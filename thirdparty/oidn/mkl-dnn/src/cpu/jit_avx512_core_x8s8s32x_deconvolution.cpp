/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "jit_avx512_core_x8s8s32x_deconvolution.hpp"

#define GET_OFF(field) offsetof(jit_deconv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

using namespace nstl;

#define wht_blk_off(d, g, ...)                             \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) : \
                           (d).blk_off(__VA_ARGS__))

status_t jit_avx512_core_x8s8s32x_deconv_fwd_kernel::init_conf(
        jit_conv_conf_t &jcp, const deconvolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md,
        memory_desc_t &dst_md, const bool with_bias,
        memory_desc_t &bias_md, const primitive_attr_t &attr) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper bias_d(&bias_md);

    if (!(mayiuse(avx512_core)
                && one_of(src_d.data_type(), data_type::u8, data_type::s8)
                && weights_d.data_type() == data_type::s8
                && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                           data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp = zero<decltype(jcp)>();

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    jcp.signed_input = src_d.data_type() == data_type::s8;
    const int ndims = jcp.ndims = dst_d.ndims();
    const bool is_1d = ndims == 3;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.is_depthwise = true && with_groups
            && utils::everyone_is(1, jcp.ic_without_padding,
                               jcp.oc_without_padding);

    /* TODO: future work, on hold until depthwise specialized kernel is
     * implemented. */
    if (jcp.is_depthwise && jcp.signed_input)
        return status::unimplemented;

    format_tag_t dat_tag = utils::pick(ndims - 3,
            format_tag::nwc, format_tag::nhwc);

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag));
        jcp.src_tag = dat_tag;
    } else {
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.src_tag != dat_tag)
        return status::unimplemented;

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, dat_tag));
        jcp.dst_tag = dat_tag;
    } else {
        jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.dst_tag != dat_tag)
        return status::unimplemented;

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;

        format_tag_t wei_tag = is_1d
            ? (jcp.is_depthwise
                    ? Goiw16g : (with_groups ? gOIw4i16o4i : OIw4i16o4i))
            : (jcp.is_depthwise
                    ? Goihw16g : (with_groups ? gOIhw4i16o4i : OIhw4i16o4i));

        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);
        if (jcp.signed_input && !jcp.is_depthwise) {
            want_wei_md.extra.flags = 0
                | memory_extra_flags::compensation_conv_s8s8
                | memory_extra_flags::scale_adjust;
            want_wei_md.extra.compensation_mask = (1 << 0)
                + (with_groups && !jcp.is_depthwise ? (1 << 1) : 0);
            want_wei_md.extra.scale_adjust =
                mayiuse(avx512_core_vnni) ? 1.f : 0.5f;
        }

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format())
        return status::unimplemented;

    jcp.with_bias = with_bias;
    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }

    jcp.prop_kind = cd.prop_kind;
    jcp.mb = src_d.dims()[0];
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    if (jcp.is_depthwise) {
        jcp.ch_block = 16;
        jcp.oc_block = 1;
        jcp.ic_block = 1;
    } else {
        jcp.ch_block = 1;
        jcp.oc_block = 16;
        jcp.ic_block = 16;

        if (jcp.ngroups == 1) {
            jcp.oc = utils::rnd_up(jcp.oc_without_padding, jcp.oc_block);
            jcp.ic = utils::rnd_up(jcp.ic_without_padding, jcp.ic_block);
        }
        if (jcp.ic % jcp.ic_block != 0)
            return status::unimplemented;
    }

    jcp.dilate_h = is_1d ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    if (!IMPLICATION(jcp.dilate_h, jcp.stride_h == 1)
            || !IMPLICATION(jcp.dilate_w, jcp.stride_w == 1))
        return status::unimplemented;

    /* padding: bottom and right */
    jcp.b_pad = (jcp.ih - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.oh + jcp.t_pad - 1);
    jcp.r_pad = (jcp.iw - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.ow + jcp.l_pad - 1);

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise)
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    jcp.ver = ver_avx512_core;
    if (mayiuse(avx512_core_vnni))
        jcp.ver = ver_vnni;
    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    jcp.dst_dt = dst_d.data_type();
    jcp.bia_dt = jcp.with_bias ? bias_d.data_type() : data_type::undef;
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;
    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());

    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    /* kernel blocking params */
    const int regs = jcp.ver == ver_vnni ? 30 : 28;
    jcp.nb_oc_blocking = nstl::min(4, jcp.nb_oc);
    for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--)
        if (jcp.nb_oc % jcp.nb_oc_blocking == 0
                && jcp.l_pad <= regs / (jcp.nb_oc_blocking + 1))
            break;

    jcp.ur_w = regs / (jcp.nb_oc_blocking + 1);
    int l_overflow = max(
            0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);

    if (jcp.ow < jcp.ur_w) {
        jcp.ur_w = jcp.ow;
        jcp.ur_w_tail = 0;
    } else {
        for (; jcp.ur_w >= 1; jcp.ur_w--) {
            /* ur_w should be multiple of stride_w in order
               to simplify logic for get_ow_start and get_ow_end */
            bool is_multiple_of_stride = jcp.ur_w % jcp.stride_w == 0;

            /* boundary conditions:
               These conditions ensure all elements close to boundary
               are computed in a single call of compute loop */
            bool left_boundary_covered = jcp.ur_w >= l_overflow * jcp.stride_w;
            jcp.ur_w_tail = jcp.ow % jcp.ur_w;
            int r_overflow_no_tail
                    = max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                                     - max(0, jcp.r_pad) - jcp.ur_w_tail)
                                    / jcp.stride_w);
            bool right_boundary_covered
                    = jcp.ur_w >= r_overflow_no_tail * jcp.stride_w;

            if (is_multiple_of_stride && left_boundary_covered
                    && right_boundary_covered)
                break;
            else if (jcp.ur_w == 1)
                /* The boundary conditions above are also important
                   to maintain simplicity of calls to icb_loop,
                   if those conditions are not satisfied,
                   then special cases will need to be added
                   to use correct l_overflow/r_overflow values
                   when different iterations of compute loop
                   work on the locations close to boundary.
                   So to keep code simple, return unimplemented
                   for extreme case when a good ur_w cannot be found.
                 */
                return status::unimplemented;
        }
    }

    jcp.wei_adj_scale =
        (weights_d.extra().flags | memory_extra_flags::scale_adjust)
        ? weights_d.extra().scale_adjust : 1.f;

    jcp.loop_order = jcp.ngroups > 1 ? loop_ngc : loop_cgn;
    return status::success;
}

bool jit_avx512_core_x8s8s32x_deconv_fwd_kernel::maybe_eltwise(int position) {
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* eltwise before sum */
        return p.contain(eltwise, 0);
    } else if (position == 1) {
        /* eltwise after sum */
        return p.contain(sum, 0) && p.contain(eltwise, 1);
    }
    return false;
}

void jit_avx512_core_x8s8s32x_deconv_fwd_kernel::compute_eltwise(int ur_w) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    eltwise_injector_->compute_vector_range(0, nb_oc_block * ur_w);
}

bool jit_avx512_core_x8s8s32x_deconv_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    switch (p.len_) {
    case 0: return true;
    case 1: return is_eltwise(0) || p.contain(sum, 0);
    case 2:
        return (p.contain(sum, 0) && is_eltwise(1))
                || (p.contain(sum, 1) && is_eltwise(0));
    default: return false;
    }

    return false;
}

void jit_avx512_core_x8s8s32x_deconv_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        dim_t count = nstl::max<dim_t>(attr.output_scales_.count_, 16);
        scratchpad.book(key_conv_adjusted_scales, sizeof(float) * count);
    }
}

void jit_avx512_core_x8s8s32x_deconv_fwd_kernel::compute_ker(int ur_w,
        int l_overflow, int r_overflow, ker_block_t last_ic_block_flag,
        bool h_padded) {

    const int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    const int ur_w_stride = jcp.signed_input ? 1 : jcp.stride_w;

    auto src_offset = [=](int oj, int icb, int ki) {
        return jcp.typesize_in
                * (((oj + jcp.l_pad - ki * (jcp.dilate_w + 1)) / jcp.stride_w)
                                  * jcp.ngroups * jcp.ic_without_padding
                          + icb * 4);
    };

    auto kernel_offset = [=](int ocb, int icb, int ki) {
        return jcp.typesize_in
                * (ocb * jcp.nb_ic * jcp.kh * jcp.kw * ch_block_all
                          + icb * jcp.oc_block * jcp.ic_block / 4
                          + ki * ch_block_all);
    };

    auto compute = [=](zmm_t vreg_acc, zmm_t vreg_wei, zmm_t vreg_src) {
        if (jcp.ver == ver_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else if (jcp.is_depthwise) {
            vpmulld(zmm_tmp, vreg_src, vreg_wei);
            vpaddd(vreg_acc, vreg_acc, zmm_tmp);
        } else {
            vpmaddubsw(zmm_tmp, vreg_src, vreg_wei);
            vpmaddwd(zmm_tmp, zmm_tmp, zmm_one);
            vpaddd(vreg_acc, vreg_acc, zmm_tmp);
        }
    };

    for (int ki = 0; ki < jcp.kw; ki++) {

        int jj_start = get_ow_start(ki, l_overflow);
        int jj_end = get_ow_end(ur_w, ki, r_overflow);

        int _start = (jcp.signed_input) ? 0 : jj_start;
        int _end = (jcp.signed_input) ? ur_w : jj_end;

        int tail_size = jcp.ic_without_padding % 4;
        int n_ic_blocks = jcp.is_depthwise ?
                1 :
                (last_ic_block_flag & ~no_last_block ?
                                div_up(jcp.ic_without_padding % jcp.ic_block,
                                        4) :
                                jcp.ic_block / 4);

        for (int icb1 = 0; icb1 < n_ic_blocks; icb1++) {
            if (h_padded == true) {
                /* fill padded area with shifted values */
                Zmm inp = zmm_inp(0, jcp.nb_oc_blocking);
                vpxord(inp, inp, inp);
                vpsubb(inp, inp, zmm_shift);
            } else {

                for (int jj = _start; jj < _end; jj += ur_w_stride) {

                    int aux_src_off = src_offset(jj, icb1, ki);

                    if (jj >= jj_start && jj < jj_end
                            && ((jj + jcp.l_pad - ki) % jcp.stride_w == 0)) {
                        if (jcp.is_depthwise) {
                            vpmovzxbd(zmm_inp(jj, jcp.nb_oc_blocking),
                                    EVEX_compress_addr(
                                              aux_reg_src, aux_src_off));
                        } else if ((last_ic_block_flag & last_sp_block)
                                && tail_size != 0 && icb1 == n_ic_blocks - 1) {
                            xmm_t xmm_tmp = xmm_t(
                                    zmm_inp(jj, jcp.nb_oc_blocking).getIdx());
                            for (int r = 0; r < tail_size; ++r)
                                vpinsrb(xmm_tmp, xmm_tmp,
                                        ptr[aux_reg_src + aux_src_off + r], r);
                            vpbroadcastd(
                                    zmm_inp(jj, jcp.nb_oc_blocking), xmm_tmp);
                        } else {
                            vpbroadcastd(zmm_inp(jj, jcp.nb_oc_blocking),
                                    EVEX_compress_addr(
                                                 aux_reg_src, aux_src_off));
                        }
                        if (jcp.signed_input)
                            vpsubb(zmm_inp(jj, jcp.nb_oc_blocking),
                                    zmm_inp(jj, jcp.nb_oc_blocking), zmm_shift);
                    } else {
                        /* fill padded area with shifted values */
                        if (jcp.signed_input) {
                            Zmm inp = zmm_inp(jj, jcp.nb_oc_blocking);
                            vpxord(inp, inp, inp);
                            vpsubb(inp, inp, zmm_shift);
                        }
                    }
                }
            }
            for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
                int aux_filt_off = kernel_offset(ocb, icb1, ki);

                if (_end - _start > 0) {
                    if (jcp.is_depthwise)
                        vpmovsxbd(zmm_wei,
                                EVEX_compress_addr(aux_reg_filt, aux_filt_off));
                    else
                        vmovups(zmm_wei,
                                EVEX_compress_addr(aux_reg_filt, aux_filt_off));
                }
                for (int jj = _start; jj < _end; jj += ur_w_stride) {
                    Zmm inp = (h_padded == true) ?
                            zmm_inp(0, jcp.nb_oc_blocking) :
                            zmm_inp(jj, jcp.nb_oc_blocking);
                    compute(zmm_out(jj, ocb), zmm_wei, inp);
                }
            }
        }
    }
}

void jit_avx512_core_x8s8s32x_deconv_fwd_kernel::kh_loop(int ur_w,
        int l_overflow, int r_overflow, ker_block_t last_ic_block_flag) {

    int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    int shift_src_ih = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw
            * jcp.ngroups * jcp.ic_without_padding;
    const int stride_h = jcp.signed_input ? 1 : jcp.stride_h;
    int shift_filt_kh = jcp.typesize_in * jcp.kw * ch_block_all * stride_h;

    Label kh_loop_label, skip_kh_loop;
    Label t_overflow_label, no_t_overflow_label, b_overflow_label,
            no_b_overflow_label;

    mov(aux_reg_src, reg_src);
    mov(aux_reg_filt, reg_filt);

    if (jcp.signed_input && jcp.ndims > 3) {
        /* Weights are transposed, so first compute 'bottom' padding. */
        mov(reg_overflow, ptr[param1 + GET_OFF(b_overflow)]);
        cmp(reg_overflow, 0);
        je(no_b_overflow_label, T_NEAR);
        L(b_overflow_label); {
            compute_ker(ur_w, 0, 0, last_ic_block_flag, true);

            add(aux_reg_filt, shift_filt_kh);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(b_overflow_label, T_NEAR);
        }
        L(no_b_overflow_label);
    }

    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    if (jcp.signed_input || ((!jcp.signed_input)
        && ((min(jcp.t_pad, jcp.b_pad) < 0)
            || ((jcp.kh - 1) * (jcp.dilate_h + 1)
                < nstl::max(jcp.t_pad, jcp.b_pad))))) {
        cmp(reg_kh, 0);
        je(skip_kh_loop, T_NEAR);
    }

    L(kh_loop_label); {
        compute_ker(ur_w, l_overflow, r_overflow, last_ic_block_flag, false);
        sub(aux_reg_src, shift_src_ih);
        add(aux_reg_filt, shift_filt_kh);
        dec(reg_kh);

        /* Insert weight compensation in stride 'holes' */
        if (jcp.signed_input && jcp.stride_h > 1) {
            Label kh_comp_loop;

            cmp(reg_kh, 0);
            je(skip_kh_loop, T_NEAR);
            mov(reg_comp_strides, jcp.stride_h - 1);
            L(kh_comp_loop);
            {
                compute_ker(
                        ur_w, 0, 0, last_ic_block_flag, true);
                add(aux_reg_filt, shift_filt_kh);
                dec(reg_comp_strides);
                cmp(reg_comp_strides, 0);
                jg(kh_comp_loop, T_NEAR);
            }
        }
        cmp(reg_kh, 0);
        jg(kh_loop_label, T_NEAR);
    }
    L(skip_kh_loop);
    if (jcp.signed_input && jcp.ndims > 3) {
        mov(reg_overflow, ptr[param1 + GET_OFF(t_overflow)]);
        cmp(reg_overflow, 0);
        je(no_t_overflow_label, T_NEAR);
        L(t_overflow_label); {
            compute_ker(ur_w, 0, 0, last_ic_block_flag, true);

            add(aux_reg_filt, shift_filt_kh);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(t_overflow_label, T_NEAR);
        }
        L(no_t_overflow_label);
    }
}

void jit_avx512_core_x8s8s32x_deconv_fwd_kernel::prepare_output(int ur_w) {
    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
        for (int ur = 0; ur < ur_w; ur++) {
            zmm_t zmm = zmm_out(ur, ocb);
            vpxord(zmm, zmm, zmm);
        }
    }
    if (jcp.signed_input) {
        xor_(reg_scratch, reg_scratch);
        Reg8 _t8 = reg_scratch.cvt8();
        mov(_t8, (int8_t)-128);
        vpbroadcastb(zmm_shift, _t8);
    }
}

void jit_avx512_core_x8s8s32x_deconv_fwd_kernel::cvt2ps(
        data_type_t type_in, zmm_t zmm_in, const Operand &op, bool mask_flag) {
    zmm_t zmm = mask_flag ? zmm_in | ktail_mask | T_z : zmm_in;
    switch (type_in) {
    case data_type::f32:
    case data_type::s32: vmovups(zmm, op); break;
    case data_type::s8: vpmovsxbd(zmm, op); break;
    case data_type::u8: vpmovzxbd(zmm, op); break;
    default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32)
        vcvtdq2ps(zmm_in, zmm_in);
}

void jit_avx512_core_x8s8s32x_deconv_fwd_kernel::store_output(
        int ur_w, bool last_oc_block) {
    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);

    if (jcp.signed_input)
        mov(reg_compensation, ptr[param1 + GET_OFF(compensation)]);

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale
            = (sum_idx != -1) ? &p.entry_[sum_idx].sum.scale : nullptr;
    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

    if (jcp.with_bias && jcp.signed_input && jcp.ver != ver_vnni) {
        mov(reg_bias_alpha, float2int(jcp.wei_adj_scale));
        vmovq(xmm_bias_alpha(), reg_bias_alpha);
        vbroadcastss(zmm_bias_alpha(), xmm_bias_alpha());
    }

    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
        const bool mask_flag = last_oc_block && ocb == jcp.nb_oc_blocking - 1;
        int scale_offset
                = jcp.is_oc_scale * (sizeof(float) * ocb * jcp.oc_block);

        auto zmm_bias = zmm_tmp;
        if (jcp.with_bias) {
            int bias_offset = jcp.typesize_bia * ocb * jcp.oc_block;
            auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
            cvt2ps(jcp.bia_dt, zmm_bias, bias_addr, mask_flag);
            if (jcp.signed_input && jcp.ver != ver_vnni)
                vmulps(zmm_bias, zmm_bias, zmm_bias_alpha());
        }
        if (jcp.signed_input) {
            int comp_offset = sizeof(int32_t) * ocb * jcp.oc_block;
            auto comp_addr = EVEX_compress_addr(reg_compensation, comp_offset);
            cvt2ps(data_type::s32, zmm_comp, comp_addr, mask_flag);
        }

        for (int ur = 0; ur < ur_w; ur++) {
            zmm_t zmm = zmm_out(ur, ocb);
            vcvtdq2ps(zmm, zmm);
            if (jcp.signed_input)
                vaddps(zmm, zmm, zmm_comp);
            if (jcp.with_bias)
                vaddps(zmm, zmm, zmm_bias);
            zmm_t mask_zmm = mask_flag ? zmm | ktail_mask | T_z : zmm;
            vmulps(mask_zmm, zmm,
                    EVEX_compress_addr(reg_ptr_scales, scale_offset));
        }
    }
    if (maybe_eltwise(0))
        compute_eltwise(ur_w);
    if (p_sum_scale) { // post_op: sum
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            const bool mask_flag
                    = last_oc_block == 1 && k == jcp.nb_oc_blocking - 1;
            for (int j = 0; j < ur_w; j++) {
                int aux_output_offset
                        = jcp.typesize_out
                        * (k * jcp.oc_block
                                  + j * jcp.oc_without_padding * jcp.ngroups);
                auto addr = EVEX_compress_addr(reg_dst, aux_output_offset);
                Zmm zmm = zmm_out(j, k);
                cvt2ps(jcp.dst_dt, zmm_prev_dst, addr, mask_flag);
                if (*p_sum_scale == 1.f)
                    vaddps(zmm, zmm_prev_dst);
                else
                    vfmadd231ps(zmm, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
        }
    }
    if (maybe_eltwise(1))
        compute_eltwise(ur_w);

    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
        const bool mask_flag = last_oc_block && ocb == jcp.nb_oc_blocking - 1;
        for (int ur = 0; ur < ur_w; ur++) {
            zmm_t zmm = zmm_out(ur, ocb);
            if (jcp.dst_dt == data_type::u8) {
                vpxord(zmm_zero, zmm_zero, zmm_zero);
                vmaxps(zmm, zmm_zero, zmm);
            }
            if (jcp.dst_dt != data_type::f32)
                vcvtps2dq(zmm, zmm);
        }
        for (int ur = 0; ur < ur_w; ur++) {
            int aux_dst_off = jcp.typesize_out
                    * (ur * jcp.ngroups * jcp.oc_without_padding
                                      + ocb * jcp.oc_block);
            auto addr = EVEX_compress_addr(reg_dst, aux_dst_off);

            zmm_t zmm = zmm_out(ur, ocb);
            zmm_t r_zmm = mask_flag ? zmm | ktail_mask : zmm;
            switch (jcp.dst_dt) {
            case data_type::f32:
            case data_type::s32: vmovups(addr, r_zmm); break;
            case data_type::s8: vpmovsdb(addr, r_zmm); break;
            case data_type::u8: vpmovusdb(addr, r_zmm); break;
            default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_avx512_core_x8s8s32x_deconv_fwd_kernel::icb_loop(
        int ur_w, int l_overflow, int r_overflow, bool is_last_sp_block) {

    int shift_src_icb = jcp.typesize_in * jcp.ic_block;
    int shift_filt_icb
            = jcp.typesize_in * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block;

    prepare_output(ur_w);

    Label skip_icb_loop, icb_loop_label;

    mov(reg_icb, jcp.nb_ic);
    L(icb_loop_label); {

        if (jcp.ic_without_padding != jcp.ic) {
            Label common_ker, end_ker;
            cmp(reg_icb, 1);
            jg(common_ker, T_NEAR);

            kh_loop(ur_w, l_overflow, r_overflow,
                    is_last_sp_block ? last_sp_block : last_ic_block);
            jmp(end_ker, T_NEAR);

            L(common_ker);
            kh_loop(ur_w, l_overflow, r_overflow, no_last_block);

            L(end_ker);
        } else {
            kh_loop(ur_w, l_overflow, r_overflow, no_last_block);
        }

        add(reg_src, shift_src_icb);
        add(reg_filt, shift_filt_icb);
        dec(reg_icb);
        cmp(reg_icb, 0);
        jg(icb_loop_label, T_NEAR);
    }

    /* come-back pointers */
    sub(reg_src, jcp.nb_ic * shift_src_icb);
    sub(reg_filt, jcp.nb_ic * shift_filt_icb);
    L(skip_icb_loop);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        Label common_store, end_store;
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - 1);
        else
            cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);
        jne(common_store, T_NEAR);

        store_output(ur_w, true);
        jmp(end_store, T_NEAR);

        L(common_store);
        store_output(ur_w, false);

        L(end_store);

    } else {
        store_output(ur_w, false);
    }
}

void jit_avx512_core_x8s8s32x_deconv_fwd_kernel::generate() {
    preamble();

    xor_(reg_scratch, reg_scratch);
    Reg16 _t = reg_scratch.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(zmm_one, _t);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.is_depthwise ?
                jcp.ngroups % jcp.ch_block :
                jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        Reg32 regw_tmp = reg_nur_w.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
    }

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_filt, ptr[param1 + GET_OFF(filt)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);

    int dst_shift = jcp.typesize_out * jcp.ur_w * jcp.ngroups
            * jcp.oc_without_padding;
    int src_shift = jcp.typesize_in * (jcp.ur_w / jcp.stride_w) * jcp.ngroups
            * jcp.ic_without_padding;

    int l_overflow = max(
            0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);
    int r_overflow
            = max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - max(0, jcp.r_pad))
                            / jcp.stride_w);

    int r_overflow1
            = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                                   - nstl::max(0, jcp.r_pad) - jcp.ur_w_tail)
                            / jcp.stride_w);
    int nur_w = jcp.ow / jcp.ur_w;
    if (r_overflow1 > 0)
        nur_w--;

    if (jcp.ur_w == jcp.ow) {
        icb_loop(jcp.ur_w, l_overflow, r_overflow, true);
    } else if (nur_w == 0) {
        icb_loop(jcp.ur_w, l_overflow, r_overflow1, jcp.ur_w_tail == 0);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
        if (jcp.ur_w_tail != 0)
            icb_loop(jcp.ur_w_tail, 0, r_overflow, true);
    } else {
        xor_(reg_nur_w, reg_nur_w);
        if (l_overflow > 0) {
            icb_loop(jcp.ur_w, l_overflow, 0, false);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            inc(reg_nur_w);
        }
        if ((l_overflow <= 0 && nur_w > 0) || (l_overflow > 0 && nur_w > 1)) {
            Label ow_loop_label;
            L(ow_loop_label);
            {
                icb_loop(jcp.ur_w, 0, 0, false);
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
                inc(reg_nur_w);
                cmp(reg_nur_w, nur_w);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_overflow1 > 0) {
            icb_loop(jcp.ur_w, 0, r_overflow1, jcp.ur_w_tail == 0);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
        }
        if (jcp.ur_w_tail != 0) {
            icb_loop(jcp.ur_w_tail, 0, r_overflow, true);
        }
    }
    postamble();

    if (jcp.with_eltwise)
        eltwise_injector_->prepare_table();
}

template <data_type_t src_type, data_type_t dst_type>
void _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<src_type,
        dst_type>::execute_forward_1d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    auto &jcp = kernel_->jcp;

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int nb_groups = jcp.nb_ch;

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales
                = scratchpad(ctx).template get<float>(key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 16);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;
    }
    size_t offset = (size_t)jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw;
    auto w = const_cast<wei_data_t *>(weights);
    int32_t *compensation
            = (jcp.signed_input) ? reinterpret_cast<int32_t *>(&w[offset]) : 0;

    parallel(0, [&](const int ithr, const int nthr) {
        int start{ 0 }, end{ 0 };
        int work_amount = jcp.mb * nb_groups * oc_chunks;
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_deconv_call_s();

        int n{ 0 }, g{ 0 }, occ{ 0 };
        if (jcp.loop_order == loop_ngc)
            nd_iterator_init(start, n, jcp.mb, g, nb_groups, occ, oc_chunks);
        else if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, occ, oc_chunks, g, nb_groups, n, jcp.mb);
        else
            assert(!"unsupported loop order");
        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g_oc = (g * jcp.ch_block * jcp.nb_oc + ocb) * jcp.oc_block;
            int g_ic = g * jcp.ch_block * jcp.ic;

            p.dst = dst + dst_d.blk_off(n, g_oc);
            p.src = src + src_d.blk_off(n, g_ic);
            p.filt = weights + wht_blk_off(weights_d, g, ocb, 0);
            p.bias = jcp.with_bias ?
                    bias + (bias_d.blk_off(g_oc) * jcp.typesize_bia) :
                    0;
            p.compensation = (jcp.signed_input) ? compensation + g_oc : 0;
            p.scales = &oscales[jcp.is_oc_scale * g_oc];
            p.t_overflow = 0;
            p.b_overflow = 0;
            p.kh_padding = jcp.kh;
            p.oc_blocks = jcp.is_depthwise ? g : ocb;

            kernel_->jit_ker(&p);

            ++start;
            if (jcp.loop_order == loop_ngc)
                nd_iterator_step(n, jcp.mb, g, nb_groups, occ, oc_chunks);
            else if (jcp.loop_order == loop_cgn)
                nd_iterator_step(occ, oc_chunks, g, nb_groups, n, jcp.mb);
            else
                assert(!"unsupported loop order");
        }
    });
}

template <data_type_t src_type, data_type_t dst_type>
void _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<src_type,
        dst_type>::execute_forward_2d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    auto &jcp = kernel_->jcp;

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int nb_groups = jcp.nb_ch;

    size_t src_h_stride = src_d.blk_off(0, 0, 1);
    size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
    size_t wht_kh_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales
                = scratchpad(ctx).template get<float>(key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 16);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;
    }
    size_t offset = (size_t)jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw;
    auto w = const_cast<wei_data_t *>(weights);
    int32_t *compensation
            = (jcp.signed_input) ? reinterpret_cast<int32_t *>(&w[offset]) : 0;

    parallel(0, [&](const int ithr, const int nthr) {
        int start{ 0 }, end{ 0 };
        int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_deconv_call_s();

        /*loop order = cgn*/
        int n{ 0 }, g{ 0 }, occ{ 0 }, oh_s{ 0 };
        if (jcp.loop_order == loop_ngc)
            nd_iterator_init(start, n, jcp.mb, g, nb_groups, occ, oc_chunks,
                    oh_s, jcp.oh);
        else if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, occ, oc_chunks, g, nb_groups, n, jcp.mb,
                    oh_s, jcp.oh);
        else
            assert(!"unsupported loop order");
        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g_oc = (g * jcp.ch_block * jcp.nb_oc + ocb) * jcp.oc_block;
            int g_ic = g * jcp.ch_block * jcp.ic;
            int work_rem = end - start;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

            auto dst_w = dst + dst_d.blk_off(n, g_oc);
            auto src_w = src + src_d.blk_off(n, g_ic);
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0);
            auto bias_w = jcp.with_bias ?
                    bias + (bias_d.blk_off(g_oc) * jcp.typesize_bia) :
                    0;
            int32_t *compensation_w
                    = (jcp.signed_input) ? compensation + g_oc : 0;

            auto scales = &oscales[jcp.is_oc_scale * g_oc];
            for (int oj = oh_s; oj < oh_e; oj++) {
                int ih_max = 0, kh_lo = 0, kh_len = 0;
                if (jcp.dilate_h != 0 && jcp.stride_h == 1) {
                    /* dilation */
                    int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    int o_t_overflow = div_up(
                            max(0, (jcp.kh - 1) * dilate_h - oj - jcp.t_pad),
                            dilate_h);
                    int o_b_overflow
                            = div_up(max(0, (jcp.kh - 1) * dilate_h + 1 - jcp.oh
                                                     + oj - jcp.b_pad),
                                    dilate_h);
                    kh_len = jcp.kh - o_t_overflow - o_b_overflow;
                    kh_lo = o_b_overflow;
                    ih_max = oj + jcp.t_pad - o_b_overflow * dilate_h;
                } else {
                    int o_t_overflow = max(
                            0, (jcp.kh - (oj + 1 + jcp.t_pad)) / jcp.stride_h);
                    int o_b_overflow
                            = max(0, ((oj + jcp.kh) - (jcp.oh + jcp.b_pad))
                                            / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1
                            - abs(jcp.oh + jcp.b_pad - (oj + 1)) % jcp.stride_h;
                    int overflow_kh_lo = (oj + jcp.t_pad) % jcp.stride_h;

                    kh_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h
                            + 1 - o_t_overflow - o_b_overflow;
                    kh_lo = overflow_kh_lo + o_b_overflow * jcp.stride_h;
                    ih_max = (oj + jcp.t_pad - kh_lo) / jcp.stride_h;
                }

                int wei_stride
                        = (!jcp.signed_input) ? kh_lo * wht_kh_stride : 0;
                p.src = src_w + ih_max * src_h_stride;
                p.dst = dst_w + oj * dst_h_stride;
                p.filt = wht_w + wei_stride;
                p.bias = bias_w;
                p.compensation = compensation_w;
                p.t_overflow = max(
                        0, jcp.kh - (kh_lo + max(0, kh_len - 1) * jcp.stride_h
                                            + 1));
                p.b_overflow = kh_lo;
                p.kh_padding = kh_len;
                p.scales = scales;
                p.oc_blocks = jcp.is_depthwise ? g : ocb;
                kernel_->jit_ker(&p);
            }
            if (jcp.loop_order == loop_ngc)
                nd_iterator_jump(start, end, n, jcp.mb, g, nb_groups, occ,
                        oc_chunks, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end, occ, oc_chunks, g, nb_groups, n,
                        jcp.mb, oh_s, jcp.oh);
            else
                assert(!"unsupported loop order");
        }
    });
}

template struct _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<data_type::u8,
        data_type::u8>;
template struct _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<data_type::u8,
        data_type::s8>;
template struct _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<data_type::u8,
        data_type::f32>;
template struct _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<data_type::u8,
        data_type::s32>;
template struct _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<data_type::s8,
        data_type::u8>;
template struct _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<data_type::s8,
        data_type::s8>;
template struct _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<data_type::s8,
        data_type::f32>;
template struct _jit_avx512_core_x8s8s32x_deconvolution_fwd_t<data_type::s8,
        data_type::s32>;
}
}
}
