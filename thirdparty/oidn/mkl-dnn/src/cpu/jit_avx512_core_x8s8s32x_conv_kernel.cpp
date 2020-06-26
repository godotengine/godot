/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_memory.hpp"

#include "jit_avx512_core_x8s8s32x_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

namespace {
void pick_loop_order(jit_conv_conf_t &jcp, int nthr)
{
    jcp.loop_order = loop_cwgn;
    if (jcp.ngroups > 1) {
        jcp.loop_order = loop_ngcw;
        if (jcp.mb < nthr)
            jcp.loop_order = jcp.ndims == 3 ? loop_nwcg : loop_nhwcg;
    }
}
}

template<typename Vmm>
bool _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::maybe_eltwise(int position)
{
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

template<typename Vmm>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::prepare_output(int ur_w)
{
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    for (int k = 0; k < nb_oc_block; k++)
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            vpxord(vmm, vmm, vmm);
        }
    if (jcp.signed_input) {
        xor_(reg_scratch, reg_scratch);
        if (jcp.is_depthwise && !jcp.is_fast_depthwise) {
            Reg32 _t32 = reg_scratch.cvt32();
            mov(_t32, (uint32_t)128);
            vpbroadcastd(vmm_shift, _t32);
        } else {
            Reg8 _t8 = reg_scratch.cvt8();
            mov(_t8, (int8_t)128);
            vpbroadcastb(vmm_shift, _t8);
        }
    }
}

template<typename Vmm>
const Vmm _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::
    vmm_mask(const Vmm vmm_in, bool mask_flag, bool store) {
    return vmm_in;
}

template<>
const Zmm _jit_avx512_core_x8s8s32x_fwd_kernel<Zmm>::
    vmm_mask(const Zmm zmm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}


template<typename Vmm>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::cvt2ps(data_type_t type_in,
        const Vmm vmm_in, const Operand &op, bool mask_flag) {
    //const Vmm vmm = mask_flag ? vmm_in | ktail_mask | T_z : vmm_in;
    const Vmm vmm = vmm_mask(vmm_in, mask_flag);
    switch (type_in) {
    case data_type::f32:
    case data_type::s32: vmovups(vmm, op); break;
    case data_type::s8: vpmovsxbd(vmm, op); break;
    case data_type::u8: vpmovzxbd(vmm, op); break;
    default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32)
        vcvtdq2ps(vmm_in, vmm_in);
}

template<typename Vmm>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::compute_eltwise(int ur_w) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    if (ur_w == jcp.ur_w)
        eltwise_injector_->compute_vector_range(0, nb_oc_block * jcp.ur_w);
    else
        for (int k = 0; k < nb_oc_block; k++)
            eltwise_injector_->compute_vector_range(k * jcp.ur_w,
                k * jcp.ur_w + ur_w);
}

template<typename Vmm>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::store_output(
        int ur_w, bool last_oc_block_flag) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    int oc_block = jcp.is_depthwise ? jcp.ch_block : jcp.oc_block;

    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
    if (jcp.signed_input)
        mov(reg_compensation, ptr[param1 + GET_OFF(compensation)]);

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = nullptr;
    if (sum_idx != -1) {
        const auto &p_entry = p.entry_[sum_idx];
        p_sum_scale = &p_entry.sum.scale;
    }

    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

    if (jcp.signed_input && jcp.ver != ver_vnni) {
        /* put 'wei_adj_scale = 0.5' for bias calculation */
        mov(reg_bias_alpha, float2int(jcp.wei_adj_scale));
        vmovq(xmm_bias_alpha(), reg_bias_alpha);
        vbroadcastss(vmm_bias_alpha(), xmm_bias_alpha());
    }

    for (int k = 0; k < nb_oc_block; k++) {
        const bool mask_flag = last_oc_block_flag && k == nb_oc_block - 1;
        int scale_offset = jcp.is_oc_scale * (sizeof(float) * k * oc_block);
        if (jcp.with_bias) {
            int bias_offset = jcp.typesize_bia * k * oc_block;
            auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);

            cvt2ps(jcp.bia_dt, vmm_bias, bias_addr, mask_flag);
            if (jcp.signed_input && jcp.ver != ver_vnni)
                /* bias *= 0.5 */
                vmulps(vmm_bias, vmm_bias, vmm_bias_alpha());
        }
        if (jcp.signed_input) {
            int comp_offset = sizeof(int32_t) * k * oc_block;
            auto comp_addr = EVEX_compress_addr(reg_compensation, comp_offset);

            cvt2ps(data_type::s32, vmm_comp, comp_addr, mask_flag);
        }
        /* add to zmm_accum: compensation, bias and permute */
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            if (jcp.is_fast_depthwise)
                vpermd(zmm_out(j, k), zmm_permute, zmm_out(j, k));
            vcvtdq2ps(vmm, vmm);
            if (jcp.signed_input)
                vaddps(vmm, vmm, vmm_comp);
            if (jcp.with_bias)
                vaddps(vmm, vmm, vmm_bias);

            const Vmm vmm_k = vmm_mask(vmm, mask_flag);
            vmulps(vmm_k, vmm,
                    EVEX_compress_addr(reg_ptr_scales, scale_offset));
        }
    }

    /* Do post-ops */
    if (maybe_eltwise(0)) compute_eltwise(ur_w);
    if (p_sum_scale) { // post_op: sum
        for (int k = 0; k < nb_oc_block; k++) {
            const bool mask_flag = last_oc_block_flag && k == nb_oc_block - 1;
            for (int j = 0; j < ur_w; j++) {
                int aux_output_offset
                        = jcp.typesize_out
                        * (k * oc_block
                                  + j * jcp.oc_without_padding * jcp.ngroups);
                auto addr = EVEX_compress_addr(reg_out, aux_output_offset);
                Vmm vmm = vmm_out(j, k);
                cvt2ps(jcp.dst_dt, vmm_prev_dst, addr, mask_flag);
                if (*p_sum_scale == 1.f)
                    vaddps(vmm, vmm_prev_dst);
                else
                    vfmadd231ps(vmm, vmm_prev_dst, zword_b[reg_ptr_sum_scale]);
            }
        }
    }
    if (maybe_eltwise(1)) compute_eltwise(ur_w);

    /* write out register to output_addr */
    for (int k = 0; k < nb_oc_block; k++) {
        const bool mask_flag = last_oc_block_flag && k == nb_oc_block - 1;
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            if (jcp.dst_dt == data_type::u8) {
                vpxord(vmm_zero, vmm_zero, vmm_zero);
                vmaxps(vmm, vmm_zero, vmm);
            }

            if (jcp.dst_dt != data_type::f32) {
                /* Note: using Zmm for rounding in Xmm/Ymm kernel
                   because there is no instruction to do rounding
                   from Xmm/Ymm -> Xmm/Ymm.
                   Embedded rounding is not supported for Xmm.
                   TODO: maybe avoid Zmm if it helps performance.*/
                Zmm zmm = zmm_out(j, k);
                vcvtps2dq(zmm, zmm);
            }
        }

        for (int j = 0; j < ur_w; j++) {
            int aux_output_offset = jcp.typesize_out
                    * (k * oc_block + j * jcp.oc_without_padding * jcp.ngroups);
            auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

            Vmm vmm = vmm_out(j, k);
            const Vmm r_vmm = vmm_mask(vmm, mask_flag, true);

            switch (jcp.dst_dt) {
            case data_type::f32:
            case data_type::s32: vmovups(addr, r_vmm); break;
            case data_type::s8: vpmovsdb(addr, r_vmm); break;
            case data_type::u8: vpmovusdb(addr, r_vmm); break;
            default: assert(!"unknown dst_dt");
            }
        }
    }

}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::compute_ker_dw(
        int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag, bool h_padded) {
    assert(!"invalid group blocking for depthwise convolution");
}

template <>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Zmm>::compute_ker_dw(
        int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag, bool h_padded) {

    auto input_spatial_index = [=](int oi, int ki) {
        return (ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l);
    };

    auto input_offset2 = [=](int ii, int ci) {
        return jcp.typesize_in * (ii * jcp.ngroups + ci * jcp.ch_block);
    };

    auto input_offset3 = [=](int oi, int ci, int ki) {
        return jcp.typesize_in * input_offset2(input_spatial_index(oi, ki), ci);
    };

    auto kernel_offset = [=](int ci, int ki) {
        return jcp.typesize_in * ((ci * jcp.kh * jcp.kw + ki) * jcp.ch_block);
    };

    auto compute = [=](Zmm vreg_acc, Zmm vreg_wei, Zmm vreg_src) {
        // okay for depthwise since src is zero-extended
        if (jcp.ver == ver_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else {
            vpmaddwd(zmm_tmp, vreg_src, vreg_wei);
            vpaddd(vreg_acc, vreg_acc, zmm_tmp);
        }
    };

    int ii_start = 0;
    int ii_end = -1;
    if (jcp.is_resrc_depthwise && !h_padded) {
        // find bounds of input spatial indices
        bool first = true;
        for (int ki = 0; ki < jcp.kw; ki++) {
            int oi_start = get_ow_start(ki, pad_l);
            int oi_end = get_ow_end(ur_w, ki, pad_r);
            for (int oi = oi_start; oi < oi_end; oi++) {
                int ii = input_spatial_index(oi, ki);
                if (first || ii < ii_start)
                    ii_start = ii;
                if (first || ii > ii_end)
                    ii_end = ii;
                first = false;
            }
        }
    }

    if (jcp.signed_input) {
        vpxord(zmm_shifted_zero, zmm_shifted_zero, zmm_shifted_zero);
        vpaddb(zmm_shifted_zero, zmm_shifted_zero, vmm_shift);
    }
    for (int ci = 0; ci < jcp.nb_ch_blocking; ci++) {
        const bool mask_flag = last_ic_block_flag != no_last_block
                && ci == jcp.nb_ch_blocking - 1;
        if (jcp.is_resrc_depthwise && !h_padded) {
            // now we can load input once and reuse up to jcp.kw times
            for (int ii = ii_start; ii <= ii_end; ii++) {
                int aux_input_offset = input_offset2(ii, ci);
                const Zmm zmm_inp_tmp = zmm_inp(ii, jcp.nb_ch_blocking);
                const Zmm zmm_inp_msk = mask_flag
                        ? zmm_inp_tmp | ktail_mask | T_z
                        : zmm_inp_tmp;
                if (jcp.is_fast_depthwise) {
                    assert(!mask_flag);
                    vbroadcasti32x4(zmm_inp_msk,
                            EVEX_compress_addr(aux_reg_inp, aux_input_offset));
                } else {
                    vpmovzxbd(zmm_inp_msk,
                            EVEX_compress_addr(aux_reg_inp, aux_input_offset));
                }
                if (jcp.signed_input)
                    vpaddb(zmm_inp_tmp, zmm_inp_tmp, vmm_shift);
            }
        }
        for (int ki = 0; ki < jcp.kw; ki++) {
            int aux_kernel_offset = kernel_offset(ci, ki);
            if (jcp.is_fast_depthwise) {
                vbroadcasti32x4(zmm_wei,
                        EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                vmovdqu8(zmm_wei | kblend_mask | T_z, zmm_wei);
            } else {
                vpmovsxbd(zmm_wei,
                        EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
            }
            if (h_padded) {
                assert(jcp.signed_input);
                for (int oi = 0; oi < ur_w; oi++)
                    compute(zmm_out(oi, ci), zmm_wei, zmm_shifted_zero);
            } else {
                const Zmm r_zmm_src = mask_flag ? zmm_src | ktail_mask : zmm_src;
                int oi_start = get_ow_start(ki, pad_l);
                int oi_end = get_ow_end(ur_w, ki, pad_r);
                int start_ = jcp.signed_input ? 0 : oi_start;
                int end_ = jcp.signed_input ? ur_w : oi_end;
                for (int oi = start_; oi < end_; oi++) {
                    if (oi >= oi_start && oi < oi_end) {
                        if (jcp.is_resrc_depthwise) {
                            int ii = input_spatial_index(oi, ki);
                            zmm_src = zmm_inp(ii, jcp.nb_ch_blocking);
                        } else {
                            int aux_input_offset = input_offset3(oi, ci, ki);
                            if (jcp.is_fast_depthwise) {
                                assert(!mask_flag);
                                vbroadcasti32x4(r_zmm_src,
                                        EVEX_compress_addr(aux_reg_inp,
                                                        aux_input_offset));
                            } else {
                                vpmovzxbd(r_zmm_src,
                                        EVEX_compress_addr(aux_reg_inp,
                                                  aux_input_offset));
                            }
                            if (jcp.signed_input)
                                vpaddb(zmm_src, zmm_src, vmm_shift);
                        }
                    } else if (jcp.signed_input) {
                        zmm_src = zmm_shifted_zero;
                    }
                    compute(zmm_out(oi, ci), zmm_wei, zmm_src);
                }
            }
        }
    }
}

template<typename Vmm>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::compute_ker(int ur_w, int pad_l,
        int pad_r, ic_block_t last_ic_block_flag, bool h_padded) {
    if (jcp.is_depthwise)
        return compute_ker_dw(ur_w, pad_l, pad_r, last_ic_block_flag, h_padded);

    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ch_block_all = jcp.ch_block * ic_block * oc_block;

    int nb_oc_block = jcp.nb_oc_blocking;

    auto input_offset = [=](int oi, int ic, int ki) {
        return jcp.typesize_in
                * ((ki * (jcp.dilate_w + 1) + oi * stride_w - pad_l)
                          * jcp.ic_without_padding * jcp.ngroups + 4 * ic);
    };
    auto kernel_offset = [=](int ii, int ic, int ki) {
        return jcp.typesize_in
                * ((ii * jcp.nb_ic * jcp.kh * jcp.kw + ki) * ch_block_all
                    + 4 * ic * oc_block);
    };
    auto compute = [=](Vmm vreg_acc, Vmm vreg_wei, Vmm vreg_src) {
        if (jcp.ver == ver_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else {
            vpmaddubsw(vmm_tmp, vreg_src, vreg_wei);
            vpmaddwd(vmm_tmp, vmm_tmp, vmm_one);
            vpaddd(vreg_acc, vreg_acc, vmm_tmp);
        }
    };

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = get_ow_start(ki, pad_l);
        int jj_end = get_ow_end(ur_w, ki, pad_r);
        int tail_size = jcp.ic_without_padding % 4;
        int _start = (jcp.signed_input) ? 0 : jj_start;
        int _end = (jcp.signed_input) ? ur_w : jj_end;
        /* Skip the last loads of input if (ic%16)/4 < ic_block/4 */
        int icb = (last_ic_block_flag != no_last_block)
            ? div_up((jcp.ic_without_padding % ic_block), 4)
            : ic_block / 4;
        for (int ic = 0; ic < icb; ic++) {
            if (h_padded == true) {
                /* fill padded area with shifted values */
                Vmm inp = vmm_inp(0,nb_oc_block);
                vpxord(inp, inp, inp);
                vpaddb(inp, inp, vmm_shift);
            } else {
                for (int jj = _start; jj < _end; jj++) {
                    int aux_input_offset = input_offset(jj, ic, ki);
                    if (jj >= jj_start && jj < jj_end) {
                        if (last_ic_block_flag == last_sp_block
                                && tail_size != 0 && ic == icb - 1) {
                            Xmm xmm_tmp = Xmm(vmm_inp(jj, nb_oc_block).getIdx());
                            for (int r = 0; r < tail_size; ++r)
                                vpinsrb(xmm_tmp, xmm_tmp,
                                    ptr[aux_reg_inp + aux_input_offset + r], r);
                            vpbroadcastd(vmm_inp(jj, nb_oc_block), xmm_tmp);
                        } else {
                            vpbroadcastd(vmm_inp(jj, nb_oc_block),
                                    EVEX_compress_addr(
                                                 aux_reg_inp, aux_input_offset));
                        }
                        if (jcp.signed_input)
                            vpaddb(vmm_inp(jj, nb_oc_block),
                                   vmm_inp(jj, nb_oc_block), vmm_shift);
                    } else {
                        /* fill padded area with shifted values */
                        if (jcp.signed_input) {
                            Vmm inp = vmm_inp(jj, nb_oc_block);
                            vpxord(inp, inp, inp);
                            vpaddb(inp, inp, vmm_shift);
                        }
                    }
                }
            }
            for (int ii = 0; ii < nb_oc_block; ii++) {
                int aux_kernel_offset = kernel_offset(ii, ic, ki);
                vmovups(vmm_wei,
                        EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                for (int jj = _start; jj < _end; jj++)  {
                    Vmm inp = (h_padded == true)
                        ? vmm_inp(0,nb_oc_block) : vmm_inp(jj, nb_oc_block);
                    compute(vmm_out(jj, ii), vmm_wei, inp);
                }
            }
        }
    }
}

template<typename Vmm>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::kh_loop(
        int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag) {
    Label kh_label, skip_kh_loop;
    Label t_overflow_label, no_t_overflow_label,
          b_overflow_label, no_b_overflow_label;

    int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * ch_block_all;
    int shift_input_ptr = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw
        * jcp.ic_without_padding * jcp.ngroups;

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);

    if (jcp.signed_input && jcp.ndims > 3) {
        mov(reg_overflow, ptr[param1 + GET_OFF(t_overflow)]);
        cmp(reg_overflow, 0);
        je(no_t_overflow_label, T_NEAR);
        L(t_overflow_label); {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            add(aux_reg_ker, shift_kernel_ptr);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(t_overflow_label, T_NEAR);
        }
        L(no_t_overflow_label);
    }
    mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    if ((jcp.signed_input) || (!jcp.signed_input &&
       (jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad))) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    L(kh_label); {
        compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, false);

        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    L(skip_kh_loop);
    if (jcp.signed_input && jcp.ndims > 3) {
        mov(reg_overflow, ptr[param1 + GET_OFF(b_overflow)]);
        cmp(reg_overflow, 0);
        je(no_b_overflow_label, T_NEAR);
        L(b_overflow_label); {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            add(aux_reg_ker, shift_kernel_ptr);
            dec(reg_overflow);
            cmp(reg_overflow, 0);
            jg(b_overflow_label, T_NEAR);
        }
        L(no_b_overflow_label);
    }
}

template<typename Vmm>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::icb_loop(
        int ur_w, int pad_l, int pad_r, bool is_last_sp_block)
{
    prepare_output(ur_w);

    // IC loop
    Label icb_label;
    mov(reg_icb, jcp.nb_ic);
    L(icb_label);
    if (jcp.ngroups % jcp.ch_block != 0 || jcp.ic_without_padding != jcp.ic) {
        Label common_ker, end_ker;

        cmp(reg_icb, 1); // The last IC block
        jne(common_ker, T_NEAR);

        kh_loop(ur_w, pad_l, pad_r,
                is_last_sp_block ? last_sp_block : last_ic_block);
        jmp(end_ker, T_NEAR);

        L(common_ker);
        kh_loop(ur_w, pad_l, pad_r, no_last_block);

        L(end_ker);
    } else {
        kh_loop(ur_w, pad_l, pad_r, no_last_block);
    }
    // End of IC Loop
    int inp_step = jcp.ic_block;
    int ker_step = jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;
    add(reg_inp, jcp.typesize_in * inp_step);
    add(reg_ker, jcp.typesize_in * ker_step);

    dec(reg_icb);
    cmp(reg_icb, 0);
    jg(icb_label, T_NEAR);

    sub(reg_inp, jcp.typesize_in * inp_step * jcp.nb_ic);
    sub(reg_ker, jcp.typesize_in * ker_step * jcp.nb_ic);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        Label common_store, end_store;

        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - jcp.nb_ch_blocking);
        else
            cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);

        jne(common_store, T_NEAR);

        store_output(ur_w, true); // last oc block
        jmp(end_store, T_NEAR);

        L(common_store);
        store_output(ur_w, false);

        L(end_store);
    } else {
        store_output(ur_w, false);
    }
}

template<typename Vmm>
void _jit_avx512_core_x8s8s32x_fwd_kernel<Vmm>::generate()
{
    Label permute_index_table;
    int inp_shift_pad = jcp.typesize_in * (jcp.ur_w * jcp.stride_w - jcp.l_pad)
        * jcp.ic_without_padding * jcp.ngroups;
    int inp_shift_pad_second_block = -1 * jcp.typesize_in * jcp.l_pad
        * jcp.ic_without_padding * jcp.ngroups;
    int inp_shift = jcp.typesize_in *
                        (jcp.ur_w * jcp.stride_w * jcp.ic_without_padding
                         * jcp.ngroups);
    int out_shift = jcp.typesize_out *
                        (jcp.ur_w * jcp.oc_without_padding * jcp.ngroups);
    preamble();

    if (jcp.is_depthwise) {
        int idx = jcp.max_regs_ur - 1;
        if (!jcp.is_resrc_depthwise)
            zmm_src = Zmm(++idx);
        if (jcp.ver != ver_vnni)
            zmm_tmp = Zmm(++idx);
        if (jcp.is_fast_depthwise)
            zmm_permute = Zmm(++idx);
        if (jcp.signed_input) {
            zmm_shifted_zero = Zmm(++idx);
            ++idx; // due to extra register used for shifts and compensations
        }
        assert(idx == ker_dw_reg_base_idx);
    }

    if (!jcp.is_depthwise && jcp.ver != ver_vnni) {
        xor_(reg_scratch, reg_scratch);
        Reg16 _t16 = reg_scratch.cvt16();
        mov(_t16, 0x1);
        vpbroadcastw(vmm_one, _t16);
    }

    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.is_depthwise
            ? jcp.ngroups % jcp.ch_block
            : jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
        Reg32 regw_tmp = reg_oi.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
    }
    if (jcp.is_fast_depthwise) {
        // prepare mask register for blending weights
        mov(reg_scratch, 0x8888444422221111);
        kmovq(kblend_mask, reg_scratch);
        // load permute indices from data section
        mov(reg_scratch, permute_index_table);
        vmovdqu32(zmm_permute, ptr[reg_scratch]);
    }

    int r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    int n_oi = jcp.ow / jcp.ur_w;
    int r_pad1 = (jcp.ur_w * n_oi - 1) * jcp.stride_w
        + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1);

    if (jcp.nb_ow == 1) {
        if (r_pad1 > 0 || jcp.ur_w_tail == 0)
            n_oi--;

        xor_(reg_oi, reg_oi);
        if (jcp.ow == jcp.ur_w) {
            icb_loop(jcp.ur_w, jcp.l_pad, r_pad, true);
        } else {
            if (n_oi == 0) {
                icb_loop(jcp.ur_w, jcp.l_pad, r_pad1, jcp.ur_w_tail == 0);
                add(reg_inp, inp_shift_pad);
                add(reg_out, out_shift);
                if (jcp.ur_w_tail != 0) {
                    icb_loop(jcp.ur_w_tail, 0, r_pad, true);
                }
            } else {
                if (jcp.l_pad > 0) {
                    icb_loop(jcp.ur_w, jcp.l_pad, 0, false);
                    add(reg_inp, inp_shift_pad);
                    add(reg_out, out_shift);

                    inc(reg_oi);
                }
                if ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1))
                {
                    Label ow_loop_label;
                    L(ow_loop_label); {
                        icb_loop(jcp.ur_w, 0, 0, false);
                        add(reg_inp, inp_shift);
                        add(reg_out, out_shift);

                        inc(reg_oi);
                        cmp(reg_oi, n_oi);
                        jl(ow_loop_label, T_NEAR);
                    }
                }
                if (r_pad1 > 0 || jcp.ur_w_tail == 0) {
                    icb_loop(jcp.ur_w, 0, r_pad1, jcp.ur_w_tail == 0);
                    add(reg_inp, inp_shift);
                    add(reg_out, out_shift);
                }
                if (jcp.ur_w_tail != 0) {
                    icb_loop(jcp.ur_w_tail, 0, r_pad, true);
                }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.
        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label,
            oi_loop_label, oi_loop_end_label;

        assert(jcp.ow_block % jcp.ur_w == 0);
        int n_oi_not_last_ow_block = jcp.ow_block / jcp.ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;
        int n_oi_last_ow_block
            = (jcp.ow - jcp.ow_block * (jcp.nb_ow - 1)) / jcp.ur_w;
        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded
                = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded
                = (r_pad1 > 0 || jcp.ur_w_tail == 0) && n_oi_last_ow_block > 0;

        if (last_ow_block_padded) n_oi_last_ow_block--;
        else if (first_ow_block_padded) n_oi_first_ow_block--;
        else if (next_last_ow_block_padded) n_oi_next_last_ow_block--;

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, 0); // is that the first ow-block ?
        jg(middle_ow_blocks_label, T_NEAR);

        // the first ow block, compute left padding
        mov(reg_oi, n_oi_first_ow_block);
        if (jcp.l_pad > 0) {
            icb_loop(jcp.ur_w, jcp.l_pad, 0, false);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);

            dec(reg_oi);
        }
        jmp(oi_loop_label, T_NEAR);

        // middle or last ow block entry
        L(middle_ow_blocks_label);

        if (jcp.l_pad > 0) {
            // just to consider left padding, not compute
            add(reg_inp, inp_shift_pad_second_block);
        }

        // set number of iteration for oi-loop
        if (n_oi_last_ow_block != n_oi_not_last_ow_block) {
            cmp(reg_owb, jcp.nb_ow - 1); // last ow-block ?
            mov(reg_oi, n_oi_last_ow_block);
            je(oi_loop_label, T_NEAR);
        }

        if (n_oi_next_last_ow_block != n_oi_not_last_ow_block) {
            cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?

            mov(reg_oi, n_oi_next_last_ow_block);
            je(oi_loop_label, T_NEAR);
        }
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label); {
            cmp(reg_oi, 0);
            jle(oi_loop_end_label, T_NEAR);

            icb_loop(jcp.ur_w, 0, 0, false);

            add(reg_inp, inp_shift);
            add(reg_out, out_shift);
            dec(reg_oi);

            jmp(oi_loop_label, T_NEAR);
        }
        L(oi_loop_end_label);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded)
            je(last_oi_label, T_NEAR);
        else
            je(end_label, T_NEAR);

        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        jl(end_label, T_NEAR);
        if (next_last_ow_block_padded)
            je(last_oi_label, T_NEAR);
        else
            je(end_label, T_NEAR);

        // that is last block
        if (!last_ow_block_padded)
            jmp(tail_label, T_NEAR);

        // last oi block with right padding
        L(last_oi_label);
        icb_loop(jcp.ur_w, 0, r_pad1, jcp.ur_w_tail == 0);
        add(reg_inp, inp_shift);
        add(reg_out, out_shift);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        jl(end_label, T_NEAR);

        // ur_w tail
        L(tail_label);
        if (jcp.ur_w_tail != 0) {
            icb_loop(jcp.ur_w_tail, 0, r_pad, true);
        }
        L(end_label);
    }
    postamble();

    if (jcp.with_eltwise)
        eltwise_injector_->prepare_table();

    if (jcp.is_fast_depthwise) {
        align(64);
        L(permute_index_table);
        const uint32_t _idx[]
                = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            dd(_idx[i]);
    }
}

bool jit_avx512_core_x8s8s32x_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr)
{
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    switch (p.len_) {
    case 0: return true;
    case 1: return is_eltwise(0) || p.contain(sum, 0);
    case 2: return (p.contain(sum, 0) && is_eltwise(1)) ||
                        (p.contain(sum, 1) && is_eltwise(0));
    default: return false;
    }

    return false;
}

status_t jit_avx512_core_x8s8s32x_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            memory_desc_t &bias_md, const primitive_attr_t &attr,
            int nthreads)
{
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    bool is_1d = ndims == 3;

    if (!(mayiuse(avx512_core)
         && one_of(src_d.data_type(), data_type::u8, data_type::s8)
         && weights_d.data_type() == data_type::s8
         && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
            data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
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
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.ur_h = 1; /* no code-unrolling by h so far */

    jcp.dilate_h = is_1d ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.signed_input = (src_d.data_type() == data_type::s8) ? true : false;
    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);

    if (jcp.is_depthwise) {
        jcp.ch_block = 16;
        jcp.ic_block = 1;
        jcp.oc_block = 1;
    } else {
        jcp.ch_block = 1;
        jcp.ic_block = 16;
        jcp.oc_block = 16;

        if (jcp.ngroups == 1) {
            /* For non grouped convolutions, pad channels by 16 if needed */
            jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
            jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        } else if (!is_1d && jcp.ngroups != 1 && jcp.ic % jcp.ic_block != 0) {
            /* For grouped convolutions, MKL-DNN doesn't support padding.
               Use Ymm when channels per group is multiple of 8,
               Xmm when channels per group is multiple of 4 */
            jcp.ic_block = jcp.ic % 8 == 0 ? 8 : 4;
            jcp.oc_block = jcp.ic_block;
        }
        if (jcp.ic % jcp.ic_block !=0 || jcp.oc % jcp.oc_block != 0)
            return status::unimplemented;
    }

    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise)
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    jcp.ver = mayiuse(avx512_core_vnni) ? ver_vnni : ver_avx512_core;
    jcp.is_fast_depthwise = true && jcp.is_depthwise && jcp.ver == ver_vnni
            && jcp.ngroups % jcp.ch_block == 0; // for groups not multiple of 16
                                                // would require byte masking
                                                // for load from src
    jcp.is_resrc_depthwise = jcp.is_depthwise && jcp.stride_w < jcp.kw
            && jcp.kw < 4 && jcp.dilate_w == 0;
    if (jcp.is_depthwise) {
        jcp.max_regs_ur = 31 - jcp.is_fast_depthwise - !jcp.is_resrc_depthwise
                - 2 * jcp.signed_input - (jcp.ver != ver_vnni);
    } else {
        jcp.max_regs_ur = jcp.ver == ver_vnni ? 31 : 28;
    }

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        format_tag_t wei_tag;
        if (jcp.ic_block == 16 || jcp.ch_block == 16) {
            if (is_1d) {
                wei_tag = with_groups
                    ? jcp.is_depthwise ? Goiw16g : gOIw4i16o4i
                    : OIw4i16o4i;
            } else {
                wei_tag = with_groups
                    ? jcp.is_depthwise ? Goihw16g : gOIhw4i16o4i
                    : OIhw4i16o4i;
            }
        } else if (with_groups && jcp.ic_block == 8) {
            wei_tag = gOIhw2i8o4i;
        } else
            wei_tag = gOIhw4o4i;

        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);
        if (jcp.signed_input) {
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

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia = jcp.with_bias
        ? types::data_type_size(bias_d.data_type())
        : 0;

    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    // Try to use 4 channel-groups at a time to avoid false sharing (depthwise)
    int nb_ch_blocking = 4;
    for ( /* init above */ ; nb_ch_blocking > 1; nb_ch_blocking--)
        if (jcp.nb_ch % nb_ch_blocking == 0)
            break;
    jcp.nb_ch_blocking = jcp.is_depthwise ? nb_ch_blocking : 1;

    // If OC blocking is incommensurate with the number of OC blocks (general
    // requirement for all convolutions), or if it results in an unrolling
    // factor smaller than the left padding (special requirement for SSD:fc6),
    // then search for a smaller OC blocking that satisfies both constraints.
    auto is_oc_blocking_ok = [&](int block) {
        int ur_w = nstl::min(jcp.ow, jcp.max_regs_ur / (block + 1));
        return jcp.nb_oc % block == 0
                && jcp.l_pad <= ur_w && jcp.ow % ur_w != 1;
    };

    // choose nb_oc work chunk size for distribution within threads
    int max_threading_nb_oc_chunk = 4;
    // Performance improvements for googlenet_v3 and resnet_50 with mb = 1;
    // TODO: generalize this condition and rewrite it in appropriate manner
    if (jcp.ver == ver_vnni && jcp.mb == 1 && jcp.kh == 3 && jcp.kw == 3
            && jcp.stride_w == 1 && jcp.ic % 64 == 0)
        max_threading_nb_oc_chunk = 2;
    jcp.nb_oc_blocking_thr_chunk =
        nstl::min(max_threading_nb_oc_chunk, jcp.nb_oc);
    for (; jcp.nb_oc_blocking_thr_chunk > 1; jcp.nb_oc_blocking_thr_chunk--) {
        if (is_oc_blocking_ok(jcp.nb_oc_blocking_thr_chunk))
            break;
    }

    // choose oc blocking for computational kernel
    jcp.nb_oc_blocking = jcp.nb_oc_blocking_thr_chunk;
    // Performance improvements for googlenet_v3 with mb = 1;
    // TODO: generalize this condition and rewrite it in appropriate manner
    const int size_treshold_for_nb_oc_blocking_reduction = 17;
    if (jcp.mb == 1 && jcp.ow <= size_treshold_for_nb_oc_blocking_reduction
            && jcp.stride_w == 1
            && !(jcp.kh == 1 && jcp.kw == 3)
            && !(jcp.kh >= 7 && jcp.oc % 64 == 0)) {
        const int max_nb_oc_blocking = 2;
        jcp.nb_oc_blocking = nstl::min(max_nb_oc_blocking, jcp.nb_oc);
        for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--)
            if (jcp.nb_oc_blocking_thr_chunk % jcp.nb_oc_blocking == 0
                && is_oc_blocking_ok(jcp.nb_oc_blocking))
                break;
    }

    if (jcp.is_resrc_depthwise)
        jcp.ur_w = (jcp.max_regs_ur - jcp.kw + jcp.stride_w)
                / (jcp.nb_ch_blocking + jcp.stride_w);
    else
        jcp.ur_w
                = jcp.max_regs_ur / (jcp.is_depthwise ? jcp.nb_ch_blocking
                                                      : jcp.nb_oc_blocking + 1);
    if (jcp.ow < jcp.ur_w)
        jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ow_block = jcp.ow;
    int base_work_amount = jcp.mb * jcp.nb_ch * jcp.oh
                         * (jcp.nb_oc / jcp.nb_oc_blocking_thr_chunk);
    float best_thr_eff
            = (float)base_work_amount / rnd_up(base_work_amount, nthreads);
    int max_nb_ow = div_up(jcp.ow, 2 * jcp.ur_w);
    for (int nb_ow = 1; nb_ow <= max_nb_ow; nb_ow++) {
        int ow_block
                = nstl::min(rnd_up(div_up(jcp.ow, nb_ow), jcp.ur_w), jcp.ow);
        if (ow_block < jcp.nb_oc_blocking_thr_chunk * jcp.oc_block
             && best_thr_eff > 0.8f)
            break;
        if (div_up(jcp.ow, ow_block) != nb_ow)
            continue;
        auto work_amount = base_work_amount * nb_ow;
        float thr_eff = (float)work_amount / rnd_up(work_amount, nthreads);
        if (ow_block >= 2 * jcp.ur_w && thr_eff > 1.1f * best_thr_eff) {
            jcp.ow_block = ow_block;
            best_thr_eff = thr_eff;
        }
        if (best_thr_eff > 0.9f)
            break;
    }
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.l_pad <= jcp.ur_w
        && IMPLICATION(!jcp.is_1stconv, jcp.ic % jcp.ic_block == 0);
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp, nthreads);

    jcp.nb_ic_L2 = jcp.nb_ic;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    jcp.wei_adj_scale =
        (weights_d.extra().flags | memory_extra_flags::scale_adjust)
        ? weights_d.extra().scale_adjust : 1.f;

    return status::success;
}

void jit_avx512_core_x8s8s32x_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        dim_t count = nstl::max(attr.output_scales_.count_, (dim_t)jcp.ic_block);
        scratchpad.book(key_conv_adjusted_scales, sizeof(float) * count);
    }
}

template struct  _jit_avx512_core_x8s8s32x_fwd_kernel<Zmm>;
template struct  _jit_avx512_core_x8s8s32x_fwd_kernel<Ymm>;
template struct  _jit_avx512_core_x8s8s32x_fwd_kernel<Xmm>;
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
