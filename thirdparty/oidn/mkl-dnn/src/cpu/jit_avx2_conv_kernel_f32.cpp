/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
* Copyright 2018 YANDEX LLC
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
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_avx2_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::format_tag;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

void jit_avx2_conv_fwd_kernel_f32::oh_step_unroll_kw(int ur_w,
        int pad_l, int pad_r, int oc_blocks)
{
    int iw = jcp.iw;
    int ih = jcp.ih;
    int id = jcp.id;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int kd = jcp.kd;
    int nb_ic = jcp.nb_ic;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = nstl::max(0, div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w
            - nstl::max(0, div_up(ki*dilate_w+pad_r-(kw-1)*dilate_w, stride_w));
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                size_t inp_off;
                if (one_of(jcp.src_tag, ncw, nchw, ncdhw))
                    inp_off = sizeof(float)*((size_t)ifm2*id*ih*iw
                        + (ki*dilate_w + jj*stride_w - pad_l));
                else
                    inp_off = sizeof(float)*((ki*dilate_w + jj*stride_w
                                - pad_l)*ic_blk + ifm2);
                vbroadcastss(Ymm(oc_blocks * ur_w + jj),
                        make_safe_addr(aux_reg_input, inp_off, reg_long_offt));
            }

            for (int ii = 0; ii < oc_blocks; ii++) {
                int ker_off = ii * nb_ic * kd * kh * kw * ic_blk * oc_blk
                        + ki * ic_blk * oc_blk + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel + sizeof(float) * ker_off]);
                for (int jj = jj_start; jj < jj_end; jj++)
                    if (mayiuse(avx2))
                        vfmadd231ps(Ymm(ur_w * ii + jj),
                                Ymm(oc_blocks * ur_w + jj), ymm15);
                    else { // Intel(R) Advanced Vector Extensions (Intel(R) AVX) support
                        vmulps(ytmp, ymm15, Ymm(oc_blocks * ur_w + jj));
                        vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj), ytmp);
                    }
            }
        }
    }
}

void jit_avx2_conv_fwd_kernel_f32::oh_step_nopad(int ur_w,
        int pad_l, int pad_r, char pad_tag,
        int oc_blocks, char oc_blocks_tag)
{
    Label kw_loop;

    int iw = jcp.iw;
    int ih = jcp.ih;
    int id = jcp.id;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int kd = jcp.kd;
    int nb_ic = jcp.nb_ic;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;

    xor_(ki_iter, ki_iter);
    L(kw_loop);
    {
        int jj_start = 0;
        int jj_end = ur_w;
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                size_t inp_off;
                if (one_of(jcp.src_tag, ncw, nchw, ncdhw))
                    inp_off = sizeof(float)*((size_t)ifm2 * id * ih * iw
                            + (jj * stride_w - pad_l));
                else
                    inp_off = sizeof(float)*((jj * stride_w - pad_l) * ic_blk
                            + ifm2);
                vbroadcastss(Ymm(oc_blocks * ur_w + jj),
                    make_safe_addr(aux_reg_input, inp_off, reg_long_offt));
            }
            for (int ii = 0; ii < oc_blocks; ii++) {
                int aux_kernel_offset =
                    ii * nb_ic * kd * kh * kw * ic_blk * oc_blk + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel
                        + sizeof(float) * aux_kernel_offset]);
                for (int jj = jj_start; jj < jj_end; jj++)
                    if (mayiuse(avx2))
                        vfmadd231ps(Ymm(ur_w * ii + jj),
                                Ymm(oc_blocks * ur_w + jj), ymm15);
                    else { // Intel AVX support
                        vmulps(ytmp, ymm15, Ymm(oc_blocks * ur_w + jj));
                        vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj), ytmp);
                    }
            }
        }
        add(aux_reg_kernel, sizeof(float) * oc_blk * ic_blk);
        add(aux_reg_input, sizeof(float) * (one_of(jcp.src_tag, ncw, nchw, ncdhw)
                ? dilate_w : ic_blk * dilate_w));

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_loop, T_NEAR);
    }
}

void jit_avx2_conv_fwd_kernel_f32::width_blk_step(int ur_w,
        int pad_l, int pad_r, char pad_tag,
        int oc_blocks, char oc_blocks_tag)
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ow = jcp.ow;
    int oh = jcp.oh;
    int od = jcp.od;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    const int inp_mult = one_of(jcp.src_tag, ncw, nchw, ncdhw)
        ? 1 : ic_blk;
    const int inp_off = one_of(jcp.src_tag, ncw, nchw, ncdhw)
        ? dilate_w : ic_blk * dilate_w;

    Label init_done, init_first;

    if (!jcp.with_sum) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        jne(init_first, T_NEAR);
    }

    for (int ii = 0; ii < oc_blocks; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            size_t offt =
                sizeof(float) * ((size_t)ii * od * oh * ow + jj) * oc_blk;
            vmovups(Ymm(ur_w * ii + jj),
                    make_safe_addr(reg_output, offt, reg_long_offt));
        }
    }

    if (jcp.with_sum && jcp.with_bias) {
        test(reg_ci_flag, FLAG_IC_FIRST);
        je(init_done, T_NEAR);

        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj),
                    yword[reg_bias + sizeof(float) * ii * oc_blk]);
    }

    jmp(init_done);

    L(init_first);
    if (this->jcp.with_bias) {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                vmovups(Ymm(ur_w * ii + jj),
                        yword[reg_bias + sizeof(float) * ii * oc_blk]);
    } else {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                uni_vpxor(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj));
    }

    L(init_done);

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
    }

    Label skip_kh_loop, skip_kd_loop, kd_loop;
    if (jcp.ndims == 5) {
        push(reg_output);
        push(oi_iter);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_input);

        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1) < jcp.f_pad) {
            cmp(reg_ki, 0);
            je(skip_kd_loop, T_NEAR);
        }
        L(kd_loop);
        mov(kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_input, aux_reg_inp_d);
        mov(aux_reg_kernel, aux_reg_ker_d);
    }

    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    Label kh_loop;
    L(kh_loop);
    {
        if (jcp.kw >= 5 && pad_l == 0 && pad_r == 0) {
            oh_step_nopad(ur_w, pad_l, pad_r, pad_tag, oc_blocks,
                    oc_blocks_tag);
            sub(aux_reg_input, sizeof(float) * kw * inp_off);
            add(aux_reg_input, sizeof(float) * iw * dilate_h * inp_mult);
        } else {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks);
            add(aux_reg_kernel, sizeof(float) * kw * oc_blk * ic_blk);
            add(aux_reg_input, sizeof(float) * iw * dilate_h * inp_mult);
        }

        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }

    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
            sizeof(float) * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mult);
        add(aux_reg_ker_d, sizeof(float) * jcp.kw * jcp.kh * jcp.oc_block
            * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_loop, T_NEAR);
        L(skip_kd_loop);

        pop(oi_iter);
        pop(reg_output);
    }

    Label regular_store;

    if (jcp.with_eltwise) {
        test(reg_ci_flag, FLAG_IC_LAST);
        je(regular_store, T_NEAR);

        eltwise_injector_->compute_vector_range(0, oc_blocks * ur_w);

        L(regular_store);
    }

    for (int ii = 0; ii < oc_blocks; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            const size_t o_off
                = sizeof(float) * ((size_t)ii * od * oh * ow + jj) * oc_blk;
            Ymm reg_out = Ymm(ur_w * ii + jj);
            vmovups(make_safe_addr(reg_output, o_off, reg_long_offt), reg_out);
        }
    }
}

inline void jit_avx2_conv_fwd_kernel_f32::solve_common(
        int oc_blocks, char oc_blocks_tag)
{
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int n_oi = jcp.ow / ur_w;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    int dilate_w = jcp.dilate_w + 1;
    int str_w = jcp.stride_w;
    const int inp_mult = one_of(jcp.src_tag, ncw, nchw, ncdhw) ? 1 : ic_blk;

    int l_pad = jcp.l_pad;
    int r_pad = nstl::max(0, (int(jcp.ow) - 1) * str_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1));
    int r_pad1 = (ur_w * n_oi - 1) * str_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1,
                    'l', oc_blocks, oc_blocks_tag); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0,
                    'l', oc_blocks, oc_blocks_tag); // "lpad"
        add(reg_input, sizeof(float) * (ur_w * str_w - l_pad) * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    Label ow_loop;
    xor_(oi_iter, oi_iter);

    if (n_oi > 0) {
        L(ow_loop);

        width_blk_step(ur_w, 0, 0,
                'm', oc_blocks, oc_blocks_tag); // "middle"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);

        inc(oi_iter);
        cmp(oi_iter, n_oi);
        jl(ow_loop, T_NEAR);
    }

    if (r_pad1 > 0 && n_oi >=0) {
        width_blk_step(ur_w, 0, r_pad1,
                'r', oc_blocks, oc_blocks_tag); // "rpad"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad,
                't', oc_blocks, oc_blocks_tag); // "tail"
}

void jit_avx2_conv_fwd_kernel_f32::generate()
{
    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_ci_flag, ptr[this->param1 + GET_OFF(flags)]);
    mov(reg_oc_blocks, ptr[this->param1 + GET_OFF(oc_blocks)]);

    int nb_oc_tail = jcp.nb_oc % jcp.nb_oc_blocking;
    Label tail, exit;

    if (jcp.nb_oc > jcp.nb_oc_blocking) {
        cmp(reg_oc_blocks, jcp.nb_oc_blocking);
        jne(nb_oc_tail ? tail : exit, T_NEAR);

        solve_common(jcp.nb_oc_blocking, '0' + jcp.nb_oc_blocking);
        jmp(exit, T_NEAR);

        if (nb_oc_tail) {
            L(tail);
            cmp(reg_oc_blocks, nb_oc_tail);
            jne(exit, T_NEAR);
            solve_common(nb_oc_tail, '0' + nb_oc_tail);
        }

        L(exit);
    } else if (jcp.nb_oc == jcp.nb_oc_blocking) {
        solve_common(jcp.nb_oc_blocking, '0' + jcp.nb_oc_blocking);
    } else {
        solve_common(nb_oc_tail, '0' + nb_oc_tail);
    }

    this->postamble();

    if (jcp.with_eltwise)
        eltwise_injector_->prepare_table();
}

bool jit_avx2_conv_fwd_kernel_f32::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
    case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
    default: return false;
    }

    return false;
}

status_t jit_avx2_conv_fwd_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr)
{
    if (!mayiuse(avx)) return status::unimplemented;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 :dst_d.dims()[ndims-2];
    jcp.ow = dst_d.dims()[ndims-1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims-2];
    jcp.kw = weights_d.dims()[with_groups + ndims-1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 :cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);

    if (ndims == 3) {
        jcp.src_tag = src_d.matches_one_of_tag(ncw, nwc, nCw8c);
        jcp.wei_tag = weights_d.matches_one_of_tag(
                Owi8o, gOwi8o, OIw8i8o, gOIw8i8o);
        jcp.dst_tag = dst_d.matches_one_of_tag(nCw8c);
    } else if (ndims == 4) {
        jcp.src_tag = src_d.matches_one_of_tag(nchw, nhwc, nChw8c);
        jcp.wei_tag = weights_d.matches_one_of_tag(
                Ohwi8o, gOhwi8o, OIhw8i8o, gOIhw8i8o);
        jcp.dst_tag = dst_d.matches_one_of_tag(nChw8c);
    } else if (ndims == 5) {
        jcp.src_tag = src_d.matches_one_of_tag(ncdhw, ndhwc, nCdhw8c);
        jcp.wei_tag = weights_d.matches_one_of_tag(
                Odhwi8o, gOdhwi8o, OIdhw8i8o, gOIdhw8i8o);
        jcp.dst_tag = dst_d.matches_one_of_tag(nCdhw8c);
    }
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (!mayiuse(avx2) && jcp.eltwise.alg != alg_kind::eltwise_relu)
            return status::unimplemented;
    }

    const int simd_w = 8;
    const bool flat = jcp.ic < simd_w;
    const bool mimo = !flat;


    /* Grouped channel offset to support 'non-blocked data' format for
     * convolution sizes with '(input_channel / ngroups) < simd' */
    jcp.nonblk_group_off =
        one_of(jcp.src_tag, ncw, nchw, ncdhw) && jcp.ngroups > 1 ? jcp.ic : 1;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        if (mimo)
            jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    bool args_ok = true
        && IMPLICATION(flat, true
                && one_of(jcp.src_tag, ncw, nwc, nchw, nhwc, ncdhw, ndhwc)
                && one_of(jcp.wei_tag, Owi8o, gOwi8o, Ohwi8o, gOhwi8o, Odhwi8o,
                    gOdhwi8o))
        && IMPLICATION(mimo, true
                && one_of(jcp.src_tag, nCw8c, nChw8c, nCdhw8c)
                && one_of(jcp.wei_tag, OIw8i8o, gOIw8i8o, OIhw8i8o, gOIhw8i8o,
                    OIdhw8i8o, gOIdhw8i8o))
        && one_of(jcp.dst_tag, nCw8c, nChw8c, nCdhw8c);
    if (!args_ok) return status::unimplemented;

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = 3;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.nb_oc_blocking = 4; /* the optimal value for the kernel */

    // Intel AVX and Intel AVX2 kernels need 2 and 1 temporary YMMs, respectively
    // Thus, we can only assign 14 or 15 YMMs for data storage
    const int num_avail_regs = mayiuse(avx2) ? 15 : 14;
    if (!mayiuse(avx2)) {
        if ((jcp.nb_oc_blocking + 1) * jcp.ur_w > num_avail_regs) {
            // current register assignment requires more YMMs than available
            // adjust one of nb_oc_block, ur_w preserving to ur_w >= l_pad
            if (jcp.ur_w > jcp.l_pad && jcp.ur_w > 1)
                jcp.ur_w -= 1;
            else
                for (int b = 3; b > 1; b--)
                    if (jcp.nb_oc % b == 0) {
                        jcp.nb_oc_blocking = b;
                        break;
                    }
        }
    }

    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    args_ok = true
        && jcp.oc % simd_w == 0
        && jcp.l_pad <= jcp.ur_w
        && IMPLICATION(jcp.kw > 7, (jcp.t_pad == 0 && jcp.l_pad == 0)
                || (jcp.stride_w == 1 && jcp.stride_h == 1))
        && IMPLICATION(mimo, jcp.ic % simd_w == 0);
    if (!args_ok) return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
        + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));

    if (r_pad_no_tail > jcp.ur_w * jcp.stride_w && jcp.ow / jcp.ur_w > 1) {
        /* recalculate ur_w, nb_oc_blocking and ur_w_tail */
        jcp.ur_w = nstl::min(r_pad_no_tail / jcp.stride_w + jcp.ur_w_tail,
                nstl::min(jcp.ow, num_avail_regs / 2));
        jcp.nb_oc_blocking = (num_avail_regs - jcp.ur_w) / jcp.ur_w;
        jcp.ur_w_tail = jcp.ow % jcp.ur_w;
        /* check again ... */
        r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
        if (jcp.ur_w < nstl::max(jcp.l_pad, r_pad_no_tail))
            return status::unimplemented;
    }
    assert(jcp.nb_oc_blocking > 0);
    assert(jcp.ur_w * (jcp.nb_oc_blocking + 1) <= num_avail_regs);

    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        jcp.nb_ic_blocking = 12;
        jcp.nb_ic_blocking_max = 16;
    } else {
        jcp.nb_ic_blocking = 1;
        jcp.nb_ic_blocking_max = jcp.nb_ic_blocking;
    }

    return status::success;
}

void jit_avx2_conv_fwd_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, sizeof(float) * jcp.oc);
}

void jit_avx2_conv_bwd_data_kernel_f32::compute_loop(int ur_w, int l_overflow,
        int r_overflow)
{
    int kw = jcp.kw;
    int kh = jcp.kh;
    int kd = jcp.kd;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int id = jcp.id;
    int ow = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    Label kd_loop, skip_kd_loop;
    Label oc_loop, skip_oc_loop;

    for (int ii = 0; ii < nb_ic_block; ii++)
        for (int jj = 0; jj < ur_w; jj++) {
            uni_vpxor(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj),
                      Ymm(ur_w * ii + jj));
        }

    if (one_of(jcp.ndims, 3, 4)) {
        cmp(reg_channel_work, 0);
        jle(skip_oc_loop, T_NEAR);
        xor_(reg_channel, reg_channel);

        mov(aux_reg_ddst_oc_loop, reg_ddst);
        mov(aux_reg_kernel_oc_loop, reg_kernel);

        L(oc_loop);
        mov(aux_reg_ddst, aux_reg_ddst_oc_loop);
        mov(aux_reg_kernel, aux_reg_kernel_oc_loop);
    }

    if (jcp.ndims == 5) {
        assert(jcp.nb_oc_blocking == 1);
        push(oi_iter);

        mov(reg_ki, ptr[this->param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_ddst);
        mov(aux_reg_ker_d, ptr[this->param1 + GET_OFF(filt)]);

        L(kd_loop);
        mov(kj, ptr[this->param1 + GET_OFF(kh_padding)]);
    } else {
        mov(kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_ddst, aux_reg_dst_d);
        mov(aux_reg_kernel, aux_reg_ker_d);
    }

    Label kh_loop, skip_kh_loop;
    cmp(kj, 0);
    jle(skip_kh_loop, T_NEAR);
    L(kh_loop); {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow); // 0;
            int jj_end = get_iw_end(ur_w, ki, r_overflow); // ur_w;
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {

                for (int jj = jj_start ; jj < jj_end; jj += stride_w) {
                    int aux_output_offset
                      = (jj + jcp.l_pad - ki) / stride_w * jcp.oc_block + ofm2;
                    vbroadcastss(Ymm(nb_ic_block * ur_w + jj / stride_w),
                            ptr[aux_reg_ddst
                            + sizeof(float) * aux_output_offset]);
                }

                for (int ii = 0; ii  < nb_ic_block; ii++) {
                    int aux_kernel_offset
                        = ii * kd * kh * kw * jcp.ic_block * jcp.oc_block
                        + ki * jcp.ic_block * jcp.oc_block
                        + ofm2 * jcp.ic_block;
                    vmovups(ymm15,
                            ptr[aux_reg_kernel
                            + sizeof(float) * aux_kernel_offset]);
                    for (int jj = jj_start; jj  < jj_end; jj += stride_w)
                        vfmadd231ps(Ymm(ur_w * ii + jj),
                                Ymm(nb_ic_block * ur_w + jj / stride_w), ymm15);
                }
            }
        }
        add(aux_reg_kernel, sizeof(float) * stride_h * kw  * oc_block
                                          * ic_block);
        sub(aux_reg_ddst, sizeof(float) * ow * oc_block);

        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }
    L(skip_kh_loop);

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d,
                sizeof(float) * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add(aux_reg_ker_d,
                sizeof(float) * jcp.kw * jcp.kh * oc_block * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_loop, T_NEAR);
        L(skip_kd_loop);

        pop(oi_iter);
    }

    if (one_of(jcp.ndims, 3, 4)) {
        int ddst_oc_shift = sizeof(float) * jcp.od * jcp.oh * jcp.ow
                          * jcp.oc_block;
        int kernel_oc_shift = sizeof(float) * jcp.kd * jcp.kh * jcp.kw
                          * jcp.ic * jcp.oc_block;

        add(aux_reg_ddst_oc_loop, ddst_oc_shift);
        add(aux_reg_kernel_oc_loop, kernel_oc_shift);

        inc(reg_channel);
        cmp(reg_channel, reg_channel_work);
        jl(oc_loop, T_NEAR);

        L(skip_oc_loop);
        mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    }

    Label no_update_label;
    cmp(reg_channel, 0);
    je(no_update_label, T_NEAR);
    for (int ii = 0; ii < nb_ic_block; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            size_t offt =
                sizeof(float) * ((size_t)ii * id * ih * iw + jj) * ic_block;
            vmovups(Ymm(15),
                    make_safe_addr(reg_dsrc, offt, reg_long_offt));
            vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj),
                    Ymm(15));

        }
    }
    L(no_update_label);

    for (int ii = 0; ii < nb_ic_block; ii++)
        for (int jj = 0; jj < ur_w; jj++) {
            size_t offt =
                sizeof(float) * ((size_t)ii * id * ih * iw + jj) * ic_block;
            vmovups(make_safe_addr(reg_dsrc, offt, reg_long_offt),
                    Ymm(ur_w * ii + jj));
        }
}

void jit_avx2_conv_bwd_data_kernel_f32::generate() {
    preamble();

    mov(reg_dsrc, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_ddst, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    mov(reg_channel_work, ptr[param1 + GET_OFF(ch_blocks)]);

    int ddst_shift = sizeof(float) * (jcp.ur_w / jcp.stride_w) * jcp.ic_block;
    int dsrc_shift = sizeof(float) * jcp.ur_w * jcp.oc_block;

    int l_overflow = nstl::max(0, (jcp.kw - 1 - jcp.l_pad) / jcp.stride_w);
    int r_overflow = nstl::max(0, (jcp.kw - 1
                    - nstl::max(0, jcp.r_pad)) / jcp.stride_w);
    int r_overflow1 = nstl::max(0, (jcp.kw - 1
                    - nstl::max(0, jcp.r_pad) - jcp.ur_w_tail) / jcp.stride_w);

    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0)
        n_oi--;

    if (jcp.ur_w == jcp.iw) {
        compute_loop(jcp.ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(jcp.ur_w, l_overflow, r_overflow1);
        add(reg_dsrc, dsrc_shift);
        add(reg_ddst, ddst_shift);
        if (jcp.ur_w_tail != 0)
            compute_loop(jcp.ur_w_tail, 0, r_overflow);
    } else {
        xor_(oi_iter, oi_iter);
        if (l_overflow > 0) {
            compute_loop(jcp.ur_w, l_overflow, 0);
            add(reg_dsrc, dsrc_shift);
            add(reg_ddst, ddst_shift);
            inc(oi_iter);
        }

        if ((l_overflow <= 0 && n_oi > 0) || (l_overflow >  0 && n_oi > 1)) {
            Label ow_loop;
            L(ow_loop); {
                compute_loop(jcp.ur_w, 0, 0);
                add(reg_dsrc, dsrc_shift);
                add(reg_ddst, ddst_shift);
                inc(oi_iter);
                cmp(oi_iter, n_oi); jl(ow_loop, T_NEAR);
            }
        }

        if (r_overflow1 > 0 ) {
            compute_loop(jcp.ur_w, 0, r_overflow1);
            add(reg_dsrc, dsrc_shift);
            add(reg_ddst, ddst_shift);
        }

        if (jcp.ur_w_tail != 0)
            compute_loop(jcp.ur_w_tail, 0, r_overflow);
    }

    this->postamble();
}

status_t jit_avx2_conv_bwd_data_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(avx2)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;

    int ndims = diff_src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims-2];
    jcp.iw = diff_src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    const int simd_w = 8;

    /* derivatives */
    jcp.idp = jcp.id + 2 * jcp.f_pad;
    jcp.ihp = jcp.ih + 2 * jcp.t_pad;
    jcp.iwp = jcp.iw + 2 * jcp.l_pad;
    jcp.ohp = jcp.oh; /* do we really need */
    jcp.owp = jcp.ow; /* padded output ??? */

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1;

    /* gemm-based convolution performs better in these cases */
    if (jcp.ic < simd_w && jcp.kw > 3 && jcp.stride_w > 1)
        return status::unimplemented;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    jcp.ic_block = (jcp.ic % simd_w) ? 1 : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    if (jcp.oc % jcp.oc_block) return status::unimplemented;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.nb_ic_blocking = 1;
    jcp.nb_oc_blocking = 1;
    jcp.ur_w = 1;

    if(one_of(ndims, 3, 4) && jcp.ow < 40)
        jcp.nb_oc_blocking = jcp.ow < 15 ? 4 : 2;

    if (ndims == 3) {
        jcp.src_tag = diff_src_d.matches_one_of_tag(nCw8c);
        jcp.wei_tag = weights_d.matches_one_of_tag(OIw8i8o, gOIw8o8i);
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(nCw8c);
    } else if (ndims == 4) {
        jcp.src_tag = diff_src_d.matches_one_of_tag(nChw8c);
        jcp.wei_tag = weights_d.matches_one_of_tag(OIhw8o8i, gOIhw8o8i);
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(nChw8c);
    } else if (ndims == 5) {
        jcp.src_tag = diff_src_d.matches_one_of_tag(nCdhw8c);
        jcp.wei_tag = weights_d.matches_one_of_tag(OIdhw8o8i, gOIdhw8o8i);
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(nCdhw8c);
    }

    bool args_ok = true
        && one_of(jcp.src_tag, nCw8c, nChw8c, nCdhw8c)
        && one_of(jcp.wei_tag, gOIw8o8i, OIw8i8o, gOIhw8o8i, OIhw8o8i,
                gOIdhw8o8i, OIdhw8o8i)
        && one_of(jcp.dst_tag, nCw8c, nChw8c, nCdhw8c)
        && jcp.stride_w == jcp.stride_h
        && jcp.stride_d == 1
        && jcp.dilate_d == 0
        && jcp.dilate_h == 0
        && jcp.dilate_w == 0
        && jcp.ic % simd_w == 0
        && jcp.oc % simd_w == 0
        && jcp.od == (jcp.idp - jcp.kd) / jcp.stride_d + 1
        && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
        && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1;
    if (!args_ok) return status::unimplemented;
    jcp.r_pad = (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad;
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih - jcp.t_pad;
    int l_overflow = nstl::max(0, (jcp.kw - 1 - jcp.l_pad) / jcp.stride_w);

    const int max_regs = 15; /* Maximun number of registers available for
                                result accumulation and delta dst data.
                                One additional register is reserved for weights
                                data. */

    /* Find the best blocking with maximum number of fma instructions
       per ur_w * nb_ic_blocking compute loops. Number of required registers
       is num_regs = ur_w * nb_ic_blocking + ur_w / stride_w <= max_regs.
       ur_w must be divisible by stride_w */
    if (jcp.stride_w + 1 > max_regs)  /* Minimal possible registers
                                         distribution exceeds max_regs */
        return status::unimplemented;

    int best_nfmas = 0;
    for (int b = 1; b <= 4; b++)
    {
        if (jcp.nb_ic % b != 0)
            continue;

        for (int u = jcp.stride_w;
             u * b + u / jcp.stride_w <= max_regs && u < jcp.iw + jcp.stride_w;
             u += jcp.stride_w)
        {
            int ur_w = nstl::min(u, jcp.iw);
            /* maximum 1 step with l_overflow so far */
            if (l_overflow * jcp.stride_w > ur_w && ur_w != jcp.iw)
                continue;
            int nfmas = utils::div_up(ur_w, jcp.stride_w) * b;
            if (nfmas > best_nfmas
               || (nfmas == best_nfmas && jcp.ur_w < ur_w)) {
                jcp.ur_w = ur_w;
                jcp.nb_ic_blocking = b;
                best_nfmas = nfmas;
            }
        }
    }
    if (best_nfmas == 0) /* can't find appropriate blocking */
        return status::unimplemented;

    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    int r_overflow_no_tail = nstl::max(0, (jcp.kw - 1 - jcp.ur_w_tail
                    - nstl::max(0, jcp.r_pad) - jcp.ur_w_tail) / jcp.stride_w);
    /* maximum 1 ur_w block with r_overflow so far */
    if (r_overflow_no_tail * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;

    if ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
        return status::unimplemented;

    return status::success;
}

void jit_avx2_conv_bwd_data_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

void jit_avx2_conv_bwd_weights_kernel_f32::generate() {
    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    compute_oh_loop_common();
    this->postamble();
}

status_t jit_avx2_conv_bwd_weights_kernel_f32::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &diff_weights_d,
        const memory_desc_wrapper &diff_dst_d) {
    if (!mayiuse(avx2)) return status::unimplemented;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : diff_weights_d.dims()[with_groups + ndims-2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims-1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    if (ndims == 3) {
        jcp.src_tag = src_d.matches_one_of_tag(ncw, nwc, nCw8c);
        jcp.wei_tag = diff_weights_d.matches_one_of_tag(
                Owi8o, gOwi8o, OIw8i8o, gOIw8i8o);
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(nCw8c);
    } else if (ndims == 4) {
        jcp.src_tag = src_d.matches_one_of_tag(nchw, nhwc, nChw8c);
        jcp.wei_tag = diff_weights_d.matches_one_of_tag(
                Ohwi8o, gOhwi8o, OIhw8i8o, gOIhw8i8o);
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(nChw8c);
    } else if (ndims == 5) {
        jcp.src_tag = src_d.matches_one_of_tag(ncdhw, ndhwc, nCdhw8c);
        jcp.wei_tag = diff_weights_d.matches_one_of_tag(
                Odhwi8o, gOdhwi8o, OIdhw8i8o, gOIdhw8i8o);
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(nCdhw8c);
    }
    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;

    const bool flat = jcp.ic == 3;
    const bool mimo = !flat;

    const int simd_w = 8;

    jcp.b_pad = nstl::max(
            0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih - jcp.t_pad);
    jcp.r_pad = nstl::max(
            0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);

    int back_pad = nstl::max(0, (jcp.od - 1) * jcp.stride_d + jcp.kd - jcp.id
        - jcp.f_pad);
    if (ndims == 5)
        if (jcp.f_pad != 0 || back_pad != 0)
            return status::unimplemented;

    const int max_h_pad = ((jcp.kh - 1) * (jcp.dilate_h + 1) + 1);
    const int max_w_pad = ((jcp.kw - 1) * (jcp.dilate_w + 1) + 1);
    const bool boundaries_ok = true
        && jcp.t_pad < max_h_pad && jcp.b_pad < max_h_pad
        && jcp.l_pad < max_w_pad && jcp.r_pad < max_w_pad;
    if (!boundaries_ok)
        return status::unimplemented;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        if (mimo)
            jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    bool args_ok = true
        && IMPLICATION(flat, true
                && one_of(jcp.src_tag, ncw, nwc, nchw, nhwc, ncdhw, ndhwc)
                && one_of(jcp.wei_tag, Owi8o, gOwi8o, Ohwi8o, gOhwi8o, Odhwi8o,
                    gOdhwi8o))
        && IMPLICATION(mimo, true
                && one_of(jcp.src_tag, nCw8c, nChw8c, nCdhw8c)
                && one_of(jcp.wei_tag, OIw8i8o, gOIw8i8o, OIhw8i8o, gOIhw8i8o,
                    OIdhw8i8o, gOIdhw8i8o))
        && one_of(jcp.dst_tag, nCw8c, nChw8c, nCdhw8c)
        && IMPLICATION(mimo, jcp.ic % simd_w == 0)
        && jcp.oc % simd_w == 0
        && jcp.kw < 14
        && jcp.kh <= jcp.t_pad + jcp.ih /* [bwd_w:r1] */
        && jcp.kh <= jcp.ih /* [bwd_w:r2] */
        && jcp.kd <= jcp.f_pad + jcp.id
        && jcp.kd <= jcp.id
        && jcp.t_pad < jcp.kh /* XXX: must fix the kernel! */
        && jcp.dilate_d == 0
        && jcp.dilate_h == 0
        && jcp.dilate_w == 0;
    if (!args_ok) return status::unimplemented;

    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    return status::success;
}

void jit_avx2_conv_bwd_weights_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, sizeof(float) * jcp.oc);
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::od_step_comeback_pointers()
{
    Label kd_comeback_loop;
    mov(kj, jcp.kd); //FIXME (Anton): this works only if f_pad = back_pad = 0
    L(kd_comeback_loop); {
        const int inp_mult = one_of(jcp.src_tag, ncw, nchw, ncdhw)
            ? 1 : jcp.ic_block;
        sub(aux_reg_input, sizeof(float) * jcp.iw * jcp.ih * inp_mult);
        sub(aux_reg_kernel, sizeof(float) * jcp.kw * jcp.kh * jcp.ic_block
                * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kd_comeback_loop, T_NEAR);
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers()
{
    mov(kj, reg_kh);
    Label kh_comeback_loop;
    L(kh_comeback_loop); {
        const int inp_mult = one_of(jcp.src_tag, ncw, nchw, ncdhw)
            ? 1 : jcp.ic_block;
        sub(reg_input, sizeof(float) * jcp.iw * inp_mult);
        sub(reg_kernel, sizeof(float) * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_loop, T_NEAR);
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_ic_block_step(
        int ur_w, int pad_l, int pad_r, int ic_block_step, int input_offset,
        int kernel_offset, int output_offset)
{
    const int kw = jcp.kw;
    const int ic_block = jcp.ic_block;
    const int oc_block = jcp.oc_block;
    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            size_t off
                = sizeof(float) * (i_kw * ic_block + i_ic) * jcp.oc_block
                + kernel_offset;
            vmovups(Ymm(i_kw * ic_block_step + i_ic), yword[reg_kernel + off]);
        }

    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        vmovups(Ymm(kw * ic_block_step + 0),
                yword[reg_output
                + sizeof(float) * i_ur * oc_block + output_offset]);

        for (int i_kw = 0; i_kw < kw; i_kw++) {
            int i_iw = i_ur * jcp.stride_w + i_kw;
            if (i_iw - pad_l < 0
                    || i_iw > (ur_w - 1) * jcp.stride_w + kw - 1 - pad_r)
                continue;
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                size_t i_off = (size_t)input_offset + sizeof(float)*(
                    one_of(jcp.src_tag, ncw, nchw, ncdhw)
                        ? (i_iw - pad_l) + i_ic
                        * ((size_t)jcp.id * jcp.ih * jcp.iw)
                        : (i_iw - pad_l) * ic_block + i_ic);
                vbroadcastss(Ymm(kw * ic_block_step + 1),
                        make_safe_addr(reg_input, i_off, reg_long_offt));
                vfmadd231ps(Ymm(i_kw * ic_block_step + i_ic),
                        Ymm(kw * ic_block_step + 0),
                        Ymm(kw * ic_block_step + 1));
            }
        }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            size_t off
                = sizeof(float) * (i_kw * ic_block + i_ic) * jcp.oc_block
                + kernel_offset;
            vmovups(yword[reg_kernel + off],
                    Ymm(i_kw * ic_block_step + i_ic));
        }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_step_disp()
{
    int ic_block_step;
    if (one_of(jcp.src_tag, ncw, nchw, ncdhw)) {
        ic_block_step = jcp.kw >= 5 ? 1 : jcp.ic_block;
    } else {
        ic_block_step = jcp.kw > 7 ? 1
        : jcp.kw > 3 ? 2
        : jcp.kw > 1 ? 4 : 8;
    }

    const int max_ur_w = jcp.ow > 56 ? 14 : 28;

    if (jcp.ow <= max_ur_w)
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else
        compute_oh_step_common(ic_block_step, max_ur_w);

    if (jcp.ndims == 5) {
        od_step_comeback_pointers();
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    } else {
        oh_step_comeback_pointers();
    }
}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_step_unroll_ow(
        int ic_block_step, int max_ur_w)
{
    UNUSED(max_ur_w);

    const int ic_block = jcp.ic_block;
    const int oc_block = jcp.oc_block;
    int inp_mul = one_of(jcp.src_tag, ncw, nchw, ncdhw) ? 1 : jcp.ic_block;
    Label kd_loop;

    const int r_pad
        = nstl::max(0,
                (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);

    if (jcp.ndims == 5) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
        mov(ki, jcp.kd);
        L(kd_loop);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    Label kh_loop;
    L(kh_loop); {
        xor_(b_ic, b_ic);
        Label ic_block_loop;
        L(ic_block_loop); {
            compute_ic_block_step(jcp.ow, jcp.l_pad, r_pad, ic_block_step, 0,
                    0, 0);
            size_t inp_icblk_stride = sizeof(float) * ic_block_step
                * (one_of(jcp.src_tag, ncw, nchw, ncdhw)
                ? jcp.id*jcp.ih*jcp.iw : 1);
            safe_add(reg_input, inp_icblk_stride, reg_long_offt);
            add(reg_kernel, sizeof(float) * ic_block_step * oc_block);
            add(b_ic, ic_block_step);
            cmp(b_ic, ic_block);
            jl(ic_block_loop, T_NEAR);
        }
        if(one_of(jcp.src_tag, ncw, nchw, ncdhw)) {
            size_t offt = sizeof(float) * jcp.id * jcp.ih * jcp.iw * ic_block;
            safe_sub(reg_input, offt, reg_long_offt);
            add(reg_input, sizeof(float) * jcp.iw);
        } else {
            add(reg_input, sizeof(float) * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel, sizeof(float) * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_input, sizeof(float) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_kernel, sizeof(float) * jcp.kh * jcp.kw * ic_block
            * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_loop, T_NEAR);
    }

}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_step_common(
        int ic_block_step, int max_ur_w)
{
    const int ic_block = jcp.ic_block;
    const int oc_block = jcp.oc_block;
    const int stride_w = jcp.stride_w;
    int inp_mul = one_of(jcp.src_tag, ncw, nchw, ncdhw) ? 1 : jcp.ic_block;
    Label kd_loop;

    const int r_pad = jcp.r_pad;

    int ur_w = nstl::min(jcp.ow, max_ur_w);
    int ur_w_trips = jcp.ow / ur_w;
    int ur_w_tail = jcp.ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0) || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }
    const int inp_mult = one_of(jcp.src_tag, ncw, nchw, ncdhw) ? 1 : ic_block;

    int input_comeback = (ur_w_trips * ur_w * stride_w - jcp.l_pad) * inp_mult;
    int output_comeback = ur_w_trips * ur_w * oc_block;

    if (jcp.ndims == 5) {
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
        mov(ki, jcp.kd);
        L(kd_loop);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    Label kh_loop;
    L(kh_loop); {
        xor_(b_ic, b_ic);
        Label ic_block_loop;
        L(ic_block_loop); {
            if (jcp.l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(ur_w,
                        jcp.l_pad, 0, ic_block_step, 0, 0, 0);
                add(reg_input, sizeof(float)
                        * (ur_w * stride_w - jcp.l_pad) * inp_mult);
                add(reg_output, sizeof(float) * ur_w * oc_block);
            }

            if (ur_w_trips > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                Label ow_block_loop;
                L(ow_block_loop); {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add(reg_input, sizeof(float) * ur_w * stride_w * inp_mult);
                    add(reg_output, sizeof(float) * ur_w * oc_block);

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_trips);
                    jl(ow_block_loop, T_NEAR);
                }
            }

            if (ur_w_tail > 0)
                compute_ic_block_step(ur_w_tail,
                        0, r_pad, ic_block_step, 0, 0, 0);

            sub(reg_input, sizeof(float) * input_comeback);
            sub(reg_output, sizeof(float) * output_comeback);

            size_t inp_icblk_stride = sizeof(float) * ic_block_step
                * (one_of(jcp.src_tag, ncw, nchw, ncdhw)
                ? jcp.id*jcp.ih*jcp.iw : 1);
            safe_add(reg_input, inp_icblk_stride, reg_long_offt);
            add(reg_kernel, sizeof(float) * ic_block_step * oc_block);

            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_loop, T_NEAR);
        }
        if (one_of(jcp.src_tag, ncw, nchw, ncdhw)) {
            size_t offt = sizeof(float) * jcp.id * jcp.ih * jcp.iw * ic_block;
            safe_sub(reg_input, offt, reg_long_offt);
            add(reg_input, sizeof(float) * jcp.iw);
        } else {
            add(reg_input, sizeof(float) * (jcp.iw - 1) * ic_block);
        }
        add(reg_kernel, sizeof(float) * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_loop, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_input, sizeof(float) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_kernel, sizeof(float) * jcp.kh * jcp.kw * ic_block
            * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_loop, T_NEAR);
    }

}

inline void jit_avx2_conv_bwd_weights_kernel_f32::compute_oh_loop_common()
{
    const int icoc_block = jcp.ic_block * jcp.oc_block;
    const int t_pad = jcp.t_pad;
    const int stride_h = jcp.stride_h;
    const int inp_mult = one_of(jcp.src_tag, ncw, nchw, ncdhw)
        ? 1 : jcp.ic_block;
    int b_pad = jcp.b_pad;

    Label oh_tpad_loop, oh_loop, oh_loop_end;

    mov(reg_kh, jcp.kh);
    xor_(reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj);
    if (t_pad > 0) {
        assert(jcp.kh <= t_pad + jcp.ih); /* [bwd_w:r1] */
        mov(reg_kh, jcp.kh <= t_pad + jcp.ih ? jcp.kh - t_pad : jcp.ih);
        add(reg_kernel, sizeof(float) * t_pad * jcp.kw * icoc_block);

        L(oh_tpad_loop); {
            compute_oh_step_disp();
            add(reg_output, sizeof(float) * jcp.ow * jcp.oc_block);
            sub(reg_kernel, sizeof(float) * stride_h * jcp.kw * icoc_block);

            inc(reg_oj);
            add(reg_ih_count, stride_h);
            add(reg_kh, stride_h);

            /* the overlap between input and kernel may not reach kernel size.
             * so far we do not support that (until we put constant here) */
            const int final_inp_ker_overlap = jcp.kh; /* [bwd_w:r2] */
            cmp(reg_kh, final_inp_ker_overlap);
            jl(oh_tpad_loop, T_NEAR);
        }

        if (t_pad % stride_h != 0) {
            int inp_corr = stride_h - t_pad % stride_h;
            add(reg_kernel, sizeof(float) * inp_corr * jcp.kw * icoc_block);
            add(reg_input, sizeof(float) * inp_corr * jcp.iw * inp_mult);
        }
    }
    cmp(reg_ih_count, jcp.ih + t_pad - jcp.kh + 1);
    jge(oh_loop_end, T_NEAR);
    cmp(reg_oj, jcp.oh);
    jge(oh_loop, T_NEAR);

    mov(reg_kh, jcp.kh);
    L(oh_loop); {
        compute_oh_step_disp();
        add(reg_input, sizeof(float) * stride_h * jcp.iw * inp_mult);
        add(reg_output, sizeof(float) * jcp.ow * jcp.oc_block);

        inc(reg_oj);
        add(reg_ih_count, stride_h);

        cmp(reg_ih_count, jcp.ih + t_pad - jcp.kh + 1);
        jge(oh_loop_end, T_NEAR);

        cmp(reg_oj, jcp.oh);
        jl(oh_loop, T_NEAR);
    }
    L(oh_loop_end);
    if (b_pad > 0) {
        Label oh_bpad_loop, oh_bpad_loop_end;
        cmp(reg_oj, jcp.oh);
        jge(oh_bpad_loop_end, T_NEAR);

        mov(reg_kh, jcp.ih + t_pad);
        sub(reg_kh, reg_ih_count);
        L(oh_bpad_loop); {
            compute_oh_step_disp();
            add(reg_input, sizeof(float) * stride_h * jcp.iw * inp_mult);
            add(reg_output, sizeof(float) * jcp.ow * jcp.oc_block);

            sub(reg_kh, stride_h);
            cmp(reg_kh, 0);
            jle(oh_bpad_loop_end, T_NEAR);

            inc(reg_oj);
            cmp(reg_oj, jcp.oh);
            jl(oh_bpad_loop, T_NEAR);
        }
        L(oh_bpad_loop_end);
    }
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
