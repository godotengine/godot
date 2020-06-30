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
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_barrier.hpp"

#include "jit_avx512_common_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)
#define KNx_L2_EFFECTIVE_CAPACITY ((512-64)*1024)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::format_tag;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

namespace {

constexpr auto small_spatial = 14;
unsigned int L1_cache_size = get_cache_size(1, true);

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(jcp.prop_kind,
                forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    // ow-threading is currently implemented for forward only
    // TODO: single code for fwd and bwd after ow-thr for bwd
    // meaningless switch was removed
    if (jcp.prop_kind == backward_data) {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial)
            ? loop_cgn : loop_gnc;
    } else {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial)
            ? loop_cwgn : loop_gncw;
    }
}

inline bool is_1stconv(const jit_conv_conf_t &jcp) {
    if (mayiuse(avx512_core))
        return (jcp.ic < 16 && jcp.ngroups == 1);
    else
        return one_of(jcp.ic, 1, 3);
}

inline bool is_ow_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}

inline bool is_owb_prefetching(const jit_conv_conf_t &jcp) {
    return (jcp.ver == ver_4fma && is_ow_threading_on(jcp));
}

}

template<typename Vmm>
void _jit_avx512_common_conv_fwd_kernel<Vmm>::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            vpxord(vmm, vmm, vmm);
            if (!is_owb_prefetching(jcp)) {
                size_t aux_output_offset = get_output_offset(j, k);
                mic_prefetcht1(EVEX_compress_addr_safe(reg_out_prf,
                            aux_output_offset, reg_out_long_offt));
            }
        }
}

template<typename Vmm>
void _jit_avx512_common_conv_fwd_kernel<Vmm>::store_output(int ur_w)
{
    Label no_update_label, store_label, eltwise_label;

    mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    }

    if (!jcp.with_sum) {
        cmp(reg_channel, 0);
        je(no_update_label, T_NEAR);
    }

    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            size_t aux_output_offset = get_output_offset(j, k);
            vaddps(vmm,
                make_safe_addr(reg_out, aux_output_offset, reg_out_long_offt));
        }

    if (!jcp.with_sum) {
        jmp(eltwise_label, T_NEAR);
    } else {
        cmp(reg_channel, 0);
        jne(eltwise_label, T_NEAR);
    }

    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_out * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                Vmm vmm = vmm_out(j, k);
                vaddps(vmm, EVEX_compress_addr(reg_bias, bias_offset));
            }
            mic_prefetcht1(EVEX_compress_addr(reg_bias, bias_offset + 64));
        }
    }

    L(eltwise_label);
    if (jcp.with_eltwise) {
        cmp(reg_channel, jcp.nb_ic - 1);
        jl(store_label, T_NEAR);

        if (ur_w == jcp.ur_w) {
            eltwise_injector_->compute_vector_range(0,
                    jcp.nb_oc_blocking * jcp.ur_w);
        } else {
            for (int k = 0; k < jcp.nb_oc_blocking; k++)
                eltwise_injector_->compute_vector_range(k * jcp.ur_w,
                        k * jcp.ur_w + ur_w);
        }
    }

    L(store_label);
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            size_t aux_output_offset = (size_t)typesize *
                ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
            vmovups(EVEX_compress_addr_safe(reg_out, aux_output_offset,
                        reg_out_long_offt), vmm);
            if (!is_owb_prefetching(jcp))
                mic_prefetcht0(EVEX_compress_addr_safe(reg_out_prf,
                            aux_output_offset, reg_out_long_offt));
        }
}

template<typename Vmm>
void _jit_avx512_common_conv_fwd_kernel<Vmm>::compute_loop_4fma_1st(int ur_w,
    int pad_l, int pad_r)
{
}

template<>
void _jit_avx512_common_conv_fwd_kernel<Zmm>::compute_loop_4fma_1st(int ur_w,
        int pad_l, int pad_r)
{
    assert(jcp.dilate_d == 0 && jcp.dilate_h == 0 && jcp.dilate_w == 0);

    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    Label kh_label, kd_label;

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_inp_prf, reg_inp_prf);
    }

    size_t max_input_offset = (size_t)jcp.typesize_in
        * ((size_t)(kw + ur_w * stride_w - pad_l)
                + (size_t)ic_block * iw * ih * jcp.id);
    assert(reg_inp_prf == reg_long_offt);
    if (max_input_offset > INT_MAX) push(reg_inp_prf);

    if (jcp.ndims == 5) {
        push(reg_out_prf);
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);
        mov(aux_reg_inp_d_prf, reg_inp_prf);

        L(kd_label);
    }
    mov(reg_kj, reg_kh);
    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_inp_prf, aux_reg_inp_d_prf);
    }

    L(kh_label);
    for (int ki = 0; ki < kw; ki += 4) {
        for (int ic = 0; ic < ic_block; ic++) {
            for (int i = 0; i < 4; i++) {
                int aux_ker_offset
                        = jcp.typesize_in
                        * ((ki + i) * oc_block
                                  + ic * kw * jcp.kh * jcp.kd * oc_block);
                if (ki + i < kw)
                    vmovups(vmm_ker(i),
                        EVEX_compress_addr(aux_reg_ker, aux_ker_offset));
                else
                    vpxord(vmm_ker(i), vmm_ker(i), vmm_ker(i));
            }

            int j_start = get_ow_start(ki, pad_l);
            int j_end = get_ow_end(ur_w, ki, pad_r);

            for (int j = j_start, prf_count=0; j < j_end; j++) {
                size_t aux_input_offset = (size_t)jcp.typesize_in
                        * ((size_t)(ki + j * stride_w
                            - pad_l) + (size_t)ic * iw * ih * jcp.id);
                v4fmaddps(vmm_out(j, 0), vmm_ker(0),
                        EVEX_compress_addr_safe(aux_reg_inp, aux_input_offset,
                        reg_long_offt));
                if (ki + prf_count < kw && prf_count < 4
                    && ((ki < 2 && j % 4) || j % 2)) {
                    int aux_ker_offset = jcp.typesize_in
                        * ((ki + prf_count) * oc_block
                        + ic * kw * jcp.kh * jcp.kd * oc_block + kw * oc_block);
                    mic_prefetcht0(EVEX_compress_addr(aux_reg_ker,
                        aux_ker_offset));
                    prf_count++;
                }
                if (ki == 0
                    && j % (64 / (stride_w * jcp.typesize_in)) == 0) {
                    mic_prefetcht0(EVEX_compress_addr_safe(aux_reg_inp_prf,
                        aux_input_offset, reg_long_offt));
                }
                if (ki == 1
                    && j % (64 / (stride_w * jcp.typesize_in)) == 0) {
                    mic_prefetcht0(EVEX_compress_addr_safe(aux_reg_inp,
                        aux_input_offset+jcp.typesize_in * iw, reg_long_offt));
                }
            }
        }
    }
    add(aux_reg_ker, jcp.typesize_in * kw * oc_block);
    add(aux_reg_inp, jcp.typesize_in * iw);
    add(aux_reg_inp_prf, jcp.typesize_in * iw);

    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d, typesize * jcp.ih * jcp.iw);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * oc_block);
        add(aux_reg_inp_d_prf, typesize * jcp.ih * jcp.iw);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);

        pop(reg_out);
        pop(reg_out_prf);
    }

    if (max_input_offset > INT_MAX) pop(reg_inp_prf);
}

template<typename Vmm>
void _jit_avx512_common_conv_fwd_kernel<Vmm>::compute_loop_4fma(int ur_w,
    int pad_l, int pad_r)
{
}

template<>
void _jit_avx512_common_conv_fwd_kernel<Zmm>::compute_loop_4fma(int ur_w,
        int pad_l, int pad_r)
{
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label, last_iter_label, loop_end_label, kd_label;
    int ker_load_number = 4;
    int shift_kernel_ptr = typesize * jcp.kw * jcp.oc_block * jcp.ic_block;
    int shift_input_ptr = typesize * (jcp.dilate_h + 1) * jcp.iw * jcp.ic_block;

    bool check_last_kh = (jcp.kh > 3);
    bool pref_current_inp = (jcp.iw < 14 || jcp.iw > 28);

    int oi_ipref_t0 = get_ow_start(0, pad_l);
    int ow_end_ipref = get_ow_end(ur_w, 0, pad_r);

    assert(jcp.oc % jcp.nb_oc_blocking == 0);

    auto kernel_offset = [=](int ocb, int ic, int ki) {
        int blk_idx = ocb * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int ic_offset = ic * jcp.oc_block;
        return typesize * (blk_offset + ic_offset);
    };
    auto kernel_loads = [=](int ki, int ic, int kk) {
        for (int ii = 0; ii < ker_load_number; ii++) {
            int aux_kernel_offset = kernel_offset(kk, ic + ii, ki);
            vmovups(vmm_ker(ii),
                EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
        }
    };
    auto prefetch_inp_next_kh = [&](int ki, int ki_start, int cnt0, int cnt1) {
        if (cnt1 >= ker_load_number && cnt0 >= ker_load_number
            && ki >= ki_start && oi_ipref_t0 < ow_end_ipref) {
            int aux_inp_offset
                    = typesize
                    * ((oi_ipref_t0 * stride_w - pad_l) * ic_block
                              + (jcp.dilate_h + 1) * jcp.iw * ic_block);
            prefetcht0(EVEX_compress_addr(aux_reg_inp,
                    aux_inp_offset));
            oi_ipref_t0++;
        }
    };

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_ker_prf, reg_ker_prf);
        mov(aux_reg_inp_prf, reg_inp_prf);
    }

    if (jcp.ndims == 5) {
        push(reg_out_prf);
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);
        mov(aux_reg_inp_d_prf, reg_inp_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);
        L(kd_label);
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }
    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
        mov(aux_reg_inp_prf, aux_reg_inp_d_prf);
    }

    align(16);
    L(kh_label);
    int kw = jcp.kw;
    if (check_last_kh) {
        for (int ki = 0; ki < kw; ki++)
            for (int ic = 0; ic < ic_block; ic += 4)
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    bool last_kernel_loads = (kk == jcp.nb_oc_blocking - 1
                        && ki == kw - 1 && (ic + 4) == ic_block);

                    if (last_kernel_loads) {
                        cmp(reg_kj, 1);
                        je(last_iter_label, T_NEAR);
                    }

                    kernel_loads(ki, ic, kk);
                    for (int oi = get_ow_start(ki, pad_l), prf_count_t1 = 0,
                             prf_count_t0 = 0;
                            oi < get_ow_end(ur_w, ki, pad_r); oi++) {
                        int aux_input_offset = typesize
                                * ((ki * (jcp.dilate_w + 1) + oi * stride_w
                                           - pad_l) * ic_block
                                                       + ic);
                        v4fmaddps(vmm_out(oi, kk), vmm_ker(0),
                            EVEX_compress_addr(aux_reg_inp, aux_input_offset));

                        if (oi % 2) {
                            if (prf_count_t0 < 4) {
                                int aux_kernel_prf;
                                if (last_kernel_loads)
                                    aux_kernel_prf= kernel_offset(0,
                                        prf_count_t0 + ic + 4
                                        - ic_block, 0) + typesize * kw
                                        * oc_block * ic_block;
                                else
                                    aux_kernel_prf = kernel_offset(kk, ic + 4
                                        + prf_count_t0, ki);
                                mic_prefetcht0(EVEX_compress_addr(aux_reg_ker,
                                    aux_kernel_prf));
                                prf_count_t0++;
                            } else if (prf_count_t1 < 4) {
                                mic_prefetcht1(EVEX_compress_addr(
                                    aux_reg_ker_prf, kernel_offset(kk, ic
                                    + prf_count_t1, ki)));
                                prf_count_t1++;
                            }
                        } else
                           prefetch_inp_next_kh(ki, 2, prf_count_t0,
                               prf_count_t1);
                    }

                    if (last_kernel_loads) {
                        jmp(loop_end_label, T_NEAR);

                        L(last_iter_label);

                        kernel_loads(ki, ic, kk);
                        for (int oi = get_ow_start(ki, pad_l), prf_count_t1 = 0,
                                 prf_count_t0 = 0;
                                oi < get_ow_end(ur_w, ki, pad_r); oi++) {
                            int aux_input_offset = typesize
                                    * ((ki * (jcp.dilate_w + 1) + oi * stride_w
                                               - pad_l) * ic_block
                                                           + ic);
                            v4fmaddps(vmm_out(oi, kk), vmm_ker(0),
                                EVEX_compress_addr(aux_reg_inp,
                                    aux_input_offset));
                            if (oi % 2) {
                                if (prf_count_t0 < 4) {
                                    mic_prefetcht0(EVEX_compress_addr(
                                        aux_reg_ker_prf, kernel_offset(0,
                                        prf_count_t0, 0)));
                                    prf_count_t0++;
                                } else if (prf_count_t1 < 4) {
                                    mic_prefetcht1(EVEX_compress_addr(
                                        aux_reg_ker_prf, kernel_offset(kk,
                                        ic + prf_count_t1, ki)));
                                    prf_count_t1++;
                                }
                            }
                        }
                        L(loop_end_label);
                    }
                }
    } else {
        for (int ki = 0; ki < kw; ki++)
            for (int ic = 0; ic < ic_block; ic += 4)
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    kernel_loads(ki, ic, kk);
                    for (int oi = get_ow_start(ki, pad_l),
                            prf_count_t1 = 0, prf_count_t0 = 0;
                            oi < get_ow_end(ur_w, ki, pad_r); oi++) {
                        int aux_input_offset = typesize
                                * ((ki * (jcp.dilate_w + 1) + oi * stride_w
                                - pad_l) * ic_block + ic);
                        v4fmaddps(vmm_out(oi, kk), vmm_ker(0),
                            EVEX_compress_addr(aux_reg_inp,
                                aux_input_offset));

                        if (!is_owb_prefetching(jcp)) {
                            if ((oi % 2) && (prf_count_t1 < 4)) {
                                mic_prefetcht1(EVEX_compress_addr(
                                    aux_reg_ker_prf, kernel_offset(kk,
                                    ic + prf_count_t1, ki)));
                                prf_count_t1++;
                            }
                        } else {
                            if (!(ki == 0 && ic == 0)
                                && !(ki == kw-1 && ic == 0) &&
                                (oi % 2) && (prf_count_t1 < 4)
                                ) {
                                mic_prefetcht0(EVEX_compress_addr(
                                    aux_reg_ker, kernel_offset(kk,
                                    ic + 4 + prf_count_t0, ki)));
                                prf_count_t0++;
                            }
                        }
                        if (!is_owb_prefetching(jcp)) {
                            if (pref_current_inp) {
                                if (ki == 0 && ic == 0 && kk == 0)
                                    mic_prefetcht0(EVEX_compress_addr(
                                        aux_reg_inp,
                                        aux_input_offset + shift_input_ptr));
                            } else {
                                if (ki == 1 && ic == 0 && kk == 0)
                                    mic_prefetcht1(EVEX_compress_addr(
                                        aux_reg_inp_prf, aux_input_offset));
                            }
                        } else {
                            int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
                            int inp_shift
                                = jcp.typesize_in * ur_w * stride_w * inp_mult;
                            bool kk_pref_slot = kk ? oi % 2 : !(oi % 2);
                            if (ki == 0 && ic == 0 && kk_pref_slot)
                                    mic_prefetcht1(EVEX_compress_addr(
                                        aux_reg_inp,
                                        aux_input_offset + inp_shift));

                            if (ki == kw - 1 && ic == 0 && kk_pref_slot)
                                    mic_prefetcht0(EVEX_compress_addr(
                                        aux_reg_inp,
                                        aux_input_offset + inp_shift));
                        }
                    }
                }
    }

    add(aux_reg_ker, shift_kernel_ptr);
    add(aux_reg_inp, shift_input_ptr);
    add(aux_reg_ker_prf, shift_kernel_ptr);
    add(aux_reg_inp_prf, shift_input_ptr);

    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * jcp.ic_block);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);
        add(aux_reg_inp_d_prf,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * jcp.ic_block);
        add(aux_reg_ker_d_prf, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);

        pop(reg_out);
        pop(reg_out_prf);
    }
}

template<typename Vmm>
void _jit_avx512_common_conv_fwd_kernel<Vmm>::compute_loop_fma(int ur_w,
        int pad_l, int pad_r)
{
    bool prf_ker = true;
    bool prf_inp = true;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;
    int id = jcp.id;
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label, kd_label;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = ic_block * nb_oc_block * kw;
    int num_ker_prfs = prf_ker ? num_ker_loads : 0;
    int num_inp_prfs = prf_inp ?
            ur_w * nstl::min(kw, stride_w) + nstl::max(0, kw - stride_w) :
            0;
    if (jcp.is_1stconv && prf_inp) {
        num_inp_prfs = div_up(num_inp_prfs, jcp.simd_w) * ic_block;
    }
    int num_prfs = num_ker_prfs + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w;
    int prf_inst_spacing
            = (prf_ker || prf_inp) ? nstl::max(1, num_fmas / num_prfs) : 1;
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_inp_prf, reg_inp_prf);
        mov(aux_reg_ker_prf, reg_ker_prf);
    }

    size_t max_input_offset = (size_t)jcp.typesize_in * ic_block * iw * ih * id;
    assert(reg_inp_prf == reg_long_offt);
    if (max_input_offset > INT_MAX) push(reg_inp_prf);


    if (jcp.ndims == 5) {
        push(reg_out_prf);
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);
        mov(aux_reg_inp_d_prf, reg_inp_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        L(kd_label);
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
        mov(aux_reg_inp_prf, aux_reg_inp_d_prf);
    }

    align(16);
    L(kh_label);
    {
        int step = 0;
        int ker_prfs = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int ic = 0; ic < ic_block; ic++) {
                int aux_kernel_offset = 0;
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        aux_kernel_offset = get_kernel_offset(ki, ic, 0, i);
                        vmovups(vmm_ker(i), EVEX_compress_addr(
                                        aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                        = (step + load_offset) % ker_pipeline_depth;
                    aux_kernel_offset
                            = get_kernel_offset(ki, ic, 0, load_offset);
                    vmovups(vmm_ker(ker_load_reg_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }

                bool ker_prf_inserted = false;
                Vmm vmm_kernel = vmm_ker(step % ker_pipeline_depth);
                int j_start = get_ow_start(ki, pad_l);
                int j_end = get_ow_end(ur_w, ki, pad_r);
                for (int j = j_start; j < j_end; j++) {
                    size_t aux_input_offset = get_input_offset(ki, ic, j, pad_l);
                    auto addr = EVEX_compress_addr_safe(aux_reg_inp,
                            aux_input_offset, reg_long_offt, true);
                    vfmadd231ps(vmm_out(j, 0), vmm_kernel, addr);
                    int fma_idx = step * ur_w + j;
                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (prf_ker && !ker_prf_inserted
                                && ker_prfs < num_ker_prfs) {
                            int ker_prf_offset
                                    = jcp.typesize_in * ker_prfs * jcp.oc_block;
                            mic_prefetcht2(EVEX_compress_addr(
                                    aux_reg_ker_prf, ker_prf_offset));
                            ker_prf_inserted = true;
                            ker_prfs++;
                        } else if (prf_inp) {
                            int inp_prf_idx = prf_slot_idx - ker_prfs;
                            if (inp_prf_idx < num_inp_prfs) {
                                size_t inp_prf_stride = nstl::max(kw, stride_w);
                                size_t inp_prf_offset;
                                if (!jcp.is_1stconv) {
                                    inp_prf_offset
                                            = ic_block * jcp.typesize_in
                                            * ((inp_prf_idx / kw)
                                            * inp_prf_stride
                                            + (inp_prf_idx % kw));
                                } else {
                                    size_t ic_prf_stride =
                                        (size_t)jcp.typesize_in * iw * ih * id;
                                    size_t iw_prf_stride
                                            = jcp.typesize_in * jcp.simd_w;
                                    inp_prf_offset = ((inp_prf_idx / ic_block)
                                            * iw_prf_stride
                                            + (inp_prf_idx % ic_block)
                                            * ic_prf_stride);
                                }
                                mic_prefetcht0(EVEX_compress_addr_safe(
                                        aux_reg_inp_prf, inp_prf_offset,
                                        reg_long_offt));
                            }
                        }
                    }
                }
                step++;
            }
        }
        add(aux_reg_ker, jcp.typesize_in * kw * oc_block * ic_block);
        if (prf_ker)
            add(aux_reg_ker_prf, jcp.typesize_in * kw * oc_block * ic_block);
        add(aux_reg_inp, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul);
        if (prf_inp)
            add(aux_reg_inp_prf,
                    jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }


    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);
        add(aux_reg_inp_d_prf,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_ker_d_prf, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);

        pop(reg_out);
        pop(reg_out_prf);
    }
    if (max_input_offset > INT_MAX) pop(reg_inp_prf);
}

template<typename Vmm>
void _jit_avx512_common_conv_fwd_kernel<Vmm>::compute_loop_fma_core(int ur_w,
    int pad_l, int pad_r)
{
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label, kd_label;
    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * jcp.oc_block
        * jcp.ic_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;
    int shift_input_ptr = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw
        * inp_mul;


    auto input_offset = [=](int oi, int ic, int ki) {
        return (size_t)jcp.typesize_in
                * ((size_t)(ki * (jcp.dilate_w + 1) + oi * stride_w - pad_l)
                * inp_mul + (size_t)ic
                * (!jcp.is_1stconv ? 1 : (size_t)jcp.iw * jcp.ih * jcp.id));
    };

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_ow_start(ki, pad_l);
            int jj_end = get_ow_end(ur_w, ki, pad_r);
            for (int ic = 0; ic < ic_block; ic++) {
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        size_t aux_input_offset = input_offset(jj, ic, ki);
                        vbroadcastss(vmm_inp(jj, nb_oc_block),
                            EVEX_compress_addr_safe(aux_reg_inp,
                            aux_input_offset, reg_long_offt));
                    }
                }
                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int aux_kernel_offset = jcp.typesize_in
                        * (ii * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd * ic_block
                        * oc_block + ki * ic_block * oc_block + ic * oc_block);
                    if (jj_end - jj_start > 0)
                        vmovups(vmm_wei, EVEX_compress_addr(aux_reg_ker,
                            aux_kernel_offset));
                    for (int jj = jj_start; jj < jj_end; jj++)
                        if (jcp.kernel_kind == expl_bcast)
                            vfmadd231ps(vmm_out(jj, ii),
                                vmm_inp(jj, nb_oc_block), vmm_wei);
                        else {
                            size_t aux_input_offset = input_offset(jj, ic, ki);
                            vfmadd231ps(vmm_out(jj, ii), vmm_wei,
                                EVEX_compress_addr_safe(aux_reg_inp,
                                aux_input_offset, reg_long_offt, true));
                        }
                }
            }
        }
        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * jcp.oc_block
                * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);

        pop(reg_out);
    }
}

template<typename Vmm>
void _jit_avx512_common_conv_fwd_kernel<Vmm>::compute_loop(int ur_w,
        int pad_l, int pad_r)
{
    if (jcp.ndims == 5) push(reg_oi);

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1) < nstl::max(jcp.f_pad, jcp.back_pad)) {
            mov(reg_kj, ptr[param1 + GET_OFF(kd_padding)]);
            cmp(reg_kj, 0);
            je(skip_compute_loop, T_NEAR);
        }
    }
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1) < nstl::max(jcp.t_pad, jcp.b_pad)) {
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
        cmp(reg_kj, 0);
        je(skip_compute_loop, T_NEAR);
    }

    if (jcp.ver == ver_4fma)
        if(jcp.is_1stconv)
            compute_loop_4fma_1st(ur_w, pad_l, pad_r);
        else
            compute_loop_4fma(ur_w, pad_l, pad_r);
    else if (jcp.ver == ver_fma)
        if ((jcp.is_1stconv && jcp.kernel_kind != expl_bcast)
                || mayiuse(avx512_mic))
            compute_loop_fma(ur_w, pad_l, pad_r);
        else
            if (jcp.kernel_kind == embd_bcast && jcp.nb_oc_blocking == 1)
                compute_loop_fma(ur_w, pad_l, pad_r);
            else
                compute_loop_fma_core(ur_w, pad_l, pad_r);
    else
        assert(!"unknown convolution version");

    L(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) pop(reg_oi);
}

template<typename Vmm>
void _jit_avx512_common_conv_fwd_kernel<Vmm>::generate()
{
    int iw = jcp.iw;
    int ow = jcp.ow;
    int ow_block = jcp.ow_block;
    int nb_ow = jcp.nb_ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift = jcp.typesize_in * ur_w * stride_w * inp_mult;
    int inp_shift_pad_second_block = -1 * jcp.typesize_in * l_pad * inp_mult;
    int out_shift = jcp.typesize_out * ur_w * jcp.oc_block;

    preamble();
    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    int r_pad = nstl::max(
            0, (ow - 1) * stride_w + (kw - 1) * dilate_w - (iw + l_pad - 1));
    int n_oi = ow / ur_w;
    int r_pad1 = (ur_w * n_oi - 1) * stride_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1);

    if (!is_ow_threading_on(jcp)) {
        // ow is being processed as a whole - with left and right paddings
        if (r_pad1 > 0) n_oi--;

        if (ow == ur_w) {
            mov(reg_inp_prf, ptr[param1 + GET_OFF(src_prf)]);
            mov(reg_out_prf, ptr[param1 + GET_OFF(dst_prf)]);
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            mov(reg_inp_prf, reg_inp);
            mov(reg_out_prf, reg_out);
            if (n_oi == 0) {
                add(reg_inp_prf, inp_shift_pad);
                add(reg_out_prf, out_shift);
                compute_loop(ur_w, l_pad, r_pad1);
                add(reg_inp, inp_shift_pad);
                add(reg_out, out_shift);
                if (ur_w_tail != 0) {
                    add(reg_inp_prf, inp_shift);
                    add(reg_out_prf, out_shift);
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            } else {
                xor_(reg_oi, reg_oi);
                if (l_pad > 0) {
                    add(reg_inp_prf, inp_shift_pad);
                    add(reg_out_prf, out_shift);
                    compute_loop(ur_w, l_pad, 0);
                    add(reg_inp, inp_shift_pad);
                    add(reg_out, out_shift);
                    inc(reg_oi);
                }
                if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        add(reg_inp_prf, inp_shift);
                        add(reg_out_prf, out_shift);
                        compute_loop(ur_w, 0, 0);
                        add(reg_inp, inp_shift);
                        add(reg_out, out_shift);
                        inc(reg_oi);
                        cmp(reg_oi, n_oi);
                        jl(ow_loop_label, T_NEAR);
                    }
                }
                if (r_pad1 > 0) {
                    add(reg_inp_prf, inp_shift);
                    add(reg_out_prf, out_shift);
                    compute_loop(ur_w, 0, r_pad1);
                    add(reg_inp, inp_shift);
                    add(reg_out, out_shift);
                }
                if (ur_w_tail != 0) {
                    add(reg_inp_prf, inp_shift);
                    add(reg_out_prf, out_shift);
                    compute_loop(ur_w_tail, 0, r_pad);
                }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.

        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label;
        Label oi_loop_label, oi_loop_start_label, oi_loop_end_label;

        assert(ow_block % ur_w == 0);
        int n_oi_not_last_ow_block = ow_block / ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;

        int n_oi_last_ow_block = (ow - ow_block * (nb_ow-1)) / ur_w;

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded) n_oi_last_ow_block--;
        else if (first_ow_block_padded) n_oi_first_ow_block--;
        else if (next_last_ow_block_padded) n_oi_next_last_ow_block--;

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, 0); // is that the first ow-block ?
        jg(middle_ow_blocks_label, T_NEAR);

        // the first ow block, compute left padding

        mov(reg_oi, n_oi_first_ow_block);
        mov(reg_inp_prf, reg_inp);
        mov(reg_out_prf, reg_out);

        if (l_pad > 0) {
            mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
            add(reg_inp_prf, inp_shift_pad);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w, l_pad, 0);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);
            dec(reg_oi);
        }
        jmp(oi_loop_label, T_NEAR);

        // middle or last ow block entry

        L(middle_ow_blocks_label);

        if (l_pad > 0) {
            // just to consider left padding, not compute
            add(reg_inp, inp_shift_pad_second_block);
            add(reg_inp_prf, inp_shift_pad_second_block);
        }

        // set number of iteration for oi-loop
        cmp(reg_owb, jcp.nb_ow - 1); // last ow-block ?
        mov(reg_oi, n_oi_last_ow_block);
        je(oi_loop_label, T_NEAR);
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        mov(reg_oi, n_oi_next_last_ow_block);
        je(oi_loop_label, T_NEAR);
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label);
        mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
        L(oi_loop_start_label);
            cmp(reg_oi, 0);
            jle(oi_loop_end_label, T_NEAR);

            add(reg_inp_prf, inp_shift);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w, 0, 0);
            add(reg_inp, inp_shift);
            add(reg_out, out_shift);
            dec(reg_oi);
            jmp(oi_loop_start_label, T_NEAR);
        L(oi_loop_end_label);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);

        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        jl(end_label, T_NEAR);
        if (next_last_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        // that is last block
        if (!last_ow_block_padded) {
            jmp(tail_label, T_NEAR);
        }

        // last oi block with right padding
        L(last_oi_label);
        mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
        add(reg_inp_prf, inp_shift);
        add(reg_out_prf, out_shift);
        compute_loop(ur_w, 0, r_pad1);
        add(reg_inp, inp_shift);
        add(reg_out, out_shift);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        jl(end_label, T_NEAR);

        L(tail_label);
        mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
        if (ur_w_tail != 0) {
            add(reg_inp_prf, inp_shift);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w_tail, 0, r_pad);
        }
        L(end_label);
    }
    postamble();

    if (jcp.with_eltwise)
        eltwise_injector_->prepare_table();
}

bool jit_avx512_common_conv_fwd_kernel::post_ops_ok(
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

status_t jit_avx512_common_conv_fwd_kernel::init_conf(
            jit_conv_conf_t &jcp, const convolution_desc_t &cd,
            memory_desc_t &src_md, memory_desc_t &weights_md,
            memory_desc_t &dst_md, memory_desc_t &bias_md,
            const primitive_attr_t &attr, int nthreads)
{
    using namespace prop_kind;

    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims-2];
    jcp.ow = dst_d.dims()[ndims-1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims-2];
    jcp.kw = weights_d.dims()[with_groups + ndims-1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    jcp.is_1stconv = is_1stconv(jcp);

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && src_d.data_type() == data_type::f32;

    const int full_simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);
    jcp.simd_w = full_simd_w;
    bool ok_to_try_xmm = true
        && mayiuse(avx512_core)
        && src_d.data_type() == data_type::f32
        && !jcp.is_1stconv
        && !ok_to_pad_channels
        && (jcp.ic % jcp.simd_w != 0 || jcp.oc % jcp.simd_w != 0)
        && (jcp.ic % 8 != 0 || jcp.oc % 8 != 0);
    if (ok_to_try_xmm)
        jcp.simd_w = 4;

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;
    jcp.aligned_threads = 0;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0;
    if (!args_ok)
        return status::unimplemented;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
    }

    auto src_tag = jcp.is_1stconv
        ? pick(ndims - 3, ncw, nchw, ncdhw)
        : ((jcp.simd_w == 4)
            ? pick(ndims - 3, nCw4c, nChw4c, nCdhw4c)
            : pick(ndims - 3, nCw16c, nChw16c, nCdhw16c));
    auto dst_tag = (jcp.simd_w == 4)
        ? pick(ndims - 3, nCw4c, nChw4c, nCdhw4c)
        : pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_tag = with_groups
        ? ((jcp.simd_w == 4)
            ? pick(ndims - 3, gOIw4i4o, gOIhw4i4o, gOIdhw4i4o)
            : pick(ndims - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o))
        : ((jcp.simd_w == 4)
            ? pick(ndims - 3, OIw4i4o, OIhw4i4o, OIdhw4i4o)
            : pick(ndims - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o));

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, src_tag));
        jcp.src_tag = src_tag;
    } else {
        jcp.src_tag = src_d.matches_one_of_tag(src_tag);
    }
    if (jcp.src_tag != src_tag)
        return status::unimplemented;

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, dst_tag));
        jcp.dst_tag = dst_tag;
    } else {
        jcp.dst_tag = dst_d.matches_one_of_tag(dst_tag);
    }
    if (jcp.dst_tag != dst_tag)
        return status::unimplemented;

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    if (mayiuse(avx512_common) &&
            src_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && dst_d.data_type() == data_type::f32) {
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops))
           jcp.ver = ver_4fma;

        if (jcp.is_1stconv) {
            // TODO: fix & remove constraints below
            bool not_for_4fma
                    = IMPLICATION(everyone_is(0, jcp.l_pad, jcp.t_pad),
                            nstl::max(jcp.kw, jcp.kh) < 7);
            bool is_dilated
                    = !everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w);
            if (one_of(true, not_for_4fma, is_dilated))
                jcp.ver = ver_fma;
            if (jcp.ver == ver_4fma) {
                wei_tag = with_groups
                    ? ((jcp.simd_w == 4)
                        ? pick(ndims - 3, gOiw4o, gOihw4o, gOidhw4o)
                        : pick(ndims - 3, gOiw16o, gOihw16o, gOidhw16o))
                    : ((jcp.simd_w == 4)
                        ? pick(ndims - 3, Oiw4o, Oihw4o, Oidhw4o)
                        : pick(ndims - 3, Oiw16o, Oihw16o, Oidhw16o));
            } else {
                wei_tag = with_groups
                    ? ((jcp.simd_w == 4)
                        ? pick(ndims - 3, gOwi4o, gOhwi4o, gOdhwi4o)
                        : pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o))
                    : ((jcp.simd_w == 4)
                        ? pick(ndims - 3, Owi4o, Ohwi4o, Odhwi4o)
                        : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o));
            }
        }
    } else {
        return status::unimplemented;
    }

    if (weights_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    }
    if (jcp.wei_tag != wei_tag)
        return status::unimplemented;

    if (jcp.is_1stconv) {
        jcp.ur_w = nstl::min(jcp.ow, regs);
    } else {
        // avx512_core guard - just to avoid possible regression for other archs
        if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
            jcp.ur_w = nstl::min(jcp.ow, regs);
        } else {
            for (int ur_w = regs; ur_w > 0; --ur_w) {
                if (jcp.ow % ur_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
        if ((ndims == 5 && jcp.ur_w <= 8) || (jcp.ur_w <= 1)) {
            jcp.ur_w = nstl::min(jcp.ow, regs);
        }
    }
    // TODO (Tanya): currently applied to Segnet convolutions only.
    // Need to try for other topologies
    if (jcp.ow > 150 && jcp.ur_w < regs/2)
        jcp.ur_w = regs;

    int n_oi = (jcp.ow / jcp.ur_w);
    int r_pad = (jcp.ur_w * n_oi - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1);
    if (jcp.l_pad > 0 && r_pad > 0)
        n_oi--;

    bool large_code_size = jcp.ur_w != jcp.ow && jcp.l_pad > 0 && r_pad > 0
            && ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1));
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.ic_block * jcp.kw;
        int mult = 1;
        if (jcp.l_pad > 0) mult += 1;
        if (r_pad > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if (ur_w * mult * num_ops_per_reg * 9.0 < max_code_size) {
                jcp.ur_w = ur_w;
                break;
            }
        }
    }

    /* Grouped channel offset to support 'non-blocked data' format for
     * convolution sizes with '(input_channel / ngroups) < simd' */
    jcp.nonblk_group_off
            = (jcp.ngroups > 1 && one_of(jcp.src_tag, ncw, nchw, ncdhw)) ?
            jcp.ic :
            1;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    auto is_ow_threading_applicable = [=]() {
        return (true && !jcp.is_1stconv && one_of(jcp.ndims, 3, 4)
                && IMPLICATION(mayiuse(avx512_mic),
                           jcp.ver == ver_4fma
                                   && IMPLICATION(jcp.mb != 1,
                                              jcp.ih == 1 && jcp.kh == 1)));
    };

    if (jcp.ver == ver_4fma && !jcp.is_1stconv) {
        if ((jcp.kw <= 5 && jcp.kh <= 5 && jcp.kw == jcp.kh && jcp.ow <= 8
                    && jcp.oh <= 8 && jcp.ow == jcp.oh)
                || (jcp.stride_h != 1 && jcp.ur_w < jcp.ow)) {
            if (jcp.nb_oc % 2 == 0) {
                jcp.nb_oc_blocking = 2;
                jcp.ur_w = nstl::min(jcp.ow, regs / jcp.nb_oc_blocking);
            }
        } else {
            for (int i = jcp.nb_oc; i > 0; i--)
                if (i * jcp.ur_w <= regs && jcp.nb_oc % i == 0) {
                    jcp.nb_oc_blocking = i;
                    break;
                }
        }
        if (jcp.ver == ver_4fma && is_ow_threading_applicable()) {
            if (jcp.nb_oc % 2 == 0 && jcp.ur_w < jcp.ow
                    && jcp.ow != 2 * jcp.ur_w) {
                jcp.nb_oc_blocking = 2;
                jcp.ur_w = nstl::min(jcp.ow, regs / jcp.nb_oc_blocking);
            }
        }
    }

    jcp.ow_block = jcp.ow;

    auto get_thr_eff = [=](int nb_oc_blocking, int ow_block) {
        int nb_ow = div_up(jcp.ow, ow_block);
        int nb_oc_chunks = div_up(jcp.nb_oc, nb_oc_blocking);
        int work_amount = jcp.mb * jcp.oh * nb_oc_chunks * nb_ow;
        float disbalance = (float)jcp.ow / rnd_up(jcp.ow, ow_block);
        float thr_eff = disbalance * (float)work_amount
            / rnd_up(work_amount, nthreads);
        return thr_eff;
    };

    auto get_ow_block = [=](int nb_oc_blocking, int ur_w, float &eff) {
        int res_ow_block = jcp.ow;
        eff = get_thr_eff(nb_oc_blocking, res_ow_block);
        if (!is_ow_threading_applicable())
            return res_ow_block;

        int L2_part = (get_cache_size(2) * 7 / 8) / typesize;
        if (jcp.ver == ver_4fma)
            L2_part /= 2;
        int size_src_chunk = jcp.ic_block * ur_w * jcp.kh;
        int size_dst_chunk = jcp.oc_block * nb_oc_blocking * ur_w;
        int size_wei_chunk = jcp.oc_block * nb_oc_blocking * jcp.ic_block
            * jcp.kw * jcp.kh;
        int nurw_cache = (L2_part - 2 * size_wei_chunk)
            / (2 * size_dst_chunk + 2 * size_src_chunk);
        // current design of generate() requires ow_block >= 2 * ur_w
        int ow_block_cache = ur_w * nstl::max(2, nurw_cache);

        int ow_block_thr = ow_block_cache;
        eff = get_thr_eff(nb_oc_blocking, ow_block_thr);

        int max_nb_ow = div_up(jcp.ow, 2 * ur_w);
        int start_nb_ow = div_up(jcp.ow, ow_block_thr);
        for (int nb_ow = start_nb_ow; nb_ow <= max_nb_ow; nb_ow++) {
            int ow_block
                = nstl::min(rnd_up(div_up(jcp.ow, nb_ow), ur_w), jcp.ow);
            float eff_threshold = (jcp.ver == ver_4fma) ? 0.8f : 0.9f;
            if (ow_block < nb_oc_blocking * jcp.oc_block && eff > eff_threshold)
                break;
            if (div_up(jcp.ow, ow_block) != nb_ow)
                continue;
            float thr_eff = get_thr_eff(nb_oc_blocking, ow_block);
            float eff_step = (jcp.ver == ver_4fma) ? 1.1f : 1.f;
            if (ow_block >= 2 * ur_w && thr_eff > eff_step * eff) {
                ow_block_thr = ow_block;
                eff = thr_eff;
            }
            eff_threshold = (jcp.ver == ver_4fma) ? 0.9f : 0.98f;
            if (eff > eff_threshold)
                break;
        }
        res_ow_block = nstl::min(jcp.ow, nstl::max(2 * ur_w, ow_block_thr));
        eff = get_thr_eff(nb_oc_blocking, res_ow_block);
        return res_ow_block;
    };


    if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
        int try_nb_oc_blocking = 2;
        unsigned int ker_inp_size = typesize * div_up(jcp.iw, jcp.stride_w)
            * jcp.ic_block * jcp.kh * jcp.kd;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block
            * try_nb_oc_blocking;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
            * jcp.oc_block * try_nb_oc_blocking * jcp.kd;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
            + ker_wei_size;

        bool embd_bcast_condition = true
            && (jcp.kw == 3 && jcp.ow <= 28 && ker_total_size < L1_cache_size)
            && !(jcp.kw == 3 && jcp.ow == 13 && jcp.ic >= 192)
            && !(jcp.kw == 3 && jcp.ow == 28 && jcp.ic >= 512);

        if (jcp.mb == 1) {
            unsigned int inp_size = jcp.mb * div_up(jcp.ih, jcp.stride_h)
                    * div_up(jcp.iw, jcp.stride_w) * jcp.ic;
            unsigned int wei_size = jcp.ic * jcp.oc * jcp.kh * jcp.kw;

            // Estimate whether we need to limit the number of threads
            // and calculate this number. Includes some heuristic.
            int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
            int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
            int job_size_min = work_amount / nthreads;
            int job_size_max = div_up(work_amount, nthreads);
            int ch_max = rnd_up(jcp.oh, job_size_max);
            int ch_min = (job_size_min == 0)
                ? jcp.oh
                : rnd_up(jcp.oh, job_size_min);
            bool not_aligned_max = ch_max % jcp.oh != 0 && ch_max / jcp.oh < 2
                    && (jcp.oh != 8 || ch_max / jcp.oh > 1);
            bool not_aligned_min = ch_min % jcp.oh != 0 && ch_min / jcp.oh < 2
                    && (jcp.oh != 8 || ch_min / jcp.oh > 1);
            bool eligible_case = (jcp.stride_h == 1 && jcp.stride_w == 1)
                    || nthreads > oc_chunks;
            if (jcp.loop_order == loop_cgn && oc_chunks > 1 && nthreads > 1
                && wei_size / inp_size > 24
                && (not_aligned_max || not_aligned_min)
                && eligible_case) {
                // Try to find nthreads > mkldnn_get_max_threads() / 2 such
                // that oc_chunks is a multiple of nthreads, or nthreads is a
                // multiple of oc_chunks. Otherwise, keep default value.
                // TODO: implement a task-based alternative without throttling.
                jcp.aligned_threads = nthreads;
                for (int i = nthreads; i > nthreads / 2; i--) {
                    if (oc_chunks % i == 0 || i % oc_chunks == 0) {
                        jcp.aligned_threads = i;
                        break;
                    }
                }
            }
        }

        if (jcp.kw > 3
                || (jcp.stride_w == 1 && jcp.stride_h == 1
                           && embd_bcast_condition)
                || ((jcp.stride_w != 1 || jcp.stride_h != 1)
                           && ((jcp.mb <= 16 && (jcp.oc <= 192 || jcp.oh <= 10)
                                      && embd_bcast_condition)))
                || (jcp.mb == 1
                           && (jcp.ur_w >= jcp.ow || jcp.is_1stconv
                                      || (jcp.ow <= 147 && jcp.oc <= 96)))) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = nstl::min(jcp.ow, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (ker_total_size < L1_cache_size && jcp.ow <= 8 && jcp.kh <= 3
                    && jcp.kw <= 3 && jcp.nb_oc % try_nb_oc_blocking == 0
                    && IMPLICATION(jcp.is_1stconv, jcp.mb == 1)
                    && IMPLICATION(jcp.mb == 1, jcp.ur_w < jcp.ow)) {
                jcp.nb_oc_blocking = try_nb_oc_blocking;
                jcp.ur_w = nstl::min(jcp.ow, 31 / (jcp.nb_oc_blocking + 1));
            }
        } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_ic_blocking = 1;
            if (IMPLICATION(jcp.is_1stconv, jcp.mb > 1)) {
                float best_thr_eff = 0.f;
                int best_nb_oc_blocking = 1;
                for (int i = nstl::min(jcp.nb_oc, 5); i > 0; i--) {
                    if (jcp.nb_oc % i == 0) {
                        float thr_eff;
                        int ur_w = nstl::min(jcp.ow, 31 / (i + 1));
                        get_ow_block(i, ur_w, thr_eff);
                        if (thr_eff > 1.05f * best_thr_eff) {
                            best_nb_oc_blocking = i;
                            best_thr_eff = thr_eff;
                        }
                    }
                }
                jcp.nb_oc_blocking = best_nb_oc_blocking;
                jcp.ur_w = nstl::min(jcp.ow, 31 / (jcp.nb_oc_blocking + 1));
            }
        }
    }

    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    args_ok = true
        && jcp.l_pad <= jcp.ur_w
        && jcp.ic <= src_d.padded_dims()[1]
        && jcp.oc <= dst_d.padded_dims()[1]
        && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
        && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok)
        return status::unimplemented;

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    if (r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    float thr_eff;
    jcp.ow_block = get_ow_block(jcp.nb_oc_blocking, jcp.ur_w, thr_eff);
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    const int L2_size = get_cache_size(2, true) / sizeof(float);
    // Source and output data needs to fit in L2,
    // leaving some space for weights and prefetching.
    int h_L2 = int(((0.6f * L2_size) / jcp.simd_w
                           - nstl::min(0, jcp.kh - jcp.stride_h) * jcp.iw)
            / (jcp.stride_h * jcp.iw + jcp.ow));
    jcp.h_blocking = nstl::max(1, nstl::min(jcp.oh, h_L2));

    if (jcp.ver == ver_4fma) {
        if (!is_ow_threading_on(jcp)) {
            for (int divf = 2, temp_nb = jcp.nb_ic_L2; divf <= jcp.nb_ic;
                  divf++) {
                size_t l2_src
                    = (size_t)jcp.iw * jcp.ic_block * jcp.ih * temp_nb * jcp.id;
                size_t l2_dst = (size_t)jcp.ow * jcp.oc_block * jcp.nb_oc_blocking
                    * jcp.oh * jcp.od;
                size_t l2_filt = (size_t)jcp.kw * jcp.oc_block * jcp.ic_block
                    * jcp.kh * jcp.nb_oc_blocking * temp_nb * jcp.kd;
                if (4 * (l2_src + l2_dst + l2_filt) > KNx_L2_EFFECTIVE_CAPACITY) {
                    if (jcp.kh == 3 && jcp.oh == 7) {
                        jcp.nb_ic_L2 = 1;
                        break;
                    }
                    temp_nb = (jcp.nb_ic_L2 % divf == 0 ? jcp.nb_ic_L2 / divf
                                    : jcp.nb_ic_L2);
                } else {
                    jcp.nb_ic_L2 = temp_nb;
                    break;
                }
            }
        } else if (jcp.ic > 64) {
            jcp.nb_ic_L2 = 2; /* according to performance data*/
        }
    }

    return status::success;
}

void jit_avx512_common_conv_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, jcp.typesize_out * jcp.oc);
}

void jit_avx512_common_conv_bwd_data_kernel_f32::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
            size_t aux_src_offset
                = (size_t)typesize * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j)
                * jcp.ic_block;
            mic_prefetcht1(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                        reg_long_offt));
        }
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::store_output(int ur_w)
{
    Label no_update_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    je(no_update_label, T_NEAR);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            size_t aux_src_offset = (size_t)typesize
                * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;
            vaddps(zmm, EVEX_compress_addr_safe(reg_src, aux_src_offset,
                        reg_long_offt));
        }
    }

    L(no_update_label);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            size_t aux_src_offset = (size_t)typesize
                * ((size_t)k * jcp.ih * jcp.iw * jcp.id + j) * jcp.ic_block;
            vmovups(EVEX_compress_addr_safe(reg_src, aux_src_offset,
                        reg_long_offt), zmm);
            mic_prefetcht0(EVEX_compress_addr_safe(reg_src_prf, aux_src_offset,
                        reg_long_offt));
        }
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_4fma(
        int ur_w, int l_overflow, int r_overflow)
{
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    Label kh_label, last_iter_label, loop_end_label, kd_label;
    int ker_load_number = 4;
    int shift_ker_ptr = typesize * kw * oc_block * ic_block;
    int shift_dst_ptr = typesize * ow * oc_block;
    int ii_dpref_t0 = get_iw_start(0, l_overflow);
    int iw_end_ipref = get_iw_end(ur_w, 0, r_overflow);

    bool check_last_kh = (jcp.kh > 3);
    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw * jcp.kd + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };
    auto kernel_loads = [=](int ki, int oc, int kk) {
        for (int ii = 0; ii < ker_load_number; ii++) {
            int aux_kernel_offset = kernel_offset(kk, oc + ii, ki);
            vmovups(zmm_ker(ii),
                EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
        }
    };
    auto prefetch_dst_next_kh = [&](int ki, int ki_start, int cnt0, int cnt1) {
        if (cnt1 >= ker_load_number && cnt0 >= ker_load_number
            && ki >= ki_start && ii_dpref_t0 < iw_end_ipref) {
            int aux_dst_offset = typesize * ((ii_dpref_t0
                + jcp.l_pad) * oc_block + jcp.ow * oc_block);
            prefetcht0(EVEX_compress_addr(aux_reg_dst, aux_dst_offset));
            ii_dpref_t0++;
        }
    };

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
        mov(aux_reg_dst_prf, reg_dst_prf);
        mov(aux_reg_ker_prf, reg_ker_prf);
    }

    if (jcp.ndims == 5) {
        push(reg_src_prf);
        push(reg_src);

        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        mov(aux_reg_ker_d, ptr[param + GET_OFF(filt)]);
        mov(aux_reg_dst_d_prf, reg_dst_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        L(kd_label);
        mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_dst_prf, aux_reg_dst_d_prf);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
    }

    align(16);
    L(kh_label);
    if (check_last_kh) {
        for (int ki = 0; ki < kw; ki++)
        for (int oc = 0; oc < oc_block; oc += 4)
        for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
            bool last_kernel_loads = (kk == jcp.nb_ic_blocking - 1
                && ki == kw - 1 && (oc + 4) == oc_block);

            if (last_kernel_loads) {
                cmp(reg_kj, 1);
                je(last_iter_label, T_NEAR);
            }

            kernel_loads(ki, oc, kk);
            for (int ii = get_iw_start(ki, l_overflow),
                    prf_count_t0 = 0, prf_count_t1 = 0;
                    ii < get_iw_end(ur_w, ki, r_overflow); ii++) {
                int aux_dst_offset = typesize
                    * ((ii + jcp.l_pad - ki) * oc_block + oc);
                v4fmaddps(zmm_out(ii, kk), zmm_ker(0),
                    EVEX_compress_addr(aux_reg_dst, aux_dst_offset));

                if (ii % 2) {
                    if (prf_count_t0 < 4) {
                        int aux_kernel_prf;
                        if (last_kernel_loads)
                            aux_kernel_prf= kernel_offset(0, prf_count_t0
                                + oc + 4 - oc_block, 0) + typesize * kw
                                * oc_block * ic_block;
                        else
                            aux_kernel_prf = kernel_offset(kk, oc + 4
                                + prf_count_t0, ki);
                        mic_prefetcht0(EVEX_compress_addr(aux_reg_ker,
                            aux_kernel_prf));
                        prf_count_t0++;
                    } else if (prf_count_t1 < 4) {
                        mic_prefetcht1(EVEX_compress_addr(aux_reg_ker_prf,
                            kernel_offset(kk, oc + prf_count_t1, ki)));
                        prf_count_t1++;
                    }
                } else
                    prefetch_dst_next_kh(ki, 2, prf_count_t0, prf_count_t1);
            }
            if (last_kernel_loads) {
                jmp(loop_end_label, T_NEAR);

                L(last_iter_label);

                kernel_loads(ki, oc, kk);
                for (int ii = get_iw_start(ki, l_overflow),
                        prf_count_t0 = 0, prf_count_t1 = 0;
                        ii < get_iw_end(ur_w, ki, r_overflow); ii++) {
                    int aux_dst_offset = typesize
                        * ((ii + jcp.l_pad - ki) * oc_block + oc);
                    v4fmaddps(zmm_out(ii, kk), zmm_ker(0),
                            EVEX_compress_addr(aux_reg_dst, aux_dst_offset));
                    if (ii % 2) {
                        if (prf_count_t0 < 4) {
                            mic_prefetcht0(EVEX_compress_addr(aux_reg_ker_prf,
                                kernel_offset(0, prf_count_t0, 0)));
                            prf_count_t0++;
                        } else if (prf_count_t1 < 4) {
                            mic_prefetcht1(EVEX_compress_addr(aux_reg_ker_prf,
                                kernel_offset(kk, oc + prf_count_t1, ki)));
                            prf_count_t1++;
                        }
                    }
                }
                L(loop_end_label);
            }
        }
    } else {
        for (int ki = 0; ki < kw; ki++)
        for (int oc = 0; oc < oc_block; oc += 4)
        for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
            kernel_loads(ki, oc, kk);

            for (int ii = get_iw_start(ki, l_overflow), prf_count_t1 = 0;
                    ii < get_iw_end(ur_w, ki, r_overflow); ii++) {
                int aux_dst_offset = typesize
                    * ((ii + jcp.l_pad - ki) * oc_block + oc);
                v4fmaddps(zmm_out(ii, kk), zmm_ker(0),
                    EVEX_compress_addr(aux_reg_dst, aux_dst_offset));
                if ((ii % 2) && (prf_count_t1 < 4)) {
                    mic_prefetcht1(EVEX_compress_addr(
                        aux_reg_ker_prf, kernel_offset(kk,
                        oc + prf_count_t1, ki)));
                    prf_count_t1++;
                }
                if ( ki == 1 && oc == 0 && kk == 0)
                    mic_prefetcht1(EVEX_compress_addr(
                        aux_reg_dst_prf, aux_dst_offset));
            }
        }
    }

    add(aux_reg_ker, shift_ker_ptr);
    sub(aux_reg_dst, shift_dst_ptr);
    add(aux_reg_ker_prf, shift_ker_ptr);
    sub(aux_reg_dst_prf, shift_dst_ptr);

    dec(reg_kj);
    cmp(reg_kj, 0);
    jg(kh_label, T_NEAR);

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d, typesize * (jcp.oh * ow) * ic_block);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * oc_block * ic_block);
        sub(aux_reg_dst_d_prf, typesize * (jcp.oh * ow) * ic_block);
        add(aux_reg_ker_d_prf, typesize * jcp.kw * jcp.kh *oc_block * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);

        pop(reg_src);
        pop(reg_src_prf);
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma(
        int ur_w, int l_overflow, int r_overflow)
{
    Label kh_label, kd_label;
    int kw = jcp.kw;
    int ow = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;
    int num_inp_prfs = ur_w * nstl::min(kw, stride_w)
                       + nstl::max(0, kw - stride_w);
    int num_prfs = num_ker_loads + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w / stride_w;
    int prf_inst_spacing = nstl::max(1, num_fmas / num_prfs);
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);

        mov(aux_reg_dst_prf, reg_dst_prf);
        mov(aux_reg_ker_prf, reg_ker_prf);
    }

    if (jcp.ndims == 5) {
        push(reg_src_prf);
        push(reg_src);

        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        mov(aux_reg_ker_d, ptr[param + GET_OFF(filt)]);
        mov(aux_reg_dst_d_prf, reg_dst_prf);
        mov(aux_reg_ker_d_prf, reg_ker_prf);

        L(kd_label);
        mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
        mov(aux_reg_dst_prf, aux_reg_dst_d_prf);
        mov(aux_reg_ker_prf, aux_reg_ker_d_prf);
    }

    L(kh_label); {
        int step = 0;
        int ker_prfs = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int oc = 0; oc < oc_block; oc++) {
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        int aux_kernel_offset = typesize * ((oc + i) * oc_block
                                + ki * ic_block * oc_block);
                        vmovups(zmm_ker(i), EVEX_compress_addr(
                                    aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                        = (step + load_offset) % ker_pipeline_depth;
                    int aux_kernel_offset = typesize * ((oc + load_offset)
                            * oc_block + ki * ic_block * oc_block);
                    vmovups(zmm_ker(ker_load_reg_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }

                bool ker_prf_inserted = false;
                auto zmm_kernel = zmm_ker(step % ker_pipeline_depth);

                int jj_start = get_iw_start(ki, l_overflow);
                int jj_end = get_iw_end(ur_w, ki, r_overflow);
                assert(stride_w != 1
                        || jj_start == nstl::max(0,
                            l_overflow - (kw - 1 - ki) * dilate_w));
                assert(stride_w != 1
                        || jj_end == ur_w - nstl::max(0,
                            r_overflow - ki * dilate_w));

                for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                    assert((jj + l_pad - ki * dilate_w) % stride_w == 0);
                    int aux_dst_offset = typesize *
                        (((jj + l_pad - ki * dilate_w)
                                / stride_w) * jcp.oc_block + oc);
                    vfmadd231ps(zmm_out(jj, 0), zmm_kernel,
                        EVEX_compress_addr(aux_reg_dst, aux_dst_offset, true));

                    int fma_idx = (step * ur_w + jj) / stride_w;
                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (!ker_prf_inserted && ker_prfs < num_ker_loads) {
                            int ker_prf_offset = typesize
                                * ker_prfs * jcp.oc_block;
                            mic_prefetcht1(EVEX_compress_addr(
                                        aux_reg_ker_prf, ker_prf_offset));
                            ker_prf_inserted = true;
                            ker_prfs++;
                        } else {
                            int inp_prf_idx = prf_slot_idx - ker_prfs;
                            if (inp_prf_idx < num_inp_prfs) {
                                int inp_prf_offset
                                    = ic_block * typesize
                                    * ((inp_prf_idx / kw) * kw
                                            + (inp_prf_idx % kw));
                                mic_prefetcht0(EVEX_compress_addr(
                                            aux_reg_dst_prf, inp_prf_offset));
                            }
                        }
                    }
                }
                step++;
            }
        }

        add(aux_reg_ker, typesize * stride_h * kw * oc_block * ic_block);
        sub(aux_reg_dst, typesize * (jcp.dilate_h + 1) * ow * oc_block);
        add(aux_reg_ker_prf, typesize * stride_h * kw * oc_block * ic_block);
        sub(aux_reg_dst_prf, typesize * (jcp.dilate_h + 1) * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add(aux_reg_ker_d, typesize * jcp.stride_d * jcp.kw * jcp.kh
                * oc_block * ic_block);
        sub(aux_reg_dst_d_prf,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add(aux_reg_ker_d_prf, typesize * jcp.stride_d * jcp.kw * jcp.kh
                * oc_block * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
    }

    if (jcp.ndims == 5)
    {
        pop(reg_src);
        pop(reg_src_prf);
    }
}

void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop_fma_core(
        int ur_w, int l_overflow, int r_overflow)
{
    int kw = jcp.kw;
    int ow = jcp.ow;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;
    Label kh_label, kd_label;

    int shift_ker_ptr = typesize * kw * oc_block * ic_block;
    int shift_dst_ptr = typesize * (jcp.dilate_h + 1) * ow * oc_block;

    auto output_offset = [=](int oi, int oc, int ki) {
        return typesize *
            (((oi + jcp.l_pad - ki * dilate_w) / stride_w) * oc_block + oc);
    };
    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw * jcp.kd + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        push(reg_src_prf);
        push(reg_src);

        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        mov(aux_reg_ker_d, ptr[param + GET_OFF(filt)]);

        L(kd_label);
        mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            for (int oc = 0; oc < oc_block; oc++) {
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        int aux_output_offset = output_offset(jj, oc, ki);
                        vbroadcastss(zmm_inp(jj, nb_ic_block),
                            ptr[aux_reg_dst + aux_output_offset]);
                    }
                }
                for (int ii = 0; ii < nb_ic_block; ii++) {
                    int aux_kernel_offset = kernel_offset(ii, oc, ki);
                    if (jj_end - jj_start > 0)
                        vmovups(zmm_wei, EVEX_compress_addr(aux_reg_ker,
                            aux_kernel_offset));
                    for (int jj = jj_start; jj < jj_end; jj += stride_w)
                        if (jcp.kernel_kind == expl_bcast)
                            vfmadd231ps(zmm_out(jj, ii),
                                zmm_inp(jj, nb_ic_block), zmm_wei);
                        else
                            vfmadd231ps(zmm_out(jj, ii), zmm_wei,
                                EVEX_compress_addr(aux_reg_dst,
                                output_offset(jj, oc, ki), true));
                }
            }
        }
        add(aux_reg_ker, shift_ker_ptr);
        sub(aux_reg_dst, shift_dst_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * ic_block);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * oc_block * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);

        pop(reg_src);
        pop(reg_src_prf);
    }
}

inline void jit_avx512_common_conv_bwd_data_kernel_f32::compute_loop(
        int ur_w, int l_overflow, int r_overflow)
{
    if (jcp.ndims == 5) push(reg_oi);

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        mov(reg_kj, ptr[param + GET_OFF(kd_padding)]);
        cmp(reg_kj, 0);
        je(skip_compute_loop, T_NEAR);
    }
    mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    cmp(reg_kj, 0);
    je(skip_compute_loop, T_NEAR);

    if (jcp.ver == ver_4fma)
        compute_loop_4fma(ur_w, l_overflow, r_overflow);
    else if (jcp.ver == ver_fma)
        if (mayiuse(avx512_mic))
            compute_loop_fma(ur_w, l_overflow, r_overflow);
        else
          if (jcp.kernel_kind == embd_bcast && jcp.nb_ic_blocking == 1)
              compute_loop_fma(ur_w, l_overflow, r_overflow);
          else
              compute_loop_fma_core(ur_w, l_overflow, r_overflow);
    else
        assert("!unknown convolution version");

    L(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) pop(reg_oi);
}

void jit_avx512_common_conv_bwd_data_kernel_f32::generate()
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int dst_shift = jcp.typesize_in * (ur_w / stride_w) * ic_block;
    int src_shift = jcp.typesize_out * ur_w * oc_block;

    preamble();

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);
    mov(reg_src_prf, ptr[param + GET_OFF(src_prf)]);
    mov(reg_dst_prf, ptr[param + GET_OFF(dst_prf)]);
    mov(reg_ker_prf, ptr[param + GET_OFF(filt_prf)]);

    int l_overflow = nstl::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow1 = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad) - ur_w_tail) / stride_w);

    int n_oi = iw / ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (ur_w == iw) {
        compute_loop(ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(ur_w, l_overflow, r_overflow1);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
        add(reg_src_prf, src_shift);
        add(reg_dst_prf, dst_shift);
        if (ur_w_tail != 0)
            compute_loop(ur_w_tail, 0, r_overflow);
    } else {
        xor_(reg_oi, reg_oi);
        if (l_overflow > 0) {
            compute_loop(ur_w, l_overflow, 0);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            add(reg_src_prf, src_shift);
            add(reg_dst_prf, dst_shift);

            inc(reg_oi);
        }
        if ((l_overflow <= 0 && n_oi > 0)
            || (l_overflow > 0 && n_oi > 1)) {
            Label ow_loop_label;
            L(ow_loop_label); {
                compute_loop(ur_w, 0, 0);
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
                add(reg_src_prf, src_shift);
                add(reg_dst_prf, dst_shift);

                inc(reg_oi);
                cmp(reg_oi, n_oi);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_overflow1 > 0) {
            compute_loop(ur_w, 0, r_overflow1);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            add(reg_src_prf, src_shift);
            add(reg_dst_prf, dst_shift);
        }
        if (ur_w_tail != 0) {
            compute_loop(ur_w_tail, 0, r_overflow);
        }
    }

    postamble();
}

status_t jit_avx512_common_conv_bwd_data_kernel_f32::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    if (!mayiuse(avx512_common)) return status::unimplemented;

    jcp = zero<decltype(jcp)>();

    jcp.simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);
    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

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
    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    jcp.r_pad = (jcp.ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.iw + jcp.l_pad - 1);
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    jcp.aligned_threads = 0;

    jcp.is_1stconv = false;

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && diff_src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    auto dat_tag = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_tag = with_groups
        ? pick(ndims - 3, gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
        : pick(ndims - 3, OIw16o16i, OIhw16o16i, OIdhw16o16i);
    jcp.src_tag = diff_src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = diff_dst_d.matches_one_of_tag(dat_tag);

    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0
        && jcp.src_tag == dat_tag
        && jcp.dst_tag == dat_tag;
    if (!args_ok)
        return status::unimplemented;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_w = jcp.stride_w;

    int regs = 28;
    if (jcp.iw <= regs)
        jcp.ur_w = jcp.iw;
    else {
        for (int ur_w = regs; ur_w > 0; --ur_w)
            if (ur_w % jcp.stride_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
    }
    int l_overflow = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad) - jcp.iw % jcp.ur_w) / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (mayiuse(avx512_common)
         && diff_dst_d.data_type() == data_type::f32
         && weights_d.data_type() == data_type::f32
         && diff_src_d.data_type() == data_type::f32) {
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops)
            && jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1) {
                jcp.ver = ver_4fma;
            }
    } else {
        return status::unimplemented;
    }

    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    if (jcp.wei_tag != wei_tag)
        return status::unimplemented;

    if (!utils::everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && jcp.ver != ver_fma)
        return status::unimplemented;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    if (jcp.ver == ver_4fma) {
        if (jcp.kw == 3 && jcp.kh == 3 && jcp.iw == 7 && jcp.ih == 7) {
            jcp.nb_ic_blocking = 2;
        } else {
            for (int i = jcp.nb_ic; i > 0; i--)
                if (i * jcp.ur_w <= regs && jcp.nb_ic % i == 0) {
                    jcp.nb_ic_blocking = i;
                    break;
                }
        }
    }

    jcp.loop_order = loop_gnc;

    bool large_code_size = (jcp.ur_w != jcp.ow)
         && ((l_overflow <= 0 && n_oi > 0) ||(l_overflow > 0 && n_oi > 1))
         && (r_overflow1 > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow1 > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if ((ur_w / jcp.stride_w) * mult * num_ops_per_reg * 9.2
                    < max_code_size) {
                if (ur_w % jcp.stride_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
    }

    if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
        int try_nb_ic_blocking = 2;
        unsigned int ker_inp_size = typesize * jcp.iw * jcp.ic_block
            * try_nb_ic_blocking * jcp.kh;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
            * jcp.oc_block * try_nb_ic_blocking;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
            + ker_wei_size;
        if (!(jcp.kw == 1 || (jcp.kw == 5 && jcp.iw < 8)
            || (jcp.kw < 5 && ((jcp.iw <= 5 || (jcp.iw > 8 && jcp.iw <= 13))
            || ker_total_size > L1_cache_size )))
                || jcp.stride_h > 1 || jcp.stride_d > 1) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = nstl::min(jcp.iw, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (!(jcp.kw > 3 || (jcp.kw == 3 && ker_total_size < L1_cache_size
                && jcp.ow > 8)) && jcp.stride_h == 1)
                if (jcp.nb_ic % try_nb_ic_blocking == 0) {
                    jcp.nb_ic_blocking = try_nb_ic_blocking;
                    jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
                    if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
                }
         } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_oc_blocking = 1;
            jcp.nb_ic_blocking = 4;
            if (jcp.nb_ic < jcp.nb_ic_blocking) jcp.nb_ic_blocking = jcp.nb_ic;
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--)
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
            if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (l_overflow * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad) - jcp.ur_w_tail) / jcp.stride_w);
    if (r_overflow_no_tail * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    if ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
        return status::unimplemented;

    pick_loop_order(jcp);

    jcp.nb_oc_L2 = jcp.nb_oc;
    if (jcp.ver == ver_4fma && (jcp.kh < 5 && jcp.kw < 5)) {
        for (int divf = 2, temp_nb = jcp.nb_oc_L2; divf <= jcp.nb_oc;
              divf++) {
            size_t l2_src = jcp.iw * jcp.ic_block * jcp.nb_ic_blocking * jcp.ih
                * jcp.id;
            size_t l2_dst = jcp.ow * jcp.oc_block * temp_nb * jcp.oh * jcp.od;
            size_t l2_filt = jcp.kw * jcp.oc_block * jcp.ic_block * jcp.kh
                * jcp.kd * jcp.nb_ic_blocking * temp_nb;
            if (4 * (l2_src + l2_dst + l2_filt) > KNx_L2_EFFECTIVE_CAPACITY) {
                if (jcp.kh == 3 && jcp.ih == 7) {
                    jcp.nb_oc_L2 = 1;
                    break;
                }
                temp_nb = (jcp.nb_oc_L2 % divf == 0 ? jcp.nb_oc_L2 / divf
                                : jcp.nb_oc_L2);
            } else {
                jcp.nb_oc_L2 = temp_nb;
                break;
            }
        }
    }

    args_ok = true
        && jcp.ic <= diff_src_d.padded_dims()[1]
        && jcp.oc <= diff_dst_d.padded_dims()[1]
        && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
        && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    return status::success;
}

void jit_avx512_common_conv_bwd_data_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

const int jit_avx512_common_conv_bwd_weights_kernel_f32::max_ur_w = 28;

void jit_avx512_common_conv_bwd_weights_kernel_f32::od_step_comeback_pointers()
{
    Label kd_comeback_label;

    /* 'depth' loop count bound by 'kd_work_size' */
    mov(kj, reg_kd_count);
    L(kd_comeback_label); {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = jcp.ver == ver_4fma ? jcp.tr_iw : jcp.iw;
        sub(reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mult);
        sub(reg_kernel,
            jcp.typesize_out * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kd_comeback_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::oh_step_comeback_pointers()
{
    Label kh_comeback_label, kd_comeback_label;
    mov(kj, reg_kh);
    L(kh_comeback_label); {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = jcp.ver == ver_4fma ? jcp.tr_iw : jcp.iw;
        sub(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mult);
        sub(reg_kernel,
            jcp.typesize_out * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_ic_block_step_fma(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{

    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vmovups(Zmm(i_kw * ic_block_step + i_ic),
                EVEX_compress_addr(reg_kernel, typesize * (i_kw * ic_block
                + i_ic) * jcp.oc_block + kernel_offset));

    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        if (i_ur == 0) {
            vmovups(Zmm(kw * ic_block_step + (i_ur + 0) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 0)
                * oc_block + output_offset));
            if (ur_w > 1) vmovups(Zmm(kw * ic_block_step + (i_ur + 1) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 1) * oc_block
                + output_offset));
            if (ur_w > 2) vmovups(Zmm(kw * ic_block_step + (i_ur + 2) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 2) * oc_block
                + output_offset));
            if (ur_w > 3) vmovups(Zmm(kw * ic_block_step + (i_ur + 3) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 3) * oc_block
                + output_offset));
        } else if (i_ur + 3 < ur_w)
            vmovups(Zmm(kw * ic_block_step + (i_ur + 3) % 4),
                EVEX_compress_addr(reg_output, typesize * (i_ur + 3) * oc_block
                + output_offset));

        for (int i_kw = 0; i_kw < kw; i_kw++) {
            int i_iw = i_ur * jcp.stride_w + i_kw * (jcp.dilate_w + 1);
            if (i_iw - pad_l < 0 || i_iw > (ur_w - 1) * jcp.stride_w +
                    (kw - 1) * (jcp.dilate_w + 1) - pad_r) continue;
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                const size_t i_offset = (size_t)input_offset
                    + (size_t)typesize * (jcp.ver == ver_4fma
                            ? (i_iw - pad_l + i_ic * jcp.tr_iw)
                            : (jcp.is_1stconv
                                ? (i_iw - pad_l) + (size_t)i_ic
                                    * ((size_t)jcp.ih*jcp.iw*jcp.id)
                                : (i_iw - pad_l) * ic_block + i_ic));
                vfmadd231ps(Zmm(i_kw * ic_block_step + i_ic),
                    Zmm(kw * ic_block_step + i_ur % 4),
                    EVEX_compress_addr_safe(reg_input, i_offset, reg_long_offt,
                        true));
            }
        }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vmovups(EVEX_compress_addr(reg_kernel, typesize
                * (i_kw * ic_block + i_ic) * jcp.oc_block + kernel_offset),
                Zmm(i_kw * ic_block_step + i_ic));
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_ic_block_step_4fma(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{
    // TODO: add prefetches to fma version as well

    assert(jcp.ver == ver_4fma);

    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    auto zmm_ker = [=](int i_kw, int i_ic) {
        return Zmm(i_kw * ic_block_step + i_ic);
    };

    auto ker_addr = [=](int i_kw, int i_ic) {
        size_t local_offset
            = jcp.typesize_out * (i_kw * ic_block + i_ic) * jcp.oc_block;
        return EVEX_compress_addr(reg_kernel, local_offset + kernel_offset);
    };

    auto inp_addr = [=](int i_iw, int i_ic, ptrdiff_t extra_offset = 0) {
        int stride = jcp.tr_iw * (jcp.is_1stconv ? jcp.ih : 1);
        int local_offset = jcp.typesize_in * (i_iw + i_ic * stride);
        return EVEX_compress_addr(reg_input,
                local_offset + input_offset + extra_offset);
    };

    auto zmm_out = [=](int i_iw) {
        // TODO: move reg calc to global member funcs
        const int out_zmm_base_idx = 28;
        return Zmm(out_zmm_base_idx + i_iw % 4);
    };

    auto out_addr = [=](int i_ur) {
        return EVEX_compress_addr(reg_output,
                jcp.typesize_in * i_ur * oc_block + output_offset);
    };

    auto pf_callback = [=](int i_ur, int i_kw, int i_ic) {
        assert(i_ur % 4 == 0);
        if (i_ur == 0)
            prefetcht1(ker_addr(i_kw, i_ic));
        if (i_ur + 4 >= ur_w)
            prefetcht0(ker_addr(i_kw, i_ic));

        const ptrdiff_t next_input_block_offset
            = jcp.typesize_in * ic_block_step * jcp.tr_iw;
        if (i_ur % 16 == 4 && i_kw == 0) {
            if (i_ur + 16 < ur_w)
                prefetcht0(inp_addr(i_ur + 16, i_ic));
            else
                prefetcht0(inp_addr(0, i_ic, next_input_block_offset));
        }
        if (i_ur % 16 == 4 && i_kw == 1) {
            if (input_wraparound)
                prefetcht1(inp_addr(i_ur, i_ic, -input_offset));
            else
                prefetcht1(inp_addr(i_ur, i_ic, next_input_block_offset));
        }
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto zmm = zmm_ker(i_kw, i_ic);
            vpxord(zmm, zmm, zmm);
        }

    for (int i_ur = 0; i_ur < ur_w; i_ur += 4) {

        for (int i = 0; i < 4; i++) {
            auto zmm = zmm_out(i_ur + i);
            if (i_ur + i < ur_w)
                vmovups(zmm, out_addr(i_ur + i));
            else
                vpxord(zmm, zmm, zmm);
            prefetcht0(out_addr(i_ur + i + 4));
        }

        for (int i_kw = 0; i_kw < kw; i_kw++)
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                int i_iw = i_ur + i_kw;
                v4fmaddps(zmm_ker(i_kw, i_ic),
                        zmm_out(i_ur), inp_addr(i_iw, i_ic));
                pf_callback(i_ur, i_kw, i_ic);
            }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto addr = ker_addr(i_kw, i_ic);
            auto zmm = zmm_ker(i_kw, i_ic);
            vaddps(zmm, zmm, addr);
            vmovups(addr, zmm);
        }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_ic_block_step(
    int ur_w, int pad_l, int pad_r,
    int ic_block_step, int input_offset, int kernel_offset,
    int output_offset, bool input_wraparound)
{
    if (jcp.ver == ver_4fma)
        compute_ic_block_step_4fma(ur_w, pad_l, pad_r,
                ic_block_step, input_offset, kernel_offset, output_offset,
                input_wraparound);
    else if (jcp.ver == ver_fma)
        compute_ic_block_step_fma(ur_w, pad_l, pad_r,
                ic_block_step, input_offset, kernel_offset, output_offset,
                input_wraparound);
    else
        assert(!"unknown convolution version");
}

void jit_avx512_common_conv_bwd_weights_kernel_f32
    ::compute_oh_step_unroll_ow_icblock(
    int ic_block_step, int max_ur_w)
{
    UNUSED(max_ur_w);

    Label kh_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;
    int iw = jcp.ver == ver_4fma ? jcp.tr_iw : jcp.iw;
    int ow = jcp.ow;

    int r_pad = nstl::max(0, (ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step) {
            const int input_offset = jcp.typesize_in
                * (jcp.ver == ver_4fma ? i_b_ic * iw : i_b_ic);
            compute_ic_block_step(jcp.ur_w, l_pad, r_pad, ic_block_step,
                input_offset, jcp.typesize_out * i_b_ic * jcp.oc_block, 0,
                i_b_ic + ic_block_step >= jcp.ic_block);
        }
        add(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul);
        add(reg_kernel, jcp.typesize_out * jcp.kw * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mul);
        add(aux_reg_kernel, jcp.typesize_out * jcp.kh * jcp.kw * ic_block
            * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32
    ::compute_oh_step_unroll_ow(
    int ic_block_step, int max_ur_w)
{
    Label kh_label, ic_block_label, kd_label;

    UNUSED(max_ur_w);

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int ow = jcp.ow;

    int r_pad = nstl::max(0,
        (ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
        - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.l_pad;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            compute_ic_block_step(ow, l_pad, r_pad, ic_block_step,
                0, 0, 0);
            size_t inp_icblk_stride = jcp.is_1stconv
                ? (size_t)jcp.ih * jcp.iw * jcp.id
                : (jcp.ver == ver_4fma ? jcp.tr_iw : 1);
            size_t input_offset
                = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            safe_add(reg_input, input_offset, reg_long_offt);
            add(reg_kernel, jcp.typesize_out * ic_block_step * oc_block);
            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }

        if (jcp.is_1stconv) {
            size_t input_offset
                = (size_t)jcp.typesize_in * jcp.id * jcp.ih * jcp.iw * ic_block;
            safe_sub(reg_input, input_offset, reg_long_offt);
            add(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw);
        } else if (jcp.ver != ver_4fma) {
            add(reg_input, jcp.typesize_in
                    * ((jcp.dilate_h + 1) * jcp.iw - 1) * ic_block);
        }
        add(reg_kernel, jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_input, jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih
                * jcp.iw * (jcp.is_1stconv ? 1 : ic_block));
        add(aux_reg_kernel, jcp.typesize_out * jcp.kh * jcp.kw * ic_block
            * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32
    ::compute_oh_step_common(
    int ic_block_step, int max_ur_w)
{
    Label kh_label, ic_block_label, ow_block_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int ow = jcp.ow;
    int r_pad = nstl::max(0, (ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    int l_pad = jcp.ver == ver_4fma ? 0 : jcp.l_pad;

    int ur_w = nstl::min(ow, max_ur_w);
    int ur_w_trips = ow / ur_w;
    int ur_w_tail = ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0)
        || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }

    int inp_mult = (jcp.is_1stconv || jcp.ver == ver_4fma) ? 1 : ic_block;
    int input_comeback = (ur_w_trips * ur_w * jcp.stride_w - l_pad) * inp_mult;
    int output_comeback = ur_w_trips * ur_w * oc_block;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label); {
        xor_(b_ic, b_ic);
        L(ic_block_label); {
            if (l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
                add(reg_input, jcp.typesize_in * (ur_w * jcp.stride_w - l_pad)
                    * inp_mult);
                add(reg_output, jcp.typesize_in * ur_w * oc_block);
            }

            if (ur_w_trips > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                L(ow_block_label); {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add(reg_input, jcp.typesize_in * ur_w * jcp.stride_w
                        * inp_mult);
                    add(reg_output, jcp.typesize_in * ur_w * oc_block);

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_trips);
                    jl(ow_block_label, T_NEAR);
                }
            }

            if (ur_w_tail > 0) compute_ic_block_step(ur_w_tail, 0, r_pad,
                ic_block_step, 0, 0, 0);

            sub(reg_input, jcp.typesize_in * input_comeback);
            sub(reg_output, jcp.typesize_in * output_comeback);
            int inp_icblk_stride = jcp.is_1stconv
                ? jcp.ih * jcp.iw * jcp.id
                : (jcp.ver == ver_4fma ? jcp.tr_iw : 1);
            size_t input_offset
                = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            safe_add(reg_input, input_offset, reg_long_offt);
            add(reg_kernel, jcp.typesize_out * ic_block_step * oc_block);

            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
        if (jcp.is_1stconv) {
            size_t input_offset
                = (size_t)jcp.typesize_in * jcp.id * jcp.ih * jcp.iw * ic_block;
            safe_sub(reg_input, input_offset, reg_long_offt);
            add(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw);
        } else if (jcp.ver != ver_4fma) {
            add(reg_input, jcp.typesize_in
                    * ((jcp.dilate_h + 1 ) * jcp.iw - 1) * ic_block);
        }
        add(reg_kernel, jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_input, jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih
                * jcp.iw * (jcp.is_1stconv ? 1 : ic_block));
        add(aux_reg_kernel, jcp.typesize_out * jcp.kh * jcp.kw * ic_block
            * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32
    ::compute_oh_step_disp()
{
    int ic_block_step = jcp.kw <= 3 ? 8 : (jcp.kw <= 7 ? 4 : 2);
    if (jcp.is_1stconv) {
        bool large_code = jcp.kw >= 7 && (jcp.l_pad > 0 || jcp.t_pad > 0);
        ic_block_step
            = (jcp.kw * jcp.ic_block <= 28 && !large_code) ? jcp.ic_block : 1;
    }

    bool too_large_to_unroll
        = (jcp.kw > 1 || jcp.kh > 1 || jcp.kd > 1)
        && (jcp.stride_w > 1 || jcp.stride_h > 1 || jcp.stride_d > 1);

    int ow = jcp.ow;
    if (jcp.ndims == 5) {
        /* NOTE: reg_kd_count = aux_reg_input = r12. The following order of
         * 'movs' must be guaranteed. */
        mov(ki, reg_kd_count);
        push(reg_kd_count);
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
    }

    if (jcp.kw <= 3 && ow <= 16 && !too_large_to_unroll)
        compute_oh_step_unroll_ow_icblock(ic_block_step, max_ur_w);
    else if (ow <= max_ur_w)
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else
        compute_oh_step_common(ic_block_step, max_ur_w);

    if (jcp.ndims == 5) {
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
        pop(reg_kd_count);
        od_step_comeback_pointers();
    } else {
        oh_step_comeback_pointers();
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::maybe_zero_kernel()
{
    Label skip_zeroing, zeroing_loop;

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jz(skip_zeroing, T_NEAR);

    Zmm zero = Zmm(0);
    vpxord(zero, zero, zero);
    xor_(reg_tmp, reg_tmp);
    L(zeroing_loop); {
        assert(jcp.oc_block * jcp.typesize_out
            == cpu_isa_traits<avx512_common>::vlen);
        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
            vmovups(ptr[reg_kernel + reg_tmp + ic1 * jcp.oc_block
                * jcp.typesize_out], zero);
        add(reg_tmp, jcp.ic_block * jcp.oc_block * jcp.typesize_out);
        cmp(reg_tmp, jcp.ic_block * jcp.oc_block * jcp.kw * jcp.kh * jcp.kd
            * jcp.typesize_out);
        jnz(zeroing_loop);
    }

    L(skip_zeroing);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::bias_kernel()
{
    Label skip_bias, bias_loop, skip_load_bias;

    mov(reg_tmp, ptr[param + GET_OFF(flags)]);
    test(reg_tmp,reg_tmp);
    jne(skip_bias, T_NEAR);

    mov(reg_bias, ptr[param + GET_OFF(bias)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    vpxord(Zmm(1), Zmm(1), Zmm(1));

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jne(skip_load_bias, T_NEAR);
    vmovups(Zmm(1), ptr[reg_bias]);

    L(skip_load_bias);

    mov(reg_oi, ptr[param + GET_OFF(d_worksize)]);
    sub(reg_oi, ptr[param + GET_OFF(d_index)]);
    mov(reg_tmp, jcp.oc_block * jcp.ow * jcp.oh * jcp.typesize_out);
    imul(reg_oi, reg_tmp);

    xor_(reg_tmp, reg_tmp);
    L(bias_loop); {
        vmovups(Zmm(0), ptr[reg_output + reg_tmp]);
        vaddps(Zmm(1), Zmm(1), Zmm(0));
        add(reg_tmp, jcp.oc_block * jcp.typesize_out);
        cmp(reg_tmp, reg_oi);
        jl(bias_loop);
    }
    vmovups(EVEX_compress_addr(reg_bias,0), Zmm(1));

    L(skip_bias);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32
    ::compute_oh_loop_common()
{
    int b_pad = jcp.b_pad;
    int t_pad = jcp.t_pad;
    bool is_dilated = jcp.dilate_h != 0;
    int dilate_h = jcp.dilate_h + 1;
    int stride_h = jcp.stride_h;
    const int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    int iw = jcp.ver == ver_4fma ? jcp.tr_iw : jcp.iw;
    Label oh_label, oh_label_end, oh_tpad_label, oh_tpad_tail_label,
            oh_bpad_label, oh_bpad_label_end, od_label, od_label_end,
            oh_dilate_label_shift, oh_dilate_label_noshift, oh_dilate_label_end;

    int ow = jcp.ow;

    mov(reg_kh, jcp.kh);
    xor_(reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj);
    /* Compute 'top' edge */
    if (t_pad > 0) {
        const int kh_range = 1 + (jcp.kh - 1) * dilate_h;
        const int overflow
            = nstl::max(0, jcp.kh - div_up(t_pad + jcp.ih, dilate_h));
        const int underflow = div_up(t_pad, dilate_h);
        const int initial_inp_ker_overlap = jcp.kh - overflow - underflow;
        mov(reg_kh, initial_inp_ker_overlap);
        add(reg_kernel, jcp.typesize_out * underflow * jcp.kw * jcp.ic_block
            * jcp.oc_block);
        // generate loop to process kernel while it remains within t_pad + ih
        if (kh_range < t_pad + jcp.ih) {
            if (is_dilated) {
                const int tail = t_pad % dilate_h;
                const int shift = tail == 0 ? 0 : dilate_h - tail;
                mov(reg_tmp, shift);
                if (tail != 0)
                    add(reg_input, jcp.typesize_in * shift * iw * inp_mult);
            }
            L(oh_tpad_label); {
                compute_oh_step_disp();
                add(reg_output, jcp.typesize_in * ow * jcp.oc_block);
                if (is_dilated) {
                    inc(reg_tmp);
                    cmp(reg_tmp, dilate_h);
                    jl(oh_dilate_label_shift, T_NEAR);
                    // unshift input as new kernel element enters
                    sub(reg_input, jcp.typesize_in * (dilate_h - 1) * iw * inp_mult);
                    xor_(reg_tmp, reg_tmp);
                }
                // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
                sub(reg_kernel, jcp.typesize_out * stride_h * jcp.kw
                                * jcp.ic_block * jcp.oc_block);
                add(reg_kh, stride_h);
                if (is_dilated) {
                    jmp(oh_dilate_label_noshift, T_NEAR);
                    L(oh_dilate_label_shift);
                    // shift input as old kernel element progresses
                    add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
                    L(oh_dilate_label_noshift);
                }
                inc(reg_oj);
                add(reg_ih_count, stride_h);

                // final number of kernel elements that overlap with input
                const int final_inp_ker_overlap
                    = nstl::min(jcp.kh, div_up(jcp.ih, dilate_h));
                cmp(reg_kh, final_inp_ker_overlap);
                jl(oh_tpad_label, T_NEAR);
            }
        }
        // need second loop to process kernel if it is larger than the input
        // (does not apply to dilations as they must have unit stride)
        if (kh_range >= jcp.ih + (t_pad % stride_h == 0 ? stride_h :
                                                        t_pad % stride_h)) {
            assert(!is_dilated);
            mov(reg_kh, jcp.ih);
            L(oh_tpad_tail_label); {
                compute_oh_step_disp();
                add(reg_output, jcp.typesize_in * ow * jcp.oc_block);
                sub(reg_kernel, jcp.typesize_out * stride_h * jcp.kw
                                * jcp.ic_block * jcp.oc_block);

                inc(reg_oj);
                add(reg_ih_count, stride_h);

                cmp(reg_ih_count, nstl::min(t_pad, jcp.oh * stride_h));
                jl(oh_tpad_tail_label, T_NEAR);
            }
        }
        // correct any excess shifts to kernel and input
        // (does not apply to dilations as they must have unit stride,
        //  kernel must fit inside input, and padding is smaller than input)
        if (t_pad <= jcp.oh * stride_h) {
            // kernel has moved beyond padding (adjust for stride effects)
            if (t_pad % stride_h != 0) {
                assert(!is_dilated);
                int inp_corr = stride_h - t_pad % stride_h;
                add(reg_kernel, jcp.typesize_out * inp_corr * jcp.kw
                                * jcp.ic_block * jcp.oc_block);
                add(reg_input, jcp.typesize_in * inp_corr * iw * inp_mult);
            }
        } else {
            // kernel still overlaps padding (complete reset)
            assert(!is_dilated);
            sub(reg_kernel, jcp.typesize_out * (t_pad - jcp.oh * stride_h)
                            * jcp.kw * jcp.ic_block * jcp.oc_block);
        }
    }

    cmp(reg_ih_count, jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h);
    jge(oh_label_end, T_NEAR);
    cmp(reg_oj, jcp.oh);
    jge(oh_label, T_NEAR);

    /* Compute middle block(s) */
    mov(reg_kh, jcp.kh);
    L(oh_label); {
        compute_oh_step_disp();
        add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
        add(reg_output, jcp.typesize_in * ow * jcp.oc_block);

        inc(reg_oj);
        add(reg_ih_count, stride_h);

        cmp(reg_ih_count, jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h);
        jge(oh_label_end, T_NEAR);

        cmp(reg_oj, jcp.oh);
        jl(oh_label, T_NEAR);
    }
    L(oh_label_end);

    /* Compute bottom edge */
    if (b_pad > 0) {
        cmp(reg_oj, jcp.oh);
        jge(oh_bpad_label_end, T_NEAR);

        if (is_dilated) {
            mov(reg_kh, jcp.kh - 1); // assumes unit stride for dilations
            mov(reg_tmp, 0);
        } else {
            mov(reg_kh, jcp.ihp - b_pad);
            sub(reg_kh, reg_ih_count);
        }
        L(oh_bpad_label);
        {
            compute_oh_step_disp();
            add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
            add(reg_output, jcp.typesize_in * ow * jcp.oc_block);
            if (is_dilated) {
                inc(reg_tmp);
                cmp(reg_tmp, dilate_h);
                jl(oh_dilate_label_end, T_NEAR);
                xor_(reg_tmp, reg_tmp);
            }
            sub(reg_kh, stride_h);
            cmp(reg_kh, 0);
            jle(oh_bpad_label_end, T_NEAR);
            if (is_dilated)
                L(oh_dilate_label_end);

            inc(reg_oj);
            cmp(reg_oj, jcp.oh);
            jl(oh_bpad_label, T_NEAR);
        }
        L(oh_bpad_label_end);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_d_loop_common() {
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    const int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    int iw = jcp.ver == ver_4fma ? jcp.tr_iw : jcp.iw;
    int ow = jcp.ow;
    const int input_backpad_overlap
            = div_up(jcp.id + jcp.f_pad - (jcp.kd - 1), jcp.stride_d);

    const size_t filter_shift
            = jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block;
    const size_t input_shift = jcp.typesize_in * jcp.ih * iw * inp_mult;
    const size_t output_shift = jcp.typesize_in * jcp.oh * ow * jcp.oc_block;

    Label d_loop_label, loop_end_label, common_block_label, fpad_end_label,
            backpad_end_label, backpad_label;

    if (jcp.with_bias) bias_kernel();

    /* initially offset 'kd' by f_pad */
    add(reg_kernel, ptr[param + GET_OFF(kd_offset)]);

    mov(reg_input_d, ptr[param + GET_OFF(src)]);
    mov(reg_output_d, ptr[param + GET_OFF(dst)]);
    mov(reg_d_index, ptr[param + GET_OFF(d_index)]);
    mov(reg_kd_count, ptr[param + GET_OFF(kd_padding)]);

    cmp(reg_d_index, ptr[param + GET_OFF(d_worksize)]);
    jge(loop_end_label, T_NEAR);

    L(d_loop_label);

    mov(reg_input, reg_input_d);
    mov(reg_output, reg_output_d);

    push(reg_input_d);
    push(reg_output_d);
    push(reg_d_index);

    compute_oh_loop_common();

    pop(reg_d_index);
    pop(reg_output_d);
    pop(reg_input_d);

    /* Compute 'front' edge */
    if (jcp.f_pad > 0) {

        /* Check if within fpad region */
        cmp(reg_d_index, div_up(jcp.f_pad, jcp.stride_d));
        jge(fpad_end_label, T_NEAR);

        /* Fpad steps */
        sub(reg_kernel, filter_shift * jcp.stride_d);
        add(reg_kd_count, jcp.stride_d);

        /* Final number of kernel elements that overlap with input */
        const int inp_ker_overlap = nstl::min(jcp.kd, jcp.id);
        cmp(reg_kd_count, inp_ker_overlap);
        jl(common_block_label, T_NEAR);

        /* Correct any excess shifts to kernel and input */
        if (jcp.f_pad <= jcp.od * jcp.stride_d) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.f_pad % jcp.stride_d != 0) {
                int inp_corr = jcp.stride_d - jcp.f_pad % jcp.stride_d;
                add(reg_kernel, filter_shift * inp_corr);
                add(reg_input_d, input_shift * inp_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub(reg_kernel, (jcp.f_pad - jcp.od * jcp.stride_d) * filter_shift);
        }

        /* Apply correction */
        mov(reg_kd_count, jcp.kd);
        jmp(common_block_label);

        L(fpad_end_label);
    }

    /* Compute bottom edge */
    if (jcp.back_pad > 0) {

        /* Check if within back_pad region */
        cmp(reg_d_index, input_backpad_overlap - 1);
        jl(backpad_end_label, T_NEAR);
        jg(backpad_label, T_NEAR);

        /* Execute overlap correction between the filter and the initial
         * back_pad region. */
        mov(reg_kd_count,
                jcp.id + jcp.f_pad - input_backpad_overlap * jcp.stride_d);
        jmp(backpad_end_label, T_NEAR);

        L(backpad_label);
        sub(reg_kd_count, jcp.stride_d);
        cmp(reg_kd_count, 0);
        jle(loop_end_label, T_NEAR);

        L(backpad_end_label);
    }

    /* Compute middle block */
    add(reg_input_d, input_shift * jcp.stride_d);

    /* Execute common block and loop */
    L(common_block_label);
    add(reg_output_d, output_shift);
    inc(reg_d_index);
    cmp(reg_d_index, ptr[param + GET_OFF(d_worksize)]);
    jl(d_loop_label, T_NEAR);

    L(loop_end_label);
}

bool jit_avx512_common_conv_bwd_weights_kernel_f32::compute_full_spat_loop() {
    // FIXME: use register mapping from the class declaration
    bool ok = jcp.ver == ver_4fma
        && everyone_is(0, jcp.dilate_h, jcp.dilate_w)
        && everyone_is(1, jcp.stride_h, jcp.stride_w);
    if (!ok) return false;
    if (jcp.l_pad != jcp.kw / 2 || jcp.t_pad != jcp.kh / 2)
        return false;

    // General code layout:
    //
    // Blocking over OH -- top level
    // (Reduces L2 pressure; not very useful right now)
    //  Loop over all KHxKW kernel -- emit_kh_kw_loop()
    //    Loop over OH block -- emit_h_loop()
    //      Loop over OW blocks -- emit_fma_block()
    //      (Supports both fully unrolled and partially unrolled versions to
    //      reduce code size)
    //          Loop over OW block -- emit_fma_step()

    int max_working_set_size = 128 * 1024;
    int pad_ow = jcp.ow;

    int inp_row_size = jcp.ic_block * jcp.tr_iw * jcp.typesize_in;
    int out_row_size = jcp.oc_block * pad_ow * jcp.typesize_in;
    int row_size = inp_row_size + out_row_size;

    int h_block_size = jcp.oh;
    int working_set_size = row_size * h_block_size;

    if (working_set_size > max_working_set_size) {
        int opt_working_set_size = 48 * 1024;
        assert(opt_working_set_size < max_working_set_size);

        while (working_set_size > opt_working_set_size) {
            for (int i = 2; i <= h_block_size; i++)
                if (i == h_block_size)
                    h_block_size = h_block_size / 2;
                else if (h_block_size % i == 0) {
                    h_block_size = h_block_size / i;
                    break;
                }
            working_set_size = row_size * h_block_size;

            if (h_block_size == 1 && working_set_size > opt_working_set_size)
                return false;
        }
    }

    // NB1: t_pad <= oh_block_size and b_pad <= last_oh_block_size (see below)
    if (h_block_size < nstl::max(1, jcp.t_pad)
            || jcp.b_pad > (jcp.oh % h_block_size == 0 ? h_block_size
                                                       : jcp.oh % h_block_size))
        return false;

    // check that we can use simple arithmetic for prefetch address
    // calculations
    // TODO: we need some traits for this check (Roma)
    int cache_line_size = 64;
    assert(jcp.ic_block * typesize == 64);
    assert(jcp.oc_block * typesize == 64);

    int num_inp_l2_pfs = jcp.tr_iw * h_block_size;
    int avg_h_loop_len = h_block_size;
    int num_inp_l2_pfs_per_fma_block
        = div_up(num_inp_l2_pfs, avg_h_loop_len * jcp.kw * jcp.kh);
    int num_out_l2_pfs = pad_ow * h_block_size;
    int num_out_l2_pfs_per_fma_block
        = div_up(num_out_l2_pfs, avg_h_loop_len * jcp.kw * jcp.kh);

    Opmask reg_h_block = k1; // 32-bit only on Intel(R) Xeon Phi(TM) processors
    Reg64 reg_kh = rax;
    Reg64 reg_kw = rbx;
    Reg64 reg_tmp = abi_not_param1;
    Reg32 reg_tmp_w = reg_tmp.cvt32();
    Reg64 reg_ohs = rdx;
    Reg64 reg_ihs = rsi;
    Reg64 reg_h = r8;
    Reg64 reg_i = r9;
    Reg64 reg_j = r10;

    Reg64 reg_inp = r13;
    Reg64 reg_out = r14;
    Reg64 reg_ker = r15;

    Reg64 reg_inp_pf_l1 = rbp;

    Reg64 reg_inp_pf_l2 = r11;
    Reg64 reg_out_pf_l2 = r12;

    Xmm reg_inp_pf_save = xmm17;
    Xmm reg_out_pf_save = xmm18;

    Reg64 reg_inp_save = abi_param1;
    Reg64 reg_out_save = reg_tmp;

    auto zmm_out = [&](int oi) { return Zmm(24 + oi % 8); };
    auto zmm_ker = [&](int ic1) { return Zmm(ic1); };
    auto inp_addr = [&](int oi, int ic1) {
        return ptr[reg_inp + (ic1 * jcp.tr_iw + oi) * jcp.typesize_in];
    };
    auto out_addr = [&](int oi, int oj = 0) {
        assert(jcp.ver == ver_4fma);
        return ptr[reg_out
            + ((oi + oj * jcp.ow) * jcp.oc_block) * jcp.typesize_in];
    };
    auto ker_addr = [&](int ic1) {
        return ptr[reg_ker + ic1 * jcp.oc_block * jcp.typesize_out];
    };

    auto emit_block = [&](int h_block_size,
            bool is_last_block, bool is_last_kh_kw_iter, bool is_last_row)
    {
        // TODO: add an fma version (Roma)
        auto pad_ow = jcp.ow;

        int ow4u = rnd_up(pad_ow, 4);
        int def_step_size = 16;

        bool has_w_tail = (pad_ow % def_step_size != 0
                || pad_ow % 4 != 0);
        bool full_w_unroll = pad_ow / def_step_size < 2 + has_w_tail;

        auto emit_step = [&](int ur_ow,
                int num_inp_l1_pfs_per_fma_step,
                int num_inp_l2_pfs_per_fma_step,
                int num_out_l2_pfs_per_fma_step, bool is_w_tail)
        {
            bool block_wraparound = is_w_tail && is_last_row;

            assert(ur_ow % 4 == 0);
            int tail_size = ow4u % ur_ow;
            int this_ur_ow
                = (is_w_tail && tail_size) ? tail_size : ur_ow;
            int ow_last_chunk4 = pad_ow % 4;
            int ow_zero_tail4 = ow_last_chunk4
                ? 4 - ow_last_chunk4 : 0;

            auto emit_out_pf = [&](int oi) {
#if 1
                if (oi + def_step_size < ur_ow || !block_wraparound)
                    mic_prefetcht0(ptr[reg_out
                            + ((def_step_size + oi)
                                * jcp.oc_block * jcp.typesize_in)]);
                else {
                    assert(block_wraparound);
                    assert(oi + def_step_size >= ur_ow);
                    mic_prefetcht0(ptr[reg_out_save
                            + ((oi + def_step_size - ur_ow)
                                * jcp.oc_block * jcp.typesize_in)]);
                }
#else
                // XXX: This is an alternative prefetching strategy that
                // always prefetches the next row. Keeping it here for
                // future experiments (Roma)
                if (!block_wraparound)
                    mic_prefetcht0(ptr[reg_out
                            + (jcp.ow + oi) * jcp.oc_block * jcp.typesize_in]);
                else
                    mic_prefetcht0(ptr[reg_out + reg_ohs
                            - ((h_block_size - 1) * jcp.ow
                                - oi) * jcp.oc_block * jcp.typesize_in]);
#endif
                if (oi < num_out_l2_pfs_per_fma_step)
                    mic_prefetcht1(ptr[reg_out_pf_l2
                            + oi * jcp.oc_block * jcp.typesize_in]);
            };

            auto emit_inp_pf = [&](int oi4, int ic1) {
                int pf_slot_idx = ic1 + oi4 / 4 * jcp.ic_block;
                int num_pf_slots = jcp.ic_block * ur_ow / 4;

                int num_pfs = num_inp_l1_pfs_per_fma_step
                    + num_inp_l2_pfs_per_fma_step;
                int pf_freq = nstl::max(1, num_pf_slots / num_pfs);

                if (pf_slot_idx % pf_freq)
                    return;

                int pf_idx = pf_slot_idx / pf_freq;

                if (pf_idx < num_inp_l2_pfs_per_fma_step)
                    mic_prefetcht1(ptr[reg_inp_pf_l2
                            + pf_idx * jcp.ic_block * jcp.typesize_in]);
                else {
                    pf_idx -= num_inp_l2_pfs_per_fma_step;
                    // prefetch the 'tail' of the cache line because most of
                    // the accesses are not aligned
                    mic_prefetcht0(ptr[reg_inp_pf_l1
                            + pf_idx * jcp.ic_block * jcp.typesize_in
                            + cache_line_size - jcp.typesize_in]);
                }
            };

            auto numloads = 4;

            int steps = this_ur_ow;
            for (int oi4 = 0; oi4 < steps; oi4 += numloads) {
                for (int oi1 = 0; oi1 < numloads; oi1++) {
                    int oi = oi4 + oi1;
                    if (!is_w_tail || oi < (this_ur_ow - ow_zero_tail4)) {
                        vmovups(zmm_out(oi), out_addr(oi));
                        emit_out_pf(oi);
                    } else {
                        auto zmm = zmm_out(oi);
                        vpxord(zmm, zmm, zmm);
                    }
                }

                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    if (jcp.ver == ver_4fma) {
                        v4fmaddps(zmm_ker(ic1),
                                zmm_out(oi4), inp_addr(oi4, ic1));
                    } else {
                        assert(!"unknown convolution version");
                    }
                        emit_inp_pf(oi4, ic1);
                }
            }
        };

        // Input is transposed and padded but we only access about jcp.iw
        // elements so use that to compute the # of cache lines in each 'row'
        int num_inp_l1_pfs
            = div_up(jcp.iw * jcp.typesize_in, cache_line_size) * jcp.ic_block;

        if (full_w_unroll) {
            emit_step(ow4u, num_inp_l1_pfs,
                    num_inp_l2_pfs_per_fma_block,
                    num_out_l2_pfs_per_fma_block, true);
            add(reg_inp_pf_l2, num_inp_l2_pfs_per_fma_block * cache_line_size);
            add(reg_out_pf_l2, num_out_l2_pfs_per_fma_block * cache_line_size);
        } else {
            Label w_loop;
            int num_w_iters = pad_ow / def_step_size;
            int num_w_iters_full = num_w_iters + has_w_tail;
            int num_inp_l1_pfs_per_fma_step
                = div_up(num_inp_l1_pfs, num_w_iters_full);
            int num_inp_l2_pfs_per_fma_step
                = div_up(num_inp_l2_pfs_per_fma_block, num_w_iters_full);
            int num_out_l2_pfs_per_fma_step
                = div_up(num_out_l2_pfs_per_fma_block, num_w_iters_full);
            mov(reg_i, num_w_iters);
            L(w_loop); {
                emit_step(def_step_size, num_inp_l1_pfs_per_fma_step,
                        num_inp_l2_pfs_per_fma_step,
                        num_out_l2_pfs_per_fma_step, false);
                add(reg_inp, def_step_size * jcp.typesize_in);
                add(reg_out, def_step_size * jcp.oc_block * jcp.typesize_in);
                add(reg_inp_pf_l1,
                        num_inp_l1_pfs_per_fma_step * cache_line_size);
                add(reg_inp_pf_l2,
                        num_inp_l2_pfs_per_fma_step * cache_line_size);
                add(reg_out_pf_l2,
                        num_out_l2_pfs_per_fma_step * cache_line_size);
                sub(reg_i, 1);
                jnz(w_loop);
            }
            if (has_w_tail) {
                emit_step(def_step_size, num_inp_l1_pfs_per_fma_step,
                        num_inp_l2_pfs_per_fma_step,
                        num_out_l2_pfs_per_fma_step, true);
                add(reg_inp_pf_l2,
                        num_inp_l2_pfs_per_fma_step * cache_line_size);
                add(reg_out_pf_l2,
                        num_out_l2_pfs_per_fma_step * cache_line_size);
            }
            // reset reg_inp and reg_out because emit_h_loop expects
            // unmodified pointers
            int w_offset = num_w_iters * def_step_size;
            sub(reg_inp, w_offset * jcp.typesize_in);
            sub(reg_out, w_offset * jcp.oc_block * jcp.typesize_in);
        }
    };

    auto emit_h_loop = [&](int h_block_size,
            bool is_last_block, bool is_last_kh_kw_iter)
    {
        Label h_loop, skip_h_loop;
        mov(reg_j, 1);
        cmp(reg_j, reg_h);
        je(skip_h_loop, T_NEAR);
        L(h_loop); {

            lea(reg_inp_pf_l1,
                    ptr[reg_inp + jcp.tr_iw * jcp.ic_block * jcp.typesize_in]);
            emit_block(h_block_size,
                    is_last_block, is_last_kh_kw_iter, false);

            add(reg_inp, jcp.tr_iw * jcp.ic_block * jcp.typesize_in);
            add(reg_out, pad_ow * jcp.oc_block * jcp.typesize_in);
            add(reg_j, 1);
            cmp(reg_j, reg_h);
            jb(h_loop);
        }

        L(skip_h_loop);

        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
            mic_prefetcht0(ker_addr(ic1));

        lea(reg_inp_pf_l1, ptr[reg_inp_save + reg_kw * jcp.typesize_in]);
        emit_block(h_block_size, is_last_block, is_last_kh_kw_iter, true);
    };

    auto emit_kh_kw_loop = [&](bool is_first_block, bool is_last_block,
            int h_block_size)
    {
        xor_(reg_kh, reg_kh);
        Label kh_loop, kh_loop_end;

        int last_oh_block_size
            = jcp.oh - rnd_up(jcp.oh - h_block_size, h_block_size);
        int oh_block_size = (is_last_block) ? last_oh_block_size : h_block_size;
        // NB1: t_pad <= oh_block_size and b_pad <= last_oh_block_size
        int ih_block_size = oh_block_size - 1 + jcp.kh
                - is_first_block * jcp.t_pad - is_last_block * jcp.b_pad;

        L(kh_loop); {
            // determine starting indices for this block
            if (is_first_block) {
                xor_(reg_tmp, reg_tmp);
                mov(reg_ohs, jcp.t_pad);
                sub(reg_ohs, reg_kh);
                cmovb(reg_ohs, reg_tmp);

                mov(reg_ihs, reg_ohs);
                sub(reg_ihs, jcp.t_pad);
                add(reg_ihs, reg_kh);
            } else {
                xor_(reg_ohs, reg_ohs);
                mov(reg_ihs, reg_kh);
            }

            // determine effective size of block based on padding
            mov(reg_tmp, oh_block_size);
            sub(reg_tmp, reg_ohs);
            mov(reg_h, ih_block_size);
            sub(reg_h, reg_ihs);
            cmp(reg_tmp, reg_h);
            cmovb(reg_h, reg_tmp);

            Label kh_loop_work;
            cmp(reg_h, 0);
            jg(kh_loop_work, T_NEAR);

            // empty h loop for this jcp.kh:
            // - set the output to 0 if necessary
            // - move ker pt
            // - jump to the end
            sub(reg_h, 1);
            Label skip_ker_zeroing;

            // The reg_ker ptr has highest bit set if the output needs to be
            // zeroed. Those who have byte-aligned their data will suffer the
            // consiquences :(
            // TODO: move the flag to a mask register? (Roma)
            test(reg_ker, 1);
            jz(skip_ker_zeroing, T_NEAR);

            Label zeroing_loop;
            vpxord(zmm0, zmm0, zmm0);
            and_(reg_ker, ~1); // temporarily clear the zeroing flag
            mov(reg_tmp, jcp.kw);
            L(zeroing_loop); {
                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
                    vmovups(ker_addr(ic1), zmm0);
                add(reg_ker, jcp.oc_block * jcp.ic_block * jcp.typesize_out);
                sub(reg_tmp, 1);
                jnz(zeroing_loop, T_NEAR);
            }
            // restore the zeroing flag (it will be cleared after the end of
            // emit_kh_kw_loop, but we may need it until then)
            or_(reg_ker, 1);
            jmp(kh_loop_end, T_NEAR);

            L(skip_ker_zeroing);
            add(reg_ker, jcp.oc_block * jcp.ic_block * jcp.kw
                * jcp.typesize_out);
            jmp(kh_loop_end, T_NEAR);

            L(kh_loop_work);

            mul_by_const(reg_ihs, reg_tmp,
                    jcp.tr_iw * jcp.ic_block * jcp.typesize_in);
            mul_by_const(reg_ohs, reg_tmp,
                    pad_ow * jcp.oc_block * jcp.typesize_in);

            add(reg_inp, reg_ihs);
            add(reg_out, reg_ohs);

            Label kw_loop;
            xor_(reg_kw, reg_kw);
            L(kw_loop); {
                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    auto zmm = zmm_ker(ic1);
                    vpxord(zmm, zmm, zmm);
                    mic_prefetcht1(ker_addr(ic1));
                }

                mov(reg_out_save, reg_out);
                mov(reg_inp_save, reg_inp);
                lea(reg_inp, ptr[reg_inp + reg_kw * jcp.typesize_in]);

#if 0
                // XXX: Generate code with special prefetches when switching
                // blocks or at the end of the last block. Disabled to reduce
                // code size and because there's no performance benefit (Roma)
                Label regular_h_loop, end_h_loop;
                cmp(reg_kw, jcp.kw - 1);
                jne(regular_h_loop, T_NEAR);
                cmp(reg_kh, jcp.kh - 1);
                jne(regular_h_loop, T_NEAR);

                emit_h_loop(oh_block_size, is_last_block, true);
                jmp(end_h_loop, T_NEAR);

                L(regular_h_loop);
                emit_h_loop(oh_block_size, is_last_block, false);

                L(end_h_loop);
#else
                emit_h_loop(oh_block_size, is_last_block, false);
#endif

                mov(reg_out, reg_out_save);
                mov(reg_inp, reg_inp_save);

                Label do_store;
                // The reg_ker ptr has highest bit set if the output needs to
                // be zeroed. Those who have byte-aligned their data will
                // suffer the consiquences :(
                mov(reg_tmp, reg_ker);
                and_(reg_ker, ~1);
                test(reg_tmp, 1);
                jnz(do_store, T_NEAR);

                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    auto zmm = zmm_ker(ic1);
                    if (jcp.ver == ver_4fma) {
                        vaddps(zmm, ker_addr(ic1));
                    } else {
                        assert(!"unknown convolution version");
                    }
                }

                L(do_store);
                for (int ic1 = 0; ic1 < jcp.ic_block; ic1++) {
                    auto zmm = zmm_ker(ic1);
                    vmovups(ker_addr(ic1), zmm);
                }

                mov(reg_ker, reg_tmp);
                add(reg_ker, jcp.ic_block * jcp.oc_block * jcp.typesize_out);
                add(reg_kw, 1);
                cmp(reg_kw, jcp.kw);
                jl(kw_loop);
            }

            sub(reg_inp, reg_ihs);
            sub(reg_out, reg_ohs);


            L(kh_loop_end);
            add(reg_kh, 1);
            cmp(reg_kh, jcp.kh);
            jl(kh_loop);
        }
    };

    mov(reg_inp, ptr[param + GET_OFF(src)]);
    mov(reg_out, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);
    mov(reg_inp_pf_l2, ptr[param + GET_OFF(src_prf)]);
    mov(reg_out_pf_l2, ptr[param + GET_OFF(dst_prf)]);
    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    or_(reg_ker, reg_tmp);

    bool single_kh_kw_loop = (h_block_size == jcp.oh);

    size_t inp_row_step = jcp.tr_iw * jcp.ic_block * jcp.typesize_in;
    size_t first_inp_block_step = inp_row_step * (h_block_size - jcp.t_pad);
    size_t inp_block_step = inp_row_step * h_block_size;
    size_t out_block_step = pad_ow * jcp.oc_block * jcp.typesize_in
        * h_block_size;

    if (!single_kh_kw_loop) {
        // Save the original prefetch pointers from the OpenMP driver
        vmovq(reg_inp_pf_save, reg_inp_pf_l2);
        vmovq(reg_out_pf_save, reg_out_pf_l2);
        mov(reg_inp_pf_l2, reg_inp);
        add(reg_inp_pf_l2, first_inp_block_step);
        mov(reg_out_pf_l2, reg_out);
        add(reg_out_pf_l2, out_block_step);
    }
    emit_kh_kw_loop(true, single_kh_kw_loop, h_block_size);

    if (!single_kh_kw_loop) {
        size_t ker_reset_offset
            = jcp.oc_block * jcp.ic_block * jcp.typesize_out * jcp.kw * jcp.kh;
        sub(reg_ker, ker_reset_offset);
        and_(reg_ker, ~1); // Clear the zeroing flag for subsequent updates

        add(reg_inp, first_inp_block_step);
        add(reg_out, out_block_step);
        mov(reg_inp_pf_l2, reg_inp);
        add(reg_inp_pf_l2, inp_block_step);
        mov(reg_out_pf_l2, reg_out);
        add(reg_out_pf_l2, out_block_step);

        int num_innermost_iters = div_up(jcp.oh, h_block_size) - 2;
        if (num_innermost_iters > 0) {
            Label h_block_loop;

            mov(reg_tmp_w, num_innermost_iters);
            kmovw(reg_h_block, reg_tmp_w);
            L(h_block_loop); {
                emit_kh_kw_loop(false, false, h_block_size);
                sub(reg_ker, ker_reset_offset);
                add(reg_inp, inp_row_step * h_block_size);
                add(reg_out, out_block_step);
                mov(reg_inp_pf_l2, reg_inp);
                add(reg_inp_pf_l2, inp_block_step);
                mov(reg_out_pf_l2, reg_out);
                add(reg_out_pf_l2, out_block_step);
                kmovw(reg_tmp_w, reg_h_block);
                sub(reg_tmp_w, 1);
                kmovw(reg_h_block, reg_tmp_w);
                jnz(h_block_loop);
            }
        }

        // Restore the original prefetch pointers that came from the OpenMP
        // driver
        vmovq(reg_inp_pf_l2, reg_inp_pf_save);
        vmovq(reg_out_pf_l2, reg_out_pf_save);
        emit_kh_kw_loop(false, true, h_block_size);
    }

    return true;
}

bool jit_avx512_common_conv_bwd_weights_kernel_f32
    ::flat_4ops_compute() {
    const auto &j = jcp;
    const bool ok = j.ver == ver_4fma && j.is_1stconv
        && everyone_is(0, j.dilate_h, j.dilate_w);
    if (!ok) return false;

    Reg64 reg_ptr_tr_src = r8;
    Reg64 reg_ptr_dst = r9;
    Reg64 reg_ptr_wei = r10;
    Reg64 reg_ptr_bia = r11;

    Reg64 reg_kh_step = rax;
    Reg64 reg_oh = abi_not_param1;
    Reg64 reg_kh = rdx;

    Reg32 reg_flag_save = ebx;
    Reg32 reg_flag = esi;

    Zmm vbia(31);

    auto zmm_wei = [&](int kh, int kw) {
        return Zmm(8 + kh * j.kw + kw);
    };
    auto zmm_dst = [&](int ow) {
        return Zmm(ow % 8);
    };

    auto addr_tr_src = [&](int kh, int iw) {
        return ptr[reg_ptr_tr_src
            + (kh * j.stride_w * j.tr_ld + iw) * jcp.typesize_in];
    };
    auto addr_dst = [&](int ow) {
        return ptr[reg_ptr_dst + ow * jcp.oc_block * jcp.typesize_in];
    };
    auto addr_wei = [&](int kh, int kw) {
        return ptr[reg_ptr_wei + (kh * j.kw + kw) * j.oc_block
            * jcp.typesize_out];
    };

    auto emit_fma_block = [&](int kh_step) {
        for (int kh = 0; kh < kh_step; ++kh) {
            for (int kw = 0; kw < j.kw; ++kw) {
                auto vwei = zmm_wei(kh, kw);
                vpxord(vwei, vwei, vwei);
            }
        }

        for (int ow = 0; ow < j.ow; ow += 4) {
            for (int _ow = ow; _ow < ow + 4; ++_ow) {
                auto vdst = zmm_dst(_ow);
                if (_ow < j.ow)
                    vmovups(vdst, addr_dst(_ow));
                else
                    vpxord(vdst, vdst, vdst);
            }

            for (int kh = 0; kh < kh_step; ++kh) {
                for (int kw = 0; kw < j.kw; ++kw) {
                    const int iw = ow + (kw % j.stride_w) * j.tr_ld
                        + (kw / j.stride_w);
                    v4fmaddps(zmm_wei(kh, kw), zmm_dst(ow),
                            addr_tr_src(kh, iw));
                    if (1 && kh == 0 && kw < 4) {
                        prefetcht1(ptr[reg_ptr_dst
                            + (j.ow + ow + kw) * jcp.oc_block
                            * jcp.typesize_in]);
                    }
                    if (j.with_bias && kh_step == 1) { /* [bwd_w:b:r1] */
                        const int off = kw + 4 - j.kw;
                        if (off >= 0 && ow + off < j.ow)
                            vaddps(vbia, vbia, zmm_dst(ow + off));
                    }
                }
            }
        }

        Label l_store;
        test(reg_flag, FLAG_MB_FIRST);
        jnz(l_store, T_NEAR);
        for (int kh = 0; kh < kh_step; ++kh) {
            for (int kw = 0; kw < j.kw; ++kw)
                vaddps(zmm_wei(kh, kw), addr_wei(kh, kw));
        }
        L(l_store);
        for (int kh = 0; kh < kh_step; ++kh) {
            for (int kw = 0; kw < j.kw; ++kw)
                vmovups(addr_wei(kh, kw), zmm_wei(kh, kw));
        }
    };

    auto emit_kh_loop = [&]() {
        const int kh_step_rem = j.kh % j.kh_step;
        xor_(reg_kh, reg_kh);
        mov(reg_kh_step, j.kh_step);

        Label l_kh_loop;
        L(l_kh_loop); {
            Label l_done;

            if (kh_step_rem != 0) {
                Label l_keep_kh_step;
                cmp(reg_kh, j.kh - j.kh_step);
                jle(l_keep_kh_step, T_NEAR);

                mov(reg_kh_step, kh_step_rem);
                emit_fma_block(kh_step_rem);
                jmp(l_done, T_NEAR);

                L(l_keep_kh_step);
            }

            emit_fma_block(j.kh_step);

            L(l_done);

            add(reg_ptr_tr_src, j.kh_step * j.stride_w * j.tr_ld
                * jcp.typesize_in);
            add(reg_ptr_wei, j.kh_step * j.kw * j.oc_block * jcp.typesize_out);
            add(reg_kh, j.kh_step);

            cmp(reg_kh, j.kh);
            jl(l_kh_loop, T_NEAR);
        }

        const int kh_steps = rnd_up(j.kh, j.kh_step);
        sub(reg_ptr_tr_src, kh_steps * j.stride_w * j.tr_ld * jcp.typesize_in);
        sub(reg_ptr_wei, kh_steps * j.kw * j.oc_block * jcp.typesize_out);
    };

    auto emit_oh_loop = [&]() {
        mov(reg_oh, j.oh);

        Label l_oh_loop;
        L(l_oh_loop); {
            Label l_restore_mb_flag, l_jump;

            cmp(reg_oh, j.oh);
            je(l_restore_mb_flag, T_NEAR);

            and_(reg_flag, ~FLAG_MB_FIRST);
            jmp(l_jump, T_NEAR);

            L(l_restore_mb_flag);
            mov(reg_flag, reg_flag_save);

            L(l_jump);

            emit_kh_loop();

            add(reg_ptr_tr_src, j.stride_h * j.stride_w * j.tr_ld
                * jcp.typesize_in);
            add(reg_ptr_dst, j.ow * j.oc_block * jcp.typesize_in);

            dec(reg_oh);
            jnz(l_oh_loop, T_NEAR);
        }
    };

    auto emit_bia_store = [&]() {
        if (!j.with_bias) return;

        Label l_bia_store, l_bia_skip;
        test(reg_flag, FLAG_IC_FIRST);
        jz(l_bia_skip);

        test(reg_flag, FLAG_MB_FIRST);
        jnz(l_bia_store, T_NEAR);
        vaddps(vbia, ptr[reg_ptr_bia]);
        L(l_bia_store);
        vmovups(ptr[reg_ptr_bia], vbia);
        L(l_bia_skip);
    };

    mov(reg_ptr_tr_src, ptr[param + GET_OFF(src)]);
    mov(reg_ptr_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ptr_wei, ptr[param + GET_OFF(filt)]);
    mov(reg_ptr_bia, ptr[param + GET_OFF(bias)]);
    mov(reg_flag_save, ptr[param + GET_OFF(flags)]);

    vpxord(vbia, vbia, vbia);
    emit_oh_loop();
    emit_bia_store();

    return true;
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::compute_loop()
{
    if (flat_4ops_compute())
        return;
    if (compute_full_spat_loop())
        return;

    maybe_zero_kernel();

    if (jcp.ndims == 5) compute_d_loop_common();
    else compute_oh_loop_common();
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::generate()
{
    preamble();

    mov(reg_input, ptr[param + GET_OFF(src)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    compute_loop();

    postamble();
}

status_t jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf(
    jit_conv_conf_t &jcp, const convolution_desc_t &cd,
    memory_desc_t &src_md, memory_desc_t &diff_weights_md,
    memory_desc_t &diff_bias_md, memory_desc_t &diff_dst_md) {
    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_bias_d(&diff_bias_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();

    jcp.simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

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

    const int kh_range = 1 + (jcp.kh - 1) * (jcp.dilate_h + 1);
    bool ok = true
        // general condition to simplify dilations
        && IMPLICATION(jcp.dilate_d != 0, jcp.stride_d == 1)
        && IMPLICATION(jcp.dilate_h != 0, jcp.stride_h == 1)
        && IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1)
        // special condition to simplify dilations in compute_oh_loop_common
        && IMPLICATION(jcp.dilate_h != 0, kh_range <= jcp.ih);
    if (!ok)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w
            + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h
            + (jcp.kh - 1) * (jcp.dilate_h + 1) - (jcp.ih + jcp.t_pad - 1));
    jcp.back_pad = nstl::max(0, (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1));

    /* XXX: currently, does not support dilation_d > 0 */
    if (ndims == 5)
        if (jcp.dilate_d > 0)
            return status::unimplemented;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.aligned_threads = 0;

    /* check for the 1st convolution */
    jcp.is_1stconv = is_1stconv(jcp);

    jcp.oc_block = jcp.simd_w;

    bool ok_to_pad_channels = true
        && jcp.ngroups == 1
        && src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels)
        jcp.oc = rnd_up(jcp.oc, jcp.simd_w);

    if (jcp.oc % jcp.oc_block)
        return status::unimplemented;

    auto dst_tag = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_tag = with_groups
        ? pick(ndims - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
        : pick(ndims - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o);

    if (diff_dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dst_tag));
        jcp.dst_tag = dst_tag;
    } else {
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(dst_tag);
    }
    if (jcp.dst_tag != dst_tag)
        return status::unimplemented;

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_bias_md, x));
    }

    jcp.nb_oc = jcp.oc / jcp.oc_block;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_pad = ((jcp.kh - 1) * (jcp.dilate_h + 1) + 1) / 2;
    const bool boundaries_ok = true
        && jcp.t_pad <= max_pad
        && jcp.b_pad <= max_pad
        && IMPLICATION(jcp.f_pad > 0, jcp.kd < jcp.id + jcp.f_pad)
        && jcp.f_pad < jcp.kd;
    if (!boundaries_ok)
        return status::unimplemented;

    /* yet another common check */
    if (jcp.kw > 14)
        return status::unimplemented;

    /* setting register strategy */
    for (int ur_w = nstl::min(max_ur_w, jcp.ow); ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) { jcp.ur_w = ur_w; break; }
    }

    if (jcp.is_1stconv) {
        auto src_tag = pick(ndims - 3, ncw, nchw, ncdhw);
        if (src_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, src_tag));
            jcp.src_tag = src_tag;
        } else {
            jcp.src_tag = src_d.matches_one_of_tag(src_tag);
            if (jcp.ic == 1 && jcp.src_tag != src_tag)
                jcp.src_tag = src_d.matches_one_of_tag(
                        pick(ndims - 3, nwc, nhwc, ndhwc));
        }
        if (jcp.src_tag == format_tag::undef)
            return status::unimplemented;

        const bool src_ok = true
            && utils::everyone_is(data_type::f32,
                src_d.data_type(), diff_weights_d.data_type(),
                diff_dst_d.data_type())
            && one_of(jcp.ic, 1, 2, 3)
            && jcp.ngroups == 1;
        if (!src_ok)
            return status::unimplemented;

        const int tr_ld = rnd_up(div_up(jcp.iw + jcp.l_pad + jcp.r_pad,
                    jcp.stride_w), 16);
        const int kh_step = nstl::max((28 - jcp.with_bias) / jcp.kw, 1);
        const int kh_step_rem = jcp.kh % kh_step;

        const auto wei_4fma_tag = with_groups
            ? pick(ndims - 3, gOiw16o, gOihw16o, gOidhw16o)
            : pick(ndims - 3, Oiw16o, Oihw16o, Oidhw16o);

        auto current_wei_tag = format_tag::undef;
        if (diff_weights_d.format_kind() != format_kind::any)
            current_wei_tag = diff_weights_d.matches_one_of_tag(wei_4fma_tag);

        const bool use_4fma = true
            && one_of(ndims, 3, 4)
            && mayiuse(avx512_mic_4ops)
            && mkldnn_thr_syncable()
            && everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w)
            && everyone_is(0, jcp.l_pad, jcp.r_pad, jcp.t_pad, jcp.b_pad)
            && jcp.kw <= 28 - jcp.with_bias
            && jcp.stride_w == 4
            && tr_ld / jcp.simd_w <= 4 /* [bwd_w:tr_src:r1] */
            && IMPLICATION(jcp.with_bias, kh_step_rem == 1) /* [bwd_w:b:r1] */
            && IMPLICATION(diff_weights_d.format_kind() != format_kind::any,
                    current_wei_tag == wei_4fma_tag);

        if (use_4fma) {
            jcp.ver = ver_4fma;
            jcp.kh_step = kh_step;
            jcp.tr_ld = tr_ld;
            jcp.ic_block = 1;
            if (diff_weights_d.format_kind() == format_kind::any)
                CHECK(memory_desc_init_by_tag(diff_weights_md, wei_4fma_tag));
            jcp.wei_tag = wei_4fma_tag;
        } else {
            jcp.ver = ver_fma;
            jcp.ic_block = jcp.ic;

            wei_tag = with_groups
                ? pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o);

            if (diff_weights_d.format_kind() == format_kind::any) {
                CHECK(memory_desc_init_by_tag(diff_weights_md, wei_tag));
                jcp.wei_tag = wei_tag;
            } else {
                jcp.wei_tag = diff_weights_d.matches_one_of_tag(wei_tag);
            }
            if (jcp.wei_tag != wei_tag)
                return status::unimplemented;
        }

        jcp.nb_ic = jcp.ic / jcp.ic_block;
    } else {
        auto src_tag = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
        if (src_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, src_tag));
            jcp.src_tag = src_tag;
        } else {
            jcp.src_tag = src_d.matches_one_of_tag(src_tag);
        }
        if (jcp.src_tag != src_tag)
            return status::unimplemented;

        if (diff_weights_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(diff_weights_md, wei_tag));
            jcp.wei_tag = wei_tag;
        } else {
            jcp.wei_tag = diff_weights_d.matches_one_of_tag(wei_tag);
        }
        if (jcp.wei_tag != wei_tag)
            return status::unimplemented;

        jcp.ic_block = jcp.simd_w;
        if (ok_to_pad_channels)
            jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        jcp.nb_ic = jcp.ic / jcp.ic_block;
        if ((mayiuse(avx512_mic) || mayiuse(avx512_core))
                && utils::everyone_is(data_type::f32,
                    src_d.data_type(), diff_weights_d.data_type(),
                    diff_dst_d.data_type())) {
            jcp.ver = ver_fma;
            if (one_of(ndims, 3, 4) && mayiuse(avx512_mic_4ops) && jcp.stride_w == 1 &&
                    everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w) &&
                    mkldnn_thr_syncable()) {
                jcp.ver = ver_4fma;
            }
        } else {
            return status::unimplemented;
        }
        if (jcp.ver == ver_4fma) {
            jcp.ur_w = jcp.ow;
            // XXX, BUGBUGBUG, but not a FIXME: this assumes that it's OK to
            // cross the right boundary. The only requirement is not to have
            // NaNs there because another multiplicand is always guaranteed to
            // be zero. This also may require the top-level driver to allocate
            // four extra guarding elements at the very end of the buffer.
            // I'm not proud of this hack, but it improves performance by
            // about 5-10% depending on the dimensions (Roma)

            const int tr_round = 4;

            jcp.tr_iw = rnd_up(jcp.iw + jcp.kw - 1, tr_round);
            jcp.tr_src_num_guard_elems = tr_round; // upper bound
        }
    }

    if (utils::one_of(jcp.ver, ver_4fma, ver_fma)) {
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
    } else
        return status::unimplemented;

    bool args_ok = true
        && jcp.ic % jcp.ic_block == 0
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic <= src_d.padded_dims()[1]
        && jcp.oc <= diff_dst_d.padded_dims()[1]
        && jcp.ic <= diff_weights_d.padded_dims()[with_groups + 1]
        && jcp.oc <= diff_weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    {   // balancing
        int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
        balance(jcp, nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b);
        jcp.nthr = nthr;
        jcp.nthr_mb = nthr_mb;
        jcp.nthr_g = nthr_g;
        jcp.nthr_oc_b = nthr_oc_b;
        jcp.nthr_ic_b = nthr_ic_b;
    }

    return status::success;
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.ver == ver_4fma) {
        if (jcp.is_1stconv) {
            const size_t tr_src_size =
                jcp.nthr / jcp.nthr_oc_b * jcp.ih * jcp.stride_w * jcp.tr_ld;
            scratchpad.book(key_conv_tr_src, jcp.typesize_in * tr_src_size);
        } else {
            // XXX: See the comment about tr_iw and guarding elements in
            // jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf()
            const size_t max_nthr = jcp.nthr_mb * jcp.ngroups * jcp.nb_ic;
            const size_t min_tr_src_size_per_thr
                = jcp.ih * jcp.ic_block * jcp.tr_iw;
            const size_t tr_src_size = max_nthr * min_tr_src_size_per_thr
                + jcp.tr_src_num_guard_elems;
            scratchpad.book(key_conv_tr_src, jcp.typesize_in * tr_src_size);
        }

        /* prepare synchronization contexts */
        if (jcp.nthr_oc_b > 1) {
            const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
            scratchpad.book(key_conv_tr_src_bctx,
                    sizeof(simple_barrier::ctx_t) * tr_src_bctx_size);
        }
    }

    if (jcp.nthr_mb > 1) {
        const int wei_size = jcp.ngroups * jcp.oc * jcp.ic
            * jcp.kh * jcp.kw * jcp.kd;
        const int bia_size = jcp.ngroups * jcp.oc;
        const size_t wei_bia_reduction_size = wei_size + bia_size;

        scratchpad.book(key_conv_wei_bia_reduction,
                jcp.typesize_out * wei_bia_reduction_size * (jcp.nthr_mb - 1));
        scratchpad.book(key_conv_wei_bia_reduction_bctx,
                sizeof(simple_barrier::ctx_t));
    }

    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, jcp.typesize_out * jcp.oc);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32::balance(
        const jit_conv_conf_t &j, int &nthr_, int &nthr_mb_, int &nthr_g_,
        int &nthr_oc_b_, int &nthr_ic_b_)
{
    nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

    const int max_threads = mkldnn_get_max_threads();

    if (max_threads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }

    if (!mkldnn_thr_syncable() && j.ver == ver_4fma) {
        // should not happen -- the driver is not ready
        // for TBB-like non-synchronous threading yet
        return;
    }

    if (j.ver == ver_4fma && j.is_1stconv) {
        nthr_g_ = 1;
        nthr_oc_b_ = 1;
        nthr_ic_b_ = nstl::min(j.nb_ic, max_threads);
        nthr_mb_ = nstl::min(max_threads / nthr_ic_b_, j.mb);
        nthr_ = nthr_mb_ * nthr_oc_b_ * nthr_ic_b_ * nthr_g_;
        return;
    }

    nthr_g_ = j.ngroups;
    const int nthr = max_threads / nthr_g_;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level optimizer
         * tries to minimize memory consumption. few notes:
         *  (n1) unclear why, but that essentially helps first convolution...
         *  (n2) assuming the reduction over minibatch is always there:
         *    - instead of 8 it should be 5 here (write ~= 2 read):
         *      kernel: temporal workspace 1 write
         *      reduction: 1 read from workspace and 1 write to the diff_wei
         *    - but experiments showed 8 works better than 5 or 6... */

        const int src_coef = j.ver == ver_4fma ? 4 : 1;
        const int dst_coef = 1;
        const int wei_coef = 8;

        return 0
            + src_coef
            * div_up(j.mb, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_ic, nthr_ic_b) * j.ic_block * j.ih * j.iw * j.id
            / j.stride_d / j.stride_h / j.stride_w /* (n1) */
            + dst_coef
            * div_up(j.mb, nthr_mb) * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * j.oc_block * j.oh * j.ow * j.od
            + wei_coef /* (n2) */
            * div_up(j.ngroups, nthr_g_)
            * div_up(j.nb_oc, nthr_oc_b) * div_up(j.nb_ic, nthr_ic_b)
            * j.kh * j.kw * j.kd * j.ic_block * j.oc_block;
    };

    int best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, j.mb * j.od);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);

            int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }

        if (!mkldnn_thr_syncable()) { assert(nthr_mb == 1); break; }
    }

    if (!mayiuse(avx512_mic)) {
        auto calc_comp_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
            return 1
                * div_up(j.mb, nthr_mb)
                * div_up(j.ngroups, nthr_g_)
                * div_up(j.nb_oc, nthr_oc_b)
                * div_up(j.nb_ic, nthr_ic_b);
        };

        /* step 2: search for a thread distribution with lower compute cost.
         * the constrains:
         *  - memory cost cannot exceed 110% of the best found in the step 1
         *  - unless compute cost is 133% lower than the current best case
         * note: both constants were found empirically */
        int best_comp_cost = calc_comp_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);
        for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
            const int nthr_par = nthr / nthr_mb;
            const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
            for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
                int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);
                int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
                int comp_cost = calc_comp_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

                const bool opt1 = comp_cost <= best_comp_cost
                    && mem_cost < 1.1 * best_mem_cost;
                const bool opt2 = 4 * comp_cost <= 3 * best_comp_cost;

                if (opt1 || opt2) {
                    best_comp_cost = comp_cost;
                    nthr_mb_ = nthr_mb;
                    nthr_oc_b_ = nthr_oc_b;
                    nthr_ic_b_ = nthr_ic_b;
                }
            }

            if (!mkldnn_thr_syncable()) { assert(nthr_mb == 1); break; }
        }
    }

    if (nthr_mb_ > max_threads/2 && nthr_mb_ < max_threads)
        nthr_mb_ = nstl::min(j.mb * j.od, max_threads);
    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;

    assert(nthr_ <= max_threads);
    assert(IMPLICATION(!mkldnn_thr_syncable(), nthr_mb_ == 1));
}

template struct  _jit_avx512_common_conv_fwd_kernel<Zmm>;
template struct  _jit_avx512_common_conv_fwd_kernel<Xmm>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
