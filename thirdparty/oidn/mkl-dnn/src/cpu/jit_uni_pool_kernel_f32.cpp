/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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
#include "utils.hpp"
#include "cpu_pooling_pd.hpp"

#include "jit_uni_pool_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;
using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_pool_call_s, field)

template <cpu_isa_t isa>
status_t jit_uni_pool_kernel_f32<isa>::init_conf(jit_pool_conf_t &jpp,
        const pooling_pd_t *ppd) {
    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(
            ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
            ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());

    bool args_ok = true
        && mayiuse(isa)
        && utils::one_of(pd.alg_kind, pooling_max,
                pooling_avg_include_padding,
                pooling_avg_exclude_padding);
    if (!args_ok) return status::unimplemented;

    const int simd_w = isa == avx512_common ? 16 : 8;
    const int ndims = src_d.ndims();

    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];

    jpp.c = utils::rnd_up(src_d.dims()[1], simd_w);
    if (jpp.c > src_d.padded_dims()[1])
        return status::unimplemented;

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = src_d.dims()[ndims-2];
    jpp.iw = src_d.dims()[ndims-1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.oh = dst_d.dims()[ndims-2];
    jpp.ow = dst_d.dims()[ndims-1];

    jpp.stride_d = (ndims == 5 ) ? pd.strides[0] : 1;
    jpp.stride_h = pd.strides[ndims-4];
    jpp.stride_w = pd.strides[ndims-3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = pd.kernel[ndims-4];
    jpp.kw = pd.kernel[ndims-3];

    jpp.f_pad = (ndims == 5 ) ? pd.padding[0][0] : 0;
    jpp.t_pad = pd.padding[0][ndims-4];
    jpp.l_pad = pd.padding[0][ndims-3];

    jpp.alg = pd.alg_kind;

    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;
    jpp.ind_dt = ppd->workspace_md()
        ? ppd->workspace_md()->data_type : data_type::undef;

    jpp.simple_alg = jpp.is_training
        || IMPLICATION(jpp.is_backward, jpp.kd <= jpp.stride_d);

    jpp.c_block = simd_w;

    jpp.nb_c = jpp.c / jpp.c_block;
    if (jpp.alg == pooling_max) {
        jpp.ur_w = isa == avx512_common ? 16 : 4;
        if (jpp.is_training)
            jpp.ur_w = isa == avx512_common ? 9 : 3;
        else if (jpp.is_backward)
            jpp.ur_w = isa == avx512_common ? 6 : 3;
    } else {
        if (jpp.is_backward)
            jpp.ur_w = isa == avx512_common ? 12 : 6;
        else
            jpp.ur_w = isa == avx512_common ? 24 : 12;
    }
    if (jpp.ow < jpp.ur_w) jpp.ur_w = jpp.ow;
    if (jpp.l_pad > jpp.ur_w) return status::unimplemented;

    jpp.ur_w_tail = jpp.ow % jpp.ur_w;

    return status::success;
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel_f32<isa>::maybe_recalculate_divisor(int jj,
        int ur_w, int pad_l, int pad_r) {
    if (jpp.alg == pooling_avg_exclude_padding) {
        int kw = jpp.kw;
        int stride_w = jpp.stride_w;

        int non_zero_kw = kw;
        non_zero_kw -= nstl::max(0, pad_l - jj*stride_w);
        non_zero_kw -= nstl::max(0, pad_r - (ur_w - 1 - jj)*stride_w);

        if (non_zero_kw != prev_kw) {
            mov(tmp_gpr, float2int((float)non_zero_kw));
            movq(xmm_tmp, tmp_gpr);
            uni_vbroadcastss(vmm_tmp, xmm_tmp);
            uni_vmulps(vmm_tmp, vmm_tmp, vmm_ker_area_h);
            prev_kw = non_zero_kw;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel_f32<isa>::avg_step(int ur_w, int pad_l,
        int pad_r) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    Label kd_label, kh_label;

    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward) {
            uni_vmovups(vreg(jj), ptr[reg_output + sizeof(float)*jj*c_block]);
            maybe_recalculate_divisor(jj, ur_w, pad_l, pad_r);
            uni_vdivps(vreg(jj), vreg(jj), vmm_tmp);
        } else {
            uni_vpxor(vreg(jj), vreg(jj), vreg(jj));
        }
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }

    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end = ur_w
                - utils::div_up(nstl::max(0, ki + pad_r - (kw-1)), stride_w);
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                if (jpp.is_backward) {
                    uni_vmovups(vreg(ur_w+jj),
                                ptr[aux_reg_input + input_offset]);
                    uni_vaddps(vreg(ur_w+jj), vreg(ur_w+jj), vreg(jj));
                    uni_vmovups(vmmword[aux_reg_input + input_offset],
                                vreg(ur_w+jj));
                } else {
                    uni_vaddps(vreg(jj), vreg(jj),
                               ptr[aux_reg_input + input_offset]);
                }
            }
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    if (jpp.simple_alg && jpp.ndims == 5)
    {
        add(aux_reg_input_d,  sizeof(float) * jpp.ih * iw * c_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        pop(reg_output);
        pop(reg_input);
    }

    if (!jpp.is_backward) {
        for (int jj = 0; jj < ur_w; jj++) {
            maybe_recalculate_divisor(jj, ur_w, pad_l, pad_r);
            uni_vdivps(vreg(jj), vreg(jj), vmm_tmp);
            uni_vmovups(vmmword[reg_output + sizeof(float)*jj*c_block],
                        vreg(jj));
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel_f32<isa>::max_step_fwd(int ur_w, int pad_l,
        int pad_r) {
    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    Label kd_label, kh_label;

    mov(tmp_gpr, float2int(nstl::numeric_limits<float>::lowest()));
    movq(xmm_tmp, tmp_gpr);
    uni_vbroadcastss(vmm_tmp, xmm_tmp);

    for (int jj = 0; jj < ur_w; jj++) {
        uni_vmovups(vreg(jj), vmm_tmp);
        if (jpp.is_training)
            uni_vpxor(vreg(2*ur_w+jj), vreg(2*ur_w+jj), vreg(2*ur_w+jj));
    }
    if (jpp.is_training)
    {
        movq(xmm_tmp, reg_k_shift);
        uni_vpbroadcastd(vmm_k_offset, xmm_tmp);
    }

    if (jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }
    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end = ur_w
                - utils::div_up(nstl::max(0, ki + pad_r - (kw-1)), stride_w);
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                uni_vmovups(vreg(ur_w+jj), ptr[aux_reg_input + input_offset]);
                if (isa == sse42) {
                    movups(vmm_mask, vreg(jj));
                    cmpps(vmm_mask, vreg(ur_w+jj), _cmp_lt_os);
                    blendvps(vreg(jj), vreg(ur_w+jj));
                    if (jpp.is_training)
                        blendvps(vreg(2*ur_w+jj), vmm_k_offset);
                } else if (isa == avx) {
                    vcmpps(vreg(3*ur_w+jj), vreg(jj), vreg(ur_w+jj),
                           _cmp_lt_os);
                    vblendvps(vreg(jj), vreg(jj), vreg(ur_w+jj),
                              vreg(3*ur_w+jj));
                    if (jpp.is_training)
                        vblendvps(vreg(2*ur_w+jj), vreg(2*ur_w+jj),
                                  vmm_k_offset, vreg(3*ur_w+jj));
                } else {
                    vcmpps(k_store_mask, vreg(jj), vreg(ur_w+jj), _cmp_lt_os);
                    vblendmps(vreg(jj) | k_store_mask, vreg(jj), vreg(ur_w+jj));
                    if (jpp.is_training)
                        vblendmps(vreg(2*ur_w+jj) | k_store_mask,
                                  vreg(2*ur_w+jj), vmm_k_offset);
                }
            }
            if (jpp.is_training) {
                if (isa == avx && !mayiuse(avx2)) {
                    avx_vpadd1(vmm_k_offset, vmm_one, xmm_tmp);
                } else {
                    uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
                }
            }
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    if (jpp.ndims == 5)
    {
        add(aux_reg_input_d,  sizeof(float) * jpp.ih * iw * c_block);
        if (jpp.is_training) {
            mov(tmp_gpr, ptr[reg_param + GET_OFF(kd_padding_shift)]);
            movq(xmm_tmp, tmp_gpr);
            uni_vpbroadcastd(vmm_tmp, xmm_tmp);
            if (isa == avx && !mayiuse(avx2)) {
                Xmm t(vmm_mask.getIdx());
                avx_vpadd1(vmm_k_offset, xmm_tmp, t);
            } else {
                uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
            }
        }

        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        pop(reg_output);
        pop(reg_input);
    }

    for (int jj = 0; jj < ur_w; jj++) {
        uni_vmovups(vmmword[reg_output + sizeof(float)*jj*c_block], vreg(jj));
        if (jpp.is_training) {
            const size_t step_index
                = jj * c_block * types::data_type_size(jpp.ind_dt);

            auto x = xreg(2 * ur_w + jj);
            if (jpp.ind_dt == data_type::u8) {
                if (isa == sse42) {
                    for (int i = 0; i < 4; ++i)
                        pextrb(ptr[reg_index + step_index + i], x, 4*i);
                } else if (isa == avx) {
                    auto y = yreg(2 * ur_w + jj);
                    if (jj == 0) {
                        movd(xmm_tmp, reg_shuf_mask);
                        uni_vpbroadcastd(vmm_tmp, xmm_tmp);
                    }
                    if (mayiuse(avx2)) {
                        vpshufb(y, y, vmm_tmp);
                        movd(ptr[reg_index + step_index], x);
                        vperm2i128(y, y, y, 0x1u);
                        movd(ptr[reg_index + step_index + 4], x);
                    } else {
                        Xmm t(vmm_mask.getIdx());
                        vextractf128(t, y, 0);
                        vpshufb(t, t, xmm_tmp);
                        movd(ptr[reg_index + step_index], t);
                        vextractf128(t, y, 1);
                        vpshufb(t, t, xmm_tmp); // ymm_tmp[:128]==ymm_tmp[127:0]
                        movd(ptr[reg_index + step_index + 4], t);
                    }
                } else {
                    auto v = vreg(2 * ur_w + jj);
                    vpmovusdb(x, v);
                    vmovups(ptr[reg_index + step_index], v | k_index_mask);
                }
            } else {
                uni_vmovups(ptr[reg_index + step_index], vreg(2*ur_w+jj));
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel_f32<isa>::max_step_bwd(int ur_w, int pad_l,
        int pad_r) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    Label kd_label, kh_label;

    for (int jj = 0; jj < ur_w; jj++) {
        uni_vmovups(vreg(jj), ptr[reg_output + sizeof(float)*jj*c_block]);

        const size_t step_index
            = jj * c_block * types::data_type_size(jpp.ind_dt);
        if (jpp.ind_dt == data_type::u8) {
            if (isa == sse42) {
                movd(xreg(ur_w+jj), ptr[reg_index + step_index]);
                pmovzxbd(vreg(ur_w+jj), xreg(ur_w+jj));
            } else if (isa == avx) {
                movq(xreg(ur_w+jj), ptr[reg_index + step_index]);
                if (!mayiuse(avx2)) {
                    avx_pmovzxbd(vreg(ur_w+jj), xreg(ur_w+jj), xmm_tmp);
                } else {
                    vpmovzxbd(vreg(ur_w+jj), xreg(ur_w+jj));
                }
            } else {
                vmovups(vreg(ur_w+jj) | k_index_mask,
                        ptr[reg_index + step_index]);
                vpmovzxbd(vreg(ur_w+jj), xreg(ur_w+jj));
            }
        } else {
            uni_vmovups(vreg(ur_w+jj), ptr[reg_index + step_index]);
        }
    }
    movq(xmm_tmp, reg_k_shift);
    uni_vpbroadcastd(vmm_k_offset, xmm_tmp);

    if (jpp.simple_alg && jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        if (isa == sse42) {
            // Save rdi since it is used in maskmovdqu
            assert(dst_ptr == rdi);
            push(dst_ptr);
        }
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        mov(reg_kd_pad_shift, ptr[reg_param + GET_OFF(kd_padding_shift)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }

    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end = ur_w
                - utils::div_up(nstl::max(0, ki + pad_r - (kw-1)), stride_w);
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                uni_vmovups(vreg(2*ur_w+jj), ptr[aux_reg_input + input_offset]);
                if (isa == sse42) {
                    mov(dst_ptr, aux_reg_input);
                    add(dst_ptr, input_offset);

                    movups(vreg(3*ur_w+jj), vreg(ur_w+jj));
                    pcmpeqd(vreg(3*ur_w+jj), vmm_k_offset);
                    addps(vreg(2*ur_w+jj), vreg(jj));
                    maskmovdqu(vreg(2*ur_w+jj), vreg(3*ur_w+jj));
                } else if (isa == avx) {
                    if (mayiuse(avx2)) {
                        vpcmpeqd(vreg(3*ur_w+jj), vreg(ur_w+jj), vmm_k_offset);
                    } else {
                        avx_pcmpeqd(vreg(3*ur_w+jj), vreg(ur_w+jj), vmm_k_offset, xmm_tmp);
                    }
                    vaddps(vreg(2*ur_w+jj), vreg(2*ur_w+jj), vreg(jj));
                    vmaskmovps(vmmword[aux_reg_input + input_offset],
                            vreg(3*ur_w+jj), vreg(2*ur_w+jj));
                } else {
                    vpcmpeqd(k_store_mask, vreg(ur_w+jj), vmm_k_offset);
                    vblendmps(vmm_tmp | k_store_mask | T_z, vreg(jj), vreg(jj));
                    vaddps(vreg(2*ur_w+jj), vreg(2*ur_w+jj), vmm_tmp);
                    vmovups(vmmword[aux_reg_input +
                        sizeof(float)*aux_input_offset], vreg(2*ur_w+jj));
                }
            }
            if (isa == avx && !mayiuse(avx2)) {
                avx_vpadd1(vmm_k_offset, vmm_one, xmm_tmp);
            } else {
                uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
            }
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }
    if (jpp.simple_alg && jpp.ndims == 5)
    {
        add(aux_reg_input_d,  sizeof(float) * jpp.ih * iw * c_block);

        mov(tmp_gpr, reg_kd_pad_shift);
        movq(xmm_tmp, tmp_gpr);
        uni_vpbroadcastd(vmm_tmp, xmm_tmp);
        if (isa == avx && !mayiuse(avx2)) {
            Xmm t(vmm_mask.getIdx());
            avx_vpadd1(vmm_k_offset, vmm_tmp, t);
        } else {
            uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
        }

        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        if (isa == sse42) {
            // Save rdi since it is used in maskmovdqu
            assert(dst_ptr == rdi);
            pop(dst_ptr);
        }
        pop(reg_output);
        pop(reg_input);
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_f32<isa>::maybe_zero_diff_src() {
    assert(jpp.c_block * sizeof(float) % cpu_isa_traits<isa>::vlen == 0);
    Label l_skip, l_zero;

    auto reg_oh = tmp_gpr;
    mov(reg_oh, ptr[reg_param + GET_OFF(oh)]);
    cmp(reg_oh, 0);
    jz(l_skip, T_NEAR);

    if (jpp.ndims == 5) {
        mov(zero_size, ptr[reg_param + GET_OFF(oh)]);
        mov(tmp_gpr, jpp.ih * jpp.iw * jpp.c_block * sizeof(float));
        imul(zero_size, tmp_gpr);
    }

    auto vzero = vmm_tmp;
    uni_vpxor(vzero, vzero, vzero);

    auto reg_off = tmp_gpr;
    xor_(reg_off, reg_off);

    L(l_zero);
    {
        const int dim = jpp.iw * jpp.c_block * sizeof(float);
        for (int i = 0; i < dim; i += cpu_isa_traits<isa>::vlen)
            uni_vmovups(ptr[reg_input + reg_off + i], vzero);
        add(reg_off, dim);
        if (jpp.ndims == 5) cmp(reg_off, zero_size);
        else cmp(reg_off, jpp.ih * dim);
        jl(l_zero, T_NEAR);
    }

    L(l_skip);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel_f32<isa>::generate() {

    this->preamble();

    int ow = jpp.ow;
    int iw = jpp.iw;
    int kw = jpp.kw;
    int kh = jpp.kh;
    int ur_w = jpp.ur_w;
    int c_block = jpp.c_block;
    int stride_w = jpp.stride_w;
    int l_pad = jpp.l_pad;
    int ur_w_tail = jpp.ur_w_tail;

    int n_oi = ow / ur_w;

    prev_kw = 0;

    int vlen = cpu_isa_traits<isa>::vlen;

#if defined(_WIN32)
    // Always mimic the Unix ABI (see the note about maskmovdqu in the header
    // file).
    xor_(rdi, rcx);
    xor_(rcx, rdi);
    xor_(rdi, rcx);
#endif

    mov(reg_input, ptr[reg_param + GET_OFF(src)]);
    mov(reg_output, ptr[reg_param + GET_OFF(dst)]);
    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
        mov(reg_index, ptr[reg_param + GET_OFF(indices)]);
    mov(reg_kh, ptr[reg_param + GET_OFF(kh_padding)]);
    mov(reg_k_shift, ptr[reg_param + GET_OFF(kh_padding_shift)]);
    mov(reg_ker_area_h, ptr[reg_param + GET_OFF(ker_area_h)]);

    if (jpp.is_backward)
        maybe_zero_diff_src();

    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
        mov(tmp_gpr, 1);
        movq(xmm_one, tmp_gpr);
        uni_vpbroadcastd(vmm_one, xmm_one);

        if (isa == avx) {
            mov(reg_shuf_mask, 0x0c080400);
        } else if (isa >= avx512_common) {
            mov(tmp_gpr.cvt32(), 0x000f);
            kmovw(k_index_mask, tmp_gpr.cvt32());
        }
    }

    int r_pad  = nstl::max(0, ((ow-1)*stride_w) + kw - 1 - (iw + l_pad - 1));
    int r_pad1 = (ur_w*n_oi - 1)*stride_w + kw - 1 - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    if (jpp.alg == pooling_avg_exclude_padding) {
        movq(xmm_ker_area_h, reg_ker_area_h);
        uni_vpbroadcastd(vmm_ker_area_h, xmm_ker_area_h);
    }

    if (jpp.alg == pooling_avg_include_padding) {
        mov(tmp_gpr, float2int((float)(kw * kh * jpp.kd)));
        movq(xmm_tmp, tmp_gpr);
        uni_vpbroadcastd(vmm_tmp, xmm_tmp);
    }
    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0) {
            step(ur_w, l_pad, r_pad1);
        } else  {
            step(ur_w, l_pad, 0);
        }

        if (isa == sse42) {
            if (n_oi < 0 && r_pad1 > 0) {
                step_high_half(ur_w, l_pad, r_pad1);
            } else  {
                step_high_half(ur_w, l_pad, 0);
            }
        }

        if (isa == sse42) {
            add(reg_input, sizeof(float)*(ur_w*stride_w-l_pad)*c_block - vlen);
            add(reg_output, sizeof(float)*ur_w*c_block - vlen);
            if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
                add(reg_index, (2 * ur_w - 1) * c_block / 2
                        * types::data_type_size(jpp.ind_dt));
        } else {
            add(reg_input, sizeof(float)*(ur_w*stride_w - l_pad)*c_block);
            add(reg_output, sizeof(float)*ur_w*c_block);
            if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
                add(reg_index, ur_w * c_block
                        * types::data_type_size(jpp.ind_dt));
        }
    }

    xor_(oi_iter, oi_iter);
    if (n_oi > 0) {
        Label ow_loop;
        L(ow_loop); {
            step(ur_w, 0, 0);

            if (isa == sse42) {
                step_high_half(ur_w, 0, 0);
            }

            if (isa == sse42) {
                add(reg_input, sizeof(float)*ur_w*stride_w*c_block - vlen);
                add(reg_output, sizeof(float)*ur_w*c_block - vlen);
                if (jpp.alg == pooling_max &&
                    (jpp.is_training || jpp.is_backward))
                    add(reg_index, (2 * ur_w - 1) * c_block / 2
                            * types::data_type_size(jpp.ind_dt));
            } else {
                add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
                add(reg_output, sizeof(float)*ur_w*c_block);
                if (jpp.alg == pooling_max &&
                    (jpp.is_training || jpp.is_backward))
                    add(reg_index, ur_w * c_block
                            * types::data_type_size(jpp.ind_dt));
            }

            inc(oi_iter);
            cmp(oi_iter, n_oi);
            jl(ow_loop, T_NEAR);
        }
    }

    if (r_pad1 > 0 && n_oi >= 0) {
        step(ur_w, 0, r_pad1);

        if (isa == sse42) {
            step_high_half(ur_w, 0, r_pad1);
        }

        if (isa == sse42) {
            add(reg_input, sizeof(float)*ur_w*stride_w*c_block - vlen);
            add(reg_output, sizeof(float)*ur_w*c_block - vlen);
            if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
                add(reg_index, (2 * ur_w - 1) * c_block / 2
                        * types::data_type_size(jpp.ind_dt));
        } else {
            add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
            add(reg_output, sizeof(float)*ur_w*c_block);
            if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
                add(reg_index, ur_w * c_block
                        * types::data_type_size(jpp.ind_dt));
        }
    }

    if (ur_w_tail != 0) {
        step(ur_w_tail, 0, r_pad);

        if (isa == sse42) {
            step_high_half(ur_w_tail, 0, r_pad);
        }
    }

    this->postamble();
}

template struct jit_uni_pool_kernel_f32<sse42>;
template struct jit_uni_pool_kernel_f32<avx>; // implements both <avx> and <avx2>
template struct jit_uni_pool_kernel_f32<avx512_common>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
