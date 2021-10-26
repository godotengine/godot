/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "jit_uni_eltwise.hpp"

#define GET_OFF(field) offsetof(jit_args, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble(size_t start_idx,
        size_t end_idx) {
    preserved_vecs_count = 0;
    vecs_to_preserve = (size_t)aux_vecs_count(alg_);
    start_idx_tail = start_idx;

    // For sse42 mask register has to be Xmm(0)
    if (isa == sse42 && vecs_to_preserve > 0) {
        size_t idx = 0;
        assert(idx < start_idx);
        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    for (size_t idx = preserved_vecs_count; idx < vecs_count; idx++) {
        if (preserved_vecs_count >= vecs_to_preserve) break;
        if (start_idx <= idx && idx < end_idx) continue;

        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
        preserved_vec_idxs[preserved_vecs_count++] = start_idx_tail++;
    }

    assert(preserved_vecs_count == vecs_to_preserve);

    if (save_state_) {
        h->push(p_table);

        if (preserved_vecs_count)
            h->sub(h->rsp, preserved_vecs_count * vlen);

        for (size_t i = 0; i < preserved_vecs_count; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen],
                    Vmm(preserved_vec_idxs[i]));

        load_table_addr();
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble_tail(size_t start_idx)
{
    size_t tail_vecs_to_preserve = start_idx_tail - start_idx;
    if (tail_vecs_to_preserve == 0) return;

    const int idx_off = vecs_to_preserve - tail_vecs_to_preserve;

    if (save_state_) {
        if (idx_off)
            h->add(h->rsp, idx_off * vlen);

        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(Vmm(preserved_vec_idxs[idx_off + i]),
                    h->ptr[h->rsp + i * vlen]);
    }

    for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
        preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;

    if (save_state_) {
        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen],
                    Vmm(preserved_vec_idxs[idx_off + i]));

        if (idx_off)
            h->sub(h->rsp, idx_off * vlen);
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_postamble() {
    if (!save_state_) return;

    for (size_t i = 0; i < preserved_vecs_count; ++i)
        h->uni_vmovups(Vmm(preserved_vec_idxs[i]),
                h->ptr[h->rsp + i * vlen]);

    if (preserved_vecs_count)
        h->add(h->rsp, preserved_vecs_count * vlen);

    h->pop(p_table);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::assign_regs() {
    vmm_mask = Vmm(preserved_vec_idxs[0]);
    vmm_aux0 = Vmm(preserved_vec_idxs[0]);
    vmm_aux1 = Vmm(preserved_vec_idxs[1]);
    vmm_aux2 = Vmm(preserved_vec_idxs[2]);
    vmm_aux3 = Vmm(preserved_vec_idxs[3]);
    vmm_aux4 = Vmm(preserved_vec_idxs[4]);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector(const Vmm &vmm_src) {
    h->uni_vminps(vmm_src, vmm_src, table_val(10));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(11));
    h->uni_vmovups(vmm_aux0, vmm_src);
    //calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(2));
    h->uni_vaddps(vmm_src, vmm_src, table_val(1));

    // tmp = floorf(fx)
    if (isa == avx512_common) {
        h->vcvtps2dq(vmm_aux1 | h->T_rd_sae, vmm_src);
        h->vcvtdq2ps(vmm_aux1, vmm_aux1);

        h->vcmpps(k_mask, vmm_aux1, vmm_src, _cmp_nle_us);
        h->vmovups(vmm_aux3 | k_mask | h->T_z, table_val(0));

        h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_aux3);
    } else {
        h->uni_vroundps(vmm_aux1, vmm_src, _op_floor);
    }

    //keep fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux1); //vmm_src = fx

    //x = x - fx * ln2
    h->uni_vfnmadd231ps(vmm_aux0, vmm_aux1, table_val(3));

    // compute 2^n
    h->uni_vcvtps2dq(vmm_aux1, vmm_src);
    h->uni_vpaddd(vmm_aux1, vmm_aux1, table_val(4));
    h->uni_vpslld(vmm_aux1, vmm_aux1, 23); //Vmm(6) = 2^-fx

    // y = p5
    h->uni_vmovups(vmm_src, table_val(9));
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(8));
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(7));
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(6));
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(0));
    // y = y * x + p0
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(5));  //exp(q)
    // y = y * 2^n
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector(const Vmm &vmm_src)
{
    const int alpha_off = 0, zero_off = 1;

    h->uni_vmovups(vmm_aux1, vmm_src);
    if (isa == sse42) {
        h->movups(vmm_mask, vmm_src);
        h->mulps(vmm_src, table_val(alpha_off));
        h->cmpps(vmm_mask, table_val(zero_off), _cmp_nle_us);
        h->blendvps(vmm_src, vmm_aux1);
    } else if (isa == avx2) {
        h->vmulps(vmm_src, vmm_src, table_val(alpha_off));
        h->vcmpgtps(vmm_mask, vmm_aux1, table_val(zero_off));
        h->vblendvps(vmm_src, vmm_src, vmm_aux1, vmm_mask);
    } else if (isa == avx512_common) {
        h->vmulps(vmm_src, vmm_src, table_val(alpha_off));
        h->vcmpps(k_mask, vmm_aux1, table_val(zero_off), _cmp_nle_us);
        h->vblendmps(vmm_src | k_mask, vmm_src, vmm_aux1);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector(
        const Vmm &vmm_src) {
    const int zero_off = 1;
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero_off));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector(const Vmm &vmm_src) {
    const int alpha_off = 23, zero_off = 24;

    // compute exponent
    h->uni_vmovups(vmm_aux2, vmm_src);
    exp_compute_vector(vmm_src);

    // alpha * (exp(x) - 1)
    h->uni_vsubps(vmm_src, vmm_src, table_val(0));
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha_off));

    // combine with mask
    if (isa == sse42) {
        h->pxor(vmm_mask, vmm_mask);
        h->cmpps(vmm_mask,  vmm_aux2, _cmp_le_os);
        h->blendvps(vmm_src, vmm_aux2);
    } else if (isa == avx2) {
        h->uni_vcmpgtps(vmm_mask, vmm_aux2, table_val(zero_off));
        h->uni_vblendvps(vmm_src, vmm_src, vmm_aux2, vmm_mask);
    } else if (isa == avx512_common) {
        h->vcmpps(k_mask, vmm_aux2, table_val(zero_off), _cmp_nle_us);
        h->vblendmps(vmm_src | k_mask, vmm_src, vmm_aux2);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector(const Vmm &vmm_src)
{
    // # comes from Taylor expansion error bound
    //  > linear_sat_point = single(sqrt(3) * 1b-12);
    // # comes from the exp formula cancellation
    //  > exp_bound_point = (single(log(3)/2));
    // # comes from rounding accuracy in float
    //  > one_sat_point = round(atanh(1 - 1b-25), single, RU);
    //  > P = fpminimax(f, [|1, 3, 5, 7, 9|], [|24... |],
    //            [linear_sat_point, exp_bound_point], relative, floating);
    //  > err_bound = D(sup(supnorm(P, tanh(x),
    //          [linear_sat_point, exp_bound_point], relative, theta)));
    //    0x1.fffd6f00b9539p-25
    //  > P;
    //    x * (0x1.fffffep-1 + x^0x1p1 * (-0x1.55539ep-2 + x^0x1p1 *
    //        (0x1.10be3ep-3 + x^0x1p1 * (-0x1.ae57b4p-5
    //        + x^0x1p1 * 0x1.09fa1p-6))))

    // register mapping
    // vmm_src contains input
    // vmm_aux0 contains mask of currently valid results.
    //     1 is need computation, 0 is already computed
    // vmm_aux1 contains current output
    // vmm_aux2, vmm_aux3 contains auxiliary values
    // vmm_aux4 contains the original sign of inputs

    Label end_tanh_label;

    auto test_exit =[&](Xbyak::Address threshold){
        // is not necessary for >AVX, but should not matter on perf
        h->uni_vmovups(vmm_aux0, vmm_src);
        if (isa == avx512_common){
            h->vcmpps(k_mask, vmm_aux0, threshold, 0x5);
            h->kortestw(k_mask, k_mask);
        } else {
            h->uni_vcmpgeps(vmm_aux0, vmm_aux0, threshold);
            h->uni_vtestps(vmm_aux0, vmm_aux0);
        }
        h->jz(end_tanh_label, Xbyak::CodeGenerator::T_NEAR);
    };

    auto blend_results=[&](Vmm vmm_partial_res){
        if (isa == avx512_common)
            h->vblendmps(vmm_aux1 | k_mask, vmm_aux1, vmm_partial_res);
        else
            h->uni_vblendvps(vmm_aux1, vmm_aux1, vmm_partial_res, vmm_aux0);
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    // mov is not necessary for >AVX, but should not matter for performance
    h->uni_vmovups(vmm_aux4, vmm_src);
    h->uni_vandps(vmm_aux4, vmm_aux4, table_val(12));
    h->uni_vandps(vmm_src, vmm_src, table_val(17));

    // if x < linear_sat_point for all inputs, we just return the input
    h->uni_vmovups(vmm_aux1, vmm_src);
    test_exit(table_val(13));

    // if one of the mask is one, we have to compute an better approx
    h->uni_vmovups(vmm_aux2, vmm_src);
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux2);
    h->uni_vmovups(vmm_aux3, table_val(22));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(21));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(20));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(19));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(18));
    h->uni_vmulps(vmm_aux3, vmm_aux3, vmm_src);

    // we blend only the result that need update
    blend_results(vmm_aux3);

    // if x < exp_bound_point, we go to return point
    test_exit(table_val(14));

    // if not we use a better approx 1 - 2 / (1 + exp(2x))
    // compute 2x
    h->uni_vmovups(vmm_aux3, vmm_src);
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux3);

    // Compute exp(2x)
    // We need to save kmask, vmm_aux0, vmm_aux1 and vmm_src as exp can use them
    // vmm_src is not more read afterwards, so we do not have to save it
    auto stack_size = 3 * vlen + (isa == avx512_common) * 4;
    h->sub(h->rsp, stack_size);
    h->uni_vmovups(h->ptr[h->rsp + 0 * vlen], vmm_aux0);
    h->uni_vmovups(h->ptr[h->rsp + 1 * vlen], vmm_aux1);
    h->uni_vmovups(h->ptr[h->rsp + 2 * vlen], vmm_src);
    if (isa == avx512_common)
        h->kmovw(h->ptr[h->rsp + 3 * vlen], k_mask);

    exp_compute_vector(vmm_aux3);

    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp + 0 * vlen]);
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp + 1 * vlen]);
    h->uni_vmovups(vmm_src, h->ptr[h->rsp + 2 * vlen]);
    if (isa == avx512_common)
        h->kmovw(k_mask, h->ptr[h->rsp + 3 * vlen]);
    h->add(h->rsp, stack_size);

    // 1 + exp(2x)
    h->uni_vaddps(vmm_aux3, vmm_aux3, table_val(0));

    // 1 - 2 / (1 + exp(2x))
    h->uni_vmovups(vmm_aux2, table_val(16));
    h->uni_vdivps(vmm_aux2, vmm_aux2, vmm_aux3);
    h->uni_vaddps(vmm_aux2, vmm_aux2, table_val(0));

    // we blend only the result that need update
    blend_results(vmm_aux2);

    // finally, we saturate to 1 if needed
    // TODO: maybe move that up if most inputs saturate in practice
    if (isa == avx512_common)
        h->vcmpps(k_mask, vmm_aux0, table_val(15), 0x5);
    else {
        h->uni_vmovups(vmm_aux0, vmm_src);
        h->uni_vcmpgeps(vmm_aux0, vmm_aux0, table_val(15));
    }
    h->uni_vmovups(vmm_aux2, table_val(0));
    blend_results(vmm_aux2);

    h->L(end_tanh_label);
    {
        // we apply the sign of x to the result and we are done
        h->uni_vmovups(vmm_src, vmm_aux1);
        h->uni_vpxor(vmm_src, vmm_src, vmm_aux4);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector(
        const Vmm &vmm_src) {
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector(const Vmm &vmm_src) {
    // compute abs(x) = _mm_and_ps(x, 01111..111));
    h->uni_vandps(vmm_src, vmm_src, table_val(0));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector(const Vmm &vmm_src)
{
    if (isa == avx512_common) {
        h->vcmpps(k_mask, vmm_src, table_val(0), _cmp_nle_us);
        h->uni_vsqrtps(vmm_aux1, vmm_src);
        h->uni_vmovups(vmm_src, table_val(0));
        h->vblendmps(vmm_src | k_mask, vmm_src, vmm_aux1);
    } else {
        h->uni_vmovups(vmm_mask, vmm_src);
        h->uni_vcmpgtps(vmm_mask, vmm_mask, table_val(0));
        h->uni_vsqrtps(vmm_aux1, vmm_src);
        h->uni_vmovups(vmm_src, table_val(0));
        h->uni_vblendvps(vmm_src, vmm_src, vmm_aux1, vmm_mask);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector(
        const Vmm &vmm_src) {
    // compute x = alpha * x + beta;
    h->uni_vmovups(vmm_aux0, table_val(0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(1));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector(
        const Vmm &vmm_src) {
    // compute bounded relu */
    h->uni_vmaxps(vmm_src, vmm_src, table_val(1));
    h->uni_vminps(vmm_src, vmm_src, table_val(0));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector(
        const Vmm &vmm_src) {
    // duplicate src
    h->uni_vmovups(vmm_aux2, vmm_src);

    h->uni_vminps(vmm_src, vmm_src, table_val(24));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(25));
    h->uni_vmovups(vmm_aux1, vmm_src);
    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(2));
    h->uni_vaddps(vmm_src, vmm_src, table_val(1));

    // tmp = floorf(fx)
    if (isa == avx512_common) {
        h->vcvtps2dq(vmm_aux0 | h->T_rd_sae, vmm_src);
        h->vcvtdq2ps(vmm_aux0, vmm_aux0);

        h->vcmpps(k_mask, vmm_aux0, vmm_src, _cmp_nle_us);
        h->vmovups(vmm_aux3 | k_mask | h->T_z, table_val(0));

        h->vsubps(vmm_aux0, vmm_aux0, vmm_aux3);
    } else {
        h->uni_vroundps(vmm_aux0, vmm_src, _op_floor);
    }

    // keep fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux0); //vmm_src = fx
    // calculation fx * ln2
    h->uni_vmulps(vmm_aux0, vmm_aux0, table_val(3));
    // x = x - fx * ln2
    h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_aux0);
    // y = p5
    h->uni_vmovups(vmm_aux3, table_val(22));
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(21));
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(20));
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(19));
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(0));
    // y = y * x + p0
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(17));

    // compute 2^(-n)
    if (isa == avx512_common) {
        h->vmulps(vmm_aux1, vmm_src, table_val(23));
        h->vcvtps2dq(vmm_aux1, vmm_aux1);
    } else {
        h->uni_vcvtps2dq(vmm_aux1, vmm_src);
        h->uni_vpsignd(vmm_aux1, vmm_aux1, table_val(23));
    }

    h->uni_vpaddd(vmm_aux1, vmm_aux1, table_val(4));
    h->uni_vpslld(vmm_aux1, vmm_aux1, 23); //vmm_aux1 = 2^-fx
    // calculate ln(1 + y)
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux1);
    // x = y; y is free; keep x for further computations
    h->uni_vmovups(vmm_src, vmm_aux3);
    // frexp()
    h->uni_vpsrld(vmm_src, vmm_src, 23);
    h->uni_vcvtdq2ps(vmm_src, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->uni_vsubps(vmm_src, vmm_src, table_val(5));

    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(6));
    // got y. (mantisa)  0.5 < y < 1
    h->uni_vorps(vmm_aux3, vmm_aux3, table_val(7));
    // y  = y - 1
    h->uni_vsubps(vmm_aux3, vmm_aux3, table_val(0));
    // y = p8
    h->uni_vmovups(vmm_aux1, table_val(16));
    // y = y * x + p7
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(15));
    // y = y * x + p6
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(14));
    // y = y * x + p5
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(13));
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(12));
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(11));
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(10));
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(9));
    // y = y * x + p0 ; p0 = 0
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(8));
    //calculate ln(2) * n
    h->uni_vmulps(vmm_src, vmm_src, table_val(3));
    h->uni_vaddps(vmm_aux1, vmm_aux1, vmm_src);
    h->uni_vaddps(vmm_aux1, vmm_aux1, vmm_aux0);

    // get vmm_mask = src > max logf
    h->uni_vmovups(vmm_mask, vmm_aux2);
    if (isa == avx512_common) {
        // y = (x < max log f) ? soft_relu(x) : x
        h->vcmpps(k_mask, vmm_mask, table_val(24), _cmp_nle_us);
        h->vblendmps(vmm_aux1 | k_mask, vmm_aux1, vmm_aux2);
    } else {
        // y = (x < max log f) ? soft_relu(x) : x
        h->uni_vcmpgtps(vmm_mask, vmm_mask, table_val(24));
        h->uni_vblendvps(vmm_aux1, vmm_aux1, vmm_aux2, vmm_mask);
    }

    h->uni_vmovups(vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector(
        const Vmm &vmm_src) {
    // we store the original sign and make x negative
    // IMPORTANT: we assume vmm_aux0 to be xmm0, as for sse4.2 path it is required
    // IMPORTANT: we use vmm_aux2 for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux2, vmm_src);
    h->uni_vandps(vmm_aux2, vmm_aux2, table_val(12));
    h->uni_vorps(vmm_src, vmm_src, table_val(12));

    exp_compute_vector(vmm_src);
    // dup exp(x)
    h->uni_vmovups(vmm_aux1, vmm_src);
    // (exp(x) + 1)
    h->uni_vaddps(vmm_aux1, vmm_aux1, table_val(0));
    // y = exp(x) / (exp(x) + 1)
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->uni_vmovups(vmm_aux3, table_val(0));
    h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src);
    if (isa == avx512_common) {
        h->vptestmd(k_mask, vmm_aux2, vmm_aux2);
        h->vblendmps(vmm_aux3 | k_mask, vmm_aux3, vmm_src);
    } else {
        h->uni_vmovups(vmm_aux0, vmm_aux2);// The mask should be xmm0 for sse4.2
        h->uni_vblendvps(vmm_aux3, vmm_aux3, vmm_src, vmm_aux0);
    }
    h->uni_vmovups(vmm_src, vmm_aux3);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_prepare_table() {
    const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            // exp(x) polynom
            0x3f800001, // [5] p0 = 1.0000001f
            0x3efffe85, // [6] p2 = 0.4999887f
            0x3e2aaa3e, // [7] p3 = 0.16666505f
            0x3d2bb1b1, // [8] p4 = 0.041917507f
            0x3c091ec1, // [9] p5 = 0.008369149f
            0x42b0c0a5, //[10] max logf = 88.3762589f
            0xc1766666, //[11] min logf = -14.5f
            // tanh(x) constants,
            0x80000000, //[12] mask to extract sign
            0x39ddb3d7, //[13] arg below which tanh(x) = x
            0x3f0c9f54, //[14] arg below which pol approx is valid
            0x41102cb4, //[15] arg after which tanh(x) = 1
            0xc0000000, //[16] -2.0f
            0x7fffffff, //[17] mask to make positive
            // tanh pol approx
            0x3f7fffff, //[18] p0
            0xbeaaa9cf, //[19] p1
            0x3e085f1f, //[20] p2
            0xbd572bda, //[21] p3
            0x3c84fd08, //[22] p4
    };

    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(cvals[i]);
    }

    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_prepare_table() {
    const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            0x42fc0000, // [5] 126
            0x807fffff, // [6] and with (to get 0.5 * mantissa)
            0x3f000000, // [7] or with (to get 0.5 * mantissa)
            // ln(1 + x) polynomial
            0xb2b4637d, // [8]  p0 = 0.0000000244f
            0x3f7fff8e, // [9]  p1 = 0.9999976971f
            0xbf001759, //[10]  p2 = -0.5002478215f
            0x3ea70608, //[11]  p3 = 0.3272714505f
            0xbea3d7bf, //[12]  p4 = -0.3153830071f
            0xbe361d04, //[13]  p5 = -0.1701777461f
            0xbfa8f1e6, //[14]  p6 = -1.3254635147f
            0xbfe1e812, //[15]  p7 = -1.7971917960f
            0xbfc4d30e, //[16]  p8 = -1.5652673123f
            // exp(x) polynomial
            0x3f800001, //[17]  p0 = 1.0000001f
            0x3f800000, //[18]  p1 = 1.0f
            0x3efffe85, //[19]  p2 = 0.4999887f
            0x3e2aaa3e, //[20]  p3 = 0.16666505f
            0x3d2bb1b1, //[21]  p4 = 0.041917507f
            0x3c091ec1, //[22]  p5 = 0.008369149f
            0xbf800000, //[23] is required for sign changing
            0x42b0c0a5, //[24] max logf = 88.3762589f
            0xc1766666  //[25] min logf = -14.5f
    };

    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d) {
            h->dd(cvals[i]);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0x7fffffff);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(beta_));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0);
}

template <cpu_isa_t isa>
int jit_uni_eltwise_injector_f32<isa>::aux_vecs_count(alg_kind_t alg_) {
    switch (alg_) {
    case alg_kind::eltwise_relu: return (alpha_ == 0.f) ? 0 : 2;
    case alg_kind::eltwise_elu: return 4;
    case alg_kind::eltwise_tanh: return 5;
    case alg_kind::eltwise_square: return 0;
    case alg_kind::eltwise_abs: return 0;
    case alg_kind::eltwise_sqrt: return 2;
    case alg_kind::eltwise_linear: return 1;
    case alg_kind::eltwise_bounded_relu: return 0;
    case alg_kind::eltwise_soft_relu: return 4;
    case alg_kind::eltwise_logistic: return 4;
    default: assert(!"unsupported eltwise algorithm");
    }

    return 0;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(size_t start_idx,
        size_t end_idx) {
    using namespace alg_kind;
    for (size_t idx = start_idx; idx < end_idx; idx++) {
        switch (alg_) {
        case eltwise_relu:
            if (alpha_ == 0.f) relu_zero_ns_compute_vector(Vmm(idx));
            else relu_compute_vector(Vmm(idx));
            break;
        case eltwise_elu: elu_compute_vector(Vmm(idx)); break;
        case eltwise_tanh: tanh_compute_vector(Vmm(idx)); break;
        case eltwise_square: square_compute_vector(Vmm(idx)); break;
        case eltwise_abs: abs_compute_vector(Vmm(idx)); break;
        case eltwise_sqrt: sqrt_compute_vector(Vmm(idx)); break;
        case eltwise_linear: linear_compute_vector(Vmm(idx)); break;
        case eltwise_bounded_relu: bounded_relu_compute_vector(Vmm(idx)); break;
        case eltwise_soft_relu: soft_relu_compute_vector(Vmm(idx)); break;
        case eltwise_logistic: logistic_compute_vector(Vmm(idx)); break;
        default: assert(!"unsupported eltwise algorithm");
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx) {
    assert(start_idx < end_idx && end_idx <= vecs_count);

    injector_preamble(start_idx, end_idx);
    compute_body(start_idx_tail, end_idx);
    injector_preamble_tail(start_idx);
    compute_body(start_idx, start_idx_tail);
    injector_postamble();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::prepare_table(bool gen_table) {
    using namespace alg_kind;

    h->align(64);
    h->L(l_table);

    if (gen_table) {
        switch (alg_) {
        case eltwise_relu: relu_prepare_table(); break;
        case eltwise_elu:
        case eltwise_tanh:
        case eltwise_logistic:
            elu_prepare_table(); break;
        case eltwise_soft_relu: soft_relu_prepare_table(); break;
        case eltwise_abs: abs_prepare_table(); break;
        case eltwise_sqrt: sqrt_prepare_table(); break;
        case eltwise_linear: linear_prepare_table(); break;
        case eltwise_bounded_relu: bounded_relu_prepare_table(); break;
        case eltwise_square: break;
        default: assert(!"unsupported eltwise algorithm");
    }
    }
}

template struct jit_uni_eltwise_injector_f32<avx512_common>;
template struct jit_uni_eltwise_injector_f32<avx2>;
template struct jit_uni_eltwise_injector_f32<sse42>;


struct jit_args {
    const float *from;
    const float *for_comparison;
    const float *to;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel_f32 : public c_compatible {
    const eltwise_desc_t &desc_;

    void (*ker_)(const jit_args *);
    void operator()(const jit_args *args) { assert(ker_); ker_(args); }

    jit_uni_eltwise_kernel_f32(const eltwise_desc_t &desc)
        : desc_(desc), ker_(nullptr) {}
    virtual ~jit_uni_eltwise_kernel_f32() {}

protected:
    bool is_bwd() const { return desc_.prop_kind == prop_kind::backward_data; }
};

/* jit kernels */
namespace {

template <cpu_isa_t isa>
struct jit_uni_relu_kernel_f32 : public jit_uni_eltwise_kernel_f32,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_relu_kernel_f32)

    void compute_step(bool vectorize, const int uf, const int shift) {
        for (int i = 0; i < uf; i++) {
            if (vectorize) {
                uni_vmovups(Vmm(i + 1), ptr[reg_from + i * shift]);
                if (is_bwd())
                    uni_vmovups(Vmm(uf + i + 1),
                                ptr[reg_for_comparison + i * shift]);
            } else {
                movss(Xmm(i + 1), ptr[reg_from + i * shift]);
                if (is_bwd())
                    movss(Xmm(uf + i + 1),
                          ptr[reg_for_comparison + i * shift]);
            }
        }

        if (isa == sse42) {
            for (int i = 0; i < uf; i++) {
                movups(Vmm(2 * uf + i + 1), Vmm(i + 1));
                mulps(Vmm(2 * uf + i + 1), vmm_ns);

                Vmm mask = Vmm(0);
                if (is_bwd()) {
                    movups(mask, Vmm(uf + i + 1));
                    cmpps(mask, vmm_zero, _cmp_nle_us);
                } else {
                    movups(mask, Vmm(i + 1));
                    cmpps(mask, vmm_zero, _cmp_nle_us);
                }
                blendvps(Vmm(2 * uf + i + 1), Vmm(i + 1));
            }
        } else {
            for (int i = 0; i < uf; i++) {
                vmulps(Vmm(2 * uf + i + 1), Vmm(i + 1), vmm_ns);
                if (isa == avx2) {
                    if (is_bwd())
                        vcmpgtps(vmm_mask, Vmm(uf + i + 1), vmm_zero);
                    else
                        vcmpgtps(vmm_mask, Vmm(i + 1), vmm_zero);

                    vblendvps(Vmm(2 * uf + i + 1), Vmm(2 * uf + i + 1),
                              Vmm(i + 1), vmm_mask);

                } else {
                    if (is_bwd())
                        vcmpps(k_mask, Vmm(uf + i + 1), vmm_zero, _cmp_nle_us);
                    else
                        vcmpps(k_mask, Vmm(i + 1), vmm_zero, _cmp_nle_us);
                    vblendmps(Vmm(2 * uf + i + 1) | k_mask, Vmm(2 * uf + i + 1),
                              Vmm(i + 1));
                }
            }
        }

        for (int i = 0; i < uf; i++) {
            if (vectorize) {
                uni_vmovups(ptr[reg_to + i * shift], Vmm(2 * uf + i + 1));
            } else {
                movss(ptr[reg_to + i * shift], Xmm(2 * uf + i + 1));
            }
        }
    }

    jit_uni_relu_kernel_f32(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel_f32(desc), jit_generator() {
        assert(desc.alg_kind == alg_kind::eltwise_relu);
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        Reg64 param = abi_param1;

        const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
        const int loop_dec[] = {simd_w, 1};
        const int uf[] = {1, 1};
        const int shift[] = {cpu_isa_traits<isa>::vlen, sizeof(float)};
        const bool loop_vectorize[] = {true, false};

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        if (is_bwd())
            mov(reg_for_comparison, ptr[param + GET_OFF(for_comparison)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        mov(imm_addr64, float2int(desc.alpha));
        movq(xmm_ns, imm_addr64);
        uni_vbroadcastss(vmm_ns, xmm_ns);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        Label loop_label[3];

        for (int id = 0; id < 2; id++) {
            L(loop_label[id]);
            cmp(reg_work_amount, uf[id] * loop_dec[id] - 1);
            jle(loop_label[id + 1], T_NEAR);

            compute_step(loop_vectorize[id], uf[id], shift[id]);

            add(reg_from, uf[id] * shift[id]);
            add(reg_to, uf[id] * shift[id]);
            if (is_bwd())
                add(reg_for_comparison, uf[id] * shift[id]);

            sub(reg_work_amount, uf[id] * loop_dec[id]);
            jmp(loop_label[id]);
        }

        L(loop_label[2]);
        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;

    Reg64 reg_from = rax;
    Reg64 reg_for_comparison = is_bwd() ? rdx : reg_from;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;

    Xmm xmm_ns = Xmm(14);

    Vmm vmm_ns = Vmm(isa == avx512_common ? 30 : 14);
    Vmm vmm_zero = Vmm(isa == avx512_common ? 31 : 15);

    Vmm vmm_mask = Vmm(isa == avx512_common ? 28 : 12);
    Opmask k_mask = Opmask(1);
};

template <cpu_isa_t isa>
struct jit_uni_kernel_fwd_f32: public jit_uni_eltwise_kernel_f32,
    public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel_fwd_f32)

    jit_uni_kernel_fwd_f32(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel_f32(desc), jit_generator() {

        eltwise_injector_ = new jit_uni_eltwise_injector_f32<isa>(this,
                desc.alg_kind, desc.alpha, desc.beta, false, r9, Opmask(1));

        using namespace alg_kind;

        assert(is_bwd() == false);
        assert(utils::one_of(desc.alg_kind, eltwise_tanh, eltwise_elu,
                    eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                    eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic));

        preamble();

        Reg64 param = abi_param1;
        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
        eltwise_injector_->load_table_addr();

        Label reminder_loop_start, reminder_loop_end;
        Label vectorized_loop_start, vectorized_loop_end;

        cmp(reg_work_amount, simd_w);
        jl(reminder_loop_start, T_NEAR);

        L(vectorized_loop_start);

        uni_vmovups(vmm_src, ptr[reg_from]);
        eltwise_injector_->compute_vector(vmm_src.getIdx());
        uni_vmovups(ptr[reg_to], vmm_src);

        add(reg_from, vlen);
        add(reg_to, vlen);

        sub(reg_work_amount, simd_w);
        cmp(reg_work_amount, simd_w);
        jge(vectorized_loop_start, T_NEAR);

        L(vectorized_loop_end);

        L(reminder_loop_start);

        cmp(reg_work_amount, 0);
        jle(reminder_loop_end, T_NEAR);

        movss(xmm_src, ptr[reg_from]);
        eltwise_injector_->compute_vector(xmm_src.getIdx());
        movss(ptr[reg_to], xmm_src);

        add(reg_from, sizeof(float));
        add(reg_to, sizeof(float));

        dec(reg_work_amount);
        jmp(reminder_loop_start, T_NEAR);

        L(reminder_loop_end);

        postamble();

        eltwise_injector_->prepare_table();

        ker_ = (decltype(ker_))this->getCode();
    }

    ~jit_uni_kernel_fwd_f32() { delete eltwise_injector_; }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                isa == avx2, Ymm, Zmm>::type;

    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int vlen   = cpu_isa_traits<isa>::vlen;

    Reg64 reg_from = rax;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;

    Xmm xmm_src = Xmm(1);
    Vmm vmm_src = Vmm(1);

    jit_uni_eltwise_injector_f32<isa> *eltwise_injector_;
};

} /* namespace */

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::pd_t::init() {
    using namespace alg_kind;

    bool ok = true
        && mayiuse(isa)
        && is_fwd()
        && utils::everyone_is(data_type::f32, desc()->data_desc.data_type)
        && !has_zero_dim_memory()
        && utils::one_of(desc()->alg_kind, eltwise_relu, eltwise_tanh,
                eltwise_elu, eltwise_square, eltwise_abs, eltwise_sqrt,
                eltwise_linear, eltwise_bounded_relu, eltwise_soft_relu,
                eltwise_logistic)
        && memory_desc_wrapper(src_md()).is_dense(true)
        && IMPLICATION(!memory_desc_wrapper(src_md()).is_dense(false),
                math::eltwise_fwd_preserves_zero(desc()->alg_kind, true))
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::jit_uni_eltwise_fwd_t(const pd_t *apd)
    : cpu_primitive_t(apd), kernel_(nullptr) {
    const auto &desc = *pd()->desc();
    switch (desc.alg_kind) {
    case alg_kind::eltwise_relu:
        kernel_ = new jit_uni_relu_kernel_f32<isa>(desc); break;
    default:
        kernel_ = new jit_uni_kernel_fwd_f32<isa>(desc);
    }
}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::~jit_uni_eltwise_fwd_t()
{ delete kernel_; }

template <cpu_isa_t isa>
void jit_uni_eltwise_fwd_t<isa>::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());

    const size_t nelems = data_d.nelems(true);

    src += data_d.offset0();
    dst += data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        const int cache_line = 16;

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_args();
        arg.from = &src[start];
        arg.for_comparison = &src[start];
        arg.to = &dst[start];
        arg.work_amount = end - start;
        if (arg.work_amount)
            (*kernel_)(&arg);
    });
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::pd_t::init() {
    bool ok = true
        && !is_fwd()
        && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu)
        && src_md()->data_type == data_type::f32
        && !has_zero_dim_memory()
        && mayiuse(isa)
        && memory_desc_wrapper(src_md()).is_dense()
        && memory_desc_wrapper(diff_dst_md()) == memory_desc_wrapper(src_md())
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::jit_uni_eltwise_bwd_t(const pd_t *apd)
    : cpu_primitive_t(apd), kernel_(nullptr) {
    const auto &desc = *pd()->desc();
    switch (desc.alg_kind) {
    case alg_kind::eltwise_relu:
        kernel_ = new jit_uni_relu_kernel_f32<isa>(desc); break;
    default: assert(!"unknown eltwise alg_kind");
    }
}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::~jit_uni_eltwise_bwd_t()
{ delete kernel_; }

template <cpu_isa_t isa>
void jit_uni_eltwise_bwd_t<isa>::execute_backward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());

    const size_t nelems = data_d.nelems();

    src += data_d.offset0();
    diff_dst += diff_data_d.offset0();
    diff_src += diff_data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        const int cache_line = 16;

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_args();
        arg.from = &diff_dst[start];
        arg.to = &diff_src[start];
        arg.for_comparison = &src[start];
        arg.work_amount = end - start;
        if (arg.work_amount)
            (*kernel_)(&arg);
    });
}

template struct jit_uni_eltwise_fwd_t<sse42>;
template struct jit_uni_eltwise_bwd_t<sse42>;
template struct jit_uni_eltwise_fwd_t<avx2>;
template struct jit_uni_eltwise_bwd_t<avx2>;
template struct jit_uni_eltwise_fwd_t<avx512_common>;
template struct jit_uni_eltwise_bwd_t<avx512_common>;

}
}
}
