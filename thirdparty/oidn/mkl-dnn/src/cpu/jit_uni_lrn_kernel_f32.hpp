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

#ifndef CPU_JIT_UNI_LRN_KERNEL_F32_HPP
#define CPU_JIT_UNI_LRN_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

enum params { VECTOR_LENGTH = 8, MAX_LOCAL_SIZE = 32 };

typedef struct {
    const float *src;
    float *dst, *scratch;
} jit_args_fwd_t;

typedef struct {
    const float *src, *diff_dst, *scratch;
    float *diff_src;
} jit_args_bwd_t;

struct nchw8c_across {
    /*  version:
    *  -1: channels 0..7,
    *   1: channels C-8 .. C-1,
    *   0: other channels
    *   3: channels only for this kernel(without prev and next)
    */
    int H, W, version;
    nchw8c_across(int h, int w, int v) : H(h), W(w), version(v) {}
};

struct nchw8c_within {
    int H, W, size;
    nchw8c_within(int h, int w, int s) : H(h), W(w), size(s) {}
};

struct nchw_across {
    int C, HW, tail;
    nchw_across(int c, int hw, int t) : C(c), HW(hw), tail(t) {}
};

struct nhwc_across {
    int C;
    nhwc_across(int c) : C(c) {}
};

template <cpu_isa_t isa>
struct jit_uni_lrn_fwd_kernel_f32 : public jit_generator {
    Xbyak::Reg64 src = rax;
    Xbyak::Reg64 dst = r8;
    Xbyak::Reg64 scratch = rdx;
    Xbyak::Reg64 imm_addr64 = rbx;
    Xbyak::Reg64 store_addr = rbp;

    Xbyak::Xmm xalpha = xmm0;
    Xbyak::Ymm yalpha = ymm0;
    Xbyak::Xmm xk = xmm1;
    Xbyak::Ymm yk = ymm1;

    float alpha;
    float k;

    int stack_space_needed = 11 * 4 * sizeof(float) + 16;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lrn_fwd_kernel_f32)

    /* cpu specific part */
    using Vmm = typename utils::conditional<isa == avx2, Ymm, Zmm>::type;

    jit_uni_lrn_fwd_kernel_f32(
        const struct nchw8c_within &J,
        float A,
        float K,
        prop_kind_t pk,
        void *code_ptr = nullptr,
        size_t code_size = 4 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_fwd_kernel_f32(
        const struct nchw8c_across &J,
        float A,
        float K,
        prop_kind_t pk,
        void *code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_fwd_kernel_f32(
        const struct nhwc_across &J,
        float A,
        float K,
        prop_kind_t pk,
        void *code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_fwd_kernel_f32(
        struct nchw_across J,
        float A,
        float K,
        prop_kind_t pk,
        void* code_ptr = nullptr,
        size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE);

    void within_body(
        int hoff, int Hoff, int woff, int Woff, int stride,
        Xbyak::Ymm ysum, Xbyak::Ymm ydst, Xbyak::Ymm ytmp, Xbyak::Ymm ysum2,
        prop_kind_t pk);
    void within_body_sse42(
        int hoff, int Hoff, int woff, int Woff, int stride, prop_kind_t pk);


    void nchw_body(int tail, int HW, prop_kind_t pk,
        Xbyak::Ymm ymask,
        Xbyak::Ymm ya,
        Xbyak::Ymm yb,
        Xbyak::Ymm yc,
        Xbyak::Ymm yd,
        Xbyak::Ymm ye,
        Xbyak::Ymm ysum);
    void nchw_body_sse42(int tail, int HW, prop_kind_t pk,
        Xbyak::Xmm xmask_lo, Xbyak::Xmm xmask_hi,
        Xbyak::Xmm xe_lo, Xbyak::Xmm xe_hi,
        Xbyak::Xmm xsum_lo, Xbyak::Xmm xsum_hi);
    void nchw_tail_sse42(int tail, Xbyak::Reg64 reg_dst,
        Xbyak::Xmm xtail_lo, Xbyak::Xmm xtail_hi);

    void operator()(jit_args_fwd_t *arg) { ker(arg); }
    void(*ker)(jit_args_fwd_t *);
};

template <cpu_isa_t isa>
struct jit_uni_lrn_bwd_kernel_f32 : public jit_generator {
    Xbyak::Reg64 src = rax;
    Xbyak::Reg64 diffsrc = r8;
    Xbyak::Reg64 diffdst = r9;
    Xbyak::Reg64 workspace = rdx;
    Xbyak::Reg64 imm_addr64 = rsi;

    Xbyak::Xmm xnalphabeta = xmm0;
    Xbyak::Ymm ynalphabeta = ymm0;

    float nalphabeta;

    int use_h_parallelizm;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lrn_bwd_kernel_f32)

    jit_uni_lrn_bwd_kernel_f32(
        const struct nchw8c_across &J,
        float A,
        float B,
        int use_h_parallel,
        void *code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);

    void operator()(jit_args_bwd_t *arg) { ker(arg); }
    void(*ker)(jit_args_bwd_t *);
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
