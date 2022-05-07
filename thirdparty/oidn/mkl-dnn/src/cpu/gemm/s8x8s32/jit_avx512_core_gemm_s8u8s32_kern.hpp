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

#ifndef IGEMM_KERNEL_GENERATOR_HPP
#define IGEMM_KERNEL_GENERATOR_HPP

#include "jit_generator.hpp"


namespace mkldnn {
namespace impl {
namespace cpu {

class jit_avx512_core_gemm_s8u8s32_kern : public jit_generator {
public:
    jit_avx512_core_gemm_s8u8s32_kern(bool beta_zero_, bool enable_offset_c_,
        bool enable_offset_r_);
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_gemm_s8u8s32_kern);

protected:
    bool beta_zero;
    bool enable_offset_c, enable_offset_r;
    bool vnni;

    void prefetch_a(const Xbyak::Address &src) {
        prefetcht0(src);
    }
    void prefetch_b(const Xbyak::Address &src) {
        prefetcht0(src);
    }
    void prefetch_c(const Xbyak::Address &src) {
        prefetchw(src);
    }
    void prefetch_x(const Xbyak::Address &src) {
        prefetcht0(src);
    }

    void c_load(const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems);
    void c_store(const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems);

    void dot_product(const Xbyak::Xmm &dst, const Xbyak::Xmm &src1,
        const Xbyak::Xmm &src2);
    void kernel_loop(int unroll_m, int unroll_n, bool cfetch);
    void remainder_kernel(int unroll_m, int unroll_n, int unroll_k, int bwidth);
    void innerloop(int unroll_m, int unroll_n);
    void outerloop(int unroll_x, int unroll_y, Xbyak::Label *&outerloop_label);

    void generate();


private:
    static const int IGEMM_UNROLL_M = 48;
    static const int IGEMM_UNROLL_N = 8;

    static const int isize = 2;
    static const int size = 4;

    // Prefetch configuration
    static const int prefetch_size_a = 32 * 5;
    static const int prefetch_size_b = 32 * 4;

    static const int offset_a = 256, offset_b = 256;
    static const int max_unroll_m = 48, max_unroll_n = 8;

    // Integer register assignments
    Xbyak::Reg64 M, N, K, A, B, C, LDC, I, J, LoopCount;
    Xbyak::Reg64 AO, BO, CO1, CO2, AA;

    // Vector register assignments
    Xbyak::Zmm dp_scratch, ones, a_regs[max_unroll_m >> 4], b_regs[2];
    Xbyak::Zmm c_regs[max_unroll_m >> 4][max_unroll_n];

    // Stack variable assignments
    int stack_alloc_size;
    Xbyak::Address arg_a, arg_b, arg_c, arg_ldc, arg_coffset_c, arg_coffset_r;
    Xbyak::Address coffset_cx, coffset_cy, coffset_rx, coffset_ry;

    void L_aligned(Xbyak::Label &label, int alignment = 16) {
        align(alignment);
        L(label);
    }
};

}
}
}

#endif /* header guard */
