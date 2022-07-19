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

#include "jit_avx512_core_gemm_s8u8s32_kern.hpp"


#ifdef _WIN32
static const bool is_windows = 1;
#else
static const bool is_windows = 0;
#endif


namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;




// Convert between vector register lengths.
static inline Xmm make_xmm(const Xmm &v) { return Xmm(v.getIdx()); }
static inline Ymm make_ymm(const Xmm &v) { return Ymm(v.getIdx()); }

// Load from or store to C.
void jit_avx512_core_gemm_s8u8s32_kern::c_load(const Xbyak::Xmm &dst,
    const Xbyak::Address &src, int nelems)
{
    switch (nelems) {
    default: vmovups(dst, src); break;
    case 8:  vmovups(make_ymm(dst), src); break;
    case 4:  vmovups(make_xmm(dst), src); break;
    case 2:  vmovlps(make_xmm(dst), src); break;
    case 1:  vmovss(make_xmm(dst), src); break;
    }
}
void jit_avx512_core_gemm_s8u8s32_kern::c_store(const Xbyak::Address &dst,
    const Xbyak::Xmm &src, int nelems)
{
    switch (nelems) {
    default: vmovups(dst, src); break;
    case 8:  vmovups(dst, make_ymm(src)); break;
    case 4:  vmovups(dst, make_xmm(src)); break;
    case 2:  vmovsd(dst, make_xmm(src)); break;
    case 1:  vmovss(dst, make_xmm(src)); break;
    }
}

// Perform length-4 dot product accumulations of unsigned and signed bytes
//  in parallel.
// Use vpdpbusd if VNNI available, otherwise emulate.
void jit_avx512_core_gemm_s8u8s32_kern::dot_product(const Xmm &dst,
    const Xmm &src1, const Xmm &src2)
{
    if (vnni)
        vpdpbusd(dst, src1, src2);
    else {
        vpmaddubsw(dp_scratch, src1, src2);
        vpmaddwd(dp_scratch, ones, dp_scratch);
        vpaddd(dst, dst, dp_scratch);
    }
}

// Inner kernel.
void jit_avx512_core_gemm_s8u8s32_kern::kernel_loop(int unroll_m, int unroll_n,
        bool cfetch)
{
    int um_vecs = (unroll_m + 15) >> 4;
    Label label_kernel_loop;

    L_aligned(label_kernel_loop); {
        for (int h = 0; h < 4; h++) {
            for (int j = 0; j < unroll_n; j++) {
                const Zmm b = b_regs[j & 1];

                vpbroadcastd(b, ptr[BO + isize *
                    (2 * j + 2 * h * unroll_n - offset_b)]);
                dot_product(c_regs[0][j], b, a_regs[0]);

                if (j == 1 && !(h & 1))
                    prefetch_b(ptr[BO + isize * (prefetch_size_b
                        + 2 * h * unroll_n - offset_b)]);
                else if (j % 3 == 0)
                    prefetch_a(ptr[AO + isize * (prefetch_size_a
                        + 32 * (j / 3) + 2 * h * unroll_m - offset_a)]);

                for (int i = 1; i < um_vecs; i++)
                    dot_product(c_regs[i][j], b, a_regs[i]);

                if (cfetch && (j == std::min(1, unroll_n - 1))) {
                    if (h == 3)
                        lea(CO2, ptr[CO2 + LDC]);
                    else if (h < um_vecs)
                        prefetch_c(ptr[CO2 + (16 * h * size)]);
                }

                if (h == 3 && j == std::min(3, unroll_n - 1))
                    lea(AA, ptr[AA + (32 * isize)]);
            }

            for (int i = 0; i < um_vecs; i++)
                vmovups(a_regs[i], ptr[AO + isize *
                (32 * i + 2 * (h + 1) * unroll_m - offset_a)]);

            if (h == 2)
                prefetch_x(ptr[AA - (offset_a * isize)]);
        }

        add(AO, 8 * isize * unroll_m);
        add(BO, 8 * isize * unroll_n);
        sub(LoopCount, 1);
        jg(label_kernel_loop, T_NEAR);
    }
}

// k remainder loop for kernel.
void jit_avx512_core_gemm_s8u8s32_kern::remainder_kernel(int unroll_m,
        int unroll_n, int unroll_k, int bwidth)
{
    if ((unroll_m > IGEMM_UNROLL_M) || (unroll_n > IGEMM_UNROLL_N)
            || (unroll_m < 0)  || (unroll_n < 0))
        return;

    int um_vecs = (unroll_m + 15) >> 4;

    for (int h = 0; h < unroll_k; h++) {
        for (int j = 0; j < unroll_n; j++) {
            Zmm b = b_regs[j & 1];
            auto b_src = ptr[BO + (-isize * offset_b
                + bwidth * (j + h * unroll_n))];

            switch (bwidth) {
            case 4:
                vpbroadcastd(b, b_src);
                break;
            case 2:
                vpbroadcastw(b, b_src);
                break;
            case 1:
                vpbroadcastb(b, b_src);
                break;
            }
            for (int i = 0; i < um_vecs; i++)
                dot_product(c_regs[i][j], b, a_regs[i]);
        }

        if (unroll_k > 1) {
            for (int i = 0; i < um_vecs; i++)
                vmovups(a_regs[i], ptr[AO + isize * (32 * i
                    + (h + 1) * 2 * unroll_m - offset_a)]);
        }
    }

    add(AO, unroll_k * unroll_m * bwidth);
    add(BO, unroll_k * unroll_n * bwidth);
}

// Inner loop.
void jit_avx512_core_gemm_s8u8s32_kern::innerloop(int unroll_m, int unroll_n)
{
    if ((unroll_m > IGEMM_UNROLL_M) || (unroll_n > IGEMM_UNROLL_N)
            || (unroll_m < 0)  || (unroll_n < 0))
        return;

    int um_vecs = (unroll_m + 15) >> 4;
    int stage1 = unroll_n, stage2 = unroll_n;

    Label label_kernel_loop_1, label_k_main_loop_2, label_kernel_loop_2;
    Label label_k_main_loop_3, label_kernel_loop_3;
    Label label_k_remainder_loop_begin, label_k_rem_4, label_k_rem_2;
    Label label_k_rem_1, label_update_begin;

    mov(AO, A);
    for (int i = 0; i < um_vecs; i++)
        vmovups(a_regs[i], ptr[AO + isize * (32 * i - offset_a)]);

    mov(LoopCount, K);
    sar(LoopCount, 4);
    jle(label_k_remainder_loop_begin, T_NEAR);

    // Main k loops, broken into three parts to time C prefetching.
    sub(LoopCount, stage1 + stage2);
    jle(label_k_main_loop_2, T_NEAR);

    kernel_loop(unroll_m, unroll_n, false);

    L_aligned(label_k_main_loop_2);
    lea(CO2, ptr[CO1 + size * (std::min(unroll_m, 16) - 1)]);
    add(LoopCount, stage1);
    jle(label_k_main_loop_3, T_NEAR);

    kernel_loop(unroll_m, unroll_n, true);

    L_aligned(label_k_main_loop_3);
    lea(CO2, ptr[CO1 + size * (std::min(unroll_m, 16) - 1)]);
    add(LoopCount, stage2);
    jle(label_k_remainder_loop_begin, T_NEAR);

    kernel_loop(unroll_m, unroll_n, true);

    // k remainder handling
    L_aligned(label_k_remainder_loop_begin);
    mov(LoopCount, K);
    test(LoopCount, 8);
    je(label_k_rem_4, T_NEAR);

    remainder_kernel(unroll_m, unroll_n, 2, 4);

    L_aligned(label_k_rem_4);
    mov(LoopCount, K);
    test(LoopCount, 4);
    je(label_k_rem_2, T_NEAR);

    remainder_kernel(unroll_m, unroll_n, 1, 4);

    L_aligned(label_k_rem_2);
    mov(LoopCount, K);
    test(LoopCount, 2);
    je(label_k_rem_1, T_NEAR);

    Zmm zero = zmm6;
    Zmm tmp = zmm5;

    vpxorq(zero, zero, zero);
    for (int i = 0; i < um_vecs; i++) {
        Zmm a = a_regs[i];
        vbroadcasti64x4(a, ptr[AO + isize * (16 * i - offset_a)]);
        vpunpcklwd(tmp, a, zero);
        vpunpckhwd(a, a, zero);
        vshufi32x4(a, tmp, a, 0x44);
        vshufi32x4(a, a, a, 0xD8);
    }

    remainder_kernel(unroll_m, unroll_n, 1, 2);

    L_aligned(label_k_rem_1);
    mov(LoopCount, K);
    test(LoopCount, 1);
    je(label_update_begin, T_NEAR);

    vpxorq(zero, zero, zero);
    for (int i = 0; i < um_vecs; i++) {
        Zmm a = a_regs[i];
        vbroadcasti32x4(a, ptr[AO + isize * (8 * i - offset_a)]);
        vpunpcklbw(tmp, a, zero);
        vpunpckhbw(a, a, zero);
        vinsertf128(make_ymm(a), make_ymm(tmp), make_xmm(a), 1);
        vpunpcklwd(tmp, a, zero);
        vpunpckhwd(a, a, zero);
        vshufi32x4(a, tmp, a, 0x44);
        vshufi32x4(a, a, a, 0xD8);
    }

    remainder_kernel(unroll_m, unroll_n, 1, 1);

    // Add offsets and update C.
    L_aligned(label_update_begin);

    if (enable_offset_r) {
        // Add row offsets.
        mov(rax, coffset_ry);
        for (int j = 0; j < unroll_n; j++) {
            Zmm row_offset = zmm0;

            vbroadcastss(row_offset, ptr[rax + size * j]);

            for (int i = 0; i < um_vecs; i++)
                vpaddd(c_regs[i][j], c_regs[i][j], row_offset);
        }
        add(coffset_ry, size * unroll_n);
    }

    if (enable_offset_c) {
        // Add column offsets.
        mov(rax, coffset_cy);
        for (int i = 0; i < um_vecs; i++) {
            Zmm col_offset = zmm0;

            c_load(col_offset, ptr[rax + size * 16 * i], unroll_m);

            for (int j = 0; j < unroll_n; j++)
                vpaddd(c_regs[i][j], c_regs[i][j], col_offset);
        }
    }

    Reg64 LDC3 = rax;
    lea(LDC3, ptr[LDC + LDC * 2]);

    // C updates.
    int c_off_j = 0;
    for (int j = 0; j < unroll_n; j++) {
        if (j > 0 && (j & 3) == 0) {
            lea(CO1, ptr[CO1 + LDC * 4]);
            c_off_j += 4;
        }

        int jj = j - c_off_j;

        for (int i = 0; i < um_vecs; i++) {
            Zmm c = c_regs[i][j];
            Zmm c_old = zmm0;
            decltype(LDC * jj) ldc_mult = (jj == 3) ? LDC3 : LDC * jj;

            auto c_mem = ptr[CO1 + ldc_mult + size * 16 * i];

            if (beta_zero)
                c_store(c_mem, c, unroll_m);
            else {
                c_load(c_old, c_mem, unroll_m);
                vpaddd(c_old, c, c_old);
                c_store(c_mem, c_old, unroll_m);
            }

            vpxorq(c, c, c);
        }
    }

    lea(CO1, ptr[CO1 + LDC * (unroll_n - c_off_j)]);
}

// Outer loop.
void jit_avx512_core_gemm_s8u8s32_kern::outerloop(int unroll_x, int unroll_y,
    Label *&cur_outerloop_label)
{
    Label label_m_loop, label_n_loop, label_n_remainder_loops[6];

    L(*cur_outerloop_label);
    cur_outerloop_label++;
    if (unroll_x >= IGEMM_UNROLL_M) {
        mov(J, M);
        cmp(J, unroll_x);
        jl(*cur_outerloop_label, T_NEAR);    // Jump to next outerloop label.
    } else {
        test(J, unroll_x);
        jle(*cur_outerloop_label, T_NEAR);
    }

    L_aligned(label_m_loop); {
        mov(CO1, C);
        add(C, unroll_x * size);

        mov(BO, B);

        mov(AA, K);
        imul(AA, AA, unroll_x * isize);
        lea(AA, ptr[A + AA + isize * prefetch_size_a]);

        if (enable_offset_c) {
            mov(rax, coffset_cx);
            mov(coffset_cy, rax);
            add(rax, unroll_x * size);
            mov(coffset_cx, rax);
        }

        if (enable_offset_r) {
            mov(rax, coffset_rx);
            mov(coffset_ry, rax);
        }

        mov(I, N);
        cmp(I, unroll_y);
        jl(label_n_remainder_loops[0], T_NEAR);

        L_aligned(label_n_loop); {
            innerloop(unroll_x, unroll_y);
            sub(I, unroll_y);
            cmp(I, unroll_y);
            jge(label_n_loop, T_NEAR);
        }

        align(16);

        int label_idx = 0;
        for (int uy = 16; uy > 0; uy >>= 1) {
            L(label_n_remainder_loops[label_idx++]);
            if (unroll_y > uy) {
                test(I, uy);
                jle(label_n_remainder_loops[label_idx], T_NEAR);

                innerloop(unroll_x, uy);
                align(16);
            }
        }
        L(label_n_remainder_loops[label_idx]);

        mov(A, AO);
        if (unroll_x >= IGEMM_UNROLL_M) {
            sub(J, unroll_x);
            cmp(J, unroll_x);
            jge(label_m_loop);
        }
    }

    align(16);
}

void jit_avx512_core_gemm_s8u8s32_kern::generate()
{
    // Prologue
    preamble();
    sub(rsp, stack_alloc_size);

    if (is_windows) {
        mov(A, arg_a);
        mov(B, arg_b);
    }

    mov(C, arg_c);
    mov(LDC, arg_ldc);

    sub(A, -offset_a * isize);
    sub(B, -offset_b * isize);

    mov(M, qword[M]);
    mov(N, qword[N]);
    mov(K, qword[K]);

    lea(LDC, ptr[LDC * size]);

    if (enable_offset_c) {
        mov(rax, arg_coffset_c);
        mov(coffset_cx, rax);
    }
    if (enable_offset_r) {
        mov(rax, arg_coffset_r);
        mov(coffset_rx, rax);
    }

    for (int i = 0; i < (max_unroll_m >> 4); i++) {
        for (int j = 0; j < max_unroll_n; j++) {
            auto &c = c_regs[i][j];
            vpxorq(c, c, c);
        }
    }

    if (!vnni) {
        mov(rax, 1);
        movq(make_xmm(ones), rax);
        vpbroadcastw(ones, make_xmm(ones));
    }

    Label outerloop_labels[8];
    Label *cur_outerloop_label = &outerloop_labels[0];

    // Main m loop.
    outerloop(IGEMM_UNROLL_M, IGEMM_UNROLL_N, cur_outerloop_label);

    // m remainder loops.
    for (int um = 32; um > 0; um >>= 1)
        if (IGEMM_UNROLL_M > um)
            outerloop(um, IGEMM_UNROLL_N, cur_outerloop_label);

    L(*cur_outerloop_label);

    // Epilogue.
    add(rsp, stack_alloc_size);
    postamble();
}


jit_avx512_core_gemm_s8u8s32_kern::jit_avx512_core_gemm_s8u8s32_kern(bool
        beta_zero_, bool enable_offset_c_, bool enable_offset_r_) :
    jit_generator(nullptr, 100000), arg_a(0), arg_b(0), arg_c(0), arg_ldc(0),
    arg_coffset_c(0), arg_coffset_r(0), coffset_cx(0), coffset_cy(0),
    coffset_rx(0), coffset_ry(0)
{
    beta_zero = beta_zero_;
    enable_offset_c = enable_offset_c_;
    enable_offset_r = enable_offset_r_;
    vnni = mayiuse(avx512_core_vnni);

    // Assign integer registers
    M = is_windows ? rcx : rdi;
    N = is_windows ? rdx : rsi;
    K = is_windows ? r8 : rdx;
    A = is_windows ? rsi : r8;
    B = r9;
    C = r10;
    LDC = r11;
    I = r12;
    J = r13;
    LoopCount = rax;
    AO = r14;
    BO = r15;
    CO1 = rbx;
    CO2 = rbp;
    AA = is_windows ? rdi : rcx;

    // Assign vector registers
    dp_scratch = zmm6;
    ones = zmm7;
    for (int i = 0; i < (max_unroll_m >> 4); i++)
        a_regs[i] = Zmm(i);
    b_regs[0] = zmm4;
    b_regs[1] = zmm5;

    int rn = 0;
    for (int i = 0; i < (max_unroll_m >> 4); i++)
        for (int j = 0; j < max_unroll_n; j++)
            c_regs[i][j] = Zmm(8 + rn++);

    // Assign stack variables.
    stack_alloc_size = 32;
    auto args_offset = stack_alloc_size + get_size_of_abi_save_regs()
        + 8 + (is_windows ? 48 : 0);

    arg_a         = ptr[rsp + (args_offset - 16)];
    arg_b         = ptr[rsp + (args_offset - 8)];
    arg_c         = ptr[rsp + (args_offset + 0)];
    arg_ldc       = ptr[rsp + (args_offset + 8)];
    arg_coffset_c = ptr[rsp + (args_offset + 16)];
    arg_coffset_r = ptr[rsp + (args_offset + 24)];

    coffset_cx = qword[rsp + 0];
    coffset_cy = qword[rsp + 8];
    coffset_rx = qword[rsp + 16];
    coffset_ry = qword[rsp + 24];

    generate();
}

}
}
}
