/*******************************************************************************
 * Copyright 2019 Intel Corporation
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

#include "jit_avx512_core_kernel_gemv_s8u8s32_kern.hpp"

#ifdef _WIN32
#define is_windows 1
#else
#define is_windows 0
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

void jit_avx512_core_gemv_s8u8s32_kern::vnni(Xbyak::Zmm acc, Xbyak::Zmm b,
                                             Xbyak::Zmm a, Xbyak::Zmm tmp,
                                             Xbyak::Zmm one, bool swap,
                                             int use_vnni) {

    if (use_vnni) {
        if (swap)
            vpdpbusd(acc, a, b);
        else
            vpdpbusd(acc, b, a);
    }

    else {
        if (swap)
            vpmaddubsw(tmp, a, b);
        else
            vpmaddubsw(tmp, b, a);
        vpmaddwd(tmp, tmp, one);
        vpaddd(acc, tmp, acc);
    }

}

void jit_avx512_core_gemv_s8u8s32_kern::n_loop_body(int start_a_idx, int start_acc_idx,
                                                    int b_idx, int nreg_acc,
                                                    Xbyak::Reg64 A, Xbyak::Reg64 lda,
                                                    Xbyak::Reg64 X, Xbyak::Zmm tmp,
                                                    Xbyak::Zmm one, bool swap, int use_vnni,
                                                    int use_mask, Xbyak::Opmask mask_n) {

    int i;
    int nreg_A = nreg_acc / 2 + (nreg_acc % 2);

    // load X + j
    if (use_mask)
        vmovdqu8(Xbyak::Zmm(b_idx) | mask_n | T_z, ptr[X]);
    else
        vmovdqu8(Xbyak::Zmm(b_idx), ptr[X]);

    xor_(r14, r14);
    // load values of A
    for (i = 0; i < nreg_A; i++) {
        if (use_mask)
            vmovdqu8(Xbyak::Zmm(start_a_idx + i) | mask_n | T_z, ptr[A + r14]);
        else
            vmovdqu8(Xbyak::Zmm(start_a_idx + i), ptr[A + r14]);
        add(r14, lda);
    }

    for (i = 0; i < nreg_A; i++) {
        // vnni (acc, b, a, tmp, one, swap, use_vnni)
        vnni(Xbyak::Zmm(start_acc_idx + i), Xbyak::Zmm(b_idx),
             Xbyak::Zmm(start_a_idx + i), tmp, one, swap, use_vnni);
    }

    for (i = 0; i < nreg_A - (nreg_acc % 2); i++) {
        if (use_mask)
            vmovdqu8(Xbyak::Zmm(start_a_idx + i) | mask_n | T_z, ptr[A + r14]);
        else
            vmovdqu8(Xbyak::Zmm(start_a_idx + i), ptr[A + r14]);
        add(r14, lda);
    }

    for (i = 0; i < nreg_A - (nreg_acc % 2); i++) {
        vnni(Xbyak::Zmm(start_acc_idx + i + nreg_A), Xbyak::Zmm(b_idx),
             Xbyak::Zmm(start_a_idx + i), tmp, one, swap, use_vnni);
    }

}

void jit_avx512_core_gemv_s8u8s32_kern::shuffle_and_add(Xbyak::Zmm dest, Xbyak::Zmm A,
                                                        Xbyak::Zmm B, Xbyak::Zmm C,
                                                        Xbyak::Zmm D) {

    vshufi32x4(dest, A, C, 0x44);
    vshufi32x4(A, A, C, 0xEE);
    vpaddd(C, dest, A); // C = A0 + A2|A1 + A3|C0 + C2|C1 + C3

    vshufi32x4(dest, B, D, 0x44);
    vshufi32x4(B, B, D, 0xEE);
    vpaddd(D, dest, B); // D = B0 + B2|B1 + B3|D0 + D2|D1 + D3

    vshufi32x4(A, C, D, 0x88);
    vshufi32x4(B, C, D, 0xDD);
    vpaddd(dest, A, B); // dest = SAi|SBi|SCi|SDi

}

void jit_avx512_core_gemv_s8u8s32_kern::update_c(int nreg_acc, Xbyak::Reg64 Y,
                                                 int start_a_idx, int start_acc_idx,
                                                 Xbyak::Xmm beta, int use_mask,
                                                 Xbyak::Opmask mask_m) {

    int l, i, k, j, last_it;
    Xbyak::Label store_label;

    l = 0;
    for (k = 0; k < nreg_acc; k += 8) {
        for (i = 0, j = k; i < 8; i += 4, j += 2) {
            if (j < nreg_acc) {
                // shuffle per block of 4 registers
                shuffle_and_add(Xbyak::Zmm(start_a_idx + l), // dest
                                Xbyak::Zmm(start_acc_idx + j), // A = acc0
                                Xbyak::Zmm(start_acc_idx + 1 + j), // B = acc1
                                Xbyak::Zmm(start_acc_idx + 4 + j), // C = acc4
                                Xbyak::Zmm(start_acc_idx + 5 + j)); // D = acc5

                // extract low and high from dest and hadd
                vextracti32x8(Xbyak::Ymm(start_a_idx + l + 1), Xbyak::Zmm(start_a_idx + l), 0);
                vextracti32x8(Xbyak::Ymm(start_a_idx + l + 2), Xbyak::Zmm(start_a_idx + l), 1);
                vphaddd(Xbyak::Ymm(start_a_idx + l),
                        Xbyak::Ymm(start_a_idx + l + 1),
                        Xbyak::Ymm(start_a_idx + l + 2));
            }
            l++;
        }

        vphaddd(Xbyak::Ymm(start_a_idx + l),
                Xbyak::Ymm(start_a_idx + l - 2),
                Xbyak::Ymm(start_a_idx + l - 1));

        l++;
    }

    // eventually add with C and store new value
    vxorps(Xbyak::Ymm(start_a_idx),
           Xbyak::Ymm(start_a_idx),
           Xbyak::Ymm(start_a_idx));
    vucomiss(beta, Xbyak::Ymm(start_a_idx));
    je(store_label, T_NEAR);

    // beta = 1
    for (k = 0, l = 2; k < nreg_acc; k += 8, l += 3) {
        // load Y and add
        last_it = (k + 8) > nreg_acc;
        if (use_mask && last_it)
            vmovdqu32(Xbyak::Ymm(start_a_idx + k / 8) | mask_m | T_z, ptr[Y + (k / 8) * 32]);
        else
            vmovdqu32(Xbyak::Ymm(start_a_idx + k / 8), ptr[Y + (k / 8) * 32]);

        vpaddd(Xbyak::Ymm(start_a_idx + l),
               Xbyak::Ymm(start_a_idx + l),
               Xbyak::Ymm(start_a_idx + k / 8));
    }

    // store
    aligned_label(store_label);
    for (k = 0, l = 2; k < nreg_acc; k += 8, l += 3) {
        last_it = (k + 8) > nreg_acc;
        if (use_mask && last_it)
            vmovdqu32(ptr[Y + (k / 8) * 32], Xbyak::Ymm(start_a_idx + l) | mask_m);
        else
            vmovdqu32(ptr[Y + (k / 8) * 32], Xbyak::Ymm(start_a_idx + l));
    }

}

template <typename T>
T jit_avx512_core_gemv_s8u8s32_kern::generate(int use_vnni) {

    Xbyak::Opmask mask_n = k1, mask_m = k2;
    Xbyak::Label one_label, m_tail_label, m_loop_label, n_loop_label;
    Xbyak::Label n_tail_label, update_c_label, end_label;
    constexpr unsigned int n_labels = (1 << unroll_m) - 1;
    Xbyak::Label m_tail_label_case[n_labels];
    Xbyak::Label n_loop_label_case[n_labels];
    Xbyak::Label n_tail_label_case[n_labels];
    Xbyak::Label update_c_label_case[n_labels];

    int i, ii;

    Xbyak::Zmm one, tmp;
    Xbyak::Reg64 n = abi_param2, m = abi_param1;
    Xbyak::Reg64 A = is_windows ? abi_param4 : abi_param3;
    Xbyak::Reg64 lda = is_windows ? abi_param3 : abi_param4;
    Xbyak::Reg64 X = is_windows ? rdi : r8;
    Xbyak::Xmm beta = xmm1;
    Xbyak::Reg64 Y = is_windows ? rsi : r9;

    bool swap = !std::is_same<T, gemv_s8u8s32_kernel_t>::value;

    // Windows: read on the stack lda, X, beta, Y

    int zmm_idx = 1;
    int nreg_acc = 1 << unroll_m;
    int nreg_A = 1 << (unroll_m - 1);
    int nreg_A_acc = nreg_acc + nreg_A;

    if (!use_vnni) {
        // set a zmm register to one
        tmp = Xbyak::Zmm(0);
        one = Xbyak::Zmm(zmm_idx + 1);
        zmm_idx += 2; // one + tmp
    }
    else {
        beta = xmm0;
    }

    preamble();

    if (is_windows) {
        mov(lda, ptr[rsp + get_size_of_abi_save_regs() + 40]);
        mov(X, ptr[rsp + get_size_of_abi_save_regs() + 48]);
        movss(beta, ptr[rsp + get_size_of_abi_save_regs() + 56]);
        mov(Y, ptr[rsp + get_size_of_abi_save_regs() + 64]);
    }

    if (use_vnni && !is_windows) {
        movaps(beta, xmm1);
    }

    mov(rax, (1 << unroll_n) - 1);
    kmovq(k3, rax);

    and_(rax, n); // rax contains n & ((1 << unroll_n) - 1)
    mov(rbx, 1);
    shlx(rbx, rbx, rax);
    sub(rbx, 1);
    kmovq(mask_n, rbx);
    // mask_n set (AVX512 only), can use rax and rbx again

    // set mask_m for update of the C matrix
    // load/store on the C matrix use Ymm so tail according to Ymm size
    mov(rax, 7); // 8 * 32 = 256 Ymm size
    and_(rax, m); // rax contains m & 7
    mov(rbx, 1);
    shlx(rbx, rbx, rax);
    sub(rbx, 1);
    kmovq(mask_m, rbx);
    // mask_m set (AVX512 only), can use rax and rbx again

    // setup register of ones when VNNI instructions not available
    if (!use_vnni) {
        vmovdqu16(one, ptr[rip + one_label]);
    }

    // M loop
    // base pointer for A rax contains a + i * lda
    // Loop stop when rax >= a + (m & mask_um) * lda = rbx
    // loop increment r10 = um * lda
    // rbp = Y + i
    mov(rax, A); // i = 0
    mov(rbx, m);
    and_(rbx, mask_um);
    imul(rbx, lda);
    add(rbx, A);
    mov(r10, lda);
    sal(r10, unroll_m);
    mov(rbp, Y);

    // N loop
    // base pointer for X r11 contains x + j
    // Loop stop when r11 >= x + n & mask_un = r12
    // loop increment un
    // r13 = rax + j = A + i * lda + j
    mov(r12, n);
    and_(r12, mask_un);
    add(r12, X);

    // M loop
    aligned_label(m_loop_label);
    cmp(rax, rbx);
    jge(m_tail_label, T_NEAR);

    // enter M loop
    for(i = 0; i < nreg_acc; i++) {
        vpxorq(Xbyak::Zmm(i + zmm_idx + nreg_A),
               Xbyak::Zmm(i + zmm_idx + nreg_A),
               Xbyak::Zmm(i + zmm_idx + nreg_A));
    }

    // N loop
    mov(r11, X); // j = 0
    mov(r13, rax);
    aligned_label(n_loop_label);
    cmp(r11, r12);
    jge(n_tail_label, T_NEAR);

    // enter N loop

    n_loop_body(zmm_idx, zmm_idx + nreg_A, zmm_idx + nreg_A_acc, nreg_acc,
                r13, lda, r11, tmp, one, swap, use_vnni, 0, mask_n);

    // increment rax with un
    add(r11, 1 << unroll_n);
    add(r13, 1 << unroll_n);
    jmp(n_loop_label, T_NEAR);
    // end N loop

    // N tail
    aligned_label(n_tail_label);

    ktestq(mask_n, k3);
    je(update_c_label, T_NEAR);
    n_loop_body(zmm_idx, zmm_idx + nreg_A, zmm_idx + nreg_A_acc, nreg_acc,
                r13, lda, r11, tmp, one, swap, use_vnni, 1, mask_n);

    // update C matrix
    aligned_label(update_c_label);

    update_c(nreg_acc, rbp, zmm_idx, zmm_idx + nreg_A, beta, 0, mask_m);

    // increment rax with um * lda
    add(rax, r10);
    add(rbp, 1 << (unroll_m + 2));
    jmp(m_loop_label, T_NEAR);
    // end M loop

    // M tail
    aligned_label(m_tail_label);

    // r10 will contain m_tail = m % unroll_m = m & (1 << unroll_m) - 1
    mov(r10, m);
    and_(r10, (1 << unroll_m) - 1);
    for (ii = 1; ii < 1 << unroll_m; ii++) {
        aligned_label(m_tail_label_case[ii-1]);
        cmp(r10, ii);
        if (ii == (1 << unroll_m) - 1)
            jne(end_label, T_NEAR);
        else
            jne(m_tail_label_case[ii], T_NEAR);

        // m_tail = i, use i accumulators

        for(i = 0; i < ii; i++) {
            vpxorq(Xbyak::Zmm(i + zmm_idx + nreg_A),
                   Xbyak::Zmm(i + zmm_idx + nreg_A),
                   Xbyak::Zmm(i + zmm_idx + nreg_A));
        }

        // N loop
        mov(r11, X); // j = 0
        mov(r13, rax);
        aligned_label(n_loop_label_case[ii - 1]);
        cmp(r11, r12);
        jge(n_tail_label_case[ii - 1], T_NEAR);

        n_loop_body(zmm_idx, zmm_idx + nreg_A, zmm_idx + nreg_A_acc, ii, r13,
                    lda, r11, tmp, one, swap, use_vnni, 0, mask_n);

        // increment rax with un
        add(r11, 1 << unroll_n);
        add(r13, 1 << unroll_n);
        jmp(n_loop_label_case[ii - 1], T_NEAR);
        // end N loop

        // N tail
        aligned_label(n_tail_label_case[ii - 1]);
        ktestq(mask_n, k3);
        je(update_c_label_case[ii - 1], T_NEAR);
        n_loop_body(zmm_idx, zmm_idx + nreg_A, zmm_idx + nreg_A_acc, ii, r13,
                    lda, r11, tmp, one, swap, use_vnni, 1, mask_n);

        // update C matrix
        aligned_label(update_c_label_case[ii - 1]);
        update_c(ii, rbp, zmm_idx, zmm_idx + nreg_A, beta, 1, mask_m);

        if (ii < ((1 << unroll_m) - 1))
            jmp(end_label, T_NEAR);
    }

    aligned_label(end_label);

    postamble();

    if (!use_vnni) {
        aligned_label(one_label);
        for (i = 0; i < size_vec_reg/8; i++)
            dq(0x0001000100010001);
    }

    return (T) getCode();
}

template jit_avx512_core_gemv_s8u8s32_kern::gemv_s8u8s32_kernel_t
jit_avx512_core_gemv_s8u8s32_kern::generate<jit_avx512_core_gemv_s8u8s32_kern::gemv_s8u8s32_kernel_t>(int);

template jit_avx512_core_gemv_s8u8s32_kern::gemv_u8s8s32_kernel_t
jit_avx512_core_gemv_s8u8s32_kern::generate<jit_avx512_core_gemv_s8u8s32_kern::gemv_u8s8s32_kernel_t>(int);

}
}
}
