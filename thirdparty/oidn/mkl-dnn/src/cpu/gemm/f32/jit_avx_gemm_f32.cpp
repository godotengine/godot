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

#include <cmath>
#include <mutex>

#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "ref_gemm_f32.hpp"
#include "gemm_utils_f32.hpp"
#include "jit_avx_gemm_f32.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#define CACHE_LINE_SIZE 64

#define STACKSIZE get_size_of_abi_save_regs()
#if _WIN32
#define STACK_K_CAPACITY 128
#else
#define STACK_K_CAPACITY 8192
#endif
#define SIZE 4
#define OFFSET 32
#define BASE_SHIFT 2
#define SECOND_FETCH 14

namespace avx_gemm_f32 {
using namespace gemm_utils;

struct xbyak_gemm : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_gemm_f32_xbyak_gemm)

    xbyak_gemm(char isTransA, char isTransB, float beta, bool hasBias = false,
            void *code_ptr = nullptr,
            size_t code_size = 80 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
    {
        using namespace Xbyak;

        const bool is_avx2 = mayiuse(avx2);
        assert(IMPLICATION(!is_avx2, mayiuse(avx)));

        const int UNROLL_M = is_avx2 ? 16 : 8;
        const int UNROLL_N = 6;

        bool isBeta0 = (beta == 0.0);
        bool isBetaN = (!isBeta0 && beta != 1.0);

        // various definitions for convenience
        auto ARG_M = abi_param1;
        auto ARG_N = abi_param2;
        auto K = abi_param3;
        auto ARG_ALPHA = abi_param4;
#ifdef _WIN32
        auto ARG_A = ptr[rsp + OFFSET_SHADOWSPACE + STACKSIZE];
        auto ARG_LDA = qword[rsp + OFFSET_SHADOWSPACE +
            sizeof(float *) + STACKSIZE];
        const auto stackOffset = OFFSET_SHADOWSPACE +
            sizeof(float *) + STACKSIZE;
        auto A = rsi;
        auto LDA = rdi;
#else
        auto ARG_A = r8;
        auto ARG_LDA = r9;
        const auto stackOffset = STACKSIZE;
        auto A = ARG_A;
        auto LDA = ARG_LDA;
#endif
        auto ARG_B = ptr[rsp + 8 + stackOffset];
        auto ARG_LDB = ptr[rsp + 16 + stackOffset];
        auto ARG_BETA = ptr[rsp + 24 + stackOffset];
        auto ARG_C = ptr[rsp + 32 + stackOffset];
        auto ARG_LDC = ptr[rsp + 40 + stackOffset];
        auto ARG_BIAS = ptr[rsp + 48 + stackOffset];
        auto ARG_WS = ptr[rsp + 56 + stackOffset];

        auto B = r11;
        auto LDB = rbx;
        auto LDC = r13;
        auto LL = rax;
        auto AO1 = abi_param2;
        auto BO1 = abi_param4;
        auto BO2 = rbp;
        auto CO1 = r14;
        auto CO2 = r15;
        auto LDB3 = r10;
        auto LDA4 = abi_param1;
        auto AA = r12;
        auto BIAS1 = abi_param1;

        auto M = qword[rsp + 0];
        auto N = qword[rsp + 8];
        auto FLAG = qword[rsp + 16];
        auto I = qword[rsp + 24];
        auto C = qword[rsp + 32];
        auto BIAS = qword[rsp + 40];
        auto ALPHA = qword[rsp + 48];
        auto BETA = qword[rsp + 64];
        auto ORIG_A = qword[rsp + 80];
        auto MASK = dword[rsp + 88];
        auto STRIDE = qword[rsp + 120];
        auto ORIG_SP = qword[rsp + 152];

        auto VALPHA = ymm1;
        auto VBETA = ymm2;
        auto VMASK = ymm3;
        auto VBIAS1 = ymm2;
        auto VBIAS2 = ymm4;

        auto PREFETCHSIZEA = 128;
        auto PREFETCHSIZEB = (!isTransB) ? -16 : 0;

        // Function for packing if needed
        auto do_pack = [&](
                int unroll_m, bool isLoad1Unmasked, bool isLoad2Unmasked) {
            Label pack2, pack3, pack4, pack10;

            int regIdx;
            Reg64 reg;

            mov(BO1, A);
            lea(AO1, ptr[rsp + 256 + OFFSET * SIZE]);

            if (isTransA) {
                lea(BO2, ptr[BO1 + LDA * 4]);
                lea(CO1, ptr[LDA + LDA * 2]);
                vmovupd(ymm7, STRIDE);
            }

            mov(LL, K);
            sar(LL, 2);
            jle(pack3, T_NEAR);
            align(16);

            L(pack2);
            if (!isTransA) {
                for (int i = 0; i < 4; i++) {
                    regIdx = (i % 2 == 0) ? 4 : 6;
                    if (isLoad1Unmasked) {
                        vmovups(Ymm(regIdx),
                                ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(Ymm(regIdx), VMASK,
                                ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                    }
                    if (unroll_m > 8) {
                        if (isLoad2Unmasked) {
                            vmovups(Ymm(regIdx + 1),
                                    ptr[BO1 + (1 * 8 - OFFSET) * SIZE]);
                        } else {
                            vmaskmovps(Ymm(regIdx + 1), VMASK,
                                    ptr[BO1 + (1 * 8 - OFFSET) * SIZE]);
                        }
                    }
                    add(BO1, LDA);

                    vmovups(ptr[AO1 + (unroll_m * i + 0 * 8 - OFFSET) * SIZE],
                            Ymm(regIdx));
                    if (unroll_m > 8) {
                        vmovups(ptr[AO1
                                        + (unroll_m * i + 1 * 8 - OFFSET)
                                                * SIZE],
                                Ymm(regIdx + 1));
                    }
                }

            } else {
                if (isLoad1Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        reg = (i % 2 == 0) ? BO1 : BO2;
                        vmovups(xmm0, ptr[reg + (0 * 8 - OFFSET) * SIZE]);
                        vmovups(xmm1,
                                ptr[reg + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        lea(BO2, ptr[reg + LDA * 2]);
                        vunpcklps(xmm4, xmm0, xmm1);
                        vunpckhps(xmm5, xmm0, xmm1);
                        vmovups(xmm0, ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        vmovups(xmm1,
                                ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        lea(BO2, ptr[BO2 + LDA * 2]);
                        vunpcklps(xmm6, xmm0, xmm1);
                        vunpckhps(xmm2, xmm0, xmm1);

                        vunpcklpd(xmm0, xmm4, xmm6);
                        vunpckhpd(xmm1, xmm4, xmm6);
                        vmovups(ptr[AO1
                                        + (unroll_m * 0 + i * 4 - OFFSET)
                                                * SIZE],
                                xmm0);
                        vmovups(ptr[AO1
                                        + (unroll_m * 1 + i * 4 - OFFSET)
                                                * SIZE],
                                xmm1);
                        vunpcklpd(xmm0, xmm5, xmm2);
                        vunpckhpd(xmm1, xmm5, xmm2);
                        vmovups(ptr[AO1
                                        + (unroll_m * 2 + i * 4 - OFFSET)
                                                * SIZE],
                                xmm0);
                        vmovups(ptr[AO1
                                        + (unroll_m * 3 + i * 4 - OFFSET)
                                                * SIZE],
                                xmm1);
                    }
                } else if (is_avx2) {
                    for (int i = 0; i < 2; i++) {
                        vmovaps(xmm4, xmm3);
                        vgatherqps(xmm0,
                                ptr[BO1 + ymm7 + ((2 * i) - OFFSET) * SIZE],
                                xmm4);
                        vmovaps(xmm4, xmm3);
                        vgatherqps(xmm1,
                                ptr[BO1 + ymm7 + ((2 * i + 1) - OFFSET) * SIZE],
                                xmm4);

                        vmovups(ptr[AO1
                                        + (unroll_m * (2 * i) + 0 * 4 - OFFSET)
                                                * SIZE],
                                xmm0);
                        vmovups(ptr[AO1
                                        + (unroll_m * (2 * i + 1) + 0 * 4
                                                  - OFFSET)
                                                * SIZE],
                                xmm1);
                    }

                    lea(BO2, ptr[BO1 + LDA * 4]);

                    for (int i = 0; i < 2; i++) {
                        vextractf128(xmm4, ymm3, 1);
                        vgatherqps(xmm0,
                                ptr[BO2 + ymm7 + ((2 * i) - OFFSET) * SIZE],
                                xmm4);
                        vextractf128(xmm4, ymm3, 1);
                        vgatherqps(xmm1,
                                ptr[BO2 + ymm7 + ((2 * i + 1) - OFFSET) * SIZE],
                                xmm4);

                        vmovups(ptr[AO1
                                        + (unroll_m * (2 * i) + 1 * 4 - OFFSET)
                                                * SIZE],
                                xmm0);
                        vmovups(ptr[AO1
                                        + (unroll_m * (2 * i + 1) + 1 * 4
                                                  - OFFSET)
                                                * SIZE],
                                xmm1);
                    }

                    lea(BO2, ptr[BO2 + LDA * 4]);
                } else {
                    vxorps(xmm4, xmm4, xmm4);
                    lea(BO2, ptr[BO1 + LDA * 4]);

                    auto el_cp = [&](int section, int ld_step) {
                        RegExp src_addr = section == 0 ? BO1 : BO2;
                        if (ld_step == 1 || ld_step == 2)
                            src_addr = src_addr + LDA * ld_step;
                        else if (ld_step == 3)
                            src_addr = src_addr + CO1;
                        src_addr = src_addr - OFFSET * SIZE;

                        vmovups(Xmm(ld_step % 2), ptr[src_addr]);
                        RegExp dst_addr = AO1
                            + (ld_step + section * 4 - OFFSET) * SIZE;
                        for (int off = 0; off < 4; ++off)
                            pextrd(ptr[dst_addr + unroll_m * off * SIZE],
                                    Xmm(ld_step % 2), off);
                    };

                    Label l_end;
                    el_cp(0, 0); cmp(M, 4 * 0 + 0 + 1); je(l_end, T_NEAR);
                    el_cp(0, 1); cmp(M, 4 * 0 + 1 + 1); je(l_end, T_NEAR);
                    el_cp(0, 2); cmp(M, 4 * 0 + 2 + 1); je(l_end, T_NEAR);
                    el_cp(0, 3); cmp(M, 4 * 0 + 3 + 1); je(l_end, T_NEAR);
                    el_cp(1, 0); cmp(M, 4 * 1 + 0 + 1); je(l_end, T_NEAR);
                    el_cp(1, 1); cmp(M, 4 * 1 + 1 + 1); je(l_end, T_NEAR);
                    el_cp(1, 2);
                    L(l_end);

                    lea(BO2, ptr[BO2 + LDA * 4]);
                }

                if (unroll_m >= 16) {
                    assert(is_avx2);
                    if (isLoad2Unmasked) {
                        for (int i = 0; i < 2; i++) {
                            vmovups(xmm0, ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                            vmovups(xmm1, ptr[BO2 + LDA * 1
                                                  + (0 * 8 - OFFSET) * SIZE]);
                            lea(BO2, ptr[BO2 + LDA * 2]);
                            vunpcklps(xmm4, xmm0, xmm1);
                            vunpckhps(xmm5, xmm0, xmm1);
                            vmovups(xmm0, ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                            vmovups(xmm1, ptr[BO2 + LDA * 1
                                                  + (0 * 8 - OFFSET) * SIZE]);
                            if (i == 0)
                                lea(BO2, ptr[BO2 + LDA * 2]);
                            vunpcklps(xmm6, xmm0, xmm1);
                            vunpckhps(xmm2, xmm0, xmm1);

                            vunpcklpd(xmm0, xmm4, xmm6);
                            vunpckhpd(xmm1, xmm4, xmm6);
                            vmovups(ptr[AO1
                                            + (unroll_m * 0 + (i + 2) * 4
                                                      - OFFSET)
                                                    * SIZE],
                                    xmm0);
                            vmovups(ptr[AO1
                                            + (unroll_m * 1 + (i + 2) * 4
                                                      - OFFSET)
                                                    * SIZE],
                                    xmm1);
                            vunpcklpd(xmm0, xmm5, xmm2);
                            vunpckhpd(xmm1, xmm5, xmm2);
                            vmovups(ptr[AO1
                                            + (unroll_m * 2 + (i + 2) * 4
                                                      - OFFSET)
                                                    * SIZE],
                                    xmm0);
                            vmovups(ptr[AO1
                                            + (unroll_m * 3 + (i + 2) * 4
                                                      - OFFSET)
                                                    * SIZE],
                                    xmm1);
                        }
                    } else {
                        for (int i = 0; i < 2; i++) {
                            vmovaps(xmm4, xmm3);
                            vgatherqps(xmm0,
                                    ptr[BO2 + ymm7 + ((2 * i) - OFFSET) * SIZE],
                                    xmm4);
                            vmovaps(xmm4, xmm3);
                            vgatherqps(xmm1,
                                    ptr[BO2 + ymm7
                                               + ((2 * i + 1) - OFFSET) * SIZE],
                                    xmm4);

                            vmovups(ptr[AO1
                                            + (unroll_m * (2 * i) + 2 * 4
                                                      - OFFSET)
                                                    * SIZE],
                                    xmm0);
                            vmovups(ptr[AO1
                                            + (unroll_m * (2 * i + 1) + 2 * 4
                                                      - OFFSET)
                                                    * SIZE],
                                    xmm1);
                        }

                        lea(BO2, ptr[BO2 + LDA * 4]);

                        for (int i = 0; i < 2; i++) {
                            vextractf128(xmm4, ymm3, 1);
                            vgatherqps(xmm0,
                                    ptr[BO2 + ymm7 + ((2 * i) - OFFSET) * SIZE],
                                    xmm4);
                            vextractf128(xmm4, ymm3, 1);
                            vgatherqps(xmm1,
                                    ptr[BO2 + ymm7
                                               + ((2 * i + 1) - OFFSET) * SIZE],
                                    xmm4);

                            vmovups(ptr[AO1
                                            + (unroll_m * (2 * i) + 3 * 4
                                                      - OFFSET)
                                                    * SIZE],
                                    xmm0);
                            vmovups(ptr[AO1
                                            + (unroll_m * (2 * i + 1) + 3 * 4
                                                      - OFFSET)
                                                    * SIZE],
                                    xmm1);
                        }

                        lea(BO2, ptr[BO2 + LDA * 4]);
                    }
                }
                add(BO1, (4 * SIZE));
            }

            add(AO1, unroll_m * 4 * SIZE);
            sub(LL, 1);
            jg(pack2, T_NEAR);
            align(16);

            L(pack3);
            mov(LL, K);
            and_(LL, 3);
            jle(pack10, T_NEAR);
            align(16);

            L(pack4);
            if (!isTransA) {
                if (isLoad1Unmasked) {
                    vmovups(ymm4, ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm4, VMASK, ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                }
                if (unroll_m > 8) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm5, ptr[BO1 + (1 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(ymm5, VMASK,
                                ptr[BO1 + (1 + 8 - OFFSET) * SIZE]);
                    }
                }
                add(BO1, LDA);
                vmovups(ptr[AO1 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE],
                        ymm4);
                if (unroll_m > 8) {
                    vmovups(ptr[AO1 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE],
                            ymm5);
                }
            } else {
                if (isLoad1Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        reg = (i % 2 == 0) ? BO1 : BO2;
                        vmovss(Xmm(i + 1), ptr[reg + (0 * 8 - OFFSET) * SIZE]);
                        vmovss(xmm0,
                                ptr[reg + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        lea(BO2, ptr[reg + LDA * 2]);
                        vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                    }
                    vunpcklpd(xmm1, xmm1, xmm2);
                    vmovups(ptr[AO1 + (unroll_m * 0 + 0 * 4 - OFFSET) * SIZE],
                            xmm1);

                    for (int i = 0; i < 2; i++) {
                        vmovss(Xmm(i + 1), ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        vmovss(xmm0,
                                ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        lea(BO2, ptr[BO2 + LDA * 2]);
                        vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                    }
                    vunpcklpd(xmm1, xmm1, xmm2);
                    vmovups(ptr[AO1 + (unroll_m * 0 + 1 * 4 - OFFSET) * SIZE],
                            xmm1);
                } else if (is_avx2) {
                    vmovaps(xmm4, xmm3);
                    vgatherqps(xmm1, ptr[BO1 + ymm7 + (0 * 8 - OFFSET) * SIZE],
                            xmm4);
                    lea(BO2, ptr[BO1 + LDA * 4]);
                    vmovups(ptr[AO1 + (unroll_m * 0 + 0 * 4 - OFFSET) * SIZE],
                            xmm1);

                    vextractf128(xmm4, ymm3, 1);
                    vgatherqps(xmm1, ptr[BO2 + ymm7 + (0 * 8 - OFFSET) * SIZE],
                            xmm4);
                    lea(BO2, ptr[BO2 + LDA * 4]);
                    vmovups(ptr[AO1 + (unroll_m * 0 + 1 * 4 - OFFSET) * SIZE],
                            xmm1);
                } else {
                    vxorps(xmm4, xmm4, xmm4);
                    lea(BO2, ptr[BO1 + LDA * 4]);

                    auto el_cp = [&](int section, int ld_step) {
                        RegExp src_addr = section == 0 ? BO1 : BO2;
                        if (ld_step == 1 || ld_step == 2)
                            src_addr = src_addr + LDA * ld_step;
                        else if (ld_step == 3)
                            src_addr = src_addr + CO1;
                        src_addr = src_addr - OFFSET * SIZE;

                        vmovss(xmm1, ptr[src_addr]);
                        RegExp dst_addr = AO1
                            + (ld_step + section * 4 - OFFSET) * SIZE;
                        movss(ptr[dst_addr], xmm1);
                    };

                    Label l_end;
                    el_cp(0, 0); cmp(M, 4 * 0 + 0 + 1); je(l_end, T_NEAR);
                    el_cp(0, 1); cmp(M, 4 * 0 + 1 + 1); je(l_end, T_NEAR);
                    el_cp(0, 2); cmp(M, 4 * 0 + 2 + 1); je(l_end, T_NEAR);
                    el_cp(0, 3); cmp(M, 4 * 0 + 3 + 1); je(l_end, T_NEAR);
                    el_cp(1, 0); cmp(M, 4 * 1 + 0 + 1); je(l_end, T_NEAR);
                    el_cp(1, 1); cmp(M, 4 * 1 + 1 + 1); je(l_end, T_NEAR);
                    el_cp(1, 2);
                    L(l_end);

                    lea(BO2, ptr[BO2 + LDA * 4]);
                }

                if (unroll_m >= 16) {
                    assert(is_avx2);
                    if (isLoad2Unmasked) {
                        for (int i = 0; i < 2; i++) {
                            vmovss(Xmm(i + 1),
                                    ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                            vmovss(xmm0, ptr[BO2 + LDA * 1
                                                 + (0 * 8 - OFFSET) * SIZE]);
                            lea(BO2, ptr[BO2 + LDA * 2]);
                            vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                        }
                        vunpcklpd(xmm1, xmm1, xmm2);
                    } else {
                        vmovaps(xmm4, xmm3);
                        vgatherqps(xmm1,
                                ptr[BO2 + ymm7 + (0 * 8 - OFFSET) * SIZE],
                                xmm4);
                        lea(BO2, ptr[BO2 + LDA * 4]);
                    }
                    vmovups(ptr[AO1 + (unroll_m * 0 + 2 * 4 - OFFSET) * SIZE],
                            xmm1);

                    if (isLoad2Unmasked) {
                        for (int i = 0; i < 2; i++) {
                            vmovss(Xmm(i + 1),
                                    ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                            vmovss(xmm0, ptr[BO2 + LDA * 1
                                                 + (0 * 8 - OFFSET) * SIZE]);
                            lea(BO2, ptr[BO2 + LDA * 2]);
                            vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                        }
                        vunpcklpd(xmm1, xmm1, xmm2);
                    } else {
                        vextractf128(xmm4, ymm3, 1);
                        vgatherqps(xmm1,
                                ptr[BO2 + ymm7 + (0 * 8 - OFFSET) * SIZE],
                                xmm4);
                    }
                    vmovups(ptr[AO1 + (unroll_m * 0 + 3 * 4 - OFFSET) * SIZE],
                            xmm1);
                }
                add(BO1, SIZE);
            }

            add(AO1, unroll_m * SIZE);
            sub(LL, 1);
            jg(pack4, T_NEAR);
            align(16);

            L(pack10);
        };

        // Fused multiply add; may become one or two instructions
        auto fma = [&](bool useFma, Ymm reg0, Ymm reg1, Ymm reg2,
                bool overWrite = false) {
            if (useFma) {
                if (is_avx2) {
                    vfmadd231ps(reg2, reg1, reg0);
                } else {
                    assert(UNROLL_M == 8);
                    auto tent_vreg = overWrite ? reg1 : ymm1;
                    vmulps(tent_vreg, reg1, reg0);
                    vaddps(reg2, reg2, tent_vreg);
                }
            } else {
                if (!overWrite) {
                    vmulps(ymm15, reg1, reg0);
                    vaddps(reg2, reg2, ymm15);
                } else {
                    vmulps(reg1, reg1, reg0);
                    vaddps(reg2, reg2, reg1);
                }
            }
        };

        // Inner kernel with k=8
        auto innerkernel8 = [&](int unroll_m, int unroll_n,
                bool isLoad1Unmasked, bool isLoad2Unmasked, bool isDirect,
                bool isCopy, bool useFma, Ymm reg00, Ymm reg01, Ymm reg02,
                Ymm reg03, Ymm reg04, Ymm reg05, Ymm reg06, Ymm reg07,
                Ymm reg08, Ymm reg09, Ymm reg10, Ymm reg11, Ymm reg12,
                Ymm reg13, Ymm reg14, Ymm reg15, Ymm reg16, Ymm reg17,
                Ymm reg18, Ymm reg19, Ymm reg20, Ymm reg21, Ymm reg22,
                Ymm reg23) {

            Ymm fmareg;

            if (!isDirect) {
                prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 0) * SIZE]);
            } else {
                prefetcht0(ptr[AO1 + LDA4]);
            }

            for (int i = 0; i < 8; i++) {
                if (isDirect) {
                    if (isLoad1Unmasked) {
                        vmovups(ymm0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(ymm0, VMASK,
                                ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    }
                    if (unroll_m >= 16) {
                        if (isLoad2Unmasked) {
                            vmovups(ymm1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        } else {
                            vmaskmovps(ymm1, VMASK,
                                    ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        }
                    }
                    add(AO1, LDA);
                }

                if (!isTransB) {
                    vbroadcastss(ymm2, ptr[BO1 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg00 : reg12;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg06 : reg18;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
                if (i == 0) {
                    if (!isTransB) {
                        prefetcht0(ptr[BO1 + PREFETCHSIZEB * SIZE]);
                    }
                }
                if (unroll_n >= 2) {
                    if (!isTransB) {
                        if (i == 1) {
                            prefetcht0(ptr[BO1 + LDB + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(
                                ymm2, ptr[BO1 + LDB * 1 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg01 : reg13;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg07 : reg19;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (isCopy) {
                    vmovups(ptr[LDA4 + (unroll_m * i + 0 * 8 - OFFSET) * SIZE],
                            ymm0);
                    if (unroll_m >= 16) {
                        vmovups(ptr[LDA4
                                        + (unroll_m * i + 1 * 8 - OFFSET)
                                                * SIZE],
                                ymm1);
                    }
                    if (i == 7) {
                        sub(LDA4, -unroll_m * 8 * SIZE);
                    }
                }

                if (unroll_n >= 3) {
                    if (!isTransB) {
                        if (i == 2) {
                            prefetcht0(
                                    ptr[BO1 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(
                                ymm2, ptr[BO1 + LDB * 2 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg02 : reg14;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg08 : reg20;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (i == 7) {
                    if (!isTransB) {
                        sub(BO1, -8 * SIZE);
                    }
                }

                if (unroll_n >= 4) {
                    if (!isTransB) {
                        if (i == 3) {
                            prefetcht0(ptr[BO2 + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(ymm2, ptr[BO2 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg03 : reg15;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg09 : reg21;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (unroll_n >= 5) {
                    if (!isTransB) {
                        if (i == 4) {
                            prefetcht0(ptr[BO2 + LDB + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(
                                ymm2, ptr[BO2 + LDB * 1 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg04 : reg16;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg10 : reg22;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (unroll_n >= 6) {
                    if (!isTransB) {
                        if (i == 5) {
                            prefetcht0(
                                    ptr[BO2 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(
                                ymm2, ptr[BO2 + LDB * 2 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg05 : reg17;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg11 : reg23;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }
                if (isTransB) {
                    prefetcht0(ptr[BO1 + BO2]);
                    add(BO1, LDB);
                }

                if (i == 0) {
                    if (unroll_m >= 4) {
                        if (!isDirect) {
                            prefetcht0(
                                    ptr[AO1 + (PREFETCHSIZEA + 2 * 8) * SIZE]);
                        } else {
                            prefetcht0(ptr[AO1 + LDA4]);
                        }
                    }
                }
                if (i == 1 || i == 2) {
                    if (unroll_m >= 8) {
                        if (!isDirect) {
                            prefetcht0(ptr[AO1
                                    + (PREFETCHSIZEA + (2 + 2 * i) * 8)
                                            * SIZE]);
                        } else {
                            prefetcht0(ptr[AO1 + LDA4]);
                        }
                    }
                }
                if (i == 3 || i == 4 || i == 5 || i == 6) {
                    if (unroll_m >= 16) {
                        if (!isDirect) {
                            prefetcht0(ptr[AO1
                                    + (PREFETCHSIZEA + (2 + 2 * i) * 8)
                                            * SIZE]);
                        } else {
                            prefetcht0(ptr[AO1 + LDA4]);
                        }
                    }
                }
                if (i == 7) {
                    if (!isTransB) {
                        if (unroll_n >= 4) {
                            sub(BO2, -8 * SIZE);
                        }
                    }
                    if (!isTransA) {
                        prefetcht2(ptr[AA]);
                        lea(AA, ptr[AA + LDA]);
                    }
                }

                if (!isDirect) {
                    if (isLoad1Unmasked) {
                        vmovups(ymm0,
                                ptr[AO1
                                        + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                                                * SIZE]);
                    } else {
                        vmaskmovps(
                                ymm0, VMASK,
                                ptr[AO1
                                        + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                                                * SIZE]);
                    }
                    if (unroll_m >= 16) {
                        if (isLoad2Unmasked) {
                            vmovups(ymm1, ptr[AO1
                                                  + (unroll_m * (i + 1) + 1 * 8
                                                            - OFFSET)
                                                          * SIZE]);
                        } else {
                            vmaskmovps(ymm1, VMASK,
                                    ptr[AO1
                                               + (unroll_m * (i + 1) + 1 * 8
                                                         - OFFSET)
                                                       * SIZE]);
                        }
                    }
                }
            }

            if (!isDirect) {
                sub(AO1, -unroll_m * 8 * SIZE);
            }
            sub(LL, 1);

        };

        // Inner kernel with k=4
        auto innerkernel4 = [&](int unroll_m, int unroll_n,
                bool isLoad1Unmasked, bool isLoad2Unmasked, bool isDirect,
                bool isCopy, bool useFma, Ymm reg00, Ymm reg01, Ymm reg02,
                Ymm reg03, Ymm reg04, Ymm reg05, Ymm reg06, Ymm reg07,
                Ymm reg08, Ymm reg09, Ymm reg10, Ymm reg11, Ymm reg12,
                Ymm reg13, Ymm reg14, Ymm reg15, Ymm reg16, Ymm reg17,
                Ymm reg18, Ymm reg19, Ymm reg20, Ymm reg21, Ymm reg22,
                Ymm reg23) {

            Ymm fmareg;

            if (!isDirect) {
                prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 0) * SIZE]);
            } else {
                prefetcht0(ptr[AO1 + LDA4]);
            }

            for (int i = 0; i < 4; i++) {
                if (isDirect) {
                    if (isLoad1Unmasked) {
                        vmovups(ymm0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(ymm0, VMASK,
                                ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    }
                    if (unroll_m >= 16) {
                        if (isLoad2Unmasked) {
                            vmovups(ymm1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        } else {
                            vmaskmovps(ymm1, VMASK,
                                    ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        }
                    }
                    add(AO1, LDA);
                }

                if (!isTransB) {
                    vbroadcastss(ymm2, ptr[BO1 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg00 : reg12;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg06 : reg18;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
                if (i == 0) {
                    if (!isTransB) {
                        prefetcht0(ptr[BO1 + PREFETCHSIZEB * SIZE]);
                    }
                }
                if (unroll_n >= 2) {
                    if (!isTransB) {
                        if (i == 1) {
                            prefetcht0(ptr[BO1 + LDB + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(
                                ymm2, ptr[BO1 + LDB * 1 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg01 : reg13;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg07 : reg19;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (isCopy) {
                    vmovups(ptr[LDA4 + (unroll_m * i + 0 * 8 - OFFSET) * SIZE],
                            ymm0);
                    if (unroll_m >= 16) {
                        vmovups(ptr[LDA4
                                        + (unroll_m * i + 1 * 8 - OFFSET)
                                                * SIZE],
                                ymm1);
                    }
                    if (i == 3) {
                        sub(LDA4, -unroll_m * 4 * SIZE);
                    }
                }

                if (unroll_n >= 3) {
                    if (!isTransB) {
                        if (i == 2) {
                            prefetcht0(
                                    ptr[BO1 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(
                                ymm2, ptr[BO1 + LDB * 2 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg02 : reg14;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg08 : reg20;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (i == 7) {
                    if (!isTransB) {
                        sub(BO1, -8 * SIZE);
                    }
                }

                if (unroll_n >= 4) {
                    if (!isTransB) {
                        if (i == 3) {
                            prefetcht0(ptr[BO2 + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(ymm2, ptr[BO2 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg03 : reg15;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg09 : reg21;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (unroll_n >= 5) {
                    if (!isTransB) {
                        if (i == 4) {
                            prefetcht0(ptr[BO2 + LDB + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(
                                ymm2, ptr[BO2 + LDB * 1 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg04 : reg16;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg10 : reg22;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (unroll_n >= 6) {
                    if (!isTransB) {
                        if (i == 5) {
                            prefetcht0(
                                    ptr[BO2 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(
                                ymm2, ptr[BO2 + LDB * 2 + (i - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg05 : reg17;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg11 : reg23;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }
                if (isTransB) {
                    prefetcht0(ptr[BO1 + BO2]);
                    add(BO1, LDB);
                }

                if (i == 0) {
                    if (unroll_m >= 4) {
                        if (!isDirect) {
                            prefetcht0(
                                    ptr[AO1 + (PREFETCHSIZEA + 2 * 8) * SIZE]);
                        } else {
                            prefetcht0(ptr[AO1 + LDA4]);
                        }
                    }
                }
                if (i == 1 || i == 2) {
                    if (unroll_m >= 8) {
                        if (!isDirect) {
                            prefetcht0(ptr[AO1
                                    + (PREFETCHSIZEA + (2 + 2 * i) * 8)
                                            * SIZE]);
                        } else {
                            prefetcht0(ptr[AO1 + LDA4]);
                        }
                    }
                }
                if (i == 3) {
                    if (!isTransB) {
                        sub(BO1, -4 * SIZE);
                        if (unroll_n >= 4) {
                            sub(BO2, -4 * SIZE);
                        }
                    }
                }

                if (!isDirect) {
                    if (isLoad1Unmasked) {
                        vmovups(ymm0,
                                ptr[AO1
                                        + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                                                * SIZE]);
                    } else {
                        vmaskmovps(
                                ymm0, VMASK,
                                ptr[AO1
                                        + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                                                * SIZE]);
                    }
                    if (unroll_m >= 16) {
                        if (isLoad2Unmasked) {
                            vmovups(ymm1, ptr[AO1
                                                  + (unroll_m * (i + 1) + 1 * 8
                                                            - OFFSET)
                                                          * SIZE]);
                        } else {
                            vmaskmovps(ymm1, VMASK,
                                    ptr[AO1
                                               + (unroll_m * (i + 1) + 1 * 8
                                                         - OFFSET)
                                                       * SIZE]);
                        }
                    }
                }
            }

            if (!isDirect) {
                sub(AO1, -unroll_m * 4 * SIZE);
            }

        };

        // Inner kernel with k=2
        auto innerkernel2 = [&](int unroll_m, int unroll_n,
                bool isLoad1Unmasked, bool isLoad2Unmasked, bool isDirect,
                bool isCopy, bool useFma, Ymm reg00, Ymm reg01, Ymm reg02,
                Ymm reg03, Ymm reg04, Ymm reg05, Ymm reg06, Ymm reg07,
                Ymm reg08, Ymm reg09, Ymm reg10, Ymm reg11, Ymm reg12,
                Ymm reg13, Ymm reg14, Ymm reg15, Ymm reg16, Ymm reg17,
                Ymm reg18, Ymm reg19, Ymm reg20, Ymm reg21, Ymm reg22,
                Ymm reg23) {

            Ymm fmareg;

            for (int i = 0; i < 2; i++) {
                if (isDirect) {
                    if (isLoad1Unmasked) {
                        vmovups(ymm0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(ymm0, VMASK,
                                ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                    }
                    if (unroll_m >= 16) {
                        if (isLoad2Unmasked) {
                            vmovups(ymm1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        } else {
                            vmaskmovps(ymm1, VMASK,
                                    ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                        }
                    }
                    add(AO1, LDA);
                }

                if (!isTransB) {
                    vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg00 : reg12;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg06 : reg18;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
                if (unroll_n >= 2) {
                    if (!isTransB) {
                        vbroadcastss(
                                ymm2, ptr[BO1 + LDB * 1 + (0 - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg01 : reg13;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg07 : reg19;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (unroll_n >= 3) {
                    if (!isTransB) {
                        if (i == 2) {
                            prefetcht0(
                                    ptr[BO1 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                        }
                        vbroadcastss(
                                ymm2, ptr[BO1 + LDB * 2 + (0 - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg02 : reg14;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg08 : reg20;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (unroll_n >= 4) {
                    if (!isTransB) {
                        vbroadcastss(ymm2, ptr[BO2 + (0 - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg03 : reg15;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg09 : reg21;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (unroll_n >= 5) {
                    if (!isTransB) {
                        vbroadcastss(
                                ymm2, ptr[BO2 + LDB * 1 + (0 - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg04 : reg16;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg10 : reg22;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (unroll_n >= 6) {
                    if (!isTransB) {
                        vbroadcastss(
                                ymm2, ptr[BO2 + LDB * 2 + (0 - OFFSET) * SIZE]);
                    } else {
                        vbroadcastss(ymm2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                    }
                    fmareg = (i % 2 == 0) ? reg05 : reg17;
                    fma(useFma, ymm0, ymm2, fmareg);
                    if (unroll_m >= 16) {
                        fmareg = (i % 2 == 0) ? reg11 : reg23;
                        fma(useFma, ymm1, ymm2, fmareg);
                    }
                }

                if (isCopy) {
                    vmovups(ptr[LDA4 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE],
                            ymm0);
                    if (unroll_m >= 16) {
                        vmovups(ptr[LDA4
                                        + (unroll_m * 0 + 1 * 8 - OFFSET)
                                                * SIZE],
                                ymm1);
                    }
                    sub(LDA4, -unroll_m * SIZE);
                }

                if (!isDirect) {
                    if (isLoad1Unmasked) {
                        vmovups(ymm0, ptr[AO1
                                              + (unroll_m * 1 + 0 * 8 - OFFSET)
                                                      * SIZE]);
                    } else {
                        vmaskmovps(ymm0, VMASK,
                                ptr[AO1
                                           + (unroll_m * 1 + 0 * 8 - OFFSET)
                                                   * SIZE]);
                    }
                    if (unroll_m >= 16) {
                        if (isLoad2Unmasked) {
                            vmovups(ymm1,
                                    ptr[AO1
                                            + (unroll_m * 1 + 1 * 8 - OFFSET)
                                                    * SIZE]);
                        } else {
                            vmaskmovps(ymm1, VMASK,
                                    ptr[AO1
                                               + (unroll_m * 1 + 1 * 8 - OFFSET)
                                                       * SIZE]);
                        }
                    }
                    sub(AO1, -unroll_m * SIZE);
                }

                if (!isTransB) {
                    sub(BO1, -SIZE);
                    if (unroll_n >= 4) {
                        sub(BO2, -SIZE);
                    }
                } else {
                    add(BO1, LDB);
                }
            }

        };

        // Inner kernel with k=1
        auto innerkernel1 = [&](int unroll_m, int unroll_n,
                bool isLoad1Unmasked, bool isLoad2Unmasked, bool isDirect,
                bool isCopy, bool useFma, Ymm reg00, Ymm reg01, Ymm reg02,
                Ymm reg03, Ymm reg04, Ymm reg05, Ymm reg06, Ymm reg07,
                Ymm reg08, Ymm reg09, Ymm reg10, Ymm reg11) {

            if (isDirect) {
                if (isLoad1Unmasked) {
                    vmovups(ymm0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm0, VMASK, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(ymm1, VMASK,
                                ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    }
                }
                add(AO1, LDA);
            }

            if (!isTransB) {
                vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
            } else {
                vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
            }
            fma(useFma, ymm0, ymm2, reg00);
            if (unroll_m >= 16) {
                fma(useFma, ymm1, ymm2, reg06);
            }

            if (unroll_n >= 2) {
                if (!isTransB) {
                    vbroadcastss(
                            ymm2, ptr[BO1 + LDB * 1 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                }
                fma(useFma, ymm0, ymm2, reg01);
                if (unroll_m >= 16) {
                    fma(useFma, ymm1, ymm2, reg07);
                }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    vbroadcastss(
                            ymm2, ptr[BO1 + LDB * 2 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                }
                fma(useFma, ymm0, ymm2, reg02);
                if (unroll_m >= 16) {
                    fma(useFma, ymm1, ymm2, reg08);
                }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    vbroadcastss(ymm2, ptr[BO2 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                }
                fma(useFma, ymm0, ymm2, reg03);
                if (unroll_m >= 16) {
                    fma(useFma, ymm1, ymm2, reg09);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    vbroadcastss(
                            ymm2, ptr[BO2 + LDB * 1 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                }
                fma(useFma, ymm0, ymm2, reg04);
                if (unroll_m >= 16) {
                    fma(useFma, ymm1, ymm2, reg10);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    vbroadcastss(
                            ymm2, ptr[BO2 + LDB * 2 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                }
                fma(useFma, ymm0, ymm2, reg05);
                if (unroll_m >= 16) {
                    fma(useFma, ymm1, ymm2, reg11);
                }
            }

            if (isCopy) {
                vmovups(ptr[LDA4 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE],
                        ymm0);
                if (unroll_m >= 16) {
                    vmovups(ptr[LDA4 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE],
                            ymm1);
                }
                sub(LDA4, -unroll_m * SIZE);
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    vmovups(ymm0,
                            ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm0, VMASK,
                            ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm1, ptr[AO1
                                              + (unroll_m * 1 + 1 * 8 - OFFSET)
                                                      * SIZE]);
                    } else {
                        vmaskmovps(ymm1, VMASK,
                                ptr[AO1
                                           + (unroll_m * 1 + 1 * 8 - OFFSET)
                                                   * SIZE]);
                    }
                }
                sub(AO1, -unroll_m * SIZE);
            }

            if (!isTransB) {
                sub(BO1, -SIZE);
                if (unroll_n >= 4) {
                    sub(BO2, -SIZE);
                }
            } else {
                add(BO1, LDB);
            }

        };

        // Main kernel; does prefetching and calls innerkernel{1,2,4,8} as
        // appropriate
        // After calculating results in registers, writes back to C matrix
        auto kernel = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
                Ymm reg00 = Ymm(4), Ymm reg01 = Ymm(5), Ymm reg02 = Ymm(6),
                Ymm reg03 = Ymm(7), Ymm reg04 = Ymm(8), Ymm reg05 = Ymm(9),
                Ymm reg06 = Ymm(10), Ymm reg07 = Ymm(11), Ymm reg08 = Ymm(12),
                Ymm reg09 = Ymm(13), Ymm reg10 = Ymm(14), Ymm reg11 = Ymm(15),
                Ymm reg12 = Ymm(4), Ymm reg13 = Ymm(5), Ymm reg14 = Ymm(6),
                Ymm reg15 = Ymm(7), Ymm reg16 = Ymm(8), Ymm reg17 = Ymm(9),
                Ymm reg18 = Ymm(10), Ymm reg19 = Ymm(11), Ymm reg20 = Ymm(12),
                Ymm reg21 = Ymm(13), Ymm reg22 = Ymm(14), Ymm reg23 = Ymm(15)) {
            if (!isDirect) {
                lea(AO1, ptr[rsp + 256 + OFFSET * SIZE]);
            } else {
                mov(AO1, A);
            }

            if (isCopy) {
                lea(LDA4, ptr[rsp + 256 + OFFSET * SIZE]);
            } else {
                lea(LDA4, ptr[LDA * 8 + (8 - 1 - OFFSET) * SIZE]);
            }

            if (isTransB) {
                lea(BO2, ptr[LDB * 4 + (8 - 1 - OFFSET) * SIZE]);
                lea(BO2, ptr[BO2 + LDB * 2]);
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    vmovups(ymm0,
                            ptr[AO1 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm0, VMASK,
                            ptr[AO1 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE]);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm1, ptr[AO1
                                              + (unroll_m * 0 + 1 * 8 - OFFSET)
                                                      * SIZE]);
                    } else {
                        vmaskmovps(ymm1, VMASK,
                                ptr[AO1
                                           + (unroll_m * 0 + 1 * 8 - OFFSET)
                                                   * SIZE]);
                    }
                }
            }

            for (int i = 4; i < 10; i++) {
                vxorps(Ymm(i), Ymm(i), Ymm(i));
                vxorps(Ymm(i + 6), Ymm(i + 6), Ymm(i + 6));
            }

            mov(LL, K);
            sar(LL, 3);

            Label kernel12, kernel13, kernel14, kernel15;
            Label kernel16, kernel17, kernel18;

            sub(LL, SECOND_FETCH);
            jle(kernel13, T_NEAR);
            align(16);

            L(kernel12);
            innerkernel8(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                    reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12,
                    reg13, reg14, reg15, reg16, reg17, reg18, reg19, reg20,
                    reg21, reg22, reg23);
            jg(kernel12, T_NEAR);
            align(16);

            L(kernel13);
            prefetcht0(ptr[CO1 + (unroll_m - 1) * SIZE]);
            if (unroll_n >= 2)
                prefetcht0(ptr[CO1 + LDC + (unroll_m - 1) * SIZE]);
            if (unroll_n >= 3)
                prefetcht0(ptr[CO1 + LDC * 2 + (unroll_m - 1) * SIZE]);
            if (unroll_n >= 4)
                prefetcht0(ptr[CO2 + (unroll_m - 1) * SIZE]);
            if (unroll_n >= 5)
                prefetcht0(ptr[CO2 + LDC + (unroll_m - 1) * SIZE]);
            if (unroll_n >= 6)
                prefetcht0(ptr[CO2 + LDC * 2 + (unroll_m - 1) * SIZE]);

            add(LL, SECOND_FETCH);
            jle(kernel15, T_NEAR);
            align(16);

            L(kernel14);
            innerkernel8(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                    reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12,
                    reg13, reg14, reg15, reg16, reg17, reg18, reg19, reg20,
                    reg21, reg22, reg23);
            jg(kernel14, T_NEAR);
            align(16);

            L(kernel15);
            test(K, 4);
            jle(kernel16, T_NEAR);
            innerkernel4(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                    reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12,
                    reg13, reg14, reg15, reg16, reg17, reg18, reg19, reg20,
                    reg21, reg22, reg23);

            L(kernel16);
            test(K, 2);
            jle(kernel17, T_NEAR);
            innerkernel2(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                    reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12,
                    reg13, reg14, reg15, reg16, reg17, reg18, reg19, reg20,
                    reg21, reg22, reg23);
            align(16);

            L(kernel17);
            if (unroll_m == 16) {
                if (unroll_n <= 3) {
                    vaddps(reg00, reg00, reg12);
                    vaddps(reg01, reg01, reg13);
                    vaddps(reg02, reg02, reg14);
                    vaddps(reg06, reg06, reg18);
                    vaddps(reg07, reg07, reg19);
                    vaddps(reg08, reg08, reg20);
                }
            }

            if (unroll_m <= 8) {
                vaddps(reg00, reg00, reg12);
                vaddps(reg01, reg01, reg13);
                vaddps(reg02, reg02, reg14);
                vaddps(reg03, reg03, reg15);
                vaddps(reg04, reg04, reg16);
                vaddps(reg05, reg05, reg17);
            }

            test(K, 1);
            jle(kernel18, T_NEAR);
            innerkernel1(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                    reg05, reg06, reg07, reg08, reg09, reg10, reg11);
            align(16);

            L(kernel18);
            vbroadcastss(VALPHA, ALPHA);

            if (isBetaN) {
                vbroadcastss(VBETA, BETA);
            }

            // Write back the results; all beta and bias cases need to be
            // handled
            switch (unroll_n) {
            case 1: mov(rax, LDC); break;
            case 2: lea(rax, ptr[LDC * 2]); break;
            case 3: lea(rax, ptr[LDC + LDC * 2]); break;
            case 4: lea(rax, ptr[LDC + LDC * 4]); break;
            case 5:
                lea(rax, ptr[LDC * 4]);
                add(rax, LDC);
                break;
            case 6:
                lea(rax, ptr[LDC + LDC * 2]);
                add(rax, rax);
                break;
            }

            if (hasBias) {
                mov(BIAS1, BIAS);
                if (isLoad1Unmasked) {
                    vmovups(VBIAS1, ptr[BIAS1 + 0 * SIZE]);
                } else {
                    vmaskmovps(VBIAS1, VMASK, ptr[BIAS1 + 0 * SIZE]);
                }
            }

            for (int i = 0; i < unroll_n; i++) {
                vmulps(Ymm(i + 4), Ymm(i + 4), VALPHA);
                if (!isBeta0) {
                    if (isLoad1Unmasked) {
                        switch (i) {
                        case 0: vmovups(ymm0, ptr[CO1 + 0 * SIZE]); break;
                        case 1: vmovups(ymm0, ptr[CO1 + LDC + 0 * SIZE]); break;
                        case 2:
                            vmovups(ymm0, ptr[CO1 + LDC * 2 + 0 * SIZE]);
                            break;
                        case 3: vmovups(ymm0, ptr[CO2 + 0 * SIZE]); break;
                        case 4: vmovups(ymm0, ptr[CO2 + LDC + 0 * SIZE]); break;
                        case 5:
                            vmovups(ymm0, ptr[CO2 + LDC * 2 + 0 * SIZE]);
                            break;
                        }
                    } else {
                        switch (i) {
                        case 0:
                            vmaskmovps(ymm0, VMASK, ptr[CO1 + 0 * SIZE]);
                            break;
                        case 1:
                            vmaskmovps(ymm0, VMASK, ptr[CO1 + LDC + 0 * SIZE]);
                            break;
                        case 2:
                            vmaskmovps(
                                    ymm0, VMASK, ptr[CO1 + LDC * 2 + 0 * SIZE]);
                            break;
                        case 3:
                            vmaskmovps(ymm0, VMASK, ptr[CO2 + 0 * SIZE]);
                            break;
                        case 4:
                            vmaskmovps(ymm0, VMASK, ptr[CO2 + LDC + 0 * SIZE]);
                            break;
                        case 5:
                            vmaskmovps(
                                    ymm0, VMASK, ptr[CO2 + LDC * 2 + 0 * SIZE]);
                            break;
                        }
                    }

                    if (!isBetaN) {
                        vaddps(Ymm(i + 4), ymm0, Ymm(i + 4));
                    } else {
                        fma(useFma, VBETA, ymm0, Ymm(i + 4), true);
                    }
                }
                if (hasBias) {
                    vaddps(Ymm(i + 4), VBIAS1, Ymm(i + 4));
                }
                if (isLoad1Unmasked) {
                    switch (i) {
                    case 0: vmovups(ptr[CO1 + 0 * SIZE], Ymm(i + 4)); break;
                    case 1:
                        vmovups(ptr[CO1 + LDC + 0 * SIZE], Ymm(i + 4));
                        break;
                    case 2:
                        vmovups(ptr[CO1 + LDC * 2 + 0 * SIZE], Ymm(i + 4));
                        break;
                    case 3: vmovups(ptr[CO2 + 0 * SIZE], Ymm(i + 4)); break;
                    case 4:
                        vmovups(ptr[CO2 + LDC + 0 * SIZE], Ymm(i + 4));
                        break;
                    case 5:
                        vmovups(ptr[CO2 + LDC * 2 + 0 * SIZE], Ymm(i + 4));
                        break;
                    }
                } else {
                    switch (i) {
                    case 0:
                        vmaskmovps(ptr[CO1 + 0 * SIZE], VMASK, Ymm(i + 4));
                        break;
                    case 1:
                        vmaskmovps(
                                ptr[CO1 + LDC + 0 * SIZE], VMASK, Ymm(i + 4));
                        break;
                    case 2:
                        vmaskmovps(ptr[CO1 + LDC * 2 + 0 * SIZE], VMASK,
                                Ymm(i + 4));
                        break;
                    case 3:
                        vmaskmovps(ptr[CO2 + 0 * SIZE], VMASK, Ymm(i + 4));
                        break;
                    case 4:
                        vmaskmovps(
                                ptr[CO2 + LDC + 0 * SIZE], VMASK, Ymm(i + 4));
                        break;
                    case 5:
                        vmaskmovps(ptr[CO2 + LDC * 2 + 0 * SIZE], VMASK,
                                Ymm(i + 4));
                        break;
                    }
                }

                if (unroll_m >= 16) {
                    // Re-use ymm4 (VBIAS2)
                    if (i == 0) {
                        if (hasBias) {
                            if (isLoad1Unmasked) {
                                vmovups(VBIAS2, ptr[BIAS1 + 8 * SIZE]);
                            } else {
                                vmaskmovps(
                                        VBIAS2, VMASK, ptr[BIAS1 + 8 * SIZE]);
                            }
                        }
                    }
                    vmulps(Ymm(i + 10), Ymm(i + 10), VALPHA);
                    if (!isBeta0) {
                        if (isLoad2Unmasked) {
                            switch (i) {
                            case 0: vmovups(ymm0, ptr[CO1 + 8 * SIZE]); break;
                            case 1:
                                vmovups(ymm0, ptr[CO1 + LDC + 8 * SIZE]);
                                break;
                            case 2:
                                vmovups(ymm0, ptr[CO1 + LDC * 2 + 8 * SIZE]);
                                break;
                            case 3: vmovups(ymm0, ptr[CO2 + 8 * SIZE]); break;
                            case 4:
                                vmovups(ymm0, ptr[CO2 + LDC + 8 * SIZE]);
                                break;
                            case 5:
                                vmovups(ymm0, ptr[CO2 + LDC * 2 + 8 * SIZE]);
                                break;
                            }
                        } else {
                            switch (i) {
                            case 0:
                                vmaskmovps(ymm0, VMASK, ptr[CO1 + 8 * SIZE]);
                                break;
                            case 1:
                                vmaskmovps(
                                        ymm0, VMASK, ptr[CO1 + LDC + 8 * SIZE]);
                                break;
                            case 2:
                                vmaskmovps(ymm0, VMASK,
                                        ptr[CO1 + LDC * 2 + 8 * SIZE]);
                                break;
                            case 3:
                                vmaskmovps(ymm0, VMASK, ptr[CO2 + 8 * SIZE]);
                                break;
                            case 4:
                                vmaskmovps(
                                        ymm0, VMASK, ptr[CO2 + LDC + 8 * SIZE]);
                                break;
                            case 5:
                                vmaskmovps(ymm0, VMASK,
                                        ptr[CO2 + LDC * 2 + 8 * SIZE]);
                                break;
                            }
                        }
                        if (!isBetaN) {
                            vaddps(Ymm(i + 10), ymm0, Ymm(i + 10));
                        } else {
                            fma(useFma, VBETA, ymm0, Ymm(i + 10), true);
                        }
                    }
                    if (hasBias) {
                        vaddps(Ymm(i + 10), VBIAS2, Ymm(i + 10));
                    }
                    if (isLoad2Unmasked) {
                        switch (i) {
                        case 0:
                            vmovups(ptr[CO1 + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 1:
                            vmovups(ptr[CO1 + LDC + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 2:
                            vmovups(ptr[CO1 + LDC * 2 + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 3:
                            vmovups(ptr[CO2 + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 4:
                            vmovups(ptr[CO2 + LDC + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 5:
                            vmovups(ptr[CO2 + LDC * 2 + 8 * SIZE], Ymm(i + 10));
                            break;
                        }
                    } else {
                        switch (i) {
                        case 0:
                            vmaskmovps(ptr[CO1 + 8 * SIZE], VMASK, Ymm(i + 10));
                            break;
                        case 1:
                            vmaskmovps(ptr[CO1 + LDC + 8 * SIZE], VMASK,
                                    Ymm(i + 10));
                            break;
                        case 2:
                            vmaskmovps(ptr[CO1 + LDC * 2 + 8 * SIZE], VMASK,
                                    Ymm(i + 10));
                            break;
                        case 3:
                            vmaskmovps(ptr[CO2 + 8 * SIZE], VMASK, Ymm(i + 10));
                            break;
                        case 4:
                            vmaskmovps(ptr[CO2 + LDC + 8 * SIZE], VMASK,
                                    Ymm(i + 10));
                            break;
                        case 5:
                            vmaskmovps(ptr[CO2 + LDC * 2 + 8 * SIZE], VMASK,
                                    Ymm(i + 10));
                            break;
                        }
                    }
                }
                if (i == 2)
                    add(CO1, rax);
            }
            if (unroll_n >= 4) {
                add(CO2, rax);
            }

            // Compute next address of B
            if (!isTransB) {
                lea(rax, ptr[K * SIZE]);
                switch (unroll_n) {
                case 1:
                    add(BO1, LDB);
                    add(BO2, LDB);
                    break;
                case 2:
                    lea(BO1, ptr[BO1 + LDB * 2]);
                    lea(BO2, ptr[BO2 + LDB * 2]);
                    break;
                case 3:
                    lea(BO1, ptr[BO1 + LDB3]);
                    lea(BO2, ptr[BO2 + LDB3]);
                    break;
                case 4:
                    lea(BO1, ptr[BO1 + LDB * 4]);
                    lea(BO2, ptr[BO2 + LDB * 4]);
                    break;
                case 5:
                    lea(BO1, ptr[BO1 + LDB * 4]);
                    add(BO1, LDB);
                    lea(BO2, ptr[BO2 + LDB * 4]);
                    add(BO2, LDB);
                    break;
                case 6:
                    lea(BO1, ptr[BO1 + LDB3 * 2]);
                    lea(BO2, ptr[BO2 + LDB3 * 2]);
                    break;
                }
                sub(BO1, rax);
                sub(BO2, rax);
            } else {
                mov(rax, LDB);
                imul(rax, K);
                sub(BO1, rax);
                add(BO1, unroll_n * SIZE);
            }
        };

        auto kernel_16x6 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy) {
            kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, true);
        };

        auto kernel_16x5 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy) {
            kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, true);
        };

        auto kernel_16x4 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy) {
            kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, true);
        };

        auto kernel_16x3 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy,
                bool useFma = true) {
            kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, useFma, Ymm(4), Ymm(5), Ymm(6), Ymm(7),
                    Ymm(8), Ymm(9), Ymm(10), Ymm(11), Ymm(12), Ymm(13), Ymm(14),
                    Ymm(15), Ymm(7), Ymm(8), Ymm(9), Ymm(7), Ymm(8), Ymm(9),
                    Ymm(13), Ymm(14), Ymm(15));
        };

        auto kernel_16x2 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy) {
            kernel_16x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, false);
        };

        auto kernel_16x1 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy) {
            kernel_16x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, false);
        };

        auto kernel_8x6 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy,
                bool useFma = true) {
            kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, useFma, Ymm(4), Ymm(5), Ymm(6), Ymm(7),
                    Ymm(8), Ymm(9), Ymm(10), Ymm(11), Ymm(12), Ymm(13), Ymm(14),
                    Ymm(15), Ymm(10), Ymm(11), Ymm(12), Ymm(13), Ymm(14),
                    Ymm(15));
        };

        auto kernel_8x5 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy) {
            kernel_8x6(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy);
        };

        auto kernel_8x4 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy) {
            kernel_8x6(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy);
        };

        auto kernel_8x3 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy,
                bool useFma = true) {
            kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, useFma, Ymm(4), Ymm(5), Ymm(6), Ymm(7),
                    Ymm(8), Ymm(9), Ymm(10), Ymm(11), Ymm(12), Ymm(13), Ymm(14),
                    Ymm(15), Ymm(7), Ymm(8), Ymm(9), Ymm(7), Ymm(8), Ymm(9),
                    Ymm(13), Ymm(14), Ymm(15));
        };

        auto kernel_8x2 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy) {
            kernel_8x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, false);
        };

        auto kernel_8x1 = [&](int unroll_m, int unroll_n, bool isLoad1Unmasked,
                bool isLoad2Unmasked, bool isDirect, bool isCopy) {
            kernel_8x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                    isDirect, isCopy, false);
        };

        // High-level subroutine; does packing if needed, then splits C matrix.
        // Operates on chunks of 16 rows, 6 columns at a time (handling tail
        // cases appropriately).
        // Masking is used for tail cases where M is not divisible by 8.
        auto subloop = [&](
                int unroll_m, bool isLoad1Unmasked, bool isLoad2Unmasked) {
            if (isTransA) {
                do_pack(unroll_m, isLoad1Unmasked, isLoad2Unmasked);
            }

            Label subloop11, subloop11mask;
            Label subloop20, subloop21, subloop22, subloop23;
            Label subloop24, subloop25;
            Label subloop30, subloop31, subloop32, subloop33;
            Label subloop34, subloop35;
            Label subloop98, subloop98mask;
            Label subloop99, subloop99mask;

            mov(CO1, C);
            lea(CO2, ptr[CO1 + LDC * 2]);
            add(CO2, LDC);
            add(C, unroll_m * SIZE);
            mov(BO1, B);
            if (!isTransB) {
                lea(BO2, qword[B + LDB3]);
            }

            if (!isTransA) {
                lea(AA, ptr[A + (unroll_m * 2 - 1 - OFFSET) * SIZE]);
                cmp(M, UNROLL_M);
                jg(subloop98, T_NEAR);

                mov(AA, ORIG_A);
                lea(AA, ptr[AA + (unroll_m - 1 - OFFSET) * SIZE]);
                L(subloop98);
            }

            mov(LL, N);
            mov(I, LL);
            if (!isTransA) {
                // If N is too small, skip copy operation
                cmp(LL, UNROLL_N * 3);
                jle(subloop30, T_NEAR);

                // If A is not aligned to cache line
                cmp(FLAG, 0);
                je(subloop30, T_NEAR);
            } else {
                cmp(LL, UNROLL_N);
                jl(subloop20, T_NEAR);
            }
            align(16);

            if (!isTransA) {
                if (unroll_m == 16) {
                    kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                            isLoad2Unmasked, true, true);
                } else {
                    kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                            isLoad2Unmasked, true, true);
                }
            } else {
                if (unroll_m == 16) {
                    kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                            isLoad2Unmasked, false, false);
                } else {
                    kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                            isLoad2Unmasked, false, false);
                }
            }

            sub(I, UNROLL_N);
            cmp(I, UNROLL_N);
            jl(subloop20, T_NEAR);
            align(16);

            L(subloop11);
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, false, false);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        false, false);
            }
            sub(I, UNROLL_N);
            cmp(I, UNROLL_N);
            jge(subloop11, T_NEAR);
            align(16);

            L(subloop20);
            cmp(I, 1);
            jne(subloop21, T_NEAR);
            if (unroll_m == 16) {
                kernel_16x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked,
                        false, false);
            } else {
                kernel_8x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, false,
                        false);
            }
            jmp(subloop99, T_NEAR);
            align(16);

            L(subloop21);
            cmp(I, 2);
            jne(subloop22, T_NEAR);
            if (unroll_m == 16) {
                kernel_16x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked,
                        false, false);
            } else {
                kernel_8x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, false,
                        false);
            }
            jmp(subloop99, T_NEAR);
            align(16);

            L(subloop22);
            cmp(I, 3);
            jne(subloop23, T_NEAR);
            if (unroll_m == 16) {
                kernel_16x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked,
                        false, false);
            } else {
                kernel_8x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, false,
                        false);
            }
            jmp(subloop99, T_NEAR);
            align(16);

            L(subloop23);
            cmp(I, 4);
            jne(subloop24, T_NEAR);
            if (unroll_m == 16) {
                kernel_16x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked,
                        false, false);
            } else {
                kernel_8x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, false,
                        false);
            }
            jmp(subloop99, T_NEAR);
            align(16);

            L(subloop24);
            cmp(I, 5);
            jne(subloop99, T_NEAR);
            if (unroll_m == 16) {
                kernel_16x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked,
                        false, false);
            } else {
                kernel_8x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, false,
                        false);
            }
            jmp(subloop99, T_NEAR);
            align(16);

            if (!isTransA) {
                L(subloop30);
                cmp(I, UNROLL_N);
                jl(subloop25, T_NEAR);
                align(16);

                L(subloop31);
                if (unroll_m == 16) {
                    kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                            isLoad2Unmasked, true, false);
                } else {
                    kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                            isLoad2Unmasked, true, false);
                }
                sub(I, UNROLL_N);
                cmp(I, UNROLL_N);
                jge(subloop31, T_NEAR);
                align(16);

                L(subloop25);
                cmp(I, 1);
                jne(subloop32, T_NEAR);
                if (unroll_m == 16) {
                    kernel_16x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                } else {
                    kernel_8x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                }
                jmp(subloop99, T_NEAR);
                align(16);

                L(subloop32);
                cmp(I, 2);
                jne(subloop33, T_NEAR);
                if (unroll_m == 16) {
                    kernel_16x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                } else {
                    kernel_8x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                }
                jmp(subloop99, T_NEAR);
                align(16);

                L(subloop33);
                cmp(I, 3);
                jne(subloop34, T_NEAR);
                if (unroll_m == 16) {
                    kernel_16x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                } else {
                    kernel_8x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                }
                jmp(subloop99, T_NEAR);
                align(16);

                L(subloop34);
                cmp(I, 4);
                jne(subloop35, T_NEAR);
                if (unroll_m == 16) {
                    kernel_16x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                } else {
                    kernel_8x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                }
                jmp(subloop99, T_NEAR);
                align(16);

                L(subloop35);
                cmp(I, 5);
                jne(subloop99, T_NEAR);
                if (unroll_m == 16) {
                    kernel_16x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                } else {
                    kernel_8x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked,
                            true, false);
                }
                align(16);
            }

            L(subloop99);
            // Compute address for A
            if (!isTransA) {
                add(A, unroll_m * SIZE);
            } else {
                mov(rax, LDA);
                imul(rax, rax, unroll_m);
                add(A, rax);
            }

            // Compute next address of BIAS
            if (hasBias) {
                add(BIAS, unroll_m * SIZE);
            }
        };

        preamble();

        Label buffer_in_ws, buffer_allocated;

        // Get the registers
        mov(B, ARG_B);
        mov(LDB, ARG_LDB);
        mov(r15, ARG_BETA);
        mov(r12, ARG_C);
        if (hasBias)
            mov(r10, ARG_BIAS);
        mov(LDC, ARG_LDC);
        mov(rbp, rsp);

        vmovss(xmm0, ptr[ARG_ALPHA]);
        vmovss(xmm1, ptr[r15]);

#if _WIN32
        mov(A, ARG_A);
        mov(LDA, ARG_LDA);
#endif

        cmp(K, STACK_K_CAPACITY);
        jg(buffer_in_ws, T_NEAR);

        // Create buffer and align to 4kB page
        lea(rax, ptr[K * SIZE]);
        sal(rax, 4);
        add(rax, 256);
        sub(rsp, rax);
        and_(rsp, -PAGE_4K);
        jmp(buffer_allocated, T_NEAR);

        L(buffer_in_ws);
        mov(rsp, ARG_WS);

        L(buffer_allocated);

        mov(ORIG_SP, rbp);
        mov(M, ARG_M);
        mov(N, ARG_N);
        mov(C, r12);
        if (hasBias)
            mov(BIAS, r10);
        vmovss(ALPHA, xmm0);
        vmovss(BETA, xmm1);
        sub(A, -OFFSET * SIZE);
        sub(B, -OFFSET * SIZE);
        mov(ORIG_A, A);
        sal(LDA, BASE_SHIFT);
        sal(LDB, BASE_SHIFT);
        sal(LDC, BASE_SHIFT);
        lea(LDB3, ptr[LDB + LDB * 2]);

        for (int i = 0; i < 8; i++) {
            mov(dword[rsp + 88 + i * 4], i);
        }

        if (isTransA && is_avx2) {
            movq(xmm0, LDA);
            vpbroadcastq(ymm1, xmm0);
            vinsertf128(ymm0, ymm0, xmm0, 1);
            vpermilpd(ymm0, ymm0, 5);
            vpaddq(ymm1, ymm1, ymm1);
            vperm2f128(ymm1, ymm1, ymm1, 8);
            vpaddq(ymm0, ymm0, ymm1);
            vmovups(STRIDE, ymm0);
        }

        // Check A alignment and leading dimension; take copy-based path as
        // needed
        mov(rax, LDA);
        or_(rax, A);
        and_(rax, 0x1f);
        mov(FLAG, rax);

        Label main0, main1, main2, main3, main999;

        cmp(M, UNROLL_M);
        jl(main0, T_NEAR);
        align(16);

        L(main1);
        subloop(UNROLL_M, true, true);
        sub(M, UNROLL_M);
        cmp(M, UNROLL_M);
        jge(main1, T_NEAR);
        align(16);

        L(main0);
        cmp(M, 0);
        jle(main999, T_NEAR);

        if (UNROLL_M > 8) {
            cmp(M, 8);
            jle(main2, T_NEAR);

            sub(M, 8);
            vbroadcastss(VMASK, M);
            vpcmpgtd(VMASK, VMASK, MASK);

            subloop(16, true, false);
            jmp(main999, T_NEAR);
            align(16);

            L(main2);
            cmp(M, 8);
            jne(main3, T_NEAR);
            subloop(8, true, true);
            jmp(main999, T_NEAR);
        }

        align(16);

        L(main3);
        vbroadcastss(VMASK, M);
        if (is_avx2) {
            vpcmpgtd(VMASK, VMASK, MASK);
        } else {
            auto xmask = Xmm(VMASK.getIdx());
            auto xmm_tmp = xmm4;

            vextractf128(xmm_tmp, VMASK, 1);
            vpcmpgtd(xmask, xmask, MASK);
            vpcmpgtd(xmm_tmp, xmm_tmp, dword[rsp + 88 + 4 * 4]); // MASK + 4
            vinsertf128(VMASK, VMASK, xmm_tmp, 1);
        }
        subloop(8, false, false);
        align(16);

        L(main999);
        // Restore original stack
        mov(rsp, ORIG_SP);

        vzeroupper();
        postamble();

        ker_ = this->getCode<ker_t>();
    }

    typedef void (*ker_t)(dim_t m, dim_t n, dim_t k,
            const float *alpha, const float *a, dim_t lda,
            const float *b, dim_t ldb, const float *beta, float *c,
            dim_t ldc, const float *bias, float *ws);

    void operator()(dim_t  m, dim_t n, dim_t k,
            const float *alpha, const float *a, dim_t lda,
            const float *b, dim_t ldb, const float *beta, float *c,
            dim_t ldc, const float *bias, float *ws) const
    {
        ker_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, bias, ws);
    }

private:
    ker_t ker_;
};

const xbyak_gemm *get_xbyak_gemm(
        bool isTransA, bool isTransB, float beta, bool hasBias) {
    auto beta_idx = [](float beta) {
        return (beta == 0.0) ? 0 : (beta == 1.0 ? 1 : 2);
    };

    // Kernel table [isTransA][isTransB][hasBias][beta (0, 1, other)]
    static xbyak_gemm *kernel_table[2][2][2][3];
    static std::once_flag initialized;
    std::call_once(initialized, [=]{
            for (bool isTransA: {false, true})
            for (bool isTransB: {false, true})
            for (bool hasBias: {false, true})
            for (float beta: {0.0f, 1.0f, 2.0f}) {
                // nocopy sgemm with bias for beta != 0.0 is not supported
                if (hasBias && beta != 0.0)
                    continue;
                kernel_table[isTransA][isTransB][hasBias][beta_idx(beta)] =
                    new xbyak_gemm(isTransA, isTransB, beta, hasBias);
            }
    });

    return kernel_table[isTransA][isTransB][hasBias][beta_idx(beta)];
}

void sgemm_nocopy_driver(const char *transa,
        const char *transb, int m, int n, int k, const float *alpha,
        const float *a, dim_t lda, const float *b, dim_t ldb, const float *beta,
        float *c, dim_t ldc, const float *bias, float *ws)
{
    bool isTransA = (*transa == 'T' || *transa == 't');
    bool isTransB = (*transb == 'T' || *transb == 't');

    int Bm, sizeM, Bn, sizeN, Bk, sizeK;

    int i, j;

    if ((m <= 0) || (n <= 0))
        return;

    if ((k <= 0) || (alpha[0] == 0.)) {

        if (beta[0] == 0.) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[i + j * ldc] = 0.0;
        } else if (beta[0] != 1.) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[i + j * ldc] *= beta[0];
        }

        return;
    }

    assert(IMPLICATION(bias != nullptr, *beta == 0.0));

    // XXX: this happens on every thread...
    bool hasBias = (bias != nullptr);
    auto ker_bn = get_xbyak_gemm(isTransA, isTransB, *beta, hasBias);
    auto ker_b1 = get_xbyak_gemm(isTransA, isTransB, 1.0, false);
    auto ker_b0 = get_xbyak_gemm(isTransA, isTransB, 0.0, false);
    assert(ker_bn && ker_b1 && ker_b0);

    int BM = 4032;
    int BN = isTransA ? 96 : 48;
    int BK = isTransB ? 96 : 256;
    const float *curA, *curB, *curBias = nullptr;
    float *curC;

    for (Bk = 0; Bk < k; Bk += sizeK) {
        sizeK = k - Bk;
        if (sizeK >= BK * 2)
            sizeK = BK;
        else {
            if (sizeK > BK)
                sizeK = (sizeK + 1) / 2;
        }

        for (Bm = 0; Bm < m; Bm += sizeM) {
            sizeM = m - Bm;
            if (sizeM >= BM * 2)
                sizeM = BM;
            else {
                if (sizeM > BM + BM / 2)
                    sizeM = (sizeM + 1) / 2;
            }

            for (Bn = 0; Bn < n; Bn += sizeN) {
                sizeN = n - Bn;
                if (sizeN >= BN * 2)
                    sizeN = BN;
                else {
                    if (sizeN > BN + BN / 2)
                        sizeN = (sizeN + 1) / 2;
                }

                if (!isTransA) {
                    curA = a + Bm + Bk * lda;
                } else {
                    curA = a + Bk + Bm * lda;
                }
                if (!isTransB) {
                    curB = b + Bk + Bn * ldb;
                } else {
                    curB = b + Bn + Bk * ldb;
                }
                curC = c + Bm + (size_t)Bn * ldc;
                if (bias != nullptr) {
                    if (Bk == 0) {
                        curBias = bias + Bm;
                    } else {
                        curBias = nullptr;
                    }
                }
                if (Bk == 0) {
                    if (*beta == 0.0 && bias == nullptr)
                        (*ker_b0)((dim_t)sizeM, (dim_t)sizeN, (dim_t)sizeK,
                                alpha, curA, lda, curB, ldb, beta, curC, ldc,
                                curBias, ws);
                    else
                        (*ker_bn)((dim_t)sizeM, (dim_t)sizeN, (dim_t)sizeK,
                                alpha, curA, lda, curB, ldb, beta, curC, ldc,
                                curBias, ws);
                } else {
                    (*ker_b1)((dim_t)sizeM, (dim_t)sizeN, (dim_t)sizeK,
                            alpha, curA, lda, curB, ldb, beta, curC, ldc,
                            curBias, ws);
                }
            }
        }
    }
}

}

mkldnn_status_t jit_avx_gemm_f32(
        const char *transa, const char *transb,
        const int *p_m, const int *p_n, const int *p_k, const float *p_alpha,
        const float *A, const int *p_lda, const float *B, const int *p_ldb,
        const float *p_beta, float *C, const int *p_ldc, const float *bias)
{
    using namespace mkldnn::impl::utils;
    using namespace avx_gemm_f32;
    using namespace gemm_utils;

    if (*p_beta != 0 && bias)
        return ref_gemm(transa, transb, p_m, p_n, p_k,
                p_alpha, A, p_lda, B, p_lda, p_beta, C, p_ldc, bias);

    int nthr = (mkldnn_in_parallel()) ? 1 : mkldnn_get_max_threads();

    int m = *p_m;
    int n = *p_n;
    int k = *p_k;
    dim_t lda = *p_lda;
    dim_t ldb = *p_ldb;
    dim_t ldc = *p_ldc;
    float beta = *p_beta;
    int MB, NB, KB;

    int nthr_m, nthr_n, nthr_k, nthr_mn;

    // Determine threading partitioning
    calc_nthr_nocopy_avx(
            m, n, k, nthr, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    assert(IMPLICATION(!mkldnn_thr_syncable(), nthr_k == 1));

    // May not happen, but just in case
    if (nthr < nthr_m * nthr_n * nthr_k)
        nthr = nthr_m * nthr_n * nthr_k;

    nthr_mn = nthr_m * nthr_n;

    unsigned char * ompstatus_ = nullptr;
    unsigned char volatile *ompstatus = nullptr;

    float *c_buffers = nullptr;
    float *ws_buffers = nullptr;

    if (nthr_k > 1) {
        ompstatus_ = (unsigned char *) malloc(
                nthr * CACHE_LINE_SIZE,
                CACHE_LINE_SIZE);
        ompstatus = (unsigned char volatile *) ompstatus_;
        assert(ompstatus);

        for (int i = 0; i < nthr; i++)
            ompstatus[i * CACHE_LINE_SIZE] = 0;

        c_buffers = (float *)malloc(nthr_m * nthr_n * (nthr_k - 1) * MB * NB
                * sizeof(float), PAGE_4K);
    }

    const size_t ws_elems_per_thr = (size_t)k * 16 + 64;
    const size_t ws_size_per_thr
            = rnd_up(ws_elems_per_thr * sizeof(float), PAGE_4K);
    if (k > STACK_K_CAPACITY) {
        ws_buffers = (float *)malloc(nthr * ws_size_per_thr, PAGE_4K);
    }

    parallel_nd(nthr, [&](const int ithr) {
        int ithr_m, ithr_n, ithr_k, ithr_mn;
        int m_from, m_to, myM;
        int n_from, n_to, myN;
        int k_from, k_to, myK;
        int cbase, ibase;
        const float *myA, *myB, *myBias = nullptr;
        float *myC = C, myBeta;
        float *ws = ws_buffers ?
                ws_buffers + ithr * ws_size_per_thr / sizeof(float) : 0;
        dim_t ld = ldc;

        int sum_later = (mkldnn_get_num_threads() < nthr_m * nthr_n * nthr_k);

        if (ithr < nthr_m * nthr_n * nthr_k) {

            ithr_mn = ithr % nthr_mn;
            ithr_m = ithr_mn % nthr_m;
            ithr_n = ithr_mn / nthr_m;
            ithr_k = ithr / nthr_mn;

            /* swap ithr_k for performance improvement */
            if (ithr_k == 0)
                ithr_k = nthr_k - 1;
            else if (ithr_k == nthr_k - 1)
                ithr_k = 0;

            m_from = MB * (ithr_m);
            m_to = MB * (ithr_m + 1);
            if (m_to > m)
                m_to = m;
            myM = m_to - m_from;

            n_from = NB * (ithr_n);
            n_to = NB * (ithr_n + 1);
            if (n_to > n)
                n_to = n;
            myN = n_to - n_from;

            k_from = KB * (ithr_k);
            k_to = KB * (ithr_k + 1);
            if (k_to > k)
                k_to = k;
            myK = k_to - k_from;

            cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);
            ibase = (ithr_m + nthr_m * ithr_n) * nthr_k;

            if ((myM > 0) && (myN > 0)) {

                if (*transa == 'N' || *transa == 'n') {
                    myA = &(A[m_from + k_from * lda]);
                } else {
                    myA = &(A[k_from + m_from * lda]);
                }
                if (*transb == 'N' || *transb == 'n') {
                    myB = &(B[k_from + n_from * ldb]);
                } else {
                    myB = &(B[n_from + k_from * ldb]);
                }
                if (ithr_k == 0) {
                    myC = &(C[m_from + n_from * ldc]);
                    myBeta = beta;
                    ld = ldc;
                    if (bias)
                        myBias = &(bias[m_from]);
                } else {
                    myC = c_buffers + (dim_t)MB * NB * (cbase + ithr_k - 1);
                    myBeta = 0.0;
                    ld = MB;
                    myBias = nullptr;
                }

                sgemm_nocopy_driver(transa, transb, myM, myN, myK, p_alpha, myA,
                        lda, myB, ldb, &myBeta, myC, ld, myBias, ws);

                if (nthr_k > 1 && !sum_later)
                    ompstatus[(ibase + ithr_k) * CACHE_LINE_SIZE] = 1;
            }

            if (nthr_k > 1 && !sum_later) {

                // sum matrices partitioned along K dimension
                int n1, n2;

                partition_unit_diff(ithr_k, nthr_k, myN, &n1, &n2);

                if (ithr_k > 0) {

                    myC = c_buffers + (dim_t)MB * NB * (cbase + ithr_k - 1)
                        + (dim_t)n1 * MB;
                    /* need to wait until main thread finishes */
                    while (ompstatus[ibase * CACHE_LINE_SIZE] != 1) {
                    };

                    /* my cache is hot */
                    sum_two_matrices(myM, n2, myC, MB,
                            &C[m_from + (n_from + n1) * ldc], ldc);
                }

                for (int ik = 1; ik < nthr_k; ++ik) {
                    if (ik != ithr_k) {

                        myC = c_buffers + (dim_t)MB * NB * (cbase + ik - 1)
                            + (dim_t)n1 * MB;

                        while (ompstatus[(ibase + ik) * CACHE_LINE_SIZE] != 1) {
                        };

                        sum_two_matrices(myM, n2, myC, MB,
                                &C[m_from + (n_from + n1) * ldc], ldc);
                    }
                }
            }
        }
    });

    // handle C summation later
    if (nthr_k > 1 && ompstatus[0] == 0) {

        parallel_nd(nthr, [&](const int ithr) {
            int ithr_m, ithr_n, ithr_k, ithr_mn;
            int m_from, m_to, myM;
            int n_from, n_to, myN;
            int cbase;
            float *myC = C;

            if (ithr < nthr_m * nthr_n * nthr_k) {

                ithr_mn = ithr % nthr_mn;
                ithr_m = ithr_mn % nthr_m;
                ithr_n = ithr_mn / nthr_m;
                ithr_k = ithr / nthr_mn;

                /* swap ithr_k for performance improvement */
                if (ithr_k == 0)
                    ithr_k = nthr_k - 1;
                else if (ithr_k == nthr_k - 1)
                    ithr_k = 0;

                m_from = MB * (ithr_m);
                m_to = MB * (ithr_m + 1);
                if (m_to > m)
                    m_to = m;
                myM = m_to - m_from;

                n_from = NB * (ithr_n);
                n_to = NB * (ithr_n + 1);
                if (n_to > n)
                    n_to = n;
                myN = n_to - n_from;

                cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

                if (nthr_k > 1) {
                    // sum matrices partitioned along K dimension
                    int n1, n2;

                    partition_unit_diff(ithr_k, nthr_k, myN, &n1, &n2);

                    if (ithr_k > 0) {

                        myC = c_buffers + (dim_t)MB * NB * (cbase + ithr_k - 1)
                            + (dim_t)n1 * MB;

                        /* my cache is hot */
                        sum_two_matrices(myM, n2, myC, MB,
                                         &C[m_from + (n_from + n1) * ldc], ldc);
                    }

                    for (int ik = 1; ik < nthr_k; ++ik) {
                        if (ik != ithr_k) {

                            myC = c_buffers + (dim_t)MB * NB * (cbase + ik - 1)
                                + (dim_t)n1 * MB;

                            sum_two_matrices(myM, n2, myC, MB,
                                             &C[m_from + (n_from + n1) * ldc], ldc);
                        }
                    }
                }
            }
        });
    }


    free(c_buffers);
    free(ompstatus_);
    free(ws_buffers);

    return mkldnn_success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
