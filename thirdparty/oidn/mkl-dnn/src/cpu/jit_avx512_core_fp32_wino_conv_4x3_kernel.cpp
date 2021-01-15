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
#include "type_helpers.hpp"
#include "utils.hpp"

#include <math.h>

#include "jit_avx512_core_fp32_wino_conv_4x3_kernel.hpp"

#define GET_OFF(field) offsetof(jit_wino_transform_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {

using namespace mkldnn::impl::utils;

unsigned int L1_cache_size = get_cache_size(1, true);
unsigned int L2_cache_size = get_cache_size(2, true);
unsigned int LLC_data_size = get_cache_size(3, false);

// the test funtion takes jcp, the candidate and the current best.
// it  returns true if the new candidate is better
int get_divisor_satisfying_cond(jit_conv_winograd_conf_t &jcp, int number,
        int default_best, bool (*test)(jit_conv_winograd_conf_t &, int, int))
{
    int best_divisor = default_best;
    auto test_num
            = [&best_divisor, test](jit_conv_winograd_conf_t &jcp, int num) {
                  if (test(jcp, num, best_divisor)) {
                      best_divisor = num;
                  }
              };

    for (int divisor = 1; divisor <= ::sqrt(number); divisor++) {
        if (number % divisor == 0) {
            test_num(jcp, divisor);
            test_num(jcp, number / divisor);
        }
    }

    return best_divisor;
}

namespace {
bool is_winograd_faster_than_direct(const jit_conv_winograd_conf_t &jcp) {
    /* Determines if current winograd implementation is faster than direct.
       Following conditions are empirical and based on performance data */
    unsigned int ncores_per_socket =
        cpu.getNumCores(Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    unsigned int nthreads = mkldnn_get_max_threads();

    if (jcp.prop_kind == prop_kind::forward_inference) {
        return jcp.mb >= 4;
    } else if (nthreads > ncores_per_socket) {
        double src_dst_transforms_per_core = alpha * alpha
            * (jcp.ic + jcp.oc)
            * jcp.mb * ((jcp.oh + tile_size - 1) / tile_size)
            * ((jcp.ow + tile_size - 1) / tile_size)
            * sizeof(float) / 1024. / 1024. / nthreads;
        double wei_transform = alpha * alpha
            * jcp.ic * jcp.oc * sizeof(float) /1024. / 1024.;

        if (jcp.prop_kind == prop_kind::backward_weights) {
            if (src_dst_transforms_per_core < 0.3
                    || (src_dst_transforms_per_core <= 28 && wei_transform < 4))
                return false;
            else
                return true;
        } else {
            if (src_dst_transforms_per_core < 2.0 || wei_transform < 0.02)
                return false;
        }
    }

    return jcp.mb > 8;
}
}

/* assumes 512 bits registers */
/* TODO: add support for strides */
/* TODO: handle the prefetch distance automatically */
typedef enum cache_t_ { L1, L2, L3 } cache_t;

template <typename data_t>
struct prefetcher_t {
    prefetcher_t(jit_generator *generator, Xbyak::Reg64 reg_base_addr,
            cache_t cache_type, size_t block_size, /* in number of elements*/
            int nb_instructions_in_block, int fma_ipc)
        : cg_(generator)
        , reg_base_addr_(reg_base_addr)
        , cache_type_(cache_type)
        , cache_block_size_(block_size)
    {
        nb_cache_lines_to_prefetch_ = cache_block_size_ / (64 / sizeof(data_t));
        prefetch_spread_
                = div_up(nb_instructions_in_block, nb_cache_lines_to_prefetch_);
        prefetch_blk_
                = div_up(nb_cache_lines_to_prefetch_, nb_instructions_in_block);

        /* assumption: when fetch in Li, data is already in L(i+1) */
        int cache_latency;
        switch (cache_type_) {
        case L1: cache_latency = 14; break;
        case L2: cache_latency = 250; break;
        case L3: cache_latency = 250; break;
        }

        prefetch_distance_ = div_up(cache_latency, nb_cache_lines_to_prefetch_);
    }

    void prefetch(int instruction_number)
    {
        if (instruction_number % prefetch_spread_ == 0) {
            for (int i = 0; (i < prefetch_blk_)
                    && (prefetches_issued_ < nb_cache_lines_to_prefetch_);
                    i++, prefetches_issued_++) {
                prefetch_inst_(cg_->EVEX_compress_addr(
                        reg_base_addr_, (cache_block_size_ * prefetch_distance_)
                                        * sizeof(data_t)
                                + (prefetches_issued_ * 64)));
            }
        }
    }

private:
    void prefetch_inst_(const Xbyak::Address &addr)
    {
        switch (cache_type_) {
        case L1: cg_->prefetcht0(addr); break;
        case L2: cg_->prefetcht1(addr); break;
        case L3: cg_->prefetcht2(addr); break;
        default:
            break; // TODO: raise an exception or put an assert
        }
    }

    jit_generator *cg_;
    Xbyak::Reg64 reg_base_addr_;
    cache_t cache_type_;
    int cache_block_size_ = 0;
    int nb_cache_lines_to_prefetch_ = 0;
    int prefetches_issued_ = 0;
    int prefetch_spread_ = 0;
    int prefetch_blk_ = 0;
    int prefetch_distance_ = 0;
};

// utilities to support kernel parameter selection
bool check_L2_block_per_thread(jit_conv_winograd_conf_t &jcp,
        int dimN_block, float C2_min, float C2_max) {
    float block_size = alpha * alpha * (2*(jcp.oc + jcp.ic)
        * dimN_block * jcp.dimN_reg_block
        + div_up(jcp.ic * jcp.oc,mkldnn_get_max_threads())) * (float)sizeof(float);
    float L2_lb = C2_min * L2_cache_size;
    float L2_ub = C2_max * L2_cache_size;
    return (block_size > L2_lb && block_size < L2_ub);
}

bool check_L1_block_gemm(jit_conv_winograd_conf_t &jcp, int dimK_block,
        int dimM_block, float C1_min, float C1_max) {
    float gemm_block_size = (dimM_block * jcp.dimM_simd_block * dimK_block
                             * jcp.dimK_reg_block * jcp.dimM_reg_block
                     + dimK_block * jcp.dimK_reg_block * jcp.dimN_reg_block
                     + dimM_block * jcp.dimM_simd_block * jcp.dimN_reg_block)
                     * (float)sizeof(float);
    float L1_lb = C1_min * L1_cache_size;
    float L1_ub = C1_max * L1_cache_size;
    return (gemm_block_size > L1_lb && gemm_block_size < L1_ub);
}
bool check_cond1(int dimN_reg_block, int dimK_block, int dimK_reg_block,
        int dimM_block, int dimM_reg_block, int dimM_simd_block, float C)
{
    float lhs = (dimM_block * dimN_reg_block * dimM_simd_block * dimM_reg_block
                        + dimM_block * dimK_block * dimK_reg_block
                                * dimM_simd_block * dimM_reg_block
                        + dimK_block * dimN_reg_block * dimK_reg_block)
            * (float)sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs < rhs);
}
bool check_cond1_bis(int dimN_reg_block, int dimK_block, int dimK_reg_block,
        int dimM_block, int dimM_reg_block, int dimM_simd_block, float C)
{
    float lhs = (dimM_block * dimM_reg_block * dimK_block * dimK_reg_block
            * dimM_simd_block + dimK_block * dimN_reg_block * dimK_reg_block)
            * (float)sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs < rhs);
}
bool check_cond2(int nb_dimN_reg_block, int dimN_reg_block, int dimK_nb_block,
        int dimK_block, int dimK_reg_block, int dimM_block, int dimM_reg_block,
        int dimM_simd_block, float C)
{
    float lhs = (nb_dimN_reg_block * dimM_block * dimN_reg_block
                              * dimM_simd_block * dimM_reg_block
                      + dimK_nb_block * dimM_block * dimK_block * dimK_reg_block
                              * dimM_simd_block * dimM_reg_block
                      + nb_dimN_reg_block * dimK_nb_block * dimK_block
                              * dimN_reg_block * dimK_reg_block)
            * (float)sizeof(float);
    float rhs = C * L2_cache_size;
    return (lhs < rhs);
}

bool check_kernel_cond(int dimM_block, int dimM_reg_block, int dimM_simd_block,
        int dimN_block, int dimN_reg_block, int dimK, float C1, float C2)
{
    float A_size = dimM_block * dimM_reg_block * dimM_simd_block * dimK
        * (float)sizeof(float);
    float B_size = dimN_block * dimN_reg_block * dimK
        * (float)sizeof(float);
    return (A_size > C1 * L2_cache_size && B_size > C2 * L2_cache_size);
}
}

using namespace mkldnn::impl::format_tag;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

void _jit_avx512_core_fp32_wino_conv_4x3_data_kernel::gemm_loop_generate()
{
    // for (int dimM_block =0; dimM_block < jcp.dimM_block; dimM_block++)
    // for (int dimM_reg_block =0; dimM_reg_block < jcp.dimM_reg_block;
    //      dimM_reg_block++) // unrolled
    //     for (int dimK_block = 0; dimK_block < jcp.dimK_block; dimK_block++)
    //         for (int dimK_reg_block= 0; dimK_reg_block < jcp.dimK_reg_block;
    //              dimK_reg_block++) // unrolled
    //             for (int tile =0; tile < jcp.dimN_reg_block; tile++)
    //                 C[dimM_block][dimM_reg_block][tile] +=
    //                 A[dimM_block][dimM_reg_block][dimK_block][dimK_reg_block]
    //                 * broadcast(B[dimK_block][tile][dimK_reg_block]);
    // Notes:
    // jcp.kernel_kind defines embedded or explicit broadcast
    // dimM_reg_block=1 for embedded bcast kernel

    auto zmm_srcA = [=]() {
        return Xbyak::Zmm(0);
    };
    auto zmm_srcB = [=](int tile) {
        int idx = 1 + tile;
        assert(idx < 1 + jcp.dimN_reg_block);
        return Xbyak::Zmm(idx);
    };
    auto zmm_dstC = [=](int dimM_reg_block, int tile) {
        int idx{0};
        if (jcp.kernel_kind == embd_bcast)
            idx = 1 + tile;
        else
            idx = 1 + jcp.dimN_reg_block
                  + dimM_reg_block * jcp.dimN_reg_block + tile;
        assert(idx < 32);
        return Xbyak::Zmm(idx);
    };

    auto prepare_output = [=]() {
        for (int dimM_reg_block = 0; dimM_reg_block < jcp.dimM_reg_block;
              dimM_reg_block++) {
            for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                Zmm zmm = zmm_dstC(dimM_reg_block, tile);
                vpxord(zmm, zmm, zmm);
            }
        }
    };
    auto store_output = [=](bool output_is_aligned) {
        Label save;
        cmp(reg_is_beta_zero, 0);
        je(save, T_NEAR);

        for (int dimM_reg_block = 0; dimM_reg_block < jcp.dimM_reg_block;
              dimM_reg_block++) {
            for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                Zmm zmm = zmm_dstC(dimM_reg_block,tile);
                int output_offset
                    = jcp.dimN_reg_block * dimM_reg_block * 64 + tile * 64;
                vaddps(zmm, zmm, EVEX_compress_addr(reg_dstC, output_offset));
            }
        }

        L(save);
        for (int dimM_reg_block = 0; dimM_reg_block < jcp.dimM_reg_block;
              dimM_reg_block++) {
            for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                Zmm zmm = zmm_dstC(dimM_reg_block,tile);
                int output_offset
                    = jcp.dimN_reg_block * dimM_reg_block * 64 + tile * 64;

                // In W_SGD, output will be reused.
                if (output_is_aligned
                    && jcp.dimK_nb_block == 1
                    && jcp.sched_policy == WSCHED_DATA_W_S_G_D
                    && (jcp.dimN * jcp.dimM * alpha * alpha
                        * sizeof(float) > 2 * LLC_data_size))
                    vmovntps(EVEX_compress_addr(reg_dstC, output_offset), zmm);
                else vmovups(EVEX_compress_addr(reg_dstC, output_offset), zmm);
            }
        }
    };

    auto inner_loops = [=]() {
        Label dimM_block_loop, dimK_block_loop;

        if (jcp.dimM_block > 1) {
            mov(reg_dimM_block_loop_cnt, jcp.dimM_block);
            L(dimM_block_loop);
        }

        prepare_output();

        if (jcp.dimK_block > 1) {
            mov(reg_dimK_block_loop_cnt, jcp.dimK_block);
            L(dimK_block_loop);
        }

        for (int dimK_reg_block = 0;
                dimK_reg_block < jcp.dimK_reg_block;
                dimK_reg_block ++) {

            if (jcp.kernel_kind == expl_bcast) {
                for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                    vbroadcastss(zmm_srcB(tile),
                        ptr[reg_srcB + 64 * tile + dimK_reg_block * 4]);
                }
            }

            /* Performing the fmas */

            for (int dimM_reg_block = 0; dimM_reg_block < jcp.dimM_reg_block;
                dimM_reg_block++) {

                vmovups(zmm_srcA(),
                    zword[reg_srcA
                            + jcp.dimK_reg_block * jcp.dimK_block * 64
                              * dimM_reg_block
                            + dimK_reg_block * 64]
                    );

                for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                    if (jcp.kernel_kind == expl_bcast)
                        vfmadd231ps(zmm_dstC(dimM_reg_block, tile), zmm_srcA(),
                            zmm_srcB(tile));
                    else
                        vfmadd231ps(zmm_dstC(dimM_reg_block, tile), zmm_srcA(),
                            EVEX_compress_addr(reg_srcB,
                                64 * tile + dimK_reg_block * 4, true));
                }
            }
        }
        add(reg_srcA, jcp.dimK_reg_block * 64);
        add(reg_srcB, jcp.dimN_reg_block * 64);
        if (jcp.dimK_block > 1) {
            sub(reg_dimK_block_loop_cnt, 1);
            jnz(dimK_block_loop);
        }

        Label unaligned_store, end_store;
        test(reg_dstC, cpu_isa_traits<avx512_core>::vlen - 1);
        jnz(unaligned_store, T_NEAR);
        store_output(true);
        jmp(end_store, T_NEAR);
        L(unaligned_store); {
            store_output(false);
        }
        L(end_store);

        if (jcp.dimM_block > 1) {
            sub(reg_srcB, jcp.dimK_block * jcp.dimN_reg_block * 64);
            add(reg_dstC, jcp.dimM_reg_block * jcp.dimN_reg_block * 64);
            if (jcp.kernel_kind == expl_bcast) {
                add(reg_srcA,
                     (jcp.dimM_reg_block-1) * jcp.dimK_reg_block * 64
                      * jcp.dimK_block);
            }
            sub(reg_dimM_block_loop_cnt, 1);
            jnz(dimM_block_loop);
        }
    };

    /* Preamble */
    preamble();

    /* kernel */
    inner_loops();

    /* Postamble */
    postamble();
    ret();
}

void _jit_avx512_core_fp32_wino_conv_4x3_data_kernel
    ::weights_transform_data_ker_generate()
{
    bool is_fwd = one_of(jcp.prop_kind,
        mkldnn_forward_training, mkldnn_forward_inference);
    int kh = jcp.kh;
    int kw = jcp.kw;

    auto zmm_temp = Xbyak::Zmm(31);
    auto zmm_zero = Xbyak::Zmm(30);

    auto zmm_M = [=](int i) {
        return Xbyak::Zmm(i);
    };
    auto zmm_MT = [=](int i) {
        return Xbyak::Zmm(i + simd_w);
    };

    auto zmm_G = [=](int i) {
        return Xbyak::Zmm(i);
    };
    auto zmm_F = [=](int i) {
        return Xbyak::Zmm(alpha + i);
    };
    auto zmm_T = [=](int i) {
        return Xbyak::Zmm(alpha + 3 + i);
    };
    auto zmm_t = [=](int i) {
        return Xbyak::Zmm(2 * alpha + 3 + i);
    };

    auto zmm_load = [=](int i) {
        return Xbyak::Zmm(i);
    };

    auto init_G = [=]() {
        mov(wreg_temp, ptr[param1 + GET_OFF(G)]);
        for (int i = 0; i < alpha; i++) {
            vbroadcastss(zmm_G(i), ptr[wreg_temp + i * typesize]);
        }
        vpxord(zmm_zero, zmm_zero, zmm_zero);
    };

    auto trans16x16 = [=]() {
        for (int i = 0; i < simd_w; i+=2 ) {
            vmovups(zmm_M(i), ptr[wreg_M + i * simd_w * 4]);
            vmovups(zmm_M(i+1), ptr[wreg_M + (i + 1) * simd_w * 4]);
            vunpcklps(zmm_MT(i), zmm_M(i), zmm_M(i+1));
            vunpckhps(zmm_MT(i+1), zmm_M(i), zmm_M(i+1));
        }
        for (int i = 0; i < simd_w; i+=4 ) {
            vunpcklpd(zmm_M(i), zmm_MT(i), zmm_MT(i+2));
            vunpckhpd(zmm_M(i+1), zmm_MT(i), zmm_MT(i+2));
            vunpcklpd(zmm_M(i+2), zmm_MT(i+1), zmm_MT(i+3));
            vunpckhpd(zmm_M(i+3), zmm_MT(i+1), zmm_MT(i+3));
        }
        for (int i = 0; i < simd_w; i += 8) {
            vshuff32x4(zmm_MT(i), zmm_M(i), zmm_M(i + 4), 0x88);
            vshuff32x4(zmm_MT(i+1), zmm_M(i+1), zmm_M(i + 5), 0x88);
            vshuff32x4(zmm_MT(i+2), zmm_M(i+2), zmm_M(i + 6), 0x88);
            vshuff32x4(zmm_MT(i+3), zmm_M(i+3), zmm_M(i + 7), 0x88);
            vshuff32x4(zmm_MT(i+4), zmm_M(i), zmm_M(i + 4), 0xdd);
            vshuff32x4(zmm_MT(i+5), zmm_M(i+1), zmm_M(i + 5), 0xdd);
            vshuff32x4(zmm_MT(i+6), zmm_M(i+2), zmm_M(i + 6), 0xdd);
            vshuff32x4(zmm_MT(i+7), zmm_M(i+3), zmm_M(i + 7), 0xdd);
        }
        {
            int i = 0;
            int mask = 0x88;
            vshuff32x4(zmm_M(0), zmm_MT(i), zmm_MT(i + 8), mask);
            vmovups(ptr[wreg_MT + 0 * 16 * 4], zmm_M(0));
            vshuff32x4(zmm_M(1), zmm_MT(i + 1), zmm_MT(i + 9), mask);
            vmovups(ptr[wreg_MT + 1 * 16 * 4], zmm_M(1));
            vshuff32x4(zmm_M(2), zmm_MT(i + 2), zmm_MT(i + 10), mask);
            vmovups(ptr[wreg_MT + 2 * 16 * 4], zmm_M(2));
            vshuff32x4(zmm_M(3), zmm_MT(i + 3), zmm_MT(i + 11), mask);
            vmovups(ptr[wreg_MT + 3 * 16 * 4], zmm_M(3));
            vshuff32x4(zmm_M(4), zmm_MT(i + 4), zmm_MT(i + 12), mask);
            vmovups(ptr[wreg_MT + 4 * 16 * 4], zmm_M(4));
            vshuff32x4(zmm_M(5), zmm_MT(i + 5), zmm_MT(i + 13), mask);
            vmovups(ptr[wreg_MT + 5 * 16 * 4], zmm_M(5));
            vshuff32x4(zmm_M(6), zmm_MT(i + 6), zmm_MT(i + 14), mask);
            vmovups(ptr[wreg_MT + 6 * 16 * 4], zmm_M(6));
            vshuff32x4(zmm_M(7), zmm_MT(i + 7), zmm_MT(i + 15), mask);
            vmovups(ptr[wreg_MT + 7 * 16 * 4], zmm_M(7));
            mask = 0xdd;
            vshuff32x4(zmm_M(8), zmm_MT(i), zmm_MT(i + 8), mask);
            vmovups(ptr[wreg_MT + 8 * 16 * 4], zmm_M(8));
            vshuff32x4(zmm_M(9), zmm_MT(i + 1), zmm_MT(i + 9), mask);
            vmovups(ptr[wreg_MT + 9 * 16 * 4], zmm_M(9));
            vshuff32x4(zmm_M(10), zmm_MT(i + 2), zmm_MT(i + 10), mask);
            vmovups(ptr[wreg_MT + 10 * 16 * 4], zmm_M(10));
            vshuff32x4(zmm_M(11), zmm_MT(i + 3), zmm_MT(i + 11), mask);
            vmovups(ptr[wreg_MT + 11 * 16 * 4], zmm_M(11));
            vshuff32x4(zmm_M(12), zmm_MT(i + 4), zmm_MT(i + 12), mask);
            vmovups(ptr[wreg_MT + 12 * 16 * 4], zmm_M(12));
            vshuff32x4(zmm_M(13), zmm_MT(i + 5), zmm_MT(i + 13), mask);
            vmovups(ptr[wreg_MT + 13 * 16 * 4], zmm_M(13));
            vshuff32x4(zmm_M(14), zmm_MT(i + 6), zmm_MT(i + 14), mask);
            vmovups(ptr[wreg_MT + 14 * 16 * 4], zmm_M(14));
            vshuff32x4(zmm_M(15), zmm_MT(i + 7), zmm_MT(i + 15), mask);
            vmovups(ptr[wreg_MT + 15 * 16 * 4], zmm_M(15));
        }
    };

    auto load_src = [=]() {
        mov(wreg_src, ptr[param1 + GET_OFF(src)]);
        mov(wreg_F, ptr[param1 + GET_OFF(M)]);
        for (int j = 0; j < kh; j++) {
            for (int i = 0; i < kw; i++) {
                if (is_fwd) {
                    for (int v1 = 0; v1 < simd_w; v1++) {
                        int offset_src = (j * kw * simd_w * simd_w
                            + i * simd_w * simd_w + v1 * simd_w) * typesize;
                        int offset_F = (j * kw * simd_w * simd_w
                            + i * simd_w * simd_w  + v1 * simd_w) * typesize;
                        vmovups(zmm_temp, ptr[wreg_src + offset_src]);
                        vmovups(ptr[wreg_F + offset_F], zmm_temp);
                    }
                } else {
                    int offset_src = ((2 - j) * kw * simd_w * simd_w
                        + (2 - i) * simd_w * simd_w) * typesize;
                    int offset_F = (j * kw * simd_w * simd_w
                        + i * simd_w * simd_w) * typesize;
                    lea(wreg_M, ptr[wreg_src + offset_src]);
                    lea(wreg_MT, ptr[wreg_F + offset_F]);
                    trans16x16();
                }
            }
        }
    };

    auto store_dst = [=]() {
        mov(wreg_dst, ptr[param1 + GET_OFF(dst)]);
        mov(wreg_Fw, ptr[param1 + GET_OFF(Mw)]);

        Label Loop_j;
        mov(wreg_cnt_j, 0);
        mov(wreg_dst_aux, wreg_dst);
        mov(wreg_Fw_aux, wreg_Fw);

        int dim5 = jcp.dimK_nb_block * (jcp.dimM_block * jcp.dimM_reg_block)
            * jcp.dimK_block * simd_w * simd_w;

        L(Loop_j);
        {
            for (int i = 0; i < alpha; i++) {
                // touch pages
                vmovups(zmm_load(0), ptr[wreg_Fw_aux
                    + (i * simd_w * simd_w) * typesize]);
                mov(wreg_dst_idx, i * dim5 * typesize);
                vmovntps(ptr[wreg_dst_aux + wreg_dst_idx], zmm_load(0));
            }
            for (int i = 0; i < alpha; i++) {
                for (int v1 = 1; v1 < simd_w; v1++) {
                    int offset_Fw = (i * simd_w * simd_w  + v1 * simd_w)
                        * typesize;
                    vmovups(zmm_load(v1), ptr[wreg_Fw_aux + offset_Fw]);
                }
                mov(wreg_dst_idx, i * dim5 * typesize);
                for (int v1 = 1; v1 < simd_w; v1++) {
                    int offset_dst = v1 * simd_w * typesize;
                    vmovntps(ptr[wreg_dst_aux + wreg_dst_idx + offset_dst],
                        zmm_load(v1));
                }
            }
            add(wreg_Fw_aux, alpha * simd_w * simd_w * typesize);
            add(wreg_dst_aux, alpha * dim5 * typesize);
            add(wreg_cnt_j, 1);
            cmp(wreg_cnt_j, alpha);
            jl(Loop_j, T_NEAR);
        }
    };

    auto trans_W_4x4_3x3 = [=]() {
        auto fma4 = [=](Zmm dst, Zmm a, Zmm b, Zmm c) {
            vmovups(dst, a);
            vfmadd231ps(dst, b, c);
        };
        auto fms4 = [=](Zmm dst, Zmm a, Zmm b, Zmm c) {
            vmulps(zmm_temp, b, c);
            vsubps(dst, a, zmm_temp);
        };
        auto fnms4 = [=](Zmm dst, Zmm a, Zmm b, Zmm c) {
            vsubps(dst, zmm_zero, a);
            vfnmadd231ps(dst, b, c);
        };

        mov(wreg_Fw, ptr[param1 + GET_OFF(Mw)]);
        mov(wreg_F, ptr[param1 + GET_OFF(M)]);
        mov(wreg_T, ptr[param1 + GET_OFF(T)]);

        Label Loop_j;
        mov(wreg_cnt_j, 0);
        L(Loop_j);
            mov(wreg_F_aux, wreg_F);
            mov(wreg_Fw_aux, wreg_Fw);
            mov(wreg_temp, wreg_cnt_j);
            shl(wreg_temp, 4 + 2);
            lea(wreg_F_aux, ptr[wreg_F + wreg_temp]);
            lea(wreg_Fw_aux, ptr[wreg_Fw + wreg_temp]);

            for (int i = 0; i < 3; i++) {
                for (int idx = 0; idx < 3; idx ++) {
                    vmovups(zmm_F(idx), ptr[wreg_F_aux + (idx * 3 * simd_w
                        * simd_w + i * simd_w * simd_w) * typesize]);
                }
                vmulps(zmm_t(0), zmm_G(0), zmm_F(2));
                fnms4(zmm_t(1), zmm_t(0), zmm_G(1), zmm_F(0));
                fma4(zmm_t(2), zmm_t(0), zmm_G(2), zmm_F(0));

                vmulps(zmm_T(0), zmm_G(3), zmm_F(0));
                fms4(zmm_T(1), zmm_t(1), zmm_G(4), zmm_F(1));
                fma4(zmm_T(2), zmm_t(1), zmm_G(4), zmm_F(1));
                fma4(zmm_T(3), zmm_t(2), zmm_G(5), zmm_F(1));
                fms4(zmm_T(4), zmm_t(2), zmm_G(5), zmm_F(1));
                vmovaps(zmm_T(5), zmm_F(2));

                for (int idx = 0; idx < 6; idx ++) {
                    vmovups(ptr[wreg_T + (idx * 3 * simd_w + i * simd_w)
                        * typesize], zmm_T(idx));
                }
            }
            for (int i = 0; i < 6; i++) {

                for (int idx = 0; idx < 3; idx ++) {
                    vmovups(zmm_T(idx), ptr[wreg_T
                        + (i * 3 * simd_w + idx * simd_w) * typesize]);
                }
                vmulps(zmm_t(0), zmm_G(0), zmm_T(2));
                fnms4(zmm_t(1), zmm_t(0), zmm_G(1), zmm_T(0));
                fma4(zmm_t(2), zmm_t(0), zmm_G(2), zmm_T(0));

                vmulps(zmm_F(0), zmm_G(3), zmm_T(0));
                fms4(zmm_F(1), zmm_t(1), zmm_G(4), zmm_T(1));
                fma4(zmm_F(2), zmm_t(1), zmm_G(4), zmm_T(1));
                fma4(zmm_F(3), zmm_t(2), zmm_G(5), zmm_T(1));
                fms4(zmm_F(4), zmm_t(2), zmm_G(5), zmm_T(1));
                vmovaps(zmm_F(5), zmm_T(2));

                for (int l = 0; l < 6; l++) {
                    vmovups(ptr[wreg_Fw_aux + (i * 6 * simd_w * simd_w
                        + l * simd_w * simd_w) * typesize], zmm_F(l));
                }
            }
        add(wreg_cnt_j, 1);
        cmp(wreg_cnt_j, 16);
        jl(Loop_j, T_NEAR);
    };

    auto inner_loops = [=]() {
        load_src();
        init_G();
        trans_W_4x4_3x3();
        store_dst();
    };

    preamble();
    inner_loops();
    postamble();
}

void _jit_avx512_core_fp32_wino_conv_4x3_data_kernel
    ::output_transform_data_ker_generate()
{
    bool is_fwd = one_of(jcp.prop_kind,
        mkldnn_forward_training, mkldnn_forward_inference);
    int outw = is_fwd ? jcp.ow : jcp.iw;
    int outh = is_fwd ? jcp.oh : jcp.ih;
    bool not_tiled = jcp.sched_policy == WSCHED_DATA_W_S_G_D;
    bool with_bias = jcp.with_bias;
    bool with_relu = jcp.with_eltwise;
    bool with_relu_postsum = jcp.with_relu_postsum;
    bool with_sum = jcp.with_sum;

    auto zmm_zero = Xbyak::Zmm(0);
    auto zmm_temp = Xbyak::Zmm(31);
    auto zmm_G = [=](int i) {
        return Xbyak::Zmm(1 + i);
    };
    auto zmm_O = [=](int i) {
        return Xbyak::Zmm(1 + alpha + i);
    };
    auto zmm_T = [=](int i) {
        return Xbyak::Zmm(1 + 2 * alpha + i);
    };
    auto zmm_t = [=](int i) {
        return Xbyak::Zmm(1 + 3 * alpha + i);
    };

    auto init_G = [=]() {
        mov(oreg_temp, ptr[param1 + GET_OFF(G)]);
        for (int i = 0; i < 6; i++) {
            vbroadcastss(zmm_G(i), ptr[oreg_temp + i * typesize]);
        }
    };

    auto load_src = [=]() {
        mov(oreg_Ow, ptr[param1 + GET_OFF(Mw)]);
        mov(oreg_src, ptr[param1 + GET_OFF(src)]);

        mov(oreg_nb_tile_block_ur, ptr[param1 + GET_OFF(nb_tile_block_ur)]);
        imul(oreg_nb_tile_block_ur, oreg_nb_tile_block_ur,
            (jcp.dimM_block * jcp.dimM_reg_block) * jcp.dimN_reg_block
            * jcp.dimM_simd_block * typesize);
        add(oreg_src, oreg_nb_tile_block_ur);

        mov(oreg_tile_block_ur, ptr[param1 + GET_OFF(tile_block_ur)]);
        imul(oreg_tile_block_ur, oreg_tile_block_ur,
            jcp.dimM_simd_block * typesize);
        add(oreg_src, oreg_tile_block_ur);

        if (not_tiled) {
            mov(oreg_tile_block, ptr[param1 + GET_OFF(tile_block)]);
            imul(oreg_tile_block, oreg_tile_block,
                jcp.dimM_nb_block * alpha * alpha * jcp.dimN_block
                * (jcp.dimM_block * jcp.dimM_reg_block) * jcp.dimN_reg_block
                * jcp.dimM_simd_block * typesize);
            add(oreg_src, oreg_tile_block);
        }

        int last4dim = jcp.dimN_block * (jcp.dimM_block * jcp.dimM_reg_block)
            * jcp.dimN_reg_block * jcp.dimM_simd_block * typesize;
        for (int j = 0; j < alpha; j++) {
            for (int i = 0; i < alpha; i++) {
                int j_base_offset = j * alpha * last4dim;
                int i_base_offset = i * last4dim;
                vmovups(zmm_temp, ptr[oreg_src + j_base_offset + i_base_offset]);
                vmovups(ptr[oreg_Ow + (j * alpha * simd_w + i * simd_w)
                    * typesize], zmm_temp);
            }
        }
    };

    auto store_dst = [=]() {
        vpxord(zmm_zero, zmm_zero, zmm_zero);
        mov(oreg_dst, ptr[param1 + GET_OFF(dst)]);
        mov(oreg_O, ptr[param1 + GET_OFF(M)]);
        mov(oreg_ydim, ptr[param1 + GET_OFF(tj)]);
        shl(oreg_ydim, 2); // tj * tile_size (==4)
        mov(oreg_xdim, ptr[param1 + GET_OFF(ti)]);
        shl(oreg_xdim, 2); // ti * tilesize (==4)

        if (with_bias)
            mov(oreg_bias, ptr[param1 + GET_OFF(bias)]);

        auto store_one = [=](int j, int i, bool is_aligned) {
            auto zmm_O = Xbyak::Zmm(31);
            auto zmm_relu_ns = Xbyak::Zmm(30);
            auto xmm_relu_ns = Xbyak::Xmm(30);
            int offset = (j * tile_size * simd_w + i * simd_w) * typesize;

            vmovups(zmm_O, ptr[oreg_O + offset]);
            if (is_fwd) {
                if (with_bias) {
                    vaddps(zmm_O, zmm_O, ptr[oreg_bias]);
                }
                if (with_relu) {
                    if (jcp.eltwise.alpha == 0) {
                        vmaxps(zmm_O, zmm_O, zmm_zero);
                    } else {
                        Opmask kmask = Opmask(7);
                        mov(imm_addr64, float2int(jcp.eltwise.alpha));
                        vmovq(xmm_relu_ns, imm_addr64);
                        vbroadcastss(zmm_relu_ns, xmm_relu_ns);
                        vcmpps(kmask, zmm_O, zmm_zero, _cmp_lt_os);
                        vmulps(zmm_O | kmask, zmm_O, zmm_relu_ns);
                    }
                }
            }
            if (with_sum) {
                vaddps(zmm_O, zmm_O, ptr[oreg_out_j + oreg_temp]);
                if (with_relu_postsum) // orig: with_relu_postsum
                    vmaxps(zmm_O, zmm_O, zmm_zero);
            }
            if (is_aligned)
                vmovntps(ptr[oreg_out_j + oreg_temp], zmm_O);
            else
                vmovups(ptr[oreg_out_j + oreg_temp], zmm_O);
        };

        auto i_loop = [=](int j, bool is_aligned) {
            for (int i = 0; i < tile_size; i++) {
                Label next;
                mov(oreg_temp, oreg_xdim);
                add(oreg_temp, i);
                cmp(oreg_temp, outw);
                jge(next, T_NEAR);
                shl(oreg_temp, 4 + 2); // * 16 * 4

                store_one(j, i, is_aligned);

                L(next);
            }
        };


        for (int j = 0; j < tile_size; j++) {
            Label next, unaligned;
            mov(oreg_temp, oreg_ydim);
            add(oreg_temp, j);
            cmp(oreg_temp, outh);
            jge(next, T_NEAR);

            mov(oreg_out_j, oreg_dst);
            imul(oreg_temp, oreg_temp, outw * simd_w * typesize);
            add(oreg_out_j, oreg_temp);

            test(oreg_dst, 63);
            jnz(unaligned, T_NEAR);

            i_loop(j, true);
            jmp(next, T_NEAR);

            L(unaligned);
            i_loop(j, false);

            L(next);
        }
    };

    auto trans_O_4x4_3x3 = [=]() {
        auto fma2 = [=](Zmm dst, Zmm v1, Zmm u1, Zmm v2, Zmm u2){
            vmulps(dst, v1, u1);
            vfmadd231ps(dst, v2, u2);
        };
        mov(oreg_Ow, ptr[param1 + GET_OFF(Mw)]);
        mov(oreg_T, ptr[param1 + GET_OFF(T)]);
        mov(oreg_O, ptr[param1 + GET_OFF(M)]);

        for (int i = 0; i < alpha; i++) {
            for (int j = 0; j < alpha; j++) {
                vmovups(zmm_O(j), ptr[oreg_Ow + (j * alpha * simd_w
                    + i * simd_w) * typesize]);
            }

            vaddps(zmm_t(0), zmm_O(1), zmm_O(2));
            vaddps(zmm_t(1), zmm_O(3), zmm_O(4));
            vsubps(zmm_t(2), zmm_O(1), zmm_O(2));
            vsubps(zmm_t(3), zmm_O(3), zmm_O(4));

            vaddps(zmm_T(0), zmm_t(0), zmm_t(1));
            vaddps(zmm_T(0), zmm_T(0), zmm_O(0));
            fma2(zmm_T(1), zmm_t(2), zmm_G(0), zmm_t(3), zmm_G(1));
            fma2(zmm_T(2), zmm_t(0), zmm_G(2), zmm_t(1), zmm_G(3));
            fma2(zmm_T(3), zmm_t(2), zmm_G(4), zmm_t(3), zmm_G(5));
            vaddps(zmm_T(3), zmm_T(3), zmm_O(5));

            for (int j = 0; j < tile_size; j++) {
                vmovups(ptr[oreg_T + (j * alpha * simd_w
                    + i * simd_w) * typesize], zmm_T(j));
            }
        }
        for (int j = 0; j < tile_size; j++) {
            for (int i = 0; i < alpha; i++) {
                vmovups(zmm_T(i), ptr[oreg_T + (j * alpha * simd_w
                    + i * simd_w) * typesize]);
            }
            vaddps(zmm_t(0), zmm_T(1), zmm_T(2));
            vaddps(zmm_t(1), zmm_T(3), zmm_T(4));
            vsubps(zmm_t(2), zmm_T(1), zmm_T(2));
            vsubps(zmm_t(3), zmm_T(3), zmm_T(4));

            vaddps(zmm_O(0), zmm_t(0), zmm_t(1));
            vaddps(zmm_O(0), zmm_O(0), zmm_T(0));
            fma2(zmm_O(1), zmm_t(2), zmm_G(0), zmm_t(3), zmm_G(1));
            fma2(zmm_O(2), zmm_t(0), zmm_G(2), zmm_t(1), zmm_G(3));
            fma2(zmm_O(3), zmm_t(2), zmm_G(4), zmm_t(3), zmm_G(5));
            vaddps(zmm_O(3), zmm_O(3), zmm_T(5));

            for (int i = 0; i < tile_size; i++) {
                vmovups(ptr[oreg_O + (j * tile_size * simd_w
                    + i * simd_w) * typesize], zmm_O(i));
            }
        }
    };

    auto inner_loops = [=]() {
        init_G();
        load_src();
        trans_O_4x4_3x3();
        store_dst();
    };

    preamble();
    inner_loops();
    postamble();
}

void _jit_avx512_core_fp32_wino_conv_4x3_data_kernel
    ::input_transform_data_ker_generate()
{
    bool is_fwd = one_of(jcp.prop_kind,
        mkldnn_forward_training, mkldnn_forward_inference);
    int inpw = is_fwd ? jcp.iw : jcp.ow;
    int inph = is_fwd ? jcp.ih : jcp.oh;
    int l_pad = is_fwd ? jcp.l_pad : jcp.iw + jcp.r_pad - jcp.ow;
    int t_pad = is_fwd ? jcp.t_pad : jcp.ih + jcp.t_pad - jcp.oh;
    int wp_max = inpw + l_pad;
    int hp_max = inph + t_pad;
    bool not_tiled = jcp.sched_policy == WSCHED_DATA_W_S_G_D;
    int G_size = 9;

    auto zmm_zero = Xbyak::Zmm(0);
    auto zmm_temp = Xbyak::Zmm(31);
    auto zmm_G = [=](int i) {
        return Xbyak::Zmm(1 + i);
    };
    auto zmm_I = [=](int i) {
        return Xbyak::Zmm(1 + G_size + i);
    };
    auto zmm_T = [=](int i) {
        return Xbyak::Zmm(1 + G_size + alpha + i);
    };
    auto zmm_t = [=](int i) {
        return Xbyak::Zmm(1 + G_size + 2 * alpha + i);
    };

    auto init_G = [=]() {
        mov(ireg_temp, ptr[param1 + GET_OFF(G)]);
        for (int i = 0; i < G_size; i++) {
            vbroadcastss(zmm_G(i), ptr[ireg_temp + i * typesize]);
        }
    };

    auto load_src = [=]() {
        mov(ireg_src, ptr[param1 + GET_OFF(src)]); // base addr of inp
        mov(ireg_I, ptr[param1 + GET_OFF(M)]);

        xor_(ireg_zero,  ireg_zero);
        vpxord(zmm_zero, zmm_zero, zmm_zero);

        mov(ireg_ydim, ptr[param1 + GET_OFF(tj)]);
        shl(ireg_ydim, 2); // tj * tile_size (==4)
        mov(ireg_xdim, ptr[param1 + GET_OFF(ti)]);
        shl(ireg_xdim, 2); // ti * tilesize (==4)

        for (int j = 0; j < alpha; j++) {
            mov(ireg_temp, ireg_ydim);
            add(ireg_temp, j);

            mov(ireg_mask_j, 0xffff);
            cmp(ireg_temp, t_pad);
            cmovl(ireg_mask_j, ireg_zero);
            cmp(ireg_temp, hp_max);
            cmovge(ireg_mask_j, ireg_zero);

            sub(ireg_temp, t_pad);
            imul(ireg_temp, ireg_temp, inpw * simd_w * typesize);
            mov(ireg_inp_j, ireg_src);
            add(ireg_inp_j, ireg_temp);

            for (int i = 0; i < alpha; i++) {

                mov(ireg_temp, ireg_xdim);
                add(ireg_temp, i);

                mov(ireg_mask, 0xffff);
                cmp(ireg_temp, l_pad);
                cmovl(ireg_mask, ireg_zero);
                cmp(ireg_temp, wp_max);
                cmovge(ireg_mask, ireg_zero);
                and_(ireg_mask, ireg_mask_j);

                sub(ireg_temp, l_pad);
                shl(ireg_temp, 4 + 2);

                vpxord(zmm_temp, zmm_temp, zmm_temp);
                Opmask kmask = Opmask(7);
                kmovw(kmask, ireg_mask_32);
                vmovups(zmm_temp | kmask, ptr[ireg_inp_j + ireg_temp]);
                vmovups(ptr[ireg_I + (j * alpha * simd_w + i * simd_w)
                    * typesize], zmm_temp);
            }
        }
    };

    auto store_Iw = [=]() {

        mov(ireg_Iw, ptr[param1 + GET_OFF(Mw)]);
        mov(ireg_output, ptr[param1 + GET_OFF(dst)]);

       bool streamout
          = jcp.dimN * jcp.dimK * alpha * alpha * sizeof(float)
            > 2 * LLC_data_size
            ? true : false;

        if (not_tiled) {
            mov(ireg_tile_block, ptr[param1 + GET_OFF(tile_block)]);
            imul(ireg_tile_block, ireg_tile_block,
                alpha * alpha * jcp.dimN_block * jcp.dimK_nb_block
                * jcp.dimK_block * jcp.dimN_reg_block * jcp.dimK_reg_block
                * typesize);
        }

        mov(ireg_nb_tile_block_ur, ptr[param1 + GET_OFF(nb_tile_block_ur)]);
        imul(ireg_nb_tile_block_ur, ireg_nb_tile_block_ur,
            jcp.dimK_nb_block * jcp.dimK_block * jcp.dimN_reg_block
            * jcp.dimK_reg_block * typesize);

        mov(ireg_tile_block_ur, ptr[param1 + GET_OFF(tile_block_ur)]);
        imul(ireg_tile_block_ur, ireg_tile_block_ur,
            jcp.dimK_reg_block * typesize);

        add(ireg_output, ireg_nb_tile_block_ur);
        add(ireg_output, ireg_tile_block_ur);
        if (not_tiled)
            add(ireg_output, ireg_tile_block);

        for (int j = 0; j < alpha; j++) {
            for (int i = 0; i < alpha; i++) {
                vmovups(zmm_temp,ptr[ireg_Iw + (j * alpha * simd_w
                    + i * simd_w) * typesize]);

                int j_base_offset =
                    j * alpha * jcp.dimN_block * jcp.dimK_nb_block
                    * jcp.dimK_block * jcp.dimN_reg_block * jcp.dimK_reg_block
                    * typesize;
                int i_base_offset =
                    i * jcp.dimN_block * jcp.dimK_nb_block * jcp.dimK_block
                    * jcp.dimN_reg_block * jcp.dimK_reg_block * typesize;

                if (not_tiled && streamout)
                    vmovntps(ptr[ireg_output + j_base_offset + i_base_offset],
                        zmm_temp);
                else
                    vmovups(ptr[ireg_output + j_base_offset + i_base_offset],
                        zmm_temp);
            }
        }
    };

    auto fma4 = [=](Zmm dst, Zmm a, Zmm b, Zmm c) {
        vmulps(zmm_temp, a, b);
        vaddps(dst, zmm_temp, c);
    };

    auto trans_I_4x4_3x3 = [=]() {
        mov(ireg_Iw, ptr[param1 + GET_OFF(Mw)]);
        mov(ireg_T, ptr[param1 + GET_OFF(T)]);
        mov(ireg_I, ptr[param1 + GET_OFF(M)]);

        mov(ireg_output, ptr[param1 + GET_OFF(dst)]); // for prefetch
        for (int i = 0; i < alpha; i++) {
            for (int idx = 0; idx < alpha; idx++) {
                vmovups(zmm_I(idx), ptr[ireg_I + (idx * alpha * simd_w
                    + i * simd_w) * typesize]);
                int j_base_offset =
                    i * alpha * jcp.dimN_block * jcp.dimK_nb_block
                    * jcp.dimK_block * jcp.dimN_reg_block * jcp.dimK_reg_block
                    * typesize;
                int idx_base_offset =
                    idx * jcp.dimN_block * jcp.dimK_nb_block * jcp.dimK_block
                    * jcp.dimN_reg_block * jcp.dimK_reg_block * typesize;
                prefetcht0(ptr[ireg_output + j_base_offset + idx_base_offset]);
            }

            fma4(zmm_t(0), zmm_I(2), zmm_G(0), zmm_I(4));
            fma4(zmm_t(1), zmm_I(1), zmm_G(0), zmm_I(3));
            fma4(zmm_t(2), zmm_I(2), zmm_G(1), zmm_I(4));
            fma4(zmm_t(3), zmm_I(1), zmm_G(1), zmm_I(3));
            fma4(zmm_t(4), zmm_I(0), zmm_G(2), zmm_I(4));
            fma4(zmm_t(5), zmm_I(1), zmm_G(2), zmm_I(5));

            fma4(zmm_T(0), zmm_I(2), zmm_G(3), zmm_t(4));
            fma4(zmm_T(1), zmm_t(1), zmm_G(4), zmm_t(0));
            fma4(zmm_T(2), zmm_t(1), zmm_G(5), zmm_t(0));
            fma4(zmm_T(3), zmm_t(3), zmm_G(6), zmm_t(2));
            fma4(zmm_T(4), zmm_t(3), zmm_G(7), zmm_t(2));
            fma4(zmm_T(5), zmm_I(3), zmm_G(8), zmm_t(5));

            for (int idx = 0; idx < alpha; idx++) {
                vmovups(ptr[ireg_T + (idx * alpha * simd_w + i * simd_w)
                    * typesize],zmm_T(idx));
            }
        }
        for (int i = 0; i < alpha; i++) {
            for (int idx = 0; idx < alpha; idx++) {
                vmovups(zmm_T(idx), ptr[ireg_T + (i * alpha * simd_w + idx
                    * simd_w) * typesize]);
            }

            fma4(zmm_t(0), zmm_T(2), zmm_G(0), zmm_T(4));
            fma4(zmm_t(1), zmm_T(1), zmm_G(0), zmm_T(3));
            fma4(zmm_t(2), zmm_T(2), zmm_G(1), zmm_T(4));
            fma4(zmm_t(3), zmm_T(1), zmm_G(1), zmm_T(3));
            fma4(zmm_t(4), zmm_T(0), zmm_G(2), zmm_T(4));
            fma4(zmm_t(5), zmm_T(1), zmm_G(2), zmm_T(5));

            fma4(zmm_I(0), zmm_T(2), zmm_G(3), zmm_t(4));
            fma4(zmm_I(1), zmm_t(1), zmm_G(4), zmm_t(0));
            fma4(zmm_I(2), zmm_t(1), zmm_G(5), zmm_t(0));
            fma4(zmm_I(3), zmm_t(3), zmm_G(6), zmm_t(2));
            fma4(zmm_I(4), zmm_t(3), zmm_G(7), zmm_t(2));
            fma4(zmm_I(5), zmm_T(3), zmm_G(8), zmm_t(5));

            for (int idx = 0; idx < alpha; idx++) {
                vmovups(ptr[ireg_Iw + (i * alpha * simd_w + idx * simd_w)
                    * typesize],zmm_I(idx));
            }
        }
    };

    auto inner_loops = [=]() {
        init_G();
        load_src();
        trans_I_4x4_3x3();
        store_Iw();
    };

    preamble();
    inner_loops();
    postamble();
}

status_t _jit_avx512_core_fp32_wino_conv_4x3_data_kernel::init_conf_common(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d)
{
    if (!mayiuse(avx512_core)) {
        return status::unimplemented;
    }

    jcp.nthr = mkldnn_get_max_threads();

    jcp.ver = ver_avx512_core;
    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    jcp.r_pad = nstl::max(
            0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);
    jcp.b_pad = nstl::max(
            0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih - jcp.t_pad);
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;

    bool ok_to_pad_channels = jcp.ngroups == 1;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    // Checking conditions not supported by these kernels
    if (!IMPLICATION(cd.alg_kind == alg_kind::convolution_auto,
               is_winograd_faster_than_direct(jcp)))
        return status::unimplemented;

    if (jcp.ngroups != 1)
        return status::unimplemented;
    if ((jcp.kh != 3) || (jcp.kw != 3))
        return status::unimplemented;
    if ((jcp.dilate_h != 0) || (jcp.dilate_w != 0))
        return status::unimplemented;
    if ((jcp.stride_h != 1) || (jcp.stride_w != 1))
        return status::unimplemented;
    if ((jcp.ic % simd_w) != 0 || (jcp.oc % simd_w) != 0)
        return status::unimplemented;

    format_tag_t dat_tag = nChw16c;
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);

    if (jcp.src_tag != dat_tag) return status::unimplemented;
    if (jcp.dst_tag != dat_tag) return status::unimplemented;

    if (!one_of(weights_d.format_kind(), format_kind::any, format_kind::wino)) {
        format_tag_t wei_tag = with_groups ? gOIhw16i16o : OIhw16i16o;
        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
        if (jcp.wei_tag != wei_tag)
            return status::unimplemented;
    }

    bool layout_consistency = true
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1]
            && (one_of(weights_d.format_kind(),
                        format_kind::any, format_kind::wino)
                    || (jcp.ic <= weights_d.padded_dims()[with_groups + 1]
                        && jcp.oc <= weights_d.padded_dims()[with_groups + 0]));
    if (!layout_consistency)
        return status::unimplemented;

    return status::success;
}

void set_kernel_dims_reg_block(jit_conv_winograd_conf_t &jcp) {

    /* ----------- dimM reg block ---------------------*/
    auto test_cond_dimM_reg_block = [](jit_conv_winograd_conf_t &jcp,
            int dimM_reg_block, int current_best) {
        int max_dimM_reg_block = jcp.kernel_kind == embd_bcast ? 1 : 4;
        return (dimM_reg_block >= 1)
                && (dimM_reg_block <= max_dimM_reg_block )
                && (dimM_reg_block > current_best);
    };
    jcp.dimM_reg_block = get_divisor_satisfying_cond(jcp,
        jcp.dimM/jcp.dimM_simd_block, 1, test_cond_dimM_reg_block);

    /* ----------- dimN reg block ---------------------*/

    auto test_cond_dimN_reg_block = [](jit_conv_winograd_conf_t &jcp,
            int dimN_reg_block, int current_best) {
        return jcp.kernel_kind == embd_bcast
            ? dimN_reg_block < jcp.nb_reg && dimN_reg_block > current_best
            : dimN_reg_block >= 1
              && (dimN_reg_block * jcp.dimM_reg_block + dimN_reg_block)
                 < jcp.nb_reg
              && dimN_reg_block > current_best;
    };
    jcp.dimN_reg_block = get_divisor_satisfying_cond(jcp,
        jcp.dimN, 1, test_cond_dimN_reg_block);
}

status_t set_wsched_DATA_W_SGD_avx512_core(jit_conv_winograd_conf_t &jcp) {
    if (jcp.ver != ver_avx512_core)
        return status::unimplemented;

    jcp.kernel_kind = embd_bcast;

    set_kernel_dims_reg_block(jcp);

    /*-------------- L2 blocking for dimN block ---------*/

    auto test_cond_dimN_block = [](jit_conv_winograd_conf_t &jcp,
        int dimN_block, int current_best) {
        return check_L2_block_per_thread(jcp, dimN_block, 0.1, 2.0)
            && (dimN_block > current_best)
            && ((jcp.dimN / dimN_block / jcp.dimN_reg_block)
            >= 1.5 * mkldnn_get_max_threads());
    };

    jcp.dimN_block = get_divisor_satisfying_cond(
            jcp, jcp.dimN / jcp.dimN_reg_block, 1, test_cond_dimN_block);
    jcp.dimN_nb_block = jcp.dimN / jcp.dimN_block / jcp.dimN_reg_block;

    if (check_L2_block_per_thread(jcp, jcp.dimN_block, 0.1, 3.2)
        && (jcp.dimN_nb_block >= 1.5 * mkldnn_get_max_threads())) {

        /* ------------------- L1 blocking for GEMM --------------*/
        /* -------------------- Choose dimK block ----------------*/

        auto test_cond_dimK_block = [](jit_conv_winograd_conf_t &jcp,
                int dimK_block, int current_best) {
            return check_L1_block_gemm(jcp, dimK_block, 1, 0.1, 0.5)
                && (dimK_block > current_best);
        };

        jcp.dimK_block = get_divisor_satisfying_cond(
                jcp, jcp.dimK / jcp.dimK_reg_block, 1, test_cond_dimK_block);

        if (check_L1_block_gemm(jcp, jcp.dimK_block, 1, 0.1, 1.0)) {
            jcp.dimK_nb_block = jcp.dimK / jcp.dimK_block / jcp.dimK_reg_block;

            /* -------------- Choose dimM block -------------------*/
            auto test_cond_dimM_block = [](jit_conv_winograd_conf_t &jcp,
                    int dimM_block, int current_best) {
                return check_L1_block_gemm(jcp, jcp.dimK_block, dimM_block,
                    0.2, 0.5) && (dimM_block > current_best);
            };

            jcp.dimM_block = get_divisor_satisfying_cond(jcp,
                jcp.dimM / (jcp.dimM_simd_block * jcp.dimM_reg_block), 1,
                test_cond_dimM_block);
            jcp.dimM_nb_block = jcp.dimM / jcp.dimM_block / jcp.dimM_reg_block
                / jcp.dimM_simd_block;

            jcp.sched_policy = WSCHED_DATA_W_SGD;
            return status::success;
        }

    }
    return status::unimplemented;
}

void set_kernel_blocking_DATA_W_S_G_D(jit_conv_winograd_conf_t &jcp) {

    set_kernel_dims_reg_block(jcp);

    //********************* Choosing dimK_block **********************//
    auto test_cond1_dimK_block = [](
            jit_conv_winograd_conf_t &jcp, int dimK_block, int current_best) {
        return check_cond1(jcp.dimN_reg_block, dimK_block, jcp.dimK_reg_block,
                       1, jcp.dimM_reg_block, jcp.dimM_simd_block, .75f)
                && (dimK_block > current_best);
    };

    auto test_cond1_bis_dimK_block = [](
            jit_conv_winograd_conf_t &jcp, int dimK_block, int current_best) {
        return check_cond1_bis(jcp.dimN_reg_block, dimK_block,
                   jcp.dimK_reg_block, 1, jcp.dimM_reg_block,
                   jcp.dimM_simd_block, .9f)
                && (dimK_block > current_best);
    };

    jcp.dimK_block = get_divisor_satisfying_cond(
            jcp, jcp.dimK / jcp.dimK_reg_block, 1, test_cond1_bis_dimK_block);
    // If we are not able to use streams, we fall back to condition [1]
    if (jcp.dimK_block < jcp.dimK / jcp.dimK_reg_block)
        jcp.dimK_block = get_divisor_satisfying_cond(
                jcp, jcp.dimK / jcp.dimK_reg_block, 1, test_cond1_dimK_block);
    jcp.dimK_nb_block = (jcp.dimK / jcp.dimK_reg_block) / jcp.dimK_block;

    //********************* Choosing dimM_block **********************//
    auto test_cond1_dimM_block = [](
            jit_conv_winograd_conf_t &jcp, int dimM_block, int current_best) {
        return check_cond1(jcp.dimN_reg_block, jcp.dimK_block,
                   jcp.dimK_reg_block, dimM_block, jcp.dimM_reg_block,
                   jcp.dimM_simd_block, .5f)
                && (dimM_block > current_best);
    };

    auto test_cond1_bis_dimM_block = [](
            jit_conv_winograd_conf_t &jcp, int dimM_block, int current_best) {
        return check_cond1_bis(jcp.dimN_reg_block, jcp.dimK_block,
                   jcp.dimK_reg_block, dimM_block, jcp.dimM_reg_block,
                   jcp.dimM_simd_block, .3f)
                && (dimM_block > current_best);
    };

    if (jcp.dimK_block < jcp.dimK / jcp.dimK_reg_block)
        jcp.dimM_block = get_divisor_satisfying_cond(
                jcp, jcp.dimM / (jcp.dimM_simd_block*jcp.dimM_reg_block), 1,
                test_cond1_dimM_block);
    else
        jcp.dimM_block = get_divisor_satisfying_cond(jcp,
                jcp.dimM / (jcp.dimM_simd_block*jcp.dimM_reg_block), 1,
                test_cond1_bis_dimM_block);
    jcp.dimM_nb_block = jcp.dimM / (jcp.dimM_simd_block * jcp.dimM_block
                        * jcp.dimM_reg_block);

    //******************* Choosing dimN_block *******************//
    auto test_cond2_dimN_block = [](
            jit_conv_winograd_conf_t &jcp, int dimN_block, int current_best) {
        return check_cond2(dimN_block, jcp.dimN_reg_block, jcp.dimK_nb_block,
                       jcp.dimK_block, jcp.dimK_reg_block, jcp.dimM_block,
                       jcp.dimM_reg_block, jcp.dimM_simd_block, .9f)
                && (dimN_block > current_best);
    };

    jcp.dimN_block = get_divisor_satisfying_cond(
            jcp, jcp.dimN / jcp.dimN_reg_block, 1, test_cond2_dimN_block);
    jcp.dimN_nb_block = jcp.dimN / (jcp.dimN_reg_block * jcp.dimN_block);
}

status_t set_wsched_DATA_W_S_G_D_avx512_core(jit_conv_winograd_conf_t &jcp) {

    jcp.kernel_kind = expl_bcast;
    set_kernel_blocking_DATA_W_S_G_D(jcp);
    if (!(check_kernel_cond(jcp.dimM_block, jcp.dimM_reg_block,
        jcp.dimM_simd_block, jcp.dimN_block, jcp.dimN_reg_block, jcp.dimK,
        .1f, .35f))) {
        jcp.kernel_kind = embd_bcast;
        set_kernel_blocking_DATA_W_S_G_D(jcp);
    }
    jcp.sched_policy = WSCHED_DATA_W_S_G_D;
    return status::success;
}

status_t _jit_avx512_core_fp32_wino_conv_4x3_data_kernel::init_conf_kernel(
        jit_conv_winograd_conf_t &jcp, int dimM, int dimN, int dimK)
{
    jcp.nb_reg = 32;
    jcp.dimN = dimN;
    jcp.dimK = dimK;
    jcp.dimM = dimM;
    jcp.sched_policy = WSCHED_INVALID;

    jcp.dimK_reg_block = 16;
    jcp.dimM_simd_block = 16;

    if (jcp.kernel_kind == embd_bcast) {
        jcp.dimM_reg_block = 1;
    }

    if (!(set_wsched_DATA_W_SGD_avx512_core(jcp) == status::success))
        set_wsched_DATA_W_S_G_D_avx512_core(jcp);

    assert(jcp.sched_policy != WSCHED_INVALID);
    return status::success;
}

bool jit_avx512_core_fp32_wino_conv_4x3_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_relu = [&](int idx) { return p.entry_[idx].is_relu(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1: return is_relu(0) || is_sum(0); // relu or sum
    case 2: return (is_sum(0) && is_relu(1))
                      || (is_relu(0) && is_sum(1)); // sum->relu or relu->sum
    case 3: return is_relu(0) && is_sum(1) && is_relu(2); // relu->sum->relu
    default: return false;
    }

    return false;
}

status_t jit_avx512_core_fp32_wino_conv_4x3_fwd_kernel::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_t &src_md, memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const primitive_attr_t &attr) {

    status_t st = init_conf_common(jcp, cd, src_md, weights_md, dst_md);

    if (st != status::success)
        return st;

    // Winograd specific initialization
    jcp.itiles = (jcp.ow + tile_size - 1) / tile_size;
    jcp.jtiles = (jcp.oh + tile_size - 1) / tile_size;
    jcp.ntiles = jcp.mb * jcp.itiles * jcp.jtiles;

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise, 0, 1);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise)
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    jcp.with_sum = p.find(primitive_kind::sum, 0) != -1;
    jcp.with_relu_postsum = p.find(primitive_kind::eltwise, 1) != -1;

    status_t res = init_conf_kernel(jcp, jcp.oc, jcp.ntiles, jcp.ic);

    jcp.ic_simd_block = jcp.dimK_reg_block;
    jcp.ic_block = jcp.dimK_block;
    jcp.nb_ic = jcp.dimK_nb_block;
    jcp.oc_simd_block = jcp.dimM_simd_block;
    jcp.oc_block = jcp.dimM_block;
    jcp.oc_reg_block = jcp.dimM_reg_block;
    jcp.ic_reg_block = 1;
    jcp.nb_oc = jcp.dimM_nb_block;
    jcp.tile_block_ur = jcp.dimN_reg_block;
    jcp.nb_tile_block_ur = jcp.dimN_block;
    jcp.tile_block = jcp.dimN_nb_block;

    /* re-create weights primitive descriptor
    and set weights wino_blocking */
    if (cd.prop_kind == mkldnn_forward_inference) {
        memory_desc_t expect_wei_md = weights_md;

        expect_wei_md.format_kind = format_kind::wino;
        expect_wei_md.data_type = data_type::f32;
        mkldnn_wino_desc_t &wd = expect_wei_md.format_desc.wino_desc;
        wd.wino_format = mkldnn_wino_wei_OBaaIBOIio;
        wd.r = 3;
        wd.alpha = 6;

        wd.ic = jcp.ic;
        wd.oc = jcp.oc;
        wd.ic_block = jcp.dimK_reg_block;
        wd.oc_block = jcp.dimM_simd_block;
        wd.ic2_block = jcp.dimK_block;
        wd.oc2_block = jcp.dimM_block * jcp.dimM_reg_block;
        size_t max_size = sizeof(float) * wd.alpha * wd.alpha * jcp.ic * jcp.oc;
        wd.size = max_size;
        wd.adj_scale = 1.f;

        if (weights_md.format_kind == format_kind::any)
            weights_md = expect_wei_md;
        if (weights_md != expect_wei_md)
            return status::unimplemented;
    }

    return res;
}

status_t jit_avx512_core_fp32_wino_conv_4x3_bwd_data_kernel::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    status_t st = init_conf_common(jcp, cd, diff_src_d, weights_d, diff_dst_d);

    if (st != status::success)
        return st;

    jcp.itiles = (jcp.iw + tile_size - 1) / tile_size;
    jcp.jtiles = (jcp.ih + tile_size - 1) / tile_size;
    jcp.ntiles = jcp.mb * jcp.itiles * jcp.jtiles;

    status_t res = init_conf_kernel(jcp, jcp.ic, jcp.ntiles, jcp.oc);

    jcp.oc_simd_block = jcp.dimK_reg_block;
    jcp.oc_block = jcp.dimK_block;
    jcp.nb_oc = jcp.dimK_nb_block;
    jcp.ic_simd_block = jcp.dimM_simd_block;
    jcp.ic_block = jcp.dimM_block;
    jcp.ic_reg_block = jcp.dimM_reg_block;
    jcp.oc_reg_block = 1;
    jcp.nb_ic = jcp.dimM_nb_block;
    jcp.tile_block_ur = jcp.dimN_reg_block;
    jcp.nb_tile_block_ur = jcp.dimN_block;
    jcp.tile_block = jcp.dimN_nb_block;

    return res;
}

void jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_kernel::
src_transform_generate() {
    constexpr int G_size = 9;
    const size_t ifwp = jcp.iw + jcp.l_pad;
    const size_t ifhp = jcp.ih + jcp.t_pad;

    auto zmm_G = [=](int i) {
        return Xbyak::Zmm(i);
    };
    auto zmm_I = [=](int i) {
        return Xbyak::Zmm(G_size + i);
    };
    auto zmm_T = [=](int i) {
        return Xbyak::Zmm(G_size + alpha + i);
    };
    auto zmm_t = [=](int i) {
        return Xbyak::Zmm(G_size + 2 * alpha + i);
    };

    auto init_G = [=]() {
        mov(reg_G, ptr[reg_transp + GET_OFF(G)]);
        for (int i = 0; i < G_size; i++) {
            vbroadcastss(zmm_G(i), ptr[reg_G + i * typesize]);
        }
    };

    auto load_src = [=]() {
        mov(reg_I, ptr[reg_transp + GET_OFF(M)]);
        xor_(reg_zero, reg_zero);

        mov(reg_ydim, reg_tj);
        shl(reg_ydim, 2); //tj * tile_size(=4)

        for (int j = 0; j < alpha; j++) {
            /* check if tile index is within physical spatial boundaries*/
            mov(reg_maskj, 0xffff);
            cmp(reg_ydim, jcp.t_pad);
            cmovl(reg_maskj, reg_zero);
            cmp(reg_ydim, ifhp);
            cmovge(reg_maskj, reg_zero);

            /*address offset for tile in src*/
            mov(reg_src_offset, reg_ydim);
            sub(reg_src_offset, jcp.t_pad); // tj*tile_size - t_pad
            imul(reg_src_offset, reg_src_offset, jcp.iw);

            mov(reg_xdim, reg_ti);
            shl(reg_xdim, 2); // xdim = ti * tile_size

            add(reg_src_offset, reg_xdim);
            sub(reg_src_offset, jcp.l_pad);
            imul(reg_src_offset, reg_src_offset, simd_w * typesize);
            for (int i = 0; i < alpha; i++) {
                /* check if tile index is within physical spatial boundaries*/
                mov(reg_maski, 0xffff);
                cmp(reg_xdim, jcp.l_pad);
                cmovl(reg_maski, reg_zero);
                cmp(reg_xdim, ifwp);
                cmovge(reg_maski, reg_zero);
                and_(reg_maski, reg_maskj);

                Opmask kmask_src = Xbyak::Opmask(7);
                auto zmm_src = Xbyak::Zmm(31);
                kmovw(kmask_src, reg_maski_32);
                vpxord(zmm_src, zmm_src, zmm_src);
                vmovups(zmm_src | kmask_src, ptr[reg_src + reg_src_offset]);
                vmovups(ptr[reg_I], zmm_src);

                add(reg_xdim, 1); //xdim = ti * tile_size + i
                add(reg_src_offset, simd_w * typesize);
                add(reg_I, simd_w * typesize);
            }
            add(reg_ydim, 1);
        }
    };

    auto fma4 = [=](Xbyak::Zmm dst, Xbyak::Zmm a, Xbyak::Zmm b, Xbyak::Zmm c) {
        vmovups(dst, c);
        vfmadd231ps(dst, a, b);
    };

    auto trans_I_3x3_4x4 = [=]() {
        //Use 24 registers
        mov(reg_I, ptr[reg_transp + GET_OFF(M)]);
        mov(reg_T, ptr[reg_transp + GET_OFF(T)]);
        for (int i = 0; i < alpha; i++) {
            for (int j = 0; j < alpha; j++) {
                size_t I_off = (j * alpha + i) * simd_w * typesize;
                vmovups(zmm_I(j), ptr[reg_I + I_off]);
            }

            fma4(zmm_t(0), zmm_I(2), zmm_G(0), zmm_I(4));
            fma4(zmm_t(1), zmm_I(1), zmm_G(0), zmm_I(3));
            fma4(zmm_t(2), zmm_I(2), zmm_G(1), zmm_I(4));
            fma4(zmm_t(3), zmm_I(1), zmm_G(1), zmm_I(3));
            fma4(zmm_t(4), zmm_I(0), zmm_G(2), zmm_I(4));
            fma4(zmm_t(5), zmm_I(1), zmm_G(2), zmm_I(5));

            fma4(zmm_T(0), zmm_I(2), zmm_G(3), zmm_t(4));
            fma4(zmm_T(1), zmm_t(1), zmm_G(4), zmm_t(0));
            fma4(zmm_T(2), zmm_t(1), zmm_G(5), zmm_t(0));
            fma4(zmm_T(3), zmm_t(3), zmm_G(6), zmm_t(2));
            fma4(zmm_T(4), zmm_t(3), zmm_G(7), zmm_t(2));
            fma4(zmm_T(5), zmm_I(3), zmm_G(8), zmm_t(5));

            for (int j = 0; j < alpha; j++) {
                vmovups(ptr[reg_T + (j * alpha + i) * simd_w * typesize],
                        zmm_T(j));
            }

        }

        for (int j = 0; j < alpha; j++) {
            for (int i = 0; i < alpha; i++) {
                vmovups(zmm_T(i), ptr[reg_T + (j * alpha + i) * simd_w * typesize]);
            }

            fma4(zmm_t(0), zmm_T(2), zmm_G(0), zmm_T(4));
            fma4(zmm_t(1), zmm_T(1), zmm_G(0), zmm_T(3));
            fma4(zmm_t(2), zmm_T(2), zmm_G(1), zmm_T(4));
            fma4(zmm_t(3), zmm_T(1), zmm_G(1), zmm_T(3));
            fma4(zmm_t(4), zmm_T(0), zmm_G(2), zmm_T(4));
            fma4(zmm_t(5), zmm_T(1), zmm_G(2), zmm_T(5));

            fma4(zmm_I(0), zmm_T(2), zmm_G(3), zmm_t(4));
            fma4(zmm_I(1), zmm_t(1), zmm_G(4), zmm_t(0));
            fma4(zmm_I(2), zmm_t(1), zmm_G(5), zmm_t(0));
            fma4(zmm_I(3), zmm_t(3), zmm_G(6), zmm_t(2));
            fma4(zmm_I(4), zmm_t(3), zmm_G(7), zmm_t(2));
            fma4(zmm_I(5), zmm_T(3), zmm_G(8), zmm_t(5));

            for (int i = 0; i < alpha; i++) {
                size_t dst_off = (j * alpha * jcp.ic_block
                    * jcp.nb_tile_block_ur * jcp.tile_block_ur
                    + i * jcp.ic_block * jcp.nb_tile_block_ur * jcp.tile_block_ur)
                    * simd_w * typesize;
                vmovups(ptr[reg_dst + dst_off], zmm_I(i));
            }
        }
    };

    auto compute_transform_SDGtWo = [=]() {
        mov(reg_ti, ptr[reg_transp + GET_OFF(ti)]);
        mov(reg_tj, ptr[reg_transp + GET_OFF(tj)]);
        mov(reg_src, ptr[reg_transp + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_transp + GET_OFF(dst)]);
        xor_(reg_tile_count, reg_tile_count);
        Label loop_mb, loop_jtiles, loop_itiles, done;
        L(loop_mb);
        {
            L(loop_jtiles);
            {
                L(loop_itiles);
                {
                    load_src();

                    trans_I_3x3_4x4();

                    add(reg_tile_count, 1);
                    cmp(reg_tile_count, jcp.nb_tile_block_ur * jcp.tile_block_ur);
                    jge(done);

                    add(reg_dst, simd_w * typesize);
                    add(reg_ti, 1);
                    cmp(reg_ti, jcp.itiles);
                    jl(loop_itiles);
                }
                xor_(reg_ti, reg_ti);
                add(reg_tj, 1);
                cmp(reg_tj, jcp.jtiles);
                jl(loop_jtiles);
            }
            xor_(reg_tj, reg_tj);
            add(reg_src, jcp.ic * jcp.iw * jcp.ih * typesize);
            jmp(loop_mb);
        }
        L(done);
    };

    auto compute_transform = [=]() {
        mov(reg_src, ptr[reg_transp + GET_OFF(src)]);
        xor_(reg_ti, reg_ti);
        xor_(reg_tj, reg_tj);

        mov(reg_dst, ptr[reg_transp + GET_OFF(dst)]);
        mov(reg_tile_count, ptr[reg_transp + GET_OFF(tile_count)]);
        imul(reg_temp, reg_tile_count, simd_w * typesize);
        add(reg_dst, reg_temp);

        Label loop_jtiles, loop_itiles, next_tile_block, next_tile;
        L(loop_jtiles);

        {
            L(loop_itiles);
            {
                load_src();

                trans_I_3x3_4x4();

                add(reg_tile_count, 1);
                cmp(reg_tile_count, jcp.nb_tile_block_ur * jcp.tile_block_ur);
                jge(next_tile_block);
                add(reg_dst, simd_w * typesize);
                jmp(next_tile);

                L(next_tile_block);
                sub(reg_dst, (jcp.nb_tile_block_ur * jcp.tile_block_ur - 1)
                        * simd_w * typesize);
                size_t tblk_off = alpha * alpha * jcp.ic_block
                    * jcp.nb_tile_block_ur * jcp.tile_block_ur
                    * simd_w * typesize;
                add(reg_dst, tblk_off);
                xor_(reg_tile_count, reg_tile_count);

                L(next_tile);
                add(reg_ti, 1);
                cmp(reg_ti, jcp.itiles);
                jl(loop_itiles);
            }
            xor_(reg_ti, reg_ti);
            add(reg_tj, 1);
            cmp(reg_tj, jcp.jtiles);
            jl(loop_jtiles);
        }
    };

    preamble();
    init_G();
    if (jcp.sched_policy == WSCHED_WEI_SDGtWo)
        compute_transform_SDGtWo();
    else
        compute_transform();
    postamble();
}

void jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_kernel::
diff_dst_transform_generate(bool with_bias) {

    constexpr int G_size = 8;
    auto zmm_G = [](int i) {
        return Xbyak::Zmm(31);
    };

    auto zmm_src = [=](int j, int i) {
        return Xbyak::Zmm(G_size + j * 4 + i);
    };

    auto zmm_bias = Xbyak::Zmm(31);

    auto load_src = [=]() {
        if (with_bias) vmovups(zmm_bias, ptr[reg_bias]);
        mov(reg_ydim, reg_tj);
        shl(reg_ydim, 2); //tj * tile_size(=4)
        for (int j = 0; j < tile_size; j++) {
            /* check if tile index is within physical spatial boundaries*/
            mov(reg_maskj, 0xffff);
            cmp(reg_ydim, jcp.oh);
            cmovge(reg_maskj, reg_zero);

            /*address offset for tile in src*/
            mov(reg_src_offset, reg_ydim);
            imul(reg_src_offset, reg_src_offset, jcp.ow);

            mov(reg_xdim, reg_ti);
            shl(reg_xdim, 2); // xdim = ti * tile_size

            add(reg_src_offset, reg_xdim);
            imul(reg_src_offset, reg_src_offset, simd_w * typesize);
            for (int i = 0; i < tile_size; i++) {
                /* check if tile index is within physical spatial boundaries*/
                mov(reg_maski, 0xffff);
                cmp(reg_xdim, jcp.ow);
                cmovge(reg_maski, reg_zero);
                and_(reg_maski, reg_maskj);

                Opmask kmask_src = Xbyak::Opmask(7);
                kmovw(kmask_src, reg_maski_32);
                vpxord(zmm_src(j, i), zmm_src(j, i), zmm_src(j, i));
                vmovups(zmm_src(j, i) | kmask_src, ptr[reg_src + reg_src_offset]);
                if (with_bias) vaddps(zmm_bias | kmask_src, zmm_bias,
                        ptr[reg_src + reg_src_offset]);

                add(reg_xdim, 1); //xdim = ti * tile_size + i
                add(reg_src_offset, simd_w * typesize);
            }
            add(reg_ydim, 1);
        }
        if(with_bias) vmovups(ptr[reg_bias], zmm_bias);
    };

    auto zmm_t = [=](int i) {
        return Xbyak::Zmm(G_size + 16 + i);
    };

    auto zmm_T = [=](int j, int i) {
        return Xbyak::Zmm(j * 4 + i);
    };

    auto movps = [=](Xbyak::Reg64 reg_dst, size_t dst_off, Xbyak::Zmm a) {
        if (jcp.sched_policy == WSCHED_WEI_SDGtWo)
            vmovups(ptr[reg_dst + dst_off], a);
        else
            vmovntps(ptr[reg_dst + dst_off], a);
    };

    auto trans_W_3x3_4x4 = [=]() {
        mov(reg_G, ptr[reg_transp + GET_OFF(G)]);
        for (int i = 0; i < tile_size; i++) {
            vbroadcastss(zmm_G(0), ptr[reg_G]);
            vmulps(zmm_t(0), zmm_src(2, i), zmm_G(0));

            vbroadcastss(zmm_G(1), ptr[reg_G + typesize]);
            vmovups(zmm_t(1), zmm_t(0));
            vfmsub231ps(zmm_t(1), zmm_src(0, i), zmm_G(1));

            vbroadcastss(zmm_G(2), ptr[reg_G + 2 * typesize]);
            vmovups(zmm_t(2), zmm_t(0));
            vfmadd231ps(zmm_t(2), zmm_src(0, i), zmm_G(2));

            vbroadcastss(zmm_G(3), ptr[reg_G + 3 * typesize]);
            vmulps(zmm_t(3), zmm_src(1, i), zmm_G(3));

            vbroadcastss(zmm_G(4), ptr[reg_G + 4 * typesize]);
            vfmadd231ps(zmm_t(3), zmm_src(3, i), zmm_G(4));

            vbroadcastss(zmm_G(5), ptr[reg_G + 5 * typesize]);
            vmulps(zmm_t(4), zmm_src(1, i), zmm_G(5));

            vbroadcastss(zmm_G(6), ptr[reg_G + 6 * typesize]);
            vfmadd231ps(zmm_t(4), zmm_src(3, i), zmm_G(6));

            vbroadcastss(zmm_G(7), ptr[reg_G + 7 * typesize]);
            vmulps(zmm_T(0, i), zmm_src(0, i), zmm_G(7));
            vsubps(zmm_T(1, i), zmm_t(1), zmm_t(3));
            vaddps(zmm_T(2, i), zmm_t(1), zmm_t(3));
            vaddps(zmm_T(3, i), zmm_t(2), zmm_t(4));
            vsubps(zmm_T(4, i), zmm_t(2), zmm_t(4));
            vmovups(zmm_T(5, i), zmm_src(3, i));
        }

        for (int j = 0; j < alpha; j++) {
            vbroadcastss(zmm_G(0), ptr[reg_G]);
            vmulps(zmm_t(0), zmm_T(j, 2), zmm_G(0));

            vbroadcastss(zmm_G(1), ptr[reg_G + typesize]);
            vmovups(zmm_t(1), zmm_t(0));
            vfmsub231ps(zmm_t(1), zmm_T(j, 0), zmm_G(1));

            vbroadcastss(zmm_G(2), ptr[reg_G + 2 * typesize]);
            vmovups(zmm_t(2), zmm_t(0));
            vfmadd231ps(zmm_t(2), zmm_T(j, 0), zmm_G(2));

            vbroadcastss(zmm_G(3), ptr[reg_G + 3 * typesize]);
            vmulps(zmm_t(3), zmm_T(j, 1), zmm_G(3));

            vbroadcastss(zmm_G(4), ptr[reg_G + 4 * typesize]);
            vfmadd231ps(zmm_t(3), zmm_T(j, 3), zmm_G(4));

            vbroadcastss(zmm_G(5), ptr[reg_G + 5 * typesize]);
            vmulps(zmm_t(4), zmm_T(j, 1), zmm_G(5));

            vbroadcastss(zmm_G(6), ptr[reg_G + 6 * typesize]);
            vfmadd231ps(zmm_t(4), zmm_T(j, 3), zmm_G(6));

            vbroadcastss(zmm_G(7), ptr[reg_G + 7 * typesize]);
            vmulps(zmm_t(0), zmm_T(j, 0), zmm_G(7));
            vsubps(zmm_t(5), zmm_t(1), zmm_t(3));
            vaddps(zmm_t(1), zmm_t(1), zmm_t(3));
            vaddps(zmm_t(6), zmm_t(2), zmm_t(4));
            vsubps(zmm_t(2), zmm_t(2), zmm_t(4));
            vmovups(zmm_t(3), zmm_T(j, 3));

            int alpha_offset = (jcp.oc / jcp.nb_oc)
                * (jcp.ntiles / jcp.tile_block) * typesize;
            int dst_off = j * alpha * alpha_offset;
            movps(reg_dst, dst_off, zmm_t(0));
            dst_off += alpha_offset;
            movps(reg_dst, dst_off, zmm_t(5));
            dst_off += alpha_offset;
            movps(reg_dst, dst_off, zmm_t(1));
            dst_off += alpha_offset;
            movps(reg_dst, dst_off, zmm_t(6));
            dst_off += alpha_offset;
            movps(reg_dst, dst_off, zmm_t(2));
            dst_off += alpha_offset;
            movps(reg_dst, dst_off, zmm_t(3));
        }

    };
    auto compute_transform_SDGtWo = [=]() {
        mov(reg_src, ptr[reg_transp + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_transp + GET_OFF(dst)]);
        if (with_bias) mov(reg_bias, ptr[reg_transp + GET_OFF(bias)]);

        xor_(reg_zero, reg_zero);
        xor_(reg_oc_ur, reg_oc_ur);
        Label loop_mb, loop_jtiles, loop_itiles, loop_oc_ur, tiles_done;

        L(loop_oc_ur);
        {
            mov(reg_ti, ptr[reg_transp + GET_OFF(ti)]);
            mov(reg_tj, ptr[reg_transp + GET_OFF(tj)]);
            xor_(reg_tile_count, reg_tile_count);
            L(loop_mb);
            {
                L(loop_jtiles);
                {
                    L(loop_itiles);
                    {
                        load_src();

                        trans_W_3x3_4x4();

                        add(reg_tile_count, 1);
                        cmp(reg_tile_count, jcp.nb_tile_block_ur * jcp.tile_block_ur);
                        jge(tiles_done);

                        add(reg_dst, jcp.oc_reg_block * simd_w * typesize);
                        add(reg_ti, 1);
                        cmp(reg_ti, jcp.itiles);
                        jl(loop_itiles);
                    }
                    xor_(reg_ti, reg_ti);
                    add(reg_tj, 1);
                    cmp(reg_tj, jcp.jtiles);
                    jl(loop_jtiles);
                }
                xor_(reg_tj, reg_tj);
                add(reg_src, jcp.oc * jcp.ow * jcp.oh * typesize);
                jmp(loop_mb);
            }

            L(tiles_done);
            mov(reg_dst, ptr[reg_transp + GET_OFF(dst)]);
            add(reg_dst, simd_w * typesize);
            mov(reg_src, ptr[reg_transp + GET_OFF(src)]);
            add(reg_src, jcp.oh * jcp.ow * simd_w * typesize);

            if (with_bias) add(reg_bias, simd_w * typesize);
            add(reg_oc_ur, 1);
            cmp(reg_oc_ur, jcp.oc_reg_block);
            jl(loop_oc_ur);
        }
    };

    auto compute_transform = [=]() {
        mov(reg_src, ptr[reg_transp + GET_OFF(src)]);
        mov(reg_G, ptr[reg_transp + GET_OFF(G)]);
        if (with_bias) mov(reg_bias, ptr[reg_transp + GET_OFF(bias)]);

        mov(reg_dst, ptr[reg_transp + GET_OFF(dst)]);
        mov(reg_tile_count, ptr[reg_transp + GET_OFF(tile_count)]);
        imul(reg_temp, reg_tile_count, jcp.oc_reg_block * simd_w * typesize);
        add(reg_dst, reg_temp);

        xor_(reg_zero, reg_zero);
        xor_(reg_oc_ur, reg_oc_ur);
        Label loop_mb, loop_jtiles, loop_itiles, loop_oc_ur, next_tile_block, next_tile;

        L(loop_oc_ur);
        {
            xor_(reg_ti, reg_ti);
            xor_(reg_tj, reg_tj);

            L(loop_jtiles);
            {
                L(loop_itiles);
                {
                    load_src();

                    trans_W_3x3_4x4();

                    add(reg_tile_count, 1);
                    cmp(reg_tile_count, jcp.nb_tile_block_ur * jcp.tile_block_ur);
                    jge(next_tile_block);
                    add(reg_dst, jcp.oc_reg_block * simd_w * typesize);
                    jmp(next_tile);

                    L(next_tile_block);
                    sub(reg_dst, (jcp.nb_tile_block_ur * jcp.tile_block_ur - 1)
                            * jcp.oc_reg_block * simd_w * typesize);
                    int tblk_off = alpha * alpha * (jcp.oc/jcp.nb_oc)
                        * (jcp.ntiles/jcp.tile_block) * typesize;
                    add(reg_dst, tblk_off);
                    xor_(reg_tile_count, reg_tile_count);

                    L(next_tile);
                    add(reg_ti, 1);
                    cmp(reg_ti, jcp.itiles);
                    jl(loop_itiles);
                }
                xor_(reg_ti, reg_ti);
                add(reg_tj, 1);
                cmp(reg_tj, jcp.jtiles);
                jl(loop_jtiles);
            }

            mov(reg_dst, ptr[reg_transp + GET_OFF(dst)]);
            mov(reg_tile_count, ptr[reg_transp + GET_OFF(tile_count)]);
            imul(reg_temp, reg_tile_count, jcp.oc_reg_block * simd_w * typesize);
            add(reg_dst, reg_temp);
            add(reg_dst, simd_w * typesize);
            mov(reg_src, ptr[reg_transp + GET_OFF(src)]);
            add(reg_src, jcp.oh * jcp.ow * simd_w * typesize);

            if (with_bias) add(reg_bias, simd_w * typesize);
            add(reg_oc_ur, 1);
            cmp(reg_oc_ur, jcp.oc_reg_block);
            jl(loop_oc_ur);
        }
    };

    preamble();
    if (jcp.sched_policy == WSCHED_WEI_SDGtWo) {
        compute_transform_SDGtWo();
    } else {
        compute_transform();
    }
    postamble();
}

void jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_kernel::
diff_weights_transform_generate(bool first_tile) {
    int G_size = 4;

    auto zmm_G = [](int i) {
        return Xbyak::Zmm(i);
    };

    auto init_G = [=]() {
        mov(reg_G, ptr[reg_transp + GET_OFF(G)]);
        for (int i = 0; i  < G_size; i++)
            vbroadcastss(zmm_G(i), ptr[reg_G + i * typesize]);
    };

    auto zmm_src = [=](int i) {
        return Xbyak::Zmm(G_size + i);
    };

    auto load_src = [=](int i) {
        for (int j = 0; j < alpha; j++) {
            size_t alpha_offset = jcp.oc_block * jcp.oc_reg_block
                * jcp.ic_block * simd_w * simd_w * typesize;
            size_t src_off = (j * alpha + i) * alpha_offset;
            vmovups(zmm_src(j), EVEX_compress_addr(reg_src, src_off));
        }
    };

    auto zmm_t = [=](int i) {
        return Xbyak::Zmm(G_size + 6 + i);
    };

    auto zmm_T = [=](int j, int i) {
        return Xbyak::Zmm(G_size + 6 + 3 + j * 6 + i);
    };

    auto zmm_dst = [=](int i) {
        return Xbyak::Zmm(G_size + i);
    };

    auto zmm_temp = Xbyak::Zmm(31);

    auto store_dst = [=](int j) {
        for (int i = 0; i < jcp.kw; i++) {
            size_t dst_off = (j * jcp.kw + i) * simd_w * simd_w * typesize;

            if (!first_tile) {
                vmovups(zmm_temp, EVEX_compress_addr(reg_dst, dst_off));
                vaddps(zmm_dst(i), zmm_dst(i), zmm_temp);
            }
            vmovntps(EVEX_compress_addr(reg_dst, dst_off), zmm_dst(i));
        }
    };

    auto compute_transform = [=] () {
        mov(reg_src, ptr[reg_transp + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_transp + GET_OFF(dst)]);

        xor_(reg_ic_simd, reg_ic_simd);
        Label loop_ic_simd;
        L(loop_ic_simd);
        {
            for (int i = 0; i < alpha; i++) {
                load_src(i);

                vaddps(zmm_t(0), zmm_src(1), zmm_src(2));
                vaddps(zmm_t(1), zmm_src(3), zmm_src(4));
                vmovups(zmm_t(2), zmm_src(5));
                vfmadd231ps(zmm_t(2), zmm_t(1), zmm_G(0));

                vaddps(zmm_T(0, i), zmm_src(0), zmm_t(0));
                vaddps(zmm_T(0, i), zmm_T(0, i), zmm_t(1));
                vsubps(zmm_T(1, i), zmm_src(1), zmm_src(2));
                vmulps(zmm_T(1, i), zmm_T(1, i), zmm_G(1));
                vsubps(zmm_temp, zmm_src(3), zmm_src(4));
                vfmadd231ps(zmm_T(1, i), zmm_temp, zmm_G(2));
                vmovups(zmm_T(2, i), zmm_t(2));
                vfmadd231ps(zmm_T(2, i), zmm_t(0), zmm_G(3));
            }

            for (int j = 0; j < jcp.kh; j++) {
                vaddps(zmm_t(0), zmm_T(j, 1), zmm_T(j, 2));
                vaddps(zmm_t(1), zmm_T(j, 3), zmm_T(j, 4));
                vmovups(zmm_t(2), zmm_T(j, 5));
                vfmadd231ps(zmm_t(2), zmm_t(1), zmm_G(0));

                vaddps(zmm_dst(0), zmm_T(j, 0), zmm_t(0));
                vaddps(zmm_dst(0), zmm_dst(0), zmm_t(1));
                vsubps(zmm_dst(1), zmm_T(j, 1), zmm_T(j, 2));
                vmulps(zmm_dst(1), zmm_dst(1), zmm_G(1));
                vsubps(zmm_temp, zmm_T(j, 3), zmm_T(j, 4));
                vfmadd231ps(zmm_dst(1), zmm_temp, zmm_G(2));
                vmovups(zmm_dst(2), zmm_t(2));
                vfmadd231ps(zmm_dst(2), zmm_t(0), zmm_G(3));

                store_dst(j);
            }

            add(reg_src, jcp.oc_reg_block * simd_w * typesize);
            add(reg_dst, simd_w * typesize);
            add(reg_ic_simd, 1);
            cmp(reg_ic_simd, simd_w);
            jl(loop_ic_simd);
        }
    };
    preamble();
    push(reg_EVEX_max_8b_offt);
    mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
    init_G();
    compute_transform();
    pop(reg_EVEX_max_8b_offt);
    postamble();
}

void jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_kernel::gemm_loop_generate(
        bool is_first_tile)
{
    auto zmm_srcA = [=]() {
        return Xbyak::Zmm(0);
    };

    auto zmm_srcB = [=] (size_t N_ur){
        return Xbyak::Zmm(N_ur + 1);
    };

    auto broadcastB = [=](size_t K_ur) {
        for (int N_bcast = 0; N_bcast < jcp.dimN_bcast_ur; N_bcast++) {
            size_t srcB_off = (K_ur * jcp.dimN_reg_block + N_bcast)
                * sizeof(float);
            vbroadcastss(zmm_srcB(N_bcast), EVEX_compress_addr(reg_srcB, srcB_off));
        }
    };

    auto load_srcA = [=] (size_t K_ur, int M_ur) {
        size_t srcA_off = (K_ur * jcp.dimM_reg_block * jcp.dimM_simd_block
                        + M_ur * jcp.dimM_simd_block) * sizeof(float);
        vmovups(zmm_srcA(), EVEX_compress_addr(reg_srcA, srcA_off));
    };

    auto zmm_dstC = [=](size_t M_reg_ur, int N_bcast){
        size_t idx = 1 // zmm_srcA
            + jcp.dimN_bcast_ur // zmm_srcB
            + M_reg_ur * jcp.dimN_bcast_ur + N_bcast;
        assert(idx < 32);
        return Xbyak::Zmm(idx);
    };
    auto prepare_accumm = [=](){
        for (int M_reg_ur = 0; M_reg_ur < jcp.dimM_reg_block; M_reg_ur++) {
            for (int N_bcast = 0; N_bcast < jcp.dimN_bcast_ur; N_bcast++) {
                Zmm zmm = zmm_dstC(M_reg_ur, N_bcast);
                vpxord(zmm, zmm, zmm);
            }
        }
    };

    auto store_dstC = [=](){
        /******** Write C back to memory *******/
        for (int M_reg = 0; M_reg < jcp.dimM_reg_block; M_reg++) {
            for (int N_ur = 0; N_ur < jcp.dimN_bcast_ur; ++N_ur) {
                Zmm zmm = zmm_dstC(M_reg, N_ur);
                size_t C_off = (N_ur * jcp.dimM_reg_block * jcp.dimM_simd_block
                             + M_reg * jcp.dimM_simd_block) * sizeof(float);
                if (!is_first_tile) {
                    vmovups(Xbyak::Zmm(0), EVEX_compress_addr(reg_dstC, C_off));
                    vaddps(zmm, zmm, Xbyak::Zmm(0));
                }
                vmovups(EVEX_compress_addr(reg_dstC, C_off), zmm);
            }
        }
    };

    auto inner_loops = [=]() {
        Label dimM_block_loop, dimK_block_loop, dimN_block_loop, dimN_bcast_ur;

        mov(reg_dimM_block_loop_cnt, jcp.dimM_block);
        L(dimM_block_loop);
        { /************* OC_block (M) loop ***********/
            mov(reg_dimN_block_loop_cnt, jcp.dimN_block);
            L(dimN_block_loop);
            { /*************** IC_block (N) loop *********/

                mov(reg_nb_dimN_bcast_ur, jcp.dimN_reg_block/jcp.dimN_bcast_ur);
                L(dimN_bcast_ur);
                {
                    prepare_accumm();

                    mov(reg_dimK_block_loop_cnt, jcp.dimK_block);
                    L(dimK_block_loop);
                    {
                     /************* nb_tile_ur(K) loop ********/
                        for (int K_ur = 0; K_ur < jcp.dimK_reg_block; K_ur++) {

                            broadcastB(K_ur);

                            for (int M_reg_ur = 0; M_reg_ur < jcp.dimM_reg_block; M_reg_ur++) {
                                load_srcA(K_ur, M_reg_ur);
                                for (int N_bcast = 0; N_bcast < jcp.dimN_bcast_ur; ++N_bcast) {
                                    vfmadd231ps(zmm_dstC(M_reg_ur, N_bcast), zmm_srcA(),
                                            zmm_srcB(N_bcast));
                                }
                            }
                        }
                        add(reg_srcA, jcp.dimK_reg_block
                                      * jcp.dimM_reg_block * jcp.dimM_simd_block
                                      * sizeof(float));
                        add(reg_srcB, jcp.dimK_reg_block
                                      * jcp.dimN_reg_block
                                      * sizeof(float));
                        sub(reg_dimK_block_loop_cnt, 1);
                        jnz(dimK_block_loop);
                    }

                    store_dstC();

                    sub(reg_srcA, jcp.dimK_block * jcp.dimK_reg_block
                                  * jcp.dimM_reg_block * jcp.dimM_simd_block
                                  * sizeof(float));
                    sub(reg_srcB, jcp.dimK_block * jcp.dimK_reg_block
                                  * jcp.dimN_reg_block
                                  * sizeof(float));
                    add(reg_srcB, jcp.dimN_bcast_ur * sizeof(float));
                    add(reg_dstC, jcp.dimN_bcast_ur
                            * jcp.dimM_reg_block * jcp.dimM_simd_block
                            * sizeof(float));
                    sub(reg_nb_dimN_bcast_ur, 1);
                    jnz(dimN_bcast_ur);
                }

                sub(reg_srcB, jcp.dimN_reg_block * sizeof(float));
                add(reg_srcB, jcp.dimK_block
                        * jcp.dimK_reg_block
                        * jcp.dimN_reg_block * sizeof(float));
                sub(reg_dimN_block_loop_cnt, 1);
                jnz(dimN_block_loop);
            }

            sub(reg_srcB, jcp.dimN_block
                          * jcp.dimK_block * jcp.dimK_reg_block
                          * jcp.dimN_reg_block
                          * sizeof(float));
            add(reg_srcA, jcp.dimK_block * jcp.dimK_reg_block
                          * jcp.dimM_reg_block * jcp.dimM_simd_block
                          * sizeof(float));
            sub(reg_dimM_block_loop_cnt, 1);
            jnz(dimM_block_loop);
        }
    };

    /* Preamble */
    preamble();

    inner_loops();

    /* Postamble */
    postamble();
    ret();
}

namespace {

void set_jcp_WEI_params(jit_conv_winograd_conf_t &jcp) {
/*M params*/
    jcp.dimM_nb_block = jcp.dimM / jcp.dimM_block / jcp.dimM_reg_block
        / jcp.dimM_simd_block;
    jcp.oc_reg_block = jcp.dimM_reg_block;
    jcp.oc_block = jcp.dimM_block;
    jcp.nb_oc = jcp.dimM_nb_block;
    /*N params*/
    jcp.dimN_nb_block = jcp.dimN / jcp.dimN_block / jcp.dimN_reg_block;
    jcp.ic_block = jcp.dimN_block;
    jcp.nb_ic = jcp.dimN_nb_block;

    /*K params*/
    jcp.dimK_nb_block = jcp.dimK / jcp.dimK_block / jcp.dimK_reg_block;
    jcp.tile_block_ur = jcp.dimK_reg_block;
    jcp.nb_tile_block_ur = jcp.dimK_block;
    jcp.tile_block = jcp.dimK_nb_block;
}

status_t set_wsched_WEI_SDGtWo(jit_conv_winograd_conf_t &jcp) {

    size_t K_blk_ur, N_blk, M_blk;
    /* IS this strategy feasible? */
    auto test_MV_large_enough = [](jit_conv_winograd_conf_t &jcp) {
        size_t M_sz = alpha * alpha * jcp.dimM * jcp.dimK * sizeof(float);
        size_t V_sz = alpha * alpha * jcp.dimN * jcp.dimK * sizeof(float);
        size_t nthreads = mkldnn_get_max_threads();
        return (((V_sz + M_sz) / nthreads) >= 2 * L2_cache_size)
            && (jcp.dimK / nthreads >= 1.0);
    };

    auto test_min_dimK_L1 = [](jit_conv_winograd_conf_t &jcp, int dimK_block_ur,
            int max_block=1) {
        size_t L1_block_M  = jcp.dimM_reg_block * jcp.dimM_simd_block * dimK_block_ur * sizeof(float);
        size_t L1_block_N = jcp.dimN_reg_block * dimK_block_ur * sizeof(float);
        size_t M_L2_block = alpha * alpha * jcp.dimM * dimK_block_ur * sizeof(float);
        size_t nthreads = mkldnn_get_max_threads();
        bool load_balance=true;
        if (!(jcp.dimK % nthreads)) {
            load_balance = ((jcp.dimK / dimK_block_ur) % nthreads == 0);
        }
        return (L1_block_M + L1_block_N >= 0.1 * L1_cache_size)
            && (L1_block_M + L1_block_N <= 0.5 * L1_cache_size)
            && load_balance
            && (M_L2_block < L2_cache_size);
    };

    auto test_dimK_ur = [](jit_conv_winograd_conf_t &jcp, int dimK_ur,
            int useless_arg=0) {
        return (dimK_ur >= 2) && (dimK_ur <= 8);
    };

    auto blocking_ok =  [&](){
        size_t M_L2_block = alpha * alpha * M_blk * jcp.dimM_reg_block * jcp.dimM_simd_block
                          * K_blk_ur * sizeof(float);
        size_t V_L2_block = alpha * alpha * N_blk * jcp.dimN_reg_block
                          * K_blk_ur * sizeof(float);
        size_t U_L2_block = alpha * alpha * M_blk * jcp.dimM_reg_block * jcp.dimM_simd_block
                          * N_blk * jcp.dimN_reg_block * sizeof(float);
        size_t L2_block = M_L2_block + V_L2_block + U_L2_block;
        /*Replace 2.375 with L2+L3 cache size*/
        return (L2_block > 0.1 * L2_cache_size) && (L2_block <= 1.2 * L2_cache_size);
    };

    if (test_MV_large_enough(jcp)) {
        if ((jcp.dimM/jcp.dimM_simd_block) % 2 == 0) {
            jcp.dimM_reg_block = 2;
        } else {
            jcp.dimM_reg_block = 1;
        }
        jcp.dimM_simd_block = jcp.oc_simd_block;
        jcp.dimN_reg_block = jcp.ic_simd_block;
        jcp.dimN_bcast_ur = 8;
        /*dimK_block and dimK_ur*/
        size_t min_dimK_block_ur = get_divisor_satisfying_cond(jcp, jcp.dimK, 1, test_min_dimK_L1);

        jcp.dimM_block = jcp.dimM/jcp.dimM_reg_block/jcp.dimM_simd_block;
        jcp.dimN_block = jcp.dimN/jcp.dimN_reg_block;
        for (K_blk_ur = min_dimK_block_ur; K_blk_ur >= 1; --K_blk_ur) {
            if (test_min_dimK_L1(jcp, K_blk_ur) && !(jcp.dimK % K_blk_ur)) {
                for (N_blk = jcp.dimN_block; N_blk >= 1; --N_blk) {
                    if (!(jcp.dimN_block % N_blk)) {
                        for (M_blk = jcp.dimM_block; M_blk >= 1; --M_blk) {
                            if (!(jcp.dimM_block % M_blk) && blocking_ok()) {
                                jcp.dimK_reg_block = get_divisor_satisfying_cond(jcp, K_blk_ur, 1, test_dimK_ur);
                                if (!test_dimK_ur(jcp, jcp.dimK_reg_block)) return status::unimplemented;
                                jcp.dimK_block = K_blk_ur / jcp.dimK_reg_block;
                                jcp.dimN_block = N_blk;
                                jcp.dimM_block = M_blk;
                                jcp.sched_policy = WSCHED_WEI_SDGtWo;
                                set_jcp_WEI_params(jcp);
                                jcp.nthr = nstl::min(mkldnn_get_max_threads(),
                                        jcp.tile_block);
                                return status::success;
                            }
                        }
                    }
                }
            }
        }
    }
    return status::unimplemented;
}

status_t set_wsched_WEI_S_D_Giot_W(jit_conv_winograd_conf_t &jcp) {
    if ((jcp.dimM/jcp.dimM_simd_block) % 2 == 0) {
        jcp.dimM_reg_block = 2;
    } else {
        jcp.dimM_reg_block = 1;
    }
    jcp.dimN_bcast_ur = 8;
    jcp.dimN_reg_block = jcp.ic_simd_block;
    jcp.dimM_simd_block = jcp.oc_simd_block;
    jcp.dimN_block = jcp.dimN / jcp.dimN_reg_block;
    jcp.dimM_block = jcp.dimM / jcp.dimM_reg_block / jcp.dimM_simd_block;
    float C1 = 0.0, C2 = 0.0;
    float C1_max = 0.5, C2_max = 1.4;
    int N_blk, M_blk, K_blk_ur;

    auto test_dimK_ur = [](jit_conv_winograd_conf_t &jcp, int dimK_ur,
            int useless_arg=0) {
        return (dimK_ur >= 2) && (dimK_ur <= 8);
    };

    auto blocking_ok = [&]() -> bool {
        size_t L1_block_M  = jcp.dimM_reg_block * jcp.dimM_simd_block * K_blk_ur * sizeof(float);
        size_t L1_block_N = jcp.dimN_reg_block * K_blk_ur * sizeof(float);
        bool L1_cond = ((L1_block_N + L1_block_M) >= C1 * L1_cache_size)
                     && ((L1_block_N + L1_block_M) <= C1_max * L1_cache_size);

        size_t nb_N_blk = jcp.dimN/N_blk/jcp.dimN_reg_block;
        size_t nb_M_blk = jcp.dimM/M_blk/jcp.dimM_reg_block/jcp.dimM_simd_block;
        size_t nb_K_blk = jcp.dimK / K_blk_ur;
        size_t nthreads = mkldnn_get_max_threads();
        bool load_balance = (nb_K_blk * nb_N_blk * nb_M_blk) >= nthreads;
        if (!(nb_K_blk % nthreads)) {
            load_balance = load_balance && (nb_K_blk % nthreads == 0);
        }

        size_t V_L2_block = alpha * alpha * N_blk * jcp.dimN_reg_block * K_blk_ur * sizeof(float);

        size_t L2_block = V_L2_block;
        /*Replace 2.375 with L2+L3 cache size*/
        bool L2_cond = (L2_block >= C2 * L2_cache_size) && (L2_block <= C2_max * L2_cache_size);
        return L1_cond && load_balance && L2_cond;
    };

    for (K_blk_ur = jcp.dimK; K_blk_ur >= 1; --K_blk_ur) {
        if (jcp.dimK % K_blk_ur == 0) {
            for (N_blk = jcp.dimN_block; N_blk >= 1; --N_blk) {
                if (jcp.dimN_block % N_blk == 0) {
                    for (M_blk = jcp.dimM_block; M_blk >= 1; --M_blk) {
                        if (jcp.dimM_block % M_blk == 0) {
                            if (blocking_ok()) {
                                jcp.dimN_block = N_blk;
                                jcp.dimM_block = M_blk;
                                jcp.dimK_reg_block = get_divisor_satisfying_cond(jcp, K_blk_ur, 1, test_dimK_ur);
                                jcp.dimK_block = K_blk_ur / jcp.dimK_reg_block;
                                jcp.sched_policy = WSCHED_WEI_S_D_Giot_W;
                                set_jcp_WEI_params(jcp);
                                return status::success;
                            }
                        }
                    }
                }
            }
        }
    }
    jcp.dimK_reg_block = 1;
    jcp.dimK_block = 1;
    jcp.sched_policy = WSCHED_WEI_S_D_Giot_W;
    set_jcp_WEI_params(jcp);
    return status::success;
}
} // namespace
status_t jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_kernel::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &diff_dst_d,
        const memory_desc_wrapper &diff_weights_d) {
    if (!mayiuse(avx512_core))
        return status::unimplemented;
    else
        jcp.ver = ver_avx512_core;

    jcp.nthr = mkldnn_get_max_threads();

    jcp.prop_kind = cd.prop_kind;
    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    jcp.mb = src_d.dims()[0];
    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];
    jcp.kh = diff_weights_d.dims()[with_groups + 2];
    jcp.kw = diff_weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.r_pad = nstl::max(
            0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);
    jcp.b_pad = nstl::max(
            0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih - jcp.t_pad);
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.with_bias = (cd.diff_bias_desc.format_kind != format_kind::undef);
    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    bool ok_to_pad_channels = jcp.ngroups == 1;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    // Winograd specific initialization
    jcp.itiles = (jcp.ow + tile_size - 1) / tile_size;
    jcp.jtiles = (jcp.oh + tile_size - 1) / tile_size;
    jcp.ntiles = jcp.mb * jcp.itiles * jcp.jtiles;

    // Winograd kernel works only for 3x3 convolution with stride 1
    if (!IMPLICATION(cd.alg_kind == alg_kind::convolution_auto,
               is_winograd_faster_than_direct(jcp)))
        return status::unimplemented;

    if (jcp.ngroups != 1)
        return status::unimplemented;
    if ((jcp.kh != 3) || (jcp.kw != 3))
        return status::unimplemented;
    if ((jcp.dilate_h != 0) || (jcp.dilate_w != 0))
        return status::unimplemented;
    if ((jcp.stride_h != 1) || (jcp.stride_w != 1))
        return status::unimplemented;
    if ((jcp.ic % simd_w) != 0 || (jcp.oc % simd_w) != 0)
        return status::unimplemented;

    format_tag_t dat_tag = nChw16c;
    format_tag_t wei_tag = with_groups ? gOIhw16i16o : OIhw16i16o;
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.wei_tag = diff_weights_d.matches_one_of_tag(wei_tag);
    jcp.dst_tag = diff_dst_d.matches_one_of_tag(dat_tag);

    if (jcp.src_tag != dat_tag) return status::unimplemented;
    if (jcp.wei_tag != wei_tag) return status::unimplemented;
    if (jcp.dst_tag != dat_tag) return status::unimplemented;

    bool layout_consistency = true
        && jcp.ic <= src_d.padded_dims()[1]
        && jcp.oc <= diff_dst_d.padded_dims()[1]
        && jcp.ic <= diff_weights_d.padded_dims()[with_groups + 1]
        && jcp.oc <= diff_weights_d.padded_dims()[with_groups + 0];
    if (!layout_consistency) return status::unimplemented;

    /******************Kernel blocking Parameters ***********/
    jcp.ic_simd_block = simd_w;
    jcp.oc_simd_block = simd_w;

    jcp.dimK = jcp.ntiles;
    jcp.dimN = jcp.ic;
    jcp.dimM = jcp.oc;
    jcp.dimM_simd_block = jcp.oc_simd_block;
    jcp.dimN_reg_block = jcp.ic_simd_block;
    jcp.sched_policy = WSCHED_INVALID;
    status_t res = set_wsched_WEI_SDGtWo(jcp);
    if (res == status::unimplemented) {
        res = set_wsched_WEI_S_D_Giot_W(jcp);
        assert(res == status::success);
    }
    return res;
}
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
