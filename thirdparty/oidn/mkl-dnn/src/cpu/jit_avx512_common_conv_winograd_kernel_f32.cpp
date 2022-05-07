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
#include "cpu_memory.hpp"

#include <math.h>

#include "jit_avx512_common_conv_winograd_kernel_f32.hpp"

#ifndef KERNEL_SIZE_THRESHOLD
#define KERNEL_SIZE_THRESHOLD 16
#endif

#define MIN_REQUIRED_DIMN_REG_BLOCK 14

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
    if (jcp.ver == ver_4fma)
        return jcp.mb >= 32;
    else
        return jcp.mb >= 16;
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
        case L2:
        case L3:
        default: cache_latency = 250; break;
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
bool check_cond1(int dimN_reg_block, int dimK_block, int dimK_reg_block,
        int dimM_block, int dimM_simd_block, float C)
{
    float lhs = (dimM_block * dimN_reg_block * dimM_simd_block
                        + dimM_block * dimK_block * dimK_reg_block
                                * dimM_simd_block
                        + dimK_block * dimN_reg_block * dimK_reg_block)
            * (float)sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs < rhs);
}

bool check_cond1_bis(int dimN_reg_block, int dimK_block, int dimK_reg_block,
        int dimM_block, int dimM_simd_block, float C)
{
    float lhs = (dimM_block * dimK_block * dimK_reg_block * dimM_simd_block
                        + dimK_block * dimN_reg_block * dimK_reg_block)
            * (float)sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs < rhs);
}

bool check_cond2(int nb_dimN_reg_block, int dimN_reg_block, int dimK_nb_block,
        int dimK_block, int dimK_reg_block, int dimM_block, int dimM_simd_block,
        float C)
{
    float lhs = (nb_dimN_reg_block * dimM_block * dimN_reg_block * dimM_simd_block
                      + dimK_nb_block * dimM_block * dimK_block * dimK_reg_block
                              * dimM_simd_block
                      + nb_dimN_reg_block * dimK_nb_block * dimK_block
                              * dimN_reg_block * dimK_reg_block)
            * (float)sizeof(float);
    float rhs = C * L2_cache_size;
    return (lhs < rhs);
}
}

using namespace mkldnn::impl::format_tag;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

void _jit_avx512_common_conv_winograd_data_kernel_f32::gemm_loop_generate(
        bool is_beta_zero)
{
    // const int dimK_simd_block = jcp.dimK_reg_block;

    // for (int dimM_block =0; dimM_block < jcp.dimM_block; dimM_block++)
    //     for (int dimK_block = 0; dimK_block < jcp.dimK_block; dimK_block++)
    //         for (int dimK_reg_block= 0; dimK_reg_block < jcp.dimK_reg_block;
    //         dimK_reg_block++)
    //                 for (int tile =0; tile < jcp.dimN_reg_block; tile++)
    //                     C[dimM_block][tile] +=
    //                     A[dimM_block][dimK_block][dimK_reg_block] *
    //                     broadcast(B[dimK_block][tile][dimK_reg_block]);
    // 1) We do register blocking on A[dimM_block][dimK_block][dimK_reg_block],
    // so we load it before the loop on tile
    // 2) the loop on tile must be fully unrolled. Don't know about the one on
    // dimK_reg_block. I think it should be

    auto inner_loops = [=]() {
        Label dimM_block_loop, dimK_block_loop;
        const int inc_dimK_reg_block = jcp.ver == ver_4fma ? 4 : 1;
        const int fma_ipc = jcp.ver == ver_4fma ? 1 : 2;

        prefetcher_t<float> L1_pf(this, reg_srcB, L1,
                jcp.dimN_reg_block * jcp.dimK_reg_block,
                jcp.dimK_reg_block * jcp.dimN_reg_block / inc_dimK_reg_block,
                fma_ipc);
        prefetcher_t<float> L2_pf(this, reg_srcB, L2,
                jcp.dimN_reg_block * jcp.dimK_reg_block,
                jcp.dimK_reg_block * jcp.dimN_reg_block / inc_dimK_reg_block,
                fma_ipc);

        if (jcp.dimM_block > 1) {
            mov(reg_dimM_block_loop_cnt, jcp.dimM_block);
            L(dimM_block_loop);
        }
        {
            // First, we zero the accumulators if first nb_ic iteration,
            // otherwise we load them
            for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                Zmm zmm(jcp.zmm_start + tile);
                if (is_beta_zero)
                    vpxord(zmm, zmm, zmm);
                else
                    vmovups(zmm, zword[reg_dstC + 64 * tile]);
            }

            if (jcp.dimK_block > 1) {
                mov(reg_dimK_block_loop_cnt, jcp.dimK_block);
                L(dimK_block_loop);
            }
            {
                auto load_A = [=](int reg_idx, int offset) {
                    for (int i = 0; i < inc_dimK_reg_block; i++)
                        vmovups(Zmm(reg_idx + i),
                                zword[reg_srcA + 64 * (offset + i)]);
                };

                // Used when doing double buffering
                int next = 0;
                if (jcp.double_buffering) {
                    load_A(next, 0);
                }
                for (int dimK_reg_block = 0;
                        dimK_reg_block < jcp.dimK_reg_block;
                        dimK_reg_block += inc_dimK_reg_block) {
                    int current;
                    /* Loading the next vector from A */
                    current = next;
                    if (jcp.double_buffering) {
                        next = (dimK_reg_block + inc_dimK_reg_block)
                                % (2 * inc_dimK_reg_block);
                        load_A(next, dimK_reg_block + inc_dimK_reg_block);
                    } else {
                        next = 0;
                        load_A(next, dimK_reg_block);
                    }
                    /* Performing the fmas */
                    for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                        Zmm zmm(jcp.zmm_start + tile);
                        if (jcp.ver != ver_avx512_core)
                            L1_pf.prefetch(
                                    dimK_reg_block * jcp.dimN_reg_block + tile);
                        if (jcp.ver == ver_4fma)
                            v4fmaddps(zmm, Zmm(current),
                                    EVEX_compress_addr(reg_srcB,
                                              64 * tile + dimK_reg_block * 4));
                        else
                            vfmadd231ps(zmm, Zmm(current),
                                    EVEX_compress_addr(reg_srcB,
                                                64 * tile + dimK_reg_block * 4,
                                                true));
                        if (jcp.ver != ver_avx512_core)
                            L2_pf.prefetch(
                                    dimK_reg_block * jcp.dimN_reg_block + tile);
                    }
                }

                add(reg_srcA, jcp.dimK_reg_block * 64);
                add(reg_srcB, jcp.dimN_reg_block * 64);
                if (jcp.dimK_block > 1) {
                    sub(reg_dimK_block_loop_cnt, 1);
                    jnz(dimK_block_loop);
                }
            }


            auto store_output = [=](bool output_is_aligned) {
                for (int tile = 0; tile < jcp.dimN_reg_block; tile++) {
                    Zmm zmm(jcp.zmm_start + tile);
                    if (output_is_aligned
                        && jcp.dimK_nb_block == 1
                        && (jcp.dimN * jcp.dimM * alpha * alpha
                            * sizeof(float) > 2 * LLC_data_size))
                        vmovntps(zword[reg_dstC + 64 * tile], zmm);
                    else
                        vmovups(zword[reg_dstC + 64 * tile], zmm);
                }
            };

            Label unaligned_store, end_store;
            test(reg_dstC, cpu_isa_traits<avx512_common>::vlen - 1);
            jnz(unaligned_store, T_NEAR);
            store_output(true);
            jmp(end_store, T_NEAR);
            L(unaligned_store); {
                store_output(false);
            }
            L(end_store);

            if (jcp.dimM_block > 1) {
                sub(reg_srcB, jcp.dimK_block * jcp.dimN_reg_block * 64);
                add(reg_dstC, jcp.dimN_reg_block * 64);
                sub(reg_dimM_block_loop_cnt, 1);
                jnz(dimM_block_loop);
            }
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

status_t _jit_avx512_common_conv_winograd_data_kernel_f32::init_conf_common(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d)
{

    if (mayiuse(avx512_core))
        return status::unimplemented;
    else if (!mayiuse(avx512_common))
        return status::unimplemented;
    else if (mayiuse(avx512_mic_4ops))
        jcp.ver = ver_4fma;
    else
        jcp.ver = ver_fma;

    jcp.nthr = mkldnn_get_max_threads();

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

    if (!IMPLICATION(cd.alg_kind == alg_kind::convolution_auto,
                is_winograd_faster_than_direct(jcp)))
        return status::unimplemented;

    // Checking conditions not supported by these kernels
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
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);

    if (jcp.src_tag != dat_tag) return status::unimplemented;
    if (jcp.wei_tag != wei_tag) return status::unimplemented;
    if (jcp.dst_tag != dat_tag) return status::unimplemented;

    bool layout_consistency = true
        && jcp.ic <= src_d.padded_dims()[1]
        && jcp.oc <= dst_d.padded_dims()[1]
        && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
        && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!layout_consistency) return status::unimplemented;

    return status::success;
}


status_t set_wsched_DATA_W_S_G_D_avx512_common(jit_conv_winograd_conf_t &jcp) {

    auto test_cond_dimN_reg_block = [](jit_conv_winograd_conf_t &jcp,
            int dimN_reg_block, int current_best) {
        return (dimN_reg_block >= MIN_REQUIRED_DIMN_REG_BLOCK)
                && (dimN_reg_block < jcp.nb_reg)
                && (dimN_reg_block < current_best);
    };
    jcp.dimN_reg_block = get_divisor_satisfying_cond(
            jcp, jcp.dimN, jcp.dimN, test_cond_dimN_reg_block);

    if (jcp.dimN_reg_block >= jcp.nb_reg) {
        auto test_cond_dimN_reg_block = [](jit_conv_winograd_conf_t &jcp,
                int dimN_reg_block, int current_best) {
            return (dimN_reg_block < jcp.nb_reg)
                    && (dimN_reg_block > current_best);
        };

        jcp.dimN_reg_block = get_divisor_satisfying_cond(
                jcp, jcp.dimN, 1, test_cond_dimN_reg_block);
    }

    //********************* Choosing dimK_block **********************//
    auto test_cond1_dimK_block = [](
            jit_conv_winograd_conf_t &jcp, int dimK_block, int current_best) {
        return check_cond1(jcp.dimN_reg_block, dimK_block, jcp.dimK_reg_block,
                       1, jcp.dimM_simd_block, .75f)
                && (dimK_block > current_best);
    };

    auto test_cond1_bis_dimK_block = [](
            jit_conv_winograd_conf_t &jcp, int dimK_block, int current_best) {
        return check_cond1_bis(jcp.dimN_reg_block, dimK_block,
                       jcp.dimK_reg_block, 1, jcp.dimM_simd_block, .9f)
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
    jcp.dimM_simd_block = 16;
    /*XXX: Why C=0.5 here but C=0.75 for dimK_block?*/
    auto test_cond1_dimM_block = [](
            jit_conv_winograd_conf_t &jcp, int dimM_block, int current_best) {
        return check_cond1(jcp.dimN_reg_block, jcp.dimK_block,
                       jcp.dimK_reg_block, dimM_block, jcp.dimM_simd_block, .5f)
                && (dimM_block > current_best);
    };

    auto test_cond1_bis_dimM_block = [](
            jit_conv_winograd_conf_t &jcp, int dimM_block, int current_best) {
        return check_cond1_bis(jcp.dimN_reg_block, jcp.dimK_block,
                       jcp.dimK_reg_block, dimM_block, jcp.dimM_simd_block, .3f)
                && (dimM_block > current_best);
    };

    if (jcp.dimK_block < jcp.dimK / jcp.dimK_reg_block)
        jcp.dimM_block = get_divisor_satisfying_cond(
                jcp, jcp.dimM / jcp.dimM_simd_block, 1, test_cond1_dimM_block);
    else
        jcp.dimM_block = get_divisor_satisfying_cond(jcp,
                jcp.dimM / jcp.dimM_simd_block, 1, test_cond1_bis_dimM_block);
    jcp.dimM_nb_block = (jcp.dimM / jcp.dimM_simd_block) / jcp.dimM_block;

    //******************* Choosing dimN_block *******************//
    auto test_cond2_dimN_block = [](
            jit_conv_winograd_conf_t &jcp, int dimN_block, int current_best) {
        return check_cond2(dimN_block, jcp.dimN_reg_block, jcp.dimK_nb_block,
                       jcp.dimK_block, jcp.dimK_reg_block, jcp.dimM_block,
                       jcp.dimM_simd_block, .5f)
                && (dimN_block > current_best);
    };

    jcp.dimN_block = get_divisor_satisfying_cond(
            jcp, jcp.dimN / jcp.dimN_reg_block, 1, test_cond2_dimN_block);
    jcp.dimN_nb_block = jcp.dimN / (jcp.dimN_reg_block * jcp.dimN_block);
    jcp.sched_policy = WSCHED_DATA_W_S_G_D;
    return status::success;
}

status_t _jit_avx512_common_conv_winograd_data_kernel_f32::init_conf_kernel(
        jit_conv_winograd_conf_t &jcp, int dimM, int dimN, int dimK)
{
    jcp.dimK_reg_block = 16;
    jcp.dimM_simd_block = 16;

    // TODO: replace double buffering with nuple buffering to maximize register
    // usage.
    // the choice of the number of buffers will then come after choosing
    // dimN_reg_block
    jcp.double_buffering = true;
    if (jcp.double_buffering)
        jcp.zmm_start = 2 * ((jcp.ver == ver_4fma) ? 4 : 2);
    else
        jcp.zmm_start = 1;
    jcp.nb_reg = 32 - jcp.zmm_start;

    jcp.dimN = dimN;
    jcp.dimK = dimK;
    jcp.dimM = dimM;

    jcp.sched_policy = WSCHED_INVALID;
    set_wsched_DATA_W_S_G_D_avx512_common(jcp);

    assert(jcp.sched_policy == WSCHED_DATA_W_S_G_D);
    return status::success;
}

bool jit_avx512_common_conv_winograd_fwd_kernel_f32::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_relu = [&](int idx) { return p.entry_[idx].is_relu(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
    case 0: return true; // no post_ops
    case 1: return is_relu(0) || is_sum(0); // relu or sum
    case 2: return (is_sum(0) && is_relu(1)) ||
                       (is_relu(0) && is_sum(1)); // sum->relu or relu->sum
    case 3: return is_relu(0) && is_sum(1) && is_relu(2); // relu->sum->relu
    default: return false;
    }

    return false;
}

status_t jit_avx512_common_conv_winograd_fwd_kernel_f32::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const primitive_attr_t &attr) {
    status_t st = init_conf_common(jcp, cd, src_d, weights_d, dst_d);

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
    if (jcp.with_eltwise) jcp.eltwise = p.entry_[eltwise_ind].eltwise;
    jcp.with_sum = p.find(primitive_kind::sum, 0) != -1;

    status_t res = init_conf_kernel(jcp, jcp.oc, jcp.ntiles, jcp.ic);
    jcp.ic_simd_block = jcp.dimK_reg_block;
    jcp.ic_block = jcp.dimK_block;
    jcp.nb_ic = jcp.dimK_nb_block;
    jcp.oc_simd_block = jcp.dimM_simd_block;
    jcp.oc_block = jcp.dimM_block;
    jcp.nb_oc = jcp.dimM_nb_block;
    jcp.tile_block_ur = jcp.dimN_reg_block;
    jcp.nb_tile_block_ur = jcp.dimN_block;
    jcp.tile_block = jcp.dimN_nb_block;
    jcp.tile_4fma_padding = 0; // only relevant for backward weights

    return res;
}

status_t jit_avx512_common_conv_winograd_bwd_data_kernel_f32::init_conf(
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
    jcp.nb_ic = jcp.dimM_nb_block;
    jcp.tile_block_ur = jcp.dimN_reg_block;
    jcp.nb_tile_block_ur = jcp.dimN_block;
    jcp.tile_block = jcp.dimN_nb_block;
    jcp.tile_4fma_padding = 0; // only relevant for backward weights

    return res;
}

void jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::transpose_ker_generate()
{
    auto load_B = [=](int reg_idx, int offset) {
        for (int i = 0; i < 4; i++) {
            vmovups(Zmm(reg_idx + i), zword[reg_origB + (offset + i) * jcp.dimN_reg_block * sizeof(float)]);
        }
    };

    preamble();
    int curr = 0;
    for (int j = 0; j < alpha; j++) {
        for (int i = 0; i < alpha; i++) {
            int origB_offset = (j * alpha + i) * jcp.dimK_4fma;
            size_t transB_offset = (size_t)(j * alpha + i) * jcp.dimK_nb_block *
                jcp.dimN_block * jcp.dimK_block * jcp.dimK_reg_block *
                jcp.dimK_4fma * jcp.dimN_reg_block * sizeof(float);
            mov(reg_transB_idx, transB_offset);
            for (int tb = 0; tb < jcp.dimK_4fma; tb+=4) {
                /*double buffering to hide load latencies*/
                int next = (curr + 4) % 8;
                if (i == 0 && tb == 0) {
                    load_B(0, origB_offset);
                }
                if (tb + 4 < (jcp.dimK_4fma -1)) {
                    load_B(next, origB_offset + 4);
                } else if (i < alpha - 1) {
                    load_B(next, origB_offset + jcp.dimK_4fma);
                }

                vunpcklps(Zmm(8), Zmm(curr), Zmm(curr + 1));
                vunpcklps(Zmm(9), Zmm(curr + 2), Zmm(curr + 3));
                vunpckhps(Zmm(curr), Zmm(curr), Zmm(curr + 1));
                vunpckhps(Zmm(curr + 1), Zmm(curr + 2), Zmm(curr + 3));

                vunpcklpd(Zmm(curr + 2), Zmm(8), Zmm(9));
                vunpckhpd(Zmm(curr + 3), Zmm(8), Zmm(9));

                vunpcklpd(Zmm(8), Zmm(curr), Zmm(curr + 1));
                vunpckhpd(Zmm(9), Zmm(curr), Zmm(curr + 1));

                vmovntps(zword[reg_transB + reg_transB_idx
                        + sizeof(float) * tb * jcp.dimN_reg_block],
                        Zmm(curr+2));
                vmovntps(zword[reg_transB + reg_transB_idx
                        + sizeof(float) * (tb + 1) * jcp.dimN_reg_block],
                        Zmm(curr+3));
                vmovntps(zword[reg_transB + reg_transB_idx
                        + sizeof(float) * (tb + 2) * jcp.dimN_reg_block],
                        Zmm(8));
                vmovntps(zword[reg_transB + reg_transB_idx
                        + sizeof(float) * (tb + 3) * jcp.dimN_reg_block],
                        Zmm(9));
                curr = next;

            }
        }
    }
    postamble();
    ret();
}
void jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::gemm_loop_generate(
        bool is_first_tile)
{
    // for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++)
    //     for (int ifm2 = 0; ifm2 < jcp.ic_block; ifm2++)
    //             for (int nb_tile_block_ur = 0; nb_tile_block_ur <
    //             jcp.nb_tile_block_ur; nb_tile_block_ur++)
    //                 for (int tile_block_ur = 0; tile_block_ur <
    //                 jcp.tile_block_ur; tile_block_ur++)
    //                     for (int ifm3 = 0; ifm3 < jcp.ic_reg_block; ++ifm3)
    //                         U[ofm2][ifm2][ofm3][ifm3][0:oc_simd_block] +=
    //                             M[ofm2][ofm3][nb_tile_block_ur][tile_block_ur][0:oc_simd_block]
    //                              *
    //                              broadcast(V[ifm2][nb_tile_block_ur][ifm3][tile_block_ur])
    auto inner_loops = [=]() {
        int inc_fma = jcp.ver == ver_4fma ? 4 : 1;
        const int fma_ipc = jcp.ver == ver_4fma ? 1 : 2;
        prefetcher_t<float> L1_pf(this, reg_srcB, L1,
                jcp.dimK_reg_block * jcp.dimN_reg_block * jcp.dimK_4fma,
                jcp.dimK_reg_block * jcp.dimN_reg_block * jcp.dimK_4fma
                        / inc_fma,
                fma_ipc);
        prefetcher_t<float> L2_pf(this, reg_srcB, L2,
                jcp.dimK_reg_block * jcp.dimN_reg_block * jcp.dimK_4fma,
                jcp.dimK_reg_block * jcp.dimN_reg_block * jcp.dimK_4fma
                        / inc_fma,
                fma_ipc);

        auto load_A = [=](int reg_idx, int offset) {
            for (int i = 0; i < inc_fma; i++) {
                vmovups(Zmm(reg_idx + i),
                        zword[reg_srcA +
                        sizeof(float) * jcp.dimM_simd_block * (offset + i)]);
            }
        };

        Label dimM_block_loop, dimK_block_loop, dimN_block_loop;
        if (jcp.dimM_block > 1) {
            mov(reg_dimM_block_loop_cnt, jcp.dimM_block);
            L(dimM_block_loop);
        }
        { /************* OC_block (M) loop ***********/
            if (jcp.dimN_block > 1) {
                mov(reg_dimN_block_loop_cnt, jcp.dimN_block);
                L(dimN_block_loop);
            }
            { /*************** IC_block (N) loop *********/
                for (int dimN_reg_block = 0;
                        dimN_reg_block < jcp.dimN_reg_block; ++dimN_reg_block) {
                    Zmm zmm(jcp.zmm_start + dimN_reg_block);
                    if (is_first_tile)
                        vpxord(zmm, zmm, zmm);
                    else
                        vmovups(zmm, zword[reg_dstC +
                                dimN_reg_block * jcp.dimM_simd_block *
                                sizeof(float)]);
                }

                if (jcp.dimK_block > 1) {
                    mov(reg_dimK_block_loop_cnt, jcp.dimK_block);
                    L(dimK_block_loop);
                }
                { /************* nb_tile_ur(K) loop ********/
                    int next = 0;
                    if (jcp.double_buffering) {
                        load_A(next, 0);
                    }
                    for (int dimK_reg_block = 0;
                            dimK_reg_block < jcp.dimK_reg_block;
                            dimK_reg_block++) {
                        int srcB_offset = dimK_reg_block * jcp.dimK_4fma
                                * jcp.dimN_reg_block;
                        for (int dimK_4fma = 0; dimK_4fma < jcp.dimK_4fma;
                                dimK_4fma += inc_fma) {
                            int current = next;
                            if (jcp.double_buffering) {
                                next = (dimK_reg_block * jcp.dimK_4fma
                                               + dimK_4fma + inc_fma)
                                        % (2 * inc_fma);
                                load_A(next, dimK_reg_block * jcp.dimK_4fma
                                                + dimK_4fma + inc_fma);
                            } else {
                                next = 0;
                                load_A(next, dimK_reg_block * jcp.dimK_4fma
                                                + dimK_4fma);
                            }
                            for (int dimN_reg_block = 0;
                                    dimN_reg_block < jcp.dimN_reg_block;
                                    ++dimN_reg_block) {
                                L1_pf.prefetch(srcB_offset / inc_fma
                                        + dimK_4fma / inc_fma
                                                * jcp.dimN_reg_block
                                        + dimN_reg_block);
                                L2_pf.prefetch(srcB_offset / inc_fma
                                        + dimK_4fma / inc_fma
                                                * jcp.dimN_reg_block
                                        + dimN_reg_block);
                                if (jcp.ver == ver_4fma) {
                                    int srcB_trans_offset = (dimK_4fma / 4) * 64
                                            + dimK_4fma % 4;
                                    v4fmaddps(
                                            Zmm(jcp.zmm_start + dimN_reg_block),
                                            Zmm(current),
                                            EVEX_compress_addr(reg_srcB,
                                                    sizeof(float) * (
                                                        srcB_offset +
                                                        srcB_trans_offset +
                                                        (dimN_reg_block % 4) * 16 +
                                                        (dimN_reg_block / 4) * 4)));
                                } else {
                                    vfmadd231ps(
                                            Zmm(jcp.zmm_start + dimN_reg_block),
                                            Zmm(current),
                                            EVEX_compress_addr(reg_srcB,
                                                sizeof(float) * (srcB_offset + dimN_reg_block),
                                                    true));
                                }
                            }
                        }
                    }
                }

                add(reg_srcA, jcp.dimK_reg_block * jcp.dimK_4fma
                                * jcp.dimM_simd_block * sizeof(float));
                add(reg_srcB, jcp.dimK_reg_block * jcp.dimN_reg_block
                                * jcp.dimK_4fma * sizeof(float));
                if (jcp.dimK_block > 1) {
                    sub(reg_dimK_block_loop_cnt, 1);
                    jnz(dimK_block_loop);
                }

                /******** Write C back to memory *******/
                for (int dimN_reg_block = 0;
                        dimN_reg_block < jcp.dimN_reg_block; ++dimN_reg_block) {
                    Zmm zmm(jcp.zmm_start + dimN_reg_block);
                    vmovups(zword[reg_dstC +
                            dimN_reg_block * jcp.dimM_simd_block * sizeof(float)],
                            zmm);
                }

                sub(reg_srcA, jcp.dimK_block * jcp.dimK_reg_block *
                        jcp.dimK_4fma * jcp.dimM_simd_block * sizeof(float));
                add(reg_dstC, jcp.dimN_reg_block * jcp.dimM_simd_block
                        * sizeof(float));
                if (jcp.dimN_block > 1) {
                    sub(reg_dimN_block_loop_cnt, 1);
                    jnz(dimN_block_loop);
                }
            }

            if (jcp.dimM_block > 1) {
                sub(reg_srcB, jcp.dimN_block * jcp.dimK_block
                                * jcp.dimK_reg_block * jcp.dimN_reg_block
                                * jcp.dimK_4fma * sizeof(float));
                add(reg_srcA, jcp.dimK_block * jcp.dimK_reg_block
                                * jcp.dimK_4fma * jcp.dimM_simd_block * sizeof(float));
                sub(reg_dimM_block_loop_cnt, 1);
                jnz(dimM_block_loop);
            }
        }
    };

    /* Preamble */
    // register used to handle long fma encoding
    preamble();
    mov(reg_srcA, reg_srcA_const);
    inner_loops();

    /* Postamble */
    postamble();
    ret();
}

namespace {
bool check_cond1_wu(int dimM_block, int dimM_simdw, int dimK_block,
        int dimK_reg_block, int dimK_4fma, int dimN_reg_block, float C)
{
    float lhs = 1.0f * dimM_block * dimN_reg_block * dimM_simdw;
    lhs += dimM_block * dimK_block * dimK_reg_block * dimK_4fma * dimM_simdw;
    lhs += dimK_block * dimN_reg_block * dimK_reg_block * dimK_4fma;
    lhs *= sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs <= rhs);
}

bool check_cond1bis_wu(int dimM_block, int dimM_simdw, int dimK_block,
        int dimK_reg_block, int dimK_4fma, int dimN_reg_block, float C)
{
    float lhs = 1.0f * dimM_block * dimK_block * dimK_reg_block * dimK_4fma
            * dimM_simdw;
    lhs += dimK_block * dimN_reg_block * dimK_reg_block * dimK_4fma;
    lhs *= sizeof(float);
    float rhs = C * L1_cache_size;
    return (lhs <= rhs);
}

bool check_cond2bis_wu(int dimM_block, int dimM_simdw, int dimK_block,
        int dimK_reg_block, int dimK_4fma, int dimN_block, int dimN_reg_block,
        float C)
{
    float lhs = 1.0f * dimM_block * dimM_simdw * dimK_block * dimK_reg_block
            * dimK_4fma;
    lhs += dimK_block * dimK_reg_block * dimK_4fma * dimN_block
            * dimN_reg_block;
    lhs *= sizeof(float);
    float rhs = C * L2_cache_size;
    return (lhs <= rhs);
}

bool check_cond2_wu(int dimM_block, int dimM_simdw, int dimK_block,
        int dimK_reg_block, int dimK_4fma, int dimN_block, int dimN_reg_block,
        float C)
{
    float lhs = 1.0f * dimM_block * dimM_simdw * dimN_block * dimN_reg_block;
    lhs += dimM_block * dimM_simdw * dimK_block * dimK_reg_block * dimK_4fma;
    lhs += dimK_block * dimK_reg_block * dimK_4fma * dimN_block
            * dimN_reg_block;
    lhs *= sizeof(float);
    float rhs = C * L2_cache_size;
    return (lhs <= rhs);
}
} // namespace

status_t set_wsched_WEI_S_D_G_W_avx512_common(jit_conv_winograd_conf_t &jcp)
{
    /*************** Choose dimN_reg_block (ic_simd_block)
     * *******************************/
    jcp.dimN = jcp.ic;
    /*Hardcoded to 16 because N = ic for bwd weights and
     innermost dimension for ic is assumed 16 in src transforms. This
     choice covers load latencies while maintaining simplicity of kernel
     for POR topologies. FIXME in future??: Will not work for future topologies
     when ic%16 != 0*/
    jcp.dimN_reg_block = jcp.ic_simd_block;

    /****************************** Choose dimK_block
     * **************************/
    // No freedom for choosing dimM_simd_block because ic_simd_block
    // is determined by input data format
    jcp.dimM_simd_block = jcp.oc_simd_block;

    auto test_cond1bis_dimK_block = [](
            jit_conv_winograd_conf_t &jcp, int dimK_block, int current_best) {
        return check_cond1bis_wu(1, jcp.dimM_simd_block, dimK_block, 1,
                       jcp.dimK_4fma, jcp.dimN_reg_block, 0.4f)
                && (dimK_block > current_best);
    };

    auto test_cond1_dimK_block = [](
            jit_conv_winograd_conf_t &jcp, int dimK_block, int current_best) {
        return check_cond1_wu(1, jcp.dimM_simd_block, dimK_block, 1,
                       jcp.dimK_4fma, jcp.dimN_reg_block, 0.4f)
                && (dimK_block > current_best);
    };

    auto test_cond2bis_dimK_block = [](
            jit_conv_winograd_conf_t &jcp, int dimK_block, int current_best) {
        return check_cond2bis_wu(1, jcp.dimM_simd_block, dimK_block, 1,
                       jcp.dimK_4fma, 1, jcp.dimN_reg_block, 0.5f)
                && (dimK_block > current_best);
    };

    auto test_cond2_dimK_block = [](
            jit_conv_winograd_conf_t &jcp, int dimK_block, int current_best) {
        return check_cond2_wu(1, jcp.dimM_simd_block, dimK_block, 1,
                       jcp.dimK_4fma, 1, jcp.dimN_reg_block, 0.1f)
                && (dimK_block > current_best);
    };

    jcp.dimK_block = get_divisor_satisfying_cond(
            jcp, jcp.dimK / jcp.dimK_4fma, 1, test_cond2bis_dimK_block);
    if (jcp.dimK_block < jcp.dimK / jcp.dimK_4fma)
        jcp.dimK_block = get_divisor_satisfying_cond(
                jcp, jcp.dimK / jcp.dimK_4fma, 1, test_cond2_dimK_block);

    jcp.dimK_reg_block = get_divisor_satisfying_cond(
            jcp, jcp.dimK_block, 1, test_cond1bis_dimK_block);
    if (jcp.dimK_reg_block < jcp.dimK_block) {
        jcp.dimK_reg_block = get_divisor_satisfying_cond(
                jcp, jcp.dimK_block, 1, test_cond1_dimK_block);
    }
    jcp.dimK_block /= jcp.dimK_reg_block;
    jcp.dimK_nb_block
            = jcp.dimK / jcp.dimK_4fma / jcp.dimK_reg_block / jcp.dimK_block;
    jcp.tile_block_ur = jcp.dimK_reg_block;
    jcp.nb_tile_block_ur = jcp.dimK_block;
    jcp.tile_block = jcp.dimK_nb_block;

    /***************************** Chose dimN block
     * ****************************/
    auto test_cond2_dimN_block = [](
            jit_conv_winograd_conf_t &jcp, int dimN_block, int current_best) {
        return check_cond2_wu(1, jcp.dimM_simd_block, jcp.dimK_block,
                       jcp.dimK_reg_block, jcp.dimK_4fma, dimN_block,
                       jcp.dimN_reg_block, 0.5f)
                && (dimN_block > current_best);
    };

    jcp.dimN_block = get_divisor_satisfying_cond(
            jcp, jcp.dimN / jcp.dimN_reg_block, 1, test_cond2_dimN_block);
    jcp.ic_block = jcp.dimN_block;
    jcp.dimN_nb_block = jcp.dimN / jcp.dimN_reg_block / jcp.dimN_block;
    jcp.nb_ic = jcp.dimN_nb_block;

    /********************************* Choose dimM block
     * ************************/
    jcp.dimM = jcp.oc;

    auto test_cond1_dimM_block = [](
            jit_conv_winograd_conf_t &jcp, int dimM_block, int current_best) {
        return check_cond1_wu(dimM_block, jcp.dimM_simd_block, 1,
                       jcp.dimK_reg_block, jcp.dimK_4fma, jcp.dimN_reg_block,
                       1.0f)
                && (dimM_block > current_best)
                && (jcp.dimM / jcp.dimM_simd_block / dimM_block) >= 2;
    };

    jcp.dimM_block = get_divisor_satisfying_cond(
            jcp, jcp.dimM / jcp.dimM_simd_block, 1, test_cond1_dimM_block);
    jcp.dimM_nb_block = (jcp.dimM / jcp.dimM_simd_block) / jcp.dimM_block;

    jcp.sched_policy = WSCHED_WEI_S_D_G_W;
    return status::success;
}

status_t jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &diff_dst_d,
        const memory_desc_wrapper &diff_weights_d)
{
    jcp.nthr = mkldnn_get_max_threads();

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
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

    if (mayiuse(avx512_core))
        return status::unimplemented;
    if (!mayiuse(avx512_common))
        return status::unimplemented;
    else if (mayiuse(avx512_mic_4ops))
        jcp.ver = ver_4fma;
    else
        jcp.ver = ver_fma;

    if (!IMPLICATION(cd.alg_kind == alg_kind::convolution_auto,
                is_winograd_faster_than_direct(jcp)))
        return status::unimplemented;
    // Winograd specific initialization
    jcp.itiles = (jcp.ow + tile_size - 1) / tile_size;
    jcp.jtiles = (jcp.oh + tile_size - 1) / tile_size;
    jcp.ntiles = jcp.mb * jcp.itiles * jcp.jtiles;

    // Winograd kernel works only for 3x3 convolution with stride 1
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

    /*************************** New Kernel Parameters
     * *****************************/
    jcp.ic_simd_block = simd_w;
    jcp.oc_simd_block = simd_w;
    jcp.dimK_4fma = 1;
    jcp.tile_4fma_padding = 0;

#define MAX_4FMA_UR 8
    if (jcp.ver == ver_4fma) {
        auto test_cond_4fma = [](jit_conv_winograd_conf_t &jcp, int dimK_4fma,
                                      int current_best) {
            return (dimK_4fma % 4 == 0) && (dimK_4fma <= MAX_4FMA_UR)
                    && (dimK_4fma > current_best);
        };
        jcp.dimK_4fma = get_divisor_satisfying_cond(
                jcp, jcp.itiles * jcp.jtiles, 4, test_cond_4fma);
        if (jcp.dimK_4fma == 1)
            jcp.dimK_4fma = 4;
        if ((jcp.itiles * jcp.jtiles) % jcp.dimK_4fma != 0)
            jcp.tile_4fma_padding = jcp.dimK_4fma
                    - ((jcp.itiles * jcp.jtiles) % jcp.dimK_4fma);
    }

    jcp.tile_4fma = jcp.dimK_4fma;
    /*NOTE: When (itiles * jtiles) % dimK_4fma != 0, transpose in diff_src
     * transform
     * will not work correctly, this is solved by applying padding.*/
    jcp.dimK = jcp.mb * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding);
    jcp.dimN = jcp.ic;
    jcp.dimM = jcp.oc;

    jcp.double_buffering = true;
    if (jcp.double_buffering)
        jcp.zmm_start = jcp.ver == ver_4fma ? 8 : 2;
    else
        jcp.zmm_start = jcp.ver == ver_4fma ? 4 : 1;
    jcp.nb_reg = 32 - jcp.zmm_start;

    jcp.sched_policy = WSCHED_INVALID;
    status_t res = set_wsched_WEI_S_D_G_W_avx512_common(jcp);
    assert(jcp.sched_policy == WSCHED_WEI_S_D_G_W);

    jcp.tile_block_ur = jcp.dimK_reg_block;
    jcp.nb_tile_block_ur = jcp.dimK_block;
    jcp.tile_block = jcp.dimK_nb_block;

    jcp.ic_block = jcp.dimN_block;
    jcp.nb_ic = jcp.dimN_nb_block;

    jcp.oc_block = jcp.dimM_block;
    jcp.nb_oc = jcp.dimM_nb_block;

    return res;

}
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
