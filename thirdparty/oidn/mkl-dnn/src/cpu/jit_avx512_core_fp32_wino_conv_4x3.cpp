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

#ifdef __INTEL_COMPILER
#include <immintrin.h>
#endif

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_fp32_wino_conv_4x3.hpp"

#ifndef _MSC_VER
#define pragma_unroll _Pragma("unroll")
#else
#define pragma_unroll
#endif


namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

template <bool is_fwd>
void _jit_avx512_core_fp32_wino_conv_4x3_t<is_fwd>
::weight_transform_data(const jit_conv_winograd_conf_t &jcp,
        float *wp, float *twp) const
{
    float G[] = {0.26890756302521f, 0.688403361344538f, 0.119514472455649f,
                 1.13777777777778f, 0.430252100840336f, 0.179271708683473f};
    const int kh = 3;
    const int kw = 3;
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[kh][kw][simd_w][simd_w];
    float T[alpha][3][simd_w];
    auto p = jit_wino_transform_call_s();

    p.src = wp;
    p.dst = twp;
    p.G = G;
    p.M = F;
    p.Mw = Fw;
    p.T = T;

    kernel_->weights_transform_data_ker(&p);
}

template<bool is_fwd>
void _jit_avx512_core_fp32_wino_conv_4x3_t<is_fwd>::output_transform_data
(int image, const jit_conv_winograd_conf_t &jcp,
    const post_ops_t &p_ops, float *toutp, float *pout_b, float *bias) const {

    float G[] = {0.625f, 1.5f, 0.390625f, 2.25f, 0.244140625f, 3.375f};
    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];
    float T[tile_size][alpha][simd_w];

    auto p = jit_wino_transform_call_s();
    p.src = toutp;
    p.dst = pout_b;
    p.G = G;
    p.M = O;
    p.Mw = Ow;
    p.T = T;
    p.bias = bias;

    int tile_base_index = image * jcp.itiles * jcp.jtiles;
    int tile_block_ur = tile_base_index % jcp.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / jcp.tile_block_ur) % jcp.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / jcp.tile_block_ur) / jcp.nb_tile_block_ur;

    for (int tj = 0; tj < jcp.jtiles; tj++) {
        for (int ti = 0; ti < jcp.itiles; ti++) {

            p.tile_block_ur = tile_block_ur;
            p.nb_tile_block_ur = nb_tile_block_ur;
            p.tile_block = tile_block;
            p.tj = tj;
            p.ti = ti;

            kernel_->output_transform_data_ker(&p);

            tile_block_ur++;
            if (tile_block_ur >= jcp.tile_block_ur) {
                tile_block_ur = 0;
                nb_tile_block_ur++;
            }
            if (nb_tile_block_ur >= jcp.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }
}

template<bool is_fwd>
void _jit_avx512_core_fp32_wino_conv_4x3_t<is_fwd>
::output_transform_tileblock_data(int tile_block,
    const jit_conv_winograd_conf_t &jcp, const post_ops_t &p_ops,
    float *toutp, float *outp, float *bias) const {

    float G[] = {0.625f, 1.5f, 0.390625f, 2.25f, 0.244140625f, 3.375f};
    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];
    float T[tile_size][alpha][simd_w];

    auto p = jit_wino_transform_call_s();
    p.src = toutp;
    p.dst = outp;
    p.G = G;
    p.M = O;
    p.Mw = Ow;
    p.T = T;
    p.bias = bias;

    int outw = is_fwd ? jcp.ow : jcp.iw;
    int outh = is_fwd ? jcp.oh : jcp.ih;

    int tile_index = tile_block * jcp.nb_tile_block_ur * jcp.tile_block_ur;

    for (int nb_tile_block_ur = 0;
        nb_tile_block_ur < jcp.nb_tile_block_ur;
        nb_tile_block_ur++) {

        for (int tile_block_ur = 0; tile_block_ur < jcp.tile_block_ur;
            tile_block_ur++) {
            int img = tile_index / (jcp.jtiles * jcp.itiles);
            int ti = tile_index % jcp.itiles;
            int tj = (tile_index / jcp.itiles) % jcp.jtiles;

            p.tile_block_ur = tile_block_ur;
            p.nb_tile_block_ur = nb_tile_block_ur;
            p.tile_block = tile_block;
            p.tj = tj;
            p.ti = ti;
            p.dst = outp + img * (jcp.dimM / jcp.dimM_simd_block)
                               * outh * outw * jcp.dimM_simd_block;

            kernel_->output_transform_data_ker(&p);

            tile_index++;
        }
    }
}


template<bool is_fwd>
void _jit_avx512_core_fp32_wino_conv_4x3_t<is_fwd>
    ::input_transform_data(int image, const jit_conv_winograd_conf_t &jcp,
        float *inp, float *tinp) const
{
    float G[] = {-2.25f, -0.390625f, 0.87890625f, -2.640625f,
                 0.625f, -0.625f, 1.5f, -1.5f, -2.640625f};

    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];
    float T[alpha][alpha][simd_w];

    auto p = jit_wino_transform_call_s();

    p.src = inp;
    p.dst = tinp;
    p.G = G;
    p.M = I;
    p.Mw = Iw;
    p.T = T;

    int tile_base_index = image * jcp.itiles * jcp.jtiles;
    int tile_block_ur = tile_base_index % jcp.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / jcp.tile_block_ur) % jcp.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / jcp.tile_block_ur) / jcp.nb_tile_block_ur;

    for (int tj = 0; tj < jcp.jtiles; tj++) {
        for (int ti = 0; ti < jcp.itiles; ti++) {

            p.tile_block_ur = tile_block_ur;
            p.nb_tile_block_ur = nb_tile_block_ur;
            p.tile_block = tile_block;
            p.tj = tj;
            p.ti = ti;

            kernel_->input_transform_data_ker(&p);

            tile_block_ur++;
            if (tile_block_ur >= jcp.tile_block_ur) {
                tile_block_ur = 0;
                nb_tile_block_ur++;
            }
            if (nb_tile_block_ur >= jcp.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }
}

template <bool is_fwd>
void _jit_avx512_core_fp32_wino_conv_4x3_t<is_fwd>
    ::input_transform_tileblock_data(int tile_block,
        const jit_conv_winograd_conf_t &jcp,
        float *inp, float *tinp) const
{
    float G[] = {-2.25f, -0.390625f, 0.87890625f, -2.640625f,
               0.625f, -0.625f, 1.5f, -1.5f, -2.640625f};
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];
    float T[alpha][alpha][simd_w];

    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int inpw = is_fwd ? jcp.iw : jcp.ow;

    array_offset_calculator<float, 5> input(inp,
        jcp.mb, jcp.dimK / simd_w, inph, inpw, simd_w);
    array_offset_calculator<float, 7> output(tinp,
        alpha, alpha,
        jcp.dimN_block, jcp.dimK_nb_block, jcp.dimK_block,
        jcp.dimN_reg_block, jcp.dimK_reg_block);

    auto p = jit_wino_transform_call_s();

    p.dst = tinp;
    p.G = G;
    p.M = I;
    p.Mw = Iw;
    p.T = T;


    int tile_index = tile_block * jcp.nb_tile_block_ur * jcp.tile_block_ur;

    for (int nb_tile_block_ur = 0;
            nb_tile_block_ur < jcp.nb_tile_block_ur;
            nb_tile_block_ur++) {

        for (int tile_block_ur = 0; tile_block_ur < jcp.tile_block_ur;
                tile_block_ur++) {

            int img = tile_index / (jcp.jtiles * jcp.itiles);
            int ti = tile_index % jcp.itiles;
            int tj = (tile_index / jcp.itiles) % jcp.jtiles;
            float *pinp_b = &(input(img, 0, 0, 0, 0));

            p.src = pinp_b;
            p.tile_block_ur = tile_block_ur;
            p.nb_tile_block_ur = nb_tile_block_ur;
            p.tj = tj;
            p.ti = ti;

            kernel_->input_transform_data_ker(&p);

            tile_index++;
        }
    }
}

template <bool is_fwd>
void _jit_avx512_core_fp32_wino_conv_4x3_t<is_fwd>::_execute_data_W_S_G_D(
        float *inp_ptr, float *out_ptr, float *wei_ptr, float *bias_ptr,
        const memory_tracking::grantor_t &scratchpad) const {
    const auto &jcp = kernel_->jcp;
    const auto &p_ops = attr_->post_ops_;

    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int inpw = is_fwd ? jcp.iw : jcp.ow;
    const int outh = is_fwd ? jcp.oh : jcp.ih;
    const int outw = is_fwd ? jcp.ow : jcp.iw;

    /* Notation:
       FWD: dimM:oc, dimN:ntiles, dimK:ic,
       BWD: dimM:ic, dimN:ntiles, dimK:oc,
       FWD/BWD: V: src/diff_dst transform, U:weight transform,
                M:dst/diff_src transform  */
    array_offset_calculator<float, 5> input(inp_ptr,
            jcp.mb, jcp.dimK/jcp.dimK_reg_block, inph, inpw,
            jcp.dimK_reg_block);
    array_offset_calculator<float, 5> output(out_ptr,
            jcp.mb, jcp.dimM/jcp.dimM_simd_block, outh, outw,
            jcp.dimM_simd_block);
    array_offset_calculator<float, 6> weights(wei_ptr,
            jcp.oc/jcp.oc_simd_block, jcp.ic/jcp.ic_simd_block, jcp.kh, jcp.kw,
            jcp.ic_simd_block, jcp.oc_simd_block);
    array_offset_calculator<float, 2> bias(bias_ptr,
            jcp.dimM/jcp.dimM_simd_block, jcp.dimM_simd_block);

    array_offset_calculator<float, 8> M(is_fwd
            ? scratchpad.template get<float>(key_wino_M)
            : scratchpad.template get<float>(key_wino_V),
            jcp.dimN_nb_block, jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimN_block, jcp.dimM_block * jcp.dimM_reg_block,
            jcp.dimN_reg_block, jcp.dimM_simd_block);

    auto wino_wei = (jcp.prop_kind == prop_kind::forward_inference)
            ? wei_ptr
            : scratchpad.template get<float>(key_wino_U);

    array_offset_calculator<float, 8> U(wino_wei,
            jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimK_nb_block,
            jcp.dimM_block * jcp.dimM_reg_block, jcp.dimK_block,
            jcp.dimK_reg_block, jcp.dimM_simd_block);
    array_offset_calculator<float, 8> V(is_fwd
            ? scratchpad.template get<float>(key_wino_V)
            : scratchpad.template get<float>(key_wino_M),
            jcp.dimN_nb_block, alpha, alpha,
            jcp.dimN_block, jcp.dimK_nb_block,
            jcp.dimK_block, jcp.dimN_reg_block, jcp.dimK_reg_block);

    const bool wants_padded_bias = jcp.with_bias
        && jcp.oc_without_padding != jcp.oc;
    float last_slice_bias[simd_w] = {0};
    if (wants_padded_bias) {
        for (int oc = 0; oc < jcp.oc_without_padding % jcp.oc_simd_block; ++oc)
            last_slice_bias[oc] = bias(jcp.dimM / jcp.dimM_simd_block - 1, oc);
    }

    {

        parallel_nd(jcp.mb, jcp.dimK_nb_block, jcp.dimK_block,
                [&](int img, int K_blk1, int K_blk2) {
                input_transform_data(img, jcp,
                    &(input(img, K_blk1 * jcp.dimK_block + K_blk2,
                            0, 0, 0)),
                        &(V(0, 0, 0, 0, K_blk1, K_blk2, 0, 0)));
                });

        if (jcp.prop_kind != prop_kind::forward_inference) {
            parallel_nd(jcp.nb_oc, jcp.nb_ic, (jcp.oc_block * jcp.oc_reg_block),
                (jcp.ic_block * jcp.ic_reg_block),
                [&](int ofm1, int ifm1, int ofm2, int ifm2) {
                    float *U_base_ptr = is_fwd
                        ? &(U(ofm1, 0, 0, ifm1, ofm2, ifm2, 0, 0))
                        : &(U(ifm1, 0, 0, ofm1, ifm2, ofm2, 0, 0));
                    weight_transform_data(jcp,
                        &(weights(
                                ofm1 * jcp.oc_block * jcp.oc_reg_block + ofm2,
                                ifm1 * jcp.ic_block * jcp.ic_reg_block + ifm2,
                                0, 0, 0, 0)),
                        U_base_ptr);
            });
        }

        parallel_nd(jcp.dimN_nb_block, alpha, alpha, jcp.dimM_nb_block,
            [&](int N_blk1, int oj, int oi, int M_blk1) {
            for (int K_blk1 = 0; K_blk1 < jcp.dimK_nb_block;
                 K_blk1++)
            for (int N_blk2 = 0; N_blk2 < jcp.dimN_block; N_blk2++)
                kernel_->gemm_loop_ker(
                        (float *)&(M(N_blk1, M_blk1, oj, oi,
                            N_blk2, 0, 0, 0)),
                        (const float *)&(U(M_blk1, oj, oi,
                            K_blk1, 0, 0, 0, 0)),
                        (const float *)&(V(N_blk1, oj, oi,
                            N_blk2, K_blk1, 0, 0, 0)), K_blk1);
        });

        parallel_nd(jcp.mb, jcp.dimM_nb_block, (jcp.dimM_block * jcp.dimM_reg_block),
                    [&](int img, int M_blk1, int M_blk2) {
            const int M_blk =
                M_blk1 * jcp.dimM_block  * jcp.dimM_reg_block + M_blk2;

            float *bias_ptr = wants_padded_bias
                && M_blk == jcp.dimM / jcp.dimM_simd_block - 1
                ? last_slice_bias : &bias(M_blk, 0);
            output_transform_data(img, jcp, p_ops,
                    &(M(0, M_blk1, 0, 0, 0, M_blk2, 0, 0)),
                    &(output(img, M_blk, 0, 0, 0)), bias_ptr);
        });

    }
}

template <bool is_fwd>
void _jit_avx512_core_fp32_wino_conv_4x3_t<is_fwd>::_execute_data_W_SGD(
        float *inp_ptr, float *out_ptr, float *wei_ptr, float *bias_ptr,
        const memory_tracking::grantor_t &scratchpad) const {
    const auto &jcp = kernel_->jcp;
    const auto &p_ops = attr_->post_ops_;

    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int inpw = is_fwd ? jcp.iw : jcp.ow;
    const int outh = is_fwd ? jcp.oh : jcp.ih;
    const int outw = is_fwd ? jcp.ow : jcp.iw;

    array_offset_calculator<float, 5> input(inp_ptr,
        jcp.mb, jcp.dimK/jcp.dimK_reg_block, inph, inpw, jcp.dimK_reg_block);
    array_offset_calculator<float, 5> output(out_ptr,
        jcp.mb, jcp.dimM/jcp.dimM_simd_block, outh, outw, jcp.dimM_simd_block);
    array_offset_calculator<float, 6> weights(wei_ptr,
        jcp.oc/jcp.oc_simd_block, jcp.ic/jcp.ic_simd_block, jcp.kh, jcp.kw,
        jcp.ic_simd_block, jcp.oc_simd_block);
    array_offset_calculator<float, 2> bias(bias_ptr,
        jcp.oc/jcp.oc_simd_block, jcp.oc_simd_block);

    auto wino_wei = (jcp.prop_kind == prop_kind::forward_inference)
                ? wei_ptr
                : scratchpad.template get<float>(key_wino_U);

    array_offset_calculator<float, 8> U(wino_wei,
            jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimK_nb_block,
            jcp.dimM_block  * jcp.dimM_reg_block, jcp.dimK_block,
            jcp.dimK_reg_block, jcp.dimM_simd_block);

    array_offset_calculator<float, 8> M(is_fwd
            ? scratchpad.template get<float>(key_wino_M)
            : scratchpad.template get<float>(key_wino_V),
            0, jcp.dimM_nb_block, alpha, alpha,
            jcp.dimN_block, jcp.dimM_block * jcp.dimM_reg_block,
            jcp.dimN_reg_block, jcp.dimM_simd_block);
    array_offset_calculator<float, 8> V(is_fwd
            ? scratchpad.template get<float>(key_wino_V)
            : scratchpad.template get<float>(key_wino_M),
            0, alpha, alpha, jcp.dimN_block,
            jcp.dimK_nb_block, jcp.dimK_block,
            jcp.dimN_reg_block, jcp.dimK_reg_block);

    const bool wants_padded_bias = jcp.with_bias
        && jcp.oc_without_padding != jcp.oc;
    float last_slice_bias[simd_w] = {0};
    if (wants_padded_bias) {
        for (int oc = 0; oc < jcp.oc_without_padding % jcp.oc_simd_block; ++oc)
            last_slice_bias[oc] = bias(jcp.dimM / jcp.dimM_simd_block - 1, oc);
    }

    if (jcp.prop_kind != prop_kind::forward_inference) {

        parallel_nd(jcp.nb_oc, jcp.nb_ic, (jcp.oc_block * jcp.oc_reg_block), (jcp.ic_block * jcp.ic_reg_block),
                    [&](int ofm1, int ifm1, int ofm2, int ifm2) {
            float *U_base_ptr = is_fwd
                              ? &(U(ofm1, 0, 0, ifm1, ofm2, ifm2, 0, 0))
                              : &(U(ifm1, 0, 0, ofm1, ifm2, ofm2, 0, 0));
            weight_transform_data(jcp,
                    &(weights(
                        ofm1 * jcp.oc_block * jcp.oc_reg_block + ofm2,
                        ifm1 * jcp.ic_block * jcp.ic_reg_block + ifm2,
                        0, 0, 0, 0)),
                    U_base_ptr);
        });
    }

    parallel_nd(jcp.tile_block, [&](int tile_block) {
        int ithr = mkldnn_get_thread_num();

        for (int K_blk1 = 0; K_blk1 < jcp.dimK_nb_block; K_blk1++) {
            for (int K_blk2 = 0; K_blk2 < jcp.dimK_block; K_blk2++) {

                input_transform_tileblock_data(
                        tile_block, jcp,
                        &(input(0, K_blk1 * jcp.dimK_block + K_blk2, 0, 0, 0)),
                        &(V(ithr, 0, 0, 0, K_blk1, K_blk2, 0, 0)));
            }
        }

        for (int oj = 0; oj < alpha; oj++) {
            for (int oi = 0; oi < alpha; oi++) {
                for (int M_blk1 = 0; M_blk1 < jcp.dimM_nb_block; M_blk1++)
                for (int K_blk1 = 0; K_blk1 < jcp.dimK_nb_block; K_blk1++)
                for (int N_blk = 0; N_blk < jcp.dimN_block; N_blk++)
                    kernel_->gemm_loop_ker(
                            (float *)&(M(ithr, M_blk1, oj, oi,
                                    N_blk, 0, 0, 0)),
                            (const float *)&(U(M_blk1, oj, oi, K_blk1,
                                    0, 0, 0, 0)),
                            (const float *)&(V(ithr, oj, oi,
                                    N_blk, K_blk1, 0, 0, 0)), K_blk1);
            }
        }

        for (int M_blk1 = 0; M_blk1 < jcp.dimM_nb_block; M_blk1++) {
            for (int M_blk2 = 0; M_blk2 < jcp.dimM_block * jcp.dimM_reg_block;
                  M_blk2++) {
                const int M_blk =
                    M_blk1 * jcp.dimM_block  * jcp.dimM_reg_block + M_blk2;

                float *bias_ptr = wants_padded_bias
                    && M_blk == jcp.dimM / jcp.dimM_simd_block - 1
                    ? last_slice_bias : &bias(M_blk, 0);

                output_transform_tileblock_data(tile_block, jcp, p_ops,
                        &(M(ithr, M_blk1, 0, 0, 0, M_blk2, 0, 0)),
                        &(output(0, M_blk, 0, 0, 0)), bias_ptr);
            }
        }
    });
}

template struct _jit_avx512_core_fp32_wino_conv_4x3_t<true>;
template struct _jit_avx512_core_fp32_wino_conv_4x3_t<false>;

namespace {

void subarray_sum(size_t num_arrs, float *output, size_t nelems,
        float *input_ptrs[], size_t input_starts[], size_t input_ends[]) {
    using namespace nstl;
    const size_t block_size = 16 * 1024 / sizeof(float);
    const size_t blocks_number = nelems / block_size;
    const size_t tail = nelems % block_size;

PRAGMA_OMP(parallel)
    {
        const int ithr = mkldnn_get_thread_num();
        const int nthr = mkldnn_get_num_threads();
        size_t start{ 0 }, end{ 0 };
        balance211(blocks_number, nthr, ithr, start, end);

        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * block_size;
            size_t end_e = start_e + block_size;
            size_t input_start = max(start_e, min(input_starts[0], end_e));
            size_t input_end = max(start_e, min(input_ends[0], end_e));

            PRAGMA_OMP_SIMD()
            for (size_t e = start_e; e < input_start; e++) {
                output[e] = 0.f;
            }

            PRAGMA_OMP_SIMD()
            for (size_t e = input_start; e < input_end; e++) {
                output[e] = input_ptrs[0][e];
            }

            PRAGMA_OMP_SIMD()
            for (size_t e = input_end; e < end_e; e++) {
                output[e] = 0.f;
            }

            for (size_t a = 1; a < num_arrs; a++) {
                input_start = max(start_e, input_starts[a]);
                input_end = min(input_ends[a], end_e);

                PRAGMA_OMP_SIMD()
                for (size_t e = input_start; e < input_end; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            size_t end_e = nelems;
            size_t input_start = max(start_e, min(input_starts[0], end_e));
            size_t input_end = max(start_e, min(input_ends[0], end_e));

            PRAGMA_OMP_SIMD()
            for (size_t e = start_e; e < input_start; e++) {
                output[e] = 0.f;
            }

            PRAGMA_OMP_SIMD()
            for (size_t e = input_start; e < input_end; e++) {
                output[e] = input_ptrs[0][e];
            }

            PRAGMA_OMP_SIMD()
            for (size_t e = input_end; e < end_e; e++) {
                output[e] = 0.f;
            }

            for (size_t a = 1; a < num_arrs; a++) {
                input_start = max(start_e, input_starts[a]);
                input_end = min(input_ends[a], end_e);

                PRAGMA_OMP_SIMD()
                for (size_t e = input_start; e < input_end; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }
    }
}

const int max_threads_number = 1024;

// Sum to the first buffer array
void array_sum(size_t num_arrs, float *output,
    size_t nelems, float *input_ptrs[], bool reduce_to_first = true) {
    const size_t block_size = 16 * 1024 / sizeof(float);
    const size_t blocks_number = nelems / block_size;
    const size_t tail = nelems % block_size;

PRAGMA_OMP(parallel)
    {
        const size_t ithr = mkldnn_get_thread_num();
        const size_t nthr = mkldnn_get_num_threads();
        size_t start{ 0 }, end{ 0 };
        balance211(blocks_number, nthr, ithr, start, end);

        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * block_size;
            size_t end_e = start_e + block_size;
            if (!reduce_to_first) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] = input_ptrs[0][e];
                }
            }
            for (size_t a = 1; a < num_arrs; a++) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            size_t end_e = nelems;
            if (!reduce_to_first) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] = input_ptrs[0][e];
                }
            }
            for (size_t a = 1; a < num_arrs; a++) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }
    }
}
} //bwdw namespace

void jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_t::
_execute_backward_weights_SDGtWo(const float *ptr_src,
        const float *ptr_diff_dst, float *ptr_diff_weights,
        float *ptr_diff_bias,
        const memory_tracking::grantor_t &scratchpad) const {
    const auto &jcp = kernel_->jcp;
    const int nthreads = jcp.nthr;

    array_offset_calculator<float, 5> src((float *)ptr_src,
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)ptr_diff_dst,
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights(ptr_diff_weights,
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);

    array_offset_calculator<float, 8> Us(scratchpad.get<float>(key_wino_U),
            0, alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    const int U_sz = nthreads * alpha * alpha * jcp.oc / jcp.nb_oc
        * jcp.ic / jcp.nb_ic;
    array_offset_calculator<float, 7>diff_weights_prv(
            scratchpad.get<float>(key_wino_U) + U_sz,
            0, jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);

    array_offset_calculator<float, 8> M(scratchpad.get<float>(key_wino_M),
            0, alpha, alpha,
            jcp.oc_block,
            jcp.nb_tile_block_ur,
            jcp.tile_block_ur,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    array_offset_calculator<float, 7> V(scratchpad.get<float>(key_wino_V),
            0, alpha, alpha,
            jcp.ic_block,
            jcp.nb_tile_block_ur,
            jcp.tile_block_ur,
            jcp.ic_simd_block);

    array_offset_calculator<float, 2> diff_bias_prv(
            scratchpad.get<float>(key_conv_bia_reduction), nthreads, jcp.oc);

    auto trans_ker_p = jit_wino_transform_call_s();
    float I[alpha][alpha][simd_w];
    float T[alpha][alpha][simd_w];
    float G_I_3x3_4x4[9] = {-2.25f, -0.390625f, 0.87890625f, -2.640625f,
               0.625f, -0.625f, 1.5f, -1.5f, -2.640625f};
    float G_W_3x3_4x4[8] = {0.26890756302521f, -0.688403361344538f, 0.119514472455649f,
       0.430252100840336f, 0.168067226890756f, 0.179271708683473f, 0.403361344537815f,
       1.13777777777778f};
    float G_O_3x3_4x4[4] = {2.25f, 0.625f, 1.5f, 0.390625f};

PRAGMA_OMP(parallel num_threads(nthreads) firstprivate(trans_ker_p, I, T))
{
    if (jcp.with_bias) {
        parallel_nd_in_omp(nthreads, jcp.oc / simd_w,
            [&](int ithr, int ofm){
                float *pdbias = &(diff_bias_prv(ithr, ofm * simd_w));
                PRAGMA_OMP_SIMD()
                for (int v = 0; v < simd_w; v++) {
                    pdbias[v] = 0.0f;
                }
        });
    }

    int ithr = mkldnn_get_thread_num();
    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
        int first_tblk = 0;
PRAGMA_OMP(for)
        for (int tblk1 = 0; tblk1 < jcp.tile_block; ++tblk1) {
            int tile_index = tblk1 * jcp.nb_tile_block_ur * jcp.tile_block_ur;
            int img = tile_index / (jcp.itiles * jcp.jtiles);
            trans_ker_p.ti = tile_index % jcp.itiles;
            trans_ker_p.tj = (tile_index / jcp.itiles) % jcp.jtiles;
            trans_ker_p.M = I;
            trans_ker_p.T = T;
            trans_ker_p.G = G_I_3x3_4x4;
            for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                int ifm = ifm1 * jcp.ic_block + ifm2;
                trans_ker_p.src = (float *)&(src(img, ifm, 0, 0, 0));
                trans_ker_p.dst = (float *)&(V(ithr, 0, 0, ifm2, 0, 0, 0));
                kernel_->src_transform(&trans_ker_p);
            }

            for (int ofm1 = 0; ofm1 < jcp.nb_oc; ++ofm1) {
                trans_ker_p.G = G_W_3x3_4x4;
                for (int ofm2 = 0; ofm2 < jcp.oc_block; ++ofm2) {
                    int ofm = (ofm1 * jcp.oc_block + ofm2) * jcp.oc_reg_block;
                    trans_ker_p.src = (float *)&(diff_dst(img, ofm, 0, 0, 0));
                    trans_ker_p.dst = (float *)&(M(ithr, 0, 0, ofm2, 0, 0, 0, 0));
                    if (jcp.with_bias && ifm1 == 0) {
                        trans_ker_p.bias = (float *)&(diff_bias_prv(ithr, ofm * simd_w));
                        kernel_->diff_dst_transform_wbias(&trans_ker_p);
                    } else {
                        kernel_->diff_dst_transform(&trans_ker_p);
                    }
                }

                for (int oj = 0; oj < alpha; ++oj) {
                    for (int oi = 0; oi < alpha; ++oi) {
                        kernel_->gemm_loop_ker_first_iter(
                                &(Us(ithr, oj, oi, 0, 0, 0, 0, 0)),
                                &(M(ithr, oj, oi, 0, 0, 0, 0, 0)),
                                &(V(ithr, oj, oi, 0, 0, 0, 0)));
                    }
                }
                trans_ker_p.G = G_O_3x3_4x4;
                for (int ofm2 = 0; ofm2 < jcp.oc_block; ++ofm2) {
                    for (int ofm3 = 0; ofm3 < jcp.oc_reg_block; ++ofm3) {
                        int ofm = (ofm1 * jcp.oc_block + ofm2) * jcp.oc_reg_block
                                + ofm3;
                        for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                            int ifm = ifm1 * jcp.ic_block + ifm2;
                            trans_ker_p.src = (float *)&(Us(ithr, 0, 0,
                                        ofm2, ifm2, 0, ofm3, 0));
                            trans_ker_p.dst = (float *)&(diff_weights_prv(ithr,
                                        ofm, ifm, 0, 0, 0, 0));
                            if (first_tblk == 0) {
                                kernel_->diff_weights_transform(&trans_ker_p);
                            } else {
                                kernel_->diff_weights_transform_accum(&trans_ker_p);
                            }
                        }
                    }
                }
            }
            ++first_tblk;
        }
    }
}

    // Reduce diff-weights
    {
        float *output = ptr_diff_weights;
        float *input_base = scratchpad.get<float>(key_wino_U) + U_sz;
        int nelems = jcp.oc * jcp.ic * jcp.kh * jcp.kw;
        float *input_ptrs[max_threads_number];
        for (int i = 0; i < nthreads; ++i) {
            input_ptrs[i] = input_base + nelems * i;
        }
        array_sum(nthreads, output, nelems, input_ptrs, false);

        if (jcp.with_bias) {
            output = ptr_diff_bias;
            input_base = scratchpad.get<float>(key_conv_bia_reduction);
            for (int i = 0; i < nthreads; ++i) {
                input_ptrs[i] = input_base + jcp.oc * i;
            }
            array_sum(nthreads, output, jcp.oc_without_padding, input_ptrs,
                    false);
        }
    }
}

void jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_t::
_execute_backward_weights_S_D_Giot_W(const float *ptr_src,
        const float *ptr_diff_dst, float *ptr_diff_weights,
        float *ptr_diff_bias,
        const memory_tracking::grantor_t &scratchpad) const {
    const auto &jcp = kernel_->jcp;
    const int nthreads = jcp.nthr;

    array_offset_calculator<float, 5> src((float *)ptr_src,
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)ptr_diff_dst,
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)ptr_diff_weights,
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 1> diff_bias((float *)ptr_diff_bias, jcp.oc);

    array_offset_calculator<float, 9> U(scratchpad.get<float>(key_wino_U),
            jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    const int U_size = jcp.oc * jcp.ic * alpha * alpha;
    array_offset_calculator<float, 10> Us(
            scratchpad.get<float>(key_wino_U) + U_size,
            0, jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    array_offset_calculator<float, 9> M(scratchpad.get<float>(key_wino_M),
            jcp.nb_oc,
            jcp.tile_block,
            alpha, alpha,
            jcp.oc_block,
            jcp.nb_tile_block_ur,
            jcp.tile_block_ur ,
            jcp.oc_reg_block,
            jcp.oc_simd_block);

    array_offset_calculator<float, 8> V(scratchpad.get<float>(key_wino_V),
            jcp.nb_ic,
            jcp.tile_block,
            alpha, alpha,
            jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block);

    array_offset_calculator<float, 2> diff_bias_prv(
            scratchpad.get<float>(key_conv_bia_reduction), nthreads, jcp.oc);

    size_t input_starts[max_threads_number] = {0};
    size_t input_ends[max_threads_number] = {0};
    size_t first_tblk = 0;

    auto trans_ker_p = jit_wino_transform_call_s();
    float G_I_3x3_4x4[9] = {-2.25f, -0.390625f, 0.87890625f, -2.640625f,
               0.625f, -0.625f, 1.5f, -1.5f, -2.640625f};
    float G_W_3x3_4x4[8] = {0.26890756302521f, -0.688403361344538f,
        0.119514472455649f, 0.430252100840336f, 0.168067226890756f,
        0.179271708683473f, 0.403361344537815f, 1.13777777777778f};
    float G_O_3x3_4x4[4] = {2.25f, 0.625f, 1.5f, 0.390625f};
    float I[alpha][alpha][simd_w];
    float T[alpha][alpha][simd_w];

PRAGMA_OMP(parallel firstprivate(first_tblk, trans_ker_p, I, T))
{
    if (jcp.with_bias) {
        parallel_nd_in_omp(nthreads, jcp.oc, [&](int ithr, int ofm) {
            diff_bias_prv(ithr, ofm) = 0.0f;
        });
    }

    trans_ker_p.G = G_I_3x3_4x4;
    trans_ker_p.M = I;
    trans_ker_p.T = T;

    parallel_nd_in_omp(jcp.nb_ic, jcp.ic_block, jcp.mb,
        [&](int ifm1, int ifm2, int img){
         size_t ifm = ifm1 * jcp.ic_block + ifm2;
         size_t tile_base_index = img * (jcp.itiles * jcp.jtiles);
         size_t tblk3 = tile_base_index  % jcp.tile_block_ur;
         size_t tblk2 = (tile_base_index / jcp.tile_block_ur)
             % jcp.nb_tile_block_ur;
         size_t tblk1 = (tile_base_index / jcp.tile_block_ur)
             / jcp.nb_tile_block_ur;
         trans_ker_p.tile_count = tblk2 * jcp.tile_block_ur + tblk3;
         trans_ker_p.src = (float *)&(src(img, ifm, 0, 0, 0));
         trans_ker_p.dst = (float *)&(V(ifm1, tblk1, 0, 0, ifm2, 0, 0, 0));
         kernel_->src_transform(&trans_ker_p);
    });

    int ithr = mkldnn_get_thread_num();
    trans_ker_p.G = G_W_3x3_4x4;
    parallel_nd_in_omp(jcp.nb_oc, jcp.oc_block, jcp.mb,
        [&](int ofm1, int ofm2, int img){
        int ofm = (ofm1 * jcp.oc_block + ofm2) * jcp.oc_reg_block;
        size_t tile_base_index = img * (jcp.itiles * jcp.jtiles);
        size_t tblk3 = tile_base_index  % jcp.tile_block_ur;
        size_t tblk2 = (tile_base_index / jcp.tile_block_ur)
            % jcp.nb_tile_block_ur;
        size_t tblk1 = (tile_base_index / jcp.tile_block_ur)
            / jcp.nb_tile_block_ur;
        trans_ker_p.tile_count = tblk2 * jcp.tile_block_ur + tblk3;
        trans_ker_p.src = (float *)&(diff_dst(img, ofm, 0, 0, 0));
        trans_ker_p.dst = (float *)&(M(ofm1, tblk1, 0, 0, ofm2, 0, 0, 0, 0));
        if (jcp.with_bias) {
            trans_ker_p.bias = (float *)&(diff_bias_prv(ithr, ofm * simd_w));
            kernel_->diff_dst_transform_wbias(&trans_ker_p);
        } else {
            kernel_->diff_dst_transform(&trans_ker_p);
        }
    });

    PRAGMA_OMP(barrier)

    parallel_nd_in_omp(jcp.nb_ic, jcp.nb_oc, alpha, alpha, jcp.tile_block,
        [&](int ifm1, int ofm1, int oj, int oi, int tblk1){
        if (first_tblk == 0) {
            input_starts[ithr] =
                (float *)&(Us(ithr, ifm1, ofm1, oj, oi, 0, 0, 0,
                            0, 0))
                - (float *)&(Us(ithr, 0, 0, 0, 0, 0, 0,
                            0, 0, 0));
            input_ends[ithr] = input_starts[ithr]
                    + jcp.oc_block * jcp.ic_block
                      * jcp.ic_simd_block * jcp.oc_reg_block
                      * jcp.oc_simd_block;
        }
        else if (tblk1 == 0) {
            input_ends[ithr] += jcp.oc_block * jcp.ic_block
                * jcp.ic_simd_block * jcp.oc_reg_block
                * jcp.oc_simd_block;
        }

        if (first_tblk == 0 || tblk1 == 0) {
            kernel_->gemm_loop_ker_first_iter(
                    &(Us(ithr, ifm1, ofm1, oj, oi,
                            0, 0, 0, 0, 0)),
                    &(M(ofm1, tblk1, oj, oi, 0, 0, 0, 0, 0)),
                    &(V(ifm1, tblk1, oj, oi, 0, 0, 0, 0)));
        } else {
            kernel_->gemm_loop_ker(
                    &(Us(ithr, ifm1, ofm1, oj, oi,
                            0, 0, 0, 0, 0)),
                    &(M(ofm1, tblk1, oj, oi, 0, 0, 0, 0, 0)),
                    &(V(ifm1, tblk1, oj, oi, 0, 0, 0, 0)));
        }
        ++first_tblk;
    });
}

    // Reduce diff-weights
    {
        float *output = &(U(0, 0, 0, 0, 0, 0, 0, 0, 0));
        size_t nelems = jcp.ic * jcp.oc * alpha * alpha;
        float *input_ptrs[max_threads_number];
        for (int i = 0; i < nthreads; ++i)
            input_ptrs[i] = output + nelems * (i + 1);
        subarray_sum(nthreads, output, nelems, input_ptrs,
                input_starts, input_ends);
    }

    trans_ker_p.G = G_O_3x3_4x4;
PRAGMA_OMP(parallel firstprivate(trans_ker_p))
    {
        parallel_nd_in_omp(jcp.nb_ic, jcp.nb_oc, jcp.oc_block, jcp.ic_block, jcp.oc_reg_block,
            [&](int ifm1, int ofm1, int ofm2, int ifm2, int ofm3){
            int ofm = (ofm1 * jcp.oc_block + ofm2)
                * jcp.oc_reg_block + ofm3;
            int ifm = ifm1 * jcp.ic_block + ifm2;
            trans_ker_p.src = (float *)&(U(ifm1, ofm1, 0, 0,
                        ofm2, ifm2, 0, ofm3, 0));
            trans_ker_p.dst = (float *)&(diff_weights(ofm, ifm,
                        0, 0, 0, 0));
            kernel_->diff_weights_transform(&trans_ker_p);
        });
    }

    if (jcp.with_bias) {
        parallel_nd(jcp.oc / simd_w, [&](int ofm1) {
            float* pbias = &(diff_bias(ofm1 * simd_w));
            float *pbias_prv = &(diff_bias_prv(0, ofm1 * simd_w));

            const int blk_sz = ofm1 == jcp.oc / simd_w - 1
                ? jcp.oc_without_padding - ofm1 * simd_w : simd_w;

            PRAGMA_OMP_SIMD()
            for (int ofm2 = 0; ofm2 < blk_sz; ++ofm2) {
                pbias[ofm2] = pbias_prv[ofm2];
            }

            for (int ithr = 1; ithr < nthreads; ++ithr) {
                pbias_prv = &(diff_bias_prv(ithr, ofm1 * simd_w));
                PRAGMA_OMP_SIMD()
                for (int ofm2 = 0; ofm2 < blk_sz; ++ofm2) {
                    pbias[ofm2] += pbias_prv[ofm2];
                }
            }
        });
    }
}

}
}
}
// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
