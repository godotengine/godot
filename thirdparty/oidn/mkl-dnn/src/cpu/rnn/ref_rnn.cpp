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

/*
  General architecture

  for diff states, we have n_states + 1 as we have n_states diff
  to propagate to the previous iteration and 1 states to propagate
  to the previous layer
  index 0 is dh for cell(t-1, l) to consume
  index 1 is dc for cell(t-1, l) to consume
  index 2 is dh for cell(t, l-1) to consume
  this indexing enables to have the same indexing for states in elemwise
  function
  only the cell execution function should be impacted

 */

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "ref_rnn.hpp"
#include "../gemm/gemm.hpp"
#include "../simple_q10n.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::memory_tracking::names;
using namespace rnn_utils;
#define AOC array_offset_calculator

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
void _ref_rnn_common_t<aprop, src_type, weights_type>::gates_reduction(
        const rnn_conf_t &rnn, const acc_data_t *ws_gates_,
        float *diff_bias_) const {
    auto body = [&](int i, int k) {
        for (int j = 0; j < rnn.mb; j++)
            diff_bias_[i * rnn.dic + k]
                    += ws_gates_[j * rnn.gates_ws_ld + i * rnn.dic + k];
    };

    // @todo block k on simd-width
#if MKLDNN_THR == MKLDNN_THR_OMP && _OPENMP >= 201307 \
    /* icc 17.0 has a problem with simd collapse */ \
    && !((defined __INTEL_COMPILER) && (__INTEL_COMPILER == 1700))
#pragma omp parallel for simd collapse(2)
    for (int i = 0; i < rnn.n_gates; i++)
        for (int k = 0; k < rnn.dic; k++)
            body(i, k);
#else
    parallel_nd(rnn.n_gates, rnn.dic, body);
#endif
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
rnn_gemm_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::gemm)) {
    assert(ldA * ldB * ldC != 0);
    extended_sgemm(&transA, &transB, &m, &n, &k, &alpha, a_, &ldA, b_, &ldB,
            &beta, c_, &ldC, nullptr, pd()->rnn_.use_jit_gemm);
}

template <>
rnn_gemm_sig((ref_rnn_fwd_u8s8_t::gemm)) {
    assert(!"non packed gemm is disabled for int8");
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
rnn_gemm_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::packed_gemm)) {
#if (USE_MKL_PACKED_GEMM)
    assert(transA == 'N');
    cblas_sgemm_compute(CblasColMajor, CblasPacked,
            (transB == 'T') ? CblasTrans : CblasNoTrans, m, n, k, a_, ldA, b_,
            ldB, beta, c_, ldC);
#else
    UNUSED(transA);
    UNUSED(transB);
    UNUSED(m);
    UNUSED(n);
    UNUSED(k);
    UNUSED(alpha);
    UNUSED(ldA);
    UNUSED(b_);
    UNUSED(ldB);
    UNUSED(beta);
    UNUSED(c_);
    UNUSED(ldC);
    assert(!"packed gemm is disabled");
#endif
}

template <>
rnn_gemm_sig((ref_rnn_fwd_u8s8_t::packed_gemm)) {
#if (USE_MKL_PACKED_GEMM)
    int8_t offseta = 0, offsetb = 0;
    int32_t offsetc = 0;
    cblas_gemm_s8u8s32_compute(CblasColMajor, (CBLAS_TRANSPOSE)CblasPacked,
            CblasNoTrans, CblasFixOffset, m, n, k, alpha, a_, ldA, offseta, b_,
            ldB, offsetb, beta, c_, ldC, &offsetc);
#else
    UNUSED(transA);
    UNUSED(transB);
    UNUSED(m);
    UNUSED(n);
    UNUSED(k);
    UNUSED(alpha);
    UNUSED(ldA);
    UNUSED(b_);
    UNUSED(ldB);
    UNUSED(beta);
    UNUSED(c_);
    UNUSED(ldC);
    assert(!"packed gemm is disabled");
#endif
}

//*************** Grid computations strategy: linear ***************//
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
rnn_grid_execution_sig(
        (_ref_rnn_common_t<aprop, src_type, weights_type>::linear_execution)) {
    AOC<src_data_t, 4> ws_states(ws_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_iter + 1, rnn.states_nld * rnn.states_ws_ld);
    AOC<float, 4> ws_c_states(ws_c_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_iter + 1, rnn.states_nld * rnn.states_ws_ld);
    AOC<float, 5> ws_diff_states(ws_diff_states_, rnn.n_layer + 1, rnn.n_dir,
            (rnn.n_states + 1), rnn.n_iter + 1,
            rnn.states_nld * rnn.states_ws_ld);
    AOC<acc_data_t, 4> ws_gates(ws_gates_, rnn.n_layer, rnn.n_dir, rnn.n_iter,
            rnn.gates_nld * rnn.gates_ws_ld);
    AOC<weights_data_t *, 3> weights_input(
            weights_layer_, rnn.n_layer, rnn.n_dir, rnn.n_parts_weights_layer);
    AOC<weights_data_t *, 3> weights_states(
            weights_states_, rnn.n_layer, rnn.n_dir, rnn.n_parts_weights_iter);
    AOC<float*, 3> bias(
        bias_, rnn.n_layer, rnn.n_dir, rnn.n_parts_bias);
    AOC<float, 3> diff_weights_layer(diff_weights_layer_, rnn.n_layer,
            rnn.n_dir,
            rnn.diff_weights_layer_nld * rnn.diff_weights_layer_ld);
    AOC<float, 3> diff_weights_iter(diff_weights_iter_, rnn.n_layer, rnn.n_dir,
            rnn.diff_weights_iter_nld * rnn.diff_weights_iter_ld);
    AOC<float, 3> diff_bias(
            diff_bias_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dic);
    AOC<float, 4> ws_grid(
            ws_grid_, rnn.n_layer, rnn.n_dir, rnn.n_iter, (int)rnn.ws_per_cell);

    // We run the grid of computation
    for (int dir = 0; dir < rnn.n_dir; dir++) {
        for (int j = 0; j < rnn.n_layer; j++) {
            int lay = (aprop == prop_kind::forward) ? j : rnn.n_layer - j - 1;

            if ((aprop == prop_kind::forward) && rnn.merge_gemm_layer) {
                (this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dic,
                        rnn.mb * rnn.n_iter, rnn.slc, 1.0,
                        weights_input(lay, dir, 0), rnn.weights_iter_ld,
                        &(ws_states(lay, dir, 1, 0)), rnn.states_ws_ld, 0.0,
                        &(ws_gates(lay, dir, 0, 0)), rnn.gates_ws_ld);
            }

            for (int i = 0; i < rnn.n_iter; i++) {
                int iter = (aprop == prop_kind::forward) ? i : rnn.n_iter - i - 1;
                (this->*cell_func)(rnn,
                        &(ws_states(lay + 1, dir, iter + 1, 0)),
                        &(ws_c_states(lay + 1, dir, iter + 1, 0)),
                        &(ws_diff_states(lay, dir, 0, iter, 0)),
                        &(weights_input(lay, dir, 0)),
                        &(weights_states(lay, dir, 0)),
                        &(bias(lay, dir, 0)),
                        &(ws_states(lay, dir, iter + 1, 0)),
                        &(ws_states(lay + 1, dir, iter, 0)),
                        &(ws_c_states(lay + 1, dir, iter, 0)),
                        &(ws_diff_states(lay + 1, dir, 0, iter, 0)),
                        &(ws_diff_states(lay, dir, 0, iter + 1, 0)),
                        &(diff_weights_layer(lay, dir, 0)),
                        &(diff_weights_iter(lay, dir, 0)),
                        &(diff_bias(lay, dir, 0)),
                        &(ws_gates(lay, dir, iter, 0)),
                        &(ws_grid(lay, dir, iter, 0)),
                        ws_cell_);
            }

            if ((aprop == prop_kind::backward) && rnn.merge_gemm_layer) {
                (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb * rnn.n_iter,
                        rnn.n_gates * rnn.dic, 1.0, weights_input(lay, dir, 0),
                        rnn.weights_layer_ld,
                        (src_data_t *)(&(ws_gates(lay, dir, 0, 0))),
                        rnn.gates_ws_ld, 0.0,
                        (acc_data_t *)(&(ws_diff_states(
                                lay, dir, rnn.n_states, 0, 0))),
                        rnn.states_ws_ld);
                gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.slc,
                        rnn.mb * rnn.n_iter, 1.0,
                        (weights_data_t *)(&(ws_gates(lay, dir, 0, 0))),
                        rnn.gates_ws_ld,
                        (src_data_t *)(&(ws_states(lay, dir, 1, 0))),
                        rnn.states_ws_ld, 1.0,
                        (acc_data_t *)(&(diff_weights_layer(lay, dir, 0))),
                        rnn.diff_weights_layer_ld);
            }
            if ((aprop == prop_kind::backward) && rnn.merge_gemm_iter) {
                gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.sic,
                        rnn.mb * rnn.n_iter, 1.0,
                        (weights_data_t *)(&(ws_gates(lay, dir, 0, 0))),
                        rnn.gates_ws_ld,
                        (src_data_t *)(&(ws_states(lay + 1, dir, 0, 0))),
                        rnn.states_ws_ld, 1.0,
                        (acc_data_t *)(&(diff_weights_iter(lay, dir, 0))),
                        rnn.diff_weights_iter_ld);
            }
        }
    }
}

//********* GRID computations strategy: utility functions **********//

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
void _ref_rnn_common_t<aprop, src_type, weights_type>::copy_init_layer(
        const rnn_conf_t &rnn, src_data_t *__restrict ws_states_,
        float *__restrict ws_diff_states_, const src_data_t *__restrict xt_,
        const float *__restrict diff_dst_layer_) const {

    AOC<src_data_t, 4> ws_states(
            ws_states_, rnn.n_dir, rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    auto xt_d = memory_desc_wrapper(pd()->src_md(0));

    parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
        auto xxt = xt_ + xt_d.blk_off(it, b);
        src_data_t *ws_l2r_ptr = &(ws_states(0, it + 1, b, 0));
        src_data_t *ws_r2l_ptr = &(ws_states(rnn.n_dir - 1, rnn.n_iter - it, b, 0));
        if (rnn.exec_dir != r2l)
            for (int c = 0; c < rnn.slc; c++)
                ws_l2r_ptr[c] = xxt[c];
        if (rnn.exec_dir != l2r)
            for (int c = 0; c < rnn.slc; c++)
                ws_r2l_ptr[c] = xxt[c];
    });
}

template <>
void ref_rnn_bwd_f32_t::copy_init_layer(const rnn_conf_t &rnn,
        src_data_t *ws_states_, float *ws_diff_states_, const src_data_t *xt_,
        const float *diff_dst_layer_) const {
    AOC<float, 6> ws_diff_states(ws_diff_states_, rnn.n_layer + 1, rnn.n_dir,
            (rnn.n_states + 1), rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    auto diff_dst_layer_d = memory_desc_wrapper(pd()->diff_dst_md(0));

    switch (rnn.exec_dir) {
    case bi_concat:
        parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
            auto diff_dst_layer_x
                    = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
            for (int s = 0; s < rnn.dic; s++) {
                ws_diff_states(rnn.n_layer, 0, rnn.n_states, it, b, s)
                        = diff_dst_layer_x[s];
                ws_diff_states(
                        rnn.n_layer, 1, rnn.n_states, rnn.n_iter - it - 1, b, s)
                        = diff_dst_layer_x[rnn.dic + s];
            }
        });
        break;
    case bi_sum:
        parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
            auto diff_dst_layer_x
                    = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
            for (int s = 0; s < rnn.dic; s++) {
                ws_diff_states(rnn.n_layer, 0, rnn.n_states, it, b, s)
                        = diff_dst_layer_x[s];
                ws_diff_states(
                        rnn.n_layer, 1, rnn.n_states, rnn.n_iter - it - 1, b, s)
                        = diff_dst_layer_x[s];
            }
        });
        break;
    case l2r:
        parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
            auto diff_dst_layer_x
                    = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
            for (int s = 0; s < rnn.dic; s++) {
                ws_diff_states(rnn.n_layer, 0, rnn.n_states, it, b, s)
                        = diff_dst_layer_x[s];
            }
        });
        break;
    case r2l:
        parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
            auto diff_dst_layer_x = diff_dst_layer_
                    + diff_dst_layer_d.blk_off(rnn.n_iter - it - 1, b);
            for (int s = 0; s < rnn.dic; s++) {
                ws_diff_states(rnn.n_layer, 0, rnn.n_states, it, b, s)
                        = diff_dst_layer_x[s];
            }
        });
        break;
    default: assert(!"Unsupported direction"); break;
    }
}

/* For int8 configuration, input iteration states may be of types f32 or u8
 * Internally h_state is always stored in u8 and c_state is always stored in f32
 * If input states are of type u8 then h state is copied and c state is dequantized
 * If input states are of type f32 then h state is quantized and c_state is copied
 * */
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
template <typename input_data_t>
void _ref_rnn_common_t<aprop, src_type, weights_type>::copy_init_iter(
        const rnn_conf_t &rnn, src_data_t *__restrict ws_states_,
        float *__restrict ws_c_states_, float *__restrict ws_diff_states_,
        const input_data_t *__restrict firstit_states_,
        const float *__restrict diff_dst_iter_) const {
    AOC<src_data_t, 5> ws_states(ws_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    AOC<float, 5> ws_c_states(ws_c_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    const bool quantize = pd()->with_src_iter()
        && pd()->src_md(1)->data_type == data_type::f32
        && rnn.dt_conf != all_f32;
    auto maybe_q = [&](input_data_t f) {
        if (quantize) {
            float qf = f * data_scale + data_shift;
            return qz_a1b0<float, src_data_t>()(qf);
        } else
            return (src_data_t)f;
    };

    const bool dequantize = pd()->with_src_iter()
        && pd()->src_md(1)->data_type == data_type::u8;
    auto maybe_deq = [&](input_data_t s) {
        if (dequantize)
            return (((float)s - data_shift) / data_scale);
        else
            return (float)s;
    };
    auto firstit_states_d = memory_desc_wrapper(pd()->src_md(1));
    if (firstit_states_) {
        parallel_nd(
                rnn.n_layer, rnn.n_dir, rnn.mb, [&](int lay, int dir, int b) {
                    for (int s = 0; s < rnn.sic; s++)
                        ws_states(lay + 1, dir, 0, b, s) = maybe_q(
                                firstit_states_[firstit_states_d.blk_off(
                                        lay, dir, 0, b, s)]);
                    if (pd()->cell_kind() == alg_kind::vanilla_lstm)
                        for (int s = 0; s < rnn.sic; s++)
                            ws_c_states(lay + 1, dir, 0, b, s) = maybe_deq(
                                    firstit_states_[firstit_states_d.blk_off(
                                            lay, dir, 1, b, s)]);
                });
    } else {
        parallel_nd(
                rnn.n_layer, rnn.n_dir, rnn.mb, [&](int lay, int dir, int b) {
                    for (int j = 0; j < rnn.sic; j++) {
                        ws_states(lay + 1, dir, 0, b, j) = (src_data_t)0;
                        ws_c_states(lay + 1, dir, 0, b, j) = 0.0f;
                    }
        });
    }
}

template <>
template <typename input_data_t>
void ref_rnn_bwd_f32_t::copy_init_iter(const rnn_conf_t &rnn,
        src_data_t *ws_states_, float *ws_c_states_, float *ws_diff_states_,
        const input_data_t *firstit_states_,
        const float *diff_dst_iter_) const {
    AOC<float, 6> ws_diff_states(ws_diff_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_states + 1, rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    auto diff_dst_iter_d = memory_desc_wrapper(pd()->diff_dst_md(1));
    if (diff_dst_iter_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.n_states, rnn.mb,
                [&](int lay, int dir, int state, int b) {
                    array_copy(&(ws_diff_states(
                                       lay, dir, state, rnn.n_iter, b, 0)),
                            diff_dst_iter_
                                    + diff_dst_iter_d.blk_off(
                                              lay, dir, state, b),
                            rnn.dic);
                });
    } else {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.n_states, rnn.mb,
                [&](int lay, int dir, int state, int i) {
                    for (int j = 0; j < rnn.dic; j++)
                        ws_diff_states(lay, dir, state, rnn.n_iter, i, j)
                                = 0.0f;
                });
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
template <typename dst_data_t>
void _ref_rnn_common_t<aprop, src_type, weights_type>::copy_res_layer(
        const rnn_conf_t &rnn, dst_data_t *dst_layer_, float *diff_src_layer,
        const src_data_t *ws_states_, const float *ws_diff_states_) const {

    auto dst_layer_d = memory_desc_wrapper(pd()->dst_md(0));
    AOC<const src_data_t, 5> ws_states(ws_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    float shift = (pd()->attr()->rnn_data_qparams_.shift_);
    float scale = (pd()->attr()->rnn_data_qparams_.scale_);

    const bool dequantize = pd()->dst_md(0)->data_type == data_type::f32
            && rnn.dt_conf != all_f32;
    auto maybe_deq = [&](src_data_t s) {
        if (dequantize)
            return (dst_data_t)(((float)s - shift) / scale);
        else
            return (dst_data_t)s;
    };
    parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
        int dir = 0;
        if (rnn.exec_dir != r2l) {
            for (int s = 0; s < rnn.dic; s++) {
                dst_layer_[dst_layer_d.blk_off(it, b, dir * rnn.dic + s)]
                        = maybe_deq(ws_states(rnn.n_layer, dir, it + 1, b, s));
            }
            dir = 1;
        }
        if (rnn.exec_dir != l2r) {
            for (int s = 0; s < rnn.dic; s++)
                switch (rnn.exec_dir) {
                case bi_sum:
                    dst_layer_[dst_layer_d.blk_off(it, b, s)]
                            += maybe_deq(ws_states(
                                    rnn.n_layer, dir, rnn.n_iter - it, b, s));
                    break;
                default:
                    dst_layer_[dst_layer_d.blk_off(it, b, dir * rnn.dic + s)]
                            = maybe_deq(ws_states(
                                    rnn.n_layer, dir, rnn.n_iter - it, b, s));
                }
        }
    });
}

template <>
template <typename dst_data_t>
void ref_rnn_bwd_f32_t::copy_res_layer(
        const rnn_conf_t &rnn, dst_data_t *dst_layer_, float *diff_src_layer_,
        const src_data_t *ws_states_, const float *ws_diff_states_) const {
    auto diff_src_layer_d = memory_desc_wrapper(pd()->diff_src_md(0));
    AOC<const float, 6> ws_diff_states(ws_diff_states_, rnn.n_layer + 1,
            rnn.n_dir, rnn.n_states + 1, rnn.n_iter + 1, rnn.mb,
            rnn.states_ws_ld);

    parallel_nd(rnn.n_iter, rnn.mb, [&](int it, int b) {
        int dir = 0;
        for (int s = 0; s < rnn.slc; s++) {
            float *dst_addr = diff_src_layer_
                    + diff_src_layer_d.blk_off(
                              (rnn.exec_dir == r2l) ? rnn.n_iter - 1 - it : it,
                              b, dir * rnn.slc + s);
            float res = ws_diff_states(0, 0, rnn.n_states, it, b, s);
            if (rnn.n_dir - 1)
                res += ws_diff_states(
                        0, 1, rnn.n_states, rnn.n_iter - 1 - it, b, s);
            dst_addr[0] = res;
        }
    });
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
template <typename output_data_t>
void _ref_rnn_common_t<aprop, src_type, weights_type>::copy_res_iter(
        const rnn_conf_t &rnn, output_data_t *dst_iter_, float *diff_src_iter_,
        const src_data_t *ws_states_, float *ws_c_states_,
        const float *ws_diff_states_) const {
    auto dst_iter_d = memory_desc_wrapper(pd()->dst_md(1));
    AOC<const src_data_t, 5> ws_states(ws_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    AOC<const float, 5> ws_c_states(ws_c_states_, rnn.n_layer + 1, rnn.n_dir,
            rnn.n_iter + 1, rnn.mb, rnn.states_ws_ld);
    float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    const bool quantize = pd()->with_dst_iter()
        && pd()->dst_md(1)->data_type == data_type::u8
        && rnn.dt_conf != all_f32;
    auto maybe_q = [&](float f) {
        if (quantize) {
            float qf = f * data_scale + data_shift;
            return qz_a1b0<float, output_data_t>()(qf);
        } else
            return (output_data_t)f;
    };

    const bool dequantize = pd()->with_dst_iter()
        && pd()->dst_md(1)->data_type == data_type::f32
        && rnn.dt_conf != all_f32;
    auto maybe_deq = [&](src_data_t s) {
        if (dequantize)
            return (output_data_t)(((float)s - data_shift) / data_scale);
        else
            return (output_data_t)s;
    };
    if (dst_iter_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.mb,
                [&](int lay, int dir, int b) {
            for (int s = 0; s < rnn.dic; s++) {
                dst_iter_[dst_iter_d.blk_off(lay, dir, 0, b, s)]
                        = maybe_deq(ws_states(lay + 1, dir, rnn.n_iter, b, s));
            }
            if (pd()->cell_kind() == alg_kind::vanilla_lstm)
                    for (int s = 0; s < rnn.dic; s++) {
                        dst_iter_[dst_iter_d.blk_off(lay, dir, 1, b, s)]
                                = maybe_q(ws_c_states(
                                        lay + 1, dir, rnn.n_iter, b, s));
                    }
            });
    }
}

template <>
template <typename output_data_t>
void ref_rnn_bwd_f32_t::copy_res_iter(
        const rnn_conf_t &rnn, output_data_t *dst_iter_, float *diff_src_iter_,
        const src_data_t *ws_states_, float *ws_c_states_,
        const float *ws_diff_states_) const {
    auto diff_src_iter_d = memory_desc_wrapper(pd()->diff_src_md(1));
    AOC<const float, 6> ws_diff_states(ws_diff_states_, rnn.n_layer + 1,
            rnn.n_dir, rnn.n_states + 1, rnn.n_iter + 1, rnn.mb,
            rnn.states_ws_ld);
    if (diff_src_iter_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.n_states, rnn.mb,
                [&](int lay, int dir, int state, int b) {
                    for (int s = 0; s < rnn.sic; s++) {
                        diff_src_iter_[diff_src_iter_d.blk_off(
                                lay, dir, state, b, s)]
                                = ws_diff_states(lay, dir, state, 0, b, s);
                    }
                });
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
rnn_bias_prepare_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::bias_prepare)) {
    /* Original set of bias provided by the user */
    AOC<const float, 5> b(
            b_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dic);
    /* Array of pointers initialized in packing */
    AOC<float *, 3> bias(bias_, rnn.n_layer, rnn.n_dir, rnn.n_parts_bias);
    AOC<float, 3> scratch_bias(
            scratch_bias_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dic);

    if (rnn.copy_bias) {
        parallel_nd(rnn.n_layer * rnn.n_dir * rnn.n_bias * rnn.dic,
                [&](size_t i) { scratch_bias_[i] = b_[i]; });
    }

    for (int i = 0; i < rnn.n_layer; i++) {
        for (int d = 0; d < rnn.n_dir; d++) {
            int offset_bias = 0;
            for (int p = 0; p < rnn.n_parts_bias; p++) {
                bias(i, d, p) = rnn.copy_bias
                        ? (float *) &scratch_bias(i, d, offset_bias)
                        : (float *) &b(i, d, offset_bias);
                offset_bias += rnn.parts_bias[p] * rnn.dic;
            }
        }
    }

}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
rnn_bias_finalize_sig(
        (_ref_rnn_common_t<aprop, src_type, weights_type>::bias_finalize)) {
    if (rnn.dt_conf != all_f32) {
        float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
        float data_scale = pd()->attr()->rnn_data_qparams_.scale_;
        float *weights_scales = pd()->attr()->rnn_weights_qparams_.scales_;
        bool scale_per_oc = pd()->attr()->rnn_weights_qparams_.mask_ != 0;
        for (int i = 0; i < rnn.n_layer * rnn.n_dir; i++)
            for (int j = 0; j < rnn.n_bias * rnn.dic; j++) {
                size_t off = i * rnn.n_bias * rnn.dic + j;
                float weights_scale
                        = scale_per_oc ? weights_scales[j] : weights_scales[0];
                scratch_bias_[off] -= (w_iter_comp[off] + w_layer_comp[off])
                        * data_shift / (weights_scale * data_scale);
            }
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
rnn_weights_assign_sig((_ref_rnn_common_t<aprop, src_type,
        weights_type>::assign_packed_weights)) {
    assert(md->format_kind == format_kind::rnn_packed);
    const auto packed_desc = md->format_desc.rnn_packed_desc;
    AOC<weights_data_t *, 3> weights(weights_,
            rnn.n_layer, rnn.n_dir, packed_desc.n_parts);

    size_t offset_packed = 0;
    for (int l = 0; l < rnn.n_layer; l++)
        for (int d = 0; d < rnn.n_dir; d++) {
            for (int p = 0; p < packed_desc.n_parts; p++) {
                weights(l, d, p) = (weights_data_t *)&w_[offset_packed];
                offset_packed
                    += packed_desc.part_pack_size[p] / sizeof(weights_data_t);
            }
        }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
rnn_weights_assign_sig(
        (_ref_rnn_common_t<aprop, src_type, weights_type>::assign_weights)) {
    assert(md->format_kind == format_kind::blocked);
    const auto &blk = md->format_desc.blocking;
    /* Original set of weights provided by the user */
    AOC<const weights_data_t, 3> w(w_,
            rnn.n_layer, rnn.n_dir, (int)blk.strides[1]);
    /* Array of pointers for each part of weights */
    AOC<weights_data_t *, 3> weights(weights_, rnn.n_layer, rnn.n_dir, n_parts);

    for (int i = 0; i < rnn.n_layer; i++)
        for (int d = 0; d < rnn.n_dir; d++) {
            size_t offset_weights = 0;
            for (int p = 0; p < n_parts; p++) {
                weights(i, d, p) = (weights_data_t *)&w(i, d, offset_weights);
                offset_weights += gates_per_part[p] * blk.strides[3];
            }
        }
}

//********************* Execution function *********************//
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
void _ref_rnn_common_t<aprop, src_type, weights_type>::execute_(
        const exec_ctx_t &ctx) const {
    const rnn_conf_t &rnn = this->pd()->rnn_;
    auto input = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC_LAYER);
    auto states = CTX_IN_MEM(const char *, MKLDNN_ARG_SRC_ITER);
    auto layer_weights_n_comp = CTX_IN_MEM(const char *, MKLDNN_ARG_WEIGHTS_LAYER);
    auto iter_weights_n_comp = CTX_IN_MEM(const char *, MKLDNN_ARG_WEIGHTS_ITER);
    auto bias = CTX_IN_MEM(const float *, MKLDNN_ARG_BIAS);

    auto dst_last_layer = rnn.is_fwd
        ? CTX_OUT_MEM(char *, MKLDNN_ARG_DST_LAYER)
        : const_cast<char *>(CTX_IN_MEM(const char *, MKLDNN_ARG_DST_LAYER));
    auto dst_last_iter = rnn.is_fwd
        ? CTX_OUT_MEM(char *, MKLDNN_ARG_DST_ITER)
        : const_cast<char *>(CTX_IN_MEM(const char *, MKLDNN_ARG_DST_ITER));

    auto diff_dst_layer = CTX_IN_MEM(const float *, MKLDNN_ARG_DIFF_DST_LAYER);
    auto diff_dst_iter = CTX_IN_MEM(const float *, MKLDNN_ARG_DIFF_DST_ITER);

    auto w_layer = reinterpret_cast<const weights_data_t *>(layer_weights_n_comp);
    auto w_iter = reinterpret_cast<const weights_data_t *>(iter_weights_n_comp);
    auto w_iter_comp = reinterpret_cast<const float *>(
            iter_weights_n_comp + rnn.weights_iter_comp_offset);
    auto w_layer_comp = reinterpret_cast<const float *>(
            layer_weights_n_comp + rnn.weights_layer_comp_offset);

    auto scratchpad = this->scratchpad(ctx);

    auto ptr_wei_layer
            = scratchpad.template get<weights_data_t *>(key_rnn_ptrs_wei_layer);
    auto ptr_wei_iter
            = scratchpad.template get<weights_data_t *>(key_rnn_ptrs_wei_iter);
    auto ptr_bias =
        scratchpad.template get<float *>(key_rnn_ptrs_bia);

    // fetchihg buffers from the workspace
    // if no workspace was provided we use the scratchpad
    char *scratch_ptr = scratchpad.template get<char>(key_rnn_space);
    char *ws_ptr = nullptr;
    if (rnn.use_workspace)
        ws_ptr = rnn.is_fwd
            ? CTX_OUT_MEM(char *, MKLDNN_ARG_WORKSPACE)
            : const_cast<char *>(CTX_IN_MEM(const char *, MKLDNN_ARG_WORKSPACE));

    char *base_ptr = rnn.use_workspace ? ws_ptr : scratch_ptr;
    acc_data_t *ws_gates = (acc_data_t *)(base_ptr + ws_gates_offset_);
    src_data_t *ws_states = (src_data_t *)(base_ptr + ws_states_offset_);
    float *ws_c_states = (float *)(base_ptr + ws_c_states_offset_);
    float *ws_diff_states = (float *)(base_ptr + ws_diff_states_offset_);
    float *ws_grid = (float *)(base_ptr + ws_grid_comp_offset_);
    float *ws_cell = (float *)(base_ptr + ws_cell_comp_offset_);

    auto diff_src_layer = CTX_OUT_MEM(float *, MKLDNN_ARG_DIFF_SRC_LAYER);
    auto diff_src_iter = CTX_OUT_MEM(float *, MKLDNN_ARG_DIFF_SRC_ITER);

    auto diff_weights_layer = CTX_OUT_MEM(float *, MKLDNN_ARG_DIFF_WEIGHTS_LAYER);
    auto diff_weights_iter = CTX_OUT_MEM(float *, MKLDNN_ARG_DIFF_WEIGHTS_ITER);
    auto diff_bias = CTX_OUT_MEM(float *, MKLDNN_ARG_DIFF_BIAS);

    // Fetching extra buffers from scratchpad
    float *ws_bias = (float *)(scratch_ptr + ws_bias_offset_);

    // initialize diff_states to 0
    if (aprop == prop_kind::backward)
        array_set(ws_diff_states, 0.0f, rnn.ws_diff_states_size / sizeof(float));

    /* Pack(if using packed gemm API) or copy(if input arrays have bad leading
     * dimension */
    (this->*bias_preparation_func)(rnn, ptr_bias, bias, ws_bias);

    (this->*weights_iter_assign_func)(rnn, pd()->weights_md(1),
            rnn.weights_iter_nld, rnn.weights_iter_ld, rnn.dic,
            rnn.sic, rnn.n_parts_weights_iter, rnn.parts_weights_iter,
            rnn.part_weights_iter_pack_size, ptr_wei_iter, w_iter,
            ptr_bias, bias, ws_bias);
    (this->*weights_layer_assign_func)(rnn, pd()->weights_md(0),
            rnn.weights_layer_nld, rnn.weights_layer_ld, rnn.dic, rnn.slc,
            rnn.n_parts_weights_layer, rnn.parts_weights_layer,
            rnn.part_weights_layer_pack_size, ptr_wei_layer, w_layer, ptr_bias,
            bias, ws_bias);

    (this->*bias_finalization_func)(rnn, ws_bias, w_iter_comp, w_layer_comp);

    // we first need to copy the initial states and input into ws
    copy_init_layer(rnn, ws_states, ws_diff_states, input, diff_dst_layer);
    if (rnn.dt_conf == f32u8f32u8 || rnn.dt_conf == f32u8f32f32
            || rnn.dt_conf == all_f32)
        copy_init_iter(rnn, ws_states, ws_c_states, ws_diff_states,
                (const float *)states, diff_dst_iter);
    else if (rnn.dt_conf == u8u8u8u8 || rnn.dt_conf == u8u8u8f32)
        copy_init_iter(rnn, ws_states, ws_c_states, ws_diff_states,
                (const uint8_t *)states, diff_dst_iter);
    else
        assert(!"unimplemented");

    // run the execution on the grid
    (this->*grid_computation)(rnn, ptr_wei_layer, ptr_wei_iter, ptr_bias,
            ws_states, ws_c_states, ws_diff_states, ws_gates, ws_cell, ws_grid,
            diff_weights_layer, diff_weights_iter, diff_bias);

    // Finally we copy the results to the result buffers
    if (rnn.dt_conf == u8u8u8f32 || rnn.dt_conf == f32u8f32f32
            || rnn.dt_conf == all_f32)
        copy_res_layer(rnn, (float *)dst_last_layer, diff_src_layer, ws_states,
                ws_diff_states);
    else if (rnn.dt_conf == u8u8u8u8 || rnn.dt_conf == f32u8f32u8)
        copy_res_layer(rnn, (uint8_t *)dst_last_layer, diff_src_layer,
                ws_states, ws_diff_states);
    else
        assert(!"unimplemented");

    if (rnn.dt_conf == f32u8f32u8 || rnn.dt_conf == f32u8f32f32
            || rnn.dt_conf == all_f32)
        copy_res_iter(rnn, (float *)dst_last_iter, diff_src_iter, ws_states,
                ws_c_states, ws_diff_states);
    else if (rnn.dt_conf == u8u8u8u8 || rnn.dt_conf == u8u8u8f32)
        copy_res_iter(rnn, (uint8_t *)dst_last_iter, diff_src_iter, ws_states,
                ws_c_states, ws_diff_states);
    else
        assert(!"unimplemented");
};

/* Fix for MSVS warning C4661 */
template<> rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution);
template<> rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution);
template<> rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution);
template<> rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru);
template<> rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru);
template<> rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru);
template<> rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru_lbr);
template<> rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru_lbr);
template<> rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru_lbr);
template<> rnn_elemwise_sig(ref_rnn_fwd_f32_t::rnn_elemwise);
template<> rnn_elemwise_sig(ref_rnn_fwd_u8s8_t::rnn_elemwise);
template<> rnn_elemwise_sig(ref_rnn_bwd_f32_t::rnn_elemwise);
template<> rnn_elemwise_sig(ref_rnn_fwd_f32_t::lstm_elemwise);
template<> rnn_elemwise_sig(ref_rnn_fwd_u8s8_t::lstm_elemwise);
template<> rnn_elemwise_sig(ref_rnn_bwd_f32_t::lstm_elemwise);
template<> rnn_elemwise_sig(ref_rnn_fwd_f32_t::gru_lbr_elemwise);
template<> rnn_elemwise_sig(ref_rnn_fwd_u8s8_t::gru_lbr_elemwise);
template<> rnn_elemwise_sig(ref_rnn_bwd_f32_t::gru_lbr_elemwise);

template struct _ref_rnn_common_t<prop_kind::forward, data_type::f32, data_type::f32>;
template struct _ref_rnn_common_t<prop_kind::forward, data_type::u8, data_type::s8>;
template struct _ref_rnn_common_t<prop_kind::backward, data_type::f32, data_type::f32>;

#undef AOC
}
}
}
