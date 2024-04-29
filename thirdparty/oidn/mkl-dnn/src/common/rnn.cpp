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

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu/gemm/os_blas.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::types;
using namespace mkldnn::impl::utils;

namespace {
memory_desc_t copy_maybe_null(const memory_desc_t *md) {
    return md ? *md : zero_md();
}

rnn_desc_t zero_rnn_desc() {
    auto rd = rnn_desc_t();
    rd.src_layer_desc = zero_md();
    rd.src_iter_desc = zero_md();
    rd.weights_layer_desc = zero_md();
    rd.weights_iter_desc = zero_md();
    rd.bias_desc = zero_md();
    rd.dst_layer_desc = zero_md();
    rd.dst_iter_desc = zero_md();
    rd.diff_src_layer_desc = zero_md();
    rd.diff_src_iter_desc = zero_md();
    rd.diff_weights_layer_desc = zero_md();
    rd.diff_weights_iter_desc = zero_md();
    rd.diff_bias_desc = zero_md();
    rd.diff_dst_layer_desc = zero_md();
    rd.diff_dst_iter_desc = zero_md();
    return rd;
}
}

/* Public C Api */

status_t mkldnn_rnn_cell_desc_init(rnn_cell_desc_t *rnn_cell_desc,
        mkldnn_alg_kind_t cell_kind, mkldnn_alg_kind_t act_f,
        unsigned int flags, float alpha, float clipping) {
    using namespace mkldnn::impl::alg_kind;

    bool args_ok = true
            && one_of(cell_kind, vanilla_rnn, vanilla_lstm, vanilla_gru,
                    gru_linear_before_reset)
            && IMPLICATION(cell_kind == vanilla_rnn,
                    one_of(act_f, eltwise_relu, eltwise_tanh, eltwise_logistic));
    if (!args_ok)
        return invalid_arguments;

    auto rcd = mkldnn_rnn_cell_desc_t();

    rcd.cell_kind = cell_kind;
    rcd.activation_kind = act_f;
    rcd.flags = flags;
    rcd.alpha = rcd.flags & mkldnn_rnn_cell_with_relu ? alpha : 0;
    rcd.clipping = rcd.flags & mkldnn_rnn_cell_with_clipping ? clipping : 0;

    *rnn_cell_desc = rcd;

    return success;
}

int mkldnn_rnn_cell_get_gates_count(const rnn_cell_desc_t *rnn_cell_desc) {
    switch (rnn_cell_desc->cell_kind) {
    case mkldnn::impl::alg_kind::vanilla_rnn: return 1;
    case mkldnn::impl::alg_kind::vanilla_gru: return 3;
    case mkldnn::impl::alg_kind::gru_linear_before_reset: return 3;
    case mkldnn::impl::alg_kind::vanilla_lstm: return 4;
    default: assert(!"unknown cell kind"); return 0;
    }
    return 0;
}

int mkldnn_rnn_cell_get_states_count(const rnn_cell_desc_t *rnn_cell_desc) {
    switch (rnn_cell_desc->cell_kind) {
    case mkldnn::impl::alg_kind::vanilla_rnn: return 1;
    case mkldnn::impl::alg_kind::vanilla_gru: return 1;
    case mkldnn::impl::alg_kind::gru_linear_before_reset: return 1;
    case mkldnn::impl::alg_kind::vanilla_lstm: return 2;
    default: assert(!"unknown cell kind"); return 0;
    }
    return 0;
}

status_t check_data_type_consistency_fwd(const rnn_cell_desc_t *rnn_cell_desc,
        prop_kind_t prop_kind, const memory_desc_t *src_layer_desc,
        const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc,
        const memory_desc_t *dst_iter_desc) {
    using namespace data_type;
    data_type_t src_layer_dt = src_layer_desc->data_type;
    data_type_t dst_layer_dt = dst_layer_desc->data_type;
    data_type_t weights_iter_dt = weights_iter_desc->data_type;
    data_type_t weights_layer_dt = weights_layer_desc->data_type;

    bool is_f32 = everyone_is(f32, src_layer_dt, dst_layer_dt, weights_iter_dt,
                          weights_layer_dt)
            && IMPLICATION(!is_zero_md(src_iter_desc),
                          src_iter_desc->data_type == f32)
            && IMPLICATION(!is_zero_md(dst_iter_desc),
                          dst_iter_desc->data_type == f32)
            && IMPLICATION(!is_zero_md(bias_desc), bias_desc->data_type == f32);

#if USE_MKL_PACKED_GEMM
    bool is_u8u8u8 = src_layer_dt == u8
            && IMPLICATION(!is_zero_md(src_iter_desc),
                             src_iter_desc->data_type == u8)
            && IMPLICATION(!is_zero_md(dst_iter_desc),
                             dst_iter_desc->data_type == u8)
            && one_of(dst_layer_dt, u8, f32)
            && everyone_is(s8, weights_iter_dt, weights_layer_dt)
            && IMPLICATION(!is_zero_md(bias_desc), bias_desc->data_type == f32);

    bool is_f32u8f32 = src_layer_dt == u8
            && IMPLICATION(!is_zero_md(src_iter_desc),
                               src_iter_desc->data_type == f32)
            && IMPLICATION(!is_zero_md(dst_iter_desc),
                               dst_iter_desc->data_type == f32)
            && one_of(dst_layer_dt, u8, f32)
            && everyone_is(s8, weights_iter_dt, weights_layer_dt)
            && IMPLICATION(!is_zero_md(bias_desc), bias_desc->data_type == f32);

    bool is_inference = prop_kind == prop_kind::forward_inference;
    bool is_lstm = rnn_cell_desc->cell_kind == mkldnn_vanilla_lstm;

    return (is_f32 || ((is_u8u8u8 || is_f32u8f32) && is_lstm && is_inference))
            ? success
            : unimplemented;
#else
    return is_f32 ? success : unimplemented;
#endif
}

status_t check_dim_consistency(const rnn_cell_desc_t *rnn_cell_desc,
        rnn_direction_t direction, int L, int D, int T, int N, int S, int G,
        int SLC, int SIC, int DLC, int DIC, const memory_desc_t *src_layer_desc,
        const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc,
        const memory_desc_t *dst_iter_desc) {
    bool args_ok;

    // * algorithm specific
    args_ok = true
        && IMPLICATION(rnn_cell_desc->cell_kind == alg_kind::vanilla_gru,
                       DIC == SIC);
    if (!args_ok) return invalid_arguments;
    int extra_bias =
            rnn_cell_desc->cell_kind == alg_kind::gru_linear_before_reset;

    // * on num layers
    args_ok = true
        && L == weights_layer_desc->dims[0]
        && L == weights_iter_desc->dims[0]
        && IMPLICATION(!is_zero_md(bias_desc), L == bias_desc->dims[0])
        && IMPLICATION(!is_zero_md(src_iter_desc), L == src_iter_desc->dims[0])
        && IMPLICATION(!is_zero_md(dst_iter_desc), L == dst_iter_desc->dims[0]);
    if (!args_ok) return invalid_arguments;

    // * on num directions
    args_ok = true
        && D == weights_layer_desc->dims[1]
        && D == weights_iter_desc->dims[1]
        && IMPLICATION(!is_zero_md(bias_desc), D == bias_desc->dims[1])
        && IMPLICATION(!is_zero_md(src_iter_desc), D == src_iter_desc->dims[1])
        && IMPLICATION(!is_zero_md(dst_iter_desc), D == dst_iter_desc->dims[1]);
    if (!args_ok) return invalid_arguments;

    // * on num iterations
    args_ok = true
        && T == src_layer_desc->dims[0]
        && T == dst_layer_desc->dims[0];
    if (!args_ok) return invalid_arguments;

    // * on mb
    args_ok = true
        && N == src_layer_desc->dims[1]
        && N == dst_layer_desc->dims[1]
        && IMPLICATION(!is_zero_md(src_iter_desc), N == src_iter_desc->dims[3])
        && IMPLICATION(!is_zero_md(dst_iter_desc), N == dst_iter_desc->dims[3]);
    if (!args_ok) return invalid_arguments;

    // * on num gates
    args_ok = true
        && G == mkldnn_rnn_cell_get_gates_count(rnn_cell_desc)
        && G == weights_layer_desc->dims[3]
        && G == weights_iter_desc->dims[3]
        && IMPLICATION(!is_zero_md(bias_desc),
                G + extra_bias == bias_desc->dims[2]);
    if (!args_ok) return invalid_arguments;

    // * on num states
    args_ok = true
        && S == mkldnn_rnn_cell_get_states_count(rnn_cell_desc)
        && IMPLICATION(!is_zero_md(src_iter_desc), S == src_iter_desc->dims[2])
        && IMPLICATION(!is_zero_md(dst_iter_desc), S == dst_iter_desc->dims[2]);
    if (!args_ok) return invalid_arguments;

    // * on slc
    args_ok = true
        && SLC == weights_layer_desc->dims[2]
        && SLC == src_layer_desc->dims[2];
    if (!args_ok) return invalid_arguments;

    // * on sic
    args_ok = true
        && SIC == weights_iter_desc->dims[2]
        && IMPLICATION(!is_zero_md(src_iter_desc),
                SIC == src_iter_desc->dims[4]);
    if (!args_ok) return invalid_arguments;

    // * on dlc
    int dlc_multiplier = (direction == mkldnn_bidirectional_concat) ? 2 : 1;
    args_ok = true
        && DLC == dlc_multiplier * DIC
        && DLC == dst_layer_desc->dims[2];
    if (!args_ok) return invalid_arguments;

    // * on dic
    args_ok = true
        && DIC == weights_layer_desc->dims[4]
        && DIC == weights_iter_desc->dims[4]
        && IMPLICATION(!is_zero_md(bias_desc), DIC == bias_desc->dims[3])
        && IMPLICATION(!is_zero_md(dst_iter_desc),
                DIC == dst_iter_desc->dims[4]);
    if (!args_ok) return invalid_arguments;

    // * unrolling/fusion conditions
    args_ok = true
        && IMPLICATION(L > 1, (dlc_multiplier * SLC) == DLC)
        && IMPLICATION(T > 1, SIC == DIC);
    if (!args_ok) return invalid_arguments;

    return success;
}

status_t MKLDNN_API mkldnn_rnn_forward_desc_init(mkldnn_rnn_desc_t *rnn_desc,
        prop_kind_t prop_kind, const rnn_cell_desc_t *rnn_cell_desc,
        const rnn_direction_t direction, const memory_desc_t *src_layer_desc,
        const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc,
        const memory_desc_t *dst_iter_desc) {
    bool args_ok = true && rnn_cell_desc != nullptr
            && !any_null(src_layer_desc, weights_layer_desc, weights_iter_desc,
                       dst_layer_desc);
    if (!args_ok) return invalid_arguments;

    //check dimensions consistency
    int L = weights_layer_desc->dims[0];
    int T = src_layer_desc->dims[0];
    int N = src_layer_desc->dims[1];
    const int D = one_of(direction, mkldnn_unidirectional_left2right,
                          mkldnn_unidirectional_right2left) ?
            1 :
            2;
    int G = mkldnn_rnn_cell_get_gates_count(rnn_cell_desc);
    int S = mkldnn_rnn_cell_get_states_count(rnn_cell_desc);
    int SLC = src_layer_desc->dims[2];
    int SIC = weights_iter_desc->dims[2];
    int DLC = dst_layer_desc->dims[2];
    int DIC = weights_layer_desc->dims[4];

    CHECK(check_dim_consistency(rnn_cell_desc, direction, L, D, T, N, S,
            G, SLC, SIC, DLC, DIC, src_layer_desc, src_iter_desc,
            weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc,
            dst_iter_desc));

    CHECK(check_data_type_consistency_fwd(rnn_cell_desc, prop_kind,
            src_layer_desc, src_iter_desc, weights_layer_desc,
            weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc));

    // Create the descriptor
    mkldnn_rnn_desc_t rd = zero_rnn_desc();

    rd.primitive_kind = primitive_kind::rnn;
    rd.prop_kind = prop_kind;
    rd.cell_desc = *rnn_cell_desc;
    rd.direction = direction;
    rd.src_layer_desc = copy_maybe_null(src_layer_desc);
    rd.src_iter_desc = copy_maybe_null(src_iter_desc);
    rd.weights_layer_desc = copy_maybe_null(weights_layer_desc);
    rd.weights_iter_desc = copy_maybe_null(weights_iter_desc);
    rd.bias_desc = copy_maybe_null(bias_desc);
    rd.dst_layer_desc = copy_maybe_null(dst_layer_desc);
    rd.dst_iter_desc = copy_maybe_null(dst_iter_desc);

    *rnn_desc = rd;

    return success;
}

status_t MKLDNN_API mkldnn_rnn_backward_desc_init(mkldnn_rnn_desc_t *rnn_desc,
        prop_kind_t prop_kind, const rnn_cell_desc_t *rnn_cell_desc,
        const rnn_direction_t direction, const memory_desc_t *src_layer_desc,
        const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer_desc,
        const memory_desc_t *diff_dst_iter_desc) {
    bool args_ok = true
            && !any_null(src_layer_desc, weights_layer_desc, weights_iter_desc,
                       dst_layer_desc, diff_src_layer_desc,
                       diff_weights_layer_desc, diff_weights_iter_desc,
                       diff_dst_layer_desc);
    if (!args_ok)
        return invalid_arguments;

    auto xnor_md = [=](const memory_desc_t *a_md, const memory_desc_t *b_md) {
        return is_zero_md(a_md) == is_zero_md(b_md);
    };

    args_ok = args_ok && xnor_md(bias_desc, diff_bias_desc)
            && xnor_md(dst_iter_desc, diff_dst_iter_desc)
            && xnor_md(src_iter_desc, diff_src_iter_desc);
    if (!args_ok)
        return invalid_arguments;

    //check dimensions consistency
    int L = weights_layer_desc->dims[0];
    int T = src_layer_desc->dims[0];
    int N = src_layer_desc->dims[1];
    const int D = one_of(direction, mkldnn_unidirectional_left2right,
                          mkldnn_unidirectional_right2left) ?
            1 :
            2;
    int G = mkldnn_rnn_cell_get_gates_count(rnn_cell_desc);
    int S = mkldnn_rnn_cell_get_states_count(rnn_cell_desc);
    int SLC = src_layer_desc->dims[2];
    int SIC = weights_iter_desc->dims[2];
    int DLC = dst_layer_desc->dims[2];
    int DIC = weights_layer_desc->dims[4];

    status_t st = check_dim_consistency(rnn_cell_desc, direction, L, D, T, N, S,
            G, SLC, SIC, DLC, DIC, src_layer_desc, src_iter_desc,
            weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc,
            dst_iter_desc);
    if (st != success) return st;

    st = check_dim_consistency(rnn_cell_desc, direction, L, D, T, N, S,
            G, SLC, SIC, DLC, DIC, diff_src_layer_desc, diff_src_iter_desc,
            diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc,
            diff_dst_layer_desc, diff_dst_iter_desc);
    if (st != success) return st;

    mkldnn_rnn_desc_t rd = zero_rnn_desc();

    rd.primitive_kind = primitive_kind::rnn;
    rd.prop_kind = prop_kind;
    rd.cell_desc = *rnn_cell_desc;
    rd.direction = direction;

    rd.src_layer_desc = copy_maybe_null(src_layer_desc);
    rd.src_iter_desc = copy_maybe_null(src_iter_desc);
    rd.weights_layer_desc = copy_maybe_null(weights_layer_desc);
    rd.weights_iter_desc = copy_maybe_null(weights_iter_desc);
    rd.bias_desc = copy_maybe_null(bias_desc);
    rd.dst_layer_desc = copy_maybe_null(dst_layer_desc);
    rd.dst_iter_desc = copy_maybe_null(dst_iter_desc);
    rd.diff_src_layer_desc = copy_maybe_null(diff_src_layer_desc);
    rd.diff_src_iter_desc = copy_maybe_null(diff_src_iter_desc);
    rd.diff_weights_layer_desc = copy_maybe_null(diff_weights_layer_desc);
    rd.diff_weights_iter_desc = copy_maybe_null(diff_weights_iter_desc);
    rd.diff_bias_desc = copy_maybe_null(diff_bias_desc);
    rd.diff_dst_layer_desc = copy_maybe_null(diff_dst_layer_desc);
    rd.diff_dst_iter_desc = copy_maybe_null(diff_dst_iter_desc);

    *rnn_desc = rd;

    return success;
}
