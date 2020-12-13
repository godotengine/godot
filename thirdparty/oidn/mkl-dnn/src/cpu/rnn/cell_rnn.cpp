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
 * Cell execution of Vanilla RNN
 */

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
using namespace rnn_utils;

template <>
float activation<alg_kind::eltwise_relu, prop_kind::forward>(
        float dd, float s, float alpha, float cliping) {
    return relu_fwd<float>(s, alpha);
}

template <>
float activation<alg_kind::eltwise_relu, prop_kind::backward>(
        float dd, float s, float alpha, float cliping) {
    return relu_bwd<float>(dd, s, alpha);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::forward>(
        float dd, float s, float alpha, float cliping) {
    return tanh_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::backward>(
        float dd, float s, float alpha, float cliping) {
    return dd * one_m_square<float>(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::forward>(
        float dd, float s, float alpha, float cliping) {
    return logistic_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::backward>(
        float dd, float s, float alpha, float cliping) {
    return dd * x_m_square<float>(s);
}

template <>
rnn_elemwise_sig(ref_rnn_fwd_f32_t::rnn_elemwise) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);

    parallel_nd(rnn.mb, [&](int i) {
        for (int j = 0; j < rnn.dic; j++) {
            const float h
                    = activation_func(0, ws_gates(i, 0, j) + bias(0, j), 0, 0);
            ws_gates(i, 0, j) = states_t_l(i, j) = h;
        }
    });
}

template <>
rnn_elemwise_sig(ref_rnn_fwd_u8s8_t::rnn_elemwise) {
    assert(!"VANILLA RNN int8 is not supported");
}

template <>
rnn_elemwise_sig(ref_rnn_bwd_f32_t::rnn_elemwise) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);
    ws_diff_states_aoc_t diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc_t diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc_t diff_states_t_lp1(rnn, diff_states_t_lp1_);

    parallel_nd(rnn.mb, [&](int i) {
        for (int j = 0; j < rnn.dic; ++j) {
            const float dH = diff_states_t_lp1(rnn.n_states, i, j)
                    + diff_states_tp1_l(0, i, j);
            auto g = ws_gates(i, 0, j);
            ws_gates(i, 0, j) = activation_func(dH, g, 0, 0);
        }
    });
}

}
}
}
