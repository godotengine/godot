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
 * Cell execution GRU
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

#define AOC array_offset_calculator
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_[0]);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);

    // 1. gemm Wx[0-2],x
    if (!rnn.merge_gemm_layer) {
        (this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dic, rnn.mb,
                rnn.slc, 1.0, w_layer_[0], rnn.weights_layer_ld,
                states_t_lm1_, rnn.states_ws_ld, 0.0, ws_gates_,
                rnn.gates_ws_ld);
    }

    // 2. gemm Wh[0-1],h
    (this->*gemm_iter_func)('N', 'N', (rnn.n_gates - 1) * rnn.dic, rnn.mb,
            rnn.sic, 1.0, w_iter_[0], rnn.weights_iter_ld, states_tm1_l_,
            rnn.states_ws_ld, 1.0, ws_gates_, rnn.gates_ws_ld);

    // 3. activation zt and rt + elemwise multiplication rt,ht-1
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            ws_gates(i, 0, j) = logistic_fwd(ws_gates(i, 0, j) + bias(0, j));
            ws_gates(i, 1, j) = logistic_fwd(ws_gates(i, 1, j) + bias(1, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 1, j);
        }
    });

    // 4. gemm Wh[2],h~t
    (this->*gemm_iter_func)('N', 'N', rnn.dic, rnn.mb, rnn.sic, 1.0, w_iter_[1],
            rnn.weights_iter_ld, states_t_l_, rnn.states_ws_ld, 1.0,
            &(ws_gates(0, 2, 0)), rnn.gates_ws_ld);

    // 5. activation h~t + calculate ht
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            ws_gates(i, 2, j) = tanh_fwd(ws_gates(i, 2, j) + bias(2, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 0, j)
                    + (1.0f - ws_gates(i, 0, j)) * ws_gates(i, 2, j);
        }
    });
}

template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru) {
    assert(!"GRU int8 is not supported");
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);
    ws_diff_w_iter_aoc_t diff_w_iter(rnn, diff_w_iter_);
    ws_diff_states_aoc_t diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc_t diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc_t diff_states_t_lp1(rnn, diff_states_t_lp1_);

    // use state memory for intermediate computations
    // TODO: use cell ws for that
    float *dhG1_ = &(diff_states_t_l(rnn.n_states, 0, 0));
    float *hG1_ = dhG1_;
    AOC<float, 2> dhG1(dhG1_, rnn.states_nld, rnn.states_ws_ld);
    AOC<float, 2> hG1(hG1_, rnn.states_nld, rnn.states_ws_ld);

    // 1. calculate dG2, dG1, and part of dht-1
    // dG2^ = dh * (1 - G0) * (1 - G2^2)
    // dG0^ = dh * (ht-1 - G2) * u * (1 - G0)
    // dht-1 (part) = dh * G0
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float h = states_tm1_l(i, j);
            float dHt = diff_states_tp1_l(0, i, j)
                    + diff_states_t_lp1(rnn.n_states, i, j);
            float dG2 = (1.0f - ws_gates(i, 0, j)) * dHt
                    * one_m_square(ws_gates(i, 2, j));
            float dG0 = (h - ws_gates(i, 2, j)) * dHt
                    * x_m_square(ws_gates(i, 0, j));

            diff_states_t_l(0, i, j) = dHt * ws_gates(i, 0, j);
            ws_gates(i, 0, j) = dG0;
            ws_gates(i, 2, j) = dG2;
        }
    });

    // 2. calculate intermediate d(hG1)
    // d(hG1) = dG2 * W2h^t
    (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb, rnn.dic, 1.0, w_iter_[1],
            rnn.weights_iter_ld, &(ws_gates(0, 2, 0)), rnn.gates_ws_ld, 0.0,
            dhG1_, rnn.states_ws_ld);

    // 3. calculate dG1^ and part of dht-1
    // dG1^ = d(hG1) * h * G1 * (1 - G1)
    // dht-1 (part) += d(hG1) * G1
    // h * G1 (required for dWh)
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float h = states_tm1_l(i, j);
            float G1 = ws_gates(i, 1, j);
            diff_states_t_l(0, i, j) += dhG1(i, j) * G1;
            ws_gates(i, 1, j) = dhG1(i, j) * h * x_m_square(G1);
            hG1(i, j) = G1 * h;
        }
    });

    // 4. calculate diff weights
    // dWh1 += dG1 * h, dWh2 += dG2 * h, dWh3 += dG3 * (G1(*)h)
    gemm('N', 'T', (rnn.n_gates - 1) * rnn.dic, rnn.sic, rnn.mb, 1.0, ws_gates_,
            rnn.gates_ws_ld, states_tm1_l_, rnn.states_ws_ld, 1.0, diff_w_iter_,
            rnn.diff_weights_iter_ld);
    gemm('N', 'T', rnn.dic, rnn.sic, rnn.mb, 1.0, &(ws_gates(0, 2, 0)),
            rnn.gates_ws_ld, hG1_, rnn.states_ws_ld, 1.0,
            &(diff_w_iter(0, 2, 0)), rnn.diff_weights_iter_ld);

    // 5. calculate diff states
    // dht-1 += dG1 * W1h + dG0 * W0h
    (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
            (rnn.n_gates - 1) * rnn.dic, 1.0, w_iter_[0],
            rnn.weights_iter_ld, ws_gates_, rnn.gates_ws_ld, 1.0,
            diff_states_t_l_, rnn.states_ws_ld);

    if (!rnn.merge_gemm_layer) {
        // dWx += [dG0 dG1 dG2] * [x]
        gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.slc, rnn.mb, 1.0, ws_gates_,
                rnn.gates_ws_ld, states_t_lm1_, rnn.states_ws_ld, 1.0,
                diff_w_layer_, rnn.diff_weights_layer_ld);
        // dx = dG2 * W2x + dG1 * W1x + dG0 * W0x
        (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dic, 1.0, w_layer_[0],
                rnn.weights_layer_ld, ws_gates_, rnn.gates_ws_ld, 0.0,
                &(diff_states_t_l(rnn.n_states, 0, 0)), rnn.states_ws_ld);
    }

    // 6. calculate diff bias
    gates_reduction(rnn, ws_gates_, diff_bias_);
}
#undef AOC

}
}
}
