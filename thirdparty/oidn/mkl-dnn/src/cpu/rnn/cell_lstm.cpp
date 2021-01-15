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
 * Cell execution LSTM
 */

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "../simple_q10n.hpp"
#include "ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
using namespace rnn_utils;

template <>
rnn_elemwise_sig(ref_rnn_fwd_f32_t::lstm_elemwise) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t c_states_t_l(rnn, c_states_t_l_);
    ws_states_aoc_t c_states_tm1_l(rnn, c_states_tm1_l_);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            ws_gates(i, 0, j) = logistic_fwd(ws_gates(i, 0, j) + bias(0, j));
            ws_gates(i, 1, j) = logistic_fwd(ws_gates(i, 1, j) + bias(1, j));
            ws_gates(i, 2, j) = tanh_fwd(ws_gates(i, 2, j) + bias(2, j));
            ws_gates(i, 3, j) = logistic_fwd(ws_gates(i, 3, j) + bias(3, j));

            float tmp = ws_gates(i, 1, j) * c_states_tm1_l(i, j)
                    + ws_gates(i, 0, j) * ws_gates(i, 2, j);
            states_t_l(i, j) = ws_gates(i, 3, j) * tanh_fwd(tmp);
            c_states_t_l(i, j) = tmp;
        }
    });
}

template <>
rnn_elemwise_sig(ref_rnn_fwd_u8s8_t::lstm_elemwise) {
    ws_gates_aoc_s32_t ws_gates_s32(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_u8_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t c_states_t_l(rnn, c_states_t_l_);
    ws_states_aoc_t c_states_tm1_l(rnn, c_states_tm1_l_);

    float *weights_scales = pd()->attr()->rnn_weights_qparams_.scales_;
    float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    auto q_d = [&](float f) {
        float qf = f * data_scale + data_shift;
        return qz_a1b0<float, src_data_t>()(qf);
    };

    auto deq_w = [&](acc_data_t s, int gate, int j) {
        return pd()->attr()->rnn_weights_qparams_.mask_ == 0 ?
                saturate<float>(s) * (1.f / (weights_scales[0] * data_scale)) :
                saturate<float>(s) * (1.f / (weights_scales[gate * rnn.dic + j]
                                                   * data_scale));
    };

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float G0 = logistic_fwd<float>(
                    deq_w(ws_gates_s32(i, 0, j), 0, j) + bias(0, j));
            float G1 = logistic_fwd<float>(
                    deq_w(ws_gates_s32(i, 1, j), 1, j) + bias(1, j));
            float G2 = tanh_fwd<float>(
                    deq_w(ws_gates_s32(i, 2, j), 2, j) + bias(2, j));
            float G3 = logistic_fwd<float>(
                    deq_w(ws_gates_s32(i, 3, j), 3, j) + bias(3, j));
            float tmp = G1 * c_states_tm1_l(i, j) + G0 * G2;
            states_t_l(i, j) = q_d(G3 * tanh_fwd(tmp));
            c_states_t_l(i, j) = tmp;
        }
    });
}

template <>
rnn_elemwise_sig(ref_rnn_bwd_f32_t::lstm_elemwise) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_t c_states_t_l(rnn, c_states_t_l_);
    ws_states_aoc_t c_states_tm1_l(rnn, c_states_tm1_l_);
    ws_diff_states_aoc_t diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc_t diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc_t diff_states_t_lp1(rnn, diff_states_t_lp1_);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float Ct = c_states_t_l(i, j);
            /// @todo save it in the workspace in fwd pass or recompute it to
            /// save bw
            float tanhCt = tanh_fwd(Ct);
            // we have 2 incoming diffs on Ht
            float dHt = diff_states_tp1_l(0, i, j)
                    + diff_states_t_lp1(rnn.n_states, i, j);
            float dCt = diff_states_tp1_l(1, i, j)
                    + one_m_square(tanhCt) * ws_gates(i, 3, j) * dHt;

            float dG1 = c_states_tm1_l(i, j) * dCt
                    * x_m_square(ws_gates(i, 1, j));
            float dG0 = ws_gates(i, 2, j) * dCt * x_m_square(ws_gates(i, 0, j));
            float dG3 = tanhCt * dHt * x_m_square(ws_gates(i, 3, j));
            float dG2
                    = ws_gates(i, 0, j) * dCt * one_m_square(ws_gates(i, 2, j));

            diff_states_t_l(1, i, j) = dCt * ws_gates(i, 1, j);

            ws_gates(i, 0, j) = dG0;
            ws_gates(i, 1, j) = dG1;
            ws_gates(i, 2, j) = dG2;
            ws_gates(i, 3, j) = dG3;
        }
    });
}

}
}
}
