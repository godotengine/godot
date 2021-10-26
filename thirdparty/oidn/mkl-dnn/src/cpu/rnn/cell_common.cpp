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
 * Common for RNN and LSTM cell execution
 */
#include "ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {
using namespace rnn_utils;

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
rnn_cell_execution_sig(
        (_ref_rnn_common_t<aprop, src_type, weights_type>::cell_execution)) {
    if (!rnn.merge_gemm_layer) {
        (this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dic, rnn.mb,
                rnn.slc, 1.0, w_layer_[0], rnn.weights_layer_ld,
                states_t_lm1_, rnn.states_ws_ld, 0.0, ws_gates_,
                rnn.gates_ws_ld);
    }
    (this->*gemm_iter_func)('N', 'N', rnn.n_gates * rnn.dic, rnn.mb, rnn.sic,
            1.0, w_iter_[0], rnn.weights_iter_ld, states_tm1_l_,
            rnn.states_ws_ld, 1.0, ws_gates_, rnn.gates_ws_ld);

    if (rnn_postgemm_ != nullptr)
        rnn_postgemm_->execute<src_data_t, acc_data_t>(rnn, ws_gates_, states_t_l_, c_states_t_l_,
            states_tm1_l_, c_states_tm1_l_, diff_states_t_l_,
            diff_states_t_lp1_, diff_states_tp1_l_, bias_[0], ws_grid_,
            ws_cell_);
    else
        (this->*elemwise_func)(rnn, ws_gates_, states_t_l_, c_states_t_l_,
                states_tm1_l_, c_states_tm1_l_, diff_states_t_l_,
                diff_states_t_lp1_, diff_states_tp1_l_, bias_[0], ws_grid_,
                ws_cell_);
}
template rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution);
template rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution);

template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution) {
    ws_diff_states_aoc_t diff_states_t_l(rnn, diff_states_t_l_);
    (this->*elemwise_func)(rnn, ws_gates_, states_t_l_, c_states_t_l_,
            states_tm1_l_, c_states_tm1_l_, diff_states_t_l_,
            diff_states_t_lp1_, diff_states_tp1_l_, bias_[0], ws_grid_,
            ws_cell_);

    /// bwd by data on the cell
    (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb, rnn.n_gates * rnn.dic,
            1.0, w_iter_[0], rnn.weights_iter_ld, ws_gates_, rnn.gates_ws_ld,
            0.0, diff_states_t_l_, rnn.states_ws_ld);

    if (!rnn.merge_gemm_layer) {
        (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dic, 1.0, w_layer_[0],
                rnn.weights_layer_ld, ws_gates_, rnn.gates_ws_ld, 0.0,
                &diff_states_t_l(rnn.n_states, 0, 0), rnn.states_ws_ld);

        /// bwd by weights on the cell
        gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.slc, rnn.mb, 1.0, ws_gates_,
                rnn.gates_ws_ld, states_t_lm1_, rnn.states_ws_ld, 1.0,
                diff_w_layer_, rnn.diff_weights_layer_ld);
    }

    if (!rnn.merge_gemm_iter)
        gemm('N', 'T', rnn.n_gates * rnn.dic, rnn.sic, rnn.mb, 1.0, ws_gates_,
                rnn.gates_ws_ld, states_tm1_l_, rnn.states_ws_ld, 1.0,
                diff_w_iter_, rnn.diff_weights_iter_ld);

    /// bwd by bias we just accumulate diffs from the gates
    gates_reduction(rnn, ws_gates_, diff_bias_);
}

}
}
}
