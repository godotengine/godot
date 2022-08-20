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

#ifndef RNN_UTILS_HPP
#define RNN_UTILS_HPP

#include "mkldnn.h"

#include "cpu_rnn_pd.hpp"


#define rnn_elemwise_sig(f)                                               \
    void f(const rnn_utils::rnn_conf_t &rnn, acc_data_t *ws_gates_,   \
            src_data_t *states_t_l_, float *c_states_t_l_,            \
            src_data_t *states_tm1_l_, float *c_states_tm1_l_,        \
            float *diff_states_t_l_, float *diff_states_t_lp1_,       \
            float *diff_states_tp1_l_, float *bias_, float *ws_grid_, \
            float *ws_cell_) const

#define rnn_cell_execution_sig(f)                                             \
    void f(const rnn_utils::rnn_conf_t &rnn, src_data_t *states_t_l_,     \
            float *c_states_t_l_, float *diff_states_t_l_,                \
            weights_data_t **w_layer_, weights_data_t **w_iter_,          \
            float **bias_, src_data_t *states_t_lm1_,                     \
            src_data_t *states_tm1_l_, float *c_states_tm1_l_,            \
            float *diff_states_t_lp1_, float *diff_states_tp1_l_,         \
            float *diff_w_layer_, float *diff_w_iter_, float *diff_bias_, \
            acc_data_t *ws_gates_, float *ws_grid_, float *ws_cell_) const

#define rnn_grid_execution_sig(f)                                                 \
    void f(const rnn_utils::rnn_conf_t &rnn, weights_data_t **weights_layer_, \
            weights_data_t **weights_states_, float **bias_,                  \
            src_data_t *ws_states_, float *ws_c_states_,                      \
            float *ws_diff_states_, acc_data_t *ws_gates_, float *ws_cell_,   \
            float *ws_grid_, float *diff_weights_layer_,                      \
            float *diff_weights_iter_, float *diff_bias_) const

#define rnn_gemm_sig(f)                                                     \
    void f(const char transA, const char transB, int m, int n, int k,   \
            const float alpha, const weights_data_t *a_, const int ldA, \
            const src_data_t *b_, const int ldB, const float beta,      \
            acc_data_t *c_, const int ldC) const

#define rnn_bias_prepare_sig(f)                                                  \
    void f(const rnn_utils::rnn_conf_t &rnn, float **bias_, const float *b_, \
            float *scratch_bias_) const

#define rnn_bias_finalize_sig(f)                                       \
    void f(const rnn_utils::rnn_conf_t &rnn, float *scratch_bias_, \
            const float *w_iter_comp, const float *w_layer_comp) const

#define rnn_weights_assign_sig(f)                                                \
    void f(const rnn_utils::rnn_conf_t &rnn, const memory_desc_t *md, int nld,   \
            int ld, int OC_size, int IC_size, const int n_parts,             \
            const int *gates_per_part, const size_t *part_weights_pack_size, \
            weights_data_t **weights_, const weights_data_t *w_,             \
            float **bias_, const float *b_, float *scratch_bias_) const


namespace mkldnn {
namespace impl {
namespace cpu {

namespace rnn_utils {

using namespace mkldnn::impl::utils;

enum execution_direction_t {
    l2r,
    r2l,
    bi_concat,
    bi_sum,
};

enum data_type_conf_t {
    all_f32,
    u8u8u8f32,
    f32u8f32f32,
    u8u8u8u8,
    f32u8f32u8
};

struct rnn_conf_t {
    execution_direction_t exec_dir;
    data_type_conf_t dt_conf;
    int n_layer, n_iter, n_dir, n_gates, n_states;
    int mb;
    int slc, sic, dic, dlc;
    int gates_ld, gates_nld, gates_ws_ld;
    int n_parts_weights_layer, parts_weights_layer[MKLDNN_RNN_MAX_N_PARTS];
    int n_parts_weights_iter, parts_weights_iter[MKLDNN_RNN_MAX_N_PARTS];
    int n_bias, n_parts_bias, parts_bias[MKLDNN_RNN_MAX_N_PARTS];
    size_t part_weights_iter_pack_size[MKLDNN_RNN_MAX_N_PARTS],
            part_weights_layer_pack_size[MKLDNN_RNN_MAX_N_PARTS];
    bool weights_layer_is_packed, weights_iter_is_packed;
    /* Size of packed data in bytes */
    size_t weights_layer_comp_offset, weights_layer_pack_size,
        weights_iter_comp_offset, weights_iter_pack_size;

    bool copy_bias;
    int weights_layer_ld, weights_layer_nld;
    int diff_weights_layer_ld, diff_weights_layer_nld;
    int weights_iter_ld, weights_iter_nld;
    int diff_weights_iter_ld, diff_weights_iter_nld;
    int states_nld, states_ws_ld;
    int weights_iter_compensation_size, weights_layer_compensation_size;
    bool is_fwd, is_training, is_lbr;
    bool use_workspace;

    /* Size of workspace for each tensor in bytes */
    size_t ws_gates_size, ws_states_size, ws_c_states_size, ws_diff_states_size,
            ws_cell_comp_size, ws_grid_comp_size, ws_per_cell, ws_bias_size;
    bool merge_gemm_iter, merge_gemm_layer, use_jit_gemm, use_layer_packed_gemm,
        use_iter_packed_gemm;
};

bool is_ldigo(const memory_desc_wrapper &md);
bool is_ldgoi(const memory_desc_wrapper &md);

int get_good_ld(int dim, int sizeof_dt);

void init_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d);

void set_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d);

void set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_h_state_offset, size_t &ws_c_state_offset,
        size_t &ws_diff_states_offset, size_t &ws_grid_comp_offset,
        size_t &ws_cell_comp_offset, size_t &ws_bias_offset,
        size_t &scratchpad_size, size_t &workspace_size);

void get_scratchpad_and_workspace_sizes(const rnn_conf_t &rnn,
        size_t &scratchpad_size, size_t &workspace_size);
status_t set_expected_desc(
        rnn_conf_t &rnn, memory_desc_t &weights_md, bool is_iter);
status_t set_good_strides(memory_desc_t &weights_md, format_tag_t tag);

template <typename T>
struct ws_gates_aoc {
    ws_gates_aoc(const rnn_conf_t &rnn, T *data)
        : gates_(data, rnn.gates_nld, rnn.gates_ws_ld), DIC_(rnn.dic) {}
    T &operator()(int batch, int gate, int dic) {
        return gates_(batch, gate * DIC_ + dic);
    }

private:
    mkldnn::impl::utils::array_offset_calculator<T, 2> gates_;
    int DIC_;
};
using ws_gates_aoc_t = ws_gates_aoc<float>;
using ws_gates_aoc_s32_t = ws_gates_aoc<int32_t>;

struct bias_aoc_t {
    bias_aoc_t(const rnn_conf_t &rnn, const float *data)
        : bias_(data, rnn.n_bias, rnn.dic) {}
    const float &operator()(int bias_n, int dic) { return bias_(bias_n, dic); }

private:
    mkldnn::impl::utils::array_offset_calculator<const float, 2> bias_;
};

template <typename T>
struct ws_states_aoc {
    ws_states_aoc(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.states_nld, rnn.states_ws_ld) {}
    T &operator()(int batch, int dic) { return state_(batch, dic); }

private:
    mkldnn::impl::utils::array_offset_calculator<T, 2> state_;
};
using ws_states_aoc_t = ws_states_aoc<float>;
using ws_states_aoc_u8_t = ws_states_aoc<uint8_t>;

struct ws_diff_states_aoc_t {
    ws_diff_states_aoc_t(const rnn_conf_t &rnn, float *data)
        : diff_states_(data, rnn.n_states + 1, rnn.n_iter + 1, rnn.states_nld,
                  rnn.states_ws_ld) {}
    float &operator()(int state_n, int batch, int dic) {
        return diff_states_(state_n, 0, batch, dic);
    }

private:
    mkldnn::impl::utils::array_offset_calculator<float, 4> diff_states_;
};

struct ws_diff_w_iter_aoc_t {
    ws_diff_w_iter_aoc_t(const rnn_conf_t &rnn, float *data)
        : diff_weights_iter_(
                  data, rnn.diff_weights_iter_nld, rnn.diff_weights_iter_ld)
        , DIC_(rnn.dic) {}
    float &operator()(int sic, int gate, int dic) {
        return diff_weights_iter_(sic, gate * DIC_ + dic);
    }

private:
    mkldnn::impl::utils::array_offset_calculator<float, 2> diff_weights_iter_;
    int DIC_;
};
}
}
}
}
#endif
