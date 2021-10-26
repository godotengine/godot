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

#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "ref_rnn.hpp"
#include "rnn_utils.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace rnn_utils;
using namespace format_tag;
using namespace rnn_packed_format;
using namespace data_type;

bool rnn_utils::is_ldigo(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked)
        return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[4] == 1
            && str[3] == dims[4] && str[1] == str[2] * dims[2]
            && str[0] == str[1] * dims[1];
};

bool rnn_utils::is_ldgoi(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked)
        return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[2] == 1
            && str[3] == dims[4] * str[4] && str[1] == str[3] * dims[3]
            && str[0] == str[1] * dims[1];
};

void rnn_utils::init_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d) {
    rnn.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    rnn.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    rnn.is_lbr = rd.cell_desc.cell_kind == mkldnn_gru_linear_before_reset;

    switch (rd.direction) {
    case mkldnn_unidirectional_left2right: rnn.exec_dir = l2r; break;
    case mkldnn_unidirectional_right2left: rnn.exec_dir = r2l; break;
    case mkldnn_bidirectional_concat: rnn.exec_dir = bi_concat; break;
    case mkldnn_bidirectional_sum: rnn.exec_dir = bi_sum; break;
    default: break;
    }

    if (everyone_is(f32, src_layer_d.data_type(), dst_layer_d.data_type(),
                weights_layer_d.data_type()))
        rnn.dt_conf = all_f32;
    else if (dst_layer_d.data_type() == u8) {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            rnn.dt_conf = u8u8u8u8;
        else
            rnn.dt_conf = f32u8f32u8;
    } else {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            rnn.dt_conf = u8u8u8f32;
        else
            rnn.dt_conf = f32u8f32f32;
    }

    rnn.n_layer = weights_layer_d.dims()[0];
    rnn.n_iter = src_layer_d.dims()[0];
    rnn.n_dir = weights_layer_d.dims()[1];
    rnn.n_gates = weights_layer_d.dims()[3];
    rnn.n_states = mkldnn_rnn_cell_get_states_count(&rd.cell_desc);
    rnn.n_bias = rnn.n_gates + rnn.is_lbr;
    rnn.mb = src_layer_d.dims()[1];
    rnn.sic = weights_iter_d.dims()[2];
    rnn.slc = weights_layer_d.dims()[2];
    rnn.dic = weights_layer_d.dims()[4];
    rnn.dlc = dst_layer_d.dims()[2];

    rnn.gates_ld = rnn.dic * rnn.n_gates;
    rnn.gates_nld = rnn.mb;
    rnn.states_nld = rnn.mb;

    /* Set the correct number of weights parts */
    bool is_orig_gru = rd.cell_desc.cell_kind == alg_kind::vanilla_gru;
    rnn.n_parts_weights_layer = 1;
    rnn.parts_weights_layer[0] = rnn.n_gates;
    rnn.parts_weights_layer[1] = 0;

    rnn.n_parts_weights_iter = is_orig_gru ? 2 : 1;
    rnn.parts_weights_iter[0] = is_orig_gru ? 2 : rnn.n_gates;
    rnn.parts_weights_iter[1] = is_orig_gru ? 1 : 0;

    rnn.n_parts_bias = 1;
    rnn.parts_bias[0] = rnn.n_bias;
    rnn.parts_bias[1] = 0;

    /* Decide wich gemm implementation to use: packed/nonpacked jit/cblas
     * and if to mergre gemm across iterations */
    bool is_int8 = rnn.dt_conf != all_f32;
    rnn.merge_gemm_layer = ((rnn.is_fwd && rnn.mb < 128) || !rnn.is_fwd)
            || is_int8;
    bool is_gru = utils::one_of(rd.cell_desc.cell_kind, alg_kind::vanilla_gru,
            alg_kind::gru_linear_before_reset);
    rnn.merge_gemm_iter = !(rnn.is_fwd || is_gru) || is_int8;
    bool is_inference = !rnn.is_training;

    rnn.use_jit_gemm = !mayiuse(avx512_mic)
            && ((is_inference && (rnn.n_layer > 1 || rnn.mb < 100))
                || (rnn.is_training && rnn.dic < 500));

    /* Decide to copy bias */
    rnn.copy_bias = rnn.dt_conf != all_f32;

#if USE_MKL_PACKED_GEMM
    rnn.use_layer_packed_gemm
            = (weights_layer_d.format_kind() == format_kind::any
                      && rnn.slc > 760 && rnn.dic > 760 && is_inference)
            || is_int8; // packed gemm is the only supported option for int8
    rnn.use_iter_packed_gemm
            = (weights_iter_d.format_kind() == format_kind::any && rnn.sic > 760
                      && rnn.dic > 760 && is_inference)
            || is_int8;
#else
    rnn.use_layer_packed_gemm = false;
    rnn.use_iter_packed_gemm = false;
#endif

    /* Set packed gemm sizes */
    if (rnn.use_layer_packed_gemm) {
        rnn.weights_layer_pack_size = 0;
        for (int p = 0; p < rnn.n_parts_weights_layer; p++) {
            int m_p = rnn.is_fwd
                ? (rnn.parts_weights_layer[p] * rnn.dic)
                : rnn.slc;
            int k_p = rnn.is_fwd
                ? rnn.slc
                : (rnn.parts_weights_layer[p] * rnn.dic);
            int n_p = rnn.merge_gemm_layer ? rnn.mb * rnn.n_iter : rnn.mb;

#if USE_MKL_PACKED_GEMM
            if (rnn.dt_conf == all_f32)
                rnn.part_weights_layer_pack_size[p] = cblas_sgemm_pack_get_size(
                        CblasAMatrix, m_p, n_p, k_p);
            else
                rnn.part_weights_layer_pack_size[p]
                        = cblas_gemm_s8u8s32_pack_get_size(
                                CblasAMatrix, m_p, n_p, k_p);
#else
            UNUSED(m_p);
            UNUSED(k_p);
            UNUSED(n_p);
            rnn.part_weights_layer_pack_size[p] = 0;
#endif
            rnn.weights_layer_pack_size += rnn.n_layer * rnn.n_dir
                    * rnn.part_weights_layer_pack_size[p];
        }
        rnn.weights_layer_comp_offset = rnn.weights_layer_pack_size;
        rnn.weights_layer_pack_size += rnn.dt_conf == all_f32 ? 0 : rnn.n_layer
                        * rnn.n_dir * rnn.n_gates * rnn.dlc * sizeof(float);
    }

    if (rnn.use_iter_packed_gemm) {
        rnn.weights_iter_pack_size = 0;
        for (int p = 0; p < rnn.n_parts_weights_iter; p++) {
            int m_p = rnn.is_fwd ? (rnn.parts_weights_iter[p] * rnn.dic) :
                                   rnn.sic;
            int k_p = rnn.is_fwd ? rnn.sic :
                                   (rnn.parts_weights_iter[p] * rnn.dic);
            int n_p = rnn.merge_gemm_iter ? rnn.mb * rnn.n_iter : rnn.mb;

#if USE_MKL_PACKED_GEMM
            if (rnn.dt_conf == all_f32)
                rnn.part_weights_iter_pack_size[p] = cblas_sgemm_pack_get_size(
                        CblasAMatrix, m_p, n_p, k_p);
            else
                rnn.part_weights_iter_pack_size[p]
                        = cblas_gemm_s8u8s32_pack_get_size(
                                CblasAMatrix, m_p, n_p, k_p);
#else
            UNUSED(m_p);
            UNUSED(k_p);
            UNUSED(n_p);
            rnn.part_weights_iter_pack_size[p] = 0;
#endif
            rnn.weights_iter_pack_size += rnn.n_layer * rnn.n_dir
                    * rnn.part_weights_iter_pack_size[p];
        }
        rnn.weights_iter_comp_offset = rnn.weights_iter_pack_size;
        rnn.weights_iter_pack_size += rnn.dt_conf == all_f32 ? 0 : rnn.n_layer
                        * rnn.n_dir * rnn.n_gates * rnn.dic * sizeof(float);
    }

}

void rnn_utils::set_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d) {

    /* Set leading dimensions for input weights arrays depending on input format
     */
    rnn.weights_layer_is_packed
            = weights_layer_d.format_kind() == format_kind::rnn_packed;
    rnn.weights_iter_is_packed
            = weights_iter_d.format_kind() == format_kind::rnn_packed;

    auto set_dims = [&](const memory_desc_wrapper &md, int &ld, int &nld) {
        ld = 0; nld = 0;
        if (md.is_blocking_desc()) {
            if (is_ldigo(md)) {
                ld = (int)md.blocking_desc().strides[2];
                nld = md.dims()[2];
            } else if (is_ldgoi(md)) {
                ld = (int)md.blocking_desc().strides[4];
                nld = md.dims()[3] * md.dims()[4];
            } else
                assert(!"unsupported weights format");
        }
    };
    set_dims(weights_layer_d, rnn.weights_layer_ld, rnn.weights_layer_nld);
    set_dims(weights_iter_d, rnn.weights_iter_ld, rnn.weights_iter_nld);
    if (!rnn.is_fwd) {
        set_dims(diff_weights_layer_d, rnn.diff_weights_layer_ld,
                rnn.diff_weights_layer_nld);
        set_dims(diff_weights_iter_d, rnn.diff_weights_iter_ld,
                rnn.diff_weights_iter_nld);
    }

    int sizeof_states_dt
            = rnn.dt_conf == all_f32 ? sizeof(float) : sizeof(uint8_t);
    rnn.states_ws_ld
            = get_good_ld(nstl::max(rnn.slc, nstl::max(rnn.sic, rnn.dic)),
                sizeof_states_dt);
    rnn.gates_ws_ld = get_good_ld(rnn.gates_ld, sizeof(float));

    /* Set workspace sizes to store:
     * states to copmute a pass
     * diff states to copmute bwd pass (training only)
     * intermediate results from the gates
     */
    rnn.use_workspace = rnn.is_training;
    rnn.ws_states_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.mb * rnn.states_ws_ld * sizeof_states_dt;
    bool is_lstm = rd.cell_desc.cell_kind == mkldnn_vanilla_lstm;
    rnn.ws_c_states_size = is_lstm
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.states_ws_ld * sizeof(float)
            : 0;
    rnn.ws_diff_states_size = rnn.is_training
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1)
                    * (rnn.n_states + 1) * rnn.mb * rnn.states_ws_ld
                    * sizeof(float)
            : (size_t)0;
    rnn.ws_gates_size = (size_t)rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.mb
            * rnn.gates_ws_ld * sizeof(float);

    /* set other sizes */
    rnn.ws_per_cell = (size_t)rnn.is_lbr * rnn.mb * rnn.dic * sizeof(float);
    rnn.ws_cell_comp_size
            = rnn.is_lbr || rnn.dt_conf != all_f32
                ? (size_t) rnn.gates_nld * rnn.gates_ws_ld * sizeof(float)
                : 0;
    rnn.ws_grid_comp_size = (size_t)rnn.is_lbr * rnn.is_training * rnn.n_layer
            * rnn.n_dir * rnn.n_iter * rnn.ws_per_cell * sizeof(float);
    rnn.ws_bias_size = (size_t)rnn.n_layer * rnn.n_dir * rnn.n_bias * rnn.dic
            * sizeof(float);
}

int rnn_utils::get_good_ld(int dim, int sizeof_dt) {
    // we want matrices leading dimentions to be 64-byte aligned,
    // and not divisible by 256 to avoid 4K aliasing effects
    int ld = rnd_up(dim, 64 / sizeof_dt);
    return (ld % 256 == 0) ? ld + 64 / sizeof_dt : ld;
}

void rnn_utils::set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_states_offset, size_t &ws_c_states_offset,
        size_t &ws_diff_states_offset, size_t &ws_grid_comp_offset,
        size_t &ws_cell_comp_offset, size_t &ws_bias_offset,
        size_t &scratchpad_size, size_t &workspace_size) {

    const size_t page_size = 4096; // 2097152;
    size_t current_offset;
    /* Mandatory workspaces: go to workspace if use_workspace, scratchpad
     * otherwise */
    current_offset = 0; // assumes the workspace base pointer is page aligned
    ws_gates_offset = current_offset;
    current_offset += rnn.ws_gates_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_states_offset = current_offset;
    current_offset += rnn.ws_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_c_states_offset = current_offset;
    current_offset += rnn.ws_c_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_diff_states_offset = current_offset;
    current_offset += rnn.ws_diff_states_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_grid_comp_offset = current_offset;
    current_offset += rnn.ws_grid_comp_size;

    current_offset = utils::rnd_up(current_offset, page_size);
    ws_cell_comp_offset = current_offset;
    current_offset += rnn.ws_cell_comp_size;

    workspace_size = rnn.use_workspace ? current_offset : 0;

    /* Optional scratchpads */
    // Assumes the scratchpad base pointer is page aligned.
    // If use_workspace, the following goes to scratchpad alone,
    // otherwise, all goes to scratchpad and continue incrementing offset
    current_offset = rnn.use_workspace ? 0 : current_offset;

    if (rnn.copy_bias) {
        current_offset = utils::rnd_up(current_offset, page_size);
        ws_bias_offset = current_offset;
        current_offset += rnn.ws_bias_size;
    }

    scratchpad_size = current_offset;
}

void rnn_utils::get_scratchpad_and_workspace_sizes(const rnn_conf_t &rnn,
        size_t &scratchpad_size, size_t &workspace_size) {
    size_t ws_gates_offset, ws_states_offset, ws_c_states_offset,
            ws_diff_states_offset, ws_grid_comp_offset, ws_cell_comp_offset,
            ws_bias_offset;
    set_offsets(rnn, ws_gates_offset, ws_states_offset, ws_diff_states_offset,
            ws_c_states_offset, ws_grid_comp_offset, ws_cell_comp_offset,
            ws_bias_offset, scratchpad_size, workspace_size);
}

status_t rnn_utils::set_good_strides(
        memory_desc_t &weights_md, format_tag_t tag) {
    auto &strides = weights_md.format_desc.blocking.strides;
    auto dims = weights_md.dims;

    if (tag == ldigo) {
        strides[2] = rnn_utils::get_good_ld((int)strides[2],
                (int)types::data_type_size(weights_md.data_type));
        strides[1] = dims[2] * strides[2];
        strides[0] = dims[1] * strides[1];
    } else if (tag == ldgoi) {
        strides[4] = rnn_utils::get_good_ld((int)strides[4],
                (int)types::data_type_size(weights_md.data_type));
        strides[3] = dims[4] * strides[4];
        strides[1] = dims[3] * strides[3];
        strides[0] = dims[1] * strides[1];
    } else
        return status::unimplemented;

    return status::success;
}

status_t rnn_utils::set_expected_desc(rnn_conf_t &rnn,
        memory_desc_t &weights_md, bool is_iter) {
    using namespace format_tag;
    bool use_packed_gemm = is_iter
        ? rnn.use_iter_packed_gemm
        : rnn.use_layer_packed_gemm;
    if (use_packed_gemm) {
        weights_md.format_kind = format_kind::rnn_packed;
        rnn_packed_desc_t &rnn_pdata = weights_md.format_desc.rnn_packed_desc;
        rnn_pdata.format = rnn.is_fwd ? mkldnn_ldigo_p : mkldnn_ldgoi_p;
        if (is_iter) {
            rnn_pdata.n = rnn.mb;
            rnn_pdata.n_parts = rnn.n_parts_weights_iter;
            array_copy(rnn_pdata.parts, rnn.parts_weights_iter,
                    MKLDNN_RNN_MAX_N_PARTS);
            array_copy(rnn_pdata.part_pack_size,
                    rnn.part_weights_iter_pack_size, MKLDNN_RNN_MAX_N_PARTS);
            rnn_pdata.offset_compensation = rnn.weights_iter_comp_offset;
            rnn_pdata.size = rnn.weights_iter_pack_size;
        } else {
            rnn_pdata.n = rnn.merge_gemm_layer ? rnn.n_iter * rnn.mb : rnn.mb;
            rnn_pdata.n_parts = rnn.n_parts_weights_layer;
            array_copy(rnn_pdata.parts, rnn.parts_weights_layer,
                    MKLDNN_RNN_MAX_N_PARTS);
            array_copy(rnn_pdata.part_pack_size,
                    rnn.part_weights_layer_pack_size, MKLDNN_RNN_MAX_N_PARTS);
            rnn_pdata.offset_compensation = rnn.weights_layer_comp_offset;
            rnn_pdata.size = rnn.weights_layer_pack_size;
        }
    } else {
        CHECK(memory_desc_init_by_tag(weights_md, rnn.is_fwd ? ldigo : ldgoi));
        // Adjust strides for good leading dimension in GEMM
        CHECK(set_good_strides(weights_md, rnn.is_fwd ? ldigo : ldgoi));
    }
    return status::success;
}

}
}
}
