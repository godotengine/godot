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

#ifndef CPU_RNN_PD_HPP
#define CPU_RNN_PD_HPP

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "rnn_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "rnn_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_rnn_fwd_pd_t : public rnn_fwd_pd_t {
    using rnn_fwd_pd_t::rnn_fwd_pd_t;

protected:
    status_t set_default_params() {
        using namespace format_tag;
        if (src_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
        if (dst_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

        // Optional parameters
        if (with_src_iter() && src_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_iter_md_, ldsnc));
        if (with_bias() && bias_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
        if (with_dst_iter() && dst_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_iter_md_, ldsnc));

        return status::success;
    }

    status_t check_layout_consistency() {
        using namespace format_tag;
        using namespace data_type;
        using namespace types;

        auto is_blocked = [&](memory_desc_t md, int ndims) {
            return md.format_kind == format_kind::blocked && md.ndims == ndims;
        };

        bool ok = true;
        ok = ok && is_blocked(src_layer_md_, 3)
                && is_blocked(dst_layer_md_, 3);
        ok = ok && IMPLICATION(!is_zero_md(&src_iter_md_),
                           is_blocked(src_iter_md_, 5))
                && IMPLICATION(!is_zero_md(&dst_iter_md_),
                           is_blocked(dst_iter_md_, 5));

        if (weights_layer_md_.format_kind == format_kind::rnn_packed)
            ok = ok && (weights_layer_md_.format_desc.rnn_packed_desc.format
                               == mkldnn_ldigo_p);
        else
            ok = ok && rnn_utils::is_ldigo(&weights_layer_md_);

        if (weights_iter_md_.format_kind == format_kind::rnn_packed)
            ok = ok && (weights_iter_md_.format_desc.rnn_packed_desc.format
                               == mkldnn_ldigo_p);
        else
            ok = ok && rnn_utils::is_ldigo(&weights_iter_md_);

        ok = ok && IMPLICATION(!is_zero_md(&bias_md_),
                           memory_desc_matches_tag(bias_md_, ldgo));

        /* Int8 is supported only for packed weights */
        data_type_t weights_iter_dt = weights_iter_md_.data_type;
        data_type_t weights_layer_dt = weights_layer_md_.data_type;
        ok = ok && IMPLICATION(
                           weights_iter_dt == s8, weights_iter_md_.format_kind
                                   == format_kind::rnn_packed);
        ok = ok && IMPLICATION(
                           weights_layer_dt == s8, weights_layer_md_.format_kind
                                   == format_kind::rnn_packed);

        return ok ? status::success : status::unimplemented;
    }
};

struct cpu_rnn_bwd_pd_t : public rnn_bwd_pd_t {
    using rnn_bwd_pd_t::rnn_bwd_pd_t;

protected:
    status_t set_default_params() {
        using namespace format_tag;
        if (src_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
        if (dst_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

        if (diff_src_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_src_layer_md_, tnc));
        if (diff_weights_layer_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(diff_weights_layer_md_, ldigo));
            CHECK(rnn_utils::set_good_strides(diff_weights_layer_md_, ldigo));
        }
        if (diff_weights_iter_md_.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(diff_weights_iter_md_, ldigo));
            CHECK(rnn_utils::set_good_strides(diff_weights_iter_md_, ldigo));
        }
        if (diff_dst_layer_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_dst_layer_md_, tnc));

        // Optional parameters
        if (with_src_iter() && src_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(src_iter_md_, ldsnc));
        if (with_bias() && bias_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
        if (with_dst_iter() && dst_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_iter_md_, ldsnc));

        if (with_src_iter() && diff_src_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_src_iter_md_, ldsnc));
        if (with_bias() && diff_bias_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_bias_md_, ldgo));
        if (with_dst_iter() && diff_dst_iter_md_.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_dst_iter_md_, ldsnc));

        return status::success;
    }

    status_t check_layout_consistency() {
        using namespace format_tag;
        using namespace types;

        auto is_blocked = [&](memory_desc_t md, int ndims) {
            return md.format_kind == format_kind::blocked && md.ndims == ndims;
        };

        bool ok = true;
        ok = ok && is_blocked(src_layer_md_, 3)
                && is_blocked(dst_layer_md_, 3);
        ok = ok && IMPLICATION(!is_zero_md(&src_iter_md_),
                           is_blocked(src_iter_md_, 5))
                && IMPLICATION(!is_zero_md(&dst_iter_md_),
                           is_blocked(dst_iter_md_, 5));

        if (weights_layer_md_.format_kind == format_kind::rnn_packed)
            ok = ok && (weights_layer_md_.format_desc.rnn_packed_desc.format
                               == mkldnn_ldgoi_p);
        else
            ok = ok && rnn_utils::is_ldgoi(&weights_layer_md_);

        if (weights_iter_md_.format_kind == format_kind::rnn_packed)
            ok = ok && (weights_iter_md_.format_desc.rnn_packed_desc.format
                               == mkldnn_ldgoi_p);
        else
            ok = ok && rnn_utils::is_ldgoi(&weights_iter_md_);

        ok = ok && IMPLICATION(!is_zero_md(&bias_md_),
                           memory_desc_matches_tag(bias_md_, ldgo));

        ok = ok && is_blocked(diff_src_layer_md_, 3)
                && is_blocked(diff_dst_layer_md_, 3);
        ok = ok && IMPLICATION(!is_zero_md(&diff_src_iter_md_),
                           is_blocked(diff_src_iter_md_, 5))
                && IMPLICATION(!is_zero_md(&diff_dst_iter_md_),
                           is_blocked(diff_dst_iter_md_, 5));

        ok = ok && rnn_utils::is_ldigo(&diff_weights_layer_md_)
                && rnn_utils::is_ldigo(&diff_weights_iter_md_);
        ok = ok && IMPLICATION(!is_zero_md(&diff_bias_md_),
                           memory_desc_matches_tag(diff_bias_md_, ldgo));

        return ok ? status::success : status::unimplemented;
    }
};
}
}
}

#endif
