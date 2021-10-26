/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;
using namespace mkldnn::impl::types;

namespace {
status_t lrn_desc_init(lrn_desc_t *lrn_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *data_desc, const memory_desc_t *diff_data_desc,
        dim_t local_size, float alpha, float beta, float k) {
    bool args_ok = true
        && !any_null(lrn_desc, data_desc)
        && one_of(alg_kind, lrn_within_channel, lrn_across_channels)
        && one_of(prop_kind, forward_training, forward_inference, backward_data)
        && IMPLICATION(prop_kind == backward_data, diff_data_desc != nullptr);
    if (!args_ok) return invalid_arguments;

    auto ld = lrn_desc_t();
    ld.primitive_kind = primitive_kind::lrn;
    ld.prop_kind = prop_kind;
    ld.alg_kind = alg_kind;

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);

    ld.data_desc = *data_desc;
    if (!is_fwd)
        ld.diff_data_desc = *diff_data_desc;
    else
        ld.diff_data_desc = zero_md();
    ld.local_size = local_size;
    ld.lrn_alpha = alpha;
    ld.lrn_beta = beta;
    ld.lrn_k = k;

    bool consistency = true
        && ld.data_desc.ndims == 4;
    if (ld.prop_kind == backward_data)
        consistency = consistency
            && ld.diff_data_desc.ndims == 4
            && array_cmp(ld.diff_data_desc.dims, ld.data_desc.dims, 4);
    if (!consistency) return invalid_arguments;

    *lrn_desc = ld;
    return success;
}
}

status_t mkldnn_lrn_forward_desc_init(lrn_desc_t *lrn_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *data_desc, dim_t local_size, float alpha,
        float beta, float k) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return lrn_desc_init(lrn_desc, prop_kind, alg_kind, data_desc, nullptr,
            local_size, alpha, beta, k);
}

status_t mkldnn_lrn_backward_desc_init(lrn_desc_t *lrn_desc,
        alg_kind_t alg_kind, const memory_desc_t *data_desc,
        const memory_desc_t *diff_data_desc, dim_t local_size, float alpha,
        float beta, float k) {
    return lrn_desc_init(lrn_desc, backward_data, alg_kind, data_desc,
            diff_data_desc, local_size, alpha, beta, k);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
