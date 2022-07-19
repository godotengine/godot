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
status_t pooling_desc_init(pooling_desc_t *pool_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t kernel, const dims_t padding_l,
        const dims_t padding_r, padding_kind_t padding_kind) {
    bool args_ok = true
        && !any_null(pool_desc, src_desc, dst_desc, strides, kernel, padding_l)
        && one_of(alg_kind, pooling_max,
                pooling_avg_include_padding,
                pooling_avg_exclude_padding)
        && one_of(padding_kind, padding_kind::padding_zero);
    if (!args_ok) return invalid_arguments;

    if (padding_r == nullptr) padding_r = padding_l;

    auto pd = pooling_desc_t();
    pd.primitive_kind = primitive_kind::pooling;
    pd.prop_kind = prop_kind;
    pd.alg_kind = alg_kind;
    pd.src_desc.ndims = src_desc->ndims;

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);

    pd.diff_src_desc = pd.src_desc = zero_md();
    pd.diff_dst_desc = pd.dst_desc = zero_md();

    (is_fwd ? pd.src_desc : pd.diff_src_desc) = *src_desc;
    (is_fwd ? pd.dst_desc : pd.diff_dst_desc) = *dst_desc;

    int sp_dims = src_desc->ndims - 2;
    utils::array_copy(pd.strides, strides, sp_dims);
    utils::array_copy(pd.kernel, kernel, sp_dims);
    utils::array_copy(pd.padding[0], padding_l, sp_dims);
    utils::array_copy(pd.padding[1], padding_r, sp_dims);

    pd.padding_kind = padding_kind;
    if (one_of(alg_kind, pooling_max, pooling_avg_include_padding,
                pooling_avg_exclude_padding)) {
        pd.accum_data_type = types::default_accum_data_type(
                src_desc->data_type, dst_desc->data_type);
    } else {
        pd.accum_data_type = dst_desc->data_type;
    }

    bool consistency = true
        && utils::one_of(src_desc->ndims, 4, 5)
        && utils::one_of(dst_desc->ndims, 4, 5)
        && src_desc->dims[0] == dst_desc->dims[0]
        && src_desc->dims[1] == dst_desc->dims[1];
    for (int i = 2; i < src_desc->ndims; ++i)
        consistency = consistency && (
                (src_desc->dims[i] - kernel[i - 2] + padding_l[i - 2]
                 + padding_r[i - 2]) / strides[i - 2] + 1
                == dst_desc->dims[i]);
    if (!consistency) return invalid_arguments;

    *pool_desc = pd;
    return success;
}
}

status_t mkldnn_pooling_forward_desc_init(pooling_desc_t *pool_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t kernel, const dims_t padding_l,
        const dims_t padding_r, padding_kind_t padding_kind) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return pooling_desc_init(pool_desc, prop_kind, alg_kind, src_desc,
            dst_desc, strides, kernel, padding_l, padding_r, padding_kind);
}

status_t mkldnn_pooling_backward_desc_init(pooling_desc_t *pool_desc,
        alg_kind_t alg_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc, const dims_t strides,
        const dims_t kernel, const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind) {
    return pooling_desc_init(pool_desc, prop_kind::backward_data, alg_kind,
            diff_src_desc, diff_dst_desc, strides, kernel, padding_l,
            padding_r, padding_kind);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
