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
#include <assert.h>

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
status_t deconv_desc_init(deconvolution_desc_t *deconv_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r, padding_kind_t padding_kind) {
    bool args_ok = true
            && !any_null(deconv_desc, src_desc, weights_desc, dst_desc, strides,
                           padding_l)
            && one_of(alg_kind, deconvolution_direct, deconvolution_winograd)
            && one_of(padding_kind, padding_kind::padding_zero);
    if (!args_ok)
        return invalid_arguments;

    if (padding_r == nullptr)
        padding_r = padding_l;

    auto dd = deconvolution_desc_t();
    dd.primitive_kind = primitive_kind::deconvolution;
    dd.prop_kind = prop_kind;
    dd.alg_kind = alg_kind;

    dd.diff_src_desc = dd.src_desc = zero_md();
    dd.diff_dst_desc = dd.dst_desc = zero_md();
    dd.diff_weights_desc = dd.weights_desc = zero_md();
    dd.diff_bias_desc = dd.bias_desc = zero_md();

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    const bool with_bias
            = bias_desc && bias_desc->format_kind != format_kind::undef;
    const bool with_groups = weights_desc->ndims == src_desc->ndims + 1;

    (prop_kind == backward_data ? dd.diff_src_desc : dd.src_desc) = *src_desc;
    (is_fwd ? dd.dst_desc : dd.diff_dst_desc) = *dst_desc;
    (prop_kind == backward_weights ? dd.diff_weights_desc : dd.weights_desc)
            = *weights_desc;
    if (with_bias)
        (prop_kind == backward_weights ? dd.diff_bias_desc : dd.bias_desc)
                = *bias_desc;

    int sp_dims = src_desc->ndims - 2;
    utils::array_copy(dd.strides, strides, sp_dims);
    utils::array_copy(dd.padding[0], padding_l, sp_dims);
    utils::array_copy(dd.padding[1], padding_r, sp_dims);
    if (dilates)
        utils::array_copy(dd.dilates, dilates, sp_dims);
    else
        utils::array_set(dd.dilates, 0, sp_dims);

    dd.padding_kind = padding_kind;
    dd.accum_data_type = types::default_accum_data_type(src_desc->data_type,
            weights_desc->data_type, dst_desc->data_type, prop_kind);

    const int g = with_groups ? weights_desc->dims[0] : 1;
    bool consistency = true
            && src_desc->ndims == dst_desc->ndims
            && utils::one_of(src_desc->ndims, 3, 4, 5)
            && utils::one_of(weights_desc->ndims, src_desc->ndims,
                    src_desc->ndims + 1)
            && (with_bias ? bias_desc->ndims == 1 : true)
            && (with_bias ? bias_desc->dims[0] == dst_desc->dims[1] : true)
            && src_desc->dims[0] == dst_desc->dims[0]
            && src_desc->dims[1] == g * weights_desc->dims[with_groups + 1]
            && dst_desc->dims[1] == g * weights_desc->dims[with_groups + 0];
    for (int i = 2; i < src_desc->ndims; ++i) {
        int src = src_desc->dims[i];
        int ker = weights_desc->dims[with_groups + i];
        int dil = dd.dilates[i - 2];
        int pad = padding_l[i - 2] + padding_r[i - 2];
        int str = strides[i - 2];
        int dst = dst_desc->dims[i];
        int ker_range = 1 + (ker - 1) * (dil + 1);

        consistency
                = consistency && (dst - ker_range + pad) / str + 1 == src;
    }
    if (!consistency)
        return invalid_arguments;

    *deconv_desc = dd;
    return success;
}
}

status_t mkldnn_deconvolution_forward_desc_init(
        deconvolution_desc_t *deconv_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_desc, const dims_t strides,
        const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return deconv_desc_init(deconv_desc, prop_kind, alg_kind, src_desc,
            weights_desc, bias_desc, dst_desc, strides, nullptr, padding_l,
            padding_r, padding_kind);
}

status_t mkldnn_dilated_deconvolution_forward_desc_init(
        deconvolution_desc_t *deconv_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_desc, const dims_t strides,
        const dims_t dilates, const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return deconv_desc_init(deconv_desc, prop_kind, alg_kind, src_desc,
            weights_desc, bias_desc, dst_desc, strides, dilates, padding_l,
            padding_r, padding_kind);
}

status_t mkldnn_deconvolution_backward_data_desc_init(
        deconvolution_desc_t *deconv_desc, alg_kind_t alg_kind,
        const memory_desc_t *diff_src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *diff_dst_desc, const dims_t strides,
        const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind) {
    return deconv_desc_init(deconv_desc, backward_data, alg_kind, diff_src_desc,
            weights_desc, nullptr, diff_dst_desc, strides, nullptr, padding_l,
            padding_r, padding_kind);
}

status_t mkldnn_dilated_deconvolution_backward_data_desc_init(
        deconvolution_desc_t *deconv_desc, alg_kind_t alg_kind,
        const memory_desc_t *diff_src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *diff_dst_desc, const dims_t strides,
        const dims_t dilates, const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind) {
    return deconv_desc_init(deconv_desc, backward_data, alg_kind, diff_src_desc,
            weights_desc, nullptr, diff_dst_desc, strides,dilates, padding_l,
            padding_r, padding_kind);
}

status_t mkldnn_deconvolution_backward_weights_desc_init(
        deconvolution_desc_t *deconv_desc, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *diff_weights_desc,
        const memory_desc_t *diff_bias_desc, const memory_desc_t *diff_dst_desc,
        const dims_t strides, const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind) {
    return deconv_desc_init(deconv_desc, backward_weights, alg_kind, src_desc,
            diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, nullptr,
            padding_l, padding_r, padding_kind);
}

status_t mkldnn_dilated_deconvolution_backward_weights_desc_init(
        deconvolution_desc_t *deconv_desc, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *diff_weights_desc,
        const memory_desc_t *diff_bias_desc, const memory_desc_t *diff_dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r, padding_kind_t padding_kind) {
    return deconv_desc_init(deconv_desc, backward_weights, alg_kind, src_desc,
            diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, dilates,
            padding_l, padding_r, padding_kind);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
