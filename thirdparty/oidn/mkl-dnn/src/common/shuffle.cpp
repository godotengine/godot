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

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::types;

namespace {
status_t shuffle_desc_init(shuffle_desc_t *shuffle_desc, prop_kind_t prop_kind,
        const memory_desc_t *data_desc, int axis, dim_t group_size) {
    bool args_ok = true
        && !any_null(shuffle_desc, data_desc)
        && one_of(prop_kind, forward_training, forward_inference,
                  backward, backward_data)
        && axis >= 0 && axis < data_desc->ndims
        && group_size > 0 && group_size <= data_desc->dims[axis];
    if (!args_ok) return invalid_arguments;

    auto sd = shuffle_desc_t();
    sd.primitive_kind = primitive_kind::shuffle;
    sd.prop_kind = prop_kind;
    sd.data_desc = *data_desc;
    sd.axis = axis;
    sd.group_size = group_size;

    bool consistency = true
        && sd.data_desc.dims[axis] % sd.group_size == 0;
    if (!consistency) return invalid_arguments;

    *shuffle_desc = sd;
    return success;
}
}

status_t mkldnn_shuffle_forward_desc_init(shuffle_desc_t *shuffle_desc,
        prop_kind_t prop_kind, const memory_desc_t *data_desc, int axis,
        dim_t group_size) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return shuffle_desc_init(shuffle_desc, prop_kind, data_desc, axis,
        group_size);
}

status_t mkldnn_shuffle_backward_desc_init(shuffle_desc_t *shuffle_desc,
        const memory_desc_t *diff_data_desc, int axis, dim_t group_size) {
    return shuffle_desc_init(shuffle_desc, backward_data, diff_data_desc, axis,
        group_size);
}

// vim: et ts=5 sw=4 cindent cino^=l0,\:0,N-s
