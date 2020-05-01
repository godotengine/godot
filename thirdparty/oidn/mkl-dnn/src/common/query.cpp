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
#include "engine.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;

status_t mkldnn_primitive_desc_query(const primitive_desc_t *primitive_desc,
        query_t what, int index, void *result) {
    if (any_null(primitive_desc, result))
        return invalid_arguments;

    return primitive_desc->query(what, index, result);
}

const memory_desc_t *mkldnn_primitive_desc_query_md(
        const primitive_desc_t *primitive_desc, query_t what, int index) {
    const memory_desc_t *res_md = nullptr;
    bool args_ok = true
        && primitive_desc != nullptr
        && (what & query::some_md) == query::some_md
        && what != query::some_md
        && mkldnn_primitive_desc_query(primitive_desc,
                what, index, &res_md) == success;
    return args_ok ? res_md : nullptr;
}

int mkldnn_primitive_desc_query_s32(const primitive_desc_t *primitive_desc,
        query_t what, int index) {
    int res_s32;
    bool args_ok = primitive_desc != nullptr
        && one_of(what, query::num_of_inputs_s32, query::num_of_outputs_s32)
        && mkldnn_primitive_desc_query(primitive_desc, what, index, &res_s32)
                == success;
    return args_ok ? res_s32 : 0;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
