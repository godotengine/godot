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

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "primitive_desc.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

status_t primitive_desc_t::query(query_t what, int idx, void *result) const {
    auto safe_ret_md = [&](const memory_desc_t *_) {
        if (_ == nullptr) return not_required;
        *(const memory_desc_t **)result = _;
        return success;
    };

    switch (what) {
        case query::engine: *(engine_t**)result = engine(); break;
        case query::primitive_kind: *(primitive_kind_t*)result = kind(); break;

        case query::scratchpad_engine:
            *(engine_t**)result = scratchpad_engine(); break;

        case query::memory_consumption_s64:
            *(dim_t *)result = scratchpad_size(scratchpad_mode::library); break;

        case query::op_d:
            if (idx != 0 || op_desc() == nullptr) return invalid_arguments;
            *(const_c_op_desc_t *)result
                = static_cast<const_c_op_desc_t>(op_desc()); break;

        case query::src_md: return safe_ret_md(src_md(idx));
        case query::diff_src_md: return safe_ret_md(diff_src_md(idx));
        case query::dst_md: return safe_ret_md(dst_md(idx));
        case query::diff_dst_md: return safe_ret_md(diff_dst_md(idx));
        case query::weights_md: return safe_ret_md(weights_md(idx));
        case query::diff_weights_md: return safe_ret_md(diff_weights_md(idx));
        case query::workspace_md:
            if (idx != 0) return status::invalid_arguments;
            return safe_ret_md(workspace_md(idx));
        case query::scratchpad_md:
            if (idx != 0) return status::invalid_arguments;
            return safe_ret_md(scratchpad_md(idx));

        case query::num_of_inputs_s32: *(int*)result = n_inputs(); break;
        case query::num_of_outputs_s32: *(int*)result = n_outputs(); break;

        case query::impl_info_str: *(const char **)result = name(); break;

        default: return unimplemented;
    }
    return success;
}

status_t mkldnn_primitive_desc_get_attr(const primitive_desc_t *primitive_desc,
        const primitive_attr_t **attr) {
    if (utils::any_null(primitive_desc, attr))
        return invalid_arguments;

    *attr = primitive_desc->attr();
    return success;
}
