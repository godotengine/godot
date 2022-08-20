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
#include <stddef.h>
#include <stdint.h>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::data_type;

namespace {
bool memory_desc_sanity_check(int ndims,const dims_t dims,
        data_type_t data_type, format_kind_t format_kind) {
    if (ndims == 0) return true;

    bool ok = true
        && dims != nullptr
        && 0 < ndims && ndims <= MKLDNN_MAX_NDIMS
        && one_of(data_type, f32, s32, s8, u8)
        && format_kind != format_kind::undef;
    if (!ok) return false;
    for (int d = 0; d < ndims; ++d)
        if (dims[d] < 0) return false;

    return true;
}

bool memory_desc_sanity_check(const memory_desc_t *md) {
    if (md == nullptr) return false;
    return memory_desc_sanity_check(md->ndims, md->dims, md->data_type,
            format_kind::any);
}
}

status_t mkldnn_memory_desc_init_by_tag(memory_desc_t *memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, format_tag_t tag) {
    if (any_null(memory_desc)) return invalid_arguments;
    if (ndims == 0 || tag == format_tag::undef) {
        *memory_desc = types::zero_md();
        return success;
    }

    format_kind_t format_kind = types::format_tag_to_kind(tag);

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc)
        && memory_desc_sanity_check(ndims, dims, data_type, format_kind);
    if (!args_ok) return invalid_arguments;

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind;

    status_t status = success;
    if (tag == format_tag::undef) {
        status = invalid_arguments;
    } else if (tag == format_tag::any) {
        // nop
    } else if (format_kind == format_kind::blocked) {
        status = memory_desc_wrapper::compute_blocking(md, tag);
    } else {
        assert(!"unreachable");
        status = invalid_arguments;
    }

    if (status == success)
        *memory_desc = md;

    return status;
}

status_t mkldnn_memory_desc_init_by_strides(memory_desc_t *memory_desc,
        int ndims, const dims_t dims, data_type_t data_type,
        const dims_t strides) {
    if (any_null(memory_desc)) return invalid_arguments;
    if (ndims == 0) {
        *memory_desc = types::zero_md();
        return success;
    }

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc)
        && memory_desc_sanity_check(ndims, dims, data_type, format_kind::any);
    if (!args_ok) return invalid_arguments;

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::blocked;

    dims_t default_strides = {0};
    if (strides == nullptr) {
        default_strides[md.ndims - 1] = 1;
        for (int d = md.ndims - 2; d >= 0; --d)
            default_strides[d] = default_strides[d + 1] * md.padded_dims[d + 1];
        strides = default_strides;
    } else {
        /* TODO: add sanity check for the provided strides */
    }

    array_copy(md.format_desc.blocking.strides, strides, md.ndims);

    *memory_desc = md;

    return status::success;
}

status_t mkldnn_memory_desc_init_submemory(memory_desc_t *md,
        const memory_desc_t *parent_md, const dims_t dims,
        const dims_t offsets) {
    if (any_null(md, parent_md) || !memory_desc_sanity_check(parent_md))
        return invalid_arguments;

    const memory_desc_wrapper src_d(parent_md);

    for (int d = 0; d < src_d.ndims(); ++d) {
        if (dims[d] < 0 || offsets[d] < 0
                || (offsets[d] + dims[d] > src_d.dims()[d]))
            return invalid_arguments;
    }

    if (src_d.format_kind() != format_kind::blocked)
        return unimplemented;

    dims_t blocks;
    src_d.compute_blocks(blocks);

    memory_desc_t dst_d = *parent_md;
    auto &dst_d_blk = dst_d.format_desc.blocking;

    /* TODO: put this into memory_desc_wrapper */
    for (int d = 0; d < src_d.ndims(); ++d) {
        /* very limited functionality for now */
        const bool ok = true
            && offsets[d] % blocks[d] == 0 /* [r1] */
            && src_d.padded_offsets()[d] == 0
            && (false
                    || dims[d] % blocks[d] == 0
                    || dims[d] < blocks[d]);
        if (!ok)
            return unimplemented;

        const bool is_right_border = offsets[d] + dims[d] == src_d.dims()[d];

        dst_d.dims[d] = dims[d];
        dst_d.padded_dims[d] = is_right_border
            ? src_d.padded_dims()[d] - offsets[d] : dst_d.dims[d];
        dst_d.padded_offsets[d] = src_d.padded_offsets()[d];
        dst_d.offset0 += /* [r1] */
            offsets[d] / blocks[d] * dst_d_blk.strides[d];
    }

    *md = dst_d;

    return success;
}

int mkldnn_memory_desc_equal(const memory_desc_t *lhs,
        const memory_desc_t *rhs) {
    if (lhs == rhs) return 1;
    if (any_null(lhs, rhs)) return 0;
    return memory_desc_wrapper(*lhs) == memory_desc_wrapper(*rhs);
}

size_t mkldnn_memory_desc_get_size(const memory_desc_t *md) {
    if (md == nullptr) return 0;
    return memory_desc_wrapper(*md).size();
}

status_t mkldnn_memory_create(memory_t **memory, const memory_desc_t *md,
        engine_t *engine, void *handle) {
    if (any_null(memory, engine)) return invalid_arguments;
    memory_desc_t z_md = types::zero_md();
    return engine->memory_create(memory, md ? md : &z_md, handle);
}

status_t mkldnn_memory_get_memory_desc(const memory_t *memory,
        const memory_desc_t **md) {
    if (any_null(memory, md)) return invalid_arguments;
    *md = memory->md();
    return success;
}

status_t mkldnn_memory_get_engine(const memory_t *memory, engine_t **engine) {
    if (any_null(memory, engine)) return invalid_arguments;
    *engine = memory->engine();
    return success;
}

status_t mkldnn_memory_get_data_handle(const memory_t *memory,
        void **handle) {
    if (any_null(handle))
        return invalid_arguments;
    if (memory == nullptr) {
        *handle = nullptr;
        return success;
    }
    return memory->get_data_handle(handle);
}

status_t mkldnn_memory_set_data_handle(memory_t *memory, void *handle) {
    if (any_null(memory)) return invalid_arguments;
    return memory->set_data_handle(handle);
}

status_t mkldnn_memory_destroy(memory_t *memory) {
    delete memory;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
