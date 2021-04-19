/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include <stdio.h>
#include <cinttypes>

#include "mkldnn_debug.h"
#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#define DPRINT(...) do { \
    int l = snprintf(str + written_len, str_len, __VA_ARGS__); \
    if (l < 0) return l; \
    if ((size_t)l >= str_len) return -1; \
    written_len += l; str_len -= l; \
} while(0)

int mkldnn_md2fmt_str(char *str, size_t str_len,
        const mkldnn_memory_desc_t *mdesc) {
    using namespace mkldnn::impl;

    if (str == nullptr || str_len <= 1u)
        return -1;

    int written_len = 0;

    if (mdesc == nullptr) {
        DPRINT("%s::%s::",
                mkldnn_dt2str(data_type::undef),
                mkldnn_fmt_kind2str(format_kind::undef));
        return written_len;
    }

    memory_desc_wrapper md(mdesc);

    DPRINT("%s:", mkldnn_dt2str(md.data_type()));

    bool padded_dims = false, padded_offsets = false;
    for (int d = 0; d < md.ndims(); ++d) {
        if (md.dims()[d] != md.padded_dims()[d]) padded_dims = true;
        if (md.padded_offsets()[d] != 0) padded_offsets = true;
    }
    bool offset0 = md.offset0();
    DPRINT("%s%s%s:",
            padded_dims ? "p" : "",
            padded_offsets ? "o" : "",
            offset0 ? "0" : "");

    DPRINT("%s:", mkldnn_fmt_kind2str(md.format_kind()));

    if (!md.is_blocking_desc()) {
        /* TODO: extend */
        DPRINT("%s:", "");
    } else {
        const auto &blk = md.blocking_desc();

        dims_t blocks;
        md.compute_blocks(blocks);

        char dim_chars[MKLDNN_MAX_NDIMS + 1];

        bool plain = true;
        for (int d = 0; d < md.ndims(); ++d) {
            dim_chars[d] = (blocks[d] == 1 ? 'a' : 'A') + (char)d;
            if (blocks[d] != 1) plain = false;
        }

        dims_t strides;
        utils::array_copy(strides, blk.strides, md.ndims());
        utils::simultaneous_sort(strides, dim_chars, md.ndims(),
                [](dim_t a, dim_t b) { return b - a; });

        dim_chars[md.ndims()] = '\0';
        DPRINT("%s", dim_chars);

        if (!plain) {
            for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
                DPRINT("%d%c", (int)blk.inner_blks[iblk],
                        'a' + (char)blk.inner_idxs[iblk]);
            }
        }

        DPRINT("%s", ":");
    }

    DPRINT("f%lx", (long)md.extra().flags);

    return written_len;
}

int mkldnn_md2dim_str(char *str, size_t str_len,
        const mkldnn_memory_desc_t *mdesc) {
    using namespace mkldnn::impl;

    if (str == nullptr || str_len <= 1)
        return -1;

    int written_len = 0;

    if (mdesc == nullptr || mdesc->ndims == 0) {
        DPRINT("%s", "");
        return written_len;
    }

    memory_desc_wrapper md(mdesc);

    for (int d = 0; d < md.ndims() - 1; ++d)
        DPRINT("%" PRId64 "x", md.dims()[d]);
    DPRINT("%" PRId64, md.dims()[md.ndims() - 1]);

    return written_len;
}

#undef  DPRINT
