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

#include <initializer_list>

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

status_t fill_blocked(memory_desc_t &md,
        std::initializer_list<int> perm,
        std::initializer_list<int> inner_blks,
        std::initializer_list<int> inner_idxs) {
    const bool ok = true
        && perm.size() == (size_t)md.ndims
        && inner_blks.size() == inner_idxs.size();
    if (!ok) return status::invalid_arguments;

    md.offset0 = 0;

    blocking_desc_t &blk = md.format_desc.blocking;

    dim_t block_size = 1;
    dims_t blocks = {0};
    utils::array_set(blocks, 1, md.ndims);

    blk.inner_nblks = (int)inner_blks.size();

    int iblk = 0;
    for (const auto &b: inner_idxs)
        blk.inner_idxs[iblk++] = b;

    iblk = 0;
    for (const auto &b: inner_blks) {
        int dim = blk.inner_idxs[iblk];
        block_size *= b;
        blocks[dim] *= b;
        blk.inner_blks[iblk++] = b;
    }

    utils::array_set(md.padded_offsets, 0, md.ndims);
    for (int d = 0; d < md.ndims; ++d)
        md.padded_dims[d] = utils::rnd_up(md.dims[d], blocks[d]);

    dim_t stride = block_size;
    // if only we use C++14, the initializer_list would have rbegin()/rend()...
    for (int d = 0; d < md.ndims; ++d)
        stride *= md.padded_dims[d] == 0 ? 1 : md.padded_dims[d] / blocks[d];

    for (const auto &d: perm) {
        if (md.padded_dims[d] == 0) {
             blk.strides[d] = 1;
             continue;
        }
        stride /= md.padded_dims[d] / blocks[d];
        blk.strides[d] = stride;
    }

    assert(stride == block_size);

    return status::success;
}

status_t memory_desc_wrapper::compute_blocking(memory_desc_t &memory_desc,
        format_tag_t tag)
{
    using namespace format_tag;

    if (memory_desc.ndims == 0) return status::invalid_arguments;

#   define C(tag, ... /* perm, inner_blks, inner_idxs */) \
    case tag: return fill_blocked(memory_desc, __VA_ARGS__)

    switch (tag) {
    C(a, {0}, {}, {});
    C(ab, {0, 1}, {}, {});
    C(abc, {0, 1, 2}, {}, {});
    C(abcd, {0, 1, 2, 3}, {}, {});
    C(abcde, {0, 1, 2, 3, 4}, {}, {});
    C(abcdef, {0, 1, 2, 3, 4, 5}, {}, {});
    C(abdec, {0, 1, 3, 4, 2}, {}, {});
    C(acb, {0, 2, 1}, {}, {});
    C(acbde, {0, 2, 1, 3, 4}, {}, {});
    C(acdb, {0, 2, 3, 1}, {}, {});
    C(acdeb, {0, 2, 3, 4, 1}, {}, {});
    C(ba, {1, 0}, {}, {});
    C(bac, {1, 0, 2}, {}, {});
    C(bacd, {1, 0, 2, 3}, {}, {});
    C(bcda, {1, 2, 3, 0}, {}, {});
    C(cba, {2, 1, 0}, {}, {});
    C(cdba, {2, 3, 1, 0}, {}, {});
    C(cdeba, {2, 3, 4, 1, 0}, {}, {});
    C(decab, {3, 4, 2, 0, 1}, {}, {});

    C(Abc4a, {0, 1, 2}, {4}, {0});
    C(aBc4b, {0, 1, 2}, {4}, {1});
    C(ABc4b16a4b, {0, 1, 2}, {4, 16, 4}, {1, 0, 1});
    C(ABc4b4a, {0, 1, 2}, {4, 4}, {1, 0});
    C(Abcd4a, {0, 1, 2, 3}, {4}, {0});
    C(aBcd4b, {0, 1, 2, 3}, {4}, {1});
    C(ABcd4b4a, {0, 1, 2, 3}, {4, 4}, {1, 0});
    C(aBCd4c16b4c, {0, 1, 2, 3}, {4, 16, 4}, {2, 1, 2});
    C(aBCd4c4b, {0, 1, 2, 3, 4}, {4, 4}, {2, 1});
    C(Abcde4a, {0, 1, 2, 3, 4}, {4}, {0});
    C(aBcde4b, {0, 1, 2, 3, 4}, {4}, {1});
    C(ABcde4b4a, {0, 1, 2, 3, 4}, {4, 4}, {1, 0});
    C(aBCde4c4b, {0, 1, 2, 3, 4}, {4, 4}, {2, 1});
    C(aBcdef4b, {0, 1, 2, 3, 4, 5}, {4}, {1});
    C(aBCdef4c4b, {0, 1, 2, 3, 4, 5}, {4, 4}, {2, 1});
    C(aBdc4b, {0, 1, 3, 2}, {4}, {1});
    C(aBdec4b, {0, 1, 3, 4, 2}, {4}, {1});
    C(aBdefc4b, {0, 1, 3, 4, 5, 2}, {4}, {1});
    C(Acb4a, {0, 2, 1}, {4}, {0});
    C(Acdb4a, {0, 2, 3, 1}, {4}, {0});
    C(Acdeb4a, {0, 2, 3, 4, 1}, {4}, {0});

    C(Abc16a, {0, 1, 2}, {16}, {0});
    C(ABc16a16b, {0, 1, 2}, {16, 16}, {0, 1});
    C(aBc16b, {0, 1, 2}, {16}, {1});
    C(ABc16b16a, {0, 1, 2}, {16, 16}, {1, 0});
    C(ABc8a16b2a, {0, 1, 2}, {8, 16, 2}, {0, 1, 0});
    C(ABc8a8b, {0, 1, 2}, {8, 8}, {0, 1});
    C(aBc8b, {0, 1, 2}, {8}, {1});
    C(ABc8b16a2b, {0, 1, 2}, {8, 16, 2}, {1, 0, 1});
    C(ABc8b8a, {0, 1, 2}, {8, 8}, {1, 0});
    C(Abcd16a, {0, 1, 2, 3}, {16}, {0});
    C(ABcd16a16b, {0, 1, 2, 3}, {16, 16}, {0, 1});
    C(aBcd16b, {0, 1, 2, 3}, {16}, {1});
    C(ABcd16b16a, {0, 1, 2, 3}, {16, 16}, {1, 0});
    C(aBCd16b16c, {0, 1, 2, 3}, {16, 16}, {1, 2});
    C(aBCd16c16b, {0, 1, 2, 3}, {16, 16}, {2, 1});
    C(ABcd4b16a4b, {0, 1, 2, 3}, {4, 16, 4}, {1, 0, 1});
    C(ABcd8a16b2a, {0, 1, 2, 3}, {8, 16, 2}, {0, 1, 0});
    C(ABcd8a8b, {0, 1, 2, 3}, {8, 8}, {0, 1});
    C(aBcd8b, {0, 1, 2, 3}, {8}, {1});
    C(ABcd8b16a2b, {0, 1, 2, 3}, {8, 16, 2}, {1, 0, 1});
    C(aBCd8b16c2b, {0, 1, 2, 3}, {8, 16, 2}, {1, 2, 1});
    C(ABcd8b8a, {0, 1, 2, 3}, {8, 8}, {1, 0});
    C(aBCd8b8c, {0, 1, 2, 3}, {8, 8}, {1, 2});
    C(aBCd8c16b2c, {0, 1, 2, 3}, {8, 16, 2}, {2, 1, 2});
    C(aBCd8c8b, {0, 1, 2, 3}, {8, 8}, {2, 1});
    C(Abcde16a, {0, 1, 2, 3, 4}, {16}, {0});
    C(ABcde16a16b, {0, 1, 2, 3, 4}, {16, 16}, {0, 1});
    C(aBcde16b, {0, 1, 2, 3, 4}, {16}, {1});
    C(ABcde16b16a, {0, 1, 2, 3, 4}, {16, 16}, {1, 0});
    C(aBCde16b16c, {0, 1, 2, 3, 4}, {16, 16}, {1, 2});
    C(aBCde16c16b, {0, 1, 2, 3, 4}, {16, 16}, {2, 1});
    C(aBCde2c8b4c, {0, 1, 2, 3, 4}, {2, 8, 4}, {2, 1, 2});
    C(aBCde4b4c, {0, 1, 2, 3, 4}, {4, 4}, {1, 2});
    C(aBCde4c16b4c, {0, 1, 2, 3, 4}, {4, 16, 4}, {2, 1, 2});
    C(Abcde8a, {0, 1, 2, 3, 4}, {8}, {0});
    C(ABcde8a8b, {0, 1, 2, 3, 4}, {8, 8}, {0, 1});
    C(aBcde8b, {0, 1, 2, 3, 4}, {8}, {1});
    C(ABcde8b16a2b, {0, 1, 2, 3, 4}, {8, 16, 2}, {1, 0, 1});
    C(aBCde8b16c2b, {0, 1, 2, 3, 4}, {8, 16, 2}, {1, 2, 1});
    C(ABcde8b8a, {0, 1, 2, 3, 4}, {8, 8}, {1, 0});
    C(aBCde8b8c, {0, 1, 2, 3, 4}, {8, 8}, {1, 2});
    C(aBCde8c16b2c, {0, 1, 2, 3, 4}, {8, 16, 2}, {2, 1, 2});
    C(aBCde8c8b, {0, 1, 2, 3, 4}, {8, 8}, {2, 1});
    C(aBcdef16b, {0, 1, 2, 3, 4, 5}, {16}, {1});
    C(aBCdef16b16c, {0, 1, 2, 3, 4, 5}, {16, 16}, {1, 2});
    C(aBCdef16c16b, {0, 1, 2, 3, 4, 5}, {16, 16}, {2, 1});
    C(aBCdef8b8c, {0, 1, 2, 3, 4, 5}, {8, 8}, {1, 2});
    C(aBCdef8c16b2c, {0, 1, 2, 3, 4, 5}, {8, 16, 2}, {2, 1, 2});
    C(aBCdef8c8b, {0, 1, 2, 3, 4, 5}, {8, 8}, {2, 1});
    C(aBdc16b, {0, 1, 3, 2}, {16}, {1});
    C(aBdc8b, {0, 1, 3, 2}, {8}, {1});
    C(aBdec16b, {0, 1, 3, 4, 2}, {16}, {1});
    C(aBdec8b, {0, 1, 3, 4, 2}, {8}, {1});
    C(aBdefc16b, {0, 1, 3, 4, 5, 2}, {16}, {1});
    C(aBdefc8b, {0, 1, 3, 4, 5, 2}, {8}, {1});
    C(Acb16a, {0, 2, 1}, {16}, {0});
    C(Acb8a, {0, 2, 1}, {8}, {0});
    C(aCBd16b16c, {0, 2, 1, 3}, {16, 16}, {1, 2});
    C(aCBde16b16c, {0, 2, 1, 3, 4}, {16, 16}, {1, 2});
    C(Acdb16a, {0, 2, 3, 1}, {16}, {0});
    C(Acdb8a, {0, 2, 3, 1}, {8}, {0});
    C(Acdeb16a, {0, 2, 3, 4, 1}, {16}, {0});
    C(Acdeb8a, {0, 2, 3, 4, 1}, {8}, {0});
    C(BAc16a16b, {1, 0, 2}, {16, 16}, {0, 1});
    C(BAcd16a16b, {1, 0, 2, 3}, {16, 16}, {0, 1});
    default: break;
    }

#undef C

    return status::invalid_arguments;
}

}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
