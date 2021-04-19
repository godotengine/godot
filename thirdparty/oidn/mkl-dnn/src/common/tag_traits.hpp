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

#ifndef TAG_TRAITS_HPP
#define TAG_TRAITS_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

enum class block_dim_t {
    _,
    _A, _B,
    _AB, _BC,
};

enum class inner_blk_t {
    _,
    _4a, _4b,
    _8a, _8b,
    _16a, _16b,

    _4b4a, _4b4c, _4c4b,
    _8a8b, _8b8a, _8b8c, _8c8b,
    _16a16b, _16a4b, _16b16a, _16b4c, _16b16c, _16c16b,

    _2c8b4c, _8a16b2a, _4b16a4b, _8b16a2b, _8b16c2b, _4c16b4c, _8c16b2c,
};

/** returns the offset within the block for weights blocked over oc and ic */
template <inner_blk_t f>
constexpr int AB_or_BC_blk_off(int x0, int x1) {
    using ib = inner_blk_t;
    static_assert(utils::one_of(f, ib::_4b4a, ib::_4b4c, ib::_4c4b, ib::_8a8b,
                ib::_8b8a, ib::_8b8c, ib::_8c8b, ib::_16a16b, ib::_16a4b,
                ib::_16b16a, ib::_16b4c, ib::_16b16c, ib::_16c16b, ib::_2c8b4c,
                ib::_8a16b2a, ib::_4b16a4b, ib::_8b16a2b, ib::_8b16c2b,
                ib::_4c16b4c, ib::_8c16b2c),
            "unexpected inner_blk format");
    return false ? 0
        : (f == ib::_4b4c) ? 4 * x0 + x1
        : (f == ib::_4b4a || f == ib::_4c4b) ? 4 * x1 + x0
        : (f == ib::_8a8b || f == ib::_8b8c) ? 8 * x0 + x1
        : (f == ib::_8b8a || f == ib::_8c8b) ? 8 * x1 + x0
        : (f == ib::_16a16b || f == ib::_16b16c) ? 16 * x0 + x1
        : (f == ib::_16b16a || f == ib::_16c16b) ? 16 * x1 + x0
        : (f == ib::_16a4b || f == ib::_16b4c) ? 4 * x0 + x1
        : (f == ib::_8a16b2a || f == ib::_8b16c2b) ? (x0 / 2) * 32 + x1 * 2 + x0 % 2
        : (f == ib::_4b16a4b || f == ib::_4c16b4c) ? (x1 / 4) * 64 + x0 * 4 + x1 % 4
        : (f == ib::_8b16a2b || f == ib::_8c16b2c) ? (x1 / 2) * 32 + x0 * 2 + x1 % 2
        : (f == ib::_2c8b4c) ? (x1 / 4) * 32 + x0 * 4 + x1 % 4
        : INT_MIN;
}

template <inner_blk_t b> struct inner_blk_traits {
    using ib = inner_blk_t;
};

template <format_tag_t> struct tag_traits {
    // block_dim_t block_dims;
    // inner_blk_t inner_blks;
    // int ndims;
};

#define DECL_TRAITS(_tag, _blk_fmt, _inner_blk, _ndims) \
template <> struct tag_traits<format_tag::_tag> { \
    static constexpr block_dim_t block_dims = block_dim_t::_blk_fmt; \
    static constexpr inner_blk_t inner_blks = inner_blk_t::_inner_blk; \
    static constexpr int ndims = _ndims; \
}

DECL_TRAITS(a, _, _, 1);
DECL_TRAITS(ab, _, _, 2);
DECL_TRAITS(abc, _, _, 3);
DECL_TRAITS(abcd, _, _, 4);
DECL_TRAITS(abcde, _, _, 5);
DECL_TRAITS(abcdef, _, _, 6);
DECL_TRAITS(abdec, _, _, 5);
DECL_TRAITS(acb, _, _, 3);
DECL_TRAITS(acbde, _, _, 5);
DECL_TRAITS(acdb, _, _, 4);
DECL_TRAITS(acdeb, _, _, 5);
DECL_TRAITS(ba, _, _, 2);
DECL_TRAITS(bac, _, _, 3);
DECL_TRAITS(bacd, _, _, 4);
DECL_TRAITS(bcda, _, _, 4);
DECL_TRAITS(cba, _, _, 3);
DECL_TRAITS(cdba, _, _, 4);
DECL_TRAITS(cdeba, _, _, 5);
DECL_TRAITS(decab, _, _, 5);

DECL_TRAITS(Abc4a, _A, _4a, 3);
DECL_TRAITS(aBc4b, _B, _4b, 3);
DECL_TRAITS(ABc4b16a4b, _AB, _4b16a4b, 3);
DECL_TRAITS(ABc4b4a, _AB, _4b4a, 3);
DECL_TRAITS(Abcd4a, _A, _4a, 4);
DECL_TRAITS(aBcd4b, _B, _4b, 4);
DECL_TRAITS(ABcd4b4a, _AB, _4b4a, 4);
DECL_TRAITS(aBCd4c16b4c, _BC, _4c16b4c, 4);
DECL_TRAITS(aBCd4c4b, _BC, _4c4b, 4);
DECL_TRAITS(Abcde4a, _A, _4a, 5);
DECL_TRAITS(aBcde4b, _B, _4b, 5);
DECL_TRAITS(ABcde4b4a, _AB, _4b4a, 5);
DECL_TRAITS(aBCde4c4b, _BC, _4c4b, 5);
DECL_TRAITS(aBcdef4b, _B, _4b, 6);
DECL_TRAITS(aBCdef4c4b, _BC, _4c4b, 6);
DECL_TRAITS(aBdc4b, _B, _4b, 4);
DECL_TRAITS(aBdec4b, _B, _4b, 5);
DECL_TRAITS(aBdefc4b, _B, _4b, 6);
DECL_TRAITS(Acb4a, _A, _4a, 3);
DECL_TRAITS(Acdb4a, _A, _4a, 4);
DECL_TRAITS(Acdeb4a, _A, _4a, 5);

DECL_TRAITS(Abc16a, _A, _16a, 3);
DECL_TRAITS(ABc16a16b, _AB, _16a16b, 3);
DECL_TRAITS(aBc16b, _B, _16b, 3);
DECL_TRAITS(ABc16b16a, _AB, _16b16a, 3);
DECL_TRAITS(ABc8a16b2a, _AB, _8a16b2a, 3);
DECL_TRAITS(ABc8a8b, _AB, _8a8b, 3);
DECL_TRAITS(aBc8b, _B, _8b, 3);
DECL_TRAITS(ABc8b16a2b, _AB, _8b16a2b, 3);
DECL_TRAITS(ABc8b8a, _AB, _8b8a, 3);
DECL_TRAITS(Abcd16a, _A, _16a, 4);
DECL_TRAITS(ABcd16a16b, _AB, _16a16b, 4);
DECL_TRAITS(aBcd16b, _B, _16b, 4);
DECL_TRAITS(ABcd16b16a, _AB, _16b16a, 4);
DECL_TRAITS(aBCd16b16c, _BC, _16b16c, 4);
DECL_TRAITS(aBCd16c16b, _BC, _16c16b, 4);
DECL_TRAITS(ABcd4b16a4b, _AB, _4b16a4b, 4);
DECL_TRAITS(ABcd8a16b2a, _AB, _8a16b2a, 4);
DECL_TRAITS(ABcd8a8b, _AB, _8a8b, 4);
DECL_TRAITS(aBcd8b, _B, _8b, 4);
DECL_TRAITS(ABcd8b16a2b, _AB, _8b16a2b, 4);
DECL_TRAITS(aBCd8b16c2b, _BC, _8b16c2b, 4);
DECL_TRAITS(ABcd8b8a, _AB, _8b8a, 4);
DECL_TRAITS(aBCd8b8c, _BC, _8b8c, 4);
DECL_TRAITS(aBCd8c16b2c, _BC, _8c16b2c, 4);
DECL_TRAITS(aBCd8c8b, _BC, _8c8b, 4);
DECL_TRAITS(Abcde16a, _A, _16a, 5);
DECL_TRAITS(ABcde16a16b, _AB, _16a16b, 5);
DECL_TRAITS(aBcde16b, _B, _16b, 5);
DECL_TRAITS(ABcde16b16a, _AB, _16b16a, 5);
DECL_TRAITS(aBCde16b16c, _BC, _16b16c, 5);
DECL_TRAITS(aBCde16c16b, _BC, _16c16b, 5);
DECL_TRAITS(aBCde4c16b4c, _BC, _4c16b4c, 5);
DECL_TRAITS(Abcde8a, _A, _8a, 5);
DECL_TRAITS(ABcde8a8b, _AB, _8a8b, 5);
DECL_TRAITS(aBcde8b, _B, _8b, 5);
DECL_TRAITS(ABcde8b16a2b, _AB, _8b16a2b, 5);
DECL_TRAITS(aBCde8b16c2b, _BC, _8b16c2b, 5);
DECL_TRAITS(ABcde8b8a, _AB, _8b8a, 5);
DECL_TRAITS(aBCde8b8c, _BC, _8b8c, 5);
DECL_TRAITS(aBCde2c8b4c, _BC, _2c8b4c, 5);
DECL_TRAITS(aBCde8c16b2c, _BC, _8c16b2c, 5);
DECL_TRAITS(aBCde4b4c, _BC, _4b4c, 5);
DECL_TRAITS(aBCde8c8b, _BC, _8c8b, 5);
DECL_TRAITS(aBcdef16b, _B, _16b, 6);
DECL_TRAITS(aBCdef16b16c, _BC, _16b16c, 6);
DECL_TRAITS(aBCdef16c16b, _BC, _16c16b, 6);
DECL_TRAITS(aBCdef8b8c, _BC, _8b8c, 6);
DECL_TRAITS(aBCdef8c16b2c, _BC, _8c16b2c, 6);
DECL_TRAITS(aBCdef8c8b, _BC, _8c8b, 6);
DECL_TRAITS(aBdc16b, _B, _16b, 4);
DECL_TRAITS(aBdc8b, _B, _8b, 4);
DECL_TRAITS(aBdec16b, _B, _16b, 5);
DECL_TRAITS(aBdec8b, _B, _8b, 5);
DECL_TRAITS(aBdefc16b, _B, _16b, 6);
DECL_TRAITS(aBdefc8b, _B, _8b, 6);
DECL_TRAITS(Acb16a, _A, _16a, 3);
DECL_TRAITS(Acb8a, _A, _8a, 3);
DECL_TRAITS(aCBd16b16c, _BC, _16b16c, 4);
DECL_TRAITS(aCBde16b16c, _BC, _16b16c, 5);
DECL_TRAITS(Acdb16a, _A, _16a, 4);
DECL_TRAITS(Acdb8a, _A, _8a, 4);
DECL_TRAITS(Acdeb16a, _A, _16a, 5);
DECL_TRAITS(Acdeb8a, _A,  _8a, 5);
DECL_TRAITS(BAc16a16b, _AB, _16a16b, 3);
DECL_TRAITS(BAcd16a16b, _AB, _16a16b, 4);

} // namespace impl
} // namespace mkldnn

#endif
