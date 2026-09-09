//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSRange.hpp
//
// Copyright 2020-2024 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "NSDefines.hpp"
#include "NSTypes.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
struct Range
{
    static Range Make(UInteger loc, UInteger len);

    Range(UInteger loc, UInteger len);

    bool     Equal(const Range& range) const;
    bool     LocationInRange(UInteger loc) const;
    UInteger Max() const;

    UInteger location;
    UInteger length;
} _NS_PACKED;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Range::Range(UInteger loc, UInteger len)
    : location(loc)
    , length(len)
{
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Range NS::Range::Make(UInteger loc, UInteger len)
{
    return Range(loc, len);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Range::Equal(const Range& range) const
{
    return (location == range.location) && (length == range.length);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Range::LocationInRange(UInteger loc) const
{
    return (!(loc < location)) && ((loc - location) < length);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::UInteger NS::Range::Max() const
{
    return location + length;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
