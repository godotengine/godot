//
// Copyright (C) 2025 Jan Kelemen
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
#pragma once
#ifndef spvUtil_H
#define spvUtil_H

#include <cstdint>
#include <type_traits>

#include "spirv.hpp11"

namespace spv {
__inline uint32_t operator&(uint32_t value, spv::MemoryAccessMask mask) { return value & (unsigned)mask; }

__inline bool operator==(uint32_t word, spv::FPEncoding encoding) { return word == (unsigned)encoding; }
__inline bool operator!=(uint32_t word, spv::FPEncoding encoding) { return !(word == encoding); }

__inline bool operator==(uint32_t word, spv::Decoration decoration) { return word == (unsigned)decoration; }
__inline bool operator!=(uint32_t word, spv::Decoration decoration) { return !(word == decoration); }

__inline bool operator==(uint32_t word, spv::Op op) { return word == (unsigned)op; }
__inline bool operator!=(uint32_t word, spv::Op op) { return !(word == op); }

__inline bool operator==(uint32_t word, spv::StorageClass storage) { return word == (unsigned)storage; }
__inline bool operator!=(uint32_t word, spv::StorageClass storage) { return !(word == storage); }

__inline bool anySet(spv::MemoryAccessMask value, spv::MemoryAccessMask mask)
{
    return (value & mask) != spv::MemoryAccessMask::MaskNone;
}

__inline bool anySet(spv::ImageOperandsMask value, spv::ImageOperandsMask mask)
{
    return (value & mask) != spv::ImageOperandsMask::MaskNone;
}

__inline bool anySet(spv::MemorySemanticsMask value, spv::MemorySemanticsMask mask)
{
    return (value & mask) != spv::MemorySemanticsMask::MaskNone;
}

__inline void addMask(uint32_t& word, spv::TensorAddressingOperandsMask mask) { word |= (unsigned)mask; }

__inline void addMask(spv::CooperativeMatrixOperandsMask& word, spv::CooperativeMatrixOperandsMask mask)
{
    word = word | mask;
}

template<typename Enum, typename To = std::underlying_type_t<Enum>>
__inline To enumCast(Enum value)
{
    return static_cast<To>(value);
}
}

#endif // spvUtil_H
