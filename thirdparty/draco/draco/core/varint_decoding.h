// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_CORE_VARINT_DECODING_H_
#define DRACO_CORE_VARINT_DECODING_H_

#include <type_traits>

#include "draco/core/bit_utils.h"
#include "draco/core/decoder_buffer.h"

namespace draco {

namespace {

// Decodes a specified unsigned integer as varint. |depth| is the current
// recursion call depth. The first call to the function must be 1.
template <typename IntTypeT>
bool DecodeVarintUnsigned(int depth, IntTypeT *out_val, DecoderBuffer *buffer) {
  constexpr IntTypeT max_depth = sizeof(IntTypeT) + 1 + (sizeof(IntTypeT) >> 3);
  if (depth > max_depth) {
    return false;
  }
  // Coding of unsigned values.
  // 0-6 bit - data
  // 7 bit - next byte?
  uint8_t in;
  if (!buffer->Decode(&in)) {
    return false;
  }
  if (in & (1 << 7)) {
    // Next byte is available, decode it first.
    if (!DecodeVarintUnsigned<IntTypeT>(depth + 1, out_val, buffer)) {
      return false;
    }
    // Append decoded info from this byte.
    *out_val <<= 7;
    *out_val |= in & ((1 << 7) - 1);
  } else {
    // Last byte reached
    *out_val = in;
  }
  return true;
}

}  // namespace

// Decodes a specified integer as varint. Note that the IntTypeT must be the
// same as the one used in the corresponding EncodeVarint() call.
// out_val is undefined if this returns false.
template <typename IntTypeT>
bool DecodeVarint(IntTypeT *out_val, DecoderBuffer *buffer) {
  if (std::is_unsigned<IntTypeT>::value) {
    if (!DecodeVarintUnsigned<IntTypeT>(1, out_val, buffer)) {
      return false;
    }
  } else {
    // IntTypeT is a signed value. Decode the symbol and convert to signed.
    typename std::make_unsigned<IntTypeT>::type symbol;
    if (!DecodeVarintUnsigned(1, &symbol, buffer)) {
      return false;
    }
    *out_val = ConvertSymbolToSignedInt(symbol);
  }
  return true;
}

}  // namespace draco

#endif  // DRACO_CORE_VARINT_DECODING_H_
