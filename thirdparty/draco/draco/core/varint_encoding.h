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
#ifndef DRACO_CORE_VARINT_ENCODING_H_
#define DRACO_CORE_VARINT_ENCODING_H_

#include <type_traits>

#include "draco/core/bit_utils.h"
#include "draco/core/encoder_buffer.h"

namespace draco {

// Encodes a specified integer as varint. Note that different coding is used
// when IntTypeT is an unsigned data type.
template <typename IntTypeT>
bool EncodeVarint(IntTypeT val, EncoderBuffer *out_buffer) {
  if (std::is_unsigned<IntTypeT>::value) {
    // Coding of unsigned values.
    // 0-6 bit - data
    // 7 bit - next byte?
    uint8_t out = 0;
    out |= val & ((1 << 7) - 1);
    if (val >= (1 << 7)) {
      out |= (1 << 7);
      if (!out_buffer->Encode(out)) {
        return false;
      }
      if (!EncodeVarint<IntTypeT>(val >> 7, out_buffer)) {
        return false;
      }
      return true;
    }
    if (!out_buffer->Encode(out)) {
      return false;
    }
  } else {
    // IntTypeT is a signed value. Convert to unsigned symbol and encode.
    const typename std::make_unsigned<IntTypeT>::type symbol =
        ConvertSignedIntToSymbol(val);
    if (!EncodeVarint(symbol, out_buffer)) {
      return false;
    }
  }
  return true;
}

}  // namespace draco

#endif  // DRACO_CORE_VARINT_ENCODING_H_
