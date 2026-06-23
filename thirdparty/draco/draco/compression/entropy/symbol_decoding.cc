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
#include "draco/compression/entropy/symbol_decoding.h"

#include <algorithm>
#include <cmath>

#include "draco/compression/entropy/rans_symbol_decoder.h"

namespace draco {

template <template <int> class SymbolDecoderT>
bool DecodeTaggedSymbols(uint32_t num_values, int num_components,
                         DecoderBuffer *src_buffer, uint32_t *out_values);

template <template <int> class SymbolDecoderT>
bool DecodeRawSymbols(uint32_t num_values, DecoderBuffer *src_buffer,
                      uint32_t *out_values);

bool DecodeSymbols(uint32_t num_values, int num_components,
                   DecoderBuffer *src_buffer, uint32_t *out_values) {
  if (num_values == 0) {
    return true;
  }
  // Decode which scheme to use.
  uint8_t scheme;
  if (!src_buffer->Decode(&scheme)) {
    return false;
  }
  if (scheme == SYMBOL_CODING_TAGGED) {
    return DecodeTaggedSymbols<RAnsSymbolDecoder>(num_values, num_components,
                                                  src_buffer, out_values);
  } else if (scheme == SYMBOL_CODING_RAW) {
    return DecodeRawSymbols<RAnsSymbolDecoder>(num_values, src_buffer,
                                               out_values);
  }
  return false;
}

template <template <int> class SymbolDecoderT>
bool DecodeTaggedSymbols(uint32_t num_values, int num_components,
                         DecoderBuffer *src_buffer, uint32_t *out_values) {
  // Decode the encoded data.
  SymbolDecoderT<5> tag_decoder;
  if (!tag_decoder.Create(src_buffer)) {
    return false;
  }

  if (!tag_decoder.StartDecoding(src_buffer)) {
    return false;
  }

  if (num_values > 0 && tag_decoder.num_symbols() == 0) {
    return false;  // Wrong number of symbols.
  }

  // src_buffer now points behind the encoded tag data (to the place where the
  // values are encoded).
  src_buffer->StartBitDecoding(false, nullptr);
  int value_id = 0;
  for (uint32_t i = 0; i < num_values; i += num_components) {
    // Decode the tag.
    const uint32_t bit_length = tag_decoder.DecodeSymbol();
    // Decode the actual value.
    for (int j = 0; j < num_components; ++j) {
      uint32_t val;
      if (!src_buffer->DecodeLeastSignificantBits32(bit_length, &val)) {
        return false;
      }
      out_values[value_id++] = val;
    }
  }
  tag_decoder.EndDecoding();
  src_buffer->EndBitDecoding();
  return true;
}

template <class SymbolDecoderT>
bool DecodeRawSymbolsInternal(uint32_t num_values, DecoderBuffer *src_buffer,
                              uint32_t *out_values) {
  SymbolDecoderT decoder;
  if (!decoder.Create(src_buffer)) {
    return false;
  }

  if (num_values > 0 && decoder.num_symbols() == 0) {
    return false;  // Wrong number of symbols.
  }

  if (!decoder.StartDecoding(src_buffer)) {
    return false;
  }
  for (uint32_t i = 0; i < num_values; ++i) {
    // Decode a symbol into the value.
    const uint32_t value = decoder.DecodeSymbol();
    out_values[i] = value;
  }
  decoder.EndDecoding();
  return true;
}

template <template <int> class SymbolDecoderT>
bool DecodeRawSymbols(uint32_t num_values, DecoderBuffer *src_buffer,
                      uint32_t *out_values) {
  uint8_t max_bit_length;
  if (!src_buffer->Decode(&max_bit_length)) {
    return false;
  }
  switch (max_bit_length) {
    case 1:
      return DecodeRawSymbolsInternal<SymbolDecoderT<1>>(num_values, src_buffer,
                                                         out_values);
    case 2:
      return DecodeRawSymbolsInternal<SymbolDecoderT<2>>(num_values, src_buffer,
                                                         out_values);
    case 3:
      return DecodeRawSymbolsInternal<SymbolDecoderT<3>>(num_values, src_buffer,
                                                         out_values);
    case 4:
      return DecodeRawSymbolsInternal<SymbolDecoderT<4>>(num_values, src_buffer,
                                                         out_values);
    case 5:
      return DecodeRawSymbolsInternal<SymbolDecoderT<5>>(num_values, src_buffer,
                                                         out_values);
    case 6:
      return DecodeRawSymbolsInternal<SymbolDecoderT<6>>(num_values, src_buffer,
                                                         out_values);
    case 7:
      return DecodeRawSymbolsInternal<SymbolDecoderT<7>>(num_values, src_buffer,
                                                         out_values);
    case 8:
      return DecodeRawSymbolsInternal<SymbolDecoderT<8>>(num_values, src_buffer,
                                                         out_values);
    case 9:
      return DecodeRawSymbolsInternal<SymbolDecoderT<9>>(num_values, src_buffer,
                                                         out_values);
    case 10:
      return DecodeRawSymbolsInternal<SymbolDecoderT<10>>(
          num_values, src_buffer, out_values);
    case 11:
      return DecodeRawSymbolsInternal<SymbolDecoderT<11>>(
          num_values, src_buffer, out_values);
    case 12:
      return DecodeRawSymbolsInternal<SymbolDecoderT<12>>(
          num_values, src_buffer, out_values);
    case 13:
      return DecodeRawSymbolsInternal<SymbolDecoderT<13>>(
          num_values, src_buffer, out_values);
    case 14:
      return DecodeRawSymbolsInternal<SymbolDecoderT<14>>(
          num_values, src_buffer, out_values);
    case 15:
      return DecodeRawSymbolsInternal<SymbolDecoderT<15>>(
          num_values, src_buffer, out_values);
    case 16:
      return DecodeRawSymbolsInternal<SymbolDecoderT<16>>(
          num_values, src_buffer, out_values);
    case 17:
      return DecodeRawSymbolsInternal<SymbolDecoderT<17>>(
          num_values, src_buffer, out_values);
    case 18:
      return DecodeRawSymbolsInternal<SymbolDecoderT<18>>(
          num_values, src_buffer, out_values);
    default:
      return false;
  }
}

}  // namespace draco
