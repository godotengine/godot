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
// File provides direct encoding of bits with arithmetic encoder interface.
#ifndef DRACO_COMPRESSION_BIT_CODERS_FOLDED_INTEGER_BIT_DECODER_H_
#define DRACO_COMPRESSION_BIT_CODERS_FOLDED_INTEGER_BIT_DECODER_H_

#include <vector>

#include "draco/core/decoder_buffer.h"

namespace draco {

// See FoldedBit32Encoder for more details.
template <class BitDecoderT>
class FoldedBit32Decoder {
 public:
  FoldedBit32Decoder() {}
  ~FoldedBit32Decoder() {}

  // Sets |source_buffer| as the buffer to decode bits from.
  bool StartDecoding(DecoderBuffer *source_buffer) {
    for (int i = 0; i < 32; i++) {
      if (!folded_number_decoders_[i].StartDecoding(source_buffer)) {
        return false;
      }
    }
    return bit_decoder_.StartDecoding(source_buffer);
  }

  // Decode one bit. Returns true if the bit is a 1, otherwise false.
  bool DecodeNextBit() { return bit_decoder_.DecodeNextBit(); }

  // Decode the next |nbits| and return the sequence in |value|. |nbits| must be
  // > 0 and <= 32.
  void DecodeLeastSignificantBits32(int nbits, uint32_t *value) {
    uint32_t result = 0;
    for (int i = 0; i < nbits; ++i) {
      const bool bit = folded_number_decoders_[i].DecodeNextBit();
      result = (result << 1) + bit;
    }
    *value = result;
  }

  void EndDecoding() {
    for (int i = 0; i < 32; i++) {
      folded_number_decoders_[i].EndDecoding();
    }
    bit_decoder_.EndDecoding();
  }

 private:
  void Clear() {
    for (int i = 0; i < 32; i++) {
      folded_number_decoders_[i].Clear();
    }
    bit_decoder_.Clear();
  }

  std::array<BitDecoderT, 32> folded_number_decoders_;
  BitDecoderT bit_decoder_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_BIT_CODERS_FOLDED_INTEGER_BIT_DECODER_H_
