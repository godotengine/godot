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
// File provides basic classes and functions for rANS coding.
#ifndef DRACO_COMPRESSION_BIT_CODERS_RANS_BIT_DECODER_H_
#define DRACO_COMPRESSION_BIT_CODERS_RANS_BIT_DECODER_H_

#include <vector>

#include "draco/compression/entropy/ans.h"
#include "draco/core/decoder_buffer.h"
#include "draco/draco_features.h"

namespace draco {

// Class for decoding a sequence of bits that were encoded with RAnsBitEncoder.
class RAnsBitDecoder {
 public:
  RAnsBitDecoder();
  ~RAnsBitDecoder();

  // Sets |source_buffer| as the buffer to decode bits from.
  // Returns false when the data is invalid.
  bool StartDecoding(DecoderBuffer *source_buffer);

  // Decode one bit. Returns true if the bit is a 1, otherwise false.
  bool DecodeNextBit();

  // Decode the next |nbits| and return the sequence in |value|. |nbits| must be
  // > 0 and <= 32.
  void DecodeLeastSignificantBits32(int nbits, uint32_t *value);

  void EndDecoding() {}

 private:
  void Clear();

  AnsDecoder ans_decoder_;
  uint8_t prob_zero_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_BIT_CODERS_RANS_BIT_DECODER_H_
