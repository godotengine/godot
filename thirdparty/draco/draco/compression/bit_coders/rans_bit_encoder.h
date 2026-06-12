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
#ifndef DRACO_COMPRESSION_BIT_CODERS_RANS_BIT_ENCODER_H_
#define DRACO_COMPRESSION_BIT_CODERS_RANS_BIT_ENCODER_H_

#include <vector>

#include "draco/core/encoder_buffer.h"

namespace draco {

// Class for encoding a sequence of bits using rANS. The probability table used
// to encode the bits is based off the total counts of bits.
// TODO(fgalligan): Investigate using an adaptive table for more compression.
class RAnsBitEncoder {
 public:
  RAnsBitEncoder();
  ~RAnsBitEncoder();

  // Must be called before any Encode* function is called.
  void StartEncoding();

  // Encode one bit. If |bit| is true encode a 1, otherwise encode a 0.
  void EncodeBit(bool bit);

  // Encode |nbits| of |value|, starting from the least significant bit.
  // |nbits| must be > 0 and <= 32.
  void EncodeLeastSignificantBits32(int nbits, uint32_t value);

  // Ends the bit encoding and stores the result into the target_buffer.
  void EndEncoding(EncoderBuffer *target_buffer);

 private:
  void Clear();

  std::vector<uint64_t> bit_counts_;
  std::vector<uint32_t> bits_;
  uint32_t local_bits_;
  uint32_t num_local_bits_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_BIT_CODERS_RANS_BIT_ENCODER_H_
