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
#ifndef DRACO_COMPRESSION_BIT_CODERS_DIRECT_BIT_ENCODER_H_
#define DRACO_COMPRESSION_BIT_CODERS_DIRECT_BIT_ENCODER_H_

#include <vector>

#include "draco/core/encoder_buffer.h"

namespace draco {

class DirectBitEncoder {
 public:
  DirectBitEncoder();
  ~DirectBitEncoder();

  // Must be called before any Encode* function is called.
  void StartEncoding();

  // Encode one bit. If |bit| is true encode a 1, otherwise encode a 0.
  void EncodeBit(bool bit) {
    if (bit) {
      local_bits_ |= 1 << (31 - num_local_bits_);
    }
    num_local_bits_++;
    if (num_local_bits_ == 32) {
      bits_.push_back(local_bits_);
      num_local_bits_ = 0;
      local_bits_ = 0;
    }
  }

  // Encode |nbits| of |value|, starting from the least significant bit.
  // |nbits| must be > 0 and <= 32.
  void EncodeLeastSignificantBits32(int nbits, uint32_t value) {
    DRACO_DCHECK_EQ(true, nbits <= 32);
    DRACO_DCHECK_EQ(true, nbits > 0);

    const int remaining = 32 - num_local_bits_;

    // Make sure there are no leading bits that should not be encoded and
    // start from here.
    value = value << (32 - nbits);
    if (nbits <= remaining) {
      value = value >> num_local_bits_;
      local_bits_ = local_bits_ | value;
      num_local_bits_ += nbits;
      if (num_local_bits_ == 32) {
        bits_.push_back(local_bits_);
        local_bits_ = 0;
        num_local_bits_ = 0;
      }
    } else {
      value = value >> (32 - nbits);
      num_local_bits_ = nbits - remaining;
      const uint32_t value_l = value >> num_local_bits_;
      local_bits_ = local_bits_ | value_l;
      bits_.push_back(local_bits_);
      local_bits_ = value << (32 - num_local_bits_);
    }
  }

  // Ends the bit encoding and stores the result into the target_buffer.
  void EndEncoding(EncoderBuffer *target_buffer);

 private:
  void Clear();

  std::vector<uint32_t> bits_;
  uint32_t local_bits_;
  uint32_t num_local_bits_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_BIT_CODERS_DIRECT_BIT_ENCODER_H_
