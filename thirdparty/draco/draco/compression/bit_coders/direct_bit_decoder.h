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
#ifndef DRACO_COMPRESSION_BIT_CODERS_DIRECT_BIT_DECODER_H_
#define DRACO_COMPRESSION_BIT_CODERS_DIRECT_BIT_DECODER_H_

#include <vector>

#include "draco/core/decoder_buffer.h"

namespace draco {

class DirectBitDecoder {
 public:
  DirectBitDecoder();
  ~DirectBitDecoder();

  // Sets |source_buffer| as the buffer to decode bits from.
  bool StartDecoding(DecoderBuffer *source_buffer);

  // Decode one bit. Returns true if the bit is a 1, otherwise false.
  bool DecodeNextBit() {
    const uint32_t selector = 1 << (31 - num_used_bits_);
    if (pos_ == bits_.end()) {
      return false;
    }
    const bool bit = *pos_ & selector;
    ++num_used_bits_;
    if (num_used_bits_ == 32) {
      ++pos_;
      num_used_bits_ = 0;
    }
    return bit;
  }

  // Decode the next |nbits| and return the sequence in |value|. |nbits| must be
  // > 0 and <= 32.
  bool DecodeLeastSignificantBits32(int nbits, uint32_t *value) {
    DRACO_DCHECK_EQ(true, nbits <= 32);
    DRACO_DCHECK_EQ(true, nbits > 0);
    const int remaining = 32 - num_used_bits_;
    if (nbits <= remaining) {
      if (pos_ == bits_.end()) {
        return false;
      }
      *value = (*pos_ << num_used_bits_) >> (32 - nbits);
      num_used_bits_ += nbits;
      if (num_used_bits_ == 32) {
        ++pos_;
        num_used_bits_ = 0;
      }
    } else {
      if (pos_ + 1 == bits_.end()) {
        return false;
      }
      const uint32_t value_l = ((*pos_) << num_used_bits_);
      num_used_bits_ = nbits - remaining;
      ++pos_;
      const uint32_t value_r = (*pos_) >> (32 - num_used_bits_);
      *value = (value_l >> (32 - num_used_bits_ - remaining)) | value_r;
    }
    return true;
  }

  void EndDecoding() {}

 private:
  void Clear();

  std::vector<uint32_t> bits_;
  std::vector<uint32_t>::const_iterator pos_;
  uint32_t num_used_bits_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_BIT_CODERS_DIRECT_BIT_DECODER_H_
