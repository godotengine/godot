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
#include "draco/compression/bit_coders/rans_bit_encoder.h"

#include "draco/compression/entropy/ans.h"
#include "draco/core/bit_utils.h"
#include "draco/core/varint_encoding.h"

namespace draco {

RAnsBitEncoder::RAnsBitEncoder() : local_bits_(0), num_local_bits_(0) {}

RAnsBitEncoder::~RAnsBitEncoder() { Clear(); }

void RAnsBitEncoder::StartEncoding() { Clear(); }

void RAnsBitEncoder::EncodeBit(bool bit) {
  if (bit) {
    bit_counts_[1]++;
    local_bits_ |= 1 << num_local_bits_;
  } else {
    bit_counts_[0]++;
  }
  num_local_bits_++;

  if (num_local_bits_ == 32) {
    bits_.push_back(local_bits_);
    num_local_bits_ = 0;
    local_bits_ = 0;
  }
}

void RAnsBitEncoder::EncodeLeastSignificantBits32(int nbits, uint32_t value) {
  DRACO_DCHECK_EQ(true, nbits <= 32);
  DRACO_DCHECK_EQ(true, nbits > 0);

  const uint32_t reversed = ReverseBits32(value) >> (32 - nbits);
  const int ones = CountOneBits32(reversed);
  bit_counts_[0] += (nbits - ones);
  bit_counts_[1] += ones;

  const int remaining = 32 - num_local_bits_;

  if (nbits <= remaining) {
    CopyBits32(&local_bits_, num_local_bits_, reversed, 0, nbits);
    num_local_bits_ += nbits;
    if (num_local_bits_ == 32) {
      bits_.push_back(local_bits_);
      local_bits_ = 0;
      num_local_bits_ = 0;
    }
  } else {
    CopyBits32(&local_bits_, num_local_bits_, reversed, 0, remaining);
    bits_.push_back(local_bits_);
    local_bits_ = 0;
    CopyBits32(&local_bits_, 0, reversed, remaining, nbits - remaining);
    num_local_bits_ = nbits - remaining;
  }
}

void RAnsBitEncoder::EndEncoding(EncoderBuffer *target_buffer) {
  uint64_t total = bit_counts_[1] + bit_counts_[0];
  if (total == 0) {
    total++;
  }

  // The probability interval [0,1] is mapped to values of [0, 256]. However,
  // the coding scheme can not deal with probabilities of 0 or 1, which is why
  // we must clamp the values to interval [1, 255]. Specifically 128
  // corresponds to 0.5 exactly. And the value can be given as uint8_t.
  const uint32_t zero_prob_raw = static_cast<uint32_t>(
      ((bit_counts_[0] / static_cast<double>(total)) * 256.0) + 0.5);

  uint8_t zero_prob = 255;
  if (zero_prob_raw < 255) {
    zero_prob = static_cast<uint8_t>(zero_prob_raw);
  }

  zero_prob += (zero_prob == 0);

  // Space for 32 bit integer and some extra space.
  std::vector<uint8_t> buffer((bits_.size() + 8) * 8);
  AnsCoder ans_coder;
  ans_write_init(&ans_coder, buffer.data());

  for (int i = num_local_bits_ - 1; i >= 0; --i) {
    const uint8_t bit = (local_bits_ >> i) & 1;
    rabs_write(&ans_coder, bit, zero_prob);
  }
  for (auto it = bits_.rbegin(); it != bits_.rend(); ++it) {
    const uint32_t bits = *it;
    for (int i = 31; i >= 0; --i) {
      const uint8_t bit = (bits >> i) & 1;
      rabs_write(&ans_coder, bit, zero_prob);
    }
  }

  const int size_in_bytes = ans_write_end(&ans_coder);
  target_buffer->Encode(zero_prob);
  EncodeVarint(static_cast<uint32_t>(size_in_bytes), target_buffer);
  target_buffer->Encode(buffer.data(), size_in_bytes);

  Clear();
}

void RAnsBitEncoder::Clear() {
  bit_counts_.assign(2, 0);
  bits_.clear();
  local_bits_ = 0;
  num_local_bits_ = 0;
}

}  // namespace draco
