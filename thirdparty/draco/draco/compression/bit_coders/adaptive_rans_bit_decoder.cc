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
#include "draco/compression/bit_coders/adaptive_rans_bit_decoder.h"

#include "draco/compression/bit_coders/adaptive_rans_bit_coding_shared.h"

namespace draco {

AdaptiveRAnsBitDecoder::AdaptiveRAnsBitDecoder() : p0_f_(0.5) {}

AdaptiveRAnsBitDecoder::~AdaptiveRAnsBitDecoder() { Clear(); }

bool AdaptiveRAnsBitDecoder::StartDecoding(DecoderBuffer *source_buffer) {
  Clear();

  uint32_t size_in_bytes;
  if (!source_buffer->Decode(&size_in_bytes)) {
    return false;
  }
  if (size_in_bytes > source_buffer->remaining_size()) {
    return false;
  }
  if (ans_read_init(&ans_decoder_,
                    reinterpret_cast<uint8_t *>(
                        const_cast<char *>(source_buffer->data_head())),
                    size_in_bytes) != 0) {
    return false;
  }
  source_buffer->Advance(size_in_bytes);
  return true;
}

bool AdaptiveRAnsBitDecoder::DecodeNextBit() {
  const uint8_t p0 = clamp_probability(p0_f_);
  const bool bit = static_cast<bool>(rabs_read(&ans_decoder_, p0));
  p0_f_ = update_probability(p0_f_, bit);
  return bit;
}

void AdaptiveRAnsBitDecoder::DecodeLeastSignificantBits32(int nbits,
                                                          uint32_t *value) {
  DRACO_DCHECK_EQ(true, nbits <= 32);
  DRACO_DCHECK_EQ(true, nbits > 0);

  uint32_t result = 0;
  while (nbits) {
    result = (result << 1) + DecodeNextBit();
    --nbits;
  }
  *value = result;
}

void AdaptiveRAnsBitDecoder::Clear() {
  ans_read_end(&ans_decoder_);
  p0_f_ = 0.5;
}

}  // namespace draco
