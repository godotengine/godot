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
#include "draco/compression/bit_coders/adaptive_rans_bit_encoder.h"

#include "draco/compression/bit_coders/adaptive_rans_bit_coding_shared.h"

namespace draco {

AdaptiveRAnsBitEncoder::AdaptiveRAnsBitEncoder() {}

AdaptiveRAnsBitEncoder::~AdaptiveRAnsBitEncoder() { Clear(); }

void AdaptiveRAnsBitEncoder::StartEncoding() { Clear(); }

void AdaptiveRAnsBitEncoder::EndEncoding(EncoderBuffer *target_buffer) {
  // Buffer for ans to write.
  std::vector<uint8_t> buffer(bits_.size() + 16);
  AnsCoder ans_coder;
  ans_write_init(&ans_coder, buffer.data());

  // Unfortunately we have to encode the bits in reversed order, while the
  // probabilities that should be given are those of the forward sequence.
  double p0_f = 0.5;
  std::vector<uint8_t> p0s;
  p0s.reserve(bits_.size());
  for (bool b : bits_) {
    p0s.push_back(clamp_probability(p0_f));
    p0_f = update_probability(p0_f, b);
  }
  auto bit = bits_.rbegin();
  auto pit = p0s.rbegin();
  while (bit != bits_.rend()) {
    rabs_write(&ans_coder, *bit, *pit);
    ++bit;
    ++pit;
  }

  const uint32_t size_in_bytes = ans_write_end(&ans_coder);
  target_buffer->Encode(size_in_bytes);
  target_buffer->Encode(buffer.data(), size_in_bytes);

  Clear();
}

void AdaptiveRAnsBitEncoder::Clear() { bits_.clear(); }

}  // namespace draco
