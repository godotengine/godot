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
#include "draco/compression/bit_coders/direct_bit_decoder.h"

namespace draco {

DirectBitDecoder::DirectBitDecoder() : pos_(bits_.end()), num_used_bits_(0) {}

DirectBitDecoder::~DirectBitDecoder() { Clear(); }

bool DirectBitDecoder::StartDecoding(DecoderBuffer *source_buffer) {
  Clear();
  uint32_t size_in_bytes;
  if (!source_buffer->Decode(&size_in_bytes)) {
    return false;
  }

  // Check that size_in_bytes is > 0 and a multiple of 4 as the encoder always
  // encodes 32 bit elements.
  if (size_in_bytes == 0 || size_in_bytes & 0x3) {
    return false;
  }
  if (size_in_bytes > source_buffer->remaining_size()) {
    return false;
  }
  const uint32_t num_32bit_elements = size_in_bytes / 4;
  bits_.resize(num_32bit_elements);
  if (!source_buffer->Decode(bits_.data(), size_in_bytes)) {
    return false;
  }
  pos_ = bits_.begin();
  num_used_bits_ = 0;
  return true;
}

void DirectBitDecoder::Clear() {
  bits_.clear();
  num_used_bits_ = 0;
  pos_ = bits_.end();
}

}  // namespace draco
