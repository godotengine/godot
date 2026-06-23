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
#include "draco/core/encoder_buffer.h"

#include <cstring>  // for memcpy

#include "draco/core/varint_encoding.h"

namespace draco {

EncoderBuffer::EncoderBuffer()
    : bit_encoder_reserved_bytes_(false), encode_bit_sequence_size_(false) {}

void EncoderBuffer::Clear() {
  buffer_.clear();
  bit_encoder_reserved_bytes_ = 0;
}

void EncoderBuffer::Resize(int64_t nbytes) { buffer_.resize(nbytes); }

bool EncoderBuffer::StartBitEncoding(int64_t required_bits, bool encode_size) {
  if (bit_encoder_active()) {
    return false;  // Bit encoding mode already active.
  }
  if (required_bits <= 0) {
    return false;  // Invalid size.
  }
  encode_bit_sequence_size_ = encode_size;
  const int64_t required_bytes = (required_bits + 7) / 8;
  bit_encoder_reserved_bytes_ = required_bytes;
  uint64_t buffer_start_size = buffer_.size();
  if (encode_size) {
    // Reserve memory for storing the encoded bit sequence size. It will be
    // filled once the bit encoding ends.
    buffer_start_size += sizeof(uint64_t);
  }
  // Resize buffer to fit the maximum size of encoded bit data.
  buffer_.resize(buffer_start_size + required_bytes);
  // Get the buffer data pointer for the bit encoder.
  const char *const data = buffer_.data() + buffer_start_size;
  bit_encoder_ =
      std::unique_ptr<BitEncoder>(new BitEncoder(const_cast<char *>(data)));
  return true;
}

void EncoderBuffer::EndBitEncoding() {
  if (!bit_encoder_active()) {
    return;
  }
  // Get the number of encoded bits and bytes (rounded up).
  const uint64_t encoded_bits = bit_encoder_->Bits();
  const uint64_t encoded_bytes = (encoded_bits + 7) / 8;
  // Flush all cached bits that are not in the bit encoder's main buffer.
  bit_encoder_->Flush(0);
  // Encode size if needed.
  if (encode_bit_sequence_size_) {
    char *out_mem = const_cast<char *>(data() + size());
    // Make the out_mem point to the memory reserved for storing the size.
    out_mem = out_mem - (bit_encoder_reserved_bytes_ + sizeof(uint64_t));

    EncoderBuffer var_size_buffer;
    EncodeVarint(encoded_bytes, &var_size_buffer);
    const uint32_t size_len = static_cast<uint32_t>(var_size_buffer.size());
    char *const dst = out_mem + size_len;
    const char *const src = out_mem + sizeof(uint64_t);
    memmove(dst, src, encoded_bytes);

    // Store the size of the encoded data.
    memcpy(out_mem, var_size_buffer.data(), size_len);

    // We need to account for the difference between the preallocated and actual
    // storage needed for storing the encoded length. This will be used later to
    // compute the correct size of |buffer_|.
    bit_encoder_reserved_bytes_ += sizeof(uint64_t) - size_len;
  }
  // Resize the underlying buffer to match the number of encoded bits.
  buffer_.resize(buffer_.size() - bit_encoder_reserved_bytes_ + encoded_bytes);
  bit_encoder_reserved_bytes_ = 0;
}

}  // namespace draco
