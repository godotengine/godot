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
#ifndef DRACO_CORE_ENCODER_BUFFER_H_
#define DRACO_CORE_ENCODER_BUFFER_H_

#include <memory>
#include <vector>

#include "draco/core/bit_utils.h"
#include "draco/core/macros.h"

namespace draco {

// Class representing a buffer that can be used for either for byte-aligned
// encoding of arbitrary data structures or for encoding of variable-length
// bit data.
class EncoderBuffer {
 public:
  EncoderBuffer();
  void Clear();
  void Resize(int64_t nbytes);

  // Start encoding a bit sequence. A maximum size of the sequence needs to
  // be known upfront.
  // If encode_size is true, the size of encoded bit sequence is stored before
  // the sequence. Decoder can then use this size to skip over the bit sequence
  // if needed.
  // Returns false on error.
  bool StartBitEncoding(int64_t required_bits, bool encode_size);

  // End the encoding of the bit sequence and return to the default byte-aligned
  // encoding.
  void EndBitEncoding();

  // Encode up to 32 bits into the buffer. Can be called only in between
  // StartBitEncoding and EndBitEncoding. Otherwise returns false.
  bool EncodeLeastSignificantBits32(int nbits, uint32_t value) {
    if (!bit_encoder_active()) {
      return false;
    }
    bit_encoder_->PutBits(value, nbits);
    return true;
  }
  // Encode an arbitrary data type.
  // Can be used only when we are not encoding a bit-sequence.
  // Returns false when the value couldn't be encoded.
  template <typename T>
  bool Encode(const T &data) {
    if (bit_encoder_active()) {
      return false;
    }
    const uint8_t *src_data = reinterpret_cast<const uint8_t *>(&data);
    buffer_.insert(buffer_.end(), src_data, src_data + sizeof(T));
    return true;
  }
  bool Encode(const void *data, size_t data_size) {
    if (bit_encoder_active()) {
      return false;
    }
    const uint8_t *src_data = reinterpret_cast<const uint8_t *>(data);
    buffer_.insert(buffer_.end(), src_data, src_data + data_size);
    return true;
  }

  bool bit_encoder_active() const { return bit_encoder_reserved_bytes_ > 0; }
  const char *data() const { return buffer_.data(); }
  size_t size() const { return buffer_.size(); }
  std::vector<char> *buffer() { return &buffer_; }

 private:
  // Internal helper class to encode bits to a bit buffer.
  class BitEncoder {
   public:
    // |data| is the buffer to write the bits into.
    explicit BitEncoder(char *data) : bit_buffer_(data), bit_offset_(0) {}

    // Write |nbits| of |data| into the bit buffer.
    void PutBits(uint32_t data, int32_t nbits) {
      DRACO_DCHECK_GE(nbits, 0);
      DRACO_DCHECK_LE(nbits, 32);
      for (int32_t bit = 0; bit < nbits; ++bit) {
        PutBit((data >> bit) & 1);
      }
    }

    // Return number of bits encoded so far.
    uint64_t Bits() const { return static_cast<uint64_t>(bit_offset_); }

    // TODO(fgalligan): Remove this function once we know we do not need the
    // old API anymore.
    // This is a function of an old API, that currently does nothing.
    void Flush(int /* left_over_bit_value */) {}

    // Return the number of bits required to store the given number
    static uint32_t BitsRequired(uint32_t x) {
      return static_cast<uint32_t>(MostSignificantBit(x));
    }

   private:
    void PutBit(uint8_t value) {
      const int byte_size = 8;
      const uint64_t off = static_cast<uint64_t>(bit_offset_);
      const uint64_t byte_offset = off / byte_size;
      const int bit_shift = off % byte_size;

      // TODO(fgalligan): Check performance if we add a branch and only do one
      // memory write if bit_shift is 7. Also try using a temporary variable to
      // hold the bits before writing to the buffer.

      bit_buffer_[byte_offset] &= ~(1 << bit_shift);
      bit_buffer_[byte_offset] |= value << bit_shift;
      bit_offset_++;
    }

    char *bit_buffer_;
    size_t bit_offset_;
  };
  friend class BufferBitCodingTest;
  // All data is stored in this vector.
  std::vector<char> buffer_;

  // Bit encoder is used when encoding variable-length bit data.
  // TODO(ostava): Currently encoder needs to be recreated each time
  // StartBitEncoding method is called. This is not necessary if BitEncoder
  // supported reset function which can easily added but let's leave that for
  // later.
  std::unique_ptr<BitEncoder> bit_encoder_;

  // The number of bytes reserved for bit encoder.
  // Values > 0 indicate we are in the bit encoding mode.
  int64_t bit_encoder_reserved_bytes_;

  // Flag used indicating that we need to store the length of the currently
  // processed bit sequence.
  bool encode_bit_sequence_size_;
};

}  // namespace draco

#endif  // DRACO_CORE_ENCODER_BUFFER_H_
