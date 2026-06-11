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
#ifndef DRACO_CORE_DECODER_BUFFER_H_
#define DRACO_CORE_DECODER_BUFFER_H_

#include <stdint.h>

#include <cstring>
#include <memory>

#include "draco/core/macros.h"
#include "draco/draco_features.h"

namespace draco {

// Class is a wrapper around input data used by MeshDecoder. It provides a
// basic interface for decoding either typed or variable-bit sized data.
class DecoderBuffer {
 public:
  DecoderBuffer();
  DecoderBuffer(const DecoderBuffer &buf) = default;

  DecoderBuffer &operator=(const DecoderBuffer &buf) = default;

  // Sets the buffer's internal data. Note that no copy of the input data is
  // made so the data owner needs to keep the data valid and unchanged for
  // runtime of the decoder.
  void Init(const char *data, size_t data_size);

  // Sets the buffer's internal data. |version| is the Draco bitstream version.
  void Init(const char *data, size_t data_size, uint16_t version);

  // Starts decoding a bit sequence.
  // decode_size must be true if the size of the encoded bit data was included,
  // during encoding. The size is then returned to out_size.
  // Returns false on error.
  bool StartBitDecoding(bool decode_size, uint64_t *out_size);

  // Ends the decoding of the bit sequence and return to the default
  // byte-aligned decoding.
  void EndBitDecoding();

  // Decodes up to 32 bits into out_val. Can be called only in between
  // StartBitDecoding and EndBitDecoding. Otherwise returns false.
  bool DecodeLeastSignificantBits32(uint32_t nbits, uint32_t *out_value) {
    if (!bit_decoder_active()) {
      return false;
    }
    return bit_decoder_.GetBits(nbits, out_value);
  }

  // Decodes an arbitrary data type.
  // Can be used only when we are not decoding a bit-sequence.
  // Returns false on error.
  template <typename T>
  bool Decode(T *out_val) {
    if (!Peek(out_val)) {
      return false;
    }
    pos_ += sizeof(T);
    return true;
  }

  bool Decode(void *out_data, size_t size_to_decode) {
    if (data_size_ < static_cast<int64_t>(pos_ + size_to_decode)) {
      return false;  // Buffer overflow.
    }
    memcpy(out_data, (data_ + pos_), size_to_decode);
    pos_ += size_to_decode;
    return true;
  }

  // Decodes an arbitrary data, but does not advance the reading position.
  template <typename T>
  bool Peek(T *out_val) {
    const size_t size_to_decode = sizeof(T);
    if (data_size_ < static_cast<int64_t>(pos_ + size_to_decode)) {
      return false;  // Buffer overflow.
    }
    memcpy(out_val, (data_ + pos_), size_to_decode);
    return true;
  }

  bool Peek(void *out_data, size_t size_to_peek) {
    if (data_size_ < static_cast<int64_t>(pos_ + size_to_peek)) {
      return false;  // Buffer overflow.
    }
    memcpy(out_data, (data_ + pos_), size_to_peek);
    return true;
  }

  // Discards #bytes from the input buffer.
  void Advance(int64_t bytes) { pos_ += bytes; }

  // Moves the parsing position to a specific offset from the beginning of the
  // input data.
  void StartDecodingFrom(int64_t offset) { pos_ = offset; }

  void set_bitstream_version(uint16_t version) { bitstream_version_ = version; }

  // Returns the data array at the current decoder position.
  const char *data_head() const { return data_ + pos_; }
  int64_t remaining_size() const { return data_size_ - pos_; }
  int64_t decoded_size() const { return pos_; }
  bool bit_decoder_active() const { return bit_mode_; }

  // Returns the bitstream associated with the data. Returns 0 if unknown.
  uint16_t bitstream_version() const { return bitstream_version_; }

 private:
  // Internal helper class to decode bits from a bit buffer.
  class BitDecoder {
   public:
    BitDecoder();
    ~BitDecoder();

    // Sets the bit buffer to |b|. |s| is the size of |b| in bytes.
    inline void reset(const void *b, size_t s) {
      bit_offset_ = 0;
      bit_buffer_ = static_cast<const uint8_t *>(b);
      bit_buffer_end_ = bit_buffer_ + s;
    }

    // Returns number of bits decoded so far.
    inline uint64_t BitsDecoded() const {
      return static_cast<uint64_t>(bit_offset_);
    }

    // Return number of bits available for decoding
    inline uint64_t AvailBits() const {
      return ((bit_buffer_end_ - bit_buffer_) * 8) - bit_offset_;
    }

    inline uint32_t EnsureBits(int k) {
      DRACO_DCHECK_LE(k, 24);
      DRACO_DCHECK_LE(static_cast<uint64_t>(k), AvailBits());

      uint32_t buf = 0;
      for (int i = 0; i < k; ++i) {
        buf |= PeekBit(i) << i;
      }
      return buf;  // Okay to return extra bits
    }

    inline void ConsumeBits(int k) { bit_offset_ += k; }

    // Returns |nbits| bits in |x|.
    inline bool GetBits(uint32_t nbits, uint32_t *x) {
      if (nbits > 32) {
        return false;
      }
      uint32_t value = 0;
      for (uint32_t bit = 0; bit < nbits; ++bit) {
        value |= GetBit() << bit;
      }
      *x = value;
      return true;
    }

   private:
    // TODO(fgalligan): Add support for error reporting on range check.
    // Returns one bit from the bit buffer.
    inline int GetBit() {
      const size_t off = bit_offset_;
      const size_t byte_offset = off >> 3;
      const int bit_shift = static_cast<int>(off & 0x7);
      if (bit_buffer_ + byte_offset < bit_buffer_end_) {
        const int bit = (bit_buffer_[byte_offset] >> bit_shift) & 1;
        bit_offset_ = off + 1;
        return bit;
      }
      return 0;
    }

    inline int PeekBit(int offset) {
      const size_t off = bit_offset_ + offset;
      const size_t byte_offset = off >> 3;
      const int bit_shift = static_cast<int>(off & 0x7);
      if (bit_buffer_ + byte_offset < bit_buffer_end_) {
        const int bit = (bit_buffer_[byte_offset] >> bit_shift) & 1;
        return bit;
      }
      return 0;
    }

    const uint8_t *bit_buffer_;
    const uint8_t *bit_buffer_end_;
    size_t bit_offset_;
  };
  friend class BufferBitCodingTest;

  const char *data_;
  int64_t data_size_;

  // Current parsing position of the decoder.
  int64_t pos_;
  BitDecoder bit_decoder_;
  bool bit_mode_;
  uint16_t bitstream_version_;
};

}  // namespace draco

#endif  // DRACO_CORE_DECODER_BUFFER_H_
