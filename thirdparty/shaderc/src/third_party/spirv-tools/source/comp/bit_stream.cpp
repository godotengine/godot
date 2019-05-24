// Copyright (c) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>
#include <type_traits>

#include "source/comp/bit_stream.h"

namespace spvtools {
namespace comp {
namespace {

// Returns if the system is little-endian. Unfortunately only works during
// runtime.
bool IsLittleEndian() {
  // This constant value allows the detection of the host machine's endianness.
  // Accessing it as an array of bytes is valid due to C++11 section 3.10
  // paragraph 10.
  static const uint16_t kFF00 = 0xff00;
  return reinterpret_cast<const unsigned char*>(&kFF00)[0] == 0;
}

// Copies bytes from the given buffer to a uint64_t buffer.
// Motivation: casting uint64_t* to uint8_t* is ok. Casting in the other
// direction is only advisable if uint8_t* is aligned to 64-bit word boundary.
std::vector<uint64_t> ToBuffer64(const void* buffer, size_t num_bytes) {
  std::vector<uint64_t> out;
  out.resize((num_bytes + 7) / 8, 0);
  memcpy(out.data(), buffer, num_bytes);
  return out;
}

// Copies uint8_t buffer to a uint64_t buffer.
std::vector<uint64_t> ToBuffer64(const std::vector<uint8_t>& in) {
  return ToBuffer64(in.data(), in.size());
}

// Returns uint64_t containing the same bits as |val|.
// Type size must be less than 8 bytes.
template <typename T>
uint64_t ToU64(T val) {
  static_assert(sizeof(T) <= 8, "Type size too big");
  uint64_t val64 = 0;
  std::memcpy(&val64, &val, sizeof(T));
  return val64;
}

// Returns value of type T containing the same bits as |val64|.
// Type size must be less than 8 bytes. Upper (unused) bits of |val64| must be
// zero (irrelevant, but is checked with assertion).
template <typename T>
T FromU64(uint64_t val64) {
  assert(sizeof(T) == 8 || (val64 >> (sizeof(T) * 8)) == 0);
  static_assert(sizeof(T) <= 8, "Type size too big");
  T val = 0;
  std::memcpy(&val, &val64, sizeof(T));
  return val;
}

// Writes bits from |val| to |writer| in chunks of size |chunk_length|.
// Signal bit is used to signal if the reader should expect another chunk:
// 0 - no more chunks to follow
// 1 - more chunks to follow
// If number of written bits reaches |max_payload| last chunk is truncated.
void WriteVariableWidthInternal(BitWriterInterface* writer, uint64_t val,
                                size_t chunk_length, size_t max_payload) {
  assert(chunk_length > 0);
  assert(chunk_length < max_payload);
  assert(max_payload == 64 || (val >> max_payload) == 0);

  if (val == 0) {
    // Split in two writes for more readable logging.
    writer->WriteBits(0, chunk_length);
    writer->WriteBits(0, 1);
    return;
  }

  size_t payload_written = 0;

  while (val) {
    if (payload_written + chunk_length >= max_payload) {
      // This has to be the last chunk.
      // There is no need for the signal bit and the chunk can be truncated.
      const size_t left_to_write = max_payload - payload_written;
      assert((val >> left_to_write) == 0);
      writer->WriteBits(val, left_to_write);
      break;
    }

    writer->WriteBits(val, chunk_length);
    payload_written += chunk_length;
    val = val >> chunk_length;

    // Write a single bit to signal if there is more to come.
    writer->WriteBits(val ? 1 : 0, 1);
  }
}

// Reads data written with WriteVariableWidthInternal. |chunk_length| and
// |max_payload| should be identical to those used to write the data.
// Returns false if the stream ends prematurely.
bool ReadVariableWidthInternal(BitReaderInterface* reader, uint64_t* val,
                               size_t chunk_length, size_t max_payload) {
  assert(chunk_length > 0);
  assert(chunk_length <= max_payload);
  size_t payload_read = 0;

  while (payload_read + chunk_length < max_payload) {
    uint64_t bits = 0;
    if (reader->ReadBits(&bits, chunk_length) != chunk_length) return false;

    *val |= bits << payload_read;
    payload_read += chunk_length;

    uint64_t more_to_come = 0;
    if (reader->ReadBits(&more_to_come, 1) != 1) return false;

    if (!more_to_come) {
      return true;
    }
  }

  // Need to read the last chunk which may be truncated. No signal bit follows.
  uint64_t bits = 0;
  const size_t left_to_read = max_payload - payload_read;
  if (reader->ReadBits(&bits, left_to_read) != left_to_read) return false;

  *val |= bits << payload_read;
  return true;
}

// Calls WriteVariableWidthInternal with the right max_payload argument.
template <typename T>
void WriteVariableWidthUnsigned(BitWriterInterface* writer, T val,
                                size_t chunk_length) {
  static_assert(std::is_unsigned<T>::value, "Type must be unsigned");
  static_assert(std::is_integral<T>::value, "Type must be integral");
  WriteVariableWidthInternal(writer, val, chunk_length, sizeof(T) * 8);
}

// Calls ReadVariableWidthInternal with the right max_payload argument.
template <typename T>
bool ReadVariableWidthUnsigned(BitReaderInterface* reader, T* val,
                               size_t chunk_length) {
  static_assert(std::is_unsigned<T>::value, "Type must be unsigned");
  static_assert(std::is_integral<T>::value, "Type must be integral");
  uint64_t val64 = 0;
  if (!ReadVariableWidthInternal(reader, &val64, chunk_length, sizeof(T) * 8))
    return false;
  *val = static_cast<T>(val64);
  assert(*val == val64);
  return true;
}

// Encodes signed |val| to an unsigned value and calls
// WriteVariableWidthInternal with the right max_payload argument.
template <typename T>
void WriteVariableWidthSigned(BitWriterInterface* writer, T val,
                              size_t chunk_length, size_t zigzag_exponent) {
  static_assert(std::is_signed<T>::value, "Type must be signed");
  static_assert(std::is_integral<T>::value, "Type must be integral");
  WriteVariableWidthInternal(writer, EncodeZigZag(val, zigzag_exponent),
                             chunk_length, sizeof(T) * 8);
}

// Calls ReadVariableWidthInternal with the right max_payload argument
// and decodes the value.
template <typename T>
bool ReadVariableWidthSigned(BitReaderInterface* reader, T* val,
                             size_t chunk_length, size_t zigzag_exponent) {
  static_assert(std::is_signed<T>::value, "Type must be signed");
  static_assert(std::is_integral<T>::value, "Type must be integral");
  uint64_t encoded = 0;
  if (!ReadVariableWidthInternal(reader, &encoded, chunk_length, sizeof(T) * 8))
    return false;

  const int64_t decoded = DecodeZigZag(encoded, zigzag_exponent);

  *val = static_cast<T>(decoded);
  assert(*val == decoded);
  return true;
}

}  // namespace

void BitWriterInterface::WriteVariableWidthU64(uint64_t val,
                                               size_t chunk_length) {
  WriteVariableWidthUnsigned(this, val, chunk_length);
}

void BitWriterInterface::WriteVariableWidthU32(uint32_t val,
                                               size_t chunk_length) {
  WriteVariableWidthUnsigned(this, val, chunk_length);
}

void BitWriterInterface::WriteVariableWidthU16(uint16_t val,
                                               size_t chunk_length) {
  WriteVariableWidthUnsigned(this, val, chunk_length);
}

void BitWriterInterface::WriteVariableWidthS64(int64_t val, size_t chunk_length,
                                               size_t zigzag_exponent) {
  WriteVariableWidthSigned(this, val, chunk_length, zigzag_exponent);
}

BitWriterWord64::BitWriterWord64(size_t reserve_bits) : end_(0) {
  buffer_.reserve(NumBitsToNumWords<64>(reserve_bits));
}

void BitWriterWord64::WriteBits(uint64_t bits, size_t num_bits) {
  // Check that |bits| and |num_bits| are valid and consistent.
  assert(num_bits <= 64);
  const bool is_little_endian = IsLittleEndian();
  assert(is_little_endian && "Big-endian architecture support not implemented");
  if (!is_little_endian) return;

  if (num_bits == 0) return;

  bits = GetLowerBits(bits, num_bits);

  EmitSequence(bits, num_bits);

  // Offset from the start of the current word.
  const size_t offset = end_ % 64;

  if (offset == 0) {
    // If no offset, simply add |bits| as a new word to the buffer_.
    buffer_.push_back(bits);
  } else {
    // Shift bits and add them to the current word after offset.
    const uint64_t first_word = bits << offset;
    buffer_.back() |= first_word;

    // If we don't overflow to the next word, there is nothing more to do.

    if (offset + num_bits > 64) {
      // We overflow to the next word.
      const uint64_t second_word = bits >> (64 - offset);
      // Add remaining bits as a new word to buffer_.
      buffer_.push_back(second_word);
    }
  }

  // Move end_ into position for next write.
  end_ += num_bits;
  assert(buffer_.size() * 64 >= end_);
}

bool BitReaderInterface::ReadVariableWidthU64(uint64_t* val,
                                              size_t chunk_length) {
  return ReadVariableWidthUnsigned(this, val, chunk_length);
}

bool BitReaderInterface::ReadVariableWidthU32(uint32_t* val,
                                              size_t chunk_length) {
  return ReadVariableWidthUnsigned(this, val, chunk_length);
}

bool BitReaderInterface::ReadVariableWidthU16(uint16_t* val,
                                              size_t chunk_length) {
  return ReadVariableWidthUnsigned(this, val, chunk_length);
}

bool BitReaderInterface::ReadVariableWidthS64(int64_t* val, size_t chunk_length,
                                              size_t zigzag_exponent) {
  return ReadVariableWidthSigned(this, val, chunk_length, zigzag_exponent);
}

BitReaderWord64::BitReaderWord64(std::vector<uint64_t>&& buffer)
    : buffer_(std::move(buffer)), pos_(0) {}

BitReaderWord64::BitReaderWord64(const std::vector<uint8_t>& buffer)
    : buffer_(ToBuffer64(buffer)), pos_(0) {}

BitReaderWord64::BitReaderWord64(const void* buffer, size_t num_bytes)
    : buffer_(ToBuffer64(buffer, num_bytes)), pos_(0) {}

size_t BitReaderWord64::ReadBits(uint64_t* bits, size_t num_bits) {
  assert(num_bits <= 64);
  const bool is_little_endian = IsLittleEndian();
  assert(is_little_endian && "Big-endian architecture support not implemented");
  if (!is_little_endian) return 0;

  if (ReachedEnd()) return 0;

  // Index of the current word.
  const size_t index = pos_ / 64;

  // Bit position in the current word where we start reading.
  const size_t offset = pos_ % 64;

  // Read all bits from the current word (it might be too much, but
  // excessive bits will be removed later).
  *bits = buffer_[index] >> offset;

  const size_t num_read_from_first_word = std::min(64 - offset, num_bits);
  pos_ += num_read_from_first_word;

  if (pos_ >= buffer_.size() * 64) {
    // Reached end of buffer_.
    EmitSequence(*bits, num_read_from_first_word);
    return num_read_from_first_word;
  }

  if (offset + num_bits > 64) {
    // Requested |num_bits| overflows to next word.
    // Write all bits from the beginning of next word to *bits after offset.
    *bits |= buffer_[index + 1] << (64 - offset);
    pos_ += offset + num_bits - 64;
  }

  // We likely have written more bits than requested. Clear excessive bits.
  *bits = GetLowerBits(*bits, num_bits);
  EmitSequence(*bits, num_bits);
  return num_bits;
}

bool BitReaderWord64::ReachedEnd() const { return pos_ >= buffer_.size() * 64; }

bool BitReaderWord64::OnlyZeroesLeft() const {
  if (ReachedEnd()) return true;

  const size_t index = pos_ / 64;
  if (index < buffer_.size() - 1) return false;

  assert(index == buffer_.size() - 1);

  const size_t offset = pos_ % 64;
  const uint64_t remaining_bits = buffer_[index] >> offset;
  return !remaining_bits;
}

}  // namespace comp
}  // namespace spvtools
