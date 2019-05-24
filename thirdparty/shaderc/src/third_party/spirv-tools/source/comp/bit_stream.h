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

// Contains utils for reading, writing and debug printing bit streams.

#ifndef SOURCE_COMP_BIT_STREAM_H_
#define SOURCE_COMP_BIT_STREAM_H_

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace spvtools {
namespace comp {

// Terminology:
// Bits - usually used for a uint64 word, first bit is the lowest.
// Stream - std::string of '0' and '1', read left-to-right,
//          i.e. first bit is at the front and not at the end as in
//          std::bitset::to_string().
// Bitset - std::bitset corresponding to uint64 bits and to reverse(stream).

// Converts number of bits to a respective number of chunks of size N.
// For example NumBitsToNumWords<8> returns how many bytes are needed to store
// |num_bits|.
template <size_t N>
inline size_t NumBitsToNumWords(size_t num_bits) {
  return (num_bits + (N - 1)) / N;
}

// Returns value of the same type as |in|, where all but the first |num_bits|
// are set to zero.
template <typename T>
inline T GetLowerBits(T in, size_t num_bits) {
  return sizeof(T) * 8 == num_bits ? in : in & T((T(1) << num_bits) - T(1));
}

// Encodes signed integer as unsigned. This is a generalized version of
// EncodeZigZag, designed to favor small positive numbers.
// Values are transformed in blocks of 2^|block_exponent|.
// If |block_exponent| is zero, then this degenerates into normal EncodeZigZag.
// Example when |block_exponent| is 1 (return value is the index):
// 0, 1, -1, -2, 2, 3, -3, -4, 4, 5, -5, -6, 6, 7, -7, -8
// Example when |block_exponent| is 2:
// 0, 1, 2, 3, -1, -2, -3, -4, 4, 5, 6, 7, -5, -6, -7, -8
inline uint64_t EncodeZigZag(int64_t val, size_t block_exponent) {
  assert(block_exponent < 64);
  const uint64_t uval = static_cast<uint64_t>(val >= 0 ? val : -val - 1);
  const uint64_t block_num =
      ((uval >> block_exponent) << 1) + (val >= 0 ? 0 : 1);
  const uint64_t pos = GetLowerBits(uval, block_exponent);
  return (block_num << block_exponent) + pos;
}

// Decodes signed integer encoded with EncodeZigZag. |block_exponent| must be
// the same.
inline int64_t DecodeZigZag(uint64_t val, size_t block_exponent) {
  assert(block_exponent < 64);
  const uint64_t block_num = val >> block_exponent;
  const uint64_t pos = GetLowerBits(val, block_exponent);
  if (block_num & 1) {
    // Negative.
    return -1LL - ((block_num >> 1) << block_exponent) - pos;
  } else {
    // Positive.
    return ((block_num >> 1) << block_exponent) + pos;
  }
}

// Converts first |num_bits| stored in uint64 to a left-to-right stream of bits.
inline std::string BitsToStream(uint64_t bits, size_t num_bits = 64) {
  std::bitset<64> bitset(bits);
  std::string str = bitset.to_string().substr(64 - num_bits);
  std::reverse(str.begin(), str.end());
  return str;
}

// Base class for writing sequences of bits.
class BitWriterInterface {
 public:
  BitWriterInterface() = default;
  virtual ~BitWriterInterface() = default;

  // Writes lower |num_bits| in |bits| to the stream.
  // |num_bits| must be no greater than 64.
  virtual void WriteBits(uint64_t bits, size_t num_bits) = 0;

  // Writes bits from value of type |T| to the stream. No encoding is done.
  // Always writes 8 * sizeof(T) bits.
  template <typename T>
  void WriteUnencoded(T val) {
    static_assert(sizeof(T) <= 64, "Type size too large");
    uint64_t bits = 0;
    memcpy(&bits, &val, sizeof(T));
    WriteBits(bits, sizeof(T) * 8);
  }

  // Writes |val| in chunks of size |chunk_length| followed by a signal bit:
  // 0 - no more chunks to follow
  // 1 - more chunks to follow
  // for example 255 is encoded into 1111 1 1111 0 for chunk length 4.
  // The last chunk can be truncated and signal bit omitted, if the entire
  // payload (for example 16 bit for uint16_t has already been written).
  void WriteVariableWidthU64(uint64_t val, size_t chunk_length);
  void WriteVariableWidthU32(uint32_t val, size_t chunk_length);
  void WriteVariableWidthU16(uint16_t val, size_t chunk_length);
  void WriteVariableWidthS64(int64_t val, size_t chunk_length,
                             size_t zigzag_exponent);

  // Returns number of bits written.
  virtual size_t GetNumBits() const = 0;

  // Provides direct access to the buffer data if implemented.
  virtual const uint8_t* GetData() const { return nullptr; }

  // Returns buffer size in bytes.
  size_t GetDataSizeBytes() const { return NumBitsToNumWords<8>(GetNumBits()); }

  // Generates and returns byte array containing written bits.
  virtual std::vector<uint8_t> GetDataCopy() const = 0;

  BitWriterInterface(const BitWriterInterface&) = delete;
  BitWriterInterface& operator=(const BitWriterInterface&) = delete;
};

// This class is an implementation of BitWriterInterface, using
// std::vector<uint64_t> to store written bits.
class BitWriterWord64 : public BitWriterInterface {
 public:
  explicit BitWriterWord64(size_t reserve_bits = 64);

  void WriteBits(uint64_t bits, size_t num_bits) override;

  size_t GetNumBits() const override { return end_; }

  const uint8_t* GetData() const override {
    return reinterpret_cast<const uint8_t*>(buffer_.data());
  }

  std::vector<uint8_t> GetDataCopy() const override {
    return std::vector<uint8_t>(GetData(), GetData() + GetDataSizeBytes());
  }

  // Sets callback to emit bit sequences after every write.
  void SetCallback(std::function<void(const std::string&)> callback) {
    callback_ = callback;
  }

 protected:
  // Sends string generated from arguments to callback_ if defined.
  void EmitSequence(uint64_t bits, size_t num_bits) const {
    if (callback_) callback_(BitsToStream(bits, num_bits));
  }

 private:
  std::vector<uint64_t> buffer_;
  // Total number of bits written so far. Named 'end' as analogy to std::end().
  size_t end_;

  // If not null, the writer will use the callback to emit the written bit
  // sequence as a string of '0' and '1'.
  std::function<void(const std::string&)> callback_;
};

// Base class for reading sequences of bits.
class BitReaderInterface {
 public:
  BitReaderInterface() {}
  virtual ~BitReaderInterface() {}

  // Reads |num_bits| from the stream, stores them in |bits|.
  // Returns number of read bits. |num_bits| must be no greater than 64.
  virtual size_t ReadBits(uint64_t* bits, size_t num_bits) = 0;

  // Reads 8 * sizeof(T) bits and stores them in |val|.
  template <typename T>
  bool ReadUnencoded(T* val) {
    static_assert(sizeof(T) <= 64, "Type size too large");
    uint64_t bits = 0;
    const size_t num_read = ReadBits(&bits, sizeof(T) * 8);
    if (num_read != sizeof(T) * 8) return false;
    memcpy(val, &bits, sizeof(T));
    return true;
  }

  // Returns number of bits already read.
  virtual size_t GetNumReadBits() const = 0;

  // These two functions define 'hard' and 'soft' EOF.
  //
  // Returns true if the end of the buffer was reached.
  virtual bool ReachedEnd() const = 0;
  // Returns true if we reached the end of the buffer or are nearing it and only
  // zero bits are left to read. Implementations of this function are allowed to
  // commit a "false negative" error if the end of the buffer was not reached,
  // i.e. it can return false even if indeed only zeroes are left.
  // It is assumed that the consumer expects that
  // the buffer stream ends with padding zeroes, and would accept this as a
  // 'soft' EOF. Implementations of this class do not necessarily need to
  // implement this, default behavior can simply delegate to ReachedEnd().
  virtual bool OnlyZeroesLeft() const { return ReachedEnd(); }

  // Reads value encoded with WriteVariableWidthXXX (see BitWriterInterface).
  // Reader and writer must use the same |chunk_length| and variable type.
  // Returns true on success, false if the bit stream ends prematurely.
  bool ReadVariableWidthU64(uint64_t* val, size_t chunk_length);
  bool ReadVariableWidthU32(uint32_t* val, size_t chunk_length);
  bool ReadVariableWidthU16(uint16_t* val, size_t chunk_length);
  bool ReadVariableWidthS64(int64_t* val, size_t chunk_length,
                            size_t zigzag_exponent);

  BitReaderInterface(const BitReaderInterface&) = delete;
  BitReaderInterface& operator=(const BitReaderInterface&) = delete;
};

// This class is an implementation of BitReaderInterface which accepts both
// uint8_t and uint64_t buffers as input. uint64_t buffers are consumed and
// owned. uint8_t buffers are copied.
class BitReaderWord64 : public BitReaderInterface {
 public:
  // Consumes and owns the buffer.
  explicit BitReaderWord64(std::vector<uint64_t>&& buffer);

  // Copies the buffer and casts it to uint64.
  // Consuming the original buffer and casting it to uint64 is difficult,
  // as it would potentially cause data misalignment and poor performance.
  explicit BitReaderWord64(const std::vector<uint8_t>& buffer);
  BitReaderWord64(const void* buffer, size_t num_bytes);

  size_t ReadBits(uint64_t* bits, size_t num_bits) override;

  size_t GetNumReadBits() const override { return pos_; }

  bool ReachedEnd() const override;
  bool OnlyZeroesLeft() const override;

  BitReaderWord64() = delete;

  // Sets callback to emit bit sequences after every read.
  void SetCallback(std::function<void(const std::string&)> callback) {
    callback_ = callback;
  }

 protected:
  // Sends string generated from arguments to callback_ if defined.
  void EmitSequence(uint64_t bits, size_t num_bits) const {
    if (callback_) callback_(BitsToStream(bits, num_bits));
  }

 private:
  const std::vector<uint64_t> buffer_;
  size_t pos_;

  // If not null, the reader will use the callback to emit the read bit
  // sequence as a string of '0' and '1'.
  std::function<void(const std::string&)> callback_;
};

}  // namespace comp
}  // namespace spvtools

#endif  // SOURCE_COMP_BIT_STREAM_H_
