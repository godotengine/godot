// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_BIT_READER_H_
#define LIB_JXL_DEC_BIT_READER_H_

// Bounds-checked bit reader; 64-bit buffer with support for deferred refills
// and switching to reading byte-aligned words.

#include <cstddef>
#include <cstdint>
#include <cstring>  // memcpy

#ifdef __BMI2__
#include <immintrin.h>
#endif

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

// Reads bits previously written to memory by BitWriter. Uses unaligned 8-byte
// little-endian loads.
class BitReader {
 public:
  static constexpr size_t kMaxBitsPerCall = 56;

  // Constructs an invalid BitReader, to be overwritten before usage.
  BitReader()
      : buf_(0),
        bits_in_buf_(0),
        next_byte_{nullptr},
        end_minus_8_{nullptr},
        first_byte_(nullptr) {}
  BitReader(const BitReader&) = delete;

  // bytes need not be aligned nor padded!
  template <class ArrayLike>
  explicit BitReader(const ArrayLike& bytes)
      : buf_(0),
        bits_in_buf_(0),
        next_byte_(bytes.data()),
        // Assumes first_byte_ >= 8.
        end_minus_8_(bytes.data() - 8 + bytes.size()),
        first_byte_(bytes.data()) {
    Refill();
  }
  ~BitReader() {
    // Close() must be called before destroying an initialized bit reader.
    // Invalid bit readers will have a nullptr in first_byte_.
    JXL_DASSERT(close_called_ || !first_byte_);
  }

  // Move operator needs to invalidate the other BitReader such that it is
  // irrelevant if we call Close() on it or not.
  BitReader& operator=(BitReader&& other) noexcept {
    // Ensure the current instance was already closed, before we overwrite it
    // with other.
    JXL_DASSERT(close_called_ || !first_byte_);

    JXL_DASSERT(!other.close_called_);
    buf_ = other.buf_;
    bits_in_buf_ = other.bits_in_buf_;
    next_byte_ = other.next_byte_;
    end_minus_8_ = other.end_minus_8_;
    first_byte_ = other.first_byte_;
    overread_bytes_ = other.overread_bytes_;
    close_called_ = other.close_called_;

    other.first_byte_ = nullptr;
    other.next_byte_ = nullptr;
    return *this;
  }
  BitReader& operator=(const BitReader& other) = delete;

  // For time-critical reads, refills can be shared by multiple reads.
  // Based on variant 4 (plus bounds-checking), see
  // fgiesen.wordpress.com/2018/02/20/reading-bits-in-far-too-many-ways-part-2/
  JXL_INLINE void Refill() {
    if (JXL_UNLIKELY(next_byte_ > end_minus_8_)) {
      BoundsCheckedRefill();
    } else {
      // It's safe to load 64 bits; insert valid (possibly nonzero) bits above
      // bits_in_buf_. The shift requires bits_in_buf_ < 64.
      buf_ |= LoadLE64(next_byte_) << bits_in_buf_;

      // Advance by bytes fully absorbed into the buffer.
      next_byte_ += (63 - bits_in_buf_) >> 3;

      // We absorbed a multiple of 8 bits, so the lower 3 bits of bits_in_buf_
      // must remain unchanged, otherwise the next refill's shifted bits will
      // not align with buf_. Set the three upper bits so the result >= 56.
      bits_in_buf_ |= 56;
      JXL_DASSERT(56 <= bits_in_buf_ && bits_in_buf_ < 64);
    }
  }

  // Returns the bits that would be returned by Read without calling Advance().
  // It is legal to PEEK at more bits than present in the bitstream (required
  // by Huffman), and those bits will be zero.
  template <size_t N>
  JXL_INLINE uint64_t PeekFixedBits() const {
    static_assert(N <= kMaxBitsPerCall, "Reading too many bits in one call.");
    JXL_DASSERT(!close_called_);
    return buf_ & ((1ULL << N) - 1);
  }

  JXL_INLINE uint64_t PeekBits(size_t nbits) const {
    JXL_DASSERT(nbits <= kMaxBitsPerCall);
    JXL_DASSERT(!close_called_);

    // Slightly faster but requires BMI2. It is infeasible to make the many
    // callers reside between begin/end_target, especially because only the
    // callers in dec_ans are time-critical. Therefore only enabled if the
    // entire binary is compiled for (and thus requires) BMI2.
#if defined(__BMI2__) && defined(__x86_64__)
    return _bzhi_u64(buf_, nbits);
#else
    const uint64_t mask = (1ULL << nbits) - 1;
    return buf_ & mask;
#endif
  }

  // Removes bits from the buffer. Need not match the previous Peek size, but
  // the buffer must contain at least num_bits (this prevents consuming more
  // than the total number of bits).
  JXL_INLINE void Consume(size_t num_bits) {
    JXL_DASSERT(!close_called_);
    JXL_DASSERT(bits_in_buf_ >= num_bits);
    if (JXL_CRASH_ON_ERROR) {
      // When JXL_CRASH_ON_ERROR is defined, it is a fatal error to read more
      // bits than available in the stream. A non-zero overread_bytes_ implies
      // that next_byte_ is already at the end of the stream, so we don't need
      // to check that.
      JXL_DASSERT(bits_in_buf_ >= num_bits + overread_bytes_ * kBitsPerByte);
    }
    bits_in_buf_ -= num_bits;
    buf_ >>= num_bits;
  }

  JXL_INLINE uint64_t ReadBits(size_t nbits) {
    JXL_DASSERT(!close_called_);
    Refill();
    const uint64_t bits = PeekBits(nbits);
    Consume(nbits);
    return bits;
  }

  template <size_t N>
  JXL_INLINE uint64_t ReadFixedBits() {
    JXL_DASSERT(!close_called_);
    Refill();
    const uint64_t bits = PeekFixedBits<N>();
    Consume(N);
    return bits;
  }

  // Equivalent to calling ReadFixedBits(1) `skip` times, but much faster.
  // `skip` is typically large.
  void SkipBits(size_t skip) {
    JXL_DASSERT(!close_called_);
    // Buffer is large enough - don't zero buf_ below.
    if (JXL_UNLIKELY(skip <= bits_in_buf_)) {
      Consume(skip);
      return;
    }

    // First deduct what we can satisfy from the buffer
    skip -= bits_in_buf_;
    bits_in_buf_ = 0;
    // Not enough to call Advance - that may leave some bits in the buffer
    // which were previously ABOVE bits_in_buf.
    buf_ = 0;

    // Skip whole bytes
    const size_t whole_bytes = skip / kBitsPerByte;
    skip %= kBitsPerByte;
    if (JXL_UNLIKELY(whole_bytes >
                     static_cast<size_t>(end_minus_8_ + 8 - next_byte_))) {
      // This is already an overflow condition (skipping past the end of the bit
      // stream). However if we increase next_byte_ too much we risk overflowing
      // that value and potentially making it valid again (next_byte_ < end).
      // This will set next_byte_ to the end of the stream and still consume
      // some bits in overread_bytes_, however the TotalBitsConsumed() will be
      // incorrect (still larger than the TotalBytes()).
      next_byte_ = end_minus_8_ + 8;
      skip += kBitsPerByte;
    } else {
      next_byte_ += whole_bytes;
    }

    Refill();
    Consume(skip);
  }

  size_t TotalBitsConsumed() const {
    const size_t bytes_read = static_cast<size_t>(next_byte_ - first_byte_);
    return (bytes_read + overread_bytes_) * kBitsPerByte - bits_in_buf_;
  }

  Status JumpToByteBoundary() {
    const size_t remainder = TotalBitsConsumed() % kBitsPerByte;
    if (remainder == 0) return true;
    if (JXL_UNLIKELY(ReadBits(kBitsPerByte - remainder) != 0)) {
      return JXL_FAILURE("Non-zero padding bits");
    }
    return true;
  }

  // For interoperability with other bitreaders (for resuming at
  // non-byte-aligned positions).
  const uint8_t* FirstByte() const { return first_byte_; }
  size_t TotalBytes() const {
    return static_cast<size_t>(end_minus_8_ + 8 - first_byte_);
  }

  // Returns whether all the bits read so far have been within the input bounds.
  // When reading past the EOF, the Read*() and Consume() functions return zeros
  // but flag a failure when calling Close() without checking this function.
  Status AllReadsWithinBounds() {
    // Mark up to which point the user checked the out of bounds condition. If
    // the user handles the condition at higher level (e.g. fetch more bytes
    // from network, return a custom JXL_FAILURE, ...), Close() should not
    // output a debug error (which would break tests with JXL_CRASH_ON_ERROR
    // even when legitimately handling the situation at higher level). This is
    // used by Bundle::CanRead.
    checked_out_of_bounds_bits_ = TotalBitsConsumed();
    if (TotalBitsConsumed() > TotalBytes() * kBitsPerByte) {
      return false;
    }
    return true;
  }

  // Close the bit reader and return whether all the previous reads were
  // successful. Close must be called once.
  Status Close() {
    JXL_DASSERT(!close_called_);
    close_called_ = true;
    if (!first_byte_) return true;
    if (TotalBitsConsumed() > checked_out_of_bounds_bits_ &&
        TotalBitsConsumed() > TotalBytes() * kBitsPerByte) {
      return JXL_FAILURE("Read more bits than available in the bit_reader");
    }
    return true;
  }

 private:
  // Separate function avoids inlining this relatively cold code into callers.
  JXL_NOINLINE void BoundsCheckedRefill() {
    const uint8_t* end = end_minus_8_ + 8;

    // Read whole bytes until we have [56, 64) bits (same as LoadLE64)
    for (; bits_in_buf_ < 64 - kBitsPerByte; bits_in_buf_ += kBitsPerByte) {
      if (next_byte_ >= end) break;
      buf_ |= static_cast<uint64_t>(*next_byte_++) << bits_in_buf_;
    }
    JXL_DASSERT(bits_in_buf_ < 64);

    // Add extra bytes as 0 at the end of the stream in the bit_buffer_. If
    // these bits are read, Close() will return a failure.
    size_t extra_bytes = (63 - bits_in_buf_) / kBitsPerByte;
    overread_bytes_ += extra_bytes;
    bits_in_buf_ += extra_bytes * kBitsPerByte;

    JXL_DASSERT(bits_in_buf_ < 64);
    JXL_DASSERT(bits_in_buf_ >= 56);
  }

  JXL_NOINLINE uint32_t BoundsCheckedReadByteAlignedWord() {
    if (next_byte_ + 1 < end_minus_8_ + 8) {
      uint32_t ret = LoadLE16(next_byte_);
      next_byte_ += 2;
      return ret;
    }
    overread_bytes_ += 2;
    return 0;
  }

  uint64_t buf_;
  size_t bits_in_buf_;  // [0, 64)
  const uint8_t* JXL_RESTRICT next_byte_;
  const uint8_t* end_minus_8_;  // for refill bounds check
  const uint8_t* first_byte_;   // for GetSpan

  // Number of bytes past the end that were loaded into the buf_. These bytes
  // are not read from memory, but instead assumed 0. It is an error (likely due
  // to an invalid stream) to Consume() more bits than specified in the range
  // passed to the constructor.
  uint64_t overread_bytes_{0};
  bool close_called_{false};

  uint64_t checked_out_of_bounds_bits_{0};
};

// Closes a BitReader when the BitReaderScopedCloser goes out of scope. When
// closing the bit reader, if the status result was failure it sets this failure
// to the passed variable pointer. Typical usage.
//
// Status ret = true;
// {
//   BitReader reader(...);
//   BitReaderScopedCloser reader_closer(&reader, &ret);
//
//   // ... code that can return errors here ...
// }
// // ... more code that doesn't use the BitReader.
// return ret;

class BitReaderScopedCloser {
 public:
  BitReaderScopedCloser(BitReader& reader, Status& status)
      : reader_(&reader), status_(&status) {}
  ~BitReaderScopedCloser() {
    if (reader_ != nullptr) {
      Status close_ret = reader_->Close();
      if (!close_ret) *status_ = close_ret;
    }
  }
  BitReaderScopedCloser(const BitReaderScopedCloser&) = delete;

 private:
  BitReader* reader_;
  Status* status_;
};

}  // namespace jxl

#endif  // LIB_JXL_DEC_BIT_READER_H_
