// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_BIT_WRITER_H_
#define LIB_JXL_ENC_BIT_WRITER_H_

// BitWriter class: unbuffered writes using unaligned 64-bit stores.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/padded_bytes.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;

struct BitWriter {
  // Upper bound on `n_bits` in each call to Write. We shift a 64-bit word by
  // 7 bits (max already valid bits in the last byte) and at least 1 bit is
  // needed to zero-initialize the bit-stream ahead (i.e. if 7 bits are valid
  // and we write 57 bits, then the next write will access a byte that was not
  // yet zero-initialized).
  static constexpr size_t kMaxBitsPerCall = 56;

  explicit BitWriter(JxlMemoryManager* memory_manager)
      : bits_written_(0), storage_(memory_manager) {}

  // Disallow copying - may lead to bugs.
  BitWriter(const BitWriter&) = delete;
  BitWriter& operator=(const BitWriter&) = delete;
  BitWriter(BitWriter&&) = default;
  BitWriter& operator=(BitWriter&&) = default;

  size_t BitsWritten() const { return bits_written_; }

  JxlMemoryManager* memory_manager() const { return storage_.memory_manager(); }

  Span<const uint8_t> GetSpan() const {
    // Callers must ensure byte alignment to avoid uninitialized bits.
    JXL_DASSERT(bits_written_ % kBitsPerByte == 0);
    return Bytes(storage_.data(), DivCeil(bits_written_, kBitsPerByte));
  }

  // Example usage: bytes = std::move(writer).TakeBytes(); Useful for the
  // top-level encoder which returns PaddedBytes, not a BitWriter.
  // *this must be an rvalue reference and is invalid afterwards.
  PaddedBytes&& TakeBytes() && {
    // Callers must ensure byte alignment to avoid uninitialized bits.
    JXL_DASSERT(bits_written_ % kBitsPerByte == 0);
    Status status = storage_.resize(DivCeil(bits_written_, kBitsPerByte));
    JXL_DASSERT(status);
    // Can never fail, because we are resizing to a lower size.
    (void)status;
    return std::move(storage_);
  }

  // Must be byte-aligned before calling.
  Status AppendByteAligned(const Span<const uint8_t>& span);

  // NOTE: no allotment needed, the other BitWriters have already been charged.
  Status AppendByteAligned(
      const std::vector<std::unique_ptr<BitWriter>>& others);

  Status AppendUnaligned(const BitWriter& other);

  // Writes bits into bytes in increasing addresses, and within a byte
  // least-significant-bit first.
  //
  // The function can write up to 56 bits in one go.
  void Write(size_t n_bits, uint64_t bits);

  // This should only rarely be used - e.g. when the current location will be
  // referenced via byte offset (TOCs point to groups), or byte-aligned reading
  // is required for speed.
  void ZeroPadToByte() {
    const size_t remainder_bits =
        RoundUpBitsToByteMultiple(bits_written_) - bits_written_;
    if (remainder_bits == 0) return;
    Write(remainder_bits, 0);
    JXL_DASSERT(bits_written_ % kBitsPerByte == 0);
  }

  Status WithMaxBits(size_t max_bits, LayerType layer,
                     AuxOut* JXL_RESTRICT aux_out,
                     const std::function<Status()>& function,
                     bool finished_histogram = false);

 private:
  class Allotment {
   public:
    explicit Allotment(size_t max_bits);
    ~Allotment();

    Allotment(const Allotment& other) = delete;
    Allotment(Allotment&& other) = delete;
    Allotment& operator=(const Allotment&) = delete;
    Allotment& operator=(Allotment&&) = delete;

    // Call after writing a histogram, but before ReclaimUnused.
    Status FinishedHistogram(BitWriter* JXL_RESTRICT writer);

    size_t HistogramBits() const {
      JXL_DASSERT(called_);
      return histogram_bits_;
    }

    Status ReclaimAndCharge(BitWriter* JXL_RESTRICT writer, LayerType layer,
                            AuxOut* JXL_RESTRICT aux_out);

   private:
    friend struct BitWriter;

    // Expands a BitWriter's storage. Must happen before calling Write or
    // ZeroPadToByte. Must call ReclaimUnused after writing to reclaim the
    // unused storage so that BitWriter memory use remains tightly bounded.
    Status Init(BitWriter* JXL_RESTRICT writer);

    Status PrivateReclaim(BitWriter* JXL_RESTRICT writer,
                          size_t* JXL_RESTRICT used_bits,
                          size_t* JXL_RESTRICT unused_bits);

    size_t prev_bits_written_;
    const size_t max_bits_;
    size_t histogram_bits_ = 0;
    bool called_ = false;
    Allotment* parent_;
  };

  size_t bits_written_;
  PaddedBytes storage_;
  Allotment* current_allotment_ = nullptr;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_BIT_WRITER_H_
