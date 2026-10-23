// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_bit_writer.h"

#include <cstdint>
#include <cstring>  // memcpy
#include <functional>
#include <memory>
#include <vector>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_aux_out.h"

namespace jxl {

Status BitWriter::WithMaxBits(size_t max_bits, LayerType layer, AuxOut* aux_out,
                              const std::function<Status()>& function,
                              bool finished_histogram) {
  BitWriter::Allotment allotment(max_bits);
  JXL_RETURN_IF_ERROR(allotment.Init(this));
  const Status result = function();
  if (result && finished_histogram) {
    JXL_RETURN_IF_ERROR(allotment.FinishedHistogram(this));
  }
  JXL_RETURN_IF_ERROR(allotment.ReclaimAndCharge(this, layer, aux_out));
  return result;
}
BitWriter::Allotment::Allotment(size_t max_bits) : max_bits_(max_bits) {}

Status BitWriter::Allotment::Init(BitWriter* JXL_RESTRICT writer) {
  prev_bits_written_ = writer->BitsWritten();
  const size_t prev_bytes = writer->storage_.size();
  const size_t next_bytes = DivCeil(max_bits_, kBitsPerByte);
  if (!writer->storage_.resize(prev_bytes + next_bytes)) {
    called_ = true;
    return false;
  }
  parent_ = writer->current_allotment_;
  writer->current_allotment_ = this;
  return true;
}

BitWriter::Allotment::~Allotment() {
  if (!called_) {
    // Not calling is a bug - unused storage will not be reclaimed.
    JXL_DEBUG_ABORT("Did not call Allotment::ReclaimUnused");
  }
}

Status BitWriter::Allotment::FinishedHistogram(BitWriter* JXL_RESTRICT writer) {
  if (writer == nullptr) return true;
  JXL_ENSURE(!called_);              // Call before ReclaimUnused
  JXL_ENSURE(histogram_bits_ == 0);  // Do not call twice
  JXL_ENSURE(writer->BitsWritten() >= prev_bits_written_);
  if (writer->BitsWritten() >= prev_bits_written_) {
    histogram_bits_ = writer->BitsWritten() - prev_bits_written_;
  }
  return true;
}

Status BitWriter::Allotment::ReclaimAndCharge(BitWriter* JXL_RESTRICT writer,
                                              LayerType layer,
                                              AuxOut* JXL_RESTRICT aux_out) {
  size_t used_bits = 0;
  size_t unused_bits = 0;
  JXL_RETURN_IF_ERROR(PrivateReclaim(writer, &used_bits, &unused_bits));

  // This may be a nested call with aux_out == null. Whenever we know that
  // aux_out is null, we can call ReclaimUnused directly.
  if (aux_out != nullptr) {
    aux_out->layer(layer).total_bits += used_bits;
    aux_out->layer(layer).histogram_bits += HistogramBits();
  }
  return true;
}

Status BitWriter::Allotment::PrivateReclaim(BitWriter* JXL_RESTRICT writer,
                                            size_t* JXL_RESTRICT used_bits,
                                            size_t* JXL_RESTRICT unused_bits) {
  JXL_DASSERT(!called_);  // Do not call twice
  called_ = true;
  if (writer == nullptr) return true;

  JXL_DASSERT(writer->BitsWritten() >= prev_bits_written_);
  *used_bits = writer->BitsWritten() - prev_bits_written_;
  JXL_DASSERT(*used_bits <= max_bits_);
  *unused_bits = max_bits_ - *used_bits;

  // Reclaim unused bytes whole bytes from writer's allotment.
  const size_t unused_bytes = *unused_bits / kBitsPerByte;  // truncate
  JXL_ENSURE(writer->storage_.size() >= unused_bytes);
  JXL_RETURN_IF_ERROR(
      writer->storage_.resize(writer->storage_.size() - unused_bytes));
  writer->current_allotment_ = parent_;
  // Ensure we don't also charge the parent for these bits.
  auto* parent = parent_;
  while (parent != nullptr) {
    parent->prev_bits_written_ += *used_bits;
    parent = parent->parent_;
  }
  return true;
}

Status BitWriter::AppendByteAligned(const Span<const uint8_t>& span) {
  if (span.empty()) return true;
  JXL_RETURN_IF_ERROR(storage_.resize(storage_.size() + span.size() +
                                      1));  // extra zero padding

  // Concatenate by copying bytes because both source and destination are bytes.
  JXL_ENSURE(BitsWritten() % kBitsPerByte == 0);
  size_t pos = BitsWritten() / kBitsPerByte;
  memcpy(storage_.data() + pos, span.data(), span.size());
  pos += span.size();
  JXL_ENSURE(pos < storage_.size());
  storage_[pos++] = 0;  // for next Write
  bits_written_ += span.size() * kBitsPerByte;
  return true;
}

Status BitWriter::AppendUnaligned(const BitWriter& other) {
  return WithMaxBits(other.BitsWritten(), LayerType::Header, nullptr, [&] {
    size_t full_bytes = other.BitsWritten() / kBitsPerByte;
    size_t remaining_bits = other.BitsWritten() % kBitsPerByte;
    for (size_t i = 0; i < full_bytes; ++i) {
      Write(8, other.storage_[i]);
    }
    if (remaining_bits > 0) {
      Write(remaining_bits,
            other.storage_[full_bytes] & ((1u << remaining_bits) - 1));
    }
    return true;
  });
}

// TODO(lode): avoid code duplication
Status BitWriter::AppendByteAligned(
    const std::vector<std::unique_ptr<BitWriter>>& others) {
  // Total size to add so we can preallocate
  size_t other_bytes = 0;
  for (const auto& writer : others) {
    JXL_ENSURE(writer->BitsWritten() % kBitsPerByte == 0);
    other_bytes += DivCeil(writer->BitsWritten(), kBitsPerByte);
  }
  if (other_bytes == 0) {
    // No bytes to append: this happens for example when creating per-group
    // storage for groups, but not writing anything in them for e.g. lossless
    // images with no alpha. Do nothing.
    return true;
  }
  JXL_RETURN_IF_ERROR(storage_.resize(storage_.size() + other_bytes +
                                      1));  // extra zero padding

  // Concatenate by copying bytes because both source and destination are bytes.
  JXL_ENSURE(BitsWritten() % kBitsPerByte == 0);
  size_t pos = DivCeil(BitsWritten(), kBitsPerByte);
  for (const auto& writer : others) {
    const Span<const uint8_t> span = writer->GetSpan();
    memcpy(storage_.data() + pos, span.data(), span.size());
    pos += span.size();
  }
  JXL_ENSURE(pos < storage_.size());
  storage_[pos++] = 0;  // for next Write
  bits_written_ += other_bytes * kBitsPerByte;
  return true;
}

// Example: let's assume that 3 bits (Rs below) have been written already:
// BYTE+0       BYTE+1       BYTE+2
// 0000 0RRR    ???? ????    ???? ????
//
// Now, we could write up to 5 bits by just shifting them left by 3 bits and
// OR'ing to BYTE-0.
//
// For n > 5 bits, we write the lowest 5 bits as above, then write the next
// lowest bits into BYTE+1 starting from its lower bits and so on.
void BitWriter::Write(size_t n_bits, uint64_t bits) {
  JXL_DASSERT((bits >> n_bits) == 0);
  JXL_DASSERT(n_bits <= kMaxBitsPerCall);
  size_t bytes_written = bits_written_ / kBitsPerByte;
  uint8_t* p = &storage_[bytes_written];
  const size_t bits_in_first_byte = bits_written_ % kBitsPerByte;
  bits <<= bits_in_first_byte;
#if JXL_BYTE_ORDER_LITTLE
  uint64_t v = *p;
  // Last (partial) or next byte to write must be zero-initialized!
  // PaddedBytes initializes the first, and Write/Append maintain this.
  JXL_DASSERT(v >> bits_in_first_byte == 0);
  v |= bits;
  memcpy(p, &v, sizeof(v));  // Write bytes: possibly more than n_bits/8
#else
  *p++ |= static_cast<uint8_t>(bits & 0xFF);
  for (size_t bits_left_to_write = n_bits + bits_in_first_byte;
       bits_left_to_write >= 9; bits_left_to_write -= 8) {
    bits >>= 8;
    *p++ = static_cast<uint8_t>(bits & 0xFF);
  }
  *p = 0;
#endif
  bits_written_ += n_bits;
}
}  // namespace jxl
