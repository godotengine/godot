// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/fields.h"

#include <algorithm>
#include <cinttypes>  // PRIu64
#include <cmath>
#include <cstddef>
#include <hwy/base.h>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/printf_macros.h"

namespace jxl {

namespace {

using ::jxl::fields_internal::VisitorBase;

struct InitVisitor : public VisitorBase {
  Status Bits(const size_t /*unused*/, const uint32_t default_value,
              uint32_t* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  Status U32(const U32Enc /*unused*/, const uint32_t default_value,
             uint32_t* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  Status U64(const uint64_t default_value,
             uint64_t* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  Status Bool(bool default_value, bool* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  Status F16(const float default_value, float* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  // Always visit conditional fields to ensure they are initialized.
  Status Conditional(bool /*condition*/) override { return true; }

  Status AllDefault(const Fields& /*fields*/,
                    bool* JXL_RESTRICT all_default) override {
    // Just initialize this field and don't skip initializing others.
    JXL_RETURN_IF_ERROR(Bool(true, all_default));
    return false;
  }

  Status VisitNested(Fields* /*fields*/) override {
    // Avoid re-initializing nested bundles (their ctors already called
    // Bundle::Init for their fields).
    return true;
  }
};

// Similar to InitVisitor, but also initializes nested fields.
struct SetDefaultVisitor : public VisitorBase {
  Status Bits(const size_t /*unused*/, const uint32_t default_value,
              uint32_t* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  Status U32(const U32Enc /*unused*/, const uint32_t default_value,
             uint32_t* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  Status U64(const uint64_t default_value,
             uint64_t* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  Status Bool(bool default_value, bool* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  Status F16(const float default_value, float* JXL_RESTRICT value) override {
    *value = default_value;
    return true;
  }

  // Always visit conditional fields to ensure they are initialized.
  Status Conditional(bool /*condition*/) override { return true; }

  Status AllDefault(const Fields& /*fields*/,
                    bool* JXL_RESTRICT all_default) override {
    // Just initialize this field and don't skip initializing others.
    JXL_RETURN_IF_ERROR(Bool(true, all_default));
    return false;
  }
};

class AllDefaultVisitor : public VisitorBase {
 public:
  explicit AllDefaultVisitor() = default;

  Status Bits(const size_t bits, const uint32_t default_value,
              uint32_t* JXL_RESTRICT value) override {
    all_default_ &= *value == default_value;
    return true;
  }

  Status U32(const U32Enc /*unused*/, const uint32_t default_value,
             uint32_t* JXL_RESTRICT value) override {
    all_default_ &= *value == default_value;
    return true;
  }

  Status U64(const uint64_t default_value,
             uint64_t* JXL_RESTRICT value) override {
    all_default_ &= *value == default_value;
    return true;
  }

  Status F16(const float default_value, float* JXL_RESTRICT value) override {
    all_default_ &= std::abs(*value - default_value) < 1E-6f;
    return true;
  }

  Status AllDefault(const Fields& /*fields*/,
                    bool* JXL_RESTRICT /*all_default*/) override {
    // Visit all fields so we can compute the actual all_default_ value.
    return false;
  }

  bool AllDefault() const { return all_default_; }

 private:
  bool all_default_ = true;
};

class ReadVisitor : public VisitorBase {
 public:
  explicit ReadVisitor(BitReader* reader) : reader_(reader) {}

  Status Bits(const size_t bits, const uint32_t /*default_value*/,
              uint32_t* JXL_RESTRICT value) override {
    *value = BitsCoder::Read(bits, reader_);
    if (!reader_->AllReadsWithinBounds()) {
      return JXL_STATUS(StatusCode::kNotEnoughBytes,
                        "Not enough bytes for header");
    }
    return true;
  }

  Status U32(const U32Enc dist, const uint32_t /*default_value*/,
             uint32_t* JXL_RESTRICT value) override {
    *value = U32Coder::Read(dist, reader_);
    if (!reader_->AllReadsWithinBounds()) {
      return JXL_STATUS(StatusCode::kNotEnoughBytes,
                        "Not enough bytes for header");
    }
    return true;
  }

  Status U64(const uint64_t /*default_value*/,
             uint64_t* JXL_RESTRICT value) override {
    *value = U64Coder::Read(reader_);
    if (!reader_->AllReadsWithinBounds()) {
      return JXL_STATUS(StatusCode::kNotEnoughBytes,
                        "Not enough bytes for header");
    }
    return true;
  }

  Status F16(const float /*default_value*/,
             float* JXL_RESTRICT value) override {
    ok_ &= F16Coder::Read(reader_, value);
    if (!reader_->AllReadsWithinBounds()) {
      return JXL_STATUS(StatusCode::kNotEnoughBytes,
                        "Not enough bytes for header");
    }
    return true;
  }

  void SetDefault(Fields* fields) override { Bundle::SetDefault(fields); }

  bool IsReading() const override { return true; }

  // This never fails because visitors are expected to keep reading until
  // EndExtensions, see comment there.
  Status BeginExtensions(uint64_t* JXL_RESTRICT extensions) override {
    JXL_QUIET_RETURN_IF_ERROR(VisitorBase::BeginExtensions(extensions));
    if (*extensions == 0) return true;

    // For each nonzero bit, i.e. extension that is present:
    for (uint64_t remaining_extensions = *extensions; remaining_extensions != 0;
         remaining_extensions &= remaining_extensions - 1) {
      const size_t idx_extension =
          Num0BitsBelowLS1Bit_Nonzero(remaining_extensions);
      // Read additional U64 (one per extension) indicating the number of bits
      // (allows skipping individual extensions).
      JXL_RETURN_IF_ERROR(U64(0, &extension_bits_[idx_extension]));
      if (!SafeAdd(total_extension_bits_, extension_bits_[idx_extension],
                   total_extension_bits_)) {
        return JXL_FAILURE("Extension bits overflowed, invalid codestream");
      }
    }
    // Used by EndExtensions to skip past any _remaining_ extensions.
    pos_after_ext_size_ = reader_->TotalBitsConsumed();
    JXL_ENSURE(pos_after_ext_size_ != 0);
    return true;
  }

  Status EndExtensions() override {
    JXL_QUIET_RETURN_IF_ERROR(VisitorBase::EndExtensions());
    // Happens if extensions == 0: don't read size, done.
    if (pos_after_ext_size_ == 0) return true;

    // Not enough bytes as set by BeginExtensions or earlier. Do not return
    // this as a JXL_FAILURE or false (which can also propagate to error
    // through e.g. JXL_RETURN_IF_ERROR), since this may be used while
    // silently checking whether there are enough bytes. If this case must be
    // treated as an error, reader_>Close() will do this, just like is already
    // done for non-extension fields.
    if (!enough_bytes_) return true;

    // Skip new fields this (old?) decoder didn't know about, if any.
    const size_t bits_read = reader_->TotalBitsConsumed();
    uint64_t end;
    if (!SafeAdd(pos_after_ext_size_, total_extension_bits_, end)) {
      return JXL_FAILURE("Invalid extension size, caused overflow");
    }
    if (bits_read > end) {
      return JXL_FAILURE("Read more extension bits than budgeted");
    }
    const size_t remaining_bits = end - bits_read;
    if (remaining_bits != 0) {
      JXL_WARNING("Skipping %" PRIuS "-bit extension(s)", remaining_bits);
      reader_->SkipBits(remaining_bits);
      if (!reader_->AllReadsWithinBounds()) {
        return JXL_STATUS(StatusCode::kNotEnoughBytes,
                          "Not enough bytes for header");
      }
    }
    return true;
  }

  Status OK() const { return ok_; }

 private:
  // Whether any error other than not enough bytes occurred.
  bool ok_ = true;

  // Whether there are enough input bytes to read from.
  bool enough_bytes_ = true;
  BitReader* const reader_;
  // May be 0 even if the corresponding extension is present.
  uint64_t extension_bits_[Bundle::kMaxExtensions] = {0};
  uint64_t total_extension_bits_ = 0;
  size_t pos_after_ext_size_ = 0;  // 0 iff extensions == 0.

  friend Status jxl::CheckHasEnoughBits(Visitor* /* visitor */,
                                        size_t /* bits */);
};

class MaxBitsVisitor : public VisitorBase {
 public:
  Status Bits(const size_t bits, const uint32_t /*default_value*/,
              uint32_t* JXL_RESTRICT /*value*/) override {
    max_bits_ += BitsCoder::MaxEncodedBits(bits);
    return true;
  }

  Status U32(const U32Enc enc, const uint32_t /*default_value*/,
             uint32_t* JXL_RESTRICT /*value*/) override {
    max_bits_ += U32Coder::MaxEncodedBits(enc);
    return true;
  }

  Status U64(const uint64_t /*default_value*/,
             uint64_t* JXL_RESTRICT /*value*/) override {
    max_bits_ += U64Coder::MaxEncodedBits();
    return true;
  }

  Status F16(const float /*default_value*/,
             float* JXL_RESTRICT /*value*/) override {
    max_bits_ += F16Coder::MaxEncodedBits();
    return true;
  }

  Status AllDefault(const Fields& /*fields*/,
                    bool* JXL_RESTRICT all_default) override {
    JXL_RETURN_IF_ERROR(Bool(true, all_default));
    return false;  // For max bits, assume nothing is default
  }

  // Always visit conditional fields to get a (loose) upper bound.
  Status Conditional(bool /*condition*/) override { return true; }

  Status BeginExtensions(uint64_t* JXL_RESTRICT /*extensions*/) override {
    // Skip - extensions are not included in "MaxBits" because their length
    // is potentially unbounded.
    return true;
  }

  Status EndExtensions() override { return true; }

  size_t MaxBits() const { return max_bits_; }

 private:
  size_t max_bits_ = 0;
};

class CanEncodeVisitor : public VisitorBase {
 public:
  explicit CanEncodeVisitor() = default;

  Status Bits(const size_t bits, const uint32_t /*default_value*/,
              uint32_t* JXL_RESTRICT value) override {
    size_t encoded_bits = 0;
    ok_ &= BitsCoder::CanEncode(bits, *value, &encoded_bits);
    encoded_bits_ += encoded_bits;
    return true;
  }

  Status U32(const U32Enc enc, const uint32_t /*default_value*/,
             uint32_t* JXL_RESTRICT value) override {
    size_t encoded_bits = 0;
    ok_ &= U32Coder::CanEncode(enc, *value, &encoded_bits);
    encoded_bits_ += encoded_bits;
    return true;
  }

  Status U64(const uint64_t /*default_value*/,
             uint64_t* JXL_RESTRICT value) override {
    size_t encoded_bits = 0;
    ok_ &= U64Coder::CanEncode(*value, &encoded_bits);
    encoded_bits_ += encoded_bits;
    return true;
  }

  Status F16(const float /*default_value*/,
             float* JXL_RESTRICT value) override {
    size_t encoded_bits = 0;
    ok_ &= F16Coder::CanEncode(*value, &encoded_bits);
    encoded_bits_ += encoded_bits;
    return true;
  }

  Status AllDefault(const Fields& fields,
                    bool* JXL_RESTRICT all_default) override {
    *all_default = Bundle::AllDefault(fields);
    JXL_RETURN_IF_ERROR(Bool(true, all_default));
    return *all_default;
  }

  Status BeginExtensions(uint64_t* JXL_RESTRICT extensions) override {
    JXL_QUIET_RETURN_IF_ERROR(VisitorBase::BeginExtensions(extensions));
    extensions_ = *extensions;
    if (*extensions != 0) {
      JXL_ENSURE(pos_after_ext_ == 0);
      pos_after_ext_ = encoded_bits_;
      JXL_ENSURE(pos_after_ext_ != 0);  // visited "extensions"
    }
    return true;
  }
  // EndExtensions = default.

  Status GetSizes(size_t* JXL_RESTRICT extension_bits,
                  size_t* JXL_RESTRICT total_bits) {
    JXL_RETURN_IF_ERROR(ok_);
    *extension_bits = 0;
    *total_bits = encoded_bits_;
    // Only if extension field was nonzero will we encode their sizes.
    if (pos_after_ext_ != 0) {
      JXL_ENSURE(encoded_bits_ >= pos_after_ext_);
      *extension_bits = encoded_bits_ - pos_after_ext_;
      // Also need to encode *extension_bits and bill it to *total_bits.
      size_t encoded_bits = 0;
      ok_ &= U64Coder::CanEncode(*extension_bits, &encoded_bits);
      *total_bits += encoded_bits;

      // TODO(janwas): support encoding individual extension sizes. We
      // currently ascribe all bits to the first and send zeros for the
      // others.
      for (size_t i = 1; i < hwy::PopCount(extensions_); ++i) {
        encoded_bits = 0;
        ok_ &= U64Coder::CanEncode(0, &encoded_bits);
        *total_bits += encoded_bits;
      }
    }
    return true;
  }

 private:
  bool ok_ = true;
  size_t encoded_bits_ = 0;
  uint64_t extensions_ = 0;
  // Snapshot of encoded_bits_ after visiting the extension field, but NOT
  // including the hidden extension sizes.
  uint64_t pos_after_ext_ = 0;
};
}  // namespace

void Bundle::Init(Fields* fields) {
  InitVisitor visitor;
  if (!visitor.Visit(fields)) {
    JXL_DEBUG_ABORT("Init should never fail");
  }
}
void Bundle::SetDefault(Fields* fields) {
  SetDefaultVisitor visitor;
  if (!visitor.Visit(fields)) {
    JXL_DEBUG_ABORT("SetDefault should never fail");
  }
}
bool Bundle::AllDefault(const Fields& fields) {
  AllDefaultVisitor visitor;
  if (!visitor.VisitConst(fields)) {
    JXL_DEBUG_ABORT("AllDefault should never fail");
  }
  return visitor.AllDefault();
}
size_t Bundle::MaxBits(const Fields& fields) {
  MaxBitsVisitor visitor;
  Status ret = visitor.VisitConst(fields);
  (void)ret;
  JXL_DASSERT(ret);
  return visitor.MaxBits();
}
Status Bundle::CanEncode(const Fields& fields, size_t* extension_bits,
                         size_t* total_bits) {
  CanEncodeVisitor visitor;
  JXL_QUIET_RETURN_IF_ERROR(visitor.VisitConst(fields));
  JXL_QUIET_RETURN_IF_ERROR(visitor.GetSizes(extension_bits, total_bits));
  return true;
}
Status Bundle::Read(BitReader* reader, Fields* fields) {
  ReadVisitor visitor(reader);
  JXL_RETURN_IF_ERROR(visitor.Visit(fields));
  return visitor.OK();
}
bool Bundle::CanRead(BitReader* reader, Fields* fields) {
  ReadVisitor visitor(reader);
  Status status = visitor.Visit(fields);
  // We are only checking here whether there are enough bytes. We still return
  // true for other errors because it means there are enough bytes to determine
  // there's an error. Use Read() to determine which error it is.
  return status.code() != StatusCode::kNotEnoughBytes;
}

size_t BitsCoder::MaxEncodedBits(const size_t bits) { return bits; }

Status BitsCoder::CanEncode(const size_t bits, const uint32_t value,
                            size_t* JXL_RESTRICT encoded_bits) {
  *encoded_bits = bits;
  if (value >= (1ULL << bits)) {
    return JXL_FAILURE("Value %u too large for %" PRIu64 " bits", value,
                       static_cast<uint64_t>(bits));
  }
  return true;
}

uint32_t BitsCoder::Read(const size_t bits, BitReader* JXL_RESTRICT reader) {
  return reader->ReadBits(bits);
}

size_t U32Coder::MaxEncodedBits(const U32Enc enc) {
  size_t extra_bits = 0;
  for (uint32_t selector = 0; selector < 4; ++selector) {
    const U32Distr d = enc.GetDistr(selector);
    if (d.IsDirect()) {
      continue;
    } else {
      extra_bits = std::max<size_t>(extra_bits, d.ExtraBits());
    }
  }
  return 2 + extra_bits;
}

Status U32Coder::CanEncode(const U32Enc enc, const uint32_t value,
                           size_t* JXL_RESTRICT encoded_bits) {
  uint32_t selector;
  size_t total_bits;
  const Status ok = ChooseSelector(enc, value, &selector, &total_bits);
  *encoded_bits = ok ? total_bits : 0;
  return ok;
}

uint32_t U32Coder::Read(const U32Enc enc, BitReader* JXL_RESTRICT reader) {
  const uint32_t selector = reader->ReadFixedBits<2>();
  const U32Distr d = enc.GetDistr(selector);
  if (d.IsDirect()) {
    return d.Direct();
  } else {
    return reader->ReadBits(d.ExtraBits()) + d.Offset();
  }
}

Status U32Coder::ChooseSelector(const U32Enc enc, const uint32_t value,
                                uint32_t* JXL_RESTRICT selector,
                                size_t* JXL_RESTRICT total_bits) {
  const size_t bits_required = 32 - Num0BitsAboveMS1Bit(value);
  JXL_ENSURE(bits_required <= 32);

  *selector = 0;
  *total_bits = 0;

  // It is difficult to verify whether Dist32Byte are sorted, so check all
  // selectors and keep the one with the fewest total_bits.
  *total_bits = 64;  // more than any valid encoding
  for (uint32_t s = 0; s < 4; ++s) {
    const U32Distr d = enc.GetDistr(s);
    if (d.IsDirect()) {
      if (d.Direct() == value) {
        *selector = s;
        *total_bits = 2;
        return true;  // Done, direct is always the best possible.
      }
      continue;
    }
    const size_t extra_bits = d.ExtraBits();
    const uint32_t offset = d.Offset();
    if (value < offset || value >= offset + (1ULL << extra_bits)) continue;

    // Better than prior encoding, remember it:
    if (2 + extra_bits < *total_bits) {
      *selector = s;
      *total_bits = 2 + extra_bits;
    }
  }

  if (*total_bits == 64) {
    return JXL_FAILURE("No feasible selector for %u", value);
  }

  return true;
}

uint64_t U64Coder::Read(BitReader* JXL_RESTRICT reader) {
  uint64_t selector = reader->ReadFixedBits<2>();
  if (selector == 0) {
    return 0;
  }
  if (selector == 1) {
    return 1 + reader->ReadFixedBits<4>();
  }
  if (selector == 2) {
    return 17 + reader->ReadFixedBits<8>();
  }

  // selector 3, varint, groups have first 12, then 8, and last 4 bits.
  uint64_t result = reader->ReadFixedBits<12>();

  uint64_t shift = 12;
  while (reader->ReadFixedBits<1>()) {
    if (shift == 60) {
      result |= static_cast<uint64_t>(reader->ReadFixedBits<4>()) << shift;
      break;
    }
    result |= static_cast<uint64_t>(reader->ReadFixedBits<8>()) << shift;
    shift += 8;
  }

  return result;
}

// Can always encode, but useful because it also returns bit size.
Status U64Coder::CanEncode(uint64_t value, size_t* JXL_RESTRICT encoded_bits) {
  if (value == 0) {
    *encoded_bits = 2;  // 2 selector bits
  } else if (value <= 16) {
    *encoded_bits = 2 + 4;  // 2 selector bits + 4 payload bits
  } else if (value <= 272) {
    *encoded_bits = 2 + 8;  // 2 selector bits + 8 payload bits
  } else {
    *encoded_bits = 2 + 12;  // 2 selector bits + 12 payload bits
    value >>= 12;
    int shift = 12;
    while (value > 0 && shift < 60) {
      *encoded_bits += 1 + 8;  // 1 continuation bit + 8 payload bits
      value >>= 8;
      shift += 8;
    }
    if (value > 0) {
      // This only could happen if shift == N - 4.
      *encoded_bits += 1 + 4;  // 1 continuation bit + 4 payload bits
    } else {
      *encoded_bits += 1;  // 1 stop bit
    }
  }

  return true;
}

Status F16Coder::Read(BitReader* JXL_RESTRICT reader,
                      float* JXL_RESTRICT value) {
  const uint32_t bits16 = reader->ReadFixedBits<16>();
  const uint32_t sign = bits16 >> 15;
  const uint32_t biased_exp = (bits16 >> 10) & 0x1F;
  const uint32_t mantissa = bits16 & 0x3FF;

  if (JXL_UNLIKELY(biased_exp == 31)) {
    return JXL_FAILURE("F16 infinity or NaN are not supported");
  }

  // Subnormal or zero
  if (JXL_UNLIKELY(biased_exp == 0)) {
    *value = (1.0f / 16384) * (mantissa * (1.0f / 1024));
    if (sign) *value = -*value;
    return true;
  }

  // Normalized: convert the representation directly (faster than ldexp/tables).
  const uint32_t biased_exp32 = biased_exp + (127 - 15);
  const uint32_t mantissa32 = mantissa << (23 - 10);
  const uint32_t bits32 = (sign << 31) | (biased_exp32 << 23) | mantissa32;
  memcpy(value, &bits32, sizeof(bits32));
  return true;
}

Status F16Coder::CanEncode(float value, size_t* JXL_RESTRICT encoded_bits) {
  *encoded_bits = MaxEncodedBits();
  if (std::isnan(value) || std::isinf(value)) {
    return JXL_FAILURE("Should not attempt to store NaN and infinity");
  }
  return std::abs(value) <= 65504.0f;
}

Status CheckHasEnoughBits(Visitor* visitor, size_t bits) {
  if (!visitor->IsReading()) return false;
  ReadVisitor* rv = static_cast<ReadVisitor*>(visitor);
  size_t have_bits = rv->reader_->TotalBytes() * kBitsPerByte;
  size_t want_bits = bits + rv->reader_->TotalBitsConsumed();
  if (have_bits < want_bits) {
    return JXL_STATUS(StatusCode::kNotEnoughBytes,
                      "Not enough bytes for header");
  }
  return true;
}

}  // namespace jxl
