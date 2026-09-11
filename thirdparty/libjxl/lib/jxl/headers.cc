// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/headers.h"

#include <cstdint>
#include <limits>

#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"

namespace jxl {
namespace {

struct Rational {
  constexpr explicit Rational(uint32_t num, uint32_t den)
      : num(num), den(den) {}

  // Returns floor(multiplicand * rational).
  constexpr uint32_t MulTruncate(uint32_t multiplicand) const {
    return static_cast<uint64_t>(multiplicand) * num / den;
  }

  uint32_t num;
  uint32_t den;
};

Rational FixedAspectRatios(uint32_t ratio) {
  JXL_DASSERT(0 != ratio && ratio < 8);
  // Other candidates: 5/4, 7/5, 14/9, 16/10, 5/3, 21/9, 12/5
  constexpr Rational kRatios[7] = {Rational(1, 1),    // square
                                   Rational(12, 10),  //
                                   Rational(4, 3),    // camera
                                   Rational(3, 2),    // mobile camera
                                   Rational(16, 9),   // camera/display
                                   Rational(5, 4),    //
                                   Rational(2, 1)};   //
  return kRatios[ratio - 1];
}

uint32_t FindAspectRatio(uint32_t xsize, uint32_t ysize) {
  for (uint32_t r = 1; r < 8; ++r) {
    if (xsize == FixedAspectRatios(r).MulTruncate(ysize)) {
      return r;
    }
  }
  return 0;  // Must send xsize instead
}

}  // namespace

size_t SizeHeader::xsize() const {
  if (ratio_ != 0) {
    return FixedAspectRatios(ratio_).MulTruncate(
        static_cast<uint32_t>(ysize()));
  }
  return small_ ? ((xsize_div8_minus_1_ + 1) * 8) : xsize_;
}

Status SizeHeader::Set(size_t xsize64, size_t ysize64) {
  constexpr size_t kDimensionCap = std::numeric_limits<uint32_t>::max();
  if (xsize64 > kDimensionCap || ysize64 > kDimensionCap) {
    return JXL_FAILURE("Image too large");
  }
  const uint32_t xsize32 = static_cast<uint32_t>(xsize64);
  const uint32_t ysize32 = static_cast<uint32_t>(ysize64);
  if (xsize64 == 0 || ysize64 == 0) return JXL_FAILURE("Empty image");
  ratio_ = FindAspectRatio(xsize32, ysize32);
  small_ = ysize64 <= 256 && (ysize64 % kBlockDim) == 0 &&
           (ratio_ != 0 || (xsize64 <= 256 && (xsize64 % kBlockDim) == 0));
  if (small_) {
    ysize_div8_minus_1_ = ysize32 / 8 - 1;
  } else {
    ysize_ = ysize32;
  }

  if (ratio_ == 0) {
    if (small_) {
      xsize_div8_minus_1_ = xsize32 / 8 - 1;
    } else {
      xsize_ = xsize32;
    }
  }
  JXL_ENSURE(xsize() == xsize64);
  JXL_ENSURE(ysize() == ysize64);
  return true;
}

Status PreviewHeader::Set(size_t xsize64, size_t ysize64) {
  const uint32_t xsize32 = static_cast<uint32_t>(xsize64);
  const uint32_t ysize32 = static_cast<uint32_t>(ysize64);
  if (xsize64 == 0 || ysize64 == 0) return JXL_FAILURE("Empty preview");
  div8_ = (xsize64 % kBlockDim) == 0 && (ysize64 % kBlockDim) == 0;
  if (div8_) {
    ysize_div8_ = ysize32 / 8;
  } else {
    ysize_ = ysize32;
  }

  ratio_ = FindAspectRatio(xsize32, ysize32);
  if (ratio_ == 0) {
    if (div8_) {
      xsize_div8_ = xsize32 / 8;
    } else {
      xsize_ = xsize32;
    }
  }
  JXL_ENSURE(xsize() == xsize64);
  JXL_ENSURE(ysize() == ysize64);
  return true;
}

size_t PreviewHeader::xsize() const {
  if (ratio_ != 0) {
    return FixedAspectRatios(ratio_).MulTruncate(
        static_cast<uint32_t>(ysize()));
  }
  return div8_ ? (xsize_div8_ * 8) : xsize_;
}

SizeHeader::SizeHeader() { Bundle::Init(this); }
Status SizeHeader::VisitFields(Visitor* JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &small_));

  if (visitor->Conditional(small_)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(5, 0, &ysize_div8_minus_1_));
  }
  if (visitor->Conditional(!small_)) {
    // (Could still be small, but non-multiple of 8.)
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(BitsOffset(9, 1), BitsOffset(13, 1),
                                           BitsOffset(18, 1), BitsOffset(30, 1),
                                           1, &ysize_));
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(3, 0, &ratio_));
  if (visitor->Conditional(ratio_ == 0 && small_)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(5, 0, &xsize_div8_minus_1_));
  }
  if (visitor->Conditional(ratio_ == 0 && !small_)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(BitsOffset(9, 1), BitsOffset(13, 1),
                                           BitsOffset(18, 1), BitsOffset(30, 1),
                                           1, &xsize_));
  }

  return true;
}

PreviewHeader::PreviewHeader() { Bundle::Init(this); }
Status PreviewHeader::VisitFields(Visitor* JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &div8_));

  if (visitor->Conditional(div8_)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(16), Val(32), BitsOffset(5, 1),
                                           BitsOffset(9, 33), 1, &ysize_div8_));
  }
  if (visitor->Conditional(!div8_)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(BitsOffset(6, 1), BitsOffset(8, 65),
                                           BitsOffset(10, 321),
                                           BitsOffset(12, 1345), 1, &ysize_));
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(3, 0, &ratio_));
  if (visitor->Conditional(ratio_ == 0 && div8_)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(16), Val(32), BitsOffset(5, 1),
                                           BitsOffset(9, 33), 1, &xsize_div8_));
  }
  if (visitor->Conditional(ratio_ == 0 && !div8_)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(BitsOffset(6, 1), BitsOffset(8, 65),
                                           BitsOffset(10, 321),
                                           BitsOffset(12, 1345), 1, &xsize_));
  }

  return true;
}

AnimationHeader::AnimationHeader() { Bundle::Init(this); }
Status AnimationHeader::VisitFields(Visitor* JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(100), Val(1000), BitsOffset(10, 1),
                                         BitsOffset(30, 1), 1, &tps_numerator));
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(1), Val(1001), BitsOffset(8, 1),
                                         BitsOffset(10, 1), 1,
                                         &tps_denominator));

  JXL_QUIET_RETURN_IF_ERROR(
      visitor->U32(Val(0), Bits(3), Bits(16), Bits(32), 0, &num_loops));

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &have_timecodes));
  return true;
}

Status ReadSizeHeader(BitReader* JXL_RESTRICT reader,
                      SizeHeader* JXL_RESTRICT size) {
  return Bundle::Read(reader, size);
}

}  // namespace jxl
