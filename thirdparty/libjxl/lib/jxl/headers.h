// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_HEADERS_H_
#define LIB_JXL_HEADERS_H_

// Codestream headers.

#include <stddef.h>
#include <stdint.h>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/field_encodings.h"

namespace jxl {

// Reserved by ISO/IEC 10918-1. LF causes files opened in text mode to be
// rejected because the marker changes to 0x0D instead. The 0xFF prefix also
// ensures there were no 7-bit transmission limitations.
static constexpr uint8_t kCodestreamMarker = 0x0A;

// Compact representation of image dimensions (best case: 9 bits) so decoders
// can preallocate early.
class SizeHeader : public Fields {
 public:
  SizeHeader();
  JXL_FIELDS_NAME(SizeHeader)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  Status Set(size_t xsize, size_t ysize);

  size_t xsize() const;
  size_t ysize() const {
    return small_ ? ((ysize_div8_minus_1_ + 1) * 8) : ysize_;
  }

 private:
  bool small_;  // xsize and ysize <= 256 and divisible by 8.

  uint32_t ysize_div8_minus_1_;
  uint32_t ysize_;

  uint32_t ratio_;
  uint32_t xsize_div8_minus_1_;
  uint32_t xsize_;
};

// (Similar to SizeHeader but different encoding because previews are smaller)
class PreviewHeader : public Fields {
 public:
  PreviewHeader();
  JXL_FIELDS_NAME(PreviewHeader)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  Status Set(size_t xsize, size_t ysize);

  size_t xsize() const;
  size_t ysize() const { return div8_ ? (ysize_div8_ * 8) : ysize_; }

 private:
  bool div8_;  // xsize and ysize divisible by 8.

  uint32_t ysize_div8_;
  uint32_t ysize_;

  uint32_t ratio_;
  uint32_t xsize_div8_;
  uint32_t xsize_;
};

struct AnimationHeader : public Fields {
  AnimationHeader();
  JXL_FIELDS_NAME(AnimationHeader)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  // Ticks per second (expressed as rational number to support NTSC)
  uint32_t tps_numerator;
  uint32_t tps_denominator;

  uint32_t num_loops;  // 0 means to repeat infinitely.

  bool have_timecodes;
};

Status ReadSizeHeader(BitReader* JXL_RESTRICT reader,
                      SizeHeader* JXL_RESTRICT size);

}  // namespace jxl

#endif  // LIB_JXL_HEADERS_H_
