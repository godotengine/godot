// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/chroma_from_luma.h"

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdlib>  // abs
#include <limits>

#include "lib/jxl/base/common.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

Status ColorCorrelation::DecodeDC(BitReader* br) {
  if (br->ReadFixedBits<1>() == 1) {
    // All default.
    return true;
  }
  SetColorFactor(U32Coder::Read(kColorFactorDist, br));
  JXL_RETURN_IF_ERROR(F16Coder::Read(br, &base_correlation_x_));
  if (std::abs(base_correlation_x_) > 4.0f) {
    return JXL_FAILURE("Base X correlation is out of range");
  }
  JXL_RETURN_IF_ERROR(F16Coder::Read(br, &base_correlation_b_));
  if (std::abs(base_correlation_b_) > 4.0f) {
    return JXL_FAILURE("Base B correlation is out of range");
  }
  ytox_dc_ = static_cast<int>(br->ReadFixedBits<kBitsPerByte>()) +
             std::numeric_limits<int8_t>::min();
  ytob_dc_ = static_cast<int>(br->ReadFixedBits<kBitsPerByte>()) +
             std::numeric_limits<int8_t>::min();
  RecomputeDCFactors();
  return true;
}

StatusOr<ColorCorrelationMap> ColorCorrelationMap::Create(
    JxlMemoryManager* memory_manager, size_t xsize, size_t ysize, bool XYB) {
  ColorCorrelationMap result;
  size_t xblocks = DivCeil(xsize, kColorTileDim);
  size_t yblocks = DivCeil(ysize, kColorTileDim);
  JXL_ASSIGN_OR_RETURN(result.ytox_map,
                       ImageSB::Create(memory_manager, xblocks, yblocks));
  JXL_ASSIGN_OR_RETURN(result.ytob_map,
                       ImageSB::Create(memory_manager, xblocks, yblocks));
  ZeroFillImage(&result.ytox_map);
  ZeroFillImage(&result.ytob_map);
  if (!XYB) {
    result.base_.base_correlation_b_ = 0;
  }
  result.base_.RecomputeDCFactors();
  return result;
}

}  // namespace jxl
