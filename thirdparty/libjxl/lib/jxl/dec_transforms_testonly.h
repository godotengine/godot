// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_TRANSFORMS_TESTONLY_H_
#define LIB_JXL_DEC_TRANSFORMS_TESTONLY_H_

// Facade for (non-inlined) inverse integral transforms.

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

enum class AcStrategyType : uint32_t;

void TransformToPixels(AcStrategyType strategy,
                       float* JXL_RESTRICT coefficients,
                       float* JXL_RESTRICT pixels, size_t pixels_stride,
                       float* JXL_RESTRICT scratch_space);

// Equivalent of the above for DC image.
void LowestFrequenciesFromDC(AcStrategyType strategy, const float* dc,
                             size_t dc_stride, float* llf,
                             float* JXL_RESTRICT scratch);

void AFVIDCT4x4(const float* JXL_RESTRICT coeffs, float* JXL_RESTRICT pixels);

}  // namespace jxl

#endif  // LIB_JXL_DEC_TRANSFORMS_TESTONLY_H_
