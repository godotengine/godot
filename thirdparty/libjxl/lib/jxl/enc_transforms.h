// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_TRANSFORMS_H_
#define LIB_JXL_ENC_TRANSFORMS_H_

// Facade for (non-inlined) integral transforms.

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

enum class AcStrategyType : uint32_t;

void TransformFromPixels(AcStrategyType strategy,
                         const float* JXL_RESTRICT pixels, size_t pixels_stride,
                         float* JXL_RESTRICT coefficients,
                         float* JXL_RESTRICT scratch_space);

// Equivalent of the above for DC image.
void DCFromLowestFrequencies(AcStrategyType strategy, const float* block,
                             float* dc, size_t dc_stride);

void AFVDCT4x4(const float* JXL_RESTRICT pixels, float* JXL_RESTRICT coeffs);

}  // namespace jxl

#endif  // LIB_JXL_ENC_TRANSFORMS_H_
