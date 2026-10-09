// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_GAMMA_CORRECT_H_
#define LIB_JXL_ENC_GAMMA_CORRECT_H_

// Deprecated: sRGB transfer function. Use JxlCms instead.

#include <cmath>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

// Values are in [0, 1].
static JXL_INLINE double Srgb8ToLinearDirect(double srgb) {
  if (srgb <= 0.0) return 0.0;
  if (srgb <= 0.04045) return srgb / 12.92;
  if (srgb >= 1.0) return 1.0;
  return std::pow((srgb + 0.055) / 1.055, 2.4);
}

// Values are in [0, 1].
static JXL_INLINE double LinearToSrgb8Direct(double linear) {
  if (linear <= 0.0) return 0.0;
  if (linear >= 1.0) return 1.0;
  if (linear <= 0.0031308) return linear * 12.92;
  return std::pow(linear, 1.0 / 2.4) * 1.055 - 0.055;
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_GAMMA_CORRECT_H_
