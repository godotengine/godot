// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_NOISE_H_
#define LIB_JXL_NOISE_H_

// Noise parameters shared by encoder/decoder.

#include <stddef.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

const float kNoisePrecision = 1 << 10;

struct NoiseParams {
  // LUT index is an intensity of pixel / mean intensity of patch
  static constexpr size_t kNumNoisePoints = 8;
  using Lut = std::array<float, kNumNoisePoints>;

  Lut lut;

  void Clear() {
    for (float& i : lut) i = 0.f;
  }
  bool HasAny() const {
    for (float i : lut) {
      if (std::abs(i) > 1e-3f) return true;
    }
    return false;
  }
};

static inline std::pair<int, float> IndexAndFrac(float x) {
  constexpr size_t kScaleNumerator = NoiseParams::kNumNoisePoints - 2;
  // TODO(user): instead of 1, this should be a proper Y range.
  constexpr float kScale = kScaleNumerator / 1.0f;
  float scaled_x = std::max(0.f, x * kScale);
  float floor_x;
  float frac_x = std::modf(scaled_x, &floor_x);
  if (JXL_UNLIKELY(scaled_x >= kScaleNumerator + 1)) {
    floor_x = kScaleNumerator;
    frac_x = 1.f;
  }
  return std::make_pair(static_cast<int>(floor_x), frac_x);
}

struct NoiseLevel {
  float noise_level;
  float intensity;
};

}  // namespace jxl

#endif  // LIB_JXL_NOISE_H_
