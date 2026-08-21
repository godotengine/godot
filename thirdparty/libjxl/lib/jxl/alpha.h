// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ALPHA_H_
#define LIB_JXL_ALPHA_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

// A very small value to avoid divisions by zero when converting to
// unpremultiplied alpha. Page 21 of the technical introduction to OpenEXR
// (https://www.openexr.com/documentation/TechnicalIntroduction.pdf) recommends
// "a power of two" that is "less than half of the smallest positive 16-bit
// floating-point value". That smallest value happens to be the denormal number
// 2^-24, so 2^-26 should be a good choice.
static constexpr float kSmallAlpha = 1.f / (1u << 26u);

struct AlphaBlendingInputLayer {
  const float* r;
  const float* g;
  const float* b;
  const float* a;
};

struct AlphaBlendingOutput {
  float* r;
  float* g;
  float* b;
  float* a;
};

// Note: The pointers in `out` are allowed to alias those in `bg` or `fg`.
// No pointer shall be null.
void PerformAlphaBlending(const AlphaBlendingInputLayer& bg,
                          const AlphaBlendingInputLayer& fg,
                          const AlphaBlendingOutput& out, size_t num_pixels,
                          bool alpha_is_premultiplied, bool clamp);
// Single plane alpha blending
void PerformAlphaBlending(const float* bg, const float* bga, const float* fg,
                          const float* fga, float* out, size_t num_pixels,
                          bool alpha_is_premultiplied, bool clamp);

void PerformAlphaWeightedAdd(const float* bg, const float* fg, const float* fga,
                             float* out, size_t num_pixels, bool clamp);

void PerformMulBlending(const float* bg, const float* fg, float* out,
                        size_t num_pixels, bool clamp);

void PremultiplyAlpha(float* JXL_RESTRICT r, float* JXL_RESTRICT g,
                      float* JXL_RESTRICT b, const float* JXL_RESTRICT a,
                      size_t num_pixels);
void UnpremultiplyAlpha(float* JXL_RESTRICT r, float* JXL_RESTRICT g,
                        float* JXL_RESTRICT b, const float* JXL_RESTRICT a,
                        size_t num_pixels);

}  // namespace jxl

#endif  // LIB_JXL_ALPHA_H_
