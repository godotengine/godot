// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/alpha.h"

#include <string.h>

#include <algorithm>

namespace jxl {

static float Clamp(float x) { return std::max(std::min(1.0f, x), 0.0f); }

void PerformAlphaBlending(const AlphaBlendingInputLayer& bg,
                          const AlphaBlendingInputLayer& fg,
                          const AlphaBlendingOutput& out, size_t num_pixels,
                          bool alpha_is_premultiplied, bool clamp) {
  if (alpha_is_premultiplied) {
    for (size_t x = 0; x < num_pixels; ++x) {
      float fga = clamp ? Clamp(fg.a[x]) : fg.a[x];
      out.r[x] = (fg.r[x] + bg.r[x] * (1.f - fga));
      out.g[x] = (fg.g[x] + bg.g[x] * (1.f - fga));
      out.b[x] = (fg.b[x] + bg.b[x] * (1.f - fga));
      out.a[x] = (1.f - (1.f - fga) * (1.f - bg.a[x]));
    }
  } else {
    for (size_t x = 0; x < num_pixels; ++x) {
      float fga = clamp ? Clamp(fg.a[x]) : fg.a[x];
      const float new_a = 1.f - (1.f - fga) * (1.f - bg.a[x]);
      const float rnew_a = (new_a > 0 ? 1.f / new_a : 0.f);
      out.r[x] = (fg.r[x] * fga + bg.r[x] * bg.a[x] * (1.f - fga)) * rnew_a;
      out.g[x] = (fg.g[x] * fga + bg.g[x] * bg.a[x] * (1.f - fga)) * rnew_a;
      out.b[x] = (fg.b[x] * fga + bg.b[x] * bg.a[x] * (1.f - fga)) * rnew_a;
      out.a[x] = new_a;
    }
  }
}
void PerformAlphaBlending(const float* bg, const float* bga, const float* fg,
                          const float* fga, float* out, size_t num_pixels,
                          bool alpha_is_premultiplied, bool clamp) {
  if (bg == bga && fg == fga) {
    for (size_t x = 0; x < num_pixels; ++x) {
      float fa = clamp ? fga[x] : Clamp(fga[x]);
      out[x] = (1.f - (1.f - fa) * (1.f - bga[x]));
    }
  } else {
    if (alpha_is_premultiplied) {
      for (size_t x = 0; x < num_pixels; ++x) {
        float fa = clamp ? fga[x] : Clamp(fga[x]);
        out[x] = (fg[x] + bg[x] * (1.f - fa));
      }
    } else {
      for (size_t x = 0; x < num_pixels; ++x) {
        float fa = clamp ? fga[x] : Clamp(fga[x]);
        const float new_a = 1.f - (1.f - fa) * (1.f - bga[x]);
        const float rnew_a = (new_a > 0 ? 1.f / new_a : 0.f);
        out[x] = (fg[x] * fa + bg[x] * bga[x] * (1.f - fa)) * rnew_a;
      }
    }
  }
}

void PerformAlphaWeightedAdd(const float* bg, const float* fg, const float* fga,
                             float* out, size_t num_pixels, bool clamp) {
  if (fg == fga) {
    memcpy(out, bg, num_pixels * sizeof(*out));
  } else if (clamp) {
    for (size_t x = 0; x < num_pixels; ++x) {
      out[x] = bg[x] + fg[x] * Clamp(fga[x]);
    }
  } else {
    for (size_t x = 0; x < num_pixels; ++x) {
      out[x] = bg[x] + fg[x] * fga[x];
    }
  }
}

void PerformMulBlending(const float* bg, const float* fg, float* out,
                        size_t num_pixels, bool clamp) {
  if (clamp) {
    for (size_t x = 0; x < num_pixels; ++x) {
      out[x] = bg[x] * Clamp(fg[x]);
    }
  } else {
    for (size_t x = 0; x < num_pixels; ++x) {
      out[x] = bg[x] * fg[x];
    }
  }
}

void PremultiplyAlpha(float* JXL_RESTRICT r, float* JXL_RESTRICT g,
                      float* JXL_RESTRICT b, const float* JXL_RESTRICT a,
                      size_t num_pixels) {
  for (size_t x = 0; x < num_pixels; ++x) {
    const float multiplier = std::max(kSmallAlpha, a[x]);
    r[x] *= multiplier;
    g[x] *= multiplier;
    b[x] *= multiplier;
  }
}

void UnpremultiplyAlpha(float* JXL_RESTRICT r, float* JXL_RESTRICT g,
                        float* JXL_RESTRICT b, const float* JXL_RESTRICT a,
                        size_t num_pixels) {
  for (size_t x = 0; x < num_pixels; ++x) {
    const float multiplier = 1.f / std::max(kSmallAlpha, a[x]);
    r[x] *= multiplier;
    g[x] *= multiplier;
    b[x] *= multiplier;
  }
}

}  // namespace jxl
