// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Gamma correction utilities.

#include "sharpyuv/sharpyuv_gamma.h"

#include <assert.h>
#include <math.h>

#include "src/webp/types.h"

// Gamma correction compensates loss of resolution during chroma subsampling.
// Size of pre-computed table for converting from gamma to linear.
#define GAMMA_TO_LINEAR_TAB_BITS 10
#define GAMMA_TO_LINEAR_TAB_SIZE (1 << GAMMA_TO_LINEAR_TAB_BITS)
static uint32_t kGammaToLinearTabS[GAMMA_TO_LINEAR_TAB_SIZE + 2];
#define LINEAR_TO_GAMMA_TAB_BITS 9
#define LINEAR_TO_GAMMA_TAB_SIZE (1 << LINEAR_TO_GAMMA_TAB_BITS)
static uint32_t kLinearToGammaTabS[LINEAR_TO_GAMMA_TAB_SIZE + 2];

static const double kGammaF = 1. / 0.45;
#define GAMMA_TO_LINEAR_BITS 16

static volatile int kGammaTablesSOk = 0;
void SharpYuvInitGammaTables(void) {
  assert(GAMMA_TO_LINEAR_BITS <= 16);
  if (!kGammaTablesSOk) {
    int v;
    const double a = 0.09929682680944;
    const double thresh = 0.018053968510807;
    const double final_scale = 1 << GAMMA_TO_LINEAR_BITS;
    // Precompute gamma to linear table.
    {
      const double norm = 1. / GAMMA_TO_LINEAR_TAB_SIZE;
      const double a_rec = 1. / (1. + a);
      for (v = 0; v <= GAMMA_TO_LINEAR_TAB_SIZE; ++v) {
        const double g = norm * v;
        double value;
        if (g <= thresh * 4.5) {
          value = g / 4.5;
        } else {
          value = pow(a_rec * (g + a), kGammaF);
        }
        kGammaToLinearTabS[v] = (uint32_t)(value * final_scale + .5);
      }
      // to prevent small rounding errors to cause read-overflow:
      kGammaToLinearTabS[GAMMA_TO_LINEAR_TAB_SIZE + 1] =
          kGammaToLinearTabS[GAMMA_TO_LINEAR_TAB_SIZE];
    }
    // Precompute linear to gamma table.
    {
      const double scale = 1. / LINEAR_TO_GAMMA_TAB_SIZE;
      for (v = 0; v <= LINEAR_TO_GAMMA_TAB_SIZE; ++v) {
        const double g = scale * v;
        double value;
        if (g <= thresh) {
          value = 4.5 * g;
        } else {
          value = (1. + a) * pow(g, 1. / kGammaF) - a;
        }
        kLinearToGammaTabS[v] =
            (uint32_t)(final_scale * value + 0.5);
      }
      // to prevent small rounding errors to cause read-overflow:
      kLinearToGammaTabS[LINEAR_TO_GAMMA_TAB_SIZE + 1] =
          kLinearToGammaTabS[LINEAR_TO_GAMMA_TAB_SIZE];
    }
    kGammaTablesSOk = 1;
  }
}

static WEBP_INLINE int Shift(int v, int shift) {
  return (shift >= 0) ? (v << shift) : (v >> -shift);
}

static WEBP_INLINE uint32_t FixedPointInterpolation(int v, uint32_t* tab,
                                                    int tab_pos_shift_right,
                                                    int tab_value_shift) {
  const uint32_t tab_pos = Shift(v, -tab_pos_shift_right);
  // fractional part, in 'tab_pos_shift' fixed-point precision
  const uint32_t x = v - (tab_pos << tab_pos_shift_right);  // fractional part
  // v0 / v1 are in kGammaToLinearBits fixed-point precision (range [0..1])
  const uint32_t v0 = Shift(tab[tab_pos + 0], tab_value_shift);
  const uint32_t v1 = Shift(tab[tab_pos + 1], tab_value_shift);
  // Final interpolation.
  const uint32_t v2 = (v1 - v0) * x;  // note: v1 >= v0.
  const int half =
      (tab_pos_shift_right > 0) ? 1 << (tab_pos_shift_right - 1) : 0;
  const uint32_t result = v0 + ((v2 + half) >> tab_pos_shift_right);
  return result;
}

uint32_t SharpYuvGammaToLinear(uint16_t v, int bit_depth) {
  const int shift = GAMMA_TO_LINEAR_TAB_BITS - bit_depth;
  if (shift > 0) {
    return kGammaToLinearTabS[v << shift];
  }
  return FixedPointInterpolation(v, kGammaToLinearTabS, -shift, 0);
}

uint16_t SharpYuvLinearToGamma(uint32_t value, int bit_depth) {
  return FixedPointInterpolation(
      value, kLinearToGammaTabS,
      (GAMMA_TO_LINEAR_BITS - LINEAR_TO_GAMMA_TAB_BITS),
      bit_depth - GAMMA_TO_LINEAR_BITS);
}
