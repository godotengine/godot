// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------

#ifndef WEBP_DSP_QUANT_H_
#define WEBP_DSP_QUANT_H_

#include <string.h>

#include "src/dsp/dsp.h"
#include "src/webp/types.h"

#if defined(WEBP_USE_NEON) && !defined(WEBP_ANDROID_NEON) && \
    !defined(WEBP_HAVE_NEON_RTCD)
#include <arm_neon.h>

#define IsFlat IsFlat_NEON

static uint32_t horizontal_add_uint32x4(const uint32x4_t a) {
#if WEBP_AARCH64
  return vaddvq_u32(a);
#else
  const uint64x2_t b = vpaddlq_u32(a);
  const uint32x2_t c = vadd_u32(vreinterpret_u32_u64(vget_low_u64(b)),
                                vreinterpret_u32_u64(vget_high_u64(b)));
  return vget_lane_u32(c, 0);
#endif
}

static WEBP_INLINE int IsFlat(const int16_t* levels, int num_blocks,
                              int thresh) {
  const int16x8_t tst_ones = vdupq_n_s16(-1);
  uint32x4_t sum = vdupq_n_u32(0);
  int i;

  for (i = 0; i < num_blocks; ++i) {
    // Set DC to zero.
    const int16x8_t a_0 = vsetq_lane_s16(0, vld1q_s16(levels), 0);
    const int16x8_t a_1 = vld1q_s16(levels + 8);

    const uint16x8_t b_0 = vshrq_n_u16(vtstq_s16(a_0, tst_ones), 15);
    const uint16x8_t b_1 = vshrq_n_u16(vtstq_s16(a_1, tst_ones), 15);

    sum = vpadalq_u16(sum, b_0);
    sum = vpadalq_u16(sum, b_1);

    levels += 16;
  }
  return thresh >= (int)horizontal_add_uint32x4(sum);
}

#else

#define IsFlat IsFlat_C

static WEBP_INLINE int IsFlat(const int16_t* levels, int num_blocks,
                              int thresh) {
  int score = 0;
  while (num_blocks-- > 0) {      // TODO(skal): refine positional scoring?
    int i;
    for (i = 1; i < 16; ++i) {    // omit DC, we're only interested in AC
      score += (levels[i] != 0);
      if (score > thresh) return 0;
    }
    levels += 16;
  }
  return 1;
}

#endif  // defined(WEBP_USE_NEON) && !defined(WEBP_ANDROID_NEON) &&
        // !defined(WEBP_HAVE_NEON_RTCD)

static WEBP_INLINE int IsFlatSource16(const uint8_t* src) {
  const uint32_t v = src[0] * 0x01010101u;
  int i;
  for (i = 0; i < 16; ++i) {
    if (memcmp(src + 0, &v, 4) || memcmp(src +  4, &v, 4) ||
        memcmp(src + 8, &v, 4) || memcmp(src + 12, &v, 4)) {
      return 0;
    }
    src += BPS;
  }
  return 1;
}

#endif  // WEBP_DSP_QUANT_H_
