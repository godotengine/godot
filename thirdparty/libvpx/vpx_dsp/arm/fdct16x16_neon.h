/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_FDCT16X16_NEON_H_
#define VPX_VPX_DSP_ARM_FDCT16X16_NEON_H_

#include <arm_neon.h>

#include "fdct_neon.h"

static INLINE void load(const int16_t *a, int stride, int16x8_t *b /*[16]*/) {
  b[0] = vld1q_s16(a);
  a += stride;
  b[1] = vld1q_s16(a);
  a += stride;
  b[2] = vld1q_s16(a);
  a += stride;
  b[3] = vld1q_s16(a);
  a += stride;
  b[4] = vld1q_s16(a);
  a += stride;
  b[5] = vld1q_s16(a);
  a += stride;
  b[6] = vld1q_s16(a);
  a += stride;
  b[7] = vld1q_s16(a);
  a += stride;
  b[8] = vld1q_s16(a);
  a += stride;
  b[9] = vld1q_s16(a);
  a += stride;
  b[10] = vld1q_s16(a);
  a += stride;
  b[11] = vld1q_s16(a);
  a += stride;
  b[12] = vld1q_s16(a);
  a += stride;
  b[13] = vld1q_s16(a);
  a += stride;
  b[14] = vld1q_s16(a);
  a += stride;
  b[15] = vld1q_s16(a);
}

// Store 8 16x8 values, assuming stride == 16.
static INLINE void store(tran_low_t *a, const int16x8_t *b /*[8]*/) {
  store_s16q_to_tran_low(a, b[0]);
  a += 16;
  store_s16q_to_tran_low(a, b[1]);
  a += 16;
  store_s16q_to_tran_low(a, b[2]);
  a += 16;
  store_s16q_to_tran_low(a, b[3]);
  a += 16;
  store_s16q_to_tran_low(a, b[4]);
  a += 16;
  store_s16q_to_tran_low(a, b[5]);
  a += 16;
  store_s16q_to_tran_low(a, b[6]);
  a += 16;
  store_s16q_to_tran_low(a, b[7]);
}

// Load step of each pass. Add and subtract clear across the input, requiring
// all 16 values to be loaded. For the first pass it also multiplies by 4.

// To maybe reduce register usage this could be combined with the load() step to
// get the first 4 and last 4 values, cross those, then load the middle 8 values
// and cross them.
static INLINE void scale_input(const int16x8_t *a /*[16]*/,
                               int16x8_t *b /*[16]*/) {
  b[0] = vshlq_n_s16(a[0], 2);
  b[1] = vshlq_n_s16(a[1], 2);
  b[2] = vshlq_n_s16(a[2], 2);
  b[3] = vshlq_n_s16(a[3], 2);
  b[4] = vshlq_n_s16(a[4], 2);
  b[5] = vshlq_n_s16(a[5], 2);
  b[6] = vshlq_n_s16(a[6], 2);
  b[7] = vshlq_n_s16(a[7], 2);

  b[8] = vshlq_n_s16(a[8], 2);
  b[9] = vshlq_n_s16(a[9], 2);
  b[10] = vshlq_n_s16(a[10], 2);
  b[11] = vshlq_n_s16(a[11], 2);
  b[12] = vshlq_n_s16(a[12], 2);
  b[13] = vshlq_n_s16(a[13], 2);
  b[14] = vshlq_n_s16(a[14], 2);
  b[15] = vshlq_n_s16(a[15], 2);
}

static INLINE void cross_input(const int16x8_t *a /*[16]*/,
                               int16x8_t *b /*[16]*/) {
  b[0] = vaddq_s16(a[0], a[15]);
  b[1] = vaddq_s16(a[1], a[14]);
  b[2] = vaddq_s16(a[2], a[13]);
  b[3] = vaddq_s16(a[3], a[12]);
  b[4] = vaddq_s16(a[4], a[11]);
  b[5] = vaddq_s16(a[5], a[10]);
  b[6] = vaddq_s16(a[6], a[9]);
  b[7] = vaddq_s16(a[7], a[8]);

  b[8] = vsubq_s16(a[7], a[8]);
  b[9] = vsubq_s16(a[6], a[9]);
  b[10] = vsubq_s16(a[5], a[10]);
  b[11] = vsubq_s16(a[4], a[11]);
  b[12] = vsubq_s16(a[3], a[12]);
  b[13] = vsubq_s16(a[2], a[13]);
  b[14] = vsubq_s16(a[1], a[14]);
  b[15] = vsubq_s16(a[0], a[15]);
}

static INLINE void load_cross(const int16_t *a, int stride,
                              int16x8_t *b /*[16]*/) {
  b[0] = vaddq_s16(vld1q_s16(a + 0 * stride), vld1q_s16(a + 15 * stride));
  b[1] = vaddq_s16(vld1q_s16(a + 1 * stride), vld1q_s16(a + 14 * stride));
  b[2] = vaddq_s16(vld1q_s16(a + 2 * stride), vld1q_s16(a + 13 * stride));
  b[3] = vaddq_s16(vld1q_s16(a + 3 * stride), vld1q_s16(a + 12 * stride));
  b[4] = vaddq_s16(vld1q_s16(a + 4 * stride), vld1q_s16(a + 11 * stride));
  b[5] = vaddq_s16(vld1q_s16(a + 5 * stride), vld1q_s16(a + 10 * stride));
  b[6] = vaddq_s16(vld1q_s16(a + 6 * stride), vld1q_s16(a + 9 * stride));
  b[7] = vaddq_s16(vld1q_s16(a + 7 * stride), vld1q_s16(a + 8 * stride));

  b[8] = vsubq_s16(vld1q_s16(a + 7 * stride), vld1q_s16(a + 8 * stride));
  b[9] = vsubq_s16(vld1q_s16(a + 6 * stride), vld1q_s16(a + 9 * stride));
  b[10] = vsubq_s16(vld1q_s16(a + 5 * stride), vld1q_s16(a + 10 * stride));
  b[11] = vsubq_s16(vld1q_s16(a + 4 * stride), vld1q_s16(a + 11 * stride));
  b[12] = vsubq_s16(vld1q_s16(a + 3 * stride), vld1q_s16(a + 12 * stride));
  b[13] = vsubq_s16(vld1q_s16(a + 2 * stride), vld1q_s16(a + 13 * stride));
  b[14] = vsubq_s16(vld1q_s16(a + 1 * stride), vld1q_s16(a + 14 * stride));
  b[15] = vsubq_s16(vld1q_s16(a + 0 * stride), vld1q_s16(a + 15 * stride));
}

// Quarter round at the beginning of the second pass. Can't use vrshr (rounding)
// because this only adds 1, not 1 << 2.
static INLINE void partial_round_shift(int16x8_t *a /*[16]*/) {
  const int16x8_t one = vdupq_n_s16(1);
  a[0] = vshrq_n_s16(vaddq_s16(a[0], one), 2);
  a[1] = vshrq_n_s16(vaddq_s16(a[1], one), 2);
  a[2] = vshrq_n_s16(vaddq_s16(a[2], one), 2);
  a[3] = vshrq_n_s16(vaddq_s16(a[3], one), 2);
  a[4] = vshrq_n_s16(vaddq_s16(a[4], one), 2);
  a[5] = vshrq_n_s16(vaddq_s16(a[5], one), 2);
  a[6] = vshrq_n_s16(vaddq_s16(a[6], one), 2);
  a[7] = vshrq_n_s16(vaddq_s16(a[7], one), 2);
  a[8] = vshrq_n_s16(vaddq_s16(a[8], one), 2);
  a[9] = vshrq_n_s16(vaddq_s16(a[9], one), 2);
  a[10] = vshrq_n_s16(vaddq_s16(a[10], one), 2);
  a[11] = vshrq_n_s16(vaddq_s16(a[11], one), 2);
  a[12] = vshrq_n_s16(vaddq_s16(a[12], one), 2);
  a[13] = vshrq_n_s16(vaddq_s16(a[13], one), 2);
  a[14] = vshrq_n_s16(vaddq_s16(a[14], one), 2);
  a[15] = vshrq_n_s16(vaddq_s16(a[15], one), 2);
}

#if CONFIG_VP9_HIGHBITDEPTH

static INLINE void highbd_scale_input(const int16x8_t *a /*[16]*/,
                                      int32x4_t *left /*[16]*/,
                                      int32x4_t *right /* [16] */) {
  left[0] = vshll_n_s16(vget_low_s16(a[0]), 2);
  left[1] = vshll_n_s16(vget_low_s16(a[1]), 2);
  left[2] = vshll_n_s16(vget_low_s16(a[2]), 2);
  left[3] = vshll_n_s16(vget_low_s16(a[3]), 2);
  left[4] = vshll_n_s16(vget_low_s16(a[4]), 2);
  left[5] = vshll_n_s16(vget_low_s16(a[5]), 2);
  left[6] = vshll_n_s16(vget_low_s16(a[6]), 2);
  left[7] = vshll_n_s16(vget_low_s16(a[7]), 2);
  left[8] = vshll_n_s16(vget_low_s16(a[8]), 2);
  left[9] = vshll_n_s16(vget_low_s16(a[9]), 2);
  left[10] = vshll_n_s16(vget_low_s16(a[10]), 2);
  left[11] = vshll_n_s16(vget_low_s16(a[11]), 2);
  left[12] = vshll_n_s16(vget_low_s16(a[12]), 2);
  left[13] = vshll_n_s16(vget_low_s16(a[13]), 2);
  left[14] = vshll_n_s16(vget_low_s16(a[14]), 2);
  left[15] = vshll_n_s16(vget_low_s16(a[15]), 2);

  right[0] = vshll_n_s16(vget_high_s16(a[0]), 2);
  right[1] = vshll_n_s16(vget_high_s16(a[1]), 2);
  right[2] = vshll_n_s16(vget_high_s16(a[2]), 2);
  right[3] = vshll_n_s16(vget_high_s16(a[3]), 2);
  right[4] = vshll_n_s16(vget_high_s16(a[4]), 2);
  right[5] = vshll_n_s16(vget_high_s16(a[5]), 2);
  right[6] = vshll_n_s16(vget_high_s16(a[6]), 2);
  right[7] = vshll_n_s16(vget_high_s16(a[7]), 2);
  right[8] = vshll_n_s16(vget_high_s16(a[8]), 2);
  right[9] = vshll_n_s16(vget_high_s16(a[9]), 2);
  right[10] = vshll_n_s16(vget_high_s16(a[10]), 2);
  right[11] = vshll_n_s16(vget_high_s16(a[11]), 2);
  right[12] = vshll_n_s16(vget_high_s16(a[12]), 2);
  right[13] = vshll_n_s16(vget_high_s16(a[13]), 2);
  right[14] = vshll_n_s16(vget_high_s16(a[14]), 2);
  right[15] = vshll_n_s16(vget_high_s16(a[15]), 2);
}

static INLINE void highbd_cross_input(const int32x4_t *a_left /*[16]*/,
                                      int32x4_t *a_right /*[16]*/,
                                      int32x4_t *b_left /*[16]*/,
                                      int32x4_t *b_right /*[16]*/) {
  b_left[0] = vaddq_s32(a_left[0], a_left[15]);
  b_left[1] = vaddq_s32(a_left[1], a_left[14]);
  b_left[2] = vaddq_s32(a_left[2], a_left[13]);
  b_left[3] = vaddq_s32(a_left[3], a_left[12]);
  b_left[4] = vaddq_s32(a_left[4], a_left[11]);
  b_left[5] = vaddq_s32(a_left[5], a_left[10]);
  b_left[6] = vaddq_s32(a_left[6], a_left[9]);
  b_left[7] = vaddq_s32(a_left[7], a_left[8]);

  b_right[0] = vaddq_s32(a_right[0], a_right[15]);
  b_right[1] = vaddq_s32(a_right[1], a_right[14]);
  b_right[2] = vaddq_s32(a_right[2], a_right[13]);
  b_right[3] = vaddq_s32(a_right[3], a_right[12]);
  b_right[4] = vaddq_s32(a_right[4], a_right[11]);
  b_right[5] = vaddq_s32(a_right[5], a_right[10]);
  b_right[6] = vaddq_s32(a_right[6], a_right[9]);
  b_right[7] = vaddq_s32(a_right[7], a_right[8]);

  b_left[8] = vsubq_s32(a_left[7], a_left[8]);
  b_left[9] = vsubq_s32(a_left[6], a_left[9]);
  b_left[10] = vsubq_s32(a_left[5], a_left[10]);
  b_left[11] = vsubq_s32(a_left[4], a_left[11]);
  b_left[12] = vsubq_s32(a_left[3], a_left[12]);
  b_left[13] = vsubq_s32(a_left[2], a_left[13]);
  b_left[14] = vsubq_s32(a_left[1], a_left[14]);
  b_left[15] = vsubq_s32(a_left[0], a_left[15]);

  b_right[8] = vsubq_s32(a_right[7], a_right[8]);
  b_right[9] = vsubq_s32(a_right[6], a_right[9]);
  b_right[10] = vsubq_s32(a_right[5], a_right[10]);
  b_right[11] = vsubq_s32(a_right[4], a_right[11]);
  b_right[12] = vsubq_s32(a_right[3], a_right[12]);
  b_right[13] = vsubq_s32(a_right[2], a_right[13]);
  b_right[14] = vsubq_s32(a_right[1], a_right[14]);
  b_right[15] = vsubq_s32(a_right[0], a_right[15]);
}

static INLINE void highbd_partial_round_shift(int32x4_t *left /*[16]*/,
                                              int32x4_t *right /* [16] */) {
  const int32x4_t one = vdupq_n_s32(1);
  left[0] = vshrq_n_s32(vaddq_s32(left[0], one), 2);
  left[1] = vshrq_n_s32(vaddq_s32(left[1], one), 2);
  left[2] = vshrq_n_s32(vaddq_s32(left[2], one), 2);
  left[3] = vshrq_n_s32(vaddq_s32(left[3], one), 2);
  left[4] = vshrq_n_s32(vaddq_s32(left[4], one), 2);
  left[5] = vshrq_n_s32(vaddq_s32(left[5], one), 2);
  left[6] = vshrq_n_s32(vaddq_s32(left[6], one), 2);
  left[7] = vshrq_n_s32(vaddq_s32(left[7], one), 2);
  left[8] = vshrq_n_s32(vaddq_s32(left[8], one), 2);
  left[9] = vshrq_n_s32(vaddq_s32(left[9], one), 2);
  left[10] = vshrq_n_s32(vaddq_s32(left[10], one), 2);
  left[11] = vshrq_n_s32(vaddq_s32(left[11], one), 2);
  left[12] = vshrq_n_s32(vaddq_s32(left[12], one), 2);
  left[13] = vshrq_n_s32(vaddq_s32(left[13], one), 2);
  left[14] = vshrq_n_s32(vaddq_s32(left[14], one), 2);
  left[15] = vshrq_n_s32(vaddq_s32(left[15], one), 2);

  right[0] = vshrq_n_s32(vaddq_s32(right[0], one), 2);
  right[1] = vshrq_n_s32(vaddq_s32(right[1], one), 2);
  right[2] = vshrq_n_s32(vaddq_s32(right[2], one), 2);
  right[3] = vshrq_n_s32(vaddq_s32(right[3], one), 2);
  right[4] = vshrq_n_s32(vaddq_s32(right[4], one), 2);
  right[5] = vshrq_n_s32(vaddq_s32(right[5], one), 2);
  right[6] = vshrq_n_s32(vaddq_s32(right[6], one), 2);
  right[7] = vshrq_n_s32(vaddq_s32(right[7], one), 2);
  right[8] = vshrq_n_s32(vaddq_s32(right[8], one), 2);
  right[9] = vshrq_n_s32(vaddq_s32(right[9], one), 2);
  right[10] = vshrq_n_s32(vaddq_s32(right[10], one), 2);
  right[11] = vshrq_n_s32(vaddq_s32(right[11], one), 2);
  right[12] = vshrq_n_s32(vaddq_s32(right[12], one), 2);
  right[13] = vshrq_n_s32(vaddq_s32(right[13], one), 2);
  right[14] = vshrq_n_s32(vaddq_s32(right[14], one), 2);
  right[15] = vshrq_n_s32(vaddq_s32(right[15], one), 2);
}

// Store 16 32x4 vectors, assuming stride == 16.
static INLINE void store16_s32(tran_low_t *a, const int32x4_t *b /*[32]*/) {
  vst1q_s32(a, b[0]);
  a += 16;
  vst1q_s32(a, b[1]);
  a += 16;
  vst1q_s32(a, b[2]);
  a += 16;
  vst1q_s32(a, b[3]);
  a += 16;
  vst1q_s32(a, b[4]);
  a += 16;
  vst1q_s32(a, b[5]);
  a += 16;
  vst1q_s32(a, b[6]);
  a += 16;
  vst1q_s32(a, b[7]);
  a += 16;
  vst1q_s32(a, b[8]);
  a += 16;
  vst1q_s32(a, b[9]);
  a += 16;
  vst1q_s32(a, b[10]);
  a += 16;
  vst1q_s32(a, b[11]);
  a += 16;
  vst1q_s32(a, b[12]);
  a += 16;
  vst1q_s32(a, b[13]);
  a += 16;
  vst1q_s32(a, b[14]);
  a += 16;
  vst1q_s32(a, b[15]);
}

#endif  // CONFIG_VP9_HIGHBITDEPTH

#endif  // VPX_VPX_DSP_ARM_FDCT16X16_NEON_H_
