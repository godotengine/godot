/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_HIGHBD_IDCT_NEON_H_
#define VPX_VPX_DSP_ARM_HIGHBD_IDCT_NEON_H_

#include <arm_neon.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/inv_txfm.h"

static INLINE void highbd_idct4x4_1_add_kernel1(uint16_t **dest,
                                                const int stride,
                                                const int16x8_t res,
                                                const int16x8_t max) {
  const uint16x4_t a0 = vld1_u16(*dest);
  const uint16x4_t a1 = vld1_u16(*dest + stride);
  const int16x8_t a = vreinterpretq_s16_u16(vcombine_u16(a0, a1));
  // Note: In some profile tests, res is quite close to +/-32767.
  // We use saturating addition.
  const int16x8_t b = vqaddq_s16(res, a);
  const int16x8_t c = vminq_s16(b, max);
  const uint16x8_t d = vqshluq_n_s16(c, 0);
  vst1_u16(*dest, vget_low_u16(d));
  *dest += stride;
  vst1_u16(*dest, vget_high_u16(d));
  *dest += stride;
}

static INLINE void idct4x4_16_kernel_bd10(const int32x4_t cospis,
                                          int32x4_t *const a) {
  int32x4_t b0, b1, b2, b3;

  transpose_s32_4x4(&a[0], &a[1], &a[2], &a[3]);
  b0 = vaddq_s32(a[0], a[2]);
  b1 = vsubq_s32(a[0], a[2]);
  b0 = vmulq_lane_s32(b0, vget_high_s32(cospis), 0);
  b1 = vmulq_lane_s32(b1, vget_high_s32(cospis), 0);
  b2 = vmulq_lane_s32(a[1], vget_high_s32(cospis), 1);
  b3 = vmulq_lane_s32(a[1], vget_low_s32(cospis), 1);
  b2 = vmlsq_lane_s32(b2, a[3], vget_low_s32(cospis), 1);
  b3 = vmlaq_lane_s32(b3, a[3], vget_high_s32(cospis), 1);
  b0 = vrshrq_n_s32(b0, DCT_CONST_BITS);
  b1 = vrshrq_n_s32(b1, DCT_CONST_BITS);
  b2 = vrshrq_n_s32(b2, DCT_CONST_BITS);
  b3 = vrshrq_n_s32(b3, DCT_CONST_BITS);
  a[0] = vaddq_s32(b0, b3);
  a[1] = vaddq_s32(b1, b2);
  a[2] = vsubq_s32(b1, b2);
  a[3] = vsubq_s32(b0, b3);
}

static INLINE void idct4x4_16_kernel_bd12(const int32x4_t cospis,
                                          int32x4_t *const a) {
  int32x4_t b0, b1, b2, b3;
  int64x2_t c[12];

  transpose_s32_4x4(&a[0], &a[1], &a[2], &a[3]);
  b0 = vaddq_s32(a[0], a[2]);
  b1 = vsubq_s32(a[0], a[2]);
  c[0] = vmull_lane_s32(vget_low_s32(b0), vget_high_s32(cospis), 0);
  c[1] = vmull_lane_s32(vget_high_s32(b0), vget_high_s32(cospis), 0);
  c[2] = vmull_lane_s32(vget_low_s32(b1), vget_high_s32(cospis), 0);
  c[3] = vmull_lane_s32(vget_high_s32(b1), vget_high_s32(cospis), 0);
  c[4] = vmull_lane_s32(vget_low_s32(a[1]), vget_high_s32(cospis), 1);
  c[5] = vmull_lane_s32(vget_high_s32(a[1]), vget_high_s32(cospis), 1);
  c[6] = vmull_lane_s32(vget_low_s32(a[1]), vget_low_s32(cospis), 1);
  c[7] = vmull_lane_s32(vget_high_s32(a[1]), vget_low_s32(cospis), 1);
  c[8] = vmull_lane_s32(vget_low_s32(a[3]), vget_low_s32(cospis), 1);
  c[9] = vmull_lane_s32(vget_high_s32(a[3]), vget_low_s32(cospis), 1);
  c[10] = vmull_lane_s32(vget_low_s32(a[3]), vget_high_s32(cospis), 1);
  c[11] = vmull_lane_s32(vget_high_s32(a[3]), vget_high_s32(cospis), 1);
  c[4] = vsubq_s64(c[4], c[8]);
  c[5] = vsubq_s64(c[5], c[9]);
  c[6] = vaddq_s64(c[6], c[10]);
  c[7] = vaddq_s64(c[7], c[11]);
  b0 = vcombine_s32(vrshrn_n_s64(c[0], DCT_CONST_BITS),
                    vrshrn_n_s64(c[1], DCT_CONST_BITS));
  b1 = vcombine_s32(vrshrn_n_s64(c[2], DCT_CONST_BITS),
                    vrshrn_n_s64(c[3], DCT_CONST_BITS));
  b2 = vcombine_s32(vrshrn_n_s64(c[4], DCT_CONST_BITS),
                    vrshrn_n_s64(c[5], DCT_CONST_BITS));
  b3 = vcombine_s32(vrshrn_n_s64(c[6], DCT_CONST_BITS),
                    vrshrn_n_s64(c[7], DCT_CONST_BITS));
  a[0] = vaddq_s32(b0, b3);
  a[1] = vaddq_s32(b1, b2);
  a[2] = vsubq_s32(b1, b2);
  a[3] = vsubq_s32(b0, b3);
}

static INLINE void highbd_add8x8(int16x8_t *const a, uint16_t *dest,
                                 const int stride, const int bd) {
  const int16x8_t max = vdupq_n_s16((1 << bd) - 1);
  const uint16_t *dst = dest;
  uint16x8_t d0, d1, d2, d3, d4, d5, d6, d7;
  uint16x8_t d0_u16, d1_u16, d2_u16, d3_u16, d4_u16, d5_u16, d6_u16, d7_u16;
  int16x8_t d0_s16, d1_s16, d2_s16, d3_s16, d4_s16, d5_s16, d6_s16, d7_s16;

  d0 = vld1q_u16(dst);
  dst += stride;
  d1 = vld1q_u16(dst);
  dst += stride;
  d2 = vld1q_u16(dst);
  dst += stride;
  d3 = vld1q_u16(dst);
  dst += stride;
  d4 = vld1q_u16(dst);
  dst += stride;
  d5 = vld1q_u16(dst);
  dst += stride;
  d6 = vld1q_u16(dst);
  dst += stride;
  d7 = vld1q_u16(dst);

  d0_s16 = vqaddq_s16(a[0], vreinterpretq_s16_u16(d0));
  d1_s16 = vqaddq_s16(a[1], vreinterpretq_s16_u16(d1));
  d2_s16 = vqaddq_s16(a[2], vreinterpretq_s16_u16(d2));
  d3_s16 = vqaddq_s16(a[3], vreinterpretq_s16_u16(d3));
  d4_s16 = vqaddq_s16(a[4], vreinterpretq_s16_u16(d4));
  d5_s16 = vqaddq_s16(a[5], vreinterpretq_s16_u16(d5));
  d6_s16 = vqaddq_s16(a[6], vreinterpretq_s16_u16(d6));
  d7_s16 = vqaddq_s16(a[7], vreinterpretq_s16_u16(d7));

  d0_s16 = vminq_s16(d0_s16, max);
  d1_s16 = vminq_s16(d1_s16, max);
  d2_s16 = vminq_s16(d2_s16, max);
  d3_s16 = vminq_s16(d3_s16, max);
  d4_s16 = vminq_s16(d4_s16, max);
  d5_s16 = vminq_s16(d5_s16, max);
  d6_s16 = vminq_s16(d6_s16, max);
  d7_s16 = vminq_s16(d7_s16, max);
  d0_u16 = vqshluq_n_s16(d0_s16, 0);
  d1_u16 = vqshluq_n_s16(d1_s16, 0);
  d2_u16 = vqshluq_n_s16(d2_s16, 0);
  d3_u16 = vqshluq_n_s16(d3_s16, 0);
  d4_u16 = vqshluq_n_s16(d4_s16, 0);
  d5_u16 = vqshluq_n_s16(d5_s16, 0);
  d6_u16 = vqshluq_n_s16(d6_s16, 0);
  d7_u16 = vqshluq_n_s16(d7_s16, 0);

  vst1q_u16(dest, d0_u16);
  dest += stride;
  vst1q_u16(dest, d1_u16);
  dest += stride;
  vst1q_u16(dest, d2_u16);
  dest += stride;
  vst1q_u16(dest, d3_u16);
  dest += stride;
  vst1q_u16(dest, d4_u16);
  dest += stride;
  vst1q_u16(dest, d5_u16);
  dest += stride;
  vst1q_u16(dest, d6_u16);
  dest += stride;
  vst1q_u16(dest, d7_u16);
}

static INLINE void idct8x8_64_half1d_bd10(
    const int32x4_t cospis0, const int32x4_t cospis1, int32x4_t *const io0,
    int32x4_t *const io1, int32x4_t *const io2, int32x4_t *const io3,
    int32x4_t *const io4, int32x4_t *const io5, int32x4_t *const io6,
    int32x4_t *const io7) {
  int32x4_t step1[8], step2[8];

  transpose_s32_8x4(io0, io1, io2, io3, io4, io5, io6, io7);

  // stage 1
  step1[4] = vmulq_lane_s32(*io1, vget_high_s32(cospis1), 1);
  step1[5] = vmulq_lane_s32(*io3, vget_high_s32(cospis1), 0);
  step1[6] = vmulq_lane_s32(*io3, vget_low_s32(cospis1), 1);
  step1[7] = vmulq_lane_s32(*io1, vget_low_s32(cospis1), 0);

  step1[4] = vmlsq_lane_s32(step1[4], *io7, vget_low_s32(cospis1), 0);
  step1[5] = vmlaq_lane_s32(step1[5], *io5, vget_low_s32(cospis1), 1);
  step1[6] = vmlsq_lane_s32(step1[6], *io5, vget_high_s32(cospis1), 0);
  step1[7] = vmlaq_lane_s32(step1[7], *io7, vget_high_s32(cospis1), 1);

  step1[4] = vrshrq_n_s32(step1[4], DCT_CONST_BITS);
  step1[5] = vrshrq_n_s32(step1[5], DCT_CONST_BITS);
  step1[6] = vrshrq_n_s32(step1[6], DCT_CONST_BITS);
  step1[7] = vrshrq_n_s32(step1[7], DCT_CONST_BITS);

  // stage 2
  step2[1] = vmulq_lane_s32(*io0, vget_high_s32(cospis0), 0);
  step2[2] = vmulq_lane_s32(*io2, vget_high_s32(cospis0), 1);
  step2[3] = vmulq_lane_s32(*io2, vget_low_s32(cospis0), 1);

  step2[0] = vmlaq_lane_s32(step2[1], *io4, vget_high_s32(cospis0), 0);
  step2[1] = vmlsq_lane_s32(step2[1], *io4, vget_high_s32(cospis0), 0);
  step2[2] = vmlsq_lane_s32(step2[2], *io6, vget_low_s32(cospis0), 1);
  step2[3] = vmlaq_lane_s32(step2[3], *io6, vget_high_s32(cospis0), 1);

  step2[0] = vrshrq_n_s32(step2[0], DCT_CONST_BITS);
  step2[1] = vrshrq_n_s32(step2[1], DCT_CONST_BITS);
  step2[2] = vrshrq_n_s32(step2[2], DCT_CONST_BITS);
  step2[3] = vrshrq_n_s32(step2[3], DCT_CONST_BITS);

  step2[4] = vaddq_s32(step1[4], step1[5]);
  step2[5] = vsubq_s32(step1[4], step1[5]);
  step2[6] = vsubq_s32(step1[7], step1[6]);
  step2[7] = vaddq_s32(step1[7], step1[6]);

  // stage 3
  step1[0] = vaddq_s32(step2[0], step2[3]);
  step1[1] = vaddq_s32(step2[1], step2[2]);
  step1[2] = vsubq_s32(step2[1], step2[2]);
  step1[3] = vsubq_s32(step2[0], step2[3]);

  step1[6] = vmulq_lane_s32(step2[6], vget_high_s32(cospis0), 0);
  step1[5] = vmlsq_lane_s32(step1[6], step2[5], vget_high_s32(cospis0), 0);
  step1[6] = vmlaq_lane_s32(step1[6], step2[5], vget_high_s32(cospis0), 0);
  step1[5] = vrshrq_n_s32(step1[5], DCT_CONST_BITS);
  step1[6] = vrshrq_n_s32(step1[6], DCT_CONST_BITS);

  // stage 4
  *io0 = vaddq_s32(step1[0], step2[7]);
  *io1 = vaddq_s32(step1[1], step1[6]);
  *io2 = vaddq_s32(step1[2], step1[5]);
  *io3 = vaddq_s32(step1[3], step2[4]);
  *io4 = vsubq_s32(step1[3], step2[4]);
  *io5 = vsubq_s32(step1[2], step1[5]);
  *io6 = vsubq_s32(step1[1], step1[6]);
  *io7 = vsubq_s32(step1[0], step2[7]);
}

static INLINE void idct8x8_64_half1d_bd12(
    const int32x4_t cospis0, const int32x4_t cospis1, int32x4_t *const io0,
    int32x4_t *const io1, int32x4_t *const io2, int32x4_t *const io3,
    int32x4_t *const io4, int32x4_t *const io5, int32x4_t *const io6,
    int32x4_t *const io7) {
  int32x2_t input1l, input1h, input3l, input3h, input5l, input5h, input7l,
      input7h;
  int32x2_t step1l[4], step1h[4];
  int32x4_t step1[8], step2[8];
  int64x2_t t64[8];
  int32x2_t t32[8];

  transpose_s32_8x4(io0, io1, io2, io3, io4, io5, io6, io7);

  // stage 1
  input1l = vget_low_s32(*io1);
  input1h = vget_high_s32(*io1);
  input3l = vget_low_s32(*io3);
  input3h = vget_high_s32(*io3);
  input5l = vget_low_s32(*io5);
  input5h = vget_high_s32(*io5);
  input7l = vget_low_s32(*io7);
  input7h = vget_high_s32(*io7);
  step1l[0] = vget_low_s32(*io0);
  step1h[0] = vget_high_s32(*io0);
  step1l[1] = vget_low_s32(*io2);
  step1h[1] = vget_high_s32(*io2);
  step1l[2] = vget_low_s32(*io4);
  step1h[2] = vget_high_s32(*io4);
  step1l[3] = vget_low_s32(*io6);
  step1h[3] = vget_high_s32(*io6);

  t64[0] = vmull_lane_s32(input1l, vget_high_s32(cospis1), 1);
  t64[1] = vmull_lane_s32(input1h, vget_high_s32(cospis1), 1);
  t64[2] = vmull_lane_s32(input3l, vget_high_s32(cospis1), 0);
  t64[3] = vmull_lane_s32(input3h, vget_high_s32(cospis1), 0);
  t64[4] = vmull_lane_s32(input3l, vget_low_s32(cospis1), 1);
  t64[5] = vmull_lane_s32(input3h, vget_low_s32(cospis1), 1);
  t64[6] = vmull_lane_s32(input1l, vget_low_s32(cospis1), 0);
  t64[7] = vmull_lane_s32(input1h, vget_low_s32(cospis1), 0);
  t64[0] = vmlsl_lane_s32(t64[0], input7l, vget_low_s32(cospis1), 0);
  t64[1] = vmlsl_lane_s32(t64[1], input7h, vget_low_s32(cospis1), 0);
  t64[2] = vmlal_lane_s32(t64[2], input5l, vget_low_s32(cospis1), 1);
  t64[3] = vmlal_lane_s32(t64[3], input5h, vget_low_s32(cospis1), 1);
  t64[4] = vmlsl_lane_s32(t64[4], input5l, vget_high_s32(cospis1), 0);
  t64[5] = vmlsl_lane_s32(t64[5], input5h, vget_high_s32(cospis1), 0);
  t64[6] = vmlal_lane_s32(t64[6], input7l, vget_high_s32(cospis1), 1);
  t64[7] = vmlal_lane_s32(t64[7], input7h, vget_high_s32(cospis1), 1);
  t32[0] = vrshrn_n_s64(t64[0], DCT_CONST_BITS);
  t32[1] = vrshrn_n_s64(t64[1], DCT_CONST_BITS);
  t32[2] = vrshrn_n_s64(t64[2], DCT_CONST_BITS);
  t32[3] = vrshrn_n_s64(t64[3], DCT_CONST_BITS);
  t32[4] = vrshrn_n_s64(t64[4], DCT_CONST_BITS);
  t32[5] = vrshrn_n_s64(t64[5], DCT_CONST_BITS);
  t32[6] = vrshrn_n_s64(t64[6], DCT_CONST_BITS);
  t32[7] = vrshrn_n_s64(t64[7], DCT_CONST_BITS);
  step1[4] = vcombine_s32(t32[0], t32[1]);
  step1[5] = vcombine_s32(t32[2], t32[3]);
  step1[6] = vcombine_s32(t32[4], t32[5]);
  step1[7] = vcombine_s32(t32[6], t32[7]);

  // stage 2
  t64[2] = vmull_lane_s32(step1l[0], vget_high_s32(cospis0), 0);
  t64[3] = vmull_lane_s32(step1h[0], vget_high_s32(cospis0), 0);
  t64[4] = vmull_lane_s32(step1l[1], vget_high_s32(cospis0), 1);
  t64[5] = vmull_lane_s32(step1h[1], vget_high_s32(cospis0), 1);
  t64[6] = vmull_lane_s32(step1l[1], vget_low_s32(cospis0), 1);
  t64[7] = vmull_lane_s32(step1h[1], vget_low_s32(cospis0), 1);
  t64[0] = vmlal_lane_s32(t64[2], step1l[2], vget_high_s32(cospis0), 0);
  t64[1] = vmlal_lane_s32(t64[3], step1h[2], vget_high_s32(cospis0), 0);
  t64[2] = vmlsl_lane_s32(t64[2], step1l[2], vget_high_s32(cospis0), 0);
  t64[3] = vmlsl_lane_s32(t64[3], step1h[2], vget_high_s32(cospis0), 0);
  t64[4] = vmlsl_lane_s32(t64[4], step1l[3], vget_low_s32(cospis0), 1);
  t64[5] = vmlsl_lane_s32(t64[5], step1h[3], vget_low_s32(cospis0), 1);
  t64[6] = vmlal_lane_s32(t64[6], step1l[3], vget_high_s32(cospis0), 1);
  t64[7] = vmlal_lane_s32(t64[7], step1h[3], vget_high_s32(cospis0), 1);
  t32[0] = vrshrn_n_s64(t64[0], DCT_CONST_BITS);
  t32[1] = vrshrn_n_s64(t64[1], DCT_CONST_BITS);
  t32[2] = vrshrn_n_s64(t64[2], DCT_CONST_BITS);
  t32[3] = vrshrn_n_s64(t64[3], DCT_CONST_BITS);
  t32[4] = vrshrn_n_s64(t64[4], DCT_CONST_BITS);
  t32[5] = vrshrn_n_s64(t64[5], DCT_CONST_BITS);
  t32[6] = vrshrn_n_s64(t64[6], DCT_CONST_BITS);
  t32[7] = vrshrn_n_s64(t64[7], DCT_CONST_BITS);
  step2[0] = vcombine_s32(t32[0], t32[1]);
  step2[1] = vcombine_s32(t32[2], t32[3]);
  step2[2] = vcombine_s32(t32[4], t32[5]);
  step2[3] = vcombine_s32(t32[6], t32[7]);

  step2[4] = vaddq_s32(step1[4], step1[5]);
  step2[5] = vsubq_s32(step1[4], step1[5]);
  step2[6] = vsubq_s32(step1[7], step1[6]);
  step2[7] = vaddq_s32(step1[7], step1[6]);

  // stage 3
  step1[0] = vaddq_s32(step2[0], step2[3]);
  step1[1] = vaddq_s32(step2[1], step2[2]);
  step1[2] = vsubq_s32(step2[1], step2[2]);
  step1[3] = vsubq_s32(step2[0], step2[3]);

  t64[2] = vmull_lane_s32(vget_low_s32(step2[6]), vget_high_s32(cospis0), 0);
  t64[3] = vmull_lane_s32(vget_high_s32(step2[6]), vget_high_s32(cospis0), 0);
  t64[0] =
      vmlsl_lane_s32(t64[2], vget_low_s32(step2[5]), vget_high_s32(cospis0), 0);
  t64[1] = vmlsl_lane_s32(t64[3], vget_high_s32(step2[5]),
                          vget_high_s32(cospis0), 0);
  t64[2] =
      vmlal_lane_s32(t64[2], vget_low_s32(step2[5]), vget_high_s32(cospis0), 0);
  t64[3] = vmlal_lane_s32(t64[3], vget_high_s32(step2[5]),
                          vget_high_s32(cospis0), 0);
  t32[0] = vrshrn_n_s64(t64[0], DCT_CONST_BITS);
  t32[1] = vrshrn_n_s64(t64[1], DCT_CONST_BITS);
  t32[2] = vrshrn_n_s64(t64[2], DCT_CONST_BITS);
  t32[3] = vrshrn_n_s64(t64[3], DCT_CONST_BITS);
  step1[5] = vcombine_s32(t32[0], t32[1]);
  step1[6] = vcombine_s32(t32[2], t32[3]);

  // stage 4
  *io0 = vaddq_s32(step1[0], step2[7]);
  *io1 = vaddq_s32(step1[1], step1[6]);
  *io2 = vaddq_s32(step1[2], step1[5]);
  *io3 = vaddq_s32(step1[3], step2[4]);
  *io4 = vsubq_s32(step1[3], step2[4]);
  *io5 = vsubq_s32(step1[2], step1[5]);
  *io6 = vsubq_s32(step1[1], step1[6]);
  *io7 = vsubq_s32(step1[0], step2[7]);
}

static INLINE void highbd_idct16x16_store_pass1(const int32x4x2_t *const out,
                                                int32_t *output) {
  // Save the result into output
  vst1q_s32(output + 0, out[0].val[0]);
  vst1q_s32(output + 4, out[0].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[1].val[0]);
  vst1q_s32(output + 4, out[1].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[2].val[0]);
  vst1q_s32(output + 4, out[2].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[3].val[0]);
  vst1q_s32(output + 4, out[3].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[4].val[0]);
  vst1q_s32(output + 4, out[4].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[5].val[0]);
  vst1q_s32(output + 4, out[5].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[6].val[0]);
  vst1q_s32(output + 4, out[6].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[7].val[0]);
  vst1q_s32(output + 4, out[7].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[8].val[0]);
  vst1q_s32(output + 4, out[8].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[9].val[0]);
  vst1q_s32(output + 4, out[9].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[10].val[0]);
  vst1q_s32(output + 4, out[10].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[11].val[0]);
  vst1q_s32(output + 4, out[11].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[12].val[0]);
  vst1q_s32(output + 4, out[12].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[13].val[0]);
  vst1q_s32(output + 4, out[13].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[14].val[0]);
  vst1q_s32(output + 4, out[14].val[1]);
  output += 16;
  vst1q_s32(output + 0, out[15].val[0]);
  vst1q_s32(output + 4, out[15].val[1]);
}

static INLINE void highbd_idct16x16_add_store(const int32x4x2_t *const out,
                                              uint16_t *dest, const int stride,
                                              const int bd) {
  // Add the result to dest
  const int16x8_t max = vdupq_n_s16((1 << bd) - 1);
  int16x8_t o[16];
  o[0] = vcombine_s16(vrshrn_n_s32(out[0].val[0], 6),
                      vrshrn_n_s32(out[0].val[1], 6));
  o[1] = vcombine_s16(vrshrn_n_s32(out[1].val[0], 6),
                      vrshrn_n_s32(out[1].val[1], 6));
  o[2] = vcombine_s16(vrshrn_n_s32(out[2].val[0], 6),
                      vrshrn_n_s32(out[2].val[1], 6));
  o[3] = vcombine_s16(vrshrn_n_s32(out[3].val[0], 6),
                      vrshrn_n_s32(out[3].val[1], 6));
  o[4] = vcombine_s16(vrshrn_n_s32(out[4].val[0], 6),
                      vrshrn_n_s32(out[4].val[1], 6));
  o[5] = vcombine_s16(vrshrn_n_s32(out[5].val[0], 6),
                      vrshrn_n_s32(out[5].val[1], 6));
  o[6] = vcombine_s16(vrshrn_n_s32(out[6].val[0], 6),
                      vrshrn_n_s32(out[6].val[1], 6));
  o[7] = vcombine_s16(vrshrn_n_s32(out[7].val[0], 6),
                      vrshrn_n_s32(out[7].val[1], 6));
  o[8] = vcombine_s16(vrshrn_n_s32(out[8].val[0], 6),
                      vrshrn_n_s32(out[8].val[1], 6));
  o[9] = vcombine_s16(vrshrn_n_s32(out[9].val[0], 6),
                      vrshrn_n_s32(out[9].val[1], 6));
  o[10] = vcombine_s16(vrshrn_n_s32(out[10].val[0], 6),
                       vrshrn_n_s32(out[10].val[1], 6));
  o[11] = vcombine_s16(vrshrn_n_s32(out[11].val[0], 6),
                       vrshrn_n_s32(out[11].val[1], 6));
  o[12] = vcombine_s16(vrshrn_n_s32(out[12].val[0], 6),
                       vrshrn_n_s32(out[12].val[1], 6));
  o[13] = vcombine_s16(vrshrn_n_s32(out[13].val[0], 6),
                       vrshrn_n_s32(out[13].val[1], 6));
  o[14] = vcombine_s16(vrshrn_n_s32(out[14].val[0], 6),
                       vrshrn_n_s32(out[14].val[1], 6));
  o[15] = vcombine_s16(vrshrn_n_s32(out[15].val[0], 6),
                       vrshrn_n_s32(out[15].val[1], 6));
  highbd_idct16x16_add8x1(o[0], max, &dest, stride);
  highbd_idct16x16_add8x1(o[1], max, &dest, stride);
  highbd_idct16x16_add8x1(o[2], max, &dest, stride);
  highbd_idct16x16_add8x1(o[3], max, &dest, stride);
  highbd_idct16x16_add8x1(o[4], max, &dest, stride);
  highbd_idct16x16_add8x1(o[5], max, &dest, stride);
  highbd_idct16x16_add8x1(o[6], max, &dest, stride);
  highbd_idct16x16_add8x1(o[7], max, &dest, stride);
  highbd_idct16x16_add8x1(o[8], max, &dest, stride);
  highbd_idct16x16_add8x1(o[9], max, &dest, stride);
  highbd_idct16x16_add8x1(o[10], max, &dest, stride);
  highbd_idct16x16_add8x1(o[11], max, &dest, stride);
  highbd_idct16x16_add8x1(o[12], max, &dest, stride);
  highbd_idct16x16_add8x1(o[13], max, &dest, stride);
  highbd_idct16x16_add8x1(o[14], max, &dest, stride);
  highbd_idct16x16_add8x1(o[15], max, &dest, stride);
}

void vpx_highbd_idct16x16_256_add_half1d(const int32_t *input, int32_t *output,
                                         uint16_t *dest, const int stride,
                                         const int bd);

#endif  // VPX_VPX_DSP_ARM_HIGHBD_IDCT_NEON_H_
