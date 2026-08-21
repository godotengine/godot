/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/highbd_idct_neon.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/inv_txfm.h"

static INLINE void highbd_idct8x8_1_add_pos_kernel(uint16_t **dest,
                                                   const int stride,
                                                   const int16x8_t res,
                                                   const int16x8_t max) {
  const uint16x8_t a = vld1q_u16(*dest);
  const int16x8_t b = vaddq_s16(res, vreinterpretq_s16_u16(a));
  const int16x8_t c = vminq_s16(b, max);
  vst1q_u16(*dest, vreinterpretq_u16_s16(c));
  *dest += stride;
}

static INLINE void highbd_idct8x8_1_add_neg_kernel(uint16_t **dest,
                                                   const int stride,
                                                   const int16x8_t res) {
  const uint16x8_t a = vld1q_u16(*dest);
  const int16x8_t b = vaddq_s16(res, vreinterpretq_s16_u16(a));
  const uint16x8_t c = vqshluq_n_s16(b, 0);
  vst1q_u16(*dest, c);
  *dest += stride;
}

void vpx_highbd_idct8x8_1_add_neon(const tran_low_t *input, uint16_t *dest,
                                   int stride, int bd) {
  const tran_low_t out0 = HIGHBD_WRAPLOW(
      dct_const_round_shift(input[0] * (tran_high_t)cospi_16_64), bd);
  const tran_low_t out1 = HIGHBD_WRAPLOW(
      dct_const_round_shift(out0 * (tran_high_t)cospi_16_64), bd);
  const int16_t a1 = ROUND_POWER_OF_TWO(out1, 5);
  const int16x8_t dc = vdupq_n_s16(a1);

  if (a1 >= 0) {
    const int16x8_t max = vdupq_n_s16((1 << bd) - 1);
    highbd_idct8x8_1_add_pos_kernel(&dest, stride, dc, max);
    highbd_idct8x8_1_add_pos_kernel(&dest, stride, dc, max);
    highbd_idct8x8_1_add_pos_kernel(&dest, stride, dc, max);
    highbd_idct8x8_1_add_pos_kernel(&dest, stride, dc, max);
    highbd_idct8x8_1_add_pos_kernel(&dest, stride, dc, max);
    highbd_idct8x8_1_add_pos_kernel(&dest, stride, dc, max);
    highbd_idct8x8_1_add_pos_kernel(&dest, stride, dc, max);
    highbd_idct8x8_1_add_pos_kernel(&dest, stride, dc, max);
  } else {
    highbd_idct8x8_1_add_neg_kernel(&dest, stride, dc);
    highbd_idct8x8_1_add_neg_kernel(&dest, stride, dc);
    highbd_idct8x8_1_add_neg_kernel(&dest, stride, dc);
    highbd_idct8x8_1_add_neg_kernel(&dest, stride, dc);
    highbd_idct8x8_1_add_neg_kernel(&dest, stride, dc);
    highbd_idct8x8_1_add_neg_kernel(&dest, stride, dc);
    highbd_idct8x8_1_add_neg_kernel(&dest, stride, dc);
    highbd_idct8x8_1_add_neg_kernel(&dest, stride, dc);
  }
}

static INLINE void idct8x8_12_half1d_bd10(
    const int32x4_t cospis0, const int32x4_t cospis1, int32x4_t *const io0,
    int32x4_t *const io1, int32x4_t *const io2, int32x4_t *const io3,
    int32x4_t *const io4, int32x4_t *const io5, int32x4_t *const io6,
    int32x4_t *const io7) {
  int32x4_t step1[8], step2[8];

  transpose_s32_4x4(io0, io1, io2, io3);

  // stage 1
  step1[4] = vmulq_lane_s32(*io1, vget_high_s32(cospis1), 1);
  step1[5] = vmulq_lane_s32(*io3, vget_high_s32(cospis1), 0);
  step1[6] = vmulq_lane_s32(*io3, vget_low_s32(cospis1), 1);
  step1[7] = vmulq_lane_s32(*io1, vget_low_s32(cospis1), 0);
  step1[4] = vrshrq_n_s32(step1[4], DCT_CONST_BITS);
  step1[5] = vrshrq_n_s32(step1[5], DCT_CONST_BITS);
  step1[6] = vrshrq_n_s32(step1[6], DCT_CONST_BITS);
  step1[7] = vrshrq_n_s32(step1[7], DCT_CONST_BITS);

  // stage 2
  step2[1] = vmulq_lane_s32(*io0, vget_high_s32(cospis0), 0);
  step2[2] = vmulq_lane_s32(*io2, vget_high_s32(cospis0), 1);
  step2[3] = vmulq_lane_s32(*io2, vget_low_s32(cospis0), 1);
  step2[1] = vrshrq_n_s32(step2[1], DCT_CONST_BITS);
  step2[2] = vrshrq_n_s32(step2[2], DCT_CONST_BITS);
  step2[3] = vrshrq_n_s32(step2[3], DCT_CONST_BITS);

  step2[4] = vaddq_s32(step1[4], step1[5]);
  step2[5] = vsubq_s32(step1[4], step1[5]);
  step2[6] = vsubq_s32(step1[7], step1[6]);
  step2[7] = vaddq_s32(step1[7], step1[6]);

  // stage 3
  step1[0] = vaddq_s32(step2[1], step2[3]);
  step1[1] = vaddq_s32(step2[1], step2[2]);
  step1[2] = vsubq_s32(step2[1], step2[2]);
  step1[3] = vsubq_s32(step2[1], step2[3]);

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

static INLINE void idct8x8_12_half1d_bd12(
    const int32x4_t cospis0, const int32x4_t cospis1, int32x4_t *const io0,
    int32x4_t *const io1, int32x4_t *const io2, int32x4_t *const io3,
    int32x4_t *const io4, int32x4_t *const io5, int32x4_t *const io6,
    int32x4_t *const io7) {
  int32x2_t input1l, input1h, input3l, input3h;
  int32x2_t step1l[2], step1h[2];
  int32x4_t step1[8], step2[8];
  int64x2_t t64[8];
  int32x2_t t32[8];

  transpose_s32_4x4(io0, io1, io2, io3);

  // stage 1
  input1l = vget_low_s32(*io1);
  input1h = vget_high_s32(*io1);
  input3l = vget_low_s32(*io3);
  input3h = vget_high_s32(*io3);
  step1l[0] = vget_low_s32(*io0);
  step1h[0] = vget_high_s32(*io0);
  step1l[1] = vget_low_s32(*io2);
  step1h[1] = vget_high_s32(*io2);

  t64[0] = vmull_lane_s32(input1l, vget_high_s32(cospis1), 1);
  t64[1] = vmull_lane_s32(input1h, vget_high_s32(cospis1), 1);
  t64[2] = vmull_lane_s32(input3l, vget_high_s32(cospis1), 0);
  t64[3] = vmull_lane_s32(input3h, vget_high_s32(cospis1), 0);
  t64[4] = vmull_lane_s32(input3l, vget_low_s32(cospis1), 1);
  t64[5] = vmull_lane_s32(input3h, vget_low_s32(cospis1), 1);
  t64[6] = vmull_lane_s32(input1l, vget_low_s32(cospis1), 0);
  t64[7] = vmull_lane_s32(input1h, vget_low_s32(cospis1), 0);
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
  t32[2] = vrshrn_n_s64(t64[2], DCT_CONST_BITS);
  t32[3] = vrshrn_n_s64(t64[3], DCT_CONST_BITS);
  t32[4] = vrshrn_n_s64(t64[4], DCT_CONST_BITS);
  t32[5] = vrshrn_n_s64(t64[5], DCT_CONST_BITS);
  t32[6] = vrshrn_n_s64(t64[6], DCT_CONST_BITS);
  t32[7] = vrshrn_n_s64(t64[7], DCT_CONST_BITS);
  step2[1] = vcombine_s32(t32[2], t32[3]);
  step2[2] = vcombine_s32(t32[4], t32[5]);
  step2[3] = vcombine_s32(t32[6], t32[7]);

  step2[4] = vaddq_s32(step1[4], step1[5]);
  step2[5] = vsubq_s32(step1[4], step1[5]);
  step2[6] = vsubq_s32(step1[7], step1[6]);
  step2[7] = vaddq_s32(step1[7], step1[6]);

  // stage 3
  step1[0] = vaddq_s32(step2[1], step2[3]);
  step1[1] = vaddq_s32(step2[1], step2[2]);
  step1[2] = vsubq_s32(step2[1], step2[2]);
  step1[3] = vsubq_s32(step2[1], step2[3]);

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

void vpx_highbd_idct8x8_12_add_neon(const tran_low_t *input, uint16_t *dest,
                                    int stride, int bd) {
  int32x4_t a[16];
  int16x8_t c[8];

  a[0] = vld1q_s32(input);
  a[1] = vld1q_s32(input + 8);
  a[2] = vld1q_s32(input + 16);
  a[3] = vld1q_s32(input + 24);

  if (bd == 8) {
    const int16x8_t cospis = vld1q_s16(kCospi);
    const int16x8_t cospisd = vaddq_s16(cospis, cospis);
    const int16x4_t cospis0 = vget_low_s16(cospis);     // cospi 0, 8, 16, 24
    const int16x4_t cospisd0 = vget_low_s16(cospisd);   // doubled 0, 8, 16, 24
    const int16x4_t cospisd1 = vget_high_s16(cospisd);  // doubled 4, 12, 20, 28
    int16x4_t b[8];

    b[0] = vmovn_s32(a[0]);
    b[1] = vmovn_s32(a[1]);
    b[2] = vmovn_s32(a[2]);
    b[3] = vmovn_s32(a[3]);

    idct8x8_12_pass1_bd8(cospis0, cospisd0, cospisd1, b);
    idct8x8_12_pass2_bd8(cospis0, cospisd0, cospisd1, b, c);
    c[0] = vrshrq_n_s16(c[0], 5);
    c[1] = vrshrq_n_s16(c[1], 5);
    c[2] = vrshrq_n_s16(c[2], 5);
    c[3] = vrshrq_n_s16(c[3], 5);
    c[4] = vrshrq_n_s16(c[4], 5);
    c[5] = vrshrq_n_s16(c[5], 5);
    c[6] = vrshrq_n_s16(c[6], 5);
    c[7] = vrshrq_n_s16(c[7], 5);
  } else {
    const int32x4_t cospis0 = vld1q_s32(kCospi32);      // cospi 0, 8, 16, 24
    const int32x4_t cospis1 = vld1q_s32(kCospi32 + 4);  // cospi 4, 12, 20, 28

    if (bd == 10) {
      idct8x8_12_half1d_bd10(cospis0, cospis1, &a[0], &a[1], &a[2], &a[3],
                             &a[4], &a[5], &a[6], &a[7]);
      idct8x8_12_half1d_bd10(cospis0, cospis1, &a[0], &a[1], &a[2], &a[3],
                             &a[8], &a[9], &a[10], &a[11]);
      idct8x8_12_half1d_bd10(cospis0, cospis1, &a[4], &a[5], &a[6], &a[7],
                             &a[12], &a[13], &a[14], &a[15]);
    } else {
      idct8x8_12_half1d_bd12(cospis0, cospis1, &a[0], &a[1], &a[2], &a[3],
                             &a[4], &a[5], &a[6], &a[7]);
      idct8x8_12_half1d_bd12(cospis0, cospis1, &a[0], &a[1], &a[2], &a[3],
                             &a[8], &a[9], &a[10], &a[11]);
      idct8x8_12_half1d_bd12(cospis0, cospis1, &a[4], &a[5], &a[6], &a[7],
                             &a[12], &a[13], &a[14], &a[15]);
    }
    c[0] = vcombine_s16(vrshrn_n_s32(a[0], 5), vrshrn_n_s32(a[4], 5));
    c[1] = vcombine_s16(vrshrn_n_s32(a[1], 5), vrshrn_n_s32(a[5], 5));
    c[2] = vcombine_s16(vrshrn_n_s32(a[2], 5), vrshrn_n_s32(a[6], 5));
    c[3] = vcombine_s16(vrshrn_n_s32(a[3], 5), vrshrn_n_s32(a[7], 5));
    c[4] = vcombine_s16(vrshrn_n_s32(a[8], 5), vrshrn_n_s32(a[12], 5));
    c[5] = vcombine_s16(vrshrn_n_s32(a[9], 5), vrshrn_n_s32(a[13], 5));
    c[6] = vcombine_s16(vrshrn_n_s32(a[10], 5), vrshrn_n_s32(a[14], 5));
    c[7] = vcombine_s16(vrshrn_n_s32(a[11], 5), vrshrn_n_s32(a[15], 5));
  }
  highbd_add8x8(c, dest, stride, bd);
}

void vpx_highbd_idct8x8_64_add_neon(const tran_low_t *input, uint16_t *dest,
                                    int stride, int bd) {
  int32x4_t a[16];
  int16x8_t c[8];

  a[0] = vld1q_s32(input);
  a[1] = vld1q_s32(input + 4);
  a[2] = vld1q_s32(input + 8);
  a[3] = vld1q_s32(input + 12);
  a[4] = vld1q_s32(input + 16);
  a[5] = vld1q_s32(input + 20);
  a[6] = vld1q_s32(input + 24);
  a[7] = vld1q_s32(input + 28);
  a[8] = vld1q_s32(input + 32);
  a[9] = vld1q_s32(input + 36);
  a[10] = vld1q_s32(input + 40);
  a[11] = vld1q_s32(input + 44);
  a[12] = vld1q_s32(input + 48);
  a[13] = vld1q_s32(input + 52);
  a[14] = vld1q_s32(input + 56);
  a[15] = vld1q_s32(input + 60);

  if (bd == 8) {
    const int16x8_t cospis = vld1q_s16(kCospi);
    const int16x4_t cospis0 = vget_low_s16(cospis);   // cospi 0, 8, 16, 24
    const int16x4_t cospis1 = vget_high_s16(cospis);  // cospi 4, 12, 20, 28
    int16x8_t b[8];

    b[0] = vcombine_s16(vmovn_s32(a[0]), vmovn_s32(a[1]));
    b[1] = vcombine_s16(vmovn_s32(a[2]), vmovn_s32(a[3]));
    b[2] = vcombine_s16(vmovn_s32(a[4]), vmovn_s32(a[5]));
    b[3] = vcombine_s16(vmovn_s32(a[6]), vmovn_s32(a[7]));
    b[4] = vcombine_s16(vmovn_s32(a[8]), vmovn_s32(a[9]));
    b[5] = vcombine_s16(vmovn_s32(a[10]), vmovn_s32(a[11]));
    b[6] = vcombine_s16(vmovn_s32(a[12]), vmovn_s32(a[13]));
    b[7] = vcombine_s16(vmovn_s32(a[14]), vmovn_s32(a[15]));

    idct8x8_64_1d_bd8(cospis0, cospis1, b);
    idct8x8_64_1d_bd8(cospis0, cospis1, b);

    c[0] = vrshrq_n_s16(b[0], 5);
    c[1] = vrshrq_n_s16(b[1], 5);
    c[2] = vrshrq_n_s16(b[2], 5);
    c[3] = vrshrq_n_s16(b[3], 5);
    c[4] = vrshrq_n_s16(b[4], 5);
    c[5] = vrshrq_n_s16(b[5], 5);
    c[6] = vrshrq_n_s16(b[6], 5);
    c[7] = vrshrq_n_s16(b[7], 5);
  } else {
    const int32x4_t cospis0 = vld1q_s32(kCospi32);      // cospi 0, 8, 16, 24
    const int32x4_t cospis1 = vld1q_s32(kCospi32 + 4);  // cospi 4, 12, 20, 28

    if (bd == 10) {
      idct8x8_64_half1d_bd10(cospis0, cospis1, &a[0], &a[1], &a[2], &a[3],
                             &a[4], &a[5], &a[6], &a[7]);
      idct8x8_64_half1d_bd10(cospis0, cospis1, &a[8], &a[9], &a[10], &a[11],
                             &a[12], &a[13], &a[14], &a[15]);
      idct8x8_64_half1d_bd10(cospis0, cospis1, &a[0], &a[8], &a[1], &a[9],
                             &a[2], &a[10], &a[3], &a[11]);
      idct8x8_64_half1d_bd10(cospis0, cospis1, &a[4], &a[12], &a[5], &a[13],
                             &a[6], &a[14], &a[7], &a[15]);
    } else {
      idct8x8_64_half1d_bd12(cospis0, cospis1, &a[0], &a[1], &a[2], &a[3],
                             &a[4], &a[5], &a[6], &a[7]);
      idct8x8_64_half1d_bd12(cospis0, cospis1, &a[8], &a[9], &a[10], &a[11],
                             &a[12], &a[13], &a[14], &a[15]);
      idct8x8_64_half1d_bd12(cospis0, cospis1, &a[0], &a[8], &a[1], &a[9],
                             &a[2], &a[10], &a[3], &a[11]);
      idct8x8_64_half1d_bd12(cospis0, cospis1, &a[4], &a[12], &a[5], &a[13],
                             &a[6], &a[14], &a[7], &a[15]);
    }
    c[0] = vcombine_s16(vrshrn_n_s32(a[0], 5), vrshrn_n_s32(a[4], 5));
    c[1] = vcombine_s16(vrshrn_n_s32(a[8], 5), vrshrn_n_s32(a[12], 5));
    c[2] = vcombine_s16(vrshrn_n_s32(a[1], 5), vrshrn_n_s32(a[5], 5));
    c[3] = vcombine_s16(vrshrn_n_s32(a[9], 5), vrshrn_n_s32(a[13], 5));
    c[4] = vcombine_s16(vrshrn_n_s32(a[2], 5), vrshrn_n_s32(a[6], 5));
    c[5] = vcombine_s16(vrshrn_n_s32(a[10], 5), vrshrn_n_s32(a[14], 5));
    c[6] = vcombine_s16(vrshrn_n_s32(a[3], 5), vrshrn_n_s32(a[7], 5));
    c[7] = vcombine_s16(vrshrn_n_s32(a[11], 5), vrshrn_n_s32(a[15], 5));
  }
  highbd_add8x8(c, dest, stride, bd);
}
