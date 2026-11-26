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
#include "vpx_dsp/inv_txfm.h"

// res is in reverse row order
static INLINE void highbd_idct4x4_1_add_kernel2(uint16_t **dest,
                                                const int stride,
                                                const int16x8_t res,
                                                const int16x8_t max) {
  const uint16x4_t a0 = vld1_u16(*dest);
  const uint16x4_t a1 = vld1_u16(*dest + stride);
  const int16x8_t a = vreinterpretq_s16_u16(vcombine_u16(a1, a0));
  // Note: In some profile tests, res is quite close to +/-32767.
  // We use saturating addition.
  const int16x8_t b = vqaddq_s16(res, a);
  const int16x8_t c = vminq_s16(b, max);
  const uint16x8_t d = vqshluq_n_s16(c, 0);
  vst1_u16(*dest, vget_high_u16(d));
  *dest += stride;
  vst1_u16(*dest, vget_low_u16(d));
  *dest += stride;
}

void vpx_highbd_idct4x4_1_add_neon(const tran_low_t *input, uint16_t *dest,
                                   int stride, int bd) {
  const int16x8_t max = vdupq_n_s16((1 << bd) - 1);
  const tran_low_t out0 = HIGHBD_WRAPLOW(
      dct_const_round_shift(input[0] * (tran_high_t)cospi_16_64), bd);
  const tran_low_t out1 = HIGHBD_WRAPLOW(
      dct_const_round_shift(out0 * (tran_high_t)cospi_16_64), bd);
  const int16_t a1 = ROUND_POWER_OF_TWO(out1, 4);
  const int16x8_t dc = vdupq_n_s16(a1);

  highbd_idct4x4_1_add_kernel1(&dest, stride, dc, max);
  highbd_idct4x4_1_add_kernel1(&dest, stride, dc, max);
}

void vpx_highbd_idct4x4_16_add_neon(const tran_low_t *input, uint16_t *dest,
                                    int stride, int bd) {
  const int16x8_t max = vdupq_n_s16((1 << bd) - 1);
  int16x8_t a[2];
  int32x4_t c[4];

  c[0] = vld1q_s32(input);
  c[1] = vld1q_s32(input + 4);
  c[2] = vld1q_s32(input + 8);
  c[3] = vld1q_s32(input + 12);

  if (bd == 8) {
    // Rows
    a[0] = vcombine_s16(vmovn_s32(c[0]), vmovn_s32(c[1]));
    a[1] = vcombine_s16(vmovn_s32(c[2]), vmovn_s32(c[3]));
    transpose_idct4x4_16_bd8(a);

    // Columns
    a[1] = vcombine_s16(vget_high_s16(a[1]), vget_low_s16(a[1]));
    transpose_idct4x4_16_bd8(a);
    a[0] = vrshrq_n_s16(a[0], 4);
    a[1] = vrshrq_n_s16(a[1], 4);
  } else {
    const int32x4_t cospis = vld1q_s32(kCospi32);

    if (bd == 10) {
      idct4x4_16_kernel_bd10(cospis, c);
      idct4x4_16_kernel_bd10(cospis, c);
    } else {
      idct4x4_16_kernel_bd12(cospis, c);
      idct4x4_16_kernel_bd12(cospis, c);
    }
    a[0] = vcombine_s16(vqrshrn_n_s32(c[0], 4), vqrshrn_n_s32(c[1], 4));
    a[1] = vcombine_s16(vqrshrn_n_s32(c[3], 4), vqrshrn_n_s32(c[2], 4));
  }

  highbd_idct4x4_1_add_kernel1(&dest, stride, a[0], max);
  highbd_idct4x4_1_add_kernel2(&dest, stride, a[1], max);
}
