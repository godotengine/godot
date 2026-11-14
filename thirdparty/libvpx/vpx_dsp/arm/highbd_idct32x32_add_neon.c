/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/inv_txfm.h"

static INLINE void highbd_idct32x32_1_add_pos_kernel(uint16_t **dest,
                                                     const int stride,
                                                     const int16x8_t res,
                                                     const int16x8_t max) {
  const uint16x8_t a0 = vld1q_u16(*dest);
  const uint16x8_t a1 = vld1q_u16(*dest + 8);
  const uint16x8_t a2 = vld1q_u16(*dest + 16);
  const uint16x8_t a3 = vld1q_u16(*dest + 24);
  const int16x8_t b0 = vaddq_s16(res, vreinterpretq_s16_u16(a0));
  const int16x8_t b1 = vaddq_s16(res, vreinterpretq_s16_u16(a1));
  const int16x8_t b2 = vaddq_s16(res, vreinterpretq_s16_u16(a2));
  const int16x8_t b3 = vaddq_s16(res, vreinterpretq_s16_u16(a3));
  const int16x8_t c0 = vminq_s16(b0, max);
  const int16x8_t c1 = vminq_s16(b1, max);
  const int16x8_t c2 = vminq_s16(b2, max);
  const int16x8_t c3 = vminq_s16(b3, max);
  vst1q_u16(*dest, vreinterpretq_u16_s16(c0));
  vst1q_u16(*dest + 8, vreinterpretq_u16_s16(c1));
  vst1q_u16(*dest + 16, vreinterpretq_u16_s16(c2));
  vst1q_u16(*dest + 24, vreinterpretq_u16_s16(c3));
  *dest += stride;
}

static INLINE void highbd_idct32x32_1_add_neg_kernel(uint16_t **dest,
                                                     const int stride,
                                                     const int16x8_t res) {
  const uint16x8_t a0 = vld1q_u16(*dest);
  const uint16x8_t a1 = vld1q_u16(*dest + 8);
  const uint16x8_t a2 = vld1q_u16(*dest + 16);
  const uint16x8_t a3 = vld1q_u16(*dest + 24);
  const int16x8_t b0 = vaddq_s16(res, vreinterpretq_s16_u16(a0));
  const int16x8_t b1 = vaddq_s16(res, vreinterpretq_s16_u16(a1));
  const int16x8_t b2 = vaddq_s16(res, vreinterpretq_s16_u16(a2));
  const int16x8_t b3 = vaddq_s16(res, vreinterpretq_s16_u16(a3));
  const uint16x8_t c0 = vqshluq_n_s16(b0, 0);
  const uint16x8_t c1 = vqshluq_n_s16(b1, 0);
  const uint16x8_t c2 = vqshluq_n_s16(b2, 0);
  const uint16x8_t c3 = vqshluq_n_s16(b3, 0);
  vst1q_u16(*dest, c0);
  vst1q_u16(*dest + 8, c1);
  vst1q_u16(*dest + 16, c2);
  vst1q_u16(*dest + 24, c3);
  *dest += stride;
}

void vpx_highbd_idct32x32_1_add_neon(const tran_low_t *input, uint16_t *dest,
                                     int stride, int bd) {
  const tran_low_t out0 = HIGHBD_WRAPLOW(
      dct_const_round_shift(input[0] * (tran_high_t)cospi_16_64), bd);
  const tran_low_t out1 = HIGHBD_WRAPLOW(
      dct_const_round_shift(out0 * (tran_high_t)cospi_16_64), bd);
  const int16_t a1 = ROUND_POWER_OF_TWO(out1, 6);
  const int16x8_t dc = vdupq_n_s16(a1);
  int i;

  if (a1 >= 0) {
    const int16x8_t max = vdupq_n_s16((1 << bd) - 1);
    for (i = 0; i < 8; ++i) {
      highbd_idct32x32_1_add_pos_kernel(&dest, stride, dc, max);
      highbd_idct32x32_1_add_pos_kernel(&dest, stride, dc, max);
      highbd_idct32x32_1_add_pos_kernel(&dest, stride, dc, max);
      highbd_idct32x32_1_add_pos_kernel(&dest, stride, dc, max);
    }
  } else {
    for (i = 0; i < 8; ++i) {
      highbd_idct32x32_1_add_neg_kernel(&dest, stride, dc);
      highbd_idct32x32_1_add_neg_kernel(&dest, stride, dc);
      highbd_idct32x32_1_add_neg_kernel(&dest, stride, dc);
      highbd_idct32x32_1_add_neg_kernel(&dest, stride, dc);
    }
  }
}
