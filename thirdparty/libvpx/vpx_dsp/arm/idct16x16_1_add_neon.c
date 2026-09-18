/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
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

static INLINE void idct16x16_1_add_pos_kernel(uint8_t **dest, const int stride,
                                              const uint8x16_t res) {
  const uint8x16_t a = vld1q_u8(*dest);
  const uint8x16_t b = vqaddq_u8(a, res);
  vst1q_u8(*dest, b);
  *dest += stride;
}

static INLINE void idct16x16_1_add_neg_kernel(uint8_t **dest, const int stride,
                                              const uint8x16_t res) {
  const uint8x16_t a = vld1q_u8(*dest);
  const uint8x16_t b = vqsubq_u8(a, res);
  vst1q_u8(*dest, b);
  *dest += stride;
}

void vpx_idct16x16_1_add_neon(const tran_low_t *input, uint8_t *dest,
                              int stride) {
  const int16_t out0 =
      WRAPLOW(dct_const_round_shift((int16_t)input[0] * cospi_16_64));
  const int16_t out1 = WRAPLOW(dct_const_round_shift(out0 * cospi_16_64));
  const int16_t a1 = ROUND_POWER_OF_TWO(out1, 6);

  if (a1 >= 0) {
    const uint8x16_t dc = create_dcq(a1);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
    idct16x16_1_add_pos_kernel(&dest, stride, dc);
  } else {
    const uint8x16_t dc = create_dcq(-a1);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
    idct16x16_1_add_neg_kernel(&dest, stride, dc);
  }
}
