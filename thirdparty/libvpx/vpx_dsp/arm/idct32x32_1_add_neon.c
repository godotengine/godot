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

static INLINE void idct32x32_1_add_pos_kernel(uint8_t **dest, const int stride,
                                              const uint8x16_t res) {
  const uint8x16_t a0 = vld1q_u8(*dest);
  const uint8x16_t a1 = vld1q_u8(*dest + 16);
  const uint8x16_t b0 = vqaddq_u8(a0, res);
  const uint8x16_t b1 = vqaddq_u8(a1, res);
  vst1q_u8(*dest, b0);
  vst1q_u8(*dest + 16, b1);
  *dest += stride;
}

static INLINE void idct32x32_1_add_neg_kernel(uint8_t **dest, const int stride,
                                              const uint8x16_t res) {
  const uint8x16_t a0 = vld1q_u8(*dest);
  const uint8x16_t a1 = vld1q_u8(*dest + 16);
  const uint8x16_t b0 = vqsubq_u8(a0, res);
  const uint8x16_t b1 = vqsubq_u8(a1, res);
  vst1q_u8(*dest, b0);
  vst1q_u8(*dest + 16, b1);
  *dest += stride;
}

void vpx_idct32x32_1_add_neon(const tran_low_t *input, uint8_t *dest,
                              int stride) {
  int i;
  const int16_t out0 =
      WRAPLOW(dct_const_round_shift((int16_t)input[0] * cospi_16_64));
  const int16_t out1 = WRAPLOW(dct_const_round_shift(out0 * cospi_16_64));
  const int16_t a1 = ROUND_POWER_OF_TWO(out1, 6);

  if (a1 >= 0) {
    const uint8x16_t dc = create_dcq(a1);
    for (i = 0; i < 32; i++) {
      idct32x32_1_add_pos_kernel(&dest, stride, dc);
    }
  } else {
    const uint8x16_t dc = create_dcq(-a1);
    for (i = 0; i < 32; i++) {
      idct32x32_1_add_neg_kernel(&dest, stride, dc);
    }
  }
}
