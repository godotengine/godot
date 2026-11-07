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
#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"

static void hadamard8x8_one_pass(int16x8_t *a0, int16x8_t *a1, int16x8_t *a2,
                                 int16x8_t *a3, int16x8_t *a4, int16x8_t *a5,
                                 int16x8_t *a6, int16x8_t *a7) {
  const int16x8_t b0 = vaddq_s16(*a0, *a1);
  const int16x8_t b1 = vsubq_s16(*a0, *a1);
  const int16x8_t b2 = vaddq_s16(*a2, *a3);
  const int16x8_t b3 = vsubq_s16(*a2, *a3);
  const int16x8_t b4 = vaddq_s16(*a4, *a5);
  const int16x8_t b5 = vsubq_s16(*a4, *a5);
  const int16x8_t b6 = vaddq_s16(*a6, *a7);
  const int16x8_t b7 = vsubq_s16(*a6, *a7);

  const int16x8_t c0 = vaddq_s16(b0, b2);
  const int16x8_t c1 = vaddq_s16(b1, b3);
  const int16x8_t c2 = vsubq_s16(b0, b2);
  const int16x8_t c3 = vsubq_s16(b1, b3);
  const int16x8_t c4 = vaddq_s16(b4, b6);
  const int16x8_t c5 = vaddq_s16(b5, b7);
  const int16x8_t c6 = vsubq_s16(b4, b6);
  const int16x8_t c7 = vsubq_s16(b5, b7);

  *a0 = vaddq_s16(c0, c4);
  *a1 = vsubq_s16(c2, c6);
  *a2 = vsubq_s16(c0, c4);
  *a3 = vaddq_s16(c2, c6);
  *a4 = vaddq_s16(c3, c7);
  *a5 = vsubq_s16(c3, c7);
  *a6 = vsubq_s16(c1, c5);
  *a7 = vaddq_s16(c1, c5);
}

void vpx_hadamard_8x8_neon(const int16_t *src_diff, ptrdiff_t src_stride,
                           tran_low_t *coeff) {
  int16x8_t a0 = vld1q_s16(src_diff);
  int16x8_t a1 = vld1q_s16(src_diff + src_stride);
  int16x8_t a2 = vld1q_s16(src_diff + 2 * src_stride);
  int16x8_t a3 = vld1q_s16(src_diff + 3 * src_stride);
  int16x8_t a4 = vld1q_s16(src_diff + 4 * src_stride);
  int16x8_t a5 = vld1q_s16(src_diff + 5 * src_stride);
  int16x8_t a6 = vld1q_s16(src_diff + 6 * src_stride);
  int16x8_t a7 = vld1q_s16(src_diff + 7 * src_stride);

  hadamard8x8_one_pass(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

  transpose_s16_8x8(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

  hadamard8x8_one_pass(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

  // Skip the second transpose because it is not required.

  store_s16q_to_tran_low(coeff + 0, a0);
  store_s16q_to_tran_low(coeff + 8, a1);
  store_s16q_to_tran_low(coeff + 16, a2);
  store_s16q_to_tran_low(coeff + 24, a3);
  store_s16q_to_tran_low(coeff + 32, a4);
  store_s16q_to_tran_low(coeff + 40, a5);
  store_s16q_to_tran_low(coeff + 48, a6);
  store_s16q_to_tran_low(coeff + 56, a7);
}

void vpx_hadamard_16x16_neon(const int16_t *src_diff, ptrdiff_t src_stride,
                             tran_low_t *coeff) {
  int i;

  /* Rearrange 16x16 to 8x32 and remove stride.
   * Top left first. */
  vpx_hadamard_8x8_neon(src_diff + 0 + 0 * src_stride, src_stride, coeff + 0);
  /* Top right. */
  vpx_hadamard_8x8_neon(src_diff + 8 + 0 * src_stride, src_stride, coeff + 64);
  /* Bottom left. */
  vpx_hadamard_8x8_neon(src_diff + 0 + 8 * src_stride, src_stride, coeff + 128);
  /* Bottom right. */
  vpx_hadamard_8x8_neon(src_diff + 8 + 8 * src_stride, src_stride, coeff + 192);

  for (i = 0; i < 64; i += 8) {
    const int16x8_t a0 = load_tran_low_to_s16q(coeff + 0);
    const int16x8_t a1 = load_tran_low_to_s16q(coeff + 64);
    const int16x8_t a2 = load_tran_low_to_s16q(coeff + 128);
    const int16x8_t a3 = load_tran_low_to_s16q(coeff + 192);

    const int16x8_t b0 = vhaddq_s16(a0, a1);
    const int16x8_t b1 = vhsubq_s16(a0, a1);
    const int16x8_t b2 = vhaddq_s16(a2, a3);
    const int16x8_t b3 = vhsubq_s16(a2, a3);

    const int16x8_t c0 = vaddq_s16(b0, b2);
    const int16x8_t c1 = vaddq_s16(b1, b3);
    const int16x8_t c2 = vsubq_s16(b0, b2);
    const int16x8_t c3 = vsubq_s16(b1, b3);

    store_s16q_to_tran_low(coeff + 0, c0);
    store_s16q_to_tran_low(coeff + 64, c1);
    store_s16q_to_tran_low(coeff + 128, c2);
    store_s16q_to_tran_low(coeff + 192, c3);

    coeff += 8;
  }
}

void vpx_hadamard_32x32_neon(const int16_t *src_diff, ptrdiff_t src_stride,
                             tran_low_t *coeff) {
  int i;

  /* Rearrange 32x32 to 16x64 and remove stride.
   * Top left first. */
  vpx_hadamard_16x16_neon(src_diff + 0 + 0 * src_stride, src_stride, coeff + 0);
  /* Top right. */
  vpx_hadamard_16x16_neon(src_diff + 16 + 0 * src_stride, src_stride,
                          coeff + 256);
  /* Bottom left. */
  vpx_hadamard_16x16_neon(src_diff + 0 + 16 * src_stride, src_stride,
                          coeff + 512);
  /* Bottom right. */
  vpx_hadamard_16x16_neon(src_diff + 16 + 16 * src_stride, src_stride,
                          coeff + 768);

  for (i = 0; i < 256; i += 8) {
    const int16x8_t a0 = load_tran_low_to_s16q(coeff + 0);
    const int16x8_t a1 = load_tran_low_to_s16q(coeff + 256);
    const int16x8_t a2 = load_tran_low_to_s16q(coeff + 512);
    const int16x8_t a3 = load_tran_low_to_s16q(coeff + 768);

    const int16x8_t b0 = vshrq_n_s16(vhaddq_s16(a0, a1), 1);
    const int16x8_t b1 = vshrq_n_s16(vhsubq_s16(a0, a1), 1);
    const int16x8_t b2 = vshrq_n_s16(vhaddq_s16(a2, a3), 1);
    const int16x8_t b3 = vshrq_n_s16(vhsubq_s16(a2, a3), 1);

    const int16x8_t c0 = vaddq_s16(b0, b2);
    const int16x8_t c1 = vaddq_s16(b1, b3);
    const int16x8_t c2 = vsubq_s16(b0, b2);
    const int16x8_t c3 = vsubq_s16(b1, b3);

    store_s16q_to_tran_low(coeff + 0, c0);
    store_s16q_to_tran_low(coeff + 256, c1);
    store_s16q_to_tran_low(coeff + 512, c2);
    store_s16q_to_tran_low(coeff + 768, c3);

    coeff += 8;
  }
}
