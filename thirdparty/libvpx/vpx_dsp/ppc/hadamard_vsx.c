/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/ppc/types_vsx.h"
#include "vpx_dsp/ppc/transpose_vsx.h"
#include "vpx_dsp/ppc/bitdepth_conversion_vsx.h"

static void vpx_hadamard_s16_8x8_one_pass(int16x8_t v[8]) {
  const int16x8_t b0 = vec_add(v[0], v[1]);
  const int16x8_t b1 = vec_sub(v[0], v[1]);
  const int16x8_t b2 = vec_add(v[2], v[3]);
  const int16x8_t b3 = vec_sub(v[2], v[3]);
  const int16x8_t b4 = vec_add(v[4], v[5]);
  const int16x8_t b5 = vec_sub(v[4], v[5]);
  const int16x8_t b6 = vec_add(v[6], v[7]);
  const int16x8_t b7 = vec_sub(v[6], v[7]);

  const int16x8_t c0 = vec_add(b0, b2);
  const int16x8_t c1 = vec_add(b1, b3);
  const int16x8_t c2 = vec_sub(b0, b2);
  const int16x8_t c3 = vec_sub(b1, b3);
  const int16x8_t c4 = vec_add(b4, b6);
  const int16x8_t c5 = vec_add(b5, b7);
  const int16x8_t c6 = vec_sub(b4, b6);
  const int16x8_t c7 = vec_sub(b5, b7);

  v[0] = vec_add(c0, c4);
  v[1] = vec_sub(c2, c6);
  v[2] = vec_sub(c0, c4);
  v[3] = vec_add(c2, c6);
  v[4] = vec_add(c3, c7);
  v[5] = vec_sub(c3, c7);
  v[6] = vec_sub(c1, c5);
  v[7] = vec_add(c1, c5);
}

void vpx_hadamard_8x8_vsx(const int16_t *src_diff, ptrdiff_t src_stride,
                          tran_low_t *coeff) {
  int16x8_t v[8];

  v[0] = vec_vsx_ld(0, src_diff);
  v[1] = vec_vsx_ld(0, src_diff + src_stride);
  v[2] = vec_vsx_ld(0, src_diff + (2 * src_stride));
  v[3] = vec_vsx_ld(0, src_diff + (3 * src_stride));
  v[4] = vec_vsx_ld(0, src_diff + (4 * src_stride));
  v[5] = vec_vsx_ld(0, src_diff + (5 * src_stride));
  v[6] = vec_vsx_ld(0, src_diff + (6 * src_stride));
  v[7] = vec_vsx_ld(0, src_diff + (7 * src_stride));

  vpx_hadamard_s16_8x8_one_pass(v);

  vpx_transpose_s16_8x8(v);

  vpx_hadamard_s16_8x8_one_pass(v);

  store_tran_low(v[0], 0, coeff);
  store_tran_low(v[1], 0, coeff + 8);
  store_tran_low(v[2], 0, coeff + 16);
  store_tran_low(v[3], 0, coeff + 24);
  store_tran_low(v[4], 0, coeff + 32);
  store_tran_low(v[5], 0, coeff + 40);
  store_tran_low(v[6], 0, coeff + 48);
  store_tran_low(v[7], 0, coeff + 56);
}

void vpx_hadamard_16x16_vsx(const int16_t *src_diff, ptrdiff_t src_stride,
                            tran_low_t *coeff) {
  int i;
  const uint16x8_t ones = vec_splat_u16(1);

  /* Rearrange 16x16 to 8x32 and remove stride.
   * Top left first. */
  vpx_hadamard_8x8_vsx(src_diff, src_stride, coeff);
  /* Top right. */
  vpx_hadamard_8x8_vsx(src_diff + 8 + 0 * src_stride, src_stride, coeff + 64);
  /* Bottom left. */
  vpx_hadamard_8x8_vsx(src_diff + 0 + 8 * src_stride, src_stride, coeff + 128);
  /* Bottom right. */
  vpx_hadamard_8x8_vsx(src_diff + 8 + 8 * src_stride, src_stride, coeff + 192);

  /* Overlay the 8x8 blocks and combine. */
  for (i = 0; i < 64; i += 8) {
    const int16x8_t a0 = load_tran_low(0, coeff);
    const int16x8_t a1 = load_tran_low(0, coeff + 64);
    const int16x8_t a2 = load_tran_low(0, coeff + 128);
    const int16x8_t a3 = load_tran_low(0, coeff + 192);

    /* Prevent the result from escaping int16_t. */
    const int16x8_t b0 = vec_sra(a0, ones);
    const int16x8_t b1 = vec_sra(a1, ones);
    const int16x8_t b2 = vec_sra(a2, ones);
    const int16x8_t b3 = vec_sra(a3, ones);

    const int16x8_t c0 = vec_add(b0, b1);
    const int16x8_t c2 = vec_add(b2, b3);
    const int16x8_t c1 = vec_sub(b0, b1);
    const int16x8_t c3 = vec_sub(b2, b3);

    const int16x8_t d0 = vec_add(c0, c2);
    const int16x8_t d1 = vec_add(c1, c3);
    const int16x8_t d2 = vec_sub(c0, c2);
    const int16x8_t d3 = vec_sub(c1, c3);

    store_tran_low(d0, 0, coeff);
    store_tran_low(d1, 0, coeff + 64);
    store_tran_low(d2, 0, coeff + 128);
    store_tran_low(d3, 0, coeff + 192);

    coeff += 8;
  }
}
