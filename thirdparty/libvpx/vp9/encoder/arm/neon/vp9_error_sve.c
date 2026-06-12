/*
 *  Copyright (c) 2024 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>

#include "./vp9_rtcd.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"
#include "vpx_dsp/arm/vpx_neon_sve_bridge.h"

int64_t vp9_block_error_sve(const tran_low_t *coeff, const tran_low_t *dqcoeff,
                            intptr_t block_size, int64_t *ssz) {
  int64x2_t err_v = vdupq_n_s64(0);
  int64x2_t ssz_v = vdupq_n_s64(0);

  assert(block_size >= 16);
  assert((block_size % 16) == 0);

  do {
    const int16x8_t c0 = load_tran_low_to_s16q(coeff);
    const int16x8_t c1 = load_tran_low_to_s16q(coeff + 8);

    const int16x8_t d0 = load_tran_low_to_s16q(dqcoeff);
    const int16x8_t d1 = load_tran_low_to_s16q(dqcoeff + 8);

    const int16x8_t diff0 = vabdq_s16(c0, d0);
    const int16x8_t diff1 = vabdq_s16(c1, d1);

    err_v = vpx_dotq_s16(err_v, diff0, diff0);
    err_v = vpx_dotq_s16(err_v, diff1, diff1);

    ssz_v = vpx_dotq_s16(ssz_v, c0, c0);
    ssz_v = vpx_dotq_s16(ssz_v, c1, c1);

    coeff += 16;
    dqcoeff += 16;
    block_size -= 16;
  } while (block_size != 0);

  *ssz = horizontal_add_int64x2(ssz_v);
  return horizontal_add_int64x2(err_v);
}

int64_t vp9_block_error_fp_sve(const tran_low_t *coeff,
                               const tran_low_t *dqcoeff, int block_size) {
  int64x2_t err = vdupq_n_s64(0);

  assert(block_size >= 16);
  assert((block_size % 16) == 0);

  do {
    const int16x8_t c0 = load_tran_low_to_s16q(coeff);
    const int16x8_t c1 = load_tran_low_to_s16q(coeff + 8);

    const int16x8_t d0 = load_tran_low_to_s16q(dqcoeff);
    const int16x8_t d1 = load_tran_low_to_s16q(dqcoeff + 8);

    const int16x8_t diff0 = vabdq_s16(c0, d0);
    const int16x8_t diff1 = vabdq_s16(c1, d1);

    err = vpx_dotq_s16(err, diff0, diff0);
    err = vpx_dotq_s16(err, diff1, diff1);

    coeff += 16;
    dqcoeff += 16;
    block_size -= 16;
  } while (block_size != 0);

  return horizontal_add_int64x2(err);
}
