/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
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

int64_t vp9_highbd_block_error_neon(const tran_low_t *coeff,
                                    const tran_low_t *dqcoeff,
                                    intptr_t block_size, int64_t *ssz, int bd) {
  uint64x2_t err_u64 = vdupq_n_u64(0);
  int64x2_t ssz_s64 = vdupq_n_s64(0);

  const int shift = 2 * (bd - 8);
  const int rounding = shift > 0 ? 1 << (shift - 1) : 0;

  assert(block_size >= 16);
  assert((block_size % 16) == 0);

  do {
    const int32x4_t c = load_tran_low_to_s32q(coeff);
    const int32x4_t d = load_tran_low_to_s32q(dqcoeff);

    const uint32x4_t diff = vreinterpretq_u32_s32(vabdq_s32(c, d));

    err_u64 = vmlal_u32(err_u64, vget_low_u32(diff), vget_low_u32(diff));
    err_u64 = vmlal_u32(err_u64, vget_high_u32(diff), vget_high_u32(diff));

    ssz_s64 = vmlal_s32(ssz_s64, vget_low_s32(c), vget_low_s32(c));
    ssz_s64 = vmlal_s32(ssz_s64, vget_high_s32(c), vget_high_s32(c));

    coeff += 4;
    dqcoeff += 4;
    block_size -= 4;
  } while (block_size != 0);

  *ssz = (horizontal_add_int64x2(ssz_s64) + rounding) >> shift;
  return ((int64_t)horizontal_add_uint64x2(err_u64) + rounding) >> shift;
}
