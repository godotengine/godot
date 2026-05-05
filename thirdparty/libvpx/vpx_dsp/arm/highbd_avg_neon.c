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

#include "./vpx_dsp_rtcd.h"
#include "./vpx_config.h"

#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"

uint32_t vpx_highbd_avg_4x4_neon(const uint8_t *s8, int p) {
  const uint16_t *a_ptr = CONVERT_TO_SHORTPTR(s8);
  const uint16x8_t a0 = load_unaligned_u16q(a_ptr + 0 * p, p);
  const uint16x8_t a1 = load_unaligned_u16q(a_ptr + 2 * p, p);
  return (horizontal_add_uint16x8(vaddq_u16(a0, a1)) + (1 << 3)) >> 4;
}

uint32_t vpx_highbd_avg_8x8_neon(const uint8_t *s8, int p) {
  const uint16_t *a_ptr = CONVERT_TO_SHORTPTR(s8);
  uint16x8_t sum, a0, a1, a2, a3, a4, a5, a6, a7;

  load_u16_8x8(a_ptr, p, &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

  sum = vaddq_u16(a0, a1);
  sum = vaddq_u16(sum, a2);
  sum = vaddq_u16(sum, a3);
  sum = vaddq_u16(sum, a4);
  sum = vaddq_u16(sum, a5);
  sum = vaddq_u16(sum, a6);
  sum = vaddq_u16(sum, a7);

  return (horizontal_add_uint16x8(sum) + (1 << 5)) >> 6;
}

// coeff: 32 bits, dynamic range [-2147483648, 2147483647].
// length: value range {16, 64, 256, 1024}.
// satd: 42 bits, dynamic range [-2147483648 * 1024, 2147483647 * 1024]
int vpx_highbd_satd_neon(const tran_low_t *coeff, int length) {
  int64x2_t sum_s64[2] = { vdupq_n_s64(0), vdupq_n_s64(0) };

  do {
    int32x4_t abs0, abs1;
    const int32x4_t s0 = load_tran_low_to_s32q(coeff);
    const int32x4_t s1 = load_tran_low_to_s32q(coeff + 4);

    abs0 = vabsq_s32(s0);
    sum_s64[0] = vpadalq_s32(sum_s64[0], abs0);
    abs1 = vabsq_s32(s1);
    sum_s64[1] = vpadalq_s32(sum_s64[1], abs1);

    length -= 8;
    coeff += 8;
  } while (length != 0);

  return (int)horizontal_add_int64x2(vaddq_s64(sum_s64[0], sum_s64[1]));
}

void vpx_highbd_minmax_8x8_neon(const uint8_t *s8, int p, const uint8_t *d8,
                                int dp, int *min, int *max) {
  const uint16_t *a_ptr = CONVERT_TO_SHORTPTR(s8);
  const uint16_t *b_ptr = CONVERT_TO_SHORTPTR(d8);

  const uint16x8_t a0 = vld1q_u16(a_ptr + 0 * p);
  const uint16x8_t a1 = vld1q_u16(a_ptr + 1 * p);
  const uint16x8_t a2 = vld1q_u16(a_ptr + 2 * p);
  const uint16x8_t a3 = vld1q_u16(a_ptr + 3 * p);
  const uint16x8_t a4 = vld1q_u16(a_ptr + 4 * p);
  const uint16x8_t a5 = vld1q_u16(a_ptr + 5 * p);
  const uint16x8_t a6 = vld1q_u16(a_ptr + 6 * p);
  const uint16x8_t a7 = vld1q_u16(a_ptr + 7 * p);

  const uint16x8_t b0 = vld1q_u16(b_ptr + 0 * dp);
  const uint16x8_t b1 = vld1q_u16(b_ptr + 1 * dp);
  const uint16x8_t b2 = vld1q_u16(b_ptr + 2 * dp);
  const uint16x8_t b3 = vld1q_u16(b_ptr + 3 * dp);
  const uint16x8_t b4 = vld1q_u16(b_ptr + 4 * dp);
  const uint16x8_t b5 = vld1q_u16(b_ptr + 5 * dp);
  const uint16x8_t b6 = vld1q_u16(b_ptr + 6 * dp);
  const uint16x8_t b7 = vld1q_u16(b_ptr + 7 * dp);

  const uint16x8_t abs_diff0 = vabdq_u16(a0, b0);
  const uint16x8_t abs_diff1 = vabdq_u16(a1, b1);
  const uint16x8_t abs_diff2 = vabdq_u16(a2, b2);
  const uint16x8_t abs_diff3 = vabdq_u16(a3, b3);
  const uint16x8_t abs_diff4 = vabdq_u16(a4, b4);
  const uint16x8_t abs_diff5 = vabdq_u16(a5, b5);
  const uint16x8_t abs_diff6 = vabdq_u16(a6, b6);
  const uint16x8_t abs_diff7 = vabdq_u16(a7, b7);

  const uint16x8_t max01 = vmaxq_u16(abs_diff0, abs_diff1);
  const uint16x8_t max23 = vmaxq_u16(abs_diff2, abs_diff3);
  const uint16x8_t max45 = vmaxq_u16(abs_diff4, abs_diff5);
  const uint16x8_t max67 = vmaxq_u16(abs_diff6, abs_diff7);

  const uint16x8_t max0123 = vmaxq_u16(max01, max23);
  const uint16x8_t max4567 = vmaxq_u16(max45, max67);
  const uint16x8_t max07 = vmaxq_u16(max0123, max4567);

  const uint16x8_t min01 = vminq_u16(abs_diff0, abs_diff1);
  const uint16x8_t min23 = vminq_u16(abs_diff2, abs_diff3);
  const uint16x8_t min45 = vminq_u16(abs_diff4, abs_diff5);
  const uint16x8_t min67 = vminq_u16(abs_diff6, abs_diff7);

  const uint16x8_t min0123 = vminq_u16(min01, min23);
  const uint16x8_t min4567 = vminq_u16(min45, min67);
  const uint16x8_t min07 = vminq_u16(min0123, min4567);

#if VPX_ARCH_AARCH64
  *min = *max = 0;  // Clear high bits
  *((uint16_t *)max) = vmaxvq_u16(max07);
  *((uint16_t *)min) = vminvq_u16(min07);
#else
  // Split into 64-bit vectors and execute pairwise min/max.
  uint16x4_t ab_max = vmax_u16(vget_high_u16(max07), vget_low_u16(max07));
  uint16x4_t ab_min = vmin_u16(vget_high_u16(min07), vget_low_u16(min07));

  // Enough runs of vpmax/min propagate the max/min values to every position.
  ab_max = vpmax_u16(ab_max, ab_max);
  ab_min = vpmin_u16(ab_min, ab_min);

  ab_max = vpmax_u16(ab_max, ab_max);
  ab_min = vpmin_u16(ab_min, ab_min);

  ab_max = vpmax_u16(ab_max, ab_max);
  ab_min = vpmin_u16(ab_min, ab_min);

  *min = *max = 0;  // Clear high bits
  // Store directly to avoid costly neon->gpr transfer.
  vst1_lane_u16((uint16_t *)max, ab_max, 0);
  vst1_lane_u16((uint16_t *)min, ab_min, 0);
#endif
}
