/*
 *  Copyright (c) 2024 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_HIGHBD_CONVOLVE8_NEON_H_
#define VPX_VPX_DSP_ARM_HIGHBD_CONVOLVE8_NEON_H_

#include <arm_neon.h>

static INLINE uint16x4_t highbd_convolve4_4_neon(
    const int16x4_t s0, const int16x4_t s1, const int16x4_t s2,
    const int16x4_t s3, const int16x4_t filters, const uint16x4_t max) {
  int32x4_t sum = vmull_lane_s16(s0, filters, 0);
  sum = vmlal_lane_s16(sum, s1, filters, 1);
  sum = vmlal_lane_s16(sum, s2, filters, 2);
  sum = vmlal_lane_s16(sum, s3, filters, 3);

  uint16x4_t res = vqrshrun_n_s32(sum, FILTER_BITS);
  return vmin_u16(res, max);
}

static INLINE uint16x8_t highbd_convolve4_8_neon(
    const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
    const int16x8_t s3, const int16x4_t filters, const uint16x8_t max) {
  int32x4_t sum0 = vmull_lane_s16(vget_low_s16(s0), filters, 0);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s1), filters, 1);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s2), filters, 2);
  sum0 = vmlal_lane_s16(sum0, vget_low_s16(s3), filters, 3);

  int32x4_t sum1 = vmull_lane_s16(vget_high_s16(s0), filters, 0);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s1), filters, 1);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s2), filters, 2);
  sum1 = vmlal_lane_s16(sum1, vget_high_s16(s3), filters, 3);

  uint16x8_t res = vcombine_u16(vqrshrun_n_s32(sum0, FILTER_BITS),
                                vqrshrun_n_s32(sum1, FILTER_BITS));
  return vminq_u16(res, max);
}

#endif  // VPX_VPX_DSP_ARM_HIGHBD_CONVOLVE8_NEON_H_
