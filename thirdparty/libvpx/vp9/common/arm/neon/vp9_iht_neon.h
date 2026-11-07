/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_COMMON_ARM_NEON_VP9_IHT_NEON_H_
#define VPX_VP9_COMMON_ARM_NEON_VP9_IHT_NEON_H_

#include <arm_neon.h>

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "vp9/common/vp9_common.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/txfm_common.h"

static INLINE void iadst4(int16x8_t *const io) {
  const int32x4_t c3 = vdupq_n_s32(sinpi_3_9);
  int16x4_t x[4];
  int32x4_t s[8], output[4];
  const int16x4_t c =
      create_s16x4_neon(sinpi_1_9, sinpi_2_9, sinpi_3_9, sinpi_4_9);

  x[0] = vget_low_s16(io[0]);
  x[1] = vget_low_s16(io[1]);
  x[2] = vget_high_s16(io[0]);
  x[3] = vget_high_s16(io[1]);

  s[0] = vmull_lane_s16(x[0], c, 0);
  s[1] = vmull_lane_s16(x[0], c, 1);
  s[2] = vmull_lane_s16(x[1], c, 2);
  s[3] = vmull_lane_s16(x[2], c, 3);
  s[4] = vmull_lane_s16(x[2], c, 0);
  s[5] = vmull_lane_s16(x[3], c, 1);
  s[6] = vmull_lane_s16(x[3], c, 3);
  s[7] = vaddl_s16(x[0], x[3]);
  s[7] = vsubw_s16(s[7], x[2]);

  s[0] = vaddq_s32(s[0], s[3]);
  s[0] = vaddq_s32(s[0], s[5]);
  s[1] = vsubq_s32(s[1], s[4]);
  s[1] = vsubq_s32(s[1], s[6]);
  s[3] = s[2];
  s[2] = vmulq_s32(c3, s[7]);

  output[0] = vaddq_s32(s[0], s[3]);
  output[1] = vaddq_s32(s[1], s[3]);
  output[2] = s[2];
  output[3] = vaddq_s32(s[0], s[1]);
  output[3] = vsubq_s32(output[3], s[3]);
  dct_const_round_shift_low_8_dual(output, &io[0], &io[1]);
}

static INLINE void iadst_half_butterfly_neon(int16x8_t *const x,
                                             const int16x4_t c) {
  // Don't add/sub before multiply, which will overflow in iadst8.
  const int32x4_t x0_lo = vmull_lane_s16(vget_low_s16(x[0]), c, 0);
  const int32x4_t x0_hi = vmull_lane_s16(vget_high_s16(x[0]), c, 0);
  const int32x4_t x1_lo = vmull_lane_s16(vget_low_s16(x[1]), c, 0);
  const int32x4_t x1_hi = vmull_lane_s16(vget_high_s16(x[1]), c, 0);
  int32x4_t t0[2], t1[2];

  t0[0] = vaddq_s32(x0_lo, x1_lo);
  t0[1] = vaddq_s32(x0_hi, x1_hi);
  t1[0] = vsubq_s32(x0_lo, x1_lo);
  t1[1] = vsubq_s32(x0_hi, x1_hi);
  x[0] = dct_const_round_shift_low_8(t0);
  x[1] = dct_const_round_shift_low_8(t1);
}

static INLINE void iadst_half_butterfly_neg_neon(int16x8_t *const x0,
                                                 int16x8_t *const x1,
                                                 const int16x4_t c) {
  // Don't add/sub before multiply, which will overflow in iadst8.
  const int32x4_t x0_lo = vmull_lane_s16(vget_low_s16(*x0), c, 1);
  const int32x4_t x0_hi = vmull_lane_s16(vget_high_s16(*x0), c, 1);
  const int32x4_t x1_lo = vmull_lane_s16(vget_low_s16(*x1), c, 1);
  const int32x4_t x1_hi = vmull_lane_s16(vget_high_s16(*x1), c, 1);
  int32x4_t t0[2], t1[2];

  t0[0] = vaddq_s32(x0_lo, x1_lo);
  t0[1] = vaddq_s32(x0_hi, x1_hi);
  t1[0] = vsubq_s32(x0_lo, x1_lo);
  t1[1] = vsubq_s32(x0_hi, x1_hi);
  *x1 = dct_const_round_shift_low_8(t0);
  *x0 = dct_const_round_shift_low_8(t1);
}

static INLINE void iadst_half_butterfly_pos_neon(int16x8_t *const x0,
                                                 int16x8_t *const x1,
                                                 const int16x4_t c) {
  // Don't add/sub before multiply, which will overflow in iadst8.
  const int32x4_t x0_lo = vmull_lane_s16(vget_low_s16(*x0), c, 0);
  const int32x4_t x0_hi = vmull_lane_s16(vget_high_s16(*x0), c, 0);
  const int32x4_t x1_lo = vmull_lane_s16(vget_low_s16(*x1), c, 0);
  const int32x4_t x1_hi = vmull_lane_s16(vget_high_s16(*x1), c, 0);
  int32x4_t t0[2], t1[2];

  t0[0] = vaddq_s32(x0_lo, x1_lo);
  t0[1] = vaddq_s32(x0_hi, x1_hi);
  t1[0] = vsubq_s32(x0_lo, x1_lo);
  t1[1] = vsubq_s32(x0_hi, x1_hi);
  *x1 = dct_const_round_shift_low_8(t0);
  *x0 = dct_const_round_shift_low_8(t1);
}

static INLINE void iadst_butterfly_lane_0_1_neon(const int16x8_t in0,
                                                 const int16x8_t in1,
                                                 const int16x4_t c,
                                                 int32x4_t *const s0,
                                                 int32x4_t *const s1) {
  s0[0] = vmull_lane_s16(vget_low_s16(in0), c, 0);
  s0[1] = vmull_lane_s16(vget_high_s16(in0), c, 0);
  s1[0] = vmull_lane_s16(vget_low_s16(in0), c, 1);
  s1[1] = vmull_lane_s16(vget_high_s16(in0), c, 1);

  s0[0] = vmlal_lane_s16(s0[0], vget_low_s16(in1), c, 1);
  s0[1] = vmlal_lane_s16(s0[1], vget_high_s16(in1), c, 1);
  s1[0] = vmlsl_lane_s16(s1[0], vget_low_s16(in1), c, 0);
  s1[1] = vmlsl_lane_s16(s1[1], vget_high_s16(in1), c, 0);
}

static INLINE void iadst_butterfly_lane_2_3_neon(const int16x8_t in0,
                                                 const int16x8_t in1,
                                                 const int16x4_t c,
                                                 int32x4_t *const s0,
                                                 int32x4_t *const s1) {
  s0[0] = vmull_lane_s16(vget_low_s16(in0), c, 2);
  s0[1] = vmull_lane_s16(vget_high_s16(in0), c, 2);
  s1[0] = vmull_lane_s16(vget_low_s16(in0), c, 3);
  s1[1] = vmull_lane_s16(vget_high_s16(in0), c, 3);

  s0[0] = vmlal_lane_s16(s0[0], vget_low_s16(in1), c, 3);
  s0[1] = vmlal_lane_s16(s0[1], vget_high_s16(in1), c, 3);
  s1[0] = vmlsl_lane_s16(s1[0], vget_low_s16(in1), c, 2);
  s1[1] = vmlsl_lane_s16(s1[1], vget_high_s16(in1), c, 2);
}

static INLINE void iadst_butterfly_lane_1_0_neon(const int16x8_t in0,
                                                 const int16x8_t in1,
                                                 const int16x4_t c,
                                                 int32x4_t *const s0,
                                                 int32x4_t *const s1) {
  s0[0] = vmull_lane_s16(vget_low_s16(in0), c, 1);
  s0[1] = vmull_lane_s16(vget_high_s16(in0), c, 1);
  s1[0] = vmull_lane_s16(vget_low_s16(in0), c, 0);
  s1[1] = vmull_lane_s16(vget_high_s16(in0), c, 0);

  s0[0] = vmlal_lane_s16(s0[0], vget_low_s16(in1), c, 0);
  s0[1] = vmlal_lane_s16(s0[1], vget_high_s16(in1), c, 0);
  s1[0] = vmlsl_lane_s16(s1[0], vget_low_s16(in1), c, 1);
  s1[1] = vmlsl_lane_s16(s1[1], vget_high_s16(in1), c, 1);
}

static INLINE void iadst_butterfly_lane_3_2_neon(const int16x8_t in0,
                                                 const int16x8_t in1,
                                                 const int16x4_t c,
                                                 int32x4_t *const s0,
                                                 int32x4_t *const s1) {
  s0[0] = vmull_lane_s16(vget_low_s16(in0), c, 3);
  s0[1] = vmull_lane_s16(vget_high_s16(in0), c, 3);
  s1[0] = vmull_lane_s16(vget_low_s16(in0), c, 2);
  s1[1] = vmull_lane_s16(vget_high_s16(in0), c, 2);

  s0[0] = vmlal_lane_s16(s0[0], vget_low_s16(in1), c, 2);
  s0[1] = vmlal_lane_s16(s0[1], vget_high_s16(in1), c, 2);
  s1[0] = vmlsl_lane_s16(s1[0], vget_low_s16(in1), c, 3);
  s1[1] = vmlsl_lane_s16(s1[1], vget_high_s16(in1), c, 3);
}

static INLINE int16x8_t add_dct_const_round_shift_low_8(
    const int32x4_t *const in0, const int32x4_t *const in1) {
  int32x4_t sum[2];

  sum[0] = vaddq_s32(in0[0], in1[0]);
  sum[1] = vaddq_s32(in0[1], in1[1]);
  return dct_const_round_shift_low_8(sum);
}

static INLINE int16x8_t sub_dct_const_round_shift_low_8(
    const int32x4_t *const in0, const int32x4_t *const in1) {
  int32x4_t sum[2];

  sum[0] = vsubq_s32(in0[0], in1[0]);
  sum[1] = vsubq_s32(in0[1], in1[1]);
  return dct_const_round_shift_low_8(sum);
}

static INLINE void iadst8(int16x8_t *const io) {
  const int16x4_t c0 =
      create_s16x4_neon(cospi_2_64, cospi_30_64, cospi_10_64, cospi_22_64);
  const int16x4_t c1 =
      create_s16x4_neon(cospi_18_64, cospi_14_64, cospi_26_64, cospi_6_64);
  const int16x4_t c2 =
      create_s16x4_neon(cospi_16_64, 0, cospi_8_64, cospi_24_64);
  int16x8_t x[8], t[4];
  int32x4_t s0[2], s1[2], s2[2], s3[2], s4[2], s5[2], s6[2], s7[2];

  x[0] = io[7];
  x[1] = io[0];
  x[2] = io[5];
  x[3] = io[2];
  x[4] = io[3];
  x[5] = io[4];
  x[6] = io[1];
  x[7] = io[6];

  // stage 1
  iadst_butterfly_lane_0_1_neon(x[0], x[1], c0, s0, s1);
  iadst_butterfly_lane_2_3_neon(x[2], x[3], c0, s2, s3);
  iadst_butterfly_lane_0_1_neon(x[4], x[5], c1, s4, s5);
  iadst_butterfly_lane_2_3_neon(x[6], x[7], c1, s6, s7);

  x[0] = add_dct_const_round_shift_low_8(s0, s4);
  x[1] = add_dct_const_round_shift_low_8(s1, s5);
  x[2] = add_dct_const_round_shift_low_8(s2, s6);
  x[3] = add_dct_const_round_shift_low_8(s3, s7);
  x[4] = sub_dct_const_round_shift_low_8(s0, s4);
  x[5] = sub_dct_const_round_shift_low_8(s1, s5);
  x[6] = sub_dct_const_round_shift_low_8(s2, s6);
  x[7] = sub_dct_const_round_shift_low_8(s3, s7);

  // stage 2
  t[0] = x[0];
  t[1] = x[1];
  t[2] = x[2];
  t[3] = x[3];
  iadst_butterfly_lane_2_3_neon(x[4], x[5], c2, s4, s5);
  iadst_butterfly_lane_3_2_neon(x[7], x[6], c2, s7, s6);

  x[0] = vaddq_s16(t[0], t[2]);
  x[1] = vaddq_s16(t[1], t[3]);
  x[2] = vsubq_s16(t[0], t[2]);
  x[3] = vsubq_s16(t[1], t[3]);
  x[4] = add_dct_const_round_shift_low_8(s4, s6);
  x[5] = add_dct_const_round_shift_low_8(s5, s7);
  x[6] = sub_dct_const_round_shift_low_8(s4, s6);
  x[7] = sub_dct_const_round_shift_low_8(s5, s7);

  // stage 3
  iadst_half_butterfly_neon(x + 2, c2);
  iadst_half_butterfly_neon(x + 6, c2);

  io[0] = x[0];
  io[1] = vnegq_s16(x[4]);
  io[2] = x[6];
  io[3] = vnegq_s16(x[2]);
  io[4] = x[3];
  io[5] = vnegq_s16(x[7]);
  io[6] = x[5];
  io[7] = vnegq_s16(x[1]);
}

void vpx_iadst16x16_256_add_half1d(const void *const input, int16_t *output,
                                   void *const dest, const int stride,
                                   const int highbd_flag);

typedef void (*iht_1d)(const void *const input, int16_t *output,
                       void *const dest, const int stride,
                       const int highbd_flag);

typedef struct {
  iht_1d cols, rows;  // vertical and horizontal
} iht_2d;

#endif  // VPX_VP9_COMMON_ARM_NEON_VP9_IHT_NEON_H_
