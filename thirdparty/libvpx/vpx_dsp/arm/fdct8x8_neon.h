/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_FDCT8X8_NEON_H_
#define VPX_VPX_DSP_ARM_FDCT8X8_NEON_H_

#include <arm_neon.h>

static INLINE void vpx_fdct8x8_pass1_notranspose_neon(int16x8_t *in,
                                                      int16x8_t *out) {
  int16x8_t s[8], x[4], t[2];

  s[0] = vaddq_s16(in[0], in[7]);
  s[1] = vaddq_s16(in[1], in[6]);
  s[2] = vaddq_s16(in[2], in[5]);
  s[3] = vaddq_s16(in[3], in[4]);
  s[4] = vsubq_s16(in[3], in[4]);
  s[5] = vsubq_s16(in[2], in[5]);
  s[6] = vsubq_s16(in[1], in[6]);
  s[7] = vsubq_s16(in[0], in[7]);
  // fdct4(step, step);
  x[0] = vaddq_s16(s[0], s[3]);
  x[1] = vaddq_s16(s[1], s[2]);
  x[2] = vsubq_s16(s[1], s[2]);
  x[3] = vsubq_s16(s[0], s[3]);

  // fdct4(step, step);
  // out[0] = (tran_low_t)fdct_round_shift((x0 + x1) * cospi_16_64)
  // out[4] = (tran_low_t)fdct_round_shift((x0 - x1) * cospi_16_64)
  butterfly_one_coeff_s16_fast(x[0], x[1], cospi_16_64, &out[0], &out[4]);
  // out[2] = (tran_low_t)fdct_round_shift(x2 * cospi_24_64 + x3 * cospi_8_64)
  // out[6] = (tran_low_t)fdct_round_shift(-x2 * cospi_8_64 + x3 * cospi_24_64)
  butterfly_two_coeff(x[3], x[2], cospi_8_64, cospi_24_64, &out[2], &out[6]);

  // Stage 2
  // t0 = (s6 - s5) * cospi_16_64;
  // t1 = (s6 + s5) * cospi_16_64;
  butterfly_one_coeff_s16_fast(s[6], s[5], cospi_16_64, &t[1], &t[0]);

  // Stage 3
  x[0] = vaddq_s16(s[4], t[0]);
  x[1] = vsubq_s16(s[4], t[0]);
  x[2] = vsubq_s16(s[7], t[1]);
  x[3] = vaddq_s16(s[7], t[1]);

  // Stage 4
  // out[1] = (tran_low_t)fdct_round_shift(x0 * cospi_28_64 + x3 * cospi_4_64)
  // out[7] = (tran_low_t)fdct_round_shift(x3 * cospi_28_64 + x0 * -cospi_4_64)
  butterfly_two_coeff(x[3], x[0], cospi_4_64, cospi_28_64, &out[1], &out[7]);

  // out[5] = (tran_low_t)fdct_round_shift(x1 * cospi_12_64 + x2 * cospi_20_64)
  // out[3] = (tran_low_t)fdct_round_shift(x2 * cospi_12_64 + x1 * -cospi_20_64)
  butterfly_two_coeff(x[2], x[1], cospi_20_64, cospi_12_64, &out[5], &out[3]);
}

static INLINE void vpx_fdct8x8_pass2_notranspose_neon(int16x8_t *in,
                                                      int16x8_t *out) {
  int16x8_t s[8], x[4], t[2];

  s[0] = vaddq_s16(in[0], in[7]);
  s[1] = vaddq_s16(in[1], in[6]);
  s[2] = vaddq_s16(in[2], in[5]);
  s[3] = vaddq_s16(in[3], in[4]);
  s[4] = vsubq_s16(in[3], in[4]);
  s[5] = vsubq_s16(in[2], in[5]);
  s[6] = vsubq_s16(in[1], in[6]);
  s[7] = vsubq_s16(in[0], in[7]);
  // fdct4(step, step);
  x[0] = vaddq_s16(s[0], s[3]);
  x[1] = vaddq_s16(s[1], s[2]);
  x[2] = vsubq_s16(s[1], s[2]);
  x[3] = vsubq_s16(s[0], s[3]);

  // fdct4(step, step);
  // out[0] = (tran_low_t)fdct_round_shift((x0 + x1) * cospi_16_64)
  // out[4] = (tran_low_t)fdct_round_shift((x0 - x1) * cospi_16_64)
  butterfly_one_coeff_s16_s32_fast_narrow(x[0], x[1], cospi_16_64, &out[0],
                                          &out[4]);
  // out[2] = (tran_low_t)fdct_round_shift(x2 * cospi_24_64 + x3 * cospi_8_64)
  // out[6] = (tran_low_t)fdct_round_shift(-x2 * cospi_8_64 + x3 * cospi_24_64)
  butterfly_two_coeff(x[3], x[2], cospi_8_64, cospi_24_64, &out[2], &out[6]);

  // Stage 2
  // t0 = (s6 - s5) * cospi_16_64;
  // t1 = (s6 + s5) * cospi_16_64;
  butterfly_one_coeff_s16_s32_fast_narrow(s[6], s[5], cospi_16_64, &t[1],
                                          &t[0]);

  // Stage 3
  x[0] = vaddq_s16(s[4], t[0]);
  x[1] = vsubq_s16(s[4], t[0]);
  x[2] = vsubq_s16(s[7], t[1]);
  x[3] = vaddq_s16(s[7], t[1]);

  // Stage 4
  // out[1] = (tran_low_t)fdct_round_shift(x0 * cospi_28_64 + x3 * cospi_4_64)
  // out[7] = (tran_low_t)fdct_round_shift(x3 * cospi_28_64 + x0 * -cospi_4_64)
  butterfly_two_coeff(x[3], x[0], cospi_4_64, cospi_28_64, &out[1], &out[7]);

  // out[5] = (tran_low_t)fdct_round_shift(x1 * cospi_12_64 + x2 * cospi_20_64)
  // out[3] = (tran_low_t)fdct_round_shift(x2 * cospi_12_64 + x1 * -cospi_20_64)
  butterfly_two_coeff(x[2], x[1], cospi_20_64, cospi_12_64, &out[5], &out[3]);
}

static INLINE void vpx_fdct8x8_pass1_neon(int16x8_t *in) {
  int16x8_t out[8];
  vpx_fdct8x8_pass1_notranspose_neon(in, out);
  // transpose 8x8
  transpose_s16_8x8(&out[0], &out[1], &out[2], &out[3], &out[4], &out[5],
                    &out[6], &out[7]);
  in[0] = out[0];
  in[1] = out[1];
  in[2] = out[2];
  in[3] = out[3];
  in[4] = out[4];
  in[5] = out[5];
  in[6] = out[6];
  in[7] = out[7];
}

static INLINE void vpx_fdct8x8_pass2_neon(int16x8_t *in) {
  int16x8_t out[8];
  vpx_fdct8x8_pass2_notranspose_neon(in, out);
  // transpose 8x8
  transpose_s16_8x8(&out[0], &out[1], &out[2], &out[3], &out[4], &out[5],
                    &out[6], &out[7]);
  in[0] = out[0];
  in[1] = out[1];
  in[2] = out[2];
  in[3] = out[3];
  in[4] = out[4];
  in[5] = out[5];
  in[6] = out[6];
  in[7] = out[7];
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE void vpx_highbd_fdct8x8_pass1_notranspose_neon(int32x4_t *left,
                                                             int32x4_t *right) {
  int32x4_t sl[8], sr[8], xl[4], xr[4], tl[4], tr[4];

  sl[0] = vaddq_s32(left[0], left[7]);
  sl[1] = vaddq_s32(left[1], left[6]);
  sl[2] = vaddq_s32(left[2], left[5]);
  sl[3] = vaddq_s32(left[3], left[4]);
  sl[4] = vsubq_s32(left[3], left[4]);
  sl[5] = vsubq_s32(left[2], left[5]);
  sl[6] = vsubq_s32(left[1], left[6]);
  sl[7] = vsubq_s32(left[0], left[7]);
  sr[0] = vaddq_s32(right[0], right[7]);
  sr[1] = vaddq_s32(right[1], right[6]);
  sr[2] = vaddq_s32(right[2], right[5]);
  sr[3] = vaddq_s32(right[3], right[4]);
  sr[4] = vsubq_s32(right[3], right[4]);
  sr[5] = vsubq_s32(right[2], right[5]);
  sr[6] = vsubq_s32(right[1], right[6]);
  sr[7] = vsubq_s32(right[0], right[7]);

  // fdct4(step, step);
  // x0 = s0 + s3;
  xl[0] = vaddq_s32(sl[0], sl[3]);
  xr[0] = vaddq_s32(sr[0], sr[3]);
  // x1 = s1 + s2;
  xl[1] = vaddq_s32(sl[1], sl[2]);
  xr[1] = vaddq_s32(sr[1], sr[2]);
  // x2 = s1 - s2;
  xl[2] = vsubq_s32(sl[1], sl[2]);
  xr[2] = vsubq_s32(sr[1], sr[2]);
  // x3 = s0 - s3;
  xl[3] = vsubq_s32(sl[0], sl[3]);
  xr[3] = vsubq_s32(sr[0], sr[3]);

  // fdct4(step, step);
  // out[0] = (tran_low_t)fdct_round_shift((x0 + x1) * cospi_16_64)
  // out[4] = (tran_low_t)fdct_round_shift((x0 - x1) * cospi_16_64)
  butterfly_one_coeff_s32_fast(xl[0], xr[0], xl[1], xr[1], cospi_16_64,
                               &left[0], &right[0], &left[4], &right[4]);
  // out[2] = (tran_low_t)fdct_round_shift(x2 * cospi_24_64 + x3 * cospi_8_64)
  // out[6] = (tran_low_t)fdct_round_shift(-x2 * cospi_8_64 + x3 * cospi_24_64)
  butterfly_two_coeff_s32(xl[3], xr[3], xl[2], xr[2], cospi_8_64, cospi_24_64,
                          &left[2], &right[2], &left[6], &right[6]);

  // Stage 2
  // t0 = (s6 - s5) * cospi_16_64;
  // t1 = (s6 + s5) * cospi_16_64;
  butterfly_one_coeff_s32_fast(sl[6], sr[6], sl[5], sr[5], cospi_16_64, &tl[1],
                               &tr[1], &tl[0], &tr[0]);

  // Stage 3
  xl[0] = vaddq_s32(sl[4], tl[0]);
  xr[0] = vaddq_s32(sr[4], tr[0]);
  xl[1] = vsubq_s32(sl[4], tl[0]);
  xr[1] = vsubq_s32(sr[4], tr[0]);
  xl[2] = vsubq_s32(sl[7], tl[1]);
  xr[2] = vsubq_s32(sr[7], tr[1]);
  xl[3] = vaddq_s32(sl[7], tl[1]);
  xr[3] = vaddq_s32(sr[7], tr[1]);

  // Stage 4
  // out[1] = (tran_low_t)fdct_round_shift(x0 * cospi_28_64 + x3 * cospi_4_64)
  // out[7] = (tran_low_t)fdct_round_shift(x3 * cospi_28_64 + x0 * -cospi_4_64)
  butterfly_two_coeff_s32(xl[3], xr[3], xl[0], xr[0], cospi_4_64, cospi_28_64,
                          &left[1], &right[1], &left[7], &right[7]);

  // out[5] = (tran_low_t)fdct_round_shift(x1 * cospi_12_64 + x2 * cospi_20_64)
  // out[3] = (tran_low_t)fdct_round_shift(x2 * cospi_12_64 + x1 * -cospi_20_64)
  butterfly_two_coeff_s32(xl[2], xr[2], xl[1], xr[1], cospi_20_64, cospi_12_64,
                          &left[5], &right[5], &left[3], &right[3]);
}

static INLINE void vpx_highbd_fdct8x8_pass2_notranspose_neon(int32x4_t *left,
                                                             int32x4_t *right) {
  int32x4_t sl[8], sr[8], xl[4], xr[4], tl[4], tr[4];

  sl[0] = vaddq_s32(left[0], left[7]);
  sl[1] = vaddq_s32(left[1], left[6]);
  sl[2] = vaddq_s32(left[2], left[5]);
  sl[3] = vaddq_s32(left[3], left[4]);
  sl[4] = vsubq_s32(left[3], left[4]);
  sl[5] = vsubq_s32(left[2], left[5]);
  sl[6] = vsubq_s32(left[1], left[6]);
  sl[7] = vsubq_s32(left[0], left[7]);
  sr[0] = vaddq_s32(right[0], right[7]);
  sr[1] = vaddq_s32(right[1], right[6]);
  sr[2] = vaddq_s32(right[2], right[5]);
  sr[3] = vaddq_s32(right[3], right[4]);
  sr[4] = vsubq_s32(right[3], right[4]);
  sr[5] = vsubq_s32(right[2], right[5]);
  sr[6] = vsubq_s32(right[1], right[6]);
  sr[7] = vsubq_s32(right[0], right[7]);

  // fdct4(step, step);
  // x0 = s0 + s3;
  xl[0] = vaddq_s32(sl[0], sl[3]);
  xr[0] = vaddq_s32(sr[0], sr[3]);
  // x1 = s1 + s2;
  xl[1] = vaddq_s32(sl[1], sl[2]);
  xr[1] = vaddq_s32(sr[1], sr[2]);
  // x2 = s1 - s2;
  xl[2] = vsubq_s32(sl[1], sl[2]);
  xr[2] = vsubq_s32(sr[1], sr[2]);
  // x3 = s0 - s3;
  xl[3] = vsubq_s32(sl[0], sl[3]);
  xr[3] = vsubq_s32(sr[0], sr[3]);

  // fdct4(step, step);
  // out[0] = (tran_low_t)fdct_round_shift((x0 + x1) * cospi_16_64)
  // out[4] = (tran_low_t)fdct_round_shift((x0 - x1) * cospi_16_64)
  butterfly_one_coeff_s32_fast(xl[0], xr[0], xl[1], xr[1], cospi_16_64,
                               &left[0], &right[0], &left[4], &right[4]);
  // out[2] = (tran_low_t)fdct_round_shift(x2 * cospi_24_64 + x3 * cospi_8_64)
  // out[6] = (tran_low_t)fdct_round_shift(-x2 * cospi_8_64 + x3 * cospi_24_64)
  butterfly_two_coeff_s32_s64_narrow(xl[3], xr[3], xl[2], xr[2], cospi_8_64,
                                     cospi_24_64, &left[2], &right[2], &left[6],
                                     &right[6]);

  // Stage 2
  // t0 = (s6 - s5) * cospi_16_64;
  // t1 = (s6 + s5) * cospi_16_64;
  butterfly_one_coeff_s32_fast(sl[6], sr[6], sl[5], sr[5], cospi_16_64, &tl[1],
                               &tr[1], &tl[0], &tr[0]);

  // Stage 3
  xl[0] = vaddq_s32(sl[4], tl[0]);
  xr[0] = vaddq_s32(sr[4], tr[0]);
  xl[1] = vsubq_s32(sl[4], tl[0]);
  xr[1] = vsubq_s32(sr[4], tr[0]);
  xl[2] = vsubq_s32(sl[7], tl[1]);
  xr[2] = vsubq_s32(sr[7], tr[1]);
  xl[3] = vaddq_s32(sl[7], tl[1]);
  xr[3] = vaddq_s32(sr[7], tr[1]);

  // Stage 4
  // out[1] = (tran_low_t)fdct_round_shift(x0 * cospi_28_64 + x3 * cospi_4_64)
  // out[7] = (tran_low_t)fdct_round_shift(x3 * cospi_28_64 + x0 * -cospi_4_64)
  butterfly_two_coeff_s32_s64_narrow(xl[3], xr[3], xl[0], xr[0], cospi_4_64,
                                     cospi_28_64, &left[1], &right[1], &left[7],
                                     &right[7]);

  // out[5] = (tran_low_t)fdct_round_shift(x1 * cospi_12_64 + x2 * cospi_20_64)
  // out[3] = (tran_low_t)fdct_round_shift(x2 * cospi_12_64 + x1 * -cospi_20_64)
  butterfly_two_coeff_s32_s64_narrow(xl[2], xr[2], xl[1], xr[1], cospi_20_64,
                                     cospi_12_64, &left[5], &right[5], &left[3],
                                     &right[3]);
}

static INLINE void vpx_highbd_fdct8x8_pass1_neon(int32x4_t *left,
                                                 int32x4_t *right) {
  vpx_highbd_fdct8x8_pass1_notranspose_neon(left, right);
  transpose_s32_8x8_2(left, right, left, right);
}

static INLINE void vpx_highbd_fdct8x8_pass2_neon(int32x4_t *left,
                                                 int32x4_t *right) {
  vpx_highbd_fdct8x8_pass2_notranspose_neon(left, right);
  transpose_s32_8x8_2(left, right, left, right);
}

#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // VPX_VPX_DSP_ARM_FDCT8X8_NEON_H_
