/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_FDCT4X4_NEON_H_
#define VPX_VPX_DSP_ARM_FDCT4X4_NEON_H_

#include <arm_neon.h>

static INLINE void vpx_fdct4x4_pass1_neon(int16x4_t *in) {
  int16x4_t out[4];

  const int16x8_t input_01 = vcombine_s16(in[0], in[1]);
  const int16x8_t input_32 = vcombine_s16(in[3], in[2]);

  // in_0 +/- in_3, in_1 +/- in_2
  const int16x8_t s_01 = vaddq_s16(input_01, input_32);
  const int16x8_t s_32 = vsubq_s16(input_01, input_32);

  // step_0 +/- step_1, step_2 +/- step_3
  const int16x4_t s_0 = vget_low_s16(s_01);
  const int16x4_t s_1 = vget_high_s16(s_01);
  const int16x4_t s_2 = vget_high_s16(s_32);
  const int16x4_t s_3 = vget_low_s16(s_32);

  // fdct_round_shift(s_0 +/- s_1) * cospi_16_64
  butterfly_one_coeff_s16_fast_half(s_0, s_1, cospi_16_64, &out[0], &out[2]);

  // s_3 * cospi_8_64 + s_2 * cospi_24_64
  // s_3 * cospi_24_64 - s_2 * cospi_8_64
  butterfly_two_coeff_half(s_3, s_2, cospi_8_64, cospi_24_64, &out[1], &out[3]);

  transpose_s16_4x4d(&out[0], &out[1], &out[2], &out[3]);

  in[0] = out[0];
  in[1] = out[1];
  in[2] = out[2];
  in[3] = out[3];
}

static INLINE void vpx_fdct4x4_pass2_neon(int16x4_t *in) {
  int16x4_t out[4];

  const int16x8_t input_01 = vcombine_s16(in[0], in[1]);
  const int16x8_t input_32 = vcombine_s16(in[3], in[2]);

  // in_0 +/- in_3, in_1 +/- in_2
  const int16x8_t s_01 = vaddq_s16(input_01, input_32);
  const int16x8_t s_32 = vsubq_s16(input_01, input_32);

  // step_0 +/- step_1, step_2 +/- step_3
  const int16x4_t s_0 = vget_low_s16(s_01);
  const int16x4_t s_1 = vget_high_s16(s_01);
  const int16x4_t s_2 = vget_high_s16(s_32);
  const int16x4_t s_3 = vget_low_s16(s_32);

  // fdct_round_shift(s_0 +/- s_1) * cospi_16_64
  butterfly_one_coeff_s16_s32_fast_narrow_half(s_0, s_1, cospi_16_64, &out[0],
                                               &out[2]);

  // s_3 * cospi_8_64 + s_2 * cospi_24_64
  // s_3 * cospi_24_64 - s_2 * cospi_8_64
  butterfly_two_coeff_half(s_3, s_2, cospi_8_64, cospi_24_64, &out[1], &out[3]);

  transpose_s16_4x4d(&out[0], &out[1], &out[2], &out[3]);

  in[0] = out[0];
  in[1] = out[1];
  in[2] = out[2];
  in[3] = out[3];
}

#if CONFIG_VP9_HIGHBITDEPTH

static INLINE void vpx_highbd_fdct4x4_pass1_neon(int32x4_t *in) {
  int32x4_t out[4];
  // in_0 +/- in_3, in_1 +/- in_2
  const int32x4_t s_0 = vaddq_s32(in[0], in[3]);
  const int32x4_t s_1 = vaddq_s32(in[1], in[2]);
  const int32x4_t s_2 = vsubq_s32(in[1], in[2]);
  const int32x4_t s_3 = vsubq_s32(in[0], in[3]);

  butterfly_one_coeff_s32_fast_half(s_0, s_1, cospi_16_64, &out[0], &out[2]);

  // out[1] = s_3 * cospi_8_64 + s_2 * cospi_24_64
  // out[3] = s_3 * cospi_24_64 - s_2 * cospi_8_64
  butterfly_two_coeff_s32_s64_narrow_half(s_3, s_2, cospi_8_64, cospi_24_64,
                                          &out[1], &out[3]);

  transpose_s32_4x4(&out[0], &out[1], &out[2], &out[3]);

  in[0] = out[0];
  in[1] = out[1];
  in[2] = out[2];
  in[3] = out[3];
}

#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // VPX_VPX_DSP_ARM_FDCT4X4_NEON_H_
