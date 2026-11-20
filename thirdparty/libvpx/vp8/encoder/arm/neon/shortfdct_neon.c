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

#include "./vp8_rtcd.h"

void vp8_short_fdct4x4_neon(int16_t *input, int16_t *output, int pitch) {
  int16x4_t d0s16, d1s16, d2s16, d3s16, d4s16, d5s16, d6s16, d7s16;
  int16x4_t d16s16, d17s16, d26s16, dEmptys16;
  uint16x4_t d4u16;
  int16x8_t q0s16, q1s16;
  int32x4_t q9s32, q10s32, q11s32, q12s32;
  int16x4x2_t v2tmp0, v2tmp1;
  int32x2x2_t v2tmp2, v2tmp3;

  d16s16 = vdup_n_s16(5352);
  d17s16 = vdup_n_s16(2217);
  q9s32 = vdupq_n_s32(14500);
  q10s32 = vdupq_n_s32(7500);
  q11s32 = vdupq_n_s32(12000);
  q12s32 = vdupq_n_s32(51000);

  // Part one
  pitch >>= 1;
  d0s16 = vld1_s16(input);
  input += pitch;
  d1s16 = vld1_s16(input);
  input += pitch;
  d2s16 = vld1_s16(input);
  input += pitch;
  d3s16 = vld1_s16(input);

  v2tmp2 = vtrn_s32(vreinterpret_s32_s16(d0s16), vreinterpret_s32_s16(d2s16));
  v2tmp3 = vtrn_s32(vreinterpret_s32_s16(d1s16), vreinterpret_s32_s16(d3s16));
  v2tmp0 = vtrn_s16(vreinterpret_s16_s32(v2tmp2.val[0]),   // d0
                    vreinterpret_s16_s32(v2tmp3.val[0]));  // d1
  v2tmp1 = vtrn_s16(vreinterpret_s16_s32(v2tmp2.val[1]),   // d2
                    vreinterpret_s16_s32(v2tmp3.val[1]));  // d3

  d4s16 = vadd_s16(v2tmp0.val[0], v2tmp1.val[1]);
  d5s16 = vadd_s16(v2tmp0.val[1], v2tmp1.val[0]);
  d6s16 = vsub_s16(v2tmp0.val[1], v2tmp1.val[0]);
  d7s16 = vsub_s16(v2tmp0.val[0], v2tmp1.val[1]);

  d4s16 = vshl_n_s16(d4s16, 3);
  d5s16 = vshl_n_s16(d5s16, 3);
  d6s16 = vshl_n_s16(d6s16, 3);
  d7s16 = vshl_n_s16(d7s16, 3);

  d0s16 = vadd_s16(d4s16, d5s16);
  d2s16 = vsub_s16(d4s16, d5s16);

  q9s32 = vmlal_s16(q9s32, d7s16, d16s16);
  q10s32 = vmlal_s16(q10s32, d7s16, d17s16);
  q9s32 = vmlal_s16(q9s32, d6s16, d17s16);
  q10s32 = vmlsl_s16(q10s32, d6s16, d16s16);

  d1s16 = vshrn_n_s32(q9s32, 12);
  d3s16 = vshrn_n_s32(q10s32, 12);

  // Part two
  v2tmp2 = vtrn_s32(vreinterpret_s32_s16(d0s16), vreinterpret_s32_s16(d2s16));
  v2tmp3 = vtrn_s32(vreinterpret_s32_s16(d1s16), vreinterpret_s32_s16(d3s16));
  v2tmp0 = vtrn_s16(vreinterpret_s16_s32(v2tmp2.val[0]),   // d0
                    vreinterpret_s16_s32(v2tmp3.val[0]));  // d1
  v2tmp1 = vtrn_s16(vreinterpret_s16_s32(v2tmp2.val[1]),   // d2
                    vreinterpret_s16_s32(v2tmp3.val[1]));  // d3

  d4s16 = vadd_s16(v2tmp0.val[0], v2tmp1.val[1]);
  d5s16 = vadd_s16(v2tmp0.val[1], v2tmp1.val[0]);
  d6s16 = vsub_s16(v2tmp0.val[1], v2tmp1.val[0]);
  d7s16 = vsub_s16(v2tmp0.val[0], v2tmp1.val[1]);

  d26s16 = vdup_n_s16(7);
  d4s16 = vadd_s16(d4s16, d26s16);

  d0s16 = vadd_s16(d4s16, d5s16);
  d2s16 = vsub_s16(d4s16, d5s16);

  q11s32 = vmlal_s16(q11s32, d7s16, d16s16);
  q12s32 = vmlal_s16(q12s32, d7s16, d17s16);

  dEmptys16 = vdup_n_s16(0);
  d4u16 = vceq_s16(d7s16, dEmptys16);

  d0s16 = vshr_n_s16(d0s16, 4);
  d2s16 = vshr_n_s16(d2s16, 4);

  q11s32 = vmlal_s16(q11s32, d6s16, d17s16);
  q12s32 = vmlsl_s16(q12s32, d6s16, d16s16);

  d4u16 = vmvn_u16(d4u16);
  d1s16 = vshrn_n_s32(q11s32, 16);
  d1s16 = vsub_s16(d1s16, vreinterpret_s16_u16(d4u16));
  d3s16 = vshrn_n_s32(q12s32, 16);

  q0s16 = vcombine_s16(d0s16, d1s16);
  q1s16 = vcombine_s16(d2s16, d3s16);

  vst1q_s16(output, q0s16);
  vst1q_s16(output + 8, q1s16);
  return;
}

void vp8_short_fdct8x4_neon(int16_t *input, int16_t *output, int pitch) {
  int16x4_t d0s16, d1s16, d2s16, d3s16, d4s16, d5s16, d6s16, d7s16;
  int16x4_t d16s16, d17s16, d26s16, d27s16, d28s16, d29s16;
  uint16x4_t d28u16, d29u16;
  uint16x8_t q14u16;
  int16x8_t q0s16, q1s16, q2s16, q3s16;
  int16x8_t q11s16, q12s16, q13s16, q14s16, q15s16, qEmptys16;
  int32x4_t q9s32, q10s32, q11s32, q12s32;
  int16x8x2_t v2tmp0, v2tmp1;
  int32x4x2_t v2tmp2, v2tmp3;

  d16s16 = vdup_n_s16(5352);
  d17s16 = vdup_n_s16(2217);
  q9s32 = vdupq_n_s32(14500);
  q10s32 = vdupq_n_s32(7500);

  // Part one
  pitch >>= 1;
  q0s16 = vld1q_s16(input);
  input += pitch;
  q1s16 = vld1q_s16(input);
  input += pitch;
  q2s16 = vld1q_s16(input);
  input += pitch;
  q3s16 = vld1q_s16(input);

  v2tmp2 =
      vtrnq_s32(vreinterpretq_s32_s16(q0s16), vreinterpretq_s32_s16(q2s16));
  v2tmp3 =
      vtrnq_s32(vreinterpretq_s32_s16(q1s16), vreinterpretq_s32_s16(q3s16));
  v2tmp0 = vtrnq_s16(vreinterpretq_s16_s32(v2tmp2.val[0]),   // q0
                     vreinterpretq_s16_s32(v2tmp3.val[0]));  // q1
  v2tmp1 = vtrnq_s16(vreinterpretq_s16_s32(v2tmp2.val[1]),   // q2
                     vreinterpretq_s16_s32(v2tmp3.val[1]));  // q3

  q11s16 = vaddq_s16(v2tmp0.val[0], v2tmp1.val[1]);
  q12s16 = vaddq_s16(v2tmp0.val[1], v2tmp1.val[0]);
  q13s16 = vsubq_s16(v2tmp0.val[1], v2tmp1.val[0]);
  q14s16 = vsubq_s16(v2tmp0.val[0], v2tmp1.val[1]);

  q11s16 = vshlq_n_s16(q11s16, 3);
  q12s16 = vshlq_n_s16(q12s16, 3);
  q13s16 = vshlq_n_s16(q13s16, 3);
  q14s16 = vshlq_n_s16(q14s16, 3);

  q0s16 = vaddq_s16(q11s16, q12s16);
  q2s16 = vsubq_s16(q11s16, q12s16);

  q11s32 = q9s32;
  q12s32 = q10s32;

  d26s16 = vget_low_s16(q13s16);
  d27s16 = vget_high_s16(q13s16);
  d28s16 = vget_low_s16(q14s16);
  d29s16 = vget_high_s16(q14s16);

  q9s32 = vmlal_s16(q9s32, d28s16, d16s16);
  q10s32 = vmlal_s16(q10s32, d28s16, d17s16);
  q11s32 = vmlal_s16(q11s32, d29s16, d16s16);
  q12s32 = vmlal_s16(q12s32, d29s16, d17s16);

  q9s32 = vmlal_s16(q9s32, d26s16, d17s16);
  q10s32 = vmlsl_s16(q10s32, d26s16, d16s16);
  q11s32 = vmlal_s16(q11s32, d27s16, d17s16);
  q12s32 = vmlsl_s16(q12s32, d27s16, d16s16);

  d2s16 = vshrn_n_s32(q9s32, 12);
  d6s16 = vshrn_n_s32(q10s32, 12);
  d3s16 = vshrn_n_s32(q11s32, 12);
  d7s16 = vshrn_n_s32(q12s32, 12);
  q1s16 = vcombine_s16(d2s16, d3s16);
  q3s16 = vcombine_s16(d6s16, d7s16);

  // Part two
  q9s32 = vdupq_n_s32(12000);
  q10s32 = vdupq_n_s32(51000);

  v2tmp2 =
      vtrnq_s32(vreinterpretq_s32_s16(q0s16), vreinterpretq_s32_s16(q2s16));
  v2tmp3 =
      vtrnq_s32(vreinterpretq_s32_s16(q1s16), vreinterpretq_s32_s16(q3s16));
  v2tmp0 = vtrnq_s16(vreinterpretq_s16_s32(v2tmp2.val[0]),   // q0
                     vreinterpretq_s16_s32(v2tmp3.val[0]));  // q1
  v2tmp1 = vtrnq_s16(vreinterpretq_s16_s32(v2tmp2.val[1]),   // q2
                     vreinterpretq_s16_s32(v2tmp3.val[1]));  // q3

  q11s16 = vaddq_s16(v2tmp0.val[0], v2tmp1.val[1]);
  q12s16 = vaddq_s16(v2tmp0.val[1], v2tmp1.val[0]);
  q13s16 = vsubq_s16(v2tmp0.val[1], v2tmp1.val[0]);
  q14s16 = vsubq_s16(v2tmp0.val[0], v2tmp1.val[1]);

  q15s16 = vdupq_n_s16(7);
  q11s16 = vaddq_s16(q11s16, q15s16);
  q0s16 = vaddq_s16(q11s16, q12s16);
  q1s16 = vsubq_s16(q11s16, q12s16);

  q11s32 = q9s32;
  q12s32 = q10s32;

  d0s16 = vget_low_s16(q0s16);
  d1s16 = vget_high_s16(q0s16);
  d2s16 = vget_low_s16(q1s16);
  d3s16 = vget_high_s16(q1s16);

  d0s16 = vshr_n_s16(d0s16, 4);
  d4s16 = vshr_n_s16(d1s16, 4);
  d2s16 = vshr_n_s16(d2s16, 4);
  d6s16 = vshr_n_s16(d3s16, 4);

  d26s16 = vget_low_s16(q13s16);
  d27s16 = vget_high_s16(q13s16);
  d28s16 = vget_low_s16(q14s16);
  d29s16 = vget_high_s16(q14s16);

  q9s32 = vmlal_s16(q9s32, d28s16, d16s16);
  q10s32 = vmlal_s16(q10s32, d28s16, d17s16);
  q11s32 = vmlal_s16(q11s32, d29s16, d16s16);
  q12s32 = vmlal_s16(q12s32, d29s16, d17s16);

  q9s32 = vmlal_s16(q9s32, d26s16, d17s16);
  q10s32 = vmlsl_s16(q10s32, d26s16, d16s16);
  q11s32 = vmlal_s16(q11s32, d27s16, d17s16);
  q12s32 = vmlsl_s16(q12s32, d27s16, d16s16);

  d1s16 = vshrn_n_s32(q9s32, 16);
  d3s16 = vshrn_n_s32(q10s32, 16);
  d5s16 = vshrn_n_s32(q11s32, 16);
  d7s16 = vshrn_n_s32(q12s32, 16);

  qEmptys16 = vdupq_n_s16(0);
  q14u16 = vceqq_s16(q14s16, qEmptys16);
  q14u16 = vmvnq_u16(q14u16);

  d28u16 = vget_low_u16(q14u16);
  d29u16 = vget_high_u16(q14u16);
  d1s16 = vsub_s16(d1s16, vreinterpret_s16_u16(d28u16));
  d5s16 = vsub_s16(d5s16, vreinterpret_s16_u16(d29u16));

  q0s16 = vcombine_s16(d0s16, d1s16);
  q1s16 = vcombine_s16(d2s16, d3s16);
  q2s16 = vcombine_s16(d4s16, d5s16);
  q3s16 = vcombine_s16(d6s16, d7s16);

  vst1q_s16(output, q0s16);
  vst1q_s16(output + 8, q1s16);
  vst1q_s16(output + 16, q2s16);
  vst1q_s16(output + 24, q3s16);
  return;
}
