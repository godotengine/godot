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

static const int16_t cospi8sqrt2minus1 = 20091;
// 35468 exceeds INT16_MAX and gets converted to a negative number. Because of
// the way it is used in vqdmulh, where the result is doubled, it can be divided
// by 2 beforehand. This saves compensating for the negative value as well as
// shifting the result.
static const int16_t sinpi8sqrt2 = 35468 >> 1;

void vp8_dequant_idct_add_neon(int16_t *input, int16_t *dq, unsigned char *dst,
                               int stride) {
  unsigned char *dst0;
  int32x2_t d14, d15;
  int16x4_t d2, d3, d4, d5, d10, d11, d12, d13;
  int16x8_t q1, q2, q3, q4, q5, q6;
  int16x8_t qEmpty = vdupq_n_s16(0);
  int32x2x2_t d2tmp0, d2tmp1;
  int16x4x2_t d2tmp2, d2tmp3;

  d14 = d15 = vdup_n_s32(0);

  // load input
  q3 = vld1q_s16(input);
  vst1q_s16(input, qEmpty);
  input += 8;
  q4 = vld1q_s16(input);
  vst1q_s16(input, qEmpty);

  // load dq
  q5 = vld1q_s16(dq);
  dq += 8;
  q6 = vld1q_s16(dq);

  // load src from dst
  dst0 = dst;
  d14 = vld1_lane_s32((const int32_t *)dst0, d14, 0);
  dst0 += stride;
  d14 = vld1_lane_s32((const int32_t *)dst0, d14, 1);
  dst0 += stride;
  d15 = vld1_lane_s32((const int32_t *)dst0, d15, 0);
  dst0 += stride;
  d15 = vld1_lane_s32((const int32_t *)dst0, d15, 1);

  q1 = vreinterpretq_s16_u16(
      vmulq_u16(vreinterpretq_u16_s16(q3), vreinterpretq_u16_s16(q5)));
  q2 = vreinterpretq_s16_u16(
      vmulq_u16(vreinterpretq_u16_s16(q4), vreinterpretq_u16_s16(q6)));

  d12 = vqadd_s16(vget_low_s16(q1), vget_low_s16(q2));
  d13 = vqsub_s16(vget_low_s16(q1), vget_low_s16(q2));

  q2 = vcombine_s16(vget_high_s16(q1), vget_high_s16(q2));

  q3 = vqdmulhq_n_s16(q2, sinpi8sqrt2);
  q4 = vqdmulhq_n_s16(q2, cospi8sqrt2minus1);

  q4 = vshrq_n_s16(q4, 1);

  q4 = vqaddq_s16(q4, q2);

  d10 = vqsub_s16(vget_low_s16(q3), vget_high_s16(q4));
  d11 = vqadd_s16(vget_high_s16(q3), vget_low_s16(q4));

  d2 = vqadd_s16(d12, d11);
  d3 = vqadd_s16(d13, d10);
  d4 = vqsub_s16(d13, d10);
  d5 = vqsub_s16(d12, d11);

  d2tmp0 = vtrn_s32(vreinterpret_s32_s16(d2), vreinterpret_s32_s16(d4));
  d2tmp1 = vtrn_s32(vreinterpret_s32_s16(d3), vreinterpret_s32_s16(d5));
  d2tmp2 = vtrn_s16(vreinterpret_s16_s32(d2tmp0.val[0]),
                    vreinterpret_s16_s32(d2tmp1.val[0]));
  d2tmp3 = vtrn_s16(vreinterpret_s16_s32(d2tmp0.val[1]),
                    vreinterpret_s16_s32(d2tmp1.val[1]));

  // loop 2
  q2 = vcombine_s16(d2tmp2.val[1], d2tmp3.val[1]);

  q3 = vqdmulhq_n_s16(q2, sinpi8sqrt2);
  q4 = vqdmulhq_n_s16(q2, cospi8sqrt2minus1);

  d12 = vqadd_s16(d2tmp2.val[0], d2tmp3.val[0]);
  d13 = vqsub_s16(d2tmp2.val[0], d2tmp3.val[0]);

  q4 = vshrq_n_s16(q4, 1);

  q4 = vqaddq_s16(q4, q2);

  d10 = vqsub_s16(vget_low_s16(q3), vget_high_s16(q4));
  d11 = vqadd_s16(vget_high_s16(q3), vget_low_s16(q4));

  d2 = vqadd_s16(d12, d11);
  d3 = vqadd_s16(d13, d10);
  d4 = vqsub_s16(d13, d10);
  d5 = vqsub_s16(d12, d11);

  d2 = vrshr_n_s16(d2, 3);
  d3 = vrshr_n_s16(d3, 3);
  d4 = vrshr_n_s16(d4, 3);
  d5 = vrshr_n_s16(d5, 3);

  d2tmp0 = vtrn_s32(vreinterpret_s32_s16(d2), vreinterpret_s32_s16(d4));
  d2tmp1 = vtrn_s32(vreinterpret_s32_s16(d3), vreinterpret_s32_s16(d5));
  d2tmp2 = vtrn_s16(vreinterpret_s16_s32(d2tmp0.val[0]),
                    vreinterpret_s16_s32(d2tmp1.val[0]));
  d2tmp3 = vtrn_s16(vreinterpret_s16_s32(d2tmp0.val[1]),
                    vreinterpret_s16_s32(d2tmp1.val[1]));

  q1 = vcombine_s16(d2tmp2.val[0], d2tmp2.val[1]);
  q2 = vcombine_s16(d2tmp3.val[0], d2tmp3.val[1]);

  q1 = vreinterpretq_s16_u16(
      vaddw_u8(vreinterpretq_u16_s16(q1), vreinterpret_u8_s32(d14)));
  q2 = vreinterpretq_s16_u16(
      vaddw_u8(vreinterpretq_u16_s16(q2), vreinterpret_u8_s32(d15)));

  d14 = vreinterpret_s32_u8(vqmovun_s16(q1));
  d15 = vreinterpret_s32_u8(vqmovun_s16(q2));

  dst0 = dst;
  vst1_lane_s32((int32_t *)dst0, d14, 0);
  dst0 += stride;
  vst1_lane_s32((int32_t *)dst0, d14, 1);
  dst0 += stride;
  vst1_lane_s32((int32_t *)dst0, d15, 0);
  dst0 += stride;
  vst1_lane_s32((int32_t *)dst0, d15, 1);
  return;
}
