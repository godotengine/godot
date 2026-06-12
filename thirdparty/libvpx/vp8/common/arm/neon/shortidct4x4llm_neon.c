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

void vp8_short_idct4x4llm_neon(int16_t *input, unsigned char *pred_ptr,
                               int pred_stride, unsigned char *dst_ptr,
                               int dst_stride) {
  int i;
  uint32x2_t d6u32 = vdup_n_u32(0);
  uint8x8_t d1u8;
  int16x4_t d2, d3, d4, d5, d10, d11, d12, d13;
  uint16x8_t q1u16;
  int16x8_t q1s16, q2s16, q3s16, q4s16;
  int32x2x2_t v2tmp0, v2tmp1;
  int16x4x2_t v2tmp2, v2tmp3;

  d2 = vld1_s16(input);
  d3 = vld1_s16(input + 4);
  d4 = vld1_s16(input + 8);
  d5 = vld1_s16(input + 12);

  // 1st for loop
  q1s16 = vcombine_s16(d2, d4);  // Swap d3 d4 here
  q2s16 = vcombine_s16(d3, d5);

  q3s16 = vqdmulhq_n_s16(q2s16, sinpi8sqrt2);
  q4s16 = vqdmulhq_n_s16(q2s16, cospi8sqrt2minus1);

  d12 = vqadd_s16(vget_low_s16(q1s16), vget_high_s16(q1s16));  // a1
  d13 = vqsub_s16(vget_low_s16(q1s16), vget_high_s16(q1s16));  // b1

  q4s16 = vshrq_n_s16(q4s16, 1);

  q4s16 = vqaddq_s16(q4s16, q2s16);

  d10 = vqsub_s16(vget_low_s16(q3s16), vget_high_s16(q4s16));  // c1
  d11 = vqadd_s16(vget_high_s16(q3s16), vget_low_s16(q4s16));  // d1

  d2 = vqadd_s16(d12, d11);
  d3 = vqadd_s16(d13, d10);
  d4 = vqsub_s16(d13, d10);
  d5 = vqsub_s16(d12, d11);

  v2tmp0 = vtrn_s32(vreinterpret_s32_s16(d2), vreinterpret_s32_s16(d4));
  v2tmp1 = vtrn_s32(vreinterpret_s32_s16(d3), vreinterpret_s32_s16(d5));
  v2tmp2 = vtrn_s16(vreinterpret_s16_s32(v2tmp0.val[0]),
                    vreinterpret_s16_s32(v2tmp1.val[0]));
  v2tmp3 = vtrn_s16(vreinterpret_s16_s32(v2tmp0.val[1]),
                    vreinterpret_s16_s32(v2tmp1.val[1]));

  // 2nd for loop
  q1s16 = vcombine_s16(v2tmp2.val[0], v2tmp3.val[0]);
  q2s16 = vcombine_s16(v2tmp2.val[1], v2tmp3.val[1]);

  q3s16 = vqdmulhq_n_s16(q2s16, sinpi8sqrt2);
  q4s16 = vqdmulhq_n_s16(q2s16, cospi8sqrt2minus1);

  d12 = vqadd_s16(vget_low_s16(q1s16), vget_high_s16(q1s16));  // a1
  d13 = vqsub_s16(vget_low_s16(q1s16), vget_high_s16(q1s16));  // b1

  q4s16 = vshrq_n_s16(q4s16, 1);

  q4s16 = vqaddq_s16(q4s16, q2s16);

  d10 = vqsub_s16(vget_low_s16(q3s16), vget_high_s16(q4s16));  // c1
  d11 = vqadd_s16(vget_high_s16(q3s16), vget_low_s16(q4s16));  // d1

  d2 = vqadd_s16(d12, d11);
  d3 = vqadd_s16(d13, d10);
  d4 = vqsub_s16(d13, d10);
  d5 = vqsub_s16(d12, d11);

  d2 = vrshr_n_s16(d2, 3);
  d3 = vrshr_n_s16(d3, 3);
  d4 = vrshr_n_s16(d4, 3);
  d5 = vrshr_n_s16(d5, 3);

  v2tmp0 = vtrn_s32(vreinterpret_s32_s16(d2), vreinterpret_s32_s16(d4));
  v2tmp1 = vtrn_s32(vreinterpret_s32_s16(d3), vreinterpret_s32_s16(d5));
  v2tmp2 = vtrn_s16(vreinterpret_s16_s32(v2tmp0.val[0]),
                    vreinterpret_s16_s32(v2tmp1.val[0]));
  v2tmp3 = vtrn_s16(vreinterpret_s16_s32(v2tmp0.val[1]),
                    vreinterpret_s16_s32(v2tmp1.val[1]));

  q1s16 = vcombine_s16(v2tmp2.val[0], v2tmp2.val[1]);
  q2s16 = vcombine_s16(v2tmp3.val[0], v2tmp3.val[1]);

  // dc_only_idct_add
  for (i = 0; i < 2; i++, q1s16 = q2s16) {
    d6u32 = vld1_lane_u32((const uint32_t *)pred_ptr, d6u32, 0);
    pred_ptr += pred_stride;
    d6u32 = vld1_lane_u32((const uint32_t *)pred_ptr, d6u32, 1);
    pred_ptr += pred_stride;

    q1u16 = vaddw_u8(vreinterpretq_u16_s16(q1s16), vreinterpret_u8_u32(d6u32));
    d1u8 = vqmovun_s16(vreinterpretq_s16_u16(q1u16));

    vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d1u8), 0);
    dst_ptr += dst_stride;
    vst1_lane_u32((uint32_t *)dst_ptr, vreinterpret_u32_u8(d1u8), 1);
    dst_ptr += dst_stride;
  }
  return;
}
