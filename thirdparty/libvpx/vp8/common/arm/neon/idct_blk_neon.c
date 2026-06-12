/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vp8_rtcd.h"

static void idct_dequant_0_2x_neon(int16_t *q, int16_t dq, unsigned char *dst,
                                   int stride) {
  unsigned char *dst0;
  int i, a0, a1;
  int16x8x2_t q2Add;
  int32x2_t d2s32 = vdup_n_s32(0), d4s32 = vdup_n_s32(0);
  uint8x8_t d2u8, d4u8;
  uint16x8_t q1u16, q2u16;

  a0 = ((q[0] * dq) + 4) >> 3;
  a1 = ((q[16] * dq) + 4) >> 3;
  q[0] = q[16] = 0;
  q2Add.val[0] = vdupq_n_s16((int16_t)a0);
  q2Add.val[1] = vdupq_n_s16((int16_t)a1);

  for (i = 0; i < 2; i++, dst += 4) {
    dst0 = dst;
    d2s32 = vld1_lane_s32((const int32_t *)dst0, d2s32, 0);
    dst0 += stride;
    d2s32 = vld1_lane_s32((const int32_t *)dst0, d2s32, 1);
    dst0 += stride;
    d4s32 = vld1_lane_s32((const int32_t *)dst0, d4s32, 0);
    dst0 += stride;
    d4s32 = vld1_lane_s32((const int32_t *)dst0, d4s32, 1);

    q1u16 = vaddw_u8(vreinterpretq_u16_s16(q2Add.val[i]),
                     vreinterpret_u8_s32(d2s32));
    q2u16 = vaddw_u8(vreinterpretq_u16_s16(q2Add.val[i]),
                     vreinterpret_u8_s32(d4s32));

    d2u8 = vqmovun_s16(vreinterpretq_s16_u16(q1u16));
    d4u8 = vqmovun_s16(vreinterpretq_s16_u16(q2u16));

    d2s32 = vreinterpret_s32_u8(d2u8);
    d4s32 = vreinterpret_s32_u8(d4u8);

    dst0 = dst;
    vst1_lane_s32((int32_t *)dst0, d2s32, 0);
    dst0 += stride;
    vst1_lane_s32((int32_t *)dst0, d2s32, 1);
    dst0 += stride;
    vst1_lane_s32((int32_t *)dst0, d4s32, 0);
    dst0 += stride;
    vst1_lane_s32((int32_t *)dst0, d4s32, 1);
  }
}

static const int16_t cospi8sqrt2minus1 = 20091;
static const int16_t sinpi8sqrt2 = 17734;
// because the lowest bit in 0x8a8c is 0, we can pre-shift this

static void idct_dequant_full_2x_neon(int16_t *q, int16_t *dq,
                                      unsigned char *dst, int stride) {
  unsigned char *dst0, *dst1;
  int32x2_t d28, d29, d30, d31;
  int16x8_t q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11;
  int16x8_t qEmpty = vdupq_n_s16(0);
  int32x4x2_t q2tmp0, q2tmp1;
  int16x8x2_t q2tmp2, q2tmp3;
  int16x4_t dLow0, dLow1, dHigh0, dHigh1;

  d28 = d29 = d30 = d31 = vdup_n_s32(0);

  // load dq
  q0 = vld1q_s16(dq);
  dq += 8;
  q1 = vld1q_s16(dq);

  // load q
  q2 = vld1q_s16(q);
  vst1q_s16(q, qEmpty);
  q += 8;
  q3 = vld1q_s16(q);
  vst1q_s16(q, qEmpty);
  q += 8;
  q4 = vld1q_s16(q);
  vst1q_s16(q, qEmpty);
  q += 8;
  q5 = vld1q_s16(q);
  vst1q_s16(q, qEmpty);

  // load src from dst
  dst0 = dst;
  dst1 = dst + 4;
  d28 = vld1_lane_s32((const int32_t *)dst0, d28, 0);
  dst0 += stride;
  d28 = vld1_lane_s32((const int32_t *)dst1, d28, 1);
  dst1 += stride;
  d29 = vld1_lane_s32((const int32_t *)dst0, d29, 0);
  dst0 += stride;
  d29 = vld1_lane_s32((const int32_t *)dst1, d29, 1);
  dst1 += stride;

  d30 = vld1_lane_s32((const int32_t *)dst0, d30, 0);
  dst0 += stride;
  d30 = vld1_lane_s32((const int32_t *)dst1, d30, 1);
  dst1 += stride;
  d31 = vld1_lane_s32((const int32_t *)dst0, d31, 0);
  d31 = vld1_lane_s32((const int32_t *)dst1, d31, 1);

  q2 = vmulq_s16(q2, q0);
  q3 = vmulq_s16(q3, q1);
  q4 = vmulq_s16(q4, q0);
  q5 = vmulq_s16(q5, q1);

  // vswp
  dLow0 = vget_low_s16(q2);
  dHigh0 = vget_high_s16(q2);
  dLow1 = vget_low_s16(q4);
  dHigh1 = vget_high_s16(q4);
  q2 = vcombine_s16(dLow0, dLow1);
  q4 = vcombine_s16(dHigh0, dHigh1);

  dLow0 = vget_low_s16(q3);
  dHigh0 = vget_high_s16(q3);
  dLow1 = vget_low_s16(q5);
  dHigh1 = vget_high_s16(q5);
  q3 = vcombine_s16(dLow0, dLow1);
  q5 = vcombine_s16(dHigh0, dHigh1);

  q6 = vqdmulhq_n_s16(q4, sinpi8sqrt2);
  q7 = vqdmulhq_n_s16(q5, sinpi8sqrt2);
  q8 = vqdmulhq_n_s16(q4, cospi8sqrt2minus1);
  q9 = vqdmulhq_n_s16(q5, cospi8sqrt2minus1);

  q10 = vqaddq_s16(q2, q3);
  q11 = vqsubq_s16(q2, q3);

  q8 = vshrq_n_s16(q8, 1);
  q9 = vshrq_n_s16(q9, 1);

  q4 = vqaddq_s16(q4, q8);
  q5 = vqaddq_s16(q5, q9);

  q2 = vqsubq_s16(q6, q5);
  q3 = vqaddq_s16(q7, q4);

  q4 = vqaddq_s16(q10, q3);
  q5 = vqaddq_s16(q11, q2);
  q6 = vqsubq_s16(q11, q2);
  q7 = vqsubq_s16(q10, q3);

  q2tmp0 = vtrnq_s32(vreinterpretq_s32_s16(q4), vreinterpretq_s32_s16(q6));
  q2tmp1 = vtrnq_s32(vreinterpretq_s32_s16(q5), vreinterpretq_s32_s16(q7));
  q2tmp2 = vtrnq_s16(vreinterpretq_s16_s32(q2tmp0.val[0]),
                     vreinterpretq_s16_s32(q2tmp1.val[0]));
  q2tmp3 = vtrnq_s16(vreinterpretq_s16_s32(q2tmp0.val[1]),
                     vreinterpretq_s16_s32(q2tmp1.val[1]));

  // loop 2
  q8 = vqdmulhq_n_s16(q2tmp2.val[1], sinpi8sqrt2);
  q9 = vqdmulhq_n_s16(q2tmp3.val[1], sinpi8sqrt2);
  q10 = vqdmulhq_n_s16(q2tmp2.val[1], cospi8sqrt2minus1);
  q11 = vqdmulhq_n_s16(q2tmp3.val[1], cospi8sqrt2minus1);

  q2 = vqaddq_s16(q2tmp2.val[0], q2tmp3.val[0]);
  q3 = vqsubq_s16(q2tmp2.val[0], q2tmp3.val[0]);

  q10 = vshrq_n_s16(q10, 1);
  q11 = vshrq_n_s16(q11, 1);

  q10 = vqaddq_s16(q2tmp2.val[1], q10);
  q11 = vqaddq_s16(q2tmp3.val[1], q11);

  q8 = vqsubq_s16(q8, q11);
  q9 = vqaddq_s16(q9, q10);

  q4 = vqaddq_s16(q2, q9);
  q5 = vqaddq_s16(q3, q8);
  q6 = vqsubq_s16(q3, q8);
  q7 = vqsubq_s16(q2, q9);

  q4 = vrshrq_n_s16(q4, 3);
  q5 = vrshrq_n_s16(q5, 3);
  q6 = vrshrq_n_s16(q6, 3);
  q7 = vrshrq_n_s16(q7, 3);

  q2tmp0 = vtrnq_s32(vreinterpretq_s32_s16(q4), vreinterpretq_s32_s16(q6));
  q2tmp1 = vtrnq_s32(vreinterpretq_s32_s16(q5), vreinterpretq_s32_s16(q7));
  q2tmp2 = vtrnq_s16(vreinterpretq_s16_s32(q2tmp0.val[0]),
                     vreinterpretq_s16_s32(q2tmp1.val[0]));
  q2tmp3 = vtrnq_s16(vreinterpretq_s16_s32(q2tmp0.val[1]),
                     vreinterpretq_s16_s32(q2tmp1.val[1]));

  q4 = vreinterpretq_s16_u16(
      vaddw_u8(vreinterpretq_u16_s16(q2tmp2.val[0]), vreinterpret_u8_s32(d28)));
  q5 = vreinterpretq_s16_u16(
      vaddw_u8(vreinterpretq_u16_s16(q2tmp2.val[1]), vreinterpret_u8_s32(d29)));
  q6 = vreinterpretq_s16_u16(
      vaddw_u8(vreinterpretq_u16_s16(q2tmp3.val[0]), vreinterpret_u8_s32(d30)));
  q7 = vreinterpretq_s16_u16(
      vaddw_u8(vreinterpretq_u16_s16(q2tmp3.val[1]), vreinterpret_u8_s32(d31)));

  d28 = vreinterpret_s32_u8(vqmovun_s16(q4));
  d29 = vreinterpret_s32_u8(vqmovun_s16(q5));
  d30 = vreinterpret_s32_u8(vqmovun_s16(q6));
  d31 = vreinterpret_s32_u8(vqmovun_s16(q7));

  dst0 = dst;
  dst1 = dst + 4;
  vst1_lane_s32((int32_t *)dst0, d28, 0);
  dst0 += stride;
  vst1_lane_s32((int32_t *)dst1, d28, 1);
  dst1 += stride;
  vst1_lane_s32((int32_t *)dst0, d29, 0);
  dst0 += stride;
  vst1_lane_s32((int32_t *)dst1, d29, 1);
  dst1 += stride;

  vst1_lane_s32((int32_t *)dst0, d30, 0);
  dst0 += stride;
  vst1_lane_s32((int32_t *)dst1, d30, 1);
  dst1 += stride;
  vst1_lane_s32((int32_t *)dst0, d31, 0);
  vst1_lane_s32((int32_t *)dst1, d31, 1);
}

void vp8_dequant_idct_add_y_block_neon(short *q, short *dq, unsigned char *dst,
                                       int stride, char *eobs) {
  int i;

  for (i = 0; i < 4; ++i) {
    if (((short *)(eobs))[0]) {
      if (((short *)eobs)[0] & 0xfefe)
        idct_dequant_full_2x_neon(q, dq, dst, stride);
      else
        idct_dequant_0_2x_neon(q, dq[0], dst, stride);
    }

    if (((short *)(eobs))[1]) {
      if (((short *)eobs)[1] & 0xfefe)
        idct_dequant_full_2x_neon(q + 32, dq, dst + 8, stride);
      else
        idct_dequant_0_2x_neon(q + 32, dq[0], dst + 8, stride);
    }
    q += 64;
    dst += 4 * stride;
    eobs += 4;
  }
}

void vp8_dequant_idct_add_uv_block_neon(short *q, short *dq,
                                        unsigned char *dst_u,
                                        unsigned char *dst_v, int stride,
                                        char *eobs) {
  if (((short *)(eobs))[0]) {
    if (((short *)eobs)[0] & 0xfefe)
      idct_dequant_full_2x_neon(q, dq, dst_u, stride);
    else
      idct_dequant_0_2x_neon(q, dq[0], dst_u, stride);
  }

  q += 32;
  dst_u += 4 * stride;

  if (((short *)(eobs))[1]) {
    if (((short *)eobs)[1] & 0xfefe)
      idct_dequant_full_2x_neon(q, dq, dst_u, stride);
    else
      idct_dequant_0_2x_neon(q, dq[0], dst_u, stride);
  }

  q += 32;

  if (((short *)(eobs))[2]) {
    if (((short *)eobs)[2] & 0xfefe)
      idct_dequant_full_2x_neon(q, dq, dst_v, stride);
    else
      idct_dequant_0_2x_neon(q, dq[0], dst_v, stride);
  }

  q += 32;
  dst_v += 4 * stride;

  if (((short *)(eobs))[3]) {
    if (((short *)eobs)[3] & 0xfefe)
      idct_dequant_full_2x_neon(q, dq, dst_v, stride);
    else
      idct_dequant_0_2x_neon(q, dq[0], dst_v, stride);
  }
}
