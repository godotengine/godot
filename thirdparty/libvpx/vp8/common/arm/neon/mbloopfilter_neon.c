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

#include "./vpx_config.h"
#include "vp8/common/arm/loopfilter_arm.h"

static INLINE void vp8_mbloop_filter_neon(uint8x16_t qblimit,  // mblimit
                                          uint8x16_t qlimit,   // limit
                                          uint8x16_t qthresh,  // thresh
                                          uint8x16_t q3,       // p2
                                          uint8x16_t q4,       // p2
                                          uint8x16_t q5,       // p1
                                          uint8x16_t q6,       // p0
                                          uint8x16_t q7,       // q0
                                          uint8x16_t q8,       // q1
                                          uint8x16_t q9,       // q2
                                          uint8x16_t q10,      // q3
                                          uint8x16_t *q4r,     // p1
                                          uint8x16_t *q5r,     // p1
                                          uint8x16_t *q6r,     // p0
                                          uint8x16_t *q7r,     // q0
                                          uint8x16_t *q8r,     // q1
                                          uint8x16_t *q9r) {   // q1
  uint8x16_t q0u8, q1u8, q11u8, q12u8, q13u8, q14u8, q15u8;
  int16x8_t q0s16, q2s16, q11s16, q12s16, q13s16, q14s16, q15s16;
  int8x16_t q1s8, q6s8, q7s8, q2s8, q11s8, q13s8;
  uint16x8_t q0u16, q11u16, q12u16, q13u16, q14u16, q15u16;
  int8x16_t q0s8, q12s8, q14s8, q15s8;
  int8x8_t d0, d1, d2, d3, d4, d5, d24, d25, d28, d29;

  q11u8 = vabdq_u8(q3, q4);
  q12u8 = vabdq_u8(q4, q5);
  q13u8 = vabdq_u8(q5, q6);
  q14u8 = vabdq_u8(q8, q7);
  q1u8 = vabdq_u8(q9, q8);
  q0u8 = vabdq_u8(q10, q9);

  q11u8 = vmaxq_u8(q11u8, q12u8);
  q12u8 = vmaxq_u8(q13u8, q14u8);
  q1u8 = vmaxq_u8(q1u8, q0u8);
  q15u8 = vmaxq_u8(q11u8, q12u8);

  q12u8 = vabdq_u8(q6, q7);

  // vp8_hevmask
  q13u8 = vcgtq_u8(q13u8, qthresh);
  q14u8 = vcgtq_u8(q14u8, qthresh);
  q15u8 = vmaxq_u8(q15u8, q1u8);

  q15u8 = vcgeq_u8(qlimit, q15u8);

  q1u8 = vabdq_u8(q5, q8);
  q12u8 = vqaddq_u8(q12u8, q12u8);

  // vp8_filter() function
  // convert to signed
  q0u8 = vdupq_n_u8(0x80);
  q9 = veorq_u8(q9, q0u8);
  q8 = veorq_u8(q8, q0u8);
  q7 = veorq_u8(q7, q0u8);
  q6 = veorq_u8(q6, q0u8);
  q5 = veorq_u8(q5, q0u8);
  q4 = veorq_u8(q4, q0u8);

  q1u8 = vshrq_n_u8(q1u8, 1);
  q12u8 = vqaddq_u8(q12u8, q1u8);

  q14u8 = vorrq_u8(q13u8, q14u8);
  q12u8 = vcgeq_u8(qblimit, q12u8);

  q2s16 = vsubl_s8(vget_low_s8(vreinterpretq_s8_u8(q7)),
                   vget_low_s8(vreinterpretq_s8_u8(q6)));
  q13s16 = vsubl_s8(vget_high_s8(vreinterpretq_s8_u8(q7)),
                    vget_high_s8(vreinterpretq_s8_u8(q6)));

  q1s8 = vqsubq_s8(vreinterpretq_s8_u8(q5), vreinterpretq_s8_u8(q8));

  q11s16 = vdupq_n_s16(3);
  q2s16 = vmulq_s16(q2s16, q11s16);
  q13s16 = vmulq_s16(q13s16, q11s16);

  q15u8 = vandq_u8(q15u8, q12u8);

  q2s16 = vaddw_s8(q2s16, vget_low_s8(q1s8));
  q13s16 = vaddw_s8(q13s16, vget_high_s8(q1s8));

  q12u8 = vdupq_n_u8(3);
  q11u8 = vdupq_n_u8(4);
  // vp8_filter = clamp(vp8_filter + 3 * ( qs0 - ps0))
  d2 = vqmovn_s16(q2s16);
  d3 = vqmovn_s16(q13s16);
  q1s8 = vcombine_s8(d2, d3);
  q1s8 = vandq_s8(q1s8, vreinterpretq_s8_u8(q15u8));
  q13s8 = vandq_s8(q1s8, vreinterpretq_s8_u8(q14u8));

  q2s8 = vqaddq_s8(q13s8, vreinterpretq_s8_u8(q11u8));
  q13s8 = vqaddq_s8(q13s8, vreinterpretq_s8_u8(q12u8));
  q2s8 = vshrq_n_s8(q2s8, 3);
  q13s8 = vshrq_n_s8(q13s8, 3);

  q7s8 = vqsubq_s8(vreinterpretq_s8_u8(q7), q2s8);
  q6s8 = vqaddq_s8(vreinterpretq_s8_u8(q6), q13s8);

  q1s8 = vbicq_s8(q1s8, vreinterpretq_s8_u8(q14u8));

  q0u16 = q11u16 = q12u16 = q13u16 = q14u16 = q15u16 = vdupq_n_u16(63);
  d5 = vdup_n_s8(9);
  d4 = vdup_n_s8(18);

  q0s16 = vmlal_s8(vreinterpretq_s16_u16(q0u16), vget_low_s8(q1s8), d5);
  q11s16 = vmlal_s8(vreinterpretq_s16_u16(q11u16), vget_high_s8(q1s8), d5);
  d5 = vdup_n_s8(27);
  q12s16 = vmlal_s8(vreinterpretq_s16_u16(q12u16), vget_low_s8(q1s8), d4);
  q13s16 = vmlal_s8(vreinterpretq_s16_u16(q13u16), vget_high_s8(q1s8), d4);
  q14s16 = vmlal_s8(vreinterpretq_s16_u16(q14u16), vget_low_s8(q1s8), d5);
  q15s16 = vmlal_s8(vreinterpretq_s16_u16(q15u16), vget_high_s8(q1s8), d5);

  d0 = vqshrn_n_s16(q0s16, 7);
  d1 = vqshrn_n_s16(q11s16, 7);
  d24 = vqshrn_n_s16(q12s16, 7);
  d25 = vqshrn_n_s16(q13s16, 7);
  d28 = vqshrn_n_s16(q14s16, 7);
  d29 = vqshrn_n_s16(q15s16, 7);

  q0s8 = vcombine_s8(d0, d1);
  q12s8 = vcombine_s8(d24, d25);
  q14s8 = vcombine_s8(d28, d29);

  q11s8 = vqsubq_s8(vreinterpretq_s8_u8(q9), q0s8);
  q0s8 = vqaddq_s8(vreinterpretq_s8_u8(q4), q0s8);
  q13s8 = vqsubq_s8(vreinterpretq_s8_u8(q8), q12s8);
  q12s8 = vqaddq_s8(vreinterpretq_s8_u8(q5), q12s8);
  q15s8 = vqsubq_s8((q7s8), q14s8);
  q14s8 = vqaddq_s8((q6s8), q14s8);

  q1u8 = vdupq_n_u8(0x80);
  *q9r = veorq_u8(vreinterpretq_u8_s8(q11s8), q1u8);
  *q8r = veorq_u8(vreinterpretq_u8_s8(q13s8), q1u8);
  *q7r = veorq_u8(vreinterpretq_u8_s8(q15s8), q1u8);
  *q6r = veorq_u8(vreinterpretq_u8_s8(q14s8), q1u8);
  *q5r = veorq_u8(vreinterpretq_u8_s8(q12s8), q1u8);
  *q4r = veorq_u8(vreinterpretq_u8_s8(q0s8), q1u8);
  return;
}

void vp8_mbloop_filter_horizontal_edge_y_neon(unsigned char *src, int pitch,
                                              unsigned char blimit,
                                              unsigned char limit,
                                              unsigned char thresh) {
  uint8x16_t qblimit, qlimit, qthresh, q3, q4;
  uint8x16_t q5, q6, q7, q8, q9, q10;

  qblimit = vdupq_n_u8(blimit);
  qlimit = vdupq_n_u8(limit);
  qthresh = vdupq_n_u8(thresh);

  src -= (pitch << 2);

  q3 = vld1q_u8(src);
  src += pitch;
  q4 = vld1q_u8(src);
  src += pitch;
  q5 = vld1q_u8(src);
  src += pitch;
  q6 = vld1q_u8(src);
  src += pitch;
  q7 = vld1q_u8(src);
  src += pitch;
  q8 = vld1q_u8(src);
  src += pitch;
  q9 = vld1q_u8(src);
  src += pitch;
  q10 = vld1q_u8(src);

  vp8_mbloop_filter_neon(qblimit, qlimit, qthresh, q3, q4, q5, q6, q7, q8, q9,
                         q10, &q4, &q5, &q6, &q7, &q8, &q9);

  src -= (pitch * 6);
  vst1q_u8(src, q4);
  src += pitch;
  vst1q_u8(src, q5);
  src += pitch;
  vst1q_u8(src, q6);
  src += pitch;
  vst1q_u8(src, q7);
  src += pitch;
  vst1q_u8(src, q8);
  src += pitch;
  vst1q_u8(src, q9);
  return;
}

void vp8_mbloop_filter_horizontal_edge_uv_neon(unsigned char *u, int pitch,
                                               unsigned char blimit,
                                               unsigned char limit,
                                               unsigned char thresh,
                                               unsigned char *v) {
  uint8x16_t qblimit, qlimit, qthresh, q3, q4;
  uint8x16_t q5, q6, q7, q8, q9, q10;
  uint8x8_t d6, d7, d8, d9, d10, d11, d12, d13, d14;
  uint8x8_t d15, d16, d17, d18, d19, d20, d21;

  qblimit = vdupq_n_u8(blimit);
  qlimit = vdupq_n_u8(limit);
  qthresh = vdupq_n_u8(thresh);

  u -= (pitch << 2);
  v -= (pitch << 2);

  d6 = vld1_u8(u);
  u += pitch;
  d7 = vld1_u8(v);
  v += pitch;
  d8 = vld1_u8(u);
  u += pitch;
  d9 = vld1_u8(v);
  v += pitch;
  d10 = vld1_u8(u);
  u += pitch;
  d11 = vld1_u8(v);
  v += pitch;
  d12 = vld1_u8(u);
  u += pitch;
  d13 = vld1_u8(v);
  v += pitch;
  d14 = vld1_u8(u);
  u += pitch;
  d15 = vld1_u8(v);
  v += pitch;
  d16 = vld1_u8(u);
  u += pitch;
  d17 = vld1_u8(v);
  v += pitch;
  d18 = vld1_u8(u);
  u += pitch;
  d19 = vld1_u8(v);
  v += pitch;
  d20 = vld1_u8(u);
  d21 = vld1_u8(v);

  q3 = vcombine_u8(d6, d7);
  q4 = vcombine_u8(d8, d9);
  q5 = vcombine_u8(d10, d11);
  q6 = vcombine_u8(d12, d13);
  q7 = vcombine_u8(d14, d15);
  q8 = vcombine_u8(d16, d17);
  q9 = vcombine_u8(d18, d19);
  q10 = vcombine_u8(d20, d21);

  vp8_mbloop_filter_neon(qblimit, qlimit, qthresh, q3, q4, q5, q6, q7, q8, q9,
                         q10, &q4, &q5, &q6, &q7, &q8, &q9);

  u -= (pitch * 6);
  v -= (pitch * 6);
  vst1_u8(u, vget_low_u8(q4));
  u += pitch;
  vst1_u8(v, vget_high_u8(q4));
  v += pitch;
  vst1_u8(u, vget_low_u8(q5));
  u += pitch;
  vst1_u8(v, vget_high_u8(q5));
  v += pitch;
  vst1_u8(u, vget_low_u8(q6));
  u += pitch;
  vst1_u8(v, vget_high_u8(q6));
  v += pitch;
  vst1_u8(u, vget_low_u8(q7));
  u += pitch;
  vst1_u8(v, vget_high_u8(q7));
  v += pitch;
  vst1_u8(u, vget_low_u8(q8));
  u += pitch;
  vst1_u8(v, vget_high_u8(q8));
  v += pitch;
  vst1_u8(u, vget_low_u8(q9));
  vst1_u8(v, vget_high_u8(q9));
  return;
}

void vp8_mbloop_filter_vertical_edge_y_neon(unsigned char *src, int pitch,
                                            unsigned char blimit,
                                            unsigned char limit,
                                            unsigned char thresh) {
  unsigned char *s1, *s2;
  uint8x16_t qblimit, qlimit, qthresh, q3, q4;
  uint8x16_t q5, q6, q7, q8, q9, q10;
  uint8x8_t d6, d7, d8, d9, d10, d11, d12, d13, d14;
  uint8x8_t d15, d16, d17, d18, d19, d20, d21;
  uint32x4x2_t q2tmp0, q2tmp1, q2tmp2, q2tmp3;
  uint16x8x2_t q2tmp4, q2tmp5, q2tmp6, q2tmp7;
  uint8x16x2_t q2tmp8, q2tmp9, q2tmp10, q2tmp11;

  qblimit = vdupq_n_u8(blimit);
  qlimit = vdupq_n_u8(limit);
  qthresh = vdupq_n_u8(thresh);

  s1 = src - 4;
  s2 = s1 + 8 * pitch;
  d6 = vld1_u8(s1);
  s1 += pitch;
  d7 = vld1_u8(s2);
  s2 += pitch;
  d8 = vld1_u8(s1);
  s1 += pitch;
  d9 = vld1_u8(s2);
  s2 += pitch;
  d10 = vld1_u8(s1);
  s1 += pitch;
  d11 = vld1_u8(s2);
  s2 += pitch;
  d12 = vld1_u8(s1);
  s1 += pitch;
  d13 = vld1_u8(s2);
  s2 += pitch;
  d14 = vld1_u8(s1);
  s1 += pitch;
  d15 = vld1_u8(s2);
  s2 += pitch;
  d16 = vld1_u8(s1);
  s1 += pitch;
  d17 = vld1_u8(s2);
  s2 += pitch;
  d18 = vld1_u8(s1);
  s1 += pitch;
  d19 = vld1_u8(s2);
  s2 += pitch;
  d20 = vld1_u8(s1);
  d21 = vld1_u8(s2);

  q3 = vcombine_u8(d6, d7);
  q4 = vcombine_u8(d8, d9);
  q5 = vcombine_u8(d10, d11);
  q6 = vcombine_u8(d12, d13);
  q7 = vcombine_u8(d14, d15);
  q8 = vcombine_u8(d16, d17);
  q9 = vcombine_u8(d18, d19);
  q10 = vcombine_u8(d20, d21);

  q2tmp0 = vtrnq_u32(vreinterpretq_u32_u8(q3), vreinterpretq_u32_u8(q7));
  q2tmp1 = vtrnq_u32(vreinterpretq_u32_u8(q4), vreinterpretq_u32_u8(q8));
  q2tmp2 = vtrnq_u32(vreinterpretq_u32_u8(q5), vreinterpretq_u32_u8(q9));
  q2tmp3 = vtrnq_u32(vreinterpretq_u32_u8(q6), vreinterpretq_u32_u8(q10));

  q2tmp4 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp0.val[0]),
                     vreinterpretq_u16_u32(q2tmp2.val[0]));
  q2tmp5 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp1.val[0]),
                     vreinterpretq_u16_u32(q2tmp3.val[0]));
  q2tmp6 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp0.val[1]),
                     vreinterpretq_u16_u32(q2tmp2.val[1]));
  q2tmp7 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp1.val[1]),
                     vreinterpretq_u16_u32(q2tmp3.val[1]));

  q2tmp8 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp4.val[0]),
                    vreinterpretq_u8_u16(q2tmp5.val[0]));
  q2tmp9 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp4.val[1]),
                    vreinterpretq_u8_u16(q2tmp5.val[1]));
  q2tmp10 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp6.val[0]),
                     vreinterpretq_u8_u16(q2tmp7.val[0]));
  q2tmp11 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp6.val[1]),
                     vreinterpretq_u8_u16(q2tmp7.val[1]));

  q3 = q2tmp8.val[0];
  q4 = q2tmp8.val[1];
  q5 = q2tmp9.val[0];
  q6 = q2tmp9.val[1];
  q7 = q2tmp10.val[0];
  q8 = q2tmp10.val[1];
  q9 = q2tmp11.val[0];
  q10 = q2tmp11.val[1];

  vp8_mbloop_filter_neon(qblimit, qlimit, qthresh, q3, q4, q5, q6, q7, q8, q9,
                         q10, &q4, &q5, &q6, &q7, &q8, &q9);

  q2tmp0 = vtrnq_u32(vreinterpretq_u32_u8(q3), vreinterpretq_u32_u8(q7));
  q2tmp1 = vtrnq_u32(vreinterpretq_u32_u8(q4), vreinterpretq_u32_u8(q8));
  q2tmp2 = vtrnq_u32(vreinterpretq_u32_u8(q5), vreinterpretq_u32_u8(q9));
  q2tmp3 = vtrnq_u32(vreinterpretq_u32_u8(q6), vreinterpretq_u32_u8(q10));

  q2tmp4 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp0.val[0]),
                     vreinterpretq_u16_u32(q2tmp2.val[0]));
  q2tmp5 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp1.val[0]),
                     vreinterpretq_u16_u32(q2tmp3.val[0]));
  q2tmp6 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp0.val[1]),
                     vreinterpretq_u16_u32(q2tmp2.val[1]));
  q2tmp7 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp1.val[1]),
                     vreinterpretq_u16_u32(q2tmp3.val[1]));

  q2tmp8 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp4.val[0]),
                    vreinterpretq_u8_u16(q2tmp5.val[0]));
  q2tmp9 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp4.val[1]),
                    vreinterpretq_u8_u16(q2tmp5.val[1]));
  q2tmp10 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp6.val[0]),
                     vreinterpretq_u8_u16(q2tmp7.val[0]));
  q2tmp11 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp6.val[1]),
                     vreinterpretq_u8_u16(q2tmp7.val[1]));

  q3 = q2tmp8.val[0];
  q4 = q2tmp8.val[1];
  q5 = q2tmp9.val[0];
  q6 = q2tmp9.val[1];
  q7 = q2tmp10.val[0];
  q8 = q2tmp10.val[1];
  q9 = q2tmp11.val[0];
  q10 = q2tmp11.val[1];

  s1 -= 7 * pitch;
  s2 -= 7 * pitch;

  vst1_u8(s1, vget_low_u8(q3));
  s1 += pitch;
  vst1_u8(s2, vget_high_u8(q3));
  s2 += pitch;
  vst1_u8(s1, vget_low_u8(q4));
  s1 += pitch;
  vst1_u8(s2, vget_high_u8(q4));
  s2 += pitch;
  vst1_u8(s1, vget_low_u8(q5));
  s1 += pitch;
  vst1_u8(s2, vget_high_u8(q5));
  s2 += pitch;
  vst1_u8(s1, vget_low_u8(q6));
  s1 += pitch;
  vst1_u8(s2, vget_high_u8(q6));
  s2 += pitch;
  vst1_u8(s1, vget_low_u8(q7));
  s1 += pitch;
  vst1_u8(s2, vget_high_u8(q7));
  s2 += pitch;
  vst1_u8(s1, vget_low_u8(q8));
  s1 += pitch;
  vst1_u8(s2, vget_high_u8(q8));
  s2 += pitch;
  vst1_u8(s1, vget_low_u8(q9));
  s1 += pitch;
  vst1_u8(s2, vget_high_u8(q9));
  s2 += pitch;
  vst1_u8(s1, vget_low_u8(q10));
  vst1_u8(s2, vget_high_u8(q10));
  return;
}

void vp8_mbloop_filter_vertical_edge_uv_neon(unsigned char *u, int pitch,
                                             unsigned char blimit,
                                             unsigned char limit,
                                             unsigned char thresh,
                                             unsigned char *v) {
  unsigned char *us, *ud;
  unsigned char *vs, *vd;
  uint8x16_t qblimit, qlimit, qthresh, q3, q4;
  uint8x16_t q5, q6, q7, q8, q9, q10;
  uint8x8_t d6, d7, d8, d9, d10, d11, d12, d13, d14;
  uint8x8_t d15, d16, d17, d18, d19, d20, d21;
  uint32x4x2_t q2tmp0, q2tmp1, q2tmp2, q2tmp3;
  uint16x8x2_t q2tmp4, q2tmp5, q2tmp6, q2tmp7;
  uint8x16x2_t q2tmp8, q2tmp9, q2tmp10, q2tmp11;

  qblimit = vdupq_n_u8(blimit);
  qlimit = vdupq_n_u8(limit);
  qthresh = vdupq_n_u8(thresh);

  us = u - 4;
  vs = v - 4;
  d6 = vld1_u8(us);
  us += pitch;
  d7 = vld1_u8(vs);
  vs += pitch;
  d8 = vld1_u8(us);
  us += pitch;
  d9 = vld1_u8(vs);
  vs += pitch;
  d10 = vld1_u8(us);
  us += pitch;
  d11 = vld1_u8(vs);
  vs += pitch;
  d12 = vld1_u8(us);
  us += pitch;
  d13 = vld1_u8(vs);
  vs += pitch;
  d14 = vld1_u8(us);
  us += pitch;
  d15 = vld1_u8(vs);
  vs += pitch;
  d16 = vld1_u8(us);
  us += pitch;
  d17 = vld1_u8(vs);
  vs += pitch;
  d18 = vld1_u8(us);
  us += pitch;
  d19 = vld1_u8(vs);
  vs += pitch;
  d20 = vld1_u8(us);
  d21 = vld1_u8(vs);

  q3 = vcombine_u8(d6, d7);
  q4 = vcombine_u8(d8, d9);
  q5 = vcombine_u8(d10, d11);
  q6 = vcombine_u8(d12, d13);
  q7 = vcombine_u8(d14, d15);
  q8 = vcombine_u8(d16, d17);
  q9 = vcombine_u8(d18, d19);
  q10 = vcombine_u8(d20, d21);

  q2tmp0 = vtrnq_u32(vreinterpretq_u32_u8(q3), vreinterpretq_u32_u8(q7));
  q2tmp1 = vtrnq_u32(vreinterpretq_u32_u8(q4), vreinterpretq_u32_u8(q8));
  q2tmp2 = vtrnq_u32(vreinterpretq_u32_u8(q5), vreinterpretq_u32_u8(q9));
  q2tmp3 = vtrnq_u32(vreinterpretq_u32_u8(q6), vreinterpretq_u32_u8(q10));

  q2tmp4 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp0.val[0]),
                     vreinterpretq_u16_u32(q2tmp2.val[0]));
  q2tmp5 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp1.val[0]),
                     vreinterpretq_u16_u32(q2tmp3.val[0]));
  q2tmp6 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp0.val[1]),
                     vreinterpretq_u16_u32(q2tmp2.val[1]));
  q2tmp7 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp1.val[1]),
                     vreinterpretq_u16_u32(q2tmp3.val[1]));

  q2tmp8 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp4.val[0]),
                    vreinterpretq_u8_u16(q2tmp5.val[0]));
  q2tmp9 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp4.val[1]),
                    vreinterpretq_u8_u16(q2tmp5.val[1]));
  q2tmp10 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp6.val[0]),
                     vreinterpretq_u8_u16(q2tmp7.val[0]));
  q2tmp11 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp6.val[1]),
                     vreinterpretq_u8_u16(q2tmp7.val[1]));

  q3 = q2tmp8.val[0];
  q4 = q2tmp8.val[1];
  q5 = q2tmp9.val[0];
  q6 = q2tmp9.val[1];
  q7 = q2tmp10.val[0];
  q8 = q2tmp10.val[1];
  q9 = q2tmp11.val[0];
  q10 = q2tmp11.val[1];

  vp8_mbloop_filter_neon(qblimit, qlimit, qthresh, q3, q4, q5, q6, q7, q8, q9,
                         q10, &q4, &q5, &q6, &q7, &q8, &q9);

  q2tmp0 = vtrnq_u32(vreinterpretq_u32_u8(q3), vreinterpretq_u32_u8(q7));
  q2tmp1 = vtrnq_u32(vreinterpretq_u32_u8(q4), vreinterpretq_u32_u8(q8));
  q2tmp2 = vtrnq_u32(vreinterpretq_u32_u8(q5), vreinterpretq_u32_u8(q9));
  q2tmp3 = vtrnq_u32(vreinterpretq_u32_u8(q6), vreinterpretq_u32_u8(q10));

  q2tmp4 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp0.val[0]),
                     vreinterpretq_u16_u32(q2tmp2.val[0]));
  q2tmp5 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp1.val[0]),
                     vreinterpretq_u16_u32(q2tmp3.val[0]));
  q2tmp6 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp0.val[1]),
                     vreinterpretq_u16_u32(q2tmp2.val[1]));
  q2tmp7 = vtrnq_u16(vreinterpretq_u16_u32(q2tmp1.val[1]),
                     vreinterpretq_u16_u32(q2tmp3.val[1]));

  q2tmp8 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp4.val[0]),
                    vreinterpretq_u8_u16(q2tmp5.val[0]));
  q2tmp9 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp4.val[1]),
                    vreinterpretq_u8_u16(q2tmp5.val[1]));
  q2tmp10 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp6.val[0]),
                     vreinterpretq_u8_u16(q2tmp7.val[0]));
  q2tmp11 = vtrnq_u8(vreinterpretq_u8_u16(q2tmp6.val[1]),
                     vreinterpretq_u8_u16(q2tmp7.val[1]));

  q3 = q2tmp8.val[0];
  q4 = q2tmp8.val[1];
  q5 = q2tmp9.val[0];
  q6 = q2tmp9.val[1];
  q7 = q2tmp10.val[0];
  q8 = q2tmp10.val[1];
  q9 = q2tmp11.val[0];
  q10 = q2tmp11.val[1];

  ud = u - 4;
  vst1_u8(ud, vget_low_u8(q3));
  ud += pitch;
  vst1_u8(ud, vget_low_u8(q4));
  ud += pitch;
  vst1_u8(ud, vget_low_u8(q5));
  ud += pitch;
  vst1_u8(ud, vget_low_u8(q6));
  ud += pitch;
  vst1_u8(ud, vget_low_u8(q7));
  ud += pitch;
  vst1_u8(ud, vget_low_u8(q8));
  ud += pitch;
  vst1_u8(ud, vget_low_u8(q9));
  ud += pitch;
  vst1_u8(ud, vget_low_u8(q10));

  vd = v - 4;
  vst1_u8(vd, vget_high_u8(q3));
  vd += pitch;
  vst1_u8(vd, vget_high_u8(q4));
  vd += pitch;
  vst1_u8(vd, vget_high_u8(q5));
  vd += pitch;
  vst1_u8(vd, vget_high_u8(q6));
  vd += pitch;
  vst1_u8(vd, vget_high_u8(q7));
  vd += pitch;
  vst1_u8(vd, vget_high_u8(q8));
  vd += pitch;
  vst1_u8(vd, vget_high_u8(q9));
  vd += pitch;
  vst1_u8(vd, vget_high_u8(q10));
  return;
}
