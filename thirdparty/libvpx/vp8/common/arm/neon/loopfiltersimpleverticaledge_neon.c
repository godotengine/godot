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
#include "./vp8_rtcd.h"
#include "vpx_ports/arm.h"

#ifdef VPX_INCOMPATIBLE_GCC
static INLINE void write_2x4(unsigned char *dst, int pitch,
                             const uint8x8x2_t result) {
  /*
   * uint8x8x2_t result
  00 01 02 03 | 04 05 06 07
  10 11 12 13 | 14 15 16 17
  ---
  * after vtrn_u8
  00 10 02 12 | 04 14 06 16
  01 11 03 13 | 05 15 07 17
  */
  const uint8x8x2_t r01_u8 = vtrn_u8(result.val[0], result.val[1]);
  const uint16x4_t x_0_4 = vreinterpret_u16_u8(r01_u8.val[0]);
  const uint16x4_t x_1_5 = vreinterpret_u16_u8(r01_u8.val[1]);
  vst1_lane_u16((uint16_t *)dst, x_0_4, 0);
  dst += pitch;
  vst1_lane_u16((uint16_t *)dst, x_1_5, 0);
  dst += pitch;
  vst1_lane_u16((uint16_t *)dst, x_0_4, 1);
  dst += pitch;
  vst1_lane_u16((uint16_t *)dst, x_1_5, 1);
  dst += pitch;
  vst1_lane_u16((uint16_t *)dst, x_0_4, 2);
  dst += pitch;
  vst1_lane_u16((uint16_t *)dst, x_1_5, 2);
  dst += pitch;
  vst1_lane_u16((uint16_t *)dst, x_0_4, 3);
  dst += pitch;
  vst1_lane_u16((uint16_t *)dst, x_1_5, 3);
}

static INLINE void write_2x8(unsigned char *dst, int pitch,
                             const uint8x8x2_t result,
                             const uint8x8x2_t result2) {
  write_2x4(dst, pitch, result);
  dst += pitch * 8;
  write_2x4(dst, pitch, result2);
}
#else
static INLINE void write_2x8(unsigned char *dst, int pitch,
                             const uint8x8x2_t result,
                             const uint8x8x2_t result2) {
  vst2_lane_u8(dst, result, 0);
  dst += pitch;
  vst2_lane_u8(dst, result, 1);
  dst += pitch;
  vst2_lane_u8(dst, result, 2);
  dst += pitch;
  vst2_lane_u8(dst, result, 3);
  dst += pitch;
  vst2_lane_u8(dst, result, 4);
  dst += pitch;
  vst2_lane_u8(dst, result, 5);
  dst += pitch;
  vst2_lane_u8(dst, result, 6);
  dst += pitch;
  vst2_lane_u8(dst, result, 7);
  dst += pitch;

  vst2_lane_u8(dst, result2, 0);
  dst += pitch;
  vst2_lane_u8(dst, result2, 1);
  dst += pitch;
  vst2_lane_u8(dst, result2, 2);
  dst += pitch;
  vst2_lane_u8(dst, result2, 3);
  dst += pitch;
  vst2_lane_u8(dst, result2, 4);
  dst += pitch;
  vst2_lane_u8(dst, result2, 5);
  dst += pitch;
  vst2_lane_u8(dst, result2, 6);
  dst += pitch;
  vst2_lane_u8(dst, result2, 7);
}
#endif  // VPX_INCOMPATIBLE_GCC

#ifdef VPX_INCOMPATIBLE_GCC
static INLINE uint8x8x4_t read_4x8(unsigned char *src, int pitch) {
  uint8x8x4_t x;
  const uint8x8_t a = vld1_u8(src);
  const uint8x8_t b = vld1_u8(src + pitch * 1);
  const uint8x8_t c = vld1_u8(src + pitch * 2);
  const uint8x8_t d = vld1_u8(src + pitch * 3);
  const uint8x8_t e = vld1_u8(src + pitch * 4);
  const uint8x8_t f = vld1_u8(src + pitch * 5);
  const uint8x8_t g = vld1_u8(src + pitch * 6);
  const uint8x8_t h = vld1_u8(src + pitch * 7);
  const uint32x2x2_t r04_u32 =
      vtrn_u32(vreinterpret_u32_u8(a), vreinterpret_u32_u8(e));
  const uint32x2x2_t r15_u32 =
      vtrn_u32(vreinterpret_u32_u8(b), vreinterpret_u32_u8(f));
  const uint32x2x2_t r26_u32 =
      vtrn_u32(vreinterpret_u32_u8(c), vreinterpret_u32_u8(g));
  const uint32x2x2_t r37_u32 =
      vtrn_u32(vreinterpret_u32_u8(d), vreinterpret_u32_u8(h));
  const uint16x4x2_t r02_u16 = vtrn_u16(vreinterpret_u16_u32(r04_u32.val[0]),
                                        vreinterpret_u16_u32(r26_u32.val[0]));
  const uint16x4x2_t r13_u16 = vtrn_u16(vreinterpret_u16_u32(r15_u32.val[0]),
                                        vreinterpret_u16_u32(r37_u32.val[0]));
  const uint8x8x2_t r01_u8 = vtrn_u8(vreinterpret_u8_u16(r02_u16.val[0]),
                                     vreinterpret_u8_u16(r13_u16.val[0]));
  const uint8x8x2_t r23_u8 = vtrn_u8(vreinterpret_u8_u16(r02_u16.val[1]),
                                     vreinterpret_u8_u16(r13_u16.val[1]));
  /*
   * after vtrn_u32
  00 01 02 03 | 40 41 42 43
  10 11 12 13 | 50 51 52 53
  20 21 22 23 | 60 61 62 63
  30 31 32 33 | 70 71 72 73
  ---
  * after vtrn_u16
  00 01 20 21 | 40 41 60 61
  02 03 22 23 | 42 43 62 63
  10 11 30 31 | 50 51 70 71
  12 13 32 33 | 52 52 72 73

  00 01 20 21 | 40 41 60 61
  10 11 30 31 | 50 51 70 71
  02 03 22 23 | 42 43 62 63
  12 13 32 33 | 52 52 72 73
  ---
  * after vtrn_u8
  00 10 20 30 | 40 50 60 70
  01 11 21 31 | 41 51 61 71
  02 12 22 32 | 42 52 62 72
  03 13 23 33 | 43 53 63 73
  */
  x.val[0] = r01_u8.val[0];
  x.val[1] = r01_u8.val[1];
  x.val[2] = r23_u8.val[0];
  x.val[3] = r23_u8.val[1];

  return x;
}
#else
static INLINE uint8x8x4_t read_4x8(unsigned char *src, int pitch) {
  uint8x8x4_t x;
  x.val[0] = x.val[1] = x.val[2] = x.val[3] = vdup_n_u8(0);
  x = vld4_lane_u8(src, x, 0);
  src += pitch;
  x = vld4_lane_u8(src, x, 1);
  src += pitch;
  x = vld4_lane_u8(src, x, 2);
  src += pitch;
  x = vld4_lane_u8(src, x, 3);
  src += pitch;
  x = vld4_lane_u8(src, x, 4);
  src += pitch;
  x = vld4_lane_u8(src, x, 5);
  src += pitch;
  x = vld4_lane_u8(src, x, 6);
  src += pitch;
  x = vld4_lane_u8(src, x, 7);
  return x;
}
#endif  // VPX_INCOMPATIBLE_GCC

static INLINE void vp8_loop_filter_simple_vertical_edge_neon(
    unsigned char *s, int p, const unsigned char *blimit) {
  unsigned char *src1;
  uint8x16_t qblimit, q0u8;
  uint8x16_t q3u8, q4u8, q5u8, q6u8, q7u8, q11u8, q12u8, q14u8, q15u8;
  int16x8_t q2s16, q13s16, q11s16;
  int8x8_t d28s8, d29s8;
  int8x16_t q2s8, q3s8, q10s8, q11s8, q14s8;
  uint8x8x4_t d0u8x4;  // d6, d7, d8, d9
  uint8x8x4_t d1u8x4;  // d10, d11, d12, d13
  uint8x8x2_t d2u8x2;  // d12, d13
  uint8x8x2_t d3u8x2;  // d14, d15

  qblimit = vdupq_n_u8(*blimit);

  src1 = s - 2;
  d0u8x4 = read_4x8(src1, p);
  src1 += p * 8;
  d1u8x4 = read_4x8(src1, p);

  q3u8 = vcombine_u8(d0u8x4.val[0], d1u8x4.val[0]);  // d6 d10
  q4u8 = vcombine_u8(d0u8x4.val[2], d1u8x4.val[2]);  // d8 d12
  q5u8 = vcombine_u8(d0u8x4.val[1], d1u8x4.val[1]);  // d7 d11
  q6u8 = vcombine_u8(d0u8x4.val[3], d1u8x4.val[3]);  // d9 d13

  q15u8 = vabdq_u8(q5u8, q4u8);
  q14u8 = vabdq_u8(q3u8, q6u8);

  q15u8 = vqaddq_u8(q15u8, q15u8);
  q14u8 = vshrq_n_u8(q14u8, 1);
  q0u8 = vdupq_n_u8(0x80);
  q11s16 = vdupq_n_s16(3);
  q15u8 = vqaddq_u8(q15u8, q14u8);

  q3u8 = veorq_u8(q3u8, q0u8);
  q4u8 = veorq_u8(q4u8, q0u8);
  q5u8 = veorq_u8(q5u8, q0u8);
  q6u8 = veorq_u8(q6u8, q0u8);

  q15u8 = vcgeq_u8(qblimit, q15u8);

  q2s16 = vsubl_s8(vget_low_s8(vreinterpretq_s8_u8(q4u8)),
                   vget_low_s8(vreinterpretq_s8_u8(q5u8)));
  q13s16 = vsubl_s8(vget_high_s8(vreinterpretq_s8_u8(q4u8)),
                    vget_high_s8(vreinterpretq_s8_u8(q5u8)));

  q14s8 = vqsubq_s8(vreinterpretq_s8_u8(q3u8), vreinterpretq_s8_u8(q6u8));

  q2s16 = vmulq_s16(q2s16, q11s16);
  q13s16 = vmulq_s16(q13s16, q11s16);

  q11u8 = vdupq_n_u8(3);
  q12u8 = vdupq_n_u8(4);

  q2s16 = vaddw_s8(q2s16, vget_low_s8(q14s8));
  q13s16 = vaddw_s8(q13s16, vget_high_s8(q14s8));

  d28s8 = vqmovn_s16(q2s16);
  d29s8 = vqmovn_s16(q13s16);
  q14s8 = vcombine_s8(d28s8, d29s8);

  q14s8 = vandq_s8(q14s8, vreinterpretq_s8_u8(q15u8));

  q2s8 = vqaddq_s8(q14s8, vreinterpretq_s8_u8(q11u8));
  q3s8 = vqaddq_s8(q14s8, vreinterpretq_s8_u8(q12u8));
  q2s8 = vshrq_n_s8(q2s8, 3);
  q14s8 = vshrq_n_s8(q3s8, 3);

  q11s8 = vqaddq_s8(vreinterpretq_s8_u8(q5u8), q2s8);
  q10s8 = vqsubq_s8(vreinterpretq_s8_u8(q4u8), q14s8);

  q6u8 = veorq_u8(vreinterpretq_u8_s8(q11s8), q0u8);
  q7u8 = veorq_u8(vreinterpretq_u8_s8(q10s8), q0u8);

  d2u8x2.val[0] = vget_low_u8(q6u8);   // d12
  d2u8x2.val[1] = vget_low_u8(q7u8);   // d14
  d3u8x2.val[0] = vget_high_u8(q6u8);  // d13
  d3u8x2.val[1] = vget_high_u8(q7u8);  // d15

  src1 = s - 1;
  write_2x8(src1, p, d2u8x2, d3u8x2);
}

void vp8_loop_filter_bvs_neon(unsigned char *y_ptr, int y_stride,
                              const unsigned char *blimit) {
  y_ptr += 4;
  vp8_loop_filter_simple_vertical_edge_neon(y_ptr, y_stride, blimit);
  y_ptr += 4;
  vp8_loop_filter_simple_vertical_edge_neon(y_ptr, y_stride, blimit);
  y_ptr += 4;
  vp8_loop_filter_simple_vertical_edge_neon(y_ptr, y_stride, blimit);
  return;
}

void vp8_loop_filter_mbvs_neon(unsigned char *y_ptr, int y_stride,
                               const unsigned char *blimit) {
  vp8_loop_filter_simple_vertical_edge_neon(y_ptr, y_stride, blimit);
  return;
}
