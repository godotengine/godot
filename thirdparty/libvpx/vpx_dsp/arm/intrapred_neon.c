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
#include "./vpx_dsp_rtcd.h"
#include "mem_neon.h"
#include "sum_neon.h"
#include "vpx/vpx_integer.h"

//------------------------------------------------------------------------------
// DC 4x4

static INLINE uint16_t dc_sum_4(const uint8_t *ref) {
  return horizontal_add_uint8x4(load_unaligned_u8_4x1(ref));
}

static INLINE void dc_store_4x4(uint8_t *dst, ptrdiff_t stride,
                                const uint8x8_t dc) {
  int i;
  for (i = 0; i < 4; ++i, dst += stride) {
    vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(dc), 0);
  }
}

void vpx_dc_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x8_t a = load_unaligned_u8_4x1(above);
  const uint8x8_t l = load_unaligned_u8_4x1(left);
  const uint16x4_t al = vget_low_u16(vaddl_u8(a, l));
  const uint16_t sum = horizontal_add_uint16x4(al);
  const uint8x8_t dc = vrshrn_n_u16(vdupq_n_u16(sum), 3);
  dc_store_4x4(dst, stride, dc);
}

void vpx_dc_left_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above, const uint8_t *left) {
  const uint16_t sum = dc_sum_4(left);
  const uint8x8_t dc = vrshrn_n_u16(vdupq_n_u16(sum), 2);
  (void)above;
  dc_store_4x4(dst, stride, dc);
}

void vpx_dc_top_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  const uint16_t sum = dc_sum_4(above);
  const uint8x8_t dc = vrshrn_n_u16(vdupq_n_u16(sum), 2);
  (void)left;
  dc_store_4x4(dst, stride, dc);
}

void vpx_dc_128_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  const uint8x8_t dc = vdup_n_u8(0x80);
  (void)above;
  (void)left;
  dc_store_4x4(dst, stride, dc);
}

//------------------------------------------------------------------------------
// DC 8x8

static INLINE uint16_t dc_sum_8(const uint8_t *ref) {
  return horizontal_add_uint8x8(vld1_u8(ref));
}

static INLINE void dc_store_8x8(uint8_t *dst, ptrdiff_t stride,
                                const uint8x8_t dc) {
  int i;
  for (i = 0; i < 8; ++i, dst += stride) {
    vst1_u8(dst, dc);
  }
}

void vpx_dc_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x8_t above_u8 = vld1_u8(above);
  const uint8x8_t left_u8 = vld1_u8(left);
  const uint16x8_t al = vaddl_u8(above_u8, left_u8);
  const uint16_t sum = horizontal_add_uint16x8(al);
  const uint8x8_t dc = vrshrn_n_u16(vdupq_n_u16(sum), 4);
  dc_store_8x8(dst, stride, dc);
}

void vpx_dc_left_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above, const uint8_t *left) {
  const uint16_t sum = dc_sum_8(left);
  const uint8x8_t dc = vrshrn_n_u16(vdupq_n_u16(sum), 3);
  (void)above;
  dc_store_8x8(dst, stride, dc);
}

void vpx_dc_top_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  const uint16_t sum = dc_sum_8(above);
  const uint8x8_t dc = vrshrn_n_u16(vdupq_n_u16(sum), 3);
  (void)left;
  dc_store_8x8(dst, stride, dc);
}

void vpx_dc_128_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  const uint8x8_t dc = vdup_n_u8(0x80);
  (void)above;
  (void)left;
  dc_store_8x8(dst, stride, dc);
}

//------------------------------------------------------------------------------
// DC 16x16

static INLINE uint16_t dc_sum_16(const uint8_t *ref) {
  return horizontal_add_uint8x16(vld1q_u8(ref));
}

static INLINE void dc_store_16x16(uint8_t *dst, ptrdiff_t stride,
                                  const uint8x16_t dc) {
  int i;
  for (i = 0; i < 16; ++i, dst += stride) {
    vst1q_u8(dst + 0, dc);
  }
}

void vpx_dc_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x16_t ref0 = vld1q_u8(above);
  const uint8x16_t ref1 = vld1q_u8(left);
  const uint16x8_t a = vpaddlq_u8(ref0);
  const uint16x8_t l = vpaddlq_u8(ref1);
  const uint16x8_t al = vaddq_u16(a, l);
  const uint16_t sum = horizontal_add_uint16x8(al);
  const uint8x16_t dc = vdupq_lane_u8(vrshrn_n_u16(vdupq_n_u16(sum), 5), 0);
  dc_store_16x16(dst, stride, dc);
}

void vpx_dc_left_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                      const uint8_t *above,
                                      const uint8_t *left) {
  const uint16_t sum = dc_sum_16(left);
  const uint8x16_t dc = vdupq_lane_u8(vrshrn_n_u16(vdupq_n_u16(sum), 4), 0);
  (void)above;
  dc_store_16x16(dst, stride, dc);
}

void vpx_dc_top_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  const uint16_t sum = dc_sum_16(above);
  const uint8x16_t dc = vdupq_lane_u8(vrshrn_n_u16(vdupq_n_u16(sum), 4), 0);
  (void)left;
  dc_store_16x16(dst, stride, dc);
}

void vpx_dc_128_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  const uint8x16_t dc = vdupq_n_u8(0x80);
  (void)above;
  (void)left;
  dc_store_16x16(dst, stride, dc);
}

//------------------------------------------------------------------------------
// DC 32x32

static INLINE uint16_t dc_sum_32(const uint8_t *ref) {
  const uint8x16_t r0 = vld1q_u8(ref + 0);
  const uint8x16_t r1 = vld1q_u8(ref + 16);
  const uint16x8_t r01 = vaddq_u16(vpaddlq_u8(r0), vpaddlq_u8(r1));
  return horizontal_add_uint16x8(r01);
}

static INLINE void dc_store_32x32(uint8_t *dst, ptrdiff_t stride,
                                  const uint8x16_t dc) {
  int i;
  for (i = 0; i < 32; ++i, dst += stride) {
    vst1q_u8(dst + 0, dc);
    vst1q_u8(dst + 16, dc);
  }
}

void vpx_dc_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x16_t a0 = vld1q_u8(above + 0);
  const uint8x16_t a1 = vld1q_u8(above + 16);
  const uint8x16_t l0 = vld1q_u8(left + 0);
  const uint8x16_t l1 = vld1q_u8(left + 16);
  const uint16x8_t a01 = vaddq_u16(vpaddlq_u8(a0), vpaddlq_u8(a1));
  const uint16x8_t l01 = vaddq_u16(vpaddlq_u8(l0), vpaddlq_u8(l1));
  const uint16x8_t al = vaddq_u16(a01, l01);
  const uint16_t sum = horizontal_add_uint16x8(al);
  const uint8x16_t dc = vdupq_lane_u8(vrshrn_n_u16(vdupq_n_u16(sum), 6), 0);
  dc_store_32x32(dst, stride, dc);
}

void vpx_dc_left_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                      const uint8_t *above,
                                      const uint8_t *left) {
  const uint16_t sum = dc_sum_32(left);
  const uint8x16_t dc = vdupq_lane_u8(vrshrn_n_u16(vdupq_n_u16(sum), 5), 0);
  (void)above;
  dc_store_32x32(dst, stride, dc);
}

void vpx_dc_top_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  const uint16_t sum = dc_sum_32(above);
  const uint8x16_t dc = vdupq_lane_u8(vrshrn_n_u16(vdupq_n_u16(sum), 5), 0);
  (void)left;
  dc_store_32x32(dst, stride, dc);
}

void vpx_dc_128_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  const uint8x16_t dc = vdupq_n_u8(0x80);
  (void)above;
  (void)left;
  dc_store_32x32(dst, stride, dc);
}

// -----------------------------------------------------------------------------

void vpx_d45_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  uint8x8_t a0, a1, a2, d0;
  uint8_t a7;
  (void)left;

  a0 = vld1_u8(above);
  a7 = above[7];

  // [ above[1], ..., above[6], x, x ]
  a1 = vext_u8(a0, a0, 1);
  // [ above[2], ..., above[7], x, x ]
  a2 = vext_u8(a0, a0, 2);

  // d0[0] = AVG3(above[0], above[1], above[2]);
  // ...
  // d0[5] = AVG3(above[5], above[6], above[7]);
  // d0[6] = x (don't care)
  // d0[7] = x (don't care)
  d0 = vrhadd_u8(vhadd_u8(a0, a2), a1);

  // We want:
  // stride=0 [ d0[0], d0[1], d0[2],    d0[3] ]
  // stride=1 [ d0[1], d0[2], d0[3],    d0[4] ]
  // stride=2 [ d0[2], d0[3], d0[4],    d0[5] ]
  // stride=2 [ d0[3], d0[4], d0[5], above[7] ]
  store_u8_4x1(dst + 0 * stride, d0);
  store_u8_4x1(dst + 1 * stride, vext_u8(d0, d0, 1));
  store_u8_4x1(dst + 2 * stride, vext_u8(d0, d0, 2));
  store_u8_4x1(dst + 3 * stride, vext_u8(d0, d0, 3));

  // We stored d0[6] above, so fixup into above[7].
  dst[3 * stride + 3] = a7;
}

void vpx_d45_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  uint8x8_t ax0, a0, a1, a7, d0;
  (void)left;

  a0 = vld1_u8(above + 0);
  a1 = vld1_u8(above + 1);
  a7 = vld1_dup_u8(above + 7);

  // We want to calculate the AVG3 result in lanes 1-7 inclusive so we can
  // shift in above[7] later, so shift a0 across by one to get the right
  // inputs:
  // [ x, above[0], ... , above[6] ]
  ax0 = vext_u8(a0, a0, 7);

  // d0[0] = x (don't care)
  // d0[1] = AVG3(above[0], above[1], above[2]);
  // ...
  // d0[7] = AVG3(above[6], above[7], above[8]);
  d0 = vrhadd_u8(vhadd_u8(ax0, a1), a0);

  // Undo the earlier ext, incrementally shift in duplicates of above[7].
  vst1_u8(dst + 0 * stride, vext_u8(d0, a7, 1));
  vst1_u8(dst + 1 * stride, vext_u8(d0, a7, 2));
  vst1_u8(dst + 2 * stride, vext_u8(d0, a7, 3));
  vst1_u8(dst + 3 * stride, vext_u8(d0, a7, 4));
  vst1_u8(dst + 4 * stride, vext_u8(d0, a7, 5));
  vst1_u8(dst + 5 * stride, vext_u8(d0, a7, 6));
  vst1_u8(dst + 6 * stride, vext_u8(d0, a7, 7));
  vst1_u8(dst + 7 * stride, a7);
}

void vpx_d45_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                  const uint8_t *above, const uint8_t *left) {
  uint8x16_t ax0, a0, a1, a15, d0;
  (void)left;

  a0 = vld1q_u8(above + 0);
  a1 = vld1q_u8(above + 1);
  a15 = vld1q_dup_u8(above + 15);

  // We want to calculate the AVG3 result in lanes 1-15 inclusive so we can
  // shift in above[15] later, so shift a0 across by one to get the right
  // inputs:
  // [ x, above[0], ... , above[14] ]
  ax0 = vextq_u8(a0, a0, 15);

  // d0[0] = x (don't care)
  // d0[1] = AVG3(above[0], above[1], above[2]);
  // ...
  // d0[15] = AVG3(above[14], above[15], above[16]);
  d0 = vrhaddq_u8(vhaddq_u8(ax0, a1), a0);

  // Undo the earlier ext, incrementally shift in duplicates of above[15].
  vst1q_u8(dst + 0 * stride, vextq_u8(d0, a15, 1));
  vst1q_u8(dst + 1 * stride, vextq_u8(d0, a15, 2));
  vst1q_u8(dst + 2 * stride, vextq_u8(d0, a15, 3));
  vst1q_u8(dst + 3 * stride, vextq_u8(d0, a15, 4));
  vst1q_u8(dst + 4 * stride, vextq_u8(d0, a15, 5));
  vst1q_u8(dst + 5 * stride, vextq_u8(d0, a15, 6));
  vst1q_u8(dst + 6 * stride, vextq_u8(d0, a15, 7));
  vst1q_u8(dst + 7 * stride, vextq_u8(d0, a15, 8));
  vst1q_u8(dst + 8 * stride, vextq_u8(d0, a15, 9));
  vst1q_u8(dst + 9 * stride, vextq_u8(d0, a15, 10));
  vst1q_u8(dst + 10 * stride, vextq_u8(d0, a15, 11));
  vst1q_u8(dst + 11 * stride, vextq_u8(d0, a15, 12));
  vst1q_u8(dst + 12 * stride, vextq_u8(d0, a15, 13));
  vst1q_u8(dst + 13 * stride, vextq_u8(d0, a15, 14));
  vst1q_u8(dst + 14 * stride, vextq_u8(d0, a15, 15));
  vst1q_u8(dst + 15 * stride, a15);
}

void vpx_d45_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                  const uint8_t *above, const uint8_t *left) {
  uint8x16_t ax0, a0, a1, a15, a16, a17, a31, d0[2];
  (void)left;

  a0 = vld1q_u8(above + 0);
  a1 = vld1q_u8(above + 1);
  a15 = vld1q_u8(above + 15);
  a16 = vld1q_u8(above + 16);
  a17 = vld1q_u8(above + 17);
  a31 = vld1q_dup_u8(above + 31);

  // We want to calculate the AVG3 result in lanes 1-15 inclusive so we can
  // shift in above[15] later, so shift a0 across by one to get the right
  // inputs:
  // [ x, above[0], ... , above[14] ]
  ax0 = vextq_u8(a0, a0, 15);

  // d0[0] = x (don't care)
  // d0[1] = AVG3(above[0], above[1], above[2]);
  // ...
  // d0[15] = AVG3(above[14], above[15], above[16]);
  d0[0] = vrhaddq_u8(vhaddq_u8(ax0, a1), a0);
  d0[1] = vrhaddq_u8(vhaddq_u8(a15, a17), a16);

  // Undo the earlier ext, incrementally shift in duplicates of above[15].
  vst1q_u8(dst + 0 * stride + 0, vextq_u8(d0[0], d0[1], 1));
  vst1q_u8(dst + 0 * stride + 16, vextq_u8(d0[1], a31, 1));
  vst1q_u8(dst + 1 * stride + 0, vextq_u8(d0[0], d0[1], 2));
  vst1q_u8(dst + 1 * stride + 16, vextq_u8(d0[1], a31, 2));
  vst1q_u8(dst + 2 * stride + 0, vextq_u8(d0[0], d0[1], 3));
  vst1q_u8(dst + 2 * stride + 16, vextq_u8(d0[1], a31, 3));
  vst1q_u8(dst + 3 * stride + 0, vextq_u8(d0[0], d0[1], 4));
  vst1q_u8(dst + 3 * stride + 16, vextq_u8(d0[1], a31, 4));
  vst1q_u8(dst + 4 * stride + 0, vextq_u8(d0[0], d0[1], 5));
  vst1q_u8(dst + 4 * stride + 16, vextq_u8(d0[1], a31, 5));
  vst1q_u8(dst + 5 * stride + 0, vextq_u8(d0[0], d0[1], 6));
  vst1q_u8(dst + 5 * stride + 16, vextq_u8(d0[1], a31, 6));
  vst1q_u8(dst + 6 * stride + 0, vextq_u8(d0[0], d0[1], 7));
  vst1q_u8(dst + 6 * stride + 16, vextq_u8(d0[1], a31, 7));
  vst1q_u8(dst + 7 * stride + 0, vextq_u8(d0[0], d0[1], 8));
  vst1q_u8(dst + 7 * stride + 16, vextq_u8(d0[1], a31, 8));
  vst1q_u8(dst + 8 * stride + 0, vextq_u8(d0[0], d0[1], 9));
  vst1q_u8(dst + 8 * stride + 16, vextq_u8(d0[1], a31, 9));
  vst1q_u8(dst + 9 * stride + 0, vextq_u8(d0[0], d0[1], 10));
  vst1q_u8(dst + 9 * stride + 16, vextq_u8(d0[1], a31, 10));
  vst1q_u8(dst + 10 * stride + 0, vextq_u8(d0[0], d0[1], 11));
  vst1q_u8(dst + 10 * stride + 16, vextq_u8(d0[1], a31, 11));
  vst1q_u8(dst + 11 * stride + 0, vextq_u8(d0[0], d0[1], 12));
  vst1q_u8(dst + 11 * stride + 16, vextq_u8(d0[1], a31, 12));
  vst1q_u8(dst + 12 * stride + 0, vextq_u8(d0[0], d0[1], 13));
  vst1q_u8(dst + 12 * stride + 16, vextq_u8(d0[1], a31, 13));
  vst1q_u8(dst + 13 * stride + 0, vextq_u8(d0[0], d0[1], 14));
  vst1q_u8(dst + 13 * stride + 16, vextq_u8(d0[1], a31, 14));
  vst1q_u8(dst + 14 * stride + 0, vextq_u8(d0[0], d0[1], 15));
  vst1q_u8(dst + 14 * stride + 16, vextq_u8(d0[1], a31, 15));
  vst1q_u8(dst + 15 * stride + 0, d0[1]);
  vst1q_u8(dst + 15 * stride + 16, a31);

  vst1q_u8(dst + 16 * stride + 0, vextq_u8(d0[1], a31, 1));
  vst1q_u8(dst + 16 * stride + 16, a31);
  vst1q_u8(dst + 17 * stride + 0, vextq_u8(d0[1], a31, 2));
  vst1q_u8(dst + 17 * stride + 16, a31);
  vst1q_u8(dst + 18 * stride + 0, vextq_u8(d0[1], a31, 3));
  vst1q_u8(dst + 18 * stride + 16, a31);
  vst1q_u8(dst + 19 * stride + 0, vextq_u8(d0[1], a31, 4));
  vst1q_u8(dst + 19 * stride + 16, a31);
  vst1q_u8(dst + 20 * stride + 0, vextq_u8(d0[1], a31, 5));
  vst1q_u8(dst + 20 * stride + 16, a31);
  vst1q_u8(dst + 21 * stride + 0, vextq_u8(d0[1], a31, 6));
  vst1q_u8(dst + 21 * stride + 16, a31);
  vst1q_u8(dst + 22 * stride + 0, vextq_u8(d0[1], a31, 7));
  vst1q_u8(dst + 22 * stride + 16, a31);
  vst1q_u8(dst + 23 * stride + 0, vextq_u8(d0[1], a31, 8));
  vst1q_u8(dst + 23 * stride + 16, a31);
  vst1q_u8(dst + 24 * stride + 0, vextq_u8(d0[1], a31, 9));
  vst1q_u8(dst + 24 * stride + 16, a31);
  vst1q_u8(dst + 25 * stride + 0, vextq_u8(d0[1], a31, 10));
  vst1q_u8(dst + 25 * stride + 16, a31);
  vst1q_u8(dst + 26 * stride + 0, vextq_u8(d0[1], a31, 11));
  vst1q_u8(dst + 26 * stride + 16, a31);
  vst1q_u8(dst + 27 * stride + 0, vextq_u8(d0[1], a31, 12));
  vst1q_u8(dst + 27 * stride + 16, a31);
  vst1q_u8(dst + 28 * stride + 0, vextq_u8(d0[1], a31, 13));
  vst1q_u8(dst + 28 * stride + 16, a31);
  vst1q_u8(dst + 29 * stride + 0, vextq_u8(d0[1], a31, 14));
  vst1q_u8(dst + 29 * stride + 16, a31);
  vst1q_u8(dst + 30 * stride + 0, vextq_u8(d0[1], a31, 15));
  vst1q_u8(dst + 30 * stride + 16, a31);
  vst1q_u8(dst + 31 * stride + 0, a31);
  vst1q_u8(dst + 31 * stride + 16, a31);
}

// -----------------------------------------------------------------------------

void vpx_d63_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  uint8x8_t a0, a1, a2, a3, d0, d1, d2, d3;
  (void)left;

  a0 = load_unaligned_u8_4x1(above + 0);
  a1 = load_unaligned_u8_4x1(above + 1);
  a2 = load_unaligned_u8_4x1(above + 2);
  a3 = load_unaligned_u8_4x1(above + 3);

  d0 = vrhadd_u8(a0, a1);
  d1 = vrhadd_u8(vhadd_u8(a0, a2), a1);
  d2 = vrhadd_u8(a1, a2);
  d3 = vrhadd_u8(vhadd_u8(a1, a3), a2);

  store_u8_4x1(dst + 0 * stride, d0);
  store_u8_4x1(dst + 1 * stride, d1);
  store_u8_4x1(dst + 2 * stride, d2);
  store_u8_4x1(dst + 3 * stride, d3);
}

void vpx_d63_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  uint8x8_t a0, a1, a2, a7, d0, d1;
  (void)left;

  a0 = vld1_u8(above + 0);
  a1 = vld1_u8(above + 1);
  a2 = vld1_u8(above + 2);
  a7 = vld1_dup_u8(above + 7);

  d0 = vrhadd_u8(a0, a1);
  d1 = vrhadd_u8(vhadd_u8(a0, a2), a1);

  vst1_u8(dst + 0 * stride, d0);
  vst1_u8(dst + 1 * stride, d1);

  d0 = vext_u8(d0, d0, 7);
  d1 = vext_u8(d1, d1, 7);

  vst1_u8(dst + 2 * stride, vext_u8(d0, a7, 2));
  vst1_u8(dst + 3 * stride, vext_u8(d1, a7, 2));
  vst1_u8(dst + 4 * stride, vext_u8(d0, a7, 3));
  vst1_u8(dst + 5 * stride, vext_u8(d1, a7, 3));
  vst1_u8(dst + 6 * stride, vext_u8(d0, a7, 4));
  vst1_u8(dst + 7 * stride, vext_u8(d1, a7, 4));
}

void vpx_d63_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                  const uint8_t *above, const uint8_t *left) {
  uint8x16_t a0, a1, a2, a15, d0, d1;
  (void)left;

  a0 = vld1q_u8(above + 0);
  a1 = vld1q_u8(above + 1);
  a2 = vld1q_u8(above + 2);
  a15 = vld1q_dup_u8(above + 15);

  d0 = vrhaddq_u8(a0, a1);
  d1 = vrhaddq_u8(vhaddq_u8(a0, a2), a1);

  vst1q_u8(dst + 0 * stride, d0);
  vst1q_u8(dst + 1 * stride, d1);

  d0 = vextq_u8(d0, d0, 15);
  d1 = vextq_u8(d1, d1, 15);

  vst1q_u8(dst + 2 * stride, vextq_u8(d0, a15, 2));
  vst1q_u8(dst + 3 * stride, vextq_u8(d1, a15, 2));
  vst1q_u8(dst + 4 * stride, vextq_u8(d0, a15, 3));
  vst1q_u8(dst + 5 * stride, vextq_u8(d1, a15, 3));
  vst1q_u8(dst + 6 * stride, vextq_u8(d0, a15, 4));
  vst1q_u8(dst + 7 * stride, vextq_u8(d1, a15, 4));
  vst1q_u8(dst + 8 * stride, vextq_u8(d0, a15, 5));
  vst1q_u8(dst + 9 * stride, vextq_u8(d1, a15, 5));
  vst1q_u8(dst + 10 * stride, vextq_u8(d0, a15, 6));
  vst1q_u8(dst + 11 * stride, vextq_u8(d1, a15, 6));
  vst1q_u8(dst + 12 * stride, vextq_u8(d0, a15, 7));
  vst1q_u8(dst + 13 * stride, vextq_u8(d1, a15, 7));
  vst1q_u8(dst + 14 * stride, vextq_u8(d0, a15, 8));
  vst1q_u8(dst + 15 * stride, vextq_u8(d1, a15, 8));
}

void vpx_d63_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                  const uint8_t *above, const uint8_t *left) {
  uint8x16_t a0, a1, a2, a16, a17, a18, a31, d0_lo, d0_hi, d1_lo, d1_hi;
  (void)left;

  a0 = vld1q_u8(above + 0);
  a1 = vld1q_u8(above + 1);
  a2 = vld1q_u8(above + 2);
  a16 = vld1q_u8(above + 16);
  a17 = vld1q_u8(above + 17);
  a18 = vld1q_u8(above + 18);
  a31 = vld1q_dup_u8(above + 31);

  d0_lo = vrhaddq_u8(a0, a1);
  d0_hi = vrhaddq_u8(a16, a17);
  d1_lo = vrhaddq_u8(vhaddq_u8(a0, a2), a1);
  d1_hi = vrhaddq_u8(vhaddq_u8(a16, a18), a17);

  vst1q_u8(dst + 0 * stride + 0, d0_lo);
  vst1q_u8(dst + 0 * stride + 16, d0_hi);
  vst1q_u8(dst + 1 * stride + 0, d1_lo);
  vst1q_u8(dst + 1 * stride + 16, d1_hi);

  d0_hi = vextq_u8(d0_lo, d0_hi, 15);
  d0_lo = vextq_u8(d0_lo, d0_lo, 15);
  d1_hi = vextq_u8(d1_lo, d1_hi, 15);
  d1_lo = vextq_u8(d1_lo, d1_lo, 15);

  vst1q_u8(dst + 2 * stride + 0, vextq_u8(d0_lo, d0_hi, 2));
  vst1q_u8(dst + 2 * stride + 16, vextq_u8(d0_hi, a31, 2));
  vst1q_u8(dst + 3 * stride + 0, vextq_u8(d1_lo, d1_hi, 2));
  vst1q_u8(dst + 3 * stride + 16, vextq_u8(d1_hi, a31, 2));
  vst1q_u8(dst + 4 * stride + 0, vextq_u8(d0_lo, d0_hi, 3));
  vst1q_u8(dst + 4 * stride + 16, vextq_u8(d0_hi, a31, 3));
  vst1q_u8(dst + 5 * stride + 0, vextq_u8(d1_lo, d1_hi, 3));
  vst1q_u8(dst + 5 * stride + 16, vextq_u8(d1_hi, a31, 3));
  vst1q_u8(dst + 6 * stride + 0, vextq_u8(d0_lo, d0_hi, 4));
  vst1q_u8(dst + 6 * stride + 16, vextq_u8(d0_hi, a31, 4));
  vst1q_u8(dst + 7 * stride + 0, vextq_u8(d1_lo, d1_hi, 4));
  vst1q_u8(dst + 7 * stride + 16, vextq_u8(d1_hi, a31, 4));
  vst1q_u8(dst + 8 * stride + 0, vextq_u8(d0_lo, d0_hi, 5));
  vst1q_u8(dst + 8 * stride + 16, vextq_u8(d0_hi, a31, 5));
  vst1q_u8(dst + 9 * stride + 0, vextq_u8(d1_lo, d1_hi, 5));
  vst1q_u8(dst + 9 * stride + 16, vextq_u8(d1_hi, a31, 5));
  vst1q_u8(dst + 10 * stride + 0, vextq_u8(d0_lo, d0_hi, 6));
  vst1q_u8(dst + 10 * stride + 16, vextq_u8(d0_hi, a31, 6));
  vst1q_u8(dst + 11 * stride + 0, vextq_u8(d1_lo, d1_hi, 6));
  vst1q_u8(dst + 11 * stride + 16, vextq_u8(d1_hi, a31, 6));
  vst1q_u8(dst + 12 * stride + 0, vextq_u8(d0_lo, d0_hi, 7));
  vst1q_u8(dst + 12 * stride + 16, vextq_u8(d0_hi, a31, 7));
  vst1q_u8(dst + 13 * stride + 0, vextq_u8(d1_lo, d1_hi, 7));
  vst1q_u8(dst + 13 * stride + 16, vextq_u8(d1_hi, a31, 7));
  vst1q_u8(dst + 14 * stride + 0, vextq_u8(d0_lo, d0_hi, 8));
  vst1q_u8(dst + 14 * stride + 16, vextq_u8(d0_hi, a31, 8));
  vst1q_u8(dst + 15 * stride + 0, vextq_u8(d1_lo, d1_hi, 8));
  vst1q_u8(dst + 15 * stride + 16, vextq_u8(d1_hi, a31, 8));
  vst1q_u8(dst + 16 * stride + 0, vextq_u8(d0_lo, d0_hi, 9));
  vst1q_u8(dst + 16 * stride + 16, vextq_u8(d0_hi, a31, 9));
  vst1q_u8(dst + 17 * stride + 0, vextq_u8(d1_lo, d1_hi, 9));
  vst1q_u8(dst + 17 * stride + 16, vextq_u8(d1_hi, a31, 9));
  vst1q_u8(dst + 18 * stride + 0, vextq_u8(d0_lo, d0_hi, 10));
  vst1q_u8(dst + 18 * stride + 16, vextq_u8(d0_hi, a31, 10));
  vst1q_u8(dst + 19 * stride + 0, vextq_u8(d1_lo, d1_hi, 10));
  vst1q_u8(dst + 19 * stride + 16, vextq_u8(d1_hi, a31, 10));
  vst1q_u8(dst + 20 * stride + 0, vextq_u8(d0_lo, d0_hi, 11));
  vst1q_u8(dst + 20 * stride + 16, vextq_u8(d0_hi, a31, 11));
  vst1q_u8(dst + 21 * stride + 0, vextq_u8(d1_lo, d1_hi, 11));
  vst1q_u8(dst + 21 * stride + 16, vextq_u8(d1_hi, a31, 11));
  vst1q_u8(dst + 22 * stride + 0, vextq_u8(d0_lo, d0_hi, 12));
  vst1q_u8(dst + 22 * stride + 16, vextq_u8(d0_hi, a31, 12));
  vst1q_u8(dst + 23 * stride + 0, vextq_u8(d1_lo, d1_hi, 12));
  vst1q_u8(dst + 23 * stride + 16, vextq_u8(d1_hi, a31, 12));
  vst1q_u8(dst + 24 * stride + 0, vextq_u8(d0_lo, d0_hi, 13));
  vst1q_u8(dst + 24 * stride + 16, vextq_u8(d0_hi, a31, 13));
  vst1q_u8(dst + 25 * stride + 0, vextq_u8(d1_lo, d1_hi, 13));
  vst1q_u8(dst + 25 * stride + 16, vextq_u8(d1_hi, a31, 13));
  vst1q_u8(dst + 26 * stride + 0, vextq_u8(d0_lo, d0_hi, 14));
  vst1q_u8(dst + 26 * stride + 16, vextq_u8(d0_hi, a31, 14));
  vst1q_u8(dst + 27 * stride + 0, vextq_u8(d1_lo, d1_hi, 14));
  vst1q_u8(dst + 27 * stride + 16, vextq_u8(d1_hi, a31, 14));
  vst1q_u8(dst + 28 * stride + 0, vextq_u8(d0_lo, d0_hi, 15));
  vst1q_u8(dst + 28 * stride + 16, vextq_u8(d0_hi, a31, 15));
  vst1q_u8(dst + 29 * stride + 0, vextq_u8(d1_lo, d1_hi, 15));
  vst1q_u8(dst + 29 * stride + 16, vextq_u8(d1_hi, a31, 15));
  vst1q_u8(dst + 30 * stride + 0, d0_hi);
  vst1q_u8(dst + 30 * stride + 16, a31);
  vst1q_u8(dst + 31 * stride + 0, d1_hi);
  vst1q_u8(dst + 31 * stride + 16, a31);
}

// -----------------------------------------------------------------------------

void vpx_d117_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  // See vpx_d117_predictor_8x8_neon for more details on the implementation.
  uint8x8_t az, a0, l0az, d0, d1, d2, d3, col0, col1;

  az = load_unaligned_u8_4x1(above - 1);
  a0 = load_unaligned_u8_4x1(above + 0);
  // [ left[0], above[-1], above[0], above[1], x, x, x, x ]
  l0az = vext_u8(vld1_dup_u8(left), az, 7);

  col0 = vdup_n_u8((above[-1] + 2 * left[0] + left[1] + 2) >> 2);
  col1 = vdup_n_u8((left[0] + 2 * left[1] + left[2] + 2) >> 2);

  d0 = vrhadd_u8(az, a0);
  d1 = vrhadd_u8(vhadd_u8(l0az, a0), az);
  d2 = vext_u8(col0, d0, 7);
  d3 = vext_u8(col1, d1, 7);

  store_u8_4x1(dst + 0 * stride, d0);
  store_u8_4x1(dst + 1 * stride, d1);
  store_u8_4x1(dst + 2 * stride, d2);
  store_u8_4x1(dst + 3 * stride, d3);
}

void vpx_d117_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  uint8x8_t az, a0, l0az, d0, d1, l0, l1, azl0, col0, col0_even, col0_odd;

  az = vld1_u8(above - 1);
  a0 = vld1_u8(above + 0);
  // [ left[0], above[-1], ... , above[5] ]
  l0az = vext_u8(vld1_dup_u8(left), az, 7);

  l0 = vld1_u8(left + 0);
  // The last lane here is unused, reading left[8] could cause a buffer
  // over-read, so just fill with a duplicate of left[0] to avoid needing to
  // materialize a zero:
  // [ left[1], ... , left[7], x ]
  l1 = vext_u8(l0, l0, 1);
  // [ above[-1], left[0], ... , left[6] ]
  azl0 = vext_u8(vld1_dup_u8(above - 1), l0, 7);

  // d0[0] = AVG2(above[-1], above[0])
  // d0[1] = AVG2(above[0], above[1])
  // ...
  // d0[7] = AVG2(above[6], above[7])
  d0 = vrhadd_u8(az, a0);

  // d1[0] = AVG3(left[0], above[-1], above[0])
  // d1[1] = AVG3(above[-1], above[0], above[1])
  // ...
  // d1[7] = AVG3(above[5], above[6], above[7])
  d1 = vrhadd_u8(vhadd_u8(l0az, a0), az);

  // The ext instruction shifts elements in from the end of the vector rather
  // than the start, so reverse the vector to put the elements to be shifted in
  // at the end. The lowest two lanes here are unused:
  // col0[7] = AVG3(above[-1], left[0], left[1])
  // col0[6] = AVG3(left[0], left[1], left[2])
  // ...
  // col0[2] = AVG3(left[4], left[5], left[6])
  // col0[1] = x (don't care)
  // col0[0] = x (don't care)
  col0 = vrev64_u8(vrhadd_u8(vhadd_u8(azl0, l1), l0));

  // We don't care about the first parameter to this uzp since we only ever use
  // the high three elements, we just use col0 again since it is already
  // available:
  // col0_even = [ x, x, x, x, x, col0[3], col0[5], col0[7] ]
  // col0_odd = [ x, x, x, x, x, col0[2], col0[4], col0[6] ]
  col0_even = vuzp_u8(col0, col0).val[1];
  col0_odd = vuzp_u8(col0, col0).val[0];

  // Incrementally shift more elements from col0 into d0/1:
  // stride=0 [ d0[0],   d0[1],   d0[2],   d0[3], d0[4], d0[5], d0[6], d0[7] ]
  // stride=1 [ d1[0],   d1[1],   d1[2],   d1[3], d1[4], d1[5], d1[6], d1[7] ]
  // stride=2 [ col0[7], d0[0],   d0[1],   d0[2], d0[3], d0[4], d0[5], d0[6] ]
  // stride=3 [ col0[6], d1[0],   d1[1],   d1[2], d1[3], d1[4], d1[5], d1[6] ]
  // stride=4 [ col0[5], col0[7], d0[0],   d0[1], d0[2], d0[3], d0[4], d0[5] ]
  // stride=5 [ col0[4], col0[6], d1[0],   d1[1], d1[2], d1[3], d1[4], d1[5] ]
  // stride=6 [ col0[3], col0[5], col0[7], d0[0], d0[1], d0[2], d0[3], d0[4] ]
  // stride=7 [ col0[2], col0[4], col0[6], d1[0], d1[1], d1[2], d1[3], d1[4] ]
  vst1_u8(dst + 0 * stride, d0);
  vst1_u8(dst + 1 * stride, d1);
  vst1_u8(dst + 2 * stride, vext_u8(col0_even, d0, 7));
  vst1_u8(dst + 3 * stride, vext_u8(col0_odd, d1, 7));
  vst1_u8(dst + 4 * stride, vext_u8(col0_even, d0, 6));
  vst1_u8(dst + 5 * stride, vext_u8(col0_odd, d1, 6));
  vst1_u8(dst + 6 * stride, vext_u8(col0_even, d0, 5));
  vst1_u8(dst + 7 * stride, vext_u8(col0_odd, d1, 5));
}

void vpx_d117_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  // See vpx_d117_predictor_8x8_neon for more details on the implementation.
  uint8x16_t az, a0, l0az, d0, d1, l0, l1, azl0, col0, col0_even, col0_odd;

  az = vld1q_u8(above - 1);
  a0 = vld1q_u8(above + 0);
  // [ left[0], above[-1], ... , above[13] ]
  l0az = vextq_u8(vld1q_dup_u8(left), az, 15);

  l0 = vld1q_u8(left + 0);
  // The last lane here is unused, reading left[16] could cause a buffer
  // over-read, so just fill with a duplicate of left[0] to avoid needing to
  // materialize a zero:
  // [ left[1], ... , left[15], x ]
  l1 = vextq_u8(l0, l0, 1);
  // [ above[-1], left[0], ... , left[14] ]
  azl0 = vextq_u8(vld1q_dup_u8(above - 1), l0, 15);

  d0 = vrhaddq_u8(az, a0);
  d1 = vrhaddq_u8(vhaddq_u8(l0az, a0), az);

  col0 = vrhaddq_u8(vhaddq_u8(azl0, l1), l0);
  col0 = vrev64q_u8(vextq_u8(col0, col0, 8));

  // The low nine lanes here are unused so the first input to the uzp is
  // unused, so just use a duplicate of col0 since we have it already. This
  // also means that the lowest lane of col0 here is unused.
  col0_even = vuzpq_u8(col0, col0).val[1];
  col0_odd = vuzpq_u8(col0, col0).val[0];

  vst1q_u8(dst + 0 * stride, d0);
  vst1q_u8(dst + 1 * stride, d1);
  vst1q_u8(dst + 2 * stride, vextq_u8(col0_even, d0, 15));
  vst1q_u8(dst + 3 * stride, vextq_u8(col0_odd, d1, 15));
  vst1q_u8(dst + 4 * stride, vextq_u8(col0_even, d0, 14));
  vst1q_u8(dst + 5 * stride, vextq_u8(col0_odd, d1, 14));
  vst1q_u8(dst + 6 * stride, vextq_u8(col0_even, d0, 13));
  vst1q_u8(dst + 7 * stride, vextq_u8(col0_odd, d1, 13));
  vst1q_u8(dst + 8 * stride, vextq_u8(col0_even, d0, 12));
  vst1q_u8(dst + 9 * stride, vextq_u8(col0_odd, d1, 12));
  vst1q_u8(dst + 10 * stride, vextq_u8(col0_even, d0, 11));
  vst1q_u8(dst + 11 * stride, vextq_u8(col0_odd, d1, 11));
  vst1q_u8(dst + 12 * stride, vextq_u8(col0_even, d0, 10));
  vst1q_u8(dst + 13 * stride, vextq_u8(col0_odd, d1, 10));
  vst1q_u8(dst + 14 * stride, vextq_u8(col0_even, d0, 9));
  vst1q_u8(dst + 15 * stride, vextq_u8(col0_odd, d1, 9));
}

void vpx_d117_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  // See vpx_d117_predictor_8x8_neon for more details on the implementation.
  uint8x16_t az, a0, a14, a15, a16, l0az, d0_lo, d0_hi, d1_lo, d1_hi, l0, l1,
      l15, l16, l17, azl0, col0_lo, col0_hi, col0_even, col0_odd;

  az = vld1q_u8(above - 1);
  a0 = vld1q_u8(above + 0);
  a14 = vld1q_u8(above + 14);
  a15 = vld1q_u8(above + 15);
  a16 = vld1q_u8(above + 16);
  // [ left[0], above[-1], ... , above[13] ]
  l0az = vextq_u8(vld1q_dup_u8(left), az, 15);

  l0 = vld1q_u8(left + 0);
  l1 = vld1q_u8(left + 1);
  l15 = vld1q_u8(left + 15);
  l16 = vld1q_u8(left + 16);
  // The last lane here is unused, reading left[32] would cause a buffer
  // over-read (observed as an address-sanitizer failure), so just fill with a
  // duplicate of left[16] to avoid needing to materialize a zero:
  // [ left[17], ... , left[31], x ]
  l17 = vextq_u8(l16, l16, 1);
  // [ above[-1], left[0], ... , left[14] ]
  azl0 = vextq_u8(vld1q_dup_u8(above - 1), l0, 15);

  d0_lo = vrhaddq_u8(az, a0);
  d0_hi = vrhaddq_u8(a15, a16);
  d1_lo = vrhaddq_u8(vhaddq_u8(l0az, a0), az);
  d1_hi = vrhaddq_u8(vhaddq_u8(a14, a16), a15);

  // The last lane of col0_hi is unused here.
  col0_lo = vrhaddq_u8(vhaddq_u8(azl0, l1), l0);
  col0_hi = vrhaddq_u8(vhaddq_u8(l15, l17), l16);

  col0_lo = vrev64q_u8(vextq_u8(col0_lo, col0_lo, 8));
  col0_hi = vrev64q_u8(vextq_u8(col0_hi, col0_hi, 8));

  // The first lane of these are unused since they are only ever called as
  // ext(col0, _, i) where i >= 1.
  col0_even = vuzpq_u8(col0_hi, col0_lo).val[1];
  col0_odd = vuzpq_u8(col0_hi, col0_lo).val[0];

  vst1q_u8(dst + 0 * stride + 0, d0_lo);
  vst1q_u8(dst + 0 * stride + 16, d0_hi);
  vst1q_u8(dst + 1 * stride + 0, d1_lo);
  vst1q_u8(dst + 1 * stride + 16, d1_hi);
  vst1q_u8(dst + 2 * stride + 0, vextq_u8(col0_even, d0_lo, 15));
  vst1q_u8(dst + 2 * stride + 16, vextq_u8(d0_lo, d0_hi, 15));
  vst1q_u8(dst + 3 * stride + 0, vextq_u8(col0_odd, d1_lo, 15));
  vst1q_u8(dst + 3 * stride + 16, vextq_u8(d1_lo, d1_hi, 15));
  vst1q_u8(dst + 4 * stride + 0, vextq_u8(col0_even, d0_lo, 14));
  vst1q_u8(dst + 4 * stride + 16, vextq_u8(d0_lo, d0_hi, 14));
  vst1q_u8(dst + 5 * stride + 0, vextq_u8(col0_odd, d1_lo, 14));
  vst1q_u8(dst + 5 * stride + 16, vextq_u8(d1_lo, d1_hi, 14));
  vst1q_u8(dst + 6 * stride + 0, vextq_u8(col0_even, d0_lo, 13));
  vst1q_u8(dst + 6 * stride + 16, vextq_u8(d0_lo, d0_hi, 13));
  vst1q_u8(dst + 7 * stride + 0, vextq_u8(col0_odd, d1_lo, 13));
  vst1q_u8(dst + 7 * stride + 16, vextq_u8(d1_lo, d1_hi, 13));
  vst1q_u8(dst + 8 * stride + 0, vextq_u8(col0_even, d0_lo, 12));
  vst1q_u8(dst + 8 * stride + 16, vextq_u8(d0_lo, d0_hi, 12));
  vst1q_u8(dst + 9 * stride + 0, vextq_u8(col0_odd, d1_lo, 12));
  vst1q_u8(dst + 9 * stride + 16, vextq_u8(d1_lo, d1_hi, 12));
  vst1q_u8(dst + 10 * stride + 0, vextq_u8(col0_even, d0_lo, 11));
  vst1q_u8(dst + 10 * stride + 16, vextq_u8(d0_lo, d0_hi, 11));
  vst1q_u8(dst + 11 * stride + 0, vextq_u8(col0_odd, d1_lo, 11));
  vst1q_u8(dst + 11 * stride + 16, vextq_u8(d1_lo, d1_hi, 11));
  vst1q_u8(dst + 12 * stride + 0, vextq_u8(col0_even, d0_lo, 10));
  vst1q_u8(dst + 12 * stride + 16, vextq_u8(d0_lo, d0_hi, 10));
  vst1q_u8(dst + 13 * stride + 0, vextq_u8(col0_odd, d1_lo, 10));
  vst1q_u8(dst + 13 * stride + 16, vextq_u8(d1_lo, d1_hi, 10));
  vst1q_u8(dst + 14 * stride + 0, vextq_u8(col0_even, d0_lo, 9));
  vst1q_u8(dst + 14 * stride + 16, vextq_u8(d0_lo, d0_hi, 9));
  vst1q_u8(dst + 15 * stride + 0, vextq_u8(col0_odd, d1_lo, 9));
  vst1q_u8(dst + 15 * stride + 16, vextq_u8(d1_lo, d1_hi, 9));
  vst1q_u8(dst + 16 * stride + 0, vextq_u8(col0_even, d0_lo, 8));
  vst1q_u8(dst + 16 * stride + 16, vextq_u8(d0_lo, d0_hi, 8));
  vst1q_u8(dst + 17 * stride + 0, vextq_u8(col0_odd, d1_lo, 8));
  vst1q_u8(dst + 17 * stride + 16, vextq_u8(d1_lo, d1_hi, 8));
  vst1q_u8(dst + 18 * stride + 0, vextq_u8(col0_even, d0_lo, 7));
  vst1q_u8(dst + 18 * stride + 16, vextq_u8(d0_lo, d0_hi, 7));
  vst1q_u8(dst + 19 * stride + 0, vextq_u8(col0_odd, d1_lo, 7));
  vst1q_u8(dst + 19 * stride + 16, vextq_u8(d1_lo, d1_hi, 7));
  vst1q_u8(dst + 20 * stride + 0, vextq_u8(col0_even, d0_lo, 6));
  vst1q_u8(dst + 20 * stride + 16, vextq_u8(d0_lo, d0_hi, 6));
  vst1q_u8(dst + 21 * stride + 0, vextq_u8(col0_odd, d1_lo, 6));
  vst1q_u8(dst + 21 * stride + 16, vextq_u8(d1_lo, d1_hi, 6));
  vst1q_u8(dst + 22 * stride + 0, vextq_u8(col0_even, d0_lo, 5));
  vst1q_u8(dst + 22 * stride + 16, vextq_u8(d0_lo, d0_hi, 5));
  vst1q_u8(dst + 23 * stride + 0, vextq_u8(col0_odd, d1_lo, 5));
  vst1q_u8(dst + 23 * stride + 16, vextq_u8(d1_lo, d1_hi, 5));
  vst1q_u8(dst + 24 * stride + 0, vextq_u8(col0_even, d0_lo, 4));
  vst1q_u8(dst + 24 * stride + 16, vextq_u8(d0_lo, d0_hi, 4));
  vst1q_u8(dst + 25 * stride + 0, vextq_u8(col0_odd, d1_lo, 4));
  vst1q_u8(dst + 25 * stride + 16, vextq_u8(d1_lo, d1_hi, 4));
  vst1q_u8(dst + 26 * stride + 0, vextq_u8(col0_even, d0_lo, 3));
  vst1q_u8(dst + 26 * stride + 16, vextq_u8(d0_lo, d0_hi, 3));
  vst1q_u8(dst + 27 * stride + 0, vextq_u8(col0_odd, d1_lo, 3));
  vst1q_u8(dst + 27 * stride + 16, vextq_u8(d1_lo, d1_hi, 3));
  vst1q_u8(dst + 28 * stride + 0, vextq_u8(col0_even, d0_lo, 2));
  vst1q_u8(dst + 28 * stride + 16, vextq_u8(d0_lo, d0_hi, 2));
  vst1q_u8(dst + 29 * stride + 0, vextq_u8(col0_odd, d1_lo, 2));
  vst1q_u8(dst + 29 * stride + 16, vextq_u8(d1_lo, d1_hi, 2));
  vst1q_u8(dst + 30 * stride + 0, vextq_u8(col0_even, d0_lo, 1));
  vst1q_u8(dst + 30 * stride + 16, vextq_u8(d0_lo, d0_hi, 1));
  vst1q_u8(dst + 31 * stride + 0, vextq_u8(col0_odd, d1_lo, 1));
  vst1q_u8(dst + 31 * stride + 16, vextq_u8(d1_lo, d1_hi, 1));
}

// -----------------------------------------------------------------------------

void vpx_d135_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x8_t XA0123 = vld1_u8(above - 1);
  const uint8x8_t L0123 = vld1_u8(left);
  const uint8x8_t L3210 = vrev64_u8(L0123);
  const uint8x8_t L3210XA012 = vext_u8(L3210, XA0123, 4);
  const uint8x8_t L210XA0123 = vext_u8(L3210, XA0123, 5);
  const uint8x8_t L10XA0123_ = vext_u8(L210XA0123, L210XA0123, 1);
  const uint8x8_t avg1 = vhadd_u8(L10XA0123_, L3210XA012);
  const uint8x8_t avg2 = vrhadd_u8(avg1, L210XA0123);

  store_u8_4x1(dst + 0 * stride, vext_u8(avg2, avg2, 3));
  store_u8_4x1(dst + 1 * stride, vext_u8(avg2, avg2, 2));
  store_u8_4x1(dst + 2 * stride, vext_u8(avg2, avg2, 1));
  store_u8_4x1(dst + 3 * stride, avg2);
}

void vpx_d135_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x8_t XA0123456 = vld1_u8(above - 1);
  const uint8x8_t A01234567 = vld1_u8(above);
  const uint8x8_t A1234567_ = vld1_u8(above + 1);
  const uint8x8_t L01234567 = vld1_u8(left);
  const uint8x8_t L76543210 = vrev64_u8(L01234567);
  const uint8x8_t L6543210X = vext_u8(L76543210, XA0123456, 1);
  const uint8x8_t L543210XA0 = vext_u8(L76543210, XA0123456, 2);
  const uint8x16_t L76543210XA0123456 = vcombine_u8(L76543210, XA0123456);
  const uint8x16_t L6543210XA01234567 = vcombine_u8(L6543210X, A01234567);
  const uint8x16_t L543210XA01234567_ = vcombine_u8(L543210XA0, A1234567_);
  const uint8x16_t avg = vhaddq_u8(L76543210XA0123456, L543210XA01234567_);
  const uint8x16_t row = vrhaddq_u8(avg, L6543210XA01234567);

  vst1_u8(dst + 0 * stride, vget_low_u8(vextq_u8(row, row, 7)));
  vst1_u8(dst + 1 * stride, vget_low_u8(vextq_u8(row, row, 6)));
  vst1_u8(dst + 2 * stride, vget_low_u8(vextq_u8(row, row, 5)));
  vst1_u8(dst + 3 * stride, vget_low_u8(vextq_u8(row, row, 4)));
  vst1_u8(dst + 4 * stride, vget_low_u8(vextq_u8(row, row, 3)));
  vst1_u8(dst + 5 * stride, vget_low_u8(vextq_u8(row, row, 2)));
  vst1_u8(dst + 6 * stride, vget_low_u8(vextq_u8(row, row, 1)));
  vst1_u8(dst + 7 * stride, vget_low_u8(row));
}

static INLINE void d135_store_16x8(
    uint8_t **dst, const ptrdiff_t stride, const uint8x16_t row_0,
    const uint8x16_t row_1, const uint8x16_t row_2, const uint8x16_t row_3,
    const uint8x16_t row_4, const uint8x16_t row_5, const uint8x16_t row_6,
    const uint8x16_t row_7) {
  vst1q_u8(*dst, row_0);
  *dst += stride;
  vst1q_u8(*dst, row_1);
  *dst += stride;
  vst1q_u8(*dst, row_2);
  *dst += stride;
  vst1q_u8(*dst, row_3);
  *dst += stride;
  vst1q_u8(*dst, row_4);
  *dst += stride;
  vst1q_u8(*dst, row_5);
  *dst += stride;
  vst1q_u8(*dst, row_6);
  *dst += stride;
  vst1q_u8(*dst, row_7);
  *dst += stride;
}

void vpx_d135_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  const uint8x16_t XA0123456789abcde = vld1q_u8(above - 1);
  const uint8x16_t A0123456789abcdef = vld1q_u8(above);
  const uint8x16_t A123456789abcdef_ = vld1q_u8(above + 1);
  const uint8x16_t L0123456789abcdef = vld1q_u8(left);
  const uint8x8_t L76543210 = vrev64_u8(vget_low_u8(L0123456789abcdef));
  const uint8x8_t Lfedcba98 = vrev64_u8(vget_high_u8(L0123456789abcdef));
  const uint8x16_t Lfedcba9876543210 = vcombine_u8(Lfedcba98, L76543210);
  const uint8x16_t Ledcba9876543210X =
      vextq_u8(Lfedcba9876543210, XA0123456789abcde, 1);
  const uint8x16_t Ldcba9876543210XA0 =
      vextq_u8(Lfedcba9876543210, XA0123456789abcde, 2);
  const uint8x16_t avg_0 = vhaddq_u8(Lfedcba9876543210, Ldcba9876543210XA0);
  const uint8x16_t avg_1 = vhaddq_u8(XA0123456789abcde, A123456789abcdef_);
  const uint8x16_t row_0 = vrhaddq_u8(avg_0, Ledcba9876543210X);
  const uint8x16_t row_1 = vrhaddq_u8(avg_1, A0123456789abcdef);

  const uint8x16_t r_0 = vextq_u8(row_0, row_1, 15);
  const uint8x16_t r_1 = vextq_u8(row_0, row_1, 14);
  const uint8x16_t r_2 = vextq_u8(row_0, row_1, 13);
  const uint8x16_t r_3 = vextq_u8(row_0, row_1, 12);
  const uint8x16_t r_4 = vextq_u8(row_0, row_1, 11);
  const uint8x16_t r_5 = vextq_u8(row_0, row_1, 10);
  const uint8x16_t r_6 = vextq_u8(row_0, row_1, 9);
  const uint8x16_t r_7 = vextq_u8(row_0, row_1, 8);
  const uint8x16_t r_8 = vextq_u8(row_0, row_1, 7);
  const uint8x16_t r_9 = vextq_u8(row_0, row_1, 6);
  const uint8x16_t r_a = vextq_u8(row_0, row_1, 5);
  const uint8x16_t r_b = vextq_u8(row_0, row_1, 4);
  const uint8x16_t r_c = vextq_u8(row_0, row_1, 3);
  const uint8x16_t r_d = vextq_u8(row_0, row_1, 2);
  const uint8x16_t r_e = vextq_u8(row_0, row_1, 1);

  d135_store_16x8(&dst, stride, r_0, r_1, r_2, r_3, r_4, r_5, r_6, r_7);
  d135_store_16x8(&dst, stride, r_8, r_9, r_a, r_b, r_c, r_d, r_e, row_0);
}

static INLINE void d135_store_32x2(uint8_t **dst, const ptrdiff_t stride,
                                   const uint8x16_t row_0,
                                   const uint8x16_t row_1,
                                   const uint8x16_t row_2) {
  uint8_t *dst2 = *dst;
  vst1q_u8(dst2, row_1);
  dst2 += 16;
  vst1q_u8(dst2, row_2);
  dst2 += 16 * stride - 16;
  vst1q_u8(dst2, row_0);
  dst2 += 16;
  vst1q_u8(dst2, row_1);
  *dst += stride;
}

void vpx_d135_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  const uint8x16_t LL0123456789abcdef = vld1q_u8(left + 16);
  const uint8x16_t LU0123456789abcdef = vld1q_u8(left);
  const uint8x8_t LL76543210 = vrev64_u8(vget_low_u8(LL0123456789abcdef));
  const uint8x8_t LU76543210 = vrev64_u8(vget_low_u8(LU0123456789abcdef));
  const uint8x8_t LLfedcba98 = vrev64_u8(vget_high_u8(LL0123456789abcdef));
  const uint8x8_t LUfedcba98 = vrev64_u8(vget_high_u8(LU0123456789abcdef));
  const uint8x16_t LLfedcba9876543210 = vcombine_u8(LLfedcba98, LL76543210);
  const uint8x16_t LUfedcba9876543210 = vcombine_u8(LUfedcba98, LU76543210);
  const uint8x16_t LLedcba9876543210Uf =
      vextq_u8(LLfedcba9876543210, LUfedcba9876543210, 1);
  const uint8x16_t LLdcba9876543210Ufe =
      vextq_u8(LLfedcba9876543210, LUfedcba9876543210, 2);
  const uint8x16_t avg_0 = vhaddq_u8(LLfedcba9876543210, LLdcba9876543210Ufe);
  const uint8x16_t row_0 = vrhaddq_u8(avg_0, LLedcba9876543210Uf);

  const uint8x16_t XAL0123456789abcde = vld1q_u8(above - 1);
  const uint8x16_t LUedcba9876543210X =
      vextq_u8(LUfedcba9876543210, XAL0123456789abcde, 1);
  const uint8x16_t LUdcba9876543210XA0 =
      vextq_u8(LUfedcba9876543210, XAL0123456789abcde, 2);
  const uint8x16_t avg_1 = vhaddq_u8(LUfedcba9876543210, LUdcba9876543210XA0);
  const uint8x16_t row_1 = vrhaddq_u8(avg_1, LUedcba9876543210X);

  const uint8x16_t AL0123456789abcdef = vld1q_u8(above);
  const uint8x16_t AL123456789abcdefg = vld1q_u8(above + 1);
  const uint8x16_t ALfR0123456789abcde = vld1q_u8(above + 15);
  const uint8x16_t AR0123456789abcdef = vld1q_u8(above + 16);
  const uint8x16_t AR123456789abcdef_ = vld1q_u8(above + 17);
  const uint8x16_t avg_2 = vhaddq_u8(XAL0123456789abcde, AL123456789abcdefg);
  const uint8x16_t row_2 = vrhaddq_u8(avg_2, AL0123456789abcdef);
  const uint8x16_t avg_3 = vhaddq_u8(ALfR0123456789abcde, AR123456789abcdef_);
  const uint8x16_t row_3 = vrhaddq_u8(avg_3, AR0123456789abcdef);

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 15);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 15);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 15);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 14);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 14);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 14);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 13);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 13);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 13);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 12);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 12);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 12);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 11);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 11);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 11);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 10);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 10);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 10);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 9);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 9);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 9);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 8);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 8);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 8);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 7);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 7);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 7);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 6);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 6);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 6);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 5);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 5);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 5);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 4);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 4);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 4);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 3);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 3);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 3);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 2);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 2);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 2);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  {
    const uint8x16_t r_0 = vextq_u8(row_0, row_1, 1);
    const uint8x16_t r_1 = vextq_u8(row_1, row_2, 1);
    const uint8x16_t r_2 = vextq_u8(row_2, row_3, 1);
    d135_store_32x2(&dst, stride, r_0, r_1, r_2);
  }

  d135_store_32x2(&dst, stride, row_0, row_1, row_2);
}

// -----------------------------------------------------------------------------

void vpx_d153_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  // See vpx_d153_predictor_8x8_neon for more details on the implementation.
  uint8x8_t az, a0, l0az, l0, l1, azl0, d0, d1, d2, d02;

  az = load_unaligned_u8_4x1(above - 1);
  a0 = load_unaligned_u8_4x1(above + 0);
  // [ left[0], above[-1], above[0], above[1], x, x, x, x ]
  l0az = vext_u8(vld1_dup_u8(left), az, 7);

  l0 = load_unaligned_u8_4x1(left + 0);
  l1 = load_unaligned_u8_4x1(left + 1);
  // [ above[-1], left[0], left[1], left[2], x, x, x, x ]
  azl0 = vext_u8(vld1_dup_u8(above - 1), l0, 7);

  d0 = vrhadd_u8(azl0, l0);
  d1 = vrhadd_u8(vhadd_u8(l0az, a0), az);
  d2 = vrhadd_u8(vhadd_u8(azl0, l1), l0);

  d02 = vrev64_u8(vzip_u8(d0, d2).val[0]);

  store_u8_4x1(dst + 0 * stride, vext_u8(d02, d1, 7));
  store_u8_4x1(dst + 1 * stride, vext_u8(d02, d1, 5));
  store_u8_4x1(dst + 2 * stride, vext_u8(d02, d1, 3));
  store_u8_4x1(dst + 3 * stride, vext_u8(d02, d1, 1));
}

void vpx_d153_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  uint8x8_t az, a0, l0az, l0, l1, azl0, d0, d1, d2, d02_lo, d02_hi;

  az = vld1_u8(above - 1);
  a0 = vld1_u8(above + 0);
  // [ left[0], above[-1], ... , above[5] ]
  l0az = vext_u8(vld1_dup_u8(left), az, 7);

  l0 = vld1_u8(left);
  // The last lane here is unused, reading left[8] could cause a buffer
  // over-read, so just fill with a duplicate of left[0] to avoid needing to
  // materialize a zero:
  // [ left[1], ... , left[7], x ]
  l1 = vext_u8(l0, l0, 1);
  // [ above[-1], left[0], ... , left[6] ]
  azl0 = vext_u8(vld1_dup_u8(above - 1), l0, 7);

  // d0[0] = AVG2(above[-1], left[0])
  // d0[1] = AVG2(left[0], left[1])
  // ...
  // d0[7] = AVG2(left[6], left[7])
  d0 = vrhadd_u8(azl0, l0);

  // d1[0] = AVG3(left[0], above[-1], above[0])
  // d1[1] = AVG3(above[-1], above[0], above[1])
  // ...
  // d1[7] = AVG3(above[5], above[6], above[7])
  d1 = vrhadd_u8(vhadd_u8(l0az, a0), az);

  // d2[0] = AVG3(above[-1], left[0], left[1])
  // d2[1] = AVG3(left[0], left[1], left[2])
  // ...
  // d2[6] = AVG3(left[5], left[6], left[7])
  // d2[7] = x (don't care)
  d2 = vrhadd_u8(vhadd_u8(azl0, l1), l0);

  // The ext instruction shifts elements in from the end of the vector rather
  // than the start, so reverse the vectors to put the elements to be shifted
  // in at the end. The lowest lane of d02_lo is unused.
  d02_lo = vzip_u8(vrev64_u8(d2), vrev64_u8(d0)).val[0];
  d02_hi = vzip_u8(vrev64_u8(d2), vrev64_u8(d0)).val[1];

  // Incrementally shift more elements from d0/d2 reversed into d1:
  // stride=0 [ d0[0], d1[0], d1[1], d1[2], d1[3], d1[4], d1[5], d1[6] ]
  // stride=1 [ d0[1], d2[0], d0[0], d1[0], d1[1], d1[2], d1[3], d1[4] ]
  // stride=2 [ d0[2], d2[1], d0[1], d2[0], d0[0], d1[0], d1[1], d1[2] ]
  // stride=3 [ d0[3], d2[2], d0[2], d2[1], d0[1], d2[0], d0[0], d1[0] ]
  // stride=4 [ d0[4], d2[3], d0[3], d2[2], d0[2], d2[1], d0[1], d2[0] ]
  // stride=5 [ d0[5], d2[4], d0[4], d2[3], d0[3], d2[2], d0[2], d2[1] ]
  // stride=6 [ d0[6], d2[5], d0[5], d2[4], d0[4], d2[3], d0[3], d2[2] ]
  // stride=7 [ d0[7], d2[6], d0[6], d2[5], d0[5], d2[4], d0[4], d2[3] ]
  vst1_u8(dst + 0 * stride, vext_u8(d02_hi, d1, 7));
  vst1_u8(dst + 1 * stride, vext_u8(d02_hi, d1, 5));
  vst1_u8(dst + 2 * stride, vext_u8(d02_hi, d1, 3));
  vst1_u8(dst + 3 * stride, vext_u8(d02_hi, d1, 1));
  vst1_u8(dst + 4 * stride, vext_u8(d02_lo, d02_hi, 7));
  vst1_u8(dst + 5 * stride, vext_u8(d02_lo, d02_hi, 5));
  vst1_u8(dst + 6 * stride, vext_u8(d02_lo, d02_hi, 3));
  vst1_u8(dst + 7 * stride, vext_u8(d02_lo, d02_hi, 1));
}

void vpx_d153_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  // See vpx_d153_predictor_8x8_neon for more details on the implementation.
  uint8x16_t az, a0, l0az, l0, l1, azl0, d0, d1, d2, d02_lo, d02_hi;

  az = vld1q_u8(above - 1);
  a0 = vld1q_u8(above + 0);
  // [ left[0], above[-1], ... , above[13] ]
  l0az = vextq_u8(vld1q_dup_u8(left), az, 15);

  l0 = vld1q_u8(left + 0);
  // The last lane here is unused, reading left[16] could cause a buffer
  // over-read, so just fill with a duplicate of left[0] to avoid needing to
  // materialize a zero:
  // [ left[1], ... , left[15], x ]
  l1 = vextq_u8(l0, l0, 1);
  // [ above[-1], left[0], ... , left[14] ]
  azl0 = vextq_u8(vld1q_dup_u8(above - 1), l0, 15);

  d0 = vrhaddq_u8(azl0, l0);
  d1 = vrhaddq_u8(vhaddq_u8(l0az, a0), az);
  d2 = vrhaddq_u8(vhaddq_u8(azl0, l1), l0);

  d0 = vrev64q_u8(vextq_u8(d0, d0, 8));
  d2 = vrev64q_u8(vextq_u8(d2, d2, 8));

  // The lowest lane of d02_lo is unused.
  d02_lo = vzipq_u8(d2, d0).val[0];
  d02_hi = vzipq_u8(d2, d0).val[1];

  vst1q_u8(dst + 0 * stride, vextq_u8(d02_hi, d1, 15));
  vst1q_u8(dst + 1 * stride, vextq_u8(d02_hi, d1, 13));
  vst1q_u8(dst + 2 * stride, vextq_u8(d02_hi, d1, 11));
  vst1q_u8(dst + 3 * stride, vextq_u8(d02_hi, d1, 9));
  vst1q_u8(dst + 4 * stride, vextq_u8(d02_hi, d1, 7));
  vst1q_u8(dst + 5 * stride, vextq_u8(d02_hi, d1, 5));
  vst1q_u8(dst + 6 * stride, vextq_u8(d02_hi, d1, 3));
  vst1q_u8(dst + 7 * stride, vextq_u8(d02_hi, d1, 1));
  vst1q_u8(dst + 8 * stride, vextq_u8(d02_lo, d02_hi, 15));
  vst1q_u8(dst + 9 * stride, vextq_u8(d02_lo, d02_hi, 13));
  vst1q_u8(dst + 10 * stride, vextq_u8(d02_lo, d02_hi, 11));
  vst1q_u8(dst + 11 * stride, vextq_u8(d02_lo, d02_hi, 9));
  vst1q_u8(dst + 12 * stride, vextq_u8(d02_lo, d02_hi, 7));
  vst1q_u8(dst + 13 * stride, vextq_u8(d02_lo, d02_hi, 5));
  vst1q_u8(dst + 14 * stride, vextq_u8(d02_lo, d02_hi, 3));
  vst1q_u8(dst + 15 * stride, vextq_u8(d02_lo, d02_hi, 1));
}

void vpx_d153_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  // See vpx_d153_predictor_8x8_neon for more details on the implementation.
  uint8x16_t az, a0, a14, a15, a16, l0az, l0, l1, l15, l16, l17, azl0, d0_lo,
      d0_hi, d1_lo, d1_hi, d2_lo, d2_hi;
  uint8x16x2_t d02_hi, d02_lo;

  az = vld1q_u8(above - 1);
  a0 = vld1q_u8(above + 0);
  a14 = vld1q_u8(above + 14);
  a15 = vld1q_u8(above + 15);
  a16 = vld1q_u8(above + 16);
  // [ left[0], above[-1], ... , above[13] ]
  l0az = vextq_u8(vld1q_dup_u8(left), az, 15);

  l0 = vld1q_u8(left);
  l1 = vld1q_u8(left + 1);
  l15 = vld1q_u8(left + 15);
  l16 = vld1q_u8(left + 16);
  // The last lane here is unused, reading left[32] would cause a buffer
  // over-read (observed as an address-sanitizer failure), so just fill with a
  // duplicate of left[16] to avoid needing to materialize a zero:
  // [ left[17], ... , left[31], x ]
  l17 = vextq_u8(l16, l16, 1);
  // [ above[-1], left[0], ... , left[14] ]
  azl0 = vextq_u8(vld1q_dup_u8(above - 1), l0, 15);

  d0_lo = vrhaddq_u8(azl0, l0);
  d0_hi = vrhaddq_u8(l15, l16);

  d1_lo = vrhaddq_u8(vhaddq_u8(l0az, a0), az);
  d1_hi = vrhaddq_u8(vhaddq_u8(a14, a16), a15);

  // The highest lane of d2_hi is unused.
  d2_lo = vrhaddq_u8(vhaddq_u8(azl0, l1), l0);
  d2_hi = vrhaddq_u8(vhaddq_u8(l15, l17), l16);

  d0_lo = vrev64q_u8(vextq_u8(d0_lo, d0_lo, 8));
  d0_hi = vrev64q_u8(vextq_u8(d0_hi, d0_hi, 8));

  d2_lo = vrev64q_u8(vextq_u8(d2_lo, d2_lo, 8));
  d2_hi = vrev64q_u8(vextq_u8(d2_hi, d2_hi, 8));

  // d02_hi.val[0][0] is unused here.
  d02_hi = vzipq_u8(d2_hi, d0_hi);
  d02_lo = vzipq_u8(d2_lo, d0_lo);

  vst1q_u8(dst + 0 * stride + 0, vextq_u8(d02_lo.val[1], d1_lo, 15));
  vst1q_u8(dst + 0 * stride + 16, vextq_u8(d1_lo, d1_hi, 15));
  vst1q_u8(dst + 1 * stride + 0, vextq_u8(d02_lo.val[1], d1_lo, 13));
  vst1q_u8(dst + 1 * stride + 16, vextq_u8(d1_lo, d1_hi, 13));
  vst1q_u8(dst + 2 * stride + 0, vextq_u8(d02_lo.val[1], d1_lo, 11));
  vst1q_u8(dst + 2 * stride + 16, vextq_u8(d1_lo, d1_hi, 11));
  vst1q_u8(dst + 3 * stride + 0, vextq_u8(d02_lo.val[1], d1_lo, 9));
  vst1q_u8(dst + 3 * stride + 16, vextq_u8(d1_lo, d1_hi, 9));
  vst1q_u8(dst + 4 * stride + 0, vextq_u8(d02_lo.val[1], d1_lo, 7));
  vst1q_u8(dst + 4 * stride + 16, vextq_u8(d1_lo, d1_hi, 7));
  vst1q_u8(dst + 5 * stride + 0, vextq_u8(d02_lo.val[1], d1_lo, 5));
  vst1q_u8(dst + 5 * stride + 16, vextq_u8(d1_lo, d1_hi, 5));
  vst1q_u8(dst + 6 * stride + 0, vextq_u8(d02_lo.val[1], d1_lo, 3));
  vst1q_u8(dst + 6 * stride + 16, vextq_u8(d1_lo, d1_hi, 3));
  vst1q_u8(dst + 7 * stride + 0, vextq_u8(d02_lo.val[1], d1_lo, 1));
  vst1q_u8(dst + 7 * stride + 16, vextq_u8(d1_lo, d1_hi, 1));
  vst1q_u8(dst + 8 * stride + 0, vextq_u8(d02_lo.val[0], d02_lo.val[1], 15));
  vst1q_u8(dst + 8 * stride + 16, vextq_u8(d02_lo.val[1], d1_lo, 15));
  vst1q_u8(dst + 9 * stride + 0, vextq_u8(d02_lo.val[0], d02_lo.val[1], 13));
  vst1q_u8(dst + 9 * stride + 16, vextq_u8(d02_lo.val[1], d1_lo, 13));
  vst1q_u8(dst + 10 * stride + 0, vextq_u8(d02_lo.val[0], d02_lo.val[1], 11));
  vst1q_u8(dst + 10 * stride + 16, vextq_u8(d02_lo.val[1], d1_lo, 11));
  vst1q_u8(dst + 11 * stride + 0, vextq_u8(d02_lo.val[0], d02_lo.val[1], 9));
  vst1q_u8(dst + 11 * stride + 16, vextq_u8(d02_lo.val[1], d1_lo, 9));
  vst1q_u8(dst + 12 * stride + 0, vextq_u8(d02_lo.val[0], d02_lo.val[1], 7));
  vst1q_u8(dst + 12 * stride + 16, vextq_u8(d02_lo.val[1], d1_lo, 7));
  vst1q_u8(dst + 13 * stride + 0, vextq_u8(d02_lo.val[0], d02_lo.val[1], 5));
  vst1q_u8(dst + 13 * stride + 16, vextq_u8(d02_lo.val[1], d1_lo, 5));
  vst1q_u8(dst + 14 * stride + 0, vextq_u8(d02_lo.val[0], d02_lo.val[1], 3));
  vst1q_u8(dst + 14 * stride + 16, vextq_u8(d02_lo.val[1], d1_lo, 3));
  vst1q_u8(dst + 15 * stride + 0, vextq_u8(d02_lo.val[0], d02_lo.val[1], 1));
  vst1q_u8(dst + 15 * stride + 16, vextq_u8(d02_lo.val[1], d1_lo, 1));
  vst1q_u8(dst + 16 * stride + 0, vextq_u8(d02_hi.val[1], d02_lo.val[0], 15));
  vst1q_u8(dst + 16 * stride + 16, vextq_u8(d02_lo.val[0], d02_lo.val[1], 15));
  vst1q_u8(dst + 17 * stride + 0, vextq_u8(d02_hi.val[1], d02_lo.val[0], 13));
  vst1q_u8(dst + 17 * stride + 16, vextq_u8(d02_lo.val[0], d02_lo.val[1], 13));
  vst1q_u8(dst + 18 * stride + 0, vextq_u8(d02_hi.val[1], d02_lo.val[0], 11));
  vst1q_u8(dst + 18 * stride + 16, vextq_u8(d02_lo.val[0], d02_lo.val[1], 11));
  vst1q_u8(dst + 19 * stride + 0, vextq_u8(d02_hi.val[1], d02_lo.val[0], 9));
  vst1q_u8(dst + 19 * stride + 16, vextq_u8(d02_lo.val[0], d02_lo.val[1], 9));
  vst1q_u8(dst + 20 * stride + 0, vextq_u8(d02_hi.val[1], d02_lo.val[0], 7));
  vst1q_u8(dst + 20 * stride + 16, vextq_u8(d02_lo.val[0], d02_lo.val[1], 7));
  vst1q_u8(dst + 21 * stride + 0, vextq_u8(d02_hi.val[1], d02_lo.val[0], 5));
  vst1q_u8(dst + 21 * stride + 16, vextq_u8(d02_lo.val[0], d02_lo.val[1], 5));
  vst1q_u8(dst + 22 * stride + 0, vextq_u8(d02_hi.val[1], d02_lo.val[0], 3));
  vst1q_u8(dst + 22 * stride + 16, vextq_u8(d02_lo.val[0], d02_lo.val[1], 3));
  vst1q_u8(dst + 23 * stride + 0, vextq_u8(d02_hi.val[1], d02_lo.val[0], 1));
  vst1q_u8(dst + 23 * stride + 16, vextq_u8(d02_lo.val[0], d02_lo.val[1], 1));
  vst1q_u8(dst + 24 * stride + 0, vextq_u8(d02_hi.val[0], d02_hi.val[1], 15));
  vst1q_u8(dst + 24 * stride + 16, vextq_u8(d02_hi.val[1], d02_lo.val[0], 15));
  vst1q_u8(dst + 25 * stride + 0, vextq_u8(d02_hi.val[0], d02_hi.val[1], 13));
  vst1q_u8(dst + 25 * stride + 16, vextq_u8(d02_hi.val[1], d02_lo.val[0], 13));
  vst1q_u8(dst + 26 * stride + 0, vextq_u8(d02_hi.val[0], d02_hi.val[1], 11));
  vst1q_u8(dst + 26 * stride + 16, vextq_u8(d02_hi.val[1], d02_lo.val[0], 11));
  vst1q_u8(dst + 27 * stride + 0, vextq_u8(d02_hi.val[0], d02_hi.val[1], 9));
  vst1q_u8(dst + 27 * stride + 16, vextq_u8(d02_hi.val[1], d02_lo.val[0], 9));
  vst1q_u8(dst + 28 * stride + 0, vextq_u8(d02_hi.val[0], d02_hi.val[1], 7));
  vst1q_u8(dst + 28 * stride + 16, vextq_u8(d02_hi.val[1], d02_lo.val[0], 7));
  vst1q_u8(dst + 29 * stride + 0, vextq_u8(d02_hi.val[0], d02_hi.val[1], 5));
  vst1q_u8(dst + 29 * stride + 16, vextq_u8(d02_hi.val[1], d02_lo.val[0], 5));
  vst1q_u8(dst + 30 * stride + 0, vextq_u8(d02_hi.val[0], d02_hi.val[1], 3));
  vst1q_u8(dst + 30 * stride + 16, vextq_u8(d02_hi.val[1], d02_lo.val[0], 3));
  vst1q_u8(dst + 31 * stride + 0, vextq_u8(d02_hi.val[0], d02_hi.val[1], 1));
  vst1q_u8(dst + 31 * stride + 16, vextq_u8(d02_hi.val[1], d02_lo.val[0], 1));
}

// -----------------------------------------------------------------------------

void vpx_d207_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  uint8x8_t l0, l3, l1, l2, c0, c1, c01, d0, d1;
  (void)above;

  // We need the low half lanes here for the c0/c1 arithmetic but the high half
  // lanes for the ext:
  // [ left[0], left[1], left[2], left[3], left[0], left[1], left[2], left[3] ]
  l0 = load_replicate_u8_4x1(left + 0);
  l3 = vld1_dup_u8(left + 3);

  // [ left[1], left[2], left[3], left[3], x, x, x, x ]
  l1 = vext_u8(l0, l3, 5);
  // [ left[2], left[3], left[3], left[3], x, x, x, x ]
  l2 = vext_u8(l0, l3, 6);

  c0 = vrhadd_u8(l0, l1);
  c1 = vrhadd_u8(vhadd_u8(l0, l2), l1);

  // [ c0[0], c1[0], c0[1], c1[1], c0[2], c1[2], c0[3], c1[3] ]
  c01 = vzip_u8(c0, c1).val[0];

  d0 = c01;
  d1 = vext_u8(c01, l3, 2);

  // Store the high half of the vector for stride={2,3} to avoid needing
  // additional ext instructions:
  // stride=0 [ c0[0], c1[0],   c0[1],   c1[1] ]
  // stride=1 [ c0[1], c1[1],   c0[2],   c1[2] ]
  // stride=2 [ c0[2], c1[2],   c0[3],   c1[3] ]
  // stride=3 [ c0[3], c1[3], left[3], left[3] ]
  store_u8_4x1(dst + 0 * stride, d0);
  store_u8_4x1(dst + 1 * stride, d1);
  store_u8_4x1_high(dst + 2 * stride, d0);
  store_u8_4x1_high(dst + 3 * stride, d1);
}

void vpx_d207_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  uint8x8_t l7, l0, l1, l2, c0, c1, c01_lo, c01_hi;
  (void)above;

  l0 = vld1_u8(left + 0);
  l7 = vld1_dup_u8(left + 7);

  // [ left[1], left[2], left[3], left[4], left[5], left[6], left[7], left[7] ]
  l1 = vext_u8(l0, l7, 1);
  // [ left[2], left[3], left[4], left[5], left[6], left[7], left[7], left[7] ]
  l2 = vext_u8(l0, l7, 2);

  c0 = vrhadd_u8(l0, l1);
  c1 = vrhadd_u8(vhadd_u8(l0, l2), l1);

  c01_lo = vzip_u8(c0, c1).val[0];
  c01_hi = vzip_u8(c0, c1).val[1];

  vst1_u8(dst + 0 * stride, c01_lo);
  vst1_u8(dst + 1 * stride, vext_u8(c01_lo, c01_hi, 2));
  vst1_u8(dst + 2 * stride, vext_u8(c01_lo, c01_hi, 4));
  vst1_u8(dst + 3 * stride, vext_u8(c01_lo, c01_hi, 6));
  vst1_u8(dst + 4 * stride, c01_hi);
  vst1_u8(dst + 5 * stride, vext_u8(c01_hi, l7, 2));
  vst1_u8(dst + 6 * stride, vext_u8(c01_hi, l7, 4));
  vst1_u8(dst + 7 * stride, vext_u8(c01_hi, l7, 6));
}

void vpx_d207_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  uint8x16_t l15, l0, l1, l2, c0, c1, c01_lo, c01_hi;
  (void)above;

  l0 = vld1q_u8(left + 0);
  l15 = vld1q_dup_u8(left + 15);

  l1 = vextq_u8(l0, l15, 1);
  l2 = vextq_u8(l0, l15, 2);

  c0 = vrhaddq_u8(l0, l1);
  c1 = vrhaddq_u8(vhaddq_u8(l0, l2), l1);

  c01_lo = vzipq_u8(c0, c1).val[0];
  c01_hi = vzipq_u8(c0, c1).val[1];

  vst1q_u8(dst + 0 * stride, c01_lo);
  vst1q_u8(dst + 1 * stride, vextq_u8(c01_lo, c01_hi, 2));
  vst1q_u8(dst + 2 * stride, vextq_u8(c01_lo, c01_hi, 4));
  vst1q_u8(dst + 3 * stride, vextq_u8(c01_lo, c01_hi, 6));
  vst1q_u8(dst + 4 * stride, vextq_u8(c01_lo, c01_hi, 8));
  vst1q_u8(dst + 5 * stride, vextq_u8(c01_lo, c01_hi, 10));
  vst1q_u8(dst + 6 * stride, vextq_u8(c01_lo, c01_hi, 12));
  vst1q_u8(dst + 7 * stride, vextq_u8(c01_lo, c01_hi, 14));
  vst1q_u8(dst + 8 * stride, c01_hi);
  vst1q_u8(dst + 9 * stride, vextq_u8(c01_hi, l15, 2));
  vst1q_u8(dst + 10 * stride, vextq_u8(c01_hi, l15, 4));
  vst1q_u8(dst + 11 * stride, vextq_u8(c01_hi, l15, 6));
  vst1q_u8(dst + 12 * stride, vextq_u8(c01_hi, l15, 8));
  vst1q_u8(dst + 13 * stride, vextq_u8(c01_hi, l15, 10));
  vst1q_u8(dst + 14 * stride, vextq_u8(c01_hi, l15, 12));
  vst1q_u8(dst + 15 * stride, vextq_u8(c01_hi, l15, 14));
}

void vpx_d207_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  uint8x16_t l0_lo, l0_hi, l1_lo, l1_hi, l2_lo, l2_hi, l31, c0_lo, c0_hi, c1_lo,
      c1_hi, c01[4];
  (void)above;

  l0_lo = vld1q_u8(left + 0);
  l0_hi = vld1q_u8(left + 16);
  l31 = vld1q_dup_u8(left + 31);

  l1_lo = vextq_u8(l0_lo, l0_hi, 1);
  l1_hi = vextq_u8(l0_hi, l31, 1);
  l2_lo = vextq_u8(l0_lo, l0_hi, 2);
  l2_hi = vextq_u8(l0_hi, l31, 2);

  c0_lo = vrhaddq_u8(l0_lo, l1_lo);
  c0_hi = vrhaddq_u8(l0_hi, l1_hi);
  c1_lo = vrhaddq_u8(vhaddq_u8(l0_lo, l2_lo), l1_lo);
  c1_hi = vrhaddq_u8(vhaddq_u8(l0_hi, l2_hi), l1_hi);

  c01[0] = vzipq_u8(c0_lo, c1_lo).val[0];
  c01[1] = vzipq_u8(c0_lo, c1_lo).val[1];
  c01[2] = vzipq_u8(c0_hi, c1_hi).val[0];
  c01[3] = vzipq_u8(c0_hi, c1_hi).val[1];

  vst1q_u8(dst + 0 * stride + 0, c01[0]);
  vst1q_u8(dst + 0 * stride + 16, c01[1]);
  vst1q_u8(dst + 1 * stride + 0, vextq_u8(c01[0], c01[1], 2));
  vst1q_u8(dst + 1 * stride + 16, vextq_u8(c01[1], c01[2], 2));
  vst1q_u8(dst + 2 * stride + 0, vextq_u8(c01[0], c01[1], 4));
  vst1q_u8(dst + 2 * stride + 16, vextq_u8(c01[1], c01[2], 4));
  vst1q_u8(dst + 3 * stride + 0, vextq_u8(c01[0], c01[1], 6));
  vst1q_u8(dst + 3 * stride + 16, vextq_u8(c01[1], c01[2], 6));
  vst1q_u8(dst + 4 * stride + 0, vextq_u8(c01[0], c01[1], 8));
  vst1q_u8(dst + 4 * stride + 16, vextq_u8(c01[1], c01[2], 8));
  vst1q_u8(dst + 5 * stride + 0, vextq_u8(c01[0], c01[1], 10));
  vst1q_u8(dst + 5 * stride + 16, vextq_u8(c01[1], c01[2], 10));
  vst1q_u8(dst + 6 * stride + 0, vextq_u8(c01[0], c01[1], 12));
  vst1q_u8(dst + 6 * stride + 16, vextq_u8(c01[1], c01[2], 12));
  vst1q_u8(dst + 7 * stride + 0, vextq_u8(c01[0], c01[1], 14));
  vst1q_u8(dst + 7 * stride + 16, vextq_u8(c01[1], c01[2], 14));
  vst1q_u8(dst + 8 * stride + 0, c01[1]);
  vst1q_u8(dst + 8 * stride + 16, c01[2]);
  vst1q_u8(dst + 9 * stride + 0, vextq_u8(c01[1], c01[2], 2));
  vst1q_u8(dst + 9 * stride + 16, vextq_u8(c01[2], c01[3], 2));
  vst1q_u8(dst + 10 * stride + 0, vextq_u8(c01[1], c01[2], 4));
  vst1q_u8(dst + 10 * stride + 16, vextq_u8(c01[2], c01[3], 4));
  vst1q_u8(dst + 11 * stride + 0, vextq_u8(c01[1], c01[2], 6));
  vst1q_u8(dst + 11 * stride + 16, vextq_u8(c01[2], c01[3], 6));
  vst1q_u8(dst + 12 * stride + 0, vextq_u8(c01[1], c01[2], 8));
  vst1q_u8(dst + 12 * stride + 16, vextq_u8(c01[2], c01[3], 8));
  vst1q_u8(dst + 13 * stride + 0, vextq_u8(c01[1], c01[2], 10));
  vst1q_u8(dst + 13 * stride + 16, vextq_u8(c01[2], c01[3], 10));
  vst1q_u8(dst + 14 * stride + 0, vextq_u8(c01[1], c01[2], 12));
  vst1q_u8(dst + 14 * stride + 16, vextq_u8(c01[2], c01[3], 12));
  vst1q_u8(dst + 15 * stride + 0, vextq_u8(c01[1], c01[2], 14));
  vst1q_u8(dst + 15 * stride + 16, vextq_u8(c01[2], c01[3], 14));
  vst1q_u8(dst + 16 * stride + 0, c01[2]);
  vst1q_u8(dst + 16 * stride + 16, c01[3]);
  vst1q_u8(dst + 17 * stride + 0, vextq_u8(c01[2], c01[3], 2));
  vst1q_u8(dst + 17 * stride + 16, vextq_u8(c01[3], l31, 2));
  vst1q_u8(dst + 18 * stride + 0, vextq_u8(c01[2], c01[3], 4));
  vst1q_u8(dst + 18 * stride + 16, vextq_u8(c01[3], l31, 4));
  vst1q_u8(dst + 19 * stride + 0, vextq_u8(c01[2], c01[3], 6));
  vst1q_u8(dst + 19 * stride + 16, vextq_u8(c01[3], l31, 6));
  vst1q_u8(dst + 20 * stride + 0, vextq_u8(c01[2], c01[3], 8));
  vst1q_u8(dst + 20 * stride + 16, vextq_u8(c01[3], l31, 8));
  vst1q_u8(dst + 21 * stride + 0, vextq_u8(c01[2], c01[3], 10));
  vst1q_u8(dst + 21 * stride + 16, vextq_u8(c01[3], l31, 10));
  vst1q_u8(dst + 22 * stride + 0, vextq_u8(c01[2], c01[3], 12));
  vst1q_u8(dst + 22 * stride + 16, vextq_u8(c01[3], l31, 12));
  vst1q_u8(dst + 23 * stride + 0, vextq_u8(c01[2], c01[3], 14));
  vst1q_u8(dst + 23 * stride + 16, vextq_u8(c01[3], l31, 14));
  vst1q_u8(dst + 24 * stride + 0, c01[3]);
  vst1q_u8(dst + 24 * stride + 16, l31);
  vst1q_u8(dst + 25 * stride + 0, vextq_u8(c01[3], l31, 2));
  vst1q_u8(dst + 25 * stride + 16, l31);
  vst1q_u8(dst + 26 * stride + 0, vextq_u8(c01[3], l31, 4));
  vst1q_u8(dst + 26 * stride + 16, l31);
  vst1q_u8(dst + 27 * stride + 0, vextq_u8(c01[3], l31, 6));
  vst1q_u8(dst + 27 * stride + 16, l31);
  vst1q_u8(dst + 28 * stride + 0, vextq_u8(c01[3], l31, 8));
  vst1q_u8(dst + 28 * stride + 16, l31);
  vst1q_u8(dst + 29 * stride + 0, vextq_u8(c01[3], l31, 10));
  vst1q_u8(dst + 29 * stride + 16, l31);
  vst1q_u8(dst + 30 * stride + 0, vextq_u8(c01[3], l31, 12));
  vst1q_u8(dst + 30 * stride + 16, l31);
  vst1q_u8(dst + 31 * stride + 0, vextq_u8(c01[3], l31, 14));
  vst1q_u8(dst + 31 * stride + 16, l31);
}

// -----------------------------------------------------------------------------

#if !HAVE_NEON_ASM

void vpx_v_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  const uint32_t d = *(const uint32_t *)above;
  int i;
  (void)left;

  for (i = 0; i < 4; i++, dst += stride) {
    *(uint32_t *)dst = d;
  }
}

void vpx_v_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  const uint8x8_t d = vld1_u8(above);
  int i;
  (void)left;

  for (i = 0; i < 8; i++, dst += stride) {
    vst1_u8(dst, d);
  }
}

void vpx_v_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  const uint8x16_t d = vld1q_u8(above);
  int i;
  (void)left;

  for (i = 0; i < 16; i++, dst += stride) {
    vst1q_u8(dst, d);
  }
}

void vpx_v_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  const uint8x16_t d0 = vld1q_u8(above);
  const uint8x16_t d1 = vld1q_u8(above + 16);
  int i;
  (void)left;

  for (i = 0; i < 32; i++) {
    // Note: performance was worse using vst2q_u8 under gcc-4.9 & clang-3.8.
    // clang-3.8 unrolled the loop fully with no filler so the cause is likely
    // the latency of the instruction.
    vst1q_u8(dst, d0);
    dst += 16;
    vst1q_u8(dst, d1);
    dst += stride - 16;
  }
}

// -----------------------------------------------------------------------------

void vpx_h_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  const uint32x2_t zero = vdup_n_u32(0);
  const uint8x8_t left_u8 =
      vreinterpret_u8_u32(vld1_lane_u32((const uint32_t *)left, zero, 0));
  uint8x8_t d;
  (void)above;

  d = vdup_lane_u8(left_u8, 0);
  vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d), 0);
  dst += stride;
  d = vdup_lane_u8(left_u8, 1);
  vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d), 0);
  dst += stride;
  d = vdup_lane_u8(left_u8, 2);
  vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d), 0);
  dst += stride;
  d = vdup_lane_u8(left_u8, 3);
  vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d), 0);
}

void vpx_h_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  const uint8x8_t left_u8 = vld1_u8(left);
  uint8x8_t d;
  (void)above;

  d = vdup_lane_u8(left_u8, 0);
  vst1_u8(dst, d);
  dst += stride;
  d = vdup_lane_u8(left_u8, 1);
  vst1_u8(dst, d);
  dst += stride;
  d = vdup_lane_u8(left_u8, 2);
  vst1_u8(dst, d);
  dst += stride;
  d = vdup_lane_u8(left_u8, 3);
  vst1_u8(dst, d);
  dst += stride;
  d = vdup_lane_u8(left_u8, 4);
  vst1_u8(dst, d);
  dst += stride;
  d = vdup_lane_u8(left_u8, 5);
  vst1_u8(dst, d);
  dst += stride;
  d = vdup_lane_u8(left_u8, 6);
  vst1_u8(dst, d);
  dst += stride;
  d = vdup_lane_u8(left_u8, 7);
  vst1_u8(dst, d);
}

static INLINE void h_store_16x8(uint8_t **dst, const ptrdiff_t stride,
                                const uint8x8_t left) {
  const uint8x16_t row_0 = vdupq_lane_u8(left, 0);
  const uint8x16_t row_1 = vdupq_lane_u8(left, 1);
  const uint8x16_t row_2 = vdupq_lane_u8(left, 2);
  const uint8x16_t row_3 = vdupq_lane_u8(left, 3);
  const uint8x16_t row_4 = vdupq_lane_u8(left, 4);
  const uint8x16_t row_5 = vdupq_lane_u8(left, 5);
  const uint8x16_t row_6 = vdupq_lane_u8(left, 6);
  const uint8x16_t row_7 = vdupq_lane_u8(left, 7);

  vst1q_u8(*dst, row_0);
  *dst += stride;
  vst1q_u8(*dst, row_1);
  *dst += stride;
  vst1q_u8(*dst, row_2);
  *dst += stride;
  vst1q_u8(*dst, row_3);
  *dst += stride;
  vst1q_u8(*dst, row_4);
  *dst += stride;
  vst1q_u8(*dst, row_5);
  *dst += stride;
  vst1q_u8(*dst, row_6);
  *dst += stride;
  vst1q_u8(*dst, row_7);
  *dst += stride;
}

void vpx_h_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  const uint8x16_t left_u8q = vld1q_u8(left);
  (void)above;

  h_store_16x8(&dst, stride, vget_low_u8(left_u8q));
  h_store_16x8(&dst, stride, vget_high_u8(left_u8q));
}

static INLINE void h_store_32x8(uint8_t **dst, const ptrdiff_t stride,
                                const uint8x8_t left) {
  const uint8x16_t row_0 = vdupq_lane_u8(left, 0);
  const uint8x16_t row_1 = vdupq_lane_u8(left, 1);
  const uint8x16_t row_2 = vdupq_lane_u8(left, 2);
  const uint8x16_t row_3 = vdupq_lane_u8(left, 3);
  const uint8x16_t row_4 = vdupq_lane_u8(left, 4);
  const uint8x16_t row_5 = vdupq_lane_u8(left, 5);
  const uint8x16_t row_6 = vdupq_lane_u8(left, 6);
  const uint8x16_t row_7 = vdupq_lane_u8(left, 7);

  vst1q_u8(*dst, row_0);  // Note clang-3.8 produced poor code w/vst2q_u8
  *dst += 16;
  vst1q_u8(*dst, row_0);
  *dst += stride - 16;
  vst1q_u8(*dst, row_1);
  *dst += 16;
  vst1q_u8(*dst, row_1);
  *dst += stride - 16;
  vst1q_u8(*dst, row_2);
  *dst += 16;
  vst1q_u8(*dst, row_2);
  *dst += stride - 16;
  vst1q_u8(*dst, row_3);
  *dst += 16;
  vst1q_u8(*dst, row_3);
  *dst += stride - 16;
  vst1q_u8(*dst, row_4);
  *dst += 16;
  vst1q_u8(*dst, row_4);
  *dst += stride - 16;
  vst1q_u8(*dst, row_5);
  *dst += 16;
  vst1q_u8(*dst, row_5);
  *dst += stride - 16;
  vst1q_u8(*dst, row_6);
  *dst += 16;
  vst1q_u8(*dst, row_6);
  *dst += stride - 16;
  vst1q_u8(*dst, row_7);
  *dst += 16;
  vst1q_u8(*dst, row_7);
  *dst += stride - 16;
}

void vpx_h_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  int i;
  (void)above;

  for (i = 0; i < 2; i++, left += 16) {
    const uint8x16_t left_u8 = vld1q_u8(left);
    h_store_32x8(&dst, stride, vget_low_u8(left_u8));
    h_store_32x8(&dst, stride, vget_high_u8(left_u8));
  }
}

// -----------------------------------------------------------------------------

static INLINE int16x8_t convert_u8_to_s16(uint8x8_t v) {
  return vreinterpretq_s16_u16(vmovl_u8(v));
}

void vpx_tm_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x8_t top_left = vld1_dup_u8(above - 1);
  const uint8x8_t left_u8 = vld1_u8(left);
  const uint8x8_t above_u8 = vld1_u8(above);
  const int16x4_t left_s16 = vget_low_s16(convert_u8_to_s16(left_u8));
  int16x8_t sub, sum;
  uint32x2_t d;

  sub = vreinterpretq_s16_u16(vsubl_u8(above_u8, top_left));
  // Avoid vcombine_s16() which generates lots of redundant code with clang-3.8.
  sub = vreinterpretq_s16_s64(
      vdupq_lane_s64(vreinterpret_s64_s16(vget_low_s16(sub)), 0));

  sum = vcombine_s16(vdup_lane_s16(left_s16, 0), vdup_lane_s16(left_s16, 1));
  sum = vaddq_s16(sum, sub);
  d = vreinterpret_u32_u8(vqmovun_s16(sum));
  vst1_lane_u32((uint32_t *)dst, d, 0);
  dst += stride;
  vst1_lane_u32((uint32_t *)dst, d, 1);
  dst += stride;

  sum = vcombine_s16(vdup_lane_s16(left_s16, 2), vdup_lane_s16(left_s16, 3));
  sum = vaddq_s16(sum, sub);
  d = vreinterpret_u32_u8(vqmovun_s16(sum));
  vst1_lane_u32((uint32_t *)dst, d, 0);
  dst += stride;
  vst1_lane_u32((uint32_t *)dst, d, 1);
}

static INLINE void tm_8_kernel(uint8_t **dst, const ptrdiff_t stride,
                               const int16x8_t left_dup, const int16x8_t sub) {
  const int16x8_t sum = vaddq_s16(left_dup, sub);
  const uint8x8_t d = vqmovun_s16(sum);
  vst1_u8(*dst, d);
  *dst += stride;
}

void vpx_tm_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x8_t top_left = vld1_dup_u8(above - 1);
  const uint8x8_t above_u8 = vld1_u8(above);
  const uint8x8_t left_u8 = vld1_u8(left);
  const int16x8_t left_s16q = convert_u8_to_s16(left_u8);
  const int16x8_t sub = vreinterpretq_s16_u16(vsubl_u8(above_u8, top_left));
  int16x4_t left_s16d = vget_low_s16(left_s16q);
  int i;

  for (i = 0; i < 2; i++, left_s16d = vget_high_s16(left_s16q)) {
    int16x8_t left_dup;

    left_dup = vdupq_lane_s16(left_s16d, 0);
    tm_8_kernel(&dst, stride, left_dup, sub);
    left_dup = vdupq_lane_s16(left_s16d, 1);
    tm_8_kernel(&dst, stride, left_dup, sub);
    left_dup = vdupq_lane_s16(left_s16d, 2);
    tm_8_kernel(&dst, stride, left_dup, sub);
    left_dup = vdupq_lane_s16(left_s16d, 3);
    tm_8_kernel(&dst, stride, left_dup, sub);
  }
}

static INLINE void tm_16_kernel(uint8_t **dst, const ptrdiff_t stride,
                                const int16x8_t left_dup, const int16x8_t sub0,
                                const int16x8_t sub1) {
  const int16x8_t sum0 = vaddq_s16(left_dup, sub0);
  const int16x8_t sum1 = vaddq_s16(left_dup, sub1);
  const uint8x8_t d0 = vqmovun_s16(sum0);
  const uint8x8_t d1 = vqmovun_s16(sum1);
  vst1_u8(*dst, d0);
  *dst += 8;
  vst1_u8(*dst, d1);
  *dst += stride - 8;
}

void vpx_tm_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x16_t top_left = vld1q_dup_u8(above - 1);
  const uint8x16_t above_u8 = vld1q_u8(above);
  const int16x8_t sub0 = vreinterpretq_s16_u16(
      vsubl_u8(vget_low_u8(above_u8), vget_low_u8(top_left)));
  const int16x8_t sub1 = vreinterpretq_s16_u16(
      vsubl_u8(vget_high_u8(above_u8), vget_high_u8(top_left)));
  int16x8_t left_dup;
  int i;

  for (i = 0; i < 2; i++, left += 8) {
    const uint8x8_t left_u8 = vld1_u8(left);
    const int16x8_t left_s16q = convert_u8_to_s16(left_u8);
    const int16x4_t left_low = vget_low_s16(left_s16q);
    const int16x4_t left_high = vget_high_s16(left_s16q);

    left_dup = vdupq_lane_s16(left_low, 0);
    tm_16_kernel(&dst, stride, left_dup, sub0, sub1);
    left_dup = vdupq_lane_s16(left_low, 1);
    tm_16_kernel(&dst, stride, left_dup, sub0, sub1);
    left_dup = vdupq_lane_s16(left_low, 2);
    tm_16_kernel(&dst, stride, left_dup, sub0, sub1);
    left_dup = vdupq_lane_s16(left_low, 3);
    tm_16_kernel(&dst, stride, left_dup, sub0, sub1);

    left_dup = vdupq_lane_s16(left_high, 0);
    tm_16_kernel(&dst, stride, left_dup, sub0, sub1);
    left_dup = vdupq_lane_s16(left_high, 1);
    tm_16_kernel(&dst, stride, left_dup, sub0, sub1);
    left_dup = vdupq_lane_s16(left_high, 2);
    tm_16_kernel(&dst, stride, left_dup, sub0, sub1);
    left_dup = vdupq_lane_s16(left_high, 3);
    tm_16_kernel(&dst, stride, left_dup, sub0, sub1);
  }
}

static INLINE void tm_32_kernel(uint8_t **dst, const ptrdiff_t stride,
                                const int16x8_t left_dup, const int16x8_t sub0,
                                const int16x8_t sub1, const int16x8_t sub2,
                                const int16x8_t sub3) {
  const int16x8_t sum0 = vaddq_s16(left_dup, sub0);
  const int16x8_t sum1 = vaddq_s16(left_dup, sub1);
  const int16x8_t sum2 = vaddq_s16(left_dup, sub2);
  const int16x8_t sum3 = vaddq_s16(left_dup, sub3);
  const uint8x8_t d0 = vqmovun_s16(sum0);
  const uint8x8_t d1 = vqmovun_s16(sum1);
  const uint8x8_t d2 = vqmovun_s16(sum2);
  const uint8x8_t d3 = vqmovun_s16(sum3);

  vst1q_u8(*dst, vcombine_u8(d0, d1));
  *dst += 16;
  vst1q_u8(*dst, vcombine_u8(d2, d3));
  *dst += stride - 16;
}

void vpx_tm_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x16_t top_left = vld1q_dup_u8(above - 1);
  const uint8x16_t above_low = vld1q_u8(above);
  const uint8x16_t above_high = vld1q_u8(above + 16);
  const int16x8_t sub0 = vreinterpretq_s16_u16(
      vsubl_u8(vget_low_u8(above_low), vget_low_u8(top_left)));
  const int16x8_t sub1 = vreinterpretq_s16_u16(
      vsubl_u8(vget_high_u8(above_low), vget_high_u8(top_left)));
  const int16x8_t sub2 = vreinterpretq_s16_u16(
      vsubl_u8(vget_low_u8(above_high), vget_low_u8(top_left)));
  const int16x8_t sub3 = vreinterpretq_s16_u16(
      vsubl_u8(vget_high_u8(above_high), vget_high_u8(top_left)));
  int16x8_t left_dup;
  int i, j;

  for (j = 0; j < 4; j++, left += 8) {
    const uint8x8_t left_u8 = vld1_u8(left);
    const int16x8_t left_s16q = convert_u8_to_s16(left_u8);
    int16x4_t left_s16d = vget_low_s16(left_s16q);
    for (i = 0; i < 2; i++, left_s16d = vget_high_s16(left_s16q)) {
      left_dup = vdupq_lane_s16(left_s16d, 0);
      tm_32_kernel(&dst, stride, left_dup, sub0, sub1, sub2, sub3);
      left_dup = vdupq_lane_s16(left_s16d, 1);
      tm_32_kernel(&dst, stride, left_dup, sub0, sub1, sub2, sub3);
      left_dup = vdupq_lane_s16(left_s16d, 2);
      tm_32_kernel(&dst, stride, left_dup, sub0, sub1, sub2, sub3);
      left_dup = vdupq_lane_s16(left_s16d, 3);
      tm_32_kernel(&dst, stride, left_dup, sub0, sub1, sub2, sub3);
    }
  }
}
#endif  // !HAVE_NEON_ASM
