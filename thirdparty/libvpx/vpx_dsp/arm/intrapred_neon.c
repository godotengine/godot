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
#include "vpx/vpx_integer.h"

//------------------------------------------------------------------------------
// DC 4x4

// 'do_above' and 'do_left' facilitate branch removal when inlined.
static INLINE void dc_4x4(uint8_t *dst, ptrdiff_t stride,
                          const uint8_t *above, const uint8_t *left,
                          int do_above, int do_left) {
  uint16x8_t sum_top;
  uint16x8_t sum_left;
  uint8x8_t dc0;

  if (do_above) {
    const uint8x8_t A = vld1_u8(above);  // top row
    const uint16x4_t p0 = vpaddl_u8(A);  // cascading summation of the top
    const uint16x4_t p1 = vpadd_u16(p0, p0);
    sum_top = vcombine_u16(p1, p1);
  }

  if (do_left) {
    const uint8x8_t L = vld1_u8(left);  // left border
    const uint16x4_t p0 = vpaddl_u8(L);  // cascading summation of the left
    const uint16x4_t p1 = vpadd_u16(p0, p0);
    sum_left = vcombine_u16(p1, p1);
  }

  if (do_above && do_left) {
    const uint16x8_t sum = vaddq_u16(sum_left, sum_top);
    dc0 = vrshrn_n_u16(sum, 3);
  } else if (do_above) {
    dc0 = vrshrn_n_u16(sum_top, 2);
  } else if (do_left) {
    dc0 = vrshrn_n_u16(sum_left, 2);
  } else {
    dc0 = vdup_n_u8(0x80);
  }

  {
    const uint8x8_t dc = vdup_lane_u8(dc0, 0);
    int i;
    for (i = 0; i < 4; ++i) {
      vst1_lane_u32((uint32_t*)(dst + i * stride), vreinterpret_u32_u8(dc), 0);
    }
  }
}

void vpx_dc_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  dc_4x4(dst, stride, above, left, 1, 1);
}

void vpx_dc_left_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above, const uint8_t *left) {
  (void)above;
  dc_4x4(dst, stride, NULL, left, 0, 1);
}

void vpx_dc_top_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  (void)left;
  dc_4x4(dst, stride, above, NULL, 1, 0);
}

void vpx_dc_128_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  (void)above;
  (void)left;
  dc_4x4(dst, stride, NULL, NULL, 0, 0);
}

//------------------------------------------------------------------------------
// DC 8x8

// 'do_above' and 'do_left' facilitate branch removal when inlined.
static INLINE void dc_8x8(uint8_t *dst, ptrdiff_t stride,
                          const uint8_t *above, const uint8_t *left,
                          int do_above, int do_left) {
  uint16x8_t sum_top;
  uint16x8_t sum_left;
  uint8x8_t dc0;

  if (do_above) {
    const uint8x8_t A = vld1_u8(above);  // top row
    const uint16x4_t p0 = vpaddl_u8(A);  // cascading summation of the top
    const uint16x4_t p1 = vpadd_u16(p0, p0);
    const uint16x4_t p2 = vpadd_u16(p1, p1);
    sum_top = vcombine_u16(p2, p2);
  }

  if (do_left) {
    const uint8x8_t L = vld1_u8(left);  // left border
    const uint16x4_t p0 = vpaddl_u8(L);  // cascading summation of the left
    const uint16x4_t p1 = vpadd_u16(p0, p0);
    const uint16x4_t p2 = vpadd_u16(p1, p1);
    sum_left = vcombine_u16(p2, p2);
  }

  if (do_above && do_left) {
    const uint16x8_t sum = vaddq_u16(sum_left, sum_top);
    dc0 = vrshrn_n_u16(sum, 4);
  } else if (do_above) {
    dc0 = vrshrn_n_u16(sum_top, 3);
  } else if (do_left) {
    dc0 = vrshrn_n_u16(sum_left, 3);
  } else {
    dc0 = vdup_n_u8(0x80);
  }

  {
    const uint8x8_t dc = vdup_lane_u8(dc0, 0);
    int i;
    for (i = 0; i < 8; ++i) {
      vst1_u32((uint32_t*)(dst + i * stride), vreinterpret_u32_u8(dc));
    }
  }
}

void vpx_dc_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  dc_8x8(dst, stride, above, left, 1, 1);
}

void vpx_dc_left_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above, const uint8_t *left) {
  (void)above;
  dc_8x8(dst, stride, NULL, left, 0, 1);
}

void vpx_dc_top_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  (void)left;
  dc_8x8(dst, stride, above, NULL, 1, 0);
}

void vpx_dc_128_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  (void)above;
  (void)left;
  dc_8x8(dst, stride, NULL, NULL, 0, 0);
}

//------------------------------------------------------------------------------
// DC 16x16

// 'do_above' and 'do_left' facilitate branch removal when inlined.
static INLINE void dc_16x16(uint8_t *dst, ptrdiff_t stride,
                            const uint8_t *above, const uint8_t *left,
                            int do_above, int do_left) {
  uint16x8_t sum_top;
  uint16x8_t sum_left;
  uint8x8_t dc0;

  if (do_above) {
    const uint8x16_t A = vld1q_u8(above);  // top row
    const uint16x8_t p0 = vpaddlq_u8(A);  // cascading summation of the top
    const uint16x4_t p1 = vadd_u16(vget_low_u16(p0), vget_high_u16(p0));
    const uint16x4_t p2 = vpadd_u16(p1, p1);
    const uint16x4_t p3 = vpadd_u16(p2, p2);
    sum_top = vcombine_u16(p3, p3);
  }

  if (do_left) {
    const uint8x16_t L = vld1q_u8(left);  // left row
    const uint16x8_t p0 = vpaddlq_u8(L);  // cascading summation of the left
    const uint16x4_t p1 = vadd_u16(vget_low_u16(p0), vget_high_u16(p0));
    const uint16x4_t p2 = vpadd_u16(p1, p1);
    const uint16x4_t p3 = vpadd_u16(p2, p2);
    sum_left = vcombine_u16(p3, p3);
  }

  if (do_above && do_left) {
    const uint16x8_t sum = vaddq_u16(sum_left, sum_top);
    dc0 = vrshrn_n_u16(sum, 5);
  } else if (do_above) {
    dc0 = vrshrn_n_u16(sum_top, 4);
  } else if (do_left) {
    dc0 = vrshrn_n_u16(sum_left, 4);
  } else {
    dc0 = vdup_n_u8(0x80);
  }

  {
    const uint8x16_t dc = vdupq_lane_u8(dc0, 0);
    int i;
    for (i = 0; i < 16; ++i) {
      vst1q_u8(dst + i * stride, dc);
    }
  }
}

void vpx_dc_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  dc_16x16(dst, stride, above, left, 1, 1);
}

void vpx_dc_left_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                      const uint8_t *above,
                                      const uint8_t *left) {
  (void)above;
  dc_16x16(dst, stride, NULL, left, 0, 1);
}

void vpx_dc_top_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  (void)left;
  dc_16x16(dst, stride, above, NULL, 1, 0);
}

void vpx_dc_128_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  (void)above;
  (void)left;
  dc_16x16(dst, stride, NULL, NULL, 0, 0);
}

//------------------------------------------------------------------------------
// DC 32x32

// 'do_above' and 'do_left' facilitate branch removal when inlined.
static INLINE void dc_32x32(uint8_t *dst, ptrdiff_t stride,
                            const uint8_t *above, const uint8_t *left,
                            int do_above, int do_left) {
  uint16x8_t sum_top;
  uint16x8_t sum_left;
  uint8x8_t dc0;

  if (do_above) {
    const uint8x16_t A0 = vld1q_u8(above);  // top row
    const uint8x16_t A1 = vld1q_u8(above + 16);
    const uint16x8_t p0 = vpaddlq_u8(A0);  // cascading summation of the top
    const uint16x8_t p1 = vpaddlq_u8(A1);
    const uint16x8_t p2 = vaddq_u16(p0, p1);
    const uint16x4_t p3 = vadd_u16(vget_low_u16(p2), vget_high_u16(p2));
    const uint16x4_t p4 = vpadd_u16(p3, p3);
    const uint16x4_t p5 = vpadd_u16(p4, p4);
    sum_top = vcombine_u16(p5, p5);
  }

  if (do_left) {
    const uint8x16_t L0 = vld1q_u8(left);  // left row
    const uint8x16_t L1 = vld1q_u8(left + 16);
    const uint16x8_t p0 = vpaddlq_u8(L0);  // cascading summation of the left
    const uint16x8_t p1 = vpaddlq_u8(L1);
    const uint16x8_t p2 = vaddq_u16(p0, p1);
    const uint16x4_t p3 = vadd_u16(vget_low_u16(p2), vget_high_u16(p2));
    const uint16x4_t p4 = vpadd_u16(p3, p3);
    const uint16x4_t p5 = vpadd_u16(p4, p4);
    sum_left = vcombine_u16(p5, p5);
  }

  if (do_above && do_left) {
    const uint16x8_t sum = vaddq_u16(sum_left, sum_top);
    dc0 = vrshrn_n_u16(sum, 6);
  } else if (do_above) {
    dc0 = vrshrn_n_u16(sum_top, 5);
  } else if (do_left) {
    dc0 = vrshrn_n_u16(sum_left, 5);
  } else {
    dc0 = vdup_n_u8(0x80);
  }

  {
    const uint8x16_t dc = vdupq_lane_u8(dc0, 0);
    int i;
    for (i = 0; i < 32; ++i) {
      vst1q_u8(dst + i * stride, dc);
      vst1q_u8(dst + i * stride + 16, dc);
    }
  }
}

void vpx_dc_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  dc_32x32(dst, stride, above, left, 1, 1);
}

void vpx_dc_left_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                      const uint8_t *above,
                                      const uint8_t *left) {
  (void)above;
  dc_32x32(dst, stride, NULL, left, 0, 1);
}

void vpx_dc_top_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  (void)left;
  dc_32x32(dst, stride, above, NULL, 1, 0);
}

void vpx_dc_128_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  (void)above;
  (void)left;
  dc_32x32(dst, stride, NULL, NULL, 0, 0);
}

// -----------------------------------------------------------------------------

void vpx_d45_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  const uint64x1_t A0 = vreinterpret_u64_u8(vld1_u8(above));  // top row
  const uint64x1_t A1 = vshr_n_u64(A0, 8);
  const uint64x1_t A2 = vshr_n_u64(A0, 16);
  const uint8x8_t ABCDEFGH = vreinterpret_u8_u64(A0);
  const uint8x8_t BCDEFGH0 = vreinterpret_u8_u64(A1);
  const uint8x8_t CDEFGH00 = vreinterpret_u8_u64(A2);
  const uint8x8_t avg1 = vhadd_u8(ABCDEFGH, CDEFGH00);
  const uint8x8_t avg2 = vrhadd_u8(avg1, BCDEFGH0);
  const uint64x1_t avg2_u64 = vreinterpret_u64_u8(avg2);
  const uint32x2_t r0 = vreinterpret_u32_u8(avg2);
  const uint32x2_t r1 = vreinterpret_u32_u64(vshr_n_u64(avg2_u64, 8));
  const uint32x2_t r2 = vreinterpret_u32_u64(vshr_n_u64(avg2_u64, 16));
  const uint32x2_t r3 = vreinterpret_u32_u64(vshr_n_u64(avg2_u64, 24));
  (void)left;
  vst1_lane_u32((uint32_t *)(dst + 0 * stride), r0, 0);
  vst1_lane_u32((uint32_t *)(dst + 1 * stride), r1, 0);
  vst1_lane_u32((uint32_t *)(dst + 2 * stride), r2, 0);
  vst1_lane_u32((uint32_t *)(dst + 3 * stride), r3, 0);
  dst[3 * stride + 3] = above[7];
}

void vpx_d45_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  static const uint8_t shuffle1[8] = { 1, 2, 3, 4, 5, 6, 7, 7 };
  static const uint8_t shuffle2[8] = { 2, 3, 4, 5, 6, 7, 7, 7 };
  const uint8x8_t sh_12345677 = vld1_u8(shuffle1);
  const uint8x8_t sh_23456777 = vld1_u8(shuffle2);
  const uint8x8_t A0 = vld1_u8(above);  // top row
  const uint8x8_t A1 = vtbl1_u8(A0, sh_12345677);
  const uint8x8_t A2 = vtbl1_u8(A0, sh_23456777);
  const uint8x8_t avg1 = vhadd_u8(A0, A2);
  uint8x8_t row = vrhadd_u8(avg1, A1);
  int i;
  (void)left;
  for (i = 0; i < 7; ++i) {
    vst1_u8(dst + i * stride, row);
    row = vtbl1_u8(row, sh_12345677);
  }
  vst1_u8(dst + i * stride, row);
}

void vpx_d45_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                  const uint8_t *above, const uint8_t *left) {
  const uint8x16_t A0 = vld1q_u8(above);  // top row
  const uint8x16_t above_right = vld1q_dup_u8(above + 15);
  const uint8x16_t A1 = vextq_u8(A0, above_right, 1);
  const uint8x16_t A2 = vextq_u8(A0, above_right, 2);
  const uint8x16_t avg1 = vhaddq_u8(A0, A2);
  uint8x16_t row = vrhaddq_u8(avg1, A1);
  int i;
  (void)left;
  for (i = 0; i < 15; ++i) {
    vst1q_u8(dst + i * stride, row);
    row = vextq_u8(row, above_right, 1);
  }
  vst1q_u8(dst + i * stride, row);
}

// -----------------------------------------------------------------------------

void vpx_d135_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x8_t XABCD_u8 = vld1_u8(above - 1);
  const uint64x1_t XABCD = vreinterpret_u64_u8(XABCD_u8);
  const uint64x1_t ____XABC = vshl_n_u64(XABCD, 32);
  const uint32x2_t zero = vdup_n_u32(0);
  const uint32x2_t IJKL = vld1_lane_u32((const uint32_t *)left, zero, 0);
  const uint8x8_t IJKL_u8 = vreinterpret_u8_u32(IJKL);
  const uint64x1_t LKJI____ = vreinterpret_u64_u8(vrev32_u8(IJKL_u8));
  const uint64x1_t LKJIXABC = vorr_u64(LKJI____, ____XABC);
  const uint8x8_t KJIXABC_ = vreinterpret_u8_u64(vshr_n_u64(LKJIXABC, 8));
  const uint8x8_t JIXABC__ = vreinterpret_u8_u64(vshr_n_u64(LKJIXABC, 16));
  const uint8_t D = vget_lane_u8(XABCD_u8, 4);
  const uint8x8_t JIXABCD_ = vset_lane_u8(D, JIXABC__, 6);
  const uint8x8_t LKJIXABC_u8 = vreinterpret_u8_u64(LKJIXABC);
  const uint8x8_t avg1 = vhadd_u8(JIXABCD_, LKJIXABC_u8);
  const uint8x8_t avg2 = vrhadd_u8(avg1, KJIXABC_);
  const uint64x1_t avg2_u64 = vreinterpret_u64_u8(avg2);
  const uint32x2_t r3 = vreinterpret_u32_u8(avg2);
  const uint32x2_t r2 = vreinterpret_u32_u64(vshr_n_u64(avg2_u64, 8));
  const uint32x2_t r1 = vreinterpret_u32_u64(vshr_n_u64(avg2_u64, 16));
  const uint32x2_t r0 = vreinterpret_u32_u64(vshr_n_u64(avg2_u64, 24));
  vst1_lane_u32((uint32_t *)(dst + 0 * stride), r0, 0);
  vst1_lane_u32((uint32_t *)(dst + 1 * stride), r1, 0);
  vst1_lane_u32((uint32_t *)(dst + 2 * stride), r2, 0);
  vst1_lane_u32((uint32_t *)(dst + 3 * stride), r3, 0);
}

#if !HAVE_NEON_ASM

void vpx_v_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  int i;
  uint32x2_t d0u32 = vdup_n_u32(0);
  (void)left;

  d0u32 = vld1_lane_u32((const uint32_t *)above, d0u32, 0);
  for (i = 0; i < 4; i++, dst += stride)
    vst1_lane_u32((uint32_t *)dst, d0u32, 0);
}

void vpx_v_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  int i;
  uint8x8_t d0u8 = vdup_n_u8(0);
  (void)left;

  d0u8 = vld1_u8(above);
  for (i = 0; i < 8; i++, dst += stride)
    vst1_u8(dst, d0u8);
}

void vpx_v_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  int i;
  uint8x16_t q0u8 = vdupq_n_u8(0);
  (void)left;

  q0u8 = vld1q_u8(above);
  for (i = 0; i < 16; i++, dst += stride)
    vst1q_u8(dst, q0u8);
}

void vpx_v_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  int i;
  uint8x16_t q0u8 = vdupq_n_u8(0);
  uint8x16_t q1u8 = vdupq_n_u8(0);
  (void)left;

  q0u8 = vld1q_u8(above);
  q1u8 = vld1q_u8(above + 16);
  for (i = 0; i < 32; i++, dst += stride) {
    vst1q_u8(dst, q0u8);
    vst1q_u8(dst + 16, q1u8);
  }
}

void vpx_h_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  uint8x8_t d0u8 = vdup_n_u8(0);
  uint32x2_t d1u32 = vdup_n_u32(0);
  (void)above;

  d1u32 = vld1_lane_u32((const uint32_t *)left, d1u32, 0);

  d0u8 = vdup_lane_u8(vreinterpret_u8_u32(d1u32), 0);
  vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d0u8), 0);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u32(d1u32), 1);
  vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d0u8), 0);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u32(d1u32), 2);
  vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d0u8), 0);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u32(d1u32), 3);
  vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d0u8), 0);
}

void vpx_h_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  uint8x8_t d0u8 = vdup_n_u8(0);
  uint64x1_t d1u64 = vdup_n_u64(0);
  (void)above;

  d1u64 = vld1_u64((const uint64_t *)left);

  d0u8 = vdup_lane_u8(vreinterpret_u8_u64(d1u64), 0);
  vst1_u8(dst, d0u8);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u64(d1u64), 1);
  vst1_u8(dst, d0u8);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u64(d1u64), 2);
  vst1_u8(dst, d0u8);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u64(d1u64), 3);
  vst1_u8(dst, d0u8);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u64(d1u64), 4);
  vst1_u8(dst, d0u8);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u64(d1u64), 5);
  vst1_u8(dst, d0u8);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u64(d1u64), 6);
  vst1_u8(dst, d0u8);
  dst += stride;
  d0u8 = vdup_lane_u8(vreinterpret_u8_u64(d1u64), 7);
  vst1_u8(dst, d0u8);
}

void vpx_h_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  int j;
  uint8x8_t d2u8 = vdup_n_u8(0);
  uint8x16_t q0u8 = vdupq_n_u8(0);
  uint8x16_t q1u8 = vdupq_n_u8(0);
  (void)above;

  q1u8 = vld1q_u8(left);
  d2u8 = vget_low_u8(q1u8);
  for (j = 0; j < 2; j++, d2u8 = vget_high_u8(q1u8)) {
    q0u8 = vdupq_lane_u8(d2u8, 0);
    vst1q_u8(dst, q0u8);
    dst += stride;
    q0u8 = vdupq_lane_u8(d2u8, 1);
    vst1q_u8(dst, q0u8);
    dst += stride;
    q0u8 = vdupq_lane_u8(d2u8, 2);
    vst1q_u8(dst, q0u8);
    dst += stride;
    q0u8 = vdupq_lane_u8(d2u8, 3);
    vst1q_u8(dst, q0u8);
    dst += stride;
    q0u8 = vdupq_lane_u8(d2u8, 4);
    vst1q_u8(dst, q0u8);
    dst += stride;
    q0u8 = vdupq_lane_u8(d2u8, 5);
    vst1q_u8(dst, q0u8);
    dst += stride;
    q0u8 = vdupq_lane_u8(d2u8, 6);
    vst1q_u8(dst, q0u8);
    dst += stride;
    q0u8 = vdupq_lane_u8(d2u8, 7);
    vst1q_u8(dst, q0u8);
    dst += stride;
  }
}

void vpx_h_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  int j, k;
  uint8x8_t d2u8 = vdup_n_u8(0);
  uint8x16_t q0u8 = vdupq_n_u8(0);
  uint8x16_t q1u8 = vdupq_n_u8(0);
  (void)above;

  for (k = 0; k < 2; k++, left += 16) {
    q1u8 = vld1q_u8(left);
    d2u8 = vget_low_u8(q1u8);
    for (j = 0; j < 2; j++, d2u8 = vget_high_u8(q1u8)) {
      q0u8 = vdupq_lane_u8(d2u8, 0);
      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q0u8);
      dst += stride;
      q0u8 = vdupq_lane_u8(d2u8, 1);
      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q0u8);
      dst += stride;
      q0u8 = vdupq_lane_u8(d2u8, 2);
      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q0u8);
      dst += stride;
      q0u8 = vdupq_lane_u8(d2u8, 3);
      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q0u8);
      dst += stride;
      q0u8 = vdupq_lane_u8(d2u8, 4);
      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q0u8);
      dst += stride;
      q0u8 = vdupq_lane_u8(d2u8, 5);
      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q0u8);
      dst += stride;
      q0u8 = vdupq_lane_u8(d2u8, 6);
      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q0u8);
      dst += stride;
      q0u8 = vdupq_lane_u8(d2u8, 7);
      vst1q_u8(dst, q0u8);
      vst1q_u8(dst + 16, q0u8);
      dst += stride;
    }
  }
}

void vpx_tm_predictor_4x4_neon(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  int i;
  uint16x8_t q1u16, q3u16;
  int16x8_t q1s16;
  uint8x8_t d0u8 = vdup_n_u8(0);
  uint32x2_t d2u32 = vdup_n_u32(0);

  d0u8 = vld1_dup_u8(above - 1);
  d2u32 = vld1_lane_u32((const uint32_t *)above, d2u32, 0);
  q3u16 = vsubl_u8(vreinterpret_u8_u32(d2u32), d0u8);
  for (i = 0; i < 4; i++, dst += stride) {
    q1u16 = vdupq_n_u16((uint16_t)left[i]);
    q1s16 = vaddq_s16(vreinterpretq_s16_u16(q1u16),
                      vreinterpretq_s16_u16(q3u16));
    d0u8 = vqmovun_s16(q1s16);
    vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(d0u8), 0);
  }
}

void vpx_tm_predictor_8x8_neon(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  int j;
  uint16x8_t q0u16, q3u16, q10u16;
  int16x8_t q0s16;
  uint16x4_t d20u16;
  uint8x8_t d0u8, d2u8, d30u8;

  d0u8 = vld1_dup_u8(above - 1);
  d30u8 = vld1_u8(left);
  d2u8 = vld1_u8(above);
  q10u16 = vmovl_u8(d30u8);
  q3u16 = vsubl_u8(d2u8, d0u8);
  d20u16 = vget_low_u16(q10u16);
  for (j = 0; j < 2; j++, d20u16 = vget_high_u16(q10u16)) {
    q0u16 = vdupq_lane_u16(d20u16, 0);
    q0s16 = vaddq_s16(vreinterpretq_s16_u16(q3u16),
                      vreinterpretq_s16_u16(q0u16));
    d0u8 = vqmovun_s16(q0s16);
    vst1_u64((uint64_t *)dst, vreinterpret_u64_u8(d0u8));
    dst += stride;
    q0u16 = vdupq_lane_u16(d20u16, 1);
    q0s16 = vaddq_s16(vreinterpretq_s16_u16(q3u16),
                      vreinterpretq_s16_u16(q0u16));
    d0u8 = vqmovun_s16(q0s16);
    vst1_u64((uint64_t *)dst, vreinterpret_u64_u8(d0u8));
    dst += stride;
    q0u16 = vdupq_lane_u16(d20u16, 2);
    q0s16 = vaddq_s16(vreinterpretq_s16_u16(q3u16),
                      vreinterpretq_s16_u16(q0u16));
    d0u8 = vqmovun_s16(q0s16);
    vst1_u64((uint64_t *)dst, vreinterpret_u64_u8(d0u8));
    dst += stride;
    q0u16 = vdupq_lane_u16(d20u16, 3);
    q0s16 = vaddq_s16(vreinterpretq_s16_u16(q3u16),
                      vreinterpretq_s16_u16(q0u16));
    d0u8 = vqmovun_s16(q0s16);
    vst1_u64((uint64_t *)dst, vreinterpret_u64_u8(d0u8));
    dst += stride;
  }
}

void vpx_tm_predictor_16x16_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  int j, k;
  uint16x8_t q0u16, q2u16, q3u16, q8u16, q10u16;
  uint8x16_t q0u8, q1u8;
  int16x8_t q0s16, q1s16, q8s16, q11s16;
  uint16x4_t d20u16;
  uint8x8_t d2u8, d3u8, d18u8, d22u8, d23u8;

  q0u8 = vld1q_dup_u8(above - 1);
  q1u8 = vld1q_u8(above);
  q2u16 = vsubl_u8(vget_low_u8(q1u8), vget_low_u8(q0u8));
  q3u16 = vsubl_u8(vget_high_u8(q1u8), vget_high_u8(q0u8));
  for (k = 0; k < 2; k++, left += 8) {
    d18u8 = vld1_u8(left);
    q10u16 = vmovl_u8(d18u8);
    d20u16 = vget_low_u16(q10u16);
    for (j = 0; j < 2; j++, d20u16 = vget_high_u16(q10u16)) {
      q0u16 = vdupq_lane_u16(d20u16, 0);
      q8u16 = vdupq_lane_u16(d20u16, 1);
      q1s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                        vreinterpretq_s16_u16(q2u16));
      q0s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                        vreinterpretq_s16_u16(q3u16));
      q11s16 = vaddq_s16(vreinterpretq_s16_u16(q8u16),
                         vreinterpretq_s16_u16(q2u16));
      q8s16 = vaddq_s16(vreinterpretq_s16_u16(q8u16),
                        vreinterpretq_s16_u16(q3u16));
      d2u8 = vqmovun_s16(q1s16);
      d3u8 = vqmovun_s16(q0s16);
      d22u8 = vqmovun_s16(q11s16);
      d23u8 = vqmovun_s16(q8s16);
      vst1_u64((uint64_t *)dst, vreinterpret_u64_u8(d2u8));
      vst1_u64((uint64_t *)(dst + 8), vreinterpret_u64_u8(d3u8));
      dst += stride;
      vst1_u64((uint64_t *)dst, vreinterpret_u64_u8(d22u8));
      vst1_u64((uint64_t *)(dst + 8), vreinterpret_u64_u8(d23u8));
      dst += stride;

      q0u16 = vdupq_lane_u16(d20u16, 2);
      q8u16 = vdupq_lane_u16(d20u16, 3);
      q1s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                        vreinterpretq_s16_u16(q2u16));
      q0s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                        vreinterpretq_s16_u16(q3u16));
      q11s16 = vaddq_s16(vreinterpretq_s16_u16(q8u16),
                         vreinterpretq_s16_u16(q2u16));
      q8s16 = vaddq_s16(vreinterpretq_s16_u16(q8u16),
                        vreinterpretq_s16_u16(q3u16));
      d2u8 = vqmovun_s16(q1s16);
      d3u8 = vqmovun_s16(q0s16);
      d22u8 = vqmovun_s16(q11s16);
      d23u8 = vqmovun_s16(q8s16);
      vst1_u64((uint64_t *)dst, vreinterpret_u64_u8(d2u8));
      vst1_u64((uint64_t *)(dst + 8), vreinterpret_u64_u8(d3u8));
      dst += stride;
      vst1_u64((uint64_t *)dst, vreinterpret_u64_u8(d22u8));
      vst1_u64((uint64_t *)(dst + 8), vreinterpret_u64_u8(d23u8));
      dst += stride;
    }
  }
}

void vpx_tm_predictor_32x32_neon(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  int j, k;
  uint16x8_t q0u16, q3u16, q8u16, q9u16, q10u16, q11u16;
  uint8x16_t q0u8, q1u8, q2u8;
  int16x8_t q12s16, q13s16, q14s16, q15s16;
  uint16x4_t d6u16;
  uint8x8_t d0u8, d1u8, d2u8, d3u8, d26u8;

  q0u8 = vld1q_dup_u8(above - 1);
  q1u8 = vld1q_u8(above);
  q2u8 = vld1q_u8(above + 16);
  q8u16 = vsubl_u8(vget_low_u8(q1u8), vget_low_u8(q0u8));
  q9u16 = vsubl_u8(vget_high_u8(q1u8), vget_high_u8(q0u8));
  q10u16 = vsubl_u8(vget_low_u8(q2u8), vget_low_u8(q0u8));
  q11u16 = vsubl_u8(vget_high_u8(q2u8), vget_high_u8(q0u8));
  for (k = 0; k < 4; k++, left += 8) {
    d26u8 = vld1_u8(left);
    q3u16 = vmovl_u8(d26u8);
    d6u16 = vget_low_u16(q3u16);
    for (j = 0; j < 2; j++, d6u16 = vget_high_u16(q3u16)) {
      q0u16 = vdupq_lane_u16(d6u16, 0);
      q12s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q8u16));
      q13s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q9u16));
      q14s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q10u16));
      q15s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q11u16));
      d0u8 = vqmovun_s16(q12s16);
      d1u8 = vqmovun_s16(q13s16);
      d2u8 = vqmovun_s16(q14s16);
      d3u8 = vqmovun_s16(q15s16);
      q0u8 = vcombine_u8(d0u8, d1u8);
      q1u8 = vcombine_u8(d2u8, d3u8);
      vst1q_u64((uint64_t *)dst, vreinterpretq_u64_u8(q0u8));
      vst1q_u64((uint64_t *)(dst + 16), vreinterpretq_u64_u8(q1u8));
      dst += stride;

      q0u16 = vdupq_lane_u16(d6u16, 1);
      q12s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q8u16));
      q13s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q9u16));
      q14s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q10u16));
      q15s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q11u16));
      d0u8 = vqmovun_s16(q12s16);
      d1u8 = vqmovun_s16(q13s16);
      d2u8 = vqmovun_s16(q14s16);
      d3u8 = vqmovun_s16(q15s16);
      q0u8 = vcombine_u8(d0u8, d1u8);
      q1u8 = vcombine_u8(d2u8, d3u8);
      vst1q_u64((uint64_t *)dst, vreinterpretq_u64_u8(q0u8));
      vst1q_u64((uint64_t *)(dst + 16), vreinterpretq_u64_u8(q1u8));
      dst += stride;

      q0u16 = vdupq_lane_u16(d6u16, 2);
      q12s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q8u16));
      q13s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q9u16));
      q14s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q10u16));
      q15s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q11u16));
      d0u8 = vqmovun_s16(q12s16);
      d1u8 = vqmovun_s16(q13s16);
      d2u8 = vqmovun_s16(q14s16);
      d3u8 = vqmovun_s16(q15s16);
      q0u8 = vcombine_u8(d0u8, d1u8);
      q1u8 = vcombine_u8(d2u8, d3u8);
      vst1q_u64((uint64_t *)dst, vreinterpretq_u64_u8(q0u8));
      vst1q_u64((uint64_t *)(dst + 16), vreinterpretq_u64_u8(q1u8));
      dst += stride;

      q0u16 = vdupq_lane_u16(d6u16, 3);
      q12s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q8u16));
      q13s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q9u16));
      q14s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q10u16));
      q15s16 = vaddq_s16(vreinterpretq_s16_u16(q0u16),
                         vreinterpretq_s16_u16(q11u16));
      d0u8 = vqmovun_s16(q12s16);
      d1u8 = vqmovun_s16(q13s16);
      d2u8 = vqmovun_s16(q14s16);
      d3u8 = vqmovun_s16(q15s16);
      q0u8 = vcombine_u8(d0u8, d1u8);
      q1u8 = vcombine_u8(d2u8, d3u8);
      vst1q_u64((uint64_t *)dst, vreinterpretq_u64_u8(q0u8));
      vst1q_u64((uint64_t *)(dst + 16), vreinterpretq_u64_u8(q1u8));
      dst += stride;
    }
  }
}
#endif  // !HAVE_NEON_ASM
