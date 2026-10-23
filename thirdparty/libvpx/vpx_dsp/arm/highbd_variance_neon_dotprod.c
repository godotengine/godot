/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vpx_dsp_rtcd.h"
#include "./vpx_config.h"

#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"
#include "vpx_ports/mem.h"

static INLINE uint32_t highbd_mse8_8xh_neon_dotprod(const uint16_t *src_ptr,
                                                    int src_stride,
                                                    const uint16_t *ref_ptr,
                                                    int ref_stride, int h) {
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  int i = h / 2;
  do {
    uint16x8_t s0, s1, r0, r1;
    uint8x16_t s, r, diff;

    s0 = vld1q_u16(src_ptr);
    src_ptr += src_stride;
    s1 = vld1q_u16(src_ptr);
    src_ptr += src_stride;
    r0 = vld1q_u16(ref_ptr);
    ref_ptr += ref_stride;
    r1 = vld1q_u16(ref_ptr);
    ref_ptr += ref_stride;

    s = vcombine_u8(vmovn_u16(s0), vmovn_u16(s1));
    r = vcombine_u8(vmovn_u16(r0), vmovn_u16(r1));

    diff = vabdq_u8(s, r);
    sse_u32 = vdotq_u32(sse_u32, diff, diff);
  } while (--i != 0);

  return horizontal_add_uint32x4(sse_u32);
}

static INLINE uint32_t highbd_mse8_16xh_neon_dotprod(const uint16_t *src_ptr,
                                                     int src_stride,
                                                     const uint16_t *ref_ptr,
                                                     int ref_stride, int h) {
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  int i = h;
  do {
    uint16x8_t s0, s1, r0, r1;
    uint8x16_t s, r, diff;

    s0 = vld1q_u16(src_ptr);
    s1 = vld1q_u16(src_ptr + 8);
    r0 = vld1q_u16(ref_ptr);
    r1 = vld1q_u16(ref_ptr + 8);

    s = vcombine_u8(vmovn_u16(s0), vmovn_u16(s1));
    r = vcombine_u8(vmovn_u16(r0), vmovn_u16(r1));

    diff = vabdq_u8(s, r);
    sse_u32 = vdotq_u32(sse_u32, diff, diff);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--i != 0);

  return horizontal_add_uint32x4(sse_u32);
}

#define HIGHBD_MSE_WXH_NEON_DOTPROD(w, h)                                      \
  uint32_t vpx_highbd_8_mse##w##x##h##_neon_dotprod(                           \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride, uint32_t *sse) {                                         \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                              \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                              \
    *sse =                                                                     \
        highbd_mse8_##w##xh_neon_dotprod(src, src_stride, ref, ref_stride, h); \
    return *sse;                                                               \
  }

HIGHBD_MSE_WXH_NEON_DOTPROD(16, 16)
HIGHBD_MSE_WXH_NEON_DOTPROD(16, 8)
HIGHBD_MSE_WXH_NEON_DOTPROD(8, 16)
HIGHBD_MSE_WXH_NEON_DOTPROD(8, 8)

#undef HIGHBD_MSE_WXH_NEON_DOTPROD
