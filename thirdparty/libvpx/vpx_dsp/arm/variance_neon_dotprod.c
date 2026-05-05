/*
 *  Copyright (c) 2021 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>

#include "./vpx_dsp_rtcd.h"
#include "./vpx_config.h"

#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"
#include "vpx_ports/mem.h"

// Process a block of width 4 four rows at a time.
static INLINE void variance_4xh_neon_dotprod(const uint8_t *src_ptr,
                                             int src_stride,
                                             const uint8_t *ref_ptr,
                                             int ref_stride, int h,
                                             uint32_t *sse, int *sum) {
  uint32x4_t src_sum = vdupq_n_u32(0);
  uint32x4_t ref_sum = vdupq_n_u32(0);
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  int i = h;
  do {
    const uint8x16_t s = load_unaligned_u8q(src_ptr, src_stride);
    const uint8x16_t r = load_unaligned_u8q(ref_ptr, ref_stride);

    const uint8x16_t abs_diff = vabdq_u8(s, r);
    sse_u32 = vdotq_u32(sse_u32, abs_diff, abs_diff);

    src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
    ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

    src_ptr += 4 * src_stride;
    ref_ptr += 4 * ref_stride;
    i -= 4;
  } while (i != 0);

  *sum = horizontal_add_int32x4(
      vreinterpretq_s32_u32(vsubq_u32(src_sum, ref_sum)));
  *sse = horizontal_add_uint32x4(sse_u32);
}

// Process a block of width 8 two rows at a time.
static INLINE void variance_8xh_neon_dotprod(const uint8_t *src_ptr,
                                             int src_stride,
                                             const uint8_t *ref_ptr,
                                             int ref_stride, int h,
                                             uint32_t *sse, int *sum) {
  uint32x4_t src_sum = vdupq_n_u32(0);
  uint32x4_t ref_sum = vdupq_n_u32(0);
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  int i = h;
  do {
    const uint8x16_t s =
        vcombine_u8(vld1_u8(src_ptr), vld1_u8(src_ptr + src_stride));
    const uint8x16_t r =
        vcombine_u8(vld1_u8(ref_ptr), vld1_u8(ref_ptr + ref_stride));

    const uint8x16_t abs_diff = vabdq_u8(s, r);
    sse_u32 = vdotq_u32(sse_u32, abs_diff, abs_diff);

    src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
    ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

    src_ptr += 2 * src_stride;
    ref_ptr += 2 * ref_stride;
    i -= 2;
  } while (i != 0);

  *sum = horizontal_add_int32x4(
      vreinterpretq_s32_u32(vsubq_u32(src_sum, ref_sum)));
  *sse = horizontal_add_uint32x4(sse_u32);
}

// Process a block of width 16 one row at a time.
static INLINE void variance_16xh_neon_dotprod(const uint8_t *src_ptr,
                                              int src_stride,
                                              const uint8_t *ref_ptr,
                                              int ref_stride, int h,
                                              uint32_t *sse, int *sum) {
  uint32x4_t src_sum = vdupq_n_u32(0);
  uint32x4_t ref_sum = vdupq_n_u32(0);
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  int i = h;
  do {
    const uint8x16_t s = vld1q_u8(src_ptr);
    const uint8x16_t r = vld1q_u8(ref_ptr);

    const uint8x16_t abs_diff = vabdq_u8(s, r);
    sse_u32 = vdotq_u32(sse_u32, abs_diff, abs_diff);

    src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
    ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--i != 0);

  *sum = horizontal_add_int32x4(
      vreinterpretq_s32_u32(vsubq_u32(src_sum, ref_sum)));
  *sse = horizontal_add_uint32x4(sse_u32);
}

// Process a block of any size where the width is divisible by 16.
static INLINE void variance_large_neon_dotprod(const uint8_t *src_ptr,
                                               int src_stride,
                                               const uint8_t *ref_ptr,
                                               int ref_stride, int w, int h,
                                               uint32_t *sse, int *sum) {
  uint32x4_t src_sum = vdupq_n_u32(0);
  uint32x4_t ref_sum = vdupq_n_u32(0);
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  int i = h;
  do {
    int j = 0;
    do {
      const uint8x16_t s = vld1q_u8(src_ptr + j);
      const uint8x16_t r = vld1q_u8(ref_ptr + j);

      const uint8x16_t abs_diff = vabdq_u8(s, r);
      sse_u32 = vdotq_u32(sse_u32, abs_diff, abs_diff);

      src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
      ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

      j += 16;
    } while (j < w);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--i != 0);

  *sum = horizontal_add_int32x4(
      vreinterpretq_s32_u32(vsubq_u32(src_sum, ref_sum)));
  *sse = horizontal_add_uint32x4(sse_u32);
}

static INLINE void variance_32xh_neon_dotprod(const uint8_t *src,
                                              int src_stride,
                                              const uint8_t *ref,
                                              int ref_stride, int h,
                                              uint32_t *sse, int *sum) {
  variance_large_neon_dotprod(src, src_stride, ref, ref_stride, 32, h, sse,
                              sum);
}

static INLINE void variance_64xh_neon_dotprod(const uint8_t *src,
                                              int src_stride,
                                              const uint8_t *ref,
                                              int ref_stride, int h,
                                              uint32_t *sse, int *sum) {
  variance_large_neon_dotprod(src, src_stride, ref, ref_stride, 64, h, sse,
                              sum);
}

void vpx_get8x8var_neon_dotprod(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *ref_ptr, int ref_stride,
                                unsigned int *sse, int *sum) {
  variance_8xh_neon_dotprod(src_ptr, src_stride, ref_ptr, ref_stride, 8, sse,
                            sum);
}

void vpx_get16x16var_neon_dotprod(const uint8_t *src_ptr, int src_stride,
                                  const uint8_t *ref_ptr, int ref_stride,
                                  unsigned int *sse, int *sum) {
  variance_16xh_neon_dotprod(src_ptr, src_stride, ref_ptr, ref_stride, 16, sse,
                             sum);
}

#define VARIANCE_WXH_NEON_DOTPROD(w, h, shift)                                \
  unsigned int vpx_variance##w##x##h##_neon_dotprod(                          \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      unsigned int *sse) {                                                    \
    int sum;                                                                  \
    variance_##w##xh_neon_dotprod(src, src_stride, ref, ref_stride, h, sse,   \
                                  &sum);                                      \
    return *sse - (uint32_t)(((int64_t)sum * sum) >> shift);                  \
  }

VARIANCE_WXH_NEON_DOTPROD(4, 4, 4)
VARIANCE_WXH_NEON_DOTPROD(4, 8, 5)

VARIANCE_WXH_NEON_DOTPROD(8, 4, 5)
VARIANCE_WXH_NEON_DOTPROD(8, 8, 6)
VARIANCE_WXH_NEON_DOTPROD(8, 16, 7)

VARIANCE_WXH_NEON_DOTPROD(16, 8, 7)
VARIANCE_WXH_NEON_DOTPROD(16, 16, 8)
VARIANCE_WXH_NEON_DOTPROD(16, 32, 9)

VARIANCE_WXH_NEON_DOTPROD(32, 16, 9)
VARIANCE_WXH_NEON_DOTPROD(32, 32, 10)
VARIANCE_WXH_NEON_DOTPROD(32, 64, 11)

VARIANCE_WXH_NEON_DOTPROD(64, 32, 11)
VARIANCE_WXH_NEON_DOTPROD(64, 64, 12)

#undef VARIANCE_WXH_NEON_DOTPROD

static INLINE unsigned int vpx_mse8xh_neon_dotprod(const unsigned char *src_ptr,
                                                   int src_stride,
                                                   const unsigned char *ref_ptr,
                                                   int ref_stride, int h) {
  uint32x2_t sse_u32[2] = { vdup_n_u32(0), vdup_n_u32(0) };

  int i = h / 2;
  do {
    uint8x8_t s0, s1, r0, r1, diff0, diff1;

    s0 = vld1_u8(src_ptr);
    src_ptr += src_stride;
    s1 = vld1_u8(src_ptr);
    src_ptr += src_stride;
    r0 = vld1_u8(ref_ptr);
    ref_ptr += ref_stride;
    r1 = vld1_u8(ref_ptr);
    ref_ptr += ref_stride;

    diff0 = vabd_u8(s0, r0);
    diff1 = vabd_u8(s1, r1);

    sse_u32[0] = vdot_u32(sse_u32[0], diff0, diff0);
    sse_u32[1] = vdot_u32(sse_u32[1], diff1, diff1);
  } while (--i != 0);

  return horizontal_add_uint32x2(vadd_u32(sse_u32[0], sse_u32[1]));
}

static INLINE unsigned int vpx_mse16xh_neon_dotprod(
    const unsigned char *src_ptr, int src_stride, const unsigned char *ref_ptr,
    int ref_stride, int h) {
  uint32x4_t sse_u32[2] = { vdupq_n_u32(0), vdupq_n_u32(0) };

  int i = h / 2;
  do {
    uint8x16_t s0, s1, r0, r1, diff0, diff1;

    s0 = vld1q_u8(src_ptr);
    src_ptr += src_stride;
    s1 = vld1q_u8(src_ptr);
    src_ptr += src_stride;
    r0 = vld1q_u8(ref_ptr);
    ref_ptr += ref_stride;
    r1 = vld1q_u8(ref_ptr);
    ref_ptr += ref_stride;

    diff0 = vabdq_u8(s0, r0);
    diff1 = vabdq_u8(s1, r1);

    sse_u32[0] = vdotq_u32(sse_u32[0], diff0, diff0);
    sse_u32[1] = vdotq_u32(sse_u32[1], diff1, diff1);
  } while (--i != 0);

  return horizontal_add_uint32x4(vaddq_u32(sse_u32[0], sse_u32[1]));
}

unsigned int vpx_get4x4sse_cs_neon_dotprod(const unsigned char *src_ptr,
                                           int src_stride,
                                           const unsigned char *ref_ptr,
                                           int ref_stride) {
  uint8x16_t s = load_unaligned_u8q(src_ptr, src_stride);
  uint8x16_t r = load_unaligned_u8q(ref_ptr, ref_stride);

  uint8x16_t abs_diff = vabdq_u8(s, r);

  uint32x4_t sse = vdotq_u32(vdupq_n_u32(0), abs_diff, abs_diff);

  return horizontal_add_uint32x4(sse);
}

#define VPX_MSE_WXH_NEON_DOTPROD(w, h)                                   \
  unsigned int vpx_mse##w##x##h##_neon_dotprod(                          \
      const unsigned char *src_ptr, int src_stride,                      \
      const unsigned char *ref_ptr, int ref_stride, unsigned int *sse) { \
    *sse = vpx_mse##w##xh_neon_dotprod(src_ptr, src_stride, ref_ptr,     \
                                       ref_stride, h);                   \
    return *sse;                                                         \
  }

VPX_MSE_WXH_NEON_DOTPROD(8, 8)
VPX_MSE_WXH_NEON_DOTPROD(8, 16)
VPX_MSE_WXH_NEON_DOTPROD(16, 8)
VPX_MSE_WXH_NEON_DOTPROD(16, 16)

#undef VPX_MSE_WXH_NEON_DOTPROD
