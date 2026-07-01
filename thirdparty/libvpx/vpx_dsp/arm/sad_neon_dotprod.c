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

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"

static INLINE unsigned int sadwxh_neon_dotprod(const uint8_t *src_ptr,
                                               int src_stride,
                                               const uint8_t *ref_ptr,
                                               int ref_stride, int w, int h) {
  // Only two accumulators are required for optimal instruction throughput of
  // the ABD, UDOT sequence on CPUs with either 2 or 4 Neon pipes.
  uint32x4_t sum[2] = { vdupq_n_u32(0), vdupq_n_u32(0) };

  int i = h;
  do {
    int j = 0;
    do {
      uint8x16_t s0, s1, r0, r1, diff0, diff1;

      s0 = vld1q_u8(src_ptr + j);
      r0 = vld1q_u8(ref_ptr + j);
      diff0 = vabdq_u8(s0, r0);
      sum[0] = vdotq_u32(sum[0], diff0, vdupq_n_u8(1));

      s1 = vld1q_u8(src_ptr + j + 16);
      r1 = vld1q_u8(ref_ptr + j + 16);
      diff1 = vabdq_u8(s1, r1);
      sum[1] = vdotq_u32(sum[1], diff1, vdupq_n_u8(1));

      j += 32;
    } while (j < w);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--i != 0);

  return horizontal_add_uint32x4(vaddq_u32(sum[0], sum[1]));
}

static INLINE unsigned int sad64xh_neon_dotprod(const uint8_t *src_ptr,
                                                int src_stride,
                                                const uint8_t *ref_ptr,
                                                int ref_stride, int h) {
  return sadwxh_neon_dotprod(src_ptr, src_stride, ref_ptr, ref_stride, 64, h);
}

static INLINE unsigned int sad32xh_neon_dotprod(const uint8_t *src_ptr,
                                                int src_stride,
                                                const uint8_t *ref_ptr,
                                                int ref_stride, int h) {
  return sadwxh_neon_dotprod(src_ptr, src_stride, ref_ptr, ref_stride, 32, h);
}

static INLINE unsigned int sad16xh_neon_dotprod(const uint8_t *src_ptr,
                                                int src_stride,
                                                const uint8_t *ref_ptr,
                                                int ref_stride, int h) {
  uint32x4_t sum[2] = { vdupq_n_u32(0), vdupq_n_u32(0) };

  int i = h / 2;
  do {
    uint8x16_t s0, s1, r0, r1, diff0, diff1;

    s0 = vld1q_u8(src_ptr);
    r0 = vld1q_u8(ref_ptr);
    diff0 = vabdq_u8(s0, r0);
    sum[0] = vdotq_u32(sum[0], diff0, vdupq_n_u8(1));

    src_ptr += src_stride;
    ref_ptr += ref_stride;

    s1 = vld1q_u8(src_ptr);
    r1 = vld1q_u8(ref_ptr);
    diff1 = vabdq_u8(s1, r1);
    sum[1] = vdotq_u32(sum[1], diff1, vdupq_n_u8(1));

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--i != 0);

  return horizontal_add_uint32x4(vaddq_u32(sum[0], sum[1]));
}

#define SAD_WXH_NEON_DOTPROD(w, h)                                         \
  unsigned int vpx_sad##w##x##h##_neon_dotprod(                            \
      const uint8_t *src, int src_stride, const uint8_t *ref,              \
      int ref_stride) {                                                    \
    return sad##w##xh_neon_dotprod(src, src_stride, ref, ref_stride, (h)); \
  }

SAD_WXH_NEON_DOTPROD(16, 8)
SAD_WXH_NEON_DOTPROD(16, 16)
SAD_WXH_NEON_DOTPROD(16, 32)

SAD_WXH_NEON_DOTPROD(32, 16)
SAD_WXH_NEON_DOTPROD(32, 32)
SAD_WXH_NEON_DOTPROD(32, 64)

SAD_WXH_NEON_DOTPROD(64, 32)
SAD_WXH_NEON_DOTPROD(64, 64)

#undef SAD_WXH_NEON_DOTPROD

#define SAD_SKIP_WXH_NEON_DOTPROD(w, h)                          \
  unsigned int vpx_sad_skip_##w##x##h##_neon_dotprod(            \
      const uint8_t *src, int src_stride, const uint8_t *ref,    \
      int ref_stride) {                                          \
    return 2 * sad##w##xh_neon_dotprod(src, 2 * src_stride, ref, \
                                       2 * ref_stride, (h) / 2); \
  }

SAD_SKIP_WXH_NEON_DOTPROD(16, 8)
SAD_SKIP_WXH_NEON_DOTPROD(16, 16)
SAD_SKIP_WXH_NEON_DOTPROD(16, 32)

SAD_SKIP_WXH_NEON_DOTPROD(32, 16)
SAD_SKIP_WXH_NEON_DOTPROD(32, 32)
SAD_SKIP_WXH_NEON_DOTPROD(32, 64)

SAD_SKIP_WXH_NEON_DOTPROD(64, 32)
SAD_SKIP_WXH_NEON_DOTPROD(64, 64)

#undef SAD_SKIP_WXH_NEON_DOTPROD

static INLINE unsigned int sadwxh_avg_neon_dotprod(const uint8_t *src_ptr,
                                                   int src_stride,
                                                   const uint8_t *ref_ptr,
                                                   int ref_stride, int w, int h,
                                                   const uint8_t *second_pred) {
  // Only two accumulators are required for optimal instruction throughput of
  // the ABD, UDOT sequence on CPUs with either 2 or 4 Neon pipes.
  uint32x4_t sum[2] = { vdupq_n_u32(0), vdupq_n_u32(0) };

  int i = h;
  do {
    int j = 0;
    do {
      uint8x16_t s0, s1, r0, r1, p0, p1, avg0, avg1, diff0, diff1;

      s0 = vld1q_u8(src_ptr + j);
      r0 = vld1q_u8(ref_ptr + j);
      p0 = vld1q_u8(second_pred);
      avg0 = vrhaddq_u8(r0, p0);
      diff0 = vabdq_u8(s0, avg0);
      sum[0] = vdotq_u32(sum[0], diff0, vdupq_n_u8(1));

      s1 = vld1q_u8(src_ptr + j + 16);
      r1 = vld1q_u8(ref_ptr + j + 16);
      p1 = vld1q_u8(second_pred + 16);
      avg1 = vrhaddq_u8(r1, p1);
      diff1 = vabdq_u8(s1, avg1);
      sum[1] = vdotq_u32(sum[1], diff1, vdupq_n_u8(1));

      j += 32;
      second_pred += 32;
    } while (j < w);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--i != 0);

  return horizontal_add_uint32x4(vaddq_u32(sum[0], sum[1]));
}

static INLINE unsigned int sad64xh_avg_neon_dotprod(
    const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
    int ref_stride, int h, const uint8_t *second_pred) {
  return sadwxh_avg_neon_dotprod(src_ptr, src_stride, ref_ptr, ref_stride, 64,
                                 h, second_pred);
}

static INLINE unsigned int sad32xh_avg_neon_dotprod(
    const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
    int ref_stride, int h, const uint8_t *second_pred) {
  return sadwxh_avg_neon_dotprod(src_ptr, src_stride, ref_ptr, ref_stride, 32,
                                 h, second_pred);
}

static INLINE unsigned int sad16xh_avg_neon_dotprod(
    const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,
    int ref_stride, int h, const uint8_t *second_pred) {
  uint32x4_t sum[2] = { vdupq_n_u32(0), vdupq_n_u32(0) };

  int i = h / 2;
  do {
    uint8x16_t s0, s1, r0, r1, p0, p1, avg0, avg1, diff0, diff1;

    s0 = vld1q_u8(src_ptr);
    r0 = vld1q_u8(ref_ptr);
    p0 = vld1q_u8(second_pred);
    avg0 = vrhaddq_u8(r0, p0);
    diff0 = vabdq_u8(s0, avg0);
    sum[0] = vdotq_u32(sum[0], diff0, vdupq_n_u8(1));

    src_ptr += src_stride;
    ref_ptr += ref_stride;
    second_pred += 16;

    s1 = vld1q_u8(src_ptr);
    r1 = vld1q_u8(ref_ptr);
    p1 = vld1q_u8(second_pred);
    avg1 = vrhaddq_u8(r1, p1);
    diff1 = vabdq_u8(s1, avg1);
    sum[1] = vdotq_u32(sum[1], diff1, vdupq_n_u8(1));

    src_ptr += src_stride;
    ref_ptr += ref_stride;
    second_pred += 16;
  } while (--i != 0);

  return horizontal_add_uint32x4(vaddq_u32(sum[0], sum[1]));
}

#define SAD_WXH_AVG_NEON_DOTPROD(w, h)                                        \
  uint32_t vpx_sad##w##x##h##_avg_neon_dotprod(                               \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      const uint8_t *second_pred) {                                           \
    return sad##w##xh_avg_neon_dotprod(src, src_stride, ref, ref_stride, (h), \
                                       second_pred);                          \
  }

SAD_WXH_AVG_NEON_DOTPROD(16, 8)
SAD_WXH_AVG_NEON_DOTPROD(16, 16)
SAD_WXH_AVG_NEON_DOTPROD(16, 32)

SAD_WXH_AVG_NEON_DOTPROD(32, 16)
SAD_WXH_AVG_NEON_DOTPROD(32, 32)
SAD_WXH_AVG_NEON_DOTPROD(32, 64)

SAD_WXH_AVG_NEON_DOTPROD(64, 32)
SAD_WXH_AVG_NEON_DOTPROD(64, 64)

#undef SAD_WXH_AVG_NEON_DOTPROD
