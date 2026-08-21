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
#include <assert.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"

static INLINE uint32_t highbd_sad4xh_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *ref_ptr,
                                          int ref_stride, int h) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr = CONVERT_TO_SHORTPTR(ref_ptr);
  uint32x4_t sum = vdupq_n_u32(0);

  do {
    uint16x4_t s = vld1_u16(src16_ptr);
    uint16x4_t r = vld1_u16(ref16_ptr);
    sum = vabal_u16(sum, s, r);

    src16_ptr += src_stride;
    ref16_ptr += ref_stride;
  } while (--h != 0);

  return horizontal_add_uint32x4(sum);
}

static INLINE uint32_t highbd_sad8xh_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *ref_ptr,
                                          int ref_stride, int h) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr = CONVERT_TO_SHORTPTR(ref_ptr);
  uint16x8_t sum = vdupq_n_u16(0);
  assert(h <= 16);

  do {
    uint16x8_t s = vld1q_u16(src16_ptr);
    uint16x8_t r = vld1q_u16(ref16_ptr);
    sum = vabaq_u16(sum, s, r);

    src16_ptr += src_stride;
    ref16_ptr += ref_stride;
  } while (--h != 0);

  return horizontal_add_uint16x8(sum);
}

static INLINE uint32_t highbd_sad16xh_neon(const uint8_t *src_ptr,
                                           int src_stride,
                                           const uint8_t *ref_ptr,
                                           int ref_stride, int h) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr = CONVERT_TO_SHORTPTR(ref_ptr);
  uint32x4_t sum_u32 = vdupq_n_u32(0);

  // 'h_overflow' is the number of 16-wide rows we can process before 16-bit
  // accumulators overflow. After hitting this limit accumulate into 32-bit
  // elements. 65535 / 4095 ~= 16, so 16 16-wide rows using two accumulators.
  const int h_overflow = 16;
  // If block height 'h' is smaller than this limit, use 'h' instead.
  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % h_limit == 0);

  do {
    uint16x8_t sum_u16[2] = { vdupq_n_u16(0), vdupq_n_u16(0) };
    int i = h_limit;
    do {
      uint16x8_t s0, s1, r0, r1;

      s0 = vld1q_u16(src16_ptr);
      r0 = vld1q_u16(ref16_ptr);
      sum_u16[0] = vabaq_u16(sum_u16[0], s0, r0);

      s1 = vld1q_u16(src16_ptr + 8);
      r1 = vld1q_u16(ref16_ptr + 8);
      sum_u16[1] = vabaq_u16(sum_u16[1], s1, r1);

      src16_ptr += src_stride;
      ref16_ptr += ref_stride;
    } while (--i != 0);

    sum_u32 = vpadalq_u16(sum_u32, sum_u16[0]);
    sum_u32 = vpadalq_u16(sum_u32, sum_u16[1]);
    h -= h_limit;
  } while (h != 0);
  return horizontal_add_uint32x4(sum_u32);
}

static INLINE uint32_t highbd_sadwxh_neon(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *ref_ptr,
                                          int ref_stride, int w, int h,
                                          const int h_overflow) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr = CONVERT_TO_SHORTPTR(ref_ptr);
  uint32x4_t sum_u32 = vdupq_n_u32(0);
  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % h_limit == 0);

  do {
    uint16x8_t sum_u16[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                              vdupq_n_u16(0) };
    int i = 0;
    do {
      int j = 0;
      do {
        uint16x8_t s0, s1, s2, s3, r0, r1, r2, r3;

        s0 = vld1q_u16(src16_ptr + j);
        r0 = vld1q_u16(ref16_ptr + j);
        sum_u16[0] = vabaq_u16(sum_u16[0], s0, r0);

        s1 = vld1q_u16(src16_ptr + j + 8);
        r1 = vld1q_u16(ref16_ptr + j + 8);
        sum_u16[1] = vabaq_u16(sum_u16[1], s1, r1);

        s2 = vld1q_u16(src16_ptr + j + 16);
        r2 = vld1q_u16(ref16_ptr + j + 16);
        sum_u16[2] = vabaq_u16(sum_u16[2], s2, r2);

        s3 = vld1q_u16(src16_ptr + j + 24);
        r3 = vld1q_u16(ref16_ptr + j + 24);
        sum_u16[3] = vabaq_u16(sum_u16[3], s3, r3);

        j += 32;
      } while (j < w);
      src16_ptr += src_stride;
      ref16_ptr += ref_stride;
    } while (++i != h_limit);

    sum_u32 = vpadalq_u16(sum_u32, sum_u16[0]);
    sum_u32 = vpadalq_u16(sum_u32, sum_u16[1]);
    sum_u32 = vpadalq_u16(sum_u32, sum_u16[2]);
    sum_u32 = vpadalq_u16(sum_u32, sum_u16[3]);
    h -= h_limit;
  } while (h != 0);
  return horizontal_add_uint32x4(sum_u32);
}

static INLINE uint32_t highbd_sad32xh_neon(const uint8_t *src_ptr,
                                           int src_stride,
                                           const uint8_t *ref_ptr,
                                           int ref_stride, int h) {
  // 'h_overflow' is the number of 32-wide rows we can process before 16-bit
  // accumulators overflow. After hitting this limit accumulate into 32-bit
  // elements. 65535 / 4095 ~= 16, so 16 32-wide rows using four accumulators.
  const int h_overflow = 16;
  return highbd_sadwxh_neon(src_ptr, src_stride, ref_ptr, ref_stride, 32, h,
                            h_overflow);
}

static INLINE uint32_t highbd_sad64xh_neon(const uint8_t *src_ptr,
                                           int src_stride,
                                           const uint8_t *ref_ptr,
                                           int ref_stride, int h) {
  // 'h_overflow' is the number of 64-wide rows we can process before 16-bit
  // accumulators overflow. After hitting this limit accumulate into 32-bit
  // elements. 65535 / 4095 ~= 16, so 8 64-wide rows using four accumulators.
  const int h_overflow = 8;
  return highbd_sadwxh_neon(src_ptr, src_stride, ref_ptr, ref_stride, 64, h,
                            h_overflow);
}

#define HBD_SAD_WXH_NEON(w, h)                                            \
  unsigned int vpx_highbd_sad##w##x##h##_neon(                            \
      const uint8_t *src, int src_stride, const uint8_t *ref,             \
      int ref_stride) {                                                   \
    return highbd_sad##w##xh_neon(src, src_stride, ref, ref_stride, (h)); \
  }

HBD_SAD_WXH_NEON(4, 4)
HBD_SAD_WXH_NEON(4, 8)

HBD_SAD_WXH_NEON(8, 4)
HBD_SAD_WXH_NEON(8, 8)
HBD_SAD_WXH_NEON(8, 16)

HBD_SAD_WXH_NEON(16, 8)
HBD_SAD_WXH_NEON(16, 16)
HBD_SAD_WXH_NEON(16, 32)

HBD_SAD_WXH_NEON(32, 16)
HBD_SAD_WXH_NEON(32, 32)
HBD_SAD_WXH_NEON(32, 64)

HBD_SAD_WXH_NEON(64, 32)
HBD_SAD_WXH_NEON(64, 64)

#undef HBD_SAD_WXH_NEON

#define HBD_SAD_SKIP_WXH_NEON(w, h)                             \
  unsigned int vpx_highbd_sad_skip_##w##x##h##_neon(            \
      const uint8_t *src, int src_stride, const uint8_t *ref,   \
      int ref_stride) {                                         \
    return 2 * highbd_sad##w##xh_neon(src, 2 * src_stride, ref, \
                                      2 * ref_stride, (h) / 2); \
  }

HBD_SAD_SKIP_WXH_NEON(4, 4)
HBD_SAD_SKIP_WXH_NEON(4, 8)

HBD_SAD_SKIP_WXH_NEON(8, 4)
HBD_SAD_SKIP_WXH_NEON(8, 8)
HBD_SAD_SKIP_WXH_NEON(8, 16)

HBD_SAD_SKIP_WXH_NEON(16, 8)
HBD_SAD_SKIP_WXH_NEON(16, 16)
HBD_SAD_SKIP_WXH_NEON(16, 32)

HBD_SAD_SKIP_WXH_NEON(32, 16)
HBD_SAD_SKIP_WXH_NEON(32, 32)
HBD_SAD_SKIP_WXH_NEON(32, 64)

HBD_SAD_SKIP_WXH_NEON(64, 32)
HBD_SAD_SKIP_WXH_NEON(64, 64)

#undef HBD_SAD_SKIP_WXH_NEON

static INLINE uint32_t highbd_sad4xh_avg_neon(const uint8_t *src_ptr,
                                              int src_stride,
                                              const uint8_t *ref_ptr,
                                              int ref_stride, int h,
                                              const uint8_t *second_pred) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr = CONVERT_TO_SHORTPTR(ref_ptr);
  const uint16_t *pred16_ptr = CONVERT_TO_SHORTPTR(second_pred);
  uint32x4_t sum = vdupq_n_u32(0);

  do {
    uint16x4_t s = vld1_u16(src16_ptr);
    uint16x4_t r = vld1_u16(ref16_ptr);
    uint16x4_t p = vld1_u16(pred16_ptr);

    uint16x4_t avg = vrhadd_u16(r, p);
    sum = vabal_u16(sum, s, avg);

    src16_ptr += src_stride;
    ref16_ptr += ref_stride;
    pred16_ptr += 4;
  } while (--h != 0);

  return horizontal_add_uint32x4(sum);
}

static INLINE uint32_t highbd_sad8xh_avg_neon(const uint8_t *src_ptr,
                                              int src_stride,
                                              const uint8_t *ref_ptr,
                                              int ref_stride, int h,
                                              const uint8_t *second_pred) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr = CONVERT_TO_SHORTPTR(ref_ptr);
  const uint16_t *pred16_ptr = CONVERT_TO_SHORTPTR(second_pred);
  uint16x8_t sum = vdupq_n_u16(0);
  assert(h <= 16);

  do {
    uint16x8_t s = vld1q_u16(src16_ptr);
    uint16x8_t r = vld1q_u16(ref16_ptr);
    uint16x8_t p = vld1q_u16(pred16_ptr);

    uint16x8_t avg = vrhaddq_u16(r, p);
    sum = vabaq_u16(sum, s, avg);

    src16_ptr += src_stride;
    ref16_ptr += ref_stride;
    pred16_ptr += 8;
  } while (--h != 0);

  return horizontal_add_uint16x8(sum);
}

static INLINE uint32_t highbd_sad16xh_avg_neon(const uint8_t *src_ptr,
                                               int src_stride,
                                               const uint8_t *ref_ptr,
                                               int ref_stride, int h,
                                               const uint8_t *second_pred) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr = CONVERT_TO_SHORTPTR(ref_ptr);
  const uint16_t *pred16_ptr = CONVERT_TO_SHORTPTR(second_pred);
  uint32x4_t sum_u32 = vdupq_n_u32(0);

  // 'h_overflow' is the number of 16-wide rows we can process before 16-bit
  // accumulators overflow. After hitting this limit accumulate into 32-bit
  // elements. 65535 / 4095 ~= 16, so 16 16-wide rows using two accumulators.
  const int h_overflow = 16;
  // If block height 'h' is smaller than this limit, use 'h' instead.
  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % h_limit == 0);

  do {
    uint16x8_t sum_u16[2] = { vdupq_n_u16(0), vdupq_n_u16(0) };
    int i = h_limit;
    do {
      uint16x8_t s0, s1, r0, r1, p0, p1;
      uint16x8_t avg0, avg1;

      s0 = vld1q_u16(src16_ptr);
      r0 = vld1q_u16(ref16_ptr);
      p0 = vld1q_u16(pred16_ptr);
      avg0 = vrhaddq_u16(r0, p0);
      sum_u16[0] = vabaq_u16(sum_u16[0], s0, avg0);

      s1 = vld1q_u16(src16_ptr + 8);
      r1 = vld1q_u16(ref16_ptr + 8);
      p1 = vld1q_u16(pred16_ptr + 8);
      avg1 = vrhaddq_u16(r1, p1);
      sum_u16[1] = vabaq_u16(sum_u16[1], s1, avg1);

      src16_ptr += src_stride;
      ref16_ptr += ref_stride;
      pred16_ptr += 16;
    } while (--i != 0);

    sum_u32 = vpadalq_u16(sum_u32, sum_u16[0]);
    sum_u32 = vpadalq_u16(sum_u32, sum_u16[1]);
    h -= h_limit;
  } while (h != 0);
  return horizontal_add_uint32x4(sum_u32);
}

static INLINE uint32_t highbd_sadwxh_avg_neon(const uint8_t *src_ptr,
                                              int src_stride,
                                              const uint8_t *ref_ptr,
                                              int ref_stride, int w, int h,
                                              const uint8_t *second_pred,
                                              const int h_overflow) {
  const uint16_t *src16_ptr = CONVERT_TO_SHORTPTR(src_ptr);
  const uint16_t *ref16_ptr = CONVERT_TO_SHORTPTR(ref_ptr);
  const uint16_t *pred16_ptr = CONVERT_TO_SHORTPTR(second_pred);
  uint32x4_t sum_u32 = vdupq_n_u32(0);
  const int h_limit = h < h_overflow ? h : h_overflow;
  assert(h % h_limit == 0);

  do {
    uint16x8_t sum_u16[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0),
                              vdupq_n_u16(0) };

    int i = h_limit;
    do {
      int j = 0;
      do {
        uint16x8_t s0, s1, s2, s3, r0, r1, r2, r3, p0, p1, p2, p3;
        uint16x8_t avg0, avg1, avg2, avg3;

        s0 = vld1q_u16(src16_ptr + j);
        r0 = vld1q_u16(ref16_ptr + j);
        p0 = vld1q_u16(pred16_ptr + j);
        avg0 = vrhaddq_u16(r0, p0);
        sum_u16[0] = vabaq_u16(sum_u16[0], s0, avg0);

        s1 = vld1q_u16(src16_ptr + j + 8);
        r1 = vld1q_u16(ref16_ptr + j + 8);
        p1 = vld1q_u16(pred16_ptr + j + 8);
        avg1 = vrhaddq_u16(r1, p1);
        sum_u16[1] = vabaq_u16(sum_u16[1], s1, avg1);

        s2 = vld1q_u16(src16_ptr + j + 16);
        r2 = vld1q_u16(ref16_ptr + j + 16);
        p2 = vld1q_u16(pred16_ptr + j + 16);
        avg2 = vrhaddq_u16(r2, p2);
        sum_u16[2] = vabaq_u16(sum_u16[2], s2, avg2);

        s3 = vld1q_u16(src16_ptr + j + 24);
        r3 = vld1q_u16(ref16_ptr + j + 24);
        p3 = vld1q_u16(pred16_ptr + j + 24);
        avg3 = vrhaddq_u16(r3, p3);
        sum_u16[3] = vabaq_u16(sum_u16[3], s3, avg3);

        j += 32;
      } while (j < w);

      src16_ptr += src_stride;
      ref16_ptr += ref_stride;
      pred16_ptr += w;
    } while (--i != 0);

    sum_u32 = vpadalq_u16(sum_u32, sum_u16[0]);
    sum_u32 = vpadalq_u16(sum_u32, sum_u16[1]);
    sum_u32 = vpadalq_u16(sum_u32, sum_u16[2]);
    sum_u32 = vpadalq_u16(sum_u32, sum_u16[3]);
    h -= h_limit;
  } while (h != 0);
  return horizontal_add_uint32x4(sum_u32);
}

static INLINE uint32_t highbd_sad32xh_avg_neon(const uint8_t *src_ptr,
                                               int src_stride,
                                               const uint8_t *ref_ptr,
                                               int ref_stride, int h,
                                               const uint8_t *second_pred) {
  // 'h_overflow' is the number of 32-wide rows we can process before 16-bit
  // accumulators overflow. After hitting this limit accumulate into 32-bit
  // elements. 65535 / 4095 ~= 16, so 16 32-wide rows using four accumulators.
  const int h_overflow = 16;
  return highbd_sadwxh_avg_neon(src_ptr, src_stride, ref_ptr, ref_stride, 32, h,
                                second_pred, h_overflow);
}

static INLINE uint32_t highbd_sad64xh_avg_neon(const uint8_t *src_ptr,
                                               int src_stride,
                                               const uint8_t *ref_ptr,
                                               int ref_stride, int h,
                                               const uint8_t *second_pred) {
  // 'h_overflow' is the number of 64-wide rows we can process before 16-bit
  // accumulators overflow. After hitting this limit accumulate into 32-bit
  // elements. 65535 / 4095 ~= 16, so 8 64-wide rows using four accumulators.
  const int h_overflow = 8;
  return highbd_sadwxh_avg_neon(src_ptr, src_stride, ref_ptr, ref_stride, 64, h,
                                second_pred, h_overflow);
}

#define HBD_SAD_WXH_AVG_NEON(w, h)                                            \
  unsigned int vpx_highbd_sad##w##x##h##_avg_neon(                            \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      const uint8_t *second_pred) {                                           \
    return highbd_sad##w##xh_avg_neon(src, src_stride, ref, ref_stride, (h),  \
                                      second_pred);                           \
  }

HBD_SAD_WXH_AVG_NEON(4, 4)
HBD_SAD_WXH_AVG_NEON(4, 8)

HBD_SAD_WXH_AVG_NEON(8, 4)
HBD_SAD_WXH_AVG_NEON(8, 8)
HBD_SAD_WXH_AVG_NEON(8, 16)

HBD_SAD_WXH_AVG_NEON(16, 8)
HBD_SAD_WXH_AVG_NEON(16, 16)
HBD_SAD_WXH_AVG_NEON(16, 32)

HBD_SAD_WXH_AVG_NEON(32, 16)
HBD_SAD_WXH_AVG_NEON(32, 32)
HBD_SAD_WXH_AVG_NEON(32, 64)

HBD_SAD_WXH_AVG_NEON(64, 32)
HBD_SAD_WXH_AVG_NEON(64, 64)

#undef HBD_SAD_WXH_AVG_NEON
