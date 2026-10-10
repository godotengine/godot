/*
 * Copyright (c) 2024 The WebM project authors. All Rights Reserved.
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS.  All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vpx_dsp_rtcd.h"
#include "./vpx_config.h"

#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"
#include "vpx_dsp/arm/vpx_neon_sve_bridge.h"
#include "vpx_ports/mem.h"

static INLINE uint32_t highbd_mse_wxh_sve(const uint16_t *src_ptr,
                                          int src_stride,
                                          const uint16_t *ref_ptr,
                                          int ref_stride, int w, int h) {
  uint64x2_t sse = vdupq_n_u64(0);

  do {
    int j = 0;
    do {
      uint16x8_t s = vld1q_u16(src_ptr + j);
      uint16x8_t r = vld1q_u16(ref_ptr + j);

      uint16x8_t diff = vabdq_u16(s, r);

      sse = vpx_dotq_u16(sse, diff, diff);

      j += 8;
    } while (j < w);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--h != 0);

  return (uint32_t)horizontal_add_uint64x2(sse);
}

#define HIGHBD_MSE_WXH_SVE(w, h)                                      \
  uint32_t vpx_highbd_10_mse##w##x##h##_sve(                          \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse) {                                \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    uint32_t sse_tmp =                                                \
        highbd_mse_wxh_sve(src, src_stride, ref, ref_stride, w, h);   \
    sse_tmp = ROUND_POWER_OF_TWO(sse_tmp, 4);                         \
    *sse = sse_tmp;                                                   \
    return sse_tmp;                                                   \
  }                                                                   \
                                                                      \
  uint32_t vpx_highbd_12_mse##w##x##h##_sve(                          \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse) {                                \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    uint32_t sse_tmp =                                                \
        highbd_mse_wxh_sve(src, src_stride, ref, ref_stride, w, h);   \
    sse_tmp = ROUND_POWER_OF_TWO(sse_tmp, 8);                         \
    *sse = sse_tmp;                                                   \
    return sse_tmp;                                                   \
  }

HIGHBD_MSE_WXH_SVE(16, 16)
HIGHBD_MSE_WXH_SVE(16, 8)
HIGHBD_MSE_WXH_SVE(8, 16)
HIGHBD_MSE_WXH_SVE(8, 8)

#undef HIGHBD_MSE_WXH_SVE

// Process a block of width 4 two rows at a time.
static INLINE void highbd_variance_4xh_sve(const uint16_t *src_ptr,
                                           int src_stride,
                                           const uint16_t *ref_ptr,
                                           int ref_stride, int h, uint64_t *sse,
                                           int64_t *sum) {
  int16x8_t sum_s16 = vdupq_n_s16(0);
  int64x2_t sse_s64 = vdupq_n_s64(0);

  do {
    const uint16x8_t s = load_unaligned_u16q(src_ptr, src_stride);
    const uint16x8_t r = load_unaligned_u16q(ref_ptr, ref_stride);

    int16x8_t diff = vreinterpretq_s16_u16(vsubq_u16(s, r));
    sum_s16 = vaddq_s16(sum_s16, diff);
    sse_s64 = vpx_dotq_s16(sse_s64, diff, diff);

    src_ptr += 2 * src_stride;
    ref_ptr += 2 * ref_stride;
    h -= 2;
  } while (h != 0);

  *sum = horizontal_add_int16x8(sum_s16);
  *sse = horizontal_add_int64x2(sse_s64);
}

static INLINE void highbd_variance_8xh_sve(const uint16_t *src_ptr,
                                           int src_stride,
                                           const uint16_t *ref_ptr,
                                           int ref_stride, int h, uint64_t *sse,
                                           int64_t *sum) {
  int32x4_t sum_s32 = vdupq_n_s32(0);
  int64x2_t sse_s64 = vdupq_n_s64(0);

  do {
    const uint16x8_t s = vld1q_u16(src_ptr);
    const uint16x8_t r = vld1q_u16(ref_ptr);

    const int16x8_t diff = vreinterpretq_s16_u16(vsubq_u16(s, r));
    sum_s32 = vpadalq_s16(sum_s32, diff);
    sse_s64 = vpx_dotq_s16(sse_s64, diff, diff);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--h != 0);

  *sum = horizontal_add_int32x4(sum_s32);
  *sse = horizontal_add_int64x2(sse_s64);
}

static INLINE void highbd_variance_16xh_sve(const uint16_t *src_ptr,
                                            int src_stride,
                                            const uint16_t *ref_ptr,
                                            int ref_stride, int h,
                                            uint64_t *sse, int64_t *sum) {
  int32x4_t sum_s32[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
  int64x2_t sse_s64[2] = { vdupq_n_s64(0), vdupq_n_s64(0) };

  do {
    const uint16x8_t s0 = vld1q_u16(src_ptr);
    const uint16x8_t s1 = vld1q_u16(src_ptr + 8);

    const uint16x8_t r0 = vld1q_u16(ref_ptr);
    const uint16x8_t r1 = vld1q_u16(ref_ptr + 8);

    const int16x8_t diff0 = vreinterpretq_s16_u16(vsubq_u16(s0, r0));
    const int16x8_t diff1 = vreinterpretq_s16_u16(vsubq_u16(s1, r1));

    sum_s32[0] = vpadalq_s16(sum_s32[0], diff0);
    sum_s32[1] = vpadalq_s16(sum_s32[1], diff1);

    sse_s64[0] = vpx_dotq_s16(sse_s64[0], diff0, diff0);
    sse_s64[1] = vpx_dotq_s16(sse_s64[1], diff1, diff1);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--h != 0);

  sum_s32[0] = vaddq_s32(sum_s32[0], sum_s32[1]);
  sse_s64[0] = vaddq_s64(sse_s64[0], sse_s64[1]);

  *sum = horizontal_add_int32x4(sum_s32[0]);
  *sse = horizontal_add_int64x2(sse_s64[0]);
}

static INLINE void highbd_variance_wxh_sve(const uint16_t *src_ptr,
                                           int src_stride,
                                           const uint16_t *ref_ptr,
                                           int ref_stride, int w, int h,
                                           uint64_t *sse, int64_t *sum) {
  int32x4_t sum_s32[4] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0),
                           vdupq_n_s32(0) };
  int64x2_t sse_s64[4] = { vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0),
                           vdupq_n_s64(0) };

  do {
    int i = 0;
    do {
      const uint16x8_t s0 = vld1q_u16(src_ptr + i);
      const uint16x8_t s1 = vld1q_u16(src_ptr + i + 8);
      const uint16x8_t s2 = vld1q_u16(src_ptr + i + 16);
      const uint16x8_t s3 = vld1q_u16(src_ptr + i + 24);

      const uint16x8_t r0 = vld1q_u16(ref_ptr + i);
      const uint16x8_t r1 = vld1q_u16(ref_ptr + i + 8);
      const uint16x8_t r2 = vld1q_u16(ref_ptr + i + 16);
      const uint16x8_t r3 = vld1q_u16(ref_ptr + i + 24);

      const int16x8_t diff0 = vreinterpretq_s16_u16(vsubq_u16(s0, r0));
      const int16x8_t diff1 = vreinterpretq_s16_u16(vsubq_u16(s1, r1));
      const int16x8_t diff2 = vreinterpretq_s16_u16(vsubq_u16(s2, r2));
      const int16x8_t diff3 = vreinterpretq_s16_u16(vsubq_u16(s3, r3));

      sum_s32[0] = vpadalq_s16(sum_s32[0], diff0);
      sum_s32[1] = vpadalq_s16(sum_s32[1], diff1);
      sum_s32[2] = vpadalq_s16(sum_s32[2], diff2);
      sum_s32[3] = vpadalq_s16(sum_s32[3], diff3);

      sse_s64[0] = vpx_dotq_s16(sse_s64[0], diff0, diff0);
      sse_s64[1] = vpx_dotq_s16(sse_s64[1], diff1, diff1);
      sse_s64[2] = vpx_dotq_s16(sse_s64[2], diff2, diff2);
      sse_s64[3] = vpx_dotq_s16(sse_s64[3], diff3, diff3);

      i += 32;
    } while (i < w);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--h != 0);

  sum_s32[0] = vaddq_s32(sum_s32[0], sum_s32[1]);
  sum_s32[2] = vaddq_s32(sum_s32[2], sum_s32[3]);
  sum_s32[0] = vaddq_s32(sum_s32[0], sum_s32[2]);

  sse_s64[0] = vaddq_s64(sse_s64[0], sse_s64[1]);
  sse_s64[2] = vaddq_s64(sse_s64[2], sse_s64[3]);
  sse_s64[0] = vaddq_s64(sse_s64[0], sse_s64[2]);

  *sum = horizontal_add_int32x4(sum_s32[0]);
  *sse = horizontal_add_int64x2(sse_s64[0]);
}

static INLINE void highbd_variance_32xh_sve(const uint16_t *src, int src_stride,
                                            const uint16_t *ref, int ref_stride,
                                            int h, uint64_t *sse,
                                            int64_t *sum) {
  highbd_variance_wxh_sve(src, src_stride, ref, ref_stride, 32, h, sse, sum);
}

static INLINE void highbd_variance_64xh_sve(const uint16_t *src, int src_stride,
                                            const uint16_t *ref, int ref_stride,
                                            int h, uint64_t *sse,
                                            int64_t *sum) {
  highbd_variance_wxh_sve(src, src_stride, ref, ref_stride, 64, h, sse, sum);
}

#define HBD_VARIANCE_WXH_SVE(w, h)                                    \
  uint32_t vpx_highbd_8_variance##w##x##h##_sve(                      \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse) {                                \
    int sum;                                                          \
    uint64_t sse_long = 0;                                            \
    int64_t sum_long = 0;                                             \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    highbd_variance_##w##xh_sve(src, src_stride, ref, ref_stride, h,  \
                                &sse_long, &sum_long);                \
    *sse = (uint32_t)sse_long;                                        \
    sum = (int)sum_long;                                              \
    return *sse - (uint32_t)(((int64_t)sum * sum) / (w * h));         \
  }                                                                   \
                                                                      \
  uint32_t vpx_highbd_10_variance##w##x##h##_sve(                     \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse) {                                \
    int sum;                                                          \
    int64_t var;                                                      \
    uint64_t sse_long = 0;                                            \
    int64_t sum_long = 0;                                             \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    highbd_variance_##w##xh_sve(src, src_stride, ref, ref_stride, h,  \
                                &sse_long, &sum_long);                \
    *sse = (uint32_t)ROUND_POWER_OF_TWO(sse_long, 4);                 \
    sum = (int)ROUND_POWER_OF_TWO(sum_long, 2);                       \
    var = (int64_t)(*sse) - (((int64_t)sum * sum) / (w * h));         \
    return (var >= 0) ? (uint32_t)var : 0;                            \
  }                                                                   \
                                                                      \
  uint32_t vpx_highbd_12_variance##w##x##h##_sve(                     \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse) {                                \
    int sum;                                                          \
    int64_t var;                                                      \
    uint64_t sse_long = 0;                                            \
    int64_t sum_long = 0;                                             \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    highbd_variance_##w##xh_sve(src, src_stride, ref, ref_stride, h,  \
                                &sse_long, &sum_long);                \
    *sse = (uint32_t)ROUND_POWER_OF_TWO(sse_long, 8);                 \
    sum = (int)ROUND_POWER_OF_TWO(sum_long, 4);                       \
    var = (int64_t)(*sse) - (((int64_t)sum * sum) / (w * h));         \
    return (var >= 0) ? (uint32_t)var : 0;                            \
  }

HBD_VARIANCE_WXH_SVE(4, 4)
HBD_VARIANCE_WXH_SVE(4, 8)

HBD_VARIANCE_WXH_SVE(8, 4)
HBD_VARIANCE_WXH_SVE(8, 8)
HBD_VARIANCE_WXH_SVE(8, 16)

HBD_VARIANCE_WXH_SVE(16, 8)
HBD_VARIANCE_WXH_SVE(16, 16)
HBD_VARIANCE_WXH_SVE(16, 32)

HBD_VARIANCE_WXH_SVE(32, 16)
HBD_VARIANCE_WXH_SVE(32, 32)
HBD_VARIANCE_WXH_SVE(32, 64)

HBD_VARIANCE_WXH_SVE(64, 32)
HBD_VARIANCE_WXH_SVE(64, 64)

#define HIGHBD_GET_VAR_SVE(s)                                         \
  void vpx_highbd_8_get##s##x##s##var_sve(                            \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse, int *sum) {                      \
    uint64_t sse_long = 0;                                            \
    int64_t sum_long = 0;                                             \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    highbd_variance_##s##xh_sve(src, src_stride, ref, ref_stride, s,  \
                                &sse_long, &sum_long);                \
    *sse = (uint32_t)sse_long;                                        \
    *sum = (int)sum_long;                                             \
  }                                                                   \
                                                                      \
  void vpx_highbd_10_get##s##x##s##var_sve(                           \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse, int *sum) {                      \
    uint64_t sse_long = 0;                                            \
    int64_t sum_long = 0;                                             \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    highbd_variance_##s##xh_sve(src, src_stride, ref, ref_stride, s,  \
                                &sse_long, &sum_long);                \
    *sse = (uint32_t)ROUND_POWER_OF_TWO(sse_long, 4);                 \
    *sum = (int)ROUND_POWER_OF_TWO(sum_long, 2);                      \
  }                                                                   \
                                                                      \
  void vpx_highbd_12_get##s##x##s##var_sve(                           \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, \
      int ref_stride, uint32_t *sse, int *sum) {                      \
    uint64_t sse_long = 0;                                            \
    int64_t sum_long = 0;                                             \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                     \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                     \
    highbd_variance_##s##xh_sve(src, src_stride, ref, ref_stride, s,  \
                                &sse_long, &sum_long);                \
    *sse = (uint32_t)ROUND_POWER_OF_TWO(sse_long, 8);                 \
    *sum = (int)ROUND_POWER_OF_TWO(sum_long, 4);                      \
  }

HIGHBD_GET_VAR_SVE(8)
HIGHBD_GET_VAR_SVE(16)
