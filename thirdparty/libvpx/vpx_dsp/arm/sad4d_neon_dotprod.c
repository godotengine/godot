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

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"

static INLINE void sad16_neon(uint8x16_t src, uint8x16_t ref,
                              uint32x4_t *const sad_sum) {
  uint8x16_t abs_diff = vabdq_u8(src, ref);
  *sad_sum = vdotq_u32(*sad_sum, abs_diff, vdupq_n_u8(1));
}

static INLINE void sad64xhx4d_neon_dotprod(const uint8_t *src, int src_stride,
                                           const uint8_t *const ref[4],
                                           int ref_stride, uint32_t res[4],
                                           int h) {
  uint32x4_t sum_lo[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };
  uint32x4_t sum_hi[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };
  uint32x4_t sum[4];

  int i = 0;
  do {
    uint8x16_t s0, s1, s2, s3;

    s0 = vld1q_u8(src + i * src_stride);
    sad16_neon(s0, vld1q_u8(ref[0] + i * ref_stride), &sum_lo[0]);
    sad16_neon(s0, vld1q_u8(ref[1] + i * ref_stride), &sum_lo[1]);
    sad16_neon(s0, vld1q_u8(ref[2] + i * ref_stride), &sum_lo[2]);
    sad16_neon(s0, vld1q_u8(ref[3] + i * ref_stride), &sum_lo[3]);

    s1 = vld1q_u8(src + i * src_stride + 16);
    sad16_neon(s1, vld1q_u8(ref[0] + i * ref_stride + 16), &sum_hi[0]);
    sad16_neon(s1, vld1q_u8(ref[1] + i * ref_stride + 16), &sum_hi[1]);
    sad16_neon(s1, vld1q_u8(ref[2] + i * ref_stride + 16), &sum_hi[2]);
    sad16_neon(s1, vld1q_u8(ref[3] + i * ref_stride + 16), &sum_hi[3]);

    s2 = vld1q_u8(src + i * src_stride + 32);
    sad16_neon(s2, vld1q_u8(ref[0] + i * ref_stride + 32), &sum_lo[0]);
    sad16_neon(s2, vld1q_u8(ref[1] + i * ref_stride + 32), &sum_lo[1]);
    sad16_neon(s2, vld1q_u8(ref[2] + i * ref_stride + 32), &sum_lo[2]);
    sad16_neon(s2, vld1q_u8(ref[3] + i * ref_stride + 32), &sum_lo[3]);

    s3 = vld1q_u8(src + i * src_stride + 48);
    sad16_neon(s3, vld1q_u8(ref[0] + i * ref_stride + 48), &sum_hi[0]);
    sad16_neon(s3, vld1q_u8(ref[1] + i * ref_stride + 48), &sum_hi[1]);
    sad16_neon(s3, vld1q_u8(ref[2] + i * ref_stride + 48), &sum_hi[2]);
    sad16_neon(s3, vld1q_u8(ref[3] + i * ref_stride + 48), &sum_hi[3]);

  } while (++i < h);

  sum[0] = vaddq_u32(sum_lo[0], sum_hi[0]);
  sum[1] = vaddq_u32(sum_lo[1], sum_hi[1]);
  sum[2] = vaddq_u32(sum_lo[2], sum_hi[2]);
  sum[3] = vaddq_u32(sum_lo[3], sum_hi[3]);

  vst1q_u32(res, horizontal_add_4d_uint32x4(sum));
}

static INLINE void sad32xhx4d_neon_dotprod(const uint8_t *src, int src_stride,
                                           const uint8_t *const ref[4],
                                           int ref_stride, uint32_t res[4],
                                           int h) {
  uint32x4_t sum_lo[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };
  uint32x4_t sum_hi[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                           vdupq_n_u32(0) };
  uint32x4_t sum[4];

  int i = 0;
  do {
    uint8x16_t s0, s1;

    s0 = vld1q_u8(src + i * src_stride);
    sad16_neon(s0, vld1q_u8(ref[0] + i * ref_stride), &sum_lo[0]);
    sad16_neon(s0, vld1q_u8(ref[1] + i * ref_stride), &sum_lo[1]);
    sad16_neon(s0, vld1q_u8(ref[2] + i * ref_stride), &sum_lo[2]);
    sad16_neon(s0, vld1q_u8(ref[3] + i * ref_stride), &sum_lo[3]);

    s1 = vld1q_u8(src + i * src_stride + 16);
    sad16_neon(s1, vld1q_u8(ref[0] + i * ref_stride + 16), &sum_hi[0]);
    sad16_neon(s1, vld1q_u8(ref[1] + i * ref_stride + 16), &sum_hi[1]);
    sad16_neon(s1, vld1q_u8(ref[2] + i * ref_stride + 16), &sum_hi[2]);
    sad16_neon(s1, vld1q_u8(ref[3] + i * ref_stride + 16), &sum_hi[3]);

  } while (++i < h);

  sum[0] = vaddq_u32(sum_lo[0], sum_hi[0]);
  sum[1] = vaddq_u32(sum_lo[1], sum_hi[1]);
  sum[2] = vaddq_u32(sum_lo[2], sum_hi[2]);
  sum[3] = vaddq_u32(sum_lo[3], sum_hi[3]);

  vst1q_u32(res, horizontal_add_4d_uint32x4(sum));
}

static INLINE void sad16xhx4d_neon_dotprod(const uint8_t *src, int src_stride,
                                           const uint8_t *const ref[4],
                                           int ref_stride, uint32_t res[4],
                                           int h) {
  uint32x4_t sum[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0),
                        vdupq_n_u32(0) };

  int i = 0;
  do {
    const uint8x16_t s = vld1q_u8(src + i * src_stride);
    sad16_neon(s, vld1q_u8(ref[0] + i * ref_stride), &sum[0]);
    sad16_neon(s, vld1q_u8(ref[1] + i * ref_stride), &sum[1]);
    sad16_neon(s, vld1q_u8(ref[2] + i * ref_stride), &sum[2]);
    sad16_neon(s, vld1q_u8(ref[3] + i * ref_stride), &sum[3]);

  } while (++i < h);

  vst1q_u32(res, horizontal_add_4d_uint32x4(sum));
}

#define SAD_WXH_4D_NEON_DOTPROD(w, h)                                      \
  void vpx_sad##w##x##h##x4d_neon_dotprod(                                 \
      const uint8_t *src_ptr, int src_stride,                              \
      const uint8_t *const ref_array[4], int ref_stride,                   \
      uint32_t sad_array[4]) {                                             \
    sad##w##xhx4d_neon_dotprod(src_ptr, src_stride, ref_array, ref_stride, \
                               sad_array, (h));                            \
  }

SAD_WXH_4D_NEON_DOTPROD(16, 8)
SAD_WXH_4D_NEON_DOTPROD(16, 16)
SAD_WXH_4D_NEON_DOTPROD(16, 32)

SAD_WXH_4D_NEON_DOTPROD(32, 16)
SAD_WXH_4D_NEON_DOTPROD(32, 32)
SAD_WXH_4D_NEON_DOTPROD(32, 64)

SAD_WXH_4D_NEON_DOTPROD(64, 32)
SAD_WXH_4D_NEON_DOTPROD(64, 64)

#undef SAD_WXH_4D_NEON_DOTPROD

#define SAD_SKIP_WXH_4D_NEON_DOTPROD(w, h)                             \
  void vpx_sad_skip_##w##x##h##x4d_neon_dotprod(                       \
      const uint8_t *src_ptr, int src_stride,                          \
      const uint8_t *const ref_array[4], int ref_stride,               \
      uint32_t sad_array[4]) {                                         \
    sad##w##xhx4d_neon_dotprod(src_ptr, 2 * src_stride, ref_array,     \
                               2 * ref_stride, sad_array, ((h) >> 1)); \
    sad_array[0] <<= 1;                                                \
    sad_array[1] <<= 1;                                                \
    sad_array[2] <<= 1;                                                \
    sad_array[3] <<= 1;                                                \
  }

SAD_SKIP_WXH_4D_NEON_DOTPROD(16, 8)
SAD_SKIP_WXH_4D_NEON_DOTPROD(16, 16)
SAD_SKIP_WXH_4D_NEON_DOTPROD(16, 32)

SAD_SKIP_WXH_4D_NEON_DOTPROD(32, 16)
SAD_SKIP_WXH_4D_NEON_DOTPROD(32, 32)
SAD_SKIP_WXH_4D_NEON_DOTPROD(32, 64)

SAD_SKIP_WXH_4D_NEON_DOTPROD(64, 32)
SAD_SKIP_WXH_4D_NEON_DOTPROD(64, 64)

#undef SAD_SKIP_WXH_4D_NEON_DOTPROD
