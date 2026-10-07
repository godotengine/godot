/*
 *  Copyright (c) 2026 The WebM project authors. All Rights Reserved.
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

#include "vpx_dsp/arm/subpel_variance_neon.h"

static inline void bil_variance_8xh_neon_dotprod(
    const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride,
    int pixel_step, int h, uint32_t *sse, int *sum, int filter_offset) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  uint32x4_t src_sum = vdupq_n_u32(0);
  uint32x4_t ref_sum = vdupq_n_u32(0);
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  do {
    uint8x8_t s0_lo = vld1_u8(src);
    uint8x8_t s1_lo = vld1_u8(src + pixel_step);
    uint16x8_t blend_l = vmull_u8(s0_lo, f0);
    blend_l = vmlal_u8(blend_l, s1_lo, f1);
    uint8x8_t s0_hi = vld1_u8(src + src_stride);
    uint8x8_t s1_hi = vld1_u8(src + src_stride + pixel_step);
    uint16x8_t blend_h = vmull_u8(s0_hi, f0);
    blend_h = vmlal_u8(blend_h, s1_hi, f1);
    uint8x16_t s =
        vcombine_u8(vrshrn_n_u16(blend_l, 3), vrshrn_n_u16(blend_h, 3));
    uint8x16_t r = load_u8_8x2(ref, ref_stride);

    src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
    ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

    uint8x16_t abs_diff = vabdq_u8(s, r);
    sse_u32 = vdotq_u32(sse_u32, abs_diff, abs_diff);

    src += 2 * src_stride;
    ref += 2 * ref_stride;
    h -= 2;
  } while (h != 0);

  int32x4_t sum_diff =
      vsubq_s32(vreinterpretq_s32_u32(src_sum), vreinterpretq_s32_u32(ref_sum));
  *sum = vaddvq_s32(sum_diff);
  *sse = vaddvq_u32(sse_u32);
}

static inline void bil_variance_neon_dotprod(const uint8_t *src, int src_stride,
                                             const uint8_t *ref, int ref_stride,
                                             int pixel_step, int w, int h,
                                             uint32_t *sse, int *sum,
                                             int filter_offset) {
  assert(w != 4);

  if (w == 8) {
    bil_variance_8xh_neon_dotprod(src, src_stride, ref, ref_stride, pixel_step,
                                  h, sse, sum, filter_offset);
    return;
  }

  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  uint32x4_t src_sum = vdupq_n_u32(0);
  uint32x4_t ref_sum = vdupq_n_u32(0);
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  do {
    int i = 0;
    do {
      uint8x16_t s0 = vld1q_u8(src + i);
      uint8x16_t s1 = vld1q_u8(src + i + pixel_step);
      uint16x8_t blend_l = vmull_u8(vget_low_u8(s0), f0);
      blend_l = vmlal_u8(blend_l, vget_low_u8(s1), f1);
      uint16x8_t blend_h = vmull_u8(vget_high_u8(s0), f0);
      blend_h = vmlal_u8(blend_h, vget_high_u8(s1), f1);
      uint8x16_t s =
          vcombine_u8(vrshrn_n_u16(blend_l, 3), vrshrn_n_u16(blend_h, 3));
      uint8x16_t r = vld1q_u8(ref + i);

      src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
      ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

      uint8x16_t abs_diff = vabdq_u8(s, r);
      sse_u32 = vdotq_u32(sse_u32, abs_diff, abs_diff);

      i += 16;
    } while (i < w);

    src += src_stride;
    ref += ref_stride;
  } while (--h != 0);

  int32x4_t sum_diff =
      vsubq_s32(vreinterpretq_s32_u32(src_sum), vreinterpretq_s32_u32(ref_sum));
  *sum = vaddvq_s32(sum_diff);
  *sse = vaddvq_u32(sse_u32);
}

static inline void avg_variance_neon_dotprod(const uint8_t *src, int src_stride,
                                             const uint8_t *ref, int ref_stride,
                                             int pixel_step, int w, int h,
                                             uint32_t *sse, int *sum) {
  assert(w >= 16 && w % 16 == 0);
  uint32x4_t src_sum = vdupq_n_u32(0);
  uint32x4_t ref_sum = vdupq_n_u32(0);
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  do {
    int i = 0;
    do {
      uint8x16_t s0 = vld1q_u8(src + i);
      uint8x16_t s1 = vld1q_u8(src + i + pixel_step);
      uint8x16_t s = vrhaddq_u8(s0, s1);
      uint8x16_t r = vld1q_u8(ref + i);

      src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
      ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

      uint8x16_t abs_diff = vabdq_u8(s, r);
      sse_u32 = vdotq_u32(sse_u32, abs_diff, abs_diff);

      i += 16;
    } while (i < w);

    src += src_stride;
    ref += ref_stride;
  } while (--h != 0);

  int32x4_t sum_diff =
      vsubq_s32(vreinterpretq_s32_u32(src_sum), vreinterpretq_s32_u32(ref_sum));
  *sum = vaddvq_s32(sum_diff);
  *sse = vaddvq_u32(sse_u32);
}

#define SUBPEL_VARIANCE_4XH_NEON_DOTPROD(h, padding)                        \
  unsigned int vpx_sub_pixel_variance4x##h##_neon_dotprod(                  \
      const uint8_t *src, int src_stride, int xoffset, int yoffset,         \
      const uint8_t *ref, int ref_stride, uint32_t *sse) {                  \
    uint8_t tmp0[4 * (h + padding)];                                        \
    uint8_t tmp1[4 * h];                                                    \
    var_filter_block2d_bil_w4(src, tmp0, src_stride, 1, (h + padding),      \
                              xoffset);                                     \
    var_filter_block2d_bil_w4(tmp0, tmp1, 4, 4, h, yoffset);                \
    return vpx_variance4x##h##_neon_dotprod(tmp1, 4, ref, ref_stride, sse); \
  }

#define SUBPEL_VARIANCE_WXH_NEON_DOTPROD(w, h, shift, padding)             \
  unsigned int vpx_sub_pixel_variance##w##x##h##_neon_dotprod(             \
      const uint8_t *src, int src_stride, int xoffset, int yoffset,        \
      const uint8_t *ref, int ref_stride, unsigned int *sse) {             \
    uint8_t tmp[w * (h + padding)];                                        \
    int sum;                                                               \
    var_filter_block2d_bil_w##w(src, tmp, src_stride, 1, h + padding,      \
                                xoffset);                                  \
    bil_variance_neon_dotprod(tmp, w, ref, ref_stride, w, w, h, sse, &sum, \
                              yoffset);                                    \
    return *sse - (uint32_t)(((int64_t)sum * sum) >> shift);               \
  }

#define SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(w, h, shift, padding)     \
  unsigned int vpx_sub_pixel_variance##w##x##h##_neon_dotprod(                 \
      const uint8_t *src, int src_stride, int xoffset, int yoffset,            \
      const uint8_t *ref, int ref_stride, unsigned int *sse) {                 \
    int sum;                                                                   \
    if (xoffset == 0) {                                                        \
      if (yoffset == 0) {                                                      \
        return vpx_variance##w##x##h##_neon_dotprod(src, src_stride, ref,      \
                                                    ref_stride, sse);          \
      } else if (yoffset == 4) {                                               \
        avg_variance_neon_dotprod(src, src_stride, ref, ref_stride,            \
                                  src_stride, w, h, sse, &sum);                \
      } else {                                                                 \
        bil_variance_neon_dotprod(src, src_stride, ref, ref_stride,            \
                                  src_stride, w, h, sse, &sum, yoffset);       \
      }                                                                        \
    } else if (xoffset == 4) {                                                 \
      uint8_t tmp[w * (h + padding)];                                          \
      if (yoffset == 0) {                                                      \
        avg_variance_neon_dotprod(src, src_stride, ref, ref_stride, 1, w, h,   \
                                  sse, &sum);                                  \
      } else if (yoffset == 4) {                                               \
        var_filter_block2d_avg(src, tmp, src_stride, 1, w, h + padding);       \
        avg_variance_neon_dotprod(tmp, w, ref, ref_stride, w, w, h, sse,       \
                                  &sum);                                       \
      } else {                                                                 \
        var_filter_block2d_avg(src, tmp, src_stride, 1, w, h + padding);       \
        bil_variance_neon_dotprod(tmp, w, ref, ref_stride, w, w, h, sse, &sum, \
                                  yoffset);                                    \
      }                                                                        \
    } else {                                                                   \
      uint8_t tmp[w * (h + padding)];                                          \
      if (yoffset == 0) {                                                      \
        bil_variance_neon_dotprod(src, src_stride, ref, ref_stride, 1, w, h,   \
                                  sse, &sum, xoffset);                         \
      } else if (yoffset == 4) {                                               \
        var_filter_block2d_bil_w##w(src, tmp, src_stride, 1, h + padding,      \
                                    xoffset);                                  \
        avg_variance_neon_dotprod(tmp, w, ref, ref_stride, w, w, h, sse,       \
                                  &sum);                                       \
      } else {                                                                 \
        var_filter_block2d_bil_w##w(src, tmp, src_stride, 1, h + padding,      \
                                    xoffset);                                  \
        bil_variance_neon_dotprod(tmp, w, ref, ref_stride, w, w, h, sse, &sum, \
                                  yoffset);                                    \
      }                                                                        \
    }                                                                          \
    return *sse - (uint32_t)(((int64_t)sum * sum) >> shift);                   \
  }

SUBPEL_VARIANCE_4XH_NEON_DOTPROD(8, 2)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(8, 4, 5, 1)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(8, 8, 6, 1)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(8, 16, 7, 1)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 8, 7, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 16, 8, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 32, 9, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 16, 9, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 32, 10, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 64, 11, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 32, 11, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 64, 12, 1)

#undef SUBPEL_VARIANCE_WXH_NEON_DOTPROD
#undef SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD
