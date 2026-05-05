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
#include "./vpx_dsp_rtcd.h"
#include "./vpx_config.h"

#include "vpx/vpx_integer.h"

#include "vpx_dsp/variance.h"
#include "vpx_dsp/arm/mem_neon.h"

// Process a block exactly 4 wide and a multiple of 2 high.
static void var_filter_block2d_bil_w4(const uint8_t *src_ptr, uint8_t *dst_ptr,
                                      int src_stride, int pixel_step,
                                      int dst_height, int filter_offset) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  int i = dst_height;
  do {
    uint8x8_t s0 = load_unaligned_u8(src_ptr, src_stride);
    uint8x8_t s1 = load_unaligned_u8(src_ptr + pixel_step, src_stride);
    uint16x8_t blend = vmlal_u8(vmull_u8(s0, f0), s1, f1);
    uint8x8_t blend_u8 = vrshrn_n_u16(blend, 3);
    vst1_u8(dst_ptr, blend_u8);

    src_ptr += 2 * src_stride;
    dst_ptr += 2 * 4;
    i -= 2;
  } while (i != 0);
}

// Process a block exactly 8 wide and any height.
static void var_filter_block2d_bil_w8(const uint8_t *src_ptr, uint8_t *dst_ptr,
                                      int src_stride, int pixel_step,
                                      int dst_height, int filter_offset) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  int i = dst_height;
  do {
    uint8x8_t s0 = vld1_u8(src_ptr);
    uint8x8_t s1 = vld1_u8(src_ptr + pixel_step);
    uint16x8_t blend = vmlal_u8(vmull_u8(s0, f0), s1, f1);
    uint8x8_t blend_u8 = vrshrn_n_u16(blend, 3);
    vst1_u8(dst_ptr, blend_u8);

    src_ptr += src_stride;
    dst_ptr += 8;
  } while (--i != 0);
}

// Process a block which is a mutiple of 16 wide and any height.
static void var_filter_block2d_bil_large(const uint8_t *src_ptr,
                                         uint8_t *dst_ptr, int src_stride,
                                         int pixel_step, int dst_width,
                                         int dst_height, int filter_offset) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  int i = dst_height;
  do {
    int j = 0;
    do {
      uint8x16_t s0 = vld1q_u8(src_ptr + j);
      uint8x16_t s1 = vld1q_u8(src_ptr + j + pixel_step);
      uint16x8_t blend_l =
          vmlal_u8(vmull_u8(vget_low_u8(s0), f0), vget_low_u8(s1), f1);
      uint16x8_t blend_h =
          vmlal_u8(vmull_u8(vget_high_u8(s0), f0), vget_high_u8(s1), f1);
      uint8x8_t out_lo = vrshrn_n_u16(blend_l, 3);
      uint8x8_t out_hi = vrshrn_n_u16(blend_h, 3);
      vst1q_u8(dst_ptr + j, vcombine_u8(out_lo, out_hi));

      j += 16;
    } while (j < dst_width);

    src_ptr += src_stride;
    dst_ptr += dst_width;
  } while (--i != 0);
}

static void var_filter_block2d_bil_w16(const uint8_t *src_ptr, uint8_t *dst_ptr,
                                       int src_stride, int pixel_step,
                                       int dst_height, int filter_offset) {
  var_filter_block2d_bil_large(src_ptr, dst_ptr, src_stride, pixel_step, 16,
                               dst_height, filter_offset);
}
static void var_filter_block2d_bil_w32(const uint8_t *src_ptr, uint8_t *dst_ptr,
                                       int src_stride, int pixel_step,
                                       int dst_height, int filter_offset) {
  var_filter_block2d_bil_large(src_ptr, dst_ptr, src_stride, pixel_step, 32,
                               dst_height, filter_offset);
}
static void var_filter_block2d_bil_w64(const uint8_t *src_ptr, uint8_t *dst_ptr,
                                       int src_stride, int pixel_step,
                                       int dst_height, int filter_offset) {
  var_filter_block2d_bil_large(src_ptr, dst_ptr, src_stride, pixel_step, 64,
                               dst_height, filter_offset);
}

static void var_filter_block2d_avg(const uint8_t *src_ptr, uint8_t *dst_ptr,
                                   int src_stride, int pixel_step,
                                   int dst_width, int dst_height) {
  int i = dst_height;

  // We only specialize on the filter values for large block sizes (>= 16x16.)
  assert(dst_width >= 16 && dst_width % 16 == 0);

  do {
    int j = 0;
    do {
      uint8x16_t s0 = vld1q_u8(src_ptr + j);
      uint8x16_t s1 = vld1q_u8(src_ptr + j + pixel_step);
      uint8x16_t avg = vrhaddq_u8(s0, s1);
      vst1q_u8(dst_ptr + j, avg);

      j += 16;
    } while (j < dst_width);

    src_ptr += src_stride;
    dst_ptr += dst_width;
  } while (--i != 0);
}

#define SUBPEL_VARIANCE_WXH_NEON(w, h, padding)                          \
  unsigned int vpx_sub_pixel_variance##w##x##h##_neon(                   \
      const uint8_t *src, int src_stride, int xoffset, int yoffset,      \
      const uint8_t *ref, int ref_stride, uint32_t *sse) {               \
    uint8_t tmp0[w * (h + padding)];                                     \
    uint8_t tmp1[w * h];                                                 \
    var_filter_block2d_bil_w##w(src, tmp0, src_stride, 1, (h + padding), \
                                xoffset);                                \
    var_filter_block2d_bil_w##w(tmp0, tmp1, w, w, h, yoffset);           \
    return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);         \
  }

#define SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON(w, h, padding)                  \
  unsigned int vpx_sub_pixel_variance##w##x##h##_neon(                       \
      const uint8_t *src, int src_stride, int xoffset, int yoffset,          \
      const uint8_t *ref, int ref_stride, unsigned int *sse) {               \
    if (xoffset == 0) {                                                      \
      if (yoffset == 0) {                                                    \
        return vpx_variance##w##x##h(src, src_stride, ref, ref_stride, sse); \
      } else if (yoffset == 4) {                                             \
        uint8_t tmp[w * h];                                                  \
        var_filter_block2d_avg(src, tmp, src_stride, src_stride, w, h);      \
        return vpx_variance##w##x##h(tmp, w, ref, ref_stride, sse);          \
      } else {                                                               \
        uint8_t tmp[w * h];                                                  \
        var_filter_block2d_bil_w##w(src, tmp, src_stride, src_stride, h,     \
                                    yoffset);                                \
        return vpx_variance##w##x##h(tmp, w, ref, ref_stride, sse);          \
      }                                                                      \
    } else if (xoffset == 4) {                                               \
      uint8_t tmp0[w * (h + padding)];                                       \
      if (yoffset == 0) {                                                    \
        var_filter_block2d_avg(src, tmp0, src_stride, 1, w, h);              \
        return vpx_variance##w##x##h(tmp0, w, ref, ref_stride, sse);         \
      } else if (yoffset == 4) {                                             \
        uint8_t tmp1[w * (h + padding)];                                     \
        var_filter_block2d_avg(src, tmp0, src_stride, 1, w, (h + padding));  \
        var_filter_block2d_avg(tmp0, tmp1, w, w, w, h);                      \
        return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);         \
      } else {                                                               \
        uint8_t tmp1[w * (h + padding)];                                     \
        var_filter_block2d_avg(src, tmp0, src_stride, 1, w, (h + padding));  \
        var_filter_block2d_bil_w##w(tmp0, tmp1, w, w, h, yoffset);           \
        return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);         \
      }                                                                      \
    } else {                                                                 \
      uint8_t tmp0[w * (h + padding)];                                       \
      if (yoffset == 0) {                                                    \
        var_filter_block2d_bil_w##w(src, tmp0, src_stride, 1, h, xoffset);   \
        return vpx_variance##w##x##h(tmp0, w, ref, ref_stride, sse);         \
      } else if (yoffset == 4) {                                             \
        uint8_t tmp1[w * h];                                                 \
        var_filter_block2d_bil_w##w(src, tmp0, src_stride, 1, (h + padding), \
                                    xoffset);                                \
        var_filter_block2d_avg(tmp0, tmp1, w, w, w, h);                      \
        return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);         \
      } else {                                                               \
        uint8_t tmp1[w * h];                                                 \
        var_filter_block2d_bil_w##w(src, tmp0, src_stride, 1, (h + padding), \
                                    xoffset);                                \
        var_filter_block2d_bil_w##w(tmp0, tmp1, w, w, h, yoffset);           \
        return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);         \
      }                                                                      \
    }                                                                        \
  }

// 4x<h> blocks are processed two rows at a time, so require an extra row of
// padding.
SUBPEL_VARIANCE_WXH_NEON(4, 4, 2)
SUBPEL_VARIANCE_WXH_NEON(4, 8, 2)

SUBPEL_VARIANCE_WXH_NEON(8, 4, 1)
SUBPEL_VARIANCE_WXH_NEON(8, 8, 1)
SUBPEL_VARIANCE_WXH_NEON(8, 16, 1)

SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON(16, 8, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON(16, 16, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON(16, 32, 1)

SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON(32, 16, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON(32, 32, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON(32, 64, 1)

SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON(64, 32, 1)
SPECIALIZED_SUBPEL_VARIANCE_WXH_NEON(64, 64, 1)

// Combine bilinear filter with vpx_comp_avg_pred for blocks having width 4.
static void avg_pred_var_filter_block2d_bil_w4(const uint8_t *src_ptr,
                                               uint8_t *dst_ptr, int src_stride,
                                               int pixel_step, int dst_height,
                                               int filter_offset,
                                               const uint8_t *second_pred) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  int i = dst_height;
  do {
    uint8x8_t s0 = load_unaligned_u8(src_ptr, src_stride);
    uint8x8_t s1 = load_unaligned_u8(src_ptr + pixel_step, src_stride);
    uint16x8_t blend = vmlal_u8(vmull_u8(s0, f0), s1, f1);
    uint8x8_t blend_u8 = vrshrn_n_u16(blend, 3);

    uint8x8_t p = vld1_u8(second_pred);
    uint8x8_t avg = vrhadd_u8(blend_u8, p);

    vst1_u8(dst_ptr, avg);

    src_ptr += 2 * src_stride;
    dst_ptr += 2 * 4;
    second_pred += 2 * 4;
    i -= 2;
  } while (i != 0);
}

// Combine bilinear filter with vpx_comp_avg_pred for blocks having width 8.
static void avg_pred_var_filter_block2d_bil_w8(const uint8_t *src_ptr,
                                               uint8_t *dst_ptr, int src_stride,
                                               int pixel_step, int dst_height,
                                               int filter_offset,
                                               const uint8_t *second_pred) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  int i = dst_height;
  do {
    uint8x8_t s0 = vld1_u8(src_ptr);
    uint8x8_t s1 = vld1_u8(src_ptr + pixel_step);
    uint16x8_t blend = vmlal_u8(vmull_u8(s0, f0), s1, f1);
    uint8x8_t blend_u8 = vrshrn_n_u16(blend, 3);

    uint8x8_t p = vld1_u8(second_pred);
    uint8x8_t avg = vrhadd_u8(blend_u8, p);

    vst1_u8(dst_ptr, avg);

    src_ptr += src_stride;
    dst_ptr += 8;
    second_pred += 8;
  } while (--i > 0);
}

// Combine bilinear filter with vpx_comp_avg_pred for large blocks.
static void avg_pred_var_filter_block2d_bil_large(
    const uint8_t *src_ptr, uint8_t *dst_ptr, int src_stride, int pixel_step,
    int dst_width, int dst_height, int filter_offset,
    const uint8_t *second_pred) {
  const uint8x8_t f0 = vdup_n_u8(8 - filter_offset);
  const uint8x8_t f1 = vdup_n_u8(filter_offset);

  int i = dst_height;
  do {
    int j = 0;
    do {
      uint8x16_t s0 = vld1q_u8(src_ptr + j);
      uint8x16_t s1 = vld1q_u8(src_ptr + j + pixel_step);
      uint16x8_t blend_l =
          vmlal_u8(vmull_u8(vget_low_u8(s0), f0), vget_low_u8(s1), f1);
      uint16x8_t blend_h =
          vmlal_u8(vmull_u8(vget_high_u8(s0), f0), vget_high_u8(s1), f1);
      uint8x16_t blend_u8 =
          vcombine_u8(vrshrn_n_u16(blend_l, 3), vrshrn_n_u16(blend_h, 3));

      uint8x16_t p = vld1q_u8(second_pred);
      uint8x16_t avg = vrhaddq_u8(blend_u8, p);

      vst1q_u8(dst_ptr + j, avg);

      j += 16;
      second_pred += 16;
    } while (j < dst_width);

    src_ptr += src_stride;
    dst_ptr += dst_width;
  } while (--i != 0);
}

// Combine bilinear filter with vpx_comp_avg_pred for blocks having width 16.
static void avg_pred_var_filter_block2d_bil_w16(
    const uint8_t *src_ptr, uint8_t *dst_ptr, int src_stride, int pixel_step,
    int dst_height, int filter_offset, const uint8_t *second_pred) {
  avg_pred_var_filter_block2d_bil_large(src_ptr, dst_ptr, src_stride,
                                        pixel_step, 16, dst_height,
                                        filter_offset, second_pred);
}

// Combine bilinear filter with vpx_comp_avg_pred for blocks having width 32.
static void avg_pred_var_filter_block2d_bil_w32(
    const uint8_t *src_ptr, uint8_t *dst_ptr, int src_stride, int pixel_step,
    int dst_height, int filter_offset, const uint8_t *second_pred) {
  avg_pred_var_filter_block2d_bil_large(src_ptr, dst_ptr, src_stride,
                                        pixel_step, 32, dst_height,
                                        filter_offset, second_pred);
}

// Combine bilinear filter with vpx_comp_avg_pred for blocks having width 64.
static void avg_pred_var_filter_block2d_bil_w64(
    const uint8_t *src_ptr, uint8_t *dst_ptr, int src_stride, int pixel_step,
    int dst_height, int filter_offset, const uint8_t *second_pred) {
  avg_pred_var_filter_block2d_bil_large(src_ptr, dst_ptr, src_stride,
                                        pixel_step, 64, dst_height,
                                        filter_offset, second_pred);
}

// Combine averaging subpel filter with vpx_comp_avg_pred.
static void avg_pred_var_filter_block2d_avg(const uint8_t *src_ptr,
                                            uint8_t *dst_ptr, int src_stride,
                                            int pixel_step, int dst_width,
                                            int dst_height,
                                            const uint8_t *second_pred) {
  int i = dst_height;

  // We only specialize on the filter values for large block sizes (>= 16x16.)
  assert(dst_width >= 16 && dst_width % 16 == 0);

  do {
    int j = 0;
    do {
      uint8x16_t s0 = vld1q_u8(src_ptr + j);
      uint8x16_t s1 = vld1q_u8(src_ptr + j + pixel_step);
      uint8x16_t avg = vrhaddq_u8(s0, s1);

      uint8x16_t p = vld1q_u8(second_pred);
      avg = vrhaddq_u8(avg, p);

      vst1q_u8(dst_ptr + j, avg);

      j += 16;
      second_pred += 16;
    } while (j < dst_width);

    src_ptr += src_stride;
    dst_ptr += dst_width;
  } while (--i != 0);
}

// Implementation of vpx_comp_avg_pred for blocks having width >= 16.
static void avg_pred(const uint8_t *src_ptr, uint8_t *dst_ptr, int src_stride,
                     int dst_width, int dst_height,
                     const uint8_t *second_pred) {
  int i = dst_height;

  // We only specialize on the filter values for large block sizes (>= 16x16.)
  assert(dst_width >= 16 && dst_width % 16 == 0);

  do {
    int j = 0;
    do {
      uint8x16_t s = vld1q_u8(src_ptr + j);
      uint8x16_t p = vld1q_u8(second_pred);

      uint8x16_t avg = vrhaddq_u8(s, p);

      vst1q_u8(dst_ptr + j, avg);

      j += 16;
      second_pred += 16;
    } while (j < dst_width);

    src_ptr += src_stride;
    dst_ptr += dst_width;
  } while (--i != 0);
}

#define SUBPEL_AVG_VARIANCE_WXH_NEON(w, h, padding)                         \
  unsigned int vpx_sub_pixel_avg_variance##w##x##h##_neon(                  \
      const uint8_t *src, int source_stride, int xoffset, int yoffset,      \
      const uint8_t *ref, int ref_stride, uint32_t *sse,                    \
      const uint8_t *second_pred) {                                         \
    uint8_t tmp0[w * (h + padding)];                                        \
    uint8_t tmp1[w * h];                                                    \
    var_filter_block2d_bil_w##w(src, tmp0, source_stride, 1, (h + padding), \
                                xoffset);                                   \
    avg_pred_var_filter_block2d_bil_w##w(tmp0, tmp1, w, w, h, yoffset,      \
                                         second_pred);                      \
    return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);            \
  }

#define SPECIALIZED_SUBPEL_AVG_VARIANCE_WXH_NEON(w, h, padding)                \
  unsigned int vpx_sub_pixel_avg_variance##w##x##h##_neon(                     \
      const uint8_t *src, int source_stride, int xoffset, int yoffset,         \
      const uint8_t *ref, int ref_stride, unsigned int *sse,                   \
      const uint8_t *second_pred) {                                            \
    if (xoffset == 0) {                                                        \
      uint8_t tmp[w * h];                                                      \
      if (yoffset == 0) {                                                      \
        avg_pred(src, tmp, source_stride, w, h, second_pred);                  \
        return vpx_variance##w##x##h(tmp, w, ref, ref_stride, sse);            \
      } else if (yoffset == 4) {                                               \
        avg_pred_var_filter_block2d_avg(src, tmp, source_stride,               \
                                        source_stride, w, h, second_pred);     \
        return vpx_variance##w##x##h(tmp, w, ref, ref_stride, sse);            \
      } else {                                                                 \
        avg_pred_var_filter_block2d_bil_w##w(                                  \
            src, tmp, source_stride, source_stride, h, yoffset, second_pred);  \
        return vpx_variance##w##x##h(tmp, w, ref, ref_stride, sse);            \
      }                                                                        \
    } else if (xoffset == 4) {                                                 \
      uint8_t tmp0[w * (h + padding)];                                         \
      if (yoffset == 0) {                                                      \
        avg_pred_var_filter_block2d_avg(src, tmp0, source_stride, 1, w, h,     \
                                        second_pred);                          \
        return vpx_variance##w##x##h(tmp0, w, ref, ref_stride, sse);           \
      } else if (yoffset == 4) {                                               \
        uint8_t tmp1[w * (h + padding)];                                       \
        var_filter_block2d_avg(src, tmp0, source_stride, 1, w, (h + padding)); \
        avg_pred_var_filter_block2d_avg(tmp0, tmp1, w, w, w, h, second_pred);  \
        return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);           \
      } else {                                                                 \
        uint8_t tmp1[w * (h + padding)];                                       \
        var_filter_block2d_avg(src, tmp0, source_stride, 1, w, (h + padding)); \
        avg_pred_var_filter_block2d_bil_w##w(tmp0, tmp1, w, w, h, yoffset,     \
                                             second_pred);                     \
        return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);           \
      }                                                                        \
    } else {                                                                   \
      uint8_t tmp0[w * (h + padding)];                                         \
      if (yoffset == 0) {                                                      \
        avg_pred_var_filter_block2d_bil_w##w(src, tmp0, source_stride, 1, h,   \
                                             xoffset, second_pred);            \
        return vpx_variance##w##x##h(tmp0, w, ref, ref_stride, sse);           \
      } else if (yoffset == 4) {                                               \
        uint8_t tmp1[w * h];                                                   \
        var_filter_block2d_bil_w##w(src, tmp0, source_stride, 1,               \
                                    (h + padding), xoffset);                   \
        avg_pred_var_filter_block2d_avg(tmp0, tmp1, w, w, w, h, second_pred);  \
        return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);           \
      } else {                                                                 \
        uint8_t tmp1[w * h];                                                   \
        var_filter_block2d_bil_w##w(src, tmp0, source_stride, 1,               \
                                    (h + padding), xoffset);                   \
        avg_pred_var_filter_block2d_bil_w##w(tmp0, tmp1, w, w, h, yoffset,     \
                                             second_pred);                     \
        return vpx_variance##w##x##h(tmp1, w, ref, ref_stride, sse);           \
      }                                                                        \
    }                                                                          \
  }

// 4x<h> blocks are processed two rows at a time, so require an extra row of
// padding.
SUBPEL_AVG_VARIANCE_WXH_NEON(4, 4, 2)
SUBPEL_AVG_VARIANCE_WXH_NEON(4, 8, 2)

SUBPEL_AVG_VARIANCE_WXH_NEON(8, 4, 1)
SUBPEL_AVG_VARIANCE_WXH_NEON(8, 8, 1)
SUBPEL_AVG_VARIANCE_WXH_NEON(8, 16, 1)

SPECIALIZED_SUBPEL_AVG_VARIANCE_WXH_NEON(16, 8, 1)
SPECIALIZED_SUBPEL_AVG_VARIANCE_WXH_NEON(16, 16, 1)
SPECIALIZED_SUBPEL_AVG_VARIANCE_WXH_NEON(16, 32, 1)

SPECIALIZED_SUBPEL_AVG_VARIANCE_WXH_NEON(32, 16, 1)
SPECIALIZED_SUBPEL_AVG_VARIANCE_WXH_NEON(32, 32, 1)
SPECIALIZED_SUBPEL_AVG_VARIANCE_WXH_NEON(32, 64, 1)

SPECIALIZED_SUBPEL_AVG_VARIANCE_WXH_NEON(64, 32, 1)
SPECIALIZED_SUBPEL_AVG_VARIANCE_WXH_NEON(64, 64, 1)
