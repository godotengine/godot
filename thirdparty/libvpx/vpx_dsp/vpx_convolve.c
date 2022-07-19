/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <string.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_convolve.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/vpx_filter.h"
#include "vpx_ports/mem.h"

static void convolve_horiz(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const InterpKernel *x_filters,
                           int x0_q4, int x_step_q4, int w, int h) {
  int x, y;
  src -= SUBPEL_TAPS / 2 - 1;
  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_x[k] * x_filter[k];
      dst[x] = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

static void convolve_avg_horiz(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *x_filters,
                               int x0_q4, int x_step_q4, int w, int h) {
  int x, y;
  src -= SUBPEL_TAPS / 2 - 1;
  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_x[k] * x_filter[k];
      dst[x] = ROUND_POWER_OF_TWO(dst[x] +
          clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS)), 1);
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

static void convolve_vert(const uint8_t *src, ptrdiff_t src_stride,
                          uint8_t *dst, ptrdiff_t dst_stride,
                          const InterpKernel *y_filters,
                          int y0_q4, int y_step_q4, int w, int h) {
  int x, y;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      const unsigned char *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_y[k * src_stride] * y_filter[k];
      dst[y * dst_stride] = clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS));
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

static void convolve_avg_vert(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *y_filters,
                              int y0_q4, int y_step_q4, int w, int h) {
  int x, y;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      const unsigned char *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_y[k * src_stride] * y_filter[k];
      dst[y * dst_stride] = ROUND_POWER_OF_TWO(dst[y * dst_stride] +
          clip_pixel(ROUND_POWER_OF_TWO(sum, FILTER_BITS)), 1);
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

static void convolve(const uint8_t *src, ptrdiff_t src_stride,
                     uint8_t *dst, ptrdiff_t dst_stride,
                     const InterpKernel *const x_filters,
                     int x0_q4, int x_step_q4,
                     const InterpKernel *const y_filters,
                     int y0_q4, int y_step_q4,
                     int w, int h) {
  // Note: Fixed size intermediate buffer, temp, places limits on parameters.
  // 2d filtering proceeds in 2 steps:
  //   (1) Interpolate horizontally into an intermediate buffer, temp.
  //   (2) Interpolate temp vertically to derive the sub-pixel result.
  // Deriving the maximum number of rows in the temp buffer (135):
  // --Smallest scaling factor is x1/2 ==> y_step_q4 = 32 (Normative).
  // --Largest block size is 64x64 pixels.
  // --64 rows in the downscaled frame span a distance of (64 - 1) * 32 in the
  //   original frame (in 1/16th pixel units).
  // --Must round-up because block may be located at sub-pixel position.
  // --Require an additional SUBPEL_TAPS rows for the 8-tap filter tails.
  // --((64 - 1) * 32 + 15) >> 4 + 8 = 135.
  uint8_t temp[135 * 64];
  int intermediate_height =
          (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

  assert(w <= 64);
  assert(h <= 64);
  assert(y_step_q4 <= 32);
  assert(x_step_q4 <= 32);

  convolve_horiz(src - src_stride * (SUBPEL_TAPS / 2 - 1), src_stride, temp, 64,
                 x_filters, x0_q4, x_step_q4, w, intermediate_height);
  convolve_vert(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst, dst_stride,
                y_filters, y0_q4, y_step_q4, w, h);
}

static const InterpKernel *get_filter_base(const int16_t *filter) {
  // NOTE: This assumes that the filter table is 256-byte aligned.
  // TODO(agrange) Modify to make independent of table alignment.
  return (const InterpKernel *)(((intptr_t)filter) & ~((intptr_t)0xFF));
}

static int get_filter_offset(const int16_t *f, const InterpKernel *base) {
  return (int)((const InterpKernel *)(intptr_t)f - base);
}

void vpx_convolve8_horiz_c(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const int16_t *filter_x, int x_step_q4,
                           const int16_t *filter_y, int y_step_q4,
                           int w, int h) {
  const InterpKernel *const filters_x = get_filter_base(filter_x);
  const int x0_q4 = get_filter_offset(filter_x, filters_x);

  (void)filter_y;
  (void)y_step_q4;

  convolve_horiz(src, src_stride, dst, dst_stride, filters_x,
                 x0_q4, x_step_q4, w, h);
}

void vpx_convolve8_avg_horiz_c(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const int16_t *filter_x, int x_step_q4,
                               const int16_t *filter_y, int y_step_q4,
                               int w, int h) {
  const InterpKernel *const filters_x = get_filter_base(filter_x);
  const int x0_q4 = get_filter_offset(filter_x, filters_x);

  (void)filter_y;
  (void)y_step_q4;

  convolve_avg_horiz(src, src_stride, dst, dst_stride, filters_x,
                     x0_q4, x_step_q4, w, h);
}

void vpx_convolve8_vert_c(const uint8_t *src, ptrdiff_t src_stride,
                          uint8_t *dst, ptrdiff_t dst_stride,
                          const int16_t *filter_x, int x_step_q4,
                          const int16_t *filter_y, int y_step_q4,
                          int w, int h) {
  const InterpKernel *const filters_y = get_filter_base(filter_y);
  const int y0_q4 = get_filter_offset(filter_y, filters_y);

  (void)filter_x;
  (void)x_step_q4;

  convolve_vert(src, src_stride, dst, dst_stride, filters_y,
                y0_q4, y_step_q4, w, h);
}

void vpx_convolve8_avg_vert_c(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const int16_t *filter_x, int x_step_q4,
                              const int16_t *filter_y, int y_step_q4,
                              int w, int h) {
  const InterpKernel *const filters_y = get_filter_base(filter_y);
  const int y0_q4 = get_filter_offset(filter_y, filters_y);

  (void)filter_x;
  (void)x_step_q4;

  convolve_avg_vert(src, src_stride, dst, dst_stride, filters_y,
                    y0_q4, y_step_q4, w, h);
}

void vpx_convolve8_c(const uint8_t *src, ptrdiff_t src_stride,
                     uint8_t *dst, ptrdiff_t dst_stride,
                     const int16_t *filter_x, int x_step_q4,
                     const int16_t *filter_y, int y_step_q4,
                     int w, int h) {
  const InterpKernel *const filters_x = get_filter_base(filter_x);
  const int x0_q4 = get_filter_offset(filter_x, filters_x);

  const InterpKernel *const filters_y = get_filter_base(filter_y);
  const int y0_q4 = get_filter_offset(filter_y, filters_y);

  convolve(src, src_stride, dst, dst_stride,
           filters_x, x0_q4, x_step_q4,
           filters_y, y0_q4, y_step_q4, w, h);
}

void vpx_convolve8_avg_c(const uint8_t *src, ptrdiff_t src_stride,
                         uint8_t *dst, ptrdiff_t dst_stride,
                         const int16_t *filter_x, int x_step_q4,
                         const int16_t *filter_y, int y_step_q4,
                         int w, int h) {
  /* Fixed size intermediate buffer places limits on parameters. */
  DECLARE_ALIGNED(16, uint8_t, temp[64 * 64]);
  assert(w <= 64);
  assert(h <= 64);

  vpx_convolve8_c(src, src_stride, temp, 64,
                  filter_x, x_step_q4, filter_y, y_step_q4, w, h);
  vpx_convolve_avg_c(temp, 64, dst, dst_stride, NULL, 0, NULL, 0, w, h);
}

void vpx_convolve_copy_c(const uint8_t *src, ptrdiff_t src_stride,
                         uint8_t *dst, ptrdiff_t dst_stride,
                         const int16_t *filter_x, int filter_x_stride,
                         const int16_t *filter_y, int filter_y_stride,
                         int w, int h) {
  int r;

  (void)filter_x;  (void)filter_x_stride;
  (void)filter_y;  (void)filter_y_stride;

  for (r = h; r > 0; --r) {
    memcpy(dst, src, w);
    src += src_stride;
    dst += dst_stride;
  }
}

void vpx_convolve_avg_c(const uint8_t *src, ptrdiff_t src_stride,
                        uint8_t *dst, ptrdiff_t dst_stride,
                        const int16_t *filter_x, int filter_x_stride,
                        const int16_t *filter_y, int filter_y_stride,
                        int w, int h) {
  int x, y;

  (void)filter_x;  (void)filter_x_stride;
  (void)filter_y;  (void)filter_y_stride;

  for (y = 0; y < h; ++y) {
    for (x = 0; x < w; ++x)
      dst[x] = ROUND_POWER_OF_TWO(dst[x] + src[x], 1);

    src += src_stride;
    dst += dst_stride;
  }
}

void vpx_scaled_horiz_c(const uint8_t *src, ptrdiff_t src_stride,
                        uint8_t *dst, ptrdiff_t dst_stride,
                        const int16_t *filter_x, int x_step_q4,
                        const int16_t *filter_y, int y_step_q4,
                        int w, int h) {
  vpx_convolve8_horiz_c(src, src_stride, dst, dst_stride, filter_x, x_step_q4,
                        filter_y, y_step_q4, w, h);
}

void vpx_scaled_vert_c(const uint8_t *src, ptrdiff_t src_stride,
                       uint8_t *dst, ptrdiff_t dst_stride,
                       const int16_t *filter_x, int x_step_q4,
                       const int16_t *filter_y, int y_step_q4,
                       int w, int h) {
  vpx_convolve8_vert_c(src, src_stride, dst, dst_stride, filter_x, x_step_q4,
                       filter_y, y_step_q4, w, h);
}

void vpx_scaled_2d_c(const uint8_t *src, ptrdiff_t src_stride,
                     uint8_t *dst, ptrdiff_t dst_stride,
                     const int16_t *filter_x, int x_step_q4,
                     const int16_t *filter_y, int y_step_q4,
                     int w, int h) {
  vpx_convolve8_c(src, src_stride, dst, dst_stride, filter_x, x_step_q4,
                  filter_y, y_step_q4, w, h);
}

void vpx_scaled_avg_horiz_c(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const int16_t *filter_x, int x_step_q4,
                            const int16_t *filter_y, int y_step_q4,
                            int w, int h) {
  vpx_convolve8_avg_horiz_c(src, src_stride, dst, dst_stride, filter_x,
                            x_step_q4, filter_y, y_step_q4, w, h);
}

void vpx_scaled_avg_vert_c(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const int16_t *filter_x, int x_step_q4,
                           const int16_t *filter_y, int y_step_q4,
                           int w, int h) {
  vpx_convolve8_avg_vert_c(src, src_stride, dst, dst_stride, filter_x,
                           x_step_q4, filter_y, y_step_q4, w, h);
}

void vpx_scaled_avg_2d_c(const uint8_t *src, ptrdiff_t src_stride,
                     uint8_t *dst, ptrdiff_t dst_stride,
                     const int16_t *filter_x, int x_step_q4,
                     const int16_t *filter_y, int y_step_q4,
                     int w, int h) {
  vpx_convolve8_avg_c(src, src_stride, dst, dst_stride, filter_x, x_step_q4,
                      filter_y, y_step_q4, w, h);
}

#if CONFIG_VP9_HIGHBITDEPTH
static void highbd_convolve_horiz(const uint8_t *src8, ptrdiff_t src_stride,
                                  uint8_t *dst8, ptrdiff_t dst_stride,
                                  const InterpKernel *x_filters,
                                  int x0_q4, int x_step_q4,
                                  int w, int h, int bd) {
  int x, y;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  src -= SUBPEL_TAPS / 2 - 1;
  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      const uint16_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_x[k] * x_filter[k];
      dst[x] = clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

static void highbd_convolve_avg_horiz(const uint8_t *src8, ptrdiff_t src_stride,
                                      uint8_t *dst8, ptrdiff_t dst_stride,
                                      const InterpKernel *x_filters,
                                      int x0_q4, int x_step_q4,
                                      int w, int h, int bd) {
  int x, y;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  src -= SUBPEL_TAPS / 2 - 1;
  for (y = 0; y < h; ++y) {
    int x_q4 = x0_q4;
    for (x = 0; x < w; ++x) {
      const uint16_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_x[k] * x_filter[k];
      dst[x] = ROUND_POWER_OF_TWO(dst[x] +
          clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd), 1);
      x_q4 += x_step_q4;
    }
    src += src_stride;
    dst += dst_stride;
  }
}

static void highbd_convolve_vert(const uint8_t *src8, ptrdiff_t src_stride,
                                 uint8_t *dst8, ptrdiff_t dst_stride,
                                 const InterpKernel *y_filters,
                                 int y0_q4, int y_step_q4, int w, int h,
                                 int bd) {
  int x, y;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      const uint16_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_y[k * src_stride] * y_filter[k];
      dst[y * dst_stride] = clip_pixel_highbd(
          ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd);
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

static void highbd_convolve_avg_vert(const uint8_t *src8, ptrdiff_t src_stride,
                                     uint8_t *dst8, ptrdiff_t dst_stride,
                                     const InterpKernel *y_filters,
                                     int y0_q4, int y_step_q4, int w, int h,
                                     int bd) {
  int x, y;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);
  for (x = 0; x < w; ++x) {
    int y_q4 = y0_q4;
    for (y = 0; y < h; ++y) {
      const uint16_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
      const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];
      int k, sum = 0;
      for (k = 0; k < SUBPEL_TAPS; ++k)
        sum += src_y[k * src_stride] * y_filter[k];
      dst[y * dst_stride] = ROUND_POWER_OF_TWO(dst[y * dst_stride] +
          clip_pixel_highbd(ROUND_POWER_OF_TWO(sum, FILTER_BITS), bd), 1);
      y_q4 += y_step_q4;
    }
    ++src;
    ++dst;
  }
}

static void highbd_convolve(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const InterpKernel *const x_filters,
                            int x0_q4, int x_step_q4,
                            const InterpKernel *const y_filters,
                            int y0_q4, int y_step_q4,
                            int w, int h, int bd) {
  // Note: Fixed size intermediate buffer, temp, places limits on parameters.
  // 2d filtering proceeds in 2 steps:
  //   (1) Interpolate horizontally into an intermediate buffer, temp.
  //   (2) Interpolate temp vertically to derive the sub-pixel result.
  // Deriving the maximum number of rows in the temp buffer (135):
  // --Smallest scaling factor is x1/2 ==> y_step_q4 = 32 (Normative).
  // --Largest block size is 64x64 pixels.
  // --64 rows in the downscaled frame span a distance of (64 - 1) * 32 in the
  //   original frame (in 1/16th pixel units).
  // --Must round-up because block may be located at sub-pixel position.
  // --Require an additional SUBPEL_TAPS rows for the 8-tap filter tails.
  // --((64 - 1) * 32 + 15) >> 4 + 8 = 135.
  uint16_t temp[64 * 135];
  int intermediate_height =
          (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

  assert(w <= 64);
  assert(h <= 64);
  assert(y_step_q4 <= 32);
  assert(x_step_q4 <= 32);

  highbd_convolve_horiz(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                        src_stride, CONVERT_TO_BYTEPTR(temp), 64,
                        x_filters, x0_q4, x_step_q4, w,
                        intermediate_height, bd);
  highbd_convolve_vert(CONVERT_TO_BYTEPTR(temp) + 64 * (SUBPEL_TAPS / 2 - 1),
                       64, dst, dst_stride, y_filters, y0_q4, y_step_q4,
                       w, h, bd);
}


void vpx_highbd_convolve8_horiz_c(const uint8_t *src, ptrdiff_t src_stride,
                                  uint8_t *dst, ptrdiff_t dst_stride,
                                  const int16_t *filter_x, int x_step_q4,
                                  const int16_t *filter_y, int y_step_q4,
                                  int w, int h, int bd) {
  const InterpKernel *const filters_x = get_filter_base(filter_x);
  const int x0_q4 = get_filter_offset(filter_x, filters_x);
  (void)filter_y;
  (void)y_step_q4;

  highbd_convolve_horiz(src, src_stride, dst, dst_stride, filters_x,
                        x0_q4, x_step_q4, w, h, bd);
}

void vpx_highbd_convolve8_avg_horiz_c(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      const int16_t *filter_x, int x_step_q4,
                                      const int16_t *filter_y, int y_step_q4,
                                      int w, int h, int bd) {
  const InterpKernel *const filters_x = get_filter_base(filter_x);
  const int x0_q4 = get_filter_offset(filter_x, filters_x);
  (void)filter_y;
  (void)y_step_q4;

  highbd_convolve_avg_horiz(src, src_stride, dst, dst_stride, filters_x,
                            x0_q4, x_step_q4, w, h, bd);
}

void vpx_highbd_convolve8_vert_c(const uint8_t *src, ptrdiff_t src_stride,
                                 uint8_t *dst, ptrdiff_t dst_stride,
                                 const int16_t *filter_x, int x_step_q4,
                                 const int16_t *filter_y, int y_step_q4,
                                 int w, int h, int bd) {
  const InterpKernel *const filters_y = get_filter_base(filter_y);
  const int y0_q4 = get_filter_offset(filter_y, filters_y);
  (void)filter_x;
  (void)x_step_q4;

  highbd_convolve_vert(src, src_stride, dst, dst_stride, filters_y,
                       y0_q4, y_step_q4, w, h, bd);
}

void vpx_highbd_convolve8_avg_vert_c(const uint8_t *src, ptrdiff_t src_stride,
                                     uint8_t *dst, ptrdiff_t dst_stride,
                                     const int16_t *filter_x, int x_step_q4,
                                     const int16_t *filter_y, int y_step_q4,
                                     int w, int h, int bd) {
  const InterpKernel *const filters_y = get_filter_base(filter_y);
  const int y0_q4 = get_filter_offset(filter_y, filters_y);
  (void)filter_x;
  (void)x_step_q4;

  highbd_convolve_avg_vert(src, src_stride, dst, dst_stride, filters_y,
                           y0_q4, y_step_q4, w, h, bd);
}

void vpx_highbd_convolve8_c(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const int16_t *filter_x, int x_step_q4,
                            const int16_t *filter_y, int y_step_q4,
                            int w, int h, int bd) {
  const InterpKernel *const filters_x = get_filter_base(filter_x);
  const int x0_q4 = get_filter_offset(filter_x, filters_x);

  const InterpKernel *const filters_y = get_filter_base(filter_y);
  const int y0_q4 = get_filter_offset(filter_y, filters_y);

  highbd_convolve(src, src_stride, dst, dst_stride,
                  filters_x, x0_q4, x_step_q4,
                  filters_y, y0_q4, y_step_q4, w, h, bd);
}

void vpx_highbd_convolve8_avg_c(const uint8_t *src, ptrdiff_t src_stride,
                                uint8_t *dst, ptrdiff_t dst_stride,
                                const int16_t *filter_x, int x_step_q4,
                                const int16_t *filter_y, int y_step_q4,
                                int w, int h, int bd) {
  // Fixed size intermediate buffer places limits on parameters.
  DECLARE_ALIGNED(16, uint16_t, temp[64 * 64]);
  assert(w <= 64);
  assert(h <= 64);

  vpx_highbd_convolve8_c(src, src_stride, CONVERT_TO_BYTEPTR(temp), 64,
                         filter_x, x_step_q4, filter_y, y_step_q4, w, h, bd);
  vpx_highbd_convolve_avg_c(CONVERT_TO_BYTEPTR(temp), 64, dst, dst_stride,
                            NULL, 0, NULL, 0, w, h, bd);
}

void vpx_highbd_convolve_copy_c(const uint8_t *src8, ptrdiff_t src_stride,
                                uint8_t *dst8, ptrdiff_t dst_stride,
                                const int16_t *filter_x, int filter_x_stride,
                                const int16_t *filter_y, int filter_y_stride,
                                int w, int h, int bd) {
  int r;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  (void)filter_x;
  (void)filter_y;
  (void)filter_x_stride;
  (void)filter_y_stride;
  (void)bd;

  for (r = h; r > 0; --r) {
    memcpy(dst, src, w * sizeof(uint16_t));
    src += src_stride;
    dst += dst_stride;
  }
}

void vpx_highbd_convolve_avg_c(const uint8_t *src8, ptrdiff_t src_stride,
                               uint8_t *dst8, ptrdiff_t dst_stride,
                               const int16_t *filter_x, int filter_x_stride,
                               const int16_t *filter_y, int filter_y_stride,
                               int w, int h, int bd) {
  int x, y;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8);
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  (void)filter_x;
  (void)filter_y;
  (void)filter_x_stride;
  (void)filter_y_stride;
  (void)bd;

  for (y = 0; y < h; ++y) {
    for (x = 0; x < w; ++x) {
      dst[x] = ROUND_POWER_OF_TWO(dst[x] + src[x], 1);
    }
    src += src_stride;
    dst += dst_stride;
  }
}
#endif
