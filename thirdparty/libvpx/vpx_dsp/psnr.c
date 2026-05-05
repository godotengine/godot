/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <assert.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/psnr.h"
#include "vpx_scale/yv12config.h"

double vpx_sse_to_psnr(double samples, double peak, double sse) {
  if (sse > 0.0) {
    const double psnr = 10.0 * log10(samples * peak * peak / sse);
    return psnr > MAX_PSNR ? MAX_PSNR : psnr;
  } else {
    return MAX_PSNR;
  }
}

/* TODO(yaowu): The block_variance calls the unoptimized versions of variance()
 * and highbd_8_variance(). It should not.
 */
static int64_t encoder_sse(const uint8_t *a, int a_stride, const uint8_t *b,
                           int b_stride, int w, int h) {
  int i, j;
  int64_t sse = 0;

  for (i = 0; i < h; i++) {
    for (j = 0; j < w; j++) {
      const int diff = a[j] - b[j];
      sse += diff * diff;
    }

    a += a_stride;
    b += b_stride;
  }

  return sse;
}

#if CONFIG_VP9_HIGHBITDEPTH
static int64_t encoder_highbd_sse(const uint8_t *a8, int a_stride,
                                  const uint8_t *b8, int b_stride, int w,
                                  int h) {
  int i, j;
  int64_t sse = 0;

  const uint16_t *a = CONVERT_TO_SHORTPTR(a8);
  const uint16_t *b = CONVERT_TO_SHORTPTR(b8);

  for (i = 0; i < h; i++) {
    for (j = 0; j < w; j++) {
      const int diff = a[j] - b[j];
      sse += diff * diff;
    }
    a += a_stride;
    b += b_stride;
  }

  return sse;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static int64_t get_sse(const uint8_t *a, int a_stride, const uint8_t *b,
                       int b_stride, int width, int height) {
  const int dw = width % 16;
  const int dh = height % 16;
  int64_t total_sse = 0;
  int x, y;

  if (dw > 0) {
    total_sse += encoder_sse(&a[width - dw], a_stride, &b[width - dw], b_stride,
                             dw, height);
  }

  if (dh > 0) {
    total_sse +=
        encoder_sse(&a[(height - dh) * a_stride], a_stride,
                    &b[(height - dh) * b_stride], b_stride, width - dw, dh);
  }

  for (y = 0; y < height / 16; ++y) {
    const uint8_t *pa = a;
    const uint8_t *pb = b;
    for (x = 0; x < width / 16; ++x) {
      total_sse += vpx_sse(pa, a_stride, pb, b_stride, 16, 16);

      pa += 16;
      pb += 16;
    }

    a += 16 * a_stride;
    b += 16 * b_stride;
  }

  return total_sse;
}

#if CONFIG_VP9_HIGHBITDEPTH
static int64_t highbd_get_sse_shift(const uint8_t *a8, int a_stride,
                                    const uint8_t *b8, int b_stride, int width,
                                    int height, unsigned int input_shift) {
  const uint16_t *a = CONVERT_TO_SHORTPTR(a8);
  const uint16_t *b = CONVERT_TO_SHORTPTR(b8);
  int64_t total_sse = 0;
  int x, y;
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      int64_t diff;
      diff = (a[x] >> input_shift) - (b[x] >> input_shift);
      total_sse += diff * diff;
    }
    a += a_stride;
    b += b_stride;
  }
  return total_sse;
}

static int64_t highbd_get_sse(const uint8_t *a, int a_stride, const uint8_t *b,
                              int b_stride, int width, int height) {
  int64_t total_sse = 0;
  int x, y;
  const int dw = width % 16;
  const int dh = height % 16;
  if (dw > 0) {
    total_sse += encoder_highbd_sse(&a[width - dw], a_stride, &b[width - dw],
                                    b_stride, dw, height);
  }
  if (dh > 0) {
    total_sse += encoder_highbd_sse(&a[(height - dh) * a_stride], a_stride,
                                    &b[(height - dh) * b_stride], b_stride,
                                    width - dw, dh);
  }
  for (y = 0; y < height / 16; ++y) {
    const uint8_t *pa = a;
    const uint8_t *pb = b;
    for (x = 0; x < width / 16; ++x) {
      total_sse += vpx_highbd_sse(pa, a_stride, pb, b_stride, 16, 16);
      pa += 16;
      pb += 16;
    }
    a += 16 * a_stride;
    b += 16 * b_stride;
  }
  return total_sse;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

int64_t vpx_get_y_sse(const YV12_BUFFER_CONFIG *a,
                      const YV12_BUFFER_CONFIG *b) {
  assert(a->y_crop_width == b->y_crop_width);
  assert(a->y_crop_height == b->y_crop_height);

  return get_sse(a->y_buffer, a->y_stride, b->y_buffer, b->y_stride,
                 a->y_crop_width, a->y_crop_height);
}

#if CONFIG_VP9_HIGHBITDEPTH
int64_t vpx_highbd_get_y_sse(const YV12_BUFFER_CONFIG *a,
                             const YV12_BUFFER_CONFIG *b) {
  assert(a->y_crop_width == b->y_crop_width);
  assert(a->y_crop_height == b->y_crop_height);
  assert((a->flags & YV12_FLAG_HIGHBITDEPTH) != 0);
  assert((b->flags & YV12_FLAG_HIGHBITDEPTH) != 0);

  return highbd_get_sse(a->y_buffer, a->y_stride, b->y_buffer, b->y_stride,
                        a->y_crop_width, a->y_crop_height);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_calc_highbd_psnr(const YV12_BUFFER_CONFIG *a,
                          const YV12_BUFFER_CONFIG *b, PSNR_STATS *psnr,
                          uint32_t bit_depth, uint32_t in_bit_depth,
                          int spatial_layer_id) {
  const int widths[3] = { a->y_crop_width, a->uv_crop_width, a->uv_crop_width };
  const int heights[3] = { a->y_crop_height, a->uv_crop_height,
                           a->uv_crop_height };
  const uint8_t *a_planes[3] = { a->y_buffer, a->u_buffer, a->v_buffer };
  const int a_strides[3] = { a->y_stride, a->uv_stride, a->uv_stride };
  const uint8_t *b_planes[3] = { b->y_buffer, b->u_buffer, b->v_buffer };
  const int b_strides[3] = { b->y_stride, b->uv_stride, b->uv_stride };
  int i;
  uint64_t total_sse = 0;
  uint32_t total_samples = 0;
  const double peak = (double)((1 << in_bit_depth) - 1);
  const unsigned int input_shift = bit_depth - in_bit_depth;

  for (i = 0; i < 3; ++i) {
    const int w = widths[i];
    const int h = heights[i];
    const uint32_t samples = w * h;
    uint64_t sse;
    if (a->flags & YV12_FLAG_HIGHBITDEPTH) {
      if (input_shift) {
        sse = highbd_get_sse_shift(a_planes[i], a_strides[i], b_planes[i],
                                   b_strides[i], w, h, input_shift);
      } else {
        sse = highbd_get_sse(a_planes[i], a_strides[i], b_planes[i],
                             b_strides[i], w, h);
      }
    } else {
      sse = get_sse(a_planes[i], a_strides[i], b_planes[i], b_strides[i], w, h);
    }
    psnr->sse[1 + i] = sse;
    psnr->samples[1 + i] = samples;
    psnr->psnr[1 + i] = vpx_sse_to_psnr(samples, peak, (double)sse);

    total_sse += sse;
    total_samples += samples;
  }

  psnr->sse[0] = total_sse;
  psnr->samples[0] = total_samples;
  psnr->psnr[0] =
      vpx_sse_to_psnr((double)total_samples, peak, (double)total_sse);
  psnr->spatial_layer_id = spatial_layer_id;
}

#endif  // !CONFIG_VP9_HIGHBITDEPTH

void vpx_calc_psnr(const YV12_BUFFER_CONFIG *a, const YV12_BUFFER_CONFIG *b,
                   PSNR_STATS *psnr, int spatial_layer_id) {
  static const double peak = 255.0;
  const int widths[3] = { a->y_crop_width, a->uv_crop_width, a->uv_crop_width };
  const int heights[3] = { a->y_crop_height, a->uv_crop_height,
                           a->uv_crop_height };
  const uint8_t *a_planes[3] = { a->y_buffer, a->u_buffer, a->v_buffer };
  const int a_strides[3] = { a->y_stride, a->uv_stride, a->uv_stride };
  const uint8_t *b_planes[3] = { b->y_buffer, b->u_buffer, b->v_buffer };
  const int b_strides[3] = { b->y_stride, b->uv_stride, b->uv_stride };
  int i;
  uint64_t total_sse = 0;
  uint32_t total_samples = 0;

  for (i = 0; i < 3; ++i) {
    const int w = widths[i];
    const int h = heights[i];
    const uint32_t samples = w * h;
    const uint64_t sse =
        get_sse(a_planes[i], a_strides[i], b_planes[i], b_strides[i], w, h);
    psnr->sse[1 + i] = sse;
    psnr->samples[1 + i] = samples;
    psnr->psnr[1 + i] = vpx_sse_to_psnr(samples, peak, (double)sse);

    total_sse += sse;
    total_samples += samples;
  }

  psnr->sse[0] = total_sse;
  psnr->samples[0] = total_samples;
  psnr->psnr[0] =
      vpx_sse_to_psnr((double)total_samples, peak, (double)total_sse);
  psnr->spatial_layer_id = spatial_layer_id;
}
