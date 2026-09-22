// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// WebPPicture tools for measuring distortion
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/webp/encode.h"

#if !(defined(WEBP_DISABLE_STATS) || defined(WEBP_REDUCE_SIZE))

#include <math.h>
#include <stdlib.h>

#include "src/webp/types.h"
#include "src/dsp/dsp.h"
#include "src/enc/vp8i_enc.h"
#include "src/utils/utils.h"

typedef double (*AccumulateFunc)(const uint8_t* src, int src_stride,
                                 const uint8_t* ref, int ref_stride,
                                 int w, int h);

//------------------------------------------------------------------------------
// local-min distortion
//
// For every pixel in the *reference* picture, we search for the local best
// match in the compressed image. This is not a symmetrical measure.

#define RADIUS 2  // search radius. Shouldn't be too large.

static double AccumulateLSIM(const uint8_t* src, int src_stride,
                             const uint8_t* ref, int ref_stride,
                             int w, int h) {
  int x, y;
  double total_sse = 0.;
  for (y = 0; y < h; ++y) {
    const int y_0 = (y - RADIUS < 0) ? 0 : y - RADIUS;
    const int y_1 = (y + RADIUS + 1 >= h) ? h : y + RADIUS + 1;
    for (x = 0; x < w; ++x) {
      const int x_0 = (x - RADIUS < 0) ? 0 : x - RADIUS;
      const int x_1 = (x + RADIUS + 1 >= w) ? w : x + RADIUS + 1;
      double best_sse = 255. * 255.;
      const double value = (double)ref[y * ref_stride + x];
      int i, j;
      for (j = y_0; j < y_1; ++j) {
        const uint8_t* const s = src + j * src_stride;
        for (i = x_0; i < x_1; ++i) {
          const double diff = s[i] - value;
          const double sse = diff * diff;
          if (sse < best_sse) best_sse = sse;
        }
      }
      total_sse += best_sse;
    }
  }
  return total_sse;
}
#undef RADIUS

static double AccumulateSSE(const uint8_t* src, int src_stride,
                            const uint8_t* ref, int ref_stride,
                            int w, int h) {
  int y;
  double total_sse = 0.;
  for (y = 0; y < h; ++y) {
    total_sse += VP8AccumulateSSE(src, ref, w);
    src += src_stride;
    ref += ref_stride;
  }
  return total_sse;
}

//------------------------------------------------------------------------------

static double AccumulateSSIM(const uint8_t* src, int src_stride,
                             const uint8_t* ref, int ref_stride,
                             int w, int h) {
  const int w0 = (w < VP8_SSIM_KERNEL) ? w : VP8_SSIM_KERNEL;
  const int w1 = w - VP8_SSIM_KERNEL - 1;
  const int h0 = (h < VP8_SSIM_KERNEL) ? h : VP8_SSIM_KERNEL;
  const int h1 = h - VP8_SSIM_KERNEL - 1;
  int x, y;
  double sum = 0.;
  for (y = 0; y < h0; ++y) {
    for (x = 0; x < w; ++x) {
      sum += VP8SSIMGetClipped(src, src_stride, ref, ref_stride, x, y, w, h);
    }
  }
  for (; y < h1; ++y) {
    for (x = 0; x < w0; ++x) {
      sum += VP8SSIMGetClipped(src, src_stride, ref, ref_stride, x, y, w, h);
    }
    for (; x < w1; ++x) {
      const int off1 = x - VP8_SSIM_KERNEL + (y - VP8_SSIM_KERNEL) * src_stride;
      const int off2 = x - VP8_SSIM_KERNEL + (y - VP8_SSIM_KERNEL) * ref_stride;
      sum += VP8SSIMGet(src + off1, src_stride, ref + off2, ref_stride);
    }
    for (; x < w; ++x) {
      sum += VP8SSIMGetClipped(src, src_stride, ref, ref_stride, x, y, w, h);
    }
  }
  for (; y < h; ++y) {
    for (x = 0; x < w; ++x) {
      sum += VP8SSIMGetClipped(src, src_stride, ref, ref_stride, x, y, w, h);
    }
  }
  return sum;
}

//------------------------------------------------------------------------------
// Distortion

// Max value returned in case of exact similarity.
static const double kMinDistortion_dB = 99.;

static double GetPSNR(double v, double size) {
  return (v > 0. && size > 0.) ? -4.3429448 * log(v / (size * 255 * 255.))
                               : kMinDistortion_dB;
}

static double GetLogSSIM(double v, double size) {
  v = (size > 0.) ? v / size : 1.;
  return (v < 1.) ? -10.0 * log10(1. - v) : kMinDistortion_dB;
}

int WebPPlaneDistortion(const uint8_t* src, size_t src_stride,
                        const uint8_t* ref, size_t ref_stride,
                        int width, int height, size_t x_step,
                        int type, float* distortion, float* result) {
  uint8_t* allocated = NULL;
  const AccumulateFunc metric = (type == 0) ? AccumulateSSE :
                                (type == 1) ? AccumulateSSIM :
                                              AccumulateLSIM;
  if (src == NULL || ref == NULL ||
      src_stride < x_step * width || ref_stride < x_step * width ||
      result == NULL || distortion == NULL) {
    return 0;
  }

  VP8SSIMDspInit();
  if (x_step != 1) {   // extract a packed plane if needed
    int x, y;
    uint8_t* tmp1;
    uint8_t* tmp2;
    allocated =
        (uint8_t*)WebPSafeMalloc(2ULL * width * height, sizeof(*allocated));
    if (allocated == NULL) return 0;
    tmp1 = allocated;
    tmp2 = tmp1 + (size_t)width * height;
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        tmp1[x + y * width] = src[x * x_step + y * src_stride];
        tmp2[x + y * width] = ref[x * x_step + y * ref_stride];
      }
    }
    src = tmp1;
    ref = tmp2;
  }
  *distortion = (float)metric(src, width, ref, width, width, height);
  WebPSafeFree(allocated);

  *result = (type == 1) ? (float)GetLogSSIM(*distortion, (double)width * height)
                        : (float)GetPSNR(*distortion, (double)width * height);
  return 1;
}

#ifdef WORDS_BIGENDIAN
#define BLUE_OFFSET 3   // uint32_t 0x000000ff is 0x00,00,00,ff in memory
#else
#define BLUE_OFFSET 0   // uint32_t 0x000000ff is 0xff,00,00,00 in memory
#endif

int WebPPictureDistortion(const WebPPicture* src, const WebPPicture* ref,
                          int type, float results[5]) {
  int w, h, c;
  int ok = 0;
  WebPPicture p0, p1;
  double total_size = 0., total_distortion = 0.;
  if (src == NULL || ref == NULL ||
      src->width != ref->width || src->height != ref->height ||
      results == NULL) {
    return 0;
  }

  VP8SSIMDspInit();
  if (!WebPPictureInit(&p0) || !WebPPictureInit(&p1)) return 0;
  w = src->width;
  h = src->height;
  if (!WebPPictureView(src, 0, 0, w, h, &p0)) goto Error;
  if (!WebPPictureView(ref, 0, 0, w, h, &p1)) goto Error;

  // We always measure distortion in ARGB space.
  if (p0.use_argb == 0 && !WebPPictureYUVAToARGB(&p0)) goto Error;
  if (p1.use_argb == 0 && !WebPPictureYUVAToARGB(&p1)) goto Error;
  for (c = 0; c < 4; ++c) {
    float distortion;
    const size_t stride0 = 4 * (size_t)p0.argb_stride;
    const size_t stride1 = 4 * (size_t)p1.argb_stride;
    // results are reported as BGRA
    const int offset = c ^ BLUE_OFFSET;
    if (!WebPPlaneDistortion((const uint8_t*)p0.argb + offset, stride0,
                             (const uint8_t*)p1.argb + offset, stride1,
                             w, h, 4, type, &distortion, results + c)) {
      goto Error;
    }
    total_distortion += distortion;
    total_size += w * h;
  }

  results[4] = (type == 1) ? (float)GetLogSSIM(total_distortion, total_size)
                           : (float)GetPSNR(total_distortion, total_size);
  ok = 1;

 Error:
  WebPPictureFree(&p0);
  WebPPictureFree(&p1);
  return ok;
}

#undef BLUE_OFFSET

#else  // defined(WEBP_DISABLE_STATS)
int WebPPlaneDistortion(const uint8_t* src, size_t src_stride,
                        const uint8_t* ref, size_t ref_stride,
                        int width, int height, size_t x_step,
                        int type, float* distortion, float* result) {
  (void)src;
  (void)src_stride;
  (void)ref;
  (void)ref_stride;
  (void)width;
  (void)height;
  (void)x_step;
  (void)type;
  if (distortion == NULL || result == NULL) return 0;
  *distortion = 0.f;
  *result = 0.f;
  return 1;
}

int WebPPictureDistortion(const WebPPicture* src, const WebPPicture* ref,
                          int type, float results[5]) {
  int i;
  (void)src;
  (void)ref;
  (void)type;
  if (results == NULL) return 0;
  for (i = 0; i < 5; ++i) results[i] = 0.f;
  return 1;
}

#endif  // !defined(WEBP_DISABLE_STATS)
