/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/convert_from.h"

#include "libyuv/basic_types.h"
#include "libyuv/convert.h"  // For I420Copy
#include "libyuv/cpu_id.h"
#include "libyuv/planar_functions.h"
#include "libyuv/rotate.h"
#include "libyuv/row.h"
#include "libyuv/scale.h"  // For ScalePlane()
#include "libyuv/video_common.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define SUBSAMPLE(v, a, s) (v < 0) ? (-((-v + a) >> s)) : ((v + a) >> s)
static __inline int Abs(int v) {
  return v >= 0 ? v : -v;
}

// I420 To any I4xx YUV format with mirroring.
static int I420ToI4xx(const uint8_t* src_y,
                      int src_stride_y,
                      const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_y,
                      int dst_stride_y,
                      uint8_t* dst_u,
                      int dst_stride_u,
                      uint8_t* dst_v,
                      int dst_stride_v,
                      int src_y_width,
                      int src_y_height,
                      int dst_uv_width,
                      int dst_uv_height) {
  const int dst_y_width = Abs(src_y_width);
  const int dst_y_height = Abs(src_y_height);
  const int src_uv_width = SUBSAMPLE(src_y_width, 1, 1);
  const int src_uv_height = SUBSAMPLE(src_y_height, 1, 1);
  if (src_y_width == 0 || src_y_height == 0 || dst_uv_width <= 0 ||
      dst_uv_height <= 0) {
    return -1;
  }
  if (dst_y) {
    ScalePlane(src_y, src_stride_y, src_y_width, src_y_height, dst_y,
               dst_stride_y, dst_y_width, dst_y_height, kFilterBilinear);
  }
  ScalePlane(src_u, src_stride_u, src_uv_width, src_uv_height, dst_u,
             dst_stride_u, dst_uv_width, dst_uv_height, kFilterBilinear);
  ScalePlane(src_v, src_stride_v, src_uv_width, src_uv_height, dst_v,
             dst_stride_v, dst_uv_width, dst_uv_height, kFilterBilinear);
  return 0;
}

// Convert 8 bit YUV to 10 bit.
LIBYUV_API
int I420ToI010(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if (!src_u || !src_v || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (halfheight - 1) * src_stride_u;
    src_v = src_v + (halfheight - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  // Convert Y plane.
  Convert8To16Plane(src_y, src_stride_y, dst_y, dst_stride_y, 1024, width,
                    height);
  // Convert UV planes.
  Convert8To16Plane(src_u, src_stride_u, dst_u, dst_stride_u, 1024, halfwidth,
                    halfheight);
  Convert8To16Plane(src_v, src_stride_v, dst_v, dst_stride_v, 1024, halfwidth,
                    halfheight);
  return 0;
}

// 420 chroma is 1/2 width, 1/2 height
// 422 chroma is 1/2 width, 1x height
LIBYUV_API
int I420ToI422(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  const int dst_uv_width = (Abs(width) + 1) >> 1;
  const int dst_uv_height = Abs(height);
  return I420ToI4xx(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, dst_uv_width,
                    dst_uv_height);
}

// 420 chroma is 1/2 width, 1/2 height
// 444 chroma is 1x width, 1x height
LIBYUV_API
int I420ToI444(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  const int dst_uv_width = Abs(width);
  const int dst_uv_height = Abs(height);
  return I420ToI4xx(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, dst_uv_width,
                    dst_uv_height);
}

// Copy to I400. Source can be I420,422,444,400,NV12,NV21
LIBYUV_API
int I400Copy(const uint8_t* src_y,
             int src_stride_y,
             uint8_t* dst_y,
             int dst_stride_y,
             int width,
             int height) {
  if (!src_y || !dst_y || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_stride_y = -src_stride_y;
  }
  CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  return 0;
}

LIBYUV_API
int I422ToYUY2(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_yuy2,
               int dst_stride_yuy2,
               int width,
               int height) {
  int y;
  void (*I422ToYUY2Row)(const uint8_t* src_y, const uint8_t* src_u,
                        const uint8_t* src_v, uint8_t* dst_yuy2, int width) =
      I422ToYUY2Row_C;
  if (!src_y || !src_u || !src_v || !dst_yuy2 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_yuy2 = dst_yuy2 + (height - 1) * dst_stride_yuy2;
    dst_stride_yuy2 = -dst_stride_yuy2;
  }
  // Coalesce rows.
  if (src_stride_y == width && src_stride_u * 2 == width &&
      src_stride_v * 2 == width && dst_stride_yuy2 == width * 2) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_yuy2 = 0;
  }
#if defined(HAS_I422TOYUY2ROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    I422ToYUY2Row = I422ToYUY2Row_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      I422ToYUY2Row = I422ToYUY2Row_SSE2;
    }
  }
#endif
#if defined(HAS_I422TOYUY2ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToYUY2Row = I422ToYUY2Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      I422ToYUY2Row = I422ToYUY2Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOYUY2ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToYUY2Row = I422ToYUY2Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      I422ToYUY2Row = I422ToYUY2Row_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToYUY2Row(src_y, src_u, src_v, dst_yuy2, width);
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_yuy2 += dst_stride_yuy2;
  }
  return 0;
}

LIBYUV_API
int I420ToYUY2(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_yuy2,
               int dst_stride_yuy2,
               int width,
               int height) {
  int y;
  void (*I422ToYUY2Row)(const uint8_t* src_y, const uint8_t* src_u,
                        const uint8_t* src_v, uint8_t* dst_yuy2, int width) =
      I422ToYUY2Row_C;
  if (!src_y || !src_u || !src_v || !dst_yuy2 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_yuy2 = dst_yuy2 + (height - 1) * dst_stride_yuy2;
    dst_stride_yuy2 = -dst_stride_yuy2;
  }
#if defined(HAS_I422TOYUY2ROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    I422ToYUY2Row = I422ToYUY2Row_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      I422ToYUY2Row = I422ToYUY2Row_SSE2;
    }
  }
#endif
#if defined(HAS_I422TOYUY2ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToYUY2Row = I422ToYUY2Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      I422ToYUY2Row = I422ToYUY2Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOYUY2ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToYUY2Row = I422ToYUY2Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      I422ToYUY2Row = I422ToYUY2Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TOYUY2ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToYUY2Row = I422ToYUY2Row_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      I422ToYUY2Row = I422ToYUY2Row_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    I422ToYUY2Row(src_y, src_u, src_v, dst_yuy2, width);
    I422ToYUY2Row(src_y + src_stride_y, src_u, src_v,
                  dst_yuy2 + dst_stride_yuy2, width);
    src_y += src_stride_y * 2;
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_yuy2 += dst_stride_yuy2 * 2;
  }
  if (height & 1) {
    I422ToYUY2Row(src_y, src_u, src_v, dst_yuy2, width);
  }
  return 0;
}

LIBYUV_API
int I422ToUYVY(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_uyvy,
               int dst_stride_uyvy,
               int width,
               int height) {
  int y;
  void (*I422ToUYVYRow)(const uint8_t* src_y, const uint8_t* src_u,
                        const uint8_t* src_v, uint8_t* dst_uyvy, int width) =
      I422ToUYVYRow_C;
  if (!src_y || !src_u || !src_v || !dst_uyvy || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_uyvy = dst_uyvy + (height - 1) * dst_stride_uyvy;
    dst_stride_uyvy = -dst_stride_uyvy;
  }
  // Coalesce rows.
  if (src_stride_y == width && src_stride_u * 2 == width &&
      src_stride_v * 2 == width && dst_stride_uyvy == width * 2) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_uyvy = 0;
  }
#if defined(HAS_I422TOUYVYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    I422ToUYVYRow = I422ToUYVYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      I422ToUYVYRow = I422ToUYVYRow_SSE2;
    }
  }
#endif
#if defined(HAS_I422TOUYVYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToUYVYRow = I422ToUYVYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      I422ToUYVYRow = I422ToUYVYRow_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOUYVYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToUYVYRow = I422ToUYVYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      I422ToUYVYRow = I422ToUYVYRow_NEON;
    }
  }
#endif
#if defined(HAS_I422TOUYVYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToUYVYRow = I422ToUYVYRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      I422ToUYVYRow = I422ToUYVYRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToUYVYRow(src_y, src_u, src_v, dst_uyvy, width);
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_uyvy += dst_stride_uyvy;
  }
  return 0;
}

LIBYUV_API
int I420ToUYVY(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_uyvy,
               int dst_stride_uyvy,
               int width,
               int height) {
  int y;
  void (*I422ToUYVYRow)(const uint8_t* src_y, const uint8_t* src_u,
                        const uint8_t* src_v, uint8_t* dst_uyvy, int width) =
      I422ToUYVYRow_C;
  if (!src_y || !src_u || !src_v || !dst_uyvy || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_uyvy = dst_uyvy + (height - 1) * dst_stride_uyvy;
    dst_stride_uyvy = -dst_stride_uyvy;
  }
#if defined(HAS_I422TOUYVYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    I422ToUYVYRow = I422ToUYVYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      I422ToUYVYRow = I422ToUYVYRow_SSE2;
    }
  }
#endif
#if defined(HAS_I422TOUYVYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToUYVYRow = I422ToUYVYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      I422ToUYVYRow = I422ToUYVYRow_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOUYVYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToUYVYRow = I422ToUYVYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      I422ToUYVYRow = I422ToUYVYRow_NEON;
    }
  }
#endif
#if defined(HAS_I422TOUYVYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToUYVYRow = I422ToUYVYRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      I422ToUYVYRow = I422ToUYVYRow_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    I422ToUYVYRow(src_y, src_u, src_v, dst_uyvy, width);
    I422ToUYVYRow(src_y + src_stride_y, src_u, src_v,
                  dst_uyvy + dst_stride_uyvy, width);
    src_y += src_stride_y * 2;
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_uyvy += dst_stride_uyvy * 2;
  }
  if (height & 1) {
    I422ToUYVYRow(src_y, src_u, src_v, dst_uyvy, width);
  }
  return 0;
}

// TODO(fbarchard): test negative height for invert.
LIBYUV_API
int I420ToNV12(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  if (!src_y || !src_u || !src_v || !dst_y || !dst_uv || width <= 0 ||
      height == 0) {
    return -1;
  }
  int halfwidth = (width + 1) / 2;
  int halfheight = height > 0 ? (height + 1) / 2 : (height - 1) / 2;
  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  MergeUVPlane(src_u, src_stride_u, src_v, src_stride_v, dst_uv, dst_stride_uv,
               halfwidth, halfheight);
  return 0;
}

LIBYUV_API
int I420ToNV21(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height) {
  return I420ToNV12(src_y, src_stride_y, src_v, src_stride_v, src_u,
                    src_stride_u, dst_y, dst_stride_y, dst_vu, dst_stride_vu,
                    width, height);
}

// Convert I422 to RGBA with matrix
static int I420ToRGBAMatrix(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_u,
                            int src_stride_u,
                            const uint8_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_rgba,
                            int dst_stride_rgba,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height) {
  int y;
  void (*I422ToRGBARow)(const uint8_t* y_buf, const uint8_t* u_buf,
                        const uint8_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I422ToRGBARow_C;
  if (!src_y || !src_u || !src_v || !dst_rgba || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgba = dst_rgba + (height - 1) * dst_stride_rgba;
    dst_stride_rgba = -dst_stride_rgba;
  }
#if defined(HAS_I422TORGBAROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToRGBARow = I422ToRGBARow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGBARow = I422ToRGBARow_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TORGBAROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToRGBARow = I422ToRGBARow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGBARow = I422ToRGBARow_AVX2;
    }
  }
#endif
#if defined(HAS_I422TORGBAROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToRGBARow = I422ToRGBARow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGBARow = I422ToRGBARow_NEON;
    }
  }
#endif
#if defined(HAS_I422TORGBAROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToRGBARow = I422ToRGBARow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGBARow = I422ToRGBARow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToRGBARow(src_y, src_u, src_v, dst_rgba, yuvconstants, width);
    dst_rgba += dst_stride_rgba;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to RGBA.
LIBYUV_API
int I420ToRGBA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_rgba,
               int dst_stride_rgba,
               int width,
               int height) {
  return I420ToRGBAMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_rgba, dst_stride_rgba,
                          &kYuvI601Constants, width, height);
}

// Convert I420 to BGRA.
LIBYUV_API
int I420ToBGRA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_bgra,
               int dst_stride_bgra,
               int width,
               int height) {
  return I420ToRGBAMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_bgra, dst_stride_bgra,
                          &kYvuI601Constants,  // Use Yvu matrix
                          width, height);
}

// Convert I420 to RGB24 with matrix
static int I420ToRGB24Matrix(const uint8_t* src_y,
                             int src_stride_y,
                             const uint8_t* src_u,
                             int src_stride_u,
                             const uint8_t* src_v,
                             int src_stride_v,
                             uint8_t* dst_rgb24,
                             int dst_stride_rgb24,
                             const struct YuvConstants* yuvconstants,
                             int width,
                             int height) {
  int y;
  void (*I422ToRGB24Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                         const uint8_t* v_buf, uint8_t* rgb_buf,
                         const struct YuvConstants* yuvconstants, int width) =
      I422ToRGB24Row_C;
  if (!src_y || !src_u || !src_v || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb24 = dst_rgb24 + (height - 1) * dst_stride_rgb24;
    dst_stride_rgb24 = -dst_stride_rgb24;
  }
#if defined(HAS_I422TORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB24Row = I422ToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB24Row = I422ToRGB24Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB24Row = I422ToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB24Row = I422ToRGB24Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToRGB24Row(src_y, src_u, src_v, dst_rgb24, yuvconstants, width);
    dst_rgb24 += dst_stride_rgb24;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to RGB24.
LIBYUV_API
int I420ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_rgb24, dst_stride_rgb24,
                           &kYuvI601Constants, width, height);
}

// Convert I420 to RAW.
LIBYUV_API
int I420ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_v,
                           src_stride_v,  // Swap U and V
                           src_u, src_stride_u, dst_raw, dst_stride_raw,
                           &kYvuI601Constants,  // Use Yvu matrix
                           width, height);
}

// Convert H420 to RGB24.
LIBYUV_API
int H420ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_rgb24, dst_stride_rgb24,
                           &kYuvH709Constants, width, height);
}

// Convert H420 to RAW.
LIBYUV_API
int H420ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_v,
                           src_stride_v,  // Swap U and V
                           src_u, src_stride_u, dst_raw, dst_stride_raw,
                           &kYvuH709Constants,  // Use Yvu matrix
                           width, height);
}

// Convert I420 to ARGB1555.
LIBYUV_API
int I420ToARGB1555(const uint8_t* src_y,
                   int src_stride_y,
                   const uint8_t* src_u,
                   int src_stride_u,
                   const uint8_t* src_v,
                   int src_stride_v,
                   uint8_t* dst_argb1555,
                   int dst_stride_argb1555,
                   int width,
                   int height) {
  int y;
  void (*I422ToARGB1555Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                            const uint8_t* v_buf, uint8_t* rgb_buf,
                            const struct YuvConstants* yuvconstants,
                            int width) = I422ToARGB1555Row_C;
  if (!src_y || !src_u || !src_v || !dst_argb1555 || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb1555 = dst_argb1555 + (height - 1) * dst_stride_argb1555;
    dst_stride_argb1555 = -dst_stride_argb1555;
  }
#if defined(HAS_I422TOARGB1555ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB1555Row = I422ToARGB1555Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGB1555Row = I422ToARGB1555Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB1555Row = I422ToARGB1555Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB1555Row = I422ToARGB1555Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToARGB1555Row(src_y, src_u, src_v, dst_argb1555, &kYuvI601Constants,
                      width);
    dst_argb1555 += dst_stride_argb1555;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to ARGB4444.
LIBYUV_API
int I420ToARGB4444(const uint8_t* src_y,
                   int src_stride_y,
                   const uint8_t* src_u,
                   int src_stride_u,
                   const uint8_t* src_v,
                   int src_stride_v,
                   uint8_t* dst_argb4444,
                   int dst_stride_argb4444,
                   int width,
                   int height) {
  int y;
  void (*I422ToARGB4444Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                            const uint8_t* v_buf, uint8_t* rgb_buf,
                            const struct YuvConstants* yuvconstants,
                            int width) = I422ToARGB4444Row_C;
  if (!src_y || !src_u || !src_v || !dst_argb4444 || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb4444 = dst_argb4444 + (height - 1) * dst_stride_argb4444;
    dst_stride_argb4444 = -dst_stride_argb4444;
  }
#if defined(HAS_I422TOARGB4444ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB4444Row = I422ToARGB4444Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGB4444Row = I422ToARGB4444Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB4444Row = I422ToARGB4444Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB4444Row = I422ToARGB4444Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToARGB4444Row(src_y, src_u, src_v, dst_argb4444, &kYuvI601Constants,
                      width);
    dst_argb4444 += dst_stride_argb4444;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to RGB565.
LIBYUV_API
int I420ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height) {
  int y;
  void (*I422ToRGB565Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                          const uint8_t* v_buf, uint8_t* rgb_buf,
                          const struct YuvConstants* yuvconstants, int width) =
      I422ToRGB565Row_C;
  if (!src_y || !src_u || !src_v || !dst_rgb565 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb565 = dst_rgb565 + (height - 1) * dst_stride_rgb565;
    dst_stride_rgb565 = -dst_stride_rgb565;
  }
#if defined(HAS_I422TORGB565ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB565Row = I422ToRGB565Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToRGB565Row(src_y, src_u, src_v, dst_rgb565, &kYuvI601Constants, width);
    dst_rgb565 += dst_stride_rgb565;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I422 to RGB565.
LIBYUV_API
int I422ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height) {
  int y;
  void (*I422ToRGB565Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                          const uint8_t* v_buf, uint8_t* rgb_buf,
                          const struct YuvConstants* yuvconstants, int width) =
      I422ToRGB565Row_C;
  if (!src_y || !src_u || !src_v || !dst_rgb565 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb565 = dst_rgb565 + (height - 1) * dst_stride_rgb565;
    dst_stride_rgb565 = -dst_stride_rgb565;
  }
#if defined(HAS_I422TORGB565ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB565Row = I422ToRGB565Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToRGB565Row(src_y, src_u, src_v, dst_rgb565, &kYuvI601Constants, width);
    dst_rgb565 += dst_stride_rgb565;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Ordered 8x8 dither for 888 to 565.  Values from 0 to 7.
static const uint8_t kDither565_4x4[16] = {
    0, 4, 1, 5, 6, 2, 7, 3, 1, 5, 0, 4, 7, 3, 6, 2,
};

// Convert I420 to RGB565 with dithering.
LIBYUV_API
int I420ToRGB565Dither(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_u,
                       int src_stride_u,
                       const uint8_t* src_v,
                       int src_stride_v,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const uint8_t* dither4x4,
                       int width,
                       int height) {
  int y;
  void (*I422ToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                        const uint8_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I422ToARGBRow_C;
  void (*ARGBToRGB565DitherRow)(const uint8_t* src_argb, uint8_t* dst_rgb,
                                const uint32_t dither4, int width) =
      ARGBToRGB565DitherRow_C;
  if (!src_y || !src_u || !src_v || !dst_rgb565 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb565 = dst_rgb565 + (height - 1) * dst_stride_rgb565;
    dst_stride_rgb565 = -dst_stride_rgb565;
  }
  if (!dither4x4) {
    dither4x4 = kDither565_4x4;
  }
#if defined(HAS_I422TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToARGBRow = I422ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGBRow = I422ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToARGBRow = I422ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGBRow = I422ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToARGBRow = I422ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGBRow = I422ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToARGBRow = I422ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGBRow = I422ToARGBRow_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTORGB565DITHERROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_Any_SSE2;
    if (IS_ALIGNED(width, 4)) {
      ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBTORGB565DITHERROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTORGB565DITHERROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTORGB565DITHERROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_MSA;
    }
  }
#endif
  {
    // Allocate a row of argb.
    align_buffer_64(row_argb, width * 4);
    for (y = 0; y < height; ++y) {
      I422ToARGBRow(src_y, src_u, src_v, row_argb, &kYuvI601Constants, width);
      ARGBToRGB565DitherRow(row_argb, dst_rgb565,
                            *(const uint32_t*)(dither4x4 + ((y & 3) << 2)),
                            width);
      dst_rgb565 += dst_stride_rgb565;
      src_y += src_stride_y;
      if (y & 1) {
        src_u += src_stride_u;
        src_v += src_stride_v;
      }
    }
    free_aligned_buffer_64(row_argb);
  }
  return 0;
}

// Convert I420 to AR30 with matrix
static int I420ToAR30Matrix(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_u,
                            int src_stride_u,
                            const uint8_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_ar30,
                            int dst_stride_ar30,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height) {
  int y;
  void (*I422ToAR30Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                        const uint8_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I422ToAR30Row_C;

  if (!src_y || !src_u || !src_v || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }

#if defined(HAS_I422TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToAR30Row = I422ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToAR30Row = I422ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToAR30Row = I422ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToAR30Row = I422ToAR30Row_AVX2;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToAR30Row(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to AR30.
LIBYUV_API
int I420ToAR30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  return I420ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_ar30, dst_stride_ar30,
                          &kYuvI601Constants, width, height);
}

// Convert H420 to AR30.
LIBYUV_API
int H420ToAR30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  return I420ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_ar30, dst_stride_ar30,
                          &kYvuH709Constants, width, height);
}

// Convert I420 to specified format
LIBYUV_API
int ConvertFromI420(const uint8_t* y,
                    int y_stride,
                    const uint8_t* u,
                    int u_stride,
                    const uint8_t* v,
                    int v_stride,
                    uint8_t* dst_sample,
                    int dst_sample_stride,
                    int width,
                    int height,
                    uint32_t fourcc) {
  uint32_t format = CanonicalFourCC(fourcc);
  int r = 0;
  if (!y || !u || !v || !dst_sample || width <= 0 || height == 0) {
    return -1;
  }
  switch (format) {
    // Single plane formats
    case FOURCC_YUY2:
      r = I420ToYUY2(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 2, width,
                     height);
      break;
    case FOURCC_UYVY:
      r = I420ToUYVY(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 2, width,
                     height);
      break;
    case FOURCC_RGBP:
      r = I420ToRGB565(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                       dst_sample_stride ? dst_sample_stride : width * 2, width,
                       height);
      break;
    case FOURCC_RGBO:
      r = I420ToARGB1555(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                         dst_sample_stride ? dst_sample_stride : width * 2,
                         width, height);
      break;
    case FOURCC_R444:
      r = I420ToARGB4444(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                         dst_sample_stride ? dst_sample_stride : width * 2,
                         width, height);
      break;
    case FOURCC_24BG:
      r = I420ToRGB24(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                      dst_sample_stride ? dst_sample_stride : width * 3, width,
                      height);
      break;
    case FOURCC_RAW:
      r = I420ToRAW(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                    dst_sample_stride ? dst_sample_stride : width * 3, width,
                    height);
      break;
    case FOURCC_ARGB:
      r = I420ToARGB(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 4, width,
                     height);
      break;
    case FOURCC_BGRA:
      r = I420ToBGRA(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 4, width,
                     height);
      break;
    case FOURCC_ABGR:
      r = I420ToABGR(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 4, width,
                     height);
      break;
    case FOURCC_RGBA:
      r = I420ToRGBA(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 4, width,
                     height);
      break;
    case FOURCC_AR30:
      r = I420ToAR30(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 4, width,
                     height);
      break;
    case FOURCC_I400:
      r = I400Copy(y, y_stride, dst_sample,
                   dst_sample_stride ? dst_sample_stride : width, width,
                   height);
      break;
    case FOURCC_NV12: {
      uint8_t* dst_uv = dst_sample + width * height;
      r = I420ToNV12(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride ? dst_sample_stride : width, dst_uv,
                     dst_sample_stride ? dst_sample_stride : width, width,
                     height);
      break;
    }
    case FOURCC_NV21: {
      uint8_t* dst_vu = dst_sample + width * height;
      r = I420ToNV21(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride ? dst_sample_stride : width, dst_vu,
                     dst_sample_stride ? dst_sample_stride : width, width,
                     height);
      break;
    }
    // TODO(fbarchard): Add M420.
    // Triplanar formats
    case FOURCC_I420:
    case FOURCC_YV12: {
      dst_sample_stride = dst_sample_stride ? dst_sample_stride : width;
      int halfstride = (dst_sample_stride + 1) / 2;
      int halfheight = (height + 1) / 2;
      uint8_t* dst_u;
      uint8_t* dst_v;
      if (format == FOURCC_YV12) {
        dst_v = dst_sample + dst_sample_stride * height;
        dst_u = dst_v + halfstride * halfheight;
      } else {
        dst_u = dst_sample + dst_sample_stride * height;
        dst_v = dst_u + halfstride * halfheight;
      }
      r = I420Copy(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                   dst_sample_stride, dst_u, halfstride, dst_v, halfstride,
                   width, height);
      break;
    }
    case FOURCC_I422:
    case FOURCC_YV16: {
      dst_sample_stride = dst_sample_stride ? dst_sample_stride : width;
      int halfstride = (dst_sample_stride + 1) / 2;
      uint8_t* dst_u;
      uint8_t* dst_v;
      if (format == FOURCC_YV16) {
        dst_v = dst_sample + dst_sample_stride * height;
        dst_u = dst_v + halfstride * height;
      } else {
        dst_u = dst_sample + dst_sample_stride * height;
        dst_v = dst_u + halfstride * height;
      }
      r = I420ToI422(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride, dst_u, halfstride, dst_v, halfstride,
                     width, height);
      break;
    }
    case FOURCC_I444:
    case FOURCC_YV24: {
      dst_sample_stride = dst_sample_stride ? dst_sample_stride : width;
      uint8_t* dst_u;
      uint8_t* dst_v;
      if (format == FOURCC_YV24) {
        dst_v = dst_sample + dst_sample_stride * height;
        dst_u = dst_v + dst_sample_stride * height;
      } else {
        dst_u = dst_sample + dst_sample_stride * height;
        dst_v = dst_u + dst_sample_stride * height;
      }
      r = I420ToI444(y, y_stride, u, u_stride, v, v_stride, dst_sample,
                     dst_sample_stride, dst_u, dst_sample_stride, dst_v,
                     dst_sample_stride, width, height);
      break;
    }
    // Formats not supported - MJPG, biplanar, some rgb formats.
    default:
      return -1;  // unknown fourcc - return failure code.
  }
  return r;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
