/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/convert.h"

#include "libyuv/basic_types.h"
#include "libyuv/cpu_id.h"
#include "libyuv/planar_functions.h"
#include "libyuv/rotate.h"
#include "libyuv/row.h"
#include "libyuv/scale.h"  // For ScalePlane()

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define SUBSAMPLE(v, a, s) (v < 0) ? (-((-v + a) >> s)) : ((v + a) >> s)
static __inline int Abs(int v) {
  return v >= 0 ? v : -v;
}

// Any I4xx To I420 format with mirroring.
static int I4xxToI420(const uint8_t* src_y,
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
                      int src_uv_width,
                      int src_uv_height) {
  const int dst_y_width = Abs(src_y_width);
  const int dst_y_height = Abs(src_y_height);
  const int dst_uv_width = SUBSAMPLE(dst_y_width, 1, 1);
  const int dst_uv_height = SUBSAMPLE(dst_y_height, 1, 1);
  if (src_uv_width == 0 || src_uv_height == 0) {
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

// Copy I420 with optional flipping.
// TODO(fbarchard): Use Scale plane which supports mirroring, but ensure
// is does row coalescing.
LIBYUV_API
int I420Copy(const uint8_t* src_y,
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

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  // Copy UV planes.
  CopyPlane(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, halfheight);
  CopyPlane(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, halfheight);
  return 0;
}

// Copy I010 with optional flipping.
LIBYUV_API
int I010Copy(const uint16_t* src_y,
             int src_stride_y,
             const uint16_t* src_u,
             int src_stride_u,
             const uint16_t* src_v,
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

  if (dst_y) {
    CopyPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  // Copy UV planes.
  CopyPlane_16(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, halfheight);
  CopyPlane_16(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, halfheight);
  return 0;
}

// Convert 10 bit YUV to 8 bit.
LIBYUV_API
int I010ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
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
  Convert16To8Plane(src_y, src_stride_y, dst_y, dst_stride_y, 16384, width,
                    height);
  // Convert UV planes.
  Convert16To8Plane(src_u, src_stride_u, dst_u, dst_stride_u, 16384, halfwidth,
                    halfheight);
  Convert16To8Plane(src_v, src_stride_v, dst_v, dst_stride_v, 16384, halfwidth,
                    halfheight);
  return 0;
}

// 422 chroma is 1/2 width, 1x height
// 420 chroma is 1/2 width, 1/2 height
LIBYUV_API
int I422ToI420(const uint8_t* src_y,
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
  const int src_uv_width = SUBSAMPLE(width, 1, 1);
  return I4xxToI420(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, src_uv_width, height);
}

// 444 chroma is 1x width, 1x height
// 420 chroma is 1/2 width, 1/2 height
LIBYUV_API
int I444ToI420(const uint8_t* src_y,
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
  return I4xxToI420(src_y, src_stride_y, src_u, src_stride_u, src_v,
                    src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                    dst_v, dst_stride_v, width, height, width, height);
}

// I400 is greyscale typically used in MJPG
LIBYUV_API
int I400ToI420(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if (!dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_stride_y = -src_stride_y;
  }
  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  SetPlane(dst_u, dst_stride_u, halfwidth, halfheight, 128);
  SetPlane(dst_v, dst_stride_v, halfwidth, halfheight, 128);
  return 0;
}

static void CopyPlane2(const uint8_t* src,
                       int src_stride_0,
                       int src_stride_1,
                       uint8_t* dst,
                       int dst_stride,
                       int width,
                       int height) {
  int y;
  void (*CopyRow)(const uint8_t* src, uint8_t* dst, int width) = CopyRow_C;
#if defined(HAS_COPYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    CopyRow = IS_ALIGNED(width, 32) ? CopyRow_SSE2 : CopyRow_Any_SSE2;
  }
#endif
#if defined(HAS_COPYROW_AVX)
  if (TestCpuFlag(kCpuHasAVX)) {
    CopyRow = IS_ALIGNED(width, 64) ? CopyRow_AVX : CopyRow_Any_AVX;
  }
#endif
#if defined(HAS_COPYROW_ERMS)
  if (TestCpuFlag(kCpuHasERMS)) {
    CopyRow = CopyRow_ERMS;
  }
#endif
#if defined(HAS_COPYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    CopyRow = IS_ALIGNED(width, 32) ? CopyRow_NEON : CopyRow_Any_NEON;
  }
#endif

  // Copy plane
  for (y = 0; y < height - 1; y += 2) {
    CopyRow(src, dst, width);
    CopyRow(src + src_stride_0, dst + dst_stride, width);
    src += src_stride_0 + src_stride_1;
    dst += dst_stride * 2;
  }
  if (height & 1) {
    CopyRow(src, dst, width);
  }
}

// Support converting from FOURCC_M420
// Useful for bandwidth constrained transports like USB 1.0 and 2.0 and for
// easy conversion to I420.
// M420 format description:
// M420 is row biplanar 420: 2 rows of Y and 1 row of UV.
// Chroma is half width / half height. (420)
// src_stride_m420 is row planar. Normally this will be the width in pixels.
//   The UV plane is half width, but 2 values, so src_stride_m420 applies to
//   this as well as the two Y planes.
static int X420ToI420(const uint8_t* src_y,
                      int src_stride_y0,
                      int src_stride_y1,
                      const uint8_t* src_uv,
                      int src_stride_uv,
                      uint8_t* dst_y,
                      int dst_stride_y,
                      uint8_t* dst_u,
                      int dst_stride_u,
                      uint8_t* dst_v,
                      int dst_stride_v,
                      int width,
                      int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if (!src_uv || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    if (dst_y) {
      dst_y = dst_y + (height - 1) * dst_stride_y;
    }
    dst_u = dst_u + (halfheight - 1) * dst_stride_u;
    dst_v = dst_v + (halfheight - 1) * dst_stride_v;
    dst_stride_y = -dst_stride_y;
    dst_stride_u = -dst_stride_u;
    dst_stride_v = -dst_stride_v;
  }
  // Coalesce rows.
  if (src_stride_y0 == width && src_stride_y1 == width &&
      dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_y0 = src_stride_y1 = dst_stride_y = 0;
  }
  // Coalesce rows.
  if (src_stride_uv == halfwidth * 2 && dst_stride_u == halfwidth &&
      dst_stride_v == halfwidth) {
    halfwidth *= halfheight;
    halfheight = 1;
    src_stride_uv = dst_stride_u = dst_stride_v = 0;
  }

  if (dst_y) {
    if (src_stride_y0 == src_stride_y1) {
      CopyPlane(src_y, src_stride_y0, dst_y, dst_stride_y, width, height);
    } else {
      CopyPlane2(src_y, src_stride_y0, src_stride_y1, dst_y, dst_stride_y,
                 width, height);
    }
  }

  // Split UV plane - NV12 / NV21
  SplitUVPlane(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v, dst_stride_v,
               halfwidth, halfheight);

  return 0;
}

// Convert NV12 to I420.
LIBYUV_API
int NV12ToI420(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return X420ToI420(src_y, src_stride_y, src_stride_y, src_uv, src_stride_uv,
                    dst_y, dst_stride_y, dst_u, dst_stride_u, dst_v,
                    dst_stride_v, width, height);
}

// Convert NV21 to I420.  Same as NV12 but u and v pointers swapped.
LIBYUV_API
int NV21ToI420(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_vu,
               int src_stride_vu,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return X420ToI420(src_y, src_stride_y, src_stride_y, src_vu, src_stride_vu,
                    dst_y, dst_stride_y, dst_v, dst_stride_v, dst_u,
                    dst_stride_u, width, height);
}

// Convert M420 to I420.
LIBYUV_API
int M420ToI420(const uint8_t* src_m420,
               int src_stride_m420,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  return X420ToI420(src_m420, src_stride_m420, src_stride_m420 * 2,
                    src_m420 + src_stride_m420 * 2, src_stride_m420 * 3, dst_y,
                    dst_stride_y, dst_u, dst_stride_u, dst_v, dst_stride_v,
                    width, height);
}

// Convert YUY2 to I420.
LIBYUV_API
int YUY2ToI420(const uint8_t* src_yuy2,
               int src_stride_yuy2,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*YUY2ToUVRow)(const uint8_t* src_yuy2, int src_stride_yuy2,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      YUY2ToUVRow_C;
  void (*YUY2ToYRow)(const uint8_t* src_yuy2, uint8_t* dst_y, int width) =
      YUY2ToYRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_yuy2 = src_yuy2 + (height - 1) * src_stride_yuy2;
    src_stride_yuy2 = -src_stride_yuy2;
  }
#if defined(HAS_YUY2TOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    YUY2ToUVRow = YUY2ToUVRow_Any_SSE2;
    YUY2ToYRow = YUY2ToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToUVRow = YUY2ToUVRow_SSE2;
      YUY2ToYRow = YUY2ToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    YUY2ToUVRow = YUY2ToUVRow_Any_AVX2;
    YUY2ToYRow = YUY2ToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToUVRow = YUY2ToUVRow_AVX2;
      YUY2ToYRow = YUY2ToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    YUY2ToYRow = YUY2ToYRow_Any_NEON;
    YUY2ToUVRow = YUY2ToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToYRow = YUY2ToYRow_NEON;
      YUY2ToUVRow = YUY2ToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    YUY2ToYRow = YUY2ToYRow_Any_MSA;
    YUY2ToUVRow = YUY2ToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_MSA;
      YUY2ToUVRow = YUY2ToUVRow_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    YUY2ToUVRow(src_yuy2, src_stride_yuy2, dst_u, dst_v, width);
    YUY2ToYRow(src_yuy2, dst_y, width);
    YUY2ToYRow(src_yuy2 + src_stride_yuy2, dst_y + dst_stride_y, width);
    src_yuy2 += src_stride_yuy2 * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    YUY2ToUVRow(src_yuy2, 0, dst_u, dst_v, width);
    YUY2ToYRow(src_yuy2, dst_y, width);
  }
  return 0;
}

// Convert UYVY to I420.
LIBYUV_API
int UYVYToI420(const uint8_t* src_uyvy,
               int src_stride_uyvy,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*UYVYToUVRow)(const uint8_t* src_uyvy, int src_stride_uyvy,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      UYVYToUVRow_C;
  void (*UYVYToYRow)(const uint8_t* src_uyvy, uint8_t* dst_y, int width) =
      UYVYToYRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uyvy = src_uyvy + (height - 1) * src_stride_uyvy;
    src_stride_uyvy = -src_stride_uyvy;
  }
#if defined(HAS_UYVYTOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    UYVYToUVRow = UYVYToUVRow_Any_SSE2;
    UYVYToYRow = UYVYToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      UYVYToUVRow = UYVYToUVRow_SSE2;
      UYVYToYRow = UYVYToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    UYVYToUVRow = UYVYToUVRow_Any_AVX2;
    UYVYToYRow = UYVYToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      UYVYToUVRow = UYVYToUVRow_AVX2;
      UYVYToYRow = UYVYToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    UYVYToYRow = UYVYToYRow_Any_NEON;
    UYVYToUVRow = UYVYToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      UYVYToYRow = UYVYToYRow_NEON;
      UYVYToUVRow = UYVYToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    UYVYToYRow = UYVYToYRow_Any_MSA;
    UYVYToUVRow = UYVYToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      UYVYToYRow = UYVYToYRow_MSA;
      UYVYToUVRow = UYVYToUVRow_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    UYVYToUVRow(src_uyvy, src_stride_uyvy, dst_u, dst_v, width);
    UYVYToYRow(src_uyvy, dst_y, width);
    UYVYToYRow(src_uyvy + src_stride_uyvy, dst_y + dst_stride_y, width);
    src_uyvy += src_stride_uyvy * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    UYVYToUVRow(src_uyvy, 0, dst_u, dst_v, width);
    UYVYToYRow(src_uyvy, dst_y, width);
  }
  return 0;
}

// Convert ARGB to I420.
LIBYUV_API
int ARGBToI420(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  if (!src_argb || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
#if defined(HAS_ARGBTOYROW_SSSE3) && defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2) && defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToYRow = ARGBToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToYRow = ARGBToYRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToUVRow = ARGBToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToYRow = ARGBToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToUVRow = ARGBToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    ARGBToUVRow(src_argb, src_stride_argb, dst_u, dst_v, width);
    ARGBToYRow(src_argb, dst_y, width);
    ARGBToYRow(src_argb + src_stride_argb, dst_y + dst_stride_y, width);
    src_argb += src_stride_argb * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    ARGBToUVRow(src_argb, 0, dst_u, dst_v, width);
    ARGBToYRow(src_argb, dst_y, width);
  }
  return 0;
}

// Convert BGRA to I420.
LIBYUV_API
int BGRAToI420(const uint8_t* src_bgra,
               int src_stride_bgra,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*BGRAToUVRow)(const uint8_t* src_bgra0, int src_stride_bgra,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      BGRAToUVRow_C;
  void (*BGRAToYRow)(const uint8_t* src_bgra, uint8_t* dst_y, int width) =
      BGRAToYRow_C;
  if (!src_bgra || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_bgra = src_bgra + (height - 1) * src_stride_bgra;
    src_stride_bgra = -src_stride_bgra;
  }
#if defined(HAS_BGRATOYROW_SSSE3) && defined(HAS_BGRATOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    BGRAToUVRow = BGRAToUVRow_Any_SSSE3;
    BGRAToYRow = BGRAToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      BGRAToUVRow = BGRAToUVRow_SSSE3;
      BGRAToYRow = BGRAToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_BGRATOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    BGRAToYRow = BGRAToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      BGRAToYRow = BGRAToYRow_NEON;
    }
  }
#endif
#if defined(HAS_BGRATOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    BGRAToUVRow = BGRAToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      BGRAToUVRow = BGRAToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_BGRATOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    BGRAToYRow = BGRAToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      BGRAToYRow = BGRAToYRow_MSA;
    }
  }
#endif
#if defined(HAS_BGRATOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    BGRAToUVRow = BGRAToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      BGRAToUVRow = BGRAToUVRow_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    BGRAToUVRow(src_bgra, src_stride_bgra, dst_u, dst_v, width);
    BGRAToYRow(src_bgra, dst_y, width);
    BGRAToYRow(src_bgra + src_stride_bgra, dst_y + dst_stride_y, width);
    src_bgra += src_stride_bgra * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    BGRAToUVRow(src_bgra, 0, dst_u, dst_v, width);
    BGRAToYRow(src_bgra, dst_y, width);
  }
  return 0;
}

// Convert ABGR to I420.
LIBYUV_API
int ABGRToI420(const uint8_t* src_abgr,
               int src_stride_abgr,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*ABGRToUVRow)(const uint8_t* src_abgr0, int src_stride_abgr,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ABGRToUVRow_C;
  void (*ABGRToYRow)(const uint8_t* src_abgr, uint8_t* dst_y, int width) =
      ABGRToYRow_C;
  if (!src_abgr || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_abgr = src_abgr + (height - 1) * src_stride_abgr;
    src_stride_abgr = -src_stride_abgr;
  }
#if defined(HAS_ABGRTOYROW_SSSE3) && defined(HAS_ABGRTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ABGRToUVRow = ABGRToUVRow_Any_SSSE3;
    ABGRToYRow = ABGRToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ABGRToUVRow = ABGRToUVRow_SSSE3;
      ABGRToYRow = ABGRToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ABGRTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ABGRToYRow = ABGRToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ABGRToYRow = ABGRToYRow_NEON;
    }
  }
#endif
#if defined(HAS_ABGRTOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ABGRToUVRow = ABGRToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ABGRToUVRow = ABGRToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_ABGRTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ABGRToYRow = ABGRToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ABGRToYRow = ABGRToYRow_MSA;
    }
  }
#endif
#if defined(HAS_ABGRTOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ABGRToUVRow = ABGRToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ABGRToUVRow = ABGRToUVRow_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    ABGRToUVRow(src_abgr, src_stride_abgr, dst_u, dst_v, width);
    ABGRToYRow(src_abgr, dst_y, width);
    ABGRToYRow(src_abgr + src_stride_abgr, dst_y + dst_stride_y, width);
    src_abgr += src_stride_abgr * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    ABGRToUVRow(src_abgr, 0, dst_u, dst_v, width);
    ABGRToYRow(src_abgr, dst_y, width);
  }
  return 0;
}

// Convert RGBA to I420.
LIBYUV_API
int RGBAToI420(const uint8_t* src_rgba,
               int src_stride_rgba,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*RGBAToUVRow)(const uint8_t* src_rgba0, int src_stride_rgba,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      RGBAToUVRow_C;
  void (*RGBAToYRow)(const uint8_t* src_rgba, uint8_t* dst_y, int width) =
      RGBAToYRow_C;
  if (!src_rgba || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgba = src_rgba + (height - 1) * src_stride_rgba;
    src_stride_rgba = -src_stride_rgba;
  }
#if defined(HAS_RGBATOYROW_SSSE3) && defined(HAS_RGBATOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RGBAToUVRow = RGBAToUVRow_Any_SSSE3;
    RGBAToYRow = RGBAToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGBAToUVRow = RGBAToUVRow_SSSE3;
      RGBAToYRow = RGBAToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RGBATOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGBAToYRow = RGBAToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RGBAToYRow = RGBAToYRow_NEON;
    }
  }
#endif
#if defined(HAS_RGBATOUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGBAToUVRow = RGBAToUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RGBAToUVRow = RGBAToUVRow_NEON;
    }
  }
#endif
#if defined(HAS_RGBATOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGBAToYRow = RGBAToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGBAToYRow = RGBAToYRow_MSA;
    }
  }
#endif
#if defined(HAS_RGBATOUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGBAToUVRow = RGBAToUVRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGBAToUVRow = RGBAToUVRow_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    RGBAToUVRow(src_rgba, src_stride_rgba, dst_u, dst_v, width);
    RGBAToYRow(src_rgba, dst_y, width);
    RGBAToYRow(src_rgba + src_stride_rgba, dst_y + dst_stride_y, width);
    src_rgba += src_stride_rgba * 2;
    dst_y += dst_stride_y * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    RGBAToUVRow(src_rgba, 0, dst_u, dst_v, width);
    RGBAToYRow(src_rgba, dst_y, width);
  }
  return 0;
}

// Convert RGB24 to I420.
LIBYUV_API
int RGB24ToI420(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_y,
                int dst_stride_y,
                uint8_t* dst_u,
                int dst_stride_u,
                uint8_t* dst_v,
                int dst_stride_v,
                int width,
                int height) {
  int y;
#if (defined(HAS_RGB24TOYROW_NEON) || defined(HAS_RGB24TOYROW_MSA))
  void (*RGB24ToUVRow)(const uint8_t* src_rgb24, int src_stride_rgb24,
                       uint8_t* dst_u, uint8_t* dst_v, int width) =
      RGB24ToUVRow_C;
  void (*RGB24ToYRow)(const uint8_t* src_rgb24, uint8_t* dst_y, int width) =
      RGB24ToYRow_C;
#else
  void (*RGB24ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RGB24ToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_rgb24 || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb24 = src_rgb24 + (height - 1) * src_stride_rgb24;
    src_stride_rgb24 = -src_stride_rgb24;
  }

// Neon version does direct RGB24 to YUV.
#if defined(HAS_RGB24TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB24ToUVRow = RGB24ToUVRow_Any_NEON;
    RGB24ToYRow = RGB24ToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RGB24ToYRow = RGB24ToYRow_NEON;
      if (IS_ALIGNED(width, 16)) {
        RGB24ToUVRow = RGB24ToUVRow_NEON;
      }
    }
  }
#elif defined(HAS_RGB24TOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGB24ToUVRow = RGB24ToUVRow_Any_MSA;
    RGB24ToYRow = RGB24ToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToYRow = RGB24ToYRow_MSA;
      RGB24ToUVRow = RGB24ToUVRow_MSA;
    }
  }
// Other platforms do intermediate conversion from RGB24 to ARGB.
#else
#if defined(HAS_RGB24TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToARGBRow = RGB24ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3) && defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2) && defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#endif

  {
#if !(defined(HAS_RGB24TOYROW_NEON) || defined(HAS_RGB24TOYROW_MSA))
    // Allocate 2 rows of ARGB.
    const int kRowSize = (width * 4 + 31) & ~31;
    align_buffer_64(row, kRowSize * 2);
#endif

    for (y = 0; y < height - 1; y += 2) {
#if (defined(HAS_RGB24TOYROW_NEON) || defined(HAS_RGB24TOYROW_MSA))
      RGB24ToUVRow(src_rgb24, src_stride_rgb24, dst_u, dst_v, width);
      RGB24ToYRow(src_rgb24, dst_y, width);
      RGB24ToYRow(src_rgb24 + src_stride_rgb24, dst_y + dst_stride_y, width);
#else
      RGB24ToARGBRow(src_rgb24, row, width);
      RGB24ToARGBRow(src_rgb24 + src_stride_rgb24, row + kRowSize, width);
      ARGBToUVRow(row, kRowSize, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + kRowSize, dst_y + dst_stride_y, width);
#endif
      src_rgb24 += src_stride_rgb24 * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if (defined(HAS_RGB24TOYROW_NEON) || defined(HAS_RGB24TOYROW_MSA))
      RGB24ToUVRow(src_rgb24, 0, dst_u, dst_v, width);
      RGB24ToYRow(src_rgb24, dst_y, width);
#else
      RGB24ToARGBRow(src_rgb24, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !(defined(HAS_RGB24TOYROW_NEON) || defined(HAS_RGB24TOYROW_MSA))
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}

// Convert RAW to I420.
LIBYUV_API
int RAWToI420(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height) {
  int y;
#if (defined(HAS_RAWTOYROW_NEON) || defined(HAS_RAWTOYROW_MSA))
  void (*RAWToUVRow)(const uint8_t* src_raw, int src_stride_raw, uint8_t* dst_u,
                     uint8_t* dst_v, int width) = RAWToUVRow_C;
  void (*RAWToYRow)(const uint8_t* src_raw, uint8_t* dst_y, int width) =
      RAWToYRow_C;
#else
  void (*RAWToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RAWToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_raw || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }

// Neon version does direct RAW to YUV.
#if defined(HAS_RAWTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RAWToUVRow = RAWToUVRow_Any_NEON;
    RAWToYRow = RAWToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RAWToYRow = RAWToYRow_NEON;
      if (IS_ALIGNED(width, 16)) {
        RAWToUVRow = RAWToUVRow_NEON;
      }
    }
  }
#elif defined(HAS_RAWTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RAWToUVRow = RAWToUVRow_Any_MSA;
    RAWToYRow = RAWToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RAWToYRow = RAWToYRow_MSA;
      RAWToUVRow = RAWToUVRow_MSA;
    }
  }
// Other platforms do intermediate conversion from RAW to ARGB.
#else
#if defined(HAS_RAWTOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RAWToARGBRow = RAWToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3) && defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2) && defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#endif

  {
#if !(defined(HAS_RAWTOYROW_NEON) || defined(HAS_RAWTOYROW_MSA))
    // Allocate 2 rows of ARGB.
    const int kRowSize = (width * 4 + 31) & ~31;
    align_buffer_64(row, kRowSize * 2);
#endif

    for (y = 0; y < height - 1; y += 2) {
#if (defined(HAS_RAWTOYROW_NEON) || defined(HAS_RAWTOYROW_MSA))
      RAWToUVRow(src_raw, src_stride_raw, dst_u, dst_v, width);
      RAWToYRow(src_raw, dst_y, width);
      RAWToYRow(src_raw + src_stride_raw, dst_y + dst_stride_y, width);
#else
      RAWToARGBRow(src_raw, row, width);
      RAWToARGBRow(src_raw + src_stride_raw, row + kRowSize, width);
      ARGBToUVRow(row, kRowSize, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + kRowSize, dst_y + dst_stride_y, width);
#endif
      src_raw += src_stride_raw * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if (defined(HAS_RAWTOYROW_NEON) || defined(HAS_RAWTOYROW_MSA))
      RAWToUVRow(src_raw, 0, dst_u, dst_v, width);
      RAWToYRow(src_raw, dst_y, width);
#else
      RAWToARGBRow(src_raw, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !(defined(HAS_RAWTOYROW_NEON) || defined(HAS_RAWTOYROW_MSA))
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}

// Convert RGB565 to I420.
LIBYUV_API
int RGB565ToI420(const uint8_t* src_rgb565,
                 int src_stride_rgb565,
                 uint8_t* dst_y,
                 int dst_stride_y,
                 uint8_t* dst_u,
                 int dst_stride_u,
                 uint8_t* dst_v,
                 int dst_stride_v,
                 int width,
                 int height) {
  int y;
#if (defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA))
  void (*RGB565ToUVRow)(const uint8_t* src_rgb565, int src_stride_rgb565,
                        uint8_t* dst_u, uint8_t* dst_v, int width) =
      RGB565ToUVRow_C;
  void (*RGB565ToYRow)(const uint8_t* src_rgb565, uint8_t* dst_y, int width) =
      RGB565ToYRow_C;
#else
  void (*RGB565ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb,
                          int width) = RGB565ToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_rgb565 || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb565 = src_rgb565 + (height - 1) * src_stride_rgb565;
    src_stride_rgb565 = -src_stride_rgb565;
  }

// Neon version does direct RGB565 to YUV.
#if defined(HAS_RGB565TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB565ToUVRow = RGB565ToUVRow_Any_NEON;
    RGB565ToYRow = RGB565ToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RGB565ToYRow = RGB565ToYRow_NEON;
      if (IS_ALIGNED(width, 16)) {
        RGB565ToUVRow = RGB565ToUVRow_NEON;
      }
    }
  }
#elif defined(HAS_RGB565TOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGB565ToUVRow = RGB565ToUVRow_Any_MSA;
    RGB565ToYRow = RGB565ToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGB565ToYRow = RGB565ToYRow_MSA;
      RGB565ToUVRow = RGB565ToUVRow_MSA;
    }
  }
// Other platforms do intermediate conversion from RGB565 to ARGB.
#else
#if defined(HAS_RGB565TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      RGB565ToARGBRow = RGB565ToARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_RGB565TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      RGB565ToARGBRow = RGB565ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3) && defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2) && defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#endif
  {
#if !(defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA))
    // Allocate 2 rows of ARGB.
    const int kRowSize = (width * 4 + 31) & ~31;
    align_buffer_64(row, kRowSize * 2);
#endif
    for (y = 0; y < height - 1; y += 2) {
#if (defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA))
      RGB565ToUVRow(src_rgb565, src_stride_rgb565, dst_u, dst_v, width);
      RGB565ToYRow(src_rgb565, dst_y, width);
      RGB565ToYRow(src_rgb565 + src_stride_rgb565, dst_y + dst_stride_y, width);
#else
      RGB565ToARGBRow(src_rgb565, row, width);
      RGB565ToARGBRow(src_rgb565 + src_stride_rgb565, row + kRowSize, width);
      ARGBToUVRow(row, kRowSize, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + kRowSize, dst_y + dst_stride_y, width);
#endif
      src_rgb565 += src_stride_rgb565 * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if (defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA))
      RGB565ToUVRow(src_rgb565, 0, dst_u, dst_v, width);
      RGB565ToYRow(src_rgb565, dst_y, width);
#else
      RGB565ToARGBRow(src_rgb565, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !(defined(HAS_RGB565TOYROW_NEON) || defined(HAS_RGB565TOYROW_MSA))
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}

// Convert ARGB1555 to I420.
LIBYUV_API
int ARGB1555ToI420(const uint8_t* src_argb1555,
                   int src_stride_argb1555,
                   uint8_t* dst_y,
                   int dst_stride_y,
                   uint8_t* dst_u,
                   int dst_stride_u,
                   uint8_t* dst_v,
                   int dst_stride_v,
                   int width,
                   int height) {
  int y;
#if (defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA))
  void (*ARGB1555ToUVRow)(const uint8_t* src_argb1555, int src_stride_argb1555,
                          uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGB1555ToUVRow_C;
  void (*ARGB1555ToYRow)(const uint8_t* src_argb1555, uint8_t* dst_y,
                         int width) = ARGB1555ToYRow_C;
#else
  void (*ARGB1555ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb,
                            int width) = ARGB1555ToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_argb1555 || !dst_y || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb1555 = src_argb1555 + (height - 1) * src_stride_argb1555;
    src_stride_argb1555 = -src_stride_argb1555;
  }

// Neon version does direct ARGB1555 to YUV.
#if defined(HAS_ARGB1555TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGB1555ToUVRow = ARGB1555ToUVRow_Any_NEON;
    ARGB1555ToYRow = ARGB1555ToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGB1555ToYRow = ARGB1555ToYRow_NEON;
      if (IS_ALIGNED(width, 16)) {
        ARGB1555ToUVRow = ARGB1555ToUVRow_NEON;
      }
    }
  }
#elif defined(HAS_ARGB1555TOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGB1555ToUVRow = ARGB1555ToUVRow_Any_MSA;
    ARGB1555ToYRow = ARGB1555ToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGB1555ToYRow = ARGB1555ToYRow_MSA;
      ARGB1555ToUVRow = ARGB1555ToUVRow_MSA;
    }
  }
// Other platforms do intermediate conversion from ARGB1555 to ARGB.
#else
#if defined(HAS_ARGB1555TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGB1555TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3) && defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2) && defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#endif
  {
#if !(defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA))
    // Allocate 2 rows of ARGB.
    const int kRowSize = (width * 4 + 31) & ~31;
    align_buffer_64(row, kRowSize * 2);
#endif

    for (y = 0; y < height - 1; y += 2) {
#if (defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA))
      ARGB1555ToUVRow(src_argb1555, src_stride_argb1555, dst_u, dst_v, width);
      ARGB1555ToYRow(src_argb1555, dst_y, width);
      ARGB1555ToYRow(src_argb1555 + src_stride_argb1555, dst_y + dst_stride_y,
                     width);
#else
      ARGB1555ToARGBRow(src_argb1555, row, width);
      ARGB1555ToARGBRow(src_argb1555 + src_stride_argb1555, row + kRowSize,
                        width);
      ARGBToUVRow(row, kRowSize, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + kRowSize, dst_y + dst_stride_y, width);
#endif
      src_argb1555 += src_stride_argb1555 * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if (defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA))
      ARGB1555ToUVRow(src_argb1555, 0, dst_u, dst_v, width);
      ARGB1555ToYRow(src_argb1555, dst_y, width);
#else
      ARGB1555ToARGBRow(src_argb1555, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !(defined(HAS_ARGB1555TOYROW_NEON) || defined(HAS_ARGB1555TOYROW_MSA))
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}

// Convert ARGB4444 to I420.
LIBYUV_API
int ARGB4444ToI420(const uint8_t* src_argb4444,
                   int src_stride_argb4444,
                   uint8_t* dst_y,
                   int dst_stride_y,
                   uint8_t* dst_u,
                   int dst_stride_u,
                   uint8_t* dst_v,
                   int dst_stride_v,
                   int width,
                   int height) {
  int y;
#if defined(HAS_ARGB4444TOYROW_NEON)
  void (*ARGB4444ToUVRow)(const uint8_t* src_argb4444, int src_stride_argb4444,
                          uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGB4444ToUVRow_C;
  void (*ARGB4444ToYRow)(const uint8_t* src_argb4444, uint8_t* dst_y,
                         int width) = ARGB4444ToYRow_C;
#else
  void (*ARGB4444ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb,
                            int width) = ARGB4444ToARGBRow_C;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
#endif
  if (!src_argb4444 || !dst_y || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb4444 = src_argb4444 + (height - 1) * src_stride_argb4444;
    src_stride_argb4444 = -src_stride_argb4444;
  }

// Neon version does direct ARGB4444 to YUV.
#if defined(HAS_ARGB4444TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGB4444ToUVRow = ARGB4444ToUVRow_Any_NEON;
    ARGB4444ToYRow = ARGB4444ToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGB4444ToYRow = ARGB4444ToYRow_NEON;
      if (IS_ALIGNED(width, 16)) {
        ARGB4444ToUVRow = ARGB4444ToUVRow_NEON;
      }
    }
  }
// Other platforms do intermediate conversion from ARGB4444 to ARGB.
#else
#if defined(HAS_ARGB4444TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGB4444TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGB4444TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3) && defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVRow = ARGBToUVRow_SSSE3;
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2) && defined(HAS_ARGBTOUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToUVRow = ARGBToUVRow_Any_AVX2;
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVRow = ARGBToUVRow_AVX2;
      ARGBToYRow = ARGBToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToUVRow = ARGBToUVRow_Any_MSA;
    ARGBToYRow = ARGBToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_MSA;
      if (IS_ALIGNED(width, 32)) {
        ARGBToUVRow = ARGBToUVRow_MSA;
      }
    }
  }
#endif
#endif

  {
#if !defined(HAS_ARGB4444TOYROW_NEON)
    // Allocate 2 rows of ARGB.
    const int kRowSize = (width * 4 + 31) & ~31;
    align_buffer_64(row, kRowSize * 2);
#endif

    for (y = 0; y < height - 1; y += 2) {
#if defined(HAS_ARGB4444TOYROW_NEON)
      ARGB4444ToUVRow(src_argb4444, src_stride_argb4444, dst_u, dst_v, width);
      ARGB4444ToYRow(src_argb4444, dst_y, width);
      ARGB4444ToYRow(src_argb4444 + src_stride_argb4444, dst_y + dst_stride_y,
                     width);
#else
      ARGB4444ToARGBRow(src_argb4444, row, width);
      ARGB4444ToARGBRow(src_argb4444 + src_stride_argb4444, row + kRowSize,
                        width);
      ARGBToUVRow(row, kRowSize, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + kRowSize, dst_y + dst_stride_y, width);
#endif
      src_argb4444 += src_stride_argb4444 * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
#if defined(HAS_ARGB4444TOYROW_NEON)
      ARGB4444ToUVRow(src_argb4444, 0, dst_u, dst_v, width);
      ARGB4444ToYRow(src_argb4444, dst_y, width);
#else
      ARGB4444ToARGBRow(src_argb4444, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
#endif
    }
#if !defined(HAS_ARGB4444TOYROW_NEON)
    free_aligned_buffer_64(row);
#endif
  }
  return 0;
}

static void SplitPixels(const uint8_t* src_u,
                        int src_pixel_stride_uv,
                        uint8_t* dst_u,
                        int width) {
  int i;
  for (i = 0; i < width; ++i) {
    *dst_u = *src_u;
    ++dst_u;
    src_u += src_pixel_stride_uv;
  }
}

// Convert Android420 to I420.
LIBYUV_API
int Android420ToI420(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     int src_pixel_stride_uv,
                     uint8_t* dst_y,
                     int dst_stride_y,
                     uint8_t* dst_u,
                     int dst_stride_u,
                     uint8_t* dst_v,
                     int dst_stride_v,
                     int width,
                     int height) {
  int y;
  const ptrdiff_t vu_off = src_v - src_u;
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

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }

  // Copy UV planes as is - I420
  if (src_pixel_stride_uv == 1) {
    CopyPlane(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, halfheight);
    CopyPlane(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, halfheight);
    return 0;
    // Split UV planes - NV21
  }
  if (src_pixel_stride_uv == 2 && vu_off == -1 &&
      src_stride_u == src_stride_v) {
    SplitUVPlane(src_v, src_stride_v, dst_v, dst_stride_v, dst_u, dst_stride_u,
                 halfwidth, halfheight);
    return 0;
    // Split UV planes - NV12
  }
  if (src_pixel_stride_uv == 2 && vu_off == 1 && src_stride_u == src_stride_v) {
    SplitUVPlane(src_u, src_stride_u, dst_u, dst_stride_u, dst_v, dst_stride_v,
                 halfwidth, halfheight);
    return 0;
  }

  for (y = 0; y < halfheight; ++y) {
    SplitPixels(src_u, src_pixel_stride_uv, dst_u, halfwidth);
    SplitPixels(src_v, src_pixel_stride_uv, dst_v, halfwidth);
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  return 0;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
