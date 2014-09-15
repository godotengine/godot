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
#include "libyuv/format_conversion.h"
#include "libyuv/planar_functions.h"
#include "libyuv/rotate.h"
#include "libyuv/scale.h"  // For ScalePlane()
#include "libyuv/video_common.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define SUBSAMPLE(v, a, s) (v < 0) ? (-((-v + a) >> s)) : ((v + a) >> s)
static __inline int Abs(int v) {
  return v >= 0 ? v : -v;
}

// I420 To any I4xx YUV format with mirroring.
static int I420ToI4xx(const uint8* src_y, int src_stride_y,
                      const uint8* src_u, int src_stride_u,
                      const uint8* src_v, int src_stride_v,
                      uint8* dst_y, int dst_stride_y,
                      uint8* dst_u, int dst_stride_u,
                      uint8* dst_v, int dst_stride_v,
                      int src_y_width, int src_y_height,
                      int dst_uv_width, int dst_uv_height) {
  if (src_y_width == 0 || src_y_height == 0 ||
      dst_uv_width <= 0 || dst_uv_height <= 0) {
    return -1;
  }
  const int dst_y_width = Abs(src_y_width);
  const int dst_y_height = Abs(src_y_height);
  const int src_uv_width = SUBSAMPLE(src_y_width, 1, 1);
  const int src_uv_height = SUBSAMPLE(src_y_height, 1, 1);
  ScalePlane(src_y, src_stride_y, src_y_width, src_y_height,
             dst_y, dst_stride_y, dst_y_width, dst_y_height,
             kFilterBilinear);
  ScalePlane(src_u, src_stride_u, src_uv_width, src_uv_height,
             dst_u, dst_stride_u, dst_uv_width, dst_uv_height,
             kFilterBilinear);
  ScalePlane(src_v, src_stride_v, src_uv_width, src_uv_height,
             dst_v, dst_stride_v, dst_uv_width, dst_uv_height,
             kFilterBilinear);
  return 0;
}

// 420 chroma is 1/2 width, 1/2 height
// 422 chroma is 1/2 width, 1x height
LIBYUV_API
int I420ToI422(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_u, int dst_stride_u,
               uint8* dst_v, int dst_stride_v,
               int width, int height) {
  const int dst_uv_width = (Abs(width) + 1) >> 1;
  const int dst_uv_height = Abs(height);
  return I420ToI4xx(src_y, src_stride_y,
                    src_u, src_stride_u,
                    src_v, src_stride_v,
                    dst_y, dst_stride_y,
                    dst_u, dst_stride_u,
                    dst_v, dst_stride_v,
                    width, height,
                    dst_uv_width, dst_uv_height);
}

// 420 chroma is 1/2 width, 1/2 height
// 444 chroma is 1x width, 1x height
LIBYUV_API
int I420ToI444(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_u, int dst_stride_u,
               uint8* dst_v, int dst_stride_v,
               int width, int height) {
  const int dst_uv_width = Abs(width);
  const int dst_uv_height = Abs(height);
  return I420ToI4xx(src_y, src_stride_y,
                    src_u, src_stride_u,
                    src_v, src_stride_v,
                    dst_y, dst_stride_y,
                    dst_u, dst_stride_u,
                    dst_v, dst_stride_v,
                    width, height,
                    dst_uv_width, dst_uv_height);
}

// 420 chroma is 1/2 width, 1/2 height
// 411 chroma is 1/4 width, 1x height
LIBYUV_API
int I420ToI411(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_u, int dst_stride_u,
               uint8* dst_v, int dst_stride_v,
               int width, int height) {
  const int dst_uv_width = (Abs(width) + 3) >> 2;
  const int dst_uv_height = Abs(height);
  return I420ToI4xx(src_y, src_stride_y,
                    src_u, src_stride_u,
                    src_v, src_stride_v,
                    dst_y, dst_stride_y,
                    dst_u, dst_stride_u,
                    dst_v, dst_stride_v,
                    width, height,
                    dst_uv_width, dst_uv_height);
}

// Copy to I400. Source can be I420,422,444,400,NV12,NV21
LIBYUV_API
int I400Copy(const uint8* src_y, int src_stride_y,
             uint8* dst_y, int dst_stride_y,
             int width, int height) {
  if (!src_y || !dst_y ||
      width <= 0 || height == 0) {
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
int I422ToYUY2(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_yuy2, int dst_stride_yuy2,
               int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_yuy2 ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_yuy2 = dst_yuy2 + (height - 1) * dst_stride_yuy2;
    dst_stride_yuy2 = -dst_stride_yuy2;
  }
  // Coalesce rows.
  if (src_stride_y == width &&
      src_stride_u * 2 == width &&
      src_stride_v * 2 == width &&
      dst_stride_yuy2 == width * 2) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_yuy2 = 0;
  }
  void (*I422ToYUY2Row)(const uint8* src_y, const uint8* src_u,
                        const uint8* src_v, uint8* dst_yuy2, int width) =
      I422ToYUY2Row_C;
#if defined(HAS_I422TOYUY2ROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && width >= 16) {
    I422ToYUY2Row = I422ToYUY2Row_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      I422ToYUY2Row = I422ToYUY2Row_SSE2;
    }
  }
#elif defined(HAS_I422TOYUY2ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 16) {
    I422ToYUY2Row = I422ToYUY2Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      I422ToYUY2Row = I422ToYUY2Row_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToYUY2Row(src_y, src_u, src_v, dst_yuy2, width);
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_yuy2 += dst_stride_yuy2;
  }
  return 0;
}

LIBYUV_API
int I420ToYUY2(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_yuy2, int dst_stride_yuy2,
               int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_yuy2 ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_yuy2 = dst_yuy2 + (height - 1) * dst_stride_yuy2;
    dst_stride_yuy2 = -dst_stride_yuy2;
  }
  void (*I422ToYUY2Row)(const uint8* src_y, const uint8* src_u,
                        const uint8* src_v, uint8* dst_yuy2, int width) =
      I422ToYUY2Row_C;
#if defined(HAS_I422TOYUY2ROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && width >= 16) {
    I422ToYUY2Row = I422ToYUY2Row_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      I422ToYUY2Row = I422ToYUY2Row_SSE2;
    }
  }
#elif defined(HAS_I422TOYUY2ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 16) {
    I422ToYUY2Row = I422ToYUY2Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      I422ToYUY2Row = I422ToYUY2Row_NEON;
    }
  }
#endif

  for (int y = 0; y < height - 1; y += 2) {
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
int I422ToUYVY(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_uyvy, int dst_stride_uyvy,
               int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_uyvy ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_uyvy = dst_uyvy + (height - 1) * dst_stride_uyvy;
    dst_stride_uyvy = -dst_stride_uyvy;
  }
  // Coalesce rows.
  if (src_stride_y == width &&
      src_stride_u * 2 == width &&
      src_stride_v * 2 == width &&
      dst_stride_uyvy == width * 2) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_uyvy = 0;
  }
  void (*I422ToUYVYRow)(const uint8* src_y, const uint8* src_u,
                        const uint8* src_v, uint8* dst_uyvy, int width) =
      I422ToUYVYRow_C;
#if defined(HAS_I422TOUYVYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && width >= 16) {
    I422ToUYVYRow = I422ToUYVYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      I422ToUYVYRow = I422ToUYVYRow_SSE2;
    }
  }
#elif defined(HAS_I422TOUYVYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 16) {
    I422ToUYVYRow = I422ToUYVYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      I422ToUYVYRow = I422ToUYVYRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToUYVYRow(src_y, src_u, src_v, dst_uyvy, width);
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_uyvy += dst_stride_uyvy;
  }
  return 0;
}

LIBYUV_API
int I420ToUYVY(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_uyvy, int dst_stride_uyvy,
               int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_uyvy ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_uyvy = dst_uyvy + (height - 1) * dst_stride_uyvy;
    dst_stride_uyvy = -dst_stride_uyvy;
  }
  void (*I422ToUYVYRow)(const uint8* src_y, const uint8* src_u,
                        const uint8* src_v, uint8* dst_uyvy, int width) =
      I422ToUYVYRow_C;
#if defined(HAS_I422TOUYVYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && width >= 16) {
    I422ToUYVYRow = I422ToUYVYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      I422ToUYVYRow = I422ToUYVYRow_SSE2;
    }
  }
#elif defined(HAS_I422TOUYVYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 16) {
    I422ToUYVYRow = I422ToUYVYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      I422ToUYVYRow = I422ToUYVYRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height - 1; y += 2) {
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

LIBYUV_API
int I420ToNV12(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_uv, int dst_stride_uv,
               int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_y || !dst_uv ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    int halfheight = (height + 1) >> 1;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_uv = dst_uv + (halfheight - 1) * dst_stride_uv;
    dst_stride_y = -dst_stride_y;
    dst_stride_uv = -dst_stride_uv;
  }
  // Coalesce rows.
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if (src_stride_y == width &&
      dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_y = 0;
  }
  // Coalesce rows.
  if (src_stride_u == halfwidth &&
      src_stride_v == halfwidth &&
      dst_stride_uv == halfwidth * 2) {
    halfwidth *= halfheight;
    halfheight = 1;
    src_stride_u = src_stride_v = dst_stride_uv = 0;
  }
  void (*MergeUVRow_)(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                      int width) = MergeUVRow_C;
#if defined(HAS_MERGEUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && halfwidth >= 16) {
    MergeUVRow_ = MergeUVRow_Any_SSE2;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow_ = MergeUVRow_Unaligned_SSE2;
      if (IS_ALIGNED(src_u, 16) && IS_ALIGNED(src_stride_u, 16) &&
          IS_ALIGNED(src_v, 16) && IS_ALIGNED(src_stride_v, 16) &&
          IS_ALIGNED(dst_uv, 16) && IS_ALIGNED(dst_stride_uv, 16)) {
        MergeUVRow_ = MergeUVRow_SSE2;
      }
    }
  }
#endif
#if defined(HAS_MERGEUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2) && halfwidth >= 32) {
    MergeUVRow_ = MergeUVRow_Any_AVX2;
    if (IS_ALIGNED(halfwidth, 32)) {
      MergeUVRow_ = MergeUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && halfwidth >= 16) {
    MergeUVRow_ = MergeUVRow_Any_NEON;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow_ = MergeUVRow_NEON;
    }
  }
#endif

  CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  for (int y = 0; y < halfheight; ++y) {
    // Merge a row of U and V into a row of UV.
    MergeUVRow_(src_u, src_v, dst_uv, halfwidth);
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_uv += dst_stride_uv;
  }
  return 0;
}

LIBYUV_API
int I420ToNV21(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_vu, int dst_stride_vu,
               int width, int height) {
  return I420ToNV12(src_y, src_stride_y,
                    src_v, src_stride_v,
                    src_u, src_stride_u,
                    dst_y, src_stride_y,
                    dst_vu, dst_stride_vu,
                    width, height);
}

// Convert I420 to ARGB.
LIBYUV_API
int I420ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  void (*I422ToARGBRow)(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* rgb_buf,
                        int width) = I422ToARGBRow_C;
#if defined(HAS_I422TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToARGBRow = I422ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGBRow = I422ToARGBRow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
        I422ToARGBRow = I422ToARGBRow_SSSE3;
      }
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2) && width >= 16) {
    I422ToARGBRow = I422ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGBRow = I422ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I422ToARGBRow = I422ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGBRow = I422ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_MIPS_DSPR2)
  if (TestCpuFlag(kCpuHasMIPS_DSPR2) && IS_ALIGNED(width, 4) &&
      IS_ALIGNED(src_y, 4) && IS_ALIGNED(src_stride_y, 4) &&
      IS_ALIGNED(src_u, 2) && IS_ALIGNED(src_stride_u, 2) &&
      IS_ALIGNED(src_v, 2) && IS_ALIGNED(src_stride_v, 2) &&
      IS_ALIGNED(dst_argb, 4) && IS_ALIGNED(dst_stride_argb, 4)) {
    I422ToARGBRow = I422ToARGBRow_MIPS_DSPR2;
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToARGBRow(src_y, src_u, src_v, dst_argb, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to BGRA.
LIBYUV_API
int I420ToBGRA(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_bgra, int dst_stride_bgra,
               int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_bgra ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_bgra = dst_bgra + (height - 1) * dst_stride_bgra;
    dst_stride_bgra = -dst_stride_bgra;
  }
  void (*I422ToBGRARow)(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* rgb_buf,
                        int width) = I422ToBGRARow_C;
#if defined(HAS_I422TOBGRAROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToBGRARow = I422ToBGRARow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToBGRARow = I422ToBGRARow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_bgra, 16) && IS_ALIGNED(dst_stride_bgra, 16)) {
        I422ToBGRARow = I422ToBGRARow_SSSE3;
      }
    }
  }
#elif defined(HAS_I422TOBGRAROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I422ToBGRARow = I422ToBGRARow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToBGRARow = I422ToBGRARow_NEON;
    }
  }
#elif defined(HAS_I422TOBGRAROW_MIPS_DSPR2)
  if (TestCpuFlag(kCpuHasMIPS_DSPR2) && IS_ALIGNED(width, 4) &&
      IS_ALIGNED(src_y, 4) && IS_ALIGNED(src_stride_y, 4) &&
      IS_ALIGNED(src_u, 2) && IS_ALIGNED(src_stride_u, 2) &&
      IS_ALIGNED(src_v, 2) && IS_ALIGNED(src_stride_v, 2) &&
      IS_ALIGNED(dst_bgra, 4) && IS_ALIGNED(dst_stride_bgra, 4)) {
    I422ToBGRARow = I422ToBGRARow_MIPS_DSPR2;
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToBGRARow(src_y, src_u, src_v, dst_bgra, width);
    dst_bgra += dst_stride_bgra;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to ABGR.
LIBYUV_API
int I420ToABGR(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_abgr, int dst_stride_abgr,
               int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_abgr ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_abgr = dst_abgr + (height - 1) * dst_stride_abgr;
    dst_stride_abgr = -dst_stride_abgr;
  }
  void (*I422ToABGRRow)(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* rgb_buf,
                        int width) = I422ToABGRRow_C;
#if defined(HAS_I422TOABGRROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToABGRRow = I422ToABGRRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToABGRRow = I422ToABGRRow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_abgr, 16) && IS_ALIGNED(dst_stride_abgr, 16)) {
        I422ToABGRRow = I422ToABGRRow_SSSE3;
      }
    }
  }
#elif defined(HAS_I422TOABGRROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I422ToABGRRow = I422ToABGRRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToABGRRow = I422ToABGRRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToABGRRow(src_y, src_u, src_v, dst_abgr, width);
    dst_abgr += dst_stride_abgr;
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
int I420ToRGBA(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_rgba, int dst_stride_rgba,
               int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_rgba ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgba = dst_rgba + (height - 1) * dst_stride_rgba;
    dst_stride_rgba = -dst_stride_rgba;
  }
  void (*I422ToRGBARow)(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* rgb_buf,
                        int width) = I422ToRGBARow_C;
#if defined(HAS_I422TORGBAROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToRGBARow = I422ToRGBARow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGBARow = I422ToRGBARow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_rgba, 16) && IS_ALIGNED(dst_stride_rgba, 16)) {
        I422ToRGBARow = I422ToRGBARow_SSSE3;
      }
    }
  }
#elif defined(HAS_I422TORGBAROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I422ToRGBARow = I422ToRGBARow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGBARow = I422ToRGBARow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToRGBARow(src_y, src_u, src_v, dst_rgba, width);
    dst_rgba += dst_stride_rgba;
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
int I420ToRGB24(const uint8* src_y, int src_stride_y,
                const uint8* src_u, int src_stride_u,
                const uint8* src_v, int src_stride_v,
                uint8* dst_rgb24, int dst_stride_rgb24,
                int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_rgb24 ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb24 = dst_rgb24 + (height - 1) * dst_stride_rgb24;
    dst_stride_rgb24 = -dst_stride_rgb24;
  }
  void (*I422ToRGB24Row)(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* rgb_buf,
                        int width) = I422ToRGB24Row_C;
#if defined(HAS_I422TORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToRGB24Row = I422ToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB24Row = I422ToRGB24Row_SSSE3;
    }
  }
#elif defined(HAS_I422TORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I422ToRGB24Row = I422ToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB24Row = I422ToRGB24Row_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToRGB24Row(src_y, src_u, src_v, dst_rgb24, width);
    dst_rgb24 += dst_stride_rgb24;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to RAW.
LIBYUV_API
int I420ToRAW(const uint8* src_y, int src_stride_y,
                const uint8* src_u, int src_stride_u,
                const uint8* src_v, int src_stride_v,
                uint8* dst_raw, int dst_stride_raw,
                int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_raw ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_raw = dst_raw + (height - 1) * dst_stride_raw;
    dst_stride_raw = -dst_stride_raw;
  }
  void (*I422ToRAWRow)(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* rgb_buf,
                        int width) = I422ToRAWRow_C;
#if defined(HAS_I422TORAWROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToRAWRow = I422ToRAWRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRAWRow = I422ToRAWRow_SSSE3;
    }
  }
#elif defined(HAS_I422TORAWROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I422ToRAWRow = I422ToRAWRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRAWRow = I422ToRAWRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToRAWRow(src_y, src_u, src_v, dst_raw, width);
    dst_raw += dst_stride_raw;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to ARGB1555.
LIBYUV_API
int I420ToARGB1555(const uint8* src_y, int src_stride_y,
                   const uint8* src_u, int src_stride_u,
                   const uint8* src_v, int src_stride_v,
                   uint8* dst_argb1555, int dst_stride_argb1555,
                   int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_argb1555 ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb1555 = dst_argb1555 + (height - 1) * dst_stride_argb1555;
    dst_stride_argb1555 = -dst_stride_argb1555;
  }
  void (*I422ToARGB1555Row)(const uint8* y_buf,
                          const uint8* u_buf,
                          const uint8* v_buf,
                          uint8* rgb_buf,
                          int width) = I422ToARGB1555Row_C;
#if defined(HAS_I422TOARGB1555ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB1555Row = I422ToARGB1555Row_SSSE3;
    }
  }
#elif defined(HAS_I422TOARGB1555ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB1555Row = I422ToARGB1555Row_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToARGB1555Row(src_y, src_u, src_v, dst_argb1555, width);
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
int I420ToARGB4444(const uint8* src_y, int src_stride_y,
                   const uint8* src_u, int src_stride_u,
                   const uint8* src_v, int src_stride_v,
                   uint8* dst_argb4444, int dst_stride_argb4444,
                   int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_argb4444 ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb4444 = dst_argb4444 + (height - 1) * dst_stride_argb4444;
    dst_stride_argb4444 = -dst_stride_argb4444;
  }
  void (*I422ToARGB4444Row)(const uint8* y_buf,
                          const uint8* u_buf,
                          const uint8* v_buf,
                          uint8* rgb_buf,
                          int width) = I422ToARGB4444Row_C;
#if defined(HAS_I422TOARGB4444ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB4444Row = I422ToARGB4444Row_SSSE3;
    }
  }
#elif defined(HAS_I422TOARGB4444ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB4444Row = I422ToARGB4444Row_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToARGB4444Row(src_y, src_u, src_v, dst_argb4444, width);
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
int I420ToRGB565(const uint8* src_y, int src_stride_y,
                 const uint8* src_u, int src_stride_u,
                 const uint8* src_v, int src_stride_v,
                 uint8* dst_rgb565, int dst_stride_rgb565,
                 int width, int height) {
  if (!src_y || !src_u || !src_v || !dst_rgb565 ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb565 = dst_rgb565 + (height - 1) * dst_stride_rgb565;
    dst_stride_rgb565 = -dst_stride_rgb565;
  }
  void (*I422ToRGB565Row)(const uint8* y_buf,
                          const uint8* u_buf,
                          const uint8* v_buf,
                          uint8* rgb_buf,
                          int width) = I422ToRGB565Row_C;
#if defined(HAS_I422TORGB565ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToRGB565Row = I422ToRGB565Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_SSSE3;
    }
  }
#elif defined(HAS_I422TORGB565ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I422ToRGB565Row = I422ToRGB565Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I422ToRGB565Row(src_y, src_u, src_v, dst_rgb565, width);
    dst_rgb565 += dst_stride_rgb565;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to specified format
LIBYUV_API
int ConvertFromI420(const uint8* y, int y_stride,
                    const uint8* u, int u_stride,
                    const uint8* v, int v_stride,
                    uint8* dst_sample, int dst_sample_stride,
                    int width, int height,
                    uint32 fourcc) {
  uint32 format = CanonicalFourCC(fourcc);
  if (!y || !u|| !v || !dst_sample ||
      width <= 0 || height == 0) {
    return -1;
  }
  int r = 0;
  switch (format) {
    // Single plane formats
    case FOURCC_YUY2:
      r = I420ToYUY2(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 2,
                     width, height);
      break;
    case FOURCC_UYVY:
      r = I420ToUYVY(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 2,
                     width, height);
      break;
    case FOURCC_RGBP:
      r = I420ToRGB565(y, y_stride,
                       u, u_stride,
                       v, v_stride,
                       dst_sample,
                       dst_sample_stride ? dst_sample_stride : width * 2,
                       width, height);
      break;
    case FOURCC_RGBO:
      r = I420ToARGB1555(y, y_stride,
                         u, u_stride,
                         v, v_stride,
                         dst_sample,
                         dst_sample_stride ? dst_sample_stride : width * 2,
                         width, height);
      break;
    case FOURCC_R444:
      r = I420ToARGB4444(y, y_stride,
                         u, u_stride,
                         v, v_stride,
                         dst_sample,
                         dst_sample_stride ? dst_sample_stride : width * 2,
                         width, height);
      break;
    case FOURCC_24BG:
      r = I420ToRGB24(y, y_stride,
                      u, u_stride,
                      v, v_stride,
                      dst_sample,
                      dst_sample_stride ? dst_sample_stride : width * 3,
                      width, height);
      break;
    case FOURCC_RAW:
      r = I420ToRAW(y, y_stride,
                    u, u_stride,
                    v, v_stride,
                    dst_sample,
                    dst_sample_stride ? dst_sample_stride : width * 3,
                    width, height);
      break;
    case FOURCC_ARGB:
      r = I420ToARGB(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 4,
                     width, height);
      break;
    case FOURCC_BGRA:
      r = I420ToBGRA(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 4,
                     width, height);
      break;
    case FOURCC_ABGR:
      r = I420ToABGR(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 4,
                     width, height);
      break;
    case FOURCC_RGBA:
      r = I420ToRGBA(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample,
                     dst_sample_stride ? dst_sample_stride : width * 4,
                     width, height);
      break;
    case FOURCC_BGGR:
      r = I420ToBayerBGGR(y, y_stride,
                          u, u_stride,
                          v, v_stride,
                          dst_sample,
                          dst_sample_stride ? dst_sample_stride : width,
                          width, height);
      break;
    case FOURCC_GBRG:
      r = I420ToBayerGBRG(y, y_stride,
                          u, u_stride,
                          v, v_stride,
                          dst_sample,
                          dst_sample_stride ? dst_sample_stride : width,
                          width, height);
      break;
    case FOURCC_GRBG:
      r = I420ToBayerGRBG(y, y_stride,
                          u, u_stride,
                          v, v_stride,
                          dst_sample,
                          dst_sample_stride ? dst_sample_stride : width,
                          width, height);
      break;
    case FOURCC_RGGB:
      r = I420ToBayerRGGB(y, y_stride,
                          u, u_stride,
                          v, v_stride,
                          dst_sample,
                          dst_sample_stride ? dst_sample_stride : width,
                          width, height);
      break;
    case FOURCC_I400:
      r = I400Copy(y, y_stride,
                   dst_sample,
                   dst_sample_stride ? dst_sample_stride : width,
                   width, height);
      break;
    case FOURCC_NV12: {
      uint8* dst_uv = dst_sample + width * height;
      r = I420ToNV12(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample,
                     dst_sample_stride ? dst_sample_stride : width,
                     dst_uv,
                     dst_sample_stride ? dst_sample_stride : width,
                     width, height);
      break;
    }
    case FOURCC_NV21: {
      uint8* dst_vu = dst_sample + width * height;
      r = I420ToNV21(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample,
                     dst_sample_stride ? dst_sample_stride : width,
                     dst_vu,
                     dst_sample_stride ? dst_sample_stride : width,
                     width, height);
      break;
    }
    // TODO(fbarchard): Add M420 and Q420.
    // Triplanar formats
    // TODO(fbarchard): halfstride instead of halfwidth
    case FOURCC_I420:
    case FOURCC_YU12:
    case FOURCC_YV12: {
      int halfwidth = (width + 1) / 2;
      int halfheight = (height + 1) / 2;
      uint8* dst_u;
      uint8* dst_v;
      if (format == FOURCC_YV12) {
        dst_v = dst_sample + width * height;
        dst_u = dst_v + halfwidth * halfheight;
      } else {
        dst_u = dst_sample + width * height;
        dst_v = dst_u + halfwidth * halfheight;
      }
      r = I420Copy(y, y_stride,
                   u, u_stride,
                   v, v_stride,
                   dst_sample, width,
                   dst_u, halfwidth,
                   dst_v, halfwidth,
                   width, height);
      break;
    }
    case FOURCC_I422:
    case FOURCC_YV16: {
      int halfwidth = (width + 1) / 2;
      uint8* dst_u;
      uint8* dst_v;
      if (format == FOURCC_YV16) {
        dst_v = dst_sample + width * height;
        dst_u = dst_v + halfwidth * height;
      } else {
        dst_u = dst_sample + width * height;
        dst_v = dst_u + halfwidth * height;
      }
      r = I420ToI422(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample, width,
                     dst_u, halfwidth,
                     dst_v, halfwidth,
                     width, height);
      break;
    }
    case FOURCC_I444:
    case FOURCC_YV24: {
      uint8* dst_u;
      uint8* dst_v;
      if (format == FOURCC_YV24) {
        dst_v = dst_sample + width * height;
        dst_u = dst_v + width * height;
      } else {
        dst_u = dst_sample + width * height;
        dst_v = dst_u + width * height;
      }
      r = I420ToI444(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample, width,
                     dst_u, width,
                     dst_v, width,
                     width, height);
      break;
    }
    case FOURCC_I411: {
      int quarterwidth = (width + 3) / 4;
      uint8* dst_u = dst_sample + width * height;
      uint8* dst_v = dst_u + quarterwidth * height;
      r = I420ToI411(y, y_stride,
                     u, u_stride,
                     v, v_stride,
                     dst_sample, width,
                     dst_u, quarterwidth,
                     dst_v, quarterwidth,
                     width, height);
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
