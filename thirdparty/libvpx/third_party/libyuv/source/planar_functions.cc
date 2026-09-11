/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/planar_functions.h"

#include <string.h>  // for memset()

#include "libyuv/cpu_id.h"
#ifdef HAVE_JPEG
#include "libyuv/mjpeg_decoder.h"
#endif
#include "libyuv/row.h"
#include "libyuv/scale_row.h"  // for ScaleRowDown2

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Copy a plane of data
LIBYUV_API
void CopyPlane(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height) {
  int y;
  void (*CopyRow)(const uint8_t* src, uint8_t* dst, int width) = CopyRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_stride_y = -dst_stride_y;
  }
  // Coalesce rows.
  if (src_stride_y == width && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_y = 0;
  }
  // Nothing to do.
  if (src_y == dst_y && src_stride_y == dst_stride_y) {
    return;
  }

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
  for (y = 0; y < height; ++y) {
    CopyRow(src_y, dst_y, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
}

// TODO(fbarchard): Consider support for negative height.
// TODO(fbarchard): Consider stride measured in bytes.
LIBYUV_API
void CopyPlane_16(const uint16_t* src_y,
                  int src_stride_y,
                  uint16_t* dst_y,
                  int dst_stride_y,
                  int width,
                  int height) {
  int y;
  void (*CopyRow)(const uint16_t* src, uint16_t* dst, int width) = CopyRow_16_C;
  // Coalesce rows.
  if (src_stride_y == width && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_y = 0;
  }
#if defined(HAS_COPYROW_16_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(width, 32)) {
    CopyRow = CopyRow_16_SSE2;
  }
#endif
#if defined(HAS_COPYROW_16_ERMS)
  if (TestCpuFlag(kCpuHasERMS)) {
    CopyRow = CopyRow_16_ERMS;
  }
#endif
#if defined(HAS_COPYROW_16_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 32)) {
    CopyRow = CopyRow_16_NEON;
  }
#endif

  // Copy plane
  for (y = 0; y < height; ++y) {
    CopyRow(src_y, dst_y, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
}

// Convert a plane of 16 bit data to 8 bit
LIBYUV_API
void Convert16To8Plane(const uint16_t* src_y,
                       int src_stride_y,
                       uint8_t* dst_y,
                       int dst_stride_y,
                       int scale,  // 16384 for 10 bits
                       int width,
                       int height) {
  int y;
  void (*Convert16To8Row)(const uint16_t* src_y, uint8_t* dst_y, int scale,
                          int width) = Convert16To8Row_C;

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_stride_y = -dst_stride_y;
  }
  // Coalesce rows.
  if (src_stride_y == width && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_y = 0;
  }
#if defined(HAS_CONVERT16TO8ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    Convert16To8Row = Convert16To8Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      Convert16To8Row = Convert16To8Row_SSSE3;
    }
  }
#endif
#if defined(HAS_CONVERT16TO8ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Convert16To8Row = Convert16To8Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      Convert16To8Row = Convert16To8Row_AVX2;
    }
  }
#endif

  // Convert plane
  for (y = 0; y < height; ++y) {
    Convert16To8Row(src_y, dst_y, scale, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
}

// Convert a plane of 8 bit data to 16 bit
LIBYUV_API
void Convert8To16Plane(const uint8_t* src_y,
                       int src_stride_y,
                       uint16_t* dst_y,
                       int dst_stride_y,
                       int scale,  // 16384 for 10 bits
                       int width,
                       int height) {
  int y;
  void (*Convert8To16Row)(const uint8_t* src_y, uint16_t* dst_y, int scale,
                          int width) = Convert8To16Row_C;

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_stride_y = -dst_stride_y;
  }
  // Coalesce rows.
  if (src_stride_y == width && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_y = 0;
  }
#if defined(HAS_CONVERT8TO16ROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    Convert8To16Row = Convert8To16Row_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      Convert8To16Row = Convert8To16Row_SSE2;
    }
  }
#endif
#if defined(HAS_CONVERT8TO16ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Convert8To16Row = Convert8To16Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      Convert8To16Row = Convert8To16Row_AVX2;
    }
  }
#endif

  // Convert plane
  for (y = 0; y < height; ++y) {
    Convert8To16Row(src_y, dst_y, scale, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
}

// Copy I422.
LIBYUV_API
int I422Copy(const uint8_t* src_y,
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
  if (!src_u || !src_v || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (height - 1) * src_stride_u;
    src_v = src_v + (height - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  CopyPlane(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, height);
  CopyPlane(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, height);
  return 0;
}

// Copy I444.
LIBYUV_API
int I444Copy(const uint8_t* src_y,
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
  if (!src_u || !src_v || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (height - 1) * src_stride_u;
    src_v = src_v + (height - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  CopyPlane(src_u, src_stride_u, dst_u, dst_stride_u, width, height);
  CopyPlane(src_v, src_stride_v, dst_v, dst_stride_v, width, height);
  return 0;
}

// Copy I400.
LIBYUV_API
int I400ToI400(const uint8_t* src_y,
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

// Convert I420 to I400.
LIBYUV_API
int I420ToI400(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height) {
  (void)src_u;
  (void)src_stride_u;
  (void)src_v;
  (void)src_stride_v;
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

// Support function for NV12 etc UV channels.
// Width and height are plane sizes (typically half pixel width).
LIBYUV_API
void SplitUVPlane(const uint8_t* src_uv,
                  int src_stride_uv,
                  uint8_t* dst_u,
                  int dst_stride_u,
                  uint8_t* dst_v,
                  int dst_stride_v,
                  int width,
                  int height) {
  int y;
  void (*SplitUVRow)(const uint8_t* src_uv, uint8_t* dst_u, uint8_t* dst_v,
                     int width) = SplitUVRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_u = dst_u + (height - 1) * dst_stride_u;
    dst_v = dst_v + (height - 1) * dst_stride_v;
    dst_stride_u = -dst_stride_u;
    dst_stride_v = -dst_stride_v;
  }
  // Coalesce rows.
  if (src_stride_uv == width * 2 && dst_stride_u == width &&
      dst_stride_v == width) {
    width *= height;
    height = 1;
    src_stride_uv = dst_stride_u = dst_stride_v = 0;
  }
#if defined(HAS_SPLITUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SplitUVRow = SplitUVRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      SplitUVRow = SplitUVRow_SSE2;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    SplitUVRow = SplitUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      SplitUVRow = SplitUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SplitUVRow = SplitUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      SplitUVRow = SplitUVRow_NEON;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    SplitUVRow = SplitUVRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      SplitUVRow = SplitUVRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    // Copy a row of UV.
    SplitUVRow(src_uv, dst_u, dst_v, width);
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
    src_uv += src_stride_uv;
  }
}

LIBYUV_API
void MergeUVPlane(const uint8_t* src_u,
                  int src_stride_u,
                  const uint8_t* src_v,
                  int src_stride_v,
                  uint8_t* dst_uv,
                  int dst_stride_uv,
                  int width,
                  int height) {
  int y;
  void (*MergeUVRow)(const uint8_t* src_u, const uint8_t* src_v,
                     uint8_t* dst_uv, int width) = MergeUVRow_C;
  // Coalesce rows.
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_uv = dst_uv + (height - 1) * dst_stride_uv;
    dst_stride_uv = -dst_stride_uv;
  }
  // Coalesce rows.
  if (src_stride_u == width && src_stride_v == width &&
      dst_stride_uv == width * 2) {
    width *= height;
    height = 1;
    src_stride_u = src_stride_v = dst_stride_uv = 0;
  }
#if defined(HAS_MERGEUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    MergeUVRow = MergeUVRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      MergeUVRow = MergeUVRow_SSE2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeUVRow = MergeUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      MergeUVRow = MergeUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeUVRow = MergeUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      MergeUVRow = MergeUVRow_NEON;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    MergeUVRow = MergeUVRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      MergeUVRow = MergeUVRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    // Merge a row of U and V into a row of UV.
    MergeUVRow(src_u, src_v, dst_uv, width);
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_uv += dst_stride_uv;
  }
}

// Support function for NV12 etc RGB channels.
// Width and height are plane sizes (typically half pixel width).
LIBYUV_API
void SplitRGBPlane(const uint8_t* src_rgb,
                   int src_stride_rgb,
                   uint8_t* dst_r,
                   int dst_stride_r,
                   uint8_t* dst_g,
                   int dst_stride_g,
                   uint8_t* dst_b,
                   int dst_stride_b,
                   int width,
                   int height) {
  int y;
  void (*SplitRGBRow)(const uint8_t* src_rgb, uint8_t* dst_r, uint8_t* dst_g,
                      uint8_t* dst_b, int width) = SplitRGBRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_r = dst_r + (height - 1) * dst_stride_r;
    dst_g = dst_g + (height - 1) * dst_stride_g;
    dst_b = dst_b + (height - 1) * dst_stride_b;
    dst_stride_r = -dst_stride_r;
    dst_stride_g = -dst_stride_g;
    dst_stride_b = -dst_stride_b;
  }
  // Coalesce rows.
  if (src_stride_rgb == width * 3 && dst_stride_r == width &&
      dst_stride_g == width && dst_stride_b == width) {
    width *= height;
    height = 1;
    src_stride_rgb = dst_stride_r = dst_stride_g = dst_stride_b = 0;
  }
#if defined(HAS_SPLITRGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    SplitRGBRow = SplitRGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      SplitRGBRow = SplitRGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_SPLITRGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SplitRGBRow = SplitRGBRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      SplitRGBRow = SplitRGBRow_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    // Copy a row of RGB.
    SplitRGBRow(src_rgb, dst_r, dst_g, dst_b, width);
    dst_r += dst_stride_r;
    dst_g += dst_stride_g;
    dst_b += dst_stride_b;
    src_rgb += src_stride_rgb;
  }
}

LIBYUV_API
void MergeRGBPlane(const uint8_t* src_r,
                   int src_stride_r,
                   const uint8_t* src_g,
                   int src_stride_g,
                   const uint8_t* src_b,
                   int src_stride_b,
                   uint8_t* dst_rgb,
                   int dst_stride_rgb,
                   int width,
                   int height) {
  int y;
  void (*MergeRGBRow)(const uint8_t* src_r, const uint8_t* src_g,
                      const uint8_t* src_b, uint8_t* dst_rgb, int width) =
      MergeRGBRow_C;
  // Coalesce rows.
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb = dst_rgb + (height - 1) * dst_stride_rgb;
    dst_stride_rgb = -dst_stride_rgb;
  }
  // Coalesce rows.
  if (src_stride_r == width && src_stride_g == width && src_stride_b == width &&
      dst_stride_rgb == width * 3) {
    width *= height;
    height = 1;
    src_stride_r = src_stride_g = src_stride_b = dst_stride_rgb = 0;
  }
#if defined(HAS_MERGERGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    MergeRGBRow = MergeRGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      MergeRGBRow = MergeRGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_MERGERGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeRGBRow = MergeRGBRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      MergeRGBRow = MergeRGBRow_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    // Merge a row of U and V into a row of RGB.
    MergeRGBRow(src_r, src_g, src_b, dst_rgb, width);
    src_r += src_stride_r;
    src_g += src_stride_g;
    src_b += src_stride_b;
    dst_rgb += dst_stride_rgb;
  }
}

// Mirror a plane of data.
void MirrorPlane(const uint8_t* src_y,
                 int src_stride_y,
                 uint8_t* dst_y,
                 int dst_stride_y,
                 int width,
                 int height) {
  int y;
  void (*MirrorRow)(const uint8_t* src, uint8_t* dst, int width) = MirrorRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_stride_y = -src_stride_y;
  }
#if defined(HAS_MIRRORROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MirrorRow = MirrorRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      MirrorRow = MirrorRow_NEON;
    }
  }
#endif
#if defined(HAS_MIRRORROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    MirrorRow = MirrorRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      MirrorRow = MirrorRow_SSSE3;
    }
  }
#endif
#if defined(HAS_MIRRORROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MirrorRow = MirrorRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      MirrorRow = MirrorRow_AVX2;
    }
  }
#endif
#if defined(HAS_MIRRORROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    MirrorRow = MirrorRow_Any_MSA;
    if (IS_ALIGNED(width, 64)) {
      MirrorRow = MirrorRow_MSA;
    }
  }
#endif

  // Mirror plane
  for (y = 0; y < height; ++y) {
    MirrorRow(src_y, dst_y, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
}

// Convert YUY2 to I422.
LIBYUV_API
int YUY2ToI422(const uint8_t* src_yuy2,
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
  void (*YUY2ToUV422Row)(const uint8_t* src_yuy2, uint8_t* dst_u,
                         uint8_t* dst_v, int width) = YUY2ToUV422Row_C;
  void (*YUY2ToYRow)(const uint8_t* src_yuy2, uint8_t* dst_y, int width) =
      YUY2ToYRow_C;
  if (!src_yuy2 || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_yuy2 = src_yuy2 + (height - 1) * src_stride_yuy2;
    src_stride_yuy2 = -src_stride_yuy2;
  }
  // Coalesce rows.
  if (src_stride_yuy2 == width * 2 && dst_stride_y == width &&
      dst_stride_u * 2 == width && dst_stride_v * 2 == width &&
      width * height <= 32768) {
    width *= height;
    height = 1;
    src_stride_yuy2 = dst_stride_y = dst_stride_u = dst_stride_v = 0;
  }
#if defined(HAS_YUY2TOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    YUY2ToUV422Row = YUY2ToUV422Row_Any_SSE2;
    YUY2ToYRow = YUY2ToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToUV422Row = YUY2ToUV422Row_SSE2;
      YUY2ToYRow = YUY2ToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    YUY2ToUV422Row = YUY2ToUV422Row_Any_AVX2;
    YUY2ToYRow = YUY2ToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToUV422Row = YUY2ToUV422Row_AVX2;
      YUY2ToYRow = YUY2ToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    YUY2ToYRow = YUY2ToYRow_Any_NEON;
    YUY2ToUV422Row = YUY2ToUV422Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToYRow = YUY2ToYRow_NEON;
      YUY2ToUV422Row = YUY2ToUV422Row_NEON;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    YUY2ToYRow = YUY2ToYRow_Any_MSA;
    YUY2ToUV422Row = YUY2ToUV422Row_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_MSA;
      YUY2ToUV422Row = YUY2ToUV422Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    YUY2ToUV422Row(src_yuy2, dst_u, dst_v, width);
    YUY2ToYRow(src_yuy2, dst_y, width);
    src_yuy2 += src_stride_yuy2;
    dst_y += dst_stride_y;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  return 0;
}

// Convert UYVY to I422.
LIBYUV_API
int UYVYToI422(const uint8_t* src_uyvy,
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
  void (*UYVYToUV422Row)(const uint8_t* src_uyvy, uint8_t* dst_u,
                         uint8_t* dst_v, int width) = UYVYToUV422Row_C;
  void (*UYVYToYRow)(const uint8_t* src_uyvy, uint8_t* dst_y, int width) =
      UYVYToYRow_C;
  if (!src_uyvy || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uyvy = src_uyvy + (height - 1) * src_stride_uyvy;
    src_stride_uyvy = -src_stride_uyvy;
  }
  // Coalesce rows.
  if (src_stride_uyvy == width * 2 && dst_stride_y == width &&
      dst_stride_u * 2 == width && dst_stride_v * 2 == width &&
      width * height <= 32768) {
    width *= height;
    height = 1;
    src_stride_uyvy = dst_stride_y = dst_stride_u = dst_stride_v = 0;
  }
#if defined(HAS_UYVYTOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    UYVYToUV422Row = UYVYToUV422Row_Any_SSE2;
    UYVYToYRow = UYVYToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      UYVYToUV422Row = UYVYToUV422Row_SSE2;
      UYVYToYRow = UYVYToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    UYVYToUV422Row = UYVYToUV422Row_Any_AVX2;
    UYVYToYRow = UYVYToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      UYVYToUV422Row = UYVYToUV422Row_AVX2;
      UYVYToYRow = UYVYToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    UYVYToYRow = UYVYToYRow_Any_NEON;
    UYVYToUV422Row = UYVYToUV422Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      UYVYToYRow = UYVYToYRow_NEON;
      UYVYToUV422Row = UYVYToUV422Row_NEON;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    UYVYToYRow = UYVYToYRow_Any_MSA;
    UYVYToUV422Row = UYVYToUV422Row_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      UYVYToYRow = UYVYToYRow_MSA;
      UYVYToUV422Row = UYVYToUV422Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    UYVYToUV422Row(src_uyvy, dst_u, dst_v, width);
    UYVYToYRow(src_uyvy, dst_y, width);
    src_uyvy += src_stride_uyvy;
    dst_y += dst_stride_y;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  return 0;
}

// Convert YUY2 to Y.
LIBYUV_API
int YUY2ToY(const uint8_t* src_yuy2,
            int src_stride_yuy2,
            uint8_t* dst_y,
            int dst_stride_y,
            int width,
            int height) {
  int y;
  void (*YUY2ToYRow)(const uint8_t* src_yuy2, uint8_t* dst_y, int width) =
      YUY2ToYRow_C;
  if (!src_yuy2 || !dst_y || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_yuy2 = src_yuy2 + (height - 1) * src_stride_yuy2;
    src_stride_yuy2 = -src_stride_yuy2;
  }
  // Coalesce rows.
  if (src_stride_yuy2 == width * 2 && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_yuy2 = dst_stride_y = 0;
  }
#if defined(HAS_YUY2TOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    YUY2ToYRow = YUY2ToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToYRow = YUY2ToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    YUY2ToYRow = YUY2ToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    YUY2ToYRow = YUY2ToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToYRow = YUY2ToYRow_NEON;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    YUY2ToYRow = YUY2ToYRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    YUY2ToYRow(src_yuy2, dst_y, width);
    src_yuy2 += src_stride_yuy2;
    dst_y += dst_stride_y;
  }
  return 0;
}

// Mirror I400 with optional flipping
LIBYUV_API
int I400Mirror(const uint8_t* src_y,
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

  MirrorPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  return 0;
}

// Mirror I420 with optional flipping
LIBYUV_API
int I420Mirror(const uint8_t* src_y,
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
  if (!src_y || !src_u || !src_v || !dst_y || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
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
    MirrorPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  MirrorPlane(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, halfheight);
  MirrorPlane(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, halfheight);
  return 0;
}

// ARGB mirror.
LIBYUV_API
int ARGBMirror(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*ARGBMirrorRow)(const uint8_t* src, uint8_t* dst, int width) =
      ARGBMirrorRow_C;
  if (!src_argb || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
#if defined(HAS_ARGBMIRRORROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBMirrorRow = ARGBMirrorRow_Any_NEON;
    if (IS_ALIGNED(width, 4)) {
      ARGBMirrorRow = ARGBMirrorRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBMIRRORROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBMirrorRow = ARGBMirrorRow_Any_SSE2;
    if (IS_ALIGNED(width, 4)) {
      ARGBMirrorRow = ARGBMirrorRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBMIRRORROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBMirrorRow = ARGBMirrorRow_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBMirrorRow = ARGBMirrorRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBMIRRORROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBMirrorRow = ARGBMirrorRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBMirrorRow = ARGBMirrorRow_MSA;
    }
  }
#endif

  // Mirror plane
  for (y = 0; y < height; ++y) {
    ARGBMirrorRow(src_argb, dst_argb, width);
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Get a blender that optimized for the CPU and pixel count.
// As there are 6 blenders to choose from, the caller should try to use
// the same blend function for all pixels if possible.
LIBYUV_API
ARGBBlendRow GetARGBBlend() {
  void (*ARGBBlendRow)(const uint8_t* src_argb, const uint8_t* src_argb1,
                       uint8_t* dst_argb, int width) = ARGBBlendRow_C;
#if defined(HAS_ARGBBLENDROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBBlendRow = ARGBBlendRow_SSSE3;
    return ARGBBlendRow;
  }
#endif
#if defined(HAS_ARGBBLENDROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBBlendRow = ARGBBlendRow_NEON;
  }
#endif
#if defined(HAS_ARGBBLENDROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBBlendRow = ARGBBlendRow_MSA;
  }
#endif
  return ARGBBlendRow;
}

// Alpha Blend 2 ARGB images and store to destination.
LIBYUV_API
int ARGBBlend(const uint8_t* src_argb0,
              int src_stride_argb0,
              const uint8_t* src_argb1,
              int src_stride_argb1,
              uint8_t* dst_argb,
              int dst_stride_argb,
              int width,
              int height) {
  int y;
  void (*ARGBBlendRow)(const uint8_t* src_argb, const uint8_t* src_argb1,
                       uint8_t* dst_argb, int width) = GetARGBBlend();
  if (!src_argb0 || !src_argb1 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb0 == width * 4 && src_stride_argb1 == width * 4 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb0 = src_stride_argb1 = dst_stride_argb = 0;
  }

  for (y = 0; y < height; ++y) {
    ARGBBlendRow(src_argb0, src_argb1, dst_argb, width);
    src_argb0 += src_stride_argb0;
    src_argb1 += src_stride_argb1;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Alpha Blend plane and store to destination.
LIBYUV_API
int BlendPlane(const uint8_t* src_y0,
               int src_stride_y0,
               const uint8_t* src_y1,
               int src_stride_y1,
               const uint8_t* alpha,
               int alpha_stride,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height) {
  int y;
  void (*BlendPlaneRow)(const uint8_t* src0, const uint8_t* src1,
                        const uint8_t* alpha, uint8_t* dst, int width) =
      BlendPlaneRow_C;
  if (!src_y0 || !src_y1 || !alpha || !dst_y || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_stride_y = -dst_stride_y;
  }

  // Coalesce rows for Y plane.
  if (src_stride_y0 == width && src_stride_y1 == width &&
      alpha_stride == width && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_y0 = src_stride_y1 = alpha_stride = dst_stride_y = 0;
  }

#if defined(HAS_BLENDPLANEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    BlendPlaneRow = BlendPlaneRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      BlendPlaneRow = BlendPlaneRow_SSSE3;
    }
  }
#endif
#if defined(HAS_BLENDPLANEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    BlendPlaneRow = BlendPlaneRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      BlendPlaneRow = BlendPlaneRow_AVX2;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    BlendPlaneRow(src_y0, src_y1, alpha, dst_y, width);
    src_y0 += src_stride_y0;
    src_y1 += src_stride_y1;
    alpha += alpha_stride;
    dst_y += dst_stride_y;
  }
  return 0;
}

#define MAXTWIDTH 2048
// Alpha Blend YUV images and store to destination.
LIBYUV_API
int I420Blend(const uint8_t* src_y0,
              int src_stride_y0,
              const uint8_t* src_u0,
              int src_stride_u0,
              const uint8_t* src_v0,
              int src_stride_v0,
              const uint8_t* src_y1,
              int src_stride_y1,
              const uint8_t* src_u1,
              int src_stride_u1,
              const uint8_t* src_v1,
              int src_stride_v1,
              const uint8_t* alpha,
              int alpha_stride,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height) {
  int y;
  // Half width/height for UV.
  int halfwidth = (width + 1) >> 1;
  void (*BlendPlaneRow)(const uint8_t* src0, const uint8_t* src1,
                        const uint8_t* alpha, uint8_t* dst, int width) =
      BlendPlaneRow_C;
  void (*ScaleRowDown2)(const uint8_t* src_ptr, ptrdiff_t src_stride,
                        uint8_t* dst_ptr, int dst_width) = ScaleRowDown2Box_C;
  if (!src_y0 || !src_u0 || !src_v0 || !src_y1 || !src_u1 || !src_v1 ||
      !alpha || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_stride_y = -dst_stride_y;
  }

  // Blend Y plane.
  BlendPlane(src_y0, src_stride_y0, src_y1, src_stride_y1, alpha, alpha_stride,
             dst_y, dst_stride_y, width, height);

#if defined(HAS_BLENDPLANEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    BlendPlaneRow = BlendPlaneRow_Any_SSSE3;
    if (IS_ALIGNED(halfwidth, 8)) {
      BlendPlaneRow = BlendPlaneRow_SSSE3;
    }
  }
#endif
#if defined(HAS_BLENDPLANEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    BlendPlaneRow = BlendPlaneRow_Any_AVX2;
    if (IS_ALIGNED(halfwidth, 32)) {
      BlendPlaneRow = BlendPlaneRow_AVX2;
    }
  }
#endif
  if (!IS_ALIGNED(width, 2)) {
    ScaleRowDown2 = ScaleRowDown2Box_Odd_C;
  }
#if defined(HAS_SCALEROWDOWN2_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleRowDown2 = ScaleRowDown2Box_Odd_NEON;
    if (IS_ALIGNED(width, 2)) {
      ScaleRowDown2 = ScaleRowDown2Box_Any_NEON;
      if (IS_ALIGNED(halfwidth, 16)) {
        ScaleRowDown2 = ScaleRowDown2Box_NEON;
      }
    }
  }
#endif
#if defined(HAS_SCALEROWDOWN2_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ScaleRowDown2 = ScaleRowDown2Box_Odd_SSSE3;
    if (IS_ALIGNED(width, 2)) {
      ScaleRowDown2 = ScaleRowDown2Box_Any_SSSE3;
      if (IS_ALIGNED(halfwidth, 16)) {
        ScaleRowDown2 = ScaleRowDown2Box_SSSE3;
      }
    }
  }
#endif
#if defined(HAS_SCALEROWDOWN2_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ScaleRowDown2 = ScaleRowDown2Box_Odd_AVX2;
    if (IS_ALIGNED(width, 2)) {
      ScaleRowDown2 = ScaleRowDown2Box_Any_AVX2;
      if (IS_ALIGNED(halfwidth, 32)) {
        ScaleRowDown2 = ScaleRowDown2Box_AVX2;
      }
    }
  }
#endif

  // Row buffer for intermediate alpha pixels.
  align_buffer_64(halfalpha, halfwidth);
  for (y = 0; y < height; y += 2) {
    // last row of odd height image use 1 row of alpha instead of 2.
    if (y == (height - 1)) {
      alpha_stride = 0;
    }
    // Subsample 2 rows of UV to half width and half height.
    ScaleRowDown2(alpha, alpha_stride, halfalpha, halfwidth);
    alpha += alpha_stride * 2;
    BlendPlaneRow(src_u0, src_u1, halfalpha, dst_u, halfwidth);
    BlendPlaneRow(src_v0, src_v1, halfalpha, dst_v, halfwidth);
    src_u0 += src_stride_u0;
    src_u1 += src_stride_u1;
    dst_u += dst_stride_u;
    src_v0 += src_stride_v0;
    src_v1 += src_stride_v1;
    dst_v += dst_stride_v;
  }
  free_aligned_buffer_64(halfalpha);
  return 0;
}

// Multiply 2 ARGB images and store to destination.
LIBYUV_API
int ARGBMultiply(const uint8_t* src_argb0,
                 int src_stride_argb0,
                 const uint8_t* src_argb1,
                 int src_stride_argb1,
                 uint8_t* dst_argb,
                 int dst_stride_argb,
                 int width,
                 int height) {
  int y;
  void (*ARGBMultiplyRow)(const uint8_t* src0, const uint8_t* src1,
                          uint8_t* dst, int width) = ARGBMultiplyRow_C;
  if (!src_argb0 || !src_argb1 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb0 == width * 4 && src_stride_argb1 == width * 4 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb0 = src_stride_argb1 = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBMULTIPLYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBMultiplyRow = ARGBMultiplyRow_Any_SSE2;
    if (IS_ALIGNED(width, 4)) {
      ARGBMultiplyRow = ARGBMultiplyRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBMULTIPLYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBMultiplyRow = ARGBMultiplyRow_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBMultiplyRow = ARGBMultiplyRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBMULTIPLYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBMultiplyRow = ARGBMultiplyRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBMultiplyRow = ARGBMultiplyRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBMULTIPLYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBMultiplyRow = ARGBMultiplyRow_Any_MSA;
    if (IS_ALIGNED(width, 4)) {
      ARGBMultiplyRow = ARGBMultiplyRow_MSA;
    }
  }
#endif

  // Multiply plane
  for (y = 0; y < height; ++y) {
    ARGBMultiplyRow(src_argb0, src_argb1, dst_argb, width);
    src_argb0 += src_stride_argb0;
    src_argb1 += src_stride_argb1;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Add 2 ARGB images and store to destination.
LIBYUV_API
int ARGBAdd(const uint8_t* src_argb0,
            int src_stride_argb0,
            const uint8_t* src_argb1,
            int src_stride_argb1,
            uint8_t* dst_argb,
            int dst_stride_argb,
            int width,
            int height) {
  int y;
  void (*ARGBAddRow)(const uint8_t* src0, const uint8_t* src1, uint8_t* dst,
                     int width) = ARGBAddRow_C;
  if (!src_argb0 || !src_argb1 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb0 == width * 4 && src_stride_argb1 == width * 4 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb0 = src_stride_argb1 = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBADDROW_SSE2) && (defined(_MSC_VER) && !defined(__clang__))
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBAddRow = ARGBAddRow_SSE2;
  }
#endif
#if defined(HAS_ARGBADDROW_SSE2) && !(defined(_MSC_VER) && !defined(__clang__))
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBAddRow = ARGBAddRow_Any_SSE2;
    if (IS_ALIGNED(width, 4)) {
      ARGBAddRow = ARGBAddRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBADDROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBAddRow = ARGBAddRow_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBAddRow = ARGBAddRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBADDROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBAddRow = ARGBAddRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBAddRow = ARGBAddRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBADDROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBAddRow = ARGBAddRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      ARGBAddRow = ARGBAddRow_MSA;
    }
  }
#endif

  // Add plane
  for (y = 0; y < height; ++y) {
    ARGBAddRow(src_argb0, src_argb1, dst_argb, width);
    src_argb0 += src_stride_argb0;
    src_argb1 += src_stride_argb1;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Subtract 2 ARGB images and store to destination.
LIBYUV_API
int ARGBSubtract(const uint8_t* src_argb0,
                 int src_stride_argb0,
                 const uint8_t* src_argb1,
                 int src_stride_argb1,
                 uint8_t* dst_argb,
                 int dst_stride_argb,
                 int width,
                 int height) {
  int y;
  void (*ARGBSubtractRow)(const uint8_t* src0, const uint8_t* src1,
                          uint8_t* dst, int width) = ARGBSubtractRow_C;
  if (!src_argb0 || !src_argb1 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb0 == width * 4 && src_stride_argb1 == width * 4 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb0 = src_stride_argb1 = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBSUBTRACTROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBSubtractRow = ARGBSubtractRow_Any_SSE2;
    if (IS_ALIGNED(width, 4)) {
      ARGBSubtractRow = ARGBSubtractRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBSUBTRACTROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBSubtractRow = ARGBSubtractRow_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBSubtractRow = ARGBSubtractRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBSUBTRACTROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBSubtractRow = ARGBSubtractRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBSubtractRow = ARGBSubtractRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBSUBTRACTROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBSubtractRow = ARGBSubtractRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      ARGBSubtractRow = ARGBSubtractRow_MSA;
    }
  }
#endif

  // Subtract plane
  for (y = 0; y < height; ++y) {
    ARGBSubtractRow(src_argb0, src_argb1, dst_argb, width);
    src_argb0 += src_stride_argb0;
    src_argb1 += src_stride_argb1;
    dst_argb += dst_stride_argb;
  }
  return 0;
}
// Convert I422 to RGBA with matrix
static int I422ToRGBAMatrix(const uint8_t* src_y,
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
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I422 to RGBA.
LIBYUV_API
int I422ToRGBA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_rgba,
               int dst_stride_rgba,
               int width,
               int height) {
  return I422ToRGBAMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_rgba, dst_stride_rgba,
                          &kYuvI601Constants, width, height);
}

// Convert I422 to BGRA.
LIBYUV_API
int I422ToBGRA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_bgra,
               int dst_stride_bgra,
               int width,
               int height) {
  return I422ToRGBAMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_bgra, dst_stride_bgra,
                          &kYvuI601Constants,  // Use Yvu matrix
                          width, height);
}

// Convert NV12 to RGB565.
LIBYUV_API
int NV12ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_uv,
                 int src_stride_uv,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height) {
  int y;
  void (*NV12ToRGB565Row)(
      const uint8_t* y_buf, const uint8_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = NV12ToRGB565Row_C;
  if (!src_y || !src_uv || !dst_rgb565 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb565 = dst_rgb565 + (height - 1) * dst_stride_rgb565;
    dst_stride_rgb565 = -dst_stride_rgb565;
  }
#if defined(HAS_NV12TORGB565ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    NV12ToRGB565Row = NV12ToRGB565Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      NV12ToRGB565Row = NV12ToRGB565Row_SSSE3;
    }
  }
#endif
#if defined(HAS_NV12TORGB565ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    NV12ToRGB565Row = NV12ToRGB565Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      NV12ToRGB565Row = NV12ToRGB565Row_AVX2;
    }
  }
#endif
#if defined(HAS_NV12TORGB565ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    NV12ToRGB565Row = NV12ToRGB565Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      NV12ToRGB565Row = NV12ToRGB565Row_NEON;
    }
  }
#endif
#if defined(HAS_NV12TORGB565ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    NV12ToRGB565Row = NV12ToRGB565Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      NV12ToRGB565Row = NV12ToRGB565Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    NV12ToRGB565Row(src_y, src_uv, dst_rgb565, &kYuvI601Constants, width);
    dst_rgb565 += dst_stride_rgb565;
    src_y += src_stride_y;
    if (y & 1) {
      src_uv += src_stride_uv;
    }
  }
  return 0;
}

// Convert RAW to RGB24.
LIBYUV_API
int RAWToRGB24(const uint8_t* src_raw,
               int src_stride_raw,
               uint8_t* dst_rgb24,
               int dst_stride_rgb24,
               int width,
               int height) {
  int y;
  void (*RAWToRGB24Row)(const uint8_t* src_rgb, uint8_t* dst_rgb24, int width) =
      RAWToRGB24Row_C;
  if (!src_raw || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }
  // Coalesce rows.
  if (src_stride_raw == width * 3 && dst_stride_rgb24 == width * 3) {
    width *= height;
    height = 1;
    src_stride_raw = dst_stride_rgb24 = 0;
  }
#if defined(HAS_RAWTORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RAWToRGB24Row = RAWToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      RAWToRGB24Row = RAWToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_RAWTORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RAWToRGB24Row = RAWToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RAWToRGB24Row = RAWToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_RAWTORGB24ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RAWToRGB24Row = RAWToRGB24Row_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RAWToRGB24Row = RAWToRGB24Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    RAWToRGB24Row(src_raw, dst_rgb24, width);
    src_raw += src_stride_raw;
    dst_rgb24 += dst_stride_rgb24;
  }
  return 0;
}

LIBYUV_API
void SetPlane(uint8_t* dst_y,
              int dst_stride_y,
              int width,
              int height,
              uint32_t value) {
  int y;
  void (*SetRow)(uint8_t * dst, uint8_t value, int width) = SetRow_C;
  if (height < 0) {
    height = -height;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_stride_y = -dst_stride_y;
  }
  // Coalesce rows.
  if (dst_stride_y == width) {
    width *= height;
    height = 1;
    dst_stride_y = 0;
  }
#if defined(HAS_SETROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SetRow = SetRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      SetRow = SetRow_NEON;
    }
  }
#endif
#if defined(HAS_SETROW_X86)
  if (TestCpuFlag(kCpuHasX86)) {
    SetRow = SetRow_Any_X86;
    if (IS_ALIGNED(width, 4)) {
      SetRow = SetRow_X86;
    }
  }
#endif
#if defined(HAS_SETROW_ERMS)
  if (TestCpuFlag(kCpuHasERMS)) {
    SetRow = SetRow_ERMS;
  }
#endif
#if defined(HAS_SETROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 16)) {
    SetRow = SetRow_MSA;
  }
#endif

  // Set plane
  for (y = 0; y < height; ++y) {
    SetRow(dst_y, value, width);
    dst_y += dst_stride_y;
  }
}

// Draw a rectangle into I420
LIBYUV_API
int I420Rect(uint8_t* dst_y,
             int dst_stride_y,
             uint8_t* dst_u,
             int dst_stride_u,
             uint8_t* dst_v,
             int dst_stride_v,
             int x,
             int y,
             int width,
             int height,
             int value_y,
             int value_u,
             int value_v) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  uint8_t* start_y = dst_y + y * dst_stride_y + x;
  uint8_t* start_u = dst_u + (y / 2) * dst_stride_u + (x / 2);
  uint8_t* start_v = dst_v + (y / 2) * dst_stride_v + (x / 2);
  if (!dst_y || !dst_u || !dst_v || width <= 0 || height == 0 || x < 0 ||
      y < 0 || value_y < 0 || value_y > 255 || value_u < 0 || value_u > 255 ||
      value_v < 0 || value_v > 255) {
    return -1;
  }

  SetPlane(start_y, dst_stride_y, width, height, value_y);
  SetPlane(start_u, dst_stride_u, halfwidth, halfheight, value_u);
  SetPlane(start_v, dst_stride_v, halfwidth, halfheight, value_v);
  return 0;
}

// Draw a rectangle into ARGB
LIBYUV_API
int ARGBRect(uint8_t* dst_argb,
             int dst_stride_argb,
             int dst_x,
             int dst_y,
             int width,
             int height,
             uint32_t value) {
  int y;
  void (*ARGBSetRow)(uint8_t * dst_argb, uint32_t value, int width) =
      ARGBSetRow_C;
  if (!dst_argb || width <= 0 || height == 0 || dst_x < 0 || dst_y < 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  dst_argb += dst_y * dst_stride_argb + dst_x * 4;
  // Coalesce rows.
  if (dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    dst_stride_argb = 0;
  }

#if defined(HAS_ARGBSETROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBSetRow = ARGBSetRow_Any_NEON;
    if (IS_ALIGNED(width, 4)) {
      ARGBSetRow = ARGBSetRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBSETROW_X86)
  if (TestCpuFlag(kCpuHasX86)) {
    ARGBSetRow = ARGBSetRow_X86;
  }
#endif
#if defined(HAS_ARGBSETROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBSetRow = ARGBSetRow_Any_MSA;
    if (IS_ALIGNED(width, 4)) {
      ARGBSetRow = ARGBSetRow_MSA;
    }
  }
#endif

  // Set plane
  for (y = 0; y < height; ++y) {
    ARGBSetRow(dst_argb, value, width);
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert unattentuated ARGB to preattenuated ARGB.
// An unattenutated ARGB alpha blend uses the formula
// p = a * f + (1 - a) * b
// where
//   p is output pixel
//   f is foreground pixel
//   b is background pixel
//   a is alpha value from foreground pixel
// An preattenutated ARGB alpha blend uses the formula
// p = f + (1 - a) * b
// where
//   f is foreground pixel premultiplied by alpha

LIBYUV_API
int ARGBAttenuate(const uint8_t* src_argb,
                  int src_stride_argb,
                  uint8_t* dst_argb,
                  int dst_stride_argb,
                  int width,
                  int height) {
  int y;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  if (!src_argb || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBATTENUATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBAttenuateRow = ARGBAttenuateRow_Any_SSSE3;
    if (IS_ALIGNED(width, 4)) {
      ARGBAttenuateRow = ARGBAttenuateRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBATTENUATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBAttenuateRow = ARGBAttenuateRow_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBAttenuateRow = ARGBAttenuateRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBATTENUATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBAttenuateRow = ARGBAttenuateRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBAttenuateRow = ARGBAttenuateRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBATTENUATEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBAttenuateRow = ARGBAttenuateRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      ARGBAttenuateRow = ARGBAttenuateRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBAttenuateRow(src_argb, dst_argb, width);
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert preattentuated ARGB to unattenuated ARGB.
LIBYUV_API
int ARGBUnattenuate(const uint8_t* src_argb,
                    int src_stride_argb,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height) {
  int y;
  void (*ARGBUnattenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                             int width) = ARGBUnattenuateRow_C;
  if (!src_argb || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBUNATTENUATEROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBUnattenuateRow = ARGBUnattenuateRow_Any_SSE2;
    if (IS_ALIGNED(width, 4)) {
      ARGBUnattenuateRow = ARGBUnattenuateRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBUNATTENUATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBUnattenuateRow = ARGBUnattenuateRow_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBUnattenuateRow = ARGBUnattenuateRow_AVX2;
    }
  }
#endif
  // TODO(fbarchard): Neon version.

  for (y = 0; y < height; ++y) {
    ARGBUnattenuateRow(src_argb, dst_argb, width);
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert ARGB to Grayed ARGB.
LIBYUV_API
int ARGBGrayTo(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*ARGBGrayRow)(const uint8_t* src_argb, uint8_t* dst_argb, int width) =
      ARGBGrayRow_C;
  if (!src_argb || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBGRAYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_SSSE3;
  }
#endif
#if defined(HAS_ARGBGRAYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_NEON;
  }
#endif
#if defined(HAS_ARGBGRAYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_MSA;
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBGrayRow(src_argb, dst_argb, width);
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Make a rectangle of ARGB gray scale.
LIBYUV_API
int ARGBGray(uint8_t* dst_argb,
             int dst_stride_argb,
             int dst_x,
             int dst_y,
             int width,
             int height) {
  int y;
  void (*ARGBGrayRow)(const uint8_t* src_argb, uint8_t* dst_argb, int width) =
      ARGBGrayRow_C;
  uint8_t* dst = dst_argb + dst_y * dst_stride_argb + dst_x * 4;
  if (!dst_argb || width <= 0 || height <= 0 || dst_x < 0 || dst_y < 0) {
    return -1;
  }
  // Coalesce rows.
  if (dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    dst_stride_argb = 0;
  }
#if defined(HAS_ARGBGRAYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_SSSE3;
  }
#endif
#if defined(HAS_ARGBGRAYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_NEON;
  }
#endif
#if defined(HAS_ARGBGRAYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_MSA;
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBGrayRow(dst, dst, width);
    dst += dst_stride_argb;
  }
  return 0;
}

// Make a rectangle of ARGB Sepia tone.
LIBYUV_API
int ARGBSepia(uint8_t* dst_argb,
              int dst_stride_argb,
              int dst_x,
              int dst_y,
              int width,
              int height) {
  int y;
  void (*ARGBSepiaRow)(uint8_t * dst_argb, int width) = ARGBSepiaRow_C;
  uint8_t* dst = dst_argb + dst_y * dst_stride_argb + dst_x * 4;
  if (!dst_argb || width <= 0 || height <= 0 || dst_x < 0 || dst_y < 0) {
    return -1;
  }
  // Coalesce rows.
  if (dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    dst_stride_argb = 0;
  }
#if defined(HAS_ARGBSEPIAROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && IS_ALIGNED(width, 8)) {
    ARGBSepiaRow = ARGBSepiaRow_SSSE3;
  }
#endif
#if defined(HAS_ARGBSEPIAROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 8)) {
    ARGBSepiaRow = ARGBSepiaRow_NEON;
  }
#endif
#if defined(HAS_ARGBSEPIAROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 8)) {
    ARGBSepiaRow = ARGBSepiaRow_MSA;
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBSepiaRow(dst, width);
    dst += dst_stride_argb;
  }
  return 0;
}

// Apply a 4x4 matrix to each ARGB pixel.
// Note: Normally for shading, but can be used to swizzle or invert.
LIBYUV_API
int ARGBColorMatrix(const uint8_t* src_argb,
                    int src_stride_argb,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    const int8_t* matrix_argb,
                    int width,
                    int height) {
  int y;
  void (*ARGBColorMatrixRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                             const int8_t* matrix_argb, int width) =
      ARGBColorMatrixRow_C;
  if (!src_argb || !dst_argb || !matrix_argb || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBCOLORMATRIXROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && IS_ALIGNED(width, 8)) {
    ARGBColorMatrixRow = ARGBColorMatrixRow_SSSE3;
  }
#endif
#if defined(HAS_ARGBCOLORMATRIXROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 8)) {
    ARGBColorMatrixRow = ARGBColorMatrixRow_NEON;
  }
#endif
#if defined(HAS_ARGBCOLORMATRIXROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 8)) {
    ARGBColorMatrixRow = ARGBColorMatrixRow_MSA;
  }
#endif
  for (y = 0; y < height; ++y) {
    ARGBColorMatrixRow(src_argb, dst_argb, matrix_argb, width);
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Apply a 4x3 matrix to each ARGB pixel.
// Deprecated.
LIBYUV_API
int RGBColorMatrix(uint8_t* dst_argb,
                   int dst_stride_argb,
                   const int8_t* matrix_rgb,
                   int dst_x,
                   int dst_y,
                   int width,
                   int height) {
  SIMD_ALIGNED(int8_t matrix_argb[16]);
  uint8_t* dst = dst_argb + dst_y * dst_stride_argb + dst_x * 4;
  if (!dst_argb || !matrix_rgb || width <= 0 || height <= 0 || dst_x < 0 ||
      dst_y < 0) {
    return -1;
  }

  // Convert 4x3 7 bit matrix to 4x4 6 bit matrix.
  matrix_argb[0] = matrix_rgb[0] / 2;
  matrix_argb[1] = matrix_rgb[1] / 2;
  matrix_argb[2] = matrix_rgb[2] / 2;
  matrix_argb[3] = matrix_rgb[3] / 2;
  matrix_argb[4] = matrix_rgb[4] / 2;
  matrix_argb[5] = matrix_rgb[5] / 2;
  matrix_argb[6] = matrix_rgb[6] / 2;
  matrix_argb[7] = matrix_rgb[7] / 2;
  matrix_argb[8] = matrix_rgb[8] / 2;
  matrix_argb[9] = matrix_rgb[9] / 2;
  matrix_argb[10] = matrix_rgb[10] / 2;
  matrix_argb[11] = matrix_rgb[11] / 2;
  matrix_argb[14] = matrix_argb[13] = matrix_argb[12] = 0;
  matrix_argb[15] = 64;  // 1.0

  return ARGBColorMatrix((const uint8_t*)(dst), dst_stride_argb, dst,
                         dst_stride_argb, &matrix_argb[0], width, height);
}

// Apply a color table each ARGB pixel.
// Table contains 256 ARGB values.
LIBYUV_API
int ARGBColorTable(uint8_t* dst_argb,
                   int dst_stride_argb,
                   const uint8_t* table_argb,
                   int dst_x,
                   int dst_y,
                   int width,
                   int height) {
  int y;
  void (*ARGBColorTableRow)(uint8_t * dst_argb, const uint8_t* table_argb,
                            int width) = ARGBColorTableRow_C;
  uint8_t* dst = dst_argb + dst_y * dst_stride_argb + dst_x * 4;
  if (!dst_argb || !table_argb || width <= 0 || height <= 0 || dst_x < 0 ||
      dst_y < 0) {
    return -1;
  }
  // Coalesce rows.
  if (dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    dst_stride_argb = 0;
  }
#if defined(HAS_ARGBCOLORTABLEROW_X86)
  if (TestCpuFlag(kCpuHasX86)) {
    ARGBColorTableRow = ARGBColorTableRow_X86;
  }
#endif
  for (y = 0; y < height; ++y) {
    ARGBColorTableRow(dst, table_argb, width);
    dst += dst_stride_argb;
  }
  return 0;
}

// Apply a color table each ARGB pixel but preserve destination alpha.
// Table contains 256 ARGB values.
LIBYUV_API
int RGBColorTable(uint8_t* dst_argb,
                  int dst_stride_argb,
                  const uint8_t* table_argb,
                  int dst_x,
                  int dst_y,
                  int width,
                  int height) {
  int y;
  void (*RGBColorTableRow)(uint8_t * dst_argb, const uint8_t* table_argb,
                           int width) = RGBColorTableRow_C;
  uint8_t* dst = dst_argb + dst_y * dst_stride_argb + dst_x * 4;
  if (!dst_argb || !table_argb || width <= 0 || height <= 0 || dst_x < 0 ||
      dst_y < 0) {
    return -1;
  }
  // Coalesce rows.
  if (dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    dst_stride_argb = 0;
  }
#if defined(HAS_RGBCOLORTABLEROW_X86)
  if (TestCpuFlag(kCpuHasX86)) {
    RGBColorTableRow = RGBColorTableRow_X86;
  }
#endif
  for (y = 0; y < height; ++y) {
    RGBColorTableRow(dst, table_argb, width);
    dst += dst_stride_argb;
  }
  return 0;
}

// ARGBQuantize is used to posterize art.
// e.g. rgb / qvalue * qvalue + qvalue / 2
// But the low levels implement efficiently with 3 parameters, and could be
// used for other high level operations.
// dst_argb[0] = (b * scale >> 16) * interval_size + interval_offset;
// where scale is 1 / interval_size as a fixed point value.
// The divide is replaces with a multiply by reciprocal fixed point multiply.
// Caveat - although SSE2 saturates, the C function does not and should be used
// with care if doing anything but quantization.
LIBYUV_API
int ARGBQuantize(uint8_t* dst_argb,
                 int dst_stride_argb,
                 int scale,
                 int interval_size,
                 int interval_offset,
                 int dst_x,
                 int dst_y,
                 int width,
                 int height) {
  int y;
  void (*ARGBQuantizeRow)(uint8_t * dst_argb, int scale, int interval_size,
                          int interval_offset, int width) = ARGBQuantizeRow_C;
  uint8_t* dst = dst_argb + dst_y * dst_stride_argb + dst_x * 4;
  if (!dst_argb || width <= 0 || height <= 0 || dst_x < 0 || dst_y < 0 ||
      interval_size < 1 || interval_size > 255) {
    return -1;
  }
  // Coalesce rows.
  if (dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    dst_stride_argb = 0;
  }
#if defined(HAS_ARGBQUANTIZEROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(width, 4)) {
    ARGBQuantizeRow = ARGBQuantizeRow_SSE2;
  }
#endif
#if defined(HAS_ARGBQUANTIZEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 8)) {
    ARGBQuantizeRow = ARGBQuantizeRow_NEON;
  }
#endif
#if defined(HAS_ARGBQUANTIZEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 8)) {
    ARGBQuantizeRow = ARGBQuantizeRow_MSA;
  }
#endif
  for (y = 0; y < height; ++y) {
    ARGBQuantizeRow(dst, scale, interval_size, interval_offset, width);
    dst += dst_stride_argb;
  }
  return 0;
}

// Computes table of cumulative sum for image where the value is the sum
// of all values above and to the left of the entry. Used by ARGBBlur.
LIBYUV_API
int ARGBComputeCumulativeSum(const uint8_t* src_argb,
                             int src_stride_argb,
                             int32_t* dst_cumsum,
                             int dst_stride32_cumsum,
                             int width,
                             int height) {
  int y;
  void (*ComputeCumulativeSumRow)(const uint8_t* row, int32_t* cumsum,
                                  const int32_t* previous_cumsum, int width) =
      ComputeCumulativeSumRow_C;
  int32_t* previous_cumsum = dst_cumsum;
  if (!dst_cumsum || !src_argb || width <= 0 || height <= 0) {
    return -1;
  }
#if defined(HAS_CUMULATIVESUMTOAVERAGEROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ComputeCumulativeSumRow = ComputeCumulativeSumRow_SSE2;
  }
#endif
  memset(dst_cumsum, 0, width * sizeof(dst_cumsum[0]) * 4);  // 4 int per pixel.
  for (y = 0; y < height; ++y) {
    ComputeCumulativeSumRow(src_argb, dst_cumsum, previous_cumsum, width);
    previous_cumsum = dst_cumsum;
    dst_cumsum += dst_stride32_cumsum;
    src_argb += src_stride_argb;
  }
  return 0;
}

// Blur ARGB image.
// Caller should allocate CumulativeSum table of width * height * 16 bytes
// aligned to 16 byte boundary. height can be radius * 2 + 2 to save memory
// as the buffer is treated as circular.
LIBYUV_API
int ARGBBlur(const uint8_t* src_argb,
             int src_stride_argb,
             uint8_t* dst_argb,
             int dst_stride_argb,
             int32_t* dst_cumsum,
             int dst_stride32_cumsum,
             int width,
             int height,
             int radius) {
  int y;
  void (*ComputeCumulativeSumRow)(const uint8_t* row, int32_t* cumsum,
                                  const int32_t* previous_cumsum, int width) =
      ComputeCumulativeSumRow_C;
  void (*CumulativeSumToAverageRow)(
      const int32_t* topleft, const int32_t* botleft, int width, int area,
      uint8_t* dst, int count) = CumulativeSumToAverageRow_C;
  int32_t* cumsum_bot_row;
  int32_t* max_cumsum_bot_row;
  int32_t* cumsum_top_row;

  if (!src_argb || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  if (radius > height) {
    radius = height;
  }
  if (radius > (width / 2 - 1)) {
    radius = width / 2 - 1;
  }
  if (radius <= 0) {
    return -1;
  }
#if defined(HAS_CUMULATIVESUMTOAVERAGEROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ComputeCumulativeSumRow = ComputeCumulativeSumRow_SSE2;
    CumulativeSumToAverageRow = CumulativeSumToAverageRow_SSE2;
  }
#endif
  // Compute enough CumulativeSum for first row to be blurred. After this
  // one row of CumulativeSum is updated at a time.
  ARGBComputeCumulativeSum(src_argb, src_stride_argb, dst_cumsum,
                           dst_stride32_cumsum, width, radius);

  src_argb = src_argb + radius * src_stride_argb;
  cumsum_bot_row = &dst_cumsum[(radius - 1) * dst_stride32_cumsum];

  max_cumsum_bot_row = &dst_cumsum[(radius * 2 + 2) * dst_stride32_cumsum];
  cumsum_top_row = &dst_cumsum[0];

  for (y = 0; y < height; ++y) {
    int top_y = ((y - radius - 1) >= 0) ? (y - radius - 1) : 0;
    int bot_y = ((y + radius) < height) ? (y + radius) : (height - 1);
    int area = radius * (bot_y - top_y);
    int boxwidth = radius * 4;
    int x;
    int n;

    // Increment cumsum_top_row pointer with circular buffer wrap around.
    if (top_y) {
      cumsum_top_row += dst_stride32_cumsum;
      if (cumsum_top_row >= max_cumsum_bot_row) {
        cumsum_top_row = dst_cumsum;
      }
    }
    // Increment cumsum_bot_row pointer with circular buffer wrap around and
    // then fill in a row of CumulativeSum.
    if ((y + radius) < height) {
      const int32_t* prev_cumsum_bot_row = cumsum_bot_row;
      cumsum_bot_row += dst_stride32_cumsum;
      if (cumsum_bot_row >= max_cumsum_bot_row) {
        cumsum_bot_row = dst_cumsum;
      }
      ComputeCumulativeSumRow(src_argb, cumsum_bot_row, prev_cumsum_bot_row,
                              width);
      src_argb += src_stride_argb;
    }

    // Left clipped.
    for (x = 0; x < radius + 1; ++x) {
      CumulativeSumToAverageRow(cumsum_top_row, cumsum_bot_row, boxwidth, area,
                                &dst_argb[x * 4], 1);
      area += (bot_y - top_y);
      boxwidth += 4;
    }

    // Middle unclipped.
    n = (width - 1) - radius - x + 1;
    CumulativeSumToAverageRow(cumsum_top_row, cumsum_bot_row, boxwidth, area,
                              &dst_argb[x * 4], n);

    // Right clipped.
    for (x += n; x <= width - 1; ++x) {
      area -= (bot_y - top_y);
      boxwidth -= 4;
      CumulativeSumToAverageRow(cumsum_top_row + (x - radius - 1) * 4,
                                cumsum_bot_row + (x - radius - 1) * 4, boxwidth,
                                area, &dst_argb[x * 4], 1);
    }
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Multiply ARGB image by a specified ARGB value.
LIBYUV_API
int ARGBShade(const uint8_t* src_argb,
              int src_stride_argb,
              uint8_t* dst_argb,
              int dst_stride_argb,
              int width,
              int height,
              uint32_t value) {
  int y;
  void (*ARGBShadeRow)(const uint8_t* src_argb, uint8_t* dst_argb, int width,
                       uint32_t value) = ARGBShadeRow_C;
  if (!src_argb || !dst_argb || width <= 0 || height == 0 || value == 0u) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBSHADEROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(width, 4)) {
    ARGBShadeRow = ARGBShadeRow_SSE2;
  }
#endif
#if defined(HAS_ARGBSHADEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 8)) {
    ARGBShadeRow = ARGBShadeRow_NEON;
  }
#endif
#if defined(HAS_ARGBSHADEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 4)) {
    ARGBShadeRow = ARGBShadeRow_MSA;
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBShadeRow(src_argb, dst_argb, width, value);
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Interpolate 2 planes by specified amount (0 to 255).
LIBYUV_API
int InterpolatePlane(const uint8_t* src0,
                     int src_stride0,
                     const uint8_t* src1,
                     int src_stride1,
                     uint8_t* dst,
                     int dst_stride,
                     int width,
                     int height,
                     int interpolation) {
  int y;
  void (*InterpolateRow)(uint8_t * dst_ptr, const uint8_t* src_ptr,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) = InterpolateRow_C;
  if (!src0 || !src1 || !dst || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst = dst + (height - 1) * dst_stride;
    dst_stride = -dst_stride;
  }
  // Coalesce rows.
  if (src_stride0 == width && src_stride1 == width && dst_stride == width) {
    width *= height;
    height = 1;
    src_stride0 = src_stride1 = dst_stride = 0;
  }
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      InterpolateRow = InterpolateRow_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    InterpolateRow = InterpolateRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    InterpolateRow(dst, src0, src1 - src0, width, interpolation);
    src0 += src_stride0;
    src1 += src_stride1;
    dst += dst_stride;
  }
  return 0;
}

// Interpolate 2 ARGB images by specified amount (0 to 255).
LIBYUV_API
int ARGBInterpolate(const uint8_t* src_argb0,
                    int src_stride_argb0,
                    const uint8_t* src_argb1,
                    int src_stride_argb1,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height,
                    int interpolation) {
  return InterpolatePlane(src_argb0, src_stride_argb0, src_argb1,
                          src_stride_argb1, dst_argb, dst_stride_argb,
                          width * 4, height, interpolation);
}

// Interpolate 2 YUV images by specified amount (0 to 255).
LIBYUV_API
int I420Interpolate(const uint8_t* src0_y,
                    int src0_stride_y,
                    const uint8_t* src0_u,
                    int src0_stride_u,
                    const uint8_t* src0_v,
                    int src0_stride_v,
                    const uint8_t* src1_y,
                    int src1_stride_y,
                    const uint8_t* src1_u,
                    int src1_stride_u,
                    const uint8_t* src1_v,
                    int src1_stride_v,
                    uint8_t* dst_y,
                    int dst_stride_y,
                    uint8_t* dst_u,
                    int dst_stride_u,
                    uint8_t* dst_v,
                    int dst_stride_v,
                    int width,
                    int height,
                    int interpolation) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if (!src0_y || !src0_u || !src0_v || !src1_y || !src1_u || !src1_v ||
      !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  InterpolatePlane(src0_y, src0_stride_y, src1_y, src1_stride_y, dst_y,
                   dst_stride_y, width, height, interpolation);
  InterpolatePlane(src0_u, src0_stride_u, src1_u, src1_stride_u, dst_u,
                   dst_stride_u, halfwidth, halfheight, interpolation);
  InterpolatePlane(src0_v, src0_stride_v, src1_v, src1_stride_v, dst_v,
                   dst_stride_v, halfwidth, halfheight, interpolation);
  return 0;
}

// Shuffle ARGB channel order.  e.g. BGRA to ARGB.
LIBYUV_API
int ARGBShuffle(const uint8_t* src_bgra,
                int src_stride_bgra,
                uint8_t* dst_argb,
                int dst_stride_argb,
                const uint8_t* shuffler,
                int width,
                int height) {
  int y;
  void (*ARGBShuffleRow)(const uint8_t* src_bgra, uint8_t* dst_argb,
                         const uint8_t* shuffler, int width) = ARGBShuffleRow_C;
  if (!src_bgra || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_bgra = src_bgra + (height - 1) * src_stride_bgra;
    src_stride_bgra = -src_stride_bgra;
  }
  // Coalesce rows.
  if (src_stride_bgra == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_bgra = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBSHUFFLEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBShuffleRow = ARGBShuffleRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      ARGBShuffleRow = ARGBShuffleRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBSHUFFLEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBShuffleRow = ARGBShuffleRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      ARGBShuffleRow = ARGBShuffleRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBSHUFFLEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBShuffleRow = ARGBShuffleRow_Any_NEON;
    if (IS_ALIGNED(width, 4)) {
      ARGBShuffleRow = ARGBShuffleRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBSHUFFLEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBShuffleRow = ARGBShuffleRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      ARGBShuffleRow = ARGBShuffleRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBShuffleRow(src_bgra, dst_argb, shuffler, width);
    src_bgra += src_stride_bgra;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Sobel ARGB effect.
static int ARGBSobelize(const uint8_t* src_argb,
                        int src_stride_argb,
                        uint8_t* dst_argb,
                        int dst_stride_argb,
                        int width,
                        int height,
                        void (*SobelRow)(const uint8_t* src_sobelx,
                                         const uint8_t* src_sobely,
                                         uint8_t* dst,
                                         int width)) {
  int y;
  void (*ARGBToYJRow)(const uint8_t* src_argb, uint8_t* dst_g, int width) =
      ARGBToYJRow_C;
  void (*SobelYRow)(const uint8_t* src_y0, const uint8_t* src_y1,
                    uint8_t* dst_sobely, int width) = SobelYRow_C;
  void (*SobelXRow)(const uint8_t* src_y0, const uint8_t* src_y1,
                    const uint8_t* src_y2, uint8_t* dst_sobely, int width) =
      SobelXRow_C;
  const int kEdge = 16;  // Extra pixels at start of row for extrude/align.
  if (!src_argb || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }

#if defined(HAS_ARGBTOYJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYJRow = ARGBToYJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYJRow = ARGBToYJRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYJRow = ARGBToYJRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToYJRow = ARGBToYJRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToYJRow = ARGBToYJRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToYJRow = ARGBToYJRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_MSA;
    }
  }
#endif

#if defined(HAS_SOBELYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SobelYRow = SobelYRow_SSE2;
  }
#endif
#if defined(HAS_SOBELYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SobelYRow = SobelYRow_NEON;
  }
#endif
#if defined(HAS_SOBELYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    SobelYRow = SobelYRow_MSA;
  }
#endif
#if defined(HAS_SOBELXROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SobelXRow = SobelXRow_SSE2;
  }
#endif
#if defined(HAS_SOBELXROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SobelXRow = SobelXRow_NEON;
  }
#endif
#if defined(HAS_SOBELXROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    SobelXRow = SobelXRow_MSA;
  }
#endif
  {
    // 3 rows with edges before/after.
    const int kRowSize = (width + kEdge + 31) & ~31;
    align_buffer_64(rows, kRowSize * 2 + (kEdge + kRowSize * 3 + kEdge));
    uint8_t* row_sobelx = rows;
    uint8_t* row_sobely = rows + kRowSize;
    uint8_t* row_y = rows + kRowSize * 2;

    // Convert first row.
    uint8_t* row_y0 = row_y + kEdge;
    uint8_t* row_y1 = row_y0 + kRowSize;
    uint8_t* row_y2 = row_y1 + kRowSize;
    ARGBToYJRow(src_argb, row_y0, width);
    row_y0[-1] = row_y0[0];
    memset(row_y0 + width, row_y0[width - 1], 16);  // Extrude 16 for valgrind.
    ARGBToYJRow(src_argb, row_y1, width);
    row_y1[-1] = row_y1[0];
    memset(row_y1 + width, row_y1[width - 1], 16);
    memset(row_y2 + width, 0, 16);

    for (y = 0; y < height; ++y) {
      // Convert next row of ARGB to G.
      if (y < (height - 1)) {
        src_argb += src_stride_argb;
      }
      ARGBToYJRow(src_argb, row_y2, width);
      row_y2[-1] = row_y2[0];
      row_y2[width] = row_y2[width - 1];

      SobelXRow(row_y0 - 1, row_y1 - 1, row_y2 - 1, row_sobelx, width);
      SobelYRow(row_y0 - 1, row_y2 - 1, row_sobely, width);
      SobelRow(row_sobelx, row_sobely, dst_argb, width);

      // Cycle thru circular queue of 3 row_y buffers.
      {
        uint8_t* row_yt = row_y0;
        row_y0 = row_y1;
        row_y1 = row_y2;
        row_y2 = row_yt;
      }

      dst_argb += dst_stride_argb;
    }
    free_aligned_buffer_64(rows);
  }
  return 0;
}

// Sobel ARGB effect.
LIBYUV_API
int ARGBSobel(const uint8_t* src_argb,
              int src_stride_argb,
              uint8_t* dst_argb,
              int dst_stride_argb,
              int width,
              int height) {
  void (*SobelRow)(const uint8_t* src_sobelx, const uint8_t* src_sobely,
                   uint8_t* dst_argb, int width) = SobelRow_C;
#if defined(HAS_SOBELROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SobelRow = SobelRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      SobelRow = SobelRow_SSE2;
    }
  }
#endif
#if defined(HAS_SOBELROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SobelRow = SobelRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      SobelRow = SobelRow_NEON;
    }
  }
#endif
#if defined(HAS_SOBELROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    SobelRow = SobelRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      SobelRow = SobelRow_MSA;
    }
  }
#endif
  return ARGBSobelize(src_argb, src_stride_argb, dst_argb, dst_stride_argb,
                      width, height, SobelRow);
}

// Sobel ARGB effect with planar output.
LIBYUV_API
int ARGBSobelToPlane(const uint8_t* src_argb,
                     int src_stride_argb,
                     uint8_t* dst_y,
                     int dst_stride_y,
                     int width,
                     int height) {
  void (*SobelToPlaneRow)(const uint8_t* src_sobelx, const uint8_t* src_sobely,
                          uint8_t* dst_, int width) = SobelToPlaneRow_C;
#if defined(HAS_SOBELTOPLANEROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SobelToPlaneRow = SobelToPlaneRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      SobelToPlaneRow = SobelToPlaneRow_SSE2;
    }
  }
#endif
#if defined(HAS_SOBELTOPLANEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SobelToPlaneRow = SobelToPlaneRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      SobelToPlaneRow = SobelToPlaneRow_NEON;
    }
  }
#endif
#if defined(HAS_SOBELTOPLANEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    SobelToPlaneRow = SobelToPlaneRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      SobelToPlaneRow = SobelToPlaneRow_MSA;
    }
  }
#endif
  return ARGBSobelize(src_argb, src_stride_argb, dst_y, dst_stride_y, width,
                      height, SobelToPlaneRow);
}

// SobelXY ARGB effect.
// Similar to Sobel, but also stores Sobel X in R and Sobel Y in B.  G = Sobel.
LIBYUV_API
int ARGBSobelXY(const uint8_t* src_argb,
                int src_stride_argb,
                uint8_t* dst_argb,
                int dst_stride_argb,
                int width,
                int height) {
  void (*SobelXYRow)(const uint8_t* src_sobelx, const uint8_t* src_sobely,
                     uint8_t* dst_argb, int width) = SobelXYRow_C;
#if defined(HAS_SOBELXYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SobelXYRow = SobelXYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      SobelXYRow = SobelXYRow_SSE2;
    }
  }
#endif
#if defined(HAS_SOBELXYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SobelXYRow = SobelXYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      SobelXYRow = SobelXYRow_NEON;
    }
  }
#endif
#if defined(HAS_SOBELXYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    SobelXYRow = SobelXYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      SobelXYRow = SobelXYRow_MSA;
    }
  }
#endif
  return ARGBSobelize(src_argb, src_stride_argb, dst_argb, dst_stride_argb,
                      width, height, SobelXYRow);
}

// Apply a 4x4 polynomial to each ARGB pixel.
LIBYUV_API
int ARGBPolynomial(const uint8_t* src_argb,
                   int src_stride_argb,
                   uint8_t* dst_argb,
                   int dst_stride_argb,
                   const float* poly,
                   int width,
                   int height) {
  int y;
  void (*ARGBPolynomialRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                            const float* poly, int width) = ARGBPolynomialRow_C;
  if (!src_argb || !dst_argb || !poly || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBPOLYNOMIALROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(width, 2)) {
    ARGBPolynomialRow = ARGBPolynomialRow_SSE2;
  }
#endif
#if defined(HAS_ARGBPOLYNOMIALROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2) && TestCpuFlag(kCpuHasFMA3) &&
      IS_ALIGNED(width, 2)) {
    ARGBPolynomialRow = ARGBPolynomialRow_AVX2;
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBPolynomialRow(src_argb, dst_argb, poly, width);
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert plane of 16 bit shorts to half floats.
// Source values are multiplied by scale before storing as half float.
LIBYUV_API
int HalfFloatPlane(const uint16_t* src_y,
                   int src_stride_y,
                   uint16_t* dst_y,
                   int dst_stride_y,
                   float scale,
                   int width,
                   int height) {
  int y;
  void (*HalfFloatRow)(const uint16_t* src, uint16_t* dst, float scale,
                       int width) = HalfFloatRow_C;
  if (!src_y || !dst_y || width <= 0 || height == 0) {
    return -1;
  }
  src_stride_y >>= 1;
  dst_stride_y >>= 1;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_stride_y = -src_stride_y;
  }
  // Coalesce rows.
  if (src_stride_y == width && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_y = 0;
  }
#if defined(HAS_HALFFLOATROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    HalfFloatRow = HalfFloatRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      HalfFloatRow = HalfFloatRow_SSE2;
    }
  }
#endif
#if defined(HAS_HALFFLOATROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    HalfFloatRow = HalfFloatRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      HalfFloatRow = HalfFloatRow_AVX2;
    }
  }
#endif
#if defined(HAS_HALFFLOATROW_F16C)
  if (TestCpuFlag(kCpuHasAVX2) && TestCpuFlag(kCpuHasF16C)) {
    HalfFloatRow =
        (scale == 1.0f) ? HalfFloat1Row_Any_F16C : HalfFloatRow_Any_F16C;
    if (IS_ALIGNED(width, 16)) {
      HalfFloatRow = (scale == 1.0f) ? HalfFloat1Row_F16C : HalfFloatRow_F16C;
    }
  }
#endif
#if defined(HAS_HALFFLOATROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    HalfFloatRow =
        (scale == 1.0f) ? HalfFloat1Row_Any_NEON : HalfFloatRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      HalfFloatRow = (scale == 1.0f) ? HalfFloat1Row_NEON : HalfFloatRow_NEON;
    }
  }
#endif
#if defined(HAS_HALFFLOATROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    HalfFloatRow = HalfFloatRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      HalfFloatRow = HalfFloatRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    HalfFloatRow(src_y, dst_y, scale, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
  return 0;
}

// Convert a buffer of bytes to floats, scale the values and store as floats.
LIBYUV_API
int ByteToFloat(const uint8_t* src_y, float* dst_y, float scale, int width) {
  void (*ByteToFloatRow)(const uint8_t* src, float* dst, float scale,
                         int width) = ByteToFloatRow_C;
  if (!src_y || !dst_y || width <= 0) {
    return -1;
  }
#if defined(HAS_BYTETOFLOATROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ByteToFloatRow = ByteToFloatRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ByteToFloatRow = ByteToFloatRow_NEON;
    }
  }
#endif

  ByteToFloatRow(src_y, dst_y, scale, width);
  return 0;
}

// Apply a lumacolortable to each ARGB pixel.
LIBYUV_API
int ARGBLumaColorTable(const uint8_t* src_argb,
                       int src_stride_argb,
                       uint8_t* dst_argb,
                       int dst_stride_argb,
                       const uint8_t* luma,
                       int width,
                       int height) {
  int y;
  void (*ARGBLumaColorTableRow)(
      const uint8_t* src_argb, uint8_t* dst_argb, int width,
      const uint8_t* luma, const uint32_t lumacoeff) = ARGBLumaColorTableRow_C;
  if (!src_argb || !dst_argb || !luma || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBLUMACOLORTABLEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && IS_ALIGNED(width, 4)) {
    ARGBLumaColorTableRow = ARGBLumaColorTableRow_SSSE3;
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBLumaColorTableRow(src_argb, dst_argb, width, luma, 0x00264b0f);
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Copy Alpha from one ARGB image to another.
LIBYUV_API
int ARGBCopyAlpha(const uint8_t* src_argb,
                  int src_stride_argb,
                  uint8_t* dst_argb,
                  int dst_stride_argb,
                  int width,
                  int height) {
  int y;
  void (*ARGBCopyAlphaRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBCopyAlphaRow_C;
  if (!src_argb || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBCOPYALPHAROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBCopyAlphaRow = ARGBCopyAlphaRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      ARGBCopyAlphaRow = ARGBCopyAlphaRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBCOPYALPHAROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBCopyAlphaRow = ARGBCopyAlphaRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      ARGBCopyAlphaRow = ARGBCopyAlphaRow_AVX2;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBCopyAlphaRow(src_argb, dst_argb, width);
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Extract just the alpha channel from ARGB.
LIBYUV_API
int ARGBExtractAlpha(const uint8_t* src_argb,
                     int src_stride_argb,
                     uint8_t* dst_a,
                     int dst_stride_a,
                     int width,
                     int height) {
  if (!src_argb || !dst_a || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb += (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_a == width) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_a = 0;
  }
  void (*ARGBExtractAlphaRow)(const uint8_t* src_argb, uint8_t* dst_a,
                              int width) = ARGBExtractAlphaRow_C;
#if defined(HAS_ARGBEXTRACTALPHAROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 8) ? ARGBExtractAlphaRow_SSE2
                                               : ARGBExtractAlphaRow_Any_SSE2;
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 32) ? ARGBExtractAlphaRow_AVX2
                                                : ARGBExtractAlphaRow_Any_AVX2;
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 16) ? ARGBExtractAlphaRow_NEON
                                                : ARGBExtractAlphaRow_Any_NEON;
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 16) ? ARGBExtractAlphaRow_MSA
                                                : ARGBExtractAlphaRow_Any_MSA;
  }
#endif

  for (int y = 0; y < height; ++y) {
    ARGBExtractAlphaRow(src_argb, dst_a, width);
    src_argb += src_stride_argb;
    dst_a += dst_stride_a;
  }
  return 0;
}

// Copy a planar Y channel to the alpha channel of a destination ARGB image.
LIBYUV_API
int ARGBCopyYToAlpha(const uint8_t* src_y,
                     int src_stride_y,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     int width,
                     int height) {
  int y;
  void (*ARGBCopyYToAlphaRow)(const uint8_t* src_y, uint8_t* dst_argb,
                              int width) = ARGBCopyYToAlphaRow_C;
  if (!src_y || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_stride_y = -src_stride_y;
  }
  // Coalesce rows.
  if (src_stride_y == width && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_argb = 0;
  }
#if defined(HAS_ARGBCOPYYTOALPHAROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBCopyYToAlphaRow = ARGBCopyYToAlphaRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      ARGBCopyYToAlphaRow = ARGBCopyYToAlphaRow_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBCOPYYTOALPHAROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBCopyYToAlphaRow = ARGBCopyYToAlphaRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      ARGBCopyYToAlphaRow = ARGBCopyYToAlphaRow_AVX2;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBCopyYToAlphaRow(src_y, dst_argb, width);
    src_y += src_stride_y;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// TODO(fbarchard): Consider if width is even Y channel can be split
// directly. A SplitUVRow_Odd function could copy the remaining chroma.

LIBYUV_API
int YUY2ToNV12(const uint8_t* src_yuy2,
               int src_stride_yuy2,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int y;
  int halfwidth = (width + 1) >> 1;
  void (*SplitUVRow)(const uint8_t* src_uv, uint8_t* dst_u, uint8_t* dst_v,
                     int width) = SplitUVRow_C;
  void (*InterpolateRow)(uint8_t * dst_ptr, const uint8_t* src_ptr,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) = InterpolateRow_C;
  if (!src_yuy2 || !dst_y || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_yuy2 = src_yuy2 + (height - 1) * src_stride_yuy2;
    src_stride_yuy2 = -src_stride_yuy2;
  }
#if defined(HAS_SPLITUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SplitUVRow = SplitUVRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      SplitUVRow = SplitUVRow_SSE2;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    SplitUVRow = SplitUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      SplitUVRow = SplitUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SplitUVRow = SplitUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      SplitUVRow = SplitUVRow_NEON;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    SplitUVRow = SplitUVRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      SplitUVRow = SplitUVRow_MSA;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      InterpolateRow = InterpolateRow_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    InterpolateRow = InterpolateRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_MSA;
    }
  }
#endif

  {
    int awidth = halfwidth * 2;
    // row of y and 2 rows of uv
    align_buffer_64(rows, awidth * 3);

    for (y = 0; y < height - 1; y += 2) {
      // Split Y from UV.
      SplitUVRow(src_yuy2, rows, rows + awidth, awidth);
      memcpy(dst_y, rows, width);
      SplitUVRow(src_yuy2 + src_stride_yuy2, rows, rows + awidth * 2, awidth);
      memcpy(dst_y + dst_stride_y, rows, width);
      InterpolateRow(dst_uv, rows + awidth, awidth, awidth, 128);
      src_yuy2 += src_stride_yuy2 * 2;
      dst_y += dst_stride_y * 2;
      dst_uv += dst_stride_uv;
    }
    if (height & 1) {
      // Split Y from UV.
      SplitUVRow(src_yuy2, rows, dst_uv, awidth);
      memcpy(dst_y, rows, width);
    }
    free_aligned_buffer_64(rows);
  }
  return 0;
}

LIBYUV_API
int UYVYToNV12(const uint8_t* src_uyvy,
               int src_stride_uyvy,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int y;
  int halfwidth = (width + 1) >> 1;
  void (*SplitUVRow)(const uint8_t* src_uv, uint8_t* dst_u, uint8_t* dst_v,
                     int width) = SplitUVRow_C;
  void (*InterpolateRow)(uint8_t * dst_ptr, const uint8_t* src_ptr,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) = InterpolateRow_C;
  if (!src_uyvy || !dst_y || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uyvy = src_uyvy + (height - 1) * src_stride_uyvy;
    src_stride_uyvy = -src_stride_uyvy;
  }
#if defined(HAS_SPLITUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SplitUVRow = SplitUVRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      SplitUVRow = SplitUVRow_SSE2;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    SplitUVRow = SplitUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      SplitUVRow = SplitUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SplitUVRow = SplitUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      SplitUVRow = SplitUVRow_NEON;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    SplitUVRow = SplitUVRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      SplitUVRow = SplitUVRow_MSA;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      InterpolateRow = InterpolateRow_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    InterpolateRow = InterpolateRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_MSA;
    }
  }
#endif

  {
    int awidth = halfwidth * 2;
    // row of y and 2 rows of uv
    align_buffer_64(rows, awidth * 3);

    for (y = 0; y < height - 1; y += 2) {
      // Split Y from UV.
      SplitUVRow(src_uyvy, rows + awidth, rows, awidth);
      memcpy(dst_y, rows, width);
      SplitUVRow(src_uyvy + src_stride_uyvy, rows + awidth * 2, rows, awidth);
      memcpy(dst_y + dst_stride_y, rows, width);
      InterpolateRow(dst_uv, rows + awidth, awidth, awidth, 128);
      src_uyvy += src_stride_uyvy * 2;
      dst_y += dst_stride_y * 2;
      dst_uv += dst_stride_uv;
    }
    if (height & 1) {
      // Split Y from UV.
      SplitUVRow(src_uyvy, dst_uv, rows, awidth);
      memcpy(dst_y, rows, width);
    }
    free_aligned_buffer_64(rows);
  }
  return 0;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
