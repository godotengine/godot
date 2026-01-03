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

#include <assert.h>
#include <string.h>  // for memset()

#include "libyuv/cpu_id.h"
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
  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_COPYROW_AVX512BW)
  if (TestCpuFlag(kCpuHasAVX512BW)) {
    CopyRow = IS_ALIGNED(width, 128) ? CopyRow_AVX512BW : CopyRow_Any_AVX512BW;
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
#if defined(HAS_COPYROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    CopyRow = CopyRow_SME;
  }
#endif
#if defined(HAS_COPYROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    CopyRow = CopyRow_RVV;
  }
#endif

  // Copy plane
  for (y = 0; y < height; ++y) {
    CopyRow(src_y, dst_y, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
}

LIBYUV_API
void CopyPlane_16(const uint16_t* src_y,
                  int src_stride_y,
                  uint16_t* dst_y,
                  int dst_stride_y,
                  int width,
                  int height) {
  CopyPlane((const uint8_t*)src_y, src_stride_y * 2, (uint8_t*)dst_y,
            dst_stride_y * 2, width * 2, height);
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

  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_CONVERT16TO8ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Convert16To8Row = Convert16To8Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      Convert16To8Row = Convert16To8Row_NEON;
    }
  }
#endif
#if defined(HAS_CONVERT16TO8ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    Convert16To8Row = Convert16To8Row_SME;
  }
#endif
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
#if defined(HAS_CONVERT16TO8ROW_AVX512BW)
  if (TestCpuFlag(kCpuHasAVX512BW)) {
    Convert16To8Row = Convert16To8Row_Any_AVX512BW;
    if (IS_ALIGNED(width, 64)) {
      Convert16To8Row = Convert16To8Row_AVX512BW;
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
                       int scale,  // 1024 for 10 bits
                       int width,
                       int height) {
  int y;
  void (*Convert8To16Row)(const uint8_t* src_y, uint16_t* dst_y, int scale,
                          int width) = Convert8To16Row_C;

  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_CONVERT8TO16ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Convert8To16Row = Convert8To16Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      Convert8To16Row = Convert8To16Row_NEON;
    }
  }
#endif
#if defined(HAS_CONVERT8TO16ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    Convert8To16Row = Convert8To16Row_SME;
  }
#endif

  // Convert plane
  for (y = 0; y < height; ++y) {
    Convert8To16Row(src_y, dst_y, scale, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
}

// Convert a plane of 8 bit data to 8 bit
LIBYUV_API
void Convert8To8Plane(const uint8_t* src_y,
                      int src_stride_y,
                      uint8_t* dst_y,
                      int dst_stride_y,
                      int scale,  // 220 for Y, 225 to UV
                      int bias,   // 16
                      int width,
                      int height) {
  int y;
  void (*Convert8To8Row)(const uint8_t* src_y, uint8_t* dst_y, int scale,
                         int bias, int width) = Convert8To8Row_C;

  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_CONVERT8TO8ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Convert8To8Row = Convert8To8Row_Any_NEON;
    if (IS_ALIGNED(width, 32)) {
      Convert8To8Row = Convert8To8Row_NEON;
    }
  }
#endif
#if defined(HAS_CONVERT8TO8ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    Convert8To8Row = Convert8To8Row_SVE2;
  }
#endif
#if defined(HAS_CONVERT8TO8ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    Convert8To8Row = Convert8To8Row_SME;
  }
#endif
#if defined(HAS_CONVERT8TO8ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Convert8To8Row = Convert8To8Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      Convert8To8Row = Convert8To8Row_AVX2;
    }
  }
#endif

  // Convert plane
  for (y = 0; y < height; ++y) {
    Convert8To8Row(src_y, dst_y, scale, bias, width);
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

  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
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
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
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

// Copy I210.
LIBYUV_API
int I210Copy(const uint16_t* src_y,
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

  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
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
    CopyPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  // Copy UV planes.
  CopyPlane_16(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, height);
  CopyPlane_16(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, height);
  return 0;
}

// Copy I410.
LIBYUV_API
int I410Copy(const uint16_t* src_y,
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
  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
      height == 0) {
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
    CopyPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  CopyPlane_16(src_u, src_stride_u, dst_u, dst_stride_u, width, height);
  CopyPlane_16(src_v, src_stride_v, dst_v, dst_stride_v, width, height);
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

// Copy NV12. Supports inverting.
LIBYUV_API
int NV12Copy(const uint8_t* src_y,
             int src_stride_y,
             const uint8_t* src_uv,
             int src_stride_uv,
             uint8_t* dst_y,
             int dst_stride_y,
             uint8_t* dst_uv,
             int dst_stride_uv,
             int width,
             int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;

  if (!src_y || !dst_y || !src_uv || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_uv = src_uv + (halfheight - 1) * src_stride_uv;
    src_stride_y = -src_stride_y;
    src_stride_uv = -src_stride_uv;
  }
  CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  CopyPlane(src_uv, src_stride_uv, dst_uv, dst_stride_uv, halfwidth * 2,
            halfheight);
  return 0;
}

// Copy NV21. Supports inverting.
LIBYUV_API
int NV21Copy(const uint8_t* src_y,
             int src_stride_y,
             const uint8_t* src_vu,
             int src_stride_vu,
             uint8_t* dst_y,
             int dst_stride_y,
             uint8_t* dst_vu,
             int dst_stride_vu,
             int width,
             int height) {
  return NV12Copy(src_y, src_stride_y, src_vu, src_stride_vu, dst_y,
                  dst_stride_y, dst_vu, dst_stride_vu, width, height);
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
  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_SPLITUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    SplitUVRow = SplitUVRow_Any_LSX;
    if (IS_ALIGNED(width, 32)) {
      SplitUVRow = SplitUVRow_LSX;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    SplitUVRow = SplitUVRow_RVV;
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
  if (width <= 0 || height == 0) {
    return;
  }
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
    if (IS_ALIGNED(width, 16)) {
      MergeUVRow = MergeUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_AVX512BW)
  if (TestCpuFlag(kCpuHasAVX512BW)) {
    MergeUVRow = MergeUVRow_Any_AVX512BW;
    if (IS_ALIGNED(width, 32)) {
      MergeUVRow = MergeUVRow_AVX512BW;
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
#if defined(HAS_MERGEUVROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    MergeUVRow = MergeUVRow_SME;
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
#if defined(HAS_MERGEUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    MergeUVRow = MergeUVRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      MergeUVRow = MergeUVRow_LSX;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    MergeUVRow = MergeUVRow_RVV;
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

// Support function for P010 etc UV channels.
// Width and height are plane sizes (typically half pixel width).
LIBYUV_API
void SplitUVPlane_16(const uint16_t* src_uv,
                     int src_stride_uv,
                     uint16_t* dst_u,
                     int dst_stride_u,
                     uint16_t* dst_v,
                     int dst_stride_v,
                     int width,
                     int height,
                     int depth) {
  int y;
  void (*SplitUVRow_16)(const uint16_t* src_uv, uint16_t* dst_u,
                        uint16_t* dst_v, int depth, int width) =
      SplitUVRow_16_C;
  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_SPLITUVROW_16_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    SplitUVRow_16 = SplitUVRow_16_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      SplitUVRow_16 = SplitUVRow_16_AVX2;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_16_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SplitUVRow_16 = SplitUVRow_16_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      SplitUVRow_16 = SplitUVRow_16_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    // Copy a row of UV.
    SplitUVRow_16(src_uv, dst_u, dst_v, depth, width);
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
    src_uv += src_stride_uv;
  }
}

LIBYUV_API
void MergeUVPlane_16(const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint16_t* dst_uv,
                     int dst_stride_uv,
                     int width,
                     int height,
                     int depth) {
  int y;
  void (*MergeUVRow_16)(const uint16_t* src_u, const uint16_t* src_v,
                        uint16_t* dst_uv, int depth, int width) =
      MergeUVRow_16_C;
  assert(depth >= 8);
  assert(depth <= 16);
  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_MERGEUVROW_16_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeUVRow_16 = MergeUVRow_16_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      MergeUVRow_16 = MergeUVRow_16_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_16_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeUVRow_16 = MergeUVRow_16_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      MergeUVRow_16 = MergeUVRow_16_NEON;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_16_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    MergeUVRow_16 = MergeUVRow_16_SME;
  }
#endif

  for (y = 0; y < height; ++y) {
    // Merge a row of U and V into a row of UV.
    MergeUVRow_16(src_u, src_v, dst_uv, depth, width);
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_uv += dst_stride_uv;
  }
}

// Convert plane from lsb to msb
LIBYUV_API
void ConvertToMSBPlane_16(const uint16_t* src_y,
                          int src_stride_y,
                          uint16_t* dst_y,
                          int dst_stride_y,
                          int width,
                          int height,
                          int depth) {
  int y;
  int scale = 1 << (16 - depth);
  void (*MultiplyRow_16)(const uint16_t* src_y, uint16_t* dst_y, int scale,
                         int width) = MultiplyRow_16_C;
  if (width <= 0 || height == 0) {
    return;
  }
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

#if defined(HAS_MULTIPLYROW_16_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MultiplyRow_16 = MultiplyRow_16_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      MultiplyRow_16 = MultiplyRow_16_AVX2;
    }
  }
#endif
#if defined(HAS_MULTIPLYROW_16_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MultiplyRow_16 = MultiplyRow_16_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      MultiplyRow_16 = MultiplyRow_16_NEON;
    }
  }
#endif
#if defined(HAS_MULTIPLYROW_16_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    MultiplyRow_16 = MultiplyRow_16_SME;
  }
#endif

  for (y = 0; y < height; ++y) {
    MultiplyRow_16(src_y, dst_y, scale, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
}

// Convert plane from msb to lsb
LIBYUV_API
void ConvertToLSBPlane_16(const uint16_t* src_y,
                          int src_stride_y,
                          uint16_t* dst_y,
                          int dst_stride_y,
                          int width,
                          int height,
                          int depth) {
  int y;
  int scale = 1 << depth;
  void (*DivideRow)(const uint16_t* src_y, uint16_t* dst_y, int scale,
                    int width) = DivideRow_16_C;
  if (width <= 0 || height == 0) {
    return;
  }
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

#if defined(HAS_DIVIDEROW_16_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    DivideRow = DivideRow_16_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      DivideRow = DivideRow_16_AVX2;
    }
  }
#endif
#if defined(HAS_DIVIDEROW_16_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    DivideRow = DivideRow_16_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      DivideRow = DivideRow_16_NEON;
    }
  }
#endif
#if defined(HAS_DIVIDEROW_16_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    DivideRow = DivideRow_16_SVE2;
  }
#endif

  for (y = 0; y < height; ++y) {
    DivideRow(src_y, dst_y, scale, width);
    src_y += src_stride_y;
    dst_y += dst_stride_y;
  }
}

// Swap U and V channels in interleaved UV plane.
LIBYUV_API
void SwapUVPlane(const uint8_t* src_uv,
                 int src_stride_uv,
                 uint8_t* dst_vu,
                 int dst_stride_vu,
                 int width,
                 int height) {
  int y;
  void (*SwapUVRow)(const uint8_t* src_uv, uint8_t* dst_vu, int width) =
      SwapUVRow_C;
  if (width <= 0 || height == 0) {
    return;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uv = src_uv + (height - 1) * src_stride_uv;
    src_stride_uv = -src_stride_uv;
  }
  // Coalesce rows.
  if (src_stride_uv == width * 2 && dst_stride_vu == width * 2) {
    width *= height;
    height = 1;
    src_stride_uv = dst_stride_vu = 0;
  }

#if defined(HAS_SWAPUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    SwapUVRow = SwapUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      SwapUVRow = SwapUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_SWAPUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    SwapUVRow = SwapUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      SwapUVRow = SwapUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_SWAPUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SwapUVRow = SwapUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      SwapUVRow = SwapUVRow_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    SwapUVRow(src_uv, dst_vu, width);
    src_uv += src_stride_uv;
    dst_vu += dst_stride_vu;
  }
}

// Convert NV21 to NV12.
LIBYUV_API
int NV21ToNV12(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_vu,
               int src_stride_vu,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;

  if (!src_vu || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }

  if (dst_y) {
    CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_vu = src_vu + (halfheight - 1) * src_stride_vu;
    src_stride_vu = -src_stride_vu;
  }

  SwapUVPlane(src_vu, src_stride_vu, dst_uv, dst_stride_uv, halfwidth,
              halfheight);
  return 0;
}

// Test if tile_height is a power of 2 (16 or 32)
#define IS_POWEROFTWO(x) (!((x) & ((x)-1)))

// Detile a plane of data
// tile width is 16 and assumed.
// tile_height is 16 or 32 for MM21.
// src_stride_y is bytes per row of source ignoring tiling. e.g. 640
// TODO: More detile row functions.
LIBYUV_API
int DetilePlane(const uint8_t* src_y,
                int src_stride_y,
                uint8_t* dst_y,
                int dst_stride_y,
                int width,
                int height,
                int tile_height) {
  const ptrdiff_t src_tile_stride = 16 * tile_height;
  int y;
  void (*DetileRow)(const uint8_t* src, ptrdiff_t src_tile_stride, uint8_t* dst,
                    int width) = DetileRow_C;
  if (!src_y || !dst_y || width <= 0 || height == 0 ||
      !IS_POWEROFTWO(tile_height)) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_stride_y = -dst_stride_y;
  }

#if defined(HAS_DETILEROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    DetileRow = DetileRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      DetileRow = DetileRow_SSE2;
    }
  }
#endif
#if defined(HAS_DETILEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    DetileRow = DetileRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      DetileRow = DetileRow_NEON;
    }
  }
#endif

  // Detile plane
  for (y = 0; y < height; ++y) {
    DetileRow(src_y, src_tile_stride, dst_y, width);
    dst_y += dst_stride_y;
    src_y += 16;
    // Advance to next row of tiles.
    if ((y & (tile_height - 1)) == (tile_height - 1)) {
      src_y = src_y - src_tile_stride + src_stride_y * tile_height;
    }
  }
  return 0;
}

// Convert a plane of 16 bit tiles of 16 x H to linear.
// tile width is 16 and assumed.
// tile_height is 16 or 32 for MT2T.
LIBYUV_API
int DetilePlane_16(const uint16_t* src_y,
                   int src_stride_y,
                   uint16_t* dst_y,
                   int dst_stride_y,
                   int width,
                   int height,
                   int tile_height) {
  const ptrdiff_t src_tile_stride = 16 * tile_height;
  int y;
  void (*DetileRow_16)(const uint16_t* src, ptrdiff_t src_tile_stride,
                       uint16_t* dst, int width) = DetileRow_16_C;
  if (!src_y || !dst_y || width <= 0 || height == 0 ||
      !IS_POWEROFTWO(tile_height)) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_stride_y = -dst_stride_y;
  }

#if defined(HAS_DETILEROW_16_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    DetileRow_16 = DetileRow_16_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      DetileRow_16 = DetileRow_16_SSE2;
    }
  }
#endif
#if defined(HAS_DETILEROW_16_AVX)
  if (TestCpuFlag(kCpuHasAVX)) {
    DetileRow_16 = DetileRow_16_Any_AVX;
    if (IS_ALIGNED(width, 16)) {
      DetileRow_16 = DetileRow_16_AVX;
    }
  }
#endif
#if defined(HAS_DETILEROW_16_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    DetileRow_16 = DetileRow_16_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      DetileRow_16 = DetileRow_16_NEON;
    }
  }
#endif

  // Detile plane
  for (y = 0; y < height; ++y) {
    DetileRow_16(src_y, src_tile_stride, dst_y, width);
    dst_y += dst_stride_y;
    src_y += 16;
    // Advance to next row of tiles.
    if ((y & (tile_height - 1)) == (tile_height - 1)) {
      src_y = src_y - src_tile_stride + src_stride_y * tile_height;
    }
  }
  return 0;
}

LIBYUV_API
void DetileSplitUVPlane(const uint8_t* src_uv,
                        int src_stride_uv,
                        uint8_t* dst_u,
                        int dst_stride_u,
                        uint8_t* dst_v,
                        int dst_stride_v,
                        int width,
                        int height,
                        int tile_height) {
  const ptrdiff_t src_tile_stride = 16 * tile_height;
  int y;
  void (*DetileSplitUVRow)(const uint8_t* src, ptrdiff_t src_tile_stride,
                           uint8_t* dst_u, uint8_t* dst_v, int width) =
      DetileSplitUVRow_C;
  assert(src_stride_uv >= 0);
  assert(tile_height > 0);
  assert(src_stride_uv > 0);

  if (width <= 0 || height == 0) {
    return;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_u = dst_u + (height - 1) * dst_stride_u;
    dst_stride_u = -dst_stride_u;
    dst_v = dst_v + (height - 1) * dst_stride_v;
    dst_stride_v = -dst_stride_v;
  }

#if defined(HAS_DETILESPLITUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    DetileSplitUVRow = DetileSplitUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      DetileSplitUVRow = DetileSplitUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_DETILESPLITUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    DetileSplitUVRow = DetileSplitUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      DetileSplitUVRow = DetileSplitUVRow_NEON;
    }
  }
#endif

  // Detile plane
  for (y = 0; y < height; ++y) {
    DetileSplitUVRow(src_uv, src_tile_stride, dst_u, dst_v, width);
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
    src_uv += 16;
    // Advance to next row of tiles.
    if ((y & (tile_height - 1)) == (tile_height - 1)) {
      src_uv = src_uv - src_tile_stride + src_stride_uv * tile_height;
    }
  }
}

LIBYUV_API
void DetileToYUY2(const uint8_t* src_y,
                  int src_stride_y,
                  const uint8_t* src_uv,
                  int src_stride_uv,
                  uint8_t* dst_yuy2,
                  int dst_stride_yuy2,
                  int width,
                  int height,
                  int tile_height) {
  const ptrdiff_t src_y_tile_stride = 16 * tile_height;
  const ptrdiff_t src_uv_tile_stride = src_y_tile_stride / 2;
  int y;
  void (*DetileToYUY2)(const uint8_t* src_y, ptrdiff_t src_y_tile_stride,
                       const uint8_t* src_uv, ptrdiff_t src_uv_tile_stride,
                       uint8_t* dst_yuy2, int width) = DetileToYUY2_C;
  assert(src_stride_y >= 0);
  assert(src_stride_y > 0);
  assert(src_stride_uv >= 0);
  assert(src_stride_uv > 0);
  assert(tile_height > 0);

  if (width <= 0 || height == 0 || tile_height <= 0) {
    return;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_yuy2 = dst_yuy2 + (height - 1) * dst_stride_yuy2;
    dst_stride_yuy2 = -dst_stride_yuy2;
  }

#if defined(HAS_DETILETOYUY2_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    DetileToYUY2 = DetileToYUY2_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      DetileToYUY2 = DetileToYUY2_NEON;
    }
  }
#endif

#if defined(HAS_DETILETOYUY2_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    DetileToYUY2 = DetileToYUY2_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      DetileToYUY2 = DetileToYUY2_SSE2;
    }
  }
#endif

  // Detile plane
  for (y = 0; y < height; ++y) {
    DetileToYUY2(src_y, src_y_tile_stride, src_uv, src_uv_tile_stride, dst_yuy2,
                 width);
    dst_yuy2 += dst_stride_yuy2;
    src_y += 16;

    if (y & 0x1)
      src_uv += 16;

    // Advance to next row of tiles.
    if ((y & (tile_height - 1)) == (tile_height - 1)) {
      src_y = src_y - src_y_tile_stride + src_stride_y * tile_height;
      src_uv = src_uv - src_uv_tile_stride + src_stride_uv * (tile_height / 2);
    }
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
  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_SPLITRGBROW_SSE41)
  if (TestCpuFlag(kCpuHasSSE41)) {
    SplitRGBRow = SplitRGBRow_Any_SSE41;
    if (IS_ALIGNED(width, 16)) {
      SplitRGBRow = SplitRGBRow_SSE41;
    }
  }
#endif
#if defined(HAS_SPLITRGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    SplitRGBRow = SplitRGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      SplitRGBRow = SplitRGBRow_AVX2;
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
#if defined(HAS_SPLITRGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    SplitRGBRow = SplitRGBRow_RVV;
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
  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_MERGERGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    MergeRGBRow = MergeRGBRow_RVV;
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

LIBYUV_NOINLINE
static void SplitARGBPlaneAlpha(const uint8_t* src_argb,
                                int src_stride_argb,
                                uint8_t* dst_r,
                                int dst_stride_r,
                                uint8_t* dst_g,
                                int dst_stride_g,
                                uint8_t* dst_b,
                                int dst_stride_b,
                                uint8_t* dst_a,
                                int dst_stride_a,
                                int width,
                                int height) {
  int y;
  void (*SplitARGBRow)(const uint8_t* src_rgb, uint8_t* dst_r, uint8_t* dst_g,
                       uint8_t* dst_b, uint8_t* dst_a, int width) =
      SplitARGBRow_C;

  assert(height > 0);

  if (width <= 0 || height == 0) {
    return;
  }
  if (src_stride_argb == width * 4 && dst_stride_r == width &&
      dst_stride_g == width && dst_stride_b == width && dst_stride_a == width) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_r = dst_stride_g = dst_stride_b =
        dst_stride_a = 0;
  }

#if defined(HAS_SPLITARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SplitARGBRow = SplitARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      SplitARGBRow = SplitARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_SPLITARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    SplitARGBRow = SplitARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      SplitARGBRow = SplitARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_SPLITARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    SplitARGBRow = SplitARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      SplitARGBRow = SplitARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_SPLITARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SplitARGBRow = SplitARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      SplitARGBRow = SplitARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_SPLITARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    SplitARGBRow = SplitARGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    SplitARGBRow(src_argb, dst_r, dst_g, dst_b, dst_a, width);
    dst_r += dst_stride_r;
    dst_g += dst_stride_g;
    dst_b += dst_stride_b;
    dst_a += dst_stride_a;
    src_argb += src_stride_argb;
  }
}

LIBYUV_NOINLINE
static void SplitARGBPlaneOpaque(const uint8_t* src_argb,
                                 int src_stride_argb,
                                 uint8_t* dst_r,
                                 int dst_stride_r,
                                 uint8_t* dst_g,
                                 int dst_stride_g,
                                 uint8_t* dst_b,
                                 int dst_stride_b,
                                 int width,
                                 int height) {
  int y;
  void (*SplitXRGBRow)(const uint8_t* src_rgb, uint8_t* dst_r, uint8_t* dst_g,
                       uint8_t* dst_b, int width) = SplitXRGBRow_C;
  assert(height > 0);

  if (width <= 0 || height == 0) {
    return;
  }
  if (src_stride_argb == width * 4 && dst_stride_r == width &&
      dst_stride_g == width && dst_stride_b == width) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_r = dst_stride_g = dst_stride_b = 0;
  }

#if defined(HAS_SPLITXRGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    SplitXRGBRow = SplitXRGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      SplitXRGBRow = SplitXRGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_SPLITXRGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    SplitXRGBRow = SplitXRGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      SplitXRGBRow = SplitXRGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_SPLITXRGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    SplitXRGBRow = SplitXRGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      SplitXRGBRow = SplitXRGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_SPLITXRGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    SplitXRGBRow = SplitXRGBRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      SplitXRGBRow = SplitXRGBRow_NEON;
    }
  }
#endif
#if defined(HAS_SPLITXRGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    SplitXRGBRow = SplitXRGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    SplitXRGBRow(src_argb, dst_r, dst_g, dst_b, width);
    dst_r += dst_stride_r;
    dst_g += dst_stride_g;
    dst_b += dst_stride_b;
    src_argb += src_stride_argb;
  }
}

LIBYUV_API
void SplitARGBPlane(const uint8_t* src_argb,
                    int src_stride_argb,
                    uint8_t* dst_r,
                    int dst_stride_r,
                    uint8_t* dst_g,
                    int dst_stride_g,
                    uint8_t* dst_b,
                    int dst_stride_b,
                    uint8_t* dst_a,
                    int dst_stride_a,
                    int width,
                    int height) {
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_r = dst_r + (height - 1) * dst_stride_r;
    dst_g = dst_g + (height - 1) * dst_stride_g;
    dst_b = dst_b + (height - 1) * dst_stride_b;
    dst_a = dst_a + (height - 1) * dst_stride_a;
    dst_stride_r = -dst_stride_r;
    dst_stride_g = -dst_stride_g;
    dst_stride_b = -dst_stride_b;
    dst_stride_a = -dst_stride_a;
  }

  if (dst_a == NULL) {
    SplitARGBPlaneOpaque(src_argb, src_stride_argb, dst_r, dst_stride_r, dst_g,
                         dst_stride_g, dst_b, dst_stride_b, width, height);
  } else {
    SplitARGBPlaneAlpha(src_argb, src_stride_argb, dst_r, dst_stride_r, dst_g,
                        dst_stride_g, dst_b, dst_stride_b, dst_a, dst_stride_a,
                        width, height);
  }
}

LIBYUV_NOINLINE
static void MergeARGBPlaneAlpha(const uint8_t* src_r,
                                int src_stride_r,
                                const uint8_t* src_g,
                                int src_stride_g,
                                const uint8_t* src_b,
                                int src_stride_b,
                                const uint8_t* src_a,
                                int src_stride_a,
                                uint8_t* dst_argb,
                                int dst_stride_argb,
                                int width,
                                int height) {
  int y;
  void (*MergeARGBRow)(const uint8_t* src_r, const uint8_t* src_g,
                       const uint8_t* src_b, const uint8_t* src_a,
                       uint8_t* dst_argb, int width) = MergeARGBRow_C;

  assert(height > 0);

  if (width <= 0 || height == 0) {
    return;
  }
  if (src_stride_r == width && src_stride_g == width && src_stride_b == width &&
      src_stride_a == width && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_r = src_stride_g = src_stride_b = src_stride_a =
        dst_stride_argb = 0;
  }
#if defined(HAS_MERGEARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    MergeARGBRow = MergeARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      MergeARGBRow = MergeARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_MERGEARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeARGBRow = MergeARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      MergeARGBRow = MergeARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeARGBRow = MergeARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      MergeARGBRow = MergeARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_MERGEARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    MergeARGBRow = MergeARGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    MergeARGBRow(src_r, src_g, src_b, src_a, dst_argb, width);
    src_r += src_stride_r;
    src_g += src_stride_g;
    src_b += src_stride_b;
    src_a += src_stride_a;
    dst_argb += dst_stride_argb;
  }
}

LIBYUV_NOINLINE
static void MergeARGBPlaneOpaque(const uint8_t* src_r,
                                 int src_stride_r,
                                 const uint8_t* src_g,
                                 int src_stride_g,
                                 const uint8_t* src_b,
                                 int src_stride_b,
                                 uint8_t* dst_argb,
                                 int dst_stride_argb,
                                 int width,
                                 int height) {
  int y;
  void (*MergeXRGBRow)(const uint8_t* src_r, const uint8_t* src_g,
                       const uint8_t* src_b, uint8_t* dst_argb, int width) =
      MergeXRGBRow_C;

  assert(height > 0);

  if (width <= 0 || height == 0) {
    return;
  }
  if (src_stride_r == width && src_stride_g == width && src_stride_b == width &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_r = src_stride_g = src_stride_b = dst_stride_argb = 0;
  }
#if defined(HAS_MERGEXRGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    MergeXRGBRow = MergeXRGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      MergeXRGBRow = MergeXRGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_MERGEXRGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeXRGBRow = MergeXRGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      MergeXRGBRow = MergeXRGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEXRGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeXRGBRow = MergeXRGBRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      MergeXRGBRow = MergeXRGBRow_NEON;
    }
  }
#endif
#if defined(HAS_MERGEXRGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    MergeXRGBRow = MergeXRGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    MergeXRGBRow(src_r, src_g, src_b, dst_argb, width);
    src_r += src_stride_r;
    src_g += src_stride_g;
    src_b += src_stride_b;
    dst_argb += dst_stride_argb;
  }
}

LIBYUV_API
void MergeARGBPlane(const uint8_t* src_r,
                    int src_stride_r,
                    const uint8_t* src_g,
                    int src_stride_g,
                    const uint8_t* src_b,
                    int src_stride_b,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height) {
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }

  if (src_a == NULL) {
    MergeARGBPlaneOpaque(src_r, src_stride_r, src_g, src_stride_g, src_b,
                         src_stride_b, dst_argb, dst_stride_argb, width,
                         height);
  } else {
    MergeARGBPlaneAlpha(src_r, src_stride_r, src_g, src_stride_g, src_b,
                        src_stride_b, src_a, src_stride_a, dst_argb,
                        dst_stride_argb, width, height);
  }
}

// TODO(yuan): Support 2 bit alpha channel.
LIBYUV_API
void MergeXR30Plane(const uint16_t* src_r,
                    int src_stride_r,
                    const uint16_t* src_g,
                    int src_stride_g,
                    const uint16_t* src_b,
                    int src_stride_b,
                    uint8_t* dst_ar30,
                    int dst_stride_ar30,
                    int width,
                    int height,
                    int depth) {
  int y;
  void (*MergeXR30Row)(const uint16_t* src_r, const uint16_t* src_g,
                       const uint16_t* src_b, uint8_t* dst_ar30, int depth,
                       int width) = MergeXR30Row_C;

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
  // Coalesce rows.
  if (src_stride_r == width && src_stride_g == width && src_stride_b == width &&
      dst_stride_ar30 == width * 4) {
    width *= height;
    height = 1;
    src_stride_r = src_stride_g = src_stride_b = dst_stride_ar30 = 0;
  }
#if defined(HAS_MERGEXR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeXR30Row = MergeXR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      MergeXR30Row = MergeXR30Row_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEXR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    if (depth == 10) {
      MergeXR30Row = MergeXR30Row_10_Any_NEON;
      if (IS_ALIGNED(width, 8)) {
        MergeXR30Row = MergeXR30Row_10_NEON;
      }
    } else {
      MergeXR30Row = MergeXR30Row_Any_NEON;
      if (IS_ALIGNED(width, 8)) {
        MergeXR30Row = MergeXR30Row_NEON;
      }
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    MergeXR30Row(src_r, src_g, src_b, dst_ar30, depth, width);
    src_r += src_stride_r;
    src_g += src_stride_g;
    src_b += src_stride_b;
    dst_ar30 += dst_stride_ar30;
  }
}

LIBYUV_NOINLINE
static void MergeAR64PlaneAlpha(const uint16_t* src_r,
                                int src_stride_r,
                                const uint16_t* src_g,
                                int src_stride_g,
                                const uint16_t* src_b,
                                int src_stride_b,
                                const uint16_t* src_a,
                                int src_stride_a,
                                uint16_t* dst_ar64,
                                int dst_stride_ar64,
                                int width,
                                int height,
                                int depth) {
  int y;
  void (*MergeAR64Row)(const uint16_t* src_r, const uint16_t* src_g,
                       const uint16_t* src_b, const uint16_t* src_a,
                       uint16_t* dst_argb, int depth, int width) =
      MergeAR64Row_C;

  if (src_stride_r == width && src_stride_g == width && src_stride_b == width &&
      src_stride_a == width && dst_stride_ar64 == width * 4) {
    width *= height;
    height = 1;
    src_stride_r = src_stride_g = src_stride_b = src_stride_a =
        dst_stride_ar64 = 0;
  }
#if defined(HAS_MERGEAR64ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeAR64Row = MergeAR64Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      MergeAR64Row = MergeAR64Row_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEAR64ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeAR64Row = MergeAR64Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      MergeAR64Row = MergeAR64Row_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    MergeAR64Row(src_r, src_g, src_b, src_a, dst_ar64, depth, width);
    src_r += src_stride_r;
    src_g += src_stride_g;
    src_b += src_stride_b;
    src_a += src_stride_a;
    dst_ar64 += dst_stride_ar64;
  }
}

LIBYUV_NOINLINE
static void MergeAR64PlaneOpaque(const uint16_t* src_r,
                                 int src_stride_r,
                                 const uint16_t* src_g,
                                 int src_stride_g,
                                 const uint16_t* src_b,
                                 int src_stride_b,
                                 uint16_t* dst_ar64,
                                 int dst_stride_ar64,
                                 int width,
                                 int height,
                                 int depth) {
  int y;
  void (*MergeXR64Row)(const uint16_t* src_r, const uint16_t* src_g,
                       const uint16_t* src_b, uint16_t* dst_argb, int depth,
                       int width) = MergeXR64Row_C;

  // Coalesce rows.
  if (src_stride_r == width && src_stride_g == width && src_stride_b == width &&
      dst_stride_ar64 == width * 4) {
    width *= height;
    height = 1;
    src_stride_r = src_stride_g = src_stride_b = dst_stride_ar64 = 0;
  }
#if defined(HAS_MERGEXR64ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeXR64Row = MergeXR64Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      MergeXR64Row = MergeXR64Row_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEXR64ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeXR64Row = MergeXR64Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      MergeXR64Row = MergeXR64Row_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    MergeXR64Row(src_r, src_g, src_b, dst_ar64, depth, width);
    src_r += src_stride_r;
    src_g += src_stride_g;
    src_b += src_stride_b;
    dst_ar64 += dst_stride_ar64;
  }
}

LIBYUV_API
void MergeAR64Plane(const uint16_t* src_r,
                    int src_stride_r,
                    const uint16_t* src_g,
                    int src_stride_g,
                    const uint16_t* src_b,
                    int src_stride_b,
                    const uint16_t* src_a,
                    int src_stride_a,
                    uint16_t* dst_ar64,
                    int dst_stride_ar64,
                    int width,
                    int height,
                    int depth) {
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar64 = dst_ar64 + (height - 1) * dst_stride_ar64;
    dst_stride_ar64 = -dst_stride_ar64;
  }

  if (src_a == NULL) {
    MergeAR64PlaneOpaque(src_r, src_stride_r, src_g, src_stride_g, src_b,
                         src_stride_b, dst_ar64, dst_stride_ar64, width, height,
                         depth);
  } else {
    MergeAR64PlaneAlpha(src_r, src_stride_r, src_g, src_stride_g, src_b,
                        src_stride_b, src_a, src_stride_a, dst_ar64,
                        dst_stride_ar64, width, height, depth);
  }
}

LIBYUV_NOINLINE
static void MergeARGB16To8PlaneAlpha(const uint16_t* src_r,
                                     int src_stride_r,
                                     const uint16_t* src_g,
                                     int src_stride_g,
                                     const uint16_t* src_b,
                                     int src_stride_b,
                                     const uint16_t* src_a,
                                     int src_stride_a,
                                     uint8_t* dst_argb,
                                     int dst_stride_argb,
                                     int width,
                                     int height,
                                     int depth) {
  int y;
  void (*MergeARGB16To8Row)(const uint16_t* src_r, const uint16_t* src_g,
                            const uint16_t* src_b, const uint16_t* src_a,
                            uint8_t* dst_argb, int depth, int width) =
      MergeARGB16To8Row_C;

  if (src_stride_r == width && src_stride_g == width && src_stride_b == width &&
      src_stride_a == width && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_r = src_stride_g = src_stride_b = src_stride_a =
        dst_stride_argb = 0;
  }
#if defined(HAS_MERGEARGB16TO8ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeARGB16To8Row = MergeARGB16To8Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      MergeARGB16To8Row = MergeARGB16To8Row_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEARGB16TO8ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeARGB16To8Row = MergeARGB16To8Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      MergeARGB16To8Row = MergeARGB16To8Row_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    MergeARGB16To8Row(src_r, src_g, src_b, src_a, dst_argb, depth, width);
    src_r += src_stride_r;
    src_g += src_stride_g;
    src_b += src_stride_b;
    src_a += src_stride_a;
    dst_argb += dst_stride_argb;
  }
}

LIBYUV_NOINLINE
static void MergeARGB16To8PlaneOpaque(const uint16_t* src_r,
                                      int src_stride_r,
                                      const uint16_t* src_g,
                                      int src_stride_g,
                                      const uint16_t* src_b,
                                      int src_stride_b,
                                      uint8_t* dst_argb,
                                      int dst_stride_argb,
                                      int width,
                                      int height,
                                      int depth) {
  int y;
  void (*MergeXRGB16To8Row)(const uint16_t* src_r, const uint16_t* src_g,
                            const uint16_t* src_b, uint8_t* dst_argb, int depth,
                            int width) = MergeXRGB16To8Row_C;

  // Coalesce rows.
  if (src_stride_r == width && src_stride_g == width && src_stride_b == width &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_r = src_stride_g = src_stride_b = dst_stride_argb = 0;
  }
#if defined(HAS_MERGEXRGB16TO8ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeXRGB16To8Row = MergeXRGB16To8Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      MergeXRGB16To8Row = MergeXRGB16To8Row_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEXRGB16TO8ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeXRGB16To8Row = MergeXRGB16To8Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      MergeXRGB16To8Row = MergeXRGB16To8Row_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    MergeXRGB16To8Row(src_r, src_g, src_b, dst_argb, depth, width);
    src_r += src_stride_r;
    src_g += src_stride_g;
    src_b += src_stride_b;
    dst_argb += dst_stride_argb;
  }
}

LIBYUV_API
void MergeARGB16To8Plane(const uint16_t* src_r,
                         int src_stride_r,
                         const uint16_t* src_g,
                         int src_stride_g,
                         const uint16_t* src_b,
                         int src_stride_b,
                         const uint16_t* src_a,
                         int src_stride_a,
                         uint8_t* dst_argb,
                         int dst_stride_argb,
                         int width,
                         int height,
                         int depth) {
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }

  if (src_a == NULL) {
    MergeARGB16To8PlaneOpaque(src_r, src_stride_r, src_g, src_stride_g, src_b,
                              src_stride_b, dst_argb, dst_stride_argb, width,
                              height, depth);
  } else {
    MergeARGB16To8PlaneAlpha(src_r, src_stride_r, src_g, src_stride_g, src_b,
                             src_stride_b, src_a, src_stride_a, dst_argb,
                             dst_stride_argb, width, height, depth);
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
#if defined(HAS_YUY2TOYROW_MSA) && defined(HAS_YUY2TOUV422ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    YUY2ToYRow = YUY2ToYRow_Any_MSA;
    YUY2ToUV422Row = YUY2ToUV422Row_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_MSA;
      YUY2ToUV422Row = YUY2ToUV422Row_MSA;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_LSX) && defined(HAS_YUY2TOUV422ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    YUY2ToYRow = YUY2ToYRow_Any_LSX;
    YUY2ToUV422Row = YUY2ToUV422Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToYRow = YUY2ToYRow_LSX;
      YUY2ToUV422Row = YUY2ToUV422Row_LSX;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_LASX) && defined(HAS_YUY2TOUV422ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    YUY2ToYRow = YUY2ToYRow_Any_LASX;
    YUY2ToUV422Row = YUY2ToUV422Row_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_LASX;
      YUY2ToUV422Row = YUY2ToUV422Row_LASX;
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
#if defined(HAS_UYVYTOYROW_MSA) && defined(HAS_UYVYTOUV422ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    UYVYToYRow = UYVYToYRow_Any_MSA;
    UYVYToUV422Row = UYVYToUV422Row_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      UYVYToYRow = UYVYToYRow_MSA;
      UYVYToUV422Row = UYVYToUV422Row_MSA;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_LSX) && defined(HAS_UYVYTOUV422ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    UYVYToYRow = UYVYToYRow_Any_LSX;
    UYVYToUV422Row = UYVYToUV422Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      UYVYToYRow = UYVYToYRow_LSX;
      UYVYToUV422Row = UYVYToUV422Row_LSX;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_LASX) && defined(HAS_UYVYTOUV422ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    UYVYToYRow = UYVYToYRow_Any_LASX;
    UYVYToUV422Row = UYVYToUV422Row_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      UYVYToYRow = UYVYToYRow_LASX;
      UYVYToUV422Row = UYVYToUV422Row_LASX;
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

// Convert UYVY to Y.
LIBYUV_API
int UYVYToY(const uint8_t* src_uyvy,
            int src_stride_uyvy,
            uint8_t* dst_y,
            int dst_stride_y,
            int width,
            int height) {
  int y;
  void (*UYVYToYRow)(const uint8_t* src_uyvy, uint8_t* dst_y, int width) =
      UYVYToYRow_C;
  if (!src_uyvy || !dst_y || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uyvy = src_uyvy + (height - 1) * src_stride_uyvy;
    src_stride_uyvy = -src_stride_uyvy;
  }
  // Coalesce rows.
  if (src_stride_uyvy == width * 2 && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_uyvy = dst_stride_y = 0;
  }
#if defined(HAS_UYVYTOYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    UYVYToYRow = UYVYToYRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      UYVYToYRow = UYVYToYRow_SSE2;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    UYVYToYRow = UYVYToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      UYVYToYRow = UYVYToYRow_AVX2;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    UYVYToYRow = UYVYToYRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      UYVYToYRow = UYVYToYRow_NEON;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    UYVYToYRow = UYVYToYRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      UYVYToYRow = UYVYToYRow_MSA;
    }
  }
#endif
#if defined(HAS_UYVYTOYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    UYVYToYRow = UYVYToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      UYVYToYRow = UYVYToYRow_LSX;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    UYVYToYRow(src_uyvy, dst_y, width);
    src_uyvy += src_stride_uyvy;
    dst_y += dst_stride_y;
  }
  return 0;
}

// Mirror a plane of data.
// See Also I400Mirror
LIBYUV_API
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
    if (IS_ALIGNED(width, 32)) {
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
#if defined(HAS_MIRRORROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    MirrorRow = MirrorRow_Any_LSX;
    if (IS_ALIGNED(width, 32)) {
      MirrorRow = MirrorRow_LSX;
    }
  }
#endif
#if defined(HAS_MIRRORROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    MirrorRow = MirrorRow_Any_LASX;
    if (IS_ALIGNED(width, 64)) {
      MirrorRow = MirrorRow_LASX;
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

// Mirror a plane of UV data.
LIBYUV_API
void MirrorUVPlane(const uint8_t* src_uv,
                   int src_stride_uv,
                   uint8_t* dst_uv,
                   int dst_stride_uv,
                   int width,
                   int height) {
  int y;
  void (*MirrorUVRow)(const uint8_t* src, uint8_t* dst, int width) =
      MirrorUVRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uv = src_uv + (height - 1) * src_stride_uv;
    src_stride_uv = -src_stride_uv;
  }
#if defined(HAS_MIRRORUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MirrorUVRow = MirrorUVRow_Any_NEON;
    if (IS_ALIGNED(width, 32)) {
      MirrorUVRow = MirrorUVRow_NEON;
    }
  }
#endif
#if defined(HAS_MIRRORUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    MirrorUVRow = MirrorUVRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      MirrorUVRow = MirrorUVRow_SSSE3;
    }
  }
#endif
#if defined(HAS_MIRRORUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MirrorUVRow = MirrorUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      MirrorUVRow = MirrorUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_MIRRORUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    MirrorUVRow = MirrorUVRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      MirrorUVRow = MirrorUVRow_MSA;
    }
  }
#endif
#if defined(HAS_MIRRORUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    MirrorUVRow = MirrorUVRow_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      MirrorUVRow = MirrorUVRow_LSX;
    }
  }
#endif
#if defined(HAS_MIRRORUVROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    MirrorUVRow = MirrorUVRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      MirrorUVRow = MirrorUVRow_LASX;
    }
  }
#endif

  // MirrorUV plane
  for (y = 0; y < height; ++y) {
    MirrorUVRow(src_uv, dst_uv, width);
    src_uv += src_stride_uv;
    dst_uv += dst_stride_uv;
  }
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

  if ((!src_y && dst_y) || !src_u || !src_v || !dst_u || !dst_v || width <= 0 ||
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

// NV12 mirror.
LIBYUV_API
int NV12Mirror(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;

  if ((!src_y && dst_y) || !src_uv || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_uv = src_uv + (halfheight - 1) * src_stride_uv;
    src_stride_y = -src_stride_y;
    src_stride_uv = -src_stride_uv;
  }

  if (dst_y) {
    MirrorPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
  }
  MirrorUVPlane(src_uv, src_stride_uv, dst_uv, dst_stride_uv, halfwidth,
                halfheight);
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
    if (IS_ALIGNED(width, 8)) {
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
#if defined(HAS_ARGBMIRRORROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBMirrorRow = ARGBMirrorRow_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      ARGBMirrorRow = ARGBMirrorRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBMIRRORROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBMirrorRow = ARGBMirrorRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      ARGBMirrorRow = ARGBMirrorRow_LASX;
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

// RGB24 mirror.
LIBYUV_API
int RGB24Mirror(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  int y;
  void (*RGB24MirrorRow)(const uint8_t* src, uint8_t* dst, int width) =
      RGB24MirrorRow_C;
  if (!src_rgb24 || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb24 = src_rgb24 + (height - 1) * src_stride_rgb24;
    src_stride_rgb24 = -src_stride_rgb24;
  }
#if defined(HAS_RGB24MIRRORROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB24MirrorRow = RGB24MirrorRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      RGB24MirrorRow = RGB24MirrorRow_NEON;
    }
  }
#endif
#if defined(HAS_RGB24MIRRORROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RGB24MirrorRow = RGB24MirrorRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGB24MirrorRow = RGB24MirrorRow_SSSE3;
    }
  }
#endif

  // Mirror plane
  for (y = 0; y < height; ++y) {
    RGB24MirrorRow(src_rgb24, dst_rgb24, width);
    src_rgb24 += src_stride_rgb24;
    dst_rgb24 += dst_stride_rgb24;
  }
  return 0;
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
                       uint8_t* dst_argb, int width) = ARGBBlendRow_C;
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
#if defined(HAS_ARGBBLENDROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBBlendRow = ARGBBlendRow_SSSE3;
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
#if defined(HAS_ARGBBLENDROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBBlendRow = ARGBBlendRow_LSX;
  }
#endif
#if defined(HAS_ARGBBLENDROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBBlendRow = ARGBBlendRow_RVV;
  }
#endif
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
#if defined(HAS_BLENDPLANEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    BlendPlaneRow = BlendPlaneRow_RVV;
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
#if defined(HAS_BLENDPLANEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    BlendPlaneRow = BlendPlaneRow_RVV;
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
#if defined(HAS_SCALEROWDOWN2_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ScaleRowDown2 = ScaleRowDown2Box_RVV;
  }
#endif

  // Row buffer for intermediate alpha pixels.
  align_buffer_64(halfalpha, halfwidth);
  if (!halfalpha)
    return 1;
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
#if defined(HAS_ARGBMULTIPLYROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    ARGBMultiplyRow = ARGBMultiplyRow_SME;
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
#if defined(HAS_ARGBMULTIPLYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBMultiplyRow = ARGBMultiplyRow_Any_LSX;
    if (IS_ALIGNED(width, 4)) {
      ARGBMultiplyRow = ARGBMultiplyRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBMULTIPLYROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBMultiplyRow = ARGBMultiplyRow_Any_LASX;
    if (IS_ALIGNED(width, 8)) {
      ARGBMultiplyRow = ARGBMultiplyRow_LASX;
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
#if defined(HAS_ARGBADDROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBAddRow = ARGBAddRow_SSE2;
  }
#endif
#if defined(HAS_ARGBADDROW_SSE2)
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
#if defined(HAS_ARGBADDROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBAddRow = ARGBAddRow_Any_LSX;
    if (IS_ALIGNED(width, 4)) {
      ARGBAddRow = ARGBAddRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBADDROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBAddRow = ARGBAddRow_Any_LASX;
    if (IS_ALIGNED(width, 8)) {
      ARGBAddRow = ARGBAddRow_LASX;
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
#if defined(HAS_ARGBSUBTRACTROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBSubtractRow = ARGBSubtractRow_Any_LSX;
    if (IS_ALIGNED(width, 4)) {
      ARGBSubtractRow = ARGBSubtractRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBSUBTRACTROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBSubtractRow = ARGBSubtractRow_Any_LASX;
    if (IS_ALIGNED(width, 8)) {
      ARGBSubtractRow = ARGBSubtractRow_LASX;
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
#if defined(HAS_RAWTORGB24ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    RAWToRGB24Row = RAWToRGB24Row_SVE2;
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
#if defined(HAS_RAWTORGB24ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RAWToRGB24Row = RAWToRGB24Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RAWToRGB24Row = RAWToRGB24Row_LSX;
    }
  }
#endif
#if defined(HAS_RAWTORGB24ROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RAWToRGB24Row = RAWToRGB24Row_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    RAWToRGB24Row(src_raw, dst_rgb24, width);
    src_raw += src_stride_raw;
    dst_rgb24 += dst_stride_rgb24;
  }
  return 0;
}

// TODO(fbarchard): Consider uint8_t value
LIBYUV_API
void SetPlane(uint8_t* dst_y,
              int dst_stride_y,
              int width,
              int height,
              uint32_t value) {
  int y;
  void (*SetRow)(uint8_t* dst, uint8_t value, int width) = SetRow_C;

  if (width <= 0 || height == 0) {
    return;
  }
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
#if defined(HAS_SETROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    SetRow = SetRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      SetRow = SetRow_LSX;
    }
  }
#endif

  // Set plane
  for (y = 0; y < height; ++y) {
    SetRow(dst_y, (uint8_t)value, width);
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
  void (*ARGBSetRow)(uint8_t* dst_argb, uint32_t value, int width) =
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
#if defined(HAS_ARGBSETROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBSetRow = ARGBSetRow_Any_LSX;
    if (IS_ALIGNED(width, 4)) {
      ARGBSetRow = ARGBSetRow_LSX;
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
#if defined(HAS_ARGBATTENUATEROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBAttenuateRow = ARGBAttenuateRow_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      ARGBAttenuateRow = ARGBAttenuateRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBATTENUATEROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBAttenuateRow = ARGBAttenuateRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      ARGBAttenuateRow = ARGBAttenuateRow_LASX;
    }
  }
#endif
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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
#if defined(HAS_ARGBGRAYROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_NEON_DotProd;
  }
#endif
#if defined(HAS_ARGBGRAYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_MSA;
  }
#endif
#if defined(HAS_ARGBGRAYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_LSX;
  }
#endif
#if defined(HAS_ARGBGRAYROW_LASX)
  if (TestCpuFlag(kCpuHasLASX) && IS_ALIGNED(width, 16)) {
    ARGBGrayRow = ARGBGrayRow_LASX;
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
#if defined(HAS_ARGBGRAYROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_NEON_DotProd;
  }
#endif
#if defined(HAS_ARGBGRAYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_MSA;
  }
#endif
#if defined(HAS_ARGBGRAYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX) && IS_ALIGNED(width, 8)) {
    ARGBGrayRow = ARGBGrayRow_LSX;
  }
#endif
#if defined(HAS_ARGBGRAYROW_LASX)
  if (TestCpuFlag(kCpuHasLASX) && IS_ALIGNED(width, 16)) {
    ARGBGrayRow = ARGBGrayRow_LASX;
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
  void (*ARGBSepiaRow)(uint8_t* dst_argb, int width) = ARGBSepiaRow_C;
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
#if defined(HAS_ARGBSEPIAROW_NEON_DOTPROD)
  if (TestCpuFlag(kCpuHasNeonDotProd) && IS_ALIGNED(width, 8)) {
    ARGBSepiaRow = ARGBSepiaRow_NEON_DotProd;
  }
#endif
#if defined(HAS_ARGBSEPIAROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 8)) {
    ARGBSepiaRow = ARGBSepiaRow_MSA;
  }
#endif
#if defined(HAS_ARGBSEPIAROW_LSX)
  if (TestCpuFlag(kCpuHasLSX) && IS_ALIGNED(width, 8)) {
    ARGBSepiaRow = ARGBSepiaRow_LSX;
  }
#endif
#if defined(HAS_ARGBSEPIAROW_LASX)
  if (TestCpuFlag(kCpuHasLASX) && IS_ALIGNED(width, 16)) {
    ARGBSepiaRow = ARGBSepiaRow_LASX;
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
#if defined(HAS_ARGBCOLORMATRIXROW_NEON_I8MM)
  if (TestCpuFlag(kCpuHasNeonI8MM) && IS_ALIGNED(width, 8)) {
    ARGBColorMatrixRow = ARGBColorMatrixRow_NEON_I8MM;
  }
#endif
#if defined(HAS_ARGBCOLORMATRIXROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 8)) {
    ARGBColorMatrixRow = ARGBColorMatrixRow_MSA;
  }
#endif
#if defined(HAS_ARGBCOLORMATRIXROW_LSX)
  if (TestCpuFlag(kCpuHasLSX) && IS_ALIGNED(width, 8)) {
    ARGBColorMatrixRow = ARGBColorMatrixRow_LSX;
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
  void (*ARGBColorTableRow)(uint8_t* dst_argb, const uint8_t* table_argb,
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
  void (*RGBColorTableRow)(uint8_t* dst_argb, const uint8_t* table_argb,
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
  void (*ARGBQuantizeRow)(uint8_t* dst_argb, int scale, int interval_size,
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
#if defined(HAS_ARGBQUANTIZEROW_LSX)
  if (TestCpuFlag(kCpuHasLSX) && IS_ALIGNED(width, 8)) {
    ARGBQuantizeRow = ARGBQuantizeRow_LSX;
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
  if (radius <= 0 || height <= 1) {
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
#if defined(HAS_ARGBSHADEROW_LSX)
  if (TestCpuFlag(kCpuHasLSX) && IS_ALIGNED(width, 4)) {
    ARGBShadeRow = ARGBShadeRow_LSX;
  }
#endif
#if defined(HAS_ARGBSHADEROW_LASX)
  if (TestCpuFlag(kCpuHasLASX) && IS_ALIGNED(width, 8)) {
    ARGBShadeRow = ARGBShadeRow_LASX;
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
  void (*InterpolateRow)(uint8_t* dst_ptr, const uint8_t* src_ptr,
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
#if defined(HAS_INTERPOLATEROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    InterpolateRow = InterpolateRow_SME;
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
#if defined(HAS_INTERPOLATEROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    InterpolateRow = InterpolateRow_Any_LSX;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_LSX;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    InterpolateRow = InterpolateRow_RVV;
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

// Interpolate 2 planes by specified amount (0 to 255).
LIBYUV_API
int InterpolatePlane_16(const uint16_t* src0,
                        int src_stride0,
                        const uint16_t* src1,
                        int src_stride1,
                        uint16_t* dst,
                        int dst_stride,
                        int width,
                        int height,
                        int interpolation) {
  int y;
  void (*InterpolateRow_16)(uint16_t* dst_ptr, const uint16_t* src_ptr,
                            ptrdiff_t src_stride, int dst_width,
                            int source_y_fraction) = InterpolateRow_16_C;
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
#if defined(HAS_INTERPOLATEROW_16_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow_16 = InterpolateRow_16_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      InterpolateRow_16 = InterpolateRow_16_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_16_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow_16 = InterpolateRow_16_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow_16 = InterpolateRow_16_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_16_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow_16 = InterpolateRow_16_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      InterpolateRow_16 = InterpolateRow_16_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_16_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    InterpolateRow_16 = InterpolateRow_16_SME;
  }
#endif
#if defined(HAS_INTERPOLATEROW_16_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    InterpolateRow_16 = InterpolateRow_16_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow_16 = InterpolateRow_16_MSA;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_16_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    InterpolateRow_16 = InterpolateRow_16_Any_LSX;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow_16 = InterpolateRow_16_LSX;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    InterpolateRow_16(dst, src0, src1 - src0, width, interpolation);
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
#if defined(HAS_ARGBSHUFFLEROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBShuffleRow = ARGBShuffleRow_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      ARGBShuffleRow = ARGBShuffleRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBSHUFFLEROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBShuffleRow = ARGBShuffleRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      ARGBShuffleRow = ARGBShuffleRow_LASX;
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

// Shuffle AR64 channel order.  e.g. AR64 to AB64.
LIBYUV_API
int AR64Shuffle(const uint16_t* src_ar64,
                int src_stride_ar64,
                uint16_t* dst_ar64,
                int dst_stride_ar64,
                const uint8_t* shuffler,
                int width,
                int height) {
  int y;
  void (*AR64ShuffleRow)(const uint8_t* src_ar64, uint8_t* dst_ar64,
                         const uint8_t* shuffler, int width) = AR64ShuffleRow_C;
  if (!src_ar64 || !dst_ar64 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_ar64 = src_ar64 + (height - 1) * src_stride_ar64;
    src_stride_ar64 = -src_stride_ar64;
  }
  // Coalesce rows.
  if (src_stride_ar64 == width * 4 && dst_stride_ar64 == width * 4) {
    width *= height;
    height = 1;
    src_stride_ar64 = dst_stride_ar64 = 0;
  }
  // Assembly versions can be reused if it's implemented with shuffle.
#if defined(HAS_ARGBSHUFFLEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    AR64ShuffleRow = ARGBShuffleRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      AR64ShuffleRow = ARGBShuffleRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBSHUFFLEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    AR64ShuffleRow = ARGBShuffleRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      AR64ShuffleRow = ARGBShuffleRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBSHUFFLEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    AR64ShuffleRow = ARGBShuffleRow_Any_NEON;
    if (IS_ALIGNED(width, 4)) {
      AR64ShuffleRow = ARGBShuffleRow_NEON;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    AR64ShuffleRow((uint8_t*)(src_ar64), (uint8_t*)(dst_ar64), shuffler,
                   width * 2);
    src_ar64 += src_stride_ar64;
    dst_ar64 += dst_stride_ar64;
  }
  return 0;
}

// Gauss blur a float plane using Gaussian 5x5 filter with
// coefficients of 1, 4, 6, 4, 1.
// Each destination pixel is a blur of the 5x5
// pixels from the source.
// Source edges are clamped.
// Edge is 2 pixels on each side, and interior is multiple of 4.
LIBYUV_API
int GaussPlane_F32(const float* src,
                   int src_stride,
                   float* dst,
                   int dst_stride,
                   int width,
                   int height) {
  int y;
  void (*GaussCol_F32)(const float* src0, const float* src1, const float* src2,
                       const float* src3, const float* src4, float* dst,
                       int width) = GaussCol_F32_C;
  void (*GaussRow_F32)(const float* src, float* dst, int width) =
      GaussRow_F32_C;
  if (!src || !dst || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src = src + (height - 1) * src_stride;
    src_stride = -src_stride;
  }

#if defined(HAS_GAUSSCOL_F32_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 8)) {
    GaussCol_F32 = GaussCol_F32_NEON;
  }
#endif
#if defined(HAS_GAUSSROW_F32_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 8)) {
    GaussRow_F32 = GaussRow_F32_NEON;
  }
#endif
  {
    // 2 pixels on each side, but aligned out to 16 bytes.
    align_buffer_64(rowbuf, (4 + width + 4) * 4);
    if (!rowbuf)
      return 1;
    memset(rowbuf, 0, 16);
    memset(rowbuf + (4 + width) * 4, 0, 16);
    float* row = (float*)(rowbuf + 16);
    const float* src0 = src;
    const float* src1 = src;
    const float* src2 = src;
    const float* src3 = src2 + ((height > 1) ? src_stride : 0);
    const float* src4 = src3 + ((height > 2) ? src_stride : 0);

    for (y = 0; y < height; ++y) {
      GaussCol_F32(src0, src1, src2, src3, src4, row, width);

      // Extrude edge by 2 floats
      row[-2] = row[-1] = row[0];
      row[width + 1] = row[width] = row[width - 1];

      GaussRow_F32(row - 2, dst, width);

      src0 = src1;
      src1 = src2;
      src2 = src3;
      src3 = src4;
      if ((y + 2) < (height - 1)) {
        src4 += src_stride;
      }
      dst += dst_stride;
    }
    free_aligned_buffer_64(rowbuf);
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
    if (IS_ALIGNED(width, 16)) {
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
#if defined(HAS_ARGBTOYJROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToYJRow = ARGBToYJRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYJRow = ARGBToYJRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBToYJRow = ARGBToYJRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGBToYJRow = ARGBToYJRow_LASX;
    }
  }
#endif
#if defined(HAS_ARGBTOYJROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBToYJRow = ARGBToYJRow_RVV;
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
    const int row_size = (width + kEdge + 31) & ~31;
    align_buffer_64(rows, row_size * 2 + (kEdge + row_size * 3 + kEdge));
    uint8_t* row_sobelx = rows;
    uint8_t* row_sobely = rows + row_size;
    uint8_t* row_y = rows + row_size * 2;

    // Convert first row.
    uint8_t* row_y0 = row_y + kEdge;
    uint8_t* row_y1 = row_y0 + row_size;
    uint8_t* row_y2 = row_y1 + row_size;
    if (!rows)
      return 1;
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
#if defined(HAS_SOBELROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    SobelRow = SobelRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      SobelRow = SobelRow_LSX;
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
#if defined(HAS_SOBELTOPLANEROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    SobelToPlaneRow = SobelToPlaneRow_Any_LSX;
    if (IS_ALIGNED(width, 32)) {
      SobelToPlaneRow = SobelToPlaneRow_LSX;
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
#if defined(HAS_SOBELXYROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    SobelXYRow = SobelXYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      SobelXYRow = SobelXYRow_LSX;
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
  if (TestCpuFlag(kCpuHasNEON)
#if defined(__arm__)
      // When scale is 1/65535 the scale * 2^-112 used to convert is a denormal.
      // But when Neon vmul is asked to multiply a normal float by that
      // denormal scale, even though the result would have been normal, it
      // flushes to zero.  The scalar version of vmul supports denormals.
      && scale >= 1.0f / 4096.0f
#endif
  ) {
    HalfFloatRow = HalfFloatRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      HalfFloatRow = HalfFloatRow_NEON;
    }
  }
#endif
#if defined(HAS_HALFFLOATROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    HalfFloatRow = scale == 1.0f ? HalfFloat1Row_SVE2 : HalfFloatRow_SVE2;
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
#if defined(HAS_HALFFLOATROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    HalfFloatRow = HalfFloatRow_Any_LSX;
    if (IS_ALIGNED(width, 32)) {
      HalfFloatRow = HalfFloatRow_LSX;
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
#if defined(HAS_ARGBEXTRACTALPHAROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBExtractAlphaRow = IS_ALIGNED(width, 16) ? ARGBExtractAlphaRow_LSX
                                                : ARGBExtractAlphaRow_Any_LSX;
  }
#endif
#if defined(HAS_ARGBEXTRACTALPHAROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBExtractAlphaRow = ARGBExtractAlphaRow_RVV;
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
#if defined(HAS_ARGBCOPYYTOALPHAROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBCopyYToAlphaRow = ARGBCopyYToAlphaRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBCopyYToAlphaRow(src_y, dst_argb, width);
    src_y += src_stride_y;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

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
  void (*YUY2ToYRow)(const uint8_t* src_yuy2, uint8_t* dst_y, int width) =
      YUY2ToYRow_C;
  void (*YUY2ToNVUVRow)(const uint8_t* src_yuy2, int stride_yuy2,
                        uint8_t* dst_uv, int width) = YUY2ToNVUVRow_C;
  if (!src_yuy2 || !dst_y || !dst_uv || width <= 0 || height == 0) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_yuy2 = src_yuy2 + (height - 1) * src_stride_yuy2;
    src_stride_yuy2 = -src_stride_yuy2;
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
#if defined(HAS_YUY2TOYROW_MSA) && defined(HAS_YUY2TOUV422ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    YUY2ToYRow = YUY2ToYRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_MSA;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_LSX) && defined(HAS_YUY2TOUV422ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    YUY2ToYRow = YUY2ToYRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToYRow = YUY2ToYRow_LSX;
    }
  }
#endif
#if defined(HAS_YUY2TOYROW_LASX) && defined(HAS_YUY2TOUV422ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    YUY2ToYRow = YUY2ToYRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToYRow = YUY2ToYRow_LASX;
    }
  }
#endif

#if defined(HAS_YUY2TONVUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    YUY2ToNVUVRow = YUY2ToNVUVRow_Any_SSE2;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToNVUVRow = YUY2ToNVUVRow_SSE2;
    }
  }
#endif
#if defined(HAS_YUY2TONVUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    YUY2ToNVUVRow = YUY2ToNVUVRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToNVUVRow = YUY2ToNVUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_YUY2TONVUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    YUY2ToNVUVRow = YUY2ToNVUVRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToNVUVRow = YUY2ToNVUVRow_NEON;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    YUY2ToYRow(src_yuy2, dst_y, width);
    YUY2ToYRow(src_yuy2 + src_stride_yuy2, dst_y + dst_stride_y, width);
    YUY2ToNVUVRow(src_yuy2, src_stride_yuy2, dst_uv, width);
    src_yuy2 += src_stride_yuy2 * 2;
    dst_y += dst_stride_y * 2;
    dst_uv += dst_stride_uv;
  }
  if (height & 1) {
    YUY2ToYRow(src_yuy2, dst_y, width);
    YUY2ToNVUVRow(src_yuy2, 0, dst_uv, width);
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
  void (*InterpolateRow)(uint8_t* dst_ptr, const uint8_t* src_ptr,
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
#if defined(HAS_SPLITUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    SplitUVRow = SplitUVRow_Any_LSX;
    if (IS_ALIGNED(width, 32)) {
      SplitUVRow = SplitUVRow_LSX;
    }
  }
#endif
#if defined(HAS_SPLITUVROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    SplitUVRow = SplitUVRow_RVV;
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
#if defined(HAS_INTERPOLATEROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    InterpolateRow = InterpolateRow_SME;
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
#if defined(HAS_INTERPOLATEROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    InterpolateRow = InterpolateRow_Any_LSX;
    if (IS_ALIGNED(width, 32)) {
      InterpolateRow = InterpolateRow_LSX;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    InterpolateRow = InterpolateRow_RVV;
  }
#endif

  {
    int awidth = halfwidth * 2;
    // row of y and 2 rows of uv
    align_buffer_64(rows, awidth * 3);
    if (!rows)
      return 1;

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

// width and height are src size allowing odd size handling.
LIBYUV_API
void HalfMergeUVPlane(const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_uv,
                      int dst_stride_uv,
                      int width,
                      int height) {
  int y;
  void (*HalfMergeUVRow)(const uint8_t* src_u, int src_stride_u,
                         const uint8_t* src_v, int src_stride_v,
                         uint8_t* dst_uv, int width) = HalfMergeUVRow_C;

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_u = src_u + (height - 1) * src_stride_u;
    src_v = src_v + (height - 1) * src_stride_v;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }
#if defined(HAS_HALFMERGEUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 16)) {
    HalfMergeUVRow = HalfMergeUVRow_NEON;
  }
#endif
#if defined(HAS_HALFMERGEUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && IS_ALIGNED(width, 16)) {
    HalfMergeUVRow = HalfMergeUVRow_SSSE3;
  }
#endif
#if defined(HAS_HALFMERGEUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2) && IS_ALIGNED(width, 32)) {
    HalfMergeUVRow = HalfMergeUVRow_AVX2;
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    // Merge a row of U and V into a row of UV.
    HalfMergeUVRow(src_u, src_stride_u, src_v, src_stride_v, dst_uv, width);
    src_u += src_stride_u * 2;
    src_v += src_stride_v * 2;
    dst_uv += dst_stride_uv;
  }
  if (height & 1) {
    HalfMergeUVRow(src_u, 0, src_v, 0, dst_uv, width);
  }
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
