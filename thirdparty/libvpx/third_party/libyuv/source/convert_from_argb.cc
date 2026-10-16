/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/convert_from_argb.h"

#include "libyuv/basic_types.h"
#include "libyuv/cpu_id.h"
#include "libyuv/planar_functions.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// ARGB little endian (bgra in memory) to I444
LIBYUV_API
int ARGBToI444(const uint8_t* src_argb,
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
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  void (*ARGBToUV444Row)(const uint8_t* src_argb, uint8_t* dst_u,
                         uint8_t* dst_v, int width) = ARGBToUV444Row_C;
  if (!src_argb || !dst_y || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_y == width &&
      dst_stride_u == width && dst_stride_v == width) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_y = dst_stride_u = dst_stride_v = 0;
  }
#if defined(HAS_ARGBTOUV444ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUV444Row = ARGBToUV444Row_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOUV444ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToUV444Row = ARGBToUV444Row_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOUV444ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToUV444Row = ARGBToUV444Row_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUV444Row = ARGBToUV444Row_MSA;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
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
#if defined(HAS_ARGBTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToYRow = ARGBToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToUV444Row(src_argb, dst_u, dst_v, width);
    ARGBToYRow(src_argb, dst_y, width);
    src_argb += src_stride_argb;
    dst_y += dst_stride_y;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  return 0;
}

// ARGB little endian (bgra in memory) to I422
LIBYUV_API
int ARGBToI422(const uint8_t* src_argb,
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
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_y == width &&
      dst_stride_u * 2 == width && dst_stride_v * 2 == width) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_y = dst_stride_u = dst_stride_v = 0;
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

  for (y = 0; y < height; ++y) {
    ARGBToUVRow(src_argb, 0, dst_u, dst_v, width);
    ARGBToYRow(src_argb, dst_y, width);
    src_argb += src_stride_argb;
    dst_y += dst_stride_y;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  return 0;
}

LIBYUV_API
int ARGBToNV12(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height) {
  int y;
  int halfwidth = (width + 1) >> 1;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  void (*MergeUVRow_)(const uint8_t* src_u, const uint8_t* src_v,
                      uint8_t* dst_uv, int width) = MergeUVRow_C;
  if (!src_argb || !dst_y || !dst_uv || width <= 0 || height == 0) {
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
#if defined(HAS_MERGEUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    MergeUVRow_ = MergeUVRow_Any_SSE2;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow_ = MergeUVRow_SSE2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeUVRow_ = MergeUVRow_Any_AVX2;
    if (IS_ALIGNED(halfwidth, 32)) {
      MergeUVRow_ = MergeUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeUVRow_ = MergeUVRow_Any_NEON;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow_ = MergeUVRow_NEON;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    MergeUVRow_ = MergeUVRow_Any_MSA;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow_ = MergeUVRow_MSA;
    }
  }
#endif
  {
    // Allocate a rows of uv.
    align_buffer_64(row_u, ((halfwidth + 31) & ~31) * 2);
    uint8_t* row_v = row_u + ((halfwidth + 31) & ~31);

    for (y = 0; y < height - 1; y += 2) {
      ARGBToUVRow(src_argb, src_stride_argb, row_u, row_v, width);
      MergeUVRow_(row_u, row_v, dst_uv, halfwidth);
      ARGBToYRow(src_argb, dst_y, width);
      ARGBToYRow(src_argb + src_stride_argb, dst_y + dst_stride_y, width);
      src_argb += src_stride_argb * 2;
      dst_y += dst_stride_y * 2;
      dst_uv += dst_stride_uv;
    }
    if (height & 1) {
      ARGBToUVRow(src_argb, 0, row_u, row_v, width);
      MergeUVRow_(row_u, row_v, dst_uv, halfwidth);
      ARGBToYRow(src_argb, dst_y, width);
    }
    free_aligned_buffer_64(row_u);
  }
  return 0;
}

// Same as NV12 but U and V swapped.
LIBYUV_API
int ARGBToNV21(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height) {
  int y;
  int halfwidth = (width + 1) >> 1;
  void (*ARGBToUVRow)(const uint8_t* src_argb0, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  void (*MergeUVRow_)(const uint8_t* src_u, const uint8_t* src_v,
                      uint8_t* dst_vu, int width) = MergeUVRow_C;
  if (!src_argb || !dst_y || !dst_vu || width <= 0 || height == 0) {
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
#if defined(HAS_MERGEUVROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    MergeUVRow_ = MergeUVRow_Any_SSE2;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow_ = MergeUVRow_SSE2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    MergeUVRow_ = MergeUVRow_Any_AVX2;
    if (IS_ALIGNED(halfwidth, 32)) {
      MergeUVRow_ = MergeUVRow_AVX2;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    MergeUVRow_ = MergeUVRow_Any_NEON;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow_ = MergeUVRow_NEON;
    }
  }
#endif
#if defined(HAS_MERGEUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    MergeUVRow_ = MergeUVRow_Any_MSA;
    if (IS_ALIGNED(halfwidth, 16)) {
      MergeUVRow_ = MergeUVRow_MSA;
    }
  }
#endif
  {
    // Allocate a rows of uv.
    align_buffer_64(row_u, ((halfwidth + 31) & ~31) * 2);
    uint8_t* row_v = row_u + ((halfwidth + 31) & ~31);

    for (y = 0; y < height - 1; y += 2) {
      ARGBToUVRow(src_argb, src_stride_argb, row_u, row_v, width);
      MergeUVRow_(row_v, row_u, dst_vu, halfwidth);
      ARGBToYRow(src_argb, dst_y, width);
      ARGBToYRow(src_argb + src_stride_argb, dst_y + dst_stride_y, width);
      src_argb += src_stride_argb * 2;
      dst_y += dst_stride_y * 2;
      dst_vu += dst_stride_vu;
    }
    if (height & 1) {
      ARGBToUVRow(src_argb, 0, row_u, row_v, width);
      MergeUVRow_(row_v, row_u, dst_vu, halfwidth);
      ARGBToYRow(src_argb, dst_y, width);
    }
    free_aligned_buffer_64(row_u);
  }
  return 0;
}

// Convert ARGB to YUY2.
LIBYUV_API
int ARGBToYUY2(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_yuy2,
               int dst_stride_yuy2,
               int width,
               int height) {
  int y;
  void (*ARGBToUVRow)(const uint8_t* src_argb, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  void (*I422ToYUY2Row)(const uint8_t* src_y, const uint8_t* src_u,
                        const uint8_t* src_v, uint8_t* dst_yuy2, int width) =
      I422ToYUY2Row_C;

  if (!src_argb || !dst_yuy2 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_yuy2 = dst_yuy2 + (height - 1) * dst_stride_yuy2;
    dst_stride_yuy2 = -dst_stride_yuy2;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_yuy2 == width * 2) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_yuy2 = 0;
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

  {
    // Allocate a rows of yuv.
    align_buffer_64(row_y, ((width + 63) & ~63) * 2);
    uint8_t* row_u = row_y + ((width + 63) & ~63);
    uint8_t* row_v = row_u + ((width + 63) & ~63) / 2;

    for (y = 0; y < height; ++y) {
      ARGBToUVRow(src_argb, 0, row_u, row_v, width);
      ARGBToYRow(src_argb, row_y, width);
      I422ToYUY2Row(row_y, row_u, row_v, dst_yuy2, width);
      src_argb += src_stride_argb;
      dst_yuy2 += dst_stride_yuy2;
    }

    free_aligned_buffer_64(row_y);
  }
  return 0;
}

// Convert ARGB to UYVY.
LIBYUV_API
int ARGBToUYVY(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_uyvy,
               int dst_stride_uyvy,
               int width,
               int height) {
  int y;
  void (*ARGBToUVRow)(const uint8_t* src_argb, int src_stride_argb,
                      uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  void (*I422ToUYVYRow)(const uint8_t* src_y, const uint8_t* src_u,
                        const uint8_t* src_v, uint8_t* dst_uyvy, int width) =
      I422ToUYVYRow_C;

  if (!src_argb || !dst_uyvy || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_uyvy = dst_uyvy + (height - 1) * dst_stride_uyvy;
    dst_stride_uyvy = -dst_stride_uyvy;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_uyvy == width * 2) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_uyvy = 0;
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

  {
    // Allocate a rows of yuv.
    align_buffer_64(row_y, ((width + 63) & ~63) * 2);
    uint8_t* row_u = row_y + ((width + 63) & ~63);
    uint8_t* row_v = row_u + ((width + 63) & ~63) / 2;

    for (y = 0; y < height; ++y) {
      ARGBToUVRow(src_argb, 0, row_u, row_v, width);
      ARGBToYRow(src_argb, row_y, width);
      I422ToUYVYRow(row_y, row_u, row_v, dst_uyvy, width);
      src_argb += src_stride_argb;
      dst_uyvy += dst_stride_uyvy;
    }

    free_aligned_buffer_64(row_y);
  }
  return 0;
}

// Convert ARGB to I400.
LIBYUV_API
int ARGBToI400(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height) {
  int y;
  void (*ARGBToYRow)(const uint8_t* src_argb, uint8_t* dst_y, int width) =
      ARGBToYRow_C;
  if (!src_argb || !dst_y || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_y == width) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_y = 0;
  }
#if defined(HAS_ARGBTOYROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOYROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToYRow = ARGBToYRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
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
#if defined(HAS_ARGBTOYROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToYRow = ARGBToYRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToYRow(src_argb, dst_y, width);
    src_argb += src_stride_argb;
    dst_y += dst_stride_y;
  }
  return 0;
}

// Shuffle table for converting ARGB to RGBA.
static const uvec8 kShuffleMaskARGBToRGBA = {
    3u, 0u, 1u, 2u, 7u, 4u, 5u, 6u, 11u, 8u, 9u, 10u, 15u, 12u, 13u, 14u};

// Convert ARGB to RGBA.
LIBYUV_API
int ARGBToRGBA(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_rgba,
               int dst_stride_rgba,
               int width,
               int height) {
  return ARGBShuffle(src_argb, src_stride_argb, dst_rgba, dst_stride_rgba,
                     (const uint8_t*)(&kShuffleMaskARGBToRGBA), width, height);
}

// Convert ARGB To RGB24.
LIBYUV_API
int ARGBToRGB24(const uint8_t* src_argb,
                int src_stride_argb,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  int y;
  void (*ARGBToRGB24Row)(const uint8_t* src_argb, uint8_t* dst_rgb, int width) =
      ARGBToRGB24Row_C;
  if (!src_argb || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_rgb24 == width * 3) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_rgb24 = 0;
  }
#if defined(HAS_ARGBTORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToRGB24Row = ARGBToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToRGB24Row = ARGBToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTORGB24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToRGB24Row = ARGBToRGB24Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToRGB24Row = ARGBToRGB24Row_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTORGB24ROW_AVX512VBMI)
  if (TestCpuFlag(kCpuHasAVX512VBMI)) {
    ARGBToRGB24Row = ARGBToRGB24Row_Any_AVX512VBMI;
    if (IS_ALIGNED(width, 32)) {
      ARGBToRGB24Row = ARGBToRGB24Row_AVX512VBMI;
    }
  }
#endif
#if defined(HAS_ARGBTORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToRGB24Row = ARGBToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToRGB24Row = ARGBToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTORGB24ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToRGB24Row = ARGBToRGB24Row_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToRGB24Row = ARGBToRGB24Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToRGB24Row(src_argb, dst_rgb24, width);
    src_argb += src_stride_argb;
    dst_rgb24 += dst_stride_rgb24;
  }
  return 0;
}

// Convert ARGB To RAW.
LIBYUV_API
int ARGBToRAW(const uint8_t* src_argb,
              int src_stride_argb,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  int y;
  void (*ARGBToRAWRow)(const uint8_t* src_argb, uint8_t* dst_rgb, int width) =
      ARGBToRAWRow_C;
  if (!src_argb || !dst_raw || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_raw == width * 3) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_raw = 0;
  }
#if defined(HAS_ARGBTORAWROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToRAWRow = ARGBToRAWRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToRAWRow = ARGBToRAWRow_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTORAWROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToRAWRow = ARGBToRAWRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      ARGBToRAWRow = ARGBToRAWRow_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTORAWROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToRAWRow = ARGBToRAWRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToRAWRow = ARGBToRAWRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTORAWROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToRAWRow = ARGBToRAWRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGBToRAWRow = ARGBToRAWRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToRAWRow(src_argb, dst_raw, width);
    src_argb += src_stride_argb;
    dst_raw += dst_stride_raw;
  }
  return 0;
}

// Ordered 8x8 dither for 888 to 565.  Values from 0 to 7.
static const uint8_t kDither565_4x4[16] = {
    0, 4, 1, 5, 6, 2, 7, 3, 1, 5, 0, 4, 7, 3, 6, 2,
};

// Convert ARGB To RGB565 with 4x4 dither matrix (16 bytes).
LIBYUV_API
int ARGBToRGB565Dither(const uint8_t* src_argb,
                       int src_stride_argb,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const uint8_t* dither4x4,
                       int width,
                       int height) {
  int y;
  void (*ARGBToRGB565DitherRow)(const uint8_t* src_argb, uint8_t* dst_rgb,
                                const uint32_t dither4, int width) =
      ARGBToRGB565DitherRow_C;
  if (!src_argb || !dst_rgb565 || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  if (!dither4x4) {
    dither4x4 = kDither565_4x4;
  }
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

  for (y = 0; y < height; ++y) {
    ARGBToRGB565DitherRow(src_argb, dst_rgb565,
                          *(const uint32_t*)(dither4x4 + ((y & 3) << 2)),
                          width);
    src_argb += src_stride_argb;
    dst_rgb565 += dst_stride_rgb565;
  }
  return 0;
}

// Convert ARGB To RGB565.
// TODO(fbarchard): Consider using dither function low level with zeros.
LIBYUV_API
int ARGBToRGB565(const uint8_t* src_argb,
                 int src_stride_argb,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height) {
  int y;
  void (*ARGBToRGB565Row)(const uint8_t* src_argb, uint8_t* dst_rgb,
                          int width) = ARGBToRGB565Row_C;
  if (!src_argb || !dst_rgb565 || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_rgb565 == width * 2) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_rgb565 = 0;
  }
#if defined(HAS_ARGBTORGB565ROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBToRGB565Row = ARGBToRGB565Row_Any_SSE2;
    if (IS_ALIGNED(width, 4)) {
      ARGBToRGB565Row = ARGBToRGB565Row_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBTORGB565ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToRGB565Row = ARGBToRGB565Row_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBToRGB565Row = ARGBToRGB565Row_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTORGB565ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToRGB565Row = ARGBToRGB565Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToRGB565Row = ARGBToRGB565Row_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTORGB565ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToRGB565Row = ARGBToRGB565Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      ARGBToRGB565Row = ARGBToRGB565Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToRGB565Row(src_argb, dst_rgb565, width);
    src_argb += src_stride_argb;
    dst_rgb565 += dst_stride_rgb565;
  }
  return 0;
}

// Convert ARGB To ARGB1555.
LIBYUV_API
int ARGBToARGB1555(const uint8_t* src_argb,
                   int src_stride_argb,
                   uint8_t* dst_argb1555,
                   int dst_stride_argb1555,
                   int width,
                   int height) {
  int y;
  void (*ARGBToARGB1555Row)(const uint8_t* src_argb, uint8_t* dst_rgb,
                            int width) = ARGBToARGB1555Row_C;
  if (!src_argb || !dst_argb1555 || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb1555 == width * 2) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb1555 = 0;
  }
#if defined(HAS_ARGBTOARGB1555ROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBToARGB1555Row = ARGBToARGB1555Row_Any_SSE2;
    if (IS_ALIGNED(width, 4)) {
      ARGBToARGB1555Row = ARGBToARGB1555Row_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBTOARGB1555ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToARGB1555Row = ARGBToARGB1555Row_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBToARGB1555Row = ARGBToARGB1555Row_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOARGB1555ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToARGB1555Row = ARGBToARGB1555Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToARGB1555Row = ARGBToARGB1555Row_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOARGB1555ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToARGB1555Row = ARGBToARGB1555Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      ARGBToARGB1555Row = ARGBToARGB1555Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToARGB1555Row(src_argb, dst_argb1555, width);
    src_argb += src_stride_argb;
    dst_argb1555 += dst_stride_argb1555;
  }
  return 0;
}

// Convert ARGB To ARGB4444.
LIBYUV_API
int ARGBToARGB4444(const uint8_t* src_argb,
                   int src_stride_argb,
                   uint8_t* dst_argb4444,
                   int dst_stride_argb4444,
                   int width,
                   int height) {
  int y;
  void (*ARGBToARGB4444Row)(const uint8_t* src_argb, uint8_t* dst_rgb,
                            int width) = ARGBToARGB4444Row_C;
  if (!src_argb || !dst_argb4444 || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_argb4444 == width * 2) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_argb4444 = 0;
  }
#if defined(HAS_ARGBTOARGB4444ROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ARGBToARGB4444Row = ARGBToARGB4444Row_Any_SSE2;
    if (IS_ALIGNED(width, 4)) {
      ARGBToARGB4444Row = ARGBToARGB4444Row_SSE2;
    }
  }
#endif
#if defined(HAS_ARGBTOARGB4444ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToARGB4444Row = ARGBToARGB4444Row_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBToARGB4444Row = ARGBToARGB4444Row_AVX2;
    }
  }
#endif
#if defined(HAS_ARGBTOARGB4444ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToARGB4444Row = ARGBToARGB4444Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToARGB4444Row = ARGBToARGB4444Row_NEON;
    }
  }
#endif
#if defined(HAS_ARGBTOARGB4444ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToARGB4444Row = ARGBToARGB4444Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      ARGBToARGB4444Row = ARGBToARGB4444Row_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToARGB4444Row(src_argb, dst_argb4444, width);
    src_argb += src_stride_argb;
    dst_argb4444 += dst_stride_argb4444;
  }
  return 0;
}

// Convert ABGR To AR30.
LIBYUV_API
int ABGRToAR30(const uint8_t* src_abgr,
               int src_stride_abgr,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  int y;
  void (*ABGRToAR30Row)(const uint8_t* src_abgr, uint8_t* dst_rgb, int width) =
      ABGRToAR30Row_C;
  if (!src_abgr || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_abgr = src_abgr + (height - 1) * src_stride_abgr;
    src_stride_abgr = -src_stride_abgr;
  }
  // Coalesce rows.
  if (src_stride_abgr == width * 4 && dst_stride_ar30 == width * 4) {
    width *= height;
    height = 1;
    src_stride_abgr = dst_stride_ar30 = 0;
  }
#if defined(HAS_ABGRTOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ABGRToAR30Row = ABGRToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 4)) {
      ABGRToAR30Row = ABGRToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_ABGRTOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ABGRToAR30Row = ABGRToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ABGRToAR30Row = ABGRToAR30Row_AVX2;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    ABGRToAR30Row(src_abgr, dst_ar30, width);
    src_abgr += src_stride_abgr;
    dst_ar30 += dst_stride_ar30;
  }
  return 0;
}

// Convert ARGB To AR30.
LIBYUV_API
int ARGBToAR30(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  int y;
  void (*ARGBToAR30Row)(const uint8_t* src_argb, uint8_t* dst_rgb, int width) =
      ARGBToAR30Row_C;
  if (!src_argb || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_ar30 == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_ar30 = 0;
  }
#if defined(HAS_ARGBTOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToAR30Row = ARGBToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 4)) {
      ARGBToAR30Row = ARGBToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_ARGBTOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ARGBToAR30Row = ARGBToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      ARGBToAR30Row = ARGBToAR30Row_AVX2;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    ARGBToAR30Row(src_argb, dst_ar30, width);
    src_argb += src_stride_argb;
    dst_ar30 += dst_stride_ar30;
  }
  return 0;
}

// Convert ARGB to J420. (JPeg full range I420).
LIBYUV_API
int ARGBToJ420(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_yj,
               int dst_stride_yj,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*ARGBToUVJRow)(const uint8_t* src_argb0, int src_stride_argb,
                       uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVJRow_C;
  void (*ARGBToYJRow)(const uint8_t* src_argb, uint8_t* dst_yj, int width) =
      ARGBToYJRow_C;
  if (!src_argb || !dst_yj || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
#if defined(HAS_ARGBTOYJROW_SSSE3) && defined(HAS_ARGBTOUVJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_SSSE3;
    ARGBToYJRow = ARGBToYJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVJRow = ARGBToUVJRow_SSSE3;
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
#if defined(HAS_ARGBTOUVJROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVJRow = ARGBToUVJRow_NEON;
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
#if defined(HAS_ARGBTOUVJROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVJRow = ARGBToUVJRow_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    ARGBToUVJRow(src_argb, src_stride_argb, dst_u, dst_v, width);
    ARGBToYJRow(src_argb, dst_yj, width);
    ARGBToYJRow(src_argb + src_stride_argb, dst_yj + dst_stride_yj, width);
    src_argb += src_stride_argb * 2;
    dst_yj += dst_stride_yj * 2;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  if (height & 1) {
    ARGBToUVJRow(src_argb, 0, dst_u, dst_v, width);
    ARGBToYJRow(src_argb, dst_yj, width);
  }
  return 0;
}

// Convert ARGB to J422. (JPeg full range I422).
LIBYUV_API
int ARGBToJ422(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_yj,
               int dst_stride_yj,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height) {
  int y;
  void (*ARGBToUVJRow)(const uint8_t* src_argb0, int src_stride_argb,
                       uint8_t* dst_u, uint8_t* dst_v, int width) =
      ARGBToUVJRow_C;
  void (*ARGBToYJRow)(const uint8_t* src_argb, uint8_t* dst_yj, int width) =
      ARGBToYJRow_C;
  if (!src_argb || !dst_yj || !dst_u || !dst_v || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_yj == width &&
      dst_stride_u * 2 == width && dst_stride_v * 2 == width) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_yj = dst_stride_u = dst_stride_v = 0;
  }
#if defined(HAS_ARGBTOYJROW_SSSE3) && defined(HAS_ARGBTOUVJROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_SSSE3;
    ARGBToYJRow = ARGBToYJRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVJRow = ARGBToUVJRow_SSSE3;
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
#if defined(HAS_ARGBTOUVJROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      ARGBToUVJRow = ARGBToUVJRow_NEON;
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
#if defined(HAS_ARGBTOUVJROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGBToUVJRow = ARGBToUVJRow_Any_MSA;
    if (IS_ALIGNED(width, 32)) {
      ARGBToUVJRow = ARGBToUVJRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToUVJRow(src_argb, 0, dst_u, dst_v, width);
    ARGBToYJRow(src_argb, dst_yj, width);
    src_argb += src_stride_argb;
    dst_yj += dst_stride_yj;
    dst_u += dst_stride_u;
    dst_v += dst_stride_v;
  }
  return 0;
}

// Convert ARGB to J400.
LIBYUV_API
int ARGBToJ400(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_yj,
               int dst_stride_yj,
               int width,
               int height) {
  int y;
  void (*ARGBToYJRow)(const uint8_t* src_argb, uint8_t* dst_yj, int width) =
      ARGBToYJRow_C;
  if (!src_argb || !dst_yj || width <= 0 || height == 0) {
    return -1;
  }
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_yj == width) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_yj = 0;
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

  for (y = 0; y < height; ++y) {
    ARGBToYJRow(src_argb, dst_yj, width);
    src_argb += src_stride_argb;
    dst_yj += dst_stride_yj;
  }
  return 0;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
