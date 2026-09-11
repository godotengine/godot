/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/scale.h"

#include <assert.h>
#include <string.h>

#include "libyuv/cpu_id.h"
#include "libyuv/planar_functions.h"  // For CopyARGB
#include "libyuv/row.h"
#include "libyuv/scale_row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

static __inline int Abs(int v) {
  return v >= 0 ? v : -v;
}

// ScaleARGB ARGB, 1/2
// This is an optimized version for scaling down a ARGB to 1/2 of
// its original size.
static void ScaleARGBDown2(int src_width,
                           int src_height,
                           int dst_width,
                           int dst_height,
                           int src_stride,
                           int dst_stride,
                           const uint8_t* src_argb,
                           uint8_t* dst_argb,
                           int x,
                           int dx,
                           int y,
                           int dy,
                           enum FilterMode filtering) {
  int j;
  int row_stride = src_stride * (dy >> 16);
  void (*ScaleARGBRowDown2)(const uint8_t* src_argb, ptrdiff_t src_stride,
                            uint8_t* dst_argb, int dst_width) =
      filtering == kFilterNone
          ? ScaleARGBRowDown2_C
          : (filtering == kFilterLinear ? ScaleARGBRowDown2Linear_C
                                        : ScaleARGBRowDown2Box_C);
  (void)src_width;
  (void)src_height;
  (void)dx;
  assert(dx == 65536 * 2);      // Test scale factor of 2.
  assert((dy & 0x1ffff) == 0);  // Test vertical scale is multiple of 2.
  // Advance to odd row, even column.
  if (filtering == kFilterBilinear) {
    src_argb += (y >> 16) * src_stride + (x >> 16) * 4;
  } else {
    src_argb += (y >> 16) * src_stride + ((x >> 16) - 1) * 4;
  }

#if defined(HAS_SCALEARGBROWDOWN2_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ScaleARGBRowDown2 =
        filtering == kFilterNone
            ? ScaleARGBRowDown2_Any_SSE2
            : (filtering == kFilterLinear ? ScaleARGBRowDown2Linear_Any_SSE2
                                          : ScaleARGBRowDown2Box_Any_SSE2);
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBRowDown2 =
          filtering == kFilterNone
              ? ScaleARGBRowDown2_SSE2
              : (filtering == kFilterLinear ? ScaleARGBRowDown2Linear_SSE2
                                            : ScaleARGBRowDown2Box_SSE2);
    }
  }
#endif
#if defined(HAS_SCALEARGBROWDOWN2_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBRowDown2 =
        filtering == kFilterNone
            ? ScaleARGBRowDown2_Any_NEON
            : (filtering == kFilterLinear ? ScaleARGBRowDown2Linear_Any_NEON
                                          : ScaleARGBRowDown2Box_Any_NEON);
    if (IS_ALIGNED(dst_width, 8)) {
      ScaleARGBRowDown2 =
          filtering == kFilterNone
              ? ScaleARGBRowDown2_NEON
              : (filtering == kFilterLinear ? ScaleARGBRowDown2Linear_NEON
                                            : ScaleARGBRowDown2Box_NEON);
    }
  }
#endif
#if defined(HAS_SCALEARGBROWDOWN2_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ScaleARGBRowDown2 =
        filtering == kFilterNone
            ? ScaleARGBRowDown2_Any_MSA
            : (filtering == kFilterLinear ? ScaleARGBRowDown2Linear_Any_MSA
                                          : ScaleARGBRowDown2Box_Any_MSA);
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBRowDown2 =
          filtering == kFilterNone
              ? ScaleARGBRowDown2_MSA
              : (filtering == kFilterLinear ? ScaleARGBRowDown2Linear_MSA
                                            : ScaleARGBRowDown2Box_MSA);
    }
  }
#endif

  if (filtering == kFilterLinear) {
    src_stride = 0;
  }
  for (j = 0; j < dst_height; ++j) {
    ScaleARGBRowDown2(src_argb, src_stride, dst_argb, dst_width);
    src_argb += row_stride;
    dst_argb += dst_stride;
  }
}

// ScaleARGB ARGB, 1/4
// This is an optimized version for scaling down a ARGB to 1/4 of
// its original size.
static void ScaleARGBDown4Box(int src_width,
                              int src_height,
                              int dst_width,
                              int dst_height,
                              int src_stride,
                              int dst_stride,
                              const uint8_t* src_argb,
                              uint8_t* dst_argb,
                              int x,
                              int dx,
                              int y,
                              int dy) {
  int j;
  // Allocate 2 rows of ARGB.
  const int kRowSize = (dst_width * 2 * 4 + 31) & ~31;
  align_buffer_64(row, kRowSize * 2);
  int row_stride = src_stride * (dy >> 16);
  void (*ScaleARGBRowDown2)(const uint8_t* src_argb, ptrdiff_t src_stride,
                            uint8_t* dst_argb, int dst_width) =
      ScaleARGBRowDown2Box_C;
  // Advance to odd row, even column.
  src_argb += (y >> 16) * src_stride + (x >> 16) * 4;
  (void)src_width;
  (void)src_height;
  (void)dx;
  assert(dx == 65536 * 4);      // Test scale factor of 4.
  assert((dy & 0x3ffff) == 0);  // Test vertical scale is multiple of 4.
#if defined(HAS_SCALEARGBROWDOWN2_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ScaleARGBRowDown2 = ScaleARGBRowDown2Box_Any_SSE2;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBRowDown2 = ScaleARGBRowDown2Box_SSE2;
    }
  }
#endif
#if defined(HAS_SCALEARGBROWDOWN2_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBRowDown2 = ScaleARGBRowDown2Box_Any_NEON;
    if (IS_ALIGNED(dst_width, 8)) {
      ScaleARGBRowDown2 = ScaleARGBRowDown2Box_NEON;
    }
  }
#endif

  for (j = 0; j < dst_height; ++j) {
    ScaleARGBRowDown2(src_argb, src_stride, row, dst_width * 2);
    ScaleARGBRowDown2(src_argb + src_stride * 2, src_stride, row + kRowSize,
                      dst_width * 2);
    ScaleARGBRowDown2(row, kRowSize, dst_argb, dst_width);
    src_argb += row_stride;
    dst_argb += dst_stride;
  }
  free_aligned_buffer_64(row);
}

// ScaleARGB ARGB Even
// This is an optimized version for scaling down a ARGB to even
// multiple of its original size.
static void ScaleARGBDownEven(int src_width,
                              int src_height,
                              int dst_width,
                              int dst_height,
                              int src_stride,
                              int dst_stride,
                              const uint8_t* src_argb,
                              uint8_t* dst_argb,
                              int x,
                              int dx,
                              int y,
                              int dy,
                              enum FilterMode filtering) {
  int j;
  int col_step = dx >> 16;
  int row_stride = (dy >> 16) * src_stride;
  void (*ScaleARGBRowDownEven)(const uint8_t* src_argb, ptrdiff_t src_stride,
                               int src_step, uint8_t* dst_argb, int dst_width) =
      filtering ? ScaleARGBRowDownEvenBox_C : ScaleARGBRowDownEven_C;
  (void)src_width;
  (void)src_height;
  assert(IS_ALIGNED(src_width, 2));
  assert(IS_ALIGNED(src_height, 2));
  src_argb += (y >> 16) * src_stride + (x >> 16) * 4;
#if defined(HAS_SCALEARGBROWDOWNEVEN_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ScaleARGBRowDownEven = filtering ? ScaleARGBRowDownEvenBox_Any_SSE2
                                     : ScaleARGBRowDownEven_Any_SSE2;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBRowDownEven =
          filtering ? ScaleARGBRowDownEvenBox_SSE2 : ScaleARGBRowDownEven_SSE2;
    }
  }
#endif
#if defined(HAS_SCALEARGBROWDOWNEVEN_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBRowDownEven = filtering ? ScaleARGBRowDownEvenBox_Any_NEON
                                     : ScaleARGBRowDownEven_Any_NEON;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBRowDownEven =
          filtering ? ScaleARGBRowDownEvenBox_NEON : ScaleARGBRowDownEven_NEON;
    }
  }
#endif
#if defined(HAS_SCALEARGBROWDOWNEVEN_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ScaleARGBRowDownEven = filtering ? ScaleARGBRowDownEvenBox_Any_MSA
                                     : ScaleARGBRowDownEven_Any_MSA;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBRowDownEven =
          filtering ? ScaleARGBRowDownEvenBox_MSA : ScaleARGBRowDownEven_MSA;
    }
  }
#endif

  if (filtering == kFilterLinear) {
    src_stride = 0;
  }
  for (j = 0; j < dst_height; ++j) {
    ScaleARGBRowDownEven(src_argb, src_stride, col_step, dst_argb, dst_width);
    src_argb += row_stride;
    dst_argb += dst_stride;
  }
}

// Scale ARGB down with bilinear interpolation.
static void ScaleARGBBilinearDown(int src_width,
                                  int src_height,
                                  int dst_width,
                                  int dst_height,
                                  int src_stride,
                                  int dst_stride,
                                  const uint8_t* src_argb,
                                  uint8_t* dst_argb,
                                  int x,
                                  int dx,
                                  int y,
                                  int dy,
                                  enum FilterMode filtering) {
  int j;
  void (*InterpolateRow)(uint8_t * dst_argb, const uint8_t* src_argb,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) = InterpolateRow_C;
  void (*ScaleARGBFilterCols)(uint8_t * dst_argb, const uint8_t* src_argb,
                              int dst_width, int x, int dx) =
      (src_width >= 32768) ? ScaleARGBFilterCols64_C : ScaleARGBFilterCols_C;
  int64_t xlast = x + (int64_t)(dst_width - 1) * dx;
  int64_t xl = (dx >= 0) ? x : xlast;
  int64_t xr = (dx >= 0) ? xlast : x;
  int clip_src_width;
  xl = (xl >> 16) & ~3;    // Left edge aligned.
  xr = (xr >> 16) + 1;     // Right most pixel used.  Bilinear uses 2 pixels.
  xr = (xr + 1 + 3) & ~3;  // 1 beyond 4 pixel aligned right most pixel.
  if (xr > src_width) {
    xr = src_width;
  }
  clip_src_width = (int)(xr - xl) * 4;  // Width aligned to 4.
  src_argb += xl * 4;
  x -= (int)(xl << 16);
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(clip_src_width, 16)) {
      InterpolateRow = InterpolateRow_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(clip_src_width, 32)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(clip_src_width, 16)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    InterpolateRow = InterpolateRow_Any_MSA;
    if (IS_ALIGNED(clip_src_width, 32)) {
      InterpolateRow = InterpolateRow_MSA;
    }
  }
#endif
#if defined(HAS_SCALEARGBFILTERCOLS_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && src_width < 32768) {
    ScaleARGBFilterCols = ScaleARGBFilterCols_SSSE3;
  }
#endif
#if defined(HAS_SCALEARGBFILTERCOLS_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBFilterCols = ScaleARGBFilterCols_Any_NEON;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBFilterCols = ScaleARGBFilterCols_NEON;
    }
  }
#endif
#if defined(HAS_SCALEARGBFILTERCOLS_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ScaleARGBFilterCols = ScaleARGBFilterCols_Any_MSA;
    if (IS_ALIGNED(dst_width, 8)) {
      ScaleARGBFilterCols = ScaleARGBFilterCols_MSA;
    }
  }
#endif
  // TODO(fbarchard): Consider not allocating row buffer for kFilterLinear.
  // Allocate a row of ARGB.
  {
    align_buffer_64(row, clip_src_width * 4);

    const int max_y = (src_height - 1) << 16;
    if (y > max_y) {
      y = max_y;
    }
    for (j = 0; j < dst_height; ++j) {
      int yi = y >> 16;
      const uint8_t* src = src_argb + yi * src_stride;
      if (filtering == kFilterLinear) {
        ScaleARGBFilterCols(dst_argb, src, dst_width, x, dx);
      } else {
        int yf = (y >> 8) & 255;
        InterpolateRow(row, src, src_stride, clip_src_width, yf);
        ScaleARGBFilterCols(dst_argb, row, dst_width, x, dx);
      }
      dst_argb += dst_stride;
      y += dy;
      if (y > max_y) {
        y = max_y;
      }
    }
    free_aligned_buffer_64(row);
  }
}

// Scale ARGB up with bilinear interpolation.
static void ScaleARGBBilinearUp(int src_width,
                                int src_height,
                                int dst_width,
                                int dst_height,
                                int src_stride,
                                int dst_stride,
                                const uint8_t* src_argb,
                                uint8_t* dst_argb,
                                int x,
                                int dx,
                                int y,
                                int dy,
                                enum FilterMode filtering) {
  int j;
  void (*InterpolateRow)(uint8_t * dst_argb, const uint8_t* src_argb,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) = InterpolateRow_C;
  void (*ScaleARGBFilterCols)(uint8_t * dst_argb, const uint8_t* src_argb,
                              int dst_width, int x, int dx) =
      filtering ? ScaleARGBFilterCols_C : ScaleARGBCols_C;
  const int max_y = (src_height - 1) << 16;
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(dst_width, 4)) {
      InterpolateRow = InterpolateRow_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(dst_width, 8)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(dst_width, 4)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    InterpolateRow = InterpolateRow_Any_MSA;
    if (IS_ALIGNED(dst_width, 8)) {
      InterpolateRow = InterpolateRow_MSA;
    }
  }
#endif
  if (src_width >= 32768) {
    ScaleARGBFilterCols =
        filtering ? ScaleARGBFilterCols64_C : ScaleARGBCols64_C;
  }
#if defined(HAS_SCALEARGBFILTERCOLS_SSSE3)
  if (filtering && TestCpuFlag(kCpuHasSSSE3) && src_width < 32768) {
    ScaleARGBFilterCols = ScaleARGBFilterCols_SSSE3;
  }
#endif
#if defined(HAS_SCALEARGBFILTERCOLS_NEON)
  if (filtering && TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBFilterCols = ScaleARGBFilterCols_Any_NEON;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBFilterCols = ScaleARGBFilterCols_NEON;
    }
  }
#endif
#if defined(HAS_SCALEARGBFILTERCOLS_MSA)
  if (filtering && TestCpuFlag(kCpuHasMSA)) {
    ScaleARGBFilterCols = ScaleARGBFilterCols_Any_MSA;
    if (IS_ALIGNED(dst_width, 8)) {
      ScaleARGBFilterCols = ScaleARGBFilterCols_MSA;
    }
  }
#endif
#if defined(HAS_SCALEARGBCOLS_SSE2)
  if (!filtering && TestCpuFlag(kCpuHasSSE2) && src_width < 32768) {
    ScaleARGBFilterCols = ScaleARGBCols_SSE2;
  }
#endif
#if defined(HAS_SCALEARGBCOLS_NEON)
  if (!filtering && TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBFilterCols = ScaleARGBCols_Any_NEON;
    if (IS_ALIGNED(dst_width, 8)) {
      ScaleARGBFilterCols = ScaleARGBCols_NEON;
    }
  }
#endif
#if defined(HAS_SCALEARGBCOLS_MSA)
  if (!filtering && TestCpuFlag(kCpuHasMSA)) {
    ScaleARGBFilterCols = ScaleARGBCols_Any_MSA;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBFilterCols = ScaleARGBCols_MSA;
    }
  }
#endif
  if (!filtering && src_width * 2 == dst_width && x < 0x8000) {
    ScaleARGBFilterCols = ScaleARGBColsUp2_C;
#if defined(HAS_SCALEARGBCOLSUP2_SSE2)
    if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(dst_width, 8)) {
      ScaleARGBFilterCols = ScaleARGBColsUp2_SSE2;
    }
#endif
  }

  if (y > max_y) {
    y = max_y;
  }

  {
    int yi = y >> 16;
    const uint8_t* src = src_argb + yi * src_stride;

    // Allocate 2 rows of ARGB.
    const int kRowSize = (dst_width * 4 + 31) & ~31;
    align_buffer_64(row, kRowSize * 2);

    uint8_t* rowptr = row;
    int rowstride = kRowSize;
    int lasty = yi;

    ScaleARGBFilterCols(rowptr, src, dst_width, x, dx);
    if (src_height > 1) {
      src += src_stride;
    }
    ScaleARGBFilterCols(rowptr + rowstride, src, dst_width, x, dx);
    src += src_stride;

    for (j = 0; j < dst_height; ++j) {
      yi = y >> 16;
      if (yi != lasty) {
        if (y > max_y) {
          y = max_y;
          yi = y >> 16;
          src = src_argb + yi * src_stride;
        }
        if (yi != lasty) {
          ScaleARGBFilterCols(rowptr, src, dst_width, x, dx);
          rowptr += rowstride;
          rowstride = -rowstride;
          lasty = yi;
          src += src_stride;
        }
      }
      if (filtering == kFilterLinear) {
        InterpolateRow(dst_argb, rowptr, 0, dst_width * 4, 0);
      } else {
        int yf = (y >> 8) & 255;
        InterpolateRow(dst_argb, rowptr, rowstride, dst_width * 4, yf);
      }
      dst_argb += dst_stride;
      y += dy;
    }
    free_aligned_buffer_64(row);
  }
}

#ifdef YUVSCALEUP
// Scale YUV to ARGB up with bilinear interpolation.
static void ScaleYUVToARGBBilinearUp(int src_width,
                                     int src_height,
                                     int dst_width,
                                     int dst_height,
                                     int src_stride_y,
                                     int src_stride_u,
                                     int src_stride_v,
                                     int dst_stride_argb,
                                     const uint8_t* src_y,
                                     const uint8_t* src_u,
                                     const uint8_t* src_v,
                                     uint8_t* dst_argb,
                                     int x,
                                     int dx,
                                     int y,
                                     int dy,
                                     enum FilterMode filtering) {
  int j;
  void (*I422ToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                        const uint8_t* v_buf, uint8_t* rgb_buf, int width) =
      I422ToARGBRow_C;
#if defined(HAS_I422TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToARGBRow = I422ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(src_width, 8)) {
      I422ToARGBRow = I422ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToARGBRow = I422ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(src_width, 16)) {
      I422ToARGBRow = I422ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToARGBRow = I422ToARGBRow_Any_NEON;
    if (IS_ALIGNED(src_width, 8)) {
      I422ToARGBRow = I422ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToARGBRow = I422ToARGBRow_Any_MSA;
    if (IS_ALIGNED(src_width, 8)) {
      I422ToARGBRow = I422ToARGBRow_MSA;
    }
  }
#endif

  void (*InterpolateRow)(uint8_t * dst_argb, const uint8_t* src_argb,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) = InterpolateRow_C;
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(dst_width, 4)) {
      InterpolateRow = InterpolateRow_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(dst_width, 8)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(dst_width, 4)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    InterpolateRow = InterpolateRow_Any_MSA;
    if (IS_ALIGNED(dst_width, 8)) {
      InterpolateRow = InterpolateRow_MSA;
    }
  }
#endif

  void (*ScaleARGBFilterCols)(uint8_t * dst_argb, const uint8_t* src_argb,
                              int dst_width, int x, int dx) =
      filtering ? ScaleARGBFilterCols_C : ScaleARGBCols_C;
  if (src_width >= 32768) {
    ScaleARGBFilterCols =
        filtering ? ScaleARGBFilterCols64_C : ScaleARGBCols64_C;
  }
#if defined(HAS_SCALEARGBFILTERCOLS_SSSE3)
  if (filtering && TestCpuFlag(kCpuHasSSSE3) && src_width < 32768) {
    ScaleARGBFilterCols = ScaleARGBFilterCols_SSSE3;
  }
#endif
#if defined(HAS_SCALEARGBFILTERCOLS_NEON)
  if (filtering && TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBFilterCols = ScaleARGBFilterCols_Any_NEON;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBFilterCols = ScaleARGBFilterCols_NEON;
    }
  }
#endif
#if defined(HAS_SCALEARGBFILTERCOLS_MSA)
  if (filtering && TestCpuFlag(kCpuHasMSA)) {
    ScaleARGBFilterCols = ScaleARGBFilterCols_Any_MSA;
    if (IS_ALIGNED(dst_width, 8)) {
      ScaleARGBFilterCols = ScaleARGBFilterCols_MSA;
    }
  }
#endif
#if defined(HAS_SCALEARGBCOLS_SSE2)
  if (!filtering && TestCpuFlag(kCpuHasSSE2) && src_width < 32768) {
    ScaleARGBFilterCols = ScaleARGBCols_SSE2;
  }
#endif
#if defined(HAS_SCALEARGBCOLS_NEON)
  if (!filtering && TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBFilterCols = ScaleARGBCols_Any_NEON;
    if (IS_ALIGNED(dst_width, 8)) {
      ScaleARGBFilterCols = ScaleARGBCols_NEON;
    }
  }
#endif
#if defined(HAS_SCALEARGBCOLS_MSA)
  if (!filtering && TestCpuFlag(kCpuHasMSA)) {
    ScaleARGBFilterCols = ScaleARGBCols_Any_MSA;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBFilterCols = ScaleARGBCols_MSA;
    }
  }
#endif
  if (!filtering && src_width * 2 == dst_width && x < 0x8000) {
    ScaleARGBFilterCols = ScaleARGBColsUp2_C;
#if defined(HAS_SCALEARGBCOLSUP2_SSE2)
    if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(dst_width, 8)) {
      ScaleARGBFilterCols = ScaleARGBColsUp2_SSE2;
    }
#endif
  }

  const int max_y = (src_height - 1) << 16;
  if (y > max_y) {
    y = max_y;
  }
  const int kYShift = 1;  // Shift Y by 1 to convert Y plane to UV coordinate.
  int yi = y >> 16;
  int uv_yi = yi >> kYShift;
  const uint8_t* src_row_y = src_y + yi * src_stride_y;
  const uint8_t* src_row_u = src_u + uv_yi * src_stride_u;
  const uint8_t* src_row_v = src_v + uv_yi * src_stride_v;

  // Allocate 2 rows of ARGB.
  const int kRowSize = (dst_width * 4 + 31) & ~31;
  align_buffer_64(row, kRowSize * 2);

  // Allocate 1 row of ARGB for source conversion.
  align_buffer_64(argb_row, src_width * 4);

  uint8_t* rowptr = row;
  int rowstride = kRowSize;
  int lasty = yi;

  // TODO(fbarchard): Convert first 2 rows of YUV to ARGB.
  ScaleARGBFilterCols(rowptr, src_row_y, dst_width, x, dx);
  if (src_height > 1) {
    src_row_y += src_stride_y;
    if (yi & 1) {
      src_row_u += src_stride_u;
      src_row_v += src_stride_v;
    }
  }
  ScaleARGBFilterCols(rowptr + rowstride, src_row_y, dst_width, x, dx);
  if (src_height > 2) {
    src_row_y += src_stride_y;
    if (!(yi & 1)) {
      src_row_u += src_stride_u;
      src_row_v += src_stride_v;
    }
  }

  for (j = 0; j < dst_height; ++j) {
    yi = y >> 16;
    if (yi != lasty) {
      if (y > max_y) {
        y = max_y;
        yi = y >> 16;
        uv_yi = yi >> kYShift;
        src_row_y = src_y + yi * src_stride_y;
        src_row_u = src_u + uv_yi * src_stride_u;
        src_row_v = src_v + uv_yi * src_stride_v;
      }
      if (yi != lasty) {
        // TODO(fbarchard): Convert the clipped region of row.
        I422ToARGBRow(src_row_y, src_row_u, src_row_v, argb_row, src_width);
        ScaleARGBFilterCols(rowptr, argb_row, dst_width, x, dx);
        rowptr += rowstride;
        rowstride = -rowstride;
        lasty = yi;
        src_row_y += src_stride_y;
        if (yi & 1) {
          src_row_u += src_stride_u;
          src_row_v += src_stride_v;
        }
      }
    }
    if (filtering == kFilterLinear) {
      InterpolateRow(dst_argb, rowptr, 0, dst_width * 4, 0);
    } else {
      int yf = (y >> 8) & 255;
      InterpolateRow(dst_argb, rowptr, rowstride, dst_width * 4, yf);
    }
    dst_argb += dst_stride_argb;
    y += dy;
  }
  free_aligned_buffer_64(row);
  free_aligned_buffer_64(row_argb);
}
#endif

// Scale ARGB to/from any dimensions, without interpolation.
// Fixed point math is used for performance: The upper 16 bits
// of x and dx is the integer part of the source position and
// the lower 16 bits are the fixed decimal part.

static void ScaleARGBSimple(int src_width,
                            int src_height,
                            int dst_width,
                            int dst_height,
                            int src_stride,
                            int dst_stride,
                            const uint8_t* src_argb,
                            uint8_t* dst_argb,
                            int x,
                            int dx,
                            int y,
                            int dy) {
  int j;
  void (*ScaleARGBCols)(uint8_t * dst_argb, const uint8_t* src_argb,
                        int dst_width, int x, int dx) =
      (src_width >= 32768) ? ScaleARGBCols64_C : ScaleARGBCols_C;
  (void)src_height;
#if defined(HAS_SCALEARGBCOLS_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && src_width < 32768) {
    ScaleARGBCols = ScaleARGBCols_SSE2;
  }
#endif
#if defined(HAS_SCALEARGBCOLS_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBCols = ScaleARGBCols_Any_NEON;
    if (IS_ALIGNED(dst_width, 8)) {
      ScaleARGBCols = ScaleARGBCols_NEON;
    }
  }
#endif
#if defined(HAS_SCALEARGBCOLS_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ScaleARGBCols = ScaleARGBCols_Any_MSA;
    if (IS_ALIGNED(dst_width, 4)) {
      ScaleARGBCols = ScaleARGBCols_MSA;
    }
  }
#endif
  if (src_width * 2 == dst_width && x < 0x8000) {
    ScaleARGBCols = ScaleARGBColsUp2_C;
#if defined(HAS_SCALEARGBCOLSUP2_SSE2)
    if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(dst_width, 8)) {
      ScaleARGBCols = ScaleARGBColsUp2_SSE2;
    }
#endif
  }

  for (j = 0; j < dst_height; ++j) {
    ScaleARGBCols(dst_argb, src_argb + (y >> 16) * src_stride, dst_width, x,
                  dx);
    dst_argb += dst_stride;
    y += dy;
  }
}

// ScaleARGB a ARGB.
// This function in turn calls a scaling function
// suitable for handling the desired resolutions.
static void ScaleARGB(const uint8_t* src,
                      int src_stride,
                      int src_width,
                      int src_height,
                      uint8_t* dst,
                      int dst_stride,
                      int dst_width,
                      int dst_height,
                      int clip_x,
                      int clip_y,
                      int clip_width,
                      int clip_height,
                      enum FilterMode filtering) {
  // Initial source x/y coordinate and step values as 16.16 fixed point.
  int x = 0;
  int y = 0;
  int dx = 0;
  int dy = 0;
  // ARGB does not support box filter yet, but allow the user to pass it.
  // Simplify filtering when possible.
  filtering = ScaleFilterReduce(src_width, src_height, dst_width, dst_height,
                                filtering);

  // Negative src_height means invert the image.
  if (src_height < 0) {
    src_height = -src_height;
    src = src + (src_height - 1) * src_stride;
    src_stride = -src_stride;
  }
  ScaleSlope(src_width, src_height, dst_width, dst_height, filtering, &x, &y,
             &dx, &dy);
  src_width = Abs(src_width);
  if (clip_x) {
    int64_t clipf = (int64_t)(clip_x)*dx;
    x += (clipf & 0xffff);
    src += (clipf >> 16) * 4;
    dst += clip_x * 4;
  }
  if (clip_y) {
    int64_t clipf = (int64_t)(clip_y)*dy;
    y += (clipf & 0xffff);
    src += (clipf >> 16) * src_stride;
    dst += clip_y * dst_stride;
  }

  // Special case for integer step values.
  if (((dx | dy) & 0xffff) == 0) {
    if (!dx || !dy) {  // 1 pixel wide and/or tall.
      filtering = kFilterNone;
    } else {
      // Optimized even scale down. ie 2, 4, 6, 8, 10x.
      if (!(dx & 0x10000) && !(dy & 0x10000)) {
        if (dx == 0x20000) {
          // Optimized 1/2 downsample.
          ScaleARGBDown2(src_width, src_height, clip_width, clip_height,
                         src_stride, dst_stride, src, dst, x, dx, y, dy,
                         filtering);
          return;
        }
        if (dx == 0x40000 && filtering == kFilterBox) {
          // Optimized 1/4 box downsample.
          ScaleARGBDown4Box(src_width, src_height, clip_width, clip_height,
                            src_stride, dst_stride, src, dst, x, dx, y, dy);
          return;
        }
        ScaleARGBDownEven(src_width, src_height, clip_width, clip_height,
                          src_stride, dst_stride, src, dst, x, dx, y, dy,
                          filtering);
        return;
      }
      // Optimized odd scale down. ie 3, 5, 7, 9x.
      if ((dx & 0x10000) && (dy & 0x10000)) {
        filtering = kFilterNone;
        if (dx == 0x10000 && dy == 0x10000) {
          // Straight copy.
          ARGBCopy(src + (y >> 16) * src_stride + (x >> 16) * 4, src_stride,
                   dst, dst_stride, clip_width, clip_height);
          return;
        }
      }
    }
  }
  if (dx == 0x10000 && (x & 0xffff) == 0) {
    // Arbitrary scale vertically, but unscaled vertically.
    ScalePlaneVertical(src_height, clip_width, clip_height, src_stride,
                       dst_stride, src, dst, x, y, dy, 4, filtering);
    return;
  }
  if (filtering && dy < 65536) {
    ScaleARGBBilinearUp(src_width, src_height, clip_width, clip_height,
                        src_stride, dst_stride, src, dst, x, dx, y, dy,
                        filtering);
    return;
  }
  if (filtering) {
    ScaleARGBBilinearDown(src_width, src_height, clip_width, clip_height,
                          src_stride, dst_stride, src, dst, x, dx, y, dy,
                          filtering);
    return;
  }
  ScaleARGBSimple(src_width, src_height, clip_width, clip_height, src_stride,
                  dst_stride, src, dst, x, dx, y, dy);
}

LIBYUV_API
int ARGBScaleClip(const uint8_t* src_argb,
                  int src_stride_argb,
                  int src_width,
                  int src_height,
                  uint8_t* dst_argb,
                  int dst_stride_argb,
                  int dst_width,
                  int dst_height,
                  int clip_x,
                  int clip_y,
                  int clip_width,
                  int clip_height,
                  enum FilterMode filtering) {
  if (!src_argb || src_width == 0 || src_height == 0 || !dst_argb ||
      dst_width <= 0 || dst_height <= 0 || clip_x < 0 || clip_y < 0 ||
      clip_width > 32768 || clip_height > 32768 ||
      (clip_x + clip_width) > dst_width ||
      (clip_y + clip_height) > dst_height) {
    return -1;
  }
  ScaleARGB(src_argb, src_stride_argb, src_width, src_height, dst_argb,
            dst_stride_argb, dst_width, dst_height, clip_x, clip_y, clip_width,
            clip_height, filtering);
  return 0;
}

// Scale an ARGB image.
LIBYUV_API
int ARGBScale(const uint8_t* src_argb,
              int src_stride_argb,
              int src_width,
              int src_height,
              uint8_t* dst_argb,
              int dst_stride_argb,
              int dst_width,
              int dst_height,
              enum FilterMode filtering) {
  if (!src_argb || src_width == 0 || src_height == 0 || src_width > 32768 ||
      src_height > 32768 || !dst_argb || dst_width <= 0 || dst_height <= 0) {
    return -1;
  }
  ScaleARGB(src_argb, src_stride_argb, src_width, src_height, dst_argb,
            dst_stride_argb, dst_width, dst_height, 0, 0, dst_width, dst_height,
            filtering);
  return 0;
}

// Scale with YUV conversion to ARGB and clipping.
LIBYUV_API
int YUVToARGBScaleClip(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_u,
                       int src_stride_u,
                       const uint8_t* src_v,
                       int src_stride_v,
                       uint32_t src_fourcc,
                       int src_width,
                       int src_height,
                       uint8_t* dst_argb,
                       int dst_stride_argb,
                       uint32_t dst_fourcc,
                       int dst_width,
                       int dst_height,
                       int clip_x,
                       int clip_y,
                       int clip_width,
                       int clip_height,
                       enum FilterMode filtering) {
  uint8_t* argb_buffer = (uint8_t*)malloc(src_width * src_height * 4);
  int r;
  (void)src_fourcc;  // TODO(fbarchard): implement and/or assert.
  (void)dst_fourcc;
  I420ToARGB(src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
             argb_buffer, src_width * 4, src_width, src_height);

  r = ARGBScaleClip(argb_buffer, src_width * 4, src_width, src_height, dst_argb,
                    dst_stride_argb, dst_width, dst_height, clip_x, clip_y,
                    clip_width, clip_height, filtering);
  free(argb_buffer);
  return r;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
