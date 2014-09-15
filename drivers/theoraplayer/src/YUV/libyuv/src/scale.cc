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
#include "libyuv/planar_functions.h"  // For CopyPlane
#include "libyuv/row.h"
#include "libyuv/scale_row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Remove this macro if OVERREAD is safe.
#define AVOID_OVERREAD 1

static __inline int Abs(int v) {
  return v >= 0 ? v : -v;
}

#define SUBSAMPLE(v, a, s) (v < 0) ? (-((-v + a) >> s)) : ((v + a) >> s)

// Scale plane, 1/2
// This is an optimized version for scaling down a plane to 1/2 of
// its original size.

static void ScalePlaneDown2(int src_width, int src_height,
                            int dst_width, int dst_height,
                            int src_stride, int dst_stride,
                            const uint8* src_ptr, uint8* dst_ptr,
                            enum FilterMode filtering) {
  int y;
  void (*ScaleRowDown2)(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width) =
    filtering == kFilterNone ? ScaleRowDown2_C :
        (filtering == kFilterLinear ? ScaleRowDown2Linear_C :
        ScaleRowDown2Box_C);
  int row_stride = src_stride << 1;
  if (!filtering) {
    src_ptr += src_stride;  // Point to odd rows.
    src_stride = 0;
  }

#if defined(HAS_SCALEROWDOWN2_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(dst_width, 16)) {
    ScaleRowDown2 = filtering ? ScaleRowDown2Box_NEON : ScaleRowDown2_NEON;
  }
#elif defined(HAS_SCALEROWDOWN2_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(dst_width, 16)) {
    ScaleRowDown2 = filtering == kFilterNone ? ScaleRowDown2_Unaligned_SSE2 :
        (filtering == kFilterLinear ? ScaleRowDown2Linear_Unaligned_SSE2 :
        ScaleRowDown2Box_Unaligned_SSE2);
    if (IS_ALIGNED(src_ptr, 16) &&
        IS_ALIGNED(src_stride, 16) && IS_ALIGNED(row_stride, 16) &&
        IS_ALIGNED(dst_ptr, 16) && IS_ALIGNED(dst_stride, 16)) {
      ScaleRowDown2 = filtering == kFilterNone ? ScaleRowDown2_SSE2 :
          (filtering == kFilterLinear ? ScaleRowDown2Linear_SSE2 :
          ScaleRowDown2Box_SSE2);
    }
  }
#elif defined(HAS_SCALEROWDOWN2_MIPS_DSPR2)
  if (TestCpuFlag(kCpuHasMIPS_DSPR2) && IS_ALIGNED(src_ptr, 4) &&
      IS_ALIGNED(src_stride, 4) && IS_ALIGNED(row_stride, 4) &&
      IS_ALIGNED(dst_ptr, 4) && IS_ALIGNED(dst_stride, 4)) {
    ScaleRowDown2 = filtering ?
        ScaleRowDown2Box_MIPS_DSPR2 : ScaleRowDown2_MIPS_DSPR2;
  }
#endif

  if (filtering == kFilterLinear) {
    src_stride = 0;
  }
  // TODO(fbarchard): Loop through source height to allow odd height.
  for (y = 0; y < dst_height; ++y) {
    ScaleRowDown2(src_ptr, src_stride, dst_ptr, dst_width);
    src_ptr += row_stride;
    dst_ptr += dst_stride;
  }
}

// Scale plane, 1/4
// This is an optimized version for scaling down a plane to 1/4 of
// its original size.

static void ScalePlaneDown4(int src_width, int src_height,
                            int dst_width, int dst_height,
                            int src_stride, int dst_stride,
                            const uint8* src_ptr, uint8* dst_ptr,
                            enum FilterMode filtering) {
  int y;
  void (*ScaleRowDown4)(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width) =
      filtering ? ScaleRowDown4Box_C : ScaleRowDown4_C;
  int row_stride = src_stride << 2;
  if (!filtering) {
    src_ptr += src_stride * 2;  // Point to row 2.
    src_stride = 0;
  }
#if defined(HAS_SCALEROWDOWN4_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(dst_width, 8)) {
    ScaleRowDown4 = filtering ? ScaleRowDown4Box_NEON : ScaleRowDown4_NEON;
  }
#elif defined(HAS_SCALEROWDOWN4_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) &&
      IS_ALIGNED(dst_width, 8) && IS_ALIGNED(row_stride, 16) &&
      IS_ALIGNED(src_ptr, 16) && IS_ALIGNED(src_stride, 16)) {
    ScaleRowDown4 = filtering ? ScaleRowDown4Box_SSE2 : ScaleRowDown4_SSE2;
  }
#elif defined(HAS_SCALEROWDOWN4_MIPS_DSPR2)
  if (TestCpuFlag(kCpuHasMIPS_DSPR2) && IS_ALIGNED(row_stride, 4) &&
      IS_ALIGNED(src_ptr, 4) && IS_ALIGNED(src_stride, 4) &&
      IS_ALIGNED(dst_ptr, 4) && IS_ALIGNED(dst_stride, 4)) {
    ScaleRowDown4 = filtering ?
        ScaleRowDown4Box_MIPS_DSPR2 : ScaleRowDown4_MIPS_DSPR2;
  }
#endif

  if (filtering == kFilterLinear) {
    src_stride = 0;
  }
  for (y = 0; y < dst_height; ++y) {
    ScaleRowDown4(src_ptr, src_stride, dst_ptr, dst_width);
    src_ptr += row_stride;
    dst_ptr += dst_stride;
  }
}

// Scale plane down, 3/4

static void ScalePlaneDown34(int src_width, int src_height,
                             int dst_width, int dst_height,
                             int src_stride, int dst_stride,
                             const uint8* src_ptr, uint8* dst_ptr,
                             enum FilterMode filtering) {
  int y;
  void (*ScaleRowDown34_0)(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width);
  void (*ScaleRowDown34_1)(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width);
  const int filter_stride = (filtering == kFilterLinear) ? 0 : src_stride;
  assert(dst_width % 3 == 0);
  if (!filtering) {
    ScaleRowDown34_0 = ScaleRowDown34_C;
    ScaleRowDown34_1 = ScaleRowDown34_C;
  } else {
    ScaleRowDown34_0 = ScaleRowDown34_0_Box_C;
    ScaleRowDown34_1 = ScaleRowDown34_1_Box_C;
  }
#if defined(HAS_SCALEROWDOWN34_NEON)
  if (TestCpuFlag(kCpuHasNEON) && (dst_width % 24 == 0)) {
    if (!filtering) {
      ScaleRowDown34_0 = ScaleRowDown34_NEON;
      ScaleRowDown34_1 = ScaleRowDown34_NEON;
    } else {
      ScaleRowDown34_0 = ScaleRowDown34_0_Box_NEON;
      ScaleRowDown34_1 = ScaleRowDown34_1_Box_NEON;
    }
  }
#endif
#if defined(HAS_SCALEROWDOWN34_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && (dst_width % 24 == 0) &&
      IS_ALIGNED(src_ptr, 16) && IS_ALIGNED(src_stride, 16)) {
    if (!filtering) {
      ScaleRowDown34_0 = ScaleRowDown34_SSSE3;
      ScaleRowDown34_1 = ScaleRowDown34_SSSE3;
    } else {
      ScaleRowDown34_0 = ScaleRowDown34_0_Box_SSSE3;
      ScaleRowDown34_1 = ScaleRowDown34_1_Box_SSSE3;
    }
  }
#endif
#if defined(HAS_SCALEROWDOWN34_MIPS_DSPR2)
  if (TestCpuFlag(kCpuHasMIPS_DSPR2) && (dst_width % 24 == 0) &&
      IS_ALIGNED(src_ptr, 4) && IS_ALIGNED(src_stride, 4) &&
      IS_ALIGNED(dst_ptr, 4) && IS_ALIGNED(dst_stride, 4)) {
    if (!filtering) {
      ScaleRowDown34_0 = ScaleRowDown34_MIPS_DSPR2;
      ScaleRowDown34_1 = ScaleRowDown34_MIPS_DSPR2;
    } else {
      ScaleRowDown34_0 = ScaleRowDown34_0_Box_MIPS_DSPR2;
      ScaleRowDown34_1 = ScaleRowDown34_1_Box_MIPS_DSPR2;
    }
  }
#endif

  for (y = 0; y < dst_height - 2; y += 3) {
    ScaleRowDown34_0(src_ptr, filter_stride, dst_ptr, dst_width);
    src_ptr += src_stride;
    dst_ptr += dst_stride;
    ScaleRowDown34_1(src_ptr, filter_stride, dst_ptr, dst_width);
    src_ptr += src_stride;
    dst_ptr += dst_stride;
    ScaleRowDown34_0(src_ptr + src_stride, -filter_stride,
                     dst_ptr, dst_width);
    src_ptr += src_stride * 2;
    dst_ptr += dst_stride;
  }

  // Remainder 1 or 2 rows with last row vertically unfiltered
  if ((dst_height % 3) == 2) {
    ScaleRowDown34_0(src_ptr, filter_stride, dst_ptr, dst_width);
    src_ptr += src_stride;
    dst_ptr += dst_stride;
    ScaleRowDown34_1(src_ptr, 0, dst_ptr, dst_width);
  } else if ((dst_height % 3) == 1) {
    ScaleRowDown34_0(src_ptr, 0, dst_ptr, dst_width);
  }
}


// Scale plane, 3/8
// This is an optimized version for scaling down a plane to 3/8
// of its original size.
//
// Uses box filter arranges like this
// aaabbbcc -> abc
// aaabbbcc    def
// aaabbbcc    ghi
// dddeeeff
// dddeeeff
// dddeeeff
// ggghhhii
// ggghhhii
// Boxes are 3x3, 2x3, 3x2 and 2x2

static void ScalePlaneDown38(int src_width, int src_height,
                             int dst_width, int dst_height,
                             int src_stride, int dst_stride,
                             const uint8* src_ptr, uint8* dst_ptr,
                             enum FilterMode filtering) {
  int y;
  void (*ScaleRowDown38_3)(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width);
  void (*ScaleRowDown38_2)(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width);
  const int filter_stride = (filtering == kFilterLinear) ? 0 : src_stride;
  assert(dst_width % 3 == 0);
  if (!filtering) {
    ScaleRowDown38_3 = ScaleRowDown38_C;
    ScaleRowDown38_2 = ScaleRowDown38_C;
  } else {
    ScaleRowDown38_3 = ScaleRowDown38_3_Box_C;
    ScaleRowDown38_2 = ScaleRowDown38_2_Box_C;
  }
#if defined(HAS_SCALEROWDOWN38_NEON)
  if (TestCpuFlag(kCpuHasNEON) && (dst_width % 12 == 0)) {
    if (!filtering) {
      ScaleRowDown38_3 = ScaleRowDown38_NEON;
      ScaleRowDown38_2 = ScaleRowDown38_NEON;
    } else {
      ScaleRowDown38_3 = ScaleRowDown38_3_Box_NEON;
      ScaleRowDown38_2 = ScaleRowDown38_2_Box_NEON;
    }
  }
#elif defined(HAS_SCALEROWDOWN38_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && (dst_width % 24 == 0) &&
      IS_ALIGNED(src_ptr, 16) && IS_ALIGNED(src_stride, 16)) {
    if (!filtering) {
      ScaleRowDown38_3 = ScaleRowDown38_SSSE3;
      ScaleRowDown38_2 = ScaleRowDown38_SSSE3;
    } else {
      ScaleRowDown38_3 = ScaleRowDown38_3_Box_SSSE3;
      ScaleRowDown38_2 = ScaleRowDown38_2_Box_SSSE3;
    }
  }
#elif defined(HAS_SCALEROWDOWN38_MIPS_DSPR2)
  if (TestCpuFlag(kCpuHasMIPS_DSPR2) && (dst_width % 12 == 0) &&
      IS_ALIGNED(src_ptr, 4) && IS_ALIGNED(src_stride, 4) &&
      IS_ALIGNED(dst_ptr, 4) && IS_ALIGNED(dst_stride, 4)) {
    if (!filtering) {
      ScaleRowDown38_3 = ScaleRowDown38_MIPS_DSPR2;
      ScaleRowDown38_2 = ScaleRowDown38_MIPS_DSPR2;
    } else {
      ScaleRowDown38_3 = ScaleRowDown38_3_Box_MIPS_DSPR2;
      ScaleRowDown38_2 = ScaleRowDown38_2_Box_MIPS_DSPR2;
    }
  }
#endif

  for (y = 0; y < dst_height - 2; y += 3) {
    ScaleRowDown38_3(src_ptr, filter_stride, dst_ptr, dst_width);
    src_ptr += src_stride * 3;
    dst_ptr += dst_stride;
    ScaleRowDown38_3(src_ptr, filter_stride, dst_ptr, dst_width);
    src_ptr += src_stride * 3;
    dst_ptr += dst_stride;
    ScaleRowDown38_2(src_ptr, filter_stride, dst_ptr, dst_width);
    src_ptr += src_stride * 2;
    dst_ptr += dst_stride;
  }

  // Remainder 1 or 2 rows with last row vertically unfiltered
  if ((dst_height % 3) == 2) {
    ScaleRowDown38_3(src_ptr, filter_stride, dst_ptr, dst_width);
    src_ptr += src_stride * 3;
    dst_ptr += dst_stride;
    ScaleRowDown38_3(src_ptr, 0, dst_ptr, dst_width);
  } else if ((dst_height % 3) == 1) {
    ScaleRowDown38_3(src_ptr, 0, dst_ptr, dst_width);
  }
}

static __inline uint32 SumBox(int iboxwidth, int iboxheight,
                              ptrdiff_t src_stride, const uint8* src_ptr) {
  uint32 sum = 0u;
  int y;
  assert(iboxwidth > 0);
  assert(iboxheight > 0);
  for (y = 0; y < iboxheight; ++y) {
    int x;
    for (x = 0; x < iboxwidth; ++x) {
      sum += src_ptr[x];
    }
    src_ptr += src_stride;
  }
  return sum;
}

static void ScalePlaneBoxRow_C(int dst_width, int boxheight,
                               int x, int dx, ptrdiff_t src_stride,
                               const uint8* src_ptr, uint8* dst_ptr) {
  int i;
  int boxwidth;
  for (i = 0; i < dst_width; ++i) {
    int ix = x >> 16;
    x += dx;
    boxwidth = (x >> 16) - ix;
    *dst_ptr++ = SumBox(boxwidth, boxheight, src_stride, src_ptr + ix) /
        (boxwidth * boxheight);
  }
}

static __inline uint32 SumPixels(int iboxwidth, const uint16* src_ptr) {
  uint32 sum = 0u;
  int x;
  assert(iboxwidth > 0);
  for (x = 0; x < iboxwidth; ++x) {
    sum += src_ptr[x];
  }
  return sum;
}

static void ScaleAddCols2_C(int dst_width, int boxheight, int x, int dx,
                            const uint16* src_ptr, uint8* dst_ptr) {
  int i;
  int scaletbl[2];
  int minboxwidth = (dx >> 16);
  int* scaleptr = scaletbl - minboxwidth;
  int boxwidth;
  scaletbl[0] = 65536 / (minboxwidth * boxheight);
  scaletbl[1] = 65536 / ((minboxwidth + 1) * boxheight);
  for (i = 0; i < dst_width; ++i) {
    int ix = x >> 16;
    x += dx;
    boxwidth = (x >> 16) - ix;
    *dst_ptr++ = SumPixels(boxwidth, src_ptr + ix) * scaleptr[boxwidth] >> 16;
  }
}

static void ScaleAddCols1_C(int dst_width, int boxheight, int x, int dx,
                            const uint16* src_ptr, uint8* dst_ptr) {
  int boxwidth = (dx >> 16);
  int scaleval = 65536 / (boxwidth * boxheight);
  int i;
  for (i = 0; i < dst_width; ++i) {
    *dst_ptr++ = SumPixels(boxwidth, src_ptr + x) * scaleval >> 16;
    x += boxwidth;
  }
}

// Scale plane down to any dimensions, with interpolation.
// (boxfilter).
//
// Same method as SimpleScale, which is fixed point, outputting
// one pixel of destination using fixed point (16.16) to step
// through source, sampling a box of pixel with simple
// averaging.
static void ScalePlaneBox(int src_width, int src_height,
                          int dst_width, int dst_height,
                          int src_stride, int dst_stride,
                          const uint8* src_ptr, uint8* dst_ptr) {
  int j;
  // Initial source x/y coordinate and step values as 16.16 fixed point.
  int x = 0;
  int y = 0;
  int dx = 0;
  int dy = 0;
  const int max_y = (src_height << 16);
  ScaleSlope(src_width, src_height, dst_width, dst_height, kFilterBox,
             &x, &y, &dx, &dy);
  src_width = Abs(src_width);
  // TODO(fbarchard): Remove this and make AddRows handle boxheight 1.
  if (!IS_ALIGNED(src_width, 16) || dst_height * 2 > src_height) {
    uint8* dst = dst_ptr;
    int j;
    for (j = 0; j < dst_height; ++j) {
      int boxheight;
      int iy = y >> 16;
      const uint8* src = src_ptr + iy * src_stride;
      y += dy;
      if (y > max_y) {
        y = max_y;
      }
      boxheight = (y >> 16) - iy;
      ScalePlaneBoxRow_C(dst_width, boxheight,
                         x, dx, src_stride,
                         src, dst);
      dst += dst_stride;
    }
    return;
  }
  {
    // Allocate a row buffer of uint16.
    align_buffer_64(row16, src_width * 2);
    void (*ScaleAddCols)(int dst_width, int boxheight, int x, int dx,
        const uint16* src_ptr, uint8* dst_ptr) =
        (dx & 0xffff) ? ScaleAddCols2_C: ScaleAddCols1_C;
    void (*ScaleAddRows)(const uint8* src_ptr, ptrdiff_t src_stride,
        uint16* dst_ptr, int src_width, int src_height) = ScaleAddRows_C;

#if defined(HAS_SCALEADDROWS_SSE2)
    if (TestCpuFlag(kCpuHasSSE2) &&
#ifdef AVOID_OVERREAD
        IS_ALIGNED(src_width, 16) &&
#endif
        IS_ALIGNED(src_ptr, 16) && IS_ALIGNED(src_stride, 16)) {
      ScaleAddRows = ScaleAddRows_SSE2;
    }
#endif

    for (j = 0; j < dst_height; ++j) {
      int boxheight;
      int iy = y >> 16;
      const uint8* src = src_ptr + iy * src_stride;
      y += dy;
      if (y > (src_height << 16)) {
        y = (src_height << 16);
      }
      boxheight = (y >> 16) - iy;
      ScaleAddRows(src, src_stride, (uint16*)(row16),
                 src_width, boxheight);
      ScaleAddCols(dst_width, boxheight, x, dx, (uint16*)(row16),
                 dst_ptr);
      dst_ptr += dst_stride;
    }
    free_aligned_buffer_64(row16);
  }
}

// Scale plane down with bilinear interpolation.
void ScalePlaneBilinearDown(int src_width, int src_height,
                            int dst_width, int dst_height,
                            int src_stride, int dst_stride,
                            const uint8* src_ptr, uint8* dst_ptr,
                            enum FilterMode filtering) {
  // Initial source x/y coordinate and step values as 16.16 fixed point.
  int x = 0;
  int y = 0;
  int dx = 0;
  int dy = 0;
  // TODO(fbarchard): Consider not allocating row buffer for kFilterLinear.
  // Allocate a row buffer.
  align_buffer_64(row, src_width);

  const int max_y = (src_height - 1) << 16;
  int j;
  void (*ScaleFilterCols)(uint8* dst_ptr, const uint8* src_ptr,
      int dst_width, int x, int dx) =
      (src_width >= 32768) ? ScaleFilterCols64_C : ScaleFilterCols_C;
  void (*InterpolateRow)(uint8* dst_ptr, const uint8* src_ptr,
      ptrdiff_t src_stride, int dst_width, int source_y_fraction) =
      InterpolateRow_C;
  ScaleSlope(src_width, src_height, dst_width, dst_height, filtering,
             &x, &y, &dx, &dy);
  src_width = Abs(src_width);

#if defined(HAS_INTERPOLATEROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && src_width >= 16) {
    InterpolateRow = InterpolateRow_Any_SSE2;
    if (IS_ALIGNED(src_width, 16)) {
      InterpolateRow = InterpolateRow_Unaligned_SSE2;
      if (IS_ALIGNED(src_ptr, 16) && IS_ALIGNED(src_stride, 16)) {
        InterpolateRow = InterpolateRow_SSE2;
      }
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && src_width >= 16) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(src_width, 16)) {
      InterpolateRow = InterpolateRow_Unaligned_SSSE3;
      if (IS_ALIGNED(src_ptr, 16) && IS_ALIGNED(src_stride, 16)) {
        InterpolateRow = InterpolateRow_SSSE3;
      }
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2) && src_width >= 32) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(src_width, 32)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && src_width >= 16) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(src_width, 16)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_MIPS_DSPR2)
  if (TestCpuFlag(kCpuHasMIPS_DSPR2) && src_width >= 4) {
    InterpolateRow = InterpolateRow_Any_MIPS_DSPR2;
    if (IS_ALIGNED(src_width, 4)) {
      InterpolateRow = InterpolateRow_MIPS_DSPR2;
    }
  }
#endif


#if defined(HAS_SCALEFILTERCOLS_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && src_width < 32768) {
    ScaleFilterCols = ScaleFilterCols_SSSE3;
  }
#endif
  if (y > max_y) {
    y = max_y;
  }

  for (j = 0; j < dst_height; ++j) {
    int yi = y >> 16;
    const uint8* src = src_ptr + yi * src_stride;
    if (filtering == kFilterLinear) {
      ScaleFilterCols(dst_ptr, src, dst_width, x, dx);
    } else {
      int yf = (y >> 8) & 255;
      InterpolateRow(row, src, src_stride, src_width, yf);
      ScaleFilterCols(dst_ptr, row, dst_width, x, dx);
    }
    dst_ptr += dst_stride;
    y += dy;
    if (y > max_y) {
      y = max_y;
    }
  }
  free_aligned_buffer_64(row);
}

// Scale up down with bilinear interpolation.
void ScalePlaneBilinearUp(int src_width, int src_height,
                          int dst_width, int dst_height,
                          int src_stride, int dst_stride,
                          const uint8* src_ptr, uint8* dst_ptr,
                          enum FilterMode filtering) {
  int j;
  // Initial source x/y coordinate and step values as 16.16 fixed point.
  int x = 0;
  int y = 0;
  int dx = 0;
  int dy = 0;
  const int max_y = (src_height - 1) << 16;
  void (*InterpolateRow)(uint8* dst_ptr, const uint8* src_ptr,
      ptrdiff_t src_stride, int dst_width, int source_y_fraction) =
      InterpolateRow_C;
  void (*ScaleFilterCols)(uint8* dst_ptr, const uint8* src_ptr,
       int dst_width, int x, int dx) =
       filtering ? ScaleFilterCols_C : ScaleCols_C;
  ScaleSlope(src_width, src_height, dst_width, dst_height, filtering,
             &x, &y, &dx, &dy);
  src_width = Abs(src_width);

#if defined(HAS_INTERPOLATEROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && dst_width >= 16) {
    InterpolateRow = InterpolateRow_Any_SSE2;
    if (IS_ALIGNED(dst_width, 16)) {
      InterpolateRow = InterpolateRow_Unaligned_SSE2;
      if (IS_ALIGNED(dst_ptr, 16) && IS_ALIGNED(dst_stride, 16)) {
        InterpolateRow = InterpolateRow_SSE2;
      }
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && dst_width >= 16) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(dst_width, 16)) {
      InterpolateRow = InterpolateRow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_ptr, 16) && IS_ALIGNED(dst_stride, 16)) {
        InterpolateRow = InterpolateRow_SSSE3;
      }
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2) && dst_width >= 32) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(dst_width, 32)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && dst_width >= 16) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(dst_width, 16)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_MIPS_DSPR2)
  if (TestCpuFlag(kCpuHasMIPS_DSPR2) && dst_width >= 4) {
    InterpolateRow = InterpolateRow_Any_MIPS_DSPR2;
    if (IS_ALIGNED(dst_width, 4)) {
      InterpolateRow = InterpolateRow_MIPS_DSPR2;
    }
  }
#endif

  if (filtering && src_width >= 32768) {
    ScaleFilterCols = ScaleFilterCols64_C;
  }
#if defined(HAS_SCALEFILTERCOLS_SSSE3)
  if (filtering && TestCpuFlag(kCpuHasSSSE3) && src_width < 32768) {
    ScaleFilterCols = ScaleFilterCols_SSSE3;
  }
#endif
  if (!filtering && src_width * 2 == dst_width && x < 0x8000) {
    ScaleFilterCols = ScaleColsUp2_C;
#if defined(HAS_SCALECOLS_SSE2)
    if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(dst_width, 8) &&
        IS_ALIGNED(src_ptr, 16) && IS_ALIGNED(src_stride, 16) &&
        IS_ALIGNED(dst_ptr, 16) && IS_ALIGNED(dst_stride, 16)) {
      ScaleFilterCols = ScaleColsUp2_SSE2;
    }
#endif
  }

  if (y > max_y) {
    y = max_y;
  }
  {
    int yi = y >> 16;
    const uint8* src = src_ptr + yi * src_stride;

    // Allocate 2 row buffers.
    const int kRowSize = (dst_width + 15) & ~15;
    align_buffer_64(row, kRowSize * 2);

    uint8* rowptr = row;
    int rowstride = kRowSize;
    int lasty = yi;

    ScaleFilterCols(rowptr, src, dst_width, x, dx);
    if (src_height > 1) {
      src += src_stride;
    }
    ScaleFilterCols(rowptr + rowstride, src, dst_width, x, dx);
    src += src_stride;

    for (j = 0; j < dst_height; ++j) {
      yi = y >> 16;
      if (yi != lasty) {
        if (y > max_y) {
          y = max_y;
          yi = y >> 16;
          src = src_ptr + yi * src_stride;
        }
        if (yi != lasty) {
          ScaleFilterCols(rowptr, src, dst_width, x, dx);
          rowptr += rowstride;
          rowstride = -rowstride;
          lasty = yi;
          src += src_stride;
        }
      }
      if (filtering == kFilterLinear) {
        InterpolateRow(dst_ptr, rowptr, 0, dst_width, 0);
      } else {
        int yf = (y >> 8) & 255;
        InterpolateRow(dst_ptr, rowptr, rowstride, dst_width, yf);
      }
      dst_ptr += dst_stride;
      y += dy;
    }
    free_aligned_buffer_64(row);
  }
}

// Scale Plane to/from any dimensions, without interpolation.
// Fixed point math is used for performance: The upper 16 bits
// of x and dx is the integer part of the source position and
// the lower 16 bits are the fixed decimal part.

static void ScalePlaneSimple(int src_width, int src_height,
                             int dst_width, int dst_height,
                             int src_stride, int dst_stride,
                             const uint8* src_ptr, uint8* dst_ptr) {
  int i;
  void (*ScaleCols)(uint8* dst_ptr, const uint8* src_ptr,
      int dst_width, int x, int dx) = ScaleCols_C;
  // Initial source x/y coordinate and step values as 16.16 fixed point.
  int x = 0;
  int y = 0;
  int dx = 0;
  int dy = 0;
  ScaleSlope(src_width, src_height, dst_width, dst_height, kFilterNone,
             &x, &y, &dx, &dy);
  src_width = Abs(src_width);

  if (src_width * 2 == dst_width && x < 0x8000) {
    ScaleCols = ScaleColsUp2_C;
#if defined(HAS_SCALECOLS_SSE2)
    if (TestCpuFlag(kCpuHasSSE2) && IS_ALIGNED(dst_width, 8) &&
        IS_ALIGNED(src_ptr, 16) && IS_ALIGNED(src_stride, 16) &&
        IS_ALIGNED(dst_ptr, 16) && IS_ALIGNED(dst_stride, 16)) {
      ScaleCols = ScaleColsUp2_SSE2;
    }
#endif
  }

  for (i = 0; i < dst_height; ++i) {
    ScaleCols(dst_ptr, src_ptr + (y >> 16) * src_stride,
              dst_width, x, dx);
    dst_ptr += dst_stride;
    y += dy;
  }
}

// Scale a plane.
// This function dispatches to a specialized scaler based on scale factor.

LIBYUV_API
void ScalePlane(const uint8* src, int src_stride,
                int src_width, int src_height,
                uint8* dst, int dst_stride,
                int dst_width, int dst_height,
                enum FilterMode filtering) {
  // Simplify filtering when possible.
  filtering = ScaleFilterReduce(src_width, src_height,
                                dst_width, dst_height,
                                filtering);

  // Negative height means invert the image.
  if (src_height < 0) {
    src_height = -src_height;
    src = src + (src_height - 1) * src_stride;
    src_stride = -src_stride;
  }

  // Use specialized scales to improve performance for common resolutions.
  // For example, all the 1/2 scalings will use ScalePlaneDown2()
  if (dst_width == src_width && dst_height == src_height) {
    // Straight copy.
    CopyPlane(src, src_stride, dst, dst_stride, dst_width, dst_height);
    return;
  }
  if (dst_width == src_width) {
    int dy = FixedDiv(src_height, dst_height);
    // Arbitrary scale vertically, but unscaled vertically.
    ScalePlaneVertical(src_height,
                       dst_width, dst_height,
                       src_stride, dst_stride, src, dst,
                       0, 0, dy, 1, filtering);
    return;
  }
  if (dst_width <= Abs(src_width) && dst_height <= src_height) {
    // Scale down.
    if (4 * dst_width == 3 * src_width &&
        4 * dst_height == 3 * src_height) {
      // optimized, 3/4
      ScalePlaneDown34(src_width, src_height, dst_width, dst_height,
                       src_stride, dst_stride, src, dst, filtering);
      return;
    }
    if (2 * dst_width == src_width && 2 * dst_height == src_height) {
      // optimized, 1/2
      ScalePlaneDown2(src_width, src_height, dst_width, dst_height,
                      src_stride, dst_stride, src, dst, filtering);
      return;
    }
    // 3/8 rounded up for odd sized chroma height.
    if (8 * dst_width == 3 * src_width &&
        dst_height == ((src_height * 3 + 7) / 8)) {
      // optimized, 3/8
      ScalePlaneDown38(src_width, src_height, dst_width, dst_height,
                       src_stride, dst_stride, src, dst, filtering);
      return;
    }
    if (4 * dst_width == src_width && 4 * dst_height == src_height &&
               filtering != kFilterBilinear) {
      // optimized, 1/4
      ScalePlaneDown4(src_width, src_height, dst_width, dst_height,
                      src_stride, dst_stride, src, dst, filtering);
      return;
    }
  }
  if (filtering == kFilterBox && dst_height * 2 < src_height) {
    ScalePlaneBox(src_width, src_height, dst_width, dst_height,
                  src_stride, dst_stride, src, dst);
    return;
  }
  if (filtering && dst_height > src_height) {
    ScalePlaneBilinearUp(src_width, src_height, dst_width, dst_height,
                         src_stride, dst_stride, src, dst, filtering);
    return;
  }
  if (filtering) {
    ScalePlaneBilinearDown(src_width, src_height, dst_width, dst_height,
                           src_stride, dst_stride, src, dst, filtering);
    return;
  }
  ScalePlaneSimple(src_width, src_height, dst_width, dst_height,
                   src_stride, dst_stride, src, dst);
}

// Scale an I420 image.
// This function in turn calls a scaling function for each plane.

LIBYUV_API
int I420Scale(const uint8* src_y, int src_stride_y,
              const uint8* src_u, int src_stride_u,
              const uint8* src_v, int src_stride_v,
              int src_width, int src_height,
              uint8* dst_y, int dst_stride_y,
              uint8* dst_u, int dst_stride_u,
              uint8* dst_v, int dst_stride_v,
              int dst_width, int dst_height,
              enum FilterMode filtering) {
  int src_halfwidth = SUBSAMPLE(src_width, 1, 1);
  int src_halfheight = SUBSAMPLE(src_height, 1, 1);
  int dst_halfwidth = SUBSAMPLE(dst_width, 1, 1);
  int dst_halfheight = SUBSAMPLE(dst_height, 1, 1);
  if (!src_y || !src_u || !src_v || src_width == 0 || src_height == 0 ||
      !dst_y || !dst_u || !dst_v || dst_width <= 0 || dst_height <= 0) {
    return -1;
  }

  ScalePlane(src_y, src_stride_y, src_width, src_height,
             dst_y, dst_stride_y, dst_width, dst_height,
             filtering);
  ScalePlane(src_u, src_stride_u, src_halfwidth, src_halfheight,
             dst_u, dst_stride_u, dst_halfwidth, dst_halfheight,
             filtering);
  ScalePlane(src_v, src_stride_v, src_halfwidth, src_halfheight,
             dst_v, dst_stride_v, dst_halfwidth, dst_halfheight,
             filtering);
  return 0;
}

// Deprecated api
LIBYUV_API
int Scale(const uint8* src_y, const uint8* src_u, const uint8* src_v,
          int src_stride_y, int src_stride_u, int src_stride_v,
          int src_width, int src_height,
          uint8* dst_y, uint8* dst_u, uint8* dst_v,
          int dst_stride_y, int dst_stride_u, int dst_stride_v,
          int dst_width, int dst_height,
          LIBYUV_BOOL interpolate) {
  return I420Scale(src_y, src_stride_y,
                   src_u, src_stride_u,
                   src_v, src_stride_v,
                   src_width, src_height,
                   dst_y, dst_stride_y,
                   dst_u, dst_stride_u,
                   dst_v, dst_stride_v,
                   dst_width, dst_height,
                   interpolate ? kFilterBox : kFilterNone);
}

// Deprecated api
LIBYUV_API
int ScaleOffset(const uint8* src, int src_width, int src_height,
                uint8* dst, int dst_width, int dst_height, int dst_yoffset,
                LIBYUV_BOOL interpolate) {
  // Chroma requires offset to multiple of 2.
  int dst_yoffset_even = dst_yoffset & ~1;
  int src_halfwidth = SUBSAMPLE(src_width, 1, 1);
  int src_halfheight = SUBSAMPLE(src_height, 1, 1);
  int dst_halfwidth = SUBSAMPLE(dst_width, 1, 1);
  int dst_halfheight = SUBSAMPLE(dst_height, 1, 1);
  int aheight = dst_height - dst_yoffset_even * 2;  // actual output height
  const uint8* src_y = src;
  const uint8* src_u = src + src_width * src_height;
  const uint8* src_v = src + src_width * src_height +
                             src_halfwidth * src_halfheight;
  uint8* dst_y = dst + dst_yoffset_even * dst_width;
  uint8* dst_u = dst + dst_width * dst_height +
                 (dst_yoffset_even >> 1) * dst_halfwidth;
  uint8* dst_v = dst + dst_width * dst_height + dst_halfwidth * dst_halfheight +
                 (dst_yoffset_even >> 1) * dst_halfwidth;
  if (!src || src_width <= 0 || src_height <= 0 ||
      !dst || dst_width <= 0 || dst_height <= 0 || dst_yoffset_even < 0 ||
      dst_yoffset_even >= dst_height) {
    return -1;
  }
  return I420Scale(src_y, src_width,
                   src_u, src_halfwidth,
                   src_v, src_halfwidth,
                   src_width, src_height,
                   dst_y, dst_width,
                   dst_u, dst_halfwidth,
                   dst_v, dst_halfwidth,
                   dst_width, aheight,
                   interpolate ? kFilterBox : kFilterNone);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
