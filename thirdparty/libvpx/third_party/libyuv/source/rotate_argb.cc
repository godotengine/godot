/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/rotate.h"

#include "libyuv/convert.h"
#include "libyuv/cpu_id.h"
#include "libyuv/planar_functions.h"
#include "libyuv/row.h"
#include "libyuv/scale_row.h" /* for ScaleARGBRowDownEven_ */

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

static void ARGBTranspose(const uint8_t* src_argb,
                          int src_stride_argb,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          int width,
                          int height) {
  int i;
  int src_pixel_step = src_stride_argb >> 2;
  void (*ScaleARGBRowDownEven)(
      const uint8_t* src_argb, ptrdiff_t src_stride_argb, int src_step,
      uint8_t* dst_argb, int dst_width) = ScaleARGBRowDownEven_C;
#if defined(HAS_SCALEARGBROWDOWNEVEN_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ScaleARGBRowDownEven = ScaleARGBRowDownEven_Any_SSE2;
    if (IS_ALIGNED(height, 4)) {  // Width of dest.
      ScaleARGBRowDownEven = ScaleARGBRowDownEven_SSE2;
    }
  }
#endif
#if defined(HAS_SCALEARGBROWDOWNEVEN_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleARGBRowDownEven = ScaleARGBRowDownEven_Any_NEON;
    if (IS_ALIGNED(height, 4)) {  // Width of dest.
      ScaleARGBRowDownEven = ScaleARGBRowDownEven_NEON;
    }
  }
#endif
#if defined(HAS_SCALEARGBROWDOWNEVEN_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ScaleARGBRowDownEven = ScaleARGBRowDownEven_Any_MSA;
    if (IS_ALIGNED(height, 4)) {  // Width of dest.
      ScaleARGBRowDownEven = ScaleARGBRowDownEven_MSA;
    }
  }
#endif

  for (i = 0; i < width; ++i) {  // column of source to row of dest.
    ScaleARGBRowDownEven(src_argb, 0, src_pixel_step, dst_argb, height);
    dst_argb += dst_stride_argb;
    src_argb += 4;
  }
}

void ARGBRotate90(const uint8_t* src_argb,
                  int src_stride_argb,
                  uint8_t* dst_argb,
                  int dst_stride_argb,
                  int width,
                  int height) {
  // Rotate by 90 is a ARGBTranspose with the source read
  // from bottom to top. So set the source pointer to the end
  // of the buffer and flip the sign of the source stride.
  src_argb += src_stride_argb * (height - 1);
  src_stride_argb = -src_stride_argb;
  ARGBTranspose(src_argb, src_stride_argb, dst_argb, dst_stride_argb, width,
                height);
}

void ARGBRotate270(const uint8_t* src_argb,
                   int src_stride_argb,
                   uint8_t* dst_argb,
                   int dst_stride_argb,
                   int width,
                   int height) {
  // Rotate by 270 is a ARGBTranspose with the destination written
  // from bottom to top. So set the destination pointer to the end
  // of the buffer and flip the sign of the destination stride.
  dst_argb += dst_stride_argb * (width - 1);
  dst_stride_argb = -dst_stride_argb;
  ARGBTranspose(src_argb, src_stride_argb, dst_argb, dst_stride_argb, width,
                height);
}

void ARGBRotate180(const uint8_t* src_argb,
                   int src_stride_argb,
                   uint8_t* dst_argb,
                   int dst_stride_argb,
                   int width,
                   int height) {
  // Swap first and last row and mirror the content. Uses a temporary row.
  align_buffer_64(row, width * 4);
  const uint8_t* src_bot = src_argb + src_stride_argb * (height - 1);
  uint8_t* dst_bot = dst_argb + dst_stride_argb * (height - 1);
  int half_height = (height + 1) >> 1;
  int y;
  void (*ARGBMirrorRow)(const uint8_t* src_argb, uint8_t* dst_argb, int width) =
      ARGBMirrorRow_C;
  void (*CopyRow)(const uint8_t* src_argb, uint8_t* dst_argb, int width) =
      CopyRow_C;
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
#if defined(HAS_COPYROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    CopyRow = IS_ALIGNED(width * 4, 32) ? CopyRow_SSE2 : CopyRow_Any_SSE2;
  }
#endif
#if defined(HAS_COPYROW_AVX)
  if (TestCpuFlag(kCpuHasAVX)) {
    CopyRow = IS_ALIGNED(width * 4, 64) ? CopyRow_AVX : CopyRow_Any_AVX;
  }
#endif
#if defined(HAS_COPYROW_ERMS)
  if (TestCpuFlag(kCpuHasERMS)) {
    CopyRow = CopyRow_ERMS;
  }
#endif
#if defined(HAS_COPYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    CopyRow = IS_ALIGNED(width * 4, 32) ? CopyRow_NEON : CopyRow_Any_NEON;
  }
#endif

  // Odd height will harmlessly mirror the middle row twice.
  for (y = 0; y < half_height; ++y) {
    ARGBMirrorRow(src_argb, row, width);      // Mirror first row into a buffer
    ARGBMirrorRow(src_bot, dst_argb, width);  // Mirror last row into first row
    CopyRow(row, dst_bot, width * 4);  // Copy first mirrored row into last
    src_argb += src_stride_argb;
    dst_argb += dst_stride_argb;
    src_bot -= src_stride_argb;
    dst_bot -= dst_stride_argb;
  }
  free_aligned_buffer_64(row);
}

LIBYUV_API
int ARGBRotate(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height,
               enum RotationMode mode) {
  if (!src_argb || width <= 0 || height == 0 || !dst_argb) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }

  switch (mode) {
    case kRotate0:
      // copy frame
      return ARGBCopy(src_argb, src_stride_argb, dst_argb, dst_stride_argb,
                      width, height);
    case kRotate90:
      ARGBRotate90(src_argb, src_stride_argb, dst_argb, dst_stride_argb, width,
                   height);
      return 0;
    case kRotate270:
      ARGBRotate270(src_argb, src_stride_argb, dst_argb, dst_stride_argb, width,
                    height);
      return 0;
    case kRotate180:
      ARGBRotate180(src_argb, src_stride_argb, dst_argb, dst_stride_argb, width,
                    height);
      return 0;
    default:
      break;
  }
  return -1;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
