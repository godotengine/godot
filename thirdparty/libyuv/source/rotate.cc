/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "libyuv/rotate.h"

#include "libyuv/convert.h"
#include "libyuv/cpu_id.h"
#include "libyuv/planar_functions.h"
#include "libyuv/rotate_row.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

LIBYUV_API
void TransposePlane(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width,
                    int height) {
  int i = height;
#if defined(HAS_TRANSPOSEWXH_SME)
  void (*TransposeWxH)(const uint8_t* src, int src_stride, uint8_t* dst,
                       int dst_stride, int width, int height) = NULL;
#endif
#if defined(HAS_TRANSPOSEWX16_MSA) || defined(HAS_TRANSPOSEWX16_LSX) || \
    defined(HAS_TRANSPOSEWX16_NEON)
  void (*TransposeWx16)(const uint8_t* src, int src_stride, uint8_t* dst,
                        int dst_stride, int width) = TransposeWx16_C;
#else
  void (*TransposeWx8)(const uint8_t* src, int src_stride, uint8_t* dst,
                       int dst_stride, int width) = TransposeWx8_C;
#endif

#if defined(HAS_TRANSPOSEWX8_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    TransposeWx8 = TransposeWx8_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      TransposeWx8 = TransposeWx8_NEON;
    }
  }
#endif
#if defined(HAS_TRANSPOSEWX16_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    TransposeWx16 = TransposeWx16_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      TransposeWx16 = TransposeWx16_NEON;
    }
  }
#endif
#if defined(HAS_TRANSPOSEWXH_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    TransposeWxH = TransposeWxH_SME;
  }
#endif
#if defined(HAS_TRANSPOSEWX8_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    TransposeWx8 = TransposeWx8_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      TransposeWx8 = TransposeWx8_SSSE3;
    }
  }
#endif
#if defined(HAS_TRANSPOSEWX8_FAST_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    TransposeWx8 = TransposeWx8_Fast_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      TransposeWx8 = TransposeWx8_Fast_SSSE3;
    }
  }
#endif
#if defined(HAS_TRANSPOSEWX16_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    TransposeWx16 = TransposeWx16_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      TransposeWx16 = TransposeWx16_MSA;
    }
  }
#endif
#if defined(HAS_TRANSPOSEWX16_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    TransposeWx16 = TransposeWx16_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      TransposeWx16 = TransposeWx16_LSX;
    }
  }
#endif

#if defined(HAS_TRANSPOSEWXH_SME)
  if (TransposeWxH) {
    TransposeWxH(src, src_stride, dst, dst_stride, width, height);
    return;
  }
#endif
#if defined(HAS_TRANSPOSEWX16_MSA) || defined(HAS_TRANSPOSEWX16_LSX) || \
    defined(HAS_TRANSPOSEWX16_NEON)
  // Work across the source in 16x16 tiles
  while (i >= 16) {
    TransposeWx16(src, src_stride, dst, dst_stride, width);
    src += 16 * src_stride;  // Go down 16 rows.
    dst += 16;               // Move over 16 columns.
    i -= 16;
  }
#else
  // Work across the source in 8x8 tiles
  while (i >= 8) {
    TransposeWx8(src, src_stride, dst, dst_stride, width);
    src += 8 * src_stride;  // Go down 8 rows.
    dst += 8;               // Move over 8 columns.
    i -= 8;
  }
#endif

  if (i > 0) {
    TransposeWxH_C(src, src_stride, dst, dst_stride, width, i);
  }
}

LIBYUV_API
void RotatePlane90(const uint8_t* src,
                   int src_stride,
                   uint8_t* dst,
                   int dst_stride,
                   int width,
                   int height) {
  // Rotate by 90 is a transpose with the source read
  // from bottom to top. So set the source pointer to the end
  // of the buffer and flip the sign of the source stride.
  src += src_stride * (height - 1);
  src_stride = -src_stride;
  TransposePlane(src, src_stride, dst, dst_stride, width, height);
}

LIBYUV_API
void RotatePlane270(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width,
                    int height) {
  // Rotate by 270 is a transpose with the destination written
  // from bottom to top. So set the destination pointer to the end
  // of the buffer and flip the sign of the destination stride.
  dst += dst_stride * (width - 1);
  dst_stride = -dst_stride;
  TransposePlane(src, src_stride, dst, dst_stride, width, height);
}

LIBYUV_API
void RotatePlane180(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width,
                    int height) {
  // Swap top and bottom row and mirror the content. Uses a temporary row.
  align_buffer_64(row, width);
  assert(row);
  if (!row)
    return;
  const uint8_t* src_bot = src + src_stride * (height - 1);
  uint8_t* dst_bot = dst + dst_stride * (height - 1);
  int half_height = (height + 1) >> 1;
  int y;
  void (*MirrorRow)(const uint8_t* src, uint8_t* dst, int width) = MirrorRow_C;
  void (*CopyRow)(const uint8_t* src, uint8_t* dst, int width) = CopyRow_C;
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

  // Odd height will harmlessly mirror the middle row twice.
  for (y = 0; y < half_height; ++y) {
    CopyRow(src, row, width);        // Copy top row into buffer
    MirrorRow(src_bot, dst, width);  // Mirror bottom row into top row
    MirrorRow(row, dst_bot, width);  // Mirror buffer into bottom row
    src += src_stride;
    dst += dst_stride;
    src_bot -= src_stride;
    dst_bot -= dst_stride;
  }
  free_aligned_buffer_64(row);
}

LIBYUV_API
void SplitTransposeUV(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width,
                      int height) {
  int i = height;
#if defined(HAS_TRANSPOSEUVWXH_SME)
  void (*TransposeUVWxH)(const uint8_t* src, int src_stride, uint8_t* dst_a,
                         int dst_stride_a, uint8_t* dst_b, int dst_stride_b,
                         int width, int height) = TransposeUVWxH_C;
#endif
#if defined(HAS_TRANSPOSEUVWX16_MSA)
  void (*TransposeUVWx16)(const uint8_t* src, int src_stride, uint8_t* dst_a,
                          int dst_stride_a, uint8_t* dst_b, int dst_stride_b,
                          int width) = TransposeUVWx16_C;
#elif defined(HAS_TRANSPOSEUVWX16_LSX)
  void (*TransposeUVWx16)(const uint8_t* src, int src_stride, uint8_t* dst_a,
                          int dst_stride_a, uint8_t* dst_b, int dst_stride_b,
                          int width) = TransposeUVWx16_C;
#else
  void (*TransposeUVWx8)(const uint8_t* src, int src_stride, uint8_t* dst_a,
                         int dst_stride_a, uint8_t* dst_b, int dst_stride_b,
                         int width) = TransposeUVWx8_C;
#endif

#if defined(HAS_TRANSPOSEUVWX16_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    TransposeUVWx16 = TransposeUVWx16_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      TransposeUVWx16 = TransposeUVWx16_MSA;
    }
  }
#elif defined(HAS_TRANSPOSEUVWX16_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    TransposeUVWx16 = TransposeUVWx16_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      TransposeUVWx16 = TransposeUVWx16_LSX;
    }
  }
#else
#if defined(HAS_TRANSPOSEUVWX8_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    TransposeUVWx8 = TransposeUVWx8_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      TransposeUVWx8 = TransposeUVWx8_NEON;
    }
  }
#endif
#if defined(HAS_TRANSPOSEUVWXH_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    TransposeUVWxH = TransposeUVWxH_SME;
  }
#endif
#if defined(HAS_TRANSPOSEUVWX8_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    TransposeUVWx8 = TransposeUVWx8_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      TransposeUVWx8 = TransposeUVWx8_SSE2;
    }
  }
#endif
#endif /* defined(HAS_TRANSPOSEUVWX16_MSA) */

#if defined(HAS_TRANSPOSEUVWXH_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    TransposeUVWxH(src, src_stride, dst_a, dst_stride_a, dst_b, dst_stride_b,
                   width, i);
    return;
  }
#endif
#if defined(HAS_TRANSPOSEUVWX16_MSA)
  // Work through the source in 8x8 tiles.
  while (i >= 16) {
    TransposeUVWx16(src, src_stride, dst_a, dst_stride_a, dst_b, dst_stride_b,
                    width);
    src += 16 * src_stride;  // Go down 16 rows.
    dst_a += 16;             // Move over 8 columns.
    dst_b += 16;             // Move over 8 columns.
    i -= 16;
  }
#elif defined(HAS_TRANSPOSEUVWX16_LSX)
  // Work through the source in 8x8 tiles.
  while (i >= 16) {
    TransposeUVWx16(src, src_stride, dst_a, dst_stride_a, dst_b, dst_stride_b,
                    width);
    src += 16 * src_stride;  // Go down 16 rows.
    dst_a += 16;             // Move over 8 columns.
    dst_b += 16;             // Move over 8 columns.
    i -= 16;
  }
#else
  // Work through the source in 8x8 tiles.
  while (i >= 8) {
    TransposeUVWx8(src, src_stride, dst_a, dst_stride_a, dst_b, dst_stride_b,
                   width);
    src += 8 * src_stride;  // Go down 8 rows.
    dst_a += 8;             // Move over 8 columns.
    dst_b += 8;             // Move over 8 columns.
    i -= 8;
  }
#endif

  if (i > 0) {
    TransposeUVWxH_C(src, src_stride, dst_a, dst_stride_a, dst_b, dst_stride_b,
                     width, i);
  }
}

LIBYUV_API
void SplitRotateUV90(const uint8_t* src,
                     int src_stride,
                     uint8_t* dst_a,
                     int dst_stride_a,
                     uint8_t* dst_b,
                     int dst_stride_b,
                     int width,
                     int height) {
  src += src_stride * (height - 1);
  src_stride = -src_stride;

  SplitTransposeUV(src, src_stride, dst_a, dst_stride_a, dst_b, dst_stride_b,
                   width, height);
}

LIBYUV_API
void SplitRotateUV270(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width,
                      int height) {
  dst_a += dst_stride_a * (width - 1);
  dst_b += dst_stride_b * (width - 1);
  dst_stride_a = -dst_stride_a;
  dst_stride_b = -dst_stride_b;

  SplitTransposeUV(src, src_stride, dst_a, dst_stride_a, dst_b, dst_stride_b,
                   width, height);
}

// Rotate 180 is a horizontal and vertical flip.
LIBYUV_API
void SplitRotateUV180(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width,
                      int height) {
  int i;
  void (*MirrorSplitUVRow)(const uint8_t* src, uint8_t* dst_u, uint8_t* dst_v,
                           int width) = MirrorSplitUVRow_C;
#if defined(HAS_MIRRORSPLITUVROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && IS_ALIGNED(width, 16)) {
    MirrorSplitUVRow = MirrorSplitUVRow_NEON;
  }
#endif
#if defined(HAS_MIRRORSPLITUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && IS_ALIGNED(width, 16)) {
    MirrorSplitUVRow = MirrorSplitUVRow_SSSE3;
  }
#endif
#if defined(HAS_MIRRORSPLITUVROW_MSA)
  if (TestCpuFlag(kCpuHasMSA) && IS_ALIGNED(width, 32)) {
    MirrorSplitUVRow = MirrorSplitUVRow_MSA;
  }
#endif
#if defined(HAS_MIRRORSPLITUVROW_LSX)
  if (TestCpuFlag(kCpuHasLSX) && IS_ALIGNED(width, 32)) {
    MirrorSplitUVRow = MirrorSplitUVRow_LSX;
  }
#endif

  dst_a += dst_stride_a * (height - 1);
  dst_b += dst_stride_b * (height - 1);

  for (i = 0; i < height; ++i) {
    MirrorSplitUVRow(src, dst_a, dst_b, width);
    src += src_stride;
    dst_a -= dst_stride_a;
    dst_b -= dst_stride_b;
  }
}

// Rotate UV and split into planar.
// width and height expected to be half size for NV12
LIBYUV_API
int SplitRotateUV(const uint8_t* src_uv,
                  int src_stride_uv,
                  uint8_t* dst_u,
                  int dst_stride_u,
                  uint8_t* dst_v,
                  int dst_stride_v,
                  int width,
                  int height,
                  enum RotationMode mode) {
  if (!src_uv || width <= 0 || height == 0 || !dst_u || !dst_v) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uv = src_uv + (height - 1) * src_stride_uv;
    src_stride_uv = -src_stride_uv;
  }

  switch (mode) {
    case kRotate0:
      SplitUVPlane(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v,
                   dst_stride_v, width, height);
      return 0;
    case kRotate90:
      SplitRotateUV90(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v,
                      dst_stride_v, width, height);
      return 0;
    case kRotate270:
      SplitRotateUV270(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v,
                       dst_stride_v, width, height);
      return 0;
    case kRotate180:
      SplitRotateUV180(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v,
                       dst_stride_v, width, height);
      return 0;
    default:
      break;
  }
  return -1;
}

LIBYUV_API
int RotatePlane(const uint8_t* src,
                int src_stride,
                uint8_t* dst,
                int dst_stride,
                int width,
                int height,
                enum RotationMode mode) {
  if (!src || width <= 0 || height == 0 || !dst) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src = src + (height - 1) * src_stride;
    src_stride = -src_stride;
  }

  switch (mode) {
    case kRotate0:
      // copy frame
      CopyPlane(src, src_stride, dst, dst_stride, width, height);
      return 0;
    case kRotate90:
      RotatePlane90(src, src_stride, dst, dst_stride, width, height);
      return 0;
    case kRotate270:
      RotatePlane270(src, src_stride, dst, dst_stride, width, height);
      return 0;
    case kRotate180:
      RotatePlane180(src, src_stride, dst, dst_stride, width, height);
      return 0;
    default:
      break;
  }
  return -1;
}

static void TransposePlane_16(const uint16_t* src,
                              int src_stride,
                              uint16_t* dst,
                              int dst_stride,
                              int width,
                              int height) {
  int i = height;
  // Work across the source in 8x8 tiles
  while (i >= 8) {
    TransposeWx8_16_C(src, src_stride, dst, dst_stride, width);
    src += 8 * src_stride;  // Go down 8 rows.
    dst += 8;               // Move over 8 columns.
    i -= 8;
  }

  if (i > 0) {
    TransposeWxH_16_C(src, src_stride, dst, dst_stride, width, i);
  }
}

static void RotatePlane90_16(const uint16_t* src,
                             int src_stride,
                             uint16_t* dst,
                             int dst_stride,
                             int width,
                             int height) {
  // Rotate by 90 is a transpose with the source read
  // from bottom to top. So set the source pointer to the end
  // of the buffer and flip the sign of the source stride.
  src += src_stride * (height - 1);
  src_stride = -src_stride;
  TransposePlane_16(src, src_stride, dst, dst_stride, width, height);
}

static void RotatePlane270_16(const uint16_t* src,
                              int src_stride,
                              uint16_t* dst,
                              int dst_stride,
                              int width,
                              int height) {
  // Rotate by 270 is a transpose with the destination written
  // from bottom to top. So set the destination pointer to the end
  // of the buffer and flip the sign of the destination stride.
  dst += dst_stride * (width - 1);
  dst_stride = -dst_stride;
  TransposePlane_16(src, src_stride, dst, dst_stride, width, height);
}

static void RotatePlane180_16(const uint16_t* src,
                              int src_stride,
                              uint16_t* dst,
                              int dst_stride,
                              int width,
                              int height) {
  const uint16_t* src_bot = src + src_stride * (height - 1);
  uint16_t* dst_bot = dst + dst_stride * (height - 1);
  int half_height = (height + 1) >> 1;
  int y;

  // Swap top and bottom row and mirror the content. Uses a temporary row.
  align_buffer_64(row, width * 2);
  uint16_t* row_tmp = (uint16_t*)row;
  assert(row);
  if (!row)
    return;

  // Odd height will harmlessly mirror the middle row twice.
  for (y = 0; y < half_height; ++y) {
    CopyRow_16_C(src, row_tmp, width);        // Copy top row into buffer
    MirrorRow_16_C(src_bot, dst, width);      // Mirror bottom row into top row
    MirrorRow_16_C(row_tmp, dst_bot, width);  // Mirror buffer into bottom row
    src += src_stride;
    dst += dst_stride;
    src_bot -= src_stride;
    dst_bot -= dst_stride;
  }
  free_aligned_buffer_64(row);
}

LIBYUV_API
int RotatePlane_16(const uint16_t* src,
                   int src_stride,
                   uint16_t* dst,
                   int dst_stride,
                   int width,
                   int height,
                   enum RotationMode mode) {
  if (!src || width <= 0 || height == 0 || !dst) {
    return -1;
  }

  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src = src + (height - 1) * src_stride;
    src_stride = -src_stride;
  }

  switch (mode) {
    case kRotate0:
      // copy frame
      CopyPlane_16(src, src_stride, dst, dst_stride, width, height);
      return 0;
    case kRotate90:
      RotatePlane90_16(src, src_stride, dst, dst_stride, width, height);
      return 0;
    case kRotate270:
      RotatePlane270_16(src, src_stride, dst, dst_stride, width, height);
      return 0;
    case kRotate180:
      RotatePlane180_16(src, src_stride, dst, dst_stride, width, height);
      return 0;
    default:
      break;
  }
  return -1;
}

LIBYUV_API
int I420Rotate(const uint8_t* src_y,
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
               int height,
               enum RotationMode mode) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if ((!src_y && dst_y) || !src_u || !src_v || width <= 0 || height == 0 ||
      !dst_y || !dst_u || !dst_v) {
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

  switch (mode) {
    case kRotate0:
      // copy frame
      return I420Copy(src_y, src_stride_y, src_u, src_stride_u, src_v,
                      src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                      dst_v, dst_stride_v, width, height);
    case kRotate90:
      RotatePlane90(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      RotatePlane90(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth,
                    halfheight);
      RotatePlane90(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth,
                    halfheight);
      return 0;
    case kRotate270:
      RotatePlane270(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      RotatePlane270(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth,
                     halfheight);
      RotatePlane270(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth,
                     halfheight);
      return 0;
    case kRotate180:
      RotatePlane180(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      RotatePlane180(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth,
                     halfheight);
      RotatePlane180(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth,
                     halfheight);
      return 0;
    default:
      break;
  }
  return -1;
}

// I422 has half width x full height UV planes, so rotate by 90 and 270
// require scaling to maintain 422 subsampling.
LIBYUV_API
int I422Rotate(const uint8_t* src_y,
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
               int height,
               enum RotationMode mode) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  int r;
  if (!src_y || !src_u || !src_v || width <= 0 || height == 0 || !dst_y ||
      !dst_u || !dst_v) {
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

  switch (mode) {
    case kRotate0:
      // Copy frame
      CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      CopyPlane(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, height);
      CopyPlane(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, height);
      return 0;

      // Note on temporary Y plane for UV.
      // Rotation of UV first fits within the Y destination plane rows.
      // Y plane is width x height
      // Y plane rotated is height x width
      // UV plane is (width / 2) x height
      // UV plane rotated is height x (width / 2)
      // UV plane rotated+scaled is (height / 2) x width.
      // UV plane rotated is a temporary that fits within the Y plane rotated.

    case kRotate90:
      RotatePlane90(src_u, src_stride_u, dst_y, dst_stride_y, halfwidth,
                    height);
      r = ScalePlane(dst_y, dst_stride_y, height, halfwidth, dst_u,
                     dst_stride_u, halfheight, width, kFilterBilinear);
      if (r != 0) {
        return r;
      }
      RotatePlane90(src_v, src_stride_v, dst_y, dst_stride_y, halfwidth,
                    height);
      r = ScalePlane(dst_y, dst_stride_y, height, halfwidth, dst_v,
                     dst_stride_v, halfheight, width, kFilterLinear);
      if (r != 0) {
        return r;
      }
      RotatePlane90(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      return 0;
    case kRotate270:
      RotatePlane270(src_u, src_stride_u, dst_y, dst_stride_y, halfwidth,
                     height);
      r = ScalePlane(dst_y, dst_stride_y, height, halfwidth, dst_u,
                     dst_stride_u, halfheight, width, kFilterBilinear);
      if (r != 0) {
        return r;
      }
      RotatePlane270(src_v, src_stride_v, dst_y, dst_stride_y, halfwidth,
                     height);
      r = ScalePlane(dst_y, dst_stride_y, height, halfwidth, dst_v,
                     dst_stride_v, halfheight, width, kFilterLinear);
      if (r != 0) {
        return r;
      }
      RotatePlane270(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      return 0;
    case kRotate180:
      RotatePlane180(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      RotatePlane180(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth,
                     height);
      RotatePlane180(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth,
                     height);
      return 0;
    default:
      break;
  }
  return -1;
}

LIBYUV_API
int I444Rotate(const uint8_t* src_y,
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
               int height,
               enum RotationMode mode) {
  if (!src_y || !src_u || !src_v || width <= 0 || height == 0 || !dst_y ||
      !dst_u || !dst_v) {
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

  switch (mode) {
    case kRotate0:
      // copy frame
      CopyPlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      CopyPlane(src_u, src_stride_u, dst_u, dst_stride_u, width, height);
      CopyPlane(src_v, src_stride_v, dst_v, dst_stride_v, width, height);
      return 0;
    case kRotate90:
      RotatePlane90(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      RotatePlane90(src_u, src_stride_u, dst_u, dst_stride_u, width, height);
      RotatePlane90(src_v, src_stride_v, dst_v, dst_stride_v, width, height);
      return 0;
    case kRotate270:
      RotatePlane270(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      RotatePlane270(src_u, src_stride_u, dst_u, dst_stride_u, width, height);
      RotatePlane270(src_v, src_stride_v, dst_v, dst_stride_v, width, height);
      return 0;
    case kRotate180:
      RotatePlane180(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      RotatePlane180(src_u, src_stride_u, dst_u, dst_stride_u, width, height);
      RotatePlane180(src_v, src_stride_v, dst_v, dst_stride_v, width, height);
      return 0;
    default:
      break;
  }
  return -1;
}

LIBYUV_API
int NV12ToI420Rotate(const uint8_t* src_y,
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
                     int height,
                     enum RotationMode mode) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if (!src_y || !src_uv || width <= 0 || height == 0 || !dst_y || !dst_u ||
      !dst_v) {
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

  switch (mode) {
    case kRotate0:
      // copy frame
      return NV12ToI420(src_y, src_stride_y, src_uv, src_stride_uv, dst_y,
                        dst_stride_y, dst_u, dst_stride_u, dst_v, dst_stride_v,
                        width, height);
    case kRotate90:
      RotatePlane90(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      SplitRotateUV90(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v,
                      dst_stride_v, halfwidth, halfheight);
      return 0;
    case kRotate270:
      RotatePlane270(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      SplitRotateUV270(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v,
                       dst_stride_v, halfwidth, halfheight);
      return 0;
    case kRotate180:
      RotatePlane180(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      SplitRotateUV180(src_uv, src_stride_uv, dst_u, dst_stride_u, dst_v,
                       dst_stride_v, halfwidth, halfheight);
      return 0;
    default:
      break;
  }
  return -1;
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

// Convert Android420 to I420 with Rotate
LIBYUV_API
int Android420ToI420Rotate(const uint8_t* src_y,
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
                           int height,
                           enum RotationMode rotation) {
  int y;
  const ptrdiff_t vu_off = src_v - src_u;
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
    RotatePlane(src_y, src_stride_y, dst_y, dst_stride_y, width, height,
                rotation);
  }

  // Copy UV planes - I420
  if (src_pixel_stride_uv == 1) {
    RotatePlane(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, halfheight,
                rotation);
    RotatePlane(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, halfheight,
                rotation);
    return 0;
  }
  // Split UV planes - NV21
  if (src_pixel_stride_uv == 2 && vu_off == -1 &&
      src_stride_u == src_stride_v) {
    SplitRotateUV(src_v, src_stride_v, dst_v, dst_stride_v, dst_u, dst_stride_u,
                  halfwidth, halfheight, rotation);
    return 0;
  }
  // Split UV planes - NV12
  if (src_pixel_stride_uv == 2 && vu_off == 1 && src_stride_u == src_stride_v) {
    SplitRotateUV(src_u, src_stride_u, dst_u, dst_stride_u, dst_v, dst_stride_v,
                  halfwidth, halfheight, rotation);
    return 0;
  }

  if (rotation == 0) {
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
  // unsupported type and/or rotation.
  return -1;
}

LIBYUV_API
int I010Rotate(const uint16_t* src_y,
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
               int height,
               enum RotationMode mode) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if (!src_y || !src_u || !src_v || width <= 0 || height == 0 || !dst_y ||
      !dst_u || !dst_v || dst_stride_y < 0) {
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

  switch (mode) {
    case kRotate0:
      // copy frame
      return I010Copy(src_y, src_stride_y, src_u, src_stride_u, src_v,
                      src_stride_v, dst_y, dst_stride_y, dst_u, dst_stride_u,
                      dst_v, dst_stride_v, width, height);
    case kRotate90:
      RotatePlane90_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      RotatePlane90_16(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth,
                       halfheight);
      RotatePlane90_16(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth,
                       halfheight);
      return 0;
    case kRotate270:
      RotatePlane270_16(src_y, src_stride_y, dst_y, dst_stride_y, width,
                        height);
      RotatePlane270_16(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth,
                        halfheight);
      RotatePlane270_16(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth,
                        halfheight);
      return 0;
    case kRotate180:
      RotatePlane180_16(src_y, src_stride_y, dst_y, dst_stride_y, width,
                        height);
      RotatePlane180_16(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth,
                        halfheight);
      RotatePlane180_16(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth,
                        halfheight);
      return 0;
    default:
      break;
  }
  return -1;
}

// I210 has half width x full height UV planes, so rotate by 90 and 270
// require scaling to maintain 422 subsampling.
LIBYUV_API
int I210Rotate(const uint16_t* src_y,
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
               int height,
               enum RotationMode mode) {
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  int r;
  if (!src_y || !src_u || !src_v || width <= 0 || height == 0 || !dst_y ||
      !dst_u || !dst_v) {
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

  switch (mode) {
    case kRotate0:
      // Copy frame
      CopyPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      CopyPlane_16(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth, height);
      CopyPlane_16(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth, height);
      return 0;

      // Note on temporary Y plane for UV.
      // Rotation of UV first fits within the Y destination plane rows.
      // Y plane is width x height
      // Y plane rotated is height x width
      // UV plane is (width / 2) x height
      // UV plane rotated is height x (width / 2)
      // UV plane rotated+scaled is (height / 2) x width.
      // UV plane rotated is a temporary that fits within the Y plane rotated.

    case kRotate90:
      RotatePlane90_16(src_u, src_stride_u, dst_y, dst_stride_y, halfwidth,
                       height);
      r = ScalePlane_16(dst_y, dst_stride_y, height, halfwidth, dst_u,
                        dst_stride_u, halfheight, width, kFilterBilinear);
      if (r != 0) {
        return r;
      }
      RotatePlane90_16(src_v, src_stride_v, dst_y, dst_stride_y, halfwidth,
                       height);
      r = ScalePlane_16(dst_y, dst_stride_y, height, halfwidth, dst_v,
                        dst_stride_v, halfheight, width, kFilterLinear);
      if (r != 0) {
        return r;
      }
      RotatePlane90_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      return 0;
    case kRotate270:
      RotatePlane270_16(src_u, src_stride_u, dst_y, dst_stride_y, halfwidth,
                        height);
      r = ScalePlane_16(dst_y, dst_stride_y, height, halfwidth, dst_u,
                        dst_stride_u, halfheight, width, kFilterBilinear);
      if (r != 0) {
        return r;
      }
      RotatePlane270_16(src_v, src_stride_v, dst_y, dst_stride_y, halfwidth,
                        height);
      r = ScalePlane_16(dst_y, dst_stride_y, height, halfwidth, dst_v,
                        dst_stride_v, halfheight, width, kFilterLinear);
      if (r != 0) {
        return r;
      }
      RotatePlane270_16(src_y, src_stride_y, dst_y, dst_stride_y, width,
                        height);
      return 0;
    case kRotate180:
      RotatePlane180_16(src_y, src_stride_y, dst_y, dst_stride_y, width,
                        height);
      RotatePlane180_16(src_u, src_stride_u, dst_u, dst_stride_u, halfwidth,
                        height);
      RotatePlane180_16(src_v, src_stride_v, dst_v, dst_stride_v, halfwidth,
                        height);
      return 0;
    default:
      break;
  }
  return -1;
}

LIBYUV_API
int I410Rotate(const uint16_t* src_y,
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
               int height,
               enum RotationMode mode) {
  if (!src_y || !src_u || !src_v || width <= 0 || height == 0 || !dst_y ||
      !dst_u || !dst_v || dst_stride_y < 0) {
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

  switch (mode) {
    case kRotate0:
      // copy frame
      CopyPlane_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      CopyPlane_16(src_u, src_stride_u, dst_u, dst_stride_u, width, height);
      CopyPlane_16(src_v, src_stride_v, dst_v, dst_stride_v, width, height);
      return 0;
    case kRotate90:
      RotatePlane90_16(src_y, src_stride_y, dst_y, dst_stride_y, width, height);
      RotatePlane90_16(src_u, src_stride_u, dst_u, dst_stride_u, width, height);
      RotatePlane90_16(src_v, src_stride_v, dst_v, dst_stride_v, width, height);
      return 0;
    case kRotate270:
      RotatePlane270_16(src_y, src_stride_y, dst_y, dst_stride_y, width,
                        height);
      RotatePlane270_16(src_u, src_stride_u, dst_u, dst_stride_u, width,
                        height);
      RotatePlane270_16(src_v, src_stride_v, dst_v, dst_stride_v, width,
                        height);
      return 0;
    case kRotate180:
      RotatePlane180_16(src_y, src_stride_y, dst_y, dst_stride_y, width,
                        height);
      RotatePlane180_16(src_u, src_stride_u, dst_u, dst_stride_u, width,
                        height);
      RotatePlane180_16(src_v, src_stride_v, dst_v, dst_stride_v, width,
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
