/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/convert_argb.h"

#include "libyuv/cpu_id.h"
#ifdef HAVE_JPEG
#include "libyuv/mjpeg_decoder.h"
#endif
#include "libyuv/planar_functions.h"  // For CopyPlane and ARGBShuffle.
#include "libyuv/rotate_argb.h"
#include "libyuv/row.h"
#include "libyuv/video_common.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Copy ARGB with optional flipping
LIBYUV_API
int ARGBCopy(const uint8_t* src_argb,
             int src_stride_argb,
             uint8_t* dst_argb,
             int dst_stride_argb,
             int width,
             int height) {
  if (!src_argb || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }

  CopyPlane(src_argb, src_stride_argb, dst_argb, dst_stride_argb, width * 4,
            height);
  return 0;
}

// Convert I420 to ARGB with matrix
static int I420ToARGBMatrix(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_u,
                            int src_stride_u,
                            const uint8_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_argb,
                            int dst_stride_argb,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height) {
  int y;
  void (*I422ToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                        const uint8_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I422ToARGBRow_C;
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
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

  for (y = 0; y < height; ++y) {
    I422ToARGBRow(src_y, src_u, src_v, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to ARGB.
LIBYUV_API
int I420ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I420ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvI601Constants, width, height);
}

// Convert I420 to ABGR.
LIBYUV_API
int I420ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I420ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuI601Constants,  // Use Yvu matrix
                          width, height);
}

// Convert J420 to ARGB.
LIBYUV_API
int J420ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I420ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvJPEGConstants, width, height);
}

// Convert J420 to ABGR.
LIBYUV_API
int J420ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I420ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuJPEGConstants,  // Use Yvu matrix
                          width, height);
}

// Convert H420 to ARGB.
LIBYUV_API
int H420ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I420ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvH709Constants, width, height);
}

// Convert H420 to ABGR.
LIBYUV_API
int H420ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I420ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuH709Constants,  // Use Yvu matrix
                          width, height);
}

// Convert I422 to ARGB with matrix
static int I422ToARGBMatrix(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_u,
                            int src_stride_u,
                            const uint8_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_argb,
                            int dst_stride_argb,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height) {
  int y;
  void (*I422ToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                        const uint8_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I422ToARGBRow_C;
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_y == width && src_stride_u * 2 == width &&
      src_stride_v * 2 == width && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_argb = 0;
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

  for (y = 0; y < height; ++y) {
    I422ToARGBRow(src_y, src_u, src_v, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I422 to ARGB.
LIBYUV_API
int I422ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I422ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvI601Constants, width, height);
}

// Convert I422 to ABGR.
LIBYUV_API
int I422ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I422ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuI601Constants,  // Use Yvu matrix
                          width, height);
}

// Convert J422 to ARGB.
LIBYUV_API
int J422ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I422ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvJPEGConstants, width, height);
}

// Convert J422 to ABGR.
LIBYUV_API
int J422ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I422ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuJPEGConstants,  // Use Yvu matrix
                          width, height);
}

// Convert H422 to ARGB.
LIBYUV_API
int H422ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I422ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvH709Constants, width, height);
}

// Convert H422 to ABGR.
LIBYUV_API
int H422ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I422ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuH709Constants,  // Use Yvu matrix
                          width, height);
}

// Convert 10 bit YUV to ARGB with matrix
// TODO(fbarchard): Consider passing scale multiplier to I210ToARGB to
// multiply 10 bit yuv into high bits to allow any number of bits.
static int I010ToAR30Matrix(const uint16_t* src_y,
                            int src_stride_y,
                            const uint16_t* src_u,
                            int src_stride_u,
                            const uint16_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_ar30,
                            int dst_stride_ar30,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height) {
  int y;
  void (*I210ToAR30Row)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I210ToAR30Row_C;
  if (!src_y || !src_u || !src_v || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_I210TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I210ToAR30Row = I210ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I210ToAR30Row = I210ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I210TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I210ToAR30Row = I210ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I210ToAR30Row = I210ToAR30Row_AVX2;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    I210ToAR30Row(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I010 to AR30.
LIBYUV_API
int I010ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  return I010ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_ar30, dst_stride_ar30,
                          &kYuvI601Constants, width, height);
}

// Convert H010 to AR30.
LIBYUV_API
int H010ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  return I010ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_ar30, dst_stride_ar30,
                          &kYuvH709Constants, width, height);
}

// Convert I010 to AB30.
LIBYUV_API
int I010ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height) {
  return I010ToAR30Matrix(src_y, src_stride_y, src_v, src_stride_v, src_u,
                          src_stride_u, dst_ab30, dst_stride_ab30,
                          &kYvuI601Constants, width, height);
}

// Convert H010 to AB30.
LIBYUV_API
int H010ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height) {
  return I010ToAR30Matrix(src_y, src_stride_y, src_v, src_stride_v, src_u,
                          src_stride_u, dst_ab30, dst_stride_ab30,
                          &kYvuH709Constants, width, height);
}

// Convert 10 bit YUV to ARGB with matrix
static int I010ToARGBMatrix(const uint16_t* src_y,
                            int src_stride_y,
                            const uint16_t* src_u,
                            int src_stride_u,
                            const uint16_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_argb,
                            int dst_stride_argb,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height) {
  int y;
  void (*I210ToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I210ToARGBRow_C;
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I210TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I210ToARGBRow = I210ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I210ToARGBRow = I210ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I210TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I210ToARGBRow = I210ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I210ToARGBRow = I210ToARGBRow_AVX2;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    I210ToARGBRow(src_y, src_u, src_v, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I010 to ARGB.
LIBYUV_API
int I010ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I010ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvI601Constants, width, height);
}

// Convert I010 to ABGR.
LIBYUV_API
int I010ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I010ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuI601Constants,  // Use Yvu matrix
                          width, height);
}

// Convert H010 to ARGB.
LIBYUV_API
int H010ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I010ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvH709Constants, width, height);
}

// Convert H010 to ABGR.
LIBYUV_API
int H010ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I010ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuH709Constants,  // Use Yvu matrix
                          width, height);
}

// Convert I444 to ARGB with matrix
static int I444ToARGBMatrix(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_u,
                            int src_stride_u,
                            const uint8_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_argb,
                            int dst_stride_argb,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height) {
  int y;
  void (*I444ToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                        const uint8_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I444ToARGBRow_C;
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_y == width && src_stride_u == width && src_stride_v == width &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_argb = 0;
  }
#if defined(HAS_I444TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I444ToARGBRow = I444ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I444ToARGBRow = I444ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I444TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I444ToARGBRow = I444ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I444ToARGBRow = I444ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I444TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I444ToARGBRow = I444ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I444ToARGBRow = I444ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I444TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I444ToARGBRow = I444ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I444ToARGBRow = I444ToARGBRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I444ToARGBRow(src_y, src_u, src_v, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I444 to ARGB.
LIBYUV_API
int I444ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I444ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvI601Constants, width, height);
}

// Convert I444 to ABGR.
LIBYUV_API
int I444ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I444ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuI601Constants,  // Use Yvu matrix
                          width, height);
}

// Convert J444 to ARGB.
LIBYUV_API
int J444ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I444ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvJPEGConstants, width, height);
}

// Convert I420 with Alpha to preattenuated ARGB.
static int I420AlphaToARGBMatrix(const uint8_t* src_y,
                                 int src_stride_y,
                                 const uint8_t* src_u,
                                 int src_stride_u,
                                 const uint8_t* src_v,
                                 int src_stride_v,
                                 const uint8_t* src_a,
                                 int src_stride_a,
                                 uint8_t* dst_argb,
                                 int dst_stride_argb,
                                 const struct YuvConstants* yuvconstants,
                                 int width,
                                 int height,
                                 int attenuate) {
  int y;
  void (*I422AlphaToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                             const uint8_t* v_buf, const uint8_t* a_buf,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) = I422AlphaToARGBRow_C;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I422ALPHATOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422AlphaToARGBRow = I422AlphaToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I422ALPHATOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422AlphaToARGBRow = I422AlphaToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I422ALPHATOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422AlphaToARGBRow = I422AlphaToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I422ALPHATOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422AlphaToARGBRow = I422AlphaToARGBRow_MSA;
    }
  }
#endif
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
    I422AlphaToARGBRow(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_a += src_stride_a;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 with Alpha to ARGB.
LIBYUV_API
int I420AlphaToARGB(const uint8_t* src_y,
                    int src_stride_y,
                    const uint8_t* src_u,
                    int src_stride_u,
                    const uint8_t* src_v,
                    int src_stride_v,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height,
                    int attenuate) {
  return I420AlphaToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                               src_stride_v, src_a, src_stride_a, dst_argb,
                               dst_stride_argb, &kYuvI601Constants, width,
                               height, attenuate);
}

// Convert I420 with Alpha to ABGR.
LIBYUV_API
int I420AlphaToABGR(const uint8_t* src_y,
                    int src_stride_y,
                    const uint8_t* src_u,
                    int src_stride_u,
                    const uint8_t* src_v,
                    int src_stride_v,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_abgr,
                    int dst_stride_abgr,
                    int width,
                    int height,
                    int attenuate) {
  return I420AlphaToARGBMatrix(
      src_y, src_stride_y, src_v, src_stride_v,  // Swap U and V
      src_u, src_stride_u, src_a, src_stride_a, dst_abgr, dst_stride_abgr,
      &kYvuI601Constants,  // Use Yvu matrix
      width, height, attenuate);
}

// Convert I400 to ARGB.
LIBYUV_API
int I400ToARGB(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*I400ToARGBRow)(const uint8_t* y_buf, uint8_t* rgb_buf, int width) =
      I400ToARGBRow_C;
  if (!src_y || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_y == width && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_argb = 0;
  }
#if defined(HAS_I400TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    I400ToARGBRow = I400ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      I400ToARGBRow = I400ToARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_I400TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I400ToARGBRow = I400ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I400ToARGBRow = I400ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I400TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I400ToARGBRow = I400ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I400ToARGBRow = I400ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I400TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I400ToARGBRow = I400ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      I400ToARGBRow = I400ToARGBRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I400ToARGBRow(src_y, dst_argb, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
  }
  return 0;
}

// Convert J400 to ARGB.
LIBYUV_API
int J400ToARGB(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*J400ToARGBRow)(const uint8_t* src_y, uint8_t* dst_argb, int width) =
      J400ToARGBRow_C;
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
#if defined(HAS_J400TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    J400ToARGBRow = J400ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      J400ToARGBRow = J400ToARGBRow_SSE2;
    }
  }
#endif
#if defined(HAS_J400TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    J400ToARGBRow = J400ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      J400ToARGBRow = J400ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_J400TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    J400ToARGBRow = J400ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      J400ToARGBRow = J400ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_J400TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    J400ToARGBRow = J400ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      J400ToARGBRow = J400ToARGBRow_MSA;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    J400ToARGBRow(src_y, dst_argb, width);
    src_y += src_stride_y;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Shuffle table for converting BGRA to ARGB.
static const uvec8 kShuffleMaskBGRAToARGB = {
    3u, 2u, 1u, 0u, 7u, 6u, 5u, 4u, 11u, 10u, 9u, 8u, 15u, 14u, 13u, 12u};

// Shuffle table for converting ABGR to ARGB.
static const uvec8 kShuffleMaskABGRToARGB = {
    2u, 1u, 0u, 3u, 6u, 5u, 4u, 7u, 10u, 9u, 8u, 11u, 14u, 13u, 12u, 15u};

// Shuffle table for converting RGBA to ARGB.
static const uvec8 kShuffleMaskRGBAToARGB = {
    1u, 2u, 3u, 0u, 5u, 6u, 7u, 4u, 9u, 10u, 11u, 8u, 13u, 14u, 15u, 12u};

// Convert BGRA to ARGB.
LIBYUV_API
int BGRAToARGB(const uint8_t* src_bgra,
               int src_stride_bgra,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return ARGBShuffle(src_bgra, src_stride_bgra, dst_argb, dst_stride_argb,
                     (const uint8_t*)(&kShuffleMaskBGRAToARGB), width, height);
}

// Convert ARGB to BGRA (same as BGRAToARGB).
LIBYUV_API
int ARGBToBGRA(const uint8_t* src_bgra,
               int src_stride_bgra,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return ARGBShuffle(src_bgra, src_stride_bgra, dst_argb, dst_stride_argb,
                     (const uint8_t*)(&kShuffleMaskBGRAToARGB), width, height);
}

// Convert ABGR to ARGB.
LIBYUV_API
int ABGRToARGB(const uint8_t* src_abgr,
               int src_stride_abgr,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return ARGBShuffle(src_abgr, src_stride_abgr, dst_argb, dst_stride_argb,
                     (const uint8_t*)(&kShuffleMaskABGRToARGB), width, height);
}

// Convert ARGB to ABGR to (same as ABGRToARGB).
LIBYUV_API
int ARGBToABGR(const uint8_t* src_abgr,
               int src_stride_abgr,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return ARGBShuffle(src_abgr, src_stride_abgr, dst_argb, dst_stride_argb,
                     (const uint8_t*)(&kShuffleMaskABGRToARGB), width, height);
}

// Convert RGBA to ARGB.
LIBYUV_API
int RGBAToARGB(const uint8_t* src_rgba,
               int src_stride_rgba,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return ARGBShuffle(src_rgba, src_stride_rgba, dst_argb, dst_stride_argb,
                     (const uint8_t*)(&kShuffleMaskRGBAToARGB), width, height);
}

// Convert RGB24 to ARGB.
LIBYUV_API
int RGB24ToARGB(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_argb,
                int dst_stride_argb,
                int width,
                int height) {
  int y;
  void (*RGB24ToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RGB24ToARGBRow_C;
  if (!src_rgb24 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb24 = src_rgb24 + (height - 1) * src_stride_rgb24;
    src_stride_rgb24 = -src_stride_rgb24;
  }
  // Coalesce rows.
  if (src_stride_rgb24 == width * 3 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_rgb24 = dst_stride_argb = 0;
  }
#if defined(HAS_RGB24TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToARGBRow = RGB24ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RGB24TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RGB24ToARGBRow = RGB24ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_RGB24TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToARGBRow = RGB24ToARGBRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    RGB24ToARGBRow(src_rgb24, dst_argb, width);
    src_rgb24 += src_stride_rgb24;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert RAW to ARGB.
LIBYUV_API
int RAWToARGB(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_argb,
              int dst_stride_argb,
              int width,
              int height) {
  int y;
  void (*RAWToARGBRow)(const uint8_t* src_rgb, uint8_t* dst_argb, int width) =
      RAWToARGBRow_C;
  if (!src_raw || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }
  // Coalesce rows.
  if (src_stride_raw == width * 3 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_raw = dst_stride_argb = 0;
  }
#if defined(HAS_RAWTOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RAWToARGBRow = RAWToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RAWToARGBRow = RAWToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RAWToARGBRow = RAWToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RAWToARGBRow = RAWToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    RAWToARGBRow(src_raw, dst_argb, width);
    src_raw += src_stride_raw;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert RGB565 to ARGB.
LIBYUV_API
int RGB565ToARGB(const uint8_t* src_rgb565,
                 int src_stride_rgb565,
                 uint8_t* dst_argb,
                 int dst_stride_argb,
                 int width,
                 int height) {
  int y;
  void (*RGB565ToARGBRow)(const uint8_t* src_rgb565, uint8_t* dst_argb,
                          int width) = RGB565ToARGBRow_C;
  if (!src_rgb565 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb565 = src_rgb565 + (height - 1) * src_stride_rgb565;
    src_stride_rgb565 = -src_stride_rgb565;
  }
  // Coalesce rows.
  if (src_stride_rgb565 == width * 2 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_rgb565 = dst_stride_argb = 0;
  }
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
#if defined(HAS_RGB565TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RGB565ToARGBRow = RGB565ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_RGB565TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      RGB565ToARGBRow = RGB565ToARGBRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    RGB565ToARGBRow(src_rgb565, dst_argb, width);
    src_rgb565 += src_stride_rgb565;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert ARGB1555 to ARGB.
LIBYUV_API
int ARGB1555ToARGB(const uint8_t* src_argb1555,
                   int src_stride_argb1555,
                   uint8_t* dst_argb,
                   int dst_stride_argb,
                   int width,
                   int height) {
  int y;
  void (*ARGB1555ToARGBRow)(const uint8_t* src_argb1555, uint8_t* dst_argb,
                            int width) = ARGB1555ToARGBRow_C;
  if (!src_argb1555 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb1555 = src_argb1555 + (height - 1) * src_stride_argb1555;
    src_stride_argb1555 = -src_stride_argb1555;
  }
  // Coalesce rows.
  if (src_stride_argb1555 == width * 2 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb1555 = dst_stride_argb = 0;
  }
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
#if defined(HAS_ARGB1555TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGB1555TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGB1555ToARGBRow(src_argb1555, dst_argb, width);
    src_argb1555 += src_stride_argb1555;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert ARGB4444 to ARGB.
LIBYUV_API
int ARGB4444ToARGB(const uint8_t* src_argb4444,
                   int src_stride_argb4444,
                   uint8_t* dst_argb,
                   int dst_stride_argb,
                   int width,
                   int height) {
  int y;
  void (*ARGB4444ToARGBRow)(const uint8_t* src_argb4444, uint8_t* dst_argb,
                            int width) = ARGB4444ToARGBRow_C;
  if (!src_argb4444 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb4444 = src_argb4444 + (height - 1) * src_stride_argb4444;
    src_stride_argb4444 = -src_stride_argb4444;
  }
  // Coalesce rows.
  if (src_stride_argb4444 == width * 2 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb4444 = dst_stride_argb = 0;
  }
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
#if defined(HAS_ARGB4444TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_NEON;
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

  for (y = 0; y < height; ++y) {
    ARGB4444ToARGBRow(src_argb4444, dst_argb, width);
    src_argb4444 += src_stride_argb4444;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert AR30 to ARGB.
LIBYUV_API
int AR30ToARGB(const uint8_t* src_ar30,
               int src_stride_ar30,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  if (!src_ar30 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_ar30 = src_ar30 + (height - 1) * src_stride_ar30;
    src_stride_ar30 = -src_stride_ar30;
  }
  // Coalesce rows.
  if (src_stride_ar30 == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_ar30 = dst_stride_argb = 0;
  }
  for (y = 0; y < height; ++y) {
    AR30ToARGBRow_C(src_ar30, dst_argb, width);
    src_ar30 += src_stride_ar30;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert AR30 to ABGR.
LIBYUV_API
int AR30ToABGR(const uint8_t* src_ar30,
               int src_stride_ar30,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  int y;
  if (!src_ar30 || !dst_abgr || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_ar30 = src_ar30 + (height - 1) * src_stride_ar30;
    src_stride_ar30 = -src_stride_ar30;
  }
  // Coalesce rows.
  if (src_stride_ar30 == width * 4 && dst_stride_abgr == width * 4) {
    width *= height;
    height = 1;
    src_stride_ar30 = dst_stride_abgr = 0;
  }
  for (y = 0; y < height; ++y) {
    AR30ToABGRRow_C(src_ar30, dst_abgr, width);
    src_ar30 += src_stride_ar30;
    dst_abgr += dst_stride_abgr;
  }
  return 0;
}

// Convert AR30 to AB30.
LIBYUV_API
int AR30ToAB30(const uint8_t* src_ar30,
               int src_stride_ar30,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height) {
  int y;
  if (!src_ar30 || !dst_ab30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_ar30 = src_ar30 + (height - 1) * src_stride_ar30;
    src_stride_ar30 = -src_stride_ar30;
  }
  // Coalesce rows.
  if (src_stride_ar30 == width * 4 && dst_stride_ab30 == width * 4) {
    width *= height;
    height = 1;
    src_stride_ar30 = dst_stride_ab30 = 0;
  }
  for (y = 0; y < height; ++y) {
    AR30ToAB30Row_C(src_ar30, dst_ab30, width);
    src_ar30 += src_stride_ar30;
    dst_ab30 += dst_stride_ab30;
  }
  return 0;
}

// Convert NV12 to ARGB with matrix
static int NV12ToARGBMatrix(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_uv,
                            int src_stride_uv,
                            uint8_t* dst_argb,
                            int dst_stride_argb,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height) {
  int y;
  void (*NV12ToARGBRow)(
      const uint8_t* y_buf, const uint8_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = NV12ToARGBRow_C;
  if (!src_y || !src_uv || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_NV12TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_NV12TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      NV12ToARGBRow = NV12ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_NV12TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_NV12TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    NV12ToARGBRow(src_y, src_uv, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    if (y & 1) {
      src_uv += src_stride_uv;
    }
  }
  return 0;
}

// Convert NV21 to ARGB with matrix
static int NV21ToARGBMatrix(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_vu,
                            int src_stride_vu,
                            uint8_t* dst_argb,
                            int dst_stride_argb,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height) {
  int y;
  void (*NV21ToARGBRow)(
      const uint8_t* y_buf, const uint8_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = NV21ToARGBRow_C;
  if (!src_y || !src_vu || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_NV21TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    NV21ToARGBRow = NV21ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      NV21ToARGBRow = NV21ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_NV21TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    NV21ToARGBRow = NV21ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      NV21ToARGBRow = NV21ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_NV21TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    NV21ToARGBRow = NV21ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      NV21ToARGBRow = NV21ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_NV21TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    NV21ToARGBRow = NV21ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      NV21ToARGBRow = NV21ToARGBRow_MSA;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    NV21ToARGBRow(src_y, src_vu, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    if (y & 1) {
      src_vu += src_stride_vu;
    }
  }
  return 0;
}

// Convert NV12 to ARGB.
LIBYUV_API
int NV12ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return NV12ToARGBMatrix(src_y, src_stride_y, src_uv, src_stride_uv, dst_argb,
                          dst_stride_argb, &kYuvI601Constants, width, height);
}

// Convert NV21 to ARGB.
LIBYUV_API
int NV21ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_vu,
               int src_stride_vu,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return NV21ToARGBMatrix(src_y, src_stride_y, src_vu, src_stride_vu, dst_argb,
                          dst_stride_argb, &kYuvI601Constants, width, height);
}

// Convert NV12 to ABGR.
// To output ABGR instead of ARGB swap the UV and use a mirrrored yuc matrix.
// To swap the UV use NV12 instead of NV21.LIBYUV_API
int NV12ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return NV21ToARGBMatrix(src_y, src_stride_y, src_uv, src_stride_uv, dst_abgr,
                          dst_stride_abgr, &kYvuI601Constants, width, height);
}

// Convert NV21 to ABGR.
LIBYUV_API
int NV21ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_vu,
               int src_stride_vu,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return NV12ToARGBMatrix(src_y, src_stride_y, src_vu, src_stride_vu, dst_abgr,
                          dst_stride_abgr, &kYvuI601Constants, width, height);
}

// TODO(fbarchard): Consider SSSE3 2 step conversion.
// Convert NV12 to RGB24 with matrix
static int NV12ToRGB24Matrix(const uint8_t* src_y,
                             int src_stride_y,
                             const uint8_t* src_uv,
                             int src_stride_uv,
                             uint8_t* dst_rgb24,
                             int dst_stride_rgb24,
                             const struct YuvConstants* yuvconstants,
                             int width,
                             int height) {
  int y;
  void (*NV12ToRGB24Row)(
      const uint8_t* y_buf, const uint8_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = NV12ToRGB24Row_C;
  if (!src_y || !src_uv || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb24 = dst_rgb24 + (height - 1) * dst_stride_rgb24;
    dst_stride_rgb24 = -dst_stride_rgb24;
  }
#if defined(HAS_NV12TORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    NV12ToRGB24Row = NV12ToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      NV12ToRGB24Row = NV12ToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_NV12TORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    NV12ToRGB24Row = NV12ToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      NV12ToRGB24Row = NV12ToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_NV12TORGB24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    NV12ToRGB24Row = NV12ToRGB24Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      NV12ToRGB24Row = NV12ToRGB24Row_AVX2;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    NV12ToRGB24Row(src_y, src_uv, dst_rgb24, yuvconstants, width);
    dst_rgb24 += dst_stride_rgb24;
    src_y += src_stride_y;
    if (y & 1) {
      src_uv += src_stride_uv;
    }
  }
  return 0;
}

// Convert NV21 to RGB24 with matrix
static int NV21ToRGB24Matrix(const uint8_t* src_y,
                             int src_stride_y,
                             const uint8_t* src_vu,
                             int src_stride_vu,
                             uint8_t* dst_rgb24,
                             int dst_stride_rgb24,
                             const struct YuvConstants* yuvconstants,
                             int width,
                             int height) {
  int y;
  void (*NV21ToRGB24Row)(
      const uint8_t* y_buf, const uint8_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = NV21ToRGB24Row_C;
  if (!src_y || !src_vu || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb24 = dst_rgb24 + (height - 1) * dst_stride_rgb24;
    dst_stride_rgb24 = -dst_stride_rgb24;
  }
#if defined(HAS_NV21TORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    NV21ToRGB24Row = NV21ToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      NV21ToRGB24Row = NV21ToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_NV21TORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    NV21ToRGB24Row = NV21ToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      NV21ToRGB24Row = NV21ToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_NV21TORGB24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    NV21ToRGB24Row = NV21ToRGB24Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      NV21ToRGB24Row = NV21ToRGB24Row_AVX2;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    NV21ToRGB24Row(src_y, src_vu, dst_rgb24, yuvconstants, width);
    dst_rgb24 += dst_stride_rgb24;
    src_y += src_stride_y;
    if (y & 1) {
      src_vu += src_stride_vu;
    }
  }
  return 0;
}

// TODO(fbarchard): NV12ToRAW can be implemented by mirrored matrix.
// Convert NV12 to RGB24.
LIBYUV_API
int NV12ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_uv,
                int src_stride_uv,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  return NV12ToRGB24Matrix(src_y, src_stride_y, src_uv, src_stride_uv,
                           dst_rgb24, dst_stride_rgb24, &kYuvI601Constants,
                           width, height);
}

// Convert NV21 to RGB24.
LIBYUV_API
int NV21ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_vu,
                int src_stride_vu,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  return NV21ToRGB24Matrix(src_y, src_stride_y, src_vu, src_stride_vu,
                           dst_rgb24, dst_stride_rgb24, &kYuvI601Constants,
                           width, height);
}

// Convert M420 to ARGB.
LIBYUV_API
int M420ToARGB(const uint8_t* src_m420,
               int src_stride_m420,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*NV12ToARGBRow)(
      const uint8_t* y_buf, const uint8_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = NV12ToARGBRow_C;
  if (!src_m420 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_NV12TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_NV12TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      NV12ToARGBRow = NV12ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_NV12TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_NV12TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_MSA;
    }
  }
#endif

  for (y = 0; y < height - 1; y += 2) {
    NV12ToARGBRow(src_m420, src_m420 + src_stride_m420 * 2, dst_argb,
                  &kYuvI601Constants, width);
    NV12ToARGBRow(src_m420 + src_stride_m420, src_m420 + src_stride_m420 * 2,
                  dst_argb + dst_stride_argb, &kYuvI601Constants, width);
    dst_argb += dst_stride_argb * 2;
    src_m420 += src_stride_m420 * 3;
  }
  if (height & 1) {
    NV12ToARGBRow(src_m420, src_m420 + src_stride_m420 * 2, dst_argb,
                  &kYuvI601Constants, width);
  }
  return 0;
}

// Convert YUY2 to ARGB.
LIBYUV_API
int YUY2ToARGB(const uint8_t* src_yuy2,
               int src_stride_yuy2,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*YUY2ToARGBRow)(const uint8_t* src_yuy2, uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants, int width) =
      YUY2ToARGBRow_C;
  if (!src_yuy2 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_yuy2 = src_yuy2 + (height - 1) * src_stride_yuy2;
    src_stride_yuy2 = -src_stride_yuy2;
  }
  // Coalesce rows.
  if (src_stride_yuy2 == width * 2 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_yuy2 = dst_stride_argb = 0;
  }
#if defined(HAS_YUY2TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    YUY2ToARGBRow = YUY2ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToARGBRow = YUY2ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_YUY2TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    YUY2ToARGBRow = YUY2ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      YUY2ToARGBRow = YUY2ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_YUY2TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    YUY2ToARGBRow = YUY2ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      YUY2ToARGBRow = YUY2ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_YUY2TOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    YUY2ToARGBRow = YUY2ToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      YUY2ToARGBRow = YUY2ToARGBRow_MSA;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    YUY2ToARGBRow(src_yuy2, dst_argb, &kYuvI601Constants, width);
    src_yuy2 += src_stride_yuy2;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert UYVY to ARGB.
LIBYUV_API
int UYVYToARGB(const uint8_t* src_uyvy,
               int src_stride_uyvy,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*UYVYToARGBRow)(const uint8_t* src_uyvy, uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants, int width) =
      UYVYToARGBRow_C;
  if (!src_uyvy || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uyvy = src_uyvy + (height - 1) * src_stride_uyvy;
    src_stride_uyvy = -src_stride_uyvy;
  }
  // Coalesce rows.
  if (src_stride_uyvy == width * 2 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_uyvy = dst_stride_argb = 0;
  }
#if defined(HAS_UYVYTOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    UYVYToARGBRow = UYVYToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      UYVYToARGBRow = UYVYToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_UYVYTOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    UYVYToARGBRow = UYVYToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      UYVYToARGBRow = UYVYToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_UYVYTOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    UYVYToARGBRow = UYVYToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      UYVYToARGBRow = UYVYToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_UYVYTOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    UYVYToARGBRow = UYVYToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      UYVYToARGBRow = UYVYToARGBRow_MSA;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    UYVYToARGBRow(src_uyvy, dst_argb, &kYuvI601Constants, width);
    src_uyvy += src_stride_uyvy;
    dst_argb += dst_stride_argb;
  }
  return 0;
}
static void WeavePixels(const uint8_t* src_u,
                        const uint8_t* src_v,
                        int src_pixel_stride_uv,
                        uint8_t* dst_uv,
                        int width) {
  int i;
  for (i = 0; i < width; ++i) {
    dst_uv[0] = *src_u;
    dst_uv[1] = *src_v;
    dst_uv += 2;
    src_u += src_pixel_stride_uv;
    src_v += src_pixel_stride_uv;
  }
}

// Convert Android420 to ARGB.
LIBYUV_API
int Android420ToARGBMatrix(const uint8_t* src_y,
                           int src_stride_y,
                           const uint8_t* src_u,
                           int src_stride_u,
                           const uint8_t* src_v,
                           int src_stride_v,
                           int src_pixel_stride_uv,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height) {
  int y;
  uint8_t* dst_uv;
  const ptrdiff_t vu_off = src_v - src_u;
  int halfwidth = (width + 1) >> 1;
  int halfheight = (height + 1) >> 1;
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    halfheight = (height + 1) >> 1;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }

  // I420
  if (src_pixel_stride_uv == 1) {
    return I420ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                            src_stride_v, dst_argb, dst_stride_argb,
                            yuvconstants, width, height);
    // NV21
  }
  if (src_pixel_stride_uv == 2 && vu_off == -1 &&
      src_stride_u == src_stride_v) {
    return NV21ToARGBMatrix(src_y, src_stride_y, src_v, src_stride_v, dst_argb,
                            dst_stride_argb, yuvconstants, width, height);
    // NV12
  }
  if (src_pixel_stride_uv == 2 && vu_off == 1 && src_stride_u == src_stride_v) {
    return NV12ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, dst_argb,
                            dst_stride_argb, yuvconstants, width, height);
  }

  // General case fallback creates NV12
  align_buffer_64(plane_uv, halfwidth * 2 * halfheight);
  dst_uv = plane_uv;
  for (y = 0; y < halfheight; ++y) {
    WeavePixels(src_u, src_v, src_pixel_stride_uv, dst_uv, halfwidth);
    src_u += src_stride_u;
    src_v += src_stride_v;
    dst_uv += halfwidth * 2;
  }
  NV12ToARGBMatrix(src_y, src_stride_y, plane_uv, halfwidth * 2, dst_argb,
                   dst_stride_argb, yuvconstants, width, height);
  free_aligned_buffer_64(plane_uv);
  return 0;
}

// Convert Android420 to ARGB.
LIBYUV_API
int Android420ToARGB(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     int src_pixel_stride_uv,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     int width,
                     int height) {
  return Android420ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                                src_stride_v, src_pixel_stride_uv, dst_argb,
                                dst_stride_argb, &kYuvI601Constants, width,
                                height);
}

// Convert Android420 to ABGR.
LIBYUV_API
int Android420ToABGR(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     int src_pixel_stride_uv,
                     uint8_t* dst_abgr,
                     int dst_stride_abgr,
                     int width,
                     int height) {
  return Android420ToARGBMatrix(src_y, src_stride_y, src_v, src_stride_v, src_u,
                                src_stride_u, src_pixel_stride_uv, dst_abgr,
                                dst_stride_abgr, &kYvuI601Constants, width,
                                height);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
