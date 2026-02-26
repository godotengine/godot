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

#include <assert.h>

#include "libyuv/convert_from_argb.h"
#include "libyuv/cpu_id.h"
#include "libyuv/planar_functions.h"  // For CopyPlane and ARGBShuffle.
#include "libyuv/rotate_argb.h"
#include "libyuv/row.h"
#include "libyuv/scale_row.h"  // For ScaleRowUp2_Linear and ScaleRowUp2_Bilinear
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

// Convert I420 to ARGB with matrix.
LIBYUV_API
int I420ToARGBMatrix(const uint8_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_I422TOARGBROW_AVX512BW)
  if (TestCpuFlag(kCpuHasAVX512BW | kCpuHasAVX512VL) ==
      (kCpuHasAVX512BW | kCpuHasAVX512VL)) {
    I422ToARGBRow = I422ToARGBRow_Any_AVX512BW;
    if (IS_ALIGNED(width, 32)) {
      I422ToARGBRow = I422ToARGBRow_AVX512BW;
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
#if defined(HAS_I422TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToARGBRow = I422ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I422TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToARGBRow = I422ToARGBRow_SME;
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
#if defined(HAS_I422TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToARGBRow = I422ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGBRow = I422ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToARGBRow = I422ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I422ToARGBRow = I422ToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I422ToARGBRow = I422ToARGBRow_RVV;
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

// Convert U420 to ARGB.
LIBYUV_API
int U420ToARGB(const uint8_t* src_y,
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
                          &kYuv2020Constants, width, height);
}

// Convert U420 to ABGR.
LIBYUV_API
int U420ToABGR(const uint8_t* src_y,
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
                          &kYvu2020Constants,  // Use Yvu matrix
                          width, height);
}

// Convert I422 to ARGB with matrix.
LIBYUV_API
int I422ToARGBMatrix(const uint8_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_I422TOARGBROW_AVX512BW)
  if (TestCpuFlag(kCpuHasAVX512BW | kCpuHasAVX512VL) ==
      (kCpuHasAVX512BW | kCpuHasAVX512VL)) {
    I422ToARGBRow = I422ToARGBRow_Any_AVX512BW;
    if (IS_ALIGNED(width, 32)) {
      I422ToARGBRow = I422ToARGBRow_AVX512BW;
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
#if defined(HAS_I422TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToARGBRow = I422ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I422TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToARGBRow = I422ToARGBRow_SME;
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
#if defined(HAS_I422TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToARGBRow = I422ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGBRow = I422ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToARGBRow = I422ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I422ToARGBRow = I422ToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I422ToARGBRow = I422ToARGBRow_RVV;
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

// Convert U422 to ARGB.
LIBYUV_API
int U422ToARGB(const uint8_t* src_y,
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
                          &kYuv2020Constants, width, height);
}

// Convert U422 to ABGR.
LIBYUV_API
int U422ToABGR(const uint8_t* src_y,
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
                          &kYvu2020Constants,  // Use Yvu matrix
                          width, height);
}

// Convert I444 to ARGB with matrix.
LIBYUV_API
int I444ToARGBMatrix(const uint8_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_I444TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I444ToARGBRow = I444ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I444TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I444ToARGBRow = I444ToARGBRow_SME;
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
#if defined(HAS_I444TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I444ToARGBRow = I444ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I444ToARGBRow = I444ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_I444TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I444ToARGBRow = I444ToARGBRow_RVV;
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

// Convert J444 to ABGR.
LIBYUV_API
int J444ToABGR(const uint8_t* src_y,
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
                          &kYvuJPEGConstants,  // Use Yvu matrix
                          width, height);
}

// Convert H444 to ARGB.
LIBYUV_API
int H444ToARGB(const uint8_t* src_y,
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
                          &kYuvH709Constants, width, height);
}

// Convert H444 to ABGR.
LIBYUV_API
int H444ToABGR(const uint8_t* src_y,
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
                          &kYvuH709Constants,  // Use Yvu matrix
                          width, height);
}

// Convert U444 to ARGB.
LIBYUV_API
int U444ToARGB(const uint8_t* src_y,
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
                          &kYuv2020Constants, width, height);
}

// Convert U444 to ABGR.
LIBYUV_API
int U444ToABGR(const uint8_t* src_y,
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
                          &kYvu2020Constants,  // Use Yvu matrix
                          width, height);
}

// Convert I444 to RGB24 with matrix.
LIBYUV_API
int I444ToRGB24Matrix(const uint8_t* src_y,
                      int src_stride_y,
                      const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_rgb24,
                      int dst_stride_rgb24,
                      const struct YuvConstants* yuvconstants,
                      int width,
                      int height) {
  int y;
  void (*I444ToRGB24Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                         const uint8_t* v_buf, uint8_t* rgb_buf,
                         const struct YuvConstants* yuvconstants, int width) =
      I444ToRGB24Row_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb24 = dst_rgb24 + (height - 1) * dst_stride_rgb24;
    dst_stride_rgb24 = -dst_stride_rgb24;
  }
  // Coalesce rows.
  if (src_stride_y == width && src_stride_u == width && src_stride_v == width &&
      dst_stride_rgb24 == width * 3) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_rgb24 = 0;
  }
#if defined(HAS_I444TORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      I444ToRGB24Row = I444ToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      I444ToRGB24Row = I444ToRGB24Row_AVX2;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I444ToRGB24Row = I444ToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I444ToRGB24Row = I444ToRGB24Row_SVE2;
  }
#endif
#if defined(HAS_I444TORGB24ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I444ToRGB24Row = I444ToRGB24Row_SME;
  }
#endif
#if defined(HAS_I444TORGB24ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I444ToRGB24Row = I444ToRGB24Row_MSA;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I444ToRGB24Row = I444ToRGB24Row_LSX;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I444ToRGB24Row = I444ToRGB24Row_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    I444ToRGB24Row(src_y, src_u, src_v, dst_rgb24, yuvconstants, width);
    dst_rgb24 += dst_stride_rgb24;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I444 to RGB24.
LIBYUV_API
int I444ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  return I444ToRGB24Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_rgb24, dst_stride_rgb24,
                           &kYuvI601Constants, width, height);
}

// Convert I444 to RAW.
LIBYUV_API
int I444ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  return I444ToRGB24Matrix(src_y, src_stride_y, src_v,
                           src_stride_v,  // Swap U and V
                           src_u, src_stride_u, dst_raw, dst_stride_raw,
                           &kYvuI601Constants,  // Use Yvu matrix
                           width, height);
}

// Convert 10 bit YUV to ARGB with matrix.
// TODO(fbarchard): Consider passing scale multiplier to I210ToARGB to
// multiply 10 bit yuv into high bits to allow any number of bits.
LIBYUV_API
int I010ToAR30Matrix(const uint16_t* src_y,
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
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_I210TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I210ToAR30Row = I210ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I210ToAR30Row = I210ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_I210TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I210ToAR30Row = I210ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_I210TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I210ToAR30Row = I210ToAR30Row_SME;
  }
#endif
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

// Convert U010 to AR30.
LIBYUV_API
int U010ToAR30(const uint16_t* src_y,
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
                          &kYuv2020Constants, width, height);
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

// Convert U010 to AB30.
LIBYUV_API
int U010ToAB30(const uint16_t* src_y,
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
                          &kYuv2020Constants, width, height);
}

// Convert 12 bit YUV to ARGB with matrix.
// TODO(fbarchard): Consider passing scale multiplier to I212ToARGB to
// multiply 12 bit yuv into high bits to allow any number of bits.
LIBYUV_API
int I012ToAR30Matrix(const uint16_t* src_y,
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
  void (*I212ToAR30Row)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I212ToAR30Row_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_I212TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I212ToAR30Row = I212ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I212ToAR30Row = I212ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I212TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I212ToAR30Row = I212ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I212ToAR30Row = I212ToAR30Row_AVX2;
    }
  }
#endif
#if defined(HAS_I212TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I212ToAR30Row = I212ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I212ToAR30Row = I212ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_I212TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I212ToAR30Row = I212ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_I212TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I212ToAR30Row = I212ToAR30Row_SME;
  }
#endif
  for (y = 0; y < height; ++y) {
    I212ToAR30Row(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert 10 bit YUV to ARGB with matrix.
// TODO(fbarchard): Consider passing scale multiplier to I210ToARGB to
// multiply 10 bit yuv into high bits to allow any number of bits.
LIBYUV_API
int I210ToAR30Matrix(const uint16_t* src_y,
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
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_I210TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I210ToAR30Row = I210ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I210ToAR30Row = I210ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_I210TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I210ToAR30Row = I210ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_I210TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I210ToAR30Row = I210ToAR30Row_SME;
  }
#endif
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
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I210 to AR30.
LIBYUV_API
int I210ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  return I210ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_ar30, dst_stride_ar30,
                          &kYuvI601Constants, width, height);
}

// Convert H210 to AR30.
LIBYUV_API
int H210ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  return I210ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_ar30, dst_stride_ar30,
                          &kYuvH709Constants, width, height);
}

// Convert U210 to AR30.
LIBYUV_API
int U210ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  return I210ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_ar30, dst_stride_ar30,
                          &kYuv2020Constants, width, height);
}

// Convert I210 to AB30.
LIBYUV_API
int I210ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height) {
  return I210ToAR30Matrix(src_y, src_stride_y, src_v, src_stride_v, src_u,
                          src_stride_u, dst_ab30, dst_stride_ab30,
                          &kYvuI601Constants, width, height);
}

// Convert H210 to AB30.
LIBYUV_API
int H210ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height) {
  return I210ToAR30Matrix(src_y, src_stride_y, src_v, src_stride_v, src_u,
                          src_stride_u, dst_ab30, dst_stride_ab30,
                          &kYvuH709Constants, width, height);
}

// Convert U210 to AB30.
LIBYUV_API
int U210ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height) {
  return I210ToAR30Matrix(src_y, src_stride_y, src_v, src_stride_v, src_u,
                          src_stride_u, dst_ab30, dst_stride_ab30,
                          &kYuv2020Constants, width, height);
}

LIBYUV_API
int I410ToAR30Matrix(const uint16_t* src_y,
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
  void (*I410ToAR30Row)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I410ToAR30Row_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_I410TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I410ToAR30Row = I410ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I410ToAR30Row = I410ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_I410TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I410ToAR30Row = I410ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_I410TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I410ToAR30Row = I410ToAR30Row_SME;
  }
#endif
#if defined(HAS_I410TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I410ToAR30Row = I410ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I410ToAR30Row = I410ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I410TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I410ToAR30Row = I410ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I410ToAR30Row = I410ToAR30Row_AVX2;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    I410ToAR30Row(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert 10 bit YUV to ARGB with matrix.
LIBYUV_API
int I010ToARGBMatrix(const uint16_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_I210TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I210ToARGBRow = I210ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I210ToARGBRow = I210ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I210TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I210ToARGBRow = I210ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I210TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I210ToARGBRow = I210ToARGBRow_SME;
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

// Convert U010 to ARGB.
LIBYUV_API
int U010ToARGB(const uint16_t* src_y,
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
                          &kYuv2020Constants, width, height);
}

// Convert U010 to ABGR.
LIBYUV_API
int U010ToABGR(const uint16_t* src_y,
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
                          &kYvu2020Constants,  // Use Yvu matrix
                          width, height);
}

// Convert 12 bit YUV to ARGB with matrix.
LIBYUV_API
int I012ToARGBMatrix(const uint16_t* src_y,
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
  void (*I212ToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I212ToARGBRow_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I212TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I212ToARGBRow = I212ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I212ToARGBRow = I212ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I212TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I212ToARGBRow = I212ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I212ToARGBRow = I212ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I212TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I212ToARGBRow = I212ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I212ToARGBRow = I212ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I212TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I212ToARGBRow = I212ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I212TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I212ToARGBRow = I212ToARGBRow_SME;
  }
#endif
  for (y = 0; y < height; ++y) {
    I212ToARGBRow(src_y, src_u, src_v, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert 10 bit 422 YUV to ARGB with matrix.
LIBYUV_API
int I210ToARGBMatrix(const uint16_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_I210TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I210ToARGBRow = I210ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I210ToARGBRow = I210ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I210TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I210ToARGBRow = I210ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I210TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I210ToARGBRow = I210ToARGBRow_SME;
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
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I210 to ARGB.
LIBYUV_API
int I210ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I210ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvI601Constants, width, height);
}

// Convert I210 to ABGR.
LIBYUV_API
int I210ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I210ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuI601Constants,  // Use Yvu matrix
                          width, height);
}

// Convert H210 to ARGB.
LIBYUV_API
int H210ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I210ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuvH709Constants, width, height);
}

// Convert H210 to ABGR.
LIBYUV_API
int H210ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I210ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvuH709Constants,  // Use Yvu matrix
                          width, height);
}

// Convert U210 to ARGB.
LIBYUV_API
int U210ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I210ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_argb, dst_stride_argb,
                          &kYuv2020Constants, width, height);
}

// Convert U210 to ABGR.
LIBYUV_API
int U210ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  return I210ToARGBMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_abgr, dst_stride_abgr,
                          &kYvu2020Constants,  // Use Yvu matrix
                          width, height);
}

LIBYUV_API
int I410ToARGBMatrix(const uint16_t* src_y,
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
  void (*I410ToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I410ToARGBRow_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I410TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I410ToARGBRow = I410ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I410ToARGBRow = I410ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I410TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I410ToARGBRow = I410ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I410ToARGBRow = I410ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I410TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I410ToARGBRow = I410ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I410TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I410ToARGBRow = I410ToARGBRow_SME;
  }
#endif
#if defined(HAS_I410TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I410ToARGBRow = I410ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I410ToARGBRow = I410ToARGBRow_AVX2;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    I410ToARGBRow(src_y, src_u, src_v, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

LIBYUV_API
int P010ToARGBMatrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height) {
  int y;
  void (*P210ToARGBRow)(
      const uint16_t* y_buf, const uint16_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = P210ToARGBRow_C;
  assert(yuvconstants);
  if (!src_y || !src_uv || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_P210TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    P210ToARGBRow = P210ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      P210ToARGBRow = P210ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_P210TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    P210ToARGBRow = P210ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      P210ToARGBRow = P210ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_P210TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    P210ToARGBRow = P210ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      P210ToARGBRow = P210ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_P210TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    P210ToARGBRow = P210ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_P210TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    P210ToARGBRow = P210ToARGBRow_SME;
  }
#endif
  for (y = 0; y < height; ++y) {
    P210ToARGBRow(src_y, src_uv, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    if (y & 1) {
      src_uv += src_stride_uv;
    }
  }
  return 0;
}

LIBYUV_API
int P210ToARGBMatrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height) {
  int y;
  void (*P210ToARGBRow)(
      const uint16_t* y_buf, const uint16_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = P210ToARGBRow_C;
  assert(yuvconstants);
  if (!src_y || !src_uv || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_P210TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    P210ToARGBRow = P210ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      P210ToARGBRow = P210ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_P210TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    P210ToARGBRow = P210ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      P210ToARGBRow = P210ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_P210TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    P210ToARGBRow = P210ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      P210ToARGBRow = P210ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_P210TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    P210ToARGBRow = P210ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_P210TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    P210ToARGBRow = P210ToARGBRow_SME;
  }
#endif
  for (y = 0; y < height; ++y) {
    P210ToARGBRow(src_y, src_uv, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_uv += src_stride_uv;
  }
  return 0;
}

LIBYUV_API
int P010ToAR30Matrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height) {
  int y;
  void (*P210ToAR30Row)(
      const uint16_t* y_buf, const uint16_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = P210ToAR30Row_C;
  assert(yuvconstants);
  if (!src_y || !src_uv || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_P210TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    P210ToAR30Row = P210ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      P210ToAR30Row = P210ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_P210TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    P210ToAR30Row = P210ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      P210ToAR30Row = P210ToAR30Row_AVX2;
    }
  }
#endif
#if defined(HAS_P210TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    P210ToAR30Row = P210ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      P210ToAR30Row = P210ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_P210TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    P210ToAR30Row = P210ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_P210TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    P210ToAR30Row = P210ToAR30Row_SME;
  }
#endif
  for (y = 0; y < height; ++y) {
    P210ToAR30Row(src_y, src_uv, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    if (y & 1) {
      src_uv += src_stride_uv;
    }
  }
  return 0;
}

LIBYUV_API
int P210ToAR30Matrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height) {
  int y;
  void (*P210ToAR30Row)(
      const uint16_t* y_buf, const uint16_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = P210ToAR30Row_C;
  assert(yuvconstants);
  if (!src_y || !src_uv || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_P210TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    P210ToAR30Row = P210ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      P210ToAR30Row = P210ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_P210TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    P210ToAR30Row = P210ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      P210ToAR30Row = P210ToAR30Row_AVX2;
    }
  }
#endif
#if defined(HAS_P210TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    P210ToAR30Row = P210ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      P210ToAR30Row = P210ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_P210TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    P210ToAR30Row = P210ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_P210TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    P210ToAR30Row = P210ToAR30Row_SME;
  }
#endif
  for (y = 0; y < height; ++y) {
    P210ToAR30Row(src_y, src_uv, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    src_uv += src_stride_uv;
  }
  return 0;
}

// Convert I420 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I420AlphaToARGBMatrix(const uint8_t* src_y,
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
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
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
#if defined(HAS_I422ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I422ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_SME;
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
#if defined(HAS_I422ALPHATOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422AlphaToARGBRow = I422AlphaToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_I422ALPHATOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      I422AlphaToARGBRow = I422AlphaToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_I422ALPHATOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_RVV;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

// Convert I422 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I422AlphaToARGBMatrix(const uint8_t* src_y,
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
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
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
#if defined(HAS_I422ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I422ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_SME;
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
#if defined(HAS_I422ALPHATOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422AlphaToARGBRow = I422AlphaToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_I422ALPHATOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      I422AlphaToARGBRow = I422AlphaToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_I422ALPHATOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I422AlphaToARGBRow = I422AlphaToARGBRow_RVV;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

  for (y = 0; y < height; ++y) {
    I422AlphaToARGBRow(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_a += src_stride_a;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I444 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I444AlphaToARGBMatrix(const uint8_t* src_y,
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
  void (*I444AlphaToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                             const uint8_t* v_buf, const uint8_t* a_buf,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) = I444AlphaToARGBRow_C;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I444ALPHATOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_SME;
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_MSA;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_RVV;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

  for (y = 0; y < height; ++y) {
    I444AlphaToARGBRow(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_a += src_stride_a;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
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

// Convert I422 with Alpha to ARGB.
LIBYUV_API
int I422AlphaToARGB(const uint8_t* src_y,
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
  return I422AlphaToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                               src_stride_v, src_a, src_stride_a, dst_argb,
                               dst_stride_argb, &kYuvI601Constants, width,
                               height, attenuate);
}

// Convert I422 with Alpha to ABGR.
LIBYUV_API
int I422AlphaToABGR(const uint8_t* src_y,
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
  return I422AlphaToARGBMatrix(
      src_y, src_stride_y, src_v, src_stride_v,  // Swap U and V
      src_u, src_stride_u, src_a, src_stride_a, dst_abgr, dst_stride_abgr,
      &kYvuI601Constants,  // Use Yvu matrix
      width, height, attenuate);
}

// Convert I444 with Alpha to ARGB.
LIBYUV_API
int I444AlphaToARGB(const uint8_t* src_y,
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
  return I444AlphaToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                               src_stride_v, src_a, src_stride_a, dst_argb,
                               dst_stride_argb, &kYuvI601Constants, width,
                               height, attenuate);
}

// Convert I444 with Alpha to ABGR.
LIBYUV_API
int I444AlphaToABGR(const uint8_t* src_y,
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
  return I444AlphaToARGBMatrix(
      src_y, src_stride_y, src_v, src_stride_v,  // Swap U and V
      src_u, src_stride_u, src_a, src_stride_a, dst_abgr, dst_stride_abgr,
      &kYvuI601Constants,  // Use Yvu matrix
      width, height, attenuate);
}

// Convert I010 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I010AlphaToARGBMatrix(const uint16_t* src_y,
                          int src_stride_y,
                          const uint16_t* src_u,
                          int src_stride_u,
                          const uint16_t* src_v,
                          int src_stride_v,
                          const uint16_t* src_a,
                          int src_stride_a,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          const struct YuvConstants* yuvconstants,
                          int width,
                          int height,
                          int attenuate) {
  int y;
  void (*I210AlphaToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                             const uint16_t* v_buf, const uint16_t* a_buf,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) = I210AlphaToARGBRow_C;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I210ALPHATOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I210AlphaToARGBRow = I210AlphaToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I210ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I210ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_SME;
  }
#endif
#if defined(HAS_I210ALPHATOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I210AlphaToARGBRow = I210AlphaToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I210ALPHATOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I210AlphaToARGBRow = I210AlphaToARGBRow_AVX2;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

  for (y = 0; y < height; ++y) {
    I210AlphaToARGBRow(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
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

// Convert I210 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I210AlphaToARGBMatrix(const uint16_t* src_y,
                          int src_stride_y,
                          const uint16_t* src_u,
                          int src_stride_u,
                          const uint16_t* src_v,
                          int src_stride_v,
                          const uint16_t* src_a,
                          int src_stride_a,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          const struct YuvConstants* yuvconstants,
                          int width,
                          int height,
                          int attenuate) {
  int y;
  void (*I210AlphaToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                             const uint16_t* v_buf, const uint16_t* a_buf,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) = I210AlphaToARGBRow_C;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I210ALPHATOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I210AlphaToARGBRow = I210AlphaToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I210ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I210ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_SME;
  }
#endif
#if defined(HAS_I210ALPHATOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I210AlphaToARGBRow = I210AlphaToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I210ALPHATOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I210AlphaToARGBRow = I210AlphaToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I210AlphaToARGBRow = I210AlphaToARGBRow_AVX2;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

  for (y = 0; y < height; ++y) {
    I210AlphaToARGBRow(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_a += src_stride_a;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I410 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I410AlphaToARGBMatrix(const uint16_t* src_y,
                          int src_stride_y,
                          const uint16_t* src_u,
                          int src_stride_u,
                          const uint16_t* src_v,
                          int src_stride_v,
                          const uint16_t* src_a,
                          int src_stride_a,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          const struct YuvConstants* yuvconstants,
                          int width,
                          int height,
                          int attenuate) {
  int y;
  void (*I410AlphaToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                             const uint16_t* v_buf, const uint16_t* a_buf,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) = I410AlphaToARGBRow_C;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I410ALPHATOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I410AlphaToARGBRow = I410AlphaToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_SME;
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I410AlphaToARGBRow = I410AlphaToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I410AlphaToARGBRow = I410AlphaToARGBRow_AVX2;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

  for (y = 0; y < height; ++y) {
    I410AlphaToARGBRow(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_a += src_stride_a;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I400 to ARGB with matrix.
LIBYUV_API
int I400ToARGBMatrix(const uint8_t* src_y,
                     int src_stride_y,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height) {
  int y;
  void (*I400ToARGBRow)(const uint8_t* y_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I400ToARGBRow_C;
  assert(yuvconstants);
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
#if defined(HAS_I400TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I400ToARGBRow = I400ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I400TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I400ToARGBRow = I400ToARGBRow_SME;
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
#if defined(HAS_I400TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I400ToARGBRow = I400ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I400ToARGBRow = I400ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_I400TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I400ToARGBRow = I400ToARGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    I400ToARGBRow(src_y, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
  }
  return 0;
}

// Convert I400 to ARGB.
LIBYUV_API
int I400ToARGB(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return I400ToARGBMatrix(src_y, src_stride_y, dst_argb, dst_stride_argb,
                          &kYuvI601Constants, width, height);
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
#if defined(HAS_J400TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    J400ToARGBRow = J400ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      J400ToARGBRow = J400ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_J400TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    J400ToARGBRow = J400ToARGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    J400ToARGBRow(src_y, dst_argb, width);
    src_y += src_stride_y;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

#ifndef __riscv
// Shuffle table for converting BGRA to ARGB.
static const uvec8 kShuffleMaskBGRAToARGB = {
    3u, 2u, 1u, 0u, 7u, 6u, 5u, 4u, 11u, 10u, 9u, 8u, 15u, 14u, 13u, 12u};

// Shuffle table for converting ABGR to ARGB.
static const uvec8 kShuffleMaskABGRToARGB = {
    2u, 1u, 0u, 3u, 6u, 5u, 4u, 7u, 10u, 9u, 8u, 11u, 14u, 13u, 12u, 15u};

// Shuffle table for converting RGBA to ARGB.
static const uvec8 kShuffleMaskRGBAToARGB = {
    1u, 2u, 3u, 0u, 5u, 6u, 7u, 4u, 9u, 10u, 11u, 8u, 13u, 14u, 15u, 12u};

// Shuffle table for converting AR64 to AB64.
static const uvec8 kShuffleMaskAR64ToAB64 = {
    4u, 5u, 2u, 3u, 0u, 1u, 6u, 7u, 12u, 13u, 10u, 11u, 8u, 9u, 14u, 15u};

// Convert BGRA to ARGB.
LIBYUV_API
int BGRAToARGB(const uint8_t* src_bgra,
               int src_stride_bgra,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return ARGBShuffle(src_bgra, src_stride_bgra, dst_argb, dst_stride_argb,
                     (const uint8_t*)&kShuffleMaskBGRAToARGB, width, height);
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
                     (const uint8_t*)&kShuffleMaskBGRAToARGB, width, height);
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
                     (const uint8_t*)&kShuffleMaskABGRToARGB, width, height);
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
                     (const uint8_t*)&kShuffleMaskABGRToARGB, width, height);
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
                     (const uint8_t*)&kShuffleMaskRGBAToARGB, width, height);
}

// Convert AR64 To AB64.
LIBYUV_API
int AR64ToAB64(const uint16_t* src_ar64,
               int src_stride_ar64,
               uint16_t* dst_ab64,
               int dst_stride_ab64,
               int width,
               int height) {
  return AR64Shuffle(src_ar64, src_stride_ar64, dst_ab64, dst_stride_ab64,
                     (const uint8_t*)&kShuffleMaskAR64ToAB64, width, height);
}
#else
// Convert BGRA to ARGB (same as ARGBToBGRA).
LIBYUV_API
int BGRAToARGB(const uint8_t* src_bgra,
               int src_stride_bgra,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return ARGBToBGRA(src_bgra, src_stride_bgra, dst_argb, dst_stride_argb, width,
                    height);
}

// Convert ARGB to BGRA.
LIBYUV_API
int ARGBToBGRA(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_bgra,
               int dst_stride_bgra,
               int width,
               int height) {
  int y;
  void (*ARGBToBGRARow)(const uint8_t* src_argb, uint8_t* dst_bgra, int width) =
      ARGBToBGRARow_C;
  if (!src_argb || !dst_bgra || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_bgra == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_bgra = 0;
  }

#if defined(HAS_ARGBTOBGRAROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBToBGRARow = ARGBToBGRARow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToBGRARow(src_argb, dst_bgra, width);
    src_argb += src_stride_argb;
    dst_bgra += dst_stride_bgra;
  }
  return 0;
}

// Convert ARGB to ABGR.
LIBYUV_API
int ARGBToABGR(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height) {
  int y;
  void (*ARGBToABGRRow)(const uint8_t* src_argb, uint8_t* dst_abgr, int width) =
      ARGBToABGRRow_C;
  if (!src_argb || !dst_abgr || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_argb == width * 4 && dst_stride_abgr == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb = dst_stride_abgr = 0;
  }

#if defined(HAS_ARGBTOABGRROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBToABGRRow = ARGBToABGRRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    ARGBToABGRRow(src_argb, dst_abgr, width);
    src_argb += src_stride_argb;
    dst_abgr += dst_stride_abgr;
  }
  return 0;
}

// Convert ABGR to ARGB (same as ARGBToABGR).
LIBYUV_API
int ABGRToARGB(const uint8_t* src_abgr,
               int src_stride_abgr,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  return ARGBToABGR(src_abgr, src_stride_abgr, dst_argb, dst_stride_argb, width,
                    height);
}

// Convert RGBA to ARGB.
LIBYUV_API
int RGBAToARGB(const uint8_t* src_rgba,
               int src_stride_rgba,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*RGBAToARGBRow)(const uint8_t* src_rgba, uint8_t* dst_argb, int width) =
      RGBAToARGBRow_C;
  if (!src_rgba || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgba = src_rgba + (height - 1) * src_stride_rgba;
    src_stride_rgba = -src_stride_rgba;
  }
  // Coalesce rows.
  if (src_stride_rgba == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_rgba = dst_stride_argb = 0;
  }

#if defined(HAS_RGBATOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RGBAToARGBRow = RGBAToARGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    RGBAToARGBRow(src_rgba, dst_argb, width);
    src_rgba += src_stride_rgba;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert AR64 To AB64.
LIBYUV_API
int AR64ToAB64(const uint16_t* src_ar64,
               int src_stride_ar64,
               uint16_t* dst_ab64,
               int dst_stride_ab64,
               int width,
               int height) {
  int y;
  void (*AR64ToAB64Row)(const uint16_t* src_ar64, uint16_t* dst_ab64,
                        int width) = AR64ToAB64Row_C;
  if (!src_ar64 || !dst_ab64 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_ar64 = src_ar64 + (height - 1) * src_stride_ar64;
    src_stride_ar64 = -src_stride_ar64;
  }
  // Coalesce rows.
  if (src_stride_ar64 == width * 4 && dst_stride_ab64 == width * 4) {
    width *= height;
    height = 1;
    src_stride_ar64 = dst_stride_ab64 = 0;
  }

#if defined(HAS_AR64TOAB64ROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    AR64ToAB64Row = AR64ToAB64Row_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    AR64ToAB64Row(src_ar64, dst_ab64, width);
    src_ar64 += src_stride_ar64;
    dst_ab64 += dst_stride_ab64;
  }
  return 0;
}
#endif

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
#if defined(HAS_RGB24TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    RGB24ToARGBRow = RGB24ToARGBRow_SVE2;
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
#if defined(HAS_RGB24TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToARGBRow = RGB24ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_RGB24TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RGB24ToARGBRow = RGB24ToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_RGB24TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RGB24ToARGBRow = RGB24ToARGBRow_RVV;
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
#if defined(HAS_RAWTOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    RAWToARGBRow = RAWToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      RAWToARGBRow = RAWToARGBRow_AVX2;
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
#if defined(HAS_RAWTOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    RAWToARGBRow = RAWToARGBRow_SVE2;
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
#if defined(HAS_RAWTOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RAWToARGBRow = RAWToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RAWToARGBRow = RAWToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RAWToARGBRow = RAWToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_RAWTOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RAWToARGBRow = RAWToARGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    RAWToARGBRow(src_raw, dst_argb, width);
    src_raw += src_stride_raw;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert RAW to RGBA.
LIBYUV_API
int RAWToRGBA(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_rgba,
              int dst_stride_rgba,
              int width,
              int height) {
  int y;
  void (*RAWToRGBARow)(const uint8_t* src_rgb, uint8_t* dst_rgba, int width) =
      RAWToRGBARow_C;
  if (!src_raw || !dst_rgba || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }
  // Coalesce rows.
  if (src_stride_raw == width * 3 && dst_stride_rgba == width * 4) {
    width *= height;
    height = 1;
    src_stride_raw = dst_stride_rgba = 0;
  }
#if defined(HAS_RAWTORGBAROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    RAWToRGBARow = RAWToRGBARow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RAWToRGBARow = RAWToRGBARow_SSSE3;
    }
  }
#endif
#if defined(HAS_RAWTORGBAROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    RAWToRGBARow = RAWToRGBARow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RAWToRGBARow = RAWToRGBARow_NEON;
    }
  }
#endif
#if defined(HAS_RAWTORGBAROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    RAWToRGBARow = RAWToRGBARow_SVE2;
  }
#endif
#if defined(HAS_RAWTORGBAROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    RAWToRGBARow = RAWToRGBARow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    RAWToRGBARow(src_raw, dst_rgba, width);
    src_raw += src_stride_raw;
    dst_rgba += dst_stride_rgba;
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
    if (IS_ALIGNED(width, 16)) {
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
#if defined(HAS_RGB565TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      RGB565ToARGBRow = RGB565ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_RGB565TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      RGB565ToARGBRow = RGB565ToARGBRow_LASX;
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
    if (IS_ALIGNED(width, 16)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_ARGB1555TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_SVE2;
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
#if defined(HAS_ARGB1555TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGB1555TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_LASX;
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
#if defined(HAS_ARGB4444TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGB4444TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_LASX;
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

// Convert AR64 to ARGB.
LIBYUV_API
int AR64ToARGB(const uint16_t* src_ar64,
               int src_stride_ar64,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*AR64ToARGBRow)(const uint16_t* src_ar64, uint8_t* dst_argb,
                        int width) = AR64ToARGBRow_C;
  if (!src_ar64 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_ar64 = src_ar64 + (height - 1) * src_stride_ar64;
    src_stride_ar64 = -src_stride_ar64;
  }
  // Coalesce rows.
  if (src_stride_ar64 == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_ar64 = dst_stride_argb = 0;
  }
#if defined(HAS_AR64TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    AR64ToARGBRow = AR64ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 4)) {
      AR64ToARGBRow = AR64ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_AR64TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    AR64ToARGBRow = AR64ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      AR64ToARGBRow = AR64ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_AR64TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    AR64ToARGBRow = AR64ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      AR64ToARGBRow = AR64ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_AR64TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    AR64ToARGBRow = AR64ToARGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    AR64ToARGBRow(src_ar64, dst_argb, width);
    src_ar64 += src_stride_ar64;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert AB64 to ARGB.
LIBYUV_API
int AB64ToARGB(const uint16_t* src_ab64,
               int src_stride_ab64,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height) {
  int y;
  void (*AB64ToARGBRow)(const uint16_t* src_ar64, uint8_t* dst_argb,
                        int width) = AB64ToARGBRow_C;
  if (!src_ab64 || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_ab64 = src_ab64 + (height - 1) * src_stride_ab64;
    src_stride_ab64 = -src_stride_ab64;
  }
  // Coalesce rows.
  if (src_stride_ab64 == width * 4 && dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_ab64 = dst_stride_argb = 0;
  }
#if defined(HAS_AB64TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    AB64ToARGBRow = AB64ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 4)) {
      AB64ToARGBRow = AB64ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_AB64TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    AB64ToARGBRow = AB64ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 8)) {
      AB64ToARGBRow = AB64ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_AB64TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    AB64ToARGBRow = AB64ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      AB64ToARGBRow = AB64ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_AB64TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    AB64ToARGBRow = AB64ToARGBRow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    AB64ToARGBRow(src_ab64, dst_argb, width);
    src_ab64 += src_stride_ab64;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert NV12 to ARGB with matrix.
LIBYUV_API
int NV12ToARGBMatrix(const uint8_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_NV12TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    NV12ToARGBRow = NV12ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_NV12TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    NV12ToARGBRow = NV12ToARGBRow_SME;
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
#if defined(HAS_NV12TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_NV12TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    NV12ToARGBRow = NV12ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      NV12ToARGBRow = NV12ToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_NV12TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    NV12ToARGBRow = NV12ToARGBRow_RVV;
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

// Convert NV21 to ARGB with matrix.
LIBYUV_API
int NV21ToARGBMatrix(const uint8_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_NV21TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    NV21ToARGBRow = NV21ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_NV21TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    NV21ToARGBRow = NV21ToARGBRow_SME;
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
#if defined(HAS_NV21TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    NV21ToARGBRow = NV21ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      NV21ToARGBRow = NV21ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_NV21TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    NV21ToARGBRow = NV21ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      NV21ToARGBRow = NV21ToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_NV21TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    NV21ToARGBRow = NV21ToARGBRow_RVV;
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
// To output ABGR instead of ARGB swap the UV and use a mirrored yuv matrix.
// To swap the UV use NV12 instead of NV21.LIBYUV_API
LIBYUV_API
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
// Convert NV12 to RGB24 with matrix.
LIBYUV_API
int NV12ToRGB24Matrix(const uint8_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_NV12TORGB24ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    NV12ToRGB24Row = NV12ToRGB24Row_SVE2;
  }
#endif
#if defined(HAS_NV12TORGB24ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    NV12ToRGB24Row = NV12ToRGB24Row_SME;
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
#if defined(HAS_NV12TORGB24ROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    NV12ToRGB24Row = NV12ToRGB24Row_RVV;
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

// Convert NV21 to RGB24 with matrix.
LIBYUV_API
int NV21ToRGB24Matrix(const uint8_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_NV21TORGB24ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    NV21ToRGB24Row = NV21ToRGB24Row_SVE2;
  }
#endif
#if defined(HAS_NV21TORGB24ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    NV21ToRGB24Row = NV21ToRGB24Row_SME;
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
#if defined(HAS_NV21TORGB24ROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    NV21ToRGB24Row = NV21ToRGB24Row_RVV;
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

// Convert NV12 to RAW.
LIBYUV_API
int NV12ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_uv,
              int src_stride_uv,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  return NV21ToRGB24Matrix(src_y, src_stride_y, src_uv, src_stride_uv, dst_raw,
                           dst_stride_raw, &kYvuI601Constants, width, height);
}

// Convert NV21 to RAW.
LIBYUV_API
int NV21ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_vu,
              int src_stride_vu,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  return NV12ToRGB24Matrix(src_y, src_stride_y, src_vu, src_stride_vu, dst_raw,
                           dst_stride_raw, &kYvuI601Constants, width, height);
}

// Convert NV21 to YUV24
int NV21ToYUV24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_vu,
                int src_stride_vu,
                uint8_t* dst_yuv24,
                int dst_stride_yuv24,
                int width,
                int height) {
  int y;
  void (*NV21ToYUV24Row)(const uint8_t* src_y, const uint8_t* src_vu,
                         uint8_t* dst_yuv24, int width) = NV21ToYUV24Row_C;
  if (!src_y || !src_vu || !dst_yuv24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_yuv24 = dst_yuv24 + (height - 1) * dst_stride_yuv24;
    dst_stride_yuv24 = -dst_stride_yuv24;
  }
#if defined(HAS_NV21TOYUV24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    NV21ToYUV24Row = NV21ToYUV24Row_Any_NEON;
    if (IS_ALIGNED(width, 16)) {
      NV21ToYUV24Row = NV21ToYUV24Row_NEON;
    }
  }
#endif
#if defined(HAS_NV21TOYUV24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    NV21ToYUV24Row = NV21ToYUV24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      NV21ToYUV24Row = NV21ToYUV24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_NV21TOYUV24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    NV21ToYUV24Row = NV21ToYUV24Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      NV21ToYUV24Row = NV21ToYUV24Row_AVX2;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    NV21ToYUV24Row(src_y, src_vu, dst_yuv24, width);
    dst_yuv24 += dst_stride_yuv24;
    src_y += src_stride_y;
    if (y & 1) {
      src_vu += src_stride_vu;
    }
  }
  return 0;
}

// Convert YUY2 to ARGB with matrix.
LIBYUV_API
int YUY2ToARGBMatrix(const uint8_t* src_yuy2,
                     int src_stride_yuy2,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
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
#if defined(HAS_YUY2TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    YUY2ToARGBRow = YUY2ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_YUY2TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    YUY2ToARGBRow = YUY2ToARGBRow_SME;
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
#if defined(HAS_YUY2TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    YUY2ToARGBRow = YUY2ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      YUY2ToARGBRow = YUY2ToARGBRow_LSX;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    YUY2ToARGBRow(src_yuy2, dst_argb, yuvconstants, width);
    src_yuy2 += src_stride_yuy2;
    dst_argb += dst_stride_argb;
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
  return YUY2ToARGBMatrix(src_yuy2, src_stride_yuy2, dst_argb, dst_stride_argb,
                          &kYuvI601Constants, width, height);
}

// Convert UYVY to ARGB with matrix.
LIBYUV_API
int UYVYToARGBMatrix(const uint8_t* src_uyvy,
                     int src_stride_uyvy,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
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
#if defined(HAS_UYVYTOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    UYVYToARGBRow = UYVYToARGBRow_SVE2;
  }
#endif
#if defined(HAS_UYVYTOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    UYVYToARGBRow = UYVYToARGBRow_SME;
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
#if defined(HAS_UYVYTOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    UYVYToARGBRow = UYVYToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      UYVYToARGBRow = UYVYToARGBRow_LSX;
    }
  }
#endif
  for (y = 0; y < height; ++y) {
    UYVYToARGBRow(src_uyvy, dst_argb, yuvconstants, width);
    src_uyvy += src_stride_uyvy;
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
  return UYVYToARGBMatrix(src_uyvy, src_stride_uyvy, dst_argb, dst_stride_argb,
                          &kYuvI601Constants, width, height);
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

// Convert Android420 to ARGB with matrix.
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
  assert(yuvconstants);
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
  if (!plane_uv)
    return 1;
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

// Convert I422 to RGBA with matrix.
LIBYUV_API
int I422ToRGBAMatrix(const uint8_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_I422TORGBAROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToRGBARow = I422ToRGBARow_SVE2;
  }
#endif
#if defined(HAS_I422TORGBAROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToRGBARow = I422ToRGBARow_SME;
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
#if defined(HAS_I422TORGBAROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToRGBARow = I422ToRGBARow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGBARow = I422ToRGBARow_LSX;
    }
  }
#endif
#if defined(HAS_I422TORGBAROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToRGBARow = I422ToRGBARow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I422ToRGBARow = I422ToRGBARow_LASX;
    }
  }
#endif
#if defined(HAS_I422TORGBAROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I422ToRGBARow = I422ToRGBARow_RVV;
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

// Convert NV12 to RGB565 with matrix.
LIBYUV_API
int NV12ToRGB565Matrix(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_uv,
                       int src_stride_uv,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width,
                       int height) {
  int y;
  void (*NV12ToRGB565Row)(
      const uint8_t* y_buf, const uint8_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = NV12ToRGB565Row_C;
  assert(yuvconstants);
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
#if defined(HAS_NV12TORGB565ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    NV12ToRGB565Row = NV12ToRGB565Row_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      NV12ToRGB565Row = NV12ToRGB565Row_LSX;
    }
  }
#endif
#if defined(HAS_NV12TORGB565ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    NV12ToRGB565Row = NV12ToRGB565Row_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      NV12ToRGB565Row = NV12ToRGB565Row_LASX;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    NV12ToRGB565Row(src_y, src_uv, dst_rgb565, yuvconstants, width);
    dst_rgb565 += dst_stride_rgb565;
    src_y += src_stride_y;
    if (y & 1) {
      src_uv += src_stride_uv;
    }
  }
  return 0;
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
  return NV12ToRGB565Matrix(src_y, src_stride_y, src_uv, src_stride_uv,
                            dst_rgb565, dst_stride_rgb565, &kYuvI601Constants,
                            width, height);
}

// Convert I422 to RGBA with matrix.
LIBYUV_API
int I420ToRGBAMatrix(const uint8_t* src_y,
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
  assert(yuvconstants);
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
#if defined(HAS_I422TORGBAROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToRGBARow = I422ToRGBARow_SVE2;
  }
#endif
#if defined(HAS_I422TORGBAROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToRGBARow = I422ToRGBARow_SME;
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
#if defined(HAS_I422TORGBAROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToRGBARow = I422ToRGBARow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGBARow = I422ToRGBARow_LSX;
    }
  }
#endif
#if defined(HAS_I422TORGBAROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToRGBARow = I422ToRGBARow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I422ToRGBARow = I422ToRGBARow_LASX;
    }
  }
#endif
#if defined(HAS_I422TORGBAROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I422ToRGBARow = I422ToRGBARow_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToRGBARow(src_y, src_u, src_v, dst_rgba, yuvconstants, width);
    dst_rgba += dst_stride_rgba;
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
int I420ToRGBA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_rgba,
               int dst_stride_rgba,
               int width,
               int height) {
  return I420ToRGBAMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_rgba, dst_stride_rgba,
                          &kYuvI601Constants, width, height);
}

// Convert I420 to BGRA.
LIBYUV_API
int I420ToBGRA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_bgra,
               int dst_stride_bgra,
               int width,
               int height) {
  return I420ToRGBAMatrix(src_y, src_stride_y, src_v,
                          src_stride_v,  // Swap U and V
                          src_u, src_stride_u, dst_bgra, dst_stride_bgra,
                          &kYvuI601Constants,  // Use Yvu matrix
                          width, height);
}

// Convert I420 to RGB24 with matrix.
LIBYUV_API
int I420ToRGB24Matrix(const uint8_t* src_y,
                      int src_stride_y,
                      const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_rgb24,
                      int dst_stride_rgb24,
                      const struct YuvConstants* yuvconstants,
                      int width,
                      int height) {
  int y;
  void (*I422ToRGB24Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                         const uint8_t* v_buf, uint8_t* rgb_buf,
                         const struct YuvConstants* yuvconstants, int width) =
      I422ToRGB24Row_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb24 = dst_rgb24 + (height - 1) * dst_stride_rgb24;
    dst_stride_rgb24 = -dst_stride_rgb24;
  }
#if defined(HAS_I422TORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB24Row = I422ToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      I422ToRGB24Row = I422ToRGB24Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB24Row = I422ToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToRGB24Row = I422ToRGB24Row_SVE2;
  }
#endif
#if defined(HAS_I422TORGB24ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToRGB24Row = I422ToRGB24Row_SME;
  }
#endif
#if defined(HAS_I422TORGB24ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB24Row = I422ToRGB24Row_MSA;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB24Row = I422ToRGB24Row_LSX;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I422ToRGB24Row = I422ToRGB24Row_LASX;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I422ToRGB24Row = I422ToRGB24Row_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToRGB24Row(src_y, src_u, src_v, dst_rgb24, yuvconstants, width);
    dst_rgb24 += dst_stride_rgb24;
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
int I420ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_rgb24, dst_stride_rgb24,
                           &kYuvI601Constants, width, height);
}

// Convert I420 to RAW.
LIBYUV_API
int I420ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_v,
                           src_stride_v,  // Swap U and V
                           src_u, src_stride_u, dst_raw, dst_stride_raw,
                           &kYvuI601Constants,  // Use Yvu matrix
                           width, height);
}

// Convert J420 to RGB24.
LIBYUV_API
int J420ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_rgb24, dst_stride_rgb24,
                           &kYuvJPEGConstants, width, height);
}

// Convert J420 to RAW.
LIBYUV_API
int J420ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_v,
                           src_stride_v,  // Swap U and V
                           src_u, src_stride_u, dst_raw, dst_stride_raw,
                           &kYvuJPEGConstants,  // Use Yvu matrix
                           width, height);
}

// Convert H420 to RGB24.
LIBYUV_API
int H420ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_rgb24, dst_stride_rgb24,
                           &kYuvH709Constants, width, height);
}

// Convert H420 to RAW.
LIBYUV_API
int H420ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  return I420ToRGB24Matrix(src_y, src_stride_y, src_v,
                           src_stride_v,  // Swap U and V
                           src_u, src_stride_u, dst_raw, dst_stride_raw,
                           &kYvuH709Constants,  // Use Yvu matrix
                           width, height);
}

// Convert I422 to RGB24 with matrix.
LIBYUV_API
int I422ToRGB24Matrix(const uint8_t* src_y,
                      int src_stride_y,
                      const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_rgb24,
                      int dst_stride_rgb24,
                      const struct YuvConstants* yuvconstants,
                      int width,
                      int height) {
  int y;
  void (*I422ToRGB24Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                         const uint8_t* v_buf, uint8_t* rgb_buf,
                         const struct YuvConstants* yuvconstants, int width) =
      I422ToRGB24Row_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb24 = dst_rgb24 + (height - 1) * dst_stride_rgb24;
    dst_stride_rgb24 = -dst_stride_rgb24;
  }
#if defined(HAS_I422TORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB24Row = I422ToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      I422ToRGB24Row = I422ToRGB24Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB24Row = I422ToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToRGB24Row = I422ToRGB24Row_SVE2;
  }
#endif
#if defined(HAS_I422TORGB24ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToRGB24Row = I422ToRGB24Row_SME;
  }
#endif
#if defined(HAS_I422TORGB24ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_MSA;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB24Row = I422ToRGB24Row_MSA;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB24Row = I422ToRGB24Row_LSX;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToRGB24Row = I422ToRGB24Row_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I422ToRGB24Row = I422ToRGB24Row_LASX;
    }
  }
#endif
#if defined(HAS_I422TORGB24ROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I422ToRGB24Row = I422ToRGB24Row_RVV;
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToRGB24Row(src_y, src_u, src_v, dst_rgb24, yuvconstants, width);
    dst_rgb24 += dst_stride_rgb24;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I422 to RGB24.
LIBYUV_API
int I422ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height) {
  return I422ToRGB24Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                           src_stride_v, dst_rgb24, dst_stride_rgb24,
                           &kYuvI601Constants, width, height);
}

// Convert I422 to RAW.
LIBYUV_API
int I422ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height) {
  return I422ToRGB24Matrix(src_y, src_stride_y, src_v,
                           src_stride_v,  // Swap U and V
                           src_u, src_stride_u, dst_raw, dst_stride_raw,
                           &kYvuI601Constants,  // Use Yvu matrix
                           width, height);
}

// Convert I420 to ARGB1555.
LIBYUV_API
int I420ToARGB1555(const uint8_t* src_y,
                   int src_stride_y,
                   const uint8_t* src_u,
                   int src_stride_u,
                   const uint8_t* src_v,
                   int src_stride_v,
                   uint8_t* dst_argb1555,
                   int dst_stride_argb1555,
                   int width,
                   int height) {
  int y;
  void (*I422ToARGB1555Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                            const uint8_t* v_buf, uint8_t* rgb_buf,
                            const struct YuvConstants* yuvconstants,
                            int width) = I422ToARGB1555Row_C;
  if (!src_y || !src_u || !src_v || !dst_argb1555 || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb1555 = dst_argb1555 + (height - 1) * dst_stride_argb1555;
    dst_stride_argb1555 = -dst_stride_argb1555;
  }
#if defined(HAS_I422TOARGB1555ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB1555Row = I422ToARGB1555Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGB1555Row = I422ToARGB1555Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB1555Row = I422ToARGB1555Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToARGB1555Row = I422ToARGB1555Row_SVE2;
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToARGB1555Row = I422ToARGB1555Row_SME;
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB1555Row = I422ToARGB1555Row_MSA;
    }
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGB1555Row = I422ToARGB1555Row_LSX;
    }
  }
#endif
#if defined(HAS_I422TOARGB1555ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToARGB1555Row = I422ToARGB1555Row_Any_LASX;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB1555Row = I422ToARGB1555Row_LASX;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToARGB1555Row(src_y, src_u, src_v, dst_argb1555, &kYuvI601Constants,
                      width);
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
int I420ToARGB4444(const uint8_t* src_y,
                   int src_stride_y,
                   const uint8_t* src_u,
                   int src_stride_u,
                   const uint8_t* src_v,
                   int src_stride_v,
                   uint8_t* dst_argb4444,
                   int dst_stride_argb4444,
                   int width,
                   int height) {
  int y;
  void (*I422ToARGB4444Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                            const uint8_t* v_buf, uint8_t* rgb_buf,
                            const struct YuvConstants* yuvconstants,
                            int width) = I422ToARGB4444Row_C;
  if (!src_y || !src_u || !src_v || !dst_argb4444 || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb4444 = dst_argb4444 + (height - 1) * dst_stride_argb4444;
    dst_stride_argb4444 = -dst_stride_argb4444;
  }
#if defined(HAS_I422TOARGB4444ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB4444Row = I422ToARGB4444Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGB4444Row = I422ToARGB4444Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB4444Row = I422ToARGB4444Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToARGB4444Row = I422ToARGB4444Row_SVE2;
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToARGB4444Row = I422ToARGB4444Row_SME;
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB4444Row = I422ToARGB4444Row_MSA;
    }
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGB4444Row = I422ToARGB4444Row_LSX;
    }
  }
#endif
#if defined(HAS_I422TOARGB4444ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToARGB4444Row = I422ToARGB4444Row_Any_LASX;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGB4444Row = I422ToARGB4444Row_LASX;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToARGB4444Row(src_y, src_u, src_v, dst_argb4444, &kYuvI601Constants,
                      width);
    dst_argb4444 += dst_stride_argb4444;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to RGB565 with specified color matrix.
LIBYUV_API
int I420ToRGB565Matrix(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_u,
                       int src_stride_u,
                       const uint8_t* src_v,
                       int src_stride_v,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width,
                       int height) {
  int y;
  void (*I422ToRGB565Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                          const uint8_t* v_buf, uint8_t* rgb_buf,
                          const struct YuvConstants* yuvconstants, int width) =
      I422ToRGB565Row_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_rgb565 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb565 = dst_rgb565 + (height - 1) * dst_stride_rgb565;
    dst_stride_rgb565 = -dst_stride_rgb565;
  }
#if defined(HAS_I422TORGB565ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB565Row = I422ToRGB565Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToRGB565Row = I422ToRGB565Row_SVE2;
  }
#endif
#if defined(HAS_I422TORGB565ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToRGB565Row = I422ToRGB565Row_SME;
  }
#endif
#if defined(HAS_I422TORGB565ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_MSA;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB565Row = I422ToRGB565Row_LSX;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I422ToRGB565Row = I422ToRGB565Row_LASX;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToRGB565Row(src_y, src_u, src_v, dst_rgb565, yuvconstants, width);
    dst_rgb565 += dst_stride_rgb565;
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
int I420ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height) {
  return I420ToRGB565Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                            src_stride_v, dst_rgb565, dst_stride_rgb565,
                            &kYuvI601Constants, width, height);
}

// Convert J420 to RGB565.
LIBYUV_API
int J420ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height) {
  return I420ToRGB565Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                            src_stride_v, dst_rgb565, dst_stride_rgb565,
                            &kYuvJPEGConstants, width, height);
}

// Convert H420 to RGB565.
LIBYUV_API
int H420ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height) {
  return I420ToRGB565Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                            src_stride_v, dst_rgb565, dst_stride_rgb565,
                            &kYuvH709Constants, width, height);
}

// Convert I422 to RGB565 with specified color matrix.
LIBYUV_API
int I422ToRGB565Matrix(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_u,
                       int src_stride_u,
                       const uint8_t* src_v,
                       int src_stride_v,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width,
                       int height) {
  int y;
  void (*I422ToRGB565Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                          const uint8_t* v_buf, uint8_t* rgb_buf,
                          const struct YuvConstants* yuvconstants, int width) =
      I422ToRGB565Row_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_rgb565 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb565 = dst_rgb565 + (height - 1) * dst_stride_rgb565;
    dst_stride_rgb565 = -dst_stride_rgb565;
  }
#if defined(HAS_I422TORGB565ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB565Row = I422ToRGB565Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToRGB565Row = I422ToRGB565Row_SVE2;
  }
#endif
#if defined(HAS_I422TORGB565ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToRGB565Row = I422ToRGB565Row_SME;
  }
#endif
#if defined(HAS_I422TORGB565ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I422ToRGB565Row = I422ToRGB565Row_MSA;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToRGB565Row = I422ToRGB565Row_LSX;
    }
  }
#endif
#if defined(HAS_I422TORGB565ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToRGB565Row = I422ToRGB565Row_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I422ToRGB565Row = I422ToRGB565Row_LASX;
    }
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToRGB565Row(src_y, src_u, src_v, dst_rgb565, yuvconstants, width);
    dst_rgb565 += dst_stride_rgb565;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I422 to RGB565.
LIBYUV_API
int I422ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height) {
  return I422ToRGB565Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                            src_stride_v, dst_rgb565, dst_stride_rgb565,
                            &kYuvI601Constants, width, height);
}

// Ordered 8x8 dither for 888 to 565.  Values from 0 to 7.
static const uint8_t kDither565_4x4[16] = {
    0, 4, 1, 5, 6, 2, 7, 3, 1, 5, 0, 4, 7, 3, 6, 2,
};

// Convert I420 to RGB565 with dithering.
LIBYUV_API
int I420ToRGB565Dither(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_u,
                       int src_stride_u,
                       const uint8_t* src_v,
                       int src_stride_v,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const uint8_t* dither4x4,
                       int width,
                       int height) {
  int y;
  void (*I422ToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                        const uint8_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I422ToARGBRow_C;
  void (*ARGBToRGB565DitherRow)(const uint8_t* src_argb, uint8_t* dst_rgb,
                                uint32_t dither4, int width) =
      ARGBToRGB565DitherRow_C;
  if (!src_y || !src_u || !src_v || !dst_rgb565 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb565 = dst_rgb565 + (height - 1) * dst_stride_rgb565;
    dst_stride_rgb565 = -dst_stride_rgb565;
  }
  if (!dither4x4) {
    dither4x4 = kDither565_4x4;
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
#if defined(HAS_I422TOARGBROW_AVX512BW)
  if (TestCpuFlag(kCpuHasAVX512BW | kCpuHasAVX512VL) ==
      (kCpuHasAVX512BW | kCpuHasAVX512VL)) {
    I422ToARGBRow = I422ToARGBRow_Any_AVX512BW;
    if (IS_ALIGNED(width, 32)) {
      I422ToARGBRow = I422ToARGBRow_AVX512BW;
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
#if defined(HAS_I422TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToARGBRow = I422ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I422TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToARGBRow = I422ToARGBRow_SME;
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
#if defined(HAS_I422TOARGBROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    I422ToARGBRow = I422ToARGBRow_Any_LSX;
    if (IS_ALIGNED(width, 16)) {
      I422ToARGBRow = I422ToARGBRow_LSX;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I422ToARGBRow = I422ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I422ToARGBRow = I422ToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_I422TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I422ToARGBRow = I422ToARGBRow_RVV;
  }
#endif
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
#if defined(HAS_ARGBTORGB565DITHERROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_SVE2;
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
#if defined(HAS_ARGBTORGB565DITHERROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_Any_LSX;
    if (IS_ALIGNED(width, 8)) {
      ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_LSX;
    }
  }
#endif
#if defined(HAS_ARGBTORGB565DITHERROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      ARGBToRGB565DitherRow = ARGBToRGB565DitherRow_LASX;
    }
  }
#endif
  {
    // Allocate a row of argb.
    align_buffer_64(row_argb, width * 4);
    if (!row_argb)
      return 1;
    for (y = 0; y < height; ++y) {
      I422ToARGBRow(src_y, src_u, src_v, row_argb, &kYuvI601Constants, width);
      ARGBToRGB565DitherRow(row_argb, dst_rgb565,
                            *(const uint32_t*)(dither4x4 + ((y & 3) << 2)),
                            width);
      dst_rgb565 += dst_stride_rgb565;
      src_y += src_stride_y;
      if (y & 1) {
        src_u += src_stride_u;
        src_v += src_stride_v;
      }
    }
    free_aligned_buffer_64(row_argb);
  }
  return 0;
}

// Convert I420 to AR30 with matrix.
LIBYUV_API
int I420ToAR30Matrix(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height) {
  int y;
  void (*I422ToAR30Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                        const uint8_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I422ToAR30Row_C;

  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }

#if defined(HAS_I422TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I422ToAR30Row = I422ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToAR30Row = I422ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I422TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I422ToAR30Row = I422ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I422ToAR30Row = I422ToAR30Row_AVX2;
    }
  }
#endif
#if defined(HAS_I422TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I422ToAR30Row = I422ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I422ToAR30Row = I422ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_I422TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I422ToAR30Row = I422ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_I422TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I422ToAR30Row = I422ToAR30Row_SME;
  }
#endif

  for (y = 0; y < height; ++y) {
    I422ToAR30Row(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    if (y & 1) {
      src_u += src_stride_u;
      src_v += src_stride_v;
    }
  }
  return 0;
}

// Convert I420 to AR30.
LIBYUV_API
int I420ToAR30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  return I420ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_ar30, dst_stride_ar30,
                          &kYuvI601Constants, width, height);
}

// Convert H420 to AR30.
LIBYUV_API
int H420ToAR30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height) {
  return I420ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                          src_stride_v, dst_ar30, dst_stride_ar30,
                          &kYvuH709Constants, width, height);
}

// Convert I420 to AB30.
LIBYUV_API
int I420ToAB30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height) {
  return I420ToAR30Matrix(src_y, src_stride_y, src_v, src_stride_v, src_u,
                          src_stride_u, dst_ab30, dst_stride_ab30,
                          &kYvuI601Constants, width, height);
}

// Convert H420 to AB30.
LIBYUV_API
int H420ToAB30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height) {
  return I420ToAR30Matrix(src_y, src_stride_y, src_v, src_stride_v, src_u,
                          src_stride_u, dst_ab30, dst_stride_ab30,
                          &kYvuH709Constants, width, height);
}

static int I420ToARGBMatrixBilinear(const uint8_t* src_y,
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
  void (*Scale2RowUp_Bilinear)(const uint8_t* src_ptr, ptrdiff_t src_stride,
                               uint8_t* dst_ptr, ptrdiff_t dst_stride,
                               int dst_width) = ScaleRowUp2_Bilinear_Any_C;
  void (*ScaleRowUp2_Linear)(const uint8_t* src_ptr, uint8_t* dst_ptr,
                             int dst_width) = ScaleRowUp2_Linear_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
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
#if defined(HAS_I444TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I444ToARGBRow = I444ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I444TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I444ToARGBRow = I444ToARGBRow_SME;
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
#if defined(HAS_I444TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I444ToARGBRow = I444ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I444ToARGBRow = I444ToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_I444TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I444ToARGBRow = I444ToARGBRow_RVV;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_SSE2;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSE2;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_SSSE3;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSSE3;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_AVX2;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_AVX2;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_NEON;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_NEON;
  }
#endif
#if defined(HAS_SCALEROWUP2_BILINEAR_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_RVV;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_RVV;
  }
#endif

  // alloc 4 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 4);
  uint8_t* temp_u_1 = row;
  uint8_t* temp_u_2 = row + row_size;
  uint8_t* temp_v_1 = row + row_size * 2;
  uint8_t* temp_v_2 = row + row_size * 3;
  if (!row)
    return 1;

  ScaleRowUp2_Linear(src_u, temp_u_1, width);
  ScaleRowUp2_Linear(src_v, temp_v_1, width);
  I444ToARGBRow(src_y, temp_u_1, temp_v_1, dst_argb, yuvconstants, width);
  dst_argb += dst_stride_argb;
  src_y += src_stride_y;

  for (y = 0; y < height - 2; y += 2) {
    Scale2RowUp_Bilinear(src_u, src_stride_u, temp_u_1, row_size, width);
    Scale2RowUp_Bilinear(src_v, src_stride_v, temp_v_1, row_size, width);
    I444ToARGBRow(src_y, temp_u_1, temp_v_1, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    I444ToARGBRow(src_y, temp_u_2, temp_v_2, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }

  if (!(height & 1)) {
    ScaleRowUp2_Linear(src_u, temp_u_1, width);
    ScaleRowUp2_Linear(src_v, temp_v_1, width);
    I444ToARGBRow(src_y, temp_u_1, temp_v_1, dst_argb, yuvconstants, width);
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int I422ToARGBMatrixLinear(const uint8_t* src_y,
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
  void (*ScaleRowUp2_Linear)(const uint8_t* src_ptr, uint8_t* dst_ptr,
                             int dst_width) = ScaleRowUp2_Linear_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
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
#if defined(HAS_I444TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I444ToARGBRow = I444ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I444TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I444ToARGBRow = I444ToARGBRow_SME;
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
#if defined(HAS_I444TOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I444ToARGBRow = I444ToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I444ToARGBRow = I444ToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_I444TOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I444ToARGBRow = I444ToARGBRow_RVV;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSE2;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSSE3;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_AVX2;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_NEON;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_RVV;
  }
#endif

  // alloc 2 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 2);
  uint8_t* temp_u = row;
  uint8_t* temp_v = row + row_size;
  if (!row)
    return 1;

  for (y = 0; y < height; ++y) {
    ScaleRowUp2_Linear(src_u, temp_u, width);
    ScaleRowUp2_Linear(src_v, temp_v, width);
    I444ToARGBRow(src_y, temp_u, temp_v, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int I420ToRGB24MatrixBilinear(const uint8_t* src_y,
                                     int src_stride_y,
                                     const uint8_t* src_u,
                                     int src_stride_u,
                                     const uint8_t* src_v,
                                     int src_stride_v,
                                     uint8_t* dst_rgb24,
                                     int dst_stride_rgb24,
                                     const struct YuvConstants* yuvconstants,
                                     int width,
                                     int height) {
  int y;
  void (*I444ToRGB24Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                         const uint8_t* v_buf, uint8_t* rgb_buf,
                         const struct YuvConstants* yuvconstants, int width) =
      I444ToRGB24Row_C;
  void (*Scale2RowUp_Bilinear)(const uint8_t* src_ptr, ptrdiff_t src_stride,
                               uint8_t* dst_ptr, ptrdiff_t dst_stride,
                               int dst_width) = ScaleRowUp2_Bilinear_Any_C;
  void (*ScaleRowUp2_Linear)(const uint8_t* src_ptr, uint8_t* dst_ptr,
                             int dst_width) = ScaleRowUp2_Linear_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb24 = dst_rgb24 + (height - 1) * dst_stride_rgb24;
    dst_stride_rgb24 = -dst_stride_rgb24;
  }
#if defined(HAS_I444TORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      I444ToRGB24Row = I444ToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      I444ToRGB24Row = I444ToRGB24Row_AVX2;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I444ToRGB24Row = I444ToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I444ToRGB24Row = I444ToRGB24Row_SVE2;
  }
#endif
#if defined(HAS_I444TORGB24ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I444ToRGB24Row = I444ToRGB24Row_SME;
  }
#endif
#if defined(HAS_I444TORGB24ROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I444ToRGB24Row = I444ToRGB24Row_MSA;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_LASX;
    if (IS_ALIGNED(width, 32)) {
      I444ToRGB24Row = I444ToRGB24Row_LASX;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I444ToRGB24Row = I444ToRGB24Row_RVV;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_SSE2;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSE2;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_SSSE3;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSSE3;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_AVX2;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_AVX2;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_NEON;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_NEON;
  }
#endif
#if defined(HAS_SCALEROWUP2_BILINEAR_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_RVV;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_RVV;
  }
#endif

  // alloc 4 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 4);
  uint8_t* temp_u_1 = row;
  uint8_t* temp_u_2 = row + row_size;
  uint8_t* temp_v_1 = row + row_size * 2;
  uint8_t* temp_v_2 = row + row_size * 3;
  if (!row)
    return 1;

  ScaleRowUp2_Linear(src_u, temp_u_1, width);
  ScaleRowUp2_Linear(src_v, temp_v_1, width);
  I444ToRGB24Row(src_y, temp_u_1, temp_v_1, dst_rgb24, yuvconstants, width);
  dst_rgb24 += dst_stride_rgb24;
  src_y += src_stride_y;

  for (y = 0; y < height - 2; y += 2) {
    Scale2RowUp_Bilinear(src_u, src_stride_u, temp_u_1, row_size, width);
    Scale2RowUp_Bilinear(src_v, src_stride_v, temp_v_1, row_size, width);
    I444ToRGB24Row(src_y, temp_u_1, temp_v_1, dst_rgb24, yuvconstants, width);
    dst_rgb24 += dst_stride_rgb24;
    src_y += src_stride_y;
    I444ToRGB24Row(src_y, temp_u_2, temp_v_2, dst_rgb24, yuvconstants, width);
    dst_rgb24 += dst_stride_rgb24;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }

  if (!(height & 1)) {
    ScaleRowUp2_Linear(src_u, temp_u_1, width);
    ScaleRowUp2_Linear(src_v, temp_v_1, width);
    I444ToRGB24Row(src_y, temp_u_1, temp_v_1, dst_rgb24, yuvconstants, width);
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int I010ToAR30MatrixBilinear(const uint16_t* src_y,
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
  void (*I410ToAR30Row)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I410ToAR30Row_C;
  void (*Scale2RowUp_Bilinear_12)(
      const uint16_t* src_ptr, ptrdiff_t src_stride, uint16_t* dst_ptr,
      ptrdiff_t dst_stride, int dst_width) = ScaleRowUp2_Bilinear_16_Any_C;
  void (*ScaleRowUp2_Linear_12)(const uint16_t* src_ptr, uint16_t* dst_ptr,
                                int dst_width) = ScaleRowUp2_Linear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_I410TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I410ToAR30Row = I410ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I410ToAR30Row = I410ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_I410TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I410ToAR30Row = I410ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_I410TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I410ToAR30Row = I410ToAR30Row_SME;
  }
#endif
#if defined(HAS_I410TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I410ToAR30Row = I410ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I410ToAR30Row = I410ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I410TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I410ToAR30Row = I410ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I410ToAR30Row = I410ToAR30Row_AVX2;
    }
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_12_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    Scale2RowUp_Bilinear_12 = ScaleRowUp2_Bilinear_12_Any_SSSE3;
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_SSSE3;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_12_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Scale2RowUp_Bilinear_12 = ScaleRowUp2_Bilinear_12_Any_AVX2;
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_AVX2;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_12_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Scale2RowUp_Bilinear_12 = ScaleRowUp2_Bilinear_12_Any_NEON;
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_NEON;
  }
#endif

  // alloc 4 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 4 * sizeof(uint16_t));
  uint16_t* temp_u_1 = (uint16_t*)(row);
  uint16_t* temp_u_2 = (uint16_t*)(row) + row_size;
  uint16_t* temp_v_1 = (uint16_t*)(row) + row_size * 2;
  uint16_t* temp_v_2 = (uint16_t*)(row) + row_size * 3;
  if (!row)
    return 1;

  ScaleRowUp2_Linear_12(src_u, temp_u_1, width);
  ScaleRowUp2_Linear_12(src_v, temp_v_1, width);
  I410ToAR30Row(src_y, temp_u_1, temp_v_1, dst_ar30, yuvconstants, width);
  dst_ar30 += dst_stride_ar30;
  src_y += src_stride_y;

  for (y = 0; y < height - 2; y += 2) {
    Scale2RowUp_Bilinear_12(src_u, src_stride_u, temp_u_1, row_size, width);
    Scale2RowUp_Bilinear_12(src_v, src_stride_v, temp_v_1, row_size, width);
    I410ToAR30Row(src_y, temp_u_1, temp_v_1, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    I410ToAR30Row(src_y, temp_u_2, temp_v_2, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }

  if (!(height & 1)) {
    ScaleRowUp2_Linear_12(src_u, temp_u_1, width);
    ScaleRowUp2_Linear_12(src_v, temp_v_1, width);
    I410ToAR30Row(src_y, temp_u_1, temp_v_1, dst_ar30, yuvconstants, width);
  }

  free_aligned_buffer_64(row);

  return 0;
}

static int I210ToAR30MatrixLinear(const uint16_t* src_y,
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
  void (*I410ToAR30Row)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I410ToAR30Row_C;
  void (*ScaleRowUp2_Linear_12)(const uint16_t* src_ptr, uint16_t* dst_ptr,
                                int dst_width) = ScaleRowUp2_Linear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_I410TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I410ToAR30Row = I410ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I410ToAR30Row = I410ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_I410TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I410ToAR30Row = I410ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_I410TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I410ToAR30Row = I410ToAR30Row_SME;
  }
#endif
#if defined(HAS_I410TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I410ToAR30Row = I410ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I410ToAR30Row = I410ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I410TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I410ToAR30Row = I410ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I410ToAR30Row = I410ToAR30Row_AVX2;
    }
  }
#endif

#if defined(HAS_SCALEROWUP2_LINEAR_12_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_SSSE3;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_12_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_AVX2;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_12_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_NEON;
  }
#endif

  // alloc 2 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 2 * sizeof(uint16_t));
  uint16_t* temp_u = (uint16_t*)(row);
  uint16_t* temp_v = (uint16_t*)(row) + row_size;
  if (!row)
    return 1;

  for (y = 0; y < height; ++y) {
    ScaleRowUp2_Linear_12(src_u, temp_u, width);
    ScaleRowUp2_Linear_12(src_v, temp_v, width);
    I410ToAR30Row(src_y, temp_u, temp_v, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  free_aligned_buffer_64(row);
  return 0;
}

static int I010ToARGBMatrixBilinear(const uint16_t* src_y,
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
  void (*I410ToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I410ToARGBRow_C;
  void (*Scale2RowUp_Bilinear_12)(
      const uint16_t* src_ptr, ptrdiff_t src_stride, uint16_t* dst_ptr,
      ptrdiff_t dst_stride, int dst_width) = ScaleRowUp2_Bilinear_16_Any_C;
  void (*ScaleRowUp2_Linear_12)(const uint16_t* src_ptr, uint16_t* dst_ptr,
                                int dst_width) = ScaleRowUp2_Linear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I410TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I410ToARGBRow = I410ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I410ToARGBRow = I410ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I410TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I410ToARGBRow = I410ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I410ToARGBRow = I410ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I410TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I410ToARGBRow = I410ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I410TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I410ToARGBRow = I410ToARGBRow_SME;
  }
#endif
#if defined(HAS_I410TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I410ToARGBRow = I410ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I410ToARGBRow = I410ToARGBRow_AVX2;
    }
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_12_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    Scale2RowUp_Bilinear_12 = ScaleRowUp2_Bilinear_12_Any_SSSE3;
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_SSSE3;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_12_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Scale2RowUp_Bilinear_12 = ScaleRowUp2_Bilinear_12_Any_AVX2;
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_AVX2;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_12_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Scale2RowUp_Bilinear_12 = ScaleRowUp2_Bilinear_12_Any_NEON;
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_NEON;
  }
#endif

  // alloc 4 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 4 * sizeof(uint16_t));
  uint16_t* temp_u_1 = (uint16_t*)(row);
  uint16_t* temp_u_2 = (uint16_t*)(row) + row_size;
  uint16_t* temp_v_1 = (uint16_t*)(row) + row_size * 2;
  uint16_t* temp_v_2 = (uint16_t*)(row) + row_size * 3;
  if (!row)
    return 1;

  ScaleRowUp2_Linear_12(src_u, temp_u_1, width);
  ScaleRowUp2_Linear_12(src_v, temp_v_1, width);
  I410ToARGBRow(src_y, temp_u_1, temp_v_1, dst_argb, yuvconstants, width);
  dst_argb += dst_stride_argb;
  src_y += src_stride_y;

  for (y = 0; y < height - 2; y += 2) {
    Scale2RowUp_Bilinear_12(src_u, src_stride_u, temp_u_1, row_size, width);
    Scale2RowUp_Bilinear_12(src_v, src_stride_v, temp_v_1, row_size, width);
    I410ToARGBRow(src_y, temp_u_1, temp_v_1, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    I410ToARGBRow(src_y, temp_u_2, temp_v_2, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }

  if (!(height & 1)) {
    ScaleRowUp2_Linear_12(src_u, temp_u_1, width);
    ScaleRowUp2_Linear_12(src_v, temp_v_1, width);
    I410ToARGBRow(src_y, temp_u_1, temp_v_1, dst_argb, yuvconstants, width);
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int I210ToARGBMatrixLinear(const uint16_t* src_y,
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
  void (*I410ToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                        const uint16_t* v_buf, uint8_t* rgb_buf,
                        const struct YuvConstants* yuvconstants, int width) =
      I410ToARGBRow_C;
  void (*ScaleRowUp2_Linear_12)(const uint16_t* src_ptr, uint16_t* dst_ptr,
                                int dst_width) = ScaleRowUp2_Linear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I410TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I410ToARGBRow = I410ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I410ToARGBRow = I410ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I410TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I410ToARGBRow = I410ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I410ToARGBRow = I410ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I410TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I410ToARGBRow = I410ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I410TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I410ToARGBRow = I410ToARGBRow_SME;
  }
#endif
#if defined(HAS_I410TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I410ToARGBRow = I410ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I410ToARGBRow = I410ToARGBRow_AVX2;
    }
  }
#endif

#if defined(HAS_SCALEROWUP2_LINEAR_12_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_SSSE3;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_12_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_AVX2;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_12_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_NEON;
  }
#endif

  // alloc 2 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 2 * sizeof(uint16_t));
  uint16_t* temp_u = (uint16_t*)(row);
  uint16_t* temp_v = (uint16_t*)(row) + row_size;
  if (!row)
    return 1;

  for (y = 0; y < height; ++y) {
    ScaleRowUp2_Linear_12(src_u, temp_u, width);
    ScaleRowUp2_Linear_12(src_v, temp_v, width);
    I410ToARGBRow(src_y, temp_u, temp_v, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int I420AlphaToARGBMatrixBilinear(
    const uint8_t* src_y,
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
  void (*I444AlphaToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                             const uint8_t* v_buf, const uint8_t* a_buf,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) = I444AlphaToARGBRow_C;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  void (*Scale2RowUp_Bilinear)(const uint8_t* src_ptr, ptrdiff_t src_stride,
                               uint8_t* dst_ptr, ptrdiff_t dst_stride,
                               int dst_width) = ScaleRowUp2_Bilinear_Any_C;
  void (*ScaleRowUp2_Linear)(const uint8_t* src_ptr, uint8_t* dst_ptr,
                             int dst_width) = ScaleRowUp2_Linear_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I444ALPHATOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_SME;
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_MSA;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_RVV;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

#if defined(HAS_SCALEROWUP2_BILINEAR_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_SSE2;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSE2;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_SSSE3;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSSE3;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_AVX2;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_AVX2;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_Any_NEON;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_NEON;
  }
#endif
#if defined(HAS_SCALEROWUP2_BILINEAR_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    Scale2RowUp_Bilinear = ScaleRowUp2_Bilinear_RVV;
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_RVV;
  }
#endif

  // alloc 4 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 4);
  uint8_t* temp_u_1 = row;
  uint8_t* temp_u_2 = row + row_size;
  uint8_t* temp_v_1 = row + row_size * 2;
  uint8_t* temp_v_2 = row + row_size * 3;
  if (!row)
    return 1;

  ScaleRowUp2_Linear(src_u, temp_u_1, width);
  ScaleRowUp2_Linear(src_v, temp_v_1, width);
  I444AlphaToARGBRow(src_y, temp_u_1, temp_v_1, src_a, dst_argb, yuvconstants,
                     width);
  if (attenuate) {
    ARGBAttenuateRow(dst_argb, dst_argb, width);
  }
  dst_argb += dst_stride_argb;
  src_y += src_stride_y;
  src_a += src_stride_a;

  for (y = 0; y < height - 2; y += 2) {
    Scale2RowUp_Bilinear(src_u, src_stride_u, temp_u_1, row_size, width);
    Scale2RowUp_Bilinear(src_v, src_stride_v, temp_v_1, row_size, width);
    I444AlphaToARGBRow(src_y, temp_u_1, temp_v_1, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_a += src_stride_a;
    I444AlphaToARGBRow(src_y, temp_u_2, temp_v_2, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
    src_a += src_stride_a;
  }

  if (!(height & 1)) {
    ScaleRowUp2_Linear(src_u, temp_u_1, width);
    ScaleRowUp2_Linear(src_v, temp_v_1, width);
    I444AlphaToARGBRow(src_y, temp_u_1, temp_v_1, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int I422AlphaToARGBMatrixLinear(const uint8_t* src_y,
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
  void (*I444AlphaToARGBRow)(const uint8_t* y_buf, const uint8_t* u_buf,
                             const uint8_t* v_buf, const uint8_t* a_buf,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) = I444AlphaToARGBRow_C;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  void (*ScaleRowUp2_Linear)(const uint8_t* src_ptr, uint8_t* dst_ptr,
                             int dst_width) = ScaleRowUp2_Linear_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I444ALPHATOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_SME;
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_MSA;
    if (IS_ALIGNED(width, 8)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_MSA;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_LASX)
  if (TestCpuFlag(kCpuHasLASX)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_Any_LASX;
    if (IS_ALIGNED(width, 16)) {
      I444AlphaToARGBRow = I444AlphaToARGBRow_LASX;
    }
  }
#endif
#if defined(HAS_I444ALPHATOARGBROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I444AlphaToARGBRow = I444AlphaToARGBRow_RVV;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

#if defined(HAS_SCALEROWUP2_LINEAR_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSE2;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSSE3;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_AVX2;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_NEON;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_RVV;
  }
#endif

  // alloc 2 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 2);
  uint8_t* temp_u = row;
  uint8_t* temp_v = row + row_size;
  if (!row)
    return 1;

  for (y = 0; y < height; ++y) {
    ScaleRowUp2_Linear(src_u, temp_u, width);
    ScaleRowUp2_Linear(src_v, temp_v, width);
    I444AlphaToARGBRow(src_y, temp_u, temp_v, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_a += src_stride_a;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int I010AlphaToARGBMatrixBilinear(
    const uint16_t* src_y,
    int src_stride_y,
    const uint16_t* src_u,
    int src_stride_u,
    const uint16_t* src_v,
    int src_stride_v,
    const uint16_t* src_a,
    int src_stride_a,
    uint8_t* dst_argb,
    int dst_stride_argb,
    const struct YuvConstants* yuvconstants,
    int width,
    int height,
    int attenuate) {
  int y;
  void (*I410AlphaToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                             const uint16_t* v_buf, const uint16_t* a_buf,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) = I410AlphaToARGBRow_C;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  void (*Scale2RowUp_Bilinear_12)(
      const uint16_t* src_ptr, ptrdiff_t src_stride, uint16_t* dst_ptr,
      ptrdiff_t dst_stride, int dst_width) = ScaleRowUp2_Bilinear_16_Any_C;
  void (*ScaleRowUp2_Linear_12)(const uint16_t* src_ptr, uint16_t* dst_ptr,
                                int dst_width) = ScaleRowUp2_Linear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I410ALPHATOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I410AlphaToARGBRow = I410AlphaToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_SME;
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I410AlphaToARGBRow = I410AlphaToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I410AlphaToARGBRow = I410AlphaToARGBRow_AVX2;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

#if defined(HAS_SCALEROWUP2_BILINEAR_12_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    Scale2RowUp_Bilinear_12 = ScaleRowUp2_Bilinear_12_Any_SSSE3;
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_SSSE3;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_12_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    Scale2RowUp_Bilinear_12 = ScaleRowUp2_Bilinear_12_Any_AVX2;
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_AVX2;
  }
#endif

#if defined(HAS_SCALEROWUP2_BILINEAR_12_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    Scale2RowUp_Bilinear_12 = ScaleRowUp2_Bilinear_12_Any_NEON;
    ScaleRowUp2_Linear_12 = ScaleRowUp2_Linear_12_Any_NEON;
  }
#endif

  // alloc 4 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 4 * sizeof(uint16_t));
  uint16_t* temp_u_1 = (uint16_t*)(row);
  uint16_t* temp_u_2 = (uint16_t*)(row) + row_size;
  uint16_t* temp_v_1 = (uint16_t*)(row) + row_size * 2;
  uint16_t* temp_v_2 = (uint16_t*)(row) + row_size * 3;
  if (!row)
    return 1;

  ScaleRowUp2_Linear_12(src_u, temp_u_1, width);
  ScaleRowUp2_Linear_12(src_v, temp_v_1, width);
  I410AlphaToARGBRow(src_y, temp_u_1, temp_v_1, src_a, dst_argb, yuvconstants,
                     width);
  if (attenuate) {
    ARGBAttenuateRow(dst_argb, dst_argb, width);
  }
  dst_argb += dst_stride_argb;
  src_y += src_stride_y;
  src_a += src_stride_a;

  for (y = 0; y < height - 2; y += 2) {
    Scale2RowUp_Bilinear_12(src_u, src_stride_u, temp_u_1, row_size, width);
    Scale2RowUp_Bilinear_12(src_v, src_stride_v, temp_v_1, row_size, width);
    I410AlphaToARGBRow(src_y, temp_u_1, temp_v_1, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_a += src_stride_a;
    I410AlphaToARGBRow(src_y, temp_u_2, temp_v_2, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_a += src_stride_a;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }

  if (!(height & 1)) {
    ScaleRowUp2_Linear_12(src_u, temp_u_1, width);
    ScaleRowUp2_Linear_12(src_v, temp_v_1, width);
    I410AlphaToARGBRow(src_y, temp_u_1, temp_v_1, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int I210AlphaToARGBMatrixLinear(const uint16_t* src_y,
                                       int src_stride_y,
                                       const uint16_t* src_u,
                                       int src_stride_u,
                                       const uint16_t* src_v,
                                       int src_stride_v,
                                       const uint16_t* src_a,
                                       int src_stride_a,
                                       uint8_t* dst_argb,
                                       int dst_stride_argb,
                                       const struct YuvConstants* yuvconstants,
                                       int width,
                                       int height,
                                       int attenuate) {
  int y;
  void (*I410AlphaToARGBRow)(const uint16_t* y_buf, const uint16_t* u_buf,
                             const uint16_t* v_buf, const uint16_t* a_buf,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) = I410AlphaToARGBRow_C;
  void (*ARGBAttenuateRow)(const uint8_t* src_argb, uint8_t* dst_argb,
                           int width) = ARGBAttenuateRow_C;
  void (*ScaleRowUp2_Linear)(const uint16_t* src_ptr, uint16_t* dst_ptr,
                             int dst_width) = ScaleRowUp2_Linear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !src_a || !dst_argb || width <= 0 ||
      height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_I410ALPHATOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I410AlphaToARGBRow = I410AlphaToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_SVE2;
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_SME;
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I410AlphaToARGBRow = I410AlphaToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_I410ALPHATOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I410AlphaToARGBRow = I410AlphaToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      I410AlphaToARGBRow = I410AlphaToARGBRow_AVX2;
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
#if defined(HAS_ARGBATTENUATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ARGBAttenuateRow = ARGBAttenuateRow_RVV;
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

#if defined(HAS_SCALEROWUP2_LINEAR_12_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_12_Any_SSSE3;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_12_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_12_Any_AVX2;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_12_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_12_Any_NEON;
  }
#endif

  // alloc 2 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 2 * sizeof(uint16_t));
  uint16_t* temp_u = (uint16_t*)(row);
  uint16_t* temp_v = (uint16_t*)(row) + row_size;
  if (!row)
    return 1;

  for (y = 0; y < height; ++y) {
    ScaleRowUp2_Linear(src_u, temp_u, width);
    ScaleRowUp2_Linear(src_v, temp_v, width);
    I410AlphaToARGBRow(src_y, temp_u, temp_v, src_a, dst_argb, yuvconstants,
                       width);
    if (attenuate) {
      ARGBAttenuateRow(dst_argb, dst_argb, width);
    }
    dst_argb += dst_stride_argb;
    src_a += src_stride_a;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  free_aligned_buffer_64(row);
  return 0;
}

static int P010ToARGBMatrixBilinear(const uint16_t* src_y,
                                    int src_stride_y,
                                    const uint16_t* src_uv,
                                    int src_stride_uv,
                                    uint8_t* dst_argb,
                                    int dst_stride_argb,
                                    const struct YuvConstants* yuvconstants,
                                    int width,
                                    int height) {
  int y;
  void (*P410ToARGBRow)(
      const uint16_t* y_buf, const uint16_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = P410ToARGBRow_C;
  void (*Scale2RowUp_Bilinear_16)(
      const uint16_t* src_ptr, ptrdiff_t src_stride, uint16_t* dst_ptr,
      ptrdiff_t dst_stride, int dst_width) = ScaleUVRowUp2_Bilinear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_uv || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_P410TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    P410ToARGBRow = P410ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      P410ToARGBRow = P410ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_P410TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    P410ToARGBRow = P410ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      P410ToARGBRow = P410ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_P410TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    P410ToARGBRow = P410ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      P410ToARGBRow = P410ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_P410TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    P410ToARGBRow = P410ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_P410TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    P410ToARGBRow = P410ToARGBRow_SME;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_SSE41
  if (TestCpuFlag(kCpuHasSSE41)) {
    Scale2RowUp_Bilinear_16 = ScaleUVRowUp2_Bilinear_16_Any_SSE41;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_AVX2
  if (TestCpuFlag(kCpuHasAVX2)) {
    Scale2RowUp_Bilinear_16 = ScaleUVRowUp2_Bilinear_16_Any_AVX2;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_NEON
  if (TestCpuFlag(kCpuHasNEON)) {
    Scale2RowUp_Bilinear_16 = ScaleUVRowUp2_Bilinear_16_Any_NEON;
  }
#endif

  // alloc 2 lines temp
  const int row_size = (2 * width + 31) & ~31;
  align_buffer_64(row, row_size * 2 * sizeof(uint16_t));
  uint16_t* temp_uv_1 = (uint16_t*)(row);
  uint16_t* temp_uv_2 = (uint16_t*)(row) + row_size;
  if (!row)
    return 1;

  Scale2RowUp_Bilinear_16(src_uv, 0, temp_uv_1, row_size, width);
  P410ToARGBRow(src_y, temp_uv_1, dst_argb, yuvconstants, width);
  dst_argb += dst_stride_argb;
  src_y += src_stride_y;

  for (y = 0; y < height - 2; y += 2) {
    Scale2RowUp_Bilinear_16(src_uv, src_stride_uv, temp_uv_1, row_size, width);
    P410ToARGBRow(src_y, temp_uv_1, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    P410ToARGBRow(src_y, temp_uv_2, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_uv += src_stride_uv;
  }

  if (!(height & 1)) {
    Scale2RowUp_Bilinear_16(src_uv, 0, temp_uv_1, row_size, width);
    P410ToARGBRow(src_y, temp_uv_1, dst_argb, yuvconstants, width);
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int P210ToARGBMatrixLinear(const uint16_t* src_y,
                                  int src_stride_y,
                                  const uint16_t* src_uv,
                                  int src_stride_uv,
                                  uint8_t* dst_argb,
                                  int dst_stride_argb,
                                  const struct YuvConstants* yuvconstants,
                                  int width,
                                  int height) {
  int y;
  void (*P410ToARGBRow)(
      const uint16_t* y_buf, const uint16_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = P410ToARGBRow_C;
  void (*ScaleRowUp2_Linear)(const uint16_t* src_uv, uint16_t* dst_uv,
                             int dst_width) = ScaleUVRowUp2_Linear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_uv || !dst_argb || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
#if defined(HAS_P410TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    P410ToARGBRow = P410ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      P410ToARGBRow = P410ToARGBRow_SSSE3;
    }
  }
#endif
#if defined(HAS_P410TOARGBROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    P410ToARGBRow = P410ToARGBRow_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      P410ToARGBRow = P410ToARGBRow_AVX2;
    }
  }
#endif
#if defined(HAS_P410TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    P410ToARGBRow = P410ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      P410ToARGBRow = P410ToARGBRow_NEON;
    }
  }
#endif
#if defined(HAS_P410TOARGBROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    P410ToARGBRow = P410ToARGBRow_SVE2;
  }
#endif
#if defined(HAS_P410TOARGBROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    P410ToARGBRow = P410ToARGBRow_SME;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_SSE41
  if (TestCpuFlag(kCpuHasSSE41)) {
    ScaleRowUp2_Linear = ScaleUVRowUp2_Linear_16_Any_SSE41;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_AVX2
  if (TestCpuFlag(kCpuHasAVX2)) {
    ScaleRowUp2_Linear = ScaleUVRowUp2_Linear_16_Any_AVX2;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_NEON
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleRowUp2_Linear = ScaleUVRowUp2_Linear_16_Any_NEON;
  }
#endif

  const int row_size = (2 * width + 31) & ~31;
  align_buffer_64(row, row_size * sizeof(uint16_t));
  uint16_t* temp_uv = (uint16_t*)(row);
  if (!row)
    return 1;

  for (y = 0; y < height; ++y) {
    ScaleRowUp2_Linear(src_uv, temp_uv, width);
    P410ToARGBRow(src_y, temp_uv, dst_argb, yuvconstants, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_uv += src_stride_uv;
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int P010ToAR30MatrixBilinear(const uint16_t* src_y,
                                    int src_stride_y,
                                    const uint16_t* src_uv,
                                    int src_stride_uv,
                                    uint8_t* dst_ar30,
                                    int dst_stride_ar30,
                                    const struct YuvConstants* yuvconstants,
                                    int width,
                                    int height) {
  int y;
  void (*P410ToAR30Row)(
      const uint16_t* y_buf, const uint16_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = P410ToAR30Row_C;
  void (*Scale2RowUp_Bilinear_16)(
      const uint16_t* src_ptr, ptrdiff_t src_stride, uint16_t* dst_ptr,
      ptrdiff_t dst_stride, int dst_width) = ScaleUVRowUp2_Bilinear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_uv || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_P410TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    P410ToAR30Row = P410ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      P410ToAR30Row = P410ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_P410TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    P410ToAR30Row = P410ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      P410ToAR30Row = P410ToAR30Row_AVX2;
    }
  }
#endif
#if defined(HAS_P410TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    P410ToAR30Row = P410ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      P410ToAR30Row = P410ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_P410TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    P410ToAR30Row = P410ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_P410TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    P410ToAR30Row = P410ToAR30Row_SME;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_SSE41
  if (TestCpuFlag(kCpuHasSSE41)) {
    Scale2RowUp_Bilinear_16 = ScaleUVRowUp2_Bilinear_16_Any_SSE41;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_AVX2
  if (TestCpuFlag(kCpuHasAVX2)) {
    Scale2RowUp_Bilinear_16 = ScaleUVRowUp2_Bilinear_16_Any_AVX2;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_NEON
  if (TestCpuFlag(kCpuHasNEON)) {
    Scale2RowUp_Bilinear_16 = ScaleUVRowUp2_Bilinear_16_Any_NEON;
  }
#endif

  // alloc 2 lines temp
  const int row_size = (2 * width + 31) & ~31;
  align_buffer_64(row, row_size * 2 * sizeof(uint16_t));
  uint16_t* temp_uv_1 = (uint16_t*)(row);
  uint16_t* temp_uv_2 = (uint16_t*)(row) + row_size;
  if (!row)
    return 1;

  Scale2RowUp_Bilinear_16(src_uv, 0, temp_uv_1, row_size, width);
  P410ToAR30Row(src_y, temp_uv_1, dst_ar30, yuvconstants, width);
  dst_ar30 += dst_stride_ar30;
  src_y += src_stride_y;

  for (y = 0; y < height - 2; y += 2) {
    Scale2RowUp_Bilinear_16(src_uv, src_stride_uv, temp_uv_1, row_size, width);
    P410ToAR30Row(src_y, temp_uv_1, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    P410ToAR30Row(src_y, temp_uv_2, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    src_uv += src_stride_uv;
  }

  if (!(height & 1)) {
    Scale2RowUp_Bilinear_16(src_uv, 0, temp_uv_1, row_size, width);
    P410ToAR30Row(src_y, temp_uv_1, dst_ar30, yuvconstants, width);
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int P210ToAR30MatrixLinear(const uint16_t* src_y,
                                  int src_stride_y,
                                  const uint16_t* src_uv,
                                  int src_stride_uv,
                                  uint8_t* dst_ar30,
                                  int dst_stride_ar30,
                                  const struct YuvConstants* yuvconstants,
                                  int width,
                                  int height) {
  int y;
  void (*P410ToAR30Row)(
      const uint16_t* y_buf, const uint16_t* uv_buf, uint8_t* rgb_buf,
      const struct YuvConstants* yuvconstants, int width) = P410ToAR30Row_C;
  void (*ScaleRowUp2_Linear)(const uint16_t* src_uv, uint16_t* dst_uv,
                             int dst_width) = ScaleUVRowUp2_Linear_16_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_uv || !dst_ar30 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_ar30 = dst_ar30 + (height - 1) * dst_stride_ar30;
    dst_stride_ar30 = -dst_stride_ar30;
  }
#if defined(HAS_P410TOAR30ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    P410ToAR30Row = P410ToAR30Row_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      P410ToAR30Row = P410ToAR30Row_SSSE3;
    }
  }
#endif
#if defined(HAS_P410TOAR30ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    P410ToAR30Row = P410ToAR30Row_Any_AVX2;
    if (IS_ALIGNED(width, 16)) {
      P410ToAR30Row = P410ToAR30Row_AVX2;
    }
  }
#endif
#if defined(HAS_P410TOAR30ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    P410ToAR30Row = P410ToAR30Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      P410ToAR30Row = P410ToAR30Row_NEON;
    }
  }
#endif
#if defined(HAS_P410TOAR30ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    P410ToAR30Row = P410ToAR30Row_SVE2;
  }
#endif
#if defined(HAS_P410TOAR30ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    P410ToAR30Row = P410ToAR30Row_SME;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_SSE41
  if (TestCpuFlag(kCpuHasSSE41)) {
    ScaleRowUp2_Linear = ScaleUVRowUp2_Linear_16_Any_SSE41;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_AVX2
  if (TestCpuFlag(kCpuHasAVX2)) {
    ScaleRowUp2_Linear = ScaleUVRowUp2_Linear_16_Any_AVX2;
  }
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_NEON
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleRowUp2_Linear = ScaleUVRowUp2_Linear_16_Any_NEON;
  }
#endif

  const int row_size = (2 * width + 31) & ~31;
  align_buffer_64(row, row_size * sizeof(uint16_t));
  uint16_t* temp_uv = (uint16_t*)(row);
  if (!row)
    return 1;

  for (y = 0; y < height; ++y) {
    ScaleRowUp2_Linear(src_uv, temp_uv, width);
    P410ToAR30Row(src_y, temp_uv, dst_ar30, yuvconstants, width);
    dst_ar30 += dst_stride_ar30;
    src_y += src_stride_y;
    src_uv += src_stride_uv;
  }

  free_aligned_buffer_64(row);
  return 0;
}

static int I422ToRGB24MatrixLinear(const uint8_t* src_y,
                                   int src_stride_y,
                                   const uint8_t* src_u,
                                   int src_stride_u,
                                   const uint8_t* src_v,
                                   int src_stride_v,
                                   uint8_t* dst_rgb24,
                                   int dst_stride_rgb24,
                                   const struct YuvConstants* yuvconstants,
                                   int width,
                                   int height) {
  int y;
  void (*I444ToRGB24Row)(const uint8_t* y_buf, const uint8_t* u_buf,
                         const uint8_t* v_buf, uint8_t* rgb_buf,
                         const struct YuvConstants* yuvconstants, int width) =
      I444ToRGB24Row_C;
  void (*ScaleRowUp2_Linear)(const uint8_t* src_ptr, uint8_t* dst_ptr,
                             int dst_width) = ScaleRowUp2_Linear_Any_C;
  assert(yuvconstants);
  if (!src_y || !src_u || !src_v || !dst_rgb24 || width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_rgb24 = dst_rgb24 + (height - 1) * dst_stride_rgb24;
    dst_stride_rgb24 = -dst_stride_rgb24;
  }
#if defined(HAS_I444TORGB24ROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      I444ToRGB24Row = I444ToRGB24Row_SSSE3;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_AVX2;
    if (IS_ALIGNED(width, 32)) {
      I444ToRGB24Row = I444ToRGB24Row_AVX2;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    I444ToRGB24Row = I444ToRGB24Row_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I444ToRGB24Row = I444ToRGB24Row_NEON;
    }
  }
#endif
#if defined(HAS_I444TORGB24ROW_SVE2)
  if (TestCpuFlag(kCpuHasSVE2)) {
    I444ToRGB24Row = I444ToRGB24Row_SVE2;
  }
#endif
#if defined(HAS_I444TORGB24ROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    I444ToRGB24Row = I444ToRGB24Row_SME;
  }
#endif
#if defined(HAS_I444TORGB24ROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    I444ToRGB24Row = I444ToRGB24Row_RVV;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSE2;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_SSSE3;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_AVX2;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_Any_NEON;
  }
#endif
#if defined(HAS_SCALEROWUP2_LINEAR_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    ScaleRowUp2_Linear = ScaleRowUp2_Linear_RVV;
  }
#endif

  // alloc 2 lines temp
  const int row_size = (width + 31) & ~31;
  align_buffer_64(row, row_size * 2);
  uint8_t* temp_u = row;
  uint8_t* temp_v = row + row_size;
  if (!row)
    return 1;

  for (y = 0; y < height; ++y) {
    ScaleRowUp2_Linear(src_u, temp_u, width);
    ScaleRowUp2_Linear(src_v, temp_v, width);
    I444ToRGB24Row(src_y, temp_u, temp_v, dst_rgb24, yuvconstants, width);
    dst_rgb24 += dst_stride_rgb24;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }

  free_aligned_buffer_64(row);
  return 0;
}

LIBYUV_API
int I422ToRGB24MatrixFilter(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_u,
                            int src_stride_u,
                            const uint8_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_rgb24,
                            int dst_stride_rgb24,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height,
                            enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I422ToRGB24Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                               src_stride_v, dst_rgb24, dst_stride_rgb24,
                               yuvconstants, width, height);
    case kFilterBilinear:
    case kFilterBox:
    case kFilterLinear:
      return I422ToRGB24MatrixLinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
          dst_rgb24, dst_stride_rgb24, yuvconstants, width, height);
  }

  return -1;
}

LIBYUV_API
int I420ToARGBMatrixFilter(const uint8_t* src_y,
                           int src_stride_y,
                           const uint8_t* src_u,
                           int src_stride_u,
                           const uint8_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I420ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                              src_stride_v, dst_argb, dst_stride_argb,
                              yuvconstants, width, height);
    case kFilterBilinear:
    case kFilterBox:
      return I420ToARGBMatrixBilinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
          dst_argb, dst_stride_argb, yuvconstants, width, height);
    case kFilterLinear:
      // Actually we can do this, but probably there's no usage.
      return -1;
  }

  return -1;
}

LIBYUV_API
int I422ToARGBMatrixFilter(const uint8_t* src_y,
                           int src_stride_y,
                           const uint8_t* src_u,
                           int src_stride_u,
                           const uint8_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I422ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                              src_stride_v, dst_argb, dst_stride_argb,
                              yuvconstants, width, height);
    case kFilterBilinear:
    case kFilterBox:
    case kFilterLinear:
      return I422ToARGBMatrixLinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
          dst_argb, dst_stride_argb, yuvconstants, width, height);
  }

  return -1;
}

LIBYUV_API
int I420ToRGB24MatrixFilter(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_u,
                            int src_stride_u,
                            const uint8_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_rgb24,
                            int dst_stride_rgb24,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height,
                            enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I420ToRGB24Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                               src_stride_v, dst_rgb24, dst_stride_rgb24,
                               yuvconstants, width, height);
    case kFilterLinear:  // TODO(fb): Implement Linear using Bilinear stride 0
    case kFilterBilinear:
    case kFilterBox:
      return I420ToRGB24MatrixBilinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
          dst_rgb24, dst_stride_rgb24, yuvconstants, width, height);
  }

  return -1;
}

LIBYUV_API
int I010ToAR30MatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_u,
                           int src_stride_u,
                           const uint16_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_ar30,
                           int dst_stride_ar30,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I010ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                              src_stride_v, dst_ar30, dst_stride_ar30,
                              yuvconstants, width, height);
    case kFilterLinear:  // TODO(fb): Implement Linear using Bilinear stride 0
    case kFilterBilinear:
    case kFilterBox:
      return I010ToAR30MatrixBilinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
          dst_ar30, dst_stride_ar30, yuvconstants, width, height);
  }

  return -1;
}

LIBYUV_API
int I210ToAR30MatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_u,
                           int src_stride_u,
                           const uint16_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_ar30,
                           int dst_stride_ar30,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I210ToAR30Matrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                              src_stride_v, dst_ar30, dst_stride_ar30,
                              yuvconstants, width, height);
    case kFilterBilinear:
    case kFilterBox:
    case kFilterLinear:
      return I210ToAR30MatrixLinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
          dst_ar30, dst_stride_ar30, yuvconstants, width, height);
  }

  return -1;
}

LIBYUV_API
int I010ToARGBMatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_u,
                           int src_stride_u,
                           const uint16_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I010ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                              src_stride_v, dst_argb, dst_stride_argb,
                              yuvconstants, width, height);
    case kFilterLinear:  // TODO(fb): Implement Linear using Bilinear stride 0
    case kFilterBilinear:
    case kFilterBox:
      return I010ToARGBMatrixBilinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
          dst_argb, dst_stride_argb, yuvconstants, width, height);
  }

  return -1;
}

LIBYUV_API
int I210ToARGBMatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_u,
                           int src_stride_u,
                           const uint16_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I210ToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u, src_v,
                              src_stride_v, dst_argb, dst_stride_argb,
                              yuvconstants, width, height);
    case kFilterBilinear:
    case kFilterBox:
    case kFilterLinear:
      return I210ToARGBMatrixLinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v,
          dst_argb, dst_stride_argb, yuvconstants, width, height);
  }

  return -1;
}

LIBYUV_API
int I420AlphaToARGBMatrixFilter(const uint8_t* src_y,
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
                                int attenuate,
                                enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I420AlphaToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u,
                                   src_v, src_stride_v, src_a, src_stride_a,
                                   dst_argb, dst_stride_argb, yuvconstants,
                                   width, height, attenuate);
    case kFilterLinear:  // TODO(fb): Implement Linear using Bilinear stride 0
    case kFilterBilinear:
    case kFilterBox:
      return I420AlphaToARGBMatrixBilinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v, src_a,
          src_stride_a, dst_argb, dst_stride_argb, yuvconstants, width, height,
          attenuate);
  }

  return -1;
}

LIBYUV_API
int I422AlphaToARGBMatrixFilter(const uint8_t* src_y,
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
                                int attenuate,
                                enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I422AlphaToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u,
                                   src_v, src_stride_v, src_a, src_stride_a,
                                   dst_argb, dst_stride_argb, yuvconstants,
                                   width, height, attenuate);
    case kFilterBilinear:
    case kFilterBox:
    case kFilterLinear:
      return I422AlphaToARGBMatrixLinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v, src_a,
          src_stride_a, dst_argb, dst_stride_argb, yuvconstants, width, height,
          attenuate);
  }

  return -1;
}

LIBYUV_API
int I010AlphaToARGBMatrixFilter(const uint16_t* src_y,
                                int src_stride_y,
                                const uint16_t* src_u,
                                int src_stride_u,
                                const uint16_t* src_v,
                                int src_stride_v,
                                const uint16_t* src_a,
                                int src_stride_a,
                                uint8_t* dst_argb,
                                int dst_stride_argb,
                                const struct YuvConstants* yuvconstants,
                                int width,
                                int height,
                                int attenuate,
                                enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I010AlphaToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u,
                                   src_v, src_stride_v, src_a, src_stride_a,
                                   dst_argb, dst_stride_argb, yuvconstants,
                                   width, height, attenuate);
    case kFilterLinear:  // TODO(fb): Implement Linear using Bilinear stride 0
    case kFilterBilinear:
    case kFilterBox:
      return I010AlphaToARGBMatrixBilinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v, src_a,
          src_stride_a, dst_argb, dst_stride_argb, yuvconstants, width, height,
          attenuate);
  }

  return -1;
}

LIBYUV_API
int I210AlphaToARGBMatrixFilter(const uint16_t* src_y,
                                int src_stride_y,
                                const uint16_t* src_u,
                                int src_stride_u,
                                const uint16_t* src_v,
                                int src_stride_v,
                                const uint16_t* src_a,
                                int src_stride_a,
                                uint8_t* dst_argb,
                                int dst_stride_argb,
                                const struct YuvConstants* yuvconstants,
                                int width,
                                int height,
                                int attenuate,
                                enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return I210AlphaToARGBMatrix(src_y, src_stride_y, src_u, src_stride_u,
                                   src_v, src_stride_v, src_a, src_stride_a,
                                   dst_argb, dst_stride_argb, yuvconstants,
                                   width, height, attenuate);
    case kFilterBilinear:
    case kFilterBox:
    case kFilterLinear:
      return I210AlphaToARGBMatrixLinear(
          src_y, src_stride_y, src_u, src_stride_u, src_v, src_stride_v, src_a,
          src_stride_a, dst_argb, dst_stride_argb, yuvconstants, width, height,
          attenuate);
  }

  return -1;
}

// TODO(fb): Verify this function works correctly.  P010 is like NV12 but 10 bit
// UV is biplanar.
LIBYUV_API
int P010ToARGBMatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_uv,
                           int src_stride_uv,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return P010ToARGBMatrix(src_y, src_stride_y, src_uv, src_stride_uv,
                              dst_argb, dst_stride_argb, yuvconstants, width,
                              height);
    case kFilterLinear:  // TODO(fb): Implement Linear using Bilinear stride 0
    case kFilterBilinear:
    case kFilterBox:
      return P010ToARGBMatrixBilinear(src_y, src_stride_y, src_uv,
                                      src_stride_uv, dst_argb, dst_stride_argb,
                                      yuvconstants, width, height);
  }

  return -1;
}

LIBYUV_API
int P210ToARGBMatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_uv,
                           int src_stride_uv,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return P210ToARGBMatrix(src_y, src_stride_y, src_uv, src_stride_uv,
                              dst_argb, dst_stride_argb, yuvconstants, width,
                              height);
    case kFilterBilinear:
    case kFilterBox:
    case kFilterLinear:
      return P210ToARGBMatrixLinear(src_y, src_stride_y, src_uv, src_stride_uv,
                                    dst_argb, dst_stride_argb, yuvconstants,
                                    width, height);
  }

  return -1;
}

LIBYUV_API
int P010ToAR30MatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_uv,
                           int src_stride_uv,
                           uint8_t* dst_ar30,
                           int dst_stride_ar30,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return P010ToAR30Matrix(src_y, src_stride_y, src_uv, src_stride_uv,
                              dst_ar30, dst_stride_ar30, yuvconstants, width,
                              height);
    case kFilterLinear:  // TODO(fb): Implement Linear using Bilinear stride 0
    case kFilterBilinear:
    case kFilterBox:
      return P010ToAR30MatrixBilinear(src_y, src_stride_y, src_uv,
                                      src_stride_uv, dst_ar30, dst_stride_ar30,
                                      yuvconstants, width, height);
  }

  return -1;
}

LIBYUV_API
int P210ToAR30MatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_uv,
                           int src_stride_uv,
                           uint8_t* dst_ar30,
                           int dst_stride_ar30,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter) {
  switch (filter) {
    case kFilterNone:
      return P210ToAR30Matrix(src_y, src_stride_y, src_uv, src_stride_uv,
                              dst_ar30, dst_stride_ar30, yuvconstants, width,
                              height);
    case kFilterBilinear:
    case kFilterBox:
    case kFilterLinear:
      return P210ToAR30MatrixLinear(src_y, src_stride_y, src_uv, src_stride_uv,
                                    dst_ar30, dst_stride_ar30, yuvconstants,
                                    width, height);
  }

  return -1;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
