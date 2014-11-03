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
#include "libyuv/format_conversion.h"
#ifdef HAVE_JPEG
#include "libyuv/mjpeg_decoder.h"
#endif
#include "libyuv/rotate_argb.h"
#include "libyuv/row.h"
#include "libyuv/video_common.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Copy ARGB with optional flipping
LIBYUV_API
int ARGBCopy(const uint8* src_argb, int src_stride_argb,
             uint8* dst_argb, int dst_stride_argb,
             int width, int height) {
  if (!src_argb || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }

  CopyPlane(src_argb, src_stride_argb, dst_argb, dst_stride_argb,
            width * 4, height);
  return 0;
}

// Convert I444 to ARGB.
LIBYUV_API
int I444ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_y || !src_u || !src_v ||
      !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_y == width &&
      src_stride_u == width &&
      src_stride_v == width &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_argb = 0;
  }
  void (*I444ToARGBRow)(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* rgb_buf,
                        int width) = I444ToARGBRow_C;
#if defined(HAS_I444TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I444ToARGBRow = I444ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I444ToARGBRow = I444ToARGBRow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
        I444ToARGBRow = I444ToARGBRow_SSSE3;
      }
    }
  }
#elif defined(HAS_I444TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I444ToARGBRow = I444ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I444ToARGBRow = I444ToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I444ToARGBRow(src_y, src_u, src_v, dst_argb, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I422 to ARGB.
LIBYUV_API
int I422ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_y || !src_u || !src_v ||
      !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_y == width &&
      src_stride_u * 2 == width &&
      src_stride_v * 2 == width &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_argb = 0;
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
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I411 to ARGB.
LIBYUV_API
int I411ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_y || !src_u || !src_v ||
      !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_y == width &&
      src_stride_u * 4 == width &&
      src_stride_v * 4 == width &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_y = src_stride_u = src_stride_v = dst_stride_argb = 0;
  }
  void (*I411ToARGBRow)(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* rgb_buf,
                        int width) = I411ToARGBRow_C;
#if defined(HAS_I411TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I411ToARGBRow = I411ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I411ToARGBRow = I411ToARGBRow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
        I411ToARGBRow = I411ToARGBRow_SSSE3;
      }
    }
  }
#elif defined(HAS_I411TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I411ToARGBRow = I411ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I411ToARGBRow = I411ToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    I411ToARGBRow(src_y, src_u, src_v, dst_argb, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    src_u += src_stride_u;
    src_v += src_stride_v;
  }
  return 0;
}

// Convert I400 to ARGB.
LIBYUV_API
int I400ToARGB_Reference(const uint8* src_y, int src_stride_y,
                         uint8* dst_argb, int dst_stride_argb,
                         int width, int height) {
  if (!src_y || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  // Coalesce rows.
  if (src_stride_y == width &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_argb = 0;
  }
  void (*YToARGBRow)(const uint8* y_buf,
                     uint8* rgb_buf,
                     int width) = YToARGBRow_C;
#if defined(HAS_YTOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && width >= 8 &&
      IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
    YToARGBRow = YToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      YToARGBRow = YToARGBRow_SSE2;
    }
  }
#elif defined(HAS_YTOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    YToARGBRow = YToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      YToARGBRow = YToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    YToARGBRow(src_y, dst_argb, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
  }
  return 0;
}

// Convert I400 to ARGB.
LIBYUV_API
int I400ToARGB(const uint8* src_y, int src_stride_y,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_y || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_y = src_y + (height - 1) * src_stride_y;
    src_stride_y = -src_stride_y;
  }
  // Coalesce rows.
  if (src_stride_y == width &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_y = dst_stride_argb = 0;
  }
  void (*I400ToARGBRow)(const uint8* src_y, uint8* dst_argb, int pix) =
      I400ToARGBRow_C;
#if defined(HAS_I400TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && width >= 8) {
    I400ToARGBRow = I400ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      I400ToARGBRow = I400ToARGBRow_Unaligned_SSE2;
      if (IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
        I400ToARGBRow = I400ToARGBRow_SSE2;
      }
    }
  }
#elif defined(HAS_I400TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    I400ToARGBRow = I400ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      I400ToARGBRow = I400ToARGBRow_NEON;
    }
  }
#endif
  for (int y = 0; y < height; ++y) {
    I400ToARGBRow(src_y, dst_argb, width);
    src_y += src_stride_y;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Shuffle table for converting BGRA to ARGB.
static uvec8 kShuffleMaskBGRAToARGB = {
  3u, 2u, 1u, 0u, 7u, 6u, 5u, 4u, 11u, 10u, 9u, 8u, 15u, 14u, 13u, 12u
};

// Shuffle table for converting ABGR to ARGB.
static uvec8 kShuffleMaskABGRToARGB = {
  2u, 1u, 0u, 3u, 6u, 5u, 4u, 7u, 10u, 9u, 8u, 11u, 14u, 13u, 12u, 15u
};

// Shuffle table for converting RGBA to ARGB.
static uvec8 kShuffleMaskRGBAToARGB = {
  1u, 2u, 3u, 0u, 5u, 6u, 7u, 4u, 9u, 10u, 11u, 8u, 13u, 14u, 15u, 12u
};

// Convert BGRA to ARGB.
LIBYUV_API
int BGRAToARGB(const uint8* src_bgra, int src_stride_bgra,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  return ARGBShuffle(src_bgra, src_stride_bgra,
                     dst_argb, dst_stride_argb,
                     (const uint8*)(&kShuffleMaskBGRAToARGB),
                     width, height);
}

// Convert ABGR to ARGB.
LIBYUV_API
int ABGRToARGB(const uint8* src_abgr, int src_stride_abgr,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  return ARGBShuffle(src_abgr, src_stride_abgr,
                     dst_argb, dst_stride_argb,
                     (const uint8*)(&kShuffleMaskABGRToARGB),
                     width, height);
}

// Convert RGBA to ARGB.
LIBYUV_API
int RGBAToARGB(const uint8* src_rgba, int src_stride_rgba,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  return ARGBShuffle(src_rgba, src_stride_rgba,
                     dst_argb, dst_stride_argb,
                     (const uint8*)(&kShuffleMaskRGBAToARGB),
                     width, height);
}

// Convert RGB24 to ARGB.
LIBYUV_API
int RGB24ToARGB(const uint8* src_rgb24, int src_stride_rgb24,
                uint8* dst_argb, int dst_stride_argb,
                int width, int height) {
  if (!src_rgb24 || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb24 = src_rgb24 + (height - 1) * src_stride_rgb24;
    src_stride_rgb24 = -src_stride_rgb24;
  }
  // Coalesce rows.
  if (src_stride_rgb24 == width * 3 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_rgb24 = dst_stride_argb = 0;
  }
  void (*RGB24ToARGBRow)(const uint8* src_rgb, uint8* dst_argb, int pix) =
      RGB24ToARGBRow_C;
#if defined(HAS_RGB24TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 16 &&
      IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RGB24ToARGBRow = RGB24ToARGBRow_SSSE3;
    }
  }
#elif defined(HAS_RGB24TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    RGB24ToARGBRow = RGB24ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RGB24ToARGBRow = RGB24ToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    RGB24ToARGBRow(src_rgb24, dst_argb, width);
    src_rgb24 += src_stride_rgb24;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert RAW to ARGB.
LIBYUV_API
int RAWToARGB(const uint8* src_raw, int src_stride_raw,
              uint8* dst_argb, int dst_stride_argb,
              int width, int height) {
  if (!src_raw || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_raw = src_raw + (height - 1) * src_stride_raw;
    src_stride_raw = -src_stride_raw;
  }
  // Coalesce rows.
  if (src_stride_raw == width * 3 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_raw = dst_stride_argb = 0;
  }
  void (*RAWToARGBRow)(const uint8* src_rgb, uint8* dst_argb, int pix) =
      RAWToARGBRow_C;
#if defined(HAS_RAWTOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 16 &&
      IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
    RAWToARGBRow = RAWToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      RAWToARGBRow = RAWToARGBRow_SSSE3;
    }
  }
#elif defined(HAS_RAWTOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    RAWToARGBRow = RAWToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RAWToARGBRow = RAWToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    RAWToARGBRow(src_raw, dst_argb, width);
    src_raw += src_stride_raw;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert RGB565 to ARGB.
LIBYUV_API
int RGB565ToARGB(const uint8* src_rgb565, int src_stride_rgb565,
                 uint8* dst_argb, int dst_stride_argb,
                 int width, int height) {
  if (!src_rgb565 || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_rgb565 = src_rgb565 + (height - 1) * src_stride_rgb565;
    src_stride_rgb565 = -src_stride_rgb565;
  }
  // Coalesce rows.
  if (src_stride_rgb565 == width * 2 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_rgb565 = dst_stride_argb = 0;
  }
  void (*RGB565ToARGBRow)(const uint8* src_rgb565, uint8* dst_argb, int pix) =
      RGB565ToARGBRow_C;
#if defined(HAS_RGB565TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && width >= 8 &&
      IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      RGB565ToARGBRow = RGB565ToARGBRow_SSE2;
    }
  }
#elif defined(HAS_RGB565TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    RGB565ToARGBRow = RGB565ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      RGB565ToARGBRow = RGB565ToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    RGB565ToARGBRow(src_rgb565, dst_argb, width);
    src_rgb565 += src_stride_rgb565;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert ARGB1555 to ARGB.
LIBYUV_API
int ARGB1555ToARGB(const uint8* src_argb1555, int src_stride_argb1555,
                   uint8* dst_argb, int dst_stride_argb,
                   int width, int height) {
  if (!src_argb1555 || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb1555 = src_argb1555 + (height - 1) * src_stride_argb1555;
    src_stride_argb1555 = -src_stride_argb1555;
  }
  // Coalesce rows.
  if (src_stride_argb1555 == width * 2 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb1555 = dst_stride_argb = 0;
  }
  void (*ARGB1555ToARGBRow)(const uint8* src_argb1555, uint8* dst_argb,
                            int pix) = ARGB1555ToARGBRow_C;
#if defined(HAS_ARGB1555TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && width >= 8 &&
      IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_SSE2;
    }
  }
#elif defined(HAS_ARGB1555TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    ARGB1555ToARGBRow = ARGB1555ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGB1555ToARGBRow = ARGB1555ToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    ARGB1555ToARGBRow(src_argb1555, dst_argb, width);
    src_argb1555 += src_stride_argb1555;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert ARGB4444 to ARGB.
LIBYUV_API
int ARGB4444ToARGB(const uint8* src_argb4444, int src_stride_argb4444,
                   uint8* dst_argb, int dst_stride_argb,
                   int width, int height) {
  if (!src_argb4444 || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_argb4444 = src_argb4444 + (height - 1) * src_stride_argb4444;
    src_stride_argb4444 = -src_stride_argb4444;
  }
  // Coalesce rows.
  if (src_stride_argb4444 == width * 2 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_argb4444 = dst_stride_argb = 0;
  }
  void (*ARGB4444ToARGBRow)(const uint8* src_argb4444, uint8* dst_argb,
                            int pix) = ARGB4444ToARGBRow_C;
#if defined(HAS_ARGB4444TOARGBROW_SSE2)
  if (TestCpuFlag(kCpuHasSSE2) && width >= 8 &&
      IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_SSE2;
    if (IS_ALIGNED(width, 8)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_SSE2;
    }
  }
#elif defined(HAS_ARGB4444TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    ARGB4444ToARGBRow = ARGB4444ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGB4444ToARGBRow = ARGB4444ToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    ARGB4444ToARGBRow(src_argb4444, dst_argb, width);
    src_argb4444 += src_stride_argb4444;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert NV12 to ARGB.
LIBYUV_API
int NV12ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_uv, int src_stride_uv,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_y || !src_uv || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  void (*NV12ToARGBRow)(const uint8* y_buf,
                        const uint8* uv_buf,
                        uint8* rgb_buf,
                        int width) = NV12ToARGBRow_C;
#if defined(HAS_NV12TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    NV12ToARGBRow = NV12ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
        NV12ToARGBRow = NV12ToARGBRow_SSSE3;
      }
    }
  }
#elif defined(HAS_NV12TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    NV12ToARGBRow = NV12ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    NV12ToARGBRow(src_y, src_uv, dst_argb, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    if (y & 1) {
      src_uv += src_stride_uv;
    }
  }
  return 0;
}

// Convert NV21 to ARGB.
LIBYUV_API
int NV21ToARGB(const uint8* src_y, int src_stride_y,
               const uint8* src_uv, int src_stride_uv,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_y || !src_uv || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  void (*NV21ToARGBRow)(const uint8* y_buf,
                        const uint8* uv_buf,
                        uint8* rgb_buf,
                        int width) = NV21ToARGBRow_C;
#if defined(HAS_NV21TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    NV21ToARGBRow = NV21ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      NV21ToARGBRow = NV21ToARGBRow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
        NV21ToARGBRow = NV21ToARGBRow_SSSE3;
      }
    }
  }
#endif
#if defined(HAS_NV21TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    NV21ToARGBRow = NV21ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      NV21ToARGBRow = NV21ToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height; ++y) {
    NV21ToARGBRow(src_y, src_uv, dst_argb, width);
    dst_argb += dst_stride_argb;
    src_y += src_stride_y;
    if (y & 1) {
      src_uv += src_stride_uv;
    }
  }
  return 0;
}

// Convert M420 to ARGB.
LIBYUV_API
int M420ToARGB(const uint8* src_m420, int src_stride_m420,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_m420 || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  void (*NV12ToARGBRow)(const uint8* y_buf,
                        const uint8* uv_buf,
                        uint8* rgb_buf,
                        int width) = NV12ToARGBRow_C;
#if defined(HAS_NV12TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    NV12ToARGBRow = NV12ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_Unaligned_SSSE3;
      if (IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
        NV12ToARGBRow = NV12ToARGBRow_SSSE3;
      }
    }
  }
#elif defined(HAS_NV12TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    NV12ToARGBRow = NV12ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      NV12ToARGBRow = NV12ToARGBRow_NEON;
    }
  }
#endif

  for (int y = 0; y < height - 1; y += 2) {
    NV12ToARGBRow(src_m420, src_m420 + src_stride_m420 * 2, dst_argb, width);
    NV12ToARGBRow(src_m420 + src_stride_m420, src_m420 + src_stride_m420 * 2,
                  dst_argb + dst_stride_argb, width);
    dst_argb += dst_stride_argb * 2;
    src_m420 += src_stride_m420 * 3;
  }
  if (height & 1) {
    NV12ToARGBRow(src_m420, src_m420 + src_stride_m420 * 2, dst_argb, width);
  }
  return 0;
}

// Convert YUY2 to ARGB.
LIBYUV_API
int YUY2ToARGB(const uint8* src_yuy2, int src_stride_yuy2,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_yuy2 || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_yuy2 = src_yuy2 + (height - 1) * src_stride_yuy2;
    src_stride_yuy2 = -src_stride_yuy2;
  }
  // Coalesce rows.
  if (src_stride_yuy2 == width * 2 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_yuy2 = dst_stride_argb = 0;
  }
  void (*YUY2ToARGBRow)(const uint8* src_yuy2, uint8* dst_argb, int pix) =
      YUY2ToARGBRow_C;
#if defined(HAS_YUY2TOARGBROW_SSSE3)
  // Posix is 16, Windows is 8.
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 16) {
    YUY2ToARGBRow = YUY2ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      YUY2ToARGBRow = YUY2ToARGBRow_Unaligned_SSSE3;
      if (IS_ALIGNED(src_yuy2, 16) && IS_ALIGNED(src_stride_yuy2, 16) &&
          IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
        YUY2ToARGBRow = YUY2ToARGBRow_SSSE3;
      }
    }
  }
#elif defined(HAS_YUY2TOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    YUY2ToARGBRow = YUY2ToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      YUY2ToARGBRow = YUY2ToARGBRow_NEON;
    }
  }
#endif
  for (int y = 0; y < height; ++y) {
    YUY2ToARGBRow(src_yuy2, dst_argb, width);
    src_yuy2 += src_stride_yuy2;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

// Convert UYVY to ARGB.
LIBYUV_API
int UYVYToARGB(const uint8* src_uyvy, int src_stride_uyvy,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height) {
  if (!src_uyvy || !dst_argb ||
      width <= 0 || height == 0) {
    return -1;
  }
  // Negative height means invert the image.
  if (height < 0) {
    height = -height;
    src_uyvy = src_uyvy + (height - 1) * src_stride_uyvy;
    src_stride_uyvy = -src_stride_uyvy;
  }
  // Coalesce rows.
  if (src_stride_uyvy == width * 2 &&
      dst_stride_argb == width * 4) {
    width *= height;
    height = 1;
    src_stride_uyvy = dst_stride_argb = 0;
  }
  void (*UYVYToARGBRow)(const uint8* src_uyvy, uint8* dst_argb, int pix) =
      UYVYToARGBRow_C;
#if defined(HAS_UYVYTOARGBROW_SSSE3)
  // Posix is 16, Windows is 8.
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 16) {
    UYVYToARGBRow = UYVYToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      UYVYToARGBRow = UYVYToARGBRow_Unaligned_SSSE3;
      if (IS_ALIGNED(src_uyvy, 16) && IS_ALIGNED(src_stride_uyvy, 16) &&
          IS_ALIGNED(dst_argb, 16) && IS_ALIGNED(dst_stride_argb, 16)) {
        UYVYToARGBRow = UYVYToARGBRow_SSSE3;
      }
    }
  }
#elif defined(HAS_UYVYTOARGBROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    UYVYToARGBRow = UYVYToARGBRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      UYVYToARGBRow = UYVYToARGBRow_NEON;
    }
  }
#endif
  for (int y = 0; y < height; ++y) {
    UYVYToARGBRow(src_uyvy, dst_argb, width);
    src_uyvy += src_stride_uyvy;
    dst_argb += dst_stride_argb;
  }
  return 0;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
