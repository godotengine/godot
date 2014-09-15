/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/format_conversion.h"

#include "libyuv/basic_types.h"
#include "libyuv/cpu_id.h"
#include "libyuv/video_common.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// generate a selector mask useful for pshufb
static uint32 GenerateSelector(int select0, int select1) {
  return (uint32)(select0) |
         (uint32)((select1 + 4) << 8) |
         (uint32)((select0 + 8) << 16) |
         (uint32)((select1 + 12) << 24);
}

static int MakeSelectors(const int blue_index,
                         const int green_index,
                         const int red_index,
                         uint32 dst_fourcc_bayer,
                         uint32* index_map) {
  // Now build a lookup table containing the indices for the four pixels in each
  // 2x2 Bayer grid.
  switch (dst_fourcc_bayer) {
    case FOURCC_BGGR:
      index_map[0] = GenerateSelector(blue_index, green_index);
      index_map[1] = GenerateSelector(green_index, red_index);
      break;
    case FOURCC_GBRG:
      index_map[0] = GenerateSelector(green_index, blue_index);
      index_map[1] = GenerateSelector(red_index, green_index);
      break;
    case FOURCC_RGGB:
      index_map[0] = GenerateSelector(red_index, green_index);
      index_map[1] = GenerateSelector(green_index, blue_index);
      break;
    case FOURCC_GRBG:
      index_map[0] = GenerateSelector(green_index, red_index);
      index_map[1] = GenerateSelector(blue_index, green_index);
      break;
    default:
      return -1;  // Bad FourCC
  }
  return 0;
}

// Converts 32 bit ARGB to Bayer RGB formats.
LIBYUV_API
int ARGBToBayer(const uint8* src_argb, int src_stride_argb,
                uint8* dst_bayer, int dst_stride_bayer,
                int width, int height,
                uint32 dst_fourcc_bayer) {
  int y;
  const int blue_index = 0;  // Offsets for ARGB format
  const int green_index = 1;
  const int red_index = 2;
  uint32 index_map[2];
  void (*ARGBToBayerRow)(const uint8* src_argb, uint8* dst_bayer,
                         uint32 selector, int pix) = ARGBToBayerRow_C;
  if (height < 0) {
    height = -height;
    src_argb = src_argb + (height - 1) * src_stride_argb;
    src_stride_argb = -src_stride_argb;
  }
#if defined(HAS_ARGBTOBAYERROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8 &&
      IS_ALIGNED(src_argb, 16) && IS_ALIGNED(src_stride_argb, 16)) {
    ARGBToBayerRow = ARGBToBayerRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      ARGBToBayerRow = ARGBToBayerRow_SSSE3;
    }
  }
#elif defined(HAS_ARGBTOBAYERROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    ARGBToBayerRow = ARGBToBayerRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToBayerRow = ARGBToBayerRow_NEON;
    }
  }
#endif
  if (MakeSelectors(blue_index, green_index, red_index,
                    dst_fourcc_bayer, index_map)) {
    return -1;  // Bad FourCC
  }

  for (y = 0; y < height; ++y) {
    ARGBToBayerRow(src_argb, dst_bayer, index_map[y & 1], width);
    src_argb += src_stride_argb;
    dst_bayer += dst_stride_bayer;
  }
  return 0;
}

#define AVG(a, b) (((a) + (b)) >> 1)

static void BayerRowBG(const uint8* src_bayer0, int src_stride_bayer,
                       uint8* dst_argb, int pix) {
  const uint8* src_bayer1 = src_bayer0 + src_stride_bayer;
  uint8 g = src_bayer0[1];
  uint8 r = src_bayer1[1];
  int x;
  for (x = 0; x < pix - 2; x += 2) {
    dst_argb[0] = src_bayer0[0];
    dst_argb[1] = AVG(g, src_bayer0[1]);
    dst_argb[2] = AVG(r, src_bayer1[1]);
    dst_argb[3] = 255U;
    dst_argb[4] = AVG(src_bayer0[0], src_bayer0[2]);
    dst_argb[5] = src_bayer0[1];
    dst_argb[6] = src_bayer1[1];
    dst_argb[7] = 255U;
    g = src_bayer0[1];
    r = src_bayer1[1];
    src_bayer0 += 2;
    src_bayer1 += 2;
    dst_argb += 8;
  }
  dst_argb[0] = src_bayer0[0];
  dst_argb[1] = AVG(g, src_bayer0[1]);
  dst_argb[2] = AVG(r, src_bayer1[1]);
  dst_argb[3] = 255U;
  if (!(pix & 1)) {
    dst_argb[4] = src_bayer0[0];
    dst_argb[5] = src_bayer0[1];
    dst_argb[6] = src_bayer1[1];
    dst_argb[7] = 255U;
  }
}

static void BayerRowRG(const uint8* src_bayer0, int src_stride_bayer,
                       uint8* dst_argb, int pix) {
  const uint8* src_bayer1 = src_bayer0 + src_stride_bayer;
  uint8 g = src_bayer0[1];
  uint8 b = src_bayer1[1];
  int x;
  for (x = 0; x < pix - 2; x += 2) {
    dst_argb[0] = AVG(b, src_bayer1[1]);
    dst_argb[1] = AVG(g, src_bayer0[1]);
    dst_argb[2] = src_bayer0[0];
    dst_argb[3] = 255U;
    dst_argb[4] = src_bayer1[1];
    dst_argb[5] = src_bayer0[1];
    dst_argb[6] = AVG(src_bayer0[0], src_bayer0[2]);
    dst_argb[7] = 255U;
    g = src_bayer0[1];
    b = src_bayer1[1];
    src_bayer0 += 2;
    src_bayer1 += 2;
    dst_argb += 8;
  }
  dst_argb[0] = AVG(b, src_bayer1[1]);
  dst_argb[1] = AVG(g, src_bayer0[1]);
  dst_argb[2] = src_bayer0[0];
  dst_argb[3] = 255U;
  if (!(pix & 1)) {
    dst_argb[4] = src_bayer1[1];
    dst_argb[5] = src_bayer0[1];
    dst_argb[6] = src_bayer0[0];
    dst_argb[7] = 255U;
  }
}

static void BayerRowGB(const uint8* src_bayer0, int src_stride_bayer,
                       uint8* dst_argb, int pix) {
  const uint8* src_bayer1 = src_bayer0 + src_stride_bayer;
  uint8 b = src_bayer0[1];
  int x;
  for (x = 0; x < pix - 2; x += 2) {
    dst_argb[0] = AVG(b, src_bayer0[1]);
    dst_argb[1] = src_bayer0[0];
    dst_argb[2] = src_bayer1[0];
    dst_argb[3] = 255U;
    dst_argb[4] = src_bayer0[1];
    dst_argb[5] = AVG(src_bayer0[0], src_bayer0[2]);
    dst_argb[6] = AVG(src_bayer1[0], src_bayer1[2]);
    dst_argb[7] = 255U;
    b = src_bayer0[1];
    src_bayer0 += 2;
    src_bayer1 += 2;
    dst_argb += 8;
  }
  dst_argb[0] = AVG(b, src_bayer0[1]);
  dst_argb[1] = src_bayer0[0];
  dst_argb[2] = src_bayer1[0];
  dst_argb[3] = 255U;
  if (!(pix & 1)) {
    dst_argb[4] = src_bayer0[1];
    dst_argb[5] = src_bayer0[0];
    dst_argb[6] = src_bayer1[0];
    dst_argb[7] = 255U;
  }
}

static void BayerRowGR(const uint8* src_bayer0, int src_stride_bayer,
                       uint8* dst_argb, int pix) {
  const uint8* src_bayer1 = src_bayer0 + src_stride_bayer;
  uint8 r = src_bayer0[1];
  int x;
  for (x = 0; x < pix - 2; x += 2) {
    dst_argb[0] = src_bayer1[0];
    dst_argb[1] = src_bayer0[0];
    dst_argb[2] = AVG(r, src_bayer0[1]);
    dst_argb[3] = 255U;
    dst_argb[4] = AVG(src_bayer1[0], src_bayer1[2]);
    dst_argb[5] = AVG(src_bayer0[0], src_bayer0[2]);
    dst_argb[6] = src_bayer0[1];
    dst_argb[7] = 255U;
    r = src_bayer0[1];
    src_bayer0 += 2;
    src_bayer1 += 2;
    dst_argb += 8;
  }
  dst_argb[0] = src_bayer1[0];
  dst_argb[1] = src_bayer0[0];
  dst_argb[2] = AVG(r, src_bayer0[1]);
  dst_argb[3] = 255U;
  if (!(pix & 1)) {
    dst_argb[4] = src_bayer1[0];
    dst_argb[5] = src_bayer0[0];
    dst_argb[6] = src_bayer0[1];
    dst_argb[7] = 255U;
  }
}

// Converts any Bayer RGB format to ARGB.
LIBYUV_API
int BayerToARGB(const uint8* src_bayer, int src_stride_bayer,
                uint8* dst_argb, int dst_stride_argb,
                int width, int height,
                uint32 src_fourcc_bayer) {
  int y;
  void (*BayerRow0)(const uint8* src_bayer, int src_stride_bayer,
                    uint8* dst_argb, int pix);
  void (*BayerRow1)(const uint8* src_bayer, int src_stride_bayer,
                    uint8* dst_argb, int pix);
  if (height < 0) {
    height = -height;
    dst_argb = dst_argb + (height - 1) * dst_stride_argb;
    dst_stride_argb = -dst_stride_argb;
  }
  switch (src_fourcc_bayer) {
    case FOURCC_BGGR:
      BayerRow0 = BayerRowBG;
      BayerRow1 = BayerRowGR;
      break;
    case FOURCC_GBRG:
      BayerRow0 = BayerRowGB;
      BayerRow1 = BayerRowRG;
      break;
    case FOURCC_GRBG:
      BayerRow0 = BayerRowGR;
      BayerRow1 = BayerRowBG;
      break;
    case FOURCC_RGGB:
      BayerRow0 = BayerRowRG;
      BayerRow1 = BayerRowGB;
      break;
    default:
      return -1;    // Bad FourCC
  }

  for (y = 0; y < height - 1; y += 2) {
    BayerRow0(src_bayer, src_stride_bayer, dst_argb, width);
    BayerRow1(src_bayer + src_stride_bayer, -src_stride_bayer,
              dst_argb + dst_stride_argb, width);
    src_bayer += src_stride_bayer * 2;
    dst_argb += dst_stride_argb * 2;
  }
  if (height & 1) {
    BayerRow0(src_bayer, src_stride_bayer, dst_argb, width);
  }
  return 0;
}

// Converts any Bayer RGB format to ARGB.
LIBYUV_API
int BayerToI420(const uint8* src_bayer, int src_stride_bayer,
                uint8* dst_y, int dst_stride_y,
                uint8* dst_u, int dst_stride_u,
                uint8* dst_v, int dst_stride_v,
                int width, int height,
                uint32 src_fourcc_bayer) {
  void (*BayerRow0)(const uint8* src_bayer, int src_stride_bayer,
                    uint8* dst_argb, int pix);
  void (*BayerRow1)(const uint8* src_bayer, int src_stride_bayer,
                    uint8* dst_argb, int pix);

  void (*ARGBToUVRow)(const uint8* src_argb0, int src_stride_argb,
                      uint8* dst_u, uint8* dst_v, int width) = ARGBToUVRow_C;
  void (*ARGBToYRow)(const uint8* src_argb, uint8* dst_y, int pix) =
      ARGBToYRow_C;
  // Negative height means invert the image.
  if (height < 0) {
    int halfheight;
    height = -height;
    halfheight = (height + 1) >> 1;
    dst_y = dst_y + (height - 1) * dst_stride_y;
    dst_u = dst_u + (halfheight - 1) * dst_stride_u;
    dst_v = dst_v + (halfheight - 1) * dst_stride_v;
    dst_stride_y = -dst_stride_y;
    dst_stride_u = -dst_stride_u;
    dst_stride_v = -dst_stride_v;
  }
#if defined(HAS_ARGBTOYROW_SSSE3) && defined(HAS_ARGBTOUVROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 16) {
    ARGBToUVRow = ARGBToUVRow_Any_SSSE3;
    ARGBToYRow = ARGBToYRow_Any_SSSE3;
    if (IS_ALIGNED(width, 16)) {
      ARGBToYRow = ARGBToYRow_Unaligned_SSSE3;
      ARGBToUVRow = ARGBToUVRow_SSSE3;
      if (IS_ALIGNED(dst_y, 16) && IS_ALIGNED(dst_stride_y, 16)) {
        ARGBToYRow = ARGBToYRow_SSSE3;
      }
    }
  }
#elif defined(HAS_ARGBTOYROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    ARGBToYRow = ARGBToYRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToYRow = ARGBToYRow_NEON;
    }
    if (width >= 16) {
      ARGBToUVRow = ARGBToUVRow_Any_NEON;
      if (IS_ALIGNED(width, 16)) {
        ARGBToUVRow = ARGBToUVRow_NEON;
      }
    }
  }
#endif

  switch (src_fourcc_bayer) {
    case FOURCC_BGGR:
      BayerRow0 = BayerRowBG;
      BayerRow1 = BayerRowGR;
      break;
    case FOURCC_GBRG:
      BayerRow0 = BayerRowGB;
      BayerRow1 = BayerRowRG;
      break;
    case FOURCC_GRBG:
      BayerRow0 = BayerRowGR;
      BayerRow1 = BayerRowBG;
      break;
    case FOURCC_RGGB:
      BayerRow0 = BayerRowRG;
      BayerRow1 = BayerRowGB;
      break;
    default:
      return -1;  // Bad FourCC
  }

  {
    // Allocate 2 rows of ARGB.
    const int kRowSize = (width * 4 + 15) & ~15;
    align_buffer_64(row, kRowSize * 2);
    int y;
    for (y = 0; y < height - 1; y += 2) {
      BayerRow0(src_bayer, src_stride_bayer, row, width);
      BayerRow1(src_bayer + src_stride_bayer, -src_stride_bayer,
                row + kRowSize, width);
      ARGBToUVRow(row, kRowSize, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
      ARGBToYRow(row + kRowSize, dst_y + dst_stride_y, width);
      src_bayer += src_stride_bayer * 2;
      dst_y += dst_stride_y * 2;
      dst_u += dst_stride_u;
      dst_v += dst_stride_v;
    }
    if (height & 1) {
      BayerRow0(src_bayer, src_stride_bayer, row, width);
      ARGBToUVRow(row, 0, dst_u, dst_v, width);
      ARGBToYRow(row, dst_y, width);
    }
    free_aligned_buffer_64(row);
  }
  return 0;
}

// Convert I420 to Bayer.
LIBYUV_API
int I420ToBayer(const uint8* src_y, int src_stride_y,
                const uint8* src_u, int src_stride_u,
                const uint8* src_v, int src_stride_v,
                uint8* dst_bayer, int dst_stride_bayer,
                int width, int height,
                uint32 dst_fourcc_bayer) {
  void (*I422ToARGBRow)(const uint8* y_buf,
                        const uint8* u_buf,
                        const uint8* v_buf,
                        uint8* rgb_buf,
                        int width) = I422ToARGBRow_C;
  void (*ARGBToBayerRow)(const uint8* src_argb, uint8* dst_bayer,
                         uint32 selector, int pix) = ARGBToBayerRow_C;
  const int blue_index = 0;  // Offsets for ARGB format
  const int green_index = 1;
  const int red_index = 2;
  uint32 index_map[2];
  // Negative height means invert the image.
  if (height < 0) {
    int halfheight;
    height = -height;
    halfheight = (height + 1) >> 1;
    src_y = src_y + (height - 1) * src_stride_y;
    src_u = src_u + (halfheight - 1) * src_stride_u;
    src_v = src_v + (halfheight - 1) * src_stride_v;
    src_stride_y = -src_stride_y;
    src_stride_u = -src_stride_u;
    src_stride_v = -src_stride_v;
  }
#if defined(HAS_I422TOARGBROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    I422ToARGBRow = I422ToARGBRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      I422ToARGBRow = I422ToARGBRow_SSSE3;
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
      IS_ALIGNED(src_v, 2) && IS_ALIGNED(src_stride_v, 2)) {
    I422ToARGBRow = I422ToARGBRow_MIPS_DSPR2;
  }
#endif

#if defined(HAS_ARGBTOBAYERROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3) && width >= 8) {
    ARGBToBayerRow = ARGBToBayerRow_Any_SSSE3;
    if (IS_ALIGNED(width, 8)) {
      ARGBToBayerRow = ARGBToBayerRow_SSSE3;
    }
  }
#elif defined(HAS_ARGBTOBAYERROW_NEON)
  if (TestCpuFlag(kCpuHasNEON) && width >= 8) {
    ARGBToBayerRow = ARGBToBayerRow_Any_NEON;
    if (IS_ALIGNED(width, 8)) {
      ARGBToBayerRow = ARGBToBayerRow_NEON;
    }
  }
#endif

  if (MakeSelectors(blue_index, green_index, red_index,
                    dst_fourcc_bayer, index_map)) {
    return -1;  // Bad FourCC
  }
  {
    // Allocate a row of ARGB.
    align_buffer_64(row, width * 4);
    int y;
    for (y = 0; y < height; ++y) {
      I422ToARGBRow(src_y, src_u, src_v, row, width);
      ARGBToBayerRow(row, dst_bayer, index_map[y & 1], width);
      dst_bayer += dst_stride_bayer;
      src_y += src_stride_y;
      if (y & 1) {
        src_u += src_stride_u;
        src_v += src_stride_v;
      }
    }
    free_aligned_buffer_64(row);
  }
  return 0;
}

#define MAKEBAYERFOURCC(BAYER)                                                 \
LIBYUV_API                                                                     \
int Bayer##BAYER##ToI420(const uint8* src_bayer, int src_stride_bayer,         \
                         uint8* dst_y, int dst_stride_y,                       \
                         uint8* dst_u, int dst_stride_u,                       \
                         uint8* dst_v, int dst_stride_v,                       \
                         int width, int height) {                              \
  return BayerToI420(src_bayer, src_stride_bayer,                              \
                     dst_y, dst_stride_y,                                      \
                     dst_u, dst_stride_u,                                      \
                     dst_v, dst_stride_v,                                      \
                     width, height,                                            \
                     FOURCC_##BAYER);                                          \
}                                                                              \
                                                                               \
LIBYUV_API                                                                     \
int I420ToBayer##BAYER(const uint8* src_y, int src_stride_y,                   \
                       const uint8* src_u, int src_stride_u,                   \
                       const uint8* src_v, int src_stride_v,                   \
                       uint8* dst_bayer, int dst_stride_bayer,                 \
                       int width, int height) {                                \
  return I420ToBayer(src_y, src_stride_y,                                      \
                     src_u, src_stride_u,                                      \
                     src_v, src_stride_v,                                      \
                     dst_bayer, dst_stride_bayer,                              \
                     width, height,                                            \
                     FOURCC_##BAYER);                                          \
}                                                                              \
                                                                               \
LIBYUV_API                                                                     \
int ARGBToBayer##BAYER(const uint8* src_argb, int src_stride_argb,             \
                       uint8* dst_bayer, int dst_stride_bayer,                 \
                       int width, int height) {                                \
  return ARGBToBayer(src_argb, src_stride_argb,                                \
                     dst_bayer, dst_stride_bayer,                              \
                     width, height,                                            \
                     FOURCC_##BAYER);                                          \
}                                                                              \
                                                                               \
LIBYUV_API                                                                     \
int Bayer##BAYER##ToARGB(const uint8* src_bayer, int src_stride_bayer,         \
                         uint8* dst_argb, int dst_stride_argb,                 \
                         int width, int height) {                              \
  return BayerToARGB(src_bayer, src_stride_bayer,                              \
                     dst_argb, dst_stride_argb,                                \
                     width, height,                                            \
                     FOURCC_##BAYER);                                          \
}

MAKEBAYERFOURCC(BGGR)
MAKEBAYERFOURCC(GBRG)
MAKEBAYERFOURCC(GRBG)
MAKEBAYERFOURCC(RGGB)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
