/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"

#include <assert.h>
#include <string.h>  // For memcpy and memset.

#include "libyuv/basic_types.h"
#include "libyuv/convert_argb.h"  // For kYuvI601Constants

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#ifdef __cplusplus
#define STATIC_CAST(type, expr) static_cast<type>(expr)
#else
#define STATIC_CAST(type, expr) (type)(expr)
#endif

// This macro controls YUV to RGB using unsigned math to extend range of
// YUV to RGB coefficients to 0 to 4 instead of 0 to 2 for more accuracy on B:
// LIBYUV_UNLIMITED_DATA

// Macros to enable unlimited data for each colorspace
// LIBYUV_UNLIMITED_BT601
// LIBYUV_UNLIMITED_BT709
// LIBYUV_UNLIMITED_BT2020

#if defined(LIBYUV_BIT_EXACT)
#define LIBYUV_UNATTENUATE_DUP 1
#endif

// llvm x86 is poor at ternary operator, so use branchless min/max.

#define USE_BRANCHLESS 1
#if defined(USE_BRANCHLESS)
static __inline int32_t clamp0(int32_t v) {
  return -(v >= 0) & v;
}
// TODO(fbarchard): make clamp255 preserve negative values.
static __inline int32_t clamp255(int32_t v) {
  return (-(v >= 255) | v) & 255;
}

static __inline int32_t clamp1023(int32_t v) {
  return (-(v >= 1023) | v) & 1023;
}

// clamp to max
static __inline int32_t ClampMax(int32_t v, int32_t max) {
  return (-(v >= max) | v) & max;
}

static __inline uint32_t Abs(int32_t v) {
  int m = -(v < 0);
  return (v + m) ^ m;
}
#else   // USE_BRANCHLESS
static __inline int32_t clamp0(int32_t v) {
  return (v < 0) ? 0 : v;
}

static __inline int32_t clamp255(int32_t v) {
  return (v > 255) ? 255 : v;
}

static __inline int32_t clamp1023(int32_t v) {
  return (v > 1023) ? 1023 : v;
}

static __inline int32_t ClampMax(int32_t v, int32_t max) {
  return (v > max) ? max : v;
}

static __inline uint32_t Abs(int32_t v) {
  return (v < 0) ? -v : v;
}
#endif  // USE_BRANCHLESS
static __inline uint32_t Clamp(int32_t val) {
  int v = clamp0(val);
  return (uint32_t)(clamp255(v));
}

static __inline uint32_t Clamp10(int32_t val) {
  int v = clamp0(val);
  return (uint32_t)(clamp1023(v));
}

// Little Endian
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86) || defined(__arm__) || defined(_M_ARM) ||     \
    (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define WRITEWORD(p, v) *(uint32_t*)(p) = v
#else
static inline void WRITEWORD(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v & 255);
  p[1] = (uint8_t)((v >> 8) & 255);
  p[2] = (uint8_t)((v >> 16) & 255);
  p[3] = (uint8_t)((v >> 24) & 255);
}
#endif

void RGB24ToARGBRow_C(const uint8_t* src_rgb24, uint8_t* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_rgb24[0];
    uint8_t g = src_rgb24[1];
    uint8_t r = src_rgb24[2];
    dst_argb[0] = b;
    dst_argb[1] = g;
    dst_argb[2] = r;
    dst_argb[3] = 255u;
    dst_argb += 4;
    src_rgb24 += 3;
  }
}

void RAWToARGBRow_C(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t r = src_raw[0];
    uint8_t g = src_raw[1];
    uint8_t b = src_raw[2];
    dst_argb[0] = b;
    dst_argb[1] = g;
    dst_argb[2] = r;
    dst_argb[3] = 255u;
    dst_argb += 4;
    src_raw += 3;
  }
}

void RAWToRGBARow_C(const uint8_t* src_raw, uint8_t* dst_rgba, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t r = src_raw[0];
    uint8_t g = src_raw[1];
    uint8_t b = src_raw[2];
    dst_rgba[0] = 255u;
    dst_rgba[1] = b;
    dst_rgba[2] = g;
    dst_rgba[3] = r;
    dst_rgba += 4;
    src_raw += 3;
  }
}

void RAWToRGB24Row_C(const uint8_t* src_raw, uint8_t* dst_rgb24, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t r = src_raw[0];
    uint8_t g = src_raw[1];
    uint8_t b = src_raw[2];
    dst_rgb24[0] = b;
    dst_rgb24[1] = g;
    dst_rgb24[2] = r;
    dst_rgb24 += 3;
    src_raw += 3;
  }
}

void RGB565ToARGBRow_C(const uint8_t* src_rgb565,
                       uint8_t* dst_argb,
                       int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = STATIC_CAST(uint8_t, src_rgb565[0] & 0x1f);
    uint8_t g = STATIC_CAST(
        uint8_t, (src_rgb565[0] >> 5) | ((src_rgb565[1] & 0x07) << 3));
    uint8_t r = STATIC_CAST(uint8_t, src_rgb565[1] >> 3);
    dst_argb[0] = STATIC_CAST(uint8_t, (b << 3) | (b >> 2));
    dst_argb[1] = STATIC_CAST(uint8_t, (g << 2) | (g >> 4));
    dst_argb[2] = STATIC_CAST(uint8_t, (r << 3) | (r >> 2));
    dst_argb[3] = 255u;
    dst_argb += 4;
    src_rgb565 += 2;
  }
}

void ARGB1555ToARGBRow_C(const uint8_t* src_argb1555,
                         uint8_t* dst_argb,
                         int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = STATIC_CAST(uint8_t, src_argb1555[0] & 0x1f);
    uint8_t g = STATIC_CAST(
        uint8_t, (src_argb1555[0] >> 5) | ((src_argb1555[1] & 0x03) << 3));
    uint8_t r = STATIC_CAST(uint8_t, (src_argb1555[1] & 0x7c) >> 2);
    uint8_t a = STATIC_CAST(uint8_t, src_argb1555[1] >> 7);
    dst_argb[0] = STATIC_CAST(uint8_t, (b << 3) | (b >> 2));
    dst_argb[1] = STATIC_CAST(uint8_t, (g << 3) | (g >> 2));
    dst_argb[2] = STATIC_CAST(uint8_t, (r << 3) | (r >> 2));
    dst_argb[3] = -a;
    dst_argb += 4;
    src_argb1555 += 2;
  }
}

void ARGB4444ToARGBRow_C(const uint8_t* src_argb4444,
                         uint8_t* dst_argb,
                         int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = STATIC_CAST(uint8_t, src_argb4444[0] & 0x0f);
    uint8_t g = STATIC_CAST(uint8_t, src_argb4444[0] >> 4);
    uint8_t r = STATIC_CAST(uint8_t, src_argb4444[1] & 0x0f);
    uint8_t a = STATIC_CAST(uint8_t, src_argb4444[1] >> 4);
    dst_argb[0] = STATIC_CAST(uint8_t, (b << 4) | b);
    dst_argb[1] = STATIC_CAST(uint8_t, (g << 4) | g);
    dst_argb[2] = STATIC_CAST(uint8_t, (r << 4) | r);
    dst_argb[3] = STATIC_CAST(uint8_t, (a << 4) | a);
    dst_argb += 4;
    src_argb4444 += 2;
  }
}

void AR30ToARGBRow_C(const uint8_t* src_ar30, uint8_t* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint32_t ar30;
    memcpy(&ar30, src_ar30, sizeof ar30);
    uint32_t b = (ar30 >> 2) & 0xff;
    uint32_t g = (ar30 >> 12) & 0xff;
    uint32_t r = (ar30 >> 22) & 0xff;
    uint32_t a = (ar30 >> 30) * 0x55;  // Replicate 2 bits to 8 bits.
    *(uint32_t*)(dst_argb) = b | (g << 8) | (r << 16) | (a << 24);
    dst_argb += 4;
    src_ar30 += 4;
  }
}

void AR30ToABGRRow_C(const uint8_t* src_ar30, uint8_t* dst_abgr, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint32_t ar30;
    memcpy(&ar30, src_ar30, sizeof ar30);
    uint32_t b = (ar30 >> 2) & 0xff;
    uint32_t g = (ar30 >> 12) & 0xff;
    uint32_t r = (ar30 >> 22) & 0xff;
    uint32_t a = (ar30 >> 30) * 0x55;  // Replicate 2 bits to 8 bits.
    *(uint32_t*)(dst_abgr) = r | (g << 8) | (b << 16) | (a << 24);
    dst_abgr += 4;
    src_ar30 += 4;
  }
}

void AR30ToAB30Row_C(const uint8_t* src_ar30, uint8_t* dst_ab30, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint32_t ar30;
    memcpy(&ar30, src_ar30, sizeof ar30);
    uint32_t b = ar30 & 0x3ff;
    uint32_t ga = ar30 & 0xc00ffc00;
    uint32_t r = (ar30 >> 20) & 0x3ff;
    *(uint32_t*)(dst_ab30) = r | ga | (b << 20);
    dst_ab30 += 4;
    src_ar30 += 4;
  }
}

void ARGBToABGRRow_C(const uint8_t* src_argb, uint8_t* dst_abgr, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_argb[0];
    uint8_t g = src_argb[1];
    uint8_t r = src_argb[2];
    uint8_t a = src_argb[3];
    dst_abgr[0] = r;
    dst_abgr[1] = g;
    dst_abgr[2] = b;
    dst_abgr[3] = a;
    dst_abgr += 4;
    src_argb += 4;
  }
}

void ARGBToBGRARow_C(const uint8_t* src_argb, uint8_t* dst_bgra, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_argb[0];
    uint8_t g = src_argb[1];
    uint8_t r = src_argb[2];
    uint8_t a = src_argb[3];
    dst_bgra[0] = a;
    dst_bgra[1] = r;
    dst_bgra[2] = g;
    dst_bgra[3] = b;
    dst_bgra += 4;
    src_argb += 4;
  }
}

void ARGBToRGBARow_C(const uint8_t* src_argb, uint8_t* dst_rgba, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_argb[0];
    uint8_t g = src_argb[1];
    uint8_t r = src_argb[2];
    uint8_t a = src_argb[3];
    dst_rgba[0] = a;
    dst_rgba[1] = b;
    dst_rgba[2] = g;
    dst_rgba[3] = r;
    dst_rgba += 4;
    src_argb += 4;
  }
}

void ARGBToRGB24Row_C(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_argb[0];
    uint8_t g = src_argb[1];
    uint8_t r = src_argb[2];
    dst_rgb[0] = b;
    dst_rgb[1] = g;
    dst_rgb[2] = r;
    dst_rgb += 3;
    src_argb += 4;
  }
}

void ARGBToRAWRow_C(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_argb[0];
    uint8_t g = src_argb[1];
    uint8_t r = src_argb[2];
    dst_rgb[0] = r;
    dst_rgb[1] = g;
    dst_rgb[2] = b;
    dst_rgb += 3;
    src_argb += 4;
  }
}

void RGBAToARGBRow_C(const uint8_t* src_rgba, uint8_t* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t a = src_rgba[0];
    uint8_t b = src_rgba[1];
    uint8_t g = src_rgba[2];
    uint8_t r = src_rgba[3];
    dst_argb[0] = b;
    dst_argb[1] = g;
    dst_argb[2] = r;
    dst_argb[3] = a;
    dst_argb += 4;
    src_rgba += 4;
  }
}

void ARGBToRGB565Row_C(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8_t b0 = src_argb[0] >> 3;
    uint8_t g0 = src_argb[1] >> 2;
    uint8_t r0 = src_argb[2] >> 3;
    uint8_t b1 = src_argb[4] >> 3;
    uint8_t g1 = src_argb[5] >> 2;
    uint8_t r1 = src_argb[6] >> 3;
    WRITEWORD(dst_rgb, b0 | (g0 << 5) | (r0 << 11) | (b1 << 16) | (g1 << 21) |
                           (r1 << 27));
    dst_rgb += 4;
    src_argb += 8;
  }
  if (width & 1) {
    uint8_t b0 = src_argb[0] >> 3;
    uint8_t g0 = src_argb[1] >> 2;
    uint8_t r0 = src_argb[2] >> 3;
    *(uint16_t*)(dst_rgb) = STATIC_CAST(uint16_t, b0 | (g0 << 5) | (r0 << 11));
  }
}

// dither4 is a row of 4 values from 4x4 dither matrix.
// The 4x4 matrix contains values to increase RGB.  When converting to
// fewer bits (565) this provides an ordered dither.
// The order in the 4x4 matrix in first byte is upper left.
// The 4 values are passed as an int, then referenced as an array, so
// endian will not affect order of the original matrix.  But the dither4
// will containing the first pixel in the lower byte for little endian
// or the upper byte for big endian.
void ARGBToRGB565DitherRow_C(const uint8_t* src_argb,
                             uint8_t* dst_rgb,
                             uint32_t dither4,
                             int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    int dither0 = ((const unsigned char*)(&dither4))[x & 3];
    int dither1 = ((const unsigned char*)(&dither4))[(x + 1) & 3];
    uint8_t b0 = STATIC_CAST(uint8_t, clamp255(src_argb[0] + dither0) >> 3);
    uint8_t g0 = STATIC_CAST(uint8_t, clamp255(src_argb[1] + dither0) >> 2);
    uint8_t r0 = STATIC_CAST(uint8_t, clamp255(src_argb[2] + dither0) >> 3);
    uint8_t b1 = STATIC_CAST(uint8_t, clamp255(src_argb[4] + dither1) >> 3);
    uint8_t g1 = STATIC_CAST(uint8_t, clamp255(src_argb[5] + dither1) >> 2);
    uint8_t r1 = STATIC_CAST(uint8_t, clamp255(src_argb[6] + dither1) >> 3);
    *(uint16_t*)(dst_rgb + 0) =
        STATIC_CAST(uint16_t, b0 | (g0 << 5) | (r0 << 11));
    *(uint16_t*)(dst_rgb + 2) =
        STATIC_CAST(uint16_t, b1 | (g1 << 5) | (r1 << 11));
    dst_rgb += 4;
    src_argb += 8;
  }
  if (width & 1) {
    int dither0 = ((const unsigned char*)(&dither4))[(width - 1) & 3];
    uint8_t b0 = STATIC_CAST(uint8_t, clamp255(src_argb[0] + dither0) >> 3);
    uint8_t g0 = STATIC_CAST(uint8_t, clamp255(src_argb[1] + dither0) >> 2);
    uint8_t r0 = STATIC_CAST(uint8_t, clamp255(src_argb[2] + dither0) >> 3);
    *(uint16_t*)(dst_rgb) = STATIC_CAST(uint16_t, b0 | (g0 << 5) | (r0 << 11));
  }
}

void ARGBToARGB1555Row_C(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8_t b0 = src_argb[0] >> 3;
    uint8_t g0 = src_argb[1] >> 3;
    uint8_t r0 = src_argb[2] >> 3;
    uint8_t a0 = src_argb[3] >> 7;
    uint8_t b1 = src_argb[4] >> 3;
    uint8_t g1 = src_argb[5] >> 3;
    uint8_t r1 = src_argb[6] >> 3;
    uint8_t a1 = src_argb[7] >> 7;
    *(uint16_t*)(dst_rgb + 0) =
        STATIC_CAST(uint16_t, b0 | (g0 << 5) | (r0 << 10) | (a0 << 15));
    *(uint16_t*)(dst_rgb + 2) =
        STATIC_CAST(uint16_t, b1 | (g1 << 5) | (r1 << 10) | (a1 << 15));
    dst_rgb += 4;
    src_argb += 8;
  }
  if (width & 1) {
    uint8_t b0 = src_argb[0] >> 3;
    uint8_t g0 = src_argb[1] >> 3;
    uint8_t r0 = src_argb[2] >> 3;
    uint8_t a0 = src_argb[3] >> 7;
    *(uint16_t*)(dst_rgb) =
        STATIC_CAST(uint16_t, b0 | (g0 << 5) | (r0 << 10) | (a0 << 15));
  }
}

void ARGBToARGB4444Row_C(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8_t b0 = src_argb[0] >> 4;
    uint8_t g0 = src_argb[1] >> 4;
    uint8_t r0 = src_argb[2] >> 4;
    uint8_t a0 = src_argb[3] >> 4;
    uint8_t b1 = src_argb[4] >> 4;
    uint8_t g1 = src_argb[5] >> 4;
    uint8_t r1 = src_argb[6] >> 4;
    uint8_t a1 = src_argb[7] >> 4;
    *(uint16_t*)(dst_rgb + 0) =
        STATIC_CAST(uint16_t, b0 | (g0 << 4) | (r0 << 8) | (a0 << 12));
    *(uint16_t*)(dst_rgb + 2) =
        STATIC_CAST(uint16_t, b1 | (g1 << 4) | (r1 << 8) | (a1 << 12));
    dst_rgb += 4;
    src_argb += 8;
  }
  if (width & 1) {
    uint8_t b0 = src_argb[0] >> 4;
    uint8_t g0 = src_argb[1] >> 4;
    uint8_t r0 = src_argb[2] >> 4;
    uint8_t a0 = src_argb[3] >> 4;
    *(uint16_t*)(dst_rgb) =
        STATIC_CAST(uint16_t, b0 | (g0 << 4) | (r0 << 8) | (a0 << 12));
  }
}

void ABGRToAR30Row_C(const uint8_t* src_abgr, uint8_t* dst_ar30, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint32_t r0 = (src_abgr[0] >> 6) | ((uint32_t)(src_abgr[0]) << 2);
    uint32_t g0 = (src_abgr[1] >> 6) | ((uint32_t)(src_abgr[1]) << 2);
    uint32_t b0 = (src_abgr[2] >> 6) | ((uint32_t)(src_abgr[2]) << 2);
    uint32_t a0 = (src_abgr[3] >> 6);
    *(uint32_t*)(dst_ar30) =
        STATIC_CAST(uint32_t, b0 | (g0 << 10) | (r0 << 20) | (a0 << 30));
    dst_ar30 += 4;
    src_abgr += 4;
  }
}

void ARGBToAR30Row_C(const uint8_t* src_argb, uint8_t* dst_ar30, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint32_t b0 = (src_argb[0] >> 6) | ((uint32_t)(src_argb[0]) << 2);
    uint32_t g0 = (src_argb[1] >> 6) | ((uint32_t)(src_argb[1]) << 2);
    uint32_t r0 = (src_argb[2] >> 6) | ((uint32_t)(src_argb[2]) << 2);
    uint32_t a0 = (src_argb[3] >> 6);
    *(uint32_t*)(dst_ar30) =
        STATIC_CAST(uint32_t, b0 | (g0 << 10) | (r0 << 20) | (a0 << 30));
    dst_ar30 += 4;
    src_argb += 4;
  }
}

void ARGBToAR64Row_C(const uint8_t* src_argb, uint16_t* dst_ar64, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint16_t b = src_argb[0] * 0x0101;
    uint16_t g = src_argb[1] * 0x0101;
    uint16_t r = src_argb[2] * 0x0101;
    uint16_t a = src_argb[3] * 0x0101;
    dst_ar64[0] = b;
    dst_ar64[1] = g;
    dst_ar64[2] = r;
    dst_ar64[3] = a;
    dst_ar64 += 4;
    src_argb += 4;
  }
}

void ARGBToAB64Row_C(const uint8_t* src_argb, uint16_t* dst_ab64, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint16_t b = src_argb[0] * 0x0101;
    uint16_t g = src_argb[1] * 0x0101;
    uint16_t r = src_argb[2] * 0x0101;
    uint16_t a = src_argb[3] * 0x0101;
    dst_ab64[0] = r;
    dst_ab64[1] = g;
    dst_ab64[2] = b;
    dst_ab64[3] = a;
    dst_ab64 += 4;
    src_argb += 4;
  }
}

void AR64ToARGBRow_C(const uint16_t* src_ar64, uint8_t* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_ar64[0] >> 8;
    uint8_t g = src_ar64[1] >> 8;
    uint8_t r = src_ar64[2] >> 8;
    uint8_t a = src_ar64[3] >> 8;
    dst_argb[0] = b;
    dst_argb[1] = g;
    dst_argb[2] = r;
    dst_argb[3] = a;
    dst_argb += 4;
    src_ar64 += 4;
  }
}

void AB64ToARGBRow_C(const uint16_t* src_ab64, uint8_t* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t r = src_ab64[0] >> 8;
    uint8_t g = src_ab64[1] >> 8;
    uint8_t b = src_ab64[2] >> 8;
    uint8_t a = src_ab64[3] >> 8;
    dst_argb[0] = b;
    dst_argb[1] = g;
    dst_argb[2] = r;
    dst_argb[3] = a;
    dst_argb += 4;
    src_ab64 += 4;
  }
}

void AR64ToAB64Row_C(const uint16_t* src_ar64, uint16_t* dst_ab64, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint16_t b = src_ar64[0];
    uint16_t g = src_ar64[1];
    uint16_t r = src_ar64[2];
    uint16_t a = src_ar64[3];
    dst_ab64[0] = r;
    dst_ab64[1] = g;
    dst_ab64[2] = b;
    dst_ab64[3] = a;
    dst_ab64 += 4;
    src_ar64 += 4;
  }
}

// TODO(fbarchard): Make shuffle compatible with SIMD versions
void AR64ShuffleRow_C(const uint8_t* src_ar64,
                      uint8_t* dst_ar64,
                      const uint8_t* shuffler,
                      int width) {
  const uint16_t* src_ar64_16 = (const uint16_t*)src_ar64;
  uint16_t* dst_ar64_16 = (uint16_t*)dst_ar64;
  int index0 = shuffler[0] / 2;
  int index1 = shuffler[2] / 2;
  int index2 = shuffler[4] / 2;
  int index3 = shuffler[6] / 2;
  // Shuffle a row of AR64.
  int x;
  for (x = 0; x < width / 2; ++x) {
    // To support in-place conversion.
    uint16_t b = src_ar64_16[index0];
    uint16_t g = src_ar64_16[index1];
    uint16_t r = src_ar64_16[index2];
    uint16_t a = src_ar64_16[index3];
    dst_ar64_16[0] = b;
    dst_ar64_16[1] = g;
    dst_ar64_16[2] = r;
    dst_ar64_16[3] = a;
    src_ar64_16 += 4;
    dst_ar64_16 += 4;
  }
}
// BT601 8 bit Y:
// b 0.114 * 219 = 24.966  = 25
// g 0.587 * 219 = 128.553 = 129
// r 0.299 * 219 = 65.481  = 66
// BT601 8 bit U:
// b  0.875  * 128 = 112.0    = 112
// g -0.5781 * 128 = −73.9968 = -74
// r -0.2969 * 128 = −38.0032 = -38
// BT601 8 bit V:
// b -0.1406 * 128 = −17.9968 = -18
// g -0.7344 * 128 = −94.0032 = -94
// r  0.875  * 128 = 112.0    = 112
static __inline uint8_t RGBToY(uint8_t r, uint8_t g, uint8_t b) {
  return STATIC_CAST(uint8_t, (66 * r + 129 * g + 25 * b + 0x1080) >> 8);
}
static __inline uint8_t RGBToU(uint8_t r, uint8_t g, uint8_t b) {
  return STATIC_CAST(uint8_t, (112 * b - 74 * g - 38 * r + 0x8000) >> 8);
}
static __inline uint8_t RGBToV(uint8_t r, uint8_t g, uint8_t b) {
  return STATIC_CAST(uint8_t, (112 * r - 94 * g - 18 * b + 0x8000) >> 8);
}
#define AVGB(a, b) (((a) + (b) + 1) >> 1)

#define MAKEROWY(NAME, R, G, B, BPP)                                       \
  void NAME##ToYRow_C(const uint8_t* src_rgb, uint8_t* dst_y, int width) { \
    int x;                                                                 \
    for (x = 0; x < width; ++x) {                                          \
      dst_y[0] = RGBToY(src_rgb[R], src_rgb[G], src_rgb[B]);               \
      src_rgb += BPP;                                                      \
      dst_y += 1;                                                          \
    }                                                                      \
  }                                                                        \
  void NAME##ToUVRow_C(const uint8_t* src_rgb, int src_stride_rgb,         \
                       uint8_t* dst_u, uint8_t* dst_v, int width) {        \
    const uint8_t* src_rgb1 = src_rgb + src_stride_rgb;                    \
    int x;                                                                 \
    for (x = 0; x < width - 1; x += 2) {                                   \
      uint8_t ab = (src_rgb[B] + src_rgb[B + BPP] + src_rgb1[B] +          \
                    src_rgb1[B + BPP] + 2) >>                              \
                   2;                                                      \
      uint8_t ag = (src_rgb[G] + src_rgb[G + BPP] + src_rgb1[G] +          \
                    src_rgb1[G + BPP] + 2) >>                              \
                   2;                                                      \
      uint8_t ar = (src_rgb[R] + src_rgb[R + BPP] + src_rgb1[R] +          \
                    src_rgb1[R + BPP] + 2) >>                              \
                   2;                                                      \
      dst_u[0] = RGBToU(ar, ag, ab);                                       \
      dst_v[0] = RGBToV(ar, ag, ab);                                       \
      src_rgb += BPP * 2;                                                  \
      src_rgb1 += BPP * 2;                                                 \
      dst_u += 1;                                                          \
      dst_v += 1;                                                          \
    }                                                                      \
    if (width & 1) {                                                       \
      uint8_t ab = (src_rgb[B] + src_rgb1[B] + 1) >> 1;                    \
      uint8_t ag = (src_rgb[G] + src_rgb1[G] + 1) >> 1;                    \
      uint8_t ar = (src_rgb[R] + src_rgb1[R] + 1) >> 1;                    \
      dst_u[0] = RGBToU(ar, ag, ab);                                       \
      dst_v[0] = RGBToV(ar, ag, ab);                                       \
    }                                                                      \
  }

MAKEROWY(ARGB, 2, 1, 0, 4)
MAKEROWY(BGRA, 1, 2, 3, 4)
MAKEROWY(ABGR, 0, 1, 2, 4)
MAKEROWY(RGBA, 3, 2, 1, 4)
MAKEROWY(RGB24, 2, 1, 0, 3)
MAKEROWY(RAW, 0, 1, 2, 3)
#undef MAKEROWY

// JPeg uses BT.601-1 full range
// y =  0.29900 * r + 0.58700 * g + 0.11400 * b
// u = -0.16874 * r - 0.33126 * g + 0.50000 * b  + center
// v =  0.50000 * r - 0.41869 * g - 0.08131 * b  + center
// JPeg 8 bit Y:
// b 0.11400 * 256 = 29.184 = 29
// g 0.58700 * 256 = 150.272 = 150
// r 0.29900 * 256 = 76.544 = 77
// JPeg 8 bit U:
// b  0.50000 * 256 = 128.0 = 128
// g -0.33126 * 256 = −84.80256 = -85
// r -0.16874 * 256 = −43.19744 = -43
// JPeg 8 bit V:
// b -0.08131 * 256 = −20.81536 = -21
// g -0.41869 * 256 = −107.18464 = -107
// r  0.50000 * 256 = 128.0 = 128

// 8 bit
static __inline uint8_t RGBToYJ(uint8_t r, uint8_t g, uint8_t b) {
  return (77 * r + 150 * g + 29 * b + 128) >> 8;
}
static __inline uint8_t RGBToUJ(uint8_t r, uint8_t g, uint8_t b) {
  return (128 * b - 85 * g - 43 * r + 0x8000) >> 8;
}
static __inline uint8_t RGBToVJ(uint8_t r, uint8_t g, uint8_t b) {
  return (128 * r - 107 * g - 21 * b + 0x8000) >> 8;
}

// ARGBToYJ_C and ARGBToUVJ_C
#define MAKEROWYJ(NAME, R, G, B, BPP)                                       \
  void NAME##ToYJRow_C(const uint8_t* src_rgb, uint8_t* dst_y, int width) { \
    int x;                                                                  \
    for (x = 0; x < width; ++x) {                                           \
      dst_y[0] = RGBToYJ(src_rgb[R], src_rgb[G], src_rgb[B]);               \
      src_rgb += BPP;                                                       \
      dst_y += 1;                                                           \
    }                                                                       \
  }                                                                         \
  void NAME##ToUVJRow_C(const uint8_t* src_rgb, int src_stride_rgb,         \
                        uint8_t* dst_u, uint8_t* dst_v, int width) {        \
    const uint8_t* src_rgb1 = src_rgb + src_stride_rgb;                     \
    int x;                                                                  \
    for (x = 0; x < width - 1; x += 2) {                                    \
      uint8_t ab = (src_rgb[B] + src_rgb[B + BPP] + src_rgb1[B] +           \
                    src_rgb1[B + BPP] + 2) >>                               \
                   2;                                                       \
      uint8_t ag = (src_rgb[G] + src_rgb[G + BPP] + src_rgb1[G] +           \
                    src_rgb1[G + BPP] + 2) >>                               \
                   2;                                                       \
      uint8_t ar = (src_rgb[R] + src_rgb[R + BPP] + src_rgb1[R] +           \
                    src_rgb1[R + BPP] + 2) >>                               \
                   2;                                                       \
      dst_u[0] = RGBToUJ(ar, ag, ab);                                       \
      dst_v[0] = RGBToVJ(ar, ag, ab);                                       \
      src_rgb += BPP * 2;                                                   \
      src_rgb1 += BPP * 2;                                                  \
      dst_u += 1;                                                           \
      dst_v += 1;                                                           \
    }                                                                       \
    if (width & 1) {                                                        \
      uint16_t ab = (src_rgb[B] + src_rgb1[B] + 1) >> 1;                    \
      uint16_t ag = (src_rgb[G] + src_rgb1[G] + 1) >> 1;                    \
      uint16_t ar = (src_rgb[R] + src_rgb1[R] + 1) >> 1;                    \
      dst_u[0] = RGBToUJ(ar, ag, ab);                                       \
      dst_v[0] = RGBToVJ(ar, ag, ab);                                       \
    }                                                                       \
  }

MAKEROWYJ(ARGB, 2, 1, 0, 4)
MAKEROWYJ(ABGR, 0, 1, 2, 4)
MAKEROWYJ(RGBA, 3, 2, 1, 4)
MAKEROWYJ(RGB24, 2, 1, 0, 3)
MAKEROWYJ(RAW, 0, 1, 2, 3)
#undef MAKEROWYJ

void RGB565ToYRow_C(const uint8_t* src_rgb565, uint8_t* dst_y, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_rgb565[0] & 0x1f;
    uint8_t g = STATIC_CAST(
        uint8_t, (src_rgb565[0] >> 5) | ((src_rgb565[1] & 0x07) << 3));
    uint8_t r = src_rgb565[1] >> 3;
    b = STATIC_CAST(uint8_t, (b << 3) | (b >> 2));
    g = STATIC_CAST(uint8_t, (g << 2) | (g >> 4));
    r = STATIC_CAST(uint8_t, (r << 3) | (r >> 2));
    dst_y[0] = RGBToY(r, g, b);
    src_rgb565 += 2;
    dst_y += 1;
  }
}

void ARGB1555ToYRow_C(const uint8_t* src_argb1555, uint8_t* dst_y, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_argb1555[0] & 0x1f;
    uint8_t g = STATIC_CAST(
        uint8_t, (src_argb1555[0] >> 5) | ((src_argb1555[1] & 0x03) << 3));
    uint8_t r = (src_argb1555[1] & 0x7c) >> 2;
    b = STATIC_CAST(uint8_t, (b << 3) | (b >> 2));
    g = STATIC_CAST(uint8_t, (g << 3) | (g >> 2));
    r = STATIC_CAST(uint8_t, (r << 3) | (r >> 2));
    dst_y[0] = RGBToY(r, g, b);
    src_argb1555 += 2;
    dst_y += 1;
  }
}

void ARGB4444ToYRow_C(const uint8_t* src_argb4444, uint8_t* dst_y, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_argb4444[0] & 0x0f;
    uint8_t g = src_argb4444[0] >> 4;
    uint8_t r = src_argb4444[1] & 0x0f;
    b = STATIC_CAST(uint8_t, (b << 4) | b);
    g = STATIC_CAST(uint8_t, (g << 4) | g);
    r = STATIC_CAST(uint8_t, (r << 4) | r);
    dst_y[0] = RGBToY(r, g, b);
    src_argb4444 += 2;
    dst_y += 1;
  }
}

void RGB565ToUVRow_C(const uint8_t* src_rgb565,
                     int src_stride_rgb565,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  const uint8_t* next_rgb565 = src_rgb565 + src_stride_rgb565;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8_t b0 = STATIC_CAST(uint8_t, src_rgb565[0] & 0x1f);
    uint8_t g0 = STATIC_CAST(
        uint8_t, (src_rgb565[0] >> 5) | ((src_rgb565[1] & 0x07) << 3));
    uint8_t r0 = STATIC_CAST(uint8_t, src_rgb565[1] >> 3);
    uint8_t b1 = STATIC_CAST(uint8_t, src_rgb565[2] & 0x1f);
    uint8_t g1 = STATIC_CAST(
        uint8_t, (src_rgb565[2] >> 5) | ((src_rgb565[3] & 0x07) << 3));
    uint8_t r1 = STATIC_CAST(uint8_t, src_rgb565[3] >> 3);
    uint8_t b2 = STATIC_CAST(uint8_t, next_rgb565[0] & 0x1f);
    uint8_t g2 = STATIC_CAST(
        uint8_t, (next_rgb565[0] >> 5) | ((next_rgb565[1] & 0x07) << 3));
    uint8_t r2 = STATIC_CAST(uint8_t, next_rgb565[1] >> 3);
    uint8_t b3 = STATIC_CAST(uint8_t, next_rgb565[2] & 0x1f);
    uint8_t g3 = STATIC_CAST(
        uint8_t, (next_rgb565[2] >> 5) | ((next_rgb565[3] & 0x07) << 3));
    uint8_t r3 = STATIC_CAST(uint8_t, next_rgb565[3] >> 3);

    b0 = STATIC_CAST(uint8_t, (b0 << 3) | (b0 >> 2));
    g0 = STATIC_CAST(uint8_t, (g0 << 2) | (g0 >> 4));
    r0 = STATIC_CAST(uint8_t, (r0 << 3) | (r0 >> 2));
    b1 = STATIC_CAST(uint8_t, (b1 << 3) | (b1 >> 2));
    g1 = STATIC_CAST(uint8_t, (g1 << 2) | (g1 >> 4));
    r1 = STATIC_CAST(uint8_t, (r1 << 3) | (r1 >> 2));
    b2 = STATIC_CAST(uint8_t, (b2 << 3) | (b2 >> 2));
    g2 = STATIC_CAST(uint8_t, (g2 << 2) | (g2 >> 4));
    r2 = STATIC_CAST(uint8_t, (r2 << 3) | (r2 >> 2));
    b3 = STATIC_CAST(uint8_t, (b3 << 3) | (b3 >> 2));
    g3 = STATIC_CAST(uint8_t, (g3 << 2) | (g3 >> 4));
    r3 = STATIC_CAST(uint8_t, (r3 << 3) | (r3 >> 2));

    uint8_t b = (b0 + b1 + b2 + b3 + 2) >> 2;
    uint8_t g = (g0 + g1 + g2 + g3 + 2) >> 2;
    uint8_t r = (r0 + r1 + r2 + r3 + 2) >> 2;
    dst_u[0] = RGBToU(r, g, b);
    dst_v[0] = RGBToV(r, g, b);

    src_rgb565 += 4;
    next_rgb565 += 4;
    dst_u += 1;
    dst_v += 1;
  }
  if (width & 1) {
    uint8_t b0 = STATIC_CAST(uint8_t, src_rgb565[0] & 0x1f);
    uint8_t g0 = STATIC_CAST(
        uint8_t, (src_rgb565[0] >> 5) | ((src_rgb565[1] & 0x07) << 3));
    uint8_t r0 = STATIC_CAST(uint8_t, src_rgb565[1] >> 3);
    uint8_t b2 = STATIC_CAST(uint8_t, next_rgb565[0] & 0x1f);
    uint8_t g2 = STATIC_CAST(
        uint8_t, (next_rgb565[0] >> 5) | ((next_rgb565[1] & 0x07) << 3));
    uint8_t r2 = STATIC_CAST(uint8_t, next_rgb565[1] >> 3);
    b0 = STATIC_CAST(uint8_t, (b0 << 3) | (b0 >> 2));
    g0 = STATIC_CAST(uint8_t, (g0 << 2) | (g0 >> 4));
    r0 = STATIC_CAST(uint8_t, (r0 << 3) | (r0 >> 2));
    b2 = STATIC_CAST(uint8_t, (b2 << 3) | (b2 >> 2));
    g2 = STATIC_CAST(uint8_t, (g2 << 2) | (g2 >> 4));
    r2 = STATIC_CAST(uint8_t, (r2 << 3) | (r2 >> 2));

    uint8_t ab = AVGB(b0, b2);
    uint8_t ag = AVGB(g0, g2);
    uint8_t ar = AVGB(r0, r2);
    dst_u[0] = RGBToU(ar, ag, ab);
    dst_v[0] = RGBToV(ar, ag, ab);
  }
}

void ARGB1555ToUVRow_C(const uint8_t* src_argb1555,
                       int src_stride_argb1555,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  const uint8_t* next_argb1555 = src_argb1555 + src_stride_argb1555;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8_t b0 = STATIC_CAST(uint8_t, src_argb1555[0] & 0x1f);
    uint8_t g0 = STATIC_CAST(
        uint8_t, (src_argb1555[0] >> 5) | ((src_argb1555[1] & 0x03) << 3));
    uint8_t r0 = STATIC_CAST(uint8_t, (src_argb1555[1] & 0x7c) >> 2);
    uint8_t b1 = STATIC_CAST(uint8_t, src_argb1555[2] & 0x1f);
    uint8_t g1 = STATIC_CAST(
        uint8_t, (src_argb1555[2] >> 5) | ((src_argb1555[3] & 0x03) << 3));
    uint8_t r1 = STATIC_CAST(uint8_t, (src_argb1555[3] & 0x7c) >> 2);
    uint8_t b2 = STATIC_CAST(uint8_t, next_argb1555[0] & 0x1f);
    uint8_t g2 = STATIC_CAST(
        uint8_t, (next_argb1555[0] >> 5) | ((next_argb1555[1] & 0x03) << 3));
    uint8_t r2 = STATIC_CAST(uint8_t, (next_argb1555[1] & 0x7c) >> 2);
    uint8_t b3 = STATIC_CAST(uint8_t, next_argb1555[2] & 0x1f);
    uint8_t g3 = STATIC_CAST(
        uint8_t, (next_argb1555[2] >> 5) | ((next_argb1555[3] & 0x03) << 3));
    uint8_t r3 = STATIC_CAST(uint8_t, (next_argb1555[3] & 0x7c) >> 2);

    b0 = STATIC_CAST(uint8_t, (b0 << 3) | (b0 >> 2));
    g0 = STATIC_CAST(uint8_t, (g0 << 3) | (g0 >> 2));
    r0 = STATIC_CAST(uint8_t, (r0 << 3) | (r0 >> 2));
    b1 = STATIC_CAST(uint8_t, (b1 << 3) | (b1 >> 2));
    g1 = STATIC_CAST(uint8_t, (g1 << 3) | (g1 >> 2));
    r1 = STATIC_CAST(uint8_t, (r1 << 3) | (r1 >> 2));
    b2 = STATIC_CAST(uint8_t, (b2 << 3) | (b2 >> 2));
    g2 = STATIC_CAST(uint8_t, (g2 << 3) | (g2 >> 2));
    r2 = STATIC_CAST(uint8_t, (r2 << 3) | (r2 >> 2));
    b3 = STATIC_CAST(uint8_t, (b3 << 3) | (b3 >> 2));
    g3 = STATIC_CAST(uint8_t, (g3 << 3) | (g3 >> 2));
    r3 = STATIC_CAST(uint8_t, (r3 << 3) | (r3 >> 2));

    uint8_t b = (b0 + b1 + b2 + b3 + 2) >> 2;
    uint8_t g = (g0 + g1 + g2 + g3 + 2) >> 2;
    uint8_t r = (r0 + r1 + r2 + r3 + 2) >> 2;
    dst_u[0] = RGBToU(r, g, b);
    dst_v[0] = RGBToV(r, g, b);

    src_argb1555 += 4;
    next_argb1555 += 4;
    dst_u += 1;
    dst_v += 1;
  }
  if (width & 1) {
    uint8_t b0 = STATIC_CAST(uint8_t, src_argb1555[0] & 0x1f);
    uint8_t g0 = STATIC_CAST(
        uint8_t, (src_argb1555[0] >> 5) | ((src_argb1555[1] & 0x03) << 3));
    uint8_t r0 = STATIC_CAST(uint8_t, (src_argb1555[1] & 0x7c) >> 2);
    uint8_t b2 = STATIC_CAST(uint8_t, next_argb1555[0] & 0x1f);
    uint8_t g2 = STATIC_CAST(
        uint8_t, (next_argb1555[0] >> 5) | ((next_argb1555[1] & 0x03) << 3));
    uint8_t r2 = STATIC_CAST(uint8_t, (next_argb1555[1] & 0x7c) >> 2);

    b0 = STATIC_CAST(uint8_t, (b0 << 3) | (b0 >> 2));
    g0 = STATIC_CAST(uint8_t, (g0 << 3) | (g0 >> 2));
    r0 = STATIC_CAST(uint8_t, (r0 << 3) | (r0 >> 2));
    b2 = STATIC_CAST(uint8_t, (b2 << 3) | (b2 >> 2));
    g2 = STATIC_CAST(uint8_t, (g2 << 3) | (g2 >> 2));
    r2 = STATIC_CAST(uint8_t, (r2 << 3) | (r2 >> 2));

    uint8_t ab = AVGB(b0, b2);
    uint8_t ag = AVGB(g0, g2);
    uint8_t ar = AVGB(r0, r2);
    dst_u[0] = RGBToU(ar, ag, ab);
    dst_v[0] = RGBToV(ar, ag, ab);
  }
}

void ARGB4444ToUVRow_C(const uint8_t* src_argb4444,
                       int src_stride_argb4444,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  const uint8_t* next_argb4444 = src_argb4444 + src_stride_argb4444;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8_t b0 = src_argb4444[0] & 0x0f;
    uint8_t g0 = src_argb4444[0] >> 4;
    uint8_t r0 = src_argb4444[1] & 0x0f;
    uint8_t b1 = src_argb4444[2] & 0x0f;
    uint8_t g1 = src_argb4444[2] >> 4;
    uint8_t r1 = src_argb4444[3] & 0x0f;
    uint8_t b2 = next_argb4444[0] & 0x0f;
    uint8_t g2 = next_argb4444[0] >> 4;
    uint8_t r2 = next_argb4444[1] & 0x0f;
    uint8_t b3 = next_argb4444[2] & 0x0f;
    uint8_t g3 = next_argb4444[2] >> 4;
    uint8_t r3 = next_argb4444[3] & 0x0f;

    b0 = STATIC_CAST(uint8_t, (b0 << 4) | b0);
    g0 = STATIC_CAST(uint8_t, (g0 << 4) | g0);
    r0 = STATIC_CAST(uint8_t, (r0 << 4) | r0);
    b1 = STATIC_CAST(uint8_t, (b1 << 4) | b1);
    g1 = STATIC_CAST(uint8_t, (g1 << 4) | g1);
    r1 = STATIC_CAST(uint8_t, (r1 << 4) | r1);
    b2 = STATIC_CAST(uint8_t, (b2 << 4) | b2);
    g2 = STATIC_CAST(uint8_t, (g2 << 4) | g2);
    r2 = STATIC_CAST(uint8_t, (r2 << 4) | r2);
    b3 = STATIC_CAST(uint8_t, (b3 << 4) | b3);
    g3 = STATIC_CAST(uint8_t, (g3 << 4) | g3);
    r3 = STATIC_CAST(uint8_t, (r3 << 4) | r3);

    uint8_t b = (b0 + b1 + b2 + b3 + 2) >> 2;
    uint8_t g = (g0 + g1 + g2 + g3 + 2) >> 2;
    uint8_t r = (r0 + r1 + r2 + r3 + 2) >> 2;
    dst_u[0] = RGBToU(r, g, b);
    dst_v[0] = RGBToV(r, g, b);

    src_argb4444 += 4;
    next_argb4444 += 4;
    dst_u += 1;
    dst_v += 1;
  }
  if (width & 1) {
    uint8_t b0 = src_argb4444[0] & 0x0f;
    uint8_t g0 = src_argb4444[0] >> 4;
    uint8_t r0 = src_argb4444[1] & 0x0f;
    uint8_t b2 = next_argb4444[0] & 0x0f;
    uint8_t g2 = next_argb4444[0] >> 4;
    uint8_t r2 = next_argb4444[1] & 0x0f;

    b0 = STATIC_CAST(uint8_t, (b0 << 4) | b0);
    g0 = STATIC_CAST(uint8_t, (g0 << 4) | g0);
    r0 = STATIC_CAST(uint8_t, (r0 << 4) | r0);
    b2 = STATIC_CAST(uint8_t, (b2 << 4) | b2);
    g2 = STATIC_CAST(uint8_t, (g2 << 4) | g2);
    r2 = STATIC_CAST(uint8_t, (r2 << 4) | r2);

    uint8_t ab = AVGB(b0, b2);
    uint8_t ag = AVGB(g0, g2);
    uint8_t ar = AVGB(r0, r2);
    dst_u[0] = RGBToU(ar, ag, ab);
    dst_v[0] = RGBToV(ar, ag, ab);
  }
}

void ARGBToUV444Row_C(const uint8_t* src_argb,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t ab = src_argb[0];
    uint8_t ag = src_argb[1];
    uint8_t ar = src_argb[2];
    dst_u[0] = RGBToU(ar, ag, ab);
    dst_v[0] = RGBToV(ar, ag, ab);
    src_argb += 4;
    dst_u += 1;
    dst_v += 1;
  }
}

void ARGBToUVJ444Row_C(const uint8_t* src_argb,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t ab = src_argb[0];
    uint8_t ag = src_argb[1];
    uint8_t ar = src_argb[2];
    dst_u[0] = RGBToUJ(ar, ag, ab);
    dst_v[0] = RGBToVJ(ar, ag, ab);
    src_argb += 4;
    dst_u += 1;
    dst_v += 1;
  }
}

void ARGBGrayRow_C(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t y = RGBToYJ(src_argb[2], src_argb[1], src_argb[0]);
    dst_argb[2] = dst_argb[1] = dst_argb[0] = y;
    dst_argb[3] = src_argb[3];
    dst_argb += 4;
    src_argb += 4;
  }
}

// Convert a row of image to Sepia tone.
void ARGBSepiaRow_C(uint8_t* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    int b = dst_argb[0];
    int g = dst_argb[1];
    int r = dst_argb[2];
    int sb = (b * 17 + g * 68 + r * 35) >> 7;
    int sg = (b * 22 + g * 88 + r * 45) >> 7;
    int sr = (b * 24 + g * 98 + r * 50) >> 7;
    // b does not over flow. a is preserved from original.
    dst_argb[0] = STATIC_CAST(uint8_t, sb);
    dst_argb[1] = STATIC_CAST(uint8_t, clamp255(sg));
    dst_argb[2] = STATIC_CAST(uint8_t, clamp255(sr));
    dst_argb += 4;
  }
}

// Apply color matrix to a row of image. Matrix is signed.
// TODO(fbarchard): Consider adding rounding (+32).
void ARGBColorMatrixRow_C(const uint8_t* src_argb,
                          uint8_t* dst_argb,
                          const int8_t* matrix_argb,
                          int width) {
  int x;
  for (x = 0; x < width; ++x) {
    int b = src_argb[0];
    int g = src_argb[1];
    int r = src_argb[2];
    int a = src_argb[3];
    int sb = (b * matrix_argb[0] + g * matrix_argb[1] + r * matrix_argb[2] +
              a * matrix_argb[3]) >>
             6;
    int sg = (b * matrix_argb[4] + g * matrix_argb[5] + r * matrix_argb[6] +
              a * matrix_argb[7]) >>
             6;
    int sr = (b * matrix_argb[8] + g * matrix_argb[9] + r * matrix_argb[10] +
              a * matrix_argb[11]) >>
             6;
    int sa = (b * matrix_argb[12] + g * matrix_argb[13] + r * matrix_argb[14] +
              a * matrix_argb[15]) >>
             6;
    dst_argb[0] = STATIC_CAST(uint8_t, Clamp(sb));
    dst_argb[1] = STATIC_CAST(uint8_t, Clamp(sg));
    dst_argb[2] = STATIC_CAST(uint8_t, Clamp(sr));
    dst_argb[3] = STATIC_CAST(uint8_t, Clamp(sa));
    src_argb += 4;
    dst_argb += 4;
  }
}

// Apply color table to a row of image.
void ARGBColorTableRow_C(uint8_t* dst_argb,
                         const uint8_t* table_argb,
                         int width) {
  int x;
  for (x = 0; x < width; ++x) {
    int b = dst_argb[0];
    int g = dst_argb[1];
    int r = dst_argb[2];
    int a = dst_argb[3];
    dst_argb[0] = table_argb[b * 4 + 0];
    dst_argb[1] = table_argb[g * 4 + 1];
    dst_argb[2] = table_argb[r * 4 + 2];
    dst_argb[3] = table_argb[a * 4 + 3];
    dst_argb += 4;
  }
}

// Apply color table to a row of image.
void RGBColorTableRow_C(uint8_t* dst_argb,
                        const uint8_t* table_argb,
                        int width) {
  int x;
  for (x = 0; x < width; ++x) {
    int b = dst_argb[0];
    int g = dst_argb[1];
    int r = dst_argb[2];
    dst_argb[0] = table_argb[b * 4 + 0];
    dst_argb[1] = table_argb[g * 4 + 1];
    dst_argb[2] = table_argb[r * 4 + 2];
    dst_argb += 4;
  }
}

void ARGBQuantizeRow_C(uint8_t* dst_argb,
                       int scale,
                       int interval_size,
                       int interval_offset,
                       int width) {
  int x;
  for (x = 0; x < width; ++x) {
    int b = dst_argb[0];
    int g = dst_argb[1];
    int r = dst_argb[2];
    dst_argb[0] = STATIC_CAST(
        uint8_t, (b * scale >> 16) * interval_size + interval_offset);
    dst_argb[1] = STATIC_CAST(
        uint8_t, (g * scale >> 16) * interval_size + interval_offset);
    dst_argb[2] = STATIC_CAST(
        uint8_t, (r * scale >> 16) * interval_size + interval_offset);
    dst_argb += 4;
  }
}

#define REPEAT8(v) (v) | ((v) << 8)
#define SHADE(f, v) v* f >> 24

void ARGBShadeRow_C(const uint8_t* src_argb,
                    uint8_t* dst_argb,
                    int width,
                    uint32_t value) {
  const uint32_t b_scale = REPEAT8(value & 0xff);
  const uint32_t g_scale = REPEAT8((value >> 8) & 0xff);
  const uint32_t r_scale = REPEAT8((value >> 16) & 0xff);
  const uint32_t a_scale = REPEAT8(value >> 24);

  int i;
  for (i = 0; i < width; ++i) {
    const uint32_t b = REPEAT8(src_argb[0]);
    const uint32_t g = REPEAT8(src_argb[1]);
    const uint32_t r = REPEAT8(src_argb[2]);
    const uint32_t a = REPEAT8(src_argb[3]);
    dst_argb[0] = SHADE(b, b_scale);
    dst_argb[1] = SHADE(g, g_scale);
    dst_argb[2] = SHADE(r, r_scale);
    dst_argb[3] = SHADE(a, a_scale);
    src_argb += 4;
    dst_argb += 4;
  }
}
#undef REPEAT8
#undef SHADE

void ARGBMultiplyRow_C(const uint8_t* src_argb,
                       const uint8_t* src_argb1,
                       uint8_t* dst_argb,
                       int width) {
  int i;
  for (i = 0; i < width; ++i) {
    const uint32_t b = src_argb[0];
    const uint32_t g = src_argb[1];
    const uint32_t r = src_argb[2];
    const uint32_t a = src_argb[3];
    const uint32_t b_scale = src_argb1[0];
    const uint32_t g_scale = src_argb1[1];
    const uint32_t r_scale = src_argb1[2];
    const uint32_t a_scale = src_argb1[3];
    dst_argb[0] = STATIC_CAST(uint8_t, (b * b_scale + 128) >> 8);
    dst_argb[1] = STATIC_CAST(uint8_t, (g * g_scale + 128) >> 8);
    dst_argb[2] = STATIC_CAST(uint8_t, (r * r_scale + 128) >> 8);
    dst_argb[3] = STATIC_CAST(uint8_t, (a * a_scale + 128) >> 8);
    src_argb += 4;
    src_argb1 += 4;
    dst_argb += 4;
  }
}

#define SHADE(f, v) clamp255(v + f)

void ARGBAddRow_C(const uint8_t* src_argb,
                  const uint8_t* src_argb1,
                  uint8_t* dst_argb,
                  int width) {
  int i;
  for (i = 0; i < width; ++i) {
    const int b = src_argb[0];
    const int g = src_argb[1];
    const int r = src_argb[2];
    const int a = src_argb[3];
    const int b_add = src_argb1[0];
    const int g_add = src_argb1[1];
    const int r_add = src_argb1[2];
    const int a_add = src_argb1[3];
    dst_argb[0] = STATIC_CAST(uint8_t, SHADE(b, b_add));
    dst_argb[1] = STATIC_CAST(uint8_t, SHADE(g, g_add));
    dst_argb[2] = STATIC_CAST(uint8_t, SHADE(r, r_add));
    dst_argb[3] = STATIC_CAST(uint8_t, SHADE(a, a_add));
    src_argb += 4;
    src_argb1 += 4;
    dst_argb += 4;
  }
}
#undef SHADE

#define SHADE(f, v) clamp0(f - v)

void ARGBSubtractRow_C(const uint8_t* src_argb,
                       const uint8_t* src_argb1,
                       uint8_t* dst_argb,
                       int width) {
  int i;
  for (i = 0; i < width; ++i) {
    const int b = src_argb[0];
    const int g = src_argb[1];
    const int r = src_argb[2];
    const int a = src_argb[3];
    const int b_sub = src_argb1[0];
    const int g_sub = src_argb1[1];
    const int r_sub = src_argb1[2];
    const int a_sub = src_argb1[3];
    dst_argb[0] = STATIC_CAST(uint8_t, SHADE(b, b_sub));
    dst_argb[1] = STATIC_CAST(uint8_t, SHADE(g, g_sub));
    dst_argb[2] = STATIC_CAST(uint8_t, SHADE(r, r_sub));
    dst_argb[3] = STATIC_CAST(uint8_t, SHADE(a, a_sub));
    src_argb += 4;
    src_argb1 += 4;
    dst_argb += 4;
  }
}
#undef SHADE

// Sobel functions which mimics SSSE3.
void SobelXRow_C(const uint8_t* src_y0,
                 const uint8_t* src_y1,
                 const uint8_t* src_y2,
                 uint8_t* dst_sobelx,
                 int width) {
  int i;
  for (i = 0; i < width; ++i) {
    int a = src_y0[i];
    int b = src_y1[i];
    int c = src_y2[i];
    int a_sub = src_y0[i + 2];
    int b_sub = src_y1[i + 2];
    int c_sub = src_y2[i + 2];
    int a_diff = a - a_sub;
    int b_diff = b - b_sub;
    int c_diff = c - c_sub;
    int sobel = Abs(a_diff + b_diff * 2 + c_diff);
    dst_sobelx[i] = (uint8_t)(clamp255(sobel));
  }
}

void SobelYRow_C(const uint8_t* src_y0,
                 const uint8_t* src_y1,
                 uint8_t* dst_sobely,
                 int width) {
  int i;
  for (i = 0; i < width; ++i) {
    int a = src_y0[i + 0];
    int b = src_y0[i + 1];
    int c = src_y0[i + 2];
    int a_sub = src_y1[i + 0];
    int b_sub = src_y1[i + 1];
    int c_sub = src_y1[i + 2];
    int a_diff = a - a_sub;
    int b_diff = b - b_sub;
    int c_diff = c - c_sub;
    int sobel = Abs(a_diff + b_diff * 2 + c_diff);
    dst_sobely[i] = (uint8_t)(clamp255(sobel));
  }
}

void SobelRow_C(const uint8_t* src_sobelx,
                const uint8_t* src_sobely,
                uint8_t* dst_argb,
                int width) {
  int i;
  for (i = 0; i < width; ++i) {
    int r = src_sobelx[i];
    int b = src_sobely[i];
    int s = clamp255(r + b);
    dst_argb[0] = (uint8_t)(s);
    dst_argb[1] = (uint8_t)(s);
    dst_argb[2] = (uint8_t)(s);
    dst_argb[3] = (uint8_t)(255u);
    dst_argb += 4;
  }
}

void SobelToPlaneRow_C(const uint8_t* src_sobelx,
                       const uint8_t* src_sobely,
                       uint8_t* dst_y,
                       int width) {
  int i;
  for (i = 0; i < width; ++i) {
    int r = src_sobelx[i];
    int b = src_sobely[i];
    int s = clamp255(r + b);
    dst_y[i] = (uint8_t)(s);
  }
}

void SobelXYRow_C(const uint8_t* src_sobelx,
                  const uint8_t* src_sobely,
                  uint8_t* dst_argb,
                  int width) {
  int i;
  for (i = 0; i < width; ++i) {
    int r = src_sobelx[i];
    int b = src_sobely[i];
    int g = clamp255(r + b);
    dst_argb[0] = (uint8_t)(b);
    dst_argb[1] = (uint8_t)(g);
    dst_argb[2] = (uint8_t)(r);
    dst_argb[3] = (uint8_t)(255u);
    dst_argb += 4;
  }
}

void J400ToARGBRow_C(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  // Copy a Y to RGB.
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t y = src_y[0];
    dst_argb[2] = dst_argb[1] = dst_argb[0] = y;
    dst_argb[3] = 255u;
    dst_argb += 4;
    ++src_y;
  }
}

// Macros to create SIMD specific yuv to rgb conversion constants.

// clang-format off

#if defined(__aarch64__) || defined(__arm__) || defined(__riscv)
// Bias values include subtract 128 from U and V, bias from Y and rounding.
// For B and R bias is negative. For G bias is positive.
#define YUVCONSTANTSBODY(YG, YB, UB, UG, VG, VR)                             \
  {{UB, VR, UG, VG, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},                     \
   {YG, (UB * 128 - YB), (UG * 128 + VG * 128 + YB), (VR * 128 - YB), YB, 0, \
    0, 0}}
#else
#define YUVCONSTANTSBODY(YG, YB, UB, UG, VG, VR)                     \
  {{UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0,          \
    UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0},         \
   {UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG,  \
    UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG}, \
   {0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR,          \
    0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR},         \
   {YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG}, \
   {YB, YB, YB, YB, YB, YB, YB, YB, YB, YB, YB, YB, YB, YB, YB, YB}}
#endif

// clang-format on

#define MAKEYUVCONSTANTS(name, YG, YB, UB, UG, VG, VR)            \
  const struct YuvConstants SIMD_ALIGNED(kYuv##name##Constants) = \
      YUVCONSTANTSBODY(YG, YB, UB, UG, VG, VR);                   \
  const struct YuvConstants SIMD_ALIGNED(kYvu##name##Constants) = \
      YUVCONSTANTSBODY(YG, YB, VR, VG, UG, UB);

// TODO(fbarchard): Generate SIMD structures from float matrix.

// BT.601 limited range YUV to RGB reference
//  R = (Y - 16) * 1.164             + V * 1.596
//  G = (Y - 16) * 1.164 - U * 0.391 - V * 0.813
//  B = (Y - 16) * 1.164 + U * 2.018
// KR = 0.299; KB = 0.114

// U and V contributions to R,G,B.
#if defined(LIBYUV_UNLIMITED_DATA) || defined(LIBYUV_UNLIMITED_BT601)
#define UB 129 /* round(2.018 * 64) */
#else
#define UB 128 /* max(128, round(2.018 * 64)) */
#endif
#define UG 25  /* round(0.391 * 64) */
#define VG 52  /* round(0.813 * 64) */
#define VR 102 /* round(1.596 * 64) */

// Y contribution to R,G,B.  Scale and bias.
#define YG 18997 /* round(1.164 * 64 * 256 * 256 / 257) */
#define YB -1160 /* 1.164 * 64 * -16 + 64 / 2 */

MAKEYUVCONSTANTS(I601, YG, YB, UB, UG, VG, VR)

#undef YG
#undef YB
#undef UB
#undef UG
#undef VG
#undef VR

// BT.601 full range YUV to RGB reference (aka JPEG)
// *  R = Y               + V * 1.40200
// *  G = Y - U * 0.34414 - V * 0.71414
// *  B = Y + U * 1.77200
// KR = 0.299; KB = 0.114

// U and V contributions to R,G,B.
#define UB 113 /* round(1.77200 * 64) */
#define UG 22  /* round(0.34414 * 64) */
#define VG 46  /* round(0.71414 * 64) */
#define VR 90  /* round(1.40200 * 64) */

// Y contribution to R,G,B.  Scale and bias.
#define YG 16320 /* round(1.000 * 64 * 256 * 256 / 257) */
#define YB 32    /* 64 / 2 */

MAKEYUVCONSTANTS(JPEG, YG, YB, UB, UG, VG, VR)

#undef YG
#undef YB
#undef UB
#undef UG
#undef VG
#undef VR

// BT.709 limited range YUV to RGB reference
//  R = (Y - 16) * 1.164             + V * 1.793
//  G = (Y - 16) * 1.164 - U * 0.213 - V * 0.533
//  B = (Y - 16) * 1.164 + U * 2.112
//  KR = 0.2126, KB = 0.0722

// U and V contributions to R,G,B.
#if defined(LIBYUV_UNLIMITED_DATA) || defined(LIBYUV_UNLIMITED_BT709)
#define UB 135 /* round(2.112 * 64) */
#else
#define UB 128 /* max(128, round(2.112 * 64)) */
#endif
#define UG 14  /* round(0.213 * 64) */
#define VG 34  /* round(0.533 * 64) */
#define VR 115 /* round(1.793 * 64) */

// Y contribution to R,G,B.  Scale and bias.
#define YG 18997 /* round(1.164 * 64 * 256 * 256 / 257) */
#define YB -1160 /* 1.164 * 64 * -16 + 64 / 2 */

MAKEYUVCONSTANTS(H709, YG, YB, UB, UG, VG, VR)

#undef YG
#undef YB
#undef UB
#undef UG
#undef VG
#undef VR

// BT.709 full range YUV to RGB reference
//  R = Y               + V * 1.5748
//  G = Y - U * 0.18732 - V * 0.46812
//  B = Y + U * 1.8556
//  KR = 0.2126, KB = 0.0722

// U and V contributions to R,G,B.
#define UB 119 /* round(1.8556 * 64) */
#define UG 12  /* round(0.18732 * 64) */
#define VG 30  /* round(0.46812 * 64) */
#define VR 101 /* round(1.5748 * 64) */

// Y contribution to R,G,B.  Scale and bias.  (same as jpeg)
#define YG 16320 /* round(1 * 64 * 256 * 256 / 257) */
#define YB 32    /* 64 / 2 */

MAKEYUVCONSTANTS(F709, YG, YB, UB, UG, VG, VR)

#undef YG
#undef YB
#undef UB
#undef UG
#undef VG
#undef VR

// BT.2020 limited range YUV to RGB reference
//  R = (Y - 16) * 1.164384                + V * 1.67867
//  G = (Y - 16) * 1.164384 - U * 0.187326 - V * 0.65042
//  B = (Y - 16) * 1.164384 + U * 2.14177
// KR = 0.2627; KB = 0.0593

// U and V contributions to R,G,B.
#if defined(LIBYUV_UNLIMITED_DATA) || defined(LIBYUV_UNLIMITED_BT2020)
#define UB 137 /* round(2.142 * 64) */
#else
#define UB 128 /* max(128, round(2.142 * 64)) */
#endif
#define UG 12  /* round(0.187326 * 64) */
#define VG 42  /* round(0.65042 * 64) */
#define VR 107 /* round(1.67867 * 64) */

// Y contribution to R,G,B.  Scale and bias.
#define YG 19003 /* round(1.164384 * 64 * 256 * 256 / 257) */
#define YB -1160 /* 1.164384 * 64 * -16 + 64 / 2 */

MAKEYUVCONSTANTS(2020, YG, YB, UB, UG, VG, VR)

#undef YG
#undef YB
#undef UB
#undef UG
#undef VG
#undef VR

// BT.2020 full range YUV to RGB reference
//  R = Y                + V * 1.474600
//  G = Y - U * 0.164553 - V * 0.571353
//  B = Y + U * 1.881400
// KR = 0.2627; KB = 0.0593

#define UB 120 /* round(1.881400 * 64) */
#define UG 11  /* round(0.164553 * 64) */
#define VG 37  /* round(0.571353 * 64) */
#define VR 94  /* round(1.474600 * 64) */

// Y contribution to R,G,B.  Scale and bias.  (same as jpeg)
#define YG 16320 /* round(1 * 64 * 256 * 256 / 257) */
#define YB 32    /* 64 / 2 */

MAKEYUVCONSTANTS(V2020, YG, YB, UB, UG, VG, VR)

#undef YG
#undef YB
#undef UB
#undef UG
#undef VG
#undef VR

#undef BB
#undef BG
#undef BR

#undef MAKEYUVCONSTANTS

#if defined(__aarch64__) || defined(__arm__) || defined(__riscv)
#define LOAD_YUV_CONSTANTS                 \
  int ub = yuvconstants->kUVCoeff[0];      \
  int vr = yuvconstants->kUVCoeff[1];      \
  int ug = yuvconstants->kUVCoeff[2];      \
  int vg = yuvconstants->kUVCoeff[3];      \
  int yg = yuvconstants->kRGBCoeffBias[0]; \
  int bb = yuvconstants->kRGBCoeffBias[1]; \
  int bg = yuvconstants->kRGBCoeffBias[2]; \
  int br = yuvconstants->kRGBCoeffBias[3]

#define CALC_RGB16                         \
  int32_t y1 = (uint32_t)(y32 * yg) >> 16; \
  int b16 = y1 + (u * ub) - bb;            \
  int g16 = y1 + bg - (u * ug + v * vg);   \
  int r16 = y1 + (v * vr) - br
#else
#define LOAD_YUV_CONSTANTS           \
  int ub = yuvconstants->kUVToB[0];  \
  int ug = yuvconstants->kUVToG[0];  \
  int vg = yuvconstants->kUVToG[1];  \
  int vr = yuvconstants->kUVToR[1];  \
  int yg = yuvconstants->kYToRgb[0]; \
  int yb = yuvconstants->kYBiasToRgb[0]

#define CALC_RGB16                                \
  int32_t y1 = ((uint32_t)(y32 * yg) >> 16) + yb; \
  int8_t ui = (int8_t)u;                          \
  int8_t vi = (int8_t)v;                          \
  ui -= 0x80;                                     \
  vi -= 0x80;                                     \
  int b16 = y1 + (ui * ub);                       \
  int g16 = y1 - (ui * ug + vi * vg);             \
  int r16 = y1 + (vi * vr)
#endif

// C reference code that mimics the YUV assembly.
// Reads 8 bit YUV and leaves result as 16 bit.
static __inline void YuvPixel(uint8_t y,
                              uint8_t u,
                              uint8_t v,
                              uint8_t* b,
                              uint8_t* g,
                              uint8_t* r,
                              const struct YuvConstants* yuvconstants) {
  LOAD_YUV_CONSTANTS;
  uint32_t y32 = y * 0x0101;
  CALC_RGB16;
  *b = STATIC_CAST(uint8_t, Clamp((int32_t)(b16) >> 6));
  *g = STATIC_CAST(uint8_t, Clamp((int32_t)(g16) >> 6));
  *r = STATIC_CAST(uint8_t, Clamp((int32_t)(r16) >> 6));
}

// Reads 8 bit YUV and leaves result as 16 bit.
static __inline void YuvPixel8_16(uint8_t y,
                                  uint8_t u,
                                  uint8_t v,
                                  int* b,
                                  int* g,
                                  int* r,
                                  const struct YuvConstants* yuvconstants) {
  LOAD_YUV_CONSTANTS;
  uint32_t y32 = y * 0x0101;
  CALC_RGB16;
  *b = b16;
  *g = g16;
  *r = r16;
}

// C reference code that mimics the YUV 16 bit assembly.
// Reads 10 bit YUV and leaves result as 16 bit.
static __inline void YuvPixel10_16(uint16_t y,
                                   uint16_t u,
                                   uint16_t v,
                                   int* b,
                                   int* g,
                                   int* r,
                                   const struct YuvConstants* yuvconstants) {
  LOAD_YUV_CONSTANTS;
  uint32_t y32 = (y << 6) | (y >> 4);
  u = STATIC_CAST(uint8_t, clamp255(u >> 2));
  v = STATIC_CAST(uint8_t, clamp255(v >> 2));
  CALC_RGB16;
  *b = b16;
  *g = g16;
  *r = r16;
}

// C reference code that mimics the YUV 16 bit assembly.
// Reads 12 bit YUV and leaves result as 16 bit.
static __inline void YuvPixel12_16(int16_t y,
                                   int16_t u,
                                   int16_t v,
                                   int* b,
                                   int* g,
                                   int* r,
                                   const struct YuvConstants* yuvconstants) {
  LOAD_YUV_CONSTANTS;
  uint32_t y32 = (y << 4) | (y >> 8);
  u = STATIC_CAST(uint8_t, clamp255(u >> 4));
  v = STATIC_CAST(uint8_t, clamp255(v >> 4));
  CALC_RGB16;
  *b = b16;
  *g = g16;
  *r = r16;
}

// C reference code that mimics the YUV 10 bit assembly.
// Reads 10 bit YUV and clamps down to 8 bit RGB.
static __inline void YuvPixel10(uint16_t y,
                                uint16_t u,
                                uint16_t v,
                                uint8_t* b,
                                uint8_t* g,
                                uint8_t* r,
                                const struct YuvConstants* yuvconstants) {
  int b16;
  int g16;
  int r16;
  YuvPixel10_16(y, u, v, &b16, &g16, &r16, yuvconstants);
  *b = STATIC_CAST(uint8_t, Clamp(b16 >> 6));
  *g = STATIC_CAST(uint8_t, Clamp(g16 >> 6));
  *r = STATIC_CAST(uint8_t, Clamp(r16 >> 6));
}

// C reference code that mimics the YUV 12 bit assembly.
// Reads 12 bit YUV and clamps down to 8 bit RGB.
static __inline void YuvPixel12(uint16_t y,
                                uint16_t u,
                                uint16_t v,
                                uint8_t* b,
                                uint8_t* g,
                                uint8_t* r,
                                const struct YuvConstants* yuvconstants) {
  int b16;
  int g16;
  int r16;
  YuvPixel12_16(y, u, v, &b16, &g16, &r16, yuvconstants);
  *b = STATIC_CAST(uint8_t, Clamp(b16 >> 6));
  *g = STATIC_CAST(uint8_t, Clamp(g16 >> 6));
  *r = STATIC_CAST(uint8_t, Clamp(r16 >> 6));
}

// C reference code that mimics the YUV 16 bit assembly.
// Reads 16 bit YUV and leaves result as 8 bit.
static __inline void YuvPixel16_8(uint16_t y,
                                  uint16_t u,
                                  uint16_t v,
                                  uint8_t* b,
                                  uint8_t* g,
                                  uint8_t* r,
                                  const struct YuvConstants* yuvconstants) {
  LOAD_YUV_CONSTANTS;
  uint32_t y32 = y;
  u = STATIC_CAST(uint16_t, clamp255(u >> 8));
  v = STATIC_CAST(uint16_t, clamp255(v >> 8));
  CALC_RGB16;
  *b = STATIC_CAST(uint8_t, Clamp((int32_t)(b16) >> 6));
  *g = STATIC_CAST(uint8_t, Clamp((int32_t)(g16) >> 6));
  *r = STATIC_CAST(uint8_t, Clamp((int32_t)(r16) >> 6));
}

// C reference code that mimics the YUV 16 bit assembly.
// Reads 16 bit YUV and leaves result as 16 bit.
static __inline void YuvPixel16_16(uint16_t y,
                                   uint16_t u,
                                   uint16_t v,
                                   int* b,
                                   int* g,
                                   int* r,
                                   const struct YuvConstants* yuvconstants) {
  LOAD_YUV_CONSTANTS;
  uint32_t y32 = y;
  u = STATIC_CAST(uint16_t, clamp255(u >> 8));
  v = STATIC_CAST(uint16_t, clamp255(v >> 8));
  CALC_RGB16;
  *b = b16;
  *g = g16;
  *r = r16;
}

// C reference code that mimics the YUV assembly.
// Reads 8 bit YUV and leaves result as 8 bit.
static __inline void YPixel(uint8_t y,
                            uint8_t* b,
                            uint8_t* g,
                            uint8_t* r,
                            const struct YuvConstants* yuvconstants) {
#if defined(__aarch64__) || defined(__arm__) || defined(__riscv)
  int yg = yuvconstants->kRGBCoeffBias[0];
  int ygb = yuvconstants->kRGBCoeffBias[4];
#else
  int ygb = yuvconstants->kYBiasToRgb[0];
  int yg = yuvconstants->kYToRgb[0];
#endif
  uint32_t y1 = (uint32_t)(y * 0x0101 * yg) >> 16;
  uint8_t b8 = STATIC_CAST(uint8_t, Clamp(((int32_t)(y1) + ygb) >> 6));
  *b = b8;
  *g = b8;
  *r = b8;
}

void I444ToARGBRow_C(const uint8_t* src_y,
                     const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width; ++x) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    src_y += 1;
    src_u += 1;
    src_v += 1;
    rgb_buf += 4;  // Advance 1 pixel.
  }
}

void I444ToRGB24Row_C(const uint8_t* src_y,
                      const uint8_t* src_u,
                      const uint8_t* src_v,
                      uint8_t* rgb_buf,
                      const struct YuvConstants* yuvconstants,
                      int width) {
  int x;
  for (x = 0; x < width; ++x) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    src_y += 1;
    src_u += 1;
    src_v += 1;
    rgb_buf += 3;  // Advance 1 pixel.
  }
}

// Also used for 420
void I422ToARGBRow_C(const uint8_t* src_y,
                     const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_y[1], src_u[0], src_v[0], rgb_buf + 4, rgb_buf + 5,
             rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

// 10 bit YUV to ARGB
void I210ToARGBRow_C(const uint16_t* src_y,
                     const uint16_t* src_u,
                     const uint16_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel10(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
               rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel10(src_y[1], src_u[0], src_v[0], rgb_buf + 4, rgb_buf + 5,
               rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel10(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
               rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void I410ToARGBRow_C(const uint16_t* src_y,
                     const uint16_t* src_u,
                     const uint16_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width; ++x) {
    YuvPixel10(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
               rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    src_y += 1;
    src_u += 1;
    src_v += 1;
    rgb_buf += 4;  // Advance 1 pixels.
  }
}

void I210AlphaToARGBRow_C(const uint16_t* src_y,
                          const uint16_t* src_u,
                          const uint16_t* src_v,
                          const uint16_t* src_a,
                          uint8_t* rgb_buf,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel10(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
               rgb_buf + 2, yuvconstants);
    rgb_buf[3] = STATIC_CAST(uint8_t, clamp255(src_a[0] >> 2));
    YuvPixel10(src_y[1], src_u[0], src_v[0], rgb_buf + 4, rgb_buf + 5,
               rgb_buf + 6, yuvconstants);
    rgb_buf[7] = STATIC_CAST(uint8_t, clamp255(src_a[1] >> 2));
    src_y += 2;
    src_u += 1;
    src_v += 1;
    src_a += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel10(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
               rgb_buf + 2, yuvconstants);
    rgb_buf[3] = STATIC_CAST(uint8_t, clamp255(src_a[0] >> 2));
  }
}

void I410AlphaToARGBRow_C(const uint16_t* src_y,
                          const uint16_t* src_u,
                          const uint16_t* src_v,
                          const uint16_t* src_a,
                          uint8_t* rgb_buf,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  int x;
  for (x = 0; x < width; ++x) {
    YuvPixel10(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
               rgb_buf + 2, yuvconstants);
    rgb_buf[3] = STATIC_CAST(uint8_t, clamp255(src_a[0] >> 2));
    src_y += 1;
    src_u += 1;
    src_v += 1;
    src_a += 1;
    rgb_buf += 4;  // Advance 1 pixels.
  }
}

// 12 bit YUV to ARGB
void I212ToARGBRow_C(const uint16_t* src_y,
                     const uint16_t* src_u,
                     const uint16_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel12(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
               rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel12(src_y[1], src_u[0], src_v[0], rgb_buf + 4, rgb_buf + 5,
               rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel12(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
               rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

static void StoreAR30(uint8_t* rgb_buf, int b, int g, int r) {
  uint32_t ar30;
  b = b >> 4;  // convert 8 bit 10.6 to 10 bit.
  g = g >> 4;
  r = r >> 4;
  b = Clamp10(b);
  g = Clamp10(g);
  r = Clamp10(r);
  ar30 = b | ((uint32_t)g << 10) | ((uint32_t)r << 20) | 0xc0000000;
  (*(uint32_t*)rgb_buf) = ar30;
}

// 10 bit YUV to 10 bit AR30
void I210ToAR30Row_C(const uint16_t* src_y,
                     const uint16_t* src_u,
                     const uint16_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  int b;
  int g;
  int r;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel10_16(src_y[0], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf, b, g, r);
    YuvPixel10_16(src_y[1], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf + 4, b, g, r);
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel10_16(src_y[0], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf, b, g, r);
  }
}

// 12 bit YUV to 10 bit AR30
void I212ToAR30Row_C(const uint16_t* src_y,
                     const uint16_t* src_u,
                     const uint16_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  int b;
  int g;
  int r;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel12_16(src_y[0], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf, b, g, r);
    YuvPixel12_16(src_y[1], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf + 4, b, g, r);
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel12_16(src_y[0], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf, b, g, r);
  }
}

void I410ToAR30Row_C(const uint16_t* src_y,
                     const uint16_t* src_u,
                     const uint16_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  int b;
  int g;
  int r;
  for (x = 0; x < width; ++x) {
    YuvPixel10_16(src_y[0], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf, b, g, r);
    src_y += 1;
    src_u += 1;
    src_v += 1;
    rgb_buf += 4;  // Advance 1 pixel.
  }
}

// P210 has 10 bits in msb of 16 bit NV12 style layout.
void P210ToARGBRow_C(const uint16_t* src_y,
                     const uint16_t* src_uv,
                     uint8_t* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel16_8(src_y[0], src_uv[0], src_uv[1], dst_argb + 0, dst_argb + 1,
                 dst_argb + 2, yuvconstants);
    dst_argb[3] = 255;
    YuvPixel16_8(src_y[1], src_uv[0], src_uv[1], dst_argb + 4, dst_argb + 5,
                 dst_argb + 6, yuvconstants);
    dst_argb[7] = 255;
    src_y += 2;
    src_uv += 2;
    dst_argb += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel16_8(src_y[0], src_uv[0], src_uv[1], dst_argb + 0, dst_argb + 1,
                 dst_argb + 2, yuvconstants);
    dst_argb[3] = 255;
  }
}

void P410ToARGBRow_C(const uint16_t* src_y,
                     const uint16_t* src_uv,
                     uint8_t* dst_argb,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width; ++x) {
    YuvPixel16_8(src_y[0], src_uv[0], src_uv[1], dst_argb + 0, dst_argb + 1,
                 dst_argb + 2, yuvconstants);
    dst_argb[3] = 255;
    src_y += 1;
    src_uv += 2;
    dst_argb += 4;  // Advance 1 pixels.
  }
}

void P210ToAR30Row_C(const uint16_t* src_y,
                     const uint16_t* src_uv,
                     uint8_t* dst_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  int b;
  int g;
  int r;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel16_16(src_y[0], src_uv[0], src_uv[1], &b, &g, &r, yuvconstants);
    StoreAR30(dst_ar30, b, g, r);
    YuvPixel16_16(src_y[1], src_uv[0], src_uv[1], &b, &g, &r, yuvconstants);
    StoreAR30(dst_ar30 + 4, b, g, r);
    src_y += 2;
    src_uv += 2;
    dst_ar30 += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel16_16(src_y[0], src_uv[0], src_uv[1], &b, &g, &r, yuvconstants);
    StoreAR30(dst_ar30, b, g, r);
  }
}

void P410ToAR30Row_C(const uint16_t* src_y,
                     const uint16_t* src_uv,
                     uint8_t* dst_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  int b;
  int g;
  int r;
  for (x = 0; x < width; ++x) {
    YuvPixel16_16(src_y[0], src_uv[0], src_uv[1], &b, &g, &r, yuvconstants);
    StoreAR30(dst_ar30, b, g, r);
    src_y += 1;
    src_uv += 2;
    dst_ar30 += 4;  // Advance 1 pixel.
  }
}

// 8 bit YUV to 10 bit AR30
// Uses same code as 10 bit YUV bit shifts the 8 bit values up to 10 bits.
void I422ToAR30Row_C(const uint8_t* src_y,
                     const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  int b;
  int g;
  int r;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel8_16(src_y[0], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf, b, g, r);
    YuvPixel8_16(src_y[1], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf + 4, b, g, r);
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel8_16(src_y[0], src_u[0], src_v[0], &b, &g, &r, yuvconstants);
    StoreAR30(rgb_buf, b, g, r);
  }
}

void I444AlphaToARGBRow_C(const uint8_t* src_y,
                          const uint8_t* src_u,
                          const uint8_t* src_v,
                          const uint8_t* src_a,
                          uint8_t* rgb_buf,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  int x;
  for (x = 0; x < width; ++x) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = src_a[0];
    src_y += 1;
    src_u += 1;
    src_v += 1;
    src_a += 1;
    rgb_buf += 4;  // Advance 1 pixel.
  }
}

void I422AlphaToARGBRow_C(const uint8_t* src_y,
                          const uint8_t* src_u,
                          const uint8_t* src_v,
                          const uint8_t* src_a,
                          uint8_t* rgb_buf,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = src_a[0];
    YuvPixel(src_y[1], src_u[0], src_v[0], rgb_buf + 4, rgb_buf + 5,
             rgb_buf + 6, yuvconstants);
    rgb_buf[7] = src_a[1];
    src_y += 2;
    src_u += 1;
    src_v += 1;
    src_a += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = src_a[0];
  }
}

void I422ToRGB24Row_C(const uint8_t* src_y,
                      const uint8_t* src_u,
                      const uint8_t* src_v,
                      uint8_t* rgb_buf,
                      const struct YuvConstants* yuvconstants,
                      int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    YuvPixel(src_y[1], src_u[0], src_v[0], rgb_buf + 3, rgb_buf + 4,
             rgb_buf + 5, yuvconstants);
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 6;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
  }
}

void I422ToARGB4444Row_C(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_argb4444,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  uint8_t b0;
  uint8_t g0;
  uint8_t r0;
  uint8_t b1;
  uint8_t g1;
  uint8_t r1;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0], &b0, &g0, &r0, yuvconstants);
    YuvPixel(src_y[1], src_u[0], src_v[0], &b1, &g1, &r1, yuvconstants);
    b0 = b0 >> 4;
    g0 = g0 >> 4;
    r0 = r0 >> 4;
    b1 = b1 >> 4;
    g1 = g1 >> 4;
    r1 = r1 >> 4;
    *(uint16_t*)(dst_argb4444 + 0) =
        STATIC_CAST(uint16_t, b0 | (g0 << 4) | (r0 << 8) | 0xf000);
    *(uint16_t*)(dst_argb4444 + 2) =
        STATIC_CAST(uint16_t, b1 | (g1 << 4) | (r1 << 8) | 0xf000);
    src_y += 2;
    src_u += 1;
    src_v += 1;
    dst_argb4444 += 4;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0], &b0, &g0, &r0, yuvconstants);
    b0 = b0 >> 4;
    g0 = g0 >> 4;
    r0 = r0 >> 4;
    *(uint16_t*)(dst_argb4444) =
        STATIC_CAST(uint16_t, b0 | (g0 << 4) | (r0 << 8) | 0xf000);
  }
}

void I422ToARGB1555Row_C(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_argb1555,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  uint8_t b0;
  uint8_t g0;
  uint8_t r0;
  uint8_t b1;
  uint8_t g1;
  uint8_t r1;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0], &b0, &g0, &r0, yuvconstants);
    YuvPixel(src_y[1], src_u[0], src_v[0], &b1, &g1, &r1, yuvconstants);
    b0 = b0 >> 3;
    g0 = g0 >> 3;
    r0 = r0 >> 3;
    b1 = b1 >> 3;
    g1 = g1 >> 3;
    r1 = r1 >> 3;
    *(uint16_t*)(dst_argb1555 + 0) =
        STATIC_CAST(uint16_t, b0 | (g0 << 5) | (r0 << 10) | 0x8000);
    *(uint16_t*)(dst_argb1555 + 2) =
        STATIC_CAST(uint16_t, b1 | (g1 << 5) | (r1 << 10) | 0x8000);
    src_y += 2;
    src_u += 1;
    src_v += 1;
    dst_argb1555 += 4;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0], &b0, &g0, &r0, yuvconstants);
    b0 = b0 >> 3;
    g0 = g0 >> 3;
    r0 = r0 >> 3;
    *(uint16_t*)(dst_argb1555) =
        STATIC_CAST(uint16_t, b0 | (g0 << 5) | (r0 << 10) | 0x8000);
  }
}

void I422ToRGB565Row_C(const uint8_t* src_y,
                       const uint8_t* src_u,
                       const uint8_t* src_v,
                       uint8_t* dst_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  uint8_t b0;
  uint8_t g0;
  uint8_t r0;
  uint8_t b1;
  uint8_t g1;
  uint8_t r1;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0], &b0, &g0, &r0, yuvconstants);
    YuvPixel(src_y[1], src_u[0], src_v[0], &b1, &g1, &r1, yuvconstants);
    b0 = b0 >> 3;
    g0 = g0 >> 2;
    r0 = r0 >> 3;
    b1 = b1 >> 3;
    g1 = g1 >> 2;
    r1 = r1 >> 3;
    *(uint16_t*)(dst_rgb565 + 0) =
        STATIC_CAST(uint16_t, b0 | (g0 << 5) | (r0 << 11));
    *(uint16_t*)(dst_rgb565 + 2) =
        STATIC_CAST(uint16_t, b1 | (g1 << 5) | (r1 << 11));
    src_y += 2;
    src_u += 1;
    src_v += 1;
    dst_rgb565 += 4;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0], &b0, &g0, &r0, yuvconstants);
    b0 = b0 >> 3;
    g0 = g0 >> 2;
    r0 = r0 >> 3;
    *(uint16_t*)(dst_rgb565 + 0) =
        STATIC_CAST(uint16_t, b0 | (g0 << 5) | (r0 << 11));
  }
}

void NV12ToARGBRow_C(const uint8_t* src_y,
                     const uint8_t* src_uv,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_uv[0], src_uv[1], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_y[1], src_uv[0], src_uv[1], rgb_buf + 4, rgb_buf + 5,
             rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    src_uv += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_uv[0], src_uv[1], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void NV21ToARGBRow_C(const uint8_t* src_y,
                     const uint8_t* src_vu,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_vu[1], src_vu[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_y[1], src_vu[1], src_vu[0], rgb_buf + 4, rgb_buf + 5,
             rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    src_vu += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_vu[1], src_vu[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void NV12ToRGB24Row_C(const uint8_t* src_y,
                      const uint8_t* src_uv,
                      uint8_t* rgb_buf,
                      const struct YuvConstants* yuvconstants,
                      int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_uv[0], src_uv[1], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    YuvPixel(src_y[1], src_uv[0], src_uv[1], rgb_buf + 3, rgb_buf + 4,
             rgb_buf + 5, yuvconstants);
    src_y += 2;
    src_uv += 2;
    rgb_buf += 6;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_uv[0], src_uv[1], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
  }
}

void NV21ToRGB24Row_C(const uint8_t* src_y,
                      const uint8_t* src_vu,
                      uint8_t* rgb_buf,
                      const struct YuvConstants* yuvconstants,
                      int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_vu[1], src_vu[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    YuvPixel(src_y[1], src_vu[1], src_vu[0], rgb_buf + 3, rgb_buf + 4,
             rgb_buf + 5, yuvconstants);
    src_y += 2;
    src_vu += 2;
    rgb_buf += 6;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_vu[1], src_vu[0], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
  }
}

void NV12ToRGB565Row_C(const uint8_t* src_y,
                       const uint8_t* src_uv,
                       uint8_t* dst_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  uint8_t b0;
  uint8_t g0;
  uint8_t r0;
  uint8_t b1;
  uint8_t g1;
  uint8_t r1;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_uv[0], src_uv[1], &b0, &g0, &r0, yuvconstants);
    YuvPixel(src_y[1], src_uv[0], src_uv[1], &b1, &g1, &r1, yuvconstants);
    b0 = b0 >> 3;
    g0 = g0 >> 2;
    r0 = r0 >> 3;
    b1 = b1 >> 3;
    g1 = g1 >> 2;
    r1 = r1 >> 3;
    *(uint16_t*)(dst_rgb565 + 0) = STATIC_CAST(uint16_t, b0) |
                                   STATIC_CAST(uint16_t, g0 << 5) |
                                   STATIC_CAST(uint16_t, r0 << 11);
    *(uint16_t*)(dst_rgb565 + 2) = STATIC_CAST(uint16_t, b1) |
                                   STATIC_CAST(uint16_t, g1 << 5) |
                                   STATIC_CAST(uint16_t, r1 << 11);
    src_y += 2;
    src_uv += 2;
    dst_rgb565 += 4;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_uv[0], src_uv[1], &b0, &g0, &r0, yuvconstants);
    b0 = b0 >> 3;
    g0 = g0 >> 2;
    r0 = r0 >> 3;
    *(uint16_t*)(dst_rgb565) = STATIC_CAST(uint16_t, b0) |
                               STATIC_CAST(uint16_t, g0 << 5) |
                               STATIC_CAST(uint16_t, r0 << 11);
  }
}

void YUY2ToARGBRow_C(const uint8_t* src_yuy2,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_yuy2[0], src_yuy2[1], src_yuy2[3], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_yuy2[2], src_yuy2[1], src_yuy2[3], rgb_buf + 4, rgb_buf + 5,
             rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_yuy2 += 4;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_yuy2[0], src_yuy2[1], src_yuy2[3], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void UYVYToARGBRow_C(const uint8_t* src_uyvy,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_uyvy[1], src_uyvy[0], src_uyvy[2], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_uyvy[3], src_uyvy[0], src_uyvy[2], rgb_buf + 4, rgb_buf + 5,
             rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_uyvy += 4;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_uyvy[1], src_uyvy[0], src_uyvy[2], rgb_buf + 0, rgb_buf + 1,
             rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void I422ToRGBARow_C(const uint8_t* src_y,
                     const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 1, rgb_buf + 2,
             rgb_buf + 3, yuvconstants);
    rgb_buf[0] = 255;
    YuvPixel(src_y[1], src_u[0], src_v[0], rgb_buf + 5, rgb_buf + 6,
             rgb_buf + 7, yuvconstants);
    rgb_buf[4] = 255;
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0], rgb_buf + 1, rgb_buf + 2,
             rgb_buf + 3, yuvconstants);
    rgb_buf[0] = 255;
  }
}

void I400ToARGBRow_C(const uint8_t* src_y,
                     uint8_t* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YPixel(src_y[0], rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YPixel(src_y[1], rgb_buf + 4, rgb_buf + 5, rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YPixel(src_y[0], rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void MirrorRow_C(const uint8_t* src, uint8_t* dst, int width) {
  int x;
  src += width - 1;
  for (x = 0; x < width - 1; x += 2) {
    dst[x] = src[0];
    dst[x + 1] = src[-1];
    src -= 2;
  }
  if (width & 1) {
    dst[width - 1] = src[0];
  }
}

void MirrorRow_16_C(const uint16_t* src, uint16_t* dst, int width) {
  int x;
  src += width - 1;
  for (x = 0; x < width - 1; x += 2) {
    dst[x] = src[0];
    dst[x + 1] = src[-1];
    src -= 2;
  }
  if (width & 1) {
    dst[width - 1] = src[0];
  }
}

void MirrorUVRow_C(const uint8_t* src_uv, uint8_t* dst_uv, int width) {
  int x;
  src_uv += (width - 1) << 1;
  for (x = 0; x < width; ++x) {
    dst_uv[0] = src_uv[0];
    dst_uv[1] = src_uv[1];
    src_uv -= 2;
    dst_uv += 2;
  }
}

void MirrorSplitUVRow_C(const uint8_t* src_uv,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  int x;
  src_uv += (width - 1) << 1;
  for (x = 0; x < width - 1; x += 2) {
    dst_u[x] = src_uv[0];
    dst_u[x + 1] = src_uv[-2];
    dst_v[x] = src_uv[1];
    dst_v[x + 1] = src_uv[-2 + 1];
    src_uv -= 4;
  }
  if (width & 1) {
    dst_u[width - 1] = src_uv[0];
    dst_v[width - 1] = src_uv[1];
  }
}

void ARGBMirrorRow_C(const uint8_t* src, uint8_t* dst, int width) {
  int x;
  const uint32_t* src32 = (const uint32_t*)(src);
  uint32_t* dst32 = (uint32_t*)(dst);
  src32 += width - 1;
  for (x = 0; x < width - 1; x += 2) {
    dst32[x] = src32[0];
    dst32[x + 1] = src32[-1];
    src32 -= 2;
  }
  if (width & 1) {
    dst32[width - 1] = src32[0];
  }
}

void RGB24MirrorRow_C(const uint8_t* src_rgb24, uint8_t* dst_rgb24, int width) {
  int x;
  src_rgb24 += width * 3 - 3;
  for (x = 0; x < width; ++x) {
    uint8_t b = src_rgb24[0];
    uint8_t g = src_rgb24[1];
    uint8_t r = src_rgb24[2];
    dst_rgb24[0] = b;
    dst_rgb24[1] = g;
    dst_rgb24[2] = r;
    src_rgb24 -= 3;
    dst_rgb24 += 3;
  }
}

void SplitUVRow_C(const uint8_t* src_uv,
                  uint8_t* dst_u,
                  uint8_t* dst_v,
                  int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_u[x] = src_uv[0];
    dst_u[x + 1] = src_uv[2];
    dst_v[x] = src_uv[1];
    dst_v[x + 1] = src_uv[3];
    src_uv += 4;
  }
  if (width & 1) {
    dst_u[width - 1] = src_uv[0];
    dst_v[width - 1] = src_uv[1];
  }
}

void MergeUVRow_C(const uint8_t* src_u,
                  const uint8_t* src_v,
                  uint8_t* dst_uv,
                  int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_uv[0] = src_u[x];
    dst_uv[1] = src_v[x];
    dst_uv[2] = src_u[x + 1];
    dst_uv[3] = src_v[x + 1];
    dst_uv += 4;
  }
  if (width & 1) {
    dst_uv[0] = src_u[width - 1];
    dst_uv[1] = src_v[width - 1];
  }
}

void DetileRow_C(const uint8_t* src,
                 ptrdiff_t src_tile_stride,
                 uint8_t* dst,
                 int width) {
  int x;
  for (x = 0; x < width - 15; x += 16) {
    memcpy(dst, src, 16);
    dst += 16;
    src += src_tile_stride;
  }
  if (width & 15) {
    memcpy(dst, src, width & 15);
  }
}

void DetileRow_16_C(const uint16_t* src,
                    ptrdiff_t src_tile_stride,
                    uint16_t* dst,
                    int width) {
  int x;
  for (x = 0; x < width - 15; x += 16) {
    memcpy(dst, src, 16 * sizeof(uint16_t));
    dst += 16;
    src += src_tile_stride;
  }
  if (width & 15) {
    memcpy(dst, src, (width & 15) * sizeof(uint16_t));
  }
}

void DetileSplitUVRow_C(const uint8_t* src_uv,
                        ptrdiff_t src_tile_stride,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  int x;
  for (x = 0; x < width - 15; x += 16) {
    SplitUVRow_C(src_uv, dst_u, dst_v, 8);
    dst_u += 8;
    dst_v += 8;
    src_uv += src_tile_stride;
  }
  if (width & 15) {
    SplitUVRow_C(src_uv, dst_u, dst_v, ((width & 15) + 1) / 2);
  }
}

void DetileToYUY2_C(const uint8_t* src_y,
                    ptrdiff_t src_y_tile_stride,
                    const uint8_t* src_uv,
                    ptrdiff_t src_uv_tile_stride,
                    uint8_t* dst_yuy2,
                    int width) {
  for (int x = 0; x < width - 15; x += 16) {
    for (int i = 0; i < 8; i++) {
      dst_yuy2[0] = src_y[0];
      dst_yuy2[1] = src_uv[0];
      dst_yuy2[2] = src_y[1];
      dst_yuy2[3] = src_uv[1];
      dst_yuy2 += 4;
      src_y += 2;
      src_uv += 2;
    }
    src_y += src_y_tile_stride - 16;
    src_uv += src_uv_tile_stride - 16;
  }
}

// Unpack MT2T into tiled P010 64 pixels at a time. MT2T's bitstream is encoded
// in 80 byte blocks representing 64 pixels each. The first 16 bytes of the
// block contain all of the lower 2 bits of each pixel packed together, and the
// next 64 bytes represent all the upper 8 bits of the pixel. The lower bits are
// packed into 1x4 blocks, whereas the upper bits are packed in normal raster
// order.
void UnpackMT2T_C(const uint8_t* src, uint16_t* dst, size_t size) {
  for (size_t i = 0; i < size; i += 80) {
    const uint8_t* src_lower_bits = src;
    const uint8_t* src_upper_bits = src + 16;

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 16; k++) {
        *dst++ = ((src_lower_bits[k] >> (j * 2)) & 0x3) << 6 |
                 (uint16_t)*src_upper_bits << 8 |
                 (uint16_t)*src_upper_bits >> 2;
        src_upper_bits++;
      }
    }

    src += 80;
  }
}

void SplitRGBRow_C(const uint8_t* src_rgb,
                   uint8_t* dst_r,
                   uint8_t* dst_g,
                   uint8_t* dst_b,
                   int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_r[x] = src_rgb[0];
    dst_g[x] = src_rgb[1];
    dst_b[x] = src_rgb[2];
    src_rgb += 3;
  }
}

void MergeRGBRow_C(const uint8_t* src_r,
                   const uint8_t* src_g,
                   const uint8_t* src_b,
                   uint8_t* dst_rgb,
                   int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_rgb[0] = src_r[x];
    dst_rgb[1] = src_g[x];
    dst_rgb[2] = src_b[x];
    dst_rgb += 3;
  }
}

void SplitARGBRow_C(const uint8_t* src_argb,
                    uint8_t* dst_r,
                    uint8_t* dst_g,
                    uint8_t* dst_b,
                    uint8_t* dst_a,
                    int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_b[x] = src_argb[0];
    dst_g[x] = src_argb[1];
    dst_r[x] = src_argb[2];
    dst_a[x] = src_argb[3];
    src_argb += 4;
  }
}

void MergeARGBRow_C(const uint8_t* src_r,
                    const uint8_t* src_g,
                    const uint8_t* src_b,
                    const uint8_t* src_a,
                    uint8_t* dst_argb,
                    int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_argb[0] = src_b[x];
    dst_argb[1] = src_g[x];
    dst_argb[2] = src_r[x];
    dst_argb[3] = src_a[x];
    dst_argb += 4;
  }
}

void MergeXR30Row_C(const uint16_t* src_r,
                    const uint16_t* src_g,
                    const uint16_t* src_b,
                    uint8_t* dst_ar30,
                    int depth,
                    int width) {
  assert(depth >= 10);
  assert(depth <= 16);
  int x;
  int shift = depth - 10;
  uint32_t* dst_ar30_32 = (uint32_t*)dst_ar30;
  for (x = 0; x < width; ++x) {
    uint32_t r = clamp1023(src_r[x] >> shift);
    uint32_t g = clamp1023(src_g[x] >> shift);
    uint32_t b = clamp1023(src_b[x] >> shift);
    dst_ar30_32[x] = b | (g << 10) | (r << 20) | 0xc0000000;
  }
}

void MergeAR64Row_C(const uint16_t* src_r,
                    const uint16_t* src_g,
                    const uint16_t* src_b,
                    const uint16_t* src_a,
                    uint16_t* dst_ar64,
                    int depth,
                    int width) {
  assert(depth >= 1);
  assert(depth <= 16);
  int x;
  int shift = 16 - depth;
  int max = (1 << depth) - 1;
  for (x = 0; x < width; ++x) {
    dst_ar64[0] = STATIC_CAST(uint16_t, ClampMax(src_b[x], max) << shift);
    dst_ar64[1] = STATIC_CAST(uint16_t, ClampMax(src_g[x], max) << shift);
    dst_ar64[2] = STATIC_CAST(uint16_t, ClampMax(src_r[x], max) << shift);
    dst_ar64[3] = STATIC_CAST(uint16_t, ClampMax(src_a[x], max) << shift);
    dst_ar64 += 4;
  }
}

void MergeARGB16To8Row_C(const uint16_t* src_r,
                         const uint16_t* src_g,
                         const uint16_t* src_b,
                         const uint16_t* src_a,
                         uint8_t* dst_argb,
                         int depth,
                         int width) {
  assert(depth >= 8);
  assert(depth <= 16);
  int x;
  int shift = depth - 8;
  for (x = 0; x < width; ++x) {
    dst_argb[0] = STATIC_CAST(uint8_t, clamp255(src_b[x] >> shift));
    dst_argb[1] = STATIC_CAST(uint8_t, clamp255(src_g[x] >> shift));
    dst_argb[2] = STATIC_CAST(uint8_t, clamp255(src_r[x] >> shift));
    dst_argb[3] = STATIC_CAST(uint8_t, clamp255(src_a[x] >> shift));
    dst_argb += 4;
  }
}

void MergeXR64Row_C(const uint16_t* src_r,
                    const uint16_t* src_g,
                    const uint16_t* src_b,
                    uint16_t* dst_ar64,
                    int depth,
                    int width) {
  assert(depth >= 1);
  assert(depth <= 16);
  int x;
  int shift = 16 - depth;
  int max = (1 << depth) - 1;
  for (x = 0; x < width; ++x) {
    dst_ar64[0] = STATIC_CAST(uint16_t, ClampMax(src_b[x], max) << shift);
    dst_ar64[1] = STATIC_CAST(uint16_t, ClampMax(src_g[x], max) << shift);
    dst_ar64[2] = STATIC_CAST(uint16_t, ClampMax(src_r[x], max) << shift);
    dst_ar64[3] = 0xffff;
    dst_ar64 += 4;
  }
}

void MergeXRGB16To8Row_C(const uint16_t* src_r,
                         const uint16_t* src_g,
                         const uint16_t* src_b,
                         uint8_t* dst_argb,
                         int depth,
                         int width) {
  assert(depth >= 8);
  assert(depth <= 16);
  int x;
  int shift = depth - 8;
  for (x = 0; x < width; ++x) {
    dst_argb[0] = STATIC_CAST(uint8_t, clamp255(src_b[x] >> shift));
    dst_argb[1] = STATIC_CAST(uint8_t, clamp255(src_g[x] >> shift));
    dst_argb[2] = STATIC_CAST(uint8_t, clamp255(src_r[x] >> shift));
    dst_argb[3] = 0xff;
    dst_argb += 4;
  }
}

void SplitXRGBRow_C(const uint8_t* src_argb,
                    uint8_t* dst_r,
                    uint8_t* dst_g,
                    uint8_t* dst_b,
                    int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_b[x] = src_argb[0];
    dst_g[x] = src_argb[1];
    dst_r[x] = src_argb[2];
    src_argb += 4;
  }
}

void MergeXRGBRow_C(const uint8_t* src_r,
                    const uint8_t* src_g,
                    const uint8_t* src_b,
                    uint8_t* dst_argb,
                    int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_argb[0] = src_b[x];
    dst_argb[1] = src_g[x];
    dst_argb[2] = src_r[x];
    dst_argb[3] = 255;
    dst_argb += 4;
  }
}

// Convert lsb formats to msb, depending on sample depth.
void MergeUVRow_16_C(const uint16_t* src_u,
                     const uint16_t* src_v,
                     uint16_t* dst_uv,
                     int depth,
                     int width) {
  int shift = 16 - depth;
  assert(depth >= 8);
  assert(depth <= 16);
  int x;
  for (x = 0; x < width; ++x) {
    dst_uv[0] = STATIC_CAST(uint16_t, src_u[x] << shift);
    dst_uv[1] = STATIC_CAST(uint16_t, src_v[x] << shift);
    dst_uv += 2;
  }
}

// Convert msb formats to lsb, depending on sample depth.
void SplitUVRow_16_C(const uint16_t* src_uv,
                     uint16_t* dst_u,
                     uint16_t* dst_v,
                     int depth,
                     int width) {
  int shift = 16 - depth;
  int x;
  assert(depth >= 8);
  assert(depth <= 16);
  for (x = 0; x < width; ++x) {
    dst_u[x] = src_uv[0] >> shift;
    dst_v[x] = src_uv[1] >> shift;
    src_uv += 2;
  }
}

void MultiplyRow_16_C(const uint16_t* src_y,
                      uint16_t* dst_y,
                      int scale,
                      int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_y[x] = STATIC_CAST(uint16_t, src_y[x] * scale);
  }
}

void DivideRow_16_C(const uint16_t* src_y,
                    uint16_t* dst_y,
                    int scale,
                    int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_y[x] = (src_y[x] * scale) >> 16;
  }
}

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 32768 = 9 bits
// 16384 = 10 bits
// 4096 = 12 bits
// 256 = 16 bits
// TODO(fbarchard): change scale to bits
#define C16TO8(v, scale) clamp255(((v) * (scale)) >> 16)

void Convert16To8Row_C(const uint16_t* src_y,
                       uint8_t* dst_y,
                       int scale,
                       int width) {
  int x;
  assert(scale >= 256);
  assert(scale <= 32768);

  for (x = 0; x < width; ++x) {
    dst_y[x] = STATIC_CAST(uint8_t, C16TO8(src_y[x], scale));
  }
}

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 1024 = 10 bits
void Convert8To16Row_C(const uint8_t* src_y,
                       uint16_t* dst_y,
                       int scale,
                       int width) {
  int x;
  scale *= 0x0101;  // replicates the byte.
  for (x = 0; x < width; ++x) {
    dst_y[x] = (src_y[x] * scale) >> 16;
  }
}

// Use scale to convert J420 to I420
// scale parameter is 8.8 fixed point but limited to 0 to 255
// Function is based on DivideRow, but adds a bias
// Does not clamp
void Convert8To8Row_C(const uint8_t* src_y,
                      uint8_t* dst_y,
                      int scale,
                      int bias,
                      int width) {
  int x;
  assert(scale >= 0);
  assert(scale <= 255);

  for (x = 0; x < width; ++x) {
    dst_y[x] = ((src_y[x] * scale) >> 8) + bias;
  }
}

void CopyRow_C(const uint8_t* src, uint8_t* dst, int count) {
  memcpy(dst, src, count);
}

void CopyRow_16_C(const uint16_t* src, uint16_t* dst, int count) {
  memcpy(dst, src, count * 2);
}

void SetRow_C(uint8_t* dst, uint8_t v8, int width) {
  memset(dst, v8, width);
}

void ARGBSetRow_C(uint8_t* dst_argb, uint32_t v32, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    memcpy(dst_argb + x * sizeof v32, &v32, sizeof v32);
  }
}

// Filter 2 rows of YUY2 UV's (422) into U and V (420).
void YUY2ToUVRow_C(const uint8_t* src_yuy2,
                   int src_stride_yuy2,
                   uint8_t* dst_u,
                   uint8_t* dst_v,
                   int width) {
  // Output a row of UV values, filtering 2 rows of YUY2.
  int x;
  for (x = 0; x < width; x += 2) {
    dst_u[0] = (src_yuy2[1] + src_yuy2[src_stride_yuy2 + 1] + 1) >> 1;
    dst_v[0] = (src_yuy2[3] + src_yuy2[src_stride_yuy2 + 3] + 1) >> 1;
    src_yuy2 += 4;
    dst_u += 1;
    dst_v += 1;
  }
}

// Filter 2 rows of YUY2 UV's (422) into UV (NV12).
void YUY2ToNVUVRow_C(const uint8_t* src_yuy2,
                     int src_stride_yuy2,
                     uint8_t* dst_uv,
                     int width) {
  // Output a row of UV values, filtering 2 rows of YUY2.
  int x;
  for (x = 0; x < width; x += 2) {
    dst_uv[0] = (src_yuy2[1] + src_yuy2[src_stride_yuy2 + 1] + 1) >> 1;
    dst_uv[1] = (src_yuy2[3] + src_yuy2[src_stride_yuy2 + 3] + 1) >> 1;
    src_yuy2 += 4;
    dst_uv += 2;
  }
}

// Copy row of YUY2 UV's (422) into U and V (422).
void YUY2ToUV422Row_C(const uint8_t* src_yuy2,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  // Output a row of UV values.
  int x;
  for (x = 0; x < width; x += 2) {
    dst_u[0] = src_yuy2[1];
    dst_v[0] = src_yuy2[3];
    src_yuy2 += 4;
    dst_u += 1;
    dst_v += 1;
  }
}

// Copy row of YUY2 Y's (422) into Y (420/422).
void YUY2ToYRow_C(const uint8_t* src_yuy2, uint8_t* dst_y, int width) {
  // Output a row of Y values.
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_y[x] = src_yuy2[0];
    dst_y[x + 1] = src_yuy2[2];
    src_yuy2 += 4;
  }
  if (width & 1) {
    dst_y[width - 1] = src_yuy2[0];
  }
}

// Filter 2 rows of UYVY UV's (422) into U and V (420).
void UYVYToUVRow_C(const uint8_t* src_uyvy,
                   int src_stride_uyvy,
                   uint8_t* dst_u,
                   uint8_t* dst_v,
                   int width) {
  // Output a row of UV values.
  int x;
  for (x = 0; x < width; x += 2) {
    dst_u[0] = (src_uyvy[0] + src_uyvy[src_stride_uyvy + 0] + 1) >> 1;
    dst_v[0] = (src_uyvy[2] + src_uyvy[src_stride_uyvy + 2] + 1) >> 1;
    src_uyvy += 4;
    dst_u += 1;
    dst_v += 1;
  }
}

// Copy row of UYVY UV's (422) into U and V (422).
void UYVYToUV422Row_C(const uint8_t* src_uyvy,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  // Output a row of UV values.
  int x;
  for (x = 0; x < width; x += 2) {
    dst_u[0] = src_uyvy[0];
    dst_v[0] = src_uyvy[2];
    src_uyvy += 4;
    dst_u += 1;
    dst_v += 1;
  }
}

// Copy row of UYVY Y's (422) into Y (420/422).
void UYVYToYRow_C(const uint8_t* src_uyvy, uint8_t* dst_y, int width) {
  // Output a row of Y values.
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_y[x] = src_uyvy[1];
    dst_y[x + 1] = src_uyvy[3];
    src_uyvy += 4;
  }
  if (width & 1) {
    dst_y[width - 1] = src_uyvy[1];
  }
}

#define BLEND(f, b, a) clamp255((((256 - a) * b) >> 8) + f)

// Blend src_argb over src_argb1 and store to dst_argb.
// dst_argb may be src_argb or src_argb1.
// This code mimics the SSSE3 version for better testability.
void ARGBBlendRow_C(const uint8_t* src_argb,
                    const uint8_t* src_argb1,
                    uint8_t* dst_argb,
                    int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint32_t fb = src_argb[0];
    uint32_t fg = src_argb[1];
    uint32_t fr = src_argb[2];
    uint32_t a = src_argb[3];
    uint32_t bb = src_argb1[0];
    uint32_t bg = src_argb1[1];
    uint32_t br = src_argb1[2];
    dst_argb[0] = STATIC_CAST(uint8_t, BLEND(fb, bb, a));
    dst_argb[1] = STATIC_CAST(uint8_t, BLEND(fg, bg, a));
    dst_argb[2] = STATIC_CAST(uint8_t, BLEND(fr, br, a));
    dst_argb[3] = 255u;

    fb = src_argb[4 + 0];
    fg = src_argb[4 + 1];
    fr = src_argb[4 + 2];
    a = src_argb[4 + 3];
    bb = src_argb1[4 + 0];
    bg = src_argb1[4 + 1];
    br = src_argb1[4 + 2];
    dst_argb[4 + 0] = STATIC_CAST(uint8_t, BLEND(fb, bb, a));
    dst_argb[4 + 1] = STATIC_CAST(uint8_t, BLEND(fg, bg, a));
    dst_argb[4 + 2] = STATIC_CAST(uint8_t, BLEND(fr, br, a));
    dst_argb[4 + 3] = 255u;
    src_argb += 8;
    src_argb1 += 8;
    dst_argb += 8;
  }

  if (width & 1) {
    uint32_t fb = src_argb[0];
    uint32_t fg = src_argb[1];
    uint32_t fr = src_argb[2];
    uint32_t a = src_argb[3];
    uint32_t bb = src_argb1[0];
    uint32_t bg = src_argb1[1];
    uint32_t br = src_argb1[2];
    dst_argb[0] = STATIC_CAST(uint8_t, BLEND(fb, bb, a));
    dst_argb[1] = STATIC_CAST(uint8_t, BLEND(fg, bg, a));
    dst_argb[2] = STATIC_CAST(uint8_t, BLEND(fr, br, a));
    dst_argb[3] = 255u;
  }
}
#undef BLEND

#define UBLEND(f, b, a) (((a)*f) + ((255 - a) * b) + 255) >> 8
void BlendPlaneRow_C(const uint8_t* src0,
                     const uint8_t* src1,
                     const uint8_t* alpha,
                     uint8_t* dst,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst[0] = UBLEND(src0[0], src1[0], alpha[0]);
    dst[1] = UBLEND(src0[1], src1[1], alpha[1]);
    src0 += 2;
    src1 += 2;
    alpha += 2;
    dst += 2;
  }
  if (width & 1) {
    dst[0] = UBLEND(src0[0], src1[0], alpha[0]);
  }
}
#undef UBLEND

#define ATTENUATE(f, a) (f * a + 255) >> 8

// Multiply source RGB by alpha and store to destination.
void ARGBAttenuateRow_C(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  int i;
  for (i = 0; i < width - 1; i += 2) {
    uint32_t b = src_argb[0];
    uint32_t g = src_argb[1];
    uint32_t r = src_argb[2];
    uint32_t a = src_argb[3];
    dst_argb[0] = ATTENUATE(b, a);
    dst_argb[1] = ATTENUATE(g, a);
    dst_argb[2] = ATTENUATE(r, a);
    dst_argb[3] = STATIC_CAST(uint8_t, a);
    b = src_argb[4];
    g = src_argb[5];
    r = src_argb[6];
    a = src_argb[7];
    dst_argb[4] = ATTENUATE(b, a);
    dst_argb[5] = ATTENUATE(g, a);
    dst_argb[6] = ATTENUATE(r, a);
    dst_argb[7] = STATIC_CAST(uint8_t, a);
    src_argb += 8;
    dst_argb += 8;
  }

  if (width & 1) {
    const uint32_t b = src_argb[0];
    const uint32_t g = src_argb[1];
    const uint32_t r = src_argb[2];
    const uint32_t a = src_argb[3];
    dst_argb[0] = ATTENUATE(b, a);
    dst_argb[1] = ATTENUATE(g, a);
    dst_argb[2] = ATTENUATE(r, a);
    dst_argb[3] = STATIC_CAST(uint8_t, a);
  }
}
#undef ATTENUATE

// Divide source RGB by alpha and store to destination.
// b = (b * 255 + (a / 2)) / a;
// g = (g * 255 + (a / 2)) / a;
// r = (r * 255 + (a / 2)) / a;
// Reciprocal method is off by 1 on some values. ie 125
// 8.8 fixed point inverse table with 1.0 in upper short and 1 / a in lower.
#define T(a) 0x01000000 + (0x10000 / a)
const uint32_t fixed_invtbl8[256] = {
    0x01000000, 0x0100ffff, T(0x02), T(0x03),   T(0x04), T(0x05), T(0x06),
    T(0x07),    T(0x08),    T(0x09), T(0x0a),   T(0x0b), T(0x0c), T(0x0d),
    T(0x0e),    T(0x0f),    T(0x10), T(0x11),   T(0x12), T(0x13), T(0x14),
    T(0x15),    T(0x16),    T(0x17), T(0x18),   T(0x19), T(0x1a), T(0x1b),
    T(0x1c),    T(0x1d),    T(0x1e), T(0x1f),   T(0x20), T(0x21), T(0x22),
    T(0x23),    T(0x24),    T(0x25), T(0x26),   T(0x27), T(0x28), T(0x29),
    T(0x2a),    T(0x2b),    T(0x2c), T(0x2d),   T(0x2e), T(0x2f), T(0x30),
    T(0x31),    T(0x32),    T(0x33), T(0x34),   T(0x35), T(0x36), T(0x37),
    T(0x38),    T(0x39),    T(0x3a), T(0x3b),   T(0x3c), T(0x3d), T(0x3e),
    T(0x3f),    T(0x40),    T(0x41), T(0x42),   T(0x43), T(0x44), T(0x45),
    T(0x46),    T(0x47),    T(0x48), T(0x49),   T(0x4a), T(0x4b), T(0x4c),
    T(0x4d),    T(0x4e),    T(0x4f), T(0x50),   T(0x51), T(0x52), T(0x53),
    T(0x54),    T(0x55),    T(0x56), T(0x57),   T(0x58), T(0x59), T(0x5a),
    T(0x5b),    T(0x5c),    T(0x5d), T(0x5e),   T(0x5f), T(0x60), T(0x61),
    T(0x62),    T(0x63),    T(0x64), T(0x65),   T(0x66), T(0x67), T(0x68),
    T(0x69),    T(0x6a),    T(0x6b), T(0x6c),   T(0x6d), T(0x6e), T(0x6f),
    T(0x70),    T(0x71),    T(0x72), T(0x73),   T(0x74), T(0x75), T(0x76),
    T(0x77),    T(0x78),    T(0x79), T(0x7a),   T(0x7b), T(0x7c), T(0x7d),
    T(0x7e),    T(0x7f),    T(0x80), T(0x81),   T(0x82), T(0x83), T(0x84),
    T(0x85),    T(0x86),    T(0x87), T(0x88),   T(0x89), T(0x8a), T(0x8b),
    T(0x8c),    T(0x8d),    T(0x8e), T(0x8f),   T(0x90), T(0x91), T(0x92),
    T(0x93),    T(0x94),    T(0x95), T(0x96),   T(0x97), T(0x98), T(0x99),
    T(0x9a),    T(0x9b),    T(0x9c), T(0x9d),   T(0x9e), T(0x9f), T(0xa0),
    T(0xa1),    T(0xa2),    T(0xa3), T(0xa4),   T(0xa5), T(0xa6), T(0xa7),
    T(0xa8),    T(0xa9),    T(0xaa), T(0xab),   T(0xac), T(0xad), T(0xae),
    T(0xaf),    T(0xb0),    T(0xb1), T(0xb2),   T(0xb3), T(0xb4), T(0xb5),
    T(0xb6),    T(0xb7),    T(0xb8), T(0xb9),   T(0xba), T(0xbb), T(0xbc),
    T(0xbd),    T(0xbe),    T(0xbf), T(0xc0),   T(0xc1), T(0xc2), T(0xc3),
    T(0xc4),    T(0xc5),    T(0xc6), T(0xc7),   T(0xc8), T(0xc9), T(0xca),
    T(0xcb),    T(0xcc),    T(0xcd), T(0xce),   T(0xcf), T(0xd0), T(0xd1),
    T(0xd2),    T(0xd3),    T(0xd4), T(0xd5),   T(0xd6), T(0xd7), T(0xd8),
    T(0xd9),    T(0xda),    T(0xdb), T(0xdc),   T(0xdd), T(0xde), T(0xdf),
    T(0xe0),    T(0xe1),    T(0xe2), T(0xe3),   T(0xe4), T(0xe5), T(0xe6),
    T(0xe7),    T(0xe8),    T(0xe9), T(0xea),   T(0xeb), T(0xec), T(0xed),
    T(0xee),    T(0xef),    T(0xf0), T(0xf1),   T(0xf2), T(0xf3), T(0xf4),
    T(0xf5),    T(0xf6),    T(0xf7), T(0xf8),   T(0xf9), T(0xfa), T(0xfb),
    T(0xfc),    T(0xfd),    T(0xfe), 0x01000100};
#undef T

#if defined(LIBYUV_UNATTENUATE_DUP)
// This code mimics the Intel SIMD version for better testability.
#define UNATTENUATE(f, ia) clamp255(((f | (f << 8)) * ia) >> 16)
#else
#define UNATTENUATE(f, ia) clamp255((f * ia) >> 8)
#endif

// mimics the Intel SIMD code for exactness.
void ARGBUnattenuateRow_C(const uint8_t* src_argb,
                          uint8_t* dst_argb,
                          int width) {
  int i;
  for (i = 0; i < width; ++i) {
    uint32_t b = src_argb[0];
    uint32_t g = src_argb[1];
    uint32_t r = src_argb[2];
    const uint32_t a = src_argb[3];
    const uint32_t ia = fixed_invtbl8[a] & 0xffff;  // 8.8 fixed point

    // Clamping should not be necessary but is free in assembly.
    dst_argb[0] = STATIC_CAST(uint8_t, UNATTENUATE(b, ia));
    dst_argb[1] = STATIC_CAST(uint8_t, UNATTENUATE(g, ia));
    dst_argb[2] = STATIC_CAST(uint8_t, UNATTENUATE(r, ia));
    dst_argb[3] = STATIC_CAST(uint8_t, a);
    src_argb += 4;
    dst_argb += 4;
  }
}

void ComputeCumulativeSumRow_C(const uint8_t* row,
                               int32_t* cumsum,
                               const int32_t* previous_cumsum,
                               int width) {
  int32_t row_sum[4] = {0, 0, 0, 0};
  int x;
  for (x = 0; x < width; ++x) {
    row_sum[0] += row[x * 4 + 0];
    row_sum[1] += row[x * 4 + 1];
    row_sum[2] += row[x * 4 + 2];
    row_sum[3] += row[x * 4 + 3];
    cumsum[x * 4 + 0] = row_sum[0] + previous_cumsum[x * 4 + 0];
    cumsum[x * 4 + 1] = row_sum[1] + previous_cumsum[x * 4 + 1];
    cumsum[x * 4 + 2] = row_sum[2] + previous_cumsum[x * 4 + 2];
    cumsum[x * 4 + 3] = row_sum[3] + previous_cumsum[x * 4 + 3];
  }
}

void CumulativeSumToAverageRow_C(const int32_t* tl,
                                 const int32_t* bl,
                                 int w,
                                 int area,
                                 uint8_t* dst,
                                 int count) {
  float ooa;
  int i;
  assert(area != 0);

  ooa = 1.0f / STATIC_CAST(float, area);
  for (i = 0; i < count; ++i) {
    dst[0] =
        (uint8_t)(STATIC_CAST(float, bl[w + 0] + tl[0] - bl[0] - tl[w + 0]) *
                  ooa);
    dst[1] =
        (uint8_t)(STATIC_CAST(float, bl[w + 1] + tl[1] - bl[1] - tl[w + 1]) *
                  ooa);
    dst[2] =
        (uint8_t)(STATIC_CAST(float, bl[w + 2] + tl[2] - bl[2] - tl[w + 2]) *
                  ooa);
    dst[3] =
        (uint8_t)(STATIC_CAST(float, bl[w + 3] + tl[3] - bl[3] - tl[w + 3]) *
                  ooa);
    dst += 4;
    tl += 4;
    bl += 4;
  }
}

// Copy pixels from rotated source to destination row with a slope.
LIBYUV_API
void ARGBAffineRow_C(const uint8_t* src_argb,
                     int src_argb_stride,
                     uint8_t* dst_argb,
                     const float* uv_dudv,
                     int width) {
  int i;
  // Render a row of pixels from source into a buffer.
  float uv[2];
  uv[0] = uv_dudv[0];
  uv[1] = uv_dudv[1];
  for (i = 0; i < width; ++i) {
    int x = (int)(uv[0]);
    int y = (int)(uv[1]);
    *(uint32_t*)(dst_argb) =
        *(const uint32_t*)(src_argb + y * src_argb_stride + x * 4);
    dst_argb += 4;
    uv[0] += uv_dudv[2];
    uv[1] += uv_dudv[3];
  }
}

// Blend 2 rows into 1.
static void HalfRow_C(const uint8_t* src_uv,
                      ptrdiff_t src_uv_stride,
                      uint8_t* dst_uv,
                      int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_uv[x] = (src_uv[x] + src_uv[src_uv_stride + x] + 1) >> 1;
  }
}

static void HalfRow_16_C(const uint16_t* src_uv,
                         ptrdiff_t src_uv_stride,
                         uint16_t* dst_uv,
                         int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_uv[x] = (src_uv[x] + src_uv[src_uv_stride + x] + 1) >> 1;
  }
}

static void HalfRow_16To8_C(const uint16_t* src_uv,
                            ptrdiff_t src_uv_stride,
                            uint8_t* dst_uv,
                            int scale,
                            int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_uv[x] = STATIC_CAST(
        uint8_t,
        C16TO8((src_uv[x] + src_uv[src_uv_stride + x] + 1) >> 1, scale));
  }
}

// C version 2x2 -> 2x1.
void InterpolateRow_C(uint8_t* dst_ptr,
                      const uint8_t* src_ptr,
                      ptrdiff_t src_stride,
                      int width,
                      int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  int x;
  assert(source_y_fraction >= 0);
  assert(source_y_fraction < 256);

  if (y1_fraction == 0) {
    memcpy(dst_ptr, src_ptr, width);
    return;
  }
  if (y1_fraction == 128) {
    HalfRow_C(src_ptr, src_stride, dst_ptr, width);
    return;
  }
  for (x = 0; x < width; ++x) {
    dst_ptr[0] = STATIC_CAST(
        uint8_t,
        (src_ptr[0] * y0_fraction + src_ptr1[0] * y1_fraction + 128) >> 8);
    ++src_ptr;
    ++src_ptr1;
    ++dst_ptr;
  }
}

// C version 2x2 -> 2x1.
void InterpolateRow_16_C(uint16_t* dst_ptr,
                         const uint16_t* src_ptr,
                         ptrdiff_t src_stride,
                         int width,
                         int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint16_t* src_ptr1 = src_ptr + src_stride;
  int x;
  assert(source_y_fraction >= 0);
  assert(source_y_fraction < 256);

  if (y1_fraction == 0) {
    memcpy(dst_ptr, src_ptr, width * 2);
    return;
  }
  if (y1_fraction == 128) {
    HalfRow_16_C(src_ptr, src_stride, dst_ptr, width);
    return;
  }
  for (x = 0; x < width; ++x) {
    dst_ptr[0] = STATIC_CAST(
        uint16_t,
        (src_ptr[0] * y0_fraction + src_ptr1[0] * y1_fraction + 128) >> 8);
    ++src_ptr;
    ++src_ptr1;
    ++dst_ptr;
  }
}

// C version 2x2 16 bit-> 2x1 8 bit.
// Use scale to convert lsb formats to msb, depending how many bits there are:
// 32768 = 9 bits
// 16384 = 10 bits
// 4096 = 12 bits
// 256 = 16 bits
// TODO(fbarchard): change scale to bits

void InterpolateRow_16To8_C(uint8_t* dst_ptr,
                            const uint16_t* src_ptr,
                            ptrdiff_t src_stride,
                            int scale,
                            int width,
                            int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint16_t* src_ptr1 = src_ptr + src_stride;
  int x;
  assert(source_y_fraction >= 0);
  assert(source_y_fraction < 256);

  if (source_y_fraction == 0) {
    Convert16To8Row_C(src_ptr, dst_ptr, scale, width);
    return;
  }
  if (source_y_fraction == 128) {
    HalfRow_16To8_C(src_ptr, src_stride, dst_ptr, scale, width);
    return;
  }
  for (x = 0; x < width; ++x) {
    dst_ptr[0] = STATIC_CAST(
        uint8_t,
        C16TO8(
            (src_ptr[0] * y0_fraction + src_ptr1[0] * y1_fraction + 128) >> 8,
            scale));
    src_ptr += 1;
    src_ptr1 += 1;
    dst_ptr += 1;
  }
}

// Use first 4 shuffler values to reorder ARGB channels.
void ARGBShuffleRow_C(const uint8_t* src_argb,
                      uint8_t* dst_argb,
                      const uint8_t* shuffler,
                      int width) {
  int index0 = shuffler[0];
  int index1 = shuffler[1];
  int index2 = shuffler[2];
  int index3 = shuffler[3];
  // Shuffle a row of ARGB.
  int x;
  for (x = 0; x < width; ++x) {
    // To support in-place conversion.
    uint8_t b = src_argb[index0];
    uint8_t g = src_argb[index1];
    uint8_t r = src_argb[index2];
    uint8_t a = src_argb[index3];
    dst_argb[0] = b;
    dst_argb[1] = g;
    dst_argb[2] = r;
    dst_argb[3] = a;
    src_argb += 4;
    dst_argb += 4;
  }
}

void I422ToYUY2Row_C(const uint8_t* src_y,
                     const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* dst_frame,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_frame[0] = src_y[0];
    dst_frame[1] = src_u[0];
    dst_frame[2] = src_y[1];
    dst_frame[3] = src_v[0];
    dst_frame += 4;
    src_y += 2;
    src_u += 1;
    src_v += 1;
  }
  if (width & 1) {
    dst_frame[0] = src_y[0];
    dst_frame[1] = src_u[0];
    dst_frame[2] = 0;
    dst_frame[3] = src_v[0];
  }
}

void I422ToUYVYRow_C(const uint8_t* src_y,
                     const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* dst_frame,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_frame[0] = src_u[0];
    dst_frame[1] = src_y[0];
    dst_frame[2] = src_v[0];
    dst_frame[3] = src_y[1];
    dst_frame += 4;
    src_y += 2;
    src_u += 1;
    src_v += 1;
  }
  if (width & 1) {
    dst_frame[0] = src_u[0];
    dst_frame[1] = src_y[0];
    dst_frame[2] = src_v[0];
    dst_frame[3] = 0;
  }
}

void ARGBPolynomialRow_C(const uint8_t* src_argb,
                         uint8_t* dst_argb,
                         const float* poly,
                         int width) {
  int i;
  for (i = 0; i < width; ++i) {
    float b = (float)(src_argb[0]);
    float g = (float)(src_argb[1]);
    float r = (float)(src_argb[2]);
    float a = (float)(src_argb[3]);
    float b2 = b * b;
    float g2 = g * g;
    float r2 = r * r;
    float a2 = a * a;
    float db = poly[0] + poly[4] * b;
    float dg = poly[1] + poly[5] * g;
    float dr = poly[2] + poly[6] * r;
    float da = poly[3] + poly[7] * a;
    float b3 = b2 * b;
    float g3 = g2 * g;
    float r3 = r2 * r;
    float a3 = a2 * a;
    db += poly[8] * b2;
    dg += poly[9] * g2;
    dr += poly[10] * r2;
    da += poly[11] * a2;
    db += poly[12] * b3;
    dg += poly[13] * g3;
    dr += poly[14] * r3;
    da += poly[15] * a3;

    dst_argb[0] = STATIC_CAST(uint8_t, Clamp((int32_t)(db)));
    dst_argb[1] = STATIC_CAST(uint8_t, Clamp((int32_t)(dg)));
    dst_argb[2] = STATIC_CAST(uint8_t, Clamp((int32_t)(dr)));
    dst_argb[3] = STATIC_CAST(uint8_t, Clamp((int32_t)(da)));
    src_argb += 4;
    dst_argb += 4;
  }
}

// Samples assumed to be unsigned in low 9, 10 or 12 bits.  Scale factor
// adjust the source integer range to the half float range desired.

// This magic constant is 2^-112. Multiplying by this
// is the same as subtracting 112 from the exponent, which
// is the difference in exponent bias between 32-bit and
// 16-bit floats. Once we've done this subtraction, we can
// simply extract the low bits of the exponent and the high
// bits of the mantissa from our float and we're done.

// Work around GCC 7 punning warning -Wstrict-aliasing
#if defined(__GNUC__)
typedef uint32_t __attribute__((__may_alias__)) uint32_alias_t;
#else
typedef uint32_t uint32_alias_t;
#endif

void HalfFloatRow_C(const uint16_t* src,
                    uint16_t* dst,
                    float scale,
                    int width) {
  int i;
  float mult = 1.9259299444e-34f * scale;
  for (i = 0; i < width; ++i) {
    float value = src[i] * mult;
    dst[i] = (uint16_t)((*(const uint32_alias_t*)&value) >> 13);
  }
}

void ByteToFloatRow_C(const uint8_t* src, float* dst, float scale, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    float value = src[i] * scale;
    dst[i] = value;
  }
}

void ARGBLumaColorTableRow_C(const uint8_t* src_argb,
                             uint8_t* dst_argb,
                             int width,
                             const uint8_t* luma,
                             uint32_t lumacoeff) {
  uint32_t bc = lumacoeff & 0xff;
  uint32_t gc = (lumacoeff >> 8) & 0xff;
  uint32_t rc = (lumacoeff >> 16) & 0xff;

  int i;
  for (i = 0; i < width - 1; i += 2) {
    // Luminance in rows, color values in columns.
    const uint8_t* luma0 =
        ((src_argb[0] * bc + src_argb[1] * gc + src_argb[2] * rc) & 0x7F00u) +
        luma;
    const uint8_t* luma1;
    dst_argb[0] = luma0[src_argb[0]];
    dst_argb[1] = luma0[src_argb[1]];
    dst_argb[2] = luma0[src_argb[2]];
    dst_argb[3] = src_argb[3];
    luma1 =
        ((src_argb[4] * bc + src_argb[5] * gc + src_argb[6] * rc) & 0x7F00u) +
        luma;
    dst_argb[4] = luma1[src_argb[4]];
    dst_argb[5] = luma1[src_argb[5]];
    dst_argb[6] = luma1[src_argb[6]];
    dst_argb[7] = src_argb[7];
    src_argb += 8;
    dst_argb += 8;
  }
  if (width & 1) {
    // Luminance in rows, color values in columns.
    const uint8_t* luma0 =
        ((src_argb[0] * bc + src_argb[1] * gc + src_argb[2] * rc) & 0x7F00u) +
        luma;
    dst_argb[0] = luma0[src_argb[0]];
    dst_argb[1] = luma0[src_argb[1]];
    dst_argb[2] = luma0[src_argb[2]];
    dst_argb[3] = src_argb[3];
  }
}

void ARGBCopyAlphaRow_C(const uint8_t* src, uint8_t* dst, int width) {
  int i;
  for (i = 0; i < width - 1; i += 2) {
    dst[3] = src[3];
    dst[7] = src[7];
    dst += 8;
    src += 8;
  }
  if (width & 1) {
    dst[3] = src[3];
  }
}

void ARGBExtractAlphaRow_C(const uint8_t* src_argb, uint8_t* dst_a, int width) {
  int i;
  for (i = 0; i < width - 1; i += 2) {
    dst_a[0] = src_argb[3];
    dst_a[1] = src_argb[7];
    dst_a += 2;
    src_argb += 8;
  }
  if (width & 1) {
    dst_a[0] = src_argb[3];
  }
}

void ARGBCopyYToAlphaRow_C(const uint8_t* src, uint8_t* dst, int width) {
  int i;
  for (i = 0; i < width - 1; i += 2) {
    dst[3] = src[0];
    dst[7] = src[1];
    dst += 8;
    src += 2;
  }
  if (width & 1) {
    dst[3] = src[0];
  }
}

// Maximum temporary width for wrappers to process at a time, in pixels.
#define MAXTWIDTH 2048

#if !(defined(_MSC_VER) && !defined(__clang__) && defined(_M_IX86)) && \
    defined(HAS_I422TORGB565ROW_SSSE3) && !defined(LIBYUV_ENABLE_ROWWIN)
// row_win.cc has asm version, but GCC uses 2 step wrapper.
void I422ToRGB565Row_SSSE3(const uint8_t* src_y,
                           const uint8_t* src_u,
                           const uint8_t* src_v,
                           uint8_t* dst_rgb565,
                           const struct YuvConstants* yuvconstants,
                           int width) {
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    I422ToARGBRow_SSSE3(src_y, src_u, src_v, row, yuvconstants, twidth);
    ARGBToRGB565Row_SSE2(row, dst_rgb565, twidth);
    src_y += twidth;
    src_u += twidth / 2;
    src_v += twidth / 2;
    dst_rgb565 += twidth * 2;
    width -= twidth;
  }
}
#endif

#if defined(HAS_I422TOARGB1555ROW_SSSE3)
void I422ToARGB1555Row_SSSE3(const uint8_t* src_y,
                             const uint8_t* src_u,
                             const uint8_t* src_v,
                             uint8_t* dst_argb1555,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    I422ToARGBRow_SSSE3(src_y, src_u, src_v, row, yuvconstants, twidth);
    ARGBToARGB1555Row_SSE2(row, dst_argb1555, twidth);
    src_y += twidth;
    src_u += twidth / 2;
    src_v += twidth / 2;
    dst_argb1555 += twidth * 2;
    width -= twidth;
  }
}
#endif

#if defined(HAS_I422TOARGB4444ROW_SSSE3)
void I422ToARGB4444Row_SSSE3(const uint8_t* src_y,
                             const uint8_t* src_u,
                             const uint8_t* src_v,
                             uint8_t* dst_argb4444,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    I422ToARGBRow_SSSE3(src_y, src_u, src_v, row, yuvconstants, twidth);
    ARGBToARGB4444Row_SSE2(row, dst_argb4444, twidth);
    src_y += twidth;
    src_u += twidth / 2;
    src_v += twidth / 2;
    dst_argb4444 += twidth * 2;
    width -= twidth;
  }
}
#endif

#if defined(HAS_NV12TORGB565ROW_SSSE3)
void NV12ToRGB565Row_SSSE3(const uint8_t* src_y,
                           const uint8_t* src_uv,
                           uint8_t* dst_rgb565,
                           const struct YuvConstants* yuvconstants,
                           int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    NV12ToARGBRow_SSSE3(src_y, src_uv, row, yuvconstants, twidth);
    ARGBToRGB565Row_SSE2(row, dst_rgb565, twidth);
    src_y += twidth;
    src_uv += twidth;
    dst_rgb565 += twidth * 2;
    width -= twidth;
  }
}
#endif

#if defined(HAS_NV12TORGB24ROW_SSSE3)
void NV12ToRGB24Row_SSSE3(const uint8_t* src_y,
                          const uint8_t* src_uv,
                          uint8_t* dst_rgb24,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    NV12ToARGBRow_SSSE3(src_y, src_uv, row, yuvconstants, twidth);
    ARGBToRGB24Row_SSSE3(row, dst_rgb24, twidth);
    src_y += twidth;
    src_uv += twidth;
    dst_rgb24 += twidth * 3;
    width -= twidth;
  }
}
#endif

#if defined(HAS_NV21TORGB24ROW_SSSE3)
void NV21ToRGB24Row_SSSE3(const uint8_t* src_y,
                          const uint8_t* src_vu,
                          uint8_t* dst_rgb24,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    NV21ToARGBRow_SSSE3(src_y, src_vu, row, yuvconstants, twidth);
    ARGBToRGB24Row_SSSE3(row, dst_rgb24, twidth);
    src_y += twidth;
    src_vu += twidth;
    dst_rgb24 += twidth * 3;
    width -= twidth;
  }
}
#endif

#if defined(HAS_NV12TORGB24ROW_AVX2)
void NV12ToRGB24Row_AVX2(const uint8_t* src_y,
                         const uint8_t* src_uv,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    NV12ToARGBRow_AVX2(src_y, src_uv, row, yuvconstants, twidth);
#if defined(HAS_ARGBTORGB24ROW_AVX2)
    ARGBToRGB24Row_AVX2(row, dst_rgb24, twidth);
#else
    ARGBToRGB24Row_SSSE3(row, dst_rgb24, twidth);
#endif
    src_y += twidth;
    src_uv += twidth;
    dst_rgb24 += twidth * 3;
    width -= twidth;
  }
}
#endif

#if defined(HAS_NV21TORGB24ROW_AVX2)
void NV21ToRGB24Row_AVX2(const uint8_t* src_y,
                         const uint8_t* src_vu,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    NV21ToARGBRow_AVX2(src_y, src_vu, row, yuvconstants, twidth);
#if defined(HAS_ARGBTORGB24ROW_AVX2)
    ARGBToRGB24Row_AVX2(row, dst_rgb24, twidth);
#else
    ARGBToRGB24Row_SSSE3(row, dst_rgb24, twidth);
#endif
    src_y += twidth;
    src_vu += twidth;
    dst_rgb24 += twidth * 3;
    width -= twidth;
  }
}
#endif

#if defined(HAS_I422TORGB565ROW_AVX2)
void I422ToRGB565Row_AVX2(const uint8_t* src_y,
                          const uint8_t* src_u,
                          const uint8_t* src_v,
                          uint8_t* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    I422ToARGBRow_AVX2(src_y, src_u, src_v, row, yuvconstants, twidth);
#if defined(HAS_ARGBTORGB565ROW_AVX2)
    ARGBToRGB565Row_AVX2(row, dst_rgb565, twidth);
#else
    ARGBToRGB565Row_SSE2(row, dst_rgb565, twidth);
#endif
    src_y += twidth;
    src_u += twidth / 2;
    src_v += twidth / 2;
    dst_rgb565 += twidth * 2;
    width -= twidth;
  }
}
#endif

#if defined(HAS_I422TOARGB1555ROW_AVX2)
void I422ToARGB1555Row_AVX2(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb1555,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    I422ToARGBRow_AVX2(src_y, src_u, src_v, row, yuvconstants, twidth);
#if defined(HAS_ARGBTOARGB1555ROW_AVX2)
    ARGBToARGB1555Row_AVX2(row, dst_argb1555, twidth);
#else
    ARGBToARGB1555Row_SSE2(row, dst_argb1555, twidth);
#endif
    src_y += twidth;
    src_u += twidth / 2;
    src_v += twidth / 2;
    dst_argb1555 += twidth * 2;
    width -= twidth;
  }
}
#endif

#if defined(HAS_I422TOARGB4444ROW_AVX2)
void I422ToARGB4444Row_AVX2(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb4444,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    I422ToARGBRow_AVX2(src_y, src_u, src_v, row, yuvconstants, twidth);
#if defined(HAS_ARGBTOARGB4444ROW_AVX2)
    ARGBToARGB4444Row_AVX2(row, dst_argb4444, twidth);
#else
    ARGBToARGB4444Row_SSE2(row, dst_argb4444, twidth);
#endif
    src_y += twidth;
    src_u += twidth / 2;
    src_v += twidth / 2;
    dst_argb4444 += twidth * 2;
    width -= twidth;
  }
}
#endif

#if defined(HAS_I422TORGB24ROW_AVX2)
void I422ToRGB24Row_AVX2(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    I422ToARGBRow_AVX2(src_y, src_u, src_v, row, yuvconstants, twidth);
#if defined(HAS_ARGBTORGB24ROW_AVX2)
    ARGBToRGB24Row_AVX2(row, dst_rgb24, twidth);
#else
    ARGBToRGB24Row_SSSE3(row, dst_rgb24, twidth);
#endif
    src_y += twidth;
    src_u += twidth / 2;
    src_v += twidth / 2;
    dst_rgb24 += twidth * 3;
    width -= twidth;
  }
}
#endif

#if defined(HAS_I444TORGB24ROW_AVX2)
void I444ToRGB24Row_AVX2(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    I444ToARGBRow_AVX2(src_y, src_u, src_v, row, yuvconstants, twidth);
#if defined(HAS_ARGBTORGB24ROW_AVX2)
    ARGBToRGB24Row_AVX2(row, dst_rgb24, twidth);
#else
    ARGBToRGB24Row_SSSE3(row, dst_rgb24, twidth);
#endif
    src_y += twidth;
    src_u += twidth;
    src_v += twidth;
    dst_rgb24 += twidth * 3;
    width -= twidth;
  }
}
#endif

#if defined(HAS_NV12TORGB565ROW_AVX2)
void NV12ToRGB565Row_AVX2(const uint8_t* src_y,
                          const uint8_t* src_uv,
                          uint8_t* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    NV12ToARGBRow_AVX2(src_y, src_uv, row, yuvconstants, twidth);
#if defined(HAS_ARGBTORGB565ROW_AVX2)
    ARGBToRGB565Row_AVX2(row, dst_rgb565, twidth);
#else
    ARGBToRGB565Row_SSE2(row, dst_rgb565, twidth);
#endif
    src_y += twidth;
    src_uv += twidth;
    dst_rgb565 += twidth * 2;
    width -= twidth;
  }
}
#endif

#ifdef HAS_RGB24TOYJROW_AVX2
// Convert 16 RGB24 pixels (64 bytes) to 16 YJ values.
void RGB24ToYJRow_AVX2(const uint8_t* src_rgb24, uint8_t* dst_yj, int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    RGB24ToARGBRow_SSSE3(src_rgb24, row, twidth);
    ARGBToYJRow_AVX2(row, dst_yj, twidth);
    src_rgb24 += twidth * 3;
    dst_yj += twidth;
    width -= twidth;
  }
}
#endif  // HAS_RGB24TOYJROW_AVX2

#ifdef HAS_RAWTOYJROW_AVX2
// Convert 32 RAW pixels (128 bytes) to 32 YJ values.
void RAWToYJRow_AVX2(const uint8_t* src_raw, uint8_t* dst_yj, int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
#ifdef HAS_RAWTOARGBROW_AVX2
    RAWToARGBRow_AVX2(src_raw, row, twidth);
#else
    RAWToARGBRow_SSSE3(src_raw, row, twidth);
#endif
    ARGBToYJRow_AVX2(row, dst_yj, twidth);
    src_raw += twidth * 3;
    dst_yj += twidth;
    width -= twidth;
  }
}
#endif  // HAS_RAWTOYJROW_AVX2

#ifdef HAS_RGB24TOYJROW_SSSE3
// Convert 16 RGB24 pixels (64 bytes) to 16 YJ values.
void RGB24ToYJRow_SSSE3(const uint8_t* src_rgb24, uint8_t* dst_yj, int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    RGB24ToARGBRow_SSSE3(src_rgb24, row, twidth);
    ARGBToYJRow_SSSE3(row, dst_yj, twidth);
    src_rgb24 += twidth * 3;
    dst_yj += twidth;
    width -= twidth;
  }
}
#endif  // HAS_RGB24TOYJROW_SSSE3

#ifdef HAS_RAWTOYJROW_SSSE3
// Convert 16 RAW pixels (64 bytes) to 16 YJ values.
void RAWToYJRow_SSSE3(const uint8_t* src_raw, uint8_t* dst_yj, int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8_t row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    RAWToARGBRow_SSSE3(src_raw, row, twidth);
    ARGBToYJRow_SSSE3(row, dst_yj, twidth);
    src_raw += twidth * 3;
    dst_yj += twidth;
    width -= twidth;
  }
}
#endif  // HAS_RAWTOYJROW_SSSE3

#ifdef HAS_INTERPOLATEROW_16TO8_AVX2
void InterpolateRow_16To8_AVX2(uint8_t* dst_ptr,
                               const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               int scale,
                               int width,
                               int source_y_fraction) {
  // Row buffer for intermediate 16 bit pixels.
  SIMD_ALIGNED(uint16_t row[MAXTWIDTH]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    InterpolateRow_16_C(row, src_ptr, src_stride, twidth, source_y_fraction);
    Convert16To8Row_AVX2(row, dst_ptr, scale, twidth);
    src_ptr += twidth;
    dst_ptr += twidth;
    width -= twidth;
  }
}
#endif  // HAS_INTERPOLATEROW_16TO8_AVX2

float ScaleSumSamples_C(const float* src, float* dst, float scale, int width) {
  float fsum = 0.f;
  int i;
  for (i = 0; i < width; ++i) {
    float v = *src++;
    fsum += v * v;
    *dst++ = v * scale;
  }
  return fsum;
}

float ScaleMaxSamples_C(const float* src, float* dst, float scale, int width) {
  float fmax = 0.f;
  int i;
  for (i = 0; i < width; ++i) {
    float v = *src++;
    float vs = v * scale;
    fmax = (v > fmax) ? v : fmax;
    *dst++ = vs;
  }
  return fmax;
}

void ScaleSamples_C(const float* src, float* dst, float scale, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    *dst++ = *src++ * scale;
  }
}

void GaussRow_C(const uint32_t* src, uint16_t* dst, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    *dst++ = STATIC_CAST(
        uint16_t,
        (src[0] + src[1] * 4 + src[2] * 6 + src[3] * 4 + src[4] + 128) >> 8);
    ++src;
  }
}

// filter 5 rows with 1, 4, 6, 4, 1 coefficients to produce 1 row.
void GaussCol_C(const uint16_t* src0,
                const uint16_t* src1,
                const uint16_t* src2,
                const uint16_t* src3,
                const uint16_t* src4,
                uint32_t* dst,
                int width) {
  int i;
  for (i = 0; i < width; ++i) {
    *dst++ = *src0++ + *src1++ * 4 + *src2++ * 6 + *src3++ * 4 + *src4++;
  }
}

void GaussRow_F32_C(const float* src, float* dst, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    *dst++ = (src[0] + src[1] * 4 + src[2] * 6 + src[3] * 4 + src[4]) *
             (1.0f / 256.0f);
    ++src;
  }
}

// filter 5 rows with 1, 4, 6, 4, 1 coefficients to produce 1 row.
void GaussCol_F32_C(const float* src0,
                    const float* src1,
                    const float* src2,
                    const float* src3,
                    const float* src4,
                    float* dst,
                    int width) {
  int i;
  for (i = 0; i < width; ++i) {
    *dst++ = *src0++ + *src1++ * 4 + *src2++ * 6 + *src3++ * 4 + *src4++;
  }
}

// Convert biplanar NV21 to packed YUV24
void NV21ToYUV24Row_C(const uint8_t* src_y,
                      const uint8_t* src_vu,
                      uint8_t* dst_yuv24,
                      int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_yuv24[0] = src_vu[0];  // V
    dst_yuv24[1] = src_vu[1];  // U
    dst_yuv24[2] = src_y[0];   // Y0
    dst_yuv24[3] = src_vu[0];  // V
    dst_yuv24[4] = src_vu[1];  // U
    dst_yuv24[5] = src_y[1];   // Y1
    src_y += 2;
    src_vu += 2;
    dst_yuv24 += 6;  // Advance 2 pixels.
  }
  if (width & 1) {
    dst_yuv24[0] = src_vu[0];  // V
    dst_yuv24[1] = src_vu[1];  // U
    dst_yuv24[2] = src_y[0];   // Y0
  }
}

// Filter 2 rows of AYUV UV's (444) into UV (420).
// AYUV is VUYA in memory.  UV for NV12 is UV order in memory.
void AYUVToUVRow_C(const uint8_t* src_ayuv,
                   int src_stride_ayuv,
                   uint8_t* dst_uv,
                   int width) {
  // Output a row of UV values, filtering 2x2 rows of AYUV.
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_uv[0] = (src_ayuv[1] + src_ayuv[5] + src_ayuv[src_stride_ayuv + 1] +
                 src_ayuv[src_stride_ayuv + 5] + 2) >>
                2;
    dst_uv[1] = (src_ayuv[0] + src_ayuv[4] + src_ayuv[src_stride_ayuv + 0] +
                 src_ayuv[src_stride_ayuv + 4] + 2) >>
                2;
    src_ayuv += 8;
    dst_uv += 2;
  }
  if (width & 1) {
    dst_uv[0] = (src_ayuv[1] + src_ayuv[src_stride_ayuv + 1] + 1) >> 1;
    dst_uv[1] = (src_ayuv[0] + src_ayuv[src_stride_ayuv + 0] + 1) >> 1;
  }
}

// Filter 2 rows of AYUV UV's (444) into VU (420).
void AYUVToVURow_C(const uint8_t* src_ayuv,
                   int src_stride_ayuv,
                   uint8_t* dst_vu,
                   int width) {
  // Output a row of VU values, filtering 2x2 rows of AYUV.
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_vu[0] = (src_ayuv[0] + src_ayuv[4] + src_ayuv[src_stride_ayuv + 0] +
                 src_ayuv[src_stride_ayuv + 4] + 2) >>
                2;
    dst_vu[1] = (src_ayuv[1] + src_ayuv[5] + src_ayuv[src_stride_ayuv + 1] +
                 src_ayuv[src_stride_ayuv + 5] + 2) >>
                2;
    src_ayuv += 8;
    dst_vu += 2;
  }
  if (width & 1) {
    dst_vu[0] = (src_ayuv[0] + src_ayuv[src_stride_ayuv + 0] + 1) >> 1;
    dst_vu[1] = (src_ayuv[1] + src_ayuv[src_stride_ayuv + 1] + 1) >> 1;
  }
}

// Copy row of AYUV Y's into Y
void AYUVToYRow_C(const uint8_t* src_ayuv, uint8_t* dst_y, int width) {
  // Output a row of Y values.
  int x;
  for (x = 0; x < width; ++x) {
    dst_y[x] = src_ayuv[2];  // v,u,y,a
    src_ayuv += 4;
  }
}

// Convert UV plane of NV12 to VU of NV21.
void SwapUVRow_C(const uint8_t* src_uv, uint8_t* dst_vu, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8_t u = src_uv[0];
    uint8_t v = src_uv[1];
    dst_vu[0] = v;
    dst_vu[1] = u;
    src_uv += 2;
    dst_vu += 2;
  }
}

void HalfMergeUVRow_C(const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_uv,
                      int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    dst_uv[0] = (src_u[0] + src_u[1] + src_u[src_stride_u] +
                 src_u[src_stride_u + 1] + 2) >>
                2;
    dst_uv[1] = (src_v[0] + src_v[1] + src_v[src_stride_v] +
                 src_v[src_stride_v + 1] + 2) >>
                2;
    src_u += 2;
    src_v += 2;
    dst_uv += 2;
  }
  if (width & 1) {
    dst_uv[0] = (src_u[0] + src_u[src_stride_u] + 1) >> 1;
    dst_uv[1] = (src_v[0] + src_v[src_stride_v] + 1) >> 1;
  }
}

#undef STATIC_CAST

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
