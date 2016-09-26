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

#include <string.h>  // For memcpy and memset.

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// llvm x86 is poor at ternary operator, so use branchless min/max.

#define USE_BRANCHLESS 1
#if USE_BRANCHLESS
static __inline int32 clamp0(int32 v) {
  return ((-(v) >> 31) & (v));
}

static __inline int32 clamp255(int32 v) {
  return (((255 - (v)) >> 31) | (v)) & 255;
}

static __inline uint32 Clamp(int32 val) {
  int v = clamp0(val);
  return (uint32)(clamp255(v));
}

static __inline uint32 Abs(int32 v) {
  int m = v >> 31;
  return (v + m) ^ m;
}
#else  // USE_BRANCHLESS
static __inline int32 clamp0(int32 v) {
  return (v < 0) ? 0 : v;
}

static __inline int32 clamp255(int32 v) {
  return (v > 255) ? 255 : v;
}

static __inline uint32 Clamp(int32 val) {
  int v = clamp0(val);
  return (uint32)(clamp255(v));
}

static __inline uint32 Abs(int32 v) {
  return (v < 0) ? -v : v;
}
#endif  // USE_BRANCHLESS

#ifdef LIBYUV_LITTLE_ENDIAN
#define WRITEWORD(p, v) *(uint32*)(p) = v
#else
static inline void WRITEWORD(uint8* p, uint32 v) {
  p[0] = (uint8)(v & 255);
  p[1] = (uint8)((v >> 8) & 255);
  p[2] = (uint8)((v >> 16) & 255);
  p[3] = (uint8)((v >> 24) & 255);
}
#endif

void RGB24ToARGBRow_C(const uint8* src_rgb24, uint8* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 b = src_rgb24[0];
    uint8 g = src_rgb24[1];
    uint8 r = src_rgb24[2];
    dst_argb[0] = b;
    dst_argb[1] = g;
    dst_argb[2] = r;
    dst_argb[3] = 255u;
    dst_argb += 4;
    src_rgb24 += 3;
  }
}

void RAWToARGBRow_C(const uint8* src_raw, uint8* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 r = src_raw[0];
    uint8 g = src_raw[1];
    uint8 b = src_raw[2];
    dst_argb[0] = b;
    dst_argb[1] = g;
    dst_argb[2] = r;
    dst_argb[3] = 255u;
    dst_argb += 4;
    src_raw += 3;
  }
}

void RAWToRGB24Row_C(const uint8* src_raw, uint8* dst_rgb24, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 r = src_raw[0];
    uint8 g = src_raw[1];
    uint8 b = src_raw[2];
    dst_rgb24[0] = b;
    dst_rgb24[1] = g;
    dst_rgb24[2] = r;
    dst_rgb24 += 3;
    src_raw += 3;
  }
}

void RGB565ToARGBRow_C(const uint8* src_rgb565, uint8* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 b = src_rgb565[0] & 0x1f;
    uint8 g = (src_rgb565[0] >> 5) | ((src_rgb565[1] & 0x07) << 3);
    uint8 r = src_rgb565[1] >> 3;
    dst_argb[0] = (b << 3) | (b >> 2);
    dst_argb[1] = (g << 2) | (g >> 4);
    dst_argb[2] = (r << 3) | (r >> 2);
    dst_argb[3] = 255u;
    dst_argb += 4;
    src_rgb565 += 2;
  }
}

void ARGB1555ToARGBRow_C(const uint8* src_argb1555, uint8* dst_argb,
                         int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 b = src_argb1555[0] & 0x1f;
    uint8 g = (src_argb1555[0] >> 5) | ((src_argb1555[1] & 0x03) << 3);
    uint8 r = (src_argb1555[1] & 0x7c) >> 2;
    uint8 a = src_argb1555[1] >> 7;
    dst_argb[0] = (b << 3) | (b >> 2);
    dst_argb[1] = (g << 3) | (g >> 2);
    dst_argb[2] = (r << 3) | (r >> 2);
    dst_argb[3] = -a;
    dst_argb += 4;
    src_argb1555 += 2;
  }
}

void ARGB4444ToARGBRow_C(const uint8* src_argb4444, uint8* dst_argb,
                         int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 b = src_argb4444[0] & 0x0f;
    uint8 g = src_argb4444[0] >> 4;
    uint8 r = src_argb4444[1] & 0x0f;
    uint8 a = src_argb4444[1] >> 4;
    dst_argb[0] = (b << 4) | b;
    dst_argb[1] = (g << 4) | g;
    dst_argb[2] = (r << 4) | r;
    dst_argb[3] = (a << 4) | a;
    dst_argb += 4;
    src_argb4444 += 2;
  }
}

void ARGBToRGB24Row_C(const uint8* src_argb, uint8* dst_rgb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 b = src_argb[0];
    uint8 g = src_argb[1];
    uint8 r = src_argb[2];
    dst_rgb[0] = b;
    dst_rgb[1] = g;
    dst_rgb[2] = r;
    dst_rgb += 3;
    src_argb += 4;
  }
}

void ARGBToRAWRow_C(const uint8* src_argb, uint8* dst_rgb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 b = src_argb[0];
    uint8 g = src_argb[1];
    uint8 r = src_argb[2];
    dst_rgb[0] = r;
    dst_rgb[1] = g;
    dst_rgb[2] = b;
    dst_rgb += 3;
    src_argb += 4;
  }
}

void ARGBToRGB565Row_C(const uint8* src_argb, uint8* dst_rgb, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8 b0 = src_argb[0] >> 3;
    uint8 g0 = src_argb[1] >> 2;
    uint8 r0 = src_argb[2] >> 3;
    uint8 b1 = src_argb[4] >> 3;
    uint8 g1 = src_argb[5] >> 2;
    uint8 r1 = src_argb[6] >> 3;
    WRITEWORD(dst_rgb, b0 | (g0 << 5) | (r0 << 11) |
              (b1 << 16) | (g1 << 21) | (r1 << 27));
    dst_rgb += 4;
    src_argb += 8;
  }
  if (width & 1) {
    uint8 b0 = src_argb[0] >> 3;
    uint8 g0 = src_argb[1] >> 2;
    uint8 r0 = src_argb[2] >> 3;
    *(uint16*)(dst_rgb) = b0 | (g0 << 5) | (r0 << 11);
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
void ARGBToRGB565DitherRow_C(const uint8* src_argb, uint8* dst_rgb,
                             const uint32 dither4, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    int dither0 = ((const unsigned char*)(&dither4))[x & 3];
    int dither1 = ((const unsigned char*)(&dither4))[(x + 1) & 3];
    uint8 b0 = clamp255(src_argb[0] + dither0) >> 3;
    uint8 g0 = clamp255(src_argb[1] + dither0) >> 2;
    uint8 r0 = clamp255(src_argb[2] + dither0) >> 3;
    uint8 b1 = clamp255(src_argb[4] + dither1) >> 3;
    uint8 g1 = clamp255(src_argb[5] + dither1) >> 2;
    uint8 r1 = clamp255(src_argb[6] + dither1) >> 3;
    WRITEWORD(dst_rgb, b0 | (g0 << 5) | (r0 << 11) |
              (b1 << 16) | (g1 << 21) | (r1 << 27));
    dst_rgb += 4;
    src_argb += 8;
  }
  if (width & 1) {
    int dither0 = ((const unsigned char*)(&dither4))[(width - 1) & 3];
    uint8 b0 = clamp255(src_argb[0] + dither0) >> 3;
    uint8 g0 = clamp255(src_argb[1] + dither0) >> 2;
    uint8 r0 = clamp255(src_argb[2] + dither0) >> 3;
    *(uint16*)(dst_rgb) = b0 | (g0 << 5) | (r0 << 11);
  }
}

void ARGBToARGB1555Row_C(const uint8* src_argb, uint8* dst_rgb, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8 b0 = src_argb[0] >> 3;
    uint8 g0 = src_argb[1] >> 3;
    uint8 r0 = src_argb[2] >> 3;
    uint8 a0 = src_argb[3] >> 7;
    uint8 b1 = src_argb[4] >> 3;
    uint8 g1 = src_argb[5] >> 3;
    uint8 r1 = src_argb[6] >> 3;
    uint8 a1 = src_argb[7] >> 7;
    *(uint32*)(dst_rgb) =
        b0 | (g0 << 5) | (r0 << 10) | (a0 << 15) |
        (b1 << 16) | (g1 << 21) | (r1 << 26) | (a1 << 31);
    dst_rgb += 4;
    src_argb += 8;
  }
  if (width & 1) {
    uint8 b0 = src_argb[0] >> 3;
    uint8 g0 = src_argb[1] >> 3;
    uint8 r0 = src_argb[2] >> 3;
    uint8 a0 = src_argb[3] >> 7;
    *(uint16*)(dst_rgb) =
        b0 | (g0 << 5) | (r0 << 10) | (a0 << 15);
  }
}

void ARGBToARGB4444Row_C(const uint8* src_argb, uint8* dst_rgb, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8 b0 = src_argb[0] >> 4;
    uint8 g0 = src_argb[1] >> 4;
    uint8 r0 = src_argb[2] >> 4;
    uint8 a0 = src_argb[3] >> 4;
    uint8 b1 = src_argb[4] >> 4;
    uint8 g1 = src_argb[5] >> 4;
    uint8 r1 = src_argb[6] >> 4;
    uint8 a1 = src_argb[7] >> 4;
    *(uint32*)(dst_rgb) =
        b0 | (g0 << 4) | (r0 << 8) | (a0 << 12) |
        (b1 << 16) | (g1 << 20) | (r1 << 24) | (a1 << 28);
    dst_rgb += 4;
    src_argb += 8;
  }
  if (width & 1) {
    uint8 b0 = src_argb[0] >> 4;
    uint8 g0 = src_argb[1] >> 4;
    uint8 r0 = src_argb[2] >> 4;
    uint8 a0 = src_argb[3] >> 4;
    *(uint16*)(dst_rgb) =
        b0 | (g0 << 4) | (r0 << 8) | (a0 << 12);
  }
}

static __inline int RGBToY(uint8 r, uint8 g, uint8 b) {
  return (66 * r + 129 * g +  25 * b + 0x1080) >> 8;
}

static __inline int RGBToU(uint8 r, uint8 g, uint8 b) {
  return (112 * b - 74 * g - 38 * r + 0x8080) >> 8;
}
static __inline int RGBToV(uint8 r, uint8 g, uint8 b) {
  return (112 * r - 94 * g - 18 * b + 0x8080) >> 8;
}

#define MAKEROWY(NAME, R, G, B, BPP) \
void NAME ## ToYRow_C(const uint8* src_argb0, uint8* dst_y, int width) {       \
  int x;                                                                       \
  for (x = 0; x < width; ++x) {                                                \
    dst_y[0] = RGBToY(src_argb0[R], src_argb0[G], src_argb0[B]);               \
    src_argb0 += BPP;                                                          \
    dst_y += 1;                                                                \
  }                                                                            \
}                                                                              \
void NAME ## ToUVRow_C(const uint8* src_rgb0, int src_stride_rgb,              \
                       uint8* dst_u, uint8* dst_v, int width) {                \
  const uint8* src_rgb1 = src_rgb0 + src_stride_rgb;                           \
  int x;                                                                       \
  for (x = 0; x < width - 1; x += 2) {                                         \
    uint8 ab = (src_rgb0[B] + src_rgb0[B + BPP] +                              \
               src_rgb1[B] + src_rgb1[B + BPP]) >> 2;                          \
    uint8 ag = (src_rgb0[G] + src_rgb0[G + BPP] +                              \
               src_rgb1[G] + src_rgb1[G + BPP]) >> 2;                          \
    uint8 ar = (src_rgb0[R] + src_rgb0[R + BPP] +                              \
               src_rgb1[R] + src_rgb1[R + BPP]) >> 2;                          \
    dst_u[0] = RGBToU(ar, ag, ab);                                             \
    dst_v[0] = RGBToV(ar, ag, ab);                                             \
    src_rgb0 += BPP * 2;                                                       \
    src_rgb1 += BPP * 2;                                                       \
    dst_u += 1;                                                                \
    dst_v += 1;                                                                \
  }                                                                            \
  if (width & 1) {                                                             \
    uint8 ab = (src_rgb0[B] + src_rgb1[B]) >> 1;                               \
    uint8 ag = (src_rgb0[G] + src_rgb1[G]) >> 1;                               \
    uint8 ar = (src_rgb0[R] + src_rgb1[R]) >> 1;                               \
    dst_u[0] = RGBToU(ar, ag, ab);                                             \
    dst_v[0] = RGBToV(ar, ag, ab);                                             \
  }                                                                            \
}

MAKEROWY(ARGB, 2, 1, 0, 4)
MAKEROWY(BGRA, 1, 2, 3, 4)
MAKEROWY(ABGR, 0, 1, 2, 4)
MAKEROWY(RGBA, 3, 2, 1, 4)
MAKEROWY(RGB24, 2, 1, 0, 3)
MAKEROWY(RAW, 0, 1, 2, 3)
#undef MAKEROWY

// JPeg uses a variation on BT.601-1 full range
// y =  0.29900 * r + 0.58700 * g + 0.11400 * b
// u = -0.16874 * r - 0.33126 * g + 0.50000 * b  + center
// v =  0.50000 * r - 0.41869 * g - 0.08131 * b  + center
// BT.601 Mpeg range uses:
// b 0.1016 * 255 = 25.908 = 25
// g 0.5078 * 255 = 129.489 = 129
// r 0.2578 * 255 = 65.739 = 66
// JPeg 8 bit Y (not used):
// b 0.11400 * 256 = 29.184 = 29
// g 0.58700 * 256 = 150.272 = 150
// r 0.29900 * 256 = 76.544 = 77
// JPeg 7 bit Y:
// b 0.11400 * 128 = 14.592 = 15
// g 0.58700 * 128 = 75.136 = 75
// r 0.29900 * 128 = 38.272 = 38
// JPeg 8 bit U:
// b  0.50000 * 255 = 127.5 = 127
// g -0.33126 * 255 = -84.4713 = -84
// r -0.16874 * 255 = -43.0287 = -43
// JPeg 8 bit V:
// b -0.08131 * 255 = -20.73405 = -20
// g -0.41869 * 255 = -106.76595 = -107
// r  0.50000 * 255 = 127.5 = 127

static __inline int RGBToYJ(uint8 r, uint8 g, uint8 b) {
  return (38 * r + 75 * g +  15 * b + 64) >> 7;
}

static __inline int RGBToUJ(uint8 r, uint8 g, uint8 b) {
  return (127 * b - 84 * g - 43 * r + 0x8080) >> 8;
}
static __inline int RGBToVJ(uint8 r, uint8 g, uint8 b) {
  return (127 * r - 107 * g - 20 * b + 0x8080) >> 8;
}

#define AVGB(a, b) (((a) + (b) + 1) >> 1)

#define MAKEROWYJ(NAME, R, G, B, BPP) \
void NAME ## ToYJRow_C(const uint8* src_argb0, uint8* dst_y, int width) {      \
  int x;                                                                       \
  for (x = 0; x < width; ++x) {                                                \
    dst_y[0] = RGBToYJ(src_argb0[R], src_argb0[G], src_argb0[B]);              \
    src_argb0 += BPP;                                                          \
    dst_y += 1;                                                                \
  }                                                                            \
}                                                                              \
void NAME ## ToUVJRow_C(const uint8* src_rgb0, int src_stride_rgb,             \
                        uint8* dst_u, uint8* dst_v, int width) {               \
  const uint8* src_rgb1 = src_rgb0 + src_stride_rgb;                           \
  int x;                                                                       \
  for (x = 0; x < width - 1; x += 2) {                                         \
    uint8 ab = AVGB(AVGB(src_rgb0[B], src_rgb1[B]),                            \
                    AVGB(src_rgb0[B + BPP], src_rgb1[B + BPP]));               \
    uint8 ag = AVGB(AVGB(src_rgb0[G], src_rgb1[G]),                            \
                    AVGB(src_rgb0[G + BPP], src_rgb1[G + BPP]));               \
    uint8 ar = AVGB(AVGB(src_rgb0[R], src_rgb1[R]),                            \
                    AVGB(src_rgb0[R + BPP], src_rgb1[R + BPP]));               \
    dst_u[0] = RGBToUJ(ar, ag, ab);                                            \
    dst_v[0] = RGBToVJ(ar, ag, ab);                                            \
    src_rgb0 += BPP * 2;                                                       \
    src_rgb1 += BPP * 2;                                                       \
    dst_u += 1;                                                                \
    dst_v += 1;                                                                \
  }                                                                            \
  if (width & 1) {                                                             \
    uint8 ab = AVGB(src_rgb0[B], src_rgb1[B]);                                 \
    uint8 ag = AVGB(src_rgb0[G], src_rgb1[G]);                                 \
    uint8 ar = AVGB(src_rgb0[R], src_rgb1[R]);                                 \
    dst_u[0] = RGBToUJ(ar, ag, ab);                                            \
    dst_v[0] = RGBToVJ(ar, ag, ab);                                            \
  }                                                                            \
}

MAKEROWYJ(ARGB, 2, 1, 0, 4)
#undef MAKEROWYJ

void RGB565ToYRow_C(const uint8* src_rgb565, uint8* dst_y, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 b = src_rgb565[0] & 0x1f;
    uint8 g = (src_rgb565[0] >> 5) | ((src_rgb565[1] & 0x07) << 3);
    uint8 r = src_rgb565[1] >> 3;
    b = (b << 3) | (b >> 2);
    g = (g << 2) | (g >> 4);
    r = (r << 3) | (r >> 2);
    dst_y[0] = RGBToY(r, g, b);
    src_rgb565 += 2;
    dst_y += 1;
  }
}

void ARGB1555ToYRow_C(const uint8* src_argb1555, uint8* dst_y, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 b = src_argb1555[0] & 0x1f;
    uint8 g = (src_argb1555[0] >> 5) | ((src_argb1555[1] & 0x03) << 3);
    uint8 r = (src_argb1555[1] & 0x7c) >> 2;
    b = (b << 3) | (b >> 2);
    g = (g << 3) | (g >> 2);
    r = (r << 3) | (r >> 2);
    dst_y[0] = RGBToY(r, g, b);
    src_argb1555 += 2;
    dst_y += 1;
  }
}

void ARGB4444ToYRow_C(const uint8* src_argb4444, uint8* dst_y, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 b = src_argb4444[0] & 0x0f;
    uint8 g = src_argb4444[0] >> 4;
    uint8 r = src_argb4444[1] & 0x0f;
    b = (b << 4) | b;
    g = (g << 4) | g;
    r = (r << 4) | r;
    dst_y[0] = RGBToY(r, g, b);
    src_argb4444 += 2;
    dst_y += 1;
  }
}

void RGB565ToUVRow_C(const uint8* src_rgb565, int src_stride_rgb565,
                     uint8* dst_u, uint8* dst_v, int width) {
  const uint8* next_rgb565 = src_rgb565 + src_stride_rgb565;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8 b0 = src_rgb565[0] & 0x1f;
    uint8 g0 = (src_rgb565[0] >> 5) | ((src_rgb565[1] & 0x07) << 3);
    uint8 r0 = src_rgb565[1] >> 3;
    uint8 b1 = src_rgb565[2] & 0x1f;
    uint8 g1 = (src_rgb565[2] >> 5) | ((src_rgb565[3] & 0x07) << 3);
    uint8 r1 = src_rgb565[3] >> 3;
    uint8 b2 = next_rgb565[0] & 0x1f;
    uint8 g2 = (next_rgb565[0] >> 5) | ((next_rgb565[1] & 0x07) << 3);
    uint8 r2 = next_rgb565[1] >> 3;
    uint8 b3 = next_rgb565[2] & 0x1f;
    uint8 g3 = (next_rgb565[2] >> 5) | ((next_rgb565[3] & 0x07) << 3);
    uint8 r3 = next_rgb565[3] >> 3;
    uint8 b = (b0 + b1 + b2 + b3);  // 565 * 4 = 787.
    uint8 g = (g0 + g1 + g2 + g3);
    uint8 r = (r0 + r1 + r2 + r3);
    b = (b << 1) | (b >> 6);  // 787 -> 888.
    r = (r << 1) | (r >> 6);
    dst_u[0] = RGBToU(r, g, b);
    dst_v[0] = RGBToV(r, g, b);
    src_rgb565 += 4;
    next_rgb565 += 4;
    dst_u += 1;
    dst_v += 1;
  }
  if (width & 1) {
    uint8 b0 = src_rgb565[0] & 0x1f;
    uint8 g0 = (src_rgb565[0] >> 5) | ((src_rgb565[1] & 0x07) << 3);
    uint8 r0 = src_rgb565[1] >> 3;
    uint8 b2 = next_rgb565[0] & 0x1f;
    uint8 g2 = (next_rgb565[0] >> 5) | ((next_rgb565[1] & 0x07) << 3);
    uint8 r2 = next_rgb565[1] >> 3;
    uint8 b = (b0 + b2);  // 565 * 2 = 676.
    uint8 g = (g0 + g2);
    uint8 r = (r0 + r2);
    b = (b << 2) | (b >> 4);  // 676 -> 888
    g = (g << 1) | (g >> 6);
    r = (r << 2) | (r >> 4);
    dst_u[0] = RGBToU(r, g, b);
    dst_v[0] = RGBToV(r, g, b);
  }
}

void ARGB1555ToUVRow_C(const uint8* src_argb1555, int src_stride_argb1555,
                       uint8* dst_u, uint8* dst_v, int width) {
  const uint8* next_argb1555 = src_argb1555 + src_stride_argb1555;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8 b0 = src_argb1555[0] & 0x1f;
    uint8 g0 = (src_argb1555[0] >> 5) | ((src_argb1555[1] & 0x03) << 3);
    uint8 r0 = (src_argb1555[1] & 0x7c) >> 2;
    uint8 b1 = src_argb1555[2] & 0x1f;
    uint8 g1 = (src_argb1555[2] >> 5) | ((src_argb1555[3] & 0x03) << 3);
    uint8 r1 = (src_argb1555[3] & 0x7c) >> 2;
    uint8 b2 = next_argb1555[0] & 0x1f;
    uint8 g2 = (next_argb1555[0] >> 5) | ((next_argb1555[1] & 0x03) << 3);
    uint8 r2 = (next_argb1555[1] & 0x7c) >> 2;
    uint8 b3 = next_argb1555[2] & 0x1f;
    uint8 g3 = (next_argb1555[2] >> 5) | ((next_argb1555[3] & 0x03) << 3);
    uint8 r3 = (next_argb1555[3] & 0x7c) >> 2;
    uint8 b = (b0 + b1 + b2 + b3);  // 555 * 4 = 777.
    uint8 g = (g0 + g1 + g2 + g3);
    uint8 r = (r0 + r1 + r2 + r3);
    b = (b << 1) | (b >> 6);  // 777 -> 888.
    g = (g << 1) | (g >> 6);
    r = (r << 1) | (r >> 6);
    dst_u[0] = RGBToU(r, g, b);
    dst_v[0] = RGBToV(r, g, b);
    src_argb1555 += 4;
    next_argb1555 += 4;
    dst_u += 1;
    dst_v += 1;
  }
  if (width & 1) {
    uint8 b0 = src_argb1555[0] & 0x1f;
    uint8 g0 = (src_argb1555[0] >> 5) | ((src_argb1555[1] & 0x03) << 3);
    uint8 r0 = (src_argb1555[1] & 0x7c) >> 2;
    uint8 b2 = next_argb1555[0] & 0x1f;
    uint8 g2 = (next_argb1555[0] >> 5) | ((next_argb1555[1] & 0x03) << 3);
    uint8 r2 = next_argb1555[1] >> 3;
    uint8 b = (b0 + b2);  // 555 * 2 = 666.
    uint8 g = (g0 + g2);
    uint8 r = (r0 + r2);
    b = (b << 2) | (b >> 4);  // 666 -> 888.
    g = (g << 2) | (g >> 4);
    r = (r << 2) | (r >> 4);
    dst_u[0] = RGBToU(r, g, b);
    dst_v[0] = RGBToV(r, g, b);
  }
}

void ARGB4444ToUVRow_C(const uint8* src_argb4444, int src_stride_argb4444,
                       uint8* dst_u, uint8* dst_v, int width) {
  const uint8* next_argb4444 = src_argb4444 + src_stride_argb4444;
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8 b0 = src_argb4444[0] & 0x0f;
    uint8 g0 = src_argb4444[0] >> 4;
    uint8 r0 = src_argb4444[1] & 0x0f;
    uint8 b1 = src_argb4444[2] & 0x0f;
    uint8 g1 = src_argb4444[2] >> 4;
    uint8 r1 = src_argb4444[3] & 0x0f;
    uint8 b2 = next_argb4444[0] & 0x0f;
    uint8 g2 = next_argb4444[0] >> 4;
    uint8 r2 = next_argb4444[1] & 0x0f;
    uint8 b3 = next_argb4444[2] & 0x0f;
    uint8 g3 = next_argb4444[2] >> 4;
    uint8 r3 = next_argb4444[3] & 0x0f;
    uint8 b = (b0 + b1 + b2 + b3);  // 444 * 4 = 666.
    uint8 g = (g0 + g1 + g2 + g3);
    uint8 r = (r0 + r1 + r2 + r3);
    b = (b << 2) | (b >> 4);  // 666 -> 888.
    g = (g << 2) | (g >> 4);
    r = (r << 2) | (r >> 4);
    dst_u[0] = RGBToU(r, g, b);
    dst_v[0] = RGBToV(r, g, b);
    src_argb4444 += 4;
    next_argb4444 += 4;
    dst_u += 1;
    dst_v += 1;
  }
  if (width & 1) {
    uint8 b0 = src_argb4444[0] & 0x0f;
    uint8 g0 = src_argb4444[0] >> 4;
    uint8 r0 = src_argb4444[1] & 0x0f;
    uint8 b2 = next_argb4444[0] & 0x0f;
    uint8 g2 = next_argb4444[0] >> 4;
    uint8 r2 = next_argb4444[1] & 0x0f;
    uint8 b = (b0 + b2);  // 444 * 2 = 555.
    uint8 g = (g0 + g2);
    uint8 r = (r0 + r2);
    b = (b << 3) | (b >> 2);  // 555 -> 888.
    g = (g << 3) | (g >> 2);
    r = (r << 3) | (r >> 2);
    dst_u[0] = RGBToU(r, g, b);
    dst_v[0] = RGBToV(r, g, b);
  }
}

void ARGBToUV444Row_C(const uint8* src_argb,
                      uint8* dst_u, uint8* dst_v, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 ab = src_argb[0];
    uint8 ag = src_argb[1];
    uint8 ar = src_argb[2];
    dst_u[0] = RGBToU(ar, ag, ab);
    dst_v[0] = RGBToV(ar, ag, ab);
    src_argb += 4;
    dst_u += 1;
    dst_v += 1;
  }
}

void ARGBToUV411Row_C(const uint8* src_argb,
                      uint8* dst_u, uint8* dst_v, int width) {
  int x;
  for (x = 0; x < width - 3; x += 4) {
    uint8 ab = (src_argb[0] + src_argb[4] + src_argb[8] + src_argb[12]) >> 2;
    uint8 ag = (src_argb[1] + src_argb[5] + src_argb[9] + src_argb[13]) >> 2;
    uint8 ar = (src_argb[2] + src_argb[6] + src_argb[10] + src_argb[14]) >> 2;
    dst_u[0] = RGBToU(ar, ag, ab);
    dst_v[0] = RGBToV(ar, ag, ab);
    src_argb += 16;
    dst_u += 1;
    dst_v += 1;
  }
  // Odd width handling mimics 'any' function which replicates last pixel.
  if ((width & 3) == 3) {
    uint8 ab = (src_argb[0] + src_argb[4] + src_argb[8] + src_argb[8]) >> 2;
    uint8 ag = (src_argb[1] + src_argb[5] + src_argb[9] + src_argb[9]) >> 2;
    uint8 ar = (src_argb[2] + src_argb[6] + src_argb[10] + src_argb[10]) >> 2;
    dst_u[0] = RGBToU(ar, ag, ab);
    dst_v[0] = RGBToV(ar, ag, ab);
  } else if ((width & 3) == 2) {
    uint8 ab = (src_argb[0] + src_argb[4]) >> 1;
    uint8 ag = (src_argb[1] + src_argb[5]) >> 1;
    uint8 ar = (src_argb[2] + src_argb[6]) >> 1;
    dst_u[0] = RGBToU(ar, ag, ab);
    dst_v[0] = RGBToV(ar, ag, ab);
  } else if ((width & 3) == 1) {
    uint8 ab = src_argb[0];
    uint8 ag = src_argb[1];
    uint8 ar = src_argb[2];
    dst_u[0] = RGBToU(ar, ag, ab);
    dst_v[0] = RGBToV(ar, ag, ab);
  }
}

void ARGBGrayRow_C(const uint8* src_argb, uint8* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    uint8 y = RGBToYJ(src_argb[2], src_argb[1], src_argb[0]);
    dst_argb[2] = dst_argb[1] = dst_argb[0] = y;
    dst_argb[3] = src_argb[3];
    dst_argb += 4;
    src_argb += 4;
  }
}

// Convert a row of image to Sepia tone.
void ARGBSepiaRow_C(uint8* dst_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    int b = dst_argb[0];
    int g = dst_argb[1];
    int r = dst_argb[2];
    int sb = (b * 17 + g * 68 + r * 35) >> 7;
    int sg = (b * 22 + g * 88 + r * 45) >> 7;
    int sr = (b * 24 + g * 98 + r * 50) >> 7;
    // b does not over flow. a is preserved from original.
    dst_argb[0] = sb;
    dst_argb[1] = clamp255(sg);
    dst_argb[2] = clamp255(sr);
    dst_argb += 4;
  }
}

// Apply color matrix to a row of image. Matrix is signed.
// TODO(fbarchard): Consider adding rounding (+32).
void ARGBColorMatrixRow_C(const uint8* src_argb, uint8* dst_argb,
                          const int8* matrix_argb, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    int b = src_argb[0];
    int g = src_argb[1];
    int r = src_argb[2];
    int a = src_argb[3];
    int sb = (b * matrix_argb[0] + g * matrix_argb[1] +
              r * matrix_argb[2] + a * matrix_argb[3]) >> 6;
    int sg = (b * matrix_argb[4] + g * matrix_argb[5] +
              r * matrix_argb[6] + a * matrix_argb[7]) >> 6;
    int sr = (b * matrix_argb[8] + g * matrix_argb[9] +
              r * matrix_argb[10] + a * matrix_argb[11]) >> 6;
    int sa = (b * matrix_argb[12] + g * matrix_argb[13] +
              r * matrix_argb[14] + a * matrix_argb[15]) >> 6;
    dst_argb[0] = Clamp(sb);
    dst_argb[1] = Clamp(sg);
    dst_argb[2] = Clamp(sr);
    dst_argb[3] = Clamp(sa);
    src_argb += 4;
    dst_argb += 4;
  }
}

// Apply color table to a row of image.
void ARGBColorTableRow_C(uint8* dst_argb, const uint8* table_argb, int width) {
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
void RGBColorTableRow_C(uint8* dst_argb, const uint8* table_argb, int width) {
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

void ARGBQuantizeRow_C(uint8* dst_argb, int scale, int interval_size,
                       int interval_offset, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    int b = dst_argb[0];
    int g = dst_argb[1];
    int r = dst_argb[2];
    dst_argb[0] = (b * scale >> 16) * interval_size + interval_offset;
    dst_argb[1] = (g * scale >> 16) * interval_size + interval_offset;
    dst_argb[2] = (r * scale >> 16) * interval_size + interval_offset;
    dst_argb += 4;
  }
}

#define REPEAT8(v) (v) | ((v) << 8)
#define SHADE(f, v) v * f >> 24

void ARGBShadeRow_C(const uint8* src_argb, uint8* dst_argb, int width,
                    uint32 value) {
  const uint32 b_scale = REPEAT8(value & 0xff);
  const uint32 g_scale = REPEAT8((value >> 8) & 0xff);
  const uint32 r_scale = REPEAT8((value >> 16) & 0xff);
  const uint32 a_scale = REPEAT8(value >> 24);

  int i;
  for (i = 0; i < width; ++i) {
    const uint32 b = REPEAT8(src_argb[0]);
    const uint32 g = REPEAT8(src_argb[1]);
    const uint32 r = REPEAT8(src_argb[2]);
    const uint32 a = REPEAT8(src_argb[3]);
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

#define REPEAT8(v) (v) | ((v) << 8)
#define SHADE(f, v) v * f >> 16

void ARGBMultiplyRow_C(const uint8* src_argb0, const uint8* src_argb1,
                       uint8* dst_argb, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    const uint32 b = REPEAT8(src_argb0[0]);
    const uint32 g = REPEAT8(src_argb0[1]);
    const uint32 r = REPEAT8(src_argb0[2]);
    const uint32 a = REPEAT8(src_argb0[3]);
    const uint32 b_scale = src_argb1[0];
    const uint32 g_scale = src_argb1[1];
    const uint32 r_scale = src_argb1[2];
    const uint32 a_scale = src_argb1[3];
    dst_argb[0] = SHADE(b, b_scale);
    dst_argb[1] = SHADE(g, g_scale);
    dst_argb[2] = SHADE(r, r_scale);
    dst_argb[3] = SHADE(a, a_scale);
    src_argb0 += 4;
    src_argb1 += 4;
    dst_argb += 4;
  }
}
#undef REPEAT8
#undef SHADE

#define SHADE(f, v) clamp255(v + f)

void ARGBAddRow_C(const uint8* src_argb0, const uint8* src_argb1,
                  uint8* dst_argb, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    const int b = src_argb0[0];
    const int g = src_argb0[1];
    const int r = src_argb0[2];
    const int a = src_argb0[3];
    const int b_add = src_argb1[0];
    const int g_add = src_argb1[1];
    const int r_add = src_argb1[2];
    const int a_add = src_argb1[3];
    dst_argb[0] = SHADE(b, b_add);
    dst_argb[1] = SHADE(g, g_add);
    dst_argb[2] = SHADE(r, r_add);
    dst_argb[3] = SHADE(a, a_add);
    src_argb0 += 4;
    src_argb1 += 4;
    dst_argb += 4;
  }
}
#undef SHADE

#define SHADE(f, v) clamp0(f - v)

void ARGBSubtractRow_C(const uint8* src_argb0, const uint8* src_argb1,
                       uint8* dst_argb, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    const int b = src_argb0[0];
    const int g = src_argb0[1];
    const int r = src_argb0[2];
    const int a = src_argb0[3];
    const int b_sub = src_argb1[0];
    const int g_sub = src_argb1[1];
    const int r_sub = src_argb1[2];
    const int a_sub = src_argb1[3];
    dst_argb[0] = SHADE(b, b_sub);
    dst_argb[1] = SHADE(g, g_sub);
    dst_argb[2] = SHADE(r, r_sub);
    dst_argb[3] = SHADE(a, a_sub);
    src_argb0 += 4;
    src_argb1 += 4;
    dst_argb += 4;
  }
}
#undef SHADE

// Sobel functions which mimics SSSE3.
void SobelXRow_C(const uint8* src_y0, const uint8* src_y1, const uint8* src_y2,
                 uint8* dst_sobelx, int width) {
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
    dst_sobelx[i] = (uint8)(clamp255(sobel));
  }
}

void SobelYRow_C(const uint8* src_y0, const uint8* src_y1,
                 uint8* dst_sobely, int width) {
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
    dst_sobely[i] = (uint8)(clamp255(sobel));
  }
}

void SobelRow_C(const uint8* src_sobelx, const uint8* src_sobely,
                uint8* dst_argb, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    int r = src_sobelx[i];
    int b = src_sobely[i];
    int s = clamp255(r + b);
    dst_argb[0] = (uint8)(s);
    dst_argb[1] = (uint8)(s);
    dst_argb[2] = (uint8)(s);
    dst_argb[3] = (uint8)(255u);
    dst_argb += 4;
  }
}

void SobelToPlaneRow_C(const uint8* src_sobelx, const uint8* src_sobely,
                       uint8* dst_y, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    int r = src_sobelx[i];
    int b = src_sobely[i];
    int s = clamp255(r + b);
    dst_y[i] = (uint8)(s);
  }
}

void SobelXYRow_C(const uint8* src_sobelx, const uint8* src_sobely,
                  uint8* dst_argb, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    int r = src_sobelx[i];
    int b = src_sobely[i];
    int g = clamp255(r + b);
    dst_argb[0] = (uint8)(b);
    dst_argb[1] = (uint8)(g);
    dst_argb[2] = (uint8)(r);
    dst_argb[3] = (uint8)(255u);
    dst_argb += 4;
  }
}

void J400ToARGBRow_C(const uint8* src_y, uint8* dst_argb, int width) {
  // Copy a Y to RGB.
  int x;
  for (x = 0; x < width; ++x) {
    uint8 y = src_y[0];
    dst_argb[2] = dst_argb[1] = dst_argb[0] = y;
    dst_argb[3] = 255u;
    dst_argb += 4;
    ++src_y;
  }
}

// TODO(fbarchard): Unify these structures to be platform independent.
// TODO(fbarchard): Generate SIMD structures from float matrix.

// BT.601 YUV to RGB reference
//  R = (Y - 16) * 1.164              - V * -1.596
//  G = (Y - 16) * 1.164 - U *  0.391 - V *  0.813
//  B = (Y - 16) * 1.164 - U * -2.018

// Y contribution to R,G,B.  Scale and bias.
#define YG 18997 /* round(1.164 * 64 * 256 * 256 / 257) */
#define YGB -1160 /* 1.164 * 64 * -16 + 64 / 2 */

// U and V contributions to R,G,B.
#define UB -128 /* max(-128, round(-2.018 * 64)) */
#define UG 25 /* round(0.391 * 64) */
#define VG 52 /* round(0.813 * 64) */
#define VR -102 /* round(-1.596 * 64) */

// Bias values to subtract 16 from Y and 128 from U and V.
#define BB (UB * 128            + YGB)
#define BG (UG * 128 + VG * 128 + YGB)
#define BR            (VR * 128 + YGB)

#if defined(__aarch64__)  // 64 bit arm
const struct YuvConstants SIMD_ALIGNED(kYuvI601Constants) = {
  { -UB, -VR, -UB, -VR, -UB, -VR, -UB, -VR },
  { -UB, -VR, -UB, -VR, -UB, -VR, -UB, -VR },
  { UG, VG, UG, VG, UG, VG, UG, VG },
  { UG, VG, UG, VG, UG, VG, UG, VG },
  { BB, BG, BR, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
const struct YuvConstants SIMD_ALIGNED(kYvuI601Constants) = {
  { -VR, -UB, -VR, -UB, -VR, -UB, -VR, -UB },
  { -VR, -UB, -VR, -UB, -VR, -UB, -VR, -UB },
  { VG, UG, VG, UG, VG, UG, VG, UG },
  { VG, UG, VG, UG, VG, UG, VG, UG },
  { BR, BG, BB, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
#elif defined(__arm__)  // 32 bit arm
const struct YuvConstants SIMD_ALIGNED(kYuvI601Constants) = {
  { -UB, -UB, -UB, -UB, -VR, -VR, -VR, -VR, 0, 0, 0, 0, 0, 0, 0, 0 },
  { UG, UG, UG, UG, VG, VG, VG, VG, 0, 0, 0, 0, 0, 0, 0, 0 },
  { BB, BG, BR, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
const struct YuvConstants SIMD_ALIGNED(kYvuI601Constants) = {
  { -VR, -VR, -VR, -VR, -UB, -UB, -UB, -UB, 0, 0, 0, 0, 0, 0, 0, 0 },
  { VG, VG, VG, VG, UG, UG, UG, UG, 0, 0, 0, 0, 0, 0, 0, 0 },
  { BR, BG, BB, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
#else
const struct YuvConstants SIMD_ALIGNED(kYuvI601Constants) = {
  { UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0,
    UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0 },
  { UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG,
    UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG },
  { 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR,
    0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR },
  { BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB },
  { BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG },
  { BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR },
  { YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG }
};
const struct YuvConstants SIMD_ALIGNED(kYvuI601Constants) = {
  { VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0,
    VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0 },
  { VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG,
    VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG },
  { 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB,
    0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB },
  { BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR },
  { BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG },
  { BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB },
  { YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG }
};
#endif

#undef BB
#undef BG
#undef BR
#undef YGB
#undef UB
#undef UG
#undef VG
#undef VR
#undef YG

// JPEG YUV to RGB reference
// *  R = Y                - V * -1.40200
// *  G = Y - U *  0.34414 - V *  0.71414
// *  B = Y - U * -1.77200

// Y contribution to R,G,B.  Scale and bias.
#define YG 16320 /* round(1.000 * 64 * 256 * 256 / 257) */
#define YGB 32  /* 64 / 2 */

// U and V contributions to R,G,B.
#define UB -113 /* round(-1.77200 * 64) */
#define UG 22 /* round(0.34414 * 64) */
#define VG 46 /* round(0.71414  * 64) */
#define VR -90 /* round(-1.40200 * 64) */

// Bias values to round, and subtract 128 from U and V.
#define BB (UB * 128            + YGB)
#define BG (UG * 128 + VG * 128 + YGB)
#define BR            (VR * 128 + YGB)

#if defined(__aarch64__)
const struct YuvConstants SIMD_ALIGNED(kYuvJPEGConstants) = {
  { -UB, -VR, -UB, -VR, -UB, -VR, -UB, -VR },
  { -UB, -VR, -UB, -VR, -UB, -VR, -UB, -VR },
  { UG, VG, UG, VG, UG, VG, UG, VG },
  { UG, VG, UG, VG, UG, VG, UG, VG },
  { BB, BG, BR, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
const struct YuvConstants SIMD_ALIGNED(kYvuJPEGConstants) = {
  { -VR, -UB, -VR, -UB, -VR, -UB, -VR, -UB },
  { -VR, -UB, -VR, -UB, -VR, -UB, -VR, -UB },
  { VG, UG, VG, UG, VG, UG, VG, UG },
  { VG, UG, VG, UG, VG, UG, VG, UG },
  { BR, BG, BB, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
#elif defined(__arm__)
const struct YuvConstants SIMD_ALIGNED(kYuvJPEGConstants) = {
  { -UB, -UB, -UB, -UB, -VR, -VR, -VR, -VR, 0, 0, 0, 0, 0, 0, 0, 0 },
  { UG, UG, UG, UG, VG, VG, VG, VG, 0, 0, 0, 0, 0, 0, 0, 0 },
  { BB, BG, BR, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
const struct YuvConstants SIMD_ALIGNED(kYvuJPEGConstants) = {
  { -VR, -VR, -VR, -VR, -UB, -UB, -UB, -UB, 0, 0, 0, 0, 0, 0, 0, 0 },
  { VG, VG, VG, VG, UG, UG, UG, UG, 0, 0, 0, 0, 0, 0, 0, 0 },
  { BR, BG, BB, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
#else
const struct YuvConstants SIMD_ALIGNED(kYuvJPEGConstants) = {
  { UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0,
    UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0 },
  { UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG,
    UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG },
  { 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR,
    0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR },
  { BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB },
  { BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG },
  { BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR },
  { YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG }
};
const struct YuvConstants SIMD_ALIGNED(kYvuJPEGConstants) = {
  { VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0,
    VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0 },
  { VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG,
    VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG },
  { 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB,
    0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB },
  { BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR },
  { BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG },
  { BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB },
  { YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG }
};
#endif

#undef BB
#undef BG
#undef BR
#undef YGB
#undef UB
#undef UG
#undef VG
#undef VR
#undef YG

// BT.709 YUV to RGB reference
// *  R = Y                - V * -1.28033
// *  G = Y - U *  0.21482 - V *  0.38059
// *  B = Y - U * -2.12798

// Y contribution to R,G,B.  Scale and bias.
#define YG 16320 /* round(1.000 * 64 * 256 * 256 / 257) */
#define YGB 32  /* 64 / 2 */

// TODO(fbarchard): Find way to express 2.12 instead of 2.0.
// U and V contributions to R,G,B.
#define UB -128 /* max(-128, round(-2.12798 * 64)) */
#define UG 14 /* round(0.21482 * 64) */
#define VG 24 /* round(0.38059  * 64) */
#define VR -82 /* round(-1.28033 * 64) */

// Bias values to round, and subtract 128 from U and V.
#define BB (UB * 128            + YGB)
#define BG (UG * 128 + VG * 128 + YGB)
#define BR            (VR * 128 + YGB)

#if defined(__aarch64__)
const struct YuvConstants SIMD_ALIGNED(kYuvH709Constants) = {
  { -UB, -VR, -UB, -VR, -UB, -VR, -UB, -VR },
  { -UB, -VR, -UB, -VR, -UB, -VR, -UB, -VR },
  { UG, VG, UG, VG, UG, VG, UG, VG },
  { UG, VG, UG, VG, UG, VG, UG, VG },
  { BB, BG, BR, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
const struct YuvConstants SIMD_ALIGNED(kYvuH709Constants) = {
  { -VR, -UB, -VR, -UB, -VR, -UB, -VR, -UB },
  { -VR, -UB, -VR, -UB, -VR, -UB, -VR, -UB },
  { VG, UG, VG, UG, VG, UG, VG, UG },
  { VG, UG, VG, UG, VG, UG, VG, UG },
  { BR, BG, BB, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
#elif defined(__arm__)
const struct YuvConstants SIMD_ALIGNED(kYuvH709Constants) = {
  { -UB, -UB, -UB, -UB, -VR, -VR, -VR, -VR, 0, 0, 0, 0, 0, 0, 0, 0 },
  { UG, UG, UG, UG, VG, VG, VG, VG, 0, 0, 0, 0, 0, 0, 0, 0 },
  { BB, BG, BR, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
const struct YuvConstants SIMD_ALIGNED(kYvuH709Constants) = {
  { -VR, -VR, -VR, -VR, -UB, -UB, -UB, -UB, 0, 0, 0, 0, 0, 0, 0, 0 },
  { VG, VG, VG, VG, UG, UG, UG, UG, 0, 0, 0, 0, 0, 0, 0, 0 },
  { BR, BG, BB, 0, 0, 0, 0, 0 },
  { 0x0101 * YG, 0, 0, 0 }
};
#else
const struct YuvConstants SIMD_ALIGNED(kYuvH709Constants) = {
  { UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0,
    UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0 },
  { UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG,
    UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG },
  { 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR,
    0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR },
  { BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB },
  { BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG },
  { BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR },
  { YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG }
};
const struct YuvConstants SIMD_ALIGNED(kYvuH709Constants) = {
  { VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0,
    VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0, VR, 0 },
  { VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG,
    VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG, VG, UG },
  { 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB,
    0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB, 0, UB },
  { BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR, BR },
  { BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG, BG },
  { BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB, BB },
  { YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG, YG }
};
#endif

#undef BB
#undef BG
#undef BR
#undef YGB
#undef UB
#undef UG
#undef VG
#undef VR
#undef YG

// C reference code that mimics the YUV assembly.
static __inline void YuvPixel(uint8 y, uint8 u, uint8 v,
                              uint8* b, uint8* g, uint8* r,
                              const struct YuvConstants* yuvconstants) {
#if defined(__aarch64__)
  int ub = -yuvconstants->kUVToRB[0];
  int ug = yuvconstants->kUVToG[0];
  int vg = yuvconstants->kUVToG[1];
  int vr = -yuvconstants->kUVToRB[1];
  int bb = yuvconstants->kUVBiasBGR[0];
  int bg = yuvconstants->kUVBiasBGR[1];
  int br = yuvconstants->kUVBiasBGR[2];
  int yg = yuvconstants->kYToRgb[0] / 0x0101;
#elif defined(__arm__)
  int ub = -yuvconstants->kUVToRB[0];
  int ug = yuvconstants->kUVToG[0];
  int vg = yuvconstants->kUVToG[4];
  int vr = -yuvconstants->kUVToRB[4];
  int bb = yuvconstants->kUVBiasBGR[0];
  int bg = yuvconstants->kUVBiasBGR[1];
  int br = yuvconstants->kUVBiasBGR[2];
  int yg = yuvconstants->kYToRgb[0] / 0x0101;
#else
  int ub = yuvconstants->kUVToB[0];
  int ug = yuvconstants->kUVToG[0];
  int vg = yuvconstants->kUVToG[1];
  int vr = yuvconstants->kUVToR[1];
  int bb = yuvconstants->kUVBiasB[0];
  int bg = yuvconstants->kUVBiasG[0];
  int br = yuvconstants->kUVBiasR[0];
  int yg = yuvconstants->kYToRgb[0];
#endif

  uint32 y1 = (uint32)(y * 0x0101 * yg) >> 16;
  *b = Clamp((int32)(-(u * ub)          + y1 + bb) >> 6);
  *g = Clamp((int32)(-(u * ug + v * vg) + y1 + bg) >> 6);
  *r = Clamp((int32)         (-(v * vr) + y1 + br) >> 6);
}

// Y contribution to R,G,B.  Scale and bias.
#define YG 18997 /* round(1.164 * 64 * 256 * 256 / 257) */
#define YGB -1160 /* 1.164 * 64 * -16 + 64 / 2 */

// C reference code that mimics the YUV assembly.
static __inline void YPixel(uint8 y, uint8* b, uint8* g, uint8* r) {
  uint32 y1 = (uint32)(y * 0x0101 * YG) >> 16;
  *b = Clamp((int32)(y1 + YGB) >> 6);
  *g = Clamp((int32)(y1 + YGB) >> 6);
  *r = Clamp((int32)(y1 + YGB) >> 6);
}

#undef YG
#undef YGB

#if !defined(LIBYUV_DISABLE_NEON) && \
    (defined(__ARM_NEON__) || defined(__aarch64__) || defined(LIBYUV_NEON))
// C mimic assembly.
// TODO(fbarchard): Remove subsampling from Neon.
void I444ToARGBRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint8 u = (src_u[0] + src_u[1] + 1) >> 1;
    uint8 v = (src_v[0] + src_v[1] + 1) >> 1;
    YuvPixel(src_y[0], u, v, rgb_buf + 0, rgb_buf + 1, rgb_buf + 2,
             yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_y[1], u, v, rgb_buf + 4, rgb_buf + 5, rgb_buf + 6,
             yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    src_u += 2;
    src_v += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}
#else
void I444ToARGBRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width; ++x) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    src_y += 1;
    src_u += 1;
    src_v += 1;
    rgb_buf += 4;  // Advance 1 pixel.
  }
}
#endif

// Also used for 420
void I422ToARGBRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_y[1], src_u[0], src_v[0],
             rgb_buf + 4, rgb_buf + 5, rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void I422AlphaToARGBRow_C(const uint8* src_y,
                          const uint8* src_u,
                          const uint8* src_v,
                          const uint8* src_a,
                          uint8* rgb_buf,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = src_a[0];
    YuvPixel(src_y[1], src_u[0], src_v[0],
             rgb_buf + 4, rgb_buf + 5, rgb_buf + 6, yuvconstants);
    rgb_buf[7] = src_a[1];
    src_y += 2;
    src_u += 1;
    src_v += 1;
    src_a += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = src_a[0];
  }
}

void I422ToRGB24Row_C(const uint8* src_y,
                      const uint8* src_u,
                      const uint8* src_v,
                      uint8* rgb_buf,
                      const struct YuvConstants* yuvconstants,
                      int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    YuvPixel(src_y[1], src_u[0], src_v[0],
             rgb_buf + 3, rgb_buf + 4, rgb_buf + 5, yuvconstants);
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 6;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
  }
}

void I422ToARGB4444Row_C(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb4444,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  uint8 b0;
  uint8 g0;
  uint8 r0;
  uint8 b1;
  uint8 g1;
  uint8 r1;
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
    *(uint32*)(dst_argb4444) = b0 | (g0 << 4) | (r0 << 8) |
        (b1 << 16) | (g1 << 20) | (r1 << 24) | 0xf000f000;
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
    *(uint16*)(dst_argb4444) = b0 | (g0 << 4) | (r0 << 8) |
        0xf000;
  }
}

void I422ToARGB1555Row_C(const uint8* src_y,
                         const uint8* src_u,
                         const uint8* src_v,
                         uint8* dst_argb1555,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  uint8 b0;
  uint8 g0;
  uint8 r0;
  uint8 b1;
  uint8 g1;
  uint8 r1;
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
    *(uint32*)(dst_argb1555) = b0 | (g0 << 5) | (r0 << 10) |
        (b1 << 16) | (g1 << 21) | (r1 << 26) | 0x80008000;
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
    *(uint16*)(dst_argb1555) = b0 | (g0 << 5) | (r0 << 10) |
        0x8000;
  }
}

void I422ToRGB565Row_C(const uint8* src_y,
                       const uint8* src_u,
                       const uint8* src_v,
                       uint8* dst_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  uint8 b0;
  uint8 g0;
  uint8 r0;
  uint8 b1;
  uint8 g1;
  uint8 r1;
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
    *(uint32*)(dst_rgb565) = b0 | (g0 << 5) | (r0 << 11) |
        (b1 << 16) | (g1 << 21) | (r1 << 27);
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
    *(uint16*)(dst_rgb565) = b0 | (g0 << 5) | (r0 << 11);
  }
}

void I411ToARGBRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 3; x += 4) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_y[1], src_u[0], src_v[0],
             rgb_buf + 4, rgb_buf + 5, rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    YuvPixel(src_y[2], src_u[0], src_v[0],
             rgb_buf + 8, rgb_buf + 9, rgb_buf + 10, yuvconstants);
    rgb_buf[11] = 255;
    YuvPixel(src_y[3], src_u[0], src_v[0],
             rgb_buf + 12, rgb_buf + 13, rgb_buf + 14, yuvconstants);
    rgb_buf[15] = 255;
    src_y += 4;
    src_u += 1;
    src_v += 1;
    rgb_buf += 16;  // Advance 4 pixels.
  }
  if (width & 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_y[1], src_u[0], src_v[0],
             rgb_buf + 4, rgb_buf + 5, rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void NV12ToARGBRow_C(const uint8* src_y,
                     const uint8* src_uv,
                     uint8* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_uv[0], src_uv[1],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_y[1], src_uv[0], src_uv[1],
             rgb_buf + 4, rgb_buf + 5, rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    src_uv += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_uv[0], src_uv[1],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void NV21ToARGBRow_C(const uint8* src_y,
                     const uint8* src_vu,
                     uint8* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_vu[1], src_vu[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_y[1], src_vu[1], src_vu[0],
             rgb_buf + 4, rgb_buf + 5, rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_y += 2;
    src_vu += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_vu[1], src_vu[0],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void NV12ToRGB565Row_C(const uint8* src_y,
                       const uint8* src_uv,
                       uint8* dst_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width) {
  uint8 b0;
  uint8 g0;
  uint8 r0;
  uint8 b1;
  uint8 g1;
  uint8 r1;
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
    *(uint32*)(dst_rgb565) = b0 | (g0 << 5) | (r0 << 11) |
        (b1 << 16) | (g1 << 21) | (r1 << 27);
    src_y += 2;
    src_uv += 2;
    dst_rgb565 += 4;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_uv[0], src_uv[1], &b0, &g0, &r0, yuvconstants);
    b0 = b0 >> 3;
    g0 = g0 >> 2;
    r0 = r0 >> 3;
    *(uint16*)(dst_rgb565) = b0 | (g0 << 5) | (r0 << 11);
  }
}

void YUY2ToARGBRow_C(const uint8* src_yuy2,
                     uint8* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_yuy2[0], src_yuy2[1], src_yuy2[3],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_yuy2[2], src_yuy2[1], src_yuy2[3],
             rgb_buf + 4, rgb_buf + 5, rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_yuy2 += 4;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_yuy2[0], src_yuy2[1], src_yuy2[3],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void UYVYToARGBRow_C(const uint8* src_uyvy,
                     uint8* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_uyvy[1], src_uyvy[0], src_uyvy[2],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
    YuvPixel(src_uyvy[3], src_uyvy[0], src_uyvy[2],
             rgb_buf + 4, rgb_buf + 5, rgb_buf + 6, yuvconstants);
    rgb_buf[7] = 255;
    src_uyvy += 4;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_uyvy[1], src_uyvy[0], src_uyvy[2],
             rgb_buf + 0, rgb_buf + 1, rgb_buf + 2, yuvconstants);
    rgb_buf[3] = 255;
  }
}

void I422ToRGBARow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* rgb_buf,
                     const struct YuvConstants* yuvconstants,
                     int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 1, rgb_buf + 2, rgb_buf + 3, yuvconstants);
    rgb_buf[0] = 255;
    YuvPixel(src_y[1], src_u[0], src_v[0],
             rgb_buf + 5, rgb_buf + 6, rgb_buf + 7, yuvconstants);
    rgb_buf[4] = 255;
    src_y += 2;
    src_u += 1;
    src_v += 1;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YuvPixel(src_y[0], src_u[0], src_v[0],
             rgb_buf + 1, rgb_buf + 2, rgb_buf + 3, yuvconstants);
    rgb_buf[0] = 255;
  }
}

void I400ToARGBRow_C(const uint8* src_y, uint8* rgb_buf, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    YPixel(src_y[0], rgb_buf + 0, rgb_buf + 1, rgb_buf + 2);
    rgb_buf[3] = 255;
    YPixel(src_y[1], rgb_buf + 4, rgb_buf + 5, rgb_buf + 6);
    rgb_buf[7] = 255;
    src_y += 2;
    rgb_buf += 8;  // Advance 2 pixels.
  }
  if (width & 1) {
    YPixel(src_y[0], rgb_buf + 0, rgb_buf + 1, rgb_buf + 2);
    rgb_buf[3] = 255;
  }
}

void MirrorRow_C(const uint8* src, uint8* dst, int width) {
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

void MirrorUVRow_C(const uint8* src_uv, uint8* dst_u, uint8* dst_v, int width) {
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

void ARGBMirrorRow_C(const uint8* src, uint8* dst, int width) {
  int x;
  const uint32* src32 = (const uint32*)(src);
  uint32* dst32 = (uint32*)(dst);
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

void SplitUVRow_C(const uint8* src_uv, uint8* dst_u, uint8* dst_v, int width) {
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

void MergeUVRow_C(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
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

void CopyRow_C(const uint8* src, uint8* dst, int count) {
  memcpy(dst, src, count);
}

void CopyRow_16_C(const uint16* src, uint16* dst, int count) {
  memcpy(dst, src, count * 2);
}

void SetRow_C(uint8* dst, uint8 v8, int width) {
  memset(dst, v8, width);
}

void ARGBSetRow_C(uint8* dst_argb, uint32 v32, int width) {
  uint32* d = (uint32*)(dst_argb);
  int x;
  for (x = 0; x < width; ++x) {
    d[x] = v32;
  }
}

// Filter 2 rows of YUY2 UV's (422) into U and V (420).
void YUY2ToUVRow_C(const uint8* src_yuy2, int src_stride_yuy2,
                   uint8* dst_u, uint8* dst_v, int width) {
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

// Copy row of YUY2 UV's (422) into U and V (422).
void YUY2ToUV422Row_C(const uint8* src_yuy2,
                      uint8* dst_u, uint8* dst_v, int width) {
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
void YUY2ToYRow_C(const uint8* src_yuy2, uint8* dst_y, int width) {
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
void UYVYToUVRow_C(const uint8* src_uyvy, int src_stride_uyvy,
                   uint8* dst_u, uint8* dst_v, int width) {
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
void UYVYToUV422Row_C(const uint8* src_uyvy,
                      uint8* dst_u, uint8* dst_v, int width) {
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
void UYVYToYRow_C(const uint8* src_uyvy, uint8* dst_y, int width) {
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

#define BLEND(f, b, a) (((256 - a) * b) >> 8) + f

// Blend src_argb0 over src_argb1 and store to dst_argb.
// dst_argb may be src_argb0 or src_argb1.
// This code mimics the SSSE3 version for better testability.
void ARGBBlendRow_C(const uint8* src_argb0, const uint8* src_argb1,
                    uint8* dst_argb, int width) {
  int x;
  for (x = 0; x < width - 1; x += 2) {
    uint32 fb = src_argb0[0];
    uint32 fg = src_argb0[1];
    uint32 fr = src_argb0[2];
    uint32 a = src_argb0[3];
    uint32 bb = src_argb1[0];
    uint32 bg = src_argb1[1];
    uint32 br = src_argb1[2];
    dst_argb[0] = BLEND(fb, bb, a);
    dst_argb[1] = BLEND(fg, bg, a);
    dst_argb[2] = BLEND(fr, br, a);
    dst_argb[3] = 255u;

    fb = src_argb0[4 + 0];
    fg = src_argb0[4 + 1];
    fr = src_argb0[4 + 2];
    a = src_argb0[4 + 3];
    bb = src_argb1[4 + 0];
    bg = src_argb1[4 + 1];
    br = src_argb1[4 + 2];
    dst_argb[4 + 0] = BLEND(fb, bb, a);
    dst_argb[4 + 1] = BLEND(fg, bg, a);
    dst_argb[4 + 2] = BLEND(fr, br, a);
    dst_argb[4 + 3] = 255u;
    src_argb0 += 8;
    src_argb1 += 8;
    dst_argb += 8;
  }

  if (width & 1) {
    uint32 fb = src_argb0[0];
    uint32 fg = src_argb0[1];
    uint32 fr = src_argb0[2];
    uint32 a = src_argb0[3];
    uint32 bb = src_argb1[0];
    uint32 bg = src_argb1[1];
    uint32 br = src_argb1[2];
    dst_argb[0] = BLEND(fb, bb, a);
    dst_argb[1] = BLEND(fg, bg, a);
    dst_argb[2] = BLEND(fr, br, a);
    dst_argb[3] = 255u;
  }
}
#undef BLEND

#define UBLEND(f, b, a) (((a) * f) + ((255 - a) * b) + 255) >> 8
void BlendPlaneRow_C(const uint8* src0, const uint8* src1,
                     const uint8* alpha, uint8* dst, int width) {
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

#define ATTENUATE(f, a) (a | (a << 8)) * (f | (f << 8)) >> 24

// Multiply source RGB by alpha and store to destination.
// This code mimics the SSSE3 version for better testability.
void ARGBAttenuateRow_C(const uint8* src_argb, uint8* dst_argb, int width) {
  int i;
  for (i = 0; i < width - 1; i += 2) {
    uint32 b = src_argb[0];
    uint32 g = src_argb[1];
    uint32 r = src_argb[2];
    uint32 a = src_argb[3];
    dst_argb[0] = ATTENUATE(b, a);
    dst_argb[1] = ATTENUATE(g, a);
    dst_argb[2] = ATTENUATE(r, a);
    dst_argb[3] = a;
    b = src_argb[4];
    g = src_argb[5];
    r = src_argb[6];
    a = src_argb[7];
    dst_argb[4] = ATTENUATE(b, a);
    dst_argb[5] = ATTENUATE(g, a);
    dst_argb[6] = ATTENUATE(r, a);
    dst_argb[7] = a;
    src_argb += 8;
    dst_argb += 8;
  }

  if (width & 1) {
    const uint32 b = src_argb[0];
    const uint32 g = src_argb[1];
    const uint32 r = src_argb[2];
    const uint32 a = src_argb[3];
    dst_argb[0] = ATTENUATE(b, a);
    dst_argb[1] = ATTENUATE(g, a);
    dst_argb[2] = ATTENUATE(r, a);
    dst_argb[3] = a;
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
const uint32 fixed_invtbl8[256] = {
  0x01000000, 0x0100ffff, T(0x02), T(0x03), T(0x04), T(0x05), T(0x06), T(0x07),
  T(0x08), T(0x09), T(0x0a), T(0x0b), T(0x0c), T(0x0d), T(0x0e), T(0x0f),
  T(0x10), T(0x11), T(0x12), T(0x13), T(0x14), T(0x15), T(0x16), T(0x17),
  T(0x18), T(0x19), T(0x1a), T(0x1b), T(0x1c), T(0x1d), T(0x1e), T(0x1f),
  T(0x20), T(0x21), T(0x22), T(0x23), T(0x24), T(0x25), T(0x26), T(0x27),
  T(0x28), T(0x29), T(0x2a), T(0x2b), T(0x2c), T(0x2d), T(0x2e), T(0x2f),
  T(0x30), T(0x31), T(0x32), T(0x33), T(0x34), T(0x35), T(0x36), T(0x37),
  T(0x38), T(0x39), T(0x3a), T(0x3b), T(0x3c), T(0x3d), T(0x3e), T(0x3f),
  T(0x40), T(0x41), T(0x42), T(0x43), T(0x44), T(0x45), T(0x46), T(0x47),
  T(0x48), T(0x49), T(0x4a), T(0x4b), T(0x4c), T(0x4d), T(0x4e), T(0x4f),
  T(0x50), T(0x51), T(0x52), T(0x53), T(0x54), T(0x55), T(0x56), T(0x57),
  T(0x58), T(0x59), T(0x5a), T(0x5b), T(0x5c), T(0x5d), T(0x5e), T(0x5f),
  T(0x60), T(0x61), T(0x62), T(0x63), T(0x64), T(0x65), T(0x66), T(0x67),
  T(0x68), T(0x69), T(0x6a), T(0x6b), T(0x6c), T(0x6d), T(0x6e), T(0x6f),
  T(0x70), T(0x71), T(0x72), T(0x73), T(0x74), T(0x75), T(0x76), T(0x77),
  T(0x78), T(0x79), T(0x7a), T(0x7b), T(0x7c), T(0x7d), T(0x7e), T(0x7f),
  T(0x80), T(0x81), T(0x82), T(0x83), T(0x84), T(0x85), T(0x86), T(0x87),
  T(0x88), T(0x89), T(0x8a), T(0x8b), T(0x8c), T(0x8d), T(0x8e), T(0x8f),
  T(0x90), T(0x91), T(0x92), T(0x93), T(0x94), T(0x95), T(0x96), T(0x97),
  T(0x98), T(0x99), T(0x9a), T(0x9b), T(0x9c), T(0x9d), T(0x9e), T(0x9f),
  T(0xa0), T(0xa1), T(0xa2), T(0xa3), T(0xa4), T(0xa5), T(0xa6), T(0xa7),
  T(0xa8), T(0xa9), T(0xaa), T(0xab), T(0xac), T(0xad), T(0xae), T(0xaf),
  T(0xb0), T(0xb1), T(0xb2), T(0xb3), T(0xb4), T(0xb5), T(0xb6), T(0xb7),
  T(0xb8), T(0xb9), T(0xba), T(0xbb), T(0xbc), T(0xbd), T(0xbe), T(0xbf),
  T(0xc0), T(0xc1), T(0xc2), T(0xc3), T(0xc4), T(0xc5), T(0xc6), T(0xc7),
  T(0xc8), T(0xc9), T(0xca), T(0xcb), T(0xcc), T(0xcd), T(0xce), T(0xcf),
  T(0xd0), T(0xd1), T(0xd2), T(0xd3), T(0xd4), T(0xd5), T(0xd6), T(0xd7),
  T(0xd8), T(0xd9), T(0xda), T(0xdb), T(0xdc), T(0xdd), T(0xde), T(0xdf),
  T(0xe0), T(0xe1), T(0xe2), T(0xe3), T(0xe4), T(0xe5), T(0xe6), T(0xe7),
  T(0xe8), T(0xe9), T(0xea), T(0xeb), T(0xec), T(0xed), T(0xee), T(0xef),
  T(0xf0), T(0xf1), T(0xf2), T(0xf3), T(0xf4), T(0xf5), T(0xf6), T(0xf7),
  T(0xf8), T(0xf9), T(0xfa), T(0xfb), T(0xfc), T(0xfd), T(0xfe), 0x01000100 };
#undef T

void ARGBUnattenuateRow_C(const uint8* src_argb, uint8* dst_argb, int width) {
  int i;
  for (i = 0; i < width; ++i) {
    uint32 b = src_argb[0];
    uint32 g = src_argb[1];
    uint32 r = src_argb[2];
    const uint32 a = src_argb[3];
    const uint32 ia = fixed_invtbl8[a] & 0xffff;  // 8.8 fixed point
    b = (b * ia) >> 8;
    g = (g * ia) >> 8;
    r = (r * ia) >> 8;
    // Clamping should not be necessary but is free in assembly.
    dst_argb[0] = clamp255(b);
    dst_argb[1] = clamp255(g);
    dst_argb[2] = clamp255(r);
    dst_argb[3] = a;
    src_argb += 4;
    dst_argb += 4;
  }
}

void ComputeCumulativeSumRow_C(const uint8* row, int32* cumsum,
                               const int32* previous_cumsum, int width) {
  int32 row_sum[4] = {0, 0, 0, 0};
  int x;
  for (x = 0; x < width; ++x) {
    row_sum[0] += row[x * 4 + 0];
    row_sum[1] += row[x * 4 + 1];
    row_sum[2] += row[x * 4 + 2];
    row_sum[3] += row[x * 4 + 3];
    cumsum[x * 4 + 0] = row_sum[0]  + previous_cumsum[x * 4 + 0];
    cumsum[x * 4 + 1] = row_sum[1]  + previous_cumsum[x * 4 + 1];
    cumsum[x * 4 + 2] = row_sum[2]  + previous_cumsum[x * 4 + 2];
    cumsum[x * 4 + 3] = row_sum[3]  + previous_cumsum[x * 4 + 3];
  }
}

void CumulativeSumToAverageRow_C(const int32* tl, const int32* bl,
                                int w, int area, uint8* dst, int count) {
  float ooa = 1.0f / area;
  int i;
  for (i = 0; i < count; ++i) {
    dst[0] = (uint8)((bl[w + 0] + tl[0] - bl[0] - tl[w + 0]) * ooa);
    dst[1] = (uint8)((bl[w + 1] + tl[1] - bl[1] - tl[w + 1]) * ooa);
    dst[2] = (uint8)((bl[w + 2] + tl[2] - bl[2] - tl[w + 2]) * ooa);
    dst[3] = (uint8)((bl[w + 3] + tl[3] - bl[3] - tl[w + 3]) * ooa);
    dst += 4;
    tl += 4;
    bl += 4;
  }
}

// Copy pixels from rotated source to destination row with a slope.
LIBYUV_API
void ARGBAffineRow_C(const uint8* src_argb, int src_argb_stride,
                     uint8* dst_argb, const float* uv_dudv, int width) {
  int i;
  // Render a row of pixels from source into a buffer.
  float uv[2];
  uv[0] = uv_dudv[0];
  uv[1] = uv_dudv[1];
  for (i = 0; i < width; ++i) {
    int x = (int)(uv[0]);
    int y = (int)(uv[1]);
    *(uint32*)(dst_argb) =
        *(const uint32*)(src_argb + y * src_argb_stride +
                                         x * 4);
    dst_argb += 4;
    uv[0] += uv_dudv[2];
    uv[1] += uv_dudv[3];
  }
}

// Blend 2 rows into 1.
static void HalfRow_C(const uint8* src_uv, ptrdiff_t src_uv_stride,
                      uint8* dst_uv, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_uv[x] = (src_uv[x] + src_uv[src_uv_stride + x] + 1) >> 1;
  }
}

static void HalfRow_16_C(const uint16* src_uv, ptrdiff_t src_uv_stride,
                         uint16* dst_uv, int width) {
  int x;
  for (x = 0; x < width; ++x) {
    dst_uv[x] = (src_uv[x] + src_uv[src_uv_stride + x] + 1) >> 1;
  }
}

// C version 2x2 -> 2x1.
void InterpolateRow_C(uint8* dst_ptr, const uint8* src_ptr,
                      ptrdiff_t src_stride,
                      int width, int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint8* src_ptr1 = src_ptr + src_stride;
  int x;
  if (y1_fraction == 0) {
    memcpy(dst_ptr, src_ptr, width);
    return;
  }
  if (y1_fraction == 128) {
    HalfRow_C(src_ptr, src_stride, dst_ptr, width);
    return;
  }
  for (x = 0; x < width - 1; x += 2) {
    dst_ptr[0] =
        (src_ptr[0] * y0_fraction + src_ptr1[0] * y1_fraction + 128) >> 8;
    dst_ptr[1] =
        (src_ptr[1] * y0_fraction + src_ptr1[1] * y1_fraction + 128) >> 8;
    src_ptr += 2;
    src_ptr1 += 2;
    dst_ptr += 2;
  }
  if (width & 1) {
    dst_ptr[0] =
        (src_ptr[0] * y0_fraction + src_ptr1[0] * y1_fraction + 128) >> 8;
  }
}

void InterpolateRow_16_C(uint16* dst_ptr, const uint16* src_ptr,
                         ptrdiff_t src_stride,
                         int width, int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint16* src_ptr1 = src_ptr + src_stride;
  int x;
  if (source_y_fraction == 0) {
    memcpy(dst_ptr, src_ptr, width * 2);
    return;
  }
  if (source_y_fraction == 128) {
    HalfRow_16_C(src_ptr, src_stride, dst_ptr, width);
    return;
  }
  for (x = 0; x < width - 1; x += 2) {
    dst_ptr[0] = (src_ptr[0] * y0_fraction + src_ptr1[0] * y1_fraction) >> 8;
    dst_ptr[1] = (src_ptr[1] * y0_fraction + src_ptr1[1] * y1_fraction) >> 8;
    src_ptr += 2;
    src_ptr1 += 2;
    dst_ptr += 2;
  }
  if (width & 1) {
    dst_ptr[0] = (src_ptr[0] * y0_fraction + src_ptr1[0] * y1_fraction) >> 8;
  }
}

// Use first 4 shuffler values to reorder ARGB channels.
void ARGBShuffleRow_C(const uint8* src_argb, uint8* dst_argb,
                      const uint8* shuffler, int width) {
  int index0 = shuffler[0];
  int index1 = shuffler[1];
  int index2 = shuffler[2];
  int index3 = shuffler[3];
  // Shuffle a row of ARGB.
  int x;
  for (x = 0; x < width; ++x) {
    // To support in-place conversion.
    uint8 b = src_argb[index0];
    uint8 g = src_argb[index1];
    uint8 r = src_argb[index2];
    uint8 a = src_argb[index3];
    dst_argb[0] = b;
    dst_argb[1] = g;
    dst_argb[2] = r;
    dst_argb[3] = a;
    src_argb += 4;
    dst_argb += 4;
  }
}

void I422ToYUY2Row_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* dst_frame, int width) {
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

void I422ToUYVYRow_C(const uint8* src_y,
                     const uint8* src_u,
                     const uint8* src_v,
                     uint8* dst_frame, int width) {
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


void ARGBPolynomialRow_C(const uint8* src_argb,
                         uint8* dst_argb,
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

    dst_argb[0] = Clamp((int32)(db));
    dst_argb[1] = Clamp((int32)(dg));
    dst_argb[2] = Clamp((int32)(dr));
    dst_argb[3] = Clamp((int32)(da));
    src_argb += 4;
    dst_argb += 4;
  }
}

void ARGBLumaColorTableRow_C(const uint8* src_argb, uint8* dst_argb, int width,
                             const uint8* luma, uint32 lumacoeff) {
  uint32 bc = lumacoeff & 0xff;
  uint32 gc = (lumacoeff >> 8) & 0xff;
  uint32 rc = (lumacoeff >> 16) & 0xff;

  int i;
  for (i = 0; i < width - 1; i += 2) {
    // Luminance in rows, color values in columns.
    const uint8* luma0 = ((src_argb[0] * bc + src_argb[1] * gc +
                           src_argb[2] * rc) & 0x7F00u) + luma;
    const uint8* luma1;
    dst_argb[0] = luma0[src_argb[0]];
    dst_argb[1] = luma0[src_argb[1]];
    dst_argb[2] = luma0[src_argb[2]];
    dst_argb[3] = src_argb[3];
    luma1 = ((src_argb[4] * bc + src_argb[5] * gc +
              src_argb[6] * rc) & 0x7F00u) + luma;
    dst_argb[4] = luma1[src_argb[4]];
    dst_argb[5] = luma1[src_argb[5]];
    dst_argb[6] = luma1[src_argb[6]];
    dst_argb[7] = src_argb[7];
    src_argb += 8;
    dst_argb += 8;
  }
  if (width & 1) {
    // Luminance in rows, color values in columns.
    const uint8* luma0 = ((src_argb[0] * bc + src_argb[1] * gc +
                           src_argb[2] * rc) & 0x7F00u) + luma;
    dst_argb[0] = luma0[src_argb[0]];
    dst_argb[1] = luma0[src_argb[1]];
    dst_argb[2] = luma0[src_argb[2]];
    dst_argb[3] = src_argb[3];
  }
}

void ARGBCopyAlphaRow_C(const uint8* src, uint8* dst, int width) {
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

void ARGBExtractAlphaRow_C(const uint8* src_argb, uint8* dst_a, int width) {
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

void ARGBCopyYToAlphaRow_C(const uint8* src, uint8* dst, int width) {
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

#if !(defined(_MSC_VER) && defined(_M_IX86)) && \
    defined(HAS_I422TORGB565ROW_SSSE3)
// row_win.cc has asm version, but GCC uses 2 step wrapper.
void I422ToRGB565Row_SSSE3(const uint8* src_y,
                           const uint8* src_u,
                           const uint8* src_v,
                           uint8* dst_rgb565,
                           const struct YuvConstants* yuvconstants,
                           int width) {
  SIMD_ALIGNED(uint8 row[MAXTWIDTH * 4]);
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
void I422ToARGB1555Row_SSSE3(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_argb1555,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8 row[MAXTWIDTH * 4]);
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
void I422ToARGB4444Row_SSSE3(const uint8* src_y,
                             const uint8* src_u,
                             const uint8* src_v,
                             uint8* dst_argb4444,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8 row[MAXTWIDTH * 4]);
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
void NV12ToRGB565Row_SSSE3(const uint8* src_y,
                           const uint8* src_uv,
                           uint8* dst_rgb565,
                           const struct YuvConstants* yuvconstants,
                           int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8 row[MAXTWIDTH * 4]);
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

#if defined(HAS_I422TORGB565ROW_AVX2)
void I422ToRGB565Row_AVX2(const uint8* src_y,
                          const uint8* src_u,
                          const uint8* src_v,
                          uint8* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  SIMD_ALIGNED(uint8 row[MAXTWIDTH * 4]);
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
void I422ToARGB1555Row_AVX2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb1555,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8 row[MAXTWIDTH * 4]);
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
void I422ToARGB4444Row_AVX2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_argb4444,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8 row[MAXTWIDTH * 4]);
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
void I422ToRGB24Row_AVX2(const uint8* src_y,
                            const uint8* src_u,
                            const uint8* src_v,
                            uint8* dst_rgb24,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8 row[MAXTWIDTH * 4]);
  while (width > 0) {
    int twidth = width > MAXTWIDTH ? MAXTWIDTH : width;
    I422ToARGBRow_AVX2(src_y, src_u, src_v, row, yuvconstants, twidth);
    // TODO(fbarchard): ARGBToRGB24Row_AVX2
    ARGBToRGB24Row_SSSE3(row, dst_rgb24, twidth);
    src_y += twidth;
    src_u += twidth / 2;
    src_v += twidth / 2;
    dst_rgb24 += twidth * 3;
    width -= twidth;
  }
}
#endif

#if defined(HAS_NV12TORGB565ROW_AVX2)
void NV12ToRGB565Row_AVX2(const uint8* src_y,
                          const uint8* src_uv,
                          uint8* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  // Row buffer for intermediate ARGB pixels.
  SIMD_ALIGNED(uint8 row[MAXTWIDTH * 4]);
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

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
