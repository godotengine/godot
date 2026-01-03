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
#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// This module is for GCC x86 and x64.
#if !defined(LIBYUV_DISABLE_X86) &&               \
    (defined(__x86_64__) || defined(__i386__)) && \
    !defined(LIBYUV_ENABLE_ROWWIN)

#if defined(HAS_ARGBTOYROW_SSSE3) || defined(HAS_ARGBGRAYROW_SSSE3)

// Constants for ARGB
static const uvec8 kARGBToY = {25u, 129u, 66u, 0u, 25u, 129u, 66u, 0u,
                               25u, 129u, 66u, 0u, 25u, 129u, 66u, 0u};

// JPeg full range.
static const uvec8 kARGBToYJ = {29u, 150u, 77u, 0u, 29u, 150u, 77u, 0u,
                                29u, 150u, 77u, 0u, 29u, 150u, 77u, 0u};

static const uvec8 kABGRToYJ = {77u, 150u, 29u, 0u, 77u, 150u, 29u, 0u,
                                77u, 150u, 29u, 0u, 77u, 150u, 29u, 0u};

static const uvec8 kRGBAToYJ = {0u, 29u, 150u, 77u, 0u, 29u, 150u, 77u,
                                0u, 29u, 150u, 77u, 0u, 29u, 150u, 77u};
#endif  // defined(HAS_ARGBTOYROW_SSSE3) || defined(HAS_ARGBGRAYROW_SSSE3)

#if defined(HAS_ARGBTOYROW_SSSE3) || defined(HAS_I422TOARGBROW_SSSE3)
// Constants for BGRA
static const uvec8 kBGRAToY = {0u, 66u, 129u, 25u, 0u, 66u, 129u, 25u,
                               0u, 66u, 129u, 25u, 0u, 66u, 129u, 25u};

// Constants for ABGR
static const uvec8 kABGRToY = {66u, 129u, 25u, 0u, 66u, 129u, 25u, 0u,
                               66u, 129u, 25u, 0u, 66u, 129u, 25u, 0u};

// Constants for RGBA.
static const uvec8 kRGBAToY = {0u, 25u, 129u, 66u, 0u, 25u, 129u, 66u,
                               0u, 25u, 129u, 66u, 0u, 25u, 129u, 66u};
// 126 (7e) - (-109..110) = 16..235
static const uvec16 kAddY16 = {0x7e80u, 0x7e80u, 0x7e80u, 0x7e80u,
                               0x7e80u, 0x7e80u, 0x7e80u, 0x7e80u};
static const uvec16 kAddY0 = {0x8080u, 0x8080u, 0x8080u, 0x8080u,
                              0x8080u, 0x8080u, 0x8080u, 0x8080u};

static const uvec16 kAddUV128 = {0x8000u, 0x8000u, 0x8000u, 0x8000u,
                                 0x8000u, 0x8000u, 0x8000u, 0x8000u};

static const uvec16 kSub128 = {0x8080u, 0x8080u, 0x8080u, 0x8080u,
                               0x8080u, 0x8080u, 0x8080u, 0x8080u};

#endif  // defined(HAS_ARGBTOYROW_SSSE3) || defined(HAS_I422TOARGBROW_SSSE3)

#ifdef HAS_RGB24TOARGBROW_SSSE3

// Shuffle table for converting RGB24 to ARGB.
static const uvec8 kShuffleMaskRGB24ToARGB = {
    0u, 1u, 2u, 12u, 3u, 4u, 5u, 13u, 6u, 7u, 8u, 14u, 9u, 10u, 11u, 15u};

// Shuffle table for converting RAW to ARGB.
static const uvec8 kShuffleMaskRAWToARGB = {
    2u, 1u, 0u, 128u, 5u, 4u, 3u, 128u, 8u, 7u, 6u, 128u, 11u, 10u, 9u, 128u};
// Shuffle table for converting RAW to ARGB.  Last 12
static const uvec8 kShuffleMaskRAWToARGB_0 = {6u,  5u,   4u,  128u, 9u,  8u,
                                              7u,  128u, 12u, 11u,  10u, 128u,
                                              15u, 14u,  13u, 128u};

// Shuffle table for converting RAW to RGBA.
static const uvec8 kShuffleMaskRAWToRGBA = {
    128u, 2u, 1u, 0u, 128u, 5u, 4u, 3u, 128u, 8u, 7u, 6u, 128u, 11u, 10u, 9u};

// Shuffle table for converting RAW to RGB24.  First 8.
static const uvec8 kShuffleMaskRAWToRGB24_0 = {
    2u,   1u,   0u,   5u,   4u,   3u,   8u,   7u,
    128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u};

// Shuffle table for converting RAW to RGB24.  Middle 8.
static const uvec8 kShuffleMaskRAWToRGB24_1 = {
    2u,   7u,   6u,   5u,   10u,  9u,   8u,   13u,
    128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u};

// Shuffle table for converting RAW to RGB24.  Last 8.
static const uvec8 kShuffleMaskRAWToRGB24_2 = {
    8u,   7u,   12u,  11u,  10u,  15u,  14u,  13u,
    128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u};

// Shuffle table for converting ARGB to RGB24.
static const uvec8 kShuffleMaskARGBToRGB24 = {
    0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 10u, 12u, 13u, 14u, 128u, 128u, 128u, 128u};

// Shuffle table for converting ARGB to RAW.
static const uvec8 kShuffleMaskARGBToRAW = {
    2u, 1u, 0u, 6u, 5u, 4u, 10u, 9u, 8u, 14u, 13u, 12u, 128u, 128u, 128u, 128u};

// Shuffle table for converting ARGBToRGB24 for I422ToRGB24.  First 8 + next 4
static const uvec8 kShuffleMaskARGBToRGB24_0 = {
    0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 128u, 128u, 128u, 128u, 10u, 12u, 13u, 14u};

// YUY2 shuf 16 Y to 32 Y.
static const vec8 kShuffleYUY2Y = {0, 0, 2,  2,  4,  4,  6,  6,
                                   8, 8, 10, 10, 12, 12, 14, 14};

// YUY2 shuf 8 UV to 16 UV.
static const vec8 kShuffleYUY2UV = {1, 3,  1, 3,  5,  7,  5,  7,
                                    9, 11, 9, 11, 13, 15, 13, 15};

// UYVY shuf 16 Y to 32 Y.
static const vec8 kShuffleUYVYY = {1, 1, 3,  3,  5,  5,  7,  7,
                                   9, 9, 11, 11, 13, 13, 15, 15};

// UYVY shuf 8 UV to 16 UV.
static const vec8 kShuffleUYVYUV = {0, 2,  0, 2,  4,  6,  4,  6,
                                    8, 10, 8, 10, 12, 14, 12, 14};

// NV21 shuf 8 VU to 16 UV.
static const lvec8 kShuffleNV21 = {
    1, 0, 1, 0, 3, 2, 3, 2, 5, 4, 5, 4, 7, 6, 7, 6,
    1, 0, 1, 0, 3, 2, 3, 2, 5, 4, 5, 4, 7, 6, 7, 6,
};
#endif  // HAS_RGB24TOARGBROW_SSSE3

#ifdef HAS_J400TOARGBROW_SSE2
void J400ToARGBRow_SSE2(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "pslld       $0x18,%%xmm5                  \n"

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"
      "lea         0x8(%0),%0                    \n"
      "punpcklbw   %%xmm0,%%xmm0                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklwd   %%xmm0,%%xmm0                 \n"
      "punpckhwd   %%xmm1,%%xmm1                 \n"
      "por         %%xmm5,%%xmm0                 \n"
      "por         %%xmm5,%%xmm1                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_y),     // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm5");
}
#endif  // HAS_J400TOARGBROW_SSE2

#ifdef HAS_RGB24TOARGBROW_SSSE3
void RGB24ToARGBRow_SSSE3(const uint8_t* src_rgb24,
                          uint8_t* dst_argb,
                          int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"  // 0xff000000
      "pslld       $0x18,%%xmm5                  \n"
      "movdqa      %3,%%xmm4                     \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm3               \n"
      "lea         0x30(%0),%0                   \n"
      "movdqa      %%xmm3,%%xmm2                 \n"
      "palignr     $0x8,%%xmm1,%%xmm2            \n"
      "pshufb      %%xmm4,%%xmm2                 \n"
      "por         %%xmm5,%%xmm2                 \n"
      "palignr     $0xc,%%xmm0,%%xmm1            \n"
      "pshufb      %%xmm4,%%xmm0                 \n"
      "movdqu      %%xmm2,0x20(%1)               \n"
      "por         %%xmm5,%%xmm0                 \n"
      "pshufb      %%xmm4,%%xmm1                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "por         %%xmm5,%%xmm1                 \n"
      "palignr     $0x4,%%xmm3,%%xmm3            \n"
      "pshufb      %%xmm4,%%xmm3                 \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "por         %%xmm5,%%xmm3                 \n"
      "movdqu      %%xmm3,0x30(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_rgb24),              // %0
        "+r"(dst_argb),               // %1
        "+r"(width)                   // %2
      : "m"(kShuffleMaskRGB24ToARGB)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

void RAWToARGBRow_SSSE3(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  asm volatile(
      "pcmpeqb     %%xmm6,%%xmm6                 \n"  // 0xff000000
      "pslld       $0x18,%%xmm6                  \n"
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      12(%0),%%xmm1                 \n"
      "movdqu      24(%0),%%xmm2                 \n"
      "movdqu      32(%0),%%xmm3                 \n"
      "lea         0x30(%0),%0                   \n"
      "pshufb      %%xmm4,%%xmm0                 \n"
      "pshufb      %%xmm4,%%xmm1                 \n"
      "pshufb      %%xmm4,%%xmm2                 \n"
      "pshufb      %%xmm5,%%xmm3                 \n"
      "por         %%xmm6,%%xmm0                 \n"
      "por         %%xmm6,%%xmm1                 \n"
      "por         %%xmm6,%%xmm2                 \n"
      "por         %%xmm6,%%xmm3                 \n"
      "movdqu      %%xmm0,0x00(%1)               \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "movdqu      %%xmm2,0x20(%1)               \n"
      "movdqu      %%xmm3,0x30(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_raw),                // %0
        "+r"(dst_argb),               // %1
        "+r"(width)                   // %2
      : "m"(kShuffleMaskRAWToARGB),   // %3
        "m"(kShuffleMaskRAWToARGB_0)  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}

void RAWToARGBRow_AVX2(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  asm volatile(
      "vpcmpeqb    %%ymm6,%%ymm6,%%ymm6          \n"  // 0xff000000
      "vpslld      $0x18,%%ymm6,%%ymm6           \n"
      "vbroadcastf128 %3,%%ymm4                  \n"  //
      "vbroadcastf128 %4,%%ymm5                  \n"  //

      LABELALIGN  //
      "1:          \n"
      "vmovdqu     (%0),%%xmm0                   \n"  // first 12
      "vinserti128 $1,12(%0),%%ymm0,%%ymm0       \n"  // second 12
      "vmovdqu     24(%0),%%xmm1                 \n"  // third 12
      "vinserti128 $1,36(%0),%%ymm1,%%ymm1       \n"  // forth 12
      "vmovdqu     48(%0),%%xmm2                 \n"  // fifth 12
      "vinserti128 $1,60(%0),%%ymm2,%%ymm2       \n"  // sixth 12
      "vmovdqu     68(%0),%%xmm3                 \n"  // seventh 12
      "vinserti128 $1,80(%0),%%ymm3,%%ymm3       \n"  // eighth 12
      "lea         96(%0),%0                     \n"
      "vpshufb     %%ymm4,%%ymm0,%%ymm0          \n"
      "vpshufb     %%ymm4,%%ymm1,%%ymm1          \n"
      "vpshufb     %%ymm4,%%ymm2,%%ymm2          \n"
      "vpshufb     %%ymm5,%%ymm3,%%ymm3          \n"
      "vpor        %%ymm6,%%ymm0,%%ymm0          \n"
      "vpor        %%ymm6,%%ymm1,%%ymm1          \n"
      "vpor        %%ymm6,%%ymm2,%%ymm2          \n"
      "vpor        %%ymm6,%%ymm3,%%ymm3          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "vmovdqu     %%ymm2,0x40(%1)               \n"
      "vmovdqu     %%ymm3,0x60(%1)               \n"
      "lea         0x80(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_raw),                // %0
        "+r"(dst_argb),               // %1
        "+r"(width)                   // %2
      : "m"(kShuffleMaskRAWToARGB),   // %3
        "m"(kShuffleMaskRAWToARGB_0)  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}

// Same code as RAWToARGB with different shuffler and A in low bits
void RAWToRGBARow_SSSE3(const uint8_t* src_raw, uint8_t* dst_rgba, int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"  // 0x000000ff
      "psrld       $0x18,%%xmm5                  \n"
      "movdqa      %3,%%xmm4                     \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm3               \n"
      "lea         0x30(%0),%0                   \n"
      "movdqa      %%xmm3,%%xmm2                 \n"
      "palignr     $0x8,%%xmm1,%%xmm2            \n"
      "pshufb      %%xmm4,%%xmm2                 \n"
      "por         %%xmm5,%%xmm2                 \n"
      "palignr     $0xc,%%xmm0,%%xmm1            \n"
      "pshufb      %%xmm4,%%xmm0                 \n"
      "movdqu      %%xmm2,0x20(%1)               \n"
      "por         %%xmm5,%%xmm0                 \n"
      "pshufb      %%xmm4,%%xmm1                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "por         %%xmm5,%%xmm1                 \n"
      "palignr     $0x4,%%xmm3,%%xmm3            \n"
      "pshufb      %%xmm4,%%xmm3                 \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "por         %%xmm5,%%xmm3                 \n"
      "movdqu      %%xmm3,0x30(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_raw),              // %0
        "+r"(dst_rgba),             // %1
        "+r"(width)                 // %2
      : "m"(kShuffleMaskRAWToRGBA)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

void RAWToRGB24Row_SSSE3(const uint8_t* src_raw,
                         uint8_t* dst_rgb24,
                         int width) {
  asm volatile(
      "movdqa      %3,%%xmm3                     \n"
      "movdqa      %4,%%xmm4                     \n"
      "movdqa      %5,%%xmm5                     \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x4(%0),%%xmm1                \n"
      "movdqu      0x8(%0),%%xmm2                \n"
      "lea         0x18(%0),%0                   \n"
      "pshufb      %%xmm3,%%xmm0                 \n"
      "pshufb      %%xmm4,%%xmm1                 \n"
      "pshufb      %%xmm5,%%xmm2                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "movq        %%xmm1,0x8(%1)                \n"
      "movq        %%xmm2,0x10(%1)               \n"
      "lea         0x18(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_raw),                  // %0
        "+r"(dst_rgb24),                // %1
        "+r"(width)                     // %2
      : "m"(kShuffleMaskRAWToRGB24_0),  // %3
        "m"(kShuffleMaskRAWToRGB24_1),  // %4
        "m"(kShuffleMaskRAWToRGB24_2)   // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

void RGB565ToARGBRow_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "mov         $0x1080108,%%eax              \n"
      "movd        %%eax,%%xmm5                  \n"
      "pshufd      $0x0,%%xmm5,%%xmm5            \n"
      "mov         $0x20802080,%%eax             \n"
      "movd        %%eax,%%xmm6                  \n"
      "pshufd      $0x0,%%xmm6,%%xmm6            \n"
      "pcmpeqb     %%xmm3,%%xmm3                 \n"
      "psllw       $0xb,%%xmm3                   \n"
      "pcmpeqb     %%xmm4,%%xmm4                 \n"
      "psllw       $0xa,%%xmm4                   \n"
      "psrlw       $0x5,%%xmm4                   \n"
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psllw       $0x8,%%xmm7                   \n"
      "sub         %0,%1                         \n"
      "sub         %0,%1                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "pand        %%xmm3,%%xmm1                 \n"
      "psllw       $0xb,%%xmm2                   \n"
      "pmulhuw     %%xmm5,%%xmm1                 \n"
      "pmulhuw     %%xmm5,%%xmm2                 \n"
      "psllw       $0x8,%%xmm1                   \n"
      "por         %%xmm2,%%xmm1                 \n"
      "pand        %%xmm4,%%xmm0                 \n"
      "pmulhuw     %%xmm6,%%xmm0                 \n"
      "por         %%xmm7,%%xmm0                 \n"
      "movdqa      %%xmm1,%%xmm2                 \n"
      "punpcklbw   %%xmm0,%%xmm1                 \n"
      "punpckhbw   %%xmm0,%%xmm2                 \n"
      "movdqu      %%xmm1,0x00(%1,%0,2)          \n"
      "movdqu      %%xmm2,0x10(%1,%0,2)          \n"
      "lea         0x10(%0),%0                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "eax", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
        "xmm6", "xmm7");
}

void ARGB1555ToARGBRow_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "mov         $0x1080108,%%eax              \n"
      "movd        %%eax,%%xmm5                  \n"
      "pshufd      $0x0,%%xmm5,%%xmm5            \n"
      "mov         $0x42004200,%%eax             \n"
      "movd        %%eax,%%xmm6                  \n"
      "pshufd      $0x0,%%xmm6,%%xmm6            \n"
      "pcmpeqb     %%xmm3,%%xmm3                 \n"
      "psllw       $0xb,%%xmm3                   \n"
      "movdqa      %%xmm3,%%xmm4                 \n"
      "psrlw       $0x6,%%xmm4                   \n"
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psllw       $0x8,%%xmm7                   \n"
      "sub         %0,%1                         \n"
      "sub         %0,%1                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "psllw       $0x1,%%xmm1                   \n"
      "psllw       $0xb,%%xmm2                   \n"
      "pand        %%xmm3,%%xmm1                 \n"
      "pmulhuw     %%xmm5,%%xmm2                 \n"
      "pmulhuw     %%xmm5,%%xmm1                 \n"
      "psllw       $0x8,%%xmm1                   \n"
      "por         %%xmm2,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "pand        %%xmm4,%%xmm0                 \n"
      "psraw       $0x8,%%xmm2                   \n"
      "pmulhuw     %%xmm6,%%xmm0                 \n"
      "pand        %%xmm7,%%xmm2                 \n"
      "por         %%xmm2,%%xmm0                 \n"
      "movdqa      %%xmm1,%%xmm2                 \n"
      "punpcklbw   %%xmm0,%%xmm1                 \n"
      "punpckhbw   %%xmm0,%%xmm2                 \n"
      "movdqu      %%xmm1,0x00(%1,%0,2)          \n"
      "movdqu      %%xmm2,0x10(%1,%0,2)          \n"
      "lea         0x10(%0),%0                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "eax", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
        "xmm6", "xmm7");
}

void ARGB4444ToARGBRow_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "mov         $0xf0f0f0f,%%eax              \n"
      "movd        %%eax,%%xmm4                  \n"
      "pshufd      $0x0,%%xmm4,%%xmm4            \n"
      "movdqa      %%xmm4,%%xmm5                 \n"
      "pslld       $0x4,%%xmm5                   \n"
      "sub         %0,%1                         \n"
      "sub         %0,%1                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "pand        %%xmm4,%%xmm0                 \n"
      "pand        %%xmm5,%%xmm2                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "movdqa      %%xmm2,%%xmm3                 \n"
      "psllw       $0x4,%%xmm1                   \n"
      "psrlw       $0x4,%%xmm3                   \n"
      "por         %%xmm1,%%xmm0                 \n"
      "por         %%xmm3,%%xmm2                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm2,%%xmm0                 \n"
      "punpckhbw   %%xmm2,%%xmm1                 \n"
      "movdqu      %%xmm0,0x00(%1,%0,2)          \n"
      "movdqu      %%xmm1,0x10(%1,%0,2)          \n"
      "lea         0x10(%0),%0                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "eax", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

void ARGBToRGB24Row_SSSE3(const uint8_t* src, uint8_t* dst, int width) {
      asm volatile("movdqa      %3,%%xmm6                     \n"

               LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm2               \n"
      "movdqu      0x30(%0),%%xmm3               \n"
      "lea         0x40(%0),%0                   \n"
      "pshufb      %%xmm6,%%xmm0                 \n"
      "pshufb      %%xmm6,%%xmm1                 \n"
      "pshufb      %%xmm6,%%xmm2                 \n"
      "pshufb      %%xmm6,%%xmm3                 \n"
      "movdqa      %%xmm1,%%xmm4                 \n"
      "psrldq      $0x4,%%xmm1                   \n"
      "pslldq      $0xc,%%xmm4                   \n"
      "movdqa      %%xmm2,%%xmm5                 \n"
      "por         %%xmm4,%%xmm0                 \n"
      "pslldq      $0x8,%%xmm5                   \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "por         %%xmm5,%%xmm1                 \n"
      "psrldq      $0x8,%%xmm2                   \n"
      "pslldq      $0x4,%%xmm3                   \n"
      "por         %%xmm3,%%xmm2                 \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "movdqu      %%xmm2,0x20(%1)               \n"
      "lea         0x30(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
               : "+r"(src),                    // %0
                 "+r"(dst),                    // %1
                 "+r"(width)                   // %2
               : "m"(kShuffleMaskARGBToRGB24)  // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
                 "xmm6");
}

void ARGBToRAWRow_SSSE3(const uint8_t* src, uint8_t* dst, int width) {
      asm volatile("movdqa      %3,%%xmm6                     \n"

               LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm2               \n"
      "movdqu      0x30(%0),%%xmm3               \n"
      "lea         0x40(%0),%0                   \n"
      "pshufb      %%xmm6,%%xmm0                 \n"
      "pshufb      %%xmm6,%%xmm1                 \n"
      "pshufb      %%xmm6,%%xmm2                 \n"
      "pshufb      %%xmm6,%%xmm3                 \n"
      "movdqa      %%xmm1,%%xmm4                 \n"
      "psrldq      $0x4,%%xmm1                   \n"
      "pslldq      $0xc,%%xmm4                   \n"
      "movdqa      %%xmm2,%%xmm5                 \n"
      "por         %%xmm4,%%xmm0                 \n"
      "pslldq      $0x8,%%xmm5                   \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "por         %%xmm5,%%xmm1                 \n"
      "psrldq      $0x8,%%xmm2                   \n"
      "pslldq      $0x4,%%xmm3                   \n"
      "por         %%xmm3,%%xmm2                 \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "movdqu      %%xmm2,0x20(%1)               \n"
      "lea         0x30(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
               : "+r"(src),                  // %0
                 "+r"(dst),                  // %1
                 "+r"(width)                 // %2
               : "m"(kShuffleMaskARGBToRAW)  // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
                 "xmm6");
}

#ifdef HAS_ARGBTORGB24ROW_AVX2
// vpermd for 12+12 to 24
static const lvec32 kPermdRGB24_AVX = {0, 1, 2, 4, 5, 6, 3, 7};

void ARGBToRGB24Row_AVX2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm6                  \n"
      "vmovdqa     %4,%%ymm7                     \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vmovdqu     0x40(%0),%%ymm2               \n"
      "vmovdqu     0x60(%0),%%ymm3               \n"
      "lea         0x80(%0),%0                   \n"
      "vpshufb     %%ymm6,%%ymm0,%%ymm0          \n"  // xxx0yyy0
      "vpshufb     %%ymm6,%%ymm1,%%ymm1          \n"
      "vpshufb     %%ymm6,%%ymm2,%%ymm2          \n"
      "vpshufb     %%ymm6,%%ymm3,%%ymm3          \n"
      "vpermd      %%ymm0,%%ymm7,%%ymm0          \n"  // pack to 24 bytes
      "vpermd      %%ymm1,%%ymm7,%%ymm1          \n"
      "vpermd      %%ymm2,%%ymm7,%%ymm2          \n"
      "vpermd      %%ymm3,%%ymm7,%%ymm3          \n"
      "vpermq      $0x3f,%%ymm1,%%ymm4           \n"  // combine 24 + 8
      "vpor        %%ymm4,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vpermq      $0xf9,%%ymm1,%%ymm1           \n"  // combine 16 + 16
      "vpermq      $0x4f,%%ymm2,%%ymm4           \n"
      "vpor        %%ymm4,%%ymm1,%%ymm1          \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "vpermq      $0xfe,%%ymm2,%%ymm2           \n"  // combine 8 + 24
      "vpermq      $0x93,%%ymm3,%%ymm3           \n"
      "vpor        %%ymm3,%%ymm2,%%ymm2          \n"
      "vmovdqu     %%ymm2,0x40(%1)               \n"
      "lea         0x60(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),                     // %0
        "+r"(dst),                     // %1
        "+r"(width)                    // %2
      : "m"(kShuffleMaskARGBToRGB24),  // %3
        "m"(kPermdRGB24_AVX)           // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_ARGBTORGB24ROW_AVX512VBMI
// Shuffle table for converting ARGBToRGB24
static const ulvec8 kPermARGBToRGB24_0 = {
    0u,  1u,  2u,  4u,  5u,  6u,  8u,  9u,  10u, 12u, 13u,
    14u, 16u, 17u, 18u, 20u, 21u, 22u, 24u, 25u, 26u, 28u,
    29u, 30u, 32u, 33u, 34u, 36u, 37u, 38u, 40u, 41u};
static const ulvec8 kPermARGBToRGB24_1 = {
    10u, 12u, 13u, 14u, 16u, 17u, 18u, 20u, 21u, 22u, 24u,
    25u, 26u, 28u, 29u, 30u, 32u, 33u, 34u, 36u, 37u, 38u,
    40u, 41u, 42u, 44u, 45u, 46u, 48u, 49u, 50u, 52u};
static const ulvec8 kPermARGBToRGB24_2 = {
    21u, 22u, 24u, 25u, 26u, 28u, 29u, 30u, 32u, 33u, 34u,
    36u, 37u, 38u, 40u, 41u, 42u, 44u, 45u, 46u, 48u, 49u,
    50u, 52u, 53u, 54u, 56u, 57u, 58u, 60u, 61u, 62u};

void ARGBToRGB24Row_AVX512VBMI(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "vmovdqa     %3,%%ymm5                     \n"
      "vmovdqa     %4,%%ymm6                     \n"
      "vmovdqa     %5,%%ymm7                     \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vmovdqu     0x40(%0),%%ymm2               \n"
      "vmovdqu     0x60(%0),%%ymm3               \n"
      "lea         0x80(%0),%0                   \n"
      "vpermt2b    %%ymm1,%%ymm5,%%ymm0          \n"
      "vpermt2b    %%ymm2,%%ymm6,%%ymm1          \n"
      "vpermt2b    %%ymm3,%%ymm7,%%ymm2          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "vmovdqu     %%ymm2,0x40(%1)               \n"
      "lea         0x60(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),                // %0
        "+r"(dst),                // %1
        "+r"(width)               // %2
      : "m"(kPermARGBToRGB24_0),  // %3
        "m"(kPermARGBToRGB24_1),  // %4
        "m"(kPermARGBToRGB24_2)   // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5", "xmm6", "xmm7");
}
#endif

#ifdef HAS_ARGBTORAWROW_AVX2
void ARGBToRAWRow_AVX2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm6                  \n"
      "vmovdqa     %4,%%ymm7                     \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vmovdqu     0x40(%0),%%ymm2               \n"
      "vmovdqu     0x60(%0),%%ymm3               \n"
      "lea         0x80(%0),%0                   \n"
      "vpshufb     %%ymm6,%%ymm0,%%ymm0          \n"  // xxx0yyy0
      "vpshufb     %%ymm6,%%ymm1,%%ymm1          \n"
      "vpshufb     %%ymm6,%%ymm2,%%ymm2          \n"
      "vpshufb     %%ymm6,%%ymm3,%%ymm3          \n"
      "vpermd      %%ymm0,%%ymm7,%%ymm0          \n"  // pack to 24 bytes
      "vpermd      %%ymm1,%%ymm7,%%ymm1          \n"
      "vpermd      %%ymm2,%%ymm7,%%ymm2          \n"
      "vpermd      %%ymm3,%%ymm7,%%ymm3          \n"
      "vpermq      $0x3f,%%ymm1,%%ymm4           \n"  // combine 24 + 8
      "vpor        %%ymm4,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vpermq      $0xf9,%%ymm1,%%ymm1           \n"  // combine 16 + 16
      "vpermq      $0x4f,%%ymm2,%%ymm4           \n"
      "vpor        %%ymm4,%%ymm1,%%ymm1          \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "vpermq      $0xfe,%%ymm2,%%ymm2           \n"  // combine 8 + 24
      "vpermq      $0x93,%%ymm3,%%ymm3           \n"
      "vpor        %%ymm3,%%ymm2,%%ymm2          \n"
      "vmovdqu     %%ymm2,0x40(%1)               \n"
      "lea         0x60(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),                   // %0
        "+r"(dst),                   // %1
        "+r"(width)                  // %2
      : "m"(kShuffleMaskARGBToRAW),  // %3
        "m"(kPermdRGB24_AVX)         // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

void ARGBToRGB565Row_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "pcmpeqb     %%xmm3,%%xmm3                 \n"
      "psrld       $0x1b,%%xmm3                  \n"
      "pcmpeqb     %%xmm4,%%xmm4                 \n"
      "psrld       $0x1a,%%xmm4                  \n"
      "pslld       $0x5,%%xmm4                   \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "pslld       $0xb,%%xmm5                   \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "pslld       $0x8,%%xmm0                   \n"
      "psrld       $0x3,%%xmm1                   \n"
      "psrld       $0x5,%%xmm2                   \n"
      "psrad       $0x10,%%xmm0                  \n"
      "pand        %%xmm3,%%xmm1                 \n"
      "pand        %%xmm4,%%xmm2                 \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "por         %%xmm2,%%xmm1                 \n"
      "por         %%xmm1,%%xmm0                 \n"
      "packssdw    %%xmm0,%%xmm0                 \n"
      "lea         0x10(%0),%0                   \n"
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

void ARGBToRGB565DitherRow_SSE2(const uint8_t* src,
                                uint8_t* dst,
                                uint32_t dither4,
                                int width) {
  asm volatile(
      "movd        %3,%%xmm6                     \n"
      "punpcklbw   %%xmm6,%%xmm6                 \n"
      "movdqa      %%xmm6,%%xmm7                 \n"
      "punpcklwd   %%xmm6,%%xmm6                 \n"
      "punpckhwd   %%xmm7,%%xmm7                 \n"
      "pcmpeqb     %%xmm3,%%xmm3                 \n"
      "psrld       $0x1b,%%xmm3                  \n"
      "pcmpeqb     %%xmm4,%%xmm4                 \n"
      "psrld       $0x1a,%%xmm4                  \n"
      "pslld       $0x5,%%xmm4                   \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "pslld       $0xb,%%xmm5                   \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "paddusb     %%xmm6,%%xmm0                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "pslld       $0x8,%%xmm0                   \n"
      "psrld       $0x3,%%xmm1                   \n"
      "psrld       $0x5,%%xmm2                   \n"
      "psrad       $0x10,%%xmm0                  \n"
      "pand        %%xmm3,%%xmm1                 \n"
      "pand        %%xmm4,%%xmm2                 \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "por         %%xmm2,%%xmm1                 \n"
      "por         %%xmm1,%%xmm0                 \n"
      "packssdw    %%xmm0,%%xmm0                 \n"
      "lea         0x10(%0),%0                   \n"
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),    // %0
        "+r"(dst),    // %1
        "+r"(width)   // %2
      : "m"(dither4)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

#ifdef HAS_ARGBTORGB565DITHERROW_AVX2
void ARGBToRGB565DitherRow_AVX2(const uint8_t* src,
                                uint8_t* dst,
                                uint32_t dither4,
                                int width) {
  asm volatile(
      "vbroadcastss %3,%%xmm6                    \n"
      "vpunpcklbw  %%xmm6,%%xmm6,%%xmm6          \n"
      "vpermq      $0xd8,%%ymm6,%%ymm6           \n"
      "vpunpcklwd  %%ymm6,%%ymm6,%%ymm6          \n"
      "vpcmpeqb    %%ymm3,%%ymm3,%%ymm3          \n"
      "vpsrld      $0x1b,%%ymm3,%%ymm3           \n"
      "vpcmpeqb    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpsrld      $0x1a,%%ymm4,%%ymm4           \n"
      "vpslld      $0x5,%%ymm4,%%ymm4            \n"
      "vpslld      $0xb,%%ymm3,%%ymm5            \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vpaddusb    %%ymm6,%%ymm0,%%ymm0          \n"
      "vpsrld      $0x5,%%ymm0,%%ymm2            \n"
      "vpsrld      $0x3,%%ymm0,%%ymm1            \n"
      "vpsrld      $0x8,%%ymm0,%%ymm0            \n"
      "vpand       %%ymm4,%%ymm2,%%ymm2          \n"
      "vpand       %%ymm3,%%ymm1,%%ymm1          \n"
      "vpand       %%ymm5,%%ymm0,%%ymm0          \n"
      "vpor        %%ymm2,%%ymm1,%%ymm1          \n"
      "vpor        %%ymm1,%%ymm0,%%ymm0          \n"
      "vpackusdw   %%ymm0,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "lea         0x20(%0),%0                   \n"
      "vmovdqu     %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),    // %0
        "+r"(dst),    // %1
        "+r"(width)   // %2
      : "m"(dither4)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBTORGB565DITHERROW_AVX2

void ARGBToARGB1555Row_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "pcmpeqb     %%xmm4,%%xmm4                 \n"
      "psrld       $0x1b,%%xmm4                  \n"
      "movdqa      %%xmm4,%%xmm5                 \n"
      "pslld       $0x5,%%xmm5                   \n"
      "movdqa      %%xmm4,%%xmm6                 \n"
      "pslld       $0xa,%%xmm6                   \n"
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "pslld       $0xf,%%xmm7                   \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm0,%%xmm3                 \n"
      "psrad       $0x10,%%xmm0                  \n"
      "psrld       $0x3,%%xmm1                   \n"
      "psrld       $0x6,%%xmm2                   \n"
      "psrld       $0x9,%%xmm3                   \n"
      "pand        %%xmm7,%%xmm0                 \n"
      "pand        %%xmm4,%%xmm1                 \n"
      "pand        %%xmm5,%%xmm2                 \n"
      "pand        %%xmm6,%%xmm3                 \n"
      "por         %%xmm1,%%xmm0                 \n"
      "por         %%xmm3,%%xmm2                 \n"
      "por         %%xmm2,%%xmm0                 \n"
      "packssdw    %%xmm0,%%xmm0                 \n"
      "lea         0x10(%0),%0                   \n"
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7");
}

void ARGBToARGB4444Row_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "pcmpeqb     %%xmm4,%%xmm4                 \n"
      "psllw       $0xc,%%xmm4                   \n"
      "movdqa      %%xmm4,%%xmm3                 \n"
      "psrlw       $0x8,%%xmm3                   \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pand        %%xmm3,%%xmm0                 \n"
      "pand        %%xmm4,%%xmm1                 \n"
      "psrlq       $0x4,%%xmm0                   \n"
      "psrlq       $0x8,%%xmm1                   \n"
      "por         %%xmm1,%%xmm0                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "lea         0x10(%0),%0                   \n"
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif  // HAS_RGB24TOARGBROW_SSSE3

/*

ARGBToAR30Row:

Red Blue
With the 8 bit value in the upper bits of a short, vpmulhuw by (1024+4) will
produce a 10 bit value in the low 10 bits of each 16 bit value. This is whats
wanted for the blue channel. The red needs to be shifted 4 left, so multiply by
(1024+4)*16 for red.

Alpha Green
Alpha and Green are already in the high bits so vpand can zero out the other
bits, keeping just 2 upper bits of alpha and 8 bit green. The same multiplier
could be used for Green - (1024+4) putting the 10 bit green in the lsb.  Alpha
would be a simple multiplier to shift it into position.  It wants a gap of 10
above the green.  Green is 10 bits, so there are 6 bits in the low short.  4
more are needed, so a multiplier of 4 gets the 2 bits into the upper 16 bits,
and then a shift of 4 is a multiply of 16, so (4*16) = 64.  Then shift the
result left 10 to position the A and G channels.
*/

// Shuffle table for converting RAW to RGB24.  Last 8.
static const uvec8 kShuffleRB30 = {128u, 0u, 128u, 2u,  128u, 4u,  128u, 6u,
                                   128u, 8u, 128u, 10u, 128u, 12u, 128u, 14u};

static const uvec8 kShuffleBR30 = {128u, 2u,  128u, 0u, 128u, 6u,  128u, 4u,
                                   128u, 10u, 128u, 8u, 128u, 14u, 128u, 12u};

static const uint32_t kMulRB10 = 1028 * 16 * 65536 + 1028;
static const uint32_t kMaskRB10 = 0x3ff003ff;
static const uint32_t kMaskAG10 = 0xc000ff00;
static const uint32_t kMulAG10 = 64 * 65536 + 1028;

void ARGBToAR30Row_SSSE3(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "movdqa      %3,%%xmm2                     \n"  // shuffler for RB
      "movd        %4,%%xmm3                     \n"  // multipler for RB
      "movd        %5,%%xmm4                     \n"  // mask for R10 B10
      "movd        %6,%%xmm5                     \n"  // mask for AG
      "movd        %7,%%xmm6                     \n"  // multipler for AG
      "pshufd      $0x0,%%xmm3,%%xmm3            \n"
      "pshufd      $0x0,%%xmm4,%%xmm4            \n"
      "pshufd      $0x0,%%xmm5,%%xmm5            \n"
      "pshufd      $0x0,%%xmm6,%%xmm6            \n"
      "sub         %0,%1                         \n"

      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"  // fetch 4 ARGB pixels
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pshufb      %%xmm2,%%xmm1                 \n"  // R0B0
      "pand        %%xmm5,%%xmm0                 \n"  // A0G0
      "pmulhuw     %%xmm3,%%xmm1                 \n"  // X2 R16 X4  B10
      "pmulhuw     %%xmm6,%%xmm0                 \n"  // X10 A2 X10 G10
      "pand        %%xmm4,%%xmm1                 \n"  // X2 R10 X10 B10
      "pslld       $10,%%xmm0                    \n"  // A2 x10 G10 x10
      "por         %%xmm1,%%xmm0                 \n"  // A2 R10 G10 B10
      "movdqu      %%xmm0,(%1,%0)                \n"  // store 4 AR30 pixels
      "add         $0x10,%0                      \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"

      : "+r"(src),          // %0
        "+r"(dst),          // %1
        "+r"(width)         // %2
      : "m"(kShuffleRB30),  // %3
        "m"(kMulRB10),      // %4
        "m"(kMaskRB10),     // %5
        "m"(kMaskAG10),     // %6
        "m"(kMulAG10)       // %7
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}

void ABGRToAR30Row_SSSE3(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "movdqa      %3,%%xmm2                     \n"  // shuffler for RB
      "movd        %4,%%xmm3                     \n"  // multipler for RB
      "movd        %5,%%xmm4                     \n"  // mask for R10 B10
      "movd        %6,%%xmm5                     \n"  // mask for AG
      "movd        %7,%%xmm6                     \n"  // multipler for AG
      "pshufd      $0x0,%%xmm3,%%xmm3            \n"
      "pshufd      $0x0,%%xmm4,%%xmm4            \n"
      "pshufd      $0x0,%%xmm5,%%xmm5            \n"
      "pshufd      $0x0,%%xmm6,%%xmm6            \n"
      "sub         %0,%1                         \n"

      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"  // fetch 4 ABGR pixels
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pshufb      %%xmm2,%%xmm1                 \n"  // R0B0
      "pand        %%xmm5,%%xmm0                 \n"  // A0G0
      "pmulhuw     %%xmm3,%%xmm1                 \n"  // X2 R16 X4  B10
      "pmulhuw     %%xmm6,%%xmm0                 \n"  // X10 A2 X10 G10
      "pand        %%xmm4,%%xmm1                 \n"  // X2 R10 X10 B10
      "pslld       $10,%%xmm0                    \n"  // A2 x10 G10 x10
      "por         %%xmm1,%%xmm0                 \n"  // A2 R10 G10 B10
      "movdqu      %%xmm0,(%1,%0)                \n"  // store 4 AR30 pixels
      "add         $0x10,%0                      \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"

      : "+r"(src),          // %0
        "+r"(dst),          // %1
        "+r"(width)         // %2
      : "m"(kShuffleBR30),  // %3  reversed shuffler
        "m"(kMulRB10),      // %4
        "m"(kMaskRB10),     // %5
        "m"(kMaskAG10),     // %6
        "m"(kMulAG10)       // %7
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}

#ifdef HAS_ARGBTOAR30ROW_AVX2
void ARGBToAR30Row_AVX2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm2                  \n"  // shuffler for RB
      "vbroadcastss %4,%%ymm3                    \n"  // multipler for RB
      "vbroadcastss %5,%%ymm4                    \n"  // mask for R10 B10
      "vbroadcastss %6,%%ymm5                    \n"  // mask for AG
      "vbroadcastss %7,%%ymm6                    \n"  // multipler for AG
      "sub         %0,%1                         \n"

      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"  // fetch 8 ARGB pixels
      "vpshufb     %%ymm2,%%ymm0,%%ymm1          \n"  // R0B0
      "vpand       %%ymm5,%%ymm0,%%ymm0          \n"  // A0G0
      "vpmulhuw    %%ymm3,%%ymm1,%%ymm1          \n"  // X2 R16 X4  B10
      "vpmulhuw    %%ymm6,%%ymm0,%%ymm0          \n"  // X10 A2 X10 G10
      "vpand       %%ymm4,%%ymm1,%%ymm1          \n"  // X2 R10 X10 B10
      "vpslld      $10,%%ymm0,%%ymm0             \n"  // A2 x10 G10 x10
      "vpor        %%ymm1,%%ymm0,%%ymm0          \n"  // A2 R10 G10 B10
      "vmovdqu     %%ymm0,(%1,%0)                \n"  // store 8 AR30 pixels
      "add         $0x20,%0                      \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"

      : "+r"(src),          // %0
        "+r"(dst),          // %1
        "+r"(width)         // %2
      : "m"(kShuffleRB30),  // %3
        "m"(kMulRB10),      // %4
        "m"(kMaskRB10),     // %5
        "m"(kMaskAG10),     // %6
        "m"(kMulAG10)       // %7
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif

#ifdef HAS_ABGRTOAR30ROW_AVX2
void ABGRToAR30Row_AVX2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm2                  \n"  // shuffler for RB
      "vbroadcastss %4,%%ymm3                    \n"  // multipler for RB
      "vbroadcastss %5,%%ymm4                    \n"  // mask for R10 B10
      "vbroadcastss %6,%%ymm5                    \n"  // mask for AG
      "vbroadcastss %7,%%ymm6                    \n"  // multipler for AG
      "sub         %0,%1                         \n"

      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"  // fetch 8 ABGR pixels
      "vpshufb     %%ymm2,%%ymm0,%%ymm1          \n"  // R0B0
      "vpand       %%ymm5,%%ymm0,%%ymm0          \n"  // A0G0
      "vpmulhuw    %%ymm3,%%ymm1,%%ymm1          \n"  // X2 R16 X4  B10
      "vpmulhuw    %%ymm6,%%ymm0,%%ymm0          \n"  // X10 A2 X10 G10
      "vpand       %%ymm4,%%ymm1,%%ymm1          \n"  // X2 R10 X10 B10
      "vpslld      $10,%%ymm0,%%ymm0             \n"  // A2 x10 G10 x10
      "vpor        %%ymm1,%%ymm0,%%ymm0          \n"  // A2 R10 G10 B10
      "vmovdqu     %%ymm0,(%1,%0)                \n"  // store 8 AR30 pixels
      "add         $0x20,%0                      \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"

      : "+r"(src),          // %0
        "+r"(dst),          // %1
        "+r"(width)         // %2
      : "m"(kShuffleBR30),  // %3  reversed shuffler
        "m"(kMulRB10),      // %4
        "m"(kMaskRB10),     // %5
        "m"(kMaskAG10),     // %6
        "m"(kMulAG10)       // %7
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif

static const uvec8 kShuffleARGBToABGR = {2,  1, 0, 3,  6,  5,  4,  7,
                                         10, 9, 8, 11, 14, 13, 12, 15};

static const uvec8 kShuffleARGBToAB64Lo = {2, 2, 1, 1, 0, 0, 3, 3,
                                           6, 6, 5, 5, 4, 4, 7, 7};
static const uvec8 kShuffleARGBToAB64Hi = {10, 10, 9,  9,  8,  8,  11, 11,
                                           14, 14, 13, 13, 12, 12, 15, 15};

void ARGBToAR64Row_SSSE3(const uint8_t* src_argb,
                         uint16_t* dst_ar64,
                         int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm0,%%xmm0                 \n"
      "punpckhbw   %%xmm1,%%xmm1                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_ar64),  // %1
        "+r"(width)      // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}

void ARGBToAB64Row_SSSE3(const uint8_t* src_argb,
                         uint16_t* dst_ab64,
                         int width) {
  asm volatile(
      "movdqa      %3,%%xmm2                     \n"
      "movdqa      %4,%%xmm3                     \n" LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pshufb      %%xmm2,%%xmm0                 \n"
      "pshufb      %%xmm3,%%xmm1                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "lea         0x10(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),             // %0
        "+r"(dst_ab64),             // %1
        "+r"(width)                 // %2
      : "m"(kShuffleARGBToAB64Lo),  // %3
        "m"(kShuffleARGBToAB64Hi)   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}

void AR64ToARGBRow_SSSE3(const uint16_t* src_ar64,
                         uint8_t* dst_argb,
                         int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "psrlw       $8,%%xmm0                     \n"
      "psrlw       $8,%%xmm1                     \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_ar64),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}

void AB64ToARGBRow_SSSE3(const uint16_t* src_ab64,
                         uint8_t* dst_argb,
                         int width) {
      asm volatile("movdqa      %3,%%xmm2                     \n"

               LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "psrlw       $8,%%xmm0                     \n"
      "psrlw       $8,%%xmm1                     \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "pshufb      %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
               : "+r"(src_ab64),          // %0
                 "+r"(dst_argb),          // %1
                 "+r"(width)              // %2
               : "m"(kShuffleARGBToABGR)  // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}

#ifdef HAS_ARGBTOAR64ROW_AVX2
void ARGBToAR64Row_AVX2(const uint8_t* src_argb,
                        uint16_t* dst_ar64,
                        int width) {
  asm volatile(
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpunpckhbw  %%ymm0,%%ymm0,%%ymm1          \n"
      "vpunpcklbw  %%ymm0,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_ar64),  // %1
        "+r"(width)      // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}
#endif

#ifdef HAS_ARGBTOAB64ROW_AVX2
void ARGBToAB64Row_AVX2(const uint8_t* src_argb,
                        uint16_t* dst_ab64,
                        int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm2                  \n"
      "vbroadcastf128 %4,%%ymm3                  \n" LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpshufb     %%ymm3,%%ymm0,%%ymm1          \n"
      "vpshufb     %%ymm2,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),             // %0
        "+r"(dst_ab64),             // %1
        "+r"(width)                 // %2
      : "m"(kShuffleARGBToAB64Lo),  // %3
        "m"(kShuffleARGBToAB64Hi)   // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif

#ifdef HAS_AR64TOARGBROW_AVX2
void AR64ToARGBRow_AVX2(const uint16_t* src_ar64,
                        uint8_t* dst_argb,
                        int width) {
  asm volatile(
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vpsrlw      $8,%%ymm0,%%ymm0              \n"
      "vpsrlw      $8,%%ymm1,%%ymm1              \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x40(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_ar64),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
        ::"memory",
        "cc", "xmm0", "xmm1");
}
#endif

#ifdef HAS_AB64TOARGBROW_AVX2
void AB64ToARGBRow_AVX2(const uint16_t* src_ab64,
                        uint8_t* dst_argb,
                        int width) {
      asm volatile("vbroadcastf128 %3,%%ymm2                  \n" LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vpsrlw      $8,%%ymm0,%%ymm0              \n"
      "vpsrlw      $8,%%ymm1,%%ymm1              \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpshufb     %%ymm2,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x40(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_ab64),          // %0
                 "+r"(dst_argb),          // %1
                 "+r"(width)              // %2
               : "m"(kShuffleARGBToABGR)  // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif

// clang-format off

// TODO(mraptis): Consider passing R, G, B multipliers as parameter.
// round parameter is register containing value to add before shift.
#define RGBTOY(round)                            \
  "1:                                        \n" \
  "movdqu    (%0),%%xmm0                     \n" \
  "movdqu    0x10(%0),%%xmm1                 \n" \
  "movdqu    0x20(%0),%%xmm2                 \n" \
  "movdqu    0x30(%0),%%xmm3                 \n" \
  "psubb     %%xmm5,%%xmm0                   \n" \
  "psubb     %%xmm5,%%xmm1                   \n" \
  "psubb     %%xmm5,%%xmm2                   \n" \
  "psubb     %%xmm5,%%xmm3                   \n" \
  "movdqu    %%xmm4,%%xmm6                   \n" \
  "pmaddubsw %%xmm0,%%xmm6                   \n" \
  "movdqu    %%xmm4,%%xmm0                   \n" \
  "pmaddubsw %%xmm1,%%xmm0                   \n" \
  "movdqu    %%xmm4,%%xmm1                   \n" \
  "pmaddubsw %%xmm2,%%xmm1                   \n" \
  "movdqu    %%xmm4,%%xmm2                   \n" \
  "pmaddubsw %%xmm3,%%xmm2                   \n" \
  "lea       0x40(%0),%0                     \n" \
  "phaddw    %%xmm0,%%xmm6                   \n" \
  "phaddw    %%xmm2,%%xmm1                   \n" \
  "prefetcht0 1280(%0)                       \n" \
  "paddw     %%" #round ",%%xmm6             \n" \
  "paddw     %%" #round ",%%xmm1             \n" \
  "psrlw     $0x8,%%xmm6                     \n" \
  "psrlw     $0x8,%%xmm1                     \n" \
  "packuswb  %%xmm1,%%xmm6                   \n" \
  "movdqu    %%xmm6,(%1)                     \n" \
  "lea       0x10(%1),%1                     \n" \
  "sub       $0x10,%2                        \n" \
  "jg        1b                              \n"

#define RGBTOY_AVX2(round)                                       \
  "1:                                        \n"                 \
  "vmovdqu    (%0),%%ymm0                    \n"                 \
  "vmovdqu    0x20(%0),%%ymm1                \n"                 \
  "vmovdqu    0x40(%0),%%ymm2                \n"                 \
  "vmovdqu    0x60(%0),%%ymm3                \n"                 \
  "vpsubb     %%ymm5, %%ymm0, %%ymm0         \n"                 \
  "vpsubb     %%ymm5, %%ymm1, %%ymm1         \n"                 \
  "vpsubb     %%ymm5, %%ymm2, %%ymm2         \n"                 \
  "vpsubb     %%ymm5, %%ymm3, %%ymm3         \n"                 \
  "vpmaddubsw %%ymm0,%%ymm4,%%ymm0           \n"                 \
  "vpmaddubsw %%ymm1,%%ymm4,%%ymm1           \n"                 \
  "vpmaddubsw %%ymm2,%%ymm4,%%ymm2           \n"                 \
  "vpmaddubsw %%ymm3,%%ymm4,%%ymm3           \n"                 \
  "lea       0x80(%0),%0                     \n"                 \
  "vphaddw    %%ymm1,%%ymm0,%%ymm0           \n" /* mutates. */  \
  "vphaddw    %%ymm3,%%ymm2,%%ymm2           \n"                 \
  "prefetcht0 1280(%0)                       \n"                 \
  "vpaddw     %%" #round ",%%ymm0,%%ymm0     \n" /* Add 16 */    \
  "vpaddw     %%" #round ",%%ymm2,%%ymm2     \n"                 \
  "vpsrlw     $0x8,%%ymm0,%%ymm0             \n"                 \
  "vpsrlw     $0x8,%%ymm2,%%ymm2             \n"                 \
  "vpackuswb  %%ymm2,%%ymm0,%%ymm0           \n" /* mutates. */  \
  "vpermd     %%ymm0,%%ymm6,%%ymm0           \n" /* unmutate. */ \
  "vmovdqu    %%ymm0,(%1)                    \n"                 \
  "lea       0x20(%1),%1                     \n"                 \
  "sub       $0x20,%2                        \n"                 \
  "jg        1b                              \n"

// clang-format on

#ifdef HAS_ARGBTOYROW_SSSE3
// Convert 16 ARGB pixels (64 bytes) to 16 Y values.
void ARGBToYRow_SSSE3(const uint8_t* src_argb, uint8_t* dst_y, int width) {
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"
      "movdqa      %5,%%xmm7                     \n"  //

      LABELALIGN ""      //
      RGBTOY(xmm7)       //
      : "+r"(src_argb),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      : "m"(kARGBToY),   // %3
        "m"(kSub128),    // %4
        "m"(kAddY16)     // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBTOYROW_SSSE3

#ifdef HAS_ARGBTOYJROW_SSSE3
// Convert 16 ARGB pixels (64 bytes) to 16 YJ values.
// Same as ARGBToYRow but different coefficients, no add 16.
void ARGBToYJRow_SSSE3(const uint8_t* src_argb, uint8_t* dst_y, int width) {
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"
      "movdqa      %5,%%xmm7                     \n"  //

      LABELALIGN ""      //
      RGBTOY(xmm7)       //
      : "+r"(src_argb),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      : "m"(kARGBToYJ),  // %3
        "m"(kSub128),    // %4
        "m"(kAddY0)      // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBTOYJROW_SSSE3

#ifdef HAS_ABGRTOYJROW_SSSE3
// Convert 16 ABGR pixels (64 bytes) to 16 YJ values.
// Same as ABGRToYRow but different coefficients, no add 16.
void ABGRToYJRow_SSSE3(const uint8_t* src_abgr, uint8_t* dst_y, int width) {
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"
      "movdqa      %5,%%xmm7                     \n"  //

      LABELALIGN ""      //
      RGBTOY(xmm7)       //
      : "+r"(src_abgr),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      : "m"(kABGRToYJ),  // %3
        "m"(kSub128),    // %4
        "m"(kAddY0)      // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ABGRTOYJROW_SSSE3

#ifdef HAS_RGBATOYJROW_SSSE3
// Convert 16 ARGB pixels (64 bytes) to 16 YJ values.
// Same as ARGBToYRow but different coefficients, no add 16.
void RGBAToYJRow_SSSE3(const uint8_t* src_rgba, uint8_t* dst_y, int width) {
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"
      "movdqa      %5,%%xmm7                     \n"  //

      LABELALIGN ""      //
      RGBTOY(xmm7)       //
      : "+r"(src_rgba),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      : "m"(kRGBAToYJ),  // %3
        "m"(kSub128),    // %4
        "m"(kAddY0)      // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_RGBATOYJROW_SSSE3

#if defined(HAS_ARGBTOYROW_AVX2) || defined(HAS_ABGRTOYROW_AVX2) || \
    defined(HAS_ARGBEXTRACTALPHAROW_AVX2)
// vpermd for vphaddw + vpackuswb vpermd.
static const lvec32 kPermdARGBToY_AVX = {0, 4, 1, 5, 2, 6, 3, 7};
#endif

#ifdef HAS_ARGBTOYROW_AVX2

// Convert 32 ARGB pixels (128 bytes) to 32 Y values.
void ARGBToYRow_AVX2(const uint8_t* src_argb, uint8_t* dst_y, int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm4                  \n"
      "vbroadcastf128 %4,%%ymm5                  \n"
      "vbroadcastf128 %5,%%ymm7                  \n"
      "vmovdqa     %6,%%ymm6                     \n"  //

      LABELALIGN ""      //
      RGBTOY_AVX2(ymm7)  //
      "vzeroupper  \n"
      : "+r"(src_argb),         // %0
        "+r"(dst_y),            // %1
        "+r"(width)             // %2
      : "m"(kARGBToY),          // %3
        "m"(kSub128),           // %4
        "m"(kAddY16),           // %5
        "m"(kPermdARGBToY_AVX)  // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBTOYROW_AVX2

#ifdef HAS_ABGRTOYROW_AVX2
// Convert 32 ABGR pixels (128 bytes) to 32 Y values.
void ABGRToYRow_AVX2(const uint8_t* src_abgr, uint8_t* dst_y, int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm4                  \n"
      "vbroadcastf128 %4,%%ymm5                  \n"
      "vbroadcastf128 %5,%%ymm7                  \n"
      "vmovdqa     %6,%%ymm6                     \n"  //

      LABELALIGN ""      //
      RGBTOY_AVX2(ymm7)  //
      "vzeroupper  \n"
      : "+r"(src_abgr),         // %0
        "+r"(dst_y),            // %1
        "+r"(width)             // %2
      : "m"(kABGRToY),          // %3
        "m"(kSub128),           // %4
        "m"(kAddY16),           // %5
        "m"(kPermdARGBToY_AVX)  // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ABGRTOYROW_AVX2

#ifdef HAS_ARGBTOYJROW_AVX2
// Convert 32 ARGB pixels (128 bytes) to 32 Y values.
void ARGBToYJRow_AVX2(const uint8_t* src_argb, uint8_t* dst_y, int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm4                  \n"
      "vbroadcastf128 %4,%%ymm5                  \n"
      "vbroadcastf128 %5,%%ymm7                  \n"
      "vmovdqa     %6,%%ymm6                     \n"  //

      LABELALIGN ""      //
      RGBTOY_AVX2(ymm7)  //
      "vzeroupper  \n"
      : "+r"(src_argb),         // %0
        "+r"(dst_y),            // %1
        "+r"(width)             // %2
      : "m"(kARGBToYJ),         // %3
        "m"(kSub128),           // %4
        "m"(kAddY0),            // %5
        "m"(kPermdARGBToY_AVX)  // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

#endif  // HAS_ARGBTOYJROW_AVX2

#ifdef HAS_ABGRTOYJROW_AVX2
// Convert 32 ABGR pixels (128 bytes) to 32 Y values.
void ABGRToYJRow_AVX2(const uint8_t* src_abgr, uint8_t* dst_y, int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm4                  \n"
      "vbroadcastf128 %4,%%ymm5                  \n"
      "vbroadcastf128 %5,%%ymm7                  \n"
      "vmovdqa     %6,%%ymm6                     \n"  //

      LABELALIGN ""      //
      RGBTOY_AVX2(ymm7)  //
      "vzeroupper  \n"
      : "+r"(src_abgr),         // %0
        "+r"(dst_y),            // %1
        "+r"(width)             // %2
      : "m"(kABGRToYJ),         // %3
        "m"(kSub128),           // %4
        "m"(kAddY0),            // %5
        "m"(kPermdARGBToY_AVX)  // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ABGRTOYJROW_AVX2

#ifdef HAS_RGBATOYJROW_AVX2
// Convert 32 ARGB pixels (128 bytes) to 32 Y values.
void RGBAToYJRow_AVX2(const uint8_t* src_rgba, uint8_t* dst_y, int width) {
  asm volatile(
      "vbroadcastf128 %3,%%ymm4                  \n"
      "vbroadcastf128 %4,%%ymm5                  \n"
      "vbroadcastf128 %5,%%ymm7                  \n"
      "vmovdqa     %6,%%ymm6                     \n"  //

      LABELALIGN ""      //
      RGBTOY_AVX2(ymm7)  //
      "vzeroupper  \n"
      : "+r"(src_rgba),         // %0
        "+r"(dst_y),            // %1
        "+r"(width)             // %2
      : "m"(kRGBAToYJ),         // %3
        "m"(kSub128),           // %4
        "m"(kAddY0),            // %5
        "m"(kPermdARGBToY_AVX)  // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_RGBATOYJROW_AVX2

#ifdef HAS_ARGBTOUV444ROW_SSSE3

// Coefficients expressed as negatives to allow 128
struct RgbUVConstants {
  vec8 kRGBToU;
  vec8 kRGBToV;
};

void ARGBToUV444MatrixRow_SSSE3(const uint8_t* src_argb,
                                uint8_t* dst_u,
                                uint8_t* dst_v,
                                int width,
                                const struct RgbUVConstants* rgbuvconstants) {
  asm volatile(
      "movdqa      %4,%%xmm3                     \n"
      "movdqa      %5,%%xmm4                     \n"
      "movdqa      %6,%%xmm5                     \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm2               \n"
      "movdqu      0x30(%0),%%xmm6               \n"
      "pmaddubsw   %%xmm3,%%xmm0                 \n"
      "pmaddubsw   %%xmm3,%%xmm1                 \n"
      "pmaddubsw   %%xmm3,%%xmm2                 \n"
      "pmaddubsw   %%xmm3,%%xmm6                 \n"
      "phaddw      %%xmm1,%%xmm0                 \n"
      "phaddw      %%xmm6,%%xmm2                 \n"
      "movdqa      %%xmm5,%%xmm1                 \n"
      "movdqa      %%xmm5,%%xmm6                 \n"
      "psubw       %%xmm0,%%xmm1                 \n"
      "psubw       %%xmm2,%%xmm6                 \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "psrlw       $0x8,%%xmm6                   \n"
      "packuswb    %%xmm6,%%xmm1                 \n"
      "movdqu      %%xmm1,(%1)                   \n"

      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm2               \n"
      "movdqu      0x30(%0),%%xmm6               \n"
      "pmaddubsw   %%xmm4,%%xmm0                 \n"
      "pmaddubsw   %%xmm4,%%xmm1                 \n"
      "pmaddubsw   %%xmm4,%%xmm2                 \n"
      "pmaddubsw   %%xmm4,%%xmm6                 \n"
      "phaddw      %%xmm1,%%xmm0                 \n"
      "phaddw      %%xmm6,%%xmm2                 \n"
      "movdqa      %%xmm5,%%xmm1                 \n"
      "movdqa      %%xmm5,%%xmm6                 \n"
      "psubw       %%xmm0,%%xmm1                 \n"
      "psubw       %%xmm2,%%xmm6                 \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "psrlw       $0x8,%%xmm6                   \n"
      "packuswb    %%xmm6,%%xmm1                 \n"
      "movdqu      %%xmm1,0x00(%1,%2,1)          \n"

      "lea         0x40(%0),%0                   \n"
      "lea         0x10(%1),%1                   \n"
      "subl        $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
#if defined(__i386__)
        "+m"(width)  // %3
#else
        "+rm"(width)  // %3
#endif
      : "m"(rgbuvconstants->kRGBToU),  // %4
        "m"(rgbuvconstants->kRGBToV),  // %5
        "m"(kAddUV128)                 // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif  // HAS_ARGBTOUV444ROW_SSSE3

#ifdef HAS_ARGBTOUV444ROW_AVX2

void ARGBToUV444MatrixRow_AVX2(const uint8_t* src_argb,
                               uint8_t* dst_u,
                               uint8_t* dst_v,
                               int width,
                               const struct RgbUVConstants* rgbuvconstants) {
  asm volatile(
      "vbroadcastf128 %4,%%ymm3                  \n"
      "vbroadcastf128 %5,%%ymm4                  \n"
      "vbroadcastf128 %6,%%ymm5                  \n"
      "vmovdqa     %7,%%ymm7                     \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vmovdqu     0x40(%0),%%ymm2               \n"
      "vmovdqu     0x60(%0),%%ymm6               \n"
      "vpmaddubsw  %%ymm3,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm3,%%ymm1,%%ymm1          \n"
      "vpmaddubsw  %%ymm3,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm3,%%ymm6,%%ymm6          \n"
      "vphaddw     %%ymm1,%%ymm0,%%ymm0          \n"  // mutates
      "vphaddw     %%ymm6,%%ymm2,%%ymm2          \n"
      "vpsubw      %%ymm0,%%ymm5,%%ymm0          \n"
      "vpsubw      %%ymm2,%%ymm5,%%ymm2          \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x8,%%ymm2,%%ymm2            \n"
      "vpackuswb   %%ymm2,%%ymm0,%%ymm0          \n"  // mutates
      "vpermd      %%ymm0,%%ymm7,%%ymm0          \n"  // unmutate.
      "vmovdqu     %%ymm0,(%1)                   \n"

      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vmovdqu     0x40(%0),%%ymm2               \n"
      "vmovdqu     0x60(%0),%%ymm6               \n"
      "vpmaddubsw  %%ymm4,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm4,%%ymm1,%%ymm1          \n"
      "vpmaddubsw  %%ymm4,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm4,%%ymm6,%%ymm6          \n"
      "vphaddw     %%ymm1,%%ymm0,%%ymm0          \n"  // mutates
      "vphaddw     %%ymm6,%%ymm2,%%ymm2          \n"
      "vpsubw      %%ymm0,%%ymm5,%%ymm0          \n"
      "vpsubw      %%ymm2,%%ymm5,%%ymm2          \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x8,%%ymm2,%%ymm2            \n"
      "vpackuswb   %%ymm2,%%ymm0,%%ymm0          \n"  // mutates
      "vpermd      %%ymm0,%%ymm7,%%ymm0          \n"  // unmutate.
      "vmovdqu     %%ymm0,(%1,%2,1)              \n"
      "lea         0x80(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"
      "subl        $0x20,%3                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
#if defined(__i386__)
        "+m"(width)  // %3
#else
        "+rm"(width)  // %3
#endif
      : "m"(rgbuvconstants->kRGBToU),  // %4
        "m"(rgbuvconstants->kRGBToV),  // %5
        "m"(kAddUV128),                // %6
        "m"(kPermdARGBToY_AVX)         // %7
      : "memory", "cc", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
        "ymm7");
}
#endif  // HAS_ARGBTOUV444ROW_AVX2

#ifdef HAS_ARGBTOUVROW_SSSE3
// 8x2 -> 4x1 ARGB pixels converted to 4 U and 4 V
// ARGBToUV does rounding average of 4 ARGB pixels
void ARGBToUVMatrixRow_SSSE3(const uint8_t* src_argb,
                             int src_stride_argb,
                             uint8_t* dst_u,
                             uint8_t* dst_v,
                             int width,
                             const struct RgbUVConstants* rgbuvconstants) {
  asm volatile(
      "pcmpeqb     %%xmm4,%%xmm4                 \n"  // 0x0101
      "pabsb       %%xmm4,%%xmm4                 \n"
      "movdqa      %5,%%xmm6                     \n"  // ARGB to U
      "movdqa      %6,%%xmm7                     \n"  // ARGB to V
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"  // Read 8 ARGB Pixels
      "movdqu      0x10(%0),%%xmm5               \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "shufps      $0x88,%%xmm5,%%xmm0           \n"  // Even pixels
      "shufps      $0xdd,%%xmm5,%%xmm1           \n"  // Odd pixels
      "movdqa      %%xmm0,%%xmm5                 \n"
      "punpcklbw   %%xmm1,%%xmm0                 \n"  // aarrgbb
      "punpckhbw   %%xmm5,%%xmm1                 \n"
      "pmaddubsw   %%xmm4,%%xmm0                 \n"  // paired add argb
      "pmaddubsw   %%xmm4,%%xmm1                 \n"

      "movdqu      0x00(%0,%4,1),%%xmm2          \n"  // Read 2nd row
      "movdqu      0x10(%0,%4,1),%%xmm5          \n"
      "movdqa      %%xmm2,%%xmm3                 \n"
      "shufps      $0x88,%%xmm5,%%xmm2           \n"  // Even
      "shufps      $0xdd,%%xmm5,%%xmm3           \n"  // Odd pixels
      "movdqa      %%xmm2,%%xmm5                 \n"
      "punpcklbw   %%xmm3,%%xmm2                 \n"  // aarrgbb
      "punpckhbw   %%xmm5,%%xmm3                 \n"
      "pmaddubsw   %%xmm4,%%xmm2                 \n"  // argb
      "pmaddubsw   %%xmm4,%%xmm3                 \n"

      "pxor        %%xmm5,%%xmm5                 \n"  // constant 0 for pavgw
      "paddw       %%xmm2,%%xmm0                 \n"
      "paddw       %%xmm3,%%xmm1                 \n"
      "psrlw       $1,%%xmm0                     \n"  // round
      "psrlw       $1,%%xmm1                     \n"
      "pavgw       %%xmm5,%%xmm0                 \n"
      "pavgw       %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"  // 4 ARGB pixels

      "movdqa      %%xmm0,%%xmm1                 \n"
      "pmaddubsw   %%xmm6,%%xmm0                 \n"  // u
      "pmaddubsw   %%xmm7,%%xmm1                 \n"  // v
      "phaddw      %%xmm1,%%xmm0                 \n"  // uuuuvvvv

      "movdqa      %7,%%xmm2                     \n"  // 0x8000
      "psubw       %%xmm0,%%xmm2                 \n"  // unsigned 0 to 0xffff
      "psrlw       $0x8,%%xmm2                   \n"
      "packuswb    %%xmm2,%%xmm2                 \n"
      "movd        %%xmm2,(%1)                   \n"  // Write 4 U's
      "shufps      $0xdd,%%xmm2,%%xmm2           \n"
      "movd        %%xmm2,0x00(%1,%2,1)          \n"  // Write 4 V's

      "lea         0x20(%0),%0                   \n"
      "lea         0x4(%1),%1                    \n"
      "subl        $0x8,%3                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
#if defined(__i386__)
        "+m"(width)  // %3
#else
        "+rm"(width)  // %3
#endif
      : "r"((intptr_t)(src_stride_argb)),  // %4
        "m"(rgbuvconstants->kRGBToU),      // %5
        "m"(rgbuvconstants->kRGBToV),      // %6
        "m"(kAddUV128)                     // %7

      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBTOUVROW_SSSE3

#ifdef HAS_ARGBTOUV444ROW_SSSE3

// RGB to BT601 coefficients
// UB   0.875 coefficient = 112
// UG -0.5781 coefficient = -74
// UR -0.2969 coefficient = -38
// VB -0.1406 coefficient = -18
// VG -0.7344 coefficient = -94
// VR   0.875 coefficient = 112

static const struct RgbUVConstants kARGBI601UVConstants = {
    {-112, 74, 38, 0, -112, 74, 38, 0, -112, 74, 38, 0, -112, 74, 38, 0},
    {18, 94, -112, 0, 18, 94, -112, 0, 18, 94, -112, 0, 18, 94, -112, 0}};

void ARGBToUV444Row_SSSE3(const uint8_t* src_argb,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  ARGBToUV444MatrixRow_SSSE3(src_argb, dst_u, dst_v, width,
                             &kARGBI601UVConstants);
}
#endif  // HAS_ARGBTOUV444ROW_SSSE3

#ifdef HAS_ARGBTOUV444ROW_AVX2
void ARGBToUV444Row_AVX2(const uint8_t* src_argb,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  ARGBToUV444MatrixRow_AVX2(src_argb, dst_u, dst_v, width,
                            &kARGBI601UVConstants);
}
#endif  // HAS_ARGBTOUV444ROW_AVX2

#ifdef HAS_ARGBTOUVROW_SSSE3
void ARGBToUVRow_SSSE3(const uint8_t* src_argb,
                       int src_stride_argb,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  ARGBToUVMatrixRow_SSSE3(src_argb, src_stride_argb, dst_u, dst_v, width,
                          &kARGBI601UVConstants);
}

static const struct RgbUVConstants kABGRI601UVConstants = {
    {38, 74, -112, 0, 38, 74, -112, 0, 38, 74, -112, 0, 38, 74, -112, 0},
    {-112, 94, 18, 0, -112, 94, 18, 0, -112, 94, 18, 0, -112, 94, 18, 0}};

void ABGRToUVRow_SSSE3(const uint8_t* src_abgr,
                       int src_stride_abgr,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  ARGBToUVMatrixRow_SSSE3(src_abgr, src_stride_abgr, dst_u, dst_v, width,
                          &kABGRI601UVConstants);
}

static const struct RgbUVConstants kBGRAI601UVConstants = {
    {0, 38, 74, -112, 0, 38, 74, -112, 0, 38, 74, -112, 0, 38, 74, -112},
    {0, -112, 94, 18, 0, -112, 94, 18, 0, -112, 94, 18, 0, -112, 94, 18}};

void BGRAToUVRow_SSSE3(const uint8_t* src_bgra,
                       int src_stride_bgra,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  ARGBToUVMatrixRow_SSSE3(src_bgra, src_stride_bgra, dst_u, dst_v, width,
                          &kBGRAI601UVConstants);
}

static const struct RgbUVConstants kRGBAI601UVConstants = {
    {0, -112, 74, 38, 0, -112, 74, 38, 0, -112, 74, 38, 0, -112, 74, 38},
    {0, 18, 94, -112, 0, 18, 94, -112, 0, 18, 94, -112, 0, 18, 94, -112}};

void RGBAToUVRow_SSSE3(const uint8_t* src_rgba,
                       int src_stride_rgba,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  ARGBToUVMatrixRow_SSSE3(src_rgba, src_stride_rgba, dst_u, dst_v, width,
                          &kRGBAI601UVConstants);
}
#endif  // HAS_ARGBTOUVROW_SSSE3

#ifdef HAS_ARGBTOUVJ444ROW_SSSE3
// RGB to JPEG coefficients
// UB  0.500    coefficient = 128
// UG -0.33126  coefficient = -85
// UR -0.16874  coefficient = -43
// VB -0.08131  coefficient = -21
// VG -0.41869  coefficient = -107
// VR 0.500     coefficient = 128

static const struct RgbUVConstants kARGBJPEGUVConstants = {
    {-128, 85, 43, 0, -128, 85, 43, 0, -128, 85, 43, 0, -128, 85, 43, 0},
    {21, 107, -128, 0, 21, 107, -128, 0, 21, 107, -128, 0, 21, 107, -128, 0}};

void ARGBToUVJ444Row_SSSE3(const uint8_t* src_argb,
                           uint8_t* dst_u,
                           uint8_t* dst_v,
                           int width) {
  ARGBToUV444MatrixRow_SSSE3(src_argb, dst_u, dst_v, width,
                             &kARGBJPEGUVConstants);
}

#endif  // HAS_ARGBTOUVJ444ROW_SSSE3

#ifdef HAS_ARGBTOUVJ444ROW_AVX2
void ARGBToUVJ444Row_AVX2(const uint8_t* src_argb,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  ARGBToUV444MatrixRow_AVX2(src_argb, dst_u, dst_v, width,
                            &kARGBJPEGUVConstants);
}
#endif  // HAS_ARGBTOUVJ444ROW_AVX2

static const struct RgbUVConstants kABGRJPEGUVConstants = {
    {43, 85, -128, 0, 43, 85, -128, 0, 43, 85, -128, 0, 43, 85, -128, 0},
    {-128, 107, 21, 0, -128, 107, 21, 0, -128, 107, 21, 0, -128, 107, 21, 0}};

#ifdef HAS_ARGBTOUVJROW_SSSE3
void ARGBToUVJRow_SSSE3(const uint8_t* src_argb,
                        int src_stride_argb,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  ARGBToUVMatrixRow_SSSE3(src_argb, src_stride_argb, dst_u, dst_v, width,
                          &kARGBJPEGUVConstants);
}

void ABGRToUVJRow_SSSE3(const uint8_t* src_abgr,
                        int src_stride_abgr,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  ARGBToUVMatrixRow_SSSE3(src_abgr, src_stride_abgr, dst_u, dst_v, width,
                          &kABGRJPEGUVConstants);
}
#endif  // HAS_ABGRTOUVJROW_SSSE3

void BGRAToYRow_SSSE3(const uint8_t* src_bgra, uint8_t* dst_y, int width) {
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"
      "movdqa      %5,%%xmm7                     \n"

      LABELALIGN ""  //
      RGBTOY(xmm7)
      : "+r"(src_bgra),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      : "m"(kBGRAToY),   // %3
        "m"(kSub128),    // %4
        "m"(kAddY16)     // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

void ABGRToYRow_SSSE3(const uint8_t* src_abgr, uint8_t* dst_y, int width) {
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"
      "movdqa      %5,%%xmm7                     \n"

      LABELALIGN ""  //
      RGBTOY(xmm7)
      : "+r"(src_abgr),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      : "m"(kABGRToY),   // %3
        "m"(kSub128),    // %4
        "m"(kAddY16)     // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

void RGBAToYRow_SSSE3(const uint8_t* src_rgba, uint8_t* dst_y, int width) {
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"
      "movdqa      %5,%%xmm7                     \n"

      LABELALIGN ""  //
      RGBTOY(xmm7)
      : "+r"(src_rgba),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      : "m"(kRGBAToY),   // %3
        "m"(kSub128),    // %4
        "m"(kAddY16)     // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}

#if defined(HAS_I422TOARGBROW_SSSE3) || defined(HAS_I422TOARGBROW_AVX2)

// Read 8 UV from 444
#define READYUV444                                                \
  "movq       (%[u_buf]),%%xmm3                               \n" \
  "movq       0x00(%[u_buf],%[v_buf],1),%%xmm1                \n" \
  "lea        0x8(%[u_buf]),%[u_buf]                          \n" \
  "punpcklbw  %%xmm1,%%xmm3                                   \n" \
  "movq       (%[y_buf]),%%xmm4                               \n" \
  "punpcklbw  %%xmm4,%%xmm4                                   \n" \
  "lea        0x8(%[y_buf]),%[y_buf]                          \n"

// Read 4 UV from 422, upsample to 8 UV
#define READYUV422                                                \
  "movd       (%[u_buf]),%%xmm3                               \n" \
  "movd       0x00(%[u_buf],%[v_buf],1),%%xmm1                \n" \
  "lea        0x4(%[u_buf]),%[u_buf]                          \n" \
  "punpcklbw  %%xmm1,%%xmm3                                   \n" \
  "punpcklwd  %%xmm3,%%xmm3                                   \n" \
  "movq       (%[y_buf]),%%xmm4                               \n" \
  "punpcklbw  %%xmm4,%%xmm4                                   \n" \
  "lea        0x8(%[y_buf]),%[y_buf]                          \n"

// Read 4 UV from 422 10 bit, upsample to 8 UV
#define READYUV210                                                \
  "movq       (%[u_buf]),%%xmm3                               \n" \
  "movq       0x00(%[u_buf],%[v_buf],1),%%xmm1                \n" \
  "lea        0x8(%[u_buf]),%[u_buf]                          \n" \
  "punpcklwd  %%xmm1,%%xmm3                                   \n" \
  "psraw      $2,%%xmm3                                       \n" \
  "packuswb   %%xmm3,%%xmm3                                   \n" \
  "punpcklwd  %%xmm3,%%xmm3                                   \n" \
  "movdqu     (%[y_buf]),%%xmm4                               \n" \
  "movdqa     %%xmm4,%%xmm2                                   \n" \
  "psllw      $6,%%xmm4                                       \n" \
  "psrlw      $4,%%xmm2                                       \n" \
  "paddw      %%xmm2,%%xmm4                                   \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                         \n"

#define READYUVA210                                               \
  "movq       (%[u_buf]),%%xmm3                               \n" \
  "movq       0x00(%[u_buf],%[v_buf],1),%%xmm1                \n" \
  "lea        0x8(%[u_buf]),%[u_buf]                          \n" \
  "punpcklwd  %%xmm1,%%xmm3                                   \n" \
  "psraw      $2,%%xmm3                                       \n" \
  "packuswb   %%xmm3,%%xmm3                                   \n" \
  "punpcklwd  %%xmm3,%%xmm3                                   \n" \
  "movdqu     (%[y_buf]),%%xmm4                               \n" \
  "movdqa     %%xmm4,%%xmm2                                   \n" \
  "psllw      $6,%%xmm4                                       \n" \
  "psrlw      $4,%%xmm2                                       \n" \
  "paddw      %%xmm2,%%xmm4                                   \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                         \n" \
  "movdqu     (%[a_buf]),%%xmm5                               \n" \
  "psraw      $2,%%xmm5                                       \n" \
  "packuswb   %%xmm5,%%xmm5                                   \n" \
  "lea        0x10(%[a_buf]),%[a_buf]                         \n"

// Read 8 UV from 444 10 bit
#define READYUV410                                                \
  "movdqu     (%[u_buf]),%%xmm3                               \n" \
  "movdqu     0x00(%[u_buf],%[v_buf],1),%%xmm2                \n" \
  "lea        0x10(%[u_buf]),%[u_buf]                         \n" \
  "psraw      $2,%%xmm3                                       \n" \
  "psraw      $2,%%xmm2                                       \n" \
  "movdqa     %%xmm3,%%xmm1                                   \n" \
  "punpcklwd  %%xmm2,%%xmm3                                   \n" \
  "punpckhwd  %%xmm2,%%xmm1                                   \n" \
  "packuswb   %%xmm1,%%xmm3                                   \n" \
  "movdqu     (%[y_buf]),%%xmm4                               \n" \
  "movdqa     %%xmm4,%%xmm2                                   \n" \
  "psllw      $6,%%xmm4                                       \n" \
  "psrlw      $4,%%xmm2                                       \n" \
  "paddw      %%xmm2,%%xmm4                                   \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                         \n"

// Read 8 UV from 444 10 bit.  With 8 Alpha.
#define READYUVA410                                               \
  "movdqu     (%[u_buf]),%%xmm3                               \n" \
  "movdqu     0x00(%[u_buf],%[v_buf],1),%%xmm2                \n" \
  "lea        0x10(%[u_buf]),%[u_buf]                         \n" \
  "psraw      $2,%%xmm3                                       \n" \
  "psraw      $2,%%xmm2                                       \n" \
  "movdqa     %%xmm3,%%xmm1                                   \n" \
  "punpcklwd  %%xmm2,%%xmm3                                   \n" \
  "punpckhwd  %%xmm2,%%xmm1                                   \n" \
  "packuswb   %%xmm1,%%xmm3                                   \n" \
  "movdqu     (%[y_buf]),%%xmm4                               \n" \
  "movdqa     %%xmm4,%%xmm2                                   \n" \
  "psllw      $6,%%xmm4                                       \n" \
  "psrlw      $4,%%xmm2                                       \n" \
  "paddw      %%xmm2,%%xmm4                                   \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                         \n" \
  "movdqu     (%[a_buf]),%%xmm5                               \n" \
  "psraw      $2,%%xmm5                                       \n" \
  "packuswb   %%xmm5,%%xmm5                                   \n" \
  "lea        0x10(%[a_buf]),%[a_buf]                         \n"

// Read 4 UV from 422 12 bit, upsample to 8 UV
#define READYUV212                                                \
  "movq       (%[u_buf]),%%xmm3                               \n" \
  "movq       0x00(%[u_buf],%[v_buf],1),%%xmm1                \n" \
  "lea        0x8(%[u_buf]),%[u_buf]                          \n" \
  "punpcklwd  %%xmm1,%%xmm3                                   \n" \
  "psraw      $0x4,%%xmm3                                     \n" \
  "packuswb   %%xmm3,%%xmm3                                   \n" \
  "punpcklwd  %%xmm3,%%xmm3                                   \n" \
  "movdqu     (%[y_buf]),%%xmm4                               \n" \
  "movdqa     %%xmm4,%%xmm2                                   \n" \
  "psllw      $4,%%xmm4                                       \n" \
  "psrlw      $8,%%xmm2                                       \n" \
  "paddw      %%xmm2,%%xmm4                                   \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                         \n"

// Read 4 UV from 422, upsample to 8 UV.  With 8 Alpha.
#define READYUVA422                                               \
  "movd       (%[u_buf]),%%xmm3                               \n" \
  "movd       0x00(%[u_buf],%[v_buf],1),%%xmm1                \n" \
  "lea        0x4(%[u_buf]),%[u_buf]                          \n" \
  "punpcklbw  %%xmm1,%%xmm3                                   \n" \
  "punpcklwd  %%xmm3,%%xmm3                                   \n" \
  "movq       (%[y_buf]),%%xmm4                               \n" \
  "punpcklbw  %%xmm4,%%xmm4                                   \n" \
  "lea        0x8(%[y_buf]),%[y_buf]                          \n" \
  "movq       (%[a_buf]),%%xmm5                               \n" \
  "lea        0x8(%[a_buf]),%[a_buf]                          \n"

// Read 8 UV from 444.  With 8 Alpha.
#define READYUVA444                                               \
  "movq       (%[u_buf]),%%xmm3                               \n" \
  "movq       0x00(%[u_buf],%[v_buf],1),%%xmm1                \n" \
  "lea        0x8(%[u_buf]),%[u_buf]                          \n" \
  "punpcklbw  %%xmm1,%%xmm3                                   \n" \
  "movq       (%[y_buf]),%%xmm4                               \n" \
  "punpcklbw  %%xmm4,%%xmm4                                   \n" \
  "lea        0x8(%[y_buf]),%[y_buf]                          \n" \
  "movq       (%[a_buf]),%%xmm5                               \n" \
  "lea        0x8(%[a_buf]),%[a_buf]                          \n"

// Read 4 UV from NV12, upsample to 8 UV
#define READNV12                                                  \
  "movq       (%[uv_buf]),%%xmm3                              \n" \
  "lea        0x8(%[uv_buf]),%[uv_buf]                        \n" \
  "punpcklwd  %%xmm3,%%xmm3                                   \n" \
  "movq       (%[y_buf]),%%xmm4                               \n" \
  "punpcklbw  %%xmm4,%%xmm4                                   \n" \
  "lea        0x8(%[y_buf]),%[y_buf]                          \n"

// Read 4 VU from NV21, upsample to 8 UV
#define READNV21                                                  \
  "movq       (%[vu_buf]),%%xmm3                              \n" \
  "lea        0x8(%[vu_buf]),%[vu_buf]                        \n" \
  "pshufb     %[kShuffleNV21], %%xmm3                         \n" \
  "movq       (%[y_buf]),%%xmm4                               \n" \
  "punpcklbw  %%xmm4,%%xmm4                                   \n" \
  "lea        0x8(%[y_buf]),%[y_buf]                          \n"

// Read 4 YUY2 with 8 Y and upsample 4 UV to 8 UV.
// xmm6 kShuffleYUY2Y,
// xmm7 kShuffleYUY2UV
#define READYUY2                                                  \
  "movdqu     (%[yuy2_buf]),%%xmm4                            \n" \
  "lea        0x10(%[yuy2_buf]),%[yuy2_buf]                   \n" \
  "movdqa     %%xmm4,%%xmm3                                   \n" \
  "pshufb     %%xmm6,%%xmm4                                   \n" \
  "pshufb     %%xmm7,%%xmm3                                   \n"

// Read 4 UYVY with 8 Y and upsample 4 UV to 8 UV.
// xmm6 kShuffleUYVYY,
// xmm7 kShuffleUYVYUV
#define READUYVY                                                  \
  "movdqu     (%[uyvy_buf]),%%xmm4                            \n" \
  "lea        0x10(%[uyvy_buf]),%[uyvy_buf]                   \n" \
  "movdqa     %%xmm4,%%xmm3                                   \n" \
  "pshufb     %%xmm6,%%xmm4                                   \n" \
  "pshufb     %%xmm7,%%xmm3                                   \n"

// Read 4 UV from P210, upsample to 8 UV
#define READP210                                                  \
  "movdqu     (%[uv_buf]),%%xmm3                              \n" \
  "lea        0x10(%[uv_buf]),%[uv_buf]                       \n" \
  "psrlw      $0x8,%%xmm3                                     \n" \
  "packuswb   %%xmm3,%%xmm3                                   \n" \
  "punpcklwd  %%xmm3,%%xmm3                                   \n" \
  "movdqu     (%[y_buf]),%%xmm4                               \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                         \n"

// Read 8 UV from P410
#define READP410                                                  \
  "movdqu     (%[uv_buf]),%%xmm3                              \n" \
  "movdqu     0x10(%[uv_buf]),%%xmm1                          \n" \
  "lea        0x20(%[uv_buf]),%[uv_buf]                       \n" \
  "psrlw      $0x8,%%xmm3                                     \n" \
  "psrlw      $0x8,%%xmm1                                     \n" \
  "packuswb   %%xmm1,%%xmm3                                   \n" \
  "movdqu     (%[y_buf]),%%xmm4                               \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                         \n"

#if defined(__x86_64__)
#define YUVTORGB_SETUP(yuvconstants)                              \
  "pcmpeqb    %%xmm13,%%xmm13                                 \n" \
  "movdqa     (%[yuvconstants]),%%xmm8                        \n" \
  "pxor       %%xmm12,%%xmm12                                 \n" \
  "movdqa     32(%[yuvconstants]),%%xmm9                      \n" \
  "psllw      $7,%%xmm13                                      \n" \
  "movdqa     64(%[yuvconstants]),%%xmm10                     \n" \
  "pshufb     %%xmm12,%%xmm13                                 \n" \
  "movdqa     96(%[yuvconstants]),%%xmm11                     \n" \
  "movdqa     128(%[yuvconstants]),%%xmm12                    \n"

// Convert 8 pixels: 8 UV and 8 Y
#define YUVTORGB16(yuvconstants)                                  \
  "psubb      %%xmm13,%%xmm3                                  \n" \
  "pmulhuw    %%xmm11,%%xmm4                                  \n" \
  "movdqa     %%xmm8,%%xmm0                                   \n" \
  "movdqa     %%xmm9,%%xmm1                                   \n" \
  "movdqa     %%xmm10,%%xmm2                                  \n" \
  "paddw      %%xmm12,%%xmm4                                  \n" \
  "pmaddubsw  %%xmm3,%%xmm0                                   \n" \
  "pmaddubsw  %%xmm3,%%xmm1                                   \n" \
  "pmaddubsw  %%xmm3,%%xmm2                                   \n" \
  "paddsw     %%xmm4,%%xmm0                                   \n" \
  "paddsw     %%xmm4,%%xmm2                                   \n" \
  "psubsw     %%xmm1,%%xmm4                                   \n" \
  "movdqa     %%xmm4,%%xmm1                                   \n"

#define YUVTORGB_REGS "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13",

#else
#define YUVTORGB_SETUP(yuvconstants)

// Convert 8 pixels: 8 UV and 8 Y
#define YUVTORGB16(yuvconstants)                                  \
  "pcmpeqb    %%xmm0,%%xmm0                                   \n" \
  "pxor       %%xmm1,%%xmm1                                   \n" \
  "psllw      $7,%%xmm0                                       \n" \
  "pshufb     %%xmm1,%%xmm0                                   \n" \
  "psubb      %%xmm0,%%xmm3                                   \n" \
  "pmulhuw    96(%[yuvconstants]),%%xmm4                      \n" \
  "movdqa     (%[yuvconstants]),%%xmm0                        \n" \
  "movdqa     32(%[yuvconstants]),%%xmm1                      \n" \
  "movdqa     64(%[yuvconstants]),%%xmm2                      \n" \
  "pmaddubsw  %%xmm3,%%xmm0                                   \n" \
  "pmaddubsw  %%xmm3,%%xmm1                                   \n" \
  "pmaddubsw  %%xmm3,%%xmm2                                   \n" \
  "movdqa     128(%[yuvconstants]),%%xmm3                     \n" \
  "paddw      %%xmm3,%%xmm4                                   \n" \
  "paddsw     %%xmm4,%%xmm0                                   \n" \
  "paddsw     %%xmm4,%%xmm2                                   \n" \
  "psubsw     %%xmm1,%%xmm4                                   \n" \
  "movdqa     %%xmm4,%%xmm1                                   \n"

#define YUVTORGB_REGS
#endif

#define YUVTORGB(yuvconstants)                                    \
  YUVTORGB16(yuvconstants)                                        \
  "psraw      $0x6,%%xmm0                                     \n" \
  "psraw      $0x6,%%xmm1                                     \n" \
  "psraw      $0x6,%%xmm2                                     \n" \
  "packuswb   %%xmm0,%%xmm0                                   \n" \
  "packuswb   %%xmm1,%%xmm1                                   \n" \
  "packuswb   %%xmm2,%%xmm2                                   \n"

// Store 8 ARGB values.
#define STOREARGB                                                  \
  "punpcklbw  %%xmm1,%%xmm0                                    \n" \
  "punpcklbw  %%xmm5,%%xmm2                                    \n" \
  "movdqa     %%xmm0,%%xmm1                                    \n" \
  "punpcklwd  %%xmm2,%%xmm0                                    \n" \
  "punpckhwd  %%xmm2,%%xmm1                                    \n" \
  "movdqu     %%xmm0,(%[dst_argb])                             \n" \
  "movdqu     %%xmm1,0x10(%[dst_argb])                         \n" \
  "lea        0x20(%[dst_argb]), %[dst_argb]                   \n"

// Store 8 RGBA values.
#define STORERGBA                                                  \
  "pcmpeqb   %%xmm5,%%xmm5                                     \n" \
  "punpcklbw %%xmm2,%%xmm1                                     \n" \
  "punpcklbw %%xmm0,%%xmm5                                     \n" \
  "movdqa    %%xmm5,%%xmm0                                     \n" \
  "punpcklwd %%xmm1,%%xmm5                                     \n" \
  "punpckhwd %%xmm1,%%xmm0                                     \n" \
  "movdqu    %%xmm5,(%[dst_rgba])                              \n" \
  "movdqu    %%xmm0,0x10(%[dst_rgba])                          \n" \
  "lea       0x20(%[dst_rgba]),%[dst_rgba]                     \n"

// Store 8 RGB24 values.
#define STORERGB24                                                      \
  "punpcklbw   %%xmm1,%%xmm0                                        \n" \
  "punpcklbw   %%xmm2,%%xmm2                                        \n" \
  "movdqa      %%xmm0,%%xmm1                                        \n" \
  "punpcklwd   %%xmm2,%%xmm0                                        \n" \
  "punpckhwd   %%xmm2,%%xmm1                                        \n" \
  "pshufb      %%xmm5,%%xmm0                                        \n" \
  "pshufb      %%xmm6,%%xmm1                                        \n" \
  "palignr     $0xc,%%xmm0,%%xmm1                                   \n" \
  "movq        %%xmm0,(%[dst_rgb24])                                \n" \
  "movdqu      %%xmm1,0x8(%[dst_rgb24])                             \n" \
  "lea         0x18(%[dst_rgb24]),%[dst_rgb24]                      \n"

// Store 8 AR30 values.
#define STOREAR30                                                  \
  "psraw      $0x4,%%xmm0                                      \n" \
  "psraw      $0x4,%%xmm1                                      \n" \
  "psraw      $0x4,%%xmm2                                      \n" \
  "pminsw     %%xmm7,%%xmm0                                    \n" \
  "pminsw     %%xmm7,%%xmm1                                    \n" \
  "pminsw     %%xmm7,%%xmm2                                    \n" \
  "pmaxsw     %%xmm6,%%xmm0                                    \n" \
  "pmaxsw     %%xmm6,%%xmm1                                    \n" \
  "pmaxsw     %%xmm6,%%xmm2                                    \n" \
  "psllw      $0x4,%%xmm2                                      \n" \
  "movdqa     %%xmm0,%%xmm3                                    \n" \
  "punpcklwd  %%xmm2,%%xmm0                                    \n" \
  "punpckhwd  %%xmm2,%%xmm3                                    \n" \
  "movdqa     %%xmm1,%%xmm2                                    \n" \
  "punpcklwd  %%xmm5,%%xmm1                                    \n" \
  "punpckhwd  %%xmm5,%%xmm2                                    \n" \
  "pslld      $0xa,%%xmm1                                      \n" \
  "pslld      $0xa,%%xmm2                                      \n" \
  "por        %%xmm1,%%xmm0                                    \n" \
  "por        %%xmm2,%%xmm3                                    \n" \
  "movdqu     %%xmm0,(%[dst_ar30])                             \n" \
  "movdqu     %%xmm3,0x10(%[dst_ar30])                         \n" \
  "lea        0x20(%[dst_ar30]), %[dst_ar30]                   \n"

void OMITFP I444ToARGBRow_SSSE3(const uint8_t* y_buf,
                                const uint8_t* u_buf,
                                const uint8_t* v_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"

    LABELALIGN
      "1:          \n"
    READYUV444
    YUVTORGB(yuvconstants)
    STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

#ifdef HAS_I444ALPHATOARGBROW_SSSE3
void OMITFP I444AlphaToARGBRow_SSSE3(const uint8_t* y_buf,
                                     const uint8_t* u_buf,
                                     const uint8_t* v_buf,
                                     const uint8_t* a_buf,
                                     uint8_t* dst_argb,
                                     const struct YuvConstants* yuvconstants,
                                     int width) {
  asm volatile(YUVTORGB_SETUP(
      yuvconstants) "sub         %[u_buf],%[v_buf]             \n"

      LABELALIGN "1:          \n" READYUVA444 YUVTORGB(yuvconstants)
                   STOREARGB
      "subl        $0x8,%[width]                 \n"
      "jg          1b                            \n"
               : [y_buf] "+r"(y_buf),        // %[y_buf]
                 [u_buf] "+r"(u_buf),        // %[u_buf]
                 [v_buf] "+r"(v_buf),        // %[v_buf]
                 [a_buf] "+r"(a_buf),        // %[a_buf]
                 [dst_argb] "+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__)
                 [width] "+m"(width)  // %[width]
#else
                 [width] "+rm"(width)  // %[width]
#endif
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3",
                 "xmm4", "xmm5");
}
#endif  // HAS_I444ALPHATOARGBROW_SSSE3

void OMITFP I422ToRGB24Row_SSSE3(const uint8_t* y_buf,
                                 const uint8_t* u_buf,
                                 const uint8_t* v_buf,
                                 uint8_t* dst_rgb24,
                                 const struct YuvConstants* yuvconstants,
                                 int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "movdqa      %[kShuffleMaskARGBToRGB24_0],%%xmm5 \n"
      "movdqa      %[kShuffleMaskARGBToRGB24],%%xmm6 \n"
      "sub         %[u_buf],%[v_buf]             \n"

    LABELALIGN
      "1:          \n"
    READYUV422
    YUVTORGB(yuvconstants)
    STORERGB24
      "subl        $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_rgb24]"+r"(dst_rgb24),  // %[dst_rgb24]
#if defined(__i386__)
    [width]"+m"(width)     // %[width]
#else
    [width]"+rm"(width)    // %[width]
#endif
  : [yuvconstants]"r"(yuvconstants),  // %[yuvconstants]
    [kShuffleMaskARGBToRGB24_0]"m"(kShuffleMaskARGBToRGB24_0),
    [kShuffleMaskARGBToRGB24]"m"(kShuffleMaskARGBToRGB24)
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}

void OMITFP I444ToRGB24Row_SSSE3(const uint8_t* y_buf,
                                 const uint8_t* u_buf,
                                 const uint8_t* v_buf,
                                 uint8_t* dst_rgb24,
                                 const struct YuvConstants* yuvconstants,
                                 int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "movdqa      %[kShuffleMaskARGBToRGB24_0],%%xmm5 \n"
      "movdqa      %[kShuffleMaskARGBToRGB24],%%xmm6 \n"
      "sub         %[u_buf],%[v_buf]             \n"

    LABELALIGN
      "1:          \n"
    READYUV444
    YUVTORGB(yuvconstants)
    STORERGB24
      "subl        $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_rgb24]"+r"(dst_rgb24),  // %[dst_rgb24]
#if defined(__i386__)
    [width]"+m"(width)     // %[width]
#else
    [width]"+rm"(width)    // %[width]
#endif
  : [yuvconstants]"r"(yuvconstants),  // %[yuvconstants]
    [kShuffleMaskARGBToRGB24_0]"m"(kShuffleMaskARGBToRGB24_0),
    [kShuffleMaskARGBToRGB24]"m"(kShuffleMaskARGBToRGB24)
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}

void OMITFP I422ToARGBRow_SSSE3(const uint8_t* y_buf,
                                const uint8_t* u_buf,
                                const uint8_t* v_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"

    LABELALIGN
      "1:          \n"
    READYUV422
    YUVTORGB(yuvconstants)
    STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void OMITFP I422ToAR30Row_SSSE3(const uint8_t* y_buf,
                                const uint8_t* u_buf,
                                const uint8_t* v_buf,
                                uint8_t* dst_ar30,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"  // AR30 constants
      "psrlw       $14,%%xmm5                    \n"
      "psllw       $4,%%xmm5                     \n"  // 2 alpha bits
      "pxor        %%xmm6,%%xmm6                 \n"  // 0 for min
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psrlw       $6,%%xmm7                     \n"  // 1023 for max

    LABELALIGN
      "1:          \n"
    READYUV422
    YUVTORGB16(yuvconstants)
    STOREAR30
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

// 10 bit YUV to ARGB
void OMITFP I210ToARGBRow_SSSE3(const uint16_t* y_buf,
                                const uint16_t* u_buf,
                                const uint16_t* v_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"

    LABELALIGN
      "1:          \n"
    READYUV210
    YUVTORGB(yuvconstants)
    STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

// 12 bit YUV to ARGB
void OMITFP I212ToARGBRow_SSSE3(const uint16_t* y_buf,
                                const uint16_t* u_buf,
                                const uint16_t* v_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"

    LABELALIGN
      "1:          \n"
    READYUV212
    YUVTORGB(yuvconstants)
    STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

// 10 bit YUV to AR30
void OMITFP I210ToAR30Row_SSSE3(const uint16_t* y_buf,
                                const uint16_t* u_buf,
                                const uint16_t* v_buf,
                                uint8_t* dst_ar30,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $14,%%xmm5                    \n"
      "psllw       $4,%%xmm5                     \n"  // 2 alpha bits
      "pxor        %%xmm6,%%xmm6                 \n"  // 0 for min
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psrlw       $6,%%xmm7                     \n"  // 1023 for max

    LABELALIGN
      "1:          \n"
    READYUV210
    YUVTORGB16(yuvconstants)
    STOREAR30
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

// 12 bit YUV to AR30
void OMITFP I212ToAR30Row_SSSE3(const uint16_t* y_buf,
                                const uint16_t* u_buf,
                                const uint16_t* v_buf,
                                uint8_t* dst_ar30,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $14,%%xmm5                    \n"
      "psllw       $4,%%xmm5                     \n"  // 2 alpha bits
      "pxor        %%xmm6,%%xmm6                 \n"  // 0 for min
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psrlw       $6,%%xmm7                     \n"  // 1023 for max

    LABELALIGN
      "1:          \n"
    READYUV212
    YUVTORGB16(yuvconstants)
    STOREAR30
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

// 10 bit YUV to ARGB
void OMITFP I410ToARGBRow_SSSE3(const uint16_t* y_buf,
                                const uint16_t* u_buf,
                                const uint16_t* v_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"

    LABELALIGN
      "1:          \n"
    READYUV410
    YUVTORGB(yuvconstants)
    STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

#ifdef HAS_I210ALPHATOARGBROW_SSSE3
// 10 bit YUVA to ARGB
void OMITFP I210AlphaToARGBRow_SSSE3(const uint16_t* y_buf,
                                     const uint16_t* u_buf,
                                     const uint16_t* v_buf,
                                     const uint16_t* a_buf,
                                     uint8_t* dst_argb,
                                     const struct YuvConstants* yuvconstants,
                                     int width) {
  asm volatile(YUVTORGB_SETUP(
      yuvconstants) "sub         %[u_buf],%[v_buf]             \n"

      LABELALIGN "1:          \n" READYUVA210 YUVTORGB(yuvconstants)
                   STOREARGB
      "subl        $0x8,%[width]                 \n"
      "jg          1b                            \n"
               : [y_buf] "+r"(y_buf),  // %[y_buf]
                 [u_buf] "+r"(u_buf),  // %[u_buf]
                 [v_buf] "+r"(v_buf),  // %[v_buf]
                 [a_buf] "+r"(a_buf),
                 [dst_argb] "+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__)
                 [width] "+m"(width)  // %[width]
#else
                 [width] "+rm"(width)  // %[width]
#endif
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3",
                 "xmm4", "xmm5");
}
#endif

#ifdef HAS_I410ALPHATOARGBROW_SSSE3
// 10 bit YUVA to ARGB
void OMITFP I410AlphaToARGBRow_SSSE3(const uint16_t* y_buf,
                                     const uint16_t* u_buf,
                                     const uint16_t* v_buf,
                                     const uint16_t* a_buf,
                                     uint8_t* dst_argb,
                                     const struct YuvConstants* yuvconstants,
                                     int width) {
  asm volatile(YUVTORGB_SETUP(
      yuvconstants) "sub         %[u_buf],%[v_buf]             \n"

      LABELALIGN "1:          \n" READYUVA410 YUVTORGB(yuvconstants)
                   STOREARGB
      "subl        $0x8,%[width]                 \n"
      "jg          1b                            \n"
               : [y_buf] "+r"(y_buf),  // %[y_buf]
                 [u_buf] "+r"(u_buf),  // %[u_buf]
                 [v_buf] "+r"(v_buf),  // %[v_buf]
                 [a_buf] "+r"(a_buf),
                 [dst_argb] "+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__)
                 [width] "+m"(width)  // %[width]
#else
                 [width] "+rm"(width)  // %[width]
#endif
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3",
                 "xmm4", "xmm5");
}
#endif

// 10 bit YUV to AR30
void OMITFP I410ToAR30Row_SSSE3(const uint16_t* y_buf,
                                const uint16_t* u_buf,
                                const uint16_t* v_buf,
                                uint8_t* dst_ar30,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $14,%%xmm5                    \n"
      "psllw       $4,%%xmm5                     \n"  // 2 alpha bits
      "pxor        %%xmm6,%%xmm6                 \n"  // 0 for min
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psrlw       $6,%%xmm7                     \n"  // 1023 for max

    LABELALIGN
      "1:          \n"
    READYUV410
    YUVTORGB16(yuvconstants)
    STOREAR30
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)   // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

#ifdef HAS_I422ALPHATOARGBROW_SSSE3
void OMITFP I422AlphaToARGBRow_SSSE3(const uint8_t* y_buf,
                                     const uint8_t* u_buf,
                                     const uint8_t* v_buf,
                                     const uint8_t* a_buf,
                                     uint8_t* dst_argb,
                                     const struct YuvConstants* yuvconstants,
                                     int width) {
  asm volatile(YUVTORGB_SETUP(
      yuvconstants) "sub         %[u_buf],%[v_buf]             \n"

      LABELALIGN "1:          \n" READYUVA422 YUVTORGB(yuvconstants)
                   STOREARGB
      "subl        $0x8,%[width]                 \n"
      "jg          1b                            \n"
               : [y_buf] "+r"(y_buf),        // %[y_buf]
                 [u_buf] "+r"(u_buf),        // %[u_buf]
                 [v_buf] "+r"(v_buf),        // %[v_buf]
                 [a_buf] "+r"(a_buf),        // %[a_buf]
                 [dst_argb] "+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__)
                 [width] "+m"(width)  // %[width]
#else
                 [width] "+rm"(width)  // %[width]
#endif
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3",
                 "xmm4", "xmm5");
}
#endif  // HAS_I422ALPHATOARGBROW_SSSE3

void OMITFP NV12ToARGBRow_SSSE3(const uint8_t* y_buf,
                                const uint8_t* uv_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile(YUVTORGB_SETUP(
      yuvconstants) "pcmpeqb     %%xmm5,%%xmm5                 \n"

      LABELALIGN "1:          \n" READNV12 YUVTORGB(yuvconstants)
                   STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
               : [y_buf] "+r"(y_buf),              // %[y_buf]
                 [uv_buf] "+r"(uv_buf),            // %[uv_buf]
                 [dst_argb] "+r"(dst_argb),        // %[dst_argb]
                 [width] "+rm"(width)              // %[width]
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3",
                 "xmm4", "xmm5");
}

void OMITFP NV21ToARGBRow_SSSE3(const uint8_t* y_buf,
                                const uint8_t* vu_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile(YUVTORGB_SETUP(
      yuvconstants) "pcmpeqb     %%xmm5,%%xmm5                 \n"

      LABELALIGN "1:          \n" READNV21 YUVTORGB(yuvconstants)
                   STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
               : [y_buf] "+r"(y_buf),               // %[y_buf]
                 [vu_buf] "+r"(vu_buf),             // %[vu_buf]
                 [dst_argb] "+r"(dst_argb),         // %[dst_argb]
                 [width] "+rm"(width)               // %[width]
               : [yuvconstants] "r"(yuvconstants),  // %[yuvconstants]
                 [kShuffleNV21] "m"(kShuffleNV21)
               : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3",
                 "xmm4", "xmm5");
}

void OMITFP YUY2ToARGBRow_SSSE3(const uint8_t* yuy2_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile(
      "movdqa      %[kShuffleYUY2Y],%%xmm6       \n"
      "movdqa      %[kShuffleYUY2UV],%%xmm7      \n" YUVTORGB_SETUP(
      yuvconstants) "pcmpeqb     %%xmm5,%%xmm5                 \n"

      LABELALIGN "1:          \n" READYUY2 YUVTORGB(yuvconstants) STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
      : [yuy2_buf] "+r"(yuy2_buf),         // %[yuy2_buf]
        [dst_argb] "+r"(dst_argb),         // %[dst_argb]
        [width] "+rm"(width)               // %[width]
      : [yuvconstants] "r"(yuvconstants),  // %[yuvconstants]
        [kShuffleYUY2Y] "m"(kShuffleYUY2Y), [kShuffleYUY2UV] "m"(kShuffleYUY2UV)
      : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
        "xmm5", "xmm6", "xmm7");
}

void OMITFP UYVYToARGBRow_SSSE3(const uint8_t* uyvy_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile(
      "movdqa      %[kShuffleUYVYY],%%xmm6       \n"
      "movdqa      %[kShuffleUYVYUV],%%xmm7      \n" YUVTORGB_SETUP(
      yuvconstants) "pcmpeqb     %%xmm5,%%xmm5                 \n"

      LABELALIGN "1:          \n" READUYVY YUVTORGB(yuvconstants) STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
      : [uyvy_buf] "+r"(uyvy_buf),         // %[uyvy_buf]
        [dst_argb] "+r"(dst_argb),         // %[dst_argb]
        [width] "+rm"(width)               // %[width]
      : [yuvconstants] "r"(yuvconstants),  // %[yuvconstants]
        [kShuffleUYVYY] "m"(kShuffleUYVYY), [kShuffleUYVYUV] "m"(kShuffleUYVYUV)
      : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3", "xmm4",
        "xmm5");
}

void OMITFP P210ToARGBRow_SSSE3(const uint16_t* y_buf,
                                const uint16_t* uv_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile(YUVTORGB_SETUP(
      yuvconstants) "pcmpeqb     %%xmm5,%%xmm5                 \n"

      LABELALIGN "1:          \n" READP210 YUVTORGB(yuvconstants)
                   STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
               : [y_buf] "+r"(y_buf),              // %[y_buf]
                 [uv_buf] "+r"(uv_buf),            // %[u_buf]
                 [dst_argb] "+r"(dst_argb),        // %[dst_argb]
                 [width] "+rm"(width)              // %[width]
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3",
                 "xmm4", "xmm5");
}

void OMITFP P410ToARGBRow_SSSE3(const uint16_t* y_buf,
                                const uint16_t* uv_buf,
                                uint8_t* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile(YUVTORGB_SETUP(
      yuvconstants) "pcmpeqb     %%xmm5,%%xmm5                 \n"

      LABELALIGN "1:          \n" READP410 YUVTORGB(yuvconstants)
                   STOREARGB
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
               : [y_buf] "+r"(y_buf),              // %[y_buf]
                 [uv_buf] "+r"(uv_buf),            // %[u_buf]
                 [dst_argb] "+r"(dst_argb),        // %[dst_argb]
                 [width] "+rm"(width)              // %[width]
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS "xmm0", "xmm1", "xmm2", "xmm3",
                 "xmm4", "xmm5");
}

void OMITFP P210ToAR30Row_SSSE3(const uint16_t* y_buf,
                                const uint16_t* uv_buf,
                                uint8_t* dst_ar30,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $14,%%xmm5                    \n"
      "psllw       $4,%%xmm5                     \n"  // 2 alpha bits
      "pxor        %%xmm6,%%xmm6                 \n"  // 0 for min
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psrlw       $6,%%xmm7                     \n"  // 1023 for max

    LABELALIGN
      "1:          \n"
    READP210
    YUVTORGB16(yuvconstants)
    STOREAR30
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),              // %[y_buf]
    [uv_buf]"+r"(uv_buf),            // %[uv_buf]
    [dst_ar30]"+r"(dst_ar30),        // %[dst_ar30]
    [width]"+rm"(width)              // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

void OMITFP P410ToAR30Row_SSSE3(const uint16_t* y_buf,
                                const uint16_t* uv_buf,
                                uint8_t* dst_ar30,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $14,%%xmm5                    \n"
      "psllw       $4,%%xmm5                     \n"  // 2 alpha bits
      "pxor        %%xmm6,%%xmm6                 \n"  // 0 for min
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psrlw       $6,%%xmm7                     \n"  // 1023 for max

    LABELALIGN
      "1:          \n"
    READP410
    YUVTORGB16(yuvconstants)
    STOREAR30
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),              // %[y_buf]
    [uv_buf]"+r"(uv_buf),            // %[uv_buf]
    [dst_ar30]"+r"(dst_ar30),        // %[dst_ar30]
    [width]"+rm"(width)              // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

void OMITFP I422ToRGBARow_SSSE3(const uint8_t* y_buf,
                                const uint8_t* u_buf,
                                const uint8_t* v_buf,
                                uint8_t* dst_rgba,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"

    LABELALIGN
      "1:          \n"
    READYUV422
    YUVTORGB(yuvconstants)
    STORERGBA
      "sub         $0x8,%[width]                 \n"
      "jg          1b                            \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_rgba]"+r"(dst_rgba),  // %[dst_rgba]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

#endif  // HAS_I422TOARGBROW_SSSE3

// Read 16 UV from 444
#define READYUV444_AVX2                                               \
  "vmovdqu    (%[u_buf]),%%xmm3                                   \n" \
  "vmovdqu    0x00(%[u_buf],%[v_buf],1),%%xmm1                    \n" \
  "lea        0x10(%[u_buf]),%[u_buf]                             \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                                 \n" \
  "vpermq     $0xd8,%%ymm1,%%ymm1                                 \n" \
  "vpunpcklbw %%ymm1,%%ymm3,%%ymm3                                \n" \
  "vmovdqu    (%[y_buf]),%%xmm4                                   \n" \
  "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n" \
  "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                             \n"

// Read 8 UV from 422, upsample to 16 UV.
#define READYUV422_AVX2                                               \
  "vmovq      (%[u_buf]),%%xmm3                                   \n" \
  "vmovq      0x00(%[u_buf],%[v_buf],1),%%xmm1                    \n" \
  "lea        0x8(%[u_buf]),%[u_buf]                              \n" \
  "vpunpcklbw %%ymm1,%%ymm3,%%ymm3                                \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                                 \n" \
  "vpunpcklwd %%ymm3,%%ymm3,%%ymm3                                \n" \
  "vmovdqu    (%[y_buf]),%%xmm4                                   \n" \
  "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n" \
  "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                             \n"

#define READYUV422_AVX512BW                                           \
  "vmovdqu    (%[u_buf]),%%xmm3                                   \n" \
  "vmovdqu    0x00(%[u_buf],%[v_buf],1),%%xmm1                    \n" \
  "vpermq     %%zmm3,%%zmm16,%%zmm3                               \n" \
  "vpermq     %%zmm1,%%zmm16,%%zmm1                               \n" \
  "lea        0x10(%[u_buf]),%[u_buf]                             \n" \
  "vpunpcklbw %%zmm1,%%zmm3,%%zmm3                                \n" \
  "vpermq     $0xd8,%%zmm3,%%zmm3                                 \n" \
  "vpunpcklwd %%zmm3,%%zmm3,%%zmm3                                \n" \
  "vmovups    (%[y_buf]),%%ymm4                                   \n" \
  "vpermq     %%zmm4,%%zmm17,%%zmm4                               \n" \
  "vpermq     $0xd8,%%zmm4,%%zmm4                                 \n" \
  "vpunpcklbw %%zmm4,%%zmm4,%%zmm4                                \n" \
  "lea        0x20(%[y_buf]),%[y_buf]                             \n"

// Read 8 UV from 210, upsample to 16 UV
// TODO(fbarchard): Consider vshufb to replace pack/unpack
// TODO(fbarchard): Consider vunpcklpd to combine the 2 registers into 1.
#define READYUV210_AVX2                                            \
  "vmovdqu    (%[u_buf]),%%xmm3                                \n" \
  "vmovdqu    0x00(%[u_buf],%[v_buf],1),%%xmm1                 \n" \
  "lea        0x10(%[u_buf]),%[u_buf]                          \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                              \n" \
  "vpermq     $0xd8,%%ymm1,%%ymm1                              \n" \
  "vpunpcklwd %%ymm1,%%ymm3,%%ymm3                             \n" \
  "vpsraw     $2,%%ymm3,%%ymm3                                 \n" \
  "vpackuswb  %%ymm3,%%ymm3,%%ymm3                             \n" \
  "vpunpcklwd %%ymm3,%%ymm3,%%ymm3                             \n" \
  "vmovdqu    (%[y_buf]),%%ymm4                                \n" \
  "vpsllw     $6,%%ymm4,%%ymm2                                 \n" \
  "vpsrlw     $4,%%ymm4,%%ymm4                                 \n" \
  "vpaddw     %%ymm2,%%ymm4,%%ymm4                             \n" \
  "lea        0x20(%[y_buf]),%[y_buf]                          \n"

// Read 8 UV from 210, upsample to 16 UV. With 16 Alpha.
#define READYUVA210_AVX2                                           \
  "vmovdqu    (%[u_buf]),%%xmm3                                \n" \
  "vmovdqu    0x00(%[u_buf],%[v_buf],1),%%xmm1                 \n" \
  "lea        0x10(%[u_buf]),%[u_buf]                          \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                              \n" \
  "vpermq     $0xd8,%%ymm1,%%ymm1                              \n" \
  "vpunpcklwd %%ymm1,%%ymm3,%%ymm3                             \n" \
  "vpsraw     $2,%%ymm3,%%ymm3                                 \n" \
  "vpackuswb  %%ymm3,%%ymm3,%%ymm3                             \n" \
  "vpunpcklwd %%ymm3,%%ymm3,%%ymm3                             \n" \
  "vmovdqu    (%[y_buf]),%%ymm4                                \n" \
  "vpsllw     $6,%%ymm4,%%ymm2                                 \n" \
  "vpsrlw     $4,%%ymm4,%%ymm4                                 \n" \
  "vpaddw     %%ymm2,%%ymm4,%%ymm4                             \n" \
  "lea        0x20(%[y_buf]),%[y_buf]                          \n" \
  "vmovdqu    (%[a_buf]),%%ymm5                                \n" \
  "vpsraw     $2,%%ymm5,%%ymm5                                 \n" \
  "vpackuswb  %%ymm5,%%ymm5,%%ymm5                             \n" \
  "lea        0x20(%[a_buf]),%[a_buf]                          \n"

// Read 16 UV from 410
#define READYUV410_AVX2                                            \
  "vmovdqu    (%[u_buf]),%%ymm3                                \n" \
  "vmovdqu    0x00(%[u_buf],%[v_buf],1),%%ymm2                 \n" \
  "lea        0x20(%[u_buf]),%[u_buf]                          \n" \
  "vpsraw     $2,%%ymm3,%%ymm3                                 \n" \
  "vpsraw     $2,%%ymm2,%%ymm2                                 \n" \
  "vpunpckhwd %%ymm2,%%ymm3,%%ymm1                             \n" \
  "vpunpcklwd %%ymm2,%%ymm3,%%ymm3                             \n" \
  "vpackuswb  %%ymm1,%%ymm3,%%ymm3                             \n" \
  "vmovdqu    (%[y_buf]),%%ymm4                                \n" \
  "vpsllw     $6,%%ymm4,%%ymm2                                 \n" \
  "vpsrlw     $4,%%ymm4,%%ymm4                                 \n" \
  "vpaddw     %%ymm2,%%ymm4,%%ymm4                             \n" \
  "lea        0x20(%[y_buf]),%[y_buf]                          \n"

// Read 8 UV from 212 12 bit, upsample to 16 UV
#define READYUV212_AVX2                                            \
  "vmovdqu    (%[u_buf]),%%xmm3                                \n" \
  "vmovdqu    0x00(%[u_buf],%[v_buf],1),%%xmm1                 \n" \
  "lea        0x10(%[u_buf]),%[u_buf]                          \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                              \n" \
  "vpermq     $0xd8,%%ymm1,%%ymm1                              \n" \
  "vpunpcklwd %%ymm1,%%ymm3,%%ymm3                             \n" \
  "vpsraw     $0x4,%%ymm3,%%ymm3                               \n" \
  "vpackuswb  %%ymm3,%%ymm3,%%ymm3                             \n" \
  "vpunpcklwd %%ymm3,%%ymm3,%%ymm3                             \n" \
  "vmovdqu    (%[y_buf]),%%ymm4                                \n" \
  "vpsllw     $4,%%ymm4,%%ymm2                                 \n" \
  "vpsrlw     $8,%%ymm4,%%ymm4                                 \n" \
  "vpaddw     %%ymm2,%%ymm4,%%ymm4                             \n" \
  "lea        0x20(%[y_buf]),%[y_buf]                          \n"

// Read 16 UV from 410. With 16 Alpha.
#define READYUVA410_AVX2                                           \
  "vmovdqu    (%[u_buf]),%%ymm3                                \n" \
  "vmovdqu    0x00(%[u_buf],%[v_buf],1),%%ymm2                 \n" \
  "lea        0x20(%[u_buf]),%[u_buf]                          \n" \
  "vpsraw     $2,%%ymm3,%%ymm3                                 \n" \
  "vpsraw     $2,%%ymm2,%%ymm2                                 \n" \
  "vpunpckhwd %%ymm2,%%ymm3,%%ymm1                             \n" \
  "vpunpcklwd %%ymm2,%%ymm3,%%ymm3                             \n" \
  "vpackuswb  %%ymm1,%%ymm3,%%ymm3                             \n" \
  "vmovdqu    (%[y_buf]),%%ymm4                                \n" \
  "vpsllw     $6,%%ymm4,%%ymm2                                 \n" \
  "vpsrlw     $4,%%ymm4,%%ymm4                                 \n" \
  "vpaddw     %%ymm2,%%ymm4,%%ymm4                             \n" \
  "lea        0x20(%[y_buf]),%[y_buf]                          \n" \
  "vmovdqu    (%[a_buf]),%%ymm5                                \n" \
  "vpsraw     $2,%%ymm5,%%ymm5                                 \n" \
  "vpackuswb  %%ymm5,%%ymm5,%%ymm5                             \n" \
  "lea        0x20(%[a_buf]),%[a_buf]                          \n"

// Read 16 UV from 444.  With 16 Alpha.
#define READYUVA444_AVX2                                              \
  "vmovdqu    (%[u_buf]),%%xmm3                                   \n" \
  "vmovdqu    0x00(%[u_buf],%[v_buf],1),%%xmm1                    \n" \
  "lea        0x10(%[u_buf]),%[u_buf]                             \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                                 \n" \
  "vpermq     $0xd8,%%ymm1,%%ymm1                                 \n" \
  "vpunpcklbw %%ymm1,%%ymm3,%%ymm3                                \n" \
  "vmovdqu    (%[y_buf]),%%xmm4                                   \n" \
  "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n" \
  "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                             \n" \
  "vmovdqu    (%[a_buf]),%%xmm5                                   \n" \
  "vpermq     $0xd8,%%ymm5,%%ymm5                                 \n" \
  "lea        0x10(%[a_buf]),%[a_buf]                             \n"

// Read 8 UV from 422, upsample to 16 UV.  With 16 Alpha.
#define READYUVA422_AVX2                                              \
  "vmovq      (%[u_buf]),%%xmm3                                   \n" \
  "vmovq      0x00(%[u_buf],%[v_buf],1),%%xmm1                    \n" \
  "lea        0x8(%[u_buf]),%[u_buf]                              \n" \
  "vpunpcklbw %%ymm1,%%ymm3,%%ymm3                                \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                                 \n" \
  "vpunpcklwd %%ymm3,%%ymm3,%%ymm3                                \n" \
  "vmovdqu    (%[y_buf]),%%xmm4                                   \n" \
  "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n" \
  "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                             \n" \
  "vmovdqu    (%[a_buf]),%%xmm5                                   \n" \
  "vpermq     $0xd8,%%ymm5,%%ymm5                                 \n" \
  "lea        0x10(%[a_buf]),%[a_buf]                             \n"

// Read 8 UV from NV12, upsample to 16 UV.
#define READNV12_AVX2                                                 \
  "vmovdqu    (%[uv_buf]),%%xmm3                                  \n" \
  "lea        0x10(%[uv_buf]),%[uv_buf]                           \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                                 \n" \
  "vpunpcklwd %%ymm3,%%ymm3,%%ymm3                                \n" \
  "vmovdqu    (%[y_buf]),%%xmm4                                   \n" \
  "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n" \
  "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                             \n"

// Read 8 VU from NV21, upsample to 16 UV.
#define READNV21_AVX2                                                 \
  "vmovdqu    (%[vu_buf]),%%xmm3                                  \n" \
  "lea        0x10(%[vu_buf]),%[vu_buf]                           \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                                 \n" \
  "vpshufb     %[kShuffleNV21], %%ymm3, %%ymm3                    \n" \
  "vmovdqu    (%[y_buf]),%%xmm4                                   \n" \
  "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n" \
  "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n" \
  "lea        0x10(%[y_buf]),%[y_buf]                             \n"

// Read 4 UV from P210, upsample to 8 UV
#define READP210_AVX2                                                 \
  "vmovdqu    (%[uv_buf]),%%ymm3                                  \n" \
  "lea        0x20(%[uv_buf]),%[uv_buf]                           \n" \
  "vpsrlw     $0x8,%%ymm3,%%ymm3                                  \n" \
  "vpackuswb  %%ymm3,%%ymm3,%%ymm3                                \n" \
  "vpunpcklwd %%ymm3,%%ymm3,%%ymm3                                \n" \
  "vmovdqu    (%[y_buf]),%%ymm4                                   \n" \
  "lea        0x20(%[y_buf]),%[y_buf]                             \n"

// Read 8 UV from P410
#define READP410_AVX2                                                 \
  "vmovdqu    (%[uv_buf]),%%ymm3                                  \n" \
  "vmovdqu    0x20(%[uv_buf]),%%ymm1                              \n" \
  "lea        0x40(%[uv_buf]),%[uv_buf]                           \n" \
  "vpsrlw     $0x8,%%ymm3,%%ymm3                                  \n" \
  "vpsrlw     $0x8,%%ymm1,%%ymm1                                  \n" \
  "vpackuswb  %%ymm1,%%ymm3,%%ymm3                                \n" \
  "vpermq     $0xd8,%%ymm3,%%ymm3                                 \n" \
  "vmovdqu    (%[y_buf]),%%ymm4                                   \n" \
  "lea        0x20(%[y_buf]),%[y_buf]                             \n"

// Read 8 YUY2 with 16 Y and upsample 8 UV to 16 UV.
// ymm6 kShuffleYUY2Y,
// ymm7 kShuffleYUY2UV
#define READYUY2_AVX2                                                 \
  "vmovdqu    (%[yuy2_buf]),%%ymm1                                \n" \
  "vpshufb    %%ymm6,%%ymm1,%%ymm4                                \n" \
  "vpshufb    %%ymm7,%%ymm1,%%ymm3                                \n" \
  "lea        0x20(%[yuy2_buf]),%[yuy2_buf]                       \n"

// Read 8 UYVY with 16 Y and upsample 8 UV to 16 UV.
// ymm6 kShuffleUYVYY,
// ymm7 kShuffleUYVYUV
#define READUYVY_AVX2                                                 \
  "vmovdqu    (%[uyvy_buf]),%%ymm1                                \n" \
  "vpshufb    %%ymm6,%%ymm1,%%ymm4                                \n" \
  "vpshufb    %%ymm7,%%ymm1,%%ymm3                                \n" \
  "lea        0x20(%[uyvy_buf]),%[uyvy_buf]                       \n"

// TODO(fbarchard): Remove broadcastb
#if defined(__x86_64__)
#define YUVTORGB_SETUP_AVX2(yuvconstants)                             \
  "vpcmpeqb    %%xmm13,%%xmm13,%%xmm13                            \n" \
  "vmovdqa     (%[yuvconstants]),%%ymm8                           \n" \
  "vpsllw      $7,%%xmm13,%%xmm13                                 \n" \
  "vmovdqa     32(%[yuvconstants]),%%ymm9                         \n" \
  "vpbroadcastb %%xmm13,%%ymm13                                   \n" \
  "vmovdqa     64(%[yuvconstants]),%%ymm10                        \n" \
  "vmovdqa     96(%[yuvconstants]),%%ymm11                        \n" \
  "vmovdqa     128(%[yuvconstants]),%%ymm12                       \n"

#define YUVTORGB_SETUP_AVX512BW(yuvconstants)                         \
  "vpcmpeqb   %%xmm13,%%xmm13,%%xmm13                             \n" \
  "movdqa     (%[yuvconstants]),%%xmm8                            \n" \
  "vpbroadcastq %%xmm8, %%zmm8                                    \n" \
  "vpsllw     $7,%%xmm13,%%xmm13                                  \n" \
  "vpbroadcastb %%xmm13,%%zmm13                                   \n" \
  "movq       32(%[yuvconstants]),%%xmm9                          \n" \
  "vpbroadcastq %%xmm9,%%zmm9                                     \n" \
  "movq       64(%[yuvconstants]),%%xmm10                         \n" \
  "vpbroadcastq %%xmm10,%%zmm10                                   \n" \
  "movq       96(%[yuvconstants]),%%xmm11                         \n" \
  "vpbroadcastq %%xmm11,%%zmm11                                   \n" \
  "movq       128(%[yuvconstants]),%%xmm12                        \n" \
  "vpbroadcastq %%xmm12,%%zmm12                                   \n" \
  "vmovups    (%[quadsplitperm]),%%zmm16                          \n" \
  "vmovups    (%[dquadsplitperm]),%%zmm17                         \n" \
  "vmovups    (%[unperm]),%%zmm18                                 \n"

#define YUVTORGB16_AVX2(yuvconstants)                                 \
  "vpsubb      %%ymm13,%%ymm3,%%ymm3                              \n" \
  "vpmulhuw    %%ymm11,%%ymm4,%%ymm4                              \n" \
  "vpmaddubsw  %%ymm3,%%ymm8,%%ymm0                               \n" \
  "vpmaddubsw  %%ymm3,%%ymm9,%%ymm1                               \n" \
  "vpmaddubsw  %%ymm3,%%ymm10,%%ymm2                              \n" \
  "vpaddw      %%ymm4,%%ymm12,%%ymm4                              \n" \
  "vpaddsw     %%ymm4,%%ymm0,%%ymm0                               \n" \
  "vpsubsw     %%ymm1,%%ymm4,%%ymm1                               \n" \
  "vpaddsw     %%ymm4,%%ymm2,%%ymm2                               \n"

#define YUVTORGB16_AVX512BW(yuvconstants)                             \
  "vpsubb      %%zmm13,%%zmm3,%%zmm3                              \n" \
  "vpmulhuw    %%zmm11,%%zmm4,%%zmm4                              \n" \
  "vpmaddubsw  %%zmm3,%%zmm8,%%zmm0                               \n" \
  "vpmaddubsw  %%zmm3,%%zmm9,%%zmm1                               \n" \
  "vpmaddubsw  %%zmm3,%%zmm10,%%zmm2                              \n" \
  "vpaddw      %%zmm4,%%zmm12,%%zmm4                              \n" \
  "vpaddsw     %%zmm4,%%zmm0,%%zmm0                               \n" \
  "vpsubsw     %%zmm1,%%zmm4,%%zmm1                               \n" \
  "vpaddsw     %%zmm4,%%zmm2,%%zmm2                               \n"

#define YUVTORGB_REGS_AVX2 "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13",
#define YUVTORGB_REGS_AVX512BW \
  "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm16", "xmm17", "xmm18",

#else  // Convert 16 pixels: 16 UV and 16 Y.

#define YUVTORGB_SETUP_AVX2(yuvconstants)
#define YUVTORGB16_AVX2(yuvconstants)                                 \
  "vpcmpeqb    %%xmm0,%%xmm0,%%xmm0                               \n" \
  "vpsllw      $7,%%xmm0,%%xmm0                                   \n" \
  "vpbroadcastb %%xmm0,%%ymm0                                     \n" \
  "vpsubb      %%ymm0,%%ymm3,%%ymm3                               \n" \
  "vpmulhuw    96(%[yuvconstants]),%%ymm4,%%ymm4                  \n" \
  "vmovdqa     (%[yuvconstants]),%%ymm0                           \n" \
  "vmovdqa     32(%[yuvconstants]),%%ymm1                         \n" \
  "vmovdqa     64(%[yuvconstants]),%%ymm2                         \n" \
  "vpmaddubsw  %%ymm3,%%ymm0,%%ymm0                               \n" \
  "vpmaddubsw  %%ymm3,%%ymm1,%%ymm1                               \n" \
  "vpmaddubsw  %%ymm3,%%ymm2,%%ymm2                               \n" \
  "vmovdqa     128(%[yuvconstants]),%%ymm3                        \n" \
  "vpaddw      %%ymm4,%%ymm3,%%ymm4                               \n" \
  "vpaddsw     %%ymm4,%%ymm0,%%ymm0                               \n" \
  "vpsubsw     %%ymm1,%%ymm4,%%ymm1                               \n" \
  "vpaddsw     %%ymm4,%%ymm2,%%ymm2                               \n"

#define YUVTORGB_REGS_AVX2
#endif

#define YUVTORGB_AVX2(yuvconstants)                                   \
  YUVTORGB16_AVX2(yuvconstants)                                       \
  "vpsraw      $0x6,%%ymm0,%%ymm0                                 \n" \
  "vpsraw      $0x6,%%ymm1,%%ymm1                                 \n" \
  "vpsraw      $0x6,%%ymm2,%%ymm2                                 \n" \
  "vpackuswb   %%ymm0,%%ymm0,%%ymm0                               \n" \
  "vpackuswb   %%ymm1,%%ymm1,%%ymm1                               \n" \
  "vpackuswb   %%ymm2,%%ymm2,%%ymm2                               \n"

#define YUVTORGB_AVX512BW(yuvconstants)                               \
  YUVTORGB16_AVX512BW(yuvconstants)                                   \
  "vpsraw     $0x6,%%zmm0,%%zmm0                                  \n" \
  "vpsraw     $0x6,%%zmm1,%%zmm1                                  \n" \
  "vpsraw     $0x6,%%zmm2,%%zmm2                                  \n" \
  "vpackuswb  %%zmm0,%%zmm0,%%zmm0                                \n" \
  "vpackuswb  %%zmm1,%%zmm1,%%zmm1                                \n" \
  "vpackuswb  %%zmm2,%%zmm2,%%zmm2                                \n"

// Store 16 ARGB values.
#define STOREARGB_AVX2                                                \
  "vpunpcklbw %%ymm1,%%ymm0,%%ymm0                                \n" \
  "vpermq     $0xd8,%%ymm0,%%ymm0                                 \n" \
  "vpunpcklbw %%ymm5,%%ymm2,%%ymm2                                \n" \
  "vpermq     $0xd8,%%ymm2,%%ymm2                                 \n" \
  "vpunpcklwd %%ymm2,%%ymm0,%%ymm1                                \n" \
  "vpunpckhwd %%ymm2,%%ymm0,%%ymm0                                \n" \
  "vmovdqu    %%ymm1,(%[dst_argb])                                \n" \
  "vmovdqu    %%ymm0,0x20(%[dst_argb])                            \n" \
  "lea        0x40(%[dst_argb]), %[dst_argb]                      \n"

// Store 32 ARGB values.
#define STOREARGB_AVX512BW                                            \
  "vpunpcklbw %%zmm1,%%zmm0,%%zmm0                                \n" \
  "vpermq     %%zmm0,%%zmm18,%%zmm0                               \n" \
  "vpunpcklbw %%zmm5,%%zmm2,%%zmm2                                \n" \
  "vpermq     %%zmm2,%%zmm18,%%zmm2                               \n" \
  "vpunpcklwd %%zmm2,%%zmm0,%%zmm1                                \n" \
  "vpunpckhwd %%zmm2,%%zmm0,%%zmm0                                \n" \
  "vmovups    %%zmm1,(%[dst_argb])                                \n" \
  "vmovups    %%zmm0,0x40(%[dst_argb])                            \n" \
  "lea        0x80(%[dst_argb]), %[dst_argb]                      \n"

// Store 16 AR30 values.
#define STOREAR30_AVX2                                                \
  "vpsraw     $0x4,%%ymm0,%%ymm0                                  \n" \
  "vpsraw     $0x4,%%ymm1,%%ymm1                                  \n" \
  "vpsraw     $0x4,%%ymm2,%%ymm2                                  \n" \
  "vpminsw    %%ymm7,%%ymm0,%%ymm0                                \n" \
  "vpminsw    %%ymm7,%%ymm1,%%ymm1                                \n" \
  "vpminsw    %%ymm7,%%ymm2,%%ymm2                                \n" \
  "vpmaxsw    %%ymm6,%%ymm0,%%ymm0                                \n" \
  "vpmaxsw    %%ymm6,%%ymm1,%%ymm1                                \n" \
  "vpmaxsw    %%ymm6,%%ymm2,%%ymm2                                \n" \
  "vpsllw     $0x4,%%ymm2,%%ymm2                                  \n" \
  "vpermq     $0xd8,%%ymm0,%%ymm0                                 \n" \
  "vpermq     $0xd8,%%ymm1,%%ymm1                                 \n" \
  "vpermq     $0xd8,%%ymm2,%%ymm2                                 \n" \
  "vpunpckhwd %%ymm2,%%ymm0,%%ymm3                                \n" \
  "vpunpcklwd %%ymm2,%%ymm0,%%ymm0                                \n" \
  "vpunpckhwd %%ymm5,%%ymm1,%%ymm2                                \n" \
  "vpunpcklwd %%ymm5,%%ymm1,%%ymm1                                \n" \
  "vpslld     $0xa,%%ymm1,%%ymm1                                  \n" \
  "vpslld     $0xa,%%ymm2,%%ymm2                                  \n" \
  "vpor       %%ymm1,%%ymm0,%%ymm0                                \n" \
  "vpor       %%ymm2,%%ymm3,%%ymm3                                \n" \
  "vmovdqu    %%ymm0,(%[dst_ar30])                                \n" \
  "vmovdqu    %%ymm3,0x20(%[dst_ar30])                            \n" \
  "lea        0x40(%[dst_ar30]), %[dst_ar30]                      \n"

#ifdef HAS_I444TOARGBROW_AVX2
// 16 pixels
// 16 UV values with 16 Y producing 16 ARGB (64 bytes).
void OMITFP I444ToARGBRow_AVX2(const uint8_t* y_buf,
                               const uint8_t* u_buf,
                               const uint8_t* v_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

    LABELALIGN
      "1:          \n"
    READYUV444_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I444TOARGBROW_AVX2

#if defined(HAS_I422TOARGBROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP I422ToARGBRow_AVX2(const uint8_t* y_buf,
                               const uint8_t* u_buf,
                               const uint8_t* v_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

    LABELALIGN
      "1:          \n"
    READYUV422_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I422TOARGBROW_AVX2

#if defined(HAS_I422TOARGBROW_AVX512BW)
static const uint64_t kSplitQuadWords[8] = {0, 2, 2, 2, 1, 2, 2, 2};
static const uint64_t kSplitDoubleQuadWords[8] = {0, 1, 4, 4, 2, 3, 4, 4};
static const uint64_t kUnpermuteAVX512[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// 32 pixels
// 16 UV values upsampled to 32 UV, mixed with 32 Y producing 32 ARGB (128
// bytes).
void OMITFP I422ToARGBRow_AVX512BW(const uint8_t* y_buf,
                                   const uint8_t* u_buf,
                                   const uint8_t* v_buf,
                                   uint8_t* dst_argb,
                                   const struct YuvConstants* yuvconstants,
                                   int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX512BW(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%xmm5,%%xmm5,%%xmm5          \n"
      "vpbroadcastq %%xmm5,%%zmm5                \n"

    LABELALIGN
      "1:          \n"
    READYUV422_AVX512BW
    YUVTORGB_AVX512BW(yuvconstants)
    STOREARGB_AVX512BW
      "sub         $0x20,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),                         // %[y_buf]
    [u_buf]"+r"(u_buf),                         // %[u_buf]
    [v_buf]"+r"(v_buf),                         // %[v_buf]
    [dst_argb]"+r"(dst_argb),                   // %[dst_argb]
    [width]"+rm"(width)                         // %[width]
  : [yuvconstants]"r"(yuvconstants),            // %[yuvconstants]
    [quadsplitperm]"r"(kSplitQuadWords),        // %[quadsplitperm]
    [dquadsplitperm]"r"(kSplitDoubleQuadWords), // %[dquadsplitperm]
    [unperm]"r"(kUnpermuteAVX512)               // %[unperm]
  : "memory", "cc", YUVTORGB_REGS_AVX512BW
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I422TOARGBROW_AVX512BW

#if defined(HAS_I422TOAR30ROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 AR30 (64 bytes).
void OMITFP I422ToAR30Row_AVX2(const uint8_t* y_buf,
                               const uint8_t* u_buf,
                               const uint8_t* v_buf,
                               uint8_t* dst_ar30,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"  // AR30 constants
      "vpsrlw      $14,%%ymm5,%%ymm5             \n"
      "vpsllw      $4,%%ymm5,%%ymm5              \n"  // 2 alpha bits
      "vpxor       %%ymm6,%%ymm6,%%ymm6          \n"  // 0 for min
      "vpcmpeqb    %%ymm7,%%ymm7,%%ymm7          \n"  // 1023 for max
      "vpsrlw      $6,%%ymm7,%%ymm7              \n"

    LABELALIGN
      "1:          \n"
    READYUV422_AVX2
    YUVTORGB16_AVX2(yuvconstants)
    STOREAR30_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_I422TOAR30ROW_AVX2

#if defined(HAS_I210TOARGBROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP I210ToARGBRow_AVX2(const uint16_t* y_buf,
                               const uint16_t* u_buf,
                               const uint16_t* v_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

    LABELALIGN
      "1:          \n"
    READYUV210_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I210TOARGBROW_AVX2

#if defined(HAS_I212TOARGBROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP I212ToARGBRow_AVX2(const uint16_t* y_buf,
                               const uint16_t* u_buf,
                               const uint16_t* v_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

    LABELALIGN
      "1:          \n"
    READYUV212_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I212TOARGBROW_AVX2

#if defined(HAS_I210TOAR30ROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 AR30 (64 bytes).
void OMITFP I210ToAR30Row_AVX2(const uint16_t* y_buf,
                               const uint16_t* u_buf,
                               const uint16_t* v_buf,
                               uint8_t* dst_ar30,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"  // AR30 constants
      "vpsrlw      $14,%%ymm5,%%ymm5             \n"
      "vpsllw      $4,%%ymm5,%%ymm5              \n"  // 2 alpha bits
      "vpxor       %%ymm6,%%ymm6,%%ymm6          \n"  // 0 for min
      "vpcmpeqb    %%ymm7,%%ymm7,%%ymm7          \n"  // 1023 for max
      "vpsrlw      $6,%%ymm7,%%ymm7              \n"

    LABELALIGN
      "1:          \n"
    READYUV210_AVX2
    YUVTORGB16_AVX2(yuvconstants)
    STOREAR30_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_I210TOAR30ROW_AVX2

#if defined(HAS_I212TOAR30ROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 AR30 (64 bytes).
void OMITFP I212ToAR30Row_AVX2(const uint16_t* y_buf,
                               const uint16_t* u_buf,
                               const uint16_t* v_buf,
                               uint8_t* dst_ar30,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"  // AR30 constants
      "vpsrlw      $14,%%ymm5,%%ymm5             \n"
      "vpsllw      $4,%%ymm5,%%ymm5              \n"  // 2 alpha bits
      "vpxor       %%ymm6,%%ymm6,%%ymm6          \n"  // 0 for min
      "vpcmpeqb    %%ymm7,%%ymm7,%%ymm7          \n"  // 1023 for max
      "vpsrlw      $6,%%ymm7,%%ymm7              \n"

    LABELALIGN
      "1:          \n"
    READYUV212_AVX2
    YUVTORGB16_AVX2(yuvconstants)
    STOREAR30_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_I212TOAR30ROW_AVX2

#if defined(HAS_I410TOARGBROW_AVX2)
// 16 pixels
// 16 UV values with 16 Y producing 16 ARGB (64 bytes).
void OMITFP I410ToARGBRow_AVX2(const uint16_t* y_buf,
                               const uint16_t* u_buf,
                               const uint16_t* v_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

    LABELALIGN
      "1:          \n"
    READYUV410_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"

  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I410TOARGBROW_AVX2

#if defined(HAS_I210ALPHATOARGBROW_AVX2)
// 16 pixels
// 8 UV, 16 Y and 16 A producing 16 ARGB (64 bytes).
void OMITFP I210AlphaToARGBRow_AVX2(const uint16_t* y_buf,
                                    const uint16_t* u_buf,
                                    const uint16_t* v_buf,
                                    const uint16_t* a_buf,
                                    uint8_t* dst_argb,
                                    const struct YuvConstants* yuvconstants,
                                    int width) {
  asm volatile(YUVTORGB_SETUP_AVX2(
      yuvconstants) "sub         %[u_buf],%[v_buf]             \n"

      LABELALIGN "1:          \n" READYUVA210_AVX2 YUVTORGB_AVX2(
                   yuvconstants) STOREARGB_AVX2
      "subl        $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"

               : [y_buf] "+r"(y_buf),        // %[y_buf]
                 [u_buf] "+r"(u_buf),        // %[u_buf]
                 [v_buf] "+r"(v_buf),        // %[v_buf]
                 [a_buf] "+r"(a_buf),        // %[a_buf]
                 [dst_argb] "+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__)
                 [width] "+m"(width)  // %[width]
#else
                 [width] "+rm"(width)  // %[width]
#endif
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm1", "xmm2",
                 "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_I210TOARGBROW_AVX2

#if defined(HAS_I410ALPHATOARGBROW_AVX2)
// 16 pixels
// 16 UV, 16 Y and 16 A producing 16 ARGB (64 bytes).
void OMITFP I410AlphaToARGBRow_AVX2(const uint16_t* y_buf,
                                    const uint16_t* u_buf,
                                    const uint16_t* v_buf,
                                    const uint16_t* a_buf,
                                    uint8_t* dst_argb,
                                    const struct YuvConstants* yuvconstants,
                                    int width) {
  asm volatile(YUVTORGB_SETUP_AVX2(
      yuvconstants) "sub         %[u_buf],%[v_buf]             \n"

      LABELALIGN "1:          \n" READYUVA410_AVX2 YUVTORGB_AVX2(
                   yuvconstants) STOREARGB_AVX2
      "subl        $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"

               : [y_buf] "+r"(y_buf),        // %[y_buf]
                 [u_buf] "+r"(u_buf),        // %[u_buf]
                 [v_buf] "+r"(v_buf),        // %[v_buf]
                 [a_buf] "+r"(a_buf),        // %[a_buf]
                 [dst_argb] "+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__)
                 [width] "+m"(width)  // %[width]
#else
                 [width] "+rm"(width)  // %[width]
#endif
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm1", "xmm2",
                 "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_I410TOARGBROW_AVX2

#if defined(HAS_I410TOAR30ROW_AVX2)
// 16 pixels
// 16 UV values with 16 Y producing 16 AR30 (64 bytes).
void OMITFP I410ToAR30Row_AVX2(const uint16_t* y_buf,
                               const uint16_t* u_buf,
                               const uint16_t* v_buf,
                               uint8_t* dst_ar30,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"  // AR30 constants
      "vpsrlw      $14,%%ymm5,%%ymm5             \n"
      "vpsllw      $4,%%ymm5,%%ymm5              \n"  // 2 alpha bits
      "vpxor       %%ymm6,%%ymm6,%%ymm6          \n"  // 0 for min
      "vpcmpeqb    %%ymm7,%%ymm7,%%ymm7          \n"  // 1023 for max
      "vpsrlw      $6,%%ymm7,%%ymm7              \n"

    LABELALIGN
      "1:          \n"
    READYUV410_AVX2
    YUVTORGB16_AVX2(yuvconstants)
    STOREAR30_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_I410TOAR30ROW_AVX2

#if defined(HAS_I444ALPHATOARGBROW_AVX2)
// 16 pixels
// 16 UV values with 16 Y and 16 A producing 16 ARGB.
void OMITFP I444AlphaToARGBRow_AVX2(const uint8_t* y_buf,
                                    const uint8_t* u_buf,
                                    const uint8_t* v_buf,
                                    const uint8_t* a_buf,
                                    uint8_t* dst_argb,
                                    const struct YuvConstants* yuvconstants,
                                    int width) {
  asm volatile(YUVTORGB_SETUP_AVX2(
      yuvconstants) "sub         %[u_buf],%[v_buf]             \n"

      LABELALIGN "1:          \n" READYUVA444_AVX2 YUVTORGB_AVX2(
                   yuvconstants) STOREARGB_AVX2
      "subl        $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : [y_buf] "+r"(y_buf),        // %[y_buf]
                 [u_buf] "+r"(u_buf),        // %[u_buf]
                 [v_buf] "+r"(v_buf),        // %[v_buf]
                 [a_buf] "+r"(a_buf),        // %[a_buf]
                 [dst_argb] "+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__)
                 [width] "+m"(width)  // %[width]
#else
                 [width] "+rm"(width)  // %[width]
#endif
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm1", "xmm2",
                 "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_I444ALPHATOARGBROW_AVX2

#if defined(HAS_I422ALPHATOARGBROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y and 16 A producing 16 ARGB.
void OMITFP I422AlphaToARGBRow_AVX2(const uint8_t* y_buf,
                                    const uint8_t* u_buf,
                                    const uint8_t* v_buf,
                                    const uint8_t* a_buf,
                                    uint8_t* dst_argb,
                                    const struct YuvConstants* yuvconstants,
                                    int width) {
  asm volatile(YUVTORGB_SETUP_AVX2(
      yuvconstants) "sub         %[u_buf],%[v_buf]             \n"

      LABELALIGN "1:          \n" READYUVA422_AVX2 YUVTORGB_AVX2(
                   yuvconstants) STOREARGB_AVX2
      "subl        $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : [y_buf] "+r"(y_buf),        // %[y_buf]
                 [u_buf] "+r"(u_buf),        // %[u_buf]
                 [v_buf] "+r"(v_buf),        // %[v_buf]
                 [a_buf] "+r"(a_buf),        // %[a_buf]
                 [dst_argb] "+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__)
                 [width] "+m"(width)  // %[width]
#else
                 [width] "+rm"(width)  // %[width]
#endif
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm1", "xmm2",
                 "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_I422ALPHATOARGBROW_AVX2

#if defined(HAS_I422TORGBAROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 RGBA (64 bytes).
void OMITFP I422ToRGBARow_AVX2(const uint8_t* y_buf,
                               const uint8_t* u_buf,
                               const uint8_t* v_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "sub         %[u_buf],%[v_buf]             \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

    LABELALIGN
      "1:          \n"
    READYUV422_AVX2
    YUVTORGB_AVX2(yuvconstants)

    // Step 3: Weave into RGBA
    "vpunpcklbw %%ymm2,%%ymm1,%%ymm1           \n"
    "vpermq     $0xd8,%%ymm1,%%ymm1            \n"
    "vpunpcklbw %%ymm0,%%ymm5,%%ymm2           \n"
    "vpermq     $0xd8,%%ymm2,%%ymm2            \n"
    "vpunpcklwd %%ymm1,%%ymm2,%%ymm0           \n"
    "vpunpckhwd %%ymm1,%%ymm2,%%ymm1           \n"
    "vmovdqu    %%ymm0,(%[dst_argb])           \n"
    "vmovdqu    %%ymm1,0x20(%[dst_argb])       \n"
    "lea        0x40(%[dst_argb]),%[dst_argb]  \n"
    "sub        $0x10,%[width]                 \n"
    "jg         1b                             \n"
    "vzeroupper                                \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I422TORGBAROW_AVX2

#if defined(HAS_NV12TOARGBROW_AVX2)
// 16 pixels.
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP NV12ToARGBRow_AVX2(const uint8_t* y_buf,
                               const uint8_t* uv_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile(YUVTORGB_SETUP_AVX2(
      yuvconstants) "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

      LABELALIGN "1:          \n" READNV12_AVX2 YUVTORGB_AVX2(
                   yuvconstants) STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : [y_buf] "+r"(y_buf),              // %[y_buf]
                 [uv_buf] "+r"(uv_buf),            // %[uv_buf]
                 [dst_argb] "+r"(dst_argb),        // %[dst_argb]
                 [width] "+rm"(width)              // %[width]
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm0", "xmm1",
                 "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_NV12TOARGBROW_AVX2

#if defined(HAS_NV21TOARGBROW_AVX2)
// 16 pixels.
// 8 VU values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP NV21ToARGBRow_AVX2(const uint8_t* y_buf,
                               const uint8_t* vu_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile(YUVTORGB_SETUP_AVX2(
      yuvconstants) "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

      LABELALIGN "1:          \n" READNV21_AVX2 YUVTORGB_AVX2(
                   yuvconstants) STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : [y_buf] "+r"(y_buf),               // %[y_buf]
                 [vu_buf] "+r"(vu_buf),             // %[vu_buf]
                 [dst_argb] "+r"(dst_argb),         // %[dst_argb]
                 [width] "+rm"(width)               // %[width]
               : [yuvconstants] "r"(yuvconstants),  // %[yuvconstants]
                 [kShuffleNV21] "m"(kShuffleNV21)
               : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm0", "xmm1",
                 "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_NV21TOARGBROW_AVX2

#if defined(HAS_YUY2TOARGBROW_AVX2)
// 16 pixels.
// 8 YUY2 values with 16 Y and 8 UV producing 16 ARGB (64 bytes).
void OMITFP YUY2ToARGBRow_AVX2(const uint8_t* yuy2_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile(
      "vbroadcastf128 %[kShuffleYUY2Y],%%ymm6    \n"
      "vbroadcastf128 %[kShuffleYUY2UV],%%ymm7   \n" YUVTORGB_SETUP_AVX2(
      yuvconstants) "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

      LABELALIGN "1:          \n" READYUY2_AVX2 YUVTORGB_AVX2(yuvconstants)
          STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : [yuy2_buf] "+r"(yuy2_buf),         // %[yuy2_buf]
        [dst_argb] "+r"(dst_argb),         // %[dst_argb]
        [width] "+rm"(width)               // %[width]
      : [yuvconstants] "r"(yuvconstants),  // %[yuvconstants]
        [kShuffleYUY2Y] "m"(kShuffleYUY2Y), [kShuffleYUY2UV] "m"(kShuffleYUY2UV)
      : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm0", "xmm1", "xmm2",
        "xmm3", "xmm4", "xmm5", "xmm6", "xmm7");
}
#endif  // HAS_YUY2TOARGBROW_AVX2

#if defined(HAS_UYVYTOARGBROW_AVX2)
// 16 pixels.
// 8 UYVY values with 16 Y and 8 UV producing 16 ARGB (64 bytes).
void OMITFP UYVYToARGBRow_AVX2(const uint8_t* uyvy_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile(
      "vbroadcastf128 %[kShuffleUYVYY],%%ymm6    \n"
      "vbroadcastf128 %[kShuffleUYVYUV],%%ymm7   \n" YUVTORGB_SETUP_AVX2(
      yuvconstants) "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

      LABELALIGN "1:          \n" READUYVY_AVX2 YUVTORGB_AVX2(yuvconstants)
          STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : [uyvy_buf] "+r"(uyvy_buf),         // %[uyvy_buf]
        [dst_argb] "+r"(dst_argb),         // %[dst_argb]
        [width] "+rm"(width)               // %[width]
      : [yuvconstants] "r"(yuvconstants),  // %[yuvconstants]
        [kShuffleUYVYY] "m"(kShuffleUYVYY), [kShuffleUYVYUV] "m"(kShuffleUYVYUV)
      : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm1", "xmm2", "xmm3",
        "xmm4", "xmm5", "xmm6", "xmm7");
}
#endif  // HAS_UYVYTOARGBROW_AVX2

#if defined(HAS_P210TOARGBROW_AVX2)
// 16 pixels.
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP P210ToARGBRow_AVX2(const uint16_t* y_buf,
                               const uint16_t* uv_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile(YUVTORGB_SETUP_AVX2(
      yuvconstants) "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

      LABELALIGN "1:          \n" READP210_AVX2 YUVTORGB_AVX2(
                   yuvconstants) STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : [y_buf] "+r"(y_buf),              // %[y_buf]
                 [uv_buf] "+r"(uv_buf),            // %[uv_buf]
                 [dst_argb] "+r"(dst_argb),        // %[dst_argb]
                 [width] "+rm"(width)              // %[width]
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm0", "xmm1",
                 "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_P210TOARGBROW_AVX2

#if defined(HAS_P410TOARGBROW_AVX2)
// 16 pixels.
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP P410ToARGBRow_AVX2(const uint16_t* y_buf,
                               const uint16_t* uv_buf,
                               uint8_t* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile(YUVTORGB_SETUP_AVX2(
      yuvconstants) "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"

      LABELALIGN "1:          \n" READP410_AVX2 YUVTORGB_AVX2(
                   yuvconstants) STOREARGB_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : [y_buf] "+r"(y_buf),              // %[y_buf]
                 [uv_buf] "+r"(uv_buf),            // %[uv_buf]
                 [dst_argb] "+r"(dst_argb),        // %[dst_argb]
                 [width] "+rm"(width)              // %[width]
               : [yuvconstants] "r"(yuvconstants)  // %[yuvconstants]
               : "memory", "cc", YUVTORGB_REGS_AVX2 "xmm0", "xmm0", "xmm1",
                 "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_P410TOARGBROW_AVX2

#if defined(HAS_P210TOAR30ROW_AVX2)
// 16 pixels
// 16 UV values with 16 Y producing 16 AR30 (64 bytes).
void OMITFP P210ToAR30Row_AVX2(const uint16_t* y_buf,
                               const uint16_t* uv_buf,
                               uint8_t* dst_ar30,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"  // AR30 constants
      "vpsrlw      $14,%%ymm5,%%ymm5             \n"
      "vpsllw      $4,%%ymm5,%%ymm5              \n"  // 2 alpha bits
      "vpxor       %%ymm6,%%ymm6,%%ymm6          \n"  // 0 for min
      "vpcmpeqb    %%ymm7,%%ymm7,%%ymm7          \n"  // 1023 for max
      "vpsrlw      $6,%%ymm7,%%ymm7              \n"

    LABELALIGN
      "1:          \n"
    READP210_AVX2
    YUVTORGB16_AVX2(yuvconstants)
    STOREAR30_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [uv_buf]"+r"(uv_buf),    // %[uv_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_P210TOAR30ROW_AVX2

#if defined(HAS_P410TOAR30ROW_AVX2)
// 16 pixels
// 16 UV values with 16 Y producing 16 AR30 (64 bytes).
void OMITFP P410ToAR30Row_AVX2(const uint16_t* y_buf,
                               const uint16_t* uv_buf,
                               uint8_t* dst_ar30,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"  // AR30 constants
      "vpsrlw      $14,%%ymm5,%%ymm5             \n"
      "vpsllw      $4,%%ymm5,%%ymm5              \n"  // 2 alpha bits
      "vpxor       %%ymm6,%%ymm6,%%ymm6          \n"  // 0 for min
      "vpcmpeqb    %%ymm7,%%ymm7,%%ymm7          \n"  // 1023 for max
      "vpsrlw      $6,%%ymm7,%%ymm7              \n"

    LABELALIGN
      "1:          \n"
    READP410_AVX2
    YUVTORGB16_AVX2(yuvconstants)
    STOREAR30_AVX2
      "sub         $0x10,%[width]                \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [uv_buf]"+r"(uv_buf),    // %[uv_buf]
    [dst_ar30]"+r"(dst_ar30),  // %[dst_ar30]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", YUVTORGB_REGS_AVX2
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_P410TOAR30ROW_AVX2

#ifdef HAS_I400TOARGBROW_SSE2
void I400ToARGBRow_SSE2(const uint8_t* y_buf,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      "movdqa      96(%3),%%xmm2                 \n"  // yg = 18997 = 1.164
      "movdqa      128(%3),%%xmm3                \n"  // ygb = 1160 = 1.164 * 16
      "pcmpeqb     %%xmm4,%%xmm4                 \n"  // 0xff000000
      "pslld       $0x18,%%xmm4                  \n"

      LABELALIGN
      "1:          \n"
      // Step 1: Scale Y contribution to 8 G values. G = (y - 16) * 1.164
      "movq      (%0),%%xmm0                     \n"
      "lea       0x8(%0),%0                      \n"
      "punpcklbw %%xmm0,%%xmm0                   \n"
      "pmulhuw   %%xmm2,%%xmm0                   \n"
      "paddsw    %%xmm3,%%xmm0                   \n"
      "psraw     $6, %%xmm0                      \n"
      "packuswb  %%xmm0,%%xmm0                   \n"

      // Step 2: Weave into ARGB
      "punpcklbw %%xmm0,%%xmm0                   \n"
      "movdqa    %%xmm0,%%xmm1                   \n"
      "punpcklwd %%xmm0,%%xmm0                   \n"
      "punpckhwd %%xmm1,%%xmm1                   \n"
      "por       %%xmm4,%%xmm0                   \n"
      "por       %%xmm4,%%xmm1                   \n"
      "movdqu    %%xmm0,(%1)                     \n"
      "movdqu    %%xmm1,0x10(%1)                 \n"
      "lea       0x20(%1),%1                     \n"

      "sub       $0x8,%2                         \n"
      "jg        1b                              \n"
      : "+r"(y_buf),       // %0
        "+r"(dst_argb),    // %1
        "+rm"(width)       // %2
      : "r"(yuvconstants)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif  // HAS_I400TOARGBROW_SSE2

#ifdef HAS_I400TOARGBROW_AVX2
// 16 pixels of Y converted to 16 pixels of ARGB (64 bytes).
// note: vpunpcklbw mutates and vpackuswb unmutates.
void I400ToARGBRow_AVX2(const uint8_t* y_buf,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      "vmovdqa     96(%3),%%ymm2                 \n"  // yg = 18997 = 1.164
      "vmovdqa     128(%3),%%ymm3                \n"  // ygb = -1160 = 1.164*16
      "vpcmpeqb    %%ymm4,%%ymm4,%%ymm4          \n"  // 0xff000000
      "vpslld      $0x18,%%ymm4,%%ymm4           \n"

      LABELALIGN
      "1:          \n"
      // Step 1: Scale Y contribution to 16 G values. G = (y - 16) * 1.164
      "vmovdqu    (%0),%%xmm0                    \n"
      "lea        0x10(%0),%0                    \n"
      "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
      "vpunpcklbw %%ymm0,%%ymm0,%%ymm0           \n"
      "vpmulhuw   %%ymm2,%%ymm0,%%ymm0           \n"
      "vpaddsw    %%ymm3,%%ymm0,%%ymm0           \n"
      "vpsraw     $0x6,%%ymm0,%%ymm0             \n"
      "vpackuswb  %%ymm0,%%ymm0,%%ymm0           \n"
      "vpunpcklbw %%ymm0,%%ymm0,%%ymm1           \n"
      "vpermq     $0xd8,%%ymm1,%%ymm1            \n"
      "vpunpcklwd %%ymm1,%%ymm1,%%ymm0           \n"
      "vpunpckhwd %%ymm1,%%ymm1,%%ymm1           \n"
      "vpor       %%ymm4,%%ymm0,%%ymm0           \n"
      "vpor       %%ymm4,%%ymm1,%%ymm1           \n"
      "vmovdqu    %%ymm0,(%1)                    \n"
      "vmovdqu    %%ymm1,0x20(%1)                \n"
      "lea        0x40(%1),%1                     \n"
      "sub        $0x10,%2                       \n"
      "jg        1b                              \n"
      "vzeroupper                                \n"
      : "+r"(y_buf),       // %0
        "+r"(dst_argb),    // %1
        "+rm"(width)       // %2
      : "r"(yuvconstants)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif  // HAS_I400TOARGBROW_AVX2

#ifdef HAS_MIRRORROW_SSSE3
// Shuffle table for reversing the bytes.
static const uvec8 kShuffleMirror = {15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u,
                                     7u,  6u,  5u,  4u,  3u,  2u,  1u, 0u};

void MirrorRow_SSSE3(const uint8_t* src, uint8_t* dst, int width) {
  intptr_t temp_width = (intptr_t)(width);
      asm volatile("movdqa      %3,%%xmm5                     \n"

               LABELALIGN
      "1:          \n"
      "movdqu      -0x10(%0,%2,1),%%xmm0         \n"
      "pshufb      %%xmm5,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
               : "+r"(src),           // %0
                 "+r"(dst),           // %1
                 "+r"(temp_width)     // %2
               : "m"(kShuffleMirror)  // %3
               : "memory", "cc", "xmm0", "xmm5");
}
#endif  // HAS_MIRRORROW_SSSE3

#ifdef HAS_MIRRORROW_AVX2
void MirrorRow_AVX2(const uint8_t* src, uint8_t* dst, int width) {
  intptr_t temp_width = (intptr_t)(width);
      asm volatile("vbroadcastf128 %3,%%ymm5                  \n"

               LABELALIGN
      "1:          \n"
      "vmovdqu     -0x20(%0,%2,1),%%ymm0         \n"
      "vpshufb     %%ymm5,%%ymm0,%%ymm0          \n"
      "vpermq      $0x4e,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src),           // %0
                 "+r"(dst),           // %1
                 "+r"(temp_width)     // %2
               : "m"(kShuffleMirror)  // %3
               : "memory", "cc", "xmm0", "xmm5");
}
#endif  // HAS_MIRRORROW_AVX2

#ifdef HAS_MIRRORUVROW_SSSE3
// Shuffle table for reversing the UV.
static const uvec8 kShuffleMirrorUV = {14u, 15u, 12u, 13u, 10u, 11u, 8u, 9u,
                                       6u,  7u,  4u,  5u,  2u,  3u,  0u, 1u};

void MirrorUVRow_SSSE3(const uint8_t* src_uv, uint8_t* dst_uv, int width) {
  intptr_t temp_width = (intptr_t)(width);
      asm volatile("movdqa      %3,%%xmm5                     \n"

               LABELALIGN
      "1:          \n"
      "movdqu      -0x10(%0,%2,2),%%xmm0         \n"
      "pshufb      %%xmm5,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
               : "+r"(src_uv),          // %0
                 "+r"(dst_uv),          // %1
                 "+r"(temp_width)       // %2
               : "m"(kShuffleMirrorUV)  // %3
               : "memory", "cc", "xmm0", "xmm5");
}
#endif  // HAS_MIRRORUVROW_SSSE3

#ifdef HAS_MIRRORUVROW_AVX2
void MirrorUVRow_AVX2(const uint8_t* src_uv, uint8_t* dst_uv, int width) {
  intptr_t temp_width = (intptr_t)(width);
      asm volatile("vbroadcastf128 %3,%%ymm5                  \n"

               LABELALIGN
      "1:          \n"
      "vmovdqu     -0x20(%0,%2,2),%%ymm0         \n"
      "vpshufb     %%ymm5,%%ymm0,%%ymm0          \n"
      "vpermq      $0x4e,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_uv),          // %0
                 "+r"(dst_uv),          // %1
                 "+r"(temp_width)       // %2
               : "m"(kShuffleMirrorUV)  // %3
               : "memory", "cc", "xmm0", "xmm5");
}
#endif  // HAS_MIRRORUVROW_AVX2

#ifdef HAS_MIRRORSPLITUVROW_SSSE3
// Shuffle table for reversing the bytes of UV channels.
static const uvec8 kShuffleMirrorSplitUV = {14u, 12u, 10u, 8u, 6u, 4u, 2u, 0u,
                                            15u, 13u, 11u, 9u, 7u, 5u, 3u, 1u};
void MirrorSplitUVRow_SSSE3(const uint8_t* src,
                            uint8_t* dst_u,
                            uint8_t* dst_v,
                            int width) {
  intptr_t temp_width = (intptr_t)(width);
  asm volatile(
      "movdqa      %4,%%xmm1                     \n"
      "lea         -0x10(%0,%3,2),%0             \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "lea         -0x10(%0),%0                  \n"
      "pshufb      %%xmm1,%%xmm0                 \n"
      "movlpd      %%xmm0,(%1)                   \n"
      "movhpd      %%xmm0,0x00(%1,%2,1)          \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $8,%3                         \n"
      "jg          1b                            \n"
      : "+r"(src),                  // %0
        "+r"(dst_u),                // %1
        "+r"(dst_v),                // %2
        "+r"(temp_width)            // %3
      : "m"(kShuffleMirrorSplitUV)  // %4
      : "memory", "cc", "xmm0", "xmm1");
}
#endif  // HAS_MIRRORSPLITUVROW_SSSE3

#ifdef HAS_RGB24MIRRORROW_SSSE3

// Shuffle first 5 pixels to last 5 mirrored.  first byte zero
static const uvec8 kShuffleMirrorRGB0 = {128u, 12u, 13u, 14u, 9u, 10u, 11u, 6u,
                                         7u,   8u,  3u,  4u,  5u, 0u,  1u,  2u};

// Shuffle last 5 pixels to first 5 mirrored.  last byte zero
static const uvec8 kShuffleMirrorRGB1 = {
    13u, 14u, 15u, 10u, 11u, 12u, 7u, 8u, 9u, 4u, 5u, 6u, 1u, 2u, 3u, 128u};

// Shuffle 5 pixels at a time (15 bytes)
void RGB24MirrorRow_SSSE3(const uint8_t* src_rgb24,
                          uint8_t* dst_rgb24,
                          int width) {
  intptr_t temp_width = (intptr_t)(width);
  src_rgb24 += width * 3 - 48;
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"  // first 5
      "movdqu      15(%0),%%xmm1                 \n"  // next 5
      "movdqu      30(%0),%%xmm2                 \n"  // next 5
      "movdqu      32(%0),%%xmm3                 \n"  // last 1 special
      "pshufb      %%xmm4,%%xmm0                 \n"
      "pshufb      %%xmm4,%%xmm1                 \n"
      "pshufb      %%xmm4,%%xmm2                 \n"
      "pshufb      %%xmm5,%%xmm3                 \n"
      "lea         -0x30(%0),%0                  \n"
      "movdqu      %%xmm0,32(%1)                 \n"  // last 5
      "movdqu      %%xmm1,17(%1)                 \n"  // next 5
      "movdqu      %%xmm2,2(%1)                  \n"  // next 5
      "movlpd      %%xmm3,0(%1)                  \n"  // first 1
      "lea         0x30(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_rgb24),          // %0
        "+r"(dst_rgb24),          // %1
        "+r"(temp_width)          // %2
      : "m"(kShuffleMirrorRGB0),  // %3
        "m"(kShuffleMirrorRGB1)   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_RGB24MIRRORROW_SSSE3

#ifdef HAS_ARGBMIRRORROW_SSE2

void ARGBMirrorRow_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  intptr_t temp_width = (intptr_t)(width);
      asm volatile("lea         -0x10(%0,%2,4),%0             \n"

               LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "pshufd      $0x1b,%%xmm0,%%xmm0           \n"
      "lea         -0x10(%0),%0                  \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
               : "+r"(src),        // %0
                 "+r"(dst),        // %1
                 "+r"(temp_width)  // %2
               :
               : "memory", "cc", "xmm0");
}
#endif  // HAS_ARGBMIRRORROW_SSE2

#ifdef HAS_ARGBMIRRORROW_AVX2
// Shuffle table for reversing the bytes.
static const ulvec32 kARGBShuffleMirror_AVX2 = {7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u};
void ARGBMirrorRow_AVX2(const uint8_t* src, uint8_t* dst, int width) {
  intptr_t temp_width = (intptr_t)(width);
      asm volatile("vmovdqu     %3,%%ymm5                     \n"

               LABELALIGN
      "1:          \n"
      "vpermd      -0x20(%0,%2,4),%%ymm5,%%ymm0  \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src),                    // %0
                 "+r"(dst),                    // %1
                 "+r"(temp_width)              // %2
               : "m"(kARGBShuffleMirror_AVX2)  // %3
               : "memory", "cc", "xmm0", "xmm5");
}
#endif  // HAS_ARGBMIRRORROW_AVX2

#ifdef HAS_SPLITUVROW_AVX2
void SplitUVRow_AVX2(const uint8_t* src_uv,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  asm volatile(
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"
      "vpsrlw      $0x8,%%ymm5,%%ymm5            \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm2            \n"
      "vpsrlw      $0x8,%%ymm1,%%ymm3            \n"
      "vpand       %%ymm5,%%ymm0,%%ymm0          \n"
      "vpand       %%ymm5,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpackuswb   %%ymm3,%%ymm2,%%ymm2          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpermq      $0xd8,%%ymm2,%%ymm2           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm2,0x00(%1,%2,1)          \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%3                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_uv),  // %0
        "+r"(dst_u),   // %1
        "+r"(dst_v),   // %2
        "+r"(width)    // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}
#endif  // HAS_SPLITUVROW_AVX2

#ifdef HAS_SPLITUVROW_SSE2
void SplitUVRow_SSE2(const uint8_t* src_uv,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $0x8,%%xmm5                   \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "pand        %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "psrlw       $0x8,%%xmm2                   \n"
      "psrlw       $0x8,%%xmm3                   \n"
      "packuswb    %%xmm3,%%xmm2                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm2,0x00(%1,%2,1)          \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_uv),  // %0
        "+r"(dst_u),   // %1
        "+r"(dst_v),   // %2
        "+r"(width)    // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}
#endif  // HAS_SPLITUVROW_SSE2

#ifdef HAS_DETILEROW_SSE2
void DetileRow_SSE2(const uint8_t* src,
                    ptrdiff_t src_tile_stride,
                    uint8_t* dst,
                    int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "sub         $0x10,%2                      \n"
      "lea         (%0,%3),%0                    \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "jg          1b                            \n"
      : "+r"(src),            // %0
        "+r"(dst),            // %1
        "+r"(width)           // %2
      : "r"(src_tile_stride)  // %3
      : "cc", "memory", "xmm0");
}
#endif  // HAS_DETILEROW_SSE2

#ifdef HAS_DETILEROW_16_SSE2
void DetileRow_16_SSE2(const uint16_t* src,
                       ptrdiff_t src_tile_stride,
                       uint16_t* dst,
                       int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         (%0,%3,2),%0                  \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src),            // %0
        "+r"(dst),            // %1
        "+r"(width)           // %2
      : "r"(src_tile_stride)  // %3
      : "cc", "memory", "xmm0", "xmm1");
}
#endif  // HAS_DETILEROW_SSE2

#ifdef HAS_DETILEROW_16_AVX
void DetileRow_16_AVX(const uint16_t* src,
                      ptrdiff_t src_tile_stride,
                      uint16_t* dst,
                      int width) {
  asm volatile(
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "lea         (%0,%3,2),%0                  \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),            // %0
        "+r"(dst),            // %1
        "+r"(width)           // %2
      : "r"(src_tile_stride)  // %3
      : "cc", "memory", "xmm0");
}
#endif  // HAS_DETILEROW_AVX

#ifdef HAS_DETILETOYUY2_SSE2
// Read 16 Y, 8 UV, and write 8 YUYV.
void DetileToYUY2_SSE2(const uint8_t* src_y,
                       ptrdiff_t src_y_tile_stride,
                       const uint8_t* src_uv,
                       ptrdiff_t src_uv_tile_stride,
                       uint8_t* dst_yuy2,
                       int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"  // Load 16 Y
      "sub         $0x10,%3                      \n"
      "lea         (%0,%4),%0                    \n"
      "movdqu      (%1),%%xmm1                   \n"  // Load 8 UV
      "lea         (%1,%5),%1                    \n"
      "movdqu      %%xmm0,%%xmm2                 \n"
      "punpcklbw   %%xmm1,%%xmm0                 \n"
      "punpckhbw   %%xmm1,%%xmm2                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "movdqu      %%xmm2,0x10(%2)               \n"
      "lea         0x20(%2),%2                   \n"
      "jg          1b                            \n"
      : "+r"(src_y),                            // %0
        "+r"(src_uv),                           // %1
        "+r"(dst_yuy2),                         // %2
        "+r"(width)                             // %3
      : "r"(src_y_tile_stride),                 // %4
        "r"(src_uv_tile_stride)                 // %5
      : "cc", "memory", "xmm0", "xmm1", "xmm2"  // Clobber list
  );
}
#endif

#ifdef HAS_DETILESPLITUVROW_SSSE3
// TODO(greenjustin): Look into generating these constants instead of loading
// them since this can cause branch mispredicts for fPIC code on 32-bit
// machines.
static const uvec8 kDeinterlaceUV = {0, 2, 4, 6, 8, 10, 12, 14,
                                     1, 3, 5, 7, 9, 11, 13, 15};

// TODO(greenjustin): Research alternatives to pshufb, since pshufb can be very
// slow on older SSE2 processors.
void DetileSplitUVRow_SSSE3(const uint8_t* src_uv,
                            ptrdiff_t src_tile_stride,
                            uint8_t* dst_u,
                            uint8_t* dst_v,
                            int width) {
  asm volatile(
      "movdqu      %4,%%xmm1                     \n"
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "lea         (%0, %5),%0                   \n"
      "pshufb      %%xmm1,%%xmm0                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"
      "movhps      %%xmm0,(%2)                   \n"
      "lea         0x8(%2),%2                    \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_uv),         // %0
        "+r"(dst_u),          // %1
        "+r"(dst_v),          // %2
        "+r"(width)           // %3
      : "m"(kDeinterlaceUV),  // %4
        "r"(src_tile_stride)  // %5
      : "cc", "memory", "xmm0", "xmm1");
}
#endif  // HAS_DETILESPLITUVROW_SSSE3

#ifdef HAS_MERGEUVROW_AVX512BW
void MergeUVRow_AVX512BW(const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_uv,
                         int width) {
      asm volatile("sub         %0,%1                         \n"

               LABELALIGN
      "1:          \n"
      "vpmovzxbw   (%0),%%zmm0                   \n"
      "vpmovzxbw   0x00(%0,%1,1),%%zmm1          \n"
      "lea         0x20(%0),%0                   \n"
      "vpsllw      $0x8,%%zmm1,%%zmm1            \n"
      "vporq       %%zmm0,%%zmm1,%%zmm2          \n"
      "vmovdqu64   %%zmm2,(%2)                   \n"
      "lea         0x40(%2),%2                   \n"
      "sub         $0x20,%3                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_u),   // %0
                 "+r"(src_v),   // %1
                 "+r"(dst_uv),  // %2
                 "+r"(width)    // %3
               :
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_MERGEUVROW_AVX512BW

#ifdef HAS_MERGEUVROW_AVX2
void MergeUVRow_AVX2(const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* dst_uv,
                     int width) {
      asm volatile("sub         %0,%1                         \n"

               LABELALIGN
      "1:          \n"
      "vpmovzxbw   (%0),%%ymm0                   \n"
      "vpmovzxbw   0x00(%0,%1,1),%%ymm1          \n"
      "lea         0x10(%0),%0                   \n"
      "vpsllw      $0x8,%%ymm1,%%ymm1            \n"
      "vpor        %%ymm0,%%ymm1,%%ymm2          \n"
      "vmovdqu     %%ymm2,(%2)                   \n"
      "lea         0x20(%2),%2                   \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_u),   // %0
                 "+r"(src_v),   // %1
                 "+r"(dst_uv),  // %2
                 "+r"(width)    // %3
               :
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_MERGEUVROW_AVX2

#ifdef HAS_MERGEUVROW_SSE2
void MergeUVRow_SSE2(const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* dst_uv,
                     int width) {
      asm volatile("sub         %0,%1                         \n"

               LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x00(%0,%1,1),%%xmm1          \n"
      "lea         0x10(%0),%0                   \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpcklbw   %%xmm1,%%xmm0                 \n"
      "punpckhbw   %%xmm1,%%xmm2                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "movdqu      %%xmm2,0x10(%2)               \n"
      "lea         0x20(%2),%2                   \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
               : "+r"(src_u),   // %0
                 "+r"(src_v),   // %1
                 "+r"(dst_uv),  // %2
                 "+r"(width)    // %3
               :
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_MERGEUVROW_SSE2

#ifdef HAS_MERGEUVROW_16_AVX2
void MergeUVRow_16_AVX2(const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint16_t* dst_uv,
                        int depth,
                        int width) {
  asm volatile(
      "vmovd       %4,%%xmm3                     \n"
      "vmovd       %5,%%xmm4                     \n"

      "sub         %0,%1                         \n"
      // 8 pixels per loop.

      LABELALIGN
      "1:          \n"
      "vpmovzxwd   (%0),%%ymm0                   \n"
      "vpmovzxwd   0x00(%0,%1,1),%%ymm1          \n"
      "lea         0x10(%0),%0                   \n"
      "vpsllw      %%xmm3,%%ymm0,%%ymm0          \n"
      "vpslld      %%xmm4,%%ymm1,%%ymm1          \n"
      "vpor        %%ymm0,%%ymm1,%%ymm2          \n"
      "vmovdqu     %%ymm2,(%2)                   \n"
      "lea         0x20(%2),%2                   \n"
      "sub         $0x8,%3                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_u),      // %0
        "+r"(src_v),      // %1
        "+r"(dst_uv),     // %2
        "+r"(width)       // %3
      : "r"(16 - depth),  // %4
        "r"(32 - depth)   // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif  // HAS_MERGEUVROW_AVX2

#ifdef HAS_SPLITUVROW_16_AVX2
const uvec8 kSplitUVShuffle16 = {0, 1, 4, 5, 8,  9,  12, 13,
                                 2, 3, 6, 7, 10, 11, 14, 15};
void SplitUVRow_16_AVX2(const uint16_t* src_uv,
                        uint16_t* dst_u,
                        uint16_t* dst_v,
                        int depth,
                        int width) {
  depth = 16 - depth;
  asm volatile(
      "vmovd       %4,%%xmm3                     \n"
      "vbroadcastf128 %5,%%ymm4                  \n"
      "sub         %1,%2                         \n"

      // 16 pixels per loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "add         $0x40,%0                      \n"

      "vpsrlw      %%xmm3,%%ymm0,%%ymm0          \n"
      "vpsrlw      %%xmm3,%%ymm1,%%ymm1          \n"
      "vpshufb     %%ymm4,%%ymm0,%%ymm0          \n"
      "vpshufb     %%ymm4,%%ymm1,%%ymm1          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpermq      $0xd8,%%ymm1,%%ymm1           \n"
      "vextractf128 $0x0,%%ymm0,(%1)             \n"
      "vextractf128 $0x0,%%ymm1,0x10(%1)         \n"
      "vextractf128 $0x1,%%ymm0,(%1,%2)          \n"
      "vextractf128 $0x1,%%ymm1,0x10(%1,%2)      \n"
      "add         $0x20,%1                      \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_uv),           // %0
        "+r"(dst_u),            // %1
        "+r"(dst_v),            // %2
        "+r"(width)             // %3
      : "r"(depth),             // %4
        "m"(kSplitUVShuffle16)  // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif  // HAS_SPLITUVROW_16_AVX2

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 128 = 9 bits
// 64 = 10 bits
// 16 = 12 bits
// 1 = 16 bits
#ifdef HAS_MULTIPLYROW_16_AVX2
void MultiplyRow_16_AVX2(const uint16_t* src_y,
                         uint16_t* dst_y,
                         int scale,
                         int width) {
  asm volatile(
      "vmovd       %3,%%xmm3                     \n"
      "vpbroadcastw %%xmm3,%%ymm3                \n"
      "sub         %0,%1                         \n"

      // 32 pixels per loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vpmullw     %%ymm3,%%ymm0,%%ymm0          \n"
      "vpmullw     %%ymm3,%%ymm1,%%ymm1          \n"
      "vmovdqu     %%ymm0,(%0,%1)                \n"
      "vmovdqu     %%ymm1,0x20(%0,%1)            \n"
      "add         $0x40,%0                      \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale)    // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm3");
}
#endif  // HAS_MULTIPLYROW_16_AVX2

// Use scale to convert msb formats to lsb, depending how many bits there are:
// 512 = 9 bits
// 1024 = 10 bits
// 4096 = 12 bits
// 65536 = 16 bits
#ifdef HAS_DIVIDEROW_16_AVX2
void DivideRow_16_AVX2(const uint16_t* src_y,
                       uint16_t* dst_y,
                       int scale,
                       int width) {
  asm volatile(
      "vmovd       %3,%%xmm3                     \n"
      "vpbroadcastw %%xmm3,%%ymm3                \n"
      "sub         %0,%1                         \n"

      // 32 pixels per loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vpmulhuw    %%ymm3,%%ymm0,%%ymm0          \n"
      "vpmulhuw    %%ymm3,%%ymm1,%%ymm1          \n"
      "vmovdqu     %%ymm0,(%0,%1)                \n"
      "vmovdqu     %%ymm1,0x20(%0,%1)            \n"
      "add         $0x40,%0                      \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width),  // %2
        "+r"(scale)   // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm3");
}
#endif  // HAS_MULTIPLYROW_16_AVX2

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 32768 = 9 bits
// 16384 = 10 bits
// 4096 = 12 bits
// 256 = 16 bits
void Convert16To8Row_SSSE3(const uint16_t* src_y,
                           uint8_t* dst_y,
                           int scale,
                           int width) {
  asm volatile(
      "movd        %3,%%xmm2                     \n"
      "punpcklwd   %%xmm2,%%xmm2                 \n"
      "pshufd      $0x0,%%xmm2,%%xmm2            \n"

      // 32 pixels per loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "add         $0x20,%0                      \n"
      "pmulhuw     %%xmm2,%%xmm0                 \n"
      "pmulhuw     %%xmm2,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "add         $0x10,%1                      \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale)    // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}

#ifdef HAS_CONVERT16TO8ROW_AVX2
void Convert16To8Row_AVX2(const uint16_t* src_y,
                          uint8_t* dst_y,
                          int scale,
                          int width) {
  asm volatile(
      "vmovd       %3,%%xmm2                     \n"
      "vpbroadcastw %%xmm2,%%ymm2                \n"

      // 32 pixels per loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "add         $0x40,%0                      \n"
      "vpmulhuw    %%ymm2,%%ymm0,%%ymm0          \n"
      "vpmulhuw    %%ymm2,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"  // mutates
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "add         $0x20,%1                      \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale)    // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_CONVERT16TO8ROW_AVX2

#ifdef HAS_CONVERT16TO8ROW_AVX512BW
void Convert16To8Row_AVX512BW(const uint16_t* src_y,
                              uint8_t* dst_y,
                              int scale,
                              int width) {
      asm volatile("vpbroadcastw %3,%%zmm2                    \n"

               // 64 pixels per loop.
               LABELALIGN
      "1:          \n"
      "vmovups     (%0),%%zmm0                   \n"
      "vmovups     0x40(%0),%%zmm1               \n"
      "add         $0x80,%0                      \n"
      "vpmulhuw    %%zmm2,%%zmm0,%%zmm0          \n"
      "vpmulhuw    %%zmm2,%%zmm1,%%zmm1          \n"
      "vpmovuswb   %%zmm0,%%ymm0                 \n"
      "vpmovuswb   %%zmm1,%%ymm1                 \n"
      "vmovups     %%ymm0,(%1)                   \n"
      "vmovups     %%ymm1,0x20(%1)               \n"
      "add         $0x40,%1                      \n"
      "sub         $0x40,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_y),  // %0
                 "+r"(dst_y),  // %1
                 "+r"(width)   // %2
               : "r"(scale)    // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_CONVERT16TO8ROW_AVX2

// Use scale to convert to lsb formats depending how many bits there are:
// 512 = 9 bits
// 1024 = 10 bits
// 4096 = 12 bits
void Convert8To16Row_SSE2(const uint8_t* src_y,
                          uint16_t* dst_y,
                          int scale,
                          int width) {
  asm volatile(
      "movd        %3,%%xmm2                     \n"
      "punpcklwd   %%xmm2,%%xmm2                 \n"
      "pshufd      $0x0,%%xmm2,%%xmm2            \n"

      // 32 pixels per loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm0,%%xmm0                 \n"
      "punpckhbw   %%xmm1,%%xmm1                 \n"
      "add         $0x10,%0                      \n"
      "pmulhuw     %%xmm2,%%xmm0                 \n"
      "pmulhuw     %%xmm2,%%xmm1                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "add         $0x20,%1                      \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale)    // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}

#ifdef HAS_CONVERT8TO16ROW_AVX2
void Convert8To16Row_AVX2(const uint8_t* src_y,
                          uint16_t* dst_y,
                          int scale,
                          int width) {
  asm volatile(
      "vmovd       %3,%%xmm2                     \n"
      "vpbroadcastw %%xmm2,%%ymm2                \n"

      // 32 pixels per loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "add         $0x20,%0                      \n"
      "vpunpckhbw  %%ymm0,%%ymm0,%%ymm1          \n"
      "vpunpcklbw  %%ymm0,%%ymm0,%%ymm0          \n"
      "vpmulhuw    %%ymm2,%%ymm0,%%ymm0          \n"
      "vpmulhuw    %%ymm2,%%ymm1,%%ymm1          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "add         $0x40,%1                      \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale)    // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_CONVERT8TO16ROW_AVX2

#ifdef HAS_SPLITRGBROW_SSSE3
// Shuffle table for converting RGB to Planar.
static const uvec8 kSplitRGBShuffle[9] = {
    {0u, 3u, 6u, 9u, 12u, 15u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u,
     128u, 128u},
    {128u, 128u, 128u, 128u, 128u, 128u, 2u, 5u, 8u, 11u, 14u, 128u, 128u, 128u,
     128u, 128u},
    {128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 1u, 4u,
     7u, 10u, 13u},
    {1u, 4u, 7u, 10u, 13u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u,
     128u, 128u},
    {128u, 128u, 128u, 128u, 128u, 0u, 3u, 6u, 9u, 12u, 15u, 128u, 128u, 128u,
     128u, 128u},
    {128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 2u, 5u,
     8u, 11u, 14u},
    {2u, 5u, 8u, 11u, 14u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u,
     128u, 128u},
    {128u, 128u, 128u, 128u, 128u, 1u, 4u, 7u, 10u, 13u, 128u, 128u, 128u, 128u,
     128u, 128u},
    {128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u, 0u, 3u, 6u, 9u,
     12u, 15u}};

void SplitRGBRow_SSSE3(const uint8_t* src_rgb,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm2               \n"
      "pshufb      0(%5), %%xmm0                 \n"
      "pshufb      16(%5), %%xmm1                \n"
      "pshufb      32(%5), %%xmm2                \n"
      "por         %%xmm1,%%xmm0                 \n"
      "por         %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"

      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm2               \n"
      "pshufb      48(%5),%%xmm0                 \n"
      "pshufb      64(%5),%%xmm1                 \n"
      "pshufb      80(%5), %%xmm2                \n"
      "por         %%xmm1,%%xmm0                 \n"
      "por         %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"

      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm2               \n"
      "pshufb      96(%5), %%xmm0                \n"
      "pshufb      112(%5), %%xmm1               \n"
      "pshufb      128(%5), %%xmm2               \n"
      "por         %%xmm1,%%xmm0                 \n"
      "por         %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%3)                   \n"
      "lea         0x10(%3),%3                   \n"
      "lea         0x30(%0),%0                   \n"
      "sub         $0x10,%4                      \n"
      "jg          1b                            \n"
      : "+r"(src_rgb),             // %0
        "+r"(dst_r),               // %1
        "+r"(dst_g),               // %2
        "+r"(dst_b),               // %3
        "+r"(width)                // %4
      : "r"(&kSplitRGBShuffle[0])  // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_SPLITRGBROW_SSSE3

#ifdef HAS_SPLITRGBROW_SSE41
// Shuffle table for converting RGB to Planar, SSE4.1. Note: these are used for
// the AVX2 implementation as well.
static const uvec8 kSplitRGBShuffleSSE41[5] = {
    {0u, 3u, 6u, 9u, 12u, 15u, 2u, 5u, 8u, 11u, 14u, 1u, 4u, 7u, 10u, 13u},
    {1u, 4u, 7u, 10u, 13u, 0u, 3u, 6u, 9u, 12u, 15u, 2u, 5u, 8u, 11u, 14u},
    {2u, 5u, 8u, 11u, 14u, 1u, 4u, 7u, 10u, 13u, 0u, 3u, 6u, 9u, 12u, 15u},
    {0u, 128u, 0u, 0u, 128u, 0u, 0u, 128u, 0u, 0u, 128u, 0u, 0u, 128u, 0u, 0u},
    {0u, 0u, 128u, 0u, 0u, 128u, 0u, 0u, 128u, 0u, 0u, 128u, 0u, 0u, 128u, 0u},
};

void SplitRGBRow_SSE41(const uint8_t* src_rgb,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       int width) {
  asm volatile(
      "movdqa      48(%5), %%xmm0                \n"
      "1:          \n"
      "movdqu      (%0),%%xmm1                   \n"
      "movdqu      0x10(%0),%%xmm2               \n"
      "movdqu      0x20(%0),%%xmm3               \n"
      "lea         0x30(%0),%0                   \n"
      "movdqa      %%xmm1, %%xmm4                \n"
      "pblendvb    %%xmm3, %%xmm1                \n"
      "pblendvb    %%xmm2, %%xmm3                \n"
      "pblendvb    %%xmm4, %%xmm2                \n"
      "palignr     $0xF, %%xmm0, %%xmm0          \n"
      "pblendvb    %%xmm2, %%xmm1                \n"
      "pblendvb    %%xmm3, %%xmm2                \n"
      "pblendvb    %%xmm4, %%xmm3                \n"
      "palignr     $0x1, %%xmm0, %%xmm0          \n"
      "pshufb      0(%5), %%xmm1                 \n"
      "pshufb      16(%5), %%xmm2                \n"
      "pshufb      32(%5), %%xmm3                \n"
      "movdqu      %%xmm1,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "movdqu      %%xmm2,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "movdqu      %%xmm3,(%3)                   \n"
      "lea         0x10(%3),%3                   \n"
      "sub         $0x10,%4                      \n"
      "jg          1b                            \n"
      : "+r"(src_rgb),                  // %0
        "+r"(dst_r),                    // %1
        "+r"(dst_g),                    // %2
        "+r"(dst_b),                    // %3
        "+r"(width)                     // %4
      : "r"(&kSplitRGBShuffleSSE41[0])  // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif  // HAS_SPLITRGBROW_SSE41

#ifdef HAS_SPLITRGBROW_AVX2
void SplitRGBRow_AVX2(const uint8_t* src_rgb,
                      uint8_t* dst_r,
                      uint8_t* dst_g,
                      uint8_t* dst_b,
                      int width) {
  asm volatile(
      "vbroadcasti128 48(%5), %%ymm0             \n"
      "vbroadcasti128 64(%5), %%ymm7             \n"
#if defined(__x86_64__)
      "vbroadcasti128 0(%5), %%ymm8              \n"
      "vbroadcasti128 16(%5), %%ymm9             \n"
      "vbroadcasti128 32(%5), %%ymm10            \n"
#endif
      "1:          \n"
      "vmovdqu     (%0),%%ymm4                   \n"
      "vmovdqu     0x20(%0),%%ymm5               \n"
      "vmovdqu     0x40(%0),%%ymm6               \n"
      "lea         0x60(%0),%0                   \n"
      "vpblendd    $240, %%ymm5, %%ymm4, %%ymm1  \n"
      "vperm2i128  $33, %%ymm6, %%ymm4, %%ymm2   \n"
      "vpblendd    $240, %%ymm6, %%ymm5, %%ymm3  \n"
      "vpblendvb   %%ymm0, %%ymm3, %%ymm1, %%ymm4 \n"
      "vpblendvb   %%ymm0, %%ymm1, %%ymm2, %%ymm5 \n"
      "vpblendvb   %%ymm0, %%ymm2, %%ymm3, %%ymm6 \n"
      "vpblendvb   %%ymm7, %%ymm5, %%ymm4, %%ymm1 \n"
      "vpblendvb   %%ymm7, %%ymm6, %%ymm5, %%ymm2 \n"
      "vpblendvb   %%ymm7, %%ymm4, %%ymm6, %%ymm3 \n"
#if defined(__x86_64__)
      "vpshufb     %%ymm8, %%ymm1, %%ymm1        \n"
      "vpshufb     %%ymm9, %%ymm2, %%ymm2        \n"
      "vpshufb     %%ymm10, %%ymm3, %%ymm3       \n"
#else
      "vbroadcasti128 0(%5), %%ymm4              \n"
      "vbroadcasti128 16(%5), %%ymm5             \n"
      "vbroadcasti128 32(%5), %%ymm6             \n"
      "vpshufb     %%ymm4, %%ymm1, %%ymm1        \n"
      "vpshufb     %%ymm5, %%ymm2, %%ymm2        \n"
      "vpshufb     %%ymm6, %%ymm3, %%ymm3        \n"
#endif
      "vmovdqu     %%ymm1,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "vmovdqu     %%ymm2,(%2)                   \n"
      "lea         0x20(%2),%2                   \n"
      "vmovdqu     %%ymm3,(%3)                   \n"
      "lea         0x20(%3),%3                   \n"
      "sub         $0x20,%4                      \n"
      "jg          1b                            \n"
      : "+r"(src_rgb),                  // %0
        "+r"(dst_r),                    // %1
        "+r"(dst_g),                    // %2
        "+r"(dst_b),                    // %3
        "+r"(width)                     // %4
      : "r"(&kSplitRGBShuffleSSE41[0])  // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7"
#if defined(__x86_64__)
        ,
        "xmm8", "xmm9", "xmm10"
#endif
  );
}
#endif  // HAS_SPLITRGBROW_AVX2

#ifdef HAS_MERGERGBROW_SSSE3
// Shuffle table for converting Planar to RGB.
static const uvec8 kMergeRGBShuffle[9] = {
    {0u, 128u, 128u, 1u, 128u, 128u, 2u, 128u, 128u, 3u, 128u, 128u, 4u, 128u,
     128u, 5u},
    {128u, 0u, 128u, 128u, 1u, 128u, 128u, 2u, 128u, 128u, 3u, 128u, 128u, 4u,
     128u, 128u},
    {128u, 128u, 0u, 128u, 128u, 1u, 128u, 128u, 2u, 128u, 128u, 3u, 128u, 128u,
     4u, 128u},
    {128u, 128u, 6u, 128u, 128u, 7u, 128u, 128u, 8u, 128u, 128u, 9u, 128u, 128u,
     10u, 128u},
    {5u, 128u, 128u, 6u, 128u, 128u, 7u, 128u, 128u, 8u, 128u, 128u, 9u, 128u,
     128u, 10u},
    {128u, 5u, 128u, 128u, 6u, 128u, 128u, 7u, 128u, 128u, 8u, 128u, 128u, 9u,
     128u, 128u},
    {128u, 11u, 128u, 128u, 12u, 128u, 128u, 13u, 128u, 128u, 14u, 128u, 128u,
     15u, 128u, 128u},
    {128u, 128u, 11u, 128u, 128u, 12u, 128u, 128u, 13u, 128u, 128u, 14u, 128u,
     128u, 15u, 128u},
    {10u, 128u, 128u, 11u, 128u, 128u, 12u, 128u, 128u, 13u, 128u, 128u, 14u,
     128u, 128u, 15u}};

void MergeRGBRow_SSSE3(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       uint8_t* dst_rgb,
                       int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      (%1),%%xmm1                   \n"
      "movdqu      (%2),%%xmm2                   \n"
      "pshufb      (%5), %%xmm0                  \n"
      "pshufb      16(%5), %%xmm1                \n"
      "pshufb      32(%5), %%xmm2                \n"
      "por         %%xmm1,%%xmm0                 \n"
      "por         %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%3)                   \n"

      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      (%1),%%xmm1                   \n"
      "movdqu      (%2),%%xmm2                   \n"
      "pshufb      48(%5), %%xmm0                \n"
      "pshufb      64(%5), %%xmm1                \n"
      "pshufb      80(%5), %%xmm2                \n"
      "por         %%xmm1,%%xmm0                 \n"
      "por         %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,16(%3)                 \n"

      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      (%1),%%xmm1                   \n"
      "movdqu      (%2),%%xmm2                   \n"
      "pshufb      96(%5), %%xmm0                \n"
      "pshufb      112(%5), %%xmm1               \n"
      "pshufb      128(%5), %%xmm2               \n"
      "por         %%xmm1,%%xmm0                 \n"
      "por         %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,32(%3)                 \n"

      "lea         0x10(%0),%0                   \n"
      "lea         0x10(%1),%1                   \n"
      "lea         0x10(%2),%2                   \n"
      "lea         0x30(%3),%3                   \n"
      "sub         $0x10,%4                      \n"
      "jg          1b                            \n"
      : "+r"(src_r),               // %0
        "+r"(src_g),               // %1
        "+r"(src_b),               // %2
        "+r"(dst_rgb),             // %3
        "+r"(width)                // %4
      : "r"(&kMergeRGBShuffle[0])  // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_MERGERGBROW_SSSE3

#ifdef HAS_MERGEARGBROW_SSE2
void MergeARGBRow_SSE2(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       const uint8_t* src_a,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "sub         %0,%2                         \n"
      "sub         %0,%3                         \n"

      LABELALIGN
      "1:          \n"

      "movq        (%0,%2),%%xmm0                \n"  // B
      "movq        (%0),%%xmm1                   \n"  // R
      "movq        (%0,%1),%%xmm2                \n"  // G
      "punpcklbw   %%xmm1,%%xmm0                 \n"  // BR
      "movq        (%0,%3),%%xmm1                \n"  // A
      "punpcklbw   %%xmm1,%%xmm2                 \n"  // GA
      "movdqa      %%xmm0,%%xmm1                 \n"  // BR
      "punpckhbw   %%xmm2,%%xmm1                 \n"  // BGRA (hi)
      "punpcklbw   %%xmm2,%%xmm0                 \n"  // BGRA (lo)
      "movdqu      %%xmm0,(%4)                   \n"
      "movdqu      %%xmm1,16(%4)                 \n"

      "lea         8(%0),%0                      \n"
      "lea         32(%4),%4                     \n"
      "sub         $0x8,%5                       \n"
      "jg          1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(src_a),     // %3
        "+r"(dst_argb),  // %4
        "+r"(width)      // %5
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif

#ifdef HAS_MERGEXRGBROW_SSE2
void MergeXRGBRow_SSE2(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "1:          \n"

      "movq        (%2),%%xmm0                   \n"  // B
      "movq        (%0),%%xmm1                   \n"  // R
      "movq        (%1),%%xmm2                   \n"  // G
      "punpcklbw   %%xmm1,%%xmm0                 \n"  // BR
      "pcmpeqd     %%xmm1,%%xmm1                 \n"  // A(255)
      "punpcklbw   %%xmm1,%%xmm2                 \n"  // GA
      "movdqa      %%xmm0,%%xmm1                 \n"  // BR
      "punpckhbw   %%xmm2,%%xmm1                 \n"  // BGRA (hi)
      "punpcklbw   %%xmm2,%%xmm0                 \n"  // BGRA (lo)
      "movdqu      %%xmm0,(%3)                   \n"
      "movdqu      %%xmm1,16(%3)                 \n"

      "lea         8(%0),%0                      \n"
      "lea         8(%1),%1                      \n"
      "lea         8(%2),%2                      \n"
      "lea         32(%3),%3                     \n"
      "sub         $0x8,%4                       \n"
      "jg          1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_argb),  // %3
        "+r"(width)      // %4
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_MERGEARGBROW_SSE2

#ifdef HAS_MERGEARGBROW_AVX2
void MergeARGBRow_AVX2(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       const uint8_t* src_a,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "sub         %0,%2                         \n"
      "sub         %0,%3                         \n"

      LABELALIGN
      "1:          \n"

      "vmovdqu     (%0,%2),%%xmm0                \n"  // B
      "vmovdqu     (%0,%1),%%xmm1                \n"  // R
      "vinserti128 $1,(%0),%%ymm0,%%ymm0         \n"  // G
      "vinserti128 $1,(%0,%3),%%ymm1,%%ymm1      \n"  // A
      "vpunpckhbw  %%ymm1,%%ymm0,%%ymm2          \n"
      "vpunpcklbw  %%ymm1,%%ymm0,%%ymm0          \n"
      "vperm2i128  $0x31,%%ymm2,%%ymm0,%%ymm1    \n"
      "vperm2i128  $0x20,%%ymm2,%%ymm0,%%ymm0    \n"
      "vpunpckhwd  %%ymm1,%%ymm0,%%ymm2          \n"
      "vpunpcklwd  %%ymm1,%%ymm0,%%ymm0          \n"
      "vperm2i128  $0x31,%%ymm2,%%ymm0,%%ymm1    \n"
      "vperm2i128  $0x20,%%ymm2,%%ymm0,%%ymm0    \n"
      "vmovdqu     %%ymm0,(%4)                   \n"  // First 8
      "vmovdqu     %%ymm1,32(%4)                 \n"  // Next 8

      "lea         16(%0),%0                     \n"
      "lea         64(%4),%4                     \n"
      "sub         $0x10,%5                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(src_a),     // %3
        "+r"(dst_argb),  // %4
        "+r"(width)      // %5
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif

#ifdef HAS_MERGEXRGBROW_AVX2
void MergeXRGBRow_AVX2(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "1:          \n"

      "vmovdqu     (%2),%%xmm0                   \n"  // B
      "vpcmpeqb    %%ymm1,%%ymm1,%%ymm1          \n"  // A(255)
      "vinserti128 $0,(%1),%%ymm1,%%ymm1         \n"  // R
      "vinserti128 $1,(%0),%%ymm0,%%ymm0         \n"  // G
      "vpunpckhbw  %%ymm1,%%ymm0,%%ymm2          \n"
      "vpunpcklbw  %%ymm1,%%ymm0,%%ymm0          \n"
      "vperm2i128  $0x31,%%ymm2,%%ymm0,%%ymm1    \n"
      "vperm2i128  $0x20,%%ymm2,%%ymm0,%%ymm0    \n"
      "vpunpckhwd  %%ymm1,%%ymm0,%%ymm2          \n"
      "vpunpcklwd  %%ymm1,%%ymm0,%%ymm0          \n"
      "vperm2i128  $0x31,%%ymm2,%%ymm0,%%ymm1    \n"
      "vperm2i128  $0x20,%%ymm2,%%ymm0,%%ymm0    \n"
      "vmovdqu     %%ymm0,(%3)                   \n"  // First 8
      "vmovdqu     %%ymm1,32(%3)                 \n"  // Next 8

      "lea         16(%0),%0                     \n"
      "lea         16(%1),%1                     \n"
      "lea         16(%2),%2                     \n"
      "lea         64(%3),%3                     \n"
      "sub         $0x10,%4                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_argb),  // %3
        "+rm"(width)     // %4
        ::"memory",
        "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_MERGEARGBROW_AVX2

#ifdef HAS_SPLITARGBROW_SSE2
void SplitARGBRow_SSE2(const uint8_t* src_argb,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       uint8_t* dst_a,
                       int width) {
  asm volatile(
      "sub         %1,%2                         \n"
      "sub         %1,%3                         \n"
      "sub         %1,%4                         \n"

      LABELALIGN
      "1:          \n"

      "movdqu      (%0),%%xmm0                   \n"  // 00-0F
      "movdqu      16(%0),%%xmm1                 \n"  // 10-1F
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpcklqdq  %%xmm1,%%xmm0                 \n"  // 00-07 10-17
      "punpckhqdq  %%xmm1,%%xmm2                 \n"  // 08-0F 18-1F
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm2,%%xmm0                 \n"  // 08192A3B4C5D6E7F (lo)
      "punpckhbw   %%xmm2,%%xmm1                 \n"  // 08192A3B4C5D6E7F (hi)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpcklqdq  %%xmm1,%%xmm0                 \n"  // 08192A3B08192A3B
      "punpckhqdq  %%xmm1,%%xmm2                 \n"  // 4C5D6E7F4C5D6E7F
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm2,%%xmm0                 \n"  // 048C159D26AE37BF (lo)
      "punpckhbw   %%xmm2,%%xmm1                 \n"  // 048C159D26AE37BF (hi)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckldq   %%xmm1,%%xmm0                 \n"  // 048C048C159D159D (BG)
      "punpckhdq   %%xmm1,%%xmm2                 \n"  // 26AE26AE37BF37BF (RA)
      "movlps      %%xmm0,(%1,%3)                \n"  // B
      "movhps      %%xmm0,(%1,%2)                \n"  // G
      "movlps      %%xmm2,(%1)                   \n"  // R
      "movhps      %%xmm2,(%1,%4)                \n"  // A

      "lea         32(%0),%0                     \n"
      "lea         8(%1),%1                      \n"
      "sub         $0x8,%5                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_r),     // %1
        "+r"(dst_g),     // %2
        "+r"(dst_b),     // %3
        "+r"(dst_a),     // %4
        "+rm"(width)     // %5
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif

#ifdef HAS_SPLITXRGBROW_SSE2
void SplitXRGBRow_SSE2(const uint8_t* src_argb,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       int width) {
  asm volatile(
      "1:          \n"

      "movdqu      (%0),%%xmm0                   \n"  // 00-0F
      "movdqu      16(%0),%%xmm1                 \n"  // 10-1F
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpcklqdq  %%xmm1,%%xmm0                 \n"  // 00-07 10-17
      "punpckhqdq  %%xmm1,%%xmm2                 \n"  // 08-0F 18-1F
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm2,%%xmm0                 \n"  // 08192A3B4C5D6E7F (lo)
      "punpckhbw   %%xmm2,%%xmm1                 \n"  // 08192A3B4C5D6E7F (hi)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpcklqdq  %%xmm1,%%xmm0                 \n"  // 08192A3B08192A3B
      "punpckhqdq  %%xmm1,%%xmm2                 \n"  // 4C5D6E7F4C5D6E7F
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm2,%%xmm0                 \n"  // 048C159D26AE37BF (lo)
      "punpckhbw   %%xmm2,%%xmm1                 \n"  // 048C159D26AE37BF (hi)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckldq   %%xmm1,%%xmm0                 \n"  // 048C048C159D159D (BG)
      "punpckhdq   %%xmm1,%%xmm2                 \n"  // 26AE26AE37BF37BF (RA)
      "movlps      %%xmm0,(%3)                   \n"  // B
      "movhps      %%xmm0,(%2)                   \n"  // G
      "movlps      %%xmm2,(%1)                   \n"  // R

      "lea         32(%0),%0                     \n"
      "lea         8(%1),%1                      \n"
      "lea         8(%2),%2                      \n"
      "lea         8(%3),%3                      \n"
      "sub         $0x8,%4                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_r),     // %1
        "+r"(dst_g),     // %2
        "+r"(dst_b),     // %3
        "+rm"(width)     // %4
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif

static const uvec8 kShuffleMaskARGBSplit = {0, 4, 8,  12, 1, 5, 9,  13,
                                            2, 6, 10, 14, 3, 7, 11, 15};
#ifdef HAS_SPLITARGBROW_SSSE3
void SplitARGBRow_SSSE3(const uint8_t* src_argb,
                        uint8_t* dst_r,
                        uint8_t* dst_g,
                        uint8_t* dst_b,
                        uint8_t* dst_a,
                        int width) {
  asm volatile(
      "movdqa      %6,%%xmm3                     \n"
      "sub         %1,%2                         \n"
      "sub         %1,%3                         \n"
      "sub         %1,%4                         \n"

      LABELALIGN
      "1:          \n"

      "movdqu      (%0),%%xmm0                   \n"  // 00-0F
      "movdqu      16(%0),%%xmm1                 \n"  // 10-1F
      "pshufb      %%xmm3,%%xmm0                 \n"  // 048C159D26AE37BF (lo)
      "pshufb      %%xmm3,%%xmm1                 \n"  // 048C159D26AE37BF (hi)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckldq   %%xmm1,%%xmm0                 \n"  // 048C048C159D159D (BG)
      "punpckhdq   %%xmm1,%%xmm2                 \n"  // 26AE26AE37BF37BF (RA)
      "movlps      %%xmm0,(%1,%3)                \n"  // B
      "movhps      %%xmm0,(%1,%2)                \n"  // G
      "movlps      %%xmm2,(%1)                   \n"  // R
      "movhps      %%xmm2,(%1,%4)                \n"  // A

      "lea         32(%0),%0                     \n"
      "lea         8(%1),%1                      \n"
      "subl        $0x8,%5                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_r),     // %1
        "+r"(dst_g),     // %2
        "+r"(dst_b),     // %3
        "+r"(dst_a),     // %4
#if defined(__i386__)
        "+m"(width)  // %5
#else
        "+rm"(width)  // %5
#endif
      : "m"(kShuffleMaskARGBSplit)  // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3");
}
#endif

#ifdef HAS_SPLITXRGBROW_SSSE3
void SplitXRGBRow_SSSE3(const uint8_t* src_argb,
                        uint8_t* dst_r,
                        uint8_t* dst_g,
                        uint8_t* dst_b,
                        int width) {
  asm volatile(
      "movdqa      %5,%%xmm3                     \n"

      LABELALIGN
      "1:          \n"

      "movdqu      (%0),%%xmm0                   \n"  // 00-0F
      "movdqu      16(%0),%%xmm1                 \n"  // 10-1F
      "pshufb      %%xmm3,%%xmm0                 \n"  // 048C159D26AE37BF (lo)
      "pshufb      %%xmm3,%%xmm1                 \n"  // 048C159D26AE37BF (hi)
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpckldq   %%xmm1,%%xmm0                 \n"  // 048C048C159D159D (BG)
      "punpckhdq   %%xmm1,%%xmm2                 \n"  // 26AE26AE37BF37BF (RA)
      "movlps      %%xmm0,(%3)                   \n"  // B
      "movhps      %%xmm0,(%2)                   \n"  // G
      "movlps      %%xmm2,(%1)                   \n"  // R

      "lea         32(%0),%0                     \n"
      "lea         8(%1),%1                      \n"
      "lea         8(%2),%2                      \n"
      "lea         8(%3),%3                      \n"
      "sub         $0x8,%4                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),             // %0
        "+r"(dst_r),                // %1
        "+r"(dst_g),                // %2
        "+r"(dst_b),                // %3
        "+r"(width)                 // %4
      : "m"(kShuffleMaskARGBSplit)  // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3");
}
#endif

#ifdef HAS_SPLITARGBROW_AVX2
static const ulvec32 kShuffleMaskARGBPermute = {0, 4, 1, 5, 2, 6, 3, 7};
void SplitARGBRow_AVX2(const uint8_t* src_argb,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       uint8_t* dst_a,
                       int width) {
  asm volatile(
      "sub         %1,%2                         \n"
      "sub         %1,%3                         \n"
      "sub         %1,%4                         \n"
      "vmovdqa     %7,%%ymm3                     \n"
      "vbroadcastf128 %6,%%ymm4                  \n"

      LABELALIGN
      "1:          \n"

      "vmovdqu     (%0),%%xmm0                   \n"  // 00-0F
      "vmovdqu     16(%0),%%xmm1                 \n"  // 10-1F
      "vinserti128 $1,32(%0),%%ymm0,%%ymm0       \n"  // 00-0F 20-2F
      "vinserti128 $1,48(%0),%%ymm1,%%ymm1       \n"  // 10-1F 30-3F
      "vpshufb     %%ymm4,%%ymm0,%%ymm0          \n"
      "vpshufb     %%ymm4,%%ymm1,%%ymm1          \n"
      "vpermd      %%ymm0,%%ymm3,%%ymm0          \n"
      "vpermd      %%ymm1,%%ymm3,%%ymm1          \n"
      "vpunpckhdq  %%ymm1,%%ymm0,%%ymm2          \n"  // GA
      "vpunpckldq  %%ymm1,%%ymm0,%%ymm0          \n"  // BR
      "vmovdqu     %%xmm0,(%1,%3)                \n"  // B
      "vextracti128 $1,%%ymm0,(%1)               \n"  // R
      "vmovdqu     %%xmm2,(%1,%2)                \n"  // G
      "vextracti128 $1,%%ymm2,(%1,%4)            \n"  // A
      "lea         64(%0),%0                     \n"
      "lea         16(%1),%1                     \n"
      "subl        $0x10,%5                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_r),     // %1
        "+r"(dst_g),     // %2
        "+r"(dst_b),     // %3
        "+r"(dst_a),     // %4
#if defined(__i386__)
        "+m"(width)  // %5
#else
        "+rm"(width)  // %5
#endif
      : "m"(kShuffleMaskARGBSplit),   // %6
        "m"(kShuffleMaskARGBPermute)  // %7
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif

#ifdef HAS_SPLITXRGBROW_AVX2
void SplitXRGBRow_AVX2(const uint8_t* src_argb,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       int width) {
  asm volatile(
      "vmovdqa     %6,%%ymm3                     \n"
      "vbroadcastf128 %5,%%ymm4                  \n"

      LABELALIGN
      "1:          \n"

      "vmovdqu     (%0),%%xmm0                   \n"  // 00-0F
      "vmovdqu     16(%0),%%xmm1                 \n"  // 10-1F
      "vinserti128 $1,32(%0),%%ymm0,%%ymm0       \n"  // 00-0F 20-2F
      "vinserti128 $1,48(%0),%%ymm1,%%ymm1       \n"  // 10-1F 30-3F
      "vpshufb     %%ymm4,%%ymm0,%%ymm0          \n"
      "vpshufb     %%ymm4,%%ymm1,%%ymm1          \n"
      "vpermd      %%ymm0,%%ymm3,%%ymm0          \n"
      "vpermd      %%ymm1,%%ymm3,%%ymm1          \n"
      "vpunpckhdq  %%ymm1,%%ymm0,%%ymm2          \n"  // GA
      "vpunpckldq  %%ymm1,%%ymm0,%%ymm0          \n"  // BR
      "vmovdqu     %%xmm0,(%3)                   \n"  // B
      "vextracti128 $1,%%ymm0,(%1)               \n"  // R
      "vmovdqu     %%xmm2,(%2)                   \n"  // G

      "lea         64(%0),%0                     \n"
      "lea         16(%1),%1                     \n"
      "lea         16(%2),%2                     \n"
      "lea         16(%3),%3                     \n"
      "sub         $0x10,%4                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),               // %0
        "+r"(dst_r),                  // %1
        "+r"(dst_g),                  // %2
        "+r"(dst_b),                  // %3
        "+r"(width)                   // %4
      : "m"(kShuffleMaskARGBSplit),   // %5
        "m"(kShuffleMaskARGBPermute)  // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif

#ifdef HAS_MERGEXR30ROW_AVX2
void MergeXR30Row_AVX2(const uint16_t* src_r,
                       const uint16_t* src_g,
                       const uint16_t* src_b,
                       uint8_t* dst_ar30,
                       int depth,
                       int width) {
  int shift = depth - 10;
  asm volatile(
      "sub         %0,%1                         \n"
      "sub         %0,%2                         \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"  // AR30 constants
      "vpsrlw      $14,%%ymm5,%%ymm5             \n"
      "vpsllw      $4,%%ymm5,%%ymm5              \n"  // 2 alpha bits
      "vpcmpeqb    %%ymm6,%%ymm6,%%ymm6          \n"
      "vpsrlw      $6,%%ymm6,%%ymm6              \n"
      "vmovd       %5,%%xmm4                     \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     (%0,%1),%%ymm1                \n"
      "vmovdqu     (%0,%2),%%ymm2                \n"
      "vpsrlw      %%xmm4,%%ymm0,%%ymm0          \n"
      "vpsrlw      %%xmm4,%%ymm1,%%ymm1          \n"
      "vpsrlw      %%xmm4,%%ymm2,%%ymm2          \n"
      "vpminuw     %%ymm0,%%ymm6,%%ymm0          \n"
      "vpminuw     %%ymm1,%%ymm6,%%ymm1          \n"
      "vpminuw     %%ymm2,%%ymm6,%%ymm2          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpermq      $0xd8,%%ymm1,%%ymm1           \n"
      "vpermq      $0xd8,%%ymm2,%%ymm2           \n"
      "vpsllw      $0x4,%%ymm0,%%ymm0            \n"  // Shift R to target bit
      "vpunpckhwd  %%ymm0,%%ymm2,%%ymm3          \n"  // RB
      "vpunpcklwd  %%ymm0,%%ymm2,%%ymm0          \n"
      "vpunpckhwd  %%ymm5,%%ymm1,%%ymm2          \n"  // AG
      "vpunpcklwd  %%ymm5,%%ymm1,%%ymm1          \n"
      "vpslld      $0xa,%%ymm1,%%ymm1            \n"  // Shift AG to target bit
      "vpslld      $0xa,%%ymm2,%%ymm2            \n"
      "vpor        %%ymm1,%%ymm0,%%ymm0          \n"  // Combine
      "vpor        %%ymm2,%%ymm3,%%ymm3          \n"
      "vmovdqu     %%ymm0,(%3)                   \n"
      "vmovdqu     %%ymm3,0x20(%3)               \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x40(%3),%3                   \n"
      "sub         $0x10,%4                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_ar30),  // %3
        "+r"(width)      // %4
#if defined(__i386__)
      : "m"(shift)  // %5
#else
      : "rm"(shift)   // %5
#endif
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif

#ifdef HAS_MERGEAR64ROW_AVX2
static const lvec32 MergeAR64Permute = {0, 4, 2, 6, 1, 5, 3, 7};
void MergeAR64Row_AVX2(const uint16_t* src_r,
                       const uint16_t* src_g,
                       const uint16_t* src_b,
                       const uint16_t* src_a,
                       uint16_t* dst_ar64,
                       int depth,
                       int width) {
  int shift = 16 - depth;
  int mask = (1 << depth) - 1;
  mask = (mask << 16) + mask;
  asm volatile(
      "sub         %0,%1                         \n"
      "sub         %0,%2                         \n"
      "sub         %0,%3                         \n"
      "vmovdqa     %8,%%ymm5                     \n"
      "vmovd       %6,%%xmm6                     \n"
      "vbroadcastss %7,%%ymm7                    \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"  // R
      "vmovdqu     (%0,%1),%%ymm1                \n"  // G
      "vmovdqu     (%0,%2),%%ymm2                \n"  // B
      "vmovdqu     (%0,%3),%%ymm3                \n"  // A
      "vpminuw     %%ymm0,%%ymm7,%%ymm0          \n"
      "vpminuw     %%ymm1,%%ymm7,%%ymm1          \n"
      "vpminuw     %%ymm2,%%ymm7,%%ymm2          \n"
      "vpminuw     %%ymm3,%%ymm7,%%ymm3          \n"
      "vpsllw      %%xmm6,%%ymm0,%%ymm0          \n"
      "vpsllw      %%xmm6,%%ymm1,%%ymm1          \n"
      "vpsllw      %%xmm6,%%ymm2,%%ymm2          \n"
      "vpsllw      %%xmm6,%%ymm3,%%ymm3          \n"
      "vpermd      %%ymm0,%%ymm5,%%ymm0          \n"
      "vpermd      %%ymm1,%%ymm5,%%ymm1          \n"
      "vpermd      %%ymm2,%%ymm5,%%ymm2          \n"
      "vpermd      %%ymm3,%%ymm5,%%ymm3          \n"
      "vpunpcklwd  %%ymm1,%%ymm2,%%ymm4          \n"  // BG(low)
      "vpunpckhwd  %%ymm1,%%ymm2,%%ymm1          \n"  // BG(hi)
      "vpunpcklwd  %%ymm3,%%ymm0,%%ymm2          \n"  // RA(low)
      "vpunpckhwd  %%ymm3,%%ymm0,%%ymm0          \n"  // RA(hi)
      "vpunpckldq  %%ymm2,%%ymm4,%%ymm3          \n"  // BGRA(1)
      "vpunpckhdq  %%ymm2,%%ymm4,%%ymm4          \n"  // BGRA(3)
      "vpunpckldq  %%ymm0,%%ymm1,%%ymm2          \n"  // BGRA(2)
      "vpunpckhdq  %%ymm0,%%ymm1,%%ymm1          \n"  // BGRA(4)
      "vmovdqu     %%ymm3,(%4)                   \n"
      "vmovdqu     %%ymm2,0x20(%4)               \n"
      "vmovdqu     %%ymm4,0x40(%4)               \n"
      "vmovdqu     %%ymm1,0x60(%4)               \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x80(%4),%4                   \n"
      "subl        $0x10,%5                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(src_a),     // %3
        "+r"(dst_ar64),  // %4
#if defined(__i386__)
        "+m"(width)  // %5
#else
        "+rm"(width)  // %5
#endif
      : "m"(shift),            // %6
        "m"(mask),             // %7
        "m"(MergeAR64Permute)  // %8
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_MERGEXR64ROW_AVX2
void MergeXR64Row_AVX2(const uint16_t* src_r,
                       const uint16_t* src_g,
                       const uint16_t* src_b,
                       uint16_t* dst_ar64,
                       int depth,
                       int width) {
  int shift = 16 - depth;
  int mask = (1 << depth) - 1;
  mask = (mask << 16) + mask;
  asm volatile(
      "sub         %0,%1                         \n"
      "sub         %0,%2                         \n"
      "vmovdqa     %7,%%ymm5                     \n"
      "vmovd       %5,%%xmm6                     \n"
      "vbroadcastss %6,%%ymm7                    \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"  // R
      "vmovdqu     (%0,%1),%%ymm1                \n"  // G
      "vmovdqu     (%0,%2),%%ymm2                \n"  // B
      "vpminuw     %%ymm0,%%ymm7,%%ymm0          \n"
      "vpminuw     %%ymm1,%%ymm7,%%ymm1          \n"
      "vpminuw     %%ymm2,%%ymm7,%%ymm2          \n"
      "vpsllw      %%xmm6,%%ymm0,%%ymm0          \n"
      "vpsllw      %%xmm6,%%ymm1,%%ymm1          \n"
      "vpsllw      %%xmm6,%%ymm2,%%ymm2          \n"
      "vpermd      %%ymm0,%%ymm5,%%ymm0          \n"
      "vpermd      %%ymm1,%%ymm5,%%ymm1          \n"
      "vpermd      %%ymm2,%%ymm5,%%ymm2          \n"
      "vpcmpeqb    %%ymm3,%%ymm3,%%ymm3          \n"  // A (0xffff)
      "vpunpcklwd  %%ymm1,%%ymm2,%%ymm4          \n"  // BG(low)
      "vpunpckhwd  %%ymm1,%%ymm2,%%ymm1          \n"  // BG(hi)
      "vpunpcklwd  %%ymm3,%%ymm0,%%ymm2          \n"  // RA(low)
      "vpunpckhwd  %%ymm3,%%ymm0,%%ymm0          \n"  // RA(hi)
      "vpunpckldq  %%ymm2,%%ymm4,%%ymm3          \n"  // BGRA(1)
      "vpunpckhdq  %%ymm2,%%ymm4,%%ymm4          \n"  // BGRA(3)
      "vpunpckldq  %%ymm0,%%ymm1,%%ymm2          \n"  // BGRA(2)
      "vpunpckhdq  %%ymm0,%%ymm1,%%ymm1          \n"  // BGRA(4)
      "vmovdqu     %%ymm3,(%3)                   \n"
      "vmovdqu     %%ymm2,0x20(%3)               \n"
      "vmovdqu     %%ymm4,0x40(%3)               \n"
      "vmovdqu     %%ymm1,0x60(%3)               \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x80(%3),%3                   \n"
      "subl        $0x10,%4                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_r),           // %0
        "+r"(src_g),           // %1
        "+r"(src_b),           // %2
        "+r"(dst_ar64),        // %3
        "+r"(width)            // %4
      : "m"(shift),            // %5
        "m"(mask),             // %6
        "m"(MergeAR64Permute)  // %7
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif

#ifdef HAS_MERGEARGB16TO8ROW_AVX2
static const uvec8 MergeARGB16To8Shuffle = {0, 8,  1, 9,  2, 10, 3, 11,
                                            4, 12, 5, 13, 6, 14, 7, 15};
void MergeARGB16To8Row_AVX2(const uint16_t* src_r,
                            const uint16_t* src_g,
                            const uint16_t* src_b,
                            const uint16_t* src_a,
                            uint8_t* dst_argb,
                            int depth,
                            int width) {
  int shift = depth - 8;
  asm volatile(
      "sub         %0,%1                         \n"
      "sub         %0,%2                         \n"
      "sub         %0,%3                         \n"
      "vbroadcastf128 %7,%%ymm5                  \n"
      "vmovd       %6,%%xmm6                     \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"  // R
      "vmovdqu     (%0,%1),%%ymm1                \n"  // G
      "vmovdqu     (%0,%2),%%ymm2                \n"  // B
      "vmovdqu     (%0,%3),%%ymm3                \n"  // A
      "vpsrlw      %%xmm6,%%ymm0,%%ymm0          \n"
      "vpsrlw      %%xmm6,%%ymm1,%%ymm1          \n"
      "vpsrlw      %%xmm6,%%ymm2,%%ymm2          \n"
      "vpsrlw      %%xmm6,%%ymm3,%%ymm3          \n"
      "vpackuswb   %%ymm1,%%ymm2,%%ymm1          \n"  // BG (planar)
      "vpackuswb   %%ymm3,%%ymm0,%%ymm0          \n"  // RA (planar)
      "vpshufb     %%ymm5,%%ymm1,%%ymm1          \n"  // BG (interleave)
      "vpshufb     %%ymm5,%%ymm0,%%ymm0          \n"  // RA (interleave)
      "vpermq      $0xd8,%%ymm1,%%ymm1           \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpunpcklwd  %%ymm0,%%ymm1,%%ymm2          \n"  // BGRA (low)
      "vpunpckhwd  %%ymm0,%%ymm1,%%ymm0          \n"  // BGRA (hi)
      "vmovdqu     %%ymm2,(%4)                   \n"
      "vmovdqu     %%ymm0,0x20(%4)               \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x40(%4),%4                   \n"
      "subl        $0x10,%5                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(src_a),     // %3
        "+r"(dst_argb),  // %4
#if defined(__i386__)
        "+m"(width)  // %5
#else
        "+rm"(width)  // %5
#endif
      : "m"(shift),                 // %6
        "m"(MergeARGB16To8Shuffle)  // %7
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif

#ifdef HAS_MERGEXRGB16TO8ROW_AVX2
void MergeXRGB16To8Row_AVX2(const uint16_t* src_r,
                            const uint16_t* src_g,
                            const uint16_t* src_b,
                            uint8_t* dst_argb,
                            int depth,
                            int width) {
  int shift = depth - 8;
  asm volatile(
      "sub         %0,%1                         \n"
      "sub         %0,%2                         \n"
      "vbroadcastf128 %6,%%ymm5                  \n"
      "vmovd       %5,%%xmm6                     \n"
      "vpcmpeqb    %%ymm3,%%ymm3,%%ymm3          \n"
      "vpsrlw      $8,%%ymm3,%%ymm3              \n"  // A (0xff)

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"  // R
      "vmovdqu     (%0,%1),%%ymm1                \n"  // G
      "vmovdqu     (%0,%2),%%ymm2                \n"  // B
      "vpsrlw      %%xmm6,%%ymm0,%%ymm0          \n"
      "vpsrlw      %%xmm6,%%ymm1,%%ymm1          \n"
      "vpsrlw      %%xmm6,%%ymm2,%%ymm2          \n"
      "vpackuswb   %%ymm1,%%ymm2,%%ymm1          \n"  // BG (planar)
      "vpackuswb   %%ymm3,%%ymm0,%%ymm0          \n"  // RA (planar)
      "vpshufb     %%ymm5,%%ymm1,%%ymm1          \n"  // BG (interleave)
      "vpshufb     %%ymm5,%%ymm0,%%ymm0          \n"  // RA (interleave)
      "vpermq      $0xd8,%%ymm1,%%ymm1           \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpunpcklwd  %%ymm0,%%ymm1,%%ymm2          \n"  // BGRA (low)
      "vpunpckhwd  %%ymm0,%%ymm1,%%ymm0          \n"  // BGRA (hi)
      "vmovdqu     %%ymm2,(%3)                   \n"
      "vmovdqu     %%ymm0,0x20(%3)               \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x40(%3),%3                   \n"
      "subl        $0x10,%4                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_r),                // %0
        "+r"(src_g),                // %1
        "+r"(src_b),                // %2
        "+r"(dst_argb),             // %3
        "+r"(width)                 // %4
      : "m"(shift),                 // %5
        "m"(MergeARGB16To8Shuffle)  // %6
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif

#ifdef HAS_COPYROW_SSE2
void CopyRow_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "test        $0xf,%0                       \n"
      "jne         2f                            \n"
      "test        $0xf,%1                       \n"
      "jne         2f                            \n"

      LABELALIGN
      "1:          \n"
      "movdqa      (%0),%%xmm0                   \n"
      "movdqa      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "movdqa      %%xmm0,(%1)                   \n"
      "movdqa      %%xmm1,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "jmp         9f                            \n"

      LABELALIGN
      "2:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          2b                            \n"

      LABELALIGN "9:          \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}
#endif  // HAS_COPYROW_SSE2

#ifdef HAS_COPYROW_AVX
void CopyRow_AVX(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x40,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}
#endif  // HAS_COPYROW_AVX

#ifdef HAS_COPYROW_AVX512BW
void CopyRow_AVX512BW(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "1:          \n"
      "vmovups     (%0),%%zmm0                   \n"
      "vmovups     0x40(%0),%%zmm1               \n"
      "lea         0x80(%0),%0                   \n"
      "vmovups     %%zmm0,(%1)                   \n"
      "vmovups     %%zmm1,0x40(%1)               \n"
      "lea         0x80(%1),%1                   \n"
      "sub         $0x80,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}
#endif  // HAS_COPYROW_AVX512

#ifdef HAS_COPYROW_ERMS
// Multiple of 1.
void CopyRow_ERMS(const uint8_t* src, uint8_t* dst, int width) {
  size_t width_tmp = (size_t)(width);
      asm volatile("rep         movsb                         \n"
               : "+S"(src),       // %0
                 "+D"(dst),       // %1
                 "+c"(width_tmp)  // %2
               :
               : "memory", "cc");
}
#endif  // HAS_COPYROW_ERMS

#ifdef HAS_ARGBCOPYALPHAROW_SSE2
// width in pixels
void ARGBCopyAlphaRow_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "pcmpeqb     %%xmm0,%%xmm0                 \n"
      "pslld       $0x18,%%xmm0                  \n"
      "pcmpeqb     %%xmm1,%%xmm1                 \n"
      "psrld       $0x8,%%xmm1                   \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm2                   \n"
      "movdqu      0x10(%0),%%xmm3               \n"
      "lea         0x20(%0),%0                   \n"
      "movdqu      (%1),%%xmm4                   \n"
      "movdqu      0x10(%1),%%xmm5               \n"
      "pand        %%xmm0,%%xmm2                 \n"
      "pand        %%xmm0,%%xmm3                 \n"
      "pand        %%xmm1,%%xmm4                 \n"
      "pand        %%xmm1,%%xmm5                 \n"
      "por         %%xmm4,%%xmm2                 \n"
      "por         %%xmm5,%%xmm3                 \n"
      "movdqu      %%xmm2,(%1)                   \n"
      "movdqu      %%xmm3,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_ARGBCOPYALPHAROW_SSE2

#ifdef HAS_ARGBCOPYALPHAROW_AVX2
// width in pixels
void ARGBCopyAlphaRow_AVX2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "vpcmpeqb    %%ymm0,%%ymm0,%%ymm0          \n"
      "vpsrld      $0x8,%%ymm0,%%ymm0            \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm1                   \n"
      "vmovdqu     0x20(%0),%%ymm2               \n"
      "lea         0x40(%0),%0                   \n"
      "vpblendvb   %%ymm0,(%1),%%ymm1,%%ymm1     \n"
      "vpblendvb   %%ymm0,0x20(%1),%%ymm2,%%ymm2 \n"
      "vmovdqu     %%ymm1,(%1)                   \n"
      "vmovdqu     %%ymm2,0x20(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_ARGBCOPYALPHAROW_AVX2

#ifdef HAS_ARGBEXTRACTALPHAROW_SSE2
// width in pixels
void ARGBExtractAlphaRow_SSE2(const uint8_t* src_argb,
                              uint8_t* dst_a,
                              int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0), %%xmm0                  \n"
      "movdqu      0x10(%0), %%xmm1              \n"
      "lea         0x20(%0), %0                  \n"
      "psrld       $0x18, %%xmm0                 \n"
      "psrld       $0x18, %%xmm1                 \n"
      "packssdw    %%xmm1, %%xmm0                \n"
      "packuswb    %%xmm0, %%xmm0                \n"
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1), %1                   \n"
      "sub         $0x8, %2                      \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_a),     // %1
        "+rm"(width)     // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}
#endif  // HAS_ARGBEXTRACTALPHAROW_SSE2

#ifdef HAS_ARGBEXTRACTALPHAROW_AVX2
static const uvec8 kShuffleAlphaShort_AVX2 = {
    3u,  128u, 128u, 128u, 7u,  128u, 128u, 128u,
    11u, 128u, 128u, 128u, 15u, 128u, 128u, 128u};

void ARGBExtractAlphaRow_AVX2(const uint8_t* src_argb,
                              uint8_t* dst_a,
                              int width) {
  asm volatile(
      "vmovdqa     %3,%%ymm4                     \n"
      "vbroadcastf128 %4,%%ymm5                  \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0), %%ymm0                  \n"
      "vmovdqu     0x20(%0), %%ymm1              \n"
      "vpshufb     %%ymm5,%%ymm0,%%ymm0          \n"  // vpsrld $0x18, %%ymm0
      "vpshufb     %%ymm5,%%ymm1,%%ymm1          \n"
      "vmovdqu     0x40(%0), %%ymm2              \n"
      "vmovdqu     0x60(%0), %%ymm3              \n"
      "lea         0x80(%0), %0                  \n"
      "vpackssdw   %%ymm1, %%ymm0, %%ymm0        \n"  // mutates
      "vpshufb     %%ymm5,%%ymm2,%%ymm2          \n"
      "vpshufb     %%ymm5,%%ymm3,%%ymm3          \n"
      "vpackssdw   %%ymm3, %%ymm2, %%ymm2        \n"  // mutates
      "vpackuswb   %%ymm2,%%ymm0,%%ymm0          \n"  // mutates.
      "vpermd      %%ymm0,%%ymm4,%%ymm0          \n"  // unmutate.
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20, %2                     \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),               // %0
        "+r"(dst_a),                  // %1
        "+rm"(width)                  // %2
      : "m"(kPermdARGBToY_AVX),       // %3
        "m"(kShuffleAlphaShort_AVX2)  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_ARGBEXTRACTALPHAROW_AVX2

#ifdef HAS_ARGBCOPYYTOALPHAROW_SSE2
// width in pixels
void ARGBCopyYToAlphaRow_SSE2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "pcmpeqb     %%xmm0,%%xmm0                 \n"
      "pslld       $0x18,%%xmm0                  \n"
      "pcmpeqb     %%xmm1,%%xmm1                 \n"
      "psrld       $0x8,%%xmm1                   \n"

      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm2                   \n"
      "lea         0x8(%0),%0                    \n"
      "punpcklbw   %%xmm2,%%xmm2                 \n"
      "punpckhwd   %%xmm2,%%xmm3                 \n"
      "punpcklwd   %%xmm2,%%xmm2                 \n"
      "movdqu      (%1),%%xmm4                   \n"
      "movdqu      0x10(%1),%%xmm5               \n"
      "pand        %%xmm0,%%xmm2                 \n"
      "pand        %%xmm0,%%xmm3                 \n"
      "pand        %%xmm1,%%xmm4                 \n"
      "pand        %%xmm1,%%xmm5                 \n"
      "por         %%xmm4,%%xmm2                 \n"
      "por         %%xmm5,%%xmm3                 \n"
      "movdqu      %%xmm2,(%1)                   \n"
      "movdqu      %%xmm3,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_ARGBCOPYYTOALPHAROW_SSE2

#ifdef HAS_ARGBCOPYYTOALPHAROW_AVX2
// width in pixels
void ARGBCopyYToAlphaRow_AVX2(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "vpcmpeqb    %%ymm0,%%ymm0,%%ymm0          \n"
      "vpsrld      $0x8,%%ymm0,%%ymm0            \n"

      LABELALIGN
      "1:          \n"
      "vpmovzxbd   (%0),%%ymm1                   \n"
      "vpmovzxbd   0x8(%0),%%ymm2                \n"
      "lea         0x10(%0),%0                   \n"
      "vpslld      $0x18,%%ymm1,%%ymm1           \n"
      "vpslld      $0x18,%%ymm2,%%ymm2           \n"
      "vpblendvb   %%ymm0,(%1),%%ymm1,%%ymm1     \n"
      "vpblendvb   %%ymm0,0x20(%1),%%ymm2,%%ymm2 \n"
      "vmovdqu     %%ymm1,(%1)                   \n"
      "vmovdqu     %%ymm2,0x20(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_ARGBCOPYYTOALPHAROW_AVX2

#ifdef HAS_SETROW_X86
void SetRow_X86(uint8_t* dst, uint8_t v8, int width) {
  size_t width_tmp = (size_t)(width >> 2);
  const uint32_t v32 = v8 * 0x01010101u;  // Duplicate byte to all bytes.
      asm volatile("rep         stosl                         \n"
               : "+D"(dst),       // %0
                 "+c"(width_tmp)  // %1
               : "a"(v32)         // %2
               : "memory", "cc");
}

void SetRow_ERMS(uint8_t* dst, uint8_t v8, int width) {
  size_t width_tmp = (size_t)(width);
      asm volatile("rep         stosb                         \n"
               : "+D"(dst),       // %0
                 "+c"(width_tmp)  // %1
               : "a"(v8)          // %2
               : "memory", "cc");
}

void ARGBSetRow_X86(uint8_t* dst_argb, uint32_t v32, int width) {
  size_t width_tmp = (size_t)(width);
      asm volatile("rep         stosl                         \n"
               : "+D"(dst_argb),  // %0
                 "+c"(width_tmp)  // %1
               : "a"(v32)         // %2
               : "memory", "cc");
}
#endif  // HAS_SETROW_X86

#ifdef HAS_YUY2TOYROW_SSE2
void YUY2ToYRow_SSE2(const uint8_t* src_yuy2, uint8_t* dst_y, int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $0x8,%%xmm5                   \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "pand        %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_yuy2),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}

void YUY2ToNVUVRow_SSE2(const uint8_t* src_yuy2,
                        int stride_yuy2,
                        uint8_t* dst_uv,
                        int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x00(%0,%3,1),%%xmm2          \n"
      "movdqu      0x10(%0,%3,1),%%xmm3          \n"
      "lea         0x20(%0),%0                   \n"
      "pavgb       %%xmm2,%%xmm0                 \n"
      "pavgb       %%xmm3,%%xmm1                 \n"
      "psrlw       $0x8,%%xmm0                   \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_yuy2),               // %0
        "+r"(dst_uv),                 // %1
        "+r"(width)                   // %2
      : "r"((intptr_t)(stride_yuy2))  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3");
}

void YUY2ToUVRow_SSE2(const uint8_t* src_yuy2,
                      int stride_yuy2,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $0x8,%%xmm5                   \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x00(%0,%4,1),%%xmm2          \n"
      "movdqu      0x10(%0,%4,1),%%xmm3          \n"
      "lea         0x20(%0),%0                   \n"
      "pavgb       %%xmm2,%%xmm0                 \n"
      "pavgb       %%xmm3,%%xmm1                 \n"
      "psrlw       $0x8,%%xmm0                   \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm1                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "movq        %%xmm1,0x00(%1,%2,1)          \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_yuy2),               // %0
        "+r"(dst_u),                  // %1
        "+r"(dst_v),                  // %2
        "+r"(width)                   // %3
      : "r"((intptr_t)(stride_yuy2))  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}

void YUY2ToUV422Row_SSE2(const uint8_t* src_yuy2,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $0x8,%%xmm5                   \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "psrlw       $0x8,%%xmm0                   \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm1                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "movq        %%xmm1,0x00(%1,%2,1)          \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_yuy2),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
        "+r"(width)      // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}

void UYVYToYRow_SSE2(const uint8_t* src_uyvy, uint8_t* dst_y, int width) {
  asm volatile(
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "psrlw       $0x8,%%xmm0                   \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      : "+r"(src_uyvy),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}

void UYVYToUVRow_SSE2(const uint8_t* src_uyvy,
                      int stride_uyvy,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $0x8,%%xmm5                   \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x00(%0,%4,1),%%xmm2          \n"
      "movdqu      0x10(%0,%4,1),%%xmm3          \n"
      "lea         0x20(%0),%0                   \n"
      "pavgb       %%xmm2,%%xmm0                 \n"
      "pavgb       %%xmm3,%%xmm1                 \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "pand        %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm1                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "movq        %%xmm1,0x00(%1,%2,1)          \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_uyvy),               // %0
        "+r"(dst_u),                  // %1
        "+r"(dst_v),                  // %2
        "+r"(width)                   // %3
      : "r"((intptr_t)(stride_uyvy))  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}

void UYVYToUV422Row_SSE2(const uint8_t* src_uyvy,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psrlw       $0x8,%%xmm5                   \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "pand        %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "pand        %%xmm5,%%xmm0                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm1                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "movq        %%xmm1,0x00(%1,%2,1)          \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_uyvy),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
        "+r"(width)      // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}
#endif  // HAS_YUY2TOYROW_SSE2

#ifdef HAS_YUY2TOYROW_AVX2
void YUY2ToYRow_AVX2(const uint8_t* src_yuy2, uint8_t* dst_y, int width) {
  asm volatile(
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"
      "vpsrlw      $0x8,%%ymm5,%%ymm5            \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpand       %%ymm5,%%ymm0,%%ymm0          \n"
      "vpand       %%ymm5,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_yuy2),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}

void YUY2ToNVUVRow_AVX2(const uint8_t* src_yuy2,
                        int stride_yuy2,
                        uint8_t* dst_uv,
                        int width) {
  asm volatile(
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vpavgb      0x00(%0,%3,1),%%ymm0,%%ymm0   \n"
      "vpavgb      0x20(%0,%3,1),%%ymm1,%%ymm1   \n"
      "lea         0x40(%0),%0                   \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x8,%%ymm1,%%ymm1            \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_yuy2),               // %0
        "+r"(dst_uv),                 // %1
        "+r"(width)                   // %2
      : "r"((intptr_t)(stride_yuy2))  // %3
      : "memory", "cc", "xmm0", "xmm1");
}

void YUY2ToUVRow_AVX2(const uint8_t* src_yuy2,
                      int stride_yuy2,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile(
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"
      "vpsrlw      $0x8,%%ymm5,%%ymm5            \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vpavgb      0x00(%0,%4,1),%%ymm0,%%ymm0   \n"
      "vpavgb      0x20(%0,%4,1),%%ymm1,%%ymm1   \n"
      "lea         0x40(%0),%0                   \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x8,%%ymm1,%%ymm1            \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpand       %%ymm5,%%ymm0,%%ymm1          \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpackuswb   %%ymm1,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm0,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm1,%%ymm1           \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vextractf128 $0x0,%%ymm1,(%1)             \n"
      "vextractf128 $0x0,%%ymm0,0x00(%1,%2,1)    \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x20,%3                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_yuy2),               // %0
        "+r"(dst_u),                  // %1
        "+r"(dst_v),                  // %2
        "+r"(width)                   // %3
      : "r"((intptr_t)(stride_yuy2))  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}

void YUY2ToUV422Row_AVX2(const uint8_t* src_yuy2,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  asm volatile(
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"
      "vpsrlw      $0x8,%%ymm5,%%ymm5            \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x8,%%ymm1,%%ymm1            \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpand       %%ymm5,%%ymm0,%%ymm1          \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpackuswb   %%ymm1,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm0,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm1,%%ymm1           \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vextractf128 $0x0,%%ymm1,(%1)             \n"
      "vextractf128 $0x0,%%ymm0,0x00(%1,%2,1)    \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x20,%3                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_yuy2),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
        "+r"(width)      // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}

void UYVYToYRow_AVX2(const uint8_t* src_uyvy, uint8_t* dst_y, int width) {
  asm volatile(
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x8,%%ymm1,%%ymm1            \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_uyvy),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}
void UYVYToUVRow_AVX2(const uint8_t* src_uyvy,
                      int stride_uyvy,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile(
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"
      "vpsrlw      $0x8,%%ymm5,%%ymm5            \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "vpavgb      0x00(%0,%4,1),%%ymm0,%%ymm0   \n"
      "vpavgb      0x20(%0,%4,1),%%ymm1,%%ymm1   \n"
      "lea         0x40(%0),%0                   \n"
      "vpand       %%ymm5,%%ymm0,%%ymm0          \n"
      "vpand       %%ymm5,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpand       %%ymm5,%%ymm0,%%ymm1          \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpackuswb   %%ymm1,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm0,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm1,%%ymm1           \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vextractf128 $0x0,%%ymm1,(%1)             \n"
      "vextractf128 $0x0,%%ymm0,0x00(%1,%2,1)    \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x20,%3                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_uyvy),               // %0
        "+r"(dst_u),                  // %1
        "+r"(dst_v),                  // %2
        "+r"(width)                   // %3
      : "r"((intptr_t)(stride_uyvy))  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}

void UYVYToUV422Row_AVX2(const uint8_t* src_uyvy,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  asm volatile(
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"
      "vpsrlw      $0x8,%%ymm5,%%ymm5            \n"
      "sub         %1,%2                         \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpand       %%ymm5,%%ymm0,%%ymm0          \n"
      "vpand       %%ymm5,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpand       %%ymm5,%%ymm0,%%ymm1          \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpackuswb   %%ymm1,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm0,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm1,%%ymm1           \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vextractf128 $0x0,%%ymm1,(%1)             \n"
      "vextractf128 $0x0,%%ymm0,0x00(%1,%2,1)    \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x20,%3                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_uyvy),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
        "+r"(width)      // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm5");
}
#endif  // HAS_YUY2TOYROW_AVX2

#ifdef HAS_ARGBBLENDROW_SSSE3
// Shuffle table for isolating alpha.
static const uvec8 kShuffleAlpha = {3u,  0x80, 3u,  0x80, 7u,  0x80, 7u,  0x80,
                                    11u, 0x80, 11u, 0x80, 15u, 0x80, 15u, 0x80};

// Blend 8 pixels at a time
void ARGBBlendRow_SSSE3(const uint8_t* src_argb,
                        const uint8_t* src_argb1,
                        uint8_t* dst_argb,
                        int width) {
  asm volatile(
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "psrlw       $0xf,%%xmm7                   \n"
      "pcmpeqb     %%xmm6,%%xmm6                 \n"
      "psrlw       $0x8,%%xmm6                   \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psllw       $0x8,%%xmm5                   \n"
      "pcmpeqb     %%xmm4,%%xmm4                 \n"
      "pslld       $0x18,%%xmm4                  \n"
      "sub         $0x4,%3                       \n"
      "jl          49f                           \n"

      // 4 pixel loop.
      LABELALIGN
      "40:         \n"
      "movdqu      (%0),%%xmm3                   \n"
      "lea         0x10(%0),%0                   \n"
      "movdqa      %%xmm3,%%xmm0                 \n"
      "pxor        %%xmm4,%%xmm3                 \n"
      "movdqu      (%1),%%xmm2                   \n"
      "pshufb      %4,%%xmm3                     \n"
      "pand        %%xmm6,%%xmm2                 \n"
      "paddw       %%xmm7,%%xmm3                 \n"
      "pmullw      %%xmm3,%%xmm2                 \n"
      "movdqu      (%1),%%xmm1                   \n"
      "lea         0x10(%1),%1                   \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "por         %%xmm4,%%xmm0                 \n"
      "pmullw      %%xmm3,%%xmm1                 \n"
      "psrlw       $0x8,%%xmm2                   \n"
      "paddusb     %%xmm2,%%xmm0                 \n"
      "pand        %%xmm5,%%xmm1                 \n"
      "paddusb     %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%3                       \n"
      "jge         40b                           \n"

      "49:         \n"
      "add         $0x3,%3                       \n"
      "jl          99f                           \n"

      // 1 pixel loop.
      "91:         \n"
      "movd        (%0),%%xmm3                   \n"
      "lea         0x4(%0),%0                    \n"
      "movdqa      %%xmm3,%%xmm0                 \n"
      "pxor        %%xmm4,%%xmm3                 \n"
      "movd        (%1),%%xmm2                   \n"
      "pshufb      %4,%%xmm3                     \n"
      "pand        %%xmm6,%%xmm2                 \n"
      "paddw       %%xmm7,%%xmm3                 \n"
      "pmullw      %%xmm3,%%xmm2                 \n"
      "movd        (%1),%%xmm1                   \n"
      "lea         0x4(%1),%1                    \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "por         %%xmm4,%%xmm0                 \n"
      "pmullw      %%xmm3,%%xmm1                 \n"
      "psrlw       $0x8,%%xmm2                   \n"
      "paddusb     %%xmm2,%%xmm0                 \n"
      "pand        %%xmm5,%%xmm1                 \n"
      "paddusb     %%xmm1,%%xmm0                 \n"
      "movd        %%xmm0,(%2)                   \n"
      "lea         0x4(%2),%2                    \n"
      "sub         $0x1,%3                       \n"
      "jge         91b                           \n"
      "99:         \n"
      : "+r"(src_argb),     // %0
        "+r"(src_argb1),    // %1
        "+r"(dst_argb),     // %2
        "+r"(width)         // %3
      : "m"(kShuffleAlpha)  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBBLENDROW_SSSE3

#ifdef HAS_BLENDPLANEROW_SSSE3
// Blend 8 pixels at a time.
// unsigned version of math
// =((A2*C2)+(B2*(255-C2))+255)/256
// signed version of math
// =(((A2-128)*C2)+((B2-128)*(255-C2))+32768+127)/256
void BlendPlaneRow_SSSE3(const uint8_t* src0,
                         const uint8_t* src1,
                         const uint8_t* alpha,
                         uint8_t* dst,
                         int width) {
  asm volatile(
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "psllw       $0x8,%%xmm5                   \n"
      "mov         $0x80808080,%%eax             \n"
      "movd        %%eax,%%xmm6                  \n"
      "pshufd      $0x0,%%xmm6,%%xmm6            \n"
      "mov         $0x807f807f,%%eax             \n"
      "movd        %%eax,%%xmm7                  \n"
      "pshufd      $0x0,%%xmm7,%%xmm7            \n"
      "sub         %2,%0                         \n"
      "sub         %2,%1                         \n"
      "sub         %2,%3                         \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "movq        (%2),%%xmm0                   \n"
      "punpcklbw   %%xmm0,%%xmm0                 \n"
      "pxor        %%xmm5,%%xmm0                 \n"
      "movq        (%0,%2,1),%%xmm1              \n"
      "movq        (%1,%2,1),%%xmm2              \n"
      "punpcklbw   %%xmm2,%%xmm1                 \n"
      "psubb       %%xmm6,%%xmm1                 \n"
      "pmaddubsw   %%xmm1,%%xmm0                 \n"
      "paddw       %%xmm7,%%xmm0                 \n"
      "psrlw       $0x8,%%xmm0                   \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movq        %%xmm0,(%3,%2,1)              \n"
      "lea         0x8(%2),%2                    \n"
      "sub         $0x8,%4                       \n"
      "jg          1b                            \n"
      : "+r"(src0),   // %0
        "+r"(src1),   // %1
        "+r"(alpha),  // %2
        "+r"(dst),    // %3
        "+rm"(width)  // %4
        ::"memory",
        "cc", "eax", "xmm0", "xmm1", "xmm2", "xmm5", "xmm6", "xmm7");
}
#endif  // HAS_BLENDPLANEROW_SSSE3

#ifdef HAS_BLENDPLANEROW_AVX2
// Blend 32 pixels at a time.
// unsigned version of math
// =((A2*C2)+(B2*(255-C2))+255)/256
// signed version of math
// =(((A2-128)*C2)+((B2-128)*(255-C2))+32768+127)/256
void BlendPlaneRow_AVX2(const uint8_t* src0,
                        const uint8_t* src1,
                        const uint8_t* alpha,
                        uint8_t* dst,
                        int width) {
  asm volatile(
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"
      "vpsllw      $0x8,%%ymm5,%%ymm5            \n"
      "mov         $0x80808080,%%eax             \n"
      "vmovd       %%eax,%%xmm6                  \n"
      "vbroadcastss %%xmm6,%%ymm6                \n"
      "mov         $0x807f807f,%%eax             \n"
      "vmovd       %%eax,%%xmm7                  \n"
      "vbroadcastss %%xmm7,%%ymm7                \n"
      "sub         %2,%0                         \n"
      "sub         %2,%1                         \n"
      "sub         %2,%3                         \n"

      // 32 pixel loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%2),%%ymm0                   \n"
      "vpunpckhbw  %%ymm0,%%ymm0,%%ymm3          \n"
      "vpunpcklbw  %%ymm0,%%ymm0,%%ymm0          \n"
      "vpxor       %%ymm5,%%ymm3,%%ymm3          \n"
      "vpxor       %%ymm5,%%ymm0,%%ymm0          \n"
      "vmovdqu     (%0,%2,1),%%ymm1              \n"
      "vmovdqu     (%1,%2,1),%%ymm2              \n"
      "vpunpckhbw  %%ymm2,%%ymm1,%%ymm4          \n"
      "vpunpcklbw  %%ymm2,%%ymm1,%%ymm1          \n"
      "vpsubb      %%ymm6,%%ymm4,%%ymm4          \n"
      "vpsubb      %%ymm6,%%ymm1,%%ymm1          \n"
      "vpmaddubsw  %%ymm4,%%ymm3,%%ymm3          \n"
      "vpmaddubsw  %%ymm1,%%ymm0,%%ymm0          \n"
      "vpaddw      %%ymm7,%%ymm3,%%ymm3          \n"
      "vpaddw      %%ymm7,%%ymm0,%%ymm0          \n"
      "vpsrlw      $0x8,%%ymm3,%%ymm3            \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpackuswb   %%ymm3,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%3,%2,1)              \n"
      "lea         0x20(%2),%2                   \n"
      "sub         $0x20,%4                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src0),   // %0
        "+r"(src1),   // %1
        "+r"(alpha),  // %2
        "+r"(dst),    // %3
        "+rm"(width)  // %4
        ::"memory",
        "cc", "eax", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_BLENDPLANEROW_AVX2

#ifdef HAS_ARGBATTENUATEROW_SSSE3
// Shuffle table duplicating alpha.
static const vec8 kAttenuateShuffle = {6,    -128, 6,    -128, 6,  -128,
                                       -128, -128, 14,   -128, 14, -128,
                                       14,   -128, -128, -128};

// Attenuate 4 pixels at a time.
void ARGBAttenuateRow_SSSE3(const uint8_t* src_argb,
                            uint8_t* dst_argb,
                            int width) {
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "pslld       $0x18,%%xmm5                  \n"
      "pxor        %%xmm6,%%xmm6                 \n"
      "pcmpeqb     %%xmm7,%%xmm7                 \n"
      "punpcklbw   %%xmm6,%%xmm7                 \n"
      "sub         %0,%1                         \n"

      // 4 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm6                   \n"
      "movdqa      %%xmm6,%%xmm0                 \n"
      "movdqa      %%xmm6,%%xmm1                 \n"
      "punpcklbw   %%xmm5,%%xmm0                 \n"
      "punpckhbw   %%xmm5,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm1,%%xmm3                 \n"
      "pshufb      %%xmm4,%%xmm2                 \n"  // a,a,a,0
      "pshufb      %%xmm4,%%xmm3                 \n"
      "pmullw      %%xmm2,%%xmm0                 \n"  // rgb * alpha
      "pmullw      %%xmm3,%%xmm1                 \n"
      "paddw       %%xmm7,%%xmm0                 \n"  // + 255
      "paddw       %%xmm7,%%xmm1                 \n"
      "psrlw       $0x8,%%xmm0                   \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "pand        %%xmm5,%%xmm6                 \n"
      "por         %%xmm6,%%xmm0                 \n"
      "movdqu      %%xmm0,(%0,%1)                \n"
      "lea         0x10(%0),%0                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),         // %0
        "+r"(dst_argb),         // %1
        "+r"(width)             // %2
      : "m"(kAttenuateShuffle)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBATTENUATEROW_SSSE3

#ifdef HAS_ARGBATTENUATEROW_AVX2

// Shuffle table duplicating alpha.
static const lvec8 kAttenuateShuffle_AVX2 = {
    6,    -128, 6,    -128, 6,    -128, -128, -128, 14,   -128, 14,
    -128, 14,   -128, -128, -128, 22,   -128, 22,   -128, 22,   -128,
    -128, -128, 30,   -128, 30,   -128, 30,   -128, -128, -128};

// Attenuate 8 pixels at a time.
void ARGBAttenuateRow_AVX2(const uint8_t* src_argb,
                           uint8_t* dst_argb,
                           int width) {
  asm volatile(
      "vmovdqa     %3,%%ymm4                     \n"
      "vpcmpeqb    %%ymm5,%%ymm5,%%ymm5          \n"
      "vpslld      $0x18,%%ymm5,%%ymm5           \n"
      "vpxor       %%ymm6,%%ymm6,%%ymm6          \n"
      "vpcmpeqb    %%ymm7,%%ymm7,%%ymm7          \n"
      "vpunpcklbw  %%ymm6,%%ymm7,%%ymm7          \n"
      "sub         %0,%1                         \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm6                   \n"
      "vpunpcklbw  %%ymm5,%%ymm6,%%ymm0          \n"
      "vpunpckhbw  %%ymm5,%%ymm6,%%ymm1          \n"
      "vpshufb     %%ymm4,%%ymm0,%%ymm2          \n"
      "vpshufb     %%ymm4,%%ymm1,%%ymm3          \n"
      "vpmullw     %%ymm2,%%ymm0,%%ymm0          \n"
      "vpmullw     %%ymm3,%%ymm1,%%ymm1          \n"
      "vpaddw      %%ymm7,%%ymm0,%%ymm0          \n"
      "vpaddw      %%ymm7,%%ymm1,%%ymm1          \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x8,%%ymm1,%%ymm1            \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vpand       %%ymm5,%%ymm6,%%ymm1          \n"
      "vpor        %%ymm1,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,0x00(%0,%1,1)          \n"
      "lea         0x20(%0),%0                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),              // %0
        "+r"(dst_argb),              // %1
        "+r"(width)                  // %2
      : "m"(kAttenuateShuffle_AVX2)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBATTENUATEROW_AVX2

#ifdef HAS_ARGBUNATTENUATEROW_SSE2
// Unattenuate 4 pixels at a time.
void ARGBUnattenuateRow_SSE2(const uint8_t* src_argb,
                             uint8_t* dst_argb,
                             int width) {
  uintptr_t alpha;
  asm volatile(
      // 4 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movzb       0x03(%0),%3                   \n"
      "punpcklbw   %%xmm0,%%xmm0                 \n"
      "movd        0x00(%4,%3,4),%%xmm2          \n"
      "movzb       0x07(%0),%3                   \n"
      "movd        0x00(%4,%3,4),%%xmm3          \n"
      "pshuflw     $0x40,%%xmm2,%%xmm2           \n"
      "pshuflw     $0x40,%%xmm3,%%xmm3           \n"
      "movlhps     %%xmm3,%%xmm2                 \n"
      "pmulhuw     %%xmm2,%%xmm0                 \n"
      "movdqu      (%0),%%xmm1                   \n"
      "movzb       0x0b(%0),%3                   \n"
      "punpckhbw   %%xmm1,%%xmm1                 \n"
      "movd        0x00(%4,%3,4),%%xmm2          \n"
      "movzb       0x0f(%0),%3                   \n"
      "movd        0x00(%4,%3,4),%%xmm3          \n"
      "pshuflw     $0x40,%%xmm2,%%xmm2           \n"
      "pshuflw     $0x40,%%xmm3,%%xmm3           \n"
      "movlhps     %%xmm3,%%xmm2                 \n"
      "pmulhuw     %%xmm2,%%xmm1                 \n"
      "lea         0x10(%0),%0                   \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),     // %0
        "+r"(dst_argb),     // %1
        "+r"(width),        // %2
        "=&r"(alpha)        // %3
      : "r"(fixed_invtbl8)  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_ARGBUNATTENUATEROW_SSE2

#ifdef HAS_ARGBUNATTENUATEROW_AVX2
// Shuffle table duplicating alpha.
static const uvec8 kUnattenShuffleAlpha_AVX2 = {
    0u, 1u, 0u, 1u, 0u, 1u, 6u, 7u, 8u, 9u, 8u, 9u, 8u, 9u, 14u, 15u};
// Unattenuate 8 pixels at a time.
void ARGBUnattenuateRow_AVX2(const uint8_t* src_argb,
                             uint8_t* dst_argb,
                             int width) {
  uintptr_t alpha;
  asm volatile(
      "sub         %0,%1                         \n"
      "vbroadcastf128 %5,%%ymm5                  \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      // replace VPGATHER
      "movzb       0x03(%0),%3                   \n"
      "vmovd       0x00(%4,%3,4),%%xmm0          \n"
      "movzb       0x07(%0),%3                   \n"
      "vmovd       0x00(%4,%3,4),%%xmm1          \n"
      "movzb       0x0b(%0),%3                   \n"
      "vpunpckldq  %%xmm1,%%xmm0,%%xmm6          \n"
      "vmovd       0x00(%4,%3,4),%%xmm2          \n"
      "movzb       0x0f(%0),%3                   \n"
      "vmovd       0x00(%4,%3,4),%%xmm3          \n"
      "movzb       0x13(%0),%3                   \n"
      "vpunpckldq  %%xmm3,%%xmm2,%%xmm7          \n"
      "vmovd       0x00(%4,%3,4),%%xmm0          \n"
      "movzb       0x17(%0),%3                   \n"
      "vmovd       0x00(%4,%3,4),%%xmm1          \n"
      "movzb       0x1b(%0),%3                   \n"
      "vpunpckldq  %%xmm1,%%xmm0,%%xmm0          \n"
      "vmovd       0x00(%4,%3,4),%%xmm2          \n"
      "movzb       0x1f(%0),%3                   \n"
      "vmovd       0x00(%4,%3,4),%%xmm3          \n"
      "vpunpckldq  %%xmm3,%%xmm2,%%xmm2          \n"
      "vpunpcklqdq %%xmm7,%%xmm6,%%xmm3          \n"
      "vpunpcklqdq %%xmm2,%%xmm0,%%xmm0          \n"
      "vinserti128 $0x1,%%xmm0,%%ymm3,%%ymm3     \n"
      // end of VPGATHER

      "vmovdqu     (%0),%%ymm6                   \n"
      "vpunpcklbw  %%ymm6,%%ymm6,%%ymm0          \n"
      "vpunpckhbw  %%ymm6,%%ymm6,%%ymm1          \n"
      "vpunpcklwd  %%ymm3,%%ymm3,%%ymm2          \n"
      "vpunpckhwd  %%ymm3,%%ymm3,%%ymm3          \n"
      "vpshufb     %%ymm5,%%ymm2,%%ymm2          \n"
      "vpshufb     %%ymm5,%%ymm3,%%ymm3          \n"
      "vpmulhuw    %%ymm2,%%ymm0,%%ymm0          \n"
      "vpmulhuw    %%ymm3,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,0x00(%0,%1,1)          \n"
      "lea         0x20(%0),%0                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),                 // %0
        "+r"(dst_argb),                 // %1
        "+r"(width),                    // %2
        "=&r"(alpha)                    // %3
      : "r"(fixed_invtbl8),             // %4
        "m"(kUnattenShuffleAlpha_AVX2)  // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBUNATTENUATEROW_AVX2

#ifdef HAS_ARGBGRAYROW_SSSE3
// Convert 8 ARGB pixels (64 bytes) to 8 Gray ARGB pixels
void ARGBGrayRow_SSSE3(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  asm volatile(
      "movdqa      %3,%%xmm4                     \n"
      "movdqa      %4,%%xmm5                     \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "psubb       %%xmm5,%%xmm0                 \n"
      "psubb       %%xmm5,%%xmm1                 \n"
      "movdqu      %%xmm4,%%xmm6                 \n"
      "pmaddubsw   %%xmm0,%%xmm6                 \n"
      "movdqu      %%xmm4,%%xmm0                 \n"
      "pmaddubsw   %%xmm1,%%xmm0                 \n"
      "phaddw      %%xmm0,%%xmm6                 \n"
      "paddw       %%xmm5,%%xmm6                 \n"
      "psrlw       $0x8,%%xmm6                   \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "movdqu      (%0),%%xmm2                   \n"
      "movdqu      0x10(%0),%%xmm3               \n"
      "lea         0x20(%0),%0                   \n"
      "psrld       $0x18,%%xmm2                  \n"
      "psrld       $0x18,%%xmm3                  \n"
      "packuswb    %%xmm3,%%xmm2                 \n"
      "packuswb    %%xmm2,%%xmm2                 \n"
      "movdqa      %%xmm6,%%xmm3                 \n"
      "punpcklbw   %%xmm6,%%xmm6                 \n"
      "punpcklbw   %%xmm2,%%xmm3                 \n"
      "movdqa      %%xmm6,%%xmm1                 \n"
      "punpcklwd   %%xmm3,%%xmm6                 \n"
      "punpckhwd   %%xmm3,%%xmm1                 \n"
      "movdqu      %%xmm6,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      : "m"(kARGBToYJ),  // %3
        "m"(kSub128)     // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif  // HAS_ARGBGRAYROW_SSSE3

#ifdef HAS_ARGBSEPIAROW_SSSE3
//    b = (r * 35 + g * 68 + b * 17) >> 7
//    g = (r * 45 + g * 88 + b * 22) >> 7
//    r = (r * 50 + g * 98 + b * 24) >> 7
// Constant for ARGB color to sepia tone
static const vec8 kARGBToSepiaB = {17, 68, 35, 0, 17, 68, 35, 0,
                                   17, 68, 35, 0, 17, 68, 35, 0};

static const vec8 kARGBToSepiaG = {22, 88, 45, 0, 22, 88, 45, 0,
                                   22, 88, 45, 0, 22, 88, 45, 0};

static const vec8 kARGBToSepiaR = {24, 98, 50, 0, 24, 98, 50, 0,
                                   24, 98, 50, 0, 24, 98, 50, 0};

// Convert 8 ARGB pixels (32 bytes) to 8 Sepia ARGB pixels.
void ARGBSepiaRow_SSSE3(uint8_t* dst_argb, int width) {
  asm volatile(
      "movdqa      %2,%%xmm2                     \n"
      "movdqa      %3,%%xmm3                     \n"
      "movdqa      %4,%%xmm4                     \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm6               \n"
      "pmaddubsw   %%xmm2,%%xmm0                 \n"
      "pmaddubsw   %%xmm2,%%xmm6                 \n"
      "phaddw      %%xmm6,%%xmm0                 \n"
      "psrlw       $0x7,%%xmm0                   \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movdqu      (%0),%%xmm5                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "pmaddubsw   %%xmm3,%%xmm5                 \n"
      "pmaddubsw   %%xmm3,%%xmm1                 \n"
      "phaddw      %%xmm1,%%xmm5                 \n"
      "psrlw       $0x7,%%xmm5                   \n"
      "packuswb    %%xmm5,%%xmm5                 \n"
      "punpcklbw   %%xmm5,%%xmm0                 \n"
      "movdqu      (%0),%%xmm5                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "pmaddubsw   %%xmm4,%%xmm5                 \n"
      "pmaddubsw   %%xmm4,%%xmm1                 \n"
      "phaddw      %%xmm1,%%xmm5                 \n"
      "psrlw       $0x7,%%xmm5                   \n"
      "packuswb    %%xmm5,%%xmm5                 \n"
      "movdqu      (%0),%%xmm6                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "psrld       $0x18,%%xmm6                  \n"
      "psrld       $0x18,%%xmm1                  \n"
      "packuswb    %%xmm1,%%xmm6                 \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "punpcklbw   %%xmm6,%%xmm5                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklwd   %%xmm5,%%xmm0                 \n"
      "punpckhwd   %%xmm5,%%xmm1                 \n"
      "movdqu      %%xmm0,(%0)                   \n"
      "movdqu      %%xmm1,0x10(%0)               \n"
      "lea         0x20(%0),%0                   \n"
      "sub         $0x8,%1                       \n"
      "jg          1b                            \n"
      : "+r"(dst_argb),      // %0
        "+r"(width)          // %1
      : "m"(kARGBToSepiaB),  // %2
        "m"(kARGBToSepiaG),  // %3
        "m"(kARGBToSepiaR)   // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif  // HAS_ARGBSEPIAROW_SSSE3

#ifdef HAS_ARGBCOLORMATRIXROW_SSSE3
// Tranform 8 ARGB pixels (32 bytes) with color matrix.
// Same as Sepia except matrix is provided.
void ARGBColorMatrixRow_SSSE3(const uint8_t* src_argb,
                              uint8_t* dst_argb,
                              const int8_t* matrix_argb,
                              int width) {
  asm volatile(
      "movdqu      (%3),%%xmm5                   \n"
      "pshufd      $0x00,%%xmm5,%%xmm2           \n"
      "pshufd      $0x55,%%xmm5,%%xmm3           \n"
      "pshufd      $0xaa,%%xmm5,%%xmm4           \n"
      "pshufd      $0xff,%%xmm5,%%xmm5           \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm7               \n"
      "pmaddubsw   %%xmm2,%%xmm0                 \n"
      "pmaddubsw   %%xmm2,%%xmm7                 \n"
      "movdqu      (%0),%%xmm6                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "pmaddubsw   %%xmm3,%%xmm6                 \n"
      "pmaddubsw   %%xmm3,%%xmm1                 \n"
      "phaddsw     %%xmm7,%%xmm0                 \n"
      "phaddsw     %%xmm1,%%xmm6                 \n"
      "psraw       $0x6,%%xmm0                   \n"
      "psraw       $0x6,%%xmm6                   \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "punpcklbw   %%xmm6,%%xmm0                 \n"
      "movdqu      (%0),%%xmm1                   \n"
      "movdqu      0x10(%0),%%xmm7               \n"
      "pmaddubsw   %%xmm4,%%xmm1                 \n"
      "pmaddubsw   %%xmm4,%%xmm7                 \n"
      "phaddsw     %%xmm7,%%xmm1                 \n"
      "movdqu      (%0),%%xmm6                   \n"
      "movdqu      0x10(%0),%%xmm7               \n"
      "pmaddubsw   %%xmm5,%%xmm6                 \n"
      "pmaddubsw   %%xmm5,%%xmm7                 \n"
      "phaddsw     %%xmm7,%%xmm6                 \n"
      "psraw       $0x6,%%xmm1                   \n"
      "psraw       $0x6,%%xmm6                   \n"
      "packuswb    %%xmm1,%%xmm1                 \n"
      "packuswb    %%xmm6,%%xmm6                 \n"
      "punpcklbw   %%xmm6,%%xmm1                 \n"
      "movdqa      %%xmm0,%%xmm6                 \n"
      "punpcklwd   %%xmm1,%%xmm0                 \n"
      "punpckhwd   %%xmm1,%%xmm6                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm6,0x10(%1)               \n"
      "lea         0x20(%0),%0                   \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(dst_argb),   // %1
        "+r"(width)       // %2
      : "r"(matrix_argb)  // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBCOLORMATRIXROW_SSSE3

#ifdef HAS_ARGBQUANTIZEROW_SSE2
// Quantize 4 ARGB pixels (16 bytes).
void ARGBQuantizeRow_SSE2(uint8_t* dst_argb,
                          int scale,
                          int interval_size,
                          int interval_offset,
                          int width) {
  asm volatile(
      "movd        %2,%%xmm2                     \n"
      "movd        %3,%%xmm3                     \n"
      "movd        %4,%%xmm4                     \n"
      "pshuflw     $0x40,%%xmm2,%%xmm2           \n"
      "pshufd      $0x44,%%xmm2,%%xmm2           \n"
      "pshuflw     $0x40,%%xmm3,%%xmm3           \n"
      "pshufd      $0x44,%%xmm3,%%xmm3           \n"
      "pshuflw     $0x40,%%xmm4,%%xmm4           \n"
      "pshufd      $0x44,%%xmm4,%%xmm4           \n"
      "pxor        %%xmm5,%%xmm5                 \n"
      "pcmpeqb     %%xmm6,%%xmm6                 \n"
      "pslld       $0x18,%%xmm6                  \n"

      // 4 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "punpcklbw   %%xmm5,%%xmm0                 \n"
      "pmulhuw     %%xmm2,%%xmm0                 \n"
      "movdqu      (%0),%%xmm1                   \n"
      "punpckhbw   %%xmm5,%%xmm1                 \n"
      "pmulhuw     %%xmm2,%%xmm1                 \n"
      "pmullw      %%xmm3,%%xmm0                 \n"
      "movdqu      (%0),%%xmm7                   \n"
      "pmullw      %%xmm3,%%xmm1                 \n"
      "pand        %%xmm6,%%xmm7                 \n"
      "paddw       %%xmm4,%%xmm0                 \n"
      "paddw       %%xmm4,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "por         %%xmm7,%%xmm0                 \n"
      "movdqu      %%xmm0,(%0)                   \n"
      "lea         0x10(%0),%0                   \n"
      "sub         $0x4,%1                       \n"
      "jg          1b                            \n"
      : "+r"(dst_argb),       // %0
        "+r"(width)           // %1
      : "r"(scale),           // %2
        "r"(interval_size),   // %3
        "r"(interval_offset)  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBQUANTIZEROW_SSE2

#ifdef HAS_ARGBSHADEROW_SSE2
// Shade 4 pixels at a time by specified value.
void ARGBShadeRow_SSE2(const uint8_t* src_argb,
                       uint8_t* dst_argb,
                       int width,
                       uint32_t value) {
  asm volatile(
      "movd        %3,%%xmm2                     \n"
      "punpcklbw   %%xmm2,%%xmm2                 \n"
      "punpcklqdq  %%xmm2,%%xmm2                 \n"

      // 4 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "lea         0x10(%0),%0                   \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm0,%%xmm0                 \n"
      "punpckhbw   %%xmm1,%%xmm1                 \n"
      "pmulhuw     %%xmm2,%%xmm0                 \n"
      "pmulhuw     %%xmm2,%%xmm1                 \n"
      "psrlw       $0x8,%%xmm0                   \n"
      "psrlw       $0x8,%%xmm1                   \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x4,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      : "r"(value)       // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_ARGBSHADEROW_SSE2

#ifdef HAS_ARGBMULTIPLYROW_SSE2
// Multiply 2 rows of ARGB pixels together, 4 pixels at a time.
void ARGBMultiplyRow_SSE2(const uint8_t* src_argb,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
      asm volatile("pxor        %%xmm5,%%xmm5                 \n"

               // 4 pixel loop.
               LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "lea         0x10(%0),%0                   \n"
      "movdqu      (%1),%%xmm2                   \n"
      "lea         0x10(%1),%1                   \n"
      "movdqu      %%xmm0,%%xmm1                 \n"
      "movdqu      %%xmm2,%%xmm3                 \n"
      "punpcklbw   %%xmm0,%%xmm0                 \n"
      "punpckhbw   %%xmm1,%%xmm1                 \n"
      "punpcklbw   %%xmm5,%%xmm2                 \n"
      "punpckhbw   %%xmm5,%%xmm3                 \n"
      "pmulhuw     %%xmm2,%%xmm0                 \n"
      "pmulhuw     %%xmm3,%%xmm1                 \n"
      "packuswb    %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%3                       \n"
      "jg          1b                            \n"
               : "+r"(src_argb),   // %0
                 "+r"(src_argb1),  // %1
                 "+r"(dst_argb),   // %2
                 "+r"(width)       // %3
               :
               : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}
#endif  // HAS_ARGBMULTIPLYROW_SSE2

#ifdef HAS_ARGBMULTIPLYROW_AVX2
// Multiply 2 rows of ARGB pixels together, 8 pixels at a time.
void ARGBMultiplyRow_AVX2(const uint8_t* src_argb,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
      asm volatile("vpxor       %%ymm5,%%ymm5,%%ymm5          \n"

               // 4 pixel loop.
               LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm1                   \n"
      "lea         0x20(%0),%0                   \n"
      "vmovdqu     (%1),%%ymm3                   \n"
      "lea         0x20(%1),%1                   \n"
      "vpunpcklbw  %%ymm1,%%ymm1,%%ymm0          \n"
      "vpunpckhbw  %%ymm1,%%ymm1,%%ymm1          \n"
      "vpunpcklbw  %%ymm5,%%ymm3,%%ymm2          \n"
      "vpunpckhbw  %%ymm5,%%ymm3,%%ymm3          \n"
      "vpmulhuw    %%ymm2,%%ymm0,%%ymm0          \n"
      "vpmulhuw    %%ymm3,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%2)                   \n"
      "lea         0x20(%2),%2                   \n"
      "sub         $0x8,%3                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_argb),   // %0
                 "+r"(src_argb1),  // %1
                 "+r"(dst_argb),   // %2
                 "+r"(width)       // %3
               :
               : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}
#endif  // HAS_ARGBMULTIPLYROW_AVX2

#ifdef HAS_ARGBADDROW_SSE2
// Add 2 rows of ARGB pixels together, 4 pixels at a time.
void ARGBAddRow_SSE2(const uint8_t* src_argb,
                     const uint8_t* src_argb1,
                     uint8_t* dst_argb,
                     int width) {
  asm volatile(
      // 4 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "lea         0x10(%0),%0                   \n"
      "movdqu      (%1),%%xmm1                   \n"
      "lea         0x10(%1),%1                   \n"
      "paddusb     %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%3                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "memory", "cc", "xmm0", "xmm1");
}
#endif  // HAS_ARGBADDROW_SSE2

#ifdef HAS_ARGBADDROW_AVX2
// Add 2 rows of ARGB pixels together, 4 pixels at a time.
void ARGBAddRow_AVX2(const uint8_t* src_argb,
                     const uint8_t* src_argb1,
                     uint8_t* dst_argb,
                     int width) {
  asm volatile(
      // 4 pixel loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "lea         0x20(%0),%0                   \n"
      "vpaddusb    (%1),%%ymm0,%%ymm0            \n"
      "lea         0x20(%1),%1                   \n"
      "vmovdqu     %%ymm0,(%2)                   \n"
      "lea         0x20(%2),%2                   \n"
      "sub         $0x8,%3                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "memory", "cc", "xmm0");
}
#endif  // HAS_ARGBADDROW_AVX2

#ifdef HAS_ARGBSUBTRACTROW_SSE2
// Subtract 2 rows of ARGB pixels, 4 pixels at a time.
void ARGBSubtractRow_SSE2(const uint8_t* src_argb,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
  asm volatile(
      // 4 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "lea         0x10(%0),%0                   \n"
      "movdqu      (%1),%%xmm1                   \n"
      "lea         0x10(%1),%1                   \n"
      "psubusb     %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%3                       \n"
      "jg          1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "memory", "cc", "xmm0", "xmm1");
}
#endif  // HAS_ARGBSUBTRACTROW_SSE2

#ifdef HAS_ARGBSUBTRACTROW_AVX2
// Subtract 2 rows of ARGB pixels, 8 pixels at a time.
void ARGBSubtractRow_AVX2(const uint8_t* src_argb,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
  asm volatile(
      // 4 pixel loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "lea         0x20(%0),%0                   \n"
      "vpsubusb    (%1),%%ymm0,%%ymm0            \n"
      "lea         0x20(%1),%1                   \n"
      "vmovdqu     %%ymm0,(%2)                   \n"
      "lea         0x20(%2),%2                   \n"
      "sub         $0x8,%3                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "memory", "cc", "xmm0");
}
#endif  // HAS_ARGBSUBTRACTROW_AVX2

#ifdef HAS_SOBELXROW_SSE2
// SobelX as a matrix is
// -1  0  1
// -2  0  2
// -1  0  1
void SobelXRow_SSE2(const uint8_t* src_y0,
                    const uint8_t* src_y1,
                    const uint8_t* src_y2,
                    uint8_t* dst_sobelx,
                    int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "sub         %0,%2                         \n"
      "sub         %0,%3                         \n"
      "pxor        %%xmm5,%%xmm5                 \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"
      "movq        0x2(%0),%%xmm1                \n"
      "punpcklbw   %%xmm5,%%xmm0                 \n"
      "punpcklbw   %%xmm5,%%xmm1                 \n"
      "psubw       %%xmm1,%%xmm0                 \n"
      "movq        0x00(%0,%1,1),%%xmm1          \n"
      "movq        0x02(%0,%1,1),%%xmm2          \n"
      "punpcklbw   %%xmm5,%%xmm1                 \n"
      "punpcklbw   %%xmm5,%%xmm2                 \n"
      "psubw       %%xmm2,%%xmm1                 \n"
      "movq        0x00(%0,%2,1),%%xmm2          \n"
      "movq        0x02(%0,%2,1),%%xmm3          \n"
      "punpcklbw   %%xmm5,%%xmm2                 \n"
      "punpcklbw   %%xmm5,%%xmm3                 \n"
      "psubw       %%xmm3,%%xmm2                 \n"
      "paddw       %%xmm2,%%xmm0                 \n"
      "paddw       %%xmm1,%%xmm0                 \n"
      "paddw       %%xmm1,%%xmm0                 \n"
      "pxor        %%xmm1,%%xmm1                 \n"
      "psubw       %%xmm0,%%xmm1                 \n"
      "pmaxsw      %%xmm1,%%xmm0                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movq        %%xmm0,0x00(%0,%3,1)          \n"
      "lea         0x8(%0),%0                    \n"
      "sub         $0x8,%4                       \n"
      "jg          1b                            \n"
      : "+r"(src_y0),      // %0
        "+r"(src_y1),      // %1
        "+r"(src_y2),      // %2
        "+r"(dst_sobelx),  // %3
        "+r"(width)        // %4
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}
#endif  // HAS_SOBELXROW_SSE2

#ifdef HAS_SOBELYROW_SSE2
// SobelY as a matrix is
// -1 -2 -1
//  0  0  0
//  1  2  1
void SobelYRow_SSE2(const uint8_t* src_y0,
                    const uint8_t* src_y1,
                    uint8_t* dst_sobely,
                    int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "sub         %0,%2                         \n"
      "pxor        %%xmm5,%%xmm5                 \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"
      "movq        0x00(%0,%1,1),%%xmm1          \n"
      "punpcklbw   %%xmm5,%%xmm0                 \n"
      "punpcklbw   %%xmm5,%%xmm1                 \n"
      "psubw       %%xmm1,%%xmm0                 \n"
      "movq        0x1(%0),%%xmm1                \n"
      "movq        0x01(%0,%1,1),%%xmm2          \n"
      "punpcklbw   %%xmm5,%%xmm1                 \n"
      "punpcklbw   %%xmm5,%%xmm2                 \n"
      "psubw       %%xmm2,%%xmm1                 \n"
      "movq        0x2(%0),%%xmm2                \n"
      "movq        0x02(%0,%1,1),%%xmm3          \n"
      "punpcklbw   %%xmm5,%%xmm2                 \n"
      "punpcklbw   %%xmm5,%%xmm3                 \n"
      "psubw       %%xmm3,%%xmm2                 \n"
      "paddw       %%xmm2,%%xmm0                 \n"
      "paddw       %%xmm1,%%xmm0                 \n"
      "paddw       %%xmm1,%%xmm0                 \n"
      "pxor        %%xmm1,%%xmm1                 \n"
      "psubw       %%xmm0,%%xmm1                 \n"
      "pmaxsw      %%xmm1,%%xmm0                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movq        %%xmm0,0x00(%0,%2,1)          \n"
      "lea         0x8(%0),%0                    \n"
      "sub         $0x8,%3                       \n"
      "jg          1b                            \n"
      : "+r"(src_y0),      // %0
        "+r"(src_y1),      // %1
        "+r"(dst_sobely),  // %2
        "+r"(width)        // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}
#endif  // HAS_SOBELYROW_SSE2

#ifdef HAS_SOBELROW_SSE2
// Adds Sobel X and Sobel Y and stores Sobel into ARGB.
// A = 255
// R = Sobel
// G = Sobel
// B = Sobel
void SobelRow_SSE2(const uint8_t* src_sobelx,
                   const uint8_t* src_sobely,
                   uint8_t* dst_argb,
                   int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "pslld       $0x18,%%xmm5                  \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x00(%0,%1,1),%%xmm1          \n"
      "lea         0x10(%0),%0                   \n"
      "paddusb     %%xmm1,%%xmm0                 \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "punpcklbw   %%xmm0,%%xmm2                 \n"
      "punpckhbw   %%xmm0,%%xmm0                 \n"
      "movdqa      %%xmm2,%%xmm1                 \n"
      "punpcklwd   %%xmm2,%%xmm1                 \n"
      "punpckhwd   %%xmm2,%%xmm2                 \n"
      "por         %%xmm5,%%xmm1                 \n"
      "por         %%xmm5,%%xmm2                 \n"
      "movdqa      %%xmm0,%%xmm3                 \n"
      "punpcklwd   %%xmm0,%%xmm3                 \n"
      "punpckhwd   %%xmm0,%%xmm0                 \n"
      "por         %%xmm5,%%xmm3                 \n"
      "por         %%xmm5,%%xmm0                 \n"
      "movdqu      %%xmm1,(%2)                   \n"
      "movdqu      %%xmm2,0x10(%2)               \n"
      "movdqu      %%xmm3,0x20(%2)               \n"
      "movdqu      %%xmm0,0x30(%2)               \n"
      "lea         0x40(%2),%2                   \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_sobelx),  // %0
        "+r"(src_sobely),  // %1
        "+r"(dst_argb),    // %2
        "+r"(width)        // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm5");
}
#endif  // HAS_SOBELROW_SSE2

#ifdef HAS_SOBELTOPLANEROW_SSE2
// Adds Sobel X and Sobel Y and stores Sobel into a plane.
void SobelToPlaneRow_SSE2(const uint8_t* src_sobelx,
                          const uint8_t* src_sobely,
                          uint8_t* dst_y,
                          int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"
      "pslld       $0x18,%%xmm5                  \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x00(%0,%1,1),%%xmm1          \n"
      "lea         0x10(%0),%0                   \n"
      "paddusb     %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_sobelx),  // %0
        "+r"(src_sobely),  // %1
        "+r"(dst_y),       // %2
        "+r"(width)        // %3
      :
      : "memory", "cc", "xmm0", "xmm1");
}
#endif  // HAS_SOBELTOPLANEROW_SSE2

#ifdef HAS_SOBELXYROW_SSE2
// Mixes Sobel X, Sobel Y and Sobel into ARGB.
// A = 255
// R = Sobel X
// G = Sobel
// B = Sobel Y
void SobelXYRow_SSE2(const uint8_t* src_sobelx,
                     const uint8_t* src_sobely,
                     uint8_t* dst_argb,
                     int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "pcmpeqb     %%xmm5,%%xmm5                 \n"

      // 8 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x00(%0,%1,1),%%xmm1          \n"
      "lea         0x10(%0),%0                   \n"
      "movdqa      %%xmm0,%%xmm2                 \n"
      "paddusb     %%xmm1,%%xmm2                 \n"
      "movdqa      %%xmm0,%%xmm3                 \n"
      "punpcklbw   %%xmm5,%%xmm3                 \n"
      "punpckhbw   %%xmm5,%%xmm0                 \n"
      "movdqa      %%xmm1,%%xmm4                 \n"
      "punpcklbw   %%xmm2,%%xmm4                 \n"
      "punpckhbw   %%xmm2,%%xmm1                 \n"
      "movdqa      %%xmm4,%%xmm6                 \n"
      "punpcklwd   %%xmm3,%%xmm6                 \n"
      "punpckhwd   %%xmm3,%%xmm4                 \n"
      "movdqa      %%xmm1,%%xmm7                 \n"
      "punpcklwd   %%xmm0,%%xmm7                 \n"
      "punpckhwd   %%xmm0,%%xmm1                 \n"
      "movdqu      %%xmm6,(%2)                   \n"
      "movdqu      %%xmm4,0x10(%2)               \n"
      "movdqu      %%xmm7,0x20(%2)               \n"
      "movdqu      %%xmm1,0x30(%2)               \n"
      "lea         0x40(%2),%2                   \n"
      "sub         $0x10,%3                      \n"
      "jg          1b                            \n"
      : "+r"(src_sobelx),  // %0
        "+r"(src_sobely),  // %1
        "+r"(dst_argb),    // %2
        "+r"(width)        // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_SOBELXYROW_SSE2

#ifdef HAS_COMPUTECUMULATIVESUMROW_SSE2
// Creates a table of cumulative sums where each value is a sum of all values
// above and to the left of the value, inclusive of the value.
void ComputeCumulativeSumRow_SSE2(const uint8_t* row,
                                  int32_t* cumsum,
                                  const int32_t* previous_cumsum,
                                  int width) {
  asm volatile(
      "pxor        %%xmm0,%%xmm0                 \n"
      "pxor        %%xmm1,%%xmm1                 \n"
      "sub         $0x4,%3                       \n"
      "jl          49f                           \n"
      "test        $0xf,%1                       \n"
      "jne         49f                           \n"

      // 4 pixel loop.
      LABELALIGN
      "40:         \n"
      "movdqu      (%0),%%xmm2                   \n"
      "lea         0x10(%0),%0                   \n"
      "movdqa      %%xmm2,%%xmm4                 \n"
      "punpcklbw   %%xmm1,%%xmm2                 \n"
      "movdqa      %%xmm2,%%xmm3                 \n"
      "punpcklwd   %%xmm1,%%xmm2                 \n"
      "punpckhwd   %%xmm1,%%xmm3                 \n"
      "punpckhbw   %%xmm1,%%xmm4                 \n"
      "movdqa      %%xmm4,%%xmm5                 \n"
      "punpcklwd   %%xmm1,%%xmm4                 \n"
      "punpckhwd   %%xmm1,%%xmm5                 \n"
      "paddd       %%xmm2,%%xmm0                 \n"
      "movdqu      (%2),%%xmm2                   \n"
      "paddd       %%xmm0,%%xmm2                 \n"
      "paddd       %%xmm3,%%xmm0                 \n"
      "movdqu      0x10(%2),%%xmm3               \n"
      "paddd       %%xmm0,%%xmm3                 \n"
      "paddd       %%xmm4,%%xmm0                 \n"
      "movdqu      0x20(%2),%%xmm4               \n"
      "paddd       %%xmm0,%%xmm4                 \n"
      "paddd       %%xmm5,%%xmm0                 \n"
      "movdqu      0x30(%2),%%xmm5               \n"
      "lea         0x40(%2),%2                   \n"
      "paddd       %%xmm0,%%xmm5                 \n"
      "movdqu      %%xmm2,(%1)                   \n"
      "movdqu      %%xmm3,0x10(%1)               \n"
      "movdqu      %%xmm4,0x20(%1)               \n"
      "movdqu      %%xmm5,0x30(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x4,%3                       \n"
      "jge         40b                           \n"

      "49:         \n"
      "add         $0x3,%3                       \n"
      "jl          19f                           \n"

      // 1 pixel loop.
      LABELALIGN
      "10:         \n"
      "movd        (%0),%%xmm2                   \n"
      "lea         0x4(%0),%0                    \n"
      "punpcklbw   %%xmm1,%%xmm2                 \n"
      "punpcklwd   %%xmm1,%%xmm2                 \n"
      "paddd       %%xmm2,%%xmm0                 \n"
      "movdqu      (%2),%%xmm2                   \n"
      "lea         0x10(%2),%2                   \n"
      "paddd       %%xmm0,%%xmm2                 \n"
      "movdqu      %%xmm2,(%1)                   \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x1,%3                       \n"
      "jge         10b                           \n"

      "19:         \n"
      : "+r"(row),              // %0
        "+r"(cumsum),           // %1
        "+r"(previous_cumsum),  // %2
        "+r"(width)             // %3
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_COMPUTECUMULATIVESUMROW_SSE2

#ifdef HAS_CUMULATIVESUMTOAVERAGEROW_SSE2
void CumulativeSumToAverageRow_SSE2(const int32_t* topleft,
                                    const int32_t* botleft,
                                    int width,
                                    int area,
                                    uint8_t* dst,
                                    int count) {
  asm volatile(
      "movd        %5,%%xmm5                     \n"
      "cvtdq2ps    %%xmm5,%%xmm5                 \n"
      "rcpss       %%xmm5,%%xmm4                 \n"
      "pshufd      $0x0,%%xmm4,%%xmm4            \n"
      "sub         $0x4,%3                       \n"
      "jl          49f                           \n"
      "cmpl        $0x80,%5                      \n"
      "ja          40f                           \n"

      "pshufd      $0x0,%%xmm5,%%xmm5            \n"
      "pcmpeqb     %%xmm6,%%xmm6                 \n"
      "psrld       $0x10,%%xmm6                  \n"
      "cvtdq2ps    %%xmm6,%%xmm6                 \n"
      "addps       %%xmm6,%%xmm5                 \n"
      "mulps       %%xmm4,%%xmm5                 \n"
      "cvtps2dq    %%xmm5,%%xmm5                 \n"
      "packssdw    %%xmm5,%%xmm5                 \n"

      // 4 pixel small loop.
      LABELALIGN
      "4:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm2               \n"
      "movdqu      0x30(%0),%%xmm3               \n"
      "psubd       0x00(%0,%4,4),%%xmm0          \n"
      "psubd       0x10(%0,%4,4),%%xmm1          \n"
      "psubd       0x20(%0,%4,4),%%xmm2          \n"
      "psubd       0x30(%0,%4,4),%%xmm3          \n"
      "lea         0x40(%0),%0                   \n"
      "psubd       (%1),%%xmm0                   \n"
      "psubd       0x10(%1),%%xmm1               \n"
      "psubd       0x20(%1),%%xmm2               \n"
      "psubd       0x30(%1),%%xmm3               \n"
      "paddd       0x00(%1,%4,4),%%xmm0          \n"
      "paddd       0x10(%1,%4,4),%%xmm1          \n"
      "paddd       0x20(%1,%4,4),%%xmm2          \n"
      "paddd       0x30(%1,%4,4),%%xmm3          \n"
      "lea         0x40(%1),%1                   \n"
      "packssdw    %%xmm1,%%xmm0                 \n"
      "packssdw    %%xmm3,%%xmm2                 \n"
      "pmulhuw     %%xmm5,%%xmm0                 \n"
      "pmulhuw     %%xmm5,%%xmm2                 \n"
      "packuswb    %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%3                       \n"
      "jge         4b                            \n"
      "jmp         49f                           \n"

      // 4 pixel loop
      LABELALIGN
      "40:         \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "movdqu      0x20(%0),%%xmm2               \n"
      "movdqu      0x30(%0),%%xmm3               \n"
      "psubd       0x00(%0,%4,4),%%xmm0          \n"
      "psubd       0x10(%0,%4,4),%%xmm1          \n"
      "psubd       0x20(%0,%4,4),%%xmm2          \n"
      "psubd       0x30(%0,%4,4),%%xmm3          \n"
      "lea         0x40(%0),%0                   \n"
      "psubd       (%1),%%xmm0                   \n"
      "psubd       0x10(%1),%%xmm1               \n"
      "psubd       0x20(%1),%%xmm2               \n"
      "psubd       0x30(%1),%%xmm3               \n"
      "paddd       0x00(%1,%4,4),%%xmm0          \n"
      "paddd       0x10(%1,%4,4),%%xmm1          \n"
      "paddd       0x20(%1,%4,4),%%xmm2          \n"
      "paddd       0x30(%1,%4,4),%%xmm3          \n"
      "lea         0x40(%1),%1                   \n"
      "cvtdq2ps    %%xmm0,%%xmm0                 \n"
      "cvtdq2ps    %%xmm1,%%xmm1                 \n"
      "mulps       %%xmm4,%%xmm0                 \n"
      "mulps       %%xmm4,%%xmm1                 \n"
      "cvtdq2ps    %%xmm2,%%xmm2                 \n"
      "cvtdq2ps    %%xmm3,%%xmm3                 \n"
      "mulps       %%xmm4,%%xmm2                 \n"
      "mulps       %%xmm4,%%xmm3                 \n"
      "cvtps2dq    %%xmm0,%%xmm0                 \n"
      "cvtps2dq    %%xmm1,%%xmm1                 \n"
      "cvtps2dq    %%xmm2,%%xmm2                 \n"
      "cvtps2dq    %%xmm3,%%xmm3                 \n"
      "packssdw    %%xmm1,%%xmm0                 \n"
      "packssdw    %%xmm3,%%xmm2                 \n"
      "packuswb    %%xmm2,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%3                       \n"
      "jge         40b                           \n"

      "49:         \n"
      "add         $0x3,%3                       \n"
      "jl          19f                           \n"

      // 1 pixel loop
      LABELALIGN
      "10:         \n"
      "movdqu      (%0),%%xmm0                   \n"
      "psubd       0x00(%0,%4,4),%%xmm0          \n"
      "lea         0x10(%0),%0                   \n"
      "psubd       (%1),%%xmm0                   \n"
      "paddd       0x00(%1,%4,4),%%xmm0          \n"
      "lea         0x10(%1),%1                   \n"
      "cvtdq2ps    %%xmm0,%%xmm0                 \n"
      "mulps       %%xmm4,%%xmm0                 \n"
      "cvtps2dq    %%xmm0,%%xmm0                 \n"
      "packssdw    %%xmm0,%%xmm0                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movd        %%xmm0,(%2)                   \n"
      "lea         0x4(%2),%2                    \n"
      "sub         $0x1,%3                       \n"
      "jge         10b                           \n"
      "19:         \n"
      : "+r"(topleft),           // %0
        "+r"(botleft),           // %1
        "+r"(dst),               // %2
        "+rm"(count)             // %3
      : "r"((intptr_t)(width)),  // %4
        "rm"(area)               // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}
#endif  // HAS_CUMULATIVESUMTOAVERAGEROW_SSE2

#ifdef HAS_ARGBAFFINEROW_SSE2
// Copy ARGB pixels from source image with slope to a row of destination.
LIBYUV_API
void ARGBAffineRow_SSE2(const uint8_t* src_argb,
                        int src_argb_stride,
                        uint8_t* dst_argb,
                        const float* src_dudv,
                        int width) {
  intptr_t src_argb_stride_temp = src_argb_stride;
  intptr_t temp;
  asm volatile(
      "movq        (%3),%%xmm2                   \n"
      "movq        0x08(%3),%%xmm7               \n"
      "shl         $0x10,%1                      \n"
      "add         $0x4,%1                       \n"
      "movd        %1,%%xmm5                     \n"
      "sub         $0x4,%4                       \n"
      "jl          49f                           \n"

      "pshufd      $0x44,%%xmm7,%%xmm7           \n"
      "pshufd      $0x0,%%xmm5,%%xmm5            \n"
      "movdqa      %%xmm2,%%xmm0                 \n"
      "addps       %%xmm7,%%xmm0                 \n"
      "movlhps     %%xmm0,%%xmm2                 \n"
      "movdqa      %%xmm7,%%xmm4                 \n"
      "addps       %%xmm4,%%xmm4                 \n"
      "movdqa      %%xmm2,%%xmm3                 \n"
      "addps       %%xmm4,%%xmm3                 \n"
      "addps       %%xmm4,%%xmm4                 \n"

      // 4 pixel loop
      LABELALIGN
      "40:         \n"
      "cvttps2dq   %%xmm2,%%xmm0                 \n"  // x,y float->int first 2
      "cvttps2dq   %%xmm3,%%xmm1                 \n"  // x,y float->int next 2
      "packssdw    %%xmm1,%%xmm0                 \n"  // x, y as 8 shorts
      "pmaddwd     %%xmm5,%%xmm0                 \n"  // off = x*4 + y*stride
      "movd        %%xmm0,%k1                    \n"
      "pshufd      $0x39,%%xmm0,%%xmm0           \n"
      "movd        %%xmm0,%k5                    \n"
      "pshufd      $0x39,%%xmm0,%%xmm0           \n"
      "movd        0x00(%0,%1,1),%%xmm1          \n"
      "movd        0x00(%0,%5,1),%%xmm6          \n"
      "punpckldq   %%xmm6,%%xmm1                 \n"
      "addps       %%xmm4,%%xmm2                 \n"
      "movq        %%xmm1,(%2)                   \n"
      "movd        %%xmm0,%k1                    \n"
      "pshufd      $0x39,%%xmm0,%%xmm0           \n"
      "movd        %%xmm0,%k5                    \n"
      "movd        0x00(%0,%1,1),%%xmm0          \n"
      "movd        0x00(%0,%5,1),%%xmm6          \n"
      "punpckldq   %%xmm6,%%xmm0                 \n"
      "addps       %%xmm4,%%xmm3                 \n"
      "movq        %%xmm0,0x08(%2)               \n"
      "lea         0x10(%2),%2                   \n"
      "sub         $0x4,%4                       \n"
      "jge         40b                           \n"

      "49:         \n"
      "add         $0x3,%4                       \n"
      "jl          19f                           \n"

      // 1 pixel loop
      LABELALIGN
      "10:         \n"
      "cvttps2dq   %%xmm2,%%xmm0                 \n"
      "packssdw    %%xmm0,%%xmm0                 \n"
      "pmaddwd     %%xmm5,%%xmm0                 \n"
      "addps       %%xmm7,%%xmm2                 \n"
      "movd        %%xmm0,%k1                    \n"
      "movd        0x00(%0,%1,1),%%xmm0          \n"
      "movd        %%xmm0,(%2)                   \n"
      "lea         0x04(%2),%2                   \n"
      "sub         $0x1,%4                       \n"
      "jge         10b                           \n"
      "19:         \n"
      : "+r"(src_argb),              // %0
        "+r"(src_argb_stride_temp),  // %1
        "+r"(dst_argb),              // %2
        "+r"(src_dudv),              // %3
        "+rm"(width),                // %4
        "=&r"(temp)                  // %5
      :
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBAFFINEROW_SSE2

#ifdef HAS_INTERPOLATEROW_SSSE3
// Bilinear filter 16x2 -> 16x1
void InterpolateRow_SSSE3(uint8_t* dst_ptr,
                          const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          int width,
                          int source_y_fraction) {
  asm volatile(
      "sub         %1,%0                         \n"
      "cmp         $0x0,%3                       \n"
      "je          100f                          \n"
      "cmp         $0x80,%3                      \n"
      "je          50f                           \n"

      "movd        %3,%%xmm0                     \n"
      "neg         %3                            \n"
      "add         $0x100,%3                     \n"
      "movd        %3,%%xmm5                     \n"
      "punpcklbw   %%xmm0,%%xmm5                 \n"
      "punpcklwd   %%xmm5,%%xmm5                 \n"
      "pshufd      $0x0,%%xmm5,%%xmm5            \n"
      "mov         $0x80808080,%%eax             \n"
      "movd        %%eax,%%xmm4                  \n"
      "pshufd      $0x0,%%xmm4,%%xmm4            \n"

      // General purpose row blend.
      LABELALIGN
      "1:          \n"
      "movdqu      (%1),%%xmm0                   \n"
      "movdqu      0x00(%1,%4,1),%%xmm2          \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm2,%%xmm0                 \n"
      "punpckhbw   %%xmm2,%%xmm1                 \n"
      "psubb       %%xmm4,%%xmm0                 \n"
      "psubb       %%xmm4,%%xmm1                 \n"
      "movdqa      %%xmm5,%%xmm2                 \n"
      "movdqa      %%xmm5,%%xmm3                 \n"
      "pmaddubsw   %%xmm0,%%xmm2                 \n"
      "pmaddubsw   %%xmm1,%%xmm3                 \n"
      "paddw       %%xmm4,%%xmm2                 \n"
      "paddw       %%xmm4,%%xmm3                 \n"
      "psrlw       $0x8,%%xmm2                   \n"
      "psrlw       $0x8,%%xmm3                   \n"
      "packuswb    %%xmm3,%%xmm2                 \n"
      "movdqu      %%xmm2,0x00(%1,%0,1)          \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "jmp         99f                           \n"

      // Blend 50 / 50.
      LABELALIGN
      "50:         \n"
      "movdqu      (%1),%%xmm0                   \n"
      "movdqu      0x00(%1,%4,1),%%xmm1          \n"
      "pavgb       %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,0x00(%1,%0,1)          \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          50b                           \n"
      "jmp         99f                           \n"

      // Blend 100 / 0 - Copy row unchanged.
      LABELALIGN
      "100:        \n"
      "movdqu      (%1),%%xmm0                   \n"
      "movdqu      %%xmm0,0x00(%1,%0,1)          \n"
      "lea         0x10(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          100b                          \n"

      "99:         \n"
      : "+r"(dst_ptr),               // %0
        "+r"(src_ptr),               // %1
        "+rm"(width),                // %2
        "+r"(source_y_fraction)      // %3
      : "r"((intptr_t)(src_stride))  // %4
      : "memory", "cc", "eax", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_INTERPOLATEROW_SSSE3

#ifdef HAS_INTERPOLATEROW_AVX2
// Bilinear filter 32x2 -> 32x1
void InterpolateRow_AVX2(uint8_t* dst_ptr,
                         const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         int width,
                         int source_y_fraction) {
  asm volatile(
      "sub         %1,%0                         \n"
      "cmp         $0x0,%3                       \n"
      "je          100f                          \n"
      "cmp         $0x80,%3                      \n"
      "je          50f                           \n"

      "vmovd       %3,%%xmm0                     \n"
      "neg         %3                            \n"
      "add         $0x100,%3                     \n"
      "vmovd       %3,%%xmm5                     \n"
      "vpunpcklbw  %%xmm0,%%xmm5,%%xmm5          \n"
      "vpunpcklwd  %%xmm5,%%xmm5,%%xmm5          \n"
      "vbroadcastss %%xmm5,%%ymm5                \n"
      "mov         $0x80808080,%%eax             \n"
      "vmovd       %%eax,%%xmm4                  \n"
      "vbroadcastss %%xmm4,%%ymm4                \n"

      // General purpose row blend.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%1),%%ymm0                   \n"
      "vmovdqu     0x00(%1,%4,1),%%ymm2          \n"
      "vpunpckhbw  %%ymm2,%%ymm0,%%ymm1          \n"
      "vpunpcklbw  %%ymm2,%%ymm0,%%ymm0          \n"
      "vpsubb      %%ymm4,%%ymm1,%%ymm1          \n"
      "vpsubb      %%ymm4,%%ymm0,%%ymm0          \n"
      "vpmaddubsw  %%ymm1,%%ymm5,%%ymm1          \n"
      "vpmaddubsw  %%ymm0,%%ymm5,%%ymm0          \n"
      "vpaddw      %%ymm4,%%ymm1,%%ymm1          \n"
      "vpaddw      %%ymm4,%%ymm0,%%ymm0          \n"
      "vpsrlw      $0x8,%%ymm1,%%ymm1            \n"
      "vpsrlw      $0x8,%%ymm0,%%ymm0            \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,0x00(%1,%0,1)          \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "jmp         99f                           \n"

      // Blend 50 / 50.
      LABELALIGN
      "50:         \n"
      "vmovdqu     (%1),%%ymm0                   \n"
      "vpavgb      0x00(%1,%4,1),%%ymm0,%%ymm0   \n"
      "vmovdqu     %%ymm0,0x00(%1,%0,1)          \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          50b                           \n"
      "jmp         99f                           \n"

      // Blend 100 / 0 - Copy row unchanged.
      LABELALIGN
      "100:        \n"
      "vmovdqu     (%1),%%ymm0                   \n"
      "vmovdqu     %%ymm0,0x00(%1,%0,1)          \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          100b                          \n"

      "99:         \n"
      "vzeroupper  \n"
      : "+r"(dst_ptr),               // %0
        "+r"(src_ptr),               // %1
        "+r"(width),                 // %2
        "+r"(source_y_fraction)      // %3
      : "r"((intptr_t)(src_stride))  // %4
      : "memory", "cc", "eax", "xmm0", "xmm1", "xmm2", "xmm4", "xmm5");
}
#endif  // HAS_INTERPOLATEROW_AVX2

#ifdef HAS_ARGBSHUFFLEROW_SSSE3
// For BGRAToARGB, ABGRToARGB, RGBAToARGB, and ARGBToRGBA.
void ARGBShuffleRow_SSSE3(const uint8_t* src_argb,
                          uint8_t* dst_argb,
                          const uint8_t* shuffler,
                          int width) {
      asm volatile("movdqu      (%3),%%xmm5                   \n"

               LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "pshufb      %%xmm5,%%xmm0                 \n"
      "pshufb      %%xmm5,%%xmm1                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
               : "+r"(src_argb),  // %0
                 "+r"(dst_argb),  // %1
                 "+r"(width)      // %2
               : "r"(shuffler)    // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm5");
}
#endif  // HAS_ARGBSHUFFLEROW_SSSE3

#ifdef HAS_ARGBSHUFFLEROW_AVX2
// For BGRAToARGB, ABGRToARGB, RGBAToARGB, and ARGBToRGBA.
void ARGBShuffleRow_AVX2(const uint8_t* src_argb,
                         uint8_t* dst_argb,
                         const uint8_t* shuffler,
                         int width) {
      asm volatile("vbroadcastf128 (%3),%%ymm5                \n"

               LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpshufb     %%ymm5,%%ymm0,%%ymm0          \n"
      "vpshufb     %%ymm5,%%ymm1,%%ymm1          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_argb),  // %0
                 "+r"(dst_argb),  // %1
                 "+r"(width)      // %2
               : "r"(shuffler)    // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm5");
}
#endif  // HAS_ARGBSHUFFLEROW_AVX2

#ifdef HAS_I422TOYUY2ROW_SSE2
void I422ToYUY2Row_SSE2(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_yuy2,
                        int width) {
      asm volatile("sub         %1,%2                         \n"

               LABELALIGN
      "1:          \n"
      "movq        (%1),%%xmm2                   \n"
      "movq        0x00(%1,%2,1),%%xmm1          \n"
      "add         $0x8,%1                       \n"
      "punpcklbw   %%xmm1,%%xmm2                 \n"
      "movdqu      (%0),%%xmm0                   \n"
      "add         $0x10,%0                      \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "punpcklbw   %%xmm2,%%xmm0                 \n"
      "punpckhbw   %%xmm2,%%xmm1                 \n"
      "movdqu      %%xmm0,(%3)                   \n"
      "movdqu      %%xmm1,0x10(%3)               \n"
      "lea         0x20(%3),%3                   \n"
      "sub         $0x10,%4                      \n"
      "jg          1b                            \n"
               : "+r"(src_y),     // %0
                 "+r"(src_u),     // %1
                 "+r"(src_v),     // %2
                 "+r"(dst_yuy2),  // %3
                 "+rm"(width)     // %4
               :
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_I422TOYUY2ROW_SSE2

#ifdef HAS_I422TOUYVYROW_SSE2
void I422ToUYVYRow_SSE2(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_uyvy,
                        int width) {
      asm volatile("sub         %1,%2                         \n"

               LABELALIGN
      "1:          \n"
      "movq        (%1),%%xmm2                   \n"
      "movq        0x00(%1,%2,1),%%xmm1          \n"
      "add         $0x8,%1                       \n"
      "punpcklbw   %%xmm1,%%xmm2                 \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqa      %%xmm2,%%xmm1                 \n"
      "add         $0x10,%0                      \n"
      "punpcklbw   %%xmm0,%%xmm1                 \n"
      "punpckhbw   %%xmm0,%%xmm2                 \n"
      "movdqu      %%xmm1,(%3)                   \n"
      "movdqu      %%xmm2,0x10(%3)               \n"
      "lea         0x20(%3),%3                   \n"
      "sub         $0x10,%4                      \n"
      "jg          1b                            \n"
               : "+r"(src_y),     // %0
                 "+r"(src_u),     // %1
                 "+r"(src_v),     // %2
                 "+r"(dst_uyvy),  // %3
                 "+rm"(width)     // %4
               :
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_I422TOUYVYROW_SSE2

#ifdef HAS_I422TOYUY2ROW_AVX2
void I422ToYUY2Row_AVX2(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_yuy2,
                        int width) {
      asm volatile("sub         %1,%2                         \n"

               LABELALIGN
      "1:          \n"
      "vpmovzxbw   (%1),%%ymm1                   \n"
      "vpmovzxbw   0x00(%1,%2,1),%%ymm2          \n"
      "add         $0x10,%1                      \n"
      "vpsllw      $0x8,%%ymm2,%%ymm2            \n"
      "vpor        %%ymm1,%%ymm2,%%ymm2          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "add         $0x20,%0                      \n"
      "vpunpcklbw  %%ymm2,%%ymm0,%%ymm1          \n"
      "vpunpckhbw  %%ymm2,%%ymm0,%%ymm2          \n"
      "vextractf128 $0x0,%%ymm1,(%3)             \n"
      "vextractf128 $0x0,%%ymm2,0x10(%3)         \n"
      "vextractf128 $0x1,%%ymm1,0x20(%3)         \n"
      "vextractf128 $0x1,%%ymm2,0x30(%3)         \n"
      "lea         0x40(%3),%3                   \n"
      "sub         $0x20,%4                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_y),     // %0
                 "+r"(src_u),     // %1
                 "+r"(src_v),     // %2
                 "+r"(dst_yuy2),  // %3
                 "+rm"(width)     // %4
               :
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_I422TOYUY2ROW_AVX2

#ifdef HAS_I422TOUYVYROW_AVX2
void I422ToUYVYRow_AVX2(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_uyvy,
                        int width) {
      asm volatile("sub         %1,%2                         \n"

               LABELALIGN
      "1:          \n"
      "vpmovzxbw   (%1),%%ymm1                   \n"
      "vpmovzxbw   0x00(%1,%2,1),%%ymm2          \n"
      "add         $0x10,%1                      \n"
      "vpsllw      $0x8,%%ymm2,%%ymm2            \n"
      "vpor        %%ymm1,%%ymm2,%%ymm2          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "add         $0x20,%0                      \n"
      "vpunpcklbw  %%ymm0,%%ymm2,%%ymm1          \n"
      "vpunpckhbw  %%ymm0,%%ymm2,%%ymm2          \n"
      "vextractf128 $0x0,%%ymm1,(%3)             \n"
      "vextractf128 $0x0,%%ymm2,0x10(%3)         \n"
      "vextractf128 $0x1,%%ymm1,0x20(%3)         \n"
      "vextractf128 $0x1,%%ymm2,0x30(%3)         \n"
      "lea         0x40(%3),%3                   \n"
      "sub         $0x20,%4                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_y),     // %0
                 "+r"(src_u),     // %1
                 "+r"(src_v),     // %2
                 "+r"(dst_uyvy),  // %3
                 "+rm"(width)     // %4
               :
               : "memory", "cc", "xmm0", "xmm1", "xmm2");
}
#endif  // HAS_I422TOUYVYROW_AVX2

#ifdef HAS_ARGBPOLYNOMIALROW_SSE2
void ARGBPolynomialRow_SSE2(const uint8_t* src_argb,
                            uint8_t* dst_argb,
                            const float* poly,
                            int width) {
      asm volatile("pxor        %%xmm3,%%xmm3                 \n"

               // 2 pixel loop.
               LABELALIGN
      "1:          \n"
      "movq        (%0),%%xmm0                   \n"
      "lea         0x8(%0),%0                    \n"
      "punpcklbw   %%xmm3,%%xmm0                 \n"
      "movdqa      %%xmm0,%%xmm4                 \n"
      "punpcklwd   %%xmm3,%%xmm0                 \n"
      "punpckhwd   %%xmm3,%%xmm4                 \n"
      "cvtdq2ps    %%xmm0,%%xmm0                 \n"
      "cvtdq2ps    %%xmm4,%%xmm4                 \n"
      "movdqa      %%xmm0,%%xmm1                 \n"
      "movdqa      %%xmm4,%%xmm5                 \n"
      "mulps       0x10(%3),%%xmm0               \n"
      "mulps       0x10(%3),%%xmm4               \n"
      "addps       (%3),%%xmm0                   \n"
      "addps       (%3),%%xmm4                   \n"
      "movdqa      %%xmm1,%%xmm2                 \n"
      "movdqa      %%xmm5,%%xmm6                 \n"
      "mulps       %%xmm1,%%xmm2                 \n"
      "mulps       %%xmm5,%%xmm6                 \n"
      "mulps       %%xmm2,%%xmm1                 \n"
      "mulps       %%xmm6,%%xmm5                 \n"
      "mulps       0x20(%3),%%xmm2               \n"
      "mulps       0x20(%3),%%xmm6               \n"
      "mulps       0x30(%3),%%xmm1               \n"
      "mulps       0x30(%3),%%xmm5               \n"
      "addps       %%xmm2,%%xmm0                 \n"
      "addps       %%xmm6,%%xmm4                 \n"
      "addps       %%xmm1,%%xmm0                 \n"
      "addps       %%xmm5,%%xmm4                 \n"
      "cvttps2dq   %%xmm0,%%xmm0                 \n"
      "cvttps2dq   %%xmm4,%%xmm4                 \n"
      "packuswb    %%xmm4,%%xmm0                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "movq        %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x2,%2                       \n"
      "jg          1b                            \n"
               : "+r"(src_argb),  // %0
                 "+r"(dst_argb),  // %1
                 "+r"(width)      // %2
               : "r"(poly)        // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
                 "xmm6");
}
#endif  // HAS_ARGBPOLYNOMIALROW_SSE2

#ifdef HAS_ARGBPOLYNOMIALROW_AVX2
void ARGBPolynomialRow_AVX2(const uint8_t* src_argb,
                            uint8_t* dst_argb,
                            const float* poly,
                            int width) {
  asm volatile(
      "vbroadcastf128 (%3),%%ymm4                \n"
      "vbroadcastf128 0x10(%3),%%ymm5            \n"
      "vbroadcastf128 0x20(%3),%%ymm6            \n"
      "vbroadcastf128 0x30(%3),%%ymm7            \n"

      // 2 pixel loop.
      LABELALIGN
      "1:          \n"
      "vpmovzxbd   (%0),%%ymm0                   \n"  // 2 ARGB pixels
      "lea         0x8(%0),%0                    \n"
      "vcvtdq2ps   %%ymm0,%%ymm0                 \n"  // X 8 floats
      "vmulps      %%ymm0,%%ymm0,%%ymm2          \n"  // X * X
      "vmulps      %%ymm7,%%ymm0,%%ymm3          \n"  // C3 * X
      "vfmadd132ps %%ymm5,%%ymm4,%%ymm0          \n"  // result = C0 + C1 * X
      "vfmadd231ps %%ymm6,%%ymm2,%%ymm0          \n"  // result += C2 * X * X
      "vfmadd231ps %%ymm3,%%ymm2,%%ymm0          \n"  // result += C3 * X * X *
                                                      // X
      "vcvttps2dq  %%ymm0,%%ymm0                 \n"
      "vpackusdw   %%ymm0,%%ymm0,%%ymm0          \n"
      "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
      "vpackuswb   %%xmm0,%%xmm0,%%xmm0          \n"
      "vmovq       %%xmm0,(%1)                   \n"
      "lea         0x8(%1),%1                    \n"
      "sub         $0x2,%2                       \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      : "r"(poly)        // %3
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
        "xmm7");
}
#endif  // HAS_ARGBPOLYNOMIALROW_AVX2

#ifdef HAS_HALFFLOATROW_SSE2
static float kScaleBias = 1.9259299444e-34f;
void HalfFloatRow_SSE2(const uint16_t* src,
                       uint16_t* dst,
                       float scale,
                       int width) {
  scale *= kScaleBias;
  asm volatile(
      "movd        %3,%%xmm4                     \n"
      "pshufd      $0x0,%%xmm4,%%xmm4            \n"
      "pxor        %%xmm5,%%xmm5                 \n"
      "sub         %0,%1                         \n"

      // 16 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm2                   \n"  // 8 shorts
      "add         $0x10,%0                      \n"
      "movdqa      %%xmm2,%%xmm3                 \n"
      "punpcklwd   %%xmm5,%%xmm2                 \n"  // 8 ints in xmm2/1
      "cvtdq2ps    %%xmm2,%%xmm2                 \n"  // 8 floats
      "punpckhwd   %%xmm5,%%xmm3                 \n"
      "cvtdq2ps    %%xmm3,%%xmm3                 \n"
      "mulps       %%xmm4,%%xmm2                 \n"
      "mulps       %%xmm4,%%xmm3                 \n"
      "psrld       $0xd,%%xmm2                   \n"
      "psrld       $0xd,%%xmm3                   \n"
      "packssdw    %%xmm3,%%xmm2                 \n"
      "movdqu      %%xmm2,-0x10(%0,%1,1)         \n"
      "sub         $0x8,%2                       \n"
      "jg          1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      : "m"(scale)   // %3
      : "memory", "cc", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_HALFFLOATROW_SSE2

#ifdef HAS_HALFFLOATROW_AVX2
void HalfFloatRow_AVX2(const uint16_t* src,
                       uint16_t* dst,
                       float scale,
                       int width) {
  scale *= kScaleBias;
  asm volatile(
      "vbroadcastss %3, %%ymm4                   \n"
      "vpxor       %%ymm5,%%ymm5,%%ymm5          \n"
      "sub         %0,%1                         \n"

      // 16 pixel loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm2                   \n"  // 16 shorts
      "add         $0x20,%0                      \n"
      "vpunpckhwd  %%ymm5,%%ymm2,%%ymm3          \n"  // mutates
      "vpunpcklwd  %%ymm5,%%ymm2,%%ymm2          \n"
      "vcvtdq2ps   %%ymm3,%%ymm3                 \n"
      "vcvtdq2ps   %%ymm2,%%ymm2                 \n"
      "vmulps      %%ymm3,%%ymm4,%%ymm3          \n"
      "vmulps      %%ymm2,%%ymm4,%%ymm2          \n"
      "vpsrld      $0xd,%%ymm3,%%ymm3            \n"
      "vpsrld      $0xd,%%ymm2,%%ymm2            \n"
      "vpackssdw   %%ymm3, %%ymm2, %%ymm2        \n"  // unmutates
      "vmovdqu     %%ymm2,-0x20(%0,%1,1)         \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"

      "vzeroupper  \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
#if defined(__x86_64__)
      : "x"(scale)  // %3
#else
      : "m"(scale)    // %3
#endif
      : "memory", "cc", "xmm2", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_HALFFLOATROW_AVX2

#ifdef HAS_HALFFLOATROW_F16C
void HalfFloatRow_F16C(const uint16_t* src,
                       uint16_t* dst,
                       float scale,
                       int width) {
  asm volatile(
      "vbroadcastss %3, %%ymm4                   \n"
      "sub         %0,%1                         \n"

      // 16 pixel loop.
      LABELALIGN
      "1:          \n"
      "vpmovzxwd   (%0),%%ymm2                   \n"  // 16 shorts -> 16 ints
      "vpmovzxwd   0x10(%0),%%ymm3               \n"
      "vcvtdq2ps   %%ymm2,%%ymm2                 \n"
      "vcvtdq2ps   %%ymm3,%%ymm3                 \n"
      "vmulps      %%ymm2,%%ymm4,%%ymm2          \n"
      "vmulps      %%ymm3,%%ymm4,%%ymm3          \n"
      "vcvtps2ph   $3, %%ymm2, %%xmm2            \n"
      "vcvtps2ph   $3, %%ymm3, %%xmm3            \n"
      "vmovdqu     %%xmm2,0x00(%0,%1,1)          \n"
      "vmovdqu     %%xmm3,0x10(%0,%1,1)          \n"
      "add         $0x20,%0                      \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
#if defined(__x86_64__)
      : "x"(scale)  // %3
#else
      : "m"(scale)    // %3
#endif
      : "memory", "cc", "xmm2", "xmm3", "xmm4");
}
#endif  // HAS_HALFFLOATROW_F16C

#ifdef HAS_HALFFLOATROW_F16C
void HalfFloat1Row_F16C(const uint16_t* src, uint16_t* dst, float, int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      // 16 pixel loop.
      LABELALIGN
      "1:          \n"
      "vpmovzxwd   (%0),%%ymm2                   \n"  // 16 shorts -> 16 ints
      "vpmovzxwd   0x10(%0),%%ymm3               \n"
      "vcvtdq2ps   %%ymm2,%%ymm2                 \n"
      "vcvtdq2ps   %%ymm3,%%ymm3                 \n"
      "vcvtps2ph   $3, %%ymm2, %%xmm2            \n"
      "vcvtps2ph   $3, %%ymm3, %%xmm3            \n"
      "vmovdqu     %%xmm2,0x00(%0,%1,1)          \n"
      "vmovdqu     %%xmm3,0x10(%0,%1,1)          \n"
      "add         $0x20,%0                      \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "memory", "cc", "xmm2", "xmm3");
}
#endif  // HAS_HALFFLOATROW_F16C

#ifdef HAS_ARGBCOLORTABLEROW_X86
// Tranform ARGB pixels with color table.
void ARGBColorTableRow_X86(uint8_t* dst_argb,
                           const uint8_t* table_argb,
                           int width) {
  uintptr_t pixel_temp;
  asm volatile(
      // 1 pixel loop.
      LABELALIGN
      "1:          \n"
      "movzb       (%0),%1                       \n"
      "lea         0x4(%0),%0                    \n"
      "movzb       0x00(%3,%1,4),%1              \n"
      "mov         %b1,-0x4(%0)                  \n"
      "movzb       -0x3(%0),%1                   \n"
      "movzb       0x01(%3,%1,4),%1              \n"
      "mov         %b1,-0x3(%0)                  \n"
      "movzb       -0x2(%0),%1                   \n"
      "movzb       0x02(%3,%1,4),%1              \n"
      "mov         %b1,-0x2(%0)                  \n"
      "movzb       -0x1(%0),%1                   \n"
      "movzb       0x03(%3,%1,4),%1              \n"
      "mov         %b1,-0x1(%0)                  \n"
      "dec         %2                            \n"
      "jg          1b                            \n"
      : "+r"(dst_argb),     // %0
        "=&d"(pixel_temp),  // %1
        "+r"(width)         // %2
      : "r"(table_argb)     // %3
      : "memory", "cc");
}
#endif  // HAS_ARGBCOLORTABLEROW_X86

#ifdef HAS_RGBCOLORTABLEROW_X86
// Tranform RGB pixels with color table.
void RGBColorTableRow_X86(uint8_t* dst_argb,
                          const uint8_t* table_argb,
                          int width) {
  uintptr_t pixel_temp;
  asm volatile(
      // 1 pixel loop.
      LABELALIGN
      "1:          \n"
      "movzb       (%0),%1                       \n"
      "lea         0x4(%0),%0                    \n"
      "movzb       0x00(%3,%1,4),%1              \n"
      "mov         %b1,-0x4(%0)                  \n"
      "movzb       -0x3(%0),%1                   \n"
      "movzb       0x01(%3,%1,4),%1              \n"
      "mov         %b1,-0x3(%0)                  \n"
      "movzb       -0x2(%0),%1                   \n"
      "movzb       0x02(%3,%1,4),%1              \n"
      "mov         %b1,-0x2(%0)                  \n"
      "dec         %2                            \n"
      "jg          1b                            \n"
      : "+r"(dst_argb),     // %0
        "=&d"(pixel_temp),  // %1
        "+r"(width)         // %2
      : "r"(table_argb)     // %3
      : "memory", "cc");
}
#endif  // HAS_RGBCOLORTABLEROW_X86

#ifdef HAS_ARGBLUMACOLORTABLEROW_SSSE3
// Tranform RGB pixels with luma table.
void ARGBLumaColorTableRow_SSSE3(const uint8_t* src_argb,
                                 uint8_t* dst_argb,
                                 int width,
                                 const uint8_t* luma,
                                 uint32_t lumacoeff) {
  uintptr_t pixel_temp;
  uintptr_t table_temp;
  asm volatile(
      "movd        %6,%%xmm3                     \n"
      "pshufd      $0x0,%%xmm3,%%xmm3            \n"
      "pcmpeqb     %%xmm4,%%xmm4                 \n"
      "psllw       $0x8,%%xmm4                   \n"
      "pxor        %%xmm5,%%xmm5                 \n"

      // 4 pixel loop.
      LABELALIGN
      "1:          \n"
      "movdqu      (%2),%%xmm0                   \n"
      "pmaddubsw   %%xmm3,%%xmm0                 \n"
      "phaddw      %%xmm0,%%xmm0                 \n"
      "pand        %%xmm4,%%xmm0                 \n"
      "punpcklwd   %%xmm5,%%xmm0                 \n"
      "movd        %%xmm0,%k1                    \n"  // 32 bit offset
      "add         %5,%1                         \n"
      "pshufd      $0x39,%%xmm0,%%xmm0           \n"

      "movzb       (%2),%0                       \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,(%3)                      \n"
      "movzb       0x1(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0x1(%3)                   \n"
      "movzb       0x2(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0x2(%3)                   \n"
      "movzb       0x3(%2),%0                    \n"
      "mov         %b0,0x3(%3)                   \n"

      "movd        %%xmm0,%k1                    \n"  // 32 bit offset
      "add         %5,%1                         \n"
      "pshufd      $0x39,%%xmm0,%%xmm0           \n"

      "movzb       0x4(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0x4(%3)                   \n"
      "movzb       0x5(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0x5(%3)                   \n"
      "movzb       0x6(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0x6(%3)                   \n"
      "movzb       0x7(%2),%0                    \n"
      "mov         %b0,0x7(%3)                   \n"

      "movd        %%xmm0,%k1                    \n"  // 32 bit offset
      "add         %5,%1                         \n"
      "pshufd      $0x39,%%xmm0,%%xmm0           \n"

      "movzb       0x8(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0x8(%3)                   \n"
      "movzb       0x9(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0x9(%3)                   \n"
      "movzb       0xa(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0xa(%3)                   \n"
      "movzb       0xb(%2),%0                    \n"
      "mov         %b0,0xb(%3)                   \n"

      "movd        %%xmm0,%k1                    \n"  // 32 bit offset
      "add         %5,%1                         \n"

      "movzb       0xc(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0xc(%3)                   \n"
      "movzb       0xd(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0xd(%3)                   \n"
      "movzb       0xe(%2),%0                    \n"
      "movzb       0x00(%1,%0,1),%0              \n"
      "mov         %b0,0xe(%3)                   \n"
      "movzb       0xf(%2),%0                    \n"
      "mov         %b0,0xf(%3)                   \n"
      "lea         0x10(%2),%2                   \n"
      "lea         0x10(%3),%3                   \n"
      "sub         $0x4,%4                       \n"
      "jg          1b                            \n"
      : "=&d"(pixel_temp),  // %0
        "=&a"(table_temp),  // %1
        "+r"(src_argb),     // %2
        "+r"(dst_argb),     // %3
        "+rm"(width)        // %4
      : "r"(luma),          // %5
        "rm"(lumacoeff)     // %6
      : "memory", "cc", "xmm0", "xmm3", "xmm4", "xmm5");
}
#endif  // HAS_ARGBLUMACOLORTABLEROW_SSSE3

static const uvec8 kYUV24Shuffle[3] = {
    {8, 9, 0, 8, 9, 1, 10, 11, 2, 10, 11, 3, 12, 13, 4, 12},
    {9, 1, 10, 11, 2, 10, 11, 3, 12, 13, 4, 12, 13, 5, 14, 15},
    {2, 10, 11, 3, 12, 13, 4, 12, 13, 5, 14, 15, 6, 14, 15, 7}};

// Convert biplanar NV21 to packed YUV24
// NV21 has VU in memory for chroma.
// YUV24 is VUY in memory
void NV21ToYUV24Row_SSSE3(const uint8_t* src_y,
                          const uint8_t* src_vu,
                          uint8_t* dst_yuv24,
                          int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "movdqa      (%4),%%xmm4                   \n"  // 3 shuffler constants
      "movdqa      16(%4),%%xmm5                 \n"
      "movdqa      32(%4),%%xmm6                 \n"
      "1:          \n"
      "movdqu      (%0),%%xmm2                   \n"  // load 16 Y values
      "movdqu      (%0,%1),%%xmm3                \n"  // load 8 VU values
      "lea         16(%0),%0                     \n"
      "movdqa      %%xmm2,%%xmm0                 \n"
      "movdqa      %%xmm2,%%xmm1                 \n"
      "shufps      $0x44,%%xmm3,%%xmm0           \n"  // Y 0..7,  UV 0..3
      "shufps      $0x99,%%xmm3,%%xmm1           \n"  // Y 4..11, UV 2..5
      "shufps      $0xee,%%xmm3,%%xmm2           \n"  // Y 8..15, UV 4..7
      "pshufb      %%xmm4, %%xmm0                \n"  // weave into YUV24
      "pshufb      %%xmm5, %%xmm1                \n"
      "pshufb      %%xmm6, %%xmm2                \n"
      "movdqu      %%xmm0,(%2)                   \n"
      "movdqu      %%xmm1,16(%2)                 \n"
      "movdqu      %%xmm2,32(%2)                 \n"
      "lea         48(%2),%2                     \n"
      "sub         $16,%3                        \n"  // 16 pixels per loop
      "jg          1b                            \n"
      : "+r"(src_y),            // %0
        "+r"(src_vu),           // %1
        "+r"(dst_yuv24),        // %2
        "+r"(width)             // %3
      : "r"(&kYUV24Shuffle[0])  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}

// Convert biplanar NV21 to packed YUV24
// NV21 has VU in memory for chroma.
// YUV24 is VUY in memory
void NV21ToYUV24Row_AVX2(const uint8_t* src_y,
                         const uint8_t* src_vu,
                         uint8_t* dst_yuv24,
                         int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "vbroadcastf128 (%4),%%ymm4                \n"  // 3 shuffler constants
      "vbroadcastf128 16(%4),%%ymm5              \n"
      "vbroadcastf128 32(%4),%%ymm6              \n"

      "1:          \n"
      "vmovdqu     (%0),%%ymm2                   \n"  // load 32 Y values
      "vmovdqu     (%0,%1),%%ymm3                \n"  // load 16 VU values
      "lea         32(%0),%0                     \n"
      "vshufps     $0x44,%%ymm3,%%ymm2,%%ymm0    \n"  // Y 0..7,  UV 0..3
      "vshufps     $0x99,%%ymm3,%%ymm2,%%ymm1    \n"  // Y 4..11, UV 2..5
      "vshufps     $0xee,%%ymm3,%%ymm2,%%ymm2    \n"  // Y 8..15, UV 4..7
      "vpshufb     %%ymm4,%%ymm0,%%ymm0          \n"  // weave into YUV24
      "vpshufb     %%ymm5,%%ymm1,%%ymm1          \n"
      "vpshufb     %%ymm6,%%ymm2,%%ymm2          \n"
      "vperm2i128  $0x20,%%ymm1,%%ymm0,%%ymm3    \n"
      "vperm2i128  $0x30,%%ymm0,%%ymm2,%%ymm0    \n"
      "vperm2i128  $0x31,%%ymm2,%%ymm1,%%ymm1    \n"
      "vmovdqu     %%ymm3,(%2)                   \n"
      "vmovdqu     %%ymm0,32(%2)                 \n"
      "vmovdqu     %%ymm1,64(%2)                 \n"
      "lea         96(%2),%2                     \n"
      "sub         $32,%3                        \n"  // 32 pixels per loop
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_y),            // %0
        "+r"(src_vu),           // %1
        "+r"(dst_yuv24),        // %2
        "+r"(width)             // %3
      : "r"(&kYUV24Shuffle[0])  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}

#ifdef HAS_NV21ToYUV24ROW_AVX512
// The following VMBI VEX256 code tests okay with the intelsde emulator.
static const lvec8 kYUV24Perm[3] = {
    {32, 33, 0,  32, 33, 1,  34, 35, 2,  34, 35, 3,  36, 37, 4,  36,
     37, 5,  38, 39, 6,  38, 39, 7,  40, 41, 8,  40, 41, 9,  42, 43},
    {10, 42, 43, 11, 44, 45, 12, 44, 45, 13, 46, 47, 14, 46, 47, 15,
     48, 49, 16, 48, 49, 17, 50, 51, 18, 50, 51, 19, 52, 53, 20, 52},
    {53, 21, 54, 55, 22, 54, 55, 23, 56, 57, 24, 56, 57, 25, 58, 59,
     26, 58, 59, 27, 60, 61, 28, 60, 61, 29, 62, 63, 30, 62, 63, 31}};

void NV21ToYUV24Row_AVX512(const uint8_t* src_y,
                           const uint8_t* src_vu,
                           uint8_t* dst_yuv24,
                           int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "vmovdqa     (%4),%%ymm4                   \n"  // 3 shuffler constants
      "vmovdqa     32(%4),%%ymm5                 \n"
      "vmovdqa     64(%4),%%ymm6                 \n" LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm2                   \n"  // load 32 Y values
      "vmovdqu     (%0,%1),%%ymm3                \n"  // load 16 VU values
      "lea         32(%0),%0                     \n"
      "vmovdqa     %%ymm2, %%ymm0                \n"
      "vmovdqa     %%ymm2, %%ymm1                \n"
      "vpermt2b    %%ymm3,%%ymm4,%%ymm0          \n"
      "vpermt2b    %%ymm3,%%ymm5,%%ymm1          \n"
      "vpermt2b    %%ymm3,%%ymm6,%%ymm2          \n"
      "vmovdqu     %%ymm0,(%2)                   \n"
      "vmovdqu     %%ymm1,32(%2)                 \n"
      "vmovdqu     %%ymm2,64(%2)                 \n"
      "lea         96(%2),%2                     \n"
      "sub         $32,%3                        \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_y),         // %0
        "+r"(src_vu),        // %1
        "+r"(dst_yuv24),     // %2
        "+r"(width)          // %3
      : "r"(&kYUV24Perm[0])  // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6");
}

#endif  // HAS_NV21ToYUV24ROW_AVX512

#ifdef HAS_SWAPUVROW_SSSE3

// Shuffle table for reversing the bytes.
static const uvec8 kShuffleUVToVU = {1u, 0u, 3u,  2u,  5u,  4u,  7u,  6u,
                                     9u, 8u, 11u, 10u, 13u, 12u, 15u, 14u};

// Convert UV plane of NV12 to VU of NV21.
void SwapUVRow_SSSE3(const uint8_t* src_uv, uint8_t* dst_vu, int width) {
      asm volatile("movdqu      %3,%%xmm5                     \n"

               LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"
      "movdqu      0x10(%0),%%xmm1               \n"
      "lea         0x20(%0),%0                   \n"
      "pshufb      %%xmm5,%%xmm0                 \n"
      "pshufb      %%xmm5,%%xmm1                 \n"
      "movdqu      %%xmm0,(%1)                   \n"
      "movdqu      %%xmm1,0x10(%1)               \n"
      "lea         0x20(%1),%1                   \n"
      "sub         $0x10,%2                      \n"
      "jg          1b                            \n"
               : "+r"(src_uv),        // %0
                 "+r"(dst_vu),        // %1
                 "+r"(width)          // %2
               : "m"(kShuffleUVToVU)  // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm5");
}
#endif  // HAS_SWAPUVROW_SSSE3

#ifdef HAS_SWAPUVROW_AVX2
void SwapUVRow_AVX2(const uint8_t* src_uv, uint8_t* dst_vu, int width) {
      asm volatile("vbroadcastf128 %3,%%ymm5                  \n"

               LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vmovdqu     0x20(%0),%%ymm1               \n"
      "lea         0x40(%0),%0                   \n"
      "vpshufb     %%ymm5,%%ymm0,%%ymm0          \n"
      "vpshufb     %%ymm5,%%ymm1,%%ymm1          \n"
      "vmovdqu     %%ymm0,(%1)                   \n"
      "vmovdqu     %%ymm1,0x20(%1)               \n"
      "lea         0x40(%1),%1                   \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
               : "+r"(src_uv),        // %0
                 "+r"(dst_vu),        // %1
                 "+r"(width)          // %2
               : "m"(kShuffleUVToVU)  // %3
               : "memory", "cc", "xmm0", "xmm1", "xmm5");
}
#endif  // HAS_SWAPUVROW_AVX2

void HalfMergeUVRow_SSSE3(const uint8_t* src_u,
                          int src_stride_u,
                          const uint8_t* src_v,
                          int src_stride_v,
                          uint8_t* dst_uv,
                          int width) {
  asm volatile(
      "pcmpeqb     %%xmm4,%%xmm4                 \n"  // 0x0101
      "pabsb       %%xmm4,%%xmm4                 \n"
      "pxor        %%xmm5,%%xmm5                 \n"

      LABELALIGN
      "1:          \n"
      "movdqu      (%0),%%xmm0                   \n"  // load 16 U values
      "movdqu      (%1),%%xmm1                   \n"  // load 16 V values
      "movdqu      0(%0,%4,1),%%xmm2             \n"  // 16 from next row
      "movdqu      0(%1,%5,1),%%xmm3             \n"
      "lea         0x10(%0),%0                   \n"
      "pmaddubsw   %%xmm4,%%xmm0                 \n"  // half size
      "pmaddubsw   %%xmm4,%%xmm1                 \n"
      "pmaddubsw   %%xmm4,%%xmm2                 \n"
      "pmaddubsw   %%xmm4,%%xmm3                 \n"
      "lea         0x10(%1),%1                   \n"
      "paddw       %%xmm2,%%xmm0                 \n"
      "paddw       %%xmm3,%%xmm1                 \n"
      "psrlw       $0x1,%%xmm0                   \n"
      "psrlw       $0x1,%%xmm1                   \n"
      "pavgw       %%xmm5,%%xmm0                 \n"
      "pavgw       %%xmm5,%%xmm1                 \n"
      "packuswb    %%xmm0,%%xmm0                 \n"
      "packuswb    %%xmm1,%%xmm1                 \n"
      "punpcklbw   %%xmm1,%%xmm0                 \n"
      "movdqu      %%xmm0,(%2)                   \n"  // store 8 UV pixels
      "lea         0x10(%2),%2                   \n"
      "sub         $0x10,%3                      \n"  // 16 src pixels per loop
      "jg          1b                            \n"
      : "+r"(src_u),                    // %0
        "+r"(src_v),                    // %1
        "+r"(dst_uv),                   // %2
        "+r"(width)                     // %3
      : "r"((intptr_t)(src_stride_u)),  // %4
        "r"((intptr_t)(src_stride_v))   // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

void HalfMergeUVRow_AVX2(const uint8_t* src_u,
                         int src_stride_u,
                         const uint8_t* src_v,
                         int src_stride_v,
                         uint8_t* dst_uv,
                         int width) {
  asm volatile(
      "vpcmpeqb    %%ymm4,%%ymm4,%%ymm4          \n"
      "vpabsb      %%ymm4,%%ymm4                 \n"
      "vpxor       %%ymm5,%%ymm5,%%ymm5          \n"

      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"  // load 32 U values
      "vmovdqu     (%1),%%ymm1                   \n"  // load 32 V values
      "vmovdqu     0(%0,%4,1),%%ymm2             \n"  // 32 from next row
      "vmovdqu     0(%1,%5,1),%%ymm3             \n"
      "lea         0x20(%0),%0                   \n"
      "vpmaddubsw  %%ymm4,%%ymm0,%%ymm0          \n"  // half size
      "vpmaddubsw  %%ymm4,%%ymm1,%%ymm1          \n"
      "vpmaddubsw  %%ymm4,%%ymm2,%%ymm2          \n"
      "vpmaddubsw  %%ymm4,%%ymm3,%%ymm3          \n"
      "lea         0x20(%1),%1                   \n"
      "vpaddw      %%ymm2,%%ymm0,%%ymm0          \n"
      "vpaddw      %%ymm3,%%ymm1,%%ymm1          \n"
      "vpsrlw      $0x1,%%ymm0,%%ymm0            \n"
      "vpsrlw      $0x1,%%ymm1,%%ymm1            \n"
      "vpavgw      %%ymm5,%%ymm0,%%ymm0          \n"
      "vpavgw      %%ymm5,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm0,%%ymm0,%%ymm0          \n"
      "vpackuswb   %%ymm1,%%ymm1,%%ymm1          \n"
      "vpunpcklbw  %%ymm1,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%2)                   \n"  // store 16 UV pixels
      "lea         0x20(%2),%2                   \n"
      "sub         $0x20,%3                      \n"  // 32 src pixels per loop
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_u),                    // %0
        "+r"(src_v),                    // %1
        "+r"(dst_uv),                   // %2
        "+r"(width)                     // %3
      : "r"((intptr_t)(src_stride_u)),  // %4
        "r"((intptr_t)(src_stride_v))   // %5
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5");
}

void ClampFloatToZero_SSE2(const float* src_x, float* dst_y, int width) {
  asm volatile(
      "pxor        %%xmm1,%%xmm1                 \n"

      LABELALIGN
      "1:          \n"
      "movd        (%0),%%xmm0                   \n"  // load float
      "maxss       %%xmm1, %%xmm0                \n"  // clamp to zero
      "add         4, %0                         \n"
      "movd        %%xmm0, (%1)                  \n"  // store float
      "add         4, %1                         \n"
      "sub         $0x4,%2                       \n"  // 1 float per loop
      "jg          1b                            \n"
      : "+r"(src_x),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      :
      : "memory", "cc", "xmm0", "xmm1");
}

#ifdef HAS_CONVERT16TO8ROW_AVX2
void Convert8To8Row_AVX2(const uint8_t* src_y,
                         uint8_t* dst_y,
                         int scale,
                         int bias,
                         int width) {
  asm volatile(
      "sub         %0,%1                         \n"
      "vmovd       %3,%%xmm2                     \n"
      "vmovd       %4,%%xmm3                     \n"
      "vpbroadcastw %%xmm2,%%ymm2                \n"
      "vpbroadcastb %%xmm3,%%ymm3                \n"
      "vpxor       %%ymm4,%%ymm4,%%ymm4          \n"
      "vpsllw      $8,%%ymm2,%%ymm2              \n"

      // 32 pixels per loop.
      LABELALIGN
      "1:          \n"
      "vmovdqu     (%0),%%ymm0                   \n"
      "vpunpckhbw  %%ymm4,%%ymm0,%%ymm1          \n"  // mutates
      "vpunpcklbw  %%ymm4,%%ymm0,%%ymm0          \n"
      "vpmulhuw    %%ymm2,%%ymm0,%%ymm0          \n"
      "vpmulhuw    %%ymm2,%%ymm1,%%ymm1          \n"
      "vpackuswb   %%ymm1,%%ymm0,%%ymm0          \n"  // unmutates
      "vpaddb      %%ymm3,%%ymm0,%%ymm0          \n"
      "vmovdqu     %%ymm0,(%0,%1)                \n"
      "add         $0x20,%0                      \n"
      "sub         $0x20,%2                      \n"
      "jg          1b                            \n"
      "vzeroupper  \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale),   // %3
        "r"(bias)     // %4
      : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4");
}
#endif  // HAS_CONVERT16TO8ROW_AVX2

#endif  // defined(__x86_64__) || defined(__i386__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
