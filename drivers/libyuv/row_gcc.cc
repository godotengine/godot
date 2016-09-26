// VERSION 2
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
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(__x86_64__) || (defined(__i386__) && !defined(_MSC_VER)))

#if defined(HAS_ARGBTOYROW_SSSE3) || defined(HAS_ARGBGRAYROW_SSSE3)

// Constants for ARGB
static vec8 kARGBToY = {
  13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33, 0
};

// JPeg full range.
static vec8 kARGBToYJ = {
  15, 75, 38, 0, 15, 75, 38, 0, 15, 75, 38, 0, 15, 75, 38, 0
};
#endif  // defined(HAS_ARGBTOYROW_SSSE3) || defined(HAS_ARGBGRAYROW_SSSE3)

#if defined(HAS_ARGBTOYROW_SSSE3) || defined(HAS_I422TOARGBROW_SSSE3)

static vec8 kARGBToU = {
  112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38, 0
};

static vec8 kARGBToUJ = {
  127, -84, -43, 0, 127, -84, -43, 0, 127, -84, -43, 0, 127, -84, -43, 0
};

static vec8 kARGBToV = {
  -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0,
};

static vec8 kARGBToVJ = {
  -20, -107, 127, 0, -20, -107, 127, 0, -20, -107, 127, 0, -20, -107, 127, 0
};

// Constants for BGRA
static vec8 kBGRAToY = {
  0, 33, 65, 13, 0, 33, 65, 13, 0, 33, 65, 13, 0, 33, 65, 13
};

static vec8 kBGRAToU = {
  0, -38, -74, 112, 0, -38, -74, 112, 0, -38, -74, 112, 0, -38, -74, 112
};

static vec8 kBGRAToV = {
  0, 112, -94, -18, 0, 112, -94, -18, 0, 112, -94, -18, 0, 112, -94, -18
};

// Constants for ABGR
static vec8 kABGRToY = {
  33, 65, 13, 0, 33, 65, 13, 0, 33, 65, 13, 0, 33, 65, 13, 0
};

static vec8 kABGRToU = {
  -38, -74, 112, 0, -38, -74, 112, 0, -38, -74, 112, 0, -38, -74, 112, 0
};

static vec8 kABGRToV = {
  112, -94, -18, 0, 112, -94, -18, 0, 112, -94, -18, 0, 112, -94, -18, 0
};

// Constants for RGBA.
static vec8 kRGBAToY = {
  0, 13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33, 0, 13, 65, 33
};

static vec8 kRGBAToU = {
  0, 112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38, 0, 112, -74, -38
};

static vec8 kRGBAToV = {
  0, -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112
};

static uvec8 kAddY16 = {
  16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u
};

// 7 bit fixed point 0.5.
static vec16 kAddYJ64 = {
  64, 64, 64, 64, 64, 64, 64, 64
};

static uvec8 kAddUV128 = {
  128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u,
  128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u
};

static uvec16 kAddUVJ128 = {
  0x8080u, 0x8080u, 0x8080u, 0x8080u, 0x8080u, 0x8080u, 0x8080u, 0x8080u
};
#endif  // defined(HAS_ARGBTOYROW_SSSE3) || defined(HAS_I422TOARGBROW_SSSE3)

#ifdef HAS_RGB24TOARGBROW_SSSE3

// Shuffle table for converting RGB24 to ARGB.
static uvec8 kShuffleMaskRGB24ToARGB = {
  0u, 1u, 2u, 12u, 3u, 4u, 5u, 13u, 6u, 7u, 8u, 14u, 9u, 10u, 11u, 15u
};

// Shuffle table for converting RAW to ARGB.
static uvec8 kShuffleMaskRAWToARGB = {
  2u, 1u, 0u, 12u, 5u, 4u, 3u, 13u, 8u, 7u, 6u, 14u, 11u, 10u, 9u, 15u
};

// Shuffle table for converting RAW to RGB24.  First 8.
static const uvec8 kShuffleMaskRAWToRGB24_0 = {
  2u, 1u, 0u, 5u, 4u, 3u, 8u, 7u,
  128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u
};

// Shuffle table for converting RAW to RGB24.  Middle 8.
static const uvec8 kShuffleMaskRAWToRGB24_1 = {
  2u, 7u, 6u, 5u, 10u, 9u, 8u, 13u,
  128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u
};

// Shuffle table for converting RAW to RGB24.  Last 8.
static const uvec8 kShuffleMaskRAWToRGB24_2 = {
  8u, 7u, 12u, 11u, 10u, 15u, 14u, 13u,
  128u, 128u, 128u, 128u, 128u, 128u, 128u, 128u
};

// Shuffle table for converting ARGB to RGB24.
static uvec8 kShuffleMaskARGBToRGB24 = {
  0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 10u, 12u, 13u, 14u, 128u, 128u, 128u, 128u
};

// Shuffle table for converting ARGB to RAW.
static uvec8 kShuffleMaskARGBToRAW = {
  2u, 1u, 0u, 6u, 5u, 4u, 10u, 9u, 8u, 14u, 13u, 12u, 128u, 128u, 128u, 128u
};

// Shuffle table for converting ARGBToRGB24 for I422ToRGB24.  First 8 + next 4
static uvec8 kShuffleMaskARGBToRGB24_0 = {
  0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 128u, 128u, 128u, 128u, 10u, 12u, 13u, 14u
};

// YUY2 shuf 16 Y to 32 Y.
static const lvec8 kShuffleYUY2Y = {
  0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14,
  0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14
};

// YUY2 shuf 8 UV to 16 UV.
static const lvec8 kShuffleYUY2UV = {
  1, 3, 1, 3, 5, 7, 5, 7, 9, 11, 9, 11, 13, 15, 13, 15,
  1, 3, 1, 3, 5, 7, 5, 7, 9, 11, 9, 11, 13, 15, 13, 15
};

// UYVY shuf 16 Y to 32 Y.
static const lvec8 kShuffleUYVYY = {
  1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15,
  1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15
};

// UYVY shuf 8 UV to 16 UV.
static const lvec8 kShuffleUYVYUV = {
  0, 2, 0, 2, 4, 6, 4, 6, 8, 10, 8, 10, 12, 14, 12, 14,
  0, 2, 0, 2, 4, 6, 4, 6, 8, 10, 8, 10, 12, 14, 12, 14
};

// NV21 shuf 8 VU to 16 UV.
static const lvec8 kShuffleNV21 = {
  1, 0, 1, 0, 3, 2, 3, 2, 5, 4, 5, 4, 7, 6, 7, 6,
  1, 0, 1, 0, 3, 2, 3, 2, 5, 4, 5, 4, 7, 6, 7, 6,
};
#endif  // HAS_RGB24TOARGBROW_SSSE3

#ifdef HAS_J400TOARGBROW_SSE2
void J400ToARGBRow_SSE2(const uint8* src_y, uint8* dst_argb, int width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "pslld     $0x18,%%xmm5                    \n"
    LABELALIGN
  "1:                                          \n"
    "movq      " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x8,0) ",%0            \n"
    "punpcklbw %%xmm0,%%xmm0                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklwd %%xmm0,%%xmm0                   \n"
    "punpckhwd %%xmm1,%%xmm1                   \n"
    "por       %%xmm5,%%xmm0                   \n"
    "por       %%xmm5,%%xmm1                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_y),     // %0
    "+r"(dst_argb),  // %1
    "+r"(width)        // %2
  :: "memory", "cc", "xmm0", "xmm1", "xmm5"
  );
}
#endif  // HAS_J400TOARGBROW_SSE2

#ifdef HAS_RGB24TOARGBROW_SSSE3
void RGB24ToARGBRow_SSSE3(const uint8* src_rgb24, uint8* dst_argb, int width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"  // generate mask 0xff000000
    "pslld     $0x18,%%xmm5                    \n"
    "movdqa    %3,%%xmm4                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm3   \n"
    "lea       " MEMLEA(0x30,0) ",%0           \n"
    "movdqa    %%xmm3,%%xmm2                   \n"
    "palignr   $0x8,%%xmm1,%%xmm2              \n"
    "pshufb    %%xmm4,%%xmm2                   \n"
    "por       %%xmm5,%%xmm2                   \n"
    "palignr   $0xc,%%xmm0,%%xmm1              \n"
    "pshufb    %%xmm4,%%xmm0                   \n"
    "movdqu    %%xmm2," MEMACCESS2(0x20,1) "   \n"
    "por       %%xmm5,%%xmm0                   \n"
    "pshufb    %%xmm4,%%xmm1                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "por       %%xmm5,%%xmm1                   \n"
    "palignr   $0x4,%%xmm3,%%xmm3              \n"
    "pshufb    %%xmm4,%%xmm3                   \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "por       %%xmm5,%%xmm3                   \n"
    "movdqu    %%xmm3," MEMACCESS2(0x30,1) "   \n"
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_rgb24),  // %0
    "+r"(dst_argb),  // %1
    "+r"(width)        // %2
  : "m"(kShuffleMaskRGB24ToARGB)  // %3
  : "memory", "cc" , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void RAWToARGBRow_SSSE3(const uint8* src_raw, uint8* dst_argb, int width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"  // generate mask 0xff000000
    "pslld     $0x18,%%xmm5                    \n"
    "movdqa    %3,%%xmm4                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm3   \n"
    "lea       " MEMLEA(0x30,0) ",%0           \n"
    "movdqa    %%xmm3,%%xmm2                   \n"
    "palignr   $0x8,%%xmm1,%%xmm2              \n"
    "pshufb    %%xmm4,%%xmm2                   \n"
    "por       %%xmm5,%%xmm2                   \n"
    "palignr   $0xc,%%xmm0,%%xmm1              \n"
    "pshufb    %%xmm4,%%xmm0                   \n"
    "movdqu    %%xmm2," MEMACCESS2(0x20,1) "   \n"
    "por       %%xmm5,%%xmm0                   \n"
    "pshufb    %%xmm4,%%xmm1                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "por       %%xmm5,%%xmm1                   \n"
    "palignr   $0x4,%%xmm3,%%xmm3              \n"
    "pshufb    %%xmm4,%%xmm3                   \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "por       %%xmm5,%%xmm3                   \n"
    "movdqu    %%xmm3," MEMACCESS2(0x30,1) "   \n"
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_raw),   // %0
    "+r"(dst_argb),  // %1
    "+r"(width)        // %2
  : "m"(kShuffleMaskRAWToARGB)  // %3
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void RAWToRGB24Row_SSSE3(const uint8* src_raw, uint8* dst_rgb24, int width) {
  asm volatile (
   "movdqa     %3,%%xmm3                       \n"
   "movdqa     %4,%%xmm4                       \n"
   "movdqa     %5,%%xmm5                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x4,0) ",%%xmm1    \n"
    "movdqu    " MEMACCESS2(0x8,0) ",%%xmm2    \n"
    "lea       " MEMLEA(0x18,0) ",%0           \n"
    "pshufb    %%xmm3,%%xmm0                   \n"
    "pshufb    %%xmm4,%%xmm1                   \n"
    "pshufb    %%xmm5,%%xmm2                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "movq      %%xmm1," MEMACCESS2(0x8,1) "    \n"
    "movq      %%xmm2," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x18,1) ",%1           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_raw),    // %0
    "+r"(dst_rgb24),  // %1
    "+r"(width)       // %2
  : "m"(kShuffleMaskRAWToRGB24_0),  // %3
    "m"(kShuffleMaskRAWToRGB24_1),  // %4
    "m"(kShuffleMaskRAWToRGB24_2)   // %5
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void RGB565ToARGBRow_SSE2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "mov       $0x1080108,%%eax                \n"
    "movd      %%eax,%%xmm5                    \n"
    "pshufd    $0x0,%%xmm5,%%xmm5              \n"
    "mov       $0x20802080,%%eax               \n"
    "movd      %%eax,%%xmm6                    \n"
    "pshufd    $0x0,%%xmm6,%%xmm6              \n"
    "pcmpeqb   %%xmm3,%%xmm3                   \n"
    "psllw     $0xb,%%xmm3                     \n"
    "pcmpeqb   %%xmm4,%%xmm4                   \n"
    "psllw     $0xa,%%xmm4                     \n"
    "psrlw     $0x5,%%xmm4                     \n"
    "pcmpeqb   %%xmm7,%%xmm7                   \n"
    "psllw     $0x8,%%xmm7                     \n"
    "sub       %0,%1                           \n"
    "sub       %0,%1                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "pand      %%xmm3,%%xmm1                   \n"
    "psllw     $0xb,%%xmm2                     \n"
    "pmulhuw   %%xmm5,%%xmm1                   \n"
    "pmulhuw   %%xmm5,%%xmm2                   \n"
    "psllw     $0x8,%%xmm1                     \n"
    "por       %%xmm2,%%xmm1                   \n"
    "pand      %%xmm4,%%xmm0                   \n"
    "pmulhuw   %%xmm6,%%xmm0                   \n"
    "por       %%xmm7,%%xmm0                   \n"
    "movdqa    %%xmm1,%%xmm2                   \n"
    "punpcklbw %%xmm0,%%xmm1                   \n"
    "punpckhbw %%xmm0,%%xmm2                   \n"
    MEMOPMEM(movdqu,xmm1,0x00,1,0,2)           //  movdqu  %%xmm1,(%1,%0,2)
    MEMOPMEM(movdqu,xmm2,0x10,1,0,2)           //  movdqu  %%xmm2,0x10(%1,%0,2)
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  :
  : "memory", "cc", "eax", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

void ARGB1555ToARGBRow_SSE2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "mov       $0x1080108,%%eax                \n"
    "movd      %%eax,%%xmm5                    \n"
    "pshufd    $0x0,%%xmm5,%%xmm5              \n"
    "mov       $0x42004200,%%eax               \n"
    "movd      %%eax,%%xmm6                    \n"
    "pshufd    $0x0,%%xmm6,%%xmm6              \n"
    "pcmpeqb   %%xmm3,%%xmm3                   \n"
    "psllw     $0xb,%%xmm3                     \n"
    "movdqa    %%xmm3,%%xmm4                   \n"
    "psrlw     $0x6,%%xmm4                     \n"
    "pcmpeqb   %%xmm7,%%xmm7                   \n"
    "psllw     $0x8,%%xmm7                     \n"
    "sub       %0,%1                           \n"
    "sub       %0,%1                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "psllw     $0x1,%%xmm1                     \n"
    "psllw     $0xb,%%xmm2                     \n"
    "pand      %%xmm3,%%xmm1                   \n"
    "pmulhuw   %%xmm5,%%xmm2                   \n"
    "pmulhuw   %%xmm5,%%xmm1                   \n"
    "psllw     $0x8,%%xmm1                     \n"
    "por       %%xmm2,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "pand      %%xmm4,%%xmm0                   \n"
    "psraw     $0x8,%%xmm2                     \n"
    "pmulhuw   %%xmm6,%%xmm0                   \n"
    "pand      %%xmm7,%%xmm2                   \n"
    "por       %%xmm2,%%xmm0                   \n"
    "movdqa    %%xmm1,%%xmm2                   \n"
    "punpcklbw %%xmm0,%%xmm1                   \n"
    "punpckhbw %%xmm0,%%xmm2                   \n"
    MEMOPMEM(movdqu,xmm1,0x00,1,0,2)           //  movdqu  %%xmm1,(%1,%0,2)
    MEMOPMEM(movdqu,xmm2,0x10,1,0,2)           //  movdqu  %%xmm2,0x10(%1,%0,2)
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  :
  : "memory", "cc", "eax", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

void ARGB4444ToARGBRow_SSE2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "mov       $0xf0f0f0f,%%eax                \n"
    "movd      %%eax,%%xmm4                    \n"
    "pshufd    $0x0,%%xmm4,%%xmm4              \n"
    "movdqa    %%xmm4,%%xmm5                   \n"
    "pslld     $0x4,%%xmm5                     \n"
    "sub       %0,%1                           \n"
    "sub       %0,%1                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "pand      %%xmm4,%%xmm0                   \n"
    "pand      %%xmm5,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm2,%%xmm3                   \n"
    "psllw     $0x4,%%xmm1                     \n"
    "psrlw     $0x4,%%xmm3                     \n"
    "por       %%xmm1,%%xmm0                   \n"
    "por       %%xmm3,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklbw %%xmm2,%%xmm0                   \n"
    "punpckhbw %%xmm2,%%xmm1                   \n"
    MEMOPMEM(movdqu,xmm0,0x00,1,0,2)           //  movdqu  %%xmm0,(%1,%0,2)
    MEMOPMEM(movdqu,xmm1,0x10,1,0,2)           //  movdqu  %%xmm1,0x10(%1,%0,2)
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  :
  : "memory", "cc", "eax", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void ARGBToRGB24Row_SSSE3(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "movdqa    %3,%%xmm6                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "pshufb    %%xmm6,%%xmm0                   \n"
    "pshufb    %%xmm6,%%xmm1                   \n"
    "pshufb    %%xmm6,%%xmm2                   \n"
    "pshufb    %%xmm6,%%xmm3                   \n"
    "movdqa    %%xmm1,%%xmm4                   \n"
    "psrldq    $0x4,%%xmm1                     \n"
    "pslldq    $0xc,%%xmm4                     \n"
    "movdqa    %%xmm2,%%xmm5                   \n"
    "por       %%xmm4,%%xmm0                   \n"
    "pslldq    $0x8,%%xmm5                     \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "por       %%xmm5,%%xmm1                   \n"
    "psrldq    $0x8,%%xmm2                     \n"
    "pslldq    $0x4,%%xmm3                     \n"
    "por       %%xmm3,%%xmm2                   \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "movdqu    %%xmm2," MEMACCESS2(0x20,1) "   \n"
    "lea       " MEMLEA(0x30,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  : "m"(kShuffleMaskARGBToRGB24)  // %3
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}

void ARGBToRAWRow_SSSE3(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "movdqa    %3,%%xmm6                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "pshufb    %%xmm6,%%xmm0                   \n"
    "pshufb    %%xmm6,%%xmm1                   \n"
    "pshufb    %%xmm6,%%xmm2                   \n"
    "pshufb    %%xmm6,%%xmm3                   \n"
    "movdqa    %%xmm1,%%xmm4                   \n"
    "psrldq    $0x4,%%xmm1                     \n"
    "pslldq    $0xc,%%xmm4                     \n"
    "movdqa    %%xmm2,%%xmm5                   \n"
    "por       %%xmm4,%%xmm0                   \n"
    "pslldq    $0x8,%%xmm5                     \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "por       %%xmm5,%%xmm1                   \n"
    "psrldq    $0x8,%%xmm2                     \n"
    "pslldq    $0x4,%%xmm3                     \n"
    "por       %%xmm3,%%xmm2                   \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "movdqu    %%xmm2," MEMACCESS2(0x20,1) "   \n"
    "lea       " MEMLEA(0x30,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  : "m"(kShuffleMaskARGBToRAW)  // %3
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}

void ARGBToRGB565Row_SSE2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "pcmpeqb   %%xmm3,%%xmm3                   \n"
    "psrld     $0x1b,%%xmm3                    \n"
    "pcmpeqb   %%xmm4,%%xmm4                   \n"
    "psrld     $0x1a,%%xmm4                    \n"
    "pslld     $0x5,%%xmm4                     \n"
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "pslld     $0xb,%%xmm5                     \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "pslld     $0x8,%%xmm0                     \n"
    "psrld     $0x3,%%xmm1                     \n"
    "psrld     $0x5,%%xmm2                     \n"
    "psrad     $0x10,%%xmm0                    \n"
    "pand      %%xmm3,%%xmm1                   \n"
    "pand      %%xmm4,%%xmm2                   \n"
    "pand      %%xmm5,%%xmm0                   \n"
    "por       %%xmm2,%%xmm1                   \n"
    "por       %%xmm1,%%xmm0                   \n"
    "packssdw  %%xmm0,%%xmm0                   \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x4,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  :: "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void ARGBToRGB565DitherRow_SSE2(const uint8* src, uint8* dst,
                                const uint32 dither4, int width) {
  asm volatile (
    "movd       %3,%%xmm6                      \n"
    "punpcklbw  %%xmm6,%%xmm6                  \n"
    "movdqa     %%xmm6,%%xmm7                  \n"
    "punpcklwd  %%xmm6,%%xmm6                  \n"
    "punpckhwd  %%xmm7,%%xmm7                  \n"
    "pcmpeqb    %%xmm3,%%xmm3                  \n"
    "psrld      $0x1b,%%xmm3                   \n"
    "pcmpeqb    %%xmm4,%%xmm4                  \n"
    "psrld      $0x1a,%%xmm4                   \n"
    "pslld      $0x5,%%xmm4                    \n"
    "pcmpeqb    %%xmm5,%%xmm5                  \n"
    "pslld      $0xb,%%xmm5                    \n"

    LABELALIGN
  "1:                                          \n"
    "movdqu     (%0),%%xmm0                    \n"
    "paddusb    %%xmm6,%%xmm0                  \n"
    "movdqa     %%xmm0,%%xmm1                  \n"
    "movdqa     %%xmm0,%%xmm2                  \n"
    "pslld      $0x8,%%xmm0                    \n"
    "psrld      $0x3,%%xmm1                    \n"
    "psrld      $0x5,%%xmm2                    \n"
    "psrad      $0x10,%%xmm0                   \n"
    "pand       %%xmm3,%%xmm1                  \n"
    "pand       %%xmm4,%%xmm2                  \n"
    "pand       %%xmm5,%%xmm0                  \n"
    "por        %%xmm2,%%xmm1                  \n"
    "por        %%xmm1,%%xmm0                  \n"
    "packssdw   %%xmm0,%%xmm0                  \n"
    "lea        0x10(%0),%0                    \n"
    "movq       %%xmm0,(%1)                    \n"
    "lea        0x8(%1),%1                     \n"
    "sub        $0x4,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  : "m"(dither4) // %3
  : "memory", "cc",
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

#ifdef HAS_ARGBTORGB565DITHERROW_AVX2
void ARGBToRGB565DitherRow_AVX2(const uint8* src, uint8* dst,
                                const uint32 dither4, int width) {
  asm volatile (
    "vbroadcastss %3,%%xmm6                    \n"
    "vpunpcklbw %%xmm6,%%xmm6,%%xmm6           \n"
    "vpermq     $0xd8,%%ymm6,%%ymm6            \n"
    "vpunpcklwd %%ymm6,%%ymm6,%%ymm6           \n"
    "vpcmpeqb   %%ymm3,%%ymm3,%%ymm3           \n"
    "vpsrld     $0x1b,%%ymm3,%%ymm3            \n"
    "vpcmpeqb   %%ymm4,%%ymm4,%%ymm4           \n"
    "vpsrld     $0x1a,%%ymm4,%%ymm4            \n"
    "vpslld     $0x5,%%ymm4,%%ymm4             \n"
    "vpslld     $0xb,%%ymm3,%%ymm5             \n"

    LABELALIGN
  "1:                                          \n"
    "vmovdqu    (%0),%%ymm0                    \n"
    "vpaddusb   %%ymm6,%%ymm0,%%ymm0           \n"
    "vpsrld     $0x5,%%ymm0,%%ymm2             \n"
    "vpsrld     $0x3,%%ymm0,%%ymm1             \n"
    "vpsrld     $0x8,%%ymm0,%%ymm0             \n"
    "vpand      %%ymm4,%%ymm2,%%ymm2           \n"
    "vpand      %%ymm3,%%ymm1,%%ymm1           \n"
    "vpand      %%ymm5,%%ymm0,%%ymm0           \n"
    "vpor       %%ymm2,%%ymm1,%%ymm1           \n"
    "vpor       %%ymm1,%%ymm0,%%ymm0           \n"
    "vpackusdw  %%ymm0,%%ymm0,%%ymm0           \n"
    "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
    "lea        0x20(%0),%0                    \n"
    "vmovdqu    %%xmm0,(%1)                    \n"
    "lea        0x10(%1),%1                    \n"
    "sub        $0x8,%2                        \n"
    "jg         1b                             \n"
    "vzeroupper                                \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  : "m"(dither4) // %3
  : "memory", "cc",
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBTORGB565DITHERROW_AVX2


void ARGBToARGB1555Row_SSE2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "pcmpeqb   %%xmm4,%%xmm4                   \n"
    "psrld     $0x1b,%%xmm4                    \n"
    "movdqa    %%xmm4,%%xmm5                   \n"
    "pslld     $0x5,%%xmm5                     \n"
    "movdqa    %%xmm4,%%xmm6                   \n"
    "pslld     $0xa,%%xmm6                     \n"
    "pcmpeqb   %%xmm7,%%xmm7                   \n"
    "pslld     $0xf,%%xmm7                     \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm3                   \n"
    "psrad     $0x10,%%xmm0                    \n"
    "psrld     $0x3,%%xmm1                     \n"
    "psrld     $0x6,%%xmm2                     \n"
    "psrld     $0x9,%%xmm3                     \n"
    "pand      %%xmm7,%%xmm0                   \n"
    "pand      %%xmm4,%%xmm1                   \n"
    "pand      %%xmm5,%%xmm2                   \n"
    "pand      %%xmm6,%%xmm3                   \n"
    "por       %%xmm1,%%xmm0                   \n"
    "por       %%xmm3,%%xmm2                   \n"
    "por       %%xmm2,%%xmm0                   \n"
    "packssdw  %%xmm0,%%xmm0                   \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x4,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  :: "memory", "cc",
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}

void ARGBToARGB4444Row_SSE2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "pcmpeqb   %%xmm4,%%xmm4                   \n"
    "psllw     $0xc,%%xmm4                     \n"
    "movdqa    %%xmm4,%%xmm3                   \n"
    "psrlw     $0x8,%%xmm3                     \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "pand      %%xmm3,%%xmm0                   \n"
    "pand      %%xmm4,%%xmm1                   \n"
    "psrlq     $0x4,%%xmm0                     \n"
    "psrlq     $0x8,%%xmm1                     \n"
    "por       %%xmm1,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x4,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(width)   // %2
  :: "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4"
  );
}
#endif  // HAS_RGB24TOARGBROW_SSSE3

#ifdef HAS_ARGBTOYROW_SSSE3
// Convert 16 ARGB pixels (64 bytes) to 16 Y values.
void ARGBToYRow_SSSE3(const uint8* src_argb, uint8* dst_y, int width) {
  asm volatile (
    "movdqa    %3,%%xmm4                       \n"
    "movdqa    %4,%%xmm5                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm4,%%xmm3                   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "phaddw    %%xmm1,%%xmm0                   \n"
    "phaddw    %%xmm3,%%xmm2                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "psrlw     $0x7,%%xmm2                     \n"
    "packuswb  %%xmm2,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  : "m"(kARGBToY),   // %3
    "m"(kAddY16)     // %4
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_ARGBTOYROW_SSSE3

#ifdef HAS_ARGBTOYJROW_SSSE3
// Convert 16 ARGB pixels (64 bytes) to 16 YJ values.
// Same as ARGBToYRow but different coefficients, no add 16, but do rounding.
void ARGBToYJRow_SSSE3(const uint8* src_argb, uint8* dst_y, int width) {
  asm volatile (
    "movdqa    %3,%%xmm4                       \n"
    "movdqa    %4,%%xmm5                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm4,%%xmm3                   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "phaddw    %%xmm1,%%xmm0                   \n"
    "phaddw    %%xmm3,%%xmm2                   \n"
    "paddw     %%xmm5,%%xmm0                   \n"
    "paddw     %%xmm5,%%xmm2                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "psrlw     $0x7,%%xmm2                     \n"
    "packuswb  %%xmm2,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  : "m"(kARGBToYJ),  // %3
    "m"(kAddYJ64)    // %4
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_ARGBTOYJROW_SSSE3

#ifdef HAS_ARGBTOYROW_AVX2
// vpermd for vphaddw + vpackuswb vpermd.
static const lvec32 kPermdARGBToY_AVX = {
  0, 4, 1, 5, 2, 6, 3, 7
};

// Convert 32 ARGB pixels (128 bytes) to 32 Y values.
void ARGBToYRow_AVX2(const uint8* src_argb, uint8* dst_y, int width) {
  asm volatile (
    "vbroadcastf128 %3,%%ymm4                  \n"
    "vbroadcastf128 %4,%%ymm5                  \n"
    "vmovdqu    %5,%%ymm6                      \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    " MEMACCESS(0) ",%%ymm0        \n"
    "vmovdqu    " MEMACCESS2(0x20,0) ",%%ymm1  \n"
    "vmovdqu    " MEMACCESS2(0x40,0) ",%%ymm2  \n"
    "vmovdqu    " MEMACCESS2(0x60,0) ",%%ymm3  \n"
    "vpmaddubsw %%ymm4,%%ymm0,%%ymm0           \n"
    "vpmaddubsw %%ymm4,%%ymm1,%%ymm1           \n"
    "vpmaddubsw %%ymm4,%%ymm2,%%ymm2           \n"
    "vpmaddubsw %%ymm4,%%ymm3,%%ymm3           \n"
    "lea       " MEMLEA(0x80,0) ",%0           \n"
    "vphaddw    %%ymm1,%%ymm0,%%ymm0           \n"  // mutates.
    "vphaddw    %%ymm3,%%ymm2,%%ymm2           \n"
    "vpsrlw     $0x7,%%ymm0,%%ymm0             \n"
    "vpsrlw     $0x7,%%ymm2,%%ymm2             \n"
    "vpackuswb  %%ymm2,%%ymm0,%%ymm0           \n"  // mutates.
    "vpermd     %%ymm0,%%ymm6,%%ymm0           \n"  // unmutate.
    "vpaddb     %%ymm5,%%ymm0,%%ymm0           \n"  // add 16 for Y
    "vmovdqu    %%ymm0," MEMACCESS(1) "        \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x20,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  : "m"(kARGBToY),   // %3
    "m"(kAddY16),    // %4
    "m"(kPermdARGBToY_AVX)  // %5
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}
#endif  // HAS_ARGBTOYROW_AVX2

#ifdef HAS_ARGBTOYJROW_AVX2
// Convert 32 ARGB pixels (128 bytes) to 32 Y values.
void ARGBToYJRow_AVX2(const uint8* src_argb, uint8* dst_y, int width) {
  asm volatile (
    "vbroadcastf128 %3,%%ymm4                  \n"
    "vbroadcastf128 %4,%%ymm5                  \n"
    "vmovdqu    %5,%%ymm6                      \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    " MEMACCESS(0) ",%%ymm0        \n"
    "vmovdqu    " MEMACCESS2(0x20,0) ",%%ymm1  \n"
    "vmovdqu    " MEMACCESS2(0x40,0) ",%%ymm2  \n"
    "vmovdqu    " MEMACCESS2(0x60,0) ",%%ymm3  \n"
    "vpmaddubsw %%ymm4,%%ymm0,%%ymm0           \n"
    "vpmaddubsw %%ymm4,%%ymm1,%%ymm1           \n"
    "vpmaddubsw %%ymm4,%%ymm2,%%ymm2           \n"
    "vpmaddubsw %%ymm4,%%ymm3,%%ymm3           \n"
    "lea       " MEMLEA(0x80,0) ",%0           \n"
    "vphaddw    %%ymm1,%%ymm0,%%ymm0           \n"  // mutates.
    "vphaddw    %%ymm3,%%ymm2,%%ymm2           \n"
    "vpaddw     %%ymm5,%%ymm0,%%ymm0           \n"  // Add .5 for rounding.
    "vpaddw     %%ymm5,%%ymm2,%%ymm2           \n"
    "vpsrlw     $0x7,%%ymm0,%%ymm0             \n"
    "vpsrlw     $0x7,%%ymm2,%%ymm2             \n"
    "vpackuswb  %%ymm2,%%ymm0,%%ymm0           \n"  // mutates.
    "vpermd     %%ymm0,%%ymm6,%%ymm0           \n"  // unmutate.
    "vmovdqu    %%ymm0," MEMACCESS(1) "        \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x20,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  : "m"(kARGBToYJ),   // %3
    "m"(kAddYJ64),    // %4
    "m"(kPermdARGBToY_AVX)  // %5
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}
#endif  // HAS_ARGBTOYJROW_AVX2

#ifdef HAS_ARGBTOUVROW_SSSE3
void ARGBToUVRow_SSSE3(const uint8* src_argb0, int src_stride_argb,
                       uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "movdqa    %5,%%xmm3                       \n"
    "movdqa    %6,%%xmm4                       \n"
    "movdqa    %7,%%xmm5                       \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,0,4,1,xmm7)            //  movdqu (%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqu,0x10,0,4,1,xmm7)            //  movdqu 0x10(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm1                   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    MEMOPREG(movdqu,0x20,0,4,1,xmm7)            //  movdqu 0x20(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm6   \n"
    MEMOPREG(movdqu,0x30,0,4,1,xmm7)            //  movdqu 0x30(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm6                   \n"

    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm7                   \n"
    "shufps    $0x88,%%xmm1,%%xmm0             \n"
    "shufps    $0xdd,%%xmm1,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqa    %%xmm2,%%xmm7                   \n"
    "shufps    $0x88,%%xmm6,%%xmm2             \n"
    "shufps    $0xdd,%%xmm6,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm2,%%xmm6                   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm3,%%xmm1                   \n"
    "pmaddubsw %%xmm3,%%xmm6                   \n"
    "phaddw    %%xmm2,%%xmm0                   \n"
    "phaddw    %%xmm6,%%xmm1                   \n"
    "psraw     $0x8,%%xmm0                     \n"
    "psraw     $0x8,%%xmm1                     \n"
    "packsswb  %%xmm1,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movlps    %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movhps,xmm0,0x00,1,2,1)           //  movhps    %%xmm0,(%1,%2,1)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_argb0),       // %0
    "+r"(dst_u),           // %1
    "+r"(dst_v),           // %2
    "+rm"(width)           // %3
  : "r"((intptr_t)(src_stride_argb)), // %4
    "m"(kARGBToV),  // %5
    "m"(kARGBToU),  // %6
    "m"(kAddUV128)  // %7
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBTOUVROW_SSSE3

#ifdef HAS_ARGBTOUVROW_AVX2
// vpshufb for vphaddw + vpackuswb packed to shorts.
static const lvec8 kShufARGBToUV_AVX = {
  0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
  0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15
};
void ARGBToUVRow_AVX2(const uint8* src_argb0, int src_stride_argb,
                      uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "vbroadcastf128 %5,%%ymm5                  \n"
    "vbroadcastf128 %6,%%ymm6                  \n"
    "vbroadcastf128 %7,%%ymm7                  \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    " MEMACCESS(0) ",%%ymm0        \n"
    "vmovdqu    " MEMACCESS2(0x20,0) ",%%ymm1  \n"
    "vmovdqu    " MEMACCESS2(0x40,0) ",%%ymm2  \n"
    "vmovdqu    " MEMACCESS2(0x60,0) ",%%ymm3  \n"
    VMEMOPREG(vpavgb,0x00,0,4,1,ymm0,ymm0)     // vpavgb (%0,%4,1),%%ymm0,%%ymm0
    VMEMOPREG(vpavgb,0x20,0,4,1,ymm1,ymm1)
    VMEMOPREG(vpavgb,0x40,0,4,1,ymm2,ymm2)
    VMEMOPREG(vpavgb,0x60,0,4,1,ymm3,ymm3)
    "lea       " MEMLEA(0x80,0) ",%0           \n"
    "vshufps    $0x88,%%ymm1,%%ymm0,%%ymm4     \n"
    "vshufps    $0xdd,%%ymm1,%%ymm0,%%ymm0     \n"
    "vpavgb     %%ymm4,%%ymm0,%%ymm0           \n"
    "vshufps    $0x88,%%ymm3,%%ymm2,%%ymm4     \n"
    "vshufps    $0xdd,%%ymm3,%%ymm2,%%ymm2     \n"
    "vpavgb     %%ymm4,%%ymm2,%%ymm2           \n"

    "vpmaddubsw %%ymm7,%%ymm0,%%ymm1           \n"
    "vpmaddubsw %%ymm7,%%ymm2,%%ymm3           \n"
    "vpmaddubsw %%ymm6,%%ymm0,%%ymm0           \n"
    "vpmaddubsw %%ymm6,%%ymm2,%%ymm2           \n"
    "vphaddw    %%ymm3,%%ymm1,%%ymm1           \n"
    "vphaddw    %%ymm2,%%ymm0,%%ymm0           \n"
    "vpsraw     $0x8,%%ymm1,%%ymm1             \n"
    "vpsraw     $0x8,%%ymm0,%%ymm0             \n"
    "vpacksswb  %%ymm0,%%ymm1,%%ymm0           \n"
    "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
    "vpshufb    %8,%%ymm0,%%ymm0               \n"
    "vpaddb     %%ymm5,%%ymm0,%%ymm0           \n"

    "vextractf128 $0x0,%%ymm0," MEMACCESS(1) " \n"
    VEXTOPMEM(vextractf128,1,ymm0,0x0,1,2,1) // vextractf128 $1,%%ymm0,(%1,%2,1)
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x20,%3                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb0),       // %0
    "+r"(dst_u),           // %1
    "+r"(dst_v),           // %2
    "+rm"(width)           // %3
  : "r"((intptr_t)(src_stride_argb)), // %4
    "m"(kAddUV128),  // %5
    "m"(kARGBToV),   // %6
    "m"(kARGBToU),   // %7
    "m"(kShufARGBToUV_AVX)  // %8
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBTOUVROW_AVX2

#ifdef HAS_ARGBTOUVJROW_AVX2
void ARGBToUVJRow_AVX2(const uint8* src_argb0, int src_stride_argb,
                       uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "vbroadcastf128 %5,%%ymm5                  \n"
    "vbroadcastf128 %6,%%ymm6                  \n"
    "vbroadcastf128 %7,%%ymm7                  \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    " MEMACCESS(0) ",%%ymm0        \n"
    "vmovdqu    " MEMACCESS2(0x20,0) ",%%ymm1  \n"
    "vmovdqu    " MEMACCESS2(0x40,0) ",%%ymm2  \n"
    "vmovdqu    " MEMACCESS2(0x60,0) ",%%ymm3  \n"
    VMEMOPREG(vpavgb,0x00,0,4,1,ymm0,ymm0)     // vpavgb (%0,%4,1),%%ymm0,%%ymm0
    VMEMOPREG(vpavgb,0x20,0,4,1,ymm1,ymm1)
    VMEMOPREG(vpavgb,0x40,0,4,1,ymm2,ymm2)
    VMEMOPREG(vpavgb,0x60,0,4,1,ymm3,ymm3)
    "lea       " MEMLEA(0x80,0) ",%0           \n"
    "vshufps    $0x88,%%ymm1,%%ymm0,%%ymm4     \n"
    "vshufps    $0xdd,%%ymm1,%%ymm0,%%ymm0     \n"
    "vpavgb     %%ymm4,%%ymm0,%%ymm0           \n"
    "vshufps    $0x88,%%ymm3,%%ymm2,%%ymm4     \n"
    "vshufps    $0xdd,%%ymm3,%%ymm2,%%ymm2     \n"
    "vpavgb     %%ymm4,%%ymm2,%%ymm2           \n"

    "vpmaddubsw %%ymm7,%%ymm0,%%ymm1           \n"
    "vpmaddubsw %%ymm7,%%ymm2,%%ymm3           \n"
    "vpmaddubsw %%ymm6,%%ymm0,%%ymm0           \n"
    "vpmaddubsw %%ymm6,%%ymm2,%%ymm2           \n"
    "vphaddw    %%ymm3,%%ymm1,%%ymm1           \n"
    "vphaddw    %%ymm2,%%ymm0,%%ymm0           \n"
    "vpaddw     %%ymm5,%%ymm0,%%ymm0           \n"
    "vpaddw     %%ymm5,%%ymm1,%%ymm1           \n"
    "vpsraw     $0x8,%%ymm1,%%ymm1             \n"
    "vpsraw     $0x8,%%ymm0,%%ymm0             \n"
    "vpacksswb  %%ymm0,%%ymm1,%%ymm0           \n"
    "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
    "vpshufb    %8,%%ymm0,%%ymm0               \n"

    "vextractf128 $0x0,%%ymm0," MEMACCESS(1) " \n"
    VEXTOPMEM(vextractf128,1,ymm0,0x0,1,2,1) // vextractf128 $1,%%ymm0,(%1,%2,1)
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x20,%3                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb0),       // %0
    "+r"(dst_u),           // %1
    "+r"(dst_v),           // %2
    "+rm"(width)           // %3
  : "r"((intptr_t)(src_stride_argb)), // %4
    "m"(kAddUVJ128),  // %5
    "m"(kARGBToVJ),  // %6
    "m"(kARGBToUJ),  // %7
    "m"(kShufARGBToUV_AVX)  // %8
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBTOUVJROW_AVX2

#ifdef HAS_ARGBTOUVJROW_SSSE3
void ARGBToUVJRow_SSSE3(const uint8* src_argb0, int src_stride_argb,
                        uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "movdqa    %5,%%xmm3                       \n"
    "movdqa    %6,%%xmm4                       \n"
    "movdqa    %7,%%xmm5                       \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,0,4,1,xmm7)            //  movdqu (%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqu,0x10,0,4,1,xmm7)            //  movdqu 0x10(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm1                   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    MEMOPREG(movdqu,0x20,0,4,1,xmm7)            //  movdqu 0x20(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm6   \n"
    MEMOPREG(movdqu,0x30,0,4,1,xmm7)            //  movdqu 0x30(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm6                   \n"

    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm7                   \n"
    "shufps    $0x88,%%xmm1,%%xmm0             \n"
    "shufps    $0xdd,%%xmm1,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqa    %%xmm2,%%xmm7                   \n"
    "shufps    $0x88,%%xmm6,%%xmm2             \n"
    "shufps    $0xdd,%%xmm6,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm2,%%xmm6                   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm3,%%xmm1                   \n"
    "pmaddubsw %%xmm3,%%xmm6                   \n"
    "phaddw    %%xmm2,%%xmm0                   \n"
    "phaddw    %%xmm6,%%xmm1                   \n"
    "paddw     %%xmm5,%%xmm0                   \n"
    "paddw     %%xmm5,%%xmm1                   \n"
    "psraw     $0x8,%%xmm0                     \n"
    "psraw     $0x8,%%xmm1                     \n"
    "packsswb  %%xmm1,%%xmm0                   \n"
    "movlps    %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movhps,xmm0,0x00,1,2,1)           //  movhps  %%xmm0,(%1,%2,1)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_argb0),       // %0
    "+r"(dst_u),           // %1
    "+r"(dst_v),           // %2
    "+rm"(width)           // %3
  : "r"((intptr_t)(src_stride_argb)), // %4
    "m"(kARGBToVJ),  // %5
    "m"(kARGBToUJ),  // %6
    "m"(kAddUVJ128)  // %7
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBTOUVJROW_SSSE3

#ifdef HAS_ARGBTOUV444ROW_SSSE3
void ARGBToUV444Row_SSSE3(const uint8* src_argb, uint8* dst_u, uint8* dst_v,
                          int width) {
  asm volatile (
    "movdqa    %4,%%xmm3                       \n"
    "movdqa    %5,%%xmm4                       \n"
    "movdqa    %6,%%xmm5                       \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm6   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm4,%%xmm6                   \n"
    "phaddw    %%xmm1,%%xmm0                   \n"
    "phaddw    %%xmm6,%%xmm2                   \n"
    "psraw     $0x8,%%xmm0                     \n"
    "psraw     $0x8,%%xmm2                     \n"
    "packsswb  %%xmm2,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm6   \n"
    "pmaddubsw %%xmm3,%%xmm0                   \n"
    "pmaddubsw %%xmm3,%%xmm1                   \n"
    "pmaddubsw %%xmm3,%%xmm2                   \n"
    "pmaddubsw %%xmm3,%%xmm6                   \n"
    "phaddw    %%xmm1,%%xmm0                   \n"
    "phaddw    %%xmm6,%%xmm2                   \n"
    "psraw     $0x8,%%xmm0                     \n"
    "psraw     $0x8,%%xmm2                     \n"
    "packsswb  %%xmm2,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    MEMOPMEM(movdqu,xmm0,0x00,1,2,1)           //  movdqu  %%xmm0,(%1,%2,1)
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_argb),        // %0
    "+r"(dst_u),           // %1
    "+r"(dst_v),           // %2
    "+rm"(width)           // %3
  : "m"(kARGBToV),  // %4
    "m"(kARGBToU),  // %5
    "m"(kAddUV128)  // %6
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm6"
  );
}
#endif  // HAS_ARGBTOUV444ROW_SSSE3

void BGRAToYRow_SSSE3(const uint8* src_bgra, uint8* dst_y, int width) {
  asm volatile (
    "movdqa    %4,%%xmm5                       \n"
    "movdqa    %3,%%xmm4                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm4,%%xmm3                   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "phaddw    %%xmm1,%%xmm0                   \n"
    "phaddw    %%xmm3,%%xmm2                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "psrlw     $0x7,%%xmm2                     \n"
    "packuswb  %%xmm2,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_bgra),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  : "m"(kBGRAToY),   // %3
    "m"(kAddY16)     // %4
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void BGRAToUVRow_SSSE3(const uint8* src_bgra0, int src_stride_bgra,
                       uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "movdqa    %5,%%xmm3                       \n"
    "movdqa    %6,%%xmm4                       \n"
    "movdqa    %7,%%xmm5                       \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,0,4,1,xmm7)            //  movdqu (%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqu,0x10,0,4,1,xmm7)            //  movdqu 0x10(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm1                   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    MEMOPREG(movdqu,0x20,0,4,1,xmm7)            //  movdqu 0x20(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm6   \n"
    MEMOPREG(movdqu,0x30,0,4,1,xmm7)            //  movdqu 0x30(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm6                   \n"

    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm7                   \n"
    "shufps    $0x88,%%xmm1,%%xmm0             \n"
    "shufps    $0xdd,%%xmm1,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqa    %%xmm2,%%xmm7                   \n"
    "shufps    $0x88,%%xmm6,%%xmm2             \n"
    "shufps    $0xdd,%%xmm6,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm2,%%xmm6                   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm3,%%xmm1                   \n"
    "pmaddubsw %%xmm3,%%xmm6                   \n"
    "phaddw    %%xmm2,%%xmm0                   \n"
    "phaddw    %%xmm6,%%xmm1                   \n"
    "psraw     $0x8,%%xmm0                     \n"
    "psraw     $0x8,%%xmm1                     \n"
    "packsswb  %%xmm1,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movlps    %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movhps,xmm0,0x00,1,2,1)           //  movhps  %%xmm0,(%1,%2,1)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_bgra0),       // %0
    "+r"(dst_u),           // %1
    "+r"(dst_v),           // %2
    "+rm"(width)           // %3
  : "r"((intptr_t)(src_stride_bgra)), // %4
    "m"(kBGRAToV),  // %5
    "m"(kBGRAToU),  // %6
    "m"(kAddUV128)  // %7
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm6", "xmm7"
  );
}

void ABGRToYRow_SSSE3(const uint8* src_abgr, uint8* dst_y, int width) {
  asm volatile (
    "movdqa    %4,%%xmm5                       \n"
    "movdqa    %3,%%xmm4                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm4,%%xmm3                   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "phaddw    %%xmm1,%%xmm0                   \n"
    "phaddw    %%xmm3,%%xmm2                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "psrlw     $0x7,%%xmm2                     \n"
    "packuswb  %%xmm2,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_abgr),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  : "m"(kABGRToY),   // %3
    "m"(kAddY16)     // %4
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void RGBAToYRow_SSSE3(const uint8* src_rgba, uint8* dst_y, int width) {
  asm volatile (
    "movdqa    %4,%%xmm5                       \n"
    "movdqa    %3,%%xmm4                       \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm4,%%xmm3                   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "phaddw    %%xmm1,%%xmm0                   \n"
    "phaddw    %%xmm3,%%xmm2                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "psrlw     $0x7,%%xmm2                     \n"
    "packuswb  %%xmm2,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_rgba),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  : "m"(kRGBAToY),   // %3
    "m"(kAddY16)     // %4
  : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void ABGRToUVRow_SSSE3(const uint8* src_abgr0, int src_stride_abgr,
                       uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "movdqa    %5,%%xmm3                       \n"
    "movdqa    %6,%%xmm4                       \n"
    "movdqa    %7,%%xmm5                       \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,0,4,1,xmm7)            //  movdqu (%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqu,0x10,0,4,1,xmm7)            //  movdqu 0x10(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm1                   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    MEMOPREG(movdqu,0x20,0,4,1,xmm7)            //  movdqu 0x20(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm6   \n"
    MEMOPREG(movdqu,0x30,0,4,1,xmm7)            //  movdqu 0x30(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm6                   \n"

    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm7                   \n"
    "shufps    $0x88,%%xmm1,%%xmm0             \n"
    "shufps    $0xdd,%%xmm1,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqa    %%xmm2,%%xmm7                   \n"
    "shufps    $0x88,%%xmm6,%%xmm2             \n"
    "shufps    $0xdd,%%xmm6,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm2,%%xmm6                   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm3,%%xmm1                   \n"
    "pmaddubsw %%xmm3,%%xmm6                   \n"
    "phaddw    %%xmm2,%%xmm0                   \n"
    "phaddw    %%xmm6,%%xmm1                   \n"
    "psraw     $0x8,%%xmm0                     \n"
    "psraw     $0x8,%%xmm1                     \n"
    "packsswb  %%xmm1,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movlps    %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movhps,xmm0,0x00,1,2,1)           //  movhps  %%xmm0,(%1,%2,1)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_abgr0),       // %0
    "+r"(dst_u),           // %1
    "+r"(dst_v),           // %2
    "+rm"(width)           // %3
  : "r"((intptr_t)(src_stride_abgr)), // %4
    "m"(kABGRToV),  // %5
    "m"(kABGRToU),  // %6
    "m"(kAddUV128)  // %7
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm6", "xmm7"
  );
}

void RGBAToUVRow_SSSE3(const uint8* src_rgba0, int src_stride_rgba,
                       uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "movdqa    %5,%%xmm3                       \n"
    "movdqa    %6,%%xmm4                       \n"
    "movdqa    %7,%%xmm5                       \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,0,4,1,xmm7)            //  movdqu (%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqu,0x10,0,4,1,xmm7)            //  movdqu 0x10(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm1                   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    MEMOPREG(movdqu,0x20,0,4,1,xmm7)            //  movdqu 0x20(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm6   \n"
    MEMOPREG(movdqu,0x30,0,4,1,xmm7)            //  movdqu 0x30(%0,%4,1),%%xmm7
    "pavgb     %%xmm7,%%xmm6                   \n"

    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm7                   \n"
    "shufps    $0x88,%%xmm1,%%xmm0             \n"
    "shufps    $0xdd,%%xmm1,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm0                   \n"
    "movdqa    %%xmm2,%%xmm7                   \n"
    "shufps    $0x88,%%xmm6,%%xmm2             \n"
    "shufps    $0xdd,%%xmm6,%%xmm7             \n"
    "pavgb     %%xmm7,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm2,%%xmm6                   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm2                   \n"
    "pmaddubsw %%xmm3,%%xmm1                   \n"
    "pmaddubsw %%xmm3,%%xmm6                   \n"
    "phaddw    %%xmm2,%%xmm0                   \n"
    "phaddw    %%xmm6,%%xmm1                   \n"
    "psraw     $0x8,%%xmm0                     \n"
    "psraw     $0x8,%%xmm1                     \n"
    "packsswb  %%xmm1,%%xmm0                   \n"
    "paddb     %%xmm5,%%xmm0                   \n"
    "movlps    %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movhps,xmm0,0x00,1,2,1)           //  movhps  %%xmm0,(%1,%2,1)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_rgba0),       // %0
    "+r"(dst_u),           // %1
    "+r"(dst_v),           // %2
    "+rm"(width)           // %3
  : "r"((intptr_t)(src_stride_rgba)), // %4
    "m"(kRGBAToV),  // %5
    "m"(kRGBAToU),  // %6
    "m"(kAddUV128)  // %7
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm6", "xmm7"
  );
}

#if defined(HAS_I422TOARGBROW_SSSE3) || defined(HAS_I422TOARGBROW_AVX2)

// Read 8 UV from 444
#define READYUV444                                                             \
    "movq       " MEMACCESS([u_buf]) ",%%xmm0                   \n"            \
    MEMOPREG(movq, 0x00, [u_buf], [v_buf], 1, xmm1)                            \
    "lea        " MEMLEA(0x8, [u_buf]) ",%[u_buf]               \n"            \
    "punpcklbw  %%xmm1,%%xmm0                                   \n"            \
    "movq       " MEMACCESS([y_buf]) ",%%xmm4                   \n"            \
    "punpcklbw  %%xmm4,%%xmm4                                   \n"            \
    "lea        " MEMLEA(0x8, [y_buf]) ",%[y_buf]               \n"

// Read 4 UV from 422, upsample to 8 UV
#define READYUV422                                                             \
    "movd       " MEMACCESS([u_buf]) ",%%xmm0                   \n"            \
    MEMOPREG(movd, 0x00, [u_buf], [v_buf], 1, xmm1)                            \
    "lea        " MEMLEA(0x4, [u_buf]) ",%[u_buf]               \n"            \
    "punpcklbw  %%xmm1,%%xmm0                                   \n"            \
    "punpcklwd  %%xmm0,%%xmm0                                   \n"            \
    "movq       " MEMACCESS([y_buf]) ",%%xmm4                   \n"            \
    "punpcklbw  %%xmm4,%%xmm4                                   \n"            \
    "lea        " MEMLEA(0x8, [y_buf]) ",%[y_buf]               \n"

// Read 4 UV from 422, upsample to 8 UV.  With 8 Alpha.
#define READYUVA422                                                            \
    "movd       " MEMACCESS([u_buf]) ",%%xmm0                   \n"            \
    MEMOPREG(movd, 0x00, [u_buf], [v_buf], 1, xmm1)                            \
    "lea        " MEMLEA(0x4, [u_buf]) ",%[u_buf]               \n"            \
    "punpcklbw  %%xmm1,%%xmm0                                   \n"            \
    "punpcklwd  %%xmm0,%%xmm0                                   \n"            \
    "movq       " MEMACCESS([y_buf]) ",%%xmm4                   \n"            \
    "punpcklbw  %%xmm4,%%xmm4                                   \n"            \
    "lea        " MEMLEA(0x8, [y_buf]) ",%[y_buf]               \n"            \
    "movq       " MEMACCESS([a_buf]) ",%%xmm5                   \n"            \
    "lea        " MEMLEA(0x8, [a_buf]) ",%[a_buf]               \n"

// Read 2 UV from 411, upsample to 8 UV.
// reading 4 bytes is an msan violation.
//    "movd       " MEMACCESS([u_buf]) ",%%xmm0                   \n"
//    MEMOPREG(movd, 0x00, [u_buf], [v_buf], 1, xmm1)
// pinsrw fails with drmemory
//  __asm pinsrw     xmm0, [esi], 0        /* U */
//  __asm pinsrw     xmm1, [esi + edi], 0  /* V */
#define READYUV411_TEMP                                                        \
    "movzwl     " MEMACCESS([u_buf]) ",%[temp]                  \n"            \
    "movd       %[temp],%%xmm0                                  \n"            \
    MEMOPARG(movzwl, 0x00, [u_buf], [v_buf], 1, [temp]) "       \n"            \
    "movd       %[temp],%%xmm1                                  \n"            \
    "lea        " MEMLEA(0x2, [u_buf]) ",%[u_buf]               \n"            \
    "punpcklbw  %%xmm1,%%xmm0                                   \n"            \
    "punpcklwd  %%xmm0,%%xmm0                                   \n"            \
    "punpckldq  %%xmm0,%%xmm0                                   \n"            \
    "movq       " MEMACCESS([y_buf]) ",%%xmm4                   \n"            \
    "punpcklbw  %%xmm4,%%xmm4                                   \n"            \
    "lea        " MEMLEA(0x8, [y_buf]) ",%[y_buf]               \n"

// Read 4 UV from NV12, upsample to 8 UV
#define READNV12                                                               \
    "movq       " MEMACCESS([uv_buf]) ",%%xmm0                  \n"            \
    "lea        " MEMLEA(0x8, [uv_buf]) ",%[uv_buf]             \n"            \
    "punpcklwd  %%xmm0,%%xmm0                                   \n"            \
    "movq       " MEMACCESS([y_buf]) ",%%xmm4                   \n"            \
    "punpcklbw  %%xmm4,%%xmm4                                   \n"            \
    "lea        " MEMLEA(0x8, [y_buf]) ",%[y_buf]               \n"

// Read 4 VU from NV21, upsample to 8 UV
#define READNV21                                                               \
    "movq       " MEMACCESS([vu_buf]) ",%%xmm0                  \n"            \
    "lea        " MEMLEA(0x8, [vu_buf]) ",%[vu_buf]             \n"            \
    "pshufb     %[kShuffleNV21], %%xmm0                         \n"            \
    "movq       " MEMACCESS([y_buf]) ",%%xmm4                   \n"            \
    "punpcklbw  %%xmm4,%%xmm4                                   \n"            \
    "lea        " MEMLEA(0x8, [y_buf]) ",%[y_buf]               \n"

// Read 4 YUY2 with 8 Y and update 4 UV to 8 UV.
#define READYUY2                                                               \
    "movdqu     " MEMACCESS([yuy2_buf]) ",%%xmm4                \n"            \
    "pshufb     %[kShuffleYUY2Y], %%xmm4                        \n"            \
    "movdqu     " MEMACCESS([yuy2_buf]) ",%%xmm0                \n"            \
    "pshufb     %[kShuffleYUY2UV], %%xmm0                       \n"            \
    "lea        " MEMLEA(0x10, [yuy2_buf]) ",%[yuy2_buf]        \n"

// Read 4 UYVY with 8 Y and update 4 UV to 8 UV.
#define READUYVY                                                               \
    "movdqu     " MEMACCESS([uyvy_buf]) ",%%xmm4                \n"            \
    "pshufb     %[kShuffleUYVYY], %%xmm4                        \n"            \
    "movdqu     " MEMACCESS([uyvy_buf]) ",%%xmm0                \n"            \
    "pshufb     %[kShuffleUYVYUV], %%xmm0                       \n"            \
    "lea        " MEMLEA(0x10, [uyvy_buf]) ",%[uyvy_buf]        \n"

#if defined(__x86_64__)
#define YUVTORGB_SETUP(yuvconstants)                                           \
    "movdqa     " MEMACCESS([yuvconstants]) ",%%xmm8            \n"            \
    "movdqa     " MEMACCESS2(32, [yuvconstants]) ",%%xmm9       \n"            \
    "movdqa     " MEMACCESS2(64, [yuvconstants]) ",%%xmm10      \n"            \
    "movdqa     " MEMACCESS2(96, [yuvconstants]) ",%%xmm11      \n"            \
    "movdqa     " MEMACCESS2(128, [yuvconstants]) ",%%xmm12     \n"            \
    "movdqa     " MEMACCESS2(160, [yuvconstants]) ",%%xmm13     \n"            \
    "movdqa     " MEMACCESS2(192, [yuvconstants]) ",%%xmm14     \n"
// Convert 8 pixels: 8 UV and 8 Y
#define YUVTORGB(yuvconstants)                                                 \
    "movdqa     %%xmm0,%%xmm1                                   \n"            \
    "movdqa     %%xmm0,%%xmm2                                   \n"            \
    "movdqa     %%xmm0,%%xmm3                                   \n"            \
    "movdqa     %%xmm11,%%xmm0                                  \n"            \
    "pmaddubsw  %%xmm8,%%xmm1                                   \n"            \
    "psubw      %%xmm1,%%xmm0                                   \n"            \
    "movdqa     %%xmm12,%%xmm1                                  \n"            \
    "pmaddubsw  %%xmm9,%%xmm2                                   \n"            \
    "psubw      %%xmm2,%%xmm1                                   \n"            \
    "movdqa     %%xmm13,%%xmm2                                  \n"            \
    "pmaddubsw  %%xmm10,%%xmm3                                  \n"            \
    "psubw      %%xmm3,%%xmm2                                   \n"            \
    "pmulhuw    %%xmm14,%%xmm4                                  \n"            \
    "paddsw     %%xmm4,%%xmm0                                   \n"            \
    "paddsw     %%xmm4,%%xmm1                                   \n"            \
    "paddsw     %%xmm4,%%xmm2                                   \n"            \
    "psraw      $0x6,%%xmm0                                     \n"            \
    "psraw      $0x6,%%xmm1                                     \n"            \
    "psraw      $0x6,%%xmm2                                     \n"            \
    "packuswb   %%xmm0,%%xmm0                                   \n"            \
    "packuswb   %%xmm1,%%xmm1                                   \n"            \
    "packuswb   %%xmm2,%%xmm2                                   \n"
#define YUVTORGB_REGS \
    "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14",

#else
#define YUVTORGB_SETUP(yuvconstants)
// Convert 8 pixels: 8 UV and 8 Y
#define YUVTORGB(yuvconstants)                                                 \
    "movdqa     %%xmm0,%%xmm1                                   \n"            \
    "movdqa     %%xmm0,%%xmm2                                   \n"            \
    "movdqa     %%xmm0,%%xmm3                                   \n"            \
    "movdqa     " MEMACCESS2(96, [yuvconstants]) ",%%xmm0       \n"            \
    "pmaddubsw  " MEMACCESS([yuvconstants]) ",%%xmm1            \n"            \
    "psubw      %%xmm1,%%xmm0                                   \n"            \
    "movdqa     " MEMACCESS2(128, [yuvconstants]) ",%%xmm1      \n"            \
    "pmaddubsw  " MEMACCESS2(32, [yuvconstants]) ",%%xmm2       \n"            \
    "psubw      %%xmm2,%%xmm1                                   \n"            \
    "movdqa     " MEMACCESS2(160, [yuvconstants]) ",%%xmm2      \n"            \
    "pmaddubsw  " MEMACCESS2(64, [yuvconstants]) ",%%xmm3       \n"            \
    "psubw      %%xmm3,%%xmm2                                   \n"            \
    "pmulhuw    " MEMACCESS2(192, [yuvconstants]) ",%%xmm4      \n"            \
    "paddsw     %%xmm4,%%xmm0                                   \n"            \
    "paddsw     %%xmm4,%%xmm1                                   \n"            \
    "paddsw     %%xmm4,%%xmm2                                   \n"            \
    "psraw      $0x6,%%xmm0                                     \n"            \
    "psraw      $0x6,%%xmm1                                     \n"            \
    "psraw      $0x6,%%xmm2                                     \n"            \
    "packuswb   %%xmm0,%%xmm0                                   \n"            \
    "packuswb   %%xmm1,%%xmm1                                   \n"            \
    "packuswb   %%xmm2,%%xmm2                                   \n"
#define YUVTORGB_REGS
#endif

// Store 8 ARGB values.
#define STOREARGB                                                              \
    "punpcklbw  %%xmm1,%%xmm0                                    \n"           \
    "punpcklbw  %%xmm5,%%xmm2                                    \n"           \
    "movdqa     %%xmm0,%%xmm1                                    \n"           \
    "punpcklwd  %%xmm2,%%xmm0                                    \n"           \
    "punpckhwd  %%xmm2,%%xmm1                                    \n"           \
    "movdqu     %%xmm0," MEMACCESS([dst_argb]) "                 \n"           \
    "movdqu     %%xmm1," MEMACCESS2(0x10, [dst_argb]) "          \n"           \
    "lea        " MEMLEA(0x20, [dst_argb]) ", %[dst_argb]        \n"

// Store 8 RGBA values.
#define STORERGBA                                                              \
    "pcmpeqb   %%xmm5,%%xmm5                                     \n"           \
    "punpcklbw %%xmm2,%%xmm1                                     \n"           \
    "punpcklbw %%xmm0,%%xmm5                                     \n"           \
    "movdqa    %%xmm5,%%xmm0                                     \n"           \
    "punpcklwd %%xmm1,%%xmm5                                     \n"           \
    "punpckhwd %%xmm1,%%xmm0                                     \n"           \
    "movdqu    %%xmm5," MEMACCESS([dst_rgba]) "                  \n"           \
    "movdqu    %%xmm0," MEMACCESS2(0x10, [dst_rgba]) "           \n"           \
    "lea       " MEMLEA(0x20, [dst_rgba]) ",%[dst_rgba]          \n"

void OMITFP I444ToARGBRow_SSSE3(const uint8* y_buf,
                                const uint8* u_buf,
                                const uint8* v_buf,
                                uint8* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    LABELALIGN
  "1:                                          \n"
    READYUV444
    YUVTORGB(yuvconstants)
    STOREARGB
    "sub       $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void OMITFP I422ToRGB24Row_SSSE3(const uint8* y_buf,
                                 const uint8* u_buf,
                                 const uint8* v_buf,
                                 uint8* dst_rgb24,
                                 const struct YuvConstants* yuvconstants,
                                 int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "movdqa    %[kShuffleMaskARGBToRGB24_0],%%xmm5 \n"
    "movdqa    %[kShuffleMaskARGBToRGB24],%%xmm6   \n"
    "sub       %[u_buf],%[v_buf]               \n"
    LABELALIGN
  "1:                                          \n"
    READYUV422
    YUVTORGB(yuvconstants)
    "punpcklbw %%xmm1,%%xmm0                   \n"
    "punpcklbw %%xmm2,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklwd %%xmm2,%%xmm0                   \n"
    "punpckhwd %%xmm2,%%xmm1                   \n"
    "pshufb    %%xmm5,%%xmm0                   \n"
    "pshufb    %%xmm6,%%xmm1                   \n"
    "palignr   $0xc,%%xmm0,%%xmm1              \n"
    "movq      %%xmm0," MEMACCESS([dst_rgb24]) "\n"
    "movdqu    %%xmm1," MEMACCESS2(0x8,[dst_rgb24]) "\n"
    "lea       " MEMLEA(0x18,[dst_rgb24]) ",%[dst_rgb24] \n"
    "subl      $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_rgb24]"+r"(dst_rgb24),  // %[dst_rgb24]
#if defined(__i386__) && defined(__pic__)
    [width]"+m"(width)     // %[width]
#else
    [width]"+rm"(width)    // %[width]
#endif
  : [yuvconstants]"r"(yuvconstants),  // %[yuvconstants]
    [kShuffleMaskARGBToRGB24_0]"m"(kShuffleMaskARGBToRGB24_0),
    [kShuffleMaskARGBToRGB24]"m"(kShuffleMaskARGBToRGB24)
  : "memory", "cc", NACL_R14 YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}

void OMITFP I422ToARGBRow_SSSE3(const uint8* y_buf,
                                const uint8* u_buf,
                                const uint8* v_buf,
                                uint8* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    LABELALIGN
  "1:                                          \n"
    READYUV422
    YUVTORGB(yuvconstants)
    STOREARGB
    "sub       $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

#ifdef HAS_I422ALPHATOARGBROW_SSSE3
void OMITFP I422AlphaToARGBRow_SSSE3(const uint8* y_buf,
                                     const uint8* u_buf,
                                     const uint8* v_buf,
                                     const uint8* a_buf,
                                     uint8* dst_argb,
                                     const struct YuvConstants* yuvconstants,
                                     int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    LABELALIGN
  "1:                                          \n"
    READYUVA422
    YUVTORGB(yuvconstants)
    STOREARGB
    "subl      $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [a_buf]"+r"(a_buf),    // %[a_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__) && defined(__pic__)
    [width]"+m"(width)     // %[width]
#else
    [width]"+rm"(width)    // %[width]
#endif
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I422ALPHATOARGBROW_SSSE3

#ifdef HAS_I411TOARGBROW_SSSE3
void OMITFP I411ToARGBRow_SSSE3(const uint8* y_buf,
                                const uint8* u_buf,
                                const uint8* v_buf,
                                uint8* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  int temp;
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    LABELALIGN
  "1:                                          \n"
    READYUV411_TEMP
    YUVTORGB(yuvconstants)
    STOREARGB
    "subl      $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [y_buf]"+r"(y_buf),        // %[y_buf]
    [u_buf]"+r"(u_buf),        // %[u_buf]
    [v_buf]"+r"(v_buf),        // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [temp]"=&r"(temp),         // %[temp]
#if defined(__i386__) && defined(__pic__)
    [width]"+m"(width)         // %[width]
#else
    [width]"+rm"(width)        // %[width]
#endif
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif

void OMITFP NV12ToARGBRow_SSSE3(const uint8* y_buf,
                                const uint8* uv_buf,
                                uint8* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    LABELALIGN
  "1:                                          \n"
    READNV12
    YUVTORGB(yuvconstants)
    STOREARGB
    "sub       $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [uv_buf]"+r"(uv_buf),    // %[uv_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
    : "memory", "cc", YUVTORGB_REGS  // Does not use r14.
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void OMITFP NV21ToARGBRow_SSSE3(const uint8* y_buf,
                                const uint8* vu_buf,
                                uint8* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    LABELALIGN
  "1:                                          \n"
    READNV21
    YUVTORGB(yuvconstants)
    STOREARGB
    "sub       $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [vu_buf]"+r"(vu_buf),    // %[vu_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants), // %[yuvconstants]
    [kShuffleNV21]"m"(kShuffleNV21)
    : "memory", "cc", YUVTORGB_REGS  // Does not use r14.
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void OMITFP YUY2ToARGBRow_SSSE3(const uint8* yuy2_buf,
                                uint8* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    LABELALIGN
  "1:                                          \n"
    READYUY2
    YUVTORGB(yuvconstants)
    STOREARGB
    "sub       $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [yuy2_buf]"+r"(yuy2_buf),    // %[yuy2_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants), // %[yuvconstants]
    [kShuffleYUY2Y]"m"(kShuffleYUY2Y),
    [kShuffleYUY2UV]"m"(kShuffleYUY2UV)
    : "memory", "cc", YUVTORGB_REGS  // Does not use r14.
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void OMITFP UYVYToARGBRow_SSSE3(const uint8* uyvy_buf,
                                uint8* dst_argb,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    LABELALIGN
  "1:                                          \n"
    READUYVY
    YUVTORGB(yuvconstants)
    STOREARGB
    "sub       $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [uyvy_buf]"+r"(uyvy_buf),    // %[uyvy_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants), // %[yuvconstants]
    [kShuffleUYVYY]"m"(kShuffleUYVYY),
    [kShuffleUYVYUV]"m"(kShuffleUYVYUV)
    : "memory", "cc", YUVTORGB_REGS  // Does not use r14.
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

void OMITFP I422ToRGBARow_SSSE3(const uint8* y_buf,
                                const uint8* u_buf,
                                const uint8* v_buf,
                                uint8* dst_rgba,
                                const struct YuvConstants* yuvconstants,
                                int width) {
  asm volatile (
    YUVTORGB_SETUP(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    LABELALIGN
  "1:                                          \n"
    READYUV422
    YUVTORGB(yuvconstants)
    STORERGBA
    "sub       $0x8,%[width]                   \n"
    "jg        1b                              \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_rgba]"+r"(dst_rgba),  // %[dst_rgba]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}

#endif  // HAS_I422TOARGBROW_SSSE3

// Read 16 UV from 444
#define READYUV444_AVX2                                                        \
    "vmovdqu    " MEMACCESS([u_buf]) ",%%xmm0                       \n"        \
    MEMOPREG(vmovdqu, 0x00, [u_buf], [v_buf], 1, xmm1)                         \
    "lea        " MEMLEA(0x10, [u_buf]) ",%[u_buf]                  \n"        \
    "vpermq     $0xd8,%%ymm0,%%ymm0                                 \n"        \
    "vpermq     $0xd8,%%ymm1,%%ymm1                                 \n"        \
    "vpunpcklbw %%ymm1,%%ymm0,%%ymm0                                \n"        \
    "vmovdqu    " MEMACCESS([y_buf]) ",%%xmm4                       \n"        \
    "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n"        \
    "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n"        \
    "lea        " MEMLEA(0x10, [y_buf]) ",%[y_buf]                  \n"

// Read 8 UV from 422, upsample to 16 UV.
#define READYUV422_AVX2                                                        \
    "vmovq      " MEMACCESS([u_buf]) ",%%xmm0                       \n"        \
    MEMOPREG(vmovq, 0x00, [u_buf], [v_buf], 1, xmm1)                           \
    "lea        " MEMLEA(0x8, [u_buf]) ",%[u_buf]                   \n"        \
    "vpunpcklbw %%ymm1,%%ymm0,%%ymm0                                \n"        \
    "vpermq     $0xd8,%%ymm0,%%ymm0                                 \n"        \
    "vpunpcklwd %%ymm0,%%ymm0,%%ymm0                                \n"        \
    "vmovdqu    " MEMACCESS([y_buf]) ",%%xmm4                       \n"        \
    "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n"        \
    "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n"        \
    "lea        " MEMLEA(0x10, [y_buf]) ",%[y_buf]                  \n"

// Read 8 UV from 422, upsample to 16 UV.  With 16 Alpha.
#define READYUVA422_AVX2                                                       \
    "vmovq      " MEMACCESS([u_buf]) ",%%xmm0                       \n"        \
    MEMOPREG(vmovq, 0x00, [u_buf], [v_buf], 1, xmm1)                           \
    "lea        " MEMLEA(0x8, [u_buf]) ",%[u_buf]                   \n"        \
    "vpunpcklbw %%ymm1,%%ymm0,%%ymm0                                \n"        \
    "vpermq     $0xd8,%%ymm0,%%ymm0                                 \n"        \
    "vpunpcklwd %%ymm0,%%ymm0,%%ymm0                                \n"        \
    "vmovdqu    " MEMACCESS([y_buf]) ",%%xmm4                       \n"        \
    "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n"        \
    "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n"        \
    "lea        " MEMLEA(0x10, [y_buf]) ",%[y_buf]                  \n"        \
    "vmovdqu    " MEMACCESS([a_buf]) ",%%xmm5                       \n"        \
    "vpermq     $0xd8,%%ymm5,%%ymm5                                 \n"        \
    "lea        " MEMLEA(0x10, [a_buf]) ",%[a_buf]                  \n"

// Read 4 UV from 411, upsample to 16 UV.
#define READYUV411_AVX2                                                        \
    "vmovd      " MEMACCESS([u_buf]) ",%%xmm0                       \n"        \
    MEMOPREG(vmovd, 0x00, [u_buf], [v_buf], 1, xmm1)                           \
    "lea        " MEMLEA(0x4, [u_buf]) ",%[u_buf]                   \n"        \
    "vpunpcklbw %%ymm1,%%ymm0,%%ymm0                                \n"        \
    "vpunpcklwd %%ymm0,%%ymm0,%%ymm0                                \n"        \
    "vpermq     $0xd8,%%ymm0,%%ymm0                                 \n"        \
    "vpunpckldq %%ymm0,%%ymm0,%%ymm0                                \n"        \
    "vmovdqu    " MEMACCESS([y_buf]) ",%%xmm4                       \n"        \
    "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n"        \
    "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n"        \
    "lea        " MEMLEA(0x10, [y_buf]) ",%[y_buf]                  \n"

// Read 8 UV from NV12, upsample to 16 UV.
#define READNV12_AVX2                                                          \
    "vmovdqu    " MEMACCESS([uv_buf]) ",%%xmm0                      \n"        \
    "lea        " MEMLEA(0x10, [uv_buf]) ",%[uv_buf]                \n"        \
    "vpermq     $0xd8,%%ymm0,%%ymm0                                 \n"        \
    "vpunpcklwd %%ymm0,%%ymm0,%%ymm0                                \n"        \
    "vmovdqu    " MEMACCESS([y_buf]) ",%%xmm4                       \n"        \
    "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n"        \
    "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n"        \
    "lea        " MEMLEA(0x10, [y_buf]) ",%[y_buf]                  \n"

// Read 8 VU from NV21, upsample to 16 UV.
#define READNV21_AVX2                                                          \
    "vmovdqu    " MEMACCESS([vu_buf]) ",%%xmm0                      \n"        \
    "lea        " MEMLEA(0x10, [vu_buf]) ",%[vu_buf]                \n"        \
    "vpermq     $0xd8,%%ymm0,%%ymm0                                 \n"        \
    "vpshufb     %[kShuffleNV21], %%ymm0, %%ymm0                    \n"        \
    "vmovdqu    " MEMACCESS([y_buf]) ",%%xmm4                       \n"        \
    "vpermq     $0xd8,%%ymm4,%%ymm4                                 \n"        \
    "vpunpcklbw %%ymm4,%%ymm4,%%ymm4                                \n"        \
    "lea        " MEMLEA(0x10, [y_buf]) ",%[y_buf]                  \n"

// Read 8 YUY2 with 16 Y and upsample 8 UV to 16 UV.
#define READYUY2_AVX2                                                          \
    "vmovdqu    " MEMACCESS([yuy2_buf]) ",%%ymm4                    \n"        \
    "vpshufb    %[kShuffleYUY2Y], %%ymm4, %%ymm4                    \n"        \
    "vmovdqu    " MEMACCESS([yuy2_buf]) ",%%ymm0                    \n"        \
    "vpshufb    %[kShuffleYUY2UV], %%ymm0, %%ymm0                   \n"        \
    "lea        " MEMLEA(0x20, [yuy2_buf]) ",%[yuy2_buf]            \n"

// Read 8 UYVY with 16 Y and upsample 8 UV to 16 UV.
#define READUYVY_AVX2                                                          \
    "vmovdqu     " MEMACCESS([uyvy_buf]) ",%%ymm4                   \n"        \
    "vpshufb     %[kShuffleUYVYY], %%ymm4, %%ymm4                   \n"        \
    "vmovdqu     " MEMACCESS([uyvy_buf]) ",%%ymm0                   \n"        \
    "vpshufb     %[kShuffleUYVYUV], %%ymm0, %%ymm0                  \n"        \
    "lea        " MEMLEA(0x20, [uyvy_buf]) ",%[uyvy_buf]            \n"

#if defined(__x86_64__)
#define YUVTORGB_SETUP_AVX2(yuvconstants)                                      \
    "vmovdqa     " MEMACCESS([yuvconstants]) ",%%ymm8            \n"           \
    "vmovdqa     " MEMACCESS2(32, [yuvconstants]) ",%%ymm9       \n"           \
    "vmovdqa     " MEMACCESS2(64, [yuvconstants]) ",%%ymm10      \n"           \
    "vmovdqa     " MEMACCESS2(96, [yuvconstants]) ",%%ymm11      \n"           \
    "vmovdqa     " MEMACCESS2(128, [yuvconstants]) ",%%ymm12     \n"           \
    "vmovdqa     " MEMACCESS2(160, [yuvconstants]) ",%%ymm13     \n"           \
    "vmovdqa     " MEMACCESS2(192, [yuvconstants]) ",%%ymm14     \n"
#define YUVTORGB_AVX2(yuvconstants)                                            \
    "vpmaddubsw  %%ymm10,%%ymm0,%%ymm2                              \n"        \
    "vpmaddubsw  %%ymm9,%%ymm0,%%ymm1                               \n"        \
    "vpmaddubsw  %%ymm8,%%ymm0,%%ymm0                               \n"        \
    "vpsubw      %%ymm2,%%ymm13,%%ymm2                              \n"        \
    "vpsubw      %%ymm1,%%ymm12,%%ymm1                              \n"        \
    "vpsubw      %%ymm0,%%ymm11,%%ymm0                              \n"        \
    "vpmulhuw    %%ymm14,%%ymm4,%%ymm4                              \n"        \
    "vpaddsw     %%ymm4,%%ymm0,%%ymm0                               \n"        \
    "vpaddsw     %%ymm4,%%ymm1,%%ymm1                               \n"        \
    "vpaddsw     %%ymm4,%%ymm2,%%ymm2                               \n"        \
    "vpsraw      $0x6,%%ymm0,%%ymm0                                 \n"        \
    "vpsraw      $0x6,%%ymm1,%%ymm1                                 \n"        \
    "vpsraw      $0x6,%%ymm2,%%ymm2                                 \n"        \
    "vpackuswb   %%ymm0,%%ymm0,%%ymm0                               \n"        \
    "vpackuswb   %%ymm1,%%ymm1,%%ymm1                               \n"        \
    "vpackuswb   %%ymm2,%%ymm2,%%ymm2                               \n"
#define YUVTORGB_REGS_AVX2 \
    "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14",
#else  // Convert 16 pixels: 16 UV and 16 Y.
#define YUVTORGB_SETUP_AVX2(yuvconstants)
#define YUVTORGB_AVX2(yuvconstants)                                            \
    "vpmaddubsw  " MEMACCESS2(64, [yuvconstants]) ",%%ymm0,%%ymm2   \n"        \
    "vpmaddubsw  " MEMACCESS2(32, [yuvconstants]) ",%%ymm0,%%ymm1   \n"        \
    "vpmaddubsw  " MEMACCESS([yuvconstants]) ",%%ymm0,%%ymm0        \n"        \
    "vmovdqu     " MEMACCESS2(160, [yuvconstants]) ",%%ymm3         \n"        \
    "vpsubw      %%ymm2,%%ymm3,%%ymm2                               \n"        \
    "vmovdqu     " MEMACCESS2(128, [yuvconstants]) ",%%ymm3         \n"        \
    "vpsubw      %%ymm1,%%ymm3,%%ymm1                               \n"        \
    "vmovdqu     " MEMACCESS2(96, [yuvconstants]) ",%%ymm3          \n"        \
    "vpsubw      %%ymm0,%%ymm3,%%ymm0                               \n"        \
    "vpmulhuw    " MEMACCESS2(192, [yuvconstants]) ",%%ymm4,%%ymm4  \n"        \
    "vpaddsw     %%ymm4,%%ymm0,%%ymm0                               \n"        \
    "vpaddsw     %%ymm4,%%ymm1,%%ymm1                               \n"        \
    "vpaddsw     %%ymm4,%%ymm2,%%ymm2                               \n"        \
    "vpsraw      $0x6,%%ymm0,%%ymm0                                 \n"        \
    "vpsraw      $0x6,%%ymm1,%%ymm1                                 \n"        \
    "vpsraw      $0x6,%%ymm2,%%ymm2                                 \n"        \
    "vpackuswb   %%ymm0,%%ymm0,%%ymm0                               \n"        \
    "vpackuswb   %%ymm1,%%ymm1,%%ymm1                               \n"        \
    "vpackuswb   %%ymm2,%%ymm2,%%ymm2                               \n"
#define YUVTORGB_REGS_AVX2
#endif

// Store 16 ARGB values.
#define STOREARGB_AVX2                                                         \
    "vpunpcklbw %%ymm1,%%ymm0,%%ymm0                                \n"        \
    "vpermq     $0xd8,%%ymm0,%%ymm0                                 \n"        \
    "vpunpcklbw %%ymm5,%%ymm2,%%ymm2                                \n"        \
    "vpermq     $0xd8,%%ymm2,%%ymm2                                 \n"        \
    "vpunpcklwd %%ymm2,%%ymm0,%%ymm1                                \n"        \
    "vpunpckhwd %%ymm2,%%ymm0,%%ymm0                                \n"        \
    "vmovdqu    %%ymm1," MEMACCESS([dst_argb]) "                    \n"        \
    "vmovdqu    %%ymm0," MEMACCESS2(0x20, [dst_argb]) "             \n"        \
    "lea       " MEMLEA(0x40, [dst_argb]) ", %[dst_argb]            \n"

#ifdef HAS_I444TOARGBROW_AVX2
// 16 pixels
// 16 UV values with 16 Y producing 16 ARGB (64 bytes).
void OMITFP I444ToARGBRow_AVX2(const uint8* y_buf,
                               const uint8* u_buf,
                               const uint8* v_buf,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    "vpcmpeqb  %%ymm5,%%ymm5,%%ymm5            \n"
    LABELALIGN
  "1:                                          \n"
    READYUV444_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
    "sub       $0x10,%[width]                  \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I444TOARGBROW_AVX2

#ifdef HAS_I411TOARGBROW_AVX2
// 16 pixels
// 4 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP I411ToARGBRow_AVX2(const uint8* y_buf,
                               const uint8* u_buf,
                               const uint8* v_buf,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    "vpcmpeqb  %%ymm5,%%ymm5,%%ymm5            \n"
    LABELALIGN
  "1:                                          \n"
    READYUV411_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
    "sub       $0x10,%[width]                  \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I411TOARGBROW_AVX2

#if defined(HAS_I422TOARGBROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP I422ToARGBRow_AVX2(const uint8* y_buf,
                               const uint8* u_buf,
                               const uint8* v_buf,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    "vpcmpeqb  %%ymm5,%%ymm5,%%ymm5            \n"
    LABELALIGN
  "1:                                          \n"
    READYUV422_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
    "sub       $0x10,%[width]                  \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I422TOARGBROW_AVX2

#if defined(HAS_I422ALPHATOARGBROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y and 16 A producing 16 ARGB.
void OMITFP I422AlphaToARGBRow_AVX2(const uint8* y_buf,
                               const uint8* u_buf,
                               const uint8* v_buf,
                               const uint8* a_buf,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    LABELALIGN
  "1:                                          \n"
    READYUVA422_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
    "subl      $0x10,%[width]                  \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [a_buf]"+r"(a_buf),    // %[a_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
#if defined(__i386__) && defined(__pic__)
    [width]"+m"(width)     // %[width]
#else
    [width]"+rm"(width)    // %[width]
#endif
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I422ALPHATOARGBROW_AVX2

#if defined(HAS_I422TORGBAROW_AVX2)
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 RGBA (64 bytes).
void OMITFP I422ToRGBARow_AVX2(const uint8* y_buf,
                               const uint8* u_buf,
                               const uint8* v_buf,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
    "sub       %[u_buf],%[v_buf]               \n"
    "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5           \n"
    LABELALIGN
  "1:                                          \n"
    READYUV422_AVX2
    YUVTORGB_AVX2(yuvconstants)

    // Step 3: Weave into RGBA
    "vpunpcklbw %%ymm2,%%ymm1,%%ymm1           \n"
    "vpermq     $0xd8,%%ymm1,%%ymm1            \n"
    "vpunpcklbw %%ymm0,%%ymm5,%%ymm2           \n"
    "vpermq     $0xd8,%%ymm2,%%ymm2            \n"
    "vpunpcklwd %%ymm1,%%ymm2,%%ymm0           \n"
    "vpunpckhwd %%ymm1,%%ymm2,%%ymm1           \n"
    "vmovdqu    %%ymm0," MEMACCESS([dst_argb]) "\n"
    "vmovdqu    %%ymm1," MEMACCESS2(0x20,[dst_argb]) "\n"
    "lea       " MEMLEA(0x40,[dst_argb]) ",%[dst_argb] \n"
    "sub       $0x10,%[width]                  \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [u_buf]"+r"(u_buf),    // %[u_buf]
    [v_buf]"+r"(v_buf),    // %[v_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
  : "memory", "cc", NACL_R14 YUVTORGB_REGS_AVX2
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_I422TORGBAROW_AVX2

#if defined(HAS_NV12TOARGBROW_AVX2)
// 16 pixels.
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP NV12ToARGBRow_AVX2(const uint8* y_buf,
                               const uint8* uv_buf,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
    "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5           \n"
    LABELALIGN
  "1:                                          \n"
    READNV12_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
    "sub       $0x10,%[width]                  \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [uv_buf]"+r"(uv_buf),    // %[uv_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants)  // %[yuvconstants]
    : "memory", "cc", YUVTORGB_REGS_AVX2  // Does not use r14.
    "xmm0", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_NV12TOARGBROW_AVX2

#if defined(HAS_NV21TOARGBROW_AVX2)
// 16 pixels.
// 8 VU values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
void OMITFP NV21ToARGBRow_AVX2(const uint8* y_buf,
                               const uint8* vu_buf,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
    "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5           \n"
    LABELALIGN
  "1:                                          \n"
    READNV21_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
    "sub       $0x10,%[width]                  \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : [y_buf]"+r"(y_buf),    // %[y_buf]
    [vu_buf]"+r"(vu_buf),    // %[vu_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants), // %[yuvconstants]
    [kShuffleNV21]"m"(kShuffleNV21)
    : "memory", "cc", YUVTORGB_REGS_AVX2  // Does not use r14.
      "xmm0", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_NV21TOARGBROW_AVX2

#if defined(HAS_YUY2TOARGBROW_AVX2)
// 16 pixels.
// 8 YUY2 values with 16 Y and 8 UV producing 16 ARGB (64 bytes).
void OMITFP YUY2ToARGBRow_AVX2(const uint8* yuy2_buf,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
    "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5           \n"
    LABELALIGN
  "1:                                          \n"
    READYUY2_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
    "sub       $0x10,%[width]                  \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : [yuy2_buf]"+r"(yuy2_buf),    // %[yuy2_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants), // %[yuvconstants]
    [kShuffleYUY2Y]"m"(kShuffleYUY2Y),
    [kShuffleYUY2UV]"m"(kShuffleYUY2UV)
    : "memory", "cc", YUVTORGB_REGS_AVX2  // Does not use r14.
      "xmm0", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_YUY2TOARGBROW_AVX2

#if defined(HAS_UYVYTOARGBROW_AVX2)
// 16 pixels.
// 8 UYVY values with 16 Y and 8 UV producing 16 ARGB (64 bytes).
void OMITFP UYVYToARGBRow_AVX2(const uint8* uyvy_buf,
                               uint8* dst_argb,
                               const struct YuvConstants* yuvconstants,
                               int width) {
  asm volatile (
    YUVTORGB_SETUP_AVX2(yuvconstants)
    "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5           \n"
    LABELALIGN
  "1:                                          \n"
    READUYVY_AVX2
    YUVTORGB_AVX2(yuvconstants)
    STOREARGB_AVX2
    "sub       $0x10,%[width]                  \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : [uyvy_buf]"+r"(uyvy_buf),    // %[uyvy_buf]
    [dst_argb]"+r"(dst_argb),  // %[dst_argb]
    [width]"+rm"(width)    // %[width]
  : [yuvconstants]"r"(yuvconstants), // %[yuvconstants]
    [kShuffleUYVYY]"m"(kShuffleUYVYY),
    [kShuffleUYVYUV]"m"(kShuffleUYVYUV)
    : "memory", "cc", YUVTORGB_REGS_AVX2  // Does not use r14.
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_UYVYTOARGBROW_AVX2

#ifdef HAS_I400TOARGBROW_SSE2
void I400ToARGBRow_SSE2(const uint8* y_buf, uint8* dst_argb, int width) {
  asm volatile (
    "mov       $0x4a354a35,%%eax               \n"  // 4a35 = 18997 = 1.164
    "movd      %%eax,%%xmm2                    \n"
    "pshufd    $0x0,%%xmm2,%%xmm2              \n"
    "mov       $0x04880488,%%eax               \n"  // 0488 = 1160 = 1.164 * 16
    "movd      %%eax,%%xmm3                    \n"
    "pshufd    $0x0,%%xmm3,%%xmm3              \n"
    "pcmpeqb   %%xmm4,%%xmm4                   \n"
    "pslld     $0x18,%%xmm4                    \n"
    LABELALIGN
  "1:                                          \n"
    // Step 1: Scale Y contribution to 8 G values. G = (y - 16) * 1.164
    "movq      " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x8,0) ",%0            \n"
    "punpcklbw %%xmm0,%%xmm0                   \n"
    "pmulhuw   %%xmm2,%%xmm0                   \n"
    "psubusw   %%xmm3,%%xmm0                   \n"
    "psrlw     $6, %%xmm0                      \n"
    "packuswb  %%xmm0,%%xmm0                   \n"

    // Step 2: Weave into ARGB
    "punpcklbw %%xmm0,%%xmm0                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklwd %%xmm0,%%xmm0                   \n"
    "punpckhwd %%xmm1,%%xmm1                   \n"
    "por       %%xmm4,%%xmm0                   \n"
    "por       %%xmm4,%%xmm1                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"

    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(y_buf),     // %0
    "+r"(dst_argb),  // %1
    "+rm"(width)     // %2
  :
  : "memory", "cc", "eax"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4"
  );
}
#endif  // HAS_I400TOARGBROW_SSE2

#ifdef HAS_I400TOARGBROW_AVX2
// 16 pixels of Y converted to 16 pixels of ARGB (64 bytes).
// note: vpunpcklbw mutates and vpackuswb unmutates.
void I400ToARGBRow_AVX2(const uint8* y_buf, uint8* dst_argb, int width) {
  asm volatile (
    "mov        $0x4a354a35,%%eax              \n" // 0488 = 1160 = 1.164 * 16
    "vmovd      %%eax,%%xmm2                   \n"
    "vbroadcastss %%xmm2,%%ymm2                \n"
    "mov        $0x4880488,%%eax               \n" // 4a35 = 18997 = 1.164
    "vmovd      %%eax,%%xmm3                   \n"
    "vbroadcastss %%xmm3,%%ymm3                \n"
    "vpcmpeqb   %%ymm4,%%ymm4,%%ymm4           \n"
    "vpslld     $0x18,%%ymm4,%%ymm4            \n"

    LABELALIGN
  "1:                                          \n"
    // Step 1: Scale Y contribution to 16 G values. G = (y - 16) * 1.164
    "vmovdqu    " MEMACCESS(0) ",%%xmm0        \n"
    "lea        " MEMLEA(0x10,0) ",%0          \n"
    "vpermq     $0xd8,%%ymm0,%%ymm0            \n"
    "vpunpcklbw %%ymm0,%%ymm0,%%ymm0           \n"
    "vpmulhuw   %%ymm2,%%ymm0,%%ymm0           \n"
    "vpsubusw   %%ymm3,%%ymm0,%%ymm0           \n"
    "vpsrlw     $0x6,%%ymm0,%%ymm0             \n"
    "vpackuswb  %%ymm0,%%ymm0,%%ymm0           \n"
    "vpunpcklbw %%ymm0,%%ymm0,%%ymm1           \n"
    "vpermq     $0xd8,%%ymm1,%%ymm1            \n"
    "vpunpcklwd %%ymm1,%%ymm1,%%ymm0           \n"
    "vpunpckhwd %%ymm1,%%ymm1,%%ymm1           \n"
    "vpor       %%ymm4,%%ymm0,%%ymm0           \n"
    "vpor       %%ymm4,%%ymm1,%%ymm1           \n"
    "vmovdqu    %%ymm0," MEMACCESS(1) "        \n"
    "vmovdqu    %%ymm1," MEMACCESS2(0x20,1) "  \n"
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "sub        $0x10,%2                       \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(y_buf),     // %0
    "+r"(dst_argb),  // %1
    "+rm"(width)     // %2
  :
  : "memory", "cc", "eax"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4"
  );
}
#endif  // HAS_I400TOARGBROW_AVX2

#ifdef HAS_MIRRORROW_SSSE3
// Shuffle table for reversing the bytes.
static uvec8 kShuffleMirror = {
  15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u
};

void MirrorRow_SSSE3(const uint8* src, uint8* dst, int width) {
  intptr_t temp_width = (intptr_t)(width);
  asm volatile (
    "movdqa    %3,%%xmm5                       \n"
    LABELALIGN
  "1:                                          \n"
    MEMOPREG(movdqu,-0x10,0,2,1,xmm0)          //  movdqu -0x10(%0,%2),%%xmm0
    "pshufb    %%xmm5,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(temp_width)  // %2
  : "m"(kShuffleMirror) // %3
  : "memory", "cc", NACL_R14
    "xmm0", "xmm5"
  );
}
#endif  // HAS_MIRRORROW_SSSE3

#ifdef HAS_MIRRORROW_AVX2
void MirrorRow_AVX2(const uint8* src, uint8* dst, int width) {
  intptr_t temp_width = (intptr_t)(width);
  asm volatile (
    "vbroadcastf128 %3,%%ymm5                  \n"
    LABELALIGN
  "1:                                          \n"
    MEMOPREG(vmovdqu,-0x20,0,2,1,ymm0)         //  vmovdqu -0x20(%0,%2),%%ymm0
    "vpshufb    %%ymm5,%%ymm0,%%ymm0           \n"
    "vpermq     $0x4e,%%ymm0,%%ymm0            \n"
    "vmovdqu    %%ymm0," MEMACCESS(1) "        \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x20,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(temp_width)  // %2
  : "m"(kShuffleMirror) // %3
  : "memory", "cc", NACL_R14
    "xmm0", "xmm5"
  );
}
#endif  // HAS_MIRRORROW_AVX2

#ifdef HAS_MIRRORUVROW_SSSE3
// Shuffle table for reversing the bytes of UV channels.
static uvec8 kShuffleMirrorUV = {
  14u, 12u, 10u, 8u, 6u, 4u, 2u, 0u, 15u, 13u, 11u, 9u, 7u, 5u, 3u, 1u
};
void MirrorUVRow_SSSE3(const uint8* src, uint8* dst_u, uint8* dst_v,
                       int width) {
  intptr_t temp_width = (intptr_t)(width);
  asm volatile (
    "movdqa    %4,%%xmm1                       \n"
    "lea       " MEMLEA4(-0x10,0,3,2) ",%0     \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(-0x10,0) ",%0          \n"
    "pshufb    %%xmm1,%%xmm0                   \n"
    "movlpd    %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movhpd,xmm0,0x00,1,2,1)           //  movhpd    %%xmm0,(%1,%2)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $8,%3                           \n"
    "jg        1b                              \n"
  : "+r"(src),      // %0
    "+r"(dst_u),    // %1
    "+r"(dst_v),    // %2
    "+r"(temp_width)  // %3
  : "m"(kShuffleMirrorUV)  // %4
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1"
  );
}
#endif  // HAS_MIRRORUVROW_SSSE3

#ifdef HAS_ARGBMIRRORROW_SSE2

void ARGBMirrorRow_SSE2(const uint8* src, uint8* dst, int width) {
  intptr_t temp_width = (intptr_t)(width);
  asm volatile (
    "lea       " MEMLEA4(-0x10,0,2,4) ",%0     \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "pshufd    $0x1b,%%xmm0,%%xmm0             \n"
    "lea       " MEMLEA(-0x10,0) ",%0          \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x4,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(temp_width)  // %2
  :
  : "memory", "cc"
    , "xmm0"
  );
}
#endif  // HAS_ARGBMIRRORROW_SSE2

#ifdef HAS_ARGBMIRRORROW_AVX2
// Shuffle table for reversing the bytes.
static const ulvec32 kARGBShuffleMirror_AVX2 = {
  7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u
};
void ARGBMirrorRow_AVX2(const uint8* src, uint8* dst, int width) {
  intptr_t temp_width = (intptr_t)(width);
  asm volatile (
    "vmovdqu    %3,%%ymm5                      \n"
    LABELALIGN
  "1:                                          \n"
    VMEMOPREG(vpermd,-0x20,0,2,4,ymm5,ymm0) // vpermd -0x20(%0,%2,4),ymm5,ymm0
    "vmovdqu    %%ymm0," MEMACCESS(1) "        \n"
    "lea        " MEMLEA(0x20,1) ",%1          \n"
    "sub        $0x8,%2                        \n"
    "jg         1b                             \n"
    "vzeroupper                                \n"
  : "+r"(src),  // %0
    "+r"(dst),  // %1
    "+r"(temp_width)  // %2
  : "m"(kARGBShuffleMirror_AVX2) // %3
  : "memory", "cc", NACL_R14
    "xmm0", "xmm5"
  );
}
#endif  // HAS_ARGBMIRRORROW_AVX2

#ifdef HAS_SPLITUVROW_AVX2
void SplitUVRow_AVX2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                     int width) {
  asm volatile (
    "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5             \n"
    "vpsrlw     $0x8,%%ymm5,%%ymm5               \n"
    "sub        %1,%2                            \n"
    LABELALIGN
  "1:                                            \n"
    "vmovdqu    " MEMACCESS(0) ",%%ymm0          \n"
    "vmovdqu    " MEMACCESS2(0x20,0) ",%%ymm1    \n"
    "lea        " MEMLEA(0x40,0) ",%0            \n"
    "vpsrlw     $0x8,%%ymm0,%%ymm2               \n"
    "vpsrlw     $0x8,%%ymm1,%%ymm3               \n"
    "vpand      %%ymm5,%%ymm0,%%ymm0             \n"
    "vpand      %%ymm5,%%ymm1,%%ymm1             \n"
    "vpackuswb  %%ymm1,%%ymm0,%%ymm0             \n"
    "vpackuswb  %%ymm3,%%ymm2,%%ymm2             \n"
    "vpermq     $0xd8,%%ymm0,%%ymm0              \n"
    "vpermq     $0xd8,%%ymm2,%%ymm2              \n"
    "vmovdqu    %%ymm0," MEMACCESS(1) "          \n"
    MEMOPMEM(vmovdqu,ymm2,0x00,1,2,1)             //  vmovdqu %%ymm2,(%1,%2)
    "lea        " MEMLEA(0x20,1) ",%1            \n"
    "sub        $0x20,%3                         \n"
    "jg         1b                               \n"
    "vzeroupper                                  \n"
  : "+r"(src_uv),     // %0
    "+r"(dst_u),      // %1
    "+r"(dst_v),      // %2
    "+r"(width)         // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
  );
}
#endif  // HAS_SPLITUVROW_AVX2

#ifdef HAS_SPLITUVROW_SSE2
void SplitUVRow_SSE2(const uint8* src_uv, uint8* dst_u, uint8* dst_v,
                     int width) {
  asm volatile (
    "pcmpeqb    %%xmm5,%%xmm5                    \n"
    "psrlw      $0x8,%%xmm5                      \n"
    "sub        %1,%2                            \n"
    LABELALIGN
  "1:                                            \n"
    "movdqu     " MEMACCESS(0) ",%%xmm0          \n"
    "movdqu     " MEMACCESS2(0x10,0) ",%%xmm1    \n"
    "lea        " MEMLEA(0x20,0) ",%0            \n"
    "movdqa     %%xmm0,%%xmm2                    \n"
    "movdqa     %%xmm1,%%xmm3                    \n"
    "pand       %%xmm5,%%xmm0                    \n"
    "pand       %%xmm5,%%xmm1                    \n"
    "packuswb   %%xmm1,%%xmm0                    \n"
    "psrlw      $0x8,%%xmm2                      \n"
    "psrlw      $0x8,%%xmm3                      \n"
    "packuswb   %%xmm3,%%xmm2                    \n"
    "movdqu     %%xmm0," MEMACCESS(1) "          \n"
    MEMOPMEM(movdqu,xmm2,0x00,1,2,1)             //  movdqu     %%xmm2,(%1,%2)
    "lea        " MEMLEA(0x10,1) ",%1            \n"
    "sub        $0x10,%3                         \n"
    "jg         1b                               \n"
  : "+r"(src_uv),     // %0
    "+r"(dst_u),      // %1
    "+r"(dst_v),      // %2
    "+r"(width)         // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
  );
}
#endif  // HAS_SPLITUVROW_SSE2

#ifdef HAS_MERGEUVROW_AVX2
void MergeUVRow_AVX2(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                     int width) {
  asm volatile (
    "sub       %0,%1                             \n"
    LABELALIGN
  "1:                                            \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm0           \n"
    MEMOPREG(vmovdqu,0x00,0,1,1,ymm1)             //  vmovdqu (%0,%1,1),%%ymm1
    "lea       " MEMLEA(0x20,0) ",%0             \n"
    "vpunpcklbw %%ymm1,%%ymm0,%%ymm2             \n"
    "vpunpckhbw %%ymm1,%%ymm0,%%ymm0             \n"
    "vextractf128 $0x0,%%ymm2," MEMACCESS(2) "   \n"
    "vextractf128 $0x0,%%ymm0," MEMACCESS2(0x10,2) "\n"
    "vextractf128 $0x1,%%ymm2," MEMACCESS2(0x20,2) "\n"
    "vextractf128 $0x1,%%ymm0," MEMACCESS2(0x30,2) "\n"
    "lea       " MEMLEA(0x40,2) ",%2             \n"
    "sub       $0x20,%3                          \n"
    "jg        1b                                \n"
    "vzeroupper                                  \n"
  : "+r"(src_u),     // %0
    "+r"(src_v),     // %1
    "+r"(dst_uv),    // %2
    "+r"(width)      // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2"
  );
}
#endif  // HAS_MERGEUVROW_AVX2

#ifdef HAS_MERGEUVROW_SSE2
void MergeUVRow_SSE2(const uint8* src_u, const uint8* src_v, uint8* dst_uv,
                     int width) {
  asm volatile (
    "sub       %0,%1                             \n"
    LABELALIGN
  "1:                                            \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0           \n"
    MEMOPREG(movdqu,0x00,0,1,1,xmm1)             //  movdqu    (%0,%1,1),%%xmm1
    "lea       " MEMLEA(0x10,0) ",%0             \n"
    "movdqa    %%xmm0,%%xmm2                     \n"
    "punpcklbw %%xmm1,%%xmm0                     \n"
    "punpckhbw %%xmm1,%%xmm2                     \n"
    "movdqu    %%xmm0," MEMACCESS(2) "           \n"
    "movdqu    %%xmm2," MEMACCESS2(0x10,2) "     \n"
    "lea       " MEMLEA(0x20,2) ",%2             \n"
    "sub       $0x10,%3                          \n"
    "jg        1b                                \n"
  : "+r"(src_u),     // %0
    "+r"(src_v),     // %1
    "+r"(dst_uv),    // %2
    "+r"(width)      // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2"
  );
}
#endif  // HAS_MERGEUVROW_SSE2

#ifdef HAS_COPYROW_SSE2
void CopyRow_SSE2(const uint8* src, uint8* dst, int count) {
  asm volatile (
    "test       $0xf,%0                        \n"
    "jne        2f                             \n"
    "test       $0xf,%1                        \n"
    "jne        2f                             \n"
    LABELALIGN
  "1:                                          \n"
    "movdqa    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqa    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "movdqa    %%xmm0," MEMACCESS(1) "         \n"
    "movdqa    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x20,%2                        \n"
    "jg        1b                              \n"
    "jmp       9f                              \n"
    LABELALIGN
  "2:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x20,%2                        \n"
    "jg        2b                              \n"
  "9:                                          \n"
  : "+r"(src),   // %0
    "+r"(dst),   // %1
    "+r"(count)  // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1"
  );
}
#endif  // HAS_COPYROW_SSE2

#ifdef HAS_COPYROW_AVX
void CopyRow_AVX(const uint8* src, uint8* dst, int count) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm0         \n"
    "vmovdqu   " MEMACCESS2(0x20,0) ",%%ymm1   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "vmovdqu   %%ymm0," MEMACCESS(1) "         \n"
    "vmovdqu   %%ymm1," MEMACCESS2(0x20,1) "   \n"
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "sub       $0x40,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src),   // %0
    "+r"(dst),   // %1
    "+r"(count)  // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1"
  );
}
#endif  // HAS_COPYROW_AVX

#ifdef HAS_COPYROW_ERMS
// Multiple of 1.
void CopyRow_ERMS(const uint8* src, uint8* dst, int width) {
  size_t width_tmp = (size_t)(width);
  asm volatile (
    "rep movsb " MEMMOVESTRING(0,1) "          \n"
  : "+S"(src),  // %0
    "+D"(dst),  // %1
    "+c"(width_tmp) // %2
  :
  : "memory", "cc"
  );
}
#endif  // HAS_COPYROW_ERMS

#ifdef HAS_ARGBCOPYALPHAROW_SSE2
// width in pixels
void ARGBCopyAlphaRow_SSE2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "pcmpeqb   %%xmm0,%%xmm0                   \n"
    "pslld     $0x18,%%xmm0                    \n"
    "pcmpeqb   %%xmm1,%%xmm1                   \n"
    "psrld     $0x8,%%xmm1                     \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm2         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm3   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "movdqu    " MEMACCESS(1) ",%%xmm4         \n"
    "movdqu    " MEMACCESS2(0x10,1) ",%%xmm5   \n"
    "pand      %%xmm0,%%xmm2                   \n"
    "pand      %%xmm0,%%xmm3                   \n"
    "pand      %%xmm1,%%xmm4                   \n"
    "pand      %%xmm1,%%xmm5                   \n"
    "por       %%xmm4,%%xmm2                   \n"
    "por       %%xmm5,%%xmm3                   \n"
    "movdqu    %%xmm2," MEMACCESS(1) "         \n"
    "movdqu    %%xmm3," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src),   // %0
    "+r"(dst),   // %1
    "+r"(width)  // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_ARGBCOPYALPHAROW_SSE2

#ifdef HAS_ARGBCOPYALPHAROW_AVX2
// width in pixels
void ARGBCopyAlphaRow_AVX2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "vpcmpeqb  %%ymm0,%%ymm0,%%ymm0            \n"
    "vpsrld    $0x8,%%ymm0,%%ymm0              \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm1         \n"
    "vmovdqu   " MEMACCESS2(0x20,0) ",%%ymm2   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "vpblendvb %%ymm0," MEMACCESS(1) ",%%ymm1,%%ymm1        \n"
    "vpblendvb %%ymm0," MEMACCESS2(0x20,1) ",%%ymm2,%%ymm2  \n"
    "vmovdqu   %%ymm1," MEMACCESS(1) "         \n"
    "vmovdqu   %%ymm2," MEMACCESS2(0x20,1) "   \n"
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src),   // %0
    "+r"(dst),   // %1
    "+r"(width)  // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2"
  );
}
#endif  // HAS_ARGBCOPYALPHAROW_AVX2

#ifdef HAS_ARGBEXTRACTALPHAROW_SSE2
// width in pixels
void ARGBExtractAlphaRow_SSE2(const uint8* src_argb, uint8* dst_a, int width) {
 asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ", %%xmm0        \n"
    "movdqu    " MEMACCESS2(0x10, 0) ", %%xmm1 \n"
    "lea       " MEMLEA(0x20, 0) ", %0         \n"
    "psrld     $0x18, %%xmm0                   \n"
    "psrld     $0x18, %%xmm1                   \n"
    "packssdw  %%xmm1, %%xmm0                  \n"
    "packuswb  %%xmm0, %%xmm0                  \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x8, 1) ", %1          \n"
    "sub       $0x8, %2                        \n"
    "jg        1b                              \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_a),     // %1
    "+rm"(width)     // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1"
  );
}
#endif  // HAS_ARGBEXTRACTALPHAROW_SSE2

#ifdef HAS_ARGBCOPYYTOALPHAROW_SSE2
// width in pixels
void ARGBCopyYToAlphaRow_SSE2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "pcmpeqb   %%xmm0,%%xmm0                   \n"
    "pslld     $0x18,%%xmm0                    \n"
    "pcmpeqb   %%xmm1,%%xmm1                   \n"
    "psrld     $0x8,%%xmm1                     \n"
    LABELALIGN
  "1:                                          \n"
    "movq      " MEMACCESS(0) ",%%xmm2         \n"
    "lea       " MEMLEA(0x8,0) ",%0            \n"
    "punpcklbw %%xmm2,%%xmm2                   \n"
    "punpckhwd %%xmm2,%%xmm3                   \n"
    "punpcklwd %%xmm2,%%xmm2                   \n"
    "movdqu    " MEMACCESS(1) ",%%xmm4         \n"
    "movdqu    " MEMACCESS2(0x10,1) ",%%xmm5   \n"
    "pand      %%xmm0,%%xmm2                   \n"
    "pand      %%xmm0,%%xmm3                   \n"
    "pand      %%xmm1,%%xmm4                   \n"
    "pand      %%xmm1,%%xmm5                   \n"
    "por       %%xmm4,%%xmm2                   \n"
    "por       %%xmm5,%%xmm3                   \n"
    "movdqu    %%xmm2," MEMACCESS(1) "         \n"
    "movdqu    %%xmm3," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src),   // %0
    "+r"(dst),   // %1
    "+r"(width)  // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_ARGBCOPYYTOALPHAROW_SSE2

#ifdef HAS_ARGBCOPYYTOALPHAROW_AVX2
// width in pixels
void ARGBCopyYToAlphaRow_AVX2(const uint8* src, uint8* dst, int width) {
  asm volatile (
    "vpcmpeqb  %%ymm0,%%ymm0,%%ymm0            \n"
    "vpsrld    $0x8,%%ymm0,%%ymm0              \n"
    LABELALIGN
  "1:                                          \n"
    "vpmovzxbd " MEMACCESS(0) ",%%ymm1         \n"
    "vpmovzxbd " MEMACCESS2(0x8,0) ",%%ymm2    \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "vpslld    $0x18,%%ymm1,%%ymm1             \n"
    "vpslld    $0x18,%%ymm2,%%ymm2             \n"
    "vpblendvb %%ymm0," MEMACCESS(1) ",%%ymm1,%%ymm1        \n"
    "vpblendvb %%ymm0," MEMACCESS2(0x20,1) ",%%ymm2,%%ymm2  \n"
    "vmovdqu   %%ymm1," MEMACCESS(1) "         \n"
    "vmovdqu   %%ymm2," MEMACCESS2(0x20,1) "   \n"
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src),   // %0
    "+r"(dst),   // %1
    "+r"(width)  // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2"
  );
}
#endif  // HAS_ARGBCOPYYTOALPHAROW_AVX2

#ifdef HAS_SETROW_X86
void SetRow_X86(uint8* dst, uint8 v8, int width) {
  size_t width_tmp = (size_t)(width >> 2);
  const uint32 v32 = v8 * 0x01010101u;  // Duplicate byte to all bytes.
  asm volatile (
    "rep stosl " MEMSTORESTRING(eax,0) "       \n"
    : "+D"(dst),       // %0
      "+c"(width_tmp)  // %1
    : "a"(v32)         // %2
    : "memory", "cc");
}

void SetRow_ERMS(uint8* dst, uint8 v8, int width) {
  size_t width_tmp = (size_t)(width);
  asm volatile (
    "rep stosb " MEMSTORESTRING(al,0) "        \n"
    : "+D"(dst),       // %0
      "+c"(width_tmp)  // %1
    : "a"(v8)          // %2
    : "memory", "cc");
}

void ARGBSetRow_X86(uint8* dst_argb, uint32 v32, int width) {
  size_t width_tmp = (size_t)(width);
  asm volatile (
    "rep stosl " MEMSTORESTRING(eax,0) "       \n"
    : "+D"(dst_argb),  // %0
      "+c"(width_tmp)  // %1
    : "a"(v32)         // %2
    : "memory", "cc");
}
#endif  // HAS_SETROW_X86

#ifdef HAS_YUY2TOYROW_SSE2
void YUY2ToYRow_SSE2(const uint8* src_yuy2, uint8* dst_y, int width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrlw     $0x8,%%xmm5                     \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pand      %%xmm5,%%xmm0                   \n"
    "pand      %%xmm5,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_yuy2),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm5"
  );
}

void YUY2ToUVRow_SSE2(const uint8* src_yuy2, int stride_yuy2,
                      uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrlw     $0x8,%%xmm5                     \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqu,0x00,0,4,1,xmm2)           //  movdqu  (%0,%4,1),%%xmm2
    MEMOPREG(movdqu,0x10,0,4,1,xmm3)           //  movdqu  0x10(%0,%4,1),%%xmm3
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "pavgb     %%xmm3,%%xmm1                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "pand      %%xmm5,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm1                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movq,xmm1,0x00,1,2,1)             //  movq    %%xmm1,(%1,%2)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_yuy2),    // %0
    "+r"(dst_u),       // %1
    "+r"(dst_v),       // %2
    "+r"(width)          // %3
  : "r"((intptr_t)(stride_yuy2))  // %4
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
  );
}

void YUY2ToUV422Row_SSE2(const uint8* src_yuy2,
                         uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrlw     $0x8,%%xmm5                     \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "pand      %%xmm5,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm1                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movq,xmm1,0x00,1,2,1)             //  movq    %%xmm1,(%1,%2)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_yuy2),    // %0
    "+r"(dst_u),       // %1
    "+r"(dst_v),       // %2
    "+r"(width)          // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm5"
  );
}

void UYVYToYRow_SSE2(const uint8* src_uyvy, uint8* dst_y, int width) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
  : "+r"(src_uyvy),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1"
  );
}

void UYVYToUVRow_SSE2(const uint8* src_uyvy, int stride_uyvy,
                      uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrlw     $0x8,%%xmm5                     \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    MEMOPREG(movdqu,0x00,0,4,1,xmm2)           //  movdqu  (%0,%4,1),%%xmm2
    MEMOPREG(movdqu,0x10,0,4,1,xmm3)           //  movdqu  0x10(%0,%4,1),%%xmm3
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pavgb     %%xmm2,%%xmm0                   \n"
    "pavgb     %%xmm3,%%xmm1                   \n"
    "pand      %%xmm5,%%xmm0                   \n"
    "pand      %%xmm5,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "pand      %%xmm5,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm1                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movq,xmm1,0x00,1,2,1)             //  movq    %%xmm1,(%1,%2)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_uyvy),    // %0
    "+r"(dst_u),       // %1
    "+r"(dst_v),       // %2
    "+r"(width)          // %3
  : "r"((intptr_t)(stride_uyvy))  // %4
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
  );
}

void UYVYToUV422Row_SSE2(const uint8* src_uyvy,
                         uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psrlw     $0x8,%%xmm5                     \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pand      %%xmm5,%%xmm0                   \n"
    "pand      %%xmm5,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "pand      %%xmm5,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm1                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    MEMOPMEM(movq,xmm1,0x00,1,2,1)             //  movq    %%xmm1,(%1,%2)
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_uyvy),    // %0
    "+r"(dst_u),       // %1
    "+r"(dst_v),       // %2
    "+r"(width)          // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm5"
  );
}
#endif  // HAS_YUY2TOYROW_SSE2

#ifdef HAS_YUY2TOYROW_AVX2
void YUY2ToYRow_AVX2(const uint8* src_yuy2, uint8* dst_y, int width) {
  asm volatile (
    "vpcmpeqb  %%ymm5,%%ymm5,%%ymm5            \n"
    "vpsrlw    $0x8,%%ymm5,%%ymm5              \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm0         \n"
    "vmovdqu   " MEMACCESS2(0x20,0) ",%%ymm1   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "vpand     %%ymm5,%%ymm0,%%ymm0            \n"
    "vpand     %%ymm5,%%ymm1,%%ymm1            \n"
    "vpackuswb %%ymm1,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vmovdqu   %%ymm0," MEMACCESS(1) "         \n"
    "lea      " MEMLEA(0x20,1) ",%1            \n"
    "sub       $0x20,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_yuy2),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm5"
  );
}

void YUY2ToUVRow_AVX2(const uint8* src_yuy2, int stride_yuy2,
                      uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "vpcmpeqb  %%ymm5,%%ymm5,%%ymm5            \n"
    "vpsrlw    $0x8,%%ymm5,%%ymm5              \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm0         \n"
    "vmovdqu   " MEMACCESS2(0x20,0) ",%%ymm1   \n"
    VMEMOPREG(vpavgb,0x00,0,4,1,ymm0,ymm0)     // vpavgb (%0,%4,1),%%ymm0,%%ymm0
    VMEMOPREG(vpavgb,0x20,0,4,1,ymm1,ymm1)
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "vpsrlw    $0x8,%%ymm0,%%ymm0              \n"
    "vpsrlw    $0x8,%%ymm1,%%ymm1              \n"
    "vpackuswb %%ymm1,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vpand     %%ymm5,%%ymm0,%%ymm1            \n"
    "vpsrlw    $0x8,%%ymm0,%%ymm0              \n"
    "vpackuswb %%ymm1,%%ymm1,%%ymm1            \n"
    "vpackuswb %%ymm0,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm1,%%ymm1             \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vextractf128 $0x0,%%ymm1," MEMACCESS(1) " \n"
    VEXTOPMEM(vextractf128,0,ymm0,0x00,1,2,1) // vextractf128 $0x0,%%ymm0,(%1,%2,1)
    "lea      " MEMLEA(0x10,1) ",%1            \n"
    "sub       $0x20,%3                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_yuy2),    // %0
    "+r"(dst_u),       // %1
    "+r"(dst_v),       // %2
    "+r"(width)          // %3
  : "r"((intptr_t)(stride_yuy2))  // %4
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm5"
  );
}

void YUY2ToUV422Row_AVX2(const uint8* src_yuy2,
                         uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "vpcmpeqb  %%ymm5,%%ymm5,%%ymm5            \n"
    "vpsrlw    $0x8,%%ymm5,%%ymm5              \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm0         \n"
    "vmovdqu   " MEMACCESS2(0x20,0) ",%%ymm1   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "vpsrlw    $0x8,%%ymm0,%%ymm0              \n"
    "vpsrlw    $0x8,%%ymm1,%%ymm1              \n"
    "vpackuswb %%ymm1,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vpand     %%ymm5,%%ymm0,%%ymm1            \n"
    "vpsrlw    $0x8,%%ymm0,%%ymm0              \n"
    "vpackuswb %%ymm1,%%ymm1,%%ymm1            \n"
    "vpackuswb %%ymm0,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm1,%%ymm1             \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vextractf128 $0x0,%%ymm1," MEMACCESS(1) " \n"
    VEXTOPMEM(vextractf128,0,ymm0,0x00,1,2,1) // vextractf128 $0x0,%%ymm0,(%1,%2,1)
    "lea      " MEMLEA(0x10,1) ",%1            \n"
    "sub       $0x20,%3                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_yuy2),    // %0
    "+r"(dst_u),       // %1
    "+r"(dst_v),       // %2
    "+r"(width)          // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm5"
  );
}

void UYVYToYRow_AVX2(const uint8* src_uyvy, uint8* dst_y, int width) {
  asm volatile (
    LABELALIGN
  "1:                                          \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm0         \n"
    "vmovdqu   " MEMACCESS2(0x20,0) ",%%ymm1   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "vpsrlw    $0x8,%%ymm0,%%ymm0              \n"
    "vpsrlw    $0x8,%%ymm1,%%ymm1              \n"
    "vpackuswb %%ymm1,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vmovdqu   %%ymm0," MEMACCESS(1) "         \n"
    "lea      " MEMLEA(0x20,1) ",%1            \n"
    "sub       $0x20,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_uyvy),  // %0
    "+r"(dst_y),     // %1
    "+r"(width)        // %2
  :
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm5"
  );
}
void UYVYToUVRow_AVX2(const uint8* src_uyvy, int stride_uyvy,
                      uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "vpcmpeqb  %%ymm5,%%ymm5,%%ymm5            \n"
    "vpsrlw    $0x8,%%ymm5,%%ymm5              \n"
    "sub       %1,%2                           \n"

    LABELALIGN
  "1:                                          \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm0         \n"
    "vmovdqu   " MEMACCESS2(0x20,0) ",%%ymm1   \n"
    VMEMOPREG(vpavgb,0x00,0,4,1,ymm0,ymm0)     // vpavgb (%0,%4,1),%%ymm0,%%ymm0
    VMEMOPREG(vpavgb,0x20,0,4,1,ymm1,ymm1)
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "vpand     %%ymm5,%%ymm0,%%ymm0            \n"
    "vpand     %%ymm5,%%ymm1,%%ymm1            \n"
    "vpackuswb %%ymm1,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vpand     %%ymm5,%%ymm0,%%ymm1            \n"
    "vpsrlw    $0x8,%%ymm0,%%ymm0              \n"
    "vpackuswb %%ymm1,%%ymm1,%%ymm1            \n"
    "vpackuswb %%ymm0,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm1,%%ymm1             \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vextractf128 $0x0,%%ymm1," MEMACCESS(1) " \n"
    VEXTOPMEM(vextractf128,0,ymm0,0x00,1,2,1) // vextractf128 $0x0,%%ymm0,(%1,%2,1)
    "lea      " MEMLEA(0x10,1) ",%1            \n"
    "sub       $0x20,%3                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_uyvy),    // %0
    "+r"(dst_u),       // %1
    "+r"(dst_v),       // %2
    "+r"(width)          // %3
  : "r"((intptr_t)(stride_uyvy))  // %4
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm5"
  );
}

void UYVYToUV422Row_AVX2(const uint8* src_uyvy,
                         uint8* dst_u, uint8* dst_v, int width) {
  asm volatile (
    "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5           \n"
    "vpsrlw     $0x8,%%ymm5,%%ymm5             \n"
    "sub       %1,%2                           \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm0         \n"
    "vmovdqu   " MEMACCESS2(0x20,0) ",%%ymm1   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "vpand     %%ymm5,%%ymm0,%%ymm0            \n"
    "vpand     %%ymm5,%%ymm1,%%ymm1            \n"
    "vpackuswb %%ymm1,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vpand     %%ymm5,%%ymm0,%%ymm1            \n"
    "vpsrlw    $0x8,%%ymm0,%%ymm0              \n"
    "vpackuswb %%ymm1,%%ymm1,%%ymm1            \n"
    "vpackuswb %%ymm0,%%ymm0,%%ymm0            \n"
    "vpermq    $0xd8,%%ymm1,%%ymm1             \n"
    "vpermq    $0xd8,%%ymm0,%%ymm0             \n"
    "vextractf128 $0x0,%%ymm1," MEMACCESS(1) " \n"
    VEXTOPMEM(vextractf128,0,ymm0,0x00,1,2,1) // vextractf128 $0x0,%%ymm0,(%1,%2,1)
    "lea      " MEMLEA(0x10,1) ",%1            \n"
    "sub       $0x20,%3                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_uyvy),    // %0
    "+r"(dst_u),       // %1
    "+r"(dst_v),       // %2
    "+r"(width)          // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm5"
  );
}
#endif  // HAS_YUY2TOYROW_AVX2

#ifdef HAS_ARGBBLENDROW_SSSE3
// Shuffle table for isolating alpha.
static uvec8 kShuffleAlpha = {
  3u, 0x80, 3u, 0x80, 7u, 0x80, 7u, 0x80,
  11u, 0x80, 11u, 0x80, 15u, 0x80, 15u, 0x80
};

// Blend 8 pixels at a time
void ARGBBlendRow_SSSE3(const uint8* src_argb0, const uint8* src_argb1,
                        uint8* dst_argb, int width) {
  asm volatile (
    "pcmpeqb   %%xmm7,%%xmm7                   \n"
    "psrlw     $0xf,%%xmm7                     \n"
    "pcmpeqb   %%xmm6,%%xmm6                   \n"
    "psrlw     $0x8,%%xmm6                     \n"
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "psllw     $0x8,%%xmm5                     \n"
    "pcmpeqb   %%xmm4,%%xmm4                   \n"
    "pslld     $0x18,%%xmm4                    \n"
    "sub       $0x4,%3                         \n"
    "jl        49f                             \n"

    // 4 pixel loop.
    LABELALIGN
  "40:                                         \n"
    "movdqu    " MEMACCESS(0) ",%%xmm3         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqa    %%xmm3,%%xmm0                   \n"
    "pxor      %%xmm4,%%xmm3                   \n"
    "movdqu    " MEMACCESS(1) ",%%xmm2         \n"
    "pshufb    %4,%%xmm3                       \n"
    "pand      %%xmm6,%%xmm2                   \n"
    "paddw     %%xmm7,%%xmm3                   \n"
    "pmullw    %%xmm3,%%xmm2                   \n"
    "movdqu    " MEMACCESS(1) ",%%xmm1         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "por       %%xmm4,%%xmm0                   \n"
    "pmullw    %%xmm3,%%xmm1                   \n"
    "psrlw     $0x8,%%xmm2                     \n"
    "paddusb   %%xmm2,%%xmm0                   \n"
    "pand      %%xmm5,%%xmm1                   \n"
    "paddusb   %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "sub       $0x4,%3                         \n"
    "jge       40b                             \n"

  "49:                                         \n"
    "add       $0x3,%3                         \n"
    "jl        99f                             \n"

    // 1 pixel loop.
  "91:                                         \n"
    "movd      " MEMACCESS(0) ",%%xmm3         \n"
    "lea       " MEMLEA(0x4,0) ",%0            \n"
    "movdqa    %%xmm3,%%xmm0                   \n"
    "pxor      %%xmm4,%%xmm3                   \n"
    "movd      " MEMACCESS(1) ",%%xmm2         \n"
    "pshufb    %4,%%xmm3                       \n"
    "pand      %%xmm6,%%xmm2                   \n"
    "paddw     %%xmm7,%%xmm3                   \n"
    "pmullw    %%xmm3,%%xmm2                   \n"
    "movd      " MEMACCESS(1) ",%%xmm1         \n"
    "lea       " MEMLEA(0x4,1) ",%1            \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "por       %%xmm4,%%xmm0                   \n"
    "pmullw    %%xmm3,%%xmm1                   \n"
    "psrlw     $0x8,%%xmm2                     \n"
    "paddusb   %%xmm2,%%xmm0                   \n"
    "pand      %%xmm5,%%xmm1                   \n"
    "paddusb   %%xmm1,%%xmm0                   \n"
    "movd      %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x4,2) ",%2            \n"
    "sub       $0x1,%3                         \n"
    "jge       91b                             \n"
  "99:                                         \n"
  : "+r"(src_argb0),    // %0
    "+r"(src_argb1),    // %1
    "+r"(dst_argb),     // %2
    "+r"(width)         // %3
  : "m"(kShuffleAlpha)  // %4
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBBLENDROW_SSSE3

#ifdef HAS_BLENDPLANEROW_SSSE3
// Blend 8 pixels at a time.
// unsigned version of math
// =((A2*C2)+(B2*(255-C2))+255)/256
// signed version of math
// =(((A2-128)*C2)+((B2-128)*(255-C2))+32768+127)/256
void BlendPlaneRow_SSSE3(const uint8* src0, const uint8* src1,
                         const uint8* alpha, uint8* dst, int width) {
  asm volatile (
    "pcmpeqb    %%xmm5,%%xmm5                  \n"
    "psllw      $0x8,%%xmm5                    \n"
    "mov        $0x80808080,%%eax              \n"
    "movd       %%eax,%%xmm6                   \n"
    "pshufd     $0x0,%%xmm6,%%xmm6             \n"
    "mov        $0x807f807f,%%eax              \n"
    "movd       %%eax,%%xmm7                   \n"
    "pshufd     $0x0,%%xmm7,%%xmm7             \n"
    "sub        %2,%0                          \n"
    "sub        %2,%1                          \n"
    "sub        %2,%3                          \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movq       (%2),%%xmm0                    \n"
    "punpcklbw  %%xmm0,%%xmm0                  \n"
    "pxor       %%xmm5,%%xmm0                  \n"
    "movq       (%0,%2,1),%%xmm1               \n"
    "movq       (%1,%2,1),%%xmm2               \n"
    "punpcklbw  %%xmm2,%%xmm1                  \n"
    "psubb      %%xmm6,%%xmm1                  \n"
    "pmaddubsw  %%xmm1,%%xmm0                  \n"
    "paddw      %%xmm7,%%xmm0                  \n"
    "psrlw      $0x8,%%xmm0                    \n"
    "packuswb   %%xmm0,%%xmm0                  \n"
    "movq       %%xmm0,(%3,%2,1)               \n"
    "lea        0x8(%2),%2                     \n"
    "sub        $0x8,%4                        \n"
    "jg        1b                              \n"
  : "+r"(src0),       // %0
    "+r"(src1),       // %1
    "+r"(alpha),      // %2
    "+r"(dst),        // %3
    "+rm"(width)      // %4
  :: "memory", "cc", "eax", "xmm0", "xmm1", "xmm2", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_BLENDPLANEROW_SSSE3

#ifdef HAS_BLENDPLANEROW_AVX2
// Blend 32 pixels at a time.
// unsigned version of math
// =((A2*C2)+(B2*(255-C2))+255)/256
// signed version of math
// =(((A2-128)*C2)+((B2-128)*(255-C2))+32768+127)/256
void BlendPlaneRow_AVX2(const uint8* src0, const uint8* src1,
                        const uint8* alpha, uint8* dst, int width) {
  asm volatile (
    "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5           \n"
    "vpsllw     $0x8,%%ymm5,%%ymm5             \n"
    "mov        $0x80808080,%%eax              \n"
    "vmovd      %%eax,%%xmm6                   \n"
    "vbroadcastss %%xmm6,%%ymm6                \n"
    "mov        $0x807f807f,%%eax              \n"
    "vmovd      %%eax,%%xmm7                   \n"
    "vbroadcastss %%xmm7,%%ymm7                \n"
    "sub        %2,%0                          \n"
    "sub        %2,%1                          \n"
    "sub        %2,%3                          \n"

    // 32 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    (%2),%%ymm0                    \n"
    "vpunpckhbw %%ymm0,%%ymm0,%%ymm3           \n"
    "vpunpcklbw %%ymm0,%%ymm0,%%ymm0           \n"
    "vpxor      %%ymm5,%%ymm3,%%ymm3           \n"
    "vpxor      %%ymm5,%%ymm0,%%ymm0           \n"
    "vmovdqu    (%0,%2,1),%%ymm1               \n"
    "vmovdqu    (%1,%2,1),%%ymm2               \n"
    "vpunpckhbw %%ymm2,%%ymm1,%%ymm4           \n"
    "vpunpcklbw %%ymm2,%%ymm1,%%ymm1           \n"
    "vpsubb     %%ymm6,%%ymm4,%%ymm4           \n"
    "vpsubb     %%ymm6,%%ymm1,%%ymm1           \n"
    "vpmaddubsw %%ymm4,%%ymm3,%%ymm3           \n"
    "vpmaddubsw %%ymm1,%%ymm0,%%ymm0           \n"
    "vpaddw     %%ymm7,%%ymm3,%%ymm3           \n"
    "vpaddw     %%ymm7,%%ymm0,%%ymm0           \n"
    "vpsrlw     $0x8,%%ymm3,%%ymm3             \n"
    "vpsrlw     $0x8,%%ymm0,%%ymm0             \n"
    "vpackuswb  %%ymm3,%%ymm0,%%ymm0           \n"
    "vmovdqu    %%ymm0,(%3,%2,1)               \n"
    "lea        0x20(%2),%2                    \n"
    "sub        $0x20,%4                       \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src0),       // %0
    "+r"(src1),       // %1
    "+r"(alpha),      // %2
    "+r"(dst),        // %3
    "+rm"(width)      // %4
  :: "memory", "cc", "eax",
     "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_BLENDPLANEROW_AVX2

#ifdef HAS_ARGBATTENUATEROW_SSSE3
// Shuffle table duplicating alpha
static uvec8 kShuffleAlpha0 = {
  3u, 3u, 3u, 3u, 3u, 3u, 128u, 128u, 7u, 7u, 7u, 7u, 7u, 7u, 128u, 128u
};
static uvec8 kShuffleAlpha1 = {
  11u, 11u, 11u, 11u, 11u, 11u, 128u, 128u,
  15u, 15u, 15u, 15u, 15u, 15u, 128u, 128u
};
// Attenuate 4 pixels at a time.
void ARGBAttenuateRow_SSSE3(const uint8* src_argb, uint8* dst_argb, int width) {
  asm volatile (
    "pcmpeqb   %%xmm3,%%xmm3                   \n"
    "pslld     $0x18,%%xmm3                    \n"
    "movdqa    %3,%%xmm4                       \n"
    "movdqa    %4,%%xmm5                       \n"

    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "pshufb    %%xmm4,%%xmm0                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm1         \n"
    "punpcklbw %%xmm1,%%xmm1                   \n"
    "pmulhuw   %%xmm1,%%xmm0                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm1         \n"
    "pshufb    %%xmm5,%%xmm1                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm2         \n"
    "punpckhbw %%xmm2,%%xmm2                   \n"
    "pmulhuw   %%xmm2,%%xmm1                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm2         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "pand      %%xmm3,%%xmm2                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "por       %%xmm2,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x4,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb),    // %0
    "+r"(dst_argb),    // %1
    "+r"(width)        // %2
  : "m"(kShuffleAlpha0),  // %3
    "m"(kShuffleAlpha1)  // %4
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_ARGBATTENUATEROW_SSSE3

#ifdef HAS_ARGBATTENUATEROW_AVX2
// Shuffle table duplicating alpha.
static const uvec8 kShuffleAlpha_AVX2 = {
  6u, 7u, 6u, 7u, 6u, 7u, 128u, 128u, 14u, 15u, 14u, 15u, 14u, 15u, 128u, 128u
};
// Attenuate 8 pixels at a time.
void ARGBAttenuateRow_AVX2(const uint8* src_argb, uint8* dst_argb, int width) {
  asm volatile (
    "vbroadcastf128 %3,%%ymm4                  \n"
    "vpcmpeqb   %%ymm5,%%ymm5,%%ymm5           \n"
    "vpslld     $0x18,%%ymm5,%%ymm5            \n"
    "sub        %0,%1                          \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    " MEMACCESS(0) ",%%ymm6        \n"
    "vpunpcklbw %%ymm6,%%ymm6,%%ymm0           \n"
    "vpunpckhbw %%ymm6,%%ymm6,%%ymm1           \n"
    "vpshufb    %%ymm4,%%ymm0,%%ymm2           \n"
    "vpshufb    %%ymm4,%%ymm1,%%ymm3           \n"
    "vpmulhuw   %%ymm2,%%ymm0,%%ymm0           \n"
    "vpmulhuw   %%ymm3,%%ymm1,%%ymm1           \n"
    "vpand      %%ymm5,%%ymm6,%%ymm6           \n"
    "vpsrlw     $0x8,%%ymm0,%%ymm0             \n"
    "vpsrlw     $0x8,%%ymm1,%%ymm1             \n"
    "vpackuswb  %%ymm1,%%ymm0,%%ymm0           \n"
    "vpor       %%ymm6,%%ymm0,%%ymm0           \n"
    MEMOPMEM(vmovdqu,ymm0,0x00,0,1,1)          //  vmovdqu %%ymm0,(%0,%1)
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "sub        $0x8,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb),    // %0
    "+r"(dst_argb),    // %1
    "+r"(width)        // %2
  : "m"(kShuffleAlpha_AVX2)  // %3
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}
#endif  // HAS_ARGBATTENUATEROW_AVX2

#ifdef HAS_ARGBUNATTENUATEROW_SSE2
// Unattenuate 4 pixels at a time.
void ARGBUnattenuateRow_SSE2(const uint8* src_argb, uint8* dst_argb,
                             int width) {
  uintptr_t alpha;
  asm volatile (
    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movzb     " MEMACCESS2(0x03,0) ",%3       \n"
    "punpcklbw %%xmm0,%%xmm0                   \n"
    MEMOPREG(movd,0x00,4,3,4,xmm2)             //  movd      0x0(%4,%3,4),%%xmm2
    "movzb     " MEMACCESS2(0x07,0) ",%3       \n"
    MEMOPREG(movd,0x00,4,3,4,xmm3)             //  movd      0x0(%4,%3,4),%%xmm3
    "pshuflw   $0x40,%%xmm2,%%xmm2             \n"
    "pshuflw   $0x40,%%xmm3,%%xmm3             \n"
    "movlhps   %%xmm3,%%xmm2                   \n"
    "pmulhuw   %%xmm2,%%xmm0                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm1         \n"
    "movzb     " MEMACCESS2(0x0b,0) ",%3       \n"
    "punpckhbw %%xmm1,%%xmm1                   \n"
    MEMOPREG(movd,0x00,4,3,4,xmm2)             //  movd      0x0(%4,%3,4),%%xmm2
    "movzb     " MEMACCESS2(0x0f,0) ",%3       \n"
    MEMOPREG(movd,0x00,4,3,4,xmm3)             //  movd      0x0(%4,%3,4),%%xmm3
    "pshuflw   $0x40,%%xmm2,%%xmm2             \n"
    "pshuflw   $0x40,%%xmm3,%%xmm3             \n"
    "movlhps   %%xmm3,%%xmm2                   \n"
    "pmulhuw   %%xmm2,%%xmm1                   \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x4,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb),     // %0
    "+r"(dst_argb),     // %1
    "+r"(width),        // %2
    "=&r"(alpha)        // %3
  : "r"(fixed_invtbl8)  // %4
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_ARGBUNATTENUATEROW_SSE2

#ifdef HAS_ARGBUNATTENUATEROW_AVX2
// Shuffle table duplicating alpha.
static const uvec8 kUnattenShuffleAlpha_AVX2 = {
  0u, 1u, 0u, 1u, 0u, 1u, 6u, 7u, 8u, 9u, 8u, 9u, 8u, 9u, 14u, 15u
};
// Unattenuate 8 pixels at a time.
void ARGBUnattenuateRow_AVX2(const uint8* src_argb, uint8* dst_argb,
                             int width) {
  uintptr_t alpha;
  asm volatile (
    "sub        %0,%1                          \n"
    "vbroadcastf128 %5,%%ymm5                  \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    // replace VPGATHER
    "movzb     " MEMACCESS2(0x03,0) ",%3       \n"
    MEMOPREG(vmovd,0x00,4,3,4,xmm0)             //  vmovd 0x0(%4,%3,4),%%xmm0
    "movzb     " MEMACCESS2(0x07,0) ",%3       \n"
    MEMOPREG(vmovd,0x00,4,3,4,xmm1)             //  vmovd 0x0(%4,%3,4),%%xmm1
    "movzb     " MEMACCESS2(0x0b,0) ",%3       \n"
    "vpunpckldq %%xmm1,%%xmm0,%%xmm6           \n"
    MEMOPREG(vmovd,0x00,4,3,4,xmm2)             //  vmovd 0x0(%4,%3,4),%%xmm2
    "movzb     " MEMACCESS2(0x0f,0) ",%3       \n"
    MEMOPREG(vmovd,0x00,4,3,4,xmm3)             //  vmovd 0x0(%4,%3,4),%%xmm3
    "movzb     " MEMACCESS2(0x13,0) ",%3       \n"
    "vpunpckldq %%xmm3,%%xmm2,%%xmm7           \n"
    MEMOPREG(vmovd,0x00,4,3,4,xmm0)             //  vmovd 0x0(%4,%3,4),%%xmm0
    "movzb     " MEMACCESS2(0x17,0) ",%3       \n"
    MEMOPREG(vmovd,0x00,4,3,4,xmm1)             //  vmovd 0x0(%4,%3,4),%%xmm1
    "movzb     " MEMACCESS2(0x1b,0) ",%3       \n"
    "vpunpckldq %%xmm1,%%xmm0,%%xmm0           \n"
    MEMOPREG(vmovd,0x00,4,3,4,xmm2)             //  vmovd 0x0(%4,%3,4),%%xmm2
    "movzb     " MEMACCESS2(0x1f,0) ",%3       \n"
    MEMOPREG(vmovd,0x00,4,3,4,xmm3)             //  vmovd 0x0(%4,%3,4),%%xmm3
    "vpunpckldq %%xmm3,%%xmm2,%%xmm2           \n"
    "vpunpcklqdq %%xmm7,%%xmm6,%%xmm3          \n"
    "vpunpcklqdq %%xmm2,%%xmm0,%%xmm0          \n"
    "vinserti128 $0x1,%%xmm0,%%ymm3,%%ymm3     \n"
    // end of VPGATHER

    "vmovdqu    " MEMACCESS(0) ",%%ymm6        \n"
    "vpunpcklbw %%ymm6,%%ymm6,%%ymm0           \n"
    "vpunpckhbw %%ymm6,%%ymm6,%%ymm1           \n"
    "vpunpcklwd %%ymm3,%%ymm3,%%ymm2           \n"
    "vpunpckhwd %%ymm3,%%ymm3,%%ymm3           \n"
    "vpshufb    %%ymm5,%%ymm2,%%ymm2           \n"
    "vpshufb    %%ymm5,%%ymm3,%%ymm3           \n"
    "vpmulhuw   %%ymm2,%%ymm0,%%ymm0           \n"
    "vpmulhuw   %%ymm3,%%ymm1,%%ymm1           \n"
    "vpackuswb  %%ymm1,%%ymm0,%%ymm0           \n"
    MEMOPMEM(vmovdqu,ymm0,0x00,0,1,1)          //  vmovdqu %%ymm0,(%0,%1)
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "sub        $0x8,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb),      // %0
    "+r"(dst_argb),      // %1
    "+r"(width),         // %2
    "=&r"(alpha)         // %3
  : "r"(fixed_invtbl8),  // %4
    "m"(kUnattenShuffleAlpha_AVX2)  // %5
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBUNATTENUATEROW_AVX2

#ifdef HAS_ARGBGRAYROW_SSSE3
// Convert 8 ARGB pixels (64 bytes) to 8 Gray ARGB pixels
void ARGBGrayRow_SSSE3(const uint8* src_argb, uint8* dst_argb, int width) {
  asm volatile (
    "movdqa    %3,%%xmm4                       \n"
    "movdqa    %4,%%xmm5                       \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "pmaddubsw %%xmm4,%%xmm0                   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "phaddw    %%xmm1,%%xmm0                   \n"
    "paddw     %%xmm5,%%xmm0                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm2         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm3   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "psrld     $0x18,%%xmm2                    \n"
    "psrld     $0x18,%%xmm3                    \n"
    "packuswb  %%xmm3,%%xmm2                   \n"
    "packuswb  %%xmm2,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm3                   \n"
    "punpcklbw %%xmm0,%%xmm0                   \n"
    "punpcklbw %%xmm2,%%xmm3                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklwd %%xmm3,%%xmm0                   \n"
    "punpckhwd %%xmm3,%%xmm1                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb),   // %0
    "+r"(dst_argb),   // %1
    "+r"(width)       // %2
  : "m"(kARGBToYJ),   // %3
    "m"(kAddYJ64)     // %4
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_ARGBGRAYROW_SSSE3

#ifdef HAS_ARGBSEPIAROW_SSSE3
//    b = (r * 35 + g * 68 + b * 17) >> 7
//    g = (r * 45 + g * 88 + b * 22) >> 7
//    r = (r * 50 + g * 98 + b * 24) >> 7
// Constant for ARGB color to sepia tone
static vec8 kARGBToSepiaB = {
  17, 68, 35, 0, 17, 68, 35, 0, 17, 68, 35, 0, 17, 68, 35, 0
};

static vec8 kARGBToSepiaG = {
  22, 88, 45, 0, 22, 88, 45, 0, 22, 88, 45, 0, 22, 88, 45, 0
};

static vec8 kARGBToSepiaR = {
  24, 98, 50, 0, 24, 98, 50, 0, 24, 98, 50, 0, 24, 98, 50, 0
};

// Convert 8 ARGB pixels (32 bytes) to 8 Sepia ARGB pixels.
void ARGBSepiaRow_SSSE3(uint8* dst_argb, int width) {
  asm volatile (
    "movdqa    %2,%%xmm2                       \n"
    "movdqa    %3,%%xmm3                       \n"
    "movdqa    %4,%%xmm4                       \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm6   \n"
    "pmaddubsw %%xmm2,%%xmm0                   \n"
    "pmaddubsw %%xmm2,%%xmm6                   \n"
    "phaddw    %%xmm6,%%xmm0                   \n"
    "psrlw     $0x7,%%xmm0                     \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm5         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "pmaddubsw %%xmm3,%%xmm5                   \n"
    "pmaddubsw %%xmm3,%%xmm1                   \n"
    "phaddw    %%xmm1,%%xmm5                   \n"
    "psrlw     $0x7,%%xmm5                     \n"
    "packuswb  %%xmm5,%%xmm5                   \n"
    "punpcklbw %%xmm5,%%xmm0                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm5         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "pmaddubsw %%xmm4,%%xmm5                   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "phaddw    %%xmm1,%%xmm5                   \n"
    "psrlw     $0x7,%%xmm5                     \n"
    "packuswb  %%xmm5,%%xmm5                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm6         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "psrld     $0x18,%%xmm6                    \n"
    "psrld     $0x18,%%xmm1                    \n"
    "packuswb  %%xmm1,%%xmm6                   \n"
    "packuswb  %%xmm6,%%xmm6                   \n"
    "punpcklbw %%xmm6,%%xmm5                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklwd %%xmm5,%%xmm0                   \n"
    "punpckhwd %%xmm5,%%xmm1                   \n"
    "movdqu    %%xmm0," MEMACCESS(0) "         \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,0) "   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "sub       $0x8,%1                         \n"
    "jg        1b                              \n"
  : "+r"(dst_argb),      // %0
    "+r"(width)          // %1
  : "m"(kARGBToSepiaB),  // %2
    "m"(kARGBToSepiaG),  // %3
    "m"(kARGBToSepiaR)   // %4
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}
#endif  // HAS_ARGBSEPIAROW_SSSE3

#ifdef HAS_ARGBCOLORMATRIXROW_SSSE3
// Tranform 8 ARGB pixels (32 bytes) with color matrix.
// Same as Sepia except matrix is provided.
void ARGBColorMatrixRow_SSSE3(const uint8* src_argb, uint8* dst_argb,
                              const int8* matrix_argb, int width) {
  asm volatile (
    "movdqu    " MEMACCESS(3) ",%%xmm5         \n"
    "pshufd    $0x00,%%xmm5,%%xmm2             \n"
    "pshufd    $0x55,%%xmm5,%%xmm3             \n"
    "pshufd    $0xaa,%%xmm5,%%xmm4             \n"
    "pshufd    $0xff,%%xmm5,%%xmm5             \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm7   \n"
    "pmaddubsw %%xmm2,%%xmm0                   \n"
    "pmaddubsw %%xmm2,%%xmm7                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm6         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "pmaddubsw %%xmm3,%%xmm6                   \n"
    "pmaddubsw %%xmm3,%%xmm1                   \n"
    "phaddsw   %%xmm7,%%xmm0                   \n"
    "phaddsw   %%xmm1,%%xmm6                   \n"
    "psraw     $0x6,%%xmm0                     \n"
    "psraw     $0x6,%%xmm6                     \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "packuswb  %%xmm6,%%xmm6                   \n"
    "punpcklbw %%xmm6,%%xmm0                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm1         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm7   \n"
    "pmaddubsw %%xmm4,%%xmm1                   \n"
    "pmaddubsw %%xmm4,%%xmm7                   \n"
    "phaddsw   %%xmm7,%%xmm1                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm6         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm7   \n"
    "pmaddubsw %%xmm5,%%xmm6                   \n"
    "pmaddubsw %%xmm5,%%xmm7                   \n"
    "phaddsw   %%xmm7,%%xmm6                   \n"
    "psraw     $0x6,%%xmm1                     \n"
    "psraw     $0x6,%%xmm6                     \n"
    "packuswb  %%xmm1,%%xmm1                   \n"
    "packuswb  %%xmm6,%%xmm6                   \n"
    "punpcklbw %%xmm6,%%xmm1                   \n"
    "movdqa    %%xmm0,%%xmm6                   \n"
    "punpcklwd %%xmm1,%%xmm0                   \n"
    "punpckhwd %%xmm1,%%xmm6                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "movdqu    %%xmm6," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb),      // %0
    "+r"(dst_argb),      // %1
    "+r"(width)          // %2
  : "r"(matrix_argb)     // %3
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBCOLORMATRIXROW_SSSE3

#ifdef HAS_ARGBQUANTIZEROW_SSE2
// Quantize 4 ARGB pixels (16 bytes).
void ARGBQuantizeRow_SSE2(uint8* dst_argb, int scale, int interval_size,
                          int interval_offset, int width) {
  asm volatile (
    "movd      %2,%%xmm2                       \n"
    "movd      %3,%%xmm3                       \n"
    "movd      %4,%%xmm4                       \n"
    "pshuflw   $0x40,%%xmm2,%%xmm2             \n"
    "pshufd    $0x44,%%xmm2,%%xmm2             \n"
    "pshuflw   $0x40,%%xmm3,%%xmm3             \n"
    "pshufd    $0x44,%%xmm3,%%xmm3             \n"
    "pshuflw   $0x40,%%xmm4,%%xmm4             \n"
    "pshufd    $0x44,%%xmm4,%%xmm4             \n"
    "pxor      %%xmm5,%%xmm5                   \n"
    "pcmpeqb   %%xmm6,%%xmm6                   \n"
    "pslld     $0x18,%%xmm6                    \n"

    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "punpcklbw %%xmm5,%%xmm0                   \n"
    "pmulhuw   %%xmm2,%%xmm0                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm1         \n"
    "punpckhbw %%xmm5,%%xmm1                   \n"
    "pmulhuw   %%xmm2,%%xmm1                   \n"
    "pmullw    %%xmm3,%%xmm0                   \n"
    "movdqu    " MEMACCESS(0) ",%%xmm7         \n"
    "pmullw    %%xmm3,%%xmm1                   \n"
    "pand      %%xmm6,%%xmm7                   \n"
    "paddw     %%xmm4,%%xmm0                   \n"
    "paddw     %%xmm4,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "por       %%xmm7,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(0) "         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "sub       $0x4,%1                         \n"
    "jg        1b                              \n"
  : "+r"(dst_argb),       // %0
    "+r"(width)           // %1
  : "r"(scale),           // %2
    "r"(interval_size),   // %3
    "r"(interval_offset)  // %4
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBQUANTIZEROW_SSE2

#ifdef HAS_ARGBSHADEROW_SSE2
// Shade 4 pixels at a time by specified value.
void ARGBShadeRow_SSE2(const uint8* src_argb, uint8* dst_argb, int width,
                       uint32 value) {
  asm volatile (
    "movd      %3,%%xmm2                       \n"
    "punpcklbw %%xmm2,%%xmm2                   \n"
    "punpcklqdq %%xmm2,%%xmm2                  \n"

    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklbw %%xmm0,%%xmm0                   \n"
    "punpckhbw %%xmm1,%%xmm1                   \n"
    "pmulhuw   %%xmm2,%%xmm0                   \n"
    "pmulhuw   %%xmm2,%%xmm1                   \n"
    "psrlw     $0x8,%%xmm0                     \n"
    "psrlw     $0x8,%%xmm1                     \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x4,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_argb),  // %1
    "+r"(width)      // %2
  : "r"(value)       // %3
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2"
  );
}
#endif  // HAS_ARGBSHADEROW_SSE2

#ifdef HAS_ARGBMULTIPLYROW_SSE2
// Multiply 2 rows of ARGB pixels together, 4 pixels at a time.
void ARGBMultiplyRow_SSE2(const uint8* src_argb0, const uint8* src_argb1,
                          uint8* dst_argb, int width) {
  asm volatile (
    "pxor      %%xmm5,%%xmm5                  \n"

    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqu    " MEMACCESS(1) ",%%xmm2         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "movdqu    %%xmm0,%%xmm1                   \n"
    "movdqu    %%xmm2,%%xmm3                   \n"
    "punpcklbw %%xmm0,%%xmm0                   \n"
    "punpckhbw %%xmm1,%%xmm1                   \n"
    "punpcklbw %%xmm5,%%xmm2                   \n"
    "punpckhbw %%xmm5,%%xmm3                   \n"
    "pmulhuw   %%xmm2,%%xmm0                   \n"
    "pmulhuw   %%xmm3,%%xmm1                   \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "sub       $0x4,%3                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb0),  // %0
    "+r"(src_argb1),  // %1
    "+r"(dst_argb),   // %2
    "+r"(width)       // %3
  :
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
  );
}
#endif  // HAS_ARGBMULTIPLYROW_SSE2

#ifdef HAS_ARGBMULTIPLYROW_AVX2
// Multiply 2 rows of ARGB pixels together, 8 pixels at a time.
void ARGBMultiplyRow_AVX2(const uint8* src_argb0, const uint8* src_argb1,
                          uint8* dst_argb, int width) {
  asm volatile (
    "vpxor      %%ymm5,%%ymm5,%%ymm5           \n"

    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    " MEMACCESS(0) ",%%ymm1        \n"
    "lea        " MEMLEA(0x20,0) ",%0          \n"
    "vmovdqu    " MEMACCESS(1) ",%%ymm3        \n"
    "lea        " MEMLEA(0x20,1) ",%1          \n"
    "vpunpcklbw %%ymm1,%%ymm1,%%ymm0           \n"
    "vpunpckhbw %%ymm1,%%ymm1,%%ymm1           \n"
    "vpunpcklbw %%ymm5,%%ymm3,%%ymm2           \n"
    "vpunpckhbw %%ymm5,%%ymm3,%%ymm3           \n"
    "vpmulhuw   %%ymm2,%%ymm0,%%ymm0           \n"
    "vpmulhuw   %%ymm3,%%ymm1,%%ymm1           \n"
    "vpackuswb  %%ymm1,%%ymm0,%%ymm0           \n"
    "vmovdqu    %%ymm0," MEMACCESS(2) "        \n"
    "lea       " MEMLEA(0x20,2) ",%2           \n"
    "sub        $0x8,%3                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb0),  // %0
    "+r"(src_argb1),  // %1
    "+r"(dst_argb),   // %2
    "+r"(width)       // %3
  :
  : "memory", "cc"
#if defined(__AVX2__)
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
#endif
  );
}
#endif  // HAS_ARGBMULTIPLYROW_AVX2

#ifdef HAS_ARGBADDROW_SSE2
// Add 2 rows of ARGB pixels together, 4 pixels at a time.
void ARGBAddRow_SSE2(const uint8* src_argb0, const uint8* src_argb1,
                     uint8* dst_argb, int width) {
  asm volatile (
    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqu    " MEMACCESS(1) ",%%xmm1         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "paddusb   %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "sub       $0x4,%3                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb0),  // %0
    "+r"(src_argb1),  // %1
    "+r"(dst_argb),   // %2
    "+r"(width)       // %3
  :
  : "memory", "cc"
    , "xmm0", "xmm1"
  );
}
#endif  // HAS_ARGBADDROW_SSE2

#ifdef HAS_ARGBADDROW_AVX2
// Add 2 rows of ARGB pixels together, 4 pixels at a time.
void ARGBAddRow_AVX2(const uint8* src_argb0, const uint8* src_argb1,
                     uint8* dst_argb, int width) {
  asm volatile (
    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    " MEMACCESS(0) ",%%ymm0        \n"
    "lea        " MEMLEA(0x20,0) ",%0          \n"
    "vpaddusb   " MEMACCESS(1) ",%%ymm0,%%ymm0 \n"
    "lea        " MEMLEA(0x20,1) ",%1          \n"
    "vmovdqu    %%ymm0," MEMACCESS(2) "        \n"
    "lea        " MEMLEA(0x20,2) ",%2          \n"
    "sub        $0x8,%3                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb0),  // %0
    "+r"(src_argb1),  // %1
    "+r"(dst_argb),   // %2
    "+r"(width)       // %3
  :
  : "memory", "cc"
    , "xmm0"
  );
}
#endif  // HAS_ARGBADDROW_AVX2

#ifdef HAS_ARGBSUBTRACTROW_SSE2
// Subtract 2 rows of ARGB pixels, 4 pixels at a time.
void ARGBSubtractRow_SSE2(const uint8* src_argb0, const uint8* src_argb1,
                          uint8* dst_argb, int width) {
  asm volatile (
    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqu    " MEMACCESS(1) ",%%xmm1         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "psubusb   %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "sub       $0x4,%3                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb0),  // %0
    "+r"(src_argb1),  // %1
    "+r"(dst_argb),   // %2
    "+r"(width)       // %3
  :
  : "memory", "cc"
    , "xmm0", "xmm1"
  );
}
#endif  // HAS_ARGBSUBTRACTROW_SSE2

#ifdef HAS_ARGBSUBTRACTROW_AVX2
// Subtract 2 rows of ARGB pixels, 8 pixels at a time.
void ARGBSubtractRow_AVX2(const uint8* src_argb0, const uint8* src_argb1,
                          uint8* dst_argb, int width) {
  asm volatile (
    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    " MEMACCESS(0) ",%%ymm0        \n"
    "lea        " MEMLEA(0x20,0) ",%0          \n"
    "vpsubusb   " MEMACCESS(1) ",%%ymm0,%%ymm0 \n"
    "lea        " MEMLEA(0x20,1) ",%1          \n"
    "vmovdqu    %%ymm0," MEMACCESS(2) "        \n"
    "lea        " MEMLEA(0x20,2) ",%2          \n"
    "sub        $0x8,%3                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb0),  // %0
    "+r"(src_argb1),  // %1
    "+r"(dst_argb),   // %2
    "+r"(width)       // %3
  :
  : "memory", "cc"
    , "xmm0"
  );
}
#endif  // HAS_ARGBSUBTRACTROW_AVX2

#ifdef HAS_SOBELXROW_SSE2
// SobelX as a matrix is
// -1  0  1
// -2  0  2
// -1  0  1
void SobelXRow_SSE2(const uint8* src_y0, const uint8* src_y1,
                    const uint8* src_y2, uint8* dst_sobelx, int width) {
  asm volatile (
    "sub       %0,%1                           \n"
    "sub       %0,%2                           \n"
    "sub       %0,%3                           \n"
    "pxor      %%xmm5,%%xmm5                   \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movq      " MEMACCESS(0) ",%%xmm0         \n"
    "movq      " MEMACCESS2(0x2,0) ",%%xmm1    \n"
    "punpcklbw %%xmm5,%%xmm0                   \n"
    "punpcklbw %%xmm5,%%xmm1                   \n"
    "psubw     %%xmm1,%%xmm0                   \n"
    MEMOPREG(movq,0x00,0,1,1,xmm1)             //  movq      (%0,%1,1),%%xmm1
    MEMOPREG(movq,0x02,0,1,1,xmm2)             //  movq      0x2(%0,%1,1),%%xmm2
    "punpcklbw %%xmm5,%%xmm1                   \n"
    "punpcklbw %%xmm5,%%xmm2                   \n"
    "psubw     %%xmm2,%%xmm1                   \n"
    MEMOPREG(movq,0x00,0,2,1,xmm2)             //  movq      (%0,%2,1),%%xmm2
    MEMOPREG(movq,0x02,0,2,1,xmm3)             //  movq      0x2(%0,%2,1),%%xmm3
    "punpcklbw %%xmm5,%%xmm2                   \n"
    "punpcklbw %%xmm5,%%xmm3                   \n"
    "psubw     %%xmm3,%%xmm2                   \n"
    "paddw     %%xmm2,%%xmm0                   \n"
    "paddw     %%xmm1,%%xmm0                   \n"
    "paddw     %%xmm1,%%xmm0                   \n"
    "pxor      %%xmm1,%%xmm1                   \n"
    "psubw     %%xmm0,%%xmm1                   \n"
    "pmaxsw    %%xmm1,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    MEMOPMEM(movq,xmm0,0x00,0,3,1)             //  movq      %%xmm0,(%0,%3,1)
    "lea       " MEMLEA(0x8,0) ",%0            \n"
    "sub       $0x8,%4                         \n"
    "jg        1b                              \n"
  : "+r"(src_y0),      // %0
    "+r"(src_y1),      // %1
    "+r"(src_y2),      // %2
    "+r"(dst_sobelx),  // %3
    "+r"(width)        // %4
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
  );
}
#endif  // HAS_SOBELXROW_SSE2

#ifdef HAS_SOBELYROW_SSE2
// SobelY as a matrix is
// -1 -2 -1
//  0  0  0
//  1  2  1
void SobelYRow_SSE2(const uint8* src_y0, const uint8* src_y1,
                    uint8* dst_sobely, int width) {
  asm volatile (
    "sub       %0,%1                           \n"
    "sub       %0,%2                           \n"
    "pxor      %%xmm5,%%xmm5                   \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movq      " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movq,0x00,0,1,1,xmm1)             //  movq      (%0,%1,1),%%xmm1
    "punpcklbw %%xmm5,%%xmm0                   \n"
    "punpcklbw %%xmm5,%%xmm1                   \n"
    "psubw     %%xmm1,%%xmm0                   \n"
    "movq      " MEMACCESS2(0x1,0) ",%%xmm1    \n"
    MEMOPREG(movq,0x01,0,1,1,xmm2)             //  movq      0x1(%0,%1,1),%%xmm2
    "punpcklbw %%xmm5,%%xmm1                   \n"
    "punpcklbw %%xmm5,%%xmm2                   \n"
    "psubw     %%xmm2,%%xmm1                   \n"
    "movq      " MEMACCESS2(0x2,0) ",%%xmm2    \n"
    MEMOPREG(movq,0x02,0,1,1,xmm3)             //  movq      0x2(%0,%1,1),%%xmm3
    "punpcklbw %%xmm5,%%xmm2                   \n"
    "punpcklbw %%xmm5,%%xmm3                   \n"
    "psubw     %%xmm3,%%xmm2                   \n"
    "paddw     %%xmm2,%%xmm0                   \n"
    "paddw     %%xmm1,%%xmm0                   \n"
    "paddw     %%xmm1,%%xmm0                   \n"
    "pxor      %%xmm1,%%xmm1                   \n"
    "psubw     %%xmm0,%%xmm1                   \n"
    "pmaxsw    %%xmm1,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    MEMOPMEM(movq,xmm0,0x00,0,2,1)             //  movq      %%xmm0,(%0,%2,1)
    "lea       " MEMLEA(0x8,0) ",%0            \n"
    "sub       $0x8,%3                         \n"
    "jg        1b                              \n"
  : "+r"(src_y0),      // %0
    "+r"(src_y1),      // %1
    "+r"(dst_sobely),  // %2
    "+r"(width)        // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
  );
}
#endif  // HAS_SOBELYROW_SSE2

#ifdef HAS_SOBELROW_SSE2
// Adds Sobel X and Sobel Y and stores Sobel into ARGB.
// A = 255
// R = Sobel
// G = Sobel
// B = Sobel
void SobelRow_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                   uint8* dst_argb, int width) {
  asm volatile (
    "sub       %0,%1                           \n"
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "pslld     $0x18,%%xmm5                    \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,0,1,1,xmm1)           //  movdqu    (%0,%1,1),%%xmm1
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "paddusb   %%xmm1,%%xmm0                   \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "punpcklbw %%xmm0,%%xmm2                   \n"
    "punpckhbw %%xmm0,%%xmm0                   \n"
    "movdqa    %%xmm2,%%xmm1                   \n"
    "punpcklwd %%xmm2,%%xmm1                   \n"
    "punpckhwd %%xmm2,%%xmm2                   \n"
    "por       %%xmm5,%%xmm1                   \n"
    "por       %%xmm5,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm3                   \n"
    "punpcklwd %%xmm0,%%xmm3                   \n"
    "punpckhwd %%xmm0,%%xmm0                   \n"
    "por       %%xmm5,%%xmm3                   \n"
    "por       %%xmm5,%%xmm0                   \n"
    "movdqu    %%xmm1," MEMACCESS(2) "         \n"
    "movdqu    %%xmm2," MEMACCESS2(0x10,2) "   \n"
    "movdqu    %%xmm3," MEMACCESS2(0x20,2) "   \n"
    "movdqu    %%xmm0," MEMACCESS2(0x30,2) "   \n"
    "lea       " MEMLEA(0x40,2) ",%2           \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_sobelx),  // %0
    "+r"(src_sobely),  // %1
    "+r"(dst_argb),    // %2
    "+r"(width)        // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm5"
  );
}
#endif  // HAS_SOBELROW_SSE2

#ifdef HAS_SOBELTOPLANEROW_SSE2
// Adds Sobel X and Sobel Y and stores Sobel into a plane.
void SobelToPlaneRow_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                          uint8* dst_y, int width) {
  asm volatile (
    "sub       %0,%1                           \n"
    "pcmpeqb   %%xmm5,%%xmm5                   \n"
    "pslld     $0x18,%%xmm5                    \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,0,1,1,xmm1)           //  movdqu    (%0,%1,1),%%xmm1
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "paddusb   %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_sobelx),  // %0
    "+r"(src_sobely),  // %1
    "+r"(dst_y),       // %2
    "+r"(width)        // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1"
  );
}
#endif  // HAS_SOBELTOPLANEROW_SSE2

#ifdef HAS_SOBELXYROW_SSE2
// Mixes Sobel X, Sobel Y and Sobel into ARGB.
// A = 255
// R = Sobel X
// G = Sobel
// B = Sobel Y
void SobelXYRow_SSE2(const uint8* src_sobelx, const uint8* src_sobely,
                     uint8* dst_argb, int width) {
  asm volatile (
    "sub       %0,%1                           \n"
    "pcmpeqb   %%xmm5,%%xmm5                   \n"

    // 8 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,0,1,1,xmm1)           //  movdqu    (%0,%1,1),%%xmm1
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm2                   \n"
    "paddusb   %%xmm1,%%xmm2                   \n"
    "movdqa    %%xmm0,%%xmm3                   \n"
    "punpcklbw %%xmm5,%%xmm3                   \n"
    "punpckhbw %%xmm5,%%xmm0                   \n"
    "movdqa    %%xmm1,%%xmm4                   \n"
    "punpcklbw %%xmm2,%%xmm4                   \n"
    "punpckhbw %%xmm2,%%xmm1                   \n"
    "movdqa    %%xmm4,%%xmm6                   \n"
    "punpcklwd %%xmm3,%%xmm6                   \n"
    "punpckhwd %%xmm3,%%xmm4                   \n"
    "movdqa    %%xmm1,%%xmm7                   \n"
    "punpcklwd %%xmm0,%%xmm7                   \n"
    "punpckhwd %%xmm0,%%xmm1                   \n"
    "movdqu    %%xmm6," MEMACCESS(2) "         \n"
    "movdqu    %%xmm4," MEMACCESS2(0x10,2) "   \n"
    "movdqu    %%xmm7," MEMACCESS2(0x20,2) "   \n"
    "movdqu    %%xmm1," MEMACCESS2(0x30,2) "   \n"
    "lea       " MEMLEA(0x40,2) ",%2           \n"
    "sub       $0x10,%3                        \n"
    "jg        1b                              \n"
  : "+r"(src_sobelx),  // %0
    "+r"(src_sobely),  // %1
    "+r"(dst_argb),    // %2
    "+r"(width)        // %3
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_SOBELXYROW_SSE2

#ifdef HAS_COMPUTECUMULATIVESUMROW_SSE2
// Creates a table of cumulative sums where each value is a sum of all values
// above and to the left of the value, inclusive of the value.
void ComputeCumulativeSumRow_SSE2(const uint8* row, int32* cumsum,
                                  const int32* previous_cumsum, int width) {
  asm volatile (
    "pxor      %%xmm0,%%xmm0                   \n"
    "pxor      %%xmm1,%%xmm1                   \n"
    "sub       $0x4,%3                         \n"
    "jl        49f                             \n"
    "test      $0xf,%1                         \n"
    "jne       49f                             \n"

  // 4 pixel loop                              \n"
    LABELALIGN
  "40:                                         \n"
    "movdqu    " MEMACCESS(0) ",%%xmm2         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqa    %%xmm2,%%xmm4                   \n"
    "punpcklbw %%xmm1,%%xmm2                   \n"
    "movdqa    %%xmm2,%%xmm3                   \n"
    "punpcklwd %%xmm1,%%xmm2                   \n"
    "punpckhwd %%xmm1,%%xmm3                   \n"
    "punpckhbw %%xmm1,%%xmm4                   \n"
    "movdqa    %%xmm4,%%xmm5                   \n"
    "punpcklwd %%xmm1,%%xmm4                   \n"
    "punpckhwd %%xmm1,%%xmm5                   \n"
    "paddd     %%xmm2,%%xmm0                   \n"
    "movdqu    " MEMACCESS(2) ",%%xmm2         \n"
    "paddd     %%xmm0,%%xmm2                   \n"
    "paddd     %%xmm3,%%xmm0                   \n"
    "movdqu    " MEMACCESS2(0x10,2) ",%%xmm3   \n"
    "paddd     %%xmm0,%%xmm3                   \n"
    "paddd     %%xmm4,%%xmm0                   \n"
    "movdqu    " MEMACCESS2(0x20,2) ",%%xmm4   \n"
    "paddd     %%xmm0,%%xmm4                   \n"
    "paddd     %%xmm5,%%xmm0                   \n"
    "movdqu    " MEMACCESS2(0x30,2) ",%%xmm5   \n"
    "lea       " MEMLEA(0x40,2) ",%2           \n"
    "paddd     %%xmm0,%%xmm5                   \n"
    "movdqu    %%xmm2," MEMACCESS(1) "         \n"
    "movdqu    %%xmm3," MEMACCESS2(0x10,1) "   \n"
    "movdqu    %%xmm4," MEMACCESS2(0x20,1) "   \n"
    "movdqu    %%xmm5," MEMACCESS2(0x30,1) "   \n"
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "sub       $0x4,%3                         \n"
    "jge       40b                             \n"

  "49:                                         \n"
    "add       $0x3,%3                         \n"
    "jl        19f                             \n"

  // 1 pixel loop                              \n"
    LABELALIGN
  "10:                                         \n"
    "movd      " MEMACCESS(0) ",%%xmm2         \n"
    "lea       " MEMLEA(0x4,0) ",%0            \n"
    "punpcklbw %%xmm1,%%xmm2                   \n"
    "punpcklwd %%xmm1,%%xmm2                   \n"
    "paddd     %%xmm2,%%xmm0                   \n"
    "movdqu    " MEMACCESS(2) ",%%xmm2         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "paddd     %%xmm0,%%xmm2                   \n"
    "movdqu    %%xmm2," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x1,%3                         \n"
    "jge       10b                             \n"

  "19:                                         \n"
  : "+r"(row),  // %0
    "+r"(cumsum),  // %1
    "+r"(previous_cumsum),  // %2
    "+r"(width)  // %3
  :
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_COMPUTECUMULATIVESUMROW_SSE2

#ifdef HAS_CUMULATIVESUMTOAVERAGEROW_SSE2
void CumulativeSumToAverageRow_SSE2(const int32* topleft, const int32* botleft,
                                    int width, int area, uint8* dst,
                                    int count) {
  asm volatile (
    "movd      %5,%%xmm5                       \n"
    "cvtdq2ps  %%xmm5,%%xmm5                   \n"
    "rcpss     %%xmm5,%%xmm4                   \n"
    "pshufd    $0x0,%%xmm4,%%xmm4              \n"
    "sub       $0x4,%3                         \n"
    "jl        49f                             \n"
    "cmpl      $0x80,%5                        \n"
    "ja        40f                             \n"

    "pshufd    $0x0,%%xmm5,%%xmm5              \n"
    "pcmpeqb   %%xmm6,%%xmm6                   \n"
    "psrld     $0x10,%%xmm6                    \n"
    "cvtdq2ps  %%xmm6,%%xmm6                   \n"
    "addps     %%xmm6,%%xmm5                   \n"
    "mulps     %%xmm4,%%xmm5                   \n"
    "cvtps2dq  %%xmm5,%%xmm5                   \n"
    "packssdw  %%xmm5,%%xmm5                   \n"

  // 4 pixel small loop                        \n"
    LABELALIGN
  "4:                                         \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    MEMOPREG(psubd,0x00,0,4,4,xmm0)            // psubd    0x00(%0,%4,4),%%xmm0
    MEMOPREG(psubd,0x10,0,4,4,xmm1)            // psubd    0x10(%0,%4,4),%%xmm1
    MEMOPREG(psubd,0x20,0,4,4,xmm2)            // psubd    0x20(%0,%4,4),%%xmm2
    MEMOPREG(psubd,0x30,0,4,4,xmm3)            // psubd    0x30(%0,%4,4),%%xmm3
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "psubd     " MEMACCESS(1) ",%%xmm0         \n"
    "psubd     " MEMACCESS2(0x10,1) ",%%xmm1   \n"
    "psubd     " MEMACCESS2(0x20,1) ",%%xmm2   \n"
    "psubd     " MEMACCESS2(0x30,1) ",%%xmm3   \n"
    MEMOPREG(paddd,0x00,1,4,4,xmm0)            // paddd    0x00(%1,%4,4),%%xmm0
    MEMOPREG(paddd,0x10,1,4,4,xmm1)            // paddd    0x10(%1,%4,4),%%xmm1
    MEMOPREG(paddd,0x20,1,4,4,xmm2)            // paddd    0x20(%1,%4,4),%%xmm2
    MEMOPREG(paddd,0x30,1,4,4,xmm3)            // paddd    0x30(%1,%4,4),%%xmm3
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "packssdw  %%xmm1,%%xmm0                   \n"
    "packssdw  %%xmm3,%%xmm2                   \n"
    "pmulhuw   %%xmm5,%%xmm0                   \n"
    "pmulhuw   %%xmm5,%%xmm2                   \n"
    "packuswb  %%xmm2,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "sub       $0x4,%3                         \n"
    "jge       4b                              \n"
    "jmp       49f                             \n"

  // 4 pixel loop                              \n"
    LABELALIGN
  "40:                                         \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "movdqu    " MEMACCESS2(0x20,0) ",%%xmm2   \n"
    "movdqu    " MEMACCESS2(0x30,0) ",%%xmm3   \n"
    MEMOPREG(psubd,0x00,0,4,4,xmm0)            // psubd    0x00(%0,%4,4),%%xmm0
    MEMOPREG(psubd,0x10,0,4,4,xmm1)            // psubd    0x10(%0,%4,4),%%xmm1
    MEMOPREG(psubd,0x20,0,4,4,xmm2)            // psubd    0x20(%0,%4,4),%%xmm2
    MEMOPREG(psubd,0x30,0,4,4,xmm3)            // psubd    0x30(%0,%4,4),%%xmm3
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "psubd     " MEMACCESS(1) ",%%xmm0         \n"
    "psubd     " MEMACCESS2(0x10,1) ",%%xmm1   \n"
    "psubd     " MEMACCESS2(0x20,1) ",%%xmm2   \n"
    "psubd     " MEMACCESS2(0x30,1) ",%%xmm3   \n"
    MEMOPREG(paddd,0x00,1,4,4,xmm0)            // paddd    0x00(%1,%4,4),%%xmm0
    MEMOPREG(paddd,0x10,1,4,4,xmm1)            // paddd    0x10(%1,%4,4),%%xmm1
    MEMOPREG(paddd,0x20,1,4,4,xmm2)            // paddd    0x20(%1,%4,4),%%xmm2
    MEMOPREG(paddd,0x30,1,4,4,xmm3)            // paddd    0x30(%1,%4,4),%%xmm3
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "cvtdq2ps  %%xmm0,%%xmm0                   \n"
    "cvtdq2ps  %%xmm1,%%xmm1                   \n"
    "mulps     %%xmm4,%%xmm0                   \n"
    "mulps     %%xmm4,%%xmm1                   \n"
    "cvtdq2ps  %%xmm2,%%xmm2                   \n"
    "cvtdq2ps  %%xmm3,%%xmm3                   \n"
    "mulps     %%xmm4,%%xmm2                   \n"
    "mulps     %%xmm4,%%xmm3                   \n"
    "cvtps2dq  %%xmm0,%%xmm0                   \n"
    "cvtps2dq  %%xmm1,%%xmm1                   \n"
    "cvtps2dq  %%xmm2,%%xmm2                   \n"
    "cvtps2dq  %%xmm3,%%xmm3                   \n"
    "packssdw  %%xmm1,%%xmm0                   \n"
    "packssdw  %%xmm3,%%xmm2                   \n"
    "packuswb  %%xmm2,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "sub       $0x4,%3                         \n"
    "jge       40b                             \n"

  "49:                                         \n"
    "add       $0x3,%3                         \n"
    "jl        19f                             \n"

  // 1 pixel loop                              \n"
    LABELALIGN
  "10:                                         \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    MEMOPREG(psubd,0x00,0,4,4,xmm0)            // psubd    0x00(%0,%4,4),%%xmm0
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "psubd     " MEMACCESS(1) ",%%xmm0         \n"
    MEMOPREG(paddd,0x00,1,4,4,xmm0)            // paddd    0x00(%1,%4,4),%%xmm0
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "cvtdq2ps  %%xmm0,%%xmm0                   \n"
    "mulps     %%xmm4,%%xmm0                   \n"
    "cvtps2dq  %%xmm0,%%xmm0                   \n"
    "packssdw  %%xmm0,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movd      %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x4,2) ",%2            \n"
    "sub       $0x1,%3                         \n"
    "jge       10b                             \n"
  "19:                                         \n"
  : "+r"(topleft),  // %0
    "+r"(botleft),  // %1
    "+r"(dst),      // %2
    "+rm"(count)    // %3
  : "r"((intptr_t)(width)),  // %4
    "rm"(area)     // %5
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}
#endif  // HAS_CUMULATIVESUMTOAVERAGEROW_SSE2

#ifdef HAS_ARGBAFFINEROW_SSE2
// Copy ARGB pixels from source image with slope to a row of destination.
LIBYUV_API
void ARGBAffineRow_SSE2(const uint8* src_argb, int src_argb_stride,
                        uint8* dst_argb, const float* src_dudv, int width) {
  intptr_t src_argb_stride_temp = src_argb_stride;
  intptr_t temp;
  asm volatile (
    "movq      " MEMACCESS(3) ",%%xmm2         \n"
    "movq      " MEMACCESS2(0x08,3) ",%%xmm7   \n"
    "shl       $0x10,%1                        \n"
    "add       $0x4,%1                         \n"
    "movd      %1,%%xmm5                       \n"
    "sub       $0x4,%4                         \n"
    "jl        49f                             \n"

    "pshufd    $0x44,%%xmm7,%%xmm7             \n"
    "pshufd    $0x0,%%xmm5,%%xmm5              \n"
    "movdqa    %%xmm2,%%xmm0                   \n"
    "addps     %%xmm7,%%xmm0                   \n"
    "movlhps   %%xmm0,%%xmm2                   \n"
    "movdqa    %%xmm7,%%xmm4                   \n"
    "addps     %%xmm4,%%xmm4                   \n"
    "movdqa    %%xmm2,%%xmm3                   \n"
    "addps     %%xmm4,%%xmm3                   \n"
    "addps     %%xmm4,%%xmm4                   \n"

  // 4 pixel loop                              \n"
    LABELALIGN
  "40:                                         \n"
    "cvttps2dq %%xmm2,%%xmm0                   \n"  // x, y float to int first 2
    "cvttps2dq %%xmm3,%%xmm1                   \n"  // x, y float to int next 2
    "packssdw  %%xmm1,%%xmm0                   \n"  // x, y as 8 shorts
    "pmaddwd   %%xmm5,%%xmm0                   \n"  // off = x * 4 + y * stride
    "movd      %%xmm0,%k1                      \n"
    "pshufd    $0x39,%%xmm0,%%xmm0             \n"
    "movd      %%xmm0,%k5                      \n"
    "pshufd    $0x39,%%xmm0,%%xmm0             \n"
    MEMOPREG(movd,0x00,0,1,1,xmm1)             //  movd      (%0,%1,1),%%xmm1
    MEMOPREG(movd,0x00,0,5,1,xmm6)             //  movd      (%0,%5,1),%%xmm6
    "punpckldq %%xmm6,%%xmm1                   \n"
    "addps     %%xmm4,%%xmm2                   \n"
    "movq      %%xmm1," MEMACCESS(2) "         \n"
    "movd      %%xmm0,%k1                      \n"
    "pshufd    $0x39,%%xmm0,%%xmm0             \n"
    "movd      %%xmm0,%k5                      \n"
    MEMOPREG(movd,0x00,0,1,1,xmm0)             //  movd      (%0,%1,1),%%xmm0
    MEMOPREG(movd,0x00,0,5,1,xmm6)             //  movd      (%0,%5,1),%%xmm6
    "punpckldq %%xmm6,%%xmm0                   \n"
    "addps     %%xmm4,%%xmm3                   \n"
    "movq      %%xmm0," MEMACCESS2(0x08,2) "   \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "sub       $0x4,%4                         \n"
    "jge       40b                             \n"

  "49:                                         \n"
    "add       $0x3,%4                         \n"
    "jl        19f                             \n"

  // 1 pixel loop                              \n"
    LABELALIGN
  "10:                                         \n"
    "cvttps2dq %%xmm2,%%xmm0                   \n"
    "packssdw  %%xmm0,%%xmm0                   \n"
    "pmaddwd   %%xmm5,%%xmm0                   \n"
    "addps     %%xmm7,%%xmm2                   \n"
    "movd      %%xmm0,%k1                      \n"
    MEMOPREG(movd,0x00,0,1,1,xmm0)             //  movd      (%0,%1,1),%%xmm0
    "movd      %%xmm0," MEMACCESS(2) "         \n"
    "lea       " MEMLEA(0x04,2) ",%2           \n"
    "sub       $0x1,%4                         \n"
    "jge       10b                             \n"
  "19:                                         \n"
  : "+r"(src_argb),  // %0
    "+r"(src_argb_stride_temp),  // %1
    "+r"(dst_argb),  // %2
    "+r"(src_dudv),  // %3
    "+rm"(width),    // %4
    "=&r"(temp)      // %5
  :
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBAFFINEROW_SSE2

#ifdef HAS_INTERPOLATEROW_SSSE3
// Bilinear filter 16x2 -> 16x1
void InterpolateRow_SSSE3(uint8* dst_ptr, const uint8* src_ptr,
                          ptrdiff_t src_stride, int dst_width,
                          int source_y_fraction) {
  asm volatile (
    "sub       %1,%0                           \n"
    "cmp       $0x0,%3                         \n"
    "je        100f                            \n"
    "cmp       $0x80,%3                        \n"
    "je        50f                             \n"

    "movd      %3,%%xmm0                       \n"
    "neg       %3                              \n"
    "add       $0x100,%3                       \n"
    "movd      %3,%%xmm5                       \n"
    "punpcklbw %%xmm0,%%xmm5                   \n"
    "punpcklwd %%xmm5,%%xmm5                   \n"
    "pshufd    $0x0,%%xmm5,%%xmm5              \n"
    "mov       $0x80808080,%%eax               \n"
    "movd      %%eax,%%xmm4                    \n"
    "pshufd    $0x0,%%xmm4,%%xmm4              \n"

    // General purpose row blend.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(1) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,1,4,1,xmm2)
    "movdqa     %%xmm0,%%xmm1                  \n"
    "punpcklbw  %%xmm2,%%xmm0                  \n"
    "punpckhbw  %%xmm2,%%xmm1                  \n"
    "psubb      %%xmm4,%%xmm0                  \n"
    "psubb      %%xmm4,%%xmm1                  \n"
    "movdqa     %%xmm5,%%xmm2                  \n"
    "movdqa     %%xmm5,%%xmm3                  \n"
    "pmaddubsw  %%xmm0,%%xmm2                  \n"
    "pmaddubsw  %%xmm1,%%xmm3                  \n"
    "paddw      %%xmm4,%%xmm2                  \n"
    "paddw      %%xmm4,%%xmm3                  \n"
    "psrlw      $0x8,%%xmm2                    \n"
    "psrlw      $0x8,%%xmm3                    \n"
    "packuswb   %%xmm3,%%xmm2                  \n"
    MEMOPMEM(movdqu,xmm2,0x00,1,0,1)
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
    "jmp       99f                             \n"

    // Blend 50 / 50.
    LABELALIGN
  "50:                                         \n"
    "movdqu    " MEMACCESS(1) ",%%xmm0         \n"
    MEMOPREG(movdqu,0x00,1,4,1,xmm1)
    "pavgb     %%xmm1,%%xmm0                   \n"
    MEMOPMEM(movdqu,xmm0,0x00,1,0,1)
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        50b                             \n"
    "jmp       99f                             \n"

    // Blend 100 / 0 - Copy row unchanged.
    LABELALIGN
  "100:                                        \n"
    "movdqu    " MEMACCESS(1) ",%%xmm0         \n"
    MEMOPMEM(movdqu,xmm0,0x00,1,0,1)
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        100b                            \n"

  "99:                                         \n"
  : "+r"(dst_ptr),     // %0
    "+r"(src_ptr),     // %1
    "+rm"(dst_width),  // %2
    "+r"(source_y_fraction)  // %3
  : "r"((intptr_t)(src_stride))  // %4
  : "memory", "cc", "eax", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_INTERPOLATEROW_SSSE3

#ifdef HAS_INTERPOLATEROW_AVX2
// Bilinear filter 32x2 -> 32x1
void InterpolateRow_AVX2(uint8* dst_ptr, const uint8* src_ptr,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) {
  asm volatile (
    "cmp       $0x0,%3                         \n"
    "je        100f                            \n"
    "sub       %1,%0                           \n"
    "cmp       $0x80,%3                        \n"
    "je        50f                             \n"

    "vmovd      %3,%%xmm0                      \n"
    "neg        %3                             \n"
    "add        $0x100,%3                      \n"
    "vmovd      %3,%%xmm5                      \n"
    "vpunpcklbw %%xmm0,%%xmm5,%%xmm5           \n"
    "vpunpcklwd %%xmm5,%%xmm5,%%xmm5           \n"
    "vbroadcastss %%xmm5,%%ymm5                \n"
    "mov        $0x80808080,%%eax              \n"
    "vmovd      %%eax,%%xmm4                   \n"
    "vbroadcastss %%xmm4,%%ymm4                \n"

    // General purpose row blend.
    LABELALIGN
  "1:                                          \n"
    "vmovdqu    " MEMACCESS(1) ",%%ymm0        \n"
    MEMOPREG(vmovdqu,0x00,1,4,1,ymm2)
    "vpunpckhbw %%ymm2,%%ymm0,%%ymm1           \n"
    "vpunpcklbw %%ymm2,%%ymm0,%%ymm0           \n"
    "vpsubb     %%ymm4,%%ymm1,%%ymm1           \n"
    "vpsubb     %%ymm4,%%ymm0,%%ymm0           \n"
    "vpmaddubsw %%ymm1,%%ymm5,%%ymm1           \n"
    "vpmaddubsw %%ymm0,%%ymm5,%%ymm0           \n"
    "vpaddw     %%ymm4,%%ymm1,%%ymm1           \n"
    "vpaddw     %%ymm4,%%ymm0,%%ymm0           \n"
    "vpsrlw     $0x8,%%ymm1,%%ymm1             \n"
    "vpsrlw     $0x8,%%ymm0,%%ymm0             \n"
    "vpackuswb  %%ymm1,%%ymm0,%%ymm0           \n"
    MEMOPMEM(vmovdqu,ymm0,0x00,1,0,1)
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x20,%2                        \n"
    "jg        1b                              \n"
    "jmp       99f                             \n"

    // Blend 50 / 50.
    LABELALIGN
  "50:                                         \n"
    "vmovdqu    " MEMACCESS(1) ",%%ymm0        \n"
    VMEMOPREG(vpavgb,0x00,1,4,1,ymm0,ymm0)     // vpavgb (%1,%4,1),%%ymm0,%%ymm0
    MEMOPMEM(vmovdqu,ymm0,0x00,1,0,1)
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x20,%2                        \n"
    "jg        50b                             \n"
    "jmp       99f                             \n"

    // Blend 100 / 0 - Copy row unchanged.
    LABELALIGN
  "100:                                        \n"
    "rep movsb " MEMMOVESTRING(1,0) "          \n"
    "jmp       999f                            \n"

  "99:                                         \n"
    "vzeroupper                                \n"
  "999:                                        \n"
  : "+D"(dst_ptr),    // %0
    "+S"(src_ptr),    // %1
    "+cm"(dst_width),  // %2
    "+r"(source_y_fraction)  // %3
  : "r"((intptr_t)(src_stride))  // %4
  : "memory", "cc", "eax", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm4", "xmm5"
  );
}
#endif  // HAS_INTERPOLATEROW_AVX2

#ifdef HAS_ARGBSHUFFLEROW_SSSE3
// For BGRAToARGB, ABGRToARGB, RGBAToARGB, and ARGBToRGBA.
void ARGBShuffleRow_SSSE3(const uint8* src_argb, uint8* dst_argb,
                          const uint8* shuffler, int width) {
  asm volatile (
    "movdqu    " MEMACCESS(3) ",%%xmm5         \n"
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "movdqu    " MEMACCESS2(0x10,0) ",%%xmm1   \n"
    "lea       " MEMLEA(0x20,0) ",%0           \n"
    "pshufb    %%xmm5,%%xmm0                   \n"
    "pshufb    %%xmm5,%%xmm1                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,1) "   \n"
    "lea       " MEMLEA(0x20,1) ",%1           \n"
    "sub       $0x8,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_argb),  // %1
    "+r"(width)        // %2
  : "r"(shuffler)    // %3
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm5"
  );
}
#endif  // HAS_ARGBSHUFFLEROW_SSSE3

#ifdef HAS_ARGBSHUFFLEROW_AVX2
// For BGRAToARGB, ABGRToARGB, RGBAToARGB, and ARGBToRGBA.
void ARGBShuffleRow_AVX2(const uint8* src_argb, uint8* dst_argb,
                         const uint8* shuffler, int width) {
  asm volatile (
    "vbroadcastf128 " MEMACCESS(3) ",%%ymm5    \n"
    LABELALIGN
  "1:                                          \n"
    "vmovdqu   " MEMACCESS(0) ",%%ymm0         \n"
    "vmovdqu   " MEMACCESS2(0x20,0) ",%%ymm1   \n"
    "lea       " MEMLEA(0x40,0) ",%0           \n"
    "vpshufb   %%ymm5,%%ymm0,%%ymm0            \n"
    "vpshufb   %%ymm5,%%ymm1,%%ymm1            \n"
    "vmovdqu   %%ymm0," MEMACCESS(1) "         \n"
    "vmovdqu   %%ymm1," MEMACCESS2(0x20,1) "   \n"
    "lea       " MEMLEA(0x40,1) ",%1           \n"
    "sub       $0x10,%2                        \n"
    "jg        1b                              \n"
    "vzeroupper                                \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_argb),  // %1
    "+r"(width)        // %2
  : "r"(shuffler)    // %3
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm5"
  );
}
#endif  // HAS_ARGBSHUFFLEROW_AVX2

#ifdef HAS_ARGBSHUFFLEROW_SSE2
// For BGRAToARGB, ABGRToARGB, RGBAToARGB, and ARGBToRGBA.
void ARGBShuffleRow_SSE2(const uint8* src_argb, uint8* dst_argb,
                         const uint8* shuffler, int width) {
  uintptr_t pixel_temp;
  asm volatile (
    "pxor      %%xmm5,%%xmm5                   \n"
    "mov       " MEMACCESS(4) ",%k2            \n"
    "cmp       $0x3000102,%k2                  \n"
    "je        3012f                           \n"
    "cmp       $0x10203,%k2                    \n"
    "je        123f                            \n"
    "cmp       $0x30201,%k2                    \n"
    "je        321f                            \n"
    "cmp       $0x2010003,%k2                  \n"
    "je        2103f                           \n"

    LABELALIGN
  "1:                                          \n"
    "movzb     " MEMACCESS(4) ",%2             \n"
    MEMOPARG(movzb,0x00,0,2,1,2) "             \n"  //  movzb     (%0,%2,1),%2
    "mov       %b2," MEMACCESS(1) "            \n"
    "movzb     " MEMACCESS2(0x1,4) ",%2        \n"
    MEMOPARG(movzb,0x00,0,2,1,2) "             \n"  //  movzb     (%0,%2,1),%2
    "mov       %b2," MEMACCESS2(0x1,1) "       \n"
    "movzb     " MEMACCESS2(0x2,4) ",%2        \n"
    MEMOPARG(movzb,0x00,0,2,1,2) "             \n"  //  movzb     (%0,%2,1),%2
    "mov       %b2," MEMACCESS2(0x2,1) "       \n"
    "movzb     " MEMACCESS2(0x3,4) ",%2        \n"
    MEMOPARG(movzb,0x00,0,2,1,2) "             \n"  //  movzb     (%0,%2,1),%2
    "mov       %b2," MEMACCESS2(0x3,1) "       \n"
    "lea       " MEMLEA(0x4,0) ",%0            \n"
    "lea       " MEMLEA(0x4,1) ",%1            \n"
    "sub       $0x1,%3                         \n"
    "jg        1b                              \n"
    "jmp       99f                             \n"

    LABELALIGN
  "123:                                        \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklbw %%xmm5,%%xmm0                   \n"
    "punpckhbw %%xmm5,%%xmm1                   \n"
    "pshufhw   $0x1b,%%xmm0,%%xmm0             \n"
    "pshuflw   $0x1b,%%xmm0,%%xmm0             \n"
    "pshufhw   $0x1b,%%xmm1,%%xmm1             \n"
    "pshuflw   $0x1b,%%xmm1,%%xmm1             \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x4,%3                         \n"
    "jg        123b                            \n"
    "jmp       99f                             \n"

    LABELALIGN
  "321:                                        \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklbw %%xmm5,%%xmm0                   \n"
    "punpckhbw %%xmm5,%%xmm1                   \n"
    "pshufhw   $0x39,%%xmm0,%%xmm0             \n"
    "pshuflw   $0x39,%%xmm0,%%xmm0             \n"
    "pshufhw   $0x39,%%xmm1,%%xmm1             \n"
    "pshuflw   $0x39,%%xmm1,%%xmm1             \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x4,%3                         \n"
    "jg        321b                            \n"
    "jmp       99f                             \n"

    LABELALIGN
  "2103:                                       \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklbw %%xmm5,%%xmm0                   \n"
    "punpckhbw %%xmm5,%%xmm1                   \n"
    "pshufhw   $0x93,%%xmm0,%%xmm0             \n"
    "pshuflw   $0x93,%%xmm0,%%xmm0             \n"
    "pshufhw   $0x93,%%xmm1,%%xmm1             \n"
    "pshuflw   $0x93,%%xmm1,%%xmm1             \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x4,%3                         \n"
    "jg        2103b                           \n"
    "jmp       99f                             \n"

    LABELALIGN
  "3012:                                       \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x10,0) ",%0           \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "punpcklbw %%xmm5,%%xmm0                   \n"
    "punpckhbw %%xmm5,%%xmm1                   \n"
    "pshufhw   $0xc6,%%xmm0,%%xmm0             \n"
    "pshuflw   $0xc6,%%xmm0,%%xmm0             \n"
    "pshufhw   $0xc6,%%xmm1,%%xmm1             \n"
    "pshuflw   $0xc6,%%xmm1,%%xmm1             \n"
    "packuswb  %%xmm1,%%xmm0                   \n"
    "movdqu    %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x10,1) ",%1           \n"
    "sub       $0x4,%3                         \n"
    "jg        3012b                           \n"

  "99:                                         \n"
  : "+r"(src_argb),     // %0
    "+r"(dst_argb),     // %1
    "=&d"(pixel_temp),  // %2
    "+r"(width)         // %3
  : "r"(shuffler)       // %4
  : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm5"
  );
}
#endif  // HAS_ARGBSHUFFLEROW_SSE2

#ifdef HAS_I422TOYUY2ROW_SSE2
void I422ToYUY2Row_SSE2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_frame, int width) {
 asm volatile (
    "sub       %1,%2                             \n"
    LABELALIGN
  "1:                                            \n"
    "movq      " MEMACCESS(1) ",%%xmm2           \n"
    MEMOPREG(movq,0x00,1,2,1,xmm3)               //  movq    (%1,%2,1),%%xmm3
    "lea       " MEMLEA(0x8,1) ",%1              \n"
    "punpcklbw %%xmm3,%%xmm2                     \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0           \n"
    "lea       " MEMLEA(0x10,0) ",%0             \n"
    "movdqa    %%xmm0,%%xmm1                     \n"
    "punpcklbw %%xmm2,%%xmm0                     \n"
    "punpckhbw %%xmm2,%%xmm1                     \n"
    "movdqu    %%xmm0," MEMACCESS(3) "           \n"
    "movdqu    %%xmm1," MEMACCESS2(0x10,3) "     \n"
    "lea       " MEMLEA(0x20,3) ",%3             \n"
    "sub       $0x10,%4                          \n"
    "jg         1b                               \n"
    : "+r"(src_y),  // %0
      "+r"(src_u),  // %1
      "+r"(src_v),  // %2
      "+r"(dst_frame),  // %3
      "+rm"(width)  // %4
    :
    : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3"
  );
}
#endif  // HAS_I422TOYUY2ROW_SSE2

#ifdef HAS_I422TOUYVYROW_SSE2
void I422ToUYVYRow_SSE2(const uint8* src_y,
                        const uint8* src_u,
                        const uint8* src_v,
                        uint8* dst_frame, int width) {
 asm volatile (
    "sub        %1,%2                            \n"
    LABELALIGN
  "1:                                            \n"
    "movq      " MEMACCESS(1) ",%%xmm2           \n"
    MEMOPREG(movq,0x00,1,2,1,xmm3)               //  movq    (%1,%2,1),%%xmm3
    "lea       " MEMLEA(0x8,1) ",%1              \n"
    "punpcklbw %%xmm3,%%xmm2                     \n"
    "movdqu    " MEMACCESS(0) ",%%xmm0           \n"
    "movdqa    %%xmm2,%%xmm1                     \n"
    "lea       " MEMLEA(0x10,0) ",%0             \n"
    "punpcklbw %%xmm0,%%xmm1                     \n"
    "punpckhbw %%xmm0,%%xmm2                     \n"
    "movdqu    %%xmm1," MEMACCESS(3) "           \n"
    "movdqu    %%xmm2," MEMACCESS2(0x10,3) "     \n"
    "lea       " MEMLEA(0x20,3) ",%3             \n"
    "sub       $0x10,%4                          \n"
    "jg         1b                               \n"
    : "+r"(src_y),  // %0
      "+r"(src_u),  // %1
      "+r"(src_v),  // %2
      "+r"(dst_frame),  // %3
      "+rm"(width)  // %4
    :
    : "memory", "cc", NACL_R14
    "xmm0", "xmm1", "xmm2", "xmm3"
  );
}
#endif  // HAS_I422TOUYVYROW_SSE2

#ifdef HAS_ARGBPOLYNOMIALROW_SSE2
void ARGBPolynomialRow_SSE2(const uint8* src_argb,
                            uint8* dst_argb, const float* poly,
                            int width) {
  asm volatile (
    "pxor      %%xmm3,%%xmm3                   \n"

    // 2 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movq      " MEMACCESS(0) ",%%xmm0         \n"
    "lea       " MEMLEA(0x8,0) ",%0            \n"
    "punpcklbw %%xmm3,%%xmm0                   \n"
    "movdqa    %%xmm0,%%xmm4                   \n"
    "punpcklwd %%xmm3,%%xmm0                   \n"
    "punpckhwd %%xmm3,%%xmm4                   \n"
    "cvtdq2ps  %%xmm0,%%xmm0                   \n"
    "cvtdq2ps  %%xmm4,%%xmm4                   \n"
    "movdqa    %%xmm0,%%xmm1                   \n"
    "movdqa    %%xmm4,%%xmm5                   \n"
    "mulps     " MEMACCESS2(0x10,3) ",%%xmm0   \n"
    "mulps     " MEMACCESS2(0x10,3) ",%%xmm4   \n"
    "addps     " MEMACCESS(3) ",%%xmm0         \n"
    "addps     " MEMACCESS(3) ",%%xmm4         \n"
    "movdqa    %%xmm1,%%xmm2                   \n"
    "movdqa    %%xmm5,%%xmm6                   \n"
    "mulps     %%xmm1,%%xmm2                   \n"
    "mulps     %%xmm5,%%xmm6                   \n"
    "mulps     %%xmm2,%%xmm1                   \n"
    "mulps     %%xmm6,%%xmm5                   \n"
    "mulps     " MEMACCESS2(0x20,3) ",%%xmm2   \n"
    "mulps     " MEMACCESS2(0x20,3) ",%%xmm6   \n"
    "mulps     " MEMACCESS2(0x30,3) ",%%xmm1   \n"
    "mulps     " MEMACCESS2(0x30,3) ",%%xmm5   \n"
    "addps     %%xmm2,%%xmm0                   \n"
    "addps     %%xmm6,%%xmm4                   \n"
    "addps     %%xmm1,%%xmm0                   \n"
    "addps     %%xmm5,%%xmm4                   \n"
    "cvttps2dq %%xmm0,%%xmm0                   \n"
    "cvttps2dq %%xmm4,%%xmm4                   \n"
    "packuswb  %%xmm4,%%xmm0                   \n"
    "packuswb  %%xmm0,%%xmm0                   \n"
    "movq      %%xmm0," MEMACCESS(1) "         \n"
    "lea       " MEMLEA(0x8,1) ",%1            \n"
    "sub       $0x2,%2                         \n"
    "jg        1b                              \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_argb),  // %1
    "+r"(width)      // %2
  : "r"(poly)        // %3
  : "memory", "cc"
    , "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6"
  );
}
#endif  // HAS_ARGBPOLYNOMIALROW_SSE2

#ifdef HAS_ARGBPOLYNOMIALROW_AVX2
void ARGBPolynomialRow_AVX2(const uint8* src_argb,
                            uint8* dst_argb, const float* poly,
                            int width) {
  asm volatile (
    "vbroadcastf128 " MEMACCESS(3) ",%%ymm4     \n"
    "vbroadcastf128 " MEMACCESS2(0x10,3) ",%%ymm5 \n"
    "vbroadcastf128 " MEMACCESS2(0x20,3) ",%%ymm6 \n"
    "vbroadcastf128 " MEMACCESS2(0x30,3) ",%%ymm7 \n"

    // 2 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "vpmovzxbd   " MEMACCESS(0) ",%%ymm0       \n"  // 2 ARGB pixels
    "lea         " MEMLEA(0x8,0) ",%0          \n"
    "vcvtdq2ps   %%ymm0,%%ymm0                 \n"  // X 8 floats
    "vmulps      %%ymm0,%%ymm0,%%ymm2          \n"  // X * X
    "vmulps      %%ymm7,%%ymm0,%%ymm3          \n"  // C3 * X
    "vfmadd132ps %%ymm5,%%ymm4,%%ymm0          \n"  // result = C0 + C1 * X
    "vfmadd231ps %%ymm6,%%ymm2,%%ymm0          \n"  // result += C2 * X * X
    "vfmadd231ps %%ymm3,%%ymm2,%%ymm0          \n"  // result += C3 * X * X * X
    "vcvttps2dq  %%ymm0,%%ymm0                 \n"
    "vpackusdw   %%ymm0,%%ymm0,%%ymm0          \n"
    "vpermq      $0xd8,%%ymm0,%%ymm0           \n"
    "vpackuswb   %%xmm0,%%xmm0,%%xmm0          \n"
    "vmovq       %%xmm0," MEMACCESS(1) "       \n"
    "lea         " MEMLEA(0x8,1) ",%1          \n"
    "sub         $0x2,%2                       \n"
    "jg          1b                            \n"
    "vzeroupper                                \n"
  : "+r"(src_argb),  // %0
    "+r"(dst_argb),  // %1
    "+r"(width)      // %2
  : "r"(poly)        // %3
  : "memory", "cc",
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
  );
}
#endif  // HAS_ARGBPOLYNOMIALROW_AVX2

#ifdef HAS_ARGBCOLORTABLEROW_X86
// Tranform ARGB pixels with color table.
void ARGBColorTableRow_X86(uint8* dst_argb, const uint8* table_argb,
                           int width) {
  uintptr_t pixel_temp;
  asm volatile (
    // 1 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movzb     " MEMACCESS(0) ",%1             \n"
    "lea       " MEMLEA(0x4,0) ",%0            \n"
    MEMOPARG(movzb,0x00,3,1,4,1) "             \n"  // movzb (%3,%1,4),%1
    "mov       %b1," MEMACCESS2(-0x4,0) "      \n"
    "movzb     " MEMACCESS2(-0x3,0) ",%1       \n"
    MEMOPARG(movzb,0x01,3,1,4,1) "             \n"  // movzb 0x1(%3,%1,4),%1
    "mov       %b1," MEMACCESS2(-0x3,0) "      \n"
    "movzb     " MEMACCESS2(-0x2,0) ",%1       \n"
    MEMOPARG(movzb,0x02,3,1,4,1) "             \n"  // movzb 0x2(%3,%1,4),%1
    "mov       %b1," MEMACCESS2(-0x2,0) "      \n"
    "movzb     " MEMACCESS2(-0x1,0) ",%1       \n"
    MEMOPARG(movzb,0x03,3,1,4,1) "             \n"  // movzb 0x3(%3,%1,4),%1
    "mov       %b1," MEMACCESS2(-0x1,0) "      \n"
    "dec       %2                              \n"
    "jg        1b                              \n"
  : "+r"(dst_argb),     // %0
    "=&d"(pixel_temp),  // %1
    "+r"(width)         // %2
  : "r"(table_argb)     // %3
  : "memory", "cc");
}
#endif  // HAS_ARGBCOLORTABLEROW_X86

#ifdef HAS_RGBCOLORTABLEROW_X86
// Tranform RGB pixels with color table.
void RGBColorTableRow_X86(uint8* dst_argb, const uint8* table_argb, int width) {
  uintptr_t pixel_temp;
  asm volatile (
    // 1 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movzb     " MEMACCESS(0) ",%1             \n"
    "lea       " MEMLEA(0x4,0) ",%0            \n"
    MEMOPARG(movzb,0x00,3,1,4,1) "             \n"  // movzb (%3,%1,4),%1
    "mov       %b1," MEMACCESS2(-0x4,0) "      \n"
    "movzb     " MEMACCESS2(-0x3,0) ",%1       \n"
    MEMOPARG(movzb,0x01,3,1,4,1) "             \n"  // movzb 0x1(%3,%1,4),%1
    "mov       %b1," MEMACCESS2(-0x3,0) "      \n"
    "movzb     " MEMACCESS2(-0x2,0) ",%1       \n"
    MEMOPARG(movzb,0x02,3,1,4,1) "             \n"  // movzb 0x2(%3,%1,4),%1
    "mov       %b1," MEMACCESS2(-0x2,0) "      \n"
    "dec       %2                              \n"
    "jg        1b                              \n"
  : "+r"(dst_argb),     // %0
    "=&d"(pixel_temp),  // %1
    "+r"(width)         // %2
  : "r"(table_argb)     // %3
  : "memory", "cc");
}
#endif  // HAS_RGBCOLORTABLEROW_X86

#ifdef HAS_ARGBLUMACOLORTABLEROW_SSSE3
// Tranform RGB pixels with luma table.
void ARGBLumaColorTableRow_SSSE3(const uint8* src_argb, uint8* dst_argb,
                                 int width,
                                 const uint8* luma, uint32 lumacoeff) {
  uintptr_t pixel_temp;
  uintptr_t table_temp;
  asm volatile (
    "movd      %6,%%xmm3                       \n"
    "pshufd    $0x0,%%xmm3,%%xmm3              \n"
    "pcmpeqb   %%xmm4,%%xmm4                   \n"
    "psllw     $0x8,%%xmm4                     \n"
    "pxor      %%xmm5,%%xmm5                   \n"

    // 4 pixel loop.
    LABELALIGN
  "1:                                          \n"
    "movdqu    " MEMACCESS(2) ",%%xmm0         \n"
    "pmaddubsw %%xmm3,%%xmm0                   \n"
    "phaddw    %%xmm0,%%xmm0                   \n"
    "pand      %%xmm4,%%xmm0                   \n"
    "punpcklwd %%xmm5,%%xmm0                   \n"
    "movd      %%xmm0,%k1                      \n"  // 32 bit offset
    "add       %5,%1                           \n"
    "pshufd    $0x39,%%xmm0,%%xmm0             \n"

    "movzb     " MEMACCESS(2) ",%0             \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS(3) "            \n"
    "movzb     " MEMACCESS2(0x1,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0x1,3) "       \n"
    "movzb     " MEMACCESS2(0x2,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0x2,3) "       \n"
    "movzb     " MEMACCESS2(0x3,2) ",%0        \n"
    "mov       %b0," MEMACCESS2(0x3,3) "       \n"

    "movd      %%xmm0,%k1                      \n"  // 32 bit offset
    "add       %5,%1                           \n"
    "pshufd    $0x39,%%xmm0,%%xmm0             \n"

    "movzb     " MEMACCESS2(0x4,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0x4,3) "       \n"
    "movzb     " MEMACCESS2(0x5,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0x5,3) "       \n"
    "movzb     " MEMACCESS2(0x6,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0x6,3) "       \n"
    "movzb     " MEMACCESS2(0x7,2) ",%0        \n"
    "mov       %b0," MEMACCESS2(0x7,3) "       \n"

    "movd      %%xmm0,%k1                      \n"  // 32 bit offset
    "add       %5,%1                           \n"
    "pshufd    $0x39,%%xmm0,%%xmm0             \n"

    "movzb     " MEMACCESS2(0x8,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0x8,3) "       \n"
    "movzb     " MEMACCESS2(0x9,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0x9,3) "       \n"
    "movzb     " MEMACCESS2(0xa,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0xa,3) "       \n"
    "movzb     " MEMACCESS2(0xb,2) ",%0        \n"
    "mov       %b0," MEMACCESS2(0xb,3) "       \n"

    "movd      %%xmm0,%k1                      \n"  // 32 bit offset
    "add       %5,%1                           \n"

    "movzb     " MEMACCESS2(0xc,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0xc,3) "       \n"
    "movzb     " MEMACCESS2(0xd,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0xd,3) "       \n"
    "movzb     " MEMACCESS2(0xe,2) ",%0        \n"
    MEMOPARG(movzb,0x00,1,0,1,0) "             \n"  // movzb     (%1,%0,1),%0
    "mov       %b0," MEMACCESS2(0xe,3) "       \n"
    "movzb     " MEMACCESS2(0xf,2) ",%0        \n"
    "mov       %b0," MEMACCESS2(0xf,3) "       \n"
    "lea       " MEMLEA(0x10,2) ",%2           \n"
    "lea       " MEMLEA(0x10,3) ",%3           \n"
    "sub       $0x4,%4                         \n"
    "jg        1b                              \n"
  : "=&d"(pixel_temp),  // %0
    "=&a"(table_temp),  // %1
    "+r"(src_argb),     // %2
    "+r"(dst_argb),     // %3
    "+rm"(width)        // %4
  : "r"(luma),          // %5
    "rm"(lumacoeff)     // %6
  : "memory", "cc", "xmm0", "xmm3", "xmm4", "xmm5"
  );
}
#endif  // HAS_ARGBLUMACOLORTABLEROW_SSSE3

#endif  // defined(__x86_64__) || defined(__i386__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
