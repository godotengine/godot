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

// This module is for Visual C 32/64 bit
#if !defined(LIBYUV_DISABLE_X86) && defined(_MSC_VER) && \
    (defined(_M_IX86) || defined(_M_X64)) &&             \
    (!defined(__clang__) || defined(LIBYUV_ENABLE_ROWWIN))

#if defined(_M_ARM64EC)
#include <intrin.h>
#elif defined(_M_X64)
#include <emmintrin.h>
#include <tmmintrin.h>  // For _mm_maddubs_epi16
#endif

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// 64 bit
#if defined(_M_X64)

// Read 8 UV from 444
#define READYUV444                                    \
  xmm3 = _mm_loadl_epi64((__m128i*)u_buf);            \
  xmm1 = _mm_loadl_epi64((__m128i*)(u_buf + offset)); \
  xmm3 = _mm_unpacklo_epi8(xmm3, xmm1);               \
  u_buf += 8;                                         \
  xmm4 = _mm_loadl_epi64((__m128i*)y_buf);            \
  xmm4 = _mm_unpacklo_epi8(xmm4, xmm4);               \
  y_buf += 8;

// Read 8 UV from 444, With 8 Alpha.
#define READYUVA444                                   \
  xmm3 = _mm_loadl_epi64((__m128i*)u_buf);            \
  xmm1 = _mm_loadl_epi64((__m128i*)(u_buf + offset)); \
  xmm3 = _mm_unpacklo_epi8(xmm3, xmm1);               \
  u_buf += 8;                                         \
  xmm4 = _mm_loadl_epi64((__m128i*)y_buf);            \
  xmm4 = _mm_unpacklo_epi8(xmm4, xmm4);               \
  y_buf += 8;                                         \
  xmm5 = _mm_loadl_epi64((__m128i*)a_buf);            \
  a_buf += 8;

// Read 4 UV from 422, upsample to 8 UV.
#define READYUV422                                        \
  xmm3 = _mm_cvtsi32_si128(*(uint32_t*)u_buf);            \
  xmm1 = _mm_cvtsi32_si128(*(uint32_t*)(u_buf + offset)); \
  xmm3 = _mm_unpacklo_epi8(xmm3, xmm1);                   \
  xmm3 = _mm_unpacklo_epi16(xmm3, xmm3);                  \
  u_buf += 4;                                             \
  xmm4 = _mm_loadl_epi64((__m128i*)y_buf);                \
  xmm4 = _mm_unpacklo_epi8(xmm4, xmm4);                   \
  y_buf += 8;

// Read 4 UV from 422, upsample to 8 UV.  With 8 Alpha.
#define READYUVA422                                       \
  xmm3 = _mm_cvtsi32_si128(*(uint32_t*)u_buf);            \
  xmm1 = _mm_cvtsi32_si128(*(uint32_t*)(u_buf + offset)); \
  xmm3 = _mm_unpacklo_epi8(xmm3, xmm1);                   \
  xmm3 = _mm_unpacklo_epi16(xmm3, xmm3);                  \
  u_buf += 4;                                             \
  xmm4 = _mm_loadl_epi64((__m128i*)y_buf);                \
  xmm4 = _mm_unpacklo_epi8(xmm4, xmm4);                   \
  y_buf += 8;                                             \
  xmm5 = _mm_loadl_epi64((__m128i*)a_buf);                \
  a_buf += 8;

// Convert 8 pixels: 8 UV and 8 Y.
#define YUVTORGB(yuvconstants)                                      \
  xmm3 = _mm_sub_epi8(xmm3, _mm_set1_epi8((char)0x80));             \
  xmm4 = _mm_mulhi_epu16(xmm4, *(__m128i*)yuvconstants->kYToRgb);   \
  xmm4 = _mm_add_epi16(xmm4, *(__m128i*)yuvconstants->kYBiasToRgb); \
  xmm0 = _mm_maddubs_epi16(*(__m128i*)yuvconstants->kUVToB, xmm3);  \
  xmm1 = _mm_maddubs_epi16(*(__m128i*)yuvconstants->kUVToG, xmm3);  \
  xmm2 = _mm_maddubs_epi16(*(__m128i*)yuvconstants->kUVToR, xmm3);  \
  xmm0 = _mm_adds_epi16(xmm4, xmm0);                                \
  xmm1 = _mm_subs_epi16(xmm4, xmm1);                                \
  xmm2 = _mm_adds_epi16(xmm4, xmm2);                                \
  xmm0 = _mm_srai_epi16(xmm0, 6);                                   \
  xmm1 = _mm_srai_epi16(xmm1, 6);                                   \
  xmm2 = _mm_srai_epi16(xmm2, 6);                                   \
  xmm0 = _mm_packus_epi16(xmm0, xmm0);                              \
  xmm1 = _mm_packus_epi16(xmm1, xmm1);                              \
  xmm2 = _mm_packus_epi16(xmm2, xmm2);

// Store 8 ARGB values.
#define STOREARGB                                    \
  xmm0 = _mm_unpacklo_epi8(xmm0, xmm1);              \
  xmm2 = _mm_unpacklo_epi8(xmm2, xmm5);              \
  xmm1 = _mm_loadu_si128(&xmm0);                     \
  xmm0 = _mm_unpacklo_epi16(xmm0, xmm2);             \
  xmm1 = _mm_unpackhi_epi16(xmm1, xmm2);             \
  _mm_storeu_si128((__m128i*)dst_argb, xmm0);        \
  _mm_storeu_si128((__m128i*)(dst_argb + 16), xmm1); \
  dst_argb += 32;

#if defined(HAS_I422TOARGBROW_SSSE3)
void I422ToARGBRow_SSSE3(const uint8_t* y_buf,
                         const uint8_t* u_buf,
                         const uint8_t* v_buf,
                         uint8_t* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  __m128i xmm0, xmm1, xmm2, xmm3, xmm4;
  const __m128i xmm5 = _mm_set1_epi8(-1);
  const ptrdiff_t offset = (uint8_t*)v_buf - (uint8_t*)u_buf;
  while (width > 0) {
    READYUV422
    YUVTORGB(yuvconstants)
    STOREARGB
    width -= 8;
  }
}
#endif

#if defined(HAS_I422ALPHATOARGBROW_SSSE3)
void I422AlphaToARGBRow_SSSE3(const uint8_t* y_buf,
                              const uint8_t* u_buf,
                              const uint8_t* v_buf,
                              const uint8_t* a_buf,
                              uint8_t* dst_argb,
                              const struct YuvConstants* yuvconstants,
                              int width) {
  __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;
  const ptrdiff_t offset = (uint8_t*)v_buf - (uint8_t*)u_buf;
  while (width > 0) {
    READYUVA422
    YUVTORGB(yuvconstants)
    STOREARGB
    width -= 8;
  }
}
#endif

#if defined(HAS_I444TOARGBROW_SSSE3)
void I444ToARGBRow_SSSE3(const uint8_t* y_buf,
                         const uint8_t* u_buf,
                         const uint8_t* v_buf,
                         uint8_t* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  __m128i xmm0, xmm1, xmm2, xmm3, xmm4;
  const __m128i xmm5 = _mm_set1_epi8(-1);
  const ptrdiff_t offset = (uint8_t*)v_buf - (uint8_t*)u_buf;
  while (width > 0) {
    READYUV444
    YUVTORGB(yuvconstants)
    STOREARGB
    width -= 8;
  }
}
#endif

#if defined(HAS_I444ALPHATOARGBROW_SSSE3)
void I444AlphaToARGBRow_SSSE3(const uint8_t* y_buf,
                              const uint8_t* u_buf,
                              const uint8_t* v_buf,
                              const uint8_t* a_buf,
                              uint8_t* dst_argb,
                              const struct YuvConstants* yuvconstants,
                              int width) {
  __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;
  const ptrdiff_t offset = (uint8_t*)v_buf - (uint8_t*)u_buf;
  while (width > 0) {
    READYUVA444
    YUVTORGB(yuvconstants)
    STOREARGB
    width -= 8;
  }
}
#endif

// 32 bit
#else  // defined(_M_X64)

// if HAS_ARGBTOUVROW_SSSE3

// 8 bit fixed point 0.5, for bias of UV.
static const ulvec8 kBiasUV128 = {
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};

// NV21 shuf 8 VU to 16 UV.
static const lvec8 kShuffleNV21 = {
    1, 0, 1, 0, 3, 2, 3, 2, 5, 4, 5, 4, 7, 6, 7, 6,
    1, 0, 1, 0, 3, 2, 3, 2, 5, 4, 5, 4, 7, 6, 7, 6,
};

// YUY2 shuf 16 Y to 32 Y.
static const lvec8 kShuffleYUY2Y = {0,  0,  2,  2,  4,  4,  6,  6,  8,  8, 10,
                                    10, 12, 12, 14, 14, 0,  0,  2,  2,  4, 4,
                                    6,  6,  8,  8,  10, 10, 12, 12, 14, 14};

// YUY2 shuf 8 UV to 16 UV.
static const lvec8 kShuffleYUY2UV = {1,  3,  1,  3,  5,  7,  5,  7,  9,  11, 9,
                                     11, 13, 15, 13, 15, 1,  3,  1,  3,  5,  7,
                                     5,  7,  9,  11, 9,  11, 13, 15, 13, 15};

// UYVY shuf 16 Y to 32 Y.
static const lvec8 kShuffleUYVYY = {1,  1,  3,  3,  5,  5,  7,  7,  9,  9, 11,
                                    11, 13, 13, 15, 15, 1,  1,  3,  3,  5, 5,
                                    7,  7,  9,  9,  11, 11, 13, 13, 15, 15};

// UYVY shuf 8 UV to 16 UV.
static const lvec8 kShuffleUYVYUV = {0,  2,  0,  2,  4,  6,  4,  6,  8,  10, 8,
                                     10, 12, 14, 12, 14, 0,  2,  0,  2,  4,  6,
                                     4,  6,  8,  10, 8,  10, 12, 14, 12, 14};

// JPeg full range.
static const vec8 kARGBToYJ = {15, 75, 38, 0, 15, 75, 38, 0,
                               15, 75, 38, 0, 15, 75, 38, 0};
// endif

// vpermd for vphaddw + vpackuswb vpermd.
static const lvec32 kPermdARGBToY_AVX = {0, 4, 1, 5, 2, 6, 3, 7};

// Constants for ARGB.
static const vec8 kARGBToY = {13, 65, 33, 0, 13, 65, 33, 0,
                              13, 65, 33, 0, 13, 65, 33, 0};

static const vec8 kARGBToU = {112, -74, -38, 0, 112, -74, -38, 0,
                              112, -74, -38, 0, 112, -74, -38, 0};

static const vec8 kARGBToUJ = {127, -84, -43, 0, 127, -84, -43, 0,
                               127, -84, -43, 0, 127, -84, -43, 0};

static const vec8 kARGBToV = {
    -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0, -18, -94, 112, 0,
};

static const vec8 kARGBToVJ = {-20, -107, 127, 0, -20, -107, 127, 0,
                               -20, -107, 127, 0, -20, -107, 127, 0};

// vpshufb for vphaddw + vpackuswb packed to shorts.
static const lvec8 kShufARGBToUV_AVX = {
    0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
    0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};

// Constants for BGRA.
static const vec8 kBGRAToY = {0, 33, 65, 13, 0, 33, 65, 13,
                              0, 33, 65, 13, 0, 33, 65, 13};

static const vec8 kBGRAToU = {0, -38, -74, 112, 0, -38, -74, 112,
                              0, -38, -74, 112, 0, -38, -74, 112};

static const vec8 kBGRAToV = {0, 112, -94, -18, 0, 112, -94, -18,
                              0, 112, -94, -18, 0, 112, -94, -18};

// Constants for ABGR.
static const vec8 kABGRToY = {33, 65, 13, 0, 33, 65, 13, 0,
                              33, 65, 13, 0, 33, 65, 13, 0};

static const vec8 kABGRToU = {-38, -74, 112, 0, -38, -74, 112, 0,
                              -38, -74, 112, 0, -38, -74, 112, 0};

static const vec8 kABGRToV = {112, -94, -18, 0, 112, -94, -18, 0,
                              112, -94, -18, 0, 112, -94, -18, 0};

// Constants for RGBA.
static const vec8 kRGBAToY = {0, 13, 65, 33, 0, 13, 65, 33,
                              0, 13, 65, 33, 0, 13, 65, 33};

static const vec8 kRGBAToU = {0, 112, -74, -38, 0, 112, -74, -38,
                              0, 112, -74, -38, 0, 112, -74, -38};

static const vec8 kRGBAToV = {0, -18, -94, 112, 0, -18, -94, 112,
                              0, -18, -94, 112, 0, -18, -94, 112};

static const uvec8 kAddY16 = {16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u,
                              16u, 16u, 16u, 16u, 16u, 16u, 16u, 16u};

// 7 bit fixed point 0.5.
static const vec16 kAddYJ64 = {64, 64, 64, 64, 64, 64, 64, 64};

// Shuffle table for converting RGB24 to ARGB.
static const uvec8 kShuffleMaskRGB24ToARGB = {
    0u, 1u, 2u, 12u, 3u, 4u, 5u, 13u, 6u, 7u, 8u, 14u, 9u, 10u, 11u, 15u};

// Shuffle table for converting RAW to ARGB.
static const uvec8 kShuffleMaskRAWToARGB = {2u, 1u, 0u, 12u, 5u,  4u,  3u, 13u,
                                            8u, 7u, 6u, 14u, 11u, 10u, 9u, 15u};

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

// Duplicates gray value 3 times and fills in alpha opaque.
__declspec(naked) void J400ToARGBRow_SSE2(const uint8_t* src_y,
                                          uint8_t* dst_argb,
                                          int width) {
  __asm {
    mov        eax, [esp + 4]  // src_y
    mov        edx, [esp + 8]  // dst_argb
    mov        ecx, [esp + 12]  // width
    pcmpeqb    xmm5, xmm5  // generate mask 0xff000000
    pslld      xmm5, 24

  convertloop:
    movq       xmm0, qword ptr [eax]
    lea        eax,  [eax + 8]
    punpcklbw  xmm0, xmm0
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm0
    punpckhwd  xmm1, xmm1
    por        xmm0, xmm5
    por        xmm1, xmm5
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop
    ret
  }
}

#ifdef HAS_J400TOARGBROW_AVX2
// Duplicates gray value 3 times and fills in alpha opaque.
__declspec(naked) void J400ToARGBRow_AVX2(const uint8_t* src_y,
                                          uint8_t* dst_argb,
                                          int width) {
  __asm {
    mov         eax, [esp + 4]  // src_y
    mov         edx, [esp + 8]  // dst_argb
    mov         ecx, [esp + 12]  // width
    vpcmpeqb    ymm5, ymm5, ymm5  // generate mask 0xff000000
    vpslld      ymm5, ymm5, 24

  convertloop:
    vmovdqu     xmm0, [eax]
    lea         eax,  [eax + 16]
    vpermq      ymm0, ymm0, 0xd8
    vpunpcklbw  ymm0, ymm0, ymm0
    vpermq      ymm0, ymm0, 0xd8
    vpunpckhwd  ymm1, ymm0, ymm0
    vpunpcklwd  ymm0, ymm0, ymm0
    vpor        ymm0, ymm0, ymm5
    vpor        ymm1, ymm1, ymm5
    vmovdqu     [edx], ymm0
    vmovdqu     [edx + 32], ymm1
    lea         edx, [edx + 64]
    sub         ecx, 16
    jg          convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_J400TOARGBROW_AVX2

__declspec(naked) void RGB24ToARGBRow_SSSE3(const uint8_t* src_rgb24,
                                            uint8_t* dst_argb,
                                            int width) {
  __asm {
    mov       eax, [esp + 4]  // src_rgb24
    mov       edx, [esp + 8]  // dst_argb
    mov       ecx, [esp + 12]  // width
    pcmpeqb   xmm5, xmm5  // generate mask 0xff000000
    pslld     xmm5, 24
    movdqa    xmm4, xmmword ptr kShuffleMaskRGB24ToARGB

 convertloop:
    movdqu    xmm0, [eax]
    movdqu    xmm1, [eax + 16]
    movdqu    xmm3, [eax + 32]
    lea       eax, [eax + 48]
    movdqa    xmm2, xmm3
    palignr   xmm2, xmm1, 8  // xmm2 = { xmm3[0:3] xmm1[8:15]}
    pshufb    xmm2, xmm4
    por       xmm2, xmm5
    palignr   xmm1, xmm0, 12  // xmm1 = { xmm3[0:7] xmm0[12:15]}
    pshufb    xmm0, xmm4
    movdqu    [edx + 32], xmm2
    por       xmm0, xmm5
    pshufb    xmm1, xmm4
    movdqu    [edx], xmm0
    por       xmm1, xmm5
    palignr   xmm3, xmm3, 4  // xmm3 = { xmm3[4:15]}
    pshufb    xmm3, xmm4
    movdqu    [edx + 16], xmm1
    por       xmm3, xmm5
    movdqu    [edx + 48], xmm3
    lea       edx, [edx + 64]
    sub       ecx, 16
    jg        convertloop
    ret
  }
}

__declspec(naked) void RAWToARGBRow_SSSE3(const uint8_t* src_raw,
                                          uint8_t* dst_argb,
                                          int width) {
  __asm {
    mov       eax, [esp + 4]  // src_raw
    mov       edx, [esp + 8]  // dst_argb
    mov       ecx, [esp + 12]  // width
    pcmpeqb   xmm5, xmm5  // generate mask 0xff000000
    pslld     xmm5, 24
    movdqa    xmm4, xmmword ptr kShuffleMaskRAWToARGB

 convertloop:
    movdqu    xmm0, [eax]
    movdqu    xmm1, [eax + 16]
    movdqu    xmm3, [eax + 32]
    lea       eax, [eax + 48]
    movdqa    xmm2, xmm3
    palignr   xmm2, xmm1, 8  // xmm2 = { xmm3[0:3] xmm1[8:15]}
    pshufb    xmm2, xmm4
    por       xmm2, xmm5
    palignr   xmm1, xmm0, 12  // xmm1 = { xmm3[0:7] xmm0[12:15]}
    pshufb    xmm0, xmm4
    movdqu    [edx + 32], xmm2
    por       xmm0, xmm5
    pshufb    xmm1, xmm4
    movdqu    [edx], xmm0
    por       xmm1, xmm5
    palignr   xmm3, xmm3, 4  // xmm3 = { xmm3[4:15]}
    pshufb    xmm3, xmm4
    movdqu    [edx + 16], xmm1
    por       xmm3, xmm5
    movdqu    [edx + 48], xmm3
    lea       edx, [edx + 64]
    sub       ecx, 16
    jg        convertloop
    ret
  }
}

__declspec(naked) void RAWToRGB24Row_SSSE3(const uint8_t* src_raw,
                                           uint8_t* dst_rgb24,
                                           int width) {
  __asm {
    mov       eax, [esp + 4]  // src_raw
    mov       edx, [esp + 8]  // dst_rgb24
    mov       ecx, [esp + 12]  // width
    movdqa    xmm3, xmmword ptr kShuffleMaskRAWToRGB24_0
    movdqa    xmm4, xmmword ptr kShuffleMaskRAWToRGB24_1
    movdqa    xmm5, xmmword ptr kShuffleMaskRAWToRGB24_2

 convertloop:
    movdqu    xmm0, [eax]
    movdqu    xmm1, [eax + 4]
    movdqu    xmm2, [eax + 8]
    lea       eax, [eax + 24]
    pshufb    xmm0, xmm3
    pshufb    xmm1, xmm4
    pshufb    xmm2, xmm5
    movq      qword ptr [edx], xmm0
    movq      qword ptr [edx + 8], xmm1
    movq      qword ptr [edx + 16], xmm2
    lea       edx, [edx + 24]
    sub       ecx, 8
    jg        convertloop
    ret
  }
}

// pmul method to replicate bits.
// Math to replicate bits:
// (v << 8) | (v << 3)
// v * 256 + v * 8
// v * (256 + 8)
// G shift of 5 is incorporated, so shift is 5 + 8 and 5 + 3
// 20 instructions.
__declspec(naked) void RGB565ToARGBRow_SSE2(const uint8_t* src_rgb565,
                                            uint8_t* dst_argb,
                                            int width) {
  __asm {
    mov       eax, 0x01080108  // generate multiplier to repeat 5 bits
    movd      xmm5, eax
    pshufd    xmm5, xmm5, 0
    mov       eax, 0x20802080  // multiplier shift by 5 and then repeat 6 bits
    movd      xmm6, eax
    pshufd    xmm6, xmm6, 0
    pcmpeqb   xmm3, xmm3  // generate mask 0xf800f800 for Red
    psllw     xmm3, 11
    pcmpeqb   xmm4, xmm4  // generate mask 0x07e007e0 for Green
    psllw     xmm4, 10
    psrlw     xmm4, 5
    pcmpeqb   xmm7, xmm7  // generate mask 0xff00ff00 for Alpha
    psllw     xmm7, 8

    mov       eax, [esp + 4]  // src_rgb565
    mov       edx, [esp + 8]  // dst_argb
    mov       ecx, [esp + 12]  // width
    sub       edx, eax
    sub       edx, eax

 convertloop:
    movdqu    xmm0, [eax]  // fetch 8 pixels of bgr565
    movdqa    xmm1, xmm0
    movdqa    xmm2, xmm0
    pand      xmm1, xmm3  // R in upper 5 bits
    psllw     xmm2, 11  // B in upper 5 bits
    pmulhuw   xmm1, xmm5  // * (256 + 8)
    pmulhuw   xmm2, xmm5  // * (256 + 8)
    psllw     xmm1, 8
    por       xmm1, xmm2  // RB
    pand      xmm0, xmm4  // G in middle 6 bits
    pmulhuw   xmm0, xmm6  // << 5 * (256 + 4)
    por       xmm0, xmm7  // AG
    movdqa    xmm2, xmm1
    punpcklbw xmm1, xmm0
    punpckhbw xmm2, xmm0
    movdqu    [eax * 2 + edx], xmm1  // store 4 pixels of ARGB
    movdqu    [eax * 2 + edx + 16], xmm2  // store next 4 pixels of ARGB
    lea       eax, [eax + 16]
    sub       ecx, 8
    jg        convertloop
    ret
  }
}

#ifdef HAS_RGB565TOARGBROW_AVX2
// pmul method to replicate bits.
// Math to replicate bits:
// (v << 8) | (v << 3)
// v * 256 + v * 8
// v * (256 + 8)
// G shift of 5 is incorporated, so shift is 5 + 8 and 5 + 3
__declspec(naked) void RGB565ToARGBRow_AVX2(const uint8_t* src_rgb565,
                                            uint8_t* dst_argb,
                                            int width) {
  __asm {
    mov        eax, 0x01080108  // generate multiplier to repeat 5 bits
    vmovd      xmm5, eax
    vbroadcastss ymm5, xmm5
    mov        eax, 0x20802080  // multiplier shift by 5 and then repeat 6 bits
    vmovd      xmm6, eax
    vbroadcastss ymm6, xmm6
    vpcmpeqb   ymm3, ymm3, ymm3  // generate mask 0xf800f800 for Red
    vpsllw     ymm3, ymm3, 11
    vpcmpeqb   ymm4, ymm4, ymm4  // generate mask 0x07e007e0 for Green
    vpsllw     ymm4, ymm4, 10
    vpsrlw     ymm4, ymm4, 5
    vpcmpeqb   ymm7, ymm7, ymm7  // generate mask 0xff00ff00 for Alpha
    vpsllw     ymm7, ymm7, 8

    mov        eax, [esp + 4]  // src_rgb565
    mov        edx, [esp + 8]  // dst_argb
    mov        ecx, [esp + 12]  // width
    sub        edx, eax
    sub        edx, eax

 convertloop:
    vmovdqu    ymm0, [eax]  // fetch 16 pixels of bgr565
    vpand      ymm1, ymm0, ymm3  // R in upper 5 bits
    vpsllw     ymm2, ymm0, 11  // B in upper 5 bits
    vpmulhuw   ymm1, ymm1, ymm5  // * (256 + 8)
    vpmulhuw   ymm2, ymm2, ymm5  // * (256 + 8)
    vpsllw     ymm1, ymm1, 8
    vpor       ymm1, ymm1, ymm2  // RB
    vpand      ymm0, ymm0, ymm4  // G in middle 6 bits
    vpmulhuw   ymm0, ymm0, ymm6  // << 5 * (256 + 4)
    vpor       ymm0, ymm0, ymm7  // AG
    vpermq     ymm0, ymm0, 0xd8  // mutate for unpack
    vpermq     ymm1, ymm1, 0xd8
    vpunpckhbw ymm2, ymm1, ymm0
    vpunpcklbw ymm1, ymm1, ymm0
    vmovdqu    [eax * 2 + edx], ymm1  // store 4 pixels of ARGB
    vmovdqu    [eax * 2 + edx + 32], ymm2  // store next 4 pixels of ARGB
    lea       eax, [eax + 32]
    sub       ecx, 16
    jg        convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_RGB565TOARGBROW_AVX2

#ifdef HAS_ARGB1555TOARGBROW_AVX2
__declspec(naked) void ARGB1555ToARGBRow_AVX2(const uint8_t* src_argb1555,
                                              uint8_t* dst_argb,
                                              int width) {
  __asm {
    mov        eax, 0x01080108  // generate multiplier to repeat 5 bits
    vmovd      xmm5, eax
    vbroadcastss ymm5, xmm5
    mov        eax, 0x42004200  // multiplier shift by 6 and then repeat 5 bits
    vmovd      xmm6, eax
    vbroadcastss ymm6, xmm6
    vpcmpeqb   ymm3, ymm3, ymm3  // generate mask 0xf800f800 for Red
    vpsllw     ymm3, ymm3, 11
    vpsrlw     ymm4, ymm3, 6  // generate mask 0x03e003e0 for Green
    vpcmpeqb   ymm7, ymm7, ymm7  // generate mask 0xff00ff00 for Alpha
    vpsllw     ymm7, ymm7, 8

    mov        eax,  [esp + 4]  // src_argb1555
    mov        edx,  [esp + 8]  // dst_argb
    mov        ecx,  [esp + 12]  // width
    sub        edx,  eax
    sub        edx,  eax

 convertloop:
    vmovdqu    ymm0, [eax]  // fetch 16 pixels of 1555
    vpsllw     ymm1, ymm0, 1  // R in upper 5 bits
    vpsllw     ymm2, ymm0, 11  // B in upper 5 bits
    vpand      ymm1, ymm1, ymm3
    vpmulhuw   ymm2, ymm2, ymm5  // * (256 + 8)
    vpmulhuw   ymm1, ymm1, ymm5  // * (256 + 8)
    vpsllw     ymm1, ymm1, 8
    vpor       ymm1, ymm1, ymm2  // RB
    vpsraw     ymm2, ymm0, 8  // A
    vpand      ymm0, ymm0, ymm4  // G in middle 5 bits
    vpmulhuw   ymm0, ymm0, ymm6  // << 6 * (256 + 8)
    vpand      ymm2, ymm2, ymm7
    vpor       ymm0, ymm0, ymm2  // AG
    vpermq     ymm0, ymm0, 0xd8  // mutate for unpack
    vpermq     ymm1, ymm1, 0xd8
    vpunpckhbw ymm2, ymm1, ymm0
    vpunpcklbw ymm1, ymm1, ymm0
    vmovdqu    [eax * 2 + edx], ymm1  // store 8 pixels of ARGB
    vmovdqu    [eax * 2 + edx + 32], ymm2  // store next 8 pixels of ARGB
    lea       eax, [eax + 32]
    sub       ecx, 16
    jg        convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGB1555TOARGBROW_AVX2

#ifdef HAS_ARGB4444TOARGBROW_AVX2
__declspec(naked) void ARGB4444ToARGBRow_AVX2(const uint8_t* src_argb4444,
                                              uint8_t* dst_argb,
                                              int width) {
  __asm {
    mov       eax,  0x0f0f0f0f  // generate mask 0x0f0f0f0f
    vmovd     xmm4, eax
    vbroadcastss ymm4, xmm4
    vpslld    ymm5, ymm4, 4  // 0xf0f0f0f0 for high nibbles
    mov       eax,  [esp + 4]  // src_argb4444
    mov       edx,  [esp + 8]  // dst_argb
    mov       ecx,  [esp + 12]  // width
    sub       edx,  eax
    sub       edx,  eax

 convertloop:
    vmovdqu    ymm0, [eax]  // fetch 16 pixels of bgra4444
    vpand      ymm2, ymm0, ymm5  // mask high nibbles
    vpand      ymm0, ymm0, ymm4  // mask low nibbles
    vpsrlw     ymm3, ymm2, 4
    vpsllw     ymm1, ymm0, 4
    vpor       ymm2, ymm2, ymm3
    vpor       ymm0, ymm0, ymm1
    vpermq     ymm0, ymm0, 0xd8  // mutate for unpack
    vpermq     ymm2, ymm2, 0xd8
    vpunpckhbw ymm1, ymm0, ymm2
    vpunpcklbw ymm0, ymm0, ymm2
    vmovdqu    [eax * 2 + edx], ymm0  // store 8 pixels of ARGB
    vmovdqu    [eax * 2 + edx + 32], ymm1  // store next 8 pixels of ARGB
    lea       eax, [eax + 32]
    sub       ecx, 16
    jg        convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGB4444TOARGBROW_AVX2

// 24 instructions
__declspec(naked) void ARGB1555ToARGBRow_SSE2(const uint8_t* src_argb1555,
                                              uint8_t* dst_argb,
                                              int width) {
  __asm {
    mov       eax, 0x01080108  // generate multiplier to repeat 5 bits
    movd      xmm5, eax
    pshufd    xmm5, xmm5, 0
    mov       eax, 0x42004200  // multiplier shift by 6 and then repeat 5 bits
    movd      xmm6, eax
    pshufd    xmm6, xmm6, 0
    pcmpeqb   xmm3, xmm3  // generate mask 0xf800f800 for Red
    psllw     xmm3, 11
    movdqa    xmm4, xmm3  // generate mask 0x03e003e0 for Green
    psrlw     xmm4, 6
    pcmpeqb   xmm7, xmm7  // generate mask 0xff00ff00 for Alpha
    psllw     xmm7, 8

    mov       eax, [esp + 4]  // src_argb1555
    mov       edx, [esp + 8]  // dst_argb
    mov       ecx, [esp + 12]  // width
    sub       edx, eax
    sub       edx, eax

 convertloop:
    movdqu    xmm0, [eax]  // fetch 8 pixels of 1555
    movdqa    xmm1, xmm0
    movdqa    xmm2, xmm0
    psllw     xmm1, 1  // R in upper 5 bits
    psllw     xmm2, 11  // B in upper 5 bits
    pand      xmm1, xmm3
    pmulhuw   xmm2, xmm5  // * (256 + 8)
    pmulhuw   xmm1, xmm5  // * (256 + 8)
    psllw     xmm1, 8
    por       xmm1, xmm2  // RB
    movdqa    xmm2, xmm0
    pand      xmm0, xmm4  // G in middle 5 bits
    psraw     xmm2, 8  // A
    pmulhuw   xmm0, xmm6  // << 6 * (256 + 8)
    pand      xmm2, xmm7
    por       xmm0, xmm2  // AG
    movdqa    xmm2, xmm1
    punpcklbw xmm1, xmm0
    punpckhbw xmm2, xmm0
    movdqu    [eax * 2 + edx], xmm1  // store 4 pixels of ARGB
    movdqu    [eax * 2 + edx + 16], xmm2  // store next 4 pixels of ARGB
    lea       eax, [eax + 16]
    sub       ecx, 8
    jg        convertloop
    ret
  }
}

// 18 instructions.
__declspec(naked) void ARGB4444ToARGBRow_SSE2(const uint8_t* src_argb4444,
                                              uint8_t* dst_argb,
                                              int width) {
  __asm {
    mov       eax, 0x0f0f0f0f  // generate mask 0x0f0f0f0f
    movd      xmm4, eax
    pshufd    xmm4, xmm4, 0
    movdqa    xmm5, xmm4  // 0xf0f0f0f0 for high nibbles
    pslld     xmm5, 4
    mov       eax, [esp + 4]  // src_argb4444
    mov       edx, [esp + 8]  // dst_argb
    mov       ecx, [esp + 12]  // width
    sub       edx, eax
    sub       edx, eax

 convertloop:
    movdqu    xmm0, [eax]  // fetch 8 pixels of bgra4444
    movdqa    xmm2, xmm0
    pand      xmm0, xmm4  // mask low nibbles
    pand      xmm2, xmm5  // mask high nibbles
    movdqa    xmm1, xmm0
    movdqa    xmm3, xmm2
    psllw     xmm1, 4
    psrlw     xmm3, 4
    por       xmm0, xmm1
    por       xmm2, xmm3
    movdqa    xmm1, xmm0
    punpcklbw xmm0, xmm2
    punpckhbw xmm1, xmm2
    movdqu    [eax * 2 + edx], xmm0  // store 4 pixels of ARGB
    movdqu    [eax * 2 + edx + 16], xmm1  // store next 4 pixels of ARGB
    lea       eax, [eax + 16]
    sub       ecx, 8
    jg        convertloop
    ret
  }
}

__declspec(naked) void ARGBToRGB24Row_SSSE3(const uint8_t* src_argb,
                                            uint8_t* dst_rgb,
                                            int width) {
  __asm {
    mov       eax, [esp + 4]  // src_argb
    mov       edx, [esp + 8]  // dst_rgb
    mov       ecx, [esp + 12]  // width
    movdqa    xmm6, xmmword ptr kShuffleMaskARGBToRGB24

 convertloop:
    movdqu    xmm0, [eax]  // fetch 16 pixels of argb
    movdqu    xmm1, [eax + 16]
    movdqu    xmm2, [eax + 32]
    movdqu    xmm3, [eax + 48]
    lea       eax, [eax + 64]
    pshufb    xmm0, xmm6  // pack 16 bytes of ARGB to 12 bytes of RGB
    pshufb    xmm1, xmm6
    pshufb    xmm2, xmm6
    pshufb    xmm3, xmm6
    movdqa    xmm4, xmm1  // 4 bytes from 1 for 0
    psrldq    xmm1, 4  // 8 bytes from 1
    pslldq    xmm4, 12  // 4 bytes from 1 for 0
    movdqa    xmm5, xmm2  // 8 bytes from 2 for 1
    por       xmm0, xmm4  // 4 bytes from 1 for 0
    pslldq    xmm5, 8  // 8 bytes from 2 for 1
    movdqu    [edx], xmm0  // store 0
    por       xmm1, xmm5  // 8 bytes from 2 for 1
    psrldq    xmm2, 8  // 4 bytes from 2
    pslldq    xmm3, 4  // 12 bytes from 3 for 2
    por       xmm2, xmm3  // 12 bytes from 3 for 2
    movdqu    [edx + 16], xmm1  // store 1
    movdqu    [edx + 32], xmm2  // store 2
    lea       edx, [edx + 48]
    sub       ecx, 16
    jg        convertloop
    ret
  }
}

__declspec(naked) void ARGBToRAWRow_SSSE3(const uint8_t* src_argb,
                                          uint8_t* dst_rgb,
                                          int width) {
  __asm {
    mov       eax, [esp + 4]  // src_argb
    mov       edx, [esp + 8]  // dst_rgb
    mov       ecx, [esp + 12]  // width
    movdqa    xmm6, xmmword ptr kShuffleMaskARGBToRAW

 convertloop:
    movdqu    xmm0, [eax]  // fetch 16 pixels of argb
    movdqu    xmm1, [eax + 16]
    movdqu    xmm2, [eax + 32]
    movdqu    xmm3, [eax + 48]
    lea       eax, [eax + 64]
    pshufb    xmm0, xmm6  // pack 16 bytes of ARGB to 12 bytes of RGB
    pshufb    xmm1, xmm6
    pshufb    xmm2, xmm6
    pshufb    xmm3, xmm6
    movdqa    xmm4, xmm1  // 4 bytes from 1 for 0
    psrldq    xmm1, 4  // 8 bytes from 1
    pslldq    xmm4, 12  // 4 bytes from 1 for 0
    movdqa    xmm5, xmm2  // 8 bytes from 2 for 1
    por       xmm0, xmm4  // 4 bytes from 1 for 0
    pslldq    xmm5, 8  // 8 bytes from 2 for 1
    movdqu    [edx], xmm0  // store 0
    por       xmm1, xmm5  // 8 bytes from 2 for 1
    psrldq    xmm2, 8  // 4 bytes from 2
    pslldq    xmm3, 4  // 12 bytes from 3 for 2
    por       xmm2, xmm3  // 12 bytes from 3 for 2
    movdqu    [edx + 16], xmm1  // store 1
    movdqu    [edx + 32], xmm2  // store 2
    lea       edx, [edx + 48]
    sub       ecx, 16
    jg        convertloop
    ret
  }
}

__declspec(naked) void ARGBToRGB565Row_SSE2(const uint8_t* src_argb,
                                            uint8_t* dst_rgb,
                                            int width) {
  __asm {
    mov       eax, [esp + 4]  // src_argb
    mov       edx, [esp + 8]  // dst_rgb
    mov       ecx, [esp + 12]  // width
    pcmpeqb   xmm3, xmm3  // generate mask 0x0000001f
    psrld     xmm3, 27
    pcmpeqb   xmm4, xmm4  // generate mask 0x000007e0
    psrld     xmm4, 26
    pslld     xmm4, 5
    pcmpeqb   xmm5, xmm5  // generate mask 0xfffff800
    pslld     xmm5, 11

 convertloop:
    movdqu    xmm0, [eax]  // fetch 4 pixels of argb
    movdqa    xmm1, xmm0  // B
    movdqa    xmm2, xmm0  // G
    pslld     xmm0, 8  // R
    psrld     xmm1, 3  // B
    psrld     xmm2, 5  // G
    psrad     xmm0, 16  // R
    pand      xmm1, xmm3  // B
    pand      xmm2, xmm4  // G
    pand      xmm0, xmm5  // R
    por       xmm1, xmm2  // BG
    por       xmm0, xmm1  // BGR
    packssdw  xmm0, xmm0
    lea       eax, [eax + 16]
    movq      qword ptr [edx], xmm0  // store 4 pixels of RGB565
    lea       edx, [edx + 8]
    sub       ecx, 4
    jg        convertloop
    ret
  }
}

__declspec(naked) void ARGBToRGB565DitherRow_SSE2(const uint8_t* src_argb,
                                                  uint8_t* dst_rgb,
                                                  uint32_t dither4,
                                                  int width) {
  __asm {

    mov       eax, [esp + 4]  // src_argb
    mov       edx, [esp + 8]  // dst_rgb
    movd      xmm6, [esp + 12]  // dither4
    mov       ecx, [esp + 16]  // width
    punpcklbw xmm6, xmm6  // make dither 16 bytes
    movdqa    xmm7, xmm6
    punpcklwd xmm6, xmm6
    punpckhwd xmm7, xmm7
    pcmpeqb   xmm3, xmm3  // generate mask 0x0000001f
    psrld     xmm3, 27
    pcmpeqb   xmm4, xmm4  // generate mask 0x000007e0
    psrld     xmm4, 26
    pslld     xmm4, 5
    pcmpeqb   xmm5, xmm5  // generate mask 0xfffff800
    pslld     xmm5, 11

 convertloop:
    movdqu    xmm0, [eax]  // fetch 4 pixels of argb
    paddusb   xmm0, xmm6  // add dither
    movdqa    xmm1, xmm0  // B
    movdqa    xmm2, xmm0  // G
    pslld     xmm0, 8  // R
    psrld     xmm1, 3  // B
    psrld     xmm2, 5  // G
    psrad     xmm0, 16  // R
    pand      xmm1, xmm3  // B
    pand      xmm2, xmm4  // G
    pand      xmm0, xmm5  // R
    por       xmm1, xmm2  // BG
    por       xmm0, xmm1  // BGR
    packssdw  xmm0, xmm0
    lea       eax, [eax + 16]
    movq      qword ptr [edx], xmm0  // store 4 pixels of RGB565
    lea       edx, [edx + 8]
    sub       ecx, 4
    jg        convertloop
    ret
  }
}

#ifdef HAS_ARGBTORGB565DITHERROW_AVX2
__declspec(naked) void ARGBToRGB565DitherRow_AVX2(const uint8_t* src_argb,
                                                  uint8_t* dst_rgb,
                                                  uint32_t dither4,
                                                  int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_rgb
    vbroadcastss xmm6, [esp + 12]  // dither4
    mov        ecx, [esp + 16]  // width
    vpunpcklbw xmm6, xmm6, xmm6  // make dither 32 bytes
    vpermq     ymm6, ymm6, 0xd8
    vpunpcklwd ymm6, ymm6, ymm6
    vpcmpeqb   ymm3, ymm3, ymm3  // generate mask 0x0000001f
    vpsrld     ymm3, ymm3, 27
    vpcmpeqb   ymm4, ymm4, ymm4  // generate mask 0x000007e0
    vpsrld     ymm4, ymm4, 26
    vpslld     ymm4, ymm4, 5
    vpslld     ymm5, ymm3, 11  // generate mask 0x0000f800

 convertloop:
    vmovdqu    ymm0, [eax]  // fetch 8 pixels of argb
    vpaddusb   ymm0, ymm0, ymm6  // add dither
    vpsrld     ymm2, ymm0, 5  // G
    vpsrld     ymm1, ymm0, 3  // B
    vpsrld     ymm0, ymm0, 8  // R
    vpand      ymm2, ymm2, ymm4  // G
    vpand      ymm1, ymm1, ymm3  // B
    vpand      ymm0, ymm0, ymm5  // R
    vpor       ymm1, ymm1, ymm2  // BG
    vpor       ymm0, ymm0, ymm1  // BGR
    vpackusdw  ymm0, ymm0, ymm0
    vpermq     ymm0, ymm0, 0xd8
    lea        eax, [eax + 32]
    vmovdqu    [edx], xmm0  // store 8 pixels of RGB565
    lea        edx, [edx + 16]
    sub        ecx, 8
    jg         convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBTORGB565DITHERROW_AVX2

// TODO(fbarchard): Improve sign extension/packing.
__declspec(naked) void ARGBToARGB1555Row_SSE2(const uint8_t* src_argb,
                                              uint8_t* dst_rgb,
                                              int width) {
  __asm {
    mov       eax, [esp + 4]  // src_argb
    mov       edx, [esp + 8]  // dst_rgb
    mov       ecx, [esp + 12]  // width
    pcmpeqb   xmm4, xmm4  // generate mask 0x0000001f
    psrld     xmm4, 27
    movdqa    xmm5, xmm4  // generate mask 0x000003e0
    pslld     xmm5, 5
    movdqa    xmm6, xmm4  // generate mask 0x00007c00
    pslld     xmm6, 10
    pcmpeqb   xmm7, xmm7  // generate mask 0xffff8000
    pslld     xmm7, 15

 convertloop:
    movdqu    xmm0, [eax]  // fetch 4 pixels of argb
    movdqa    xmm1, xmm0  // B
    movdqa    xmm2, xmm0  // G
    movdqa    xmm3, xmm0  // R
    psrad     xmm0, 16  // A
    psrld     xmm1, 3  // B
    psrld     xmm2, 6  // G
    psrld     xmm3, 9  // R
    pand      xmm0, xmm7  // A
    pand      xmm1, xmm4  // B
    pand      xmm2, xmm5  // G
    pand      xmm3, xmm6  // R
    por       xmm0, xmm1  // BA
    por       xmm2, xmm3  // GR
    por       xmm0, xmm2  // BGRA
    packssdw  xmm0, xmm0
    lea       eax, [eax + 16]
    movq      qword ptr [edx], xmm0  // store 4 pixels of ARGB1555
    lea       edx, [edx + 8]
    sub       ecx, 4
    jg        convertloop
    ret
  }
}

__declspec(naked) void ARGBToARGB4444Row_SSE2(const uint8_t* src_argb,
                                              uint8_t* dst_rgb,
                                              int width) {
  __asm {
    mov       eax, [esp + 4]  // src_argb
    mov       edx, [esp + 8]  // dst_rgb
    mov       ecx, [esp + 12]  // width
    pcmpeqb   xmm4, xmm4  // generate mask 0xf000f000
    psllw     xmm4, 12
    movdqa    xmm3, xmm4  // generate mask 0x00f000f0
    psrlw     xmm3, 8

 convertloop:
    movdqu    xmm0, [eax]  // fetch 4 pixels of argb
    movdqa    xmm1, xmm0
    pand      xmm0, xmm3  // low nibble
    pand      xmm1, xmm4  // high nibble
    psrld     xmm0, 4
    psrld     xmm1, 8
    por       xmm0, xmm1
    packuswb  xmm0, xmm0
    lea       eax, [eax + 16]
    movq      qword ptr [edx], xmm0  // store 4 pixels of ARGB4444
    lea       edx, [edx + 8]
    sub       ecx, 4
    jg        convertloop
    ret
  }
}

#ifdef HAS_ARGBTORGB565ROW_AVX2
__declspec(naked) void ARGBToRGB565Row_AVX2(const uint8_t* src_argb,
                                            uint8_t* dst_rgb,
                                            int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_rgb
    mov        ecx, [esp + 12]  // width
    vpcmpeqb   ymm3, ymm3, ymm3  // generate mask 0x0000001f
    vpsrld     ymm3, ymm3, 27
    vpcmpeqb   ymm4, ymm4, ymm4  // generate mask 0x000007e0
    vpsrld     ymm4, ymm4, 26
    vpslld     ymm4, ymm4, 5
    vpslld     ymm5, ymm3, 11  // generate mask 0x0000f800

 convertloop:
    vmovdqu    ymm0, [eax]  // fetch 8 pixels of argb
    vpsrld     ymm2, ymm0, 5  // G
    vpsrld     ymm1, ymm0, 3  // B
    vpsrld     ymm0, ymm0, 8  // R
    vpand      ymm2, ymm2, ymm4  // G
    vpand      ymm1, ymm1, ymm3  // B
    vpand      ymm0, ymm0, ymm5  // R
    vpor       ymm1, ymm1, ymm2  // BG
    vpor       ymm0, ymm0, ymm1  // BGR
    vpackusdw  ymm0, ymm0, ymm0
    vpermq     ymm0, ymm0, 0xd8
    lea        eax, [eax + 32]
    vmovdqu    [edx], xmm0  // store 8 pixels of RGB565
    lea        edx, [edx + 16]
    sub        ecx, 8
    jg         convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBTORGB565ROW_AVX2

#ifdef HAS_ARGBTOARGB1555ROW_AVX2
__declspec(naked) void ARGBToARGB1555Row_AVX2(const uint8_t* src_argb,
                                              uint8_t* dst_rgb,
                                              int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_rgb
    mov        ecx, [esp + 12]  // width
    vpcmpeqb   ymm4, ymm4, ymm4
    vpsrld     ymm4, ymm4, 27  // generate mask 0x0000001f
    vpslld     ymm5, ymm4, 5  // generate mask 0x000003e0
    vpslld     ymm6, ymm4, 10  // generate mask 0x00007c00
    vpcmpeqb   ymm7, ymm7, ymm7  // generate mask 0xffff8000
    vpslld     ymm7, ymm7, 15

 convertloop:
    vmovdqu    ymm0, [eax]  // fetch 8 pixels of argb
    vpsrld     ymm3, ymm0, 9  // R
    vpsrld     ymm2, ymm0, 6  // G
    vpsrld     ymm1, ymm0, 3  // B
    vpsrad     ymm0, ymm0, 16  // A
    vpand      ymm3, ymm3, ymm6  // R
    vpand      ymm2, ymm2, ymm5  // G
    vpand      ymm1, ymm1, ymm4  // B
    vpand      ymm0, ymm0, ymm7  // A
    vpor       ymm0, ymm0, ymm1  // BA
    vpor       ymm2, ymm2, ymm3  // GR
    vpor       ymm0, ymm0, ymm2  // BGRA
    vpackssdw  ymm0, ymm0, ymm0
    vpermq     ymm0, ymm0, 0xd8
    lea        eax, [eax + 32]
    vmovdqu    [edx], xmm0  // store 8 pixels of ARGB1555
    lea        edx, [edx + 16]
    sub        ecx, 8
    jg         convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBTOARGB1555ROW_AVX2

#ifdef HAS_ARGBTOARGB4444ROW_AVX2
__declspec(naked) void ARGBToARGB4444Row_AVX2(const uint8_t* src_argb,
                                              uint8_t* dst_rgb,
                                              int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_rgb
    mov        ecx, [esp + 12]  // width
    vpcmpeqb   ymm4, ymm4, ymm4  // generate mask 0xf000f000
    vpsllw     ymm4, ymm4, 12
    vpsrlw     ymm3, ymm4, 8  // generate mask 0x00f000f0

 convertloop:
    vmovdqu    ymm0, [eax]  // fetch 8 pixels of argb
    vpand      ymm1, ymm0, ymm4  // high nibble
    vpand      ymm0, ymm0, ymm3  // low nibble
    vpsrld     ymm1, ymm1, 8
    vpsrld     ymm0, ymm0, 4
    vpor       ymm0, ymm0, ymm1
    vpackuswb  ymm0, ymm0, ymm0
    vpermq     ymm0, ymm0, 0xd8
    lea        eax, [eax + 32]
    vmovdqu    [edx], xmm0  // store 8 pixels of ARGB4444
    lea        edx, [edx + 16]
    sub        ecx, 8
    jg         convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBTOARGB4444ROW_AVX2

// Convert 16 ARGB pixels (64 bytes) to 16 Y values.
__declspec(naked) void ARGBToYRow_SSSE3(const uint8_t* src_argb,
                                        uint8_t* dst_y,
                                        int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_y */
    mov        ecx, [esp + 12] /* width */
    movdqa     xmm4, xmmword ptr kARGBToY
    movdqa     xmm5, xmmword ptr kAddY16

 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop
    ret
  }
}

#ifdef HAS_ARGBTOUVROW_SSSE3

// Convert 16 ARGB pixels (64 bytes) to 16 YJ values.
// Same as ARGBToYRow but different coefficients, no add 16, but do rounding.
__declspec(naked) void ARGBToYJRow_SSSE3(const uint8_t* src_argb,
                                         uint8_t* dst_y,
                                         int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_y */
    mov        ecx, [esp + 12] /* width */
    movdqa     xmm4, xmmword ptr kARGBToYJ
    movdqa     xmm5, xmmword ptr kAddYJ64

 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    paddw      xmm0, xmm5  // Add .5 for rounding.
    paddw      xmm2, xmm5
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop
    ret
  }
}
#endif

#ifdef HAS_ARGBTOYROW_AVX2

// Convert 32 ARGB pixels (128 bytes) to 32 Y values.
__declspec(naked) void ARGBToYRow_AVX2(const uint8_t* src_argb,
                                       uint8_t* dst_y,
                                       int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_y */
    mov        ecx, [esp + 12] /* width */
    vbroadcastf128 ymm4, xmmword ptr kARGBToY
    vbroadcastf128 ymm5, xmmword ptr kAddY16
    vmovdqu    ymm6, ymmword ptr kPermdARGBToY_AVX

 convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vmovdqu    ymm2, [eax + 64]
    vmovdqu    ymm3, [eax + 96]
    vpmaddubsw ymm0, ymm0, ymm4
    vpmaddubsw ymm1, ymm1, ymm4
    vpmaddubsw ymm2, ymm2, ymm4
    vpmaddubsw ymm3, ymm3, ymm4
    lea        eax, [eax + 128]
    vphaddw    ymm0, ymm0, ymm1  // mutates.
    vphaddw    ymm2, ymm2, ymm3
    vpsrlw     ymm0, ymm0, 7
    vpsrlw     ymm2, ymm2, 7
    vpackuswb  ymm0, ymm0, ymm2  // mutates.
    vpermd     ymm0, ymm6, ymm0  // For vphaddw + vpackuswb mutation.
    vpaddb     ymm0, ymm0, ymm5  // add 16 for Y
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         convertloop
    vzeroupper
    ret
  }
}
#endif  //  HAS_ARGBTOYROW_AVX2

#ifdef HAS_ARGBTOYJROW_AVX2
// Convert 32 ARGB pixels (128 bytes) to 32 Y values.
__declspec(naked) void ARGBToYJRow_AVX2(const uint8_t* src_argb,
                                        uint8_t* dst_y,
                                        int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_y */
    mov        ecx, [esp + 12] /* width */
    vbroadcastf128 ymm4, xmmword ptr kARGBToYJ
    vbroadcastf128 ymm5, xmmword ptr kAddYJ64
    vmovdqu    ymm6, ymmword ptr kPermdARGBToY_AVX

 convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vmovdqu    ymm2, [eax + 64]
    vmovdqu    ymm3, [eax + 96]
    vpmaddubsw ymm0, ymm0, ymm4
    vpmaddubsw ymm1, ymm1, ymm4
    vpmaddubsw ymm2, ymm2, ymm4
    vpmaddubsw ymm3, ymm3, ymm4
    lea        eax, [eax + 128]
    vphaddw    ymm0, ymm0, ymm1  // mutates.
    vphaddw    ymm2, ymm2, ymm3
    vpaddw     ymm0, ymm0, ymm5  // Add .5 for rounding.
    vpaddw     ymm2, ymm2, ymm5
    vpsrlw     ymm0, ymm0, 7
    vpsrlw     ymm2, ymm2, 7
    vpackuswb  ymm0, ymm0, ymm2  // mutates.
    vpermd     ymm0, ymm6, ymm0  // For vphaddw + vpackuswb mutation.
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         convertloop

    vzeroupper
    ret
  }
}
#endif  //  HAS_ARGBTOYJROW_AVX2

__declspec(naked) void BGRAToYRow_SSSE3(const uint8_t* src_argb,
                                        uint8_t* dst_y,
                                        int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_y */
    mov        ecx, [esp + 12] /* width */
    movdqa     xmm4, xmmword ptr kBGRAToY
    movdqa     xmm5, xmmword ptr kAddY16

 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop
    ret
  }
}

__declspec(naked) void ABGRToYRow_SSSE3(const uint8_t* src_argb,
                                        uint8_t* dst_y,
                                        int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_y */
    mov        ecx, [esp + 12] /* width */
    movdqa     xmm4, xmmword ptr kABGRToY
    movdqa     xmm5, xmmword ptr kAddY16

 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop
    ret
  }
}

__declspec(naked) void RGBAToYRow_SSSE3(const uint8_t* src_argb,
                                        uint8_t* dst_y,
                                        int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_y */
    mov        ecx, [esp + 12] /* width */
    movdqa     xmm4, xmmword ptr kRGBAToY
    movdqa     xmm5, xmmword ptr kAddY16

 convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm2, xmm4
    pmaddubsw  xmm3, xmm4
    lea        eax, [eax + 64]
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psrlw      xmm0, 7
    psrlw      xmm2, 7
    packuswb   xmm0, xmm2
    paddb      xmm0, xmm5
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop
    ret
  }
}

#ifdef HAS_ARGBTOUVROW_SSSE3

__declspec(naked) void ARGBToUVRow_SSSE3(const uint8_t* src_argb,
                                         int src_stride_argb,
                                         uint8_t* dst_u,
                                         uint8_t* dst_v,
                                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_argb
    mov        esi, [esp + 8 + 8]  // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    movdqa     xmm5, xmmword ptr kBiasUV128
    movdqa     xmm6, xmmword ptr kARGBToV
    movdqa     xmm7, xmmword ptr kARGBToU
    sub        edi, edx  // stride from u to v

 convertloop:
         /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm1, [eax + 16]
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm2, [eax + 32]
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4

    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

        // step 2 - convert to U and V
        // from here down is very similar to Y code except
        // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5  // -> unsigned

        // step 3 - store 8 U and 8 V values
    movlps     qword ptr [edx], xmm0  // U
    movhps     qword ptr [edx + edi], xmm0  // V
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) void ARGBToUVJRow_SSSE3(const uint8_t* src_argb,
                                          int src_stride_argb,
                                          uint8_t* dst_u,
                                          uint8_t* dst_v,
                                          int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_argb
    mov        esi, [esp + 8 + 8]  // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    // TODO: change biasuv to 0x8000
    movdqa     xmm5, xmmword ptr kBiasUV128
        // TODO: use negated coefficients to allow -128
    movdqa     xmm6, xmmword ptr kARGBToVJ
    movdqa     xmm7, xmmword ptr kARGBToUJ
    sub        edi, edx  // stride from u to v

 convertloop:
         /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm1, [eax + 16]
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm2, [eax + 32]
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4

    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

        // step 2 - convert to U and V
        // from here down is very similar to Y code except
        // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
        // TODO: negate by subtracting from 0x8000
    paddw      xmm0, xmm5  // +.5 rounding -> unsigned
    paddw      xmm1, xmm5
    psraw      xmm0, 8
    psraw      xmm1, 8
    // TODO: packuswb
    packsswb   xmm0, xmm1

        // step 3 - store 8 U and 8 V values
    movlps     qword ptr [edx], xmm0  // U
    movhps     qword ptr [edx + edi], xmm0  // V
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}
#endif

#ifdef HAS_ARGBTOUVROW_AVX2
__declspec(naked) void ARGBToUVRow_AVX2(const uint8_t* src_argb,
                                        int src_stride_argb,
                                        uint8_t* dst_u,
                                        uint8_t* dst_v,
                                        int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_argb
    mov        esi, [esp + 8 + 8]  // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    vbroadcastf128 ymm5, xmmword ptr kBiasUV128
    vbroadcastf128 ymm6, xmmword ptr kARGBToV
    vbroadcastf128 ymm7, xmmword ptr kARGBToU
    sub        edi, edx   // stride from u to v

 convertloop:
        /* step 1 - subsample 32x2 argb pixels to 16x1 */
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vmovdqu    ymm2, [eax + 64]
    vmovdqu    ymm3, [eax + 96]
    vpavgb     ymm0, ymm0, [eax + esi]
    vpavgb     ymm1, ymm1, [eax + esi + 32]
    vpavgb     ymm2, ymm2, [eax + esi + 64]
    vpavgb     ymm3, ymm3, [eax + esi + 96]
    lea        eax,  [eax + 128]
    vshufps    ymm4, ymm0, ymm1, 0x88
    vshufps    ymm0, ymm0, ymm1, 0xdd
    vpavgb     ymm0, ymm0, ymm4  // mutated by vshufps
    vshufps    ymm4, ymm2, ymm3, 0x88
    vshufps    ymm2, ymm2, ymm3, 0xdd
    vpavgb     ymm2, ymm2, ymm4  // mutated by vshufps

        // step 2 - convert to U and V
        // from here down is very similar to Y code except
        // instead of 32 different pixels, its 16 pixels of U and 16 of V
    vpmaddubsw ymm1, ymm0, ymm7  // U
    vpmaddubsw ymm3, ymm2, ymm7
    vpmaddubsw ymm0, ymm0, ymm6  // V
    vpmaddubsw ymm2, ymm2, ymm6
    vphaddw    ymm1, ymm1, ymm3  // mutates
    vphaddw    ymm0, ymm0, ymm2
    vpsraw     ymm1, ymm1, 8
    vpsraw     ymm0, ymm0, 8
    vpacksswb  ymm0, ymm1, ymm0  // mutates
    vpermq     ymm0, ymm0, 0xd8  // For vpacksswb
    vpshufb    ymm0, ymm0, ymmword ptr kShufARGBToUV_AVX  // for vshufps/vphaddw
    vpaddb     ymm0, ymm0, ymm5  // -> unsigned

        // step 3 - store 16 U and 16 V values
    vextractf128 [edx], ymm0, 0  // U
    vextractf128 [edx + edi], ymm0, 1  // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBTOUVROW_AVX2

#ifdef HAS_ARGBTOUVJROW_AVX2
__declspec(naked) void ARGBToUVJRow_AVX2(const uint8_t* src_argb,
                                         int src_stride_argb,
                                         uint8_t* dst_u,
                                         uint8_t* dst_v,
                                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_argb
    mov        esi, [esp + 8 + 8]  // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    vbroadcastf128 ymm5, xmmword ptr kBiasUV128
    vbroadcastf128 ymm6, xmmword ptr kARGBToVJ
    vbroadcastf128 ymm7, xmmword ptr kARGBToUJ
    sub        edi, edx   // stride from u to v

 convertloop:
        /* step 1 - subsample 32x2 argb pixels to 16x1 */
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vmovdqu    ymm2, [eax + 64]
    vmovdqu    ymm3, [eax + 96]
    vpavgb     ymm0, ymm0, [eax + esi]
    vpavgb     ymm1, ymm1, [eax + esi + 32]
    vpavgb     ymm2, ymm2, [eax + esi + 64]
    vpavgb     ymm3, ymm3, [eax + esi + 96]
    lea        eax,  [eax + 128]
    vshufps    ymm4, ymm0, ymm1, 0x88
    vshufps    ymm0, ymm0, ymm1, 0xdd
    vpavgb     ymm0, ymm0, ymm4  // mutated by vshufps
    vshufps    ymm4, ymm2, ymm3, 0x88
    vshufps    ymm2, ymm2, ymm3, 0xdd
    vpavgb     ymm2, ymm2, ymm4  // mutated by vshufps

        // step 2 - convert to U and V
        // from here down is very similar to Y code except
        // instead of 32 different pixels, its 16 pixels of U and 16 of V
    vpmaddubsw ymm1, ymm0, ymm7  // U
    vpmaddubsw ymm3, ymm2, ymm7
    vpmaddubsw ymm0, ymm0, ymm6  // V
    vpmaddubsw ymm2, ymm2, ymm6
    vphaddw    ymm1, ymm1, ymm3  // mutates
    vphaddw    ymm0, ymm0, ymm2
    vpaddw     ymm1, ymm1, ymm5  // +.5 rounding -> unsigned
    vpaddw     ymm0, ymm0, ymm5
    vpsraw     ymm1, ymm1, 8
    vpsraw     ymm0, ymm0, 8
    vpacksswb  ymm0, ymm1, ymm0  // mutates
    vpermq     ymm0, ymm0, 0xd8  // For vpacksswb
    vpshufb    ymm0, ymm0, ymmword ptr kShufARGBToUV_AVX  // for vshufps/vphaddw

        // step 3 - store 16 U and 16 V values
    vextractf128 [edx], ymm0, 0  // U
    vextractf128 [edx + edi], ymm0, 1  // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBTOUVJROW_AVX2

__declspec(naked) void ARGBToUV444Row_SSSE3(const uint8_t* src_argb,
                                            uint8_t* dst_u,
                                            uint8_t* dst_v,
                                            int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]  // src_argb
    mov        edx, [esp + 4 + 8]  // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // width
    movdqa     xmm5, xmmword ptr kBiasUV128
    movdqa     xmm6, xmmword ptr kARGBToV
    movdqa     xmm7, xmmword ptr kARGBToU
    sub        edi, edx    // stride from u to v

 convertloop:
        /* convert to U and V */
    movdqu     xmm0, [eax]  // U
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm7
    pmaddubsw  xmm1, xmm7
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm3, xmm7
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psraw      xmm0, 8
    psraw      xmm2, 8
    packsswb   xmm0, xmm2
    paddb      xmm0, xmm5
    movdqu     [edx], xmm0

    movdqu     xmm0, [eax]  // V
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]
    pmaddubsw  xmm0, xmm6
    pmaddubsw  xmm1, xmm6
    pmaddubsw  xmm2, xmm6
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm1
    phaddw     xmm2, xmm3
    psraw      xmm0, 8
    psraw      xmm2, 8
    packsswb   xmm0, xmm2
    paddb      xmm0, xmm5
    lea        eax,  [eax + 64]
    movdqu     [edx + edi], xmm0
    lea        edx,  [edx + 16]
    sub        ecx,  16
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) void BGRAToUVRow_SSSE3(const uint8_t* src_argb,
                                         int src_stride_argb,
                                         uint8_t* dst_u,
                                         uint8_t* dst_v,
                                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_argb
    mov        esi, [esp + 8 + 8]  // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    movdqa     xmm5, xmmword ptr kBiasUV128
    movdqa     xmm6, xmmword ptr kBGRAToV
    movdqa     xmm7, xmmword ptr kBGRAToU
    sub        edi, edx  // stride from u to v

 convertloop:
         /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm1, [eax + 16]
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm2, [eax + 32]
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4

    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

        // step 2 - convert to U and V
        // from here down is very similar to Y code except
        // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5  // -> unsigned

        // step 3 - store 8 U and 8 V values
    movlps     qword ptr [edx], xmm0  // U
    movhps     qword ptr [edx + edi], xmm0  // V
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) void ABGRToUVRow_SSSE3(const uint8_t* src_argb,
                                         int src_stride_argb,
                                         uint8_t* dst_u,
                                         uint8_t* dst_v,
                                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_argb
    mov        esi, [esp + 8 + 8]  // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    movdqa     xmm5, xmmword ptr kBiasUV128
    movdqa     xmm6, xmmword ptr kABGRToV
    movdqa     xmm7, xmmword ptr kABGRToU
    sub        edi, edx  // stride from u to v

 convertloop:
         /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm1, [eax + 16]
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm2, [eax + 32]
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4

    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

        // step 2 - convert to U and V
        // from here down is very similar to Y code except
        // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5  // -> unsigned

        // step 3 - store 8 U and 8 V values
    movlps     qword ptr [edx], xmm0  // U
    movhps     qword ptr [edx + edi], xmm0  // V
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) void RGBAToUVRow_SSSE3(const uint8_t* src_argb,
                                         int src_stride_argb,
                                         uint8_t* dst_u,
                                         uint8_t* dst_v,
                                         int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_argb
    mov        esi, [esp + 8 + 8]  // src_stride_argb
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    movdqa     xmm5, xmmword ptr kBiasUV128
    movdqa     xmm6, xmmword ptr kRGBAToV
    movdqa     xmm7, xmmword ptr kRGBAToU
    sub        edi, edx  // stride from u to v

 convertloop:
         /* step 1 - subsample 16x2 argb pixels to 8x1 */
    movdqu     xmm0, [eax]
    movdqu     xmm4, [eax + esi]
    pavgb      xmm0, xmm4
    movdqu     xmm1, [eax + 16]
    movdqu     xmm4, [eax + esi + 16]
    pavgb      xmm1, xmm4
    movdqu     xmm2, [eax + 32]
    movdqu     xmm4, [eax + esi + 32]
    pavgb      xmm2, xmm4
    movdqu     xmm3, [eax + 48]
    movdqu     xmm4, [eax + esi + 48]
    pavgb      xmm3, xmm4

    lea        eax,  [eax + 64]
    movdqa     xmm4, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm4, xmm1, 0xdd
    pavgb      xmm0, xmm4
    movdqa     xmm4, xmm2
    shufps     xmm2, xmm3, 0x88
    shufps     xmm4, xmm3, 0xdd
    pavgb      xmm2, xmm4

        // step 2 - convert to U and V
        // from here down is very similar to Y code except
        // instead of 16 different pixels, its 8 pixels of U and 8 of V
    movdqa     xmm1, xmm0
    movdqa     xmm3, xmm2
    pmaddubsw  xmm0, xmm7  // U
    pmaddubsw  xmm2, xmm7
    pmaddubsw  xmm1, xmm6  // V
    pmaddubsw  xmm3, xmm6
    phaddw     xmm0, xmm2
    phaddw     xmm1, xmm3
    psraw      xmm0, 8
    psraw      xmm1, 8
    packsswb   xmm0, xmm1
    paddb      xmm0, xmm5  // -> unsigned

        // step 3 - store 8 U and 8 V values
    movlps     qword ptr [edx], xmm0  // U
    movhps     qword ptr [edx + edi], xmm0  // V
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

// Read 16 UV from 444
#define READYUV444_AVX2 \
  __asm {                                                                      \
    __asm vmovdqu    xmm3, [esi] /* U */                                       \
    __asm vmovdqu    xmm1, [esi + edi] /* V */                                 \
    __asm lea        esi,  [esi + 16]                                          \
    __asm vpermq     ymm3, ymm3, 0xd8                                          \
    __asm vpermq     ymm1, ymm1, 0xd8                                          \
    __asm vpunpcklbw ymm3, ymm3, ymm1 /* UV */                                 \
    __asm vmovdqu    xmm4, [eax] /* Y */                                       \
    __asm vpermq     ymm4, ymm4, 0xd8                                          \
    __asm vpunpcklbw ymm4, ymm4, ymm4                                          \
    __asm lea        eax, [eax + 16]}

// Read 16 UV from 444.  With 16 Alpha.
#define READYUVA444_AVX2 \
  __asm {                                                                      \
    __asm vmovdqu    xmm3, [esi] /* U */                                       \
    __asm vmovdqu    xmm1, [esi + edi] /* V */                                 \
    __asm lea        esi,  [esi + 16]                                          \
    __asm vpermq     ymm3, ymm3, 0xd8                                          \
    __asm vpermq     ymm1, ymm1, 0xd8                                          \
    __asm vpunpcklbw ymm3, ymm3, ymm1 /* UV */                                 \
    __asm vmovdqu    xmm4, [eax] /* Y */                                       \
    __asm vpermq     ymm4, ymm4, 0xd8                                          \
    __asm vpunpcklbw ymm4, ymm4, ymm4                                          \
    __asm lea        eax, [eax + 16]                                           \
    __asm vmovdqu    xmm5, [ebp] /* A */                                       \
    __asm vpermq     ymm5, ymm5, 0xd8                                          \
    __asm lea        ebp, [ebp + 16]}

// Read 8 UV from 422, upsample to 16 UV.
#define READYUV422_AVX2 \
  __asm {                                                                      \
    __asm vmovq      xmm3, qword ptr [esi] /* U */                             \
    __asm vmovq      xmm1, qword ptr [esi + edi] /* V */                       \
    __asm lea        esi,  [esi + 8]                                           \
    __asm vpunpcklbw ymm3, ymm3, ymm1 /* UV */                                 \
    __asm vpermq     ymm3, ymm3, 0xd8                                          \
    __asm vpunpcklwd ymm3, ymm3, ymm3 /* UVUV (upsample) */                    \
    __asm vmovdqu    xmm4, [eax] /* Y */                                       \
    __asm vpermq     ymm4, ymm4, 0xd8                                          \
    __asm vpunpcklbw ymm4, ymm4, ymm4                                          \
    __asm lea        eax, [eax + 16]}

// Read 8 UV from 422, upsample to 16 UV.  With 16 Alpha.
#define READYUVA422_AVX2 \
  __asm {                                                                      \
    __asm vmovq      xmm3, qword ptr [esi] /* U */                             \
    __asm vmovq      xmm1, qword ptr [esi + edi] /* V */                       \
    __asm lea        esi,  [esi + 8]                                           \
    __asm vpunpcklbw ymm3, ymm3, ymm1 /* UV */                                 \
    __asm vpermq     ymm3, ymm3, 0xd8                                          \
    __asm vpunpcklwd ymm3, ymm3, ymm3 /* UVUV (upsample) */                    \
    __asm vmovdqu    xmm4, [eax] /* Y */                                       \
    __asm vpermq     ymm4, ymm4, 0xd8                                          \
    __asm vpunpcklbw ymm4, ymm4, ymm4                                          \
    __asm lea        eax, [eax + 16]                                           \
    __asm vmovdqu    xmm5, [ebp] /* A */                                       \
    __asm vpermq     ymm5, ymm5, 0xd8                                          \
    __asm lea        ebp, [ebp + 16]}

// Read 8 UV from NV12, upsample to 16 UV.
#define READNV12_AVX2 \
  __asm {                                                                      \
    __asm vmovdqu    xmm3, [esi] /* UV */                                      \
    __asm lea        esi,  [esi + 16]                                          \
    __asm vpermq     ymm3, ymm3, 0xd8                                          \
    __asm vpunpcklwd ymm3, ymm3, ymm3 /* UVUV (upsample) */                    \
    __asm vmovdqu    xmm4, [eax] /* Y */                                       \
    __asm vpermq     ymm4, ymm4, 0xd8                                          \
    __asm vpunpcklbw ymm4, ymm4, ymm4                                          \
    __asm lea        eax, [eax + 16]}

// Read 8 UV from NV21, upsample to 16 UV.
#define READNV21_AVX2 \
  __asm {                                                                      \
    __asm vmovdqu    xmm3, [esi] /* UV */                                      \
    __asm lea        esi,  [esi + 16]                                          \
    __asm vpermq     ymm3, ymm3, 0xd8                                          \
    __asm vpshufb    ymm3, ymm3, ymmword ptr kShuffleNV21                      \
    __asm vmovdqu    xmm4, [eax] /* Y */                                       \
    __asm vpermq     ymm4, ymm4, 0xd8                                          \
    __asm vpunpcklbw ymm4, ymm4, ymm4                                          \
    __asm lea        eax, [eax + 16]}

// Read 8 YUY2 with 16 Y and upsample 8 UV to 16 UV.
#define READYUY2_AVX2 \
  __asm {                                                                      \
    __asm vmovdqu    ymm4, [eax] /* YUY2 */                                    \
    __asm vpshufb    ymm4, ymm4, ymmword ptr kShuffleYUY2Y                     \
    __asm vmovdqu    ymm3, [eax] /* UV */                                      \
    __asm vpshufb    ymm3, ymm3, ymmword ptr kShuffleYUY2UV                    \
    __asm lea        eax, [eax + 32]}

// Read 8 UYVY with 16 Y and upsample 8 UV to 16 UV.
#define READUYVY_AVX2 \
  __asm {                                                                      \
    __asm vmovdqu    ymm4, [eax] /* UYVY */                                    \
    __asm vpshufb    ymm4, ymm4, ymmword ptr kShuffleUYVYY                     \
    __asm vmovdqu    ymm3, [eax] /* UV */                                      \
    __asm vpshufb    ymm3, ymm3, ymmword ptr kShuffleUYVYUV                    \
    __asm lea        eax, [eax + 32]}

// Convert 16 pixels: 16 UV and 16 Y.
#define YUVTORGB_AVX2(YuvConstants) \
  __asm {                                                                      \
    __asm vpsubb     ymm3, ymm3, ymmword ptr kBiasUV128                        \
    __asm vpmulhuw   ymm4, ymm4, ymmword ptr [YuvConstants + KYTORGB]          \
    __asm vmovdqa    ymm0, ymmword ptr [YuvConstants + KUVTOB]                 \
    __asm vmovdqa    ymm1, ymmword ptr [YuvConstants + KUVTOG]                 \
    __asm vmovdqa    ymm2, ymmword ptr [YuvConstants + KUVTOR]                 \
    __asm vpmaddubsw ymm0, ymm0, ymm3 /* B UV */                               \
    __asm vpmaddubsw ymm1, ymm1, ymm3 /* G UV */                               \
    __asm vpmaddubsw ymm2, ymm2, ymm3 /* B UV */                               \
    __asm vmovdqu    ymm3, ymmword ptr [YuvConstants + KYBIASTORGB]            \
    __asm vpaddw     ymm4, ymm3, ymm4                                          \
    __asm vpaddsw    ymm0, ymm0, ymm4                                          \
    __asm vpsubsw    ymm1, ymm4, ymm1                                          \
    __asm vpaddsw    ymm2, ymm2, ymm4                                          \
    __asm vpsraw     ymm0, ymm0, 6                                             \
    __asm vpsraw     ymm1, ymm1, 6                                             \
    __asm vpsraw     ymm2, ymm2, 6                                             \
    __asm vpackuswb  ymm0, ymm0, ymm0                                          \
    __asm vpackuswb  ymm1, ymm1, ymm1                                          \
    __asm vpackuswb  ymm2, ymm2, ymm2}

// Store 16 ARGB values.
#define STOREARGB_AVX2 \
  __asm {                                                                      \
    __asm vpunpcklbw ymm0, ymm0, ymm1 /* BG */                                 \
    __asm vpermq     ymm0, ymm0, 0xd8                                          \
    __asm vpunpcklbw ymm2, ymm2, ymm5 /* RA */                                 \
    __asm vpermq     ymm2, ymm2, 0xd8                                          \
    __asm vpunpcklwd ymm1, ymm0, ymm2 /* BGRA first 8 pixels */                \
    __asm vpunpckhwd ymm0, ymm0, ymm2 /* BGRA next 8 pixels */                 \
    __asm vmovdqu    0[edx], ymm1                                              \
    __asm vmovdqu    32[edx], ymm0                                             \
    __asm lea        edx,  [edx + 64]}

// Store 16 RGBA values.
#define STORERGBA_AVX2 \
  __asm {                                                                      \
    __asm vpunpcklbw ymm1, ymm1, ymm2 /* GR */                                 \
    __asm vpermq     ymm1, ymm1, 0xd8                                          \
    __asm vpunpcklbw ymm2, ymm5, ymm0 /* AB */                                 \
    __asm vpermq     ymm2, ymm2, 0xd8                                          \
    __asm vpunpcklwd ymm0, ymm2, ymm1 /* ABGR first 8 pixels */                \
    __asm vpunpckhwd ymm1, ymm2, ymm1 /* ABGR next 8 pixels */                 \
    __asm vmovdqu    [edx], ymm0                                               \
    __asm vmovdqu    [edx + 32], ymm1                                          \
    __asm lea        edx,  [edx + 64]}

#ifdef HAS_I422TOARGBROW_AVX2
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
__declspec(naked) void I422ToARGBRow_AVX2(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    mov        eax, [esp + 12 + 4]  // Y
    mov        esi, [esp + 12 + 8]  // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ebx, [esp + 12 + 20]  // yuvconstants
    mov        ecx, [esp + 12 + 24]  // width
    sub        edi, esi
    vpcmpeqb   ymm5, ymm5, ymm5  // generate 0xffffffffffffffff for alpha

 convertloop:
    READYUV422_AVX2
    YUVTORGB_AVX2(ebx)
    STOREARGB_AVX2

    sub        ecx, 16
    jg         convertloop

    pop        ebx
    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_I422TOARGBROW_AVX2

#ifdef HAS_I422ALPHATOARGBROW_AVX2
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y and 16 A producing 16 ARGB.
__declspec(naked) void I422AlphaToARGBRow_AVX2(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    const uint8_t* a_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    push       ebp
    mov        eax, [esp + 16 + 4]  // Y
    mov        esi, [esp + 16 + 8]  // U
    mov        edi, [esp + 16 + 12]  // V
    mov        ebp, [esp + 16 + 16]  // A
    mov        edx, [esp + 16 + 20]  // argb
    mov        ebx, [esp + 16 + 24]  // yuvconstants
    mov        ecx, [esp + 16 + 28]  // width
    sub        edi, esi

 convertloop:
    READYUVA422_AVX2
    YUVTORGB_AVX2(ebx)
    STOREARGB_AVX2

    sub        ecx, 16
    jg         convertloop

    pop        ebp
    pop        ebx
    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_I422ALPHATOARGBROW_AVX2

#ifdef HAS_I444TOARGBROW_AVX2
// 16 pixels
// 16 UV values with 16 Y producing 16 ARGB (64 bytes).
__declspec(naked) void I444ToARGBRow_AVX2(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    mov        eax, [esp + 12 + 4]  // Y
    mov        esi, [esp + 12 + 8]  // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ebx, [esp + 12 + 20]  // yuvconstants
    mov        ecx, [esp + 12 + 24]  // width
    sub        edi, esi
    vpcmpeqb   ymm5, ymm5, ymm5  // generate 0xffffffffffffffff for alpha
 convertloop:
    READYUV444_AVX2
    YUVTORGB_AVX2(ebx)
    STOREARGB_AVX2

    sub        ecx, 16
    jg         convertloop

    pop        ebx
    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_I444TOARGBROW_AVX2

#ifdef HAS_I444ALPHATOARGBROW_AVX2
// 16 pixels
// 16 UV values with 16 Y producing 16 ARGB (64 bytes).
__declspec(naked) void I444AlphaToARGBRow_AVX2(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    const uint8_t* a_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
  push       esi
  push       edi
  push       ebx
  push       ebp
  mov        eax, [esp + 16 + 4]  // Y
  mov        esi, [esp + 16 + 8]  // U
  mov        edi, [esp + 16 + 12]  // V
  mov        ebp, [esp + 16 + 16]  // A
  mov        edx, [esp + 16 + 20]  // argb
  mov        ebx, [esp + 16 + 24]  // yuvconstants
  mov        ecx, [esp + 16 + 28]  // width
  sub        edi, esi
  convertloop:
  READYUVA444_AVX2
  YUVTORGB_AVX2(ebx)
  STOREARGB_AVX2

  sub        ecx, 16
  jg         convertloop

  pop        ebp
  pop        ebx
  pop        edi
  pop        esi
  vzeroupper
  ret
  }
}
#endif  // HAS_I444AlphaTOARGBROW_AVX2

#ifdef HAS_NV12TOARGBROW_AVX2
// 16 pixels.
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
__declspec(naked) void NV12ToARGBRow_AVX2(
    const uint8_t* y_buf,
    const uint8_t* uv_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       ebx
    mov        eax, [esp + 8 + 4]  // Y
    mov        esi, [esp + 8 + 8]  // UV
    mov        edx, [esp + 8 + 12]  // argb
    mov        ebx, [esp + 8 + 16]  // yuvconstants
    mov        ecx, [esp + 8 + 20]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate 0xffffffffffffffff for alpha

 convertloop:
    READNV12_AVX2
    YUVTORGB_AVX2(ebx)
    STOREARGB_AVX2

    sub        ecx, 16
    jg         convertloop

    pop        ebx
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_NV12TOARGBROW_AVX2

#ifdef HAS_NV21TOARGBROW_AVX2
// 16 pixels.
// 8 VU values upsampled to 16 UV, mixed with 16 Y producing 16 ARGB (64 bytes).
__declspec(naked) void NV21ToARGBRow_AVX2(
    const uint8_t* y_buf,
    const uint8_t* vu_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       ebx
    mov        eax, [esp + 8 + 4]  // Y
    mov        esi, [esp + 8 + 8]  // VU
    mov        edx, [esp + 8 + 12]  // argb
    mov        ebx, [esp + 8 + 16]  // yuvconstants
    mov        ecx, [esp + 8 + 20]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate 0xffffffffffffffff for alpha

 convertloop:
    READNV21_AVX2
    YUVTORGB_AVX2(ebx)
    STOREARGB_AVX2

    sub        ecx, 16
    jg         convertloop

    pop        ebx
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_NV21TOARGBROW_AVX2

#ifdef HAS_YUY2TOARGBROW_AVX2
// 16 pixels.
// 8 YUY2 values with 16 Y and 8 UV producing 16 ARGB (64 bytes).
__declspec(naked) void YUY2ToARGBRow_AVX2(
    const uint8_t* src_yuy2,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       ebx
    mov        eax, [esp + 4 + 4]  // yuy2
    mov        edx, [esp + 4 + 8]  // argb
    mov        ebx, [esp + 4 + 12]  // yuvconstants
    mov        ecx, [esp + 4 + 16]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate 0xffffffffffffffff for alpha

 convertloop:
    READYUY2_AVX2
    YUVTORGB_AVX2(ebx)
    STOREARGB_AVX2

    sub        ecx, 16
    jg         convertloop

    pop        ebx
    vzeroupper
    ret
  }
}
#endif  // HAS_YUY2TOARGBROW_AVX2

#ifdef HAS_UYVYTOARGBROW_AVX2
// 16 pixels.
// 8 UYVY values with 16 Y and 8 UV producing 16 ARGB (64 bytes).
__declspec(naked) void UYVYToARGBRow_AVX2(
    const uint8_t* src_uyvy,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       ebx
    mov        eax, [esp + 4 + 4]  // uyvy
    mov        edx, [esp + 4 + 8]  // argb
    mov        ebx, [esp + 4 + 12]  // yuvconstants
    mov        ecx, [esp + 4 + 16]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate 0xffffffffffffffff for alpha

 convertloop:
    READUYVY_AVX2
    YUVTORGB_AVX2(ebx)
    STOREARGB_AVX2

    sub        ecx, 16
    jg         convertloop

    pop        ebx
    vzeroupper
    ret
  }
}
#endif  // HAS_UYVYTOARGBROW_AVX2

#ifdef HAS_I422TORGBAROW_AVX2
// 16 pixels
// 8 UV values upsampled to 16 UV, mixed with 16 Y producing 16 RGBA (64 bytes).
__declspec(naked) void I422ToRGBARow_AVX2(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    mov        eax, [esp + 12 + 4]  // Y
    mov        esi, [esp + 12 + 8]  // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // abgr
    mov        ebx, [esp + 12 + 20]  // yuvconstants
    mov        ecx, [esp + 12 + 24]  // width
    sub        edi, esi
    vpcmpeqb   ymm5, ymm5, ymm5  // generate 0xffffffffffffffff for alpha

 convertloop:
    READYUV422_AVX2
    YUVTORGB_AVX2(ebx)
    STORERGBA_AVX2

    sub        ecx, 16
    jg         convertloop

    pop        ebx
    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_I422TORGBAROW_AVX2

#if defined(HAS_I422TOARGBROW_SSSE3)
// TODO(fbarchard): Read that does half size on Y and treats 420 as 444.
// Allows a conversion with half size scaling.

// Read 8 UV from 444.
#define READYUV444 \
  __asm {                                                                      \
    __asm movq       xmm3, qword ptr [esi] /* U */                             \
    __asm movq       xmm1, qword ptr [esi + edi] /* V */                       \
    __asm lea        esi,  [esi + 8]                                           \
    __asm punpcklbw  xmm3, xmm1 /* UV */                                       \
    __asm movq       xmm4, qword ptr [eax]                                     \
    __asm punpcklbw  xmm4, xmm4                                                \
    __asm lea        eax, [eax + 8]}

// Read 4 UV from 444.  With 8 Alpha.
#define READYUVA444 \
  __asm {                                                                      \
    __asm movq       xmm3, qword ptr [esi] /* U */                             \
    __asm movq       xmm1, qword ptr [esi + edi] /* V */                       \
    __asm lea        esi,  [esi + 8]                                           \
    __asm punpcklbw  xmm3, xmm1 /* UV */                                       \
    __asm movq       xmm4, qword ptr [eax]                                     \
    __asm punpcklbw  xmm4, xmm4                                                \
    __asm lea        eax, [eax + 8]                                            \
    __asm movq       xmm5, qword ptr [ebp] /* A */                             \
    __asm lea        ebp, [ebp + 8]}

// Read 4 UV from 422, upsample to 8 UV.
#define READYUV422 \
  __asm {                                                                      \
    __asm movd       xmm3, [esi] /* U */                                       \
    __asm movd       xmm1, [esi + edi] /* V */                                 \
    __asm lea        esi,  [esi + 4]                                           \
    __asm punpcklbw  xmm3, xmm1 /* UV */                                       \
    __asm punpcklwd  xmm3, xmm3 /* UVUV (upsample) */                          \
    __asm movq       xmm4, qword ptr [eax]                                     \
    __asm punpcklbw  xmm4, xmm4                                                \
    __asm lea        eax, [eax + 8]}

// Read 4 UV from 422, upsample to 8 UV.  With 8 Alpha.
#define READYUVA422 \
  __asm {                                                                      \
    __asm movd       xmm3, [esi] /* U */                                       \
    __asm movd       xmm1, [esi + edi] /* V */                                 \
    __asm lea        esi,  [esi + 4]                                           \
    __asm punpcklbw  xmm3, xmm1 /* UV */                                       \
    __asm punpcklwd  xmm3, xmm3 /* UVUV (upsample) */                          \
    __asm movq       xmm4, qword ptr [eax] /* Y */                             \
    __asm punpcklbw  xmm4, xmm4                                                \
    __asm lea        eax, [eax + 8]                                            \
    __asm movq       xmm5, qword ptr [ebp] /* A */                             \
    __asm lea        ebp, [ebp + 8]}

// Read 4 UV from NV12, upsample to 8 UV.
#define READNV12 \
  __asm {                                                                      \
    __asm movq       xmm3, qword ptr [esi] /* UV */                            \
    __asm lea        esi,  [esi + 8]                                           \
    __asm punpcklwd  xmm3, xmm3 /* UVUV (upsample) */                          \
    __asm movq       xmm4, qword ptr [eax]                                     \
    __asm punpcklbw  xmm4, xmm4                                                \
    __asm lea        eax, [eax + 8]}

// Read 4 VU from NV21, upsample to 8 UV.
#define READNV21 \
  __asm {                                                                      \
    __asm movq       xmm3, qword ptr [esi] /* UV */                            \
    __asm lea        esi,  [esi + 8]                                           \
    __asm pshufb     xmm3, xmmword ptr kShuffleNV21                            \
    __asm movq       xmm4, qword ptr [eax]                                     \
    __asm punpcklbw  xmm4, xmm4                                                \
    __asm lea        eax, [eax + 8]}

// Read 4 YUY2 with 8 Y and upsample 4 UV to 8 UV.
#define READYUY2 \
  __asm {                                                                      \
    __asm movdqu     xmm4, [eax] /* YUY2 */                                    \
    __asm pshufb     xmm4, xmmword ptr kShuffleYUY2Y                           \
    __asm movdqu     xmm3, [eax] /* UV */                                      \
    __asm pshufb     xmm3, xmmword ptr kShuffleYUY2UV                          \
    __asm lea        eax, [eax + 16]}

// Read 4 UYVY with 8 Y and upsample 4 UV to 8 UV.
#define READUYVY \
  __asm {                                                                      \
    __asm movdqu     xmm4, [eax] /* UYVY */                                    \
    __asm pshufb     xmm4, xmmword ptr kShuffleUYVYY                           \
    __asm movdqu     xmm3, [eax] /* UV */                                      \
    __asm pshufb     xmm3, xmmword ptr kShuffleUYVYUV                          \
    __asm lea        eax, [eax + 16]}

// Convert 8 pixels: 8 UV and 8 Y.
#define YUVTORGB(YuvConstants) \
  __asm {                                                                      \
    __asm psubb      xmm3, xmmword ptr kBiasUV128                              \
    __asm pmulhuw    xmm4, xmmword ptr [YuvConstants + KYTORGB]                \
    __asm movdqa     xmm0, xmmword ptr [YuvConstants + KUVTOB]                 \
    __asm movdqa     xmm1, xmmword ptr [YuvConstants + KUVTOG]                 \
    __asm movdqa     xmm2, xmmword ptr [YuvConstants + KUVTOR]                 \
    __asm pmaddubsw  xmm0, xmm3                                                \
    __asm pmaddubsw  xmm1, xmm3                                                \
    __asm pmaddubsw  xmm2, xmm3                                                \
    __asm movdqa     xmm3, xmmword ptr [YuvConstants + KYBIASTORGB]            \
    __asm paddw      xmm4, xmm3                                                \
    __asm paddsw     xmm0, xmm4                                                \
    __asm paddsw     xmm2, xmm4                                                \
    __asm psubsw     xmm4, xmm1                                                \
    __asm movdqa     xmm1, xmm4                                                \
    __asm psraw      xmm0, 6                                                   \
    __asm psraw      xmm1, 6                                                   \
    __asm psraw      xmm2, 6                                                   \
    __asm packuswb   xmm0, xmm0 /* B */                                        \
    __asm packuswb   xmm1, xmm1 /* G */                                        \
    __asm packuswb   xmm2, xmm2 /* R */             \
  }

// Store 8 ARGB values.
#define STOREARGB \
  __asm {                                                                      \
    __asm punpcklbw  xmm0, xmm1 /* BG */                                       \
    __asm punpcklbw  xmm2, xmm5 /* RA */                                       \
    __asm movdqa     xmm1, xmm0                                                \
    __asm punpcklwd  xmm0, xmm2 /* BGRA first 4 pixels */                      \
    __asm punpckhwd  xmm1, xmm2 /* BGRA next 4 pixels */                       \
    __asm movdqu     0[edx], xmm0                                              \
    __asm movdqu     16[edx], xmm1                                             \
    __asm lea        edx,  [edx + 32]}

// Store 8 BGRA values.
#define STOREBGRA \
  __asm {                                                                      \
    __asm pcmpeqb    xmm5, xmm5 /* generate 0xffffffff for alpha */            \
    __asm punpcklbw  xmm1, xmm0 /* GB */                                       \
    __asm punpcklbw  xmm5, xmm2 /* AR */                                       \
    __asm movdqa     xmm0, xmm5                                                \
    __asm punpcklwd  xmm5, xmm1 /* BGRA first 4 pixels */                      \
    __asm punpckhwd  xmm0, xmm1 /* BGRA next 4 pixels */                       \
    __asm movdqu     0[edx], xmm5                                              \
    __asm movdqu     16[edx], xmm0                                             \
    __asm lea        edx,  [edx + 32]}

// Store 8 RGBA values.
#define STORERGBA \
  __asm {                                                                      \
    __asm pcmpeqb    xmm5, xmm5 /* generate 0xffffffff for alpha */            \
    __asm punpcklbw  xmm1, xmm2 /* GR */                                       \
    __asm punpcklbw  xmm5, xmm0 /* AB */                                       \
    __asm movdqa     xmm0, xmm5                                                \
    __asm punpcklwd  xmm5, xmm1 /* RGBA first 4 pixels */                      \
    __asm punpckhwd  xmm0, xmm1 /* RGBA next 4 pixels */                       \
    __asm movdqu     0[edx], xmm5                                              \
    __asm movdqu     16[edx], xmm0                                             \
    __asm lea        edx,  [edx + 32]}

// Store 8 RGB24 values.
#define STORERGB24 \
  __asm {/* Weave into RRGB */                                                 \
    __asm punpcklbw  xmm0, xmm1 /* BG */                                       \
    __asm punpcklbw  xmm2, xmm2 /* RR */                                       \
    __asm movdqa     xmm1, xmm0                                                \
    __asm punpcklwd  xmm0, xmm2 /* BGRR first 4 pixels */                      \
    __asm punpckhwd  xmm1, xmm2 /* BGRR next 4 pixels */ /* RRGB -> RGB24 */   \
    __asm pshufb     xmm0, xmm5 /* Pack first 8 and last 4 bytes. */           \
    __asm pshufb     xmm1, xmm6 /* Pack first 12 bytes. */                     \
    __asm palignr    xmm1, xmm0, 12 /* last 4 bytes of xmm0 + 12 xmm1 */       \
    __asm movq       qword ptr 0[edx], xmm0 /* First 8 bytes */                \
    __asm movdqu     8[edx], xmm1 /* Last 16 bytes */                          \
    __asm lea        edx,  [edx + 24]}

// Store 8 RGB565 values.
#define STORERGB565 \
  __asm {/* Weave into RRGB */                                                 \
    __asm punpcklbw  xmm0, xmm1 /* BG */                                       \
    __asm punpcklbw  xmm2, xmm2 /* RR */                                       \
    __asm movdqa     xmm1, xmm0                                                \
    __asm punpcklwd  xmm0, xmm2 /* BGRR first 4 pixels */                      \
    __asm punpckhwd  xmm1, xmm2 /* BGRR next 4 pixels */ /* RRGB -> RGB565 */  \
    __asm movdqa     xmm3, xmm0 /* B  first 4 pixels of argb */                \
    __asm movdqa     xmm2, xmm0 /* G */                                        \
    __asm pslld      xmm0, 8 /* R */                                           \
    __asm psrld      xmm3, 3 /* B */                                           \
    __asm psrld      xmm2, 5 /* G */                                           \
    __asm psrad      xmm0, 16 /* R */                                          \
    __asm pand       xmm3, xmm5 /* B */                                        \
    __asm pand       xmm2, xmm6 /* G */                                        \
    __asm pand       xmm0, xmm7 /* R */                                        \
    __asm por        xmm3, xmm2 /* BG */                                       \
    __asm por        xmm0, xmm3 /* BGR */                                      \
    __asm movdqa     xmm3, xmm1 /* B  next 4 pixels of argb */                 \
    __asm movdqa     xmm2, xmm1 /* G */                                        \
    __asm pslld      xmm1, 8 /* R */                                           \
    __asm psrld      xmm3, 3 /* B */                                           \
    __asm psrld      xmm2, 5 /* G */                                           \
    __asm psrad      xmm1, 16 /* R */                                          \
    __asm pand       xmm3, xmm5 /* B */                                        \
    __asm pand       xmm2, xmm6 /* G */                                        \
    __asm pand       xmm1, xmm7 /* R */                                        \
    __asm por        xmm3, xmm2 /* BG */                                       \
    __asm por        xmm1, xmm3 /* BGR */                                      \
    __asm packssdw   xmm0, xmm1                                                \
    __asm movdqu     0[edx], xmm0 /* store 8 pixels of RGB565 */               \
    __asm lea        edx, [edx + 16]}

// 8 pixels.
// 8 UV values, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) void I444ToARGBRow_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    mov        eax, [esp + 12 + 4]  // Y
    mov        esi, [esp + 12 + 8]  // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ebx, [esp + 12 + 20]  // yuvconstants
    mov        ecx, [esp + 12 + 24]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5  // generate 0xffffffff for alpha

 convertloop:
    READYUV444
    YUVTORGB(ebx)
    STOREARGB

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels.
// 8 UV values, mixed with 8 Y and 8A producing 8 ARGB (32 bytes).
__declspec(naked) void I444AlphaToARGBRow_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    const uint8_t* a_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    push       ebp
    mov        eax, [esp + 16 + 4]  // Y
    mov        esi, [esp + 16 + 8]  // U
    mov        edi, [esp + 16 + 12]  // V
    mov        ebp, [esp + 16 + 16]  // A
    mov        edx, [esp + 16 + 20]  // argb
    mov        ebx, [esp + 16 + 24]  // yuvconstants
    mov        ecx, [esp + 16 + 28]  // width
    sub        edi, esi

 convertloop:
    READYUVA444
    YUVTORGB(ebx)
    STOREARGB

    sub        ecx, 8
    jg         convertloop

    pop        ebp
    pop        ebx
    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 RGB24 (24 bytes).
__declspec(naked) void I422ToRGB24Row_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    uint8_t* dst_rgb24,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    mov        eax, [esp + 12 + 4]  // Y
    mov        esi, [esp + 12 + 8]  // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ebx, [esp + 12 + 20]  // yuvconstants
    mov        ecx, [esp + 12 + 24]  // width
    sub        edi, esi
    movdqa     xmm5, xmmword ptr kShuffleMaskARGBToRGB24_0
    movdqa     xmm6, xmmword ptr kShuffleMaskARGBToRGB24

 convertloop:
    READYUV422
    YUVTORGB(ebx)
    STORERGB24

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels.
// 8 UV values, mixed with 8 Y producing 8 RGB24 (24 bytes).
__declspec(naked) void I444ToRGB24Row_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    uint8_t* dst_rgb24,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    mov        eax, [esp + 12 + 4]  // Y
    mov        esi, [esp + 12 + 8]  // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ebx, [esp + 12 + 20]  // yuvconstants
    mov        ecx, [esp + 12 + 24]  // width
    sub        edi, esi
    movdqa     xmm5, xmmword ptr kShuffleMaskARGBToRGB24_0
    movdqa     xmm6, xmmword ptr kShuffleMaskARGBToRGB24

 convertloop:
    READYUV444
    YUVTORGB(ebx)
    STORERGB24

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 RGB565 (16 bytes).
__declspec(naked) void I422ToRGB565Row_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    uint8_t* rgb565_buf,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    mov        eax, [esp + 12 + 4]  // Y
    mov        esi, [esp + 12 + 8]  // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ebx, [esp + 12 + 20]  // yuvconstants
    mov        ecx, [esp + 12 + 24]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5  // generate mask 0x0000001f
    psrld      xmm5, 27
    pcmpeqb    xmm6, xmm6  // generate mask 0x000007e0
    psrld      xmm6, 26
    pslld      xmm6, 5
    pcmpeqb    xmm7, xmm7  // generate mask 0xfffff800
    pslld      xmm7, 11

 convertloop:
    READYUV422
    YUVTORGB(ebx)
    STORERGB565

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) void I422ToARGBRow_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    mov        eax, [esp + 12 + 4]  // Y
    mov        esi, [esp + 12 + 8]  // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ebx, [esp + 12 + 20]  // yuvconstants
    mov        ecx, [esp + 12 + 24]  // width
    sub        edi, esi
    pcmpeqb    xmm5, xmm5  // generate 0xffffffff for alpha

 convertloop:
    READYUV422
    YUVTORGB(ebx)
    STOREARGB

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels.
// 4 UV values upsampled to 8 UV, mixed with 8 Y and 8 A producing 8 ARGB.
__declspec(naked) void I422AlphaToARGBRow_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    const uint8_t* a_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    push       ebp
    mov        eax, [esp + 16 + 4]  // Y
    mov        esi, [esp + 16 + 8]  // U
    mov        edi, [esp + 16 + 12]  // V
    mov        ebp, [esp + 16 + 16]  // A
    mov        edx, [esp + 16 + 20]  // argb
    mov        ebx, [esp + 16 + 24]  // yuvconstants
    mov        ecx, [esp + 16 + 28]  // width
    sub        edi, esi

 convertloop:
    READYUVA422
    YUVTORGB(ebx)
    STOREARGB

    sub        ecx, 8
    jg         convertloop

    pop        ebp
    pop        ebx
    pop        edi
    pop        esi
    ret
  }
}

// 8 pixels.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) void NV12ToARGBRow_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* uv_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       ebx
    mov        eax, [esp + 8 + 4]  // Y
    mov        esi, [esp + 8 + 8]  // UV
    mov        edx, [esp + 8 + 12]  // argb
    mov        ebx, [esp + 8 + 16]  // yuvconstants
    mov        ecx, [esp + 8 + 20]  // width
    pcmpeqb    xmm5, xmm5  // generate 0xffffffff for alpha

 convertloop:
    READNV12
    YUVTORGB(ebx)
    STOREARGB

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    pop        esi
    ret
  }
}

// 8 pixels.
// 4 UV values upsampled to 8 UV, mixed with 8 Y producing 8 ARGB (32 bytes).
__declspec(naked) void NV21ToARGBRow_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* vu_buf,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       ebx
    mov        eax, [esp + 8 + 4]  // Y
    mov        esi, [esp + 8 + 8]  // VU
    mov        edx, [esp + 8 + 12]  // argb
    mov        ebx, [esp + 8 + 16]  // yuvconstants
    mov        ecx, [esp + 8 + 20]  // width
    pcmpeqb    xmm5, xmm5  // generate 0xffffffff for alpha

 convertloop:
    READNV21
    YUVTORGB(ebx)
    STOREARGB

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    pop        esi
    ret
  }
}

// 8 pixels.
// 4 YUY2 values with 8 Y and 4 UV producing 8 ARGB (32 bytes).
__declspec(naked) void YUY2ToARGBRow_SSSE3(
    const uint8_t* src_yuy2,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       ebx
    mov        eax, [esp + 4 + 4]  // yuy2
    mov        edx, [esp + 4 + 8]  // argb
    mov        ebx, [esp + 4 + 12]  // yuvconstants
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm5, xmm5  // generate 0xffffffff for alpha

 convertloop:
    READYUY2
    YUVTORGB(ebx)
    STOREARGB

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    ret
  }
}

// 8 pixels.
// 4 UYVY values with 8 Y and 4 UV producing 8 ARGB (32 bytes).
__declspec(naked) void UYVYToARGBRow_SSSE3(
    const uint8_t* src_uyvy,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       ebx
    mov        eax, [esp + 4 + 4]  // uyvy
    mov        edx, [esp + 4 + 8]  // argb
    mov        ebx, [esp + 4 + 12]  // yuvconstants
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm5, xmm5  // generate 0xffffffff for alpha

 convertloop:
    READUYVY
    YUVTORGB(ebx)
    STOREARGB

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    ret
  }
}

__declspec(naked) void I422ToRGBARow_SSSE3(
    const uint8_t* y_buf,
    const uint8_t* u_buf,
    const uint8_t* v_buf,
    uint8_t* dst_rgba,
    const struct YuvConstants* yuvconstants,
    int width) {
  __asm {
    push       esi
    push       edi
    push       ebx
    mov        eax, [esp + 12 + 4]  // Y
    mov        esi, [esp + 12 + 8]  // U
    mov        edi, [esp + 12 + 12]  // V
    mov        edx, [esp + 12 + 16]  // argb
    mov        ebx, [esp + 12 + 20]  // yuvconstants
    mov        ecx, [esp + 12 + 24]  // width
    sub        edi, esi

 convertloop:
    READYUV422
    YUVTORGB(ebx)
    STORERGBA

    sub        ecx, 8
    jg         convertloop

    pop        ebx
    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_I422TOARGBROW_SSSE3

// I400ToARGBRow_SSE2 is disabled due to new yuvconstant parameter
#ifdef HAS_I400TOARGBROW_SSE2
// 8 pixels of Y converted to 8 pixels of ARGB (32 bytes).
__declspec(naked) void I400ToARGBRow_SSE2(const uint8_t* y_buf,
                                          uint8_t* rgb_buf,
                                          const struct YuvConstants*,
                                          int width) {
  __asm {
    mov        eax, 0x4a354a35  // 4a35 = 18997 = round(1.164 * 64 * 256)
    movd       xmm2, eax
    pshufd     xmm2, xmm2,0
    mov        eax, 0x04880488  // 0488 = 1160 = round(1.164 * 64 * 16)
    movd       xmm3, eax
    pshufd     xmm3, xmm3, 0
    pcmpeqb    xmm4, xmm4  // generate mask 0xff000000
    pslld      xmm4, 24

    mov        eax, [esp + 4]  // Y
    mov        edx, [esp + 8]  // rgb
    mov        ecx, [esp + 12]  // width

 convertloop:
        // Step 1: Scale Y contribution to 8 G values. G = (y - 16) * 1.164
    movq       xmm0, qword ptr [eax]
    lea        eax, [eax + 8]
    punpcklbw  xmm0, xmm0  // Y.Y
    pmulhuw    xmm0, xmm2
    psubusw    xmm0, xmm3
    psrlw      xmm0, 6
    packuswb   xmm0, xmm0        // G

        // Step 2: Weave into ARGB
    punpcklbw  xmm0, xmm0  // GG
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm0  // BGRA first 4 pixels
    punpckhwd  xmm1, xmm1  // BGRA next 4 pixels
    por        xmm0, xmm4
    por        xmm1, xmm4
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx,  [edx + 32]
    sub        ecx, 8
    jg         convertloop
    ret
  }
}
#endif  // HAS_I400TOARGBROW_SSE2

#ifdef HAS_I400TOARGBROW_AVX2
// 16 pixels of Y converted to 16 pixels of ARGB (64 bytes).
// note: vpunpcklbw mutates and vpackuswb unmutates.
__declspec(naked) void I400ToARGBRow_AVX2(const uint8_t* y_buf,
                                          uint8_t* rgb_buf,
                                          const struct YuvConstants*,
                                          int width) {
  __asm {
    mov        eax, 0x4a354a35  // 4a35 = 18997 = round(1.164 * 64 * 256)
    vmovd      xmm2, eax
    vbroadcastss ymm2, xmm2
    mov        eax, 0x04880488  // 0488 = 1160 = round(1.164 * 64 * 16)
    vmovd      xmm3, eax
    vbroadcastss ymm3, xmm3
    vpcmpeqb   ymm4, ymm4, ymm4  // generate mask 0xff000000
    vpslld     ymm4, ymm4, 24

    mov        eax, [esp + 4]  // Y
    mov        edx, [esp + 8]  // rgb
    mov        ecx, [esp + 12]  // width

 convertloop:
        // Step 1: Scale Y contriportbution to 16 G values. G = (y - 16) * 1.164
    vmovdqu    xmm0, [eax]
    lea        eax, [eax + 16]
    vpermq     ymm0, ymm0, 0xd8  // vpunpcklbw mutates
    vpunpcklbw ymm0, ymm0, ymm0  // Y.Y
    vpmulhuw   ymm0, ymm0, ymm2
    vpsubusw   ymm0, ymm0, ymm3
    vpsrlw     ymm0, ymm0, 6
    vpackuswb  ymm0, ymm0, ymm0        // G.  still mutated: 3120

        // TODO(fbarchard): Weave alpha with unpack.
        // Step 2: Weave into ARGB
    vpunpcklbw ymm1, ymm0, ymm0  // GG - mutates
    vpermq     ymm1, ymm1, 0xd8
    vpunpcklwd ymm0, ymm1, ymm1  // GGGG first 8 pixels
    vpunpckhwd ymm1, ymm1, ymm1  // GGGG next 8 pixels
    vpor       ymm0, ymm0, ymm4
    vpor       ymm1, ymm1, ymm4
    vmovdqu    [edx], ymm0
    vmovdqu    [edx + 32], ymm1
    lea        edx,  [edx + 64]
    sub        ecx, 16
    jg         convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_I400TOARGBROW_AVX2

#ifdef HAS_MIRRORROW_SSSE3
// Shuffle table for reversing the bytes.
static const uvec8 kShuffleMirror = {15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u,
                                     7u,  6u,  5u,  4u,  3u,  2u,  1u, 0u};

// TODO(fbarchard): Replace lea with -16 offset.
__declspec(naked) void MirrorRow_SSSE3(const uint8_t* src,
                                       uint8_t* dst,
                                       int width) {
  __asm {
    mov       eax, [esp + 4]  // src
    mov       edx, [esp + 8]  // dst
    mov       ecx, [esp + 12]  // width
    movdqa    xmm5, xmmword ptr kShuffleMirror

 convertloop:
    movdqu    xmm0, [eax - 16 + ecx]
    pshufb    xmm0, xmm5
    movdqu    [edx], xmm0
    lea       edx, [edx + 16]
    sub       ecx, 16
    jg        convertloop
    ret
  }
}
#endif  // HAS_MIRRORROW_SSSE3

#ifdef HAS_MIRRORROW_AVX2
__declspec(naked) void MirrorRow_AVX2(const uint8_t* src,
                                      uint8_t* dst,
                                      int width) {
  __asm {
    mov       eax, [esp + 4]  // src
    mov       edx, [esp + 8]  // dst
    mov       ecx, [esp + 12]  // width
    vbroadcastf128 ymm5, xmmword ptr kShuffleMirror

 convertloop:
    vmovdqu   ymm0, [eax - 32 + ecx]
    vpshufb   ymm0, ymm0, ymm5
    vpermq    ymm0, ymm0, 0x4e  // swap high and low halfs
    vmovdqu   [edx], ymm0
    lea       edx, [edx + 32]
    sub       ecx, 32
    jg        convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_MIRRORROW_AVX2

#ifdef HAS_MIRRORSPLITUVROW_SSSE3
// Shuffle table for reversing the bytes of UV channels.
static const uvec8 kShuffleMirrorUV = {14u, 12u, 10u, 8u, 6u, 4u, 2u, 0u,
                                       15u, 13u, 11u, 9u, 7u, 5u, 3u, 1u};

__declspec(naked) void MirrorSplitUVRow_SSSE3(const uint8_t* src,
                                              uint8_t* dst_u,
                                              uint8_t* dst_v,
                                              int width) {
  __asm {
    push      edi
    mov       eax, [esp + 4 + 4]  // src
    mov       edx, [esp + 4 + 8]  // dst_u
    mov       edi, [esp + 4 + 12]  // dst_v
    mov       ecx, [esp + 4 + 16]  // width
    movdqa    xmm1, xmmword ptr kShuffleMirrorUV
    lea       eax, [eax + ecx * 2 - 16]
    sub       edi, edx

 convertloop:
    movdqu    xmm0, [eax]
    lea       eax, [eax - 16]
    pshufb    xmm0, xmm1
    movlpd    qword ptr [edx], xmm0
    movhpd    qword ptr [edx + edi], xmm0
    lea       edx, [edx + 8]
    sub       ecx, 8
    jg        convertloop

    pop       edi
    ret
  }
}
#endif  // HAS_MIRRORSPLITUVROW_SSSE3

#ifdef HAS_ARGBMIRRORROW_SSE2
__declspec(naked) void ARGBMirrorRow_SSE2(const uint8_t* src,
                                          uint8_t* dst,
                                          int width) {
  __asm {
    mov       eax, [esp + 4]  // src
    mov       edx, [esp + 8]  // dst
    mov       ecx, [esp + 12]  // width
    lea       eax, [eax - 16 + ecx * 4]  // last 4 pixels.

 convertloop:
    movdqu    xmm0, [eax]
    lea       eax, [eax - 16]
    pshufd    xmm0, xmm0, 0x1b
    movdqu    [edx], xmm0
    lea       edx, [edx + 16]
    sub       ecx, 4
    jg        convertloop
    ret
  }
}
#endif  // HAS_ARGBMIRRORROW_SSE2

#ifdef HAS_ARGBMIRRORROW_AVX2
// Shuffle table for reversing the bytes.
static const ulvec32 kARGBShuffleMirror_AVX2 = {7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u};

__declspec(naked) void ARGBMirrorRow_AVX2(const uint8_t* src,
                                          uint8_t* dst,
                                          int width) {
  __asm {
    mov       eax, [esp + 4]  // src
    mov       edx, [esp + 8]  // dst
    mov       ecx, [esp + 12]  // width
    vmovdqu   ymm5, ymmword ptr kARGBShuffleMirror_AVX2

 convertloop:
    vpermd    ymm0, ymm5, [eax - 32 + ecx * 4]  // permute dword order
    vmovdqu   [edx], ymm0
    lea       edx, [edx + 32]
    sub       ecx, 8
    jg        convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBMIRRORROW_AVX2

#ifdef HAS_SPLITUVROW_SSE2
__declspec(naked) void SplitUVRow_SSE2(const uint8_t* src_uv,
                                       uint8_t* dst_u,
                                       uint8_t* dst_v,
                                       int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]  // src_uv
    mov        edx, [esp + 4 + 8]  // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm5, xmm5  // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    movdqa     xmm2, xmm0
    movdqa     xmm3, xmm1
    pand       xmm0, xmm5  // even bytes
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    psrlw      xmm2, 8  // odd bytes
    psrlw      xmm3, 8
    packuswb   xmm2, xmm3
    movdqu     [edx], xmm0
    movdqu     [edx + edi], xmm2
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}

#endif  // HAS_SPLITUVROW_SSE2

#ifdef HAS_SPLITUVROW_AVX2
__declspec(naked) void SplitUVRow_AVX2(const uint8_t* src_uv,
                                       uint8_t* dst_u,
                                       uint8_t* dst_v,
                                       int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]  // src_uv
    mov        edx, [esp + 4 + 8]  // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpsrlw     ymm2, ymm0, 8  // odd bytes
    vpsrlw     ymm3, ymm1, 8
    vpand      ymm0, ymm0, ymm5  // even bytes
    vpand      ymm1, ymm1, ymm5
    vpackuswb  ymm0, ymm0, ymm1
    vpackuswb  ymm2, ymm2, ymm3
    vpermq     ymm0, ymm0, 0xd8
    vpermq     ymm2, ymm2, 0xd8
    vmovdqu    [edx], ymm0
    vmovdqu    [edx + edi], ymm2
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    vzeroupper
    ret
  }
}
#endif  // HAS_SPLITUVROW_AVX2

#ifdef HAS_MERGEUVROW_SSE2
__declspec(naked) void MergeUVRow_SSE2(const uint8_t* src_u,
                                       const uint8_t* src_v,
                                       uint8_t* dst_uv,
                                       int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]  // src_u
    mov        edx, [esp + 4 + 8]  // src_v
    mov        edi, [esp + 4 + 12]  // dst_uv
    mov        ecx, [esp + 4 + 16]  // width
    sub        edx, eax

  convertloop:
    movdqu     xmm0, [eax]  // read 16 U's
    movdqu     xmm1, [eax + edx]  // and 16 V's
    lea        eax,  [eax + 16]
    movdqa     xmm2, xmm0
    punpcklbw  xmm0, xmm1  // first 8 UV pairs
    punpckhbw  xmm2, xmm1  // next 8 UV pairs
    movdqu     [edi], xmm0
    movdqu     [edi + 16], xmm2
    lea        edi, [edi + 32]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}
#endif  //  HAS_MERGEUVROW_SSE2

#ifdef HAS_MERGEUVROW_AVX2
__declspec(naked) void MergeUVRow_AVX2(const uint8_t* src_u,
                                       const uint8_t* src_v,
                                       uint8_t* dst_uv,
                                       int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]  // src_u
    mov        edx, [esp + 4 + 8]  // src_v
    mov        edi, [esp + 4 + 12]  // dst_uv
    mov        ecx, [esp + 4 + 16]  // width
    sub        edx, eax

  convertloop:
    vpmovzxbw  ymm0, [eax]
    vpmovzxbw  ymm1, [eax + edx]
    lea        eax,  [eax + 16]
    vpsllw     ymm1, ymm1, 8
    vpor       ymm2, ymm1, ymm0
    vmovdqu    [edi], ymm2
    lea        edi, [edi + 32]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    vzeroupper
    ret
  }
}
#endif  //  HAS_MERGEUVROW_AVX2

#ifdef HAS_COPYROW_SSE2
// CopyRow copys 'width' bytes using a 16 byte load/store, 32 bytes at time.
__declspec(naked) void CopyRow_SSE2(const uint8_t* src,
                                    uint8_t* dst,
                                    int width) {
  __asm {
    mov        eax, [esp + 4]  // src
    mov        edx, [esp + 8]  // dst
    mov        ecx, [esp + 12]  // width
    test       eax, 15
    jne        convertloopu
    test       edx, 15
    jne        convertloopu

  convertloopa:
    movdqa     xmm0, [eax]
    movdqa     xmm1, [eax + 16]
    lea        eax, [eax + 32]
    movdqa     [edx], xmm0
    movdqa     [edx + 16], xmm1
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         convertloopa
    ret

  convertloopu:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax, [eax + 32]
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         convertloopu
    ret
  }
}
#endif  // HAS_COPYROW_SSE2

#ifdef HAS_COPYROW_AVX
// CopyRow copys 'width' bytes using a 32 byte load/store, 64 bytes at time.
__declspec(naked) void CopyRow_AVX(const uint8_t* src,
                                   uint8_t* dst,
                                   int width) {
  __asm {
    mov        eax, [esp + 4]  // src
    mov        edx, [esp + 8]  // dst
    mov        ecx, [esp + 12]  // width

  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax, [eax + 64]
    vmovdqu    [edx], ymm0
    vmovdqu    [edx + 32], ymm1
    lea        edx, [edx + 64]
    sub        ecx, 64
    jg         convertloop

    vzeroupper
    ret
  }
}
#endif  // HAS_COPYROW_AVX

// Multiple of 1.
__declspec(naked) void CopyRow_ERMS(const uint8_t* src,
                                    uint8_t* dst,
                                    int width) {
  __asm {
    mov        eax, esi
    mov        edx, edi
    mov        esi, [esp + 4]  // src
    mov        edi, [esp + 8]  // dst
    mov        ecx, [esp + 12]  // width
    rep movsb
    mov        edi, edx
    mov        esi, eax
    ret
  }
}

#ifdef HAS_ARGBCOPYALPHAROW_SSE2
// width in pixels
__declspec(naked) void ARGBCopyAlphaRow_SSE2(const uint8_t* src,
                                             uint8_t* dst,
                                             int width) {
  __asm {
    mov        eax, [esp + 4]  // src
    mov        edx, [esp + 8]  // dst
    mov        ecx, [esp + 12]  // width
    pcmpeqb    xmm0, xmm0  // generate mask 0xff000000
    pslld      xmm0, 24
    pcmpeqb    xmm1, xmm1  // generate mask 0x00ffffff
    psrld      xmm1, 8

  convertloop:
    movdqu     xmm2, [eax]
    movdqu     xmm3, [eax + 16]
    lea        eax, [eax + 32]
    movdqu     xmm4, [edx]
    movdqu     xmm5, [edx + 16]
    pand       xmm2, xmm0
    pand       xmm3, xmm0
    pand       xmm4, xmm1
    pand       xmm5, xmm1
    por        xmm2, xmm4
    por        xmm3, xmm5
    movdqu     [edx], xmm2
    movdqu     [edx + 16], xmm3
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    ret
  }
}
#endif  // HAS_ARGBCOPYALPHAROW_SSE2

#ifdef HAS_ARGBCOPYALPHAROW_AVX2
// width in pixels
__declspec(naked) void ARGBCopyAlphaRow_AVX2(const uint8_t* src,
                                             uint8_t* dst,
                                             int width) {
  __asm {
    mov        eax, [esp + 4]  // src
    mov        edx, [esp + 8]  // dst
    mov        ecx, [esp + 12]  // width
    vpcmpeqb   ymm0, ymm0, ymm0
    vpsrld     ymm0, ymm0, 8  // generate mask 0x00ffffff

  convertloop:
    vmovdqu    ymm1, [eax]
    vmovdqu    ymm2, [eax + 32]
    lea        eax, [eax + 64]
    vpblendvb  ymm1, ymm1, [edx], ymm0
    vpblendvb  ymm2, ymm2, [edx + 32], ymm0
    vmovdqu    [edx], ymm1
    vmovdqu    [edx + 32], ymm2
    lea        edx, [edx + 64]
    sub        ecx, 16
    jg         convertloop

    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBCOPYALPHAROW_AVX2

#ifdef HAS_ARGBEXTRACTALPHAROW_SSE2
// width in pixels
__declspec(naked) void ARGBExtractAlphaRow_SSE2(const uint8_t* src_argb,
                                                uint8_t* dst_a,
                                                int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_a
    mov        ecx, [esp + 12]  // width

  extractloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax, [eax + 32]
    psrld      xmm0, 24
    psrld      xmm1, 24
    packssdw   xmm0, xmm1
    packuswb   xmm0, xmm0
    movq       qword ptr [edx], xmm0
    lea        edx, [edx + 8]
    sub        ecx, 8
    jg         extractloop

    ret
  }
}
#endif  // HAS_ARGBEXTRACTALPHAROW_SSE2

#ifdef HAS_ARGBEXTRACTALPHAROW_AVX2
// width in pixels
__declspec(naked) void ARGBExtractAlphaRow_AVX2(const uint8_t* src_argb,
                                                uint8_t* dst_a,
                                                int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_a
    mov        ecx, [esp + 12]  // width
    vmovdqa    ymm4, ymmword ptr kPermdARGBToY_AVX

  extractloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vpsrld     ymm0, ymm0, 24
    vpsrld     ymm1, ymm1, 24
    vmovdqu    ymm2, [eax + 64]
    vmovdqu    ymm3, [eax + 96]
    lea        eax, [eax + 128]
    vpackssdw  ymm0, ymm0, ymm1  // mutates
    vpsrld     ymm2, ymm2, 24
    vpsrld     ymm3, ymm3, 24
    vpackssdw  ymm2, ymm2, ymm3  // mutates
    vpackuswb  ymm0, ymm0, ymm2  // mutates
    vpermd     ymm0, ymm4, ymm0  // unmutate
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         extractloop

    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBEXTRACTALPHAROW_AVX2

#ifdef HAS_ARGBCOPYYTOALPHAROW_SSE2
// width in pixels
__declspec(naked) void ARGBCopyYToAlphaRow_SSE2(const uint8_t* src,
                                                uint8_t* dst,
                                                int width) {
  __asm {
    mov        eax, [esp + 4]  // src
    mov        edx, [esp + 8]  // dst
    mov        ecx, [esp + 12]  // width
    pcmpeqb    xmm0, xmm0  // generate mask 0xff000000
    pslld      xmm0, 24
    pcmpeqb    xmm1, xmm1  // generate mask 0x00ffffff
    psrld      xmm1, 8

  convertloop:
    movq       xmm2, qword ptr [eax]  // 8 Y's
    lea        eax, [eax + 8]
    punpcklbw  xmm2, xmm2
    punpckhwd  xmm3, xmm2
    punpcklwd  xmm2, xmm2
    movdqu     xmm4, [edx]
    movdqu     xmm5, [edx + 16]
    pand       xmm2, xmm0
    pand       xmm3, xmm0
    pand       xmm4, xmm1
    pand       xmm5, xmm1
    por        xmm2, xmm4
    por        xmm3, xmm5
    movdqu     [edx], xmm2
    movdqu     [edx + 16], xmm3
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    ret
  }
}
#endif  // HAS_ARGBCOPYYTOALPHAROW_SSE2

#ifdef HAS_ARGBCOPYYTOALPHAROW_AVX2
// width in pixels
__declspec(naked) void ARGBCopyYToAlphaRow_AVX2(const uint8_t* src,
                                                uint8_t* dst,
                                                int width) {
  __asm {
    mov        eax, [esp + 4]  // src
    mov        edx, [esp + 8]  // dst
    mov        ecx, [esp + 12]  // width
    vpcmpeqb   ymm0, ymm0, ymm0
    vpsrld     ymm0, ymm0, 8  // generate mask 0x00ffffff

  convertloop:
    vpmovzxbd  ymm1, qword ptr [eax]
    vpmovzxbd  ymm2, qword ptr [eax + 8]
    lea        eax, [eax + 16]
    vpslld     ymm1, ymm1, 24
    vpslld     ymm2, ymm2, 24
    vpblendvb  ymm1, ymm1, [edx], ymm0
    vpblendvb  ymm2, ymm2, [edx + 32], ymm0
    vmovdqu    [edx], ymm1
    vmovdqu    [edx + 32], ymm2
    lea        edx, [edx + 64]
    sub        ecx, 16
    jg         convertloop

    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBCOPYYTOALPHAROW_AVX2

#ifdef HAS_SETROW_X86
// Write 'width' bytes using an 8 bit value repeated.
// width should be multiple of 4.
__declspec(naked) void SetRow_X86(uint8_t* dst, uint8_t v8, int width) {
  __asm {
    movzx      eax, byte ptr [esp + 8]  // v8
    mov        edx, 0x01010101  // Duplicate byte to all bytes.
    mul        edx  // overwrites edx with upper part of result.
    mov        edx, edi
    mov        edi, [esp + 4]  // dst
    mov        ecx, [esp + 12]  // width
    shr        ecx, 2
    rep stosd
    mov        edi, edx
    ret
  }
}

// Write 'width' bytes using an 8 bit value repeated.
__declspec(naked) void SetRow_ERMS(uint8_t* dst, uint8_t v8, int width) {
  __asm {
    mov        edx, edi
    mov        edi, [esp + 4]  // dst
    mov        eax, [esp + 8]  // v8
    mov        ecx, [esp + 12]  // width
    rep stosb
    mov        edi, edx
    ret
  }
}

// Write 'width' 32 bit values.
__declspec(naked) void ARGBSetRow_X86(uint8_t* dst_argb,
                                      uint32_t v32,
                                      int width) {
  __asm {
    mov        edx, edi
    mov        edi, [esp + 4]  // dst
    mov        eax, [esp + 8]  // v32
    mov        ecx, [esp + 12]  // width
    rep stosd
    mov        edi, edx
    ret
  }
}
#endif  // HAS_SETROW_X86

#ifdef HAS_YUY2TOYROW_AVX2
__declspec(naked) void YUY2ToYRow_AVX2(const uint8_t* src_yuy2,
                                       uint8_t* dst_y,
                                       int width) {
  __asm {
    mov        eax, [esp + 4]  // src_yuy2
    mov        edx, [esp + 8]  // dst_y
    mov        ecx, [esp + 12]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8

  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpand      ymm0, ymm0, ymm5  // even bytes are Y
    vpand      ymm1, ymm1, ymm5
    vpackuswb  ymm0, ymm0, ymm1  // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         convertloop
    vzeroupper
    ret
  }
}

__declspec(naked) void YUY2ToUVRow_AVX2(const uint8_t* src_yuy2,
                                        int stride_yuy2,
                                        uint8_t* dst_u,
                                        uint8_t* dst_v,
                                        int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_yuy2
    mov        esi, [esp + 8 + 8]  // stride_yuy2
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vpavgb     ymm0, ymm0, [eax + esi]
    vpavgb     ymm1, ymm1, [eax + esi + 32]
    lea        eax,  [eax + 64]
    vpsrlw     ymm0, ymm0, 8  // YUYV -> UVUV
    vpsrlw     ymm1, ymm1, 8
    vpackuswb  ymm0, ymm0, ymm1  // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vpand      ymm1, ymm0, ymm5  // U
    vpsrlw     ymm0, ymm0, 8  // V
    vpackuswb  ymm1, ymm1, ymm1  // mutates.
    vpackuswb  ymm0, ymm0, ymm0  // mutates.
    vpermq     ymm1, ymm1, 0xd8
    vpermq     ymm0, ymm0, 0xd8
    vextractf128 [edx], ymm1, 0  // U
    vextractf128 [edx + edi], ymm0, 0  // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}

__declspec(naked) void YUY2ToUV422Row_AVX2(const uint8_t* src_yuy2,
                                           uint8_t* dst_u,
                                           uint8_t* dst_v,
                                           int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]  // src_yuy2
    mov        edx, [esp + 4 + 8]  // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpsrlw     ymm0, ymm0, 8  // YUYV -> UVUV
    vpsrlw     ymm1, ymm1, 8
    vpackuswb  ymm0, ymm0, ymm1  // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vpand      ymm1, ymm0, ymm5  // U
    vpsrlw     ymm0, ymm0, 8  // V
    vpackuswb  ymm1, ymm1, ymm1  // mutates.
    vpackuswb  ymm0, ymm0, ymm0  // mutates.
    vpermq     ymm1, ymm1, 0xd8
    vpermq     ymm0, ymm0, 0xd8
    vextractf128 [edx], ymm1, 0  // U
    vextractf128 [edx + edi], ymm0, 0  // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    vzeroupper
    ret
  }
}

__declspec(naked) void UYVYToYRow_AVX2(const uint8_t* src_uyvy,
                                       uint8_t* dst_y,
                                       int width) {
  __asm {
    mov        eax, [esp + 4]  // src_uyvy
    mov        edx, [esp + 8]  // dst_y
    mov        ecx, [esp + 12]  // width

  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpsrlw     ymm0, ymm0, 8  // odd bytes are Y
    vpsrlw     ymm1, ymm1, 8
    vpackuswb  ymm0, ymm0, ymm1  // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 32
    jg         convertloop
    vzeroupper
    ret
  }
}

__declspec(naked) void UYVYToUVRow_AVX2(const uint8_t* src_uyvy,
                                        int stride_uyvy,
                                        uint8_t* dst_u,
                                        uint8_t* dst_v,
                                        int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_yuy2
    mov        esi, [esp + 8 + 8]  // stride_yuy2
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    vpavgb     ymm0, ymm0, [eax + esi]
    vpavgb     ymm1, ymm1, [eax + esi + 32]
    lea        eax,  [eax + 64]
    vpand      ymm0, ymm0, ymm5  // UYVY -> UVUV
    vpand      ymm1, ymm1, ymm5
    vpackuswb  ymm0, ymm0, ymm1  // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vpand      ymm1, ymm0, ymm5  // U
    vpsrlw     ymm0, ymm0, 8  // V
    vpackuswb  ymm1, ymm1, ymm1  // mutates.
    vpackuswb  ymm0, ymm0, ymm0  // mutates.
    vpermq     ymm1, ymm1, 0xd8
    vpermq     ymm0, ymm0, 0xd8
    vextractf128 [edx], ymm1, 0  // U
    vextractf128 [edx + edi], ymm0, 0  // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}

__declspec(naked) void UYVYToUV422Row_AVX2(const uint8_t* src_uyvy,
                                           uint8_t* dst_u,
                                           uint8_t* dst_v,
                                           int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]  // src_yuy2
    mov        edx, [esp + 4 + 8]  // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // width
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0x00ff00ff
    vpsrlw     ymm5, ymm5, 8
    sub        edi, edx

  convertloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax,  [eax + 64]
    vpand      ymm0, ymm0, ymm5  // UYVY -> UVUV
    vpand      ymm1, ymm1, ymm5
    vpackuswb  ymm0, ymm0, ymm1  // mutates.
    vpermq     ymm0, ymm0, 0xd8
    vpand      ymm1, ymm0, ymm5  // U
    vpsrlw     ymm0, ymm0, 8  // V
    vpackuswb  ymm1, ymm1, ymm1  // mutates.
    vpackuswb  ymm0, ymm0, ymm0  // mutates.
    vpermq     ymm1, ymm1, 0xd8
    vpermq     ymm0, ymm0, 0xd8
    vextractf128 [edx], ymm1, 0  // U
    vextractf128 [edx + edi], ymm0, 0  // V
    lea        edx, [edx + 16]
    sub        ecx, 32
    jg         convertloop

    pop        edi
    vzeroupper
    ret
  }
}
#endif  // HAS_YUY2TOYROW_AVX2

#ifdef HAS_YUY2TOYROW_SSE2
__declspec(naked) void YUY2ToYRow_SSE2(const uint8_t* src_yuy2,
                                       uint8_t* dst_y,
                                       int width) {
  __asm {
    mov        eax, [esp + 4]  // src_yuy2
    mov        edx, [esp + 8]  // dst_y
    mov        ecx, [esp + 12]  // width
    pcmpeqb    xmm5, xmm5  // generate mask 0x00ff00ff
    psrlw      xmm5, 8

  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    pand       xmm0, xmm5  // even bytes are Y
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop
    ret
  }
}

__declspec(naked) void YUY2ToUVRow_SSE2(const uint8_t* src_yuy2,
                                        int stride_yuy2,
                                        uint8_t* dst_u,
                                        uint8_t* dst_v,
                                        int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_yuy2
    mov        esi, [esp + 8 + 8]  // stride_yuy2
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    pcmpeqb    xmm5, xmm5  // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + esi]
    movdqu     xmm3, [eax + esi + 16]
    lea        eax,  [eax + 32]
    pavgb      xmm0, xmm2
    pavgb      xmm1, xmm3
    psrlw      xmm0, 8  // YUYV -> UVUV
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8  // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) void YUY2ToUV422Row_SSE2(const uint8_t* src_yuy2,
                                           uint8_t* dst_u,
                                           uint8_t* dst_v,
                                           int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]  // src_yuy2
    mov        edx, [esp + 4 + 8]  // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm5, xmm5  // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    psrlw      xmm0, 8  // YUYV -> UVUV
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8  // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}

__declspec(naked) void UYVYToYRow_SSE2(const uint8_t* src_uyvy,
                                       uint8_t* dst_y,
                                       int width) {
  __asm {
    mov        eax, [esp + 4]  // src_uyvy
    mov        edx, [esp + 8]  // dst_y
    mov        ecx, [esp + 12]  // width

  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    psrlw      xmm0, 8  // odd bytes are Y
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop
    ret
  }
}

__declspec(naked) void UYVYToUVRow_SSE2(const uint8_t* src_uyvy,
                                        int stride_uyvy,
                                        uint8_t* dst_u,
                                        uint8_t* dst_v,
                                        int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_yuy2
    mov        esi, [esp + 8 + 8]  // stride_yuy2
    mov        edx, [esp + 8 + 12]  // dst_u
    mov        edi, [esp + 8 + 16]  // dst_v
    mov        ecx, [esp + 8 + 20]  // width
    pcmpeqb    xmm5, xmm5  // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + esi]
    movdqu     xmm3, [eax + esi + 16]
    lea        eax,  [eax + 32]
    pavgb      xmm0, xmm2
    pavgb      xmm1, xmm3
    pand       xmm0, xmm5  // UYVY -> UVUV
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8  // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) void UYVYToUV422Row_SSE2(const uint8_t* src_uyvy,
                                           uint8_t* dst_u,
                                           uint8_t* dst_v,
                                           int width) {
  __asm {
    push       edi
    mov        eax, [esp + 4 + 4]  // src_yuy2
    mov        edx, [esp + 4 + 8]  // dst_u
    mov        edi, [esp + 4 + 12]  // dst_v
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm5, xmm5  // generate mask 0x00ff00ff
    psrlw      xmm5, 8
    sub        edi, edx

  convertloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax,  [eax + 32]
    pand       xmm0, xmm5  // UYVY -> UVUV
    pand       xmm1, xmm5
    packuswb   xmm0, xmm1
    movdqa     xmm1, xmm0
    pand       xmm0, xmm5  // U
    packuswb   xmm0, xmm0
    psrlw      xmm1, 8  // V
    packuswb   xmm1, xmm1
    movq       qword ptr [edx], xmm0
    movq       qword ptr [edx + edi], xmm1
    lea        edx, [edx + 8]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    ret
  }
}
#endif  // HAS_YUY2TOYROW_SSE2

#ifdef HAS_BLENDPLANEROW_SSSE3
// Blend 8 pixels at a time.
// unsigned version of math
// =((A2*C2)+(B2*(255-C2))+255)/256
// signed version of math
// =(((A2-128)*C2)+((B2-128)*(255-C2))+32768+127)/256
__declspec(naked) void BlendPlaneRow_SSSE3(const uint8_t* src0,
                                           const uint8_t* src1,
                                           const uint8_t* alpha,
                                           uint8_t* dst,
                                           int width) {
  __asm {
    push       esi
    push       edi
    pcmpeqb    xmm5, xmm5  // generate mask 0xff00ff00
    psllw      xmm5, 8
    mov        eax, 0x80808080  // 128 for biasing image to signed.
    movd       xmm6, eax
    pshufd     xmm6, xmm6, 0x00

    mov        eax, 0x807f807f  // 32768 + 127 for unbias and round.
    movd       xmm7, eax
    pshufd     xmm7, xmm7, 0x00
    mov        eax, [esp + 8 + 4]  // src0
    mov        edx, [esp + 8 + 8]  // src1
    mov        esi, [esp + 8 + 12]  // alpha
    mov        edi, [esp + 8 + 16]  // dst
    mov        ecx, [esp + 8 + 20]  // width
    sub        eax, esi
    sub        edx, esi
    sub        edi, esi

        // 8 pixel loop.
  convertloop8:
    movq       xmm0, qword ptr [esi]  // alpha
    punpcklbw  xmm0, xmm0
    pxor       xmm0, xmm5  // a, 255-a
    movq       xmm1, qword ptr [eax + esi]  // src0
    movq       xmm2, qword ptr [edx + esi]  // src1
    punpcklbw  xmm1, xmm2
    psubb      xmm1, xmm6  // bias src0/1 - 128
    pmaddubsw  xmm0, xmm1
    paddw      xmm0, xmm7  // unbias result - 32768 and round.
    psrlw      xmm0, 8
    packuswb   xmm0, xmm0
    movq       qword ptr [edi + esi], xmm0
    lea        esi, [esi + 8]
    sub        ecx, 8
    jg         convertloop8

    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_BLENDPLANEROW_SSSE3

#ifdef HAS_BLENDPLANEROW_AVX2
// Blend 32 pixels at a time.
// unsigned version of math
// =((A2*C2)+(B2*(255-C2))+255)/256
// signed version of math
// =(((A2-128)*C2)+((B2-128)*(255-C2))+32768+127)/256
__declspec(naked) void BlendPlaneRow_AVX2(const uint8_t* src0,
                                          const uint8_t* src1,
                                          const uint8_t* alpha,
                                          uint8_t* dst,
                                          int width) {
  __asm {
    push        esi
    push        edi
    vpcmpeqb    ymm5, ymm5, ymm5  // generate mask 0xff00ff00
    vpsllw      ymm5, ymm5, 8
    mov         eax, 0x80808080  // 128 for biasing image to signed.
    vmovd       xmm6, eax
    vbroadcastss ymm6, xmm6
    mov         eax, 0x807f807f  // 32768 + 127 for unbias and round.
    vmovd       xmm7, eax
    vbroadcastss ymm7, xmm7
    mov         eax, [esp + 8 + 4]  // src0
    mov         edx, [esp + 8 + 8]  // src1
    mov         esi, [esp + 8 + 12]  // alpha
    mov         edi, [esp + 8 + 16]  // dst
    mov         ecx, [esp + 8 + 20]  // width
    sub         eax, esi
    sub         edx, esi
    sub         edi, esi

        // 32 pixel loop.
  convertloop32:
    vmovdqu     ymm0, [esi]  // alpha
    vpunpckhbw  ymm3, ymm0, ymm0  // 8..15, 24..31
    vpunpcklbw  ymm0, ymm0, ymm0  // 0..7, 16..23
    vpxor       ymm3, ymm3, ymm5  // a, 255-a
    vpxor       ymm0, ymm0, ymm5  // a, 255-a
    vmovdqu     ymm1, [eax + esi]  // src0
    vmovdqu     ymm2, [edx + esi]  // src1
    vpunpckhbw  ymm4, ymm1, ymm2
    vpunpcklbw  ymm1, ymm1, ymm2
    vpsubb      ymm4, ymm4, ymm6  // bias src0/1 - 128
    vpsubb      ymm1, ymm1, ymm6  // bias src0/1 - 128
    vpmaddubsw  ymm3, ymm3, ymm4
    vpmaddubsw  ymm0, ymm0, ymm1
    vpaddw      ymm3, ymm3, ymm7  // unbias result - 32768 and round.
    vpaddw      ymm0, ymm0, ymm7  // unbias result - 32768 and round.
    vpsrlw      ymm3, ymm3, 8
    vpsrlw      ymm0, ymm0, 8
    vpackuswb   ymm0, ymm0, ymm3
    vmovdqu     [edi + esi], ymm0
    lea         esi, [esi + 32]
    sub         ecx, 32
    jg          convertloop32

    pop         edi
    pop         esi
    vzeroupper
    ret
  }
}
#endif  // HAS_BLENDPLANEROW_AVX2

#ifdef HAS_ARGBBLENDROW_SSSE3
// Shuffle table for isolating alpha.
static const uvec8 kShuffleAlpha = {3u,  0x80, 3u,  0x80, 7u,  0x80, 7u,  0x80,
                                    11u, 0x80, 11u, 0x80, 15u, 0x80, 15u, 0x80};

// Blend 8 pixels at a time.
__declspec(naked) void ARGBBlendRow_SSSE3(const uint8_t* src_argb,
                                          const uint8_t* src_argb1,
                                          uint8_t* dst_argb,
                                          int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_argb
    mov        esi, [esp + 4 + 8]  // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    pcmpeqb    xmm7, xmm7  // generate constant 0x0001
    psrlw      xmm7, 15
    pcmpeqb    xmm6, xmm6  // generate mask 0x00ff00ff
    psrlw      xmm6, 8
    pcmpeqb    xmm5, xmm5  // generate mask 0xff00ff00
    psllw      xmm5, 8
    pcmpeqb    xmm4, xmm4  // generate mask 0xff000000
    pslld      xmm4, 24
    sub        ecx, 4
    jl         convertloop4b  // less than 4 pixels?

        // 4 pixel loop.
  convertloop4:
    movdqu     xmm3, [eax]  // src argb
    lea        eax, [eax + 16]
    movdqa     xmm0, xmm3  // src argb
    pxor       xmm3, xmm4  // ~alpha
    movdqu     xmm2, [esi]  // _r_b
    pshufb     xmm3, xmmword ptr kShuffleAlpha  // alpha
    pand       xmm2, xmm6  // _r_b
    paddw      xmm3, xmm7  // 256 - alpha
    pmullw     xmm2, xmm3  // _r_b * alpha
    movdqu     xmm1, [esi]  // _a_g
    lea        esi, [esi + 16]
    psrlw      xmm1, 8  // _a_g
    por        xmm0, xmm4  // set alpha to 255
    pmullw     xmm1, xmm3  // _a_g * alpha
    psrlw      xmm2, 8  // _r_b convert to 8 bits again
    paddusb    xmm0, xmm2  // + src argb
    pand       xmm1, xmm5  // a_g_ convert to 8 bits again
    paddusb    xmm0, xmm1  // + src argb
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 4
    jge        convertloop4

  convertloop4b:
    add        ecx, 4 - 1
    jl         convertloop1b

            // 1 pixel loop.
  convertloop1:
    movd       xmm3, [eax]  // src argb
    lea        eax, [eax + 4]
    movdqa     xmm0, xmm3  // src argb
    pxor       xmm3, xmm4  // ~alpha
    movd       xmm2, [esi]  // _r_b
    pshufb     xmm3, xmmword ptr kShuffleAlpha  // alpha
    pand       xmm2, xmm6  // _r_b
    paddw      xmm3, xmm7  // 256 - alpha
    pmullw     xmm2, xmm3  // _r_b * alpha
    movd       xmm1, [esi]  // _a_g
    lea        esi, [esi + 4]
    psrlw      xmm1, 8  // _a_g
    por        xmm0, xmm4  // set alpha to 255
    pmullw     xmm1, xmm3  // _a_g * alpha
    psrlw      xmm2, 8  // _r_b convert to 8 bits again
    paddusb    xmm0, xmm2  // + src argb
    pand       xmm1, xmm5  // a_g_ convert to 8 bits again
    paddusb    xmm0, xmm1  // + src argb
    movd       [edx], xmm0
    lea        edx, [edx + 4]
    sub        ecx, 1
    jge        convertloop1

  convertloop1b:
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBBLENDROW_SSSE3

#ifdef HAS_ARGBATTENUATEROW_SSSE3
// Shuffle table duplicating alpha.
static const uvec8 kShuffleAlpha0 = {
    3u, 3u, 3u, 3u, 3u, 3u, 128u, 128u, 7u, 7u, 7u, 7u, 7u, 7u, 128u, 128u,
};
static const uvec8 kShuffleAlpha1 = {
    11u, 11u, 11u, 11u, 11u, 11u, 128u, 128u,
    15u, 15u, 15u, 15u, 15u, 15u, 128u, 128u,
};
__declspec(naked) void ARGBAttenuateRow_SSSE3(const uint8_t* src_argb,
                                              uint8_t* dst_argb,
                                              int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_argb
    mov        ecx, [esp + 12]  // width
    pcmpeqb    xmm3, xmm3  // generate mask 0xff000000
    pslld      xmm3, 24
    movdqa     xmm4, xmmword ptr kShuffleAlpha0
    movdqa     xmm5, xmmword ptr kShuffleAlpha1

 convertloop:
    movdqu     xmm0, [eax]  // read 4 pixels
    pshufb     xmm0, xmm4  // isolate first 2 alphas
    movdqu     xmm1, [eax]  // read 4 pixels
    punpcklbw  xmm1, xmm1  // first 2 pixel rgbs
    pmulhuw    xmm0, xmm1  // rgb * a
    movdqu     xmm1, [eax]  // read 4 pixels
    pshufb     xmm1, xmm5  // isolate next 2 alphas
    movdqu     xmm2, [eax]  // read 4 pixels
    punpckhbw  xmm2, xmm2  // next 2 pixel rgbs
    pmulhuw    xmm1, xmm2  // rgb * a
    movdqu     xmm2, [eax]  // mask original alpha
    lea        eax, [eax + 16]
    pand       xmm2, xmm3
    psrlw      xmm0, 8
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    por        xmm0, xmm2  // copy original alpha
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 4
    jg         convertloop

    ret
  }
}
#endif  // HAS_ARGBATTENUATEROW_SSSE3

#ifdef HAS_ARGBATTENUATEROW_AVX2
// Shuffle table duplicating alpha.
static const uvec8 kShuffleAlpha_AVX2 = {6u,   7u,   6u,   7u,  6u,  7u,
                                         128u, 128u, 14u,  15u, 14u, 15u,
                                         14u,  15u,  128u, 128u};
__declspec(naked) void ARGBAttenuateRow_AVX2(const uint8_t* src_argb,
                                             uint8_t* dst_argb,
                                             int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_argb
    mov        ecx, [esp + 12]  // width
    sub        edx, eax
    vbroadcastf128 ymm4, xmmword ptr kShuffleAlpha_AVX2
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0xff000000
    vpslld     ymm5, ymm5, 24

 convertloop:
    vmovdqu    ymm6, [eax]  // read 8 pixels.
    vpunpcklbw ymm0, ymm6, ymm6  // low 4 pixels. mutated.
    vpunpckhbw ymm1, ymm6, ymm6  // high 4 pixels. mutated.
    vpshufb    ymm2, ymm0, ymm4  // low 4 alphas
    vpshufb    ymm3, ymm1, ymm4  // high 4 alphas
    vpmulhuw   ymm0, ymm0, ymm2  // rgb * a
    vpmulhuw   ymm1, ymm1, ymm3  // rgb * a
    vpand      ymm6, ymm6, ymm5  // isolate alpha
    vpsrlw     ymm0, ymm0, 8
    vpsrlw     ymm1, ymm1, 8
    vpackuswb  ymm0, ymm0, ymm1  // unmutated.
    vpor       ymm0, ymm0, ymm6  // copy original alpha
    vmovdqu    [eax + edx], ymm0
    lea        eax, [eax + 32]
    sub        ecx, 8
    jg         convertloop

    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBATTENUATEROW_AVX2

#ifdef HAS_ARGBUNATTENUATEROW_SSE2
// Unattenuate 4 pixels at a time.
__declspec(naked) void ARGBUnattenuateRow_SSE2(const uint8_t* src_argb,
                                               uint8_t* dst_argb,
                                               int width) {
  __asm {
    push       ebx
    push       esi
    push       edi
    mov        eax, [esp + 12 + 4]  // src_argb
    mov        edx, [esp + 12 + 8]  // dst_argb
    mov        ecx, [esp + 12 + 12]  // width
    lea        ebx, fixed_invtbl8

 convertloop:
    movdqu     xmm0, [eax]  // read 4 pixels
    movzx      esi, byte ptr [eax + 3]  // first alpha
    movzx      edi, byte ptr [eax + 7]  // second alpha
    punpcklbw  xmm0, xmm0  // first 2
    movd       xmm2, dword ptr [ebx + esi * 4]
    movd       xmm3, dword ptr [ebx + edi * 4]
    pshuflw    xmm2, xmm2, 040h  // first 4 inv_alpha words.  1, a, a, a
    pshuflw    xmm3, xmm3, 040h  // next 4 inv_alpha words
    movlhps    xmm2, xmm3
    pmulhuw    xmm0, xmm2  // rgb * a

    movdqu     xmm1, [eax]  // read 4 pixels
    movzx      esi, byte ptr [eax + 11]  // third alpha
    movzx      edi, byte ptr [eax + 15]  // forth alpha
    punpckhbw  xmm1, xmm1  // next 2
    movd       xmm2, dword ptr [ebx + esi * 4]
    movd       xmm3, dword ptr [ebx + edi * 4]
    pshuflw    xmm2, xmm2, 040h  // first 4 inv_alpha words
    pshuflw    xmm3, xmm3, 040h  // next 4 inv_alpha words
    movlhps    xmm2, xmm3
    pmulhuw    xmm1, xmm2  // rgb * a
    lea        eax, [eax + 16]
    packuswb   xmm0, xmm1
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 4
    jg         convertloop

    pop        edi
    pop        esi
    pop        ebx
    ret
  }
}
#endif  // HAS_ARGBUNATTENUATEROW_SSE2

#ifdef HAS_ARGBUNATTENUATEROW_AVX2
// Shuffle table duplicating alpha.
static const uvec8 kUnattenShuffleAlpha_AVX2 = {
    0u, 1u, 0u, 1u, 0u, 1u, 6u, 7u, 8u, 9u, 8u, 9u, 8u, 9u, 14u, 15u};
// TODO(fbarchard): Enable USE_GATHER for future hardware if faster.
// USE_GATHER is not on by default, due to being a slow instruction.
#ifdef USE_GATHER
__declspec(naked) void ARGBUnattenuateRow_AVX2(const uint8_t* src_argb,
                                               uint8_t* dst_argb,
                                               int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_argb
    mov        ecx, [esp + 12]  // width
    sub        edx, eax
    vbroadcastf128 ymm4, xmmword ptr kUnattenShuffleAlpha_AVX2

 convertloop:
    vmovdqu    ymm6, [eax]  // read 8 pixels.
    vpcmpeqb   ymm5, ymm5, ymm5  // generate mask 0xffffffff for gather.
    vpsrld     ymm2, ymm6, 24  // alpha in low 8 bits.
    vpunpcklbw ymm0, ymm6, ymm6  // low 4 pixels. mutated.
    vpunpckhbw ymm1, ymm6, ymm6  // high 4 pixels. mutated.
    vpgatherdd ymm3, [ymm2 * 4 + fixed_invtbl8], ymm5  // ymm5 cleared.  1, a
    vpunpcklwd ymm2, ymm3, ymm3  // low 4 inverted alphas. mutated. 1, 1, a, a
    vpunpckhwd ymm3, ymm3, ymm3  // high 4 inverted alphas. mutated.
    vpshufb    ymm2, ymm2, ymm4  // replicate low 4 alphas. 1, a, a, a
    vpshufb    ymm3, ymm3, ymm4  // replicate high 4 alphas
    vpmulhuw   ymm0, ymm0, ymm2  // rgb * ia
    vpmulhuw   ymm1, ymm1, ymm3  // rgb * ia
    vpackuswb  ymm0, ymm0, ymm1  // unmutated.
    vmovdqu    [eax + edx], ymm0
    lea        eax, [eax + 32]
    sub        ecx, 8
    jg         convertloop

    vzeroupper
    ret
  }
}
#else   // USE_GATHER
__declspec(naked) void ARGBUnattenuateRow_AVX2(const uint8_t* src_argb,
                                               uint8_t* dst_argb,
                                               int width) {
  __asm {

    push       ebx
    push       esi
    push       edi
    mov        eax, [esp + 12 + 4]  // src_argb
    mov        edx, [esp + 12 + 8]  // dst_argb
    mov        ecx, [esp + 12 + 12]  // width
    sub        edx, eax
    lea        ebx, fixed_invtbl8
    vbroadcastf128 ymm5, xmmword ptr kUnattenShuffleAlpha_AVX2

 convertloop:
        // replace VPGATHER
    movzx      esi, byte ptr [eax + 3]  // alpha0
    movzx      edi, byte ptr [eax + 7]  // alpha1
    vmovd      xmm0, dword ptr [ebx + esi * 4]  // [1,a0]
    vmovd      xmm1, dword ptr [ebx + edi * 4]  // [1,a1]
    movzx      esi, byte ptr [eax + 11]  // alpha2
    movzx      edi, byte ptr [eax + 15]  // alpha3
    vpunpckldq xmm6, xmm0, xmm1  // [1,a1,1,a0]
    vmovd      xmm2, dword ptr [ebx + esi * 4]  // [1,a2]
    vmovd      xmm3, dword ptr [ebx + edi * 4]  // [1,a3]
    movzx      esi, byte ptr [eax + 19]  // alpha4
    movzx      edi, byte ptr [eax + 23]  // alpha5
    vpunpckldq xmm7, xmm2, xmm3  // [1,a3,1,a2]
    vmovd      xmm0, dword ptr [ebx + esi * 4]  // [1,a4]
    vmovd      xmm1, dword ptr [ebx + edi * 4]  // [1,a5]
    movzx      esi, byte ptr [eax + 27]  // alpha6
    movzx      edi, byte ptr [eax + 31]  // alpha7
    vpunpckldq xmm0, xmm0, xmm1  // [1,a5,1,a4]
    vmovd      xmm2, dword ptr [ebx + esi * 4]  // [1,a6]
    vmovd      xmm3, dword ptr [ebx + edi * 4]  // [1,a7]
    vpunpckldq xmm2, xmm2, xmm3  // [1,a7,1,a6]
    vpunpcklqdq xmm3, xmm6, xmm7  // [1,a3,1,a2,1,a1,1,a0]
    vpunpcklqdq xmm0, xmm0, xmm2  // [1,a7,1,a6,1,a5,1,a4]
    vinserti128 ymm3, ymm3, xmm0, 1                // [1,a7,1,a6,1,a5,1,a4,1,a3,1,a2,1,a1,1,a0]
    // end of VPGATHER

    vmovdqu    ymm6, [eax]  // read 8 pixels.
    vpunpcklbw ymm0, ymm6, ymm6  // low 4 pixels. mutated.
    vpunpckhbw ymm1, ymm6, ymm6  // high 4 pixels. mutated.
    vpunpcklwd ymm2, ymm3, ymm3  // low 4 inverted alphas. mutated. 1, 1, a, a
    vpunpckhwd ymm3, ymm3, ymm3  // high 4 inverted alphas. mutated.
    vpshufb    ymm2, ymm2, ymm5  // replicate low 4 alphas. 1, a, a, a
    vpshufb    ymm3, ymm3, ymm5  // replicate high 4 alphas
    vpmulhuw   ymm0, ymm0, ymm2  // rgb * ia
    vpmulhuw   ymm1, ymm1, ymm3  // rgb * ia
    vpackuswb  ymm0, ymm0, ymm1             // unmutated.
    vmovdqu    [eax + edx], ymm0
    lea        eax, [eax + 32]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    pop        ebx
    vzeroupper
    ret
  }
}
#endif  // USE_GATHER
#endif  // HAS_ARGBATTENUATEROW_AVX2

#ifdef HAS_ARGBGRAYROW_SSSE3
// Convert 8 ARGB pixels (64 bytes) to 8 Gray ARGB pixels.
__declspec(naked) void ARGBGrayRow_SSSE3(const uint8_t* src_argb,
                                         uint8_t* dst_argb,
                                         int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_argb */
    mov        ecx, [esp + 12] /* width */
    movdqa     xmm4, xmmword ptr kARGBToYJ
    movdqa     xmm5, xmmword ptr kAddYJ64

 convertloop:
    movdqu     xmm0, [eax]  // G
    movdqu     xmm1, [eax + 16]
    pmaddubsw  xmm0, xmm4
    pmaddubsw  xmm1, xmm4
    phaddw     xmm0, xmm1
    paddw      xmm0, xmm5  // Add .5 for rounding.
    psrlw      xmm0, 7
    packuswb   xmm0, xmm0  // 8 G bytes
    movdqu     xmm2, [eax]  // A
    movdqu     xmm3, [eax + 16]
    lea        eax, [eax + 32]
    psrld      xmm2, 24
    psrld      xmm3, 24
    packuswb   xmm2, xmm3
    packuswb   xmm2, xmm2  // 8 A bytes
    movdqa     xmm3, xmm0  // Weave into GG, GA, then GGGA
    punpcklbw  xmm0, xmm0  // 8 GG words
    punpcklbw  xmm3, xmm2  // 8 GA words
    movdqa     xmm1, xmm0
    punpcklwd  xmm0, xmm3  // GGGA first 4
    punpckhwd  xmm1, xmm3  // GGGA next 4
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop
    ret
  }
}
#endif  // HAS_ARGBGRAYROW_SSSE3

#ifdef HAS_ARGBSEPIAROW_SSSE3
//    b = (r * 35 + g * 68 + b * 17) >> 7
//    g = (r * 45 + g * 88 + b * 22) >> 7
//    r = (r * 50 + g * 98 + b * 24) >> 7
// Constant for ARGB color to sepia tone.
static const vec8 kARGBToSepiaB = {17, 68, 35, 0, 17, 68, 35, 0,
                                   17, 68, 35, 0, 17, 68, 35, 0};

static const vec8 kARGBToSepiaG = {22, 88, 45, 0, 22, 88, 45, 0,
                                   22, 88, 45, 0, 22, 88, 45, 0};

static const vec8 kARGBToSepiaR = {24, 98, 50, 0, 24, 98, 50, 0,
                                   24, 98, 50, 0, 24, 98, 50, 0};

// Convert 8 ARGB pixels (32 bytes) to 8 Sepia ARGB pixels.
__declspec(naked) void ARGBSepiaRow_SSSE3(uint8_t* dst_argb, int width) {
  __asm {
    mov        eax, [esp + 4] /* dst_argb */
    mov        ecx, [esp + 8] /* width */
    movdqa     xmm2, xmmword ptr kARGBToSepiaB
    movdqa     xmm3, xmmword ptr kARGBToSepiaG
    movdqa     xmm4, xmmword ptr kARGBToSepiaR

 convertloop:
    movdqu     xmm0, [eax]  // B
    movdqu     xmm6, [eax + 16]
    pmaddubsw  xmm0, xmm2
    pmaddubsw  xmm6, xmm2
    phaddw     xmm0, xmm6
    psrlw      xmm0, 7
    packuswb   xmm0, xmm0  // 8 B values
    movdqu     xmm5, [eax]  // G
    movdqu     xmm1, [eax + 16]
    pmaddubsw  xmm5, xmm3
    pmaddubsw  xmm1, xmm3
    phaddw     xmm5, xmm1
    psrlw      xmm5, 7
    packuswb   xmm5, xmm5  // 8 G values
    punpcklbw  xmm0, xmm5  // 8 BG values
    movdqu     xmm5, [eax]  // R
    movdqu     xmm1, [eax + 16]
    pmaddubsw  xmm5, xmm4
    pmaddubsw  xmm1, xmm4
    phaddw     xmm5, xmm1
    psrlw      xmm5, 7
    packuswb   xmm5, xmm5  // 8 R values
    movdqu     xmm6, [eax]  // A
    movdqu     xmm1, [eax + 16]
    psrld      xmm6, 24
    psrld      xmm1, 24
    packuswb   xmm6, xmm1
    packuswb   xmm6, xmm6  // 8 A values
    punpcklbw  xmm5, xmm6  // 8 RA values
    movdqa     xmm1, xmm0  // Weave BG, RA together
    punpcklwd  xmm0, xmm5  // BGRA first 4
    punpckhwd  xmm1, xmm5  // BGRA next 4
    movdqu     [eax], xmm0
    movdqu     [eax + 16], xmm1
    lea        eax, [eax + 32]
    sub        ecx, 8
    jg         convertloop
    ret
  }
}
#endif  // HAS_ARGBSEPIAROW_SSSE3

#ifdef HAS_ARGBCOLORMATRIXROW_SSSE3
// Tranform 8 ARGB pixels (32 bytes) with color matrix.
// Same as Sepia except matrix is provided.
// TODO(fbarchard): packuswbs only use half of the reg. To make RGBA, combine R
// and B into a high and low, then G/A, unpackl/hbw and then unpckl/hwd.
__declspec(naked) void ARGBColorMatrixRow_SSSE3(const uint8_t* src_argb,
                                                uint8_t* dst_argb,
                                                const int8_t* matrix_argb,
                                                int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_argb */
    mov        ecx, [esp + 12] /* matrix_argb */
    movdqu     xmm5, [ecx]
    pshufd     xmm2, xmm5, 0x00
    pshufd     xmm3, xmm5, 0x55
    pshufd     xmm4, xmm5, 0xaa
    pshufd     xmm5, xmm5, 0xff
    mov        ecx, [esp + 16] /* width */

 convertloop:
    movdqu     xmm0, [eax]  // B
    movdqu     xmm7, [eax + 16]
    pmaddubsw  xmm0, xmm2
    pmaddubsw  xmm7, xmm2
    movdqu     xmm6, [eax]  // G
    movdqu     xmm1, [eax + 16]
    pmaddubsw  xmm6, xmm3
    pmaddubsw  xmm1, xmm3
    phaddsw    xmm0, xmm7  // B
    phaddsw    xmm6, xmm1  // G
    psraw      xmm0, 6  // B
    psraw      xmm6, 6  // G
    packuswb   xmm0, xmm0  // 8 B values
    packuswb   xmm6, xmm6  // 8 G values
    punpcklbw  xmm0, xmm6  // 8 BG values
    movdqu     xmm1, [eax]  // R
    movdqu     xmm7, [eax + 16]
    pmaddubsw  xmm1, xmm4
    pmaddubsw  xmm7, xmm4
    phaddsw    xmm1, xmm7  // R
    movdqu     xmm6, [eax]  // A
    movdqu     xmm7, [eax + 16]
    pmaddubsw  xmm6, xmm5
    pmaddubsw  xmm7, xmm5
    phaddsw    xmm6, xmm7  // A
    psraw      xmm1, 6  // R
    psraw      xmm6, 6  // A
    packuswb   xmm1, xmm1  // 8 R values
    packuswb   xmm6, xmm6  // 8 A values
    punpcklbw  xmm1, xmm6  // 8 RA values
    movdqa     xmm6, xmm0  // Weave BG, RA together
    punpcklwd  xmm0, xmm1  // BGRA first 4
    punpckhwd  xmm6, xmm1  // BGRA next 4
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm6
    lea        eax, [eax + 32]
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop
    ret
  }
}
#endif  // HAS_ARGBCOLORMATRIXROW_SSSE3

#ifdef HAS_ARGBQUANTIZEROW_SSE2
// Quantize 4 ARGB pixels (16 bytes).
__declspec(naked) void ARGBQuantizeRow_SSE2(uint8_t* dst_argb,
                                            int scale,
                                            int interval_size,
                                            int interval_offset,
                                            int width) {
  __asm {
    mov        eax, [esp + 4] /* dst_argb */
    movd       xmm2, [esp + 8] /* scale */
    movd       xmm3, [esp + 12] /* interval_size */
    movd       xmm4, [esp + 16] /* interval_offset */
    mov        ecx, [esp + 20] /* width */
    pshuflw    xmm2, xmm2, 040h
    pshufd     xmm2, xmm2, 044h
    pshuflw    xmm3, xmm3, 040h
    pshufd     xmm3, xmm3, 044h
    pshuflw    xmm4, xmm4, 040h
    pshufd     xmm4, xmm4, 044h
    pxor       xmm5, xmm5  // constant 0
    pcmpeqb    xmm6, xmm6  // generate mask 0xff000000
    pslld      xmm6, 24

 convertloop:
    movdqu     xmm0, [eax]  // read 4 pixels
    punpcklbw  xmm0, xmm5  // first 2 pixels
    pmulhuw    xmm0, xmm2  // pixel * scale >> 16
    movdqu     xmm1, [eax]  // read 4 pixels
    punpckhbw  xmm1, xmm5  // next 2 pixels
    pmulhuw    xmm1, xmm2
    pmullw     xmm0, xmm3  // * interval_size
    movdqu     xmm7, [eax]  // read 4 pixels
    pmullw     xmm1, xmm3
    pand       xmm7, xmm6  // mask alpha
    paddw      xmm0, xmm4  // + interval_size / 2
    paddw      xmm1, xmm4
    packuswb   xmm0, xmm1
    por        xmm0, xmm7
    movdqu     [eax], xmm0
    lea        eax, [eax + 16]
    sub        ecx, 4
    jg         convertloop
    ret
  }
}
#endif  // HAS_ARGBQUANTIZEROW_SSE2

#ifdef HAS_ARGBSHADEROW_SSE2
// Shade 4 pixels at a time by specified value.
__declspec(naked) void ARGBShadeRow_SSE2(const uint8_t* src_argb,
                                         uint8_t* dst_argb,
                                         int width,
                                         uint32_t value) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_argb
    mov        ecx, [esp + 12]  // width
    movd       xmm2, [esp + 16]  // value
    punpcklbw  xmm2, xmm2
    punpcklqdq xmm2, xmm2

 convertloop:
    movdqu     xmm0, [eax]  // read 4 pixels
    lea        eax, [eax + 16]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm0  // first 2
    punpckhbw  xmm1, xmm1  // next 2
    pmulhuw    xmm0, xmm2  // argb * value
    pmulhuw    xmm1, xmm2  // argb * value
    psrlw      xmm0, 8
    psrlw      xmm1, 8
    packuswb   xmm0, xmm1
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 4
    jg         convertloop

    ret
  }
}
#endif  // HAS_ARGBSHADEROW_SSE2

#ifdef HAS_ARGBMULTIPLYROW_SSE2
// Multiply 2 rows of ARGB pixels together, 4 pixels at a time.
__declspec(naked) void ARGBMultiplyRow_SSE2(const uint8_t* src_argb,
                                            const uint8_t* src_argb1,
                                            uint8_t* dst_argb,
                                            int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_argb
    mov        esi, [esp + 4 + 8]  // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    pxor       xmm5, xmm5  // constant 0

 convertloop:
    movdqu     xmm0, [eax]  // read 4 pixels from src_argb
    movdqu     xmm2, [esi]  // read 4 pixels from src_argb1
    movdqu     xmm1, xmm0
    movdqu     xmm3, xmm2
    punpcklbw  xmm0, xmm0  // first 2
    punpckhbw  xmm1, xmm1  // next 2
    punpcklbw  xmm2, xmm5  // first 2
    punpckhbw  xmm3, xmm5  // next 2
    pmulhuw    xmm0, xmm2  // src_argb * src_argb1 first 2
    pmulhuw    xmm1, xmm3  // src_argb * src_argb1 next 2
    lea        eax, [eax + 16]
    lea        esi, [esi + 16]
    packuswb   xmm0, xmm1
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 4
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_ARGBMULTIPLYROW_SSE2

#ifdef HAS_ARGBADDROW_SSE2
// Add 2 rows of ARGB pixels together, 4 pixels at a time.
// TODO(fbarchard): Port this to posix, neon and other math functions.
__declspec(naked) void ARGBAddRow_SSE2(const uint8_t* src_argb,
                                       const uint8_t* src_argb1,
                                       uint8_t* dst_argb,
                                       int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_argb
    mov        esi, [esp + 4 + 8]  // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width

    sub        ecx, 4
    jl         convertloop49

 convertloop4:
    movdqu     xmm0, [eax]  // read 4 pixels from src_argb
    lea        eax, [eax + 16]
    movdqu     xmm1, [esi]  // read 4 pixels from src_argb1
    lea        esi, [esi + 16]
    paddusb    xmm0, xmm1  // src_argb + src_argb1
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 4
    jge        convertloop4

 convertloop49:
    add        ecx, 4 - 1
    jl         convertloop19

 convertloop1:
    movd       xmm0, [eax]  // read 1 pixels from src_argb
    lea        eax, [eax + 4]
    movd       xmm1, [esi]  // read 1 pixels from src_argb1
    lea        esi, [esi + 4]
    paddusb    xmm0, xmm1  // src_argb + src_argb1
    movd       [edx], xmm0
    lea        edx, [edx + 4]
    sub        ecx, 1
    jge        convertloop1

 convertloop19:
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBADDROW_SSE2

#ifdef HAS_ARGBSUBTRACTROW_SSE2
// Subtract 2 rows of ARGB pixels together, 4 pixels at a time.
__declspec(naked) void ARGBSubtractRow_SSE2(const uint8_t* src_argb,
                                            const uint8_t* src_argb1,
                                            uint8_t* dst_argb,
                                            int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_argb
    mov        esi, [esp + 4 + 8]  // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width

 convertloop:
    movdqu     xmm0, [eax]  // read 4 pixels from src_argb
    lea        eax, [eax + 16]
    movdqu     xmm1, [esi]  // read 4 pixels from src_argb1
    lea        esi, [esi + 16]
    psubusb    xmm0, xmm1  // src_argb - src_argb1
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 4
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_ARGBSUBTRACTROW_SSE2

#ifdef HAS_ARGBMULTIPLYROW_AVX2
// Multiply 2 rows of ARGB pixels together, 8 pixels at a time.
__declspec(naked) void ARGBMultiplyRow_AVX2(const uint8_t* src_argb,
                                            const uint8_t* src_argb1,
                                            uint8_t* dst_argb,
                                            int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_argb
    mov        esi, [esp + 4 + 8]  // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    vpxor      ymm5, ymm5, ymm5  // constant 0

 convertloop:
    vmovdqu    ymm1, [eax]  // read 8 pixels from src_argb
    lea        eax, [eax + 32]
    vmovdqu    ymm3, [esi]  // read 8 pixels from src_argb1
    lea        esi, [esi + 32]
    vpunpcklbw ymm0, ymm1, ymm1  // low 4
    vpunpckhbw ymm1, ymm1, ymm1  // high 4
    vpunpcklbw ymm2, ymm3, ymm5  // low 4
    vpunpckhbw ymm3, ymm3, ymm5  // high 4
    vpmulhuw   ymm0, ymm0, ymm2  // src_argb * src_argb1 low 4
    vpmulhuw   ymm1, ymm1, ymm3  // src_argb * src_argb1 high 4
    vpackuswb  ymm0, ymm0, ymm1
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBMULTIPLYROW_AVX2

#ifdef HAS_ARGBADDROW_AVX2
// Add 2 rows of ARGB pixels together, 8 pixels at a time.
__declspec(naked) void ARGBAddRow_AVX2(const uint8_t* src_argb,
                                       const uint8_t* src_argb1,
                                       uint8_t* dst_argb,
                                       int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_argb
    mov        esi, [esp + 4 + 8]  // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width

 convertloop:
    vmovdqu    ymm0, [eax]  // read 8 pixels from src_argb
    lea        eax, [eax + 32]
    vpaddusb   ymm0, ymm0, [esi]  // add 8 pixels from src_argb1
    lea        esi, [esi + 32]
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBADDROW_AVX2

#ifdef HAS_ARGBSUBTRACTROW_AVX2
// Subtract 2 rows of ARGB pixels together, 8 pixels at a time.
__declspec(naked) void ARGBSubtractRow_AVX2(const uint8_t* src_argb,
                                            const uint8_t* src_argb1,
                                            uint8_t* dst_argb,
                                            int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_argb
    mov        esi, [esp + 4 + 8]  // src_argb1
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width

 convertloop:
    vmovdqu    ymm0, [eax]  // read 8 pixels from src_argb
    lea        eax, [eax + 32]
    vpsubusb   ymm0, ymm0, [esi]  // src_argb - src_argb1
    lea        esi, [esi + 32]
    vmovdqu    [edx], ymm0
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBSUBTRACTROW_AVX2

#ifdef HAS_SOBELXROW_SSE2
// SobelX as a matrix is
// -1  0  1
// -2  0  2
// -1  0  1
__declspec(naked) void SobelXRow_SSE2(const uint8_t* src_y0,
                                      const uint8_t* src_y1,
                                      const uint8_t* src_y2,
                                      uint8_t* dst_sobelx,
                                      int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_y0
    mov        esi, [esp + 8 + 8]  // src_y1
    mov        edi, [esp + 8 + 12]  // src_y2
    mov        edx, [esp + 8 + 16]  // dst_sobelx
    mov        ecx, [esp + 8 + 20]  // width
    sub        esi, eax
    sub        edi, eax
    sub        edx, eax
    pxor       xmm5, xmm5  // constant 0

 convertloop:
    movq       xmm0, qword ptr [eax]  // read 8 pixels from src_y0[0]
    movq       xmm1, qword ptr [eax + 2]  // read 8 pixels from src_y0[2]
    punpcklbw  xmm0, xmm5
    punpcklbw  xmm1, xmm5
    psubw      xmm0, xmm1
    movq       xmm1, qword ptr [eax + esi]  // read 8 pixels from src_y1[0]
    movq       xmm2, qword ptr [eax + esi + 2]  // read 8 pixels from src_y1[2]
    punpcklbw  xmm1, xmm5
    punpcklbw  xmm2, xmm5
    psubw      xmm1, xmm2
    movq       xmm2, qword ptr [eax + edi]  // read 8 pixels from src_y2[0]
    movq       xmm3, qword ptr [eax + edi + 2]  // read 8 pixels from src_y2[2]
    punpcklbw  xmm2, xmm5
    punpcklbw  xmm3, xmm5
    psubw      xmm2, xmm3
    paddw      xmm0, xmm2
    paddw      xmm0, xmm1
    paddw      xmm0, xmm1
    pxor       xmm1, xmm1  // abs = max(xmm0, -xmm0).  SSSE3 could use pabsw
    psubw      xmm1, xmm0
    pmaxsw     xmm0, xmm1
    packuswb   xmm0, xmm0
    movq       qword ptr [eax + edx], xmm0
    lea        eax, [eax + 8]
    sub        ecx, 8
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_SOBELXROW_SSE2

#ifdef HAS_SOBELYROW_SSE2
// SobelY as a matrix is
// -1 -2 -1
//  0  0  0
//  1  2  1
__declspec(naked) void SobelYRow_SSE2(const uint8_t* src_y0,
                                      const uint8_t* src_y1,
                                      uint8_t* dst_sobely,
                                      int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_y0
    mov        esi, [esp + 4 + 8]  // src_y1
    mov        edx, [esp + 4 + 12]  // dst_sobely
    mov        ecx, [esp + 4 + 16]  // width
    sub        esi, eax
    sub        edx, eax
    pxor       xmm5, xmm5  // constant 0

 convertloop:
    movq       xmm0, qword ptr [eax]  // read 8 pixels from src_y0[0]
    movq       xmm1, qword ptr [eax + esi]  // read 8 pixels from src_y1[0]
    punpcklbw  xmm0, xmm5
    punpcklbw  xmm1, xmm5
    psubw      xmm0, xmm1
    movq       xmm1, qword ptr [eax + 1]  // read 8 pixels from src_y0[1]
    movq       xmm2, qword ptr [eax + esi + 1]  // read 8 pixels from src_y1[1]
    punpcklbw  xmm1, xmm5
    punpcklbw  xmm2, xmm5
    psubw      xmm1, xmm2
    movq       xmm2, qword ptr [eax + 2]  // read 8 pixels from src_y0[2]
    movq       xmm3, qword ptr [eax + esi + 2]  // read 8 pixels from src_y1[2]
    punpcklbw  xmm2, xmm5
    punpcklbw  xmm3, xmm5
    psubw      xmm2, xmm3
    paddw      xmm0, xmm2
    paddw      xmm0, xmm1
    paddw      xmm0, xmm1
    pxor       xmm1, xmm1  // abs = max(xmm0, -xmm0).  SSSE3 could use pabsw
    psubw      xmm1, xmm0
    pmaxsw     xmm0, xmm1
    packuswb   xmm0, xmm0
    movq       qword ptr [eax + edx], xmm0
    lea        eax, [eax + 8]
    sub        ecx, 8
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_SOBELYROW_SSE2

#ifdef HAS_SOBELROW_SSE2
// Adds Sobel X and Sobel Y and stores Sobel into ARGB.
// A = 255
// R = Sobel
// G = Sobel
// B = Sobel
__declspec(naked) void SobelRow_SSE2(const uint8_t* src_sobelx,
                                     const uint8_t* src_sobely,
                                     uint8_t* dst_argb,
                                     int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_sobelx
    mov        esi, [esp + 4 + 8]  // src_sobely
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    sub        esi, eax
    pcmpeqb    xmm5, xmm5  // alpha 255
    pslld      xmm5, 24  // 0xff000000

 convertloop:
    movdqu     xmm0, [eax]  // read 16 pixels src_sobelx
    movdqu     xmm1, [eax + esi]  // read 16 pixels src_sobely
    lea        eax, [eax + 16]
    paddusb    xmm0, xmm1  // sobel = sobelx + sobely
    movdqa     xmm2, xmm0  // GG
    punpcklbw  xmm2, xmm0  // First 8
    punpckhbw  xmm0, xmm0  // Next 8
    movdqa     xmm1, xmm2  // GGGG
    punpcklwd  xmm1, xmm2  // First 4
    punpckhwd  xmm2, xmm2  // Next 4
    por        xmm1, xmm5  // GGGA
    por        xmm2, xmm5
    movdqa     xmm3, xmm0  // GGGG
    punpcklwd  xmm3, xmm0  // Next 4
    punpckhwd  xmm0, xmm0  // Last 4
    por        xmm3, xmm5  // GGGA
    por        xmm0, xmm5
    movdqu     [edx], xmm1
    movdqu     [edx + 16], xmm2
    movdqu     [edx + 32], xmm3
    movdqu     [edx + 48], xmm0
    lea        edx, [edx + 64]
    sub        ecx, 16
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_SOBELROW_SSE2

#ifdef HAS_SOBELTOPLANEROW_SSE2
// Adds Sobel X and Sobel Y and stores Sobel into a plane.
__declspec(naked) void SobelToPlaneRow_SSE2(const uint8_t* src_sobelx,
                                            const uint8_t* src_sobely,
                                            uint8_t* dst_y,
                                            int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_sobelx
    mov        esi, [esp + 4 + 8]  // src_sobely
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    sub        esi, eax

 convertloop:
    movdqu     xmm0, [eax]  // read 16 pixels src_sobelx
    movdqu     xmm1, [eax + esi]  // read 16 pixels src_sobely
    lea        eax, [eax + 16]
    paddusb    xmm0, xmm1  // sobel = sobelx + sobely
    movdqu     [edx], xmm0
    lea        edx, [edx + 16]
    sub        ecx, 16
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_SOBELTOPLANEROW_SSE2

#ifdef HAS_SOBELXYROW_SSE2
// Mixes Sobel X, Sobel Y and Sobel into ARGB.
// A = 255
// R = Sobel X
// G = Sobel
// B = Sobel Y
__declspec(naked) void SobelXYRow_SSE2(const uint8_t* src_sobelx,
                                       const uint8_t* src_sobely,
                                       uint8_t* dst_argb,
                                       int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4]  // src_sobelx
    mov        esi, [esp + 4 + 8]  // src_sobely
    mov        edx, [esp + 4 + 12]  // dst_argb
    mov        ecx, [esp + 4 + 16]  // width
    sub        esi, eax
    pcmpeqb    xmm5, xmm5  // alpha 255

 convertloop:
    movdqu     xmm0, [eax]  // read 16 pixels src_sobelx
    movdqu     xmm1, [eax + esi]  // read 16 pixels src_sobely
    lea        eax, [eax + 16]
    movdqa     xmm2, xmm0
    paddusb    xmm2, xmm1  // sobel = sobelx + sobely
    movdqa     xmm3, xmm0  // XA
    punpcklbw  xmm3, xmm5
    punpckhbw  xmm0, xmm5
    movdqa     xmm4, xmm1  // YS
    punpcklbw  xmm4, xmm2
    punpckhbw  xmm1, xmm2
    movdqa     xmm6, xmm4  // YSXA
    punpcklwd  xmm6, xmm3  // First 4
    punpckhwd  xmm4, xmm3  // Next 4
    movdqa     xmm7, xmm1  // YSXA
    punpcklwd  xmm7, xmm0  // Next 4
    punpckhwd  xmm1, xmm0  // Last 4
    movdqu     [edx], xmm6
    movdqu     [edx + 16], xmm4
    movdqu     [edx + 32], xmm7
    movdqu     [edx + 48], xmm1
    lea        edx, [edx + 64]
    sub        ecx, 16
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_SOBELXYROW_SSE2

#ifdef HAS_CUMULATIVESUMTOAVERAGEROW_SSE2
// Consider float CumulativeSum.
// Consider calling CumulativeSum one row at time as needed.
// Consider circular CumulativeSum buffer of radius * 2 + 1 height.
// Convert cumulative sum for an area to an average for 1 pixel.
// topleft is pointer to top left of CumulativeSum buffer for area.
// botleft is pointer to bottom left of CumulativeSum buffer.
// width is offset from left to right of area in CumulativeSum buffer measured
//   in number of ints.
// area is the number of pixels in the area being averaged.
// dst points to pixel to store result to.
// count is number of averaged pixels to produce.
// Does 4 pixels at a time.
// This function requires alignment on accumulation buffer pointers.
void CumulativeSumToAverageRow_SSE2(const int32_t* topleft,
                                    const int32_t* botleft,
                                    int width,
                                    int area,
                                    uint8_t* dst,
                                    int count) {
  __asm {
    mov        eax, topleft  // eax topleft
    mov        esi, botleft  // esi botleft
    mov        edx, width
    movd       xmm5, area
    mov        edi, dst
    mov        ecx, count
    cvtdq2ps   xmm5, xmm5
    rcpss      xmm4, xmm5  // 1.0f / area
    pshufd     xmm4, xmm4, 0
    sub        ecx, 4
    jl         l4b

    cmp        area, 128  // 128 pixels will not overflow 15 bits.
    ja         l4

    pshufd     xmm5, xmm5, 0  // area
    pcmpeqb    xmm6, xmm6  // constant of 65536.0 - 1 = 65535.0
    psrld      xmm6, 16
    cvtdq2ps   xmm6, xmm6
    addps      xmm5, xmm6  // (65536.0 + area - 1)
    mulps      xmm5, xmm4  // (65536.0 + area - 1) * 1 / area
    cvtps2dq   xmm5, xmm5  // 0.16 fixed point
    packssdw   xmm5, xmm5  // 16 bit shorts

        // 4 pixel loop small blocks.
  s4:
        // top left
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]

    // - top right
    psubd      xmm0, [eax + edx * 4]
    psubd      xmm1, [eax + edx * 4 + 16]
    psubd      xmm2, [eax + edx * 4 + 32]
    psubd      xmm3, [eax + edx * 4 + 48]
    lea        eax, [eax + 64]

    // - bottom left
    psubd      xmm0, [esi]
    psubd      xmm1, [esi + 16]
    psubd      xmm2, [esi + 32]
    psubd      xmm3, [esi + 48]

    // + bottom right
    paddd      xmm0, [esi + edx * 4]
    paddd      xmm1, [esi + edx * 4 + 16]
    paddd      xmm2, [esi + edx * 4 + 32]
    paddd      xmm3, [esi + edx * 4 + 48]
    lea        esi, [esi + 64]

    packssdw   xmm0, xmm1  // pack 4 pixels into 2 registers
    packssdw   xmm2, xmm3

    pmulhuw    xmm0, xmm5
    pmulhuw    xmm2, xmm5

    packuswb   xmm0, xmm2
    movdqu     [edi], xmm0
    lea        edi, [edi + 16]
    sub        ecx, 4
    jge        s4

    jmp        l4b

            // 4 pixel loop
  l4:
        // top left
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    movdqu     xmm2, [eax + 32]
    movdqu     xmm3, [eax + 48]

    // - top right
    psubd      xmm0, [eax + edx * 4]
    psubd      xmm1, [eax + edx * 4 + 16]
    psubd      xmm2, [eax + edx * 4 + 32]
    psubd      xmm3, [eax + edx * 4 + 48]
    lea        eax, [eax + 64]

    // - bottom left
    psubd      xmm0, [esi]
    psubd      xmm1, [esi + 16]
    psubd      xmm2, [esi + 32]
    psubd      xmm3, [esi + 48]

    // + bottom right
    paddd      xmm0, [esi + edx * 4]
    paddd      xmm1, [esi + edx * 4 + 16]
    paddd      xmm2, [esi + edx * 4 + 32]
    paddd      xmm3, [esi + edx * 4 + 48]
    lea        esi, [esi + 64]

    cvtdq2ps   xmm0, xmm0  // Average = Sum * 1 / Area
    cvtdq2ps   xmm1, xmm1
    mulps      xmm0, xmm4
    mulps      xmm1, xmm4
    cvtdq2ps   xmm2, xmm2
    cvtdq2ps   xmm3, xmm3
    mulps      xmm2, xmm4
    mulps      xmm3, xmm4
    cvtps2dq   xmm0, xmm0
    cvtps2dq   xmm1, xmm1
    cvtps2dq   xmm2, xmm2
    cvtps2dq   xmm3, xmm3
    packssdw   xmm0, xmm1
    packssdw   xmm2, xmm3
    packuswb   xmm0, xmm2
    movdqu     [edi], xmm0
    lea        edi, [edi + 16]
    sub        ecx, 4
    jge        l4

  l4b:
    add        ecx, 4 - 1
    jl         l1b

            // 1 pixel loop
  l1:
    movdqu     xmm0, [eax]
    psubd      xmm0, [eax + edx * 4]
    lea        eax, [eax + 16]
    psubd      xmm0, [esi]
    paddd      xmm0, [esi + edx * 4]
    lea        esi, [esi + 16]
    cvtdq2ps   xmm0, xmm0
    mulps      xmm0, xmm4
    cvtps2dq   xmm0, xmm0
    packssdw   xmm0, xmm0
    packuswb   xmm0, xmm0
    movd       dword ptr [edi], xmm0
    lea        edi, [edi + 4]
    sub        ecx, 1
    jge        l1
  l1b:
  }
}
#endif  // HAS_CUMULATIVESUMTOAVERAGEROW_SSE2

#ifdef HAS_COMPUTECUMULATIVESUMROW_SSE2
// Creates a table of cumulative sums where each value is a sum of all values
// above and to the left of the value.
void ComputeCumulativeSumRow_SSE2(const uint8_t* row,
                                  int32_t* cumsum,
                                  const int32_t* previous_cumsum,
                                  int width) {
  __asm {
    mov        eax, row
    mov        edx, cumsum
    mov        esi, previous_cumsum
    mov        ecx, width
    pxor       xmm0, xmm0
    pxor       xmm1, xmm1

    sub        ecx, 4
    jl         l4b
    test       edx, 15
    jne        l4b

        // 4 pixel loop
  l4:
    movdqu     xmm2, [eax]  // 4 argb pixels 16 bytes.
    lea        eax, [eax + 16]
    movdqa     xmm4, xmm2

    punpcklbw  xmm2, xmm1
    movdqa     xmm3, xmm2
    punpcklwd  xmm2, xmm1
    punpckhwd  xmm3, xmm1

    punpckhbw  xmm4, xmm1
    movdqa     xmm5, xmm4
    punpcklwd  xmm4, xmm1
    punpckhwd  xmm5, xmm1

    paddd      xmm0, xmm2
    movdqu     xmm2, [esi]  // previous row above.
    paddd      xmm2, xmm0

    paddd      xmm0, xmm3
    movdqu     xmm3, [esi + 16]
    paddd      xmm3, xmm0

    paddd      xmm0, xmm4
    movdqu     xmm4, [esi + 32]
    paddd      xmm4, xmm0

    paddd      xmm0, xmm5
    movdqu     xmm5, [esi + 48]
    lea        esi, [esi + 64]
    paddd      xmm5, xmm0

    movdqu     [edx], xmm2
    movdqu     [edx + 16], xmm3
    movdqu     [edx + 32], xmm4
    movdqu     [edx + 48], xmm5

    lea        edx, [edx + 64]
    sub        ecx, 4
    jge        l4

  l4b:
    add        ecx, 4 - 1
    jl         l1b

            // 1 pixel loop
  l1:
    movd       xmm2, dword ptr [eax]  // 1 argb pixel
    lea        eax, [eax + 4]
    punpcklbw  xmm2, xmm1
    punpcklwd  xmm2, xmm1
    paddd      xmm0, xmm2
    movdqu     xmm2, [esi]
    lea        esi, [esi + 16]
    paddd      xmm2, xmm0
    movdqu     [edx], xmm2
    lea        edx, [edx + 16]
    sub        ecx, 1
    jge        l1

 l1b:
  }
}
#endif  // HAS_COMPUTECUMULATIVESUMROW_SSE2

#ifdef HAS_ARGBAFFINEROW_SSE2
// Copy ARGB pixels from source image with slope to a row of destination.
__declspec(naked) LIBYUV_API void ARGBAffineRow_SSE2(const uint8_t* src_argb,
                                                     int src_argb_stride,
                                                     uint8_t* dst_argb,
                                                     const float* uv_dudv,
                                                     int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 12]  // src_argb
    mov        esi, [esp + 16]  // stride
    mov        edx, [esp + 20]  // dst_argb
    mov        ecx, [esp + 24]  // pointer to uv_dudv
    movq       xmm2, qword ptr [ecx]  // uv
    movq       xmm7, qword ptr [ecx + 8]  // dudv
    mov        ecx, [esp + 28]  // width
    shl        esi, 16  // 4, stride
    add        esi, 4
    movd       xmm5, esi
    sub        ecx, 4
    jl         l4b

        // setup for 4 pixel loop
    pshufd     xmm7, xmm7, 0x44  // dup dudv
    pshufd     xmm5, xmm5, 0  // dup 4, stride
    movdqa     xmm0, xmm2  // x0, y0, x1, y1
    addps      xmm0, xmm7
    movlhps    xmm2, xmm0
    movdqa     xmm4, xmm7
    addps      xmm4, xmm4  // dudv *= 2
    movdqa     xmm3, xmm2  // x2, y2, x3, y3
    addps      xmm3, xmm4
    addps      xmm4, xmm4  // dudv *= 4

        // 4 pixel loop
  l4:
    cvttps2dq  xmm0, xmm2  // x, y float to int first 2
    cvttps2dq  xmm1, xmm3  // x, y float to int next 2
    packssdw   xmm0, xmm1  // x, y as 8 shorts
    pmaddwd    xmm0, xmm5  // offsets = x * 4 + y * stride.
    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // shift right
    movd       edi, xmm0
    pshufd     xmm0, xmm0, 0x39  // shift right
    movd       xmm1, [eax + esi]  // read pixel 0
    movd       xmm6, [eax + edi]  // read pixel 1
    punpckldq  xmm1, xmm6  // combine pixel 0 and 1
    addps      xmm2, xmm4  // x, y += dx, dy first 2
    movq       qword ptr [edx], xmm1
    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // shift right
    movd       edi, xmm0
    movd       xmm6, [eax + esi]  // read pixel 2
    movd       xmm0, [eax + edi]  // read pixel 3
    punpckldq  xmm6, xmm0  // combine pixel 2 and 3
    addps      xmm3, xmm4  // x, y += dx, dy next 2
    movq       qword ptr 8[edx], xmm6
    lea        edx, [edx + 16]
    sub        ecx, 4
    jge        l4

  l4b:
    add        ecx, 4 - 1
    jl         l1b

            // 1 pixel loop
  l1:
    cvttps2dq  xmm0, xmm2  // x, y float to int
    packssdw   xmm0, xmm0  // x, y as shorts
    pmaddwd    xmm0, xmm5  // offset = x * 4 + y * stride
    addps      xmm2, xmm7  // x, y += dx, dy
    movd       esi, xmm0
    movd       xmm0, [eax + esi]  // copy a pixel
    movd       [edx], xmm0
    lea        edx, [edx + 4]
    sub        ecx, 1
    jge        l1
  l1b:
    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBAFFINEROW_SSE2

#ifdef HAS_INTERPOLATEROW_AVX2
// Bilinear filter 32x2 -> 32x1
__declspec(naked) void InterpolateRow_AVX2(uint8_t* dst_ptr,
                                           const uint8_t* src_ptr,
                                           ptrdiff_t src_stride,
                                           int dst_width,
                                           int source_y_fraction) {
  __asm {
    push       esi
    push       edi
    mov        edi, [esp + 8 + 4]  // dst_ptr
    mov        esi, [esp + 8 + 8]  // src_ptr
    mov        edx, [esp + 8 + 12]  // src_stride
    mov        ecx, [esp + 8 + 16]  // dst_width
    mov        eax, [esp + 8 + 20]  // source_y_fraction (0..255)
    // Dispatch to specialized filters if applicable.
    cmp        eax, 0
    je         xloop100  // 0 / 256.  Blend 100 / 0.
    sub        edi, esi
    cmp        eax, 128
    je         xloop50  // 128 /256 is 0.50.  Blend 50 / 50.

    vmovd      xmm0, eax  // high fraction 0..255
    neg        eax
    add        eax, 256
    vmovd      xmm5, eax  // low fraction 256..1
    vpunpcklbw xmm5, xmm5, xmm0
    vpunpcklwd xmm5, xmm5, xmm5
    vbroadcastss ymm5, xmm5

    mov        eax, 0x80808080  // 128b for bias and rounding.
    vmovd      xmm4, eax
    vbroadcastss ymm4, xmm4

  xloop:
    vmovdqu    ymm0, [esi]
    vmovdqu    ymm2, [esi + edx]
    vpunpckhbw ymm1, ymm0, ymm2  // mutates
    vpunpcklbw ymm0, ymm0, ymm2
    vpsubb     ymm1, ymm1, ymm4  // bias to signed image
    vpsubb     ymm0, ymm0, ymm4
    vpmaddubsw ymm1, ymm5, ymm1
    vpmaddubsw ymm0, ymm5, ymm0
    vpaddw     ymm1, ymm1, ymm4  // unbias and round
    vpaddw     ymm0, ymm0, ymm4
    vpsrlw     ymm1, ymm1, 8
    vpsrlw     ymm0, ymm0, 8
    vpackuswb  ymm0, ymm0, ymm1            // unmutates
    vmovdqu    [esi + edi], ymm0
    lea        esi, [esi + 32]
    sub        ecx, 32
    jg         xloop
    jmp        xloop99

        // Blend 50 / 50.
 xloop50:
   vmovdqu    ymm0, [esi]
   vpavgb     ymm0, ymm0, [esi + edx]
   vmovdqu    [esi + edi], ymm0
   lea        esi, [esi + 32]
   sub        ecx, 32
   jg         xloop50
   jmp        xloop99

        // Blend 100 / 0 - Copy row unchanged.
 xloop100:
   rep movsb

  xloop99:
    pop        edi
    pop        esi
    vzeroupper
    ret
  }
}
#endif  // HAS_INTERPOLATEROW_AVX2

// Bilinear filter 16x2 -> 16x1
// TODO(fbarchard): Consider allowing 256 using memcpy.
__declspec(naked) void InterpolateRow_SSSE3(uint8_t* dst_ptr,
                                            const uint8_t* src_ptr,
                                            ptrdiff_t src_stride,
                                            int dst_width,
                                            int source_y_fraction) {
  __asm {
    push       esi
    push       edi

    mov        edi, [esp + 8 + 4]  // dst_ptr
    mov        esi, [esp + 8 + 8]  // src_ptr
    mov        edx, [esp + 8 + 12]  // src_stride
    mov        ecx, [esp + 8 + 16]  // dst_width
    mov        eax, [esp + 8 + 20]  // source_y_fraction (0..255)
    sub        edi, esi
        // Dispatch to specialized filters if applicable.
    cmp        eax, 0
    je         xloop100  // 0 /256.  Blend 100 / 0.
    cmp        eax, 128
    je         xloop50  // 128 / 256 is 0.50.  Blend 50 / 50.

    movd       xmm0, eax  // high fraction 0..255
    neg        eax
    add        eax, 256
    movd       xmm5, eax  // low fraction 255..1
    punpcklbw  xmm5, xmm0
    punpcklwd  xmm5, xmm5
    pshufd     xmm5, xmm5, 0
    mov        eax, 0x80808080  // 128 for biasing image to signed.
    movd       xmm4, eax
    pshufd     xmm4, xmm4, 0x00

  xloop:
    movdqu     xmm0, [esi]
    movdqu     xmm2, [esi + edx]
    movdqu     xmm1, xmm0
    punpcklbw  xmm0, xmm2
    punpckhbw  xmm1, xmm2
    psubb      xmm0, xmm4            // bias image by -128
    psubb      xmm1, xmm4
    movdqa     xmm2, xmm5
    movdqa     xmm3, xmm5
    pmaddubsw  xmm2, xmm0
    pmaddubsw  xmm3, xmm1
    paddw      xmm2, xmm4
    paddw      xmm3, xmm4
    psrlw      xmm2, 8
    psrlw      xmm3, 8
    packuswb   xmm2, xmm3
    movdqu     [esi + edi], xmm2
    lea        esi, [esi + 16]
    sub        ecx, 16
    jg         xloop
    jmp        xloop99

        // Blend 50 / 50.
  xloop50:
    movdqu     xmm0, [esi]
    movdqu     xmm1, [esi + edx]
    pavgb      xmm0, xmm1
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    sub        ecx, 16
    jg         xloop50
    jmp        xloop99

        // Blend 100 / 0 - Copy row unchanged.
  xloop100:
    movdqu     xmm0, [esi]
    movdqu     [esi + edi], xmm0
    lea        esi, [esi + 16]
    sub        ecx, 16
    jg         xloop100

  xloop99:
    pop        edi
    pop        esi
    ret
  }
}

// For BGRAToARGB, ABGRToARGB, RGBAToARGB, and ARGBToRGBA.
__declspec(naked) void ARGBShuffleRow_SSSE3(const uint8_t* src_argb,
                                            uint8_t* dst_argb,
                                            const uint8_t* shuffler,
                                            int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_argb
    mov        ecx, [esp + 12]  // shuffler
    movdqu     xmm5, [ecx]
    mov        ecx, [esp + 16]  // width

  wloop:
    movdqu     xmm0, [eax]
    movdqu     xmm1, [eax + 16]
    lea        eax, [eax + 32]
    pshufb     xmm0, xmm5
    pshufb     xmm1, xmm5
    movdqu     [edx], xmm0
    movdqu     [edx + 16], xmm1
    lea        edx, [edx + 32]
    sub        ecx, 8
    jg         wloop
    ret
  }
}

#ifdef HAS_ARGBSHUFFLEROW_AVX2
__declspec(naked) void ARGBShuffleRow_AVX2(const uint8_t* src_argb,
                                           uint8_t* dst_argb,
                                           const uint8_t* shuffler,
                                           int width) {
  __asm {
    mov        eax, [esp + 4]  // src_argb
    mov        edx, [esp + 8]  // dst_argb
    mov        ecx, [esp + 12]  // shuffler
    vbroadcastf128 ymm5, [ecx]  // same shuffle in high as low.
    mov        ecx, [esp + 16]  // width

  wloop:
    vmovdqu    ymm0, [eax]
    vmovdqu    ymm1, [eax + 32]
    lea        eax, [eax + 64]
    vpshufb    ymm0, ymm0, ymm5
    vpshufb    ymm1, ymm1, ymm5
    vmovdqu    [edx], ymm0
    vmovdqu    [edx + 32], ymm1
    lea        edx, [edx + 64]
    sub        ecx, 16
    jg         wloop

    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBSHUFFLEROW_AVX2

// YUY2 - Macro-pixel = 2 image pixels
// Y0U0Y1V0....Y2U2Y3V2...Y4U4Y5V4....

// UYVY - Macro-pixel = 2 image pixels
// U0Y0V0Y1

__declspec(naked) void I422ToYUY2Row_SSE2(const uint8_t* src_y,
                                          const uint8_t* src_u,
                                          const uint8_t* src_v,
                                          uint8_t* dst_frame,
                                          int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_y
    mov        esi, [esp + 8 + 8]  // src_u
    mov        edx, [esp + 8 + 12]  // src_v
    mov        edi, [esp + 8 + 16]  // dst_frame
    mov        ecx, [esp + 8 + 20]  // width
    sub        edx, esi

  convertloop:
    movq       xmm2, qword ptr [esi]  // U
    movq       xmm3, qword ptr [esi + edx]  // V
    lea        esi, [esi + 8]
    punpcklbw  xmm2, xmm3  // UV
    movdqu     xmm0, [eax]  // Y
    lea        eax, [eax + 16]
    movdqa     xmm1, xmm0
    punpcklbw  xmm0, xmm2  // YUYV
    punpckhbw  xmm1, xmm2
    movdqu     [edi], xmm0
    movdqu     [edi + 16], xmm1
    lea        edi, [edi + 32]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

__declspec(naked) void I422ToUYVYRow_SSE2(const uint8_t* src_y,
                                          const uint8_t* src_u,
                                          const uint8_t* src_v,
                                          uint8_t* dst_frame,
                                          int width) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4]  // src_y
    mov        esi, [esp + 8 + 8]  // src_u
    mov        edx, [esp + 8 + 12]  // src_v
    mov        edi, [esp + 8 + 16]  // dst_frame
    mov        ecx, [esp + 8 + 20]  // width
    sub        edx, esi

  convertloop:
    movq       xmm2, qword ptr [esi]  // U
    movq       xmm3, qword ptr [esi + edx]  // V
    lea        esi, [esi + 8]
    punpcklbw  xmm2, xmm3  // UV
    movdqu     xmm0, [eax]  // Y
    movdqa     xmm1, xmm2
    lea        eax, [eax + 16]
    punpcklbw  xmm1, xmm0  // UYVY
    punpckhbw  xmm2, xmm0
    movdqu     [edi], xmm1
    movdqu     [edi + 16], xmm2
    lea        edi, [edi + 32]
    sub        ecx, 16
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}

#ifdef HAS_ARGBPOLYNOMIALROW_SSE2
__declspec(naked) void ARGBPolynomialRow_SSE2(const uint8_t* src_argb,
                                              uint8_t* dst_argb,
                                              const float* poly,
                                              int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4] /* src_argb */
    mov        edx, [esp + 4 + 8] /* dst_argb */
    mov        esi, [esp + 4 + 12] /* poly */
    mov        ecx, [esp + 4 + 16] /* width */
    pxor       xmm3, xmm3  // 0 constant for zero extending bytes to ints.

        // 2 pixel loop.
 convertloop:
        //    pmovzxbd  xmm0, dword ptr [eax]  // BGRA pixel
        //    pmovzxbd  xmm4, dword ptr [eax + 4]  // BGRA pixel
    movq       xmm0, qword ptr [eax]  // BGRABGRA
    lea        eax, [eax + 8]
    punpcklbw  xmm0, xmm3
    movdqa     xmm4, xmm0
    punpcklwd  xmm0, xmm3  // pixel 0
    punpckhwd  xmm4, xmm3  // pixel 1
    cvtdq2ps   xmm0, xmm0  // 4 floats
    cvtdq2ps   xmm4, xmm4
    movdqa     xmm1, xmm0  // X
    movdqa     xmm5, xmm4
    mulps      xmm0, [esi + 16]  // C1 * X
    mulps      xmm4, [esi + 16]
    addps      xmm0, [esi]  // result = C0 + C1 * X
    addps      xmm4, [esi]
    movdqa     xmm2, xmm1
    movdqa     xmm6, xmm5
    mulps      xmm2, xmm1  // X * X
    mulps      xmm6, xmm5
    mulps      xmm1, xmm2  // X * X * X
    mulps      xmm5, xmm6
    mulps      xmm2, [esi + 32]  // C2 * X * X
    mulps      xmm6, [esi + 32]
    mulps      xmm1, [esi + 48]  // C3 * X * X * X
    mulps      xmm5, [esi + 48]
    addps      xmm0, xmm2  // result += C2 * X * X
    addps      xmm4, xmm6
    addps      xmm0, xmm1  // result += C3 * X * X * X
    addps      xmm4, xmm5
    cvttps2dq  xmm0, xmm0
    cvttps2dq  xmm4, xmm4
    packuswb   xmm0, xmm4
    packuswb   xmm0, xmm0
    movq       qword ptr [edx], xmm0
    lea        edx, [edx + 8]
    sub        ecx, 2
    jg         convertloop
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBPOLYNOMIALROW_SSE2

#ifdef HAS_ARGBPOLYNOMIALROW_AVX2
__declspec(naked) void ARGBPolynomialRow_AVX2(const uint8_t* src_argb,
                                              uint8_t* dst_argb,
                                              const float* poly,
                                              int width) {
  __asm {
    mov        eax, [esp + 4] /* src_argb */
    mov        edx, [esp + 8] /* dst_argb */
    mov        ecx, [esp + 12] /* poly */
    vbroadcastf128 ymm4, [ecx]  // C0
    vbroadcastf128 ymm5, [ecx + 16]  // C1
    vbroadcastf128 ymm6, [ecx + 32]  // C2
    vbroadcastf128 ymm7, [ecx + 48]  // C3
    mov        ecx, [esp + 16] /* width */

    // 2 pixel loop.
 convertloop:
    vpmovzxbd   ymm0, qword ptr [eax]  // 2 BGRA pixels
    lea         eax, [eax + 8]
    vcvtdq2ps   ymm0, ymm0  // X 8 floats
    vmulps      ymm2, ymm0, ymm0  // X * X
    vmulps      ymm3, ymm0, ymm7  // C3 * X
    vfmadd132ps ymm0, ymm4, ymm5  // result = C0 + C1 * X
    vfmadd231ps ymm0, ymm2, ymm6  // result += C2 * X * X
    vfmadd231ps ymm0, ymm2, ymm3  // result += C3 * X * X * X
    vcvttps2dq  ymm0, ymm0
    vpackusdw   ymm0, ymm0, ymm0  // b0g0r0a0_00000000_b0g0r0a0_00000000
    vpermq      ymm0, ymm0, 0xd8  // b0g0r0a0_b0g0r0a0_00000000_00000000
    vpackuswb   xmm0, xmm0, xmm0  // bgrabgra_00000000_00000000_00000000
    vmovq       qword ptr [edx], xmm0
    lea         edx, [edx + 8]
    sub         ecx, 2
    jg          convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_ARGBPOLYNOMIALROW_AVX2

#ifdef HAS_HALFFLOATROW_SSE2
static float kExpBias = 1.9259299444e-34f;
__declspec(naked) void HalfFloatRow_SSE2(const uint16_t* src,
                                         uint16_t* dst,
                                         float scale,
                                         int width) {
  __asm {
    mov        eax, [esp + 4] /* src */
    mov        edx, [esp + 8] /* dst */
    movd       xmm4, dword ptr [esp + 12] /* scale */
    mov        ecx, [esp + 16] /* width */
    mulss      xmm4, kExpBias
    pshufd     xmm4, xmm4, 0
    pxor       xmm5, xmm5
    sub        edx, eax

        // 8 pixel loop.
 convertloop:
    movdqu      xmm2, xmmword ptr [eax]  // 8 shorts
    add         eax, 16
    movdqa      xmm3, xmm2
    punpcklwd   xmm2, xmm5
    cvtdq2ps    xmm2, xmm2  // convert 8 ints to floats
    punpckhwd   xmm3, xmm5
    cvtdq2ps    xmm3, xmm3
    mulps       xmm2, xmm4
    mulps       xmm3, xmm4
    psrld       xmm2, 13
    psrld       xmm3, 13
    packssdw    xmm2, xmm3
    movdqu      [eax + edx - 16], xmm2
    sub         ecx, 8
    jg          convertloop
    ret
  }
}
#endif  // HAS_HALFFLOATROW_SSE2

#ifdef HAS_HALFFLOATROW_AVX2
__declspec(naked) void HalfFloatRow_AVX2(const uint16_t* src,
                                         uint16_t* dst,
                                         float scale,
                                         int width) {
  __asm {
    mov        eax, [esp + 4] /* src */
    mov        edx, [esp + 8] /* dst */
    movd       xmm4, dword ptr [esp + 12] /* scale */
    mov        ecx, [esp + 16] /* width */

    vmulss     xmm4, xmm4, kExpBias
    vbroadcastss ymm4, xmm4
    vpxor      ymm5, ymm5, ymm5
    sub        edx, eax

        // 16 pixel loop.
 convertloop:
    vmovdqu     ymm2, [eax]  // 16 shorts
    add         eax, 32
    vpunpckhwd  ymm3, ymm2, ymm5  // convert 16 shorts to 16 ints
    vpunpcklwd  ymm2, ymm2, ymm5
    vcvtdq2ps   ymm3, ymm3  // convert 16 ints to floats
    vcvtdq2ps   ymm2, ymm2
    vmulps      ymm3, ymm3, ymm4  // scale to adjust exponent for 5 bit range.
    vmulps      ymm2, ymm2, ymm4
    vpsrld      ymm3, ymm3, 13  // float convert to 8 half floats truncate
    vpsrld      ymm2, ymm2, 13
    vpackssdw   ymm2, ymm2, ymm3
    vmovdqu     [eax + edx - 32], ymm2
    sub         ecx, 16
    jg          convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_HALFFLOATROW_AVX2

#ifdef HAS_HALFFLOATROW_F16C
__declspec(naked) void HalfFloatRow_F16C(const uint16_t* src,
                                         uint16_t* dst,
                                         float scale,
                                         int width) {
  __asm {
    mov        eax, [esp + 4] /* src */
    mov        edx, [esp + 8] /* dst */
    vbroadcastss ymm4, [esp + 12] /* scale */
    mov        ecx, [esp + 16] /* width */
    sub        edx, eax

        // 16 pixel loop.
 convertloop:
    vpmovzxwd   ymm2, xmmword ptr [eax]  // 8 shorts -> 8 ints
    vpmovzxwd   ymm3, xmmword ptr [eax + 16]  // 8 more shorts
    add         eax, 32
    vcvtdq2ps   ymm2, ymm2  // convert 8 ints to floats
    vcvtdq2ps   ymm3, ymm3
    vmulps      ymm2, ymm2, ymm4  // scale to normalized range 0 to 1
    vmulps      ymm3, ymm3, ymm4
    vcvtps2ph   xmm2, ymm2, 3  // float convert to 8 half floats truncate
    vcvtps2ph   xmm3, ymm3, 3
    vmovdqu     [eax + edx + 32], xmm2
    vmovdqu     [eax + edx + 32 + 16], xmm3
    sub         ecx, 16
    jg          convertloop
    vzeroupper
    ret
  }
}
#endif  // HAS_HALFFLOATROW_F16C

#ifdef HAS_ARGBCOLORTABLEROW_X86
// Tranform ARGB pixels with color table.
__declspec(naked) void ARGBColorTableRow_X86(uint8_t* dst_argb,
                                             const uint8_t* table_argb,
                                             int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4] /* dst_argb */
    mov        esi, [esp + 4 + 8] /* table_argb */
    mov        ecx, [esp + 4 + 12] /* width */

    // 1 pixel loop.
  convertloop:
    movzx      edx, byte ptr [eax]
    lea        eax, [eax + 4]
    movzx      edx, byte ptr [esi + edx * 4]
    mov        byte ptr [eax - 4], dl
    movzx      edx, byte ptr [eax - 4 + 1]
    movzx      edx, byte ptr [esi + edx * 4 + 1]
    mov        byte ptr [eax - 4 + 1], dl
    movzx      edx, byte ptr [eax - 4 + 2]
    movzx      edx, byte ptr [esi + edx * 4 + 2]
    mov        byte ptr [eax - 4 + 2], dl
    movzx      edx, byte ptr [eax - 4 + 3]
    movzx      edx, byte ptr [esi + edx * 4 + 3]
    mov        byte ptr [eax - 4 + 3], dl
    dec        ecx
    jg         convertloop
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBCOLORTABLEROW_X86

#ifdef HAS_RGBCOLORTABLEROW_X86
// Tranform RGB pixels with color table.
__declspec(naked) void RGBColorTableRow_X86(uint8_t* dst_argb,
                                            const uint8_t* table_argb,
                                            int width) {
  __asm {
    push       esi
    mov        eax, [esp + 4 + 4] /* dst_argb */
    mov        esi, [esp + 4 + 8] /* table_argb */
    mov        ecx, [esp + 4 + 12] /* width */

    // 1 pixel loop.
  convertloop:
    movzx      edx, byte ptr [eax]
    lea        eax, [eax + 4]
    movzx      edx, byte ptr [esi + edx * 4]
    mov        byte ptr [eax - 4], dl
    movzx      edx, byte ptr [eax - 4 + 1]
    movzx      edx, byte ptr [esi + edx * 4 + 1]
    mov        byte ptr [eax - 4 + 1], dl
    movzx      edx, byte ptr [eax - 4 + 2]
    movzx      edx, byte ptr [esi + edx * 4 + 2]
    mov        byte ptr [eax - 4 + 2], dl
    dec        ecx
    jg         convertloop

    pop        esi
    ret
  }
}
#endif  // HAS_RGBCOLORTABLEROW_X86

#ifdef HAS_ARGBLUMACOLORTABLEROW_SSSE3
// Tranform RGB pixels with luma table.
__declspec(naked) void ARGBLumaColorTableRow_SSSE3(const uint8_t* src_argb,
                                                   uint8_t* dst_argb,
                                                   int width,
                                                   const uint8_t* luma,
                                                   uint32_t lumacoeff) {
  __asm {
    push       esi
    push       edi
    mov        eax, [esp + 8 + 4] /* src_argb */
    mov        edi, [esp + 8 + 8] /* dst_argb */
    mov        ecx, [esp + 8 + 12] /* width */
    movd       xmm2, dword ptr [esp + 8 + 16]  // luma table
    movd       xmm3, dword ptr [esp + 8 + 20]  // lumacoeff
    pshufd     xmm2, xmm2, 0
    pshufd     xmm3, xmm3, 0
    pcmpeqb    xmm4, xmm4  // generate mask 0xff00ff00
    psllw      xmm4, 8
    pxor       xmm5, xmm5

        // 4 pixel loop.
  convertloop:
    movdqu     xmm0, xmmword ptr [eax]  // generate luma ptr
    pmaddubsw  xmm0, xmm3
    phaddw     xmm0, xmm0
    pand       xmm0, xmm4  // mask out low bits
    punpcklwd  xmm0, xmm5
    paddd      xmm0, xmm2  // add table base
    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // 00111001 to rotate right 32

    movzx      edx, byte ptr [eax]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi], dl
    movzx      edx, byte ptr [eax + 1]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 1], dl
    movzx      edx, byte ptr [eax + 2]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 2], dl
    movzx      edx, byte ptr [eax + 3]  // copy alpha.
    mov        byte ptr [edi + 3], dl

    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // 00111001 to rotate right 32

    movzx      edx, byte ptr [eax + 4]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 4], dl
    movzx      edx, byte ptr [eax + 5]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 5], dl
    movzx      edx, byte ptr [eax + 6]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 6], dl
    movzx      edx, byte ptr [eax + 7]  // copy alpha.
    mov        byte ptr [edi + 7], dl

    movd       esi, xmm0
    pshufd     xmm0, xmm0, 0x39  // 00111001 to rotate right 32

    movzx      edx, byte ptr [eax + 8]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 8], dl
    movzx      edx, byte ptr [eax + 9]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 9], dl
    movzx      edx, byte ptr [eax + 10]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 10], dl
    movzx      edx, byte ptr [eax + 11]  // copy alpha.
    mov        byte ptr [edi + 11], dl

    movd       esi, xmm0

    movzx      edx, byte ptr [eax + 12]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 12], dl
    movzx      edx, byte ptr [eax + 13]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 13], dl
    movzx      edx, byte ptr [eax + 14]
    movzx      edx, byte ptr [esi + edx]
    mov        byte ptr [edi + 14], dl
    movzx      edx, byte ptr [eax + 15]  // copy alpha.
    mov        byte ptr [edi + 15], dl

    lea        eax, [eax + 16]
    lea        edi, [edi + 16]
    sub        ecx, 4
    jg         convertloop

    pop        edi
    pop        esi
    ret
  }
}
#endif  // HAS_ARGBLUMACOLORTABLEROW_SSSE3

#endif  // defined(_M_X64)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_X86) && (defined(_M_IX86) || defined(_M_X64))
