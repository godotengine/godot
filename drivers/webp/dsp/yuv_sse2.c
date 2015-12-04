// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// YUV->RGB conversion functions
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./yuv.h"

#if defined(WEBP_USE_SSE2)

#include <emmintrin.h>
#include <string.h>   // for memcpy

typedef union {   // handy struct for converting SSE2 registers
  int32_t i32[4];
  uint8_t u8[16];
  __m128i m;
} VP8kCstSSE2;

#if defined(WEBP_YUV_USE_SSE2_TABLES)

#include "./yuv_tables_sse2.h"

WEBP_TSAN_IGNORE_FUNCTION void VP8YUVInitSSE2(void) {}

#else

static int done_sse2 = 0;
static VP8kCstSSE2 VP8kUtoRGBA[256], VP8kVtoRGBA[256], VP8kYtoRGBA[256];

WEBP_TSAN_IGNORE_FUNCTION void VP8YUVInitSSE2(void) {
  if (!done_sse2) {
    int i;
    for (i = 0; i < 256; ++i) {
      VP8kYtoRGBA[i].i32[0] =
        VP8kYtoRGBA[i].i32[1] =
        VP8kYtoRGBA[i].i32[2] = (i - 16) * kYScale + YUV_HALF2;
      VP8kYtoRGBA[i].i32[3] = 0xff << YUV_FIX2;

      VP8kUtoRGBA[i].i32[0] = 0;
      VP8kUtoRGBA[i].i32[1] = -kUToG * (i - 128);
      VP8kUtoRGBA[i].i32[2] =  kUToB * (i - 128);
      VP8kUtoRGBA[i].i32[3] = 0;

      VP8kVtoRGBA[i].i32[0] =  kVToR * (i - 128);
      VP8kVtoRGBA[i].i32[1] = -kVToG * (i - 128);
      VP8kVtoRGBA[i].i32[2] = 0;
      VP8kVtoRGBA[i].i32[3] = 0;
    }
    done_sse2 = 1;

#if 0   // code used to generate 'yuv_tables_sse2.h'
    printf("static const VP8kCstSSE2 VP8kYtoRGBA[256] = {\n");
    for (i = 0; i < 256; ++i) {
      printf("  {{0x%.8x, 0x%.8x, 0x%.8x, 0x%.8x}},\n",
             VP8kYtoRGBA[i].i32[0], VP8kYtoRGBA[i].i32[1],
             VP8kYtoRGBA[i].i32[2], VP8kYtoRGBA[i].i32[3]);
    }
    printf("};\n\n");
    printf("static const VP8kCstSSE2 VP8kUtoRGBA[256] = {\n");
    for (i = 0; i < 256; ++i) {
      printf("  {{0, 0x%.8x, 0x%.8x, 0}},\n",
             VP8kUtoRGBA[i].i32[1], VP8kUtoRGBA[i].i32[2]);
    }
    printf("};\n\n");
    printf("static VP8kCstSSE2 VP8kVtoRGBA[256] = {\n");
    for (i = 0; i < 256; ++i) {
      printf("  {{0x%.8x, 0x%.8x, 0, 0}},\n",
             VP8kVtoRGBA[i].i32[0], VP8kVtoRGBA[i].i32[1]);
    }
    printf("};\n\n");
#endif
  }
}

#endif  // WEBP_YUV_USE_SSE2_TABLES

//-----------------------------------------------------------------------------

static WEBP_INLINE __m128i LoadUVPart(int u, int v) {
  const __m128i u_part = _mm_loadu_si128(&VP8kUtoRGBA[u].m);
  const __m128i v_part = _mm_loadu_si128(&VP8kVtoRGBA[v].m);
  const __m128i uv_part = _mm_add_epi32(u_part, v_part);
  return uv_part;
}

static WEBP_INLINE __m128i GetRGBA32bWithUV(int y, const __m128i uv_part) {
  const __m128i y_part = _mm_loadu_si128(&VP8kYtoRGBA[y].m);
  const __m128i rgba1 = _mm_add_epi32(y_part, uv_part);
  const __m128i rgba2 = _mm_srai_epi32(rgba1, YUV_FIX2);
  return rgba2;
}

static WEBP_INLINE __m128i GetRGBA32b(int y, int u, int v) {
  const __m128i uv_part = LoadUVPart(u, v);
  return GetRGBA32bWithUV(y, uv_part);
}

static WEBP_INLINE void YuvToRgbSSE2(uint8_t y, uint8_t u, uint8_t v,
                                     uint8_t* const rgb) {
  const __m128i tmp0 = GetRGBA32b(y, u, v);
  const __m128i tmp1 = _mm_packs_epi32(tmp0, tmp0);
  const __m128i tmp2 = _mm_packus_epi16(tmp1, tmp1);
  // Note: we store 8 bytes at a time, not 3 bytes! -> memory stomp
  _mm_storel_epi64((__m128i*)rgb, tmp2);
}

static WEBP_INLINE void YuvToBgrSSE2(uint8_t y, uint8_t u, uint8_t v,
                                     uint8_t* const bgr) {
  const __m128i tmp0 = GetRGBA32b(y, u, v);
  const __m128i tmp1 = _mm_shuffle_epi32(tmp0, _MM_SHUFFLE(3, 0, 1, 2));
  const __m128i tmp2 = _mm_packs_epi32(tmp1, tmp1);
  const __m128i tmp3 = _mm_packus_epi16(tmp2, tmp2);
  // Note: we store 8 bytes at a time, not 3 bytes! -> memory stomp
  _mm_storel_epi64((__m128i*)bgr, tmp3);
}

//-----------------------------------------------------------------------------
// Convert spans of 32 pixels to various RGB formats for the fancy upsampler.

void VP8YuvToRgba32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                    uint8_t* dst) {
  int n;
  for (n = 0; n < 32; n += 4) {
    const __m128i tmp0_1 = GetRGBA32b(y[n + 0], u[n + 0], v[n + 0]);
    const __m128i tmp0_2 = GetRGBA32b(y[n + 1], u[n + 1], v[n + 1]);
    const __m128i tmp0_3 = GetRGBA32b(y[n + 2], u[n + 2], v[n + 2]);
    const __m128i tmp0_4 = GetRGBA32b(y[n + 3], u[n + 3], v[n + 3]);
    const __m128i tmp1_1 = _mm_packs_epi32(tmp0_1, tmp0_2);
    const __m128i tmp1_2 = _mm_packs_epi32(tmp0_3, tmp0_4);
    const __m128i tmp2 = _mm_packus_epi16(tmp1_1, tmp1_2);
    _mm_storeu_si128((__m128i*)dst, tmp2);
    dst += 4 * 4;
  }
}

void VP8YuvToBgra32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                    uint8_t* dst) {
  int n;
  for (n = 0; n < 32; n += 2) {
    const __m128i tmp0_1 = GetRGBA32b(y[n + 0], u[n + 0], v[n + 0]);
    const __m128i tmp0_2 = GetRGBA32b(y[n + 1], u[n + 1], v[n + 1]);
    const __m128i tmp1_1 = _mm_shuffle_epi32(tmp0_1, _MM_SHUFFLE(3, 0, 1, 2));
    const __m128i tmp1_2 = _mm_shuffle_epi32(tmp0_2, _MM_SHUFFLE(3, 0, 1, 2));
    const __m128i tmp2_1 = _mm_packs_epi32(tmp1_1, tmp1_2);
    const __m128i tmp3 = _mm_packus_epi16(tmp2_1, tmp2_1);
    _mm_storel_epi64((__m128i*)dst, tmp3);
    dst += 4 * 2;
  }
}

void VP8YuvToRgb32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                   uint8_t* dst) {
  int n;
  uint8_t tmp0[2 * 3 + 5 + 15];
  uint8_t* const tmp = (uint8_t*)((uintptr_t)(tmp0 + 15) & ~15);  // align
  for (n = 0; n < 30; ++n) {   // we directly stomp the *dst memory
    YuvToRgbSSE2(y[n], u[n], v[n], dst + n * 3);
  }
  // Last two pixels are special: we write in a tmp buffer before sending
  // to dst.
  YuvToRgbSSE2(y[n + 0], u[n + 0], v[n + 0], tmp + 0);
  YuvToRgbSSE2(y[n + 1], u[n + 1], v[n + 1], tmp + 3);
  memcpy(dst + n * 3, tmp, 2 * 3);
}

void VP8YuvToBgr32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                   uint8_t* dst) {
  int n;
  uint8_t tmp0[2 * 3 + 5 + 15];
  uint8_t* const tmp = (uint8_t*)((uintptr_t)(tmp0 + 15) & ~15);  // align
  for (n = 0; n < 30; ++n) {
    YuvToBgrSSE2(y[n], u[n], v[n], dst + n * 3);
  }
  YuvToBgrSSE2(y[n + 0], u[n + 0], v[n + 0], tmp + 0);
  YuvToBgrSSE2(y[n + 1], u[n + 1], v[n + 1], tmp + 3);
  memcpy(dst + n * 3, tmp, 2 * 3);
}

//-----------------------------------------------------------------------------
// Arbitrary-length row conversion functions

static void YuvToRgbaRow(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                         uint8_t* dst, int len) {
  int n;
  for (n = 0; n + 4 <= len; n += 4) {
    const __m128i uv_0 = LoadUVPart(u[0], v[0]);
    const __m128i uv_1 = LoadUVPart(u[1], v[1]);
    const __m128i tmp0_1 = GetRGBA32bWithUV(y[0], uv_0);
    const __m128i tmp0_2 = GetRGBA32bWithUV(y[1], uv_0);
    const __m128i tmp0_3 = GetRGBA32bWithUV(y[2], uv_1);
    const __m128i tmp0_4 = GetRGBA32bWithUV(y[3], uv_1);
    const __m128i tmp1_1 = _mm_packs_epi32(tmp0_1, tmp0_2);
    const __m128i tmp1_2 = _mm_packs_epi32(tmp0_3, tmp0_4);
    const __m128i tmp2 = _mm_packus_epi16(tmp1_1, tmp1_2);
    _mm_storeu_si128((__m128i*)dst, tmp2);
    dst += 4 * 4;
    y += 4;
    u += 2;
    v += 2;
  }
  // Finish off
  while (n < len) {
    VP8YuvToRgba(y[0], u[0], v[0], dst);
    dst += 4;
    ++y;
    u += (n & 1);
    v += (n & 1);
    ++n;
  }
}

static void YuvToBgraRow(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                         uint8_t* dst, int len) {
  int n;
  for (n = 0; n + 2 <= len; n += 2) {
    const __m128i uv_0 = LoadUVPart(u[0], v[0]);
    const __m128i tmp0_1 = GetRGBA32bWithUV(y[0], uv_0);
    const __m128i tmp0_2 = GetRGBA32bWithUV(y[1], uv_0);
    const __m128i tmp1_1 = _mm_shuffle_epi32(tmp0_1, _MM_SHUFFLE(3, 0, 1, 2));
    const __m128i tmp1_2 = _mm_shuffle_epi32(tmp0_2, _MM_SHUFFLE(3, 0, 1, 2));
    const __m128i tmp2_1 = _mm_packs_epi32(tmp1_1, tmp1_2);
    const __m128i tmp3 = _mm_packus_epi16(tmp2_1, tmp2_1);
    _mm_storel_epi64((__m128i*)dst, tmp3);
    dst += 4 * 2;
    y += 2;
    ++u;
    ++v;
  }
  // Finish off
  if (len & 1) {
    VP8YuvToBgra(y[0], u[0], v[0], dst);
  }
}

static void YuvToArgbRow(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                         uint8_t* dst, int len) {
  int n;
  for (n = 0; n + 2 <= len; n += 2) {
    const __m128i uv_0 = LoadUVPart(u[0], v[0]);
    const __m128i tmp0_1 = GetRGBA32bWithUV(y[0], uv_0);
    const __m128i tmp0_2 = GetRGBA32bWithUV(y[1], uv_0);
    const __m128i tmp1_1 = _mm_shuffle_epi32(tmp0_1, _MM_SHUFFLE(2, 1, 0, 3));
    const __m128i tmp1_2 = _mm_shuffle_epi32(tmp0_2, _MM_SHUFFLE(2, 1, 0, 3));
    const __m128i tmp2_1 = _mm_packs_epi32(tmp1_1, tmp1_2);
    const __m128i tmp3 = _mm_packus_epi16(tmp2_1, tmp2_1);
    _mm_storel_epi64((__m128i*)dst, tmp3);
    dst += 4 * 2;
    y += 2;
    ++u;
    ++v;
  }
  // Finish off
  if (len & 1) {
    VP8YuvToArgb(y[0], u[0], v[0], dst);
  }
}

static void YuvToRgbRow(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                        uint8_t* dst, int len) {
  int n;
  for (n = 0; n + 2 < len; ++n) {   // we directly stomp the *dst memory
    YuvToRgbSSE2(y[0], u[0], v[0], dst);  // stomps 8 bytes
    dst += 3;
    ++y;
    u += (n & 1);
    v += (n & 1);
  }
  VP8YuvToRgb(y[0], u[0], v[0], dst);
  if (len > 1) {
    VP8YuvToRgb(y[1], u[n & 1], v[n & 1], dst + 3);
  }
}

static void YuvToBgrRow(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                        uint8_t* dst, int len) {
  int n;
  for (n = 0; n + 2 < len; ++n) {   // we directly stomp the *dst memory
    YuvToBgrSSE2(y[0], u[0], v[0], dst);  // stomps 8 bytes
    dst += 3;
    ++y;
    u += (n & 1);
    v += (n & 1);
  }
  VP8YuvToBgr(y[0], u[0], v[0], dst + 0);
  if (len > 1) {
    VP8YuvToBgr(y[1], u[n & 1], v[n & 1], dst + 3);
  }
}

//------------------------------------------------------------------------------
// Entry point

extern void WebPInitSamplersSSE2(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPInitSamplersSSE2(void) {
  WebPSamplers[MODE_RGB]  = YuvToRgbRow;
  WebPSamplers[MODE_RGBA] = YuvToRgbaRow;
  WebPSamplers[MODE_BGR]  = YuvToBgrRow;
  WebPSamplers[MODE_BGRA] = YuvToBgraRow;
  WebPSamplers[MODE_ARGB] = YuvToArgbRow;
}

//------------------------------------------------------------------------------
// RGB24/32 -> YUV converters

// Load eight 16b-words from *src.
#define LOAD_16(src) _mm_loadu_si128((const __m128i*)(src))
// Store either 16b-words into *dst
#define STORE_16(V, dst) _mm_storeu_si128((__m128i*)(dst), (V))

// Convert 8 packed RGB or BGR samples to r[], g[], b[]
static WEBP_INLINE void RGB24PackedToPlanar(const uint8_t* const rgb,
                                            __m128i* const r,
                                            __m128i* const g,
                                            __m128i* const b,
                                            int input_is_bgr) {
  const __m128i zero = _mm_setzero_si128();
  // in0: r0 g0 b0 r1 | g1 b1 r2 g2 | b2 r3 g3 b3 | r4 g4 b4 r5
  // in1: b2 r3 g3 b3 | r4 g4 b4 r5 | g5 b5 r6 g6 | b6 r7 g7 b7
  const __m128i in0 = LOAD_16(rgb + 0);
  const __m128i in1 = LOAD_16(rgb + 8);
  // A0: | r2 g2 b2 r3 | g3 b3 r4 g4 | b4 r5 ...
  // A1:                   ... b2 r3 | g3 b3 r4 g4 | b4 r5 g5 b5 |
  const __m128i A0 = _mm_srli_si128(in0, 6);
  const __m128i A1 = _mm_slli_si128(in1, 6);
  // B0: r0 r2 g0 g2 | b0 b2 r1 r3 | g1 g3 b1 b3 | r2 r4 b2 b4
  // B1: g3 g5 b3 b5 | r4 r6 g4 g6 | b4 b6 r5 r7 | g5 g7 b5 b7
  const __m128i B0 = _mm_unpacklo_epi8(in0, A0);
  const __m128i B1 = _mm_unpackhi_epi8(A1, in1);
  // C0: r1 r3 g1 g3 | b1 b3 r2 r4 | b2 b4 ...
  // C1:                 ... g3 g5 | b3 b5 r4 r6 | g4 g6 b4 b6
  const __m128i C0 = _mm_srli_si128(B0, 6);
  const __m128i C1 = _mm_slli_si128(B1, 6);
  // D0: r0 r1 r2 r3 | g0 g1 g2 g3 | b0 b1 b2 b3 | r1 r2 r3 r4
  // D1: b3 b4 b5 b6 | r4 r5 r6 r7 | g4 g5 g6 g7 | b4 b5 b6 b7 |
  const __m128i D0 = _mm_unpacklo_epi8(B0, C0);
  const __m128i D1 = _mm_unpackhi_epi8(C1, B1);
  // r4 r5 r6 r7 | g4 g5 g6 g7 | b4 b5 b6 b7 | 0
  const __m128i D2 = _mm_srli_si128(D1, 4);
  // r0 r1 r2 r3 | r4 r5 r6 r7 | g0 g1 g2 g3 | g4 g5 g6 g7
  const __m128i E0 = _mm_unpacklo_epi32(D0, D2);
  // b0 b1 b2 b3 | b4 b5 b6 b7 | r1 r2 r3 r4 | 0
  const __m128i E1 = _mm_unpackhi_epi32(D0, D2);
  // g0 g1 g2 g3 | g4 g5 g6 g7 | 0
  const __m128i E2 = _mm_srli_si128(E0, 8);
  const __m128i F0 = _mm_unpacklo_epi8(E0, zero);  // -> R
  const __m128i F1 = _mm_unpacklo_epi8(E1, zero);  // -> B
  const __m128i F2 = _mm_unpacklo_epi8(E2, zero);  // -> G
  *g = F2;
  if (input_is_bgr) {
    *r = F1;
    *b = F0;
  } else {
    *r = F0;
    *b = F1;
  }
}

// Convert 8 packed ARGB to r[], g[], b[]
static WEBP_INLINE void RGB32PackedToPlanar(const uint32_t* const argb,
                                            __m128i* const r,
                                            __m128i* const g,
                                            __m128i* const b) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i in0 = LOAD_16(argb + 0);    // argb3 | argb2 | argb1 | argb0
  const __m128i in1 = LOAD_16(argb + 4);    // argb7 | argb6 | argb5 | argb4
  // column-wise transpose
  const __m128i A0 = _mm_unpacklo_epi8(in0, in1);
  const __m128i A1 = _mm_unpackhi_epi8(in0, in1);
  const __m128i B0 = _mm_unpacklo_epi8(A0, A1);
  const __m128i B1 = _mm_unpackhi_epi8(A0, A1);
  // C0 = g7 g6 ... g1 g0 | b7 b6 ... b1 b0
  // C1 = a7 a6 ... a1 a0 | r7 r6 ... r1 r0
  const __m128i C0 = _mm_unpacklo_epi8(B0, B1);
  const __m128i C1 = _mm_unpackhi_epi8(B0, B1);
  // store 16b
  *r = _mm_unpacklo_epi8(C1, zero);
  *g = _mm_unpackhi_epi8(C0, zero);
  *b = _mm_unpacklo_epi8(C0, zero);
}

// This macro computes (RG * MULT_RG + GB * MULT_GB + ROUNDER) >> DESCALE_FIX
// It's a macro and not a function because we need to use immediate values with
// srai_epi32, e.g.
#define TRANSFORM(RG_LO, RG_HI, GB_LO, GB_HI, MULT_RG, MULT_GB, \
                  ROUNDER, DESCALE_FIX, OUT) do {               \
  const __m128i V0_lo = _mm_madd_epi16(RG_LO, MULT_RG);         \
  const __m128i V0_hi = _mm_madd_epi16(RG_HI, MULT_RG);         \
  const __m128i V1_lo = _mm_madd_epi16(GB_LO, MULT_GB);         \
  const __m128i V1_hi = _mm_madd_epi16(GB_HI, MULT_GB);         \
  const __m128i V2_lo = _mm_add_epi32(V0_lo, V1_lo);            \
  const __m128i V2_hi = _mm_add_epi32(V0_hi, V1_hi);            \
  const __m128i V3_lo = _mm_add_epi32(V2_lo, ROUNDER);          \
  const __m128i V3_hi = _mm_add_epi32(V2_hi, ROUNDER);          \
  const __m128i V5_lo = _mm_srai_epi32(V3_lo, DESCALE_FIX);     \
  const __m128i V5_hi = _mm_srai_epi32(V3_hi, DESCALE_FIX);     \
  (OUT) = _mm_packs_epi32(V5_lo, V5_hi);                        \
} while (0)

#define MK_CST_16(A, B) _mm_set_epi16((B), (A), (B), (A), (B), (A), (B), (A))
static WEBP_INLINE void ConvertRGBToY(const __m128i* const R,
                                      const __m128i* const G,
                                      const __m128i* const B,
                                      __m128i* const Y) {
  const __m128i kRG_y = MK_CST_16(16839, 33059 - 16384);
  const __m128i kGB_y = MK_CST_16(16384, 6420);
  const __m128i kHALF_Y = _mm_set1_epi32((16 << YUV_FIX) + YUV_HALF);

  const __m128i RG_lo = _mm_unpacklo_epi16(*R, *G);
  const __m128i RG_hi = _mm_unpackhi_epi16(*R, *G);
  const __m128i GB_lo = _mm_unpacklo_epi16(*G, *B);
  const __m128i GB_hi = _mm_unpackhi_epi16(*G, *B);
  TRANSFORM(RG_lo, RG_hi, GB_lo, GB_hi, kRG_y, kGB_y, kHALF_Y, YUV_FIX, *Y);
}

static WEBP_INLINE void ConvertRGBToUV(const __m128i* const R,
                                       const __m128i* const G,
                                       const __m128i* const B,
                                       __m128i* const U, __m128i* const V) {
  const __m128i kRG_u = MK_CST_16(-9719, -19081);
  const __m128i kGB_u = MK_CST_16(0, 28800);
  const __m128i kRG_v = MK_CST_16(28800, 0);
  const __m128i kGB_v = MK_CST_16(-24116, -4684);
  const __m128i kHALF_UV = _mm_set1_epi32(((128 << YUV_FIX) + YUV_HALF) << 2);

  const __m128i RG_lo = _mm_unpacklo_epi16(*R, *G);
  const __m128i RG_hi = _mm_unpackhi_epi16(*R, *G);
  const __m128i GB_lo = _mm_unpacklo_epi16(*G, *B);
  const __m128i GB_hi = _mm_unpackhi_epi16(*G, *B);
  TRANSFORM(RG_lo, RG_hi, GB_lo, GB_hi, kRG_u, kGB_u,
            kHALF_UV, YUV_FIX + 2, *U);
  TRANSFORM(RG_lo, RG_hi, GB_lo, GB_hi, kRG_v, kGB_v,
            kHALF_UV, YUV_FIX + 2, *V);
}

#undef MK_CST_16
#undef TRANSFORM

static void ConvertRGB24ToY(const uint8_t* rgb, uint8_t* y, int width) {
  const int max_width = width & ~15;
  int i;
  for (i = 0; i < max_width; i += 16, rgb += 3 * 16) {
    __m128i r, g, b, Y0, Y1;
    RGB24PackedToPlanar(rgb + 0 * 8, &r, &g, &b, 0);
    ConvertRGBToY(&r, &g, &b, &Y0);
    RGB24PackedToPlanar(rgb + 3 * 8, &r, &g, &b, 0);
    ConvertRGBToY(&r, &g, &b, &Y1);
    STORE_16(_mm_packus_epi16(Y0, Y1), y + i);
  }
  for (; i < width; ++i, rgb += 3) {   // left-over
    y[i] = VP8RGBToY(rgb[0], rgb[1], rgb[2], YUV_HALF);
  }
}

static void ConvertBGR24ToY(const uint8_t* bgr, uint8_t* y, int width) {
  int i;
  const int max_width = width & ~15;
  for (i = 0; i < max_width; i += 16, bgr += 3 * 16) {
    __m128i r, g, b, Y0, Y1;
    RGB24PackedToPlanar(bgr + 0 * 8, &r, &g, &b, 1);
    ConvertRGBToY(&r, &g, &b, &Y0);
    RGB24PackedToPlanar(bgr + 3 * 8, &r, &g, &b, 1);
    ConvertRGBToY(&r, &g, &b, &Y1);
    STORE_16(_mm_packus_epi16(Y0, Y1), y + i);
  }
  for (; i < width; ++i, bgr += 3) {  // left-over
    y[i] = VP8RGBToY(bgr[2], bgr[1], bgr[0], YUV_HALF);
  }
}

static void ConvertARGBToY(const uint32_t* argb, uint8_t* y, int width) {
  const int max_width = width & ~15;
  int i;
  for (i = 0; i < max_width; i += 16) {
    __m128i r, g, b, Y0, Y1;
    RGB32PackedToPlanar(&argb[i + 0], &r, &g, &b);
    ConvertRGBToY(&r, &g, &b, &Y0);
    RGB32PackedToPlanar(&argb[i + 8], &r, &g, &b);
    ConvertRGBToY(&r, &g, &b, &Y1);
    STORE_16(_mm_packus_epi16(Y0, Y1), y + i);
  }
  for (; i < width; ++i) {   // left-over
    const uint32_t p = argb[i];
    y[i] = VP8RGBToY((p >> 16) & 0xff, (p >> 8) & 0xff, (p >>  0) & 0xff,
                     YUV_HALF);
  }
}

// Horizontal add (doubled) of two 16b values, result is 16b.
// in: A | B | C | D | ... -> out: 2*(A+B) | 2*(C+D) | ...
static void HorizontalAddPack(const __m128i* const A, const __m128i* const B,
                              __m128i* const out) {
  const __m128i k2 = _mm_set1_epi16(2);
  const __m128i C = _mm_madd_epi16(*A, k2);
  const __m128i D = _mm_madd_epi16(*B, k2);
  *out = _mm_packs_epi32(C, D);
}

static void ConvertARGBToUV(const uint32_t* argb, uint8_t* u, uint8_t* v,
                            int src_width, int do_store) {
  const int max_width = src_width & ~31;
  int i;
  for (i = 0; i < max_width; i += 32, u += 16, v += 16) {
    __m128i r0, g0, b0, r1, g1, b1, U0, V0, U1, V1;
    RGB32PackedToPlanar(&argb[i +  0], &r0, &g0, &b0);
    RGB32PackedToPlanar(&argb[i +  8], &r1, &g1, &b1);
    HorizontalAddPack(&r0, &r1, &r0);
    HorizontalAddPack(&g0, &g1, &g0);
    HorizontalAddPack(&b0, &b1, &b0);
    ConvertRGBToUV(&r0, &g0, &b0, &U0, &V0);

    RGB32PackedToPlanar(&argb[i + 16], &r0, &g0, &b0);
    RGB32PackedToPlanar(&argb[i + 24], &r1, &g1, &b1);
    HorizontalAddPack(&r0, &r1, &r0);
    HorizontalAddPack(&g0, &g1, &g0);
    HorizontalAddPack(&b0, &b1, &b0);
    ConvertRGBToUV(&r0, &g0, &b0, &U1, &V1);

    U0 = _mm_packus_epi16(U0, U1);
    V0 = _mm_packus_epi16(V0, V1);
    if (!do_store) {
      const __m128i prev_u = LOAD_16(u);
      const __m128i prev_v = LOAD_16(v);
      U0 = _mm_avg_epu8(U0, prev_u);
      V0 = _mm_avg_epu8(V0, prev_v);
    }
    STORE_16(U0, u);
    STORE_16(V0, v);
  }
  if (i < src_width) {  // left-over
    WebPConvertARGBToUV_C(argb + i, u, v, src_width - i, do_store);
  }
}

// Convert 16 packed ARGB 16b-values to r[], g[], b[]
static WEBP_INLINE void RGBA32PackedToPlanar_16b(const uint16_t* const rgbx,
                                                 __m128i* const r,
                                                 __m128i* const g,
                                                 __m128i* const b) {
  const __m128i in0 = LOAD_16(rgbx +  0);  // r0 | g0 | b0 |x| r1 | g1 | b1 |x
  const __m128i in1 = LOAD_16(rgbx +  8);  // r2 | g2 | b2 |x| r3 | g3 | b3 |x
  const __m128i in2 = LOAD_16(rgbx + 16);  // r4 | ...
  const __m128i in3 = LOAD_16(rgbx + 24);  // r6 | ...
  // column-wise transpose
  const __m128i A0 = _mm_unpacklo_epi16(in0, in1);
  const __m128i A1 = _mm_unpackhi_epi16(in0, in1);
  const __m128i A2 = _mm_unpacklo_epi16(in2, in3);
  const __m128i A3 = _mm_unpackhi_epi16(in2, in3);
  const __m128i B0 = _mm_unpacklo_epi16(A0, A1);  // r0 r1 r2 r3 | g0 g1 ..
  const __m128i B1 = _mm_unpackhi_epi16(A0, A1);  // b0 b1 b2 b3 | x x x x
  const __m128i B2 = _mm_unpacklo_epi16(A2, A3);  // r4 r5 r6 r7 | g4 g5 ..
  const __m128i B3 = _mm_unpackhi_epi16(A2, A3);  // b4 b5 b6 b7 | x x x x
  *r = _mm_unpacklo_epi64(B0, B2);
  *g = _mm_unpackhi_epi64(B0, B2);
  *b = _mm_unpacklo_epi64(B1, B3);
}

static void ConvertRGBA32ToUV(const uint16_t* rgb,
                              uint8_t* u, uint8_t* v, int width) {
  const int max_width = width & ~15;
  const uint16_t* const last_rgb = rgb + 4 * max_width;
  while (rgb < last_rgb) {
    __m128i r, g, b, U0, V0, U1, V1;
    RGBA32PackedToPlanar_16b(rgb +  0, &r, &g, &b);
    ConvertRGBToUV(&r, &g, &b, &U0, &V0);
    RGBA32PackedToPlanar_16b(rgb + 32, &r, &g, &b);
    ConvertRGBToUV(&r, &g, &b, &U1, &V1);
    STORE_16(_mm_packus_epi16(U0, U1), u);
    STORE_16(_mm_packus_epi16(V0, V1), v);
    u += 16;
    v += 16;
    rgb += 2 * 32;
  }
  if (max_width < width) {  // left-over
    WebPConvertRGBA32ToUV_C(rgb, u, v, width - max_width);
  }
}

//------------------------------------------------------------------------------

extern void WebPInitConvertARGBToYUVSSE2(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPInitConvertARGBToYUVSSE2(void) {
  WebPConvertARGBToY = ConvertARGBToY;
  WebPConvertARGBToUV = ConvertARGBToUV;

  WebPConvertRGB24ToY = ConvertRGB24ToY;
  WebPConvertBGR24ToY = ConvertBGR24ToY;

  WebPConvertRGBA32ToUV = ConvertRGBA32ToUV;
}

#else  // !WEBP_USE_SSE2

WEBP_DSP_INIT_STUB(WebPInitSamplersSSE2)
WEBP_DSP_INIT_STUB(WebPInitConvertARGBToYUVSSE2)

#endif  // WEBP_USE_SSE2
