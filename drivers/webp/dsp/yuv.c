// Copyright 2010 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// YUV->RGB conversion function
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./yuv.h"


#if defined(WEBP_YUV_USE_TABLE)

static int done = 0;

static WEBP_INLINE uint8_t clip(int v, int max_value) {
  return v < 0 ? 0 : v > max_value ? max_value : v;
}

int16_t VP8kVToR[256], VP8kUToB[256];
int32_t VP8kVToG[256], VP8kUToG[256];
uint8_t VP8kClip[YUV_RANGE_MAX - YUV_RANGE_MIN];
uint8_t VP8kClip4Bits[YUV_RANGE_MAX - YUV_RANGE_MIN];

void VP8YUVInit(void) {
  int i;
  if (done) {
    return;
  }
#ifndef USE_YUVj
  for (i = 0; i < 256; ++i) {
    VP8kVToR[i] = (89858 * (i - 128) + YUV_HALF) >> YUV_FIX;
    VP8kUToG[i] = -22014 * (i - 128) + YUV_HALF;
    VP8kVToG[i] = -45773 * (i - 128);
    VP8kUToB[i] = (113618 * (i - 128) + YUV_HALF) >> YUV_FIX;
  }
  for (i = YUV_RANGE_MIN; i < YUV_RANGE_MAX; ++i) {
    const int k = ((i - 16) * 76283 + YUV_HALF) >> YUV_FIX;
    VP8kClip[i - YUV_RANGE_MIN] = clip(k, 255);
    VP8kClip4Bits[i - YUV_RANGE_MIN] = clip((k + 8) >> 4, 15);
  }
#else
  for (i = 0; i < 256; ++i) {
    VP8kVToR[i] = (91881 * (i - 128) + YUV_HALF) >> YUV_FIX;
    VP8kUToG[i] = -22554 * (i - 128) + YUV_HALF;
    VP8kVToG[i] = -46802 * (i - 128);
    VP8kUToB[i] = (116130 * (i - 128) + YUV_HALF) >> YUV_FIX;
  }
  for (i = YUV_RANGE_MIN; i < YUV_RANGE_MAX; ++i) {
    const int k = i;
    VP8kClip[i - YUV_RANGE_MIN] = clip(k, 255);
    VP8kClip4Bits[i - YUV_RANGE_MIN] = clip((k + 8) >> 4, 15);
  }
#endif

  done = 1;
}

#else

void VP8YUVInit(void) {}

#endif  // WEBP_YUV_USE_TABLE

//-----------------------------------------------------------------------------
// SSE2 extras

#if defined(WEBP_USE_SSE2)

#ifdef FANCY_UPSAMPLING

#include <emmintrin.h>
#include <string.h>   // for memcpy

typedef union {   // handy struct for converting SSE2 registers
  int32_t i32[4];
  uint8_t u8[16];
  __m128i m;
} VP8kCstSSE2;

static int done_sse2 = 0;
static VP8kCstSSE2 VP8kUtoRGBA[256], VP8kVtoRGBA[256], VP8kYtoRGBA[256];

void VP8YUVInitSSE2(void) {
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
  }
}

static WEBP_INLINE __m128i VP8GetRGBA32b(int y, int u, int v) {
  const __m128i u_part = _mm_loadu_si128(&VP8kUtoRGBA[u].m);
  const __m128i v_part = _mm_loadu_si128(&VP8kVtoRGBA[v].m);
  const __m128i y_part = _mm_loadu_si128(&VP8kYtoRGBA[y].m);
  const __m128i uv_part = _mm_add_epi32(u_part, v_part);
  const __m128i rgba1 = _mm_add_epi32(y_part, uv_part);
  const __m128i rgba2 = _mm_srai_epi32(rgba1, YUV_FIX2);
  return rgba2;
}

static WEBP_INLINE void VP8YuvToRgbSSE2(uint8_t y, uint8_t u, uint8_t v,
                                        uint8_t* const rgb) {
  const __m128i tmp0 = VP8GetRGBA32b(y, u, v);
  const __m128i tmp1 = _mm_packs_epi32(tmp0, tmp0);
  const __m128i tmp2 = _mm_packus_epi16(tmp1, tmp1);
  // Note: we store 8 bytes at a time, not 3 bytes! -> memory stomp
  _mm_storel_epi64((__m128i*)rgb, tmp2);
}

static WEBP_INLINE void VP8YuvToBgrSSE2(uint8_t y, uint8_t u, uint8_t v,
                                        uint8_t* const bgr) {
  const __m128i tmp0 = VP8GetRGBA32b(y, u, v);
  const __m128i tmp1 = _mm_shuffle_epi32(tmp0, _MM_SHUFFLE(3, 0, 1, 2));
  const __m128i tmp2 = _mm_packs_epi32(tmp1, tmp1);
  const __m128i tmp3 = _mm_packus_epi16(tmp2, tmp2);
  // Note: we store 8 bytes at a time, not 3 bytes! -> memory stomp
  _mm_storel_epi64((__m128i*)bgr, tmp3);
}

void VP8YuvToRgba32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                    uint8_t* dst) {
  int n;
  for (n = 0; n < 32; n += 4) {
    const __m128i tmp0_1 = VP8GetRGBA32b(y[n + 0], u[n + 0], v[n + 0]);
    const __m128i tmp0_2 = VP8GetRGBA32b(y[n + 1], u[n + 1], v[n + 1]);
    const __m128i tmp0_3 = VP8GetRGBA32b(y[n + 2], u[n + 2], v[n + 2]);
    const __m128i tmp0_4 = VP8GetRGBA32b(y[n + 3], u[n + 3], v[n + 3]);
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
    const __m128i tmp0_1 = VP8GetRGBA32b(y[n + 0], u[n + 0], v[n + 0]);
    const __m128i tmp0_2 = VP8GetRGBA32b(y[n + 1], u[n + 1], v[n + 1]);
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
    VP8YuvToRgbSSE2(y[n], u[n], v[n], dst + n * 3);
  }
  // Last two pixels are special: we write in a tmp buffer before sending
  // to dst.
  VP8YuvToRgbSSE2(y[n + 0], u[n + 0], v[n + 0], tmp + 0);
  VP8YuvToRgbSSE2(y[n + 1], u[n + 1], v[n + 1], tmp + 3);
  memcpy(dst + n * 3, tmp, 2 * 3);
}

void VP8YuvToBgr32(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                   uint8_t* dst) {
  int n;
  uint8_t tmp0[2 * 3 + 5 + 15];
  uint8_t* const tmp = (uint8_t*)((uintptr_t)(tmp0 + 15) & ~15);  // align
  for (n = 0; n < 30; ++n) {
    VP8YuvToBgrSSE2(y[n], u[n], v[n], dst + n * 3);
  }
  VP8YuvToBgrSSE2(y[n + 0], u[n + 0], v[n + 0], tmp + 0);
  VP8YuvToBgrSSE2(y[n + 1], u[n + 1], v[n + 1], tmp + 3);
  memcpy(dst + n * 3, tmp, 2 * 3);
}

#else

void VP8YUVInitSSE2(void) {}

#endif  // FANCY_UPSAMPLING

#endif  // WEBP_USE_SSE2

