// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//   ARGB making functions (SSE2 version).
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./dsp.h"

#if defined(WEBP_USE_SSE2)

#include <assert.h>
#include <emmintrin.h>
#include <string.h>

static WEBP_INLINE uint32_t MakeARGB32(int a, int r, int g, int b) {
  return (((uint32_t)a << 24) | (r << 16) | (g << 8) | b);
}

static void PackARGB(const uint8_t* a, const uint8_t* r, const uint8_t* g,
                     const uint8_t* b, int len, uint32_t* out) {
  if (g == r + 1) {  // RGBA input order. Need to swap R and B.
    int i = 0;
    const int len_max = len & ~3;  // max length processed in main loop
    const __m128i red_blue_mask = _mm_set1_epi32(0x00ff00ffu);
    assert(b == r + 2);
    assert(a == r + 3);
    for (; i < len_max; i += 4) {
      const __m128i A = _mm_loadu_si128((const __m128i*)(r + 4 * i));
      const __m128i B = _mm_and_si128(A, red_blue_mask);     // R 0 B 0
      const __m128i C = _mm_andnot_si128(red_blue_mask, A);  // 0 G 0 A
      const __m128i D = _mm_shufflelo_epi16(B, _MM_SHUFFLE(2, 3, 0, 1));
      const __m128i E = _mm_shufflehi_epi16(D, _MM_SHUFFLE(2, 3, 0, 1));
      const __m128i F = _mm_or_si128(E, C);
      _mm_storeu_si128((__m128i*)(out + i), F);
    }
    for (; i < len; ++i) {
      out[i] = MakeARGB32(a[4 * i], r[4 * i], g[4 * i], b[4 * i]);
    }
  } else {
    assert(g == b + 1);
    assert(r == b + 2);
    assert(a == b + 3);
    memcpy(out, b, len * 4);
  }
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8EncDspARGBInitSSE2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8EncDspARGBInitSSE2(void) {
  VP8PackARGB = PackARGB;
}

#else  // !WEBP_USE_SSE2

WEBP_DSP_INIT_STUB(VP8EncDspARGBInitSSE2)

#endif  // WEBP_USE_SSE2
