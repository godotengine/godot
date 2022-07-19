// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// SSE2 version of distortion calculation
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_SSE2)

#include <assert.h>
#include <emmintrin.h>

#include "src/dsp/common_sse2.h"

#if !defined(WEBP_DISABLE_STATS)

// Helper function
static WEBP_INLINE void SubtractAndSquare_SSE2(const __m128i a, const __m128i b,
                                               __m128i* const sum) {
  // take abs(a-b) in 8b
  const __m128i a_b = _mm_subs_epu8(a, b);
  const __m128i b_a = _mm_subs_epu8(b, a);
  const __m128i abs_a_b = _mm_or_si128(a_b, b_a);
  // zero-extend to 16b
  const __m128i zero = _mm_setzero_si128();
  const __m128i C0 = _mm_unpacklo_epi8(abs_a_b, zero);
  const __m128i C1 = _mm_unpackhi_epi8(abs_a_b, zero);
  // multiply with self
  const __m128i sum1 = _mm_madd_epi16(C0, C0);
  const __m128i sum2 = _mm_madd_epi16(C1, C1);
  *sum = _mm_add_epi32(sum1, sum2);
}

//------------------------------------------------------------------------------
// SSIM / PSNR entry point

static uint32_t AccumulateSSE_SSE2(const uint8_t* src1,
                                   const uint8_t* src2, int len) {
  int i = 0;
  uint32_t sse2 = 0;
  if (len >= 16) {
    const int limit = len - 32;
    int32_t tmp[4];
    __m128i sum1;
    __m128i sum = _mm_setzero_si128();
    __m128i a0 = _mm_loadu_si128((const __m128i*)&src1[i]);
    __m128i b0 = _mm_loadu_si128((const __m128i*)&src2[i]);
    i += 16;
    while (i <= limit) {
      const __m128i a1 = _mm_loadu_si128((const __m128i*)&src1[i]);
      const __m128i b1 = _mm_loadu_si128((const __m128i*)&src2[i]);
      __m128i sum2;
      i += 16;
      SubtractAndSquare_SSE2(a0, b0, &sum1);
      sum = _mm_add_epi32(sum, sum1);
      a0 = _mm_loadu_si128((const __m128i*)&src1[i]);
      b0 = _mm_loadu_si128((const __m128i*)&src2[i]);
      i += 16;
      SubtractAndSquare_SSE2(a1, b1, &sum2);
      sum = _mm_add_epi32(sum, sum2);
    }
    SubtractAndSquare_SSE2(a0, b0, &sum1);
    sum = _mm_add_epi32(sum, sum1);
    _mm_storeu_si128((__m128i*)tmp, sum);
    sse2 += (tmp[3] + tmp[2] + tmp[1] + tmp[0]);
  }

  for (; i < len; ++i) {
    const int32_t diff = src1[i] - src2[i];
    sse2 += diff * diff;
  }
  return sse2;
}
#endif  // !defined(WEBP_DISABLE_STATS)

#if !defined(WEBP_REDUCE_SIZE)

static uint32_t HorizontalAdd16b_SSE2(const __m128i* const m) {
  uint16_t tmp[8];
  const __m128i a = _mm_srli_si128(*m, 8);
  const __m128i b = _mm_add_epi16(*m, a);
  _mm_storeu_si128((__m128i*)tmp, b);
  return (uint32_t)tmp[3] + tmp[2] + tmp[1] + tmp[0];
}

static uint32_t HorizontalAdd32b_SSE2(const __m128i* const m) {
  const __m128i a = _mm_srli_si128(*m, 8);
  const __m128i b = _mm_add_epi32(*m, a);
  const __m128i c = _mm_add_epi32(b, _mm_srli_si128(b, 4));
  return (uint32_t)_mm_cvtsi128_si32(c);
}

static const uint16_t kWeight[] = { 1, 2, 3, 4, 3, 2, 1, 0 };

#define ACCUMULATE_ROW(WEIGHT) do {                         \
  /* compute row weight (Wx * Wy) */                        \
  const __m128i Wy = _mm_set1_epi16((WEIGHT));              \
  const __m128i W = _mm_mullo_epi16(Wx, Wy);                \
  /* process 8 bytes at a time (7 bytes, actually) */       \
  const __m128i a0 = _mm_loadl_epi64((const __m128i*)src1); \
  const __m128i b0 = _mm_loadl_epi64((const __m128i*)src2); \
  /* convert to 16b and multiply by weight */               \
  const __m128i a1 = _mm_unpacklo_epi8(a0, zero);           \
  const __m128i b1 = _mm_unpacklo_epi8(b0, zero);           \
  const __m128i wa1 = _mm_mullo_epi16(a1, W);               \
  const __m128i wb1 = _mm_mullo_epi16(b1, W);               \
  /* accumulate */                                          \
  xm  = _mm_add_epi16(xm, wa1);                             \
  ym  = _mm_add_epi16(ym, wb1);                             \
  xxm = _mm_add_epi32(xxm, _mm_madd_epi16(a1, wa1));        \
  xym = _mm_add_epi32(xym, _mm_madd_epi16(a1, wb1));        \
  yym = _mm_add_epi32(yym, _mm_madd_epi16(b1, wb1));        \
  src1 += stride1;                                          \
  src2 += stride2;                                          \
} while (0)

static double SSIMGet_SSE2(const uint8_t* src1, int stride1,
                           const uint8_t* src2, int stride2) {
  VP8DistoStats stats;
  const __m128i zero = _mm_setzero_si128();
  __m128i xm = zero, ym = zero;                // 16b accums
  __m128i xxm = zero, yym = zero, xym = zero;  // 32b accum
  const __m128i Wx = _mm_loadu_si128((const __m128i*)kWeight);
  assert(2 * VP8_SSIM_KERNEL + 1 == 7);
  ACCUMULATE_ROW(1);
  ACCUMULATE_ROW(2);
  ACCUMULATE_ROW(3);
  ACCUMULATE_ROW(4);
  ACCUMULATE_ROW(3);
  ACCUMULATE_ROW(2);
  ACCUMULATE_ROW(1);
  stats.xm  = HorizontalAdd16b_SSE2(&xm);
  stats.ym  = HorizontalAdd16b_SSE2(&ym);
  stats.xxm = HorizontalAdd32b_SSE2(&xxm);
  stats.xym = HorizontalAdd32b_SSE2(&xym);
  stats.yym = HorizontalAdd32b_SSE2(&yym);
  return VP8SSIMFromStats(&stats);
}

#endif  // !defined(WEBP_REDUCE_SIZE)

extern void VP8SSIMDspInitSSE2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8SSIMDspInitSSE2(void) {
#if !defined(WEBP_DISABLE_STATS)
  VP8AccumulateSSE = AccumulateSSE_SSE2;
#endif
#if !defined(WEBP_REDUCE_SIZE)
  VP8SSIMGet = SSIMGet_SSE2;
#endif
}

#else  // !WEBP_USE_SSE2

WEBP_DSP_INIT_STUB(VP8SSIMDspInitSSE2)

#endif  // WEBP_USE_SSE2
