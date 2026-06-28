// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Utilities for processing transparent channel, SSE4.1 variant.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/dsp/cpu.h"
#include "src/webp/types.h"
#include "src/dsp/dsp.h"

#if defined(WEBP_USE_SSE41)
#include <emmintrin.h>
#include <smmintrin.h>

//------------------------------------------------------------------------------

static int ExtractAlpha_SSE41(const uint8_t* WEBP_RESTRICT argb,
                              int argb_stride, int width, int height,
                              uint8_t* WEBP_RESTRICT alpha, int alpha_stride) {
  // alpha_and stores an 'and' operation of all the alpha[] values. The final
  // value is not 0xff if any of the alpha[] is not equal to 0xff.
  uint32_t alpha_and = 0xff;
  int i, j;
  const __m128i all_0xff = _mm_set1_epi32(~0);
  __m128i all_alphas = all_0xff;

  // We must be able to access 3 extra bytes after the last written byte
  // 'src[4 * width - 4]', because we don't know if alpha is the first or the
  // last byte of the quadruplet.
  const int limit = (width - 1) & ~15;
  const __m128i kCstAlpha0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                          -1, -1, -1, -1, 12, 8, 4, 0);
  const __m128i kCstAlpha1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                          12, 8, 4, 0, -1, -1, -1, -1);
  const __m128i kCstAlpha2 = _mm_set_epi8(-1, -1, -1, -1, 12, 8, 4, 0,
                                          -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i kCstAlpha3 = _mm_set_epi8(12, 8, 4, 0, -1, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1);
  for (j = 0; j < height; ++j) {
    const __m128i* src = (const __m128i*)argb;
    for (i = 0; i < limit; i += 16) {
      // load 64 argb bytes
      const __m128i a0 = _mm_loadu_si128(src + 0);
      const __m128i a1 = _mm_loadu_si128(src + 1);
      const __m128i a2 = _mm_loadu_si128(src + 2);
      const __m128i a3 = _mm_loadu_si128(src + 3);
      const __m128i b0 = _mm_shuffle_epi8(a0, kCstAlpha0);
      const __m128i b1 = _mm_shuffle_epi8(a1, kCstAlpha1);
      const __m128i b2 = _mm_shuffle_epi8(a2, kCstAlpha2);
      const __m128i b3 = _mm_shuffle_epi8(a3, kCstAlpha3);
      const __m128i c0 = _mm_or_si128(b0, b1);
      const __m128i c1 = _mm_or_si128(b2, b3);
      const __m128i d0 = _mm_or_si128(c0, c1);
      // store
      _mm_storeu_si128((__m128i*)&alpha[i], d0);
      // accumulate sixteen alpha 'and' in parallel
      all_alphas = _mm_and_si128(all_alphas, d0);
      src += 4;
    }
    for (; i < width; ++i) {
      const uint32_t alpha_value = argb[4 * i];
      alpha[i] = alpha_value;
      alpha_and &= alpha_value;
    }
    argb += argb_stride;
    alpha += alpha_stride;
  }
  // Combine the sixteen alpha 'and' into an 8-bit mask.
  alpha_and |= 0xff00u;  // pretend the upper bits [8..15] were tested ok.
  alpha_and &= _mm_movemask_epi8(_mm_cmpeq_epi8(all_alphas, all_0xff));
  return (alpha_and == 0xffffu);
}

//------------------------------------------------------------------------------
// Entry point

extern void WebPInitAlphaProcessingSSE41(void);

WEBP_TSAN_IGNORE_FUNCTION void WebPInitAlphaProcessingSSE41(void) {
  WebPExtractAlpha = ExtractAlpha_SSE41;
}

#else  // !WEBP_USE_SSE41

WEBP_DSP_INIT_STUB(WebPInitAlphaProcessingSSE41)

#endif  // WEBP_USE_SSE41
