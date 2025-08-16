// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// SSE4 version of some decoding functions.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_SSE41)

#include <smmintrin.h>
#include "src/dec/vp8i_dec.h"
#include "src/utils/utils.h"

static void HE16_SSE41(uint8_t* dst) {     // horizontal
  int j;
  const __m128i kShuffle3 = _mm_set1_epi8(3);
  for (j = 16; j > 0; --j) {
    const __m128i in = _mm_cvtsi32_si128(WebPMemToInt32(dst - 4));
    const __m128i values = _mm_shuffle_epi8(in, kShuffle3);
    _mm_storeu_si128((__m128i*)dst, values);
    dst += BPS;
  }
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8DspInitSSE41(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8DspInitSSE41(void) {
  VP8PredLuma16[3] = HE16_SSE41;
}

#else  // !WEBP_USE_SSE41

WEBP_DSP_INIT_STUB(VP8DspInitSSE41)

#endif  // WEBP_USE_SSE41
