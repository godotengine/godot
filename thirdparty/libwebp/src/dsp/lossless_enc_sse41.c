// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// SSE4.1 variant of methods for lossless encoder
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_SSE41)
#include <assert.h>
#include <smmintrin.h>
#include "src/dsp/lossless.h"

// For sign-extended multiplying constants, pre-shifted by 5:
#define CST_5b(X)  (((int16_t)((uint16_t)(X) << 8)) >> 5)

//------------------------------------------------------------------------------
// Subtract-Green Transform

static void SubtractGreenFromBlueAndRed_SSE41(uint32_t* argb_data,
                                              int num_pixels) {
  int i;
  const __m128i kCstShuffle = _mm_set_epi8(-1, 13, -1, 13, -1, 9, -1, 9,
                                           -1,  5, -1,  5, -1, 1, -1, 1);
  for (i = 0; i + 4 <= num_pixels; i += 4) {
    const __m128i in = _mm_loadu_si128((__m128i*)&argb_data[i]);
    const __m128i in_0g0g = _mm_shuffle_epi8(in, kCstShuffle);
    const __m128i out = _mm_sub_epi8(in, in_0g0g);
    _mm_storeu_si128((__m128i*)&argb_data[i], out);
  }
  // fallthrough and finish off with plain-C
  if (i != num_pixels) {
    VP8LSubtractGreenFromBlueAndRed_C(argb_data + i, num_pixels - i);
  }
}

//------------------------------------------------------------------------------
// Color Transform

#define SPAN 8
static void CollectColorBlueTransforms_SSE41(const uint32_t* argb, int stride,
                                             int tile_width, int tile_height,
                                             int green_to_blue, int red_to_blue,
                                             int histo[]) {
  const __m128i mults_r = _mm_set1_epi16(CST_5b(red_to_blue));
  const __m128i mults_g = _mm_set1_epi16(CST_5b(green_to_blue));
  const __m128i mask_g = _mm_set1_epi16(0xff00);   // green mask
  const __m128i mask_gb = _mm_set1_epi32(0xffff);  // green/blue mask
  const __m128i mask_b = _mm_set1_epi16(0x00ff);   // blue mask
  const __m128i shuffler_lo = _mm_setr_epi8(-1, 2, -1, 6, -1, 10, -1, 14, -1,
                                            -1, -1, -1, -1, -1, -1, -1);
  const __m128i shuffler_hi = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1,
                                            2, -1, 6, -1, 10, -1, 14);
  int y;
  for (y = 0; y < tile_height; ++y) {
    const uint32_t* const src = argb + y * stride;
    int i, x;
    for (x = 0; x + SPAN <= tile_width; x += SPAN) {
      uint16_t values[SPAN];
      const __m128i in0 = _mm_loadu_si128((__m128i*)&src[x + 0]);
      const __m128i in1 = _mm_loadu_si128((__m128i*)&src[x + SPAN / 2]);
      const __m128i r0 = _mm_shuffle_epi8(in0, shuffler_lo);
      const __m128i r1 = _mm_shuffle_epi8(in1, shuffler_hi);
      const __m128i r = _mm_or_si128(r0, r1);         // r 0
      const __m128i gb0 = _mm_and_si128(in0, mask_gb);
      const __m128i gb1 = _mm_and_si128(in1, mask_gb);
      const __m128i gb = _mm_packus_epi32(gb0, gb1);  // g b
      const __m128i g = _mm_and_si128(gb, mask_g);    // g 0
      const __m128i A = _mm_mulhi_epi16(r, mults_r);  // x dbr
      const __m128i B = _mm_mulhi_epi16(g, mults_g);  // x dbg
      const __m128i C = _mm_sub_epi8(gb, B);          // x b'
      const __m128i D = _mm_sub_epi8(C, A);           // x b''
      const __m128i E = _mm_and_si128(D, mask_b);     // 0 b''
      _mm_storeu_si128((__m128i*)values, E);
      for (i = 0; i < SPAN; ++i) ++histo[values[i]];
    }
  }
  {
    const int left_over = tile_width & (SPAN - 1);
    if (left_over > 0) {
      VP8LCollectColorBlueTransforms_C(argb + tile_width - left_over, stride,
                                       left_over, tile_height,
                                       green_to_blue, red_to_blue, histo);
    }
  }
}

static void CollectColorRedTransforms_SSE41(const uint32_t* argb, int stride,
                                            int tile_width, int tile_height,
                                            int green_to_red, int histo[]) {
  const __m128i mults_g = _mm_set1_epi16(CST_5b(green_to_red));
  const __m128i mask_g = _mm_set1_epi32(0x00ff00);  // green mask
  const __m128i mask = _mm_set1_epi16(0xff);

  int y;
  for (y = 0; y < tile_height; ++y) {
    const uint32_t* const src = argb + y * stride;
    int i, x;
    for (x = 0; x + SPAN <= tile_width; x += SPAN) {
      uint16_t values[SPAN];
      const __m128i in0 = _mm_loadu_si128((__m128i*)&src[x + 0]);
      const __m128i in1 = _mm_loadu_si128((__m128i*)&src[x + SPAN / 2]);
      const __m128i g0 = _mm_and_si128(in0, mask_g);  // 0 0  | g 0
      const __m128i g1 = _mm_and_si128(in1, mask_g);
      const __m128i g = _mm_packus_epi32(g0, g1);     // g 0
      const __m128i A0 = _mm_srli_epi32(in0, 16);     // 0 0  | x r
      const __m128i A1 = _mm_srli_epi32(in1, 16);
      const __m128i A = _mm_packus_epi32(A0, A1);     // x r
      const __m128i B = _mm_mulhi_epi16(g, mults_g);  // x dr
      const __m128i C = _mm_sub_epi8(A, B);           // x r'
      const __m128i D = _mm_and_si128(C, mask);       // 0 r'
      _mm_storeu_si128((__m128i*)values, D);
      for (i = 0; i < SPAN; ++i) ++histo[values[i]];
    }
  }
  {
    const int left_over = tile_width & (SPAN - 1);
    if (left_over > 0) {
      VP8LCollectColorRedTransforms_C(argb + tile_width - left_over, stride,
                                      left_over, tile_height, green_to_red,
                                      histo);
    }
  }
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8LEncDspInitSSE41(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitSSE41(void) {
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_SSE41;
  VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_SSE41;
  VP8LCollectColorRedTransforms = CollectColorRedTransforms_SSE41;
}

#else  // !WEBP_USE_SSE41

WEBP_DSP_INIT_STUB(VP8LEncDspInitSSE41)

#endif  // WEBP_USE_SSE41
