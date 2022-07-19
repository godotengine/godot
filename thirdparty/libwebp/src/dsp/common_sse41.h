// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// SSE4 code common to several files.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#ifndef WEBP_DSP_COMMON_SSE41_H_
#define WEBP_DSP_COMMON_SSE41_H_

#ifdef __cplusplus
extern "C" {
#endif

#if defined(WEBP_USE_SSE41)
#include <smmintrin.h>

//------------------------------------------------------------------------------
// Channel mixing.
// Shuffles the input buffer as A0 0 0 A1 0 0 A2 ...
#define WEBP_SSE41_SHUFF(OUT, IN0, IN1)    \
  OUT##0 = _mm_shuffle_epi8(*IN0, shuff0); \
  OUT##1 = _mm_shuffle_epi8(*IN0, shuff1); \
  OUT##2 = _mm_shuffle_epi8(*IN0, shuff2); \
  OUT##3 = _mm_shuffle_epi8(*IN1, shuff0); \
  OUT##4 = _mm_shuffle_epi8(*IN1, shuff1); \
  OUT##5 = _mm_shuffle_epi8(*IN1, shuff2);

// Pack the planar buffers
// rrrr... rrrr... gggg... gggg... bbbb... bbbb....
// triplet by triplet in the output buffer rgb as rgbrgbrgbrgb ...
static WEBP_INLINE void VP8PlanarTo24b_SSE41(
    __m128i* const in0, __m128i* const in1, __m128i* const in2,
    __m128i* const in3, __m128i* const in4, __m128i* const in5) {
  __m128i R0, R1, R2, R3, R4, R5;
  __m128i G0, G1, G2, G3, G4, G5;
  __m128i B0, B1, B2, B3, B4, B5;

  // Process R.
  {
    const __m128i shuff0 = _mm_set_epi8(
        5, -1, -1, 4, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0);
    const __m128i shuff1 = _mm_set_epi8(
        -1, 10, -1, -1, 9, -1, -1, 8, -1, -1, 7, -1, -1, 6, -1, -1);
    const __m128i shuff2 = _mm_set_epi8(
     -1, -1, 15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1, -1, 11, -1);
    WEBP_SSE41_SHUFF(R, in0, in1)
  }

  // Process G.
  {
    // Same as before, just shifted to the left by one and including the right
    // padding.
    const __m128i shuff0 = _mm_set_epi8(
        -1, -1, 4, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0, -1);
    const __m128i shuff1 = _mm_set_epi8(
        10, -1, -1, 9, -1, -1, 8, -1, -1, 7, -1, -1, 6, -1, -1, 5);
    const __m128i shuff2 = _mm_set_epi8(
     -1, 15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1, -1, 11, -1, -1);
    WEBP_SSE41_SHUFF(G, in2, in3)
  }

  // Process B.
  {
    const __m128i shuff0 = _mm_set_epi8(
        -1, 4, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0, -1, -1);
    const __m128i shuff1 = _mm_set_epi8(
        -1, -1, 9, -1, -1, 8, -1, -1, 7, -1, -1, 6, -1, -1, 5, -1);
    const __m128i shuff2 = _mm_set_epi8(
      15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1, -1, 11, -1, -1, 10);
    WEBP_SSE41_SHUFF(B, in4, in5)
  }

  // OR the different channels.
  {
    const __m128i RG0 = _mm_or_si128(R0, G0);
    const __m128i RG1 = _mm_or_si128(R1, G1);
    const __m128i RG2 = _mm_or_si128(R2, G2);
    const __m128i RG3 = _mm_or_si128(R3, G3);
    const __m128i RG4 = _mm_or_si128(R4, G4);
    const __m128i RG5 = _mm_or_si128(R5, G5);
    *in0 = _mm_or_si128(RG0, B0);
    *in1 = _mm_or_si128(RG1, B1);
    *in2 = _mm_or_si128(RG2, B2);
    *in3 = _mm_or_si128(RG3, B3);
    *in4 = _mm_or_si128(RG4, B4);
    *in5 = _mm_or_si128(RG5, B5);
  }
}

#undef WEBP_SSE41_SHUFF

// Convert four packed four-channel buffers like argbargbargbargb... into the
// split channels aaaaa ... rrrr ... gggg .... bbbbb ......
static WEBP_INLINE void VP8L32bToPlanar_SSE41(__m128i* const in0,
                                              __m128i* const in1,
                                              __m128i* const in2,
                                              __m128i* const in3) {
  // aaaarrrrggggbbbb
  const __m128i shuff0 =
      _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  const __m128i A0 = _mm_shuffle_epi8(*in0, shuff0);
  const __m128i A1 = _mm_shuffle_epi8(*in1, shuff0);
  const __m128i A2 = _mm_shuffle_epi8(*in2, shuff0);
  const __m128i A3 = _mm_shuffle_epi8(*in3, shuff0);
  // A0A1R0R1
  // G0G1B0B1
  // A2A3R2R3
  // G0G1B0B1
  const __m128i B0 = _mm_unpacklo_epi32(A0, A1);
  const __m128i B1 = _mm_unpackhi_epi32(A0, A1);
  const __m128i B2 = _mm_unpacklo_epi32(A2, A3);
  const __m128i B3 = _mm_unpackhi_epi32(A2, A3);
  *in3 = _mm_unpacklo_epi64(B0, B2);
  *in2 = _mm_unpackhi_epi64(B0, B2);
  *in1 = _mm_unpacklo_epi64(B1, B3);
  *in0 = _mm_unpackhi_epi64(B1, B3);
}

#endif  // WEBP_USE_SSE41

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WEBP_DSP_COMMON_SSE41_H_
