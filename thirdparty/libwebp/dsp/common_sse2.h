// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// SSE2 code common to several files.
//
// Author: Vincent Rabaud (vrabaud@google.com)

#ifndef WEBP_DSP_COMMON_SSE2_H_
#define WEBP_DSP_COMMON_SSE2_H_

#ifdef __cplusplus
extern "C" {
#endif

#if defined(WEBP_USE_SSE2)

#include <emmintrin.h>

//------------------------------------------------------------------------------
// Quite useful macro for debugging. Left here for convenience.

#if 0
#include <stdio.h>
static WEBP_INLINE void PrintReg(const __m128i r, const char* const name,
                                 int size) {
  int n;
  union {
    __m128i r;
    uint8_t i8[16];
    uint16_t i16[8];
    uint32_t i32[4];
    uint64_t i64[2];
  } tmp;
  tmp.r = r;
  fprintf(stderr, "%s\t: ", name);
  if (size == 8) {
    for (n = 0; n < 16; ++n) fprintf(stderr, "%.2x ", tmp.i8[n]);
  } else if (size == 16) {
    for (n = 0; n < 8; ++n) fprintf(stderr, "%.4x ", tmp.i16[n]);
  } else if (size == 32) {
    for (n = 0; n < 4; ++n) fprintf(stderr, "%.8x ", tmp.i32[n]);
  } else {
    for (n = 0; n < 2; ++n) fprintf(stderr, "%.16lx ", tmp.i64[n]);
  }
  fprintf(stderr, "\n");
}
#endif

//------------------------------------------------------------------------------
// Math functions.

// Return the sum of all the 8b in the register.
static WEBP_INLINE int VP8HorizontalAdd8b(const __m128i* const a) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i sad8x2 = _mm_sad_epu8(*a, zero);
  // sum the two sads: sad8x2[0:1] + sad8x2[8:9]
  const __m128i sum = _mm_add_epi32(sad8x2, _mm_shuffle_epi32(sad8x2, 2));
  return _mm_cvtsi128_si32(sum);
}

// Transpose two 4x4 16b matrices horizontally stored in registers.
static WEBP_INLINE void VP8Transpose_2_4x4_16b(
    const __m128i* const in0, const __m128i* const in1,
    const __m128i* const in2, const __m128i* const in3, __m128i* const out0,
    __m128i* const out1, __m128i* const out2, __m128i* const out3) {
  // Transpose the two 4x4.
  // a00 a01 a02 a03   b00 b01 b02 b03
  // a10 a11 a12 a13   b10 b11 b12 b13
  // a20 a21 a22 a23   b20 b21 b22 b23
  // a30 a31 a32 a33   b30 b31 b32 b33
  const __m128i transpose0_0 = _mm_unpacklo_epi16(*in0, *in1);
  const __m128i transpose0_1 = _mm_unpacklo_epi16(*in2, *in3);
  const __m128i transpose0_2 = _mm_unpackhi_epi16(*in0, *in1);
  const __m128i transpose0_3 = _mm_unpackhi_epi16(*in2, *in3);
  // a00 a10 a01 a11   a02 a12 a03 a13
  // a20 a30 a21 a31   a22 a32 a23 a33
  // b00 b10 b01 b11   b02 b12 b03 b13
  // b20 b30 b21 b31   b22 b32 b23 b33
  const __m128i transpose1_0 = _mm_unpacklo_epi32(transpose0_0, transpose0_1);
  const __m128i transpose1_1 = _mm_unpacklo_epi32(transpose0_2, transpose0_3);
  const __m128i transpose1_2 = _mm_unpackhi_epi32(transpose0_0, transpose0_1);
  const __m128i transpose1_3 = _mm_unpackhi_epi32(transpose0_2, transpose0_3);
  // a00 a10 a20 a30 a01 a11 a21 a31
  // b00 b10 b20 b30 b01 b11 b21 b31
  // a02 a12 a22 a32 a03 a13 a23 a33
  // b02 b12 a22 b32 b03 b13 b23 b33
  *out0 = _mm_unpacklo_epi64(transpose1_0, transpose1_1);
  *out1 = _mm_unpackhi_epi64(transpose1_0, transpose1_1);
  *out2 = _mm_unpacklo_epi64(transpose1_2, transpose1_3);
  *out3 = _mm_unpackhi_epi64(transpose1_2, transpose1_3);
  // a00 a10 a20 a30   b00 b10 b20 b30
  // a01 a11 a21 a31   b01 b11 b21 b31
  // a02 a12 a22 a32   b02 b12 b22 b32
  // a03 a13 a23 a33   b03 b13 b23 b33
}

#endif  // WEBP_USE_SSE2

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WEBP_DSP_COMMON_SSE2_H_
