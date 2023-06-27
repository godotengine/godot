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

//------------------------------------------------------------------------------
// Channel mixing.

// Function used several times in VP8PlanarTo24b.
// It samples the in buffer as follows: one every two unsigned char is stored
// at the beginning of the buffer, while the other half is stored at the end.
#define VP8PlanarTo24bHelper(IN, OUT)                            \
  do {                                                           \
    const __m128i v_mask = _mm_set1_epi16(0x00ff);               \
    /* Take one every two upper 8b values.*/                     \
    (OUT##0) = _mm_packus_epi16(_mm_and_si128((IN##0), v_mask),  \
                                _mm_and_si128((IN##1), v_mask)); \
    (OUT##1) = _mm_packus_epi16(_mm_and_si128((IN##2), v_mask),  \
                                _mm_and_si128((IN##3), v_mask)); \
    (OUT##2) = _mm_packus_epi16(_mm_and_si128((IN##4), v_mask),  \
                                _mm_and_si128((IN##5), v_mask)); \
    /* Take one every two lower 8b values.*/                     \
    (OUT##3) = _mm_packus_epi16(_mm_srli_epi16((IN##0), 8),      \
                                _mm_srli_epi16((IN##1), 8));     \
    (OUT##4) = _mm_packus_epi16(_mm_srli_epi16((IN##2), 8),      \
                                _mm_srli_epi16((IN##3), 8));     \
    (OUT##5) = _mm_packus_epi16(_mm_srli_epi16((IN##4), 8),      \
                                _mm_srli_epi16((IN##5), 8));     \
  } while (0)

// Pack the planar buffers
// rrrr... rrrr... gggg... gggg... bbbb... bbbb....
// triplet by triplet in the output buffer rgb as rgbrgbrgbrgb ...
static WEBP_INLINE void VP8PlanarTo24b_SSE2(
    __m128i* const in0, __m128i* const in1, __m128i* const in2,
    __m128i* const in3, __m128i* const in4, __m128i* const in5) {
  // The input is 6 registers of sixteen 8b but for the sake of explanation,
  // let's take 6 registers of four 8b values.
  // To pack, we will keep taking one every two 8b integer and move it
  // around as follows:
  // Input:
  //   r0r1r2r3 | r4r5r6r7 | g0g1g2g3 | g4g5g6g7 | b0b1b2b3 | b4b5b6b7
  // Split the 6 registers in two sets of 3 registers: the first set as the even
  // 8b bytes, the second the odd ones:
  //   r0r2r4r6 | g0g2g4g6 | b0b2b4b6 | r1r3r5r7 | g1g3g5g7 | b1b3b5b7
  // Repeat the same permutations twice more:
  //   r0r4g0g4 | b0b4r1r5 | g1g5b1b5 | r2r6g2g6 | b2b6r3r7 | g3g7b3b7
  //   r0g0b0r1 | g1b1r2g2 | b2r3g3b3 | r4g4b4r5 | g5b5r6g6 | b6r7g7b7
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  VP8PlanarTo24bHelper(*in, tmp);
  VP8PlanarTo24bHelper(tmp, *in);
  VP8PlanarTo24bHelper(*in, tmp);
  // We need to do it two more times than the example as we have sixteen bytes.
  {
    __m128i out0, out1, out2, out3, out4, out5;
    VP8PlanarTo24bHelper(tmp, out);
    VP8PlanarTo24bHelper(out, *in);
  }
}

#undef VP8PlanarTo24bHelper

// Convert four packed four-channel buffers like argbargbargbargb... into the
// split channels aaaaa ... rrrr ... gggg .... bbbbb ......
static WEBP_INLINE void VP8L32bToPlanar_SSE2(__m128i* const in0,
                                             __m128i* const in1,
                                             __m128i* const in2,
                                             __m128i* const in3) {
  // Column-wise transpose.
  const __m128i A0 = _mm_unpacklo_epi8(*in0, *in1);
  const __m128i A1 = _mm_unpackhi_epi8(*in0, *in1);
  const __m128i A2 = _mm_unpacklo_epi8(*in2, *in3);
  const __m128i A3 = _mm_unpackhi_epi8(*in2, *in3);
  const __m128i B0 = _mm_unpacklo_epi8(A0, A1);
  const __m128i B1 = _mm_unpackhi_epi8(A0, A1);
  const __m128i B2 = _mm_unpacklo_epi8(A2, A3);
  const __m128i B3 = _mm_unpackhi_epi8(A2, A3);
  // C0 = g7 g6 ... g1 g0 | b7 b6 ... b1 b0
  // C1 = a7 a6 ... a1 a0 | r7 r6 ... r1 r0
  const __m128i C0 = _mm_unpacklo_epi8(B0, B1);
  const __m128i C1 = _mm_unpackhi_epi8(B0, B1);
  const __m128i C2 = _mm_unpacklo_epi8(B2, B3);
  const __m128i C3 = _mm_unpackhi_epi8(B2, B3);
  // Gather the channels.
  *in0 = _mm_unpackhi_epi64(C1, C3);
  *in1 = _mm_unpacklo_epi64(C1, C3);
  *in2 = _mm_unpackhi_epi64(C0, C2);
  *in3 = _mm_unpacklo_epi64(C0, C2);
}

#endif  // WEBP_USE_SSE2

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WEBP_DSP_COMMON_SSE2_H_
