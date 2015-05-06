// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// SSE2 version of speed-critical encoding functions.
//
// Author: Christian Duvivier (cduvivier@google.com)

#include "./dsp.h"

#if defined(WEBP_USE_SSE2)
#include <stdlib.h>  // for abs()
#include <emmintrin.h>

#include "../enc/vp8enci.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Compute susceptibility based on DCT-coeff histograms:
// the higher, the "easier" the macroblock is to compress.

static int CollectHistogramSSE2(const uint8_t* ref, const uint8_t* pred,
                                int start_block, int end_block) {
  int histo[MAX_COEFF_THRESH + 1] = { 0 };
  int16_t out[16];
  int j, k;
  const __m128i max_coeff_thresh = _mm_set1_epi16(MAX_COEFF_THRESH);
  for (j = start_block; j < end_block; ++j) {
    VP8FTransform(ref + VP8DspScan[j], pred + VP8DspScan[j], out);

    // Convert coefficients to bin (within out[]).
    {
      // Load.
      const __m128i out0 = _mm_loadu_si128((__m128i*)&out[0]);
      const __m128i out1 = _mm_loadu_si128((__m128i*)&out[8]);
      // sign(out) = out >> 15  (0x0000 if positive, 0xffff if negative)
      const __m128i sign0 = _mm_srai_epi16(out0, 15);
      const __m128i sign1 = _mm_srai_epi16(out1, 15);
      // abs(out) = (out ^ sign) - sign
      const __m128i xor0 = _mm_xor_si128(out0, sign0);
      const __m128i xor1 = _mm_xor_si128(out1, sign1);
      const __m128i abs0 = _mm_sub_epi16(xor0, sign0);
      const __m128i abs1 = _mm_sub_epi16(xor1, sign1);
      // v = abs(out) >> 2
      const __m128i v0 = _mm_srai_epi16(abs0, 2);
      const __m128i v1 = _mm_srai_epi16(abs1, 2);
      // bin = min(v, MAX_COEFF_THRESH)
      const __m128i bin0 = _mm_min_epi16(v0, max_coeff_thresh);
      const __m128i bin1 = _mm_min_epi16(v1, max_coeff_thresh);
      // Store.
      _mm_storeu_si128((__m128i*)&out[0], bin0);
      _mm_storeu_si128((__m128i*)&out[8], bin1);
    }

    // Use bin to update histogram.
    for (k = 0; k < 16; ++k) {
      histo[out[k]]++;
    }
  }

  return VP8GetAlpha(histo);
}

//------------------------------------------------------------------------------
// Transforms (Paragraph 14.4)

// Does one or two inverse transforms.
static void ITransformSSE2(const uint8_t* ref, const int16_t* in, uint8_t* dst,
                           int do_two) {
  // This implementation makes use of 16-bit fixed point versions of two
  // multiply constants:
  //    K1 = sqrt(2) * cos (pi/8) ~= 85627 / 2^16
  //    K2 = sqrt(2) * sin (pi/8) ~= 35468 / 2^16
  //
  // To be able to use signed 16-bit integers, we use the following trick to
  // have constants within range:
  // - Associated constants are obtained by subtracting the 16-bit fixed point
  //   version of one:
  //      k = K - (1 << 16)  =>  K = k + (1 << 16)
  //      K1 = 85267  =>  k1 =  20091
  //      K2 = 35468  =>  k2 = -30068
  // - The multiplication of a variable by a constant become the sum of the
  //   variable and the multiplication of that variable by the associated
  //   constant:
  //      (x * K) >> 16 = (x * (k + (1 << 16))) >> 16 = ((x * k ) >> 16) + x
  const __m128i k1 = _mm_set1_epi16(20091);
  const __m128i k2 = _mm_set1_epi16(-30068);
  __m128i T0, T1, T2, T3;

  // Load and concatenate the transform coefficients (we'll do two inverse
  // transforms in parallel). In the case of only one inverse transform, the
  // second half of the vectors will just contain random value we'll never
  // use nor store.
  __m128i in0, in1, in2, in3;
  {
    in0 = _mm_loadl_epi64((__m128i*)&in[0]);
    in1 = _mm_loadl_epi64((__m128i*)&in[4]);
    in2 = _mm_loadl_epi64((__m128i*)&in[8]);
    in3 = _mm_loadl_epi64((__m128i*)&in[12]);
    // a00 a10 a20 a30   x x x x
    // a01 a11 a21 a31   x x x x
    // a02 a12 a22 a32   x x x x
    // a03 a13 a23 a33   x x x x
    if (do_two) {
      const __m128i inB0 = _mm_loadl_epi64((__m128i*)&in[16]);
      const __m128i inB1 = _mm_loadl_epi64((__m128i*)&in[20]);
      const __m128i inB2 = _mm_loadl_epi64((__m128i*)&in[24]);
      const __m128i inB3 = _mm_loadl_epi64((__m128i*)&in[28]);
      in0 = _mm_unpacklo_epi64(in0, inB0);
      in1 = _mm_unpacklo_epi64(in1, inB1);
      in2 = _mm_unpacklo_epi64(in2, inB2);
      in3 = _mm_unpacklo_epi64(in3, inB3);
      // a00 a10 a20 a30   b00 b10 b20 b30
      // a01 a11 a21 a31   b01 b11 b21 b31
      // a02 a12 a22 a32   b02 b12 b22 b32
      // a03 a13 a23 a33   b03 b13 b23 b33
    }
  }

  // Vertical pass and subsequent transpose.
  {
    // First pass, c and d calculations are longer because of the "trick"
    // multiplications.
    const __m128i a = _mm_add_epi16(in0, in2);
    const __m128i b = _mm_sub_epi16(in0, in2);
    // c = MUL(in1, K2) - MUL(in3, K1) = MUL(in1, k2) - MUL(in3, k1) + in1 - in3
    const __m128i c1 = _mm_mulhi_epi16(in1, k2);
    const __m128i c2 = _mm_mulhi_epi16(in3, k1);
    const __m128i c3 = _mm_sub_epi16(in1, in3);
    const __m128i c4 = _mm_sub_epi16(c1, c2);
    const __m128i c = _mm_add_epi16(c3, c4);
    // d = MUL(in1, K1) + MUL(in3, K2) = MUL(in1, k1) + MUL(in3, k2) + in1 + in3
    const __m128i d1 = _mm_mulhi_epi16(in1, k1);
    const __m128i d2 = _mm_mulhi_epi16(in3, k2);
    const __m128i d3 = _mm_add_epi16(in1, in3);
    const __m128i d4 = _mm_add_epi16(d1, d2);
    const __m128i d = _mm_add_epi16(d3, d4);

    // Second pass.
    const __m128i tmp0 = _mm_add_epi16(a, d);
    const __m128i tmp1 = _mm_add_epi16(b, c);
    const __m128i tmp2 = _mm_sub_epi16(b, c);
    const __m128i tmp3 = _mm_sub_epi16(a, d);

    // Transpose the two 4x4.
    // a00 a01 a02 a03   b00 b01 b02 b03
    // a10 a11 a12 a13   b10 b11 b12 b13
    // a20 a21 a22 a23   b20 b21 b22 b23
    // a30 a31 a32 a33   b30 b31 b32 b33
    const __m128i transpose0_0 = _mm_unpacklo_epi16(tmp0, tmp1);
    const __m128i transpose0_1 = _mm_unpacklo_epi16(tmp2, tmp3);
    const __m128i transpose0_2 = _mm_unpackhi_epi16(tmp0, tmp1);
    const __m128i transpose0_3 = _mm_unpackhi_epi16(tmp2, tmp3);
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
    T0 = _mm_unpacklo_epi64(transpose1_0, transpose1_1);
    T1 = _mm_unpackhi_epi64(transpose1_0, transpose1_1);
    T2 = _mm_unpacklo_epi64(transpose1_2, transpose1_3);
    T3 = _mm_unpackhi_epi64(transpose1_2, transpose1_3);
    // a00 a10 a20 a30   b00 b10 b20 b30
    // a01 a11 a21 a31   b01 b11 b21 b31
    // a02 a12 a22 a32   b02 b12 b22 b32
    // a03 a13 a23 a33   b03 b13 b23 b33
  }

  // Horizontal pass and subsequent transpose.
  {
    // First pass, c and d calculations are longer because of the "trick"
    // multiplications.
    const __m128i four = _mm_set1_epi16(4);
    const __m128i dc = _mm_add_epi16(T0, four);
    const __m128i a =  _mm_add_epi16(dc, T2);
    const __m128i b =  _mm_sub_epi16(dc, T2);
    // c = MUL(T1, K2) - MUL(T3, K1) = MUL(T1, k2) - MUL(T3, k1) + T1 - T3
    const __m128i c1 = _mm_mulhi_epi16(T1, k2);
    const __m128i c2 = _mm_mulhi_epi16(T3, k1);
    const __m128i c3 = _mm_sub_epi16(T1, T3);
    const __m128i c4 = _mm_sub_epi16(c1, c2);
    const __m128i c = _mm_add_epi16(c3, c4);
    // d = MUL(T1, K1) + MUL(T3, K2) = MUL(T1, k1) + MUL(T3, k2) + T1 + T3
    const __m128i d1 = _mm_mulhi_epi16(T1, k1);
    const __m128i d2 = _mm_mulhi_epi16(T3, k2);
    const __m128i d3 = _mm_add_epi16(T1, T3);
    const __m128i d4 = _mm_add_epi16(d1, d2);
    const __m128i d = _mm_add_epi16(d3, d4);

    // Second pass.
    const __m128i tmp0 = _mm_add_epi16(a, d);
    const __m128i tmp1 = _mm_add_epi16(b, c);
    const __m128i tmp2 = _mm_sub_epi16(b, c);
    const __m128i tmp3 = _mm_sub_epi16(a, d);
    const __m128i shifted0 = _mm_srai_epi16(tmp0, 3);
    const __m128i shifted1 = _mm_srai_epi16(tmp1, 3);
    const __m128i shifted2 = _mm_srai_epi16(tmp2, 3);
    const __m128i shifted3 = _mm_srai_epi16(tmp3, 3);

    // Transpose the two 4x4.
    // a00 a01 a02 a03   b00 b01 b02 b03
    // a10 a11 a12 a13   b10 b11 b12 b13
    // a20 a21 a22 a23   b20 b21 b22 b23
    // a30 a31 a32 a33   b30 b31 b32 b33
    const __m128i transpose0_0 = _mm_unpacklo_epi16(shifted0, shifted1);
    const __m128i transpose0_1 = _mm_unpacklo_epi16(shifted2, shifted3);
    const __m128i transpose0_2 = _mm_unpackhi_epi16(shifted0, shifted1);
    const __m128i transpose0_3 = _mm_unpackhi_epi16(shifted2, shifted3);
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
    T0 = _mm_unpacklo_epi64(transpose1_0, transpose1_1);
    T1 = _mm_unpackhi_epi64(transpose1_0, transpose1_1);
    T2 = _mm_unpacklo_epi64(transpose1_2, transpose1_3);
    T3 = _mm_unpackhi_epi64(transpose1_2, transpose1_3);
    // a00 a10 a20 a30   b00 b10 b20 b30
    // a01 a11 a21 a31   b01 b11 b21 b31
    // a02 a12 a22 a32   b02 b12 b22 b32
    // a03 a13 a23 a33   b03 b13 b23 b33
  }

  // Add inverse transform to 'ref' and store.
  {
    const __m128i zero = _mm_set1_epi16(0);
    // Load the reference(s).
    __m128i ref0, ref1, ref2, ref3;
    if (do_two) {
      // Load eight bytes/pixels per line.
      ref0 = _mm_loadl_epi64((__m128i*)&ref[0 * BPS]);
      ref1 = _mm_loadl_epi64((__m128i*)&ref[1 * BPS]);
      ref2 = _mm_loadl_epi64((__m128i*)&ref[2 * BPS]);
      ref3 = _mm_loadl_epi64((__m128i*)&ref[3 * BPS]);
    } else {
      // Load four bytes/pixels per line.
      ref0 = _mm_cvtsi32_si128(*(int*)&ref[0 * BPS]);
      ref1 = _mm_cvtsi32_si128(*(int*)&ref[1 * BPS]);
      ref2 = _mm_cvtsi32_si128(*(int*)&ref[2 * BPS]);
      ref3 = _mm_cvtsi32_si128(*(int*)&ref[3 * BPS]);
    }
    // Convert to 16b.
    ref0 = _mm_unpacklo_epi8(ref0, zero);
    ref1 = _mm_unpacklo_epi8(ref1, zero);
    ref2 = _mm_unpacklo_epi8(ref2, zero);
    ref3 = _mm_unpacklo_epi8(ref3, zero);
    // Add the inverse transform(s).
    ref0 = _mm_add_epi16(ref0, T0);
    ref1 = _mm_add_epi16(ref1, T1);
    ref2 = _mm_add_epi16(ref2, T2);
    ref3 = _mm_add_epi16(ref3, T3);
    // Unsigned saturate to 8b.
    ref0 = _mm_packus_epi16(ref0, ref0);
    ref1 = _mm_packus_epi16(ref1, ref1);
    ref2 = _mm_packus_epi16(ref2, ref2);
    ref3 = _mm_packus_epi16(ref3, ref3);
    // Store the results.
    if (do_two) {
      // Store eight bytes/pixels per line.
      _mm_storel_epi64((__m128i*)&dst[0 * BPS], ref0);
      _mm_storel_epi64((__m128i*)&dst[1 * BPS], ref1);
      _mm_storel_epi64((__m128i*)&dst[2 * BPS], ref2);
      _mm_storel_epi64((__m128i*)&dst[3 * BPS], ref3);
    } else {
      // Store four bytes/pixels per line.
      *((int32_t *)&dst[0 * BPS]) = _mm_cvtsi128_si32(ref0);
      *((int32_t *)&dst[1 * BPS]) = _mm_cvtsi128_si32(ref1);
      *((int32_t *)&dst[2 * BPS]) = _mm_cvtsi128_si32(ref2);
      *((int32_t *)&dst[3 * BPS]) = _mm_cvtsi128_si32(ref3);
    }
  }
}

static void FTransformSSE2(const uint8_t* src, const uint8_t* ref,
                           int16_t* out) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i seven = _mm_set1_epi16(7);
  const __m128i k7500 = _mm_set1_epi32(7500);
  const __m128i k14500 = _mm_set1_epi32(14500);
  const __m128i k51000 = _mm_set1_epi32(51000);
  const __m128i k12000_plus_one = _mm_set1_epi32(12000 + (1 << 16));
  const __m128i k5352_2217 = _mm_set_epi16(5352,  2217, 5352,  2217,
                                           5352,  2217, 5352,  2217);
  const __m128i k2217_5352 = _mm_set_epi16(2217, -5352, 2217, -5352,
                                           2217, -5352, 2217, -5352);

  __m128i v01, v32;

  // Difference between src and ref and initial transpose.
  {
    // Load src and convert to 16b.
    const __m128i src0 = _mm_loadl_epi64((__m128i*)&src[0 * BPS]);
    const __m128i src1 = _mm_loadl_epi64((__m128i*)&src[1 * BPS]);
    const __m128i src2 = _mm_loadl_epi64((__m128i*)&src[2 * BPS]);
    const __m128i src3 = _mm_loadl_epi64((__m128i*)&src[3 * BPS]);
    const __m128i src_0 = _mm_unpacklo_epi8(src0, zero);
    const __m128i src_1 = _mm_unpacklo_epi8(src1, zero);
    const __m128i src_2 = _mm_unpacklo_epi8(src2, zero);
    const __m128i src_3 = _mm_unpacklo_epi8(src3, zero);
    // Load ref and convert to 16b.
    const __m128i ref0 = _mm_loadl_epi64((__m128i*)&ref[0 * BPS]);
    const __m128i ref1 = _mm_loadl_epi64((__m128i*)&ref[1 * BPS]);
    const __m128i ref2 = _mm_loadl_epi64((__m128i*)&ref[2 * BPS]);
    const __m128i ref3 = _mm_loadl_epi64((__m128i*)&ref[3 * BPS]);
    const __m128i ref_0 = _mm_unpacklo_epi8(ref0, zero);
    const __m128i ref_1 = _mm_unpacklo_epi8(ref1, zero);
    const __m128i ref_2 = _mm_unpacklo_epi8(ref2, zero);
    const __m128i ref_3 = _mm_unpacklo_epi8(ref3, zero);
    // Compute difference.
    const __m128i diff0 = _mm_sub_epi16(src_0, ref_0);
    const __m128i diff1 = _mm_sub_epi16(src_1, ref_1);
    const __m128i diff2 = _mm_sub_epi16(src_2, ref_2);
    const __m128i diff3 = _mm_sub_epi16(src_3, ref_3);

    // Transpose.
    // 00 01 02 03   0 0 0 0
    // 10 11 12 13   0 0 0 0
    // 20 21 22 23   0 0 0 0
    // 30 31 32 33   0 0 0 0
    const __m128i transpose0_0 = _mm_unpacklo_epi16(diff0, diff1);
    const __m128i transpose0_1 = _mm_unpacklo_epi16(diff2, diff3);
    // 00 10 01 11   02 12 03 13
    // 20 30 21 31   22 32 23 33
    const __m128i v23 = _mm_unpackhi_epi32(transpose0_0, transpose0_1);
    v01 = _mm_unpacklo_epi32(transpose0_0, transpose0_1);
    v32 = _mm_shuffle_epi32(v23, _MM_SHUFFLE(1, 0, 3, 2));
    // a02 a12 a22 a32   a03 a13 a23 a33
    // a00 a10 a20 a30   a01 a11 a21 a31
    // a03 a13 a23 a33   a02 a12 a22 a32
  }

  // First pass and subsequent transpose.
  {
    // Same operations are done on the (0,3) and (1,2) pairs.
    // b0 = (a0 + a3) << 3
    // b1 = (a1 + a2) << 3
    // b3 = (a0 - a3) << 3
    // b2 = (a1 - a2) << 3
    const __m128i a01 = _mm_add_epi16(v01, v32);
    const __m128i a32 = _mm_sub_epi16(v01, v32);
    const __m128i b01 = _mm_slli_epi16(a01, 3);
    const __m128i b32 = _mm_slli_epi16(a32, 3);
    const __m128i b11 = _mm_unpackhi_epi64(b01, b01);
    const __m128i b22 = _mm_unpackhi_epi64(b32, b32);

    // e0 = b0 + b1
    // e2 = b0 - b1
    const __m128i e0 = _mm_add_epi16(b01, b11);
    const __m128i e2 = _mm_sub_epi16(b01, b11);
    const __m128i e02 = _mm_unpacklo_epi64(e0, e2);

    // e1 = (b3 * 5352 + b2 * 2217 + 14500) >> 12
    // e3 = (b3 * 2217 - b2 * 5352 +  7500) >> 12
    const __m128i b23 = _mm_unpacklo_epi16(b22, b32);
    const __m128i c1 = _mm_madd_epi16(b23, k5352_2217);
    const __m128i c3 = _mm_madd_epi16(b23, k2217_5352);
    const __m128i d1 = _mm_add_epi32(c1, k14500);
    const __m128i d3 = _mm_add_epi32(c3, k7500);
    const __m128i e1 = _mm_srai_epi32(d1, 12);
    const __m128i e3 = _mm_srai_epi32(d3, 12);
    const __m128i e13 = _mm_packs_epi32(e1, e3);

    // Transpose.
    // 00 01 02 03  20 21 22 23
    // 10 11 12 13  30 31 32 33
    const __m128i transpose0_0 = _mm_unpacklo_epi16(e02, e13);
    const __m128i transpose0_1 = _mm_unpackhi_epi16(e02, e13);
    // 00 10 01 11   02 12 03 13
    // 20 30 21 31   22 32 23 33
    const __m128i v23 = _mm_unpackhi_epi32(transpose0_0, transpose0_1);
    v01 = _mm_unpacklo_epi32(transpose0_0, transpose0_1);
    v32 = _mm_shuffle_epi32(v23, _MM_SHUFFLE(1, 0, 3, 2));
    // 02 12 22 32   03 13 23 33
    // 00 10 20 30   01 11 21 31
    // 03 13 23 33   02 12 22 32
  }

  // Second pass
  {
    // Same operations are done on the (0,3) and (1,2) pairs.
    // a0 = v0 + v3
    // a1 = v1 + v2
    // a3 = v0 - v3
    // a2 = v1 - v2
    const __m128i a01 = _mm_add_epi16(v01, v32);
    const __m128i a32 = _mm_sub_epi16(v01, v32);
    const __m128i a11 = _mm_unpackhi_epi64(a01, a01);
    const __m128i a22 = _mm_unpackhi_epi64(a32, a32);

    // d0 = (a0 + a1 + 7) >> 4;
    // d2 = (a0 - a1 + 7) >> 4;
    const __m128i b0 = _mm_add_epi16(a01, a11);
    const __m128i b2 = _mm_sub_epi16(a01, a11);
    const __m128i c0 = _mm_add_epi16(b0, seven);
    const __m128i c2 = _mm_add_epi16(b2, seven);
    const __m128i d0 = _mm_srai_epi16(c0, 4);
    const __m128i d2 = _mm_srai_epi16(c2, 4);

    // f1 = ((b3 * 5352 + b2 * 2217 + 12000) >> 16)
    // f3 = ((b3 * 2217 - b2 * 5352 + 51000) >> 16)
    const __m128i b23 = _mm_unpacklo_epi16(a22, a32);
    const __m128i c1 = _mm_madd_epi16(b23, k5352_2217);
    const __m128i c3 = _mm_madd_epi16(b23, k2217_5352);
    const __m128i d1 = _mm_add_epi32(c1, k12000_plus_one);
    const __m128i d3 = _mm_add_epi32(c3, k51000);
    const __m128i e1 = _mm_srai_epi32(d1, 16);
    const __m128i e3 = _mm_srai_epi32(d3, 16);
    const __m128i f1 = _mm_packs_epi32(e1, e1);
    const __m128i f3 = _mm_packs_epi32(e3, e3);
    // f1 = f1 + (a3 != 0);
    // The compare will return (0xffff, 0) for (==0, !=0). To turn that into the
    // desired (0, 1), we add one earlier through k12000_plus_one.
    const __m128i g1 = _mm_add_epi16(f1, _mm_cmpeq_epi16(a32, zero));

    _mm_storel_epi64((__m128i*)&out[ 0], d0);
    _mm_storel_epi64((__m128i*)&out[ 4], g1);
    _mm_storel_epi64((__m128i*)&out[ 8], d2);
    _mm_storel_epi64((__m128i*)&out[12], f3);
  }
}

//------------------------------------------------------------------------------
// Metric

static int SSE4x4SSE2(const uint8_t* a, const uint8_t* b) {
  const __m128i zero = _mm_set1_epi16(0);

  // Load values.
  const __m128i a0 = _mm_loadl_epi64((__m128i*)&a[BPS * 0]);
  const __m128i a1 = _mm_loadl_epi64((__m128i*)&a[BPS * 1]);
  const __m128i a2 = _mm_loadl_epi64((__m128i*)&a[BPS * 2]);
  const __m128i a3 = _mm_loadl_epi64((__m128i*)&a[BPS * 3]);
  const __m128i b0 = _mm_loadl_epi64((__m128i*)&b[BPS * 0]);
  const __m128i b1 = _mm_loadl_epi64((__m128i*)&b[BPS * 1]);
  const __m128i b2 = _mm_loadl_epi64((__m128i*)&b[BPS * 2]);
  const __m128i b3 = _mm_loadl_epi64((__m128i*)&b[BPS * 3]);

  // Combine pair of lines and convert to 16b.
  const __m128i a01 = _mm_unpacklo_epi32(a0, a1);
  const __m128i a23 = _mm_unpacklo_epi32(a2, a3);
  const __m128i b01 = _mm_unpacklo_epi32(b0, b1);
  const __m128i b23 = _mm_unpacklo_epi32(b2, b3);
  const __m128i a01s = _mm_unpacklo_epi8(a01, zero);
  const __m128i a23s = _mm_unpacklo_epi8(a23, zero);
  const __m128i b01s = _mm_unpacklo_epi8(b01, zero);
  const __m128i b23s = _mm_unpacklo_epi8(b23, zero);

  // Compute differences; (a-b)^2 = (abs(a-b))^2 = (sat8(a-b) + sat8(b-a))^2
  // TODO(cduvivier): Dissassemble and figure out why this is fastest. We don't
  //                  need absolute values, there is no need to do calculation
  //                  in 8bit as we are already in 16bit, ... Yet this is what
  //                  benchmarks the fastest!
  const __m128i d0 = _mm_subs_epu8(a01s, b01s);
  const __m128i d1 = _mm_subs_epu8(b01s, a01s);
  const __m128i d2 = _mm_subs_epu8(a23s, b23s);
  const __m128i d3 = _mm_subs_epu8(b23s, a23s);

  // Square and add them all together.
  const __m128i madd0 = _mm_madd_epi16(d0, d0);
  const __m128i madd1 = _mm_madd_epi16(d1, d1);
  const __m128i madd2 = _mm_madd_epi16(d2, d2);
  const __m128i madd3 = _mm_madd_epi16(d3, d3);
  const __m128i sum0 = _mm_add_epi32(madd0, madd1);
  const __m128i sum1 = _mm_add_epi32(madd2, madd3);
  const __m128i sum2 = _mm_add_epi32(sum0, sum1);
  int32_t tmp[4];
  _mm_storeu_si128((__m128i*)tmp, sum2);
  return (tmp[3] + tmp[2] + tmp[1] + tmp[0]);
}

//------------------------------------------------------------------------------
// Texture distortion
//
// We try to match the spectral content (weighted) between source and
// reconstructed samples.

// Hadamard transform
// Returns the difference between the weighted sum of the absolute value of
// transformed coefficients.
static int TTransformSSE2(const uint8_t* inA, const uint8_t* inB,
                          const uint16_t* const w) {
  int32_t sum[4];
  __m128i tmp_0, tmp_1, tmp_2, tmp_3;
  const __m128i zero = _mm_setzero_si128();
  const __m128i one = _mm_set1_epi16(1);
  const __m128i three = _mm_set1_epi16(3);

  // Load, combine and tranpose inputs.
  {
    const __m128i inA_0 = _mm_loadl_epi64((__m128i*)&inA[BPS * 0]);
    const __m128i inA_1 = _mm_loadl_epi64((__m128i*)&inA[BPS * 1]);
    const __m128i inA_2 = _mm_loadl_epi64((__m128i*)&inA[BPS * 2]);
    const __m128i inA_3 = _mm_loadl_epi64((__m128i*)&inA[BPS * 3]);
    const __m128i inB_0 = _mm_loadl_epi64((__m128i*)&inB[BPS * 0]);
    const __m128i inB_1 = _mm_loadl_epi64((__m128i*)&inB[BPS * 1]);
    const __m128i inB_2 = _mm_loadl_epi64((__m128i*)&inB[BPS * 2]);
    const __m128i inB_3 = _mm_loadl_epi64((__m128i*)&inB[BPS * 3]);

    // Combine inA and inB (we'll do two transforms in parallel).
    const __m128i inAB_0 = _mm_unpacklo_epi8(inA_0, inB_0);
    const __m128i inAB_1 = _mm_unpacklo_epi8(inA_1, inB_1);
    const __m128i inAB_2 = _mm_unpacklo_epi8(inA_2, inB_2);
    const __m128i inAB_3 = _mm_unpacklo_epi8(inA_3, inB_3);
    // a00 b00 a01 b01 a02 b03 a03 b03   0 0 0 0 0 0 0 0
    // a10 b10 a11 b11 a12 b12 a13 b13   0 0 0 0 0 0 0 0
    // a20 b20 a21 b21 a22 b22 a23 b23   0 0 0 0 0 0 0 0
    // a30 b30 a31 b31 a32 b32 a33 b33   0 0 0 0 0 0 0 0

    // Transpose the two 4x4, discarding the filling zeroes.
    const __m128i transpose0_0 = _mm_unpacklo_epi8(inAB_0, inAB_2);
    const __m128i transpose0_1 = _mm_unpacklo_epi8(inAB_1, inAB_3);
    // a00 a20  b00 b20  a01 a21  b01 b21  a02 a22  b02 b22  a03 a23  b03 b23
    // a10 a30  b10 b30  a11 a31  b11 b31  a12 a32  b12 b32  a13 a33  b13 b33
    const __m128i transpose1_0 = _mm_unpacklo_epi8(transpose0_0, transpose0_1);
    const __m128i transpose1_1 = _mm_unpackhi_epi8(transpose0_0, transpose0_1);
    // a00 a10 a20 a30  b00 b10 b20 b30  a01 a11 a21 a31  b01 b11 b21 b31
    // a02 a12 a22 a32  b02 b12 b22 b32  a03 a13 a23 a33  b03 b13 b23 b33

    // Convert to 16b.
    tmp_0 = _mm_unpacklo_epi8(transpose1_0, zero);
    tmp_1 = _mm_unpackhi_epi8(transpose1_0, zero);
    tmp_2 = _mm_unpacklo_epi8(transpose1_1, zero);
    tmp_3 = _mm_unpackhi_epi8(transpose1_1, zero);
    // a00 a10 a20 a30   b00 b10 b20 b30
    // a01 a11 a21 a31   b01 b11 b21 b31
    // a02 a12 a22 a32   b02 b12 b22 b32
    // a03 a13 a23 a33   b03 b13 b23 b33
  }

  // Horizontal pass and subsequent transpose.
  {
    // Calculate a and b (two 4x4 at once).
    const __m128i a0 = _mm_slli_epi16(_mm_add_epi16(tmp_0, tmp_2), 2);
    const __m128i a1 = _mm_slli_epi16(_mm_add_epi16(tmp_1, tmp_3), 2);
    const __m128i a2 = _mm_slli_epi16(_mm_sub_epi16(tmp_1, tmp_3), 2);
    const __m128i a3 = _mm_slli_epi16(_mm_sub_epi16(tmp_0, tmp_2), 2);
    // b0_extra = (a0 != 0);
    const __m128i b0_extra = _mm_andnot_si128(_mm_cmpeq_epi16 (a0, zero), one);
    const __m128i b0_base = _mm_add_epi16(a0, a1);
    const __m128i b1 = _mm_add_epi16(a3, a2);
    const __m128i b2 = _mm_sub_epi16(a3, a2);
    const __m128i b3 = _mm_sub_epi16(a0, a1);
    const __m128i b0 = _mm_add_epi16(b0_base, b0_extra);
    // a00 a01 a02 a03   b00 b01 b02 b03
    // a10 a11 a12 a13   b10 b11 b12 b13
    // a20 a21 a22 a23   b20 b21 b22 b23
    // a30 a31 a32 a33   b30 b31 b32 b33

    // Transpose the two 4x4.
    const __m128i transpose0_0 = _mm_unpacklo_epi16(b0, b1);
    const __m128i transpose0_1 = _mm_unpacklo_epi16(b2, b3);
    const __m128i transpose0_2 = _mm_unpackhi_epi16(b0, b1);
    const __m128i transpose0_3 = _mm_unpackhi_epi16(b2, b3);
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
    tmp_0 = _mm_unpacklo_epi64(transpose1_0, transpose1_1);
    tmp_1 = _mm_unpackhi_epi64(transpose1_0, transpose1_1);
    tmp_2 = _mm_unpacklo_epi64(transpose1_2, transpose1_3);
    tmp_3 = _mm_unpackhi_epi64(transpose1_2, transpose1_3);
    // a00 a10 a20 a30   b00 b10 b20 b30
    // a01 a11 a21 a31   b01 b11 b21 b31
    // a02 a12 a22 a32   b02 b12 b22 b32
    // a03 a13 a23 a33   b03 b13 b23 b33
  }

  // Vertical pass and difference of weighted sums.
  {
    // Load all inputs.
    // TODO(cduvivier): Make variable declarations and allocations aligned so
    //                  we can use _mm_load_si128 instead of _mm_loadu_si128.
    const __m128i w_0 = _mm_loadu_si128((__m128i*)&w[0]);
    const __m128i w_8 = _mm_loadu_si128((__m128i*)&w[8]);

    // Calculate a and b (two 4x4 at once).
    const __m128i a0 = _mm_add_epi16(tmp_0, tmp_2);
    const __m128i a1 = _mm_add_epi16(tmp_1, tmp_3);
    const __m128i a2 = _mm_sub_epi16(tmp_1, tmp_3);
    const __m128i a3 = _mm_sub_epi16(tmp_0, tmp_2);
    const __m128i b0 = _mm_add_epi16(a0, a1);
    const __m128i b1 = _mm_add_epi16(a3, a2);
    const __m128i b2 = _mm_sub_epi16(a3, a2);
    const __m128i b3 = _mm_sub_epi16(a0, a1);

    // Separate the transforms of inA and inB.
    __m128i A_b0 = _mm_unpacklo_epi64(b0, b1);
    __m128i A_b2 = _mm_unpacklo_epi64(b2, b3);
    __m128i B_b0 = _mm_unpackhi_epi64(b0, b1);
    __m128i B_b2 = _mm_unpackhi_epi64(b2, b3);

    {
      // sign(b) = b >> 15  (0x0000 if positive, 0xffff if negative)
      const __m128i sign_A_b0 = _mm_srai_epi16(A_b0, 15);
      const __m128i sign_A_b2 = _mm_srai_epi16(A_b2, 15);
      const __m128i sign_B_b0 = _mm_srai_epi16(B_b0, 15);
      const __m128i sign_B_b2 = _mm_srai_epi16(B_b2, 15);

      // b = abs(b) = (b ^ sign) - sign
      A_b0 = _mm_xor_si128(A_b0, sign_A_b0);
      A_b2 = _mm_xor_si128(A_b2, sign_A_b2);
      B_b0 = _mm_xor_si128(B_b0, sign_B_b0);
      B_b2 = _mm_xor_si128(B_b2, sign_B_b2);
      A_b0 = _mm_sub_epi16(A_b0, sign_A_b0);
      A_b2 = _mm_sub_epi16(A_b2, sign_A_b2);
      B_b0 = _mm_sub_epi16(B_b0, sign_B_b0);
      B_b2 = _mm_sub_epi16(B_b2, sign_B_b2);
    }

    // b = abs(b) + 3
    A_b0 = _mm_add_epi16(A_b0, three);
    A_b2 = _mm_add_epi16(A_b2, three);
    B_b0 = _mm_add_epi16(B_b0, three);
    B_b2 = _mm_add_epi16(B_b2, three);

    // abs((b + (b<0) + 3) >> 3) = (abs(b) + 3) >> 3
    // b = (abs(b) + 3) >> 3
    A_b0 = _mm_srai_epi16(A_b0, 3);
    A_b2 = _mm_srai_epi16(A_b2, 3);
    B_b0 = _mm_srai_epi16(B_b0, 3);
    B_b2 = _mm_srai_epi16(B_b2, 3);

    // weighted sums
    A_b0 = _mm_madd_epi16(A_b0, w_0);
    A_b2 = _mm_madd_epi16(A_b2, w_8);
    B_b0 = _mm_madd_epi16(B_b0, w_0);
    B_b2 = _mm_madd_epi16(B_b2, w_8);
    A_b0 = _mm_add_epi32(A_b0, A_b2);
    B_b0 = _mm_add_epi32(B_b0, B_b2);

    // difference of weighted sums
    A_b0 = _mm_sub_epi32(A_b0, B_b0);
    _mm_storeu_si128((__m128i*)&sum[0], A_b0);
  }
  return sum[0] + sum[1] + sum[2] + sum[3];
}

static int Disto4x4SSE2(const uint8_t* const a, const uint8_t* const b,
                        const uint16_t* const w) {
  const int diff_sum = TTransformSSE2(a, b, w);
  return (abs(diff_sum) + 8) >> 4;
}

static int Disto16x16SSE2(const uint8_t* const a, const uint8_t* const b,
                          const uint16_t* const w) {
  int D = 0;
  int x, y;
  for (y = 0; y < 16 * BPS; y += 4 * BPS) {
    for (x = 0; x < 16; x += 4) {
      D += Disto4x4SSE2(a + x + y, b + x + y, w);
    }
  }
  return D;
}


//------------------------------------------------------------------------------
// Quantization
//

// Simple quantization
static int QuantizeBlockSSE2(int16_t in[16], int16_t out[16],
                             int n, const VP8Matrix* const mtx) {
  const __m128i max_coeff_2047 = _mm_set1_epi16(2047);
  const __m128i zero = _mm_set1_epi16(0);
  __m128i sign0, sign8;
  __m128i coeff0, coeff8;
  __m128i out0, out8;
  __m128i packed_out;

  // Load all inputs.
  // TODO(cduvivier): Make variable declarations and allocations aligned so that
  //                  we can use _mm_load_si128 instead of _mm_loadu_si128.
  __m128i in0 = _mm_loadu_si128((__m128i*)&in[0]);
  __m128i in8 = _mm_loadu_si128((__m128i*)&in[8]);
  const __m128i sharpen0 = _mm_loadu_si128((__m128i*)&mtx->sharpen_[0]);
  const __m128i sharpen8 = _mm_loadu_si128((__m128i*)&mtx->sharpen_[8]);
  const __m128i iq0 = _mm_loadu_si128((__m128i*)&mtx->iq_[0]);
  const __m128i iq8 = _mm_loadu_si128((__m128i*)&mtx->iq_[8]);
  const __m128i bias0 = _mm_loadu_si128((__m128i*)&mtx->bias_[0]);
  const __m128i bias8 = _mm_loadu_si128((__m128i*)&mtx->bias_[8]);
  const __m128i q0 = _mm_loadu_si128((__m128i*)&mtx->q_[0]);
  const __m128i q8 = _mm_loadu_si128((__m128i*)&mtx->q_[8]);
  const __m128i zthresh0 = _mm_loadu_si128((__m128i*)&mtx->zthresh_[0]);
  const __m128i zthresh8 = _mm_loadu_si128((__m128i*)&mtx->zthresh_[8]);

  // sign(in) = in >> 15  (0x0000 if positive, 0xffff if negative)
  sign0 = _mm_srai_epi16(in0, 15);
  sign8 = _mm_srai_epi16(in8, 15);

  // coeff = abs(in) = (in ^ sign) - sign
  coeff0 = _mm_xor_si128(in0, sign0);
  coeff8 = _mm_xor_si128(in8, sign8);
  coeff0 = _mm_sub_epi16(coeff0, sign0);
  coeff8 = _mm_sub_epi16(coeff8, sign8);

  // coeff = abs(in) + sharpen
  coeff0 = _mm_add_epi16(coeff0, sharpen0);
  coeff8 = _mm_add_epi16(coeff8, sharpen8);

  // if (coeff > 2047) coeff = 2047
  coeff0 = _mm_min_epi16(coeff0, max_coeff_2047);
  coeff8 = _mm_min_epi16(coeff8, max_coeff_2047);

  // out = (coeff * iQ + B) >> QFIX;
  {
    // doing calculations with 32b precision (QFIX=17)
    // out = (coeff * iQ)
    __m128i coeff_iQ0H = _mm_mulhi_epu16(coeff0, iq0);
    __m128i coeff_iQ0L = _mm_mullo_epi16(coeff0, iq0);
    __m128i coeff_iQ8H = _mm_mulhi_epu16(coeff8, iq8);
    __m128i coeff_iQ8L = _mm_mullo_epi16(coeff8, iq8);
    __m128i out_00 = _mm_unpacklo_epi16(coeff_iQ0L, coeff_iQ0H);
    __m128i out_04 = _mm_unpackhi_epi16(coeff_iQ0L, coeff_iQ0H);
    __m128i out_08 = _mm_unpacklo_epi16(coeff_iQ8L, coeff_iQ8H);
    __m128i out_12 = _mm_unpackhi_epi16(coeff_iQ8L, coeff_iQ8H);
    // expand bias from 16b to 32b
    __m128i bias_00 = _mm_unpacklo_epi16(bias0, zero);
    __m128i bias_04 = _mm_unpackhi_epi16(bias0, zero);
    __m128i bias_08 = _mm_unpacklo_epi16(bias8, zero);
    __m128i bias_12 = _mm_unpackhi_epi16(bias8, zero);
    // out = (coeff * iQ + B)
    out_00 = _mm_add_epi32(out_00, bias_00);
    out_04 = _mm_add_epi32(out_04, bias_04);
    out_08 = _mm_add_epi32(out_08, bias_08);
    out_12 = _mm_add_epi32(out_12, bias_12);
    // out = (coeff * iQ + B) >> QFIX;
    out_00 = _mm_srai_epi32(out_00, QFIX);
    out_04 = _mm_srai_epi32(out_04, QFIX);
    out_08 = _mm_srai_epi32(out_08, QFIX);
    out_12 = _mm_srai_epi32(out_12, QFIX);
    // pack result as 16b
    out0 = _mm_packs_epi32(out_00, out_04);
    out8 = _mm_packs_epi32(out_08, out_12);
  }

  // get sign back (if (sign[j]) out_n = -out_n)
  out0 = _mm_xor_si128(out0, sign0);
  out8 = _mm_xor_si128(out8, sign8);
  out0 = _mm_sub_epi16(out0, sign0);
  out8 = _mm_sub_epi16(out8, sign8);

  // in = out * Q
  in0 = _mm_mullo_epi16(out0, q0);
  in8 = _mm_mullo_epi16(out8, q8);

  // if (coeff <= mtx->zthresh_) {in=0; out=0;}
  {
    __m128i cmp0 = _mm_cmpgt_epi16(coeff0, zthresh0);
    __m128i cmp8 = _mm_cmpgt_epi16(coeff8, zthresh8);
    in0 = _mm_and_si128(in0, cmp0);
    in8 = _mm_and_si128(in8, cmp8);
    _mm_storeu_si128((__m128i*)&in[0], in0);
    _mm_storeu_si128((__m128i*)&in[8], in8);
    out0 = _mm_and_si128(out0, cmp0);
    out8 = _mm_and_si128(out8, cmp8);
  }

  // zigzag the output before storing it.
  //
  // The zigzag pattern can almost be reproduced with a small sequence of
  // shuffles. After it, we only need to swap the 7th (ending up in third
  // position instead of twelfth) and 8th values.
  {
    __m128i outZ0, outZ8;
    outZ0 = _mm_shufflehi_epi16(out0,  _MM_SHUFFLE(2, 1, 3, 0));
    outZ0 = _mm_shuffle_epi32  (outZ0, _MM_SHUFFLE(3, 1, 2, 0));
    outZ0 = _mm_shufflehi_epi16(outZ0, _MM_SHUFFLE(3, 1, 0, 2));
    outZ8 = _mm_shufflelo_epi16(out8,  _MM_SHUFFLE(3, 0, 2, 1));
    outZ8 = _mm_shuffle_epi32  (outZ8, _MM_SHUFFLE(3, 1, 2, 0));
    outZ8 = _mm_shufflelo_epi16(outZ8, _MM_SHUFFLE(1, 3, 2, 0));
    _mm_storeu_si128((__m128i*)&out[0], outZ0);
    _mm_storeu_si128((__m128i*)&out[8], outZ8);
    packed_out = _mm_packs_epi16(outZ0, outZ8);
  }
  {
    const int16_t outZ_12 = out[12];
    const int16_t outZ_3 = out[3];
    out[3] = outZ_12;
    out[12] = outZ_3;
  }

  // detect if all 'out' values are zeroes or not
  {
    int32_t tmp[4];
    _mm_storeu_si128((__m128i*)tmp, packed_out);
    if (n) {
      tmp[0] &= ~0xff;
    }
    return (tmp[3] || tmp[2] || tmp[1] || tmp[0]);
  }
}

extern void VP8EncDspInitSSE2(void);
void VP8EncDspInitSSE2(void) {
  VP8CollectHistogram = CollectHistogramSSE2;
  VP8EncQuantizeBlock = QuantizeBlockSSE2;
  VP8ITransform = ITransformSSE2;
  VP8FTransform = FTransformSSE2;
  VP8SSE4x4 = SSE4x4SSE2;
  VP8TDisto4x4 = Disto4x4SSE2;
  VP8TDisto16x16 = Disto16x16SSE2;
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif   // WEBP_USE_SSE2
