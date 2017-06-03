// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// SSE2 version of speed-critical encoding functions.
//
// Author: Christian Duvivier (cduvivier@google.com)

#include "./dsp.h"

#if defined(WEBP_USE_SSE2)
#include <assert.h>
#include <stdlib.h>  // for abs()
#include <emmintrin.h>

#include "./common_sse2.h"
#include "../enc/cost_enc.h"
#include "../enc/vp8i_enc.h"

//------------------------------------------------------------------------------
// Transforms (Paragraph 14.4)

// Does one or two inverse transforms.
static void ITransform(const uint8_t* ref, const int16_t* in, uint8_t* dst,
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
    in0 = _mm_loadl_epi64((const __m128i*)&in[0]);
    in1 = _mm_loadl_epi64((const __m128i*)&in[4]);
    in2 = _mm_loadl_epi64((const __m128i*)&in[8]);
    in3 = _mm_loadl_epi64((const __m128i*)&in[12]);
    // a00 a10 a20 a30   x x x x
    // a01 a11 a21 a31   x x x x
    // a02 a12 a22 a32   x x x x
    // a03 a13 a23 a33   x x x x
    if (do_two) {
      const __m128i inB0 = _mm_loadl_epi64((const __m128i*)&in[16]);
      const __m128i inB1 = _mm_loadl_epi64((const __m128i*)&in[20]);
      const __m128i inB2 = _mm_loadl_epi64((const __m128i*)&in[24]);
      const __m128i inB3 = _mm_loadl_epi64((const __m128i*)&in[28]);
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
    VP8Transpose_2_4x4_16b(&tmp0, &tmp1, &tmp2, &tmp3, &T0, &T1, &T2, &T3);
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
    VP8Transpose_2_4x4_16b(&shifted0, &shifted1, &shifted2, &shifted3, &T0, &T1,
                           &T2, &T3);
  }

  // Add inverse transform to 'ref' and store.
  {
    const __m128i zero = _mm_setzero_si128();
    // Load the reference(s).
    __m128i ref0, ref1, ref2, ref3;
    if (do_two) {
      // Load eight bytes/pixels per line.
      ref0 = _mm_loadl_epi64((const __m128i*)&ref[0 * BPS]);
      ref1 = _mm_loadl_epi64((const __m128i*)&ref[1 * BPS]);
      ref2 = _mm_loadl_epi64((const __m128i*)&ref[2 * BPS]);
      ref3 = _mm_loadl_epi64((const __m128i*)&ref[3 * BPS]);
    } else {
      // Load four bytes/pixels per line.
      ref0 = _mm_cvtsi32_si128(WebPMemToUint32(&ref[0 * BPS]));
      ref1 = _mm_cvtsi32_si128(WebPMemToUint32(&ref[1 * BPS]));
      ref2 = _mm_cvtsi32_si128(WebPMemToUint32(&ref[2 * BPS]));
      ref3 = _mm_cvtsi32_si128(WebPMemToUint32(&ref[3 * BPS]));
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
      WebPUint32ToMem(&dst[0 * BPS], _mm_cvtsi128_si32(ref0));
      WebPUint32ToMem(&dst[1 * BPS], _mm_cvtsi128_si32(ref1));
      WebPUint32ToMem(&dst[2 * BPS], _mm_cvtsi128_si32(ref2));
      WebPUint32ToMem(&dst[3 * BPS], _mm_cvtsi128_si32(ref3));
    }
  }
}

static void FTransformPass1(const __m128i* const in01,
                            const __m128i* const in23,
                            __m128i* const out01,
                            __m128i* const out32) {
  const __m128i k937 = _mm_set1_epi32(937);
  const __m128i k1812 = _mm_set1_epi32(1812);

  const __m128i k88p = _mm_set_epi16(8, 8, 8, 8, 8, 8, 8, 8);
  const __m128i k88m = _mm_set_epi16(-8, 8, -8, 8, -8, 8, -8, 8);
  const __m128i k5352_2217p = _mm_set_epi16(2217, 5352, 2217, 5352,
                                            2217, 5352, 2217, 5352);
  const __m128i k5352_2217m = _mm_set_epi16(-5352, 2217, -5352, 2217,
                                            -5352, 2217, -5352, 2217);

  // *in01 = 00 01 10 11 02 03 12 13
  // *in23 = 20 21 30 31 22 23 32 33
  const __m128i shuf01_p = _mm_shufflehi_epi16(*in01, _MM_SHUFFLE(2, 3, 0, 1));
  const __m128i shuf23_p = _mm_shufflehi_epi16(*in23, _MM_SHUFFLE(2, 3, 0, 1));
  // 00 01 10 11 03 02 13 12
  // 20 21 30 31 23 22 33 32
  const __m128i s01 = _mm_unpacklo_epi64(shuf01_p, shuf23_p);
  const __m128i s32 = _mm_unpackhi_epi64(shuf01_p, shuf23_p);
  // 00 01 10 11 20 21 30 31
  // 03 02 13 12 23 22 33 32
  const __m128i a01 = _mm_add_epi16(s01, s32);
  const __m128i a32 = _mm_sub_epi16(s01, s32);
  // [d0 + d3 | d1 + d2 | ...] = [a0 a1 | a0' a1' | ... ]
  // [d0 - d3 | d1 - d2 | ...] = [a3 a2 | a3' a2' | ... ]

  const __m128i tmp0   = _mm_madd_epi16(a01, k88p);  // [ (a0 + a1) << 3, ... ]
  const __m128i tmp2   = _mm_madd_epi16(a01, k88m);  // [ (a0 - a1) << 3, ... ]
  const __m128i tmp1_1 = _mm_madd_epi16(a32, k5352_2217p);
  const __m128i tmp3_1 = _mm_madd_epi16(a32, k5352_2217m);
  const __m128i tmp1_2 = _mm_add_epi32(tmp1_1, k1812);
  const __m128i tmp3_2 = _mm_add_epi32(tmp3_1, k937);
  const __m128i tmp1   = _mm_srai_epi32(tmp1_2, 9);
  const __m128i tmp3   = _mm_srai_epi32(tmp3_2, 9);
  const __m128i s03    = _mm_packs_epi32(tmp0, tmp2);
  const __m128i s12    = _mm_packs_epi32(tmp1, tmp3);
  const __m128i s_lo   = _mm_unpacklo_epi16(s03, s12);   // 0 1 0 1 0 1...
  const __m128i s_hi   = _mm_unpackhi_epi16(s03, s12);   // 2 3 2 3 2 3
  const __m128i v23    = _mm_unpackhi_epi32(s_lo, s_hi);
  *out01 = _mm_unpacklo_epi32(s_lo, s_hi);
  *out32 = _mm_shuffle_epi32(v23, _MM_SHUFFLE(1, 0, 3, 2));  // 3 2 3 2 3 2..
}

static void FTransformPass2(const __m128i* const v01, const __m128i* const v32,
                            int16_t* out) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i seven = _mm_set1_epi16(7);
  const __m128i k5352_2217 = _mm_set_epi16(5352,  2217, 5352,  2217,
                                           5352,  2217, 5352,  2217);
  const __m128i k2217_5352 = _mm_set_epi16(2217, -5352, 2217, -5352,
                                           2217, -5352, 2217, -5352);
  const __m128i k12000_plus_one = _mm_set1_epi32(12000 + (1 << 16));
  const __m128i k51000 = _mm_set1_epi32(51000);

  // Same operations are done on the (0,3) and (1,2) pairs.
  // a3 = v0 - v3
  // a2 = v1 - v2
  const __m128i a32 = _mm_sub_epi16(*v01, *v32);
  const __m128i a22 = _mm_unpackhi_epi64(a32, a32);

  const __m128i b23 = _mm_unpacklo_epi16(a22, a32);
  const __m128i c1 = _mm_madd_epi16(b23, k5352_2217);
  const __m128i c3 = _mm_madd_epi16(b23, k2217_5352);
  const __m128i d1 = _mm_add_epi32(c1, k12000_plus_one);
  const __m128i d3 = _mm_add_epi32(c3, k51000);
  const __m128i e1 = _mm_srai_epi32(d1, 16);
  const __m128i e3 = _mm_srai_epi32(d3, 16);
  // f1 = ((b3 * 5352 + b2 * 2217 + 12000) >> 16)
  // f3 = ((b3 * 2217 - b2 * 5352 + 51000) >> 16)
  const __m128i f1 = _mm_packs_epi32(e1, e1);
  const __m128i f3 = _mm_packs_epi32(e3, e3);
  // g1 = f1 + (a3 != 0);
  // The compare will return (0xffff, 0) for (==0, !=0). To turn that into the
  // desired (0, 1), we add one earlier through k12000_plus_one.
  // -> g1 = f1 + 1 - (a3 == 0)
  const __m128i g1 = _mm_add_epi16(f1, _mm_cmpeq_epi16(a32, zero));

  // a0 = v0 + v3
  // a1 = v1 + v2
  const __m128i a01 = _mm_add_epi16(*v01, *v32);
  const __m128i a01_plus_7 = _mm_add_epi16(a01, seven);
  const __m128i a11 = _mm_unpackhi_epi64(a01, a01);
  const __m128i c0 = _mm_add_epi16(a01_plus_7, a11);
  const __m128i c2 = _mm_sub_epi16(a01_plus_7, a11);
  // d0 = (a0 + a1 + 7) >> 4;
  // d2 = (a0 - a1 + 7) >> 4;
  const __m128i d0 = _mm_srai_epi16(c0, 4);
  const __m128i d2 = _mm_srai_epi16(c2, 4);

  const __m128i d0_g1 = _mm_unpacklo_epi64(d0, g1);
  const __m128i d2_f3 = _mm_unpacklo_epi64(d2, f3);
  _mm_storeu_si128((__m128i*)&out[0], d0_g1);
  _mm_storeu_si128((__m128i*)&out[8], d2_f3);
}

static void FTransform(const uint8_t* src, const uint8_t* ref, int16_t* out) {
  const __m128i zero = _mm_setzero_si128();
  // Load src.
  const __m128i src0 = _mm_loadl_epi64((const __m128i*)&src[0 * BPS]);
  const __m128i src1 = _mm_loadl_epi64((const __m128i*)&src[1 * BPS]);
  const __m128i src2 = _mm_loadl_epi64((const __m128i*)&src[2 * BPS]);
  const __m128i src3 = _mm_loadl_epi64((const __m128i*)&src[3 * BPS]);
  // 00 01 02 03 *
  // 10 11 12 13 *
  // 20 21 22 23 *
  // 30 31 32 33 *
  // Shuffle.
  const __m128i src_0 = _mm_unpacklo_epi16(src0, src1);
  const __m128i src_1 = _mm_unpacklo_epi16(src2, src3);
  // 00 01 10 11 02 03 12 13 * * ...
  // 20 21 30 31 22 22 32 33 * * ...

  // Load ref.
  const __m128i ref0 = _mm_loadl_epi64((const __m128i*)&ref[0 * BPS]);
  const __m128i ref1 = _mm_loadl_epi64((const __m128i*)&ref[1 * BPS]);
  const __m128i ref2 = _mm_loadl_epi64((const __m128i*)&ref[2 * BPS]);
  const __m128i ref3 = _mm_loadl_epi64((const __m128i*)&ref[3 * BPS]);
  const __m128i ref_0 = _mm_unpacklo_epi16(ref0, ref1);
  const __m128i ref_1 = _mm_unpacklo_epi16(ref2, ref3);

  // Convert both to 16 bit.
  const __m128i src_0_16b = _mm_unpacklo_epi8(src_0, zero);
  const __m128i src_1_16b = _mm_unpacklo_epi8(src_1, zero);
  const __m128i ref_0_16b = _mm_unpacklo_epi8(ref_0, zero);
  const __m128i ref_1_16b = _mm_unpacklo_epi8(ref_1, zero);

  // Compute the difference.
  const __m128i row01 = _mm_sub_epi16(src_0_16b, ref_0_16b);
  const __m128i row23 = _mm_sub_epi16(src_1_16b, ref_1_16b);
  __m128i v01, v32;

  // First pass
  FTransformPass1(&row01, &row23, &v01, &v32);

  // Second pass
  FTransformPass2(&v01, &v32, out);
}

static void FTransform2(const uint8_t* src, const uint8_t* ref, int16_t* out) {
  const __m128i zero = _mm_setzero_si128();

  // Load src and convert to 16b.
  const __m128i src0 = _mm_loadl_epi64((const __m128i*)&src[0 * BPS]);
  const __m128i src1 = _mm_loadl_epi64((const __m128i*)&src[1 * BPS]);
  const __m128i src2 = _mm_loadl_epi64((const __m128i*)&src[2 * BPS]);
  const __m128i src3 = _mm_loadl_epi64((const __m128i*)&src[3 * BPS]);
  const __m128i src_0 = _mm_unpacklo_epi8(src0, zero);
  const __m128i src_1 = _mm_unpacklo_epi8(src1, zero);
  const __m128i src_2 = _mm_unpacklo_epi8(src2, zero);
  const __m128i src_3 = _mm_unpacklo_epi8(src3, zero);
  // Load ref and convert to 16b.
  const __m128i ref0 = _mm_loadl_epi64((const __m128i*)&ref[0 * BPS]);
  const __m128i ref1 = _mm_loadl_epi64((const __m128i*)&ref[1 * BPS]);
  const __m128i ref2 = _mm_loadl_epi64((const __m128i*)&ref[2 * BPS]);
  const __m128i ref3 = _mm_loadl_epi64((const __m128i*)&ref[3 * BPS]);
  const __m128i ref_0 = _mm_unpacklo_epi8(ref0, zero);
  const __m128i ref_1 = _mm_unpacklo_epi8(ref1, zero);
  const __m128i ref_2 = _mm_unpacklo_epi8(ref2, zero);
  const __m128i ref_3 = _mm_unpacklo_epi8(ref3, zero);
  // Compute difference. -> 00 01 02 03  00' 01' 02' 03'
  const __m128i diff0 = _mm_sub_epi16(src_0, ref_0);
  const __m128i diff1 = _mm_sub_epi16(src_1, ref_1);
  const __m128i diff2 = _mm_sub_epi16(src_2, ref_2);
  const __m128i diff3 = _mm_sub_epi16(src_3, ref_3);

  // Unpack and shuffle
  // 00 01 02 03   0 0 0 0
  // 10 11 12 13   0 0 0 0
  // 20 21 22 23   0 0 0 0
  // 30 31 32 33   0 0 0 0
  const __m128i shuf01l = _mm_unpacklo_epi32(diff0, diff1);
  const __m128i shuf23l = _mm_unpacklo_epi32(diff2, diff3);
  const __m128i shuf01h = _mm_unpackhi_epi32(diff0, diff1);
  const __m128i shuf23h = _mm_unpackhi_epi32(diff2, diff3);
  __m128i v01l, v32l;
  __m128i v01h, v32h;

  // First pass
  FTransformPass1(&shuf01l, &shuf23l, &v01l, &v32l);
  FTransformPass1(&shuf01h, &shuf23h, &v01h, &v32h);

  // Second pass
  FTransformPass2(&v01l, &v32l, out + 0);
  FTransformPass2(&v01h, &v32h, out + 16);
}

static void FTransformWHTRow(const int16_t* const in, __m128i* const out) {
  const __m128i kMult = _mm_set_epi16(-1, 1, -1, 1, 1, 1, 1, 1);
  const __m128i src0 = _mm_loadl_epi64((__m128i*)&in[0 * 16]);
  const __m128i src1 = _mm_loadl_epi64((__m128i*)&in[1 * 16]);
  const __m128i src2 = _mm_loadl_epi64((__m128i*)&in[2 * 16]);
  const __m128i src3 = _mm_loadl_epi64((__m128i*)&in[3 * 16]);
  const __m128i A01 = _mm_unpacklo_epi16(src0, src1);  // A0 A1 | ...
  const __m128i A23 = _mm_unpacklo_epi16(src2, src3);  // A2 A3 | ...
  const __m128i B0 = _mm_adds_epi16(A01, A23);    // a0 | a1 | ...
  const __m128i B1 = _mm_subs_epi16(A01, A23);    // a3 | a2 | ...
  const __m128i C0 = _mm_unpacklo_epi32(B0, B1);  // a0 | a1 | a3 | a2 | ...
  const __m128i C1 = _mm_unpacklo_epi32(B1, B0);  // a3 | a2 | a0 | a1 | ...
  const __m128i D = _mm_unpacklo_epi64(C0, C1);   // a0 a1 a3 a2 a3 a2 a0 a1
  *out = _mm_madd_epi16(D, kMult);
}

static void FTransformWHT(const int16_t* in, int16_t* out) {
  // Input is 12b signed.
  __m128i row0, row1, row2, row3;
  // Rows are 14b signed.
  FTransformWHTRow(in + 0 * 64, &row0);
  FTransformWHTRow(in + 1 * 64, &row1);
  FTransformWHTRow(in + 2 * 64, &row2);
  FTransformWHTRow(in + 3 * 64, &row3);

  {
    // The a* are 15b signed.
    const __m128i a0 = _mm_add_epi32(row0, row2);
    const __m128i a1 = _mm_add_epi32(row1, row3);
    const __m128i a2 = _mm_sub_epi32(row1, row3);
    const __m128i a3 = _mm_sub_epi32(row0, row2);
    const __m128i a0a3 = _mm_packs_epi32(a0, a3);
    const __m128i a1a2 = _mm_packs_epi32(a1, a2);

    // The b* are 16b signed.
    const __m128i b0b1 = _mm_add_epi16(a0a3, a1a2);
    const __m128i b3b2 = _mm_sub_epi16(a0a3, a1a2);
    const __m128i tmp_b2b3 = _mm_unpackhi_epi64(b3b2, b3b2);
    const __m128i b2b3 = _mm_unpacklo_epi64(tmp_b2b3, b3b2);

    _mm_storeu_si128((__m128i*)&out[0], _mm_srai_epi16(b0b1, 1));
    _mm_storeu_si128((__m128i*)&out[8], _mm_srai_epi16(b2b3, 1));
  }
}

//------------------------------------------------------------------------------
// Compute susceptibility based on DCT-coeff histograms:
// the higher, the "easier" the macroblock is to compress.

static void CollectHistogram(const uint8_t* ref, const uint8_t* pred,
                             int start_block, int end_block,
                             VP8Histogram* const histo) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i max_coeff_thresh = _mm_set1_epi16(MAX_COEFF_THRESH);
  int j;
  int distribution[MAX_COEFF_THRESH + 1] = { 0 };
  for (j = start_block; j < end_block; ++j) {
    int16_t out[16];
    int k;

    FTransform(ref + VP8DspScan[j], pred + VP8DspScan[j], out);

    // Convert coefficients to bin (within out[]).
    {
      // Load.
      const __m128i out0 = _mm_loadu_si128((__m128i*)&out[0]);
      const __m128i out1 = _mm_loadu_si128((__m128i*)&out[8]);
      const __m128i d0 = _mm_sub_epi16(zero, out0);
      const __m128i d1 = _mm_sub_epi16(zero, out1);
      const __m128i abs0 = _mm_max_epi16(out0, d0);   // abs(v), 16b
      const __m128i abs1 = _mm_max_epi16(out1, d1);
      // v = abs(out) >> 3
      const __m128i v0 = _mm_srai_epi16(abs0, 3);
      const __m128i v1 = _mm_srai_epi16(abs1, 3);
      // bin = min(v, MAX_COEFF_THRESH)
      const __m128i bin0 = _mm_min_epi16(v0, max_coeff_thresh);
      const __m128i bin1 = _mm_min_epi16(v1, max_coeff_thresh);
      // Store.
      _mm_storeu_si128((__m128i*)&out[0], bin0);
      _mm_storeu_si128((__m128i*)&out[8], bin1);
    }

    // Convert coefficients to bin.
    for (k = 0; k < 16; ++k) {
      ++distribution[out[k]];
    }
  }
  VP8SetHistogramData(distribution, histo);
}

//------------------------------------------------------------------------------
// Intra predictions

// helper for chroma-DC predictions
static WEBP_INLINE void Put8x8uv(uint8_t v, uint8_t* dst) {
  int j;
  const __m128i values = _mm_set1_epi8(v);
  for (j = 0; j < 8; ++j) {
    _mm_storel_epi64((__m128i*)(dst + j * BPS), values);
  }
}

static WEBP_INLINE void Put16(uint8_t v, uint8_t* dst) {
  int j;
  const __m128i values = _mm_set1_epi8(v);
  for (j = 0; j < 16; ++j) {
    _mm_store_si128((__m128i*)(dst + j * BPS), values);
  }
}

static WEBP_INLINE void Fill(uint8_t* dst, int value, int size) {
  if (size == 4) {
    int j;
    for (j = 0; j < 4; ++j) {
      memset(dst + j * BPS, value, 4);
    }
  } else if (size == 8) {
    Put8x8uv(value, dst);
  } else {
    Put16(value, dst);
  }
}

static WEBP_INLINE void VE8uv(uint8_t* dst, const uint8_t* top) {
  int j;
  const __m128i top_values = _mm_loadl_epi64((const __m128i*)top);
  for (j = 0; j < 8; ++j) {
    _mm_storel_epi64((__m128i*)(dst + j * BPS), top_values);
  }
}

static WEBP_INLINE void VE16(uint8_t* dst, const uint8_t* top) {
  const __m128i top_values = _mm_load_si128((const __m128i*)top);
  int j;
  for (j = 0; j < 16; ++j) {
    _mm_store_si128((__m128i*)(dst + j * BPS), top_values);
  }
}

static WEBP_INLINE void VerticalPred(uint8_t* dst,
                                     const uint8_t* top, int size) {
  if (top != NULL) {
    if (size == 8) {
      VE8uv(dst, top);
    } else {
      VE16(dst, top);
    }
  } else {
    Fill(dst, 127, size);
  }
}

static WEBP_INLINE void HE8uv(uint8_t* dst, const uint8_t* left) {
  int j;
  for (j = 0; j < 8; ++j) {
    const __m128i values = _mm_set1_epi8(left[j]);
    _mm_storel_epi64((__m128i*)dst, values);
    dst += BPS;
  }
}

static WEBP_INLINE void HE16(uint8_t* dst, const uint8_t* left) {
  int j;
  for (j = 0; j < 16; ++j) {
    const __m128i values = _mm_set1_epi8(left[j]);
    _mm_store_si128((__m128i*)dst, values);
    dst += BPS;
  }
}

static WEBP_INLINE void HorizontalPred(uint8_t* dst,
                                       const uint8_t* left, int size) {
  if (left != NULL) {
    if (size == 8) {
      HE8uv(dst, left);
    } else {
      HE16(dst, left);
    }
  } else {
    Fill(dst, 129, size);
  }
}

static WEBP_INLINE void TM(uint8_t* dst, const uint8_t* left,
                           const uint8_t* top, int size) {
  const __m128i zero = _mm_setzero_si128();
  int y;
  if (size == 8) {
    const __m128i top_values = _mm_loadl_epi64((const __m128i*)top);
    const __m128i top_base = _mm_unpacklo_epi8(top_values, zero);
    for (y = 0; y < 8; ++y, dst += BPS) {
      const int val = left[y] - left[-1];
      const __m128i base = _mm_set1_epi16(val);
      const __m128i out = _mm_packus_epi16(_mm_add_epi16(base, top_base), zero);
      _mm_storel_epi64((__m128i*)dst, out);
    }
  } else {
    const __m128i top_values = _mm_load_si128((const __m128i*)top);
    const __m128i top_base_0 = _mm_unpacklo_epi8(top_values, zero);
    const __m128i top_base_1 = _mm_unpackhi_epi8(top_values, zero);
    for (y = 0; y < 16; ++y, dst += BPS) {
      const int val = left[y] - left[-1];
      const __m128i base = _mm_set1_epi16(val);
      const __m128i out_0 = _mm_add_epi16(base, top_base_0);
      const __m128i out_1 = _mm_add_epi16(base, top_base_1);
      const __m128i out = _mm_packus_epi16(out_0, out_1);
      _mm_store_si128((__m128i*)dst, out);
    }
  }
}

static WEBP_INLINE void TrueMotion(uint8_t* dst, const uint8_t* left,
                                   const uint8_t* top, int size) {
  if (left != NULL) {
    if (top != NULL) {
      TM(dst, left, top, size);
    } else {
      HorizontalPred(dst, left, size);
    }
  } else {
    // true motion without left samples (hence: with default 129 value)
    // is equivalent to VE prediction where you just copy the top samples.
    // Note that if top samples are not available, the default value is
    // then 129, and not 127 as in the VerticalPred case.
    if (top != NULL) {
      VerticalPred(dst, top, size);
    } else {
      Fill(dst, 129, size);
    }
  }
}

static WEBP_INLINE void DC8uv(uint8_t* dst, const uint8_t* left,
                              const uint8_t* top) {
  const __m128i top_values = _mm_loadl_epi64((const __m128i*)top);
  const __m128i left_values = _mm_loadl_epi64((const __m128i*)left);
  const __m128i combined = _mm_unpacklo_epi64(top_values, left_values);
  const int DC = VP8HorizontalAdd8b(&combined) + 8;
  Put8x8uv(DC >> 4, dst);
}

static WEBP_INLINE void DC8uvNoLeft(uint8_t* dst, const uint8_t* top) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i top_values = _mm_loadl_epi64((const __m128i*)top);
  const __m128i sum = _mm_sad_epu8(top_values, zero);
  const int DC = _mm_cvtsi128_si32(sum) + 4;
  Put8x8uv(DC >> 3, dst);
}

static WEBP_INLINE void DC8uvNoTop(uint8_t* dst, const uint8_t* left) {
  // 'left' is contiguous so we can reuse the top summation.
  DC8uvNoLeft(dst, left);
}

static WEBP_INLINE void DC8uvNoTopLeft(uint8_t* dst) {
  Put8x8uv(0x80, dst);
}

static WEBP_INLINE void DC8uvMode(uint8_t* dst, const uint8_t* left,
                                  const uint8_t* top) {
  if (top != NULL) {
    if (left != NULL) {  // top and left present
      DC8uv(dst, left, top);
    } else {  // top, but no left
      DC8uvNoLeft(dst, top);
    }
  } else if (left != NULL) {  // left but no top
    DC8uvNoTop(dst, left);
  } else {  // no top, no left, nothing.
    DC8uvNoTopLeft(dst);
  }
}

static WEBP_INLINE void DC16(uint8_t* dst, const uint8_t* left,
                             const uint8_t* top) {
  const __m128i top_row = _mm_load_si128((const __m128i*)top);
  const __m128i left_row = _mm_load_si128((const __m128i*)left);
  const int DC =
      VP8HorizontalAdd8b(&top_row) + VP8HorizontalAdd8b(&left_row) + 16;
  Put16(DC >> 5, dst);
}

static WEBP_INLINE void DC16NoLeft(uint8_t* dst, const uint8_t* top) {
  const __m128i top_row = _mm_load_si128((const __m128i*)top);
  const int DC = VP8HorizontalAdd8b(&top_row) + 8;
  Put16(DC >> 4, dst);
}

static WEBP_INLINE void DC16NoTop(uint8_t* dst, const uint8_t* left) {
  // 'left' is contiguous so we can reuse the top summation.
  DC16NoLeft(dst, left);
}

static WEBP_INLINE void DC16NoTopLeft(uint8_t* dst) {
  Put16(0x80, dst);
}

static WEBP_INLINE void DC16Mode(uint8_t* dst, const uint8_t* left,
                                 const uint8_t* top) {
  if (top != NULL) {
    if (left != NULL) {  // top and left present
      DC16(dst, left, top);
    } else {  // top, but no left
      DC16NoLeft(dst, top);
    }
  } else if (left != NULL) {  // left but no top
    DC16NoTop(dst, left);
  } else {  // no top, no left, nothing.
    DC16NoTopLeft(dst);
  }
}

//------------------------------------------------------------------------------
// 4x4 predictions

#define DST(x, y) dst[(x) + (y) * BPS]
#define AVG3(a, b, c) (((a) + 2 * (b) + (c) + 2) >> 2)
#define AVG2(a, b) (((a) + (b) + 1) >> 1)

// We use the following 8b-arithmetic tricks:
//     (a + 2 * b + c + 2) >> 2 = (AC + b + 1) >> 1
//   where: AC = (a + c) >> 1 = [(a + c + 1) >> 1] - [(a^c) & 1]
// and:
//     (a + 2 * b + c + 2) >> 2 = (AB + BC + 1) >> 1 - (ab|bc)&lsb
//   where: AC = (a + b + 1) >> 1,   BC = (b + c + 1) >> 1
//   and ab = a ^ b, bc = b ^ c, lsb = (AC^BC)&1

static WEBP_INLINE void VE4(uint8_t* dst, const uint8_t* top) {  // vertical
  const __m128i one = _mm_set1_epi8(1);
  const __m128i ABCDEFGH = _mm_loadl_epi64((__m128i*)(top - 1));
  const __m128i BCDEFGH0 = _mm_srli_si128(ABCDEFGH, 1);
  const __m128i CDEFGH00 = _mm_srli_si128(ABCDEFGH, 2);
  const __m128i a = _mm_avg_epu8(ABCDEFGH, CDEFGH00);
  const __m128i lsb = _mm_and_si128(_mm_xor_si128(ABCDEFGH, CDEFGH00), one);
  const __m128i b = _mm_subs_epu8(a, lsb);
  const __m128i avg = _mm_avg_epu8(b, BCDEFGH0);
  const uint32_t vals = _mm_cvtsi128_si32(avg);
  int i;
  for (i = 0; i < 4; ++i) {
    WebPUint32ToMem(dst + i * BPS, vals);
  }
}

static WEBP_INLINE void HE4(uint8_t* dst, const uint8_t* top) {  // horizontal
  const int X = top[-1];
  const int I = top[-2];
  const int J = top[-3];
  const int K = top[-4];
  const int L = top[-5];
  WebPUint32ToMem(dst + 0 * BPS, 0x01010101U * AVG3(X, I, J));
  WebPUint32ToMem(dst + 1 * BPS, 0x01010101U * AVG3(I, J, K));
  WebPUint32ToMem(dst + 2 * BPS, 0x01010101U * AVG3(J, K, L));
  WebPUint32ToMem(dst + 3 * BPS, 0x01010101U * AVG3(K, L, L));
}

static WEBP_INLINE void DC4(uint8_t* dst, const uint8_t* top) {
  uint32_t dc = 4;
  int i;
  for (i = 0; i < 4; ++i) dc += top[i] + top[-5 + i];
  Fill(dst, dc >> 3, 4);
}

static WEBP_INLINE void LD4(uint8_t* dst, const uint8_t* top) {  // Down-Left
  const __m128i one = _mm_set1_epi8(1);
  const __m128i ABCDEFGH = _mm_loadl_epi64((const __m128i*)top);
  const __m128i BCDEFGH0 = _mm_srli_si128(ABCDEFGH, 1);
  const __m128i CDEFGH00 = _mm_srli_si128(ABCDEFGH, 2);
  const __m128i CDEFGHH0 = _mm_insert_epi16(CDEFGH00, top[7], 3);
  const __m128i avg1 = _mm_avg_epu8(ABCDEFGH, CDEFGHH0);
  const __m128i lsb = _mm_and_si128(_mm_xor_si128(ABCDEFGH, CDEFGHH0), one);
  const __m128i avg2 = _mm_subs_epu8(avg1, lsb);
  const __m128i abcdefg = _mm_avg_epu8(avg2, BCDEFGH0);
  WebPUint32ToMem(dst + 0 * BPS, _mm_cvtsi128_si32(               abcdefg    ));
  WebPUint32ToMem(dst + 1 * BPS, _mm_cvtsi128_si32(_mm_srli_si128(abcdefg, 1)));
  WebPUint32ToMem(dst + 2 * BPS, _mm_cvtsi128_si32(_mm_srli_si128(abcdefg, 2)));
  WebPUint32ToMem(dst + 3 * BPS, _mm_cvtsi128_si32(_mm_srli_si128(abcdefg, 3)));
}

static WEBP_INLINE void VR4(uint8_t* dst,
                            const uint8_t* top) {  // Vertical-Right
  const __m128i one = _mm_set1_epi8(1);
  const int I = top[-2];
  const int J = top[-3];
  const int K = top[-4];
  const int X = top[-1];
  const __m128i XABCD = _mm_loadl_epi64((const __m128i*)(top - 1));
  const __m128i ABCD0 = _mm_srli_si128(XABCD, 1);
  const __m128i abcd = _mm_avg_epu8(XABCD, ABCD0);
  const __m128i _XABCD = _mm_slli_si128(XABCD, 1);
  const __m128i IXABCD = _mm_insert_epi16(_XABCD, I | (X << 8), 0);
  const __m128i avg1 = _mm_avg_epu8(IXABCD, ABCD0);
  const __m128i lsb = _mm_and_si128(_mm_xor_si128(IXABCD, ABCD0), one);
  const __m128i avg2 = _mm_subs_epu8(avg1, lsb);
  const __m128i efgh = _mm_avg_epu8(avg2, XABCD);
  WebPUint32ToMem(dst + 0 * BPS, _mm_cvtsi128_si32(               abcd    ));
  WebPUint32ToMem(dst + 1 * BPS, _mm_cvtsi128_si32(               efgh    ));
  WebPUint32ToMem(dst + 2 * BPS, _mm_cvtsi128_si32(_mm_slli_si128(abcd, 1)));
  WebPUint32ToMem(dst + 3 * BPS, _mm_cvtsi128_si32(_mm_slli_si128(efgh, 1)));

  // these two are hard to implement in SSE2, so we keep the C-version:
  DST(0, 2) = AVG3(J, I, X);
  DST(0, 3) = AVG3(K, J, I);
}

static WEBP_INLINE void VL4(uint8_t* dst,
                            const uint8_t* top) {  // Vertical-Left
  const __m128i one = _mm_set1_epi8(1);
  const __m128i ABCDEFGH = _mm_loadl_epi64((const __m128i*)top);
  const __m128i BCDEFGH_ = _mm_srli_si128(ABCDEFGH, 1);
  const __m128i CDEFGH__ = _mm_srli_si128(ABCDEFGH, 2);
  const __m128i avg1 = _mm_avg_epu8(ABCDEFGH, BCDEFGH_);
  const __m128i avg2 = _mm_avg_epu8(CDEFGH__, BCDEFGH_);
  const __m128i avg3 = _mm_avg_epu8(avg1, avg2);
  const __m128i lsb1 = _mm_and_si128(_mm_xor_si128(avg1, avg2), one);
  const __m128i ab = _mm_xor_si128(ABCDEFGH, BCDEFGH_);
  const __m128i bc = _mm_xor_si128(CDEFGH__, BCDEFGH_);
  const __m128i abbc = _mm_or_si128(ab, bc);
  const __m128i lsb2 = _mm_and_si128(abbc, lsb1);
  const __m128i avg4 = _mm_subs_epu8(avg3, lsb2);
  const uint32_t extra_out = _mm_cvtsi128_si32(_mm_srli_si128(avg4, 4));
  WebPUint32ToMem(dst + 0 * BPS, _mm_cvtsi128_si32(               avg1    ));
  WebPUint32ToMem(dst + 1 * BPS, _mm_cvtsi128_si32(               avg4    ));
  WebPUint32ToMem(dst + 2 * BPS, _mm_cvtsi128_si32(_mm_srli_si128(avg1, 1)));
  WebPUint32ToMem(dst + 3 * BPS, _mm_cvtsi128_si32(_mm_srli_si128(avg4, 1)));

  // these two are hard to get and irregular
  DST(3, 2) = (extra_out >> 0) & 0xff;
  DST(3, 3) = (extra_out >> 8) & 0xff;
}

static WEBP_INLINE void RD4(uint8_t* dst, const uint8_t* top) {  // Down-right
  const __m128i one = _mm_set1_epi8(1);
  const __m128i LKJIXABC = _mm_loadl_epi64((const __m128i*)(top - 5));
  const __m128i LKJIXABCD = _mm_insert_epi16(LKJIXABC, top[3], 4);
  const __m128i KJIXABCD_ = _mm_srli_si128(LKJIXABCD, 1);
  const __m128i JIXABCD__ = _mm_srli_si128(LKJIXABCD, 2);
  const __m128i avg1 = _mm_avg_epu8(JIXABCD__, LKJIXABCD);
  const __m128i lsb = _mm_and_si128(_mm_xor_si128(JIXABCD__, LKJIXABCD), one);
  const __m128i avg2 = _mm_subs_epu8(avg1, lsb);
  const __m128i abcdefg = _mm_avg_epu8(avg2, KJIXABCD_);
  WebPUint32ToMem(dst + 3 * BPS, _mm_cvtsi128_si32(               abcdefg    ));
  WebPUint32ToMem(dst + 2 * BPS, _mm_cvtsi128_si32(_mm_srli_si128(abcdefg, 1)));
  WebPUint32ToMem(dst + 1 * BPS, _mm_cvtsi128_si32(_mm_srli_si128(abcdefg, 2)));
  WebPUint32ToMem(dst + 0 * BPS, _mm_cvtsi128_si32(_mm_srli_si128(abcdefg, 3)));
}

static WEBP_INLINE void HU4(uint8_t* dst, const uint8_t* top) {
  const int I = top[-2];
  const int J = top[-3];
  const int K = top[-4];
  const int L = top[-5];
  DST(0, 0) =             AVG2(I, J);
  DST(2, 0) = DST(0, 1) = AVG2(J, K);
  DST(2, 1) = DST(0, 2) = AVG2(K, L);
  DST(1, 0) =             AVG3(I, J, K);
  DST(3, 0) = DST(1, 1) = AVG3(J, K, L);
  DST(3, 1) = DST(1, 2) = AVG3(K, L, L);
  DST(3, 2) = DST(2, 2) =
  DST(0, 3) = DST(1, 3) = DST(2, 3) = DST(3, 3) = L;
}

static WEBP_INLINE void HD4(uint8_t* dst, const uint8_t* top) {
  const int X = top[-1];
  const int I = top[-2];
  const int J = top[-3];
  const int K = top[-4];
  const int L = top[-5];
  const int A = top[0];
  const int B = top[1];
  const int C = top[2];

  DST(0, 0) = DST(2, 1) = AVG2(I, X);
  DST(0, 1) = DST(2, 2) = AVG2(J, I);
  DST(0, 2) = DST(2, 3) = AVG2(K, J);
  DST(0, 3)             = AVG2(L, K);

  DST(3, 0)             = AVG3(A, B, C);
  DST(2, 0)             = AVG3(X, A, B);
  DST(1, 0) = DST(3, 1) = AVG3(I, X, A);
  DST(1, 1) = DST(3, 2) = AVG3(J, I, X);
  DST(1, 2) = DST(3, 3) = AVG3(K, J, I);
  DST(1, 3)             = AVG3(L, K, J);
}

static WEBP_INLINE void TM4(uint8_t* dst, const uint8_t* top) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i top_values = _mm_cvtsi32_si128(WebPMemToUint32(top));
  const __m128i top_base = _mm_unpacklo_epi8(top_values, zero);
  int y;
  for (y = 0; y < 4; ++y, dst += BPS) {
    const int val = top[-2 - y] - top[-1];
    const __m128i base = _mm_set1_epi16(val);
    const __m128i out = _mm_packus_epi16(_mm_add_epi16(base, top_base), zero);
    WebPUint32ToMem(dst, _mm_cvtsi128_si32(out));
  }
}

#undef DST
#undef AVG3
#undef AVG2

//------------------------------------------------------------------------------
// luma 4x4 prediction

// Left samples are top[-5 .. -2], top_left is top[-1], top are
// located at top[0..3], and top right is top[4..7]
static void Intra4Preds(uint8_t* dst, const uint8_t* top) {
  DC4(I4DC4 + dst, top);
  TM4(I4TM4 + dst, top);
  VE4(I4VE4 + dst, top);
  HE4(I4HE4 + dst, top);
  RD4(I4RD4 + dst, top);
  VR4(I4VR4 + dst, top);
  LD4(I4LD4 + dst, top);
  VL4(I4VL4 + dst, top);
  HD4(I4HD4 + dst, top);
  HU4(I4HU4 + dst, top);
}

//------------------------------------------------------------------------------
// Chroma 8x8 prediction (paragraph 12.2)

static void IntraChromaPreds(uint8_t* dst, const uint8_t* left,
                             const uint8_t* top) {
  // U block
  DC8uvMode(C8DC8 + dst, left, top);
  VerticalPred(C8VE8 + dst, top, 8);
  HorizontalPred(C8HE8 + dst, left, 8);
  TrueMotion(C8TM8 + dst, left, top, 8);
  // V block
  dst += 8;
  if (top != NULL) top += 8;
  if (left != NULL) left += 16;
  DC8uvMode(C8DC8 + dst, left, top);
  VerticalPred(C8VE8 + dst, top, 8);
  HorizontalPred(C8HE8 + dst, left, 8);
  TrueMotion(C8TM8 + dst, left, top, 8);
}

//------------------------------------------------------------------------------
// luma 16x16 prediction (paragraph 12.3)

static void Intra16Preds(uint8_t* dst,
                         const uint8_t* left, const uint8_t* top) {
  DC16Mode(I16DC16 + dst, left, top);
  VerticalPred(I16VE16 + dst, top, 16);
  HorizontalPred(I16HE16 + dst, left, 16);
  TrueMotion(I16TM16 + dst, left, top, 16);
}

//------------------------------------------------------------------------------
// Metric

static WEBP_INLINE void SubtractAndAccumulate(const __m128i a, const __m128i b,
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

static WEBP_INLINE int SSE_16xN(const uint8_t* a, const uint8_t* b,
                                int num_pairs) {
  __m128i sum = _mm_setzero_si128();
  int32_t tmp[4];
  int i;

  for (i = 0; i < num_pairs; ++i) {
    const __m128i a0 = _mm_loadu_si128((const __m128i*)&a[BPS * 0]);
    const __m128i b0 = _mm_loadu_si128((const __m128i*)&b[BPS * 0]);
    const __m128i a1 = _mm_loadu_si128((const __m128i*)&a[BPS * 1]);
    const __m128i b1 = _mm_loadu_si128((const __m128i*)&b[BPS * 1]);
    __m128i sum1, sum2;
    SubtractAndAccumulate(a0, b0, &sum1);
    SubtractAndAccumulate(a1, b1, &sum2);
    sum = _mm_add_epi32(sum, _mm_add_epi32(sum1, sum2));
    a += 2 * BPS;
    b += 2 * BPS;
  }
  _mm_storeu_si128((__m128i*)tmp, sum);
  return (tmp[3] + tmp[2] + tmp[1] + tmp[0]);
}

static int SSE16x16(const uint8_t* a, const uint8_t* b) {
  return SSE_16xN(a, b, 8);
}

static int SSE16x8(const uint8_t* a, const uint8_t* b) {
  return SSE_16xN(a, b, 4);
}

#define LOAD_8x16b(ptr) \
  _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(ptr)), zero)

static int SSE8x8(const uint8_t* a, const uint8_t* b) {
  const __m128i zero = _mm_setzero_si128();
  int num_pairs = 4;
  __m128i sum = zero;
  int32_t tmp[4];
  while (num_pairs-- > 0) {
    const __m128i a0 = LOAD_8x16b(&a[BPS * 0]);
    const __m128i a1 = LOAD_8x16b(&a[BPS * 1]);
    const __m128i b0 = LOAD_8x16b(&b[BPS * 0]);
    const __m128i b1 = LOAD_8x16b(&b[BPS * 1]);
    // subtract
    const __m128i c0 = _mm_subs_epi16(a0, b0);
    const __m128i c1 = _mm_subs_epi16(a1, b1);
    // multiply/accumulate with self
    const __m128i d0 = _mm_madd_epi16(c0, c0);
    const __m128i d1 = _mm_madd_epi16(c1, c1);
    // collect
    const __m128i sum01 = _mm_add_epi32(d0, d1);
    sum = _mm_add_epi32(sum, sum01);
    a += 2 * BPS;
    b += 2 * BPS;
  }
  _mm_storeu_si128((__m128i*)tmp, sum);
  return (tmp[3] + tmp[2] + tmp[1] + tmp[0]);
}
#undef LOAD_8x16b

static int SSE4x4(const uint8_t* a, const uint8_t* b) {
  const __m128i zero = _mm_setzero_si128();

  // Load values. Note that we read 8 pixels instead of 4,
  // but the a/b buffers are over-allocated to that effect.
  const __m128i a0 = _mm_loadl_epi64((const __m128i*)&a[BPS * 0]);
  const __m128i a1 = _mm_loadl_epi64((const __m128i*)&a[BPS * 1]);
  const __m128i a2 = _mm_loadl_epi64((const __m128i*)&a[BPS * 2]);
  const __m128i a3 = _mm_loadl_epi64((const __m128i*)&a[BPS * 3]);
  const __m128i b0 = _mm_loadl_epi64((const __m128i*)&b[BPS * 0]);
  const __m128i b1 = _mm_loadl_epi64((const __m128i*)&b[BPS * 1]);
  const __m128i b2 = _mm_loadl_epi64((const __m128i*)&b[BPS * 2]);
  const __m128i b3 = _mm_loadl_epi64((const __m128i*)&b[BPS * 3]);
  // Combine pair of lines.
  const __m128i a01 = _mm_unpacklo_epi32(a0, a1);
  const __m128i a23 = _mm_unpacklo_epi32(a2, a3);
  const __m128i b01 = _mm_unpacklo_epi32(b0, b1);
  const __m128i b23 = _mm_unpacklo_epi32(b2, b3);
  // Convert to 16b.
  const __m128i a01s = _mm_unpacklo_epi8(a01, zero);
  const __m128i a23s = _mm_unpacklo_epi8(a23, zero);
  const __m128i b01s = _mm_unpacklo_epi8(b01, zero);
  const __m128i b23s = _mm_unpacklo_epi8(b23, zero);
  // subtract, square and accumulate
  const __m128i d0 = _mm_subs_epi16(a01s, b01s);
  const __m128i d1 = _mm_subs_epi16(a23s, b23s);
  const __m128i e0 = _mm_madd_epi16(d0, d0);
  const __m128i e1 = _mm_madd_epi16(d1, d1);
  const __m128i sum = _mm_add_epi32(e0, e1);

  int32_t tmp[4];
  _mm_storeu_si128((__m128i*)tmp, sum);
  return (tmp[3] + tmp[2] + tmp[1] + tmp[0]);
}

//------------------------------------------------------------------------------

static void Mean16x4(const uint8_t* ref, uint32_t dc[4]) {
  const __m128i mask = _mm_set1_epi16(0x00ff);
  const __m128i a0 = _mm_loadu_si128((const __m128i*)&ref[BPS * 0]);
  const __m128i a1 = _mm_loadu_si128((const __m128i*)&ref[BPS * 1]);
  const __m128i a2 = _mm_loadu_si128((const __m128i*)&ref[BPS * 2]);
  const __m128i a3 = _mm_loadu_si128((const __m128i*)&ref[BPS * 3]);
  const __m128i b0 = _mm_srli_epi16(a0, 8);     // hi byte
  const __m128i b1 = _mm_srli_epi16(a1, 8);
  const __m128i b2 = _mm_srli_epi16(a2, 8);
  const __m128i b3 = _mm_srli_epi16(a3, 8);
  const __m128i c0 = _mm_and_si128(a0, mask);   // lo byte
  const __m128i c1 = _mm_and_si128(a1, mask);
  const __m128i c2 = _mm_and_si128(a2, mask);
  const __m128i c3 = _mm_and_si128(a3, mask);
  const __m128i d0 = _mm_add_epi32(b0, c0);
  const __m128i d1 = _mm_add_epi32(b1, c1);
  const __m128i d2 = _mm_add_epi32(b2, c2);
  const __m128i d3 = _mm_add_epi32(b3, c3);
  const __m128i e0 = _mm_add_epi32(d0, d1);
  const __m128i e1 = _mm_add_epi32(d2, d3);
  const __m128i f0 = _mm_add_epi32(e0, e1);
  uint16_t tmp[8];
  _mm_storeu_si128((__m128i*)tmp, f0);
  dc[0] = tmp[0] + tmp[1];
  dc[1] = tmp[2] + tmp[3];
  dc[2] = tmp[4] + tmp[5];
  dc[3] = tmp[6] + tmp[7];
}

//------------------------------------------------------------------------------
// Texture distortion
//
// We try to match the spectral content (weighted) between source and
// reconstructed samples.

// Hadamard transform
// Returns the weighted sum of the absolute value of transformed coefficients.
// w[] contains a row-major 4 by 4 symmetric matrix.
static int TTransform(const uint8_t* inA, const uint8_t* inB,
                      const uint16_t* const w) {
  int32_t sum[4];
  __m128i tmp_0, tmp_1, tmp_2, tmp_3;
  const __m128i zero = _mm_setzero_si128();

  // Load and combine inputs.
  {
    const __m128i inA_0 = _mm_loadl_epi64((const __m128i*)&inA[BPS * 0]);
    const __m128i inA_1 = _mm_loadl_epi64((const __m128i*)&inA[BPS * 1]);
    const __m128i inA_2 = _mm_loadl_epi64((const __m128i*)&inA[BPS * 2]);
    const __m128i inA_3 = _mm_loadl_epi64((const __m128i*)&inA[BPS * 3]);
    const __m128i inB_0 = _mm_loadl_epi64((const __m128i*)&inB[BPS * 0]);
    const __m128i inB_1 = _mm_loadl_epi64((const __m128i*)&inB[BPS * 1]);
    const __m128i inB_2 = _mm_loadl_epi64((const __m128i*)&inB[BPS * 2]);
    const __m128i inB_3 = _mm_loadl_epi64((const __m128i*)&inB[BPS * 3]);

    // Combine inA and inB (we'll do two transforms in parallel).
    const __m128i inAB_0 = _mm_unpacklo_epi32(inA_0, inB_0);
    const __m128i inAB_1 = _mm_unpacklo_epi32(inA_1, inB_1);
    const __m128i inAB_2 = _mm_unpacklo_epi32(inA_2, inB_2);
    const __m128i inAB_3 = _mm_unpacklo_epi32(inA_3, inB_3);
    tmp_0 = _mm_unpacklo_epi8(inAB_0, zero);
    tmp_1 = _mm_unpacklo_epi8(inAB_1, zero);
    tmp_2 = _mm_unpacklo_epi8(inAB_2, zero);
    tmp_3 = _mm_unpacklo_epi8(inAB_3, zero);
    // a00 a01 a02 a03   b00 b01 b02 b03
    // a10 a11 a12 a13   b10 b11 b12 b13
    // a20 a21 a22 a23   b20 b21 b22 b23
    // a30 a31 a32 a33   b30 b31 b32 b33
  }

  // Vertical pass first to avoid a transpose (vertical and horizontal passes
  // are commutative because w/kWeightY is symmetric) and subsequent transpose.
  {
    // Calculate a and b (two 4x4 at once).
    const __m128i a0 = _mm_add_epi16(tmp_0, tmp_2);
    const __m128i a1 = _mm_add_epi16(tmp_1, tmp_3);
    const __m128i a2 = _mm_sub_epi16(tmp_1, tmp_3);
    const __m128i a3 = _mm_sub_epi16(tmp_0, tmp_2);
    const __m128i b0 = _mm_add_epi16(a0, a1);
    const __m128i b1 = _mm_add_epi16(a3, a2);
    const __m128i b2 = _mm_sub_epi16(a3, a2);
    const __m128i b3 = _mm_sub_epi16(a0, a1);
    // a00 a01 a02 a03   b00 b01 b02 b03
    // a10 a11 a12 a13   b10 b11 b12 b13
    // a20 a21 a22 a23   b20 b21 b22 b23
    // a30 a31 a32 a33   b30 b31 b32 b33

    // Transpose the two 4x4.
    VP8Transpose_2_4x4_16b(&b0, &b1, &b2, &b3, &tmp_0, &tmp_1, &tmp_2, &tmp_3);
  }

  // Horizontal pass and difference of weighted sums.
  {
    // Load all inputs.
    const __m128i w_0 = _mm_loadu_si128((const __m128i*)&w[0]);
    const __m128i w_8 = _mm_loadu_si128((const __m128i*)&w[8]);

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
      const __m128i d0 = _mm_sub_epi16(zero, A_b0);
      const __m128i d1 = _mm_sub_epi16(zero, A_b2);
      const __m128i d2 = _mm_sub_epi16(zero, B_b0);
      const __m128i d3 = _mm_sub_epi16(zero, B_b2);
      A_b0 = _mm_max_epi16(A_b0, d0);   // abs(v), 16b
      A_b2 = _mm_max_epi16(A_b2, d1);
      B_b0 = _mm_max_epi16(B_b0, d2);
      B_b2 = _mm_max_epi16(B_b2, d3);
    }

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

static int Disto4x4(const uint8_t* const a, const uint8_t* const b,
                    const uint16_t* const w) {
  const int diff_sum = TTransform(a, b, w);
  return abs(diff_sum) >> 5;
}

static int Disto16x16(const uint8_t* const a, const uint8_t* const b,
                      const uint16_t* const w) {
  int D = 0;
  int x, y;
  for (y = 0; y < 16 * BPS; y += 4 * BPS) {
    for (x = 0; x < 16; x += 4) {
      D += Disto4x4(a + x + y, b + x + y, w);
    }
  }
  return D;
}

//------------------------------------------------------------------------------
// Quantization
//

static WEBP_INLINE int DoQuantizeBlock(int16_t in[16], int16_t out[16],
                                       const uint16_t* const sharpen,
                                       const VP8Matrix* const mtx) {
  const __m128i max_coeff_2047 = _mm_set1_epi16(MAX_LEVEL);
  const __m128i zero = _mm_setzero_si128();
  __m128i coeff0, coeff8;
  __m128i out0, out8;
  __m128i packed_out;

  // Load all inputs.
  __m128i in0 = _mm_loadu_si128((__m128i*)&in[0]);
  __m128i in8 = _mm_loadu_si128((__m128i*)&in[8]);
  const __m128i iq0 = _mm_loadu_si128((const __m128i*)&mtx->iq_[0]);
  const __m128i iq8 = _mm_loadu_si128((const __m128i*)&mtx->iq_[8]);
  const __m128i q0 = _mm_loadu_si128((const __m128i*)&mtx->q_[0]);
  const __m128i q8 = _mm_loadu_si128((const __m128i*)&mtx->q_[8]);

  // extract sign(in)  (0x0000 if positive, 0xffff if negative)
  const __m128i sign0 = _mm_cmpgt_epi16(zero, in0);
  const __m128i sign8 = _mm_cmpgt_epi16(zero, in8);

  // coeff = abs(in) = (in ^ sign) - sign
  coeff0 = _mm_xor_si128(in0, sign0);
  coeff8 = _mm_xor_si128(in8, sign8);
  coeff0 = _mm_sub_epi16(coeff0, sign0);
  coeff8 = _mm_sub_epi16(coeff8, sign8);

  // coeff = abs(in) + sharpen
  if (sharpen != NULL) {
    const __m128i sharpen0 = _mm_loadu_si128((const __m128i*)&sharpen[0]);
    const __m128i sharpen8 = _mm_loadu_si128((const __m128i*)&sharpen[8]);
    coeff0 = _mm_add_epi16(coeff0, sharpen0);
    coeff8 = _mm_add_epi16(coeff8, sharpen8);
  }

  // out = (coeff * iQ + B) >> QFIX
  {
    // doing calculations with 32b precision (QFIX=17)
    // out = (coeff * iQ)
    const __m128i coeff_iQ0H = _mm_mulhi_epu16(coeff0, iq0);
    const __m128i coeff_iQ0L = _mm_mullo_epi16(coeff0, iq0);
    const __m128i coeff_iQ8H = _mm_mulhi_epu16(coeff8, iq8);
    const __m128i coeff_iQ8L = _mm_mullo_epi16(coeff8, iq8);
    __m128i out_00 = _mm_unpacklo_epi16(coeff_iQ0L, coeff_iQ0H);
    __m128i out_04 = _mm_unpackhi_epi16(coeff_iQ0L, coeff_iQ0H);
    __m128i out_08 = _mm_unpacklo_epi16(coeff_iQ8L, coeff_iQ8H);
    __m128i out_12 = _mm_unpackhi_epi16(coeff_iQ8L, coeff_iQ8H);
    // out = (coeff * iQ + B)
    const __m128i bias_00 = _mm_loadu_si128((const __m128i*)&mtx->bias_[0]);
    const __m128i bias_04 = _mm_loadu_si128((const __m128i*)&mtx->bias_[4]);
    const __m128i bias_08 = _mm_loadu_si128((const __m128i*)&mtx->bias_[8]);
    const __m128i bias_12 = _mm_loadu_si128((const __m128i*)&mtx->bias_[12]);
    out_00 = _mm_add_epi32(out_00, bias_00);
    out_04 = _mm_add_epi32(out_04, bias_04);
    out_08 = _mm_add_epi32(out_08, bias_08);
    out_12 = _mm_add_epi32(out_12, bias_12);
    // out = QUANTDIV(coeff, iQ, B, QFIX)
    out_00 = _mm_srai_epi32(out_00, QFIX);
    out_04 = _mm_srai_epi32(out_04, QFIX);
    out_08 = _mm_srai_epi32(out_08, QFIX);
    out_12 = _mm_srai_epi32(out_12, QFIX);

    // pack result as 16b
    out0 = _mm_packs_epi32(out_00, out_04);
    out8 = _mm_packs_epi32(out_08, out_12);

    // if (coeff > 2047) coeff = 2047
    out0 = _mm_min_epi16(out0, max_coeff_2047);
    out8 = _mm_min_epi16(out8, max_coeff_2047);
  }

  // get sign back (if (sign[j]) out_n = -out_n)
  out0 = _mm_xor_si128(out0, sign0);
  out8 = _mm_xor_si128(out8, sign8);
  out0 = _mm_sub_epi16(out0, sign0);
  out8 = _mm_sub_epi16(out8, sign8);

  // in = out * Q
  in0 = _mm_mullo_epi16(out0, q0);
  in8 = _mm_mullo_epi16(out8, q8);

  _mm_storeu_si128((__m128i*)&in[0], in0);
  _mm_storeu_si128((__m128i*)&in[8], in8);

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
  return (_mm_movemask_epi8(_mm_cmpeq_epi8(packed_out, zero)) != 0xffff);
}

static int QuantizeBlock(int16_t in[16], int16_t out[16],
                         const VP8Matrix* const mtx) {
  return DoQuantizeBlock(in, out, &mtx->sharpen_[0], mtx);
}

static int QuantizeBlockWHT(int16_t in[16], int16_t out[16],
                            const VP8Matrix* const mtx) {
  return DoQuantizeBlock(in, out, NULL, mtx);
}

static int Quantize2Blocks(int16_t in[32], int16_t out[32],
                           const VP8Matrix* const mtx) {
  int nz;
  const uint16_t* const sharpen = &mtx->sharpen_[0];
  nz  = DoQuantizeBlock(in + 0 * 16, out + 0 * 16, sharpen, mtx) << 0;
  nz |= DoQuantizeBlock(in + 1 * 16, out + 1 * 16, sharpen, mtx) << 1;
  return nz;
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8EncDspInitSSE2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8EncDspInitSSE2(void) {
  VP8CollectHistogram = CollectHistogram;
  VP8EncPredLuma16 = Intra16Preds;
  VP8EncPredChroma8 = IntraChromaPreds;
  VP8EncPredLuma4 = Intra4Preds;
  VP8EncQuantizeBlock = QuantizeBlock;
  VP8EncQuantize2Blocks = Quantize2Blocks;
  VP8EncQuantizeBlockWHT = QuantizeBlockWHT;
  VP8ITransform = ITransform;
  VP8FTransform = FTransform;
  VP8FTransform2 = FTransform2;
  VP8FTransformWHT = FTransformWHT;
  VP8SSE16x16 = SSE16x16;
  VP8SSE16x8 = SSE16x8;
  VP8SSE8x8 = SSE8x8;
  VP8SSE4x4 = SSE4x4;
  VP8TDisto4x4 = Disto4x4;
  VP8TDisto16x16 = Disto16x16;
  VP8Mean16x4 = Mean16x4;
}

//------------------------------------------------------------------------------
// SSIM / PSNR entry point (TODO(skal): move to its own file later)

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
      SubtractAndAccumulate(a0, b0, &sum1);
      sum = _mm_add_epi32(sum, sum1);
      a0 = _mm_loadu_si128((const __m128i*)&src1[i]);
      b0 = _mm_loadu_si128((const __m128i*)&src2[i]);
      i += 16;
      SubtractAndAccumulate(a1, b1, &sum2);
      sum = _mm_add_epi32(sum, sum2);
    }
    SubtractAndAccumulate(a0, b0, &sum1);
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

static uint32_t HorizontalAdd16b(const __m128i* const m) {
  uint16_t tmp[8];
  const __m128i a = _mm_srli_si128(*m, 8);
  const __m128i b = _mm_add_epi16(*m, a);
  _mm_storeu_si128((__m128i*)tmp, b);
  return (uint32_t)tmp[3] + tmp[2] + tmp[1] + tmp[0];
}

static uint32_t HorizontalAdd32b(const __m128i* const m) {
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
  stats.xm  = HorizontalAdd16b(&xm);
  stats.ym  = HorizontalAdd16b(&ym);
  stats.xxm = HorizontalAdd32b(&xxm);
  stats.xym = HorizontalAdd32b(&xym);
  stats.yym = HorizontalAdd32b(&yym);
  return VP8SSIMFromStats(&stats);
}

extern void VP8SSIMDspInitSSE2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8SSIMDspInitSSE2(void) {
  VP8AccumulateSSE = AccumulateSSE_SSE2;
  VP8SSIMGet = SSIMGet_SSE2;
}

#else  // !WEBP_USE_SSE2

WEBP_DSP_INIT_STUB(VP8EncDspInitSSE2)
WEBP_DSP_INIT_STUB(VP8SSIMDspInitSSE2)

#endif  // WEBP_USE_SSE2
