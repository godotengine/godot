/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>  // SSE2

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_dsp/x86/fwd_txfm_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"
#include "vpx_ports/mem.h"

// TODO(jingning) The high bit-depth functions need rework for performance.
// After we properly fix the high bit-depth function implementations, this
// file's dependency should be substantially simplified.
#if DCT_HIGH_BIT_DEPTH
#define ADD_EPI16 _mm_adds_epi16
#define SUB_EPI16 _mm_subs_epi16

#else
#define ADD_EPI16 _mm_add_epi16
#define SUB_EPI16 _mm_sub_epi16
#endif

void FDCT4x4_2D(const int16_t *input, tran_low_t *output, int stride) {
  // This 2D transform implements 4 vertical 1D transforms followed
  // by 4 horizontal 1D transforms.  The multiplies and adds are as given
  // by Chen, Smith and Fralick ('77).  The commands for moving the data
  // around have been minimized by hand.
  // For the purposes of the comments, the 16 inputs are referred to at i0
  // through iF (in raster order), intermediate variables are a0, b0, c0
  // through f, and correspond to the in-place computations mapped to input
  // locations.  The outputs, o0 through oF are labeled according to the
  // output locations.

  // Constants
  // These are the coefficients used for the multiplies.
  // In the comments, pN means cos(N pi /64) and mN is -cos(N pi /64),
  // where cospi_N_64 = cos(N pi /64)
  const __m128i k__cospi_A =
      octa_set_epi16(cospi_16_64, cospi_16_64, cospi_16_64, cospi_16_64,
                     cospi_16_64, -cospi_16_64, cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_B =
      octa_set_epi16(cospi_16_64, -cospi_16_64, cospi_16_64, -cospi_16_64,
                     cospi_16_64, cospi_16_64, cospi_16_64, cospi_16_64);
  const __m128i k__cospi_C =
      octa_set_epi16(cospi_8_64, cospi_24_64, cospi_8_64, cospi_24_64,
                     cospi_24_64, -cospi_8_64, cospi_24_64, -cospi_8_64);
  const __m128i k__cospi_D =
      octa_set_epi16(cospi_24_64, -cospi_8_64, cospi_24_64, -cospi_8_64,
                     cospi_8_64, cospi_24_64, cospi_8_64, cospi_24_64);
  const __m128i k__cospi_E =
      octa_set_epi16(cospi_16_64, cospi_16_64, cospi_16_64, cospi_16_64,
                     cospi_16_64, cospi_16_64, cospi_16_64, cospi_16_64);
  const __m128i k__cospi_F =
      octa_set_epi16(cospi_16_64, -cospi_16_64, cospi_16_64, -cospi_16_64,
                     cospi_16_64, -cospi_16_64, cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_G =
      octa_set_epi16(cospi_8_64, cospi_24_64, cospi_8_64, cospi_24_64,
                     -cospi_8_64, -cospi_24_64, -cospi_8_64, -cospi_24_64);
  const __m128i k__cospi_H =
      octa_set_epi16(cospi_24_64, -cospi_8_64, cospi_24_64, -cospi_8_64,
                     -cospi_24_64, cospi_8_64, -cospi_24_64, cospi_8_64);

  const __m128i k__DCT_CONST_ROUNDING = _mm_set1_epi32(DCT_CONST_ROUNDING);
  // This second rounding constant saves doing some extra adds at the end
  const __m128i k__DCT_CONST_ROUNDING2 =
      _mm_set1_epi32(DCT_CONST_ROUNDING + (DCT_CONST_ROUNDING << 1));
  const int DCT_CONST_BITS2 = DCT_CONST_BITS + 2;
  const __m128i k__nonzero_bias_a = _mm_setr_epi16(0, 1, 1, 1, 1, 1, 1, 1);
  const __m128i k__nonzero_bias_b = _mm_setr_epi16(1, 0, 0, 0, 0, 0, 0, 0);
  __m128i in0, in1;
#if DCT_HIGH_BIT_DEPTH
  __m128i cmp0, cmp1;
  int test, overflow;
#endif

  // Load inputs.
  in0 = _mm_loadl_epi64((const __m128i *)(input + 0 * stride));
  in1 = _mm_loadl_epi64((const __m128i *)(input + 1 * stride));
  in1 = _mm_unpacklo_epi64(
      in1, _mm_loadl_epi64((const __m128i *)(input + 2 * stride)));
  in0 = _mm_unpacklo_epi64(
      in0, _mm_loadl_epi64((const __m128i *)(input + 3 * stride)));
// in0 = [i0 i1 i2 i3 iC iD iE iF]
// in1 = [i4 i5 i6 i7 i8 i9 iA iB]
#if DCT_HIGH_BIT_DEPTH
  // Check inputs small enough to use optimised code
  cmp0 = _mm_xor_si128(_mm_cmpgt_epi16(in0, _mm_set1_epi16(0x3ff)),
                       _mm_cmplt_epi16(in0, _mm_set1_epi16((int16_t)0xfc00)));
  cmp1 = _mm_xor_si128(_mm_cmpgt_epi16(in1, _mm_set1_epi16(0x3ff)),
                       _mm_cmplt_epi16(in1, _mm_set1_epi16((int16_t)0xfc00)));
  test = _mm_movemask_epi8(_mm_or_si128(cmp0, cmp1));
  if (test) {
    vpx_highbd_fdct4x4_c(input, output, stride);
    return;
  }
#endif  // DCT_HIGH_BIT_DEPTH

  // multiply by 16 to give some extra precision
  in0 = _mm_slli_epi16(in0, 4);
  in1 = _mm_slli_epi16(in1, 4);
  // if (i == 0 && input[0]) input[0] += 1;
  // add 1 to the upper left pixel if it is non-zero, which helps reduce
  // the round-trip error
  {
    // The mask will only contain whether the first value is zero, all
    // other comparison will fail as something shifted by 4 (above << 4)
    // can never be equal to one. To increment in the non-zero case, we
    // add the mask and one for the first element:
    //   - if zero, mask = -1, v = v - 1 + 1 = v
    //   - if non-zero, mask = 0, v = v + 0 + 1 = v + 1
    __m128i mask = _mm_cmpeq_epi16(in0, k__nonzero_bias_a);
    in0 = _mm_add_epi16(in0, mask);
    in0 = _mm_add_epi16(in0, k__nonzero_bias_b);
  }
  // There are 4 total stages, alternating between an add/subtract stage
  // followed by an multiply-and-add stage.
  {
    // Stage 1: Add/subtract

    // in0 = [i0 i1 i2 i3 iC iD iE iF]
    // in1 = [i4 i5 i6 i7 i8 i9 iA iB]
    const __m128i r0 = _mm_unpacklo_epi16(in0, in1);
    const __m128i r1 = _mm_unpackhi_epi16(in0, in1);
    // r0 = [i0 i4 i1 i5 i2 i6 i3 i7]
    // r1 = [iC i8 iD i9 iE iA iF iB]
    const __m128i r2 = _mm_shuffle_epi32(r0, 0xB4);
    const __m128i r3 = _mm_shuffle_epi32(r1, 0xB4);
    // r2 = [i0 i4 i1 i5 i3 i7 i2 i6]
    // r3 = [iC i8 iD i9 iF iB iE iA]

    const __m128i t0 = _mm_add_epi16(r2, r3);
    const __m128i t1 = _mm_sub_epi16(r2, r3);
    // t0 = [a0 a4 a1 a5 a3 a7 a2 a6]
    // t1 = [aC a8 aD a9 aF aB aE aA]

    // Stage 2: multiply by constants (which gets us into 32 bits).
    // The constants needed here are:
    // k__cospi_A = [p16 p16 p16 p16 p16 m16 p16 m16]
    // k__cospi_B = [p16 m16 p16 m16 p16 p16 p16 p16]
    // k__cospi_C = [p08 p24 p08 p24 p24 m08 p24 m08]
    // k__cospi_D = [p24 m08 p24 m08 p08 p24 p08 p24]
    const __m128i u0 = _mm_madd_epi16(t0, k__cospi_A);
    const __m128i u2 = _mm_madd_epi16(t0, k__cospi_B);
    const __m128i u1 = _mm_madd_epi16(t1, k__cospi_C);
    const __m128i u3 = _mm_madd_epi16(t1, k__cospi_D);
    // Then add and right-shift to get back to 16-bit range
    const __m128i v0 = _mm_add_epi32(u0, k__DCT_CONST_ROUNDING);
    const __m128i v1 = _mm_add_epi32(u1, k__DCT_CONST_ROUNDING);
    const __m128i v2 = _mm_add_epi32(u2, k__DCT_CONST_ROUNDING);
    const __m128i v3 = _mm_add_epi32(u3, k__DCT_CONST_ROUNDING);
    const __m128i w0 = _mm_srai_epi32(v0, DCT_CONST_BITS);
    const __m128i w1 = _mm_srai_epi32(v1, DCT_CONST_BITS);
    const __m128i w2 = _mm_srai_epi32(v2, DCT_CONST_BITS);
    const __m128i w3 = _mm_srai_epi32(v3, DCT_CONST_BITS);
    // w0 = [b0 b1 b7 b6]
    // w1 = [b8 b9 bF bE]
    // w2 = [b4 b5 b3 b2]
    // w3 = [bC bD bB bA]
    const __m128i x0 = _mm_packs_epi32(w0, w1);
    const __m128i x1 = _mm_packs_epi32(w2, w3);
#if DCT_HIGH_BIT_DEPTH
    overflow = check_epi16_overflow_x2(&x0, &x1);
    if (overflow) {
      vpx_highbd_fdct4x4_c(input, output, stride);
      return;
    }
#endif  // DCT_HIGH_BIT_DEPTH
    // x0 = [b0 b1 b7 b6 b8 b9 bF bE]
    // x1 = [b4 b5 b3 b2 bC bD bB bA]
    in0 = _mm_shuffle_epi32(x0, 0xD8);
    in1 = _mm_shuffle_epi32(x1, 0x8D);
    // in0 = [b0 b1 b8 b9 b7 b6 bF bE]
    // in1 = [b3 b2 bB bA b4 b5 bC bD]
  }
  {
    // vertical DCTs finished. Now we do the horizontal DCTs.
    // Stage 3: Add/subtract

    const __m128i t0 = ADD_EPI16(in0, in1);
    const __m128i t1 = SUB_EPI16(in0, in1);
// t0 = [c0 c1 c8 c9  c4  c5  cC  cD]
// t1 = [c3 c2 cB cA -c7 -c6 -cF -cE]
#if DCT_HIGH_BIT_DEPTH
    overflow = check_epi16_overflow_x2(&t0, &t1);
    if (overflow) {
      vpx_highbd_fdct4x4_c(input, output, stride);
      return;
    }
#endif  // DCT_HIGH_BIT_DEPTH

    // Stage 4: multiply by constants (which gets us into 32 bits).
    {
      // The constants needed here are:
      // k__cospi_E = [p16 p16 p16 p16 p16 p16 p16 p16]
      // k__cospi_F = [p16 m16 p16 m16 p16 m16 p16 m16]
      // k__cospi_G = [p08 p24 p08 p24 m08 m24 m08 m24]
      // k__cospi_H = [p24 m08 p24 m08 m24 p08 m24 p08]
      const __m128i u0 = _mm_madd_epi16(t0, k__cospi_E);
      const __m128i u1 = _mm_madd_epi16(t0, k__cospi_F);
      const __m128i u2 = _mm_madd_epi16(t1, k__cospi_G);
      const __m128i u3 = _mm_madd_epi16(t1, k__cospi_H);
      // Then add and right-shift to get back to 16-bit range
      // but this combines the final right-shift as well to save operations
      // This unusual rounding operations is to maintain bit-accurate
      // compatibility with the c version of this function which has two
      // rounding steps in a row.
      const __m128i v0 = _mm_add_epi32(u0, k__DCT_CONST_ROUNDING2);
      const __m128i v1 = _mm_add_epi32(u1, k__DCT_CONST_ROUNDING2);
      const __m128i v2 = _mm_add_epi32(u2, k__DCT_CONST_ROUNDING2);
      const __m128i v3 = _mm_add_epi32(u3, k__DCT_CONST_ROUNDING2);
      const __m128i w0 = _mm_srai_epi32(v0, DCT_CONST_BITS2);
      const __m128i w1 = _mm_srai_epi32(v1, DCT_CONST_BITS2);
      const __m128i w2 = _mm_srai_epi32(v2, DCT_CONST_BITS2);
      const __m128i w3 = _mm_srai_epi32(v3, DCT_CONST_BITS2);
      // w0 = [o0 o4 o8 oC]
      // w1 = [o2 o6 oA oE]
      // w2 = [o1 o5 o9 oD]
      // w3 = [o3 o7 oB oF]
      // remember the o's are numbered according to the correct output location
      const __m128i x0 = _mm_packs_epi32(w0, w1);
      const __m128i x1 = _mm_packs_epi32(w2, w3);
#if DCT_HIGH_BIT_DEPTH
      overflow = check_epi16_overflow_x2(&x0, &x1);
      if (overflow) {
        vpx_highbd_fdct4x4_c(input, output, stride);
        return;
      }
#endif  // DCT_HIGH_BIT_DEPTH
      {
        // x0 = [o0 o4 o8 oC o2 o6 oA oE]
        // x1 = [o1 o5 o9 oD o3 o7 oB oF]
        const __m128i y0 = _mm_unpacklo_epi16(x0, x1);
        const __m128i y1 = _mm_unpackhi_epi16(x0, x1);
        // y0 = [o0 o1 o4 o5 o8 o9 oC oD]
        // y1 = [o2 o3 o6 o7 oA oB oE oF]
        in0 = _mm_unpacklo_epi32(y0, y1);
        // in0 = [o0 o1 o2 o3 o4 o5 o6 o7]
        in1 = _mm_unpackhi_epi32(y0, y1);
        // in1 = [o8 o9 oA oB oC oD oE oF]
      }
    }
  }
  // Post-condition (v + 1) >> 2 is now incorporated into previous
  // add and right-shift commands.  Only 2 store instructions needed
  // because we are using the fact that 1/3 are stored just after 0/2.
  storeu_output(&in0, output + 0 * 4);
  storeu_output(&in1, output + 2 * 4);
}

void FDCT8x8_2D(const int16_t *input, tran_low_t *output, int stride) {
  int pass;
  // Constants
  //    When we use them, in one case, they are all the same. In all others
  //    it's a pair of them that we need to repeat four times. This is done
  //    by constructing the 32 bit constant corresponding to that pair.
  const __m128i k__cospi_p16_p16 = _mm_set1_epi16(cospi_16_64);
  const __m128i k__cospi_p16_m16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_p24_p08 = pair_set_epi16(cospi_24_64, cospi_8_64);
  const __m128i k__cospi_m08_p24 = pair_set_epi16(-cospi_8_64, cospi_24_64);
  const __m128i k__cospi_p28_p04 = pair_set_epi16(cospi_28_64, cospi_4_64);
  const __m128i k__cospi_m04_p28 = pair_set_epi16(-cospi_4_64, cospi_28_64);
  const __m128i k__cospi_p12_p20 = pair_set_epi16(cospi_12_64, cospi_20_64);
  const __m128i k__cospi_m20_p12 = pair_set_epi16(-cospi_20_64, cospi_12_64);
  const __m128i k__DCT_CONST_ROUNDING = _mm_set1_epi32(DCT_CONST_ROUNDING);
#if DCT_HIGH_BIT_DEPTH
  int overflow;
#endif
  // Load input
  __m128i in0 = _mm_load_si128((const __m128i *)(input + 0 * stride));
  __m128i in1 = _mm_load_si128((const __m128i *)(input + 1 * stride));
  __m128i in2 = _mm_load_si128((const __m128i *)(input + 2 * stride));
  __m128i in3 = _mm_load_si128((const __m128i *)(input + 3 * stride));
  __m128i in4 = _mm_load_si128((const __m128i *)(input + 4 * stride));
  __m128i in5 = _mm_load_si128((const __m128i *)(input + 5 * stride));
  __m128i in6 = _mm_load_si128((const __m128i *)(input + 6 * stride));
  __m128i in7 = _mm_load_si128((const __m128i *)(input + 7 * stride));
  // Pre-condition input (shift by two)
  in0 = _mm_slli_epi16(in0, 2);
  in1 = _mm_slli_epi16(in1, 2);
  in2 = _mm_slli_epi16(in2, 2);
  in3 = _mm_slli_epi16(in3, 2);
  in4 = _mm_slli_epi16(in4, 2);
  in5 = _mm_slli_epi16(in5, 2);
  in6 = _mm_slli_epi16(in6, 2);
  in7 = _mm_slli_epi16(in7, 2);

  // We do two passes, first the columns, then the rows. The results of the
  // first pass are transposed so that the same column code can be reused. The
  // results of the second pass are also transposed so that the rows (processed
  // as columns) are put back in row positions.
  for (pass = 0; pass < 2; pass++) {
    // To store results of each pass before the transpose.
    __m128i res0, res1, res2, res3, res4, res5, res6, res7;
    // Add/subtract
    const __m128i q0 = ADD_EPI16(in0, in7);
    const __m128i q1 = ADD_EPI16(in1, in6);
    const __m128i q2 = ADD_EPI16(in2, in5);
    const __m128i q3 = ADD_EPI16(in3, in4);
    const __m128i q4 = SUB_EPI16(in3, in4);
    const __m128i q5 = SUB_EPI16(in2, in5);
    const __m128i q6 = SUB_EPI16(in1, in6);
    const __m128i q7 = SUB_EPI16(in0, in7);
#if DCT_HIGH_BIT_DEPTH
    if (pass == 1) {
      overflow =
          check_epi16_overflow_x8(&q0, &q1, &q2, &q3, &q4, &q5, &q6, &q7);
      if (overflow) {
        vpx_highbd_fdct8x8_c(input, output, stride);
        return;
      }
    }
#endif  // DCT_HIGH_BIT_DEPTH
    // Work on first four results
    {
      // Add/subtract
      const __m128i r0 = ADD_EPI16(q0, q3);
      const __m128i r1 = ADD_EPI16(q1, q2);
      const __m128i r2 = SUB_EPI16(q1, q2);
      const __m128i r3 = SUB_EPI16(q0, q3);
#if DCT_HIGH_BIT_DEPTH
      overflow = check_epi16_overflow_x4(&r0, &r1, &r2, &r3);
      if (overflow) {
        vpx_highbd_fdct8x8_c(input, output, stride);
        return;
      }
#endif  // DCT_HIGH_BIT_DEPTH
      // Interleave to do the multiply by constants which gets us into 32bits
      {
        const __m128i t0 = _mm_unpacklo_epi16(r0, r1);
        const __m128i t1 = _mm_unpackhi_epi16(r0, r1);
        const __m128i t2 = _mm_unpacklo_epi16(r2, r3);
        const __m128i t3 = _mm_unpackhi_epi16(r2, r3);
        const __m128i u0 = _mm_madd_epi16(t0, k__cospi_p16_p16);
        const __m128i u1 = _mm_madd_epi16(t1, k__cospi_p16_p16);
        const __m128i u2 = _mm_madd_epi16(t0, k__cospi_p16_m16);
        const __m128i u3 = _mm_madd_epi16(t1, k__cospi_p16_m16);
        const __m128i u4 = _mm_madd_epi16(t2, k__cospi_p24_p08);
        const __m128i u5 = _mm_madd_epi16(t3, k__cospi_p24_p08);
        const __m128i u6 = _mm_madd_epi16(t2, k__cospi_m08_p24);
        const __m128i u7 = _mm_madd_epi16(t3, k__cospi_m08_p24);
        // dct_const_round_shift
        const __m128i v0 = _mm_add_epi32(u0, k__DCT_CONST_ROUNDING);
        const __m128i v1 = _mm_add_epi32(u1, k__DCT_CONST_ROUNDING);
        const __m128i v2 = _mm_add_epi32(u2, k__DCT_CONST_ROUNDING);
        const __m128i v3 = _mm_add_epi32(u3, k__DCT_CONST_ROUNDING);
        const __m128i v4 = _mm_add_epi32(u4, k__DCT_CONST_ROUNDING);
        const __m128i v5 = _mm_add_epi32(u5, k__DCT_CONST_ROUNDING);
        const __m128i v6 = _mm_add_epi32(u6, k__DCT_CONST_ROUNDING);
        const __m128i v7 = _mm_add_epi32(u7, k__DCT_CONST_ROUNDING);
        const __m128i w0 = _mm_srai_epi32(v0, DCT_CONST_BITS);
        const __m128i w1 = _mm_srai_epi32(v1, DCT_CONST_BITS);
        const __m128i w2 = _mm_srai_epi32(v2, DCT_CONST_BITS);
        const __m128i w3 = _mm_srai_epi32(v3, DCT_CONST_BITS);
        const __m128i w4 = _mm_srai_epi32(v4, DCT_CONST_BITS);
        const __m128i w5 = _mm_srai_epi32(v5, DCT_CONST_BITS);
        const __m128i w6 = _mm_srai_epi32(v6, DCT_CONST_BITS);
        const __m128i w7 = _mm_srai_epi32(v7, DCT_CONST_BITS);
        // Combine
        res0 = _mm_packs_epi32(w0, w1);
        res4 = _mm_packs_epi32(w2, w3);
        res2 = _mm_packs_epi32(w4, w5);
        res6 = _mm_packs_epi32(w6, w7);
#if DCT_HIGH_BIT_DEPTH
        overflow = check_epi16_overflow_x4(&res0, &res4, &res2, &res6);
        if (overflow) {
          vpx_highbd_fdct8x8_c(input, output, stride);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
      }
    }
    // Work on next four results
    {
      // Interleave to do the multiply by constants which gets us into 32bits
      const __m128i d0 = _mm_unpacklo_epi16(q6, q5);
      const __m128i d1 = _mm_unpackhi_epi16(q6, q5);
      const __m128i e0 = _mm_madd_epi16(d0, k__cospi_p16_m16);
      const __m128i e1 = _mm_madd_epi16(d1, k__cospi_p16_m16);
      const __m128i e2 = _mm_madd_epi16(d0, k__cospi_p16_p16);
      const __m128i e3 = _mm_madd_epi16(d1, k__cospi_p16_p16);
      // dct_const_round_shift
      const __m128i f0 = _mm_add_epi32(e0, k__DCT_CONST_ROUNDING);
      const __m128i f1 = _mm_add_epi32(e1, k__DCT_CONST_ROUNDING);
      const __m128i f2 = _mm_add_epi32(e2, k__DCT_CONST_ROUNDING);
      const __m128i f3 = _mm_add_epi32(e3, k__DCT_CONST_ROUNDING);
      const __m128i s0 = _mm_srai_epi32(f0, DCT_CONST_BITS);
      const __m128i s1 = _mm_srai_epi32(f1, DCT_CONST_BITS);
      const __m128i s2 = _mm_srai_epi32(f2, DCT_CONST_BITS);
      const __m128i s3 = _mm_srai_epi32(f3, DCT_CONST_BITS);
      // Combine
      const __m128i r0 = _mm_packs_epi32(s0, s1);
      const __m128i r1 = _mm_packs_epi32(s2, s3);
#if DCT_HIGH_BIT_DEPTH
      overflow = check_epi16_overflow_x2(&r0, &r1);
      if (overflow) {
        vpx_highbd_fdct8x8_c(input, output, stride);
        return;
      }
#endif  // DCT_HIGH_BIT_DEPTH
      {
        // Add/subtract
        const __m128i x0 = ADD_EPI16(q4, r0);
        const __m128i x1 = SUB_EPI16(q4, r0);
        const __m128i x2 = SUB_EPI16(q7, r1);
        const __m128i x3 = ADD_EPI16(q7, r1);
#if DCT_HIGH_BIT_DEPTH
        overflow = check_epi16_overflow_x4(&x0, &x1, &x2, &x3);
        if (overflow) {
          vpx_highbd_fdct8x8_c(input, output, stride);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
        // Interleave to do the multiply by constants which gets us into 32bits
        {
          const __m128i t0 = _mm_unpacklo_epi16(x0, x3);
          const __m128i t1 = _mm_unpackhi_epi16(x0, x3);
          const __m128i t2 = _mm_unpacklo_epi16(x1, x2);
          const __m128i t3 = _mm_unpackhi_epi16(x1, x2);
          const __m128i u0 = _mm_madd_epi16(t0, k__cospi_p28_p04);
          const __m128i u1 = _mm_madd_epi16(t1, k__cospi_p28_p04);
          const __m128i u2 = _mm_madd_epi16(t0, k__cospi_m04_p28);
          const __m128i u3 = _mm_madd_epi16(t1, k__cospi_m04_p28);
          const __m128i u4 = _mm_madd_epi16(t2, k__cospi_p12_p20);
          const __m128i u5 = _mm_madd_epi16(t3, k__cospi_p12_p20);
          const __m128i u6 = _mm_madd_epi16(t2, k__cospi_m20_p12);
          const __m128i u7 = _mm_madd_epi16(t3, k__cospi_m20_p12);
          // dct_const_round_shift
          const __m128i v0 = _mm_add_epi32(u0, k__DCT_CONST_ROUNDING);
          const __m128i v1 = _mm_add_epi32(u1, k__DCT_CONST_ROUNDING);
          const __m128i v2 = _mm_add_epi32(u2, k__DCT_CONST_ROUNDING);
          const __m128i v3 = _mm_add_epi32(u3, k__DCT_CONST_ROUNDING);
          const __m128i v4 = _mm_add_epi32(u4, k__DCT_CONST_ROUNDING);
          const __m128i v5 = _mm_add_epi32(u5, k__DCT_CONST_ROUNDING);
          const __m128i v6 = _mm_add_epi32(u6, k__DCT_CONST_ROUNDING);
          const __m128i v7 = _mm_add_epi32(u7, k__DCT_CONST_ROUNDING);
          const __m128i w0 = _mm_srai_epi32(v0, DCT_CONST_BITS);
          const __m128i w1 = _mm_srai_epi32(v1, DCT_CONST_BITS);
          const __m128i w2 = _mm_srai_epi32(v2, DCT_CONST_BITS);
          const __m128i w3 = _mm_srai_epi32(v3, DCT_CONST_BITS);
          const __m128i w4 = _mm_srai_epi32(v4, DCT_CONST_BITS);
          const __m128i w5 = _mm_srai_epi32(v5, DCT_CONST_BITS);
          const __m128i w6 = _mm_srai_epi32(v6, DCT_CONST_BITS);
          const __m128i w7 = _mm_srai_epi32(v7, DCT_CONST_BITS);
          // Combine
          res1 = _mm_packs_epi32(w0, w1);
          res7 = _mm_packs_epi32(w2, w3);
          res5 = _mm_packs_epi32(w4, w5);
          res3 = _mm_packs_epi32(w6, w7);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&res1, &res7, &res5, &res3);
          if (overflow) {
            vpx_highbd_fdct8x8_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
      }
    }
    // Transpose the 8x8.
    {
      // 00 01 02 03 04 05 06 07
      // 10 11 12 13 14 15 16 17
      // 20 21 22 23 24 25 26 27
      // 30 31 32 33 34 35 36 37
      // 40 41 42 43 44 45 46 47
      // 50 51 52 53 54 55 56 57
      // 60 61 62 63 64 65 66 67
      // 70 71 72 73 74 75 76 77
      const __m128i tr0_0 = _mm_unpacklo_epi16(res0, res1);
      const __m128i tr0_1 = _mm_unpacklo_epi16(res2, res3);
      const __m128i tr0_2 = _mm_unpackhi_epi16(res0, res1);
      const __m128i tr0_3 = _mm_unpackhi_epi16(res2, res3);
      const __m128i tr0_4 = _mm_unpacklo_epi16(res4, res5);
      const __m128i tr0_5 = _mm_unpacklo_epi16(res6, res7);
      const __m128i tr0_6 = _mm_unpackhi_epi16(res4, res5);
      const __m128i tr0_7 = _mm_unpackhi_epi16(res6, res7);
      // 00 10 01 11 02 12 03 13
      // 20 30 21 31 22 32 23 33
      // 04 14 05 15 06 16 07 17
      // 24 34 25 35 26 36 27 37
      // 40 50 41 51 42 52 43 53
      // 60 70 61 71 62 72 63 73
      // 54 54 55 55 56 56 57 57
      // 64 74 65 75 66 76 67 77
      const __m128i tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1);
      const __m128i tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3);
      const __m128i tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1);
      const __m128i tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3);
      const __m128i tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5);
      const __m128i tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7);
      const __m128i tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5);
      const __m128i tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7);
      // 00 10 20 30 01 11 21 31
      // 40 50 60 70 41 51 61 71
      // 02 12 22 32 03 13 23 33
      // 42 52 62 72 43 53 63 73
      // 04 14 24 34 05 15 21 36
      // 44 54 64 74 45 55 61 76
      // 06 16 26 36 07 17 27 37
      // 46 56 66 76 47 57 67 77
      in0 = _mm_unpacklo_epi64(tr1_0, tr1_4);
      in1 = _mm_unpackhi_epi64(tr1_0, tr1_4);
      in2 = _mm_unpacklo_epi64(tr1_2, tr1_6);
      in3 = _mm_unpackhi_epi64(tr1_2, tr1_6);
      in4 = _mm_unpacklo_epi64(tr1_1, tr1_5);
      in5 = _mm_unpackhi_epi64(tr1_1, tr1_5);
      in6 = _mm_unpacklo_epi64(tr1_3, tr1_7);
      in7 = _mm_unpackhi_epi64(tr1_3, tr1_7);
      // 00 10 20 30 40 50 60 70
      // 01 11 21 31 41 51 61 71
      // 02 12 22 32 42 52 62 72
      // 03 13 23 33 43 53 63 73
      // 04 14 24 34 44 54 64 74
      // 05 15 25 35 45 55 65 75
      // 06 16 26 36 46 56 66 76
      // 07 17 27 37 47 57 67 77
    }
  }
  // Post-condition output and store it
  {
    // Post-condition (division by two)
    //    division of two 16 bits signed numbers using shifts
    //    n / 2 = (n - (n >> 15)) >> 1
    const __m128i sign_in0 = _mm_srai_epi16(in0, 15);
    const __m128i sign_in1 = _mm_srai_epi16(in1, 15);
    const __m128i sign_in2 = _mm_srai_epi16(in2, 15);
    const __m128i sign_in3 = _mm_srai_epi16(in3, 15);
    const __m128i sign_in4 = _mm_srai_epi16(in4, 15);
    const __m128i sign_in5 = _mm_srai_epi16(in5, 15);
    const __m128i sign_in6 = _mm_srai_epi16(in6, 15);
    const __m128i sign_in7 = _mm_srai_epi16(in7, 15);
    in0 = _mm_sub_epi16(in0, sign_in0);
    in1 = _mm_sub_epi16(in1, sign_in1);
    in2 = _mm_sub_epi16(in2, sign_in2);
    in3 = _mm_sub_epi16(in3, sign_in3);
    in4 = _mm_sub_epi16(in4, sign_in4);
    in5 = _mm_sub_epi16(in5, sign_in5);
    in6 = _mm_sub_epi16(in6, sign_in6);
    in7 = _mm_sub_epi16(in7, sign_in7);
    in0 = _mm_srai_epi16(in0, 1);
    in1 = _mm_srai_epi16(in1, 1);
    in2 = _mm_srai_epi16(in2, 1);
    in3 = _mm_srai_epi16(in3, 1);
    in4 = _mm_srai_epi16(in4, 1);
    in5 = _mm_srai_epi16(in5, 1);
    in6 = _mm_srai_epi16(in6, 1);
    in7 = _mm_srai_epi16(in7, 1);
    // store results
    store_output(&in0, (output + 0 * 8));
    store_output(&in1, (output + 1 * 8));
    store_output(&in2, (output + 2 * 8));
    store_output(&in3, (output + 3 * 8));
    store_output(&in4, (output + 4 * 8));
    store_output(&in5, (output + 5 * 8));
    store_output(&in6, (output + 6 * 8));
    store_output(&in7, (output + 7 * 8));
  }
}

void FDCT16x16_2D(const int16_t *input, tran_low_t *output, int stride) {
  // The 2D transform is done with two passes which are actually pretty
  // similar. In the first one, we transform the columns and transpose
  // the results. In the second one, we transform the rows. To achieve that,
  // as the first pass results are transposed, we transpose the columns (that
  // is the transposed rows) and transpose the results (so that it goes back
  // in normal/row positions).
  int pass;
  // We need an intermediate buffer between passes.
  DECLARE_ALIGNED(16, int16_t, intermediate[256]);
  const int16_t *in = input;
  int16_t *out0 = intermediate;
  tran_low_t *out1 = output;
  // Constants
  //    When we use them, in one case, they are all the same. In all others
  //    it's a pair of them that we need to repeat four times. This is done
  //    by constructing the 32 bit constant corresponding to that pair.
  const __m128i k__cospi_p16_p16 = _mm_set1_epi16(cospi_16_64);
  const __m128i k__cospi_p16_m16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_p24_p08 = pair_set_epi16(cospi_24_64, cospi_8_64);
  const __m128i k__cospi_p08_m24 = pair_set_epi16(cospi_8_64, -cospi_24_64);
  const __m128i k__cospi_m08_p24 = pair_set_epi16(-cospi_8_64, cospi_24_64);
  const __m128i k__cospi_p28_p04 = pair_set_epi16(cospi_28_64, cospi_4_64);
  const __m128i k__cospi_m04_p28 = pair_set_epi16(-cospi_4_64, cospi_28_64);
  const __m128i k__cospi_p12_p20 = pair_set_epi16(cospi_12_64, cospi_20_64);
  const __m128i k__cospi_m20_p12 = pair_set_epi16(-cospi_20_64, cospi_12_64);
  const __m128i k__cospi_p30_p02 = pair_set_epi16(cospi_30_64, cospi_2_64);
  const __m128i k__cospi_p14_p18 = pair_set_epi16(cospi_14_64, cospi_18_64);
  const __m128i k__cospi_m02_p30 = pair_set_epi16(-cospi_2_64, cospi_30_64);
  const __m128i k__cospi_m18_p14 = pair_set_epi16(-cospi_18_64, cospi_14_64);
  const __m128i k__cospi_p22_p10 = pair_set_epi16(cospi_22_64, cospi_10_64);
  const __m128i k__cospi_p06_p26 = pair_set_epi16(cospi_6_64, cospi_26_64);
  const __m128i k__cospi_m10_p22 = pair_set_epi16(-cospi_10_64, cospi_22_64);
  const __m128i k__cospi_m26_p06 = pair_set_epi16(-cospi_26_64, cospi_6_64);
  const __m128i k__DCT_CONST_ROUNDING = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i kOne = _mm_set1_epi16(1);
  // Do the two transform/transpose passes
  for (pass = 0; pass < 2; ++pass) {
    // We process eight columns (transposed rows in second pass) at a time.
    int column_start;
#if DCT_HIGH_BIT_DEPTH
    int overflow;
#endif
    for (column_start = 0; column_start < 16; column_start += 8) {
      __m128i in00, in01, in02, in03, in04, in05, in06, in07;
      __m128i in08, in09, in10, in11, in12, in13, in14, in15;
      __m128i input0, input1, input2, input3, input4, input5, input6, input7;
      __m128i step1_0, step1_1, step1_2, step1_3;
      __m128i step1_4, step1_5, step1_6, step1_7;
      __m128i step2_1, step2_2, step2_3, step2_4, step2_5, step2_6;
      __m128i step3_0, step3_1, step3_2, step3_3;
      __m128i step3_4, step3_5, step3_6, step3_7;
      __m128i res00, res01, res02, res03, res04, res05, res06, res07;
      __m128i res08, res09, res10, res11, res12, res13, res14, res15;
      // Load and pre-condition input.
      if (0 == pass) {
        in00 = _mm_load_si128((const __m128i *)(in + 0 * stride));
        in01 = _mm_load_si128((const __m128i *)(in + 1 * stride));
        in02 = _mm_load_si128((const __m128i *)(in + 2 * stride));
        in03 = _mm_load_si128((const __m128i *)(in + 3 * stride));
        in04 = _mm_load_si128((const __m128i *)(in + 4 * stride));
        in05 = _mm_load_si128((const __m128i *)(in + 5 * stride));
        in06 = _mm_load_si128((const __m128i *)(in + 6 * stride));
        in07 = _mm_load_si128((const __m128i *)(in + 7 * stride));
        in08 = _mm_load_si128((const __m128i *)(in + 8 * stride));
        in09 = _mm_load_si128((const __m128i *)(in + 9 * stride));
        in10 = _mm_load_si128((const __m128i *)(in + 10 * stride));
        in11 = _mm_load_si128((const __m128i *)(in + 11 * stride));
        in12 = _mm_load_si128((const __m128i *)(in + 12 * stride));
        in13 = _mm_load_si128((const __m128i *)(in + 13 * stride));
        in14 = _mm_load_si128((const __m128i *)(in + 14 * stride));
        in15 = _mm_load_si128((const __m128i *)(in + 15 * stride));
        // x = x << 2
        in00 = _mm_slli_epi16(in00, 2);
        in01 = _mm_slli_epi16(in01, 2);
        in02 = _mm_slli_epi16(in02, 2);
        in03 = _mm_slli_epi16(in03, 2);
        in04 = _mm_slli_epi16(in04, 2);
        in05 = _mm_slli_epi16(in05, 2);
        in06 = _mm_slli_epi16(in06, 2);
        in07 = _mm_slli_epi16(in07, 2);
        in08 = _mm_slli_epi16(in08, 2);
        in09 = _mm_slli_epi16(in09, 2);
        in10 = _mm_slli_epi16(in10, 2);
        in11 = _mm_slli_epi16(in11, 2);
        in12 = _mm_slli_epi16(in12, 2);
        in13 = _mm_slli_epi16(in13, 2);
        in14 = _mm_slli_epi16(in14, 2);
        in15 = _mm_slli_epi16(in15, 2);
      } else {
        in00 = _mm_load_si128((const __m128i *)(in + 0 * 16));
        in01 = _mm_load_si128((const __m128i *)(in + 1 * 16));
        in02 = _mm_load_si128((const __m128i *)(in + 2 * 16));
        in03 = _mm_load_si128((const __m128i *)(in + 3 * 16));
        in04 = _mm_load_si128((const __m128i *)(in + 4 * 16));
        in05 = _mm_load_si128((const __m128i *)(in + 5 * 16));
        in06 = _mm_load_si128((const __m128i *)(in + 6 * 16));
        in07 = _mm_load_si128((const __m128i *)(in + 7 * 16));
        in08 = _mm_load_si128((const __m128i *)(in + 8 * 16));
        in09 = _mm_load_si128((const __m128i *)(in + 9 * 16));
        in10 = _mm_load_si128((const __m128i *)(in + 10 * 16));
        in11 = _mm_load_si128((const __m128i *)(in + 11 * 16));
        in12 = _mm_load_si128((const __m128i *)(in + 12 * 16));
        in13 = _mm_load_si128((const __m128i *)(in + 13 * 16));
        in14 = _mm_load_si128((const __m128i *)(in + 14 * 16));
        in15 = _mm_load_si128((const __m128i *)(in + 15 * 16));
        // x = (x + 1) >> 2
        in00 = _mm_add_epi16(in00, kOne);
        in01 = _mm_add_epi16(in01, kOne);
        in02 = _mm_add_epi16(in02, kOne);
        in03 = _mm_add_epi16(in03, kOne);
        in04 = _mm_add_epi16(in04, kOne);
        in05 = _mm_add_epi16(in05, kOne);
        in06 = _mm_add_epi16(in06, kOne);
        in07 = _mm_add_epi16(in07, kOne);
        in08 = _mm_add_epi16(in08, kOne);
        in09 = _mm_add_epi16(in09, kOne);
        in10 = _mm_add_epi16(in10, kOne);
        in11 = _mm_add_epi16(in11, kOne);
        in12 = _mm_add_epi16(in12, kOne);
        in13 = _mm_add_epi16(in13, kOne);
        in14 = _mm_add_epi16(in14, kOne);
        in15 = _mm_add_epi16(in15, kOne);
        in00 = _mm_srai_epi16(in00, 2);
        in01 = _mm_srai_epi16(in01, 2);
        in02 = _mm_srai_epi16(in02, 2);
        in03 = _mm_srai_epi16(in03, 2);
        in04 = _mm_srai_epi16(in04, 2);
        in05 = _mm_srai_epi16(in05, 2);
        in06 = _mm_srai_epi16(in06, 2);
        in07 = _mm_srai_epi16(in07, 2);
        in08 = _mm_srai_epi16(in08, 2);
        in09 = _mm_srai_epi16(in09, 2);
        in10 = _mm_srai_epi16(in10, 2);
        in11 = _mm_srai_epi16(in11, 2);
        in12 = _mm_srai_epi16(in12, 2);
        in13 = _mm_srai_epi16(in13, 2);
        in14 = _mm_srai_epi16(in14, 2);
        in15 = _mm_srai_epi16(in15, 2);
      }
      in += 8;
      // Calculate input for the first 8 results.
      {
        input0 = ADD_EPI16(in00, in15);
        input1 = ADD_EPI16(in01, in14);
        input2 = ADD_EPI16(in02, in13);
        input3 = ADD_EPI16(in03, in12);
        input4 = ADD_EPI16(in04, in11);
        input5 = ADD_EPI16(in05, in10);
        input6 = ADD_EPI16(in06, in09);
        input7 = ADD_EPI16(in07, in08);
#if DCT_HIGH_BIT_DEPTH
        overflow = check_epi16_overflow_x8(&input0, &input1, &input2, &input3,
                                           &input4, &input5, &input6, &input7);
        if (overflow) {
          vpx_highbd_fdct16x16_c(input, output, stride);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
      }
      // Calculate input for the next 8 results.
      {
        step1_0 = SUB_EPI16(in07, in08);
        step1_1 = SUB_EPI16(in06, in09);
        step1_2 = SUB_EPI16(in05, in10);
        step1_3 = SUB_EPI16(in04, in11);
        step1_4 = SUB_EPI16(in03, in12);
        step1_5 = SUB_EPI16(in02, in13);
        step1_6 = SUB_EPI16(in01, in14);
        step1_7 = SUB_EPI16(in00, in15);
#if DCT_HIGH_BIT_DEPTH
        overflow =
            check_epi16_overflow_x8(&step1_0, &step1_1, &step1_2, &step1_3,
                                    &step1_4, &step1_5, &step1_6, &step1_7);
        if (overflow) {
          vpx_highbd_fdct16x16_c(input, output, stride);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
      }
      // Work on the first eight values; fdct8(input, even_results);
      {
        // Add/subtract
        const __m128i q0 = ADD_EPI16(input0, input7);
        const __m128i q1 = ADD_EPI16(input1, input6);
        const __m128i q2 = ADD_EPI16(input2, input5);
        const __m128i q3 = ADD_EPI16(input3, input4);
        const __m128i q4 = SUB_EPI16(input3, input4);
        const __m128i q5 = SUB_EPI16(input2, input5);
        const __m128i q6 = SUB_EPI16(input1, input6);
        const __m128i q7 = SUB_EPI16(input0, input7);
#if DCT_HIGH_BIT_DEPTH
        overflow =
            check_epi16_overflow_x8(&q0, &q1, &q2, &q3, &q4, &q5, &q6, &q7);
        if (overflow) {
          vpx_highbd_fdct16x16_c(input, output, stride);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
        // Work on first four results
        {
          // Add/subtract
          const __m128i r0 = ADD_EPI16(q0, q3);
          const __m128i r1 = ADD_EPI16(q1, q2);
          const __m128i r2 = SUB_EPI16(q1, q2);
          const __m128i r3 = SUB_EPI16(q0, q3);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&r0, &r1, &r2, &r3);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH

          // Interleave to do the multiply by constants which gets us
          // into 32 bits.
          {
            const __m128i t0 = _mm_unpacklo_epi16(r0, r1);
            const __m128i t1 = _mm_unpackhi_epi16(r0, r1);
            const __m128i t2 = _mm_unpacklo_epi16(r2, r3);
            const __m128i t3 = _mm_unpackhi_epi16(r2, r3);
            res00 = mult_round_shift(&t0, &t1, &k__cospi_p16_p16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
            res08 = mult_round_shift(&t0, &t1, &k__cospi_p16_m16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
            res04 = mult_round_shift(&t2, &t3, &k__cospi_p24_p08,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
            res12 = mult_round_shift(&t2, &t3, &k__cospi_m08_p24,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
            overflow = check_epi16_overflow_x4(&res00, &res08, &res04, &res12);
            if (overflow) {
              vpx_highbd_fdct16x16_c(input, output, stride);
              return;
            }
#endif  // DCT_HIGH_BIT_DEPTH
          }
        }
        // Work on next four results
        {
          // Interleave to do the multiply by constants which gets us
          // into 32 bits.
          const __m128i d0 = _mm_unpacklo_epi16(q6, q5);
          const __m128i d1 = _mm_unpackhi_epi16(q6, q5);
          const __m128i r0 =
              mult_round_shift(&d0, &d1, &k__cospi_p16_m16,
                               &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          const __m128i r1 =
              mult_round_shift(&d0, &d1, &k__cospi_p16_p16,
                               &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x2(&r0, &r1);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          {
            // Add/subtract
            const __m128i x0 = ADD_EPI16(q4, r0);
            const __m128i x1 = SUB_EPI16(q4, r0);
            const __m128i x2 = SUB_EPI16(q7, r1);
            const __m128i x3 = ADD_EPI16(q7, r1);
#if DCT_HIGH_BIT_DEPTH
            overflow = check_epi16_overflow_x4(&x0, &x1, &x2, &x3);
            if (overflow) {
              vpx_highbd_fdct16x16_c(input, output, stride);
              return;
            }
#endif  // DCT_HIGH_BIT_DEPTH

            // Interleave to do the multiply by constants which gets us
            // into 32 bits.
            {
              const __m128i t0 = _mm_unpacklo_epi16(x0, x3);
              const __m128i t1 = _mm_unpackhi_epi16(x0, x3);
              const __m128i t2 = _mm_unpacklo_epi16(x1, x2);
              const __m128i t3 = _mm_unpackhi_epi16(x1, x2);
              res02 = mult_round_shift(&t0, &t1, &k__cospi_p28_p04,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
              res14 = mult_round_shift(&t0, &t1, &k__cospi_m04_p28,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
              res10 = mult_round_shift(&t2, &t3, &k__cospi_p12_p20,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
              res06 = mult_round_shift(&t2, &t3, &k__cospi_m20_p12,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
              overflow =
                  check_epi16_overflow_x4(&res02, &res14, &res10, &res06);
              if (overflow) {
                vpx_highbd_fdct16x16_c(input, output, stride);
                return;
              }
#endif  // DCT_HIGH_BIT_DEPTH
            }
          }
        }
      }
      // Work on the next eight values; step1 -> odd_results
      {
        // step 2
        {
          const __m128i t0 = _mm_unpacklo_epi16(step1_5, step1_2);
          const __m128i t1 = _mm_unpackhi_epi16(step1_5, step1_2);
          const __m128i t2 = _mm_unpacklo_epi16(step1_4, step1_3);
          const __m128i t3 = _mm_unpackhi_epi16(step1_4, step1_3);
          step2_2 = mult_round_shift(&t0, &t1, &k__cospi_p16_m16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_3 = mult_round_shift(&t2, &t3, &k__cospi_p16_m16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_5 = mult_round_shift(&t0, &t1, &k__cospi_p16_p16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_4 = mult_round_shift(&t2, &t3, &k__cospi_p16_p16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x4(&step2_2, &step2_3, &step2_5, &step2_4);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // step 3
        {
          step3_0 = ADD_EPI16(step1_0, step2_3);
          step3_1 = ADD_EPI16(step1_1, step2_2);
          step3_2 = SUB_EPI16(step1_1, step2_2);
          step3_3 = SUB_EPI16(step1_0, step2_3);
          step3_4 = SUB_EPI16(step1_7, step2_4);
          step3_5 = SUB_EPI16(step1_6, step2_5);
          step3_6 = ADD_EPI16(step1_6, step2_5);
          step3_7 = ADD_EPI16(step1_7, step2_4);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&step3_0, &step3_1, &step3_2, &step3_3,
                                      &step3_4, &step3_5, &step3_6, &step3_7);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // step 4
        {
          const __m128i t0 = _mm_unpacklo_epi16(step3_1, step3_6);
          const __m128i t1 = _mm_unpackhi_epi16(step3_1, step3_6);
          const __m128i t2 = _mm_unpacklo_epi16(step3_2, step3_5);
          const __m128i t3 = _mm_unpackhi_epi16(step3_2, step3_5);
          step2_1 = mult_round_shift(&t0, &t1, &k__cospi_m08_p24,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_2 = mult_round_shift(&t2, &t3, &k__cospi_p24_p08,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_6 = mult_round_shift(&t0, &t1, &k__cospi_p24_p08,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          step2_5 = mult_round_shift(&t2, &t3, &k__cospi_p08_m24,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x4(&step2_1, &step2_2, &step2_6, &step2_5);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // step 5
        {
          step1_0 = ADD_EPI16(step3_0, step2_1);
          step1_1 = SUB_EPI16(step3_0, step2_1);
          step1_2 = ADD_EPI16(step3_3, step2_2);
          step1_3 = SUB_EPI16(step3_3, step2_2);
          step1_4 = SUB_EPI16(step3_4, step2_5);
          step1_5 = ADD_EPI16(step3_4, step2_5);
          step1_6 = SUB_EPI16(step3_7, step2_6);
          step1_7 = ADD_EPI16(step3_7, step2_6);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&step1_0, &step1_1, &step1_2, &step1_3,
                                      &step1_4, &step1_5, &step1_6, &step1_7);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // step 6
        {
          const __m128i t0 = _mm_unpacklo_epi16(step1_0, step1_7);
          const __m128i t1 = _mm_unpackhi_epi16(step1_0, step1_7);
          const __m128i t2 = _mm_unpacklo_epi16(step1_1, step1_6);
          const __m128i t3 = _mm_unpackhi_epi16(step1_1, step1_6);
          res01 = mult_round_shift(&t0, &t1, &k__cospi_p30_p02,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res09 = mult_round_shift(&t2, &t3, &k__cospi_p14_p18,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res15 = mult_round_shift(&t0, &t1, &k__cospi_m02_p30,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res07 = mult_round_shift(&t2, &t3, &k__cospi_m18_p14,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&res01, &res09, &res15, &res07);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i t0 = _mm_unpacklo_epi16(step1_2, step1_5);
          const __m128i t1 = _mm_unpackhi_epi16(step1_2, step1_5);
          const __m128i t2 = _mm_unpacklo_epi16(step1_3, step1_4);
          const __m128i t3 = _mm_unpackhi_epi16(step1_3, step1_4);
          res05 = mult_round_shift(&t0, &t1, &k__cospi_p22_p10,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res13 = mult_round_shift(&t2, &t3, &k__cospi_p06_p26,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res11 = mult_round_shift(&t0, &t1, &k__cospi_m10_p22,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          res03 = mult_round_shift(&t2, &t3, &k__cospi_m26_p06,
                                   &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&res05, &res13, &res11, &res03);
          if (overflow) {
            vpx_highbd_fdct16x16_c(input, output, stride);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
      }
      // Transpose the results, do it as two 8x8 transposes.
      transpose_and_output8x8(&res00, &res01, &res02, &res03, &res04, &res05,
                              &res06, &res07, pass, out0, out1);
      transpose_and_output8x8(&res08, &res09, &res10, &res11, &res12, &res13,
                              &res14, &res15, pass, out0 + 8, out1 + 8);
      if (pass == 0) {
        out0 += 8 * 16;
      } else {
        out1 += 8 * 16;
      }
    }
    // Setup in/out for next pass.
    in = intermediate;
  }
}

#undef ADD_EPI16
#undef SUB_EPI16
