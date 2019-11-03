/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

#define RECON_AND_STORE4X4(dest, in_x) \
{                                                     \
  __m128i d0 = _mm_cvtsi32_si128(*(const int *)(dest)); \
  d0 = _mm_unpacklo_epi8(d0, zero); \
  d0 = _mm_add_epi16(in_x, d0); \
  d0 = _mm_packus_epi16(d0, d0); \
  *(int *)(dest) = _mm_cvtsi128_si32(d0); \
}

void vpx_idct4x4_16_add_sse2(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i eight = _mm_set1_epi16(8);
  const __m128i cst = _mm_setr_epi16(
      (int16_t)cospi_16_64, (int16_t)cospi_16_64, (int16_t)cospi_16_64,
      (int16_t)-cospi_16_64, (int16_t)cospi_24_64, (int16_t)-cospi_8_64,
      (int16_t)cospi_8_64, (int16_t)cospi_24_64);
  const __m128i rounding = _mm_set1_epi32(DCT_CONST_ROUNDING);
  __m128i input0, input1, input2, input3;

  // Rows
  input0 = load_input_data(input);
  input2 = load_input_data(input + 8);

  // Construct i3, i1, i3, i1, i2, i0, i2, i0
  input0 = _mm_shufflelo_epi16(input0, 0xd8);
  input0 = _mm_shufflehi_epi16(input0, 0xd8);
  input2 = _mm_shufflelo_epi16(input2, 0xd8);
  input2 = _mm_shufflehi_epi16(input2, 0xd8);

  input1 = _mm_unpackhi_epi32(input0, input0);
  input0 = _mm_unpacklo_epi32(input0, input0);
  input3 = _mm_unpackhi_epi32(input2, input2);
  input2 = _mm_unpacklo_epi32(input2, input2);

  // Stage 1
  input0 = _mm_madd_epi16(input0, cst);
  input1 = _mm_madd_epi16(input1, cst);
  input2 = _mm_madd_epi16(input2, cst);
  input3 = _mm_madd_epi16(input3, cst);

  input0 = _mm_add_epi32(input0, rounding);
  input1 = _mm_add_epi32(input1, rounding);
  input2 = _mm_add_epi32(input2, rounding);
  input3 = _mm_add_epi32(input3, rounding);

  input0 = _mm_srai_epi32(input0, DCT_CONST_BITS);
  input1 = _mm_srai_epi32(input1, DCT_CONST_BITS);
  input2 = _mm_srai_epi32(input2, DCT_CONST_BITS);
  input3 = _mm_srai_epi32(input3, DCT_CONST_BITS);

  // Stage 2
  input0 = _mm_packs_epi32(input0, input1);
  input1 = _mm_packs_epi32(input2, input3);

  // Transpose
  input2 = _mm_unpacklo_epi16(input0, input1);
  input3 = _mm_unpackhi_epi16(input0, input1);
  input0 = _mm_unpacklo_epi32(input2, input3);
  input1 = _mm_unpackhi_epi32(input2, input3);

  // Switch column2, column 3, and then, we got:
  // input2: column1, column 0;  input3: column2, column 3.
  input1 = _mm_shuffle_epi32(input1, 0x4e);
  input2 = _mm_add_epi16(input0, input1);
  input3 = _mm_sub_epi16(input0, input1);

  // Columns
  // Construct i3, i1, i3, i1, i2, i0, i2, i0
  input0 = _mm_unpacklo_epi32(input2, input2);
  input1 = _mm_unpackhi_epi32(input2, input2);
  input2 = _mm_unpackhi_epi32(input3, input3);
  input3 = _mm_unpacklo_epi32(input3, input3);

  // Stage 1
  input0 = _mm_madd_epi16(input0, cst);
  input1 = _mm_madd_epi16(input1, cst);
  input2 = _mm_madd_epi16(input2, cst);
  input3 = _mm_madd_epi16(input3, cst);

  input0 = _mm_add_epi32(input0, rounding);
  input1 = _mm_add_epi32(input1, rounding);
  input2 = _mm_add_epi32(input2, rounding);
  input3 = _mm_add_epi32(input3, rounding);

  input0 = _mm_srai_epi32(input0, DCT_CONST_BITS);
  input1 = _mm_srai_epi32(input1, DCT_CONST_BITS);
  input2 = _mm_srai_epi32(input2, DCT_CONST_BITS);
  input3 = _mm_srai_epi32(input3, DCT_CONST_BITS);

  // Stage 2
  input0 = _mm_packs_epi32(input0, input2);
  input1 = _mm_packs_epi32(input1, input3);

  // Transpose
  input2 = _mm_unpacklo_epi16(input0, input1);
  input3 = _mm_unpackhi_epi16(input0, input1);
  input0 = _mm_unpacklo_epi32(input2, input3);
  input1 = _mm_unpackhi_epi32(input2, input3);

  // Switch column2, column 3, and then, we got:
  // input2: column1, column 0;  input3: column2, column 3.
  input1 = _mm_shuffle_epi32(input1, 0x4e);
  input2 = _mm_add_epi16(input0, input1);
  input3 = _mm_sub_epi16(input0, input1);

  // Final round and shift
  input2 = _mm_add_epi16(input2, eight);
  input3 = _mm_add_epi16(input3, eight);

  input2 = _mm_srai_epi16(input2, 4);
  input3 = _mm_srai_epi16(input3, 4);

  // Reconstruction and Store
  {
    __m128i d0 = _mm_cvtsi32_si128(*(const int *)(dest));
    __m128i d2 = _mm_cvtsi32_si128(*(const int *)(dest + stride * 2));
    d0 = _mm_unpacklo_epi32(d0,
                            _mm_cvtsi32_si128(*(const int *)(dest + stride)));
    d2 = _mm_unpacklo_epi32(
        _mm_cvtsi32_si128(*(const int *)(dest + stride * 3)), d2);
    d0 = _mm_unpacklo_epi8(d0, zero);
    d2 = _mm_unpacklo_epi8(d2, zero);
    d0 = _mm_add_epi16(d0, input2);
    d2 = _mm_add_epi16(d2, input3);
    d0 = _mm_packus_epi16(d0, d2);
    // store input0
    *(int *)dest = _mm_cvtsi128_si32(d0);
    // store input1
    d0 = _mm_srli_si128(d0, 4);
    *(int *)(dest + stride) = _mm_cvtsi128_si32(d0);
    // store input2
    d0 = _mm_srli_si128(d0, 4);
    *(int *)(dest + stride * 3) = _mm_cvtsi128_si32(d0);
    // store input3
    d0 = _mm_srli_si128(d0, 4);
    *(int *)(dest + stride * 2) = _mm_cvtsi128_si32(d0);
  }
}

void vpx_idct4x4_1_add_sse2(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  __m128i dc_value;
  const __m128i zero = _mm_setzero_si128();
  int a;

  a = (int)dct_const_round_shift(input[0] * cospi_16_64);
  a = (int)dct_const_round_shift(a * cospi_16_64);
  a = ROUND_POWER_OF_TWO(a, 4);

  dc_value = _mm_set1_epi16(a);

  RECON_AND_STORE4X4(dest + 0 * stride, dc_value);
  RECON_AND_STORE4X4(dest + 1 * stride, dc_value);
  RECON_AND_STORE4X4(dest + 2 * stride, dc_value);
  RECON_AND_STORE4X4(dest + 3 * stride, dc_value);
}

static INLINE void transpose_4x4(__m128i *res) {
  const __m128i tr0_0 = _mm_unpacklo_epi16(res[0], res[1]);
  const __m128i tr0_1 = _mm_unpackhi_epi16(res[0], res[1]);

  res[0] = _mm_unpacklo_epi16(tr0_0, tr0_1);
  res[1] = _mm_unpackhi_epi16(tr0_0, tr0_1);
}

void idct4_sse2(__m128i *in) {
  const __m128i k__cospi_p16_p16 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i k__cospi_p16_m16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_p24_m08 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i k__cospi_p08_p24 = pair_set_epi16(cospi_8_64, cospi_24_64);
  const __m128i k__DCT_CONST_ROUNDING = _mm_set1_epi32(DCT_CONST_ROUNDING);
  __m128i u[8], v[8];

  transpose_4x4(in);
  // stage 1
  u[0] = _mm_unpacklo_epi16(in[0], in[1]);
  u[1] = _mm_unpackhi_epi16(in[0], in[1]);
  v[0] = _mm_madd_epi16(u[0], k__cospi_p16_p16);
  v[1] = _mm_madd_epi16(u[0], k__cospi_p16_m16);
  v[2] = _mm_madd_epi16(u[1], k__cospi_p24_m08);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p08_p24);

  u[0] = _mm_add_epi32(v[0], k__DCT_CONST_ROUNDING);
  u[1] = _mm_add_epi32(v[1], k__DCT_CONST_ROUNDING);
  u[2] = _mm_add_epi32(v[2], k__DCT_CONST_ROUNDING);
  u[3] = _mm_add_epi32(v[3], k__DCT_CONST_ROUNDING);

  v[0] = _mm_srai_epi32(u[0], DCT_CONST_BITS);
  v[1] = _mm_srai_epi32(u[1], DCT_CONST_BITS);
  v[2] = _mm_srai_epi32(u[2], DCT_CONST_BITS);
  v[3] = _mm_srai_epi32(u[3], DCT_CONST_BITS);

  u[0] = _mm_packs_epi32(v[0], v[1]);
  u[1] = _mm_packs_epi32(v[3], v[2]);

  // stage 2
  in[0] = _mm_add_epi16(u[0], u[1]);
  in[1] = _mm_sub_epi16(u[0], u[1]);
  in[1] = _mm_shuffle_epi32(in[1], 0x4E);
}

void iadst4_sse2(__m128i *in) {
  const __m128i k__sinpi_p01_p04 = pair_set_epi16(sinpi_1_9, sinpi_4_9);
  const __m128i k__sinpi_p03_p02 = pair_set_epi16(sinpi_3_9, sinpi_2_9);
  const __m128i k__sinpi_p02_m01 = pair_set_epi16(sinpi_2_9, -sinpi_1_9);
  const __m128i k__sinpi_p03_m04 = pair_set_epi16(sinpi_3_9, -sinpi_4_9);
  const __m128i k__sinpi_p03_p03 = _mm_set1_epi16((int16_t)sinpi_3_9);
  const __m128i kZero = _mm_set1_epi16(0);
  const __m128i k__DCT_CONST_ROUNDING = _mm_set1_epi32(DCT_CONST_ROUNDING);
  __m128i u[8], v[8], in7;

  transpose_4x4(in);
  in7 = _mm_srli_si128(in[1], 8);
  in7 = _mm_add_epi16(in7, in[0]);
  in7 = _mm_sub_epi16(in7, in[1]);

  u[0] = _mm_unpacklo_epi16(in[0], in[1]);
  u[1] = _mm_unpackhi_epi16(in[0], in[1]);
  u[2] = _mm_unpacklo_epi16(in7, kZero);
  u[3] = _mm_unpackhi_epi16(in[0], kZero);

  v[0] = _mm_madd_epi16(u[0], k__sinpi_p01_p04);  // s0 + s3
  v[1] = _mm_madd_epi16(u[1], k__sinpi_p03_p02);  // s2 + s5
  v[2] = _mm_madd_epi16(u[2], k__sinpi_p03_p03);  // x2
  v[3] = _mm_madd_epi16(u[0], k__sinpi_p02_m01);  // s1 - s4
  v[4] = _mm_madd_epi16(u[1], k__sinpi_p03_m04);  // s2 - s6
  v[5] = _mm_madd_epi16(u[3], k__sinpi_p03_p03);  // s2

  u[0] = _mm_add_epi32(v[0], v[1]);
  u[1] = _mm_add_epi32(v[3], v[4]);
  u[2] = v[2];
  u[3] = _mm_add_epi32(u[0], u[1]);
  u[4] = _mm_slli_epi32(v[5], 2);
  u[5] = _mm_add_epi32(u[3], v[5]);
  u[6] = _mm_sub_epi32(u[5], u[4]);

  v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
  v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
  v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
  v[3] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);

  u[0] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
  u[1] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
  u[2] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
  u[3] = _mm_srai_epi32(v[3], DCT_CONST_BITS);

  in[0] = _mm_packs_epi32(u[0], u[1]);
  in[1] = _mm_packs_epi32(u[2], u[3]);
}

#define TRANSPOSE_8X8(in0, in1, in2, in3, in4, in5, in6, in7, \
                      out0, out1, out2, out3, out4, out5, out6, out7) \
  {                                                     \
    const __m128i tr0_0 = _mm_unpacklo_epi16(in0, in1); \
    const __m128i tr0_1 = _mm_unpacklo_epi16(in2, in3); \
    const __m128i tr0_2 = _mm_unpackhi_epi16(in0, in1); \
    const __m128i tr0_3 = _mm_unpackhi_epi16(in2, in3); \
    const __m128i tr0_4 = _mm_unpacklo_epi16(in4, in5); \
    const __m128i tr0_5 = _mm_unpacklo_epi16(in6, in7); \
    const __m128i tr0_6 = _mm_unpackhi_epi16(in4, in5); \
    const __m128i tr0_7 = _mm_unpackhi_epi16(in6, in7); \
                                                        \
    const __m128i tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    const __m128i tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3); \
    const __m128i tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
    const __m128i tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3); \
    const __m128i tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5); \
    const __m128i tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7); \
    const __m128i tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5); \
    const __m128i tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7); \
                                                            \
    out0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    out1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    out2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    out3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
    out4 = _mm_unpacklo_epi64(tr1_1, tr1_5); \
    out5 = _mm_unpackhi_epi64(tr1_1, tr1_5); \
    out6 = _mm_unpacklo_epi64(tr1_3, tr1_7); \
    out7 = _mm_unpackhi_epi64(tr1_3, tr1_7); \
  }

#define TRANSPOSE_4X8_10(tmp0, tmp1, tmp2, tmp3, \
                         out0, out1, out2, out3) \
  {                                              \
    const __m128i tr0_0 = _mm_unpackhi_epi16(tmp0, tmp1); \
    const __m128i tr0_1 = _mm_unpacklo_epi16(tmp1, tmp0); \
    const __m128i tr0_4 = _mm_unpacklo_epi16(tmp2, tmp3); \
    const __m128i tr0_5 = _mm_unpackhi_epi16(tmp3, tmp2); \
    \
    const __m128i tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    const __m128i tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
    const __m128i tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5); \
    const __m128i tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5); \
    \
    out0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    out1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    out2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    out3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
  }

#define TRANSPOSE_8X8_10(in0, in1, in2, in3, out0, out1) \
  {                                            \
    const __m128i tr0_0 = _mm_unpacklo_epi16(in0, in1); \
    const __m128i tr0_1 = _mm_unpacklo_epi16(in2, in3); \
    out0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    out1 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
  }

// Define Macro for multiplying elements by constants and adding them together.
#define MULTIPLICATION_AND_ADD(lo_0, hi_0, lo_1, hi_1, \
                               cst0, cst1, cst2, cst3, res0, res1, res2, res3) \
  {   \
      tmp0 = _mm_madd_epi16(lo_0, cst0); \
      tmp1 = _mm_madd_epi16(hi_0, cst0); \
      tmp2 = _mm_madd_epi16(lo_0, cst1); \
      tmp3 = _mm_madd_epi16(hi_0, cst1); \
      tmp4 = _mm_madd_epi16(lo_1, cst2); \
      tmp5 = _mm_madd_epi16(hi_1, cst2); \
      tmp6 = _mm_madd_epi16(lo_1, cst3); \
      tmp7 = _mm_madd_epi16(hi_1, cst3); \
      \
      tmp0 = _mm_add_epi32(tmp0, rounding); \
      tmp1 = _mm_add_epi32(tmp1, rounding); \
      tmp2 = _mm_add_epi32(tmp2, rounding); \
      tmp3 = _mm_add_epi32(tmp3, rounding); \
      tmp4 = _mm_add_epi32(tmp4, rounding); \
      tmp5 = _mm_add_epi32(tmp5, rounding); \
      tmp6 = _mm_add_epi32(tmp6, rounding); \
      tmp7 = _mm_add_epi32(tmp7, rounding); \
      \
      tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS); \
      tmp1 = _mm_srai_epi32(tmp1, DCT_CONST_BITS); \
      tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS); \
      tmp3 = _mm_srai_epi32(tmp3, DCT_CONST_BITS); \
      tmp4 = _mm_srai_epi32(tmp4, DCT_CONST_BITS); \
      tmp5 = _mm_srai_epi32(tmp5, DCT_CONST_BITS); \
      tmp6 = _mm_srai_epi32(tmp6, DCT_CONST_BITS); \
      tmp7 = _mm_srai_epi32(tmp7, DCT_CONST_BITS); \
      \
      res0 = _mm_packs_epi32(tmp0, tmp1); \
      res1 = _mm_packs_epi32(tmp2, tmp3); \
      res2 = _mm_packs_epi32(tmp4, tmp5); \
      res3 = _mm_packs_epi32(tmp6, tmp7); \
  }

#define MULTIPLICATION_AND_ADD_2(lo_0, hi_0, cst0, cst1, res0, res1) \
  {   \
      tmp0 = _mm_madd_epi16(lo_0, cst0); \
      tmp1 = _mm_madd_epi16(hi_0, cst0); \
      tmp2 = _mm_madd_epi16(lo_0, cst1); \
      tmp3 = _mm_madd_epi16(hi_0, cst1); \
      \
      tmp0 = _mm_add_epi32(tmp0, rounding); \
      tmp1 = _mm_add_epi32(tmp1, rounding); \
      tmp2 = _mm_add_epi32(tmp2, rounding); \
      tmp3 = _mm_add_epi32(tmp3, rounding); \
      \
      tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS); \
      tmp1 = _mm_srai_epi32(tmp1, DCT_CONST_BITS); \
      tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS); \
      tmp3 = _mm_srai_epi32(tmp3, DCT_CONST_BITS); \
      \
      res0 = _mm_packs_epi32(tmp0, tmp1); \
      res1 = _mm_packs_epi32(tmp2, tmp3); \
  }

#define IDCT8(in0, in1, in2, in3, in4, in5, in6, in7, \
              out0, out1, out2, out3, out4, out5, out6, out7)  \
  { \
  /* Stage1 */      \
  { \
    const __m128i lo_17 = _mm_unpacklo_epi16(in1, in7); \
    const __m128i hi_17 = _mm_unpackhi_epi16(in1, in7); \
    const __m128i lo_35 = _mm_unpacklo_epi16(in3, in5); \
    const __m128i hi_35 = _mm_unpackhi_epi16(in3, in5); \
    \
    MULTIPLICATION_AND_ADD(lo_17, hi_17, lo_35, hi_35, stg1_0, \
                          stg1_1, stg1_2, stg1_3, stp1_4,      \
                          stp1_7, stp1_5, stp1_6)              \
  } \
    \
  /* Stage2 */ \
  { \
    const __m128i lo_04 = _mm_unpacklo_epi16(in0, in4); \
    const __m128i hi_04 = _mm_unpackhi_epi16(in0, in4); \
    const __m128i lo_26 = _mm_unpacklo_epi16(in2, in6); \
    const __m128i hi_26 = _mm_unpackhi_epi16(in2, in6); \
    \
    MULTIPLICATION_AND_ADD(lo_04, hi_04, lo_26, hi_26, stg2_0, \
                           stg2_1, stg2_2, stg2_3, stp2_0,     \
                           stp2_1, stp2_2, stp2_3)             \
    \
    stp2_4 = _mm_adds_epi16(stp1_4, stp1_5); \
    stp2_5 = _mm_subs_epi16(stp1_4, stp1_5); \
    stp2_6 = _mm_subs_epi16(stp1_7, stp1_6); \
    stp2_7 = _mm_adds_epi16(stp1_7, stp1_6); \
  } \
    \
  /* Stage3 */ \
  { \
    const __m128i lo_56 = _mm_unpacklo_epi16(stp2_6, stp2_5); \
    const __m128i hi_56 = _mm_unpackhi_epi16(stp2_6, stp2_5); \
    \
    stp1_0 = _mm_adds_epi16(stp2_0, stp2_3); \
    stp1_1 = _mm_adds_epi16(stp2_1, stp2_2); \
    stp1_2 = _mm_subs_epi16(stp2_1, stp2_2); \
    stp1_3 = _mm_subs_epi16(stp2_0, stp2_3); \
    \
    tmp0 = _mm_madd_epi16(lo_56, stg2_1); \
    tmp1 = _mm_madd_epi16(hi_56, stg2_1); \
    tmp2 = _mm_madd_epi16(lo_56, stg2_0); \
    tmp3 = _mm_madd_epi16(hi_56, stg2_0); \
    \
    tmp0 = _mm_add_epi32(tmp0, rounding); \
    tmp1 = _mm_add_epi32(tmp1, rounding); \
    tmp2 = _mm_add_epi32(tmp2, rounding); \
    tmp3 = _mm_add_epi32(tmp3, rounding); \
    \
    tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS); \
    tmp1 = _mm_srai_epi32(tmp1, DCT_CONST_BITS); \
    tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS); \
    tmp3 = _mm_srai_epi32(tmp3, DCT_CONST_BITS); \
    \
    stp1_5 = _mm_packs_epi32(tmp0, tmp1); \
    stp1_6 = _mm_packs_epi32(tmp2, tmp3); \
  } \
  \
  /* Stage4  */ \
  out0 = _mm_adds_epi16(stp1_0, stp2_7); \
  out1 = _mm_adds_epi16(stp1_1, stp1_6); \
  out2 = _mm_adds_epi16(stp1_2, stp1_5); \
  out3 = _mm_adds_epi16(stp1_3, stp2_4); \
  out4 = _mm_subs_epi16(stp1_3, stp2_4); \
  out5 = _mm_subs_epi16(stp1_2, stp1_5); \
  out6 = _mm_subs_epi16(stp1_1, stp1_6); \
  out7 = _mm_subs_epi16(stp1_0, stp2_7); \
  }

void vpx_idct8x8_64_add_sse2(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i rounding = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i final_rounding = _mm_set1_epi16(1 << 4);
  const __m128i stg1_0 = pair_set_epi16(cospi_28_64, -cospi_4_64);
  const __m128i stg1_1 = pair_set_epi16(cospi_4_64, cospi_28_64);
  const __m128i stg1_2 = pair_set_epi16(-cospi_20_64, cospi_12_64);
  const __m128i stg1_3 = pair_set_epi16(cospi_12_64, cospi_20_64);
  const __m128i stg2_0 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i stg2_1 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i stg2_2 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i stg2_3 = pair_set_epi16(cospi_8_64, cospi_24_64);

  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i stp1_0, stp1_1, stp1_2, stp1_3, stp1_4, stp1_5, stp1_6, stp1_7;
  __m128i stp2_0, stp2_1, stp2_2, stp2_3, stp2_4, stp2_5, stp2_6, stp2_7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  int i;

  // Load input data.
  in0 = load_input_data(input);
  in1 = load_input_data(input + 8 * 1);
  in2 = load_input_data(input + 8 * 2);
  in3 = load_input_data(input + 8 * 3);
  in4 = load_input_data(input + 8 * 4);
  in5 = load_input_data(input + 8 * 5);
  in6 = load_input_data(input + 8 * 6);
  in7 = load_input_data(input + 8 * 7);

  // 2-D
  for (i = 0; i < 2; i++) {
    // 8x8 Transpose is copied from vpx_fdct8x8_sse2()
    TRANSPOSE_8X8(in0, in1, in2, in3, in4, in5, in6, in7,
                  in0, in1, in2, in3, in4, in5, in6, in7);

    // 4-stage 1D idct8x8
    IDCT8(in0, in1, in2, in3, in4, in5, in6, in7,
          in0, in1, in2, in3, in4, in5, in6, in7);
  }

  // Final rounding and shift
  in0 = _mm_adds_epi16(in0, final_rounding);
  in1 = _mm_adds_epi16(in1, final_rounding);
  in2 = _mm_adds_epi16(in2, final_rounding);
  in3 = _mm_adds_epi16(in3, final_rounding);
  in4 = _mm_adds_epi16(in4, final_rounding);
  in5 = _mm_adds_epi16(in5, final_rounding);
  in6 = _mm_adds_epi16(in6, final_rounding);
  in7 = _mm_adds_epi16(in7, final_rounding);

  in0 = _mm_srai_epi16(in0, 5);
  in1 = _mm_srai_epi16(in1, 5);
  in2 = _mm_srai_epi16(in2, 5);
  in3 = _mm_srai_epi16(in3, 5);
  in4 = _mm_srai_epi16(in4, 5);
  in5 = _mm_srai_epi16(in5, 5);
  in6 = _mm_srai_epi16(in6, 5);
  in7 = _mm_srai_epi16(in7, 5);

  RECON_AND_STORE(dest + 0 * stride, in0);
  RECON_AND_STORE(dest + 1 * stride, in1);
  RECON_AND_STORE(dest + 2 * stride, in2);
  RECON_AND_STORE(dest + 3 * stride, in3);
  RECON_AND_STORE(dest + 4 * stride, in4);
  RECON_AND_STORE(dest + 5 * stride, in5);
  RECON_AND_STORE(dest + 6 * stride, in6);
  RECON_AND_STORE(dest + 7 * stride, in7);
}

void vpx_idct8x8_1_add_sse2(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  __m128i dc_value;
  const __m128i zero = _mm_setzero_si128();
  int a;

  a = (int)dct_const_round_shift(input[0] * cospi_16_64);
  a = (int)dct_const_round_shift(a * cospi_16_64);
  a = ROUND_POWER_OF_TWO(a, 5);

  dc_value = _mm_set1_epi16(a);

  RECON_AND_STORE(dest + 0 * stride, dc_value);
  RECON_AND_STORE(dest + 1 * stride, dc_value);
  RECON_AND_STORE(dest + 2 * stride, dc_value);
  RECON_AND_STORE(dest + 3 * stride, dc_value);
  RECON_AND_STORE(dest + 4 * stride, dc_value);
  RECON_AND_STORE(dest + 5 * stride, dc_value);
  RECON_AND_STORE(dest + 6 * stride, dc_value);
  RECON_AND_STORE(dest + 7 * stride, dc_value);
}

void idct8_sse2(__m128i *in) {
  const __m128i rounding = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i stg1_0 = pair_set_epi16(cospi_28_64, -cospi_4_64);
  const __m128i stg1_1 = pair_set_epi16(cospi_4_64, cospi_28_64);
  const __m128i stg1_2 = pair_set_epi16(-cospi_20_64, cospi_12_64);
  const __m128i stg1_3 = pair_set_epi16(cospi_12_64, cospi_20_64);
  const __m128i stg2_0 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i stg2_1 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i stg2_2 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i stg2_3 = pair_set_epi16(cospi_8_64, cospi_24_64);

  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i stp1_0, stp1_1, stp1_2, stp1_3, stp1_4, stp1_5, stp1_6, stp1_7;
  __m128i stp2_0, stp2_1, stp2_2, stp2_3, stp2_4, stp2_5, stp2_6, stp2_7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

  // 8x8 Transpose is copied from vpx_fdct8x8_sse2()
  TRANSPOSE_8X8(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7],
                in0, in1, in2, in3, in4, in5, in6, in7);

  // 4-stage 1D idct8x8
  IDCT8(in0, in1, in2, in3, in4, in5, in6, in7,
        in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]);
}

void iadst8_sse2(__m128i *in) {
  const __m128i k__cospi_p02_p30 = pair_set_epi16(cospi_2_64, cospi_30_64);
  const __m128i k__cospi_p30_m02 = pair_set_epi16(cospi_30_64, -cospi_2_64);
  const __m128i k__cospi_p10_p22 = pair_set_epi16(cospi_10_64, cospi_22_64);
  const __m128i k__cospi_p22_m10 = pair_set_epi16(cospi_22_64, -cospi_10_64);
  const __m128i k__cospi_p18_p14 = pair_set_epi16(cospi_18_64, cospi_14_64);
  const __m128i k__cospi_p14_m18 = pair_set_epi16(cospi_14_64, -cospi_18_64);
  const __m128i k__cospi_p26_p06 = pair_set_epi16(cospi_26_64, cospi_6_64);
  const __m128i k__cospi_p06_m26 = pair_set_epi16(cospi_6_64, -cospi_26_64);
  const __m128i k__cospi_p08_p24 = pair_set_epi16(cospi_8_64, cospi_24_64);
  const __m128i k__cospi_p24_m08 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i k__cospi_m24_p08 = pair_set_epi16(-cospi_24_64, cospi_8_64);
  const __m128i k__cospi_p16_m16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_p16_p16 = _mm_set1_epi16((int16_t)cospi_16_64);
  const __m128i k__const_0 = _mm_set1_epi16(0);
  const __m128i k__DCT_CONST_ROUNDING = _mm_set1_epi32(DCT_CONST_ROUNDING);

  __m128i u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15;
  __m128i v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;
  __m128i w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15;
  __m128i s0, s1, s2, s3, s4, s5, s6, s7;
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;

  // transpose
  array_transpose_8x8(in, in);

  // properly aligned for butterfly input
  in0 = in[7];
  in1 = in[0];
  in2 = in[5];
  in3 = in[2];
  in4 = in[3];
  in5 = in[4];
  in6 = in[1];
  in7 = in[6];

  // column transformation
  // stage 1
  // interleave and multiply/add into 32-bit integer
  s0 = _mm_unpacklo_epi16(in0, in1);
  s1 = _mm_unpackhi_epi16(in0, in1);
  s2 = _mm_unpacklo_epi16(in2, in3);
  s3 = _mm_unpackhi_epi16(in2, in3);
  s4 = _mm_unpacklo_epi16(in4, in5);
  s5 = _mm_unpackhi_epi16(in4, in5);
  s6 = _mm_unpacklo_epi16(in6, in7);
  s7 = _mm_unpackhi_epi16(in6, in7);

  u0 = _mm_madd_epi16(s0, k__cospi_p02_p30);
  u1 = _mm_madd_epi16(s1, k__cospi_p02_p30);
  u2 = _mm_madd_epi16(s0, k__cospi_p30_m02);
  u3 = _mm_madd_epi16(s1, k__cospi_p30_m02);
  u4 = _mm_madd_epi16(s2, k__cospi_p10_p22);
  u5 = _mm_madd_epi16(s3, k__cospi_p10_p22);
  u6 = _mm_madd_epi16(s2, k__cospi_p22_m10);
  u7 = _mm_madd_epi16(s3, k__cospi_p22_m10);
  u8 = _mm_madd_epi16(s4, k__cospi_p18_p14);
  u9 = _mm_madd_epi16(s5, k__cospi_p18_p14);
  u10 = _mm_madd_epi16(s4, k__cospi_p14_m18);
  u11 = _mm_madd_epi16(s5, k__cospi_p14_m18);
  u12 = _mm_madd_epi16(s6, k__cospi_p26_p06);
  u13 = _mm_madd_epi16(s7, k__cospi_p26_p06);
  u14 = _mm_madd_epi16(s6, k__cospi_p06_m26);
  u15 = _mm_madd_epi16(s7, k__cospi_p06_m26);

  // addition
  w0 = _mm_add_epi32(u0, u8);
  w1 = _mm_add_epi32(u1, u9);
  w2 = _mm_add_epi32(u2, u10);
  w3 = _mm_add_epi32(u3, u11);
  w4 = _mm_add_epi32(u4, u12);
  w5 = _mm_add_epi32(u5, u13);
  w6 = _mm_add_epi32(u6, u14);
  w7 = _mm_add_epi32(u7, u15);
  w8 = _mm_sub_epi32(u0, u8);
  w9 = _mm_sub_epi32(u1, u9);
  w10 = _mm_sub_epi32(u2, u10);
  w11 = _mm_sub_epi32(u3, u11);
  w12 = _mm_sub_epi32(u4, u12);
  w13 = _mm_sub_epi32(u5, u13);
  w14 = _mm_sub_epi32(u6, u14);
  w15 = _mm_sub_epi32(u7, u15);

  // shift and rounding
  v0 = _mm_add_epi32(w0, k__DCT_CONST_ROUNDING);
  v1 = _mm_add_epi32(w1, k__DCT_CONST_ROUNDING);
  v2 = _mm_add_epi32(w2, k__DCT_CONST_ROUNDING);
  v3 = _mm_add_epi32(w3, k__DCT_CONST_ROUNDING);
  v4 = _mm_add_epi32(w4, k__DCT_CONST_ROUNDING);
  v5 = _mm_add_epi32(w5, k__DCT_CONST_ROUNDING);
  v6 = _mm_add_epi32(w6, k__DCT_CONST_ROUNDING);
  v7 = _mm_add_epi32(w7, k__DCT_CONST_ROUNDING);
  v8 = _mm_add_epi32(w8, k__DCT_CONST_ROUNDING);
  v9 = _mm_add_epi32(w9, k__DCT_CONST_ROUNDING);
  v10 = _mm_add_epi32(w10, k__DCT_CONST_ROUNDING);
  v11 = _mm_add_epi32(w11, k__DCT_CONST_ROUNDING);
  v12 = _mm_add_epi32(w12, k__DCT_CONST_ROUNDING);
  v13 = _mm_add_epi32(w13, k__DCT_CONST_ROUNDING);
  v14 = _mm_add_epi32(w14, k__DCT_CONST_ROUNDING);
  v15 = _mm_add_epi32(w15, k__DCT_CONST_ROUNDING);

  u0 = _mm_srai_epi32(v0, DCT_CONST_BITS);
  u1 = _mm_srai_epi32(v1, DCT_CONST_BITS);
  u2 = _mm_srai_epi32(v2, DCT_CONST_BITS);
  u3 = _mm_srai_epi32(v3, DCT_CONST_BITS);
  u4 = _mm_srai_epi32(v4, DCT_CONST_BITS);
  u5 = _mm_srai_epi32(v5, DCT_CONST_BITS);
  u6 = _mm_srai_epi32(v6, DCT_CONST_BITS);
  u7 = _mm_srai_epi32(v7, DCT_CONST_BITS);
  u8 = _mm_srai_epi32(v8, DCT_CONST_BITS);
  u9 = _mm_srai_epi32(v9, DCT_CONST_BITS);
  u10 = _mm_srai_epi32(v10, DCT_CONST_BITS);
  u11 = _mm_srai_epi32(v11, DCT_CONST_BITS);
  u12 = _mm_srai_epi32(v12, DCT_CONST_BITS);
  u13 = _mm_srai_epi32(v13, DCT_CONST_BITS);
  u14 = _mm_srai_epi32(v14, DCT_CONST_BITS);
  u15 = _mm_srai_epi32(v15, DCT_CONST_BITS);

  // back to 16-bit and pack 8 integers into __m128i
  in[0] = _mm_packs_epi32(u0, u1);
  in[1] = _mm_packs_epi32(u2, u3);
  in[2] = _mm_packs_epi32(u4, u5);
  in[3] = _mm_packs_epi32(u6, u7);
  in[4] = _mm_packs_epi32(u8, u9);
  in[5] = _mm_packs_epi32(u10, u11);
  in[6] = _mm_packs_epi32(u12, u13);
  in[7] = _mm_packs_epi32(u14, u15);

  // stage 2
  s0 = _mm_add_epi16(in[0], in[2]);
  s1 = _mm_add_epi16(in[1], in[3]);
  s2 = _mm_sub_epi16(in[0], in[2]);
  s3 = _mm_sub_epi16(in[1], in[3]);
  u0 = _mm_unpacklo_epi16(in[4], in[5]);
  u1 = _mm_unpackhi_epi16(in[4], in[5]);
  u2 = _mm_unpacklo_epi16(in[6], in[7]);
  u3 = _mm_unpackhi_epi16(in[6], in[7]);

  v0 = _mm_madd_epi16(u0, k__cospi_p08_p24);
  v1 = _mm_madd_epi16(u1, k__cospi_p08_p24);
  v2 = _mm_madd_epi16(u0, k__cospi_p24_m08);
  v3 = _mm_madd_epi16(u1, k__cospi_p24_m08);
  v4 = _mm_madd_epi16(u2, k__cospi_m24_p08);
  v5 = _mm_madd_epi16(u3, k__cospi_m24_p08);
  v6 = _mm_madd_epi16(u2, k__cospi_p08_p24);
  v7 = _mm_madd_epi16(u3, k__cospi_p08_p24);

  w0 = _mm_add_epi32(v0, v4);
  w1 = _mm_add_epi32(v1, v5);
  w2 = _mm_add_epi32(v2, v6);
  w3 = _mm_add_epi32(v3, v7);
  w4 = _mm_sub_epi32(v0, v4);
  w5 = _mm_sub_epi32(v1, v5);
  w6 = _mm_sub_epi32(v2, v6);
  w7 = _mm_sub_epi32(v3, v7);

  v0 = _mm_add_epi32(w0, k__DCT_CONST_ROUNDING);
  v1 = _mm_add_epi32(w1, k__DCT_CONST_ROUNDING);
  v2 = _mm_add_epi32(w2, k__DCT_CONST_ROUNDING);
  v3 = _mm_add_epi32(w3, k__DCT_CONST_ROUNDING);
  v4 = _mm_add_epi32(w4, k__DCT_CONST_ROUNDING);
  v5 = _mm_add_epi32(w5, k__DCT_CONST_ROUNDING);
  v6 = _mm_add_epi32(w6, k__DCT_CONST_ROUNDING);
  v7 = _mm_add_epi32(w7, k__DCT_CONST_ROUNDING);

  u0 = _mm_srai_epi32(v0, DCT_CONST_BITS);
  u1 = _mm_srai_epi32(v1, DCT_CONST_BITS);
  u2 = _mm_srai_epi32(v2, DCT_CONST_BITS);
  u3 = _mm_srai_epi32(v3, DCT_CONST_BITS);
  u4 = _mm_srai_epi32(v4, DCT_CONST_BITS);
  u5 = _mm_srai_epi32(v5, DCT_CONST_BITS);
  u6 = _mm_srai_epi32(v6, DCT_CONST_BITS);
  u7 = _mm_srai_epi32(v7, DCT_CONST_BITS);

  // back to 16-bit intergers
  s4 = _mm_packs_epi32(u0, u1);
  s5 = _mm_packs_epi32(u2, u3);
  s6 = _mm_packs_epi32(u4, u5);
  s7 = _mm_packs_epi32(u6, u7);

  // stage 3
  u0 = _mm_unpacklo_epi16(s2, s3);
  u1 = _mm_unpackhi_epi16(s2, s3);
  u2 = _mm_unpacklo_epi16(s6, s7);
  u3 = _mm_unpackhi_epi16(s6, s7);

  v0 = _mm_madd_epi16(u0, k__cospi_p16_p16);
  v1 = _mm_madd_epi16(u1, k__cospi_p16_p16);
  v2 = _mm_madd_epi16(u0, k__cospi_p16_m16);
  v3 = _mm_madd_epi16(u1, k__cospi_p16_m16);
  v4 = _mm_madd_epi16(u2, k__cospi_p16_p16);
  v5 = _mm_madd_epi16(u3, k__cospi_p16_p16);
  v6 = _mm_madd_epi16(u2, k__cospi_p16_m16);
  v7 = _mm_madd_epi16(u3, k__cospi_p16_m16);

  u0 = _mm_add_epi32(v0, k__DCT_CONST_ROUNDING);
  u1 = _mm_add_epi32(v1, k__DCT_CONST_ROUNDING);
  u2 = _mm_add_epi32(v2, k__DCT_CONST_ROUNDING);
  u3 = _mm_add_epi32(v3, k__DCT_CONST_ROUNDING);
  u4 = _mm_add_epi32(v4, k__DCT_CONST_ROUNDING);
  u5 = _mm_add_epi32(v5, k__DCT_CONST_ROUNDING);
  u6 = _mm_add_epi32(v6, k__DCT_CONST_ROUNDING);
  u7 = _mm_add_epi32(v7, k__DCT_CONST_ROUNDING);

  v0 = _mm_srai_epi32(u0, DCT_CONST_BITS);
  v1 = _mm_srai_epi32(u1, DCT_CONST_BITS);
  v2 = _mm_srai_epi32(u2, DCT_CONST_BITS);
  v3 = _mm_srai_epi32(u3, DCT_CONST_BITS);
  v4 = _mm_srai_epi32(u4, DCT_CONST_BITS);
  v5 = _mm_srai_epi32(u5, DCT_CONST_BITS);
  v6 = _mm_srai_epi32(u6, DCT_CONST_BITS);
  v7 = _mm_srai_epi32(u7, DCT_CONST_BITS);

  s2 = _mm_packs_epi32(v0, v1);
  s3 = _mm_packs_epi32(v2, v3);
  s6 = _mm_packs_epi32(v4, v5);
  s7 = _mm_packs_epi32(v6, v7);

  in[0] = s0;
  in[1] = _mm_sub_epi16(k__const_0, s4);
  in[2] = s6;
  in[3] = _mm_sub_epi16(k__const_0, s2);
  in[4] = s3;
  in[5] = _mm_sub_epi16(k__const_0, s7);
  in[6] = s5;
  in[7] = _mm_sub_epi16(k__const_0, s1);
}

void vpx_idct8x8_12_add_sse2(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i rounding = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i final_rounding = _mm_set1_epi16(1 << 4);
  const __m128i stg1_0 = pair_set_epi16(cospi_28_64, -cospi_4_64);
  const __m128i stg1_1 = pair_set_epi16(cospi_4_64, cospi_28_64);
  const __m128i stg1_2 = pair_set_epi16(-cospi_20_64, cospi_12_64);
  const __m128i stg1_3 = pair_set_epi16(cospi_12_64, cospi_20_64);
  const __m128i stg2_0 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i stg2_1 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i stg2_2 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i stg2_3 = pair_set_epi16(cospi_8_64, cospi_24_64);
  const __m128i stg3_0 = pair_set_epi16(-cospi_16_64, cospi_16_64);

  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i stp1_0, stp1_1, stp1_2, stp1_3, stp1_4, stp1_5, stp1_6, stp1_7;
  __m128i stp2_0, stp2_1, stp2_2, stp2_3, stp2_4, stp2_5, stp2_6, stp2_7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

  // Rows. Load 4-row input data.
  in0 = load_input_data(input);
  in1 = load_input_data(input + 8 * 1);
  in2 = load_input_data(input + 8 * 2);
  in3 = load_input_data(input + 8 * 3);

  // 8x4 Transpose
  TRANSPOSE_8X8_10(in0, in1, in2, in3, in0, in1);
  // Stage1
  {
    const __m128i lo_17 = _mm_unpackhi_epi16(in0, zero);
    const __m128i lo_35 = _mm_unpackhi_epi16(in1, zero);

    tmp0 = _mm_madd_epi16(lo_17, stg1_0);
    tmp2 = _mm_madd_epi16(lo_17, stg1_1);
    tmp4 = _mm_madd_epi16(lo_35, stg1_2);
    tmp6 = _mm_madd_epi16(lo_35, stg1_3);

    tmp0 = _mm_add_epi32(tmp0, rounding);
    tmp2 = _mm_add_epi32(tmp2, rounding);
    tmp4 = _mm_add_epi32(tmp4, rounding);
    tmp6 = _mm_add_epi32(tmp6, rounding);
    tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS);
    tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS);
    tmp4 = _mm_srai_epi32(tmp4, DCT_CONST_BITS);
    tmp6 = _mm_srai_epi32(tmp6, DCT_CONST_BITS);

    stp1_4 = _mm_packs_epi32(tmp0, tmp2);
    stp1_5 = _mm_packs_epi32(tmp4, tmp6);
  }

  // Stage2
  {
    const __m128i lo_04 = _mm_unpacklo_epi16(in0, zero);
    const __m128i lo_26 = _mm_unpacklo_epi16(in1, zero);

    tmp0 = _mm_madd_epi16(lo_04, stg2_0);
    tmp2 = _mm_madd_epi16(lo_04, stg2_1);
    tmp4 = _mm_madd_epi16(lo_26, stg2_2);
    tmp6 = _mm_madd_epi16(lo_26, stg2_3);

    tmp0 = _mm_add_epi32(tmp0, rounding);
    tmp2 = _mm_add_epi32(tmp2, rounding);
    tmp4 = _mm_add_epi32(tmp4, rounding);
    tmp6 = _mm_add_epi32(tmp6, rounding);
    tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS);
    tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS);
    tmp4 = _mm_srai_epi32(tmp4, DCT_CONST_BITS);
    tmp6 = _mm_srai_epi32(tmp6, DCT_CONST_BITS);

    stp2_0 = _mm_packs_epi32(tmp0, tmp2);
    stp2_2 = _mm_packs_epi32(tmp6, tmp4);

    tmp0 = _mm_adds_epi16(stp1_4, stp1_5);
    tmp1 = _mm_subs_epi16(stp1_4, stp1_5);

    stp2_4 = tmp0;
    stp2_5 = _mm_unpacklo_epi64(tmp1, zero);
    stp2_6 = _mm_unpackhi_epi64(tmp1, zero);
  }

  // Stage3
  {
    const __m128i lo_56 = _mm_unpacklo_epi16(stp2_5, stp2_6);

    tmp4 = _mm_adds_epi16(stp2_0, stp2_2);
    tmp6 = _mm_subs_epi16(stp2_0, stp2_2);

    stp1_2 = _mm_unpackhi_epi64(tmp6, tmp4);
    stp1_3 = _mm_unpacklo_epi64(tmp6, tmp4);

    tmp0 = _mm_madd_epi16(lo_56, stg3_0);
    tmp2 = _mm_madd_epi16(lo_56, stg2_0);  // stg3_1 = stg2_0

    tmp0 = _mm_add_epi32(tmp0, rounding);
    tmp2 = _mm_add_epi32(tmp2, rounding);
    tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS);
    tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS);

    stp1_5 = _mm_packs_epi32(tmp0, tmp2);
  }

  // Stage4
  tmp0 = _mm_adds_epi16(stp1_3, stp2_4);
  tmp1 = _mm_adds_epi16(stp1_2, stp1_5);
  tmp2 = _mm_subs_epi16(stp1_3, stp2_4);
  tmp3 = _mm_subs_epi16(stp1_2, stp1_5);

  TRANSPOSE_4X8_10(tmp0, tmp1, tmp2, tmp3, in0, in1, in2, in3)

  IDCT8(in0, in1, in2, in3, zero, zero, zero, zero,
        in0, in1, in2, in3, in4, in5, in6, in7);
  // Final rounding and shift
  in0 = _mm_adds_epi16(in0, final_rounding);
  in1 = _mm_adds_epi16(in1, final_rounding);
  in2 = _mm_adds_epi16(in2, final_rounding);
  in3 = _mm_adds_epi16(in3, final_rounding);
  in4 = _mm_adds_epi16(in4, final_rounding);
  in5 = _mm_adds_epi16(in5, final_rounding);
  in6 = _mm_adds_epi16(in6, final_rounding);
  in7 = _mm_adds_epi16(in7, final_rounding);

  in0 = _mm_srai_epi16(in0, 5);
  in1 = _mm_srai_epi16(in1, 5);
  in2 = _mm_srai_epi16(in2, 5);
  in3 = _mm_srai_epi16(in3, 5);
  in4 = _mm_srai_epi16(in4, 5);
  in5 = _mm_srai_epi16(in5, 5);
  in6 = _mm_srai_epi16(in6, 5);
  in7 = _mm_srai_epi16(in7, 5);

  RECON_AND_STORE(dest + 0 * stride, in0);
  RECON_AND_STORE(dest + 1 * stride, in1);
  RECON_AND_STORE(dest + 2 * stride, in2);
  RECON_AND_STORE(dest + 3 * stride, in3);
  RECON_AND_STORE(dest + 4 * stride, in4);
  RECON_AND_STORE(dest + 5 * stride, in5);
  RECON_AND_STORE(dest + 6 * stride, in6);
  RECON_AND_STORE(dest + 7 * stride, in7);
}

#define IDCT16 \
  /* Stage2 */ \
  { \
    const __m128i lo_1_15 = _mm_unpacklo_epi16(in[1], in[15]); \
    const __m128i hi_1_15 = _mm_unpackhi_epi16(in[1], in[15]); \
    const __m128i lo_9_7 = _mm_unpacklo_epi16(in[9], in[7]);   \
    const __m128i hi_9_7 = _mm_unpackhi_epi16(in[9], in[7]);   \
    const __m128i lo_5_11 = _mm_unpacklo_epi16(in[5], in[11]); \
    const __m128i hi_5_11 = _mm_unpackhi_epi16(in[5], in[11]); \
    const __m128i lo_13_3 = _mm_unpacklo_epi16(in[13], in[3]); \
    const __m128i hi_13_3 = _mm_unpackhi_epi16(in[13], in[3]); \
    \
    MULTIPLICATION_AND_ADD(lo_1_15, hi_1_15, lo_9_7, hi_9_7, \
                           stg2_0, stg2_1, stg2_2, stg2_3, \
                           stp2_8, stp2_15, stp2_9, stp2_14) \
    \
    MULTIPLICATION_AND_ADD(lo_5_11, hi_5_11, lo_13_3, hi_13_3, \
                           stg2_4, stg2_5, stg2_6, stg2_7, \
                           stp2_10, stp2_13, stp2_11, stp2_12) \
  } \
    \
  /* Stage3 */ \
  { \
    const __m128i lo_2_14 = _mm_unpacklo_epi16(in[2], in[14]); \
    const __m128i hi_2_14 = _mm_unpackhi_epi16(in[2], in[14]); \
    const __m128i lo_10_6 = _mm_unpacklo_epi16(in[10], in[6]); \
    const __m128i hi_10_6 = _mm_unpackhi_epi16(in[10], in[6]); \
    \
    MULTIPLICATION_AND_ADD(lo_2_14, hi_2_14, lo_10_6, hi_10_6, \
                           stg3_0, stg3_1, stg3_2, stg3_3, \
                           stp1_4, stp1_7, stp1_5, stp1_6) \
    \
    stp1_8_0 = _mm_add_epi16(stp2_8, stp2_9);  \
    stp1_9 = _mm_sub_epi16(stp2_8, stp2_9);    \
    stp1_10 = _mm_sub_epi16(stp2_11, stp2_10); \
    stp1_11 = _mm_add_epi16(stp2_11, stp2_10); \
    \
    stp1_12_0 = _mm_add_epi16(stp2_12, stp2_13); \
    stp1_13 = _mm_sub_epi16(stp2_12, stp2_13); \
    stp1_14 = _mm_sub_epi16(stp2_15, stp2_14); \
    stp1_15 = _mm_add_epi16(stp2_15, stp2_14); \
  } \
  \
  /* Stage4 */ \
  { \
    const __m128i lo_0_8 = _mm_unpacklo_epi16(in[0], in[8]); \
    const __m128i hi_0_8 = _mm_unpackhi_epi16(in[0], in[8]); \
    const __m128i lo_4_12 = _mm_unpacklo_epi16(in[4], in[12]); \
    const __m128i hi_4_12 = _mm_unpackhi_epi16(in[4], in[12]); \
    \
    const __m128i lo_9_14 = _mm_unpacklo_epi16(stp1_9, stp1_14); \
    const __m128i hi_9_14 = _mm_unpackhi_epi16(stp1_9, stp1_14); \
    const __m128i lo_10_13 = _mm_unpacklo_epi16(stp1_10, stp1_13); \
    const __m128i hi_10_13 = _mm_unpackhi_epi16(stp1_10, stp1_13); \
    \
    MULTIPLICATION_AND_ADD(lo_0_8, hi_0_8, lo_4_12, hi_4_12, \
                           stg4_0, stg4_1, stg4_2, stg4_3, \
                           stp2_0, stp2_1, stp2_2, stp2_3) \
    \
    stp2_4 = _mm_add_epi16(stp1_4, stp1_5); \
    stp2_5 = _mm_sub_epi16(stp1_4, stp1_5); \
    stp2_6 = _mm_sub_epi16(stp1_7, stp1_6); \
    stp2_7 = _mm_add_epi16(stp1_7, stp1_6); \
    \
    MULTIPLICATION_AND_ADD(lo_9_14, hi_9_14, lo_10_13, hi_10_13, \
                           stg4_4, stg4_5, stg4_6, stg4_7, \
                           stp2_9, stp2_14, stp2_10, stp2_13) \
  } \
    \
  /* Stage5 */ \
  { \
    const __m128i lo_6_5 = _mm_unpacklo_epi16(stp2_6, stp2_5); \
    const __m128i hi_6_5 = _mm_unpackhi_epi16(stp2_6, stp2_5); \
    \
    stp1_0 = _mm_add_epi16(stp2_0, stp2_3); \
    stp1_1 = _mm_add_epi16(stp2_1, stp2_2); \
    stp1_2 = _mm_sub_epi16(stp2_1, stp2_2); \
    stp1_3 = _mm_sub_epi16(stp2_0, stp2_3); \
    \
    tmp0 = _mm_madd_epi16(lo_6_5, stg4_1); \
    tmp1 = _mm_madd_epi16(hi_6_5, stg4_1); \
    tmp2 = _mm_madd_epi16(lo_6_5, stg4_0); \
    tmp3 = _mm_madd_epi16(hi_6_5, stg4_0); \
    \
    tmp0 = _mm_add_epi32(tmp0, rounding); \
    tmp1 = _mm_add_epi32(tmp1, rounding); \
    tmp2 = _mm_add_epi32(tmp2, rounding); \
    tmp3 = _mm_add_epi32(tmp3, rounding); \
    \
    tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS); \
    tmp1 = _mm_srai_epi32(tmp1, DCT_CONST_BITS); \
    tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS); \
    tmp3 = _mm_srai_epi32(tmp3, DCT_CONST_BITS); \
    \
    stp1_5 = _mm_packs_epi32(tmp0, tmp1); \
    stp1_6 = _mm_packs_epi32(tmp2, tmp3); \
    \
    stp1_8 = _mm_add_epi16(stp1_8_0, stp1_11);  \
    stp1_9 = _mm_add_epi16(stp2_9, stp2_10);    \
    stp1_10 = _mm_sub_epi16(stp2_9, stp2_10);   \
    stp1_11 = _mm_sub_epi16(stp1_8_0, stp1_11); \
    \
    stp1_12 = _mm_sub_epi16(stp1_15, stp1_12_0); \
    stp1_13 = _mm_sub_epi16(stp2_14, stp2_13);   \
    stp1_14 = _mm_add_epi16(stp2_14, stp2_13);   \
    stp1_15 = _mm_add_epi16(stp1_15, stp1_12_0); \
  } \
    \
  /* Stage6 */ \
  { \
    const __m128i lo_10_13 = _mm_unpacklo_epi16(stp1_10, stp1_13); \
    const __m128i hi_10_13 = _mm_unpackhi_epi16(stp1_10, stp1_13); \
    const __m128i lo_11_12 = _mm_unpacklo_epi16(stp1_11, stp1_12); \
    const __m128i hi_11_12 = _mm_unpackhi_epi16(stp1_11, stp1_12); \
    \
    stp2_0 = _mm_add_epi16(stp1_0, stp2_7); \
    stp2_1 = _mm_add_epi16(stp1_1, stp1_6); \
    stp2_2 = _mm_add_epi16(stp1_2, stp1_5); \
    stp2_3 = _mm_add_epi16(stp1_3, stp2_4); \
    stp2_4 = _mm_sub_epi16(stp1_3, stp2_4); \
    stp2_5 = _mm_sub_epi16(stp1_2, stp1_5); \
    stp2_6 = _mm_sub_epi16(stp1_1, stp1_6); \
    stp2_7 = _mm_sub_epi16(stp1_0, stp2_7); \
    \
    MULTIPLICATION_AND_ADD(lo_10_13, hi_10_13, lo_11_12, hi_11_12, \
                           stg6_0, stg4_0, stg6_0, stg4_0, \
                           stp2_10, stp2_13, stp2_11, stp2_12) \
  }

#define IDCT16_10 \
    /* Stage2 */ \
    { \
      const __m128i lo_1_15 = _mm_unpacklo_epi16(in[1], zero); \
      const __m128i hi_1_15 = _mm_unpackhi_epi16(in[1], zero); \
      const __m128i lo_13_3 = _mm_unpacklo_epi16(zero, in[3]); \
      const __m128i hi_13_3 = _mm_unpackhi_epi16(zero, in[3]); \
      \
      MULTIPLICATION_AND_ADD(lo_1_15, hi_1_15, lo_13_3, hi_13_3, \
                             stg2_0, stg2_1, stg2_6, stg2_7, \
                             stp1_8_0, stp1_15, stp1_11, stp1_12_0) \
    } \
      \
    /* Stage3 */ \
    { \
      const __m128i lo_2_14 = _mm_unpacklo_epi16(in[2], zero); \
      const __m128i hi_2_14 = _mm_unpackhi_epi16(in[2], zero); \
      \
      MULTIPLICATION_AND_ADD_2(lo_2_14, hi_2_14, \
                               stg3_0, stg3_1,  \
                               stp2_4, stp2_7) \
      \
      stp1_9  =  stp1_8_0; \
      stp1_10 =  stp1_11;  \
      \
      stp1_13 = stp1_12_0; \
      stp1_14 = stp1_15;   \
    } \
    \
    /* Stage4 */ \
    { \
      const __m128i lo_0_8 = _mm_unpacklo_epi16(in[0], zero); \
      const __m128i hi_0_8 = _mm_unpackhi_epi16(in[0], zero); \
      \
      const __m128i lo_9_14 = _mm_unpacklo_epi16(stp1_9, stp1_14); \
      const __m128i hi_9_14 = _mm_unpackhi_epi16(stp1_9, stp1_14); \
      const __m128i lo_10_13 = _mm_unpacklo_epi16(stp1_10, stp1_13); \
      const __m128i hi_10_13 = _mm_unpackhi_epi16(stp1_10, stp1_13); \
      \
      MULTIPLICATION_AND_ADD_2(lo_0_8, hi_0_8, \
                               stg4_0, stg4_1, \
                               stp1_0, stp1_1) \
      stp2_5 = stp2_4; \
      stp2_6 = stp2_7; \
      \
      MULTIPLICATION_AND_ADD(lo_9_14, hi_9_14, lo_10_13, hi_10_13, \
                             stg4_4, stg4_5, stg4_6, stg4_7, \
                             stp2_9, stp2_14, stp2_10, stp2_13) \
    } \
      \
    /* Stage5 */ \
    { \
      const __m128i lo_6_5 = _mm_unpacklo_epi16(stp2_6, stp2_5); \
      const __m128i hi_6_5 = _mm_unpackhi_epi16(stp2_6, stp2_5); \
      \
      stp1_2 = stp1_1; \
      stp1_3 = stp1_0; \
      \
      tmp0 = _mm_madd_epi16(lo_6_5, stg4_1); \
      tmp1 = _mm_madd_epi16(hi_6_5, stg4_1); \
      tmp2 = _mm_madd_epi16(lo_6_5, stg4_0); \
      tmp3 = _mm_madd_epi16(hi_6_5, stg4_0); \
      \
      tmp0 = _mm_add_epi32(tmp0, rounding); \
      tmp1 = _mm_add_epi32(tmp1, rounding); \
      tmp2 = _mm_add_epi32(tmp2, rounding); \
      tmp3 = _mm_add_epi32(tmp3, rounding); \
      \
      tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS); \
      tmp1 = _mm_srai_epi32(tmp1, DCT_CONST_BITS); \
      tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS); \
      tmp3 = _mm_srai_epi32(tmp3, DCT_CONST_BITS); \
      \
      stp1_5 = _mm_packs_epi32(tmp0, tmp1); \
      stp1_6 = _mm_packs_epi32(tmp2, tmp3); \
      \
      stp1_8 = _mm_add_epi16(stp1_8_0, stp1_11);  \
      stp1_9 = _mm_add_epi16(stp2_9, stp2_10);    \
      stp1_10 = _mm_sub_epi16(stp2_9, stp2_10);   \
      stp1_11 = _mm_sub_epi16(stp1_8_0, stp1_11); \
      \
      stp1_12 = _mm_sub_epi16(stp1_15, stp1_12_0); \
      stp1_13 = _mm_sub_epi16(stp2_14, stp2_13);   \
      stp1_14 = _mm_add_epi16(stp2_14, stp2_13);   \
      stp1_15 = _mm_add_epi16(stp1_15, stp1_12_0); \
    } \
      \
    /* Stage6 */ \
    { \
      const __m128i lo_10_13 = _mm_unpacklo_epi16(stp1_10, stp1_13); \
      const __m128i hi_10_13 = _mm_unpackhi_epi16(stp1_10, stp1_13); \
      const __m128i lo_11_12 = _mm_unpacklo_epi16(stp1_11, stp1_12); \
      const __m128i hi_11_12 = _mm_unpackhi_epi16(stp1_11, stp1_12); \
      \
      stp2_0 = _mm_add_epi16(stp1_0, stp2_7); \
      stp2_1 = _mm_add_epi16(stp1_1, stp1_6); \
      stp2_2 = _mm_add_epi16(stp1_2, stp1_5); \
      stp2_3 = _mm_add_epi16(stp1_3, stp2_4); \
      stp2_4 = _mm_sub_epi16(stp1_3, stp2_4); \
      stp2_5 = _mm_sub_epi16(stp1_2, stp1_5); \
      stp2_6 = _mm_sub_epi16(stp1_1, stp1_6); \
      stp2_7 = _mm_sub_epi16(stp1_0, stp2_7); \
      \
      MULTIPLICATION_AND_ADD(lo_10_13, hi_10_13, lo_11_12, hi_11_12, \
                             stg6_0, stg4_0, stg6_0, stg4_0, \
                             stp2_10, stp2_13, stp2_11, stp2_12) \
    }

void vpx_idct16x16_256_add_sse2(const tran_low_t *input, uint8_t *dest,
                                int stride) {
  const __m128i rounding = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i final_rounding = _mm_set1_epi16(1 << 5);
  const __m128i zero = _mm_setzero_si128();

  const __m128i stg2_0 = pair_set_epi16(cospi_30_64, -cospi_2_64);
  const __m128i stg2_1 = pair_set_epi16(cospi_2_64, cospi_30_64);
  const __m128i stg2_2 = pair_set_epi16(cospi_14_64, -cospi_18_64);
  const __m128i stg2_3 = pair_set_epi16(cospi_18_64, cospi_14_64);
  const __m128i stg2_4 = pair_set_epi16(cospi_22_64, -cospi_10_64);
  const __m128i stg2_5 = pair_set_epi16(cospi_10_64, cospi_22_64);
  const __m128i stg2_6 = pair_set_epi16(cospi_6_64, -cospi_26_64);
  const __m128i stg2_7 = pair_set_epi16(cospi_26_64, cospi_6_64);

  const __m128i stg3_0 = pair_set_epi16(cospi_28_64, -cospi_4_64);
  const __m128i stg3_1 = pair_set_epi16(cospi_4_64, cospi_28_64);
  const __m128i stg3_2 = pair_set_epi16(cospi_12_64, -cospi_20_64);
  const __m128i stg3_3 = pair_set_epi16(cospi_20_64, cospi_12_64);

  const __m128i stg4_0 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i stg4_1 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i stg4_2 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i stg4_3 = pair_set_epi16(cospi_8_64, cospi_24_64);
  const __m128i stg4_4 = pair_set_epi16(-cospi_8_64, cospi_24_64);
  const __m128i stg4_5 = pair_set_epi16(cospi_24_64, cospi_8_64);
  const __m128i stg4_6 = pair_set_epi16(-cospi_24_64, -cospi_8_64);
  const __m128i stg4_7 = pair_set_epi16(-cospi_8_64, cospi_24_64);

  const __m128i stg6_0 = pair_set_epi16(-cospi_16_64, cospi_16_64);

  __m128i in[16], l[16], r[16], *curr1;
  __m128i stp1_0, stp1_1, stp1_2, stp1_3, stp1_4, stp1_5, stp1_6, stp1_7,
          stp1_8, stp1_9, stp1_10, stp1_11, stp1_12, stp1_13, stp1_14, stp1_15,
          stp1_8_0, stp1_12_0;
  __m128i stp2_0, stp2_1, stp2_2, stp2_3, stp2_4, stp2_5, stp2_6, stp2_7,
          stp2_8, stp2_9, stp2_10, stp2_11, stp2_12, stp2_13, stp2_14, stp2_15;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  int i;

  curr1 = l;
  for (i = 0; i < 2; i++) {
    // 1-D idct

    // Load input data.
    in[0] = load_input_data(input);
    in[8] = load_input_data(input + 8 * 1);
    in[1] = load_input_data(input + 8 * 2);
    in[9] = load_input_data(input + 8 * 3);
    in[2] = load_input_data(input + 8 * 4);
    in[10] = load_input_data(input + 8 * 5);
    in[3] = load_input_data(input + 8 * 6);
    in[11] = load_input_data(input + 8 * 7);
    in[4] = load_input_data(input + 8 * 8);
    in[12] = load_input_data(input + 8 * 9);
    in[5] = load_input_data(input + 8 * 10);
    in[13] = load_input_data(input + 8 * 11);
    in[6] = load_input_data(input + 8 * 12);
    in[14] = load_input_data(input + 8 * 13);
    in[7] = load_input_data(input + 8 * 14);
    in[15] = load_input_data(input + 8 * 15);

    array_transpose_8x8(in, in);
    array_transpose_8x8(in + 8, in + 8);

    IDCT16

    // Stage7
    curr1[0] = _mm_add_epi16(stp2_0, stp1_15);
    curr1[1] = _mm_add_epi16(stp2_1, stp1_14);
    curr1[2] = _mm_add_epi16(stp2_2, stp2_13);
    curr1[3] = _mm_add_epi16(stp2_3, stp2_12);
    curr1[4] = _mm_add_epi16(stp2_4, stp2_11);
    curr1[5] = _mm_add_epi16(stp2_5, stp2_10);
    curr1[6] = _mm_add_epi16(stp2_6, stp1_9);
    curr1[7] = _mm_add_epi16(stp2_7, stp1_8);
    curr1[8] = _mm_sub_epi16(stp2_7, stp1_8);
    curr1[9] = _mm_sub_epi16(stp2_6, stp1_9);
    curr1[10] = _mm_sub_epi16(stp2_5, stp2_10);
    curr1[11] = _mm_sub_epi16(stp2_4, stp2_11);
    curr1[12] = _mm_sub_epi16(stp2_3, stp2_12);
    curr1[13] = _mm_sub_epi16(stp2_2, stp2_13);
    curr1[14] = _mm_sub_epi16(stp2_1, stp1_14);
    curr1[15] = _mm_sub_epi16(stp2_0, stp1_15);

    curr1 = r;
    input += 128;
  }
  for (i = 0; i < 2; i++) {
    int j;
    // 1-D idct
    array_transpose_8x8(l + i * 8, in);
    array_transpose_8x8(r + i * 8, in + 8);

    IDCT16

    // 2-D
    in[0] = _mm_add_epi16(stp2_0, stp1_15);
    in[1] = _mm_add_epi16(stp2_1, stp1_14);
    in[2] = _mm_add_epi16(stp2_2, stp2_13);
    in[3] = _mm_add_epi16(stp2_3, stp2_12);
    in[4] = _mm_add_epi16(stp2_4, stp2_11);
    in[5] = _mm_add_epi16(stp2_5, stp2_10);
    in[6] = _mm_add_epi16(stp2_6, stp1_9);
    in[7] = _mm_add_epi16(stp2_7, stp1_8);
    in[8] = _mm_sub_epi16(stp2_7, stp1_8);
    in[9] = _mm_sub_epi16(stp2_6, stp1_9);
    in[10] = _mm_sub_epi16(stp2_5, stp2_10);
    in[11] = _mm_sub_epi16(stp2_4, stp2_11);
    in[12] = _mm_sub_epi16(stp2_3, stp2_12);
    in[13] = _mm_sub_epi16(stp2_2, stp2_13);
    in[14] = _mm_sub_epi16(stp2_1, stp1_14);
    in[15] = _mm_sub_epi16(stp2_0, stp1_15);

    for (j = 0; j < 16; ++j) {
      // Final rounding and shift
      in[j] = _mm_adds_epi16(in[j], final_rounding);
      in[j] = _mm_srai_epi16(in[j], 6);
      RECON_AND_STORE(dest + j * stride, in[j]);
    }

    dest += 8;
  }
}

void vpx_idct16x16_1_add_sse2(const tran_low_t *input, uint8_t *dest,
                              int stride) {
  __m128i dc_value;
  const __m128i zero = _mm_setzero_si128();
  int a, i;

  a = (int)dct_const_round_shift(input[0] * cospi_16_64);
  a = (int)dct_const_round_shift(a * cospi_16_64);
  a = ROUND_POWER_OF_TWO(a, 6);

  dc_value = _mm_set1_epi16(a);

  for (i = 0; i < 16; ++i) {
    RECON_AND_STORE(dest +  0, dc_value);
    RECON_AND_STORE(dest +  8, dc_value);
    dest += stride;
  }
}

static void iadst16_8col(__m128i *in) {
  // perform 16x16 1-D ADST for 8 columns
  __m128i s[16], x[16], u[32], v[32];
  const __m128i k__cospi_p01_p31 = pair_set_epi16(cospi_1_64, cospi_31_64);
  const __m128i k__cospi_p31_m01 = pair_set_epi16(cospi_31_64, -cospi_1_64);
  const __m128i k__cospi_p05_p27 = pair_set_epi16(cospi_5_64, cospi_27_64);
  const __m128i k__cospi_p27_m05 = pair_set_epi16(cospi_27_64, -cospi_5_64);
  const __m128i k__cospi_p09_p23 = pair_set_epi16(cospi_9_64, cospi_23_64);
  const __m128i k__cospi_p23_m09 = pair_set_epi16(cospi_23_64, -cospi_9_64);
  const __m128i k__cospi_p13_p19 = pair_set_epi16(cospi_13_64, cospi_19_64);
  const __m128i k__cospi_p19_m13 = pair_set_epi16(cospi_19_64, -cospi_13_64);
  const __m128i k__cospi_p17_p15 = pair_set_epi16(cospi_17_64, cospi_15_64);
  const __m128i k__cospi_p15_m17 = pair_set_epi16(cospi_15_64, -cospi_17_64);
  const __m128i k__cospi_p21_p11 = pair_set_epi16(cospi_21_64, cospi_11_64);
  const __m128i k__cospi_p11_m21 = pair_set_epi16(cospi_11_64, -cospi_21_64);
  const __m128i k__cospi_p25_p07 = pair_set_epi16(cospi_25_64, cospi_7_64);
  const __m128i k__cospi_p07_m25 = pair_set_epi16(cospi_7_64, -cospi_25_64);
  const __m128i k__cospi_p29_p03 = pair_set_epi16(cospi_29_64, cospi_3_64);
  const __m128i k__cospi_p03_m29 = pair_set_epi16(cospi_3_64, -cospi_29_64);
  const __m128i k__cospi_p04_p28 = pair_set_epi16(cospi_4_64, cospi_28_64);
  const __m128i k__cospi_p28_m04 = pair_set_epi16(cospi_28_64, -cospi_4_64);
  const __m128i k__cospi_p20_p12 = pair_set_epi16(cospi_20_64, cospi_12_64);
  const __m128i k__cospi_p12_m20 = pair_set_epi16(cospi_12_64, -cospi_20_64);
  const __m128i k__cospi_m28_p04 = pair_set_epi16(-cospi_28_64, cospi_4_64);
  const __m128i k__cospi_m12_p20 = pair_set_epi16(-cospi_12_64, cospi_20_64);
  const __m128i k__cospi_p08_p24 = pair_set_epi16(cospi_8_64, cospi_24_64);
  const __m128i k__cospi_p24_m08 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i k__cospi_m24_p08 = pair_set_epi16(-cospi_24_64, cospi_8_64);
  const __m128i k__cospi_m16_m16 = _mm_set1_epi16((int16_t)-cospi_16_64);
  const __m128i k__cospi_p16_p16 = _mm_set1_epi16((int16_t)cospi_16_64);
  const __m128i k__cospi_p16_m16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_m16_p16 = pair_set_epi16(-cospi_16_64, cospi_16_64);
  const __m128i k__DCT_CONST_ROUNDING = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i kZero = _mm_set1_epi16(0);

  u[0] = _mm_unpacklo_epi16(in[15], in[0]);
  u[1] = _mm_unpackhi_epi16(in[15], in[0]);
  u[2] = _mm_unpacklo_epi16(in[13], in[2]);
  u[3] = _mm_unpackhi_epi16(in[13], in[2]);
  u[4] = _mm_unpacklo_epi16(in[11], in[4]);
  u[5] = _mm_unpackhi_epi16(in[11], in[4]);
  u[6] = _mm_unpacklo_epi16(in[9], in[6]);
  u[7] = _mm_unpackhi_epi16(in[9], in[6]);
  u[8] = _mm_unpacklo_epi16(in[7], in[8]);
  u[9] = _mm_unpackhi_epi16(in[7], in[8]);
  u[10] = _mm_unpacklo_epi16(in[5], in[10]);
  u[11] = _mm_unpackhi_epi16(in[5], in[10]);
  u[12] = _mm_unpacklo_epi16(in[3], in[12]);
  u[13] = _mm_unpackhi_epi16(in[3], in[12]);
  u[14] = _mm_unpacklo_epi16(in[1], in[14]);
  u[15] = _mm_unpackhi_epi16(in[1], in[14]);

  v[0] = _mm_madd_epi16(u[0], k__cospi_p01_p31);
  v[1] = _mm_madd_epi16(u[1], k__cospi_p01_p31);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p31_m01);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p31_m01);
  v[4] = _mm_madd_epi16(u[2], k__cospi_p05_p27);
  v[5] = _mm_madd_epi16(u[3], k__cospi_p05_p27);
  v[6] = _mm_madd_epi16(u[2], k__cospi_p27_m05);
  v[7] = _mm_madd_epi16(u[3], k__cospi_p27_m05);
  v[8] = _mm_madd_epi16(u[4], k__cospi_p09_p23);
  v[9] = _mm_madd_epi16(u[5], k__cospi_p09_p23);
  v[10] = _mm_madd_epi16(u[4], k__cospi_p23_m09);
  v[11] = _mm_madd_epi16(u[5], k__cospi_p23_m09);
  v[12] = _mm_madd_epi16(u[6], k__cospi_p13_p19);
  v[13] = _mm_madd_epi16(u[7], k__cospi_p13_p19);
  v[14] = _mm_madd_epi16(u[6], k__cospi_p19_m13);
  v[15] = _mm_madd_epi16(u[7], k__cospi_p19_m13);
  v[16] = _mm_madd_epi16(u[8], k__cospi_p17_p15);
  v[17] = _mm_madd_epi16(u[9], k__cospi_p17_p15);
  v[18] = _mm_madd_epi16(u[8], k__cospi_p15_m17);
  v[19] = _mm_madd_epi16(u[9], k__cospi_p15_m17);
  v[20] = _mm_madd_epi16(u[10], k__cospi_p21_p11);
  v[21] = _mm_madd_epi16(u[11], k__cospi_p21_p11);
  v[22] = _mm_madd_epi16(u[10], k__cospi_p11_m21);
  v[23] = _mm_madd_epi16(u[11], k__cospi_p11_m21);
  v[24] = _mm_madd_epi16(u[12], k__cospi_p25_p07);
  v[25] = _mm_madd_epi16(u[13], k__cospi_p25_p07);
  v[26] = _mm_madd_epi16(u[12], k__cospi_p07_m25);
  v[27] = _mm_madd_epi16(u[13], k__cospi_p07_m25);
  v[28] = _mm_madd_epi16(u[14], k__cospi_p29_p03);
  v[29] = _mm_madd_epi16(u[15], k__cospi_p29_p03);
  v[30] = _mm_madd_epi16(u[14], k__cospi_p03_m29);
  v[31] = _mm_madd_epi16(u[15], k__cospi_p03_m29);

  u[0] = _mm_add_epi32(v[0], v[16]);
  u[1] = _mm_add_epi32(v[1], v[17]);
  u[2] = _mm_add_epi32(v[2], v[18]);
  u[3] = _mm_add_epi32(v[3], v[19]);
  u[4] = _mm_add_epi32(v[4], v[20]);
  u[5] = _mm_add_epi32(v[5], v[21]);
  u[6] = _mm_add_epi32(v[6], v[22]);
  u[7] = _mm_add_epi32(v[7], v[23]);
  u[8] = _mm_add_epi32(v[8], v[24]);
  u[9] = _mm_add_epi32(v[9], v[25]);
  u[10] = _mm_add_epi32(v[10], v[26]);
  u[11] = _mm_add_epi32(v[11], v[27]);
  u[12] = _mm_add_epi32(v[12], v[28]);
  u[13] = _mm_add_epi32(v[13], v[29]);
  u[14] = _mm_add_epi32(v[14], v[30]);
  u[15] = _mm_add_epi32(v[15], v[31]);
  u[16] = _mm_sub_epi32(v[0], v[16]);
  u[17] = _mm_sub_epi32(v[1], v[17]);
  u[18] = _mm_sub_epi32(v[2], v[18]);
  u[19] = _mm_sub_epi32(v[3], v[19]);
  u[20] = _mm_sub_epi32(v[4], v[20]);
  u[21] = _mm_sub_epi32(v[5], v[21]);
  u[22] = _mm_sub_epi32(v[6], v[22]);
  u[23] = _mm_sub_epi32(v[7], v[23]);
  u[24] = _mm_sub_epi32(v[8], v[24]);
  u[25] = _mm_sub_epi32(v[9], v[25]);
  u[26] = _mm_sub_epi32(v[10], v[26]);
  u[27] = _mm_sub_epi32(v[11], v[27]);
  u[28] = _mm_sub_epi32(v[12], v[28]);
  u[29] = _mm_sub_epi32(v[13], v[29]);
  u[30] = _mm_sub_epi32(v[14], v[30]);
  u[31] = _mm_sub_epi32(v[15], v[31]);

  v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
  v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
  v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
  v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
  v[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
  v[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
  v[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
  v[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);
  v[8] = _mm_add_epi32(u[8], k__DCT_CONST_ROUNDING);
  v[9] = _mm_add_epi32(u[9], k__DCT_CONST_ROUNDING);
  v[10] = _mm_add_epi32(u[10], k__DCT_CONST_ROUNDING);
  v[11] = _mm_add_epi32(u[11], k__DCT_CONST_ROUNDING);
  v[12] = _mm_add_epi32(u[12], k__DCT_CONST_ROUNDING);
  v[13] = _mm_add_epi32(u[13], k__DCT_CONST_ROUNDING);
  v[14] = _mm_add_epi32(u[14], k__DCT_CONST_ROUNDING);
  v[15] = _mm_add_epi32(u[15], k__DCT_CONST_ROUNDING);
  v[16] = _mm_add_epi32(u[16], k__DCT_CONST_ROUNDING);
  v[17] = _mm_add_epi32(u[17], k__DCT_CONST_ROUNDING);
  v[18] = _mm_add_epi32(u[18], k__DCT_CONST_ROUNDING);
  v[19] = _mm_add_epi32(u[19], k__DCT_CONST_ROUNDING);
  v[20] = _mm_add_epi32(u[20], k__DCT_CONST_ROUNDING);
  v[21] = _mm_add_epi32(u[21], k__DCT_CONST_ROUNDING);
  v[22] = _mm_add_epi32(u[22], k__DCT_CONST_ROUNDING);
  v[23] = _mm_add_epi32(u[23], k__DCT_CONST_ROUNDING);
  v[24] = _mm_add_epi32(u[24], k__DCT_CONST_ROUNDING);
  v[25] = _mm_add_epi32(u[25], k__DCT_CONST_ROUNDING);
  v[26] = _mm_add_epi32(u[26], k__DCT_CONST_ROUNDING);
  v[27] = _mm_add_epi32(u[27], k__DCT_CONST_ROUNDING);
  v[28] = _mm_add_epi32(u[28], k__DCT_CONST_ROUNDING);
  v[29] = _mm_add_epi32(u[29], k__DCT_CONST_ROUNDING);
  v[30] = _mm_add_epi32(u[30], k__DCT_CONST_ROUNDING);
  v[31] = _mm_add_epi32(u[31], k__DCT_CONST_ROUNDING);

  u[0] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
  u[1] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
  u[2] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
  u[3] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
  u[4] = _mm_srai_epi32(v[4], DCT_CONST_BITS);
  u[5] = _mm_srai_epi32(v[5], DCT_CONST_BITS);
  u[6] = _mm_srai_epi32(v[6], DCT_CONST_BITS);
  u[7] = _mm_srai_epi32(v[7], DCT_CONST_BITS);
  u[8] = _mm_srai_epi32(v[8], DCT_CONST_BITS);
  u[9] = _mm_srai_epi32(v[9], DCT_CONST_BITS);
  u[10] = _mm_srai_epi32(v[10], DCT_CONST_BITS);
  u[11] = _mm_srai_epi32(v[11], DCT_CONST_BITS);
  u[12] = _mm_srai_epi32(v[12], DCT_CONST_BITS);
  u[13] = _mm_srai_epi32(v[13], DCT_CONST_BITS);
  u[14] = _mm_srai_epi32(v[14], DCT_CONST_BITS);
  u[15] = _mm_srai_epi32(v[15], DCT_CONST_BITS);
  u[16] = _mm_srai_epi32(v[16], DCT_CONST_BITS);
  u[17] = _mm_srai_epi32(v[17], DCT_CONST_BITS);
  u[18] = _mm_srai_epi32(v[18], DCT_CONST_BITS);
  u[19] = _mm_srai_epi32(v[19], DCT_CONST_BITS);
  u[20] = _mm_srai_epi32(v[20], DCT_CONST_BITS);
  u[21] = _mm_srai_epi32(v[21], DCT_CONST_BITS);
  u[22] = _mm_srai_epi32(v[22], DCT_CONST_BITS);
  u[23] = _mm_srai_epi32(v[23], DCT_CONST_BITS);
  u[24] = _mm_srai_epi32(v[24], DCT_CONST_BITS);
  u[25] = _mm_srai_epi32(v[25], DCT_CONST_BITS);
  u[26] = _mm_srai_epi32(v[26], DCT_CONST_BITS);
  u[27] = _mm_srai_epi32(v[27], DCT_CONST_BITS);
  u[28] = _mm_srai_epi32(v[28], DCT_CONST_BITS);
  u[29] = _mm_srai_epi32(v[29], DCT_CONST_BITS);
  u[30] = _mm_srai_epi32(v[30], DCT_CONST_BITS);
  u[31] = _mm_srai_epi32(v[31], DCT_CONST_BITS);

  s[0] = _mm_packs_epi32(u[0], u[1]);
  s[1] = _mm_packs_epi32(u[2], u[3]);
  s[2] = _mm_packs_epi32(u[4], u[5]);
  s[3] = _mm_packs_epi32(u[6], u[7]);
  s[4] = _mm_packs_epi32(u[8], u[9]);
  s[5] = _mm_packs_epi32(u[10], u[11]);
  s[6] = _mm_packs_epi32(u[12], u[13]);
  s[7] = _mm_packs_epi32(u[14], u[15]);
  s[8] = _mm_packs_epi32(u[16], u[17]);
  s[9] = _mm_packs_epi32(u[18], u[19]);
  s[10] = _mm_packs_epi32(u[20], u[21]);
  s[11] = _mm_packs_epi32(u[22], u[23]);
  s[12] = _mm_packs_epi32(u[24], u[25]);
  s[13] = _mm_packs_epi32(u[26], u[27]);
  s[14] = _mm_packs_epi32(u[28], u[29]);
  s[15] = _mm_packs_epi32(u[30], u[31]);

  // stage 2
  u[0] = _mm_unpacklo_epi16(s[8], s[9]);
  u[1] = _mm_unpackhi_epi16(s[8], s[9]);
  u[2] = _mm_unpacklo_epi16(s[10], s[11]);
  u[3] = _mm_unpackhi_epi16(s[10], s[11]);
  u[4] = _mm_unpacklo_epi16(s[12], s[13]);
  u[5] = _mm_unpackhi_epi16(s[12], s[13]);
  u[6] = _mm_unpacklo_epi16(s[14], s[15]);
  u[7] = _mm_unpackhi_epi16(s[14], s[15]);

  v[0] = _mm_madd_epi16(u[0], k__cospi_p04_p28);
  v[1] = _mm_madd_epi16(u[1], k__cospi_p04_p28);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p28_m04);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p28_m04);
  v[4] = _mm_madd_epi16(u[2], k__cospi_p20_p12);
  v[5] = _mm_madd_epi16(u[3], k__cospi_p20_p12);
  v[6] = _mm_madd_epi16(u[2], k__cospi_p12_m20);
  v[7] = _mm_madd_epi16(u[3], k__cospi_p12_m20);
  v[8] = _mm_madd_epi16(u[4], k__cospi_m28_p04);
  v[9] = _mm_madd_epi16(u[5], k__cospi_m28_p04);
  v[10] = _mm_madd_epi16(u[4], k__cospi_p04_p28);
  v[11] = _mm_madd_epi16(u[5], k__cospi_p04_p28);
  v[12] = _mm_madd_epi16(u[6], k__cospi_m12_p20);
  v[13] = _mm_madd_epi16(u[7], k__cospi_m12_p20);
  v[14] = _mm_madd_epi16(u[6], k__cospi_p20_p12);
  v[15] = _mm_madd_epi16(u[7], k__cospi_p20_p12);

  u[0] = _mm_add_epi32(v[0], v[8]);
  u[1] = _mm_add_epi32(v[1], v[9]);
  u[2] = _mm_add_epi32(v[2], v[10]);
  u[3] = _mm_add_epi32(v[3], v[11]);
  u[4] = _mm_add_epi32(v[4], v[12]);
  u[5] = _mm_add_epi32(v[5], v[13]);
  u[6] = _mm_add_epi32(v[6], v[14]);
  u[7] = _mm_add_epi32(v[7], v[15]);
  u[8] = _mm_sub_epi32(v[0], v[8]);
  u[9] = _mm_sub_epi32(v[1], v[9]);
  u[10] = _mm_sub_epi32(v[2], v[10]);
  u[11] = _mm_sub_epi32(v[3], v[11]);
  u[12] = _mm_sub_epi32(v[4], v[12]);
  u[13] = _mm_sub_epi32(v[5], v[13]);
  u[14] = _mm_sub_epi32(v[6], v[14]);
  u[15] = _mm_sub_epi32(v[7], v[15]);

  v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
  v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
  v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
  v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
  v[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
  v[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
  v[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
  v[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);
  v[8] = _mm_add_epi32(u[8], k__DCT_CONST_ROUNDING);
  v[9] = _mm_add_epi32(u[9], k__DCT_CONST_ROUNDING);
  v[10] = _mm_add_epi32(u[10], k__DCT_CONST_ROUNDING);
  v[11] = _mm_add_epi32(u[11], k__DCT_CONST_ROUNDING);
  v[12] = _mm_add_epi32(u[12], k__DCT_CONST_ROUNDING);
  v[13] = _mm_add_epi32(u[13], k__DCT_CONST_ROUNDING);
  v[14] = _mm_add_epi32(u[14], k__DCT_CONST_ROUNDING);
  v[15] = _mm_add_epi32(u[15], k__DCT_CONST_ROUNDING);

  u[0] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
  u[1] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
  u[2] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
  u[3] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
  u[4] = _mm_srai_epi32(v[4], DCT_CONST_BITS);
  u[5] = _mm_srai_epi32(v[5], DCT_CONST_BITS);
  u[6] = _mm_srai_epi32(v[6], DCT_CONST_BITS);
  u[7] = _mm_srai_epi32(v[7], DCT_CONST_BITS);
  u[8] = _mm_srai_epi32(v[8], DCT_CONST_BITS);
  u[9] = _mm_srai_epi32(v[9], DCT_CONST_BITS);
  u[10] = _mm_srai_epi32(v[10], DCT_CONST_BITS);
  u[11] = _mm_srai_epi32(v[11], DCT_CONST_BITS);
  u[12] = _mm_srai_epi32(v[12], DCT_CONST_BITS);
  u[13] = _mm_srai_epi32(v[13], DCT_CONST_BITS);
  u[14] = _mm_srai_epi32(v[14], DCT_CONST_BITS);
  u[15] = _mm_srai_epi32(v[15], DCT_CONST_BITS);

  x[0] = _mm_add_epi16(s[0], s[4]);
  x[1] = _mm_add_epi16(s[1], s[5]);
  x[2] = _mm_add_epi16(s[2], s[6]);
  x[3] = _mm_add_epi16(s[3], s[7]);
  x[4] = _mm_sub_epi16(s[0], s[4]);
  x[5] = _mm_sub_epi16(s[1], s[5]);
  x[6] = _mm_sub_epi16(s[2], s[6]);
  x[7] = _mm_sub_epi16(s[3], s[7]);
  x[8] = _mm_packs_epi32(u[0], u[1]);
  x[9] = _mm_packs_epi32(u[2], u[3]);
  x[10] = _mm_packs_epi32(u[4], u[5]);
  x[11] = _mm_packs_epi32(u[6], u[7]);
  x[12] = _mm_packs_epi32(u[8], u[9]);
  x[13] = _mm_packs_epi32(u[10], u[11]);
  x[14] = _mm_packs_epi32(u[12], u[13]);
  x[15] = _mm_packs_epi32(u[14], u[15]);

  // stage 3
  u[0] = _mm_unpacklo_epi16(x[4], x[5]);
  u[1] = _mm_unpackhi_epi16(x[4], x[5]);
  u[2] = _mm_unpacklo_epi16(x[6], x[7]);
  u[3] = _mm_unpackhi_epi16(x[6], x[7]);
  u[4] = _mm_unpacklo_epi16(x[12], x[13]);
  u[5] = _mm_unpackhi_epi16(x[12], x[13]);
  u[6] = _mm_unpacklo_epi16(x[14], x[15]);
  u[7] = _mm_unpackhi_epi16(x[14], x[15]);

  v[0] = _mm_madd_epi16(u[0], k__cospi_p08_p24);
  v[1] = _mm_madd_epi16(u[1], k__cospi_p08_p24);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p24_m08);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p24_m08);
  v[4] = _mm_madd_epi16(u[2], k__cospi_m24_p08);
  v[5] = _mm_madd_epi16(u[3], k__cospi_m24_p08);
  v[6] = _mm_madd_epi16(u[2], k__cospi_p08_p24);
  v[7] = _mm_madd_epi16(u[3], k__cospi_p08_p24);
  v[8] = _mm_madd_epi16(u[4], k__cospi_p08_p24);
  v[9] = _mm_madd_epi16(u[5], k__cospi_p08_p24);
  v[10] = _mm_madd_epi16(u[4], k__cospi_p24_m08);
  v[11] = _mm_madd_epi16(u[5], k__cospi_p24_m08);
  v[12] = _mm_madd_epi16(u[6], k__cospi_m24_p08);
  v[13] = _mm_madd_epi16(u[7], k__cospi_m24_p08);
  v[14] = _mm_madd_epi16(u[6], k__cospi_p08_p24);
  v[15] = _mm_madd_epi16(u[7], k__cospi_p08_p24);

  u[0] = _mm_add_epi32(v[0], v[4]);
  u[1] = _mm_add_epi32(v[1], v[5]);
  u[2] = _mm_add_epi32(v[2], v[6]);
  u[3] = _mm_add_epi32(v[3], v[7]);
  u[4] = _mm_sub_epi32(v[0], v[4]);
  u[5] = _mm_sub_epi32(v[1], v[5]);
  u[6] = _mm_sub_epi32(v[2], v[6]);
  u[7] = _mm_sub_epi32(v[3], v[7]);
  u[8] = _mm_add_epi32(v[8], v[12]);
  u[9] = _mm_add_epi32(v[9], v[13]);
  u[10] = _mm_add_epi32(v[10], v[14]);
  u[11] = _mm_add_epi32(v[11], v[15]);
  u[12] = _mm_sub_epi32(v[8], v[12]);
  u[13] = _mm_sub_epi32(v[9], v[13]);
  u[14] = _mm_sub_epi32(v[10], v[14]);
  u[15] = _mm_sub_epi32(v[11], v[15]);

  u[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
  u[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
  u[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
  u[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
  u[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
  u[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
  u[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
  u[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);
  u[8] = _mm_add_epi32(u[8], k__DCT_CONST_ROUNDING);
  u[9] = _mm_add_epi32(u[9], k__DCT_CONST_ROUNDING);
  u[10] = _mm_add_epi32(u[10], k__DCT_CONST_ROUNDING);
  u[11] = _mm_add_epi32(u[11], k__DCT_CONST_ROUNDING);
  u[12] = _mm_add_epi32(u[12], k__DCT_CONST_ROUNDING);
  u[13] = _mm_add_epi32(u[13], k__DCT_CONST_ROUNDING);
  u[14] = _mm_add_epi32(u[14], k__DCT_CONST_ROUNDING);
  u[15] = _mm_add_epi32(u[15], k__DCT_CONST_ROUNDING);

  v[0] = _mm_srai_epi32(u[0], DCT_CONST_BITS);
  v[1] = _mm_srai_epi32(u[1], DCT_CONST_BITS);
  v[2] = _mm_srai_epi32(u[2], DCT_CONST_BITS);
  v[3] = _mm_srai_epi32(u[3], DCT_CONST_BITS);
  v[4] = _mm_srai_epi32(u[4], DCT_CONST_BITS);
  v[5] = _mm_srai_epi32(u[5], DCT_CONST_BITS);
  v[6] = _mm_srai_epi32(u[6], DCT_CONST_BITS);
  v[7] = _mm_srai_epi32(u[7], DCT_CONST_BITS);
  v[8] = _mm_srai_epi32(u[8], DCT_CONST_BITS);
  v[9] = _mm_srai_epi32(u[9], DCT_CONST_BITS);
  v[10] = _mm_srai_epi32(u[10], DCT_CONST_BITS);
  v[11] = _mm_srai_epi32(u[11], DCT_CONST_BITS);
  v[12] = _mm_srai_epi32(u[12], DCT_CONST_BITS);
  v[13] = _mm_srai_epi32(u[13], DCT_CONST_BITS);
  v[14] = _mm_srai_epi32(u[14], DCT_CONST_BITS);
  v[15] = _mm_srai_epi32(u[15], DCT_CONST_BITS);

  s[0] = _mm_add_epi16(x[0], x[2]);
  s[1] = _mm_add_epi16(x[1], x[3]);
  s[2] = _mm_sub_epi16(x[0], x[2]);
  s[3] = _mm_sub_epi16(x[1], x[3]);
  s[4] = _mm_packs_epi32(v[0], v[1]);
  s[5] = _mm_packs_epi32(v[2], v[3]);
  s[6] = _mm_packs_epi32(v[4], v[5]);
  s[7] = _mm_packs_epi32(v[6], v[7]);
  s[8] = _mm_add_epi16(x[8], x[10]);
  s[9] = _mm_add_epi16(x[9], x[11]);
  s[10] = _mm_sub_epi16(x[8], x[10]);
  s[11] = _mm_sub_epi16(x[9], x[11]);
  s[12] = _mm_packs_epi32(v[8], v[9]);
  s[13] = _mm_packs_epi32(v[10], v[11]);
  s[14] = _mm_packs_epi32(v[12], v[13]);
  s[15] = _mm_packs_epi32(v[14], v[15]);

  // stage 4
  u[0] = _mm_unpacklo_epi16(s[2], s[3]);
  u[1] = _mm_unpackhi_epi16(s[2], s[3]);
  u[2] = _mm_unpacklo_epi16(s[6], s[7]);
  u[3] = _mm_unpackhi_epi16(s[6], s[7]);
  u[4] = _mm_unpacklo_epi16(s[10], s[11]);
  u[5] = _mm_unpackhi_epi16(s[10], s[11]);
  u[6] = _mm_unpacklo_epi16(s[14], s[15]);
  u[7] = _mm_unpackhi_epi16(s[14], s[15]);

  v[0] = _mm_madd_epi16(u[0], k__cospi_m16_m16);
  v[1] = _mm_madd_epi16(u[1], k__cospi_m16_m16);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p16_m16);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p16_m16);
  v[4] = _mm_madd_epi16(u[2], k__cospi_p16_p16);
  v[5] = _mm_madd_epi16(u[3], k__cospi_p16_p16);
  v[6] = _mm_madd_epi16(u[2], k__cospi_m16_p16);
  v[7] = _mm_madd_epi16(u[3], k__cospi_m16_p16);
  v[8] = _mm_madd_epi16(u[4], k__cospi_p16_p16);
  v[9] = _mm_madd_epi16(u[5], k__cospi_p16_p16);
  v[10] = _mm_madd_epi16(u[4], k__cospi_m16_p16);
  v[11] = _mm_madd_epi16(u[5], k__cospi_m16_p16);
  v[12] = _mm_madd_epi16(u[6], k__cospi_m16_m16);
  v[13] = _mm_madd_epi16(u[7], k__cospi_m16_m16);
  v[14] = _mm_madd_epi16(u[6], k__cospi_p16_m16);
  v[15] = _mm_madd_epi16(u[7], k__cospi_p16_m16);

  u[0] = _mm_add_epi32(v[0], k__DCT_CONST_ROUNDING);
  u[1] = _mm_add_epi32(v[1], k__DCT_CONST_ROUNDING);
  u[2] = _mm_add_epi32(v[2], k__DCT_CONST_ROUNDING);
  u[3] = _mm_add_epi32(v[3], k__DCT_CONST_ROUNDING);
  u[4] = _mm_add_epi32(v[4], k__DCT_CONST_ROUNDING);
  u[5] = _mm_add_epi32(v[5], k__DCT_CONST_ROUNDING);
  u[6] = _mm_add_epi32(v[6], k__DCT_CONST_ROUNDING);
  u[7] = _mm_add_epi32(v[7], k__DCT_CONST_ROUNDING);
  u[8] = _mm_add_epi32(v[8], k__DCT_CONST_ROUNDING);
  u[9] = _mm_add_epi32(v[9], k__DCT_CONST_ROUNDING);
  u[10] = _mm_add_epi32(v[10], k__DCT_CONST_ROUNDING);
  u[11] = _mm_add_epi32(v[11], k__DCT_CONST_ROUNDING);
  u[12] = _mm_add_epi32(v[12], k__DCT_CONST_ROUNDING);
  u[13] = _mm_add_epi32(v[13], k__DCT_CONST_ROUNDING);
  u[14] = _mm_add_epi32(v[14], k__DCT_CONST_ROUNDING);
  u[15] = _mm_add_epi32(v[15], k__DCT_CONST_ROUNDING);

  v[0] = _mm_srai_epi32(u[0], DCT_CONST_BITS);
  v[1] = _mm_srai_epi32(u[1], DCT_CONST_BITS);
  v[2] = _mm_srai_epi32(u[2], DCT_CONST_BITS);
  v[3] = _mm_srai_epi32(u[3], DCT_CONST_BITS);
  v[4] = _mm_srai_epi32(u[4], DCT_CONST_BITS);
  v[5] = _mm_srai_epi32(u[5], DCT_CONST_BITS);
  v[6] = _mm_srai_epi32(u[6], DCT_CONST_BITS);
  v[7] = _mm_srai_epi32(u[7], DCT_CONST_BITS);
  v[8] = _mm_srai_epi32(u[8], DCT_CONST_BITS);
  v[9] = _mm_srai_epi32(u[9], DCT_CONST_BITS);
  v[10] = _mm_srai_epi32(u[10], DCT_CONST_BITS);
  v[11] = _mm_srai_epi32(u[11], DCT_CONST_BITS);
  v[12] = _mm_srai_epi32(u[12], DCT_CONST_BITS);
  v[13] = _mm_srai_epi32(u[13], DCT_CONST_BITS);
  v[14] = _mm_srai_epi32(u[14], DCT_CONST_BITS);
  v[15] = _mm_srai_epi32(u[15], DCT_CONST_BITS);

  in[0] = s[0];
  in[1] = _mm_sub_epi16(kZero, s[8]);
  in[2] = s[12];
  in[3] = _mm_sub_epi16(kZero, s[4]);
  in[4] = _mm_packs_epi32(v[4], v[5]);
  in[5] = _mm_packs_epi32(v[12], v[13]);
  in[6] = _mm_packs_epi32(v[8], v[9]);
  in[7] = _mm_packs_epi32(v[0], v[1]);
  in[8] = _mm_packs_epi32(v[2], v[3]);
  in[9] = _mm_packs_epi32(v[10], v[11]);
  in[10] = _mm_packs_epi32(v[14], v[15]);
  in[11] = _mm_packs_epi32(v[6], v[7]);
  in[12] = s[5];
  in[13] = _mm_sub_epi16(kZero, s[13]);
  in[14] = s[9];
  in[15] = _mm_sub_epi16(kZero, s[1]);
}

static void idct16_8col(__m128i *in) {
  const __m128i k__cospi_p30_m02 = pair_set_epi16(cospi_30_64, -cospi_2_64);
  const __m128i k__cospi_p02_p30 = pair_set_epi16(cospi_2_64, cospi_30_64);
  const __m128i k__cospi_p14_m18 = pair_set_epi16(cospi_14_64, -cospi_18_64);
  const __m128i k__cospi_p18_p14 = pair_set_epi16(cospi_18_64, cospi_14_64);
  const __m128i k__cospi_p22_m10 = pair_set_epi16(cospi_22_64, -cospi_10_64);
  const __m128i k__cospi_p10_p22 = pair_set_epi16(cospi_10_64, cospi_22_64);
  const __m128i k__cospi_p06_m26 = pair_set_epi16(cospi_6_64, -cospi_26_64);
  const __m128i k__cospi_p26_p06 = pair_set_epi16(cospi_26_64, cospi_6_64);
  const __m128i k__cospi_p28_m04 = pair_set_epi16(cospi_28_64, -cospi_4_64);
  const __m128i k__cospi_p04_p28 = pair_set_epi16(cospi_4_64, cospi_28_64);
  const __m128i k__cospi_p12_m20 = pair_set_epi16(cospi_12_64, -cospi_20_64);
  const __m128i k__cospi_p20_p12 = pair_set_epi16(cospi_20_64, cospi_12_64);
  const __m128i k__cospi_p16_p16 = _mm_set1_epi16((int16_t)cospi_16_64);
  const __m128i k__cospi_p16_m16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_p24_m08 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i k__cospi_p08_p24 = pair_set_epi16(cospi_8_64, cospi_24_64);
  const __m128i k__cospi_m08_p24 = pair_set_epi16(-cospi_8_64, cospi_24_64);
  const __m128i k__cospi_p24_p08 = pair_set_epi16(cospi_24_64, cospi_8_64);
  const __m128i k__cospi_m24_m08 = pair_set_epi16(-cospi_24_64, -cospi_8_64);
  const __m128i k__cospi_m16_p16 = pair_set_epi16(-cospi_16_64, cospi_16_64);
  const __m128i k__DCT_CONST_ROUNDING = _mm_set1_epi32(DCT_CONST_ROUNDING);
  __m128i v[16], u[16], s[16], t[16];

  // stage 1
  s[0] = in[0];
  s[1] = in[8];
  s[2] = in[4];
  s[3] = in[12];
  s[4] = in[2];
  s[5] = in[10];
  s[6] = in[6];
  s[7] = in[14];
  s[8] = in[1];
  s[9] = in[9];
  s[10] = in[5];
  s[11] = in[13];
  s[12] = in[3];
  s[13] = in[11];
  s[14] = in[7];
  s[15] = in[15];

  // stage 2
  u[0] = _mm_unpacklo_epi16(s[8], s[15]);
  u[1] = _mm_unpackhi_epi16(s[8], s[15]);
  u[2] = _mm_unpacklo_epi16(s[9], s[14]);
  u[3] = _mm_unpackhi_epi16(s[9], s[14]);
  u[4] = _mm_unpacklo_epi16(s[10], s[13]);
  u[5] = _mm_unpackhi_epi16(s[10], s[13]);
  u[6] = _mm_unpacklo_epi16(s[11], s[12]);
  u[7] = _mm_unpackhi_epi16(s[11], s[12]);

  v[0] = _mm_madd_epi16(u[0], k__cospi_p30_m02);
  v[1] = _mm_madd_epi16(u[1], k__cospi_p30_m02);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p02_p30);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p02_p30);
  v[4] = _mm_madd_epi16(u[2], k__cospi_p14_m18);
  v[5] = _mm_madd_epi16(u[3], k__cospi_p14_m18);
  v[6] = _mm_madd_epi16(u[2], k__cospi_p18_p14);
  v[7] = _mm_madd_epi16(u[3], k__cospi_p18_p14);
  v[8] = _mm_madd_epi16(u[4], k__cospi_p22_m10);
  v[9] = _mm_madd_epi16(u[5], k__cospi_p22_m10);
  v[10] = _mm_madd_epi16(u[4], k__cospi_p10_p22);
  v[11] = _mm_madd_epi16(u[5], k__cospi_p10_p22);
  v[12] = _mm_madd_epi16(u[6], k__cospi_p06_m26);
  v[13] = _mm_madd_epi16(u[7], k__cospi_p06_m26);
  v[14] = _mm_madd_epi16(u[6], k__cospi_p26_p06);
  v[15] = _mm_madd_epi16(u[7], k__cospi_p26_p06);

  u[0] = _mm_add_epi32(v[0], k__DCT_CONST_ROUNDING);
  u[1] = _mm_add_epi32(v[1], k__DCT_CONST_ROUNDING);
  u[2] = _mm_add_epi32(v[2], k__DCT_CONST_ROUNDING);
  u[3] = _mm_add_epi32(v[3], k__DCT_CONST_ROUNDING);
  u[4] = _mm_add_epi32(v[4], k__DCT_CONST_ROUNDING);
  u[5] = _mm_add_epi32(v[5], k__DCT_CONST_ROUNDING);
  u[6] = _mm_add_epi32(v[6], k__DCT_CONST_ROUNDING);
  u[7] = _mm_add_epi32(v[7], k__DCT_CONST_ROUNDING);
  u[8] = _mm_add_epi32(v[8], k__DCT_CONST_ROUNDING);
  u[9] = _mm_add_epi32(v[9], k__DCT_CONST_ROUNDING);
  u[10] = _mm_add_epi32(v[10], k__DCT_CONST_ROUNDING);
  u[11] = _mm_add_epi32(v[11], k__DCT_CONST_ROUNDING);
  u[12] = _mm_add_epi32(v[12], k__DCT_CONST_ROUNDING);
  u[13] = _mm_add_epi32(v[13], k__DCT_CONST_ROUNDING);
  u[14] = _mm_add_epi32(v[14], k__DCT_CONST_ROUNDING);
  u[15] = _mm_add_epi32(v[15], k__DCT_CONST_ROUNDING);

  u[0] = _mm_srai_epi32(u[0], DCT_CONST_BITS);
  u[1] = _mm_srai_epi32(u[1], DCT_CONST_BITS);
  u[2] = _mm_srai_epi32(u[2], DCT_CONST_BITS);
  u[3] = _mm_srai_epi32(u[3], DCT_CONST_BITS);
  u[4] = _mm_srai_epi32(u[4], DCT_CONST_BITS);
  u[5] = _mm_srai_epi32(u[5], DCT_CONST_BITS);
  u[6] = _mm_srai_epi32(u[6], DCT_CONST_BITS);
  u[7] = _mm_srai_epi32(u[7], DCT_CONST_BITS);
  u[8] = _mm_srai_epi32(u[8], DCT_CONST_BITS);
  u[9] = _mm_srai_epi32(u[9], DCT_CONST_BITS);
  u[10] = _mm_srai_epi32(u[10], DCT_CONST_BITS);
  u[11] = _mm_srai_epi32(u[11], DCT_CONST_BITS);
  u[12] = _mm_srai_epi32(u[12], DCT_CONST_BITS);
  u[13] = _mm_srai_epi32(u[13], DCT_CONST_BITS);
  u[14] = _mm_srai_epi32(u[14], DCT_CONST_BITS);
  u[15] = _mm_srai_epi32(u[15], DCT_CONST_BITS);

  s[8]  = _mm_packs_epi32(u[0], u[1]);
  s[15] = _mm_packs_epi32(u[2], u[3]);
  s[9]  = _mm_packs_epi32(u[4], u[5]);
  s[14] = _mm_packs_epi32(u[6], u[7]);
  s[10] = _mm_packs_epi32(u[8], u[9]);
  s[13] = _mm_packs_epi32(u[10], u[11]);
  s[11] = _mm_packs_epi32(u[12], u[13]);
  s[12] = _mm_packs_epi32(u[14], u[15]);

  // stage 3
  t[0] = s[0];
  t[1] = s[1];
  t[2] = s[2];
  t[3] = s[3];
  u[0] = _mm_unpacklo_epi16(s[4], s[7]);
  u[1] = _mm_unpackhi_epi16(s[4], s[7]);
  u[2] = _mm_unpacklo_epi16(s[5], s[6]);
  u[3] = _mm_unpackhi_epi16(s[5], s[6]);

  v[0] = _mm_madd_epi16(u[0], k__cospi_p28_m04);
  v[1] = _mm_madd_epi16(u[1], k__cospi_p28_m04);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p04_p28);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p04_p28);
  v[4] = _mm_madd_epi16(u[2], k__cospi_p12_m20);
  v[5] = _mm_madd_epi16(u[3], k__cospi_p12_m20);
  v[6] = _mm_madd_epi16(u[2], k__cospi_p20_p12);
  v[7] = _mm_madd_epi16(u[3], k__cospi_p20_p12);

  u[0] = _mm_add_epi32(v[0], k__DCT_CONST_ROUNDING);
  u[1] = _mm_add_epi32(v[1], k__DCT_CONST_ROUNDING);
  u[2] = _mm_add_epi32(v[2], k__DCT_CONST_ROUNDING);
  u[3] = _mm_add_epi32(v[3], k__DCT_CONST_ROUNDING);
  u[4] = _mm_add_epi32(v[4], k__DCT_CONST_ROUNDING);
  u[5] = _mm_add_epi32(v[5], k__DCT_CONST_ROUNDING);
  u[6] = _mm_add_epi32(v[6], k__DCT_CONST_ROUNDING);
  u[7] = _mm_add_epi32(v[7], k__DCT_CONST_ROUNDING);

  u[0] = _mm_srai_epi32(u[0], DCT_CONST_BITS);
  u[1] = _mm_srai_epi32(u[1], DCT_CONST_BITS);
  u[2] = _mm_srai_epi32(u[2], DCT_CONST_BITS);
  u[3] = _mm_srai_epi32(u[3], DCT_CONST_BITS);
  u[4] = _mm_srai_epi32(u[4], DCT_CONST_BITS);
  u[5] = _mm_srai_epi32(u[5], DCT_CONST_BITS);
  u[6] = _mm_srai_epi32(u[6], DCT_CONST_BITS);
  u[7] = _mm_srai_epi32(u[7], DCT_CONST_BITS);

  t[4] = _mm_packs_epi32(u[0], u[1]);
  t[7] = _mm_packs_epi32(u[2], u[3]);
  t[5] = _mm_packs_epi32(u[4], u[5]);
  t[6] = _mm_packs_epi32(u[6], u[7]);
  t[8] = _mm_add_epi16(s[8], s[9]);
  t[9] = _mm_sub_epi16(s[8], s[9]);
  t[10] = _mm_sub_epi16(s[11], s[10]);
  t[11] = _mm_add_epi16(s[10], s[11]);
  t[12] = _mm_add_epi16(s[12], s[13]);
  t[13] = _mm_sub_epi16(s[12], s[13]);
  t[14] = _mm_sub_epi16(s[15], s[14]);
  t[15] = _mm_add_epi16(s[14], s[15]);

  // stage 4
  u[0] = _mm_unpacklo_epi16(t[0], t[1]);
  u[1] = _mm_unpackhi_epi16(t[0], t[1]);
  u[2] = _mm_unpacklo_epi16(t[2], t[3]);
  u[3] = _mm_unpackhi_epi16(t[2], t[3]);
  u[4] = _mm_unpacklo_epi16(t[9], t[14]);
  u[5] = _mm_unpackhi_epi16(t[9], t[14]);
  u[6] = _mm_unpacklo_epi16(t[10], t[13]);
  u[7] = _mm_unpackhi_epi16(t[10], t[13]);

  v[0] = _mm_madd_epi16(u[0], k__cospi_p16_p16);
  v[1] = _mm_madd_epi16(u[1], k__cospi_p16_p16);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p16_m16);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p16_m16);
  v[4] = _mm_madd_epi16(u[2], k__cospi_p24_m08);
  v[5] = _mm_madd_epi16(u[3], k__cospi_p24_m08);
  v[6] = _mm_madd_epi16(u[2], k__cospi_p08_p24);
  v[7] = _mm_madd_epi16(u[3], k__cospi_p08_p24);
  v[8] = _mm_madd_epi16(u[4], k__cospi_m08_p24);
  v[9] = _mm_madd_epi16(u[5], k__cospi_m08_p24);
  v[10] = _mm_madd_epi16(u[4], k__cospi_p24_p08);
  v[11] = _mm_madd_epi16(u[5], k__cospi_p24_p08);
  v[12] = _mm_madd_epi16(u[6], k__cospi_m24_m08);
  v[13] = _mm_madd_epi16(u[7], k__cospi_m24_m08);
  v[14] = _mm_madd_epi16(u[6], k__cospi_m08_p24);
  v[15] = _mm_madd_epi16(u[7], k__cospi_m08_p24);

  u[0] = _mm_add_epi32(v[0], k__DCT_CONST_ROUNDING);
  u[1] = _mm_add_epi32(v[1], k__DCT_CONST_ROUNDING);
  u[2] = _mm_add_epi32(v[2], k__DCT_CONST_ROUNDING);
  u[3] = _mm_add_epi32(v[3], k__DCT_CONST_ROUNDING);
  u[4] = _mm_add_epi32(v[4], k__DCT_CONST_ROUNDING);
  u[5] = _mm_add_epi32(v[5], k__DCT_CONST_ROUNDING);
  u[6] = _mm_add_epi32(v[6], k__DCT_CONST_ROUNDING);
  u[7] = _mm_add_epi32(v[7], k__DCT_CONST_ROUNDING);
  u[8] = _mm_add_epi32(v[8], k__DCT_CONST_ROUNDING);
  u[9] = _mm_add_epi32(v[9], k__DCT_CONST_ROUNDING);
  u[10] = _mm_add_epi32(v[10], k__DCT_CONST_ROUNDING);
  u[11] = _mm_add_epi32(v[11], k__DCT_CONST_ROUNDING);
  u[12] = _mm_add_epi32(v[12], k__DCT_CONST_ROUNDING);
  u[13] = _mm_add_epi32(v[13], k__DCT_CONST_ROUNDING);
  u[14] = _mm_add_epi32(v[14], k__DCT_CONST_ROUNDING);
  u[15] = _mm_add_epi32(v[15], k__DCT_CONST_ROUNDING);

  u[0] = _mm_srai_epi32(u[0], DCT_CONST_BITS);
  u[1] = _mm_srai_epi32(u[1], DCT_CONST_BITS);
  u[2] = _mm_srai_epi32(u[2], DCT_CONST_BITS);
  u[3] = _mm_srai_epi32(u[3], DCT_CONST_BITS);
  u[4] = _mm_srai_epi32(u[4], DCT_CONST_BITS);
  u[5] = _mm_srai_epi32(u[5], DCT_CONST_BITS);
  u[6] = _mm_srai_epi32(u[6], DCT_CONST_BITS);
  u[7] = _mm_srai_epi32(u[7], DCT_CONST_BITS);
  u[8] = _mm_srai_epi32(u[8], DCT_CONST_BITS);
  u[9] = _mm_srai_epi32(u[9], DCT_CONST_BITS);
  u[10] = _mm_srai_epi32(u[10], DCT_CONST_BITS);
  u[11] = _mm_srai_epi32(u[11], DCT_CONST_BITS);
  u[12] = _mm_srai_epi32(u[12], DCT_CONST_BITS);
  u[13] = _mm_srai_epi32(u[13], DCT_CONST_BITS);
  u[14] = _mm_srai_epi32(u[14], DCT_CONST_BITS);
  u[15] = _mm_srai_epi32(u[15], DCT_CONST_BITS);

  s[0] = _mm_packs_epi32(u[0], u[1]);
  s[1] = _mm_packs_epi32(u[2], u[3]);
  s[2] = _mm_packs_epi32(u[4], u[5]);
  s[3] = _mm_packs_epi32(u[6], u[7]);
  s[4] = _mm_add_epi16(t[4], t[5]);
  s[5] = _mm_sub_epi16(t[4], t[5]);
  s[6] = _mm_sub_epi16(t[7], t[6]);
  s[7] = _mm_add_epi16(t[6], t[7]);
  s[8] = t[8];
  s[15] = t[15];
  s[9]  = _mm_packs_epi32(u[8], u[9]);
  s[14] = _mm_packs_epi32(u[10], u[11]);
  s[10] = _mm_packs_epi32(u[12], u[13]);
  s[13] = _mm_packs_epi32(u[14], u[15]);
  s[11] = t[11];
  s[12] = t[12];

  // stage 5
  t[0] = _mm_add_epi16(s[0], s[3]);
  t[1] = _mm_add_epi16(s[1], s[2]);
  t[2] = _mm_sub_epi16(s[1], s[2]);
  t[3] = _mm_sub_epi16(s[0], s[3]);
  t[4] = s[4];
  t[7] = s[7];

  u[0] = _mm_unpacklo_epi16(s[5], s[6]);
  u[1] = _mm_unpackhi_epi16(s[5], s[6]);
  v[0] = _mm_madd_epi16(u[0], k__cospi_m16_p16);
  v[1] = _mm_madd_epi16(u[1], k__cospi_m16_p16);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p16_p16);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p16_p16);
  u[0] = _mm_add_epi32(v[0], k__DCT_CONST_ROUNDING);
  u[1] = _mm_add_epi32(v[1], k__DCT_CONST_ROUNDING);
  u[2] = _mm_add_epi32(v[2], k__DCT_CONST_ROUNDING);
  u[3] = _mm_add_epi32(v[3], k__DCT_CONST_ROUNDING);
  u[0] = _mm_srai_epi32(u[0], DCT_CONST_BITS);
  u[1] = _mm_srai_epi32(u[1], DCT_CONST_BITS);
  u[2] = _mm_srai_epi32(u[2], DCT_CONST_BITS);
  u[3] = _mm_srai_epi32(u[3], DCT_CONST_BITS);
  t[5] = _mm_packs_epi32(u[0], u[1]);
  t[6] = _mm_packs_epi32(u[2], u[3]);

  t[8] = _mm_add_epi16(s[8], s[11]);
  t[9] = _mm_add_epi16(s[9], s[10]);
  t[10] = _mm_sub_epi16(s[9], s[10]);
  t[11] = _mm_sub_epi16(s[8], s[11]);
  t[12] = _mm_sub_epi16(s[15], s[12]);
  t[13] = _mm_sub_epi16(s[14], s[13]);
  t[14] = _mm_add_epi16(s[13], s[14]);
  t[15] = _mm_add_epi16(s[12], s[15]);

  // stage 6
  s[0] = _mm_add_epi16(t[0], t[7]);
  s[1] = _mm_add_epi16(t[1], t[6]);
  s[2] = _mm_add_epi16(t[2], t[5]);
  s[3] = _mm_add_epi16(t[3], t[4]);
  s[4] = _mm_sub_epi16(t[3], t[4]);
  s[5] = _mm_sub_epi16(t[2], t[5]);
  s[6] = _mm_sub_epi16(t[1], t[6]);
  s[7] = _mm_sub_epi16(t[0], t[7]);
  s[8] = t[8];
  s[9] = t[9];

  u[0] = _mm_unpacklo_epi16(t[10], t[13]);
  u[1] = _mm_unpackhi_epi16(t[10], t[13]);
  u[2] = _mm_unpacklo_epi16(t[11], t[12]);
  u[3] = _mm_unpackhi_epi16(t[11], t[12]);

  v[0] = _mm_madd_epi16(u[0], k__cospi_m16_p16);
  v[1] = _mm_madd_epi16(u[1], k__cospi_m16_p16);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p16_p16);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p16_p16);
  v[4] = _mm_madd_epi16(u[2], k__cospi_m16_p16);
  v[5] = _mm_madd_epi16(u[3], k__cospi_m16_p16);
  v[6] = _mm_madd_epi16(u[2], k__cospi_p16_p16);
  v[7] = _mm_madd_epi16(u[3], k__cospi_p16_p16);

  u[0] = _mm_add_epi32(v[0], k__DCT_CONST_ROUNDING);
  u[1] = _mm_add_epi32(v[1], k__DCT_CONST_ROUNDING);
  u[2] = _mm_add_epi32(v[2], k__DCT_CONST_ROUNDING);
  u[3] = _mm_add_epi32(v[3], k__DCT_CONST_ROUNDING);
  u[4] = _mm_add_epi32(v[4], k__DCT_CONST_ROUNDING);
  u[5] = _mm_add_epi32(v[5], k__DCT_CONST_ROUNDING);
  u[6] = _mm_add_epi32(v[6], k__DCT_CONST_ROUNDING);
  u[7] = _mm_add_epi32(v[7], k__DCT_CONST_ROUNDING);

  u[0] = _mm_srai_epi32(u[0], DCT_CONST_BITS);
  u[1] = _mm_srai_epi32(u[1], DCT_CONST_BITS);
  u[2] = _mm_srai_epi32(u[2], DCT_CONST_BITS);
  u[3] = _mm_srai_epi32(u[3], DCT_CONST_BITS);
  u[4] = _mm_srai_epi32(u[4], DCT_CONST_BITS);
  u[5] = _mm_srai_epi32(u[5], DCT_CONST_BITS);
  u[6] = _mm_srai_epi32(u[6], DCT_CONST_BITS);
  u[7] = _mm_srai_epi32(u[7], DCT_CONST_BITS);

  s[10] = _mm_packs_epi32(u[0], u[1]);
  s[13] = _mm_packs_epi32(u[2], u[3]);
  s[11] = _mm_packs_epi32(u[4], u[5]);
  s[12] = _mm_packs_epi32(u[6], u[7]);
  s[14] = t[14];
  s[15] = t[15];

  // stage 7
  in[0] = _mm_add_epi16(s[0], s[15]);
  in[1] = _mm_add_epi16(s[1], s[14]);
  in[2] = _mm_add_epi16(s[2], s[13]);
  in[3] = _mm_add_epi16(s[3], s[12]);
  in[4] = _mm_add_epi16(s[4], s[11]);
  in[5] = _mm_add_epi16(s[5], s[10]);
  in[6] = _mm_add_epi16(s[6], s[9]);
  in[7] = _mm_add_epi16(s[7], s[8]);
  in[8] = _mm_sub_epi16(s[7], s[8]);
  in[9] = _mm_sub_epi16(s[6], s[9]);
  in[10] = _mm_sub_epi16(s[5], s[10]);
  in[11] = _mm_sub_epi16(s[4], s[11]);
  in[12] = _mm_sub_epi16(s[3], s[12]);
  in[13] = _mm_sub_epi16(s[2], s[13]);
  in[14] = _mm_sub_epi16(s[1], s[14]);
  in[15] = _mm_sub_epi16(s[0], s[15]);
}

void idct16_sse2(__m128i *in0, __m128i *in1) {
  array_transpose_16x16(in0, in1);
  idct16_8col(in0);
  idct16_8col(in1);
}

void iadst16_sse2(__m128i *in0, __m128i *in1) {
  array_transpose_16x16(in0, in1);
  iadst16_8col(in0);
  iadst16_8col(in1);
}

void vpx_idct16x16_10_add_sse2(const tran_low_t *input, uint8_t *dest,
                               int stride) {
  const __m128i rounding = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i final_rounding = _mm_set1_epi16(1 << 5);
  const __m128i zero = _mm_setzero_si128();

  const __m128i stg2_0 = pair_set_epi16(cospi_30_64, -cospi_2_64);
  const __m128i stg2_1 = pair_set_epi16(cospi_2_64, cospi_30_64);
  const __m128i stg2_6 = pair_set_epi16(cospi_6_64, -cospi_26_64);
  const __m128i stg2_7 = pair_set_epi16(cospi_26_64, cospi_6_64);

  const __m128i stg3_0 = pair_set_epi16(cospi_28_64, -cospi_4_64);
  const __m128i stg3_1 = pair_set_epi16(cospi_4_64, cospi_28_64);

  const __m128i stg4_0 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i stg4_1 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i stg4_4 = pair_set_epi16(-cospi_8_64, cospi_24_64);
  const __m128i stg4_5 = pair_set_epi16(cospi_24_64, cospi_8_64);
  const __m128i stg4_6 = pair_set_epi16(-cospi_24_64, -cospi_8_64);
  const __m128i stg4_7 = pair_set_epi16(-cospi_8_64, cospi_24_64);

  const __m128i stg6_0 = pair_set_epi16(-cospi_16_64, cospi_16_64);
  __m128i in[16], l[16];
  __m128i stp1_0, stp1_1, stp1_2, stp1_3, stp1_4, stp1_5, stp1_6,
          stp1_8, stp1_9, stp1_10, stp1_11, stp1_12, stp1_13, stp1_14, stp1_15,
          stp1_8_0, stp1_12_0;
  __m128i stp2_0, stp2_1, stp2_2, stp2_3, stp2_4, stp2_5, stp2_6, stp2_7,
          stp2_8, stp2_9, stp2_10, stp2_11, stp2_12, stp2_13, stp2_14;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  int i;
  // First 1-D inverse DCT
  // Load input data.
  in[0] = load_input_data(input);
  in[1] = load_input_data(input + 8 * 2);
  in[2] = load_input_data(input + 8 * 4);
  in[3] = load_input_data(input + 8 * 6);

  TRANSPOSE_8X4(in[0], in[1], in[2], in[3], in[0], in[1]);

  // Stage2
  {
    const __m128i lo_1_15 = _mm_unpackhi_epi16(in[0], zero);
    const __m128i lo_13_3 = _mm_unpackhi_epi16(zero, in[1]);

    tmp0 = _mm_madd_epi16(lo_1_15, stg2_0);
    tmp2 = _mm_madd_epi16(lo_1_15, stg2_1);
    tmp5 = _mm_madd_epi16(lo_13_3, stg2_6);
    tmp7 = _mm_madd_epi16(lo_13_3, stg2_7);

    tmp0 = _mm_add_epi32(tmp0, rounding);
    tmp2 = _mm_add_epi32(tmp2, rounding);
    tmp5 = _mm_add_epi32(tmp5, rounding);
    tmp7 = _mm_add_epi32(tmp7, rounding);

    tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS);
    tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS);
    tmp5 = _mm_srai_epi32(tmp5, DCT_CONST_BITS);
    tmp7 = _mm_srai_epi32(tmp7, DCT_CONST_BITS);

    stp2_8  = _mm_packs_epi32(tmp0, tmp2);
    stp2_11 = _mm_packs_epi32(tmp5, tmp7);
  }

  // Stage3
  {
    const __m128i lo_2_14 = _mm_unpacklo_epi16(in[1], zero);

    tmp0 = _mm_madd_epi16(lo_2_14, stg3_0);
    tmp2 = _mm_madd_epi16(lo_2_14, stg3_1);

    tmp0 = _mm_add_epi32(tmp0, rounding);
    tmp2 = _mm_add_epi32(tmp2, rounding);
    tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS);
    tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS);

    stp1_13 = _mm_unpackhi_epi64(stp2_11, zero);
    stp1_14 = _mm_unpackhi_epi64(stp2_8, zero);

    stp1_4 = _mm_packs_epi32(tmp0, tmp2);
  }

  // Stage4
  {
    const __m128i lo_0_8 = _mm_unpacklo_epi16(in[0], zero);
    const __m128i lo_9_14 = _mm_unpacklo_epi16(stp2_8, stp1_14);
    const __m128i lo_10_13 = _mm_unpacklo_epi16(stp2_11, stp1_13);

    tmp0 = _mm_madd_epi16(lo_0_8, stg4_0);
    tmp2 = _mm_madd_epi16(lo_0_8, stg4_1);
    tmp1 = _mm_madd_epi16(lo_9_14, stg4_4);
    tmp3 = _mm_madd_epi16(lo_9_14, stg4_5);
    tmp5 = _mm_madd_epi16(lo_10_13, stg4_6);
    tmp7 = _mm_madd_epi16(lo_10_13, stg4_7);

    tmp0 = _mm_add_epi32(tmp0, rounding);
    tmp2 = _mm_add_epi32(tmp2, rounding);
    tmp1 = _mm_add_epi32(tmp1, rounding);
    tmp3 = _mm_add_epi32(tmp3, rounding);
    tmp5 = _mm_add_epi32(tmp5, rounding);
    tmp7 = _mm_add_epi32(tmp7, rounding);

    tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS);
    tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS);
    tmp1 = _mm_srai_epi32(tmp1, DCT_CONST_BITS);
    tmp3 = _mm_srai_epi32(tmp3, DCT_CONST_BITS);
    tmp5 = _mm_srai_epi32(tmp5, DCT_CONST_BITS);
    tmp7 = _mm_srai_epi32(tmp7, DCT_CONST_BITS);

    stp1_0 = _mm_packs_epi32(tmp0, tmp0);
    stp1_1 = _mm_packs_epi32(tmp2, tmp2);
    stp2_9 = _mm_packs_epi32(tmp1, tmp3);
    stp2_10 = _mm_packs_epi32(tmp5, tmp7);

    stp2_6 = _mm_unpackhi_epi64(stp1_4, zero);
  }

  // Stage5 and Stage6
  {
    tmp0 = _mm_add_epi16(stp2_8, stp2_11);
    tmp1 = _mm_sub_epi16(stp2_8, stp2_11);
    tmp2 = _mm_add_epi16(stp2_9, stp2_10);
    tmp3 = _mm_sub_epi16(stp2_9, stp2_10);

    stp1_9  = _mm_unpacklo_epi64(tmp2, zero);
    stp1_10 = _mm_unpacklo_epi64(tmp3, zero);
    stp1_8  = _mm_unpacklo_epi64(tmp0, zero);
    stp1_11 = _mm_unpacklo_epi64(tmp1, zero);

    stp1_13 = _mm_unpackhi_epi64(tmp3, zero);
    stp1_14 = _mm_unpackhi_epi64(tmp2, zero);
    stp1_12 = _mm_unpackhi_epi64(tmp1, zero);
    stp1_15 = _mm_unpackhi_epi64(tmp0, zero);
  }

  // Stage6
  {
    const __m128i lo_6_5 = _mm_unpacklo_epi16(stp2_6, stp1_4);
    const __m128i lo_10_13 = _mm_unpacklo_epi16(stp1_10, stp1_13);
    const __m128i lo_11_12 = _mm_unpacklo_epi16(stp1_11, stp1_12);

    tmp1 = _mm_madd_epi16(lo_6_5, stg4_1);
    tmp3 = _mm_madd_epi16(lo_6_5, stg4_0);
    tmp0 = _mm_madd_epi16(lo_10_13, stg6_0);
    tmp2 = _mm_madd_epi16(lo_10_13, stg4_0);
    tmp4 = _mm_madd_epi16(lo_11_12, stg6_0);
    tmp6 = _mm_madd_epi16(lo_11_12, stg4_0);

    tmp1 = _mm_add_epi32(tmp1, rounding);
    tmp3 = _mm_add_epi32(tmp3, rounding);
    tmp0 = _mm_add_epi32(tmp0, rounding);
    tmp2 = _mm_add_epi32(tmp2, rounding);
    tmp4 = _mm_add_epi32(tmp4, rounding);
    tmp6 = _mm_add_epi32(tmp6, rounding);

    tmp1 = _mm_srai_epi32(tmp1, DCT_CONST_BITS);
    tmp3 = _mm_srai_epi32(tmp3, DCT_CONST_BITS);
    tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS);
    tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS);
    tmp4 = _mm_srai_epi32(tmp4, DCT_CONST_BITS);
    tmp6 = _mm_srai_epi32(tmp6, DCT_CONST_BITS);

    stp1_6 = _mm_packs_epi32(tmp3, tmp1);

    stp2_10 = _mm_packs_epi32(tmp0, zero);
    stp2_13 = _mm_packs_epi32(tmp2, zero);
    stp2_11 = _mm_packs_epi32(tmp4, zero);
    stp2_12 = _mm_packs_epi32(tmp6, zero);

    tmp0 = _mm_add_epi16(stp1_0, stp1_4);
    tmp1 = _mm_sub_epi16(stp1_0, stp1_4);
    tmp2 = _mm_add_epi16(stp1_1, stp1_6);
    tmp3 = _mm_sub_epi16(stp1_1, stp1_6);

    stp2_0 = _mm_unpackhi_epi64(tmp0, zero);
    stp2_1 = _mm_unpacklo_epi64(tmp2, zero);
    stp2_2 = _mm_unpackhi_epi64(tmp2, zero);
    stp2_3 = _mm_unpacklo_epi64(tmp0, zero);
    stp2_4 = _mm_unpacklo_epi64(tmp1, zero);
    stp2_5 = _mm_unpackhi_epi64(tmp3, zero);
    stp2_6 = _mm_unpacklo_epi64(tmp3, zero);
    stp2_7 = _mm_unpackhi_epi64(tmp1, zero);
  }

  // Stage7. Left 8x16 only.
  l[0] = _mm_add_epi16(stp2_0, stp1_15);
  l[1] = _mm_add_epi16(stp2_1, stp1_14);
  l[2] = _mm_add_epi16(stp2_2, stp2_13);
  l[3] = _mm_add_epi16(stp2_3, stp2_12);
  l[4] = _mm_add_epi16(stp2_4, stp2_11);
  l[5] = _mm_add_epi16(stp2_5, stp2_10);
  l[6] = _mm_add_epi16(stp2_6, stp1_9);
  l[7] = _mm_add_epi16(stp2_7, stp1_8);
  l[8] = _mm_sub_epi16(stp2_7, stp1_8);
  l[9] = _mm_sub_epi16(stp2_6, stp1_9);
  l[10] = _mm_sub_epi16(stp2_5, stp2_10);
  l[11] = _mm_sub_epi16(stp2_4, stp2_11);
  l[12] = _mm_sub_epi16(stp2_3, stp2_12);
  l[13] = _mm_sub_epi16(stp2_2, stp2_13);
  l[14] = _mm_sub_epi16(stp2_1, stp1_14);
  l[15] = _mm_sub_epi16(stp2_0, stp1_15);

  // Second 1-D inverse transform, performed per 8x16 block
  for (i = 0; i < 2; i++) {
    int j;
    array_transpose_4X8(l + 8 * i, in);

    IDCT16_10

    // Stage7
    in[0] = _mm_add_epi16(stp2_0, stp1_15);
    in[1] = _mm_add_epi16(stp2_1, stp1_14);
    in[2] = _mm_add_epi16(stp2_2, stp2_13);
    in[3] = _mm_add_epi16(stp2_3, stp2_12);
    in[4] = _mm_add_epi16(stp2_4, stp2_11);
    in[5] = _mm_add_epi16(stp2_5, stp2_10);
    in[6] = _mm_add_epi16(stp2_6, stp1_9);
    in[7] = _mm_add_epi16(stp2_7, stp1_8);
    in[8] = _mm_sub_epi16(stp2_7, stp1_8);
    in[9] = _mm_sub_epi16(stp2_6, stp1_9);
    in[10] = _mm_sub_epi16(stp2_5, stp2_10);
    in[11] = _mm_sub_epi16(stp2_4, stp2_11);
    in[12] = _mm_sub_epi16(stp2_3, stp2_12);
    in[13] = _mm_sub_epi16(stp2_2, stp2_13);
    in[14] = _mm_sub_epi16(stp2_1, stp1_14);
    in[15] = _mm_sub_epi16(stp2_0, stp1_15);

    for (j = 0; j < 16; ++j) {
      // Final rounding and shift
      in[j] = _mm_adds_epi16(in[j], final_rounding);
      in[j] = _mm_srai_epi16(in[j], 6);
      RECON_AND_STORE(dest + j * stride, in[j]);
    }

    dest += 8;
  }
}

#define LOAD_DQCOEFF(reg, input) \
  {  \
    reg = load_input_data(input); \
    input += 8; \
  }  \

#define IDCT32_34 \
/* Stage1 */ \
{ \
  const __m128i zero = _mm_setzero_si128();\
  const __m128i lo_1_31 = _mm_unpacklo_epi16(in[1], zero); \
  const __m128i hi_1_31 = _mm_unpackhi_epi16(in[1], zero); \
  \
  const __m128i lo_25_7= _mm_unpacklo_epi16(zero, in[7]); \
  const __m128i hi_25_7 = _mm_unpackhi_epi16(zero, in[7]); \
  \
  const __m128i lo_5_27 = _mm_unpacklo_epi16(in[5], zero); \
  const __m128i hi_5_27 = _mm_unpackhi_epi16(in[5], zero); \
  \
  const __m128i lo_29_3 = _mm_unpacklo_epi16(zero, in[3]); \
  const __m128i hi_29_3 = _mm_unpackhi_epi16(zero, in[3]); \
  \
  MULTIPLICATION_AND_ADD_2(lo_1_31, hi_1_31, stg1_0, \
                         stg1_1, stp1_16, stp1_31); \
  MULTIPLICATION_AND_ADD_2(lo_25_7, hi_25_7, stg1_6, \
                         stg1_7, stp1_19, stp1_28); \
  MULTIPLICATION_AND_ADD_2(lo_5_27, hi_5_27, stg1_8, \
                         stg1_9, stp1_20, stp1_27); \
  MULTIPLICATION_AND_ADD_2(lo_29_3, hi_29_3, stg1_14, \
                         stg1_15, stp1_23, stp1_24); \
} \
\
/* Stage2 */ \
{ \
  const __m128i zero = _mm_setzero_si128();\
  const __m128i lo_2_30 = _mm_unpacklo_epi16(in[2], zero); \
  const __m128i hi_2_30 = _mm_unpackhi_epi16(in[2], zero); \
  \
  const __m128i lo_26_6 = _mm_unpacklo_epi16(zero, in[6]); \
  const __m128i hi_26_6 = _mm_unpackhi_epi16(zero, in[6]); \
  \
  MULTIPLICATION_AND_ADD_2(lo_2_30, hi_2_30, stg2_0, \
                         stg2_1, stp2_8, stp2_15); \
  MULTIPLICATION_AND_ADD_2(lo_26_6, hi_26_6, stg2_6, \
                         stg2_7, stp2_11, stp2_12); \
  \
  stp2_16 = stp1_16; \
  stp2_19 = stp1_19; \
  \
  stp2_20 = stp1_20; \
  stp2_23 = stp1_23; \
  \
  stp2_24 = stp1_24; \
  stp2_27 = stp1_27; \
  \
  stp2_28 = stp1_28; \
  stp2_31 = stp1_31; \
} \
\
/* Stage3 */ \
{ \
  const __m128i zero = _mm_setzero_si128();\
  const __m128i lo_4_28 = _mm_unpacklo_epi16(in[4], zero); \
  const __m128i hi_4_28 = _mm_unpackhi_epi16(in[4], zero); \
  \
  const __m128i lo_17_30 = _mm_unpacklo_epi16(stp1_16, stp1_31); \
  const __m128i hi_17_30 = _mm_unpackhi_epi16(stp1_16, stp1_31); \
  const __m128i lo_18_29 = _mm_unpacklo_epi16(stp1_19, stp1_28); \
  const __m128i hi_18_29 = _mm_unpackhi_epi16(stp1_19, stp1_28); \
  \
  const __m128i lo_21_26 = _mm_unpacklo_epi16(stp1_20, stp1_27); \
  const __m128i hi_21_26 = _mm_unpackhi_epi16(stp1_20, stp1_27); \
  const __m128i lo_22_25 = _mm_unpacklo_epi16(stp1_23, stp1_24); \
  const __m128i hi_22_25 = _mm_unpackhi_epi16(stp1_23, stp2_24); \
  \
  MULTIPLICATION_AND_ADD_2(lo_4_28, hi_4_28, stg3_0, \
                         stg3_1, stp1_4, stp1_7); \
  \
  stp1_8 = stp2_8; \
  stp1_11 = stp2_11; \
  stp1_12 = stp2_12; \
  stp1_15 = stp2_15; \
  \
  MULTIPLICATION_AND_ADD(lo_17_30, hi_17_30, lo_18_29, hi_18_29, stg3_4, \
                         stg3_5, stg3_6, stg3_4, stp1_17, stp1_30, \
                         stp1_18, stp1_29) \
  MULTIPLICATION_AND_ADD(lo_21_26, hi_21_26, lo_22_25, hi_22_25, stg3_8, \
                         stg3_9, stg3_10, stg3_8, stp1_21, stp1_26, \
                         stp1_22, stp1_25) \
  \
  stp1_16 = stp2_16; \
  stp1_31 = stp2_31; \
  stp1_19 = stp2_19; \
  stp1_20 = stp2_20; \
  stp1_23 = stp2_23; \
  stp1_24 = stp2_24; \
  stp1_27 = stp2_27; \
  stp1_28 = stp2_28; \
} \
\
/* Stage4 */ \
{ \
  const __m128i zero = _mm_setzero_si128();\
  const __m128i lo_0_16 = _mm_unpacklo_epi16(in[0], zero); \
  const __m128i hi_0_16 = _mm_unpackhi_epi16(in[0], zero); \
  \
  const __m128i lo_9_14 = _mm_unpacklo_epi16(stp2_8, stp2_15); \
  const __m128i hi_9_14 = _mm_unpackhi_epi16(stp2_8, stp2_15); \
  const __m128i lo_10_13 = _mm_unpacklo_epi16(stp2_11, stp2_12); \
  const __m128i hi_10_13 = _mm_unpackhi_epi16(stp2_11, stp2_12); \
  \
  MULTIPLICATION_AND_ADD_2(lo_0_16, hi_0_16, stg4_0, \
                         stg4_1, stp2_0, stp2_1); \
  \
  stp2_4 = stp1_4; \
  stp2_5 = stp1_4; \
  stp2_6 = stp1_7; \
  stp2_7 = stp1_7; \
  \
  MULTIPLICATION_AND_ADD(lo_9_14, hi_9_14, lo_10_13, hi_10_13, stg4_4, \
                         stg4_5, stg4_6, stg4_4, stp2_9, stp2_14, \
                         stp2_10, stp2_13) \
  \
  stp2_8 = stp1_8; \
  stp2_15 = stp1_15; \
  stp2_11 = stp1_11; \
  stp2_12 = stp1_12; \
  \
  stp2_16 = _mm_add_epi16(stp1_16, stp1_19); \
  stp2_17 = _mm_add_epi16(stp1_17, stp1_18); \
  stp2_18 = _mm_sub_epi16(stp1_17, stp1_18); \
  stp2_19 = _mm_sub_epi16(stp1_16, stp1_19); \
  stp2_20 = _mm_sub_epi16(stp1_23, stp1_20); \
  stp2_21 = _mm_sub_epi16(stp1_22, stp1_21); \
  stp2_22 = _mm_add_epi16(stp1_22, stp1_21); \
  stp2_23 = _mm_add_epi16(stp1_23, stp1_20); \
  \
  stp2_24 = _mm_add_epi16(stp1_24, stp1_27); \
  stp2_25 = _mm_add_epi16(stp1_25, stp1_26); \
  stp2_26 = _mm_sub_epi16(stp1_25, stp1_26); \
  stp2_27 = _mm_sub_epi16(stp1_24, stp1_27); \
  stp2_28 = _mm_sub_epi16(stp1_31, stp1_28); \
  stp2_29 = _mm_sub_epi16(stp1_30, stp1_29); \
  stp2_30 = _mm_add_epi16(stp1_29, stp1_30); \
  stp2_31 = _mm_add_epi16(stp1_28, stp1_31); \
} \
\
/* Stage5 */ \
{ \
  const __m128i lo_6_5 = _mm_unpacklo_epi16(stp2_6, stp2_5); \
  const __m128i hi_6_5 = _mm_unpackhi_epi16(stp2_6, stp2_5); \
  const __m128i lo_18_29 = _mm_unpacklo_epi16(stp2_18, stp2_29); \
  const __m128i hi_18_29 = _mm_unpackhi_epi16(stp2_18, stp2_29); \
  \
  const __m128i lo_19_28 = _mm_unpacklo_epi16(stp2_19, stp2_28); \
  const __m128i hi_19_28 = _mm_unpackhi_epi16(stp2_19, stp2_28); \
  const __m128i lo_20_27 = _mm_unpacklo_epi16(stp2_20, stp2_27); \
  const __m128i hi_20_27 = _mm_unpackhi_epi16(stp2_20, stp2_27); \
  \
  const __m128i lo_21_26 = _mm_unpacklo_epi16(stp2_21, stp2_26); \
  const __m128i hi_21_26 = _mm_unpackhi_epi16(stp2_21, stp2_26); \
  \
  stp1_0 = stp2_0; \
  stp1_1 = stp2_1; \
  stp1_2 = stp2_1; \
  stp1_3 = stp2_0; \
  \
  tmp0 = _mm_madd_epi16(lo_6_5, stg4_1); \
  tmp1 = _mm_madd_epi16(hi_6_5, stg4_1); \
  tmp2 = _mm_madd_epi16(lo_6_5, stg4_0); \
  tmp3 = _mm_madd_epi16(hi_6_5, stg4_0); \
  \
  tmp0 = _mm_add_epi32(tmp0, rounding); \
  tmp1 = _mm_add_epi32(tmp1, rounding); \
  tmp2 = _mm_add_epi32(tmp2, rounding); \
  tmp3 = _mm_add_epi32(tmp3, rounding); \
  \
  tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS); \
  tmp1 = _mm_srai_epi32(tmp1, DCT_CONST_BITS); \
  tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS); \
  tmp3 = _mm_srai_epi32(tmp3, DCT_CONST_BITS); \
  \
  stp1_5 = _mm_packs_epi32(tmp0, tmp1); \
  stp1_6 = _mm_packs_epi32(tmp2, tmp3); \
  \
  stp1_4 = stp2_4; \
  stp1_7 = stp2_7; \
  \
  stp1_8 = _mm_add_epi16(stp2_8, stp2_11); \
  stp1_9 = _mm_add_epi16(stp2_9, stp2_10); \
  stp1_10 = _mm_sub_epi16(stp2_9, stp2_10); \
  stp1_11 = _mm_sub_epi16(stp2_8, stp2_11); \
  stp1_12 = _mm_sub_epi16(stp2_15, stp2_12); \
  stp1_13 = _mm_sub_epi16(stp2_14, stp2_13); \
  stp1_14 = _mm_add_epi16(stp2_14, stp2_13); \
  stp1_15 = _mm_add_epi16(stp2_15, stp2_12); \
  \
  stp1_16 = stp2_16; \
  stp1_17 = stp2_17; \
  \
  MULTIPLICATION_AND_ADD(lo_18_29, hi_18_29, lo_19_28, hi_19_28, stg4_4, \
                         stg4_5, stg4_4, stg4_5, stp1_18, stp1_29, \
                         stp1_19, stp1_28) \
  MULTIPLICATION_AND_ADD(lo_20_27, hi_20_27, lo_21_26, hi_21_26, stg4_6, \
                         stg4_4, stg4_6, stg4_4, stp1_20, stp1_27, \
                         stp1_21, stp1_26) \
  \
  stp1_22 = stp2_22; \
  stp1_23 = stp2_23; \
  stp1_24 = stp2_24; \
  stp1_25 = stp2_25; \
  stp1_30 = stp2_30; \
  stp1_31 = stp2_31; \
} \
\
/* Stage6 */ \
{ \
  const __m128i lo_10_13 = _mm_unpacklo_epi16(stp1_10, stp1_13); \
  const __m128i hi_10_13 = _mm_unpackhi_epi16(stp1_10, stp1_13); \
  const __m128i lo_11_12 = _mm_unpacklo_epi16(stp1_11, stp1_12); \
  const __m128i hi_11_12 = _mm_unpackhi_epi16(stp1_11, stp1_12); \
  \
  stp2_0 = _mm_add_epi16(stp1_0, stp1_7); \
  stp2_1 = _mm_add_epi16(stp1_1, stp1_6); \
  stp2_2 = _mm_add_epi16(stp1_2, stp1_5); \
  stp2_3 = _mm_add_epi16(stp1_3, stp1_4); \
  stp2_4 = _mm_sub_epi16(stp1_3, stp1_4); \
  stp2_5 = _mm_sub_epi16(stp1_2, stp1_5); \
  stp2_6 = _mm_sub_epi16(stp1_1, stp1_6); \
  stp2_7 = _mm_sub_epi16(stp1_0, stp1_7); \
  \
  stp2_8 = stp1_8; \
  stp2_9 = stp1_9; \
  stp2_14 = stp1_14; \
  stp2_15 = stp1_15; \
  \
  MULTIPLICATION_AND_ADD(lo_10_13, hi_10_13, lo_11_12, hi_11_12, \
                         stg6_0, stg4_0, stg6_0, stg4_0, stp2_10, \
                         stp2_13, stp2_11, stp2_12) \
  \
  stp2_16 = _mm_add_epi16(stp1_16, stp1_23); \
  stp2_17 = _mm_add_epi16(stp1_17, stp1_22); \
  stp2_18 = _mm_add_epi16(stp1_18, stp1_21); \
  stp2_19 = _mm_add_epi16(stp1_19, stp1_20); \
  stp2_20 = _mm_sub_epi16(stp1_19, stp1_20); \
  stp2_21 = _mm_sub_epi16(stp1_18, stp1_21); \
  stp2_22 = _mm_sub_epi16(stp1_17, stp1_22); \
  stp2_23 = _mm_sub_epi16(stp1_16, stp1_23); \
  \
  stp2_24 = _mm_sub_epi16(stp1_31, stp1_24); \
  stp2_25 = _mm_sub_epi16(stp1_30, stp1_25); \
  stp2_26 = _mm_sub_epi16(stp1_29, stp1_26); \
  stp2_27 = _mm_sub_epi16(stp1_28, stp1_27); \
  stp2_28 = _mm_add_epi16(stp1_27, stp1_28); \
  stp2_29 = _mm_add_epi16(stp1_26, stp1_29); \
  stp2_30 = _mm_add_epi16(stp1_25, stp1_30); \
  stp2_31 = _mm_add_epi16(stp1_24, stp1_31); \
} \
\
/* Stage7 */ \
{ \
  const __m128i lo_20_27 = _mm_unpacklo_epi16(stp2_20, stp2_27); \
  const __m128i hi_20_27 = _mm_unpackhi_epi16(stp2_20, stp2_27); \
  const __m128i lo_21_26 = _mm_unpacklo_epi16(stp2_21, stp2_26); \
  const __m128i hi_21_26 = _mm_unpackhi_epi16(stp2_21, stp2_26); \
  \
  const __m128i lo_22_25 = _mm_unpacklo_epi16(stp2_22, stp2_25); \
  const __m128i hi_22_25 = _mm_unpackhi_epi16(stp2_22, stp2_25); \
  const __m128i lo_23_24 = _mm_unpacklo_epi16(stp2_23, stp2_24); \
  const __m128i hi_23_24 = _mm_unpackhi_epi16(stp2_23, stp2_24); \
  \
  stp1_0 = _mm_add_epi16(stp2_0, stp2_15); \
  stp1_1 = _mm_add_epi16(stp2_1, stp2_14); \
  stp1_2 = _mm_add_epi16(stp2_2, stp2_13); \
  stp1_3 = _mm_add_epi16(stp2_3, stp2_12); \
  stp1_4 = _mm_add_epi16(stp2_4, stp2_11); \
  stp1_5 = _mm_add_epi16(stp2_5, stp2_10); \
  stp1_6 = _mm_add_epi16(stp2_6, stp2_9); \
  stp1_7 = _mm_add_epi16(stp2_7, stp2_8); \
  stp1_8 = _mm_sub_epi16(stp2_7, stp2_8); \
  stp1_9 = _mm_sub_epi16(stp2_6, stp2_9); \
  stp1_10 = _mm_sub_epi16(stp2_5, stp2_10); \
  stp1_11 = _mm_sub_epi16(stp2_4, stp2_11); \
  stp1_12 = _mm_sub_epi16(stp2_3, stp2_12); \
  stp1_13 = _mm_sub_epi16(stp2_2, stp2_13); \
  stp1_14 = _mm_sub_epi16(stp2_1, stp2_14); \
  stp1_15 = _mm_sub_epi16(stp2_0, stp2_15); \
  \
  stp1_16 = stp2_16; \
  stp1_17 = stp2_17; \
  stp1_18 = stp2_18; \
  stp1_19 = stp2_19; \
  \
  MULTIPLICATION_AND_ADD(lo_20_27, hi_20_27, lo_21_26, hi_21_26, stg6_0, \
                         stg4_0, stg6_0, stg4_0, stp1_20, stp1_27, \
                         stp1_21, stp1_26) \
  MULTIPLICATION_AND_ADD(lo_22_25, hi_22_25, lo_23_24, hi_23_24, stg6_0, \
                         stg4_0, stg6_0, stg4_0, stp1_22, stp1_25, \
                         stp1_23, stp1_24) \
  \
  stp1_28 = stp2_28; \
  stp1_29 = stp2_29; \
  stp1_30 = stp2_30; \
  stp1_31 = stp2_31; \
}


#define IDCT32 \
/* Stage1 */ \
{ \
  const __m128i lo_1_31 = _mm_unpacklo_epi16(in[1], in[31]); \
  const __m128i hi_1_31 = _mm_unpackhi_epi16(in[1], in[31]); \
  const __m128i lo_17_15 = _mm_unpacklo_epi16(in[17], in[15]); \
  const __m128i hi_17_15 = _mm_unpackhi_epi16(in[17], in[15]); \
  \
  const __m128i lo_9_23 = _mm_unpacklo_epi16(in[9], in[23]); \
  const __m128i hi_9_23 = _mm_unpackhi_epi16(in[9], in[23]); \
  const __m128i lo_25_7= _mm_unpacklo_epi16(in[25], in[7]); \
  const __m128i hi_25_7 = _mm_unpackhi_epi16(in[25], in[7]); \
  \
  const __m128i lo_5_27 = _mm_unpacklo_epi16(in[5], in[27]); \
  const __m128i hi_5_27 = _mm_unpackhi_epi16(in[5], in[27]); \
  const __m128i lo_21_11 = _mm_unpacklo_epi16(in[21], in[11]); \
  const __m128i hi_21_11 = _mm_unpackhi_epi16(in[21], in[11]); \
  \
  const __m128i lo_13_19 = _mm_unpacklo_epi16(in[13], in[19]); \
  const __m128i hi_13_19 = _mm_unpackhi_epi16(in[13], in[19]); \
  const __m128i lo_29_3 = _mm_unpacklo_epi16(in[29], in[3]); \
  const __m128i hi_29_3 = _mm_unpackhi_epi16(in[29], in[3]); \
  \
  MULTIPLICATION_AND_ADD(lo_1_31, hi_1_31, lo_17_15, hi_17_15, stg1_0, \
                         stg1_1, stg1_2, stg1_3, stp1_16, stp1_31, \
                         stp1_17, stp1_30) \
  MULTIPLICATION_AND_ADD(lo_9_23, hi_9_23, lo_25_7, hi_25_7, stg1_4, \
                         stg1_5, stg1_6, stg1_7, stp1_18, stp1_29, \
                         stp1_19, stp1_28) \
  MULTIPLICATION_AND_ADD(lo_5_27, hi_5_27, lo_21_11, hi_21_11, stg1_8, \
                         stg1_9, stg1_10, stg1_11, stp1_20, stp1_27, \
                         stp1_21, stp1_26) \
  MULTIPLICATION_AND_ADD(lo_13_19, hi_13_19, lo_29_3, hi_29_3, stg1_12, \
                         stg1_13, stg1_14, stg1_15, stp1_22, stp1_25, \
                         stp1_23, stp1_24) \
} \
\
/* Stage2 */ \
{ \
  const __m128i lo_2_30 = _mm_unpacklo_epi16(in[2], in[30]); \
  const __m128i hi_2_30 = _mm_unpackhi_epi16(in[2], in[30]); \
  const __m128i lo_18_14 = _mm_unpacklo_epi16(in[18], in[14]); \
  const __m128i hi_18_14 = _mm_unpackhi_epi16(in[18], in[14]); \
  \
  const __m128i lo_10_22 = _mm_unpacklo_epi16(in[10], in[22]); \
  const __m128i hi_10_22 = _mm_unpackhi_epi16(in[10], in[22]); \
  const __m128i lo_26_6 = _mm_unpacklo_epi16(in[26], in[6]); \
  const __m128i hi_26_6 = _mm_unpackhi_epi16(in[26], in[6]); \
  \
  MULTIPLICATION_AND_ADD(lo_2_30, hi_2_30, lo_18_14, hi_18_14, stg2_0, \
                         stg2_1, stg2_2, stg2_3, stp2_8, stp2_15, stp2_9, \
                         stp2_14) \
  MULTIPLICATION_AND_ADD(lo_10_22, hi_10_22, lo_26_6, hi_26_6, stg2_4, \
                         stg2_5, stg2_6, stg2_7, stp2_10, stp2_13, \
                         stp2_11, stp2_12) \
  \
  stp2_16 = _mm_add_epi16(stp1_16, stp1_17); \
  stp2_17 = _mm_sub_epi16(stp1_16, stp1_17); \
  stp2_18 = _mm_sub_epi16(stp1_19, stp1_18); \
  stp2_19 = _mm_add_epi16(stp1_19, stp1_18); \
  \
  stp2_20 = _mm_add_epi16(stp1_20, stp1_21); \
  stp2_21 = _mm_sub_epi16(stp1_20, stp1_21); \
  stp2_22 = _mm_sub_epi16(stp1_23, stp1_22); \
  stp2_23 = _mm_add_epi16(stp1_23, stp1_22); \
  \
  stp2_24 = _mm_add_epi16(stp1_24, stp1_25); \
  stp2_25 = _mm_sub_epi16(stp1_24, stp1_25); \
  stp2_26 = _mm_sub_epi16(stp1_27, stp1_26); \
  stp2_27 = _mm_add_epi16(stp1_27, stp1_26); \
  \
  stp2_28 = _mm_add_epi16(stp1_28, stp1_29); \
  stp2_29 = _mm_sub_epi16(stp1_28, stp1_29); \
  stp2_30 = _mm_sub_epi16(stp1_31, stp1_30); \
  stp2_31 = _mm_add_epi16(stp1_31, stp1_30); \
} \
\
/* Stage3 */ \
{ \
  const __m128i lo_4_28 = _mm_unpacklo_epi16(in[4], in[28]); \
  const __m128i hi_4_28 = _mm_unpackhi_epi16(in[4], in[28]); \
  const __m128i lo_20_12 = _mm_unpacklo_epi16(in[20], in[12]); \
  const __m128i hi_20_12 = _mm_unpackhi_epi16(in[20], in[12]); \
  \
  const __m128i lo_17_30 = _mm_unpacklo_epi16(stp2_17, stp2_30); \
  const __m128i hi_17_30 = _mm_unpackhi_epi16(stp2_17, stp2_30); \
  const __m128i lo_18_29 = _mm_unpacklo_epi16(stp2_18, stp2_29); \
  const __m128i hi_18_29 = _mm_unpackhi_epi16(stp2_18, stp2_29); \
  \
  const __m128i lo_21_26 = _mm_unpacklo_epi16(stp2_21, stp2_26); \
  const __m128i hi_21_26 = _mm_unpackhi_epi16(stp2_21, stp2_26); \
  const __m128i lo_22_25 = _mm_unpacklo_epi16(stp2_22, stp2_25); \
  const __m128i hi_22_25 = _mm_unpackhi_epi16(stp2_22, stp2_25); \
  \
  MULTIPLICATION_AND_ADD(lo_4_28, hi_4_28, lo_20_12, hi_20_12, stg3_0, \
                         stg3_1, stg3_2, stg3_3, stp1_4, stp1_7, stp1_5, \
                         stp1_6) \
  \
  stp1_8 = _mm_add_epi16(stp2_8, stp2_9); \
  stp1_9 = _mm_sub_epi16(stp2_8, stp2_9); \
  stp1_10 = _mm_sub_epi16(stp2_11, stp2_10); \
  stp1_11 = _mm_add_epi16(stp2_11, stp2_10); \
  stp1_12 = _mm_add_epi16(stp2_12, stp2_13); \
  stp1_13 = _mm_sub_epi16(stp2_12, stp2_13); \
  stp1_14 = _mm_sub_epi16(stp2_15, stp2_14); \
  stp1_15 = _mm_add_epi16(stp2_15, stp2_14); \
  \
  MULTIPLICATION_AND_ADD(lo_17_30, hi_17_30, lo_18_29, hi_18_29, stg3_4, \
                         stg3_5, stg3_6, stg3_4, stp1_17, stp1_30, \
                         stp1_18, stp1_29) \
  MULTIPLICATION_AND_ADD(lo_21_26, hi_21_26, lo_22_25, hi_22_25, stg3_8, \
                         stg3_9, stg3_10, stg3_8, stp1_21, stp1_26, \
                         stp1_22, stp1_25) \
  \
  stp1_16 = stp2_16; \
  stp1_31 = stp2_31; \
  stp1_19 = stp2_19; \
  stp1_20 = stp2_20; \
  stp1_23 = stp2_23; \
  stp1_24 = stp2_24; \
  stp1_27 = stp2_27; \
  stp1_28 = stp2_28; \
} \
\
/* Stage4 */ \
{ \
  const __m128i lo_0_16 = _mm_unpacklo_epi16(in[0], in[16]); \
  const __m128i hi_0_16 = _mm_unpackhi_epi16(in[0], in[16]); \
  const __m128i lo_8_24 = _mm_unpacklo_epi16(in[8], in[24]); \
  const __m128i hi_8_24 = _mm_unpackhi_epi16(in[8], in[24]); \
  \
  const __m128i lo_9_14 = _mm_unpacklo_epi16(stp1_9, stp1_14); \
  const __m128i hi_9_14 = _mm_unpackhi_epi16(stp1_9, stp1_14); \
  const __m128i lo_10_13 = _mm_unpacklo_epi16(stp1_10, stp1_13); \
  const __m128i hi_10_13 = _mm_unpackhi_epi16(stp1_10, stp1_13); \
  \
  MULTIPLICATION_AND_ADD(lo_0_16, hi_0_16, lo_8_24, hi_8_24, stg4_0, \
                         stg4_1, stg4_2, stg4_3, stp2_0, stp2_1, \
                         stp2_2, stp2_3) \
  \
  stp2_4 = _mm_add_epi16(stp1_4, stp1_5); \
  stp2_5 = _mm_sub_epi16(stp1_4, stp1_5); \
  stp2_6 = _mm_sub_epi16(stp1_7, stp1_6); \
  stp2_7 = _mm_add_epi16(stp1_7, stp1_6); \
  \
  MULTIPLICATION_AND_ADD(lo_9_14, hi_9_14, lo_10_13, hi_10_13, stg4_4, \
                         stg4_5, stg4_6, stg4_4, stp2_9, stp2_14, \
                         stp2_10, stp2_13) \
  \
  stp2_8 = stp1_8; \
  stp2_15 = stp1_15; \
  stp2_11 = stp1_11; \
  stp2_12 = stp1_12; \
  \
  stp2_16 = _mm_add_epi16(stp1_16, stp1_19); \
  stp2_17 = _mm_add_epi16(stp1_17, stp1_18); \
  stp2_18 = _mm_sub_epi16(stp1_17, stp1_18); \
  stp2_19 = _mm_sub_epi16(stp1_16, stp1_19); \
  stp2_20 = _mm_sub_epi16(stp1_23, stp1_20); \
  stp2_21 = _mm_sub_epi16(stp1_22, stp1_21); \
  stp2_22 = _mm_add_epi16(stp1_22, stp1_21); \
  stp2_23 = _mm_add_epi16(stp1_23, stp1_20); \
  \
  stp2_24 = _mm_add_epi16(stp1_24, stp1_27); \
  stp2_25 = _mm_add_epi16(stp1_25, stp1_26); \
  stp2_26 = _mm_sub_epi16(stp1_25, stp1_26); \
  stp2_27 = _mm_sub_epi16(stp1_24, stp1_27); \
  stp2_28 = _mm_sub_epi16(stp1_31, stp1_28); \
  stp2_29 = _mm_sub_epi16(stp1_30, stp1_29); \
  stp2_30 = _mm_add_epi16(stp1_29, stp1_30); \
  stp2_31 = _mm_add_epi16(stp1_28, stp1_31); \
} \
\
/* Stage5 */ \
{ \
  const __m128i lo_6_5 = _mm_unpacklo_epi16(stp2_6, stp2_5); \
  const __m128i hi_6_5 = _mm_unpackhi_epi16(stp2_6, stp2_5); \
  const __m128i lo_18_29 = _mm_unpacklo_epi16(stp2_18, stp2_29); \
  const __m128i hi_18_29 = _mm_unpackhi_epi16(stp2_18, stp2_29); \
  \
  const __m128i lo_19_28 = _mm_unpacklo_epi16(stp2_19, stp2_28); \
  const __m128i hi_19_28 = _mm_unpackhi_epi16(stp2_19, stp2_28); \
  const __m128i lo_20_27 = _mm_unpacklo_epi16(stp2_20, stp2_27); \
  const __m128i hi_20_27 = _mm_unpackhi_epi16(stp2_20, stp2_27); \
  \
  const __m128i lo_21_26 = _mm_unpacklo_epi16(stp2_21, stp2_26); \
  const __m128i hi_21_26 = _mm_unpackhi_epi16(stp2_21, stp2_26); \
  \
  stp1_0 = _mm_add_epi16(stp2_0, stp2_3); \
  stp1_1 = _mm_add_epi16(stp2_1, stp2_2); \
  stp1_2 = _mm_sub_epi16(stp2_1, stp2_2); \
  stp1_3 = _mm_sub_epi16(stp2_0, stp2_3); \
  \
  tmp0 = _mm_madd_epi16(lo_6_5, stg4_1); \
  tmp1 = _mm_madd_epi16(hi_6_5, stg4_1); \
  tmp2 = _mm_madd_epi16(lo_6_5, stg4_0); \
  tmp3 = _mm_madd_epi16(hi_6_5, stg4_0); \
  \
  tmp0 = _mm_add_epi32(tmp0, rounding); \
  tmp1 = _mm_add_epi32(tmp1, rounding); \
  tmp2 = _mm_add_epi32(tmp2, rounding); \
  tmp3 = _mm_add_epi32(tmp3, rounding); \
  \
  tmp0 = _mm_srai_epi32(tmp0, DCT_CONST_BITS); \
  tmp1 = _mm_srai_epi32(tmp1, DCT_CONST_BITS); \
  tmp2 = _mm_srai_epi32(tmp2, DCT_CONST_BITS); \
  tmp3 = _mm_srai_epi32(tmp3, DCT_CONST_BITS); \
  \
  stp1_5 = _mm_packs_epi32(tmp0, tmp1); \
  stp1_6 = _mm_packs_epi32(tmp2, tmp3); \
  \
  stp1_4 = stp2_4; \
  stp1_7 = stp2_7; \
  \
  stp1_8 = _mm_add_epi16(stp2_8, stp2_11); \
  stp1_9 = _mm_add_epi16(stp2_9, stp2_10); \
  stp1_10 = _mm_sub_epi16(stp2_9, stp2_10); \
  stp1_11 = _mm_sub_epi16(stp2_8, stp2_11); \
  stp1_12 = _mm_sub_epi16(stp2_15, stp2_12); \
  stp1_13 = _mm_sub_epi16(stp2_14, stp2_13); \
  stp1_14 = _mm_add_epi16(stp2_14, stp2_13); \
  stp1_15 = _mm_add_epi16(stp2_15, stp2_12); \
  \
  stp1_16 = stp2_16; \
  stp1_17 = stp2_17; \
  \
  MULTIPLICATION_AND_ADD(lo_18_29, hi_18_29, lo_19_28, hi_19_28, stg4_4, \
                         stg4_5, stg4_4, stg4_5, stp1_18, stp1_29, \
                         stp1_19, stp1_28) \
  MULTIPLICATION_AND_ADD(lo_20_27, hi_20_27, lo_21_26, hi_21_26, stg4_6, \
                         stg4_4, stg4_6, stg4_4, stp1_20, stp1_27, \
                         stp1_21, stp1_26) \
  \
  stp1_22 = stp2_22; \
  stp1_23 = stp2_23; \
  stp1_24 = stp2_24; \
  stp1_25 = stp2_25; \
  stp1_30 = stp2_30; \
  stp1_31 = stp2_31; \
} \
\
/* Stage6 */ \
{ \
  const __m128i lo_10_13 = _mm_unpacklo_epi16(stp1_10, stp1_13); \
  const __m128i hi_10_13 = _mm_unpackhi_epi16(stp1_10, stp1_13); \
  const __m128i lo_11_12 = _mm_unpacklo_epi16(stp1_11, stp1_12); \
  const __m128i hi_11_12 = _mm_unpackhi_epi16(stp1_11, stp1_12); \
  \
  stp2_0 = _mm_add_epi16(stp1_0, stp1_7); \
  stp2_1 = _mm_add_epi16(stp1_1, stp1_6); \
  stp2_2 = _mm_add_epi16(stp1_2, stp1_5); \
  stp2_3 = _mm_add_epi16(stp1_3, stp1_4); \
  stp2_4 = _mm_sub_epi16(stp1_3, stp1_4); \
  stp2_5 = _mm_sub_epi16(stp1_2, stp1_5); \
  stp2_6 = _mm_sub_epi16(stp1_1, stp1_6); \
  stp2_7 = _mm_sub_epi16(stp1_0, stp1_7); \
  \
  stp2_8 = stp1_8; \
  stp2_9 = stp1_9; \
  stp2_14 = stp1_14; \
  stp2_15 = stp1_15; \
  \
  MULTIPLICATION_AND_ADD(lo_10_13, hi_10_13, lo_11_12, hi_11_12, \
                         stg6_0, stg4_0, stg6_0, stg4_0, stp2_10, \
                         stp2_13, stp2_11, stp2_12) \
  \
  stp2_16 = _mm_add_epi16(stp1_16, stp1_23); \
  stp2_17 = _mm_add_epi16(stp1_17, stp1_22); \
  stp2_18 = _mm_add_epi16(stp1_18, stp1_21); \
  stp2_19 = _mm_add_epi16(stp1_19, stp1_20); \
  stp2_20 = _mm_sub_epi16(stp1_19, stp1_20); \
  stp2_21 = _mm_sub_epi16(stp1_18, stp1_21); \
  stp2_22 = _mm_sub_epi16(stp1_17, stp1_22); \
  stp2_23 = _mm_sub_epi16(stp1_16, stp1_23); \
  \
  stp2_24 = _mm_sub_epi16(stp1_31, stp1_24); \
  stp2_25 = _mm_sub_epi16(stp1_30, stp1_25); \
  stp2_26 = _mm_sub_epi16(stp1_29, stp1_26); \
  stp2_27 = _mm_sub_epi16(stp1_28, stp1_27); \
  stp2_28 = _mm_add_epi16(stp1_27, stp1_28); \
  stp2_29 = _mm_add_epi16(stp1_26, stp1_29); \
  stp2_30 = _mm_add_epi16(stp1_25, stp1_30); \
  stp2_31 = _mm_add_epi16(stp1_24, stp1_31); \
} \
\
/* Stage7 */ \
{ \
  const __m128i lo_20_27 = _mm_unpacklo_epi16(stp2_20, stp2_27); \
  const __m128i hi_20_27 = _mm_unpackhi_epi16(stp2_20, stp2_27); \
  const __m128i lo_21_26 = _mm_unpacklo_epi16(stp2_21, stp2_26); \
  const __m128i hi_21_26 = _mm_unpackhi_epi16(stp2_21, stp2_26); \
  \
  const __m128i lo_22_25 = _mm_unpacklo_epi16(stp2_22, stp2_25); \
  const __m128i hi_22_25 = _mm_unpackhi_epi16(stp2_22, stp2_25); \
  const __m128i lo_23_24 = _mm_unpacklo_epi16(stp2_23, stp2_24); \
  const __m128i hi_23_24 = _mm_unpackhi_epi16(stp2_23, stp2_24); \
  \
  stp1_0 = _mm_add_epi16(stp2_0, stp2_15); \
  stp1_1 = _mm_add_epi16(stp2_1, stp2_14); \
  stp1_2 = _mm_add_epi16(stp2_2, stp2_13); \
  stp1_3 = _mm_add_epi16(stp2_3, stp2_12); \
  stp1_4 = _mm_add_epi16(stp2_4, stp2_11); \
  stp1_5 = _mm_add_epi16(stp2_5, stp2_10); \
  stp1_6 = _mm_add_epi16(stp2_6, stp2_9); \
  stp1_7 = _mm_add_epi16(stp2_7, stp2_8); \
  stp1_8 = _mm_sub_epi16(stp2_7, stp2_8); \
  stp1_9 = _mm_sub_epi16(stp2_6, stp2_9); \
  stp1_10 = _mm_sub_epi16(stp2_5, stp2_10); \
  stp1_11 = _mm_sub_epi16(stp2_4, stp2_11); \
  stp1_12 = _mm_sub_epi16(stp2_3, stp2_12); \
  stp1_13 = _mm_sub_epi16(stp2_2, stp2_13); \
  stp1_14 = _mm_sub_epi16(stp2_1, stp2_14); \
  stp1_15 = _mm_sub_epi16(stp2_0, stp2_15); \
  \
  stp1_16 = stp2_16; \
  stp1_17 = stp2_17; \
  stp1_18 = stp2_18; \
  stp1_19 = stp2_19; \
  \
  MULTIPLICATION_AND_ADD(lo_20_27, hi_20_27, lo_21_26, hi_21_26, stg6_0, \
                         stg4_0, stg6_0, stg4_0, stp1_20, stp1_27, \
                         stp1_21, stp1_26) \
  MULTIPLICATION_AND_ADD(lo_22_25, hi_22_25, lo_23_24, hi_23_24, stg6_0, \
                         stg4_0, stg6_0, stg4_0, stp1_22, stp1_25, \
                         stp1_23, stp1_24) \
  \
  stp1_28 = stp2_28; \
  stp1_29 = stp2_29; \
  stp1_30 = stp2_30; \
  stp1_31 = stp2_31; \
}

// Only upper-left 8x8 has non-zero coeff
void vpx_idct32x32_34_add_sse2(const tran_low_t *input, uint8_t *dest,
                               int stride) {
  const __m128i rounding = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i final_rounding = _mm_set1_epi16(1<<5);

  // idct constants for each stage
  const __m128i stg1_0 = pair_set_epi16(cospi_31_64, -cospi_1_64);
  const __m128i stg1_1 = pair_set_epi16(cospi_1_64, cospi_31_64);
  const __m128i stg1_6 = pair_set_epi16(cospi_7_64, -cospi_25_64);
  const __m128i stg1_7 = pair_set_epi16(cospi_25_64, cospi_7_64);
  const __m128i stg1_8 = pair_set_epi16(cospi_27_64, -cospi_5_64);
  const __m128i stg1_9 = pair_set_epi16(cospi_5_64, cospi_27_64);
  const __m128i stg1_14 = pair_set_epi16(cospi_3_64, -cospi_29_64);
  const __m128i stg1_15 = pair_set_epi16(cospi_29_64, cospi_3_64);

  const __m128i stg2_0 = pair_set_epi16(cospi_30_64, -cospi_2_64);
  const __m128i stg2_1 = pair_set_epi16(cospi_2_64, cospi_30_64);
  const __m128i stg2_6 = pair_set_epi16(cospi_6_64, -cospi_26_64);
  const __m128i stg2_7 = pair_set_epi16(cospi_26_64, cospi_6_64);

  const __m128i stg3_0 = pair_set_epi16(cospi_28_64, -cospi_4_64);
  const __m128i stg3_1 = pair_set_epi16(cospi_4_64, cospi_28_64);
  const __m128i stg3_4 = pair_set_epi16(-cospi_4_64, cospi_28_64);
  const __m128i stg3_5 = pair_set_epi16(cospi_28_64, cospi_4_64);
  const __m128i stg3_6 = pair_set_epi16(-cospi_28_64, -cospi_4_64);
  const __m128i stg3_8 = pair_set_epi16(-cospi_20_64, cospi_12_64);
  const __m128i stg3_9 = pair_set_epi16(cospi_12_64, cospi_20_64);
  const __m128i stg3_10 = pair_set_epi16(-cospi_12_64, -cospi_20_64);

  const __m128i stg4_0 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i stg4_1 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i stg4_4 = pair_set_epi16(-cospi_8_64, cospi_24_64);
  const __m128i stg4_5 = pair_set_epi16(cospi_24_64, cospi_8_64);
  const __m128i stg4_6 = pair_set_epi16(-cospi_24_64, -cospi_8_64);

  const __m128i stg6_0 = pair_set_epi16(-cospi_16_64, cospi_16_64);

  __m128i in[32], col[32];
  __m128i stp1_0, stp1_1, stp1_2, stp1_3, stp1_4, stp1_5, stp1_6, stp1_7,
          stp1_8, stp1_9, stp1_10, stp1_11, stp1_12, stp1_13, stp1_14, stp1_15,
          stp1_16, stp1_17, stp1_18, stp1_19, stp1_20, stp1_21, stp1_22,
          stp1_23, stp1_24, stp1_25, stp1_26, stp1_27, stp1_28, stp1_29,
          stp1_30, stp1_31;
  __m128i stp2_0, stp2_1, stp2_2, stp2_3, stp2_4, stp2_5, stp2_6, stp2_7,
          stp2_8, stp2_9, stp2_10, stp2_11, stp2_12, stp2_13, stp2_14, stp2_15,
          stp2_16, stp2_17, stp2_18, stp2_19, stp2_20, stp2_21, stp2_22,
          stp2_23, stp2_24, stp2_25, stp2_26, stp2_27, stp2_28, stp2_29,
          stp2_30, stp2_31;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  int i;

  // Load input data. Only need to load the top left 8x8 block.
  in[0] = load_input_data(input);
  in[1] = load_input_data(input + 32);
  in[2] = load_input_data(input + 64);
  in[3] = load_input_data(input + 96);
  in[4] = load_input_data(input + 128);
  in[5] = load_input_data(input + 160);
  in[6] = load_input_data(input + 192);
  in[7] = load_input_data(input + 224);

  for (i = 8; i < 32; ++i) {
    in[i] = _mm_setzero_si128();
  }

  array_transpose_8x8(in, in);
  // TODO(hkuang): Following transposes are unnecessary. But remove them will
  // lead to performance drop on some devices.
  array_transpose_8x8(in + 8, in + 8);
  array_transpose_8x8(in + 16, in + 16);
  array_transpose_8x8(in + 24, in + 24);

  IDCT32_34

  // 1_D: Store 32 intermediate results for each 8x32 block.
  col[0] = _mm_add_epi16(stp1_0, stp1_31);
  col[1] = _mm_add_epi16(stp1_1, stp1_30);
  col[2] = _mm_add_epi16(stp1_2, stp1_29);
  col[3] = _mm_add_epi16(stp1_3, stp1_28);
  col[4] = _mm_add_epi16(stp1_4, stp1_27);
  col[5] = _mm_add_epi16(stp1_5, stp1_26);
  col[6] = _mm_add_epi16(stp1_6, stp1_25);
  col[7] = _mm_add_epi16(stp1_7, stp1_24);
  col[8] = _mm_add_epi16(stp1_8, stp1_23);
  col[9] = _mm_add_epi16(stp1_9, stp1_22);
  col[10] = _mm_add_epi16(stp1_10, stp1_21);
  col[11] = _mm_add_epi16(stp1_11, stp1_20);
  col[12] = _mm_add_epi16(stp1_12, stp1_19);
  col[13] = _mm_add_epi16(stp1_13, stp1_18);
  col[14] = _mm_add_epi16(stp1_14, stp1_17);
  col[15] = _mm_add_epi16(stp1_15, stp1_16);
  col[16] = _mm_sub_epi16(stp1_15, stp1_16);
  col[17] = _mm_sub_epi16(stp1_14, stp1_17);
  col[18] = _mm_sub_epi16(stp1_13, stp1_18);
  col[19] = _mm_sub_epi16(stp1_12, stp1_19);
  col[20] = _mm_sub_epi16(stp1_11, stp1_20);
  col[21] = _mm_sub_epi16(stp1_10, stp1_21);
  col[22] = _mm_sub_epi16(stp1_9, stp1_22);
  col[23] = _mm_sub_epi16(stp1_8, stp1_23);
  col[24] = _mm_sub_epi16(stp1_7, stp1_24);
  col[25] = _mm_sub_epi16(stp1_6, stp1_25);
  col[26] = _mm_sub_epi16(stp1_5, stp1_26);
  col[27] = _mm_sub_epi16(stp1_4, stp1_27);
  col[28] = _mm_sub_epi16(stp1_3, stp1_28);
  col[29] = _mm_sub_epi16(stp1_2, stp1_29);
  col[30] = _mm_sub_epi16(stp1_1, stp1_30);
  col[31] = _mm_sub_epi16(stp1_0, stp1_31);
  for (i = 0; i < 4; i++) {
    int j;
    const __m128i zero = _mm_setzero_si128();
    // Transpose 32x8 block to 8x32 block
    array_transpose_8x8(col + i * 8, in);
    IDCT32_34

    // 2_D: Calculate the results and store them to destination.
    in[0] = _mm_add_epi16(stp1_0, stp1_31);
    in[1] = _mm_add_epi16(stp1_1, stp1_30);
    in[2] = _mm_add_epi16(stp1_2, stp1_29);
    in[3] = _mm_add_epi16(stp1_3, stp1_28);
    in[4] = _mm_add_epi16(stp1_4, stp1_27);
    in[5] = _mm_add_epi16(stp1_5, stp1_26);
    in[6] = _mm_add_epi16(stp1_6, stp1_25);
    in[7] = _mm_add_epi16(stp1_7, stp1_24);
    in[8] = _mm_add_epi16(stp1_8, stp1_23);
    in[9] = _mm_add_epi16(stp1_9, stp1_22);
    in[10] = _mm_add_epi16(stp1_10, stp1_21);
    in[11] = _mm_add_epi16(stp1_11, stp1_20);
    in[12] = _mm_add_epi16(stp1_12, stp1_19);
    in[13] = _mm_add_epi16(stp1_13, stp1_18);
    in[14] = _mm_add_epi16(stp1_14, stp1_17);
    in[15] = _mm_add_epi16(stp1_15, stp1_16);
    in[16] = _mm_sub_epi16(stp1_15, stp1_16);
    in[17] = _mm_sub_epi16(stp1_14, stp1_17);
    in[18] = _mm_sub_epi16(stp1_13, stp1_18);
    in[19] = _mm_sub_epi16(stp1_12, stp1_19);
    in[20] = _mm_sub_epi16(stp1_11, stp1_20);
    in[21] = _mm_sub_epi16(stp1_10, stp1_21);
    in[22] = _mm_sub_epi16(stp1_9, stp1_22);
    in[23] = _mm_sub_epi16(stp1_8, stp1_23);
    in[24] = _mm_sub_epi16(stp1_7, stp1_24);
    in[25] = _mm_sub_epi16(stp1_6, stp1_25);
    in[26] = _mm_sub_epi16(stp1_5, stp1_26);
    in[27] = _mm_sub_epi16(stp1_4, stp1_27);
    in[28] = _mm_sub_epi16(stp1_3, stp1_28);
    in[29] = _mm_sub_epi16(stp1_2, stp1_29);
    in[30] = _mm_sub_epi16(stp1_1, stp1_30);
    in[31] = _mm_sub_epi16(stp1_0, stp1_31);

    for (j = 0; j < 32; ++j) {
      // Final rounding and shift
      in[j] = _mm_adds_epi16(in[j], final_rounding);
      in[j] = _mm_srai_epi16(in[j], 6);
      RECON_AND_STORE(dest + j * stride, in[j]);
    }

    dest += 8;
  }
}

void vpx_idct32x32_1024_add_sse2(const tran_low_t *input, uint8_t *dest,
                                 int stride) {
  const __m128i rounding = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i final_rounding = _mm_set1_epi16(1 << 5);
  const __m128i zero = _mm_setzero_si128();

  // idct constants for each stage
  const __m128i stg1_0 = pair_set_epi16(cospi_31_64, -cospi_1_64);
  const __m128i stg1_1 = pair_set_epi16(cospi_1_64, cospi_31_64);
  const __m128i stg1_2 = pair_set_epi16(cospi_15_64, -cospi_17_64);
  const __m128i stg1_3 = pair_set_epi16(cospi_17_64, cospi_15_64);
  const __m128i stg1_4 = pair_set_epi16(cospi_23_64, -cospi_9_64);
  const __m128i stg1_5 = pair_set_epi16(cospi_9_64, cospi_23_64);
  const __m128i stg1_6 = pair_set_epi16(cospi_7_64, -cospi_25_64);
  const __m128i stg1_7 = pair_set_epi16(cospi_25_64, cospi_7_64);
  const __m128i stg1_8 = pair_set_epi16(cospi_27_64, -cospi_5_64);
  const __m128i stg1_9 = pair_set_epi16(cospi_5_64, cospi_27_64);
  const __m128i stg1_10 = pair_set_epi16(cospi_11_64, -cospi_21_64);
  const __m128i stg1_11 = pair_set_epi16(cospi_21_64, cospi_11_64);
  const __m128i stg1_12 = pair_set_epi16(cospi_19_64, -cospi_13_64);
  const __m128i stg1_13 = pair_set_epi16(cospi_13_64, cospi_19_64);
  const __m128i stg1_14 = pair_set_epi16(cospi_3_64, -cospi_29_64);
  const __m128i stg1_15 = pair_set_epi16(cospi_29_64, cospi_3_64);

  const __m128i stg2_0 = pair_set_epi16(cospi_30_64, -cospi_2_64);
  const __m128i stg2_1 = pair_set_epi16(cospi_2_64, cospi_30_64);
  const __m128i stg2_2 = pair_set_epi16(cospi_14_64, -cospi_18_64);
  const __m128i stg2_3 = pair_set_epi16(cospi_18_64, cospi_14_64);
  const __m128i stg2_4 = pair_set_epi16(cospi_22_64, -cospi_10_64);
  const __m128i stg2_5 = pair_set_epi16(cospi_10_64, cospi_22_64);
  const __m128i stg2_6 = pair_set_epi16(cospi_6_64, -cospi_26_64);
  const __m128i stg2_7 = pair_set_epi16(cospi_26_64, cospi_6_64);

  const __m128i stg3_0 = pair_set_epi16(cospi_28_64, -cospi_4_64);
  const __m128i stg3_1 = pair_set_epi16(cospi_4_64, cospi_28_64);
  const __m128i stg3_2 = pair_set_epi16(cospi_12_64, -cospi_20_64);
  const __m128i stg3_3 = pair_set_epi16(cospi_20_64, cospi_12_64);
  const __m128i stg3_4 = pair_set_epi16(-cospi_4_64, cospi_28_64);
  const __m128i stg3_5 = pair_set_epi16(cospi_28_64, cospi_4_64);
  const __m128i stg3_6 = pair_set_epi16(-cospi_28_64, -cospi_4_64);
  const __m128i stg3_8 = pair_set_epi16(-cospi_20_64, cospi_12_64);
  const __m128i stg3_9 = pair_set_epi16(cospi_12_64, cospi_20_64);
  const __m128i stg3_10 = pair_set_epi16(-cospi_12_64, -cospi_20_64);

  const __m128i stg4_0 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i stg4_1 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i stg4_2 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i stg4_3 = pair_set_epi16(cospi_8_64, cospi_24_64);
  const __m128i stg4_4 = pair_set_epi16(-cospi_8_64, cospi_24_64);
  const __m128i stg4_5 = pair_set_epi16(cospi_24_64, cospi_8_64);
  const __m128i stg4_6 = pair_set_epi16(-cospi_24_64, -cospi_8_64);

  const __m128i stg6_0 = pair_set_epi16(-cospi_16_64, cospi_16_64);

  __m128i in[32], col[128], zero_idx[16];
  __m128i stp1_0, stp1_1, stp1_2, stp1_3, stp1_4, stp1_5, stp1_6, stp1_7,
          stp1_8, stp1_9, stp1_10, stp1_11, stp1_12, stp1_13, stp1_14, stp1_15,
          stp1_16, stp1_17, stp1_18, stp1_19, stp1_20, stp1_21, stp1_22,
          stp1_23, stp1_24, stp1_25, stp1_26, stp1_27, stp1_28, stp1_29,
          stp1_30, stp1_31;
  __m128i stp2_0, stp2_1, stp2_2, stp2_3, stp2_4, stp2_5, stp2_6, stp2_7,
          stp2_8, stp2_9, stp2_10, stp2_11, stp2_12, stp2_13, stp2_14, stp2_15,
          stp2_16, stp2_17, stp2_18, stp2_19, stp2_20, stp2_21, stp2_22,
          stp2_23, stp2_24, stp2_25, stp2_26, stp2_27, stp2_28, stp2_29,
          stp2_30, stp2_31;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  int i, j, i32;

  for (i = 0; i < 4; i++) {
    i32 = (i << 5);
    // First 1-D idct
    // Load input data.
    LOAD_DQCOEFF(in[0], input);
    LOAD_DQCOEFF(in[8], input);
    LOAD_DQCOEFF(in[16], input);
    LOAD_DQCOEFF(in[24], input);
    LOAD_DQCOEFF(in[1], input);
    LOAD_DQCOEFF(in[9], input);
    LOAD_DQCOEFF(in[17], input);
    LOAD_DQCOEFF(in[25], input);
    LOAD_DQCOEFF(in[2], input);
    LOAD_DQCOEFF(in[10], input);
    LOAD_DQCOEFF(in[18], input);
    LOAD_DQCOEFF(in[26], input);
    LOAD_DQCOEFF(in[3], input);
    LOAD_DQCOEFF(in[11], input);
    LOAD_DQCOEFF(in[19], input);
    LOAD_DQCOEFF(in[27], input);

    LOAD_DQCOEFF(in[4], input);
    LOAD_DQCOEFF(in[12], input);
    LOAD_DQCOEFF(in[20], input);
    LOAD_DQCOEFF(in[28], input);
    LOAD_DQCOEFF(in[5], input);
    LOAD_DQCOEFF(in[13], input);
    LOAD_DQCOEFF(in[21], input);
    LOAD_DQCOEFF(in[29], input);
    LOAD_DQCOEFF(in[6], input);
    LOAD_DQCOEFF(in[14], input);
    LOAD_DQCOEFF(in[22], input);
    LOAD_DQCOEFF(in[30], input);
    LOAD_DQCOEFF(in[7], input);
    LOAD_DQCOEFF(in[15], input);
    LOAD_DQCOEFF(in[23], input);
    LOAD_DQCOEFF(in[31], input);

    // checking if all entries are zero
    zero_idx[0] = _mm_or_si128(in[0], in[1]);
    zero_idx[1] = _mm_or_si128(in[2], in[3]);
    zero_idx[2] = _mm_or_si128(in[4], in[5]);
    zero_idx[3] = _mm_or_si128(in[6], in[7]);
    zero_idx[4] = _mm_or_si128(in[8], in[9]);
    zero_idx[5] = _mm_or_si128(in[10], in[11]);
    zero_idx[6] = _mm_or_si128(in[12], in[13]);
    zero_idx[7] = _mm_or_si128(in[14], in[15]);
    zero_idx[8] = _mm_or_si128(in[16], in[17]);
    zero_idx[9] = _mm_or_si128(in[18], in[19]);
    zero_idx[10] = _mm_or_si128(in[20], in[21]);
    zero_idx[11] = _mm_or_si128(in[22], in[23]);
    zero_idx[12] = _mm_or_si128(in[24], in[25]);
    zero_idx[13] = _mm_or_si128(in[26], in[27]);
    zero_idx[14] = _mm_or_si128(in[28], in[29]);
    zero_idx[15] = _mm_or_si128(in[30], in[31]);

    zero_idx[0] = _mm_or_si128(zero_idx[0], zero_idx[1]);
    zero_idx[1] = _mm_or_si128(zero_idx[2], zero_idx[3]);
    zero_idx[2] = _mm_or_si128(zero_idx[4], zero_idx[5]);
    zero_idx[3] = _mm_or_si128(zero_idx[6], zero_idx[7]);
    zero_idx[4] = _mm_or_si128(zero_idx[8], zero_idx[9]);
    zero_idx[5] = _mm_or_si128(zero_idx[10], zero_idx[11]);
    zero_idx[6] = _mm_or_si128(zero_idx[12], zero_idx[13]);
    zero_idx[7] = _mm_or_si128(zero_idx[14], zero_idx[15]);

    zero_idx[8] = _mm_or_si128(zero_idx[0], zero_idx[1]);
    zero_idx[9] = _mm_or_si128(zero_idx[2], zero_idx[3]);
    zero_idx[10] = _mm_or_si128(zero_idx[4], zero_idx[5]);
    zero_idx[11] = _mm_or_si128(zero_idx[6], zero_idx[7]);
    zero_idx[12] = _mm_or_si128(zero_idx[8], zero_idx[9]);
    zero_idx[13] = _mm_or_si128(zero_idx[10], zero_idx[11]);
    zero_idx[14] = _mm_or_si128(zero_idx[12], zero_idx[13]);

    if (_mm_movemask_epi8(_mm_cmpeq_epi32(zero_idx[14], zero)) == 0xFFFF) {
      col[i32 + 0] = _mm_setzero_si128();
      col[i32 + 1] = _mm_setzero_si128();
      col[i32 + 2] = _mm_setzero_si128();
      col[i32 + 3] = _mm_setzero_si128();
      col[i32 + 4] = _mm_setzero_si128();
      col[i32 + 5] = _mm_setzero_si128();
      col[i32 + 6] = _mm_setzero_si128();
      col[i32 + 7] = _mm_setzero_si128();
      col[i32 + 8] = _mm_setzero_si128();
      col[i32 + 9] = _mm_setzero_si128();
      col[i32 + 10] = _mm_setzero_si128();
      col[i32 + 11] = _mm_setzero_si128();
      col[i32 + 12] = _mm_setzero_si128();
      col[i32 + 13] = _mm_setzero_si128();
      col[i32 + 14] = _mm_setzero_si128();
      col[i32 + 15] = _mm_setzero_si128();
      col[i32 + 16] = _mm_setzero_si128();
      col[i32 + 17] = _mm_setzero_si128();
      col[i32 + 18] = _mm_setzero_si128();
      col[i32 + 19] = _mm_setzero_si128();
      col[i32 + 20] = _mm_setzero_si128();
      col[i32 + 21] = _mm_setzero_si128();
      col[i32 + 22] = _mm_setzero_si128();
      col[i32 + 23] = _mm_setzero_si128();
      col[i32 + 24] = _mm_setzero_si128();
      col[i32 + 25] = _mm_setzero_si128();
      col[i32 + 26] = _mm_setzero_si128();
      col[i32 + 27] = _mm_setzero_si128();
      col[i32 + 28] = _mm_setzero_si128();
      col[i32 + 29] = _mm_setzero_si128();
      col[i32 + 30] = _mm_setzero_si128();
      col[i32 + 31] = _mm_setzero_si128();
      continue;
    }

    // Transpose 32x8 block to 8x32 block
    array_transpose_8x8(in, in);
    array_transpose_8x8(in + 8, in + 8);
    array_transpose_8x8(in + 16, in + 16);
    array_transpose_8x8(in + 24, in + 24);

    IDCT32

    // 1_D: Store 32 intermediate results for each 8x32 block.
    col[i32 + 0] = _mm_add_epi16(stp1_0, stp1_31);
    col[i32 + 1] = _mm_add_epi16(stp1_1, stp1_30);
    col[i32 + 2] = _mm_add_epi16(stp1_2, stp1_29);
    col[i32 + 3] = _mm_add_epi16(stp1_3, stp1_28);
    col[i32 + 4] = _mm_add_epi16(stp1_4, stp1_27);
    col[i32 + 5] = _mm_add_epi16(stp1_5, stp1_26);
    col[i32 + 6] = _mm_add_epi16(stp1_6, stp1_25);
    col[i32 + 7] = _mm_add_epi16(stp1_7, stp1_24);
    col[i32 + 8] = _mm_add_epi16(stp1_8, stp1_23);
    col[i32 + 9] = _mm_add_epi16(stp1_9, stp1_22);
    col[i32 + 10] = _mm_add_epi16(stp1_10, stp1_21);
    col[i32 + 11] = _mm_add_epi16(stp1_11, stp1_20);
    col[i32 + 12] = _mm_add_epi16(stp1_12, stp1_19);
    col[i32 + 13] = _mm_add_epi16(stp1_13, stp1_18);
    col[i32 + 14] = _mm_add_epi16(stp1_14, stp1_17);
    col[i32 + 15] = _mm_add_epi16(stp1_15, stp1_16);
    col[i32 + 16] = _mm_sub_epi16(stp1_15, stp1_16);
    col[i32 + 17] = _mm_sub_epi16(stp1_14, stp1_17);
    col[i32 + 18] = _mm_sub_epi16(stp1_13, stp1_18);
    col[i32 + 19] = _mm_sub_epi16(stp1_12, stp1_19);
    col[i32 + 20] = _mm_sub_epi16(stp1_11, stp1_20);
    col[i32 + 21] = _mm_sub_epi16(stp1_10, stp1_21);
    col[i32 + 22] = _mm_sub_epi16(stp1_9, stp1_22);
    col[i32 + 23] = _mm_sub_epi16(stp1_8, stp1_23);
    col[i32 + 24] = _mm_sub_epi16(stp1_7, stp1_24);
    col[i32 + 25] = _mm_sub_epi16(stp1_6, stp1_25);
    col[i32 + 26] = _mm_sub_epi16(stp1_5, stp1_26);
    col[i32 + 27] = _mm_sub_epi16(stp1_4, stp1_27);
    col[i32 + 28] = _mm_sub_epi16(stp1_3, stp1_28);
    col[i32 + 29] = _mm_sub_epi16(stp1_2, stp1_29);
    col[i32 + 30] = _mm_sub_epi16(stp1_1, stp1_30);
    col[i32 + 31] = _mm_sub_epi16(stp1_0, stp1_31);
  }
  for (i = 0; i < 4; i++) {
    // Second 1-D idct
    j = i << 3;

    // Transpose 32x8 block to 8x32 block
    array_transpose_8x8(col + j, in);
    array_transpose_8x8(col + j + 32, in + 8);
    array_transpose_8x8(col + j + 64, in + 16);
    array_transpose_8x8(col + j + 96, in + 24);

    IDCT32

    // 2_D: Calculate the results and store them to destination.
    in[0] = _mm_add_epi16(stp1_0, stp1_31);
    in[1] = _mm_add_epi16(stp1_1, stp1_30);
    in[2] = _mm_add_epi16(stp1_2, stp1_29);
    in[3] = _mm_add_epi16(stp1_3, stp1_28);
    in[4] = _mm_add_epi16(stp1_4, stp1_27);
    in[5] = _mm_add_epi16(stp1_5, stp1_26);
    in[6] = _mm_add_epi16(stp1_6, stp1_25);
    in[7] = _mm_add_epi16(stp1_7, stp1_24);
    in[8] = _mm_add_epi16(stp1_8, stp1_23);
    in[9] = _mm_add_epi16(stp1_9, stp1_22);
    in[10] = _mm_add_epi16(stp1_10, stp1_21);
    in[11] = _mm_add_epi16(stp1_11, stp1_20);
    in[12] = _mm_add_epi16(stp1_12, stp1_19);
    in[13] = _mm_add_epi16(stp1_13, stp1_18);
    in[14] = _mm_add_epi16(stp1_14, stp1_17);
    in[15] = _mm_add_epi16(stp1_15, stp1_16);
    in[16] = _mm_sub_epi16(stp1_15, stp1_16);
    in[17] = _mm_sub_epi16(stp1_14, stp1_17);
    in[18] = _mm_sub_epi16(stp1_13, stp1_18);
    in[19] = _mm_sub_epi16(stp1_12, stp1_19);
    in[20] = _mm_sub_epi16(stp1_11, stp1_20);
    in[21] = _mm_sub_epi16(stp1_10, stp1_21);
    in[22] = _mm_sub_epi16(stp1_9, stp1_22);
    in[23] = _mm_sub_epi16(stp1_8, stp1_23);
    in[24] = _mm_sub_epi16(stp1_7, stp1_24);
    in[25] = _mm_sub_epi16(stp1_6, stp1_25);
    in[26] = _mm_sub_epi16(stp1_5, stp1_26);
    in[27] = _mm_sub_epi16(stp1_4, stp1_27);
    in[28] = _mm_sub_epi16(stp1_3, stp1_28);
    in[29] = _mm_sub_epi16(stp1_2, stp1_29);
    in[30] = _mm_sub_epi16(stp1_1, stp1_30);
    in[31] = _mm_sub_epi16(stp1_0, stp1_31);

    for (j = 0; j < 32; ++j) {
      // Final rounding and shift
      in[j] = _mm_adds_epi16(in[j], final_rounding);
      in[j] = _mm_srai_epi16(in[j], 6);
      RECON_AND_STORE(dest + j * stride, in[j]);
    }

    dest += 8;
  }
}

void vpx_idct32x32_1_add_sse2(const tran_low_t *input, uint8_t *dest,
                              int stride) {
  __m128i dc_value;
  const __m128i zero = _mm_setzero_si128();
  int a, j;

  a = (int)dct_const_round_shift(input[0] * cospi_16_64);
  a = (int)dct_const_round_shift(a * cospi_16_64);
  a = ROUND_POWER_OF_TWO(a, 6);

  dc_value = _mm_set1_epi16(a);

  for (j = 0; j < 32; ++j) {
    RECON_AND_STORE(dest +  0 + j * stride, dc_value);
    RECON_AND_STORE(dest +  8 + j * stride, dc_value);
    RECON_AND_STORE(dest + 16 + j * stride, dc_value);
    RECON_AND_STORE(dest + 24 + j * stride, dc_value);
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE __m128i clamp_high_sse2(__m128i value, int bd) {
  __m128i ubounded, retval;
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i one = _mm_set1_epi16(1);
  const __m128i max = _mm_subs_epi16(_mm_slli_epi16(one, bd), one);
  ubounded = _mm_cmpgt_epi16(value, max);
  retval = _mm_andnot_si128(ubounded, value);
  ubounded = _mm_and_si128(ubounded, max);
  retval = _mm_or_si128(retval, ubounded);
  retval = _mm_and_si128(retval, _mm_cmpgt_epi16(retval, zero));
  return retval;
}

void vpx_highbd_idct4x4_16_add_sse2(const tran_low_t *input, uint8_t *dest8,
                                    int stride, int bd) {
  tran_low_t out[4 * 4];
  tran_low_t *outptr = out;
  int i, j;
  __m128i inptr[4];
  __m128i sign_bits[2];
  __m128i temp_mm, min_input, max_input;
  int test;
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);
  int optimised_cols = 0;
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i eight = _mm_set1_epi16(8);
  const __m128i max = _mm_set1_epi16(12043);
  const __m128i min = _mm_set1_epi16(-12043);
  // Load input into __m128i
  inptr[0] = _mm_loadu_si128((const __m128i *)input);
  inptr[1] = _mm_loadu_si128((const __m128i *)(input + 4));
  inptr[2] = _mm_loadu_si128((const __m128i *)(input + 8));
  inptr[3] = _mm_loadu_si128((const __m128i *)(input + 12));

  // Pack to 16 bits
  inptr[0] = _mm_packs_epi32(inptr[0], inptr[1]);
  inptr[1] = _mm_packs_epi32(inptr[2], inptr[3]);

  max_input = _mm_max_epi16(inptr[0], inptr[1]);
  min_input = _mm_min_epi16(inptr[0], inptr[1]);
  max_input = _mm_cmpgt_epi16(max_input, max);
  min_input = _mm_cmplt_epi16(min_input, min);
  temp_mm = _mm_or_si128(max_input, min_input);
  test = _mm_movemask_epi8(temp_mm);

  if (!test) {
    // Do the row transform
    idct4_sse2(inptr);

    // Check the min & max values
    max_input = _mm_max_epi16(inptr[0], inptr[1]);
    min_input = _mm_min_epi16(inptr[0], inptr[1]);
    max_input = _mm_cmpgt_epi16(max_input, max);
    min_input = _mm_cmplt_epi16(min_input, min);
    temp_mm = _mm_or_si128(max_input, min_input);
    test = _mm_movemask_epi8(temp_mm);

    if (test) {
      transpose_4x4(inptr);
      sign_bits[0] = _mm_cmplt_epi16(inptr[0], zero);
      sign_bits[1] = _mm_cmplt_epi16(inptr[1], zero);
      inptr[3] = _mm_unpackhi_epi16(inptr[1], sign_bits[1]);
      inptr[2] = _mm_unpacklo_epi16(inptr[1], sign_bits[1]);
      inptr[1] = _mm_unpackhi_epi16(inptr[0], sign_bits[0]);
      inptr[0] = _mm_unpacklo_epi16(inptr[0], sign_bits[0]);
      _mm_storeu_si128((__m128i *)outptr, inptr[0]);
      _mm_storeu_si128((__m128i *)(outptr + 4), inptr[1]);
      _mm_storeu_si128((__m128i *)(outptr + 8), inptr[2]);
      _mm_storeu_si128((__m128i *)(outptr + 12), inptr[3]);
    } else {
      // Set to use the optimised transform for the column
      optimised_cols = 1;
    }
  } else {
    // Run the un-optimised row transform
    for (i = 0; i < 4; ++i) {
      vpx_highbd_idct4_c(input, outptr, bd);
      input += 4;
      outptr += 4;
    }
  }

  if (optimised_cols) {
    idct4_sse2(inptr);

    // Final round and shift
    inptr[0] = _mm_add_epi16(inptr[0], eight);
    inptr[1] = _mm_add_epi16(inptr[1], eight);

    inptr[0] = _mm_srai_epi16(inptr[0], 4);
    inptr[1] = _mm_srai_epi16(inptr[1], 4);

    // Reconstruction and Store
    {
      __m128i d0 = _mm_loadl_epi64((const __m128i *)dest);
      __m128i d2 = _mm_loadl_epi64((const __m128i *)(dest + stride * 2));
      d0 = _mm_unpacklo_epi64(
          d0, _mm_loadl_epi64((const __m128i *)(dest + stride)));
      d2 = _mm_unpacklo_epi64(
          d2, _mm_loadl_epi64((const __m128i *)(dest + stride * 3)));
      d0 = clamp_high_sse2(_mm_adds_epi16(d0, inptr[0]), bd);
      d2 = clamp_high_sse2(_mm_adds_epi16(d2, inptr[1]), bd);
      // store input0
      _mm_storel_epi64((__m128i *)dest, d0);
      // store input1
      d0 = _mm_srli_si128(d0, 8);
      _mm_storel_epi64((__m128i *)(dest + stride), d0);
      // store input2
      _mm_storel_epi64((__m128i *)(dest + stride * 2), d2);
      // store input3
      d2 = _mm_srli_si128(d2, 8);
      _mm_storel_epi64((__m128i *)(dest + stride * 3), d2);
    }
  } else {
    // Run the un-optimised column transform
    tran_low_t temp_in[4], temp_out[4];
    // Columns
    for (i = 0; i < 4; ++i) {
      for (j = 0; j < 4; ++j)
        temp_in[j] = out[j * 4 + i];
      vpx_highbd_idct4_c(temp_in, temp_out, bd);
      for (j = 0; j < 4; ++j) {
        dest[j * stride + i] = highbd_clip_pixel_add(
            dest[j * stride + i], ROUND_POWER_OF_TWO(temp_out[j], 4), bd);
      }
    }
  }
}

void vpx_highbd_idct8x8_64_add_sse2(const tran_low_t *input, uint8_t *dest8,
                                    int stride, int bd) {
  tran_low_t out[8 * 8];
  tran_low_t *outptr = out;
  int i, j, test;
  __m128i inptr[8];
  __m128i min_input, max_input, temp1, temp2, sign_bits;
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i sixteen = _mm_set1_epi16(16);
  const __m128i max = _mm_set1_epi16(6201);
  const __m128i min = _mm_set1_epi16(-6201);
  int optimised_cols = 0;

  // Load input into __m128i & pack to 16 bits
  for (i = 0; i < 8; i++) {
    temp1 = _mm_loadu_si128((const __m128i *)(input + 8 * i));
    temp2 = _mm_loadu_si128((const __m128i *)(input + 8 * i + 4));
    inptr[i] = _mm_packs_epi32(temp1, temp2);
  }

  // Find the min & max for the row transform
  max_input = _mm_max_epi16(inptr[0], inptr[1]);
  min_input = _mm_min_epi16(inptr[0], inptr[1]);
  for (i = 2; i < 8; i++) {
    max_input = _mm_max_epi16(max_input, inptr[i]);
    min_input = _mm_min_epi16(min_input, inptr[i]);
  }
  max_input = _mm_cmpgt_epi16(max_input, max);
  min_input = _mm_cmplt_epi16(min_input, min);
  temp1 = _mm_or_si128(max_input, min_input);
  test = _mm_movemask_epi8(temp1);

  if (!test) {
    // Do the row transform
    idct8_sse2(inptr);

    // Find the min & max for the column transform
    max_input = _mm_max_epi16(inptr[0], inptr[1]);
    min_input = _mm_min_epi16(inptr[0], inptr[1]);
    for (i = 2; i < 8; i++) {
      max_input = _mm_max_epi16(max_input, inptr[i]);
      min_input = _mm_min_epi16(min_input, inptr[i]);
    }
    max_input = _mm_cmpgt_epi16(max_input, max);
    min_input = _mm_cmplt_epi16(min_input, min);
    temp1 = _mm_or_si128(max_input, min_input);
    test = _mm_movemask_epi8(temp1);

    if (test) {
      array_transpose_8x8(inptr, inptr);
      for (i = 0; i < 8; i++) {
        sign_bits = _mm_cmplt_epi16(inptr[i], zero);
        temp1 = _mm_unpackhi_epi16(inptr[i], sign_bits);
        temp2 = _mm_unpacklo_epi16(inptr[i], sign_bits);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (2 * i + 1)), temp1);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (2 * i)), temp2);
      }
    } else {
      // Set to use the optimised transform for the column
      optimised_cols = 1;
    }
  } else {
    // Run the un-optimised row transform
    for (i = 0; i < 8; ++i) {
      vpx_highbd_idct8_c(input, outptr, bd);
      input += 8;
      outptr += 8;
    }
  }

  if (optimised_cols) {
    idct8_sse2(inptr);

    // Final round & shift and Reconstruction and Store
    {
      __m128i d[8];
      for (i = 0; i < 8; i++) {
        inptr[i] = _mm_add_epi16(inptr[i], sixteen);
        d[i] = _mm_loadu_si128((const __m128i *)(dest + stride*i));
        inptr[i] = _mm_srai_epi16(inptr[i], 5);
        d[i] = clamp_high_sse2(_mm_adds_epi16(d[i], inptr[i]), bd);
        // Store
        _mm_storeu_si128((__m128i *)(dest + stride*i), d[i]);
      }
    }
  } else {
    // Run the un-optimised column transform
    tran_low_t temp_in[8], temp_out[8];
    for (i = 0; i < 8; ++i) {
      for (j = 0; j < 8; ++j)
        temp_in[j] = out[j * 8 + i];
      vpx_highbd_idct8_c(temp_in, temp_out, bd);
      for (j = 0; j < 8; ++j) {
        dest[j * stride + i] = highbd_clip_pixel_add(
            dest[j * stride + i], ROUND_POWER_OF_TWO(temp_out[j], 5), bd);
      }
    }
  }
}

void vpx_highbd_idct8x8_10_add_sse2(const tran_low_t *input, uint8_t *dest8,
                                    int stride, int bd) {
  tran_low_t out[8 * 8] = { 0 };
  tran_low_t *outptr = out;
  int i, j, test;
  __m128i inptr[8];
  __m128i min_input, max_input, temp1, temp2, sign_bits;
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i sixteen = _mm_set1_epi16(16);
  const __m128i max = _mm_set1_epi16(6201);
  const __m128i min = _mm_set1_epi16(-6201);
  int optimised_cols = 0;

  // Load input into __m128i & pack to 16 bits
  for (i = 0; i < 8; i++) {
    temp1 = _mm_loadu_si128((const __m128i *)(input + 8 * i));
    temp2 = _mm_loadu_si128((const __m128i *)(input + 8 * i + 4));
    inptr[i] = _mm_packs_epi32(temp1, temp2);
  }

  // Find the min & max for the row transform
  // only first 4 row has non-zero coefs
  max_input = _mm_max_epi16(inptr[0], inptr[1]);
  min_input = _mm_min_epi16(inptr[0], inptr[1]);
  for (i = 2; i < 4; i++) {
    max_input = _mm_max_epi16(max_input, inptr[i]);
    min_input = _mm_min_epi16(min_input, inptr[i]);
  }
  max_input = _mm_cmpgt_epi16(max_input, max);
  min_input = _mm_cmplt_epi16(min_input, min);
  temp1 = _mm_or_si128(max_input, min_input);
  test = _mm_movemask_epi8(temp1);

  if (!test) {
    // Do the row transform
    idct8_sse2(inptr);

    // Find the min & max for the column transform
    // N.B. Only first 4 cols contain non-zero coeffs
    max_input = _mm_max_epi16(inptr[0], inptr[1]);
    min_input = _mm_min_epi16(inptr[0], inptr[1]);
    for (i = 2; i < 8; i++) {
      max_input = _mm_max_epi16(max_input, inptr[i]);
      min_input = _mm_min_epi16(min_input, inptr[i]);
    }
    max_input = _mm_cmpgt_epi16(max_input, max);
    min_input = _mm_cmplt_epi16(min_input, min);
    temp1 = _mm_or_si128(max_input, min_input);
    test = _mm_movemask_epi8(temp1);

    if (test) {
      // Use fact only first 4 rows contain non-zero coeffs
      array_transpose_4X8(inptr, inptr);
      for (i = 0; i < 4; i++) {
        sign_bits = _mm_cmplt_epi16(inptr[i], zero);
        temp1 = _mm_unpackhi_epi16(inptr[i], sign_bits);
        temp2 = _mm_unpacklo_epi16(inptr[i], sign_bits);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (2 * i + 1)), temp1);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (2 * i)), temp2);
      }
    } else {
      // Set to use the optimised transform for the column
      optimised_cols = 1;
    }
  } else {
    // Run the un-optimised row transform
    for (i = 0; i < 4; ++i) {
      vpx_highbd_idct8_c(input, outptr, bd);
      input += 8;
      outptr += 8;
    }
  }

  if (optimised_cols) {
    idct8_sse2(inptr);

    // Final round & shift and Reconstruction and Store
    {
      __m128i d[8];
      for (i = 0; i < 8; i++) {
        inptr[i] = _mm_add_epi16(inptr[i], sixteen);
        d[i] = _mm_loadu_si128((const __m128i *)(dest + stride*i));
        inptr[i] = _mm_srai_epi16(inptr[i], 5);
        d[i] = clamp_high_sse2(_mm_adds_epi16(d[i], inptr[i]), bd);
        // Store
        _mm_storeu_si128((__m128i *)(dest + stride*i), d[i]);
      }
    }
  } else {
    // Run the un-optimised column transform
    tran_low_t temp_in[8], temp_out[8];
    for (i = 0; i < 8; ++i) {
      for (j = 0; j < 8; ++j)
        temp_in[j] = out[j * 8 + i];
      vpx_highbd_idct8_c(temp_in, temp_out, bd);
      for (j = 0; j < 8; ++j) {
        dest[j * stride + i] = highbd_clip_pixel_add(
            dest[j * stride + i], ROUND_POWER_OF_TWO(temp_out[j], 5), bd);
      }
    }
  }
}

void vpx_highbd_idct16x16_256_add_sse2(const tran_low_t *input, uint8_t *dest8,
                                       int stride, int bd) {
  tran_low_t out[16 * 16];
  tran_low_t *outptr = out;
  int i, j, test;
  __m128i inptr[32];
  __m128i min_input, max_input, temp1, temp2, sign_bits;
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i rounding = _mm_set1_epi16(32);
  const __m128i max = _mm_set1_epi16(3155);
  const __m128i min = _mm_set1_epi16(-3155);
  int optimised_cols = 0;

  // Load input into __m128i & pack to 16 bits
  for (i = 0; i < 16; i++) {
    temp1 = _mm_loadu_si128((const __m128i *)(input + 16 * i));
    temp2 = _mm_loadu_si128((const __m128i *)(input + 16 * i + 4));
    inptr[i] = _mm_packs_epi32(temp1, temp2);
    temp1 = _mm_loadu_si128((const __m128i *)(input + 16 * i + 8));
    temp2 = _mm_loadu_si128((const __m128i *)(input + 16 * i + 12));
    inptr[i + 16] = _mm_packs_epi32(temp1, temp2);
  }

  // Find the min & max for the row transform
  max_input = _mm_max_epi16(inptr[0], inptr[1]);
  min_input = _mm_min_epi16(inptr[0], inptr[1]);
  for (i = 2; i < 32; i++) {
    max_input = _mm_max_epi16(max_input, inptr[i]);
    min_input = _mm_min_epi16(min_input, inptr[i]);
  }
  max_input = _mm_cmpgt_epi16(max_input, max);
  min_input = _mm_cmplt_epi16(min_input, min);
  temp1 = _mm_or_si128(max_input, min_input);
  test = _mm_movemask_epi8(temp1);

  if (!test) {
    // Do the row transform
    idct16_sse2(inptr, inptr + 16);

    // Find the min & max for the column transform
    max_input = _mm_max_epi16(inptr[0], inptr[1]);
    min_input = _mm_min_epi16(inptr[0], inptr[1]);
    for (i = 2; i < 32; i++) {
      max_input = _mm_max_epi16(max_input, inptr[i]);
      min_input = _mm_min_epi16(min_input, inptr[i]);
    }
    max_input = _mm_cmpgt_epi16(max_input, max);
    min_input = _mm_cmplt_epi16(min_input, min);
    temp1 = _mm_or_si128(max_input, min_input);
    test = _mm_movemask_epi8(temp1);

    if (test) {
      array_transpose_16x16(inptr, inptr + 16);
      for (i = 0; i < 16; i++) {
        sign_bits = _mm_cmplt_epi16(inptr[i], zero);
        temp1 = _mm_unpacklo_epi16(inptr[i], sign_bits);
        temp2 = _mm_unpackhi_epi16(inptr[i], sign_bits);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (i * 4)), temp1);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (i * 4 + 1)), temp2);
        sign_bits = _mm_cmplt_epi16(inptr[i + 16], zero);
        temp1 = _mm_unpacklo_epi16(inptr[i + 16], sign_bits);
        temp2 = _mm_unpackhi_epi16(inptr[i + 16], sign_bits);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (i * 4 + 2)), temp1);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (i * 4 + 3)), temp2);
      }
    } else {
      // Set to use the optimised transform for the column
      optimised_cols = 1;
    }
  } else {
    // Run the un-optimised row transform
    for (i = 0; i < 16; ++i) {
      vpx_highbd_idct16_c(input, outptr, bd);
      input += 16;
      outptr += 16;
    }
  }

  if (optimised_cols) {
    idct16_sse2(inptr, inptr + 16);

    // Final round & shift and Reconstruction and Store
    {
      __m128i d[2];
      for (i = 0; i < 16; i++) {
        inptr[i   ] = _mm_add_epi16(inptr[i   ], rounding);
        inptr[i+16] = _mm_add_epi16(inptr[i+16], rounding);
        d[0] = _mm_loadu_si128((const __m128i *)(dest + stride*i));
        d[1] = _mm_loadu_si128((const __m128i *)(dest + stride*i + 8));
        inptr[i   ] = _mm_srai_epi16(inptr[i   ], 6);
        inptr[i+16] = _mm_srai_epi16(inptr[i+16], 6);
        d[0] = clamp_high_sse2(_mm_add_epi16(d[0], inptr[i   ]), bd);
        d[1] = clamp_high_sse2(_mm_add_epi16(d[1], inptr[i+16]), bd);
        // Store
        _mm_storeu_si128((__m128i *)(dest + stride*i), d[0]);
        _mm_storeu_si128((__m128i *)(dest + stride*i + 8), d[1]);
      }
    }
  } else {
    // Run the un-optimised column transform
    tran_low_t temp_in[16], temp_out[16];
    for (i = 0; i < 16; ++i) {
      for (j = 0; j < 16; ++j)
        temp_in[j] = out[j * 16 + i];
      vpx_highbd_idct16_c(temp_in, temp_out, bd);
      for (j = 0; j < 16; ++j) {
        dest[j * stride + i] = highbd_clip_pixel_add(
            dest[j * stride + i], ROUND_POWER_OF_TWO(temp_out[j], 6), bd);
      }
    }
  }
}

void vpx_highbd_idct16x16_10_add_sse2(const tran_low_t *input, uint8_t *dest8,
                                      int stride, int bd) {
  tran_low_t out[16 * 16] = { 0 };
  tran_low_t *outptr = out;
  int i, j, test;
  __m128i inptr[32];
  __m128i min_input, max_input, temp1, temp2, sign_bits;
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);
  const __m128i zero = _mm_set1_epi16(0);
  const __m128i rounding = _mm_set1_epi16(32);
  const __m128i max = _mm_set1_epi16(3155);
  const __m128i min = _mm_set1_epi16(-3155);
  int optimised_cols = 0;

  // Load input into __m128i & pack to 16 bits
  for (i = 0; i < 16; i++) {
    temp1 = _mm_loadu_si128((const __m128i *)(input + 16 * i));
    temp2 = _mm_loadu_si128((const __m128i *)(input + 16 * i + 4));
    inptr[i] = _mm_packs_epi32(temp1, temp2);
    temp1 = _mm_loadu_si128((const __m128i *)(input + 16 * i + 8));
    temp2 = _mm_loadu_si128((const __m128i *)(input + 16 * i + 12));
    inptr[i + 16] = _mm_packs_epi32(temp1, temp2);
  }

  // Find the min & max for the row transform
  // Since all non-zero dct coefficients are in upper-left 4x4 area,
  // we only need to consider first 4 rows here.
  max_input = _mm_max_epi16(inptr[0], inptr[1]);
  min_input = _mm_min_epi16(inptr[0], inptr[1]);
  for (i = 2; i < 4; i++) {
    max_input = _mm_max_epi16(max_input, inptr[i]);
    min_input = _mm_min_epi16(min_input, inptr[i]);
  }
  max_input = _mm_cmpgt_epi16(max_input, max);
  min_input = _mm_cmplt_epi16(min_input, min);
  temp1 = _mm_or_si128(max_input, min_input);
  test = _mm_movemask_epi8(temp1);

  if (!test) {
    // Do the row transform (N.B. This transposes inptr)
    idct16_sse2(inptr, inptr + 16);

    // Find the min & max for the column transform
    // N.B. Only first 4 cols contain non-zero coeffs
    max_input = _mm_max_epi16(inptr[0], inptr[1]);
    min_input = _mm_min_epi16(inptr[0], inptr[1]);
    for (i = 2; i < 16; i++) {
      max_input = _mm_max_epi16(max_input, inptr[i]);
      min_input = _mm_min_epi16(min_input, inptr[i]);
    }
    max_input = _mm_cmpgt_epi16(max_input, max);
    min_input = _mm_cmplt_epi16(min_input, min);
    temp1 = _mm_or_si128(max_input, min_input);
    test = _mm_movemask_epi8(temp1);

    if (test) {
      // Use fact only first 4 rows contain non-zero coeffs
      array_transpose_8x8(inptr, inptr);
      array_transpose_8x8(inptr + 8, inptr + 16);
      for (i = 0; i < 4; i++) {
        sign_bits = _mm_cmplt_epi16(inptr[i], zero);
        temp1 = _mm_unpacklo_epi16(inptr[i], sign_bits);
        temp2 = _mm_unpackhi_epi16(inptr[i], sign_bits);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (i * 4)), temp1);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (i * 4 + 1)), temp2);
        sign_bits = _mm_cmplt_epi16(inptr[i + 16], zero);
        temp1 = _mm_unpacklo_epi16(inptr[i + 16], sign_bits);
        temp2 = _mm_unpackhi_epi16(inptr[i + 16], sign_bits);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (i * 4 + 2)), temp1);
        _mm_storeu_si128((__m128i *)(outptr + 4 * (i * 4 + 3)), temp2);
      }
    } else {
      // Set to use the optimised transform for the column
      optimised_cols = 1;
    }
  } else {
    // Run the un-optimised row transform
    for (i = 0; i < 4; ++i) {
      vpx_highbd_idct16_c(input, outptr, bd);
      input += 16;
      outptr += 16;
    }
  }

  if (optimised_cols) {
    idct16_sse2(inptr, inptr + 16);

    // Final round & shift and Reconstruction and Store
    {
      __m128i d[2];
      for (i = 0; i < 16; i++) {
        inptr[i   ] = _mm_add_epi16(inptr[i   ], rounding);
        inptr[i+16] = _mm_add_epi16(inptr[i+16], rounding);
        d[0] = _mm_loadu_si128((const __m128i *)(dest + stride*i));
        d[1] = _mm_loadu_si128((const __m128i *)(dest + stride*i + 8));
        inptr[i   ] = _mm_srai_epi16(inptr[i   ], 6);
        inptr[i+16] = _mm_srai_epi16(inptr[i+16], 6);
        d[0] = clamp_high_sse2(_mm_add_epi16(d[0], inptr[i   ]), bd);
        d[1] = clamp_high_sse2(_mm_add_epi16(d[1], inptr[i+16]), bd);
        // Store
        _mm_storeu_si128((__m128i *)(dest + stride*i), d[0]);
        _mm_storeu_si128((__m128i *)(dest + stride*i + 8), d[1]);
      }
    }
  } else {
    // Run the un-optimised column transform
    tran_low_t temp_in[16], temp_out[16];
    for (i = 0; i < 16; ++i) {
      for (j = 0; j < 16; ++j)
        temp_in[j] = out[j * 16 + i];
      vpx_highbd_idct16_c(temp_in, temp_out, bd);
      for (j = 0; j < 16; ++j) {
        dest[j * stride + i] = highbd_clip_pixel_add(
            dest[j * stride + i], ROUND_POWER_OF_TWO(temp_out[j], 6), bd);
      }
    }
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
