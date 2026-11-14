/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>  // SSE2

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/transpose_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

static INLINE void transpose_16bit_4(__m128i *res) {
  const __m128i tr0_0 = _mm_unpacklo_epi16(res[0], res[1]);
  const __m128i tr0_1 = _mm_unpackhi_epi16(res[0], res[1]);

  res[0] = _mm_unpacklo_epi16(tr0_0, tr0_1);
  res[1] = _mm_unpackhi_epi16(tr0_0, tr0_1);
}

void vpx_idct4x4_16_add_sse2(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  const __m128i eight = _mm_set1_epi16(8);
  __m128i in[2];

  // Rows
  in[0] = load_input_data8(input);
  in[1] = load_input_data8(input + 8);
  idct4_sse2(in);

  // Columns
  idct4_sse2(in);

  // Final round and shift
  in[0] = _mm_add_epi16(in[0], eight);
  in[1] = _mm_add_epi16(in[1], eight);
  in[0] = _mm_srai_epi16(in[0], 4);
  in[1] = _mm_srai_epi16(in[1], 4);

  recon_and_store4x4_sse2(in, dest, stride);
}

void vpx_idct4x4_1_add_sse2(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  const __m128i zero = _mm_setzero_si128();
  int a;
  __m128i dc_value, d[2];

  a = (int)dct_const_round_shift((int16_t)input[0] * cospi_16_64);
  a = (int)dct_const_round_shift(a * cospi_16_64);
  a = ROUND_POWER_OF_TWO(a, 4);

  dc_value = _mm_set1_epi16(a);

  // Reconstruction and Store
  d[0] = _mm_cvtsi32_si128(*(const int *)(dest));
  d[1] = _mm_cvtsi32_si128(*(const int *)(dest + stride * 3));
  d[0] = _mm_unpacklo_epi32(d[0],
                            _mm_cvtsi32_si128(*(const int *)(dest + stride)));
  d[1] = _mm_unpacklo_epi32(
      _mm_cvtsi32_si128(*(const int *)(dest + stride * 2)), d[1]);
  d[0] = _mm_unpacklo_epi8(d[0], zero);
  d[1] = _mm_unpacklo_epi8(d[1], zero);
  d[0] = _mm_add_epi16(d[0], dc_value);
  d[1] = _mm_add_epi16(d[1], dc_value);
  d[0] = _mm_packus_epi16(d[0], d[1]);

  *(int *)dest = _mm_cvtsi128_si32(d[0]);
  d[0] = _mm_srli_si128(d[0], 4);
  *(int *)(dest + stride) = _mm_cvtsi128_si32(d[0]);
  d[0] = _mm_srli_si128(d[0], 4);
  *(int *)(dest + stride * 2) = _mm_cvtsi128_si32(d[0]);
  d[0] = _mm_srli_si128(d[0], 4);
  *(int *)(dest + stride * 3) = _mm_cvtsi128_si32(d[0]);
}

void idct4_sse2(__m128i *const in) {
  const __m128i k__cospi_p16_p16 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i k__cospi_p16_m16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_p24_m08 = pair_set_epi16(cospi_24_64, -cospi_8_64);
  const __m128i k__cospi_p08_p24 = pair_set_epi16(cospi_8_64, cospi_24_64);
  __m128i u[2];

  transpose_16bit_4(in);
  // stage 1
  u[0] = _mm_unpacklo_epi16(in[0], in[1]);
  u[1] = _mm_unpackhi_epi16(in[0], in[1]);
  u[0] = idct_calc_wraplow_sse2(k__cospi_p16_p16, k__cospi_p16_m16, u[0]);
  u[1] = idct_calc_wraplow_sse2(k__cospi_p08_p24, k__cospi_p24_m08, u[1]);

  // stage 2
  in[0] = _mm_add_epi16(u[0], u[1]);
  in[1] = _mm_sub_epi16(u[0], u[1]);
  in[1] = _mm_shuffle_epi32(in[1], 0x4E);
}

void iadst4_sse2(__m128i *const in) {
  const __m128i k__sinpi_1_3 = pair_set_epi16(sinpi_1_9, sinpi_3_9);
  const __m128i k__sinpi_4_2 = pair_set_epi16(sinpi_4_9, sinpi_2_9);
  const __m128i k__sinpi_2_3 = pair_set_epi16(sinpi_2_9, sinpi_3_9);
  const __m128i k__sinpi_1_4 = pair_set_epi16(sinpi_1_9, sinpi_4_9);
  const __m128i k__sinpi_12_n3 =
      pair_set_epi16(sinpi_1_9 + sinpi_2_9, -sinpi_3_9);
  __m128i u[4], v[5];

  // 00 01 20 21  02 03 22 23
  // 10 11 30 31  12 13 32 33
  const __m128i tr0_0 = _mm_unpacklo_epi32(in[0], in[1]);
  const __m128i tr0_1 = _mm_unpackhi_epi32(in[0], in[1]);

  // 00 01 10 11  20 21 30 31
  // 02 03 12 13  22 23 32 33
  in[0] = _mm_unpacklo_epi32(tr0_0, tr0_1);
  in[1] = _mm_unpackhi_epi32(tr0_0, tr0_1);

  v[0] = _mm_madd_epi16(in[0], k__sinpi_1_3);    // s_1 * x0 + s_3 * x1
  v[1] = _mm_madd_epi16(in[1], k__sinpi_4_2);    // s_4 * x2 + s_2 * x3
  v[2] = _mm_madd_epi16(in[0], k__sinpi_2_3);    // s_2 * x0 + s_3 * x1
  v[3] = _mm_madd_epi16(in[1], k__sinpi_1_4);    // s_1 * x2 + s_4 * x3
  v[4] = _mm_madd_epi16(in[0], k__sinpi_12_n3);  // (s_1 + s_2) * x0 - s_3 * x1
  in[0] = _mm_sub_epi16(in[0], in[1]);           // x0 - x2
  in[1] = _mm_srli_epi32(in[1], 16);
  in[0] = _mm_add_epi16(in[0], in[1]);
  in[0] = _mm_slli_epi32(in[0], 16);  // x0 - x2 + x3

  u[0] = _mm_add_epi32(v[0], v[1]);
  u[1] = _mm_sub_epi32(v[2], v[3]);
  u[2] = _mm_madd_epi16(in[0], k__sinpi_1_3);
  u[3] = _mm_sub_epi32(v[1], v[3]);
  u[3] = _mm_add_epi32(u[3], v[4]);

  u[0] = dct_const_round_shift_sse2(u[0]);
  u[1] = dct_const_round_shift_sse2(u[1]);
  u[2] = dct_const_round_shift_sse2(u[2]);
  u[3] = dct_const_round_shift_sse2(u[3]);

  in[0] = _mm_packs_epi32(u[0], u[1]);
  in[1] = _mm_packs_epi32(u[2], u[3]);
}

static INLINE void load_buffer_8x8(const tran_low_t *const input,
                                   __m128i *const in) {
  in[0] = load_input_data8(input + 0 * 8);
  in[1] = load_input_data8(input + 1 * 8);
  in[2] = load_input_data8(input + 2 * 8);
  in[3] = load_input_data8(input + 3 * 8);
  in[4] = load_input_data8(input + 4 * 8);
  in[5] = load_input_data8(input + 5 * 8);
  in[6] = load_input_data8(input + 6 * 8);
  in[7] = load_input_data8(input + 7 * 8);
}

void vpx_idct8x8_64_add_sse2(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  __m128i in[8];
  int i;

  // Load input data.
  load_buffer_8x8(input, in);

  // 2-D
  for (i = 0; i < 2; i++) {
    vpx_idct8_sse2(in);
  }

  write_buffer_8x8(in, dest, stride);
}

void vpx_idct8x8_12_add_sse2(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  __m128i io[8];

  io[0] = load_input_data4(input + 0 * 8);
  io[1] = load_input_data4(input + 1 * 8);
  io[2] = load_input_data4(input + 2 * 8);
  io[3] = load_input_data4(input + 3 * 8);

  idct8x8_12_add_kernel_sse2(io);
  write_buffer_8x8(io, dest, stride);
}

static INLINE void recon_and_store_8_dual(uint8_t *const dest,
                                          const __m128i in_x,
                                          const int stride) {
  const __m128i zero = _mm_setzero_si128();
  __m128i d0, d1;

  d0 = _mm_loadl_epi64((__m128i *)(dest + 0 * stride));
  d1 = _mm_loadl_epi64((__m128i *)(dest + 1 * stride));
  d0 = _mm_unpacklo_epi8(d0, zero);
  d1 = _mm_unpacklo_epi8(d1, zero);
  d0 = _mm_add_epi16(in_x, d0);
  d1 = _mm_add_epi16(in_x, d1);
  d0 = _mm_packus_epi16(d0, d1);
  _mm_storel_epi64((__m128i *)(dest + 0 * stride), d0);
  _mm_storeh_pi((__m64 *)(dest + 1 * stride), _mm_castsi128_ps(d0));
}

void vpx_idct8x8_1_add_sse2(const tran_low_t *input, uint8_t *dest,
                            int stride) {
  __m128i dc_value;
  tran_high_t a1;
  tran_low_t out =
      WRAPLOW(dct_const_round_shift((int16_t)input[0] * cospi_16_64));

  out = WRAPLOW(dct_const_round_shift(out * cospi_16_64));
  a1 = ROUND_POWER_OF_TWO(out, 5);
  dc_value = _mm_set1_epi16((int16_t)a1);

  recon_and_store_8_dual(dest, dc_value, stride);
  dest += 2 * stride;
  recon_and_store_8_dual(dest, dc_value, stride);
  dest += 2 * stride;
  recon_and_store_8_dual(dest, dc_value, stride);
  dest += 2 * stride;
  recon_and_store_8_dual(dest, dc_value, stride);
}

void vpx_idct8_sse2(__m128i *const in) {
  // 8x8 Transpose is copied from vpx_fdct8x8_sse2()
  transpose_16bit_8x8(in, in);

  // 4-stage 1D idct8x8
  idct8(in, in);
}

void iadst8_sse2(__m128i *const in) {
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
  const __m128i k__cospi_p16_p16 = _mm_set1_epi16(cospi_16_64);
  const __m128i kZero = _mm_setzero_si128();
  __m128i s[8], u[16], v[8], w[16];

  // transpose
  transpose_16bit_8x8(in, in);

  // column transformation
  // stage 1
  // interleave and multiply/add into 32-bit integer
  s[0] = _mm_unpacklo_epi16(in[7], in[0]);
  s[1] = _mm_unpackhi_epi16(in[7], in[0]);
  s[2] = _mm_unpacklo_epi16(in[5], in[2]);
  s[3] = _mm_unpackhi_epi16(in[5], in[2]);
  s[4] = _mm_unpacklo_epi16(in[3], in[4]);
  s[5] = _mm_unpackhi_epi16(in[3], in[4]);
  s[6] = _mm_unpacklo_epi16(in[1], in[6]);
  s[7] = _mm_unpackhi_epi16(in[1], in[6]);

  u[0] = _mm_madd_epi16(s[0], k__cospi_p02_p30);
  u[1] = _mm_madd_epi16(s[1], k__cospi_p02_p30);
  u[2] = _mm_madd_epi16(s[0], k__cospi_p30_m02);
  u[3] = _mm_madd_epi16(s[1], k__cospi_p30_m02);
  u[4] = _mm_madd_epi16(s[2], k__cospi_p10_p22);
  u[5] = _mm_madd_epi16(s[3], k__cospi_p10_p22);
  u[6] = _mm_madd_epi16(s[2], k__cospi_p22_m10);
  u[7] = _mm_madd_epi16(s[3], k__cospi_p22_m10);
  u[8] = _mm_madd_epi16(s[4], k__cospi_p18_p14);
  u[9] = _mm_madd_epi16(s[5], k__cospi_p18_p14);
  u[10] = _mm_madd_epi16(s[4], k__cospi_p14_m18);
  u[11] = _mm_madd_epi16(s[5], k__cospi_p14_m18);
  u[12] = _mm_madd_epi16(s[6], k__cospi_p26_p06);
  u[13] = _mm_madd_epi16(s[7], k__cospi_p26_p06);
  u[14] = _mm_madd_epi16(s[6], k__cospi_p06_m26);
  u[15] = _mm_madd_epi16(s[7], k__cospi_p06_m26);

  // addition
  w[0] = _mm_add_epi32(u[0], u[8]);
  w[1] = _mm_add_epi32(u[1], u[9]);
  w[2] = _mm_add_epi32(u[2], u[10]);
  w[3] = _mm_add_epi32(u[3], u[11]);
  w[4] = _mm_add_epi32(u[4], u[12]);
  w[5] = _mm_add_epi32(u[5], u[13]);
  w[6] = _mm_add_epi32(u[6], u[14]);
  w[7] = _mm_add_epi32(u[7], u[15]);
  w[8] = _mm_sub_epi32(u[0], u[8]);
  w[9] = _mm_sub_epi32(u[1], u[9]);
  w[10] = _mm_sub_epi32(u[2], u[10]);
  w[11] = _mm_sub_epi32(u[3], u[11]);
  w[12] = _mm_sub_epi32(u[4], u[12]);
  w[13] = _mm_sub_epi32(u[5], u[13]);
  w[14] = _mm_sub_epi32(u[6], u[14]);
  w[15] = _mm_sub_epi32(u[7], u[15]);

  // shift and rounding
  u[0] = dct_const_round_shift_sse2(w[0]);
  u[1] = dct_const_round_shift_sse2(w[1]);
  u[2] = dct_const_round_shift_sse2(w[2]);
  u[3] = dct_const_round_shift_sse2(w[3]);
  u[4] = dct_const_round_shift_sse2(w[4]);
  u[5] = dct_const_round_shift_sse2(w[5]);
  u[6] = dct_const_round_shift_sse2(w[6]);
  u[7] = dct_const_round_shift_sse2(w[7]);
  u[8] = dct_const_round_shift_sse2(w[8]);
  u[9] = dct_const_round_shift_sse2(w[9]);
  u[10] = dct_const_round_shift_sse2(w[10]);
  u[11] = dct_const_round_shift_sse2(w[11]);
  u[12] = dct_const_round_shift_sse2(w[12]);
  u[13] = dct_const_round_shift_sse2(w[13]);
  u[14] = dct_const_round_shift_sse2(w[14]);
  u[15] = dct_const_round_shift_sse2(w[15]);

  // back to 16-bit and pack 8 integers into __m128i
  in[0] = _mm_packs_epi32(u[0], u[1]);
  in[1] = _mm_packs_epi32(u[2], u[3]);
  in[2] = _mm_packs_epi32(u[4], u[5]);
  in[3] = _mm_packs_epi32(u[6], u[7]);
  in[4] = _mm_packs_epi32(u[8], u[9]);
  in[5] = _mm_packs_epi32(u[10], u[11]);
  in[6] = _mm_packs_epi32(u[12], u[13]);
  in[7] = _mm_packs_epi32(u[14], u[15]);

  // stage 2
  s[0] = _mm_add_epi16(in[0], in[2]);
  s[1] = _mm_add_epi16(in[1], in[3]);
  s[2] = _mm_sub_epi16(in[0], in[2]);
  s[3] = _mm_sub_epi16(in[1], in[3]);
  u[0] = _mm_unpacklo_epi16(in[4], in[5]);
  u[1] = _mm_unpackhi_epi16(in[4], in[5]);
  u[2] = _mm_unpacklo_epi16(in[6], in[7]);
  u[3] = _mm_unpackhi_epi16(in[6], in[7]);

  v[0] = _mm_madd_epi16(u[0], k__cospi_p08_p24);
  v[1] = _mm_madd_epi16(u[1], k__cospi_p08_p24);
  v[2] = _mm_madd_epi16(u[0], k__cospi_p24_m08);
  v[3] = _mm_madd_epi16(u[1], k__cospi_p24_m08);
  v[4] = _mm_madd_epi16(u[2], k__cospi_m24_p08);
  v[5] = _mm_madd_epi16(u[3], k__cospi_m24_p08);
  v[6] = _mm_madd_epi16(u[2], k__cospi_p08_p24);
  v[7] = _mm_madd_epi16(u[3], k__cospi_p08_p24);

  w[0] = _mm_add_epi32(v[0], v[4]);
  w[1] = _mm_add_epi32(v[1], v[5]);
  w[2] = _mm_add_epi32(v[2], v[6]);
  w[3] = _mm_add_epi32(v[3], v[7]);
  w[4] = _mm_sub_epi32(v[0], v[4]);
  w[5] = _mm_sub_epi32(v[1], v[5]);
  w[6] = _mm_sub_epi32(v[2], v[6]);
  w[7] = _mm_sub_epi32(v[3], v[7]);

  u[0] = dct_const_round_shift_sse2(w[0]);
  u[1] = dct_const_round_shift_sse2(w[1]);
  u[2] = dct_const_round_shift_sse2(w[2]);
  u[3] = dct_const_round_shift_sse2(w[3]);
  u[4] = dct_const_round_shift_sse2(w[4]);
  u[5] = dct_const_round_shift_sse2(w[5]);
  u[6] = dct_const_round_shift_sse2(w[6]);
  u[7] = dct_const_round_shift_sse2(w[7]);

  // back to 16-bit intergers
  s[4] = _mm_packs_epi32(u[0], u[1]);
  s[5] = _mm_packs_epi32(u[2], u[3]);
  s[6] = _mm_packs_epi32(u[4], u[5]);
  s[7] = _mm_packs_epi32(u[6], u[7]);

  // stage 3
  u[0] = _mm_unpacklo_epi16(s[2], s[3]);
  u[1] = _mm_unpackhi_epi16(s[2], s[3]);
  u[2] = _mm_unpacklo_epi16(s[6], s[7]);
  u[3] = _mm_unpackhi_epi16(s[6], s[7]);

  s[2] = idct_calc_wraplow_sse2(u[0], u[1], k__cospi_p16_p16);
  s[3] = idct_calc_wraplow_sse2(u[0], u[1], k__cospi_p16_m16);
  s[6] = idct_calc_wraplow_sse2(u[2], u[3], k__cospi_p16_p16);
  s[7] = idct_calc_wraplow_sse2(u[2], u[3], k__cospi_p16_m16);

  in[0] = s[0];
  in[1] = _mm_sub_epi16(kZero, s[4]);
  in[2] = s[6];
  in[3] = _mm_sub_epi16(kZero, s[2]);
  in[4] = s[3];
  in[5] = _mm_sub_epi16(kZero, s[7]);
  in[6] = s[5];
  in[7] = _mm_sub_epi16(kZero, s[1]);
}

static INLINE void idct16_load8x8(const tran_low_t *const input,
                                  __m128i *const in) {
  in[0] = load_input_data8(input + 0 * 16);
  in[1] = load_input_data8(input + 1 * 16);
  in[2] = load_input_data8(input + 2 * 16);
  in[3] = load_input_data8(input + 3 * 16);
  in[4] = load_input_data8(input + 4 * 16);
  in[5] = load_input_data8(input + 5 * 16);
  in[6] = load_input_data8(input + 6 * 16);
  in[7] = load_input_data8(input + 7 * 16);
}

void vpx_idct16x16_256_add_sse2(const tran_low_t *input, uint8_t *dest,
                                int stride) {
  __m128i l[16], r[16], out[16], *in;
  int i;

  in = l;
  for (i = 0; i < 2; i++) {
    idct16_load8x8(input, in);
    transpose_16bit_8x8(in, in);
    idct16_load8x8(input + 8, in + 8);
    transpose_16bit_8x8(in + 8, in + 8);
    idct16_8col(in, in);
    in = r;
    input += 128;
  }

  for (i = 0; i < 16; i += 8) {
    int j;
    transpose_16bit_8x8(l + i, out);
    transpose_16bit_8x8(r + i, out + 8);
    idct16_8col(out, out);

    for (j = 0; j < 16; ++j) {
      write_buffer_8x1(dest + j * stride, out[j]);
    }

    dest += 8;
  }
}

void vpx_idct16x16_38_add_sse2(const tran_low_t *input, uint8_t *dest,
                               int stride) {
  __m128i in[16], temp[16], out[16];
  int i;

  idct16_load8x8(input, in);
  transpose_16bit_8x8(in, in);

  for (i = 8; i < 16; i++) {
    in[i] = _mm_setzero_si128();
  }
  idct16_8col(in, temp);

  for (i = 0; i < 16; i += 8) {
    int j;
    transpose_16bit_8x8(temp + i, in);
    idct16_8col(in, out);

    for (j = 0; j < 16; ++j) {
      write_buffer_8x1(dest + j * stride, out[j]);
    }

    dest += 8;
  }
}

void vpx_idct16x16_10_add_sse2(const tran_low_t *input, uint8_t *dest,
                               int stride) {
  __m128i in[16], l[16];
  int i;

  // First 1-D inverse DCT
  // Load input data.
  in[0] = load_input_data4(input + 0 * 16);
  in[1] = load_input_data4(input + 1 * 16);
  in[2] = load_input_data4(input + 2 * 16);
  in[3] = load_input_data4(input + 3 * 16);

  idct16x16_10_pass1(in, l);

  // Second 1-D inverse transform, performed per 8x16 block
  for (i = 0; i < 16; i += 8) {
    int j;
    idct16x16_10_pass2(l + i, in);

    for (j = 0; j < 16; ++j) {
      write_buffer_8x1(dest + j * stride, in[j]);
    }

    dest += 8;
  }
}

static INLINE void recon_and_store_16(uint8_t *const dest, const __m128i in_x) {
  const __m128i zero = _mm_setzero_si128();
  __m128i d0, d1;

  d0 = _mm_load_si128((__m128i *)(dest));
  d1 = _mm_unpackhi_epi8(d0, zero);
  d0 = _mm_unpacklo_epi8(d0, zero);
  d0 = _mm_add_epi16(in_x, d0);
  d1 = _mm_add_epi16(in_x, d1);
  d0 = _mm_packus_epi16(d0, d1);
  _mm_store_si128((__m128i *)(dest), d0);
}

void vpx_idct16x16_1_add_sse2(const tran_low_t *input, uint8_t *dest,
                              int stride) {
  __m128i dc_value;
  int i;
  tran_high_t a1;
  tran_low_t out =
      WRAPLOW(dct_const_round_shift((int16_t)input[0] * cospi_16_64));

  out = WRAPLOW(dct_const_round_shift(out * cospi_16_64));
  a1 = ROUND_POWER_OF_TWO(out, 6);
  dc_value = _mm_set1_epi16((int16_t)a1);

  for (i = 0; i < 16; ++i) {
    recon_and_store_16(dest, dc_value);
    dest += stride;
  }
}

void vpx_iadst16_8col_sse2(__m128i *const in) {
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
  const __m128i k__cospi_m16_m16 = _mm_set1_epi16(-cospi_16_64);
  const __m128i k__cospi_p16_p16 = _mm_set1_epi16(cospi_16_64);
  const __m128i k__cospi_p16_m16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_m16_p16 = pair_set_epi16(-cospi_16_64, cospi_16_64);
  const __m128i kZero = _mm_setzero_si128();

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

  u[0] = dct_const_round_shift_sse2(u[0]);
  u[1] = dct_const_round_shift_sse2(u[1]);
  u[2] = dct_const_round_shift_sse2(u[2]);
  u[3] = dct_const_round_shift_sse2(u[3]);
  u[4] = dct_const_round_shift_sse2(u[4]);
  u[5] = dct_const_round_shift_sse2(u[5]);
  u[6] = dct_const_round_shift_sse2(u[6]);
  u[7] = dct_const_round_shift_sse2(u[7]);
  u[8] = dct_const_round_shift_sse2(u[8]);
  u[9] = dct_const_round_shift_sse2(u[9]);
  u[10] = dct_const_round_shift_sse2(u[10]);
  u[11] = dct_const_round_shift_sse2(u[11]);
  u[12] = dct_const_round_shift_sse2(u[12]);
  u[13] = dct_const_round_shift_sse2(u[13]);
  u[14] = dct_const_round_shift_sse2(u[14]);
  u[15] = dct_const_round_shift_sse2(u[15]);
  u[16] = dct_const_round_shift_sse2(u[16]);
  u[17] = dct_const_round_shift_sse2(u[17]);
  u[18] = dct_const_round_shift_sse2(u[18]);
  u[19] = dct_const_round_shift_sse2(u[19]);
  u[20] = dct_const_round_shift_sse2(u[20]);
  u[21] = dct_const_round_shift_sse2(u[21]);
  u[22] = dct_const_round_shift_sse2(u[22]);
  u[23] = dct_const_round_shift_sse2(u[23]);
  u[24] = dct_const_round_shift_sse2(u[24]);
  u[25] = dct_const_round_shift_sse2(u[25]);
  u[26] = dct_const_round_shift_sse2(u[26]);
  u[27] = dct_const_round_shift_sse2(u[27]);
  u[28] = dct_const_round_shift_sse2(u[28]);
  u[29] = dct_const_round_shift_sse2(u[29]);
  u[30] = dct_const_round_shift_sse2(u[30]);
  u[31] = dct_const_round_shift_sse2(u[31]);

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

  u[0] = dct_const_round_shift_sse2(u[0]);
  u[1] = dct_const_round_shift_sse2(u[1]);
  u[2] = dct_const_round_shift_sse2(u[2]);
  u[3] = dct_const_round_shift_sse2(u[3]);
  u[4] = dct_const_round_shift_sse2(u[4]);
  u[5] = dct_const_round_shift_sse2(u[5]);
  u[6] = dct_const_round_shift_sse2(u[6]);
  u[7] = dct_const_round_shift_sse2(u[7]);
  u[8] = dct_const_round_shift_sse2(u[8]);
  u[9] = dct_const_round_shift_sse2(u[9]);
  u[10] = dct_const_round_shift_sse2(u[10]);
  u[11] = dct_const_round_shift_sse2(u[11]);
  u[12] = dct_const_round_shift_sse2(u[12]);
  u[13] = dct_const_round_shift_sse2(u[13]);
  u[14] = dct_const_round_shift_sse2(u[14]);
  u[15] = dct_const_round_shift_sse2(u[15]);

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

  v[0] = dct_const_round_shift_sse2(u[0]);
  v[1] = dct_const_round_shift_sse2(u[1]);
  v[2] = dct_const_round_shift_sse2(u[2]);
  v[3] = dct_const_round_shift_sse2(u[3]);
  v[4] = dct_const_round_shift_sse2(u[4]);
  v[5] = dct_const_round_shift_sse2(u[5]);
  v[6] = dct_const_round_shift_sse2(u[6]);
  v[7] = dct_const_round_shift_sse2(u[7]);
  v[8] = dct_const_round_shift_sse2(u[8]);
  v[9] = dct_const_round_shift_sse2(u[9]);
  v[10] = dct_const_round_shift_sse2(u[10]);
  v[11] = dct_const_round_shift_sse2(u[11]);
  v[12] = dct_const_round_shift_sse2(u[12]);
  v[13] = dct_const_round_shift_sse2(u[13]);
  v[14] = dct_const_round_shift_sse2(u[14]);
  v[15] = dct_const_round_shift_sse2(u[15]);

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

  in[7] = idct_calc_wraplow_sse2(u[0], u[1], k__cospi_m16_m16);
  in[8] = idct_calc_wraplow_sse2(u[0], u[1], k__cospi_p16_m16);
  in[4] = idct_calc_wraplow_sse2(u[2], u[3], k__cospi_p16_p16);
  in[11] = idct_calc_wraplow_sse2(u[2], u[3], k__cospi_m16_p16);
  in[6] = idct_calc_wraplow_sse2(u[4], u[5], k__cospi_p16_p16);
  in[9] = idct_calc_wraplow_sse2(u[4], u[5], k__cospi_m16_p16);
  in[5] = idct_calc_wraplow_sse2(u[6], u[7], k__cospi_m16_m16);
  in[10] = idct_calc_wraplow_sse2(u[6], u[7], k__cospi_p16_m16);

  in[0] = s[0];
  in[1] = _mm_sub_epi16(kZero, s[8]);
  in[2] = s[12];
  in[3] = _mm_sub_epi16(kZero, s[4]);
  in[12] = s[5];
  in[13] = _mm_sub_epi16(kZero, s[13]);
  in[14] = s[9];
  in[15] = _mm_sub_epi16(kZero, s[1]);
}

void idct16_sse2(__m128i *const in0, __m128i *const in1) {
  transpose_16bit_16x16(in0, in1);
  idct16_8col(in0, in0);
  idct16_8col(in1, in1);
}

void iadst16_sse2(__m128i *const in0, __m128i *const in1) {
  transpose_16bit_16x16(in0, in1);
  vpx_iadst16_8col_sse2(in0);
  vpx_iadst16_8col_sse2(in1);
}

// Group the coefficient calculation into smaller functions to prevent stack
// spillover in 32x32 idct optimizations:
// quarter_1: 0-7
// quarter_2: 8-15
// quarter_3_4: 16-23, 24-31

// For each 8x32 block __m128i in[32],
// Input with index, 0, 4
// output pixels: 0-7 in __m128i out[32]
static INLINE void idct32_34_8x32_quarter_1(const __m128i *const in /*in[32]*/,
                                            __m128i *const out /*out[8]*/) {
  const __m128i zero = _mm_setzero_si128();
  __m128i step1[8], step2[8];

  // stage 3
  butterfly(in[4], zero, cospi_28_64, cospi_4_64, &step1[4], &step1[7]);

  // stage 4
  step2[0] = butterfly_cospi16(in[0]);
  step2[4] = step1[4];
  step2[5] = step1[4];
  step2[6] = step1[7];
  step2[7] = step1[7];

  // stage 5
  step1[0] = step2[0];
  step1[1] = step2[0];
  step1[2] = step2[0];
  step1[3] = step2[0];
  step1[4] = step2[4];
  butterfly(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5], &step1[6]);
  step1[7] = step2[7];

  // stage 6
  out[0] = _mm_add_epi16(step1[0], step1[7]);
  out[1] = _mm_add_epi16(step1[1], step1[6]);
  out[2] = _mm_add_epi16(step1[2], step1[5]);
  out[3] = _mm_add_epi16(step1[3], step1[4]);
  out[4] = _mm_sub_epi16(step1[3], step1[4]);
  out[5] = _mm_sub_epi16(step1[2], step1[5]);
  out[6] = _mm_sub_epi16(step1[1], step1[6]);
  out[7] = _mm_sub_epi16(step1[0], step1[7]);
}

// For each 8x32 block __m128i in[32],
// Input with index, 2, 6
// output pixels: 8-15 in __m128i out[32]
static INLINE void idct32_34_8x32_quarter_2(const __m128i *const in /*in[32]*/,
                                            __m128i *const out /*out[16]*/) {
  const __m128i zero = _mm_setzero_si128();
  __m128i step1[16], step2[16];

  // stage 2
  butterfly(in[2], zero, cospi_30_64, cospi_2_64, &step2[8], &step2[15]);
  butterfly(zero, in[6], cospi_6_64, cospi_26_64, &step2[11], &step2[12]);

  // stage 3
  step1[8] = step2[8];
  step1[9] = step2[8];
  step1[14] = step2[15];
  step1[15] = step2[15];
  step1[10] = step2[11];
  step1[11] = step2[11];
  step1[12] = step2[12];
  step1[13] = step2[12];

  idct32_8x32_quarter_2_stage_4_to_6(step1, out);
}

static INLINE void idct32_34_8x32_quarter_1_2(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i temp[16];
  idct32_34_8x32_quarter_1(in, temp);
  idct32_34_8x32_quarter_2(in, temp);
  // stage 7
  add_sub_butterfly(temp, out, 16);
}

// For each 8x32 block __m128i in[32],
// Input with odd index, 1, 3, 5, 7
// output pixels: 16-23, 24-31 in __m128i out[32]
static INLINE void idct32_34_8x32_quarter_3_4(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  const __m128i zero = _mm_setzero_si128();
  __m128i step1[32];

  // stage 1
  butterfly(in[1], zero, cospi_31_64, cospi_1_64, &step1[16], &step1[31]);
  butterfly(zero, in[7], cospi_7_64, cospi_25_64, &step1[19], &step1[28]);
  butterfly(in[5], zero, cospi_27_64, cospi_5_64, &step1[20], &step1[27]);
  butterfly(zero, in[3], cospi_3_64, cospi_29_64, &step1[23], &step1[24]);

  // stage 3
  butterfly(step1[31], step1[16], cospi_28_64, cospi_4_64, &step1[17],
            &step1[30]);
  butterfly(step1[28], step1[19], -cospi_4_64, cospi_28_64, &step1[18],
            &step1[29]);
  butterfly(step1[27], step1[20], cospi_12_64, cospi_20_64, &step1[21],
            &step1[26]);
  butterfly(step1[24], step1[23], -cospi_20_64, cospi_12_64, &step1[22],
            &step1[25]);

  idct32_8x32_quarter_3_4_stage_4_to_7(step1, out);
}

void idct32_34_8x32_sse2(const __m128i *const in /*in[32]*/,
                         __m128i *const out /*out[32]*/) {
  __m128i temp[32];

  idct32_34_8x32_quarter_1_2(in, temp);
  idct32_34_8x32_quarter_3_4(in, temp);
  // final stage
  add_sub_butterfly(temp, out, 32);
}

// Only upper-left 8x8 has non-zero coeff
void vpx_idct32x32_34_add_sse2(const tran_low_t *input, uint8_t *dest,
                               int stride) {
  __m128i io[32], col[32];
  int i;

  // Load input data. Only need to load the top left 8x8 block.
  load_transpose_16bit_8x8(input, 32, io);
  idct32_34_8x32_sse2(io, col);

  for (i = 0; i < 32; i += 8) {
    int j;
    transpose_16bit_8x8(col + i, io);
    idct32_34_8x32_sse2(io, io);

    for (j = 0; j < 32; ++j) {
      write_buffer_8x1(dest + j * stride, io[j]);
    }

    dest += 8;
  }
}

// For each 8x32 block __m128i in[32],
// Input with index, 0, 4, 8, 12, 16, 20, 24, 28
// output pixels: 0-7 in __m128i out[32]
static INLINE void idct32_1024_8x32_quarter_1(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[8]*/) {
  __m128i step1[8], step2[8];

  // stage 3
  butterfly(in[4], in[28], cospi_28_64, cospi_4_64, &step1[4], &step1[7]);
  butterfly(in[20], in[12], cospi_12_64, cospi_20_64, &step1[5], &step1[6]);

  // stage 4
  butterfly(in[0], in[16], cospi_16_64, cospi_16_64, &step2[1], &step2[0]);
  butterfly(in[8], in[24], cospi_24_64, cospi_8_64, &step2[2], &step2[3]);
  step2[4] = _mm_add_epi16(step1[4], step1[5]);
  step2[5] = _mm_sub_epi16(step1[4], step1[5]);
  step2[6] = _mm_sub_epi16(step1[7], step1[6]);
  step2[7] = _mm_add_epi16(step1[7], step1[6]);

  // stage 5
  step1[0] = _mm_add_epi16(step2[0], step2[3]);
  step1[1] = _mm_add_epi16(step2[1], step2[2]);
  step1[2] = _mm_sub_epi16(step2[1], step2[2]);
  step1[3] = _mm_sub_epi16(step2[0], step2[3]);
  step1[4] = step2[4];
  butterfly(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5], &step1[6]);
  step1[7] = step2[7];

  // stage 6
  out[0] = _mm_add_epi16(step1[0], step1[7]);
  out[1] = _mm_add_epi16(step1[1], step1[6]);
  out[2] = _mm_add_epi16(step1[2], step1[5]);
  out[3] = _mm_add_epi16(step1[3], step1[4]);
  out[4] = _mm_sub_epi16(step1[3], step1[4]);
  out[5] = _mm_sub_epi16(step1[2], step1[5]);
  out[6] = _mm_sub_epi16(step1[1], step1[6]);
  out[7] = _mm_sub_epi16(step1[0], step1[7]);
}

// For each 8x32 block __m128i in[32],
// Input with index, 2, 6, 10, 14, 18, 22, 26, 30
// output pixels: 8-15 in __m128i out[32]
static INLINE void idct32_1024_8x32_quarter_2(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[16]*/) {
  __m128i step1[16], step2[16];

  // stage 2
  butterfly(in[2], in[30], cospi_30_64, cospi_2_64, &step2[8], &step2[15]);
  butterfly(in[18], in[14], cospi_14_64, cospi_18_64, &step2[9], &step2[14]);
  butterfly(in[10], in[22], cospi_22_64, cospi_10_64, &step2[10], &step2[13]);
  butterfly(in[26], in[6], cospi_6_64, cospi_26_64, &step2[11], &step2[12]);

  // stage 3
  step1[8] = _mm_add_epi16(step2[8], step2[9]);
  step1[9] = _mm_sub_epi16(step2[8], step2[9]);
  step1[10] = _mm_sub_epi16(step2[11], step2[10]);
  step1[11] = _mm_add_epi16(step2[11], step2[10]);
  step1[12] = _mm_add_epi16(step2[12], step2[13]);
  step1[13] = _mm_sub_epi16(step2[12], step2[13]);
  step1[14] = _mm_sub_epi16(step2[15], step2[14]);
  step1[15] = _mm_add_epi16(step2[15], step2[14]);

  idct32_8x32_quarter_2_stage_4_to_6(step1, out);
}

static INLINE void idct32_1024_8x32_quarter_1_2(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i temp[16];
  idct32_1024_8x32_quarter_1(in, temp);
  idct32_1024_8x32_quarter_2(in, temp);
  // stage 7
  add_sub_butterfly(temp, out, 16);
}

// For each 8x32 block __m128i in[32],
// Input with odd index,
// 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
// output pixels: 16-23, 24-31 in __m128i out[32]
static INLINE void idct32_1024_8x32_quarter_3_4(
    const __m128i *const in /*in[32]*/, __m128i *const out /*out[32]*/) {
  __m128i step1[32], step2[32];

  // stage 1
  butterfly(in[1], in[31], cospi_31_64, cospi_1_64, &step1[16], &step1[31]);
  butterfly(in[17], in[15], cospi_15_64, cospi_17_64, &step1[17], &step1[30]);
  butterfly(in[9], in[23], cospi_23_64, cospi_9_64, &step1[18], &step1[29]);
  butterfly(in[25], in[7], cospi_7_64, cospi_25_64, &step1[19], &step1[28]);

  butterfly(in[5], in[27], cospi_27_64, cospi_5_64, &step1[20], &step1[27]);
  butterfly(in[21], in[11], cospi_11_64, cospi_21_64, &step1[21], &step1[26]);

  butterfly(in[13], in[19], cospi_19_64, cospi_13_64, &step1[22], &step1[25]);
  butterfly(in[29], in[3], cospi_3_64, cospi_29_64, &step1[23], &step1[24]);

  // stage 2
  step2[16] = _mm_add_epi16(step1[16], step1[17]);
  step2[17] = _mm_sub_epi16(step1[16], step1[17]);
  step2[18] = _mm_sub_epi16(step1[19], step1[18]);
  step2[19] = _mm_add_epi16(step1[19], step1[18]);
  step2[20] = _mm_add_epi16(step1[20], step1[21]);
  step2[21] = _mm_sub_epi16(step1[20], step1[21]);
  step2[22] = _mm_sub_epi16(step1[23], step1[22]);
  step2[23] = _mm_add_epi16(step1[23], step1[22]);

  step2[24] = _mm_add_epi16(step1[24], step1[25]);
  step2[25] = _mm_sub_epi16(step1[24], step1[25]);
  step2[26] = _mm_sub_epi16(step1[27], step1[26]);
  step2[27] = _mm_add_epi16(step1[27], step1[26]);
  step2[28] = _mm_add_epi16(step1[28], step1[29]);
  step2[29] = _mm_sub_epi16(step1[28], step1[29]);
  step2[30] = _mm_sub_epi16(step1[31], step1[30]);
  step2[31] = _mm_add_epi16(step1[31], step1[30]);

  // stage 3
  step1[16] = step2[16];
  step1[31] = step2[31];
  butterfly(step2[30], step2[17], cospi_28_64, cospi_4_64, &step1[17],
            &step1[30]);
  butterfly(step2[29], step2[18], -cospi_4_64, cospi_28_64, &step1[18],
            &step1[29]);
  step1[19] = step2[19];
  step1[20] = step2[20];
  butterfly(step2[26], step2[21], cospi_12_64, cospi_20_64, &step1[21],
            &step1[26]);
  butterfly(step2[25], step2[22], -cospi_20_64, cospi_12_64, &step1[22],
            &step1[25]);
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[27] = step2[27];
  step1[28] = step2[28];

  idct32_8x32_quarter_3_4_stage_4_to_7(step1, out);
}

void idct32_1024_8x32(const __m128i *const in /*in[32]*/,
                      __m128i *const out /*out[32]*/) {
  __m128i temp[32];

  idct32_1024_8x32_quarter_1_2(in, temp);
  idct32_1024_8x32_quarter_3_4(in, temp);
  // final stage
  add_sub_butterfly(temp, out, 32);
}

void vpx_idct32x32_1024_add_sse2(const tran_low_t *input, uint8_t *dest,
                                 int stride) {
  __m128i col[4][32], io[32];
  int i;

  // rows
  for (i = 0; i < 4; i++) {
    load_transpose_16bit_8x8(&input[0], 32, &io[0]);
    load_transpose_16bit_8x8(&input[8], 32, &io[8]);
    load_transpose_16bit_8x8(&input[16], 32, &io[16]);
    load_transpose_16bit_8x8(&input[24], 32, &io[24]);
    idct32_1024_8x32(io, col[i]);
    input += 32 << 3;
  }

  // columns
  for (i = 0; i < 32; i += 8) {
    // Transpose 32x8 block to 8x32 block
    transpose_16bit_8x8(col[0] + i, io);
    transpose_16bit_8x8(col[1] + i, io + 8);
    transpose_16bit_8x8(col[2] + i, io + 16);
    transpose_16bit_8x8(col[3] + i, io + 24);

    idct32_1024_8x32(io, io);
    store_buffer_8x32(io, dest, stride);
    dest += 8;
  }
}

void vpx_idct32x32_135_add_sse2(const tran_low_t *input, uint8_t *dest,
                                int stride) {
  __m128i col[2][32], in[32], out[32];
  int i;

  for (i = 16; i < 32; i++) {
    in[i] = _mm_setzero_si128();
  }

  // rows
  for (i = 0; i < 2; i++) {
    load_transpose_16bit_8x8(&input[0], 32, &in[0]);
    load_transpose_16bit_8x8(&input[8], 32, &in[8]);
    idct32_1024_8x32(in, col[i]);
    input += 32 << 3;
  }

  // columns
  for (i = 0; i < 32; i += 8) {
    transpose_16bit_8x8(col[0] + i, in);
    transpose_16bit_8x8(col[1] + i, in + 8);
    idct32_1024_8x32(in, out);
    store_buffer_8x32(out, dest, stride);
    dest += 8;
  }
}

void vpx_idct32x32_1_add_sse2(const tran_low_t *input, uint8_t *dest,
                              int stride) {
  __m128i dc_value;
  int j;
  tran_high_t a1;
  tran_low_t out =
      WRAPLOW(dct_const_round_shift((int16_t)input[0] * cospi_16_64));

  out = WRAPLOW(dct_const_round_shift(out * cospi_16_64));
  a1 = ROUND_POWER_OF_TWO(out, 6);
  dc_value = _mm_set1_epi16((int16_t)a1);

  for (j = 0; j < 32; ++j) {
    recon_and_store_16(dest + j * stride + 0, dc_value);
    recon_and_store_16(dest + j * stride + 16, dc_value);
  }
}
