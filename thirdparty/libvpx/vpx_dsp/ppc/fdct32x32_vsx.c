/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx_dsp/ppc/transpose_vsx.h"
#include "vpx_dsp/ppc/txfm_common_vsx.h"
#include "vpx_dsp/ppc/types_vsx.h"

// Returns ((a +/- b) * cospi16 + (2 << 13)) >> 14.
static INLINE void single_butterfly(int16x8_t a, int16x8_t b, int16x8_t *add,
                                    int16x8_t *sub) {
  // Since a + b can overflow 16 bits, the multiplication is distributed
  // (a * c +/- b * c).
  const int32x4_t ac_e = vec_mule(a, cospi16_v);
  const int32x4_t ac_o = vec_mulo(a, cospi16_v);
  const int32x4_t bc_e = vec_mule(b, cospi16_v);
  const int32x4_t bc_o = vec_mulo(b, cospi16_v);

  // Reuse the same multiplies for sum and difference.
  const int32x4_t sum_e = vec_add(ac_e, bc_e);
  const int32x4_t sum_o = vec_add(ac_o, bc_o);
  const int32x4_t diff_e = vec_sub(ac_e, bc_e);
  const int32x4_t diff_o = vec_sub(ac_o, bc_o);

  // Add rounding offset
  const int32x4_t rsum_o = vec_add(sum_o, vec_dct_const_rounding);
  const int32x4_t rsum_e = vec_add(sum_e, vec_dct_const_rounding);
  const int32x4_t rdiff_o = vec_add(diff_o, vec_dct_const_rounding);
  const int32x4_t rdiff_e = vec_add(diff_e, vec_dct_const_rounding);

  const int32x4_t ssum_o = vec_sra(rsum_o, vec_dct_const_bits);
  const int32x4_t ssum_e = vec_sra(rsum_e, vec_dct_const_bits);
  const int32x4_t sdiff_o = vec_sra(rdiff_o, vec_dct_const_bits);
  const int32x4_t sdiff_e = vec_sra(rdiff_e, vec_dct_const_bits);

  // There's no pack operation for even and odd, so we need to permute.
  *add = (int16x8_t)vec_perm(ssum_e, ssum_o, vec_perm_odd_even_pack);
  *sub = (int16x8_t)vec_perm(sdiff_e, sdiff_o, vec_perm_odd_even_pack);
}

// Returns (a * c1 +/- b * c2 + (2 << 13)) >> 14
static INLINE void double_butterfly(int16x8_t a, int16x8_t c1, int16x8_t b,
                                    int16x8_t c2, int16x8_t *add,
                                    int16x8_t *sub) {
  const int32x4_t ac1_o = vec_mulo(a, c1);
  const int32x4_t ac1_e = vec_mule(a, c1);
  const int32x4_t ac2_o = vec_mulo(a, c2);
  const int32x4_t ac2_e = vec_mule(a, c2);

  const int32x4_t bc1_o = vec_mulo(b, c1);
  const int32x4_t bc1_e = vec_mule(b, c1);
  const int32x4_t bc2_o = vec_mulo(b, c2);
  const int32x4_t bc2_e = vec_mule(b, c2);

  const int32x4_t sum_o = vec_add(ac1_o, bc2_o);
  const int32x4_t sum_e = vec_add(ac1_e, bc2_e);
  const int32x4_t diff_o = vec_sub(ac2_o, bc1_o);
  const int32x4_t diff_e = vec_sub(ac2_e, bc1_e);

  // Add rounding offset
  const int32x4_t rsum_o = vec_add(sum_o, vec_dct_const_rounding);
  const int32x4_t rsum_e = vec_add(sum_e, vec_dct_const_rounding);
  const int32x4_t rdiff_o = vec_add(diff_o, vec_dct_const_rounding);
  const int32x4_t rdiff_e = vec_add(diff_e, vec_dct_const_rounding);

  const int32x4_t ssum_o = vec_sra(rsum_o, vec_dct_const_bits);
  const int32x4_t ssum_e = vec_sra(rsum_e, vec_dct_const_bits);
  const int32x4_t sdiff_o = vec_sra(rdiff_o, vec_dct_const_bits);
  const int32x4_t sdiff_e = vec_sra(rdiff_e, vec_dct_const_bits);

  // There's no pack operation for even and odd, so we need to permute.
  *add = (int16x8_t)vec_perm(ssum_e, ssum_o, vec_perm_odd_even_pack);
  *sub = (int16x8_t)vec_perm(sdiff_e, sdiff_o, vec_perm_odd_even_pack);
}

// While other architecture combine the load and the stage 1 operations, Power9
// benchmarking show no benefit in such an approach.
static INLINE void load(const int16_t *a, int stride, int16x8_t *b) {
  // Tried out different combinations of load and shift instructions, this is
  // the fastest one.
  {
    const int16x8_t l0 = vec_vsx_ld(0, a);
    const int16x8_t l1 = vec_vsx_ld(0, a + stride);
    const int16x8_t l2 = vec_vsx_ld(0, a + 2 * stride);
    const int16x8_t l3 = vec_vsx_ld(0, a + 3 * stride);
    const int16x8_t l4 = vec_vsx_ld(0, a + 4 * stride);
    const int16x8_t l5 = vec_vsx_ld(0, a + 5 * stride);
    const int16x8_t l6 = vec_vsx_ld(0, a + 6 * stride);
    const int16x8_t l7 = vec_vsx_ld(0, a + 7 * stride);

    const int16x8_t l8 = vec_vsx_ld(0, a + 8 * stride);
    const int16x8_t l9 = vec_vsx_ld(0, a + 9 * stride);
    const int16x8_t l10 = vec_vsx_ld(0, a + 10 * stride);
    const int16x8_t l11 = vec_vsx_ld(0, a + 11 * stride);
    const int16x8_t l12 = vec_vsx_ld(0, a + 12 * stride);
    const int16x8_t l13 = vec_vsx_ld(0, a + 13 * stride);
    const int16x8_t l14 = vec_vsx_ld(0, a + 14 * stride);
    const int16x8_t l15 = vec_vsx_ld(0, a + 15 * stride);

    b[0] = vec_sl(l0, vec_dct_scale_log2);
    b[1] = vec_sl(l1, vec_dct_scale_log2);
    b[2] = vec_sl(l2, vec_dct_scale_log2);
    b[3] = vec_sl(l3, vec_dct_scale_log2);
    b[4] = vec_sl(l4, vec_dct_scale_log2);
    b[5] = vec_sl(l5, vec_dct_scale_log2);
    b[6] = vec_sl(l6, vec_dct_scale_log2);
    b[7] = vec_sl(l7, vec_dct_scale_log2);

    b[8] = vec_sl(l8, vec_dct_scale_log2);
    b[9] = vec_sl(l9, vec_dct_scale_log2);
    b[10] = vec_sl(l10, vec_dct_scale_log2);
    b[11] = vec_sl(l11, vec_dct_scale_log2);
    b[12] = vec_sl(l12, vec_dct_scale_log2);
    b[13] = vec_sl(l13, vec_dct_scale_log2);
    b[14] = vec_sl(l14, vec_dct_scale_log2);
    b[15] = vec_sl(l15, vec_dct_scale_log2);
  }
  {
    const int16x8_t l16 = vec_vsx_ld(0, a + 16 * stride);
    const int16x8_t l17 = vec_vsx_ld(0, a + 17 * stride);
    const int16x8_t l18 = vec_vsx_ld(0, a + 18 * stride);
    const int16x8_t l19 = vec_vsx_ld(0, a + 19 * stride);
    const int16x8_t l20 = vec_vsx_ld(0, a + 20 * stride);
    const int16x8_t l21 = vec_vsx_ld(0, a + 21 * stride);
    const int16x8_t l22 = vec_vsx_ld(0, a + 22 * stride);
    const int16x8_t l23 = vec_vsx_ld(0, a + 23 * stride);

    const int16x8_t l24 = vec_vsx_ld(0, a + 24 * stride);
    const int16x8_t l25 = vec_vsx_ld(0, a + 25 * stride);
    const int16x8_t l26 = vec_vsx_ld(0, a + 26 * stride);
    const int16x8_t l27 = vec_vsx_ld(0, a + 27 * stride);
    const int16x8_t l28 = vec_vsx_ld(0, a + 28 * stride);
    const int16x8_t l29 = vec_vsx_ld(0, a + 29 * stride);
    const int16x8_t l30 = vec_vsx_ld(0, a + 30 * stride);
    const int16x8_t l31 = vec_vsx_ld(0, a + 31 * stride);

    b[16] = vec_sl(l16, vec_dct_scale_log2);
    b[17] = vec_sl(l17, vec_dct_scale_log2);
    b[18] = vec_sl(l18, vec_dct_scale_log2);
    b[19] = vec_sl(l19, vec_dct_scale_log2);
    b[20] = vec_sl(l20, vec_dct_scale_log2);
    b[21] = vec_sl(l21, vec_dct_scale_log2);
    b[22] = vec_sl(l22, vec_dct_scale_log2);
    b[23] = vec_sl(l23, vec_dct_scale_log2);

    b[24] = vec_sl(l24, vec_dct_scale_log2);
    b[25] = vec_sl(l25, vec_dct_scale_log2);
    b[26] = vec_sl(l26, vec_dct_scale_log2);
    b[27] = vec_sl(l27, vec_dct_scale_log2);
    b[28] = vec_sl(l28, vec_dct_scale_log2);
    b[29] = vec_sl(l29, vec_dct_scale_log2);
    b[30] = vec_sl(l30, vec_dct_scale_log2);
    b[31] = vec_sl(l31, vec_dct_scale_log2);
  }
}

static INLINE void store(tran_low_t *a, const int16x8_t *b) {
  vec_vsx_st(b[0], 0, a);
  vec_vsx_st(b[8], 0, a + 8);
  vec_vsx_st(b[16], 0, a + 16);
  vec_vsx_st(b[24], 0, a + 24);

  vec_vsx_st(b[1], 0, a + 32);
  vec_vsx_st(b[9], 0, a + 40);
  vec_vsx_st(b[17], 0, a + 48);
  vec_vsx_st(b[25], 0, a + 56);

  vec_vsx_st(b[2], 0, a + 64);
  vec_vsx_st(b[10], 0, a + 72);
  vec_vsx_st(b[18], 0, a + 80);
  vec_vsx_st(b[26], 0, a + 88);

  vec_vsx_st(b[3], 0, a + 96);
  vec_vsx_st(b[11], 0, a + 104);
  vec_vsx_st(b[19], 0, a + 112);
  vec_vsx_st(b[27], 0, a + 120);

  vec_vsx_st(b[4], 0, a + 128);
  vec_vsx_st(b[12], 0, a + 136);
  vec_vsx_st(b[20], 0, a + 144);
  vec_vsx_st(b[28], 0, a + 152);

  vec_vsx_st(b[5], 0, a + 160);
  vec_vsx_st(b[13], 0, a + 168);
  vec_vsx_st(b[21], 0, a + 176);
  vec_vsx_st(b[29], 0, a + 184);

  vec_vsx_st(b[6], 0, a + 192);
  vec_vsx_st(b[14], 0, a + 200);
  vec_vsx_st(b[22], 0, a + 208);
  vec_vsx_st(b[30], 0, a + 216);

  vec_vsx_st(b[7], 0, a + 224);
  vec_vsx_st(b[15], 0, a + 232);
  vec_vsx_st(b[23], 0, a + 240);
  vec_vsx_st(b[31], 0, a + 248);
}

// Returns 1 if negative 0 if positive
static INLINE int16x8_t vec_sign_s16(int16x8_t a) {
  return vec_sr(a, vec_shift_sign_s16);
}

// Add 2 if positive, 1 if negative, and shift by 2.
static INLINE int16x8_t sub_round_shift(const int16x8_t a) {
  const int16x8_t sign = vec_sign_s16(a);
  return vec_sra(vec_sub(vec_add(a, vec_twos_s16), sign), vec_dct_scale_log2);
}

// Add 1 if positive, 2 if negative, and shift by 2.
// In practice, add 1, then add the sign bit, then shift without rounding.
static INLINE int16x8_t add_round_shift_s16(const int16x8_t a) {
  const int16x8_t sign = vec_sign_s16(a);
  return vec_sra(vec_add(vec_add(a, vec_ones_s16), sign), vec_dct_scale_log2);
}

static void fdct32_vsx(const int16x8_t *in, int16x8_t *out, int pass) {
  int16x8_t temp0[32];  // Hold stages: 1, 4, 7
  int16x8_t temp1[32];  // Hold stages: 2, 5
  int16x8_t temp2[32];  // Hold stages: 3, 6
  int i;

  // Stage 1
  // Unrolling this loops actually slows down Power9 benchmarks
  for (i = 0; i < 16; i++) {
    temp0[i] = vec_add(in[i], in[31 - i]);
    // pass through to stage 3.
    temp1[i + 16] = vec_sub(in[15 - i], in[i + 16]);
  }

  // Stage 2
  // Unrolling this loops actually slows down Power9 benchmarks
  for (i = 0; i < 8; i++) {
    temp1[i] = vec_add(temp0[i], temp0[15 - i]);
    temp1[i + 8] = vec_sub(temp0[7 - i], temp0[i + 8]);
  }

  // Apply butterflies (in place) on pass through to stage 3.
  single_butterfly(temp1[27], temp1[20], &temp1[27], &temp1[20]);
  single_butterfly(temp1[26], temp1[21], &temp1[26], &temp1[21]);
  single_butterfly(temp1[25], temp1[22], &temp1[25], &temp1[22]);
  single_butterfly(temp1[24], temp1[23], &temp1[24], &temp1[23]);

  // dump the magnitude by 4, hence the intermediate values are within
  // the range of 16 bits.
  if (pass) {
    temp1[0] = add_round_shift_s16(temp1[0]);
    temp1[1] = add_round_shift_s16(temp1[1]);
    temp1[2] = add_round_shift_s16(temp1[2]);
    temp1[3] = add_round_shift_s16(temp1[3]);
    temp1[4] = add_round_shift_s16(temp1[4]);
    temp1[5] = add_round_shift_s16(temp1[5]);
    temp1[6] = add_round_shift_s16(temp1[6]);
    temp1[7] = add_round_shift_s16(temp1[7]);
    temp1[8] = add_round_shift_s16(temp1[8]);
    temp1[9] = add_round_shift_s16(temp1[9]);
    temp1[10] = add_round_shift_s16(temp1[10]);
    temp1[11] = add_round_shift_s16(temp1[11]);
    temp1[12] = add_round_shift_s16(temp1[12]);
    temp1[13] = add_round_shift_s16(temp1[13]);
    temp1[14] = add_round_shift_s16(temp1[14]);
    temp1[15] = add_round_shift_s16(temp1[15]);

    temp1[16] = add_round_shift_s16(temp1[16]);
    temp1[17] = add_round_shift_s16(temp1[17]);
    temp1[18] = add_round_shift_s16(temp1[18]);
    temp1[19] = add_round_shift_s16(temp1[19]);
    temp1[20] = add_round_shift_s16(temp1[20]);
    temp1[21] = add_round_shift_s16(temp1[21]);
    temp1[22] = add_round_shift_s16(temp1[22]);
    temp1[23] = add_round_shift_s16(temp1[23]);
    temp1[24] = add_round_shift_s16(temp1[24]);
    temp1[25] = add_round_shift_s16(temp1[25]);
    temp1[26] = add_round_shift_s16(temp1[26]);
    temp1[27] = add_round_shift_s16(temp1[27]);
    temp1[28] = add_round_shift_s16(temp1[28]);
    temp1[29] = add_round_shift_s16(temp1[29]);
    temp1[30] = add_round_shift_s16(temp1[30]);
    temp1[31] = add_round_shift_s16(temp1[31]);
  }

  // Stage 3
  temp2[0] = vec_add(temp1[0], temp1[7]);
  temp2[1] = vec_add(temp1[1], temp1[6]);
  temp2[2] = vec_add(temp1[2], temp1[5]);
  temp2[3] = vec_add(temp1[3], temp1[4]);
  temp2[5] = vec_sub(temp1[2], temp1[5]);
  temp2[6] = vec_sub(temp1[1], temp1[6]);
  temp2[8] = temp1[8];
  temp2[9] = temp1[9];

  single_butterfly(temp1[13], temp1[10], &temp2[13], &temp2[10]);
  single_butterfly(temp1[12], temp1[11], &temp2[12], &temp2[11]);
  temp2[14] = temp1[14];
  temp2[15] = temp1[15];

  temp2[18] = vec_add(temp1[18], temp1[21]);
  temp2[19] = vec_add(temp1[19], temp1[20]);

  temp2[20] = vec_sub(temp1[19], temp1[20]);
  temp2[21] = vec_sub(temp1[18], temp1[21]);

  temp2[26] = vec_sub(temp1[29], temp1[26]);
  temp2[27] = vec_sub(temp1[28], temp1[27]);

  temp2[28] = vec_add(temp1[28], temp1[27]);
  temp2[29] = vec_add(temp1[29], temp1[26]);

  // Pass through Stage 4
  temp0[7] = vec_sub(temp1[0], temp1[7]);
  temp0[4] = vec_sub(temp1[3], temp1[4]);
  temp0[16] = vec_add(temp1[16], temp1[23]);
  temp0[17] = vec_add(temp1[17], temp1[22]);
  temp0[22] = vec_sub(temp1[17], temp1[22]);
  temp0[23] = vec_sub(temp1[16], temp1[23]);
  temp0[24] = vec_sub(temp1[31], temp1[24]);
  temp0[25] = vec_sub(temp1[30], temp1[25]);
  temp0[30] = vec_add(temp1[30], temp1[25]);
  temp0[31] = vec_add(temp1[31], temp1[24]);

  // Stage 4
  temp0[0] = vec_add(temp2[0], temp2[3]);
  temp0[1] = vec_add(temp2[1], temp2[2]);
  temp0[2] = vec_sub(temp2[1], temp2[2]);
  temp0[3] = vec_sub(temp2[0], temp2[3]);
  single_butterfly(temp2[6], temp2[5], &temp0[6], &temp0[5]);

  temp0[9] = vec_add(temp2[9], temp2[10]);
  temp0[10] = vec_sub(temp2[9], temp2[10]);
  temp0[13] = vec_sub(temp2[14], temp2[13]);
  temp0[14] = vec_add(temp2[14], temp2[13]);

  double_butterfly(temp2[29], cospi8_v, temp2[18], cospi24_v, &temp0[29],
                   &temp0[18]);
  double_butterfly(temp2[28], cospi8_v, temp2[19], cospi24_v, &temp0[28],
                   &temp0[19]);
  double_butterfly(temp2[27], cospi24_v, temp2[20], cospi8m_v, &temp0[27],
                   &temp0[20]);
  double_butterfly(temp2[26], cospi24_v, temp2[21], cospi8m_v, &temp0[26],
                   &temp0[21]);

  // Pass through Stage 5
  temp1[8] = vec_add(temp2[8], temp2[11]);
  temp1[11] = vec_sub(temp2[8], temp2[11]);
  temp1[12] = vec_sub(temp2[15], temp2[12]);
  temp1[15] = vec_add(temp2[15], temp2[12]);

  // Stage 5
  // 0 and 1 pass through to 0 and 16 at the end
  single_butterfly(temp0[0], temp0[1], &out[0], &out[16]);

  // 2 and 3 pass through to 8 and 24 at the end
  double_butterfly(temp0[3], cospi8_v, temp0[2], cospi24_v, &out[8], &out[24]);

  temp1[4] = vec_add(temp0[4], temp0[5]);
  temp1[5] = vec_sub(temp0[4], temp0[5]);
  temp1[6] = vec_sub(temp0[7], temp0[6]);
  temp1[7] = vec_add(temp0[7], temp0[6]);

  double_butterfly(temp0[14], cospi8_v, temp0[9], cospi24_v, &temp1[14],
                   &temp1[9]);
  double_butterfly(temp0[13], cospi24_v, temp0[10], cospi8m_v, &temp1[13],
                   &temp1[10]);

  temp1[17] = vec_add(temp0[17], temp0[18]);
  temp1[18] = vec_sub(temp0[17], temp0[18]);

  temp1[21] = vec_sub(temp0[22], temp0[21]);
  temp1[22] = vec_add(temp0[22], temp0[21]);

  temp1[25] = vec_add(temp0[25], temp0[26]);
  temp1[26] = vec_sub(temp0[25], temp0[26]);

  temp1[29] = vec_sub(temp0[30], temp0[29]);
  temp1[30] = vec_add(temp0[30], temp0[29]);

  // Pass through Stage 6
  temp2[16] = vec_add(temp0[16], temp0[19]);
  temp2[19] = vec_sub(temp0[16], temp0[19]);
  temp2[20] = vec_sub(temp0[23], temp0[20]);
  temp2[23] = vec_add(temp0[23], temp0[20]);
  temp2[24] = vec_add(temp0[24], temp0[27]);
  temp2[27] = vec_sub(temp0[24], temp0[27]);
  temp2[28] = vec_sub(temp0[31], temp0[28]);
  temp2[31] = vec_add(temp0[31], temp0[28]);

  // Stage 6
  // 4 and 7 pass through to 4 and 28 at the end
  double_butterfly(temp1[7], cospi4_v, temp1[4], cospi28_v, &out[4], &out[28]);
  // 5 and 6 pass through to 20 and 12 at the end
  double_butterfly(temp1[6], cospi20_v, temp1[5], cospi12_v, &out[20],
                   &out[12]);
  temp2[8] = vec_add(temp1[8], temp1[9]);
  temp2[9] = vec_sub(temp1[8], temp1[9]);
  temp2[10] = vec_sub(temp1[11], temp1[10]);
  temp2[11] = vec_add(temp1[11], temp1[10]);
  temp2[12] = vec_add(temp1[12], temp1[13]);
  temp2[13] = vec_sub(temp1[12], temp1[13]);
  temp2[14] = vec_sub(temp1[15], temp1[14]);
  temp2[15] = vec_add(temp1[15], temp1[14]);

  double_butterfly(temp1[30], cospi4_v, temp1[17], cospi28_v, &temp2[30],
                   &temp2[17]);
  double_butterfly(temp1[29], cospi28_v, temp1[18], cospi4m_v, &temp2[29],
                   &temp2[18]);
  double_butterfly(temp1[26], cospi20_v, temp1[21], cospi12_v, &temp2[26],
                   &temp2[21]);
  double_butterfly(temp1[25], cospi12_v, temp1[22], cospi20m_v, &temp2[25],
                   &temp2[22]);

  // Stage 7
  double_butterfly(temp2[15], cospi2_v, temp2[8], cospi30_v, &out[2], &out[30]);
  double_butterfly(temp2[14], cospi18_v, temp2[9], cospi14_v, &out[18],
                   &out[14]);
  double_butterfly(temp2[13], cospi10_v, temp2[10], cospi22_v, &out[10],
                   &out[22]);
  double_butterfly(temp2[12], cospi26_v, temp2[11], cospi6_v, &out[26],
                   &out[6]);

  temp0[16] = vec_add(temp2[16], temp2[17]);
  temp0[17] = vec_sub(temp2[16], temp2[17]);
  temp0[18] = vec_sub(temp2[19], temp2[18]);
  temp0[19] = vec_add(temp2[19], temp2[18]);
  temp0[20] = vec_add(temp2[20], temp2[21]);
  temp0[21] = vec_sub(temp2[20], temp2[21]);
  temp0[22] = vec_sub(temp2[23], temp2[22]);
  temp0[23] = vec_add(temp2[23], temp2[22]);
  temp0[24] = vec_add(temp2[24], temp2[25]);
  temp0[25] = vec_sub(temp2[24], temp2[25]);
  temp0[26] = vec_sub(temp2[27], temp2[26]);
  temp0[27] = vec_add(temp2[27], temp2[26]);
  temp0[28] = vec_add(temp2[28], temp2[29]);
  temp0[29] = vec_sub(temp2[28], temp2[29]);
  temp0[30] = vec_sub(temp2[31], temp2[30]);
  temp0[31] = vec_add(temp2[31], temp2[30]);

  // Final stage --- outputs indices are bit-reversed.
  double_butterfly(temp0[31], cospi1_v, temp0[16], cospi31_v, &out[1],
                   &out[31]);
  double_butterfly(temp0[30], cospi17_v, temp0[17], cospi15_v, &out[17],
                   &out[15]);
  double_butterfly(temp0[29], cospi9_v, temp0[18], cospi23_v, &out[9],
                   &out[23]);
  double_butterfly(temp0[28], cospi25_v, temp0[19], cospi7_v, &out[25],
                   &out[7]);
  double_butterfly(temp0[27], cospi5_v, temp0[20], cospi27_v, &out[5],
                   &out[27]);
  double_butterfly(temp0[26], cospi21_v, temp0[21], cospi11_v, &out[21],
                   &out[11]);
  double_butterfly(temp0[25], cospi13_v, temp0[22], cospi19_v, &out[13],
                   &out[19]);
  double_butterfly(temp0[24], cospi29_v, temp0[23], cospi3_v, &out[29],
                   &out[3]);

  if (pass == 0) {
    for (i = 0; i < 32; i++) {
      out[i] = sub_round_shift(out[i]);
    }
  }
}

void vpx_fdct32x32_rd_vsx(const int16_t *input, tran_low_t *out, int stride) {
  int16x8_t temp0[32];
  int16x8_t temp1[32];
  int16x8_t temp2[32];
  int16x8_t temp3[32];
  int16x8_t temp4[32];
  int16x8_t temp5[32];
  int16x8_t temp6[32];

  // Process in 8x32 columns.
  load(input, stride, temp0);
  fdct32_vsx(temp0, temp1, 0);

  load(input + 8, stride, temp0);
  fdct32_vsx(temp0, temp2, 0);

  load(input + 16, stride, temp0);
  fdct32_vsx(temp0, temp3, 0);

  load(input + 24, stride, temp0);
  fdct32_vsx(temp0, temp4, 0);

  // Generate the top row by munging the first set of 8 from each one
  // together.
  transpose_8x8(&temp1[0], &temp0[0]);
  transpose_8x8(&temp2[0], &temp0[8]);
  transpose_8x8(&temp3[0], &temp0[16]);
  transpose_8x8(&temp4[0], &temp0[24]);

  fdct32_vsx(temp0, temp5, 1);

  transpose_8x8(&temp5[0], &temp6[0]);
  transpose_8x8(&temp5[8], &temp6[8]);
  transpose_8x8(&temp5[16], &temp6[16]);
  transpose_8x8(&temp5[24], &temp6[24]);

  store(out, temp6);

  // Second row of 8x32.
  transpose_8x8(&temp1[8], &temp0[0]);
  transpose_8x8(&temp2[8], &temp0[8]);
  transpose_8x8(&temp3[8], &temp0[16]);
  transpose_8x8(&temp4[8], &temp0[24]);

  fdct32_vsx(temp0, temp5, 1);

  transpose_8x8(&temp5[0], &temp6[0]);
  transpose_8x8(&temp5[8], &temp6[8]);
  transpose_8x8(&temp5[16], &temp6[16]);
  transpose_8x8(&temp5[24], &temp6[24]);

  store(out + 8 * 32, temp6);

  // Third row of 8x32
  transpose_8x8(&temp1[16], &temp0[0]);
  transpose_8x8(&temp2[16], &temp0[8]);
  transpose_8x8(&temp3[16], &temp0[16]);
  transpose_8x8(&temp4[16], &temp0[24]);

  fdct32_vsx(temp0, temp5, 1);

  transpose_8x8(&temp5[0], &temp6[0]);
  transpose_8x8(&temp5[8], &temp6[8]);
  transpose_8x8(&temp5[16], &temp6[16]);
  transpose_8x8(&temp5[24], &temp6[24]);

  store(out + 16 * 32, temp6);

  // Final row of 8x32.
  transpose_8x8(&temp1[24], &temp0[0]);
  transpose_8x8(&temp2[24], &temp0[8]);
  transpose_8x8(&temp3[24], &temp0[16]);
  transpose_8x8(&temp4[24], &temp0[24]);

  fdct32_vsx(temp0, temp5, 1);

  transpose_8x8(&temp5[0], &temp6[0]);
  transpose_8x8(&temp5[8], &temp6[8]);
  transpose_8x8(&temp5[16], &temp6[16]);
  transpose_8x8(&temp5[24], &temp6[24]);

  store(out + 24 * 32, temp6);
}
