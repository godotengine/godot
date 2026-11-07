/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_X86_HIGHBD_INV_TXFM_SSE2_H_
#define VPX_VPX_DSP_X86_HIGHBD_INV_TXFM_SSE2_H_

#include <emmintrin.h>  // SSE2

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/inv_txfm.h"
#include "vpx_dsp/x86/transpose_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

// Note: There is no 64-bit bit-level shifting SIMD instruction. All
// coefficients are left shifted by 2, so that dct_const_round_shift() can be
// done by right shifting 2 bytes.

static INLINE void extend_64bit(const __m128i in,
                                __m128i *const out /*out[2]*/) {
  out[0] = _mm_unpacklo_epi32(in, in);  // 0, 0, 1, 1
  out[1] = _mm_unpackhi_epi32(in, in);  // 2, 2, 3, 3
}

static INLINE __m128i wraplow_16bit_shift4(const __m128i in0, const __m128i in1,
                                           const __m128i rounding) {
  __m128i temp[2];
  temp[0] = _mm_add_epi32(in0, rounding);
  temp[1] = _mm_add_epi32(in1, rounding);
  temp[0] = _mm_srai_epi32(temp[0], 4);
  temp[1] = _mm_srai_epi32(temp[1], 4);
  return _mm_packs_epi32(temp[0], temp[1]);
}

static INLINE __m128i wraplow_16bit_shift5(const __m128i in0, const __m128i in1,
                                           const __m128i rounding) {
  __m128i temp[2];
  temp[0] = _mm_add_epi32(in0, rounding);
  temp[1] = _mm_add_epi32(in1, rounding);
  temp[0] = _mm_srai_epi32(temp[0], 5);
  temp[1] = _mm_srai_epi32(temp[1], 5);
  return _mm_packs_epi32(temp[0], temp[1]);
}

static INLINE __m128i dct_const_round_shift_64bit(const __m128i in) {
  const __m128i t =
      _mm_add_epi64(in, pair_set_epi32(DCT_CONST_ROUNDING << 2, 0));
  return _mm_srli_si128(t, 2);
}

static INLINE __m128i pack_4(const __m128i in0, const __m128i in1) {
  const __m128i t0 = _mm_unpacklo_epi32(in0, in1);  // 0, 2
  const __m128i t1 = _mm_unpackhi_epi32(in0, in1);  // 1, 3
  return _mm_unpacklo_epi32(t0, t1);                // 0, 1, 2, 3
}

static INLINE void abs_extend_64bit_sse2(const __m128i in,
                                         __m128i *const out /*out[2]*/,
                                         __m128i *const sign /*sign[2]*/) {
  sign[0] = _mm_srai_epi32(in, 31);
  out[0] = _mm_xor_si128(in, sign[0]);
  out[0] = _mm_sub_epi32(out[0], sign[0]);
  sign[1] = _mm_unpackhi_epi32(sign[0], sign[0]);  // 64-bit sign of 2, 3
  sign[0] = _mm_unpacklo_epi32(sign[0], sign[0]);  // 64-bit sign of 0, 1
  out[1] = _mm_unpackhi_epi32(out[0], out[0]);     // 2, 3
  out[0] = _mm_unpacklo_epi32(out[0], out[0]);     // 0, 1
}

// Note: cospi must be non negative.
static INLINE __m128i multiply_apply_sign_sse2(const __m128i in,
                                               const __m128i sign,
                                               const __m128i cospi) {
  __m128i out = _mm_mul_epu32(in, cospi);
  out = _mm_xor_si128(out, sign);
  return _mm_sub_epi64(out, sign);
}

// Note: c must be non negative.
static INLINE __m128i multiplication_round_shift_sse2(
    const __m128i *const in /*in[2]*/, const __m128i *const sign /*sign[2]*/,
    const int c) {
  const __m128i pair_c = pair_set_epi32(c << 2, 0);
  __m128i t0, t1;

  assert(c >= 0);
  t0 = multiply_apply_sign_sse2(in[0], sign[0], pair_c);
  t1 = multiply_apply_sign_sse2(in[1], sign[1], pair_c);
  t0 = dct_const_round_shift_64bit(t0);
  t1 = dct_const_round_shift_64bit(t1);

  return pack_4(t0, t1);
}

// Note: c must be non negative.
static INLINE __m128i multiplication_neg_round_shift_sse2(
    const __m128i *const in /*in[2]*/, const __m128i *const sign /*sign[2]*/,
    const int c) {
  const __m128i pair_c = pair_set_epi32(c << 2, 0);
  __m128i t0, t1;

  assert(c >= 0);
  t0 = multiply_apply_sign_sse2(in[0], sign[0], pair_c);
  t1 = multiply_apply_sign_sse2(in[1], sign[1], pair_c);
  t0 = _mm_sub_epi64(_mm_setzero_si128(), t0);
  t1 = _mm_sub_epi64(_mm_setzero_si128(), t1);
  t0 = dct_const_round_shift_64bit(t0);
  t1 = dct_const_round_shift_64bit(t1);

  return pack_4(t0, t1);
}

// Note: c0 and c1 must be non negative.
static INLINE void highbd_butterfly_sse2(const __m128i in0, const __m128i in1,
                                         const int c0, const int c1,
                                         __m128i *const out0,
                                         __m128i *const out1) {
  const __m128i pair_c0 = pair_set_epi32(c0 << 2, 0);
  const __m128i pair_c1 = pair_set_epi32(c1 << 2, 0);
  __m128i temp1[4], temp2[4], sign1[2], sign2[2];

  assert(c0 >= 0);
  assert(c1 >= 0);
  abs_extend_64bit_sse2(in0, temp1, sign1);
  abs_extend_64bit_sse2(in1, temp2, sign2);
  temp1[2] = multiply_apply_sign_sse2(temp1[0], sign1[0], pair_c1);
  temp1[3] = multiply_apply_sign_sse2(temp1[1], sign1[1], pair_c1);
  temp1[0] = multiply_apply_sign_sse2(temp1[0], sign1[0], pair_c0);
  temp1[1] = multiply_apply_sign_sse2(temp1[1], sign1[1], pair_c0);
  temp2[2] = multiply_apply_sign_sse2(temp2[0], sign2[0], pair_c0);
  temp2[3] = multiply_apply_sign_sse2(temp2[1], sign2[1], pair_c0);
  temp2[0] = multiply_apply_sign_sse2(temp2[0], sign2[0], pair_c1);
  temp2[1] = multiply_apply_sign_sse2(temp2[1], sign2[1], pair_c1);
  temp1[0] = _mm_sub_epi64(temp1[0], temp2[0]);
  temp1[1] = _mm_sub_epi64(temp1[1], temp2[1]);
  temp2[0] = _mm_add_epi64(temp1[2], temp2[2]);
  temp2[1] = _mm_add_epi64(temp1[3], temp2[3]);
  temp1[0] = dct_const_round_shift_64bit(temp1[0]);
  temp1[1] = dct_const_round_shift_64bit(temp1[1]);
  temp2[0] = dct_const_round_shift_64bit(temp2[0]);
  temp2[1] = dct_const_round_shift_64bit(temp2[1]);
  *out0 = pack_4(temp1[0], temp1[1]);
  *out1 = pack_4(temp2[0], temp2[1]);
}

// Note: c0 and c1 must be non negative.
static INLINE void highbd_partial_butterfly_sse2(const __m128i in, const int c0,
                                                 const int c1,
                                                 __m128i *const out0,
                                                 __m128i *const out1) {
  __m128i temp[2], sign[2];

  assert(c0 >= 0);
  assert(c1 >= 0);
  abs_extend_64bit_sse2(in, temp, sign);
  *out0 = multiplication_round_shift_sse2(temp, sign, c0);
  *out1 = multiplication_round_shift_sse2(temp, sign, c1);
}

// Note: c0 and c1 must be non negative.
static INLINE void highbd_partial_butterfly_neg_sse2(const __m128i in,
                                                     const int c0, const int c1,
                                                     __m128i *const out0,
                                                     __m128i *const out1) {
  __m128i temp[2], sign[2];

  assert(c0 >= 0);
  assert(c1 >= 0);
  abs_extend_64bit_sse2(in, temp, sign);
  *out0 = multiplication_neg_round_shift_sse2(temp, sign, c1);
  *out1 = multiplication_round_shift_sse2(temp, sign, c0);
}

static INLINE void highbd_butterfly_cospi16_sse2(const __m128i in0,
                                                 const __m128i in1,
                                                 __m128i *const out0,
                                                 __m128i *const out1) {
  __m128i temp1[2], temp2, sign[2];

  temp2 = _mm_add_epi32(in0, in1);
  abs_extend_64bit_sse2(temp2, temp1, sign);
  *out0 = multiplication_round_shift_sse2(temp1, sign, cospi_16_64);
  temp2 = _mm_sub_epi32(in0, in1);
  abs_extend_64bit_sse2(temp2, temp1, sign);
  *out1 = multiplication_round_shift_sse2(temp1, sign, cospi_16_64);
}

// Only do addition and subtraction butterfly, size = 16, 32
static INLINE void highbd_add_sub_butterfly(const __m128i *in, __m128i *out,
                                            int size) {
  int i = 0;
  const int num = size >> 1;
  const int bound = size - 1;
  while (i < num) {
    out[i] = _mm_add_epi32(in[i], in[bound - i]);
    out[bound - i] = _mm_sub_epi32(in[i], in[bound - i]);
    i++;
  }
}

static INLINE void highbd_idct8_stage4(const __m128i *const in,
                                       __m128i *const out) {
  out[0] = _mm_add_epi32(in[0], in[7]);
  out[1] = _mm_add_epi32(in[1], in[6]);
  out[2] = _mm_add_epi32(in[2], in[5]);
  out[3] = _mm_add_epi32(in[3], in[4]);
  out[4] = _mm_sub_epi32(in[3], in[4]);
  out[5] = _mm_sub_epi32(in[2], in[5]);
  out[6] = _mm_sub_epi32(in[1], in[6]);
  out[7] = _mm_sub_epi32(in[0], in[7]);
}

static INLINE void highbd_idct8x8_final_round(__m128i *const io) {
  io[0] = wraplow_16bit_shift5(io[0], io[8], _mm_set1_epi32(16));
  io[1] = wraplow_16bit_shift5(io[1], io[9], _mm_set1_epi32(16));
  io[2] = wraplow_16bit_shift5(io[2], io[10], _mm_set1_epi32(16));
  io[3] = wraplow_16bit_shift5(io[3], io[11], _mm_set1_epi32(16));
  io[4] = wraplow_16bit_shift5(io[4], io[12], _mm_set1_epi32(16));
  io[5] = wraplow_16bit_shift5(io[5], io[13], _mm_set1_epi32(16));
  io[6] = wraplow_16bit_shift5(io[6], io[14], _mm_set1_epi32(16));
  io[7] = wraplow_16bit_shift5(io[7], io[15], _mm_set1_epi32(16));
}

static INLINE void highbd_idct16_4col_stage7(const __m128i *const in,
                                             __m128i *const out) {
  out[0] = _mm_add_epi32(in[0], in[15]);
  out[1] = _mm_add_epi32(in[1], in[14]);
  out[2] = _mm_add_epi32(in[2], in[13]);
  out[3] = _mm_add_epi32(in[3], in[12]);
  out[4] = _mm_add_epi32(in[4], in[11]);
  out[5] = _mm_add_epi32(in[5], in[10]);
  out[6] = _mm_add_epi32(in[6], in[9]);
  out[7] = _mm_add_epi32(in[7], in[8]);
  out[8] = _mm_sub_epi32(in[7], in[8]);
  out[9] = _mm_sub_epi32(in[6], in[9]);
  out[10] = _mm_sub_epi32(in[5], in[10]);
  out[11] = _mm_sub_epi32(in[4], in[11]);
  out[12] = _mm_sub_epi32(in[3], in[12]);
  out[13] = _mm_sub_epi32(in[2], in[13]);
  out[14] = _mm_sub_epi32(in[1], in[14]);
  out[15] = _mm_sub_epi32(in[0], in[15]);
}

static INLINE __m128i add_clamp(const __m128i in0, const __m128i in1,
                                const int bd) {
  const __m128i zero = _mm_setzero_si128();
  // Faster than _mm_set1_epi16((1 << bd) - 1).
  const __m128i one = _mm_set1_epi16(1);
  const __m128i max = _mm_sub_epi16(_mm_slli_epi16(one, bd), one);
  __m128i d;

  d = _mm_adds_epi16(in0, in1);
  d = _mm_max_epi16(d, zero);
  d = _mm_min_epi16(d, max);

  return d;
}

static INLINE void highbd_idct_1_add_kernel(const tran_low_t *input,
                                            uint16_t *dest, int stride, int bd,
                                            const int size) {
  int a1, i, j;
  tran_low_t out;
  __m128i dc, d;

  out = HIGHBD_WRAPLOW(
      dct_const_round_shift(input[0] * (tran_high_t)cospi_16_64), bd);
  out =
      HIGHBD_WRAPLOW(dct_const_round_shift(out * (tran_high_t)cospi_16_64), bd);
  a1 = ROUND_POWER_OF_TWO(out, (size == 8) ? 5 : 6);
  dc = _mm_set1_epi16(a1);

  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; j += 8) {
      d = _mm_load_si128((const __m128i *)(&dest[j]));
      d = add_clamp(d, dc, bd);
      _mm_store_si128((__m128i *)(&dest[j]), d);
    }
    dest += stride;
  }
}

static INLINE void recon_and_store_4(const __m128i in, uint16_t *const dest,
                                     const int bd) {
  __m128i d;

  d = _mm_loadl_epi64((const __m128i *)dest);
  d = add_clamp(d, in, bd);
  _mm_storel_epi64((__m128i *)dest, d);
}

static INLINE void recon_and_store_4x2(const __m128i in, uint16_t *const dest,
                                       const int stride, const int bd) {
  __m128i d;

  d = _mm_loadl_epi64((const __m128i *)(dest + 0 * stride));
  d = _mm_castps_si128(
      _mm_loadh_pi(_mm_castsi128_ps(d), (const __m64 *)(dest + 1 * stride)));
  d = add_clamp(d, in, bd);
  _mm_storel_epi64((__m128i *)(dest + 0 * stride), d);
  _mm_storeh_pi((__m64 *)(dest + 1 * stride), _mm_castsi128_ps(d));
}

static INLINE void recon_and_store_4x4(const __m128i *const in, uint16_t *dest,
                                       const int stride, const int bd) {
  recon_and_store_4x2(in[0], dest, stride, bd);
  dest += 2 * stride;
  recon_and_store_4x2(in[1], dest, stride, bd);
}

static INLINE void recon_and_store_8(const __m128i in, uint16_t **const dest,
                                     const int stride, const int bd) {
  __m128i d;

  d = _mm_load_si128((const __m128i *)(*dest));
  d = add_clamp(d, in, bd);
  _mm_store_si128((__m128i *)(*dest), d);
  *dest += stride;
}

static INLINE void recon_and_store_8x8(const __m128i *const in, uint16_t *dest,
                                       const int stride, const int bd) {
  recon_and_store_8(in[0], &dest, stride, bd);
  recon_and_store_8(in[1], &dest, stride, bd);
  recon_and_store_8(in[2], &dest, stride, bd);
  recon_and_store_8(in[3], &dest, stride, bd);
  recon_and_store_8(in[4], &dest, stride, bd);
  recon_and_store_8(in[5], &dest, stride, bd);
  recon_and_store_8(in[6], &dest, stride, bd);
  recon_and_store_8(in[7], &dest, stride, bd);
}

static INLINE __m128i load_pack_8_32bit(const tran_low_t *const input) {
  const __m128i t0 = _mm_load_si128((const __m128i *)(input + 0));
  const __m128i t1 = _mm_load_si128((const __m128i *)(input + 4));
  return _mm_packs_epi32(t0, t1);
}

static INLINE void highbd_load_pack_transpose_32bit_8x8(const tran_low_t *input,
                                                        const int stride,
                                                        __m128i *const in) {
  in[0] = load_pack_8_32bit(input + 0 * stride);
  in[1] = load_pack_8_32bit(input + 1 * stride);
  in[2] = load_pack_8_32bit(input + 2 * stride);
  in[3] = load_pack_8_32bit(input + 3 * stride);
  in[4] = load_pack_8_32bit(input + 4 * stride);
  in[5] = load_pack_8_32bit(input + 5 * stride);
  in[6] = load_pack_8_32bit(input + 6 * stride);
  in[7] = load_pack_8_32bit(input + 7 * stride);
  transpose_16bit_8x8(in, in);
}

static INLINE void highbd_load_transpose_32bit_8x4(const tran_low_t *input,
                                                   const int stride,
                                                   __m128i *in) {
  in[0] = _mm_load_si128((const __m128i *)(input + 0 * stride + 0));
  in[1] = _mm_load_si128((const __m128i *)(input + 0 * stride + 4));
  in[2] = _mm_load_si128((const __m128i *)(input + 1 * stride + 0));
  in[3] = _mm_load_si128((const __m128i *)(input + 1 * stride + 4));
  in[4] = _mm_load_si128((const __m128i *)(input + 2 * stride + 0));
  in[5] = _mm_load_si128((const __m128i *)(input + 2 * stride + 4));
  in[6] = _mm_load_si128((const __m128i *)(input + 3 * stride + 0));
  in[7] = _mm_load_si128((const __m128i *)(input + 3 * stride + 4));
  transpose_32bit_8x4(in, in);
}

static INLINE void highbd_load_transpose_32bit_4x4(const tran_low_t *input,
                                                   const int stride,
                                                   __m128i *in) {
  in[0] = _mm_load_si128((const __m128i *)(input + 0 * stride));
  in[1] = _mm_load_si128((const __m128i *)(input + 1 * stride));
  in[2] = _mm_load_si128((const __m128i *)(input + 2 * stride));
  in[3] = _mm_load_si128((const __m128i *)(input + 3 * stride));
  transpose_32bit_4x4(in, in);
}

static INLINE void highbd_write_buffer_8(uint16_t *dest, const __m128i in,
                                         const int bd) {
  const __m128i final_rounding = _mm_set1_epi16(1 << 5);
  __m128i out;

  out = _mm_adds_epi16(in, final_rounding);
  out = _mm_srai_epi16(out, 6);
  recon_and_store_8(out, &dest, 0, bd);
}

static INLINE void highbd_write_buffer_4(uint16_t *const dest, const __m128i in,
                                         const int bd) {
  const __m128i final_rounding = _mm_set1_epi32(1 << 5);
  __m128i out;

  out = _mm_add_epi32(in, final_rounding);
  out = _mm_srai_epi32(out, 6);
  out = _mm_packs_epi32(out, out);
  recon_and_store_4(out, dest, bd);
}

#endif  // VPX_VPX_DSP_X86_HIGHBD_INV_TXFM_SSE2_H_
