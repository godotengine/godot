/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_X86_INV_TXFM_SSE2_H_
#define VPX_VPX_DSP_X86_INV_TXFM_SSE2_H_

#include <emmintrin.h>  // SSE2

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/inv_txfm.h"
#include "vpx_dsp/x86/transpose_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

static INLINE void idct8x8_12_transpose_16bit_4x8(const __m128i *const in,
                                                  __m128i *const out) {
  // Unpack 16 bit elements. Goes from:
  // in[0]: 30 31 32 33  00 01 02 03
  // in[1]: 20 21 22 23  10 11 12 13
  // in[2]: 40 41 42 43  70 71 72 73
  // in[3]: 50 51 52 53  60 61 62 63
  // to:
  // tr0_0: 00 10 01 11  02 12 03 13
  // tr0_1: 20 30 21 31  22 32 23 33
  // tr0_2: 40 50 41 51  42 52 43 53
  // tr0_3: 60 70 61 71  62 72 63 73
  const __m128i tr0_0 = _mm_unpackhi_epi16(in[0], in[1]);
  const __m128i tr0_1 = _mm_unpacklo_epi16(in[1], in[0]);
  const __m128i tr0_2 = _mm_unpacklo_epi16(in[2], in[3]);
  const __m128i tr0_3 = _mm_unpackhi_epi16(in[3], in[2]);

  // Unpack 32 bit elements resulting in:
  // tr1_0: 00 10 20 30  01 11 21 31
  // tr1_1: 02 12 22 32  03 13 23 33
  // tr1_2: 40 50 60 70  41 51 61 71
  // tr1_3: 42 52 62 72  43 53 63 73
  const __m128i tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1);
  const __m128i tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3);
  const __m128i tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1);
  const __m128i tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30  40 50 60 70
  // out[1]: 01 11 21 31  41 51 61 71
  // out[2]: 02 12 22 32  42 52 62 72
  // out[3]: 03 13 23 33  43 53 63 73
  out[0] = _mm_unpacklo_epi64(tr1_0, tr1_1);
  out[1] = _mm_unpackhi_epi64(tr1_0, tr1_1);
  out[2] = _mm_unpacklo_epi64(tr1_2, tr1_3);
  out[3] = _mm_unpackhi_epi64(tr1_2, tr1_3);
}

static INLINE __m128i dct_const_round_shift_sse2(const __m128i in) {
  const __m128i t = _mm_add_epi32(in, _mm_set1_epi32(DCT_CONST_ROUNDING));
  return _mm_srai_epi32(t, DCT_CONST_BITS);
}

static INLINE __m128i idct_madd_round_shift_sse2(const __m128i in,
                                                 const __m128i cospi) {
  const __m128i t = _mm_madd_epi16(in, cospi);
  return dct_const_round_shift_sse2(t);
}

// Calculate the dot product between in0/1 and x and wrap to short.
static INLINE __m128i idct_calc_wraplow_sse2(const __m128i in0,
                                             const __m128i in1,
                                             const __m128i x) {
  const __m128i t0 = idct_madd_round_shift_sse2(in0, x);
  const __m128i t1 = idct_madd_round_shift_sse2(in1, x);
  return _mm_packs_epi32(t0, t1);
}

// Multiply elements by constants and add them together.
static INLINE void butterfly(const __m128i in0, const __m128i in1, const int c0,
                             const int c1, __m128i *const out0,
                             __m128i *const out1) {
  const __m128i cst0 = pair_set_epi16(c0, -c1);
  const __m128i cst1 = pair_set_epi16(c1, c0);
  const __m128i lo = _mm_unpacklo_epi16(in0, in1);
  const __m128i hi = _mm_unpackhi_epi16(in0, in1);
  *out0 = idct_calc_wraplow_sse2(lo, hi, cst0);
  *out1 = idct_calc_wraplow_sse2(lo, hi, cst1);
}

static INLINE __m128i butterfly_cospi16(const __m128i in) {
  const __m128i cst = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i lo = _mm_unpacklo_epi16(in, _mm_setzero_si128());
  const __m128i hi = _mm_unpackhi_epi16(in, _mm_setzero_si128());
  return idct_calc_wraplow_sse2(lo, hi, cst);
}

// Functions to allow 8 bit optimisations to be used when profile 0 is used with
// highbitdepth enabled
static INLINE __m128i load_input_data4(const tran_low_t *data) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m128i zero = _mm_setzero_si128();
  const __m128i in = _mm_load_si128((const __m128i *)data);
  return _mm_packs_epi32(in, zero);
#else
  return _mm_loadl_epi64((const __m128i *)data);
#endif
}

static INLINE __m128i load_input_data8(const tran_low_t *data) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m128i in0 = _mm_load_si128((const __m128i *)data);
  const __m128i in1 = _mm_load_si128((const __m128i *)(data + 4));
  return _mm_packs_epi32(in0, in1);
#else
  return _mm_load_si128((const __m128i *)data);
#endif
}

static INLINE void load_transpose_16bit_8x8(const tran_low_t *input,
                                            const int stride,
                                            __m128i *const in) {
  in[0] = load_input_data8(input + 0 * stride);
  in[1] = load_input_data8(input + 1 * stride);
  in[2] = load_input_data8(input + 2 * stride);
  in[3] = load_input_data8(input + 3 * stride);
  in[4] = load_input_data8(input + 4 * stride);
  in[5] = load_input_data8(input + 5 * stride);
  in[6] = load_input_data8(input + 6 * stride);
  in[7] = load_input_data8(input + 7 * stride);
  transpose_16bit_8x8(in, in);
}

static INLINE void recon_and_store(uint8_t *const dest, const __m128i in_x) {
  const __m128i zero = _mm_setzero_si128();
  __m128i d0 = _mm_loadl_epi64((__m128i *)(dest));
  d0 = _mm_unpacklo_epi8(d0, zero);
  d0 = _mm_add_epi16(in_x, d0);
  d0 = _mm_packus_epi16(d0, d0);
  _mm_storel_epi64((__m128i *)(dest), d0);
}

static INLINE void round_shift_8x8(const __m128i *const in,
                                   __m128i *const out) {
  const __m128i final_rounding = _mm_set1_epi16(1 << 4);

  out[0] = _mm_add_epi16(in[0], final_rounding);
  out[1] = _mm_add_epi16(in[1], final_rounding);
  out[2] = _mm_add_epi16(in[2], final_rounding);
  out[3] = _mm_add_epi16(in[3], final_rounding);
  out[4] = _mm_add_epi16(in[4], final_rounding);
  out[5] = _mm_add_epi16(in[5], final_rounding);
  out[6] = _mm_add_epi16(in[6], final_rounding);
  out[7] = _mm_add_epi16(in[7], final_rounding);

  out[0] = _mm_srai_epi16(out[0], 5);
  out[1] = _mm_srai_epi16(out[1], 5);
  out[2] = _mm_srai_epi16(out[2], 5);
  out[3] = _mm_srai_epi16(out[3], 5);
  out[4] = _mm_srai_epi16(out[4], 5);
  out[5] = _mm_srai_epi16(out[5], 5);
  out[6] = _mm_srai_epi16(out[6], 5);
  out[7] = _mm_srai_epi16(out[7], 5);
}

static INLINE void write_buffer_8x8(const __m128i *const in,
                                    uint8_t *const dest, const int stride) {
  __m128i t[8];

  round_shift_8x8(in, t);

  recon_and_store(dest + 0 * stride, t[0]);
  recon_and_store(dest + 1 * stride, t[1]);
  recon_and_store(dest + 2 * stride, t[2]);
  recon_and_store(dest + 3 * stride, t[3]);
  recon_and_store(dest + 4 * stride, t[4]);
  recon_and_store(dest + 5 * stride, t[5]);
  recon_and_store(dest + 6 * stride, t[6]);
  recon_and_store(dest + 7 * stride, t[7]);
}

static INLINE void recon_and_store4x4_sse2(const __m128i *const in,
                                           uint8_t *const dest,
                                           const int stride) {
  const __m128i zero = _mm_setzero_si128();
  __m128i d[2];

  // Reconstruction and Store
  d[0] = _mm_cvtsi32_si128(*(const int *)(dest));
  d[1] = _mm_cvtsi32_si128(*(const int *)(dest + stride * 3));
  d[0] = _mm_unpacklo_epi32(d[0],
                            _mm_cvtsi32_si128(*(const int *)(dest + stride)));
  d[1] = _mm_unpacklo_epi32(
      _mm_cvtsi32_si128(*(const int *)(dest + stride * 2)), d[1]);
  d[0] = _mm_unpacklo_epi8(d[0], zero);
  d[1] = _mm_unpacklo_epi8(d[1], zero);
  d[0] = _mm_add_epi16(d[0], in[0]);
  d[1] = _mm_add_epi16(d[1], in[1]);
  d[0] = _mm_packus_epi16(d[0], d[1]);

  *(int *)dest = _mm_cvtsi128_si32(d[0]);
  d[0] = _mm_srli_si128(d[0], 4);
  *(int *)(dest + stride) = _mm_cvtsi128_si32(d[0]);
  d[0] = _mm_srli_si128(d[0], 4);
  *(int *)(dest + stride * 2) = _mm_cvtsi128_si32(d[0]);
  d[0] = _mm_srli_si128(d[0], 4);
  *(int *)(dest + stride * 3) = _mm_cvtsi128_si32(d[0]);
}

static INLINE void store_buffer_8x32(__m128i *in, uint8_t *dst, int stride) {
  const __m128i final_rounding = _mm_set1_epi16(1 << 5);
  int j = 0;
  while (j < 32) {
    in[j] = _mm_adds_epi16(in[j], final_rounding);
    in[j + 1] = _mm_adds_epi16(in[j + 1], final_rounding);

    in[j] = _mm_srai_epi16(in[j], 6);
    in[j + 1] = _mm_srai_epi16(in[j + 1], 6);

    recon_and_store(dst, in[j]);
    dst += stride;
    recon_and_store(dst, in[j + 1]);
    dst += stride;
    j += 2;
  }
}

static INLINE void write_buffer_8x1(uint8_t *const dest, const __m128i in) {
  const __m128i final_rounding = _mm_set1_epi16(1 << 5);
  __m128i out;
  out = _mm_adds_epi16(in, final_rounding);
  out = _mm_srai_epi16(out, 6);
  recon_and_store(dest, out);
}

// Only do addition and subtraction butterfly, size = 16, 32
static INLINE void add_sub_butterfly(const __m128i *in, __m128i *out,
                                     int size) {
  int i = 0;
  const int num = size >> 1;
  const int bound = size - 1;
  while (i < num) {
    out[i] = _mm_add_epi16(in[i], in[bound - i]);
    out[bound - i] = _mm_sub_epi16(in[i], in[bound - i]);
    i++;
  }
}

static INLINE void idct8(const __m128i *const in /*in[8]*/,
                         __m128i *const out /*out[8]*/) {
  __m128i step1[8], step2[8];

  // stage 1
  butterfly(in[1], in[7], cospi_28_64, cospi_4_64, &step1[4], &step1[7]);
  butterfly(in[5], in[3], cospi_12_64, cospi_20_64, &step1[5], &step1[6]);

  // stage 2
  butterfly(in[0], in[4], cospi_16_64, cospi_16_64, &step2[1], &step2[0]);
  butterfly(in[2], in[6], cospi_24_64, cospi_8_64, &step2[2], &step2[3]);

  step2[4] = _mm_add_epi16(step1[4], step1[5]);
  step2[5] = _mm_sub_epi16(step1[4], step1[5]);
  step2[6] = _mm_sub_epi16(step1[7], step1[6]);
  step2[7] = _mm_add_epi16(step1[7], step1[6]);

  // stage 3
  step1[0] = _mm_add_epi16(step2[0], step2[3]);
  step1[1] = _mm_add_epi16(step2[1], step2[2]);
  step1[2] = _mm_sub_epi16(step2[1], step2[2]);
  step1[3] = _mm_sub_epi16(step2[0], step2[3]);
  butterfly(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5], &step1[6]);

  // stage 4
  out[0] = _mm_add_epi16(step1[0], step2[7]);
  out[1] = _mm_add_epi16(step1[1], step1[6]);
  out[2] = _mm_add_epi16(step1[2], step1[5]);
  out[3] = _mm_add_epi16(step1[3], step2[4]);
  out[4] = _mm_sub_epi16(step1[3], step2[4]);
  out[5] = _mm_sub_epi16(step1[2], step1[5]);
  out[6] = _mm_sub_epi16(step1[1], step1[6]);
  out[7] = _mm_sub_epi16(step1[0], step2[7]);
}

static INLINE void idct8x8_12_add_kernel_sse2(__m128i *const io /*io[8]*/) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i cp_16_16 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i cp_16_n16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  __m128i step1[8], step2[8], tmp[4];

  transpose_16bit_4x4(io, io);
  // io[0]: 00 10 20 30  01 11 21 31
  // io[1]: 02 12 22 32  03 13 23 33

  // stage 1
  {
    const __m128i cp_28_n4 = pair_set_epi16(cospi_28_64, -cospi_4_64);
    const __m128i cp_4_28 = pair_set_epi16(cospi_4_64, cospi_28_64);
    const __m128i cp_n20_12 = pair_set_epi16(-cospi_20_64, cospi_12_64);
    const __m128i cp_12_20 = pair_set_epi16(cospi_12_64, cospi_20_64);
    const __m128i lo_1 = _mm_unpackhi_epi16(io[0], zero);
    const __m128i lo_3 = _mm_unpackhi_epi16(io[1], zero);
    step1[4] = idct_calc_wraplow_sse2(cp_28_n4, cp_4_28, lo_1);    // step1 4&7
    step1[5] = idct_calc_wraplow_sse2(cp_n20_12, cp_12_20, lo_3);  // step1 5&6
  }

  // stage 2
  {
    const __m128i cp_24_n8 = pair_set_epi16(cospi_24_64, -cospi_8_64);
    const __m128i cp_8_24 = pair_set_epi16(cospi_8_64, cospi_24_64);
    const __m128i lo_0 = _mm_unpacklo_epi16(io[0], zero);
    const __m128i lo_2 = _mm_unpacklo_epi16(io[1], zero);
    const __m128i t = idct_madd_round_shift_sse2(cp_16_16, lo_0);
    step2[0] = _mm_packs_epi32(t, t);                            // step2 0&1
    step2[2] = idct_calc_wraplow_sse2(cp_8_24, cp_24_n8, lo_2);  // step2 3&2
    step2[4] = _mm_add_epi16(step1[4], step1[5]);                // step2 4&7
    step2[5] = _mm_sub_epi16(step1[4], step1[5]);                // step2 5&6
    step2[6] = _mm_unpackhi_epi64(step2[5], zero);               // step2 6
  }

  // stage 3
  {
    const __m128i lo_65 = _mm_unpacklo_epi16(step2[6], step2[5]);
    tmp[0] = _mm_add_epi16(step2[0], step2[2]);                     // step1 0&1
    tmp[1] = _mm_sub_epi16(step2[0], step2[2]);                     // step1 3&2
    step1[2] = _mm_unpackhi_epi64(tmp[1], tmp[0]);                  // step1 2&1
    step1[3] = _mm_unpacklo_epi64(tmp[1], tmp[0]);                  // step1 3&0
    step1[5] = idct_calc_wraplow_sse2(cp_16_n16, cp_16_16, lo_65);  // step1 5&6
  }

  // stage 4
  tmp[0] = _mm_add_epi16(step1[3], step2[4]);  // output 3&0
  tmp[1] = _mm_add_epi16(step1[2], step1[5]);  // output 2&1
  tmp[2] = _mm_sub_epi16(step1[3], step2[4]);  // output 4&7
  tmp[3] = _mm_sub_epi16(step1[2], step1[5]);  // output 5&6

  idct8x8_12_transpose_16bit_4x8(tmp, io);
  io[4] = io[5] = io[6] = io[7] = zero;

  idct8(io, io);
}

static INLINE void idct16_8col(const __m128i *const in /*in[16]*/,
                               __m128i *const out /*out[16]*/) {
  __m128i step1[16], step2[16];

  // stage 2
  butterfly(in[1], in[15], cospi_30_64, cospi_2_64, &step2[8], &step2[15]);
  butterfly(in[9], in[7], cospi_14_64, cospi_18_64, &step2[9], &step2[14]);
  butterfly(in[5], in[11], cospi_22_64, cospi_10_64, &step2[10], &step2[13]);
  butterfly(in[13], in[3], cospi_6_64, cospi_26_64, &step2[11], &step2[12]);

  // stage 3
  butterfly(in[2], in[14], cospi_28_64, cospi_4_64, &step1[4], &step1[7]);
  butterfly(in[10], in[6], cospi_12_64, cospi_20_64, &step1[5], &step1[6]);
  step1[8] = _mm_add_epi16(step2[8], step2[9]);
  step1[9] = _mm_sub_epi16(step2[8], step2[9]);
  step1[10] = _mm_sub_epi16(step2[11], step2[10]);
  step1[11] = _mm_add_epi16(step2[10], step2[11]);
  step1[12] = _mm_add_epi16(step2[12], step2[13]);
  step1[13] = _mm_sub_epi16(step2[12], step2[13]);
  step1[14] = _mm_sub_epi16(step2[15], step2[14]);
  step1[15] = _mm_add_epi16(step2[14], step2[15]);

  // stage 4
  butterfly(in[0], in[8], cospi_16_64, cospi_16_64, &step2[1], &step2[0]);
  butterfly(in[4], in[12], cospi_24_64, cospi_8_64, &step2[2], &step2[3]);
  butterfly(step1[14], step1[9], cospi_24_64, cospi_8_64, &step2[9],
            &step2[14]);
  butterfly(step1[10], step1[13], -cospi_8_64, -cospi_24_64, &step2[13],
            &step2[10]);
  step2[5] = _mm_sub_epi16(step1[4], step1[5]);
  step1[4] = _mm_add_epi16(step1[4], step1[5]);
  step2[6] = _mm_sub_epi16(step1[7], step1[6]);
  step1[7] = _mm_add_epi16(step1[6], step1[7]);
  step2[8] = step1[8];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[15] = step1[15];

  // stage 5
  step1[0] = _mm_add_epi16(step2[0], step2[3]);
  step1[1] = _mm_add_epi16(step2[1], step2[2]);
  step1[2] = _mm_sub_epi16(step2[1], step2[2]);
  step1[3] = _mm_sub_epi16(step2[0], step2[3]);
  butterfly(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5], &step1[6]);
  step1[8] = _mm_add_epi16(step2[8], step2[11]);
  step1[9] = _mm_add_epi16(step2[9], step2[10]);
  step1[10] = _mm_sub_epi16(step2[9], step2[10]);
  step1[11] = _mm_sub_epi16(step2[8], step2[11]);
  step1[12] = _mm_sub_epi16(step2[15], step2[12]);
  step1[13] = _mm_sub_epi16(step2[14], step2[13]);
  step1[14] = _mm_add_epi16(step2[14], step2[13]);
  step1[15] = _mm_add_epi16(step2[15], step2[12]);

  // stage 6
  step2[0] = _mm_add_epi16(step1[0], step1[7]);
  step2[1] = _mm_add_epi16(step1[1], step1[6]);
  step2[2] = _mm_add_epi16(step1[2], step1[5]);
  step2[3] = _mm_add_epi16(step1[3], step1[4]);
  step2[4] = _mm_sub_epi16(step1[3], step1[4]);
  step2[5] = _mm_sub_epi16(step1[2], step1[5]);
  step2[6] = _mm_sub_epi16(step1[1], step1[6]);
  step2[7] = _mm_sub_epi16(step1[0], step1[7]);
  butterfly(step1[13], step1[10], cospi_16_64, cospi_16_64, &step2[10],
            &step2[13]);
  butterfly(step1[12], step1[11], cospi_16_64, cospi_16_64, &step2[11],
            &step2[12]);

  // stage 7
  out[0] = _mm_add_epi16(step2[0], step1[15]);
  out[1] = _mm_add_epi16(step2[1], step1[14]);
  out[2] = _mm_add_epi16(step2[2], step2[13]);
  out[3] = _mm_add_epi16(step2[3], step2[12]);
  out[4] = _mm_add_epi16(step2[4], step2[11]);
  out[5] = _mm_add_epi16(step2[5], step2[10]);
  out[6] = _mm_add_epi16(step2[6], step1[9]);
  out[7] = _mm_add_epi16(step2[7], step1[8]);
  out[8] = _mm_sub_epi16(step2[7], step1[8]);
  out[9] = _mm_sub_epi16(step2[6], step1[9]);
  out[10] = _mm_sub_epi16(step2[5], step2[10]);
  out[11] = _mm_sub_epi16(step2[4], step2[11]);
  out[12] = _mm_sub_epi16(step2[3], step2[12]);
  out[13] = _mm_sub_epi16(step2[2], step2[13]);
  out[14] = _mm_sub_epi16(step2[1], step1[14]);
  out[15] = _mm_sub_epi16(step2[0], step1[15]);
}

static INLINE void idct16x16_10_pass1(const __m128i *const input /*input[4]*/,
                                      __m128i *const output /*output[16]*/) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i k__cospi_p16_p16 = pair_set_epi16(cospi_16_64, cospi_16_64);
  const __m128i k__cospi_m16_p16 = pair_set_epi16(-cospi_16_64, cospi_16_64);
  __m128i step1[16], step2[16];

  transpose_16bit_4x4(input, output);

  // stage 2
  {
    const __m128i k__cospi_p30_m02 = pair_set_epi16(cospi_30_64, -cospi_2_64);
    const __m128i k__cospi_p02_p30 = pair_set_epi16(cospi_2_64, cospi_30_64);
    const __m128i k__cospi_p06_m26 = pair_set_epi16(cospi_6_64, -cospi_26_64);
    const __m128i k__cospi_p26_p06 = pair_set_epi16(cospi_26_64, cospi_6_64);
    const __m128i lo_1_15 = _mm_unpackhi_epi16(output[0], zero);
    const __m128i lo_13_3 = _mm_unpackhi_epi16(zero, output[1]);
    step2[8] = idct_calc_wraplow_sse2(k__cospi_p30_m02, k__cospi_p02_p30,
                                      lo_1_15);  // step2 8&15
    step2[11] = idct_calc_wraplow_sse2(k__cospi_p06_m26, k__cospi_p26_p06,
                                       lo_13_3);  // step2 11&12
  }

  // stage 3
  {
    const __m128i k__cospi_p28_m04 = pair_set_epi16(cospi_28_64, -cospi_4_64);
    const __m128i k__cospi_p04_p28 = pair_set_epi16(cospi_4_64, cospi_28_64);
    const __m128i lo_2_14 = _mm_unpacklo_epi16(output[1], zero);
    step1[4] = idct_calc_wraplow_sse2(k__cospi_p28_m04, k__cospi_p04_p28,
                                      lo_2_14);  // step1 4&7
    step1[13] = _mm_unpackhi_epi64(step2[11], zero);
    step1[14] = _mm_unpackhi_epi64(step2[8], zero);
  }

  // stage 4
  {
    const __m128i k__cospi_m08_p24 = pair_set_epi16(-cospi_8_64, cospi_24_64);
    const __m128i k__cospi_p24_p08 = pair_set_epi16(cospi_24_64, cospi_8_64);
    const __m128i k__cospi_m24_m08 = pair_set_epi16(-cospi_24_64, -cospi_8_64);
    const __m128i lo_0_8 = _mm_unpacklo_epi16(output[0], zero);
    const __m128i lo_9_14 = _mm_unpacklo_epi16(step2[8], step1[14]);
    const __m128i lo_10_13 = _mm_unpacklo_epi16(step2[11], step1[13]);
    const __m128i t = idct_madd_round_shift_sse2(lo_0_8, k__cospi_p16_p16);
    step1[0] = _mm_packs_epi32(t, t);  // step2 0&1
    step2[9] = idct_calc_wraplow_sse2(k__cospi_m08_p24, k__cospi_p24_p08,
                                      lo_9_14);  // step2 9&14
    step2[10] = idct_calc_wraplow_sse2(k__cospi_m24_m08, k__cospi_m08_p24,
                                       lo_10_13);  // step2 10&13
    step2[6] = _mm_unpackhi_epi64(step1[4], zero);
  }

  // stage 5
  {
    const __m128i lo_5_6 = _mm_unpacklo_epi16(step1[4], step2[6]);
    step1[6] = idct_calc_wraplow_sse2(k__cospi_p16_p16, k__cospi_m16_p16,
                                      lo_5_6);  // step1 6&5
    step1[8] = _mm_add_epi16(step2[8], step2[11]);
    step1[9] = _mm_add_epi16(step2[9], step2[10]);
    step1[10] = _mm_sub_epi16(step2[9], step2[10]);
    step1[11] = _mm_sub_epi16(step2[8], step2[11]);
    step1[12] = _mm_unpackhi_epi64(step1[11], zero);
    step1[13] = _mm_unpackhi_epi64(step1[10], zero);
    step1[14] = _mm_unpackhi_epi64(step1[9], zero);
    step1[15] = _mm_unpackhi_epi64(step1[8], zero);
  }

  // stage 6
  {
    const __m128i lo_10_13 = _mm_unpacklo_epi16(step1[10], step1[13]);
    const __m128i lo_11_12 = _mm_unpacklo_epi16(step1[11], step1[12]);
    step2[10] = idct_calc_wraplow_sse2(k__cospi_m16_p16, k__cospi_p16_p16,
                                       lo_10_13);  // step2 10&13
    step2[11] = idct_calc_wraplow_sse2(k__cospi_m16_p16, k__cospi_p16_p16,
                                       lo_11_12);  // step2 11&12
    step2[13] = _mm_unpackhi_epi64(step2[10], zero);
    step2[12] = _mm_unpackhi_epi64(step2[11], zero);
    step2[3] = _mm_add_epi16(step1[0], step1[4]);
    step2[1] = _mm_add_epi16(step1[0], step1[6]);
    step2[6] = _mm_sub_epi16(step1[0], step1[6]);
    step2[4] = _mm_sub_epi16(step1[0], step1[4]);
    step2[0] = _mm_unpackhi_epi64(step2[3], zero);
    step2[2] = _mm_unpackhi_epi64(step2[1], zero);
    step2[5] = _mm_unpackhi_epi64(step2[6], zero);
    step2[7] = _mm_unpackhi_epi64(step2[4], zero);
  }

  // stage 7. Left 8x16 only.
  output[0] = _mm_add_epi16(step2[0], step1[15]);
  output[1] = _mm_add_epi16(step2[1], step1[14]);
  output[2] = _mm_add_epi16(step2[2], step2[13]);
  output[3] = _mm_add_epi16(step2[3], step2[12]);
  output[4] = _mm_add_epi16(step2[4], step2[11]);
  output[5] = _mm_add_epi16(step2[5], step2[10]);
  output[6] = _mm_add_epi16(step2[6], step1[9]);
  output[7] = _mm_add_epi16(step2[7], step1[8]);
  output[8] = _mm_sub_epi16(step2[7], step1[8]);
  output[9] = _mm_sub_epi16(step2[6], step1[9]);
  output[10] = _mm_sub_epi16(step2[5], step2[10]);
  output[11] = _mm_sub_epi16(step2[4], step2[11]);
  output[12] = _mm_sub_epi16(step2[3], step2[12]);
  output[13] = _mm_sub_epi16(step2[2], step2[13]);
  output[14] = _mm_sub_epi16(step2[1], step1[14]);
  output[15] = _mm_sub_epi16(step2[0], step1[15]);
}

static INLINE void idct16x16_10_pass2(__m128i *const l /*l[8]*/,
                                      __m128i *const io /*io[16]*/) {
  const __m128i zero = _mm_setzero_si128();
  __m128i step1[16], step2[16];

  transpose_16bit_4x8(l, io);

  // stage 2
  butterfly(io[1], zero, cospi_30_64, cospi_2_64, &step2[8], &step2[15]);
  butterfly(zero, io[3], cospi_6_64, cospi_26_64, &step2[11], &step2[12]);

  // stage 3
  butterfly(io[2], zero, cospi_28_64, cospi_4_64, &step1[4], &step1[7]);

  // stage 4
  step1[0] = butterfly_cospi16(io[0]);
  butterfly(step2[15], step2[8], cospi_24_64, cospi_8_64, &step2[9],
            &step2[14]);
  butterfly(step2[11], step2[12], -cospi_8_64, -cospi_24_64, &step2[13],
            &step2[10]);

  // stage 5
  butterfly(step1[7], step1[4], cospi_16_64, cospi_16_64, &step1[5], &step1[6]);
  step1[8] = _mm_add_epi16(step2[8], step2[11]);
  step1[9] = _mm_add_epi16(step2[9], step2[10]);
  step1[10] = _mm_sub_epi16(step2[9], step2[10]);
  step1[11] = _mm_sub_epi16(step2[8], step2[11]);
  step1[12] = _mm_sub_epi16(step2[15], step2[12]);
  step1[13] = _mm_sub_epi16(step2[14], step2[13]);
  step1[14] = _mm_add_epi16(step2[14], step2[13]);
  step1[15] = _mm_add_epi16(step2[15], step2[12]);

  // stage 6
  step2[0] = _mm_add_epi16(step1[0], step1[7]);
  step2[1] = _mm_add_epi16(step1[0], step1[6]);
  step2[2] = _mm_add_epi16(step1[0], step1[5]);
  step2[3] = _mm_add_epi16(step1[0], step1[4]);
  step2[4] = _mm_sub_epi16(step1[0], step1[4]);
  step2[5] = _mm_sub_epi16(step1[0], step1[5]);
  step2[6] = _mm_sub_epi16(step1[0], step1[6]);
  step2[7] = _mm_sub_epi16(step1[0], step1[7]);
  butterfly(step1[13], step1[10], cospi_16_64, cospi_16_64, &step2[10],
            &step2[13]);
  butterfly(step1[12], step1[11], cospi_16_64, cospi_16_64, &step2[11],
            &step2[12]);

  // stage 7
  io[0] = _mm_add_epi16(step2[0], step1[15]);
  io[1] = _mm_add_epi16(step2[1], step1[14]);
  io[2] = _mm_add_epi16(step2[2], step2[13]);
  io[3] = _mm_add_epi16(step2[3], step2[12]);
  io[4] = _mm_add_epi16(step2[4], step2[11]);
  io[5] = _mm_add_epi16(step2[5], step2[10]);
  io[6] = _mm_add_epi16(step2[6], step1[9]);
  io[7] = _mm_add_epi16(step2[7], step1[8]);
  io[8] = _mm_sub_epi16(step2[7], step1[8]);
  io[9] = _mm_sub_epi16(step2[6], step1[9]);
  io[10] = _mm_sub_epi16(step2[5], step2[10]);
  io[11] = _mm_sub_epi16(step2[4], step2[11]);
  io[12] = _mm_sub_epi16(step2[3], step2[12]);
  io[13] = _mm_sub_epi16(step2[2], step2[13]);
  io[14] = _mm_sub_epi16(step2[1], step1[14]);
  io[15] = _mm_sub_epi16(step2[0], step1[15]);
}

static INLINE void idct32_8x32_quarter_2_stage_4_to_6(
    __m128i *const step1 /*step1[16]*/, __m128i *const out /*out[16]*/) {
  __m128i step2[32];

  // stage 4
  step2[8] = step1[8];
  step2[15] = step1[15];
  butterfly(step1[14], step1[9], cospi_24_64, cospi_8_64, &step2[9],
            &step2[14]);
  butterfly(step1[13], step1[10], -cospi_8_64, cospi_24_64, &step2[10],
            &step2[13]);
  step2[11] = step1[11];
  step2[12] = step1[12];

  // stage 5
  step1[8] = _mm_add_epi16(step2[8], step2[11]);
  step1[9] = _mm_add_epi16(step2[9], step2[10]);
  step1[10] = _mm_sub_epi16(step2[9], step2[10]);
  step1[11] = _mm_sub_epi16(step2[8], step2[11]);
  step1[12] = _mm_sub_epi16(step2[15], step2[12]);
  step1[13] = _mm_sub_epi16(step2[14], step2[13]);
  step1[14] = _mm_add_epi16(step2[14], step2[13]);
  step1[15] = _mm_add_epi16(step2[15], step2[12]);

  // stage 6
  out[8] = step1[8];
  out[9] = step1[9];
  butterfly(step1[13], step1[10], cospi_16_64, cospi_16_64, &out[10], &out[13]);
  butterfly(step1[12], step1[11], cospi_16_64, cospi_16_64, &out[11], &out[12]);
  out[14] = step1[14];
  out[15] = step1[15];
}

static INLINE void idct32_8x32_quarter_3_4_stage_4_to_7(
    __m128i *const step1 /*step1[32]*/, __m128i *const out /*out[32]*/) {
  __m128i step2[32];

  // stage 4
  step2[16] = _mm_add_epi16(step1[16], step1[19]);
  step2[17] = _mm_add_epi16(step1[17], step1[18]);
  step2[18] = _mm_sub_epi16(step1[17], step1[18]);
  step2[19] = _mm_sub_epi16(step1[16], step1[19]);
  step2[20] = _mm_sub_epi16(step1[23], step1[20]);
  step2[21] = _mm_sub_epi16(step1[22], step1[21]);
  step2[22] = _mm_add_epi16(step1[22], step1[21]);
  step2[23] = _mm_add_epi16(step1[23], step1[20]);

  step2[24] = _mm_add_epi16(step1[24], step1[27]);
  step2[25] = _mm_add_epi16(step1[25], step1[26]);
  step2[26] = _mm_sub_epi16(step1[25], step1[26]);
  step2[27] = _mm_sub_epi16(step1[24], step1[27]);
  step2[28] = _mm_sub_epi16(step1[31], step1[28]);
  step2[29] = _mm_sub_epi16(step1[30], step1[29]);
  step2[30] = _mm_add_epi16(step1[29], step1[30]);
  step2[31] = _mm_add_epi16(step1[28], step1[31]);

  // stage 5
  step1[16] = step2[16];
  step1[17] = step2[17];
  butterfly(step2[29], step2[18], cospi_24_64, cospi_8_64, &step1[18],
            &step1[29]);
  butterfly(step2[28], step2[19], cospi_24_64, cospi_8_64, &step1[19],
            &step1[28]);
  butterfly(step2[27], step2[20], -cospi_8_64, cospi_24_64, &step1[20],
            &step1[27]);
  butterfly(step2[26], step2[21], -cospi_8_64, cospi_24_64, &step1[21],
            &step1[26]);
  step1[22] = step2[22];
  step1[23] = step2[23];
  step1[24] = step2[24];
  step1[25] = step2[25];
  step1[30] = step2[30];
  step1[31] = step2[31];

  // stage 6
  out[16] = _mm_add_epi16(step1[16], step1[23]);
  out[17] = _mm_add_epi16(step1[17], step1[22]);
  out[18] = _mm_add_epi16(step1[18], step1[21]);
  out[19] = _mm_add_epi16(step1[19], step1[20]);
  step2[20] = _mm_sub_epi16(step1[19], step1[20]);
  step2[21] = _mm_sub_epi16(step1[18], step1[21]);
  step2[22] = _mm_sub_epi16(step1[17], step1[22]);
  step2[23] = _mm_sub_epi16(step1[16], step1[23]);

  step2[24] = _mm_sub_epi16(step1[31], step1[24]);
  step2[25] = _mm_sub_epi16(step1[30], step1[25]);
  step2[26] = _mm_sub_epi16(step1[29], step1[26]);
  step2[27] = _mm_sub_epi16(step1[28], step1[27]);
  out[28] = _mm_add_epi16(step1[27], step1[28]);
  out[29] = _mm_add_epi16(step1[26], step1[29]);
  out[30] = _mm_add_epi16(step1[25], step1[30]);
  out[31] = _mm_add_epi16(step1[24], step1[31]);

  // stage 7
  butterfly(step2[27], step2[20], cospi_16_64, cospi_16_64, &out[20], &out[27]);
  butterfly(step2[26], step2[21], cospi_16_64, cospi_16_64, &out[21], &out[26]);
  butterfly(step2[25], step2[22], cospi_16_64, cospi_16_64, &out[22], &out[25]);
  butterfly(step2[24], step2[23], cospi_16_64, cospi_16_64, &out[23], &out[24]);
}

void idct4_sse2(__m128i *const in);
void vpx_idct8_sse2(__m128i *const in);
void idct16_sse2(__m128i *const in0, __m128i *const in1);
void iadst4_sse2(__m128i *const in);
void iadst8_sse2(__m128i *const in);
void vpx_iadst16_8col_sse2(__m128i *const in);
void iadst16_sse2(__m128i *const in0, __m128i *const in1);
void idct32_1024_8x32(const __m128i *const in, __m128i *const out);
void idct32_34_8x32_sse2(const __m128i *const in, __m128i *const out);
void idct32_34_8x32_ssse3(const __m128i *const in, __m128i *const out);

#endif  // VPX_VPX_DSP_X86_INV_TXFM_SSE2_H_
