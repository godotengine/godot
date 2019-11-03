/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_DSP_X86_INV_TXFM_SSE2_H_
#define VPX_DSP_X86_INV_TXFM_SSE2_H_

#include <emmintrin.h>  // SSE2
#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/inv_txfm.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

// perform 8x8 transpose
static INLINE void array_transpose_8x8(__m128i *in, __m128i *res) {
  const __m128i tr0_0 = _mm_unpacklo_epi16(in[0], in[1]);
  const __m128i tr0_1 = _mm_unpacklo_epi16(in[2], in[3]);
  const __m128i tr0_2 = _mm_unpackhi_epi16(in[0], in[1]);
  const __m128i tr0_3 = _mm_unpackhi_epi16(in[2], in[3]);
  const __m128i tr0_4 = _mm_unpacklo_epi16(in[4], in[5]);
  const __m128i tr0_5 = _mm_unpacklo_epi16(in[6], in[7]);
  const __m128i tr0_6 = _mm_unpackhi_epi16(in[4], in[5]);
  const __m128i tr0_7 = _mm_unpackhi_epi16(in[6], in[7]);

  const __m128i tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1);
  const __m128i tr1_1 = _mm_unpacklo_epi32(tr0_4, tr0_5);
  const __m128i tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1);
  const __m128i tr1_3 = _mm_unpackhi_epi32(tr0_4, tr0_5);
  const __m128i tr1_4 = _mm_unpacklo_epi32(tr0_2, tr0_3);
  const __m128i tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7);
  const __m128i tr1_6 = _mm_unpackhi_epi32(tr0_2, tr0_3);
  const __m128i tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7);

  res[0] = _mm_unpacklo_epi64(tr1_0, tr1_1);
  res[1] = _mm_unpackhi_epi64(tr1_0, tr1_1);
  res[2] = _mm_unpacklo_epi64(tr1_2, tr1_3);
  res[3] = _mm_unpackhi_epi64(tr1_2, tr1_3);
  res[4] = _mm_unpacklo_epi64(tr1_4, tr1_5);
  res[5] = _mm_unpackhi_epi64(tr1_4, tr1_5);
  res[6] = _mm_unpacklo_epi64(tr1_6, tr1_7);
  res[7] = _mm_unpackhi_epi64(tr1_6, tr1_7);
}

#define TRANSPOSE_8X4(in0, in1, in2, in3, out0, out1) \
  {                                                     \
    const __m128i tr0_0 = _mm_unpacklo_epi16(in0, in1); \
    const __m128i tr0_1 = _mm_unpacklo_epi16(in2, in3); \
                                                        \
    in0 = _mm_unpacklo_epi32(tr0_0, tr0_1);  /* i1 i0 */  \
    in1 = _mm_unpackhi_epi32(tr0_0, tr0_1);  /* i3 i2 */  \
  }

static INLINE void array_transpose_4X8(__m128i *in, __m128i * out) {
  const __m128i tr0_0 = _mm_unpacklo_epi16(in[0], in[1]);
  const __m128i tr0_1 = _mm_unpacklo_epi16(in[2], in[3]);
  const __m128i tr0_4 = _mm_unpacklo_epi16(in[4], in[5]);
  const __m128i tr0_5 = _mm_unpacklo_epi16(in[6], in[7]);

  const __m128i tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1);
  const __m128i tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1);
  const __m128i tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5);
  const __m128i tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5);

  out[0] = _mm_unpacklo_epi64(tr1_0, tr1_4);
  out[1] = _mm_unpackhi_epi64(tr1_0, tr1_4);
  out[2] = _mm_unpacklo_epi64(tr1_2, tr1_6);
  out[3] = _mm_unpackhi_epi64(tr1_2, tr1_6);
}

static INLINE void array_transpose_16x16(__m128i *res0, __m128i *res1) {
  __m128i tbuf[8];
  array_transpose_8x8(res0, res0);
  array_transpose_8x8(res1, tbuf);
  array_transpose_8x8(res0 + 8, res1);
  array_transpose_8x8(res1 + 8, res1 + 8);

  res0[8] = tbuf[0];
  res0[9] = tbuf[1];
  res0[10] = tbuf[2];
  res0[11] = tbuf[3];
  res0[12] = tbuf[4];
  res0[13] = tbuf[5];
  res0[14] = tbuf[6];
  res0[15] = tbuf[7];
}

// Function to allow 8 bit optimisations to be used when profile 0 is used with
// highbitdepth enabled
static INLINE __m128i load_input_data(const tran_low_t *data) {
#if CONFIG_VP9_HIGHBITDEPTH
  return octa_set_epi16(data[0], data[1], data[2], data[3], data[4], data[5],
      data[6], data[7]);
#else
  return _mm_load_si128((const __m128i *)data);
#endif
}

static INLINE void load_buffer_8x16(const tran_low_t *input, __m128i *in) {
  in[0]  = load_input_data(input + 0 * 16);
  in[1]  = load_input_data(input + 1 * 16);
  in[2]  = load_input_data(input + 2 * 16);
  in[3]  = load_input_data(input + 3 * 16);
  in[4]  = load_input_data(input + 4 * 16);
  in[5]  = load_input_data(input + 5 * 16);
  in[6]  = load_input_data(input + 6 * 16);
  in[7]  = load_input_data(input + 7 * 16);

  in[8]  = load_input_data(input + 8 * 16);
  in[9]  = load_input_data(input + 9 * 16);
  in[10]  = load_input_data(input + 10 * 16);
  in[11]  = load_input_data(input + 11 * 16);
  in[12]  = load_input_data(input + 12 * 16);
  in[13]  = load_input_data(input + 13 * 16);
  in[14]  = load_input_data(input + 14 * 16);
  in[15]  = load_input_data(input + 15 * 16);
}

#define RECON_AND_STORE(dest, in_x) \
  {                                                     \
     __m128i d0 = _mm_loadl_epi64((__m128i *)(dest)); \
      d0 = _mm_unpacklo_epi8(d0, zero); \
      d0 = _mm_add_epi16(in_x, d0); \
      d0 = _mm_packus_epi16(d0, d0); \
      _mm_storel_epi64((__m128i *)(dest), d0); \
  }

static INLINE void write_buffer_8x16(uint8_t *dest, __m128i *in, int stride) {
  const __m128i final_rounding = _mm_set1_epi16(1<<5);
  const __m128i zero = _mm_setzero_si128();
  // Final rounding and shift
  in[0] = _mm_adds_epi16(in[0], final_rounding);
  in[1] = _mm_adds_epi16(in[1], final_rounding);
  in[2] = _mm_adds_epi16(in[2], final_rounding);
  in[3] = _mm_adds_epi16(in[3], final_rounding);
  in[4] = _mm_adds_epi16(in[4], final_rounding);
  in[5] = _mm_adds_epi16(in[5], final_rounding);
  in[6] = _mm_adds_epi16(in[6], final_rounding);
  in[7] = _mm_adds_epi16(in[7], final_rounding);
  in[8] = _mm_adds_epi16(in[8], final_rounding);
  in[9] = _mm_adds_epi16(in[9], final_rounding);
  in[10] = _mm_adds_epi16(in[10], final_rounding);
  in[11] = _mm_adds_epi16(in[11], final_rounding);
  in[12] = _mm_adds_epi16(in[12], final_rounding);
  in[13] = _mm_adds_epi16(in[13], final_rounding);
  in[14] = _mm_adds_epi16(in[14], final_rounding);
  in[15] = _mm_adds_epi16(in[15], final_rounding);

  in[0] = _mm_srai_epi16(in[0], 6);
  in[1] = _mm_srai_epi16(in[1], 6);
  in[2] = _mm_srai_epi16(in[2], 6);
  in[3] = _mm_srai_epi16(in[3], 6);
  in[4] = _mm_srai_epi16(in[4], 6);
  in[5] = _mm_srai_epi16(in[5], 6);
  in[6] = _mm_srai_epi16(in[6], 6);
  in[7] = _mm_srai_epi16(in[7], 6);
  in[8] = _mm_srai_epi16(in[8], 6);
  in[9] = _mm_srai_epi16(in[9], 6);
  in[10] = _mm_srai_epi16(in[10], 6);
  in[11] = _mm_srai_epi16(in[11], 6);
  in[12] = _mm_srai_epi16(in[12], 6);
  in[13] = _mm_srai_epi16(in[13], 6);
  in[14] = _mm_srai_epi16(in[14], 6);
  in[15] = _mm_srai_epi16(in[15], 6);

  RECON_AND_STORE(dest +  0 * stride, in[0]);
  RECON_AND_STORE(dest +  1 * stride, in[1]);
  RECON_AND_STORE(dest +  2 * stride, in[2]);
  RECON_AND_STORE(dest +  3 * stride, in[3]);
  RECON_AND_STORE(dest +  4 * stride, in[4]);
  RECON_AND_STORE(dest +  5 * stride, in[5]);
  RECON_AND_STORE(dest +  6 * stride, in[6]);
  RECON_AND_STORE(dest +  7 * stride, in[7]);
  RECON_AND_STORE(dest +  8 * stride, in[8]);
  RECON_AND_STORE(dest +  9 * stride, in[9]);
  RECON_AND_STORE(dest + 10 * stride, in[10]);
  RECON_AND_STORE(dest + 11 * stride, in[11]);
  RECON_AND_STORE(dest + 12 * stride, in[12]);
  RECON_AND_STORE(dest + 13 * stride, in[13]);
  RECON_AND_STORE(dest + 14 * stride, in[14]);
  RECON_AND_STORE(dest + 15 * stride, in[15]);
}

void idct4_sse2(__m128i *in);
void idct8_sse2(__m128i *in);
void idct16_sse2(__m128i *in0, __m128i *in1);
void iadst4_sse2(__m128i *in);
void iadst8_sse2(__m128i *in);
void iadst16_sse2(__m128i *in0, __m128i *in1);

#endif  // VPX_DSP_X86_INV_TXFM_SSE2_H_
