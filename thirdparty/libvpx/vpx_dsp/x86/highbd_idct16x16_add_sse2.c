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
#include "vpx_dsp/x86/highbd_inv_txfm_sse2.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/transpose_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

static INLINE void highbd_idct16_4col_stage5(const __m128i *const in,
                                             __m128i *const out) {
  // stage 5
  out[0] = _mm_add_epi32(in[0], in[3]);
  out[1] = _mm_add_epi32(in[1], in[2]);
  out[2] = _mm_sub_epi32(in[1], in[2]);
  out[3] = _mm_sub_epi32(in[0], in[3]);
  highbd_butterfly_cospi16_sse2(in[6], in[5], &out[6], &out[5]);
  out[8] = _mm_add_epi32(in[8], in[11]);
  out[9] = _mm_add_epi32(in[9], in[10]);
  out[10] = _mm_sub_epi32(in[9], in[10]);
  out[11] = _mm_sub_epi32(in[8], in[11]);
  out[12] = _mm_sub_epi32(in[15], in[12]);
  out[13] = _mm_sub_epi32(in[14], in[13]);
  out[14] = _mm_add_epi32(in[14], in[13]);
  out[15] = _mm_add_epi32(in[15], in[12]);
}

static INLINE void highbd_idct16_4col_stage6(const __m128i *const in,
                                             __m128i *const out) {
  out[0] = _mm_add_epi32(in[0], in[7]);
  out[1] = _mm_add_epi32(in[1], in[6]);
  out[2] = _mm_add_epi32(in[2], in[5]);
  out[3] = _mm_add_epi32(in[3], in[4]);
  out[4] = _mm_sub_epi32(in[3], in[4]);
  out[5] = _mm_sub_epi32(in[2], in[5]);
  out[6] = _mm_sub_epi32(in[1], in[6]);
  out[7] = _mm_sub_epi32(in[0], in[7]);
  out[8] = in[8];
  out[9] = in[9];
  highbd_butterfly_cospi16_sse2(in[13], in[10], &out[13], &out[10]);
  highbd_butterfly_cospi16_sse2(in[12], in[11], &out[12], &out[11]);
  out[14] = in[14];
  out[15] = in[15];
}

static INLINE void highbd_idct16_4col(__m128i *const io /*io[16]*/) {
  __m128i step1[16], step2[16];

  // stage 2
  highbd_butterfly_sse2(io[1], io[15], cospi_30_64, cospi_2_64, &step2[8],
                        &step2[15]);
  highbd_butterfly_sse2(io[9], io[7], cospi_14_64, cospi_18_64, &step2[9],
                        &step2[14]);
  highbd_butterfly_sse2(io[5], io[11], cospi_22_64, cospi_10_64, &step2[10],
                        &step2[13]);
  highbd_butterfly_sse2(io[13], io[3], cospi_6_64, cospi_26_64, &step2[11],
                        &step2[12]);

  // stage 3
  highbd_butterfly_sse2(io[2], io[14], cospi_28_64, cospi_4_64, &step1[4],
                        &step1[7]);
  highbd_butterfly_sse2(io[10], io[6], cospi_12_64, cospi_20_64, &step1[5],
                        &step1[6]);
  step1[8] = _mm_add_epi32(step2[8], step2[9]);
  step1[9] = _mm_sub_epi32(step2[8], step2[9]);
  step1[10] = _mm_sub_epi32(step2[10], step2[11]);  // step1[10] = -step1[10]
  step1[11] = _mm_add_epi32(step2[10], step2[11]);
  step1[12] = _mm_add_epi32(step2[13], step2[12]);
  step1[13] = _mm_sub_epi32(step2[13], step2[12]);  // step1[13] = -step1[13]
  step1[14] = _mm_sub_epi32(step2[15], step2[14]);
  step1[15] = _mm_add_epi32(step2[15], step2[14]);

  // stage 4
  highbd_butterfly_cospi16_sse2(io[0], io[8], &step2[0], &step2[1]);
  highbd_butterfly_sse2(io[4], io[12], cospi_24_64, cospi_8_64, &step2[2],
                        &step2[3]);
  highbd_butterfly_sse2(step1[14], step1[9], cospi_24_64, cospi_8_64, &step2[9],
                        &step2[14]);
  highbd_butterfly_sse2(step1[10], step1[13], cospi_8_64, cospi_24_64,
                        &step2[13], &step2[10]);
  step2[5] = _mm_sub_epi32(step1[4], step1[5]);
  step1[4] = _mm_add_epi32(step1[4], step1[5]);
  step2[6] = _mm_sub_epi32(step1[7], step1[6]);
  step1[7] = _mm_add_epi32(step1[7], step1[6]);
  step2[8] = step1[8];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[15] = step1[15];

  highbd_idct16_4col_stage5(step2, step1);
  highbd_idct16_4col_stage6(step1, step2);
  highbd_idct16_4col_stage7(step2, io);
}

static INLINE void highbd_idct16x16_38_4col(__m128i *const io /*io[16]*/) {
  __m128i step1[16], step2[16];
  __m128i temp1[2], sign[2];

  // stage 2
  highbd_partial_butterfly_sse2(io[1], cospi_30_64, cospi_2_64, &step2[8],
                                &step2[15]);
  highbd_partial_butterfly_neg_sse2(io[7], cospi_14_64, cospi_18_64, &step2[9],
                                    &step2[14]);
  highbd_partial_butterfly_sse2(io[5], cospi_22_64, cospi_10_64, &step2[10],
                                &step2[13]);
  highbd_partial_butterfly_neg_sse2(io[3], cospi_6_64, cospi_26_64, &step2[11],
                                    &step2[12]);

  // stage 3
  highbd_partial_butterfly_sse2(io[2], cospi_28_64, cospi_4_64, &step1[4],
                                &step1[7]);
  highbd_partial_butterfly_neg_sse2(io[6], cospi_12_64, cospi_20_64, &step1[5],
                                    &step1[6]);
  step1[8] = _mm_add_epi32(step2[8], step2[9]);
  step1[9] = _mm_sub_epi32(step2[8], step2[9]);
  step1[10] = _mm_sub_epi32(step2[10], step2[11]);  // step1[10] = -step1[10]
  step1[11] = _mm_add_epi32(step2[10], step2[11]);
  step1[12] = _mm_add_epi32(step2[13], step2[12]);
  step1[13] = _mm_sub_epi32(step2[13], step2[12]);  // step1[13] = -step1[13]
  step1[14] = _mm_sub_epi32(step2[15], step2[14]);
  step1[15] = _mm_add_epi32(step2[15], step2[14]);

  // stage 4
  abs_extend_64bit_sse2(io[0], temp1, sign);
  step2[0] = multiplication_round_shift_sse2(temp1, sign, cospi_16_64);
  step2[1] = step2[0];
  highbd_partial_butterfly_sse2(io[4], cospi_24_64, cospi_8_64, &step2[2],
                                &step2[3]);
  highbd_butterfly_sse2(step1[14], step1[9], cospi_24_64, cospi_8_64, &step2[9],
                        &step2[14]);
  highbd_butterfly_sse2(step1[10], step1[13], cospi_8_64, cospi_24_64,
                        &step2[13], &step2[10]);
  step2[5] = _mm_sub_epi32(step1[4], step1[5]);
  step1[4] = _mm_add_epi32(step1[4], step1[5]);
  step2[6] = _mm_sub_epi32(step1[7], step1[6]);
  step1[7] = _mm_add_epi32(step1[7], step1[6]);
  step2[8] = step1[8];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[15] = step1[15];

  highbd_idct16_4col_stage5(step2, step1);
  highbd_idct16_4col_stage6(step1, step2);
  highbd_idct16_4col_stage7(step2, io);
}

static INLINE void highbd_idct16x16_10_4col(__m128i *const io /*io[16]*/) {
  __m128i step1[16], step2[16];
  __m128i temp[2], sign[2];

  // stage 2
  highbd_partial_butterfly_sse2(io[1], cospi_30_64, cospi_2_64, &step2[8],
                                &step2[15]);
  highbd_partial_butterfly_neg_sse2(io[3], cospi_6_64, cospi_26_64, &step2[11],
                                    &step2[12]);

  // stage 3
  highbd_partial_butterfly_sse2(io[2], cospi_28_64, cospi_4_64, &step1[4],
                                &step1[7]);
  step1[8] = step2[8];
  step1[9] = step2[8];
  step1[10] =
      _mm_sub_epi32(_mm_setzero_si128(), step2[11]);  // step1[10] = -step1[10]
  step1[11] = step2[11];
  step1[12] = step2[12];
  step1[13] =
      _mm_sub_epi32(_mm_setzero_si128(), step2[12]);  // step1[13] = -step1[13]
  step1[14] = step2[15];
  step1[15] = step2[15];

  // stage 4
  abs_extend_64bit_sse2(io[0], temp, sign);
  step2[0] = multiplication_round_shift_sse2(temp, sign, cospi_16_64);
  step2[1] = step2[0];
  step2[2] = _mm_setzero_si128();
  step2[3] = _mm_setzero_si128();
  highbd_butterfly_sse2(step1[14], step1[9], cospi_24_64, cospi_8_64, &step2[9],
                        &step2[14]);
  highbd_butterfly_sse2(step1[10], step1[13], cospi_8_64, cospi_24_64,
                        &step2[13], &step2[10]);
  step2[5] = step1[4];
  step2[6] = step1[7];
  step2[8] = step1[8];
  step2[11] = step1[11];
  step2[12] = step1[12];
  step2[15] = step1[15];

  highbd_idct16_4col_stage5(step2, step1);
  highbd_idct16_4col_stage6(step1, step2);
  highbd_idct16_4col_stage7(step2, io);
}

void vpx_highbd_idct16x16_256_add_sse2(const tran_low_t *input, uint16_t *dest,
                                       int stride, int bd) {
  int i;
  __m128i out[16], *in;

  if (bd == 8) {
    __m128i l[16], r[16];

    in = l;
    for (i = 0; i < 2; i++) {
      highbd_load_pack_transpose_32bit_8x8(&input[0], 16, &in[0]);
      highbd_load_pack_transpose_32bit_8x8(&input[8], 16, &in[8]);
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
        highbd_write_buffer_8(dest + j * stride, out[j], bd);
      }
      dest += 8;
    }
  } else {
    __m128i all[4][16];

    for (i = 0; i < 4; i++) {
      in = all[i];
      highbd_load_transpose_32bit_8x4(&input[0], 16, &in[0]);
      highbd_load_transpose_32bit_8x4(&input[8], 16, &in[8]);
      highbd_idct16_4col(in);
      input += 4 * 16;
    }

    for (i = 0; i < 16; i += 4) {
      int j;
      transpose_32bit_4x4(all[0] + i, out + 0);
      transpose_32bit_4x4(all[1] + i, out + 4);
      transpose_32bit_4x4(all[2] + i, out + 8);
      transpose_32bit_4x4(all[3] + i, out + 12);
      highbd_idct16_4col(out);

      for (j = 0; j < 16; ++j) {
        highbd_write_buffer_4(dest + j * stride, out[j], bd);
      }
      dest += 4;
    }
  }
}

void vpx_highbd_idct16x16_38_add_sse2(const tran_low_t *input, uint16_t *dest,
                                      int stride, int bd) {
  int i;
  __m128i out[16];

  if (bd == 8) {
    __m128i in[16], temp[16];

    highbd_load_pack_transpose_32bit_8x8(input, 16, in);
    for (i = 8; i < 16; i++) {
      in[i] = _mm_setzero_si128();
    }
    idct16_8col(in, temp);

    for (i = 0; i < 16; i += 8) {
      int j;
      transpose_16bit_8x8(temp + i, in);
      idct16_8col(in, out);

      for (j = 0; j < 16; ++j) {
        highbd_write_buffer_8(dest + j * stride, out[j], bd);
      }
      dest += 8;
    }
  } else {
    __m128i all[2][16], *in;

    for (i = 0; i < 2; i++) {
      in = all[i];
      highbd_load_transpose_32bit_8x4(input, 16, in);
      highbd_idct16x16_38_4col(in);
      input += 4 * 16;
    }

    for (i = 0; i < 16; i += 4) {
      int j;
      transpose_32bit_4x4(all[0] + i, out + 0);
      transpose_32bit_4x4(all[1] + i, out + 4);
      highbd_idct16x16_38_4col(out);

      for (j = 0; j < 16; ++j) {
        highbd_write_buffer_4(dest + j * stride, out[j], bd);
      }
      dest += 4;
    }
  }
}

void vpx_highbd_idct16x16_10_add_sse2(const tran_low_t *input, uint16_t *dest,
                                      int stride, int bd) {
  int i;
  __m128i out[16];

  if (bd == 8) {
    __m128i in[16], l[16];

    in[0] = load_pack_8_32bit(input + 0 * 16);
    in[1] = load_pack_8_32bit(input + 1 * 16);
    in[2] = load_pack_8_32bit(input + 2 * 16);
    in[3] = load_pack_8_32bit(input + 3 * 16);

    idct16x16_10_pass1(in, l);

    for (i = 0; i < 16; i += 8) {
      int j;
      idct16x16_10_pass2(l + i, in);

      for (j = 0; j < 16; ++j) {
        highbd_write_buffer_8(dest + j * stride, in[j], bd);
      }
      dest += 8;
    }
  } else {
    __m128i all[2][16], *in;

    for (i = 0; i < 2; i++) {
      in = all[i];
      highbd_load_transpose_32bit_4x4(input, 16, in);
      highbd_idct16x16_10_4col(in);
      input += 4 * 16;
    }

    for (i = 0; i < 16; i += 4) {
      int j;
      transpose_32bit_4x4(&all[0][i], out);
      highbd_idct16x16_10_4col(out);

      for (j = 0; j < 16; ++j) {
        highbd_write_buffer_4(dest + j * stride, out[j], bd);
      }
      dest += 4;
    }
  }
}

void vpx_highbd_idct16x16_1_add_sse2(const tran_low_t *input, uint16_t *dest,
                                     int stride, int bd) {
  highbd_idct_1_add_kernel(input, dest, stride, bd, 16);
}
