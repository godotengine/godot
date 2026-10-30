/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <smmintrin.h>  // SSE4.1

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/x86/highbd_inv_txfm_sse2.h"
#include "vpx_dsp/x86/highbd_inv_txfm_sse4.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/inv_txfm_ssse3.h"
#include "vpx_dsp/x86/transpose_sse2.h"

void vpx_highbd_idct8x8_half1d_sse4_1(__m128i *const io) {
  __m128i step1[8], step2[8];

  transpose_32bit_4x4x2(io, io);

  // stage 1
  step1[0] = io[0];
  step1[2] = io[4];
  step1[1] = io[2];
  step1[3] = io[6];
  highbd_butterfly_sse4_1(io[1], io[7], cospi_28_64, cospi_4_64, &step1[4],
                          &step1[7]);
  highbd_butterfly_sse4_1(io[5], io[3], cospi_12_64, cospi_20_64, &step1[5],
                          &step1[6]);

  // stage 2
  highbd_butterfly_cospi16_sse4_1(step1[0], step1[2], &step2[0], &step2[1]);
  highbd_butterfly_sse4_1(step1[1], step1[3], cospi_24_64, cospi_8_64,
                          &step2[2], &step2[3]);
  step2[4] = _mm_add_epi32(step1[4], step1[5]);
  step2[5] = _mm_sub_epi32(step1[4], step1[5]);
  step2[6] = _mm_sub_epi32(step1[7], step1[6]);
  step2[7] = _mm_add_epi32(step1[7], step1[6]);

  // stage 3
  step1[0] = _mm_add_epi32(step2[0], step2[3]);
  step1[1] = _mm_add_epi32(step2[1], step2[2]);
  step1[2] = _mm_sub_epi32(step2[1], step2[2]);
  step1[3] = _mm_sub_epi32(step2[0], step2[3]);
  step1[4] = step2[4];
  highbd_butterfly_cospi16_sse4_1(step2[6], step2[5], &step1[6], &step1[5]);
  step1[7] = step2[7];

  // stage 4
  highbd_idct8_stage4(step1, io);
}

static void highbd_idct8x8_12_half1d(__m128i *const io) {
  __m128i temp1[2], step1[8], step2[8];

  transpose_32bit_4x4(io, io);

  // stage 1
  step1[0] = io[0];
  step1[1] = io[2];
  extend_64bit(io[1], temp1);
  step1[4] = multiplication_round_shift_sse4_1(temp1, cospi_28_64);
  step1[7] = multiplication_round_shift_sse4_1(temp1, cospi_4_64);
  extend_64bit(io[3], temp1);
  step1[5] = multiplication_round_shift_sse4_1(temp1, -cospi_20_64);
  step1[6] = multiplication_round_shift_sse4_1(temp1, cospi_12_64);

  // stage 2
  extend_64bit(step1[0], temp1);
  step2[0] = multiplication_round_shift_sse4_1(temp1, cospi_16_64);
  extend_64bit(step1[1], temp1);
  step2[2] = multiplication_round_shift_sse4_1(temp1, cospi_24_64);
  step2[3] = multiplication_round_shift_sse4_1(temp1, cospi_8_64);
  step2[4] = _mm_add_epi32(step1[4], step1[5]);
  step2[5] = _mm_sub_epi32(step1[4], step1[5]);
  step2[6] = _mm_sub_epi32(step1[7], step1[6]);
  step2[7] = _mm_add_epi32(step1[7], step1[6]);

  // stage 3
  step1[0] = _mm_add_epi32(step2[0], step2[3]);
  step1[1] = _mm_add_epi32(step2[0], step2[2]);
  step1[2] = _mm_sub_epi32(step2[0], step2[2]);
  step1[3] = _mm_sub_epi32(step2[0], step2[3]);
  step1[4] = step2[4];
  highbd_butterfly_cospi16_sse4_1(step2[6], step2[5], &step1[6], &step1[5]);
  step1[7] = step2[7];

  // stage 4
  highbd_idct8_stage4(step1, io);
}

void vpx_highbd_idct8x8_64_add_sse4_1(const tran_low_t *input, uint16_t *dest,
                                      int stride, int bd) {
  __m128i io[16];

  io[0] = _mm_load_si128((const __m128i *)(input + 0 * 8 + 0));
  io[4] = _mm_load_si128((const __m128i *)(input + 0 * 8 + 4));
  io[1] = _mm_load_si128((const __m128i *)(input + 1 * 8 + 0));
  io[5] = _mm_load_si128((const __m128i *)(input + 1 * 8 + 4));
  io[2] = _mm_load_si128((const __m128i *)(input + 2 * 8 + 0));
  io[6] = _mm_load_si128((const __m128i *)(input + 2 * 8 + 4));
  io[3] = _mm_load_si128((const __m128i *)(input + 3 * 8 + 0));
  io[7] = _mm_load_si128((const __m128i *)(input + 3 * 8 + 4));

  if (bd == 8) {
    __m128i io_short[8];

    io_short[0] = _mm_packs_epi32(io[0], io[4]);
    io_short[1] = _mm_packs_epi32(io[1], io[5]);
    io_short[2] = _mm_packs_epi32(io[2], io[6]);
    io_short[3] = _mm_packs_epi32(io[3], io[7]);
    io[8] = _mm_load_si128((const __m128i *)(input + 4 * 8 + 0));
    io[12] = _mm_load_si128((const __m128i *)(input + 4 * 8 + 4));
    io[9] = _mm_load_si128((const __m128i *)(input + 5 * 8 + 0));
    io[13] = _mm_load_si128((const __m128i *)(input + 5 * 8 + 4));
    io[10] = _mm_load_si128((const __m128i *)(input + 6 * 8 + 0));
    io[14] = _mm_load_si128((const __m128i *)(input + 6 * 8 + 4));
    io[11] = _mm_load_si128((const __m128i *)(input + 7 * 8 + 0));
    io[15] = _mm_load_si128((const __m128i *)(input + 7 * 8 + 4));
    io_short[4] = _mm_packs_epi32(io[8], io[12]);
    io_short[5] = _mm_packs_epi32(io[9], io[13]);
    io_short[6] = _mm_packs_epi32(io[10], io[14]);
    io_short[7] = _mm_packs_epi32(io[11], io[15]);

    vpx_idct8_sse2(io_short);
    vpx_idct8_sse2(io_short);
    round_shift_8x8(io_short, io);
  } else {
    __m128i temp[4];

    vpx_highbd_idct8x8_half1d_sse4_1(io);

    io[8] = _mm_load_si128((const __m128i *)(input + 4 * 8 + 0));
    io[12] = _mm_load_si128((const __m128i *)(input + 4 * 8 + 4));
    io[9] = _mm_load_si128((const __m128i *)(input + 5 * 8 + 0));
    io[13] = _mm_load_si128((const __m128i *)(input + 5 * 8 + 4));
    io[10] = _mm_load_si128((const __m128i *)(input + 6 * 8 + 0));
    io[14] = _mm_load_si128((const __m128i *)(input + 6 * 8 + 4));
    io[11] = _mm_load_si128((const __m128i *)(input + 7 * 8 + 0));
    io[15] = _mm_load_si128((const __m128i *)(input + 7 * 8 + 4));
    vpx_highbd_idct8x8_half1d_sse4_1(&io[8]);

    temp[0] = io[4];
    temp[1] = io[5];
    temp[2] = io[6];
    temp[3] = io[7];
    io[4] = io[8];
    io[5] = io[9];
    io[6] = io[10];
    io[7] = io[11];
    vpx_highbd_idct8x8_half1d_sse4_1(io);

    io[8] = temp[0];
    io[9] = temp[1];
    io[10] = temp[2];
    io[11] = temp[3];
    vpx_highbd_idct8x8_half1d_sse4_1(&io[8]);

    highbd_idct8x8_final_round(io);
  }

  recon_and_store_8x8(io, dest, stride, bd);
}

void vpx_highbd_idct8x8_12_add_sse4_1(const tran_low_t *input, uint16_t *dest,
                                      int stride, int bd) {
  const __m128i zero = _mm_setzero_si128();
  __m128i io[16];

  io[0] = _mm_load_si128((const __m128i *)(input + 0 * 8 + 0));
  io[1] = _mm_load_si128((const __m128i *)(input + 1 * 8 + 0));
  io[2] = _mm_load_si128((const __m128i *)(input + 2 * 8 + 0));
  io[3] = _mm_load_si128((const __m128i *)(input + 3 * 8 + 0));

  if (bd == 8) {
    __m128i io_short[8];

    io_short[0] = _mm_packs_epi32(io[0], zero);
    io_short[1] = _mm_packs_epi32(io[1], zero);
    io_short[2] = _mm_packs_epi32(io[2], zero);
    io_short[3] = _mm_packs_epi32(io[3], zero);

    idct8x8_12_add_kernel_ssse3(io_short);
    round_shift_8x8(io_short, io);
  } else {
    __m128i temp[4];

    highbd_idct8x8_12_half1d(io);

    temp[0] = io[4];
    temp[1] = io[5];
    temp[2] = io[6];
    temp[3] = io[7];
    highbd_idct8x8_12_half1d(io);

    io[8] = temp[0];
    io[9] = temp[1];
    io[10] = temp[2];
    io[11] = temp[3];
    highbd_idct8x8_12_half1d(&io[8]);

    highbd_idct8x8_final_round(io);
  }

  recon_and_store_8x8(io, dest, stride, bd);
}
