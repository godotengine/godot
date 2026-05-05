/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp9_rtcd.h"
#include "vp9/common/vp9_idct.h"
#include "vpx_dsp/x86/highbd_inv_txfm_sse4.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/transpose_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

static INLINE void highbd_iadst_half_butterfly_sse4_1(const __m128i in,
                                                      const int c,
                                                      __m128i *const s) {
  const __m128i pair_c = pair_set_epi32(4 * c, 0);
  __m128i x[2];

  extend_64bit(in, x);
  s[0] = _mm_mul_epi32(pair_c, x[0]);
  s[1] = _mm_mul_epi32(pair_c, x[1]);
}

static INLINE void highbd_iadst_butterfly_sse4_1(const __m128i in0,
                                                 const __m128i in1,
                                                 const int c0, const int c1,
                                                 __m128i *const s0,
                                                 __m128i *const s1) {
  const __m128i pair_c0 = pair_set_epi32(4 * c0, 0);
  const __m128i pair_c1 = pair_set_epi32(4 * c1, 0);
  __m128i t00[2], t01[2], t10[2], t11[2];
  __m128i x0[2], x1[2];

  extend_64bit(in0, x0);
  extend_64bit(in1, x1);
  t00[0] = _mm_mul_epi32(pair_c0, x0[0]);
  t00[1] = _mm_mul_epi32(pair_c0, x0[1]);
  t01[0] = _mm_mul_epi32(pair_c0, x1[0]);
  t01[1] = _mm_mul_epi32(pair_c0, x1[1]);
  t10[0] = _mm_mul_epi32(pair_c1, x0[0]);
  t10[1] = _mm_mul_epi32(pair_c1, x0[1]);
  t11[0] = _mm_mul_epi32(pair_c1, x1[0]);
  t11[1] = _mm_mul_epi32(pair_c1, x1[1]);

  s0[0] = _mm_add_epi64(t00[0], t11[0]);
  s0[1] = _mm_add_epi64(t00[1], t11[1]);
  s1[0] = _mm_sub_epi64(t10[0], t01[0]);
  s1[1] = _mm_sub_epi64(t10[1], t01[1]);
}

static void highbd_iadst8_sse4_1(__m128i *const io) {
  __m128i s0[2], s1[2], s2[2], s3[2], s4[2], s5[2], s6[2], s7[2];
  __m128i x0[2], x1[2], x2[2], x3[2], x4[2], x5[2], x6[2], x7[2];

  transpose_32bit_4x4x2(io, io);

  // stage 1
  highbd_iadst_butterfly_sse4_1(io[7], io[0], cospi_2_64, cospi_30_64, s0, s1);
  highbd_iadst_butterfly_sse4_1(io[3], io[4], cospi_18_64, cospi_14_64, s4, s5);
  x0[0] = _mm_add_epi64(s0[0], s4[0]);
  x0[1] = _mm_add_epi64(s0[1], s4[1]);
  x1[0] = _mm_add_epi64(s1[0], s5[0]);
  x1[1] = _mm_add_epi64(s1[1], s5[1]);
  x4[0] = _mm_sub_epi64(s0[0], s4[0]);
  x4[1] = _mm_sub_epi64(s0[1], s4[1]);
  x5[0] = _mm_sub_epi64(s1[0], s5[0]);
  x5[1] = _mm_sub_epi64(s1[1], s5[1]);

  highbd_iadst_butterfly_sse4_1(io[5], io[2], cospi_10_64, cospi_22_64, s2, s3);
  highbd_iadst_butterfly_sse4_1(io[1], io[6], cospi_26_64, cospi_6_64, s6, s7);
  x2[0] = _mm_add_epi64(s2[0], s6[0]);
  x2[1] = _mm_add_epi64(s2[1], s6[1]);
  x3[0] = _mm_add_epi64(s3[0], s7[0]);
  x3[1] = _mm_add_epi64(s3[1], s7[1]);
  x6[0] = _mm_sub_epi64(s2[0], s6[0]);
  x6[1] = _mm_sub_epi64(s2[1], s6[1]);
  x7[0] = _mm_sub_epi64(s3[0], s7[0]);
  x7[1] = _mm_sub_epi64(s3[1], s7[1]);

  x0[0] = dct_const_round_shift_64bit(x0[0]);
  x0[1] = dct_const_round_shift_64bit(x0[1]);
  x1[0] = dct_const_round_shift_64bit(x1[0]);
  x1[1] = dct_const_round_shift_64bit(x1[1]);
  x2[0] = dct_const_round_shift_64bit(x2[0]);
  x2[1] = dct_const_round_shift_64bit(x2[1]);
  x3[0] = dct_const_round_shift_64bit(x3[0]);
  x3[1] = dct_const_round_shift_64bit(x3[1]);
  x4[0] = dct_const_round_shift_64bit(x4[0]);
  x4[1] = dct_const_round_shift_64bit(x4[1]);
  x5[0] = dct_const_round_shift_64bit(x5[0]);
  x5[1] = dct_const_round_shift_64bit(x5[1]);
  x6[0] = dct_const_round_shift_64bit(x6[0]);
  x6[1] = dct_const_round_shift_64bit(x6[1]);
  x7[0] = dct_const_round_shift_64bit(x7[0]);
  x7[1] = dct_const_round_shift_64bit(x7[1]);
  s0[0] = pack_4(x0[0], x0[1]);  // s0 = x0;
  s1[0] = pack_4(x1[0], x1[1]);  // s1 = x1;
  s2[0] = pack_4(x2[0], x2[1]);  // s2 = x2;
  s3[0] = pack_4(x3[0], x3[1]);  // s3 = x3;
  x4[0] = pack_4(x4[0], x4[1]);
  x5[0] = pack_4(x5[0], x5[1]);
  x6[0] = pack_4(x6[0], x6[1]);
  x7[0] = pack_4(x7[0], x7[1]);

  // stage 2
  x0[0] = _mm_add_epi32(s0[0], s2[0]);
  x1[0] = _mm_add_epi32(s1[0], s3[0]);
  x2[0] = _mm_sub_epi32(s0[0], s2[0]);
  x3[0] = _mm_sub_epi32(s1[0], s3[0]);

  highbd_iadst_butterfly_sse4_1(x4[0], x5[0], cospi_8_64, cospi_24_64, s4, s5);
  highbd_iadst_butterfly_sse4_1(x7[0], x6[0], cospi_24_64, cospi_8_64, s7, s6);

  x4[0] = _mm_add_epi64(s4[0], s6[0]);
  x4[1] = _mm_add_epi64(s4[1], s6[1]);
  x5[0] = _mm_add_epi64(s5[0], s7[0]);
  x5[1] = _mm_add_epi64(s5[1], s7[1]);
  x6[0] = _mm_sub_epi64(s4[0], s6[0]);
  x6[1] = _mm_sub_epi64(s4[1], s6[1]);
  x7[0] = _mm_sub_epi64(s5[0], s7[0]);
  x7[1] = _mm_sub_epi64(s5[1], s7[1]);
  x4[0] = dct_const_round_shift_64bit(x4[0]);
  x4[1] = dct_const_round_shift_64bit(x4[1]);
  x5[0] = dct_const_round_shift_64bit(x5[0]);
  x5[1] = dct_const_round_shift_64bit(x5[1]);
  x6[0] = dct_const_round_shift_64bit(x6[0]);
  x6[1] = dct_const_round_shift_64bit(x6[1]);
  x7[0] = dct_const_round_shift_64bit(x7[0]);
  x7[1] = dct_const_round_shift_64bit(x7[1]);
  x4[0] = pack_4(x4[0], x4[1]);
  x5[0] = pack_4(x5[0], x5[1]);
  x6[0] = pack_4(x6[0], x6[1]);
  x7[0] = pack_4(x7[0], x7[1]);

  // stage 3
  s2[0] = _mm_add_epi32(x2[0], x3[0]);
  s3[0] = _mm_sub_epi32(x2[0], x3[0]);
  s6[0] = _mm_add_epi32(x6[0], x7[0]);
  s7[0] = _mm_sub_epi32(x6[0], x7[0]);
  highbd_iadst_half_butterfly_sse4_1(s2[0], cospi_16_64, s2);
  highbd_iadst_half_butterfly_sse4_1(s3[0], cospi_16_64, s3);
  highbd_iadst_half_butterfly_sse4_1(s6[0], cospi_16_64, s6);
  highbd_iadst_half_butterfly_sse4_1(s7[0], cospi_16_64, s7);

  x2[0] = dct_const_round_shift_64bit(s2[0]);
  x2[1] = dct_const_round_shift_64bit(s2[1]);
  x3[0] = dct_const_round_shift_64bit(s3[0]);
  x3[1] = dct_const_round_shift_64bit(s3[1]);
  x6[0] = dct_const_round_shift_64bit(s6[0]);
  x6[1] = dct_const_round_shift_64bit(s6[1]);
  x7[0] = dct_const_round_shift_64bit(s7[0]);
  x7[1] = dct_const_round_shift_64bit(s7[1]);
  x2[0] = pack_4(x2[0], x2[1]);
  x3[0] = pack_4(x3[0], x3[1]);
  x6[0] = pack_4(x6[0], x6[1]);
  x7[0] = pack_4(x7[0], x7[1]);

  io[0] = x0[0];
  io[1] = _mm_sub_epi32(_mm_setzero_si128(), x4[0]);
  io[2] = x6[0];
  io[3] = _mm_sub_epi32(_mm_setzero_si128(), x2[0]);
  io[4] = x3[0];
  io[5] = _mm_sub_epi32(_mm_setzero_si128(), x7[0]);
  io[6] = x5[0];
  io[7] = _mm_sub_epi32(_mm_setzero_si128(), x1[0]);
}

void vp9_highbd_iht8x8_64_add_sse4_1(const tran_low_t *input, uint16_t *dest,
                                     int stride, int tx_type, int bd) {
  __m128i io[16];

  io[0] = _mm_load_si128((const __m128i *)(input + 0 * 8 + 0));
  io[4] = _mm_load_si128((const __m128i *)(input + 0 * 8 + 4));
  io[1] = _mm_load_si128((const __m128i *)(input + 1 * 8 + 0));
  io[5] = _mm_load_si128((const __m128i *)(input + 1 * 8 + 4));
  io[2] = _mm_load_si128((const __m128i *)(input + 2 * 8 + 0));
  io[6] = _mm_load_si128((const __m128i *)(input + 2 * 8 + 4));
  io[3] = _mm_load_si128((const __m128i *)(input + 3 * 8 + 0));
  io[7] = _mm_load_si128((const __m128i *)(input + 3 * 8 + 4));
  io[8] = _mm_load_si128((const __m128i *)(input + 4 * 8 + 0));
  io[12] = _mm_load_si128((const __m128i *)(input + 4 * 8 + 4));
  io[9] = _mm_load_si128((const __m128i *)(input + 5 * 8 + 0));
  io[13] = _mm_load_si128((const __m128i *)(input + 5 * 8 + 4));
  io[10] = _mm_load_si128((const __m128i *)(input + 6 * 8 + 0));
  io[14] = _mm_load_si128((const __m128i *)(input + 6 * 8 + 4));
  io[11] = _mm_load_si128((const __m128i *)(input + 7 * 8 + 0));
  io[15] = _mm_load_si128((const __m128i *)(input + 7 * 8 + 4));

  if (bd == 8) {
    __m128i io_short[8];

    io_short[0] = _mm_packs_epi32(io[0], io[4]);
    io_short[1] = _mm_packs_epi32(io[1], io[5]);
    io_short[2] = _mm_packs_epi32(io[2], io[6]);
    io_short[3] = _mm_packs_epi32(io[3], io[7]);
    io_short[4] = _mm_packs_epi32(io[8], io[12]);
    io_short[5] = _mm_packs_epi32(io[9], io[13]);
    io_short[6] = _mm_packs_epi32(io[10], io[14]);
    io_short[7] = _mm_packs_epi32(io[11], io[15]);

    if (tx_type == DCT_DCT || tx_type == ADST_DCT) {
      vpx_idct8_sse2(io_short);
    } else {
      iadst8_sse2(io_short);
    }
    if (tx_type == DCT_DCT || tx_type == DCT_ADST) {
      vpx_idct8_sse2(io_short);
    } else {
      iadst8_sse2(io_short);
    }
    round_shift_8x8(io_short, io);
  } else {
    __m128i temp[4];

    if (tx_type == DCT_DCT || tx_type == ADST_DCT) {
      vpx_highbd_idct8x8_half1d_sse4_1(io);
      vpx_highbd_idct8x8_half1d_sse4_1(&io[8]);
    } else {
      highbd_iadst8_sse4_1(io);
      highbd_iadst8_sse4_1(&io[8]);
    }

    temp[0] = io[4];
    temp[1] = io[5];
    temp[2] = io[6];
    temp[3] = io[7];
    io[4] = io[8];
    io[5] = io[9];
    io[6] = io[10];
    io[7] = io[11];

    if (tx_type == DCT_DCT || tx_type == DCT_ADST) {
      vpx_highbd_idct8x8_half1d_sse4_1(io);
      io[8] = temp[0];
      io[9] = temp[1];
      io[10] = temp[2];
      io[11] = temp[3];
      vpx_highbd_idct8x8_half1d_sse4_1(&io[8]);
    } else {
      highbd_iadst8_sse4_1(io);
      io[8] = temp[0];
      io[9] = temp[1];
      io[10] = temp[2];
      io[11] = temp[3];
      highbd_iadst8_sse4_1(&io[8]);
    }
    highbd_idct8x8_final_round(io);
  }
  recon_and_store_8x8(io, dest, stride, bd);
}
