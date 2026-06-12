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

static INLINE void highbd_iadst4_sse4_1(__m128i *const io) {
  const __m128i pair_c1 = pair_set_epi32(4 * sinpi_1_9, 0);
  const __m128i pair_c2 = pair_set_epi32(4 * sinpi_2_9, 0);
  const __m128i pair_c3 = pair_set_epi32(4 * sinpi_3_9, 0);
  const __m128i pair_c4 = pair_set_epi32(4 * sinpi_4_9, 0);
  __m128i s0[2], s1[2], s2[2], s3[2], s4[2], s5[2], s6[2], t0[2], t1[2], t2[2];
  __m128i temp[2];

  transpose_32bit_4x4(io, io);

  extend_64bit(io[0], temp);
  s0[0] = _mm_mul_epi32(pair_c1, temp[0]);
  s0[1] = _mm_mul_epi32(pair_c1, temp[1]);
  s1[0] = _mm_mul_epi32(pair_c2, temp[0]);
  s1[1] = _mm_mul_epi32(pair_c2, temp[1]);

  extend_64bit(io[1], temp);
  s2[0] = _mm_mul_epi32(pair_c3, temp[0]);
  s2[1] = _mm_mul_epi32(pair_c3, temp[1]);

  extend_64bit(io[2], temp);
  s3[0] = _mm_mul_epi32(pair_c4, temp[0]);
  s3[1] = _mm_mul_epi32(pair_c4, temp[1]);
  s4[0] = _mm_mul_epi32(pair_c1, temp[0]);
  s4[1] = _mm_mul_epi32(pair_c1, temp[1]);

  extend_64bit(io[3], temp);
  s5[0] = _mm_mul_epi32(pair_c2, temp[0]);
  s5[1] = _mm_mul_epi32(pair_c2, temp[1]);
  s6[0] = _mm_mul_epi32(pair_c4, temp[0]);
  s6[1] = _mm_mul_epi32(pair_c4, temp[1]);

  t0[0] = _mm_add_epi64(s0[0], s3[0]);
  t0[1] = _mm_add_epi64(s0[1], s3[1]);
  t0[0] = _mm_add_epi64(t0[0], s5[0]);
  t0[1] = _mm_add_epi64(t0[1], s5[1]);
  t1[0] = _mm_sub_epi64(s1[0], s4[0]);
  t1[1] = _mm_sub_epi64(s1[1], s4[1]);
  t1[0] = _mm_sub_epi64(t1[0], s6[0]);
  t1[1] = _mm_sub_epi64(t1[1], s6[1]);
  temp[0] = _mm_sub_epi32(io[0], io[2]);
  temp[0] = _mm_add_epi32(temp[0], io[3]);
  extend_64bit(temp[0], temp);
  t2[0] = _mm_mul_epi32(pair_c3, temp[0]);
  t2[1] = _mm_mul_epi32(pair_c3, temp[1]);

  s0[0] = _mm_add_epi64(t0[0], s2[0]);
  s0[1] = _mm_add_epi64(t0[1], s2[1]);
  s1[0] = _mm_add_epi64(t1[0], s2[0]);
  s1[1] = _mm_add_epi64(t1[1], s2[1]);
  s3[0] = _mm_add_epi64(t0[0], t1[0]);
  s3[1] = _mm_add_epi64(t0[1], t1[1]);
  s3[0] = _mm_sub_epi64(s3[0], s2[0]);
  s3[1] = _mm_sub_epi64(s3[1], s2[1]);

  s0[0] = dct_const_round_shift_64bit(s0[0]);
  s0[1] = dct_const_round_shift_64bit(s0[1]);
  s1[0] = dct_const_round_shift_64bit(s1[0]);
  s1[1] = dct_const_round_shift_64bit(s1[1]);
  s2[0] = dct_const_round_shift_64bit(t2[0]);
  s2[1] = dct_const_round_shift_64bit(t2[1]);
  s3[0] = dct_const_round_shift_64bit(s3[0]);
  s3[1] = dct_const_round_shift_64bit(s3[1]);
  io[0] = pack_4(s0[0], s0[1]);
  io[1] = pack_4(s1[0], s1[1]);
  io[2] = pack_4(s2[0], s2[1]);
  io[3] = pack_4(s3[0], s3[1]);
}

void vp9_highbd_iht4x4_16_add_sse4_1(const tran_low_t *input, uint16_t *dest,
                                     int stride, int tx_type, int bd) {
  __m128i io[4];

  io[0] = _mm_load_si128((const __m128i *)(input + 0));
  io[1] = _mm_load_si128((const __m128i *)(input + 4));
  io[2] = _mm_load_si128((const __m128i *)(input + 8));
  io[3] = _mm_load_si128((const __m128i *)(input + 12));

  if (bd == 8) {
    __m128i io_short[2];

    io_short[0] = _mm_packs_epi32(io[0], io[1]);
    io_short[1] = _mm_packs_epi32(io[2], io[3]);
    if (tx_type == DCT_DCT || tx_type == ADST_DCT) {
      idct4_sse2(io_short);
    } else {
      iadst4_sse2(io_short);
    }
    if (tx_type == DCT_DCT || tx_type == DCT_ADST) {
      idct4_sse2(io_short);
    } else {
      iadst4_sse2(io_short);
    }
    io_short[0] = _mm_add_epi16(io_short[0], _mm_set1_epi16(8));
    io_short[1] = _mm_add_epi16(io_short[1], _mm_set1_epi16(8));
    io[0] = _mm_srai_epi16(io_short[0], 4);
    io[1] = _mm_srai_epi16(io_short[1], 4);
  } else {
    if (tx_type == DCT_DCT || tx_type == ADST_DCT) {
      highbd_idct4_sse4_1(io);
    } else {
      highbd_iadst4_sse4_1(io);
    }
    if (tx_type == DCT_DCT || tx_type == DCT_ADST) {
      highbd_idct4_sse4_1(io);
    } else {
      highbd_iadst4_sse4_1(io);
    }
    io[0] = wraplow_16bit_shift4(io[0], io[1], _mm_set1_epi32(8));
    io[1] = wraplow_16bit_shift4(io[2], io[3], _mm_set1_epi32(8));
  }

  recon_and_store_4x4(io, dest, stride, bd);
}
