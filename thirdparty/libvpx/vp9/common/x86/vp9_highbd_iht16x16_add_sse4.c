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

static void highbd_iadst16_4col_sse4_1(__m128i *const io /*io[16]*/) {
  __m128i s0[2], s1[2], s2[2], s3[2], s4[2], s5[2], s6[2], s7[2], s8[2], s9[2],
      s10[2], s11[2], s12[2], s13[2], s14[2], s15[2];
  __m128i x0[2], x1[2], x2[2], x3[2], x4[2], x5[2], x6[2], x7[2], x8[2], x9[2],
      x10[2], x11[2], x12[2], x13[2], x14[2], x15[2];

  // stage 1
  highbd_iadst_butterfly_sse4_1(io[15], io[0], cospi_1_64, cospi_31_64, s0, s1);
  highbd_iadst_butterfly_sse4_1(io[13], io[2], cospi_5_64, cospi_27_64, s2, s3);
  highbd_iadst_butterfly_sse4_1(io[11], io[4], cospi_9_64, cospi_23_64, s4, s5);
  highbd_iadst_butterfly_sse4_1(io[9], io[6], cospi_13_64, cospi_19_64, s6, s7);
  highbd_iadst_butterfly_sse4_1(io[7], io[8], cospi_17_64, cospi_15_64, s8, s9);
  highbd_iadst_butterfly_sse4_1(io[5], io[10], cospi_21_64, cospi_11_64, s10,
                                s11);
  highbd_iadst_butterfly_sse4_1(io[3], io[12], cospi_25_64, cospi_7_64, s12,
                                s13);
  highbd_iadst_butterfly_sse4_1(io[1], io[14], cospi_29_64, cospi_3_64, s14,
                                s15);

  x0[0] = _mm_add_epi64(s0[0], s8[0]);
  x0[1] = _mm_add_epi64(s0[1], s8[1]);
  x1[0] = _mm_add_epi64(s1[0], s9[0]);
  x1[1] = _mm_add_epi64(s1[1], s9[1]);
  x2[0] = _mm_add_epi64(s2[0], s10[0]);
  x2[1] = _mm_add_epi64(s2[1], s10[1]);
  x3[0] = _mm_add_epi64(s3[0], s11[0]);
  x3[1] = _mm_add_epi64(s3[1], s11[1]);
  x4[0] = _mm_add_epi64(s4[0], s12[0]);
  x4[1] = _mm_add_epi64(s4[1], s12[1]);
  x5[0] = _mm_add_epi64(s5[0], s13[0]);
  x5[1] = _mm_add_epi64(s5[1], s13[1]);
  x6[0] = _mm_add_epi64(s6[0], s14[0]);
  x6[1] = _mm_add_epi64(s6[1], s14[1]);
  x7[0] = _mm_add_epi64(s7[0], s15[0]);
  x7[1] = _mm_add_epi64(s7[1], s15[1]);
  x8[0] = _mm_sub_epi64(s0[0], s8[0]);
  x8[1] = _mm_sub_epi64(s0[1], s8[1]);
  x9[0] = _mm_sub_epi64(s1[0], s9[0]);
  x9[1] = _mm_sub_epi64(s1[1], s9[1]);
  x10[0] = _mm_sub_epi64(s2[0], s10[0]);
  x10[1] = _mm_sub_epi64(s2[1], s10[1]);
  x11[0] = _mm_sub_epi64(s3[0], s11[0]);
  x11[1] = _mm_sub_epi64(s3[1], s11[1]);
  x12[0] = _mm_sub_epi64(s4[0], s12[0]);
  x12[1] = _mm_sub_epi64(s4[1], s12[1]);
  x13[0] = _mm_sub_epi64(s5[0], s13[0]);
  x13[1] = _mm_sub_epi64(s5[1], s13[1]);
  x14[0] = _mm_sub_epi64(s6[0], s14[0]);
  x14[1] = _mm_sub_epi64(s6[1], s14[1]);
  x15[0] = _mm_sub_epi64(s7[0], s15[0]);
  x15[1] = _mm_sub_epi64(s7[1], s15[1]);

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
  x8[0] = dct_const_round_shift_64bit(x8[0]);
  x8[1] = dct_const_round_shift_64bit(x8[1]);
  x9[0] = dct_const_round_shift_64bit(x9[0]);
  x9[1] = dct_const_round_shift_64bit(x9[1]);
  x10[0] = dct_const_round_shift_64bit(x10[0]);
  x10[1] = dct_const_round_shift_64bit(x10[1]);
  x11[0] = dct_const_round_shift_64bit(x11[0]);
  x11[1] = dct_const_round_shift_64bit(x11[1]);
  x12[0] = dct_const_round_shift_64bit(x12[0]);
  x12[1] = dct_const_round_shift_64bit(x12[1]);
  x13[0] = dct_const_round_shift_64bit(x13[0]);
  x13[1] = dct_const_round_shift_64bit(x13[1]);
  x14[0] = dct_const_round_shift_64bit(x14[0]);
  x14[1] = dct_const_round_shift_64bit(x14[1]);
  x15[0] = dct_const_round_shift_64bit(x15[0]);
  x15[1] = dct_const_round_shift_64bit(x15[1]);
  x0[0] = pack_4(x0[0], x0[1]);
  x1[0] = pack_4(x1[0], x1[1]);
  x2[0] = pack_4(x2[0], x2[1]);
  x3[0] = pack_4(x3[0], x3[1]);
  x4[0] = pack_4(x4[0], x4[1]);
  x5[0] = pack_4(x5[0], x5[1]);
  x6[0] = pack_4(x6[0], x6[1]);
  x7[0] = pack_4(x7[0], x7[1]);
  x8[0] = pack_4(x8[0], x8[1]);
  x9[0] = pack_4(x9[0], x9[1]);
  x10[0] = pack_4(x10[0], x10[1]);
  x11[0] = pack_4(x11[0], x11[1]);
  x12[0] = pack_4(x12[0], x12[1]);
  x13[0] = pack_4(x13[0], x13[1]);
  x14[0] = pack_4(x14[0], x14[1]);
  x15[0] = pack_4(x15[0], x15[1]);

  // stage 2
  s0[0] = x0[0];
  s1[0] = x1[0];
  s2[0] = x2[0];
  s3[0] = x3[0];
  s4[0] = x4[0];
  s5[0] = x5[0];
  s6[0] = x6[0];
  s7[0] = x7[0];
  x0[0] = _mm_add_epi32(s0[0], s4[0]);
  x1[0] = _mm_add_epi32(s1[0], s5[0]);
  x2[0] = _mm_add_epi32(s2[0], s6[0]);
  x3[0] = _mm_add_epi32(s3[0], s7[0]);
  x4[0] = _mm_sub_epi32(s0[0], s4[0]);
  x5[0] = _mm_sub_epi32(s1[0], s5[0]);
  x6[0] = _mm_sub_epi32(s2[0], s6[0]);
  x7[0] = _mm_sub_epi32(s3[0], s7[0]);

  highbd_iadst_butterfly_sse4_1(x8[0], x9[0], cospi_4_64, cospi_28_64, s8, s9);
  highbd_iadst_butterfly_sse4_1(x10[0], x11[0], cospi_20_64, cospi_12_64, s10,
                                s11);
  highbd_iadst_butterfly_sse4_1(x13[0], x12[0], cospi_28_64, cospi_4_64, s13,
                                s12);
  highbd_iadst_butterfly_sse4_1(x15[0], x14[0], cospi_12_64, cospi_20_64, s15,
                                s14);

  x8[0] = _mm_add_epi64(s8[0], s12[0]);
  x8[1] = _mm_add_epi64(s8[1], s12[1]);
  x9[0] = _mm_add_epi64(s9[0], s13[0]);
  x9[1] = _mm_add_epi64(s9[1], s13[1]);
  x10[0] = _mm_add_epi64(s10[0], s14[0]);
  x10[1] = _mm_add_epi64(s10[1], s14[1]);
  x11[0] = _mm_add_epi64(s11[0], s15[0]);
  x11[1] = _mm_add_epi64(s11[1], s15[1]);
  x12[0] = _mm_sub_epi64(s8[0], s12[0]);
  x12[1] = _mm_sub_epi64(s8[1], s12[1]);
  x13[0] = _mm_sub_epi64(s9[0], s13[0]);
  x13[1] = _mm_sub_epi64(s9[1], s13[1]);
  x14[0] = _mm_sub_epi64(s10[0], s14[0]);
  x14[1] = _mm_sub_epi64(s10[1], s14[1]);
  x15[0] = _mm_sub_epi64(s11[0], s15[0]);
  x15[1] = _mm_sub_epi64(s11[1], s15[1]);
  x8[0] = dct_const_round_shift_64bit(x8[0]);
  x8[1] = dct_const_round_shift_64bit(x8[1]);
  x9[0] = dct_const_round_shift_64bit(x9[0]);
  x9[1] = dct_const_round_shift_64bit(x9[1]);
  x10[0] = dct_const_round_shift_64bit(x10[0]);
  x10[1] = dct_const_round_shift_64bit(x10[1]);
  x11[0] = dct_const_round_shift_64bit(x11[0]);
  x11[1] = dct_const_round_shift_64bit(x11[1]);
  x12[0] = dct_const_round_shift_64bit(x12[0]);
  x12[1] = dct_const_round_shift_64bit(x12[1]);
  x13[0] = dct_const_round_shift_64bit(x13[0]);
  x13[1] = dct_const_round_shift_64bit(x13[1]);
  x14[0] = dct_const_round_shift_64bit(x14[0]);
  x14[1] = dct_const_round_shift_64bit(x14[1]);
  x15[0] = dct_const_round_shift_64bit(x15[0]);
  x15[1] = dct_const_round_shift_64bit(x15[1]);
  x8[0] = pack_4(x8[0], x8[1]);
  x9[0] = pack_4(x9[0], x9[1]);
  x10[0] = pack_4(x10[0], x10[1]);
  x11[0] = pack_4(x11[0], x11[1]);
  x12[0] = pack_4(x12[0], x12[1]);
  x13[0] = pack_4(x13[0], x13[1]);
  x14[0] = pack_4(x14[0], x14[1]);
  x15[0] = pack_4(x15[0], x15[1]);

  // stage 3
  s0[0] = x0[0];
  s1[0] = x1[0];
  s2[0] = x2[0];
  s3[0] = x3[0];
  highbd_iadst_butterfly_sse4_1(x4[0], x5[0], cospi_8_64, cospi_24_64, s4, s5);
  highbd_iadst_butterfly_sse4_1(x7[0], x6[0], cospi_24_64, cospi_8_64, s7, s6);
  s8[0] = x8[0];
  s9[0] = x9[0];
  s10[0] = x10[0];
  s11[0] = x11[0];
  highbd_iadst_butterfly_sse4_1(x12[0], x13[0], cospi_8_64, cospi_24_64, s12,
                                s13);
  highbd_iadst_butterfly_sse4_1(x15[0], x14[0], cospi_24_64, cospi_8_64, s15,
                                s14);

  x0[0] = _mm_add_epi32(s0[0], s2[0]);
  x1[0] = _mm_add_epi32(s1[0], s3[0]);
  x2[0] = _mm_sub_epi32(s0[0], s2[0]);
  x3[0] = _mm_sub_epi32(s1[0], s3[0]);
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
  x8[0] = _mm_add_epi32(s8[0], s10[0]);
  x9[0] = _mm_add_epi32(s9[0], s11[0]);
  x10[0] = _mm_sub_epi32(s8[0], s10[0]);
  x11[0] = _mm_sub_epi32(s9[0], s11[0]);
  x12[0] = _mm_add_epi64(s12[0], s14[0]);
  x12[1] = _mm_add_epi64(s12[1], s14[1]);
  x13[0] = _mm_add_epi64(s13[0], s15[0]);
  x13[1] = _mm_add_epi64(s13[1], s15[1]);
  x14[0] = _mm_sub_epi64(s12[0], s14[0]);
  x14[1] = _mm_sub_epi64(s12[1], s14[1]);
  x15[0] = _mm_sub_epi64(s13[0], s15[0]);
  x15[1] = _mm_sub_epi64(s13[1], s15[1]);
  x12[0] = dct_const_round_shift_64bit(x12[0]);
  x12[1] = dct_const_round_shift_64bit(x12[1]);
  x13[0] = dct_const_round_shift_64bit(x13[0]);
  x13[1] = dct_const_round_shift_64bit(x13[1]);
  x14[0] = dct_const_round_shift_64bit(x14[0]);
  x14[1] = dct_const_round_shift_64bit(x14[1]);
  x15[0] = dct_const_round_shift_64bit(x15[0]);
  x15[1] = dct_const_round_shift_64bit(x15[1]);
  x12[0] = pack_4(x12[0], x12[1]);
  x13[0] = pack_4(x13[0], x13[1]);
  x14[0] = pack_4(x14[0], x14[1]);
  x15[0] = pack_4(x15[0], x15[1]);

  // stage 4
  s2[0] = _mm_add_epi32(x2[0], x3[0]);
  s3[0] = _mm_sub_epi32(x2[0], x3[0]);
  s6[0] = _mm_add_epi32(x7[0], x6[0]);
  s7[0] = _mm_sub_epi32(x7[0], x6[0]);
  s10[0] = _mm_add_epi32(x11[0], x10[0]);
  s11[0] = _mm_sub_epi32(x11[0], x10[0]);
  s14[0] = _mm_add_epi32(x14[0], x15[0]);
  s15[0] = _mm_sub_epi32(x14[0], x15[0]);
  highbd_iadst_half_butterfly_sse4_1(s2[0], -cospi_16_64, s2);
  highbd_iadst_half_butterfly_sse4_1(s3[0], cospi_16_64, s3);
  highbd_iadst_half_butterfly_sse4_1(s6[0], cospi_16_64, s6);
  highbd_iadst_half_butterfly_sse4_1(s7[0], cospi_16_64, s7);
  highbd_iadst_half_butterfly_sse4_1(s10[0], cospi_16_64, s10);
  highbd_iadst_half_butterfly_sse4_1(s11[0], cospi_16_64, s11);
  highbd_iadst_half_butterfly_sse4_1(s14[0], -cospi_16_64, s14);
  highbd_iadst_half_butterfly_sse4_1(s15[0], cospi_16_64, s15);

  x2[0] = dct_const_round_shift_64bit(s2[0]);
  x2[1] = dct_const_round_shift_64bit(s2[1]);
  x3[0] = dct_const_round_shift_64bit(s3[0]);
  x3[1] = dct_const_round_shift_64bit(s3[1]);
  x6[0] = dct_const_round_shift_64bit(s6[0]);
  x6[1] = dct_const_round_shift_64bit(s6[1]);
  x7[0] = dct_const_round_shift_64bit(s7[0]);
  x7[1] = dct_const_round_shift_64bit(s7[1]);
  x10[0] = dct_const_round_shift_64bit(s10[0]);
  x10[1] = dct_const_round_shift_64bit(s10[1]);
  x11[0] = dct_const_round_shift_64bit(s11[0]);
  x11[1] = dct_const_round_shift_64bit(s11[1]);
  x14[0] = dct_const_round_shift_64bit(s14[0]);
  x14[1] = dct_const_round_shift_64bit(s14[1]);
  x15[0] = dct_const_round_shift_64bit(s15[0]);
  x15[1] = dct_const_round_shift_64bit(s15[1]);
  x2[0] = pack_4(x2[0], x2[1]);
  x3[0] = pack_4(x3[0], x3[1]);
  x6[0] = pack_4(x6[0], x6[1]);
  x7[0] = pack_4(x7[0], x7[1]);
  x10[0] = pack_4(x10[0], x10[1]);
  x11[0] = pack_4(x11[0], x11[1]);
  x14[0] = pack_4(x14[0], x14[1]);
  x15[0] = pack_4(x15[0], x15[1]);

  io[0] = x0[0];
  io[1] = _mm_sub_epi32(_mm_setzero_si128(), x8[0]);
  io[2] = x12[0];
  io[3] = _mm_sub_epi32(_mm_setzero_si128(), x4[0]);
  io[4] = x6[0];
  io[5] = x14[0];
  io[6] = x10[0];
  io[7] = x2[0];
  io[8] = x3[0];
  io[9] = x11[0];
  io[10] = x15[0];
  io[11] = x7[0];
  io[12] = x5[0];
  io[13] = _mm_sub_epi32(_mm_setzero_si128(), x13[0]);
  io[14] = x9[0];
  io[15] = _mm_sub_epi32(_mm_setzero_si128(), x1[0]);
}

void vp9_highbd_iht16x16_256_add_sse4_1(const tran_low_t *input, uint16_t *dest,
                                        int stride, int tx_type, int bd) {
  int i;
  __m128i out[16], *in;

  if (bd == 8) {
    __m128i l[16], r[16];

    in = l;
    for (i = 0; i < 2; i++) {
      highbd_load_pack_transpose_32bit_8x8(&input[0], 16, &in[0]);
      highbd_load_pack_transpose_32bit_8x8(&input[8], 16, &in[8]);
      if (tx_type == DCT_DCT || tx_type == ADST_DCT) {
        idct16_8col(in, in);
      } else {
        vpx_iadst16_8col_sse2(in);
      }
      in = r;
      input += 128;
    }

    for (i = 0; i < 16; i += 8) {
      int j;
      transpose_16bit_8x8(l + i, out);
      transpose_16bit_8x8(r + i, out + 8);
      if (tx_type == DCT_DCT || tx_type == DCT_ADST) {
        idct16_8col(out, out);
      } else {
        vpx_iadst16_8col_sse2(out);
      }

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
      if (tx_type == DCT_DCT || tx_type == ADST_DCT) {
        vpx_highbd_idct16_4col_sse4_1(in);
      } else {
        highbd_iadst16_4col_sse4_1(in);
      }
      input += 4 * 16;
    }

    for (i = 0; i < 16; i += 4) {
      int j;
      transpose_32bit_4x4(all[0] + i, out + 0);
      transpose_32bit_4x4(all[1] + i, out + 4);
      transpose_32bit_4x4(all[2] + i, out + 8);
      transpose_32bit_4x4(all[3] + i, out + 12);
      if (tx_type == DCT_DCT || tx_type == DCT_ADST) {
        vpx_highbd_idct16_4col_sse4_1(out);
      } else {
        highbd_iadst16_4col_sse4_1(out);
      }

      for (j = 0; j < 16; ++j) {
        highbd_write_buffer_4(dest + j * stride, out[j], bd);
      }
      dest += 4;
    }
  }
}
