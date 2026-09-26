/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/inv_txfm_msa.h"

static void idct32x8_row_transpose_store(const int16_t *input,
                                         int16_t *tmp_buf) {
  v8i16 m0, m1, m2, m3, m4, m5, m6, m7, n0, n1, n2, n3, n4, n5, n6, n7;

  /* 1st & 2nd 8x8 */
  LD_SH8(input, 32, m0, n0, m1, n1, m2, n2, m3, n3);
  LD_SH8((input + 8), 32, m4, n4, m5, n5, m6, n6, m7, n7);
  TRANSPOSE8x8_SH_SH(m0, n0, m1, n1, m2, n2, m3, n3, m0, n0, m1, n1, m2, n2, m3,
                     n3);
  TRANSPOSE8x8_SH_SH(m4, n4, m5, n5, m6, n6, m7, n7, m4, n4, m5, n5, m6, n6, m7,
                     n7);
  ST_SH8(m0, n0, m1, n1, m2, n2, m3, n3, (tmp_buf), 8);
  ST_SH4(m4, n4, m5, n5, (tmp_buf + 8 * 8), 8);
  ST_SH4(m6, n6, m7, n7, (tmp_buf + 12 * 8), 8);

  /* 3rd & 4th 8x8 */
  LD_SH8((input + 16), 32, m0, n0, m1, n1, m2, n2, m3, n3);
  LD_SH8((input + 24), 32, m4, n4, m5, n5, m6, n6, m7, n7);
  TRANSPOSE8x8_SH_SH(m0, n0, m1, n1, m2, n2, m3, n3, m0, n0, m1, n1, m2, n2, m3,
                     n3);
  TRANSPOSE8x8_SH_SH(m4, n4, m5, n5, m6, n6, m7, n7, m4, n4, m5, n5, m6, n6, m7,
                     n7);
  ST_SH4(m0, n0, m1, n1, (tmp_buf + 16 * 8), 8);
  ST_SH4(m2, n2, m3, n3, (tmp_buf + 20 * 8), 8);
  ST_SH4(m4, n4, m5, n5, (tmp_buf + 24 * 8), 8);
  ST_SH4(m6, n6, m7, n7, (tmp_buf + 28 * 8), 8);
}

static void idct32x8_row_even_process_store(int16_t *tmp_buf,
                                            int16_t *tmp_eve_buf) {
  v8i16 vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  v8i16 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  v8i16 stp0, stp1, stp2, stp3, stp4, stp5, stp6, stp7;

  /* Even stage 1 */
  LD_SH8(tmp_buf, 32, reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7);

  DOTP_CONST_PAIR(reg1, reg7, cospi_28_64, cospi_4_64, reg1, reg7);
  DOTP_CONST_PAIR(reg5, reg3, cospi_12_64, cospi_20_64, reg5, reg3);
  BUTTERFLY_4(reg1, reg7, reg3, reg5, vec1, vec3, vec2, vec0);
  DOTP_CONST_PAIR(vec2, vec0, cospi_16_64, cospi_16_64, loc2, loc3);

  loc1 = vec3;
  loc0 = vec1;

  DOTP_CONST_PAIR(reg0, reg4, cospi_16_64, cospi_16_64, reg0, reg4);
  DOTP_CONST_PAIR(reg2, reg6, cospi_24_64, cospi_8_64, reg2, reg6);
  BUTTERFLY_4(reg4, reg0, reg2, reg6, vec1, vec3, vec2, vec0);
  BUTTERFLY_4(vec0, vec1, loc1, loc0, stp3, stp0, stp7, stp4);
  BUTTERFLY_4(vec2, vec3, loc3, loc2, stp2, stp1, stp6, stp5);

  /* Even stage 2 */
  LD_SH8((tmp_buf + 16), 32, reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7);
  DOTP_CONST_PAIR(reg0, reg7, cospi_30_64, cospi_2_64, reg0, reg7);
  DOTP_CONST_PAIR(reg4, reg3, cospi_14_64, cospi_18_64, reg4, reg3);
  DOTP_CONST_PAIR(reg2, reg5, cospi_22_64, cospi_10_64, reg2, reg5);
  DOTP_CONST_PAIR(reg6, reg1, cospi_6_64, cospi_26_64, reg6, reg1);

  vec0 = reg0 + reg4;
  reg0 = reg0 - reg4;
  reg4 = reg6 + reg2;
  reg6 = reg6 - reg2;
  reg2 = reg1 + reg5;
  reg1 = reg1 - reg5;
  reg5 = reg7 + reg3;
  reg7 = reg7 - reg3;
  reg3 = vec0;

  vec1 = reg2;
  reg2 = reg3 + reg4;
  reg3 = reg3 - reg4;
  reg4 = reg5 - vec1;
  reg5 = reg5 + vec1;

  DOTP_CONST_PAIR(reg7, reg0, cospi_24_64, cospi_8_64, reg0, reg7);
  DOTP_CONST_PAIR((-reg6), reg1, cospi_24_64, cospi_8_64, reg6, reg1);

  vec0 = reg0 - reg6;
  reg0 = reg0 + reg6;
  vec1 = reg7 - reg1;
  reg7 = reg7 + reg1;

  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, reg6, reg1);
  DOTP_CONST_PAIR(reg4, reg3, cospi_16_64, cospi_16_64, reg3, reg4);

  /* Even stage 3 : Dependency on Even stage 1 & Even stage 2 */
  BUTTERFLY_4(stp0, stp1, reg7, reg5, loc1, loc3, loc2, loc0);
  ST_SH(loc0, (tmp_eve_buf + 15 * 8));
  ST_SH(loc1, (tmp_eve_buf));
  ST_SH(loc2, (tmp_eve_buf + 14 * 8));
  ST_SH(loc3, (tmp_eve_buf + 8));

  BUTTERFLY_4(stp2, stp3, reg4, reg1, loc1, loc3, loc2, loc0);
  ST_SH(loc0, (tmp_eve_buf + 13 * 8));
  ST_SH(loc1, (tmp_eve_buf + 2 * 8));
  ST_SH(loc2, (tmp_eve_buf + 12 * 8));
  ST_SH(loc3, (tmp_eve_buf + 3 * 8));

  /* Store 8 */
  BUTTERFLY_4(stp4, stp5, reg6, reg3, loc1, loc3, loc2, loc0);
  ST_SH(loc0, (tmp_eve_buf + 11 * 8));
  ST_SH(loc1, (tmp_eve_buf + 4 * 8));
  ST_SH(loc2, (tmp_eve_buf + 10 * 8));
  ST_SH(loc3, (tmp_eve_buf + 5 * 8));

  BUTTERFLY_4(stp6, stp7, reg2, reg0, loc1, loc3, loc2, loc0);
  ST_SH(loc0, (tmp_eve_buf + 9 * 8));
  ST_SH(loc1, (tmp_eve_buf + 6 * 8));
  ST_SH(loc2, (tmp_eve_buf + 8 * 8));
  ST_SH(loc3, (tmp_eve_buf + 7 * 8));
}

static void idct32x8_row_odd_process_store(int16_t *tmp_buf,
                                           int16_t *tmp_odd_buf) {
  v8i16 vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  v8i16 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;

  /* Odd stage 1 */
  reg0 = LD_SH(tmp_buf + 8);
  reg1 = LD_SH(tmp_buf + 7 * 8);
  reg2 = LD_SH(tmp_buf + 9 * 8);
  reg3 = LD_SH(tmp_buf + 15 * 8);
  reg4 = LD_SH(tmp_buf + 17 * 8);
  reg5 = LD_SH(tmp_buf + 23 * 8);
  reg6 = LD_SH(tmp_buf + 25 * 8);
  reg7 = LD_SH(tmp_buf + 31 * 8);

  DOTP_CONST_PAIR(reg0, reg7, cospi_31_64, cospi_1_64, reg0, reg7);
  DOTP_CONST_PAIR(reg4, reg3, cospi_15_64, cospi_17_64, reg3, reg4);
  DOTP_CONST_PAIR(reg2, reg5, cospi_23_64, cospi_9_64, reg2, reg5);
  DOTP_CONST_PAIR(reg6, reg1, cospi_7_64, cospi_25_64, reg1, reg6);

  vec0 = reg0 + reg3;
  reg0 = reg0 - reg3;
  reg3 = reg7 + reg4;
  reg7 = reg7 - reg4;
  reg4 = reg1 + reg2;
  reg1 = reg1 - reg2;
  reg2 = reg6 + reg5;
  reg6 = reg6 - reg5;
  reg5 = vec0;

  /* 4 Stores */
  ADD2(reg5, reg4, reg3, reg2, vec0, vec1);
  ST_SH2(vec0, vec1, (tmp_odd_buf + 4 * 8), 8);

  SUB2(reg5, reg4, reg3, reg2, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_24_64, cospi_8_64, vec0, vec1);
  ST_SH2(vec0, vec1, (tmp_odd_buf), 8);

  /* 4 Stores */
  DOTP_CONST_PAIR(reg7, reg0, cospi_28_64, cospi_4_64, reg0, reg7);
  DOTP_CONST_PAIR(reg6, reg1, -cospi_4_64, cospi_28_64, reg1, reg6);
  BUTTERFLY_4(reg0, reg7, reg6, reg1, vec0, vec1, vec2, vec3);
  ST_SH2(vec0, vec1, (tmp_odd_buf + 6 * 8), 8);

  DOTP_CONST_PAIR(vec2, vec3, cospi_24_64, cospi_8_64, vec2, vec3);
  ST_SH2(vec2, vec3, (tmp_odd_buf + 2 * 8), 8);

  /* Odd stage 2 */
  /* 8 loads */
  reg0 = LD_SH(tmp_buf + 3 * 8);
  reg1 = LD_SH(tmp_buf + 5 * 8);
  reg2 = LD_SH(tmp_buf + 11 * 8);
  reg3 = LD_SH(tmp_buf + 13 * 8);
  reg4 = LD_SH(tmp_buf + 19 * 8);
  reg5 = LD_SH(tmp_buf + 21 * 8);
  reg6 = LD_SH(tmp_buf + 27 * 8);
  reg7 = LD_SH(tmp_buf + 29 * 8);

  DOTP_CONST_PAIR(reg1, reg6, cospi_27_64, cospi_5_64, reg1, reg6);
  DOTP_CONST_PAIR(reg5, reg2, cospi_11_64, cospi_21_64, reg2, reg5);
  DOTP_CONST_PAIR(reg3, reg4, cospi_19_64, cospi_13_64, reg3, reg4);
  DOTP_CONST_PAIR(reg7, reg0, cospi_3_64, cospi_29_64, reg0, reg7);

  /* 4 Stores */
  SUB4(reg1, reg2, reg6, reg5, reg0, reg3, reg7, reg4, vec0, vec1, vec2, vec3);
  DOTP_CONST_PAIR(vec1, vec0, cospi_12_64, cospi_20_64, loc0, loc1);
  DOTP_CONST_PAIR(vec3, vec2, -cospi_20_64, cospi_12_64, loc2, loc3);

  BUTTERFLY_4(loc3, loc2, loc0, loc1, vec1, vec0, vec2, vec3);
  ST_SH2(vec0, vec1, (tmp_odd_buf + 12 * 8), 3 * 8);

  DOTP_CONST_PAIR(vec3, vec2, -cospi_8_64, cospi_24_64, vec0, vec1);
  ST_SH2(vec0, vec1, (tmp_odd_buf + 10 * 8), 8);

  /* 4 Stores */
  ADD4(reg1, reg2, reg6, reg5, reg0, reg3, reg7, reg4, vec1, vec2, vec0, vec3);
  BUTTERFLY_4(vec0, vec3, vec2, vec1, reg0, reg1, reg3, reg2);
  ST_SH(reg0, (tmp_odd_buf + 13 * 8));
  ST_SH(reg1, (tmp_odd_buf + 14 * 8));

  DOTP_CONST_PAIR(reg3, reg2, -cospi_8_64, cospi_24_64, reg0, reg1);
  ST_SH2(reg0, reg1, (tmp_odd_buf + 8 * 8), 8);

  /* Odd stage 3 : Dependency on Odd stage 1 & Odd stage 2 */

  /* Load 8 & Store 8 */
  LD_SH4(tmp_odd_buf, 8, reg0, reg1, reg2, reg3);
  LD_SH4((tmp_odd_buf + 8 * 8), 8, reg4, reg5, reg6, reg7);

  ADD4(reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7, loc0, loc1, loc2, loc3);
  ST_SH4(loc0, loc1, loc2, loc3, tmp_odd_buf, 8);

  SUB2(reg0, reg4, reg1, reg5, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc0, loc1);

  SUB2(reg2, reg6, reg3, reg7, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc2, loc3);
  ST_SH4(loc0, loc1, loc2, loc3, (tmp_odd_buf + 8 * 8), 8);

  /* Load 8 & Store 8 */
  LD_SH4((tmp_odd_buf + 4 * 8), 8, reg1, reg2, reg0, reg3);
  LD_SH4((tmp_odd_buf + 12 * 8), 8, reg4, reg5, reg6, reg7);

  ADD4(reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7, loc0, loc1, loc2, loc3);
  ST_SH4(loc0, loc1, loc2, loc3, (tmp_odd_buf + 4 * 8), 8);

  SUB2(reg0, reg4, reg3, reg7, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc0, loc1);

  SUB2(reg1, reg5, reg2, reg6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc2, loc3);
  ST_SH4(loc0, loc1, loc2, loc3, (tmp_odd_buf + 12 * 8), 8);
}

static void idct_butterfly_transpose_store(int16_t *tmp_buf,
                                           int16_t *tmp_eve_buf,
                                           int16_t *tmp_odd_buf, int16_t *dst) {
  v8i16 vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  v8i16 m0, m1, m2, m3, m4, m5, m6, m7, n0, n1, n2, n3, n4, n5, n6, n7;

  /* FINAL BUTTERFLY : Dependency on Even & Odd */
  vec0 = LD_SH(tmp_odd_buf);
  vec1 = LD_SH(tmp_odd_buf + 9 * 8);
  vec2 = LD_SH(tmp_odd_buf + 14 * 8);
  vec3 = LD_SH(tmp_odd_buf + 6 * 8);
  loc0 = LD_SH(tmp_eve_buf);
  loc1 = LD_SH(tmp_eve_buf + 8 * 8);
  loc2 = LD_SH(tmp_eve_buf + 4 * 8);
  loc3 = LD_SH(tmp_eve_buf + 12 * 8);

  ADD4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m0, m4, m2, m6);

  ST_SH((loc0 - vec3), (tmp_buf + 31 * 8));
  ST_SH((loc1 - vec2), (tmp_buf + 23 * 8));
  ST_SH((loc2 - vec1), (tmp_buf + 27 * 8));
  ST_SH((loc3 - vec0), (tmp_buf + 19 * 8));

  /* Load 8 & Store 8 */
  vec0 = LD_SH(tmp_odd_buf + 4 * 8);
  vec1 = LD_SH(tmp_odd_buf + 13 * 8);
  vec2 = LD_SH(tmp_odd_buf + 10 * 8);
  vec3 = LD_SH(tmp_odd_buf + 3 * 8);
  loc0 = LD_SH(tmp_eve_buf + 2 * 8);
  loc1 = LD_SH(tmp_eve_buf + 10 * 8);
  loc2 = LD_SH(tmp_eve_buf + 6 * 8);
  loc3 = LD_SH(tmp_eve_buf + 14 * 8);

  ADD4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m1, m5, m3, m7);

  ST_SH((loc0 - vec3), (tmp_buf + 29 * 8));
  ST_SH((loc1 - vec2), (tmp_buf + 21 * 8));
  ST_SH((loc2 - vec1), (tmp_buf + 25 * 8));
  ST_SH((loc3 - vec0), (tmp_buf + 17 * 8));

  /* Load 8 & Store 8 */
  vec0 = LD_SH(tmp_odd_buf + 2 * 8);
  vec1 = LD_SH(tmp_odd_buf + 11 * 8);
  vec2 = LD_SH(tmp_odd_buf + 12 * 8);
  vec3 = LD_SH(tmp_odd_buf + 7 * 8);
  loc0 = LD_SH(tmp_eve_buf + 1 * 8);
  loc1 = LD_SH(tmp_eve_buf + 9 * 8);
  loc2 = LD_SH(tmp_eve_buf + 5 * 8);
  loc3 = LD_SH(tmp_eve_buf + 13 * 8);

  ADD4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n0, n4, n2, n6);

  ST_SH((loc0 - vec3), (tmp_buf + 30 * 8));
  ST_SH((loc1 - vec2), (tmp_buf + 22 * 8));
  ST_SH((loc2 - vec1), (tmp_buf + 26 * 8));
  ST_SH((loc3 - vec0), (tmp_buf + 18 * 8));

  /* Load 8 & Store 8 */
  vec0 = LD_SH(tmp_odd_buf + 5 * 8);
  vec1 = LD_SH(tmp_odd_buf + 15 * 8);
  vec2 = LD_SH(tmp_odd_buf + 8 * 8);
  vec3 = LD_SH(tmp_odd_buf + 1 * 8);
  loc0 = LD_SH(tmp_eve_buf + 3 * 8);
  loc1 = LD_SH(tmp_eve_buf + 11 * 8);
  loc2 = LD_SH(tmp_eve_buf + 7 * 8);
  loc3 = LD_SH(tmp_eve_buf + 15 * 8);

  ADD4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n1, n5, n3, n7);

  ST_SH((loc0 - vec3), (tmp_buf + 28 * 8));
  ST_SH((loc1 - vec2), (tmp_buf + 20 * 8));
  ST_SH((loc2 - vec1), (tmp_buf + 24 * 8));
  ST_SH((loc3 - vec0), (tmp_buf + 16 * 8));

  /* Transpose : 16 vectors */
  /* 1st & 2nd 8x8 */
  TRANSPOSE8x8_SH_SH(m0, n0, m1, n1, m2, n2, m3, n3, m0, n0, m1, n1, m2, n2, m3,
                     n3);
  ST_SH4(m0, n0, m1, n1, (dst + 0), 32);
  ST_SH4(m2, n2, m3, n3, (dst + 4 * 32), 32);

  TRANSPOSE8x8_SH_SH(m4, n4, m5, n5, m6, n6, m7, n7, m4, n4, m5, n5, m6, n6, m7,
                     n7);
  ST_SH4(m4, n4, m5, n5, (dst + 8), 32);
  ST_SH4(m6, n6, m7, n7, (dst + 8 + 4 * 32), 32);

  /* 3rd & 4th 8x8 */
  LD_SH8((tmp_buf + 8 * 16), 8, m0, n0, m1, n1, m2, n2, m3, n3);
  LD_SH8((tmp_buf + 12 * 16), 8, m4, n4, m5, n5, m6, n6, m7, n7);
  TRANSPOSE8x8_SH_SH(m0, n0, m1, n1, m2, n2, m3, n3, m0, n0, m1, n1, m2, n2, m3,
                     n3);
  ST_SH4(m0, n0, m1, n1, (dst + 16), 32);
  ST_SH4(m2, n2, m3, n3, (dst + 16 + 4 * 32), 32);

  TRANSPOSE8x8_SH_SH(m4, n4, m5, n5, m6, n6, m7, n7, m4, n4, m5, n5, m6, n6, m7,
                     n7);
  ST_SH4(m4, n4, m5, n5, (dst + 24), 32);
  ST_SH4(m6, n6, m7, n7, (dst + 24 + 4 * 32), 32);
}

static void idct32x8_1d_rows_msa(const int16_t *input, int16_t *output) {
  DECLARE_ALIGNED(32, int16_t, tmp_buf[8 * 32]);
  DECLARE_ALIGNED(32, int16_t, tmp_odd_buf[16 * 8]);
  DECLARE_ALIGNED(32, int16_t, tmp_eve_buf[16 * 8]);

  idct32x8_row_transpose_store(input, &tmp_buf[0]);
  idct32x8_row_even_process_store(&tmp_buf[0], &tmp_eve_buf[0]);
  idct32x8_row_odd_process_store(&tmp_buf[0], &tmp_odd_buf[0]);
  idct_butterfly_transpose_store(&tmp_buf[0], &tmp_eve_buf[0], &tmp_odd_buf[0],
                                 output);
}

static void idct8x32_column_even_process_store(int16_t *tmp_buf,
                                               int16_t *tmp_eve_buf) {
  v8i16 vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  v8i16 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  v8i16 stp0, stp1, stp2, stp3, stp4, stp5, stp6, stp7;

  /* Even stage 1 */
  LD_SH8(tmp_buf, (4 * 32), reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7);
  tmp_buf += (2 * 32);

  DOTP_CONST_PAIR(reg1, reg7, cospi_28_64, cospi_4_64, reg1, reg7);
  DOTP_CONST_PAIR(reg5, reg3, cospi_12_64, cospi_20_64, reg5, reg3);
  BUTTERFLY_4(reg1, reg7, reg3, reg5, vec1, vec3, vec2, vec0);
  DOTP_CONST_PAIR(vec2, vec0, cospi_16_64, cospi_16_64, loc2, loc3);

  loc1 = vec3;
  loc0 = vec1;

  DOTP_CONST_PAIR(reg0, reg4, cospi_16_64, cospi_16_64, reg0, reg4);
  DOTP_CONST_PAIR(reg2, reg6, cospi_24_64, cospi_8_64, reg2, reg6);
  BUTTERFLY_4(reg4, reg0, reg2, reg6, vec1, vec3, vec2, vec0);
  BUTTERFLY_4(vec0, vec1, loc1, loc0, stp3, stp0, stp7, stp4);
  BUTTERFLY_4(vec2, vec3, loc3, loc2, stp2, stp1, stp6, stp5);

  /* Even stage 2 */
  /* Load 8 */
  LD_SH8(tmp_buf, (4 * 32), reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7);

  DOTP_CONST_PAIR(reg0, reg7, cospi_30_64, cospi_2_64, reg0, reg7);
  DOTP_CONST_PAIR(reg4, reg3, cospi_14_64, cospi_18_64, reg4, reg3);
  DOTP_CONST_PAIR(reg2, reg5, cospi_22_64, cospi_10_64, reg2, reg5);
  DOTP_CONST_PAIR(reg6, reg1, cospi_6_64, cospi_26_64, reg6, reg1);

  vec0 = reg0 + reg4;
  reg0 = reg0 - reg4;
  reg4 = reg6 + reg2;
  reg6 = reg6 - reg2;
  reg2 = reg1 + reg5;
  reg1 = reg1 - reg5;
  reg5 = reg7 + reg3;
  reg7 = reg7 - reg3;
  reg3 = vec0;

  vec1 = reg2;
  reg2 = reg3 + reg4;
  reg3 = reg3 - reg4;
  reg4 = reg5 - vec1;
  reg5 = reg5 + vec1;

  DOTP_CONST_PAIR(reg7, reg0, cospi_24_64, cospi_8_64, reg0, reg7);
  DOTP_CONST_PAIR((-reg6), reg1, cospi_24_64, cospi_8_64, reg6, reg1);

  vec0 = reg0 - reg6;
  reg0 = reg0 + reg6;
  vec1 = reg7 - reg1;
  reg7 = reg7 + reg1;

  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, reg6, reg1);
  DOTP_CONST_PAIR(reg4, reg3, cospi_16_64, cospi_16_64, reg3, reg4);

  /* Even stage 3 : Dependency on Even stage 1 & Even stage 2 */
  /* Store 8 */
  BUTTERFLY_4(stp0, stp1, reg7, reg5, loc1, loc3, loc2, loc0);
  ST_SH2(loc1, loc3, tmp_eve_buf, 8);
  ST_SH2(loc2, loc0, (tmp_eve_buf + 14 * 8), 8);

  BUTTERFLY_4(stp2, stp3, reg4, reg1, loc1, loc3, loc2, loc0);
  ST_SH2(loc1, loc3, (tmp_eve_buf + 2 * 8), 8);
  ST_SH2(loc2, loc0, (tmp_eve_buf + 12 * 8), 8);

  /* Store 8 */
  BUTTERFLY_4(stp4, stp5, reg6, reg3, loc1, loc3, loc2, loc0);
  ST_SH2(loc1, loc3, (tmp_eve_buf + 4 * 8), 8);
  ST_SH2(loc2, loc0, (tmp_eve_buf + 10 * 8), 8);

  BUTTERFLY_4(stp6, stp7, reg2, reg0, loc1, loc3, loc2, loc0);
  ST_SH2(loc1, loc3, (tmp_eve_buf + 6 * 8), 8);
  ST_SH2(loc2, loc0, (tmp_eve_buf + 8 * 8), 8);
}

static void idct8x32_column_odd_process_store(int16_t *tmp_buf,
                                              int16_t *tmp_odd_buf) {
  v8i16 vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  v8i16 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;

  /* Odd stage 1 */
  reg0 = LD_SH(tmp_buf + 32);
  reg1 = LD_SH(tmp_buf + 7 * 32);
  reg2 = LD_SH(tmp_buf + 9 * 32);
  reg3 = LD_SH(tmp_buf + 15 * 32);
  reg4 = LD_SH(tmp_buf + 17 * 32);
  reg5 = LD_SH(tmp_buf + 23 * 32);
  reg6 = LD_SH(tmp_buf + 25 * 32);
  reg7 = LD_SH(tmp_buf + 31 * 32);

  DOTP_CONST_PAIR(reg0, reg7, cospi_31_64, cospi_1_64, reg0, reg7);
  DOTP_CONST_PAIR(reg4, reg3, cospi_15_64, cospi_17_64, reg3, reg4);
  DOTP_CONST_PAIR(reg2, reg5, cospi_23_64, cospi_9_64, reg2, reg5);
  DOTP_CONST_PAIR(reg6, reg1, cospi_7_64, cospi_25_64, reg1, reg6);

  vec0 = reg0 + reg3;
  reg0 = reg0 - reg3;
  reg3 = reg7 + reg4;
  reg7 = reg7 - reg4;
  reg4 = reg1 + reg2;
  reg1 = reg1 - reg2;
  reg2 = reg6 + reg5;
  reg6 = reg6 - reg5;
  reg5 = vec0;

  /* 4 Stores */
  ADD2(reg5, reg4, reg3, reg2, vec0, vec1);
  ST_SH2(vec0, vec1, (tmp_odd_buf + 4 * 8), 8);
  SUB2(reg5, reg4, reg3, reg2, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_24_64, cospi_8_64, vec0, vec1);
  ST_SH2(vec0, vec1, tmp_odd_buf, 8);

  /* 4 Stores */
  DOTP_CONST_PAIR(reg7, reg0, cospi_28_64, cospi_4_64, reg0, reg7);
  DOTP_CONST_PAIR(reg6, reg1, -cospi_4_64, cospi_28_64, reg1, reg6);
  BUTTERFLY_4(reg0, reg7, reg6, reg1, vec0, vec1, vec2, vec3);
  ST_SH2(vec0, vec1, (tmp_odd_buf + 6 * 8), 8);
  DOTP_CONST_PAIR(vec2, vec3, cospi_24_64, cospi_8_64, vec2, vec3);
  ST_SH2(vec2, vec3, (tmp_odd_buf + 2 * 8), 8);

  /* Odd stage 2 */
  /* 8 loads */
  reg0 = LD_SH(tmp_buf + 3 * 32);
  reg1 = LD_SH(tmp_buf + 5 * 32);
  reg2 = LD_SH(tmp_buf + 11 * 32);
  reg3 = LD_SH(tmp_buf + 13 * 32);
  reg4 = LD_SH(tmp_buf + 19 * 32);
  reg5 = LD_SH(tmp_buf + 21 * 32);
  reg6 = LD_SH(tmp_buf + 27 * 32);
  reg7 = LD_SH(tmp_buf + 29 * 32);

  DOTP_CONST_PAIR(reg1, reg6, cospi_27_64, cospi_5_64, reg1, reg6);
  DOTP_CONST_PAIR(reg5, reg2, cospi_11_64, cospi_21_64, reg2, reg5);
  DOTP_CONST_PAIR(reg3, reg4, cospi_19_64, cospi_13_64, reg3, reg4);
  DOTP_CONST_PAIR(reg7, reg0, cospi_3_64, cospi_29_64, reg0, reg7);

  /* 4 Stores */
  SUB4(reg1, reg2, reg6, reg5, reg0, reg3, reg7, reg4, vec0, vec1, vec2, vec3);
  DOTP_CONST_PAIR(vec1, vec0, cospi_12_64, cospi_20_64, loc0, loc1);
  DOTP_CONST_PAIR(vec3, vec2, -cospi_20_64, cospi_12_64, loc2, loc3);
  BUTTERFLY_4(loc2, loc3, loc1, loc0, vec0, vec1, vec3, vec2);
  ST_SH2(vec0, vec1, (tmp_odd_buf + 12 * 8), 3 * 8);
  DOTP_CONST_PAIR(vec3, vec2, -cospi_8_64, cospi_24_64, vec0, vec1);
  ST_SH2(vec0, vec1, (tmp_odd_buf + 10 * 8), 8);

  /* 4 Stores */
  ADD4(reg0, reg3, reg1, reg2, reg5, reg6, reg4, reg7, vec0, vec1, vec2, vec3);
  BUTTERFLY_4(vec0, vec3, vec2, vec1, reg0, reg1, reg3, reg2);
  ST_SH2(reg0, reg1, (tmp_odd_buf + 13 * 8), 8);
  DOTP_CONST_PAIR(reg3, reg2, -cospi_8_64, cospi_24_64, reg0, reg1);
  ST_SH2(reg0, reg1, (tmp_odd_buf + 8 * 8), 8);

  /* Odd stage 3 : Dependency on Odd stage 1 & Odd stage 2 */
  /* Load 8 & Store 8 */
  LD_SH4(tmp_odd_buf, 8, reg0, reg1, reg2, reg3);
  LD_SH4((tmp_odd_buf + 8 * 8), 8, reg4, reg5, reg6, reg7);

  ADD4(reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7, loc0, loc1, loc2, loc3);
  ST_SH4(loc0, loc1, loc2, loc3, tmp_odd_buf, 8);

  SUB2(reg0, reg4, reg1, reg5, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc0, loc1);

  SUB2(reg2, reg6, reg3, reg7, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc2, loc3);
  ST_SH4(loc0, loc1, loc2, loc3, (tmp_odd_buf + 8 * 8), 8);

  /* Load 8 & Store 8 */
  LD_SH4((tmp_odd_buf + 4 * 8), 8, reg1, reg2, reg0, reg3);
  LD_SH4((tmp_odd_buf + 12 * 8), 8, reg4, reg5, reg6, reg7);

  ADD4(reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7, loc0, loc1, loc2, loc3);
  ST_SH4(loc0, loc1, loc2, loc3, (tmp_odd_buf + 4 * 8), 8);

  SUB2(reg0, reg4, reg3, reg7, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc0, loc1);

  SUB2(reg1, reg5, reg2, reg6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc2, loc3);
  ST_SH4(loc0, loc1, loc2, loc3, (tmp_odd_buf + 12 * 8), 8);
}

static void idct8x32_column_butterfly_addblk(int16_t *tmp_eve_buf,
                                             int16_t *tmp_odd_buf, uint8_t *dst,
                                             int32_t dst_stride) {
  v8i16 vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  v8i16 m0, m1, m2, m3, m4, m5, m6, m7, n0, n1, n2, n3, n4, n5, n6, n7;

  /* FINAL BUTTERFLY : Dependency on Even & Odd */
  vec0 = LD_SH(tmp_odd_buf);
  vec1 = LD_SH(tmp_odd_buf + 9 * 8);
  vec2 = LD_SH(tmp_odd_buf + 14 * 8);
  vec3 = LD_SH(tmp_odd_buf + 6 * 8);
  loc0 = LD_SH(tmp_eve_buf);
  loc1 = LD_SH(tmp_eve_buf + 8 * 8);
  loc2 = LD_SH(tmp_eve_buf + 4 * 8);
  loc3 = LD_SH(tmp_eve_buf + 12 * 8);

  ADD4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m0, m4, m2, m6);
  SRARI_H4_SH(m0, m2, m4, m6, 6);
  VP9_ADDBLK_ST8x4_UB(dst, (4 * dst_stride), m0, m2, m4, m6);

  SUB4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m6, m2, m4, m0);
  SRARI_H4_SH(m0, m2, m4, m6, 6);
  VP9_ADDBLK_ST8x4_UB((dst + 19 * dst_stride), (4 * dst_stride), m0, m2, m4,
                      m6);

  /* Load 8 & Store 8 */
  vec0 = LD_SH(tmp_odd_buf + 4 * 8);
  vec1 = LD_SH(tmp_odd_buf + 13 * 8);
  vec2 = LD_SH(tmp_odd_buf + 10 * 8);
  vec3 = LD_SH(tmp_odd_buf + 3 * 8);
  loc0 = LD_SH(tmp_eve_buf + 2 * 8);
  loc1 = LD_SH(tmp_eve_buf + 10 * 8);
  loc2 = LD_SH(tmp_eve_buf + 6 * 8);
  loc3 = LD_SH(tmp_eve_buf + 14 * 8);

  ADD4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m1, m5, m3, m7);
  SRARI_H4_SH(m1, m3, m5, m7, 6);
  VP9_ADDBLK_ST8x4_UB((dst + 2 * dst_stride), (4 * dst_stride), m1, m3, m5, m7);

  SUB4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m7, m3, m5, m1);
  SRARI_H4_SH(m1, m3, m5, m7, 6);
  VP9_ADDBLK_ST8x4_UB((dst + 17 * dst_stride), (4 * dst_stride), m1, m3, m5,
                      m7);

  /* Load 8 & Store 8 */
  vec0 = LD_SH(tmp_odd_buf + 2 * 8);
  vec1 = LD_SH(tmp_odd_buf + 11 * 8);
  vec2 = LD_SH(tmp_odd_buf + 12 * 8);
  vec3 = LD_SH(tmp_odd_buf + 7 * 8);
  loc0 = LD_SH(tmp_eve_buf + 1 * 8);
  loc1 = LD_SH(tmp_eve_buf + 9 * 8);
  loc2 = LD_SH(tmp_eve_buf + 5 * 8);
  loc3 = LD_SH(tmp_eve_buf + 13 * 8);

  ADD4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n0, n4, n2, n6);
  SRARI_H4_SH(n0, n2, n4, n6, 6);
  VP9_ADDBLK_ST8x4_UB((dst + 1 * dst_stride), (4 * dst_stride), n0, n2, n4, n6);

  SUB4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n6, n2, n4, n0);
  SRARI_H4_SH(n0, n2, n4, n6, 6);
  VP9_ADDBLK_ST8x4_UB((dst + 18 * dst_stride), (4 * dst_stride), n0, n2, n4,
                      n6);

  /* Load 8 & Store 8 */
  vec0 = LD_SH(tmp_odd_buf + 5 * 8);
  vec1 = LD_SH(tmp_odd_buf + 15 * 8);
  vec2 = LD_SH(tmp_odd_buf + 8 * 8);
  vec3 = LD_SH(tmp_odd_buf + 1 * 8);
  loc0 = LD_SH(tmp_eve_buf + 3 * 8);
  loc1 = LD_SH(tmp_eve_buf + 11 * 8);
  loc2 = LD_SH(tmp_eve_buf + 7 * 8);
  loc3 = LD_SH(tmp_eve_buf + 15 * 8);

  ADD4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n1, n5, n3, n7);
  SRARI_H4_SH(n1, n3, n5, n7, 6);
  VP9_ADDBLK_ST8x4_UB((dst + 3 * dst_stride), (4 * dst_stride), n1, n3, n5, n7);

  SUB4(loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n7, n3, n5, n1);
  SRARI_H4_SH(n1, n3, n5, n7, 6);
  VP9_ADDBLK_ST8x4_UB((dst + 16 * dst_stride), (4 * dst_stride), n1, n3, n5,
                      n7);
}

static void idct8x32_1d_columns_addblk_msa(int16_t *input, uint8_t *dst,
                                           int32_t dst_stride) {
  DECLARE_ALIGNED(32, int16_t, tmp_odd_buf[16 * 8]);
  DECLARE_ALIGNED(32, int16_t, tmp_eve_buf[16 * 8]);

  idct8x32_column_even_process_store(input, &tmp_eve_buf[0]);
  idct8x32_column_odd_process_store(input, &tmp_odd_buf[0]);
  idct8x32_column_butterfly_addblk(&tmp_eve_buf[0], &tmp_odd_buf[0], dst,
                                   dst_stride);
}

void vpx_idct32x32_1024_add_msa(const int16_t *input, uint8_t *dst,
                                int32_t dst_stride) {
  int32_t i;
  DECLARE_ALIGNED(32, int16_t, out_arr[32 * 32]);
  int16_t *out_ptr = out_arr;

  /* transform rows */
  for (i = 0; i < 4; ++i) {
    /* process 32 * 8 block */
    idct32x8_1d_rows_msa((input + (i << 8)), (out_ptr + (i << 8)));
  }

  /* transform columns */
  for (i = 0; i < 4; ++i) {
    /* process 8 * 32 block */
    idct8x32_1d_columns_addblk_msa((out_ptr + (i << 3)), (dst + (i << 3)),
                                   dst_stride);
  }
}

void vpx_idct32x32_34_add_msa(const int16_t *input, uint8_t *dst,
                              int32_t dst_stride) {
  int32_t i;
  DECLARE_ALIGNED(32, int16_t, out_arr[32 * 32]);
  int16_t *out_ptr = out_arr;

  for (i = 32; i--;) {
    __asm__ __volatile__(
        "sw     $zero,      0(%[out_ptr])     \n\t"
        "sw     $zero,      4(%[out_ptr])     \n\t"
        "sw     $zero,      8(%[out_ptr])     \n\t"
        "sw     $zero,     12(%[out_ptr])     \n\t"
        "sw     $zero,     16(%[out_ptr])     \n\t"
        "sw     $zero,     20(%[out_ptr])     \n\t"
        "sw     $zero,     24(%[out_ptr])     \n\t"
        "sw     $zero,     28(%[out_ptr])     \n\t"
        "sw     $zero,     32(%[out_ptr])     \n\t"
        "sw     $zero,     36(%[out_ptr])     \n\t"
        "sw     $zero,     40(%[out_ptr])     \n\t"
        "sw     $zero,     44(%[out_ptr])     \n\t"
        "sw     $zero,     48(%[out_ptr])     \n\t"
        "sw     $zero,     52(%[out_ptr])     \n\t"
        "sw     $zero,     56(%[out_ptr])     \n\t"
        "sw     $zero,     60(%[out_ptr])     \n\t"

        :
        : [out_ptr] "r"(out_ptr));

    out_ptr += 32;
  }

  out_ptr = out_arr;

  /* rows: only upper-left 8x8 has non-zero coeff */
  idct32x8_1d_rows_msa(input, out_ptr);

  /* transform columns */
  for (i = 0; i < 4; ++i) {
    /* process 8 * 32 block */
    idct8x32_1d_columns_addblk_msa((out_ptr + (i << 3)), (dst + (i << 3)),
                                   dst_stride);
  }
}

void vpx_idct32x32_1_add_msa(const int16_t *input, uint8_t *dst,
                             int32_t dst_stride) {
  int32_t i;
  int16_t out;
  v16u8 dst0, dst1, dst2, dst3, tmp0, tmp1, tmp2, tmp3;
  v8i16 res0, res1, res2, res3, res4, res5, res6, res7, vec;

  out = ROUND_POWER_OF_TWO((input[0] * cospi_16_64), DCT_CONST_BITS);
  out = ROUND_POWER_OF_TWO((out * cospi_16_64), DCT_CONST_BITS);
  out = ROUND_POWER_OF_TWO(out, 6);

  vec = __msa_fill_h(out);

  for (i = 16; i--;) {
    LD_UB2(dst, 16, dst0, dst1);
    LD_UB2(dst + dst_stride, 16, dst2, dst3);

    UNPCK_UB_SH(dst0, res0, res4);
    UNPCK_UB_SH(dst1, res1, res5);
    UNPCK_UB_SH(dst2, res2, res6);
    UNPCK_UB_SH(dst3, res3, res7);
    ADD4(res0, vec, res1, vec, res2, vec, res3, vec, res0, res1, res2, res3);
    ADD4(res4, vec, res5, vec, res6, vec, res7, vec, res4, res5, res6, res7);
    CLIP_SH4_0_255(res0, res1, res2, res3);
    CLIP_SH4_0_255(res4, res5, res6, res7);
    PCKEV_B4_UB(res4, res0, res5, res1, res6, res2, res7, res3, tmp0, tmp1,
                tmp2, tmp3);

    ST_UB2(tmp0, tmp1, dst, 16);
    dst += dst_stride;
    ST_UB2(tmp2, tmp3, dst, 16);
    dst += dst_stride;
  }
}
