/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/loongarch/fwd_txfm_lsx.h"

#define UNPCK_UB_SH(_in, _out0, _out1)   \
  do {                                   \
    _out0 = __lsx_vsllwil_hu_bu(_in, 0); \
    _out1 = __lsx_vexth_hu_bu(_in);      \
  } while (0)

static void idct32x8_row_transpose_store(const int16_t *input,
                                         int16_t *tmp_buf) {
  __m128i m0, m1, m2, m3, m4, m5, m6, m7;
  __m128i n0, n1, n2, n3, n4, n5, n6, n7;

  /* 1st & 2nd 8x8 */
  DUP4_ARG2(__lsx_vld, input, 0, input, 64, input, 128, input, 192, m0, n0, m1,
            n1);
  DUP4_ARG2(__lsx_vld, input, 256, input, 320, input, 384, input, 448, m2, n2,
            m3, n3);
  DUP4_ARG2(__lsx_vld, input, 16, input, 80, input, 144, input, 208, m4, n4, m5,
            n5);
  DUP4_ARG2(__lsx_vld, input, 272, input, 336, input, 400, input, 464, m6, n6,
            m7, n7);

  LSX_TRANSPOSE8x8_H(m0, n0, m1, n1, m2, n2, m3, n3, m0, n0, m1, n1, m2, n2, m3,
                     n3);
  LSX_TRANSPOSE8x8_H(m4, n4, m5, n5, m6, n6, m7, n7, m4, n4, m5, n5, m6, n6, m7,
                     n7);

  __lsx_vst(m0, tmp_buf, 0);
  __lsx_vst(n0, tmp_buf, 16);
  __lsx_vst(m1, tmp_buf, 32);
  __lsx_vst(n1, tmp_buf, 48);
  __lsx_vst(m2, tmp_buf, 64);
  __lsx_vst(n2, tmp_buf, 80);
  __lsx_vst(m3, tmp_buf, 96);
  __lsx_vst(n3, tmp_buf, 112);
  __lsx_vst(m4, tmp_buf, 128);
  __lsx_vst(n4, tmp_buf, 144);
  __lsx_vst(m5, tmp_buf, 160);
  __lsx_vst(n5, tmp_buf, 176);
  __lsx_vst(m6, tmp_buf, 192);
  __lsx_vst(n6, tmp_buf, 208);
  __lsx_vst(m7, tmp_buf, 224);
  __lsx_vst(n7, tmp_buf, 240);

  /* 3rd & 4th 8x8 */
  DUP4_ARG2(__lsx_vld, input, 32, input, 96, input, 160, input, 224, m0, n0, m1,
            n1);
  DUP4_ARG2(__lsx_vld, input, 288, input, 352, input, 416, input, 480, m2, n2,
            m3, n3);
  DUP4_ARG2(__lsx_vld, input, 48, input, 112, input, 176, input, 240, m4, n4,
            m5, n5);
  DUP4_ARG2(__lsx_vld, input, 304, input, 368, input, 432, input, 496, m6, n6,
            m7, n7);

  LSX_TRANSPOSE8x8_H(m0, n0, m1, n1, m2, n2, m3, n3, m0, n0, m1, n1, m2, n2, m3,
                     n3);
  LSX_TRANSPOSE8x8_H(m4, n4, m5, n5, m6, n6, m7, n7, m4, n4, m5, n5, m6, n6, m7,
                     n7);

  __lsx_vst(m0, tmp_buf, 256);
  __lsx_vst(n0, tmp_buf, 272);
  __lsx_vst(m1, tmp_buf, 288);
  __lsx_vst(n1, tmp_buf, 304);
  __lsx_vst(m2, tmp_buf, 320);
  __lsx_vst(n2, tmp_buf, 336);
  __lsx_vst(m3, tmp_buf, 352);
  __lsx_vst(n3, tmp_buf, 368);
  __lsx_vst(m4, tmp_buf, 384);
  __lsx_vst(n4, tmp_buf, 400);
  __lsx_vst(m5, tmp_buf, 416);
  __lsx_vst(n5, tmp_buf, 432);
  __lsx_vst(m6, tmp_buf, 448);
  __lsx_vst(n6, tmp_buf, 464);
  __lsx_vst(m7, tmp_buf, 480);
  __lsx_vst(n7, tmp_buf, 496);
}

static void idct32x8_row_even_process_store(int16_t *tmp_buf,
                                            int16_t *tmp_eve_buf) {
  __m128i vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i stp0, stp1, stp2, stp3, stp4, stp5, stp6, stp7;
  __m128i tmp0;

  /* Even stage 1 */
  DUP4_ARG2(__lsx_vld, tmp_buf, 0, tmp_buf, 64, tmp_buf, 128, tmp_buf, 192,
            reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_buf, 256, tmp_buf, 320, tmp_buf, 384, tmp_buf, 448,
            reg4, reg5, reg6, reg7);

  DOTP_CONST_PAIR(reg1, reg7, cospi_28_64, cospi_4_64, reg1, reg7);
  DOTP_CONST_PAIR(reg5, reg3, cospi_12_64, cospi_20_64, reg5, reg3);
  LSX_BUTTERFLY_4_H(reg1, reg7, reg3, reg5, vec1, vec3, vec2, vec0);
  DOTP_CONST_PAIR(vec2, vec0, cospi_16_64, cospi_16_64, loc2, loc3);

  loc1 = vec3;
  loc0 = vec1;

  DOTP_CONST_PAIR(reg0, reg4, cospi_16_64, cospi_16_64, reg0, reg4);
  DOTP_CONST_PAIR(reg2, reg6, cospi_24_64, cospi_8_64, reg2, reg6);
  LSX_BUTTERFLY_4_H(reg4, reg0, reg2, reg6, vec1, vec3, vec2, vec0);
  LSX_BUTTERFLY_4_H(vec0, vec1, loc1, loc0, stp3, stp0, stp7, stp4);
  LSX_BUTTERFLY_4_H(vec2, vec3, loc3, loc2, stp2, stp1, stp6, stp5);

  /* Even stage 2 */
  DUP4_ARG2(__lsx_vld, tmp_buf, 32, tmp_buf, 96, tmp_buf, 160, tmp_buf, 224,
            reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_buf, 288, tmp_buf, 352, tmp_buf, 416, tmp_buf, 480,
            reg4, reg5, reg6, reg7);
  DOTP_CONST_PAIR(reg0, reg7, cospi_30_64, cospi_2_64, reg0, reg7);
  DOTP_CONST_PAIR(reg4, reg3, cospi_14_64, cospi_18_64, reg4, reg3);
  DOTP_CONST_PAIR(reg2, reg5, cospi_22_64, cospi_10_64, reg2, reg5);
  DOTP_CONST_PAIR(reg6, reg1, cospi_6_64, cospi_26_64, reg6, reg1);

  vec0 = __lsx_vadd_h(reg0, reg4);
  reg0 = __lsx_vsub_h(reg0, reg4);
  reg4 = __lsx_vadd_h(reg6, reg2);
  reg6 = __lsx_vsub_h(reg6, reg2);
  reg2 = __lsx_vadd_h(reg1, reg5);
  reg1 = __lsx_vsub_h(reg1, reg5);
  reg5 = __lsx_vadd_h(reg7, reg3);
  reg7 = __lsx_vsub_h(reg7, reg3);
  reg3 = vec0;

  vec1 = reg2;
  reg2 = __lsx_vadd_h(reg3, reg4);
  reg3 = __lsx_vsub_h(reg3, reg4);
  reg4 = __lsx_vsub_h(reg5, vec1);
  reg5 = __lsx_vadd_h(reg5, vec1);

  tmp0 = __lsx_vneg_h(reg6);
  DOTP_CONST_PAIR(reg7, reg0, cospi_24_64, cospi_8_64, reg0, reg7);
  DOTP_CONST_PAIR(tmp0, reg1, cospi_24_64, cospi_8_64, reg6, reg1);

  vec0 = __lsx_vsub_h(reg0, reg6);
  reg0 = __lsx_vadd_h(reg0, reg6);
  vec1 = __lsx_vsub_h(reg7, reg1);
  reg7 = __lsx_vadd_h(reg7, reg1);

  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, reg6, reg1);
  DOTP_CONST_PAIR(reg4, reg3, cospi_16_64, cospi_16_64, reg3, reg4);

  /* Even stage 3 : Dependency on Even stage 1 & Even stage 2 */
  LSX_BUTTERFLY_4_H(stp0, stp1, reg7, reg5, loc1, loc3, loc2, loc0);
  __lsx_vst(loc0, tmp_eve_buf, 240);
  __lsx_vst(loc1, tmp_eve_buf, 0);
  __lsx_vst(loc2, tmp_eve_buf, 224);
  __lsx_vst(loc3, tmp_eve_buf, 16);

  LSX_BUTTERFLY_4_H(stp2, stp3, reg4, reg1, loc1, loc3, loc2, loc0);
  __lsx_vst(loc0, tmp_eve_buf, 208);
  __lsx_vst(loc1, tmp_eve_buf, 32);
  __lsx_vst(loc2, tmp_eve_buf, 192);
  __lsx_vst(loc3, tmp_eve_buf, 48);

  /* Store 8 */
  LSX_BUTTERFLY_4_H(stp4, stp5, reg6, reg3, loc1, loc3, loc2, loc0);
  __lsx_vst(loc0, tmp_eve_buf, 176);
  __lsx_vst(loc1, tmp_eve_buf, 64);
  __lsx_vst(loc2, tmp_eve_buf, 160);
  __lsx_vst(loc3, tmp_eve_buf, 80);

  LSX_BUTTERFLY_4_H(stp6, stp7, reg2, reg0, loc1, loc3, loc2, loc0);
  __lsx_vst(loc0, tmp_eve_buf, 144);
  __lsx_vst(loc1, tmp_eve_buf, 96);
  __lsx_vst(loc2, tmp_eve_buf, 128);
  __lsx_vst(loc3, tmp_eve_buf, 112);
}

static void idct32x8_row_odd_process_store(int16_t *tmp_buf,
                                           int16_t *tmp_odd_buf) {
  __m128i vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;

  /* Odd stage 1 */
  DUP4_ARG2(__lsx_vld, tmp_buf, 16, tmp_buf, 112, tmp_buf, 144, tmp_buf, 240,
            reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_buf, 272, tmp_buf, 368, tmp_buf, 400, tmp_buf, 496,
            reg4, reg5, reg6, reg7);

  DOTP_CONST_PAIR(reg0, reg7, cospi_31_64, cospi_1_64, reg0, reg7);
  DOTP_CONST_PAIR(reg4, reg3, cospi_15_64, cospi_17_64, reg3, reg4);
  DOTP_CONST_PAIR(reg2, reg5, cospi_23_64, cospi_9_64, reg2, reg5);
  DOTP_CONST_PAIR(reg6, reg1, cospi_7_64, cospi_25_64, reg1, reg6);

  vec0 = __lsx_vadd_h(reg0, reg3);
  reg0 = __lsx_vsub_h(reg0, reg3);
  reg3 = __lsx_vadd_h(reg7, reg4);
  reg7 = __lsx_vsub_h(reg7, reg4);
  reg4 = __lsx_vadd_h(reg1, reg2);
  reg1 = __lsx_vsub_h(reg1, reg2);
  reg2 = __lsx_vadd_h(reg6, reg5);
  reg6 = __lsx_vsub_h(reg6, reg5);
  reg5 = vec0;

  /* 4 Stores */
  DUP2_ARG2(__lsx_vadd_h, reg5, reg4, reg3, reg2, vec0, vec1);
  __lsx_vst(vec0, tmp_odd_buf, 64);
  __lsx_vst(vec1, tmp_odd_buf, 80);

  DUP2_ARG2(__lsx_vsub_h, reg5, reg4, reg3, reg2, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_24_64, cospi_8_64, vec0, vec1);
  __lsx_vst(vec0, tmp_odd_buf, 0);
  __lsx_vst(vec1, tmp_odd_buf, 16);

  /* 4 Stores */
  DOTP_CONST_PAIR(reg7, reg0, cospi_28_64, cospi_4_64, reg0, reg7);
  DOTP_CONST_PAIR(reg6, reg1, -cospi_4_64, cospi_28_64, reg1, reg6);
  LSX_BUTTERFLY_4_H(reg0, reg7, reg6, reg1, vec0, vec1, vec2, vec3);
  __lsx_vst(vec0, tmp_odd_buf, 96);
  __lsx_vst(vec1, tmp_odd_buf, 112);

  DOTP_CONST_PAIR(vec2, vec3, cospi_24_64, cospi_8_64, vec2, vec3);
  __lsx_vst(vec2, tmp_odd_buf, 32);
  __lsx_vst(vec3, tmp_odd_buf, 48);

  /* Odd stage 2 */
  /* 8 loads */
  DUP4_ARG2(__lsx_vld, tmp_buf, 48, tmp_buf, 80, tmp_buf, 176, tmp_buf, 208,
            reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_buf, 304, tmp_buf, 336, tmp_buf, 432, tmp_buf, 464,
            reg4, reg5, reg6, reg7);

  DOTP_CONST_PAIR(reg1, reg6, cospi_27_64, cospi_5_64, reg1, reg6);
  DOTP_CONST_PAIR(reg5, reg2, cospi_11_64, cospi_21_64, reg2, reg5);
  DOTP_CONST_PAIR(reg3, reg4, cospi_19_64, cospi_13_64, reg3, reg4);
  DOTP_CONST_PAIR(reg7, reg0, cospi_3_64, cospi_29_64, reg0, reg7);

  /* 4 Stores */
  DUP4_ARG2(__lsx_vsub_h, reg1, reg2, reg6, reg5, reg0, reg3, reg7, reg4, vec0,
            vec1, vec2, vec3);
  DOTP_CONST_PAIR(vec1, vec0, cospi_12_64, cospi_20_64, loc0, loc1);
  DOTP_CONST_PAIR(vec3, vec2, -cospi_20_64, cospi_12_64, loc2, loc3);

  LSX_BUTTERFLY_4_H(loc3, loc2, loc0, loc1, vec1, vec0, vec2, vec3);
  __lsx_vst(vec0, tmp_odd_buf, 192);
  __lsx_vst(vec1, tmp_odd_buf, 240);

  DOTP_CONST_PAIR(vec3, vec2, -cospi_8_64, cospi_24_64, vec0, vec1);
  __lsx_vst(vec0, tmp_odd_buf, 160);
  __lsx_vst(vec1, tmp_odd_buf, 176);

  /* 4 Stores */
  DUP4_ARG2(__lsx_vadd_h, reg1, reg2, reg6, reg5, reg0, reg3, reg7, reg4, vec1,
            vec2, vec0, vec3);
  LSX_BUTTERFLY_4_H(vec0, vec3, vec2, vec1, reg0, reg1, reg3, reg2);
  __lsx_vst(reg0, tmp_odd_buf, 208);
  __lsx_vst(reg1, tmp_odd_buf, 224);

  DOTP_CONST_PAIR(reg3, reg2, -cospi_8_64, cospi_24_64, reg0, reg1);
  __lsx_vst(reg0, tmp_odd_buf, 128);
  __lsx_vst(reg1, tmp_odd_buf, 144);

  /* Odd stage 3 : Dependency on Odd stage 1 & Odd stage 2 */

  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 0, tmp_odd_buf, 16, tmp_odd_buf, 32,
            tmp_odd_buf, 48, reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 128, tmp_odd_buf, 144, tmp_odd_buf, 160,
            tmp_odd_buf, 176, reg4, reg5, reg6, reg7);
  DUP4_ARG2(__lsx_vadd_h, reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7, loc0,
            loc1, loc2, loc3);
  __lsx_vst(loc0, tmp_odd_buf, 0);
  __lsx_vst(loc1, tmp_odd_buf, 16);
  __lsx_vst(loc2, tmp_odd_buf, 32);
  __lsx_vst(loc3, tmp_odd_buf, 48);

  DUP2_ARG2(__lsx_vsub_h, reg0, reg4, reg1, reg5, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc0, loc1);

  DUP2_ARG2(__lsx_vsub_h, reg2, reg6, reg3, reg7, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc2, loc3);
  __lsx_vst(loc0, tmp_odd_buf, 128);
  __lsx_vst(loc1, tmp_odd_buf, 144);
  __lsx_vst(loc2, tmp_odd_buf, 160);
  __lsx_vst(loc3, tmp_odd_buf, 176);

  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 64, tmp_odd_buf, 80, tmp_odd_buf, 96,
            tmp_odd_buf, 112, reg1, reg2, reg0, reg3);
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 192, tmp_odd_buf, 208, tmp_odd_buf, 224,
            tmp_odd_buf, 240, reg4, reg5, reg6, reg7);

  DUP4_ARG2(__lsx_vadd_h, reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7, loc0,
            loc1, loc2, loc3);
  __lsx_vst(loc0, tmp_odd_buf, 64);
  __lsx_vst(loc1, tmp_odd_buf, 80);
  __lsx_vst(loc2, tmp_odd_buf, 96);
  __lsx_vst(loc3, tmp_odd_buf, 112);

  DUP2_ARG2(__lsx_vsub_h, reg0, reg4, reg3, reg7, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc0, loc1);
  DUP2_ARG2(__lsx_vsub_h, reg1, reg5, reg2, reg6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc2, loc3);
  __lsx_vst(loc0, tmp_odd_buf, 192);
  __lsx_vst(loc1, tmp_odd_buf, 208);
  __lsx_vst(loc2, tmp_odd_buf, 224);
  __lsx_vst(loc3, tmp_odd_buf, 240);
}

static void idct_butterfly_transpose_store(int16_t *tmp_buf,
                                           int16_t *tmp_eve_buf,
                                           int16_t *tmp_odd_buf, int16_t *dst) {
  __m128i vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  __m128i m0, m1, m2, m3, m4, m5, m6, m7;
  __m128i n0, n1, n2, n3, n4, n5, n6, n7;
  __m128i reg0, reg1, reg2, reg3;

  /* FINAL BUTTERFLY : Dependency on Even & Odd */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 0, tmp_odd_buf, 144, tmp_odd_buf, 224,
            tmp_odd_buf, 96, vec0, vec1, vec2, vec3);
  DUP4_ARG2(__lsx_vld, tmp_eve_buf, 0, tmp_eve_buf, 128, tmp_eve_buf, 64,
            tmp_eve_buf, 192, loc0, loc1, loc2, loc3);

  DUP4_ARG2(__lsx_vadd_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m0,
            m4, m2, m6);
  DUP4_ARG2(__lsx_vsub_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, reg0,
            reg1, reg2, reg3);
  __lsx_vst(reg0, tmp_buf, 496);
  __lsx_vst(reg1, tmp_buf, 368);
  __lsx_vst(reg2, tmp_buf, 432);
  __lsx_vst(reg3, tmp_buf, 304);

  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 64, tmp_odd_buf, 208, tmp_odd_buf, 160,
            tmp_odd_buf, 48, vec0, vec1, vec2, vec3);
  DUP4_ARG2(__lsx_vld, tmp_eve_buf, 32, tmp_eve_buf, 160, tmp_eve_buf, 96,
            tmp_eve_buf, 224, loc0, loc1, loc2, loc3);

  DUP4_ARG2(__lsx_vadd_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m1,
            m5, m3, m7);
  DUP4_ARG2(__lsx_vsub_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, reg0,
            reg1, reg2, reg3);
  __lsx_vst(reg0, tmp_buf, 464);
  __lsx_vst(reg1, tmp_buf, 336);
  __lsx_vst(reg2, tmp_buf, 400);
  __lsx_vst(reg3, tmp_buf, 272);

  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 32, tmp_odd_buf, 176, tmp_odd_buf, 192,
            tmp_odd_buf, 112, vec0, vec1, vec2, vec3);
  DUP4_ARG2(__lsx_vld, tmp_eve_buf, 16, tmp_eve_buf, 144, tmp_eve_buf, 80,
            tmp_eve_buf, 208, loc0, loc1, loc2, loc3);

  DUP4_ARG2(__lsx_vadd_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n0,
            n4, n2, n6);
  DUP4_ARG2(__lsx_vsub_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, reg0,
            reg1, reg2, reg3);
  __lsx_vst(reg0, tmp_buf, 480);
  __lsx_vst(reg1, tmp_buf, 352);
  __lsx_vst(reg2, tmp_buf, 416);
  __lsx_vst(reg3, tmp_buf, 288);

  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 80, tmp_odd_buf, 240, tmp_odd_buf, 128,
            tmp_odd_buf, 16, vec0, vec1, vec2, vec3);
  DUP4_ARG2(__lsx_vld, tmp_eve_buf, 48, tmp_eve_buf, 176, tmp_eve_buf, 112,
            tmp_eve_buf, 240, loc0, loc1, loc2, loc3);
  DUP4_ARG2(__lsx_vadd_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n1,
            n5, n3, n7);
  DUP4_ARG2(__lsx_vsub_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, reg0,
            reg1, reg2, reg3);
  __lsx_vst(reg0, tmp_buf, 448);
  __lsx_vst(reg1, tmp_buf, 320);
  __lsx_vst(reg2, tmp_buf, 384);
  __lsx_vst(reg3, tmp_buf, 256);

  /* Transpose : 16 vectors */
  /* 1st & 2nd 8x8 */
  LSX_TRANSPOSE8x8_H(m0, n0, m1, n1, m2, n2, m3, n3, m0, n0, m1, n1, m2, n2, m3,
                     n3);
  __lsx_vst(m0, dst, 0);
  __lsx_vst(n0, dst, 64);
  __lsx_vst(m1, dst, 128);
  __lsx_vst(n1, dst, 192);
  __lsx_vst(m2, dst, 256);
  __lsx_vst(n2, dst, 320);
  __lsx_vst(m3, dst, 384);
  __lsx_vst(n3, dst, 448);

  LSX_TRANSPOSE8x8_H(m4, n4, m5, n5, m6, n6, m7, n7, m4, n4, m5, n5, m6, n6, m7,
                     n7);
  __lsx_vst(m4, dst, 16);
  __lsx_vst(n4, dst, 80);
  __lsx_vst(m5, dst, 144);
  __lsx_vst(n5, dst, 208);
  __lsx_vst(m6, dst, 272);
  __lsx_vst(n6, dst, 336);
  __lsx_vst(m7, dst, 400);
  __lsx_vst(n7, dst, 464);

  /* 3rd & 4th 8x8 */
  DUP4_ARG2(__lsx_vld, tmp_buf, 256, tmp_buf, 272, tmp_buf, 288, tmp_buf, 304,
            m0, n0, m1, n1);
  DUP4_ARG2(__lsx_vld, tmp_buf, 320, tmp_buf, 336, tmp_buf, 352, tmp_buf, 368,
            m2, n2, m3, n3);
  DUP4_ARG2(__lsx_vld, tmp_buf, 384, tmp_buf, 400, tmp_buf, 416, tmp_buf, 432,
            m4, n4, m5, n5);
  DUP4_ARG2(__lsx_vld, tmp_buf, 448, tmp_buf, 464, tmp_buf, 480, tmp_buf, 496,
            m6, n6, m7, n7);
  LSX_TRANSPOSE8x8_H(m0, n0, m1, n1, m2, n2, m3, n3, m0, n0, m1, n1, m2, n2, m3,
                     n3);
  LSX_TRANSPOSE8x8_H(m4, n4, m5, n5, m6, n6, m7, n7, m4, n4, m5, n5, m6, n6, m7,
                     n7);
  __lsx_vst(m0, dst, 32);
  __lsx_vst(n0, dst, 96);
  __lsx_vst(m1, dst, 160);
  __lsx_vst(n1, dst, 224);
  __lsx_vst(m2, dst, 288);
  __lsx_vst(n2, dst, 352);
  __lsx_vst(m3, dst, 416);
  __lsx_vst(n3, dst, 480);
  __lsx_vst(m4, dst, 48);
  __lsx_vst(n4, dst, 112);
  __lsx_vst(m5, dst, 176);
  __lsx_vst(n5, dst, 240);
  __lsx_vst(m6, dst, 304);
  __lsx_vst(n6, dst, 368);
  __lsx_vst(m7, dst, 432);
  __lsx_vst(n7, dst, 496);
}

static void idct32x8_1d_rows_lsx(const int16_t *input, int16_t *output) {
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
  __m128i vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i stp0, stp1, stp2, stp3, stp4, stp5, stp6, stp7;
  __m128i tmp0;

  /* Even stage 1 */
  DUP4_ARG2(__lsx_vld, tmp_buf, 0, tmp_buf, 256, tmp_buf, 512, tmp_buf, 768,
            reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_buf, 1024, tmp_buf, 1280, tmp_buf, 1536, tmp_buf,
            1792, reg4, reg5, reg6, reg7);
  tmp_buf += 64;

  DOTP_CONST_PAIR(reg1, reg7, cospi_28_64, cospi_4_64, reg1, reg7);
  DOTP_CONST_PAIR(reg5, reg3, cospi_12_64, cospi_20_64, reg5, reg3);
  LSX_BUTTERFLY_4_H(reg1, reg7, reg3, reg5, vec1, vec3, vec2, vec0);
  DOTP_CONST_PAIR(vec2, vec0, cospi_16_64, cospi_16_64, loc2, loc3);

  loc1 = vec3;
  loc0 = vec1;

  DOTP_CONST_PAIR(reg0, reg4, cospi_16_64, cospi_16_64, reg0, reg4);
  DOTP_CONST_PAIR(reg2, reg6, cospi_24_64, cospi_8_64, reg2, reg6);
  LSX_BUTTERFLY_4_H(reg4, reg0, reg2, reg6, vec1, vec3, vec2, vec0);
  LSX_BUTTERFLY_4_H(vec0, vec1, loc1, loc0, stp3, stp0, stp7, stp4);
  LSX_BUTTERFLY_4_H(vec2, vec3, loc3, loc2, stp2, stp1, stp6, stp5);

  /* Even stage 2 */
  /* Load 8 */
  DUP4_ARG2(__lsx_vld, tmp_buf, 0, tmp_buf, 256, tmp_buf, 512, tmp_buf, 768,
            reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_buf, 1024, tmp_buf, 1280, tmp_buf, 1536, tmp_buf,
            1792, reg4, reg5, reg6, reg7);
  DOTP_CONST_PAIR(reg0, reg7, cospi_30_64, cospi_2_64, reg0, reg7);
  DOTP_CONST_PAIR(reg4, reg3, cospi_14_64, cospi_18_64, reg4, reg3);
  DOTP_CONST_PAIR(reg2, reg5, cospi_22_64, cospi_10_64, reg2, reg5);
  DOTP_CONST_PAIR(reg6, reg1, cospi_6_64, cospi_26_64, reg6, reg1);

  vec0 = __lsx_vadd_h(reg0, reg4);
  reg0 = __lsx_vsub_h(reg0, reg4);
  reg4 = __lsx_vadd_h(reg6, reg2);
  reg6 = __lsx_vsub_h(reg6, reg2);
  reg2 = __lsx_vadd_h(reg1, reg5);
  reg1 = __lsx_vsub_h(reg1, reg5);
  reg5 = __lsx_vadd_h(reg7, reg3);
  reg7 = __lsx_vsub_h(reg7, reg3);
  reg3 = vec0;

  vec1 = reg2;
  reg2 = __lsx_vadd_h(reg3, reg4);
  reg3 = __lsx_vsub_h(reg3, reg4);
  reg4 = __lsx_vsub_h(reg5, vec1);
  reg5 = __lsx_vadd_h(reg5, vec1);

  tmp0 = __lsx_vneg_h(reg6);
  DOTP_CONST_PAIR(reg7, reg0, cospi_24_64, cospi_8_64, reg0, reg7);
  DOTP_CONST_PAIR(tmp0, reg1, cospi_24_64, cospi_8_64, reg6, reg1);

  vec0 = __lsx_vsub_h(reg0, reg6);
  reg0 = __lsx_vadd_h(reg0, reg6);
  vec1 = __lsx_vsub_h(reg7, reg1);
  reg7 = __lsx_vadd_h(reg7, reg1);

  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, reg6, reg1);
  DOTP_CONST_PAIR(reg4, reg3, cospi_16_64, cospi_16_64, reg3, reg4);

  /* Even stage 3 : Dependency on Even stage 1 & Even stage 2 */
  /* Store 8 */
  LSX_BUTTERFLY_4_H(stp0, stp1, reg7, reg5, loc1, loc3, loc2, loc0);
  __lsx_vst(loc1, tmp_eve_buf, 0);
  __lsx_vst(loc3, tmp_eve_buf, 16);
  __lsx_vst(loc2, tmp_eve_buf, 224);
  __lsx_vst(loc0, tmp_eve_buf, 240);

  LSX_BUTTERFLY_4_H(stp2, stp3, reg4, reg1, loc1, loc3, loc2, loc0);
  __lsx_vst(loc1, tmp_eve_buf, 32);
  __lsx_vst(loc3, tmp_eve_buf, 48);
  __lsx_vst(loc2, tmp_eve_buf, 192);
  __lsx_vst(loc0, tmp_eve_buf, 208);

  /* Store 8 */
  LSX_BUTTERFLY_4_H(stp4, stp5, reg6, reg3, loc1, loc3, loc2, loc0);
  __lsx_vst(loc1, tmp_eve_buf, 64);
  __lsx_vst(loc3, tmp_eve_buf, 80);
  __lsx_vst(loc2, tmp_eve_buf, 160);
  __lsx_vst(loc0, tmp_eve_buf, 176);

  LSX_BUTTERFLY_4_H(stp6, stp7, reg2, reg0, loc1, loc3, loc2, loc0);
  __lsx_vst(loc1, tmp_eve_buf, 96);
  __lsx_vst(loc3, tmp_eve_buf, 112);
  __lsx_vst(loc2, tmp_eve_buf, 128);
  __lsx_vst(loc0, tmp_eve_buf, 144);
}

static void idct8x32_column_odd_process_store(int16_t *tmp_buf,
                                              int16_t *tmp_odd_buf) {
  __m128i vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;

  /* Odd stage 1 */
  DUP4_ARG2(__lsx_vld, tmp_buf, 64, tmp_buf, 448, tmp_buf, 576, tmp_buf, 960,
            reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_buf, 1088, tmp_buf, 1472, tmp_buf, 1600, tmp_buf,
            1984, reg4, reg5, reg6, reg7);

  DOTP_CONST_PAIR(reg0, reg7, cospi_31_64, cospi_1_64, reg0, reg7);
  DOTP_CONST_PAIR(reg4, reg3, cospi_15_64, cospi_17_64, reg3, reg4);
  DOTP_CONST_PAIR(reg2, reg5, cospi_23_64, cospi_9_64, reg2, reg5);
  DOTP_CONST_PAIR(reg6, reg1, cospi_7_64, cospi_25_64, reg1, reg6);

  vec0 = __lsx_vadd_h(reg0, reg3);
  reg0 = __lsx_vsub_h(reg0, reg3);
  reg3 = __lsx_vadd_h(reg7, reg4);
  reg7 = __lsx_vsub_h(reg7, reg4);
  reg4 = __lsx_vadd_h(reg1, reg2);
  reg1 = __lsx_vsub_h(reg1, reg2);
  reg2 = __lsx_vadd_h(reg6, reg5);
  reg6 = __lsx_vsub_h(reg6, reg5);
  reg5 = vec0;

  /* 4 Stores */
  DUP2_ARG2(__lsx_vadd_h, reg5, reg4, reg3, reg2, vec0, vec1);
  __lsx_vst(vec0, tmp_odd_buf, 64);
  __lsx_vst(vec1, tmp_odd_buf, 80);
  DUP2_ARG2(__lsx_vsub_h, reg5, reg4, reg3, reg2, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_24_64, cospi_8_64, vec0, vec1);
  __lsx_vst(vec0, tmp_odd_buf, 0);
  __lsx_vst(vec1, tmp_odd_buf, 16);

  /* 4 Stores */
  DOTP_CONST_PAIR(reg7, reg0, cospi_28_64, cospi_4_64, reg0, reg7);
  DOTP_CONST_PAIR(reg6, reg1, -cospi_4_64, cospi_28_64, reg1, reg6);
  LSX_BUTTERFLY_4_H(reg0, reg7, reg6, reg1, vec0, vec1, vec2, vec3);
  DOTP_CONST_PAIR(vec2, vec3, cospi_24_64, cospi_8_64, vec2, vec3);
  __lsx_vst(vec0, tmp_odd_buf, 96);
  __lsx_vst(vec1, tmp_odd_buf, 112);
  __lsx_vst(vec2, tmp_odd_buf, 32);
  __lsx_vst(vec3, tmp_odd_buf, 48);

  /* Odd stage 2 */
  /* 8 loads */
  DUP4_ARG2(__lsx_vld, tmp_buf, 192, tmp_buf, 320, tmp_buf, 704, tmp_buf, 832,
            reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_buf, 1216, tmp_buf, 1344, tmp_buf, 1728, tmp_buf,
            1856, reg4, reg5, reg6, reg7);
  DOTP_CONST_PAIR(reg1, reg6, cospi_27_64, cospi_5_64, reg1, reg6);
  DOTP_CONST_PAIR(reg5, reg2, cospi_11_64, cospi_21_64, reg2, reg5);
  DOTP_CONST_PAIR(reg3, reg4, cospi_19_64, cospi_13_64, reg3, reg4);
  DOTP_CONST_PAIR(reg7, reg0, cospi_3_64, cospi_29_64, reg0, reg7);

  /* 4 Stores */
  DUP4_ARG2(__lsx_vsub_h, reg1, reg2, reg6, reg5, reg0, reg3, reg7, reg4, vec0,
            vec1, vec2, vec3);
  DOTP_CONST_PAIR(vec1, vec0, cospi_12_64, cospi_20_64, loc0, loc1);
  DOTP_CONST_PAIR(vec3, vec2, -cospi_20_64, cospi_12_64, loc2, loc3);
  LSX_BUTTERFLY_4_H(loc2, loc3, loc1, loc0, vec0, vec1, vec3, vec2);
  __lsx_vst(vec0, tmp_odd_buf, 192);
  __lsx_vst(vec1, tmp_odd_buf, 240);
  DOTP_CONST_PAIR(vec3, vec2, -cospi_8_64, cospi_24_64, vec0, vec1);
  __lsx_vst(vec0, tmp_odd_buf, 160);
  __lsx_vst(vec1, tmp_odd_buf, 176);

  /* 4 Stores */
  DUP4_ARG2(__lsx_vadd_h, reg0, reg3, reg1, reg2, reg5, reg6, reg4, reg7, vec0,
            vec1, vec2, vec3);
  LSX_BUTTERFLY_4_H(vec0, vec3, vec2, vec1, reg0, reg1, reg3, reg2);
  __lsx_vst(reg0, tmp_odd_buf, 208);
  __lsx_vst(reg1, tmp_odd_buf, 224);
  DOTP_CONST_PAIR(reg3, reg2, -cospi_8_64, cospi_24_64, reg0, reg1);
  __lsx_vst(reg0, tmp_odd_buf, 128);
  __lsx_vst(reg1, tmp_odd_buf, 144);

  /* Odd stage 3 : Dependency on Odd stage 1 & Odd stage 2 */
  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 0, tmp_odd_buf, 16, tmp_odd_buf, 32,
            tmp_odd_buf, 48, reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 128, tmp_odd_buf, 144, tmp_odd_buf, 160,
            tmp_odd_buf, 176, reg4, reg5, reg6, reg7);
  DUP4_ARG2(__lsx_vadd_h, reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7, loc0,
            loc1, loc2, loc3);
  __lsx_vst(loc0, tmp_odd_buf, 0);
  __lsx_vst(loc1, tmp_odd_buf, 16);
  __lsx_vst(loc2, tmp_odd_buf, 32);
  __lsx_vst(loc3, tmp_odd_buf, 48);

  DUP2_ARG2(__lsx_vsub_h, reg0, reg4, reg1, reg5, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc0, loc1);
  DUP2_ARG2(__lsx_vsub_h, reg2, reg6, reg3, reg7, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc2, loc3);
  __lsx_vst(loc0, tmp_odd_buf, 128);
  __lsx_vst(loc1, tmp_odd_buf, 144);
  __lsx_vst(loc2, tmp_odd_buf, 160);
  __lsx_vst(loc3, tmp_odd_buf, 176);

  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 64, tmp_odd_buf, 80, tmp_odd_buf, 96,
            tmp_odd_buf, 112, reg1, reg2, reg0, reg3);
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 192, tmp_odd_buf, 208, tmp_odd_buf, 224,
            tmp_odd_buf, 240, reg4, reg5, reg6, reg7);
  DUP4_ARG2(__lsx_vadd_h, reg0, reg4, reg1, reg5, reg2, reg6, reg3, reg7, loc0,
            loc1, loc2, loc3);
  __lsx_vst(loc0, tmp_odd_buf, 64);
  __lsx_vst(loc1, tmp_odd_buf, 80);
  __lsx_vst(loc2, tmp_odd_buf, 96);
  __lsx_vst(loc3, tmp_odd_buf, 112);

  DUP2_ARG2(__lsx_vsub_h, reg0, reg4, reg3, reg7, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc0, loc1);
  DUP2_ARG2(__lsx_vsub_h, reg1, reg5, reg2, reg6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_16_64, cospi_16_64, loc2, loc3);
  __lsx_vst(loc0, tmp_odd_buf, 192);
  __lsx_vst(loc1, tmp_odd_buf, 208);
  __lsx_vst(loc2, tmp_odd_buf, 224);
  __lsx_vst(loc3, tmp_odd_buf, 240);
}

static void idct8x32_column_butterfly_addblk(int16_t *tmp_eve_buf,
                                             int16_t *tmp_odd_buf, uint8_t *dst,
                                             int32_t dst_stride) {
  __m128i vec0, vec1, vec2, vec3, loc0, loc1, loc2, loc3;
  __m128i m0, m1, m2, m3, m4, m5, m6, m7;
  __m128i n0, n1, n2, n3, n4, n5, n6, n7;
  int32_t stride = dst_stride << 2;
  int32_t stride2 = stride << 1;
  int32_t stride3 = stride + stride2;

  /* FINAL BUTTERFLY : Dependency on Even & Odd */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 0, tmp_odd_buf, 144, tmp_odd_buf, 224,
            tmp_odd_buf, 96, vec0, vec1, vec2, vec3);
  DUP4_ARG2(__lsx_vld, tmp_eve_buf, 0, tmp_eve_buf, 128, tmp_eve_buf, 64,
            tmp_eve_buf, 192, loc0, loc1, loc2, loc3);

  DUP4_ARG2(__lsx_vadd_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m0,
            m4, m2, m6);
  DUP4_ARG2(__lsx_vsrari_h, m0, 6, m2, 6, m4, 6, m6, 6, m0, m2, m4, m6);
  VP9_ADDBLK_ST8x4_UB(dst, stride, stride2, stride3, m0, m2, m4, m6);
  DUP4_ARG2(__lsx_vsub_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m6,
            m2, m4, m0);
  DUP4_ARG2(__lsx_vsrari_h, m0, 6, m2, 6, m4, 6, m6, 6, m0, m2, m4, m6);
  VP9_ADDBLK_ST8x4_UB((dst + 19 * dst_stride), stride, stride2, stride3, m0, m2,
                      m4, m6);

  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 64, tmp_odd_buf, 208, tmp_odd_buf, 160,
            tmp_odd_buf, 48, vec0, vec1, vec2, vec3);
  DUP4_ARG2(__lsx_vld, tmp_eve_buf, 32, tmp_eve_buf, 160, tmp_eve_buf, 96,
            tmp_eve_buf, 224, loc0, loc1, loc2, loc3);

  DUP4_ARG2(__lsx_vadd_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m1,
            m5, m3, m7);
  DUP4_ARG2(__lsx_vsrari_h, m1, 6, m3, 6, m5, 6, m7, 6, m1, m3, m5, m7);
  VP9_ADDBLK_ST8x4_UB((dst + 2 * dst_stride), stride, stride2, stride3, m1, m3,
                      m5, m7);
  DUP4_ARG2(__lsx_vsub_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, m7,
            m3, m5, m1);
  DUP4_ARG2(__lsx_vsrari_h, m1, 6, m3, 6, m5, 6, m7, 6, m1, m3, m5, m7);
  VP9_ADDBLK_ST8x4_UB((dst + 17 * dst_stride), stride, stride2, stride3, m1, m3,
                      m5, m7);

  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 32, tmp_odd_buf, 176, tmp_odd_buf, 192,
            tmp_odd_buf, 112, vec0, vec1, vec2, vec3);
  DUP4_ARG2(__lsx_vld, tmp_eve_buf, 16, tmp_eve_buf, 144, tmp_eve_buf, 80,
            tmp_eve_buf, 208, loc0, loc1, loc2, loc3);
  DUP4_ARG2(__lsx_vadd_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n0,
            n4, n2, n6);
  DUP4_ARG2(__lsx_vsrari_h, n0, 6, n2, 6, n4, 6, n6, 6, n0, n2, n4, n6);
  VP9_ADDBLK_ST8x4_UB((dst + dst_stride), stride, stride2, stride3, n0, n2, n4,
                      n6);
  DUP4_ARG2(__lsx_vsub_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n6,
            n2, n4, n0);
  DUP4_ARG2(__lsx_vsrari_h, n0, 6, n2, 6, n4, 6, n6, 6, n0, n2, n4, n6);
  VP9_ADDBLK_ST8x4_UB((dst + 18 * dst_stride), stride, stride2, stride3, n0, n2,
                      n4, n6);

  /* Load 8 & Store 8 */
  DUP4_ARG2(__lsx_vld, tmp_odd_buf, 80, tmp_odd_buf, 240, tmp_odd_buf, 128,
            tmp_odd_buf, 16, vec0, vec1, vec2, vec3);
  DUP4_ARG2(__lsx_vld, tmp_eve_buf, 48, tmp_eve_buf, 176, tmp_eve_buf, 112,
            tmp_eve_buf, 240, loc0, loc1, loc2, loc3);
  DUP4_ARG2(__lsx_vadd_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n1,
            n5, n3, n7);
  DUP4_ARG2(__lsx_vsrari_h, n1, 6, n3, 6, n5, 6, n7, 6, n1, n3, n5, n7);
  VP9_ADDBLK_ST8x4_UB((dst + 3 * dst_stride), stride, stride2, stride3, n1, n3,
                      n5, n7);
  DUP4_ARG2(__lsx_vsub_h, loc0, vec3, loc1, vec2, loc2, vec1, loc3, vec0, n7,
            n3, n5, n1);
  DUP4_ARG2(__lsx_vsrari_h, n1, 6, n3, 6, n5, 6, n7, 6, n1, n3, n5, n7);
  VP9_ADDBLK_ST8x4_UB((dst + 16 * dst_stride), stride, stride2, stride3, n1, n3,
                      n5, n7);
}

static void idct8x32_1d_columns_addblk_lsx(int16_t *input, uint8_t *dst,
                                           int32_t dst_stride) {
  DECLARE_ALIGNED(32, int16_t, tmp_odd_buf[16 * 8]);
  DECLARE_ALIGNED(32, int16_t, tmp_eve_buf[16 * 8]);

  idct8x32_column_even_process_store(input, &tmp_eve_buf[0]);
  idct8x32_column_odd_process_store(input, &tmp_odd_buf[0]);
  idct8x32_column_butterfly_addblk(&tmp_eve_buf[0], &tmp_odd_buf[0], dst,
                                   dst_stride);
}

void vpx_idct32x32_1024_add_lsx(const int16_t *input, uint8_t *dst,
                                int32_t dst_stride) {
  int32_t i;
  DECLARE_ALIGNED(32, int16_t, out_arr[32 * 32]);
  int16_t *out_ptr = out_arr;

  /* transform rows */
  for (i = 0; i < 4; ++i) {
    /* process 32 * 8 block */
    idct32x8_1d_rows_lsx((input + (i << 8)), (out_ptr + (i << 8)));
  }

  for (i = 0; i < 4; ++i) {
    /* process 8 * 32 block */
    idct8x32_1d_columns_addblk_lsx((out_ptr + (i << 3)), (dst + (i << 3)),
                                   dst_stride);
  }
}

void vpx_idct32x32_34_add_lsx(const int16_t *input, uint8_t *dst,
                              int32_t dst_stride) {
  int32_t i;
  DECLARE_ALIGNED(32, int16_t, out_arr[32 * 32]);
  int16_t *out_ptr = out_arr;
  __m128i zero = __lsx_vldi(0);

  for (i = 32; i--;) {
    __lsx_vst(zero, out_ptr, 0);
    __lsx_vst(zero, out_ptr, 16);
    __lsx_vst(zero, out_ptr, 32);
    __lsx_vst(zero, out_ptr, 48);
    out_ptr += 32;
  }

  out_ptr = out_arr;

  /* rows: only upper-left 8x8 has non-zero coeff */
  idct32x8_1d_rows_lsx(input, out_ptr);

  /* transform columns */
  for (i = 0; i < 4; ++i) {
    /* process 8 * 32 block */
    idct8x32_1d_columns_addblk_lsx((out_ptr + (i << 3)), (dst + (i << 3)),
                                   dst_stride);
  }
}

void vpx_idct32x32_1_add_lsx(const int16_t *input, uint8_t *dst,
                             int32_t dst_stride) {
  int32_t i;
  int16_t out;
  __m128i dst0, dst1, dst2, dst3, tmp0, tmp1, tmp2, tmp3;
  __m128i res0, res1, res2, res3, res4, res5, res6, res7, vec;

  out = ROUND_POWER_OF_TWO((input[0] * cospi_16_64), DCT_CONST_BITS);
  out = ROUND_POWER_OF_TWO((out * cospi_16_64), DCT_CONST_BITS);
  out = ROUND_POWER_OF_TWO(out, 6);

  vec = __lsx_vreplgr2vr_h(out);

  for (i = 16; i--;) {
    DUP2_ARG2(__lsx_vld, dst, 0, dst, 16, dst0, dst1);
    dst2 = __lsx_vldx(dst, dst_stride);
    dst3 = __lsx_vldx(dst + 16, dst_stride);

    UNPCK_UB_SH(dst0, res0, res4);
    UNPCK_UB_SH(dst1, res1, res5);
    UNPCK_UB_SH(dst2, res2, res6);
    UNPCK_UB_SH(dst3, res3, res7);

    DUP4_ARG2(__lsx_vadd_h, res0, vec, res1, vec, res2, vec, res3, vec, res0,
              res1, res2, res3);
    DUP4_ARG2(__lsx_vadd_h, res4, vec, res5, vec, res6, vec, res7, vec, res4,
              res5, res6, res7);
    DUP4_ARG3(__lsx_vssrarni_bu_h, res4, res0, 0, res5, res1, 0, res6, res2, 0,
              res7, res3, 0, tmp0, tmp1, tmp2, tmp3);
    __lsx_vst(tmp0, dst, 0);
    __lsx_vst(tmp1, dst, 16);
    dst += dst_stride;
    __lsx_vst(tmp2, dst, 0);
    __lsx_vst(tmp3, dst, 16);
    dst += dst_stride;
  }
}
