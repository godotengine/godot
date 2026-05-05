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
#include "vpx_dsp/fwd_txfm.h"

#define UNPCK_SH_SW(in, out0, out1)  \
  do {                               \
    out0 = __lsx_vsllwil_w_h(in, 0); \
    out1 = __lsx_vexth_w_h(in);      \
  } while (0)

static void fdct8x32_1d_column_load_butterfly(const int16_t *input,
                                              int32_t src_stride,
                                              int16_t *temp_buff) {
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i step0, step1, step2, step3;
  __m128i in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1;
  __m128i step0_1, step1_1, step2_1, step3_1;

  int32_t stride = src_stride << 1;
  int32_t stride2 = stride << 1;
  int32_t stride3 = stride2 + stride;
  const int16_t *input_tmp = (int16_t *)input;

  in0 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, stride, input_tmp, stride2, in1, in2);
  in3 = __lsx_vldx(input_tmp, stride3);

  input_tmp += stride2;
  in0_1 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, stride, input_tmp, stride2, in1_1, in2_1);
  in3_1 = __lsx_vldx(input_tmp, stride3);

  input_tmp = input + (src_stride * 24);
  in4_1 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, stride, input_tmp, stride2, in5_1, in6_1);
  in7_1 = __lsx_vldx(input_tmp, stride3);

  input_tmp += stride2;
  in4 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, stride, input_tmp, stride2, in5, in6);
  in7 = __lsx_vldx(input_tmp, stride3);

  DUP4_ARG2(__lsx_vslli_h, in0, 2, in1, 2, in2, 2, in3, 2, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vslli_h, in4, 2, in5, 2, in6, 2, in7, 2, in4, in5, in6, in7);
  DUP4_ARG2(__lsx_vslli_h, in0_1, 2, in1_1, 2, in2_1, 2, in3_1, 2, in0_1, in1_1,
            in2_1, in3_1);
  DUP4_ARG2(__lsx_vslli_h, in4_1, 2, in5_1, 2, in6_1, 2, in7_1, 2, in4_1, in5_1,
            in6_1, in7_1);
  LSX_BUTTERFLY_8_H(in0, in1, in2, in3, in4, in5, in6, in7, step0, step1, step2,
                    step3, in4, in5, in6, in7);
  LSX_BUTTERFLY_8_H(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1,
                    step0_1, step1_1, step2_1, step3_1, in4_1, in5_1, in6_1,
                    in7_1);

  __lsx_vst(step0, temp_buff, 0);
  __lsx_vst(step1, temp_buff, 16);
  __lsx_vst(step2, temp_buff, 32);
  __lsx_vst(step3, temp_buff, 48);

  __lsx_vst(in4, temp_buff, 448);
  __lsx_vst(in5, temp_buff, 464);
  __lsx_vst(in6, temp_buff, 480);
  __lsx_vst(in7, temp_buff, 496);

  __lsx_vst(step0_1, temp_buff, 64);
  __lsx_vst(step1_1, temp_buff, 80);
  __lsx_vst(step2_1, temp_buff, 96);
  __lsx_vst(step3_1, temp_buff, 112);

  __lsx_vst(in4_1, temp_buff, 384);
  __lsx_vst(in5_1, temp_buff, 400);
  __lsx_vst(in6_1, temp_buff, 416);
  __lsx_vst(in7_1, temp_buff, 432);

  /* 3rd and 4th set */
  input_tmp = input + (src_stride * 8);
  in0 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, stride, input_tmp, stride2, in1, in2);
  in3 = __lsx_vldx(input_tmp, stride3);

  input_tmp += stride2;
  in0_1 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, stride, input_tmp, stride2, in1_1, in2_1);
  in3_1 = __lsx_vldx(input_tmp, stride3);

  input_tmp += stride2;
  in4_1 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, stride, input_tmp, stride2, in5_1, in6_1);
  in7_1 = __lsx_vldx(input_tmp, stride3);

  input_tmp += stride2;
  in4 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, stride, input_tmp, stride2, in5, in6);
  in7 = __lsx_vldx(input_tmp, stride3);
  DUP4_ARG2(__lsx_vslli_h, in0, 2, in1, 2, in2, 2, in3, 2, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vslli_h, in4, 2, in5, 2, in6, 2, in7, 2, in4, in5, in6, in7);
  DUP4_ARG2(__lsx_vslli_h, in0_1, 2, in1_1, 2, in2_1, 2, in3_1, 2, in0_1, in1_1,
            in2_1, in3_1);
  DUP4_ARG2(__lsx_vslli_h, in4_1, 2, in5_1, 2, in6_1, 2, in7_1, 2, in4_1, in5_1,
            in6_1, in7_1);

  LSX_BUTTERFLY_8_H(in0, in1, in2, in3, in4, in5, in6, in7, step0, step1, step2,
                    step3, in4, in5, in6, in7);
  LSX_BUTTERFLY_8_H(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1,
                    step0_1, step1_1, step2_1, step3_1, in4_1, in5_1, in6_1,
                    in7_1);

  __lsx_vst(step0, temp_buff, 128);
  __lsx_vst(step1, temp_buff, 144);
  __lsx_vst(step2, temp_buff, 160);
  __lsx_vst(step3, temp_buff, 176);

  __lsx_vst(in4, temp_buff, 320);
  __lsx_vst(in5, temp_buff, 336);
  __lsx_vst(in6, temp_buff, 352);
  __lsx_vst(in7, temp_buff, 368);

  __lsx_vst(step0_1, temp_buff, 192);
  __lsx_vst(step1_1, temp_buff, 208);
  __lsx_vst(step2_1, temp_buff, 224);
  __lsx_vst(step3_1, temp_buff, 240);

  __lsx_vst(in4_1, temp_buff, 256);
  __lsx_vst(in5_1, temp_buff, 272);
  __lsx_vst(in6_1, temp_buff, 288);
  __lsx_vst(in7_1, temp_buff, 304);
}

static void fdct8x32_1d_column_even_store(int16_t *input, int16_t *temp) {
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i in8, in9, in10, in11, in12, in13, in14, in15;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  __m128i temp0, temp1;

  /* fdct even */
  DUP4_ARG2(__lsx_vld, input, 0, input, 16, input, 32, input, 48, in0, in1, in2,
            in3);
  DUP4_ARG2(__lsx_vld, input, 192, input, 208, input, 224, input, 240, in12,
            in13, in14, in15);
  LSX_BUTTERFLY_8_H(in0, in1, in2, in3, in12, in13, in14, in15, vec0, vec1,
                    vec2, vec3, in12, in13, in14, in15);
  DUP4_ARG2(__lsx_vld, input, 64, input, 80, input, 96, input, 112, in4, in5,
            in6, in7);
  DUP4_ARG2(__lsx_vld, input, 128, input, 144, input, 160, input, 176, in8, in9,
            in10, in11);
  LSX_BUTTERFLY_8_H(in4, in5, in6, in7, in8, in9, in10, in11, vec4, vec5, vec6,
                    vec7, in8, in9, in10, in11);

  /* Stage 3 */
  DUP4_ARG2(__lsx_vadd_h, vec0, vec7, vec1, vec6, vec2, vec5, vec3, vec4, in0,
            in1, in2, in3);
  LSX_BUTTERFLY_4_H(in0, in1, in2, in3, temp0, in4, in1, in0);
  DOTP_CONST_PAIR(temp0, in4, cospi_16_64, cospi_16_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  __lsx_vst(temp0, temp, 0);
  __lsx_vst(temp1, temp, 1024);

  DOTP_CONST_PAIR(in0, in1, cospi_24_64, cospi_8_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  __lsx_vst(temp0, temp, 512);
  __lsx_vst(temp1, temp, 1536);

  DUP4_ARG2(__lsx_vsub_h, vec0, vec7, vec1, vec6, vec2, vec5, vec3, vec4, vec7,
            vec6, vec5, vec4);
  DOTP_CONST_PAIR(vec6, vec5, cospi_16_64, cospi_16_64, vec5, vec6);
  DUP2_ARG2(__lsx_vadd_h, vec4, vec5, vec7, vec6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_28_64, cospi_4_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  __lsx_vst(temp0, temp, 256);
  __lsx_vst(temp1, temp, 1792);

  DUP2_ARG2(__lsx_vsub_h, vec4, vec5, vec7, vec6, vec4, vec7);
  DOTP_CONST_PAIR(vec7, vec4, cospi_12_64, cospi_20_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  __lsx_vst(temp0, temp, 1280);
  __lsx_vst(temp1, temp, 768);

  DOTP_CONST_PAIR(in13, in10, cospi_16_64, cospi_16_64, vec2, vec5);
  DOTP_CONST_PAIR(in12, in11, cospi_16_64, cospi_16_64, vec3, vec4);
  DUP4_ARG2(__lsx_vadd_h, in8, vec3, in9, vec2, in14, vec5, in15, vec4, in0,
            vec1, vec6, in2);
  DOTP_CONST_PAIR(vec6, vec1, cospi_24_64, cospi_8_64, in1, in3);
  DUP2_ARG2(__lsx_vadd_h, in0, in1, in2, in3, vec0, vec7);
  DOTP_CONST_PAIR(vec7, vec0, cospi_30_64, cospi_2_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  __lsx_vst(temp0, temp, 128);
  __lsx_vst(temp1, temp, 1920);

  DUP2_ARG2(__lsx_vsub_h, in0, in1, in2, in3, in0, in2);
  DOTP_CONST_PAIR(in2, in0, cospi_14_64, cospi_18_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  __lsx_vst(temp0, temp, 1152);
  __lsx_vst(temp1, temp, 896);

  DUP2_ARG2(__lsx_vsub_h, in9, vec2, in14, vec5, vec2, vec5);
  temp0 = __lsx_vneg_h(vec2);
  DOTP_CONST_PAIR(temp0, vec5, cospi_24_64, cospi_8_64, in2, in1);
  DUP4_ARG2(__lsx_vsub_h, in8, vec3, in15, vec4, in3, in2, in0, in1, in3, in0,
            vec2, vec5);
  DOTP_CONST_PAIR(vec5, vec2, cospi_22_64, cospi_10_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  __lsx_vst(temp0, temp, 640);
  __lsx_vst(temp1, temp, 1408);

  DUP2_ARG2(__lsx_vadd_h, in3, in2, in0, in1, vec3, vec4);
  DOTP_CONST_PAIR(vec4, vec3, cospi_6_64, cospi_26_64, temp0, temp1);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  __lsx_vst(temp0, temp, 384);
  __lsx_vst(temp1, temp, 1664);
}

static void fdct8x32_1d_column_odd_store(int16_t *input, int16_t *temp_ptr) {
  __m128i in16, in17, in18, in19, in20, in21, in22, in23;
  __m128i in24, in25, in26, in27, in28, in29, in30, in31, vec4, vec5;
  __m128i tmp0, tmp1;

  DUP4_ARG2(__lsx_vld, input, 64, input, 80, input, 160, input, 176, in20, in21,
            in26, in27);

  DOTP_CONST_PAIR(in27, in20, cospi_16_64, cospi_16_64, in20, in27);
  DOTP_CONST_PAIR(in26, in21, cospi_16_64, cospi_16_64, in21, in26);

  DUP4_ARG2(__lsx_vld, input, 32, input, 48, input, 192, input, 208, in18, in19,
            in28, in29);

  vec4 = __lsx_vsub_h(in19, in20);
  __lsx_vst(vec4, input, 64);
  vec4 = __lsx_vsub_h(in18, in21);
  __lsx_vst(vec4, input, 80);
  vec4 = __lsx_vsub_h(in29, in26);
  __lsx_vst(vec4, input, 160);
  vec4 = __lsx_vsub_h(in28, in27);
  __lsx_vst(vec4, input, 176);

  in21 = __lsx_vadd_h(in18, in21);
  in20 = __lsx_vadd_h(in19, in20);
  in27 = __lsx_vadd_h(in28, in27);
  in26 = __lsx_vadd_h(in29, in26);

  DUP4_ARG2(__lsx_vld, input, 96, input, 112, input, 128, input, 144, in22,
            in23, in24, in25);
  DOTP_CONST_PAIR(in25, in22, cospi_16_64, cospi_16_64, in22, in25);
  DOTP_CONST_PAIR(in24, in23, cospi_16_64, cospi_16_64, in23, in24);

  DUP4_ARG2(__lsx_vld, input, 0, input, 16, input, 224, input, 240, in16, in17,
            in30, in31);

  vec4 = __lsx_vsub_h(in17, in22);
  __lsx_vst(vec4, input, 32);
  vec4 = __lsx_vsub_h(in16, in23);
  __lsx_vst(vec4, input, 48);
  vec4 = __lsx_vsub_h(in31, in24);
  __lsx_vst(vec4, input, 192);
  vec4 = __lsx_vsub_h(in30, in25);
  __lsx_vst(vec4, input, 208);

  DUP4_ARG2(__lsx_vadd_h, in16, in23, in17, in22, in30, in25, in31, in24, in16,
            in17, in30, in31);
  DOTP_CONST_PAIR(in26, in21, cospi_24_64, cospi_8_64, in18, in29);
  DOTP_CONST_PAIR(in27, in20, cospi_24_64, cospi_8_64, in19, in28);
  DUP4_ARG2(__lsx_vadd_h, in16, in19, in17, in18, in30, in29, in31, in28, in27,
            in22, in21, in25);
  DOTP_CONST_PAIR(in21, in22, cospi_28_64, cospi_4_64, in26, in24);
  DUP2_ARG2(__lsx_vadd_h, in27, in26, in25, in24, in23, in20);
  DOTP_CONST_PAIR(in20, in23, cospi_31_64, cospi_1_64, vec4, vec5);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  __lsx_vst(vec5, temp_ptr, 0);
  __lsx_vst(vec4, temp_ptr, 1920);

  DUP2_ARG2(__lsx_vsub_h, in27, in26, in25, in24, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_15_64, cospi_17_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  __lsx_vst(vec5, temp_ptr, 896);
  __lsx_vst(vec4, temp_ptr, 1024);

  DUP4_ARG2(__lsx_vsub_h, in17, in18, in16, in19, in31, in28, in30, in29, in23,
            in26, in24, in20);
  tmp0 = __lsx_vneg_h(in23);
  DOTP_CONST_PAIR(tmp0, in20, cospi_28_64, cospi_4_64, in27, in25);
  DUP2_ARG2(__lsx_vsub_h, in26, in27, in24, in25, in23, in20);
  DOTP_CONST_PAIR(in20, in23, cospi_23_64, cospi_9_64, vec4, vec5);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  __lsx_vst(vec4, temp_ptr, 1408);
  __lsx_vst(vec5, temp_ptr, 512);

  DUP2_ARG2(__lsx_vadd_h, in26, in27, in24, in25, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_7_64, cospi_25_64, vec4, vec5);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  __lsx_vst(vec4, temp_ptr, 384);
  __lsx_vst(vec5, temp_ptr, 1536);

  DUP4_ARG2(__lsx_vld, input, 32, input, 48, input, 64, input, 80, in22, in23,
            in20, in21);
  DUP4_ARG2(__lsx_vld, input, 160, input, 176, input, 192, input, 208, in26,
            in27, in24, in25);
  in16 = in20;
  in17 = in21;
  DUP2_ARG1(__lsx_vneg_h, in16, in17, tmp0, tmp1);
  DOTP_CONST_PAIR(tmp0, in27, cospi_24_64, cospi_8_64, in20, in27);
  DOTP_CONST_PAIR(tmp1, in26, cospi_24_64, cospi_8_64, in21, in26);
  DUP4_ARG2(__lsx_vsub_h, in23, in20, in22, in21, in25, in26, in24, in27, in28,
            in17, in18, in31);
  DOTP_CONST_PAIR(in18, in17, cospi_12_64, cospi_20_64, in29, in30);
  DUP2_ARG2(__lsx_vadd_h, in28, in29, in31, in30, in16, in19);
  DOTP_CONST_PAIR(in19, in16, cospi_27_64, cospi_5_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  __lsx_vst(vec5, temp_ptr, 1664);
  __lsx_vst(vec4, temp_ptr, 256);

  DUP2_ARG2(__lsx_vsub_h, in28, in29, in31, in30, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_11_64, cospi_21_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  __lsx_vst(vec5, temp_ptr, 640);
  __lsx_vst(vec4, temp_ptr, 1280);

  DUP4_ARG2(__lsx_vadd_h, in22, in21, in23, in20, in24, in27, in25, in26, in16,
            in29, in30, in19);
  tmp0 = __lsx_vneg_h(in16);
  DOTP_CONST_PAIR(tmp0, in19, cospi_12_64, cospi_20_64, in28, in31);
  DUP2_ARG2(__lsx_vsub_h, in29, in28, in30, in31, in16, in19);
  DOTP_CONST_PAIR(in19, in16, cospi_19_64, cospi_13_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  __lsx_vst(vec5, temp_ptr, 1152);
  __lsx_vst(vec4, temp_ptr, 768);

  DUP2_ARG2(__lsx_vadd_h, in29, in28, in30, in31, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_3_64, cospi_29_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  __lsx_vst(vec5, temp_ptr, 128);
  __lsx_vst(vec4, temp_ptr, 1792);
}

static void fdct8x32_1d_column(const int16_t *input, int32_t src_stride,
                               int16_t *tmp_buf, int16_t *tmp_buf_big) {
  fdct8x32_1d_column_load_butterfly(input, src_stride, tmp_buf);
  fdct8x32_1d_column_even_store(tmp_buf, tmp_buf_big);
  fdct8x32_1d_column_odd_store(tmp_buf + 128, (tmp_buf_big + 32));
}

static void fdct8x32_1d_row_load_butterfly(int16_t *temp_buff,
                                           int16_t *output) {
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i in8, in9, in10, in11, in12, in13, in14, in15;
  __m128i step0, step1, step2, step3, step4, step5, step6, step7;

  DUP4_ARG2(__lsx_vld, temp_buff, 0, temp_buff, 64, temp_buff, 128, temp_buff,
            192, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vld, temp_buff, 256, temp_buff, 320, temp_buff, 384,
            temp_buff, 448, in4, in5, in6, in7);
  DUP4_ARG2(__lsx_vld, temp_buff, 48, temp_buff, 112, temp_buff, 176, temp_buff,
            240, in8, in9, in10, in11);
  DUP4_ARG2(__lsx_vld, temp_buff, 304, temp_buff, 368, temp_buff, 432,
            temp_buff, 496, in12, in13, in14, in15);
  LSX_TRANSPOSE8x8_H(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  LSX_TRANSPOSE8x8_H(in8, in9, in10, in11, in12, in13, in14, in15, in8, in9,
                     in10, in11, in12, in13, in14, in15);
  LSX_BUTTERFLY_16_H(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                     in11, in12, in13, in14, in15, step0, step1, step2, step3,
                     step4, step5, step6, step7, in8, in9, in10, in11, in12,
                     in13, in14, in15);

  __lsx_vst(step0, output, 0);
  __lsx_vst(step1, output, 16);
  __lsx_vst(step2, output, 32);
  __lsx_vst(step3, output, 48);
  __lsx_vst(step4, output, 64);
  __lsx_vst(step5, output, 80);
  __lsx_vst(step6, output, 96);
  __lsx_vst(step7, output, 112);

  __lsx_vst(in8, output, 384);
  __lsx_vst(in9, output, 400);
  __lsx_vst(in10, output, 416);
  __lsx_vst(in11, output, 432);
  __lsx_vst(in12, output, 448);
  __lsx_vst(in13, output, 464);
  __lsx_vst(in14, output, 480);
  __lsx_vst(in15, output, 496);

  /* 2nd set */
  DUP4_ARG2(__lsx_vld, temp_buff, 16, temp_buff, 80, temp_buff, 144, temp_buff,
            208, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vld, temp_buff, 272, temp_buff, 336, temp_buff, 400,
            temp_buff, 464, in4, in5, in6, in7);
  DUP4_ARG2(__lsx_vld, temp_buff, 32, temp_buff, 96, temp_buff, 160, temp_buff,
            224, in8, in9, in10, in11);
  DUP4_ARG2(__lsx_vld, temp_buff, 288, temp_buff, 352, temp_buff, 416,
            temp_buff, 480, in12, in13, in14, in15);
  LSX_TRANSPOSE8x8_H(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  LSX_TRANSPOSE8x8_H(in8, in9, in10, in11, in12, in13, in14, in15, in8, in9,
                     in10, in11, in12, in13, in14, in15);
  LSX_BUTTERFLY_16_H(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                     in11, in12, in13, in14, in15, step0, step1, step2, step3,
                     step4, step5, step6, step7, in8, in9, in10, in11, in12,
                     in13, in14, in15);

  __lsx_vst(step0, output, 128);
  __lsx_vst(step1, output, 144);
  __lsx_vst(step2, output, 160);
  __lsx_vst(step3, output, 176);
  __lsx_vst(step4, output, 192);
  __lsx_vst(step5, output, 208);
  __lsx_vst(step6, output, 224);
  __lsx_vst(step7, output, 240);

  __lsx_vst(in8, output, 256);
  __lsx_vst(in9, output, 272);
  __lsx_vst(in10, output, 288);
  __lsx_vst(in11, output, 304);
  __lsx_vst(in12, output, 320);
  __lsx_vst(in13, output, 336);
  __lsx_vst(in14, output, 352);
  __lsx_vst(in15, output, 368);
}

static void fdct8x32_1d_row_even_4x(int16_t *input, int16_t *interm_ptr,
                                    int16_t *out) {
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i in8, in9, in10, in11, in12, in13, in14, in15;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  __m128i vec0_l, vec1_l, vec2_l, vec3_l, vec4_l, vec5_l, vec6_l, vec7_l;
  __m128i vec0_r, vec1_r, vec2_r, vec3_r, vec4_r, vec5_r, vec6_r, vec7_r;
  __m128i tmp0_w, tmp1_w, tmp2_w, tmp3_w;

  /* fdct32 even */
  /* stage 2 */
  DUP4_ARG2(__lsx_vld, input, 0, input, 16, input, 32, input, 48, in0, in1, in2,
            in3);
  DUP4_ARG2(__lsx_vld, input, 64, input, 80, input, 96, input, 112, in4, in5,
            in6, in7);
  DUP4_ARG2(__lsx_vld, input, 128, input, 144, input, 160, input, 176, in8, in9,
            in10, in11);
  DUP4_ARG2(__lsx_vld, input, 192, input, 208, input, 224, input, 240, in12,
            in13, in14, in15);

  LSX_BUTTERFLY_16_H(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                     in11, in12, in13, in14, in15, vec0, vec1, vec2, vec3, vec4,
                     vec5, vec6, vec7, in8, in9, in10, in11, in12, in13, in14,
                     in15);

  __lsx_vst(vec0, interm_ptr, 0);
  __lsx_vst(vec1, interm_ptr, 16);
  __lsx_vst(vec2, interm_ptr, 32);
  __lsx_vst(vec3, interm_ptr, 48);
  __lsx_vst(vec4, interm_ptr, 64);
  __lsx_vst(vec5, interm_ptr, 80);
  __lsx_vst(vec6, interm_ptr, 96);
  __lsx_vst(vec7, interm_ptr, 112);

  __lsx_vst(in8, interm_ptr, 128);
  __lsx_vst(in9, interm_ptr, 144);
  __lsx_vst(in10, interm_ptr, 160);
  __lsx_vst(in11, interm_ptr, 176);
  __lsx_vst(in12, interm_ptr, 192);
  __lsx_vst(in13, interm_ptr, 208);
  __lsx_vst(in14, interm_ptr, 224);
  __lsx_vst(in15, interm_ptr, 240);

  /* Stage 3 */
  UNPCK_SH_SW(vec0, vec0_l, vec0_r);
  UNPCK_SH_SW(vec1, vec1_l, vec1_r);
  UNPCK_SH_SW(vec2, vec2_l, vec2_r);
  UNPCK_SH_SW(vec3, vec3_l, vec3_r);
  UNPCK_SH_SW(vec4, vec4_l, vec4_r);
  UNPCK_SH_SW(vec5, vec5_l, vec5_r);
  UNPCK_SH_SW(vec6, vec6_l, vec6_r);
  UNPCK_SH_SW(vec7, vec7_l, vec7_r);
  DUP4_ARG2(__lsx_vadd_w, vec0_r, vec7_r, vec1_r, vec6_r, vec2_r, vec5_r,
            vec3_r, vec4_r, tmp0_w, tmp1_w, tmp2_w, tmp3_w);
  LSX_BUTTERFLY_4_W(tmp0_w, tmp1_w, tmp2_w, tmp3_w, vec4_r, vec6_r, vec7_r,
                    vec5_r);
  DUP4_ARG2(__lsx_vadd_w, vec0_l, vec7_l, vec1_l, vec6_l, vec2_l, vec5_l,
            vec3_l, vec4_l, vec0_r, vec1_r, vec2_r, vec3_r);

  tmp3_w = __lsx_vadd_w(vec0_r, vec3_r);
  vec0_r = __lsx_vsub_w(vec0_r, vec3_r);
  vec3_r = __lsx_vadd_w(vec1_r, vec2_r);
  vec1_r = __lsx_vsub_w(vec1_r, vec2_r);

  DOTP_CONST_PAIR_W(vec4_r, vec6_r, tmp3_w, vec3_r, cospi_16_64, cospi_16_64,
                    vec4_r, tmp3_w, vec6_r, vec3_r);
  FDCT32_POSTPROC_NEG_W(vec4_r);
  FDCT32_POSTPROC_NEG_W(tmp3_w);
  FDCT32_POSTPROC_NEG_W(vec6_r);
  FDCT32_POSTPROC_NEG_W(vec3_r);
  DUP2_ARG2(__lsx_vpickev_h, vec4_r, tmp3_w, vec6_r, vec3_r, vec4, vec5);
  __lsx_vst(vec5, out, 0);
  __lsx_vst(vec4, out, 16);

  DOTP_CONST_PAIR_W(vec5_r, vec7_r, vec0_r, vec1_r, cospi_24_64, cospi_8_64,
                    vec4_r, tmp3_w, vec6_r, vec3_r);
  FDCT32_POSTPROC_NEG_W(vec4_r);
  FDCT32_POSTPROC_NEG_W(tmp3_w);
  FDCT32_POSTPROC_NEG_W(vec6_r);
  FDCT32_POSTPROC_NEG_W(vec3_r);
  DUP2_ARG2(__lsx_vpickev_h, vec4_r, tmp3_w, vec6_r, vec3_r, vec4, vec5);
  __lsx_vst(vec5, out, 32);
  __lsx_vst(vec4, out, 48);

  DUP4_ARG2(__lsx_vld, interm_ptr, 0, interm_ptr, 16, interm_ptr, 32,
            interm_ptr, 48, vec0, vec1, vec2, vec3);
  DUP4_ARG2(__lsx_vld, interm_ptr, 64, interm_ptr, 80, interm_ptr, 96,
            interm_ptr, 112, vec4, vec5, vec6, vec7);
  DUP4_ARG2(__lsx_vsub_h, vec3, vec4, vec2, vec5, vec1, vec6, vec0, vec7, vec4,
            vec5, vec6, vec7);
  DOTP_CONST_PAIR(vec6, vec5, cospi_16_64, cospi_16_64, vec5, vec6);
  DUP2_ARG2(__lsx_vadd_h, vec4, vec5, vec7, vec6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_28_64, cospi_4_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  __lsx_vst(in4, out, 64);
  __lsx_vst(in5, out, 112);

  DUP2_ARG2(__lsx_vsub_h, vec4, vec5, vec7, vec6, vec4, vec7);
  DOTP_CONST_PAIR(vec7, vec4, cospi_12_64, cospi_20_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  __lsx_vst(in4, out, 80);
  __lsx_vst(in5, out, 96);

  DUP4_ARG2(__lsx_vld, interm_ptr, 128, interm_ptr, 144, interm_ptr, 160,
            interm_ptr, 176, in8, in9, in10, in11);
  DUP4_ARG2(__lsx_vld, interm_ptr, 192, interm_ptr, 208, interm_ptr, 224,
            interm_ptr, 240, in12, in13, in14, in15);
  DOTP_CONST_PAIR(in13, in10, cospi_16_64, cospi_16_64, vec2, vec5);
  DOTP_CONST_PAIR(in12, in11, cospi_16_64, cospi_16_64, vec3, vec4);
  DUP4_ARG2(__lsx_vadd_h, in8, vec3, in9, vec2, in14, vec5, in15, vec4, in0,
            vec1, vec6, in2);
  DOTP_CONST_PAIR(vec6, vec1, cospi_24_64, cospi_8_64, in1, in3);
  DUP2_ARG2(__lsx_vadd_h, in0, in1, in2, in3, vec0, vec7);
  DOTP_CONST_PAIR(vec7, vec0, cospi_30_64, cospi_2_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  __lsx_vst(in4, out, 128);
  __lsx_vst(in5, out, 240);

  DUP2_ARG2(__lsx_vsub_h, in0, in1, in2, in3, in0, in2);
  DOTP_CONST_PAIR(in2, in0, cospi_14_64, cospi_18_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  __lsx_vst(in4, out, 144);
  __lsx_vst(in5, out, 224);

  DUP2_ARG2(__lsx_vsub_h, in9, vec2, in14, vec5, vec2, vec5);
  tmp0_w = __lsx_vneg_h(vec2);
  DOTP_CONST_PAIR(tmp0_w, vec5, cospi_24_64, cospi_8_64, in2, in1);
  DUP4_ARG2(__lsx_vsub_h, in8, vec3, in15, vec4, in3, in2, in0, in1, in3, in0,
            vec2, vec5);
  DOTP_CONST_PAIR(vec5, vec2, cospi_22_64, cospi_10_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  __lsx_vst(in4, out, 160);
  __lsx_vst(in5, out, 208);

  DUP2_ARG2(__lsx_vadd_h, in3, in2, in0, in1, vec3, vec4);
  DOTP_CONST_PAIR(vec4, vec3, cospi_6_64, cospi_26_64, in4, in5);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  __lsx_vst(in4, out, 192);
  __lsx_vst(in5, out, 176);
}

static void fdct8x32_1d_row_even(int16_t *temp, int16_t *out) {
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i in8, in9, in10, in11, in12, in13, in14, in15;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, temp0, temp1;

  /* fdct32 even */
  /* stage 2 */
  DUP4_ARG2(__lsx_vld, temp, 0, temp, 16, temp, 32, temp, 48, in0, in1, in2,
            in3);
  DUP4_ARG2(__lsx_vld, temp, 64, temp, 80, temp, 96, temp, 112, in4, in5, in6,
            in7);
  DUP4_ARG2(__lsx_vld, temp, 128, temp, 144, temp, 160, temp, 176, in8, in9,
            in10, in11);
  DUP4_ARG2(__lsx_vld, temp, 192, temp, 208, temp, 224, temp, 240, in12, in13,
            in14, in15);

  LSX_BUTTERFLY_16_H(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                     in11, in12, in13, in14, in15, vec0, vec1, vec2, vec3, vec4,
                     vec5, vec6, vec7, in8, in9, in10, in11, in12, in13, in14,
                     in15);
  /* Stage 3 */
  DUP4_ARG2(__lsx_vadd_h, vec0, vec7, vec1, vec6, vec2, vec5, vec3, vec4, in0,
            in1, in2, in3);
  LSX_BUTTERFLY_4_H(in0, in1, in2, in3, temp0, in4, in1, in0);
  DOTP_CONST_PAIR(temp0, in4, cospi_16_64, cospi_16_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  __lsx_vst(temp0, out, 0);
  __lsx_vst(temp1, out, 16);

  DOTP_CONST_PAIR(in0, in1, cospi_24_64, cospi_8_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  __lsx_vst(temp0, out, 32);
  __lsx_vst(temp1, out, 48);

  DUP4_ARG2(__lsx_vsub_h, vec3, vec4, vec2, vec5, vec1, vec6, vec0, vec7, vec4,
            vec5, vec6, vec7);
  DOTP_CONST_PAIR(vec6, vec5, cospi_16_64, cospi_16_64, vec5, vec6);
  DUP2_ARG2(__lsx_vadd_h, vec4, vec5, vec7, vec6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_28_64, cospi_4_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  __lsx_vst(temp0, out, 64);
  __lsx_vst(temp1, out, 112);

  DUP2_ARG2(__lsx_vsub_h, vec4, vec5, vec7, vec6, vec4, vec7);
  DOTP_CONST_PAIR(vec7, vec4, cospi_12_64, cospi_20_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  __lsx_vst(temp0, out, 80);
  __lsx_vst(temp1, out, 96);

  DOTP_CONST_PAIR(in13, in10, cospi_16_64, cospi_16_64, vec2, vec5);
  DOTP_CONST_PAIR(in12, in11, cospi_16_64, cospi_16_64, vec3, vec4);
  DUP4_ARG2(__lsx_vadd_h, in8, vec3, in9, vec2, in14, vec5, in15, vec4, in0,
            vec1, vec6, in2);
  DOTP_CONST_PAIR(vec6, vec1, cospi_24_64, cospi_8_64, in1, in3);
  DUP2_ARG2(__lsx_vadd_h, in0, in1, in2, in3, vec0, vec7);
  DOTP_CONST_PAIR(vec7, vec0, cospi_30_64, cospi_2_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  __lsx_vst(temp0, out, 128);
  __lsx_vst(temp1, out, 240);

  DUP2_ARG2(__lsx_vsub_h, in0, in1, in2, in3, in0, in2);
  DOTP_CONST_PAIR(in2, in0, cospi_14_64, cospi_18_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  __lsx_vst(temp0, out, 144);
  __lsx_vst(temp1, out, 224);

  DUP2_ARG2(__lsx_vsub_h, in9, vec2, in14, vec5, vec2, vec5);
  temp0 = __lsx_vneg_h(vec2);
  DOTP_CONST_PAIR(temp0, vec5, cospi_24_64, cospi_8_64, in2, in1);
  DUP4_ARG2(__lsx_vsub_h, in8, vec3, in15, vec4, in3, in2, in0, in1, in3, in0,
            vec2, vec5)
  DOTP_CONST_PAIR(vec5, vec2, cospi_22_64, cospi_10_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  __lsx_vst(temp0, out, 160);
  __lsx_vst(temp1, out, 208);

  DUP2_ARG2(__lsx_vadd_h, in3, in2, in0, in1, vec3, vec4);
  DOTP_CONST_PAIR(vec4, vec3, cospi_6_64, cospi_26_64, temp0, temp1);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  __lsx_vst(temp0, out, 192);
  __lsx_vst(temp1, out, 176);
}

static void fdct8x32_1d_row_odd(int16_t *temp, int16_t *interm_ptr,
                                int16_t *out) {
  __m128i in16, in17, in18, in19, in20, in21, in22, in23;
  __m128i in24, in25, in26, in27, in28, in29, in30, in31, vec4, vec5;
  __m128i tmp0, tmp1;

  in20 = __lsx_vld(temp, 64);
  in21 = __lsx_vld(temp, 80);
  in26 = __lsx_vld(temp, 160);
  in27 = __lsx_vld(temp, 176);

  DOTP_CONST_PAIR(in27, in20, cospi_16_64, cospi_16_64, in20, in27);
  DOTP_CONST_PAIR(in26, in21, cospi_16_64, cospi_16_64, in21, in26);

  in18 = __lsx_vld(temp, 32);
  in19 = __lsx_vld(temp, 48);
  in28 = __lsx_vld(temp, 192);
  in29 = __lsx_vld(temp, 208);

  vec4 = __lsx_vsub_h(in19, in20);
  __lsx_vst(vec4, interm_ptr, 64);
  vec4 = __lsx_vsub_h(in18, in21);
  __lsx_vst(vec4, interm_ptr, 176);
  vec4 = __lsx_vsub_h(in28, in27);
  __lsx_vst(vec4, interm_ptr, 112);
  vec4 = __lsx_vsub_h(in29, in26);
  __lsx_vst(vec4, interm_ptr, 128);

  DUP4_ARG2(__lsx_vadd_h, in18, in21, in19, in20, in28, in27, in29, in26, in21,
            in20, in27, in26);

  in22 = __lsx_vld(temp, 96);
  in23 = __lsx_vld(temp, 112);
  in24 = __lsx_vld(temp, 128);
  in25 = __lsx_vld(temp, 144);

  DOTP_CONST_PAIR(in25, in22, cospi_16_64, cospi_16_64, in22, in25);
  DOTP_CONST_PAIR(in24, in23, cospi_16_64, cospi_16_64, in23, in24);

  in16 = __lsx_vld(temp, 0);
  in17 = __lsx_vld(temp, 16);
  in30 = __lsx_vld(temp, 224);
  in31 = __lsx_vld(temp, 240);

  vec4 = __lsx_vsub_h(in17, in22);
  __lsx_vst(vec4, interm_ptr, 80);
  vec4 = __lsx_vsub_h(in30, in25);
  __lsx_vst(vec4, interm_ptr, 96);
  vec4 = __lsx_vsub_h(in31, in24);
  __lsx_vst(vec4, interm_ptr, 144);
  vec4 = __lsx_vsub_h(in16, in23);
  __lsx_vst(vec4, interm_ptr, 160);

  DUP4_ARG2(__lsx_vadd_h, in16, in23, in17, in22, in30, in25, in31, in24, in16,
            in17, in30, in31);
  DOTP_CONST_PAIR(in26, in21, cospi_24_64, cospi_8_64, in18, in29);
  DOTP_CONST_PAIR(in27, in20, cospi_24_64, cospi_8_64, in19, in28);

  DUP4_ARG2(__lsx_vadd_h, in16, in19, in17, in18, in30, in29, in31, in28, in27,
            in22, in21, in25);
  DOTP_CONST_PAIR(in21, in22, cospi_28_64, cospi_4_64, in26, in24);
  DUP2_ARG2(__lsx_vadd_h, in27, in26, in25, in24, in23, in20);

  DOTP_CONST_PAIR(in20, in23, cospi_31_64, cospi_1_64, vec4, vec5);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  __lsx_vst(vec5, out, 0);
  __lsx_vst(vec4, out, 240);

  DUP2_ARG2(__lsx_vsub_h, in27, in26, in25, in24, in22, in21);

  DOTP_CONST_PAIR(in21, in22, cospi_15_64, cospi_17_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  __lsx_vst(vec5, out, 224);
  __lsx_vst(vec4, out, 16);

  DUP4_ARG2(__lsx_vsub_h, in17, in18, in16, in19, in31, in28, in30, in29, in23,
            in26, in24, in20);
  tmp0 = __lsx_vneg_h(in23);
  DOTP_CONST_PAIR(tmp0, in20, cospi_28_64, cospi_4_64, in27, in25);
  DUP2_ARG2(__lsx_vsub_h, in26, in27, in24, in25, in23, in20);

  DOTP_CONST_PAIR(in20, in23, cospi_23_64, cospi_9_64, vec4, vec5);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  __lsx_vst(vec4, out, 32);
  __lsx_vst(vec5, out, 208);

  DUP2_ARG2(__lsx_vadd_h, in26, in27, in24, in25, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_7_64, cospi_25_64, vec4, vec5);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  __lsx_vst(vec4, out, 48);
  __lsx_vst(vec5, out, 192);

  in20 = __lsx_vld(interm_ptr, 64);
  in21 = __lsx_vld(interm_ptr, 176);
  in27 = __lsx_vld(interm_ptr, 112);
  in26 = __lsx_vld(interm_ptr, 128);

  in16 = in20;
  in17 = in21;
  DUP2_ARG1(__lsx_vneg_h, in16, in17, tmp0, tmp1);
  DOTP_CONST_PAIR(tmp0, in27, cospi_24_64, cospi_8_64, in20, in27);
  DOTP_CONST_PAIR(tmp1, in26, cospi_24_64, cospi_8_64, in21, in26);

  in22 = __lsx_vld(interm_ptr, 80);
  in25 = __lsx_vld(interm_ptr, 96);
  in24 = __lsx_vld(interm_ptr, 144);
  in23 = __lsx_vld(interm_ptr, 160);

  DUP4_ARG2(__lsx_vsub_h, in23, in20, in22, in21, in25, in26, in24, in27, in28,
            in17, in18, in31);
  DOTP_CONST_PAIR(in18, in17, cospi_12_64, cospi_20_64, in29, in30);
  DUP2_ARG2(__lsx_vadd_h, in28, in29, in31, in30, in16, in19);
  DOTP_CONST_PAIR(in19, in16, cospi_27_64, cospi_5_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  __lsx_vst(vec5, out, 64);
  __lsx_vst(vec4, out, 176);

  DUP2_ARG2(__lsx_vsub_h, in28, in29, in31, in30, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_11_64, cospi_21_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  __lsx_vst(vec5, out, 80);
  __lsx_vst(vec4, out, 160);

  DUP4_ARG2(__lsx_vadd_h, in22, in21, in23, in20, in24, in27, in25, in26, in16,
            in29, in30, in19);
  tmp0 = __lsx_vneg_h(in16);
  DOTP_CONST_PAIR(tmp0, in19, cospi_12_64, cospi_20_64, in28, in31);
  DUP2_ARG2(__lsx_vsub_h, in29, in28, in30, in31, in16, in19);

  DOTP_CONST_PAIR(in19, in16, cospi_19_64, cospi_13_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  __lsx_vst(vec5, out, 144);
  __lsx_vst(vec4, out, 96);

  DUP2_ARG2(__lsx_vadd_h, in29, in28, in30, in31, in17, in18);

  DOTP_CONST_PAIR(in18, in17, cospi_3_64, cospi_29_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  __lsx_vst(vec4, out, 112);
  __lsx_vst(vec5, out, 128);
}

static void fdct8x32_1d_row_transpose_store(int16_t *temp, int16_t *output) {
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1;

  /* 1st set */
  in0 = __lsx_vld(temp, 0);
  in4 = __lsx_vld(temp, 64);
  in2 = __lsx_vld(temp, 128);
  in6 = __lsx_vld(temp, 192);
  in1 = __lsx_vld(temp, 256);
  in7 = __lsx_vld(temp, 304);
  in3 = __lsx_vld(temp, 384);
  in5 = __lsx_vld(temp, 432);

  LSX_TRANSPOSE8x8_H(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);

  /* 2nd set */
  in0_1 = __lsx_vld(temp, 32);
  in1_1 = __lsx_vld(temp, 464);
  in2_1 = __lsx_vld(temp, 160);
  in3_1 = __lsx_vld(temp, 336);
  in4_1 = __lsx_vld(temp, 96);
  in5_1 = __lsx_vld(temp, 352);
  in6_1 = __lsx_vld(temp, 224);
  in7_1 = __lsx_vld(temp, 480);

  __lsx_vst(in0, output, 0);
  __lsx_vst(in1, output, 64);
  __lsx_vst(in2, output, 128);
  __lsx_vst(in3, output, 192);
  __lsx_vst(in4, output, 256);
  __lsx_vst(in5, output, 320);
  __lsx_vst(in6, output, 384);
  __lsx_vst(in7, output, 448);

  LSX_TRANSPOSE8x8_H(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1,
                     in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1);

  /* 3rd set */
  in0 = __lsx_vld(temp, 16);
  in1 = __lsx_vld(temp, 272);
  in2 = __lsx_vld(temp, 144);
  in3 = __lsx_vld(temp, 400);
  in4 = __lsx_vld(temp, 80);
  in5 = __lsx_vld(temp, 416);
  in6 = __lsx_vld(temp, 208);
  in7 = __lsx_vld(temp, 288);

  __lsx_vst(in0_1, output, 16);
  __lsx_vst(in1_1, output, 80);
  __lsx_vst(in2_1, output, 144);
  __lsx_vst(in3_1, output, 208);
  __lsx_vst(in4_1, output, 272);
  __lsx_vst(in5_1, output, 336);
  __lsx_vst(in6_1, output, 400);
  __lsx_vst(in7_1, output, 464);

  LSX_TRANSPOSE8x8_H(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);

  __lsx_vst(in0, output, 32);
  __lsx_vst(in1, output, 96);
  __lsx_vst(in2, output, 160);
  __lsx_vst(in3, output, 224);
  __lsx_vst(in4, output, 288);
  __lsx_vst(in5, output, 352);
  __lsx_vst(in6, output, 416);
  __lsx_vst(in7, output, 480);

  /* 4th set */
  in0_1 = __lsx_vld(temp, 48);
  in1_1 = __lsx_vld(temp, 448);
  in2_1 = __lsx_vld(temp, 176);
  in3_1 = __lsx_vld(temp, 320);
  in4_1 = __lsx_vld(temp, 112);
  in5_1 = __lsx_vld(temp, 368);
  in6_1 = __lsx_vld(temp, 240);
  in7_1 = __lsx_vld(temp, 496);

  LSX_TRANSPOSE8x8_H(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1,
                     in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1);

  __lsx_vst(in0_1, output, 48);
  __lsx_vst(in1_1, output, 112);
  __lsx_vst(in2_1, output, 176);
  __lsx_vst(in3_1, output, 240);
  __lsx_vst(in4_1, output, 304);
  __lsx_vst(in5_1, output, 368);
  __lsx_vst(in6_1, output, 432);
  __lsx_vst(in7_1, output, 496);
}

static void fdct32x8_1d_row(int16_t *temp, int16_t *temp_buf, int16_t *output) {
  fdct8x32_1d_row_load_butterfly(temp, temp_buf);
  fdct8x32_1d_row_even(temp_buf, temp_buf);
  fdct8x32_1d_row_odd(temp_buf + 128, temp, temp_buf + 128);
  fdct8x32_1d_row_transpose_store(temp_buf, output);
}

static void fdct32x8_1d_row_4x(int16_t *tmp_buf_big, int16_t *tmp_buf,
                               int16_t *output) {
  fdct8x32_1d_row_load_butterfly(tmp_buf_big, tmp_buf);
  fdct8x32_1d_row_even_4x(tmp_buf, tmp_buf_big, tmp_buf);
  fdct8x32_1d_row_odd(tmp_buf + 128, tmp_buf_big, tmp_buf + 128);
  fdct8x32_1d_row_transpose_store(tmp_buf, output);
}

void vpx_fdct32x32_lsx(const int16_t *input, int16_t *output,
                       int32_t src_stride) {
  int i;
  DECLARE_ALIGNED(32, int16_t, tmp_buf_big[1024]);
  DECLARE_ALIGNED(32, int16_t, tmp_buf[256]);

  /* column transform */
  for (i = 0; i < 4; ++i) {
    fdct8x32_1d_column(input + (8 * i), src_stride, tmp_buf,
                       tmp_buf_big + (8 * i));
  }

  /* row transform */
  fdct32x8_1d_row_4x(tmp_buf_big, tmp_buf, output);

  /* row transform */
  for (i = 1; i < 4; ++i) {
    fdct32x8_1d_row(tmp_buf_big + (i * 256), tmp_buf, output + (i * 256));
  }
}

static void fdct8x32_1d_row_even_rd(int16_t *temp, int16_t *out) {
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i in8, in9, in10, in11, in12, in13, in14, in15;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, temp0, temp1;

  /* fdct32 even */
  /* stage 2 */
  DUP4_ARG2(__lsx_vld, temp, 0, temp, 16, temp, 32, temp, 48, in0, in1, in2,
            in3);
  DUP4_ARG2(__lsx_vld, temp, 64, temp, 80, temp, 96, temp, 112, in4, in5, in6,
            in7);
  DUP4_ARG2(__lsx_vld, temp, 128, temp, 144, temp, 160, temp, 176, in8, in9,
            in10, in11);
  DUP4_ARG2(__lsx_vld, temp, 192, temp, 208, temp, 224, temp, 240, in12, in13,
            in14, in15);
  LSX_BUTTERFLY_16_H(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                     in11, in12, in13, in14, in15, vec0, vec1, vec2, vec3, vec4,
                     vec5, vec6, vec7, in8, in9, in10, in11, in12, in13, in14,
                     in15);

  FDCT_POSTPROC_2V_NEG_H(vec0, vec1);
  FDCT_POSTPROC_2V_NEG_H(vec2, vec3);
  FDCT_POSTPROC_2V_NEG_H(vec4, vec5);
  FDCT_POSTPROC_2V_NEG_H(vec6, vec7);
  FDCT_POSTPROC_2V_NEG_H(in8, in9);
  FDCT_POSTPROC_2V_NEG_H(in10, in11);
  FDCT_POSTPROC_2V_NEG_H(in12, in13);
  FDCT_POSTPROC_2V_NEG_H(in14, in15);

  /* Stage 3 */
  DUP4_ARG2(__lsx_vadd_h, vec0, vec7, vec1, vec6, vec2, vec5, vec3, vec4, in0,
            in1, in2, in3);

  temp0 = __lsx_vadd_h(in0, in3);
  in0 = __lsx_vsub_h(in0, in3);
  in3 = __lsx_vadd_h(in1, in2);
  in1 = __lsx_vsub_h(in1, in2);

  DOTP_CONST_PAIR(temp0, in3, cospi_16_64, cospi_16_64, temp1, temp0);
  __lsx_vst(temp0, out, 0);
  __lsx_vst(temp1, out, 16);

  DOTP_CONST_PAIR(in0, in1, cospi_24_64, cospi_8_64, temp1, temp0);
  __lsx_vst(temp0, out, 32);
  __lsx_vst(temp1, out, 48);

  DUP4_ARG2(__lsx_vsub_h, vec3, vec4, vec2, vec5, vec1, vec6, vec0, vec7, vec4,
            vec5, vec6, vec7);
  DOTP_CONST_PAIR(vec6, vec5, cospi_16_64, cospi_16_64, vec5, vec6);
  DUP2_ARG2(__lsx_vadd_h, vec4, vec5, vec7, vec6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_28_64, cospi_4_64, temp1, temp0);
  __lsx_vst(temp0, out, 64);
  __lsx_vst(temp1, out, 112);

  DUP2_ARG2(__lsx_vsub_h, vec4, vec5, vec7, vec6, vec4, vec7);
  DOTP_CONST_PAIR(vec7, vec4, cospi_12_64, cospi_20_64, temp1, temp0);
  __lsx_vst(temp0, out, 80);
  __lsx_vst(temp1, out, 96);

  DOTP_CONST_PAIR(in13, in10, cospi_16_64, cospi_16_64, vec2, vec5);
  DOTP_CONST_PAIR(in12, in11, cospi_16_64, cospi_16_64, vec3, vec4);
  DUP4_ARG2(__lsx_vadd_h, in8, vec3, in9, vec2, in14, vec5, in15, vec4, in0,
            vec1, vec6, in2);
  DOTP_CONST_PAIR(vec6, vec1, cospi_24_64, cospi_8_64, in1, in3);
  DUP2_ARG2(__lsx_vadd_h, in0, in1, in2, in3, vec0, vec7);
  DOTP_CONST_PAIR(vec7, vec0, cospi_30_64, cospi_2_64, temp1, temp0);
  __lsx_vst(temp0, out, 128);
  __lsx_vst(temp1, out, 240);

  DUP2_ARG2(__lsx_vsub_h, in0, in1, in2, in3, in0, in2);
  DOTP_CONST_PAIR(in2, in0, cospi_14_64, cospi_18_64, temp1, temp0);
  __lsx_vst(temp0, out, 144);
  __lsx_vst(temp1, out, 224);

  DUP2_ARG2(__lsx_vsub_h, in9, vec2, in14, vec5, vec2, vec5);
  temp0 = __lsx_vneg_h(vec2);
  DOTP_CONST_PAIR(temp0, vec5, cospi_24_64, cospi_8_64, in2, in1);
  DUP4_ARG2(__lsx_vsub_h, in8, vec3, in15, vec4, in3, in2, in0, in1, in3, in0,
            vec2, vec5);
  DOTP_CONST_PAIR(vec5, vec2, cospi_22_64, cospi_10_64, temp1, temp0);
  __lsx_vst(temp0, out, 160);
  __lsx_vst(temp1, out, 208);

  DUP2_ARG2(__lsx_vadd_h, in3, in2, in0, in1, vec3, vec4);
  DOTP_CONST_PAIR(vec4, vec3, cospi_6_64, cospi_26_64, temp0, temp1);
  __lsx_vst(temp0, out, 192);
  __lsx_vst(temp1, out, 176);
}

static void fdct8x32_1d_row_odd_rd(int16_t *temp, int16_t *interm_ptr,
                                   int16_t *out) {
  __m128i in16, in17, in18, in19, in20, in21, in22, in23;
  __m128i in24, in25, in26, in27, in28, in29, in30, in31;
  __m128i vec4, vec5, tmp0, tmp1;

  in20 = __lsx_vld(temp, 64);
  in21 = __lsx_vld(temp, 80);
  in26 = __lsx_vld(temp, 160);
  in27 = __lsx_vld(temp, 176);

  DOTP_CONST_PAIR(in27, in20, cospi_16_64, cospi_16_64, in20, in27);
  DOTP_CONST_PAIR(in26, in21, cospi_16_64, cospi_16_64, in21, in26);

  FDCT_POSTPROC_2V_NEG_H(in20, in21);
  FDCT_POSTPROC_2V_NEG_H(in26, in27);

  in18 = __lsx_vld(temp, 32);
  in19 = __lsx_vld(temp, 48);
  in28 = __lsx_vld(temp, 192);
  in29 = __lsx_vld(temp, 208);

  FDCT_POSTPROC_2V_NEG_H(in18, in19);
  FDCT_POSTPROC_2V_NEG_H(in28, in29);

  vec4 = __lsx_vsub_h(in19, in20);
  __lsx_vst(vec4, interm_ptr, 64);
  vec4 = __lsx_vsub_h(in18, in21);
  __lsx_vst(vec4, interm_ptr, 176);
  vec4 = __lsx_vsub_h(in29, in26);
  __lsx_vst(vec4, interm_ptr, 128);
  vec4 = __lsx_vsub_h(in28, in27);
  __lsx_vst(vec4, interm_ptr, 112);

  DUP4_ARG2(__lsx_vadd_h, in18, in21, in19, in20, in28, in27, in29, in26, in21,
            in20, in27, in26);

  in22 = __lsx_vld(temp, 96);
  in23 = __lsx_vld(temp, 112);
  in24 = __lsx_vld(temp, 128);
  in25 = __lsx_vld(temp, 144);

  DOTP_CONST_PAIR(in25, in22, cospi_16_64, cospi_16_64, in22, in25);
  DOTP_CONST_PAIR(in24, in23, cospi_16_64, cospi_16_64, in23, in24);
  FDCT_POSTPROC_2V_NEG_H(in22, in23);
  FDCT_POSTPROC_2V_NEG_H(in24, in25);

  in16 = __lsx_vld(temp, 0);
  in17 = __lsx_vld(temp, 16);
  in30 = __lsx_vld(temp, 224);
  in31 = __lsx_vld(temp, 240);

  FDCT_POSTPROC_2V_NEG_H(in16, in17);
  FDCT_POSTPROC_2V_NEG_H(in30, in31);

  vec4 = __lsx_vsub_h(in17, in22);
  __lsx_vst(vec4, interm_ptr, 80);
  vec4 = __lsx_vsub_h(in30, in25);
  __lsx_vst(vec4, interm_ptr, 96);
  vec4 = __lsx_vsub_h(in31, in24);
  __lsx_vst(vec4, interm_ptr, 144);
  vec4 = __lsx_vsub_h(in16, in23);
  __lsx_vst(vec4, interm_ptr, 160);

  DUP4_ARG2(__lsx_vadd_h, in16, in23, in17, in22, in30, in25, in31, in24, in16,
            in17, in30, in31);
  DOTP_CONST_PAIR(in26, in21, cospi_24_64, cospi_8_64, in18, in29);
  DOTP_CONST_PAIR(in27, in20, cospi_24_64, cospi_8_64, in19, in28);
  DUP4_ARG2(__lsx_vadd_h, in16, in19, in17, in18, in30, in29, in31, in28, in27,
            in22, in21, in25);
  DOTP_CONST_PAIR(in21, in22, cospi_28_64, cospi_4_64, in26, in24);
  DUP2_ARG2(__lsx_vadd_h, in27, in26, in25, in24, in23, in20);
  DOTP_CONST_PAIR(in20, in23, cospi_31_64, cospi_1_64, vec4, vec5);
  __lsx_vst(vec5, out, 0);
  __lsx_vst(vec4, out, 240);

  DUP2_ARG2(__lsx_vsub_h, in27, in26, in25, in24, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_15_64, cospi_17_64, vec5, vec4);
  __lsx_vst(vec5, out, 224);
  __lsx_vst(vec4, out, 16);

  DUP4_ARG2(__lsx_vsub_h, in17, in18, in16, in19, in31, in28, in30, in29, in23,
            in26, in24, in20);
  tmp0 = __lsx_vneg_h(in23);
  DOTP_CONST_PAIR(tmp0, in20, cospi_28_64, cospi_4_64, in27, in25);
  DUP2_ARG2(__lsx_vsub_h, in26, in27, in24, in25, in23, in20);
  DOTP_CONST_PAIR(in20, in23, cospi_23_64, cospi_9_64, vec4, vec5);
  __lsx_vst(vec4, out, 32);
  __lsx_vst(vec5, out, 208);

  DUP2_ARG2(__lsx_vadd_h, in26, in27, in24, in25, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_7_64, cospi_25_64, vec4, vec5);
  __lsx_vst(vec4, out, 48);
  __lsx_vst(vec5, out, 192);

  in20 = __lsx_vld(interm_ptr, 64);
  in21 = __lsx_vld(interm_ptr, 176);
  in27 = __lsx_vld(interm_ptr, 112);
  in26 = __lsx_vld(interm_ptr, 128);

  in16 = in20;
  in17 = in21;
  DUP2_ARG1(__lsx_vneg_h, in16, in17, tmp0, tmp1);
  DOTP_CONST_PAIR(tmp0, in27, cospi_24_64, cospi_8_64, in20, in27);
  DOTP_CONST_PAIR(tmp1, in26, cospi_24_64, cospi_8_64, in21, in26);

  in22 = __lsx_vld(interm_ptr, 80);
  in25 = __lsx_vld(interm_ptr, 96);
  in24 = __lsx_vld(interm_ptr, 144);
  in23 = __lsx_vld(interm_ptr, 160);

  DUP4_ARG2(__lsx_vsub_h, in23, in20, in22, in21, in25, in26, in24, in27, in28,
            in17, in18, in31);
  DOTP_CONST_PAIR(in18, in17, cospi_12_64, cospi_20_64, in29, in30);
  in16 = __lsx_vadd_h(in28, in29);
  in19 = __lsx_vadd_h(in31, in30);
  DOTP_CONST_PAIR(in19, in16, cospi_27_64, cospi_5_64, vec5, vec4);
  __lsx_vst(vec5, out, 64);
  __lsx_vst(vec4, out, 176);

  DUP2_ARG2(__lsx_vsub_h, in28, in29, in31, in30, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_11_64, cospi_21_64, vec5, vec4);
  __lsx_vst(vec5, out, 80);
  __lsx_vst(vec4, out, 160);

  DUP4_ARG2(__lsx_vadd_h, in22, in21, in23, in20, in24, in27, in25, in26, in16,
            in29, in30, in19);
  tmp0 = __lsx_vneg_h(in16);
  DOTP_CONST_PAIR(tmp0, in19, cospi_12_64, cospi_20_64, in28, in31);
  DUP2_ARG2(__lsx_vsub_h, in29, in28, in30, in31, in16, in19);
  DOTP_CONST_PAIR(in19, in16, cospi_19_64, cospi_13_64, vec5, vec4);
  __lsx_vst(vec5, out, 144);
  __lsx_vst(vec4, out, 96);

  DUP2_ARG2(__lsx_vadd_h, in29, in28, in30, in31, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_3_64, cospi_29_64, vec5, vec4);
  __lsx_vst(vec4, out, 112);
  __lsx_vst(vec5, out, 128);
}

static void fdct32x8_1d_row_rd(int16_t *tmp_buf_big, int16_t *tmp_buf,
                               int16_t *output) {
  fdct8x32_1d_row_load_butterfly(tmp_buf_big, tmp_buf);
  fdct8x32_1d_row_even_rd(tmp_buf, tmp_buf);
  fdct8x32_1d_row_odd_rd((tmp_buf + 128), tmp_buf_big, (tmp_buf + 128));
  fdct8x32_1d_row_transpose_store(tmp_buf, output);
}

void vpx_fdct32x32_rd_lsx(const int16_t *input, int16_t *out,
                          int32_t src_stride) {
  int32_t i;
  DECLARE_ALIGNED(32, int16_t, tmp_buf_big[1024]);
  DECLARE_ALIGNED(32, int16_t, tmp_buf[256]);

  /* column transform */
  for (i = 0; i < 4; ++i) {
    fdct8x32_1d_column(input + (8 * i), src_stride, &tmp_buf[0],
                       &tmp_buf_big[0] + (8 * i));
  }
  /* row transform */
  for (i = 0; i < 4; ++i) {
    fdct32x8_1d_row_rd(&tmp_buf_big[0] + (8 * i * 32), &tmp_buf[0],
                       out + (8 * i * 32));
  }
}
