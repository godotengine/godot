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
#include "vpx_dsp/mips/fwd_txfm_msa.h"

static void fdct8x32_1d_column_load_butterfly(const int16_t *input,
                                              int32_t src_stride,
                                              int16_t *temp_buff) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 step0, step1, step2, step3;
  v8i16 in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1;
  v8i16 step0_1, step1_1, step2_1, step3_1;

  /* 1st and 2nd set */
  LD_SH4(input, src_stride, in0, in1, in2, in3);
  LD_SH4(input + (28 * src_stride), src_stride, in4, in5, in6, in7);
  LD_SH4(input + (4 * src_stride), src_stride, in0_1, in1_1, in2_1, in3_1);
  LD_SH4(input + (24 * src_stride), src_stride, in4_1, in5_1, in6_1, in7_1);
  SLLI_4V(in0, in1, in2, in3, 2);
  SLLI_4V(in4, in5, in6, in7, 2);
  SLLI_4V(in0_1, in1_1, in2_1, in3_1, 2);
  SLLI_4V(in4_1, in5_1, in6_1, in7_1, 2);
  BUTTERFLY_8(in0, in1, in2, in3, in4, in5, in6, in7, step0, step1, step2,
              step3, in4, in5, in6, in7);
  BUTTERFLY_8(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1, step0_1,
              step1_1, step2_1, step3_1, in4_1, in5_1, in6_1, in7_1);
  ST_SH4(step0, step1, step2, step3, temp_buff, 8);
  ST_SH4(in4, in5, in6, in7, temp_buff + (28 * 8), 8);
  ST_SH4(step0_1, step1_1, step2_1, step3_1, temp_buff + (4 * 8), 8);
  ST_SH4(in4_1, in5_1, in6_1, in7_1, temp_buff + (24 * 8), 8);

  /* 3rd and 4th set */
  LD_SH4(input + (8 * src_stride), src_stride, in0, in1, in2, in3);
  LD_SH4(input + (20 * src_stride), src_stride, in4, in5, in6, in7);
  LD_SH4(input + (12 * src_stride), src_stride, in0_1, in1_1, in2_1, in3_1);
  LD_SH4(input + (16 * src_stride), src_stride, in4_1, in5_1, in6_1, in7_1);
  SLLI_4V(in0, in1, in2, in3, 2);
  SLLI_4V(in4, in5, in6, in7, 2);
  SLLI_4V(in0_1, in1_1, in2_1, in3_1, 2);
  SLLI_4V(in4_1, in5_1, in6_1, in7_1, 2);
  BUTTERFLY_8(in0, in1, in2, in3, in4, in5, in6, in7, step0, step1, step2,
              step3, in4, in5, in6, in7);
  BUTTERFLY_8(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1, step0_1,
              step1_1, step2_1, step3_1, in4_1, in5_1, in6_1, in7_1);
  ST_SH4(step0, step1, step2, step3, temp_buff + (8 * 8), 8);
  ST_SH4(in4, in5, in6, in7, temp_buff + (20 * 8), 8);
  ST_SH4(step0_1, step1_1, step2_1, step3_1, temp_buff + (12 * 8), 8);
  ST_SH4(in4_1, in5_1, in6_1, in7_1, temp_buff + (15 * 8) + 8, 8);
}

static void fdct8x32_1d_column_even_store(int16_t *input, int16_t *temp) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 in8, in9, in10, in11, in12, in13, in14, in15;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8i16 temp0, temp1;

  /* fdct even */
  LD_SH4(input, 8, in0, in1, in2, in3);
  LD_SH4(input + 96, 8, in12, in13, in14, in15);
  BUTTERFLY_8(in0, in1, in2, in3, in12, in13, in14, in15, vec0, vec1, vec2,
              vec3, in12, in13, in14, in15);
  LD_SH4(input + 32, 8, in4, in5, in6, in7);
  LD_SH4(input + 64, 8, in8, in9, in10, in11);
  BUTTERFLY_8(in4, in5, in6, in7, in8, in9, in10, in11, vec4, vec5, vec6, vec7,
              in8, in9, in10, in11);

  /* Stage 3 */
  ADD4(vec0, vec7, vec1, vec6, vec2, vec5, vec3, vec4, in0, in1, in2, in3);
  BUTTERFLY_4(in0, in1, in2, in3, temp0, in4, in1, in0);
  DOTP_CONST_PAIR(temp0, in4, cospi_16_64, cospi_16_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  ST_SH(temp0, temp);
  ST_SH(temp1, temp + 512);

  DOTP_CONST_PAIR(in0, in1, cospi_24_64, cospi_8_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  ST_SH(temp0, temp + 256);
  ST_SH(temp1, temp + 768);

  SUB4(vec0, vec7, vec1, vec6, vec2, vec5, vec3, vec4, vec7, vec6, vec5, vec4);
  DOTP_CONST_PAIR(vec6, vec5, cospi_16_64, cospi_16_64, vec5, vec6);
  ADD2(vec4, vec5, vec7, vec6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_28_64, cospi_4_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  ST_SH(temp0, temp + 128);
  ST_SH(temp1, temp + 896);

  SUB2(vec4, vec5, vec7, vec6, vec4, vec7);
  DOTP_CONST_PAIR(vec7, vec4, cospi_12_64, cospi_20_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  ST_SH(temp0, temp + 640);
  ST_SH(temp1, temp + 384);

  DOTP_CONST_PAIR(in13, in10, cospi_16_64, cospi_16_64, vec2, vec5);
  DOTP_CONST_PAIR(in12, in11, cospi_16_64, cospi_16_64, vec3, vec4);
  ADD4(in8, vec3, in9, vec2, in14, vec5, in15, vec4, in0, vec1, vec6, in2);
  DOTP_CONST_PAIR(vec6, vec1, cospi_24_64, cospi_8_64, in1, in3);
  ADD2(in0, in1, in2, in3, vec0, vec7);
  DOTP_CONST_PAIR(vec7, vec0, cospi_30_64, cospi_2_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  ST_SH(temp0, temp + 64);
  ST_SH(temp1, temp + 960);

  SUB2(in0, in1, in2, in3, in0, in2);
  DOTP_CONST_PAIR(in2, in0, cospi_14_64, cospi_18_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  ST_SH(temp0, temp + 576);
  ST_SH(temp1, temp + 448);

  SUB2(in9, vec2, in14, vec5, vec2, vec5);
  DOTP_CONST_PAIR((-vec2), vec5, cospi_24_64, cospi_8_64, in2, in1);
  SUB4(in8, vec3, in15, vec4, in3, in2, in0, in1, in3, in0, vec2, vec5);
  DOTP_CONST_PAIR(vec5, vec2, cospi_22_64, cospi_10_64, temp1, temp0);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  ST_SH(temp0, temp + 320);
  ST_SH(temp1, temp + 704);

  ADD2(in3, in2, in0, in1, vec3, vec4);
  DOTP_CONST_PAIR(vec4, vec3, cospi_6_64, cospi_26_64, temp0, temp1);
  FDCT32_POSTPROC_2V_POS_H(temp0, temp1);
  ST_SH(temp0, temp + 192);
  ST_SH(temp1, temp + 832);
}

static void fdct8x32_1d_column_odd_store(int16_t *input, int16_t *temp_ptr) {
  v8i16 in16, in17, in18, in19, in20, in21, in22, in23;
  v8i16 in24, in25, in26, in27, in28, in29, in30, in31, vec4, vec5;

  in20 = LD_SH(input + 32);
  in21 = LD_SH(input + 40);
  in26 = LD_SH(input + 80);
  in27 = LD_SH(input + 88);

  DOTP_CONST_PAIR(in27, in20, cospi_16_64, cospi_16_64, in20, in27);
  DOTP_CONST_PAIR(in26, in21, cospi_16_64, cospi_16_64, in21, in26);

  in18 = LD_SH(input + 16);
  in19 = LD_SH(input + 24);
  in28 = LD_SH(input + 96);
  in29 = LD_SH(input + 104);

  vec4 = in19 - in20;
  ST_SH(vec4, input + 32);
  vec4 = in18 - in21;
  ST_SH(vec4, input + 40);
  vec4 = in29 - in26;
  ST_SH(vec4, input + 80);
  vec4 = in28 - in27;
  ST_SH(vec4, input + 88);

  in21 = in18 + in21;
  in20 = in19 + in20;
  in27 = in28 + in27;
  in26 = in29 + in26;

  LD_SH4(input + 48, 8, in22, in23, in24, in25);
  DOTP_CONST_PAIR(in25, in22, cospi_16_64, cospi_16_64, in22, in25);
  DOTP_CONST_PAIR(in24, in23, cospi_16_64, cospi_16_64, in23, in24);

  in16 = LD_SH(input);
  in17 = LD_SH(input + 8);
  in30 = LD_SH(input + 112);
  in31 = LD_SH(input + 120);

  vec4 = in17 - in22;
  ST_SH(vec4, input + 16);
  vec4 = in16 - in23;
  ST_SH(vec4, input + 24);
  vec4 = in31 - in24;
  ST_SH(vec4, input + 96);
  vec4 = in30 - in25;
  ST_SH(vec4, input + 104);

  ADD4(in16, in23, in17, in22, in30, in25, in31, in24, in16, in17, in30, in31);
  DOTP_CONST_PAIR(in26, in21, cospi_24_64, cospi_8_64, in18, in29);
  DOTP_CONST_PAIR(in27, in20, cospi_24_64, cospi_8_64, in19, in28);
  ADD4(in16, in19, in17, in18, in30, in29, in31, in28, in27, in22, in21, in25);
  DOTP_CONST_PAIR(in21, in22, cospi_28_64, cospi_4_64, in26, in24);
  ADD2(in27, in26, in25, in24, in23, in20);
  DOTP_CONST_PAIR(in20, in23, cospi_31_64, cospi_1_64, vec4, vec5);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  ST_SH(vec5, temp_ptr);
  ST_SH(vec4, temp_ptr + 960);

  SUB2(in27, in26, in25, in24, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_15_64, cospi_17_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  ST_SH(vec5, temp_ptr + 448);
  ST_SH(vec4, temp_ptr + 512);

  SUB4(in17, in18, in16, in19, in31, in28, in30, in29, in23, in26, in24, in20);
  DOTP_CONST_PAIR((-in23), in20, cospi_28_64, cospi_4_64, in27, in25);
  SUB2(in26, in27, in24, in25, in23, in20);
  DOTP_CONST_PAIR(in20, in23, cospi_23_64, cospi_9_64, vec4, vec5);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  ST_SH(vec4, temp_ptr + 704);
  ST_SH(vec5, temp_ptr + 256);

  ADD2(in26, in27, in24, in25, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_7_64, cospi_25_64, vec4, vec5);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  ST_SH(vec4, temp_ptr + 192);
  ST_SH(vec5, temp_ptr + 768);

  LD_SH4(input + 16, 8, in22, in23, in20, in21);
  LD_SH4(input + 80, 8, in26, in27, in24, in25);
  in16 = in20;
  in17 = in21;
  DOTP_CONST_PAIR(-in16, in27, cospi_24_64, cospi_8_64, in20, in27);
  DOTP_CONST_PAIR(-in17, in26, cospi_24_64, cospi_8_64, in21, in26);
  SUB4(in23, in20, in22, in21, in25, in26, in24, in27, in28, in17, in18, in31);
  DOTP_CONST_PAIR(in18, in17, cospi_12_64, cospi_20_64, in29, in30);
  ADD2(in28, in29, in31, in30, in16, in19);
  DOTP_CONST_PAIR(in19, in16, cospi_27_64, cospi_5_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  ST_SH(vec5, temp_ptr + 832);
  ST_SH(vec4, temp_ptr + 128);

  SUB2(in28, in29, in31, in30, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_11_64, cospi_21_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  ST_SH(vec5, temp_ptr + 320);
  ST_SH(vec4, temp_ptr + 640);
  ADD4(in22, in21, in23, in20, in24, in27, in25, in26, in16, in29, in30, in19);
  DOTP_CONST_PAIR(-in16, in19, cospi_12_64, cospi_20_64, in28, in31);
  SUB2(in29, in28, in30, in31, in16, in19);
  DOTP_CONST_PAIR(in19, in16, cospi_19_64, cospi_13_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  ST_SH(vec5, temp_ptr + 576);
  ST_SH(vec4, temp_ptr + 384);

  ADD2(in29, in28, in30, in31, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_3_64, cospi_29_64, vec5, vec4);
  FDCT32_POSTPROC_2V_POS_H(vec5, vec4);
  ST_SH(vec5, temp_ptr + 64);
  ST_SH(vec4, temp_ptr + 896);
}

static void fdct8x32_1d_column(const int16_t *input, int32_t src_stride,
                               int16_t *tmp_buf, int16_t *tmp_buf_big) {
  fdct8x32_1d_column_load_butterfly(input, src_stride, tmp_buf);
  fdct8x32_1d_column_even_store(tmp_buf, tmp_buf_big);
  fdct8x32_1d_column_odd_store(tmp_buf + 128, (tmp_buf_big + 32));
}

static void fdct8x32_1d_row_load_butterfly(int16_t *temp_buff,
                                           int16_t *output) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 in8, in9, in10, in11, in12, in13, in14, in15;
  v8i16 step0, step1, step2, step3, step4, step5, step6, step7;

  LD_SH8(temp_buff, 32, in0, in1, in2, in3, in4, in5, in6, in7);
  LD_SH8(temp_buff + 24, 32, in8, in9, in10, in11, in12, in13, in14, in15);
  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  TRANSPOSE8x8_SH_SH(in8, in9, in10, in11, in12, in13, in14, in15, in8, in9,
                     in10, in11, in12, in13, in14, in15);
  BUTTERFLY_16(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
               in12, in13, in14, in15, step0, step1, step2, step3, step4, step5,
               step6, step7, in8, in9, in10, in11, in12, in13, in14, in15);
  ST_SH8(step0, step1, step2, step3, step4, step5, step6, step7, output, 8);
  ST_SH8(in8, in9, in10, in11, in12, in13, in14, in15, (output + 24 * 8), 8);

  /* 2nd set */
  LD_SH8(temp_buff + 8, 32, in0, in1, in2, in3, in4, in5, in6, in7);
  LD_SH8(temp_buff + 16, 32, in8, in9, in10, in11, in12, in13, in14, in15);
  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  TRANSPOSE8x8_SH_SH(in8, in9, in10, in11, in12, in13, in14, in15, in8, in9,
                     in10, in11, in12, in13, in14, in15);
  BUTTERFLY_16(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
               in12, in13, in14, in15, step0, step1, step2, step3, step4, step5,
               step6, step7, in8, in9, in10, in11, in12, in13, in14, in15);
  ST_SH8(step0, step1, step2, step3, step4, step5, step6, step7,
         (output + 8 * 8), 8);
  ST_SH8(in8, in9, in10, in11, in12, in13, in14, in15, (output + 16 * 8), 8);
}

static void fdct8x32_1d_row_even_4x(int16_t *input, int16_t *interm_ptr,
                                    int16_t *out) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 in8, in9, in10, in11, in12, in13, in14, in15;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v4i32 vec0_l, vec1_l, vec2_l, vec3_l, vec4_l, vec5_l, vec6_l, vec7_l;
  v4i32 vec0_r, vec1_r, vec2_r, vec3_r, vec4_r, vec5_r, vec6_r, vec7_r;
  v4i32 tmp0_w, tmp1_w, tmp2_w, tmp3_w;

  /* fdct32 even */
  /* stage 2 */
  LD_SH8(input, 8, in0, in1, in2, in3, in4, in5, in6, in7);
  LD_SH8(input + 64, 8, in8, in9, in10, in11, in12, in13, in14, in15);

  BUTTERFLY_16(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
               in12, in13, in14, in15, vec0, vec1, vec2, vec3, vec4, vec5, vec6,
               vec7, in8, in9, in10, in11, in12, in13, in14, in15);
  ST_SH8(vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, interm_ptr, 8);
  ST_SH8(in8, in9, in10, in11, in12, in13, in14, in15, interm_ptr + 64, 8);

  /* Stage 3 */
  UNPCK_SH_SW(vec0, vec0_l, vec0_r);
  UNPCK_SH_SW(vec1, vec1_l, vec1_r);
  UNPCK_SH_SW(vec2, vec2_l, vec2_r);
  UNPCK_SH_SW(vec3, vec3_l, vec3_r);
  UNPCK_SH_SW(vec4, vec4_l, vec4_r);
  UNPCK_SH_SW(vec5, vec5_l, vec5_r);
  UNPCK_SH_SW(vec6, vec6_l, vec6_r);
  UNPCK_SH_SW(vec7, vec7_l, vec7_r);
  ADD4(vec0_r, vec7_r, vec1_r, vec6_r, vec2_r, vec5_r, vec3_r, vec4_r, tmp0_w,
       tmp1_w, tmp2_w, tmp3_w);
  BUTTERFLY_4(tmp0_w, tmp1_w, tmp2_w, tmp3_w, vec4_r, vec6_r, vec7_r, vec5_r);
  ADD4(vec0_l, vec7_l, vec1_l, vec6_l, vec2_l, vec5_l, vec3_l, vec4_l, vec0_r,
       vec1_r, vec2_r, vec3_r);

  tmp3_w = vec0_r + vec3_r;
  vec0_r = vec0_r - vec3_r;
  vec3_r = vec1_r + vec2_r;
  vec1_r = vec1_r - vec2_r;

  DOTP_CONST_PAIR_W(vec4_r, vec6_r, tmp3_w, vec3_r, cospi_16_64, cospi_16_64,
                    vec4_r, tmp3_w, vec6_r, vec3_r);
  FDCT32_POSTPROC_NEG_W(vec4_r);
  FDCT32_POSTPROC_NEG_W(tmp3_w);
  FDCT32_POSTPROC_NEG_W(vec6_r);
  FDCT32_POSTPROC_NEG_W(vec3_r);
  PCKEV_H2_SH(vec4_r, tmp3_w, vec6_r, vec3_r, vec4, vec5);
  ST_SH2(vec5, vec4, out, 8);

  DOTP_CONST_PAIR_W(vec5_r, vec7_r, vec0_r, vec1_r, cospi_24_64, cospi_8_64,
                    vec4_r, tmp3_w, vec6_r, vec3_r);
  FDCT32_POSTPROC_NEG_W(vec4_r);
  FDCT32_POSTPROC_NEG_W(tmp3_w);
  FDCT32_POSTPROC_NEG_W(vec6_r);
  FDCT32_POSTPROC_NEG_W(vec3_r);
  PCKEV_H2_SH(vec4_r, tmp3_w, vec6_r, vec3_r, vec4, vec5);
  ST_SH2(vec5, vec4, out + 16, 8);

  LD_SH8(interm_ptr, 8, vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7);
  SUB4(vec3, vec4, vec2, vec5, vec1, vec6, vec0, vec7, vec4, vec5, vec6, vec7);
  DOTP_CONST_PAIR(vec6, vec5, cospi_16_64, cospi_16_64, vec5, vec6);
  ADD2(vec4, vec5, vec7, vec6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_28_64, cospi_4_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  ST_SH(in4, out + 32);
  ST_SH(in5, out + 56);

  SUB2(vec4, vec5, vec7, vec6, vec4, vec7);
  DOTP_CONST_PAIR(vec7, vec4, cospi_12_64, cospi_20_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  ST_SH(in4, out + 40);
  ST_SH(in5, out + 48);

  LD_SH8(interm_ptr + 64, 8, in8, in9, in10, in11, in12, in13, in14, in15);
  DOTP_CONST_PAIR(in13, in10, cospi_16_64, cospi_16_64, vec2, vec5);
  DOTP_CONST_PAIR(in12, in11, cospi_16_64, cospi_16_64, vec3, vec4);
  ADD4(in8, vec3, in9, vec2, in14, vec5, in15, vec4, in0, vec1, vec6, in2);
  DOTP_CONST_PAIR(vec6, vec1, cospi_24_64, cospi_8_64, in1, in3);
  ADD2(in0, in1, in2, in3, vec0, vec7);
  DOTP_CONST_PAIR(vec7, vec0, cospi_30_64, cospi_2_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  ST_SH(in4, out + 64);
  ST_SH(in5, out + 120);

  SUB2(in0, in1, in2, in3, in0, in2);
  DOTP_CONST_PAIR(in2, in0, cospi_14_64, cospi_18_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  ST_SH(in4, out + 72);
  ST_SH(in5, out + 112);

  SUB2(in9, vec2, in14, vec5, vec2, vec5);
  DOTP_CONST_PAIR((-vec2), vec5, cospi_24_64, cospi_8_64, in2, in1);
  SUB4(in8, vec3, in15, vec4, in3, in2, in0, in1, in3, in0, vec2, vec5);
  DOTP_CONST_PAIR(vec5, vec2, cospi_22_64, cospi_10_64, in5, in4);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  ST_SH(in4, out + 80);
  ST_SH(in5, out + 104);

  ADD2(in3, in2, in0, in1, vec3, vec4);
  DOTP_CONST_PAIR(vec4, vec3, cospi_6_64, cospi_26_64, in4, in5);
  FDCT_POSTPROC_2V_NEG_H(in4, in5);
  ST_SH(in4, out + 96);
  ST_SH(in5, out + 88);
}

static void fdct8x32_1d_row_even(int16_t *temp, int16_t *out) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 in8, in9, in10, in11, in12, in13, in14, in15;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, temp0, temp1;

  /* fdct32 even */
  /* stage 2 */
  LD_SH8(temp, 8, in0, in1, in2, in3, in4, in5, in6, in7);
  LD_SH8(temp + 64, 8, in8, in9, in10, in11, in12, in13, in14, in15);

  BUTTERFLY_16(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
               in12, in13, in14, in15, vec0, vec1, vec2, vec3, vec4, vec5, vec6,
               vec7, in8, in9, in10, in11, in12, in13, in14, in15);

  /* Stage 3 */
  ADD4(vec0, vec7, vec1, vec6, vec2, vec5, vec3, vec4, in0, in1, in2, in3);
  BUTTERFLY_4(in0, in1, in2, in3, temp0, in4, in1, in0);
  DOTP_CONST_PAIR(temp0, in4, cospi_16_64, cospi_16_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  ST_SH(temp0, out);
  ST_SH(temp1, out + 8);

  DOTP_CONST_PAIR(in0, in1, cospi_24_64, cospi_8_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  ST_SH(temp0, out + 16);
  ST_SH(temp1, out + 24);

  SUB4(vec3, vec4, vec2, vec5, vec1, vec6, vec0, vec7, vec4, vec5, vec6, vec7);
  DOTP_CONST_PAIR(vec6, vec5, cospi_16_64, cospi_16_64, vec5, vec6);
  ADD2(vec4, vec5, vec7, vec6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_28_64, cospi_4_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  ST_SH(temp0, out + 32);
  ST_SH(temp1, out + 56);

  SUB2(vec4, vec5, vec7, vec6, vec4, vec7);
  DOTP_CONST_PAIR(vec7, vec4, cospi_12_64, cospi_20_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  ST_SH(temp0, out + 40);
  ST_SH(temp1, out + 48);

  DOTP_CONST_PAIR(in13, in10, cospi_16_64, cospi_16_64, vec2, vec5);
  DOTP_CONST_PAIR(in12, in11, cospi_16_64, cospi_16_64, vec3, vec4);
  ADD4(in8, vec3, in9, vec2, in14, vec5, in15, vec4, in0, vec1, vec6, in2);
  DOTP_CONST_PAIR(vec6, vec1, cospi_24_64, cospi_8_64, in1, in3);
  ADD2(in0, in1, in2, in3, vec0, vec7);
  DOTP_CONST_PAIR(vec7, vec0, cospi_30_64, cospi_2_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  ST_SH(temp0, out + 64);
  ST_SH(temp1, out + 120);

  SUB2(in0, in1, in2, in3, in0, in2);
  DOTP_CONST_PAIR(in2, in0, cospi_14_64, cospi_18_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  ST_SH(temp0, out + 72);
  ST_SH(temp1, out + 112);

  SUB2(in9, vec2, in14, vec5, vec2, vec5);
  DOTP_CONST_PAIR((-vec2), vec5, cospi_24_64, cospi_8_64, in2, in1);
  SUB4(in8, vec3, in15, vec4, in3, in2, in0, in1, in3, in0, vec2, vec5)
  DOTP_CONST_PAIR(vec5, vec2, cospi_22_64, cospi_10_64, temp1, temp0);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  ST_SH(temp0, out + 80);
  ST_SH(temp1, out + 104);

  ADD2(in3, in2, in0, in1, vec3, vec4);
  DOTP_CONST_PAIR(vec4, vec3, cospi_6_64, cospi_26_64, temp0, temp1);
  FDCT_POSTPROC_2V_NEG_H(temp0, temp1);
  ST_SH(temp0, out + 96);
  ST_SH(temp1, out + 88);
}

static void fdct8x32_1d_row_odd(int16_t *temp, int16_t *interm_ptr,
                                int16_t *out) {
  v8i16 in16, in17, in18, in19, in20, in21, in22, in23;
  v8i16 in24, in25, in26, in27, in28, in29, in30, in31, vec4, vec5;

  in20 = LD_SH(temp + 32);
  in21 = LD_SH(temp + 40);
  in26 = LD_SH(temp + 80);
  in27 = LD_SH(temp + 88);

  DOTP_CONST_PAIR(in27, in20, cospi_16_64, cospi_16_64, in20, in27);
  DOTP_CONST_PAIR(in26, in21, cospi_16_64, cospi_16_64, in21, in26);

  in18 = LD_SH(temp + 16);
  in19 = LD_SH(temp + 24);
  in28 = LD_SH(temp + 96);
  in29 = LD_SH(temp + 104);

  vec4 = in19 - in20;
  ST_SH(vec4, interm_ptr + 32);
  vec4 = in18 - in21;
  ST_SH(vec4, interm_ptr + 88);
  vec4 = in28 - in27;
  ST_SH(vec4, interm_ptr + 56);
  vec4 = in29 - in26;
  ST_SH(vec4, interm_ptr + 64);

  ADD4(in18, in21, in19, in20, in28, in27, in29, in26, in21, in20, in27, in26);

  in22 = LD_SH(temp + 48);
  in23 = LD_SH(temp + 56);
  in24 = LD_SH(temp + 64);
  in25 = LD_SH(temp + 72);

  DOTP_CONST_PAIR(in25, in22, cospi_16_64, cospi_16_64, in22, in25);
  DOTP_CONST_PAIR(in24, in23, cospi_16_64, cospi_16_64, in23, in24);

  in16 = LD_SH(temp);
  in17 = LD_SH(temp + 8);
  in30 = LD_SH(temp + 112);
  in31 = LD_SH(temp + 120);

  vec4 = in17 - in22;
  ST_SH(vec4, interm_ptr + 40);
  vec4 = in30 - in25;
  ST_SH(vec4, interm_ptr + 48);
  vec4 = in31 - in24;
  ST_SH(vec4, interm_ptr + 72);
  vec4 = in16 - in23;
  ST_SH(vec4, interm_ptr + 80);

  ADD4(in16, in23, in17, in22, in30, in25, in31, in24, in16, in17, in30, in31);
  DOTP_CONST_PAIR(in26, in21, cospi_24_64, cospi_8_64, in18, in29);
  DOTP_CONST_PAIR(in27, in20, cospi_24_64, cospi_8_64, in19, in28);

  ADD4(in16, in19, in17, in18, in30, in29, in31, in28, in27, in22, in21, in25);
  DOTP_CONST_PAIR(in21, in22, cospi_28_64, cospi_4_64, in26, in24);
  ADD2(in27, in26, in25, in24, in23, in20);

  DOTP_CONST_PAIR(in20, in23, cospi_31_64, cospi_1_64, vec4, vec5);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  ST_SH(vec5, out);
  ST_SH(vec4, out + 120);

  SUB2(in27, in26, in25, in24, in22, in21);

  DOTP_CONST_PAIR(in21, in22, cospi_15_64, cospi_17_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  ST_SH(vec5, out + 112);
  ST_SH(vec4, out + 8);

  SUB4(in17, in18, in16, in19, in31, in28, in30, in29, in23, in26, in24, in20);
  DOTP_CONST_PAIR((-in23), in20, cospi_28_64, cospi_4_64, in27, in25);
  SUB2(in26, in27, in24, in25, in23, in20);

  DOTP_CONST_PAIR(in20, in23, cospi_23_64, cospi_9_64, vec4, vec5);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  ST_SH(vec4, out + 16);
  ST_SH(vec5, out + 104);

  ADD2(in26, in27, in24, in25, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_7_64, cospi_25_64, vec4, vec5);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  ST_SH(vec4, out + 24);
  ST_SH(vec5, out + 96);

  in20 = LD_SH(interm_ptr + 32);
  in21 = LD_SH(interm_ptr + 88);
  in27 = LD_SH(interm_ptr + 56);
  in26 = LD_SH(interm_ptr + 64);

  in16 = in20;
  in17 = in21;
  DOTP_CONST_PAIR(-in16, in27, cospi_24_64, cospi_8_64, in20, in27);
  DOTP_CONST_PAIR(-in17, in26, cospi_24_64, cospi_8_64, in21, in26);

  in22 = LD_SH(interm_ptr + 40);
  in25 = LD_SH(interm_ptr + 48);
  in24 = LD_SH(interm_ptr + 72);
  in23 = LD_SH(interm_ptr + 80);

  SUB4(in23, in20, in22, in21, in25, in26, in24, in27, in28, in17, in18, in31);
  DOTP_CONST_PAIR(in18, in17, cospi_12_64, cospi_20_64, in29, in30);
  ADD2(in28, in29, in31, in30, in16, in19);
  DOTP_CONST_PAIR(in19, in16, cospi_27_64, cospi_5_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  ST_SH(vec5, out + 32);
  ST_SH(vec4, out + 88);

  SUB2(in28, in29, in31, in30, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_11_64, cospi_21_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  ST_SH(vec5, out + 40);
  ST_SH(vec4, out + 80);

  ADD4(in22, in21, in23, in20, in24, in27, in25, in26, in16, in29, in30, in19);
  DOTP_CONST_PAIR(-in16, in19, cospi_12_64, cospi_20_64, in28, in31);
  SUB2(in29, in28, in30, in31, in16, in19);

  DOTP_CONST_PAIR(in19, in16, cospi_19_64, cospi_13_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  ST_SH(vec5, out + 72);
  ST_SH(vec4, out + 48);

  ADD2(in29, in28, in30, in31, in17, in18);

  DOTP_CONST_PAIR(in18, in17, cospi_3_64, cospi_29_64, vec5, vec4);
  FDCT_POSTPROC_2V_NEG_H(vec5, vec4);
  ST_SH(vec4, out + 56);
  ST_SH(vec5, out + 64);
}

static void fdct8x32_1d_row_transpose_store(int16_t *temp, int16_t *output) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1;

  /* 1st set */
  in0 = LD_SH(temp);
  in4 = LD_SH(temp + 32);
  in2 = LD_SH(temp + 64);
  in6 = LD_SH(temp + 96);
  in1 = LD_SH(temp + 128);
  in7 = LD_SH(temp + 152);
  in3 = LD_SH(temp + 192);
  in5 = LD_SH(temp + 216);

  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);

  /* 2nd set */
  in0_1 = LD_SH(temp + 16);
  in1_1 = LD_SH(temp + 232);
  in2_1 = LD_SH(temp + 80);
  in3_1 = LD_SH(temp + 168);
  in4_1 = LD_SH(temp + 48);
  in5_1 = LD_SH(temp + 176);
  in6_1 = LD_SH(temp + 112);
  in7_1 = LD_SH(temp + 240);

  ST_SH8(in0, in1, in2, in3, in4, in5, in6, in7, output, 32);
  TRANSPOSE8x8_SH_SH(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1,
                     in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1);

  /* 3rd set */
  in0 = LD_SH(temp + 8);
  in1 = LD_SH(temp + 136);
  in2 = LD_SH(temp + 72);
  in3 = LD_SH(temp + 200);
  in4 = LD_SH(temp + 40);
  in5 = LD_SH(temp + 208);
  in6 = LD_SH(temp + 104);
  in7 = LD_SH(temp + 144);

  ST_SH8(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1, output + 8,
         32);
  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  ST_SH8(in0, in1, in2, in3, in4, in5, in6, in7, output + 16, 32);

  /* 4th set */
  in0_1 = LD_SH(temp + 24);
  in1_1 = LD_SH(temp + 224);
  in2_1 = LD_SH(temp + 88);
  in3_1 = LD_SH(temp + 160);
  in4_1 = LD_SH(temp + 56);
  in5_1 = LD_SH(temp + 184);
  in6_1 = LD_SH(temp + 120);
  in7_1 = LD_SH(temp + 248);

  TRANSPOSE8x8_SH_SH(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1,
                     in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1);
  ST_SH8(in0_1, in1_1, in2_1, in3_1, in4_1, in5_1, in6_1, in7_1, output + 24,
         32);
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

void vpx_fdct32x32_msa(const int16_t *input, int16_t *output,
                       int32_t src_stride) {
  int32_t i;
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
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 in8, in9, in10, in11, in12, in13, in14, in15;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, temp0, temp1;

  /* fdct32 even */
  /* stage 2 */
  LD_SH8(temp, 8, in0, in1, in2, in3, in4, in5, in6, in7);
  LD_SH8(temp + 64, 8, in8, in9, in10, in11, in12, in13, in14, in15);

  BUTTERFLY_16(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
               in12, in13, in14, in15, vec0, vec1, vec2, vec3, vec4, vec5, vec6,
               vec7, in8, in9, in10, in11, in12, in13, in14, in15);
  FDCT_POSTPROC_2V_NEG_H(vec0, vec1);
  FDCT_POSTPROC_2V_NEG_H(vec2, vec3);
  FDCT_POSTPROC_2V_NEG_H(vec4, vec5);
  FDCT_POSTPROC_2V_NEG_H(vec6, vec7);
  FDCT_POSTPROC_2V_NEG_H(in8, in9);
  FDCT_POSTPROC_2V_NEG_H(in10, in11);
  FDCT_POSTPROC_2V_NEG_H(in12, in13);
  FDCT_POSTPROC_2V_NEG_H(in14, in15);

  /* Stage 3 */
  ADD4(vec0, vec7, vec1, vec6, vec2, vec5, vec3, vec4, in0, in1, in2, in3);

  temp0 = in0 + in3;
  in0 = in0 - in3;
  in3 = in1 + in2;
  in1 = in1 - in2;

  DOTP_CONST_PAIR(temp0, in3, cospi_16_64, cospi_16_64, temp1, temp0);
  ST_SH(temp0, out);
  ST_SH(temp1, out + 8);

  DOTP_CONST_PAIR(in0, in1, cospi_24_64, cospi_8_64, temp1, temp0);
  ST_SH(temp0, out + 16);
  ST_SH(temp1, out + 24);

  SUB4(vec3, vec4, vec2, vec5, vec1, vec6, vec0, vec7, vec4, vec5, vec6, vec7);
  DOTP_CONST_PAIR(vec6, vec5, cospi_16_64, cospi_16_64, vec5, vec6);
  ADD2(vec4, vec5, vec7, vec6, vec0, vec1);
  DOTP_CONST_PAIR(vec1, vec0, cospi_28_64, cospi_4_64, temp1, temp0);
  ST_SH(temp0, out + 32);
  ST_SH(temp1, out + 56);

  SUB2(vec4, vec5, vec7, vec6, vec4, vec7);
  DOTP_CONST_PAIR(vec7, vec4, cospi_12_64, cospi_20_64, temp1, temp0);
  ST_SH(temp0, out + 40);
  ST_SH(temp1, out + 48);

  DOTP_CONST_PAIR(in13, in10, cospi_16_64, cospi_16_64, vec2, vec5);
  DOTP_CONST_PAIR(in12, in11, cospi_16_64, cospi_16_64, vec3, vec4);
  ADD4(in8, vec3, in9, vec2, in14, vec5, in15, vec4, in0, vec1, vec6, in2);
  DOTP_CONST_PAIR(vec6, vec1, cospi_24_64, cospi_8_64, in1, in3);
  ADD2(in0, in1, in2, in3, vec0, vec7);
  DOTP_CONST_PAIR(vec7, vec0, cospi_30_64, cospi_2_64, temp1, temp0);
  ST_SH(temp0, out + 64);
  ST_SH(temp1, out + 120);

  SUB2(in0, in1, in2, in3, in0, in2);
  DOTP_CONST_PAIR(in2, in0, cospi_14_64, cospi_18_64, temp1, temp0);
  ST_SH(temp0, out + 72);
  ST_SH(temp1, out + 112);

  SUB2(in9, vec2, in14, vec5, vec2, vec5);
  DOTP_CONST_PAIR((-vec2), vec5, cospi_24_64, cospi_8_64, in2, in1);
  SUB4(in8, vec3, in15, vec4, in3, in2, in0, in1, in3, in0, vec2, vec5);
  DOTP_CONST_PAIR(vec5, vec2, cospi_22_64, cospi_10_64, temp1, temp0);
  ST_SH(temp0, out + 80);
  ST_SH(temp1, out + 104);

  ADD2(in3, in2, in0, in1, vec3, vec4);
  DOTP_CONST_PAIR(vec4, vec3, cospi_6_64, cospi_26_64, temp0, temp1);
  ST_SH(temp0, out + 96);
  ST_SH(temp1, out + 88);
}

static void fdct8x32_1d_row_odd_rd(int16_t *temp, int16_t *interm_ptr,
                                   int16_t *out) {
  v8i16 in16, in17, in18, in19, in20, in21, in22, in23;
  v8i16 in24, in25, in26, in27, in28, in29, in30, in31;
  v8i16 vec4, vec5;

  in20 = LD_SH(temp + 32);
  in21 = LD_SH(temp + 40);
  in26 = LD_SH(temp + 80);
  in27 = LD_SH(temp + 88);

  DOTP_CONST_PAIR(in27, in20, cospi_16_64, cospi_16_64, in20, in27);
  DOTP_CONST_PAIR(in26, in21, cospi_16_64, cospi_16_64, in21, in26);

  FDCT_POSTPROC_2V_NEG_H(in20, in21);
  FDCT_POSTPROC_2V_NEG_H(in26, in27);

  in18 = LD_SH(temp + 16);
  in19 = LD_SH(temp + 24);
  in28 = LD_SH(temp + 96);
  in29 = LD_SH(temp + 104);

  FDCT_POSTPROC_2V_NEG_H(in18, in19);
  FDCT_POSTPROC_2V_NEG_H(in28, in29);

  vec4 = in19 - in20;
  ST_SH(vec4, interm_ptr + 32);
  vec4 = in18 - in21;
  ST_SH(vec4, interm_ptr + 88);
  vec4 = in29 - in26;
  ST_SH(vec4, interm_ptr + 64);
  vec4 = in28 - in27;
  ST_SH(vec4, interm_ptr + 56);

  ADD4(in18, in21, in19, in20, in28, in27, in29, in26, in21, in20, in27, in26);

  in22 = LD_SH(temp + 48);
  in23 = LD_SH(temp + 56);
  in24 = LD_SH(temp + 64);
  in25 = LD_SH(temp + 72);

  DOTP_CONST_PAIR(in25, in22, cospi_16_64, cospi_16_64, in22, in25);
  DOTP_CONST_PAIR(in24, in23, cospi_16_64, cospi_16_64, in23, in24);
  FDCT_POSTPROC_2V_NEG_H(in22, in23);
  FDCT_POSTPROC_2V_NEG_H(in24, in25);

  in16 = LD_SH(temp);
  in17 = LD_SH(temp + 8);
  in30 = LD_SH(temp + 112);
  in31 = LD_SH(temp + 120);

  FDCT_POSTPROC_2V_NEG_H(in16, in17);
  FDCT_POSTPROC_2V_NEG_H(in30, in31);

  vec4 = in17 - in22;
  ST_SH(vec4, interm_ptr + 40);
  vec4 = in30 - in25;
  ST_SH(vec4, interm_ptr + 48);
  vec4 = in31 - in24;
  ST_SH(vec4, interm_ptr + 72);
  vec4 = in16 - in23;
  ST_SH(vec4, interm_ptr + 80);

  ADD4(in16, in23, in17, in22, in30, in25, in31, in24, in16, in17, in30, in31);
  DOTP_CONST_PAIR(in26, in21, cospi_24_64, cospi_8_64, in18, in29);
  DOTP_CONST_PAIR(in27, in20, cospi_24_64, cospi_8_64, in19, in28);
  ADD4(in16, in19, in17, in18, in30, in29, in31, in28, in27, in22, in21, in25);
  DOTP_CONST_PAIR(in21, in22, cospi_28_64, cospi_4_64, in26, in24);
  ADD2(in27, in26, in25, in24, in23, in20);
  DOTP_CONST_PAIR(in20, in23, cospi_31_64, cospi_1_64, vec4, vec5);
  ST_SH(vec5, out);
  ST_SH(vec4, out + 120);

  SUB2(in27, in26, in25, in24, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_15_64, cospi_17_64, vec5, vec4);
  ST_SH(vec5, out + 112);
  ST_SH(vec4, out + 8);

  SUB4(in17, in18, in16, in19, in31, in28, in30, in29, in23, in26, in24, in20);
  DOTP_CONST_PAIR((-in23), in20, cospi_28_64, cospi_4_64, in27, in25);
  SUB2(in26, in27, in24, in25, in23, in20);
  DOTP_CONST_PAIR(in20, in23, cospi_23_64, cospi_9_64, vec4, vec5);
  ST_SH(vec4, out + 16);
  ST_SH(vec5, out + 104);

  ADD2(in26, in27, in24, in25, in22, in21);
  DOTP_CONST_PAIR(in21, in22, cospi_7_64, cospi_25_64, vec4, vec5);
  ST_SH(vec4, out + 24);
  ST_SH(vec5, out + 96);

  in20 = LD_SH(interm_ptr + 32);
  in21 = LD_SH(interm_ptr + 88);
  in27 = LD_SH(interm_ptr + 56);
  in26 = LD_SH(interm_ptr + 64);

  in16 = in20;
  in17 = in21;
  DOTP_CONST_PAIR(-in16, in27, cospi_24_64, cospi_8_64, in20, in27);
  DOTP_CONST_PAIR(-in17, in26, cospi_24_64, cospi_8_64, in21, in26);

  in22 = LD_SH(interm_ptr + 40);
  in25 = LD_SH(interm_ptr + 48);
  in24 = LD_SH(interm_ptr + 72);
  in23 = LD_SH(interm_ptr + 80);

  SUB4(in23, in20, in22, in21, in25, in26, in24, in27, in28, in17, in18, in31);
  DOTP_CONST_PAIR(in18, in17, cospi_12_64, cospi_20_64, in29, in30);
  in16 = in28 + in29;
  in19 = in31 + in30;
  DOTP_CONST_PAIR(in19, in16, cospi_27_64, cospi_5_64, vec5, vec4);
  ST_SH(vec5, out + 32);
  ST_SH(vec4, out + 88);

  SUB2(in28, in29, in31, in30, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_11_64, cospi_21_64, vec5, vec4);
  ST_SH(vec5, out + 40);
  ST_SH(vec4, out + 80);

  ADD4(in22, in21, in23, in20, in24, in27, in25, in26, in16, in29, in30, in19);
  DOTP_CONST_PAIR(-in16, in19, cospi_12_64, cospi_20_64, in28, in31);
  SUB2(in29, in28, in30, in31, in16, in19);
  DOTP_CONST_PAIR(in19, in16, cospi_19_64, cospi_13_64, vec5, vec4);
  ST_SH(vec5, out + 72);
  ST_SH(vec4, out + 48);

  ADD2(in29, in28, in30, in31, in17, in18);
  DOTP_CONST_PAIR(in18, in17, cospi_3_64, cospi_29_64, vec5, vec4);
  ST_SH(vec4, out + 56);
  ST_SH(vec5, out + 64);
}

static void fdct32x8_1d_row_rd(int16_t *tmp_buf_big, int16_t *tmp_buf,
                               int16_t *output) {
  fdct8x32_1d_row_load_butterfly(tmp_buf_big, tmp_buf);
  fdct8x32_1d_row_even_rd(tmp_buf, tmp_buf);
  fdct8x32_1d_row_odd_rd((tmp_buf + 128), tmp_buf_big, (tmp_buf + 128));
  fdct8x32_1d_row_transpose_store(tmp_buf, output);
}

void vpx_fdct32x32_rd_msa(const int16_t *input, int16_t *out,
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

void vpx_fdct32x32_1_msa(const int16_t *input, int16_t *out, int32_t stride) {
  int sum, i;
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v4i32 vec_w = { 0 };

  for (i = 0; i < 16; ++i) {
    LD_SH4(input, 8, in0, in1, in2, in3);
    input += stride;
    LD_SH4(input, 8, in4, in5, in6, in7);
    input += stride;
    ADD4(in0, in1, in2, in3, in4, in5, in6, in7, in0, in2, in4, in6);
    ADD2(in0, in2, in4, in6, in0, in4);
    vec_w += __msa_hadd_s_w(in0, in0);
    vec_w += __msa_hadd_s_w(in4, in4);
  }

  sum = HADD_SW_S32(vec_w);
  out[0] = (int16_t)(sum >> 3);
}
