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

void vpx_fdct8x8_1_msa(const int16_t *input, tran_low_t *out, int32_t stride) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v4i32 vec_w;

  LD_SH8(input, stride, in0, in1, in2, in3, in4, in5, in6, in7);
  ADD4(in0, in1, in2, in3, in4, in5, in6, in7, in0, in2, in4, in6);
  ADD2(in0, in2, in4, in6, in0, in4);
  vec_w = __msa_hadd_s_w(in0, in0);
  vec_w += __msa_hadd_s_w(in4, in4);
  out[0] = HADD_SW_S32(vec_w);
  out[1] = 0;
}

#if !CONFIG_VP9_HIGHBITDEPTH
void fdct8x16_1d_column(const int16_t *input, int16_t *tmp_ptr,
                        int32_t src_stride) {
  v8i16 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 in8, in9, in10, in11, in12, in13, in14, in15;
  v8i16 stp21, stp22, stp23, stp24, stp25, stp26, stp30;
  v8i16 stp31, stp32, stp33, stp34, stp35, stp36, stp37;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, cnst0, cnst1, cnst4, cnst5;
  v8i16 coeff = { cospi_16_64, -cospi_16_64, cospi_8_64,  cospi_24_64,
                  -cospi_8_64, -cospi_24_64, cospi_12_64, cospi_20_64 };
  v8i16 coeff1 = { cospi_2_64,  cospi_30_64, cospi_14_64, cospi_18_64,
                   cospi_10_64, cospi_22_64, cospi_6_64,  cospi_26_64 };
  v8i16 coeff2 = {
    -cospi_2_64, -cospi_10_64, -cospi_18_64, -cospi_26_64, 0, 0, 0, 0
  };

  LD_SH16(input, src_stride, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9,
          in10, in11, in12, in13, in14, in15);
  SLLI_4V(in0, in1, in2, in3, 2);
  SLLI_4V(in4, in5, in6, in7, 2);
  SLLI_4V(in8, in9, in10, in11, 2);
  SLLI_4V(in12, in13, in14, in15, 2);
  ADD4(in0, in15, in1, in14, in2, in13, in3, in12, tmp0, tmp1, tmp2, tmp3);
  ADD4(in4, in11, in5, in10, in6, in9, in7, in8, tmp4, tmp5, tmp6, tmp7);
  FDCT8x16_EVEN(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp0, tmp1,
                tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
  ST_SH8(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp_ptr, 32);
  SUB4(in0, in15, in1, in14, in2, in13, in3, in12, in15, in14, in13, in12);
  SUB4(in4, in11, in5, in10, in6, in9, in7, in8, in11, in10, in9, in8);

  tmp_ptr += 16;

  /* stp 1 */
  ILVL_H2_SH(in10, in13, in11, in12, vec2, vec4);
  ILVR_H2_SH(in10, in13, in11, in12, vec3, vec5);

  cnst4 = __msa_splati_h(coeff, 0);
  stp25 = DOT_SHIFT_RIGHT_PCK_H(vec2, vec3, cnst4);

  cnst5 = __msa_splati_h(coeff, 1);
  cnst5 = __msa_ilvev_h(cnst5, cnst4);
  stp22 = DOT_SHIFT_RIGHT_PCK_H(vec2, vec3, cnst5);
  stp24 = DOT_SHIFT_RIGHT_PCK_H(vec4, vec5, cnst4);
  stp23 = DOT_SHIFT_RIGHT_PCK_H(vec4, vec5, cnst5);

  /* stp2 */
  BUTTERFLY_4(in8, in9, stp22, stp23, stp30, stp31, stp32, stp33);
  BUTTERFLY_4(in15, in14, stp25, stp24, stp37, stp36, stp35, stp34);
  ILVL_H2_SH(stp36, stp31, stp35, stp32, vec2, vec4);
  ILVR_H2_SH(stp36, stp31, stp35, stp32, vec3, vec5);
  SPLATI_H2_SH(coeff, 2, 3, cnst0, cnst1);
  cnst0 = __msa_ilvev_h(cnst0, cnst1);
  stp26 = DOT_SHIFT_RIGHT_PCK_H(vec2, vec3, cnst0);

  cnst0 = __msa_splati_h(coeff, 4);
  cnst1 = __msa_ilvev_h(cnst1, cnst0);
  stp21 = DOT_SHIFT_RIGHT_PCK_H(vec2, vec3, cnst1);

  BUTTERFLY_4(stp30, stp37, stp26, stp21, in8, in15, in14, in9);
  ILVRL_H2_SH(in15, in8, vec1, vec0);
  SPLATI_H2_SH(coeff1, 0, 1, cnst0, cnst1);
  cnst0 = __msa_ilvev_h(cnst0, cnst1);

  in8 = DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0);
  ST_SH(in8, tmp_ptr);

  cnst0 = __msa_splati_h(coeff2, 0);
  cnst0 = __msa_ilvev_h(cnst1, cnst0);
  in8 = DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0);
  ST_SH(in8, tmp_ptr + 224);

  ILVRL_H2_SH(in14, in9, vec1, vec0);
  SPLATI_H2_SH(coeff1, 2, 3, cnst0, cnst1);
  cnst1 = __msa_ilvev_h(cnst1, cnst0);

  in8 = DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst1);
  ST_SH(in8, tmp_ptr + 128);

  cnst1 = __msa_splati_h(coeff2, 2);
  cnst0 = __msa_ilvev_h(cnst0, cnst1);
  in8 = DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0);
  ST_SH(in8, tmp_ptr + 96);

  SPLATI_H2_SH(coeff, 2, 5, cnst0, cnst1);
  cnst1 = __msa_ilvev_h(cnst1, cnst0);

  stp25 = DOT_SHIFT_RIGHT_PCK_H(vec4, vec5, cnst1);

  cnst1 = __msa_splati_h(coeff, 3);
  cnst1 = __msa_ilvev_h(cnst0, cnst1);
  stp22 = DOT_SHIFT_RIGHT_PCK_H(vec4, vec5, cnst1);

  /* stp4 */
  ADD2(stp34, stp25, stp33, stp22, in13, in10);

  ILVRL_H2_SH(in13, in10, vec1, vec0);
  SPLATI_H2_SH(coeff1, 4, 5, cnst0, cnst1);
  cnst0 = __msa_ilvev_h(cnst0, cnst1);
  in8 = DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0);
  ST_SH(in8, tmp_ptr + 64);

  cnst0 = __msa_splati_h(coeff2, 1);
  cnst0 = __msa_ilvev_h(cnst1, cnst0);
  in8 = DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0);
  ST_SH(in8, tmp_ptr + 160);

  SUB2(stp34, stp25, stp33, stp22, in12, in11);
  ILVRL_H2_SH(in12, in11, vec1, vec0);
  SPLATI_H2_SH(coeff1, 6, 7, cnst0, cnst1);
  cnst1 = __msa_ilvev_h(cnst1, cnst0);

  in8 = DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst1);
  ST_SH(in8, tmp_ptr + 192);

  cnst1 = __msa_splati_h(coeff2, 3);
  cnst0 = __msa_ilvev_h(cnst0, cnst1);
  in8 = DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0);
  ST_SH(in8, tmp_ptr + 32);
}

void fdct16x8_1d_row(int16_t *input, int16_t *output) {
  v8i16 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 in8, in9, in10, in11, in12, in13, in14, in15;

  LD_SH8(input, 16, in0, in1, in2, in3, in4, in5, in6, in7);
  LD_SH8((input + 8), 16, in8, in9, in10, in11, in12, in13, in14, in15);
  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  TRANSPOSE8x8_SH_SH(in8, in9, in10, in11, in12, in13, in14, in15, in8, in9,
                     in10, in11, in12, in13, in14, in15);
  ADD4(in0, 1, in1, 1, in2, 1, in3, 1, in0, in1, in2, in3);
  ADD4(in4, 1, in5, 1, in6, 1, in7, 1, in4, in5, in6, in7);
  ADD4(in8, 1, in9, 1, in10, 1, in11, 1, in8, in9, in10, in11);
  ADD4(in12, 1, in13, 1, in14, 1, in15, 1, in12, in13, in14, in15);
  SRA_4V(in0, in1, in2, in3, 2);
  SRA_4V(in4, in5, in6, in7, 2);
  SRA_4V(in8, in9, in10, in11, 2);
  SRA_4V(in12, in13, in14, in15, 2);
  BUTTERFLY_16(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
               in12, in13, in14, in15, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6,
               tmp7, in8, in9, in10, in11, in12, in13, in14, in15);
  ST_SH8(in8, in9, in10, in11, in12, in13, in14, in15, input, 16);
  FDCT8x16_EVEN(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp0, tmp1,
                tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
  LD_SH8(input, 16, in8, in9, in10, in11, in12, in13, in14, in15);
  FDCT8x16_ODD(in8, in9, in10, in11, in12, in13, in14, in15, in0, in1, in2, in3,
               in4, in5, in6, in7);
  TRANSPOSE8x8_SH_SH(tmp0, in0, tmp1, in1, tmp2, in2, tmp3, in3, tmp0, in0,
                     tmp1, in1, tmp2, in2, tmp3, in3);
  ST_SH8(tmp0, in0, tmp1, in1, tmp2, in2, tmp3, in3, output, 16);
  TRANSPOSE8x8_SH_SH(tmp4, in4, tmp5, in5, tmp6, in6, tmp7, in7, tmp4, in4,
                     tmp5, in5, tmp6, in6, tmp7, in7);
  ST_SH8(tmp4, in4, tmp5, in5, tmp6, in6, tmp7, in7, output + 8, 16);
}

void vpx_fdct4x4_msa(const int16_t *input, int16_t *output,
                     int32_t src_stride) {
  v8i16 in0, in1, in2, in3;

  LD_SH4(input, src_stride, in0, in1, in2, in3);

  /* fdct4 pre-process */
  {
    v8i16 vec, mask;
    v16i8 zero = { 0 };
    v16i8 one = __msa_ldi_b(1);

    mask = (v8i16)__msa_sldi_b(zero, one, 15);
    SLLI_4V(in0, in1, in2, in3, 4);
    vec = __msa_ceqi_h(in0, 0);
    vec = vec ^ 255;
    vec = mask & vec;
    in0 += vec;
  }

  VP9_FDCT4(in0, in1, in2, in3, in0, in1, in2, in3);
  TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);
  VP9_FDCT4(in0, in1, in2, in3, in0, in1, in2, in3);
  TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);
  ADD4(in0, 1, in1, 1, in2, 1, in3, 1, in0, in1, in2, in3);
  SRA_4V(in0, in1, in2, in3, 2);
  PCKEV_D2_SH(in1, in0, in3, in2, in0, in2);
  ST_SH2(in0, in2, output, 8);
}

void vpx_fdct8x8_msa(const int16_t *input, int16_t *output,
                     int32_t src_stride) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;

  LD_SH8(input, src_stride, in0, in1, in2, in3, in4, in5, in6, in7);
  SLLI_4V(in0, in1, in2, in3, 2);
  SLLI_4V(in4, in5, in6, in7, 2);
  VP9_FDCT8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
            in5, in6, in7);
  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  VP9_FDCT8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
            in5, in6, in7);
  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  SRLI_AVE_S_4V_H(in0, in1, in2, in3, in4, in5, in6, in7);
  ST_SH8(in0, in1, in2, in3, in4, in5, in6, in7, output, 8);
}

void vpx_fdct16x16_msa(const int16_t *input, int16_t *output,
                       int32_t src_stride) {
  int32_t i;
  DECLARE_ALIGNED(32, int16_t, tmp_buf[16 * 16]);

  /* column transform */
  for (i = 0; i < 2; ++i) {
    fdct8x16_1d_column((input + 8 * i), (&tmp_buf[0] + 8 * i), src_stride);
  }

  /* row transform */
  for (i = 0; i < 2; ++i) {
    fdct16x8_1d_row((&tmp_buf[0] + (128 * i)), (output + (128 * i)));
  }
}

void vpx_fdct16x16_1_msa(const int16_t *input, int16_t *out, int32_t stride) {
  int sum, i;
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v4i32 vec_w = { 0 };

  for (i = 0; i < 4; ++i) {
    LD_SH2(input, 8, in0, in1);
    input += stride;
    LD_SH2(input, 8, in2, in3);
    input += stride;
    LD_SH2(input, 8, in4, in5);
    input += stride;
    LD_SH2(input, 8, in6, in7);
    input += stride;
    ADD4(in0, in1, in2, in3, in4, in5, in6, in7, in0, in2, in4, in6);
    ADD2(in0, in2, in4, in6, in0, in4);
    vec_w += __msa_hadd_s_w(in0, in0);
    vec_w += __msa_hadd_s_w(in4, in4);
  }

  sum = HADD_SW_S32(vec_w);
  out[0] = (int16_t)(sum >> 1);
}
#endif  // !CONFIG_VP9_HIGHBITDEPTH
