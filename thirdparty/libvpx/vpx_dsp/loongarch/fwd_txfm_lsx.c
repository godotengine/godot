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

#define LSX_TRANSPOSE4x4_H(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  do {                                                                         \
    __m128i _s0, _s1, _s2, _s3, _t0, _t1, _t2, _t3;                            \
                                                                               \
    DUP2_ARG2(__lsx_vilvl_h, _in2, _in0, _in3, _in1, _s0, _s1);                \
    DUP2_ARG2(__lsx_vilvh_h, _in2, _in0, _in3, _in1, _s2, _s3);                \
    _t0 = __lsx_vilvl_h(_s1, _s0);                                             \
    _t1 = __lsx_vilvh_h(_s1, _s0);                                             \
    _t2 = __lsx_vilvl_h(_s3, _s2);                                             \
    _t3 = __lsx_vilvh_h(_s3, _s2);                                             \
    DUP2_ARG2(__lsx_vpickev_d, _t2, _t0, _t3, _t1, _out0, _out2);              \
    DUP2_ARG2(__lsx_vpickod_d, _t2, _t0, _t3, _t1, _out1, _out3);              \
  } while (0)

#if !CONFIG_VP9_HIGHBITDEPTH
void fdct8x16_1d_column(const int16_t *input, int16_t *tmp_ptr,
                        int32_t src_stride) {
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i in8, in9, in10, in11, in12, in13, in14, in15;
  __m128i stp21, stp22, stp23, stp24, stp25, stp26, stp30;
  __m128i stp31, stp32, stp33, stp34, stp35, stp36, stp37;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, cnst0, cnst1, cnst4, cnst5;
  __m128i coeff = { 0x187e3b21d2bf2d41, 0x238e3537e782c4df };
  __m128i coeff1 = { 0x289a317906463fb1, 0x12943d3f1e2b3871 };
  __m128i coeff2 = { 0xed6cd766c78fc04f, 0x0 };

  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t src_stride6 = src_stride4 + src_stride2;
  int32_t src_stride8 = src_stride4 << 1;
  int16_t *input_tmp = (int16_t *)input;
  in0 = __lsx_vld(input_tmp, 0);
  DUP4_ARG2(__lsx_vldx, input_tmp, src_stride2, input_tmp, src_stride4,
            input_tmp, src_stride6, input_tmp, src_stride8, in1, in2, in3, in4);
  input_tmp += src_stride4;
  DUP4_ARG2(__lsx_vldx, input_tmp, src_stride2, input_tmp, src_stride4,
            input_tmp, src_stride6, input_tmp, src_stride8, in5, in6, in7, in8);
  input_tmp += src_stride4;
  DUP4_ARG2(__lsx_vldx, input_tmp, src_stride2, input_tmp, src_stride4,
            input_tmp, src_stride6, input_tmp, src_stride8, in9, in10, in11,
            in12);
  input_tmp += src_stride4;
  DUP2_ARG2(__lsx_vldx, input_tmp, src_stride2, input_tmp, src_stride4, in13,
            in14);
  input_tmp += src_stride2;
  in15 = __lsx_vldx(input_tmp, src_stride2);

  DUP4_ARG2(__lsx_vslli_h, in0, 2, in1, 2, in2, 2, in3, 2, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vslli_h, in4, 2, in5, 2, in6, 2, in7, 2, in4, in5, in6, in7);
  DUP4_ARG2(__lsx_vslli_h, in8, 2, in9, 2, in10, 2, in11, 2, in8, in9, in10,
            in11);
  DUP4_ARG2(__lsx_vslli_h, in12, 2, in13, 2, in14, 2, in15, 2, in12, in13, in14,
            in15);
  DUP4_ARG2(__lsx_vadd_h, in0, in15, in1, in14, in2, in13, in3, in12, tmp0,
            tmp1, tmp2, tmp3);
  DUP4_ARG2(__lsx_vadd_h, in4, in11, in5, in10, in6, in9, in7, in8, tmp4, tmp5,
            tmp6, tmp7);
  FDCT8x16_EVEN(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp0, tmp1,
                tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
  __lsx_vst(tmp0, tmp_ptr, 0);
  __lsx_vst(tmp1, tmp_ptr, 64);
  __lsx_vst(tmp2, tmp_ptr, 128);
  __lsx_vst(tmp3, tmp_ptr, 192);
  __lsx_vst(tmp4, tmp_ptr, 256);
  __lsx_vst(tmp5, tmp_ptr, 320);
  __lsx_vst(tmp6, tmp_ptr, 384);
  __lsx_vst(tmp7, tmp_ptr, 448);
  DUP4_ARG2(__lsx_vsub_h, in0, in15, in1, in14, in2, in13, in3, in12, in15,
            in14, in13, in12);
  DUP4_ARG2(__lsx_vsub_h, in4, in11, in5, in10, in6, in9, in7, in8, in11, in10,
            in9, in8);

  tmp_ptr += 16;

  /* stp 1 */
  DUP2_ARG2(__lsx_vilvh_h, in10, in13, in11, in12, vec2, vec4);
  DUP2_ARG2(__lsx_vilvl_h, in10, in13, in11, in12, vec3, vec5);

  cnst4 = __lsx_vreplvei_h(coeff, 0);
  DOT_SHIFT_RIGHT_PCK_H(vec2, vec3, cnst4, stp25);

  cnst5 = __lsx_vreplvei_h(coeff, 1);
  cnst5 = __lsx_vpackev_h(cnst5, cnst4);
  DOT_SHIFT_RIGHT_PCK_H(vec2, vec3, cnst5, stp22);
  DOT_SHIFT_RIGHT_PCK_H(vec4, vec5, cnst4, stp24);
  DOT_SHIFT_RIGHT_PCK_H(vec4, vec5, cnst5, stp23);

  /* stp2 */
  LSX_BUTTERFLY_4_H(in8, in9, stp22, stp23, stp30, stp31, stp32, stp33);
  LSX_BUTTERFLY_4_H(in15, in14, stp25, stp24, stp37, stp36, stp35, stp34);
  DUP2_ARG2(__lsx_vilvh_h, stp36, stp31, stp35, stp32, vec2, vec4);
  DUP2_ARG2(__lsx_vilvl_h, stp36, stp31, stp35, stp32, vec3, vec5);
  DUP2_ARG2(__lsx_vreplvei_h, coeff, 2, coeff, 3, cnst0, cnst1);
  cnst0 = __lsx_vpackev_h(cnst0, cnst1);
  DOT_SHIFT_RIGHT_PCK_H(vec2, vec3, cnst0, stp26);

  cnst0 = __lsx_vreplvei_h(coeff, 4);
  cnst1 = __lsx_vpackev_h(cnst1, cnst0);
  DOT_SHIFT_RIGHT_PCK_H(vec2, vec3, cnst1, stp21);

  LSX_BUTTERFLY_4_H(stp30, stp37, stp26, stp21, in8, in15, in14, in9);
  vec1 = __lsx_vilvl_h(in15, in8);
  vec0 = __lsx_vilvh_h(in15, in8);

  DUP2_ARG2(__lsx_vreplvei_h, coeff1, 0, coeff1, 1, cnst0, cnst1);
  cnst0 = __lsx_vpackev_h(cnst0, cnst1);

  DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0, in8);
  __lsx_vst(in8, tmp_ptr, 0);

  cnst0 = __lsx_vreplvei_h(coeff2, 0);
  cnst0 = __lsx_vpackev_h(cnst1, cnst0);
  DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0, in8);
  __lsx_vst(in8, tmp_ptr, 448);

  vec1 = __lsx_vilvl_h(in14, in9);
  vec0 = __lsx_vilvh_h(in14, in9);
  DUP2_ARG2(__lsx_vreplvei_h, coeff1, 2, coeff1, 3, cnst0, cnst1);
  cnst1 = __lsx_vpackev_h(cnst1, cnst0);

  DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst1, in8);
  __lsx_vst(in8, tmp_ptr, 256);

  cnst1 = __lsx_vreplvei_h(coeff2, 2);
  cnst0 = __lsx_vpackev_h(cnst0, cnst1);
  DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0, in8);
  __lsx_vst(in8, tmp_ptr, 192);

  DUP2_ARG2(__lsx_vreplvei_h, coeff, 2, coeff, 5, cnst0, cnst1);
  cnst1 = __lsx_vpackev_h(cnst1, cnst0);
  DOT_SHIFT_RIGHT_PCK_H(vec4, vec5, cnst1, stp25);

  cnst1 = __lsx_vreplvei_h(coeff, 3);
  cnst1 = __lsx_vpackev_h(cnst0, cnst1);
  DOT_SHIFT_RIGHT_PCK_H(vec4, vec5, cnst1, stp22);

  /* stp4 */
  DUP2_ARG2(__lsx_vadd_h, stp34, stp25, stp33, stp22, in13, in10);

  vec1 = __lsx_vilvl_h(in13, in10);
  vec0 = __lsx_vilvh_h(in13, in10);
  DUP2_ARG2(__lsx_vreplvei_h, coeff1, 4, coeff1, 5, cnst0, cnst1);
  cnst0 = __lsx_vpackev_h(cnst0, cnst1);
  DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0, in8);
  __lsx_vst(in8, tmp_ptr, 128);

  cnst0 = __lsx_vreplvei_h(coeff2, 1);
  cnst0 = __lsx_vpackev_h(cnst1, cnst0);
  DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0, in8);
  __lsx_vst(in8, tmp_ptr, 320);

  DUP2_ARG2(__lsx_vsub_h, stp34, stp25, stp33, stp22, in12, in11);
  vec1 = __lsx_vilvl_h(in12, in11);
  vec0 = __lsx_vilvh_h(in12, in11);
  DUP2_ARG2(__lsx_vreplvei_h, coeff1, 6, coeff1, 7, cnst0, cnst1);
  cnst1 = __lsx_vpackev_h(cnst1, cnst0);

  DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst1, in8);
  __lsx_vst(in8, tmp_ptr, 384);

  cnst1 = __lsx_vreplvei_h(coeff2, 3);
  cnst0 = __lsx_vpackev_h(cnst0, cnst1);
  DOT_SHIFT_RIGHT_PCK_H(vec0, vec1, cnst0, in8);
  __lsx_vst(in8, tmp_ptr, 64);
}

void fdct16x8_1d_row(int16_t *input, int16_t *output) {
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  __m128i in8, in9, in10, in11, in12, in13, in14, in15;
  int16_t *input_tmp = input;

  DUP4_ARG2(__lsx_vld, input, 0, input, 32, input, 64, input, 96, in0, in1, in2,
            in3);
  DUP4_ARG2(__lsx_vld, input, 128, input, 160, input, 192, input, 224, in4, in5,
            in6, in7);
  DUP4_ARG2(__lsx_vld, input_tmp, 16, input_tmp, 48, input_tmp, 80, input_tmp,
            112, in8, in9, in10, in11);
  DUP4_ARG2(__lsx_vld, input_tmp, 144, input_tmp, 176, input_tmp, 208,
            input_tmp, 240, in12, in13, in14, in15);

  LSX_TRANSPOSE8x8_H(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  LSX_TRANSPOSE8x8_H(in8, in9, in10, in11, in12, in13, in14, in15, in8, in9,
                     in10, in11, in12, in13, in14, in15);
  DUP4_ARG2(__lsx_vaddi_hu, in0, 1, in1, 1, in2, 1, in3, 1, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vaddi_hu, in4, 1, in5, 1, in6, 1, in7, 1, in4, in5, in6, in7);
  DUP4_ARG2(__lsx_vaddi_hu, in8, 1, in9, 1, in10, 1, in11, 1, in8, in9, in10,
            in11);
  DUP4_ARG2(__lsx_vaddi_hu, in12, 1, in13, 1, in14, 1, in15, 1, in12, in13,
            in14, in15);

  DUP4_ARG2(__lsx_vsrai_h, in0, 2, in1, 2, in2, 2, in3, 2, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vsrai_h, in4, 2, in5, 2, in6, 2, in7, 2, in4, in5, in6, in7);
  DUP4_ARG2(__lsx_vsrai_h, in8, 2, in9, 2, in10, 2, in11, 2, in8, in9, in10,
            in11);
  DUP4_ARG2(__lsx_vsrai_h, in12, 2, in13, 2, in14, 2, in15, 2, in12, in13, in14,
            in15);
  LSX_BUTTERFLY_16_H(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                     in11, in12, in13, in14, in15, tmp0, tmp1, tmp2, tmp3, tmp4,
                     tmp5, tmp6, tmp7, in8, in9, in10, in11, in12, in13, in14,
                     in15);
  __lsx_vst(in8, input, 0);
  __lsx_vst(in9, input, 32);
  __lsx_vst(in10, input, 64);
  __lsx_vst(in11, input, 96);
  __lsx_vst(in12, input, 128);
  __lsx_vst(in13, input, 160);
  __lsx_vst(in14, input, 192);
  __lsx_vst(in15, input, 224);

  FDCT8x16_EVEN(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp0, tmp1,
                tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
  DUP4_ARG2(__lsx_vld, input, 0, input, 32, input, 64, input, 96, in8, in9,
            in10, in11);
  DUP4_ARG2(__lsx_vld, input, 128, input, 160, input, 192, input, 224, in12,
            in13, in14, in15);
  FDCT8x16_ODD(in8, in9, in10, in11, in12, in13, in14, in15, in0, in1, in2, in3,
               in4, in5, in6, in7);
  LSX_TRANSPOSE8x8_H(tmp0, in0, tmp1, in1, tmp2, in2, tmp3, in3, tmp0, in0,
                     tmp1, in1, tmp2, in2, tmp3, in3);
  __lsx_vst(tmp0, output, 0);
  __lsx_vst(in0, output, 32);
  __lsx_vst(tmp1, output, 64);
  __lsx_vst(in1, output, 96);
  __lsx_vst(tmp2, output, 128);
  __lsx_vst(in2, output, 160);
  __lsx_vst(tmp3, output, 192);
  __lsx_vst(in3, output, 224);

  LSX_TRANSPOSE8x8_H(tmp4, in4, tmp5, in5, tmp6, in6, tmp7, in7, tmp4, in4,
                     tmp5, in5, tmp6, in6, tmp7, in7);
  __lsx_vst(tmp4, output, 16);
  __lsx_vst(in4, output, 48);
  __lsx_vst(tmp5, output, 80);
  __lsx_vst(in5, output, 112);
  __lsx_vst(tmp6, output, 144);
  __lsx_vst(in6, output, 176);
  __lsx_vst(tmp7, output, 208);
  __lsx_vst(in7, output, 240);
}

void vpx_fdct4x4_lsx(const int16_t *input, int16_t *output,
                     int32_t src_stride) {
  __m128i in0, in1, in2, in3;

  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t src_stride6 = src_stride4 + src_stride2;

  in0 = __lsx_vld(input, 0);
  DUP2_ARG2(__lsx_vldx, input, src_stride2, input, src_stride4, in1, in2);
  in3 = __lsx_vldx(input, src_stride6);

  /* fdct4 pre-process */
  {
    __m128i vec, mask;
    __m128i zero = __lsx_vldi(0);

    mask = __lsx_vinsgr2vr_b(zero, 1, 0);
    DUP4_ARG2(__lsx_vslli_h, in0, 4, in1, 4, in2, 4, in3, 4, in0, in1, in2,
              in3);
    vec = __lsx_vseqi_h(in0, 0);
    vec = __lsx_vxori_b(vec, 255);
    vec = __lsx_vand_v(mask, vec);
    in0 = __lsx_vadd_h(in0, vec);
  }

  VP9_FDCT4(in0, in1, in2, in3, in0, in1, in2, in3);
  LSX_TRANSPOSE4x4_H(in0, in1, in2, in3, in0, in1, in2, in3);
  VP9_FDCT4(in0, in1, in2, in3, in0, in1, in2, in3);
  LSX_TRANSPOSE4x4_H(in0, in1, in2, in3, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vaddi_hu, in0, 1, in1, 1, in2, 1, in3, 1, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vsrai_h, in0, 2, in1, 2, in2, 2, in3, 2, in0, in1, in2, in3);
  DUP2_ARG2(__lsx_vpickev_d, in1, in0, in3, in2, in0, in2);
  __lsx_vst(in0, output, 0);
  __lsx_vst(in2, output, 16);
}

void vpx_fdct8x8_lsx(const int16_t *input, int16_t *output,
                     int32_t src_stride) {
  __m128i in0, in1, in2, in3, in4, in5, in6, in7;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t src_stride6 = src_stride4 + src_stride2;
  int16_t *input_tmp = (int16_t *)input;

  in0 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, src_stride2, input_tmp, src_stride4, in1,
            in2);
  in3 = __lsx_vldx(input_tmp, src_stride6);
  input_tmp += src_stride4;
  in4 = __lsx_vld(input_tmp, 0);
  DUP2_ARG2(__lsx_vldx, input_tmp, src_stride2, input_tmp, src_stride4, in5,
            in6);
  in7 = __lsx_vldx(input_tmp, src_stride6);

  DUP4_ARG2(__lsx_vslli_h, in0, 2, in1, 2, in2, 2, in3, 2, in0, in1, in2, in3);
  DUP4_ARG2(__lsx_vslli_h, in4, 2, in5, 2, in6, 2, in7, 2, in4, in5, in6, in7);

  VP9_FDCT8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
            in5, in6, in7);
  LSX_TRANSPOSE8x8_H(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  VP9_FDCT8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
            in5, in6, in7);
  LSX_TRANSPOSE8x8_H(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  SRLI_AVE_S_4V_H(in0, in1, in2, in3, in4, in5, in6, in7);

  __lsx_vst(in0, output, 0);
  __lsx_vst(in1, output, 16);
  __lsx_vst(in2, output, 32);
  __lsx_vst(in3, output, 48);
  __lsx_vst(in4, output, 64);
  __lsx_vst(in5, output, 80);
  __lsx_vst(in6, output, 96);
  __lsx_vst(in7, output, 112);
}

void vpx_fdct16x16_lsx(const int16_t *input, int16_t *output,
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
#endif  // !CONFIG_VP9_HIGHBITDEPTH
