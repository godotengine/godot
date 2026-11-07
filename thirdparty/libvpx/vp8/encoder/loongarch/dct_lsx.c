/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdint.h>
#include "./vp8_rtcd.h"
#include "vpx_util/loongson_intrinsics.h"

#define LSX_TRANSPOSE4x4_H(_in0, _in1, _in2, _in3, _out0, _out1, _out2, _out3) \
  {                                                                            \
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
  }

#define SET_DOTP_VALUES(coeff, val0, val1, val2, const1, const2)           \
  {                                                                        \
    __m128i tmp0_m, tmp1_m, tmp2_m;                                        \
                                                                           \
    tmp0_m = __lsx_vreplvei_h(coeff, val0);                                \
    DUP2_ARG2(__lsx_vreplvei_h, coeff, val1, coeff, val2, tmp1_m, tmp2_m); \
    DUP2_ARG2(__lsx_vpackev_h, tmp1_m, tmp0_m, tmp0_m, tmp2_m, const1,     \
              const2);                                                     \
  }

#define RET_1_IF_NZERO_H(_in)           \
  ({                                    \
    __m128i tmp_m;                      \
    __m128i one_m = __lsx_vldi(0x401);  \
    __m128i max_m = __lsx_vldi(0xFF);   \
                                        \
    tmp_m = __lsx_vseqi_h(_in, 0);      \
    tmp_m = __lsx_vxor_v(tmp_m, max_m); \
    tmp_m = __lsx_vand_v(tmp_m, one_m); \
                                        \
    tmp_m;                              \
  })

void vp8_short_fdct4x4_lsx(int16_t *input, int16_t *output, int32_t pitch) {
  __m128i in0, in1, in2, in3;
  __m128i tmp0, tmp1, tmp2, tmp3, const0, const1;
  __m128i coeff = { 0x38a4eb1814e808a9, 0x659061a82ee01d4c };
  __m128i out0, out1, out2, out3;
  __m128i zero = __lsx_vldi(0);
  int32_t pitch2 = pitch << 1;
  int32_t pitch3 = pitch2 + pitch;

  in0 = __lsx_vld(input, 0);
  DUP2_ARG2(__lsx_vldx, input, pitch, input, pitch2, in1, in2);
  in3 = __lsx_vldx(input, pitch3);

  LSX_TRANSPOSE4x4_H(in0, in1, in2, in3, in0, in1, in2, in3);
  LSX_BUTTERFLY_4_H(in0, in1, in2, in3, tmp0, tmp1, in1, in3);
  DUP4_ARG2(__lsx_vslli_h, tmp0, 3, tmp1, 3, in1, 3, in3, 3, tmp0, tmp1, in1,
            in3);
  in0 = __lsx_vadd_h(tmp0, tmp1);
  in2 = __lsx_vsub_h(tmp0, tmp1);
  SET_DOTP_VALUES(coeff, 0, 1, 2, const0, const1);
  tmp0 = __lsx_vilvl_h(in3, in1);
  in1 = __lsx_vreplvei_h(coeff, 3);
  out0 = __lsx_vpackev_h(zero, in1);
  coeff = __lsx_vilvl_h(zero, coeff);
  out1 = __lsx_vreplvei_w(coeff, 0);
  DUP2_ARG3(__lsx_vdp2add_w_h, out0, tmp0, const0, out1, tmp0, const1, out0,
            out1);
  DUP2_ARG3(__lsx_vsrani_h_w, out0, out0, 12, out1, out1, 12, in1, in3);
  LSX_TRANSPOSE4x4_H(in0, in1, in2, in3, in0, in1, in2, in3);
  LSX_BUTTERFLY_4_H(in0, in1, in2, in3, tmp0, tmp1, in1, in3);
  tmp2 = __lsx_vadd_h(tmp0, tmp1);
  tmp3 = __lsx_vsub_h(tmp0, tmp1);
  DUP2_ARG2(__lsx_vaddi_hu, tmp2, 7, tmp3, 7, in0, in2);
  DUP2_ARG2(__lsx_vsrai_h, in0, 4, in2, 4, in0, in2);
  DUP2_ARG2(__lsx_vilvl_h, zero, in0, zero, in2, out0, out2);
  tmp1 = RET_1_IF_NZERO_H(in3);
  DUP2_ARG2(__lsx_vilvl_h, zero, tmp1, in3, in1, tmp1, tmp0);
  DUP2_ARG2(__lsx_vreplvei_w, coeff, 2, coeff, 3, out3, out1);
  out3 = __lsx_vadd_w(out3, out1);
  out1 = __lsx_vreplvei_w(coeff, 1);
  DUP2_ARG3(__lsx_vdp2add_w_h, out1, tmp0, const0, out3, tmp0, const1, out1,
            out3);
  DUP2_ARG2(__lsx_vsrai_w, out1, 16, out3, 16, out1, out3);
  out1 = __lsx_vadd_w(out1, tmp1);
  DUP2_ARG2(__lsx_vpickev_h, out1, out0, out3, out2, in0, in2);
  __lsx_vst(in0, output, 0);
  __lsx_vst(in2, output, 16);
}

void vp8_short_fdct8x4_lsx(int16_t *input, int16_t *output, int32_t pitch) {
  __m128i in0, in1, in2, in3, temp0, temp1, tmp0, tmp1;
  __m128i const0, const1, const2, vec0_w, vec1_w, vec2_w, vec3_w;
  __m128i coeff = { 0x38a4eb1814e808a9, 0x659061a82ee01d4c };
  __m128i zero = __lsx_vldi(0);
  int32_t pitch2 = pitch << 1;
  int32_t pitch3 = pitch2 + pitch;

  in0 = __lsx_vld(input, 0);
  DUP2_ARG2(__lsx_vldx, input, pitch, input, pitch2, in1, in2);
  in3 = __lsx_vldx(input, pitch3);
  LSX_TRANSPOSE4x4_H(in0, in1, in2, in3, in0, in1, in2, in3);

  LSX_BUTTERFLY_4_H(in0, in1, in2, in3, temp0, temp1, in1, in3);
  DUP4_ARG2(__lsx_vslli_h, temp0, 3, temp1, 3, in1, 3, in3, 3, temp0, temp1,
            in1, in3);
  in0 = __lsx_vadd_h(temp0, temp1);
  in2 = __lsx_vsub_h(temp0, temp1);
  SET_DOTP_VALUES(coeff, 0, 1, 2, const1, const2);
  temp0 = __lsx_vreplvei_h(coeff, 3);
  vec1_w = __lsx_vpackev_h(zero, temp0);
  coeff = __lsx_vilvh_h(zero, coeff);
  vec3_w = __lsx_vreplvei_w(coeff, 0);
  tmp1 = __lsx_vilvl_h(in3, in1);
  tmp0 = __lsx_vilvh_h(in3, in1);
  vec0_w = vec1_w;
  vec2_w = vec3_w;
  DUP4_ARG3(__lsx_vdp2add_w_h, vec0_w, tmp1, const1, vec1_w, tmp0, const1,
            vec2_w, tmp1, const2, vec3_w, tmp0, const2, vec0_w, vec1_w, vec2_w,
            vec3_w);
  DUP2_ARG3(__lsx_vsrani_h_w, vec1_w, vec0_w, 12, vec3_w, vec2_w, 12, in1, in3);
  LSX_TRANSPOSE4x4_H(in0, in1, in2, in3, in0, in1, in2, in3);

  LSX_BUTTERFLY_4_H(in0, in1, in2, in3, temp0, temp1, in1, in3);
  in0 = __lsx_vadd_h(temp0, temp1);
  in0 = __lsx_vaddi_hu(in0, 7);
  in2 = __lsx_vsub_h(temp0, temp1);
  in2 = __lsx_vaddi_hu(in2, 7);
  in0 = __lsx_vsrai_h(in0, 4);
  in2 = __lsx_vsrai_h(in2, 4);
  DUP2_ARG2(__lsx_vreplvei_w, coeff, 2, coeff, 3, vec3_w, vec1_w);
  vec3_w = __lsx_vadd_w(vec3_w, vec1_w);
  vec1_w = __lsx_vreplvei_w(coeff, 1);
  const0 = RET_1_IF_NZERO_H(in3);
  tmp1 = __lsx_vilvl_h(in3, in1);
  tmp0 = __lsx_vilvh_h(in3, in1);
  vec0_w = vec1_w;
  vec2_w = vec3_w;
  DUP4_ARG3(__lsx_vdp2add_w_h, vec0_w, tmp1, const1, vec1_w, tmp0, const1,
            vec2_w, tmp1, const2, vec3_w, tmp0, const2, vec0_w, vec1_w, vec2_w,
            vec3_w);
  DUP2_ARG3(__lsx_vsrani_h_w, vec1_w, vec0_w, 16, vec3_w, vec2_w, 16, in1, in3);
  in1 = __lsx_vadd_h(in1, const0);
  DUP2_ARG2(__lsx_vpickev_d, in1, in0, in3, in2, temp0, temp1);
  __lsx_vst(temp0, output, 0);
  __lsx_vst(temp1, output, 16);

  DUP2_ARG2(__lsx_vpickod_d, in1, in0, in3, in2, in0, in2);
  __lsx_vst(in0, output, 32);
  __lsx_vst(in2, output, 48);
}
