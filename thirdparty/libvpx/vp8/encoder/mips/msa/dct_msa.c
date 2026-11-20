/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"
#include "vp8/common/mips/msa/vp8_macros_msa.h"

#define TRANSPOSE4x4_H(in0, in1, in2, in3, out0, out1, out2, out3) \
  {                                                                \
    v8i16 s0_m, s1_m, tp0_m, tp1_m, tp2_m, tp3_m;                  \
                                                                   \
    ILVR_H2_SH(in2, in0, in3, in1, s0_m, s1_m);                    \
    ILVRL_H2_SH(s1_m, s0_m, tp0_m, tp1_m);                         \
    ILVL_H2_SH(in2, in0, in3, in1, s0_m, s1_m);                    \
    ILVRL_H2_SH(s1_m, s0_m, tp2_m, tp3_m);                         \
    PCKEV_D2_SH(tp2_m, tp0_m, tp3_m, tp1_m, out0, out2);           \
    PCKOD_D2_SH(tp2_m, tp0_m, tp3_m, tp1_m, out1, out3);           \
  }

#define SET_DOTP_VALUES(coeff, val0, val1, val2, const1, const2)   \
  {                                                                \
    v8i16 tmp0_m;                                                  \
                                                                   \
    SPLATI_H3_SH(coeff, val0, val1, val2, tmp0_m, const1, const2); \
    ILVEV_H2_SH(tmp0_m, const1, const2, tmp0_m, const1, const2);   \
  }

#define RET_1_IF_NZERO_H(in0)      \
  ({                               \
    v8i16 tmp0_m;                  \
    v8i16 one_m = __msa_ldi_h(1);  \
                                   \
    tmp0_m = __msa_ceqi_h(in0, 0); \
    tmp0_m = tmp0_m ^ 255;         \
    tmp0_m = one_m & tmp0_m;       \
                                   \
    tmp0_m;                        \
  })

#define RET_1_IF_NZERO_W(in0)      \
  ({                               \
    v4i32 tmp0_m;                  \
    v4i32 one_m = __msa_ldi_w(1);  \
                                   \
    tmp0_m = __msa_ceqi_w(in0, 0); \
    tmp0_m = tmp0_m ^ 255;         \
    tmp0_m = one_m & tmp0_m;       \
                                   \
    tmp0_m;                        \
  })

#define RET_1_IF_NEG_W(in0)          \
  ({                                 \
    v4i32 tmp0_m;                    \
                                     \
    v4i32 one_m = __msa_ldi_w(1);    \
    tmp0_m = __msa_clti_s_w(in0, 0); \
    tmp0_m = one_m & tmp0_m;         \
                                     \
    tmp0_m;                          \
  })

void vp8_short_fdct4x4_msa(int16_t *input, int16_t *output, int32_t pitch) {
  v8i16 in0, in1, in2, in3;
  v8i16 temp0, temp1;
  v8i16 const0, const1;
  v8i16 coeff = { 2217, 5352, -5352, 14500, 7500, 12000, 25000, 26000 };
  v4i32 out0, out1, out2, out3;
  v8i16 zero = { 0 };

  LD_SH4(input, pitch / 2, in0, in1, in2, in3);
  TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);

  BUTTERFLY_4(in0, in1, in2, in3, temp0, temp1, in1, in3);
  SLLI_4V(temp0, temp1, in1, in3, 3);
  in0 = temp0 + temp1;
  in2 = temp0 - temp1;
  SET_DOTP_VALUES(coeff, 0, 1, 2, const0, const1);
  temp0 = __msa_ilvr_h(in3, in1);
  in1 = __msa_splati_h(coeff, 3);
  out0 = (v4i32)__msa_ilvev_h(zero, in1);
  coeff = __msa_ilvl_h(zero, coeff);
  out1 = __msa_splati_w((v4i32)coeff, 0);
  DPADD_SH2_SW(temp0, temp0, const0, const1, out0, out1);
  out0 >>= 12;
  out1 >>= 12;
  PCKEV_H2_SH(out0, out0, out1, out1, in1, in3);
  TRANSPOSE4x4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);

  BUTTERFLY_4(in0, in1, in2, in3, temp0, temp1, in1, in3);
  in0 = temp0 + temp1 + 7;
  in2 = temp0 - temp1 + 7;
  in0 >>= 4;
  in2 >>= 4;
  ILVR_H2_SW(zero, in0, zero, in2, out0, out2);
  temp1 = RET_1_IF_NZERO_H(in3);
  ILVR_H2_SH(zero, temp1, in3, in1, temp1, temp0);
  SPLATI_W2_SW(coeff, 2, out3, out1);
  out3 += out1;
  out1 = __msa_splati_w((v4i32)coeff, 1);
  DPADD_SH2_SW(temp0, temp0, const0, const1, out1, out3);
  out1 >>= 16;
  out3 >>= 16;
  out1 += (v4i32)temp1;
  PCKEV_H2_SH(out1, out0, out3, out2, in0, in2);
  ST_SH2(in0, in2, output, 8);
}

void vp8_short_fdct8x4_msa(int16_t *input, int16_t *output, int32_t pitch) {
  v8i16 in0, in1, in2, in3;
  v8i16 temp0, temp1, tmp0, tmp1;
  v8i16 const0, const1, const2;
  v8i16 coeff = { 2217, 5352, -5352, 14500, 7500, 12000, 25000, 26000 };
  v8i16 zero = { 0 };
  v4i32 vec0_w, vec1_w, vec2_w, vec3_w;

  LD_SH4(input, pitch / 2, in0, in1, in2, in3);
  TRANSPOSE4x4_H(in0, in1, in2, in3, in0, in1, in2, in3);

  BUTTERFLY_4(in0, in1, in2, in3, temp0, temp1, in1, in3);
  SLLI_4V(temp0, temp1, in1, in3, 3);
  in0 = temp0 + temp1;
  in2 = temp0 - temp1;
  SET_DOTP_VALUES(coeff, 0, 1, 2, const1, const2);
  temp0 = __msa_splati_h(coeff, 3);
  vec1_w = (v4i32)__msa_ilvev_h(zero, temp0);
  coeff = __msa_ilvl_h(zero, coeff);
  vec3_w = __msa_splati_w((v4i32)coeff, 0);
  ILVRL_H2_SH(in3, in1, tmp1, tmp0);
  vec0_w = vec1_w;
  vec2_w = vec3_w;
  DPADD_SH4_SW(tmp1, tmp0, tmp1, tmp0, const1, const1, const2, const2, vec0_w,
               vec1_w, vec2_w, vec3_w);
  SRA_4V(vec1_w, vec0_w, vec3_w, vec2_w, 12);
  PCKEV_H2_SH(vec1_w, vec0_w, vec3_w, vec2_w, in1, in3);
  TRANSPOSE4x4_H(in0, in1, in2, in3, in0, in1, in2, in3);

  BUTTERFLY_4(in0, in1, in2, in3, temp0, temp1, in1, in3);
  in0 = temp0 + temp1 + 7;
  in2 = temp0 - temp1 + 7;
  in0 >>= 4;
  in2 >>= 4;
  SPLATI_W2_SW(coeff, 2, vec3_w, vec1_w);
  vec3_w += vec1_w;
  vec1_w = __msa_splati_w((v4i32)coeff, 1);
  const0 = RET_1_IF_NZERO_H(in3);
  ILVRL_H2_SH(in3, in1, tmp1, tmp0);
  vec0_w = vec1_w;
  vec2_w = vec3_w;
  DPADD_SH4_SW(tmp1, tmp0, tmp1, tmp0, const1, const1, const2, const2, vec0_w,
               vec1_w, vec2_w, vec3_w);
  SRA_4V(vec1_w, vec0_w, vec3_w, vec2_w, 16);
  PCKEV_H2_SH(vec1_w, vec0_w, vec3_w, vec2_w, in1, in3);
  in1 += const0;
  PCKEV_D2_SH(in1, in0, in3, in2, temp0, temp1);
  ST_SH2(temp0, temp1, output, 8);

  PCKOD_D2_SH(in1, in0, in3, in2, in0, in2);
  ST_SH2(in0, in2, output + 16, 8);
}

void vp8_short_walsh4x4_msa(int16_t *input, int16_t *output, int32_t pitch) {
  v8i16 in0_h, in1_h, in2_h, in3_h;
  v4i32 in0_w, in1_w, in2_w, in3_w, temp0, temp1, temp2, temp3;

  LD_SH4(input, pitch / 2, in0_h, in1_h, in2_h, in3_h);
  TRANSPOSE4x4_SH_SH(in0_h, in1_h, in2_h, in3_h, in0_h, in1_h, in2_h, in3_h);

  UNPCK_R_SH_SW(in0_h, in0_w);
  UNPCK_R_SH_SW(in1_h, in1_w);
  UNPCK_R_SH_SW(in2_h, in2_w);
  UNPCK_R_SH_SW(in3_h, in3_w);
  BUTTERFLY_4(in0_w, in1_w, in3_w, in2_w, temp0, temp3, temp2, temp1);
  SLLI_4V(temp0, temp1, temp2, temp3, 2);
  BUTTERFLY_4(temp0, temp1, temp2, temp3, in0_w, in1_w, in2_w, in3_w);
  temp0 = RET_1_IF_NZERO_W(temp0);
  in0_w += temp0;
  TRANSPOSE4x4_SW_SW(in0_w, in1_w, in2_w, in3_w, in0_w, in1_w, in2_w, in3_w);

  BUTTERFLY_4(in0_w, in1_w, in3_w, in2_w, temp0, temp3, temp2, temp1);
  BUTTERFLY_4(temp0, temp1, temp2, temp3, in0_w, in1_w, in2_w, in3_w);
  in0_w += RET_1_IF_NEG_W(in0_w);
  in1_w += RET_1_IF_NEG_W(in1_w);
  in2_w += RET_1_IF_NEG_W(in2_w);
  in3_w += RET_1_IF_NEG_W(in3_w);
  ADD4(in0_w, 3, in1_w, 3, in2_w, 3, in3_w, 3, in0_w, in1_w, in2_w, in3_w);
  SRA_4V(in0_w, in1_w, in2_w, in3_w, 3);
  PCKEV_H2_SH(in1_w, in0_w, in3_w, in2_w, in0_h, in1_h);
  ST_SH2(in0_h, in1_h, output, 8);
}
