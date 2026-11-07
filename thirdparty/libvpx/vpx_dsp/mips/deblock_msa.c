/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/macros_msa.h"

extern const int16_t vpx_rv[];

#define VPX_TRANSPOSE8x16_UB_UB(                                            \
    in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, out3, out4,   \
    out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15) \
  {                                                                         \
    v8i16 temp0, temp1, temp2, temp3, temp4;                                \
    v8i16 temp5, temp6, temp7, temp8, temp9;                                \
                                                                            \
    ILVR_B4_SH(in1, in0, in3, in2, in5, in4, in7, in6, temp0, temp1, temp2, \
               temp3);                                                      \
    ILVR_H2_SH(temp1, temp0, temp3, temp2, temp4, temp5);                   \
    ILVRL_W2_SH(temp5, temp4, temp6, temp7);                                \
    ILVL_H2_SH(temp1, temp0, temp3, temp2, temp4, temp5);                   \
    ILVRL_W2_SH(temp5, temp4, temp8, temp9);                                \
    ILVL_B4_SH(in1, in0, in3, in2, in5, in4, in7, in6, temp0, temp1, temp2, \
               temp3);                                                      \
    ILVR_H2_SH(temp1, temp0, temp3, temp2, temp4, temp5);                   \
    ILVRL_W2_UB(temp5, temp4, out8, out10);                                 \
    ILVL_H2_SH(temp1, temp0, temp3, temp2, temp4, temp5);                   \
    ILVRL_W2_UB(temp5, temp4, out12, out14);                                \
    out0 = (v16u8)temp6;                                                    \
    out2 = (v16u8)temp7;                                                    \
    out4 = (v16u8)temp8;                                                    \
    out6 = (v16u8)temp9;                                                    \
    out9 = (v16u8)__msa_ilvl_d((v2i64)out8, (v2i64)out8);                   \
    out11 = (v16u8)__msa_ilvl_d((v2i64)out10, (v2i64)out10);                \
    out13 = (v16u8)__msa_ilvl_d((v2i64)out12, (v2i64)out12);                \
    out15 = (v16u8)__msa_ilvl_d((v2i64)out14, (v2i64)out14);                \
    out1 = (v16u8)__msa_ilvl_d((v2i64)out0, (v2i64)out0);                   \
    out3 = (v16u8)__msa_ilvl_d((v2i64)out2, (v2i64)out2);                   \
    out5 = (v16u8)__msa_ilvl_d((v2i64)out4, (v2i64)out4);                   \
    out7 = (v16u8)__msa_ilvl_d((v2i64)out6, (v2i64)out6);                   \
  }

#define VPX_AVER_IF_RETAIN(above2_in, above1_in, src_in, below1_in, below2_in, \
                           ref, out)                                           \
  {                                                                            \
    v16u8 temp0, temp1;                                                        \
                                                                               \
    temp1 = __msa_aver_u_b(above2_in, above1_in);                              \
    temp0 = __msa_aver_u_b(below2_in, below1_in);                              \
    temp1 = __msa_aver_u_b(temp1, temp0);                                      \
    out = __msa_aver_u_b(src_in, temp1);                                       \
    temp0 = __msa_asub_u_b(src_in, above2_in);                                 \
    temp1 = __msa_asub_u_b(src_in, above1_in);                                 \
    temp0 = (temp0 < ref);                                                     \
    temp1 = (temp1 < ref);                                                     \
    temp0 = temp0 & temp1;                                                     \
    temp1 = __msa_asub_u_b(src_in, below1_in);                                 \
    temp1 = (temp1 < ref);                                                     \
    temp0 = temp0 & temp1;                                                     \
    temp1 = __msa_asub_u_b(src_in, below2_in);                                 \
    temp1 = (temp1 < ref);                                                     \
    temp0 = temp0 & temp1;                                                     \
    out = __msa_bmz_v(out, src_in, temp0);                                     \
  }

#define TRANSPOSE12x16_B(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9,    \
                         in10, in11, in12, in13, in14, in15)                  \
  {                                                                           \
    v8i16 temp0, temp1, temp2, temp3, temp4;                                  \
    v8i16 temp5, temp6, temp7, temp8, temp9;                                  \
                                                                              \
    ILVR_B2_SH(in1, in0, in3, in2, temp0, temp1);                             \
    ILVRL_H2_SH(temp1, temp0, temp2, temp3);                                  \
    ILVR_B2_SH(in5, in4, in7, in6, temp0, temp1);                             \
    ILVRL_H2_SH(temp1, temp0, temp4, temp5);                                  \
    ILVRL_W2_SH(temp4, temp2, temp0, temp1);                                  \
    ILVRL_W2_SH(temp5, temp3, temp2, temp3);                                  \
    ILVR_B2_SH(in9, in8, in11, in10, temp4, temp5);                           \
    ILVR_B2_SH(in9, in8, in11, in10, temp4, temp5);                           \
    ILVRL_H2_SH(temp5, temp4, temp6, temp7);                                  \
    ILVR_B2_SH(in13, in12, in15, in14, temp4, temp5);                         \
    ILVRL_H2_SH(temp5, temp4, temp8, temp9);                                  \
    ILVRL_W2_SH(temp8, temp6, temp4, temp5);                                  \
    ILVRL_W2_SH(temp9, temp7, temp6, temp7);                                  \
    ILVL_B2_SH(in1, in0, in3, in2, temp8, temp9);                             \
    ILVR_D2_UB(temp4, temp0, temp5, temp1, in0, in2);                         \
    in1 = (v16u8)__msa_ilvl_d((v2i64)temp4, (v2i64)temp0);                    \
    in3 = (v16u8)__msa_ilvl_d((v2i64)temp5, (v2i64)temp1);                    \
    ILVL_B2_SH(in5, in4, in7, in6, temp0, temp1);                             \
    ILVR_D2_UB(temp6, temp2, temp7, temp3, in4, in6);                         \
    in5 = (v16u8)__msa_ilvl_d((v2i64)temp6, (v2i64)temp2);                    \
    in7 = (v16u8)__msa_ilvl_d((v2i64)temp7, (v2i64)temp3);                    \
    ILVL_B4_SH(in9, in8, in11, in10, in13, in12, in15, in14, temp2, temp3,    \
               temp4, temp5);                                                 \
    ILVR_H4_SH(temp9, temp8, temp1, temp0, temp3, temp2, temp5, temp4, temp6, \
               temp7, temp8, temp9);                                          \
    ILVR_W2_SH(temp7, temp6, temp9, temp8, temp0, temp1);                     \
    in8 = (v16u8)__msa_ilvr_d((v2i64)temp1, (v2i64)temp0);                    \
    in9 = (v16u8)__msa_ilvl_d((v2i64)temp1, (v2i64)temp0);                    \
    ILVL_W2_SH(temp7, temp6, temp9, temp8, temp2, temp3);                     \
    in10 = (v16u8)__msa_ilvr_d((v2i64)temp3, (v2i64)temp2);                   \
    in11 = (v16u8)__msa_ilvl_d((v2i64)temp3, (v2i64)temp2);                   \
  }

#define VPX_TRANSPOSE12x8_UB_UB(in0, in1, in2, in3, in4, in5, in6, in7, in8, \
                                in9, in10, in11)                             \
  {                                                                          \
    v8i16 temp0, temp1, temp2, temp3;                                        \
    v8i16 temp4, temp5, temp6, temp7;                                        \
                                                                             \
    ILVR_B2_SH(in1, in0, in3, in2, temp0, temp1);                            \
    ILVRL_H2_SH(temp1, temp0, temp2, temp3);                                 \
    ILVR_B2_SH(in5, in4, in7, in6, temp0, temp1);                            \
    ILVRL_H2_SH(temp1, temp0, temp4, temp5);                                 \
    ILVRL_W2_SH(temp4, temp2, temp0, temp1);                                 \
    ILVRL_W2_SH(temp5, temp3, temp2, temp3);                                 \
    ILVL_B2_SH(in1, in0, in3, in2, temp4, temp5);                            \
    temp4 = __msa_ilvr_h(temp5, temp4);                                      \
    ILVL_B2_SH(in5, in4, in7, in6, temp6, temp7);                            \
    temp5 = __msa_ilvr_h(temp7, temp6);                                      \
    ILVRL_W2_SH(temp5, temp4, temp6, temp7);                                 \
    in0 = (v16u8)temp0;                                                      \
    in2 = (v16u8)temp1;                                                      \
    in4 = (v16u8)temp2;                                                      \
    in6 = (v16u8)temp3;                                                      \
    in8 = (v16u8)temp6;                                                      \
    in10 = (v16u8)temp7;                                                     \
    in1 = (v16u8)__msa_ilvl_d((v2i64)temp0, (v2i64)temp0);                   \
    in3 = (v16u8)__msa_ilvl_d((v2i64)temp1, (v2i64)temp1);                   \
    in5 = (v16u8)__msa_ilvl_d((v2i64)temp2, (v2i64)temp2);                   \
    in7 = (v16u8)__msa_ilvl_d((v2i64)temp3, (v2i64)temp3);                   \
    in9 = (v16u8)__msa_ilvl_d((v2i64)temp6, (v2i64)temp6);                   \
    in11 = (v16u8)__msa_ilvl_d((v2i64)temp7, (v2i64)temp7);                  \
  }

static void postproc_down_across_chroma_msa(uint8_t *src_ptr, uint8_t *dst_ptr,
                                            int32_t src_stride,
                                            int32_t dst_stride, int32_t cols,
                                            uint8_t *f) {
  uint8_t *p_src = src_ptr;
  uint8_t *p_dst = dst_ptr;
  uint8_t *f_orig = f;
  uint8_t *p_dst_st = dst_ptr;
  uint16_t col;
  uint64_t out0, out1, out2, out3;
  v16u8 above2, above1, below2, below1, src, ref, ref_temp;
  v16u8 inter0, inter1, inter2, inter3, inter4, inter5;
  v16u8 inter6, inter7, inter8, inter9, inter10, inter11;

  for (col = (cols / 16); col--;) {
    ref = LD_UB(f);
    LD_UB2(p_src - 2 * src_stride, src_stride, above2, above1);
    src = LD_UB(p_src);
    LD_UB2(p_src + 1 * src_stride, src_stride, below1, below2);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter0);
    above2 = LD_UB(p_src + 3 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter1);
    above1 = LD_UB(p_src + 4 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter2);
    src = LD_UB(p_src + 5 * src_stride);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref, inter3);
    below1 = LD_UB(p_src + 6 * src_stride);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref, inter4);
    below2 = LD_UB(p_src + 7 * src_stride);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter5);
    above2 = LD_UB(p_src + 8 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter6);
    above1 = LD_UB(p_src + 9 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter7);
    ST_UB8(inter0, inter1, inter2, inter3, inter4, inter5, inter6, inter7,
           p_dst, dst_stride);

    p_dst += 16;
    p_src += 16;
    f += 16;
  }

  if (0 != (cols / 16)) {
    ref = LD_UB(f);
    LD_UB2(p_src - 2 * src_stride, src_stride, above2, above1);
    src = LD_UB(p_src);
    LD_UB2(p_src + 1 * src_stride, src_stride, below1, below2);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter0);
    above2 = LD_UB(p_src + 3 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter1);
    above1 = LD_UB(p_src + 4 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter2);
    src = LD_UB(p_src + 5 * src_stride);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref, inter3);
    below1 = LD_UB(p_src + 6 * src_stride);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref, inter4);
    below2 = LD_UB(p_src + 7 * src_stride);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter5);
    above2 = LD_UB(p_src + 8 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter6);
    above1 = LD_UB(p_src + 9 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter7);
    out0 = __msa_copy_u_d((v2i64)inter0, 0);
    out1 = __msa_copy_u_d((v2i64)inter1, 0);
    out2 = __msa_copy_u_d((v2i64)inter2, 0);
    out3 = __msa_copy_u_d((v2i64)inter3, 0);
    SD4(out0, out1, out2, out3, p_dst, dst_stride);

    out0 = __msa_copy_u_d((v2i64)inter4, 0);
    out1 = __msa_copy_u_d((v2i64)inter5, 0);
    out2 = __msa_copy_u_d((v2i64)inter6, 0);
    out3 = __msa_copy_u_d((v2i64)inter7, 0);
    SD4(out0, out1, out2, out3, p_dst + 4 * dst_stride, dst_stride);
  }

  f = f_orig;
  p_dst = dst_ptr - 2;
  LD_UB8(p_dst, dst_stride, inter0, inter1, inter2, inter3, inter4, inter5,
         inter6, inter7);

  for (col = 0; col < (cols / 8); ++col) {
    ref = LD_UB(f);
    f += 8;
    VPX_TRANSPOSE12x8_UB_UB(inter0, inter1, inter2, inter3, inter4, inter5,
                            inter6, inter7, inter8, inter9, inter10, inter11);
    if (0 == col) {
      above2 = inter2;
      above1 = inter2;
    } else {
      above2 = inter0;
      above1 = inter1;
    }
    src = inter2;
    below1 = inter3;
    below2 = inter4;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 0);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref_temp, inter2);
    above2 = inter5;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 1);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref_temp, inter3);
    above1 = inter6;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 2);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref_temp, inter4);
    src = inter7;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 3);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref_temp, inter5);
    below1 = inter8;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 4);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref_temp, inter6);
    below2 = inter9;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 5);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref_temp, inter7);
    if (col == (cols / 8 - 1)) {
      above2 = inter9;
    } else {
      above2 = inter10;
    }
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 6);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref_temp, inter8);
    if (col == (cols / 8 - 1)) {
      above1 = inter9;
    } else {
      above1 = inter11;
    }
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 7);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref_temp, inter9);
    TRANSPOSE8x8_UB_UB(inter2, inter3, inter4, inter5, inter6, inter7, inter8,
                       inter9, inter2, inter3, inter4, inter5, inter6, inter7,
                       inter8, inter9);
    p_dst += 8;
    LD_UB2(p_dst, dst_stride, inter0, inter1);
    ST8x1_UB(inter2, p_dst_st);
    ST8x1_UB(inter3, (p_dst_st + 1 * dst_stride));
    LD_UB2(p_dst + 2 * dst_stride, dst_stride, inter2, inter3);
    ST8x1_UB(inter4, (p_dst_st + 2 * dst_stride));
    ST8x1_UB(inter5, (p_dst_st + 3 * dst_stride));
    LD_UB2(p_dst + 4 * dst_stride, dst_stride, inter4, inter5);
    ST8x1_UB(inter6, (p_dst_st + 4 * dst_stride));
    ST8x1_UB(inter7, (p_dst_st + 5 * dst_stride));
    LD_UB2(p_dst + 6 * dst_stride, dst_stride, inter6, inter7);
    ST8x1_UB(inter8, (p_dst_st + 6 * dst_stride));
    ST8x1_UB(inter9, (p_dst_st + 7 * dst_stride));
    p_dst_st += 8;
  }
}

static void postproc_down_across_luma_msa(uint8_t *src_ptr, uint8_t *dst_ptr,
                                          int32_t src_stride,
                                          int32_t dst_stride, int32_t cols,
                                          uint8_t *f) {
  uint8_t *p_src = src_ptr;
  uint8_t *p_dst = dst_ptr;
  uint8_t *p_dst_st = dst_ptr;
  uint8_t *f_orig = f;
  uint16_t col;
  uint64_t out0, out1, out2, out3;
  v16u8 above2, above1, below2, below1;
  v16u8 src, ref, ref_temp;
  v16u8 inter0, inter1, inter2, inter3, inter4, inter5, inter6;
  v16u8 inter7, inter8, inter9, inter10, inter11;
  v16u8 inter12, inter13, inter14, inter15;

  for (col = (cols / 16); col--;) {
    ref = LD_UB(f);
    LD_UB2(p_src - 2 * src_stride, src_stride, above2, above1);
    src = LD_UB(p_src);
    LD_UB2(p_src + 1 * src_stride, src_stride, below1, below2);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter0);
    above2 = LD_UB(p_src + 3 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter1);
    above1 = LD_UB(p_src + 4 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter2);
    src = LD_UB(p_src + 5 * src_stride);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref, inter3);
    below1 = LD_UB(p_src + 6 * src_stride);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref, inter4);
    below2 = LD_UB(p_src + 7 * src_stride);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter5);
    above2 = LD_UB(p_src + 8 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter6);
    above1 = LD_UB(p_src + 9 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter7);
    src = LD_UB(p_src + 10 * src_stride);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref, inter8);
    below1 = LD_UB(p_src + 11 * src_stride);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref, inter9);
    below2 = LD_UB(p_src + 12 * src_stride);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter10);
    above2 = LD_UB(p_src + 13 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter11);
    above1 = LD_UB(p_src + 14 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter12);
    src = LD_UB(p_src + 15 * src_stride);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref, inter13);
    below1 = LD_UB(p_src + 16 * src_stride);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref, inter14);
    below2 = LD_UB(p_src + 17 * src_stride);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter15);
    ST_UB8(inter0, inter1, inter2, inter3, inter4, inter5, inter6, inter7,
           p_dst, dst_stride);
    ST_UB8(inter8, inter9, inter10, inter11, inter12, inter13, inter14, inter15,
           p_dst + 8 * dst_stride, dst_stride);
    p_src += 16;
    p_dst += 16;
    f += 16;
  }

  if (0 != (cols / 16)) {
    ref = LD_UB(f);
    LD_UB2(p_src - 2 * src_stride, src_stride, above2, above1);
    src = LD_UB(p_src);
    LD_UB2(p_src + 1 * src_stride, src_stride, below1, below2);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter0);
    above2 = LD_UB(p_src + 3 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter1);
    above1 = LD_UB(p_src + 4 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter2);
    src = LD_UB(p_src + 5 * src_stride);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref, inter3);
    below1 = LD_UB(p_src + 6 * src_stride);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref, inter4);
    below2 = LD_UB(p_src + 7 * src_stride);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter5);
    above2 = LD_UB(p_src + 8 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter6);
    above1 = LD_UB(p_src + 9 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter7);
    src = LD_UB(p_src + 10 * src_stride);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref, inter8);
    below1 = LD_UB(p_src + 11 * src_stride);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref, inter9);
    below2 = LD_UB(p_src + 12 * src_stride);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter10);
    above2 = LD_UB(p_src + 13 * src_stride);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref, inter11);
    above1 = LD_UB(p_src + 14 * src_stride);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref, inter12);
    src = LD_UB(p_src + 15 * src_stride);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref, inter13);
    below1 = LD_UB(p_src + 16 * src_stride);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref, inter14);
    below2 = LD_UB(p_src + 17 * src_stride);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref, inter15);
    out0 = __msa_copy_u_d((v2i64)inter0, 0);
    out1 = __msa_copy_u_d((v2i64)inter1, 0);
    out2 = __msa_copy_u_d((v2i64)inter2, 0);
    out3 = __msa_copy_u_d((v2i64)inter3, 0);
    SD4(out0, out1, out2, out3, p_dst, dst_stride);

    out0 = __msa_copy_u_d((v2i64)inter4, 0);
    out1 = __msa_copy_u_d((v2i64)inter5, 0);
    out2 = __msa_copy_u_d((v2i64)inter6, 0);
    out3 = __msa_copy_u_d((v2i64)inter7, 0);
    SD4(out0, out1, out2, out3, p_dst + 4 * dst_stride, dst_stride);

    out0 = __msa_copy_u_d((v2i64)inter8, 0);
    out1 = __msa_copy_u_d((v2i64)inter9, 0);
    out2 = __msa_copy_u_d((v2i64)inter10, 0);
    out3 = __msa_copy_u_d((v2i64)inter11, 0);
    SD4(out0, out1, out2, out3, p_dst + 8 * dst_stride, dst_stride);

    out0 = __msa_copy_u_d((v2i64)inter12, 0);
    out1 = __msa_copy_u_d((v2i64)inter13, 0);
    out2 = __msa_copy_u_d((v2i64)inter14, 0);
    out3 = __msa_copy_u_d((v2i64)inter15, 0);
    SD4(out0, out1, out2, out3, p_dst + 12 * dst_stride, dst_stride);
  }

  f = f_orig;
  p_dst = dst_ptr - 2;
  LD_UB8(p_dst, dst_stride, inter0, inter1, inter2, inter3, inter4, inter5,
         inter6, inter7);
  LD_UB8(p_dst + 8 * dst_stride, dst_stride, inter8, inter9, inter10, inter11,
         inter12, inter13, inter14, inter15);

  for (col = 0; col < cols / 8; ++col) {
    ref = LD_UB(f);
    f += 8;
    TRANSPOSE12x16_B(inter0, inter1, inter2, inter3, inter4, inter5, inter6,
                     inter7, inter8, inter9, inter10, inter11, inter12, inter13,
                     inter14, inter15);
    if (0 == col) {
      above2 = inter2;
      above1 = inter2;
    } else {
      above2 = inter0;
      above1 = inter1;
    }

    src = inter2;
    below1 = inter3;
    below2 = inter4;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 0);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref_temp, inter2);
    above2 = inter5;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 1);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref_temp, inter3);
    above1 = inter6;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 2);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref_temp, inter4);
    src = inter7;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 3);
    VPX_AVER_IF_RETAIN(below1, below2, above2, above1, src, ref_temp, inter5);
    below1 = inter8;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 4);
    VPX_AVER_IF_RETAIN(below2, above2, above1, src, below1, ref_temp, inter6);
    below2 = inter9;
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 5);
    VPX_AVER_IF_RETAIN(above2, above1, src, below1, below2, ref_temp, inter7);
    if (col == (cols / 8 - 1)) {
      above2 = inter9;
    } else {
      above2 = inter10;
    }
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 6);
    VPX_AVER_IF_RETAIN(above1, src, below1, below2, above2, ref_temp, inter8);
    if (col == (cols / 8 - 1)) {
      above1 = inter9;
    } else {
      above1 = inter11;
    }
    ref_temp = (v16u8)__msa_splati_b((v16i8)ref, 7);
    VPX_AVER_IF_RETAIN(src, below1, below2, above2, above1, ref_temp, inter9);
    VPX_TRANSPOSE8x16_UB_UB(inter2, inter3, inter4, inter5, inter6, inter7,
                            inter8, inter9, inter2, inter3, inter4, inter5,
                            inter6, inter7, inter8, inter9, inter10, inter11,
                            inter12, inter13, inter14, inter15, above2, above1);

    p_dst += 8;
    LD_UB2(p_dst, dst_stride, inter0, inter1);
    ST8x1_UB(inter2, p_dst_st);
    ST8x1_UB(inter3, (p_dst_st + 1 * dst_stride));
    LD_UB2(p_dst + 2 * dst_stride, dst_stride, inter2, inter3);
    ST8x1_UB(inter4, (p_dst_st + 2 * dst_stride));
    ST8x1_UB(inter5, (p_dst_st + 3 * dst_stride));
    LD_UB2(p_dst + 4 * dst_stride, dst_stride, inter4, inter5);
    ST8x1_UB(inter6, (p_dst_st + 4 * dst_stride));
    ST8x1_UB(inter7, (p_dst_st + 5 * dst_stride));
    LD_UB2(p_dst + 6 * dst_stride, dst_stride, inter6, inter7);
    ST8x1_UB(inter8, (p_dst_st + 6 * dst_stride));
    ST8x1_UB(inter9, (p_dst_st + 7 * dst_stride));
    LD_UB2(p_dst + 8 * dst_stride, dst_stride, inter8, inter9);
    ST8x1_UB(inter10, (p_dst_st + 8 * dst_stride));
    ST8x1_UB(inter11, (p_dst_st + 9 * dst_stride));
    LD_UB2(p_dst + 10 * dst_stride, dst_stride, inter10, inter11);
    ST8x1_UB(inter12, (p_dst_st + 10 * dst_stride));
    ST8x1_UB(inter13, (p_dst_st + 11 * dst_stride));
    LD_UB2(p_dst + 12 * dst_stride, dst_stride, inter12, inter13);
    ST8x1_UB(inter14, (p_dst_st + 12 * dst_stride));
    ST8x1_UB(inter15, (p_dst_st + 13 * dst_stride));
    LD_UB2(p_dst + 14 * dst_stride, dst_stride, inter14, inter15);
    ST8x1_UB(above2, (p_dst_st + 14 * dst_stride));
    ST8x1_UB(above1, (p_dst_st + 15 * dst_stride));
    p_dst_st += 8;
  }
}

void vpx_post_proc_down_and_across_mb_row_msa(uint8_t *src, uint8_t *dst,
                                              int32_t src_stride,
                                              int32_t dst_stride, int32_t cols,
                                              uint8_t *f, int32_t size) {
  if (8 == size) {
    postproc_down_across_chroma_msa(src, dst, src_stride, dst_stride, cols, f);
  } else if (16 == size) {
    postproc_down_across_luma_msa(src, dst, src_stride, dst_stride, cols, f);
  }
}

void vpx_mbpost_proc_across_ip_msa(uint8_t *src, int32_t pitch, int32_t rows,
                                   int32_t cols, int32_t flimit) {
  int32_t row, col, cnt;
  uint8_t *src_dup = src;
  v16u8 src0, src1, tmp_orig;
  v16u8 tmp = { 0 };
  v16i8 zero = { 0 };
  v8u16 sum_h, src_r_h, src_l_h;
  v4u32 src_r_w;
  v4i32 flimit_vec;

  flimit_vec = __msa_fill_w(flimit);
  for (row = rows; row--;) {
    int32_t sum_sq;
    int32_t sum = 0;
    src0 = (v16u8)__msa_fill_b(src_dup[0]);
    ST8x1_UB(src0, (src_dup - 8));

    src0 = (v16u8)__msa_fill_b(src_dup[cols - 1]);
    ST_UB(src0, src_dup + cols);
    src_dup[cols + 16] = src_dup[cols - 1];
    tmp_orig = (v16u8)__msa_ldi_b(0);
    tmp_orig[15] = tmp[15];
    src1 = LD_UB(src_dup - 8);
    src1[15] = 0;
    ILVRL_B2_UH(zero, src1, src_r_h, src_l_h);
    src_r_w = __msa_dotp_u_w(src_r_h, src_r_h);
    src_r_w += __msa_dotp_u_w(src_l_h, src_l_h);
    sum_sq = HADD_SW_S32(src_r_w) + 16;
    sum_h = __msa_hadd_u_h(src1, src1);
    sum = HADD_UH_U32(sum_h);
    {
      v16u8 src7, src8, src_r, src_l;
      v16i8 mask;
      v8u16 add_r, add_l;
      v8i16 sub_r, sub_l, sum_r, sum_l, mask0, mask1;
      v4i32 sum_sq0, sum_sq1, sum_sq2, sum_sq3;
      v4i32 sub0, sub1, sub2, sub3;
      v4i32 sum0_w, sum1_w, sum2_w, sum3_w;
      v4i32 mul0, mul1, mul2, mul3;
      v4i32 total0, total1, total2, total3;
      v8i16 const8 = __msa_fill_h(8);

      src7 = LD_UB(src_dup + 7);
      src8 = LD_UB(src_dup - 8);
      for (col = 0; col < (cols >> 4); ++col) {
        ILVRL_B2_UB(src7, src8, src_r, src_l);
        HSUB_UB2_SH(src_r, src_l, sub_r, sub_l);

        sum_r[0] = sum + sub_r[0];
        for (cnt = 0; cnt < 7; ++cnt) {
          sum_r[cnt + 1] = sum_r[cnt] + sub_r[cnt + 1];
        }
        sum_l[0] = sum_r[7] + sub_l[0];
        for (cnt = 0; cnt < 7; ++cnt) {
          sum_l[cnt + 1] = sum_l[cnt] + sub_l[cnt + 1];
        }
        sum = sum_l[7];
        src1 = LD_UB(src_dup + 16 * col);
        ILVRL_B2_UH(zero, src1, src_r_h, src_l_h);
        src7 = (v16u8)((const8 + sum_r + (v8i16)src_r_h) >> 4);
        src8 = (v16u8)((const8 + sum_l + (v8i16)src_l_h) >> 4);
        tmp = (v16u8)__msa_pckev_b((v16i8)src8, (v16i8)src7);

        HADD_UB2_UH(src_r, src_l, add_r, add_l);
        UNPCK_SH_SW(sub_r, sub0, sub1);
        UNPCK_SH_SW(sub_l, sub2, sub3);
        ILVR_H2_SW(zero, add_r, zero, add_l, sum0_w, sum2_w);
        ILVL_H2_SW(zero, add_r, zero, add_l, sum1_w, sum3_w);
        MUL4(sum0_w, sub0, sum1_w, sub1, sum2_w, sub2, sum3_w, sub3, mul0, mul1,
             mul2, mul3);
        sum_sq0[0] = sum_sq + mul0[0];
        for (cnt = 0; cnt < 3; ++cnt) {
          sum_sq0[cnt + 1] = sum_sq0[cnt] + mul0[cnt + 1];
        }
        sum_sq1[0] = sum_sq0[3] + mul1[0];
        for (cnt = 0; cnt < 3; ++cnt) {
          sum_sq1[cnt + 1] = sum_sq1[cnt] + mul1[cnt + 1];
        }
        sum_sq2[0] = sum_sq1[3] + mul2[0];
        for (cnt = 0; cnt < 3; ++cnt) {
          sum_sq2[cnt + 1] = sum_sq2[cnt] + mul2[cnt + 1];
        }
        sum_sq3[0] = sum_sq2[3] + mul3[0];
        for (cnt = 0; cnt < 3; ++cnt) {
          sum_sq3[cnt + 1] = sum_sq3[cnt] + mul3[cnt + 1];
        }
        sum_sq = sum_sq3[3];

        UNPCK_SH_SW(sum_r, sum0_w, sum1_w);
        UNPCK_SH_SW(sum_l, sum2_w, sum3_w);
        total0 = sum_sq0 * __msa_ldi_w(15);
        total0 -= sum0_w * sum0_w;
        total1 = sum_sq1 * __msa_ldi_w(15);
        total1 -= sum1_w * sum1_w;
        total2 = sum_sq2 * __msa_ldi_w(15);
        total2 -= sum2_w * sum2_w;
        total3 = sum_sq3 * __msa_ldi_w(15);
        total3 -= sum3_w * sum3_w;
        total0 = (total0 < flimit_vec);
        total1 = (total1 < flimit_vec);
        total2 = (total2 < flimit_vec);
        total3 = (total3 < flimit_vec);
        PCKEV_H2_SH(total1, total0, total3, total2, mask0, mask1);
        mask = __msa_pckev_b((v16i8)mask1, (v16i8)mask0);
        tmp = __msa_bmz_v(tmp, src1, (v16u8)mask);

        if (col == 0) {
          uint64_t src_d;

          src_d = __msa_copy_u_d((v2i64)tmp_orig, 1);
          SD(src_d, (src_dup - 8));
        }

        src7 = LD_UB(src_dup + 16 * (col + 1) + 7);
        src8 = LD_UB(src_dup + 16 * (col + 1) - 8);
        ST_UB(tmp, (src_dup + (16 * col)));
      }

      src_dup += pitch;
    }
  }
}

void vpx_mbpost_proc_down_msa(uint8_t *dst_ptr, int32_t pitch, int32_t rows,
                              int32_t cols, int32_t flimit) {
  int32_t row, col, cnt, i;
  v4i32 flimit_vec;
  v16u8 dst7, dst8, dst_r_b, dst_l_b;
  v16i8 mask;
  v8u16 add_r, add_l;
  v8i16 dst_r_h, dst_l_h, sub_r, sub_l, mask0, mask1;
  v4i32 sub0, sub1, sub2, sub3, total0, total1, total2, total3;

  flimit_vec = __msa_fill_w(flimit);

  for (col = 0; col < (cols >> 4); ++col) {
    uint8_t *dst_tmp = &dst_ptr[col << 4];
    v16u8 dst;
    v16i8 zero = { 0 };
    v16u8 tmp[16];
    v8i16 mult0, mult1, rv2_0, rv2_1;
    v8i16 sum0_h = { 0 };
    v8i16 sum1_h = { 0 };
    v4i32 mul0 = { 0 };
    v4i32 mul1 = { 0 };
    v4i32 mul2 = { 0 };
    v4i32 mul3 = { 0 };
    v4i32 sum0_w, sum1_w, sum2_w, sum3_w;
    v4i32 add0, add1, add2, add3;
    const int16_t *rv2[16];

    dst = LD_UB(dst_tmp);
    for (cnt = (col << 4), i = 0; i < 16; ++cnt) {
      rv2[i] = vpx_rv + (i & 7);
      ++i;
    }
    for (cnt = -8; cnt < 0; ++cnt) {
      ST_UB(dst, dst_tmp + cnt * pitch);
    }

    dst = LD_UB((dst_tmp + (rows - 1) * pitch));
    for (cnt = rows; cnt < rows + 17; ++cnt) {
      ST_UB(dst, dst_tmp + cnt * pitch);
    }
    for (cnt = -8; cnt <= 6; ++cnt) {
      dst = LD_UB(dst_tmp + (cnt * pitch));
      UNPCK_UB_SH(dst, dst_r_h, dst_l_h);
      MUL2(dst_r_h, dst_r_h, dst_l_h, dst_l_h, mult0, mult1);
      mul0 += (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)mult0);
      mul1 += (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)mult0);
      mul2 += (v4i32)__msa_ilvr_h((v8i16)zero, (v8i16)mult1);
      mul3 += (v4i32)__msa_ilvl_h((v8i16)zero, (v8i16)mult1);
      ADD2(sum0_h, dst_r_h, sum1_h, dst_l_h, sum0_h, sum1_h);
    }

    for (row = 0; row < (rows + 8); ++row) {
      for (i = 0; i < 8; ++i) {
        rv2_0[i] = *(rv2[i] + (row & 127));
        rv2_1[i] = *(rv2[i + 8] + (row & 127));
      }
      dst7 = LD_UB(dst_tmp + (7 * pitch));
      dst8 = LD_UB(dst_tmp - (8 * pitch));
      ILVRL_B2_UB(dst7, dst8, dst_r_b, dst_l_b);

      HSUB_UB2_SH(dst_r_b, dst_l_b, sub_r, sub_l);
      UNPCK_SH_SW(sub_r, sub0, sub1);
      UNPCK_SH_SW(sub_l, sub2, sub3);
      sum0_h += sub_r;
      sum1_h += sub_l;

      HADD_UB2_UH(dst_r_b, dst_l_b, add_r, add_l);

      ILVRL_H2_SW(zero, add_r, add0, add1);
      ILVRL_H2_SW(zero, add_l, add2, add3);
      mul0 += add0 * sub0;
      mul1 += add1 * sub1;
      mul2 += add2 * sub2;
      mul3 += add3 * sub3;
      dst = LD_UB(dst_tmp);
      ILVRL_B2_SH(zero, dst, dst_r_h, dst_l_h);
      dst7 = (v16u8)((rv2_0 + sum0_h + dst_r_h) >> 4);
      dst8 = (v16u8)((rv2_1 + sum1_h + dst_l_h) >> 4);
      tmp[row & 15] = (v16u8)__msa_pckev_b((v16i8)dst8, (v16i8)dst7);

      UNPCK_SH_SW(sum0_h, sum0_w, sum1_w);
      UNPCK_SH_SW(sum1_h, sum2_w, sum3_w);
      total0 = mul0 * __msa_ldi_w(15);
      total0 -= sum0_w * sum0_w;
      total1 = mul1 * __msa_ldi_w(15);
      total1 -= sum1_w * sum1_w;
      total2 = mul2 * __msa_ldi_w(15);
      total2 -= sum2_w * sum2_w;
      total3 = mul3 * __msa_ldi_w(15);
      total3 -= sum3_w * sum3_w;
      total0 = (total0 < flimit_vec);
      total1 = (total1 < flimit_vec);
      total2 = (total2 < flimit_vec);
      total3 = (total3 < flimit_vec);
      PCKEV_H2_SH(total1, total0, total3, total2, mask0, mask1);
      mask = __msa_pckev_b((v16i8)mask1, (v16i8)mask0);
      tmp[row & 15] = __msa_bmz_v(tmp[row & 15], dst, (v16u8)mask);

      if (row >= 8) {
        ST_UB(tmp[(row - 8) & 15], (dst_tmp - 8 * pitch));
      }

      dst_tmp += pitch;
    }
  }
}
