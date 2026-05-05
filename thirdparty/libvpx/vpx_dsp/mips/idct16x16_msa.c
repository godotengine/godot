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

void vpx_idct16_1d_rows_msa(const int16_t *input, int16_t *output) {
  v8i16 loc0, loc1, loc2, loc3;
  v8i16 reg0, reg2, reg4, reg6, reg8, reg10, reg12, reg14;
  v8i16 reg3, reg13, reg11, reg5, reg7, reg9, reg1, reg15;
  v8i16 tmp5, tmp6, tmp7;

  LD_SH8(input, 16, reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7);
  input += 8;
  LD_SH8(input, 16, reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15);

  TRANSPOSE8x8_SH_SH(reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg0, reg1,
                     reg2, reg3, reg4, reg5, reg6, reg7);
  TRANSPOSE8x8_SH_SH(reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15, reg8,
                     reg9, reg10, reg11, reg12, reg13, reg14, reg15);
  DOTP_CONST_PAIR(reg2, reg14, cospi_28_64, cospi_4_64, reg2, reg14);
  DOTP_CONST_PAIR(reg10, reg6, cospi_12_64, cospi_20_64, reg10, reg6);
  BUTTERFLY_4(reg2, reg14, reg6, reg10, loc0, loc1, reg14, reg2);
  DOTP_CONST_PAIR(reg14, reg2, cospi_16_64, cospi_16_64, loc2, loc3);
  DOTP_CONST_PAIR(reg0, reg8, cospi_16_64, cospi_16_64, reg0, reg8);
  DOTP_CONST_PAIR(reg4, reg12, cospi_24_64, cospi_8_64, reg4, reg12);
  BUTTERFLY_4(reg8, reg0, reg4, reg12, reg2, reg6, reg10, reg14);
  SUB4(reg2, loc1, reg14, loc0, reg6, loc3, reg10, loc2, reg0, reg12, reg4,
       reg8);
  ADD4(reg2, loc1, reg14, loc0, reg6, loc3, reg10, loc2, reg2, reg14, reg6,
       reg10);

  /* stage 2 */
  DOTP_CONST_PAIR(reg1, reg15, cospi_30_64, cospi_2_64, reg1, reg15);
  DOTP_CONST_PAIR(reg9, reg7, cospi_14_64, cospi_18_64, loc2, loc3);

  reg9 = reg1 - loc2;
  reg1 = reg1 + loc2;
  reg7 = reg15 - loc3;
  reg15 = reg15 + loc3;

  DOTP_CONST_PAIR(reg5, reg11, cospi_22_64, cospi_10_64, reg5, reg11);
  DOTP_CONST_PAIR(reg13, reg3, cospi_6_64, cospi_26_64, loc0, loc1);
  BUTTERFLY_4(loc0, loc1, reg11, reg5, reg13, reg3, reg11, reg5);

  loc1 = reg15 + reg3;
  reg3 = reg15 - reg3;
  loc2 = reg2 + loc1;
  reg15 = reg2 - loc1;

  loc1 = reg1 + reg13;
  reg13 = reg1 - reg13;
  loc0 = reg0 + loc1;
  loc1 = reg0 - loc1;
  tmp6 = loc0;
  tmp7 = loc1;
  reg0 = loc2;

  DOTP_CONST_PAIR(reg7, reg9, cospi_24_64, cospi_8_64, reg7, reg9);
  DOTP_CONST_PAIR((-reg5), (-reg11), cospi_8_64, cospi_24_64, reg5, reg11);

  loc0 = reg9 + reg5;
  reg5 = reg9 - reg5;
  reg2 = reg6 + loc0;
  reg1 = reg6 - loc0;

  loc0 = reg7 + reg11;
  reg11 = reg7 - reg11;
  loc1 = reg4 + loc0;
  loc2 = reg4 - loc0;
  tmp5 = loc1;

  DOTP_CONST_PAIR(reg5, reg11, cospi_16_64, cospi_16_64, reg5, reg11);
  BUTTERFLY_4(reg8, reg10, reg11, reg5, loc0, reg4, reg9, loc1);

  reg10 = loc0;
  reg11 = loc1;

  DOTP_CONST_PAIR(reg3, reg13, cospi_16_64, cospi_16_64, reg3, reg13);
  BUTTERFLY_4(reg12, reg14, reg13, reg3, reg8, reg6, reg7, reg5);

  reg13 = loc2;

  /* Transpose and store the output */
  reg12 = tmp5;
  reg14 = tmp6;
  reg3 = tmp7;

  /* transpose block */
  TRANSPOSE8x8_SH_SH(reg0, reg2, reg4, reg6, reg8, reg10, reg12, reg14, reg0,
                     reg2, reg4, reg6, reg8, reg10, reg12, reg14);
  ST_SH8(reg0, reg2, reg4, reg6, reg8, reg10, reg12, reg14, output, 16);

  /* transpose block */
  TRANSPOSE8x8_SH_SH(reg3, reg13, reg11, reg5, reg7, reg9, reg1, reg15, reg3,
                     reg13, reg11, reg5, reg7, reg9, reg1, reg15);
  ST_SH8(reg3, reg13, reg11, reg5, reg7, reg9, reg1, reg15, (output + 8), 16);
}

void vpx_idct16_1d_columns_addblk_msa(int16_t *input, uint8_t *dst,
                                      int32_t dst_stride) {
  v8i16 loc0, loc1, loc2, loc3;
  v8i16 reg0, reg2, reg4, reg6, reg8, reg10, reg12, reg14;
  v8i16 reg3, reg13, reg11, reg5, reg7, reg9, reg1, reg15;
  v8i16 tmp5, tmp6, tmp7;

  /* load up 8x8 */
  LD_SH8(input, 16, reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7);
  input += 8 * 16;
  /* load bottom 8x8 */
  LD_SH8(input, 16, reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15);

  DOTP_CONST_PAIR(reg2, reg14, cospi_28_64, cospi_4_64, reg2, reg14);
  DOTP_CONST_PAIR(reg10, reg6, cospi_12_64, cospi_20_64, reg10, reg6);
  BUTTERFLY_4(reg2, reg14, reg6, reg10, loc0, loc1, reg14, reg2);
  DOTP_CONST_PAIR(reg14, reg2, cospi_16_64, cospi_16_64, loc2, loc3);
  DOTP_CONST_PAIR(reg0, reg8, cospi_16_64, cospi_16_64, reg0, reg8);
  DOTP_CONST_PAIR(reg4, reg12, cospi_24_64, cospi_8_64, reg4, reg12);
  BUTTERFLY_4(reg8, reg0, reg4, reg12, reg2, reg6, reg10, reg14);

  reg0 = reg2 - loc1;
  reg2 = reg2 + loc1;
  reg12 = reg14 - loc0;
  reg14 = reg14 + loc0;
  reg4 = reg6 - loc3;
  reg6 = reg6 + loc3;
  reg8 = reg10 - loc2;
  reg10 = reg10 + loc2;

  /* stage 2 */
  DOTP_CONST_PAIR(reg1, reg15, cospi_30_64, cospi_2_64, reg1, reg15);
  DOTP_CONST_PAIR(reg9, reg7, cospi_14_64, cospi_18_64, loc2, loc3);

  reg9 = reg1 - loc2;
  reg1 = reg1 + loc2;
  reg7 = reg15 - loc3;
  reg15 = reg15 + loc3;

  DOTP_CONST_PAIR(reg5, reg11, cospi_22_64, cospi_10_64, reg5, reg11);
  DOTP_CONST_PAIR(reg13, reg3, cospi_6_64, cospi_26_64, loc0, loc1);
  BUTTERFLY_4(loc0, loc1, reg11, reg5, reg13, reg3, reg11, reg5);

  loc1 = reg15 + reg3;
  reg3 = reg15 - reg3;
  loc2 = reg2 + loc1;
  reg15 = reg2 - loc1;

  loc1 = reg1 + reg13;
  reg13 = reg1 - reg13;
  loc0 = reg0 + loc1;
  loc1 = reg0 - loc1;
  tmp6 = loc0;
  tmp7 = loc1;
  reg0 = loc2;

  DOTP_CONST_PAIR(reg7, reg9, cospi_24_64, cospi_8_64, reg7, reg9);
  DOTP_CONST_PAIR((-reg5), (-reg11), cospi_8_64, cospi_24_64, reg5, reg11);

  loc0 = reg9 + reg5;
  reg5 = reg9 - reg5;
  reg2 = reg6 + loc0;
  reg1 = reg6 - loc0;

  loc0 = reg7 + reg11;
  reg11 = reg7 - reg11;
  loc1 = reg4 + loc0;
  loc2 = reg4 - loc0;
  tmp5 = loc1;

  DOTP_CONST_PAIR(reg5, reg11, cospi_16_64, cospi_16_64, reg5, reg11);
  BUTTERFLY_4(reg8, reg10, reg11, reg5, loc0, reg4, reg9, loc1);

  reg10 = loc0;
  reg11 = loc1;

  DOTP_CONST_PAIR(reg3, reg13, cospi_16_64, cospi_16_64, reg3, reg13);
  BUTTERFLY_4(reg12, reg14, reg13, reg3, reg8, reg6, reg7, reg5);
  reg13 = loc2;

  /* Transpose and store the output */
  reg12 = tmp5;
  reg14 = tmp6;
  reg3 = tmp7;

  SRARI_H4_SH(reg0, reg2, reg4, reg6, 6);
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, reg0, reg2, reg4, reg6);
  dst += (4 * dst_stride);
  SRARI_H4_SH(reg8, reg10, reg12, reg14, 6);
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, reg8, reg10, reg12, reg14);
  dst += (4 * dst_stride);
  SRARI_H4_SH(reg3, reg13, reg11, reg5, 6);
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, reg3, reg13, reg11, reg5);
  dst += (4 * dst_stride);
  SRARI_H4_SH(reg7, reg9, reg1, reg15, 6);
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, reg7, reg9, reg1, reg15);
}

void vpx_idct16x16_256_add_msa(const int16_t *input, uint8_t *dst,
                               int32_t dst_stride) {
  int32_t i;
  DECLARE_ALIGNED(32, int16_t, out_arr[16 * 16]);
  int16_t *out = out_arr;

  /* transform rows */
  for (i = 0; i < 2; ++i) {
    /* process 16 * 8 block */
    vpx_idct16_1d_rows_msa((input + (i << 7)), (out + (i << 7)));
  }

  /* transform columns */
  for (i = 0; i < 2; ++i) {
    /* process 8 * 16 block */
    vpx_idct16_1d_columns_addblk_msa((out + (i << 3)), (dst + (i << 3)),
                                     dst_stride);
  }
}

void vpx_idct16x16_10_add_msa(const int16_t *input, uint8_t *dst,
                              int32_t dst_stride) {
  uint8_t i;
  DECLARE_ALIGNED(32, int16_t, out_arr[16 * 16]);
  int16_t *out = out_arr;

  /* process 16 * 8 block */
  vpx_idct16_1d_rows_msa(input, out);

  /* short case just considers top 4 rows as valid output */
  out += 4 * 16;
  for (i = 12; i--;) {
    __asm__ __volatile__(
        "sw     $zero,   0(%[out])     \n\t"
        "sw     $zero,   4(%[out])     \n\t"
        "sw     $zero,   8(%[out])     \n\t"
        "sw     $zero,  12(%[out])     \n\t"
        "sw     $zero,  16(%[out])     \n\t"
        "sw     $zero,  20(%[out])     \n\t"
        "sw     $zero,  24(%[out])     \n\t"
        "sw     $zero,  28(%[out])     \n\t"

        :
        : [out] "r"(out));

    out += 16;
  }

  out = out_arr;

  /* transform columns */
  for (i = 0; i < 2; ++i) {
    /* process 8 * 16 block */
    vpx_idct16_1d_columns_addblk_msa((out + (i << 3)), (dst + (i << 3)),
                                     dst_stride);
  }
}

void vpx_idct16x16_1_add_msa(const int16_t *input, uint8_t *dst,
                             int32_t dst_stride) {
  uint8_t i;
  int16_t out;
  v8i16 vec, res0, res1, res2, res3, res4, res5, res6, res7;
  v16u8 dst0, dst1, dst2, dst3, tmp0, tmp1, tmp2, tmp3;

  out = ROUND_POWER_OF_TWO((input[0] * cospi_16_64), DCT_CONST_BITS);
  out = ROUND_POWER_OF_TWO((out * cospi_16_64), DCT_CONST_BITS);
  out = ROUND_POWER_OF_TWO(out, 6);

  vec = __msa_fill_h(out);

  for (i = 4; i--;) {
    LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);
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
    ST_UB4(tmp0, tmp1, tmp2, tmp3, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

void vpx_iadst16_1d_rows_msa(const int16_t *input, int16_t *output) {
  v8i16 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  v8i16 l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15;

  /* load input data */
  LD_SH16(input, 8, l0, l8, l1, l9, l2, l10, l3, l11, l4, l12, l5, l13, l6, l14,
          l7, l15);
  TRANSPOSE8x8_SH_SH(l0, l1, l2, l3, l4, l5, l6, l7, l0, l1, l2, l3, l4, l5, l6,
                     l7);
  TRANSPOSE8x8_SH_SH(l8, l9, l10, l11, l12, l13, l14, l15, l8, l9, l10, l11,
                     l12, l13, l14, l15);

  /* ADST in horizontal */
  VP9_IADST8x16_1D(l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                   l14, l15, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11,
                   r12, r13, r14, r15);

  l1 = -r8;
  l3 = -r4;
  l13 = -r13;
  l15 = -r1;

  TRANSPOSE8x8_SH_SH(r0, l1, r12, l3, r6, r14, r10, r2, l0, l1, l2, l3, l4, l5,
                     l6, l7);
  ST_SH8(l0, l1, l2, l3, l4, l5, l6, l7, output, 16);
  TRANSPOSE8x8_SH_SH(r3, r11, r15, r7, r5, l13, r9, l15, l8, l9, l10, l11, l12,
                     l13, l14, l15);
  ST_SH8(l8, l9, l10, l11, l12, l13, l14, l15, (output + 8), 16);
}

void vpx_iadst16_1d_columns_addblk_msa(int16_t *input, uint8_t *dst,
                                       int32_t dst_stride) {
  v8i16 v0, v2, v4, v6, k0, k1, k2, k3;
  v8i16 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  v8i16 out0, out1, out2, out3, out4, out5, out6, out7;
  v8i16 out8, out9, out10, out11, out12, out13, out14, out15;
  v8i16 g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15;
  v8i16 h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11;
  v8i16 res0, res1, res2, res3, res4, res5, res6, res7;
  v8i16 res8, res9, res10, res11, res12, res13, res14, res15;
  v16u8 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
  v16u8 dst8, dst9, dst10, dst11, dst12, dst13, dst14, dst15;
  v16i8 zero = { 0 };

  r0 = LD_SH(input + 0 * 16);
  r3 = LD_SH(input + 3 * 16);
  r4 = LD_SH(input + 4 * 16);
  r7 = LD_SH(input + 7 * 16);
  r8 = LD_SH(input + 8 * 16);
  r11 = LD_SH(input + 11 * 16);
  r12 = LD_SH(input + 12 * 16);
  r15 = LD_SH(input + 15 * 16);

  /* stage 1 */
  k0 = VP9_SET_COSPI_PAIR(cospi_1_64, cospi_31_64);
  k1 = VP9_SET_COSPI_PAIR(cospi_31_64, -cospi_1_64);
  k2 = VP9_SET_COSPI_PAIR(cospi_17_64, cospi_15_64);
  k3 = VP9_SET_COSPI_PAIR(cospi_15_64, -cospi_17_64);
  MADD_BF(r15, r0, r7, r8, k0, k1, k2, k3, g0, g1, g2, g3);
  k0 = VP9_SET_COSPI_PAIR(cospi_9_64, cospi_23_64);
  k1 = VP9_SET_COSPI_PAIR(cospi_23_64, -cospi_9_64);
  k2 = VP9_SET_COSPI_PAIR(cospi_25_64, cospi_7_64);
  k3 = VP9_SET_COSPI_PAIR(cospi_7_64, -cospi_25_64);
  MADD_BF(r11, r4, r3, r12, k0, k1, k2, k3, g8, g9, g10, g11);
  BUTTERFLY_4(g0, g2, g10, g8, h8, h9, v2, v0);
  k0 = VP9_SET_COSPI_PAIR(cospi_4_64, cospi_28_64);
  k1 = VP9_SET_COSPI_PAIR(cospi_28_64, -cospi_4_64);
  k2 = VP9_SET_COSPI_PAIR(-cospi_28_64, cospi_4_64);
  MADD_BF(g1, g3, g9, g11, k0, k1, k2, k0, h0, h1, h2, h3);

  r1 = LD_SH(input + 1 * 16);
  r2 = LD_SH(input + 2 * 16);
  r5 = LD_SH(input + 5 * 16);
  r6 = LD_SH(input + 6 * 16);
  r9 = LD_SH(input + 9 * 16);
  r10 = LD_SH(input + 10 * 16);
  r13 = LD_SH(input + 13 * 16);
  r14 = LD_SH(input + 14 * 16);

  k0 = VP9_SET_COSPI_PAIR(cospi_5_64, cospi_27_64);
  k1 = VP9_SET_COSPI_PAIR(cospi_27_64, -cospi_5_64);
  k2 = VP9_SET_COSPI_PAIR(cospi_21_64, cospi_11_64);
  k3 = VP9_SET_COSPI_PAIR(cospi_11_64, -cospi_21_64);
  MADD_BF(r13, r2, r5, r10, k0, k1, k2, k3, g4, g5, g6, g7);
  k0 = VP9_SET_COSPI_PAIR(cospi_13_64, cospi_19_64);
  k1 = VP9_SET_COSPI_PAIR(cospi_19_64, -cospi_13_64);
  k2 = VP9_SET_COSPI_PAIR(cospi_29_64, cospi_3_64);
  k3 = VP9_SET_COSPI_PAIR(cospi_3_64, -cospi_29_64);
  MADD_BF(r9, r6, r1, r14, k0, k1, k2, k3, g12, g13, g14, g15);
  BUTTERFLY_4(g4, g6, g14, g12, h10, h11, v6, v4);
  BUTTERFLY_4(h8, h9, h11, h10, out0, out1, h11, h10);
  out1 = -out1;
  SRARI_H2_SH(out0, out1, 6);
  dst0 = LD_UB(dst + 0 * dst_stride);
  dst1 = LD_UB(dst + 15 * dst_stride);
  ILVR_B2_SH(zero, dst0, zero, dst1, res0, res1);
  ADD2(res0, out0, res1, out1, res0, res1);
  CLIP_SH2_0_255(res0, res1);
  PCKEV_B2_SH(res0, res0, res1, res1, res0, res1);
  ST8x1_UB(res0, dst);
  ST8x1_UB(res1, dst + 15 * dst_stride);

  k0 = VP9_SET_COSPI_PAIR(cospi_12_64, cospi_20_64);
  k1 = VP9_SET_COSPI_PAIR(-cospi_20_64, cospi_12_64);
  k2 = VP9_SET_COSPI_PAIR(cospi_20_64, -cospi_12_64);
  MADD_BF(g7, g5, g15, g13, k0, k1, k2, k0, h4, h5, h6, h7);
  BUTTERFLY_4(h0, h2, h6, h4, out8, out9, out11, out10);
  out8 = -out8;

  SRARI_H2_SH(out8, out9, 6);
  dst8 = LD_UB(dst + 1 * dst_stride);
  dst9 = LD_UB(dst + 14 * dst_stride);
  ILVR_B2_SH(zero, dst8, zero, dst9, res8, res9);
  ADD2(res8, out8, res9, out9, res8, res9);
  CLIP_SH2_0_255(res8, res9);
  PCKEV_B2_SH(res8, res8, res9, res9, res8, res9);
  ST8x1_UB(res8, dst + dst_stride);
  ST8x1_UB(res9, dst + 14 * dst_stride);

  k0 = VP9_SET_COSPI_PAIR(cospi_8_64, cospi_24_64);
  k1 = VP9_SET_COSPI_PAIR(cospi_24_64, -cospi_8_64);
  k2 = VP9_SET_COSPI_PAIR(-cospi_24_64, cospi_8_64);
  MADD_BF(v0, v2, v4, v6, k0, k1, k2, k0, out4, out6, out5, out7);
  out4 = -out4;
  SRARI_H2_SH(out4, out5, 6);
  dst4 = LD_UB(dst + 3 * dst_stride);
  dst5 = LD_UB(dst + 12 * dst_stride);
  ILVR_B2_SH(zero, dst4, zero, dst5, res4, res5);
  ADD2(res4, out4, res5, out5, res4, res5);
  CLIP_SH2_0_255(res4, res5);
  PCKEV_B2_SH(res4, res4, res5, res5, res4, res5);
  ST8x1_UB(res4, dst + 3 * dst_stride);
  ST8x1_UB(res5, dst + 12 * dst_stride);

  MADD_BF(h1, h3, h5, h7, k0, k1, k2, k0, out12, out14, out13, out15);
  out13 = -out13;
  SRARI_H2_SH(out12, out13, 6);
  dst12 = LD_UB(dst + 2 * dst_stride);
  dst13 = LD_UB(dst + 13 * dst_stride);
  ILVR_B2_SH(zero, dst12, zero, dst13, res12, res13);
  ADD2(res12, out12, res13, out13, res12, res13);
  CLIP_SH2_0_255(res12, res13);
  PCKEV_B2_SH(res12, res12, res13, res13, res12, res13);
  ST8x1_UB(res12, dst + 2 * dst_stride);
  ST8x1_UB(res13, dst + 13 * dst_stride);

  k0 = VP9_SET_COSPI_PAIR(cospi_16_64, cospi_16_64);
  k3 = VP9_SET_COSPI_PAIR(-cospi_16_64, cospi_16_64);
  MADD_SHORT(out6, out7, k0, k3, out6, out7);
  SRARI_H2_SH(out6, out7, 6);
  dst6 = LD_UB(dst + 4 * dst_stride);
  dst7 = LD_UB(dst + 11 * dst_stride);
  ILVR_B2_SH(zero, dst6, zero, dst7, res6, res7);
  ADD2(res6, out6, res7, out7, res6, res7);
  CLIP_SH2_0_255(res6, res7);
  PCKEV_B2_SH(res6, res6, res7, res7, res6, res7);
  ST8x1_UB(res6, dst + 4 * dst_stride);
  ST8x1_UB(res7, dst + 11 * dst_stride);

  MADD_SHORT(out10, out11, k0, k3, out10, out11);
  SRARI_H2_SH(out10, out11, 6);
  dst10 = LD_UB(dst + 6 * dst_stride);
  dst11 = LD_UB(dst + 9 * dst_stride);
  ILVR_B2_SH(zero, dst10, zero, dst11, res10, res11);
  ADD2(res10, out10, res11, out11, res10, res11);
  CLIP_SH2_0_255(res10, res11);
  PCKEV_B2_SH(res10, res10, res11, res11, res10, res11);
  ST8x1_UB(res10, dst + 6 * dst_stride);
  ST8x1_UB(res11, dst + 9 * dst_stride);

  k1 = VP9_SET_COSPI_PAIR(-cospi_16_64, -cospi_16_64);
  k2 = VP9_SET_COSPI_PAIR(cospi_16_64, -cospi_16_64);
  MADD_SHORT(h10, h11, k1, k2, out2, out3);
  SRARI_H2_SH(out2, out3, 6);
  dst2 = LD_UB(dst + 7 * dst_stride);
  dst3 = LD_UB(dst + 8 * dst_stride);
  ILVR_B2_SH(zero, dst2, zero, dst3, res2, res3);
  ADD2(res2, out2, res3, out3, res2, res3);
  CLIP_SH2_0_255(res2, res3);
  PCKEV_B2_SH(res2, res2, res3, res3, res2, res3);
  ST8x1_UB(res2, dst + 7 * dst_stride);
  ST8x1_UB(res3, dst + 8 * dst_stride);

  MADD_SHORT(out14, out15, k1, k2, out14, out15);
  SRARI_H2_SH(out14, out15, 6);
  dst14 = LD_UB(dst + 5 * dst_stride);
  dst15 = LD_UB(dst + 10 * dst_stride);
  ILVR_B2_SH(zero, dst14, zero, dst15, res14, res15);
  ADD2(res14, out14, res15, out15, res14, res15);
  CLIP_SH2_0_255(res14, res15);
  PCKEV_B2_SH(res14, res14, res15, res15, res14, res15);
  ST8x1_UB(res14, dst + 5 * dst_stride);
  ST8x1_UB(res15, dst + 10 * dst_stride);
}
