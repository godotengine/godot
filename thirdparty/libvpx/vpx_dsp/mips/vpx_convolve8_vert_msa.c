/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/vpx_convolve_msa.h"

static void common_vt_8t_4w_msa(const uint8_t *src, int32_t src_stride,
                                uint8_t *dst, int32_t dst_stride,
                                int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v16i8 src10_r, src32_r, src54_r, src76_r, src98_r, src21_r, src43_r;
  v16i8 src65_r, src87_r, src109_r, src2110, src4332, src6554, src8776;
  v16i8 src10998, filt0, filt1, filt2, filt3;
  v16u8 out;
  v8i16 filt, out10, out32;

  src -= (3 * src_stride);

  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  LD_SB7(src, src_stride, src0, src1, src2, src3, src4, src5, src6);
  src += (7 * src_stride);

  ILVR_B4_SB(src1, src0, src3, src2, src5, src4, src2, src1, src10_r, src32_r,
             src54_r, src21_r);
  ILVR_B2_SB(src4, src3, src6, src5, src43_r, src65_r);
  ILVR_D3_SB(src21_r, src10_r, src43_r, src32_r, src65_r, src54_r, src2110,
             src4332, src6554);
  XORI_B3_128_SB(src2110, src4332, src6554);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src7, src8, src9, src10);
    src += (4 * src_stride);

    ILVR_B4_SB(src7, src6, src8, src7, src9, src8, src10, src9, src76_r,
               src87_r, src98_r, src109_r);
    ILVR_D2_SB(src87_r, src76_r, src109_r, src98_r, src8776, src10998);
    XORI_B2_128_SB(src8776, src10998);
    out10 = FILT_8TAP_DPADD_S_H(src2110, src4332, src6554, src8776, filt0,
                                filt1, filt2, filt3);
    out32 = FILT_8TAP_DPADD_S_H(src4332, src6554, src8776, src10998, filt0,
                                filt1, filt2, filt3);
    SRARI_H2_SH(out10, out32, FILTER_BITS);
    SAT_SH2_SH(out10, out32, 7);
    out = PCKEV_XORI128_UB(out10, out32);
    ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
    dst += (4 * dst_stride);

    src2110 = src6554;
    src4332 = src8776;
    src6554 = src10998;
    src6 = src10;
  }
}

static void common_vt_8t_8w_msa(const uint8_t *src, int32_t src_stride,
                                uint8_t *dst, int32_t dst_stride,
                                int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v16i8 src10_r, src32_r, src54_r, src76_r, src98_r, src21_r, src43_r;
  v16i8 src65_r, src87_r, src109_r, filt0, filt1, filt2, filt3;
  v16u8 tmp0, tmp1;
  v8i16 filt, out0_r, out1_r, out2_r, out3_r;

  src -= (3 * src_stride);

  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  LD_SB7(src, src_stride, src0, src1, src2, src3, src4, src5, src6);
  XORI_B7_128_SB(src0, src1, src2, src3, src4, src5, src6);
  src += (7 * src_stride);
  ILVR_B4_SB(src1, src0, src3, src2, src5, src4, src2, src1, src10_r, src32_r,
             src54_r, src21_r);
  ILVR_B2_SB(src4, src3, src6, src5, src43_r, src65_r);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src7, src8, src9, src10);
    XORI_B4_128_SB(src7, src8, src9, src10);
    src += (4 * src_stride);

    ILVR_B4_SB(src7, src6, src8, src7, src9, src8, src10, src9, src76_r,
               src87_r, src98_r, src109_r);
    out0_r = FILT_8TAP_DPADD_S_H(src10_r, src32_r, src54_r, src76_r, filt0,
                                 filt1, filt2, filt3);
    out1_r = FILT_8TAP_DPADD_S_H(src21_r, src43_r, src65_r, src87_r, filt0,
                                 filt1, filt2, filt3);
    out2_r = FILT_8TAP_DPADD_S_H(src32_r, src54_r, src76_r, src98_r, filt0,
                                 filt1, filt2, filt3);
    out3_r = FILT_8TAP_DPADD_S_H(src43_r, src65_r, src87_r, src109_r, filt0,
                                 filt1, filt2, filt3);
    SRARI_H4_SH(out0_r, out1_r, out2_r, out3_r, FILTER_BITS);
    SAT_SH4_SH(out0_r, out1_r, out2_r, out3_r, 7);
    tmp0 = PCKEV_XORI128_UB(out0_r, out1_r);
    tmp1 = PCKEV_XORI128_UB(out2_r, out3_r);
    ST8x4_UB(tmp0, tmp1, dst, dst_stride);
    dst += (4 * dst_stride);

    src10_r = src54_r;
    src32_r = src76_r;
    src54_r = src98_r;
    src21_r = src65_r;
    src43_r = src87_r;
    src65_r = src109_r;
    src6 = src10;
  }
}

static void common_vt_8t_16w_msa(const uint8_t *src, int32_t src_stride,
                                 uint8_t *dst, int32_t dst_stride,
                                 int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v16i8 filt0, filt1, filt2, filt3;
  v16i8 src10_r, src32_r, src54_r, src76_r, src98_r, src21_r, src43_r;
  v16i8 src65_r, src87_r, src109_r, src10_l, src32_l, src54_l, src76_l;
  v16i8 src98_l, src21_l, src43_l, src65_l, src87_l, src109_l;
  v16u8 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt, out0_r, out1_r, out2_r, out3_r, out0_l, out1_l, out2_l, out3_l;

  src -= (3 * src_stride);

  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  LD_SB7(src, src_stride, src0, src1, src2, src3, src4, src5, src6);
  XORI_B7_128_SB(src0, src1, src2, src3, src4, src5, src6);
  src += (7 * src_stride);
  ILVR_B4_SB(src1, src0, src3, src2, src5, src4, src2, src1, src10_r, src32_r,
             src54_r, src21_r);
  ILVR_B2_SB(src4, src3, src6, src5, src43_r, src65_r);
  ILVL_B4_SB(src1, src0, src3, src2, src5, src4, src2, src1, src10_l, src32_l,
             src54_l, src21_l);
  ILVL_B2_SB(src4, src3, src6, src5, src43_l, src65_l);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src7, src8, src9, src10);
    XORI_B4_128_SB(src7, src8, src9, src10);
    src += (4 * src_stride);

    ILVR_B4_SB(src7, src6, src8, src7, src9, src8, src10, src9, src76_r,
               src87_r, src98_r, src109_r);
    ILVL_B4_SB(src7, src6, src8, src7, src9, src8, src10, src9, src76_l,
               src87_l, src98_l, src109_l);
    out0_r = FILT_8TAP_DPADD_S_H(src10_r, src32_r, src54_r, src76_r, filt0,
                                 filt1, filt2, filt3);
    out1_r = FILT_8TAP_DPADD_S_H(src21_r, src43_r, src65_r, src87_r, filt0,
                                 filt1, filt2, filt3);
    out2_r = FILT_8TAP_DPADD_S_H(src32_r, src54_r, src76_r, src98_r, filt0,
                                 filt1, filt2, filt3);
    out3_r = FILT_8TAP_DPADD_S_H(src43_r, src65_r, src87_r, src109_r, filt0,
                                 filt1, filt2, filt3);
    out0_l = FILT_8TAP_DPADD_S_H(src10_l, src32_l, src54_l, src76_l, filt0,
                                 filt1, filt2, filt3);
    out1_l = FILT_8TAP_DPADD_S_H(src21_l, src43_l, src65_l, src87_l, filt0,
                                 filt1, filt2, filt3);
    out2_l = FILT_8TAP_DPADD_S_H(src32_l, src54_l, src76_l, src98_l, filt0,
                                 filt1, filt2, filt3);
    out3_l = FILT_8TAP_DPADD_S_H(src43_l, src65_l, src87_l, src109_l, filt0,
                                 filt1, filt2, filt3);
    SRARI_H4_SH(out0_r, out1_r, out2_r, out3_r, FILTER_BITS);
    SRARI_H4_SH(out0_l, out1_l, out2_l, out3_l, FILTER_BITS);
    SAT_SH4_SH(out0_r, out1_r, out2_r, out3_r, 7);
    SAT_SH4_SH(out0_l, out1_l, out2_l, out3_l, 7);
    PCKEV_B4_UB(out0_l, out0_r, out1_l, out1_r, out2_l, out2_r, out3_l, out3_r,
                tmp0, tmp1, tmp2, tmp3);
    XORI_B4_128_UB(tmp0, tmp1, tmp2, tmp3);
    ST_UB4(tmp0, tmp1, tmp2, tmp3, dst, dst_stride);
    dst += (4 * dst_stride);

    src10_r = src54_r;
    src32_r = src76_r;
    src54_r = src98_r;
    src21_r = src65_r;
    src43_r = src87_r;
    src65_r = src109_r;
    src10_l = src54_l;
    src32_l = src76_l;
    src54_l = src98_l;
    src21_l = src65_l;
    src43_l = src87_l;
    src65_l = src109_l;
    src6 = src10;
  }
}

static void common_vt_8t_16w_mult_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter, int32_t height,
                                      int32_t width) {
  const uint8_t *src_tmp;
  uint8_t *dst_tmp;
  uint32_t loop_cnt, cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v16i8 filt0, filt1, filt2, filt3;
  v16i8 src10_r, src32_r, src54_r, src76_r, src98_r, src21_r, src43_r;
  v16i8 src65_r, src87_r, src109_r, src10_l, src32_l, src54_l, src76_l;
  v16i8 src98_l, src21_l, src43_l, src65_l, src87_l, src109_l;
  v16u8 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt, out0_r, out1_r, out2_r, out3_r, out0_l, out1_l, out2_l, out3_l;

  src -= (3 * src_stride);

  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  for (cnt = (width >> 4); cnt--;) {
    src_tmp = src;
    dst_tmp = dst;

    LD_SB7(src_tmp, src_stride, src0, src1, src2, src3, src4, src5, src6);
    XORI_B7_128_SB(src0, src1, src2, src3, src4, src5, src6);
    src_tmp += (7 * src_stride);
    ILVR_B4_SB(src1, src0, src3, src2, src5, src4, src2, src1, src10_r, src32_r,
               src54_r, src21_r);
    ILVR_B2_SB(src4, src3, src6, src5, src43_r, src65_r);
    ILVL_B4_SB(src1, src0, src3, src2, src5, src4, src2, src1, src10_l, src32_l,
               src54_l, src21_l);
    ILVL_B2_SB(src4, src3, src6, src5, src43_l, src65_l);

    for (loop_cnt = (height >> 2); loop_cnt--;) {
      LD_SB4(src_tmp, src_stride, src7, src8, src9, src10);
      XORI_B4_128_SB(src7, src8, src9, src10);
      src_tmp += (4 * src_stride);
      ILVR_B4_SB(src7, src6, src8, src7, src9, src8, src10, src9, src76_r,
                 src87_r, src98_r, src109_r);
      ILVL_B4_SB(src7, src6, src8, src7, src9, src8, src10, src9, src76_l,
                 src87_l, src98_l, src109_l);
      out0_r = FILT_8TAP_DPADD_S_H(src10_r, src32_r, src54_r, src76_r, filt0,
                                   filt1, filt2, filt3);
      out1_r = FILT_8TAP_DPADD_S_H(src21_r, src43_r, src65_r, src87_r, filt0,
                                   filt1, filt2, filt3);
      out2_r = FILT_8TAP_DPADD_S_H(src32_r, src54_r, src76_r, src98_r, filt0,
                                   filt1, filt2, filt3);
      out3_r = FILT_8TAP_DPADD_S_H(src43_r, src65_r, src87_r, src109_r, filt0,
                                   filt1, filt2, filt3);
      out0_l = FILT_8TAP_DPADD_S_H(src10_l, src32_l, src54_l, src76_l, filt0,
                                   filt1, filt2, filt3);
      out1_l = FILT_8TAP_DPADD_S_H(src21_l, src43_l, src65_l, src87_l, filt0,
                                   filt1, filt2, filt3);
      out2_l = FILT_8TAP_DPADD_S_H(src32_l, src54_l, src76_l, src98_l, filt0,
                                   filt1, filt2, filt3);
      out3_l = FILT_8TAP_DPADD_S_H(src43_l, src65_l, src87_l, src109_l, filt0,
                                   filt1, filt2, filt3);
      SRARI_H4_SH(out0_r, out1_r, out2_r, out3_r, FILTER_BITS);
      SRARI_H4_SH(out0_l, out1_l, out2_l, out3_l, FILTER_BITS);
      SAT_SH4_SH(out0_r, out1_r, out2_r, out3_r, 7);
      SAT_SH4_SH(out0_l, out1_l, out2_l, out3_l, 7);
      PCKEV_B4_UB(out0_l, out0_r, out1_l, out1_r, out2_l, out2_r, out3_l,
                  out3_r, tmp0, tmp1, tmp2, tmp3);
      XORI_B4_128_UB(tmp0, tmp1, tmp2, tmp3);
      ST_UB4(tmp0, tmp1, tmp2, tmp3, dst_tmp, dst_stride);
      dst_tmp += (4 * dst_stride);

      src10_r = src54_r;
      src32_r = src76_r;
      src54_r = src98_r;
      src21_r = src65_r;
      src43_r = src87_r;
      src65_r = src109_r;
      src10_l = src54_l;
      src32_l = src76_l;
      src54_l = src98_l;
      src21_l = src65_l;
      src43_l = src87_l;
      src65_l = src109_l;
      src6 = src10;
    }

    src += 16;
    dst += 16;
  }
}

static void common_vt_8t_32w_msa(const uint8_t *src, int32_t src_stride,
                                 uint8_t *dst, int32_t dst_stride,
                                 int8_t *filter, int32_t height) {
  common_vt_8t_16w_mult_msa(src, src_stride, dst, dst_stride, filter, height,
                            32);
}

static void common_vt_8t_64w_msa(const uint8_t *src, int32_t src_stride,
                                 uint8_t *dst, int32_t dst_stride,
                                 int8_t *filter, int32_t height) {
  common_vt_8t_16w_mult_msa(src, src_stride, dst, dst_stride, filter, height,
                            64);
}

static void common_vt_2t_4x4_msa(const uint8_t *src, int32_t src_stride,
                                 uint8_t *dst, int32_t dst_stride,
                                 int8_t *filter) {
  v16i8 src0, src1, src2, src3, src4;
  v16i8 src10_r, src32_r, src21_r, src43_r, src2110, src4332;
  v16u8 filt0;
  v8i16 filt;
  v8u16 tmp0, tmp1;

  filt = LD_SH(filter);
  filt0 = (v16u8)__msa_splati_h(filt, 0);

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  src += (5 * src_stride);

  ILVR_B4_SB(src1, src0, src2, src1, src3, src2, src4, src3, src10_r, src21_r,
             src32_r, src43_r);
  ILVR_D2_SB(src21_r, src10_r, src43_r, src32_r, src2110, src4332);
  DOTP_UB2_UH(src2110, src4332, filt0, filt0, tmp0, tmp1);
  SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
  src2110 = __msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);
  ST4x4_UB(src2110, src2110, 0, 1, 2, 3, dst, dst_stride);
}

static void common_vt_2t_4x8_msa(const uint8_t *src, int32_t src_stride,
                                 uint8_t *dst, int32_t dst_stride,
                                 int8_t *filter) {
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;
  v16i8 src10_r, src32_r, src54_r, src76_r, src21_r, src43_r;
  v16i8 src65_r, src87_r, src2110, src4332, src6554, src8776;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v16u8 filt0;
  v8i16 filt;

  filt = LD_SH(filter);
  filt0 = (v16u8)__msa_splati_h(filt, 0);

  LD_SB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  src += (8 * src_stride);

  src8 = LD_SB(src);
  src += src_stride;

  ILVR_B4_SB(src1, src0, src2, src1, src3, src2, src4, src3, src10_r, src21_r,
             src32_r, src43_r);
  ILVR_B4_SB(src5, src4, src6, src5, src7, src6, src8, src7, src54_r, src65_r,
             src76_r, src87_r);
  ILVR_D4_SB(src21_r, src10_r, src43_r, src32_r, src65_r, src54_r, src87_r,
             src76_r, src2110, src4332, src6554, src8776);
  DOTP_UB4_UH(src2110, src4332, src6554, src8776, filt0, filt0, filt0, filt0,
              tmp0, tmp1, tmp2, tmp3);
  SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, FILTER_BITS);
  PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, src2110, src4332);
  ST4x4_UB(src2110, src2110, 0, 1, 2, 3, dst, dst_stride);
  ST4x4_UB(src4332, src4332, 0, 1, 2, 3, dst + 4 * dst_stride, dst_stride);
}

static void common_vt_2t_4w_msa(const uint8_t *src, int32_t src_stride,
                                uint8_t *dst, int32_t dst_stride,
                                int8_t *filter, int32_t height) {
  if (4 == height) {
    common_vt_2t_4x4_msa(src, src_stride, dst, dst_stride, filter);
  } else if (8 == height) {
    common_vt_2t_4x8_msa(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_vt_2t_8x4_msa(const uint8_t *src, int32_t src_stride,
                                 uint8_t *dst, int32_t dst_stride,
                                 int8_t *filter) {
  v16u8 src0, src1, src2, src3, src4, vec0, vec1, vec2, vec3, filt0;
  v16i8 out0, out1;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  /* rearranging filter_y */
  filt = LD_SH(filter);
  filt0 = (v16u8)__msa_splati_h(filt, 0);

  LD_UB5(src, src_stride, src0, src1, src2, src3, src4);
  ILVR_B2_UB(src1, src0, src2, src1, vec0, vec1);
  ILVR_B2_UB(src3, src2, src4, src3, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, tmp0, tmp1,
              tmp2, tmp3);
  SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, FILTER_BITS);
  PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, out0, out1);
  ST8x4_UB(out0, out1, dst, dst_stride);
}

static void common_vt_2t_8x8mult_msa(const uint8_t *src, int32_t src_stride,
                                     uint8_t *dst, int32_t dst_stride,
                                     int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7, src8;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, filt0;
  v16i8 out0, out1;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  /* rearranging filter_y */
  filt = LD_SH(filter);
  filt0 = (v16u8)__msa_splati_h(filt, 0);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 3); loop_cnt--;) {
    LD_UB8(src, src_stride, src1, src2, src3, src4, src5, src6, src7, src8);
    src += (8 * src_stride);

    ILVR_B4_UB(src1, src0, src2, src1, src3, src2, src4, src3, vec0, vec1, vec2,
               vec3);
    ILVR_B4_UB(src5, src4, src6, src5, src7, src6, src8, src7, vec4, vec5, vec6,
               vec7);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, tmp0, tmp1,
                tmp2, tmp3);
    SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, FILTER_BITS);
    PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, out0, out1);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);

    DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, tmp0, tmp1,
                tmp2, tmp3);
    SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, FILTER_BITS);
    PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, out0, out1);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);

    src0 = src8;
  }
}

static void common_vt_2t_8w_msa(const uint8_t *src, int32_t src_stride,
                                uint8_t *dst, int32_t dst_stride,
                                int8_t *filter, int32_t height) {
  if (4 == height) {
    common_vt_2t_8x4_msa(src, src_stride, dst, dst_stride, filter);
  } else {
    common_vt_2t_8x8mult_msa(src, src_stride, dst, dst_stride, filter, height);
  }
}

static void common_vt_2t_16w_msa(const uint8_t *src, int32_t src_stride,
                                 uint8_t *dst, int32_t dst_stride,
                                 int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, filt0;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  /* rearranging filter_y */
  filt = LD_SH(filter);
  filt0 = (v16u8)__msa_splati_h(filt, 0);

  src0 = LD_UB(src);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);

    ILVR_B2_UB(src1, src0, src2, src1, vec0, vec2);
    ILVL_B2_UB(src1, src0, src2, src1, vec1, vec3);
    DOTP_UB2_UH(vec0, vec1, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_ST_SB(tmp0, tmp1, dst);
    dst += dst_stride;

    ILVR_B2_UB(src3, src2, src4, src3, vec4, vec6);
    ILVL_B2_UB(src3, src2, src4, src3, vec5, vec7);
    DOTP_UB2_UH(vec2, vec3, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_ST_SB(tmp2, tmp3, dst);
    dst += dst_stride;

    DOTP_UB2_UH(vec4, vec5, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_ST_SB(tmp0, tmp1, dst);
    dst += dst_stride;

    DOTP_UB2_UH(vec6, vec7, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_ST_SB(tmp2, tmp3, dst);
    dst += dst_stride;

    src0 = src4;
  }
}

static void common_vt_2t_32w_msa(const uint8_t *src, int32_t src_stride,
                                 uint8_t *dst, int32_t dst_stride,
                                 int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, filt0;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  /* rearranging filter_y */
  filt = LD_SH(filter);
  filt0 = (v16u8)__msa_splati_h(filt, 0);

  src0 = LD_UB(src);
  src5 = LD_UB(src + 16);
  src += src_stride;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_UB4(src, src_stride, src1, src2, src3, src4);
    ILVR_B2_UB(src1, src0, src2, src1, vec0, vec2);
    ILVL_B2_UB(src1, src0, src2, src1, vec1, vec3);

    LD_UB4(src + 16, src_stride, src6, src7, src8, src9);
    src += (4 * src_stride);

    DOTP_UB2_UH(vec0, vec1, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_ST_SB(tmp0, tmp1, dst);
    DOTP_UB2_UH(vec2, vec3, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_ST_SB(tmp2, tmp3, dst + dst_stride);

    ILVR_B2_UB(src3, src2, src4, src3, vec4, vec6);
    ILVL_B2_UB(src3, src2, src4, src3, vec5, vec7);
    DOTP_UB2_UH(vec4, vec5, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_ST_SB(tmp0, tmp1, dst + 2 * dst_stride);

    DOTP_UB2_UH(vec6, vec7, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_ST_SB(tmp2, tmp3, dst + 3 * dst_stride);

    ILVR_B2_UB(src6, src5, src7, src6, vec0, vec2);
    ILVL_B2_UB(src6, src5, src7, src6, vec1, vec3);
    DOTP_UB2_UH(vec0, vec1, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_ST_SB(tmp0, tmp1, dst + 16);

    DOTP_UB2_UH(vec2, vec3, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_ST_SB(tmp2, tmp3, dst + 16 + dst_stride);

    ILVR_B2_UB(src8, src7, src9, src8, vec4, vec6);
    ILVL_B2_UB(src8, src7, src9, src8, vec5, vec7);
    DOTP_UB2_UH(vec4, vec5, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_ST_SB(tmp0, tmp1, dst + 16 + 2 * dst_stride);

    DOTP_UB2_UH(vec6, vec7, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_ST_SB(tmp2, tmp3, dst + 16 + 3 * dst_stride);
    dst += (4 * dst_stride);

    src0 = src4;
    src5 = src9;
  }
}

static void common_vt_2t_64w_msa(const uint8_t *src, int32_t src_stride,
                                 uint8_t *dst, int32_t dst_stride,
                                 int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v16u8 src11, vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, filt0;
  v8u16 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  v8i16 filt;

  /* rearranging filter_y */
  filt = LD_SH(filter);
  filt0 = (v16u8)__msa_splati_h(filt, 0);

  LD_UB4(src, 16, src0, src3, src6, src9);
  src += src_stride;

  for (loop_cnt = (height >> 1); loop_cnt--;) {
    LD_UB2(src, src_stride, src1, src2);
    LD_UB2(src + 16, src_stride, src4, src5);
    LD_UB2(src + 32, src_stride, src7, src8);
    LD_UB2(src + 48, src_stride, src10, src11);
    src += (2 * src_stride);

    ILVR_B2_UB(src1, src0, src2, src1, vec0, vec2);
    ILVL_B2_UB(src1, src0, src2, src1, vec1, vec3);
    DOTP_UB2_UH(vec0, vec1, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_ST_SB(tmp0, tmp1, dst);

    DOTP_UB2_UH(vec2, vec3, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_ST_SB(tmp2, tmp3, dst + dst_stride);

    ILVR_B2_UB(src4, src3, src5, src4, vec4, vec6);
    ILVL_B2_UB(src4, src3, src5, src4, vec5, vec7);
    DOTP_UB2_UH(vec4, vec5, filt0, filt0, tmp4, tmp5);
    SRARI_H2_UH(tmp4, tmp5, FILTER_BITS);
    PCKEV_ST_SB(tmp4, tmp5, dst + 16);

    DOTP_UB2_UH(vec6, vec7, filt0, filt0, tmp6, tmp7);
    SRARI_H2_UH(tmp6, tmp7, FILTER_BITS);
    PCKEV_ST_SB(tmp6, tmp7, dst + 16 + dst_stride);

    ILVR_B2_UB(src7, src6, src8, src7, vec0, vec2);
    ILVL_B2_UB(src7, src6, src8, src7, vec1, vec3);
    DOTP_UB2_UH(vec0, vec1, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_ST_SB(tmp0, tmp1, dst + 32);

    DOTP_UB2_UH(vec2, vec3, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    PCKEV_ST_SB(tmp2, tmp3, dst + 32 + dst_stride);

    ILVR_B2_UB(src10, src9, src11, src10, vec4, vec6);
    ILVL_B2_UB(src10, src9, src11, src10, vec5, vec7);
    DOTP_UB2_UH(vec4, vec5, filt0, filt0, tmp4, tmp5);
    SRARI_H2_UH(tmp4, tmp5, FILTER_BITS);
    PCKEV_ST_SB(tmp4, tmp5, dst + 48);

    DOTP_UB2_UH(vec6, vec7, filt0, filt0, tmp6, tmp7);
    SRARI_H2_UH(tmp6, tmp7, FILTER_BITS);
    PCKEV_ST_SB(tmp6, tmp7, dst + 48 + dst_stride);
    dst += (2 * dst_stride);

    src0 = src2;
    src3 = src5;
    src6 = src8;
    src9 = src11;
  }
}

void vpx_convolve8_vert_msa(const uint8_t *src, ptrdiff_t src_stride,
                            uint8_t *dst, ptrdiff_t dst_stride,
                            const InterpKernel *filter, int x0_q4,
                            int32_t x_step_q4, int y0_q4, int y_step_q4, int w,
                            int h) {
  const int16_t *const filter_y = filter[y0_q4];
  int8_t cnt, filt_ver[8];

  assert(y_step_q4 == 16);
  assert(((const int32_t *)filter_y)[1] != 0x800000);

  for (cnt = 8; cnt--;) {
    filt_ver[cnt] = filter_y[cnt];
  }

  if (vpx_get_filter_taps(filter_y) == 2) {
    switch (w) {
      case 4:
        common_vt_2t_4w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                            &filt_ver[3], h);
        break;
      case 8:
        common_vt_2t_8w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                            &filt_ver[3], h);
        break;
      case 16:
        common_vt_2t_16w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                             &filt_ver[3], h);
        break;
      case 32:
        common_vt_2t_32w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                             &filt_ver[3], h);
        break;
      case 64:
        common_vt_2t_64w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                             &filt_ver[3], h);
        break;
      default:
        vpx_convolve8_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                             x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  } else {
    switch (w) {
      case 4:
        common_vt_8t_4w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                            filt_ver, h);
        break;
      case 8:
        common_vt_8t_8w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                            filt_ver, h);
        break;
      case 16:
        common_vt_8t_16w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                             filt_ver, h);
        break;
      case 32:
        common_vt_8t_32w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                             filt_ver, h);
        break;
      case 64:
        common_vt_8t_64w_msa(src, (int32_t)src_stride, dst, (int32_t)dst_stride,
                             filt_ver, h);
        break;
      default:
        vpx_convolve8_vert_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                             x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  }
}
