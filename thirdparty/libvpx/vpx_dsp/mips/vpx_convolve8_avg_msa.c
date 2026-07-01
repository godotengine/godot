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

static void common_hv_8ht_8vt_and_aver_dst_4w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  uint32_t loop_cnt;
  uint32_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v16u8 dst0 = { 0 }, mask0, mask1, mask2, mask3, res;
  v16i8 filt_hz0, filt_hz1, filt_hz2, filt_hz3;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8i16 hz_out7, hz_out8, hz_out9, res0, res1, vec0, vec1, vec2, vec3, vec4;
  v8i16 filt, filt_vt0, filt_vt1, filt_vt2, filt_vt3;

  mask0 = LD_UB(&mc_filt_mask_arr[16]);
  src -= (3 + 3 * src_stride);

  /* rearranging filter */
  filt = LD_SH(filter_horiz);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt_hz0, filt_hz1, filt_hz2, filt_hz3);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;
  mask3 = mask0 + 6;

  LD_SB7(src, src_stride, src0, src1, src2, src3, src4, src5, src6);
  XORI_B7_128_SB(src0, src1, src2, src3, src4, src5, src6);
  src += (7 * src_stride);

  hz_out0 = HORIZ_8TAP_FILT(src0, src1, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  hz_out2 = HORIZ_8TAP_FILT(src2, src3, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  hz_out4 = HORIZ_8TAP_FILT(src4, src5, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  hz_out5 = HORIZ_8TAP_FILT(src5, src6, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  SLDI_B2_SH(hz_out2, hz_out4, hz_out0, hz_out2, hz_out1, hz_out3, 8);

  filt = LD_SH(filter_vert);
  SPLATI_H4_SH(filt, 0, 1, 2, 3, filt_vt0, filt_vt1, filt_vt2, filt_vt3);

  ILVEV_B2_SH(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
  vec2 = (v8i16)__msa_ilvev_b((v16i8)hz_out5, (v16i8)hz_out4);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src7, src8, src9, src10);
    XORI_B4_128_SB(src7, src8, src9, src10);
    src += (4 * src_stride);

    LW4(dst, dst_stride, tp0, tp1, tp2, tp3);
    INSERT_W4_UB(tp0, tp1, tp2, tp3, dst0);
    hz_out7 = HORIZ_8TAP_FILT(src7, src8, mask0, mask1, mask2, mask3, filt_hz0,
                              filt_hz1, filt_hz2, filt_hz3);
    hz_out6 = (v8i16)__msa_sldi_b((v16i8)hz_out7, (v16i8)hz_out5, 8);
    vec3 = (v8i16)__msa_ilvev_b((v16i8)hz_out7, (v16i8)hz_out6);
    res0 = FILT_8TAP_DPADD_S_H(vec0, vec1, vec2, vec3, filt_vt0, filt_vt1,
                               filt_vt2, filt_vt3);

    hz_out9 = HORIZ_8TAP_FILT(src9, src10, mask0, mask1, mask2, mask3, filt_hz0,
                              filt_hz1, filt_hz2, filt_hz3);
    hz_out8 = (v8i16)__msa_sldi_b((v16i8)hz_out9, (v16i8)hz_out7, 8);
    vec4 = (v8i16)__msa_ilvev_b((v16i8)hz_out9, (v16i8)hz_out8);
    res1 = FILT_8TAP_DPADD_S_H(vec1, vec2, vec3, vec4, filt_vt0, filt_vt1,
                               filt_vt2, filt_vt3);

    SRARI_H2_SH(res0, res1, FILTER_BITS);
    SAT_SH2_SH(res0, res1, 7);
    res = PCKEV_XORI128_UB(res0, res1);
    res = (v16u8)__msa_aver_u_b(res, dst0);
    ST4x4_UB(res, res, 0, 1, 2, 3, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out5 = hz_out9;
    vec0 = vec2;
    vec1 = vec3;
    vec2 = vec4;
  }
}

static void common_hv_8ht_8vt_and_aver_dst_8w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  uint32_t loop_cnt;
  uint64_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v16i8 filt_hz0, filt_hz1, filt_hz2, filt_hz3;
  v8i16 filt, filt_vt0, filt_vt1, filt_vt2, filt_vt3;
  v16u8 dst0 = { 0 }, dst1 = { 0 }, mask0, mask1, mask2, mask3;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8i16 hz_out7, hz_out8, hz_out9, hz_out10, tmp0, tmp1, tmp2, tmp3;
  v8i16 out0, out1, out2, out3, out4, out5, out6, out7, out8, out9;

  mask0 = LD_UB(&mc_filt_mask_arr[0]);
  src -= (3 + 3 * src_stride);

  /* rearranging filter */
  filt = LD_SH(filter_horiz);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt_hz0, filt_hz1, filt_hz2, filt_hz3);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;
  mask3 = mask0 + 6;

  LD_SB7(src, src_stride, src0, src1, src2, src3, src4, src5, src6);
  src += (7 * src_stride);

  XORI_B7_128_SB(src0, src1, src2, src3, src4, src5, src6);
  hz_out0 = HORIZ_8TAP_FILT(src0, src0, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  hz_out1 = HORIZ_8TAP_FILT(src1, src1, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  hz_out2 = HORIZ_8TAP_FILT(src2, src2, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  hz_out3 = HORIZ_8TAP_FILT(src3, src3, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  hz_out4 = HORIZ_8TAP_FILT(src4, src4, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  hz_out5 = HORIZ_8TAP_FILT(src5, src5, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);
  hz_out6 = HORIZ_8TAP_FILT(src6, src6, mask0, mask1, mask2, mask3, filt_hz0,
                            filt_hz1, filt_hz2, filt_hz3);

  filt = LD_SH(filter_vert);
  SPLATI_H4_SH(filt, 0, 1, 2, 3, filt_vt0, filt_vt1, filt_vt2, filt_vt3);

  ILVEV_B2_SH(hz_out0, hz_out1, hz_out2, hz_out3, out0, out1);
  ILVEV_B2_SH(hz_out4, hz_out5, hz_out1, hz_out2, out2, out4);
  ILVEV_B2_SH(hz_out3, hz_out4, hz_out5, hz_out6, out5, out6);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src7, src8, src9, src10);
    XORI_B4_128_SB(src7, src8, src9, src10);
    src += (4 * src_stride);

    LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
    INSERT_D2_UB(tp0, tp1, dst0);
    INSERT_D2_UB(tp2, tp3, dst1);

    hz_out7 = HORIZ_8TAP_FILT(src7, src7, mask0, mask1, mask2, mask3, filt_hz0,
                              filt_hz1, filt_hz2, filt_hz3);
    out3 = (v8i16)__msa_ilvev_b((v16i8)hz_out7, (v16i8)hz_out6);
    tmp0 = FILT_8TAP_DPADD_S_H(out0, out1, out2, out3, filt_vt0, filt_vt1,
                               filt_vt2, filt_vt3);

    hz_out8 = HORIZ_8TAP_FILT(src8, src8, mask0, mask1, mask2, mask3, filt_hz0,
                              filt_hz1, filt_hz2, filt_hz3);
    out7 = (v8i16)__msa_ilvev_b((v16i8)hz_out8, (v16i8)hz_out7);
    tmp1 = FILT_8TAP_DPADD_S_H(out4, out5, out6, out7, filt_vt0, filt_vt1,
                               filt_vt2, filt_vt3);

    hz_out9 = HORIZ_8TAP_FILT(src9, src9, mask0, mask1, mask2, mask3, filt_hz0,
                              filt_hz1, filt_hz2, filt_hz3);
    out8 = (v8i16)__msa_ilvev_b((v16i8)hz_out9, (v16i8)hz_out8);
    tmp2 = FILT_8TAP_DPADD_S_H(out1, out2, out3, out8, filt_vt0, filt_vt1,
                               filt_vt2, filt_vt3);

    hz_out10 = HORIZ_8TAP_FILT(src10, src10, mask0, mask1, mask2, mask3,
                               filt_hz0, filt_hz1, filt_hz2, filt_hz3);
    out9 = (v8i16)__msa_ilvev_b((v16i8)hz_out10, (v16i8)hz_out9);
    tmp3 = FILT_8TAP_DPADD_S_H(out5, out6, out7, out9, filt_vt0, filt_vt1,
                               filt_vt2, filt_vt3);

    SRARI_H4_SH(tmp0, tmp1, tmp2, tmp3, FILTER_BITS);
    SAT_SH4_SH(tmp0, tmp1, tmp2, tmp3, 7);
    CONVERT_UB_AVG_ST8x4_UB(tmp0, tmp1, tmp2, tmp3, dst0, dst1, dst,
                            dst_stride);
    dst += (4 * dst_stride);

    hz_out6 = hz_out10;
    out0 = out2;
    out1 = out3;
    out2 = out8;
    out4 = out6;
    out5 = out7;
    out6 = out9;
  }
}

static void common_hv_8ht_8vt_and_aver_dst_16w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 2; multiple8_cnt--;) {
    common_hv_8ht_8vt_and_aver_dst_8w_msa(src, src_stride, dst, dst_stride,
                                          filter_horiz, filter_vert, height);
    src += 8;
    dst += 8;
  }
}

static void common_hv_8ht_8vt_and_aver_dst_32w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 4; multiple8_cnt--;) {
    common_hv_8ht_8vt_and_aver_dst_8w_msa(src, src_stride, dst, dst_stride,
                                          filter_horiz, filter_vert, height);
    src += 8;
    dst += 8;
  }
}

static void common_hv_8ht_8vt_and_aver_dst_64w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 8; multiple8_cnt--;) {
    common_hv_8ht_8vt_and_aver_dst_8w_msa(src, src_stride, dst, dst_stride,
                                          filter_horiz, filter_vert, height);
    src += 8;
    dst += 8;
  }
}

static void common_hv_2ht_2vt_and_aver_dst_4x4_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert) {
  uint32_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, src4, mask;
  v16u8 filt_hz, filt_vt, vec0, vec1;
  v16u8 dst0 = { 0 }, out;
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, tmp0, tmp1, filt;

  mask = LD_SB(&mc_filt_mask_arr[16]);

  /* rearranging filter */
  filt = LD_UH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h((v8i16)filt, 0);

  filt = LD_UH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src1, mask, filt_hz, FILTER_BITS);
  hz_out2 = HORIZ_2TAP_FILT_UH(src2, src3, mask, filt_hz, FILTER_BITS);
  hz_out4 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
  hz_out1 = (v8u16)__msa_sldi_b((v16i8)hz_out2, (v16i8)hz_out0, 8);
  hz_out3 = (v8u16)__msa_pckod_d((v2i64)hz_out4, (v2i64)hz_out2);
  ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);

  LW4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_W4_UB(tp0, tp1, tp2, tp3, dst0);
  DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
  SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
  out = (v16u8)__msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);
  out = __msa_aver_u_b(out, dst0);
  ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
}

static void common_hv_2ht_2vt_and_aver_dst_4x8_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert) {
  uint32_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, mask;
  v16u8 filt_hz, filt_vt, vec0, vec1, vec2, vec3, res0, res1;
  v16u8 dst0 = { 0 }, dst1 = { 0 };
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8u16 hz_out7, hz_out8, tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  mask = LD_SB(&mc_filt_mask_arr[16]);

  /* rearranging filter */
  filt = LD_SH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h(filt, 0);

  filt = LD_SH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h(filt, 0);

  LD_SB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  src += (8 * src_stride);
  src8 = LD_SB(src);

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src1, mask, filt_hz, FILTER_BITS);
  hz_out2 = HORIZ_2TAP_FILT_UH(src2, src3, mask, filt_hz, FILTER_BITS);
  hz_out4 = HORIZ_2TAP_FILT_UH(src4, src5, mask, filt_hz, FILTER_BITS);
  hz_out6 = HORIZ_2TAP_FILT_UH(src6, src7, mask, filt_hz, FILTER_BITS);
  hz_out8 = HORIZ_2TAP_FILT_UH(src8, src8, mask, filt_hz, FILTER_BITS);
  SLDI_B3_UH(hz_out2, hz_out4, hz_out6, hz_out0, hz_out2, hz_out4, hz_out1,
             hz_out3, hz_out5, 8);
  hz_out7 = (v8u16)__msa_pckod_d((v2i64)hz_out8, (v2i64)hz_out6);

  LW4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_W4_UB(tp0, tp1, tp2, tp3, dst0);
  LW4(dst + 4 * dst_stride, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_W4_UB(tp0, tp1, tp2, tp3, dst1);
  ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
  ILVEV_B2_UB(hz_out4, hz_out5, hz_out6, hz_out7, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt_vt, filt_vt, filt_vt, filt_vt, tmp0,
              tmp1, tmp2, tmp3);
  SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, FILTER_BITS);
  PCKEV_B2_UB(tmp1, tmp0, tmp3, tmp2, res0, res1);
  AVER_UB2_UB(res0, dst0, res1, dst1, res0, res1);
  ST4x8_UB(res0, res1, dst, dst_stride);
}

static void common_hv_2ht_2vt_and_aver_dst_4w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  if (4 == height) {
    common_hv_2ht_2vt_and_aver_dst_4x4_msa(src, src_stride, dst, dst_stride,
                                           filter_horiz, filter_vert);
  } else if (8 == height) {
    common_hv_2ht_2vt_and_aver_dst_4x8_msa(src, src_stride, dst, dst_stride,
                                           filter_horiz, filter_vert);
  }
}

static void common_hv_2ht_2vt_and_aver_dst_8x4_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert) {
  uint64_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, src4, mask;
  v16u8 filt_hz, filt_vt, dst0 = { 0 }, dst1 = { 0 }, vec0, vec1, vec2, vec3;
  v8u16 hz_out0, hz_out1, tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  mask = LD_SB(&mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_SH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h(filt, 0);

  filt = LD_SH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h(filt, 0);

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  src += (5 * src_stride);

  LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_D2_UB(tp0, tp1, dst0);
  INSERT_D2_UB(tp2, tp3, dst1);
  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);
  hz_out1 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);
  vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
  tmp0 = __msa_dotp_u_h(vec0, filt_vt);

  hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);
  vec1 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
  tmp1 = __msa_dotp_u_h(vec1, filt_vt);

  hz_out1 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);
  vec2 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
  tmp2 = __msa_dotp_u_h(vec2, filt_vt);

  hz_out0 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
  vec3 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
  tmp3 = __msa_dotp_u_h(vec3, filt_vt);

  SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, FILTER_BITS);
  PCKEV_AVG_ST8x4_UB(tmp0, tmp1, tmp2, tmp3, dst0, dst1, dst, dst_stride);
}

static void common_hv_2ht_2vt_and_aver_dst_8x8mult_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  uint32_t loop_cnt;
  uint64_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, src4, mask;
  v16u8 filt_hz, filt_vt, vec0, dst0 = { 0 }, dst1 = { 0 };
  v8u16 hz_out0, hz_out1, tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  mask = LD_SB(&mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_SH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h(filt, 0);

  filt = LD_SH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h(filt, 0);

  src0 = LD_SB(src);
  src += src_stride;

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);

    hz_out1 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp0 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp1 = __msa_dotp_u_h(vec0, filt_vt);

    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);

    hz_out1 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp2 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp3 = __msa_dotp_u_h(vec0, filt_vt);

    SRARI_H2_UH(tmp2, tmp3, FILTER_BITS);
    LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
    INSERT_D2_UB(tp0, tp1, dst0);
    INSERT_D2_UB(tp2, tp3, dst1);
    PCKEV_AVG_ST8x4_UB(tmp0, tmp1, tmp2, tmp3, dst0, dst1, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void common_hv_2ht_2vt_and_aver_dst_8w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  if (4 == height) {
    common_hv_2ht_2vt_and_aver_dst_8x4_msa(src, src_stride, dst, dst_stride,
                                           filter_horiz, filter_vert);
  } else {
    common_hv_2ht_2vt_and_aver_dst_8x8mult_msa(
        src, src_stride, dst, dst_stride, filter_horiz, filter_vert, height);
  }
}

static void common_hv_2ht_2vt_and_aver_dst_16w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask;
  v16u8 filt_hz, filt_vt, vec0, vec1, dst0, dst1, dst2, dst3;
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, tmp0, tmp1;
  v8i16 filt;

  mask = LD_SB(&mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_SH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h(filt, 0);

  filt = LD_SH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h(filt, 0);

  LD_SB2(src, 8, src0, src1);
  src += src_stride;

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);
  hz_out2 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src2, src4, src6);
    LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);
    LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);

    hz_out1 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);
    hz_out3 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_AVG_ST_UB(tmp1, tmp0, dst0, dst);
    dst += dst_stride;

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_AVG_ST_UB(tmp1, tmp0, dst1, dst);
    dst += dst_stride;

    hz_out1 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    hz_out3 = HORIZ_2TAP_FILT_UH(src5, src5, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_AVG_ST_UB(tmp1, tmp0, dst2, dst);
    dst += dst_stride;

    hz_out0 = HORIZ_2TAP_FILT_UH(src6, src6, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src7, src7, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
    PCKEV_AVG_ST_UB(tmp1, tmp0, dst3, dst);
    dst += dst_stride;
  }
}

static void common_hv_2ht_2vt_and_aver_dst_32w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 2; multiple8_cnt--;) {
    common_hv_2ht_2vt_and_aver_dst_16w_msa(src, src_stride, dst, dst_stride,
                                           filter_horiz, filter_vert, height);
    src += 16;
    dst += 16;
  }
}

static void common_hv_2ht_2vt_and_aver_dst_64w_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter_horiz, int8_t *filter_vert, int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 4; multiple8_cnt--;) {
    common_hv_2ht_2vt_and_aver_dst_16w_msa(src, src_stride, dst, dst_stride,
                                           filter_horiz, filter_vert, height);
    src += 16;
    dst += 16;
  }
}

void vpx_convolve8_avg_msa(const uint8_t *src, ptrdiff_t src_stride,
                           uint8_t *dst, ptrdiff_t dst_stride,
                           const InterpKernel *filter, int x0_q4, int x_step_q4,
                           int y0_q4, int y_step_q4, int w, int h) {
  const int16_t *const filter_x = filter[x0_q4];
  const int16_t *const filter_y = filter[y0_q4];
  int8_t cnt, filt_hor[8], filt_ver[8];

  assert(x_step_q4 == 16);
  assert(y_step_q4 == 16);
  assert(((const int32_t *)filter_x)[1] != 0x800000);
  assert(((const int32_t *)filter_y)[1] != 0x800000);

  for (cnt = 0; cnt < 8; ++cnt) {
    filt_hor[cnt] = filter_x[cnt];
    filt_ver[cnt] = filter_y[cnt];
  }

  if (vpx_get_filter_taps(filter_x) == 2 &&
      vpx_get_filter_taps(filter_y) == 2) {
    switch (w) {
      case 4:
        common_hv_2ht_2vt_and_aver_dst_4w_msa(src, (int32_t)src_stride, dst,
                                              (int32_t)dst_stride, &filt_hor[3],
                                              &filt_ver[3], h);
        break;
      case 8:
        common_hv_2ht_2vt_and_aver_dst_8w_msa(src, (int32_t)src_stride, dst,
                                              (int32_t)dst_stride, &filt_hor[3],
                                              &filt_ver[3], h);
        break;
      case 16:
        common_hv_2ht_2vt_and_aver_dst_16w_msa(src, (int32_t)src_stride, dst,
                                               (int32_t)dst_stride,
                                               &filt_hor[3], &filt_ver[3], h);
        break;
      case 32:
        common_hv_2ht_2vt_and_aver_dst_32w_msa(src, (int32_t)src_stride, dst,
                                               (int32_t)dst_stride,
                                               &filt_hor[3], &filt_ver[3], h);
        break;
      case 64:
        common_hv_2ht_2vt_and_aver_dst_64w_msa(src, (int32_t)src_stride, dst,
                                               (int32_t)dst_stride,
                                               &filt_hor[3], &filt_ver[3], h);
        break;
      default:
        vpx_convolve8_avg_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                            x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  } else if (vpx_get_filter_taps(filter_x) == 2 ||
             vpx_get_filter_taps(filter_y) == 2) {
    vpx_convolve8_avg_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                        x_step_q4, y0_q4, y_step_q4, w, h);
  } else {
    switch (w) {
      case 4:
        common_hv_8ht_8vt_and_aver_dst_4w_msa(src, (int32_t)src_stride, dst,
                                              (int32_t)dst_stride, filt_hor,
                                              filt_ver, h);
        break;
      case 8:
        common_hv_8ht_8vt_and_aver_dst_8w_msa(src, (int32_t)src_stride, dst,
                                              (int32_t)dst_stride, filt_hor,
                                              filt_ver, h);
        break;
      case 16:
        common_hv_8ht_8vt_and_aver_dst_16w_msa(src, (int32_t)src_stride, dst,
                                               (int32_t)dst_stride, filt_hor,
                                               filt_ver, h);
        break;
      case 32:
        common_hv_8ht_8vt_and_aver_dst_32w_msa(src, (int32_t)src_stride, dst,
                                               (int32_t)dst_stride, filt_hor,
                                               filt_ver, h);
        break;
      case 64:
        common_hv_8ht_8vt_and_aver_dst_64w_msa(src, (int32_t)src_stride, dst,
                                               (int32_t)dst_stride, filt_hor,
                                               filt_ver, h);
        break;
      default:
        vpx_convolve8_avg_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                            x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  }
}
