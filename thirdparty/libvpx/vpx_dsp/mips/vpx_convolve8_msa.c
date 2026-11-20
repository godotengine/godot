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

const uint8_t mc_filt_mask_arr[16 * 3] = {
  /* 8 width cases */
  0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
  /* 4 width cases */
  0, 1, 1, 2, 2, 3, 3, 4, 16, 17, 17, 18, 18, 19, 19, 20,
  /* 4 width cases */
  8, 9, 9, 10, 10, 11, 11, 12, 24, 25, 25, 26, 26, 27, 27, 28
};

static void common_hv_8ht_8vt_4w_msa(const uint8_t *src, int32_t src_stride,
                                     uint8_t *dst, int32_t dst_stride,
                                     int8_t *filter_horiz, int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v16i8 filt_hz0, filt_hz1, filt_hz2, filt_hz3;
  v16u8 mask0, mask1, mask2, mask3, out;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8i16 hz_out7, hz_out8, hz_out9, tmp0, tmp1, out0, out1, out2, out3, out4;
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

  ILVEV_B2_SH(hz_out0, hz_out1, hz_out2, hz_out3, out0, out1);
  out2 = (v8i16)__msa_ilvev_b((v16i8)hz_out5, (v16i8)hz_out4);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src7, src8, src9, src10);
    XORI_B4_128_SB(src7, src8, src9, src10);
    src += (4 * src_stride);

    hz_out7 = HORIZ_8TAP_FILT(src7, src8, mask0, mask1, mask2, mask3, filt_hz0,
                              filt_hz1, filt_hz2, filt_hz3);
    hz_out6 = (v8i16)__msa_sldi_b((v16i8)hz_out7, (v16i8)hz_out5, 8);
    out3 = (v8i16)__msa_ilvev_b((v16i8)hz_out7, (v16i8)hz_out6);
    tmp0 = FILT_8TAP_DPADD_S_H(out0, out1, out2, out3, filt_vt0, filt_vt1,
                               filt_vt2, filt_vt3);

    hz_out9 = HORIZ_8TAP_FILT(src9, src10, mask0, mask1, mask2, mask3, filt_hz0,
                              filt_hz1, filt_hz2, filt_hz3);
    hz_out8 = (v8i16)__msa_sldi_b((v16i8)hz_out9, (v16i8)hz_out7, 8);
    out4 = (v8i16)__msa_ilvev_b((v16i8)hz_out9, (v16i8)hz_out8);
    tmp1 = FILT_8TAP_DPADD_S_H(out1, out2, out3, out4, filt_vt0, filt_vt1,
                               filt_vt2, filt_vt3);
    SRARI_H2_SH(tmp0, tmp1, FILTER_BITS);
    SAT_SH2_SH(tmp0, tmp1, 7);
    out = PCKEV_XORI128_UB(tmp0, tmp1);
    ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out5 = hz_out9;
    out0 = out2;
    out1 = out3;
    out2 = out4;
  }
}

static void common_hv_8ht_8vt_8w_msa(const uint8_t *src, int32_t src_stride,
                                     uint8_t *dst, int32_t dst_stride,
                                     int8_t *filter_horiz, int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  v16i8 filt_hz0, filt_hz1, filt_hz2, filt_hz3;
  v16u8 mask0, mask1, mask2, mask3, vec0, vec1;
  v8i16 filt, filt_vt0, filt_vt1, filt_vt2, filt_vt3;
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
    src += (4 * src_stride);

    XORI_B4_128_SB(src7, src8, src9, src10);

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
    vec0 = PCKEV_XORI128_UB(tmp0, tmp1);
    vec1 = PCKEV_XORI128_UB(tmp2, tmp3);
    ST8x4_UB(vec0, vec1, dst, dst_stride);
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

static void common_hv_8ht_8vt_16w_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter_horiz, int8_t *filter_vert,
                                      int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 2; multiple8_cnt--;) {
    common_hv_8ht_8vt_8w_msa(src, src_stride, dst, dst_stride, filter_horiz,
                             filter_vert, height);
    src += 8;
    dst += 8;
  }
}

static void common_hv_8ht_8vt_32w_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter_horiz, int8_t *filter_vert,
                                      int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 4; multiple8_cnt--;) {
    common_hv_8ht_8vt_8w_msa(src, src_stride, dst, dst_stride, filter_horiz,
                             filter_vert, height);
    src += 8;
    dst += 8;
  }
}

static void common_hv_8ht_8vt_64w_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter_horiz, int8_t *filter_vert,
                                      int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 8; multiple8_cnt--;) {
    common_hv_8ht_8vt_8w_msa(src, src_stride, dst, dst_stride, filter_horiz,
                             filter_vert, height);
    src += 8;
    dst += 8;
  }
}

static void common_hv_2ht_2vt_4x4_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter_horiz,
                                      int8_t *filter_vert) {
  v16i8 src0, src1, src2, src3, src4, mask;
  v16u8 filt_vt, filt_hz, vec0, vec1, res0, res1;
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, filt, tmp0, tmp1;

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
  DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
  SRARI_H2_UH(tmp0, tmp1, FILTER_BITS);
  PCKEV_B2_UB(tmp0, tmp0, tmp1, tmp1, res0, res1);
  ST4x4_UB(res0, res1, 0, 1, 0, 1, dst, dst_stride);
}

static void common_hv_2ht_2vt_4x8_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter_horiz,
                                      int8_t *filter_vert) {
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, mask;
  v16i8 res0, res1, res2, res3;
  v16u8 filt_hz, filt_vt, vec0, vec1, vec2, vec3;
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8u16 hz_out7, hz_out8, vec4, vec5, vec6, vec7, filt;

  mask = LD_SB(&mc_filt_mask_arr[16]);

  /* rearranging filter */
  filt = LD_UH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h((v8i16)filt, 0);

  filt = LD_UH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h((v8i16)filt, 0);

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

  ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
  ILVEV_B2_UB(hz_out4, hz_out5, hz_out6, hz_out7, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt_vt, filt_vt, filt_vt, filt_vt, vec4,
              vec5, vec6, vec7);
  SRARI_H4_UH(vec4, vec5, vec6, vec7, FILTER_BITS);
  PCKEV_B4_SB(vec4, vec4, vec5, vec5, vec6, vec6, vec7, vec7, res0, res1, res2,
              res3);
  ST4x4_UB(res0, res1, 0, 1, 0, 1, dst, dst_stride);
  dst += (4 * dst_stride);
  ST4x4_UB(res2, res3, 0, 1, 0, 1, dst, dst_stride);
}

static void common_hv_2ht_2vt_4w_msa(const uint8_t *src, int32_t src_stride,
                                     uint8_t *dst, int32_t dst_stride,
                                     int8_t *filter_horiz, int8_t *filter_vert,
                                     int32_t height) {
  if (4 == height) {
    common_hv_2ht_2vt_4x4_msa(src, src_stride, dst, dst_stride, filter_horiz,
                              filter_vert);
  } else if (8 == height) {
    common_hv_2ht_2vt_4x8_msa(src, src_stride, dst, dst_stride, filter_horiz,
                              filter_vert);
  }
}

static void common_hv_2ht_2vt_8x4_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter_horiz,
                                      int8_t *filter_vert) {
  v16i8 src0, src1, src2, src3, src4, mask, out0, out1;
  v16u8 filt_hz, filt_vt, vec0, vec1, vec2, vec3;
  v8u16 hz_out0, hz_out1, tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  mask = LD_SB(&mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_SH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h(filt, 0);

  filt = LD_SH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h(filt, 0);

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

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
  PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, out0, out1);
  ST8x4_UB(out0, out1, dst, dst_stride);
}

static void common_hv_2ht_2vt_8x8mult_msa(const uint8_t *src,
                                          int32_t src_stride, uint8_t *dst,
                                          int32_t dst_stride,
                                          int8_t *filter_horiz,
                                          int8_t *filter_vert, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, mask, out0, out1;
  v16u8 filt_hz, filt_vt, vec0;
  v8u16 hz_out0, hz_out1, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
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

  for (loop_cnt = (height >> 3); loop_cnt--;) {
    LD_SB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);

    hz_out1 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp1 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp2 = __msa_dotp_u_h(vec0, filt_vt);

    SRARI_H2_UH(tmp1, tmp2, FILTER_BITS);

    hz_out1 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp3 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    LD_SB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp4 = __msa_dotp_u_h(vec0, filt_vt);

    SRARI_H2_UH(tmp3, tmp4, FILTER_BITS);
    PCKEV_B2_SB(tmp2, tmp1, tmp4, tmp3, out0, out1);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out1 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp5 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp6 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out1 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp7 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp8 = __msa_dotp_u_h(vec0, filt_vt);

    SRARI_H4_UH(tmp5, tmp6, tmp7, tmp8, FILTER_BITS);
    PCKEV_B2_SB(tmp6, tmp5, tmp8, tmp7, out0, out1);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void common_hv_2ht_2vt_8w_msa(const uint8_t *src, int32_t src_stride,
                                     uint8_t *dst, int32_t dst_stride,
                                     int8_t *filter_horiz, int8_t *filter_vert,
                                     int32_t height) {
  if (4 == height) {
    common_hv_2ht_2vt_8x4_msa(src, src_stride, dst, dst_stride, filter_horiz,
                              filter_vert);
  } else {
    common_hv_2ht_2vt_8x8mult_msa(src, src_stride, dst, dst_stride,
                                  filter_horiz, filter_vert, height);
  }
}

static void common_hv_2ht_2vt_16w_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter_horiz, int8_t *filter_vert,
                                      int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask;
  v16u8 filt_hz, filt_vt, vec0, vec1;
  v8u16 tmp1, tmp2, hz_out0, hz_out1, hz_out2, hz_out3;
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

    hz_out1 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, FILTER_BITS);
    hz_out3 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp1, tmp2);
    SRARI_H2_UH(tmp1, tmp2, FILTER_BITS);
    PCKEV_ST_SB(tmp1, tmp2, dst);
    dst += dst_stride;

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp1, tmp2);
    SRARI_H2_UH(tmp1, tmp2, FILTER_BITS);
    PCKEV_ST_SB(tmp1, tmp2, dst);
    dst += dst_stride;

    hz_out1 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, FILTER_BITS);
    hz_out3 = HORIZ_2TAP_FILT_UH(src5, src5, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp1, tmp2);
    SRARI_H2_UH(tmp1, tmp2, FILTER_BITS);
    PCKEV_ST_SB(tmp1, tmp2, dst);
    dst += dst_stride;

    hz_out0 = HORIZ_2TAP_FILT_UH(src6, src6, mask, filt_hz, FILTER_BITS);
    hz_out2 = HORIZ_2TAP_FILT_UH(src7, src7, mask, filt_hz, FILTER_BITS);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp1, tmp2);
    SRARI_H2_UH(tmp1, tmp2, FILTER_BITS);
    PCKEV_ST_SB(tmp1, tmp2, dst);
    dst += dst_stride;
  }
}

static void common_hv_2ht_2vt_32w_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter_horiz, int8_t *filter_vert,
                                      int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 2; multiple8_cnt--;) {
    common_hv_2ht_2vt_16w_msa(src, src_stride, dst, dst_stride, filter_horiz,
                              filter_vert, height);
    src += 16;
    dst += 16;
  }
}

static void common_hv_2ht_2vt_64w_msa(const uint8_t *src, int32_t src_stride,
                                      uint8_t *dst, int32_t dst_stride,
                                      int8_t *filter_horiz, int8_t *filter_vert,
                                      int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 4; multiple8_cnt--;) {
    common_hv_2ht_2vt_16w_msa(src, src_stride, dst, dst_stride, filter_horiz,
                              filter_vert, height);
    src += 16;
    dst += 16;
  }
}

void vpx_convolve8_msa(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                       ptrdiff_t dst_stride, const InterpKernel *filter,
                       int x0_q4, int32_t x_step_q4, int y0_q4,
                       int32_t y_step_q4, int32_t w, int32_t h) {
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
        common_hv_2ht_2vt_4w_msa(src, (int32_t)src_stride, dst,
                                 (int32_t)dst_stride, &filt_hor[3],
                                 &filt_ver[3], (int32_t)h);
        break;
      case 8:
        common_hv_2ht_2vt_8w_msa(src, (int32_t)src_stride, dst,
                                 (int32_t)dst_stride, &filt_hor[3],
                                 &filt_ver[3], (int32_t)h);
        break;
      case 16:
        common_hv_2ht_2vt_16w_msa(src, (int32_t)src_stride, dst,
                                  (int32_t)dst_stride, &filt_hor[3],
                                  &filt_ver[3], (int32_t)h);
        break;
      case 32:
        common_hv_2ht_2vt_32w_msa(src, (int32_t)src_stride, dst,
                                  (int32_t)dst_stride, &filt_hor[3],
                                  &filt_ver[3], (int32_t)h);
        break;
      case 64:
        common_hv_2ht_2vt_64w_msa(src, (int32_t)src_stride, dst,
                                  (int32_t)dst_stride, &filt_hor[3],
                                  &filt_ver[3], (int32_t)h);
        break;
      default:
        vpx_convolve8_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                        x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  } else if (vpx_get_filter_taps(filter_x) == 2 ||
             vpx_get_filter_taps(filter_y) == 2) {
    vpx_convolve8_c(src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4,
                    y0_q4, y_step_q4, w, h);
  } else {
    switch (w) {
      case 4:
        common_hv_8ht_8vt_4w_msa(src, (int32_t)src_stride, dst,
                                 (int32_t)dst_stride, filt_hor, filt_ver,
                                 (int32_t)h);
        break;
      case 8:
        common_hv_8ht_8vt_8w_msa(src, (int32_t)src_stride, dst,
                                 (int32_t)dst_stride, filt_hor, filt_ver,
                                 (int32_t)h);
        break;
      case 16:
        common_hv_8ht_8vt_16w_msa(src, (int32_t)src_stride, dst,
                                  (int32_t)dst_stride, filt_hor, filt_ver,
                                  (int32_t)h);
        break;
      case 32:
        common_hv_8ht_8vt_32w_msa(src, (int32_t)src_stride, dst,
                                  (int32_t)dst_stride, filt_hor, filt_ver,
                                  (int32_t)h);
        break;
      case 64:
        common_hv_8ht_8vt_64w_msa(src, (int32_t)src_stride, dst,
                                  (int32_t)dst_stride, filt_hor, filt_ver,
                                  (int32_t)h);
        break;
      default:
        vpx_convolve8_c(src, src_stride, dst, dst_stride, filter, x0_q4,
                        x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  }
}

static void filter_horiz_w4_msa(const uint8_t *src_x, ptrdiff_t src_pitch,
                                uint8_t *dst, const int16_t *x_filter) {
  uint64_t srcd0, srcd1, srcd2, srcd3;
  uint32_t res;
  v16u8 src0 = { 0 }, src1 = { 0 }, dst0;
  v16i8 out0, out1;
  v16i8 shf1 = { 0, 8, 16, 24, 4, 12, 20, 28, 1, 9, 17, 25, 5, 13, 21, 29 };
  v16i8 shf2 = shf1 + 2;
  v16i8 filt_shf0 = { 0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 8, 9, 8, 9, 8, 9 };
  v16i8 filt_shf1 = filt_shf0 + 2;
  v16i8 filt_shf2 = filt_shf0 + 4;
  v16i8 filt_shf3 = filt_shf0 + 6;
  v8i16 filt, src0_h, src1_h, src2_h, src3_h, filt0, filt1, filt2, filt3;

  LD4(src_x, src_pitch, srcd0, srcd1, srcd2, srcd3);
  INSERT_D2_UB(srcd0, srcd1, src0);
  INSERT_D2_UB(srcd2, srcd3, src1);
  VSHF_B2_SB(src0, src1, src0, src1, shf1, shf2, out0, out1);
  XORI_B2_128_SB(out0, out1);
  UNPCK_SB_SH(out0, src0_h, src1_h);
  UNPCK_SB_SH(out1, src2_h, src3_h);

  filt = LD_SH(x_filter);
  VSHF_B2_SH(filt, filt, filt, filt, filt_shf0, filt_shf1, filt0, filt1);
  VSHF_B2_SH(filt, filt, filt, filt, filt_shf2, filt_shf3, filt2, filt3);

  src0_h *= filt0;
  src0_h += src1_h * filt1;
  src0_h += src2_h * filt2;
  src0_h += src3_h * filt3;

  src1_h = (v8i16)__msa_sldi_b((v16i8)src0_h, (v16i8)src0_h, 8);

  src0_h = __msa_adds_s_h(src0_h, src1_h);
  src0_h = __msa_srari_h(src0_h, FILTER_BITS);
  src0_h = __msa_sat_s_h(src0_h, 7);
  dst0 = PCKEV_XORI128_UB(src0_h, src0_h);
  res = __msa_copy_u_w((v4i32)dst0, 0);
  SW(res, dst);
}

static void filter_horiz_w8_msa(const uint8_t *src_x, ptrdiff_t src_pitch,
                                uint8_t *dst, const int16_t *x_filter) {
  uint64_t srcd0, srcd1, srcd2, srcd3;
  v16u8 src0 = { 0 }, src1 = { 0 }, src2 = { 0 }, src3 = { 0 };
  v16u8 tmp0, tmp1, tmp2, tmp3, dst0;
  v16i8 out0, out1, out2, out3;
  v16i8 shf1 = { 0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27 };
  v16i8 shf2 = shf1 + 4;
  v8i16 filt, src0_h, src1_h, src2_h, src3_h, src4_h, src5_h, src6_h, src7_h;
  v8i16 filt0, filt1, filt2, filt3, filt4, filt5, filt6, filt7;

  LD4(src_x, src_pitch, srcd0, srcd1, srcd2, srcd3);
  INSERT_D2_UB(srcd0, srcd1, src0);
  INSERT_D2_UB(srcd2, srcd3, src1);
  LD4(src_x + 4 * src_pitch, src_pitch, srcd0, srcd1, srcd2, srcd3);
  INSERT_D2_UB(srcd0, srcd1, src2);
  INSERT_D2_UB(srcd2, srcd3, src3);

  filt = LD_SH(x_filter);
  SPLATI_H4_SH(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);
  SPLATI_H4_SH(filt, 4, 5, 6, 7, filt4, filt5, filt6, filt7);

  // transpose
  VSHF_B2_UB(src0, src1, src0, src1, shf1, shf2, tmp0, tmp1);
  VSHF_B2_UB(src2, src3, src2, src3, shf1, shf2, tmp2, tmp3);
  ILVRL_W2_SB(tmp2, tmp0, out0, out1);
  ILVRL_W2_SB(tmp3, tmp1, out2, out3);

  XORI_B4_128_SB(out0, out1, out2, out3);
  UNPCK_SB_SH(out0, src0_h, src1_h);
  UNPCK_SB_SH(out1, src2_h, src3_h);
  UNPCK_SB_SH(out2, src4_h, src5_h);
  UNPCK_SB_SH(out3, src6_h, src7_h);

  src0_h *= filt0;
  src4_h *= filt4;
  src0_h += src1_h * filt1;
  src4_h += src5_h * filt5;
  src0_h += src2_h * filt2;
  src4_h += src6_h * filt6;
  src0_h += src3_h * filt3;
  src4_h += src7_h * filt7;

  src0_h = __msa_adds_s_h(src0_h, src4_h);
  src0_h = __msa_srari_h(src0_h, FILTER_BITS);
  src0_h = __msa_sat_s_h(src0_h, 7);
  dst0 = PCKEV_XORI128_UB(src0_h, src0_h);
  ST8x1_UB(dst0, dst);
}

static void filter_horiz_w16_msa(const uint8_t *src_x, ptrdiff_t src_pitch,
                                 uint8_t *dst, const int16_t *x_filter) {
  uint64_t srcd0, srcd1, srcd2, srcd3;
  v16u8 src0 = { 0 }, src1 = { 0 }, src2 = { 0 }, src3 = { 0 };
  v16u8 src4 = { 0 }, src5 = { 0 }, src6 = { 0 }, src7 = { 0 };
  v16u8 tmp0, tmp1, tmp2, tmp3, dst0;
  v16i8 out0, out1, out2, out3, out4, out5, out6, out7;
  v16i8 shf1 = { 0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27 };
  v16i8 shf2 = shf1 + 4;
  v8i16 filt, src0_h, src1_h, src2_h, src3_h, src4_h, src5_h, src6_h, src7_h;
  v8i16 filt0, filt1, filt2, filt3, filt4, filt5, filt6, filt7;
  v8i16 dst0_h, dst1_h, dst2_h, dst3_h;

  LD4(src_x, src_pitch, srcd0, srcd1, srcd2, srcd3);
  INSERT_D2_UB(srcd0, srcd1, src0);
  INSERT_D2_UB(srcd2, srcd3, src1);
  LD4(src_x + 4 * src_pitch, src_pitch, srcd0, srcd1, srcd2, srcd3);
  INSERT_D2_UB(srcd0, srcd1, src2);
  INSERT_D2_UB(srcd2, srcd3, src3);
  LD4(src_x + 8 * src_pitch, src_pitch, srcd0, srcd1, srcd2, srcd3);
  INSERT_D2_UB(srcd0, srcd1, src4);
  INSERT_D2_UB(srcd2, srcd3, src5);
  LD4(src_x + 12 * src_pitch, src_pitch, srcd0, srcd1, srcd2, srcd3);
  INSERT_D2_UB(srcd0, srcd1, src6);
  INSERT_D2_UB(srcd2, srcd3, src7);

  filt = LD_SH(x_filter);
  SPLATI_H4_SH(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);
  SPLATI_H4_SH(filt, 4, 5, 6, 7, filt4, filt5, filt6, filt7);

  // transpose
  VSHF_B2_UB(src0, src1, src0, src1, shf1, shf2, tmp0, tmp1);
  VSHF_B2_UB(src2, src3, src2, src3, shf1, shf2, tmp2, tmp3);
  ILVRL_W2_SB(tmp2, tmp0, out0, out1);
  ILVRL_W2_SB(tmp3, tmp1, out2, out3);
  XORI_B4_128_SB(out0, out1, out2, out3);

  UNPCK_SB_SH(out0, src0_h, src1_h);
  UNPCK_SB_SH(out1, src2_h, src3_h);
  UNPCK_SB_SH(out2, src4_h, src5_h);
  UNPCK_SB_SH(out3, src6_h, src7_h);

  VSHF_B2_UB(src4, src5, src4, src5, shf1, shf2, tmp0, tmp1);
  VSHF_B2_UB(src6, src7, src6, src7, shf1, shf2, tmp2, tmp3);
  ILVRL_W2_SB(tmp2, tmp0, out4, out5);
  ILVRL_W2_SB(tmp3, tmp1, out6, out7);
  XORI_B4_128_SB(out4, out5, out6, out7);

  dst0_h = src0_h * filt0;
  dst1_h = src4_h * filt4;
  dst0_h += src1_h * filt1;
  dst1_h += src5_h * filt5;
  dst0_h += src2_h * filt2;
  dst1_h += src6_h * filt6;
  dst0_h += src3_h * filt3;
  dst1_h += src7_h * filt7;

  UNPCK_SB_SH(out4, src0_h, src1_h);
  UNPCK_SB_SH(out5, src2_h, src3_h);
  UNPCK_SB_SH(out6, src4_h, src5_h);
  UNPCK_SB_SH(out7, src6_h, src7_h);

  dst2_h = src0_h * filt0;
  dst3_h = src4_h * filt4;
  dst2_h += src1_h * filt1;
  dst3_h += src5_h * filt5;
  dst2_h += src2_h * filt2;
  dst3_h += src6_h * filt6;
  dst2_h += src3_h * filt3;
  dst3_h += src7_h * filt7;

  ADDS_SH2_SH(dst0_h, dst1_h, dst2_h, dst3_h, dst0_h, dst2_h);
  SRARI_H2_SH(dst0_h, dst2_h, FILTER_BITS);
  SAT_SH2_SH(dst0_h, dst2_h, 7);
  dst0 = PCKEV_XORI128_UB(dst0_h, dst2_h);
  ST_UB(dst0, dst);
}

static void transpose4x4_to_dst(const uint8_t *src, uint8_t *dst,
                                ptrdiff_t dst_stride) {
  v16u8 in0;
  v16i8 out0 = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

  in0 = LD_UB(src);
  out0 = __msa_vshf_b(out0, (v16i8)in0, (v16i8)in0);
  ST4x4_UB(out0, out0, 0, 1, 2, 3, dst, dst_stride);
}

static void transpose8x8_to_dst(const uint8_t *src, uint8_t *dst,
                                ptrdiff_t dst_stride) {
  v16u8 in0, in1, in2, in3, out0, out1, out2, out3, tmp0, tmp1, tmp2, tmp3;
  v16i8 shf1 = { 0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27 };
  v16i8 shf2 = shf1 + 4;

  LD_UB4(src, 16, in0, in1, in2, in3);
  VSHF_B2_UB(in0, in1, in0, in1, shf1, shf2, tmp0, tmp1);
  VSHF_B2_UB(in2, in3, in2, in3, shf1, shf2, tmp2, tmp3);
  ILVRL_W2_UB(tmp2, tmp0, out0, out1);
  ILVRL_W2_UB(tmp3, tmp1, out2, out3);
  ST8x4_UB(out0, out1, dst, dst_stride);
  ST8x4_UB(out2, out3, dst + 4 * dst_stride, dst_stride);
}

static void transpose16x16_to_dst(const uint8_t *src, uint8_t *dst,
                                  ptrdiff_t dst_stride) {
  v16u8 in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12;
  v16u8 in13, in14, in15, out0, out1, out2, out3, out4, out5, out6, out7, out8;
  v16u8 out9, out10, out11, out12, out13, out14, out15;

  LD_UB8(src, 16, in0, in1, in2, in3, in4, in5, in6, in7);
  LD_UB8(src + 16 * 8, 16, in8, in9, in10, in11, in12, in13, in14, in15);

  TRANSPOSE16x8_UB_UB(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                      in11, in12, in13, in14, in15, out0, out1, out2, out3,
                      out4, out5, out6, out7);
  ST_UB8(out0, out1, out2, out3, out4, out5, out6, out7, dst, dst_stride);
  dst += 8 * dst_stride;

  SLDI_B4_0_UB(in0, in1, in2, in3, in0, in1, in2, in3, 8);
  SLDI_B4_0_UB(in4, in5, in6, in7, in4, in5, in6, in7, 8);
  SLDI_B4_0_UB(in8, in9, in10, in11, in8, in9, in10, in11, 8);
  SLDI_B4_0_UB(in12, in13, in14, in15, in12, in13, in14, in15, 8);

  TRANSPOSE16x8_UB_UB(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
                      in11, in12, in13, in14, in15, out8, out9, out10, out11,
                      out12, out13, out14, out15);
  ST_UB8(out8, out9, out10, out11, out12, out13, out14, out15, dst, dst_stride);
}

static void scaledconvolve_horiz_w4(const uint8_t *src, ptrdiff_t src_stride,
                                    uint8_t *dst, ptrdiff_t dst_stride,
                                    const InterpKernel *x_filters, int x0_q4,
                                    int x_step_q4, int h) {
  DECLARE_ALIGNED(16, uint8_t, temp[4 * 4]);
  int y, z, i;
  src -= SUBPEL_TAPS / 2 - 1;

  for (y = 0; y < h; y += 4) {
    int x_q4 = x0_q4;
    for (z = 0; z < 4; ++z) {
      const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];

      if (x_q4 & SUBPEL_MASK) {
        filter_horiz_w4_msa(src_x, src_stride, temp + (z * 4), x_filter);
      } else {
        for (i = 0; i < 4; ++i) {
          temp[z * 4 + i] = src_x[i * src_stride + 3];
        }
      }

      x_q4 += x_step_q4;
    }

    transpose4x4_to_dst(temp, dst, dst_stride);

    src += src_stride * 4;
    dst += dst_stride * 4;
  }
}

static void scaledconvolve_horiz_w8(const uint8_t *src, ptrdiff_t src_stride,
                                    uint8_t *dst, ptrdiff_t dst_stride,
                                    const InterpKernel *x_filters, int x0_q4,
                                    int x_step_q4, int h) {
  DECLARE_ALIGNED(16, uint8_t, temp[8 * 8]);
  int y, z, i;
  src -= SUBPEL_TAPS / 2 - 1;

  // This function processes 8x8 areas. The intermediate height is not always
  // a multiple of 8, so force it to be a multiple of 8 here.
  y = h + (8 - (h & 0x7));

  do {
    int x_q4 = x0_q4;
    for (z = 0; z < 8; ++z) {
      const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
      const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];

      if (x_q4 & SUBPEL_MASK) {
        filter_horiz_w8_msa(src_x, src_stride, temp + (z * 8), x_filter);
      } else {
        for (i = 0; i < 8; ++i) {
          temp[z * 8 + i] = src_x[3 + i * src_stride];
        }
      }

      x_q4 += x_step_q4;
    }

    transpose8x8_to_dst(temp, dst, dst_stride);

    src += src_stride * 8;
    dst += dst_stride * 8;
  } while (y -= 8);
}

static void scaledconvolve_horiz_mul16(const uint8_t *src, ptrdiff_t src_stride,
                                       uint8_t *dst, ptrdiff_t dst_stride,
                                       const InterpKernel *x_filters, int x0_q4,
                                       int x_step_q4, int w, int h) {
  DECLARE_ALIGNED(16, uint8_t, temp[16 * 16]);
  int x, y, z, i;

  src -= SUBPEL_TAPS / 2 - 1;

  // This function processes 16x16 areas.  The intermediate height is not always
  // a multiple of 16, so force it to be a multiple of 8 here.
  y = h + (16 - (h & 0xF));

  do {
    int x_q4 = x0_q4;
    for (x = 0; x < w; x += 16) {
      for (z = 0; z < 16; ++z) {
        const uint8_t *const src_x = &src[x_q4 >> SUBPEL_BITS];
        const int16_t *const x_filter = x_filters[x_q4 & SUBPEL_MASK];

        if (x_q4 & SUBPEL_MASK) {
          filter_horiz_w16_msa(src_x, src_stride, temp + (z * 16), x_filter);
        } else {
          for (i = 0; i < 16; ++i) {
            temp[z * 16 + i] = src_x[3 + i * src_stride];
          }
        }

        x_q4 += x_step_q4;
      }

      transpose16x16_to_dst(temp, dst + x, dst_stride);
    }

    src += src_stride * 16;
    dst += dst_stride * 16;
  } while (y -= 16);
}

static void filter_vert_w4_msa(const uint8_t *src_y, ptrdiff_t src_pitch,
                               uint8_t *dst, const int16_t *y_filter) {
  uint32_t srcw0, srcw1, srcw2, srcw3, srcw4, srcw5, srcw6, srcw7;
  uint32_t res;
  v16u8 src0 = { 0 }, src1 = { 0 }, dst0;
  v16i8 out0, out1;
  v16i8 shf1 = { 0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23 };
  v16i8 shf2 = shf1 + 8;
  v16i8 filt_shf0 = { 0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 8, 9, 8, 9, 8, 9 };
  v16i8 filt_shf1 = filt_shf0 + 2;
  v16i8 filt_shf2 = filt_shf0 + 4;
  v16i8 filt_shf3 = filt_shf0 + 6;
  v8i16 filt, src0_h, src1_h, src2_h, src3_h;
  v8i16 filt0, filt1, filt2, filt3;

  LW4(src_y, src_pitch, srcw0, srcw1, srcw2, srcw3);
  LW4(src_y + 4 * src_pitch, src_pitch, srcw4, srcw5, srcw6, srcw7);
  INSERT_W4_UB(srcw0, srcw1, srcw2, srcw3, src0);
  INSERT_W4_UB(srcw4, srcw5, srcw6, srcw7, src1);
  VSHF_B2_SB(src0, src1, src0, src1, shf1, shf2, out0, out1);
  XORI_B2_128_SB(out0, out1);
  UNPCK_SB_SH(out0, src0_h, src1_h);
  UNPCK_SB_SH(out1, src2_h, src3_h);

  filt = LD_SH(y_filter);
  VSHF_B2_SH(filt, filt, filt, filt, filt_shf0, filt_shf1, filt0, filt1);
  VSHF_B2_SH(filt, filt, filt, filt, filt_shf2, filt_shf3, filt2, filt3);

  src0_h *= filt0;
  src0_h += src1_h * filt1;
  src0_h += src2_h * filt2;
  src0_h += src3_h * filt3;

  src1_h = (v8i16)__msa_sldi_b((v16i8)src0_h, (v16i8)src0_h, 8);

  src0_h = __msa_adds_s_h(src0_h, src1_h);
  src0_h = __msa_srari_h(src0_h, FILTER_BITS);
  src0_h = __msa_sat_s_h(src0_h, 7);
  dst0 = PCKEV_XORI128_UB(src0_h, src0_h);
  res = __msa_copy_u_w((v4i32)dst0, 0);
  SW(res, dst);
}

static void filter_vert_w8_msa(const uint8_t *src_y, ptrdiff_t src_pitch,
                               uint8_t *dst, const int16_t *y_filter) {
  uint64_t srcd0, srcd1, srcd2, srcd3;
  v16u8 dst0;
  v16i8 src0 = { 0 }, src1 = { 0 }, src2 = { 0 }, src3 = { 0 };
  v8i16 filt, src0_h, src1_h, src2_h, src3_h, src4_h, src5_h, src6_h, src7_h;
  v8i16 filt0, filt1, filt2, filt3, filt4, filt5, filt6, filt7;

  LD4(src_y, src_pitch, srcd0, srcd1, srcd2, srcd3);
  INSERT_D2_SB(srcd0, srcd1, src0);
  INSERT_D2_SB(srcd2, srcd3, src1);
  LD4(src_y + 4 * src_pitch, src_pitch, srcd0, srcd1, srcd2, srcd3);
  INSERT_D2_SB(srcd0, srcd1, src2);
  INSERT_D2_SB(srcd2, srcd3, src3);

  filt = LD_SH(y_filter);
  SPLATI_H4_SH(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);
  SPLATI_H4_SH(filt, 4, 5, 6, 7, filt4, filt5, filt6, filt7);

  XORI_B4_128_SB(src0, src1, src2, src3);
  UNPCK_SB_SH(src0, src0_h, src1_h);
  UNPCK_SB_SH(src1, src2_h, src3_h);
  UNPCK_SB_SH(src2, src4_h, src5_h);
  UNPCK_SB_SH(src3, src6_h, src7_h);

  src0_h *= filt0;
  src4_h *= filt4;
  src0_h += src1_h * filt1;
  src4_h += src5_h * filt5;
  src0_h += src2_h * filt2;
  src4_h += src6_h * filt6;
  src0_h += src3_h * filt3;
  src4_h += src7_h * filt7;

  src0_h = __msa_adds_s_h(src0_h, src4_h);
  src0_h = __msa_srari_h(src0_h, FILTER_BITS);
  src0_h = __msa_sat_s_h(src0_h, 7);
  dst0 = PCKEV_XORI128_UB(src0_h, src0_h);
  ST8x1_UB(dst0, dst);
}

static void filter_vert_mul_w16_msa(const uint8_t *src_y, ptrdiff_t src_pitch,
                                    uint8_t *dst, const int16_t *y_filter,
                                    int w) {
  int x;
  v16u8 dst0;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7;
  v8i16 filt, src0_h, src1_h, src2_h, src3_h, src4_h, src5_h, src6_h, src7_h;
  v8i16 src8_h, src9_h, src10_h, src11_h, src12_h, src13_h, src14_h, src15_h;
  v8i16 filt0, filt1, filt2, filt3, filt4, filt5, filt6, filt7;

  filt = LD_SH(y_filter);
  SPLATI_H4_SH(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);
  SPLATI_H4_SH(filt, 4, 5, 6, 7, filt4, filt5, filt6, filt7);

  for (x = 0; x < w; x += 16) {
    LD_SB8(src_y, src_pitch, src0, src1, src2, src3, src4, src5, src6, src7);
    src_y += 16;

    XORI_B4_128_SB(src0, src1, src2, src3);
    XORI_B4_128_SB(src4, src5, src6, src7);
    UNPCK_SB_SH(src0, src0_h, src1_h);
    UNPCK_SB_SH(src1, src2_h, src3_h);
    UNPCK_SB_SH(src2, src4_h, src5_h);
    UNPCK_SB_SH(src3, src6_h, src7_h);
    UNPCK_SB_SH(src4, src8_h, src9_h);
    UNPCK_SB_SH(src5, src10_h, src11_h);
    UNPCK_SB_SH(src6, src12_h, src13_h);
    UNPCK_SB_SH(src7, src14_h, src15_h);

    src0_h *= filt0;
    src1_h *= filt0;
    src8_h *= filt4;
    src9_h *= filt4;
    src0_h += src2_h * filt1;
    src1_h += src3_h * filt1;
    src8_h += src10_h * filt5;
    src9_h += src11_h * filt5;
    src0_h += src4_h * filt2;
    src1_h += src5_h * filt2;
    src8_h += src12_h * filt6;
    src9_h += src13_h * filt6;
    src0_h += src6_h * filt3;
    src1_h += src7_h * filt3;
    src8_h += src14_h * filt7;
    src9_h += src15_h * filt7;

    ADDS_SH2_SH(src0_h, src8_h, src1_h, src9_h, src0_h, src1_h);
    SRARI_H2_SH(src0_h, src1_h, FILTER_BITS);
    SAT_SH2_SH(src0_h, src1_h, 7);
    dst0 = PCKEV_XORI128_UB(src0_h, src1_h);
    ST_UB(dst0, dst);
    dst += 16;
  }
}

static void scaledconvolve_vert_w4(const uint8_t *src, ptrdiff_t src_stride,
                                   uint8_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *y_filters, int y0_q4,
                                   int y_step_q4, int h) {
  int y;
  int y_q4 = y0_q4;

  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (y = 0; y < h; ++y) {
    const uint8_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
    const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];

    if (y_q4 & SUBPEL_MASK) {
      filter_vert_w4_msa(src_y, src_stride, &dst[y * dst_stride], y_filter);
    } else {
      uint32_t srcd = LW(src_y + 3 * src_stride);
      SW(srcd, dst + y * dst_stride);
    }

    y_q4 += y_step_q4;
  }
}

static void scaledconvolve_vert_w8(const uint8_t *src, ptrdiff_t src_stride,
                                   uint8_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *y_filters, int y0_q4,
                                   int y_step_q4, int h) {
  int y;
  int y_q4 = y0_q4;

  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (y = 0; y < h; ++y) {
    const uint8_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
    const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];

    if (y_q4 & SUBPEL_MASK) {
      filter_vert_w8_msa(src_y, src_stride, &dst[y * dst_stride], y_filter);
    } else {
      uint64_t srcd = LD(src_y + 3 * src_stride);
      SD(srcd, dst + y * dst_stride);
    }

    y_q4 += y_step_q4;
  }
}

static void scaledconvolve_vert_mul16(const uint8_t *src, ptrdiff_t src_stride,
                                      uint8_t *dst, ptrdiff_t dst_stride,
                                      const InterpKernel *y_filters, int y0_q4,
                                      int y_step_q4, int w, int h) {
  int x, y;
  int y_q4 = y0_q4;
  src -= src_stride * (SUBPEL_TAPS / 2 - 1);

  for (y = 0; y < h; ++y) {
    const uint8_t *src_y = &src[(y_q4 >> SUBPEL_BITS) * src_stride];
    const int16_t *const y_filter = y_filters[y_q4 & SUBPEL_MASK];

    if (y_q4 & SUBPEL_MASK) {
      filter_vert_mul_w16_msa(src_y, src_stride, &dst[y * dst_stride], y_filter,
                              w);
    } else {
      for (x = 0; x < w; ++x) {
        dst[x + y * dst_stride] = src_y[x + 3 * src_stride];
      }
    }

    y_q4 += y_step_q4;
  }
}

void vpx_scaled_2d_msa(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                       ptrdiff_t dst_stride, const InterpKernel *filter,
                       int x0_q4, int x_step_q4, int y0_q4, int y_step_q4,
                       int w, int h) {
  // Note: Fixed size intermediate buffer, temp, places limits on parameters.
  // 2d filtering proceeds in 2 steps:
  //   (1) Interpolate horizontally into an intermediate buffer, temp.
  //   (2) Interpolate temp vertically to derive the sub-pixel result.
  // Deriving the maximum number of rows in the temp buffer (135):
  // --Smallest scaling factor is x1/2 ==> y_step_q4 = 32 (Normative).
  // --Largest block size is 64x64 pixels.
  // --64 rows in the downscaled frame span a distance of (64 - 1) * 32 in the
  //   original frame (in 1/16th pixel units).
  // --Must round-up because block may be located at sub-pixel position.
  // --Require an additional SUBPEL_TAPS rows for the 8-tap filter tails.
  // --((64 - 1) * 32 + 15) >> 4 + 8 = 135.
  // --Require an additional 8 rows for the horiz_w8 transpose tail.
  DECLARE_ALIGNED(16, uint8_t, temp[(135 + 8) * 64]);
  const int intermediate_height =
      (((h - 1) * y_step_q4 + y0_q4) >> SUBPEL_BITS) + SUBPEL_TAPS;

  assert(w <= 64);
  assert(h <= 64);
  assert(y_step_q4 <= 32 || (y_step_q4 <= 64 && h <= 32));
  assert(x_step_q4 <= 64);

  if ((0 == x0_q4) && (16 == x_step_q4) && (0 == y0_q4) && (16 == y_step_q4)) {
    vpx_convolve_copy_msa(src, src_stride, dst, dst_stride, filter, x0_q4,
                          x_step_q4, y0_q4, y_step_q4, w, h);
  } else {
    if (w >= 16) {
      scaledconvolve_horiz_mul16(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                                 src_stride, temp, 64, filter, x0_q4, x_step_q4,
                                 w, intermediate_height);
    } else if (w == 8) {
      scaledconvolve_horiz_w8(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                              src_stride, temp, 64, filter, x0_q4, x_step_q4,
                              intermediate_height);
    } else {
      scaledconvolve_horiz_w4(src - src_stride * (SUBPEL_TAPS / 2 - 1),
                              src_stride, temp, 64, filter, x0_q4, x_step_q4,
                              intermediate_height);
    }

    if (w >= 16) {
      scaledconvolve_vert_mul16(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst,
                                dst_stride, filter, y0_q4, y_step_q4, w, h);
    } else if (w == 8) {
      scaledconvolve_vert_w8(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst,
                             dst_stride, filter, y0_q4, y_step_q4, h);
    } else {
      scaledconvolve_vert_w4(temp + 64 * (SUBPEL_TAPS / 2 - 1), 64, dst,
                             dst_stride, filter, y0_q4, y_step_q4, h);
    }
  }
}
