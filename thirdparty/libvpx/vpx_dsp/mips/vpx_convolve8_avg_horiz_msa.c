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

static void common_hz_8t_and_aver_dst_4x4_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  uint32_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, filt0, filt1, filt2, filt3;
  v16u8 dst0 = { 0 }, res;
  v16u8 mask0, mask1, mask2, mask3;
  v8i16 filt, res0, res1;

  mask0 = LD_UB(&mc_filt_mask_arr[16]);
  src -= 3;

  /* rearranging filter */
  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;
  mask3 = mask0 + 6;

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  XORI_B4_128_SB(src0, src1, src2, src3);
  HORIZ_8TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, mask3,
                             filt0, filt1, filt2, filt3, res0, res1);
  LW4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_W4_UB(tp0, tp1, tp2, tp3, dst0);
  SRARI_H2_SH(res0, res1, FILTER_BITS);
  SAT_SH2_SH(res0, res1, 7);
  res = PCKEV_XORI128_UB(res0, res1);
  res = (v16u8)__msa_aver_u_b(res, dst0);
  ST4x4_UB(res, res, 0, 1, 2, 3, dst, dst_stride);
}

static void common_hz_8t_and_aver_dst_4x8_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  uint32_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, filt0, filt1, filt2, filt3;
  v16u8 mask0, mask1, mask2, mask3, res0, res1, res2, res3;
  v16u8 dst0 = { 0 }, dst1 = { 0 };
  v8i16 filt, vec0, vec1, vec2, vec3;

  mask0 = LD_UB(&mc_filt_mask_arr[16]);
  src -= 3;

  /* rearranging filter */
  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;
  mask3 = mask0 + 6;

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  XORI_B4_128_SB(src0, src1, src2, src3);
  src += (4 * src_stride);
  LW4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_W4_UB(tp0, tp1, tp2, tp3, dst0);
  LW4(dst + 4 * dst_stride, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_W4_UB(tp0, tp1, tp2, tp3, dst1);
  HORIZ_8TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, mask3,
                             filt0, filt1, filt2, filt3, vec0, vec1);
  LD_SB4(src, src_stride, src0, src1, src2, src3);
  XORI_B4_128_SB(src0, src1, src2, src3);
  HORIZ_8TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, mask3,
                             filt0, filt1, filt2, filt3, vec2, vec3);
  SRARI_H4_SH(vec0, vec1, vec2, vec3, FILTER_BITS);
  SAT_SH4_SH(vec0, vec1, vec2, vec3, 7);
  PCKEV_B4_UB(vec0, vec0, vec1, vec1, vec2, vec2, vec3, vec3, res0, res1, res2,
              res3);
  ILVR_D2_UB(res1, res0, res3, res2, res0, res2);
  XORI_B2_128_UB(res0, res2);
  AVER_UB2_UB(res0, dst0, res2, dst1, res0, res2);
  ST4x8_UB(res0, res2, dst, dst_stride);
}

static void common_hz_8t_and_aver_dst_4w_msa(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  if (4 == height) {
    common_hz_8t_and_aver_dst_4x4_msa(src, src_stride, dst, dst_stride, filter);
  } else if (8 == height) {
    common_hz_8t_and_aver_dst_4x8_msa(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_hz_8t_and_aver_dst_8w_msa(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  int32_t loop_cnt;
  int64_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, filt0, filt1, filt2, filt3;
  v16u8 mask0, mask1, mask2, mask3, dst0 = { 0 }, dst1 = { 0 };
  v8i16 filt, out0, out1, out2, out3;

  mask0 = LD_UB(&mc_filt_mask_arr[0]);
  src -= 3;

  /* rearranging filter */
  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;
  mask3 = mask0 + 6;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    XORI_B4_128_SB(src0, src1, src2, src3);
    src += (4 * src_stride);
    HORIZ_8TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2,
                               mask3, filt0, filt1, filt2, filt3, out0, out1,
                               out2, out3);
    LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
    INSERT_D2_UB(tp0, tp1, dst0);
    INSERT_D2_UB(tp2, tp3, dst1);
    SRARI_H4_SH(out0, out1, out2, out3, FILTER_BITS);
    SAT_SH4_SH(out0, out1, out2, out3, 7);
    CONVERT_UB_AVG_ST8x4_UB(out0, out1, out2, out3, dst0, dst1, dst,
                            dst_stride);
    dst += (4 * dst_stride);
  }
}

static void common_hz_8t_and_aver_dst_16w_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  int32_t loop_cnt;
  v16i8 src0, src1, src2, src3, filt0, filt1, filt2, filt3;
  v16u8 mask0, mask1, mask2, mask3, dst0, dst1;
  v8i16 filt, out0, out1, out2, out3;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8i16 vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15;

  mask0 = LD_UB(&mc_filt_mask_arr[0]);
  src -= 3;

  /* rearranging filter */
  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;
  mask3 = mask0 + 6;

  for (loop_cnt = height >> 1; loop_cnt--;) {
    LD_SB2(src, src_stride, src0, src2);
    LD_SB2(src + 8, src_stride, src1, src3);
    src += (2 * src_stride);

    XORI_B4_128_SB(src0, src1, src2, src3);
    VSHF_B4_SH(src0, src0, mask0, mask1, mask2, mask3, vec0, vec4, vec8, vec12);
    VSHF_B4_SH(src1, src1, mask0, mask1, mask2, mask3, vec1, vec5, vec9, vec13);
    VSHF_B4_SH(src2, src2, mask0, mask1, mask2, mask3, vec2, vec6, vec10,
               vec14);
    VSHF_B4_SH(src3, src3, mask0, mask1, mask2, mask3, vec3, vec7, vec11,
               vec15);
    DOTP_SB4_SH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    DOTP_SB4_SH(vec8, vec9, vec10, vec11, filt2, filt2, filt2, filt2, vec8,
                vec9, vec10, vec11);
    DPADD_SB4_SH(vec4, vec5, vec6, vec7, filt1, filt1, filt1, filt1, vec0, vec1,
                 vec2, vec3);
    DPADD_SB4_SH(vec12, vec13, vec14, vec15, filt3, filt3, filt3, filt3, vec8,
                 vec9, vec10, vec11);
    ADDS_SH4_SH(vec0, vec8, vec1, vec9, vec2, vec10, vec3, vec11, out0, out1,
                out2, out3);
    LD_UB2(dst, dst_stride, dst0, dst1);
    SRARI_H4_SH(out0, out1, out2, out3, FILTER_BITS);
    SAT_SH4_SH(out0, out1, out2, out3, 7);
    PCKEV_XORI128_AVG_ST_UB(out1, out0, dst0, dst);
    dst += dst_stride;
    PCKEV_XORI128_AVG_ST_UB(out3, out2, dst1, dst);
    dst += dst_stride;
  }
}

static void common_hz_8t_and_aver_dst_32w_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, filt0, filt1, filt2, filt3;
  v16u8 dst1, dst2, mask0, mask1, mask2, mask3;
  v8i16 filt, out0, out1, out2, out3;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8i16 vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15;

  mask0 = LD_UB(&mc_filt_mask_arr[0]);
  src -= 3;

  /* rearranging filter */
  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;
  mask3 = mask0 + 6;

  for (loop_cnt = height; loop_cnt--;) {
    src0 = LD_SB(src);
    src2 = LD_SB(src + 16);
    src3 = LD_SB(src + 24);
    src1 = __msa_sldi_b(src2, src0, 8);
    src += src_stride;

    XORI_B4_128_SB(src0, src1, src2, src3);
    VSHF_B4_SH(src0, src0, mask0, mask1, mask2, mask3, vec0, vec4, vec8, vec12);
    VSHF_B4_SH(src1, src1, mask0, mask1, mask2, mask3, vec1, vec5, vec9, vec13);
    VSHF_B4_SH(src2, src2, mask0, mask1, mask2, mask3, vec2, vec6, vec10,
               vec14);
    VSHF_B4_SH(src3, src3, mask0, mask1, mask2, mask3, vec3, vec7, vec11,
               vec15);
    DOTP_SB4_SH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    DOTP_SB4_SH(vec8, vec9, vec10, vec11, filt2, filt2, filt2, filt2, vec8,
                vec9, vec10, vec11);
    DPADD_SB4_SH(vec4, vec5, vec6, vec7, filt1, filt1, filt1, filt1, vec0, vec1,
                 vec2, vec3);
    DPADD_SB4_SH(vec12, vec13, vec14, vec15, filt3, filt3, filt3, filt3, vec8,
                 vec9, vec10, vec11);
    ADDS_SH4_SH(vec0, vec8, vec1, vec9, vec2, vec10, vec3, vec11, out0, out1,
                out2, out3);
    SRARI_H4_SH(out0, out1, out2, out3, FILTER_BITS);
    SAT_SH4_SH(out0, out1, out2, out3, 7);
    LD_UB2(dst, 16, dst1, dst2);
    PCKEV_XORI128_AVG_ST_UB(out1, out0, dst1, dst);
    PCKEV_XORI128_AVG_ST_UB(out3, out2, dst2, dst + 16);
    dst += dst_stride;
  }
}

static void common_hz_8t_and_aver_dst_64w_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt, cnt;
  v16i8 src0, src1, src2, src3, filt0, filt1, filt2, filt3;
  v16u8 dst1, dst2, mask0, mask1, mask2, mask3;
  v8i16 filt, out0, out1, out2, out3;
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8i16 vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15;

  mask0 = LD_UB(&mc_filt_mask_arr[0]);
  src -= 3;

  /* rearranging filter */
  filt = LD_SH(filter);
  SPLATI_H4_SB(filt, 0, 1, 2, 3, filt0, filt1, filt2, filt3);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;
  mask3 = mask0 + 6;

  for (loop_cnt = height; loop_cnt--;) {
    for (cnt = 0; cnt < 2; ++cnt) {
      src0 = LD_SB(&src[cnt << 5]);
      src2 = LD_SB(&src[16 + (cnt << 5)]);
      src3 = LD_SB(&src[24 + (cnt << 5)]);
      src1 = __msa_sldi_b(src2, src0, 8);

      XORI_B4_128_SB(src0, src1, src2, src3);
      VSHF_B4_SH(src0, src0, mask0, mask1, mask2, mask3, vec0, vec4, vec8,
                 vec12);
      VSHF_B4_SH(src1, src1, mask0, mask1, mask2, mask3, vec1, vec5, vec9,
                 vec13);
      VSHF_B4_SH(src2, src2, mask0, mask1, mask2, mask3, vec2, vec6, vec10,
                 vec14);
      VSHF_B4_SH(src3, src3, mask0, mask1, mask2, mask3, vec3, vec7, vec11,
                 vec15);
      DOTP_SB4_SH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0,
                  vec1, vec2, vec3);
      DOTP_SB4_SH(vec8, vec9, vec10, vec11, filt2, filt2, filt2, filt2, vec8,
                  vec9, vec10, vec11);
      DPADD_SB4_SH(vec4, vec5, vec6, vec7, filt1, filt1, filt1, filt1, vec0,
                   vec1, vec2, vec3);
      DPADD_SB4_SH(vec12, vec13, vec14, vec15, filt3, filt3, filt3, filt3, vec8,
                   vec9, vec10, vec11);
      ADDS_SH4_SH(vec0, vec8, vec1, vec9, vec2, vec10, vec3, vec11, out0, out1,
                  out2, out3);
      SRARI_H4_SH(out0, out1, out2, out3, FILTER_BITS);
      SAT_SH4_SH(out0, out1, out2, out3, 7);
      LD_UB2(&dst[cnt << 5], 16, dst1, dst2);
      PCKEV_XORI128_AVG_ST_UB(out1, out0, dst1, &dst[cnt << 5]);
      PCKEV_XORI128_AVG_ST_UB(out3, out2, dst2, &dst[16 + (cnt << 5)]);
    }

    src += src_stride;
    dst += dst_stride;
  }
}

static void common_hz_2t_and_aver_dst_4x4_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  uint32_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, mask;
  v16u8 filt0, dst0 = { 0 }, vec0, vec1, res;
  v8u16 vec2, vec3, filt;

  mask = LD_SB(&mc_filt_mask_arr[16]);

  /* rearranging filter */
  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  LW4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_W4_UB(tp0, tp1, tp2, tp3, dst0);
  VSHF_B2_UB(src0, src1, src2, src3, mask, mask, vec0, vec1);
  DOTP_UB2_UH(vec0, vec1, filt0, filt0, vec2, vec3);
  SRARI_H2_UH(vec2, vec3, FILTER_BITS);
  res = (v16u8)__msa_pckev_b((v16i8)vec3, (v16i8)vec2);
  res = (v16u8)__msa_aver_u_b(res, dst0);
  ST4x4_UB(res, res, 0, 1, 2, 3, dst, dst_stride);
}

static void common_hz_2t_and_aver_dst_4x8_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  uint32_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask;
  v16u8 filt0, vec0, vec1, vec2, vec3, res0, res1, res2, res3;
  v16u8 dst0 = { 0 }, dst1 = { 0 };
  v8u16 vec4, vec5, vec6, vec7, filt;

  mask = LD_SB(&mc_filt_mask_arr[16]);

  /* rearranging filter */
  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  LW4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_W4_UB(tp0, tp1, tp2, tp3, dst0);
  LW4(dst + 4 * dst_stride, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_W4_UB(tp0, tp1, tp2, tp3, dst1);
  VSHF_B2_UB(src0, src1, src2, src3, mask, mask, vec0, vec1);
  VSHF_B2_UB(src4, src5, src6, src7, mask, mask, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec4, vec5,
              vec6, vec7);
  SRARI_H4_UH(vec4, vec5, vec6, vec7, FILTER_BITS);
  PCKEV_B4_UB(vec4, vec4, vec5, vec5, vec6, vec6, vec7, vec7, res0, res1, res2,
              res3);
  ILVR_D2_UB(res1, res0, res3, res2, res0, res2);
  AVER_UB2_UB(res0, dst0, res2, dst1, res0, res2);
  ST4x8_UB(res0, res2, dst, dst_stride);
}

static void common_hz_2t_and_aver_dst_4w_msa(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  if (4 == height) {
    common_hz_2t_and_aver_dst_4x4_msa(src, src_stride, dst, dst_stride, filter);
  } else if (8 == height) {
    common_hz_2t_and_aver_dst_4x8_msa(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_hz_2t_and_aver_dst_8x4_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  int64_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, mask;
  v16u8 filt0, dst0 = { 0 }, dst1 = { 0 };
  v8u16 vec0, vec1, vec2, vec3, filt;

  mask = LD_SB(&mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
  VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
              vec2, vec3);
  SRARI_H4_UH(vec0, vec1, vec2, vec3, FILTER_BITS);
  LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_D2_UB(tp0, tp1, dst0);
  INSERT_D2_UB(tp2, tp3, dst1);
  PCKEV_AVG_ST8x4_UB(vec0, vec1, vec2, vec3, dst0, dst1, dst, dst_stride);
}

static void common_hz_2t_and_aver_dst_8x8mult_msa(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter, int32_t height) {
  int64_t tp0, tp1, tp2, tp3;
  v16i8 src0, src1, src2, src3, mask;
  v16u8 filt0, dst0 = { 0 }, dst1 = { 0 };
  v8u16 vec0, vec1, vec2, vec3, filt;

  mask = LD_SB(&mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  src += (4 * src_stride);
  VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
  VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
              vec2, vec3);
  SRARI_H4_UH(vec0, vec1, vec2, vec3, FILTER_BITS);
  LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_D2_UB(tp0, tp1, dst0);
  INSERT_D2_UB(tp2, tp3, dst1);
  LD_SB4(src, src_stride, src0, src1, src2, src3);
  src += (4 * src_stride);
  PCKEV_AVG_ST8x4_UB(vec0, vec1, vec2, vec3, dst0, dst1, dst, dst_stride);
  dst += (4 * dst_stride);

  VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
  VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
              vec2, vec3);
  SRARI_H4_UH(vec0, vec1, vec2, vec3, FILTER_BITS);
  LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
  INSERT_D2_UB(tp0, tp1, dst0);
  INSERT_D2_UB(tp2, tp3, dst1);
  PCKEV_AVG_ST8x4_UB(vec0, vec1, vec2, vec3, dst0, dst1, dst, dst_stride);
  dst += (4 * dst_stride);

  if (16 == height) {
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);

    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    SRARI_H4_UH(vec0, vec1, vec2, vec3, FILTER_BITS);
    LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
    INSERT_D2_UB(tp0, tp1, dst0);
    INSERT_D2_UB(tp2, tp3, dst1);
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    PCKEV_AVG_ST8x4_UB(vec0, vec1, vec2, vec3, dst0, dst1, dst, dst_stride);
    dst += (4 * dst_stride);

    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    SRARI_H4_UH(vec0, vec1, vec2, vec3, FILTER_BITS);
    LD4(dst, dst_stride, tp0, tp1, tp2, tp3);
    INSERT_D2_UB(tp0, tp1, dst0);
    INSERT_D2_UB(tp2, tp3, dst1);
    PCKEV_AVG_ST8x4_UB(vec0, vec1, vec2, vec3, dst0, dst1, dst, dst_stride);
  }
}

static void common_hz_2t_and_aver_dst_8w_msa(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  if (4 == height) {
    common_hz_2t_and_aver_dst_8x4_msa(src, src_stride, dst, dst_stride, filter);
  } else {
    common_hz_2t_and_aver_dst_8x8mult_msa(src, src_stride, dst, dst_stride,
                                          filter, height);
  }
}

static void common_hz_2t_and_aver_dst_16w_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask;
  v16u8 filt0, dst0, dst1, dst2, dst3;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 res0, res1, res2, res3, res4, res5, res6, res7, filt;

  mask = LD_SB(&mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB4(src, src_stride, src0, src2, src4, src6);
  LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
  src += (4 * src_stride);

  VSHF_B2_UB(src0, src0, src1, src1, mask, mask, vec0, vec1);
  VSHF_B2_UB(src2, src2, src3, src3, mask, mask, vec2, vec3);
  VSHF_B2_UB(src4, src4, src5, src5, mask, mask, vec4, vec5);
  VSHF_B2_UB(src6, src6, src7, src7, mask, mask, vec6, vec7);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, res0, res1,
              res2, res3);
  DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, res4, res5,
              res6, res7);
  SRARI_H4_UH(res0, res1, res2, res3, FILTER_BITS);
  SRARI_H4_UH(res4, res5, res6, res7, FILTER_BITS);
  LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);
  PCKEV_AVG_ST_UB(res1, res0, dst0, dst);
  dst += dst_stride;
  PCKEV_AVG_ST_UB(res3, res2, dst1, dst);
  dst += dst_stride;
  PCKEV_AVG_ST_UB(res5, res4, dst2, dst);
  dst += dst_stride;
  PCKEV_AVG_ST_UB(res7, res6, dst3, dst);
  dst += dst_stride;

  for (loop_cnt = (height >> 2) - 1; loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src2, src4, src6);
    LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);

    VSHF_B2_UB(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UB(src2, src2, src3, src3, mask, mask, vec2, vec3);
    VSHF_B2_UB(src4, src4, src5, src5, mask, mask, vec4, vec5);
    VSHF_B2_UB(src6, src6, src7, src7, mask, mask, vec6, vec7);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, res0, res1,
                res2, res3);
    DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, res4, res5,
                res6, res7);
    SRARI_H4_UH(res0, res1, res2, res3, FILTER_BITS);
    SRARI_H4_UH(res4, res5, res6, res7, FILTER_BITS);
    LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);
    PCKEV_AVG_ST_UB(res1, res0, dst0, dst);
    dst += dst_stride;
    PCKEV_AVG_ST_UB(res3, res2, dst1, dst);
    dst += dst_stride;
    PCKEV_AVG_ST_UB(res5, res4, dst2, dst);
    dst += dst_stride;
    PCKEV_AVG_ST_UB(res7, res6, dst3, dst);
    dst += dst_stride;
  }
}

static void common_hz_2t_and_aver_dst_32w_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask;
  v16u8 filt0, dst0, dst1, dst2, dst3;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 res0, res1, res2, res3, res4, res5, res6, res7, filt;

  mask = LD_SB(&mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  for (loop_cnt = (height >> 1); loop_cnt--;) {
    src0 = LD_SB(src);
    src2 = LD_SB(src + 16);
    src3 = LD_SB(src + 24);
    src1 = __msa_sldi_b(src2, src0, 8);
    src += src_stride;
    src4 = LD_SB(src);
    src6 = LD_SB(src + 16);
    src7 = LD_SB(src + 24);
    src5 = __msa_sldi_b(src6, src4, 8);
    src += src_stride;

    VSHF_B2_UB(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UB(src2, src2, src3, src3, mask, mask, vec2, vec3);
    VSHF_B2_UB(src4, src4, src5, src5, mask, mask, vec4, vec5);
    VSHF_B2_UB(src6, src6, src7, src7, mask, mask, vec6, vec7);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, res0, res1,
                res2, res3);
    DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, res4, res5,
                res6, res7);
    SRARI_H4_UH(res0, res1, res2, res3, FILTER_BITS);
    SRARI_H4_UH(res4, res5, res6, res7, FILTER_BITS);
    LD_UB2(dst, 16, dst0, dst1);
    PCKEV_AVG_ST_UB(res1, res0, dst0, dst);
    PCKEV_AVG_ST_UB(res3, res2, dst1, (dst + 16));
    dst += dst_stride;
    LD_UB2(dst, 16, dst2, dst3);
    PCKEV_AVG_ST_UB(res5, res4, dst2, dst);
    PCKEV_AVG_ST_UB(res7, res6, dst3, (dst + 16));
    dst += dst_stride;
  }
}

static void common_hz_2t_and_aver_dst_64w_msa(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask;
  v16u8 filt0, dst0, dst1, dst2, dst3;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 out0, out1, out2, out3, out4, out5, out6, out7, filt;

  mask = LD_SB(&mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  for (loop_cnt = height; loop_cnt--;) {
    LD_SB4(src, 16, src0, src2, src4, src6);
    src7 = LD_SB(src + 56);
    SLDI_B3_SB(src2, src4, src6, src0, src2, src4, src1, src3, src5, 8);
    src += src_stride;

    VSHF_B2_UB(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UB(src2, src2, src3, src3, mask, mask, vec2, vec3);
    VSHF_B2_UB(src4, src4, src5, src5, mask, mask, vec4, vec5);
    VSHF_B2_UB(src6, src6, src7, src7, mask, mask, vec6, vec7);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, out0, out1,
                out2, out3);
    DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, out4, out5,
                out6, out7);
    SRARI_H4_UH(out0, out1, out2, out3, FILTER_BITS);
    SRARI_H4_UH(out4, out5, out6, out7, FILTER_BITS);
    LD_UB4(dst, 16, dst0, dst1, dst2, dst3);
    PCKEV_AVG_ST_UB(out1, out0, dst0, dst);
    PCKEV_AVG_ST_UB(out3, out2, dst1, dst + 16);
    PCKEV_AVG_ST_UB(out5, out4, dst2, dst + 32);
    PCKEV_AVG_ST_UB(out7, out6, dst3, dst + 48);
    dst += dst_stride;
  }
}

void vpx_convolve8_avg_horiz_msa(const uint8_t *src, ptrdiff_t src_stride,
                                 uint8_t *dst, ptrdiff_t dst_stride,
                                 const InterpKernel *filter, int x0_q4,
                                 int x_step_q4, int y0_q4, int y_step_q4, int w,
                                 int h) {
  const int16_t *const filter_x = filter[x0_q4];
  int8_t cnt, filt_hor[8];

  assert(x_step_q4 == 16);
  assert(((const int32_t *)filter_x)[1] != 0x800000);

  for (cnt = 0; cnt < 8; ++cnt) {
    filt_hor[cnt] = filter_x[cnt];
  }

  if (vpx_get_filter_taps(filter_x) == 2) {
    switch (w) {
      case 4:
        common_hz_2t_and_aver_dst_4w_msa(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, &filt_hor[3], h);
        break;
      case 8:
        common_hz_2t_and_aver_dst_8w_msa(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, &filt_hor[3], h);
        break;
      case 16:
        common_hz_2t_and_aver_dst_16w_msa(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, &filt_hor[3], h);
        break;
      case 32:
        common_hz_2t_and_aver_dst_32w_msa(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, &filt_hor[3], h);
        break;
      case 64:
        common_hz_2t_and_aver_dst_64w_msa(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, &filt_hor[3], h);
        break;
      default:
        vpx_convolve8_avg_horiz_c(src, src_stride, dst, dst_stride, filter,
                                  x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  } else {
    switch (w) {
      case 4:
        common_hz_8t_and_aver_dst_4w_msa(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, filt_hor, h);
        break;
      case 8:
        common_hz_8t_and_aver_dst_8w_msa(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, filt_hor, h);
        break;
      case 16:
        common_hz_8t_and_aver_dst_16w_msa(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, filt_hor, h);
        break;
      case 32:
        common_hz_8t_and_aver_dst_32w_msa(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, filt_hor, h);
        break;
      case 64:
        common_hz_8t_and_aver_dst_64w_msa(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, filt_hor, h);
        break;
      default:
        vpx_convolve8_avg_horiz_c(src, src_stride, dst, dst_stride, filter,
                                  x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  }
}
