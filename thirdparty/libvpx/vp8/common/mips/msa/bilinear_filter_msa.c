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
#include "vpx_ports/mem.h"
#include "vp8/common/filter.h"
#include "vp8/common/mips/msa/vp8_macros_msa.h"

DECLARE_ALIGNED(16, static const int8_t, vp8_bilinear_filters_msa[7][2]) = {
  { 112, 16 }, { 96, 32 }, { 80, 48 }, { 64, 64 },
  { 48, 80 },  { 32, 96 }, { 16, 112 }
};

static const uint8_t vp8_mc_filt_mask_arr[16 * 3] = {
  /* 8 width cases */
  0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
  /* 4 width cases */
  0, 1, 1, 2, 2, 3, 3, 4, 16, 17, 17, 18, 18, 19, 19, 20,
  /* 4 width cases */
  8, 9, 9, 10, 10, 11, 11, 12, 24, 25, 25, 26, 26, 27, 27, 28
};

static void common_hz_2t_4x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  v16i8 src0, src1, src2, src3, mask;
  v16u8 filt0, vec0, vec1, res0, res1;
  v8u16 vec2, vec3, filt;

  mask = LD_SB(&vp8_mc_filt_mask_arr[16]);

  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  VSHF_B2_UB(src0, src1, src2, src3, mask, mask, vec0, vec1);
  DOTP_UB2_UH(vec0, vec1, filt0, filt0, vec2, vec3);
  SRARI_H2_UH(vec2, vec3, VP8_FILTER_SHIFT);
  PCKEV_B2_UB(vec2, vec2, vec3, vec3, res0, res1);
  ST4x4_UB(res0, res1, 0, 1, 0, 1, dst, dst_stride);
}

static void common_hz_2t_4x8_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  v16u8 vec0, vec1, vec2, vec3, filt0;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask;
  v16i8 res0, res1, res2, res3;
  v8u16 vec4, vec5, vec6, vec7, filt;

  mask = LD_SB(&vp8_mc_filt_mask_arr[16]);

  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  VSHF_B2_UB(src0, src1, src2, src3, mask, mask, vec0, vec1);
  VSHF_B2_UB(src4, src5, src6, src7, mask, mask, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec4, vec5,
              vec6, vec7);
  SRARI_H4_UH(vec4, vec5, vec6, vec7, VP8_FILTER_SHIFT);
  PCKEV_B4_SB(vec4, vec4, vec5, vec5, vec6, vec6, vec7, vec7, res0, res1, res2,
              res3);
  ST4x4_UB(res0, res1, 0, 1, 0, 1, dst, dst_stride);
  dst += (4 * dst_stride);
  ST4x4_UB(res2, res3, 0, 1, 0, 1, dst, dst_stride);
}

static void common_hz_2t_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  if (4 == height) {
    common_hz_2t_4x4_msa(src, src_stride, dst, dst_stride, filter);
  } else if (8 == height) {
    common_hz_2t_4x8_msa(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_hz_2t_8x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  v16u8 filt0;
  v16i8 src0, src1, src2, src3, mask;
  v8u16 vec0, vec1, vec2, vec3, filt;

  mask = LD_SB(&vp8_mc_filt_mask_arr[0]);

  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
  VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
              vec2, vec3);
  SRARI_H4_UH(vec0, vec1, vec2, vec3, VP8_FILTER_SHIFT);
  PCKEV_B2_SB(vec1, vec0, vec3, vec2, src0, src1);
  ST8x4_UB(src0, src1, dst, dst_stride);
}

static void common_hz_2t_8x8mult_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter, int32_t height) {
  v16u8 filt0;
  v16i8 src0, src1, src2, src3, mask, out0, out1;
  v8u16 vec0, vec1, vec2, vec3, filt;

  mask = LD_SB(&vp8_mc_filt_mask_arr[0]);

  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  src += (4 * src_stride);

  VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
  VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
              vec2, vec3);
  SRARI_H4_UH(vec0, vec1, vec2, vec3, VP8_FILTER_SHIFT);

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  src += (4 * src_stride);

  PCKEV_B2_SB(vec1, vec0, vec3, vec2, out0, out1);
  ST8x4_UB(out0, out1, dst, dst_stride);
  dst += (4 * dst_stride);

  VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
  VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
              vec2, vec3);
  SRARI_H4_UH(vec0, vec1, vec2, vec3, VP8_FILTER_SHIFT);
  PCKEV_B2_SB(vec1, vec0, vec3, vec2, out0, out1);
  ST8x4_UB(out0, out1, dst, dst_stride);
  dst += (4 * dst_stride);

  if (16 == height) {
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);

    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    SRARI_H4_UH(vec0, vec1, vec2, vec3, VP8_FILTER_SHIFT);
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);

    PCKEV_B2_SB(vec1, vec0, vec3, vec2, out0, out1);
    ST8x4_UB(out0, out1, dst, dst_stride);

    VSHF_B2_UH(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UH(src2, src2, src3, src3, mask, mask, vec2, vec3);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, vec0, vec1,
                vec2, vec3);
    SRARI_H4_UH(vec0, vec1, vec2, vec3, VP8_FILTER_SHIFT);
    PCKEV_B2_SB(vec1, vec0, vec3, vec2, out0, out1);
    ST8x4_UB(out0, out1, dst + 4 * dst_stride, dst_stride);
  }
}

static void common_hz_2t_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  if (4 == height) {
    common_hz_2t_8x4_msa(src, src_stride, dst, dst_stride, filter);
  } else {
    common_hz_2t_8x8mult_msa(src, src_stride, dst, dst_stride, filter, height);
  }
}

static void common_hz_2t_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask;
  v16u8 filt0, vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  v8u16 out0, out1, out2, out3, out4, out5, out6, out7, filt;

  mask = LD_SB(&vp8_mc_filt_mask_arr[0]);

  loop_cnt = (height >> 2) - 1;

  filt = LD_UH(filter);
  filt0 = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB4(src, src_stride, src0, src2, src4, src6);
  LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
  src += (4 * src_stride);

  VSHF_B2_UB(src0, src0, src1, src1, mask, mask, vec0, vec1);
  VSHF_B2_UB(src2, src2, src3, src3, mask, mask, vec2, vec3);
  VSHF_B2_UB(src4, src4, src5, src5, mask, mask, vec4, vec5);
  VSHF_B2_UB(src6, src6, src7, src7, mask, mask, vec6, vec7);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, out0, out1,
              out2, out3);
  DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, out4, out5,
              out6, out7);
  SRARI_H4_UH(out0, out1, out2, out3, VP8_FILTER_SHIFT);
  SRARI_H4_UH(out4, out5, out6, out7, VP8_FILTER_SHIFT);
  PCKEV_ST_SB(out0, out1, dst);
  dst += dst_stride;
  PCKEV_ST_SB(out2, out3, dst);
  dst += dst_stride;
  PCKEV_ST_SB(out4, out5, dst);
  dst += dst_stride;
  PCKEV_ST_SB(out6, out7, dst);
  dst += dst_stride;

  for (; loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src2, src4, src6);
    LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);

    VSHF_B2_UB(src0, src0, src1, src1, mask, mask, vec0, vec1);
    VSHF_B2_UB(src2, src2, src3, src3, mask, mask, vec2, vec3);
    VSHF_B2_UB(src4, src4, src5, src5, mask, mask, vec4, vec5);
    VSHF_B2_UB(src6, src6, src7, src7, mask, mask, vec6, vec7);
    DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, out0, out1,
                out2, out3);
    DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, out4, out5,
                out6, out7);
    SRARI_H4_UH(out0, out1, out2, out3, VP8_FILTER_SHIFT);
    SRARI_H4_UH(out4, out5, out6, out7, VP8_FILTER_SHIFT);
    PCKEV_ST_SB(out0, out1, dst);
    dst += dst_stride;
    PCKEV_ST_SB(out2, out3, dst);
    dst += dst_stride;
    PCKEV_ST_SB(out4, out5, dst);
    dst += dst_stride;
    PCKEV_ST_SB(out6, out7, dst);
    dst += dst_stride;
  }
}

static void common_vt_2t_4x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
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
  SRARI_H2_UH(tmp0, tmp1, VP8_FILTER_SHIFT);
  src2110 = __msa_pckev_b((v16i8)tmp1, (v16i8)tmp0);
  ST4x4_UB(src2110, src2110, 0, 1, 2, 3, dst, dst_stride);
}

static void common_vt_2t_4x8_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
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
  SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, VP8_FILTER_SHIFT);
  PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, src2110, src4332);
  ST4x4_UB(src2110, src2110, 0, 1, 2, 3, dst, dst_stride);
  ST4x4_UB(src4332, src4332, 0, 1, 2, 3, dst + 4 * dst_stride, dst_stride);
}

static void common_vt_2t_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  if (4 == height) {
    common_vt_2t_4x4_msa(src, src_stride, dst, dst_stride, filter);
  } else if (8 == height) {
    common_vt_2t_4x8_msa(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_vt_2t_8x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  v16u8 src0, src1, src2, src3, src4, vec0, vec1, vec2, vec3, filt0;
  v16i8 out0, out1;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  filt = LD_SH(filter);
  filt0 = (v16u8)__msa_splati_h(filt, 0);

  LD_UB5(src, src_stride, src0, src1, src2, src3, src4);
  ILVR_B2_UB(src1, src0, src2, src1, vec0, vec1);
  ILVR_B2_UB(src3, src2, src4, src3, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt0, filt0, filt0, filt0, tmp0, tmp1,
              tmp2, tmp3);
  SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, VP8_FILTER_SHIFT);
  PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, out0, out1);
  ST8x4_UB(out0, out1, dst, dst_stride);
}

static void common_vt_2t_8x8mult_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7, src8;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, filt0;
  v16i8 out0, out1;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

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
    SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, VP8_FILTER_SHIFT);
    PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, out0, out1);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);

    DOTP_UB4_UH(vec4, vec5, vec6, vec7, filt0, filt0, filt0, filt0, tmp0, tmp1,
                tmp2, tmp3);
    SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, VP8_FILTER_SHIFT);
    PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, out0, out1);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);

    src0 = src8;
  }
}

static void common_vt_2t_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  if (4 == height) {
    common_vt_2t_8x4_msa(src, src_stride, dst, dst_stride, filter);
  } else {
    common_vt_2t_8x8mult_msa(src, src_stride, dst, dst_stride, filter, height);
  }
}

static void common_vt_2t_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16u8 src0, src1, src2, src3, src4;
  v16u8 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, filt0;
  v8u16 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

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
    SRARI_H2_UH(tmp0, tmp1, VP8_FILTER_SHIFT);
    PCKEV_ST_SB(tmp0, tmp1, dst);
    dst += dst_stride;

    ILVR_B2_UB(src3, src2, src4, src3, vec4, vec6);
    ILVL_B2_UB(src3, src2, src4, src3, vec5, vec7);
    DOTP_UB2_UH(vec2, vec3, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, VP8_FILTER_SHIFT);
    PCKEV_ST_SB(tmp2, tmp3, dst);
    dst += dst_stride;

    DOTP_UB2_UH(vec4, vec5, filt0, filt0, tmp0, tmp1);
    SRARI_H2_UH(tmp0, tmp1, VP8_FILTER_SHIFT);
    PCKEV_ST_SB(tmp0, tmp1, dst);
    dst += dst_stride;

    DOTP_UB2_UH(vec6, vec7, filt0, filt0, tmp2, tmp3);
    SRARI_H2_UH(tmp2, tmp3, VP8_FILTER_SHIFT);
    PCKEV_ST_SB(tmp2, tmp3, dst);
    dst += dst_stride;

    src0 = src4;
  }
}

static void common_hv_2ht_2vt_4x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert) {
  v16i8 src0, src1, src2, src3, src4, mask;
  v16u8 filt_vt, filt_hz, vec0, vec1, res0, res1;
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, filt, tmp0, tmp1;

  mask = LD_SB(&vp8_mc_filt_mask_arr[16]);

  filt = LD_UH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h((v8i16)filt, 0);
  filt = LD_UH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src1, mask, filt_hz, VP8_FILTER_SHIFT);
  hz_out2 = HORIZ_2TAP_FILT_UH(src2, src3, mask, filt_hz, VP8_FILTER_SHIFT);
  hz_out4 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, VP8_FILTER_SHIFT);
  hz_out1 = (v8u16)__msa_sldi_b((v16i8)hz_out2, (v16i8)hz_out0, 8);
  hz_out3 = (v8u16)__msa_pckod_d((v2i64)hz_out4, (v2i64)hz_out2);

  ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
  DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp0, tmp1);
  SRARI_H2_UH(tmp0, tmp1, VP8_FILTER_SHIFT);
  PCKEV_B2_UB(tmp0, tmp0, tmp1, tmp1, res0, res1);
  ST4x4_UB(res0, res1, 0, 1, 0, 1, dst, dst_stride);
}

static void common_hv_2ht_2vt_4x8_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert) {
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8, mask;
  v16i8 res0, res1, res2, res3;
  v16u8 filt_hz, filt_vt, vec0, vec1, vec2, vec3;
  v8u16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8u16 hz_out7, hz_out8, vec4, vec5, vec6, vec7, filt;

  mask = LD_SB(&vp8_mc_filt_mask_arr[16]);

  filt = LD_UH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h((v8i16)filt, 0);
  filt = LD_UH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h((v8i16)filt, 0);

  LD_SB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  src += (8 * src_stride);
  src8 = LD_SB(src);

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src1, mask, filt_hz, VP8_FILTER_SHIFT);
  hz_out2 = HORIZ_2TAP_FILT_UH(src2, src3, mask, filt_hz, VP8_FILTER_SHIFT);
  hz_out4 = HORIZ_2TAP_FILT_UH(src4, src5, mask, filt_hz, VP8_FILTER_SHIFT);
  hz_out6 = HORIZ_2TAP_FILT_UH(src6, src7, mask, filt_hz, VP8_FILTER_SHIFT);
  hz_out8 = HORIZ_2TAP_FILT_UH(src8, src8, mask, filt_hz, VP8_FILTER_SHIFT);
  SLDI_B3_UH(hz_out2, hz_out4, hz_out6, hz_out0, hz_out2, hz_out4, hz_out1,
             hz_out3, hz_out5, 8);
  hz_out7 = (v8u16)__msa_pckod_d((v2i64)hz_out8, (v2i64)hz_out6);

  ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
  ILVEV_B2_UB(hz_out4, hz_out5, hz_out6, hz_out7, vec2, vec3);
  DOTP_UB4_UH(vec0, vec1, vec2, vec3, filt_vt, filt_vt, filt_vt, filt_vt, vec4,
              vec5, vec6, vec7);
  SRARI_H4_UH(vec4, vec5, vec6, vec7, VP8_FILTER_SHIFT);
  PCKEV_B4_SB(vec4, vec4, vec5, vec5, vec6, vec6, vec7, vec7, res0, res1, res2,
              res3);
  ST4x4_UB(res0, res1, 0, 1, 0, 1, dst, dst_stride);
  dst += (4 * dst_stride);
  ST4x4_UB(res2, res3, 0, 1, 0, 1, dst, dst_stride);
}

static void common_hv_2ht_2vt_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  if (4 == height) {
    common_hv_2ht_2vt_4x4_msa(src, src_stride, dst, dst_stride, filter_horiz,
                              filter_vert);
  } else if (8 == height) {
    common_hv_2ht_2vt_4x8_msa(src, src_stride, dst, dst_stride, filter_horiz,
                              filter_vert);
  }
}

static void common_hv_2ht_2vt_8x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert) {
  v16i8 src0, src1, src2, src3, src4, mask, out0, out1;
  v16u8 filt_hz, filt_vt, vec0, vec1, vec2, vec3;
  v8u16 hz_out0, hz_out1, tmp0, tmp1, tmp2, tmp3;
  v8i16 filt;

  mask = LD_SB(&vp8_mc_filt_mask_arr[0]);

  filt = LD_SH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h(filt, 0);
  filt = LD_SH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h(filt, 0);

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, VP8_FILTER_SHIFT);
  hz_out1 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, VP8_FILTER_SHIFT);
  vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
  tmp0 = __msa_dotp_u_h(vec0, filt_vt);

  hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, VP8_FILTER_SHIFT);
  vec1 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
  tmp1 = __msa_dotp_u_h(vec1, filt_vt);

  hz_out1 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, VP8_FILTER_SHIFT);
  vec2 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
  tmp2 = __msa_dotp_u_h(vec2, filt_vt);

  hz_out0 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, VP8_FILTER_SHIFT);
  vec3 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
  tmp3 = __msa_dotp_u_h(vec3, filt_vt);

  SRARI_H4_UH(tmp0, tmp1, tmp2, tmp3, VP8_FILTER_SHIFT);
  PCKEV_B2_SB(tmp1, tmp0, tmp3, tmp2, out0, out1);
  ST8x4_UB(out0, out1, dst, dst_stride);
}

static void common_hv_2ht_2vt_8x8mult_msa(
    uint8_t *RESTRICT src, int32_t src_stride, uint8_t *RESTRICT dst,
    int32_t dst_stride, const int8_t *filter_horiz, const int8_t *filter_vert,
    int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, mask, out0, out1;
  v16u8 filt_hz, filt_vt, vec0;
  v8u16 hz_out0, hz_out1, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
  v8i16 filt;

  mask = LD_SB(&vp8_mc_filt_mask_arr[0]);

  filt = LD_SH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h(filt, 0);
  filt = LD_SH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h(filt, 0);

  src0 = LD_SB(src);
  src += src_stride;

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, VP8_FILTER_SHIFT);

  for (loop_cnt = (height >> 3); loop_cnt--;) {
    LD_SB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);

    hz_out1 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, VP8_FILTER_SHIFT);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp1 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, VP8_FILTER_SHIFT);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp2 = __msa_dotp_u_h(vec0, filt_vt);

    SRARI_H2_UH(tmp1, tmp2, VP8_FILTER_SHIFT);

    hz_out1 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, VP8_FILTER_SHIFT);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp3 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, VP8_FILTER_SHIFT);
    LD_SB4(src, src_stride, src1, src2, src3, src4);
    src += (4 * src_stride);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp4 = __msa_dotp_u_h(vec0, filt_vt);

    SRARI_H2_UH(tmp3, tmp4, VP8_FILTER_SHIFT);
    PCKEV_B2_SB(tmp2, tmp1, tmp4, tmp3, out0, out1);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out1 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, VP8_FILTER_SHIFT);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp5 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, VP8_FILTER_SHIFT);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp6 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out1 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, VP8_FILTER_SHIFT);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp7 = __msa_dotp_u_h(vec0, filt_vt);

    hz_out0 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, VP8_FILTER_SHIFT);
    vec0 = (v16u8)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out1);
    tmp8 = __msa_dotp_u_h(vec0, filt_vt);

    SRARI_H4_UH(tmp5, tmp6, tmp7, tmp8, VP8_FILTER_SHIFT);
    PCKEV_B2_SB(tmp6, tmp5, tmp8, tmp7, out0, out1);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void common_hv_2ht_2vt_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  if (4 == height) {
    common_hv_2ht_2vt_8x4_msa(src, src_stride, dst, dst_stride, filter_horiz,
                              filter_vert);
  } else {
    common_hv_2ht_2vt_8x8mult_msa(src, src_stride, dst, dst_stride,
                                  filter_horiz, filter_vert, height);
  }
}

static void common_hv_2ht_2vt_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert,
                                      int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, mask;
  v16u8 filt_hz, filt_vt, vec0, vec1;
  v8u16 tmp1, tmp2, hz_out0, hz_out1, hz_out2, hz_out3;
  v8i16 filt;

  mask = LD_SB(&vp8_mc_filt_mask_arr[0]);

  /* rearranging filter */
  filt = LD_SH(filter_horiz);
  filt_hz = (v16u8)__msa_splati_h(filt, 0);
  filt = LD_SH(filter_vert);
  filt_vt = (v16u8)__msa_splati_h(filt, 0);

  LD_SB2(src, 8, src0, src1);
  src += src_stride;

  hz_out0 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, VP8_FILTER_SHIFT);
  hz_out2 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, VP8_FILTER_SHIFT);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src2, src4, src6);
    LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);

    hz_out1 = HORIZ_2TAP_FILT_UH(src0, src0, mask, filt_hz, VP8_FILTER_SHIFT);
    hz_out3 = HORIZ_2TAP_FILT_UH(src1, src1, mask, filt_hz, VP8_FILTER_SHIFT);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp1, tmp2);
    SRARI_H2_UH(tmp1, tmp2, VP8_FILTER_SHIFT);
    PCKEV_ST_SB(tmp1, tmp2, dst);
    dst += dst_stride;

    hz_out0 = HORIZ_2TAP_FILT_UH(src2, src2, mask, filt_hz, VP8_FILTER_SHIFT);
    hz_out2 = HORIZ_2TAP_FILT_UH(src3, src3, mask, filt_hz, VP8_FILTER_SHIFT);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp1, tmp2);
    SRARI_H2_UH(tmp1, tmp2, VP8_FILTER_SHIFT);
    PCKEV_ST_SB(tmp1, tmp2, dst);
    dst += dst_stride;

    hz_out1 = HORIZ_2TAP_FILT_UH(src4, src4, mask, filt_hz, VP8_FILTER_SHIFT);
    hz_out3 = HORIZ_2TAP_FILT_UH(src5, src5, mask, filt_hz, VP8_FILTER_SHIFT);
    ILVEV_B2_UB(hz_out0, hz_out1, hz_out2, hz_out3, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp1, tmp2);
    SRARI_H2_UH(tmp1, tmp2, VP8_FILTER_SHIFT);
    PCKEV_ST_SB(tmp1, tmp2, dst);
    dst += dst_stride;

    hz_out0 = HORIZ_2TAP_FILT_UH(src6, src6, mask, filt_hz, VP8_FILTER_SHIFT);
    hz_out2 = HORIZ_2TAP_FILT_UH(src7, src7, mask, filt_hz, VP8_FILTER_SHIFT);
    ILVEV_B2_UB(hz_out1, hz_out0, hz_out3, hz_out2, vec0, vec1);
    DOTP_UB2_UH(vec0, vec1, filt_vt, filt_vt, tmp1, tmp2);
    SRARI_H2_UH(tmp1, tmp2, VP8_FILTER_SHIFT);
    PCKEV_ST_SB(tmp1, tmp2, dst);
    dst += dst_stride;
  }
}

void vp8_bilinear_predict4x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 int32_t xoffset, int32_t yoffset,
                                 uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_bilinear_filters_msa[xoffset - 1];
  const int8_t *v_filter = vp8_bilinear_filters_msa[yoffset - 1];

  if (yoffset) {
    if (xoffset) {
      common_hv_2ht_2vt_4w_msa(src, src_stride, dst, dst_stride, h_filter,
                               v_filter, 4);
    } else {
      common_vt_2t_4w_msa(src, src_stride, dst, dst_stride, v_filter, 4);
    }
  } else {
    if (xoffset) {
      common_hz_2t_4w_msa(src, src_stride, dst, dst_stride, h_filter, 4);
    } else {
      uint32_t tp0, tp1, tp2, tp3;

      LW4(src, src_stride, tp0, tp1, tp2, tp3);
      SW4(tp0, tp1, tp2, tp3, dst, dst_stride);
    }
  }
}

void vp8_bilinear_predict8x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 int32_t xoffset, int32_t yoffset,
                                 uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_bilinear_filters_msa[xoffset - 1];
  const int8_t *v_filter = vp8_bilinear_filters_msa[yoffset - 1];

  if (yoffset) {
    if (xoffset) {
      common_hv_2ht_2vt_8w_msa(src, src_stride, dst, dst_stride, h_filter,
                               v_filter, 4);
    } else {
      common_vt_2t_8w_msa(src, src_stride, dst, dst_stride, v_filter, 4);
    }
  } else {
    if (xoffset) {
      common_hz_2t_8w_msa(src, src_stride, dst, dst_stride, h_filter, 4);
    } else {
      vp8_copy_mem8x4(src, src_stride, dst, dst_stride);
    }
  }
}

void vp8_bilinear_predict8x8_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 int32_t xoffset, int32_t yoffset,
                                 uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_bilinear_filters_msa[xoffset - 1];
  const int8_t *v_filter = vp8_bilinear_filters_msa[yoffset - 1];

  if (yoffset) {
    if (xoffset) {
      common_hv_2ht_2vt_8w_msa(src, src_stride, dst, dst_stride, h_filter,
                               v_filter, 8);
    } else {
      common_vt_2t_8w_msa(src, src_stride, dst, dst_stride, v_filter, 8);
    }
  } else {
    if (xoffset) {
      common_hz_2t_8w_msa(src, src_stride, dst, dst_stride, h_filter, 8);
    } else {
      vp8_copy_mem8x8(src, src_stride, dst, dst_stride);
    }
  }
}

void vp8_bilinear_predict16x16_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                   int32_t xoffset, int32_t yoffset,
                                   uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_bilinear_filters_msa[xoffset - 1];
  const int8_t *v_filter = vp8_bilinear_filters_msa[yoffset - 1];

  if (yoffset) {
    if (xoffset) {
      common_hv_2ht_2vt_16w_msa(src, src_stride, dst, dst_stride, h_filter,
                                v_filter, 16);
    } else {
      common_vt_2t_16w_msa(src, src_stride, dst, dst_stride, v_filter, 16);
    }
  } else {
    if (xoffset) {
      common_hz_2t_16w_msa(src, src_stride, dst, dst_stride, h_filter, 16);
    } else {
      vp8_copy_mem16x16(src, src_stride, dst, dst_stride);
    }
  }
}
