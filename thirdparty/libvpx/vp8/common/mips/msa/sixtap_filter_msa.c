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

DECLARE_ALIGNED(16, static const int8_t, vp8_subpel_filters_msa[7][8]) = {
  { 0, -6, 123, 12, -1, 0, 0, 0 },
  { 2, -11, 108, 36, -8, 1, 0, 0 }, /* New 1/4 pel 6 tap filter */
  { 0, -9, 93, 50, -6, 0, 0, 0 },
  { 3, -16, 77, 77, -16, 3, 0, 0 }, /* New 1/2 pel 6 tap filter */
  { 0, -6, 50, 93, -9, 0, 0, 0 },
  { 1, -8, 36, 108, -11, 2, 0, 0 }, /* New 1/4 pel 6 tap filter */
  { 0, -1, 12, 123, -6, 0, 0, 0 },
};

static const uint8_t vp8_mc_filt_mask_arr[16 * 3] = {
  /* 8 width cases */
  0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
  /* 4 width cases */
  0, 1, 1, 2, 2, 3, 3, 4, 16, 17, 17, 18, 18, 19, 19, 20,
  /* 4 width cases */
  8, 9, 9, 10, 10, 11, 11, 12, 24, 25, 25, 26, 26, 27, 27, 28
};

#define HORIZ_6TAP_FILT(src0, src1, mask0, mask1, mask2, filt_h0, filt_h1, \
                        filt_h2)                                           \
  ({                                                                       \
    v16i8 _6tap_vec0_m, _6tap_vec1_m, _6tap_vec2_m;                        \
    v8i16 _6tap_out_m;                                                     \
                                                                           \
    VSHF_B3_SB(src0, src1, src0, src1, src0, src1, mask0, mask1, mask2,    \
               _6tap_vec0_m, _6tap_vec1_m, _6tap_vec2_m);                  \
    _6tap_out_m = DPADD_SH3_SH(_6tap_vec0_m, _6tap_vec1_m, _6tap_vec2_m,   \
                               filt_h0, filt_h1, filt_h2);                 \
                                                                           \
    _6tap_out_m = __msa_srari_h(_6tap_out_m, VP8_FILTER_SHIFT);            \
    _6tap_out_m = __msa_sat_s_h(_6tap_out_m, 7);                           \
                                                                           \
    _6tap_out_m;                                                           \
  })

#define HORIZ_6TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,   \
                                   mask2, filt0, filt1, filt2, out0, out1) \
  {                                                                        \
    v16i8 _6tap_4wid_vec0_m, _6tap_4wid_vec1_m, _6tap_4wid_vec2_m,         \
        _6tap_4wid_vec3_m, _6tap_4wid_vec4_m, _6tap_4wid_vec5_m;           \
                                                                           \
    VSHF_B2_SB(src0, src1, src2, src3, mask0, mask0, _6tap_4wid_vec0_m,    \
               _6tap_4wid_vec1_m);                                         \
    DOTP_SB2_SH(_6tap_4wid_vec0_m, _6tap_4wid_vec1_m, filt0, filt0, out0,  \
                out1);                                                     \
    VSHF_B2_SB(src0, src1, src2, src3, mask1, mask1, _6tap_4wid_vec2_m,    \
               _6tap_4wid_vec3_m);                                         \
    DPADD_SB2_SH(_6tap_4wid_vec2_m, _6tap_4wid_vec3_m, filt1, filt1, out0, \
                 out1);                                                    \
    VSHF_B2_SB(src0, src1, src2, src3, mask2, mask2, _6tap_4wid_vec4_m,    \
               _6tap_4wid_vec5_m);                                         \
    DPADD_SB2_SH(_6tap_4wid_vec4_m, _6tap_4wid_vec5_m, filt2, filt2, out0, \
                 out1);                                                    \
  }

#define HORIZ_6TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,    \
                                   mask2, filt0, filt1, filt2, out0, out1,  \
                                   out2, out3)                              \
  {                                                                         \
    v16i8 _6tap_8wid_vec0_m, _6tap_8wid_vec1_m, _6tap_8wid_vec2_m,          \
        _6tap_8wid_vec3_m, _6tap_8wid_vec4_m, _6tap_8wid_vec5_m,            \
        _6tap_8wid_vec6_m, _6tap_8wid_vec7_m;                               \
                                                                            \
    VSHF_B2_SB(src0, src0, src1, src1, mask0, mask0, _6tap_8wid_vec0_m,     \
               _6tap_8wid_vec1_m);                                          \
    VSHF_B2_SB(src2, src2, src3, src3, mask0, mask0, _6tap_8wid_vec2_m,     \
               _6tap_8wid_vec3_m);                                          \
    DOTP_SB4_SH(_6tap_8wid_vec0_m, _6tap_8wid_vec1_m, _6tap_8wid_vec2_m,    \
                _6tap_8wid_vec3_m, filt0, filt0, filt0, filt0, out0, out1,  \
                out2, out3);                                                \
    VSHF_B2_SB(src0, src0, src1, src1, mask1, mask1, _6tap_8wid_vec0_m,     \
               _6tap_8wid_vec1_m);                                          \
    VSHF_B2_SB(src2, src2, src3, src3, mask1, mask1, _6tap_8wid_vec2_m,     \
               _6tap_8wid_vec3_m);                                          \
    VSHF_B2_SB(src0, src0, src1, src1, mask2, mask2, _6tap_8wid_vec4_m,     \
               _6tap_8wid_vec5_m);                                          \
    VSHF_B2_SB(src2, src2, src3, src3, mask2, mask2, _6tap_8wid_vec6_m,     \
               _6tap_8wid_vec7_m);                                          \
    DPADD_SB4_SH(_6tap_8wid_vec0_m, _6tap_8wid_vec1_m, _6tap_8wid_vec2_m,   \
                 _6tap_8wid_vec3_m, filt1, filt1, filt1, filt1, out0, out1, \
                 out2, out3);                                               \
    DPADD_SB4_SH(_6tap_8wid_vec4_m, _6tap_8wid_vec5_m, _6tap_8wid_vec6_m,   \
                 _6tap_8wid_vec7_m, filt2, filt2, filt2, filt2, out0, out1, \
                 out2, out3);                                               \
  }

#define FILT_4TAP_DPADD_S_H(vec0, vec1, filt0, filt1)                 \
  ({                                                                  \
    v8i16 _4tap_dpadd_tmp0;                                           \
                                                                      \
    _4tap_dpadd_tmp0 = __msa_dotp_s_h((v16i8)vec0, (v16i8)filt0);     \
    _4tap_dpadd_tmp0 =                                                \
        __msa_dpadd_s_h(_4tap_dpadd_tmp0, (v16i8)vec1, (v16i8)filt1); \
                                                                      \
    _4tap_dpadd_tmp0;                                                 \
  })

#define HORIZ_4TAP_FILT(src0, src1, mask0, mask1, filt_h0, filt_h1)        \
  ({                                                                       \
    v16i8 _4tap_vec0_m, _4tap_vec1_m;                                      \
    v8i16 _4tap_out_m;                                                     \
                                                                           \
    VSHF_B2_SB(src0, src1, src0, src1, mask0, mask1, _4tap_vec0_m,         \
               _4tap_vec1_m);                                              \
    _4tap_out_m =                                                          \
        FILT_4TAP_DPADD_S_H(_4tap_vec0_m, _4tap_vec1_m, filt_h0, filt_h1); \
                                                                           \
    _4tap_out_m = __msa_srari_h(_4tap_out_m, VP8_FILTER_SHIFT);            \
    _4tap_out_m = __msa_sat_s_h(_4tap_out_m, 7);                           \
                                                                           \
    _4tap_out_m;                                                           \
  })

#define HORIZ_4TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,   \
                                   filt0, filt1, out0, out1)               \
  {                                                                        \
    v16i8 _4tap_4wid_vec0_m, _4tap_4wid_vec1_m, _4tap_4wid_vec2_m,         \
        _4tap_4wid_vec3_m;                                                 \
                                                                           \
    VSHF_B2_SB(src0, src1, src2, src3, mask0, mask0, _4tap_4wid_vec0_m,    \
               _4tap_4wid_vec1_m);                                         \
    DOTP_SB2_SH(_4tap_4wid_vec0_m, _4tap_4wid_vec1_m, filt0, filt0, out0,  \
                out1);                                                     \
    VSHF_B2_SB(src0, src1, src2, src3, mask1, mask1, _4tap_4wid_vec2_m,    \
               _4tap_4wid_vec3_m);                                         \
    DPADD_SB2_SH(_4tap_4wid_vec2_m, _4tap_4wid_vec3_m, filt1, filt1, out0, \
                 out1);                                                    \
  }

#define HORIZ_4TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,    \
                                   filt0, filt1, out0, out1, out2, out3)    \
  {                                                                         \
    v16i8 _4tap_8wid_vec0_m, _4tap_8wid_vec1_m, _4tap_8wid_vec2_m,          \
        _4tap_8wid_vec3_m;                                                  \
                                                                            \
    VSHF_B2_SB(src0, src0, src1, src1, mask0, mask0, _4tap_8wid_vec0_m,     \
               _4tap_8wid_vec1_m);                                          \
    VSHF_B2_SB(src2, src2, src3, src3, mask0, mask0, _4tap_8wid_vec2_m,     \
               _4tap_8wid_vec3_m);                                          \
    DOTP_SB4_SH(_4tap_8wid_vec0_m, _4tap_8wid_vec1_m, _4tap_8wid_vec2_m,    \
                _4tap_8wid_vec3_m, filt0, filt0, filt0, filt0, out0, out1,  \
                out2, out3);                                                \
    VSHF_B2_SB(src0, src0, src1, src1, mask1, mask1, _4tap_8wid_vec0_m,     \
               _4tap_8wid_vec1_m);                                          \
    VSHF_B2_SB(src2, src2, src3, src3, mask1, mask1, _4tap_8wid_vec2_m,     \
               _4tap_8wid_vec3_m);                                          \
    DPADD_SB4_SH(_4tap_8wid_vec0_m, _4tap_8wid_vec1_m, _4tap_8wid_vec2_m,   \
                 _4tap_8wid_vec3_m, filt1, filt1, filt1, filt1, out0, out1, \
                 out2, out3);                                               \
  }

static void common_hz_6t_4x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  v16i8 src0, src1, src2, src3, filt0, filt1, filt2;
  v16u8 mask0, mask1, mask2, out;
  v8i16 filt, out0, out1;

  mask0 = LD_UB(&vp8_mc_filt_mask_arr[16]);
  src -= 2;

  filt = LD_SH(filter);
  SPLATI_H3_SB(filt, 0, 1, 2, filt0, filt1, filt2);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  XORI_B4_128_SB(src0, src1, src2, src3);
  HORIZ_6TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, filt0,
                             filt1, filt2, out0, out1);
  SRARI_H2_SH(out0, out1, VP8_FILTER_SHIFT);
  SAT_SH2_SH(out0, out1, 7);
  out = PCKEV_XORI128_UB(out0, out1);
  ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
}

static void common_hz_6t_4x8_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  v16i8 src0, src1, src2, src3, filt0, filt1, filt2;
  v16u8 mask0, mask1, mask2, out;
  v8i16 filt, out0, out1, out2, out3;

  mask0 = LD_UB(&vp8_mc_filt_mask_arr[16]);
  src -= 2;

  filt = LD_SH(filter);
  SPLATI_H3_SB(filt, 0, 1, 2, filt0, filt1, filt2);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  XORI_B4_128_SB(src0, src1, src2, src3);
  src += (4 * src_stride);
  HORIZ_6TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, filt0,
                             filt1, filt2, out0, out1);
  LD_SB4(src, src_stride, src0, src1, src2, src3);
  XORI_B4_128_SB(src0, src1, src2, src3);
  HORIZ_6TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, filt0,
                             filt1, filt2, out2, out3);
  SRARI_H4_SH(out0, out1, out2, out3, VP8_FILTER_SHIFT);
  SAT_SH4_SH(out0, out1, out2, out3, 7);
  out = PCKEV_XORI128_UB(out0, out1);
  ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
  dst += (4 * dst_stride);
  out = PCKEV_XORI128_UB(out2, out3);
  ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
}

static void common_hz_6t_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  if (4 == height) {
    common_hz_6t_4x4_msa(src, src_stride, dst, dst_stride, filter);
  } else if (8 == height) {
    common_hz_6t_4x8_msa(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_hz_6t_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, filt0, filt1, filt2;
  v16u8 mask0, mask1, mask2, tmp0, tmp1;
  v8i16 filt, out0, out1, out2, out3;

  mask0 = LD_UB(&vp8_mc_filt_mask_arr[0]);
  src -= 2;

  filt = LD_SH(filter);
  SPLATI_H3_SB(filt, 0, 1, 2, filt0, filt1, filt2);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  XORI_B4_128_SB(src0, src1, src2, src3);
  src += (4 * src_stride);
  HORIZ_6TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, filt0,
                             filt1, filt2, out0, out1, out2, out3);
  SRARI_H4_SH(out0, out1, out2, out3, VP8_FILTER_SHIFT);
  SAT_SH4_SH(out0, out1, out2, out3, 7);
  tmp0 = PCKEV_XORI128_UB(out0, out1);
  tmp1 = PCKEV_XORI128_UB(out2, out3);
  ST8x4_UB(tmp0, tmp1, dst, dst_stride);
  dst += (4 * dst_stride);

  for (loop_cnt = (height >> 2) - 1; loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    XORI_B4_128_SB(src0, src1, src2, src3);
    src += (4 * src_stride);
    HORIZ_6TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2,
                               filt0, filt1, filt2, out0, out1, out2, out3);
    SRARI_H4_SH(out0, out1, out2, out3, VP8_FILTER_SHIFT);
    SAT_SH4_SH(out0, out1, out2, out3, 7);
    tmp0 = PCKEV_XORI128_UB(out0, out1);
    tmp1 = PCKEV_XORI128_UB(out2, out3);
    ST8x4_UB(tmp0, tmp1, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void common_hz_6t_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, filt0, filt1, filt2;
  v16u8 mask0, mask1, mask2, out;
  v8i16 filt, out0, out1, out2, out3, out4, out5, out6, out7;

  mask0 = LD_UB(&vp8_mc_filt_mask_arr[0]);
  src -= 2;

  filt = LD_SH(filter);
  SPLATI_H3_SB(filt, 0, 1, 2, filt0, filt1, filt2);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src2, src4, src6);
    LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
    XORI_B8_128_SB(src0, src1, src2, src3, src4, src5, src6, src7);
    src += (4 * src_stride);

    HORIZ_6TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2,
                               filt0, filt1, filt2, out0, out1, out2, out3);
    HORIZ_6TAP_8WID_4VECS_FILT(src4, src5, src6, src7, mask0, mask1, mask2,
                               filt0, filt1, filt2, out4, out5, out6, out7);
    SRARI_H4_SH(out0, out1, out2, out3, VP8_FILTER_SHIFT);
    SRARI_H4_SH(out4, out5, out6, out7, VP8_FILTER_SHIFT);
    SAT_SH4_SH(out0, out1, out2, out3, 7);
    SAT_SH4_SH(out4, out5, out6, out7, 7);
    out = PCKEV_XORI128_UB(out0, out1);
    ST_UB(out, dst);
    dst += dst_stride;
    out = PCKEV_XORI128_UB(out2, out3);
    ST_UB(out, dst);
    dst += dst_stride;
    out = PCKEV_XORI128_UB(out4, out5);
    ST_UB(out, dst);
    dst += dst_stride;
    out = PCKEV_XORI128_UB(out6, out7);
    ST_UB(out, dst);
    dst += dst_stride;
  }
}

static void common_vt_6t_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;
  v16i8 src10_r, src32_r, src54_r, src76_r, src21_r, src43_r, src65_r;
  v16i8 src87_r, src2110, src4332, src6554, src8776, filt0, filt1, filt2;
  v16u8 out;
  v8i16 filt, out10, out32;

  src -= (2 * src_stride);

  filt = LD_SH(filter);
  SPLATI_H3_SB(filt, 0, 1, 2, filt0, filt1, filt2);

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  src += (5 * src_stride);

  ILVR_B4_SB(src1, src0, src2, src1, src3, src2, src4, src3, src10_r, src21_r,
             src32_r, src43_r);
  ILVR_D2_SB(src21_r, src10_r, src43_r, src32_r, src2110, src4332);
  XORI_B2_128_SB(src2110, src4332);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src5, src6, src7, src8);
    src += (4 * src_stride);

    ILVR_B4_SB(src5, src4, src6, src5, src7, src6, src8, src7, src54_r, src65_r,
               src76_r, src87_r);
    ILVR_D2_SB(src65_r, src54_r, src87_r, src76_r, src6554, src8776);
    XORI_B2_128_SB(src6554, src8776);
    out10 = DPADD_SH3_SH(src2110, src4332, src6554, filt0, filt1, filt2);
    out32 = DPADD_SH3_SH(src4332, src6554, src8776, filt0, filt1, filt2);
    SRARI_H2_SH(out10, out32, VP8_FILTER_SHIFT);
    SAT_SH2_SH(out10, out32, 7);
    out = PCKEV_XORI128_UB(out10, out32);
    ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
    dst += (4 * dst_stride);

    src2110 = src6554;
    src4332 = src8776;
    src4 = src8;
  }
}

static void common_vt_6t_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src7, src8, src9, src10;
  v16i8 src10_r, src32_r, src76_r, src98_r, src21_r, src43_r, src87_r;
  v16i8 src109_r, filt0, filt1, filt2;
  v16u8 tmp0, tmp1;
  v8i16 filt, out0_r, out1_r, out2_r, out3_r;

  src -= (2 * src_stride);

  filt = LD_SH(filter);
  SPLATI_H3_SB(filt, 0, 1, 2, filt0, filt1, filt2);

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  src += (5 * src_stride);

  XORI_B5_128_SB(src0, src1, src2, src3, src4);
  ILVR_B4_SB(src1, src0, src3, src2, src2, src1, src4, src3, src10_r, src32_r,
             src21_r, src43_r);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src7, src8, src9, src10);
    XORI_B4_128_SB(src7, src8, src9, src10);
    src += (4 * src_stride);

    ILVR_B4_SB(src7, src4, src8, src7, src9, src8, src10, src9, src76_r,
               src87_r, src98_r, src109_r);
    out0_r = DPADD_SH3_SH(src10_r, src32_r, src76_r, filt0, filt1, filt2);
    out1_r = DPADD_SH3_SH(src21_r, src43_r, src87_r, filt0, filt1, filt2);
    out2_r = DPADD_SH3_SH(src32_r, src76_r, src98_r, filt0, filt1, filt2);
    out3_r = DPADD_SH3_SH(src43_r, src87_r, src109_r, filt0, filt1, filt2);
    SRARI_H4_SH(out0_r, out1_r, out2_r, out3_r, VP8_FILTER_SHIFT);
    SAT_SH4_SH(out0_r, out1_r, out2_r, out3_r, 7);
    tmp0 = PCKEV_XORI128_UB(out0_r, out1_r);
    tmp1 = PCKEV_XORI128_UB(out2_r, out3_r);
    ST8x4_UB(tmp0, tmp1, dst, dst_stride);
    dst += (4 * dst_stride);

    src10_r = src76_r;
    src32_r = src98_r;
    src21_r = src87_r;
    src43_r = src109_r;
    src4 = src10;
  }
}

static void common_vt_6t_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;
  v16i8 src10_r, src32_r, src54_r, src76_r, src21_r, src43_r, src65_r;
  v16i8 src87_r, src10_l, src32_l, src54_l, src76_l, src21_l, src43_l;
  v16i8 src65_l, src87_l, filt0, filt1, filt2;
  v16u8 tmp0, tmp1, tmp2, tmp3;
  v8i16 out0_r, out1_r, out2_r, out3_r, out0_l, out1_l, out2_l, out3_l, filt;

  src -= (2 * src_stride);

  filt = LD_SH(filter);
  SPLATI_H3_SB(filt, 0, 1, 2, filt0, filt1, filt2);

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  src += (5 * src_stride);

  XORI_B5_128_SB(src0, src1, src2, src3, src4);
  ILVR_B4_SB(src1, src0, src3, src2, src4, src3, src2, src1, src10_r, src32_r,
             src43_r, src21_r);
  ILVL_B4_SB(src1, src0, src3, src2, src4, src3, src2, src1, src10_l, src32_l,
             src43_l, src21_l);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src5, src6, src7, src8);
    src += (4 * src_stride);

    XORI_B4_128_SB(src5, src6, src7, src8);
    ILVR_B4_SB(src5, src4, src6, src5, src7, src6, src8, src7, src54_r, src65_r,
               src76_r, src87_r);
    ILVL_B4_SB(src5, src4, src6, src5, src7, src6, src8, src7, src54_l, src65_l,
               src76_l, src87_l);
    out0_r = DPADD_SH3_SH(src10_r, src32_r, src54_r, filt0, filt1, filt2);
    out1_r = DPADD_SH3_SH(src21_r, src43_r, src65_r, filt0, filt1, filt2);
    out2_r = DPADD_SH3_SH(src32_r, src54_r, src76_r, filt0, filt1, filt2);
    out3_r = DPADD_SH3_SH(src43_r, src65_r, src87_r, filt0, filt1, filt2);
    out0_l = DPADD_SH3_SH(src10_l, src32_l, src54_l, filt0, filt1, filt2);
    out1_l = DPADD_SH3_SH(src21_l, src43_l, src65_l, filt0, filt1, filt2);
    out2_l = DPADD_SH3_SH(src32_l, src54_l, src76_l, filt0, filt1, filt2);
    out3_l = DPADD_SH3_SH(src43_l, src65_l, src87_l, filt0, filt1, filt2);
    SRARI_H4_SH(out0_r, out1_r, out2_r, out3_r, VP8_FILTER_SHIFT);
    SRARI_H4_SH(out0_l, out1_l, out2_l, out3_l, VP8_FILTER_SHIFT);
    SAT_SH4_SH(out0_r, out1_r, out2_r, out3_r, 7);
    SAT_SH4_SH(out0_l, out1_l, out2_l, out3_l, 7);
    PCKEV_B4_UB(out0_l, out0_r, out1_l, out1_r, out2_l, out2_r, out3_l, out3_r,
                tmp0, tmp1, tmp2, tmp3);
    XORI_B4_128_UB(tmp0, tmp1, tmp2, tmp3);
    ST_UB4(tmp0, tmp1, tmp2, tmp3, dst, dst_stride);
    dst += (4 * dst_stride);

    src10_r = src54_r;
    src32_r = src76_r;
    src21_r = src65_r;
    src43_r = src87_r;
    src10_l = src54_l;
    src32_l = src76_l;
    src21_l = src65_l;
    src43_l = src87_l;
    src4 = src8;
  }
}

static void common_hv_6ht_6vt_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;
  v16i8 filt_hz0, filt_hz1, filt_hz2;
  v16u8 mask0, mask1, mask2, out;
  v8i16 tmp0, tmp1;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8i16 hz_out7, filt, filt_vt0, filt_vt1, filt_vt2, out0, out1, out2, out3;

  mask0 = LD_UB(&vp8_mc_filt_mask_arr[16]);
  src -= (2 + 2 * src_stride);

  filt = LD_SH(filter_horiz);
  SPLATI_H3_SB(filt, 0, 1, 2, filt_hz0, filt_hz1, filt_hz2);
  filt = LD_SH(filter_vert);
  SPLATI_H3_SH(filt, 0, 1, 2, filt_vt0, filt_vt1, filt_vt2);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  src += (5 * src_stride);

  XORI_B5_128_SB(src0, src1, src2, src3, src4);
  hz_out0 = HORIZ_6TAP_FILT(src0, src1, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out2 = HORIZ_6TAP_FILT(src2, src3, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out1 = (v8i16)__msa_sldi_b((v16i8)hz_out2, (v16i8)hz_out0, 8);
  hz_out3 = HORIZ_6TAP_FILT(src3, src4, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  ILVEV_B2_SH(hz_out0, hz_out1, hz_out2, hz_out3, out0, out1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB2(src, src_stride, src5, src6);
    src += (2 * src_stride);

    XORI_B2_128_SB(src5, src6);
    hz_out5 = HORIZ_6TAP_FILT(src5, src6, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    hz_out4 = (v8i16)__msa_sldi_b((v16i8)hz_out5, (v16i8)hz_out3, 8);

    LD_SB2(src, src_stride, src7, src8);
    src += (2 * src_stride);

    XORI_B2_128_SB(src7, src8);
    hz_out7 = HORIZ_6TAP_FILT(src7, src8, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    hz_out6 = (v8i16)__msa_sldi_b((v16i8)hz_out7, (v16i8)hz_out5, 8);

    out2 = (v8i16)__msa_ilvev_b((v16i8)hz_out5, (v16i8)hz_out4);
    tmp0 = DPADD_SH3_SH(out0, out1, out2, filt_vt0, filt_vt1, filt_vt2);

    out3 = (v8i16)__msa_ilvev_b((v16i8)hz_out7, (v16i8)hz_out6);
    tmp1 = DPADD_SH3_SH(out1, out2, out3, filt_vt0, filt_vt1, filt_vt2);

    SRARI_H2_SH(tmp0, tmp1, 7);
    SAT_SH2_SH(tmp0, tmp1, 7);
    out = PCKEV_XORI128_UB(tmp0, tmp1);
    ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out3 = hz_out7;
    out0 = out2;
    out1 = out3;
  }
}

static void common_hv_6ht_6vt_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;
  v16i8 filt_hz0, filt_hz1, filt_hz2;
  v16u8 mask0, mask1, mask2, vec0, vec1;
  v8i16 filt, filt_vt0, filt_vt1, filt_vt2;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8i16 hz_out7, hz_out8, out0, out1, out2, out3, out4, out5, out6, out7;
  v8i16 tmp0, tmp1, tmp2, tmp3;

  mask0 = LD_UB(&vp8_mc_filt_mask_arr[0]);
  src -= (2 + 2 * src_stride);

  filt = LD_SH(filter_horiz);
  SPLATI_H3_SB(filt, 0, 1, 2, filt_hz0, filt_hz1, filt_hz2);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  src += (5 * src_stride);

  XORI_B5_128_SB(src0, src1, src2, src3, src4);
  hz_out0 = HORIZ_6TAP_FILT(src0, src0, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out1 = HORIZ_6TAP_FILT(src1, src1, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out2 = HORIZ_6TAP_FILT(src2, src2, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out3 = HORIZ_6TAP_FILT(src3, src3, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out4 = HORIZ_6TAP_FILT(src4, src4, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);

  filt = LD_SH(filter_vert);
  SPLATI_H3_SH(filt, 0, 1, 2, filt_vt0, filt_vt1, filt_vt2);

  ILVEV_B2_SH(hz_out0, hz_out1, hz_out2, hz_out3, out0, out1);
  ILVEV_B2_SH(hz_out1, hz_out2, hz_out3, hz_out4, out3, out4);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src5, src6, src7, src8);
    src += (4 * src_stride);

    XORI_B4_128_SB(src5, src6, src7, src8);
    hz_out5 = HORIZ_6TAP_FILT(src5, src5, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    out2 = (v8i16)__msa_ilvev_b((v16i8)hz_out5, (v16i8)hz_out4);
    tmp0 = DPADD_SH3_SH(out0, out1, out2, filt_vt0, filt_vt1, filt_vt2);

    hz_out6 = HORIZ_6TAP_FILT(src6, src6, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    out5 = (v8i16)__msa_ilvev_b((v16i8)hz_out6, (v16i8)hz_out5);
    tmp1 = DPADD_SH3_SH(out3, out4, out5, filt_vt0, filt_vt1, filt_vt2);

    hz_out7 = HORIZ_6TAP_FILT(src7, src7, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    out7 = (v8i16)__msa_ilvev_b((v16i8)hz_out7, (v16i8)hz_out6);
    tmp2 = DPADD_SH3_SH(out1, out2, out7, filt_vt0, filt_vt1, filt_vt2);

    hz_out8 = HORIZ_6TAP_FILT(src8, src8, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    out6 = (v8i16)__msa_ilvev_b((v16i8)hz_out8, (v16i8)hz_out7);
    tmp3 = DPADD_SH3_SH(out4, out5, out6, filt_vt0, filt_vt1, filt_vt2);

    SRARI_H4_SH(tmp0, tmp1, tmp2, tmp3, 7);
    SAT_SH4_SH(tmp0, tmp1, tmp2, tmp3, 7);
    vec0 = PCKEV_XORI128_UB(tmp0, tmp1);
    vec1 = PCKEV_XORI128_UB(tmp2, tmp3);
    ST8x4_UB(vec0, vec1, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out4 = hz_out8;
    out0 = out2;
    out1 = out7;
    out3 = out5;
    out4 = out6;
  }
}

static void common_hv_6ht_6vt_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert,
                                      int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 2; multiple8_cnt--;) {
    common_hv_6ht_6vt_8w_msa(src, src_stride, dst, dst_stride, filter_horiz,
                             filter_vert, height);
    src += 8;
    dst += 8;
  }
}

static void common_hz_4t_4x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  v16i8 src0, src1, src2, src3, filt0, filt1, mask0, mask1;
  v8i16 filt, out0, out1;
  v16u8 out;

  mask0 = LD_SB(&vp8_mc_filt_mask_arr[16]);
  src -= 1;

  filt = LD_SH(filter);
  SPLATI_H2_SB(filt, 0, 1, filt0, filt1);

  mask1 = mask0 + 2;

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  XORI_B4_128_SB(src0, src1, src2, src3);
  HORIZ_4TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0, filt1,
                             out0, out1);
  SRARI_H2_SH(out0, out1, VP8_FILTER_SHIFT);
  SAT_SH2_SH(out0, out1, 7);
  out = PCKEV_XORI128_UB(out0, out1);
  ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
}

static void common_hz_4t_4x8_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  v16i8 src0, src1, src2, src3, filt0, filt1, mask0, mask1;
  v16u8 out;
  v8i16 filt, out0, out1, out2, out3;

  mask0 = LD_SB(&vp8_mc_filt_mask_arr[16]);
  src -= 1;

  filt = LD_SH(filter);
  SPLATI_H2_SB(filt, 0, 1, filt0, filt1);

  mask1 = mask0 + 2;

  LD_SB4(src, src_stride, src0, src1, src2, src3);
  src += (4 * src_stride);

  XORI_B4_128_SB(src0, src1, src2, src3);
  HORIZ_4TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0, filt1,
                             out0, out1);
  LD_SB4(src, src_stride, src0, src1, src2, src3);
  XORI_B4_128_SB(src0, src1, src2, src3);
  HORIZ_4TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0, filt1,
                             out2, out3);
  SRARI_H4_SH(out0, out1, out2, out3, VP8_FILTER_SHIFT);
  SAT_SH4_SH(out0, out1, out2, out3, 7);
  out = PCKEV_XORI128_UB(out0, out1);
  ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
  dst += (4 * dst_stride);
  out = PCKEV_XORI128_UB(out2, out3);
  ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
}

static void common_hz_4t_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  if (4 == height) {
    common_hz_4t_4x4_msa(src, src_stride, dst, dst_stride, filter);
  } else if (8 == height) {
    common_hz_4t_4x8_msa(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_hz_4t_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, filt0, filt1, mask0, mask1;
  v16u8 tmp0, tmp1;
  v8i16 filt, out0, out1, out2, out3;

  mask0 = LD_SB(&vp8_mc_filt_mask_arr[0]);
  src -= 1;

  filt = LD_SH(filter);
  SPLATI_H2_SB(filt, 0, 1, filt0, filt1);

  mask1 = mask0 + 2;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src1, src2, src3);
    src += (4 * src_stride);

    XORI_B4_128_SB(src0, src1, src2, src3);
    HORIZ_4TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0,
                               filt1, out0, out1, out2, out3);
    SRARI_H4_SH(out0, out1, out2, out3, VP8_FILTER_SHIFT);
    SAT_SH4_SH(out0, out1, out2, out3, 7);
    tmp0 = PCKEV_XORI128_UB(out0, out1);
    tmp1 = PCKEV_XORI128_UB(out2, out3);
    ST8x4_UB(tmp0, tmp1, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void common_hz_4t_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16i8 filt0, filt1, mask0, mask1;
  v8i16 filt, out0, out1, out2, out3, out4, out5, out6, out7;
  v16u8 out;

  mask0 = LD_SB(&vp8_mc_filt_mask_arr[0]);
  src -= 1;

  filt = LD_SH(filter);
  SPLATI_H2_SB(filt, 0, 1, filt0, filt1);

  mask1 = mask0 + 2;

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src0, src2, src4, src6);
    LD_SB4(src + 8, src_stride, src1, src3, src5, src7);
    src += (4 * src_stride);

    XORI_B8_128_SB(src0, src1, src2, src3, src4, src5, src6, src7);
    HORIZ_4TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0,
                               filt1, out0, out1, out2, out3);
    HORIZ_4TAP_8WID_4VECS_FILT(src4, src5, src6, src7, mask0, mask1, filt0,
                               filt1, out4, out5, out6, out7);
    SRARI_H4_SH(out0, out1, out2, out3, VP8_FILTER_SHIFT);
    SRARI_H4_SH(out4, out5, out6, out7, VP8_FILTER_SHIFT);
    SAT_SH4_SH(out0, out1, out2, out3, 7);
    SAT_SH4_SH(out4, out5, out6, out7, 7);
    out = PCKEV_XORI128_UB(out0, out1);
    ST_UB(out, dst);
    dst += dst_stride;
    out = PCKEV_XORI128_UB(out2, out3);
    ST_UB(out, dst);
    dst += dst_stride;
    out = PCKEV_XORI128_UB(out4, out5);
    ST_UB(out, dst);
    dst += dst_stride;
    out = PCKEV_XORI128_UB(out6, out7);
    ST_UB(out, dst);
    dst += dst_stride;
  }
}

static void common_vt_4t_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5;
  v16i8 src10_r, src32_r, src54_r, src21_r, src43_r, src65_r;
  v16i8 src2110, src4332, filt0, filt1;
  v8i16 filt, out10, out32;
  v16u8 out;

  src -= src_stride;

  filt = LD_SH(filter);
  SPLATI_H2_SB(filt, 0, 1, filt0, filt1);

  LD_SB3(src, src_stride, src0, src1, src2);
  src += (3 * src_stride);

  ILVR_B2_SB(src1, src0, src2, src1, src10_r, src21_r);

  src2110 = (v16i8)__msa_ilvr_d((v2i64)src21_r, (v2i64)src10_r);
  src2110 = (v16i8)__msa_xori_b((v16u8)src2110, 128);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB3(src, src_stride, src3, src4, src5);
    src += (3 * src_stride);
    ILVR_B2_SB(src3, src2, src4, src3, src32_r, src43_r);
    src4332 = (v16i8)__msa_ilvr_d((v2i64)src43_r, (v2i64)src32_r);
    src4332 = (v16i8)__msa_xori_b((v16u8)src4332, 128);
    out10 = FILT_4TAP_DPADD_S_H(src2110, src4332, filt0, filt1);

    src2 = LD_SB(src);
    src += (src_stride);
    ILVR_B2_SB(src5, src4, src2, src5, src54_r, src65_r);
    src2110 = (v16i8)__msa_ilvr_d((v2i64)src65_r, (v2i64)src54_r);
    src2110 = (v16i8)__msa_xori_b((v16u8)src2110, 128);
    out32 = FILT_4TAP_DPADD_S_H(src4332, src2110, filt0, filt1);
    SRARI_H2_SH(out10, out32, VP8_FILTER_SHIFT);
    SAT_SH2_SH(out10, out32, 7);
    out = PCKEV_XORI128_UB(out10, out32);
    ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void common_vt_4t_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src7, src8, src9, src10;
  v16i8 src10_r, src72_r, src98_r, src21_r, src87_r, src109_r, filt0, filt1;
  v16u8 tmp0, tmp1;
  v8i16 filt, out0_r, out1_r, out2_r, out3_r;

  src -= src_stride;

  filt = LD_SH(filter);
  SPLATI_H2_SB(filt, 0, 1, filt0, filt1);

  LD_SB3(src, src_stride, src0, src1, src2);
  src += (3 * src_stride);

  XORI_B3_128_SB(src0, src1, src2);
  ILVR_B2_SB(src1, src0, src2, src1, src10_r, src21_r);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src7, src8, src9, src10);
    src += (4 * src_stride);

    XORI_B4_128_SB(src7, src8, src9, src10);
    ILVR_B4_SB(src7, src2, src8, src7, src9, src8, src10, src9, src72_r,
               src87_r, src98_r, src109_r);
    out0_r = FILT_4TAP_DPADD_S_H(src10_r, src72_r, filt0, filt1);
    out1_r = FILT_4TAP_DPADD_S_H(src21_r, src87_r, filt0, filt1);
    out2_r = FILT_4TAP_DPADD_S_H(src72_r, src98_r, filt0, filt1);
    out3_r = FILT_4TAP_DPADD_S_H(src87_r, src109_r, filt0, filt1);
    SRARI_H4_SH(out0_r, out1_r, out2_r, out3_r, VP8_FILTER_SHIFT);
    SAT_SH4_SH(out0_r, out1_r, out2_r, out3_r, 7);
    tmp0 = PCKEV_XORI128_UB(out0_r, out1_r);
    tmp1 = PCKEV_XORI128_UB(out2_r, out3_r);
    ST8x4_UB(tmp0, tmp1, dst, dst_stride);
    dst += (4 * dst_stride);

    src10_r = src98_r;
    src21_r = src109_r;
    src2 = src10;
  }
}

static void common_vt_4t_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6;
  v16i8 src10_r, src32_r, src54_r, src21_r, src43_r, src65_r, src10_l;
  v16i8 src32_l, src54_l, src21_l, src43_l, src65_l, filt0, filt1;
  v16u8 tmp0, tmp1, tmp2, tmp3;
  v8i16 filt, out0_r, out1_r, out2_r, out3_r, out0_l, out1_l, out2_l, out3_l;

  src -= src_stride;

  filt = LD_SH(filter);
  SPLATI_H2_SB(filt, 0, 1, filt0, filt1);

  LD_SB3(src, src_stride, src0, src1, src2);
  src += (3 * src_stride);

  XORI_B3_128_SB(src0, src1, src2);
  ILVR_B2_SB(src1, src0, src2, src1, src10_r, src21_r);
  ILVL_B2_SB(src1, src0, src2, src1, src10_l, src21_l);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src3, src4, src5, src6);
    src += (4 * src_stride);

    XORI_B4_128_SB(src3, src4, src5, src6);
    ILVR_B4_SB(src3, src2, src4, src3, src5, src4, src6, src5, src32_r, src43_r,
               src54_r, src65_r);
    ILVL_B4_SB(src3, src2, src4, src3, src5, src4, src6, src5, src32_l, src43_l,
               src54_l, src65_l);
    out0_r = FILT_4TAP_DPADD_S_H(src10_r, src32_r, filt0, filt1);
    out1_r = FILT_4TAP_DPADD_S_H(src21_r, src43_r, filt0, filt1);
    out2_r = FILT_4TAP_DPADD_S_H(src32_r, src54_r, filt0, filt1);
    out3_r = FILT_4TAP_DPADD_S_H(src43_r, src65_r, filt0, filt1);
    out0_l = FILT_4TAP_DPADD_S_H(src10_l, src32_l, filt0, filt1);
    out1_l = FILT_4TAP_DPADD_S_H(src21_l, src43_l, filt0, filt1);
    out2_l = FILT_4TAP_DPADD_S_H(src32_l, src54_l, filt0, filt1);
    out3_l = FILT_4TAP_DPADD_S_H(src43_l, src65_l, filt0, filt1);
    SRARI_H4_SH(out0_r, out1_r, out2_r, out3_r, VP8_FILTER_SHIFT);
    SRARI_H4_SH(out0_l, out1_l, out2_l, out3_l, VP8_FILTER_SHIFT);
    SAT_SH4_SH(out0_r, out1_r, out2_r, out3_r, 7);
    SAT_SH4_SH(out0_l, out1_l, out2_l, out3_l, 7);
    PCKEV_B4_UB(out0_l, out0_r, out1_l, out1_r, out2_l, out2_r, out3_l, out3_r,
                tmp0, tmp1, tmp2, tmp3);
    XORI_B4_128_UB(tmp0, tmp1, tmp2, tmp3);
    ST_UB4(tmp0, tmp1, tmp2, tmp3, dst, dst_stride);
    dst += (4 * dst_stride);

    src10_r = src54_r;
    src21_r = src65_r;
    src10_l = src54_l;
    src21_l = src65_l;
    src2 = src6;
  }
}

static void common_hv_4ht_4vt_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, filt_hz0, filt_hz1;
  v16u8 mask0, mask1, out;
  v8i16 filt, filt_vt0, filt_vt1, tmp0, tmp1, vec0, vec1, vec2;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5;

  mask0 = LD_UB(&vp8_mc_filt_mask_arr[16]);
  src -= (1 + 1 * src_stride);

  filt = LD_SH(filter_horiz);
  SPLATI_H2_SB(filt, 0, 1, filt_hz0, filt_hz1);

  mask1 = mask0 + 2;

  LD_SB3(src, src_stride, src0, src1, src2);
  src += (3 * src_stride);

  XORI_B3_128_SB(src0, src1, src2);
  hz_out0 = HORIZ_4TAP_FILT(src0, src1, mask0, mask1, filt_hz0, filt_hz1);
  hz_out1 = HORIZ_4TAP_FILT(src1, src2, mask0, mask1, filt_hz0, filt_hz1);
  vec0 = (v8i16)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);

  filt = LD_SH(filter_vert);
  SPLATI_H2_SH(filt, 0, 1, filt_vt0, filt_vt1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src3, src4, src5, src6);
    src += (4 * src_stride);

    XORI_B2_128_SB(src3, src4);
    hz_out3 = HORIZ_4TAP_FILT(src3, src4, mask0, mask1, filt_hz0, filt_hz1);
    hz_out2 = (v8i16)__msa_sldi_b((v16i8)hz_out3, (v16i8)hz_out1, 8);
    vec1 = (v8i16)__msa_ilvev_b((v16i8)hz_out3, (v16i8)hz_out2);
    tmp0 = FILT_4TAP_DPADD_S_H(vec0, vec1, filt_vt0, filt_vt1);

    XORI_B2_128_SB(src5, src6);
    hz_out5 = HORIZ_4TAP_FILT(src5, src6, mask0, mask1, filt_hz0, filt_hz1);
    hz_out4 = (v8i16)__msa_sldi_b((v16i8)hz_out5, (v16i8)hz_out3, 8);
    vec2 = (v8i16)__msa_ilvev_b((v16i8)hz_out5, (v16i8)hz_out4);
    tmp1 = FILT_4TAP_DPADD_S_H(vec1, vec2, filt_vt0, filt_vt1);

    SRARI_H2_SH(tmp0, tmp1, 7);
    SAT_SH2_SH(tmp0, tmp1, 7);
    out = PCKEV_XORI128_UB(tmp0, tmp1);
    ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out1 = hz_out5;
    vec0 = vec2;
  }
}

static void common_hv_4ht_4vt_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, filt_hz0, filt_hz1;
  v16u8 mask0, mask1, out0, out1;
  v8i16 filt, filt_vt0, filt_vt1, tmp0, tmp1, tmp2, tmp3;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3;
  v8i16 vec0, vec1, vec2, vec3, vec4;

  mask0 = LD_UB(&vp8_mc_filt_mask_arr[0]);
  src -= (1 + 1 * src_stride);

  filt = LD_SH(filter_horiz);
  SPLATI_H2_SB(filt, 0, 1, filt_hz0, filt_hz1);

  mask1 = mask0 + 2;

  LD_SB3(src, src_stride, src0, src1, src2);
  src += (3 * src_stride);

  XORI_B3_128_SB(src0, src1, src2);
  hz_out0 = HORIZ_4TAP_FILT(src0, src0, mask0, mask1, filt_hz0, filt_hz1);
  hz_out1 = HORIZ_4TAP_FILT(src1, src1, mask0, mask1, filt_hz0, filt_hz1);
  hz_out2 = HORIZ_4TAP_FILT(src2, src2, mask0, mask1, filt_hz0, filt_hz1);
  ILVEV_B2_SH(hz_out0, hz_out1, hz_out1, hz_out2, vec0, vec2);

  filt = LD_SH(filter_vert);
  SPLATI_H2_SH(filt, 0, 1, filt_vt0, filt_vt1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src3, src4, src5, src6);
    src += (4 * src_stride);

    XORI_B4_128_SB(src3, src4, src5, src6);
    hz_out3 = HORIZ_4TAP_FILT(src3, src3, mask0, mask1, filt_hz0, filt_hz1);
    vec1 = (v8i16)__msa_ilvev_b((v16i8)hz_out3, (v16i8)hz_out2);
    tmp0 = FILT_4TAP_DPADD_S_H(vec0, vec1, filt_vt0, filt_vt1);

    hz_out0 = HORIZ_4TAP_FILT(src4, src4, mask0, mask1, filt_hz0, filt_hz1);
    vec3 = (v8i16)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out3);
    tmp1 = FILT_4TAP_DPADD_S_H(vec2, vec3, filt_vt0, filt_vt1);

    hz_out1 = HORIZ_4TAP_FILT(src5, src5, mask0, mask1, filt_hz0, filt_hz1);
    vec4 = (v8i16)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp2 = FILT_4TAP_DPADD_S_H(vec1, vec4, filt_vt0, filt_vt1);

    hz_out2 = HORIZ_4TAP_FILT(src6, src6, mask0, mask1, filt_hz0, filt_hz1);
    ILVEV_B2_SH(hz_out3, hz_out0, hz_out1, hz_out2, vec0, vec1);
    tmp3 = FILT_4TAP_DPADD_S_H(vec0, vec1, filt_vt0, filt_vt1);

    SRARI_H4_SH(tmp0, tmp1, tmp2, tmp3, 7);
    SAT_SH4_SH(tmp0, tmp1, tmp2, tmp3, 7);
    out0 = PCKEV_XORI128_UB(tmp0, tmp1);
    out1 = PCKEV_XORI128_UB(tmp2, tmp3);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);

    vec0 = vec4;
    vec2 = vec1;
  }
}

static void common_hv_4ht_4vt_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert,
                                      int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 2; multiple8_cnt--;) {
    common_hv_4ht_4vt_8w_msa(src, src_stride, dst, dst_stride, filter_horiz,
                             filter_vert, height);
    src += 8;
    dst += 8;
  }
}

static void common_hv_6ht_4vt_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6;
  v16i8 filt_hz0, filt_hz1, filt_hz2;
  v16u8 res0, res1, mask0, mask1, mask2;
  v8i16 filt, filt_vt0, filt_vt1, tmp0, tmp1, vec0, vec1, vec2;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5;

  mask0 = LD_UB(&vp8_mc_filt_mask_arr[16]);
  src -= (2 + 1 * src_stride);

  filt = LD_SH(filter_horiz);
  SPLATI_H3_SB(filt, 0, 1, 2, filt_hz0, filt_hz1, filt_hz2);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;

  LD_SB3(src, src_stride, src0, src1, src2);
  src += (3 * src_stride);

  XORI_B3_128_SB(src0, src1, src2);
  hz_out0 = HORIZ_6TAP_FILT(src0, src1, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out1 = HORIZ_6TAP_FILT(src1, src2, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  vec0 = (v8i16)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);

  filt = LD_SH(filter_vert);
  SPLATI_H2_SH(filt, 0, 1, filt_vt0, filt_vt1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src3, src4, src5, src6);
    src += (4 * src_stride);

    XORI_B4_128_SB(src3, src4, src5, src6);
    hz_out3 = HORIZ_6TAP_FILT(src3, src4, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    hz_out2 = (v8i16)__msa_sldi_b((v16i8)hz_out3, (v16i8)hz_out1, 8);
    vec1 = (v8i16)__msa_ilvev_b((v16i8)hz_out3, (v16i8)hz_out2);
    tmp0 = FILT_4TAP_DPADD_S_H(vec0, vec1, filt_vt0, filt_vt1);

    hz_out5 = HORIZ_6TAP_FILT(src5, src6, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    hz_out4 = (v8i16)__msa_sldi_b((v16i8)hz_out5, (v16i8)hz_out3, 8);
    vec2 = (v8i16)__msa_ilvev_b((v16i8)hz_out5, (v16i8)hz_out4);
    tmp1 = FILT_4TAP_DPADD_S_H(vec1, vec2, filt_vt0, filt_vt1);

    SRARI_H2_SH(tmp0, tmp1, 7);
    SAT_SH2_SH(tmp0, tmp1, 7);
    PCKEV_B2_UB(tmp0, tmp0, tmp1, tmp1, res0, res1);
    XORI_B2_128_UB(res0, res1);
    ST4x4_UB(res0, res1, 0, 1, 0, 1, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out1 = hz_out5;
    vec0 = vec2;
  }
}

static void common_hv_6ht_4vt_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6;
  v16i8 filt_hz0, filt_hz1, filt_hz2, mask0, mask1, mask2;
  v8i16 filt, filt_vt0, filt_vt1, hz_out0, hz_out1, hz_out2, hz_out3;
  v8i16 tmp0, tmp1, tmp2, tmp3, vec0, vec1, vec2, vec3;
  v16u8 out0, out1;

  mask0 = LD_SB(&vp8_mc_filt_mask_arr[0]);
  src -= (2 + src_stride);

  filt = LD_SH(filter_horiz);
  SPLATI_H3_SB(filt, 0, 1, 2, filt_hz0, filt_hz1, filt_hz2);

  mask1 = mask0 + 2;
  mask2 = mask0 + 4;

  LD_SB3(src, src_stride, src0, src1, src2);
  src += (3 * src_stride);

  XORI_B3_128_SB(src0, src1, src2);
  hz_out0 = HORIZ_6TAP_FILT(src0, src0, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out1 = HORIZ_6TAP_FILT(src1, src1, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out2 = HORIZ_6TAP_FILT(src2, src2, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  ILVEV_B2_SH(hz_out0, hz_out1, hz_out1, hz_out2, vec0, vec2);

  filt = LD_SH(filter_vert);
  SPLATI_H2_SH(filt, 0, 1, filt_vt0, filt_vt1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src3, src4, src5, src6);
    src += (4 * src_stride);

    XORI_B4_128_SB(src3, src4, src5, src6);

    hz_out3 = HORIZ_6TAP_FILT(src3, src3, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    vec1 = (v8i16)__msa_ilvev_b((v16i8)hz_out3, (v16i8)hz_out2);
    tmp0 = FILT_4TAP_DPADD_S_H(vec0, vec1, filt_vt0, filt_vt1);

    hz_out0 = HORIZ_6TAP_FILT(src4, src4, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    vec3 = (v8i16)__msa_ilvev_b((v16i8)hz_out0, (v16i8)hz_out3);
    tmp1 = FILT_4TAP_DPADD_S_H(vec2, vec3, filt_vt0, filt_vt1);

    hz_out1 = HORIZ_6TAP_FILT(src5, src5, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    vec0 = (v8i16)__msa_ilvev_b((v16i8)hz_out1, (v16i8)hz_out0);
    tmp2 = FILT_4TAP_DPADD_S_H(vec1, vec0, filt_vt0, filt_vt1);

    hz_out2 = HORIZ_6TAP_FILT(src6, src6, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    ILVEV_B2_SH(hz_out3, hz_out0, hz_out1, hz_out2, vec1, vec2);
    tmp3 = FILT_4TAP_DPADD_S_H(vec1, vec2, filt_vt0, filt_vt1);

    SRARI_H4_SH(tmp0, tmp1, tmp2, tmp3, 7);
    SAT_SH4_SH(tmp0, tmp1, tmp2, tmp3, 7);
    out0 = PCKEV_XORI128_UB(tmp0, tmp1);
    out1 = PCKEV_XORI128_UB(tmp2, tmp3);
    ST8x4_UB(out0, out1, dst, dst_stride);
    dst += (4 * dst_stride);
  }
}

static void common_hv_6ht_4vt_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert,
                                      int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 2; multiple8_cnt--;) {
    common_hv_6ht_4vt_8w_msa(src, src_stride, dst, dst_stride, filter_horiz,
                             filter_vert, height);
    src += 8;
    dst += 8;
  }
}

static void common_hv_4ht_6vt_4w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;
  v16i8 filt_hz0, filt_hz1, mask0, mask1;
  v16u8 out;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8i16 hz_out7, tmp0, tmp1, out0, out1, out2, out3;
  v8i16 filt, filt_vt0, filt_vt1, filt_vt2;

  mask0 = LD_SB(&vp8_mc_filt_mask_arr[16]);

  src -= (1 + 2 * src_stride);

  filt = LD_SH(filter_horiz);
  SPLATI_H2_SB(filt, 0, 1, filt_hz0, filt_hz1);

  mask1 = mask0 + 2;

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  src += (5 * src_stride);

  XORI_B5_128_SB(src0, src1, src2, src3, src4);
  hz_out0 = HORIZ_4TAP_FILT(src0, src1, mask0, mask1, filt_hz0, filt_hz1);
  hz_out2 = HORIZ_4TAP_FILT(src2, src3, mask0, mask1, filt_hz0, filt_hz1);
  hz_out3 = HORIZ_4TAP_FILT(src3, src4, mask0, mask1, filt_hz0, filt_hz1);
  hz_out1 = (v8i16)__msa_sldi_b((v16i8)hz_out2, (v16i8)hz_out0, 8);
  ILVEV_B2_SH(hz_out0, hz_out1, hz_out2, hz_out3, out0, out1);

  filt = LD_SH(filter_vert);
  SPLATI_H3_SH(filt, 0, 1, 2, filt_vt0, filt_vt1, filt_vt2);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src5, src6, src7, src8);
    XORI_B4_128_SB(src5, src6, src7, src8);
    src += (4 * src_stride);

    hz_out5 = HORIZ_4TAP_FILT(src5, src6, mask0, mask1, filt_hz0, filt_hz1);
    hz_out4 = (v8i16)__msa_sldi_b((v16i8)hz_out5, (v16i8)hz_out3, 8);
    out2 = (v8i16)__msa_ilvev_b((v16i8)hz_out5, (v16i8)hz_out4);
    tmp0 = DPADD_SH3_SH(out0, out1, out2, filt_vt0, filt_vt1, filt_vt2);

    hz_out7 = HORIZ_4TAP_FILT(src7, src8, mask0, mask1, filt_hz0, filt_hz1);
    hz_out6 = (v8i16)__msa_sldi_b((v16i8)hz_out7, (v16i8)hz_out5, 8);
    out3 = (v8i16)__msa_ilvev_b((v16i8)hz_out7, (v16i8)hz_out6);
    tmp1 = DPADD_SH3_SH(out1, out2, out3, filt_vt0, filt_vt1, filt_vt2);

    SRARI_H2_SH(tmp0, tmp1, 7);
    SAT_SH2_SH(tmp0, tmp1, 7);
    out = PCKEV_XORI128_UB(tmp0, tmp1);
    ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out3 = hz_out7;
    out0 = out2;
    out1 = out3;
  }
}

static void common_hv_4ht_6vt_8w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;
  v16i8 filt_hz0, filt_hz1, mask0, mask1;
  v8i16 filt, filt_vt0, filt_vt1, filt_vt2, tmp0, tmp1, tmp2, tmp3;
  v8i16 hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  v8i16 hz_out7, hz_out8, out0, out1, out2, out3, out4, out5, out6, out7;
  v16u8 vec0, vec1;

  mask0 = LD_SB(&vp8_mc_filt_mask_arr[0]);
  src -= (1 + 2 * src_stride);

  filt = LD_SH(filter_horiz);
  SPLATI_H2_SB(filt, 0, 1, filt_hz0, filt_hz1);

  mask1 = mask0 + 2;

  LD_SB5(src, src_stride, src0, src1, src2, src3, src4);
  src += (5 * src_stride);

  XORI_B5_128_SB(src0, src1, src2, src3, src4);
  hz_out0 = HORIZ_4TAP_FILT(src0, src0, mask0, mask1, filt_hz0, filt_hz1);
  hz_out1 = HORIZ_4TAP_FILT(src1, src1, mask0, mask1, filt_hz0, filt_hz1);
  hz_out2 = HORIZ_4TAP_FILT(src2, src2, mask0, mask1, filt_hz0, filt_hz1);
  hz_out3 = HORIZ_4TAP_FILT(src3, src3, mask0, mask1, filt_hz0, filt_hz1);
  hz_out4 = HORIZ_4TAP_FILT(src4, src4, mask0, mask1, filt_hz0, filt_hz1);
  ILVEV_B2_SH(hz_out0, hz_out1, hz_out2, hz_out3, out0, out1);
  ILVEV_B2_SH(hz_out1, hz_out2, hz_out3, hz_out4, out3, out4);

  filt = LD_SH(filter_vert);
  SPLATI_H3_SH(filt, 0, 1, 2, filt_vt0, filt_vt1, filt_vt2);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    LD_SB4(src, src_stride, src5, src6, src7, src8);
    src += (4 * src_stride);

    XORI_B4_128_SB(src5, src6, src7, src8);

    hz_out5 = HORIZ_4TAP_FILT(src5, src5, mask0, mask1, filt_hz0, filt_hz1);
    out2 = (v8i16)__msa_ilvev_b((v16i8)hz_out5, (v16i8)hz_out4);
    tmp0 = DPADD_SH3_SH(out0, out1, out2, filt_vt0, filt_vt1, filt_vt2);

    hz_out6 = HORIZ_4TAP_FILT(src6, src6, mask0, mask1, filt_hz0, filt_hz1);
    out5 = (v8i16)__msa_ilvev_b((v16i8)hz_out6, (v16i8)hz_out5);
    tmp1 = DPADD_SH3_SH(out3, out4, out5, filt_vt0, filt_vt1, filt_vt2);

    hz_out7 = HORIZ_4TAP_FILT(src7, src7, mask0, mask1, filt_hz0, filt_hz1);
    out6 = (v8i16)__msa_ilvev_b((v16i8)hz_out7, (v16i8)hz_out6);
    tmp2 = DPADD_SH3_SH(out1, out2, out6, filt_vt0, filt_vt1, filt_vt2);

    hz_out8 = HORIZ_4TAP_FILT(src8, src8, mask0, mask1, filt_hz0, filt_hz1);
    out7 = (v8i16)__msa_ilvev_b((v16i8)hz_out8, (v16i8)hz_out7);
    tmp3 = DPADD_SH3_SH(out4, out5, out7, filt_vt0, filt_vt1, filt_vt2);

    SRARI_H4_SH(tmp0, tmp1, tmp2, tmp3, 7);
    SAT_SH4_SH(tmp0, tmp1, tmp2, tmp3, 7);
    vec0 = PCKEV_XORI128_UB(tmp0, tmp1);
    vec1 = PCKEV_XORI128_UB(tmp2, tmp3);
    ST8x4_UB(vec0, vec1, dst, dst_stride);
    dst += (4 * dst_stride);

    hz_out4 = hz_out8;
    out0 = out2;
    out1 = out6;
    out3 = out5;
    out4 = out7;
  }
}

static void common_hv_4ht_6vt_16w_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert,
                                      int32_t height) {
  int32_t multiple8_cnt;
  for (multiple8_cnt = 2; multiple8_cnt--;) {
    common_hv_4ht_6vt_8w_msa(src, src_stride, dst, dst_stride, filter_horiz,
                             filter_vert, height);
    src += 8;
    dst += 8;
  }
}

void vp8_sixtap_predict4x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                               int32_t xoffset, int32_t yoffset,
                               uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_subpel_filters_msa[xoffset - 1];
  const int8_t *v_filter = vp8_subpel_filters_msa[yoffset - 1];

  if (yoffset) {
    if (xoffset) {
      switch (xoffset) {
        case 2:
        case 4:
        case 6:
          switch (yoffset) {
            case 2:
            case 4:
            case 6:
              common_hv_6ht_6vt_4w_msa(src, src_stride, dst, dst_stride,
                                       h_filter, v_filter, 4);
              break;

            case 1:
            case 3:
            case 5:
            case 7:
              common_hv_6ht_4vt_4w_msa(src, src_stride, dst, dst_stride,
                                       h_filter, v_filter + 1, 4);
              break;
          }
          break;

        case 1:
        case 3:
        case 5:
        case 7:
          switch (yoffset) {
            case 2:
            case 4:
            case 6:
              common_hv_4ht_6vt_4w_msa(src, src_stride, dst, dst_stride,
                                       h_filter + 1, v_filter, 4);
              break;

            case 1:
            case 3:
            case 5:
            case 7:
              common_hv_4ht_4vt_4w_msa(src, src_stride, dst, dst_stride,
                                       h_filter + 1, v_filter + 1, 4);
              break;
          }
          break;
      }
    } else {
      switch (yoffset) {
        case 2:
        case 4:
        case 6:
          common_vt_6t_4w_msa(src, src_stride, dst, dst_stride, v_filter, 4);
          break;

        case 1:
        case 3:
        case 5:
        case 7:
          common_vt_4t_4w_msa(src, src_stride, dst, dst_stride, v_filter + 1,
                              4);
          break;
      }
    }
  } else {
    switch (xoffset) {
      case 0: {
        uint32_t tp0, tp1, tp2, tp3;

        LW4(src, src_stride, tp0, tp1, tp2, tp3);
        SW4(tp0, tp1, tp2, tp3, dst, dst_stride);
        break;
      }
      case 2:
      case 4:
      case 6:
        common_hz_6t_4w_msa(src, src_stride, dst, dst_stride, h_filter, 4);
        break;

      case 1:
      case 3:
      case 5:
      case 7:
        common_hz_4t_4w_msa(src, src_stride, dst, dst_stride, h_filter + 1, 4);
        break;
    }
  }
}

void vp8_sixtap_predict8x4_msa(uint8_t *RESTRICT src, int32_t src_stride,
                               int32_t xoffset, int32_t yoffset,
                               uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_subpel_filters_msa[xoffset - 1];
  const int8_t *v_filter = vp8_subpel_filters_msa[yoffset - 1];

  if (yoffset) {
    if (xoffset) {
      switch (xoffset) {
        case 2:
        case 4:
        case 6:
          switch (yoffset) {
            case 2:
            case 4:
            case 6:
              common_hv_6ht_6vt_8w_msa(src, src_stride, dst, dst_stride,
                                       h_filter, v_filter, 4);
              break;

            case 1:
            case 3:
            case 5:
            case 7:
              common_hv_6ht_4vt_8w_msa(src, src_stride, dst, dst_stride,
                                       h_filter, v_filter + 1, 4);
              break;
          }
          break;

        case 1:
        case 3:
        case 5:
        case 7:
          switch (yoffset) {
            case 2:
            case 4:
            case 6:
              common_hv_4ht_6vt_8w_msa(src, src_stride, dst, dst_stride,
                                       h_filter + 1, v_filter, 4);
              break;

            case 1:
            case 3:
            case 5:
            case 7:
              common_hv_4ht_4vt_8w_msa(src, src_stride, dst, dst_stride,
                                       h_filter + 1, v_filter + 1, 4);
              break;
          }
          break;
      }
    } else {
      switch (yoffset) {
        case 2:
        case 4:
        case 6:
          common_vt_6t_8w_msa(src, src_stride, dst, dst_stride, v_filter, 4);
          break;

        case 1:
        case 3:
        case 5:
        case 7:
          common_vt_4t_8w_msa(src, src_stride, dst, dst_stride, v_filter + 1,
                              4);
          break;
      }
    }
  } else {
    switch (xoffset) {
      case 0: vp8_copy_mem8x4(src, src_stride, dst, dst_stride); break;
      case 2:
      case 4:
      case 6:
        common_hz_6t_8w_msa(src, src_stride, dst, dst_stride, h_filter, 4);
        break;

      case 1:
      case 3:
      case 5:
      case 7:
        common_hz_4t_8w_msa(src, src_stride, dst, dst_stride, h_filter + 1, 4);
        break;
    }
  }
}

void vp8_sixtap_predict8x8_msa(uint8_t *RESTRICT src, int32_t src_stride,
                               int32_t xoffset, int32_t yoffset,
                               uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_subpel_filters_msa[xoffset - 1];
  const int8_t *v_filter = vp8_subpel_filters_msa[yoffset - 1];

  if (yoffset) {
    if (xoffset) {
      switch (xoffset) {
        case 2:
        case 4:
        case 6:
          switch (yoffset) {
            case 2:
            case 4:
            case 6:
              common_hv_6ht_6vt_8w_msa(src, src_stride, dst, dst_stride,
                                       h_filter, v_filter, 8);
              break;

            case 1:
            case 3:
            case 5:
            case 7:
              common_hv_6ht_4vt_8w_msa(src, src_stride, dst, dst_stride,
                                       h_filter, v_filter + 1, 8);
              break;
          }
          break;

        case 1:
        case 3:
        case 5:
        case 7:
          switch (yoffset) {
            case 2:
            case 4:
            case 6:
              common_hv_4ht_6vt_8w_msa(src, src_stride, dst, dst_stride,
                                       h_filter + 1, v_filter, 8);
              break;

            case 1:
            case 3:
            case 5:
            case 7:
              common_hv_4ht_4vt_8w_msa(src, src_stride, dst, dst_stride,
                                       h_filter + 1, v_filter + 1, 8);
              break;
          }
          break;
      }
    } else {
      switch (yoffset) {
        case 2:
        case 4:
        case 6:
          common_vt_6t_8w_msa(src, src_stride, dst, dst_stride, v_filter, 8);
          break;

        case 1:
        case 3:
        case 5:
        case 7:
          common_vt_4t_8w_msa(src, src_stride, dst, dst_stride, v_filter + 1,
                              8);
          break;
      }
    }
  } else {
    switch (xoffset) {
      case 0: vp8_copy_mem8x8(src, src_stride, dst, dst_stride); break;
      case 2:
      case 4:
      case 6:
        common_hz_6t_8w_msa(src, src_stride, dst, dst_stride, h_filter, 8);
        break;

      case 1:
      case 3:
      case 5:
      case 7:
        common_hz_4t_8w_msa(src, src_stride, dst, dst_stride, h_filter + 1, 8);
        break;
    }
  }
}

void vp8_sixtap_predict16x16_msa(uint8_t *RESTRICT src, int32_t src_stride,
                                 int32_t xoffset, int32_t yoffset,
                                 uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_subpel_filters_msa[xoffset - 1];
  const int8_t *v_filter = vp8_subpel_filters_msa[yoffset - 1];

  if (yoffset) {
    if (xoffset) {
      switch (xoffset) {
        case 2:
        case 4:
        case 6:
          switch (yoffset) {
            case 2:
            case 4:
            case 6:
              common_hv_6ht_6vt_16w_msa(src, src_stride, dst, dst_stride,
                                        h_filter, v_filter, 16);
              break;

            case 1:
            case 3:
            case 5:
            case 7:
              common_hv_6ht_4vt_16w_msa(src, src_stride, dst, dst_stride,
                                        h_filter, v_filter + 1, 16);
              break;
          }
          break;

        case 1:
        case 3:
        case 5:
        case 7:
          switch (yoffset) {
            case 2:
            case 4:
            case 6:
              common_hv_4ht_6vt_16w_msa(src, src_stride, dst, dst_stride,
                                        h_filter + 1, v_filter, 16);
              break;

            case 1:
            case 3:
            case 5:
            case 7:
              common_hv_4ht_4vt_16w_msa(src, src_stride, dst, dst_stride,
                                        h_filter + 1, v_filter + 1, 16);
              break;
          }
          break;
      }
    } else {
      switch (yoffset) {
        case 2:
        case 4:
        case 6:
          common_vt_6t_16w_msa(src, src_stride, dst, dst_stride, v_filter, 16);
          break;

        case 1:
        case 3:
        case 5:
        case 7:
          common_vt_4t_16w_msa(src, src_stride, dst, dst_stride, v_filter + 1,
                               16);
          break;
      }
    }
  } else {
    switch (xoffset) {
      case 0: vp8_copy_mem16x16(src, src_stride, dst, dst_stride); break;
      case 2:
      case 4:
      case 6:
        common_hz_6t_16w_msa(src, src_stride, dst, dst_stride, h_filter, 16);
        break;

      case 1:
      case 3:
      case 5:
      case 7:
        common_hz_4t_16w_msa(src, src_stride, dst, dst_stride, h_filter + 1,
                             16);
        break;
    }
  }
}
