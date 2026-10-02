/*
 * Copyright (c) 2021 Loongson Technology Corporation Limited
 * Contributed by Lu Wang <wanglu@loongson.cn>
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS.  All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"
#include "vp8/common/filter.h"
#include "vpx_ports/mem.h"
#include "vpx_util/loongson_intrinsics.h"

DECLARE_ALIGNED(16, static const int8_t, vp8_subpel_filters_lsx[7][8]) = {
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

static INLINE __m128i dpadd_h3(__m128i in0, __m128i in1, __m128i in2,
                               __m128i coeff0, __m128i coeff1, __m128i coeff2) {
  __m128i out0_m;

  out0_m = __lsx_vdp2_h_b(in0, coeff0);
  out0_m = __lsx_vdp2add_h_b(out0_m, in1, coeff1);
  out0_m = __lsx_vdp2add_h_b(out0_m, in2, coeff2);

  return out0_m;
}

static INLINE __m128i horiz_6tap_filt(__m128i src0, __m128i src1, __m128i mask0,
                                      __m128i mask1, __m128i mask2,
                                      __m128i filt_h0, __m128i filt_h1,
                                      __m128i filt_h2) {
  __m128i vec0_m, vec1_m, vec2_m;
  __m128i hz_out_m;

  DUP2_ARG3(__lsx_vshuf_b, src1, src0, mask0, src1, src0, mask1, vec0_m,
            vec1_m);
  vec2_m = __lsx_vshuf_b(src1, src0, mask2);
  hz_out_m = dpadd_h3(vec0_m, vec1_m, vec2_m, filt_h0, filt_h1, filt_h2);
  hz_out_m = __lsx_vsrari_h(hz_out_m, VP8_FILTER_SHIFT);
  hz_out_m = __lsx_vsat_h(hz_out_m, 7);

  return hz_out_m;
}

static INLINE __m128i filt_4tap_dpadd_h(__m128i vec0, __m128i vec1,
                                        __m128i filt0, __m128i filt1) {
  __m128i tmp_m;

  tmp_m = __lsx_vdp2_h_b(vec0, filt0);
  tmp_m = __lsx_vdp2add_h_b(tmp_m, vec1, filt1);

  return tmp_m;
}

static INLINE __m128i horiz_4tap_filt(__m128i src0, __m128i src1, __m128i mask0,
                                      __m128i mask1, __m128i filt_h0,
                                      __m128i filt_h1) {
  __m128i vec0_m, vec1_m, hz_out_m;

  DUP2_ARG3(__lsx_vshuf_b, src1, src0, mask0, src1, src0, mask1, vec0_m,
            vec1_m);
  hz_out_m = filt_4tap_dpadd_h(vec0_m, vec1_m, filt_h0, filt_h1);
  hz_out_m = __lsx_vsrari_h(hz_out_m, VP8_FILTER_SHIFT);
  hz_out_m = __lsx_vsat_h(hz_out_m, 7);

  return hz_out_m;
}

#define HORIZ_6TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,   \
                                   mask2, filt0, filt1, filt2, out0, out1) \
  do {                                                                     \
    __m128i vec0_m, vec1_m, vec2_m, vec3_m, vec4_m, vec5_m;                \
                                                                           \
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, mask0, src3, src2, mask0, vec0_m, \
              vec1_m);                                                     \
    DUP2_ARG2(__lsx_vdp2_h_b, vec0_m, filt0, vec1_m, filt0, out0, out1);   \
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, mask1, src3, src2, mask1, vec2_m, \
              vec3_m);                                                     \
    DUP2_ARG3(__lsx_vdp2add_h_b, out0, vec2_m, filt1, out1, vec3_m, filt1, \
              out0, out1);                                                 \
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, mask2, src3, src2, mask2, vec4_m, \
              vec5_m);                                                     \
    DUP2_ARG3(__lsx_vdp2add_h_b, out0, vec4_m, filt2, out1, vec5_m, filt2, \
              out0, out1);                                                 \
  } while (0)

#define HORIZ_6TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,    \
                                   mask2, filt0, filt1, filt2, out0, out1,  \
                                   out2, out3)                              \
  do {                                                                      \
    __m128i vec0_m, vec1_m, vec2_m, vec3_m, vec4_m, vec5_m, vec6_m, vec7_m; \
                                                                            \
    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask0, src1, src1, mask0, vec0_m,  \
              vec1_m);                                                      \
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask0, src3, src3, mask0, vec2_m,  \
              vec3_m);                                                      \
    DUP4_ARG2(__lsx_vdp2_h_b, vec0_m, filt0, vec1_m, filt0, vec2_m, filt0,  \
              vec3_m, filt0, out0, out1, out2, out3);                       \
    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask1, src1, src1, mask1, vec0_m,  \
              vec1_m);                                                      \
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask1, src3, src3, mask1, vec2_m,  \
              vec3_m);                                                      \
    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask2, src1, src1, mask2, vec4_m,  \
              vec5_m);                                                      \
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask2, src3, src3, mask2, vec6_m,  \
              vec7_m);                                                      \
    DUP4_ARG3(__lsx_vdp2add_h_b, out0, vec0_m, filt1, out1, vec1_m, filt1,  \
              out2, vec2_m, filt1, out3, vec3_m, filt1, out0, out1, out2,   \
              out3);                                                        \
    DUP4_ARG3(__lsx_vdp2add_h_b, out0, vec4_m, filt2, out1, vec5_m, filt2,  \
              out2, vec6_m, filt2, out3, vec7_m, filt2, out0, out1, out2,   \
              out3);                                                        \
  } while (0)

#define HORIZ_4TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,   \
                                   filt0, filt1, out0, out1)               \
  do {                                                                     \
    __m128i vec0_m, vec1_m, vec2_m, vec3_m;                                \
                                                                           \
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, mask0, src3, src2, mask0, vec0_m, \
              vec1_m);                                                     \
    DUP2_ARG2(__lsx_vdp2_h_b, vec0_m, filt0, vec1_m, filt0, out0, out1);   \
    DUP2_ARG3(__lsx_vshuf_b, src1, src0, mask1, src3, src2, mask1, vec2_m, \
              vec3_m);                                                     \
    DUP2_ARG3(__lsx_vdp2add_h_b, out0, vec2_m, filt1, out1, vec3_m, filt1, \
              out0, out1);                                                 \
  } while (0)

#define HORIZ_4TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1,   \
                                   filt0, filt1, out0, out1, out2, out3)   \
  do {                                                                     \
    __m128i vec0_m, vec1_m, vec2_m, vec3_m;                                \
                                                                           \
    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask0, src1, src1, mask0, vec0_m, \
              vec1_m);                                                     \
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask0, src3, src3, mask0, vec2_m, \
              vec3_m);                                                     \
    DUP4_ARG2(__lsx_vdp2_h_b, vec0_m, filt0, vec1_m, filt0, vec2_m, filt0, \
              vec3_m, filt0, out0, out1, out2, out3);                      \
    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask1, src1, src1, mask1, vec0_m, \
              vec1_m);                                                     \
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask1, src3, src3, mask1, vec2_m, \
              vec3_m);                                                     \
    DUP4_ARG3(__lsx_vdp2add_h_b, out0, vec0_m, filt1, out1, vec1_m, filt1, \
              out2, vec2_m, filt1, out3, vec3_m, filt1, out0, out1, out2,  \
              out3);                                                       \
  } while (0)

static inline void common_hz_6t_4x4_lsx(uint8_t *RESTRICT src,
                                        int32_t src_stride,
                                        uint8_t *RESTRICT dst,
                                        int32_t dst_stride,
                                        const int8_t *filter) {
  __m128i src0, src1, src2, src3, filt0, filt1, filt2;
  __m128i mask0, mask1, mask2, out0, out1;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 16);
  src -= 2;

  DUP2_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filt0, filt1);
  filt2 = __lsx_vldrepl_h(filter, 4);

  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);
  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src1, src2);
  src3 = __lsx_vldx(src, src_stride_x3);

  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  HORIZ_6TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, filt0,
                             filt1, filt2, out0, out1);
  out0 = __lsx_vssrarni_b_h(out1, out0, VP8_FILTER_SHIFT);
  out0 = __lsx_vxori_b(out0, 128);

  __lsx_vstelm_w(out0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 3);
}

static void common_hz_6t_4x8_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  __m128i src0, src1, src2, src3, filt0, filt1, filt2;
  __m128i mask0, mask1, mask2, out0, out1, out2, out3;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride_x2 << 1;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 16);
  src -= 2;

  DUP2_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filt0, filt1);
  filt2 = __lsx_vldrepl_h(filter, 4);
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);

  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src1, src2);
  src3 = __lsx_vldx(src, src_stride_x3);
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  src += src_stride_x4;
  HORIZ_6TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, filt0,
                             filt1, filt2, out0, out1);

  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src1, src2);
  src3 = __lsx_vldx(src, src_stride_x3);
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  HORIZ_6TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, filt0,
                             filt1, filt2, out2, out3);

  DUP2_ARG3(__lsx_vssrarni_b_h, out1, out0, VP8_FILTER_SHIFT, out3, out2,
            VP8_FILTER_SHIFT, out0, out1);
  DUP2_ARG2(__lsx_vxori_b, out0, 128, out1, 128, out0, out1);
  __lsx_vstelm_w(out0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 3);
  dst += dst_stride;

  __lsx_vstelm_w(out1, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(out1, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(out1, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(out1, dst, 0, 3);
}

static void common_hz_6t_4w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  if (height == 4) {
    common_hz_6t_4x4_lsx(src, src_stride, dst, dst_stride, filter);
  } else if (height == 8) {
    common_hz_6t_4x8_lsx(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_hz_6t_8w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, filt0, filt1, filt2;
  __m128i mask0, mask1, mask2, tmp0, tmp1;
  __m128i filt, out0, out1, out2, out3;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 0);
  src -= 2;

  filt = __lsx_vld(filter, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt0, filt1);
  filt2 = __lsx_vreplvei_h(filt, 2);
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);

  DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
            src_stride_x3, src0, src1, src2, src3);
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  src += src_stride_x4;
  HORIZ_6TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, filt0,
                             filt1, filt2, out0, out1, out2, out3);
  DUP2_ARG3(__lsx_vssrarni_b_h, out1, out0, VP8_FILTER_SHIFT, out3, out2,
            VP8_FILTER_SHIFT, tmp0, tmp1);
  DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
  __lsx_vstelm_d(tmp0, dst, 0, 0);
  __lsx_vstelm_d(tmp0, dst + dst_stride, 0, 1);
  __lsx_vstelm_d(tmp1, dst + dst_stride_x2, 0, 0);
  __lsx_vstelm_d(tmp1, dst + dst_stride_x3, 0, 1);
  dst += dst_stride_x4;

  for (loop_cnt = (height >> 2) - 1; loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    src += src_stride_x4;
    HORIZ_6TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2,
                               filt0, filt1, filt2, out0, out1, out2, out3);
    DUP2_ARG3(__lsx_vssrarni_b_h, out1, out0, VP8_FILTER_SHIFT, out3, out2,
              VP8_FILTER_SHIFT, tmp0, tmp1);
    DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
    __lsx_vstelm_d(tmp0, dst, 0, 0);
    __lsx_vstelm_d(tmp0, dst + dst_stride, 0, 1);
    __lsx_vstelm_d(tmp1, dst + dst_stride_x2, 0, 0);
    __lsx_vstelm_d(tmp1, dst + dst_stride_x3, 0, 1);
    dst += dst_stride_x4;
  }
}

static void common_hz_6t_16w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, filt0, filt1, filt2;
  __m128i mask0, mask1, mask2, out;
  __m128i filt, out0, out1, out2, out3, out4, out5, out6, out7;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 0);
  src -= 2;

  filt = __lsx_vld(filter, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt0, filt1);
  filt2 = __lsx_vreplvei_h(filt, 2);
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src0, src2, src4, src6);
    src += 8;
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src1, src3, src5, src7);
    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    DUP4_ARG2(__lsx_vxori_b, src4, 128, src5, 128, src6, 128, src7, 128, src4,
              src5, src6, src7);
    src += src_stride_x4 - 8;

    HORIZ_6TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2,
                               filt0, filt1, filt2, out0, out1, out2, out3);
    HORIZ_6TAP_8WID_4VECS_FILT(src4, src5, src6, src7, mask0, mask1, mask2,
                               filt0, filt1, filt2, out4, out5, out6, out7);
    DUP4_ARG2(__lsx_vsrari_h, out0, VP8_FILTER_SHIFT, out1, VP8_FILTER_SHIFT,
              out2, VP8_FILTER_SHIFT, out3, VP8_FILTER_SHIFT, out0, out1, out2,
              out3);
    DUP4_ARG2(__lsx_vsrari_h, out4, VP8_FILTER_SHIFT, out5, VP8_FILTER_SHIFT,
              out6, VP8_FILTER_SHIFT, out7, VP8_FILTER_SHIFT, out4, out5, out6,
              out7);
    DUP4_ARG2(__lsx_vsat_h, out0, 7, out1, 7, out2, 7, out3, 7, out0, out1,
              out2, out3);
    DUP4_ARG2(__lsx_vsat_h, out4, 7, out5, 7, out6, 7, out7, 7, out4, out5,
              out6, out7);
    out = __lsx_vpickev_b(out1, out0);
    out = __lsx_vxori_b(out, 128);
    __lsx_vst(out, dst, 0);
    out = __lsx_vpickev_b(out3, out2);
    out = __lsx_vxori_b(out, 128);
    __lsx_vstx(out, dst, dst_stride);
    out = __lsx_vpickev_b(out5, out4);
    out = __lsx_vxori_b(out, 128);
    __lsx_vstx(out, dst, dst_stride_x2);
    out = __lsx_vpickev_b(out7, out6);
    out = __lsx_vxori_b(out, 128);
    __lsx_vstx(out, dst, dst_stride_x3);
    dst += dst_stride_x4;
  }
}

static void common_vt_6t_4w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8;
  __m128i src10_r, src32_r, src54_r, src76_r, src21_r, src43_r, src65_r;
  __m128i src87_r, src2110, src4332, src6554, src8776, filt0, filt1, filt2;
  __m128i out0, out1;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;

  DUP2_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filt0, filt1);
  filt2 = __lsx_vldrepl_h(filter, 4);

  DUP2_ARG2(__lsx_vldx, src, -src_stride_x2, src, -src_stride, src0, src1);
  src2 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src3, src4);
  src += src_stride_x3;

  DUP4_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, src3, src2, src4, src3,
            src10_r, src21_r, src32_r, src43_r);
  DUP2_ARG2(__lsx_vilvl_d, src21_r, src10_r, src43_r, src32_r, src2110,
            src4332);
  DUP2_ARG2(__lsx_vxori_b, src2110, 128, src4332, 128, src2110, src4332);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    src5 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src6, src7);
    src8 = __lsx_vldx(src, src_stride_x3);
    src += src_stride_x4;

    DUP4_ARG2(__lsx_vilvl_b, src5, src4, src6, src5, src7, src6, src8, src7,
              src54_r, src65_r, src76_r, src87_r);
    DUP2_ARG2(__lsx_vilvl_d, src65_r, src54_r, src87_r, src76_r, src6554,
              src8776);
    DUP2_ARG2(__lsx_vxori_b, src6554, 128, src8776, 128, src6554, src8776);
    out0 = dpadd_h3(src2110, src4332, src6554, filt0, filt1, filt2);
    out1 = dpadd_h3(src4332, src6554, src8776, filt0, filt1, filt2);

    out0 = __lsx_vssrarni_b_h(out1, out0, VP8_FILTER_SHIFT);
    out0 = __lsx_vxori_b(out0, 128);

    __lsx_vstelm_w(out0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_w(out0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_w(out0, dst, 0, 2);
    dst += dst_stride;
    __lsx_vstelm_w(out0, dst, 0, 3);
    dst += dst_stride;

    src2110 = src6554;
    src4332 = src8776;
    src4 = src8;
  }
}

static void common_vt_6t_8w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, src4, src7, src8, src9, src10;
  __m128i src10_r, src32_r, src76_r, src98_r, src21_r, src43_r, src87_r;
  __m128i src109_r, filt0, filt1, filt2;
  __m128i tmp0, tmp1;
  __m128i filt, out0_r, out1_r, out2_r, out3_r;

  src -= src_stride_x2;
  filt = __lsx_vld(filter, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt0, filt1);
  filt2 = __lsx_vreplvei_h(filt, 2);

  DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
            src_stride_x3, src0, src1, src2, src3);
  src += src_stride_x4;
  src4 = __lsx_vld(src, 0);
  src += src_stride;

  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  src4 = __lsx_vxori_b(src4, 128);
  DUP4_ARG2(__lsx_vilvl_b, src1, src0, src3, src2, src2, src1, src4, src3,
            src10_r, src32_r, src21_r, src43_r);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src7, src8, src9, src10);
    DUP4_ARG2(__lsx_vxori_b, src7, 128, src8, 128, src9, 128, src10, 128, src7,
              src8, src9, src10);
    src += src_stride_x4;

    DUP4_ARG2(__lsx_vilvl_b, src7, src4, src8, src7, src9, src8, src10, src9,
              src76_r, src87_r, src98_r, src109_r);
    out0_r = dpadd_h3(src10_r, src32_r, src76_r, filt0, filt1, filt2);
    out1_r = dpadd_h3(src21_r, src43_r, src87_r, filt0, filt1, filt2);
    out2_r = dpadd_h3(src32_r, src76_r, src98_r, filt0, filt1, filt2);
    out3_r = dpadd_h3(src43_r, src87_r, src109_r, filt0, filt1, filt2);
    DUP2_ARG3(__lsx_vssrarni_b_h, out1_r, out0_r, VP8_FILTER_SHIFT, out3_r,
              out2_r, VP8_FILTER_SHIFT, tmp0, tmp1);
    DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
    __lsx_vstelm_d(tmp0, dst, 0, 0);
    __lsx_vstelm_d(tmp0, dst + dst_stride, 0, 1);
    __lsx_vstelm_d(tmp1, dst + dst_stride_x2, 0, 0);
    __lsx_vstelm_d(tmp1, dst + dst_stride_x3, 0, 1);
    dst += dst_stride_x4;

    src10_r = src76_r;
    src32_r = src98_r;
    src21_r = src87_r;
    src43_r = src109_r;
    src4 = src10;
  }
}

static void common_vt_6t_16w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8;
  __m128i src10_r, src32_r, src54_r, src76_r, src21_r, src43_r, src65_r;
  __m128i src87_r, src10_l, src32_l, src54_l, src76_l, src21_l, src43_l;
  __m128i src65_l, src87_l, filt0, filt1, filt2;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i filt, out0_r, out1_r, out2_r, out3_r, out0_l, out1_l, out2_l, out3_l;

  src -= src_stride_x2;
  filt = __lsx_vld(filter, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt0, filt1);
  filt2 = __lsx_vreplvei_h(filt, 2);

  DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
            src_stride_x3, src0, src1, src2, src3);
  src += src_stride_x4;
  src4 = __lsx_vldx(src, 0);
  src += src_stride;

  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  src4 = __lsx_vxori_b(src4, 128);
  DUP4_ARG2(__lsx_vilvl_b, src1, src0, src3, src2, src4, src3, src2, src1,
            src10_r, src32_r, src43_r, src21_r);
  DUP4_ARG2(__lsx_vilvh_b, src1, src0, src3, src2, src4, src3, src2, src1,
            src10_l, src32_l, src43_l, src21_l);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src5, src6, src7, src8);
    src += src_stride_x4;

    DUP4_ARG2(__lsx_vxori_b, src5, 128, src6, 128, src7, 128, src8, 128, src5,
              src6, src7, src8);
    DUP4_ARG2(__lsx_vilvl_b, src5, src4, src6, src5, src7, src6, src8, src7,
              src54_r, src65_r, src76_r, src87_r);
    DUP4_ARG2(__lsx_vilvh_b, src5, src4, src6, src5, src7, src6, src8, src7,
              src54_l, src65_l, src76_l, src87_l);
    out0_r = dpadd_h3(src10_r, src32_r, src54_r, filt0, filt1, filt2);
    out1_r = dpadd_h3(src21_r, src43_r, src65_r, filt0, filt1, filt2);
    out2_r = dpadd_h3(src32_r, src54_r, src76_r, filt0, filt1, filt2);
    out3_r = dpadd_h3(src43_r, src65_r, src87_r, filt0, filt1, filt2);
    out0_l = dpadd_h3(src10_l, src32_l, src54_l, filt0, filt1, filt2);
    out1_l = dpadd_h3(src21_l, src43_l, src65_l, filt0, filt1, filt2);
    out2_l = dpadd_h3(src32_l, src54_l, src76_l, filt0, filt1, filt2);
    out3_l = dpadd_h3(src43_l, src65_l, src87_l, filt0, filt1, filt2);
    DUP4_ARG3(__lsx_vssrarni_b_h, out0_l, out0_r, VP8_FILTER_SHIFT, out1_l,
              out1_r, VP8_FILTER_SHIFT, out2_l, out2_r, VP8_FILTER_SHIFT,
              out3_l, out3_r, VP8_FILTER_SHIFT, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp2, 128, tmp3, 128, tmp0,
              tmp1, tmp2, tmp3);
    __lsx_vstx(tmp0, dst, 0);
    __lsx_vstx(tmp1, dst, dst_stride);
    __lsx_vstx(tmp2, dst, dst_stride_x2);
    __lsx_vstx(tmp3, dst, dst_stride_x3);
    dst += dst_stride_x4;

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

static void common_hv_6ht_6vt_4w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8, tmp0, tmp1;
  __m128i filt_hz0, filt_hz1, filt_hz2, mask0, mask1, mask2;
  __m128i hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  __m128i hz_out7, filt_vt0, filt_vt1, filt_vt2, out0, out1, out2, out3;
  __m128i shuff = { 0x0F0E0D0C0B0A0908, 0x1716151413121110 };
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 16);
  src -= 2;

  DUP2_ARG2(__lsx_vldrepl_h, filter_horiz, 0, filter_horiz, 2, filt_hz0,
            filt_hz1);
  filt_hz2 = __lsx_vldrepl_h(filter_horiz, 4);
  DUP2_ARG2(__lsx_vldrepl_h, filter_vert, 0, filter_vert, 2, filt_vt0,
            filt_vt1);
  filt_vt2 = __lsx_vldrepl_h(filter_vert, 4);

  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);

  DUP2_ARG2(__lsx_vldx, src, -src_stride_x2, src, -src_stride, src0, src1);
  src2 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src3, src4);
  src += src_stride_x3;

  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  src4 = __lsx_vxori_b(src4, 128);

  hz_out0 = horiz_6tap_filt(src0, src1, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out2 = horiz_6tap_filt(src2, src3, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out1 = __lsx_vshuf_b(hz_out2, hz_out0, shuff);
  hz_out3 = horiz_6tap_filt(src3, src4, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  DUP2_ARG2(__lsx_vpackev_b, hz_out1, hz_out0, hz_out3, hz_out2, out0, out1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    src5 = __lsx_vld(src, 0);
    src6 = __lsx_vldx(src, src_stride);
    src += src_stride_x2;

    DUP2_ARG2(__lsx_vxori_b, src5, 128, src6, 128, src5, src6);
    hz_out5 = horiz_6tap_filt(src5, src6, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    hz_out4 = __lsx_vshuf_b(hz_out5, hz_out3, shuff);

    src7 = __lsx_vld(src, 0);
    src8 = __lsx_vldx(src, src_stride);
    src += src_stride_x2;

    DUP2_ARG2(__lsx_vxori_b, src7, 128, src8, 128, src7, src8);
    hz_out7 = horiz_6tap_filt(src7, src8, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    hz_out6 = __lsx_vshuf_b(hz_out7, hz_out5, shuff);

    out2 = __lsx_vpackev_b(hz_out5, hz_out4);
    tmp0 = dpadd_h3(out0, out1, out2, filt_vt0, filt_vt1, filt_vt2);

    out3 = __lsx_vpackev_b(hz_out7, hz_out6);
    tmp1 = dpadd_h3(out1, out2, out3, filt_vt0, filt_vt1, filt_vt2);

    tmp0 = __lsx_vssrarni_b_h(tmp1, tmp0, 7);
    tmp0 = __lsx_vxori_b(tmp0, 128);
    __lsx_vstelm_w(tmp0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 2);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 3);
    dst += dst_stride;

    hz_out3 = hz_out7;
    out0 = out2;
    out1 = out3;
  }
}

static void common_hv_6ht_6vt_8w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8;
  __m128i filt_hz0, filt_hz1, filt_hz2;
  __m128i mask0, mask1, mask2, vec0, vec1;
  __m128i filt, filt_vt0, filt_vt1, filt_vt2;
  __m128i hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  __m128i hz_out7, hz_out8, out0, out1, out2, out3, out4, out5, out6, out7;
  __m128i tmp0, tmp1, tmp2, tmp3;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 0);
  src -= (2 + src_stride_x2);

  filt = __lsx_vld(filter_horiz, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt_hz0, filt_hz1);
  filt_hz2 = __lsx_vreplvei_h(filt, 2);

  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);
  DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
            src_stride_x3, src0, src1, src2, src3);
  src += src_stride_x4;
  src4 = __lsx_vldx(src, 0);
  src += src_stride;

  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  src4 = __lsx_vxori_b(src4, 128);

  hz_out0 = horiz_6tap_filt(src0, src0, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out1 = horiz_6tap_filt(src1, src1, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out2 = horiz_6tap_filt(src2, src2, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out3 = horiz_6tap_filt(src3, src3, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out4 = horiz_6tap_filt(src4, src4, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  filt = __lsx_vld(filter_vert, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt_vt0, filt_vt1);
  filt_vt2 = __lsx_vreplvei_h(filt, 2);

  DUP4_ARG2(__lsx_vpackev_b, hz_out1, hz_out0, hz_out3, hz_out2, hz_out2,
            hz_out1, hz_out4, hz_out3, out0, out1, out3, out4);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src5, src6, src7, src8);
    src += src_stride_x4;

    DUP4_ARG2(__lsx_vxori_b, src5, 128, src6, 128, src7, 128, src8, 128, src5,
              src6, src7, src8);
    hz_out5 = horiz_6tap_filt(src5, src5, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    out2 = __lsx_vpackev_b(hz_out5, hz_out4);
    tmp0 = dpadd_h3(out0, out1, out2, filt_vt0, filt_vt1, filt_vt2);

    hz_out6 = horiz_6tap_filt(src6, src6, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    out5 = __lsx_vpackev_b(hz_out6, hz_out5);
    tmp1 = dpadd_h3(out3, out4, out5, filt_vt0, filt_vt1, filt_vt2);

    hz_out7 = horiz_6tap_filt(src7, src7, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    out7 = __lsx_vpackev_b(hz_out7, hz_out6);
    tmp2 = dpadd_h3(out1, out2, out7, filt_vt0, filt_vt1, filt_vt2);

    hz_out8 = horiz_6tap_filt(src8, src8, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    out6 = __lsx_vpackev_b(hz_out8, hz_out7);
    tmp3 = dpadd_h3(out4, out5, out6, filt_vt0, filt_vt1, filt_vt2);

    DUP2_ARG3(__lsx_vssrarni_b_h, tmp1, tmp0, VP8_FILTER_SHIFT, tmp3, tmp2,
              VP8_FILTER_SHIFT, vec0, vec1);
    DUP2_ARG2(__lsx_vxori_b, vec0, 128, vec1, 128, vec0, vec1);

    __lsx_vstelm_d(vec0, dst, 0, 0);
    __lsx_vstelm_d(vec0, dst + dst_stride, 0, 1);
    __lsx_vstelm_d(vec1, dst + dst_stride_x2, 0, 0);
    __lsx_vstelm_d(vec1, dst + dst_stride_x3, 0, 1);
    dst += dst_stride_x4;

    hz_out4 = hz_out8;
    out0 = out2;
    out1 = out7;
    out3 = out5;
    out4 = out6;
  }
}

static void common_hv_6ht_6vt_16w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert,
                                      int32_t height) {
  common_hv_6ht_6vt_8w_lsx(src, src_stride, dst, dst_stride, filter_horiz,
                           filter_vert, height);
  common_hv_6ht_6vt_8w_lsx(src + 8, src_stride, dst + 8, dst_stride,
                           filter_horiz, filter_vert, height);
}

static void common_hz_4t_4x4_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  __m128i src0, src1, src2, src3, filt0, filt1, mask0, mask1;
  __m128i out0, out1;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 16);
  src -= 1;

  DUP2_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filt0, filt1);
  mask1 = __lsx_vaddi_bu(mask0, 2);

  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src1, src2);
  src3 = __lsx_vldx(src, src_stride_x3);
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  HORIZ_4TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0, filt1,
                             out0, out1);

  out0 = __lsx_vssrarni_b_h(out1, out0, VP8_FILTER_SHIFT);
  out0 = __lsx_vxori_b(out0, 128);

  __lsx_vstelm_w(out0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 3);
}

static void common_hz_4t_4x8_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter) {
  __m128i src0, src1, src2, src3, filt0, filt1, mask0, mask1;
  __m128i out0, out1, out2, out3;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 16);
  src -= 1;

  DUP2_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filt0, filt1);
  mask1 = __lsx_vaddi_bu(mask0, 2);

  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src1, src2);
  src3 = __lsx_vldx(src, src_stride_x3);
  src += src_stride_x4;
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  HORIZ_4TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0, filt1,
                             out0, out1);

  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src1, src2);
  src3 = __lsx_vldx(src, src_stride_x3);
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  HORIZ_4TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0, filt1,
                             out2, out3);
  DUP2_ARG3(__lsx_vssrarni_b_h, out1, out0, VP8_FILTER_SHIFT, out3, out2,
            VP8_FILTER_SHIFT, out0, out1);
  DUP2_ARG2(__lsx_vxori_b, out0, 128, out1, 128, out0, out1);
  __lsx_vstelm_w(out0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(out0, dst, 0, 3);
  dst += dst_stride;

  __lsx_vstelm_w(out1, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(out1, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(out1, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(out1, dst, 0, 3);
}

static void common_hz_4t_4w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  if (height == 4) {
    common_hz_4t_4x4_lsx(src, src_stride, dst, dst_stride, filter);
  } else if (height == 8) {
    common_hz_4t_4x8_lsx(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_hz_4t_8w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, filt0, filt1, mask0, mask1;
  __m128i tmp0, tmp1;
  __m128i filt, out0, out1, out2, out3;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 0);
  src -= 1;

  filt = __lsx_vld(filter, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt0, filt1);
  mask1 = __lsx_vaddi_bu(mask0, 2);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src0, src1, src2, src3);
    src += src_stride_x4;

    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    HORIZ_4TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0,
                               filt1, out0, out1, out2, out3);
    DUP2_ARG3(__lsx_vssrarni_b_h, out1, out0, VP8_FILTER_SHIFT, out3, out2,
              VP8_FILTER_SHIFT, tmp0, tmp1);
    DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
    __lsx_vstelm_d(tmp0, dst, 0, 0);
    __lsx_vstelm_d(tmp0, dst + dst_stride, 0, 1);
    __lsx_vstelm_d(tmp1, dst + dst_stride_x2, 0, 0);
    __lsx_vstelm_d(tmp1, dst + dst_stride_x3, 0, 1);
    dst += dst_stride_x4;
  }
}

static void common_hz_4t_16w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i filt0, filt1, mask0, mask1;
  __m128i filt, out0, out1, out2, out3, out4, out5, out6, out7;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 0);
  src -= 1;

  filt = __lsx_vld(filter, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt0, filt1);
  mask1 = __lsx_vaddi_bu(mask0, 2);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src0, src2, src4, src6);
    src += 8;
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src1, src3, src5, src7);
    src += src_stride_x4 - 8;

    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    DUP4_ARG2(__lsx_vxori_b, src4, 128, src5, 128, src6, 128, src7, 128, src4,
              src5, src6, src7);
    HORIZ_4TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, filt0,
                               filt1, out0, out1, out2, out3);
    HORIZ_4TAP_8WID_4VECS_FILT(src4, src5, src6, src7, mask0, mask1, filt0,
                               filt1, out4, out5, out6, out7);
    DUP4_ARG3(__lsx_vssrarni_b_h, out1, out0, VP8_FILTER_SHIFT, out3, out2,
              VP8_FILTER_SHIFT, out5, out4, VP8_FILTER_SHIFT, out7, out6,
              VP8_FILTER_SHIFT, out0, out1, out2, out3);
    DUP4_ARG2(__lsx_vxori_b, out0, 128, out1, 128, out2, 128, out3, 128, out0,
              out1, out2, out3);
    __lsx_vstx(out0, dst, 0);
    __lsx_vstx(out1, dst, dst_stride);
    __lsx_vstx(out2, dst, dst_stride_x2);
    __lsx_vstx(out3, dst, dst_stride_x3);
    dst += dst_stride_x4;
  }
}

static void common_vt_4t_4w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  __m128i src0, src1, src2, src3, src4, src5;
  __m128i src10_r, src32_r, src54_r, src21_r, src43_r, src65_r;
  __m128i src2110, src4332, filt0, filt1, out0, out1;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;

  DUP2_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filt0, filt1);
  DUP2_ARG2(__lsx_vldx, src, -src_stride, src, src_stride, src0, src2);
  src1 = __lsx_vld(src, 0);
  src += src_stride_x2;

  DUP2_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, src10_r, src21_r);

  src2110 = __lsx_vilvl_d(src21_r, src10_r);
  src2110 = __lsx_vxori_b(src2110, 128);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    src3 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src4, src5);
    src += src_stride_x3;
    DUP2_ARG2(__lsx_vilvl_b, src3, src2, src4, src3, src32_r, src43_r);
    src4332 = __lsx_vilvl_d(src43_r, src32_r);
    src4332 = __lsx_vxori_b(src4332, 128);
    out0 = filt_4tap_dpadd_h(src2110, src4332, filt0, filt1);

    src2 = __lsx_vld(src, 0);
    src += src_stride;
    DUP2_ARG2(__lsx_vilvl_b, src5, src4, src2, src5, src54_r, src65_r);
    src2110 = __lsx_vilvl_d(src65_r, src54_r);
    src2110 = __lsx_vxori_b(src2110, 128);
    out1 = filt_4tap_dpadd_h(src4332, src2110, filt0, filt1);
    out0 = __lsx_vssrarni_b_h(out1, out0, VP8_FILTER_SHIFT);
    out0 = __lsx_vxori_b(out0, 128);

    __lsx_vstelm_w(out0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_w(out0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_w(out0, dst, 0, 2);
    dst += dst_stride;
    __lsx_vstelm_w(out0, dst, 0, 3);
    dst += dst_stride;
  }
}

static void common_vt_4t_8w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                uint8_t *RESTRICT dst, int32_t dst_stride,
                                const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src7, src8, src9, src10;
  __m128i src10_r, src72_r, src98_r, src21_r, src87_r, src109_r, filt0, filt1;
  __m128i tmp0, tmp1;
  __m128i filt, out0_r, out1_r, out2_r, out3_r;

  src -= src_stride;
  filt = __lsx_vld(filter, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt0, filt1);

  DUP2_ARG2(__lsx_vldx, src, 0, src, src_stride, src0, src1);
  src2 = __lsx_vldx(src, src_stride_x2);
  src += src_stride_x3;

  DUP2_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src0, src1);
  src2 = __lsx_vxori_b(src2, 128);
  DUP2_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, src10_r, src21_r);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src7, src8, src9, src10);
    src += src_stride_x4;

    DUP4_ARG2(__lsx_vxori_b, src7, 128, src8, 128, src9, 128, src10, 128, src7,
              src8, src9, src10);
    DUP4_ARG2(__lsx_vilvl_b, src7, src2, src8, src7, src9, src8, src10, src9,
              src72_r, src87_r, src98_r, src109_r);
    out0_r = filt_4tap_dpadd_h(src10_r, src72_r, filt0, filt1);
    out1_r = filt_4tap_dpadd_h(src21_r, src87_r, filt0, filt1);
    out2_r = filt_4tap_dpadd_h(src72_r, src98_r, filt0, filt1);
    out3_r = filt_4tap_dpadd_h(src87_r, src109_r, filt0, filt1);
    DUP2_ARG3(__lsx_vssrarni_b_h, out1_r, out0_r, VP8_FILTER_SHIFT, out3_r,
              out2_r, VP8_FILTER_SHIFT, tmp0, tmp1);
    DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
    __lsx_vstelm_d(tmp0, dst, 0, 0);
    __lsx_vstelm_d(tmp0, dst + dst_stride, 0, 1);
    __lsx_vstelm_d(tmp1, dst + dst_stride_x2, 0, 0);
    __lsx_vstelm_d(tmp1, dst + dst_stride_x3, 0, 1);
    dst += dst_stride_x4;

    src10_r = src98_r;
    src21_r = src109_r;
    src2 = src10;
  }
}

static void common_vt_4t_16w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                 uint8_t *RESTRICT dst, int32_t dst_stride,
                                 const int8_t *filter, int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, src4, src5, src6;
  __m128i src10_r, src32_r, src54_r, src21_r, src43_r, src65_r, src10_l;
  __m128i src32_l, src54_l, src21_l, src43_l, src65_l, filt0, filt1;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i filt, out0_r, out1_r, out2_r, out3_r, out0_l, out1_l, out2_l, out3_l;

  src -= src_stride;
  filt = __lsx_vld(filter, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt0, filt1);

  DUP2_ARG2(__lsx_vldx, src, 0, src, src_stride, src0, src1);
  src2 = __lsx_vldx(src, src_stride_x2);
  src += src_stride_x3;

  DUP2_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src0, src1);
  src2 = __lsx_vxori_b(src2, 128);
  DUP2_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, src10_r, src21_r);
  DUP2_ARG2(__lsx_vilvh_b, src1, src0, src2, src1, src10_l, src21_l);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src3, src4, src5, src6);
    src += src_stride_x4;

    DUP4_ARG2(__lsx_vxori_b, src3, 128, src4, 128, src5, 128, src6, 128, src3,
              src4, src5, src6);
    DUP4_ARG2(__lsx_vilvl_b, src3, src2, src4, src3, src5, src4, src6, src5,
              src32_r, src43_r, src54_r, src65_r);
    DUP4_ARG2(__lsx_vilvh_b, src3, src2, src4, src3, src5, src4, src6, src5,
              src32_l, src43_l, src54_l, src65_l);
    out0_r = filt_4tap_dpadd_h(src10_r, src32_r, filt0, filt1);
    out1_r = filt_4tap_dpadd_h(src21_r, src43_r, filt0, filt1);
    out2_r = filt_4tap_dpadd_h(src32_r, src54_r, filt0, filt1);
    out3_r = filt_4tap_dpadd_h(src43_r, src65_r, filt0, filt1);
    out0_l = filt_4tap_dpadd_h(src10_l, src32_l, filt0, filt1);
    out1_l = filt_4tap_dpadd_h(src21_l, src43_l, filt0, filt1);
    out2_l = filt_4tap_dpadd_h(src32_l, src54_l, filt0, filt1);
    out3_l = filt_4tap_dpadd_h(src43_l, src65_l, filt0, filt1);
    DUP4_ARG3(__lsx_vssrarni_b_h, out0_l, out0_r, VP8_FILTER_SHIFT, out1_l,
              out1_r, VP8_FILTER_SHIFT, out2_l, out2_r, VP8_FILTER_SHIFT,
              out3_l, out3_r, VP8_FILTER_SHIFT, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp2, 128, tmp3, 128, tmp0,
              tmp1, tmp2, tmp3);
    __lsx_vstx(tmp0, dst, 0);
    __lsx_vstx(tmp1, dst, dst_stride);
    __lsx_vstx(tmp2, dst, dst_stride_x2);
    __lsx_vstx(tmp3, dst, dst_stride_x3);
    dst += dst_stride_x4;

    src10_r = src54_r;
    src21_r = src65_r;
    src10_l = src54_l;
    src21_l = src65_l;
    src2 = src6;
  }
}

static void common_hv_4ht_4vt_4w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  __m128i src0, src1, src2, src3, src4, src5, src6, filt_hz0, filt_hz1;
  __m128i mask0, mask1, filt_vt0, filt_vt1, tmp0, tmp1, vec0, vec1, vec2;
  __m128i hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5;
  __m128i shuff = { 0x0F0E0D0C0B0A0908, 0x1716151413121110 };
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 16);
  src -= 1;

  DUP2_ARG2(__lsx_vldrepl_h, filter_horiz, 0, filter_horiz, 2, filt_hz0,
            filt_hz1);
  mask1 = __lsx_vaddi_bu(mask0, 2);

  src1 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, -src_stride, src, src_stride, src0, src2);
  src += src_stride_x2;

  DUP2_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src0, src1);
  src2 = __lsx_vxori_b(src2, 128);
  hz_out0 = horiz_4tap_filt(src0, src1, mask0, mask1, filt_hz0, filt_hz1);
  hz_out1 = horiz_4tap_filt(src1, src2, mask0, mask1, filt_hz0, filt_hz1);
  vec0 = __lsx_vpackev_b(hz_out1, hz_out0);

  DUP2_ARG2(__lsx_vldrepl_h, filter_vert, 0, filter_vert, 2, filt_vt0,
            filt_vt1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    src3 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src4, src5);
    src6 = __lsx_vldx(src, src_stride_x3);
    src += src_stride_x4;

    DUP2_ARG2(__lsx_vxori_b, src3, 128, src4, 128, src3, src4);
    hz_out3 = horiz_4tap_filt(src3, src4, mask0, mask1, filt_hz0, filt_hz1);
    hz_out2 = __lsx_vshuf_b(hz_out3, hz_out1, shuff);
    vec1 = __lsx_vpackev_b(hz_out3, hz_out2);
    tmp0 = filt_4tap_dpadd_h(vec0, vec1, filt_vt0, filt_vt1);

    DUP2_ARG2(__lsx_vxori_b, src5, 128, src6, 128, src5, src6);
    hz_out5 = horiz_4tap_filt(src5, src6, mask0, mask1, filt_hz0, filt_hz1);
    hz_out4 = __lsx_vshuf_b(hz_out5, hz_out3, shuff);
    vec2 = __lsx_vpackev_b(hz_out5, hz_out4);
    tmp1 = filt_4tap_dpadd_h(vec1, vec2, filt_vt0, filt_vt1);

    tmp0 = __lsx_vssrarni_b_h(tmp1, tmp0, 7);
    tmp0 = __lsx_vxori_b(tmp0, 128);
    __lsx_vstelm_w(tmp0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 2);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 3);
    dst += dst_stride;

    hz_out1 = hz_out5;
    vec0 = vec2;
  }
}

static inline void common_hv_4ht_4vt_8w_lsx(
    uint8_t *RESTRICT src, int32_t src_stride, uint8_t *RESTRICT dst,
    int32_t dst_stride, const int8_t *filter_horiz, const int8_t *filter_vert,
    int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, src4, src5, src6, filt_hz0, filt_hz1;
  __m128i mask0, mask1, out0, out1;
  __m128i filt, filt_vt0, filt_vt1, tmp0, tmp1, tmp2, tmp3;
  __m128i hz_out0, hz_out1, hz_out2, hz_out3;
  __m128i vec0, vec1, vec2, vec3, vec4;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 0);
  src -= 1 + src_stride;

  filt = __lsx_vld(filter_horiz, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt_hz0, filt_hz1);
  mask1 = __lsx_vaddi_bu(mask0, 2);

  DUP2_ARG2(__lsx_vldx, src, 0, src, src_stride, src0, src1);
  src2 = __lsx_vldx(src, src_stride_x2);
  src += src_stride_x3;

  DUP2_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src0, src1);
  src2 = __lsx_vxori_b(src2, 128);
  hz_out0 = horiz_4tap_filt(src0, src0, mask0, mask1, filt_hz0, filt_hz1);
  hz_out1 = horiz_4tap_filt(src1, src1, mask0, mask1, filt_hz0, filt_hz1);
  hz_out2 = horiz_4tap_filt(src2, src2, mask0, mask1, filt_hz0, filt_hz1);
  DUP2_ARG2(__lsx_vpackev_b, hz_out1, hz_out0, hz_out2, hz_out1, vec0, vec2);

  filt = __lsx_vld(filter_vert, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt_vt0, filt_vt1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src3, src4, src5, src6);
    src += src_stride_x4;

    DUP4_ARG2(__lsx_vxori_b, src3, 128, src4, 128, src5, 128, src6, 128, src3,
              src4, src5, src6);
    hz_out3 = horiz_4tap_filt(src3, src3, mask0, mask1, filt_hz0, filt_hz1);
    vec1 = __lsx_vpackev_b(hz_out3, hz_out2);
    tmp0 = filt_4tap_dpadd_h(vec0, vec1, filt_vt0, filt_vt1);

    hz_out0 = horiz_4tap_filt(src4, src4, mask0, mask1, filt_hz0, filt_hz1);
    vec3 = __lsx_vpackev_b(hz_out0, hz_out3);
    tmp1 = filt_4tap_dpadd_h(vec2, vec3, filt_vt0, filt_vt1);

    hz_out1 = horiz_4tap_filt(src5, src5, mask0, mask1, filt_hz0, filt_hz1);
    vec4 = __lsx_vpackev_b(hz_out1, hz_out0);
    tmp2 = filt_4tap_dpadd_h(vec1, vec4, filt_vt0, filt_vt1);

    hz_out2 = horiz_4tap_filt(src6, src6, mask0, mask1, filt_hz0, filt_hz1);
    DUP2_ARG2(__lsx_vpackev_b, hz_out0, hz_out3, hz_out2, hz_out1, vec0, vec1);
    tmp3 = filt_4tap_dpadd_h(vec0, vec1, filt_vt0, filt_vt1);

    DUP2_ARG3(__lsx_vssrarni_b_h, tmp1, tmp0, 7, tmp3, tmp2, 7, out0, out1);
    DUP2_ARG2(__lsx_vxori_b, out0, 128, out1, 128, out0, out1);
    __lsx_vstelm_d(out0, dst, 0, 0);
    __lsx_vstelm_d(out0, dst + dst_stride, 0, 1);
    __lsx_vstelm_d(out1, dst + dst_stride_x2, 0, 0);
    __lsx_vstelm_d(out1, dst + dst_stride_x3, 0, 1);
    dst += dst_stride_x4;

    vec0 = vec4;
    vec2 = vec1;
  }
}

static void common_hv_4ht_4vt_16w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert,
                                      int32_t height) {
  common_hv_4ht_4vt_8w_lsx(src, src_stride, dst, dst_stride, filter_horiz,
                           filter_vert, height);
  common_hv_4ht_4vt_8w_lsx(src + 8, src_stride, dst + 8, dst_stride,
                           filter_horiz, filter_vert, height);
}

static void common_hv_6ht_4vt_4w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  __m128i src0, src1, src2, src3, src4, src5, src6;
  __m128i filt_hz0, filt_hz1, filt_hz2, mask0, mask1, mask2;
  __m128i filt_vt0, filt_vt1, tmp0, tmp1, vec0, vec1, vec2;
  __m128i hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5;
  __m128i shuff = { 0x0F0E0D0C0B0A0908, 0x1716151413121110 };
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 16);
  src -= 2;

  DUP2_ARG2(__lsx_vldrepl_h, filter_horiz, 0, filter_horiz, 2, filt_hz0,
            filt_hz1);
  filt_hz2 = __lsx_vldrepl_h(filter_horiz, 4);
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);

  src1 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, -src_stride, src, src_stride, src0, src2);
  src += src_stride_x2;

  DUP2_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src0, src1);
  src2 = __lsx_vxori_b(src2, 128);

  hz_out0 = horiz_6tap_filt(src0, src1, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out1 = horiz_6tap_filt(src1, src2, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  vec0 = __lsx_vpackev_b(hz_out1, hz_out0);

  DUP2_ARG2(__lsx_vldrepl_h, filter_vert, 0, filter_vert, 2, filt_vt0,
            filt_vt1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    src3 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src4, src5);
    src6 = __lsx_vldx(src, src_stride_x3);
    src += src_stride_x4;
    DUP4_ARG2(__lsx_vxori_b, src3, 128, src4, 128, src5, 128, src6, 128, src3,
              src4, src5, src6);

    hz_out3 = horiz_6tap_filt(src3, src4, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    hz_out2 = __lsx_vshuf_b(hz_out3, hz_out1, shuff);
    vec1 = __lsx_vpackev_b(hz_out3, hz_out2);
    tmp0 = filt_4tap_dpadd_h(vec0, vec1, filt_vt0, filt_vt1);

    hz_out5 = horiz_6tap_filt(src5, src6, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    hz_out4 = __lsx_vshuf_b(hz_out5, hz_out3, shuff);
    vec2 = __lsx_vpackev_b(hz_out5, hz_out4);
    tmp1 = filt_4tap_dpadd_h(vec1, vec2, filt_vt0, filt_vt1);

    DUP2_ARG3(__lsx_vssrarni_b_h, tmp0, tmp0, 7, tmp1, tmp1, 7, tmp0, tmp1);
    DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);

    __lsx_vstelm_w(tmp0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_w(tmp1, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_w(tmp1, dst, 0, 1);
    dst += dst_stride;

    hz_out1 = hz_out5;
    vec0 = vec2;
  }
}

static inline void common_hv_6ht_4vt_8w_lsx(
    uint8_t *RESTRICT src, int32_t src_stride, uint8_t *RESTRICT dst,
    int32_t dst_stride, const int8_t *filter_horiz, const int8_t *filter_vert,
    int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;

  __m128i src0, src1, src2, src3, src4, src5, src6;
  __m128i filt_hz0, filt_hz1, filt_hz2, mask0, mask1, mask2;
  __m128i filt, filt_vt0, filt_vt1, hz_out0, hz_out1, hz_out2, hz_out3;
  __m128i tmp0, tmp1, tmp2, tmp3, vec0, vec1, vec2, vec3;
  __m128i out0, out1;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 0);
  src -= (2 + src_stride);

  filt = __lsx_vld(filter_horiz, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt_hz0, filt_hz1);
  filt_hz2 = __lsx_vreplvei_h(filt, 2);
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);

  DUP2_ARG2(__lsx_vldx, src, 0, src, src_stride, src0, src1);
  src2 = __lsx_vldx(src, src_stride_x2);
  src += src_stride_x3;

  DUP2_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src0, src1);
  src2 = __lsx_vxori_b(src2, 128);
  hz_out0 = horiz_6tap_filt(src0, src0, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out1 = horiz_6tap_filt(src1, src1, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  hz_out2 = horiz_6tap_filt(src2, src2, mask0, mask1, mask2, filt_hz0, filt_hz1,
                            filt_hz2);
  DUP2_ARG2(__lsx_vpackev_b, hz_out1, hz_out0, hz_out2, hz_out1, vec0, vec2);

  filt = __lsx_vld(filter_vert, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt_vt0, filt_vt1);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src3, src4, src5, src6);
    src += src_stride_x4;
    DUP4_ARG2(__lsx_vxori_b, src3, 128, src4, 128, src5, 128, src6, 128, src3,
              src4, src5, src6);

    hz_out3 = horiz_6tap_filt(src3, src3, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    vec1 = __lsx_vpackev_b(hz_out3, hz_out2);
    tmp0 = filt_4tap_dpadd_h(vec0, vec1, filt_vt0, filt_vt1);

    hz_out0 = horiz_6tap_filt(src4, src4, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    vec3 = __lsx_vpackev_b(hz_out0, hz_out3);
    tmp1 = filt_4tap_dpadd_h(vec2, vec3, filt_vt0, filt_vt1);

    hz_out1 = horiz_6tap_filt(src5, src5, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    vec0 = __lsx_vpackev_b(hz_out1, hz_out0);
    tmp2 = filt_4tap_dpadd_h(vec1, vec0, filt_vt0, filt_vt1);

    hz_out2 = horiz_6tap_filt(src6, src6, mask0, mask1, mask2, filt_hz0,
                              filt_hz1, filt_hz2);
    DUP2_ARG2(__lsx_vpackev_b, hz_out0, hz_out3, hz_out2, hz_out1, vec1, vec2);
    tmp3 = filt_4tap_dpadd_h(vec1, vec2, filt_vt0, filt_vt1);

    DUP2_ARG3(__lsx_vssrarni_b_h, tmp1, tmp0, 7, tmp3, tmp2, 7, out0, out1);
    DUP2_ARG2(__lsx_vxori_b, out0, 128, out1, 128, out0, out1);
    __lsx_vstelm_d(out0, dst, 0, 0);
    __lsx_vstelm_d(out0, dst + dst_stride, 0, 1);
    __lsx_vstelm_d(out1, dst + dst_stride_x2, 0, 0);
    __lsx_vstelm_d(out1, dst + dst_stride_x3, 0, 1);
    dst += dst_stride_x4;
  }
}

static void common_hv_6ht_4vt_16w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert,
                                      int32_t height) {
  common_hv_6ht_4vt_8w_lsx(src, src_stride, dst, dst_stride, filter_horiz,
                           filter_vert, height);
  common_hv_6ht_4vt_8w_lsx(src + 8, src_stride, dst + 8, dst_stride,
                           filter_horiz, filter_vert, height);
}

static void common_hv_4ht_6vt_4w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                     uint8_t *RESTRICT dst, int32_t dst_stride,
                                     const int8_t *filter_horiz,
                                     const int8_t *filter_vert,
                                     int32_t height) {
  uint32_t loop_cnt;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8;
  __m128i filt_hz0, filt_hz1, filt_vt0, filt_vt1, filt_vt2, mask0, mask1;
  __m128i hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  __m128i hz_out7, tmp0, tmp1, out0, out1, out2, out3;
  __m128i shuff = { 0x0F0E0D0C0B0A0908, 0x1716151413121110 };
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 16);

  src -= 1;

  DUP2_ARG2(__lsx_vldrepl_h, filter_horiz, 0, filter_horiz, 2, filt_hz0,
            filt_hz1);
  mask1 = __lsx_vaddi_bu(mask0, 2);

  DUP4_ARG2(__lsx_vldx, src, -src_stride_x2, src, -src_stride, src, src_stride,
            src, src_stride_x2, src0, src1, src3, src4);
  src2 = __lsx_vld(src, 0);
  src += src_stride_x3;

  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  src4 = __lsx_vxori_b(src4, 128);
  hz_out0 = horiz_4tap_filt(src0, src1, mask0, mask1, filt_hz0, filt_hz1);
  hz_out2 = horiz_4tap_filt(src2, src3, mask0, mask1, filt_hz0, filt_hz1);
  hz_out3 = horiz_4tap_filt(src3, src4, mask0, mask1, filt_hz0, filt_hz1);
  hz_out1 = __lsx_vshuf_b(hz_out2, hz_out0, shuff);
  DUP2_ARG2(__lsx_vpackev_b, hz_out1, hz_out0, hz_out3, hz_out2, out0, out1);

  DUP2_ARG2(__lsx_vldrepl_h, filter_vert, 0, filter_vert, 2, filt_vt0,
            filt_vt1);
  filt_vt2 = __lsx_vldrepl_h(filter_vert, 4);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    src5 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride_x2, src6, src7);
    src8 = __lsx_vldx(src, src_stride_x3);
    DUP4_ARG2(__lsx_vxori_b, src5, 128, src6, 128, src7, 128, src8, 128, src5,
              src6, src7, src8);
    src += src_stride_x4;

    hz_out5 = horiz_4tap_filt(src5, src6, mask0, mask1, filt_hz0, filt_hz1);
    hz_out4 = __lsx_vshuf_b(hz_out5, hz_out3, shuff);
    out2 = __lsx_vpackev_b(hz_out5, hz_out4);
    tmp0 = dpadd_h3(out0, out1, out2, filt_vt0, filt_vt1, filt_vt2);

    hz_out7 = horiz_4tap_filt(src7, src8, mask0, mask1, filt_hz0, filt_hz1);
    hz_out6 = __lsx_vshuf_b(hz_out7, hz_out5, shuff);
    out3 = __lsx_vpackev_b(hz_out7, hz_out6);
    tmp1 = dpadd_h3(out1, out2, out3, filt_vt0, filt_vt1, filt_vt2);

    tmp0 = __lsx_vssrarni_b_h(tmp1, tmp0, 7);
    tmp0 = __lsx_vxori_b(tmp0, 128);
    __lsx_vstelm_w(tmp0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 2);
    dst += dst_stride;
    __lsx_vstelm_w(tmp0, dst, 0, 3);
    dst += dst_stride;

    hz_out3 = hz_out7;
    out0 = out2;
    out1 = out3;
  }
}

static inline void common_hv_4ht_6vt_8w_lsx(
    uint8_t *RESTRICT src, int32_t src_stride, uint8_t *RESTRICT dst,
    int32_t dst_stride, const int8_t *filter_horiz, const int8_t *filter_vert,
    int32_t height) {
  uint32_t loop_cnt;
  int32_t src_stride_x2 = src_stride << 1;
  int32_t src_stride_x3 = src_stride_x2 + src_stride;
  int32_t src_stride_x4 = src_stride << 2;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8;
  __m128i filt_hz0, filt_hz1, mask0, mask1;
  __m128i filt, filt_vt0, filt_vt1, filt_vt2, tmp0, tmp1, tmp2, tmp3;
  __m128i hz_out0, hz_out1, hz_out2, hz_out3, hz_out4, hz_out5, hz_out6;
  __m128i hz_out7, hz_out8, out0, out1, out2, out3, out4, out5, out6, out7;
  __m128i vec0, vec1;

  mask0 = __lsx_vld(vp8_mc_filt_mask_arr, 0);
  src -= 1 + src_stride_x2;

  filt = __lsx_vld(filter_horiz, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt_hz0, filt_hz1);
  mask1 = __lsx_vaddi_bu(mask0, 2);

  DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
            src_stride_x3, src0, src1, src2, src3);
  src += src_stride_x4;
  src4 = __lsx_vld(src, 0);
  src += src_stride;

  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  src4 = __lsx_vxori_b(src4, 128);
  hz_out0 = horiz_4tap_filt(src0, src0, mask0, mask1, filt_hz0, filt_hz1);
  hz_out1 = horiz_4tap_filt(src1, src1, mask0, mask1, filt_hz0, filt_hz1);
  hz_out2 = horiz_4tap_filt(src2, src2, mask0, mask1, filt_hz0, filt_hz1);
  hz_out3 = horiz_4tap_filt(src3, src3, mask0, mask1, filt_hz0, filt_hz1);
  hz_out4 = horiz_4tap_filt(src4, src4, mask0, mask1, filt_hz0, filt_hz1);
  DUP2_ARG2(__lsx_vpackev_b, hz_out1, hz_out0, hz_out3, hz_out2, out0, out1);
  DUP2_ARG2(__lsx_vpackev_b, hz_out2, hz_out1, hz_out4, hz_out3, out3, out4);

  filt = __lsx_vld(filter_vert, 0);
  DUP2_ARG2(__lsx_vreplvei_h, filt, 0, filt, 1, filt_vt0, filt_vt1);
  filt_vt2 = __lsx_vreplvei_h(filt, 2);

  for (loop_cnt = (height >> 2); loop_cnt--;) {
    DUP4_ARG2(__lsx_vldx, src, 0, src, src_stride, src, src_stride_x2, src,
              src_stride_x3, src5, src6, src7, src8);
    src += src_stride_x4;

    DUP4_ARG2(__lsx_vxori_b, src5, 128, src6, 128, src7, 128, src8, 128, src5,
              src6, src7, src8);
    hz_out5 = horiz_4tap_filt(src5, src5, mask0, mask1, filt_hz0, filt_hz1);
    out2 = __lsx_vpackev_b(hz_out5, hz_out4);
    tmp0 = dpadd_h3(out0, out1, out2, filt_vt0, filt_vt1, filt_vt2);

    hz_out6 = horiz_4tap_filt(src6, src6, mask0, mask1, filt_hz0, filt_hz1);
    out5 = __lsx_vpackev_b(hz_out6, hz_out5);
    tmp1 = dpadd_h3(out3, out4, out5, filt_vt0, filt_vt1, filt_vt2);

    hz_out7 = horiz_4tap_filt(src7, src7, mask0, mask1, filt_hz0, filt_hz1);
    out6 = __lsx_vpackev_b(hz_out7, hz_out6);
    tmp2 = dpadd_h3(out1, out2, out6, filt_vt0, filt_vt1, filt_vt2);

    hz_out8 = horiz_4tap_filt(src8, src8, mask0, mask1, filt_hz0, filt_hz1);
    out7 = __lsx_vpackev_b(hz_out8, hz_out7);
    tmp3 = dpadd_h3(out4, out5, out7, filt_vt0, filt_vt1, filt_vt2);
    DUP2_ARG3(__lsx_vssrarni_b_h, tmp1, tmp0, 7, tmp3, tmp2, 7, vec0, vec1);
    DUP2_ARG2(__lsx_vxori_b, vec0, 128, vec1, 128, vec0, vec1);
    __lsx_vstelm_d(vec0, dst, 0, 0);
    __lsx_vstelm_d(vec0, dst + dst_stride, 0, 1);
    __lsx_vstelm_d(vec1, dst + dst_stride_x2, 0, 0);
    __lsx_vstelm_d(vec1, dst + dst_stride_x3, 0, 1);
    dst += dst_stride_x4;
    hz_out4 = hz_out8;
    out0 = out2;
    out1 = out6;
    out3 = out5;
    out4 = out7;
  }
}

static void common_hv_4ht_6vt_16w_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                      uint8_t *RESTRICT dst, int32_t dst_stride,
                                      const int8_t *filter_horiz,
                                      const int8_t *filter_vert,
                                      int32_t height) {
  common_hv_4ht_6vt_8w_lsx(src, src_stride, dst, dst_stride, filter_horiz,
                           filter_vert, height);
  common_hv_4ht_6vt_8w_lsx(src + 8, src_stride, dst + 8, dst_stride,
                           filter_horiz, filter_vert, height);
}

typedef void (*PVp8SixtapPredictFunc1)(
    uint8_t *RESTRICT src, int32_t src_stride, uint8_t *RESTRICT dst,
    int32_t dst_stride, const int8_t *filter_horiz, const int8_t *filter_vert,
    int32_t height);

typedef void (*PVp8SixtapPredictFunc2)(uint8_t *RESTRICT src,
                                       int32_t src_stride,
                                       uint8_t *RESTRICT dst,
                                       int32_t dst_stride, const int8_t *filter,
                                       int32_t height);

void vp8_sixtap_predict4x4_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                               int32_t xoffset, int32_t yoffset,
                               uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_subpel_filters_lsx[xoffset - 1];
  const int8_t *v_filter = vp8_subpel_filters_lsx[yoffset - 1];

  static PVp8SixtapPredictFunc1 Predict4x4Funcs1[4] = {
    common_hv_6ht_6vt_4w_lsx,
    common_hv_6ht_4vt_4w_lsx,
    common_hv_4ht_6vt_4w_lsx,
    common_hv_4ht_4vt_4w_lsx,
  };

  static PVp8SixtapPredictFunc2 Predict4x4Funcs2[4] = { common_vt_6t_4w_lsx,
                                                        common_vt_4t_4w_lsx,
                                                        common_hz_6t_4w_lsx,
                                                        common_hz_4t_4w_lsx };
  if (yoffset < 8 && xoffset < 8) {
    if (yoffset) {
      if (xoffset) {
        switch (xoffset & 1) {
          case 0:
            switch (yoffset & 1) {
              case 0:
                Predict4x4Funcs1[0](src, src_stride, dst, dst_stride, h_filter,
                                    v_filter, 4);
                break;
              case 1:
                Predict4x4Funcs1[1](src, src_stride, dst, dst_stride, h_filter,
                                    v_filter + 1, 4);
                break;
            }
            break;

          case 1:
            switch (yoffset & 1) {
              case 0:
                Predict4x4Funcs1[2](src, src_stride, dst, dst_stride,
                                    h_filter + 1, v_filter, 4);
                break;

              case 1:
                Predict4x4Funcs1[3](src, src_stride, dst, dst_stride,
                                    h_filter + 1, v_filter + 1, 4);
                break;
            }
            break;
        }
      } else {
        switch (yoffset & 1) {
          case 0:
            Predict4x4Funcs2[0](src, src_stride, dst, dst_stride, v_filter, 4);
            break;

          case 1:
            Predict4x4Funcs2[1](src, src_stride, dst, dst_stride, v_filter + 1,
                                4);
            break;
        }
      }
    } else {
      switch (xoffset) {
        case 0: {
          __m128i tp0;

          tp0 = __lsx_vldrepl_w(src, 0);
          src += src_stride;
          __lsx_vstelm_w(tp0, dst, 0, 0);
          dst += dst_stride;
          tp0 = __lsx_vldrepl_w(src, 0);
          src += src_stride;
          __lsx_vstelm_w(tp0, dst, 0, 0);
          dst += dst_stride;
          tp0 = __lsx_vldrepl_w(src, 0);
          src += src_stride;
          __lsx_vstelm_w(tp0, dst, 0, 0);
          dst += dst_stride;
          tp0 = __lsx_vldrepl_w(src, 0);
          __lsx_vstelm_w(tp0, dst, 0, 0);

          break;
        }
        case 2:
        case 4:
        case 6:
          Predict4x4Funcs2[2](src, src_stride, dst, dst_stride, h_filter, 4);
          break;
      }
      switch (xoffset & 1) {
        case 1:
          Predict4x4Funcs2[3](src, src_stride, dst, dst_stride, h_filter + 1,
                              4);
          break;
      }
    }
  }
}

void vp8_sixtap_predict8x8_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                               int32_t xoffset, int32_t yoffset,
                               uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_subpel_filters_lsx[xoffset - 1];
  const int8_t *v_filter = vp8_subpel_filters_lsx[yoffset - 1];

  static PVp8SixtapPredictFunc1 Predict8x8Funcs1[4] = {
    common_hv_6ht_6vt_8w_lsx,
    common_hv_6ht_4vt_8w_lsx,
    common_hv_4ht_6vt_8w_lsx,
    common_hv_4ht_4vt_8w_lsx,
  };

  static PVp8SixtapPredictFunc2 Predict8x8Funcs2[4] = { common_vt_6t_8w_lsx,
                                                        common_vt_4t_8w_lsx,
                                                        common_hz_6t_8w_lsx,
                                                        common_hz_4t_8w_lsx };

  if (yoffset < 8 && xoffset < 8) {
    if (yoffset) {
      if (xoffset) {
        switch (xoffset & 1) {
          case 0:
            switch (yoffset & 1) {
              case 0:
                Predict8x8Funcs1[0](src, src_stride, dst, dst_stride, h_filter,
                                    v_filter, 8);
                break;

              case 1:
                Predict8x8Funcs1[1](src, src_stride, dst, dst_stride, h_filter,
                                    v_filter + 1, 8);
                break;
            }
            break;

          case 1:
            switch (yoffset & 1) {
              case 0:
                Predict8x8Funcs1[2](src, src_stride, dst, dst_stride,
                                    h_filter + 1, v_filter, 8);
                break;

              case 1:
                Predict8x8Funcs1[3](src, src_stride, dst, dst_stride,
                                    h_filter + 1, v_filter + 1, 8);
                break;
            }
            break;
        }
      } else {
        switch (yoffset & 1) {
          case 0:
            Predict8x8Funcs2[0](src, src_stride, dst, dst_stride, v_filter, 8);
            break;

          case 1:
            Predict8x8Funcs2[1](src, src_stride, dst, dst_stride, v_filter + 1,
                                8);
            break;
        }
      }
    } else {
      switch (xoffset & 1) {
        case 1:
          Predict8x8Funcs2[3](src, src_stride, dst, dst_stride, h_filter + 1,
                              8);
          break;
      }
      switch (xoffset) {
        case 0: vp8_copy_mem8x8(src, src_stride, dst, dst_stride); break;
        case 2:
        case 4:
        case 6:
          Predict8x8Funcs2[2](src, src_stride, dst, dst_stride, h_filter, 8);
          break;
      }
    }
  }
}

void vp8_sixtap_predict16x16_lsx(uint8_t *RESTRICT src, int32_t src_stride,
                                 int32_t xoffset, int32_t yoffset,
                                 uint8_t *RESTRICT dst, int32_t dst_stride) {
  const int8_t *h_filter = vp8_subpel_filters_lsx[xoffset - 1];
  const int8_t *v_filter = vp8_subpel_filters_lsx[yoffset - 1];

  static PVp8SixtapPredictFunc1 Predict16x16Funcs1[4] = {
    common_hv_6ht_6vt_16w_lsx,
    common_hv_6ht_4vt_16w_lsx,
    common_hv_4ht_6vt_16w_lsx,
    common_hv_4ht_4vt_16w_lsx,
  };

  static PVp8SixtapPredictFunc2 Predict16x16Funcs2[4] = {
    common_vt_6t_16w_lsx, common_vt_4t_16w_lsx, common_hz_6t_16w_lsx,
    common_hz_4t_16w_lsx
  };

  if (yoffset < 8 && xoffset < 8) {
    if (yoffset) {
      if (xoffset) {
        switch (xoffset & 1) {
          case 0:
            switch (yoffset & 1) {
              case 0:
                Predict16x16Funcs1[0](src, src_stride, dst, dst_stride,
                                      h_filter, v_filter, 16);
                break;

              case 1:
                Predict16x16Funcs1[1](src, src_stride, dst, dst_stride,
                                      h_filter, v_filter + 1, 16);
                break;
            }
            break;

          case 1:
            switch (yoffset & 1) {
              case 0:
                Predict16x16Funcs1[2](src, src_stride, dst, dst_stride,
                                      h_filter + 1, v_filter, 16);
                break;

              case 1:
                Predict16x16Funcs1[3](src, src_stride, dst, dst_stride,
                                      h_filter + 1, v_filter + 1, 16);
                break;
            }
            break;
        }
      } else {
        switch (yoffset & 1) {
          case 0:
            Predict16x16Funcs2[0](src, src_stride, dst, dst_stride, v_filter,
                                  16);
            break;

          case 1:
            Predict16x16Funcs2[1](src, src_stride, dst, dst_stride,
                                  v_filter + 1, 16);
            break;
        }
      }
    } else {
      switch (xoffset & 1) {
        case 1:
          Predict16x16Funcs2[3](src, src_stride, dst, dst_stride, h_filter + 1,
                                16);
          break;
      }
      switch (xoffset) {
        case 0: vp8_copy_mem16x16(src, src_stride, dst, dst_stride); break;
        case 2:
        case 4:
        case 6:
          Predict16x16Funcs2[2](src, src_stride, dst, dst_stride, h_filter, 16);
          break;
      }
    }
  }
}
