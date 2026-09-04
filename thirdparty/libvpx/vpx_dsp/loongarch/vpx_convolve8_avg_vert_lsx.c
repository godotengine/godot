/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/loongarch/vpx_convolve_lsx.h"

static void common_vt_8t_and_aver_dst_4w_lsx(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride,
                                             const int8_t *filter,
                                             int32_t height) {
  uint32_t loop_cnt = (height >> 2);
  uint8_t *dst_tmp = dst;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  __m128i reg0, reg1, reg2, reg3, reg4;
  __m128i filter0, filter1, filter2, filter3;
  __m128i out0, out1;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride + src_stride2;
  int32_t src_stride4 = src_stride2 << 1;
  uint8_t *src_tmp0 = (uint8_t *)src - src_stride3;

  DUP4_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filter, 4, filter, 6,
            filter0, filter1, filter2, filter3);
  src0 = __lsx_vld(src_tmp0, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp0, src_stride, src_tmp0, src_stride2, src1,
            src2);
  src3 = __lsx_vldx(src_tmp0, src_stride3);
  src_tmp0 += src_stride4;
  src4 = __lsx_vld(src_tmp0, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp0, src_stride, src_tmp0, src_stride2, src5,
            src6);
  src_tmp0 += src_stride3;
  DUP4_ARG2(__lsx_vilvl_b, src1, src0, src3, src2, src5, src4, src2, src1, tmp0,
            tmp1, tmp2, tmp3);
  DUP2_ARG2(__lsx_vilvl_b, src4, src3, src6, src5, tmp4, tmp5);
  DUP2_ARG2(__lsx_vilvl_d, tmp3, tmp0, tmp4, tmp1, reg0, reg1);
  reg2 = __lsx_vilvl_d(tmp5, tmp2);
  DUP2_ARG2(__lsx_vxori_b, reg0, 128, reg1, 128, reg0, reg1);
  reg2 = __lsx_vxori_b(reg2, 128);

  for (; loop_cnt--;) {
    src7 = __lsx_vld(src_tmp0, 0);
    DUP2_ARG2(__lsx_vldx, src_tmp0, src_stride, src_tmp0, src_stride2, src8,
              src9);
    src10 = __lsx_vldx(src_tmp0, src_stride3);
    src_tmp0 += src_stride4;
    src0 = __lsx_vldrepl_w(dst_tmp, 0);
    dst_tmp += dst_stride;
    src1 = __lsx_vldrepl_w(dst_tmp, 0);
    dst_tmp += dst_stride;
    src2 = __lsx_vldrepl_w(dst_tmp, 0);
    dst_tmp += dst_stride;
    src3 = __lsx_vldrepl_w(dst_tmp, 0);
    dst_tmp += dst_stride;
    DUP2_ARG2(__lsx_vilvl_w, src1, src0, src3, src2, src0, src1);
    src0 = __lsx_vilvl_d(src1, src0);
    DUP4_ARG2(__lsx_vilvl_b, src7, src6, src8, src7, src9, src8, src10, src9,
              tmp0, tmp1, tmp2, tmp3);
    DUP2_ARG2(__lsx_vilvl_d, tmp1, tmp0, tmp3, tmp2, reg3, reg4);
    DUP2_ARG2(__lsx_vxori_b, reg3, 128, reg4, 128, reg3, reg4);
    out0 = filt_8tap_dpadd_s_h(reg0, reg1, reg2, reg3, filter0, filter1,
                               filter2, filter3);
    out1 = filt_8tap_dpadd_s_h(reg1, reg2, reg3, reg4, filter0, filter1,
                               filter2, filter3);
    out0 = __lsx_vssrarni_b_h(out1, out0, 7);
    out0 = __lsx_vxori_b(out0, 128);
    out0 = __lsx_vavgr_bu(out0, src0);
    __lsx_vstelm_w(out0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_w(out0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_w(out0, dst, 0, 2);
    dst += dst_stride;
    __lsx_vstelm_w(out0, dst, 0, 3);
    dst += dst_stride;
    reg0 = reg2;
    reg1 = reg3;
    reg2 = reg4;
    src6 = src10;
  }
}

static void common_vt_8t_and_aver_dst_8w_lsx(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride,
                                             const int8_t *filter,
                                             int32_t height) {
  uint32_t loop_cnt = height >> 2;
  uint8_t *dst_tmp = dst;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5;
  __m128i filter0, filter1, filter2, filter3;
  __m128i out0, out1, out2, out3;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride + src_stride2;
  int32_t src_stride4 = src_stride2 << 1;
  uint8_t *src_tmp0 = (uint8_t *)src - src_stride3;

  DUP4_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filter, 4, filter, 6,
            filter0, filter1, filter2, filter3);

  src0 = __lsx_vld(src_tmp0, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp0, src_stride, src_tmp0, src_stride2, src1,
            src2);
  src3 = __lsx_vldx(src_tmp0, src_stride3);
  src_tmp0 += src_stride4;
  src4 = __lsx_vld(src_tmp0, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp0, src_stride, src_tmp0, src_stride2, src5,
            src6);
  src_tmp0 += src_stride3;
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  DUP2_ARG2(__lsx_vxori_b, src4, 128, src5, 128, src4, src5);
  src6 = __lsx_vxori_b(src6, 128);
  DUP4_ARG2(__lsx_vilvl_b, src1, src0, src3, src2, src5, src4, src2, src1, reg0,
            reg1, reg2, reg3);
  DUP2_ARG2(__lsx_vilvl_b, src4, src3, src6, src5, reg4, reg5);

  for (; loop_cnt--;) {
    src7 = __lsx_vld(src_tmp0, 0);
    DUP2_ARG2(__lsx_vldx, src_tmp0, src_stride, src_tmp0, src_stride2, src8,
              src9);
    src10 = __lsx_vldx(src_tmp0, src_stride3);
    src_tmp0 += src_stride4;
    src0 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    src1 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    src2 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    src3 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    DUP2_ARG2(__lsx_vilvl_d, src1, src0, src3, src2, src0, src1);
    DUP4_ARG2(__lsx_vxori_b, src7, 128, src8, 128, src9, 128, src10, 128, src7,
              src8, src9, src10);
    DUP4_ARG2(__lsx_vilvl_b, src7, src6, src8, src7, src9, src8, src10, src9,
              tmp0, tmp1, tmp2, tmp3);
    out0 = filt_8tap_dpadd_s_h(reg0, reg1, reg2, tmp0, filter0, filter1,
                               filter2, filter3);
    out1 = filt_8tap_dpadd_s_h(reg3, reg4, reg5, tmp1, filter0, filter1,
                               filter2, filter3);
    out2 = filt_8tap_dpadd_s_h(reg1, reg2, tmp0, tmp2, filter0, filter1,
                               filter2, filter3);
    out3 = filt_8tap_dpadd_s_h(reg4, reg5, tmp1, tmp3, filter0, filter1,
                               filter2, filter3);
    DUP2_ARG3(__lsx_vssrarni_b_h, out1, out0, 7, out3, out2, 7, out0, out1);
    DUP2_ARG2(__lsx_vxori_b, out0, 128, out1, 128, out0, out1);
    DUP2_ARG2(__lsx_vavgr_bu, out0, src0, out1, src1, out0, out1);
    __lsx_vstelm_d(out0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(out0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_d(out1, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(out1, dst, 0, 1);
    dst += dst_stride;
    reg0 = reg2;
    reg1 = tmp0;
    reg2 = tmp2;
    reg3 = reg5;
    reg4 = tmp1;
    reg5 = tmp3;
    src6 = src10;
  }
}

static void common_vt_8t_and_aver_dst_16w_mult_lsx(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    const int8_t *filter, int32_t height, int32_t width) {
  uint8_t *src_tmp;
  uint32_t cnt = width >> 4;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10;
  __m128i filter0, filter1, filter2, filter3;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5;
  __m128i reg6, reg7, reg8, reg9, reg10, reg11;
  __m128i tmp0, tmp1, tmp2, tmp3;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride + src_stride2;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t dst_stride2 = dst_stride << 1;
  int32_t dst_stride3 = dst_stride2 + dst_stride;
  int32_t dst_stride4 = dst_stride2 << 1;
  uint8_t *src_tmp0 = (uint8_t *)src - src_stride3;

  DUP4_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filter, 4, filter, 6,
            filter0, filter1, filter2, filter3);
  for (; cnt--;) {
    uint32_t loop_cnt = height >> 2;
    uint8_t *dst_reg = dst;

    src_tmp = src_tmp0;
    src0 = __lsx_vld(src_tmp, 0);
    DUP2_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src1,
              src2);
    src3 = __lsx_vldx(src_tmp, src_stride3);
    src_tmp += src_stride4;
    src4 = __lsx_vld(src_tmp, 0);
    DUP2_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src5,
              src6);
    src_tmp += src_stride3;
    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    DUP2_ARG2(__lsx_vxori_b, src4, 128, src5, 128, src4, src5);
    src6 = __lsx_vxori_b(src6, 128);
    DUP4_ARG2(__lsx_vilvl_b, src1, src0, src3, src2, src5, src4, src2, src1,
              reg0, reg1, reg2, reg3);
    DUP2_ARG2(__lsx_vilvl_b, src4, src3, src6, src5, reg4, reg5);
    DUP4_ARG2(__lsx_vilvh_b, src1, src0, src3, src2, src5, src4, src2, src1,
              reg6, reg7, reg8, reg9);
    DUP2_ARG2(__lsx_vilvh_b, src4, src3, src6, src5, reg10, reg11);
    for (; loop_cnt--;) {
      src7 = __lsx_vld(src_tmp, 0);
      DUP2_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src8,
                src9);
      src10 = __lsx_vldx(src_tmp, src_stride3);
      src_tmp += src_stride4;
      DUP4_ARG2(__lsx_vxori_b, src7, 128, src8, 128, src9, 128, src10, 128,
                src7, src8, src9, src10);
      DUP4_ARG2(__lsx_vilvl_b, src7, src6, src8, src7, src9, src8, src10, src9,
                src0, src1, src2, src3);
      DUP4_ARG2(__lsx_vilvh_b, src7, src6, src8, src7, src9, src8, src10, src9,
                src4, src5, src7, src8);
      tmp0 = filt_8tap_dpadd_s_h(reg0, reg1, reg2, src0, filter0, filter1,
                                 filter2, filter3);
      tmp1 = filt_8tap_dpadd_s_h(reg3, reg4, reg5, src1, filter0, filter1,
                                 filter2, filter3);
      tmp2 = filt_8tap_dpadd_s_h(reg6, reg7, reg8, src4, filter0, filter1,
                                 filter2, filter3);
      tmp3 = filt_8tap_dpadd_s_h(reg9, reg10, reg11, src5, filter0, filter1,
                                 filter2, filter3);
      DUP2_ARG3(__lsx_vssrarni_b_h, tmp2, tmp0, 7, tmp3, tmp1, 7, tmp0, tmp1);
      DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
      tmp2 = __lsx_vld(dst_reg, 0);
      tmp3 = __lsx_vldx(dst_reg, dst_stride);
      DUP2_ARG2(__lsx_vavgr_bu, tmp0, tmp2, tmp1, tmp3, tmp0, tmp1);
      __lsx_vst(tmp0, dst_reg, 0);
      __lsx_vstx(tmp1, dst_reg, dst_stride);
      tmp0 = filt_8tap_dpadd_s_h(reg1, reg2, src0, src2, filter0, filter1,
                                 filter2, filter3);
      tmp1 = filt_8tap_dpadd_s_h(reg4, reg5, src1, src3, filter0, filter1,
                                 filter2, filter3);
      tmp2 = filt_8tap_dpadd_s_h(reg7, reg8, src4, src7, filter0, filter1,
                                 filter2, filter3);
      tmp3 = filt_8tap_dpadd_s_h(reg10, reg11, src5, src8, filter0, filter1,
                                 filter2, filter3);
      DUP2_ARG3(__lsx_vssrarni_b_h, tmp2, tmp0, 7, tmp3, tmp1, 7, tmp0, tmp1);
      DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
      tmp2 = __lsx_vldx(dst_reg, dst_stride2);
      tmp3 = __lsx_vldx(dst_reg, dst_stride3);
      DUP2_ARG2(__lsx_vavgr_bu, tmp0, tmp2, tmp1, tmp3, tmp0, tmp1);
      __lsx_vstx(tmp0, dst_reg, dst_stride2);
      __lsx_vstx(tmp1, dst_reg, dst_stride3);
      dst_reg += dst_stride4;

      reg0 = reg2;
      reg1 = src0;
      reg2 = src2;
      reg3 = reg5;
      reg4 = src1;
      reg5 = src3;
      reg6 = reg8;
      reg7 = src4;
      reg8 = src7;
      reg9 = reg11;
      reg10 = src5;
      reg11 = src8;
      src6 = src10;
    }
    src_tmp0 += 16;
    dst += 16;
  }
}

static void common_vt_8t_and_aver_dst_16w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              const int8_t *filter,
                                              int32_t height) {
  common_vt_8t_and_aver_dst_16w_mult_lsx(src, src_stride, dst, dst_stride,
                                         filter, height, 16);
}

static void common_vt_8t_and_aver_dst_32w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              const int8_t *filter,
                                              int32_t height) {
  common_vt_8t_and_aver_dst_16w_mult_lsx(src, src_stride, dst, dst_stride,
                                         filter, height, 32);
}

static void common_vt_8t_and_aver_dst_64w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              const int8_t *filter,
                                              int32_t height) {
  common_vt_8t_and_aver_dst_16w_mult_lsx(src, src_stride, dst, dst_stride,
                                         filter, height, 64);
}

static void common_vt_2t_and_aver_dst_4x4_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  __m128i src0, src1, src2, src3, src4;
  __m128i dst0, dst1, dst2, dst3, out, filt0, src2110, src4332;
  __m128i src10_r, src32_r, src21_r, src43_r;
  __m128i tmp0, tmp1;
  uint8_t *dst_tmp = dst;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  filt0 = __lsx_vldrepl_h(filter, 0);
  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
  src3 = __lsx_vldx(src, src_stride3);
  src += src_stride4;
  src4 = __lsx_vld(src, 0);
  src += src_stride;

  dst0 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst1 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_w(dst_tmp, 0);
  dst0 = __lsx_vilvl_w(dst1, dst0);
  dst1 = __lsx_vilvl_w(dst3, dst2);
  dst0 = __lsx_vilvl_d(dst1, dst0);
  DUP4_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, src3, src2, src4, src3,
            src10_r, src21_r, src32_r, src43_r);
  DUP2_ARG2(__lsx_vilvl_d, src21_r, src10_r, src43_r, src32_r, src2110,
            src4332);
  DUP2_ARG2(__lsx_vdp2_h_bu, src2110, filt0, src4332, filt0, tmp0, tmp1);
  tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
  out = __lsx_vavgr_bu(tmp0, dst0);
  __lsx_vstelm_w(out, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(out, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(out, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(out, dst, 0, 3);
  dst += dst_stride;
}

static void common_vt_2t_and_aver_dst_4x8_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  __m128i dst0, dst1, dst2, dst3, dst4;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8, src87_r;
  __m128i src10_r, src32_r, src54_r, src76_r, src21_r, src43_r, src65_r;
  __m128i src2110, src4332, src6554, src8776, filt0;
  __m128i tmp0, tmp1, tmp2, tmp3;
  uint8_t *dst_tmp = dst;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  filt0 = __lsx_vldrepl_h(filter, 0);
  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
  src3 = __lsx_vldx(src, src_stride3);
  src += src_stride4;
  src4 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src5, src6);
  src7 = __lsx_vldx(src, src_stride3);
  src += src_stride4;
  src8 = __lsx_vld(src, 0);

  dst0 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst1 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst0 = __lsx_vilvl_w(dst1, dst0);
  dst1 = __lsx_vilvl_w(dst3, dst2);
  dst0 = __lsx_vilvl_d(dst1, dst0);

  dst1 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst4 = __lsx_vldrepl_w(dst_tmp, 0);
  dst1 = __lsx_vilvl_w(dst2, dst1);
  dst2 = __lsx_vilvl_w(dst4, dst3);
  dst1 = __lsx_vilvl_d(dst2, dst1);

  DUP4_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, src3, src2, src4, src3,
            src10_r, src21_r, src32_r, src43_r);
  DUP4_ARG2(__lsx_vilvl_b, src5, src4, src6, src5, src7, src6, src8, src7,
            src54_r, src65_r, src76_r, src87_r);
  DUP4_ARG2(__lsx_vilvl_d, src21_r, src10_r, src43_r, src32_r, src65_r, src54_r,
            src87_r, src76_r, src2110, src4332, src6554, src8776);
  DUP4_ARG2(__lsx_vdp2_h_bu, src2110, filt0, src4332, filt0, src6554, filt0,
            src8776, filt0, tmp0, tmp1, tmp2, tmp3);
  DUP2_ARG3(__lsx_vssrarni_bu_h, tmp1, tmp0, FILTER_BITS, tmp3, tmp2,
            FILTER_BITS, tmp0, tmp2);
  DUP2_ARG2(__lsx_vavgr_bu, tmp0, dst0, tmp2, dst1, tmp0, tmp2);
  __lsx_vstelm_w(tmp0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(tmp0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(tmp0, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(tmp0, dst, 0, 3);
  dst += dst_stride;

  __lsx_vstelm_w(tmp2, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(tmp2, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(tmp2, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(tmp2, dst, 0, 3);
}

static void common_vt_2t_and_aver_dst_4w_lsx(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  if (height == 4) {
    common_vt_2t_and_aver_dst_4x4_lsx(src, src_stride, dst, dst_stride, filter);
  } else if (height == 8) {
    common_vt_2t_and_aver_dst_4x8_lsx(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_vt_2t_and_aver_dst_8x4_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  __m128i src0, src1, src2, src3, src4;
  __m128i dst0, dst1, dst2, dst3, vec0, vec1, vec2, vec3, filt0;
  __m128i tmp0, tmp1, tmp2, tmp3;
  uint8_t *dst_tmp = dst;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  filt0 = __lsx_vldrepl_h(filter, 0);
  src0 = __lsx_vld(src, 0);
  DUP4_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src, src_stride3,
            src, src_stride4, src1, src2, src3, src4);
  dst0 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst1 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  DUP2_ARG2(__lsx_vilvl_d, dst1, dst0, dst3, dst2, dst0, dst1);
  DUP2_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, vec0, vec1);
  DUP2_ARG2(__lsx_vilvl_b, src3, src2, src4, src3, vec2, vec3);
  DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3, filt0,
            tmp0, tmp1, tmp2, tmp3);
  DUP2_ARG3(__lsx_vssrarni_bu_h, tmp1, tmp0, FILTER_BITS, tmp3, tmp2,
            FILTER_BITS, tmp0, tmp2);
  DUP2_ARG2(__lsx_vavgr_bu, tmp0, dst0, tmp2, dst1, tmp0, tmp2);
  __lsx_vstelm_d(tmp0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_d(tmp0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_d(tmp2, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_d(tmp2, dst, 0, 1);
}

static void common_vt_2t_and_aver_dst_8x8mult_lsx(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter, int32_t height) {
  uint32_t loop_cnt = (height >> 3);
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8;
  __m128i dst0, dst1, dst2, dst3, dst4, dst5;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, filt0;
  __m128i tmp0, tmp1, tmp2, tmp3;
  uint8_t *dst_tmp = dst;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;

  filt0 = __lsx_vldrepl_h(filter, 0);
  src0 = __lsx_vld(src, 0);
  src += src_stride;

  for (; loop_cnt--;) {
    src1 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src2, src3);
    src4 = __lsx_vldx(src, src_stride3);
    src += src_stride4;
    src5 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src6, src7);
    src8 = __lsx_vldx(src, src_stride3);
    src += src_stride4;

    dst0 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst1 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst2 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst3 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    DUP2_ARG2(__lsx_vilvl_d, dst1, dst0, dst3, dst2, dst0, dst1);

    dst2 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst3 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst4 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst5 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    DUP2_ARG2(__lsx_vilvl_d, dst3, dst2, dst5, dst4, dst2, dst3);

    DUP4_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, src3, src2, src4, src3,
              vec0, vec1, vec2, vec3);
    DUP4_ARG2(__lsx_vilvl_b, src5, src4, src6, src5, src7, src6, src8, src7,
              vec4, vec5, vec6, vec7);
    DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3,
              filt0, tmp0, tmp1, tmp2, tmp3);

    DUP2_ARG3(__lsx_vssrarni_bu_h, tmp1, tmp0, FILTER_BITS, tmp3, tmp2,
              FILTER_BITS, tmp0, tmp2);
    DUP2_ARG2(__lsx_vavgr_bu, tmp0, dst0, tmp2, dst1, tmp0, tmp2);
    __lsx_vstelm_d(tmp0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(tmp0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_d(tmp2, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(tmp2, dst, 0, 1);
    dst += dst_stride;

    DUP4_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, vec6, filt0, vec7,
              filt0, tmp0, tmp1, tmp2, tmp3);
    DUP2_ARG3(__lsx_vssrarni_bu_h, tmp1, tmp0, FILTER_BITS, tmp3, tmp2,
              FILTER_BITS, tmp0, tmp2);
    DUP2_ARG2(__lsx_vavgr_bu, tmp0, dst2, tmp2, dst3, tmp0, tmp2);
    __lsx_vstelm_d(tmp0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(tmp0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_d(tmp2, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(tmp2, dst, 0, 1);
    dst += dst_stride;

    src0 = src8;
  }
}

static void common_vt_2t_and_aver_dst_8w_lsx(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  if (height == 4) {
    common_vt_2t_and_aver_dst_8x4_lsx(src, src_stride, dst, dst_stride, filter);
  } else {
    common_vt_2t_and_aver_dst_8x8mult_lsx(src, src_stride, dst, dst_stride,
                                          filter, height);
  }
}

static void common_vt_2t_and_aver_dst_16w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt = (height >> 2);
  __m128i src0, src1, src2, src3, src4, dst0, dst1, dst2, dst3, filt0;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  __m128i tmp0, tmp1;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t dst_stride2 = dst_stride << 1;
  int32_t dst_stride3 = dst_stride2 + dst_stride;

  filt0 = __lsx_vldrepl_h(filter, 0);
  src0 = __lsx_vld(src, 0);
  src += src_stride;

  for (; loop_cnt--;) {
    src1 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src2, src3);
    src4 = __lsx_vldx(src, src_stride3);
    src += src_stride4;

    dst0 = __lsx_vld(dst, 0);
    DUP2_ARG2(__lsx_vldx, dst, dst_stride, dst, dst_stride2, dst1, dst2);
    dst3 = __lsx_vldx(dst, dst_stride3);

    DUP2_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, vec0, vec2);
    DUP2_ARG2(__lsx_vilvh_b, src1, src0, src2, src1, vec1, vec3);
    DUP2_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst0);
    __lsx_vst(tmp0, dst, 0);
    dst += dst_stride;

    DUP2_ARG2(__lsx_vilvl_b, src3, src2, src4, src3, vec4, vec6);
    DUP2_ARG2(__lsx_vilvh_b, src3, src2, src4, src3, vec5, vec7);
    DUP2_ARG2(__lsx_vdp2_h_bu, vec2, filt0, vec3, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst1);
    __lsx_vst(tmp0, dst, 0);
    dst += dst_stride;

    DUP2_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst2);
    __lsx_vst(tmp0, dst, 0);
    dst += dst_stride;

    DUP2_ARG2(__lsx_vdp2_h_bu, vec6, filt0, vec7, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst3);
    __lsx_vst(tmp0, dst, 0);
    dst += dst_stride;

    src0 = src4;
  }
}

static void common_vt_2t_and_aver_dst_32w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt = (height >> 2);
  uint8_t *src_tmp1;
  uint8_t *dst_tmp1;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, src8, src9;
  __m128i dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, filt0;
  __m128i tmp0, tmp1;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t dst_stride2 = dst_stride << 1;
  int32_t dst_stride3 = dst_stride2 + dst_stride;

  filt0 = __lsx_vldrepl_h(filter, 0);
  DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src5);
  src += src_stride;

  for (; loop_cnt--;) {
    src1 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src2, src3);
    src4 = __lsx_vldx(src, src_stride3);

    dst0 = __lsx_vld(dst, 0);
    DUP2_ARG2(__lsx_vldx, dst, dst_stride, dst, dst_stride2, dst1, dst2);
    dst3 = __lsx_vldx(dst, dst_stride3);

    DUP2_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, vec0, vec2);
    DUP2_ARG2(__lsx_vilvh_b, src1, src0, src2, src1, vec1, vec3);

    src_tmp1 = src + 16;
    src6 = __lsx_vld(src_tmp1, 0);
    DUP2_ARG2(__lsx_vldx, src_tmp1, src_stride, src_tmp1, src_stride2, src7,
              src8);
    src9 = __lsx_vldx(src_tmp1, src_stride3);

    dst_tmp1 = dst + 16;
    dst4 = __lsx_vld(dst_tmp1, 0);
    DUP2_ARG2(__lsx_vldx, dst_tmp1, dst_stride, dst_tmp1, dst_stride2, dst5,
              dst6);
    dst7 = __lsx_vldx(dst_tmp1, dst_stride3);
    src += src_stride4;

    DUP2_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst0);
    __lsx_vst(tmp0, dst, 0);

    DUP2_ARG2(__lsx_vdp2_h_bu, vec2, filt0, vec3, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst1);
    __lsx_vstx(tmp0, dst, dst_stride);

    DUP2_ARG2(__lsx_vilvl_b, src3, src2, src4, src3, vec4, vec6);
    DUP2_ARG2(__lsx_vilvh_b, src3, src2, src4, src3, vec5, vec7);
    DUP2_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst2);
    __lsx_vstx(tmp0, dst, dst_stride2);

    DUP2_ARG2(__lsx_vdp2_h_bu, vec6, filt0, vec7, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst3);
    __lsx_vstx(tmp0, dst, dst_stride3);

    DUP2_ARG2(__lsx_vilvl_b, src6, src5, src7, src6, vec0, vec2);
    DUP2_ARG2(__lsx_vilvh_b, src6, src5, src7, src6, vec1, vec3);
    DUP2_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst4);
    __lsx_vst(tmp0, dst, 16);

    DUP2_ARG2(__lsx_vdp2_h_bu, vec2, filt0, vec3, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst5);
    dst += dst_stride;
    __lsx_vst(tmp0, dst, 16);

    DUP2_ARG2(__lsx_vilvl_b, src8, src7, src9, src8, vec4, vec6);
    DUP2_ARG2(__lsx_vilvh_b, src8, src7, src9, src8, vec5, vec7);
    DUP2_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst6);
    dst += dst_stride;
    __lsx_vst(tmp0, dst, 16);

    DUP2_ARG2(__lsx_vdp2_h_bu, vec6, filt0, vec7, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst7);
    dst += dst_stride;
    __lsx_vst(tmp0, dst, 16);
    dst += dst_stride;

    src0 = src4;
    src5 = src9;
  }
}

static void common_vt_2t_and_aver_dst_64w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt = (height >> 1);
  int32_t src_stride2 = src_stride << 1;
  int32_t dst_stride2 = dst_stride << 1;
  uint8_t *src_tmp1;
  uint8_t *dst_tmp1;
  __m128i src0, src1, src2, src3, src4, src5;
  __m128i src6, src7, src8, src9, src10, src11, filt0;
  __m128i dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  __m128i tmp0, tmp1;

  filt0 = __lsx_vldrepl_h(filter, 0);
  DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src3, src6,
            src9);
  src += src_stride;

  for (; loop_cnt--;) {
    src2 = __lsx_vldx(src, src_stride);
    dst1 = __lsx_vldx(dst, dst_stride);
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src1, src4, src7,
              src10);
    DUP4_ARG2(__lsx_vld, dst, 0, dst, 16, dst, 32, dst, 48, dst0, dst2, dst4,
              dst6);
    src_tmp1 = (uint8_t *)src + 16;
    src5 = __lsx_vldx(src_tmp1, src_stride);
    src_tmp1 = src_tmp1 + 16;
    src8 = __lsx_vldx(src_tmp1, src_stride);
    src_tmp1 = src_tmp1 + 16;
    src11 = __lsx_vldx(src_tmp1, src_stride);

    dst_tmp1 = dst + 16;
    dst3 = __lsx_vldx(dst_tmp1, dst_stride);
    dst_tmp1 = dst + 32;
    dst5 = __lsx_vldx(dst_tmp1, dst_stride);
    dst_tmp1 = dst + 48;
    dst7 = __lsx_vldx(dst_tmp1, dst_stride);
    src += src_stride2;

    DUP2_ARG2(__lsx_vilvl_b, src1, src0, src2, src1, vec0, vec2);
    DUP2_ARG2(__lsx_vilvh_b, src1, src0, src2, src1, vec1, vec3);
    DUP2_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst0);
    __lsx_vst(tmp0, dst, 0);

    DUP2_ARG2(__lsx_vdp2_h_bu, vec2, filt0, vec3, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst1);
    __lsx_vstx(tmp0, dst, dst_stride);

    DUP2_ARG2(__lsx_vilvl_b, src4, src3, src5, src4, vec4, vec6);
    DUP2_ARG2(__lsx_vilvh_b, src4, src3, src5, src4, vec5, vec7);
    DUP2_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst2);
    __lsx_vst(tmp0, dst, 16);

    dst_tmp1 = dst + 16;
    DUP2_ARG2(__lsx_vdp2_h_bu, vec6, filt0, vec7, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst3);
    __lsx_vstx(tmp0, dst_tmp1, dst_stride);

    DUP2_ARG2(__lsx_vilvl_b, src7, src6, src8, src7, vec0, vec2);
    DUP2_ARG2(__lsx_vilvh_b, src7, src6, src8, src7, vec1, vec3);
    DUP2_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst4);
    __lsx_vst(tmp0, dst, 32);

    dst_tmp1 = dst_tmp1 + 16;
    DUP2_ARG2(__lsx_vdp2_h_bu, vec2, filt0, vec3, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst5);
    __lsx_vstx(tmp0, dst_tmp1, dst_stride);

    DUP2_ARG2(__lsx_vilvl_b, src10, src9, src11, src10, vec4, vec6);
    DUP2_ARG2(__lsx_vilvh_b, src10, src9, src11, src10, vec5, vec7);
    DUP2_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst6);
    __lsx_vst(tmp0, dst, 48);

    dst_tmp1 = dst_tmp1 + 16;
    DUP2_ARG2(__lsx_vdp2_h_bu, vec6, filt0, vec7, filt0, tmp0, tmp1);
    tmp0 = __lsx_vssrarni_bu_h(tmp1, tmp0, FILTER_BITS);
    tmp0 = __lsx_vavgr_bu(tmp0, dst7);
    __lsx_vstx(tmp0, dst_tmp1, dst_stride);
    dst += dst_stride2;

    src0 = src2;
    src3 = src5;
    src6 = src8;
    src9 = src11;
  }
}

void vpx_convolve8_avg_vert_lsx(const uint8_t *src, ptrdiff_t src_stride,
                                uint8_t *dst, ptrdiff_t dst_stride,
                                const InterpKernel *filter, int x0_q4,
                                int x_step_q4, int y0_q4, int y_step_q4, int w,
                                int h) {
  const int16_t *const filter_y = filter[y0_q4];
  int8_t cnt, filt_ver[8];

  assert(y_step_q4 == 16);
  assert(((const int32_t *)filter_y)[1] != 0x800000);

  for (cnt = 0; cnt < 8; ++cnt) {
    filt_ver[cnt] = filter_y[cnt];
  }

  if (vpx_get_filter_taps(filter_y) == 2) {
    switch (w) {
      case 4:
        common_vt_2t_and_aver_dst_4w_lsx(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, &filt_ver[3], h);
        break;
      case 8:
        common_vt_2t_and_aver_dst_8w_lsx(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, &filt_ver[3], h);
        break;
      case 16:
        common_vt_2t_and_aver_dst_16w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, &filt_ver[3], h);
        break;
      case 32:
        common_vt_2t_and_aver_dst_32w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, &filt_ver[3], h);
        break;
      case 64:
        common_vt_2t_and_aver_dst_64w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, &filt_ver[3], h);
        break;
      default:
        vpx_convolve8_avg_vert_c(src, src_stride, dst, dst_stride, filter,
                                 x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  } else {
    switch (w) {
      case 4:
        common_vt_8t_and_aver_dst_4w_lsx(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, filt_ver, h);
        break;
      case 8:
        common_vt_8t_and_aver_dst_8w_lsx(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, filt_ver, h);
        break;
      case 16:
        common_vt_8t_and_aver_dst_16w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, filt_ver, h);

        break;
      case 32:
        common_vt_8t_and_aver_dst_32w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, filt_ver, h);
        break;
      case 64:
        common_vt_8t_and_aver_dst_64w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, filt_ver, h);
        break;
      default:
        vpx_convolve8_avg_vert_c(src, src_stride, dst, dst_stride, filter,
                                 x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  }
}
