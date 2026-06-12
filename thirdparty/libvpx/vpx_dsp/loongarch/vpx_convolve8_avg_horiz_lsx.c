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

static const uint8_t mc_filt_mask_arr[16 * 3] = {
  /* 8 width cases */
  0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
  /* 4 width cases */
  0, 1, 1, 2, 2, 3, 3, 4, 16, 17, 17, 18, 18, 19, 19, 20,
  /* 4 width cases */
  8, 9, 9, 10, 10, 11, 11, 12, 24, 25, 25, 26, 26, 27, 27, 28
};

static void common_hz_8t_and_aver_dst_4x4_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  uint8_t *dst_tmp = dst;
  __m128i src0, src1, src2, src3;
  __m128i filter0, filter1, filter2, filter3;
  __m128i mask0, mask1, mask2, mask3;
  __m128i tmp0, tmp1;
  __m128i dst0, dst1, dst2, dst3;

  mask0 = __lsx_vld(mc_filt_mask_arr, 16);
  src -= 3;
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);
  mask3 = __lsx_vaddi_bu(mask0, 6);
  DUP4_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filter, 4, filter, 6,
            filter0, filter1, filter2, filter3);
  LSX_LD_4(src, src_stride, src0, src1, src2, src3);
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  HORIZ_8TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, mask3,
                             filter0, filter1, filter2, filter3, tmp0, tmp1);
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
  tmp0 = __lsx_vssrarni_b_h(tmp1, tmp0, 7);
  tmp0 = __lsx_vxori_b(tmp0, 128);
  dst0 = __lsx_vavgr_bu(tmp0, dst0);
  __lsx_vstelm_w(dst0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(dst0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(dst0, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(dst0, dst, 0, 3);
}

static void common_hz_8t_and_aver_dst_4x8_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  uint8_t *dst_tmp = dst;
  __m128i src0, src1, src2, src3, filter0, filter1, filter2, filter3;
  __m128i mask0, mask1, mask2, mask3, tmp0, tmp1, tmp2, tmp3;
  __m128i dst0, dst1;

  mask0 = __lsx_vld(mc_filt_mask_arr, 16);
  src -= 3;
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);
  mask3 = __lsx_vaddi_bu(mask0, 6);
  DUP4_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filter, 4, filter, 6,
            filter0, filter1, filter2, filter3);

  LSX_LD_4(src, src_stride, src0, src1, src2, src3);
  src += src_stride;
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  tmp0 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  tmp1 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  tmp2 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  tmp3 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  tmp0 = __lsx_vilvl_w(tmp1, tmp0);
  tmp1 = __lsx_vilvl_w(tmp3, tmp2);
  dst0 = __lsx_vilvl_d(tmp1, tmp0);

  tmp0 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  tmp1 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  tmp2 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  tmp3 = __lsx_vldrepl_w(dst_tmp, 0);
  tmp0 = __lsx_vilvl_w(tmp1, tmp0);
  tmp1 = __lsx_vilvl_w(tmp3, tmp2);
  dst1 = __lsx_vilvl_d(tmp1, tmp0);

  HORIZ_8TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, mask3,
                             filter0, filter1, filter2, filter3, tmp0, tmp1);
  LSX_LD_4(src, src_stride, src0, src1, src2, src3);
  DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
            src1, src2, src3);
  HORIZ_8TAP_4WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2, mask3,
                             filter0, filter1, filter2, filter3, tmp2, tmp3);
  DUP4_ARG3(__lsx_vssrarni_b_h, tmp0, tmp0, 7, tmp1, tmp1, 7, tmp2, tmp2, 7,
            tmp3, tmp3, 7, tmp0, tmp1, tmp2, tmp3);
  DUP2_ARG2(__lsx_vilvl_d, tmp1, tmp0, tmp3, tmp2, tmp0, tmp1);
  DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
  DUP2_ARG2(__lsx_vavgr_bu, tmp0, dst0, tmp1, dst1, dst0, dst1);
  __lsx_vstelm_w(dst0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(dst0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(dst0, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(dst0, dst, 0, 3);
  dst += dst_stride;
  __lsx_vstelm_w(dst1, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(dst1, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(dst1, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(dst1, dst, 0, 3);
}

static void common_hz_8t_and_aver_dst_4w_lsx(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  if (height == 4) {
    common_hz_8t_and_aver_dst_4x4_lsx(src, src_stride, dst, dst_stride, filter);
  } else if (height == 8) {
    common_hz_8t_and_aver_dst_4x8_lsx(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_hz_8t_and_aver_dst_8w_lsx(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  int32_t loop_cnt = height >> 2;
  uint8_t *dst_tmp = dst;
  __m128i src0, src1, src2, src3, filter0, filter1, filter2, filter3;
  __m128i mask0, mask1, mask2, mask3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i dst0, dst1, dst2, dst3;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;
  uint8_t *_src = (uint8_t *)src - 3;

  mask0 = __lsx_vld(mc_filt_mask_arr, 0);
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);
  mask3 = __lsx_vaddi_bu(mask0, 6);
  DUP4_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filter, 4, filter, 6,
            filter0, filter1, filter2, filter3);
  for (; loop_cnt--;) {
    src0 = __lsx_vld(_src, 0);
    DUP2_ARG2(__lsx_vldx, _src, src_stride, _src, src_stride2, src1, src2);
    src3 = __lsx_vldx(_src, src_stride3);
    _src += src_stride4;
    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    HORIZ_8TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2,
                               mask3, filter0, filter1, filter2, filter3, tmp0,
                               tmp1, tmp2, tmp3);
    dst0 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst1 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst2 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst3 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    DUP2_ARG2(__lsx_vilvl_d, dst1, dst0, dst3, dst2, dst0, dst1);
    DUP2_ARG3(__lsx_vssrarni_b_h, tmp1, tmp0, 7, tmp3, tmp2, 7, tmp0, tmp1);
    DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
    DUP2_ARG2(__lsx_vavgr_bu, tmp0, dst0, tmp1, dst1, dst0, dst1);
    __lsx_vstelm_d(dst0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(dst0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_d(dst1, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(dst1, dst, 0, 1);
    dst += dst_stride;
  }
}

static void common_hz_8t_and_aver_dst_16w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  int32_t loop_cnt = height >> 1;
  int32_t dst_stride2 = dst_stride << 1;
  uint8_t *dst_tmp = dst;
  __m128i src0, src1, src2, src3, filter0, filter1, filter2, filter3;
  __m128i mask0, mask1, mask2, mask3, dst0, dst1, dst2, dst3;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;

  mask0 = __lsx_vld(mc_filt_mask_arr, 0);
  src -= 3;
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);
  mask3 = __lsx_vaddi_bu(mask0, 6);
  DUP4_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filter, 4, filter, 6,
            filter0, filter1, filter2, filter3);

  for (; loop_cnt--;) {
    DUP2_ARG2(__lsx_vld, src, 0, src, 8, src0, src1);
    src += src_stride;
    DUP2_ARG2(__lsx_vld, src, 0, src, 8, src2, src3);
    src += src_stride;
    dst0 = __lsx_vld(dst_tmp, 0);
    dst1 = __lsx_vldx(dst_tmp, dst_stride);
    dst_tmp += dst_stride2;
    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, mask0, src1, src1, mask0, src2, src2,
              mask0, src3, src3, mask0, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, mask1, src1, src1, mask1, src2, src2,
              mask1, src3, src3, mask1, tmp4, tmp5, tmp6, tmp7);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, mask2, src1, src1, mask2, src2, src2,
              mask2, src3, src3, mask2, tmp8, tmp9, tmp10, tmp11);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, mask3, src1, src1, mask3, src2, src2,
              mask3, src3, src3, mask3, tmp12, tmp13, tmp14, tmp15);
    DUP4_ARG2(__lsx_vdp2_h_b, tmp0, filter0, tmp1, filter0, tmp2, filter0, tmp3,
              filter0, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vdp2_h_b, tmp8, filter2, tmp9, filter2, tmp10, filter2,
              tmp11, filter2, tmp8, tmp9, tmp10, tmp11);
    DUP4_ARG3(__lsx_vdp2add_h_b, tmp0, tmp4, filter1, tmp1, tmp5, filter1, tmp2,
              tmp6, filter1, tmp3, tmp7, filter1, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG3(__lsx_vdp2add_h_b, tmp8, tmp12, filter3, tmp9, tmp13, filter3,
              tmp10, tmp14, filter3, tmp11, tmp15, filter3, tmp4, tmp5, tmp6,
              tmp7);
    DUP4_ARG2(__lsx_vsadd_h, tmp0, tmp4, tmp1, tmp5, tmp2, tmp6, tmp3, tmp7,
              tmp0, tmp1, tmp2, tmp3);
    DUP2_ARG3(__lsx_vssrarni_b_h, tmp1, tmp0, 7, tmp3, tmp2, 7, dst2, dst3);
    DUP2_ARG2(__lsx_vxori_b, dst2, 128, dst3, 128, dst2, dst3);
    DUP2_ARG2(__lsx_vavgr_bu, dst0, dst2, dst1, dst3, dst0, dst1);
    __lsx_vst(dst0, dst, 0);
    __lsx_vstx(dst1, dst, dst_stride);
    dst += dst_stride2;
  }
}

static void common_hz_8t_and_aver_dst_32w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt = height;
  uint8_t *dst_tmp = dst;
  __m128i src0, src1, src2, src3, filter0, filter1, filter2, filter3;
  __m128i mask0, mask1, mask2, mask3, dst0, dst1;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
  __m128i shuff = { 0x0F0E0D0C0B0A0908, 0x1716151413121110 };

  mask0 = __lsx_vld(mc_filt_mask_arr, 0);
  src -= 3;
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);
  mask3 = __lsx_vaddi_bu(mask0, 6);
  DUP4_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filter, 4, filter, 6,
            filter0, filter1, filter2, filter3);
  for (; loop_cnt--;) {
    DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src2);
    src3 = __lsx_vld(src, 24);
    src1 = __lsx_vshuf_b(src2, src0, shuff);
    src += src_stride;
    DUP2_ARG2(__lsx_vld, dst_tmp, 0, dst, 16, dst0, dst1);
    dst_tmp += dst_stride;
    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, mask0, src1, src1, mask0, src2, src2,
              mask0, src3, src3, mask0, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, mask1, src1, src1, mask1, src2, src2,
              mask1, src3, src3, mask1, tmp4, tmp5, tmp6, tmp7);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, mask2, src1, src1, mask2, src2, src2,
              mask2, src3, src3, mask2, tmp8, tmp9, tmp10, tmp11);
    DUP4_ARG3(__lsx_vshuf_b, src0, src0, mask3, src1, src1, mask3, src2, src2,
              mask3, src3, src3, mask3, tmp12, tmp13, tmp14, tmp15);
    DUP4_ARG2(__lsx_vdp2_h_b, tmp0, filter0, tmp1, filter0, tmp2, filter0, tmp3,
              filter0, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG2(__lsx_vdp2_h_b, tmp8, filter2, tmp9, filter2, tmp10, filter2,
              tmp11, filter2, tmp8, tmp9, tmp10, tmp11);
    DUP4_ARG3(__lsx_vdp2add_h_b, tmp0, tmp4, filter1, tmp1, tmp5, filter1, tmp2,
              tmp6, filter1, tmp3, tmp7, filter1, tmp0, tmp1, tmp2, tmp3);
    DUP4_ARG3(__lsx_vdp2add_h_b, tmp8, tmp12, filter3, tmp9, tmp13, filter3,
              tmp10, tmp14, filter3, tmp11, tmp15, filter3, tmp4, tmp5, tmp6,
              tmp7);
    DUP4_ARG2(__lsx_vsadd_h, tmp0, tmp4, tmp1, tmp5, tmp2, tmp6, tmp3, tmp7,
              tmp0, tmp1, tmp2, tmp3);
    DUP2_ARG3(__lsx_vssrarni_b_h, tmp1, tmp0, 7, tmp3, tmp2, 7, tmp0, tmp1);
    DUP2_ARG2(__lsx_vxori_b, tmp0, 128, tmp1, 128, tmp0, tmp1);
    DUP2_ARG2(__lsx_vavgr_bu, dst0, tmp0, dst1, tmp1, dst0, dst1);
    __lsx_vst(dst0, dst, 0);
    __lsx_vst(dst1, dst, 16);
    dst += dst_stride;
  }
}

static void common_hz_8t_and_aver_dst_64w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  int32_t loop_cnt = height;
  __m128i src0, src1, src2, src3;
  __m128i filter0, filter1, filter2, filter3;
  __m128i mask0, mask1, mask2, mask3;
  __m128i out0, out1, out2, out3, dst0, dst1;
  __m128i shuff = { 0x0F0E0D0C0B0A0908, 0x1716151413121110 };

  mask0 = __lsx_vld(mc_filt_mask_arr, 0);
  src -= 3;
  DUP2_ARG2(__lsx_vaddi_bu, mask0, 2, mask0, 4, mask1, mask2);
  mask3 = __lsx_vaddi_bu(mask0, 6);
  DUP4_ARG2(__lsx_vldrepl_h, filter, 0, filter, 2, filter, 4, filter, 6,
            filter0, filter1, filter2, filter3);
  for (; loop_cnt--;) {
    DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src2);
    src3 = __lsx_vld(src, 24);
    src1 = __lsx_vshuf_b(src2, src0, shuff);
    DUP2_ARG2(__lsx_vld, dst, 0, dst, 16, dst0, dst1);
    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    HORIZ_8TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2,
                               mask3, filter0, filter1, filter2, filter3, out0,
                               out1, out2, out3);
    DUP2_ARG3(__lsx_vssrarni_b_h, out1, out0, 7, out3, out2, 7, out0, out1);
    DUP2_ARG2(__lsx_vxori_b, out0, 128, out1, 128, out0, out1);
    DUP2_ARG2(__lsx_vavgr_bu, out0, dst0, out1, dst1, out0, out1);
    __lsx_vst(out0, dst, 0);
    __lsx_vst(out1, dst, 16);

    DUP2_ARG2(__lsx_vld, src, 32, src, 48, src0, src2);
    src3 = __lsx_vld(src, 56);
    src1 = __lsx_vshuf_b(src2, src0, shuff);
    DUP2_ARG2(__lsx_vld, dst, 32, dst, 48, dst0, dst1);
    DUP4_ARG2(__lsx_vxori_b, src0, 128, src1, 128, src2, 128, src3, 128, src0,
              src1, src2, src3);
    HORIZ_8TAP_8WID_4VECS_FILT(src0, src1, src2, src3, mask0, mask1, mask2,
                               mask3, filter0, filter1, filter2, filter3, out0,
                               out1, out2, out3);
    DUP2_ARG3(__lsx_vssrarni_b_h, out1, out0, 7, out3, out2, 7, out0, out1);
    DUP2_ARG2(__lsx_vxori_b, out0, 128, out1, 128, out0, out1);
    DUP2_ARG2(__lsx_vavgr_bu, out0, dst0, out1, dst1, out0, out1);
    __lsx_vst(out0, dst, 32);
    __lsx_vst(out1, dst, 48);
    src += src_stride;
    dst += dst_stride;
  }
}

static void common_hz_2t_and_aver_dst_4x4_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  __m128i src0, src1, src2, src3, mask;
  __m128i dst0, dst1, dst2, dst3, vec0, vec1, filt0;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  uint8_t *dst_tmp = dst;

  mask = __lsx_vld(mc_filt_mask_arr, 16);

  /* rearranging filter */
  filt0 = __lsx_vldrepl_h(filter, 0);

  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
  src3 = __lsx_vldx(src, src_stride3);
  dst0 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst1 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  DUP2_ARG2(__lsx_vilvl_w, dst1, dst0, dst3, dst2, dst0, dst1);
  dst0 = __lsx_vilvl_d(dst1, dst0);
  DUP2_ARG3(__lsx_vshuf_b, src1, src0, mask, src3, src2, mask, vec0, vec1);
  DUP2_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec0, vec1);
  vec0 = __lsx_vssrarni_bu_h(vec1, vec0, FILTER_BITS);
  vec0 = __lsx_vavgr_bu(vec0, dst0);
  __lsx_vstelm_w(vec0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(vec0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(vec0, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(vec0, dst, 0, 3);
}

static void common_hz_2t_and_aver_dst_4x8_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, mask;
  __m128i filt0, vec0, vec1, vec2, vec3, res0, res1, res2, res3;
  __m128i dst0, dst1, dst2, dst3, dst4;
  __m128i vec4, vec5, vec6, vec7;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;
  uint8_t *src_tmp1 = (uint8_t *)src + src_stride4;
  uint8_t *dst_tmp = dst;

  mask = __lsx_vld(mc_filt_mask_arr, 16);

  /* rearranging filter */
  filt0 = __lsx_vldrepl_h(filter, 0);

  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
  src3 = __lsx_vldx(src, src_stride3);

  src4 = __lsx_vld(src_tmp1, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp1, src_stride, src_tmp1, src_stride2, src5,
            src6);
  src7 = __lsx_vldx(src_tmp1, src_stride3);

  dst0 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst1 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  DUP2_ARG2(__lsx_vilvl_w, dst1, dst0, dst3, dst2, dst0, dst1);
  dst0 = __lsx_vilvl_d(dst1, dst0);

  dst1 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst4 = __lsx_vldrepl_w(dst_tmp, 0);
  dst_tmp += dst_stride;
  DUP2_ARG2(__lsx_vilvl_w, dst2, dst1, dst4, dst3, dst1, dst2);
  dst1 = __lsx_vilvl_d(dst2, dst1);

  DUP2_ARG3(__lsx_vshuf_b, src1, src0, mask, src3, src2, mask, vec0, vec1);
  DUP2_ARG3(__lsx_vshuf_b, src5, src4, mask, src7, src6, mask, vec2, vec3);
  DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3, filt0,
            vec4, vec5, vec6, vec7);
  DUP4_ARG3(__lsx_vssrarni_bu_h, vec4, vec4, FILTER_BITS, vec5, vec5,
            FILTER_BITS, vec6, vec6, FILTER_BITS, vec7, vec7, FILTER_BITS, res0,
            res1, res2, res3);
  DUP2_ARG2(__lsx_vilvl_d, res1, res0, res3, res2, res0, res2);
  DUP2_ARG2(__lsx_vavgr_bu, res0, dst0, res2, dst1, res0, res2);

  __lsx_vstelm_w(res0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(res0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(res0, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(res0, dst, 0, 3);
  dst += dst_stride;

  __lsx_vstelm_w(res2, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_w(res2, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_w(res2, dst, 0, 2);
  dst += dst_stride;
  __lsx_vstelm_w(res2, dst, 0, 3);
  dst += dst_stride;
}

static void common_hz_2t_and_aver_dst_4w_lsx(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  if (height == 4) {
    common_hz_2t_and_aver_dst_4x4_lsx(src, src_stride, dst, dst_stride, filter);
  } else if (height == 8) {
    common_hz_2t_and_aver_dst_4x8_lsx(src, src_stride, dst, dst_stride, filter);
  }
}

static void common_hz_2t_and_aver_dst_8x4_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter) {
  __m128i src0, src1, src2, src3, mask;
  __m128i filt0, dst0, dst1, dst2, dst3;
  __m128i vec0, vec1, vec2, vec3;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  uint8_t *dst_tmp = dst;

  mask = __lsx_vld(mc_filt_mask_arr, 0);

  /* rearranging filter */
  filt0 = __lsx_vldrepl_h(filter, 0);

  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
  src3 = __lsx_vldx(src, src_stride3);
  DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask, src1, src1, mask, vec0, vec1);
  DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask, src3, src3, mask, vec2, vec3);
  DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3, filt0,
            vec0, vec1, vec2, vec3);
  DUP2_ARG3(__lsx_vssrarni_bu_h, vec1, vec0, FILTER_BITS, vec3, vec2,
            FILTER_BITS, vec0, vec1);
  dst0 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst1 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;

  DUP2_ARG2(__lsx_vilvl_d, dst1, dst0, dst3, dst2, dst0, dst1);
  DUP2_ARG2(__lsx_vavgr_bu, vec0, dst0, vec1, dst1, vec0, vec1);
  __lsx_vstelm_d(vec0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_d(vec0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_d(vec1, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_d(vec1, dst, 0, 1);
}

static void common_hz_2t_and_aver_dst_8x8mult_lsx(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int8_t *filter, int32_t height) {
  __m128i src0, src1, src2, src3, mask;
  __m128i filt0, dst0, dst1, dst2, dst3;
  __m128i vec0, vec1, vec2, vec3;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  uint8_t *dst_tmp = dst;

  mask = __lsx_vld(mc_filt_mask_arr, 0);

  /* rearranging filter */
  filt0 = __lsx_vldrepl_h(filter, 0);

  LSX_LD_4(src, src_stride, src0, src1, src2, src3);
  src += src_stride;

  DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask, src1, src1, mask, vec0, vec1);
  DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask, src3, src3, mask, vec2, vec3);
  DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3, filt0,
            vec0, vec1, vec2, vec3);
  DUP2_ARG3(__lsx_vssrarni_bu_h, vec1, vec0, FILTER_BITS, vec3, vec2,
            FILTER_BITS, vec0, vec2);
  dst0 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst1 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  DUP2_ARG2(__lsx_vilvl_d, dst1, dst0, dst3, dst2, dst0, dst1);

  DUP2_ARG2(__lsx_vavgr_bu, vec0, dst0, vec2, dst1, vec0, vec2);
  __lsx_vstelm_d(vec0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_d(vec0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_d(vec2, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_d(vec2, dst, 0, 1);
  dst += dst_stride;

  LSX_LD_4(src, src_stride, src0, src1, src2, src3);
  src += src_stride;
  DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask, src1, src1, mask, vec0, vec1);
  DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask, src3, src3, mask, vec2, vec3);

  DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3, filt0,
            vec0, vec1, vec2, vec3);
  DUP2_ARG3(__lsx_vssrarni_bu_h, vec1, vec0, FILTER_BITS, vec3, vec2,
            FILTER_BITS, vec0, vec2);
  dst0 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst1 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst2 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  dst3 = __lsx_vldrepl_d(dst_tmp, 0);
  dst_tmp += dst_stride;
  DUP2_ARG2(__lsx_vilvl_d, dst1, dst0, dst3, dst2, dst0, dst1);
  DUP2_ARG2(__lsx_vavgr_bu, vec0, dst0, vec2, dst1, vec0, vec2);
  __lsx_vstelm_d(vec0, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_d(vec0, dst, 0, 1);
  dst += dst_stride;
  __lsx_vstelm_d(vec2, dst, 0, 0);
  dst += dst_stride;
  __lsx_vstelm_d(vec2, dst, 0, 1);
  dst += dst_stride;

  if (height == 16) {
    LSX_LD_4(src, src_stride, src0, src1, src2, src3);
    src += src_stride;

    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask, src1, src1, mask, vec0, vec1);
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask, src3, src3, mask, vec2, vec3);
    DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3,
              filt0, vec0, vec1, vec2, vec3);
    DUP2_ARG3(__lsx_vssrarni_bu_h, vec1, vec0, FILTER_BITS, vec3, vec2,
              FILTER_BITS, vec0, vec2);
    dst0 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst1 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst2 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst3 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    DUP2_ARG2(__lsx_vilvl_d, dst1, dst0, dst3, dst2, dst0, dst1);
    DUP2_ARG2(__lsx_vavgr_bu, vec0, dst0, vec2, dst1, vec0, vec2);
    __lsx_vstelm_d(vec0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(vec0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_d(vec2, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(vec2, dst, 0, 1);
    dst += dst_stride;

    src0 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src1, src2);
    src3 = __lsx_vldx(src, src_stride3);
    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask, src1, src1, mask, vec0, vec1);
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask, src3, src3, mask, vec2, vec3);
    DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3,
              filt0, vec0, vec1, vec2, vec3);
    DUP2_ARG3(__lsx_vssrarni_bu_h, vec1, vec0, FILTER_BITS, vec3, vec2,
              FILTER_BITS, vec0, vec2);
    dst0 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst1 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst2 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    dst3 = __lsx_vldrepl_d(dst_tmp, 0);
    dst_tmp += dst_stride;
    DUP2_ARG2(__lsx_vilvl_d, dst1, dst0, dst3, dst2, dst0, dst1);
    DUP2_ARG2(__lsx_vavgr_bu, vec0, dst0, vec2, dst1, vec0, vec2);
    __lsx_vstelm_d(vec0, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(vec0, dst, 0, 1);
    dst += dst_stride;
    __lsx_vstelm_d(vec2, dst, 0, 0);
    dst += dst_stride;
    __lsx_vstelm_d(vec2, dst, 0, 1);
    dst += dst_stride;
  }
}

static void common_hz_2t_and_aver_dst_8w_lsx(const uint8_t *src,
                                             int32_t src_stride, uint8_t *dst,
                                             int32_t dst_stride, int8_t *filter,
                                             int32_t height) {
  if (height == 4) {
    common_hz_2t_and_aver_dst_8x4_lsx(src, src_stride, dst, dst_stride, filter);
  } else {
    common_hz_2t_and_aver_dst_8x8mult_lsx(src, src_stride, dst, dst_stride,
                                          filter, height);
  }
}

static void common_hz_2t_and_aver_dst_16w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt = (height >> 2) - 1;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, mask;
  __m128i filt0, dst0;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  __m128i res0, res1, res2, res3, res4, res5, res6, res7;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;
  uint8_t *src_tmp1 = (uint8_t *)src + 8;

  mask = __lsx_vld(mc_filt_mask_arr, 0);

  /* rearranging filter */
  filt0 = __lsx_vldrepl_h(filter, 0);

  src0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src2, src4);
  src6 = __lsx_vldx(src, src_stride3);
  src += src_stride4;

  src1 = __lsx_vld(src_tmp1, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp1, src_stride, src_tmp1, src_stride2, src3,
            src5);
  src7 = __lsx_vldx(src_tmp1, src_stride3);
  src_tmp1 += src_stride4;

  DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask, src1, src1, mask, vec0, vec1);
  DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask, src3, src3, mask, vec2, vec3);
  DUP2_ARG3(__lsx_vshuf_b, src4, src4, mask, src5, src5, mask, vec4, vec5);
  DUP2_ARG3(__lsx_vshuf_b, src6, src6, mask, src7, src7, mask, vec6, vec7);

  DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3, filt0,
            res0, res1, res2, res3);
  DUP4_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, vec6, filt0, vec7, filt0,
            res4, res5, res6, res7);
  DUP4_ARG3(__lsx_vssrarni_bu_h, res1, res0, FILTER_BITS, res3, res2,
            FILTER_BITS, res5, res4, FILTER_BITS, res7, res6, FILTER_BITS, res0,
            res2, res4, res6);
  dst0 = __lsx_vld(dst, 0);
  res0 = __lsx_vavgr_bu(res0, dst0);
  __lsx_vst(res0, dst, 0);
  dst += dst_stride;

  dst0 = __lsx_vld(dst, 0);
  res2 = __lsx_vavgr_bu(res2, dst0);
  __lsx_vst(res2, dst, 0);
  dst += dst_stride;

  dst0 = __lsx_vld(dst, 0);
  res4 = __lsx_vavgr_bu(res4, dst0);
  __lsx_vst(res4, dst, 0);
  dst += dst_stride;

  dst0 = __lsx_vld(dst, 0);
  res6 = __lsx_vavgr_bu(res6, dst0);
  __lsx_vst(res6, dst, 0);
  dst += dst_stride;

  for (; loop_cnt--;) {
    src0 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src2, src4);
    src6 = __lsx_vldx(src, src_stride3);
    src += src_stride4;

    src1 = __lsx_vld(src_tmp1, 0);
    DUP2_ARG2(__lsx_vldx, src_tmp1, src_stride, src_tmp1, src_stride2, src3,
              src5);
    src7 = __lsx_vldx(src_tmp1, src_stride3);
    src_tmp1 += src_stride4;

    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask, src1, src1, mask, vec0, vec1);
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask, src3, src3, mask, vec2, vec3);
    DUP2_ARG3(__lsx_vshuf_b, src4, src4, mask, src5, src5, mask, vec4, vec5);
    DUP2_ARG3(__lsx_vshuf_b, src6, src6, mask, src7, src7, mask, vec6, vec7);
    DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3,
              filt0, res0, res1, res2, res3);
    DUP4_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, vec6, filt0, vec7,
              filt0, res4, res5, res6, res7);

    DUP4_ARG3(__lsx_vssrarni_bu_h, res1, res0, FILTER_BITS, res3, res2,
              FILTER_BITS, res5, res4, FILTER_BITS, res7, res6, FILTER_BITS,
              res0, res2, res4, res6);
    dst0 = __lsx_vld(dst, 0);
    res0 = __lsx_vavgr_bu(res0, dst0);
    __lsx_vst(res0, dst, 0);
    dst += dst_stride;

    dst0 = __lsx_vld(dst, 0);
    res2 = __lsx_vavgr_bu(res2, dst0);
    __lsx_vst(res2, dst, 0);
    dst += dst_stride;

    dst0 = __lsx_vld(dst, 0);
    res4 = __lsx_vavgr_bu(res4, dst0);
    __lsx_vst(res4, dst, 0);
    dst += dst_stride;

    dst0 = __lsx_vld(dst, 0);
    res6 = __lsx_vavgr_bu(res6, dst0);
    __lsx_vst(res6, dst, 0);
    dst += dst_stride;
  }
}

static void common_hz_2t_and_aver_dst_32w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt = (height >> 1);
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, mask;
  __m128i filt0, dst0, dst1;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  __m128i res0, res1, res2, res3, res4, res5, res6, res7;
  __m128i shuff = { 0x0F0E0D0C0B0A0908, 0x1716151413121110 };

  mask = __lsx_vld(mc_filt_mask_arr, 0);

  /* rearranging filter */
  filt0 = __lsx_vldrepl_h(filter, 0);

  for (; loop_cnt--;) {
    src0 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vld, src, 16, src, 24, src2, src3);
    src1 = __lsx_vshuf_b(src2, src0, shuff);
    src += src_stride;
    src4 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vld, src, 16, src, 24, src6, src7);
    src5 = __lsx_vshuf_b(src6, src4, shuff);
    src += src_stride;

    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask, src1, src1, mask, vec0, vec1);
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask, src3, src3, mask, vec2, vec3);
    DUP2_ARG3(__lsx_vshuf_b, src4, src4, mask, src5, src5, mask, vec4, vec5);
    DUP2_ARG3(__lsx_vshuf_b, src6, src6, mask, src7, src7, mask, vec6, vec7);

    DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3,
              filt0, res0, res1, res2, res3);
    DUP4_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, vec6, filt0, vec7,
              filt0, res4, res5, res6, res7);
    DUP4_ARG3(__lsx_vssrarni_bu_h, res1, res0, FILTER_BITS, res3, res2,
              FILTER_BITS, res5, res4, FILTER_BITS, res7, res6, FILTER_BITS,
              res0, res2, res4, res6);

    DUP2_ARG2(__lsx_vld, dst, 0, dst, 16, dst0, dst1);
    res0 = __lsx_vavgr_bu(res0, dst0);
    __lsx_vst(res0, dst, 0);
    res2 = __lsx_vavgr_bu(res2, dst1);
    __lsx_vst(res2, dst, 16);
    dst += dst_stride;

    DUP2_ARG2(__lsx_vld, dst, 0, dst, 16, dst0, dst1);
    res4 = __lsx_vavgr_bu(res4, dst0);
    __lsx_vst(res4, dst, 0);
    res6 = __lsx_vavgr_bu(res6, dst1);
    __lsx_vst(res6, dst, 16);
    dst += dst_stride;
  }
}

static void common_hz_2t_and_aver_dst_64w_lsx(const uint8_t *src,
                                              int32_t src_stride, uint8_t *dst,
                                              int32_t dst_stride,
                                              int8_t *filter, int32_t height) {
  uint32_t loop_cnt = height;
  __m128i src0, src1, src2, src3, src4, src5, src6, src7, mask;
  __m128i filt0, dst0, dst1, dst2, dst3;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  __m128i out0, out1, out2, out3, out4, out5, out6, out7;
  __m128i shuff = { 0x0F0E0D0C0B0A0908, 0x1716151413121110 };

  mask = __lsx_vld(mc_filt_mask_arr, 0);

  /* rearranging filter */
  filt0 = __lsx_vldrepl_h(filter, 0);

  for (; loop_cnt--;) {
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src2, src4,
              src6);
    src7 = __lsx_vld(src, 56);
    DUP2_ARG3(__lsx_vshuf_b, src2, src0, shuff, src4, src2, shuff, src1, src3);
    src5 = __lsx_vshuf_b(src6, src4, shuff);
    src += src_stride;

    DUP2_ARG3(__lsx_vshuf_b, src0, src0, mask, src1, src1, mask, vec0, vec1);
    DUP2_ARG3(__lsx_vshuf_b, src2, src2, mask, src3, src3, mask, vec2, vec3);
    DUP2_ARG3(__lsx_vshuf_b, src4, src4, mask, src5, src5, mask, vec4, vec5);
    DUP2_ARG3(__lsx_vshuf_b, src6, src6, mask, src7, src7, mask, vec6, vec7);
    DUP4_ARG2(__lsx_vdp2_h_bu, vec0, filt0, vec1, filt0, vec2, filt0, vec3,
              filt0, out0, out1, out2, out3);
    DUP4_ARG2(__lsx_vdp2_h_bu, vec4, filt0, vec5, filt0, vec6, filt0, vec7,
              filt0, out4, out5, out6, out7);

    DUP4_ARG3(__lsx_vssrarni_bu_h, out1, out0, FILTER_BITS, out3, out2,
              FILTER_BITS, out5, out4, FILTER_BITS, out7, out6, FILTER_BITS,
              out0, out2, out4, out6);

    DUP4_ARG2(__lsx_vld, dst, 0, dst, 16, dst, 32, dst, 48, dst0, dst1, dst2,
              dst3);
    out0 = __lsx_vavgr_bu(out0, dst0);
    __lsx_vst(out0, dst, 0);
    out2 = __lsx_vavgr_bu(out2, dst1);
    __lsx_vst(out2, dst, 16);
    out4 = __lsx_vavgr_bu(out4, dst2);
    __lsx_vst(out4, dst, 32);
    out6 = __lsx_vavgr_bu(out6, dst3);
    __lsx_vst(out6, dst, 48);
    dst += dst_stride;
  }
}

void vpx_convolve8_avg_horiz_lsx(const uint8_t *src, ptrdiff_t src_stride,
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
        common_hz_2t_and_aver_dst_4w_lsx(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, &filt_hor[3], h);
        break;
      case 8:
        common_hz_2t_and_aver_dst_8w_lsx(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, &filt_hor[3], h);
        break;
      case 16:
        common_hz_2t_and_aver_dst_16w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, &filt_hor[3], h);
        break;

      case 32:
        common_hz_2t_and_aver_dst_32w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, &filt_hor[3], h);
        break;
      case 64:
        common_hz_2t_and_aver_dst_64w_lsx(src, (int32_t)src_stride, dst,
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
        common_hz_8t_and_aver_dst_4w_lsx(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, filt_hor, h);
        break;
      case 8:
        common_hz_8t_and_aver_dst_8w_lsx(src, (int32_t)src_stride, dst,
                                         (int32_t)dst_stride, filt_hor, h);
        break;
      case 16:
        common_hz_8t_and_aver_dst_16w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, filt_hor, h);
        break;
      case 32:
        common_hz_8t_and_aver_dst_32w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, filt_hor, h);
        break;
      case 64:
        common_hz_8t_and_aver_dst_64w_lsx(src, (int32_t)src_stride, dst,
                                          (int32_t)dst_stride, filt_hor, h);
        break;
      default:
        vpx_convolve8_avg_horiz_c(src, src_stride, dst, dst_stride, filter,
                                  x0_q4, x_step_q4, y0_q4, y_step_q4, w, h);
        break;
    }
  }
}
