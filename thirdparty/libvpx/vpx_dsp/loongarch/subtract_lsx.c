/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_util/loongson_intrinsics.h"

static void sub_blk_4x4_lsx(const uint8_t *src_ptr, int32_t src_stride,
                            const uint8_t *pred_ptr, int32_t pred_stride,
                            int16_t *diff_ptr, int32_t diff_stride) {
  __m128i src0, src1, src2, src3;
  __m128i pred0, pred1, pred2, pred3;
  __m128i diff0, diff1;
  __m128i reg0, reg1;
  int32_t src_stride2 = src_stride << 1;
  int32_t pred_stride2 = pred_stride << 1;
  int32_t diff_stride2 = diff_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t pred_stride3 = pred_stride2 + pred_stride;
  int32_t diff_stride3 = diff_stride2 + diff_stride;

  DUP4_ARG2(__lsx_vldrepl_w, src_ptr, 0, src_ptr + src_stride, 0,
            src_ptr + src_stride2, 0, src_ptr + src_stride3, 0, src0, src1,
            src2, src3);
  DUP4_ARG2(__lsx_vldrepl_w, pred_ptr, 0, pred_ptr + pred_stride, 0,
            pred_ptr + pred_stride2, 0, pred_ptr + pred_stride3, 0, pred0,
            pred1, pred2, pred3);
  DUP4_ARG2(__lsx_vilvl_w, src1, src0, src3, src2, pred1, pred0, pred3, pred2,
            src0, src2, pred0, pred2);
  DUP2_ARG2(__lsx_vilvl_d, src2, src0, pred2, pred0, src0, pred0);
  reg0 = __lsx_vilvl_b(src0, pred0);
  reg1 = __lsx_vilvh_b(src0, pred0);
  DUP2_ARG2(__lsx_vhsubw_hu_bu, reg0, reg0, reg1, reg1, diff0, diff1);
  __lsx_vstelm_d(diff0, diff_ptr, 0, 0);
  __lsx_vstelm_d(diff0, diff_ptr + diff_stride, 0, 1);
  __lsx_vstelm_d(diff1, diff_ptr + diff_stride2, 0, 0);
  __lsx_vstelm_d(diff1, diff_ptr + diff_stride3, 0, 1);
}

static void sub_blk_8x8_lsx(const uint8_t *src_ptr, int32_t src_stride,
                            const uint8_t *pred_ptr, int32_t pred_stride,
                            int16_t *diff_ptr, int32_t diff_stride) {
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  int32_t src_stride2 = src_stride << 1;
  int32_t pred_stride2 = pred_stride << 1;
  int32_t dst_stride = diff_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t pred_stride3 = pred_stride2 + pred_stride;
  int32_t dst_stride2 = dst_stride << 1;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t pred_stride4 = pred_stride2 << 1;
  int32_t dst_stride3 = dst_stride + dst_stride2;

  DUP4_ARG2(__lsx_vldrepl_d, src_ptr, 0, src_ptr + src_stride, 0,
            src_ptr + src_stride2, 0, src_ptr + src_stride3, 0, src0, src1,
            src2, src3);
  DUP4_ARG2(__lsx_vldrepl_d, pred_ptr, 0, pred_ptr + pred_stride, 0,
            pred_ptr + pred_stride2, 0, pred_ptr + pred_stride3, 0, pred0,
            pred1, pred2, pred3);
  src_ptr += src_stride4;
  pred_ptr += pred_stride4;

  DUP4_ARG2(__lsx_vldrepl_d, src_ptr, 0, src_ptr + src_stride, 0,
            src_ptr + src_stride2, 0, src_ptr + src_stride3, 0, src4, src5,
            src6, src7);
  DUP4_ARG2(__lsx_vldrepl_d, pred_ptr, 0, pred_ptr + pred_stride, 0,
            pred_ptr + pred_stride2, 0, pred_ptr + pred_stride3, 0, pred4,
            pred5, pred6, pred7);

  DUP4_ARG2(__lsx_vilvl_b, src0, pred0, src1, pred1, src2, pred2, src3, pred3,
            reg0, reg1, reg2, reg3);
  DUP4_ARG2(__lsx_vilvl_b, src4, pred4, src5, pred5, src6, pred6, src7, pred7,
            reg4, reg5, reg6, reg7);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, reg0, reg0, reg1, reg1, reg2, reg2, reg3, reg3,
            src0, src1, src2, src3);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, reg4, reg4, reg5, reg5, reg6, reg6, reg7, reg7,
            src4, src5, src6, src7);
  __lsx_vst(src0, diff_ptr, 0);
  __lsx_vstx(src1, diff_ptr, dst_stride);
  __lsx_vstx(src2, diff_ptr, dst_stride2);
  __lsx_vstx(src3, diff_ptr, dst_stride3);
  diff_ptr += dst_stride2;
  __lsx_vst(src4, diff_ptr, 0);
  __lsx_vstx(src5, diff_ptr, dst_stride);
  __lsx_vstx(src6, diff_ptr, dst_stride2);
  __lsx_vstx(src7, diff_ptr, dst_stride3);
}

static void sub_blk_16x16_lsx(const uint8_t *src, int32_t src_stride,
                              const uint8_t *pred, int32_t pred_stride,
                              int16_t *diff, int32_t diff_stride) {
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  int32_t src_stride2 = src_stride << 1;
  int32_t pred_stride2 = pred_stride << 1;
  int32_t dst_stride = diff_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t pred_stride3 = pred_stride2 + pred_stride;
  int32_t dst_stride2 = dst_stride << 1;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t pred_stride4 = pred_stride2 << 1;
  int32_t dst_stride3 = dst_stride + dst_stride2;
  int16_t *diff_tmp = diff + 8;

  DUP2_ARG2(__lsx_vld, src, 0, pred, 0, src0, pred0);
  DUP4_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src, src_stride3,
            src, src_stride4, src1, src2, src3, src4);
  DUP4_ARG2(__lsx_vldx, pred, pred_stride, pred, pred_stride2, pred,
            pred_stride3, pred, pred_stride4, pred1, pred2, pred3, pred4);
  src += src_stride4;
  pred += pred_stride4;
  DUP4_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src, src_stride3,
            pred, pred_stride, src5, src6, src7, pred5);
  DUP2_ARG2(__lsx_vldx, pred, pred_stride2, pred, pred_stride3, pred6, pred7);
  src += src_stride4;
  pred += pred_stride4;
  DUP4_ARG2(__lsx_vilvl_b, src0, pred0, src1, pred1, src2, pred2, src3, pred3,
            reg0, reg2, reg4, reg6);
  DUP4_ARG2(__lsx_vilvh_b, src0, pred0, src1, pred1, src2, pred2, src3, pred3,
            reg1, reg3, reg5, reg7);
  DUP4_ARG2(__lsx_vilvl_b, src4, pred4, src5, pred5, src6, pred6, src7, pred7,
            tmp0, tmp2, tmp4, tmp6);
  DUP4_ARG2(__lsx_vilvh_b, src4, pred4, src5, pred5, src6, pred6, src7, pred7,
            tmp1, tmp3, tmp5, tmp7);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, reg0, reg0, reg1, reg1, reg2, reg2, reg3, reg3,
            src0, src1, src2, src3);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, reg4, reg4, reg5, reg5, reg6, reg6, reg7, reg7,
            src4, src5, src6, src7);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, tmp0, tmp0, tmp1, tmp1, tmp2, tmp2, tmp3, tmp3,
            pred0, pred1, pred2, pred3);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, tmp4, tmp4, tmp5, tmp5, tmp6, tmp6, tmp7, tmp7,
            pred4, pred5, pred6, pred7);
  __lsx_vst(src0, diff, 0);
  __lsx_vstx(src2, diff, dst_stride);
  __lsx_vstx(src4, diff, dst_stride2);
  __lsx_vstx(src6, diff, dst_stride3);
  __lsx_vst(src1, diff_tmp, 0);
  __lsx_vstx(src3, diff_tmp, dst_stride);
  __lsx_vstx(src5, diff_tmp, dst_stride2);
  __lsx_vstx(src7, diff_tmp, dst_stride3);
  diff += dst_stride2;
  diff_tmp += dst_stride2;
  __lsx_vst(pred0, diff, 0);
  __lsx_vstx(pred2, diff, dst_stride);
  __lsx_vstx(pred4, diff, dst_stride2);
  __lsx_vstx(pred6, diff, dst_stride3);
  __lsx_vst(pred1, diff_tmp, 0);
  __lsx_vstx(pred3, diff_tmp, dst_stride);
  __lsx_vstx(pred5, diff_tmp, dst_stride2);
  __lsx_vstx(pred7, diff_tmp, dst_stride3);
  diff += dst_stride2;
  diff_tmp += dst_stride2;
  DUP2_ARG2(__lsx_vld, src, 0, pred, 0, src0, pred0);
  DUP4_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src, src_stride3,
            src, src_stride4, src1, src2, src3, src4);
  DUP4_ARG2(__lsx_vldx, pred, pred_stride, pred, pred_stride2, pred,
            pred_stride3, pred, pred_stride4, pred1, pred2, pred3, pred4);
  src += src_stride4;
  pred += pred_stride4;
  DUP4_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src, src_stride3,
            pred, pred_stride, src5, src6, src7, pred5);
  DUP2_ARG2(__lsx_vldx, pred, pred_stride2, pred, pred_stride3, pred6, pred7);
  DUP4_ARG2(__lsx_vilvl_b, src0, pred0, src1, pred1, src2, pred2, src3, pred3,
            reg0, reg2, reg4, reg6);
  DUP4_ARG2(__lsx_vilvh_b, src0, pred0, src1, pred1, src2, pred2, src3, pred3,
            reg1, reg3, reg5, reg7);
  DUP4_ARG2(__lsx_vilvl_b, src4, pred4, src5, pred5, src6, pred6, src7, pred7,
            tmp0, tmp2, tmp4, tmp6);
  DUP4_ARG2(__lsx_vilvh_b, src4, pred4, src5, pred5, src6, pred6, src7, pred7,
            tmp1, tmp3, tmp5, tmp7);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, reg0, reg0, reg1, reg1, reg2, reg2, reg3, reg3,
            src0, src1, src2, src3);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, reg4, reg4, reg5, reg5, reg6, reg6, reg7, reg7,
            src4, src5, src6, src7);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, tmp0, tmp0, tmp1, tmp1, tmp2, tmp2, tmp3, tmp3,
            pred0, pred1, pred2, pred3);
  DUP4_ARG2(__lsx_vhsubw_hu_bu, tmp4, tmp4, tmp5, tmp5, tmp6, tmp6, tmp7, tmp7,
            pred4, pred5, pred6, pred7);
  __lsx_vst(src0, diff, 0);
  __lsx_vstx(src2, diff, dst_stride);
  __lsx_vstx(src4, diff, dst_stride2);
  __lsx_vstx(src6, diff, dst_stride3);
  __lsx_vst(src1, diff_tmp, 0);
  __lsx_vstx(src3, diff_tmp, dst_stride);
  __lsx_vstx(src5, diff_tmp, dst_stride2);
  __lsx_vstx(src7, diff_tmp, dst_stride3);
  diff += dst_stride2;
  diff_tmp += dst_stride2;
  __lsx_vst(pred0, diff, 0);
  __lsx_vstx(pred2, diff, dst_stride);
  __lsx_vstx(pred4, diff, dst_stride2);
  __lsx_vstx(pred6, diff, dst_stride3);
  __lsx_vst(pred1, diff_tmp, 0);
  __lsx_vstx(pred3, diff_tmp, dst_stride);
  __lsx_vstx(pred5, diff_tmp, dst_stride2);
  __lsx_vstx(pred7, diff_tmp, dst_stride3);
}

static void sub_blk_32x32_lsx(const uint8_t *src, int32_t src_stride,
                              const uint8_t *pred, int32_t pred_stride,
                              int16_t *diff, int32_t diff_stride) {
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  uint32_t loop_cnt;
  int32_t src_stride2 = src_stride << 1;
  int32_t pred_stride2 = pred_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t pred_stride3 = pred_stride2 + pred_stride;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t pred_stride4 = pred_stride2 << 1;

  for (loop_cnt = 8; loop_cnt--;) {
    const uint8_t *src_tmp = src + 16;
    const uint8_t *pred_tmp = pred + 16;
    DUP4_ARG2(__lsx_vld, src, 0, src_tmp, 0, pred, 0, pred_tmp, 0, src0, src1,
              pred0, pred1);
    DUP4_ARG2(__lsx_vldx, src, src_stride, src_tmp, src_stride, src,
              src_stride2, src_tmp, src_stride2, src2, src3, src4, src5);
    DUP4_ARG2(__lsx_vldx, src, src_stride3, src_tmp, src_stride3, pred,
              pred_stride, pred_tmp, pred_stride, src6, src7, pred2, pred3);
    DUP4_ARG2(__lsx_vldx, pred, pred_stride2, pred_tmp, pred_stride2, pred,
              pred_stride3, pred_tmp, pred_stride3, pred4, pred5, pred6, pred7);
    DUP4_ARG2(__lsx_vilvl_b, src0, pred0, src1, pred1, src2, pred2, src3, pred3,
              reg0, reg2, reg4, reg6);
    DUP4_ARG2(__lsx_vilvh_b, src0, pred0, src1, pred1, src2, pred2, src3, pred3,
              reg1, reg3, reg5, reg7);
    DUP4_ARG2(__lsx_vilvl_b, src4, pred4, src5, pred5, src6, pred6, src7, pred7,
              tmp0, tmp2, tmp4, tmp6);
    DUP4_ARG2(__lsx_vilvh_b, src4, pred4, src5, pred5, src6, pred6, src7, pred7,
              tmp1, tmp3, tmp5, tmp7);
    DUP4_ARG2(__lsx_vhsubw_hu_bu, reg0, reg0, reg1, reg1, reg2, reg2, reg3,
              reg3, src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vhsubw_hu_bu, reg4, reg4, reg5, reg5, reg6, reg6, reg7,
              reg7, src4, src5, src6, src7);
    DUP4_ARG2(__lsx_vhsubw_hu_bu, tmp0, tmp0, tmp1, tmp1, tmp2, tmp2, tmp3,
              tmp3, pred0, pred1, pred2, pred3);
    DUP4_ARG2(__lsx_vhsubw_hu_bu, tmp4, tmp4, tmp5, tmp5, tmp6, tmp6, tmp7,
              tmp7, pred4, pred5, pred6, pred7);
    src += src_stride4;
    pred += pred_stride4;
    __lsx_vst(src0, diff, 0);
    __lsx_vst(src1, diff, 16);
    __lsx_vst(src2, diff, 32);
    __lsx_vst(src3, diff, 48);
    diff += diff_stride;
    __lsx_vst(src4, diff, 0);
    __lsx_vst(src5, diff, 16);
    __lsx_vst(src6, diff, 32);
    __lsx_vst(src7, diff, 48);
    diff += diff_stride;
    __lsx_vst(pred0, diff, 0);
    __lsx_vst(pred1, diff, 16);
    __lsx_vst(pred2, diff, 32);
    __lsx_vst(pred3, diff, 48);
    diff += diff_stride;
    __lsx_vst(pred4, diff, 0);
    __lsx_vst(pred5, diff, 16);
    __lsx_vst(pred6, diff, 32);
    __lsx_vst(pred7, diff, 48);
    diff += diff_stride;
  }
}

static void sub_blk_64x64_lsx(const uint8_t *src, int32_t src_stride,
                              const uint8_t *pred, int32_t pred_stride,
                              int16_t *diff, int32_t diff_stride) {
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  uint32_t loop_cnt;

  for (loop_cnt = 32; loop_cnt--;) {
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    DUP4_ARG2(__lsx_vld, pred, 0, pred, 16, pred, 32, pred, 48, pred0, pred1,
              pred2, pred3);
    src += src_stride;
    pred += pred_stride;
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src4, src5, src6,
              src7);
    DUP4_ARG2(__lsx_vld, pred, 0, pred, 16, pred, 32, pred, 48, pred4, pred5,
              pred6, pred7);
    src += src_stride;
    pred += pred_stride;

    DUP4_ARG2(__lsx_vilvl_b, src0, pred0, src1, pred1, src2, pred2, src3, pred3,
              reg0, reg2, reg4, reg6);
    DUP4_ARG2(__lsx_vilvh_b, src0, pred0, src1, pred1, src2, pred2, src3, pred3,
              reg1, reg3, reg5, reg7);
    DUP4_ARG2(__lsx_vilvl_b, src4, pred4, src5, pred5, src6, pred6, src7, pred7,
              tmp0, tmp2, tmp4, tmp6);
    DUP4_ARG2(__lsx_vilvh_b, src4, pred4, src5, pred5, src6, pred6, src7, pred7,
              tmp1, tmp3, tmp5, tmp7);
    DUP4_ARG2(__lsx_vhsubw_hu_bu, reg0, reg0, reg1, reg1, reg2, reg2, reg3,
              reg3, src0, src1, src2, src3);
    DUP4_ARG2(__lsx_vhsubw_hu_bu, reg4, reg4, reg5, reg5, reg6, reg6, reg7,
              reg7, src4, src5, src6, src7);
    DUP4_ARG2(__lsx_vhsubw_hu_bu, tmp0, tmp0, tmp1, tmp1, tmp2, tmp2, tmp3,
              tmp3, pred0, pred1, pred2, pred3);
    DUP4_ARG2(__lsx_vhsubw_hu_bu, tmp4, tmp4, tmp5, tmp5, tmp6, tmp6, tmp7,
              tmp7, pred4, pred5, pred6, pred7);
    __lsx_vst(src0, diff, 0);
    __lsx_vst(src1, diff, 16);
    __lsx_vst(src2, diff, 32);
    __lsx_vst(src3, diff, 48);
    __lsx_vst(src4, diff, 64);
    __lsx_vst(src5, diff, 80);
    __lsx_vst(src6, diff, 96);
    __lsx_vst(src7, diff, 112);
    diff += diff_stride;
    __lsx_vst(pred0, diff, 0);
    __lsx_vst(pred1, diff, 16);
    __lsx_vst(pred2, diff, 32);
    __lsx_vst(pred3, diff, 48);
    __lsx_vst(pred4, diff, 64);
    __lsx_vst(pred5, diff, 80);
    __lsx_vst(pred6, diff, 96);
    __lsx_vst(pred7, diff, 112);
    diff += diff_stride;
  }
}

void vpx_subtract_block_lsx(int32_t rows, int32_t cols, int16_t *diff_ptr,
                            ptrdiff_t diff_stride, const uint8_t *src_ptr,
                            ptrdiff_t src_stride, const uint8_t *pred_ptr,
                            ptrdiff_t pred_stride) {
  if (rows == cols) {
    switch (rows) {
      case 4:
        sub_blk_4x4_lsx(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
                        diff_stride);
        break;
      case 8:
        sub_blk_8x8_lsx(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
                        diff_stride);
        break;
      case 16:
        sub_blk_16x16_lsx(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
                          diff_stride);
        break;
      case 32:
        sub_blk_32x32_lsx(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
                          diff_stride);
        break;
      case 64:
        sub_blk_64x64_lsx(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
                          diff_stride);
        break;
      default:
        vpx_subtract_block_c(rows, cols, diff_ptr, diff_stride, src_ptr,
                             src_stride, pred_ptr, pred_stride);
        break;
    }
  } else {
    vpx_subtract_block_c(rows, cols, diff_ptr, diff_stride, src_ptr, src_stride,
                         pred_ptr, pred_stride);
  }
}
