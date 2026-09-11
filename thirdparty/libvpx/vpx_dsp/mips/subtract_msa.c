/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/macros_msa.h"

static void sub_blk_4x4_msa(const uint8_t *src_ptr, int32_t src_stride,
                            const uint8_t *pred_ptr, int32_t pred_stride,
                            int16_t *diff_ptr, int32_t diff_stride) {
  uint32_t src0, src1, src2, src3;
  uint32_t pred0, pred1, pred2, pred3;
  v16i8 src = { 0 };
  v16i8 pred = { 0 };
  v16u8 src_l0, src_l1;
  v8i16 diff0, diff1;

  LW4(src_ptr, src_stride, src0, src1, src2, src3);
  LW4(pred_ptr, pred_stride, pred0, pred1, pred2, pred3);
  INSERT_W4_SB(src0, src1, src2, src3, src);
  INSERT_W4_SB(pred0, pred1, pred2, pred3, pred);
  ILVRL_B2_UB(src, pred, src_l0, src_l1);
  HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
  ST8x4_UB(diff0, diff1, diff_ptr, (2 * diff_stride));
}

static void sub_blk_8x8_msa(const uint8_t *src_ptr, int32_t src_stride,
                            const uint8_t *pred_ptr, int32_t pred_stride,
                            int16_t *diff_ptr, int32_t diff_stride) {
  uint32_t loop_cnt;
  uint64_t src0, src1, pred0, pred1;
  v16i8 src = { 0 };
  v16i8 pred = { 0 };
  v16u8 src_l0, src_l1;
  v8i16 diff0, diff1;

  for (loop_cnt = 4; loop_cnt--;) {
    LD2(src_ptr, src_stride, src0, src1);
    src_ptr += (2 * src_stride);
    LD2(pred_ptr, pred_stride, pred0, pred1);
    pred_ptr += (2 * pred_stride);

    INSERT_D2_SB(src0, src1, src);
    INSERT_D2_SB(pred0, pred1, pred);
    ILVRL_B2_UB(src, pred, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff_ptr, diff_stride);
    diff_ptr += (2 * diff_stride);
  }
}

static void sub_blk_16x16_msa(const uint8_t *src, int32_t src_stride,
                              const uint8_t *pred, int32_t pred_stride,
                              int16_t *diff, int32_t diff_stride) {
  int8_t count;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16i8 pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7;
  v16u8 src_l0, src_l1;
  v8i16 diff0, diff1;

  for (count = 2; count--;) {
    LD_SB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
    src += (8 * src_stride);

    LD_SB8(pred, pred_stride, pred0, pred1, pred2, pred3, pred4, pred5, pred6,
           pred7);
    pred += (8 * pred_stride);

    ILVRL_B2_UB(src0, pred0, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src1, pred1, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src2, pred2, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src3, pred3, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src4, pred4, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src5, pred5, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src6, pred6, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src7, pred7, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    diff += diff_stride;
  }
}

static void sub_blk_32x32_msa(const uint8_t *src, int32_t src_stride,
                              const uint8_t *pred, int32_t pred_stride,
                              int16_t *diff, int32_t diff_stride) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16i8 pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7;
  v16u8 src_l0, src_l1;
  v8i16 diff0, diff1;

  for (loop_cnt = 8; loop_cnt--;) {
    LD_SB2(src, 16, src0, src1);
    src += src_stride;
    LD_SB2(src, 16, src2, src3);
    src += src_stride;
    LD_SB2(src, 16, src4, src5);
    src += src_stride;
    LD_SB2(src, 16, src6, src7);
    src += src_stride;

    LD_SB2(pred, 16, pred0, pred1);
    pred += pred_stride;
    LD_SB2(pred, 16, pred2, pred3);
    pred += pred_stride;
    LD_SB2(pred, 16, pred4, pred5);
    pred += pred_stride;
    LD_SB2(pred, 16, pred6, pred7);
    pred += pred_stride;

    ILVRL_B2_UB(src0, pred0, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    ILVRL_B2_UB(src1, pred1, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 16, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src2, pred2, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    ILVRL_B2_UB(src3, pred3, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 16, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src4, pred4, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    ILVRL_B2_UB(src5, pred5, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 16, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src6, pred6, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    ILVRL_B2_UB(src7, pred7, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 16, 8);
    diff += diff_stride;
  }
}

static void sub_blk_64x64_msa(const uint8_t *src, int32_t src_stride,
                              const uint8_t *pred, int32_t pred_stride,
                              int16_t *diff, int32_t diff_stride) {
  uint32_t loop_cnt;
  v16i8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16i8 pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7;
  v16u8 src_l0, src_l1;
  v8i16 diff0, diff1;

  for (loop_cnt = 32; loop_cnt--;) {
    LD_SB4(src, 16, src0, src1, src2, src3);
    src += src_stride;
    LD_SB4(src, 16, src4, src5, src6, src7);
    src += src_stride;

    LD_SB4(pred, 16, pred0, pred1, pred2, pred3);
    pred += pred_stride;
    LD_SB4(pred, 16, pred4, pred5, pred6, pred7);
    pred += pred_stride;

    ILVRL_B2_UB(src0, pred0, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    ILVRL_B2_UB(src1, pred1, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 16, 8);
    ILVRL_B2_UB(src2, pred2, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 32, 8);
    ILVRL_B2_UB(src3, pred3, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 48, 8);
    diff += diff_stride;

    ILVRL_B2_UB(src4, pred4, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff, 8);
    ILVRL_B2_UB(src5, pred5, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 16, 8);
    ILVRL_B2_UB(src6, pred6, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 32, 8);
    ILVRL_B2_UB(src7, pred7, src_l0, src_l1);
    HSUB_UB2_SH(src_l0, src_l1, diff0, diff1);
    ST_SH2(diff0, diff1, diff + 48, 8);
    diff += diff_stride;
  }
}

void vpx_subtract_block_msa(int32_t rows, int32_t cols, int16_t *diff_ptr,
                            ptrdiff_t diff_stride, const uint8_t *src_ptr,
                            ptrdiff_t src_stride, const uint8_t *pred_ptr,
                            ptrdiff_t pred_stride) {
  if (rows == cols) {
    switch (rows) {
      case 4:
        sub_blk_4x4_msa(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
                        diff_stride);
        break;
      case 8:
        sub_blk_8x8_msa(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
                        diff_stride);
        break;
      case 16:
        sub_blk_16x16_msa(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
                          diff_stride);
        break;
      case 32:
        sub_blk_32x32_msa(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
                          diff_stride);
        break;
      case 64:
        sub_blk_64x64_msa(src_ptr, src_stride, pred_ptr, pred_stride, diff_ptr,
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
