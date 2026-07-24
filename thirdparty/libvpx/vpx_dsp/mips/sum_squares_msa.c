/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "./macros_msa.h"

uint64_t vpx_sum_squares_2d_i16_msa(const int16_t *src, int src_stride,
                                    int size) {
  int row, col;
  uint64_t ss_res = 0;
  v4i32 mul0, mul1;
  v2i64 res0 = { 0 };

  if (4 == size) {
    uint64_t src0, src1, src2, src3;
    v8i16 diff0 = { 0 };
    v8i16 diff1 = { 0 };

    LD4(src, src_stride, src0, src1, src2, src3);
    INSERT_D2_SH(src0, src1, diff0);
    INSERT_D2_SH(src2, src3, diff1);
    DOTP_SH2_SW(diff0, diff1, diff0, diff1, mul0, mul1);
    mul0 += mul1;
    res0 = __msa_hadd_s_d(mul0, mul0);
    res0 += __msa_splati_d(res0, 1);
    ss_res = (uint64_t)__msa_copy_s_d(res0, 0);
  } else if (8 == size) {
    v8i16 src0, src1, src2, src3, src4, src5, src6, src7;

    LD_SH8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
    DOTP_SH2_SW(src0, src1, src0, src1, mul0, mul1);
    DPADD_SH2_SW(src2, src3, src2, src3, mul0, mul1);
    DPADD_SH2_SW(src4, src5, src4, src5, mul0, mul1);
    DPADD_SH2_SW(src6, src7, src6, src7, mul0, mul1);
    mul0 += mul1;
    res0 = __msa_hadd_s_d(mul0, mul0);
    res0 += __msa_splati_d(res0, 1);
    ss_res = (uint64_t)__msa_copy_s_d(res0, 0);
  } else if (16 == size) {
    v8i16 src0, src1, src2, src3, src4, src5, src6, src7;

    LD_SH8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
    DOTP_SH2_SW(src0, src1, src0, src1, mul0, mul1);
    DPADD_SH2_SW(src2, src3, src2, src3, mul0, mul1);
    DPADD_SH2_SW(src4, src5, src4, src5, mul0, mul1);
    DPADD_SH2_SW(src6, src7, src6, src7, mul0, mul1);
    LD_SH8(src + 8, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
    src += 8 * src_stride;
    DPADD_SH2_SW(src0, src1, src0, src1, mul0, mul1);
    DPADD_SH2_SW(src2, src3, src2, src3, mul0, mul1);
    DPADD_SH2_SW(src4, src5, src4, src5, mul0, mul1);
    DPADD_SH2_SW(src6, src7, src6, src7, mul0, mul1);
    LD_SH8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
    DPADD_SH2_SW(src0, src1, src0, src1, mul0, mul1);
    DPADD_SH2_SW(src2, src3, src2, src3, mul0, mul1);
    DPADD_SH2_SW(src4, src5, src4, src5, mul0, mul1);
    DPADD_SH2_SW(src6, src7, src6, src7, mul0, mul1);
    LD_SH8(src + 8, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
    DPADD_SH2_SW(src0, src1, src0, src1, mul0, mul1);
    DPADD_SH2_SW(src2, src3, src2, src3, mul0, mul1);
    DPADD_SH2_SW(src4, src5, src4, src5, mul0, mul1);
    DPADD_SH2_SW(src6, src7, src6, src7, mul0, mul1);
    mul0 += mul1;
    res0 += __msa_hadd_s_d(mul0, mul0);

    res0 += __msa_splati_d(res0, 1);
    ss_res = (uint64_t)__msa_copy_s_d(res0, 0);
  } else if (0 == (size % 16)) {
    v8i16 src0, src1, src2, src3, src4, src5, src6, src7;

    for (row = 0; row < (size >> 4); row++) {
      for (col = 0; col < size; col += 16) {
        const int16_t *src_ptr = src + col;
        LD_SH8(src_ptr, src_stride, src0, src1, src2, src3, src4, src5, src6,
               src7);
        DOTP_SH2_SW(src0, src1, src0, src1, mul0, mul1);
        DPADD_SH2_SW(src2, src3, src2, src3, mul0, mul1);
        DPADD_SH2_SW(src4, src5, src4, src5, mul0, mul1);
        DPADD_SH2_SW(src6, src7, src6, src7, mul0, mul1);
        LD_SH8(src_ptr + 8, src_stride, src0, src1, src2, src3, src4, src5,
               src6, src7);
        src_ptr += 8 * src_stride;
        DPADD_SH2_SW(src0, src1, src0, src1, mul0, mul1);
        DPADD_SH2_SW(src2, src3, src2, src3, mul0, mul1);
        DPADD_SH2_SW(src4, src5, src4, src5, mul0, mul1);
        DPADD_SH2_SW(src6, src7, src6, src7, mul0, mul1);
        LD_SH8(src_ptr, src_stride, src0, src1, src2, src3, src4, src5, src6,
               src7);
        DPADD_SH2_SW(src0, src1, src0, src1, mul0, mul1);
        DPADD_SH2_SW(src2, src3, src2, src3, mul0, mul1);
        DPADD_SH2_SW(src4, src5, src4, src5, mul0, mul1);
        DPADD_SH2_SW(src6, src7, src6, src7, mul0, mul1);
        LD_SH8(src_ptr + 8, src_stride, src0, src1, src2, src3, src4, src5,
               src6, src7);
        DPADD_SH2_SW(src0, src1, src0, src1, mul0, mul1);
        DPADD_SH2_SW(src2, src3, src2, src3, mul0, mul1);
        DPADD_SH2_SW(src4, src5, src4, src5, mul0, mul1);
        DPADD_SH2_SW(src6, src7, src6, src7, mul0, mul1);
        mul0 += mul1;
        res0 += __msa_hadd_s_d(mul0, mul0);
      }

      src += 16 * src_stride;
    }

    res0 += __msa_splati_d(res0, 1);
    ss_res = (uint64_t)__msa_copy_s_d(res0, 0);
  } else {
    int16_t val;

    for (row = 0; row < size; row++) {
      for (col = 0; col < size; col++) {
        val = src[col];
        ss_res += val * val;
      }

      src += src_stride;
    }
  }

  return ss_res;
}
