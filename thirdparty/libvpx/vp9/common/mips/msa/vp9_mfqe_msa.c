/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp9_rtcd.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vpx_dsp/mips/macros_msa.h"

static void filter_by_weight8x8_msa(const uint8_t *src_ptr, int32_t src_stride,
                                    uint8_t *dst_ptr, int32_t dst_stride,
                                    int32_t src_weight) {
  int32_t dst_weight = (1 << MFQE_PRECISION) - src_weight;
  int32_t row;
  uint64_t src0_d, src1_d, dst0_d, dst1_d;
  v16i8 src0 = { 0 };
  v16i8 src1 = { 0 };
  v16i8 dst0 = { 0 };
  v16i8 dst1 = { 0 };
  v8i16 src_wt, dst_wt, res_h_r, res_h_l, src_r, src_l, dst_r, dst_l;

  src_wt = __msa_fill_h(src_weight);
  dst_wt = __msa_fill_h(dst_weight);

  for (row = 2; row--;) {
    LD2(src_ptr, src_stride, src0_d, src1_d);
    src_ptr += (2 * src_stride);
    LD2(dst_ptr, dst_stride, dst0_d, dst1_d);
    INSERT_D2_SB(src0_d, src1_d, src0);
    INSERT_D2_SB(dst0_d, dst1_d, dst0);

    LD2(src_ptr, src_stride, src0_d, src1_d);
    src_ptr += (2 * src_stride);
    LD2((dst_ptr + 2 * dst_stride), dst_stride, dst0_d, dst1_d);
    INSERT_D2_SB(src0_d, src1_d, src1);
    INSERT_D2_SB(dst0_d, dst1_d, dst1);

    UNPCK_UB_SH(src0, src_r, src_l);
    UNPCK_UB_SH(dst0, dst_r, dst_l);
    res_h_r = (src_r * src_wt);
    res_h_r += (dst_r * dst_wt);
    res_h_l = (src_l * src_wt);
    res_h_l += (dst_l * dst_wt);
    SRARI_H2_SH(res_h_r, res_h_l, MFQE_PRECISION);
    dst0 = (v16i8)__msa_pckev_b((v16i8)res_h_l, (v16i8)res_h_r);
    ST8x2_UB(dst0, dst_ptr, dst_stride);
    dst_ptr += (2 * dst_stride);

    UNPCK_UB_SH(src1, src_r, src_l);
    UNPCK_UB_SH(dst1, dst_r, dst_l);
    res_h_r = (src_r * src_wt);
    res_h_r += (dst_r * dst_wt);
    res_h_l = (src_l * src_wt);
    res_h_l += (dst_l * dst_wt);
    SRARI_H2_SH(res_h_r, res_h_l, MFQE_PRECISION);
    dst1 = (v16i8)__msa_pckev_b((v16i8)res_h_l, (v16i8)res_h_r);
    ST8x2_UB(dst1, dst_ptr, dst_stride);
    dst_ptr += (2 * dst_stride);
  }
}

static void filter_by_weight16x16_msa(const uint8_t *src_ptr,
                                      int32_t src_stride, uint8_t *dst_ptr,
                                      int32_t dst_stride, int32_t src_weight) {
  int32_t dst_weight = (1 << MFQE_PRECISION) - src_weight;
  int32_t row;
  v16i8 src0, src1, src2, src3, dst0, dst1, dst2, dst3;
  v8i16 src_wt, dst_wt, res_h_r, res_h_l, src_r, src_l, dst_r, dst_l;

  src_wt = __msa_fill_h(src_weight);
  dst_wt = __msa_fill_h(dst_weight);

  for (row = 4; row--;) {
    LD_SB4(src_ptr, src_stride, src0, src1, src2, src3);
    src_ptr += (4 * src_stride);
    LD_SB4(dst_ptr, dst_stride, dst0, dst1, dst2, dst3);

    UNPCK_UB_SH(src0, src_r, src_l);
    UNPCK_UB_SH(dst0, dst_r, dst_l);
    res_h_r = (src_r * src_wt);
    res_h_r += (dst_r * dst_wt);
    res_h_l = (src_l * src_wt);
    res_h_l += (dst_l * dst_wt);
    SRARI_H2_SH(res_h_r, res_h_l, MFQE_PRECISION);
    PCKEV_ST_SB(res_h_r, res_h_l, dst_ptr);
    dst_ptr += dst_stride;

    UNPCK_UB_SH(src1, src_r, src_l);
    UNPCK_UB_SH(dst1, dst_r, dst_l);
    res_h_r = (src_r * src_wt);
    res_h_r += (dst_r * dst_wt);
    res_h_l = (src_l * src_wt);
    res_h_l += (dst_l * dst_wt);
    SRARI_H2_SH(res_h_r, res_h_l, MFQE_PRECISION);
    PCKEV_ST_SB(res_h_r, res_h_l, dst_ptr);
    dst_ptr += dst_stride;

    UNPCK_UB_SH(src2, src_r, src_l);
    UNPCK_UB_SH(dst2, dst_r, dst_l);
    res_h_r = (src_r * src_wt);
    res_h_r += (dst_r * dst_wt);
    res_h_l = (src_l * src_wt);
    res_h_l += (dst_l * dst_wt);
    SRARI_H2_SH(res_h_r, res_h_l, MFQE_PRECISION);
    PCKEV_ST_SB(res_h_r, res_h_l, dst_ptr);
    dst_ptr += dst_stride;

    UNPCK_UB_SH(src3, src_r, src_l);
    UNPCK_UB_SH(dst3, dst_r, dst_l);
    res_h_r = (src_r * src_wt);
    res_h_r += (dst_r * dst_wt);
    res_h_l = (src_l * src_wt);
    res_h_l += (dst_l * dst_wt);
    SRARI_H2_SH(res_h_r, res_h_l, MFQE_PRECISION);
    PCKEV_ST_SB(res_h_r, res_h_l, dst_ptr);
    dst_ptr += dst_stride;
  }
}

void vp9_filter_by_weight8x8_msa(const uint8_t *src, int src_stride,
                                 uint8_t *dst, int dst_stride, int src_weight) {
  filter_by_weight8x8_msa(src, src_stride, dst, dst_stride, src_weight);
}

void vp9_filter_by_weight16x16_msa(const uint8_t *src, int src_stride,
                                   uint8_t *dst, int dst_stride,
                                   int src_weight) {
  filter_by_weight16x16_msa(src, src_stride, dst, dst_stride, src_weight);
}
