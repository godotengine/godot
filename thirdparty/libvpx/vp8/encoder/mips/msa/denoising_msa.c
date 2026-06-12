/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>
#include "./vp8_rtcd.h"
#include "vp8/common/mips/msa/vp8_macros_msa.h"
#include "vp8/encoder/denoising.h"

int32_t vp8_denoiser_filter_msa(uint8_t *mc_running_avg_y_ptr,
                                int32_t mc_avg_y_stride,
                                uint8_t *running_avg_y_ptr,
                                int32_t avg_y_stride, uint8_t *sig_ptr,
                                int32_t sig_stride, uint32_t motion_magnitude,
                                int32_t increase_denoising) {
  uint8_t *running_avg_y_start = running_avg_y_ptr;
  uint8_t *sig_start = sig_ptr;
  int32_t cnt = 0;
  int32_t sum_diff = 0;
  int32_t shift_inc1 = 3;
  int32_t delta = 0;
  int32_t sum_diff_thresh;
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 src8, src9, src10, src11, src12, src13, src14, src15;
  v16u8 mc_running_avg_y0, running_avg_y, sig0;
  v16u8 mc_running_avg_y1, running_avg_y1, sig1;
  v16u8 coeff0, coeff1;
  v8i16 diff0, diff1, abs_diff0, abs_diff1, abs_diff_neg0, abs_diff_neg1;
  v8i16 adjust0, adjust1, adjust2, adjust3;
  v8i16 shift_inc1_vec = { 0 };
  v8i16 col_sum0 = { 0 };
  v8i16 col_sum1 = { 0 };
  v8i16 col_sum2 = { 0 };
  v8i16 col_sum3 = { 0 };
  v8i16 temp0_h, temp1_h, temp2_h, temp3_h, cmp, delta_vec;
  v4i32 temp0_w;
  v2i64 temp0_d, temp1_d;
  v8i16 zero = { 0 };
  v8i16 one = __msa_ldi_h(1);
  v8i16 four = __msa_ldi_h(4);
  v8i16 val_127 = __msa_ldi_h(127);
  v8i16 adj_val = { 6, 4, 3, 0, -6, -4, -3, 0 };

  if (motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD) {
    adj_val = __msa_add_a_h(adj_val, one);
    if (increase_denoising) {
      adj_val = __msa_add_a_h(adj_val, one);
      shift_inc1 = 4;
    }

    temp0_h = zero - adj_val;
    adj_val = (v8i16)__msa_ilvev_d((v2i64)temp0_h, (v2i64)adj_val);
  }

  adj_val = __msa_insert_h(adj_val, 3, cnt);
  adj_val = __msa_insert_h(adj_val, 7, cnt);
  shift_inc1_vec = __msa_fill_h(shift_inc1);

  for (cnt = 8; cnt--;) {
    v8i16 mask0 = { 0 };
    v8i16 mask1 = { 0 };

    mc_running_avg_y0 = LD_UB(mc_running_avg_y_ptr);
    sig0 = LD_UB(sig_ptr);
    sig_ptr += sig_stride;
    mc_running_avg_y_ptr += mc_avg_y_stride;

    mc_running_avg_y1 = LD_UB(mc_running_avg_y_ptr);
    sig1 = LD_UB(sig_ptr);

    ILVRL_B2_UB(mc_running_avg_y0, sig0, coeff0, coeff1);
    HSUB_UB2_SH(coeff0, coeff1, diff0, diff1);
    abs_diff0 = __msa_add_a_h(diff0, zero);
    abs_diff1 = __msa_add_a_h(diff1, zero);
    cmp = __msa_clei_s_h(abs_diff0, 15);
    cmp = cmp & one;
    mask0 += cmp;
    cmp = __msa_clei_s_h(abs_diff0, 7);
    cmp = cmp & one;
    mask0 += cmp;
    cmp = abs_diff0 < shift_inc1_vec;
    cmp = cmp & one;
    mask0 += cmp;
    cmp = __msa_clei_s_h(abs_diff1, 15);
    cmp = cmp & one;
    mask1 += cmp;
    cmp = __msa_clei_s_h(abs_diff1, 7);
    cmp = cmp & one;
    mask1 += cmp;
    cmp = abs_diff1 < shift_inc1_vec;
    cmp = cmp & one;
    mask1 += cmp;
    temp0_h = __msa_clei_s_h(diff0, 0);
    temp0_h = temp0_h & four;
    mask0 += temp0_h;
    temp1_h = __msa_clei_s_h(diff1, 0);
    temp1_h = temp1_h & four;
    mask1 += temp1_h;
    VSHF_H2_SH(adj_val, adj_val, adj_val, adj_val, mask0, mask1, adjust0,
               adjust1);
    temp2_h = __msa_ceqi_h(adjust0, 0);
    temp3_h = __msa_ceqi_h(adjust1, 0);
    adjust0 = (v8i16)__msa_bmnz_v((v16u8)adjust0, (v16u8)diff0, (v16u8)temp2_h);
    adjust1 = (v8i16)__msa_bmnz_v((v16u8)adjust1, (v16u8)diff1, (v16u8)temp3_h);
    ADD2(col_sum0, adjust0, col_sum1, adjust1, col_sum0, col_sum1);
    UNPCK_UB_SH(sig0, temp0_h, temp1_h);
    ADD2(temp0_h, adjust0, temp1_h, adjust1, temp0_h, temp1_h);
    MAXI_SH2_SH(temp0_h, temp1_h, 0);
    SAT_UH2_SH(temp0_h, temp1_h, 7);
    temp2_h = (v8i16)__msa_pckev_b((v16i8)temp3_h, (v16i8)temp2_h);
    running_avg_y = (v16u8)__msa_pckev_b((v16i8)temp1_h, (v16i8)temp0_h);
    running_avg_y =
        __msa_bmnz_v(running_avg_y, mc_running_avg_y0, (v16u8)temp2_h);
    ST_UB(running_avg_y, running_avg_y_ptr);
    running_avg_y_ptr += avg_y_stride;

    mask0 = zero;
    mask1 = zero;
    ILVRL_B2_UB(mc_running_avg_y1, sig1, coeff0, coeff1);
    HSUB_UB2_SH(coeff0, coeff1, diff0, diff1);
    abs_diff0 = __msa_add_a_h(diff0, zero);
    abs_diff1 = __msa_add_a_h(diff1, zero);
    cmp = __msa_clei_s_h(abs_diff0, 15);
    cmp = cmp & one;
    mask0 += cmp;
    cmp = __msa_clei_s_h(abs_diff0, 7);
    cmp = cmp & one;
    mask0 += cmp;
    cmp = abs_diff0 < shift_inc1_vec;
    cmp = cmp & one;
    mask0 += cmp;
    cmp = __msa_clei_s_h(abs_diff1, 15);
    cmp = cmp & one;
    mask1 += cmp;
    cmp = __msa_clei_s_h(abs_diff1, 7);
    cmp = cmp & one;
    mask1 += cmp;
    cmp = abs_diff1 < shift_inc1_vec;
    cmp = cmp & one;
    mask1 += cmp;
    temp0_h = __msa_clei_s_h(diff0, 0);
    temp0_h = temp0_h & four;
    mask0 += temp0_h;
    temp1_h = __msa_clei_s_h(diff1, 0);
    temp1_h = temp1_h & four;
    mask1 += temp1_h;
    VSHF_H2_SH(adj_val, adj_val, adj_val, adj_val, mask0, mask1, adjust0,
               adjust1);
    temp2_h = __msa_ceqi_h(adjust0, 0);
    temp3_h = __msa_ceqi_h(adjust1, 0);
    adjust0 = (v8i16)__msa_bmnz_v((v16u8)adjust0, (v16u8)diff0, (v16u8)temp2_h);
    adjust1 = (v8i16)__msa_bmnz_v((v16u8)adjust1, (v16u8)diff1, (v16u8)temp3_h);
    ADD2(col_sum0, adjust0, col_sum1, adjust1, col_sum0, col_sum1);
    UNPCK_UB_SH(sig1, temp0_h, temp1_h);
    ADD2(temp0_h, adjust0, temp1_h, adjust1, temp0_h, temp1_h);
    MAXI_SH2_SH(temp0_h, temp1_h, 0);
    SAT_UH2_SH(temp0_h, temp1_h, 7);
    temp2_h = (v8i16)__msa_pckev_b((v16i8)temp3_h, (v16i8)temp2_h);
    running_avg_y = (v16u8)__msa_pckev_b((v16i8)temp1_h, (v16i8)temp0_h);
    running_avg_y =
        __msa_bmnz_v(running_avg_y, mc_running_avg_y1, (v16u8)temp2_h);
    ST_UB(running_avg_y, running_avg_y_ptr);
    sig_ptr += sig_stride;
    mc_running_avg_y_ptr += mc_avg_y_stride;
    running_avg_y_ptr += avg_y_stride;
  }

  col_sum0 = __msa_min_s_h(col_sum0, val_127);
  col_sum1 = __msa_min_s_h(col_sum1, val_127);
  temp0_h = col_sum0 + col_sum1;
  temp0_w = __msa_hadd_s_w(temp0_h, temp0_h);
  temp0_d = __msa_hadd_s_d(temp0_w, temp0_w);
  temp1_d = __msa_splati_d(temp0_d, 1);
  temp0_d += temp1_d;
  sum_diff = __msa_copy_s_w((v4i32)temp0_d, 0);
  sig_ptr -= sig_stride * 16;
  mc_running_avg_y_ptr -= mc_avg_y_stride * 16;
  running_avg_y_ptr -= avg_y_stride * 16;

  if (increase_denoising) {
    sum_diff_thresh = SUM_DIFF_THRESHOLD_HIGH;
  }

  if (abs(sum_diff) > sum_diff_thresh) {
    delta = ((abs(sum_diff) - sum_diff_thresh) >> 8) + 1;
    delta_vec = __msa_fill_h(delta);
    if (delta < 4) {
      for (cnt = 8; cnt--;) {
        running_avg_y = LD_UB(running_avg_y_ptr);
        mc_running_avg_y0 = LD_UB(mc_running_avg_y_ptr);
        sig0 = LD_UB(sig_ptr);
        sig_ptr += sig_stride;
        mc_running_avg_y_ptr += mc_avg_y_stride;
        running_avg_y_ptr += avg_y_stride;
        mc_running_avg_y1 = LD_UB(mc_running_avg_y_ptr);
        sig1 = LD_UB(sig_ptr);
        running_avg_y1 = LD_UB(running_avg_y_ptr);
        ILVRL_B2_UB(mc_running_avg_y0, sig0, coeff0, coeff1);
        HSUB_UB2_SH(coeff0, coeff1, diff0, diff1);
        abs_diff0 = __msa_add_a_h(diff0, zero);
        abs_diff1 = __msa_add_a_h(diff1, zero);
        temp0_h = abs_diff0 < delta_vec;
        temp1_h = abs_diff1 < delta_vec;
        abs_diff0 = (v8i16)__msa_bmz_v((v16u8)abs_diff0, (v16u8)delta_vec,
                                       (v16u8)temp0_h);
        abs_diff1 = (v8i16)__msa_bmz_v((v16u8)abs_diff1, (v16u8)delta_vec,
                                       (v16u8)temp1_h);
        SUB2(zero, abs_diff0, zero, abs_diff1, abs_diff_neg0, abs_diff_neg1);
        abs_diff_neg0 = zero - abs_diff0;
        abs_diff_neg1 = zero - abs_diff1;
        temp0_h = __msa_clei_s_h(diff0, 0);
        temp1_h = __msa_clei_s_h(diff1, 0);
        adjust0 = (v8i16)__msa_bmnz_v((v16u8)abs_diff0, (v16u8)abs_diff_neg0,
                                      (v16u8)temp0_h);
        adjust1 = (v8i16)__msa_bmnz_v((v16u8)abs_diff1, (v16u8)abs_diff_neg1,
                                      (v16u8)temp1_h);
        ILVRL_B2_SH(zero, running_avg_y, temp2_h, temp3_h);
        ADD2(temp2_h, adjust0, temp3_h, adjust1, adjust2, adjust3);
        MAXI_SH2_SH(adjust2, adjust3, 0);
        SAT_UH2_SH(adjust2, adjust3, 7);
        temp0_h = __msa_ceqi_h(diff0, 0);
        temp1_h = __msa_ceqi_h(diff1, 0);
        adjust2 =
            (v8i16)__msa_bmz_v((v16u8)adjust2, (v16u8)temp2_h, (v16u8)temp0_h);
        adjust3 =
            (v8i16)__msa_bmz_v((v16u8)adjust3, (v16u8)temp3_h, (v16u8)temp1_h);
        adjust0 =
            (v8i16)__msa_bmnz_v((v16u8)adjust0, (v16u8)zero, (v16u8)temp0_h);
        adjust1 =
            (v8i16)__msa_bmnz_v((v16u8)adjust1, (v16u8)zero, (v16u8)temp1_h);
        ADD2(col_sum2, adjust0, col_sum3, adjust1, col_sum2, col_sum3);
        running_avg_y = (v16u8)__msa_pckev_b((v16i8)adjust3, (v16i8)adjust2);
        ST_UB(running_avg_y, running_avg_y_ptr - avg_y_stride);
        ILVRL_B2_UB(mc_running_avg_y1, sig1, coeff0, coeff1);
        HSUB_UB2_SH(coeff0, coeff1, diff0, diff1);
        abs_diff0 = __msa_add_a_h(diff0, zero);
        abs_diff1 = __msa_add_a_h(diff1, zero);
        temp0_h = abs_diff0 < delta_vec;
        temp1_h = abs_diff1 < delta_vec;
        abs_diff0 = (v8i16)__msa_bmz_v((v16u8)abs_diff0, (v16u8)delta_vec,
                                       (v16u8)temp0_h);
        abs_diff1 = (v8i16)__msa_bmz_v((v16u8)abs_diff1, (v16u8)delta_vec,
                                       (v16u8)temp1_h);
        SUB2(zero, abs_diff0, zero, abs_diff1, abs_diff_neg0, abs_diff_neg1);
        temp0_h = __msa_clei_s_h(diff0, 0);
        temp1_h = __msa_clei_s_h(diff1, 0);
        adjust0 = (v8i16)__msa_bmnz_v((v16u8)abs_diff0, (v16u8)abs_diff_neg0,
                                      (v16u8)temp0_h);
        adjust1 = (v8i16)__msa_bmnz_v((v16u8)abs_diff1, (v16u8)abs_diff_neg1,
                                      (v16u8)temp1_h);
        ILVRL_H2_SH(zero, running_avg_y1, temp2_h, temp3_h);
        ADD2(temp2_h, adjust0, temp3_h, adjust1, adjust2, adjust3);
        MAXI_SH2_SH(adjust2, adjust3, 0);
        SAT_UH2_SH(adjust2, adjust3, 7);
        temp0_h = __msa_ceqi_h(diff0, 0);
        temp1_h = __msa_ceqi_h(diff1, 0);
        adjust2 =
            (v8i16)__msa_bmz_v((v16u8)adjust2, (v16u8)temp2_h, (v16u8)temp0_h);
        adjust3 =
            (v8i16)__msa_bmz_v((v16u8)adjust3, (v16u8)temp3_h, (v16u8)temp1_h);
        adjust0 =
            (v8i16)__msa_bmz_v((v16u8)adjust0, (v16u8)zero, (v16u8)temp0_h);
        adjust1 =
            (v8i16)__msa_bmz_v((v16u8)adjust1, (v16u8)zero, (v16u8)temp1_h);
        ADD2(col_sum2, adjust0, col_sum3, adjust1, col_sum2, col_sum3);
        running_avg_y = (v16u8)__msa_pckev_b((v16i8)adjust3, (v16i8)adjust2);
        ST_UB(running_avg_y, running_avg_y_ptr);
        running_avg_y_ptr += avg_y_stride;
      }

      col_sum2 = __msa_min_s_h(col_sum2, val_127);
      col_sum3 = __msa_min_s_h(col_sum3, val_127);
      temp0_h = col_sum2 + col_sum3;
      temp0_w = __msa_hadd_s_w(temp0_h, temp0_h);
      temp0_d = __msa_hadd_s_d(temp0_w, temp0_w);
      temp1_d = __msa_splati_d(temp0_d, 1);
      temp0_d += (v2i64)temp1_d;
      sum_diff = __msa_copy_s_w((v4i32)temp0_d, 0);
      if (abs(sum_diff) > SUM_DIFF_THRESHOLD) {
        return COPY_BLOCK;
      }
    } else {
      return COPY_BLOCK;
    }
  }

  LD_UB8(sig_start, sig_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  sig_start += (8 * sig_stride);
  LD_UB8(sig_start, sig_stride, src8, src9, src10, src11, src12, src13, src14,
         src15);

  ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, running_avg_y_start,
         avg_y_stride);
  running_avg_y_start += (8 * avg_y_stride);
  ST_UB8(src8, src9, src10, src11, src12, src13, src14, src15,
         running_avg_y_start, avg_y_stride);

  return FILTER_BLOCK;
}

int32_t vp8_denoiser_filter_uv_msa(
    uint8_t *mc_running_avg_y_ptr, int32_t mc_avg_y_stride,
    uint8_t *running_avg_y_ptr, int32_t avg_y_stride, uint8_t *sig_ptr,
    int32_t sig_stride, uint32_t motion_magnitude, int32_t increase_denoising) {
  uint8_t *running_avg_y_start = running_avg_y_ptr;
  uint8_t *sig_start = sig_ptr;
  int32_t cnt = 0;
  int32_t sum_diff = 0;
  int32_t shift_inc1 = 3;
  int32_t delta = 0;
  int32_t sum_block = 0;
  int32_t sum_diff_thresh;
  int64_t dst0, dst1, src0, src1, src2, src3;
  v16u8 mc_running_avg_y0, running_avg_y, sig0;
  v16u8 mc_running_avg_y1, running_avg_y1, sig1;
  v16u8 sig2, sig3, sig4, sig5, sig6, sig7;
  v16u8 coeff0;
  v8i16 diff0, abs_diff0, abs_diff_neg0;
  v8i16 adjust0, adjust2;
  v8i16 shift_inc1_vec = { 0 };
  v8i16 col_sum0 = { 0 };
  v8i16 temp0_h, temp2_h, cmp, delta_vec;
  v4i32 temp0_w;
  v2i64 temp0_d, temp1_d;
  v16i8 zero = { 0 };
  v8i16 one = __msa_ldi_h(1);
  v8i16 four = __msa_ldi_h(4);
  v8i16 adj_val = { 6, 4, 3, 0, -6, -4, -3, 0 };

  sig0 = LD_UB(sig_ptr);
  sig_ptr += sig_stride;
  temp0_h = (v8i16)__msa_ilvr_b(zero, (v16i8)sig0);
  sig1 = LD_UB(sig_ptr);
  sig_ptr += sig_stride;
  temp0_h += (v8i16)__msa_ilvr_b(zero, (v16i8)sig1);
  sig2 = LD_UB(sig_ptr);
  sig_ptr += sig_stride;
  temp0_h += (v8i16)__msa_ilvr_b(zero, (v16i8)sig2);
  sig3 = LD_UB(sig_ptr);
  sig_ptr += sig_stride;
  temp0_h += (v8i16)__msa_ilvr_b(zero, (v16i8)sig3);
  sig4 = LD_UB(sig_ptr);
  sig_ptr += sig_stride;
  temp0_h += (v8i16)__msa_ilvr_b(zero, (v16i8)sig4);
  sig5 = LD_UB(sig_ptr);
  sig_ptr += sig_stride;
  temp0_h += (v8i16)__msa_ilvr_b(zero, (v16i8)sig5);
  sig6 = LD_UB(sig_ptr);
  sig_ptr += sig_stride;
  temp0_h += (v8i16)__msa_ilvr_b(zero, (v16i8)sig6);
  sig7 = LD_UB(sig_ptr);
  sig_ptr += sig_stride;
  temp0_h += (v8i16)__msa_ilvr_b(zero, (v16i8)sig7);
  temp0_w = __msa_hadd_s_w(temp0_h, temp0_h);
  temp0_d = __msa_hadd_s_d(temp0_w, temp0_w);
  temp1_d = __msa_splati_d(temp0_d, 1);
  temp0_d += temp1_d;
  sum_block = __msa_copy_s_w((v4i32)temp0_d, 0);
  sig_ptr -= sig_stride * 8;

  if (abs(sum_block - (128 * 8 * 8)) < SUM_DIFF_FROM_AVG_THRESH_UV) {
    return COPY_BLOCK;
  }

  if (motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD) {
    adj_val = __msa_add_a_h(adj_val, one);

    if (increase_denoising) {
      adj_val = __msa_add_a_h(adj_val, one);
      shift_inc1 = 4;
    }

    temp0_h = (v8i16)zero - adj_val;
    adj_val = (v8i16)__msa_ilvev_d((v2i64)temp0_h, (v2i64)adj_val);
  }

  adj_val = __msa_insert_h(adj_val, 3, cnt);
  adj_val = __msa_insert_h(adj_val, 7, cnt);
  shift_inc1_vec = __msa_fill_h(shift_inc1);
  for (cnt = 4; cnt--;) {
    v8i16 mask0 = { 0 };
    mc_running_avg_y0 = LD_UB(mc_running_avg_y_ptr);
    sig0 = LD_UB(sig_ptr);
    sig_ptr += sig_stride;
    mc_running_avg_y_ptr += mc_avg_y_stride;
    mc_running_avg_y1 = LD_UB(mc_running_avg_y_ptr);
    sig1 = LD_UB(sig_ptr);
    coeff0 = (v16u8)__msa_ilvr_b((v16i8)mc_running_avg_y0, (v16i8)sig0);
    diff0 = __msa_hsub_u_h(coeff0, coeff0);
    abs_diff0 = __msa_add_a_h(diff0, (v8i16)zero);
    cmp = __msa_clei_s_h(abs_diff0, 15);
    cmp = cmp & one;
    mask0 += cmp;
    cmp = __msa_clei_s_h(abs_diff0, 7);
    cmp = cmp & one;
    mask0 += cmp;
    cmp = abs_diff0 < shift_inc1_vec;
    cmp = cmp & one;
    mask0 += cmp;
    temp0_h = __msa_clei_s_h(diff0, 0);
    temp0_h = temp0_h & four;
    mask0 += temp0_h;
    adjust0 = __msa_vshf_h(mask0, adj_val, adj_val);
    temp2_h = __msa_ceqi_h(adjust0, 0);
    adjust0 = (v8i16)__msa_bmnz_v((v16u8)adjust0, (v16u8)diff0, (v16u8)temp2_h);
    col_sum0 += adjust0;
    temp0_h = (v8i16)__msa_ilvr_b(zero, (v16i8)sig0);
    temp0_h += adjust0;
    temp0_h = __msa_maxi_s_h(temp0_h, 0);
    temp0_h = (v8i16)__msa_sat_u_h((v8u16)temp0_h, 7);
    temp2_h = (v8i16)__msa_pckev_b((v16i8)temp2_h, (v16i8)temp2_h);
    running_avg_y = (v16u8)__msa_pckev_b((v16i8)temp0_h, (v16i8)temp0_h);
    running_avg_y =
        __msa_bmnz_v(running_avg_y, mc_running_avg_y0, (v16u8)temp2_h);
    dst0 = __msa_copy_s_d((v2i64)running_avg_y, 0);
    SD(dst0, running_avg_y_ptr);
    running_avg_y_ptr += avg_y_stride;

    mask0 = __msa_ldi_h(0);
    coeff0 = (v16u8)__msa_ilvr_b((v16i8)mc_running_avg_y1, (v16i8)sig1);
    diff0 = __msa_hsub_u_h(coeff0, coeff0);
    abs_diff0 = __msa_add_a_h(diff0, (v8i16)zero);
    cmp = __msa_clei_s_h(abs_diff0, 15);
    cmp = cmp & one;
    mask0 += cmp;
    cmp = __msa_clei_s_h(abs_diff0, 7);
    cmp = cmp & one;
    mask0 += cmp;
    cmp = abs_diff0 < shift_inc1_vec;
    cmp = cmp & one;
    mask0 += cmp;
    temp0_h = __msa_clei_s_h(diff0, 0);
    temp0_h = temp0_h & four;
    mask0 += temp0_h;
    adjust0 = __msa_vshf_h(mask0, adj_val, adj_val);
    temp2_h = __msa_ceqi_h(adjust0, 0);
    adjust0 = (v8i16)__msa_bmnz_v((v16u8)adjust0, (v16u8)diff0, (v16u8)temp2_h);
    col_sum0 += adjust0;
    temp0_h = (v8i16)__msa_ilvr_b(zero, (v16i8)sig1);
    temp0_h += adjust0;
    temp0_h = __msa_maxi_s_h(temp0_h, 0);
    temp0_h = (v8i16)__msa_sat_u_h((v8u16)temp0_h, 7);

    temp2_h = (v8i16)__msa_pckev_b((v16i8)temp2_h, (v16i8)temp2_h);
    running_avg_y = (v16u8)__msa_pckev_b((v16i8)temp0_h, (v16i8)temp0_h);
    running_avg_y =
        __msa_bmnz_v(running_avg_y, mc_running_avg_y1, (v16u8)temp2_h);
    dst1 = __msa_copy_s_d((v2i64)running_avg_y, 0);
    SD(dst1, running_avg_y_ptr);

    sig_ptr += sig_stride;
    mc_running_avg_y_ptr += mc_avg_y_stride;
    running_avg_y_ptr += avg_y_stride;
  }

  temp0_h = col_sum0;
  temp0_w = __msa_hadd_s_w(temp0_h, temp0_h);
  temp0_d = __msa_hadd_s_d(temp0_w, temp0_w);
  temp1_d = __msa_splati_d(temp0_d, 1);
  temp0_d += temp1_d;
  sum_diff = __msa_copy_s_w((v4i32)temp0_d, 0);
  sig_ptr -= sig_stride * 8;
  mc_running_avg_y_ptr -= mc_avg_y_stride * 8;
  running_avg_y_ptr -= avg_y_stride * 8;
  sum_diff_thresh = SUM_DIFF_THRESHOLD_UV;

  if (increase_denoising) {
    sum_diff_thresh = SUM_DIFF_THRESHOLD_HIGH_UV;
  }

  if (abs(sum_diff) > sum_diff_thresh) {
    delta = ((abs(sum_diff) - sum_diff_thresh) >> 8) + 1;
    delta_vec = __msa_fill_h(delta);
    if (delta < 4) {
      for (cnt = 4; cnt--;) {
        running_avg_y = LD_UB(running_avg_y_ptr);
        mc_running_avg_y0 = LD_UB(mc_running_avg_y_ptr);
        sig0 = LD_UB(sig_ptr);
        /* Update pointers for next iteration. */
        sig_ptr += sig_stride;
        mc_running_avg_y_ptr += mc_avg_y_stride;
        running_avg_y_ptr += avg_y_stride;

        mc_running_avg_y1 = LD_UB(mc_running_avg_y_ptr);
        sig1 = LD_UB(sig_ptr);
        running_avg_y1 = LD_UB(running_avg_y_ptr);

        coeff0 = (v16u8)__msa_ilvr_b((v16i8)mc_running_avg_y0, (v16i8)sig0);
        diff0 = __msa_hsub_u_h(coeff0, coeff0);
        abs_diff0 = __msa_add_a_h(diff0, (v8i16)zero);
        temp0_h = delta_vec < abs_diff0;
        abs_diff0 = (v8i16)__msa_bmnz_v((v16u8)abs_diff0, (v16u8)delta_vec,
                                        (v16u8)temp0_h);
        abs_diff_neg0 = (v8i16)zero - abs_diff0;
        temp0_h = __msa_clei_s_h(diff0, 0);
        adjust0 = (v8i16)__msa_bmz_v((v16u8)abs_diff0, (v16u8)abs_diff_neg0,
                                     (v16u8)temp0_h);
        temp2_h = (v8i16)__msa_ilvr_b(zero, (v16i8)running_avg_y);
        adjust2 = temp2_h + adjust0;
        adjust2 = __msa_maxi_s_h(adjust2, 0);
        adjust2 = (v8i16)__msa_sat_u_h((v8u16)adjust2, 7);
        temp0_h = __msa_ceqi_h(diff0, 0);
        adjust2 =
            (v8i16)__msa_bmnz_v((v16u8)adjust2, (v16u8)temp2_h, (v16u8)temp0_h);
        adjust0 =
            (v8i16)__msa_bmnz_v((v16u8)adjust0, (v16u8)zero, (v16u8)temp0_h);
        col_sum0 += adjust0;
        running_avg_y = (v16u8)__msa_pckev_b((v16i8)adjust2, (v16i8)adjust2);
        dst0 = __msa_copy_s_d((v2i64)running_avg_y, 0);
        SD(dst0, running_avg_y_ptr - avg_y_stride);

        coeff0 = (v16u8)__msa_ilvr_b((v16i8)mc_running_avg_y1, (v16i8)sig1);
        diff0 = __msa_hsub_u_h(coeff0, coeff0);
        abs_diff0 = __msa_add_a_h(diff0, (v8i16)zero);
        temp0_h = delta_vec < abs_diff0;
        abs_diff0 = (v8i16)__msa_bmnz_v((v16u8)abs_diff0, (v16u8)delta_vec,
                                        (v16u8)temp0_h);
        abs_diff_neg0 = (v8i16)zero - abs_diff0;
        temp0_h = __msa_clei_s_h(diff0, 0);
        adjust0 = (v8i16)__msa_bmz_v((v16u8)abs_diff0, (v16u8)abs_diff_neg0,
                                     (v16u8)temp0_h);
        temp2_h = (v8i16)__msa_ilvr_b(zero, (v16i8)running_avg_y1);
        adjust2 = temp2_h + adjust0;
        adjust2 = __msa_maxi_s_h(adjust2, 0);
        adjust2 = (v8i16)__msa_sat_u_h((v8u16)adjust2, 7);
        temp0_h = __msa_ceqi_h(diff0, 0);
        adjust2 =
            (v8i16)__msa_bmnz_v((v16u8)adjust2, (v16u8)temp2_h, (v16u8)temp0_h);
        adjust0 =
            (v8i16)__msa_bmnz_v((v16u8)adjust0, (v16u8)zero, (v16u8)temp0_h);
        col_sum0 += adjust0;
        running_avg_y = (v16u8)__msa_pckev_b((v16i8)adjust2, (v16i8)adjust2);
        dst1 = __msa_copy_s_d((v2i64)running_avg_y, 0);
        SD(dst1, running_avg_y_ptr);
        running_avg_y_ptr += avg_y_stride;
      }

      temp0_h = col_sum0;
      temp0_w = __msa_hadd_s_w(temp0_h, temp0_h);
      temp0_d = __msa_hadd_s_d(temp0_w, temp0_w);
      temp1_d = __msa_splati_d(temp0_d, 1);
      temp0_d += temp1_d;
      sum_diff = __msa_copy_s_w((v4i32)temp0_d, 0);

      if (abs(sum_diff) > sum_diff_thresh) {
        return COPY_BLOCK;
      }
    } else {
      return COPY_BLOCK;
    }
  }

  LD4(sig_start, sig_stride, src0, src1, src2, src3);
  sig_start += (4 * sig_stride);
  SD4(src0, src1, src2, src3, running_avg_y_start, avg_y_stride);
  running_avg_y_start += (4 * avg_y_stride);

  LD4(sig_start, sig_stride, src0, src1, src2, src3);
  SD4(src0, src1, src2, src3, running_avg_y_start, avg_y_stride);

  return FILTER_BLOCK;
}
