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
#include "vp8/common/mips/msa/vp8_macros_msa.h"
#include "vp8/encoder/block.h"

static int8_t fast_quantize_b_msa(int16_t *coeff_ptr, int16_t *round,
                                  int16_t *quant, int16_t *de_quant,
                                  int16_t *q_coeff, int16_t *dq_coeff) {
  int32_t cnt, eob;
  v16i8 inv_zig_zag = { 0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15 };
  v8i16 round0, round1;
  v8i16 sign_z0, sign_z1;
  v8i16 q_coeff0, q_coeff1;
  v8i16 x0, x1, de_quant0, de_quant1;
  v8i16 coeff0, coeff1, z0, z1;
  v8i16 quant0, quant1, quant2, quant3;
  v8i16 zero = { 0 };
  v8i16 inv_zig_zag0, inv_zig_zag1;
  v8i16 zigzag_mask0 = { 0, 1, 4, 8, 5, 2, 3, 6 };
  v8i16 zigzag_mask1 = { 9, 12, 13, 10, 7, 11, 14, 15 };
  v8i16 temp0_h, temp1_h, temp2_h, temp3_h;
  v4i32 temp0_w, temp1_w, temp2_w, temp3_w;

  ILVRL_B2_SH(zero, inv_zig_zag, inv_zig_zag0, inv_zig_zag1);
  eob = -1;
  LD_SH2(coeff_ptr, 8, coeff0, coeff1);
  VSHF_H2_SH(coeff0, coeff1, coeff0, coeff1, zigzag_mask0, zigzag_mask1, z0,
             z1);
  LD_SH2(round, 8, coeff0, coeff1);
  VSHF_H2_SH(coeff0, coeff1, coeff0, coeff1, zigzag_mask0, zigzag_mask1, round0,
             round1);
  LD_SH2(quant, 8, coeff0, coeff1);
  VSHF_H2_SH(coeff0, coeff1, coeff0, coeff1, zigzag_mask0, zigzag_mask1, quant0,
             quant2);
  sign_z0 = z0 >> 15;
  sign_z1 = z1 >> 15;
  x0 = __msa_add_a_h(z0, zero);
  x1 = __msa_add_a_h(z1, zero);
  ILVL_H2_SH(quant0, quant0, quant2, quant2, quant1, quant3);
  ILVR_H2_SH(quant0, quant0, quant2, quant2, quant0, quant2);
  ILVL_H2_SH(round0, x0, round1, x1, temp1_h, temp3_h);
  ILVR_H2_SH(round0, x0, round1, x1, temp0_h, temp2_h);
  DOTP_SH4_SW(temp0_h, temp1_h, temp2_h, temp3_h, quant0, quant1, quant2,
              quant3, temp0_w, temp1_w, temp2_w, temp3_w);
  SRA_4V(temp0_w, temp1_w, temp2_w, temp3_w, 16);
  PCKEV_H2_SH(temp1_w, temp0_w, temp3_w, temp2_w, x0, x1);
  x0 = x0 ^ sign_z0;
  x1 = x1 ^ sign_z1;
  SUB2(x0, sign_z0, x1, sign_z1, x0, x1);
  VSHF_H2_SH(x0, x1, x0, x1, inv_zig_zag0, inv_zig_zag1, q_coeff0, q_coeff1);
  ST_SH2(q_coeff0, q_coeff1, q_coeff, 8);
  LD_SH2(de_quant, 8, de_quant0, de_quant1);
  q_coeff0 *= de_quant0;
  q_coeff1 *= de_quant1;
  ST_SH2(q_coeff0, q_coeff1, dq_coeff, 8);

  for (cnt = 0; cnt < 16; ++cnt) {
    if ((cnt <= 7) && (x1[7 - cnt] != 0)) {
      eob = (15 - cnt);
      break;
    }

    if ((cnt > 7) && (x0[7 - (cnt - 8)] != 0)) {
      eob = (7 - (cnt - 8));
      break;
    }
  }

  return (int8_t)(eob + 1);
}

static int8_t exact_regular_quantize_b_msa(
    int16_t *zbin_boost, int16_t *coeff_ptr, int16_t *zbin, int16_t *round,
    int16_t *quant, int16_t *quant_shift, int16_t *de_quant, int16_t zbin_oq_in,
    int16_t *q_coeff, int16_t *dq_coeff) {
  int32_t cnt, eob;
  int16_t *boost_temp = zbin_boost;
  v16i8 inv_zig_zag = { 0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15 };
  v8i16 round0, round1;
  v8i16 sign_z0, sign_z1;
  v8i16 q_coeff0, q_coeff1;
  v8i16 z_bin0, z_bin1, zbin_o_q;
  v8i16 x0, x1, sign_x0, sign_x1, de_quant0, de_quant1;
  v8i16 coeff0, coeff1, z0, z1;
  v8i16 quant0, quant1, quant2, quant3;
  v8i16 zero = { 0 };
  v8i16 inv_zig_zag0, inv_zig_zag1;
  v8i16 zigzag_mask0 = { 0, 1, 4, 8, 5, 2, 3, 6 };
  v8i16 zigzag_mask1 = { 9, 12, 13, 10, 7, 11, 14, 15 };
  v8i16 temp0_h, temp1_h, temp2_h, temp3_h;
  v4i32 temp0_w, temp1_w, temp2_w, temp3_w;

  ILVRL_B2_SH(zero, inv_zig_zag, inv_zig_zag0, inv_zig_zag1);
  zbin_o_q = __msa_fill_h(zbin_oq_in);
  eob = -1;
  LD_SH2(coeff_ptr, 8, coeff0, coeff1);
  VSHF_H2_SH(coeff0, coeff1, coeff0, coeff1, zigzag_mask0, zigzag_mask1, z0,
             z1);
  LD_SH2(round, 8, coeff0, coeff1);
  VSHF_H2_SH(coeff0, coeff1, coeff0, coeff1, zigzag_mask0, zigzag_mask1, round0,
             round1);
  LD_SH2(quant, 8, coeff0, coeff1);
  VSHF_H2_SH(coeff0, coeff1, coeff0, coeff1, zigzag_mask0, zigzag_mask1, quant0,
             quant2);
  LD_SH2(zbin, 8, coeff0, coeff1);
  VSHF_H2_SH(coeff0, coeff1, coeff0, coeff1, zigzag_mask0, zigzag_mask1, z_bin0,
             z_bin1);
  sign_z0 = z0 >> 15;
  sign_z1 = z1 >> 15;
  x0 = __msa_add_a_h(z0, zero);
  x1 = __msa_add_a_h(z1, zero);
  SUB2(x0, z_bin0, x1, z_bin1, z_bin0, z_bin1);
  SUB2(z_bin0, zbin_o_q, z_bin1, zbin_o_q, z_bin0, z_bin1);
  ILVL_H2_SH(quant0, quant0, quant2, quant2, quant1, quant3);
  ILVR_H2_SH(quant0, quant0, quant2, quant2, quant0, quant2);
  ILVL_H2_SH(round0, x0, round1, x1, temp1_h, temp3_h);
  ILVR_H2_SH(round0, x0, round1, x1, temp0_h, temp2_h);
  DOTP_SH4_SW(temp0_h, temp1_h, temp2_h, temp3_h, quant0, quant1, quant2,
              quant3, temp0_w, temp1_w, temp2_w, temp3_w);
  SRA_4V(temp0_w, temp1_w, temp2_w, temp3_w, 16);
  PCKEV_H2_SH(temp1_w, temp0_w, temp3_w, temp2_w, temp0_h, temp2_h);
  LD_SH2(quant_shift, 8, coeff0, coeff1);
  VSHF_H2_SH(coeff0, coeff1, coeff0, coeff1, zigzag_mask0, zigzag_mask1, quant0,
             quant2);
  ILVL_H2_SH(quant0, quant0, quant2, quant2, quant1, quant3);
  ILVR_H2_SH(quant0, quant0, quant2, quant2, quant0, quant2);
  ADD2(x0, round0, x1, round1, x0, x1);
  ILVL_H2_SH(temp0_h, x0, temp2_h, x1, temp1_h, temp3_h);
  ILVR_H2_SH(temp0_h, x0, temp2_h, x1, temp0_h, temp2_h);
  DOTP_SH4_SW(temp0_h, temp1_h, temp2_h, temp3_h, quant0, quant1, quant2,
              quant3, temp0_w, temp1_w, temp2_w, temp3_w);
  SRA_4V(temp0_w, temp1_w, temp2_w, temp3_w, 16);
  PCKEV_H2_SH(temp1_w, temp0_w, temp3_w, temp2_w, x0, x1);
  sign_x0 = x0 ^ sign_z0;
  sign_x1 = x1 ^ sign_z1;
  SUB2(sign_x0, sign_z0, sign_x1, sign_z1, sign_x0, sign_x1);
  for (cnt = 0; cnt < 16; ++cnt) {
    if (cnt <= 7) {
      if (boost_temp[0] <= z_bin0[cnt]) {
        if (x0[cnt]) {
          eob = cnt;
          boost_temp = zbin_boost;
        } else {
          boost_temp++;
        }
      } else {
        sign_x0[cnt] = 0;
        boost_temp++;
      }
    } else {
      if (boost_temp[0] <= z_bin1[cnt - 8]) {
        if (x1[cnt - 8]) {
          eob = cnt;
          boost_temp = zbin_boost;
        } else {
          boost_temp++;
        }
      } else {
        sign_x1[cnt - 8] = 0;
        boost_temp++;
      }
    }
  }

  VSHF_H2_SH(sign_x0, sign_x1, sign_x0, sign_x1, inv_zig_zag0, inv_zig_zag1,
             q_coeff0, q_coeff1);
  ST_SH2(q_coeff0, q_coeff1, q_coeff, 8);
  LD_SH2(de_quant, 8, de_quant0, de_quant1);
  MUL2(de_quant0, q_coeff0, de_quant1, q_coeff1, de_quant0, de_quant1);
  ST_SH2(de_quant0, de_quant1, dq_coeff, 8);

  return (int8_t)(eob + 1);
}

void vp8_fast_quantize_b_msa(BLOCK *b, BLOCKD *d) {
  int16_t *coeff_ptr = b->coeff;
  int16_t *round_ptr = b->round;
  int16_t *quant_ptr = b->quant_fast;
  int16_t *qcoeff_ptr = d->qcoeff;
  int16_t *dqcoeff_ptr = d->dqcoeff;
  int16_t *dequant_ptr = d->dequant;

  *d->eob = fast_quantize_b_msa(coeff_ptr, round_ptr, quant_ptr, dequant_ptr,
                                qcoeff_ptr, dqcoeff_ptr);
}

void vp8_regular_quantize_b_msa(BLOCK *b, BLOCKD *d) {
  int16_t *zbin_boost_ptr = b->zrun_zbin_boost;
  int16_t *coeff_ptr = b->coeff;
  int16_t *zbin_ptr = b->zbin;
  int16_t *round_ptr = b->round;
  int16_t *quant_ptr = b->quant;
  int16_t *quant_shift_ptr = b->quant_shift;
  int16_t *qcoeff_ptr = d->qcoeff;
  int16_t *dqcoeff_ptr = d->dqcoeff;
  int16_t *dequant_ptr = d->dequant;
  int16_t zbin_oq_value = b->zbin_extra;

  *d->eob = exact_regular_quantize_b_msa(
      zbin_boost_ptr, coeff_ptr, zbin_ptr, round_ptr, quant_ptr,
      quant_shift_ptr, dequant_ptr, zbin_oq_value, qcoeff_ptr, dqcoeff_ptr);
}
