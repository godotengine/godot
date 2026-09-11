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

int32_t vp8_block_error_msa(int16_t *coeff_ptr, int16_t *dq_coeff_ptr) {
  int32_t err = 0;
  uint32_t loop_cnt;
  v8i16 coeff, dq_coeff, coeff0, coeff1;
  v4i32 diff0, diff1;
  v2i64 err0 = { 0 };
  v2i64 err1 = { 0 };

  for (loop_cnt = 2; loop_cnt--;) {
    coeff = LD_SH(coeff_ptr);
    dq_coeff = LD_SH(dq_coeff_ptr);
    ILVRL_H2_SH(coeff, dq_coeff, coeff0, coeff1);
    HSUB_UH2_SW(coeff0, coeff1, diff0, diff1);
    DPADD_SD2_SD(diff0, diff1, err0, err1);
    coeff_ptr += 8;
    dq_coeff_ptr += 8;
  }

  err0 += __msa_splati_d(err0, 1);
  err1 += __msa_splati_d(err1, 1);
  err = __msa_copy_s_d(err0, 0);
  err += __msa_copy_s_d(err1, 0);

  return err;
}

int32_t vp8_mbblock_error_msa(MACROBLOCK *mb, int32_t dc) {
  BLOCK *be;
  BLOCKD *bd;
  int16_t *coeff_ptr, *dq_coeff_ptr;
  int32_t err = 0;
  uint32_t loop_cnt;
  v8i16 coeff, coeff0, coeff1, coeff2, coeff3, coeff4;
  v8i16 dq_coeff, dq_coeff2, dq_coeff3, dq_coeff4;
  v4i32 diff0, diff1;
  v2i64 err0, err1;
  v16u8 zero = { 0 };
  v16u8 mask0 = (v16u8)__msa_ldi_b(255);

  if (1 == dc) {
    mask0 = (v16u8)__msa_insve_w((v4i32)mask0, 0, (v4i32)zero);
  }

  for (loop_cnt = 0; loop_cnt < 8; ++loop_cnt) {
    be = &mb->block[2 * loop_cnt];
    bd = &mb->e_mbd.block[2 * loop_cnt];
    coeff_ptr = be->coeff;
    dq_coeff_ptr = bd->dqcoeff;
    coeff = LD_SH(coeff_ptr);
    dq_coeff = LD_SH(dq_coeff_ptr);
    coeff_ptr += 8;
    dq_coeff_ptr += 8;
    coeff2 = LD_SH(coeff_ptr);
    dq_coeff2 = LD_SH(dq_coeff_ptr);
    be = &mb->block[2 * loop_cnt + 1];
    bd = &mb->e_mbd.block[2 * loop_cnt + 1];
    coeff_ptr = be->coeff;
    dq_coeff_ptr = bd->dqcoeff;
    coeff3 = LD_SH(coeff_ptr);
    dq_coeff3 = LD_SH(dq_coeff_ptr);
    coeff_ptr += 8;
    dq_coeff_ptr += 8;
    coeff4 = LD_SH(coeff_ptr);
    dq_coeff4 = LD_SH(dq_coeff_ptr);
    ILVRL_H2_SH(coeff, dq_coeff, coeff0, coeff1);
    HSUB_UH2_SW(coeff0, coeff1, diff0, diff1);
    diff0 = (v4i32)__msa_bmnz_v(zero, (v16u8)diff0, mask0);
    DOTP_SW2_SD(diff0, diff1, diff0, diff1, err0, err1);
    ILVRL_H2_SH(coeff2, dq_coeff2, coeff0, coeff1);
    HSUB_UH2_SW(coeff0, coeff1, diff0, diff1);
    DPADD_SD2_SD(diff0, diff1, err0, err1);
    err0 += __msa_splati_d(err0, 1);
    err1 += __msa_splati_d(err1, 1);
    err += __msa_copy_s_d(err0, 0);
    err += __msa_copy_s_d(err1, 0);

    ILVRL_H2_SH(coeff3, dq_coeff3, coeff0, coeff1);
    HSUB_UH2_SW(coeff0, coeff1, diff0, diff1);
    diff0 = (v4i32)__msa_bmnz_v(zero, (v16u8)diff0, mask0);
    DOTP_SW2_SD(diff0, diff1, diff0, diff1, err0, err1);
    ILVRL_H2_SH(coeff4, dq_coeff4, coeff0, coeff1);
    HSUB_UH2_SW(coeff0, coeff1, diff0, diff1);
    DPADD_SD2_SD(diff0, diff1, err0, err1);
    err0 += __msa_splati_d(err0, 1);
    err1 += __msa_splati_d(err1, 1);
    err += __msa_copy_s_d(err0, 0);
    err += __msa_copy_s_d(err1, 0);
  }

  return err;
}

int32_t vp8_mbuverror_msa(MACROBLOCK *mb) {
  BLOCK *be;
  BLOCKD *bd;
  int16_t *coeff_ptr, *dq_coeff_ptr;
  int32_t err = 0;
  uint32_t loop_cnt;
  v8i16 coeff, coeff0, coeff1, coeff2, coeff3, coeff4;
  v8i16 dq_coeff, dq_coeff2, dq_coeff3, dq_coeff4;
  v4i32 diff0, diff1;
  v2i64 err0, err1, err_dup0, err_dup1;

  for (loop_cnt = 16; loop_cnt < 24; loop_cnt += 2) {
    be = &mb->block[loop_cnt];
    bd = &mb->e_mbd.block[loop_cnt];
    coeff_ptr = be->coeff;
    dq_coeff_ptr = bd->dqcoeff;
    coeff = LD_SH(coeff_ptr);
    dq_coeff = LD_SH(dq_coeff_ptr);
    coeff_ptr += 8;
    dq_coeff_ptr += 8;
    coeff2 = LD_SH(coeff_ptr);
    dq_coeff2 = LD_SH(dq_coeff_ptr);
    be = &mb->block[loop_cnt + 1];
    bd = &mb->e_mbd.block[loop_cnt + 1];
    coeff_ptr = be->coeff;
    dq_coeff_ptr = bd->dqcoeff;
    coeff3 = LD_SH(coeff_ptr);
    dq_coeff3 = LD_SH(dq_coeff_ptr);
    coeff_ptr += 8;
    dq_coeff_ptr += 8;
    coeff4 = LD_SH(coeff_ptr);
    dq_coeff4 = LD_SH(dq_coeff_ptr);

    ILVRL_H2_SH(coeff, dq_coeff, coeff0, coeff1);
    HSUB_UH2_SW(coeff0, coeff1, diff0, diff1);
    DOTP_SW2_SD(diff0, diff1, diff0, diff1, err0, err1);

    ILVRL_H2_SH(coeff2, dq_coeff2, coeff0, coeff1);
    HSUB_UH2_SW(coeff0, coeff1, diff0, diff1);
    DPADD_SD2_SD(diff0, diff1, err0, err1);
    err_dup0 = __msa_splati_d(err0, 1);
    err_dup1 = __msa_splati_d(err1, 1);
    ADD2(err0, err_dup0, err1, err_dup1, err0, err1);
    err += __msa_copy_s_d(err0, 0);
    err += __msa_copy_s_d(err1, 0);

    ILVRL_H2_SH(coeff3, dq_coeff3, coeff0, coeff1);
    HSUB_UH2_SW(coeff0, coeff1, diff0, diff1);
    DOTP_SW2_SD(diff0, diff1, diff0, diff1, err0, err1);
    ILVRL_H2_SH(coeff4, dq_coeff4, coeff0, coeff1);
    HSUB_UH2_SW(coeff0, coeff1, diff0, diff1);
    DPADD_SD2_SD(diff0, diff1, err0, err1);
    err_dup0 = __msa_splati_d(err0, 1);
    err_dup1 = __msa_splati_d(err1, 1);
    ADD2(err0, err_dup0, err1, err_dup1, err0, err1);
    err += __msa_copy_s_d(err0, 0);
    err += __msa_copy_s_d(err1, 0);
  }

  return err;
}
