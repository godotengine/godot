/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "./vp9_rtcd.h"
#include "vpx_dsp/mips/macros_msa.h"

#define BLOCK_ERROR_BLOCKSIZE_MSA(BSize)                                     \
  static int64_t block_error_##BSize##size_msa(                              \
      const int16_t *coeff_ptr, const int16_t *dq_coeff_ptr, int64_t *ssz) { \
    int64_t err = 0;                                                         \
    uint32_t loop_cnt;                                                       \
    v8i16 coeff, dq_coeff, coeff_r_h, coeff_l_h;                             \
    v4i32 diff_r, diff_l, coeff_r_w, coeff_l_w;                              \
    v2i64 sq_coeff_r, sq_coeff_l;                                            \
    v2i64 err0, err_dup0, err1, err_dup1;                                    \
                                                                             \
    coeff = LD_SH(coeff_ptr);                                                \
    dq_coeff = LD_SH(dq_coeff_ptr);                                          \
    UNPCK_SH_SW(coeff, coeff_r_w, coeff_l_w);                                \
    ILVRL_H2_SH(coeff, dq_coeff, coeff_r_h, coeff_l_h);                      \
    HSUB_UH2_SW(coeff_r_h, coeff_l_h, diff_r, diff_l);                       \
    DOTP_SW2_SD(coeff_r_w, coeff_l_w, coeff_r_w, coeff_l_w, sq_coeff_r,      \
                sq_coeff_l);                                                 \
    DOTP_SW2_SD(diff_r, diff_l, diff_r, diff_l, err0, err1);                 \
                                                                             \
    coeff = LD_SH(coeff_ptr + 8);                                            \
    dq_coeff = LD_SH(dq_coeff_ptr + 8);                                      \
    UNPCK_SH_SW(coeff, coeff_r_w, coeff_l_w);                                \
    ILVRL_H2_SH(coeff, dq_coeff, coeff_r_h, coeff_l_h);                      \
    HSUB_UH2_SW(coeff_r_h, coeff_l_h, diff_r, diff_l);                       \
    DPADD_SD2_SD(coeff_r_w, coeff_l_w, sq_coeff_r, sq_coeff_l);              \
    DPADD_SD2_SD(diff_r, diff_l, err0, err1);                                \
                                                                             \
    coeff_ptr += 16;                                                         \
    dq_coeff_ptr += 16;                                                      \
                                                                             \
    for (loop_cnt = ((BSize >> 4) - 1); loop_cnt--;) {                       \
      coeff = LD_SH(coeff_ptr);                                              \
      dq_coeff = LD_SH(dq_coeff_ptr);                                        \
      UNPCK_SH_SW(coeff, coeff_r_w, coeff_l_w);                              \
      ILVRL_H2_SH(coeff, dq_coeff, coeff_r_h, coeff_l_h);                    \
      HSUB_UH2_SW(coeff_r_h, coeff_l_h, diff_r, diff_l);                     \
      DPADD_SD2_SD(coeff_r_w, coeff_l_w, sq_coeff_r, sq_coeff_l);            \
      DPADD_SD2_SD(diff_r, diff_l, err0, err1);                              \
                                                                             \
      coeff = LD_SH(coeff_ptr + 8);                                          \
      dq_coeff = LD_SH(dq_coeff_ptr + 8);                                    \
      UNPCK_SH_SW(coeff, coeff_r_w, coeff_l_w);                              \
      ILVRL_H2_SH(coeff, dq_coeff, coeff_r_h, coeff_l_h);                    \
      HSUB_UH2_SW(coeff_r_h, coeff_l_h, diff_r, diff_l);                     \
      DPADD_SD2_SD(coeff_r_w, coeff_l_w, sq_coeff_r, sq_coeff_l);            \
      DPADD_SD2_SD(diff_r, diff_l, err0, err1);                              \
                                                                             \
      coeff_ptr += 16;                                                       \
      dq_coeff_ptr += 16;                                                    \
    }                                                                        \
                                                                             \
    err_dup0 = __msa_splati_d(sq_coeff_r, 1);                                \
    err_dup1 = __msa_splati_d(sq_coeff_l, 1);                                \
    sq_coeff_r += err_dup0;                                                  \
    sq_coeff_l += err_dup1;                                                  \
    *ssz = __msa_copy_s_d(sq_coeff_r, 0);                                    \
    *ssz += __msa_copy_s_d(sq_coeff_l, 0);                                   \
                                                                             \
    err_dup0 = __msa_splati_d(err0, 1);                                      \
    err_dup1 = __msa_splati_d(err1, 1);                                      \
    err0 += err_dup0;                                                        \
    err1 += err_dup1;                                                        \
    err = __msa_copy_s_d(err0, 0);                                           \
    err += __msa_copy_s_d(err1, 0);                                          \
                                                                             \
    return err;                                                              \
  }

#if !CONFIG_VP9_HIGHBITDEPTH
BLOCK_ERROR_BLOCKSIZE_MSA(16);
BLOCK_ERROR_BLOCKSIZE_MSA(64);
BLOCK_ERROR_BLOCKSIZE_MSA(256);
BLOCK_ERROR_BLOCKSIZE_MSA(1024);

int64_t vp9_block_error_msa(const tran_low_t *coeff_ptr,
                            const tran_low_t *dq_coeff_ptr, intptr_t blk_size,
                            int64_t *ssz) {
  int64_t err;
  const int16_t *coeff = (const int16_t *)coeff_ptr;
  const int16_t *dq_coeff = (const int16_t *)dq_coeff_ptr;

  switch (blk_size) {
    case 16: err = block_error_16size_msa(coeff, dq_coeff, ssz); break;
    case 64: err = block_error_64size_msa(coeff, dq_coeff, ssz); break;
    case 256: err = block_error_256size_msa(coeff, dq_coeff, ssz); break;
    case 1024: err = block_error_1024size_msa(coeff, dq_coeff, ssz); break;
    default:
      err = vp9_block_error_c(coeff_ptr, dq_coeff_ptr, blk_size, ssz);
      break;
  }

  return err;
}
#endif  // !CONFIG_VP9_HIGHBITDEPTH
