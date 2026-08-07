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
#include "vpx_dsp/mips/loopfilter_msa.h"

void vpx_lpf_horizontal_4_msa(uint8_t *src, int32_t pitch,
                              const uint8_t *b_limit_ptr,
                              const uint8_t *limit_ptr,
                              const uint8_t *thresh_ptr) {
  uint64_t p1_d, p0_d, q0_d, q1_d;
  v16u8 mask, hev, flat, thresh, b_limit, limit;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0, p1_out, p0_out, q0_out, q1_out;

  /* load vector elements */
  LD_UB8((src - 4 * pitch), pitch, p3, p2, p1, p0, q0, q1, q2, q3);

  thresh = (v16u8)__msa_fill_b(*thresh_ptr);
  b_limit = (v16u8)__msa_fill_b(*b_limit_ptr);
  limit = (v16u8)__msa_fill_b(*limit_ptr);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1_out, p0_out, q0_out, q1_out);

  p1_d = __msa_copy_u_d((v2i64)p1_out, 0);
  p0_d = __msa_copy_u_d((v2i64)p0_out, 0);
  q0_d = __msa_copy_u_d((v2i64)q0_out, 0);
  q1_d = __msa_copy_u_d((v2i64)q1_out, 0);
  SD4(p1_d, p0_d, q0_d, q1_d, (src - 2 * pitch), pitch);
}

void vpx_lpf_horizontal_4_dual_msa(uint8_t *src, int32_t pitch,
                                   const uint8_t *b_limit0_ptr,
                                   const uint8_t *limit0_ptr,
                                   const uint8_t *thresh0_ptr,
                                   const uint8_t *b_limit1_ptr,
                                   const uint8_t *limit1_ptr,
                                   const uint8_t *thresh1_ptr) {
  v16u8 mask, hev, flat, thresh0, b_limit0, limit0, thresh1, b_limit1, limit1;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;

  /* load vector elements */
  LD_UB8((src - 4 * pitch), pitch, p3, p2, p1, p0, q0, q1, q2, q3);

  thresh0 = (v16u8)__msa_fill_b(*thresh0_ptr);
  thresh1 = (v16u8)__msa_fill_b(*thresh1_ptr);
  thresh0 = (v16u8)__msa_ilvr_d((v2i64)thresh1, (v2i64)thresh0);

  b_limit0 = (v16u8)__msa_fill_b(*b_limit0_ptr);
  b_limit1 = (v16u8)__msa_fill_b(*b_limit1_ptr);
  b_limit0 = (v16u8)__msa_ilvr_d((v2i64)b_limit1, (v2i64)b_limit0);

  limit0 = (v16u8)__msa_fill_b(*limit0_ptr);
  limit1 = (v16u8)__msa_fill_b(*limit1_ptr);
  limit0 = (v16u8)__msa_ilvr_d((v2i64)limit1, (v2i64)limit0);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit0, b_limit0, thresh0, hev,
               mask, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1, p0, q0, q1);

  ST_UB4(p1, p0, q0, q1, (src - 2 * pitch), pitch);
}

void vpx_lpf_vertical_4_msa(uint8_t *src, int32_t pitch,
                            const uint8_t *b_limit_ptr,
                            const uint8_t *limit_ptr,
                            const uint8_t *thresh_ptr) {
  v16u8 mask, hev, flat, limit, thresh, b_limit;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v8i16 vec0, vec1, vec2, vec3;

  LD_UB8((src - 4), pitch, p3, p2, p1, p0, q0, q1, q2, q3);

  thresh = (v16u8)__msa_fill_b(*thresh_ptr);
  b_limit = (v16u8)__msa_fill_b(*b_limit_ptr);
  limit = (v16u8)__msa_fill_b(*limit_ptr);

  TRANSPOSE8x8_UB_UB(p3, p2, p1, p0, q0, q1, q2, q3, p3, p2, p1, p0, q0, q1, q2,
                     q3);
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1, p0, q0, q1);
  ILVR_B2_SH(p0, p1, q1, q0, vec0, vec1);
  ILVRL_H2_SH(vec1, vec0, vec2, vec3);

  src -= 2;
  ST4x4_UB(vec2, vec2, 0, 1, 2, 3, src, pitch);
  src += 4 * pitch;
  ST4x4_UB(vec3, vec3, 0, 1, 2, 3, src, pitch);
}

void vpx_lpf_vertical_4_dual_msa(uint8_t *src, int32_t pitch,
                                 const uint8_t *b_limit0_ptr,
                                 const uint8_t *limit0_ptr,
                                 const uint8_t *thresh0_ptr,
                                 const uint8_t *b_limit1_ptr,
                                 const uint8_t *limit1_ptr,
                                 const uint8_t *thresh1_ptr) {
  v16u8 mask, hev, flat;
  v16u8 thresh0, b_limit0, limit0, thresh1, b_limit1, limit1;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 row0, row1, row2, row3, row4, row5, row6, row7;
  v16u8 row8, row9, row10, row11, row12, row13, row14, row15;
  v8i16 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

  LD_UB8(src - 4, pitch, row0, row1, row2, row3, row4, row5, row6, row7);
  LD_UB8(src - 4 + (8 * pitch), pitch, row8, row9, row10, row11, row12, row13,
         row14, row15);

  TRANSPOSE16x8_UB_UB(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p3, p2,
                      p1, p0, q0, q1, q2, q3);

  thresh0 = (v16u8)__msa_fill_b(*thresh0_ptr);
  thresh1 = (v16u8)__msa_fill_b(*thresh1_ptr);
  thresh0 = (v16u8)__msa_ilvr_d((v2i64)thresh1, (v2i64)thresh0);

  b_limit0 = (v16u8)__msa_fill_b(*b_limit0_ptr);
  b_limit1 = (v16u8)__msa_fill_b(*b_limit1_ptr);
  b_limit0 = (v16u8)__msa_ilvr_d((v2i64)b_limit1, (v2i64)b_limit0);

  limit0 = (v16u8)__msa_fill_b(*limit0_ptr);
  limit1 = (v16u8)__msa_fill_b(*limit1_ptr);
  limit0 = (v16u8)__msa_ilvr_d((v2i64)limit1, (v2i64)limit0);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit0, b_limit0, thresh0, hev,
               mask, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1, p0, q0, q1);
  ILVR_B2_SH(p0, p1, q1, q0, tmp0, tmp1);
  ILVRL_H2_SH(tmp1, tmp0, tmp2, tmp3);
  ILVL_B2_SH(p0, p1, q1, q0, tmp0, tmp1);
  ILVRL_H2_SH(tmp1, tmp0, tmp4, tmp5);

  src -= 2;

  ST4x8_UB(tmp2, tmp3, src, pitch);
  src += (8 * pitch);
  ST4x8_UB(tmp4, tmp5, src, pitch);
}
