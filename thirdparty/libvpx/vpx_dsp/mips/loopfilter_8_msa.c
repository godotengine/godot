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

void vpx_lpf_horizontal_8_msa(uint8_t *src, int32_t pitch,
                              const uint8_t *b_limit_ptr,
                              const uint8_t *limit_ptr,
                              const uint8_t *thresh_ptr) {
  uint64_t p2_d, p1_d, p0_d, q0_d, q1_d, q2_d;
  v16u8 mask, hev, flat, thresh, b_limit, limit;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 p2_out, p1_out, p0_out, q0_out, q1_out, q2_out;
  v8i16 p2_filter8, p1_filter8, p0_filter8, q0_filter8, q1_filter8, q2_filter8;
  v8u16 p3_r, p2_r, p1_r, p0_r, q3_r, q2_r, q1_r, q0_r;
  v16i8 zero = { 0 };

  /* load vector elements */
  LD_UB8((src - 4 * pitch), pitch, p3, p2, p1, p0, q0, q1, q2, q3);

  thresh = (v16u8)__msa_fill_b(*thresh_ptr);
  b_limit = (v16u8)__msa_fill_b(*b_limit_ptr);
  limit = (v16u8)__msa_fill_b(*limit_ptr);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP9_FLAT4(p3, p2, p0, q0, q2, q3, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1_out, p0_out, q0_out, q1_out);

  flat = (v16u8)__msa_ilvr_d((v2i64)zero, (v2i64)flat);

  if (__msa_test_bz_v(flat)) {
    p1_d = __msa_copy_u_d((v2i64)p1_out, 0);
    p0_d = __msa_copy_u_d((v2i64)p0_out, 0);
    q0_d = __msa_copy_u_d((v2i64)q0_out, 0);
    q1_d = __msa_copy_u_d((v2i64)q1_out, 0);
    SD4(p1_d, p0_d, q0_d, q1_d, (src - 2 * pitch), pitch);
  } else {
    ILVR_B8_UH(zero, p3, zero, p2, zero, p1, zero, p0, zero, q0, zero, q1, zero,
               q2, zero, q3, p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r);
    VP9_FILTER8(p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r, p2_filter8,
                p1_filter8, p0_filter8, q0_filter8, q1_filter8, q2_filter8);

    /* convert 16 bit output data into 8 bit */
    PCKEV_B4_SH(zero, p2_filter8, zero, p1_filter8, zero, p0_filter8, zero,
                q0_filter8, p2_filter8, p1_filter8, p0_filter8, q0_filter8);
    PCKEV_B2_SH(zero, q1_filter8, zero, q2_filter8, q1_filter8, q2_filter8);

    /* store pixel values */
    p2_out = __msa_bmnz_v(p2, (v16u8)p2_filter8, flat);
    p1_out = __msa_bmnz_v(p1_out, (v16u8)p1_filter8, flat);
    p0_out = __msa_bmnz_v(p0_out, (v16u8)p0_filter8, flat);
    q0_out = __msa_bmnz_v(q0_out, (v16u8)q0_filter8, flat);
    q1_out = __msa_bmnz_v(q1_out, (v16u8)q1_filter8, flat);
    q2_out = __msa_bmnz_v(q2, (v16u8)q2_filter8, flat);

    p2_d = __msa_copy_u_d((v2i64)p2_out, 0);
    p1_d = __msa_copy_u_d((v2i64)p1_out, 0);
    p0_d = __msa_copy_u_d((v2i64)p0_out, 0);
    q0_d = __msa_copy_u_d((v2i64)q0_out, 0);
    q1_d = __msa_copy_u_d((v2i64)q1_out, 0);
    q2_d = __msa_copy_u_d((v2i64)q2_out, 0);

    src -= 3 * pitch;

    SD4(p2_d, p1_d, p0_d, q0_d, src, pitch);
    src += (4 * pitch);
    SD(q1_d, src);
    src += pitch;
    SD(q2_d, src);
  }
}

void vpx_lpf_horizontal_8_dual_msa(
    uint8_t *src, int32_t pitch, const uint8_t *b_limit0, const uint8_t *limit0,
    const uint8_t *thresh0, const uint8_t *b_limit1, const uint8_t *limit1,
    const uint8_t *thresh1) {
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 p2_out, p1_out, p0_out, q0_out, q1_out, q2_out;
  v16u8 flat, mask, hev, tmp, thresh, b_limit, limit;
  v8u16 p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r;
  v8u16 p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l;
  v8i16 p2_filt8_r, p1_filt8_r, p0_filt8_r, q0_filt8_r, q1_filt8_r, q2_filt8_r;
  v8i16 p2_filt8_l, p1_filt8_l, p0_filt8_l, q0_filt8_l, q1_filt8_l, q2_filt8_l;
  v16u8 zero = { 0 };

  /* load vector elements */
  LD_UB8(src - (4 * pitch), pitch, p3, p2, p1, p0, q0, q1, q2, q3);

  thresh = (v16u8)__msa_fill_b(*thresh0);
  tmp = (v16u8)__msa_fill_b(*thresh1);
  thresh = (v16u8)__msa_ilvr_d((v2i64)tmp, (v2i64)thresh);

  b_limit = (v16u8)__msa_fill_b(*b_limit0);
  tmp = (v16u8)__msa_fill_b(*b_limit1);
  b_limit = (v16u8)__msa_ilvr_d((v2i64)tmp, (v2i64)b_limit);

  limit = (v16u8)__msa_fill_b(*limit0);
  tmp = (v16u8)__msa_fill_b(*limit1);
  limit = (v16u8)__msa_ilvr_d((v2i64)tmp, (v2i64)limit);

  /* mask and hev */
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP9_FLAT4(p3, p2, p0, q0, q2, q3, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1_out, p0_out, q0_out, q1_out);

  if (__msa_test_bz_v(flat)) {
    ST_UB4(p1_out, p0_out, q0_out, q1_out, (src - 2 * pitch), pitch);
  } else {
    ILVR_B8_UH(zero, p3, zero, p2, zero, p1, zero, p0, zero, q0, zero, q1, zero,
               q2, zero, q3, p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r);
    VP9_FILTER8(p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r, p2_filt8_r,
                p1_filt8_r, p0_filt8_r, q0_filt8_r, q1_filt8_r, q2_filt8_r);

    ILVL_B4_UH(zero, p3, zero, p2, zero, p1, zero, p0, p3_l, p2_l, p1_l, p0_l);
    ILVL_B4_UH(zero, q0, zero, q1, zero, q2, zero, q3, q0_l, q1_l, q2_l, q3_l);
    VP9_FILTER8(p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l, p2_filt8_l,
                p1_filt8_l, p0_filt8_l, q0_filt8_l, q1_filt8_l, q2_filt8_l);

    /* convert 16 bit output data into 8 bit */
    PCKEV_B4_SH(p2_filt8_l, p2_filt8_r, p1_filt8_l, p1_filt8_r, p0_filt8_l,
                p0_filt8_r, q0_filt8_l, q0_filt8_r, p2_filt8_r, p1_filt8_r,
                p0_filt8_r, q0_filt8_r);
    PCKEV_B2_SH(q1_filt8_l, q1_filt8_r, q2_filt8_l, q2_filt8_r, q1_filt8_r,
                q2_filt8_r);

    /* store pixel values */
    p2_out = __msa_bmnz_v(p2, (v16u8)p2_filt8_r, flat);
    p1_out = __msa_bmnz_v(p1_out, (v16u8)p1_filt8_r, flat);
    p0_out = __msa_bmnz_v(p0_out, (v16u8)p0_filt8_r, flat);
    q0_out = __msa_bmnz_v(q0_out, (v16u8)q0_filt8_r, flat);
    q1_out = __msa_bmnz_v(q1_out, (v16u8)q1_filt8_r, flat);
    q2_out = __msa_bmnz_v(q2, (v16u8)q2_filt8_r, flat);

    src -= 3 * pitch;

    ST_UB4(p2_out, p1_out, p0_out, q0_out, src, pitch);
    src += (4 * pitch);
    ST_UB2(q1_out, q2_out, src, pitch);
    src += (2 * pitch);
  }
}

void vpx_lpf_vertical_8_msa(uint8_t *src, int32_t pitch,
                            const uint8_t *b_limit_ptr,
                            const uint8_t *limit_ptr,
                            const uint8_t *thresh_ptr) {
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 p1_out, p0_out, q0_out, q1_out;
  v16u8 flat, mask, hev, thresh, b_limit, limit;
  v8u16 p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r;
  v8i16 p2_filt8_r, p1_filt8_r, p0_filt8_r, q0_filt8_r, q1_filt8_r, q2_filt8_r;
  v16u8 zero = { 0 };
  v8i16 vec0, vec1, vec2, vec3, vec4;

  /* load vector elements */
  LD_UB8(src - 4, pitch, p3, p2, p1, p0, q0, q1, q2, q3);

  TRANSPOSE8x8_UB_UB(p3, p2, p1, p0, q0, q1, q2, q3, p3, p2, p1, p0, q0, q1, q2,
                     q3);

  thresh = (v16u8)__msa_fill_b(*thresh_ptr);
  b_limit = (v16u8)__msa_fill_b(*b_limit_ptr);
  limit = (v16u8)__msa_fill_b(*limit_ptr);

  /* mask and hev */
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  /* flat4 */
  VP9_FLAT4(p3, p2, p0, q0, q2, q3, flat);
  /* filter4 */
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1_out, p0_out, q0_out, q1_out);

  flat = (v16u8)__msa_ilvr_d((v2i64)zero, (v2i64)flat);

  if (__msa_test_bz_v(flat)) {
    /* Store 4 pixels p1-_q1 */
    ILVR_B2_SH(p0_out, p1_out, q1_out, q0_out, vec0, vec1);
    ILVRL_H2_SH(vec1, vec0, vec2, vec3);

    src -= 2;
    ST4x4_UB(vec2, vec2, 0, 1, 2, 3, src, pitch);
    src += 4 * pitch;
    ST4x4_UB(vec3, vec3, 0, 1, 2, 3, src, pitch);
  } else {
    ILVR_B8_UH(zero, p3, zero, p2, zero, p1, zero, p0, zero, q0, zero, q1, zero,
               q2, zero, q3, p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r);
    VP9_FILTER8(p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r, p2_filt8_r,
                p1_filt8_r, p0_filt8_r, q0_filt8_r, q1_filt8_r, q2_filt8_r);
    /* convert 16 bit output data into 8 bit */
    PCKEV_B4_SH(p2_filt8_r, p2_filt8_r, p1_filt8_r, p1_filt8_r, p0_filt8_r,
                p0_filt8_r, q0_filt8_r, q0_filt8_r, p2_filt8_r, p1_filt8_r,
                p0_filt8_r, q0_filt8_r);
    PCKEV_B2_SH(q1_filt8_r, q1_filt8_r, q2_filt8_r, q2_filt8_r, q1_filt8_r,
                q2_filt8_r);

    /* store pixel values */
    p2 = __msa_bmnz_v(p2, (v16u8)p2_filt8_r, flat);
    p1 = __msa_bmnz_v(p1_out, (v16u8)p1_filt8_r, flat);
    p0 = __msa_bmnz_v(p0_out, (v16u8)p0_filt8_r, flat);
    q0 = __msa_bmnz_v(q0_out, (v16u8)q0_filt8_r, flat);
    q1 = __msa_bmnz_v(q1_out, (v16u8)q1_filt8_r, flat);
    q2 = __msa_bmnz_v(q2, (v16u8)q2_filt8_r, flat);

    /* Store 6 pixels p2-_q2 */
    ILVR_B2_SH(p1, p2, q0, p0, vec0, vec1);
    ILVRL_H2_SH(vec1, vec0, vec2, vec3);
    vec4 = (v8i16)__msa_ilvr_b((v16i8)q2, (v16i8)q1);

    src -= 3;
    ST4x4_UB(vec2, vec2, 0, 1, 2, 3, src, pitch);
    ST2x4_UB(vec4, 0, src + 4, pitch);
    src += (4 * pitch);
    ST4x4_UB(vec3, vec3, 0, 1, 2, 3, src, pitch);
    ST2x4_UB(vec4, 4, src + 4, pitch);
  }
}

void vpx_lpf_vertical_8_dual_msa(uint8_t *src, int32_t pitch,
                                 const uint8_t *b_limit0, const uint8_t *limit0,
                                 const uint8_t *thresh0,
                                 const uint8_t *b_limit1, const uint8_t *limit1,
                                 const uint8_t *thresh1) {
  uint8_t *temp_src;
  v16u8 p3, p2, p1, p0, q3, q2, q1, q0;
  v16u8 p1_out, p0_out, q0_out, q1_out;
  v16u8 flat, mask, hev, thresh, b_limit, limit;
  v16u8 row4, row5, row6, row7, row12, row13, row14, row15;
  v8u16 p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r;
  v8u16 p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l;
  v8i16 p2_filt8_r, p1_filt8_r, p0_filt8_r, q0_filt8_r, q1_filt8_r, q2_filt8_r;
  v8i16 p2_filt8_l, p1_filt8_l, p0_filt8_l, q0_filt8_l, q1_filt8_l, q2_filt8_l;
  v16u8 zero = { 0 };
  v8i16 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;

  temp_src = src - 4;

  LD_UB8(temp_src, pitch, p0, p1, p2, p3, row4, row5, row6, row7);
  temp_src += (8 * pitch);
  LD_UB8(temp_src, pitch, q3, q2, q1, q0, row12, row13, row14, row15);

  /* transpose 16x8 matrix into 8x16 */
  TRANSPOSE16x8_UB_UB(p0, p1, p2, p3, row4, row5, row6, row7, q3, q2, q1, q0,
                      row12, row13, row14, row15, p3, p2, p1, p0, q0, q1, q2,
                      q3);

  thresh = (v16u8)__msa_fill_b(*thresh0);
  vec0 = (v8i16)__msa_fill_b(*thresh1);
  thresh = (v16u8)__msa_ilvr_d((v2i64)vec0, (v2i64)thresh);

  b_limit = (v16u8)__msa_fill_b(*b_limit0);
  vec0 = (v8i16)__msa_fill_b(*b_limit1);
  b_limit = (v16u8)__msa_ilvr_d((v2i64)vec0, (v2i64)b_limit);

  limit = (v16u8)__msa_fill_b(*limit0);
  vec0 = (v8i16)__msa_fill_b(*limit1);
  limit = (v16u8)__msa_ilvr_d((v2i64)vec0, (v2i64)limit);

  /* mask and hev */
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  /* flat4 */
  VP9_FLAT4(p3, p2, p0, q0, q2, q3, flat);
  /* filter4 */
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1_out, p0_out, q0_out, q1_out);

  if (__msa_test_bz_v(flat)) {
    ILVR_B2_SH(p0_out, p1_out, q1_out, q0_out, vec0, vec1);
    ILVRL_H2_SH(vec1, vec0, vec2, vec3);
    ILVL_B2_SH(p0_out, p1_out, q1_out, q0_out, vec0, vec1);
    ILVRL_H2_SH(vec1, vec0, vec4, vec5);

    src -= 2;
    ST4x8_UB(vec2, vec3, src, pitch);
    src += 8 * pitch;
    ST4x8_UB(vec4, vec5, src, pitch);
  } else {
    ILVR_B8_UH(zero, p3, zero, p2, zero, p1, zero, p0, zero, q0, zero, q1, zero,
               q2, zero, q3, p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r);
    VP9_FILTER8(p3_r, p2_r, p1_r, p0_r, q0_r, q1_r, q2_r, q3_r, p2_filt8_r,
                p1_filt8_r, p0_filt8_r, q0_filt8_r, q1_filt8_r, q2_filt8_r);

    ILVL_B4_UH(zero, p3, zero, p2, zero, p1, zero, p0, p3_l, p2_l, p1_l, p0_l);
    ILVL_B4_UH(zero, q0, zero, q1, zero, q2, zero, q3, q0_l, q1_l, q2_l, q3_l);

    /* filter8 */
    VP9_FILTER8(p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l, p2_filt8_l,
                p1_filt8_l, p0_filt8_l, q0_filt8_l, q1_filt8_l, q2_filt8_l);

    /* convert 16 bit output data into 8 bit */
    PCKEV_B4_SH(p2_filt8_l, p2_filt8_r, p1_filt8_l, p1_filt8_r, p0_filt8_l,
                p0_filt8_r, q0_filt8_l, q0_filt8_r, p2_filt8_r, p1_filt8_r,
                p0_filt8_r, q0_filt8_r);
    PCKEV_B2_SH(q1_filt8_l, q1_filt8_r, q2_filt8_l, q2_filt8_r, q1_filt8_r,
                q2_filt8_r);

    /* store pixel values */
    p2 = __msa_bmnz_v(p2, (v16u8)p2_filt8_r, flat);
    p1 = __msa_bmnz_v(p1_out, (v16u8)p1_filt8_r, flat);
    p0 = __msa_bmnz_v(p0_out, (v16u8)p0_filt8_r, flat);
    q0 = __msa_bmnz_v(q0_out, (v16u8)q0_filt8_r, flat);
    q1 = __msa_bmnz_v(q1_out, (v16u8)q1_filt8_r, flat);
    q2 = __msa_bmnz_v(q2, (v16u8)q2_filt8_r, flat);

    ILVR_B2_SH(p1, p2, q0, p0, vec0, vec1);
    ILVRL_H2_SH(vec1, vec0, vec3, vec4);
    ILVL_B2_SH(p1, p2, q0, p0, vec0, vec1);
    ILVRL_H2_SH(vec1, vec0, vec6, vec7);
    ILVRL_B2_SH(q2, q1, vec2, vec5);

    src -= 3;
    ST4x4_UB(vec3, vec3, 0, 1, 2, 3, src, pitch);
    ST2x4_UB(vec2, 0, src + 4, pitch);
    src += (4 * pitch);
    ST4x4_UB(vec4, vec4, 0, 1, 2, 3, src, pitch);
    ST2x4_UB(vec2, 4, src + 4, pitch);
    src += (4 * pitch);
    ST4x4_UB(vec6, vec6, 0, 1, 2, 3, src, pitch);
    ST2x4_UB(vec5, 0, src + 4, pitch);
    src += (4 * pitch);
    ST4x4_UB(vec7, vec7, 0, 1, 2, 3, src, pitch);
    ST2x4_UB(vec5, 4, src + 4, pitch);
  }
}
