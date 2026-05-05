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
#include "vpx_dsp/loongarch/loopfilter_lsx.h"

void vpx_lpf_horizontal_4_lsx(uint8_t *src, int32_t pitch,
                              const uint8_t *b_limit_ptr,
                              const uint8_t *limit_ptr,
                              const uint8_t *thresh_ptr) {
  __m128i mask, hev, flat, thresh, b_limit, limit;
  __m128i p3, p2, p1, p0, q3, q2, q1, q0, p1_out, p0_out, q0_out, q1_out;
  int32_t pitch2 = pitch << 1;
  int32_t pitch3 = pitch2 + pitch;
  int32_t pitch4 = pitch2 << 1;

  DUP4_ARG2(__lsx_vldx, src, -pitch4, src, -pitch3, src, -pitch2, src, -pitch,
            p3, p2, p1, p0);
  q0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, pitch, src, pitch2, q1, q2);
  q3 = __lsx_vldx(src, pitch3);

  thresh = __lsx_vldrepl_b(thresh_ptr, 0);
  b_limit = __lsx_vldrepl_b(b_limit_ptr, 0);
  limit = __lsx_vldrepl_b(limit_ptr, 0);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1_out, p0_out, q0_out, q1_out);

  __lsx_vstelm_d(p1_out, src - pitch2, 0, 0);
  __lsx_vstelm_d(p0_out, src - pitch, 0, 0);
  __lsx_vstelm_d(q0_out, src, 0, 0);
  __lsx_vstelm_d(q1_out, src + pitch, 0, 0);
}

void vpx_lpf_horizontal_4_dual_lsx(uint8_t *src, int32_t pitch,
                                   const uint8_t *b_limit0_ptr,
                                   const uint8_t *limit0_ptr,
                                   const uint8_t *thresh0_ptr,
                                   const uint8_t *b_limit1_ptr,
                                   const uint8_t *limit1_ptr,
                                   const uint8_t *thresh1_ptr) {
  __m128i mask, hev, flat, thresh0, b_limit0, limit0, thresh1, b_limit1, limit1;
  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  int32_t pitch2 = pitch << 1;
  int32_t pitch3 = pitch2 + pitch;
  int32_t pitch4 = pitch2 << 1;

  DUP4_ARG2(__lsx_vldx, src, -pitch4, src, -pitch3, src, -pitch2, src, -pitch,
            p3, p2, p1, p0);
  q0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, pitch, src, pitch2, q1, q2);
  q3 = __lsx_vldx(src, pitch3);

  thresh0 = __lsx_vldrepl_b(thresh0_ptr, 0);
  thresh1 = __lsx_vldrepl_b(thresh1_ptr, 0);
  thresh0 = __lsx_vilvl_d(thresh1, thresh0);

  b_limit0 = __lsx_vldrepl_b(b_limit0_ptr, 0);
  b_limit1 = __lsx_vldrepl_b(b_limit1_ptr, 0);
  b_limit0 = __lsx_vilvl_d(b_limit1, b_limit0);

  limit0 = __lsx_vldrepl_b(limit0_ptr, 0);
  limit1 = __lsx_vldrepl_b(limit1_ptr, 0);
  limit0 = __lsx_vilvl_d(limit1, limit0);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit0, b_limit0, thresh0, hev,
               mask, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1, p0, q0, q1);

  __lsx_vstx(p1, src, -pitch2);
  __lsx_vstx(p0, src, -pitch);
  __lsx_vst(q0, src, 0);
  __lsx_vstx(q1, src, pitch);
}

void vpx_lpf_vertical_4_lsx(uint8_t *src, int32_t pitch,
                            const uint8_t *b_limit_ptr,
                            const uint8_t *limit_ptr,
                            const uint8_t *thresh_ptr) {
  __m128i mask, hev, flat, limit, thresh, b_limit;
  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i vec0, vec1, vec2, vec3;
  int32_t pitch2 = pitch << 1;
  int32_t pitch3 = pitch2 + pitch;
  int32_t pitch4 = pitch2 << 1;
  uint8_t *src_tmp = src - 4;

  p3 = __lsx_vld(src_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp, pitch, src_tmp, pitch2, p2, p1);
  p0 = __lsx_vldx(src_tmp, pitch3);
  src_tmp += pitch4;
  q0 = __lsx_vld(src_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp, pitch, src_tmp, pitch2, q1, q2);
  q3 = __lsx_vldx(src_tmp, pitch3);

  thresh = __lsx_vldrepl_b(thresh_ptr, 0);
  b_limit = __lsx_vldrepl_b(b_limit_ptr, 0);
  limit = __lsx_vldrepl_b(limit_ptr, 0);

  LSX_TRANSPOSE8x8_B(p3, p2, p1, p0, q0, q1, q2, q3, p3, p2, p1, p0, q0, q1, q2,
                     q3);
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1, p0, q0, q1);
  DUP2_ARG2(__lsx_vilvl_b, p0, p1, q1, q0, vec0, vec1);
  vec2 = __lsx_vilvl_h(vec1, vec0);
  vec3 = __lsx_vilvh_h(vec1, vec0);

  src -= 2;
  __lsx_vstelm_w(vec2, src, 0, 0);
  src += pitch;
  __lsx_vstelm_w(vec2, src, 0, 1);
  src += pitch;
  __lsx_vstelm_w(vec2, src, 0, 2);
  src += pitch;
  __lsx_vstelm_w(vec2, src, 0, 3);
  src += pitch;

  __lsx_vstelm_w(vec3, src, 0, 0);
  __lsx_vstelm_w(vec3, src + pitch, 0, 1);
  __lsx_vstelm_w(vec3, src + pitch2, 0, 2);
  __lsx_vstelm_w(vec3, src + pitch3, 0, 3);
}

void vpx_lpf_vertical_4_dual_lsx(uint8_t *src, int32_t pitch,
                                 const uint8_t *b_limit0_ptr,
                                 const uint8_t *limit0_ptr,
                                 const uint8_t *thresh0_ptr,
                                 const uint8_t *b_limit1_ptr,
                                 const uint8_t *limit1_ptr,
                                 const uint8_t *thresh1_ptr) {
  __m128i mask, hev, flat;
  __m128i thresh0, b_limit0, limit0, thresh1, b_limit1, limit1;
  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i row0, row1, row2, row3, row4, row5, row6, row7;
  __m128i row8, row9, row10, row11, row12, row13, row14, row15;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
  int32_t pitch2 = pitch << 1;
  int32_t pitch3 = pitch2 + pitch;
  int32_t pitch4 = pitch2 << 1;
  uint8_t *src_tmp = src - 4;

  row0 = __lsx_vld(src_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp, pitch, src_tmp, pitch2, row1, row2);
  row3 = __lsx_vldx(src_tmp, pitch3);
  src_tmp += pitch4;
  row4 = __lsx_vld(src_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp, pitch, src_tmp, pitch2, row5, row6);
  row7 = __lsx_vldx(src_tmp, pitch3);
  src_tmp += pitch4;
  row8 = __lsx_vld(src_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp, pitch, src_tmp, pitch2, row9, row10);
  row11 = __lsx_vldx(src_tmp, pitch3);
  src_tmp += pitch4;
  row12 = __lsx_vld(src_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp, pitch, src_tmp, pitch2, row13, row14);
  row15 = __lsx_vldx(src_tmp, pitch3);

  LSX_TRANSPOSE16x8_B(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p3, p2,
                      p1, p0, q0, q1, q2, q3);

  thresh0 = __lsx_vldrepl_b(thresh0_ptr, 0);
  thresh1 = __lsx_vldrepl_b(thresh1_ptr, 0);
  thresh0 = __lsx_vilvl_d(thresh1, thresh0);

  b_limit0 = __lsx_vldrepl_b(b_limit0_ptr, 0);
  b_limit1 = __lsx_vldrepl_b(b_limit1_ptr, 0);
  b_limit0 = __lsx_vilvl_d(b_limit1, b_limit0);

  limit0 = __lsx_vldrepl_b(limit0_ptr, 0);
  limit1 = __lsx_vldrepl_b(limit1_ptr, 0);
  limit0 = __lsx_vilvl_d(limit1, limit0);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit0, b_limit0, thresh0, hev,
               mask, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1, p0, q0, q1);
  DUP2_ARG2(__lsx_vilvl_b, p0, p1, q1, q0, tmp0, tmp1);
  tmp2 = __lsx_vilvl_h(tmp1, tmp0);
  tmp3 = __lsx_vilvh_h(tmp1, tmp0);
  DUP2_ARG2(__lsx_vilvh_b, p0, p1, q1, q0, tmp0, tmp1);
  tmp4 = __lsx_vilvl_h(tmp1, tmp0);
  tmp5 = __lsx_vilvh_h(tmp1, tmp0);

  src -= 2;
  __lsx_vstelm_w(tmp2, src, 0, 0);
  __lsx_vstelm_w(tmp2, src + pitch, 0, 1);
  __lsx_vstelm_w(tmp2, src + pitch2, 0, 2);
  __lsx_vstelm_w(tmp2, src + pitch3, 0, 3);
  src += pitch4;
  __lsx_vstelm_w(tmp3, src, 0, 0);
  __lsx_vstelm_w(tmp3, src + pitch, 0, 1);
  __lsx_vstelm_w(tmp3, src + pitch2, 0, 2);
  __lsx_vstelm_w(tmp3, src + pitch3, 0, 3);
  src += pitch4;
  __lsx_vstelm_w(tmp4, src, 0, 0);
  __lsx_vstelm_w(tmp4, src + pitch, 0, 1);
  __lsx_vstelm_w(tmp4, src + pitch2, 0, 2);
  __lsx_vstelm_w(tmp4, src + pitch3, 0, 3);
  src += pitch4;
  __lsx_vstelm_w(tmp5, src, 0, 0);
  __lsx_vstelm_w(tmp5, src + pitch, 0, 1);
  __lsx_vstelm_w(tmp5, src + pitch2, 0, 2);
  __lsx_vstelm_w(tmp5, src + pitch3, 0, 3);
}
