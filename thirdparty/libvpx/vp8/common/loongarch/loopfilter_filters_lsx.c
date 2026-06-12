/*
 * Copyright (c) 2021 Loongson Technology Corporation Limited
 * Contributed by Lu Wang <wanglu@loongson.cn>
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS. All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"
#include "vp8/common/loopfilter.h"
#include "vpx_util/loongson_intrinsics.h"

#define VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev)        \
  do {                                                       \
    __m128i p1_m, p0_m, q0_m, q1_m, filt, q0_sub_p0, t1, t2; \
    const __m128i cnst4b = __lsx_vldi(4);                    \
    const __m128i cnst3b = __lsx_vldi(3);                    \
                                                             \
    p1_m = __lsx_vxori_b(p1, 0x80);                          \
    p0_m = __lsx_vxori_b(p0, 0x80);                          \
    q0_m = __lsx_vxori_b(q0, 0x80);                          \
    q1_m = __lsx_vxori_b(q1, 0x80);                          \
                                                             \
    filt = __lsx_vssub_b(p1_m, q1_m);                        \
    filt = __lsx_vand_v(filt, hev);                          \
    q0_sub_p0 = __lsx_vssub_b(q0_m, p0_m);                   \
    filt = __lsx_vsadd_b(filt, q0_sub_p0);                   \
    filt = __lsx_vsadd_b(filt, q0_sub_p0);                   \
    filt = __lsx_vsadd_b(filt, q0_sub_p0);                   \
    filt = __lsx_vand_v(filt, mask);                         \
    t1 = __lsx_vsadd_b(filt, cnst4b);                        \
    t1 = __lsx_vsra_b(t1, cnst3b);                           \
    t2 = __lsx_vsadd_b(filt, cnst3b);                        \
    t2 = __lsx_vsra_b(t2, cnst3b);                           \
    q0_m = __lsx_vssub_b(q0_m, t1);                          \
    q0 = __lsx_vxori_b(q0_m, 0x80);                          \
    p0_m = __lsx_vsadd_b(p0_m, t2);                          \
    p0 = __lsx_vxori_b(p0_m, 0x80);                          \
    filt = __lsx_vsrari_b(t1, 1);                            \
    hev = __lsx_vxori_b(hev, 0xff);                          \
    filt = __lsx_vand_v(filt, hev);                          \
    q1_m = __lsx_vssub_b(q1_m, filt);                        \
    q1 = __lsx_vxori_b(q1_m, 0x80);                          \
    p1_m = __lsx_vsadd_b(p1_m, filt);                        \
    p1 = __lsx_vxori_b(p1_m, 0x80);                          \
  } while (0)

#define VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev) \
  do {                                                  \
    __m128i p2_m, p1_m, p0_m, q2_m, q1_m, q0_m;         \
    __m128i u, filt, t1, t2, filt_sign, q0_sub_p0;      \
    __m128i filt_r, filt_l;                             \
    __m128i temp0, temp1, temp2, temp3;                 \
    const __m128i cnst4b = __lsx_vldi(4);               \
    const __m128i cnst3b = __lsx_vldi(3);               \
    const __m128i cnst9h = __lsx_vldi(1033);            \
    const __m128i cnst63h = __lsx_vldi(1087);           \
                                                        \
    p2_m = __lsx_vxori_b(p2, 0x80);                     \
    p1_m = __lsx_vxori_b(p1, 0x80);                     \
    p0_m = __lsx_vxori_b(p0, 0x80);                     \
    q0_m = __lsx_vxori_b(q0, 0x80);                     \
    q1_m = __lsx_vxori_b(q1, 0x80);                     \
    q2_m = __lsx_vxori_b(q2, 0x80);                     \
                                                        \
    filt = __lsx_vssub_b(p1_m, q1_m);                   \
    q0_sub_p0 = __lsx_vssub_b(q0_m, p0_m);              \
    filt = __lsx_vsadd_b(filt, q0_sub_p0);              \
    filt = __lsx_vsadd_b(filt, q0_sub_p0);              \
    filt = __lsx_vsadd_b(filt, q0_sub_p0);              \
    filt = __lsx_vand_v(filt, mask);                    \
                                                        \
    t2 = __lsx_vand_v(filt, hev);                       \
    hev = __lsx_vxori_b(hev, 0xff);                     \
    filt = __lsx_vand_v(hev, filt);                     \
    t1 = __lsx_vsadd_b(t2, cnst4b);                     \
    t1 = __lsx_vsra_b(t1, cnst3b);                      \
    t2 = __lsx_vsadd_b(t2, cnst3b);                     \
    t2 = __lsx_vsra_b(t2, cnst3b);                      \
    q0_m = __lsx_vssub_b(q0_m, t1);                     \
    p0_m = __lsx_vsadd_b(p0_m, t2);                     \
    filt_sign = __lsx_vslti_b(filt, 0);                 \
    filt_r = __lsx_vilvl_b(filt_sign, filt);            \
    filt_l = __lsx_vilvh_b(filt_sign, filt);            \
    temp0 = __lsx_vmul_h(filt_r, cnst9h);               \
    temp1 = __lsx_vadd_h(temp0, cnst63h);               \
    temp2 = __lsx_vmul_h(filt_l, cnst9h);               \
    temp3 = __lsx_vadd_h(temp2, cnst63h);               \
                                                        \
    u = __lsx_vssrani_b_h(temp3, temp1, 7);             \
    q2_m = __lsx_vssub_b(q2_m, u);                      \
    p2_m = __lsx_vsadd_b(p2_m, u);                      \
    q2 = __lsx_vxori_b(q2_m, 0x80);                     \
    p2 = __lsx_vxori_b(p2_m, 0x80);                     \
                                                        \
    temp1 = __lsx_vadd_h(temp1, temp0);                 \
    temp3 = __lsx_vadd_h(temp3, temp2);                 \
                                                        \
    u = __lsx_vssrani_b_h(temp3, temp1, 7);             \
    q1_m = __lsx_vssub_b(q1_m, u);                      \
    p1_m = __lsx_vsadd_b(p1_m, u);                      \
    q1 = __lsx_vxori_b(q1_m, 0x80);                     \
    p1 = __lsx_vxori_b(p1_m, 0x80);                     \
                                                        \
    temp1 = __lsx_vadd_h(temp1, temp0);                 \
    temp3 = __lsx_vadd_h(temp3, temp2);                 \
                                                        \
    u = __lsx_vssrani_b_h(temp3, temp1, 7);             \
    q0_m = __lsx_vssub_b(q0_m, u);                      \
    p0_m = __lsx_vsadd_b(p0_m, u);                      \
    q0 = __lsx_vxori_b(q0_m, 0x80);                     \
    p0 = __lsx_vxori_b(p0_m, 0x80);                     \
  } while (0)

#define LPF_MASK_HEV(p3_in, p2_in, p1_in, p0_in, q0_in, q1_in, q2_in, q3_in, \
                     limit_in, b_limit_in, thresh_in, hev_out, mask_out,     \
                     flat_out)                                               \
  do {                                                                       \
    __m128i p3_asub_p2_m, p2_asub_p1_m, p1_asub_p0_m, q1_asub_q0_m;          \
    __m128i p1_asub_q1_m, p0_asub_q0_m, q3_asub_q2_m, q2_asub_q1_m;          \
                                                                             \
    p3_asub_p2_m = __lsx_vabsd_bu(p3_in, p2_in);                             \
    p2_asub_p1_m = __lsx_vabsd_bu(p2_in, p1_in);                             \
    p1_asub_p0_m = __lsx_vabsd_bu(p1_in, p0_in);                             \
    q1_asub_q0_m = __lsx_vabsd_bu(q1_in, q0_in);                             \
    q2_asub_q1_m = __lsx_vabsd_bu(q2_in, q1_in);                             \
    q3_asub_q2_m = __lsx_vabsd_bu(q3_in, q2_in);                             \
    p0_asub_q0_m = __lsx_vabsd_bu(p0_in, q0_in);                             \
    p1_asub_q1_m = __lsx_vabsd_bu(p1_in, q1_in);                             \
    flat_out = __lsx_vmax_bu(p1_asub_p0_m, q1_asub_q0_m);                    \
    hev_out = __lsx_vslt_bu(thresh_in, flat_out);                            \
    p0_asub_q0_m = __lsx_vsadd_bu(p0_asub_q0_m, p0_asub_q0_m);               \
    p1_asub_q1_m = __lsx_vsrli_b(p1_asub_q1_m, 1);                           \
    p0_asub_q0_m = __lsx_vsadd_bu(p0_asub_q0_m, p1_asub_q1_m);               \
    mask_out = __lsx_vslt_bu(b_limit_in, p0_asub_q0_m);                      \
    mask_out = __lsx_vmax_bu(flat_out, mask_out);                            \
    p3_asub_p2_m = __lsx_vmax_bu(p3_asub_p2_m, p2_asub_p1_m);                \
    mask_out = __lsx_vmax_bu(p3_asub_p2_m, mask_out);                        \
    q2_asub_q1_m = __lsx_vmax_bu(q2_asub_q1_m, q3_asub_q2_m);                \
    mask_out = __lsx_vmax_bu(q2_asub_q1_m, mask_out);                        \
    mask_out = __lsx_vslt_bu(limit_in, mask_out);                            \
    mask_out = __lsx_vxori_b(mask_out, 0xff);                                \
  } while (0)

#define VP8_ST6x1_B(in0, in0_idx, in1, in1_idx, pdst, stride) \
  do {                                                        \
    __lsx_vstelm_w(in0, pdst, 0, in0_idx);                    \
    __lsx_vstelm_h(in1, pdst + stride, 0, in1_idx);           \
  } while (0)

static void loop_filter_horizontal_4_dual_lsx(uint8_t *src, int32_t pitch,
                                              const uint8_t *b_limit0_ptr,
                                              const uint8_t *limit0_ptr,
                                              const uint8_t *thresh0_ptr,
                                              const uint8_t *b_limit1_ptr,
                                              const uint8_t *limit1_ptr,
                                              const uint8_t *thresh1_ptr) {
  int32_t pitch_x2 = pitch << 1;
  int32_t pitch_x3 = pitch_x2 + pitch;
  int32_t pitch_x4 = pitch << 2;

  __m128i mask, hev, flat;
  __m128i thresh0, b_limit0, limit0, thresh1, b_limit1, limit1;
  __m128i p3, p2, p1, p0, q3, q2, q1, q0;

  DUP4_ARG2(__lsx_vldx, src, -pitch_x4, src, -pitch_x3, src, -pitch_x2, src,
            -pitch, p3, p2, p1, p0);
  q0 = __lsx_vld(src, 0);
  DUP2_ARG2(__lsx_vldx, src, pitch, src, pitch_x2, q1, q2);
  q3 = __lsx_vldx(src, pitch_x3);

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
  VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev);

  __lsx_vstx(p1, src, -pitch_x2);
  __lsx_vstx(p0, src, -pitch);
  __lsx_vst(q0, src, 0);
  __lsx_vstx(q1, src, pitch);
}

static void loop_filter_vertical_4_dual_lsx(uint8_t *src, int32_t pitch,
                                            const uint8_t *b_limit0_ptr,
                                            const uint8_t *limit0_ptr,
                                            const uint8_t *thresh0_ptr,
                                            const uint8_t *b_limit1_ptr,
                                            const uint8_t *limit1_ptr,
                                            const uint8_t *thresh1_ptr) {
  uint8_t *src_tmp0 = src - 4;
  int32_t pitch_x2 = pitch << 1;
  int32_t pitch_x3 = pitch_x2 + pitch;
  int32_t pitch_x4 = pitch << 2;
  __m128i mask, hev, flat;
  __m128i thresh0, b_limit0, limit0, thresh1, b_limit1, limit1;
  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i row0, row1, row2, row3, row4, row5, row6, row7;
  __m128i row8, row9, row10, row11, row12, row13, row14, row15;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

  row0 = __lsx_vld(src_tmp0, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp0, pitch, src_tmp0, pitch_x2, row1, row2);
  row3 = __lsx_vldx(src_tmp0, pitch_x3);
  src_tmp0 += pitch_x4;
  row4 = __lsx_vld(src_tmp0, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp0, pitch, src_tmp0, pitch_x2, row5, row6);
  row7 = __lsx_vldx(src_tmp0, pitch_x3);
  src_tmp0 += pitch_x4;

  row8 = __lsx_vld(src_tmp0, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp0, pitch, src_tmp0, pitch_x2, row9, row10);
  row11 = __lsx_vldx(src_tmp0, pitch_x3);
  src_tmp0 += pitch_x4;
  row12 = __lsx_vld(src_tmp0, 0);
  DUP2_ARG2(__lsx_vldx, src_tmp0, pitch, src_tmp0, pitch_x2, row13, row14);
  row15 = __lsx_vldx(src_tmp0, pitch_x3);

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
  VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev);

  DUP2_ARG2(__lsx_vilvl_b, p0, p1, q1, q0, tmp0, tmp1);
  tmp2 = __lsx_vilvl_h(tmp1, tmp0);
  tmp3 = __lsx_vilvh_h(tmp1, tmp0);
  DUP2_ARG2(__lsx_vilvh_b, p0, p1, q1, q0, tmp0, tmp1);
  tmp4 = __lsx_vilvl_h(tmp1, tmp0);
  tmp5 = __lsx_vilvh_h(tmp1, tmp0);

  src -= 2;
  __lsx_vstelm_w(tmp2, src, 0, 0);
  src += pitch;
  __lsx_vstelm_w(tmp2, src, 0, 1);
  src += pitch;
  __lsx_vstelm_w(tmp2, src, 0, 2);
  src += pitch;
  __lsx_vstelm_w(tmp2, src, 0, 3);
  src += pitch;

  __lsx_vstelm_w(tmp3, src, 0, 0);
  src += pitch;
  __lsx_vstelm_w(tmp3, src, 0, 1);
  src += pitch;
  __lsx_vstelm_w(tmp3, src, 0, 2);
  src += pitch;
  __lsx_vstelm_w(tmp3, src, 0, 3);
  src += pitch;

  __lsx_vstelm_w(tmp4, src, 0, 0);
  src += pitch;
  __lsx_vstelm_w(tmp4, src, 0, 1);
  src += pitch;
  __lsx_vstelm_w(tmp4, src, 0, 2);
  src += pitch;
  __lsx_vstelm_w(tmp4, src, 0, 3);
  src += pitch;

  __lsx_vstelm_w(tmp5, src, 0, 0);
  src += pitch;
  __lsx_vstelm_w(tmp5, src, 0, 1);
  src += pitch;
  __lsx_vstelm_w(tmp5, src, 0, 2);
  src += pitch;
  __lsx_vstelm_w(tmp5, src, 0, 3);
}

static void loop_filter_horizontal_edge_uv_lsx(uint8_t *src_u, uint8_t *src_v,
                                               int32_t pitch,
                                               const uint8_t b_limit_in,
                                               const uint8_t limit_in,
                                               const uint8_t thresh_in) {
  int32_t pitch_x2 = pitch << 1;
  int32_t pitch_x3 = pitch_x2 + pitch;
  int32_t pitch_x4 = pitch << 2;

  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i mask, hev, flat, thresh, limit, b_limit;
  __m128i p3_u, p2_u, p1_u, p0_u, q3_u, q2_u, q1_u, q0_u;
  __m128i p3_v, p2_v, p1_v, p0_v, q3_v, q2_v, q1_v, q0_v;

  thresh = __lsx_vreplgr2vr_b(thresh_in);
  limit = __lsx_vreplgr2vr_b(limit_in);
  b_limit = __lsx_vreplgr2vr_b(b_limit_in);

  DUP4_ARG2(__lsx_vldx, src_u, -pitch_x4, src_u, -pitch_x3, src_u, -pitch_x2,
            src_u, -pitch, p3_u, p2_u, p1_u, p0_u);
  q0_u = __lsx_vld(src_u, 0);
  DUP2_ARG2(__lsx_vldx, src_u, pitch, src_u, pitch_x2, q1_u, q2_u);
  q3_u = __lsx_vldx(src_u, pitch_x3);

  DUP4_ARG2(__lsx_vldx, src_v, -pitch_x4, src_v, -pitch_x3, src_v, -pitch_x2,
            src_v, -pitch, p3_v, p2_v, p1_v, p0_v);
  q0_v = __lsx_vld(src_v, 0);
  DUP2_ARG2(__lsx_vldx, src_v, pitch, src_v, pitch_x2, q1_v, q2_v);
  q3_v = __lsx_vldx(src_v, pitch_x3);

  /* right 8 element of p3 are u pixel and
     left 8 element of p3 are v pixel */
  DUP4_ARG2(__lsx_vilvl_d, p3_v, p3_u, p2_v, p2_u, p1_v, p1_u, p0_v, p0_u, p3,
            p2, p1, p0);
  DUP4_ARG2(__lsx_vilvl_d, q0_v, q0_u, q1_v, q1_u, q2_v, q2_u, q3_v, q3_u, q0,
            q1, q2, q3);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev);

  __lsx_vstelm_d(q1, src_u + pitch, 0, 0);
  __lsx_vstelm_d(q0, src_u, 0, 0);
  __lsx_vstelm_d(p0, src_u - pitch, 0, 0);
  __lsx_vstelm_d(p1, src_u - pitch_x2, 0, 0);

  __lsx_vstelm_d(q1, src_v + pitch, 0, 1);
  __lsx_vstelm_d(q0, src_v, 0, 1);
  __lsx_vstelm_d(p0, src_v - pitch, 0, 1);
  __lsx_vstelm_d(p1, src_v - pitch_x2, 0, 1);
}

static void loop_filter_vertical_edge_uv_lsx(uint8_t *src_u, uint8_t *src_v,
                                             int32_t pitch,
                                             const uint8_t b_limit_in,
                                             const uint8_t limit_in,
                                             const uint8_t thresh_in) {
  uint8_t *src_u_tmp, *src_v_tmp;
  int32_t pitch_x2 = pitch << 1;
  int32_t pitch_x3 = pitch_x2 + pitch;
  int32_t pitch_x4 = pitch << 2;

  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i mask, hev, flat, thresh, limit, b_limit;
  __m128i row0, row1, row2, row3, row4, row5, row6, row7, row8;
  __m128i row9, row10, row11, row12, row13, row14, row15;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

  thresh = __lsx_vreplgr2vr_b(thresh_in);
  limit = __lsx_vreplgr2vr_b(limit_in);
  b_limit = __lsx_vreplgr2vr_b(b_limit_in);

  src_u_tmp = src_u - 4;
  row0 = __lsx_vld(src_u_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_u_tmp, pitch, src_u_tmp, pitch_x2, row1, row2);
  row3 = __lsx_vldx(src_u_tmp, pitch_x3);
  src_u_tmp += pitch_x4;
  row4 = __lsx_vld(src_u_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_u_tmp, pitch, src_u_tmp, pitch_x2, row5, row6);
  row7 = __lsx_vldx(src_u_tmp, pitch_x3);

  src_v_tmp = src_v - 4;
  row8 = __lsx_vld(src_v_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_v_tmp, pitch, src_v_tmp, pitch_x2, row9, row10);
  row11 = __lsx_vldx(src_v_tmp, pitch_x3);
  src_v_tmp += pitch_x4;
  row12 = __lsx_vld(src_v_tmp, 0);
  DUP2_ARG2(__lsx_vldx, src_v_tmp, pitch, src_v_tmp, pitch_x2, row13, row14);
  row15 = __lsx_vldx(src_v_tmp, pitch_x3);

  LSX_TRANSPOSE16x8_B(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p3, p2,
                      p1, p0, q0, q1, q2, q3);
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev);

  DUP2_ARG2(__lsx_vilvl_b, p0, p1, q1, q0, tmp0, tmp1);
  tmp2 = __lsx_vilvl_h(tmp1, tmp0);
  tmp3 = __lsx_vilvh_h(tmp1, tmp0);

  tmp0 = __lsx_vilvh_b(p0, p1);
  tmp1 = __lsx_vilvh_b(q1, q0);
  tmp4 = __lsx_vilvl_h(tmp1, tmp0);
  tmp5 = __lsx_vilvh_h(tmp1, tmp0);

  src_u_tmp += 2;
  __lsx_vstelm_w(tmp2, src_u_tmp - pitch_x4, 0, 0);
  __lsx_vstelm_w(tmp2, src_u_tmp - pitch_x3, 0, 1);
  __lsx_vstelm_w(tmp2, src_u_tmp - pitch_x2, 0, 2);
  __lsx_vstelm_w(tmp2, src_u_tmp - pitch, 0, 3);

  __lsx_vstelm_w(tmp3, src_u_tmp, 0, 0);
  __lsx_vstelm_w(tmp3, src_u_tmp + pitch, 0, 1);
  __lsx_vstelm_w(tmp3, src_u_tmp + pitch_x2, 0, 2);
  __lsx_vstelm_w(tmp3, src_u_tmp + pitch_x3, 0, 3);

  src_v_tmp += 2;
  __lsx_vstelm_w(tmp4, src_v_tmp - pitch_x4, 0, 0);
  __lsx_vstelm_w(tmp4, src_v_tmp - pitch_x3, 0, 1);
  __lsx_vstelm_w(tmp4, src_v_tmp - pitch_x2, 0, 2);
  __lsx_vstelm_w(tmp4, src_v_tmp - pitch, 0, 3);

  __lsx_vstelm_w(tmp5, src_v_tmp, 0, 0);
  __lsx_vstelm_w(tmp5, src_v_tmp + pitch, 0, 1);
  __lsx_vstelm_w(tmp5, src_v_tmp + pitch_x2, 0, 2);
  __lsx_vstelm_w(tmp5, src_v_tmp + pitch_x3, 0, 3);
}

static inline void mbloop_filter_horizontal_edge_y_lsx(
    uint8_t *src, int32_t pitch, const uint8_t b_limit_in,
    const uint8_t limit_in, const uint8_t thresh_in) {
  uint8_t *temp_src;
  int32_t pitch_x2 = pitch << 1;
  int32_t pitch_x3 = pitch_x2 + pitch;
  int32_t pitch_x4 = pitch << 2;

  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i mask, hev, flat, thresh, limit, b_limit;

  DUP2_ARG2(__lsx_vldrepl_b, &b_limit_in, 0, &limit_in, 0, b_limit, limit);
  thresh = __lsx_vldrepl_b(&thresh_in, 0);

  temp_src = src - pitch_x4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, p3, p2, p1, p0);
  temp_src += pitch_x4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, q0, q1, q2, q3);
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev);

  temp_src = src - pitch_x3;
  __lsx_vstx(p2, temp_src, 0);
  __lsx_vstx(p1, temp_src, pitch);
  __lsx_vstx(p0, temp_src, pitch_x2);
  __lsx_vstx(q0, temp_src, pitch_x3);
  temp_src += pitch_x4;
  __lsx_vstx(q1, temp_src, 0);
  __lsx_vstx(q2, temp_src, pitch);
}

static inline void mbloop_filter_horizontal_edge_uv_lsx(
    uint8_t *src_u, uint8_t *src_v, int32_t pitch, const uint8_t b_limit_in,
    const uint8_t limit_in, const uint8_t thresh_in) {
  uint8_t *temp_src;
  int32_t pitch_x2 = pitch << 1;
  int32_t pitch_x3 = pitch_x2 + pitch;
  int32_t pitch_x4 = pitch << 2;
  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i mask, hev, flat, thresh, limit, b_limit;
  __m128i p3_u, p2_u, p1_u, p0_u, q3_u, q2_u, q1_u, q0_u;
  __m128i p3_v, p2_v, p1_v, p0_v, q3_v, q2_v, q1_v, q0_v;

  DUP2_ARG2(__lsx_vldrepl_b, &b_limit_in, 0, &limit_in, 0, b_limit, limit);
  thresh = __lsx_vldrepl_b(&thresh_in, 0);

  temp_src = src_u - pitch_x4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, p3_u, p2_u, p1_u, p0_u);
  temp_src += pitch_x4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, q0_u, q1_u, q2_u, q3_u);
  temp_src = src_v - pitch_x4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, p3_v, p2_v, p1_v, p0_v);
  temp_src += pitch_x4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, q0_v, q1_v, q2_v, q3_v);

  DUP4_ARG2(__lsx_vilvl_d, p3_v, p3_u, p2_v, p2_u, p1_v, p1_u, p0_v, p0_u, p3,
            p2, p1, p0);
  DUP4_ARG2(__lsx_vilvl_d, q0_v, q0_u, q1_v, q1_u, q2_v, q2_u, q3_v, q3_u, q0,
            q1, q2, q3);
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev);

  src_u -= pitch_x3;
  __lsx_vstelm_d(p2, src_u, 0, 0);
  __lsx_vstelm_d(p1, src_u + pitch, 0, 0);
  __lsx_vstelm_d(p0, src_u + pitch_x2, 0, 0);
  __lsx_vstelm_d(q0, src_u + pitch_x3, 0, 0);
  src_u += pitch_x4;
  __lsx_vstelm_d(q1, src_u, 0, 0);
  src_u += pitch;
  __lsx_vstelm_d(q2, src_u, 0, 0);

  src_v -= pitch_x3;
  __lsx_vstelm_d(p2, src_v, 0, 1);
  __lsx_vstelm_d(p1, src_v + pitch, 0, 1);
  __lsx_vstelm_d(p0, src_v + pitch_x2, 0, 1);
  __lsx_vstelm_d(q0, src_v + pitch_x3, 0, 1);
  src_v += pitch_x4;
  __lsx_vstelm_d(q1, src_v, 0, 1);
  src_v += pitch;
  __lsx_vstelm_d(q2, src_v, 0, 1);
}

static inline void mbloop_filter_vertical_edge_y_lsx(uint8_t *src,
                                                     int32_t pitch,
                                                     const uint8_t b_limit_in,
                                                     const uint8_t limit_in,
                                                     const uint8_t thresh_in) {
  uint8_t *temp_src;
  int32_t pitch_x2 = pitch << 1;
  int32_t pitch_x3 = pitch_x2 + pitch;
  int32_t pitch_x4 = pitch << 2;

  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i mask, hev, flat, thresh, limit, b_limit;
  __m128i row0, row1, row2, row3, row4, row5, row6, row7, row8;
  __m128i row9, row10, row11, row12, row13, row14, row15;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

  DUP2_ARG2(__lsx_vldrepl_b, &b_limit_in, 0, &limit_in, 0, b_limit, limit);
  thresh = __lsx_vldrepl_b(&thresh_in, 0);
  temp_src = src - 4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, row0, row1, row2, row3);
  temp_src += pitch_x4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, row4, row5, row6, row7);
  temp_src += pitch_x4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, row8, row9, row10, row11);
  temp_src += pitch_x4;
  DUP4_ARG2(__lsx_vldx, temp_src, 0, temp_src, pitch, temp_src, pitch_x2,
            temp_src, pitch_x3, row12, row13, row14, row15);
  temp_src -= pitch_x4;
  LSX_TRANSPOSE16x8_B(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p3, p2,
                      p1, p0, q0, q1, q2, q3);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev);
  DUP2_ARG2(__lsx_vilvl_b, p1, p2, q0, p0, tmp0, tmp1);
  tmp3 = __lsx_vilvl_h(tmp1, tmp0);
  tmp4 = __lsx_vilvh_h(tmp1, tmp0);
  DUP2_ARG2(__lsx_vilvh_b, p1, p2, q0, p0, tmp0, tmp1);
  tmp6 = __lsx_vilvl_h(tmp1, tmp0);
  tmp7 = __lsx_vilvh_h(tmp1, tmp0);
  tmp2 = __lsx_vilvl_b(q2, q1);
  tmp5 = __lsx_vilvh_b(q2, q1);

  temp_src = src - 3;
  VP8_ST6x1_B(tmp3, 0, tmp2, 0, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp3, 1, tmp2, 1, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp3, 2, tmp2, 2, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp3, 3, tmp2, 3, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp4, 0, tmp2, 4, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp4, 1, tmp2, 5, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp4, 2, tmp2, 6, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp4, 3, tmp2, 7, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp6, 0, tmp5, 0, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp6, 1, tmp5, 1, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp6, 2, tmp5, 2, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp6, 3, tmp5, 3, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp7, 0, tmp5, 4, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp7, 1, tmp5, 5, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp7, 2, tmp5, 6, temp_src, 4);
  temp_src += pitch;
  VP8_ST6x1_B(tmp7, 3, tmp5, 7, temp_src, 4);
}

static inline void mbloop_filter_vertical_edge_uv_lsx(
    uint8_t *src_u, uint8_t *src_v, int32_t pitch, const uint8_t b_limit_in,
    const uint8_t limit_in, const uint8_t thresh_in) {
  int32_t pitch_x2 = pitch << 1;
  int32_t pitch_x3 = pitch_x2 + pitch;
  int32_t pitch_x4 = pitch << 2;
  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i mask, hev, flat, thresh, limit, b_limit;
  __m128i row0, row1, row2, row3, row4, row5, row6, row7, row8;
  __m128i row9, row10, row11, row12, row13, row14, row15;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

  DUP2_ARG2(__lsx_vldrepl_b, &b_limit_in, 0, &limit_in, 0, b_limit, limit);
  thresh = __lsx_vldrepl_b(&thresh_in, 0);

  src_u -= 4;
  DUP4_ARG2(__lsx_vldx, src_u, 0, src_u, pitch, src_u, pitch_x2, src_u,
            pitch_x3, row0, row1, row2, row3);
  src_u += pitch_x4;
  DUP4_ARG2(__lsx_vldx, src_u, 0, src_u, pitch, src_u, pitch_x2, src_u,
            pitch_x3, row4, row5, row6, row7);
  src_v -= 4;
  DUP4_ARG2(__lsx_vldx, src_v, 0, src_v, pitch, src_v, pitch_x2, src_v,
            pitch_x3, row8, row9, row10, row11);
  src_v += pitch_x4;
  DUP4_ARG2(__lsx_vldx, src_v, 0, src_v, pitch, src_v, pitch_x2, src_v,
            pitch_x3, row12, row13, row14, row15);
  LSX_TRANSPOSE16x8_B(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p3, p2,
                      p1, p0, q0, q1, q2, q3);

  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP8_MBFILTER(p2, p1, p0, q0, q1, q2, mask, hev);

  DUP2_ARG2(__lsx_vilvl_b, p1, p2, q0, p0, tmp0, tmp1);
  tmp3 = __lsx_vilvl_h(tmp1, tmp0);
  tmp4 = __lsx_vilvh_h(tmp1, tmp0);
  DUP2_ARG2(__lsx_vilvh_b, p1, p2, q0, p0, tmp0, tmp1);
  tmp6 = __lsx_vilvl_h(tmp1, tmp0);
  tmp7 = __lsx_vilvh_h(tmp1, tmp0);
  tmp2 = __lsx_vilvl_b(q2, q1);
  tmp5 = __lsx_vilvh_b(q2, q1);

  src_u += 1 - pitch_x4;
  VP8_ST6x1_B(tmp3, 0, tmp2, 0, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_B(tmp3, 1, tmp2, 1, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_B(tmp3, 2, tmp2, 2, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_B(tmp3, 3, tmp2, 3, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_B(tmp4, 0, tmp2, 4, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_B(tmp4, 1, tmp2, 5, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_B(tmp4, 2, tmp2, 6, src_u, 4);
  src_u += pitch;
  VP8_ST6x1_B(tmp4, 3, tmp2, 7, src_u, 4);

  src_v += 1 - pitch_x4;
  VP8_ST6x1_B(tmp6, 0, tmp5, 0, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_B(tmp6, 1, tmp5, 1, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_B(tmp6, 2, tmp5, 2, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_B(tmp6, 3, tmp5, 3, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_B(tmp7, 0, tmp5, 4, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_B(tmp7, 1, tmp5, 5, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_B(tmp7, 2, tmp5, 6, src_v, 4);
  src_v += pitch;
  VP8_ST6x1_B(tmp7, 3, tmp5, 7, src_v, 4);
}

void vp8_loop_filter_mbh_lsx(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v,
                             int32_t pitch_y, int32_t pitch_u_v,
                             loop_filter_info *lpf_info_ptr) {
  mbloop_filter_horizontal_edge_y_lsx(src_y, pitch_y, *lpf_info_ptr->mblim,
                                      *lpf_info_ptr->lim,
                                      *lpf_info_ptr->hev_thr);
  if (src_u) {
    mbloop_filter_horizontal_edge_uv_lsx(
        src_u, src_v, pitch_u_v, *lpf_info_ptr->mblim, *lpf_info_ptr->lim,
        *lpf_info_ptr->hev_thr);
  }
}

void vp8_loop_filter_mbv_lsx(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v,
                             int32_t pitch_y, int32_t pitch_u_v,
                             loop_filter_info *lpf_info_ptr) {
  mbloop_filter_vertical_edge_y_lsx(src_y, pitch_y, *lpf_info_ptr->mblim,
                                    *lpf_info_ptr->lim, *lpf_info_ptr->hev_thr);
  if (src_u) {
    mbloop_filter_vertical_edge_uv_lsx(src_u, src_v, pitch_u_v,
                                       *lpf_info_ptr->mblim, *lpf_info_ptr->lim,
                                       *lpf_info_ptr->hev_thr);
  }
}

void vp8_loop_filter_bh_lsx(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v,
                            int32_t pitch_y, int32_t pitch_u_v,
                            loop_filter_info *lpf_info_ptr) {
  loop_filter_horizontal_4_dual_lsx(src_y + 4 * pitch_y, pitch_y,
                                    lpf_info_ptr->blim, lpf_info_ptr->lim,
                                    lpf_info_ptr->hev_thr, lpf_info_ptr->blim,
                                    lpf_info_ptr->lim, lpf_info_ptr->hev_thr);
  loop_filter_horizontal_4_dual_lsx(src_y + 8 * pitch_y, pitch_y,
                                    lpf_info_ptr->blim, lpf_info_ptr->lim,
                                    lpf_info_ptr->hev_thr, lpf_info_ptr->blim,
                                    lpf_info_ptr->lim, lpf_info_ptr->hev_thr);
  loop_filter_horizontal_4_dual_lsx(src_y + 12 * pitch_y, pitch_y,
                                    lpf_info_ptr->blim, lpf_info_ptr->lim,
                                    lpf_info_ptr->hev_thr, lpf_info_ptr->blim,
                                    lpf_info_ptr->lim, lpf_info_ptr->hev_thr);
  if (src_u) {
    loop_filter_horizontal_edge_uv_lsx(
        src_u + (4 * pitch_u_v), src_v + (4 * pitch_u_v), pitch_u_v,
        *lpf_info_ptr->blim, *lpf_info_ptr->lim, *lpf_info_ptr->hev_thr);
  }
}

void vp8_loop_filter_bv_lsx(uint8_t *src_y, uint8_t *src_u, uint8_t *src_v,
                            int32_t pitch_y, int32_t pitch_u_v,
                            loop_filter_info *lpf_info_ptr) {
  loop_filter_vertical_4_dual_lsx(src_y + 4, pitch_y, lpf_info_ptr->blim,
                                  lpf_info_ptr->lim, lpf_info_ptr->hev_thr,
                                  lpf_info_ptr->blim, lpf_info_ptr->lim,
                                  lpf_info_ptr->hev_thr);
  loop_filter_vertical_4_dual_lsx(src_y + 8, pitch_y, lpf_info_ptr->blim,
                                  lpf_info_ptr->lim, lpf_info_ptr->hev_thr,
                                  lpf_info_ptr->blim, lpf_info_ptr->lim,
                                  lpf_info_ptr->hev_thr);
  loop_filter_vertical_4_dual_lsx(src_y + 12, pitch_y, lpf_info_ptr->blim,
                                  lpf_info_ptr->lim, lpf_info_ptr->hev_thr,
                                  lpf_info_ptr->blim, lpf_info_ptr->lim,
                                  lpf_info_ptr->hev_thr);
  if (src_u) {
    loop_filter_vertical_edge_uv_lsx(src_u + 4, src_v + 4, pitch_u_v,
                                     *lpf_info_ptr->blim, *lpf_info_ptr->lim,
                                     *lpf_info_ptr->hev_thr);
  }
}
