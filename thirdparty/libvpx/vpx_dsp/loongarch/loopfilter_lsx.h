/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_LOONGARCH_LOOPFILTER_LSX_H_
#define VPX_VPX_DSP_LOONGARCH_LOOPFILTER_LSX_H_

#include "vpx_util/loongson_intrinsics.h"

#define LPF_MASK_HEV(p3_in, p2_in, p1_in, p0_in, q0_in, q1_in, q2_in, q3_in, \
                     limit_in, b_limit_in, thresh_in, hev_out, mask_out,     \
                     flat_out)                                               \
  do {                                                                       \
    __m128i p3_asub_p2_m, p2_asub_p1_m, p1_asub_p0_m, q1_asub_q0_m;          \
    __m128i p1_asub_q1_m, p0_asub_q0_m, q3_asub_q2_m, q2_asub_q1_m;          \
                                                                             \
    /* absolute subtraction of pixel values */                               \
    p3_asub_p2_m = __lsx_vabsd_bu(p3_in, p2_in);                             \
    p2_asub_p1_m = __lsx_vabsd_bu(p2_in, p1_in);                             \
    p1_asub_p0_m = __lsx_vabsd_bu(p1_in, p0_in);                             \
    q1_asub_q0_m = __lsx_vabsd_bu(q1_in, q0_in);                             \
    q2_asub_q1_m = __lsx_vabsd_bu(q2_in, q1_in);                             \
    q3_asub_q2_m = __lsx_vabsd_bu(q3_in, q2_in);                             \
    p0_asub_q0_m = __lsx_vabsd_bu(p0_in, q0_in);                             \
    p1_asub_q1_m = __lsx_vabsd_bu(p1_in, q1_in);                             \
                                                                             \
    /* calculation of hev */                                                 \
    flat_out = __lsx_vmax_bu(p1_asub_p0_m, q1_asub_q0_m);                    \
    hev_out = __lsx_vslt_bu(thresh_in, flat_out);                            \
                                                                             \
    /* calculation of mask */                                                \
    p0_asub_q0_m = __lsx_vsadd_bu(p0_asub_q0_m, p0_asub_q0_m);               \
    p1_asub_q1_m = __lsx_vsrli_b(p1_asub_q1_m, 1);                           \
    p0_asub_q0_m = __lsx_vsadd_bu(p0_asub_q0_m, p1_asub_q1_m);               \
    mask_out = __lsx_vslt_bu(b_limit_in, p0_asub_q0_m);                      \
    mask_out = __lsx_vmax_bu(flat_out, mask_out);                            \
    p3_asub_p2_m = __lsx_vmax_bu(p3_asub_p2_m, p2_asub_p1_m);                \
    mask_out = __lsx_vmax_bu(p3_asub_p2_m, mask_out);                        \
    q2_asub_q1_m = __lsx_vmax_bu(q2_asub_q1_m, q3_asub_q2_m);                \
    mask_out = __lsx_vmax_bu(q2_asub_q1_m, mask_out);                        \
                                                                             \
    mask_out = __lsx_vslt_bu(limit_in, mask_out);                            \
    mask_out = __lsx_vxori_b(mask_out, 0xff);                                \
  } while (0)

#define VP9_FLAT4(p3_in, p2_in, p0_in, q0_in, q2_in, q3_in, flat_out)          \
  do {                                                                         \
    __m128i p2_asub_p0, q2_asub_q0, p3_asub_p0, q3_asub_q0;                    \
    __m128i flat4_tmp = __lsx_vldi(1);                                         \
                                                                               \
    DUP4_ARG2(__lsx_vabsd_bu, p2_in, p0_in, q2_in, q0_in, p3_in, p0_in, q3_in, \
              q0_in, p2_asub_p0, q2_asub_q0, p3_asub_p0, q3_asub_q0);          \
    p2_asub_p0 = __lsx_vmax_bu(p2_asub_p0, q2_asub_q0);                        \
    flat_out = __lsx_vmax_bu(p2_asub_p0, flat_out);                            \
    p3_asub_p0 = __lsx_vmax_bu(p3_asub_p0, q3_asub_q0);                        \
    flat_out = __lsx_vmax_bu(p3_asub_p0, flat_out);                            \
                                                                               \
    flat_out = __lsx_vslt_bu(flat4_tmp, flat_out);                             \
    flat_out = __lsx_vxori_b(flat_out, 0xff);                                  \
    flat_out = flat_out & (mask);                                              \
  } while (0)

#define VP9_FLAT5(p7_in, p6_in, p5_in, p4_in, p0_in, q0_in, q4_in, q5_in,      \
                  q6_in, q7_in, flat_in, flat2_out)                            \
  do {                                                                         \
    __m128i flat5_tmp = __lsx_vldi(1);                                         \
    __m128i p4_asub_p0, q4_asub_q0, p5_asub_p0, q5_asub_q0;                    \
    __m128i p6_asub_p0, q6_asub_q0, p7_asub_p0, q7_asub_q0;                    \
    DUP4_ARG2(__lsx_vabsd_bu, p4_in, p0_in, q4_in, q0_in, p5_in, p0_in, q5_in, \
              q0_in, p4_asub_p0, q4_asub_q0, p5_asub_p0, q5_asub_q0);          \
    DUP4_ARG2(__lsx_vabsd_bu, p6_in, p0_in, q6_in, q0_in, p7_in, p0_in, q7_in, \
              q0_in, p6_asub_p0, q6_asub_q0, p7_asub_p0, q7_asub_q0);          \
                                                                               \
    DUP2_ARG2(__lsx_vmax_bu, p4_asub_p0, q4_asub_q0, p5_asub_p0, q5_asub_q0,   \
              p4_asub_p0, flat2_out);                                          \
    flat2_out = __lsx_vmax_bu(p4_asub_p0, flat2_out);                          \
    p6_asub_p0 = __lsx_vmax_bu(p6_asub_p0, q6_asub_q0);                        \
    flat2_out = __lsx_vmax_bu(p6_asub_p0, flat2_out);                          \
    p7_asub_p0 = __lsx_vmax_bu(p7_asub_p0, q7_asub_q0);                        \
    flat2_out = __lsx_vmax_bu(p7_asub_p0, flat2_out);                          \
    flat2_out = __lsx_vslt_bu(flat5_tmp, flat2_out);                           \
    flat2_out = __lsx_vxori_b(flat2_out, 0xff);                                \
    flat2_out = flat2_out & flat_in;                                           \
  } while (0)

#define VP9_LPF_FILTER4_4W(p1_in, p0_in, q0_in, q1_in, mask, hev, p1_out,  \
                           p0_out, q0_out, q1_out)                         \
  do {                                                                     \
    __m128i p1_m, p0_m, q0_m, q1_m, filt, q0_sub_p0, t1, t2;               \
    const __m128i cnst4b = __lsx_vldi(4);                                  \
    const __m128i cnst3b = __lsx_vldi(3);                                  \
    DUP4_ARG2(__lsx_vxori_b, p1_in, 0x80, p0_in, 0x80, q0_in, 0x80, q1_in, \
              0x80, p1_m, p0_m, q0_m, q1_m);                               \
    filt = __lsx_vssub_b(p1_m, q1_m);                                      \
    filt &= hev;                                                           \
                                                                           \
    q0_sub_p0 = __lsx_vssub_b(q0_m, p0_m);                                 \
    filt = __lsx_vsadd_b(filt, q0_sub_p0);                                 \
    filt = __lsx_vsadd_b(filt, q0_sub_p0);                                 \
    filt = __lsx_vsadd_b(filt, q0_sub_p0);                                 \
    filt &= mask;                                                          \
    DUP2_ARG2(__lsx_vsadd_b, filt, cnst4b, filt, cnst3b, t1, t2);          \
    DUP2_ARG2(__lsx_vsrai_b, t1, 3, t2, 3, t1, t2);                        \
                                                                           \
    q0_m = __lsx_vssub_b(q0_m, t1);                                        \
    p0_m = __lsx_vsadd_b(p0_m, t2);                                        \
    DUP2_ARG2(__lsx_vxori_b, q0_m, 0x80, p0_m, 0x80, q0_out, p0_out);      \
                                                                           \
    filt = __lsx_vsrari_b(t1, 1);                                          \
    hev = __lsx_vxori_b(hev, 0xff);                                        \
    filt &= hev;                                                           \
    q1_m = __lsx_vssub_b(q1_m, filt);                                      \
    p1_m = __lsx_vsadd_b(p1_m, filt);                                      \
    DUP2_ARG2(__lsx_vxori_b, q1_m, 0x80, p1_m, 0x80, q1_out, p1_out);      \
  } while (0)

#define VP9_FILTER8(p3_in, p2_in, p1_in, p0_in, q0_in, q1_in, q2_in, q3_in, \
                    p2_filt8_out, p1_filt8_out, p0_filt8_out, q0_filt8_out, \
                    q1_filt8_out, q2_filt8_out)                             \
  do {                                                                      \
    __m128i tmp_filt8_0, tmp_filt8_1, tmp_filt8_2;                          \
                                                                            \
    tmp_filt8_2 = __lsx_vadd_h(p2_in, p1_in);                               \
    tmp_filt8_2 = __lsx_vadd_h(tmp_filt8_2, p0_in);                         \
    tmp_filt8_0 = __lsx_vslli_h(p3_in, 1);                                  \
                                                                            \
    tmp_filt8_0 = __lsx_vadd_h(tmp_filt8_0, tmp_filt8_2);                   \
    tmp_filt8_0 = __lsx_vadd_h(tmp_filt8_0, q0_in);                         \
    tmp_filt8_1 = __lsx_vadd_h(tmp_filt8_0, p3_in);                         \
    tmp_filt8_1 = __lsx_vadd_h(tmp_filt8_1, p2_in);                         \
    p2_filt8_out = __lsx_vsrari_h(tmp_filt8_1, 3);                          \
                                                                            \
    tmp_filt8_1 = __lsx_vadd_h(tmp_filt8_0, p1_in);                         \
    tmp_filt8_1 = __lsx_vadd_h(tmp_filt8_1, q1_in);                         \
    p1_filt8_out = __lsx_vsrari_h(tmp_filt8_1, 3);                          \
                                                                            \
    tmp_filt8_1 = __lsx_vadd_h(q2_in, q1_in);                               \
    tmp_filt8_1 = __lsx_vadd_h(tmp_filt8_1, q0_in);                         \
    tmp_filt8_2 = __lsx_vadd_h(tmp_filt8_2, tmp_filt8_1);                   \
    tmp_filt8_0 = __lsx_vadd_h(tmp_filt8_2, p0_in);                         \
    tmp_filt8_0 = __lsx_vadd_h(tmp_filt8_0, p3_in);                         \
    p0_filt8_out = __lsx_vsrari_h(tmp_filt8_0, 3);                          \
                                                                            \
    tmp_filt8_0 = __lsx_vadd_h(q2_in, q3_in);                               \
    tmp_filt8_0 = __lsx_vadd_h(p0_in, tmp_filt8_0);                         \
    tmp_filt8_0 = __lsx_vadd_h(tmp_filt8_0, tmp_filt8_1);                   \
    tmp_filt8_1 = __lsx_vadd_h(q3_in, q3_in);                               \
    tmp_filt8_1 = __lsx_vadd_h(tmp_filt8_1, tmp_filt8_0);                   \
    q2_filt8_out = __lsx_vsrari_h(tmp_filt8_1, 3);                          \
                                                                            \
    tmp_filt8_0 = __lsx_vadd_h(tmp_filt8_2, q3_in);                         \
    tmp_filt8_1 = __lsx_vadd_h(tmp_filt8_0, q0_in);                         \
    q0_filt8_out = __lsx_vsrari_h(tmp_filt8_1, 3);                          \
                                                                            \
    tmp_filt8_1 = __lsx_vsub_h(tmp_filt8_0, p2_in);                         \
    tmp_filt8_0 = __lsx_vadd_h(q1_in, q3_in);                               \
    tmp_filt8_1 = __lsx_vadd_h(tmp_filt8_0, tmp_filt8_1);                   \
    q1_filt8_out = __lsx_vsrari_h(tmp_filt8_1, 3);                          \
  } while (0)

#endif  // VPX_VPX_DSP_LOONGARCH_LOOPFILTER_LSX_H_
