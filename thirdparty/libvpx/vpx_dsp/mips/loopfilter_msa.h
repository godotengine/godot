/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_MIPS_LOOPFILTER_MSA_H_
#define VPX_VPX_DSP_MIPS_LOOPFILTER_MSA_H_

#include "vpx_dsp/mips/macros_msa.h"

#define VP9_LPF_FILTER4_4W(p1_in, p0_in, q0_in, q1_in, mask, hev, p1_out, \
                           p0_out, q0_out, q1_out)                        \
  {                                                                       \
    v16i8 p1_m, p0_m, q0_m, q1_m, filt, q0_sub_p0, t1, t2;                \
    const v16i8 cnst4b = __msa_ldi_b(4);                                  \
    const v16i8 cnst3b = __msa_ldi_b(3);                                  \
                                                                          \
    p1_m = (v16i8)__msa_xori_b(p1_in, 0x80);                              \
    p0_m = (v16i8)__msa_xori_b(p0_in, 0x80);                              \
    q0_m = (v16i8)__msa_xori_b(q0_in, 0x80);                              \
    q1_m = (v16i8)__msa_xori_b(q1_in, 0x80);                              \
                                                                          \
    filt = __msa_subs_s_b(p1_m, q1_m);                                    \
    filt &= hev;                                                          \
    q0_sub_p0 = __msa_subs_s_b(q0_m, p0_m);                               \
    filt = __msa_adds_s_b(filt, q0_sub_p0);                               \
    filt = __msa_adds_s_b(filt, q0_sub_p0);                               \
    filt = __msa_adds_s_b(filt, q0_sub_p0);                               \
    filt &= mask;                                                         \
    t1 = __msa_adds_s_b(filt, cnst4b);                                    \
    t1 >>= cnst3b;                                                        \
    t2 = __msa_adds_s_b(filt, cnst3b);                                    \
    t2 >>= cnst3b;                                                        \
    q0_m = __msa_subs_s_b(q0_m, t1);                                      \
    q0_out = __msa_xori_b((v16u8)q0_m, 0x80);                             \
    p0_m = __msa_adds_s_b(p0_m, t2);                                      \
    p0_out = __msa_xori_b((v16u8)p0_m, 0x80);                             \
    filt = __msa_srari_b(t1, 1);                                          \
    hev = __msa_xori_b(hev, 0xff);                                        \
    filt &= hev;                                                          \
    q1_m = __msa_subs_s_b(q1_m, filt);                                    \
    q1_out = __msa_xori_b((v16u8)q1_m, 0x80);                             \
    p1_m = __msa_adds_s_b(p1_m, filt);                                    \
    p1_out = __msa_xori_b((v16u8)p1_m, 0x80);                             \
  }

#define VP9_FLAT4(p3_in, p2_in, p0_in, q0_in, q2_in, q3_in, flat_out)    \
  {                                                                      \
    v16u8 tmp_flat4, p2_a_sub_p0, q2_a_sub_q0, p3_a_sub_p0, q3_a_sub_q0; \
    v16u8 zero_in = { 0 };                                               \
                                                                         \
    tmp_flat4 = __msa_ori_b(zero_in, 1);                                 \
    p2_a_sub_p0 = __msa_asub_u_b(p2_in, p0_in);                          \
    q2_a_sub_q0 = __msa_asub_u_b(q2_in, q0_in);                          \
    p3_a_sub_p0 = __msa_asub_u_b(p3_in, p0_in);                          \
    q3_a_sub_q0 = __msa_asub_u_b(q3_in, q0_in);                          \
                                                                         \
    p2_a_sub_p0 = __msa_max_u_b(p2_a_sub_p0, q2_a_sub_q0);               \
    flat_out = __msa_max_u_b(p2_a_sub_p0, flat_out);                     \
    p3_a_sub_p0 = __msa_max_u_b(p3_a_sub_p0, q3_a_sub_q0);               \
    flat_out = __msa_max_u_b(p3_a_sub_p0, flat_out);                     \
                                                                         \
    flat_out = (tmp_flat4 < (v16u8)flat_out);                            \
    flat_out = __msa_xori_b(flat_out, 0xff);                             \
    flat_out = flat_out & (mask);                                        \
  }

#define VP9_FLAT5(p7_in, p6_in, p5_in, p4_in, p0_in, q0_in, q4_in, q5_in, \
                  q6_in, q7_in, flat_in, flat2_out)                       \
  {                                                                       \
    v16u8 tmp_flat5, zero_in = { 0 };                                     \
    v16u8 p4_a_sub_p0, q4_a_sub_q0, p5_a_sub_p0, q5_a_sub_q0;             \
    v16u8 p6_a_sub_p0, q6_a_sub_q0, p7_a_sub_p0, q7_a_sub_q0;             \
                                                                          \
    tmp_flat5 = __msa_ori_b(zero_in, 1);                                  \
    p4_a_sub_p0 = __msa_asub_u_b(p4_in, p0_in);                           \
    q4_a_sub_q0 = __msa_asub_u_b(q4_in, q0_in);                           \
    p5_a_sub_p0 = __msa_asub_u_b(p5_in, p0_in);                           \
    q5_a_sub_q0 = __msa_asub_u_b(q5_in, q0_in);                           \
    p6_a_sub_p0 = __msa_asub_u_b(p6_in, p0_in);                           \
    q6_a_sub_q0 = __msa_asub_u_b(q6_in, q0_in);                           \
    p7_a_sub_p0 = __msa_asub_u_b(p7_in, p0_in);                           \
    q7_a_sub_q0 = __msa_asub_u_b(q7_in, q0_in);                           \
                                                                          \
    p4_a_sub_p0 = __msa_max_u_b(p4_a_sub_p0, q4_a_sub_q0);                \
    flat2_out = __msa_max_u_b(p5_a_sub_p0, q5_a_sub_q0);                  \
    flat2_out = __msa_max_u_b(p4_a_sub_p0, flat2_out);                    \
    p6_a_sub_p0 = __msa_max_u_b(p6_a_sub_p0, q6_a_sub_q0);                \
    flat2_out = __msa_max_u_b(p6_a_sub_p0, flat2_out);                    \
    p7_a_sub_p0 = __msa_max_u_b(p7_a_sub_p0, q7_a_sub_q0);                \
    flat2_out = __msa_max_u_b(p7_a_sub_p0, flat2_out);                    \
                                                                          \
    flat2_out = (tmp_flat5 < (v16u8)flat2_out);                           \
    flat2_out = __msa_xori_b(flat2_out, 0xff);                            \
    flat2_out = flat2_out & flat_in;                                      \
  }

#define VP9_FILTER8(p3_in, p2_in, p1_in, p0_in, q0_in, q1_in, q2_in, q3_in, \
                    p2_filt8_out, p1_filt8_out, p0_filt8_out, q0_filt8_out, \
                    q1_filt8_out, q2_filt8_out)                             \
  {                                                                         \
    v8u16 tmp_filt8_0, tmp_filt8_1, tmp_filt8_2;                            \
                                                                            \
    tmp_filt8_2 = p2_in + p1_in + p0_in;                                    \
    tmp_filt8_0 = p3_in << 1;                                               \
                                                                            \
    tmp_filt8_0 = tmp_filt8_0 + tmp_filt8_2 + q0_in;                        \
    tmp_filt8_1 = tmp_filt8_0 + p3_in + p2_in;                              \
    p2_filt8_out = (v8i16)__msa_srari_h((v8i16)tmp_filt8_1, 3);             \
                                                                            \
    tmp_filt8_1 = tmp_filt8_0 + p1_in + q1_in;                              \
    p1_filt8_out = (v8i16)__msa_srari_h((v8i16)tmp_filt8_1, 3);             \
                                                                            \
    tmp_filt8_1 = q2_in + q1_in + q0_in;                                    \
    tmp_filt8_2 = tmp_filt8_2 + tmp_filt8_1;                                \
    tmp_filt8_0 = tmp_filt8_2 + (p0_in);                                    \
    tmp_filt8_0 = tmp_filt8_0 + (p3_in);                                    \
    p0_filt8_out = (v8i16)__msa_srari_h((v8i16)tmp_filt8_0, 3);             \
                                                                            \
    tmp_filt8_0 = q2_in + q3_in;                                            \
    tmp_filt8_0 = p0_in + tmp_filt8_1 + tmp_filt8_0;                        \
    tmp_filt8_1 = q3_in + q3_in;                                            \
    tmp_filt8_1 = tmp_filt8_1 + tmp_filt8_0;                                \
    q2_filt8_out = (v8i16)__msa_srari_h((v8i16)tmp_filt8_1, 3);             \
                                                                            \
    tmp_filt8_0 = tmp_filt8_2 + q3_in;                                      \
    tmp_filt8_1 = tmp_filt8_0 + q0_in;                                      \
    q0_filt8_out = (v8i16)__msa_srari_h((v8i16)tmp_filt8_1, 3);             \
                                                                            \
    tmp_filt8_1 = tmp_filt8_0 - p2_in;                                      \
    tmp_filt8_0 = q1_in + q3_in;                                            \
    tmp_filt8_1 = tmp_filt8_0 + tmp_filt8_1;                                \
    q1_filt8_out = (v8i16)__msa_srari_h((v8i16)tmp_filt8_1, 3);             \
  }

#define LPF_MASK_HEV(p3_in, p2_in, p1_in, p0_in, q0_in, q1_in, q2_in, q3_in, \
                     limit_in, b_limit_in, thresh_in, hev_out, mask_out,     \
                     flat_out)                                               \
  {                                                                          \
    v16u8 p3_asub_p2_m, p2_asub_p1_m, p1_asub_p0_m, q1_asub_q0_m;            \
    v16u8 p1_asub_q1_m, p0_asub_q0_m, q3_asub_q2_m, q2_asub_q1_m;            \
                                                                             \
    /* absolute subtraction of pixel values */                               \
    p3_asub_p2_m = __msa_asub_u_b(p3_in, p2_in);                             \
    p2_asub_p1_m = __msa_asub_u_b(p2_in, p1_in);                             \
    p1_asub_p0_m = __msa_asub_u_b(p1_in, p0_in);                             \
    q1_asub_q0_m = __msa_asub_u_b(q1_in, q0_in);                             \
    q2_asub_q1_m = __msa_asub_u_b(q2_in, q1_in);                             \
    q3_asub_q2_m = __msa_asub_u_b(q3_in, q2_in);                             \
    p0_asub_q0_m = __msa_asub_u_b(p0_in, q0_in);                             \
    p1_asub_q1_m = __msa_asub_u_b(p1_in, q1_in);                             \
                                                                             \
    /* calculation of hev */                                                 \
    flat_out = __msa_max_u_b(p1_asub_p0_m, q1_asub_q0_m);                    \
    hev_out = thresh_in < (v16u8)flat_out;                                   \
                                                                             \
    /* calculation of mask */                                                \
    p0_asub_q0_m = __msa_adds_u_b(p0_asub_q0_m, p0_asub_q0_m);               \
    p1_asub_q1_m >>= 1;                                                      \
    p0_asub_q0_m = __msa_adds_u_b(p0_asub_q0_m, p1_asub_q1_m);               \
                                                                             \
    mask_out = b_limit_in < p0_asub_q0_m;                                    \
    mask_out = __msa_max_u_b(flat_out, mask_out);                            \
    p3_asub_p2_m = __msa_max_u_b(p3_asub_p2_m, p2_asub_p1_m);                \
    mask_out = __msa_max_u_b(p3_asub_p2_m, mask_out);                        \
    q2_asub_q1_m = __msa_max_u_b(q2_asub_q1_m, q3_asub_q2_m);                \
    mask_out = __msa_max_u_b(q2_asub_q1_m, mask_out);                        \
                                                                             \
    mask_out = limit_in < (v16u8)mask_out;                                   \
    mask_out = __msa_xori_b(mask_out, 0xff);                                 \
  }
#endif  // VPX_VPX_DSP_MIPS_LOOPFILTER_MSA_H_
