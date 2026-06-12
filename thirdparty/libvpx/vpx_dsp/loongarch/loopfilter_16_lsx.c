/*
 * Copyright (c) 2022 Loongson Technology Corporation Limited
 * Contributed by Hecai Yuan <yuanhecai@loongson.cn>
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS.  All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/loongarch/loopfilter_lsx.h"
#include "vpx_ports/mem.h"

#define LSX_LD_8(_src, _stride, _stride2, _stride3, _stride4, _in0, _in1, \
                 _in2, _in3, _in4, _in5, _in6, _in7)                      \
  do {                                                                    \
    _in0 = __lsx_vld(_src, 0);                                            \
    _in1 = __lsx_vldx(_src, _stride);                                     \
    _in2 = __lsx_vldx(_src, _stride2);                                    \
    _in3 = __lsx_vldx(_src, _stride3);                                    \
    _src += _stride4;                                                     \
    _in4 = __lsx_vld(_src, 0);                                            \
    _in5 = __lsx_vldx(_src, _stride);                                     \
    _in6 = __lsx_vldx(_src, _stride2);                                    \
    _in7 = __lsx_vldx(_src, _stride3);                                    \
  } while (0)

#define LSX_ST_8(_dst0, _dst1, _dst2, _dst3, _dst4, _dst5, _dst6, _dst7, _dst, \
                 _stride, _stride2, _stride3, _stride4)                        \
  do {                                                                         \
    __lsx_vst(_dst0, _dst, 0);                                                 \
    __lsx_vstx(_dst1, _dst, _stride);                                          \
    __lsx_vstx(_dst2, _dst, _stride2);                                         \
    __lsx_vstx(_dst3, _dst, _stride3);                                         \
    _dst += _stride4;                                                          \
    __lsx_vst(_dst4, _dst, 0);                                                 \
    __lsx_vstx(_dst5, _dst, _stride);                                          \
    __lsx_vstx(_dst6, _dst, _stride2);                                         \
    __lsx_vstx(_dst7, _dst, _stride3);                                         \
  } while (0)

static int32_t hz_lpf_t4_and_t8_16w(uint8_t *dst, int32_t stride,
                                    uint8_t *filter48,
                                    const uint8_t *b_limit_ptr,
                                    const uint8_t *limit_ptr,
                                    const uint8_t *thresh_ptr) {
  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i p2_out, p1_out, p0_out, q0_out, q1_out, q2_out;
  __m128i flat, mask, hev, thresh, b_limit, limit;
  __m128i p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l;
  __m128i p3_h, p2_h, p1_h, p0_h, q0_h, q1_h, q2_h, q3_h;
  __m128i p2_filt8_l, p1_filt8_l, p0_filt8_l;
  __m128i q0_filt8_l, q1_filt8_l, q2_filt8_l;
  __m128i p2_filt8_h, p1_filt8_h, p0_filt8_h;
  __m128i q0_filt8_h, q1_filt8_h, q2_filt8_h;

  int32_t stride2 = stride << 1;
  int32_t stride3 = stride2 + stride;
  int32_t stride4 = stride2 << 1;

  /* load vector elements */
  DUP4_ARG2(__lsx_vldx, dst, -stride4, dst, -stride3, dst, -stride2, dst,
            -stride, p3, p2, p1, p0);

  q0 = __lsx_vld(dst, 0);
  DUP2_ARG2(__lsx_vldx, dst, stride, dst, stride2, q1, q2);
  q3 = __lsx_vldx(dst, stride3);

  thresh = __lsx_vldrepl_b(thresh_ptr, 0);
  b_limit = __lsx_vldrepl_b(b_limit_ptr, 0);
  limit = __lsx_vldrepl_b(limit_ptr, 0);
  /* mask and hev */
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  VP9_FLAT4(p3, p2, p0, q0, q2, q3, flat);
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1_out, p0_out, q0_out, q1_out);

  if (__lsx_bz_v(flat)) {
    __lsx_vstx(p1_out, dst, -stride2);
    __lsx_vstx(p0_out, dst, -stride);
    __lsx_vst(q0_out, dst, 0);
    __lsx_vstx(q1_out, dst, stride);

    return 1;
  }

  DUP4_ARG2(__lsx_vsllwil_hu_bu, p3, 0, p2, 0, p1, 0, p0, 0, p3_l, p2_l, p1_l,
            p0_l);
  DUP4_ARG2(__lsx_vsllwil_hu_bu, q0, 0, q1, 0, q2, 0, q3, 0, q0_l, q1_l, q2_l,
            q3_l);

  VP9_FILTER8(p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l, p2_filt8_l,
              p1_filt8_l, p0_filt8_l, q0_filt8_l, q1_filt8_l, q2_filt8_l);

  DUP4_ARG1(__lsx_vexth_hu_bu, p3, p2, p1, p0, p3_h, p2_h, p1_h, p0_h);
  DUP4_ARG1(__lsx_vexth_hu_bu, q0, q1, q2, q3, q0_h, q1_h, q2_h, q3_h);
  VP9_FILTER8(p3_h, p2_h, p1_h, p0_h, q0_h, q1_h, q2_h, q3_h, p2_filt8_h,
              p1_filt8_h, p0_filt8_h, q0_filt8_h, q1_filt8_h, q2_filt8_h);

  /* convert 16 bit output data into 8 bit */
  DUP4_ARG2(__lsx_vpickev_b, p2_filt8_h, p2_filt8_l, p1_filt8_h, p1_filt8_l,
            p0_filt8_h, p0_filt8_l, q0_filt8_h, q0_filt8_l, p2_filt8_l,
            p1_filt8_l, p0_filt8_l, q0_filt8_l);
  DUP2_ARG2(__lsx_vpickev_b, q1_filt8_h, q1_filt8_l, q2_filt8_h, q2_filt8_l,
            q1_filt8_l, q2_filt8_l);

  /* store pixel values */
  DUP4_ARG3(__lsx_vbitsel_v, p2, p2_filt8_l, flat, p1_out, p1_filt8_l, flat,
            p0_out, p0_filt8_l, flat, q0_out, q0_filt8_l, flat, p2_out, p1_out,
            p0_out, q0_out);
  DUP2_ARG3(__lsx_vbitsel_v, q1_out, q1_filt8_l, flat, q2, q2_filt8_l, flat,
            q1_out, q2_out);

  __lsx_vst(p2_out, filter48, 0);
  __lsx_vst(p1_out, filter48, 16);
  __lsx_vst(p0_out, filter48, 32);
  __lsx_vst(q0_out, filter48, 48);
  __lsx_vst(q1_out, filter48, 64);
  __lsx_vst(q2_out, filter48, 80);
  __lsx_vst(flat, filter48, 96);

  return 0;
}

static void hz_lpf_t16_16w(uint8_t *dst, int32_t stride, uint8_t *filter48) {
  int32_t stride2 = stride << 1;
  int32_t stride3 = stride2 + stride;
  int32_t stride4 = stride2 << 1;
  uint8_t *dst_tmp0 = dst - stride4;
  uint8_t *dst_tmp1 = dst + stride4;

  __m128i flat, flat2, filter8;
  __m128i p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7;
  __m128i out_h, out_l;
  v8u16 p7_l_in, p6_l_in, p5_l_in, p4_l_in;
  v8u16 p3_l_in, p2_l_in, p1_l_in, p0_l_in;
  v8u16 q7_l_in, q6_l_in, q5_l_in, q4_l_in;
  v8u16 q3_l_in, q2_l_in, q1_l_in, q0_l_in;
  v8u16 p7_h_in, p6_h_in, p5_h_in, p4_h_in;
  v8u16 p3_h_in, p2_h_in, p1_h_in, p0_h_in;
  v8u16 q7_h_in, q6_h_in, q5_h_in, q4_h_in;
  v8u16 q3_h_in, q2_h_in, q1_h_in, q0_h_in;
  v8u16 tmp0_l, tmp1_l, tmp0_h, tmp1_h;

  flat = __lsx_vld(filter48, 96);

  DUP4_ARG2(__lsx_vldx, dst_tmp0, -stride4, dst_tmp0, -stride3, dst_tmp0,
            -stride2, dst_tmp0, -stride, p7, p6, p5, p4);

  p3 = __lsx_vld(dst_tmp0, 0);
  DUP2_ARG2(__lsx_vldx, dst_tmp0, stride, dst_tmp0, stride2, p2, p1);
  p0 = __lsx_vldx(dst_tmp0, stride3);

  q0 = __lsx_vld(dst, 0);
  DUP2_ARG2(__lsx_vldx, dst, stride, dst, stride2, q1, q2);
  q3 = __lsx_vldx(dst, stride3);

  q4 = __lsx_vld(dst_tmp1, 0);
  DUP2_ARG2(__lsx_vldx, dst_tmp1, stride, dst_tmp1, stride2, q5, q6);
  q7 = __lsx_vldx(dst_tmp1, stride3);

  VP9_FLAT5(p7, p6, p5, p4, p0, q0, q4, q5, q6, q7, flat, flat2);

  if (__lsx_bz_v(flat2)) {
    DUP4_ARG2(__lsx_vld, filter48, 0, filter48, 16, filter48, 32, filter48, 48,
              p2, p1, p0, q0);
    DUP2_ARG2(__lsx_vld, filter48, 64, filter48, 80, q1, q2);
    __lsx_vstx(p2, dst, -stride3);
    __lsx_vstx(p1, dst, -stride2);
    __lsx_vstx(p0, dst, -stride);
    __lsx_vst(q0, dst, 0);
    __lsx_vstx(q1, dst, stride);
    __lsx_vstx(q2, dst, stride2);
  } else {
    dst = dst_tmp0 - stride3;

    p7_l_in = (v8u16)__lsx_vsllwil_hu_bu(p7, 0);
    p6_l_in = (v8u16)__lsx_vsllwil_hu_bu(p6, 0);
    p5_l_in = (v8u16)__lsx_vsllwil_hu_bu(p5, 0);
    p4_l_in = (v8u16)__lsx_vsllwil_hu_bu(p4, 0);
    p3_l_in = (v8u16)__lsx_vsllwil_hu_bu(p3, 0);
    p2_l_in = (v8u16)__lsx_vsllwil_hu_bu(p2, 0);
    p1_l_in = (v8u16)__lsx_vsllwil_hu_bu(p1, 0);
    p0_l_in = (v8u16)__lsx_vsllwil_hu_bu(p0, 0);
    q0_l_in = (v8u16)__lsx_vsllwil_hu_bu(q0, 0);

    tmp0_l = p7_l_in << 3;
    tmp0_l -= p7_l_in;
    tmp0_l += p6_l_in;
    tmp0_l += q0_l_in;
    tmp1_l = p6_l_in + p5_l_in;
    tmp1_l += p4_l_in;
    tmp1_l += p3_l_in;
    tmp1_l += p2_l_in;
    tmp1_l += p1_l_in;
    tmp1_l += p0_l_in;
    tmp1_l += tmp0_l;

    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    p7_h_in = (v8u16)__lsx_vexth_hu_bu(p7);
    p6_h_in = (v8u16)__lsx_vexth_hu_bu(p6);
    p5_h_in = (v8u16)__lsx_vexth_hu_bu(p5);
    p4_h_in = (v8u16)__lsx_vexth_hu_bu(p4);
    p3_h_in = (v8u16)__lsx_vexth_hu_bu(p3);
    p2_h_in = (v8u16)__lsx_vexth_hu_bu(p2);
    p1_h_in = (v8u16)__lsx_vexth_hu_bu(p1);
    p0_h_in = (v8u16)__lsx_vexth_hu_bu(p0);
    q0_h_in = (v8u16)__lsx_vexth_hu_bu(q0);

    tmp0_h = p7_h_in << 3;
    tmp0_h -= p7_h_in;
    tmp0_h += p6_h_in;
    tmp0_h += q0_h_in;
    tmp1_h = p6_h_in + p5_h_in;
    tmp1_h += p4_h_in;
    tmp1_h += p3_h_in;
    tmp1_h += p2_h_in;
    tmp1_h += p1_h_in;
    tmp1_h += p0_h_in;
    tmp1_h += tmp0_h;

    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    p6 = __lsx_vbitsel_v(p6, out_l, flat2);
    __lsx_vst(p6, dst, 0);
    dst += stride;

    /* p5 */
    q1_l_in = (v8u16)__lsx_vsllwil_hu_bu(q1, 0);
    tmp0_l = p5_l_in - p6_l_in;
    tmp0_l += q1_l_in;
    tmp0_l -= p7_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    q1_h_in = (v8u16)__lsx_vexth_hu_bu(q1);
    tmp0_h = p5_h_in - p6_h_in;
    tmp0_h += q1_h_in;
    tmp0_h -= p7_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    p5 = __lsx_vbitsel_v(p5, out_l, flat2);
    __lsx_vst(p5, dst, 0);
    dst += stride;

    /* p4 */
    q2_l_in = (v8u16)__lsx_vsllwil_hu_bu(q2, 0);
    tmp0_l = p4_l_in - p5_l_in;
    tmp0_l += q2_l_in;
    tmp0_l -= p7_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    q2_h_in = (v8u16)__lsx_vexth_hu_bu(q2);
    tmp0_h = p4_h_in - p5_h_in;
    tmp0_h += q2_h_in;
    tmp0_h -= p7_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    p4 = __lsx_vbitsel_v(p4, out_l, flat2);
    __lsx_vst(p4, dst, 0);
    dst += stride;

    /* p3 */
    q3_l_in = (v8u16)__lsx_vsllwil_hu_bu(q3, 0);
    tmp0_l = p3_l_in - p4_l_in;
    tmp0_l += q3_l_in;
    tmp0_l -= p7_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    q3_h_in = (v8u16)__lsx_vexth_hu_bu(q3);
    tmp0_h = p3_h_in - p4_h_in;
    tmp0_h += q3_h_in;
    tmp0_h -= p7_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    p3 = __lsx_vbitsel_v(p3, out_l, flat2);
    __lsx_vst(p3, dst, 0);
    dst += stride;

    /* p2 */
    q4_l_in = (v8u16)__lsx_vsllwil_hu_bu(q4, 0);
    filter8 = __lsx_vld(filter48, 0);
    tmp0_l = p2_l_in - p3_l_in;
    tmp0_l += q4_l_in;
    tmp0_l -= p7_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    q4_h_in = (v8u16)__lsx_vexth_hu_bu(q4);
    tmp0_h = p2_h_in - p3_h_in;
    tmp0_h += q4_h_in;
    tmp0_h -= p7_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
    __lsx_vst(filter8, dst, 0);
    dst += stride;

    /* p1 */
    q5_l_in = (v8u16)__lsx_vsllwil_hu_bu(q5, 0);
    filter8 = __lsx_vld(filter48, 16);
    tmp0_l = p1_l_in - p2_l_in;
    tmp0_l += q5_l_in;
    tmp0_l -= p7_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    q5_h_in = (v8u16)__lsx_vexth_hu_bu(q5);
    tmp0_h = p1_h_in - p2_h_in;
    tmp0_h += q5_h_in;
    tmp0_h -= p7_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
    __lsx_vst(filter8, dst, 0);
    dst += stride;

    /* p0 */
    q6_l_in = (v8u16)__lsx_vsllwil_hu_bu(q6, 0);
    filter8 = __lsx_vld(filter48, 32);
    tmp0_l = p0_l_in - p1_l_in;
    tmp0_l += q6_l_in;
    tmp0_l -= p7_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    q6_h_in = (v8u16)__lsx_vexth_hu_bu(q6);
    tmp0_h = p0_h_in - p1_h_in;
    tmp0_h += q6_h_in;
    tmp0_h -= p7_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
    __lsx_vst(filter8, dst, 0);
    dst += stride;

    /* q0 */
    q7_l_in = (v8u16)__lsx_vsllwil_hu_bu(q7, 0);
    filter8 = __lsx_vld(filter48, 48);
    tmp0_l = q7_l_in - p0_l_in;
    tmp0_l += q0_l_in;
    tmp0_l -= p7_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    q7_h_in = (v8u16)__lsx_vexth_hu_bu(q7);
    tmp0_h = q7_h_in - p0_h_in;
    tmp0_h += q0_h_in;
    tmp0_h -= p7_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
    __lsx_vst(filter8, dst, 0);
    dst += stride;

    /* q1 */
    filter8 = __lsx_vld(filter48, 64);
    tmp0_l = q7_l_in - q0_l_in;
    tmp0_l += q1_l_in;
    tmp0_l -= p6_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    tmp0_h = q7_h_in - q0_h_in;
    tmp0_h += q1_h_in;
    tmp0_h -= p6_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
    __lsx_vst(filter8, dst, 0);
    dst += stride;

    /* q2 */
    filter8 = __lsx_vld(filter48, 80);
    tmp0_l = q7_l_in - q1_l_in;
    tmp0_l += q2_l_in;
    tmp0_l -= p5_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    tmp0_h = q7_h_in - q1_h_in;
    tmp0_h += q2_h_in;
    tmp0_h -= p5_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
    __lsx_vst(filter8, dst, 0);
    dst += stride;

    /* q3 */
    tmp0_l = q7_l_in - q2_l_in;
    tmp0_l += q3_l_in;
    tmp0_l -= p4_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    tmp0_h = q7_h_in - q2_h_in;
    tmp0_h += q3_h_in;
    tmp0_h -= p4_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    q3 = __lsx_vbitsel_v(q3, out_l, flat2);
    __lsx_vst(q3, dst, 0);
    dst += stride;

    /* q4 */
    tmp0_l = q7_l_in - q3_l_in;
    tmp0_l += q4_l_in;
    tmp0_l -= p3_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    tmp0_h = q7_h_in - q3_h_in;
    tmp0_h += q4_h_in;
    tmp0_h -= p3_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    q4 = __lsx_vbitsel_v(q4, out_l, flat2);
    __lsx_vst(q4, dst, 0);
    dst += stride;

    /* q5 */
    tmp0_l = q7_l_in - q4_l_in;
    tmp0_l += q5_l_in;
    tmp0_l -= p2_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    tmp0_h = q7_h_in - q4_h_in;
    tmp0_h += q5_h_in;
    tmp0_h -= p2_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    q5 = __lsx_vbitsel_v(q5, out_l, flat2);
    __lsx_vst(q5, dst, 0);
    dst += stride;

    /* q6 */
    tmp0_l = q7_l_in - q5_l_in;
    tmp0_l += q6_l_in;
    tmp0_l -= p1_l_in;
    tmp1_l += tmp0_l;
    out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);

    tmp0_h = q7_h_in - q5_h_in;
    tmp0_h += q6_h_in;
    tmp0_h -= p1_h_in;
    tmp1_h += tmp0_h;
    out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

    out_l = __lsx_vpickev_b(out_h, out_l);
    q6 = __lsx_vbitsel_v(q6, out_l, flat2);
    __lsx_vst(q6, dst, 0);
  }
}

static void mb_lpf_horizontal_edge_dual(uint8_t *dst, int32_t stride,
                                        const uint8_t *b_limit_ptr,
                                        const uint8_t *limit_ptr,
                                        const uint8_t *thresh_ptr) {
  DECLARE_ALIGNED(16, uint8_t, filter48[16 * 8]);
  uint8_t early_exit = 0;

  early_exit = hz_lpf_t4_and_t8_16w(dst, stride, &filter48[0], b_limit_ptr,
                                    limit_ptr, thresh_ptr);

  if (early_exit == 0) {
    hz_lpf_t16_16w(dst, stride, filter48);
  }
}

static void mb_lpf_horizontal_edge(uint8_t *dst, int32_t stride,
                                   const uint8_t *b_limit_ptr,
                                   const uint8_t *limit_ptr,
                                   const uint8_t *thresh_ptr, int32_t count) {
  if (count == 1) {
    __m128i flat2, mask, hev, flat, thresh, b_limit, limit;
    __m128i p3, p2, p1, p0, q3, q2, q1, q0, p7, p6, p5, p4, q4, q5, q6, q7;
    __m128i p2_out, p1_out, p0_out, q0_out, q1_out, q2_out;
    __m128i p0_filter16, p1_filter16;
    __m128i p2_filter8, p1_filter8, p0_filter8;
    __m128i q0_filter8, q1_filter8, q2_filter8;
    __m128i p7_l, p6_l, p5_l, p4_l, q7_l, q6_l, q5_l, q4_l;
    __m128i p3_l, p2_l, p1_l, p0_l, q3_l, q2_l, q1_l, q0_l;
    __m128i zero = __lsx_vldi(0);
    __m128i tmp0, tmp1, tmp2;

    int32_t stride2 = stride << 1;
    int32_t stride3 = 2 + stride;
    int32_t stride4 = stride << 2;
    uint8_t *dst_tmp0 = dst - stride4;
    uint8_t *dst_tmp1 = dst + stride4;

    /* load vector elements */
    DUP4_ARG2(__lsx_vldx, dst, -stride4, dst, -stride3, dst, -stride2, dst,
              -stride, p3, p2, p1, p0);
    q0 = __lsx_vld(dst, 0);
    DUP2_ARG2(__lsx_vldx, dst, stride, dst, stride2, q1, q2);
    q3 = __lsx_vldx(dst, stride3);

    thresh = __lsx_vldrepl_b(thresh_ptr, 0);
    b_limit = __lsx_vldrepl_b(b_limit_ptr, 0);
    limit = __lsx_vldrepl_b(limit_ptr, 0);

    /* filter_mask* */
    LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
                 mask, flat);
    VP9_FLAT4(p3, p2, p0, q0, q2, q3, flat);
    VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1_out, p0_out, q0_out,
                       q1_out);
    flat = __lsx_vilvl_d(zero, flat);
    if (__lsx_bz_v(flat)) {
      __lsx_vstelm_d(p1_out, dst - stride2, 0, 0);
      __lsx_vstelm_d(p0_out, dst - stride, 0, 0);
      __lsx_vstelm_d(q0_out, dst, 0, 0);
      __lsx_vstelm_d(q1_out, dst + stride, 0, 0);
    } else {
      /* convert 8 bit input data into 16 bit */
      DUP4_ARG2(__lsx_vilvl_b, zero, p3, zero, p2, zero, p1, zero, p0, p3_l,
                p2_l, p1_l, p0_l);
      DUP4_ARG2(__lsx_vilvl_b, zero, q0, zero, q1, zero, q2, zero, q3, q0_l,
                q1_l, q2_l, q3_l);
      VP9_FILTER8(p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l, p2_filter8,
                  p1_filter8, p0_filter8, q0_filter8, q1_filter8, q2_filter8);

      /* convert 16 bit output data into 8 bit */
      DUP4_ARG2(__lsx_vpickev_b, zero, p2_filter8, zero, p1_filter8, zero,
                p0_filter8, zero, q0_filter8, p2_filter8, p1_filter8,
                p0_filter8, q0_filter8);
      DUP2_ARG2(__lsx_vpickev_b, zero, q1_filter8, zero, q2_filter8, q1_filter8,
                q2_filter8);

      /* store pixel values */
      p2_out = __lsx_vbitsel_v(p2, p2_filter8, flat);
      p1_out = __lsx_vbitsel_v(p1_out, p1_filter8, flat);
      p0_out = __lsx_vbitsel_v(p0_out, p0_filter8, flat);
      q0_out = __lsx_vbitsel_v(q0_out, q0_filter8, flat);
      q1_out = __lsx_vbitsel_v(q1_out, q1_filter8, flat);
      q2_out = __lsx_vbitsel_v(q2, q2_filter8, flat);

      /* load 16 vector elements */
      DUP4_ARG2(__lsx_vldx, dst_tmp0, -stride4, dst_tmp0, -stride3, dst_tmp0,
                -stride2, dst_tmp0, -stride, p7, p6, p5, p4);
      q4 = __lsx_vld(dst_tmp1, 0);
      DUP2_ARG2(__lsx_vldx, dst_tmp1, stride, dst_tmp1, stride2, q5, q6);
      q7 = __lsx_vldx(dst_tmp1, stride3);

      VP9_FLAT5(p7, p6, p5, p4, p0, q0, q4, q5, q6, q7, flat, flat2);

      if (__lsx_bz_v(flat2)) {
        dst -= stride3;
        __lsx_vstelm_d(p2_out, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(p1_out, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(p0_out, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(q0_out, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(q1_out, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(q2_out, dst, 0, 0);
      } else {
        /* LSB(right) 8 pixel operation */
        DUP4_ARG2(__lsx_vilvl_b, zero, p7, zero, p6, zero, p5, zero, p4, p7_l,
                  p6_l, p5_l, p4_l);
        DUP4_ARG2(__lsx_vilvl_b, zero, q4, zero, q5, zero, q6, zero, q7, q4_l,
                  q5_l, q6_l, q7_l);

        tmp0 = __lsx_vslli_h(p7_l, 3);
        tmp0 = __lsx_vsub_h(tmp0, p7_l);
        tmp0 = __lsx_vadd_h(tmp0, p6_l);
        tmp0 = __lsx_vadd_h(tmp0, q0_l);

        dst = dst_tmp0 - stride3;

        /* calculation of p6 and p5 */
        tmp1 = __lsx_vadd_h(p6_l, p5_l);
        tmp1 = __lsx_vadd_h(tmp1, p4_l);
        tmp1 = __lsx_vadd_h(tmp1, p3_l);
        tmp1 = __lsx_vadd_h(tmp1, p2_l);
        tmp1 = __lsx_vadd_h(tmp1, p1_l);
        tmp1 = __lsx_vadd_h(tmp1, p0_l);
        tmp1 = __lsx_vadd_h(tmp1, tmp0);
        p0_filter16 = __lsx_vsrari_h(tmp1, 4);
        tmp0 = __lsx_vsub_h(p5_l, p6_l);
        tmp0 = __lsx_vadd_h(tmp0, q1_l);
        tmp0 = __lsx_vsub_h(tmp0, p7_l);
        tmp1 = __lsx_vadd_h(tmp1, tmp0);
        p1_filter16 = __lsx_vsrari_h(tmp1, 4);
        DUP2_ARG2(__lsx_vpickev_b, zero, p0_filter16, zero, p1_filter16,
                  p0_filter16, p1_filter16);
        DUP2_ARG3(__lsx_vbitsel_v, p6, p0_filter16, flat2, p5, p1_filter16,
                  flat2, p0_filter16, p1_filter16);
        __lsx_vstelm_d(p0_filter16, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(p1_filter16, dst, 0, 0);
        dst += stride;

        /* calculation of p4 and p3 */
        tmp0 = __lsx_vsub_h(p4_l, p5_l);
        tmp0 = __lsx_vadd_h(tmp0, q2_l);
        tmp0 = __lsx_vsub_h(tmp0, p7_l);
        tmp2 = __lsx_vsub_h(p3_l, p4_l);
        tmp2 = __lsx_vadd_h(tmp2, q3_l);
        tmp2 = __lsx_vsub_h(tmp2, p7_l);
        tmp1 = __lsx_vadd_h(tmp1, tmp0);
        p0_filter16 = __lsx_vsrari_h(tmp1, 4);
        tmp1 = __lsx_vadd_h(tmp1, tmp2);
        p1_filter16 = __lsx_vsrari_h(tmp1, 4);
        DUP2_ARG2(__lsx_vpickev_b, zero, p0_filter16, zero, p1_filter16,
                  p0_filter16, p1_filter16);
        DUP2_ARG3(__lsx_vbitsel_v, p4, p0_filter16, flat2, p3, p1_filter16,
                  flat2, p0_filter16, p1_filter16);
        __lsx_vstelm_d(p0_filter16, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(p1_filter16, dst, 0, 0);
        dst += stride;

        /* calculation of p2 and p1 */
        tmp0 = __lsx_vsub_h(p2_l, p3_l);
        tmp0 = __lsx_vadd_h(tmp0, q4_l);
        tmp0 = __lsx_vsub_h(tmp0, p7_l);
        tmp2 = __lsx_vsub_h(p1_l, p2_l);
        tmp2 = __lsx_vadd_h(tmp2, q5_l);
        tmp2 = __lsx_vsub_h(tmp2, p7_l);
        tmp1 = __lsx_vadd_h(tmp1, tmp0);
        p0_filter16 = __lsx_vsrari_h(tmp1, 4);
        tmp1 = __lsx_vadd_h(tmp1, tmp2);
        p1_filter16 = __lsx_vsrari_h(tmp1, 4);
        DUP2_ARG2(__lsx_vpickev_b, zero, p0_filter16, zero, p1_filter16,
                  p0_filter16, p1_filter16);
        DUP2_ARG3(__lsx_vbitsel_v, p2_out, p0_filter16, flat2, p1_out,
                  p1_filter16, flat2, p0_filter16, p1_filter16);
        __lsx_vstelm_d(p0_filter16, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(p1_filter16, dst, 0, 0);
        dst += stride;

        /* calculation of p0 and q0 */
        tmp0 = __lsx_vsub_h(p0_l, p1_l);
        tmp0 = __lsx_vadd_h(tmp0, q6_l);
        tmp0 = __lsx_vsub_h(tmp0, p7_l);
        tmp2 = __lsx_vsub_h(q7_l, p0_l);
        tmp2 = __lsx_vadd_h(tmp2, q0_l);
        tmp2 = __lsx_vsub_h(tmp2, p7_l);
        tmp1 = __lsx_vadd_h(tmp1, tmp0);
        p0_filter16 = __lsx_vsrari_h(tmp1, 4);
        tmp1 = __lsx_vadd_h(tmp1, tmp2);
        p1_filter16 = __lsx_vsrari_h(tmp1, 4);
        DUP2_ARG2(__lsx_vpickev_b, zero, p0_filter16, zero, p1_filter16,
                  p0_filter16, p1_filter16);
        DUP2_ARG3(__lsx_vbitsel_v, p0_out, p0_filter16, flat2, q0_out,
                  p1_filter16, flat2, p0_filter16, p1_filter16);
        __lsx_vstelm_d(p0_filter16, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(p1_filter16, dst, 0, 0);
        dst += stride;

        /* calculation of q1 and q2 */
        tmp0 = __lsx_vsub_h(q7_l, q0_l);
        tmp0 = __lsx_vadd_h(tmp0, q1_l);
        tmp0 = __lsx_vsub_h(tmp0, p6_l);
        tmp2 = __lsx_vsub_h(q7_l, q1_l);
        tmp2 = __lsx_vadd_h(tmp2, q2_l);
        tmp2 = __lsx_vsub_h(tmp2, p5_l);
        tmp1 = __lsx_vadd_h(tmp1, tmp0);
        p0_filter16 = __lsx_vsrari_h(tmp1, 4);
        tmp1 = __lsx_vadd_h(tmp1, tmp2);
        p1_filter16 = __lsx_vsrari_h(tmp1, 4);
        DUP2_ARG2(__lsx_vpickev_b, zero, p0_filter16, zero, p1_filter16,
                  p0_filter16, p1_filter16);
        DUP2_ARG3(__lsx_vbitsel_v, q1_out, p0_filter16, flat2, q2_out,
                  p1_filter16, flat2, p0_filter16, p1_filter16);
        __lsx_vstelm_d(p0_filter16, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(p1_filter16, dst, 0, 0);
        dst += stride;

        /* calculation of q3 and q4 */
        tmp0 = __lsx_vsub_h(q7_l, q2_l);
        tmp0 = __lsx_vadd_h(tmp0, q3_l);
        tmp0 = __lsx_vsub_h(tmp0, p4_l);
        tmp2 = __lsx_vsub_h(q7_l, q3_l);
        tmp2 = __lsx_vadd_h(tmp2, q4_l);
        tmp2 = __lsx_vsub_h(tmp2, p3_l);
        tmp1 = __lsx_vadd_h(tmp1, tmp0);
        p0_filter16 = __lsx_vsrari_h(tmp1, 4);
        tmp1 = __lsx_vadd_h(tmp1, tmp2);
        p1_filter16 = __lsx_vsrari_h(tmp1, 4);
        DUP2_ARG2(__lsx_vpickev_b, zero, p0_filter16, zero, p1_filter16,
                  p0_filter16, p1_filter16);
        DUP2_ARG3(__lsx_vbitsel_v, q3, p0_filter16, flat2, q4, p1_filter16,
                  flat2, p0_filter16, p1_filter16);
        __lsx_vstelm_d(p0_filter16, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(p1_filter16, dst, 0, 0);
        dst += stride;

        /* calculation of q5 and q6 */
        tmp0 = __lsx_vsub_h(q7_l, q4_l);
        tmp0 = __lsx_vadd_h(tmp0, q5_l);
        tmp0 = __lsx_vsub_h(tmp0, p2_l);
        tmp2 = __lsx_vsub_h(q7_l, q5_l);
        tmp2 = __lsx_vadd_h(tmp2, q6_l);
        tmp2 = __lsx_vsub_h(tmp2, p1_l);
        tmp1 = __lsx_vadd_h(tmp1, tmp0);
        p0_filter16 = __lsx_vsrari_h(tmp1, 4);
        tmp1 = __lsx_vadd_h(tmp1, tmp2);
        p1_filter16 = __lsx_vsrari_h(tmp1, 4);
        DUP2_ARG2(__lsx_vpickev_b, zero, p0_filter16, zero, p1_filter16,
                  p0_filter16, p1_filter16);
        DUP2_ARG3(__lsx_vbitsel_v, q5, p0_filter16, flat2, q6, p1_filter16,
                  flat2, p0_filter16, p1_filter16);
        __lsx_vstelm_d(p0_filter16, dst, 0, 0);
        dst += stride;
        __lsx_vstelm_d(p1_filter16, dst, 0, 0);
      }
    }
  } else {
    mb_lpf_horizontal_edge_dual(dst, stride, b_limit_ptr, limit_ptr,
                                thresh_ptr);
  }
}

void vpx_lpf_horizontal_16_dual_lsx(uint8_t *dst, int32_t stride,
                                    const uint8_t *b_limit_ptr,
                                    const uint8_t *limit_ptr,
                                    const uint8_t *thresh_ptr) {
  mb_lpf_horizontal_edge(dst, stride, b_limit_ptr, limit_ptr, thresh_ptr, 2);
}

static void transpose_16x16(uint8_t *input, int32_t in_stride, uint8_t *output,
                            int32_t out_stride) {
  __m128i row0, row1, row2, row3, row4, row5, row6, row7;
  __m128i row8, row9, row10, row11, row12, row13, row14, row15;
  __m128i tmp0, tmp1, tmp4, tmp5, tmp6, tmp7;
  __m128i tmp2, tmp3;
  __m128i p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7;
  int32_t in_stride2 = in_stride << 1;
  int32_t in_stride3 = in_stride2 + in_stride;
  int32_t in_stride4 = in_stride2 << 1;
  int32_t out_stride2 = out_stride << 1;
  int32_t out_stride3 = out_stride2 + out_stride;
  int32_t out_stride4 = out_stride2 << 1;

  LSX_LD_8(input, in_stride, in_stride2, in_stride3, in_stride4, row0, row1,
           row2, row3, row4, row5, row6, row7);
  input += in_stride4;
  LSX_LD_8(input, in_stride, in_stride2, in_stride3, in_stride4, row8, row9,
           row10, row11, row12, row13, row14, row15);

  LSX_TRANSPOSE16x8_B(row0, row1, row2, row3, row4, row5, row6, row7, row8,
                      row9, row10, row11, row12, row13, row14, row15, p7, p6,
                      p5, p4, p3, p2, p1, p0);

  /* transpose 16x8 matrix into 8x16 */
  /* total 8 intermediate register and 32 instructions */
  q7 = __lsx_vpackod_d(row8, row0);
  q6 = __lsx_vpackod_d(row9, row1);
  q5 = __lsx_vpackod_d(row10, row2);
  q4 = __lsx_vpackod_d(row11, row3);
  q3 = __lsx_vpackod_d(row12, row4);
  q2 = __lsx_vpackod_d(row13, row5);
  q1 = __lsx_vpackod_d(row14, row6);
  q0 = __lsx_vpackod_d(row15, row7);

  DUP2_ARG2(__lsx_vpackev_b, q6, q7, q4, q5, tmp0, tmp1);
  DUP2_ARG2(__lsx_vpackod_b, q6, q7, q4, q5, tmp4, tmp5);

  DUP2_ARG2(__lsx_vpackev_b, q2, q3, q0, q1, q5, q7);
  DUP2_ARG2(__lsx_vpackod_b, q2, q3, q0, q1, tmp6, tmp7);

  DUP2_ARG2(__lsx_vpackev_h, tmp1, tmp0, q7, q5, tmp2, tmp3);
  q0 = __lsx_vpackev_w(tmp3, tmp2);
  q4 = __lsx_vpackod_w(tmp3, tmp2);

  tmp2 = __lsx_vpackod_h(tmp1, tmp0);
  tmp3 = __lsx_vpackod_h(q7, q5);
  q2 = __lsx_vpackev_w(tmp3, tmp2);
  q6 = __lsx_vpackod_w(tmp3, tmp2);

  DUP2_ARG2(__lsx_vpackev_h, tmp5, tmp4, tmp7, tmp6, tmp2, tmp3);
  q1 = __lsx_vpackev_w(tmp3, tmp2);
  q5 = __lsx_vpackod_w(tmp3, tmp2);

  tmp2 = __lsx_vpackod_h(tmp5, tmp4);
  tmp3 = __lsx_vpackod_h(tmp7, tmp6);
  q3 = __lsx_vpackev_w(tmp3, tmp2);
  q7 = __lsx_vpackod_w(tmp3, tmp2);

  LSX_ST_8(p7, p6, p5, p4, p3, p2, p1, p0, output, out_stride, out_stride2,
           out_stride3, out_stride4);
  output += out_stride4;
  LSX_ST_8(q0, q1, q2, q3, q4, q5, q6, q7, output, out_stride, out_stride2,
           out_stride3, out_stride4);
}

static int32_t vt_lpf_t4_and_t8_16w(uint8_t *dst, uint8_t *filter48,
                                    uint8_t *dst_org, int32_t stride,
                                    const uint8_t *b_limit_ptr,
                                    const uint8_t *limit_ptr,
                                    const uint8_t *thresh_ptr) {
  int32_t stride2 = stride << 1;
  int32_t stride3 = stride2 + stride;
  int32_t stride4 = stride2 << 1;
  __m128i p3, p2, p1, p0, q3, q2, q1, q0;
  __m128i p2_out, p1_out, p0_out, q0_out, q1_out, q2_out;
  __m128i flat, mask, hev, thresh, b_limit, limit;
  __m128i p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l;
  __m128i p3_h, p2_h, p1_h, p0_h, q0_h, q1_h, q2_h, q3_h;
  __m128i p2_filt8_l, p1_filt8_l, p0_filt8_l;
  __m128i q0_filt8_l, q1_filt8_l, q2_filt8_l;
  __m128i p2_filt8_h, p1_filt8_h, p0_filt8_h;
  __m128i q0_filt8_h, q1_filt8_h, q2_filt8_h;
  __m128i vec0, vec1, vec2, vec3, vec4, vec5;

  /* load vector elements */
  DUP4_ARG2(__lsx_vld, dst, -64, dst, -48, dst, -32, dst, -16, p3, p2, p1, p0);
  DUP4_ARG2(__lsx_vld, dst, 0, dst, 16, dst, 32, dst, 48, q0, q1, q2, q3);

  thresh = __lsx_vldrepl_b(thresh_ptr, 0);
  b_limit = __lsx_vldrepl_b(b_limit_ptr, 0);
  limit = __lsx_vldrepl_b(limit_ptr, 0);

  /* mask and hev */
  LPF_MASK_HEV(p3, p2, p1, p0, q0, q1, q2, q3, limit, b_limit, thresh, hev,
               mask, flat);
  /* flat4 */
  VP9_FLAT4(p3, p2, p0, q0, q2, q3, flat);
  /* filter4 */
  VP9_LPF_FILTER4_4W(p1, p0, q0, q1, mask, hev, p1_out, p0_out, q0_out, q1_out);

  /* if flat is zero for all pixels, then no need to calculate other filter */
  if (__lsx_bz_v(flat)) {
    DUP2_ARG2(__lsx_vilvl_b, p0_out, p1_out, q1_out, q0_out, vec0, vec1);
    vec2 = __lsx_vilvl_h(vec1, vec0);
    vec3 = __lsx_vilvh_h(vec1, vec0);
    DUP2_ARG2(__lsx_vilvh_b, p0_out, p1_out, q1_out, q0_out, vec0, vec1);
    vec4 = __lsx_vilvl_h(vec1, vec0);
    vec5 = __lsx_vilvh_h(vec1, vec0);

    dst_org -= 2;
    __lsx_vstelm_w(vec2, dst_org, 0, 0);
    __lsx_vstelm_w(vec2, dst_org + stride, 0, 1);
    __lsx_vstelm_w(vec2, dst_org + stride2, 0, 2);
    __lsx_vstelm_w(vec2, dst_org + stride3, 0, 3);
    dst_org += stride4;
    __lsx_vstelm_w(vec3, dst_org, 0, 0);
    __lsx_vstelm_w(vec3, dst_org + stride, 0, 1);
    __lsx_vstelm_w(vec3, dst_org + stride2, 0, 2);
    __lsx_vstelm_w(vec3, dst_org + stride3, 0, 3);
    dst_org += stride4;
    __lsx_vstelm_w(vec4, dst_org, 0, 0);
    __lsx_vstelm_w(vec4, dst_org + stride, 0, 1);
    __lsx_vstelm_w(vec4, dst_org + stride2, 0, 2);
    __lsx_vstelm_w(vec4, dst_org + stride3, 0, 3);
    dst_org += stride4;
    __lsx_vstelm_w(vec5, dst_org, 0, 0);
    __lsx_vstelm_w(vec5, dst_org + stride, 0, 1);
    __lsx_vstelm_w(vec5, dst_org + stride2, 0, 2);
    __lsx_vstelm_w(vec5, dst_org + stride3, 0, 3);

    return 1;
  }

  DUP4_ARG2(__lsx_vsllwil_hu_bu, p3, 0, p2, 0, p1, 0, p0, 0, p3_l, p2_l, p1_l,
            p0_l);
  DUP4_ARG2(__lsx_vsllwil_hu_bu, q0, 0, q1, 0, q2, 0, q3, 0, q0_l, q1_l, q2_l,
            q3_l);
  VP9_FILTER8(p3_l, p2_l, p1_l, p0_l, q0_l, q1_l, q2_l, q3_l, p2_filt8_l,
              p1_filt8_l, p0_filt8_l, q0_filt8_l, q1_filt8_l, q2_filt8_l);
  DUP4_ARG1(__lsx_vexth_hu_bu, p3, p2, p1, p0, p3_h, p2_h, p1_h, p0_h);
  DUP4_ARG1(__lsx_vexth_hu_bu, q0, q1, q2, q3, q0_h, q1_h, q2_h, q3_h);
  VP9_FILTER8(p3_h, p2_h, p1_h, p0_h, q0_h, q1_h, q2_h, q3_h, p2_filt8_h,
              p1_filt8_h, p0_filt8_h, q0_filt8_h, q1_filt8_h, q2_filt8_h);

  /* convert 16 bit output data into 8 bit */
  DUP4_ARG2(__lsx_vpickev_b, p2_filt8_h, p2_filt8_l, p1_filt8_h, p1_filt8_l,
            p0_filt8_h, p0_filt8_l, q0_filt8_h, q0_filt8_l, p2_filt8_l,
            p1_filt8_l, p0_filt8_l, q0_filt8_l);
  DUP2_ARG2(__lsx_vpickev_b, q1_filt8_h, q1_filt8_l, q2_filt8_h, q2_filt8_l,
            q1_filt8_l, q2_filt8_l);

  /* store pixel values */
  p2_out = __lsx_vbitsel_v(p2, p2_filt8_l, flat);
  p1_out = __lsx_vbitsel_v(p1_out, p1_filt8_l, flat);
  p0_out = __lsx_vbitsel_v(p0_out, p0_filt8_l, flat);
  q0_out = __lsx_vbitsel_v(q0_out, q0_filt8_l, flat);
  q1_out = __lsx_vbitsel_v(q1_out, q1_filt8_l, flat);
  q2_out = __lsx_vbitsel_v(q2, q2_filt8_l, flat);

  __lsx_vst(p2_out, filter48, 0);
  __lsx_vst(p1_out, filter48, 16);
  __lsx_vst(p0_out, filter48, 32);
  __lsx_vst(q0_out, filter48, 48);
  __lsx_vst(q1_out, filter48, 64);
  __lsx_vst(q2_out, filter48, 80);
  __lsx_vst(flat, filter48, 96);

  return 0;
}

static int32_t vt_lpf_t16_16w(uint8_t *dst, uint8_t *dst_org, int32_t stride,
                              uint8_t *filter48) {
  __m128i flat, flat2, filter8;
  __m128i p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7;
  __m128i out_l, out_h;
  v8u16 p7_l_in, p6_l_in, p5_l_in, p4_l_in;
  v8u16 p3_l_in, p2_l_in, p1_l_in, p0_l_in;
  v8u16 q7_l_in, q6_l_in, q5_l_in, q4_l_in;
  v8u16 q3_l_in, q2_l_in, q1_l_in, q0_l_in;
  v8u16 p7_h_in, p6_h_in, p5_h_in, p4_h_in;
  v8u16 p3_h_in, p2_h_in, p1_h_in, p0_h_in;
  v8u16 q7_h_in, q6_h_in, q5_h_in, q4_h_in;
  v8u16 q3_h_in, q2_h_in, q1_h_in, q0_h_in;
  v8u16 tmp0_l, tmp1_l, tmp0_h, tmp1_h;
  uint8_t *dst_tmp = dst - 128;

  flat = __lsx_vld(filter48, 96);

  DUP4_ARG2(__lsx_vld, dst_tmp, 0, dst_tmp, 16, dst_tmp, 32, dst_tmp, 48, p7,
            p6, p5, p4);
  DUP4_ARG2(__lsx_vld, dst_tmp, 64, dst_tmp, 80, dst_tmp, 96, dst_tmp, 112, p3,
            p2, p1, p0);
  DUP4_ARG2(__lsx_vld, dst, 0, dst, 16, dst, 32, dst, 48, q0, q1, q2, q3);
  DUP4_ARG2(__lsx_vld, dst, 64, dst, 80, dst, 96, dst, 112, q4, q5, q6, q7);

  VP9_FLAT5(p7, p6, p5, p4, p0, q0, q4, q5, q6, q7, flat, flat2);
  /* if flat2 is zero for all pixels, then no need to calculate other filter */
  if (__lsx_bz_v(flat2)) {
    __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;

    DUP4_ARG2(__lsx_vld, filter48, 0, filter48, 16, filter48, 32, filter48, 48,
              p2, p1, p0, q0);
    DUP2_ARG2(__lsx_vld, filter48, 64, filter48, 80, q1, q2);

    DUP2_ARG2(__lsx_vilvl_b, p1, p2, q0, p0, vec0, vec1);
    vec3 = __lsx_vilvl_h(vec1, vec0);
    vec4 = __lsx_vilvh_h(vec1, vec0);
    DUP2_ARG2(__lsx_vilvh_b, p1, p2, q0, p0, vec0, vec1);
    vec6 = __lsx_vilvl_h(vec1, vec0);
    vec7 = __lsx_vilvh_h(vec1, vec0);
    vec2 = __lsx_vilvl_b(q2, q1);
    vec5 = __lsx_vilvh_b(q2, q1);

    dst_org -= 3;
    __lsx_vstelm_w(vec3, dst_org, 0, 0);
    __lsx_vstelm_h(vec2, dst_org, 4, 0);
    dst_org += stride;
    __lsx_vstelm_w(vec3, dst_org, 0, 1);
    __lsx_vstelm_h(vec2, dst_org, 4, 1);
    dst_org += stride;
    __lsx_vstelm_w(vec3, dst_org, 0, 2);
    __lsx_vstelm_h(vec2, dst_org, 4, 2);
    dst_org += stride;
    __lsx_vstelm_w(vec3, dst_org, 0, 3);
    __lsx_vstelm_h(vec2, dst_org, 4, 3);
    dst_org += stride;
    __lsx_vstelm_w(vec4, dst_org, 0, 0);
    __lsx_vstelm_h(vec2, dst_org, 4, 4);
    dst_org += stride;
    __lsx_vstelm_w(vec4, dst_org, 0, 1);
    __lsx_vstelm_h(vec2, dst_org, 4, 5);
    dst_org += stride;
    __lsx_vstelm_w(vec4, dst_org, 0, 2);
    __lsx_vstelm_h(vec2, dst_org, 4, 6);
    dst_org += stride;
    __lsx_vstelm_w(vec4, dst_org, 0, 3);
    __lsx_vstelm_h(vec2, dst_org, 4, 7);
    dst_org += stride;
    __lsx_vstelm_w(vec6, dst_org, 0, 0);
    __lsx_vstelm_h(vec5, dst_org, 4, 0);
    dst_org += stride;
    __lsx_vstelm_w(vec6, dst_org, 0, 1);
    __lsx_vstelm_h(vec5, dst_org, 4, 1);
    dst_org += stride;
    __lsx_vstelm_w(vec6, dst_org, 0, 2);
    __lsx_vstelm_h(vec5, dst_org, 4, 2);
    dst_org += stride;
    __lsx_vstelm_w(vec6, dst_org, 0, 3);
    __lsx_vstelm_h(vec5, dst_org, 4, 3);
    dst_org += stride;
    __lsx_vstelm_w(vec7, dst_org, 0, 0);
    __lsx_vstelm_h(vec5, dst_org, 4, 4);
    dst_org += stride;
    __lsx_vstelm_w(vec7, dst_org, 0, 1);
    __lsx_vstelm_h(vec5, dst_org, 4, 5);
    dst_org += stride;
    __lsx_vstelm_w(vec7, dst_org, 0, 2);
    __lsx_vstelm_h(vec5, dst_org, 4, 6);
    dst_org += stride;
    __lsx_vstelm_w(vec7, dst_org, 0, 3);
    __lsx_vstelm_h(vec5, dst_org, 4, 7);

    return 1;
  }

  dst -= 7 * 16;

  p7_l_in = (v8u16)__lsx_vsllwil_hu_bu(p7, 0);
  p6_l_in = (v8u16)__lsx_vsllwil_hu_bu(p6, 0);
  p5_l_in = (v8u16)__lsx_vsllwil_hu_bu(p5, 0);
  p4_l_in = (v8u16)__lsx_vsllwil_hu_bu(p4, 0);
  p3_l_in = (v8u16)__lsx_vsllwil_hu_bu(p3, 0);
  p2_l_in = (v8u16)__lsx_vsllwil_hu_bu(p2, 0);
  p1_l_in = (v8u16)__lsx_vsllwil_hu_bu(p1, 0);
  p0_l_in = (v8u16)__lsx_vsllwil_hu_bu(p0, 0);
  q0_l_in = (v8u16)__lsx_vsllwil_hu_bu(q0, 0);

  tmp0_l = p7_l_in << 3;
  tmp0_l -= p7_l_in;
  tmp0_l += p6_l_in;
  tmp0_l += q0_l_in;
  tmp1_l = p6_l_in + p5_l_in;
  tmp1_l += p4_l_in;
  tmp1_l += p3_l_in;
  tmp1_l += p2_l_in;
  tmp1_l += p1_l_in;
  tmp1_l += p0_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  p7_h_in = (v8u16)__lsx_vexth_hu_bu(p7);
  p6_h_in = (v8u16)__lsx_vexth_hu_bu(p6);
  p5_h_in = (v8u16)__lsx_vexth_hu_bu(p5);
  p4_h_in = (v8u16)__lsx_vexth_hu_bu(p4);
  p3_h_in = (v8u16)__lsx_vexth_hu_bu(p3);
  p2_h_in = (v8u16)__lsx_vexth_hu_bu(p2);
  p1_h_in = (v8u16)__lsx_vexth_hu_bu(p1);
  p0_h_in = (v8u16)__lsx_vexth_hu_bu(p0);
  q0_h_in = (v8u16)__lsx_vexth_hu_bu(q0);

  tmp0_h = p7_h_in << 3;
  tmp0_h -= p7_h_in;
  tmp0_h += p6_h_in;
  tmp0_h += q0_h_in;
  tmp1_h = p6_h_in + p5_h_in;
  tmp1_h += p4_h_in;
  tmp1_h += p3_h_in;
  tmp1_h += p2_h_in;
  tmp1_h += p1_h_in;
  tmp1_h += p0_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);

  out_l = __lsx_vpickev_b(out_h, out_l);
  p6 = __lsx_vbitsel_v(p6, out_l, flat2);
  __lsx_vst(p6, dst, 0);

  /* p5 */
  q1_l_in = (v8u16)__lsx_vsllwil_hu_bu(q1, 0);
  tmp0_l = p5_l_in - p6_l_in;
  tmp0_l += q1_l_in;
  tmp0_l -= p7_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  q1_h_in = (v8u16)__lsx_vexth_hu_bu(q1);
  tmp0_h = p5_h_in - p6_h_in;
  tmp0_h += q1_h_in;
  tmp0_h -= p7_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  p5 = __lsx_vbitsel_v(p5, out_l, flat2);
  __lsx_vst(p5, dst, 16);

  /* p4 */
  q2_l_in = (v8u16)__lsx_vsllwil_hu_bu(q2, 0);
  tmp0_l = p4_l_in - p5_l_in;
  tmp0_l += q2_l_in;
  tmp0_l -= p7_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  q2_h_in = (v8u16)__lsx_vexth_hu_bu(q2);
  tmp0_h = p4_h_in - p5_h_in;
  tmp0_h += q2_h_in;
  tmp0_h -= p7_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  p4 = __lsx_vbitsel_v(p4, out_l, flat2);
  __lsx_vst(p4, dst, 16 * 2);

  /* p3 */
  q3_l_in = (v8u16)__lsx_vsllwil_hu_bu(q3, 0);
  tmp0_l = p3_l_in - p4_l_in;
  tmp0_l += q3_l_in;
  tmp0_l -= p7_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  q3_h_in = (v8u16)__lsx_vexth_hu_bu(q3);
  tmp0_h = p3_h_in - p4_h_in;
  tmp0_h += q3_h_in;
  tmp0_h -= p7_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  p3 = __lsx_vbitsel_v(p3, out_l, flat2);
  __lsx_vst(p3, dst, 16 * 3);

  /* p2 */
  q4_l_in = (v8u16)__lsx_vsllwil_hu_bu(q4, 0);
  filter8 = __lsx_vld(filter48, 0);
  tmp0_l = p2_l_in - p3_l_in;
  tmp0_l += q4_l_in;
  tmp0_l -= p7_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  q4_h_in = (v8u16)__lsx_vexth_hu_bu(q4);
  tmp0_h = p2_h_in - p3_h_in;
  tmp0_h += q4_h_in;
  tmp0_h -= p7_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
  __lsx_vst(filter8, dst, 16 * 4);

  /* p1 */
  q5_l_in = (v8u16)__lsx_vsllwil_hu_bu(q5, 0);
  filter8 = __lsx_vld(filter48, 16);
  tmp0_l = p1_l_in - p2_l_in;
  tmp0_l += q5_l_in;
  tmp0_l -= p7_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  q5_h_in = (v8u16)__lsx_vexth_hu_bu(q5);
  tmp0_h = p1_h_in - p2_h_in;
  tmp0_h += q5_h_in;
  tmp0_h -= p7_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)(tmp1_h), 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
  __lsx_vst(filter8, dst, 16 * 5);

  /* p0 */
  q6_l_in = (v8u16)__lsx_vsllwil_hu_bu(q6, 0);
  filter8 = __lsx_vld(filter48, 32);
  tmp0_l = p0_l_in - p1_l_in;
  tmp0_l += q6_l_in;
  tmp0_l -= p7_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  q6_h_in = (v8u16)__lsx_vexth_hu_bu(q6);
  tmp0_h = p0_h_in - p1_h_in;
  tmp0_h += q6_h_in;
  tmp0_h -= p7_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
  __lsx_vst(filter8, dst, 16 * 6);

  /* q0 */
  q7_l_in = (v8u16)__lsx_vsllwil_hu_bu(q7, 0);
  filter8 = __lsx_vld(filter48, 48);
  tmp0_l = q7_l_in - p0_l_in;
  tmp0_l += q0_l_in;
  tmp0_l -= p7_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  q7_h_in = (v8u16)__lsx_vexth_hu_bu(q7);
  tmp0_h = q7_h_in - p0_h_in;
  tmp0_h += q0_h_in;
  tmp0_h -= p7_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
  __lsx_vst(filter8, dst, 16 * 7);

  /* q1 */
  filter8 = __lsx_vld(filter48, 64);
  tmp0_l = q7_l_in - q0_l_in;
  tmp0_l += q1_l_in;
  tmp0_l -= p6_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  tmp0_h = q7_h_in - q0_h_in;
  tmp0_h += q1_h_in;
  tmp0_h -= p6_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
  __lsx_vst(filter8, dst, 16 * 8);

  /* q2 */
  filter8 = __lsx_vld(filter48, 80);
  tmp0_l = q7_l_in - q1_l_in;
  tmp0_l += q2_l_in;
  tmp0_l -= p5_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  tmp0_h = q7_h_in - q1_h_in;
  tmp0_h += q2_h_in;
  tmp0_h -= p5_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  filter8 = __lsx_vbitsel_v(filter8, out_l, flat2);
  __lsx_vst(filter8, dst, 16 * 9);

  /* q3 */
  tmp0_l = q7_l_in - q2_l_in;
  tmp0_l += q3_l_in;
  tmp0_l -= p4_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  tmp0_h = q7_h_in - q2_h_in;
  tmp0_h += q3_h_in;
  tmp0_h -= p4_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  q3 = __lsx_vbitsel_v(q3, out_l, flat2);
  __lsx_vst(q3, dst, 16 * 10);

  /* q4 */
  tmp0_l = q7_l_in - q3_l_in;
  tmp0_l += q4_l_in;
  tmp0_l -= p3_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  tmp0_h = q7_h_in - q3_h_in;
  tmp0_h += q4_h_in;
  tmp0_h -= p3_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  q4 = __lsx_vbitsel_v(q4, out_l, flat2);
  __lsx_vst(q4, dst, 16 * 11);

  /* q5 */
  tmp0_l = q7_l_in - q4_l_in;
  tmp0_l += q5_l_in;
  tmp0_l -= p2_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  tmp0_h = q7_h_in - q4_h_in;
  tmp0_h += q5_h_in;
  tmp0_h -= p2_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  q5 = __lsx_vbitsel_v(q5, out_l, flat2);
  __lsx_vst(q5, dst, 16 * 12);

  /* q6 */
  tmp0_l = q7_l_in - q5_l_in;
  tmp0_l += q6_l_in;
  tmp0_l -= p1_l_in;
  tmp1_l += tmp0_l;
  out_l = __lsx_vsrari_h((__m128i)tmp1_l, 4);
  tmp0_h = q7_h_in - q5_h_in;
  tmp0_h += q6_h_in;
  tmp0_h -= p1_h_in;
  tmp1_h += tmp0_h;
  out_h = __lsx_vsrari_h((__m128i)tmp1_h, 4);
  out_l = __lsx_vpickev_b(out_h, out_l);
  q6 = __lsx_vbitsel_v(q6, out_l, flat2);
  __lsx_vst(q6, dst, 16 * 13);

  return 0;
}

void vpx_lpf_vertical_16_dual_lsx(uint8_t *src, int32_t pitch,
                                  const uint8_t *b_limit_ptr,
                                  const uint8_t *limit_ptr,
                                  const uint8_t *thresh_ptr) {
  uint8_t early_exit = 0;
  DECLARE_ALIGNED(16, uint8_t, transposed_input[16 * 24]);
  uint8_t *filter48 = &transposed_input[16 * 16];

  transpose_16x16((src - 8), pitch, &transposed_input[0], 16);

  early_exit =
      vt_lpf_t4_and_t8_16w((transposed_input + 16 * 8), &filter48[0], src,
                           pitch, b_limit_ptr, limit_ptr, thresh_ptr);

  if (early_exit == 0) {
    early_exit =
        vt_lpf_t16_16w((transposed_input + 16 * 8), src, pitch, &filter48[0]);

    if (early_exit == 0) {
      transpose_16x16(transposed_input, 16, (src - 8), pitch);
    }
  }
}
