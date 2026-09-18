/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_util/loongson_intrinsics.h"

static INLINE __m128i sad_ub2_uh(__m128i in0, __m128i in1, __m128i ref0,
                                 __m128i ref1) {
  __m128i diff0_m, diff1_m, sad_m0;
  __m128i sad_m = __lsx_vldi(0);

  diff0_m = __lsx_vabsd_bu(in0, ref0);
  diff1_m = __lsx_vabsd_bu(in1, ref1);

  sad_m0 = __lsx_vhaddw_hu_bu(diff0_m, diff0_m);
  sad_m = __lsx_vadd_h(sad_m, sad_m0);
  sad_m0 = __lsx_vhaddw_hu_bu(diff1_m, diff1_m);
  sad_m = __lsx_vadd_h(sad_m, sad_m0);

  return sad_m;
}

static INLINE uint32_t hadd_uw_u32(__m128i in) {
  __m128i res0_m;
  uint32_t sum_m;

  res0_m = __lsx_vhaddw_du_wu(in, in);
  res0_m = __lsx_vhaddw_qu_du(res0_m, res0_m);
  sum_m = __lsx_vpickve2gr_w(res0_m, 0);

  return sum_m;
}

static INLINE uint32_t hadd_uh_u32(__m128i in) {
  __m128i res_m;
  uint32_t sum_m;

  res_m = __lsx_vhaddw_wu_hu(in, in);
  sum_m = hadd_uw_u32(res_m);

  return sum_m;
}

static INLINE int32_t hadd_sw_s32(__m128i in) {
  __m128i res0_m;
  int32_t sum_m;

  res0_m = __lsx_vhaddw_d_w(in, in);
  res0_m = __lsx_vhaddw_q_d(res0_m, res0_m);
  sum_m = __lsx_vpickve2gr_w(res0_m, 0);

  return sum_m;
}

static uint32_t sad_8width_lsx(const uint8_t *src, int32_t src_stride,
                               const uint8_t *ref, int32_t ref_stride,
                               int32_t height) {
  int32_t ht_cnt;
  uint32_t res;
  __m128i src0, src1, src2, src3, ref0, ref1, ref2, ref3, sad_tmp;
  __m128i sad = __lsx_vldi(0);

  for (ht_cnt = (height >> 2); ht_cnt--;) {
    DUP2_ARG2(__lsx_vld, src, 0, ref, 0, src0, ref0);
    src += src_stride;
    ref += ref_stride;
    DUP2_ARG2(__lsx_vld, src, 0, ref, 0, src1, ref1);
    src += src_stride;
    ref += ref_stride;
    DUP2_ARG2(__lsx_vld, src, 0, ref, 0, src2, ref2);
    src += src_stride;
    ref += ref_stride;
    DUP2_ARG2(__lsx_vld, src, 0, ref, 0, src3, ref3);
    src += src_stride;
    ref += ref_stride;
    DUP4_ARG2(__lsx_vpickev_d, src1, src0, src3, src2, ref1, ref0, ref3, ref2,
              src0, src1, ref0, ref1);
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad = __lsx_vadd_h(sad, sad_tmp);
  }
  res = hadd_uh_u32(sad);
  return res;
}

static uint32_t sad_16width_lsx(const uint8_t *src, int32_t src_stride,
                                const uint8_t *ref, int32_t ref_stride,
                                int32_t height) {
  int32_t ht_cnt = (height >> 2);
  uint32_t res;
  __m128i src0, src1, ref0, ref1, sad_tmp;
  __m128i sad = __lsx_vldi(0);
  int32_t src_stride2 = src_stride << 1;
  int32_t ref_stride2 = ref_stride << 1;

  for (; ht_cnt--;) {
    DUP2_ARG2(__lsx_vld, src, 0, ref, 0, src0, ref0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, ref, ref_stride, src1, ref1);
    src += src_stride2;
    ref += ref_stride2;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad = __lsx_vadd_h(sad, sad_tmp);

    DUP2_ARG2(__lsx_vld, src, 0, ref, 0, src0, ref0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, ref, ref_stride, src1, ref1);
    src += src_stride2;
    ref += ref_stride2;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad = __lsx_vadd_h(sad, sad_tmp);
  }

  res = hadd_uh_u32(sad);
  return res;
}

static uint32_t sad_32width_lsx(const uint8_t *src, int32_t src_stride,
                                const uint8_t *ref, int32_t ref_stride,
                                int32_t height) {
  int32_t ht_cnt = (height >> 2);
  uint32_t res;
  __m128i src0, src1, ref0, ref1;
  __m128i sad_tmp;
  __m128i sad = __lsx_vldi(0);

  for (; ht_cnt--;) {
    DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src1);
    src += src_stride;
    DUP2_ARG2(__lsx_vld, ref, 0, ref, 16, ref0, ref1);
    ref += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad = __lsx_vadd_h(sad, sad_tmp);

    DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src1);
    src += src_stride;
    DUP2_ARG2(__lsx_vld, ref, 0, ref, 16, ref0, ref1);
    ref += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad = __lsx_vadd_h(sad, sad_tmp);

    DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src1);
    src += src_stride;
    DUP2_ARG2(__lsx_vld, ref, 0, ref, 16, ref0, ref1);
    ref += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad = __lsx_vadd_h(sad, sad_tmp);

    DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src1);
    src += src_stride;
    DUP2_ARG2(__lsx_vld, ref, 0, ref, 16, ref0, ref1);
    ref += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad = __lsx_vadd_h(sad, sad_tmp);
  }
  res = hadd_uh_u32(sad);
  return res;
}

static uint32_t sad_64width_lsx(const uint8_t *src, int32_t src_stride,
                                const uint8_t *ref, int32_t ref_stride,
                                int32_t height) {
  int32_t ht_cnt = (height >> 1);
  uint32_t sad = 0;
  __m128i src0, src1, src2, src3;
  __m128i ref0, ref1, ref2, ref3;
  __m128i sad_tmp;
  __m128i sad0 = __lsx_vldi(0);
  __m128i sad1 = sad0;

  for (; ht_cnt--;) {
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    src += src_stride;
    DUP4_ARG2(__lsx_vld, ref, 0, ref, 16, ref, 32, ref, 48, ref0, ref1, ref2,
              ref3);
    ref += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, ref2, ref3);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);

    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    src += src_stride;
    DUP4_ARG2(__lsx_vld, ref, 0, ref, 16, ref, 32, ref, 48, ref0, ref1, ref2,
              ref3);
    ref += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, ref2, ref3);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);
  }

  sad = hadd_uh_u32(sad0);
  sad += hadd_uh_u32(sad1);

  return sad;
}

static void sad_8width_x4d_lsx(const uint8_t *src_ptr, int32_t src_stride,
                               const uint8_t *const aref_ptr[],
                               int32_t ref_stride, int32_t height,
                               uint32_t *sad_array) {
  int32_t ht_cnt = (height >> 2);
  const uint8_t *ref0_ptr, *ref1_ptr, *ref2_ptr, *ref3_ptr;
  __m128i src0, src1, src2, src3, sad_tmp;
  __m128i ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7;
  __m128i ref8, ref9, ref10, ref11, ref12, ref13, ref14, ref15;
  __m128i sad0 = __lsx_vldi(0);
  __m128i sad1 = sad0;
  __m128i sad2 = sad0;
  __m128i sad3 = sad0;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t ref_stride2 = ref_stride << 1;
  int32_t ref_stride3 = ref_stride2 + ref_stride;
  int32_t ref_stride4 = ref_stride2 << 1;

  ref0_ptr = aref_ptr[0];
  ref1_ptr = aref_ptr[1];
  ref2_ptr = aref_ptr[2];
  ref3_ptr = aref_ptr[3];

  for (; ht_cnt--;) {
    src0 = __lsx_vld(src_ptr, 0);
    DUP2_ARG2(__lsx_vldx, src_ptr, src_stride, src_ptr, src_stride2, src1,
              src2);
    src3 = __lsx_vldx(src_ptr, src_stride3);
    src_ptr += src_stride4;
    ref0 = __lsx_vld(ref0_ptr, 0);
    DUP2_ARG2(__lsx_vldx, ref0_ptr, ref_stride, ref0_ptr, ref_stride2, ref1,
              ref2);
    ref3 = __lsx_vldx(ref0_ptr, ref_stride3);
    ref0_ptr += ref_stride4;
    ref4 = __lsx_vld(ref1_ptr, 0);
    DUP2_ARG2(__lsx_vldx, ref1_ptr, ref_stride, ref1_ptr, ref_stride2, ref5,
              ref6);
    ref7 = __lsx_vldx(ref1_ptr, ref_stride3);
    ref1_ptr += ref_stride4;
    ref8 = __lsx_vld(ref2_ptr, 0);
    DUP2_ARG2(__lsx_vldx, ref2_ptr, ref_stride, ref2_ptr, ref_stride2, ref9,
              ref10);
    ref11 = __lsx_vldx(ref2_ptr, ref_stride3);
    ref2_ptr += ref_stride4;
    ref12 = __lsx_vld(ref3_ptr, 0);
    DUP2_ARG2(__lsx_vldx, ref3_ptr, ref_stride, ref3_ptr, ref_stride2, ref13,
              ref14);
    ref15 = __lsx_vldx(ref3_ptr, ref_stride3);
    ref3_ptr += ref_stride4;

    DUP2_ARG2(__lsx_vpickev_d, src1, src0, src3, src2, src0, src1);
    DUP2_ARG2(__lsx_vpickev_d, ref1, ref0, ref3, ref2, ref0, ref1);
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);

    DUP2_ARG2(__lsx_vpickev_d, ref5, ref4, ref7, ref6, ref0, ref1);
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);

    DUP2_ARG2(__lsx_vpickev_d, ref9, ref8, ref11, ref10, ref0, ref1);
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad2 = __lsx_vadd_h(sad2, sad_tmp);

    DUP2_ARG2(__lsx_vpickev_d, ref13, ref12, ref15, ref14, ref0, ref1);
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad3 = __lsx_vadd_h(sad3, sad_tmp);
  }
  sad_array[0] = hadd_uh_u32(sad0);
  sad_array[1] = hadd_uh_u32(sad1);
  sad_array[2] = hadd_uh_u32(sad2);
  sad_array[3] = hadd_uh_u32(sad3);
}

static void sad_16width_x4d_lsx(const uint8_t *src_ptr, int32_t src_stride,
                                const uint8_t *const aref_ptr[],
                                int32_t ref_stride, int32_t height,
                                uint32_t *sad_array) {
  int32_t ht_cnt = (height >> 1);
  const uint8_t *ref0_ptr, *ref1_ptr, *ref2_ptr, *ref3_ptr;
  __m128i src, ref0, ref1, ref2, ref3, diff, sad_tmp;
  __m128i sad0 = __lsx_vldi(0);
  __m128i sad1 = sad0;
  __m128i sad2 = sad0;
  __m128i sad3 = sad0;

  ref0_ptr = aref_ptr[0];
  ref1_ptr = aref_ptr[1];
  ref2_ptr = aref_ptr[2];
  ref3_ptr = aref_ptr[3];

  for (; ht_cnt--;) {
    src = __lsx_vld(src_ptr, 0);
    src_ptr += src_stride;
    ref0 = __lsx_vld(ref0_ptr, 0);
    ref0_ptr += ref_stride;
    ref1 = __lsx_vld(ref1_ptr, 0);
    ref1_ptr += ref_stride;
    ref2 = __lsx_vld(ref2_ptr, 0);
    ref2_ptr += ref_stride;
    ref3 = __lsx_vld(ref3_ptr, 0);
    ref3_ptr += ref_stride;

    diff = __lsx_vabsd_bu(src, ref0);
    sad_tmp = __lsx_vhaddw_hu_bu(diff, diff);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);
    diff = __lsx_vabsd_bu(src, ref1);
    sad_tmp = __lsx_vhaddw_hu_bu(diff, diff);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);
    diff = __lsx_vabsd_bu(src, ref2);
    sad_tmp = __lsx_vhaddw_hu_bu(diff, diff);
    sad2 = __lsx_vadd_h(sad2, sad_tmp);
    diff = __lsx_vabsd_bu(src, ref3);
    sad_tmp = __lsx_vhaddw_hu_bu(diff, diff);
    sad3 = __lsx_vadd_h(sad3, sad_tmp);

    src = __lsx_vld(src_ptr, 0);
    src_ptr += src_stride;
    ref0 = __lsx_vld(ref0_ptr, 0);
    ref0_ptr += ref_stride;
    ref1 = __lsx_vld(ref1_ptr, 0);
    ref1_ptr += ref_stride;
    ref2 = __lsx_vld(ref2_ptr, 0);
    ref2_ptr += ref_stride;
    ref3 = __lsx_vld(ref3_ptr, 0);
    ref3_ptr += ref_stride;

    diff = __lsx_vabsd_bu(src, ref0);
    sad_tmp = __lsx_vhaddw_hu_bu(diff, diff);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);
    diff = __lsx_vabsd_bu(src, ref1);
    sad_tmp = __lsx_vhaddw_hu_bu(diff, diff);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);
    diff = __lsx_vabsd_bu(src, ref2);
    sad_tmp = __lsx_vhaddw_hu_bu(diff, diff);
    sad2 = __lsx_vadd_h(sad2, sad_tmp);
    diff = __lsx_vabsd_bu(src, ref3);
    sad_tmp = __lsx_vhaddw_hu_bu(diff, diff);
    sad3 = __lsx_vadd_h(sad3, sad_tmp);
  }
  sad_array[0] = hadd_uh_u32(sad0);
  sad_array[1] = hadd_uh_u32(sad1);
  sad_array[2] = hadd_uh_u32(sad2);
  sad_array[3] = hadd_uh_u32(sad3);
}

static void sad_32width_x4d_lsx(const uint8_t *src, int32_t src_stride,
                                const uint8_t *const aref_ptr[],
                                int32_t ref_stride, int32_t height,
                                uint32_t *sad_array) {
  const uint8_t *ref0_ptr, *ref1_ptr, *ref2_ptr, *ref3_ptr;
  int32_t ht_cnt = height;
  __m128i src0, src1, ref0, ref1, sad_tmp;
  __m128i sad0 = __lsx_vldi(0);
  __m128i sad1 = sad0;
  __m128i sad2 = sad0;
  __m128i sad3 = sad0;

  ref0_ptr = aref_ptr[0];
  ref1_ptr = aref_ptr[1];
  ref2_ptr = aref_ptr[2];
  ref3_ptr = aref_ptr[3];

  for (; ht_cnt--;) {
    DUP2_ARG2(__lsx_vld, src, 0, src, 16, src0, src1);
    src += src_stride;

    DUP2_ARG2(__lsx_vld, ref0_ptr, 0, ref0_ptr, 16, ref0, ref1);
    ref0_ptr += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);

    DUP2_ARG2(__lsx_vld, ref1_ptr, 0, ref1_ptr, 16, ref0, ref1);
    ref1_ptr += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);

    DUP2_ARG2(__lsx_vld, ref2_ptr, 0, ref2_ptr, 16, ref0, ref1);
    ref2_ptr += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad2 = __lsx_vadd_h(sad2, sad_tmp);

    DUP2_ARG2(__lsx_vld, ref3_ptr, 0, ref3_ptr, 16, ref0, ref1);
    ref3_ptr += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad3 = __lsx_vadd_h(sad3, sad_tmp);
  }
  sad_array[0] = hadd_uh_u32(sad0);
  sad_array[1] = hadd_uh_u32(sad1);
  sad_array[2] = hadd_uh_u32(sad2);
  sad_array[3] = hadd_uh_u32(sad3);
}

static void sad_64width_x4d_lsx(const uint8_t *src, int32_t src_stride,
                                const uint8_t *const aref_ptr[],
                                int32_t ref_stride, int32_t height,
                                uint32_t *sad_array) {
  const uint8_t *ref0_ptr, *ref1_ptr, *ref2_ptr, *ref3_ptr;
  int32_t ht_cnt = height;
  __m128i src0, src1, src2, src3;
  __m128i ref0, ref1, ref2, ref3;
  __m128i sad, sad_tmp;

  __m128i sad0_0 = __lsx_vldi(0);
  __m128i sad0_1 = sad0_0;
  __m128i sad1_0 = sad0_0;
  __m128i sad1_1 = sad0_0;
  __m128i sad2_0 = sad0_0;
  __m128i sad2_1 = sad0_0;
  __m128i sad3_0 = sad0_0;
  __m128i sad3_1 = sad0_0;

  ref0_ptr = aref_ptr[0];
  ref1_ptr = aref_ptr[1];
  ref2_ptr = aref_ptr[2];
  ref3_ptr = aref_ptr[3];

  for (; ht_cnt--;) {
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    src += src_stride;

    DUP4_ARG2(__lsx_vld, ref0_ptr, 0, ref0_ptr, 16, ref0_ptr, 32, ref0_ptr, 48,
              ref0, ref1, ref2, ref3);
    ref0_ptr += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad0_0 = __lsx_vadd_h(sad0_0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, ref2, ref3);
    sad0_1 = __lsx_vadd_h(sad0_1, sad_tmp);

    DUP4_ARG2(__lsx_vld, ref1_ptr, 0, ref1_ptr, 16, ref1_ptr, 32, ref1_ptr, 48,
              ref0, ref1, ref2, ref3);
    ref1_ptr += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad1_0 = __lsx_vadd_h(sad1_0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, ref2, ref3);
    sad1_1 = __lsx_vadd_h(sad1_1, sad_tmp);

    DUP4_ARG2(__lsx_vld, ref2_ptr, 0, ref2_ptr, 16, ref2_ptr, 32, ref2_ptr, 48,
              ref0, ref1, ref2, ref3);
    ref2_ptr += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad2_0 = __lsx_vadd_h(sad2_0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, ref2, ref3);
    sad2_1 = __lsx_vadd_h(sad2_1, sad_tmp);

    DUP4_ARG2(__lsx_vld, ref3_ptr, 0, ref3_ptr, 16, ref3_ptr, 32, ref3_ptr, 48,
              ref0, ref1, ref2, ref3);
    ref3_ptr += ref_stride;
    sad_tmp = sad_ub2_uh(src0, src1, ref0, ref1);
    sad3_0 = __lsx_vadd_h(sad3_0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, ref2, ref3);
    sad3_1 = __lsx_vadd_h(sad3_1, sad_tmp);
  }
  sad = __lsx_vhaddw_wu_hu(sad0_0, sad0_0);
  sad_tmp = __lsx_vhaddw_wu_hu(sad0_1, sad0_1);
  sad = __lsx_vadd_w(sad, sad_tmp);
  sad_array[0] = hadd_uw_u32(sad);

  sad = __lsx_vhaddw_wu_hu(sad1_0, sad1_0);
  sad_tmp = __lsx_vhaddw_wu_hu(sad1_1, sad1_1);
  sad = __lsx_vadd_w(sad, sad_tmp);
  sad_array[1] = hadd_uw_u32(sad);

  sad = __lsx_vhaddw_wu_hu(sad2_0, sad2_0);
  sad_tmp = __lsx_vhaddw_wu_hu(sad2_1, sad2_1);
  sad = __lsx_vadd_w(sad, sad_tmp);
  sad_array[2] = hadd_uw_u32(sad);

  sad = __lsx_vhaddw_wu_hu(sad3_0, sad3_0);
  sad_tmp = __lsx_vhaddw_wu_hu(sad3_1, sad3_1);
  sad = __lsx_vadd_w(sad, sad_tmp);
  sad_array[3] = hadd_uw_u32(sad);
}

static uint32_t avgsad_32width_lsx(const uint8_t *src, int32_t src_stride,
                                   const uint8_t *ref, int32_t ref_stride,
                                   int32_t height, const uint8_t *sec_pred) {
  int32_t res, ht_cnt = (height >> 2);
  __m128i src0, src1, src2, src3, src4, src5, src6, src7;
  __m128i ref0, ref1, ref2, ref3, ref4, ref5, ref6, ref7;
  __m128i pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7;
  __m128i comp0, comp1, sad_tmp;
  __m128i sad = __lsx_vldi(0);
  uint8_t *src_tmp, *ref_tmp;
  int32_t src_stride2 = src_stride << 1;
  int32_t src_stride3 = src_stride2 + src_stride;
  int32_t src_stride4 = src_stride2 << 1;
  int32_t ref_stride2 = ref_stride << 1;
  int32_t ref_stride3 = ref_stride2 + ref_stride;
  int32_t ref_stride4 = ref_stride2 << 1;

  for (; ht_cnt--;) {
    src_tmp = (uint8_t *)src + 16;
    src0 = __lsx_vld(src, 0);
    DUP2_ARG2(__lsx_vldx, src, src_stride, src, src_stride2, src2, src4);
    src6 = __lsx_vldx(src, src_stride3);
    src1 = __lsx_vld(src_tmp, 0);
    DUP2_ARG2(__lsx_vldx, src_tmp, src_stride, src_tmp, src_stride2, src3,
              src5);
    src7 = __lsx_vldx(src_tmp, src_stride3);
    src += src_stride4;

    ref_tmp = (uint8_t *)ref + 16;
    ref0 = __lsx_vld(ref, 0);
    DUP2_ARG2(__lsx_vldx, ref, ref_stride, ref, ref_stride2, ref2, ref4);
    ref6 = __lsx_vldx(ref, ref_stride3);
    ref1 = __lsx_vld(ref_tmp, 0);
    DUP2_ARG2(__lsx_vldx, ref_tmp, ref_stride, ref_tmp, ref_stride2, ref3,
              ref5);
    ref7 = __lsx_vldx(ref_tmp, ref_stride3);
    ref += ref_stride4;

    DUP4_ARG2(__lsx_vld, sec_pred, 0, sec_pred, 32, sec_pred, 64, sec_pred, 96,
              pred0, pred2, pred4, pred6);
    DUP4_ARG2(__lsx_vld, sec_pred, 16, sec_pred, 48, sec_pred, 80, sec_pred,
              112, pred1, pred3, pred5, pred7);
    sec_pred += 128;

    DUP2_ARG2(__lsx_vavgr_bu, pred0, ref0, pred1, ref1, comp0, comp1);
    sad_tmp = sad_ub2_uh(src0, src1, comp0, comp1);
    sad = __lsx_vadd_h(sad, sad_tmp);
    DUP2_ARG2(__lsx_vavgr_bu, pred2, ref2, pred3, ref3, comp0, comp1);
    sad_tmp = sad_ub2_uh(src2, src3, comp0, comp1);
    sad = __lsx_vadd_h(sad, sad_tmp);
    DUP2_ARG2(__lsx_vavgr_bu, pred4, ref4, pred5, ref5, comp0, comp1);
    sad_tmp = sad_ub2_uh(src4, src5, comp0, comp1);
    sad = __lsx_vadd_h(sad, sad_tmp);
    DUP2_ARG2(__lsx_vavgr_bu, pred6, ref6, pred7, ref7, comp0, comp1);
    sad_tmp = sad_ub2_uh(src6, src7, comp0, comp1);
    sad = __lsx_vadd_h(sad, sad_tmp);
  }
  res = hadd_uh_u32(sad);
  return res;
}

static uint32_t avgsad_64width_lsx(const uint8_t *src, int32_t src_stride,
                                   const uint8_t *ref, int32_t ref_stride,
                                   int32_t height, const uint8_t *sec_pred) {
  int32_t res, ht_cnt = (height >> 2);
  __m128i src0, src1, src2, src3, ref0, ref1, ref2, ref3;
  __m128i comp0, comp1, comp2, comp3, pred0, pred1, pred2, pred3;
  __m128i sad, sad_tmp;
  __m128i sad0 = __lsx_vldi(0);
  __m128i sad1 = sad0;

  for (; ht_cnt--;) {
    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    src += src_stride;
    DUP4_ARG2(__lsx_vld, ref, 0, ref, 16, ref, 32, ref, 48, ref0, ref1, ref2,
              ref3);
    ref += ref_stride;
    DUP4_ARG2(__lsx_vld, sec_pred, 0, sec_pred, 16, sec_pred, 32, sec_pred, 48,
              pred0, pred1, pred2, pred3);
    sec_pred += 64;
    DUP4_ARG2(__lsx_vavgr_bu, pred0, ref0, pred1, ref1, pred2, ref2, pred3,
              ref3, comp0, comp1, comp2, comp3);
    sad_tmp = sad_ub2_uh(src0, src1, comp0, comp1);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, comp2, comp3);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);

    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    src += src_stride;
    DUP4_ARG2(__lsx_vld, ref, 0, ref, 16, ref, 32, ref, 48, ref0, ref1, ref2,
              ref3);
    ref += ref_stride;
    DUP4_ARG2(__lsx_vld, sec_pred, 0, sec_pred, 16, sec_pred, 32, sec_pred, 48,
              pred0, pred1, pred2, pred3);
    sec_pred += 64;
    DUP4_ARG2(__lsx_vavgr_bu, pred0, ref0, pred1, ref1, pred2, ref2, pred3,
              ref3, comp0, comp1, comp2, comp3);
    sad_tmp = sad_ub2_uh(src0, src1, comp0, comp1);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, comp2, comp3);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);

    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    src += src_stride;
    DUP4_ARG2(__lsx_vld, ref, 0, ref, 16, ref, 32, ref, 48, ref0, ref1, ref2,
              ref3);
    ref += ref_stride;
    DUP4_ARG2(__lsx_vld, sec_pred, 0, sec_pred, 16, sec_pred, 32, sec_pred, 48,
              pred0, pred1, pred2, pred3);
    sec_pred += 64;
    DUP4_ARG2(__lsx_vavgr_bu, pred0, ref0, pred1, ref1, pred2, ref2, pred3,
              ref3, comp0, comp1, comp2, comp3);
    sad_tmp = sad_ub2_uh(src0, src1, comp0, comp1);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, comp2, comp3);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);

    DUP4_ARG2(__lsx_vld, src, 0, src, 16, src, 32, src, 48, src0, src1, src2,
              src3);
    src += src_stride;
    DUP4_ARG2(__lsx_vld, ref, 0, ref, 16, ref, 32, ref, 48, ref0, ref1, ref2,
              ref3);
    ref += ref_stride;
    DUP4_ARG2(__lsx_vld, sec_pred, 0, sec_pred, 16, sec_pred, 32, sec_pred, 48,
              pred0, pred1, pred2, pred3);
    sec_pred += 64;
    DUP4_ARG2(__lsx_vavgr_bu, pred0, ref0, pred1, ref1, pred2, ref2, pred3,
              ref3, comp0, comp1, comp2, comp3);
    sad_tmp = sad_ub2_uh(src0, src1, comp0, comp1);
    sad0 = __lsx_vadd_h(sad0, sad_tmp);
    sad_tmp = sad_ub2_uh(src2, src3, comp2, comp3);
    sad1 = __lsx_vadd_h(sad1, sad_tmp);
  }
  sad = __lsx_vhaddw_wu_hu(sad0, sad0);
  sad_tmp = __lsx_vhaddw_wu_hu(sad1, sad1);
  sad = __lsx_vadd_w(sad, sad_tmp);

  res = hadd_sw_s32(sad);
  return res;
}

#define VPX_SAD_8xHT_LSX(height)                                             \
  uint32_t vpx_sad8x##height##_lsx(const uint8_t *src, int32_t src_stride,   \
                                   const uint8_t *ref, int32_t ref_stride) { \
    return sad_8width_lsx(src, src_stride, ref, ref_stride, height);         \
  }

#define VPX_SAD_16xHT_LSX(height)                                             \
  uint32_t vpx_sad16x##height##_lsx(const uint8_t *src, int32_t src_stride,   \
                                    const uint8_t *ref, int32_t ref_stride) { \
    return sad_16width_lsx(src, src_stride, ref, ref_stride, height);         \
  }

#define VPX_SAD_32xHT_LSX(height)                                             \
  uint32_t vpx_sad32x##height##_lsx(const uint8_t *src, int32_t src_stride,   \
                                    const uint8_t *ref, int32_t ref_stride) { \
    return sad_32width_lsx(src, src_stride, ref, ref_stride, height);         \
  }

#define VPX_SAD_64xHT_LSX(height)                                             \
  uint32_t vpx_sad64x##height##_lsx(const uint8_t *src, int32_t src_stride,   \
                                    const uint8_t *ref, int32_t ref_stride) { \
    return sad_64width_lsx(src, src_stride, ref, ref_stride, height);         \
  }

#define VPX_SAD_8xHTx4D_LSX(height)                                       \
  void vpx_sad8x##height##x4d_lsx(const uint8_t *src, int32_t src_stride, \
                                  const uint8_t *const refs[4],           \
                                  int32_t ref_stride, uint32_t sads[4]) { \
    sad_8width_x4d_lsx(src, src_stride, refs, ref_stride, height, sads);  \
  }

#define VPX_SAD_16xHTx4D_LSX(height)                                       \
  void vpx_sad16x##height##x4d_lsx(const uint8_t *src, int32_t src_stride, \
                                   const uint8_t *const refs[],            \
                                   int32_t ref_stride, uint32_t *sads) {   \
    sad_16width_x4d_lsx(src, src_stride, refs, ref_stride, height, sads);  \
  }

#define VPX_SAD_32xHTx4D_LSX(height)                                       \
  void vpx_sad32x##height##x4d_lsx(const uint8_t *src, int32_t src_stride, \
                                   const uint8_t *const refs[],            \
                                   int32_t ref_stride, uint32_t *sads) {   \
    sad_32width_x4d_lsx(src, src_stride, refs, ref_stride, height, sads);  \
  }

#define VPX_SAD_64xHTx4D_LSX(height)                                       \
  void vpx_sad64x##height##x4d_lsx(const uint8_t *src, int32_t src_stride, \
                                   const uint8_t *const refs[],            \
                                   int32_t ref_stride, uint32_t *sads) {   \
    sad_64width_x4d_lsx(src, src_stride, refs, ref_stride, height, sads);  \
  }

#define VPX_AVGSAD_32xHT_LSX(height)                                    \
  uint32_t vpx_sad32x##height##_avg_lsx(                                \
      const uint8_t *src, int32_t src_stride, const uint8_t *ref,       \
      int32_t ref_stride, const uint8_t *second_pred) {                 \
    return avgsad_32width_lsx(src, src_stride, ref, ref_stride, height, \
                              second_pred);                             \
  }

#define VPX_AVGSAD_64xHT_LSX(height)                                    \
  uint32_t vpx_sad64x##height##_avg_lsx(                                \
      const uint8_t *src, int32_t src_stride, const uint8_t *ref,       \
      int32_t ref_stride, const uint8_t *second_pred) {                 \
    return avgsad_64width_lsx(src, src_stride, ref, ref_stride, height, \
                              second_pred);                             \
  }

#define SAD64                                                             \
  VPX_SAD_64xHT_LSX(64) VPX_SAD_64xHTx4D_LSX(64) VPX_SAD_64xHTx4D_LSX(32) \
      VPX_AVGSAD_64xHT_LSX(64)

SAD64

#define SAD32                                                             \
  VPX_SAD_32xHT_LSX(32) VPX_SAD_32xHTx4D_LSX(32) VPX_SAD_32xHTx4D_LSX(64) \
      VPX_AVGSAD_32xHT_LSX(32)

SAD32

#define SAD16 VPX_SAD_16xHT_LSX(16) VPX_SAD_16xHTx4D_LSX(16)

SAD16

#define SAD8 VPX_SAD_8xHT_LSX(8) VPX_SAD_8xHTx4D_LSX(8)

SAD8

#undef SAD64
#undef SAD32
#undef SAD16
#undef SAD8
