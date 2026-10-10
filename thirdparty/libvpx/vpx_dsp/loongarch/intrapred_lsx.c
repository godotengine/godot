/*
 * Copyright (c) 2021 Loongson Technology Corporation Limited
 * Contributed by Lu Wang <wanglu@loongson.cn>
 *
 * Use of this source code is governed by a BSD-style license
 * that can be found in the LICENSE file in the root of the source
 * tree. An additional intellectual property rights grant can be found
 * in the file PATENTS.  All contributing project authors may
 * be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_util/loongson_intrinsics.h"

static inline void intra_predict_dc_8x8_lsx(const uint8_t *src_top,
                                            const uint8_t *src_left,
                                            uint8_t *dst, int32_t dst_stride) {
  uint64_t val0, val1;
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i store, sum_h, sum_w, sum_d;
  __m128i src = { 0 };

  val0 = *(const uint64_t *)src_top;
  val1 = *(const uint64_t *)src_left;
  DUP2_ARG3(__lsx_vinsgr2vr_d, src, val0, 0, src, val1, 1, src, src);
  sum_h = __lsx_vhaddw_hu_bu(src, src);
  sum_w = __lsx_vhaddw_wu_hu(sum_h, sum_h);
  sum_d = __lsx_vhaddw_du_wu(sum_w, sum_w);
  sum_w = __lsx_vpickev_w(sum_d, sum_d);
  sum_d = __lsx_vhaddw_du_wu(sum_w, sum_w);
  sum_w = __lsx_vsrari_w(sum_d, 4);
  store = __lsx_vreplvei_b(sum_w, 0);

  __lsx_vstelm_d(store, dst, 0, 0);
  __lsx_vstelm_d(store, dst + dst_stride, 0, 0);
  __lsx_vstelm_d(store, dst + dst_stride_x2, 0, 0);
  __lsx_vstelm_d(store, dst + dst_stride_x3, 0, 0);
  dst += dst_stride_x4;
  __lsx_vstelm_d(store, dst, 0, 0);
  __lsx_vstelm_d(store, dst + dst_stride, 0, 0);
  __lsx_vstelm_d(store, dst + dst_stride_x2, 0, 0);
  __lsx_vstelm_d(store, dst + dst_stride_x3, 0, 0);
}

static inline void intra_predict_dc_16x16_lsx(const uint8_t *src_top,
                                              const uint8_t *src_left,
                                              uint8_t *dst,
                                              int32_t dst_stride) {
  int32_t dst_stride_x2 = dst_stride << 1;
  int32_t dst_stride_x3 = dst_stride_x2 + dst_stride;
  int32_t dst_stride_x4 = dst_stride << 2;
  __m128i top, left, out;
  __m128i sum_h, sum_top, sum_left;
  __m128i sum_w;
  __m128i sum_d;

  DUP2_ARG2(__lsx_vld, src_top, 0, src_left, 0, top, left);
  DUP2_ARG2(__lsx_vhaddw_hu_bu, top, top, left, left, sum_top, sum_left);
  sum_h = __lsx_vadd_h(sum_top, sum_left);
  sum_w = __lsx_vhaddw_wu_hu(sum_h, sum_h);
  sum_d = __lsx_vhaddw_du_wu(sum_w, sum_w);
  sum_w = __lsx_vpickev_w(sum_d, sum_d);
  sum_d = __lsx_vhaddw_du_wu(sum_w, sum_w);
  sum_w = __lsx_vsrari_w(sum_d, 5);
  out = __lsx_vreplvei_b(sum_w, 0);

  __lsx_vstx(out, dst, 0);
  __lsx_vstx(out, dst, dst_stride);
  __lsx_vstx(out, dst, dst_stride_x2);
  __lsx_vstx(out, dst, dst_stride_x3);
  dst += dst_stride_x4;
  __lsx_vstx(out, dst, 0);
  __lsx_vstx(out, dst, dst_stride);
  __lsx_vstx(out, dst, dst_stride_x2);
  __lsx_vstx(out, dst, dst_stride_x3);
  dst += dst_stride_x4;
  __lsx_vstx(out, dst, 0);
  __lsx_vstx(out, dst, dst_stride);
  __lsx_vstx(out, dst, dst_stride_x2);
  __lsx_vstx(out, dst, dst_stride_x3);
  dst += dst_stride_x4;
  __lsx_vstx(out, dst, 0);
  __lsx_vstx(out, dst, dst_stride);
  __lsx_vstx(out, dst, dst_stride_x2);
  __lsx_vstx(out, dst, dst_stride_x3);
}

void vpx_dc_predictor_8x8_lsx(uint8_t *dst, ptrdiff_t y_stride,
                              const uint8_t *above, const uint8_t *left) {
  intra_predict_dc_8x8_lsx(above, left, dst, y_stride);
}

void vpx_dc_predictor_16x16_lsx(uint8_t *dst, ptrdiff_t y_stride,
                                const uint8_t *above, const uint8_t *left) {
  intra_predict_dc_16x16_lsx(above, left, dst, y_stride);
}
