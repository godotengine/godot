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
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"

static INLINE __m128i calculate_qcoeff(__m128i coeff, __m128i coeff_abs,
                                       __m128i round, __m128i quant,
                                       __m128i shift, __m128i cmp_mask) {
  __m128i rounded, qcoeff;

  rounded = __lsx_vsadd_h(coeff_abs, round);
  qcoeff = __lsx_vmuh_h(rounded, quant);
  qcoeff = __lsx_vadd_h(rounded, qcoeff);
  qcoeff = __lsx_vmuh_h(qcoeff, shift);
  qcoeff = __lsx_vsigncov_h(coeff, qcoeff);
  qcoeff = __lsx_vand_v(qcoeff, cmp_mask);

  return qcoeff;
}

static INLINE void calculate_dqcoeff_and_store(__m128i qcoeff, __m128i dequant,
                                               int16_t *dqcoeff) {
  __m128i dqcoeff16 = __lsx_vmul_h(qcoeff, dequant);
  __lsx_vst(dqcoeff16, dqcoeff, 0);
}

static INLINE void calculate_dqcoeff_and_store_32x32(__m128i qcoeff,
                                                     __m128i dequant,
                                                     int16_t *dqcoeff) {
  // Un-sign to bias rounding like C.
  __m128i low, high, dqcoeff32_0, dqcoeff32_1, res;
  __m128i zero = __lsx_vldi(0);
  __m128i coeff = __lsx_vabsd_h(qcoeff, zero);

  const __m128i sign_0 = __lsx_vilvl_h(qcoeff, zero);
  const __m128i sign_1 = __lsx_vilvh_h(qcoeff, zero);

  low = __lsx_vmul_h(coeff, dequant);
  high = __lsx_vmuh_h(coeff, dequant);
  dqcoeff32_0 = __lsx_vilvl_h(high, low);
  dqcoeff32_1 = __lsx_vilvh_h(high, low);

  // "Divide" by 2.
  dqcoeff32_0 = __lsx_vsrai_w(dqcoeff32_0, 1);
  dqcoeff32_1 = __lsx_vsrai_w(dqcoeff32_1, 1);
  dqcoeff32_0 = __lsx_vsigncov_w(sign_0, dqcoeff32_0);
  dqcoeff32_1 = __lsx_vsigncov_w(sign_1, dqcoeff32_1);
  res = __lsx_vpickev_h(dqcoeff32_1, dqcoeff32_0);
  __lsx_vst(res, dqcoeff, 0);
}

static INLINE __m128i scan_for_eob(__m128i coeff0, __m128i coeff1,
                                   const int16_t *scan, int index,
                                   __m128i zero) {
  const __m128i zero_coeff0 = __lsx_vseq_h(coeff0, zero);
  const __m128i zero_coeff1 = __lsx_vseq_h(coeff1, zero);
  __m128i scan0 = __lsx_vld(scan + index, 0);
  __m128i scan1 = __lsx_vld(scan + index + 8, 0);
  __m128i eob0, eob1;

  eob0 = __lsx_vandn_v(zero_coeff0, scan0);
  eob1 = __lsx_vandn_v(zero_coeff1, scan1);
  return __lsx_vmax_h(eob0, eob1);
}

static INLINE int16_t accumulate_eob(__m128i eob) {
  __m128i eob_shuffled;
  int16_t res_m;

  eob_shuffled = __lsx_vshuf4i_w(eob, 0xe);
  eob = __lsx_vmax_h(eob, eob_shuffled);
  eob_shuffled = __lsx_vshuf4i_h(eob, 0xe);
  eob = __lsx_vmax_h(eob, eob_shuffled);
  eob_shuffled = __lsx_vshuf4i_h(eob, 0x1);
  eob = __lsx_vmax_h(eob, eob_shuffled);
  res_m = __lsx_vpickve2gr_h(eob, 1);

  return res_m;
}

#if !CONFIG_VP9_HIGHBITDEPTH

void vpx_quantize_b_lsx(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                        const struct macroblock_plane *const mb_plane,
                        tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                        const int16_t *dequant_ptr, uint16_t *eob_ptr,
                        const struct ScanOrder *const scan_order) {
  __m128i zero = __lsx_vldi(0);
  int index = 16;
  const int16_t *iscan = scan_order->iscan;

  __m128i zbin, round, quant, dequant, quant_shift;
  __m128i coeff0, coeff1;
  __m128i qcoeff0, qcoeff1;
  __m128i cmp_mask0, cmp_mask1;
  __m128i eob, eob0;

  zbin = __lsx_vld(mb_plane->zbin, 0);
  round = __lsx_vld(mb_plane->round, 0);
  quant = __lsx_vld(mb_plane->quant, 0);
  dequant = __lsx_vld(dequant_ptr, 0);
  quant_shift = __lsx_vld(mb_plane->quant_shift, 0);
  // Handle one DC and first 15 AC.
  DUP2_ARG2(__lsx_vld, coeff_ptr, 0, coeff_ptr, 16, coeff0, coeff1);
  qcoeff0 = __lsx_vabsd_h(coeff0, zero);
  qcoeff1 = __lsx_vabsd_h(coeff1, zero);

  cmp_mask0 = __lsx_vsle_h(zbin, qcoeff0);
  zbin = __lsx_vilvh_d(zbin, zbin);
  cmp_mask1 = __lsx_vsle_h(zbin, qcoeff1);

  qcoeff0 =
      calculate_qcoeff(coeff0, qcoeff0, round, quant, quant_shift, cmp_mask0);
  round = __lsx_vilvh_d(round, round);
  quant = __lsx_vilvh_d(quant, quant);
  quant_shift = __lsx_vilvh_d(quant_shift, quant_shift);
  qcoeff1 =
      calculate_qcoeff(coeff1, qcoeff1, round, quant, quant_shift, cmp_mask1);

  __lsx_vst(qcoeff0, qcoeff_ptr, 0);
  __lsx_vst(qcoeff1, qcoeff_ptr, 16);

  calculate_dqcoeff_and_store(qcoeff0, dequant, dqcoeff_ptr);
  dequant = __lsx_vilvh_d(dequant, dequant);
  calculate_dqcoeff_and_store(qcoeff1, dequant, dqcoeff_ptr + 8);

  eob = scan_for_eob(qcoeff0, qcoeff1, iscan, 0, zero);
  // AC only loop.
  while (index < n_coeffs) {
    coeff0 = __lsx_vld(coeff_ptr + index, 0);
    coeff1 = __lsx_vld(coeff_ptr + index + 8, 0);

    qcoeff0 = __lsx_vabsd_h(coeff0, zero);
    qcoeff1 = __lsx_vabsd_h(coeff1, zero);

    cmp_mask0 = __lsx_vsle_h(zbin, qcoeff0);
    cmp_mask1 = __lsx_vsle_h(zbin, qcoeff1);

    qcoeff0 =
        calculate_qcoeff(coeff0, qcoeff0, round, quant, quant_shift, cmp_mask0);
    qcoeff1 =
        calculate_qcoeff(coeff1, qcoeff1, round, quant, quant_shift, cmp_mask1);

    __lsx_vst(qcoeff0, qcoeff_ptr + index, 0);
    __lsx_vst(qcoeff1, qcoeff_ptr + index + 8, 0);

    calculate_dqcoeff_and_store(qcoeff0, dequant, dqcoeff_ptr + index);
    calculate_dqcoeff_and_store(qcoeff1, dequant, dqcoeff_ptr + index + 8);

    eob0 = scan_for_eob(qcoeff0, qcoeff1, iscan, index, zero);
    eob = __lsx_vmax_h(eob, eob0);

    index += 16;
  }

  *eob_ptr = accumulate_eob(eob);
}

void vpx_quantize_b_32x32_lsx(const tran_low_t *coeff_ptr,
                              const struct macroblock_plane *const mb_plane,
                              tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                              const int16_t *dequant_ptr, uint16_t *eob_ptr,
                              const struct ScanOrder *const scan_order) {
  __m128i zero = __lsx_vldi(0);
  int index;
  const int16_t *iscan = scan_order->iscan;

  __m128i zbin, round, quant, dequant, quant_shift;
  __m128i coeff0, coeff1, qcoeff0, qcoeff1, cmp_mask0, cmp_mask1;
  __m128i eob = zero, eob0;

  zbin = __lsx_vld(mb_plane->zbin, 0);
  zbin = __lsx_vsrari_h(zbin, 1);
  round = __lsx_vld(mb_plane->round, 0);
  round = __lsx_vsrari_h(round, 1);

  quant = __lsx_vld(mb_plane->quant, 0);
  dequant = __lsx_vld(dequant_ptr, 0);
  quant_shift = __lsx_vld(mb_plane->quant_shift, 0);
  quant_shift = __lsx_vslli_h(quant_shift, 1);
  // Handle one DC and first 15 AC.
  DUP2_ARG2(__lsx_vld, coeff_ptr, 0, coeff_ptr, 16, coeff0, coeff1);
  qcoeff0 = __lsx_vabsd_h(coeff0, zero);
  qcoeff1 = __lsx_vabsd_h(coeff1, zero);

  cmp_mask0 = __lsx_vsle_h(zbin, qcoeff0);
  // remove DC from zbin
  zbin = __lsx_vilvh_d(zbin, zbin);
  cmp_mask1 = __lsx_vsle_h(zbin, qcoeff1);

  qcoeff0 =
      calculate_qcoeff(coeff0, qcoeff0, round, quant, quant_shift, cmp_mask0);
  // remove DC in quant_shift, quant, quant_shift
  round = __lsx_vilvh_d(round, round);
  quant = __lsx_vilvh_d(quant, quant);
  quant_shift = __lsx_vilvh_d(quant_shift, quant_shift);
  qcoeff1 =
      calculate_qcoeff(coeff1, qcoeff1, round, quant, quant_shift, cmp_mask1);
  __lsx_vst(qcoeff0, qcoeff_ptr, 0);
  __lsx_vst(qcoeff1, qcoeff_ptr, 16);

  calculate_dqcoeff_and_store_32x32(qcoeff0, dequant, dqcoeff_ptr);
  dequant = __lsx_vilvh_d(dequant, dequant);
  calculate_dqcoeff_and_store_32x32(qcoeff1, dequant, dqcoeff_ptr + 8);
  eob = scan_for_eob(qcoeff0, qcoeff1, iscan, 0, zero);
  // AC only loop.
  for (index = 16; index < 32 * 32; index += 16) {
    coeff0 = __lsx_vld(coeff_ptr + index, 0);
    coeff1 = __lsx_vld(coeff_ptr + index + 8, 0);

    qcoeff0 = __lsx_vabsd_h(coeff0, zero);
    qcoeff1 = __lsx_vabsd_h(coeff1, zero);

    cmp_mask0 = __lsx_vsle_h(zbin, qcoeff0);
    cmp_mask1 = __lsx_vsle_h(zbin, qcoeff1);

    qcoeff0 =
        calculate_qcoeff(coeff0, qcoeff0, round, quant, quant_shift, cmp_mask0);
    qcoeff1 =
        calculate_qcoeff(coeff1, qcoeff1, round, quant, quant_shift, cmp_mask1);
    __lsx_vst(qcoeff0, qcoeff_ptr + index, 0);
    __lsx_vst(qcoeff1, qcoeff_ptr + index + 8, 0);

    calculate_dqcoeff_and_store_32x32(qcoeff0, dequant, dqcoeff_ptr + index);
    calculate_dqcoeff_and_store_32x32(qcoeff1, dequant,
                                      dqcoeff_ptr + 8 + index);
    eob0 = scan_for_eob(qcoeff0, qcoeff1, iscan, index, zero);
    eob = __lsx_vmax_h(eob, eob0);
  }

  *eob_ptr = accumulate_eob(eob);
}
#endif
