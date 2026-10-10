/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <tmmintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/bitdepth_conversion_sse2.h"
#include "vpx_dsp/x86/quantize_sse2.h"
#include "vpx_dsp/x86/quantize_ssse3.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"

void vpx_quantize_b_ssse3(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                          const struct macroblock_plane *const mb_plane,
                          tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                          const int16_t *dequant_ptr, uint16_t *eob_ptr,
                          const struct ScanOrder *const scan_order) {
  const __m128i zero = _mm_setzero_si128();
  int index = 16;
  const int16_t *iscan = scan_order->iscan;

  __m128i zbin, round, quant, dequant, shift;
  __m128i coeff0, coeff1;
  __m128i qcoeff0, qcoeff1;
  __m128i cmp_mask0, cmp_mask1;
  __m128i eob, eob0;

  load_b_values(mb_plane, &zbin, &round, &quant, dequant_ptr, &dequant, &shift);

  // Do DC and first 15 AC.
  coeff0 = load_tran_low(coeff_ptr);
  coeff1 = load_tran_low(coeff_ptr + 8);

  qcoeff0 = _mm_abs_epi16(coeff0);
  qcoeff1 = _mm_abs_epi16(coeff1);

  cmp_mask0 = _mm_cmpgt_epi16(qcoeff0, zbin);
  zbin = _mm_unpackhi_epi64(zbin, zbin);  // Switch DC to AC
  cmp_mask1 = _mm_cmpgt_epi16(qcoeff1, zbin);

  calculate_qcoeff(&qcoeff0, round, quant, shift);
  round = _mm_unpackhi_epi64(round, round);
  quant = _mm_unpackhi_epi64(quant, quant);
  shift = _mm_unpackhi_epi64(shift, shift);
  calculate_qcoeff(&qcoeff1, round, quant, shift);

  // Reinsert signs
  qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
  qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

  // Mask out zbin threshold coeffs
  qcoeff0 = _mm_and_si128(qcoeff0, cmp_mask0);
  qcoeff1 = _mm_and_si128(qcoeff1, cmp_mask1);

  store_tran_low(qcoeff0, qcoeff_ptr);
  store_tran_low(qcoeff1, qcoeff_ptr + 8);

  calculate_dqcoeff_and_store(qcoeff0, dequant, dqcoeff_ptr);
  dequant = _mm_unpackhi_epi64(dequant, dequant);
  calculate_dqcoeff_and_store(qcoeff1, dequant, dqcoeff_ptr + 8);

  eob = scan_for_eob(&qcoeff0, &qcoeff1, iscan, 0, zero);

  // AC only loop.
  while (index < n_coeffs) {
    coeff0 = load_tran_low(coeff_ptr + index);
    coeff1 = load_tran_low(coeff_ptr + index + 8);

    qcoeff0 = _mm_abs_epi16(coeff0);
    qcoeff1 = _mm_abs_epi16(coeff1);

    cmp_mask0 = _mm_cmpgt_epi16(qcoeff0, zbin);
    cmp_mask1 = _mm_cmpgt_epi16(qcoeff1, zbin);

    calculate_qcoeff(&qcoeff0, round, quant, shift);
    calculate_qcoeff(&qcoeff1, round, quant, shift);

    qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
    qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

    qcoeff0 = _mm_and_si128(qcoeff0, cmp_mask0);
    qcoeff1 = _mm_and_si128(qcoeff1, cmp_mask1);

    store_tran_low(qcoeff0, qcoeff_ptr + index);
    store_tran_low(qcoeff1, qcoeff_ptr + index + 8);

    calculate_dqcoeff_and_store(qcoeff0, dequant, dqcoeff_ptr + index);
    calculate_dqcoeff_and_store(qcoeff1, dequant, dqcoeff_ptr + index + 8);

    eob0 = scan_for_eob(&qcoeff0, &qcoeff1, iscan, index, zero);
    eob = _mm_max_epi16(eob, eob0);

    index += 16;
  }

  *eob_ptr = accumulate_eob(eob);
}

void vpx_quantize_b_32x32_ssse3(const tran_low_t *coeff_ptr,
                                const struct macroblock_plane *const mb_plane,
                                tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                                const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                const struct ScanOrder *const scan_order) {
  const __m128i zero = _mm_setzero_si128();
  int index;
  const int16_t *iscan = scan_order->iscan;

  __m128i zbin, round, quant, dequant, shift;
  __m128i coeff0, coeff1;
  __m128i qcoeff0, qcoeff1;
  __m128i cmp_mask0, cmp_mask1;
  __m128i all_zero;
  __m128i eob = zero, eob0;

  load_b_values32x32(mb_plane, &zbin, &round, &quant, dequant_ptr, &dequant,
                     &shift);

  // Do DC and first 15 AC.
  coeff0 = load_tran_low(coeff_ptr);
  coeff1 = load_tran_low(coeff_ptr + 8);

  qcoeff0 = _mm_abs_epi16(coeff0);
  qcoeff1 = _mm_abs_epi16(coeff1);

  cmp_mask0 = _mm_cmpgt_epi16(qcoeff0, zbin);
  zbin = _mm_unpackhi_epi64(zbin, zbin);  // Switch DC to AC.
  cmp_mask1 = _mm_cmpgt_epi16(qcoeff1, zbin);

  all_zero = _mm_or_si128(cmp_mask0, cmp_mask1);
  if (_mm_movemask_epi8(all_zero) == 0) {
    _mm_store_si128((__m128i *)(qcoeff_ptr), zero);
    _mm_store_si128((__m128i *)(qcoeff_ptr + 8), zero);
    _mm_store_si128((__m128i *)(dqcoeff_ptr), zero);
    _mm_store_si128((__m128i *)(dqcoeff_ptr + 8), zero);
#if CONFIG_VP9_HIGHBITDEPTH
    _mm_store_si128((__m128i *)(qcoeff_ptr + 4), zero);
    _mm_store_si128((__m128i *)(qcoeff_ptr + 12), zero);
    _mm_store_si128((__m128i *)(dqcoeff_ptr + 4), zero);
    _mm_store_si128((__m128i *)(dqcoeff_ptr + 12), zero);
#endif  // CONFIG_HIGHBITDEPTH

    round = _mm_unpackhi_epi64(round, round);
    quant = _mm_unpackhi_epi64(quant, quant);
    shift = _mm_unpackhi_epi64(shift, shift);
    dequant = _mm_unpackhi_epi64(dequant, dequant);
  } else {
    calculate_qcoeff(&qcoeff0, round, quant, shift);
    round = _mm_unpackhi_epi64(round, round);
    quant = _mm_unpackhi_epi64(quant, quant);
    shift = _mm_unpackhi_epi64(shift, shift);
    calculate_qcoeff(&qcoeff1, round, quant, shift);

    // Reinsert signs.
    qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
    qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

    // Mask out zbin threshold coeffs.
    qcoeff0 = _mm_and_si128(qcoeff0, cmp_mask0);
    qcoeff1 = _mm_and_si128(qcoeff1, cmp_mask1);

    store_tran_low(qcoeff0, qcoeff_ptr);
    store_tran_low(qcoeff1, qcoeff_ptr + 8);

    calculate_dqcoeff_and_store_32x32(qcoeff0, dequant, zero, dqcoeff_ptr);
    dequant = _mm_unpackhi_epi64(dequant, dequant);
    calculate_dqcoeff_and_store_32x32(qcoeff1, dequant, zero, dqcoeff_ptr + 8);

    eob = scan_for_eob(&qcoeff0, &qcoeff1, iscan, 0, zero);
  }

  // AC only loop.
  for (index = 16; index < 32 * 32; index += 16) {
    coeff0 = load_tran_low(coeff_ptr + index);
    coeff1 = load_tran_low(coeff_ptr + index + 8);

    qcoeff0 = _mm_abs_epi16(coeff0);
    qcoeff1 = _mm_abs_epi16(coeff1);

    cmp_mask0 = _mm_cmpgt_epi16(qcoeff0, zbin);
    cmp_mask1 = _mm_cmpgt_epi16(qcoeff1, zbin);

    all_zero = _mm_or_si128(cmp_mask0, cmp_mask1);
    if (_mm_movemask_epi8(all_zero) == 0) {
      _mm_store_si128((__m128i *)(qcoeff_ptr + index), zero);
      _mm_store_si128((__m128i *)(qcoeff_ptr + index + 8), zero);
      _mm_store_si128((__m128i *)(dqcoeff_ptr + index), zero);
      _mm_store_si128((__m128i *)(dqcoeff_ptr + index + 8), zero);
#if CONFIG_VP9_HIGHBITDEPTH
      _mm_store_si128((__m128i *)(qcoeff_ptr + index + 4), zero);
      _mm_store_si128((__m128i *)(qcoeff_ptr + index + 12), zero);
      _mm_store_si128((__m128i *)(dqcoeff_ptr + index + 4), zero);
      _mm_store_si128((__m128i *)(dqcoeff_ptr + index + 12), zero);
#endif  // CONFIG_VP9_HIGHBITDEPTH
      continue;
    }

    calculate_qcoeff(&qcoeff0, round, quant, shift);
    calculate_qcoeff(&qcoeff1, round, quant, shift);

    qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
    qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

    qcoeff0 = _mm_and_si128(qcoeff0, cmp_mask0);
    qcoeff1 = _mm_and_si128(qcoeff1, cmp_mask1);

    store_tran_low(qcoeff0, qcoeff_ptr + index);
    store_tran_low(qcoeff1, qcoeff_ptr + index + 8);

    calculate_dqcoeff_and_store_32x32(qcoeff0, dequant, zero,
                                      dqcoeff_ptr + index);
    calculate_dqcoeff_and_store_32x32(qcoeff1, dequant, zero,
                                      dqcoeff_ptr + 8 + index);

    eob0 = scan_for_eob(&qcoeff0, &qcoeff1, iscan, index, zero);
    eob = _mm_max_epi16(eob, eob0);
  }

  *eob_ptr = accumulate_eob(eob);
}
