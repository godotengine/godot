/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <tmmintrin.h>

#include "./vp9_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/x86/bitdepth_conversion_sse2.h"
#include "vpx_dsp/x86/quantize_sse2.h"
#include "vpx_dsp/x86/quantize_ssse3.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"

void vp9_quantize_fp_ssse3(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                           const struct macroblock_plane *const mb_plane,
                           tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                           const int16_t *dequant_ptr, uint16_t *eob_ptr,
                           const struct ScanOrder *const scan_order) {
  const __m128i zero = _mm_setzero_si128();
  __m128i thr;
  int nzflag;
  int index = 16;
  __m128i round, quant, dequant;
  __m128i coeff0, coeff1;
  __m128i qcoeff0, qcoeff1;
  __m128i eob;
  const int16_t *iscan = scan_order->iscan;

  // Setup global values.
  load_fp_values(mb_plane, &round, &quant, dequant_ptr, &dequant);

  // Do DC and first 15 AC.
  coeff0 = load_tran_low(coeff_ptr);
  coeff1 = load_tran_low(coeff_ptr + 8);

  qcoeff0 = _mm_abs_epi16(coeff0);
  qcoeff1 = _mm_abs_epi16(coeff1);

  qcoeff0 = _mm_adds_epi16(qcoeff0, round);
  qcoeff0 = _mm_mulhi_epi16(qcoeff0, quant);

  round = _mm_unpackhi_epi64(round, round);
  quant = _mm_unpackhi_epi64(quant, quant);

  qcoeff1 = _mm_adds_epi16(qcoeff1, round);
  qcoeff1 = _mm_mulhi_epi16(qcoeff1, quant);

  // Reinsert signs.
  qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
  qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

  store_tran_low(qcoeff0, qcoeff_ptr);
  store_tran_low(qcoeff1, qcoeff_ptr + 8);

  qcoeff0 = _mm_mullo_epi16(qcoeff0, dequant);
  dequant = _mm_unpackhi_epi64(dequant, dequant);
  qcoeff1 = _mm_mullo_epi16(qcoeff1, dequant);

  store_tran_low(qcoeff0, dqcoeff_ptr);
  store_tran_low(qcoeff1, dqcoeff_ptr + 8);

  eob = scan_for_eob(&qcoeff0, &qcoeff1, iscan, 0, zero);

  thr = _mm_srai_epi16(dequant, 1);

  // AC only loop.
  while (index < n_coeffs) {
    coeff0 = load_tran_low(coeff_ptr + index);
    coeff1 = load_tran_low(coeff_ptr + index + 8);

    qcoeff0 = _mm_abs_epi16(coeff0);
    qcoeff1 = _mm_abs_epi16(coeff1);

    nzflag = _mm_movemask_epi8(_mm_cmpgt_epi16(qcoeff0, thr)) |
             _mm_movemask_epi8(_mm_cmpgt_epi16(qcoeff1, thr));

    if (nzflag) {
      __m128i eob0;
      qcoeff0 = _mm_adds_epi16(qcoeff0, round);
      qcoeff1 = _mm_adds_epi16(qcoeff1, round);
      qcoeff0 = _mm_mulhi_epi16(qcoeff0, quant);
      qcoeff1 = _mm_mulhi_epi16(qcoeff1, quant);

      // Reinsert signs.
      qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
      qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

      store_tran_low(qcoeff0, qcoeff_ptr + index);
      store_tran_low(qcoeff1, qcoeff_ptr + index + 8);

      qcoeff0 = _mm_mullo_epi16(qcoeff0, dequant);
      qcoeff1 = _mm_mullo_epi16(qcoeff1, dequant);

      store_tran_low(qcoeff0, dqcoeff_ptr + index);
      store_tran_low(qcoeff1, dqcoeff_ptr + index + 8);

      eob0 = scan_for_eob(&qcoeff0, &qcoeff1, iscan, index, zero);
      eob = _mm_max_epi16(eob, eob0);
    } else {
      store_zero_tran_low(qcoeff_ptr + index);
      store_zero_tran_low(qcoeff_ptr + index + 8);

      store_zero_tran_low(dqcoeff_ptr + index);
      store_zero_tran_low(dqcoeff_ptr + index + 8);
    }

    index += 16;
  }

  *eob_ptr = accumulate_eob(eob);
}

void vp9_quantize_fp_32x32_ssse3(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                                 const struct macroblock_plane *const mb_plane,
                                 tran_low_t *qcoeff_ptr,
                                 tran_low_t *dqcoeff_ptr,
                                 const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                 const struct ScanOrder *const scan_order) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i one_s16 = _mm_set1_epi16(1);
  __m128i thr;
  int nzflag;
  int index = 16;
  __m128i round, quant, dequant;
  __m128i coeff0, coeff1;
  __m128i qcoeff0, qcoeff1;
  __m128i eob;
  const int16_t *iscan = scan_order->iscan;

  // Setup global values.
  load_fp_values(mb_plane, &round, &quant, dequant_ptr, &dequant);
  // The 32x32 halves round.
  round = _mm_add_epi16(round, one_s16);
  round = _mm_srli_epi16(round, 1);

  // The 16x16 shifts by 16, the 32x32 shifts by 15. We want to use pmulhw so
  // upshift quant to account for this.
  quant = _mm_slli_epi16(quant, 1);

  // Do DC and first 15 AC.
  coeff0 = load_tran_low(coeff_ptr);
  coeff1 = load_tran_low(coeff_ptr + 8);

  qcoeff0 = _mm_abs_epi16(coeff0);
  qcoeff1 = _mm_abs_epi16(coeff1);

  qcoeff0 = _mm_adds_epi16(qcoeff0, round);
  qcoeff0 = _mm_mulhi_epi16(qcoeff0, quant);

  round = _mm_unpackhi_epi64(round, round);
  quant = _mm_unpackhi_epi64(quant, quant);

  qcoeff1 = _mm_adds_epi16(qcoeff1, round);
  qcoeff1 = _mm_mulhi_epi16(qcoeff1, quant);

  // Reinsert signs.
  qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
  qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

  store_tran_low(qcoeff0, qcoeff_ptr);
  store_tran_low(qcoeff1, qcoeff_ptr + 8);

  // Get the abs value of qcoeff again so we can use shifts for division.
  qcoeff0 = _mm_abs_epi16(qcoeff0);
  qcoeff1 = _mm_abs_epi16(qcoeff1);

  qcoeff0 = _mm_mullo_epi16(qcoeff0, dequant);
  dequant = _mm_unpackhi_epi64(dequant, dequant);
  qcoeff1 = _mm_mullo_epi16(qcoeff1, dequant);

  // Divide by 2.
  qcoeff0 = _mm_srli_epi16(qcoeff0, 1);
  qcoeff1 = _mm_srli_epi16(qcoeff1, 1);

  // Reinsert signs.
  qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
  qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

  store_tran_low(qcoeff0, dqcoeff_ptr);
  store_tran_low(qcoeff1, dqcoeff_ptr + 8);

  eob = scan_for_eob(&qcoeff0, &qcoeff1, iscan, 0, zero);

  thr = _mm_srai_epi16(dequant, 2);

  // AC only loop.
  while (index < n_coeffs) {
    coeff0 = load_tran_low(coeff_ptr + index);
    coeff1 = load_tran_low(coeff_ptr + index + 8);

    qcoeff0 = _mm_abs_epi16(coeff0);
    qcoeff1 = _mm_abs_epi16(coeff1);

    nzflag = _mm_movemask_epi8(_mm_cmpgt_epi16(qcoeff0, thr)) |
             _mm_movemask_epi8(_mm_cmpgt_epi16(qcoeff1, thr));

    if (nzflag) {
      qcoeff0 = _mm_adds_epi16(qcoeff0, round);
      qcoeff1 = _mm_adds_epi16(qcoeff1, round);
      qcoeff0 = _mm_mulhi_epi16(qcoeff0, quant);
      qcoeff1 = _mm_mulhi_epi16(qcoeff1, quant);

      // Reinsert signs.
      qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
      qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

      store_tran_low(qcoeff0, qcoeff_ptr + index);
      store_tran_low(qcoeff1, qcoeff_ptr + index + 8);

      // Get the abs value of qcoeff again so we can use shifts for division.
      qcoeff0 = _mm_abs_epi16(qcoeff0);
      qcoeff1 = _mm_abs_epi16(qcoeff1);

      qcoeff0 = _mm_mullo_epi16(qcoeff0, dequant);
      qcoeff1 = _mm_mullo_epi16(qcoeff1, dequant);

      // Divide by 2.
      qcoeff0 = _mm_srli_epi16(qcoeff0, 1);
      qcoeff1 = _mm_srli_epi16(qcoeff1, 1);

      // Reinsert signs.
      qcoeff0 = _mm_sign_epi16(qcoeff0, coeff0);
      qcoeff1 = _mm_sign_epi16(qcoeff1, coeff1);

      store_tran_low(qcoeff0, dqcoeff_ptr + index);
      store_tran_low(qcoeff1, dqcoeff_ptr + index + 8);
    } else {
      store_zero_tran_low(qcoeff_ptr + index);
      store_zero_tran_low(qcoeff_ptr + index + 8);

      store_zero_tran_low(dqcoeff_ptr + index);
      store_zero_tran_low(dqcoeff_ptr + index + 8);
    }

    if (nzflag) {
      const __m128i eob0 = scan_for_eob(&qcoeff0, &qcoeff1, iscan, index, zero);
      eob = _mm_max_epi16(eob, eob0);
    }
    index += 16;
  }

  *eob_ptr = accumulate_eob(eob);
}
