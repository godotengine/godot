/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/quantize.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"

void vpx_quantize_dc(const tran_low_t *coeff_ptr, int n_coeffs,
                     const int16_t *round_ptr, const int16_t quant,
                     tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                     const int16_t dequant, uint16_t *eob_ptr) {
  const int rc = 0;
  const int coeff = coeff_ptr[rc];
  const int coeff_sign = (coeff >> 31);
  const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
  int tmp, eob = -1;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  tmp = clamp(abs_coeff + round_ptr[rc != 0], INT16_MIN, INT16_MAX);
  tmp = (tmp * quant) >> 16;
  qcoeff_ptr[rc] = (tmp ^ coeff_sign) - coeff_sign;
  dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant;
  if (tmp) eob = 0;

  *eob_ptr = eob + 1;
}

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_quantize_dc(const tran_low_t *coeff_ptr, int n_coeffs,
                            const int16_t *round_ptr, const int16_t quant,
                            tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                            const int16_t dequant, uint16_t *eob_ptr) {
  int eob = -1;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  {
    const int coeff = coeff_ptr[0];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
    const int64_t tmp = abs_coeff + round_ptr[0];
    const int abs_qcoeff = (int)((tmp * quant) >> 16);
    qcoeff_ptr[0] = (tran_low_t)((abs_qcoeff ^ coeff_sign) - coeff_sign);
    dqcoeff_ptr[0] = qcoeff_ptr[0] * dequant;
    if (abs_qcoeff) eob = 0;
  }

  *eob_ptr = eob + 1;
}
#endif

void vpx_quantize_dc_32x32(const tran_low_t *coeff_ptr,
                           const int16_t *round_ptr, const int16_t quant,
                           tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                           const int16_t dequant, uint16_t *eob_ptr) {
  const int n_coeffs = 1024;
  const int rc = 0;
  const int coeff = coeff_ptr[rc];
  const int coeff_sign = (coeff >> 31);
  const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
  int tmp, eob = -1;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  tmp = clamp(abs_coeff + ROUND_POWER_OF_TWO(round_ptr[rc != 0], 1), INT16_MIN,
              INT16_MAX);
  tmp = (tmp * quant) >> 15;
  qcoeff_ptr[rc] = (tmp ^ coeff_sign) - coeff_sign;
  dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant / 2;
  if (tmp) eob = 0;

  *eob_ptr = eob + 1;
}

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_quantize_dc_32x32(const tran_low_t *coeff_ptr,
                                  const int16_t *round_ptr, const int16_t quant,
                                  tran_low_t *qcoeff_ptr,
                                  tran_low_t *dqcoeff_ptr,
                                  const int16_t dequant, uint16_t *eob_ptr) {
  const int n_coeffs = 1024;
  int eob = -1;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  {
    const int coeff = coeff_ptr[0];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
    const int64_t tmp = abs_coeff + ROUND_POWER_OF_TWO(round_ptr[0], 1);
    const int abs_qcoeff = (int)((tmp * quant) >> 15);
    qcoeff_ptr[0] = (tran_low_t)((abs_qcoeff ^ coeff_sign) - coeff_sign);
    dqcoeff_ptr[0] = qcoeff_ptr[0] * dequant / 2;
    if (abs_qcoeff) eob = 0;
  }

  *eob_ptr = eob + 1;
}
#endif

void vpx_quantize_b_c(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                      const struct macroblock_plane *const mb_plane,
                      tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                      const int16_t *dequant_ptr, uint16_t *eob_ptr,
                      const struct ScanOrder *const scan_order) {
  int i, non_zero_count = (int)n_coeffs, eob = -1;
  const int zbins[2] = { mb_plane->zbin[0], mb_plane->zbin[1] };
  const int nzbins[2] = { zbins[0] * -1, zbins[1] * -1 };
  const int16_t *round_ptr = mb_plane->round;
  const int16_t *quant_ptr = mb_plane->quant;
  const int16_t *quant_shift_ptr = mb_plane->quant_shift;
  const int16_t *scan = scan_order->scan;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  // Pre-scan pass
  for (i = (int)n_coeffs - 1; i >= 0; i--) {
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];

    if (coeff < zbins[rc != 0] && coeff > nzbins[rc != 0])
      non_zero_count--;
    else
      break;
  }

  // Quantization pass: All coefficients with index >= zero_flag are
  // skippable. Note: zero_flag can be zero.
  for (i = 0; i < non_zero_count; i++) {
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;

    if (abs_coeff >= zbins[rc != 0]) {
      int tmp = clamp(abs_coeff + round_ptr[rc != 0], INT16_MIN, INT16_MAX);
      tmp = ((((tmp * quant_ptr[rc != 0]) >> 16) + tmp) *
             quant_shift_ptr[rc != 0]) >>
            16;  // quantization
      qcoeff_ptr[rc] = (tmp ^ coeff_sign) - coeff_sign;
      dqcoeff_ptr[rc] = (tran_low_t)(qcoeff_ptr[rc] * dequant_ptr[rc != 0]);

      if (tmp) eob = i;
    }
  }
  *eob_ptr = eob + 1;
}

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_quantize_b_c(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                             const struct macroblock_plane *const mb_plane,
                             tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                             const int16_t *dequant_ptr, uint16_t *eob_ptr,
                             const struct ScanOrder *const scan_order) {
  int i, non_zero_count = (int)n_coeffs, eob = -1;
  const int zbins[2] = { mb_plane->zbin[0], mb_plane->zbin[1] };
  const int nzbins[2] = { zbins[0] * -1, zbins[1] * -1 };
  const int16_t *round_ptr = mb_plane->round;
  const int16_t *quant_ptr = mb_plane->quant;
  const int16_t *quant_shift_ptr = mb_plane->quant_shift;
  const int16_t *scan = scan_order->scan;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  // Pre-scan pass
  for (i = (int)n_coeffs - 1; i >= 0; i--) {
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];

    if (coeff < zbins[rc != 0] && coeff > nzbins[rc != 0])
      non_zero_count--;
    else
      break;
  }

  // Quantization pass: All coefficients with index >= zero_flag are
  // skippable. Note: zero_flag can be zero.
  for (i = 0; i < non_zero_count; i++) {
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;

    if (abs_coeff >= zbins[rc != 0]) {
      const int64_t tmp1 = abs_coeff + round_ptr[rc != 0];
      const int64_t tmp2 = ((tmp1 * quant_ptr[rc != 0]) >> 16) + tmp1;
      const int abs_qcoeff = (int)((tmp2 * quant_shift_ptr[rc != 0]) >> 16);
      qcoeff_ptr[rc] = (tran_low_t)((abs_qcoeff ^ coeff_sign) - coeff_sign);
      dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant_ptr[rc != 0];
      if (abs_qcoeff) eob = i;
    }
  }
  *eob_ptr = eob + 1;
}
#endif

void vpx_quantize_b_32x32_c(const tran_low_t *coeff_ptr,
                            const struct macroblock_plane *const mb_plane,
                            tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                            const int16_t *dequant_ptr, uint16_t *eob_ptr,
                            const struct ScanOrder *const scan_order) {
  const int n_coeffs = 32 * 32;
  const int zbins[2] = { ROUND_POWER_OF_TWO(mb_plane->zbin[0], 1),
                         ROUND_POWER_OF_TWO(mb_plane->zbin[1], 1) };
  const int nzbins[2] = { zbins[0] * -1, zbins[1] * -1 };
  const int16_t *round_ptr = mb_plane->round;
  const int16_t *quant_ptr = mb_plane->quant;
  const int16_t *quant_shift_ptr = mb_plane->quant_shift;
  const int16_t *scan = scan_order->scan;

  int idx = 0;
  int idx_arr[32 * 32 /* n_coeffs */];
  int i, eob = -1;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  // Pre-scan pass
  for (i = 0; i < n_coeffs; i++) {
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];

    // If the coefficient is out of the base ZBIN range, keep it for
    // quantization.
    if (coeff >= zbins[rc != 0] || coeff <= nzbins[rc != 0]) idx_arr[idx++] = i;
  }

  // Quantization pass: only process the coefficients selected in
  // pre-scan pass. Note: idx can be zero.
  for (i = 0; i < idx; i++) {
    const int rc = scan[idx_arr[i]];
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = (coeff >> 31);
    int tmp;
    int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
    abs_coeff += ROUND_POWER_OF_TWO(round_ptr[rc != 0], 1);
    abs_coeff = clamp(abs_coeff, INT16_MIN, INT16_MAX);
    tmp = ((((abs_coeff * quant_ptr[rc != 0]) >> 16) + abs_coeff) *
           quant_shift_ptr[rc != 0]) >>
          15;

    qcoeff_ptr[rc] = (tmp ^ coeff_sign) - coeff_sign;
#if (VPX_ARCH_X86 || VPX_ARCH_X86_64) && !CONFIG_VP9_HIGHBITDEPTH
    // When tran_low_t is only 16 bits dqcoeff can outrange it. Rather than
    // truncating with a cast, saturate the value. This is easier to implement
    // on x86 and preserves the sign of the value.
    dqcoeff_ptr[rc] =
        clamp(qcoeff_ptr[rc] * dequant_ptr[rc != 0] / 2, INT16_MIN, INT16_MAX);
#else
    dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant_ptr[rc != 0] / 2;
#endif  // VPX_ARCH_X86 && CONFIG_VP9_HIGHBITDEPTH

    if (tmp) eob = idx_arr[i];
  }
  *eob_ptr = eob + 1;
}

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_quantize_b_32x32_c(
    const tran_low_t *coeff_ptr, const struct macroblock_plane *const mb_plane,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,
    uint16_t *eob_ptr, const struct ScanOrder *const scan_order) {
  const intptr_t n_coeffs = 32 * 32;
  const int zbins[2] = { ROUND_POWER_OF_TWO(mb_plane->zbin[0], 1),
                         ROUND_POWER_OF_TWO(mb_plane->zbin[1], 1) };
  const int nzbins[2] = { zbins[0] * -1, zbins[1] * -1 };
  const int16_t *round_ptr = mb_plane->round;
  const int16_t *quant_ptr = mb_plane->quant;
  const int16_t *quant_shift_ptr = mb_plane->quant_shift;
  const int16_t *scan = scan_order->scan;

  int idx = 0;
  int idx_arr[1024];
  int i, eob = -1;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  // Pre-scan pass
  for (i = 0; i < n_coeffs; i++) {
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];

    // If the coefficient is out of the base ZBIN range, keep it for
    // quantization.
    if (coeff >= zbins[rc != 0] || coeff <= nzbins[rc != 0]) idx_arr[idx++] = i;
  }

  // Quantization pass: only process the coefficients selected in
  // pre-scan pass. Note: idx can be zero.
  for (i = 0; i < idx; i++) {
    const int rc = scan[idx_arr[i]];
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
    const int64_t tmp1 = abs_coeff + ROUND_POWER_OF_TWO(round_ptr[rc != 0], 1);
    const int64_t tmp2 = ((tmp1 * quant_ptr[rc != 0]) >> 16) + tmp1;
    const int abs_qcoeff = (int)((tmp2 * quant_shift_ptr[rc != 0]) >> 15);
    qcoeff_ptr[rc] = (tran_low_t)((abs_qcoeff ^ coeff_sign) - coeff_sign);
    dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant_ptr[rc != 0] / 2;
    if (abs_qcoeff) eob = idx_arr[i];
  }
  *eob_ptr = eob + 1;
}
#endif
