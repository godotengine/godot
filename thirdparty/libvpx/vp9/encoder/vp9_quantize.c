/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <math.h>
#include "./vpx_dsp_rtcd.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/bitops.h"
#include "vpx_ports/mem.h"

#include "vp9/common/vp9_quant_common.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/common/vp9_seg_common.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_quantize.h"
#include "vp9/encoder/vp9_rd.h"

void vp9_quantize_fp_c(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                       const struct macroblock_plane *const mb_plane,
                       tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                       const int16_t *dequant_ptr, uint16_t *eob_ptr,
                       const struct ScanOrder *const scan_order) {
  int i, eob = -1;
  const int16_t *round_ptr = mb_plane->round_fp;
  const int16_t *quant_ptr = mb_plane->quant_fp;
  const int16_t *scan = scan_order->scan;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  // Quantization pass: All coefficients with index >= zero_flag are
  // skippable. Note: zero_flag can be zero.
  for (i = 0; i < n_coeffs; i++) {
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;

    int tmp = clamp(abs_coeff + round_ptr[rc != 0], INT16_MIN, INT16_MAX);
    tmp = (tmp * quant_ptr[rc != 0]) >> 16;

    qcoeff_ptr[rc] = (tmp ^ coeff_sign) - coeff_sign;
    dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant_ptr[rc != 0];

    if (tmp) eob = i;
  }
  *eob_ptr = eob + 1;
}

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_quantize_fp_c(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                              const struct macroblock_plane *const mb_plane,
                              tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                              const int16_t *dequant_ptr, uint16_t *eob_ptr,
                              const struct ScanOrder *const scan_order) {
  int i;
  int eob = -1;
  const int16_t *round_ptr = mb_plane->round_fp;
  const int16_t *quant_ptr = mb_plane->quant_fp;
  const int16_t *scan = scan_order->scan;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  // Quantization pass: All coefficients with index >= zero_flag are
  // skippable. Note: zero_flag can be zero.
  for (i = 0; i < n_coeffs; i++) {
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
    const int64_t tmp = abs_coeff + round_ptr[rc != 0];
    const int abs_qcoeff = (int)((tmp * quant_ptr[rc != 0]) >> 16);
    qcoeff_ptr[rc] = (tran_low_t)(abs_qcoeff ^ coeff_sign) - coeff_sign;
    dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant_ptr[rc != 0];
    if (abs_qcoeff) eob = i;
  }
  *eob_ptr = eob + 1;
}
#endif

// TODO(jingning) Refactor this file and combine functions with similar
// operations.
void vp9_quantize_fp_32x32_c(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                             const struct macroblock_plane *const mb_plane,
                             tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                             const int16_t *dequant_ptr, uint16_t *eob_ptr,
                             const struct ScanOrder *const scan_order) {
  int i, eob = -1;
  const int16_t *round_ptr = mb_plane->round_fp;
  const int16_t *quant_ptr = mb_plane->quant_fp;
  const int16_t *scan = scan_order->scan;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  for (i = 0; i < n_coeffs; i++) {
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = (coeff >> 31);
    int tmp = 0;
    int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;

    if (abs_coeff >= (dequant_ptr[rc != 0] >> 2)) {
      abs_coeff += ROUND_POWER_OF_TWO(round_ptr[rc != 0], 1);
      abs_coeff = clamp(abs_coeff, INT16_MIN, INT16_MAX);
      tmp = (abs_coeff * quant_ptr[rc != 0]) >> 15;
      qcoeff_ptr[rc] = (tmp ^ coeff_sign) - coeff_sign;
      dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant_ptr[rc != 0] / 2;
    }

    if (tmp) eob = i;
  }
  *eob_ptr = eob + 1;
}

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_quantize_fp_32x32_c(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs,
    const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const struct ScanOrder *const scan_order) {
  int i, eob = -1;
  const int16_t *round_ptr = mb_plane->round_fp;
  const int16_t *quant_ptr = mb_plane->quant_fp;
  const int16_t *scan = scan_order->scan;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  for (i = 0; i < n_coeffs; i++) {
    int abs_qcoeff = 0;
    const int rc = scan[i];
    const int coeff = coeff_ptr[rc];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;

    if (abs_coeff >= (dequant_ptr[rc != 0] >> 2)) {
      const int64_t tmp = abs_coeff + ROUND_POWER_OF_TWO(round_ptr[rc != 0], 1);
      abs_qcoeff = (int)((tmp * quant_ptr[rc != 0]) >> 15);
      qcoeff_ptr[rc] = (tran_low_t)((abs_qcoeff ^ coeff_sign) - coeff_sign);
      dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant_ptr[rc != 0] / 2;
    }

    if (abs_qcoeff) eob = i;
  }
  *eob_ptr = eob + 1;
}
#endif

static void invert_quant(int16_t *quant, int16_t *shift, int d) {
  unsigned int t;
  int l, m;
  t = (unsigned int)d;
  l = get_msb(t);
  m = 1 + (1 << (16 + l)) / d;
  *quant = (int16_t)(m - (1 << 16));
  *shift = 1 << (16 - l);
}

static int get_qzbin_factor(int q, vpx_bit_depth_t bit_depth) {
  const int quant = vp9_dc_quant(q, 0, bit_depth);
#if CONFIG_VP9_HIGHBITDEPTH
  switch (bit_depth) {
    case VPX_BITS_8: return q == 0 ? 64 : (quant < 148 ? 84 : 80);
    case VPX_BITS_10: return q == 0 ? 64 : (quant < 592 ? 84 : 80);
    default:
      assert(bit_depth == VPX_BITS_12);
      return q == 0 ? 64 : (quant < 2368 ? 84 : 80);
  }
#else
  (void)bit_depth;
  return q == 0 ? 64 : (quant < 148 ? 84 : 80);
#endif
}

void vp9_init_quantizer(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  QUANTS *const quants = &cpi->quants;
  int i, q, quant;

  for (q = 0; q < QINDEX_RANGE; q++) {
    int qzbin_factor = get_qzbin_factor(q, cm->bit_depth);
    int qrounding_factor = q == 0 ? 64 : 48;
    const int sharpness_adjustment = 16 * (7 - cpi->oxcf.sharpness) / 7;

    if (cpi->oxcf.sharpness > 0 && q > 0) {
      qzbin_factor = 64 + sharpness_adjustment;
      qrounding_factor = 64 - sharpness_adjustment;
    }

    for (i = 0; i < 2; ++i) {
      int qrounding_factor_fp = i == 0 ? 48 : 42;
      if (q == 0) qrounding_factor_fp = 64;
      if (cpi->oxcf.sharpness > 0)
        qrounding_factor_fp = 64 - sharpness_adjustment;
      // y
      quant = i == 0 ? vp9_dc_quant(q, cm->y_dc_delta_q, cm->bit_depth)
                     : vp9_ac_quant(q, 0, cm->bit_depth);
      invert_quant(&quants->y_quant[q][i], &quants->y_quant_shift[q][i], quant);
      quants->y_quant_fp[q][i] = (1 << 16) / quant;
      quants->y_round_fp[q][i] = (qrounding_factor_fp * quant) >> 7;
      quants->y_zbin[q][i] = ROUND_POWER_OF_TWO(qzbin_factor * quant, 7);
      quants->y_round[q][i] = (qrounding_factor * quant) >> 7;
      cpi->y_dequant[q][i] = quant;

      // uv
      quant = i == 0 ? vp9_dc_quant(q, cm->uv_dc_delta_q, cm->bit_depth)
                     : vp9_ac_quant(q, cm->uv_ac_delta_q, cm->bit_depth);
      invert_quant(&quants->uv_quant[q][i], &quants->uv_quant_shift[q][i],
                   quant);
      quants->uv_quant_fp[q][i] = (1 << 16) / quant;
      quants->uv_round_fp[q][i] = (qrounding_factor_fp * quant) >> 7;
      quants->uv_zbin[q][i] = ROUND_POWER_OF_TWO(qzbin_factor * quant, 7);
      quants->uv_round[q][i] = (qrounding_factor * quant) >> 7;
      cpi->uv_dequant[q][i] = quant;
    }

    for (i = 2; i < 8; i++) {
      quants->y_quant[q][i] = quants->y_quant[q][1];
      quants->y_quant_fp[q][i] = quants->y_quant_fp[q][1];
      quants->y_round_fp[q][i] = quants->y_round_fp[q][1];
      quants->y_quant_shift[q][i] = quants->y_quant_shift[q][1];
      quants->y_zbin[q][i] = quants->y_zbin[q][1];
      quants->y_round[q][i] = quants->y_round[q][1];
      cpi->y_dequant[q][i] = cpi->y_dequant[q][1];

      quants->uv_quant[q][i] = quants->uv_quant[q][1];
      quants->uv_quant_fp[q][i] = quants->uv_quant_fp[q][1];
      quants->uv_round_fp[q][i] = quants->uv_round_fp[q][1];
      quants->uv_quant_shift[q][i] = quants->uv_quant_shift[q][1];
      quants->uv_zbin[q][i] = quants->uv_zbin[q][1];
      quants->uv_round[q][i] = quants->uv_round[q][1];
      cpi->uv_dequant[q][i] = cpi->uv_dequant[q][1];
    }
  }
}

void vp9_init_plane_quantizers(VP9_COMP *cpi, MACROBLOCK *x) {
  const VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  QUANTS *const quants = &cpi->quants;
  const int segment_id = xd->mi[0]->segment_id;
  const int qindex = vp9_get_qindex(&cm->seg, segment_id, cm->base_qindex);
  const int rdmult = vp9_compute_rd_mult(cpi, qindex + cm->y_dc_delta_q);
  int i;

  // Y
  x->plane[0].quant = quants->y_quant[qindex];
  x->plane[0].quant_fp = quants->y_quant_fp[qindex];
  x->plane[0].round_fp = quants->y_round_fp[qindex];
  x->plane[0].quant_shift = quants->y_quant_shift[qindex];
  x->plane[0].zbin = quants->y_zbin[qindex];
  x->plane[0].round = quants->y_round[qindex];
  xd->plane[0].dequant = cpi->y_dequant[qindex];
  x->plane[0].quant_thred[0] = x->plane[0].zbin[0] * x->plane[0].zbin[0];
  x->plane[0].quant_thred[1] = x->plane[0].zbin[1] * x->plane[0].zbin[1];

  // UV
  for (i = 1; i < 3; i++) {
    x->plane[i].quant = quants->uv_quant[qindex];
    x->plane[i].quant_fp = quants->uv_quant_fp[qindex];
    x->plane[i].round_fp = quants->uv_round_fp[qindex];
    x->plane[i].quant_shift = quants->uv_quant_shift[qindex];
    x->plane[i].zbin = quants->uv_zbin[qindex];
    x->plane[i].round = quants->uv_round[qindex];
    xd->plane[i].dequant = cpi->uv_dequant[qindex];
    x->plane[i].quant_thred[0] = x->plane[i].zbin[0] * x->plane[i].zbin[0];
    x->plane[i].quant_thred[1] = x->plane[i].zbin[1] * x->plane[i].zbin[1];
  }

  x->skip_block = segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP);
  x->q_index = qindex;

  set_error_per_bit(x, rdmult);

  vp9_initialize_me_consts(cpi, x, x->q_index);
}

void vp9_frame_init_quantizer(VP9_COMP *cpi) {
  vp9_init_plane_quantizers(cpi, &cpi->td.mb);
}

void vp9_set_quantizer(VP9_COMP *cpi, int q, int ext_rc_delta_q_uv) {
  VP9_COMMON *cm = &cpi->common;
  // quantizer has to be reinitialized with vp9_init_quantizer() if any
  // delta_q changes.
  cm->base_qindex = q;
  cm->y_dc_delta_q = 0;
  cm->uv_dc_delta_q = 0;
  cm->uv_ac_delta_q = 0;

  if (ext_rc_delta_q_uv != 0) {
    cm->uv_dc_delta_q = cm->uv_ac_delta_q = ext_rc_delta_q_uv;
    vp9_init_quantizer(cpi);
    return;
  }

  if (cpi->oxcf.delta_q_uv != 0) {
    cm->uv_dc_delta_q = cm->uv_ac_delta_q = cpi->oxcf.delta_q_uv;
    vp9_init_quantizer(cpi);
  }
}

// Table that converts 0-63 Q-range values passed in outside to the Qindex
// range used internally.
static const int quantizer_to_qindex[] = {
  0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  44,  48,
  52,  56,  60,  64,  68,  72,  76,  80,  84,  88,  92,  96,  100,
  104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152,
  156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
  208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 249, 255,
};

int vp9_quantizer_to_qindex(int quantizer) {
  return quantizer_to_qindex[quantizer];
}

int vp9_qindex_to_quantizer(int qindex) {
  int quantizer;

  for (quantizer = 0; quantizer < 64; ++quantizer)
    if (quantizer_to_qindex[quantizer] >= qindex) return quantizer;

  return 63;
}
