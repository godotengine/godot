/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"

static VPX_FORCE_INLINE void init_one_qp(const __m128i *p, __m256i *qp) {
  const __m128i sign = _mm_srai_epi16(*p, 15);
  const __m128i dc = _mm_unpacklo_epi16(*p, sign);
  const __m128i ac = _mm_unpackhi_epi16(*p, sign);
  *qp = _mm256_insertf128_si256(_mm256_castsi128_si256(dc), ac, 1);
}

static VPX_FORCE_INLINE void update_qp(__m256i *qp) {
  int i;
  for (i = 0; i < 5; ++i) {
    qp[i] = _mm256_permute2x128_si256(qp[i], qp[i], 0x11);
  }
}

static VPX_FORCE_INLINE void init_qp(
    const struct macroblock_plane *const mb_plane, const int16_t *dequant_ptr,
    __m256i *qp, int log_scale) {
  const __m128i zbin = _mm_loadu_si128((const __m128i *)mb_plane->zbin);
  const __m128i round = _mm_loadu_si128((const __m128i *)mb_plane->round);
  const __m128i quant = _mm_loadu_si128((const __m128i *)mb_plane->quant);
  const __m128i dequant = _mm_loadu_si128((const __m128i *)dequant_ptr);
  const __m128i quant_shift =
      _mm_loadu_si128((const __m128i *)mb_plane->quant_shift);
  init_one_qp(&zbin, &qp[0]);
  init_one_qp(&round, &qp[1]);
  init_one_qp(&quant, &qp[2]);
  init_one_qp(&dequant, &qp[3]);
  init_one_qp(&quant_shift, &qp[4]);
  if (log_scale > 0) {
    const __m256i rnd = _mm256_set1_epi32((int16_t)(1 << (log_scale - 1)));
    qp[0] = _mm256_add_epi32(qp[0], rnd);
    qp[0] = _mm256_srai_epi32(qp[0], log_scale);

    qp[1] = _mm256_add_epi32(qp[1], rnd);
    qp[1] = _mm256_srai_epi32(qp[1], log_scale);
  }
  // Subtracting 1 here eliminates a _mm256_cmpeq_epi32() instruction when
  // calculating the zbin mask.
  qp[0] = _mm256_sub_epi32(qp[0], _mm256_set1_epi32(1));
}

// Note:
// *x is vector multiplied by *y which is 16 int32_t parallel multiplication
// and right shift 16.  The output, 16 int32_t is save in *p.
static VPX_FORCE_INLINE __m256i mm256_mul_shift_epi32(const __m256i *x,
                                                      const __m256i *y) {
  __m256i prod_lo = _mm256_mul_epi32(*x, *y);
  __m256i prod_hi = _mm256_srli_epi64(*x, 32);
  const __m256i mult_hi = _mm256_srli_epi64(*y, 32);
  const __m256i mask = _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1);
  prod_hi = _mm256_mul_epi32(prod_hi, mult_hi);
  prod_lo = _mm256_srli_epi64(prod_lo, 16);
  prod_lo = _mm256_and_si256(prod_lo, mask);
  prod_hi = _mm256_srli_epi64(prod_hi, 16);
  prod_hi = _mm256_slli_epi64(prod_hi, 32);
  return _mm256_or_si256(prod_lo, prod_hi);
}

static VPX_FORCE_INLINE __m256i get_max_lane_eob(const int16_t *iscan_ptr,
                                                 __m256i eobmax,
                                                 __m256i nz_mask) {
  const __m256i packed_nz_mask = _mm256_packs_epi32(nz_mask, nz_mask);
  const __m256i packed_nz_mask_perm =
      _mm256_permute4x64_epi64(packed_nz_mask, 0xD8);
  const __m256i iscan =
      _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)iscan_ptr));
  const __m256i nz_iscan = _mm256_and_si256(iscan, packed_nz_mask_perm);
  return _mm256_max_epi16(eobmax, nz_iscan);
}

// Get the max eob from the lower 128 bits.
static VPX_FORCE_INLINE uint16_t get_max_eob(__m256i eob) {
  __m256i eob_s;
  eob_s = _mm256_shuffle_epi32(eob, 0xe);
  eob = _mm256_max_epi16(eob, eob_s);
  eob_s = _mm256_shufflelo_epi16(eob, 0xe);
  eob = _mm256_max_epi16(eob, eob_s);
  eob_s = _mm256_shufflelo_epi16(eob, 1);
  eob = _mm256_max_epi16(eob, eob_s);
#if defined(_MSC_VER) && (_MSC_VER < 1910)
  return _mm_cvtsi128_si32(_mm256_extracti128_si256(eob, 0)) & 0xffff;
#else
  return (uint16_t)_mm256_extract_epi16(eob, 0);
#endif
}

static VPX_FORCE_INLINE void quantize(const __m256i *qp,
                                      const tran_low_t *coeff_ptr,
                                      const int16_t *iscan_ptr,
                                      tran_low_t *qcoeff, tran_low_t *dqcoeff,
                                      __m256i *eob) {
  const __m256i coeff = _mm256_loadu_si256((const __m256i *)coeff_ptr);
  const __m256i abs_coeff = _mm256_abs_epi32(coeff);
  const __m256i zbin_mask = _mm256_cmpgt_epi32(abs_coeff, qp[0]);

  if (_mm256_movemask_epi8(zbin_mask) == 0) {
    const __m256i zero = _mm256_setzero_si256();
    _mm256_storeu_si256((__m256i *)qcoeff, zero);
    _mm256_storeu_si256((__m256i *)dqcoeff, zero);
    return;
  }
  {
    const __m256i tmp_rnd =
        _mm256_and_si256(_mm256_add_epi32(abs_coeff, qp[1]), zbin_mask);
    const __m256i tmp = mm256_mul_shift_epi32(&tmp_rnd, &qp[2]);
    const __m256i tmp2 = _mm256_add_epi32(tmp, tmp_rnd);
    const __m256i abs_q = mm256_mul_shift_epi32(&tmp2, &qp[4]);
    const __m256i abs_dq = _mm256_mullo_epi32(abs_q, qp[3]);
    const __m256i nz_mask = _mm256_cmpgt_epi32(abs_q, _mm256_setzero_si256());
    const __m256i q = _mm256_sign_epi32(abs_q, coeff);
    const __m256i dq = _mm256_sign_epi32(abs_dq, coeff);

    _mm256_storeu_si256((__m256i *)qcoeff, q);
    _mm256_storeu_si256((__m256i *)dqcoeff, dq);

    *eob = get_max_lane_eob(iscan_ptr, *eob, nz_mask);
  }
}

void vpx_highbd_quantize_b_avx2(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                                const struct macroblock_plane *const mb_plane,
                                tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                                const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                const struct ScanOrder *const scan_order) {
  const int step = 8;
  __m256i eob = _mm256_setzero_si256();
  __m256i qp[5];
  const int16_t *iscan = scan_order->iscan;

  init_qp(mb_plane, dequant_ptr, qp, 0);

  quantize(qp, coeff_ptr, iscan, qcoeff_ptr, dqcoeff_ptr, &eob);

  coeff_ptr += step;
  qcoeff_ptr += step;
  dqcoeff_ptr += step;
  iscan += step;
  n_coeffs -= step;

  update_qp(qp);

  while (n_coeffs > 0) {
    quantize(qp, coeff_ptr, iscan, qcoeff_ptr, dqcoeff_ptr, &eob);

    coeff_ptr += step;
    qcoeff_ptr += step;
    dqcoeff_ptr += step;
    iscan += step;
    n_coeffs -= step;
  }

  *eob_ptr = get_max_eob(eob);
}

static VPX_FORCE_INLINE __m256i mm256_mul_shift_epi32_logscale(const __m256i *x,
                                                               const __m256i *y,
                                                               int log_scale) {
  __m256i prod_lo = _mm256_mul_epi32(*x, *y);
  __m256i prod_hi = _mm256_srli_epi64(*x, 32);
  const __m256i mult_hi = _mm256_srli_epi64(*y, 32);
  const __m256i mask = _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1);
  prod_hi = _mm256_mul_epi32(prod_hi, mult_hi);
  prod_lo = _mm256_srli_epi64(prod_lo, 16 - log_scale);
  prod_lo = _mm256_and_si256(prod_lo, mask);
  prod_hi = _mm256_srli_epi64(prod_hi, 16 - log_scale);
  prod_hi = _mm256_slli_epi64(prod_hi, 32);
  return _mm256_or_si256(prod_lo, prod_hi);
}

static VPX_FORCE_INLINE void quantize_b_32x32(
    const __m256i *qp, const tran_low_t *coeff_ptr, const int16_t *iscan_ptr,
    tran_low_t *qcoeff, tran_low_t *dqcoeff, __m256i *eob) {
  const __m256i coeff = _mm256_loadu_si256((const __m256i *)coeff_ptr);
  const __m256i abs_coeff = _mm256_abs_epi32(coeff);
  const __m256i zbin_mask = _mm256_cmpgt_epi32(abs_coeff, qp[0]);

  if (_mm256_movemask_epi8(zbin_mask) == 0) {
    const __m256i zero = _mm256_setzero_si256();
    _mm256_storeu_si256((__m256i *)qcoeff, zero);
    _mm256_storeu_si256((__m256i *)dqcoeff, zero);
    return;
  }

  {
    const __m256i tmp_rnd =
        _mm256_and_si256(_mm256_add_epi32(abs_coeff, qp[1]), zbin_mask);
    // const int64_t tmp2 = ((tmpw * quant_ptr[rc != 0]) >> 16) + tmpw;
    const __m256i tmp = mm256_mul_shift_epi32_logscale(&tmp_rnd, &qp[2], 0);
    const __m256i tmp2 = _mm256_add_epi32(tmp, tmp_rnd);
    // const int abs_qcoeff = (int)((tmp2 * quant_shift_ptr[rc != 0]) >> 15);
    const __m256i abs_q = mm256_mul_shift_epi32_logscale(&tmp2, &qp[4], 1);
    const __m256i abs_dq =
        _mm256_srli_epi32(_mm256_mullo_epi32(abs_q, qp[3]), 1);
    const __m256i nz_mask = _mm256_cmpgt_epi32(abs_q, _mm256_setzero_si256());
    const __m256i q = _mm256_sign_epi32(abs_q, coeff);
    const __m256i dq = _mm256_sign_epi32(abs_dq, coeff);

    _mm256_storeu_si256((__m256i *)qcoeff, q);
    _mm256_storeu_si256((__m256i *)dqcoeff, dq);

    *eob = get_max_lane_eob(iscan_ptr, *eob, nz_mask);
  }
}

void vpx_highbd_quantize_b_32x32_avx2(
    const tran_low_t *coeff_ptr, const struct macroblock_plane *const mb_plane,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,
    uint16_t *eob_ptr, const struct ScanOrder *const scan_order) {
  const unsigned int step = 8;
  intptr_t n_coeffs = 32 * 32;
  const int16_t *iscan = scan_order->iscan;
  __m256i eob = _mm256_setzero_si256();
  __m256i qp[5];

  init_qp(mb_plane, dequant_ptr, qp, 1);

  quantize_b_32x32(qp, coeff_ptr, iscan, qcoeff_ptr, dqcoeff_ptr, &eob);

  coeff_ptr += step;
  qcoeff_ptr += step;
  dqcoeff_ptr += step;
  iscan += step;
  n_coeffs -= step;

  update_qp(qp);

  while (n_coeffs > 0) {
    quantize_b_32x32(qp, coeff_ptr, iscan, qcoeff_ptr, dqcoeff_ptr, &eob);

    coeff_ptr += step;
    qcoeff_ptr += step;
    dqcoeff_ptr += step;
    iscan += step;
    n_coeffs -= step;
  }

  *eob_ptr = get_max_eob(eob);
}
