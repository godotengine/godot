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
#include <immintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"

static VPX_FORCE_INLINE void load_b_values_avx2(
    const struct macroblock_plane *mb_plane, __m256i *zbin, __m256i *round,
    __m256i *quant, const int16_t *dequant_ptr, __m256i *dequant,
    __m256i *shift, int log_scale) {
  *zbin =
      _mm256_castsi128_si256(_mm_load_si128((const __m128i *)mb_plane->zbin));
  *zbin = _mm256_permute4x64_epi64(*zbin, 0x54);
  if (log_scale > 0) {
    const __m256i rnd = _mm256_set1_epi16((int16_t)(1 << (log_scale - 1)));
    *zbin = _mm256_add_epi16(*zbin, rnd);
    *zbin = _mm256_srai_epi16(*zbin, log_scale);
  }
  // Subtracting 1 here eliminates a _mm256_cmpeq_epi16() instruction when
  // calculating the zbin mask. (See quantize_b_logscale{0,1,2}_16)
  *zbin = _mm256_sub_epi16(*zbin, _mm256_set1_epi16(1));

  *round =
      _mm256_castsi128_si256(_mm_load_si128((const __m128i *)mb_plane->round));
  *round = _mm256_permute4x64_epi64(*round, 0x54);
  if (log_scale > 0) {
    const __m256i rnd = _mm256_set1_epi16((int16_t)(1 << (log_scale - 1)));
    *round = _mm256_add_epi16(*round, rnd);
    *round = _mm256_srai_epi16(*round, log_scale);
  }

  *quant =
      _mm256_castsi128_si256(_mm_load_si128((const __m128i *)mb_plane->quant));
  *quant = _mm256_permute4x64_epi64(*quant, 0x54);
  *dequant =
      _mm256_castsi128_si256(_mm_load_si128((const __m128i *)dequant_ptr));
  *dequant = _mm256_permute4x64_epi64(*dequant, 0x54);
  *shift = _mm256_castsi128_si256(
      _mm_load_si128((const __m128i *)mb_plane->quant_shift));
  *shift = _mm256_permute4x64_epi64(*shift, 0x54);
}

static VPX_FORCE_INLINE __m256i
load_coefficients_avx2(const tran_low_t *coeff_ptr) {
#if CONFIG_VP9_HIGHBITDEPTH
  // typedef int32_t tran_low_t;
  const __m256i coeff1 = _mm256_loadu_si256((const __m256i *)coeff_ptr);
  const __m256i coeff2 = _mm256_loadu_si256((const __m256i *)(coeff_ptr + 8));
  return _mm256_packs_epi32(coeff1, coeff2);
#else
  // typedef int16_t tran_low_t;
  return _mm256_loadu_si256((const __m256i *)coeff_ptr);
#endif
}

static VPX_FORCE_INLINE void store_coefficients_avx2(__m256i coeff_vals,
                                                     tran_low_t *coeff_ptr) {
#if CONFIG_VP9_HIGHBITDEPTH
  // typedef int32_t tran_low_t;
  __m256i coeff_sign = _mm256_srai_epi16(coeff_vals, 15);
  __m256i coeff_vals_lo = _mm256_unpacklo_epi16(coeff_vals, coeff_sign);
  __m256i coeff_vals_hi = _mm256_unpackhi_epi16(coeff_vals, coeff_sign);
  _mm256_storeu_si256((__m256i *)coeff_ptr, coeff_vals_lo);
  _mm256_storeu_si256((__m256i *)(coeff_ptr + 8), coeff_vals_hi);
#else
  // typedef int16_t tran_low_t;
  _mm256_storeu_si256((__m256i *)coeff_ptr, coeff_vals);
#endif
}

static VPX_FORCE_INLINE __m256i
quantize_b_16(const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
              tran_low_t *dqcoeff_ptr, __m256i *v_quant, __m256i *v_dequant,
              __m256i *v_round, __m256i *v_zbin, __m256i *v_quant_shift) {
  const __m256i v_coeff = load_coefficients_avx2(coeff_ptr);
  const __m256i v_abs_coeff = _mm256_abs_epi16(v_coeff);
  const __m256i v_zbin_mask = _mm256_cmpgt_epi16(v_abs_coeff, *v_zbin);

  if (_mm256_movemask_epi8(v_zbin_mask) == 0) {
    _mm256_storeu_si256((__m256i *)qcoeff_ptr, _mm256_setzero_si256());
    _mm256_storeu_si256((__m256i *)dqcoeff_ptr, _mm256_setzero_si256());
#if CONFIG_VP9_HIGHBITDEPTH
    _mm256_store_si256((__m256i *)(qcoeff_ptr + 8), _mm256_setzero_si256());
    _mm256_store_si256((__m256i *)(dqcoeff_ptr + 8), _mm256_setzero_si256());
#endif  // CONFIG_VP9_HIGHBITDEPTH
    return _mm256_setzero_si256();
  }
  {
    // tmp = v_zbin_mask ? (int64_t)abs_coeff + log_scaled_round : 0
    const __m256i v_tmp_rnd =
        _mm256_and_si256(_mm256_adds_epi16(v_abs_coeff, *v_round), v_zbin_mask);

    const __m256i v_tmp32_a = _mm256_mulhi_epi16(v_tmp_rnd, *v_quant);
    const __m256i v_tmp32_b = _mm256_add_epi16(v_tmp32_a, v_tmp_rnd);
    const __m256i v_tmp32 = _mm256_mulhi_epi16(v_tmp32_b, *v_quant_shift);
    const __m256i v_nz_mask =
        _mm256_cmpgt_epi16(v_tmp32, _mm256_setzero_si256());
    const __m256i v_qcoeff = _mm256_sign_epi16(v_tmp32, v_coeff);
#if CONFIG_VP9_HIGHBITDEPTH
    const __m256i low = _mm256_mullo_epi16(v_qcoeff, *v_dequant);
    const __m256i high = _mm256_mulhi_epi16(v_qcoeff, *v_dequant);

    const __m256i v_dqcoeff_lo = _mm256_unpacklo_epi16(low, high);
    const __m256i v_dqcoeff_hi = _mm256_unpackhi_epi16(low, high);
#else
    const __m256i v_dqcoeff = _mm256_mullo_epi16(v_qcoeff, *v_dequant);
#endif

    store_coefficients_avx2(v_qcoeff, qcoeff_ptr);
#if CONFIG_VP9_HIGHBITDEPTH
    _mm256_storeu_si256((__m256i *)(dqcoeff_ptr), v_dqcoeff_lo);
    _mm256_storeu_si256((__m256i *)(dqcoeff_ptr + 8), v_dqcoeff_hi);
#else
    store_coefficients_avx2(v_dqcoeff, dqcoeff_ptr);
#endif
    return v_nz_mask;
  }
}

static VPX_FORCE_INLINE __m256i get_max_lane_eob(const int16_t *iscan,
                                                 __m256i v_eobmax,
                                                 __m256i v_mask) {
#if CONFIG_VP9_HIGHBITDEPTH
  const __m256i v_iscan = _mm256_permute4x64_epi64(
      _mm256_loadu_si256((const __m256i *)iscan), 0xD8);
#else
  const __m256i v_iscan = _mm256_loadu_si256((const __m256i *)iscan);
#endif
  const __m256i v_nz_iscan = _mm256_and_si256(v_iscan, v_mask);
  return _mm256_max_epi16(v_eobmax, v_nz_iscan);
}

static VPX_FORCE_INLINE int16_t accumulate_eob256(__m256i eob256) {
  const __m128i eob_lo = _mm256_castsi256_si128(eob256);
  const __m128i eob_hi = _mm256_extractf128_si256(eob256, 1);
  __m128i eob = _mm_max_epi16(eob_lo, eob_hi);
  __m128i eob_shuffled = _mm_shuffle_epi32(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0x1);
  eob = _mm_max_epi16(eob, eob_shuffled);
  return _mm_extract_epi16(eob, 1);
}

void vpx_quantize_b_avx2(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                         const struct macroblock_plane *const mb_plane,
                         tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                         const int16_t *dequant_ptr, uint16_t *eob_ptr,
                         const struct ScanOrder *const scan_order) {
  __m256i v_zbin, v_round, v_quant, v_dequant, v_quant_shift, v_nz_mask;
  __m256i v_eobmax = _mm256_setzero_si256();
  intptr_t count;
  const int16_t *iscan = scan_order->iscan;

  load_b_values_avx2(mb_plane, &v_zbin, &v_round, &v_quant, dequant_ptr,
                     &v_dequant, &v_quant_shift, 0);
  // Do DC and first 15 AC.
  v_nz_mask = quantize_b_16(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, &v_quant,
                            &v_dequant, &v_round, &v_zbin, &v_quant_shift);

  v_eobmax = get_max_lane_eob(iscan, v_eobmax, v_nz_mask);

  v_round = _mm256_unpackhi_epi64(v_round, v_round);
  v_quant = _mm256_unpackhi_epi64(v_quant, v_quant);
  v_dequant = _mm256_unpackhi_epi64(v_dequant, v_dequant);
  v_quant_shift = _mm256_unpackhi_epi64(v_quant_shift, v_quant_shift);
  v_zbin = _mm256_unpackhi_epi64(v_zbin, v_zbin);

  for (count = n_coeffs - 16; count > 0; count -= 16) {
    coeff_ptr += 16;
    qcoeff_ptr += 16;
    dqcoeff_ptr += 16;
    iscan += 16;
    v_nz_mask = quantize_b_16(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, &v_quant,
                              &v_dequant, &v_round, &v_zbin, &v_quant_shift);

    v_eobmax = get_max_lane_eob(iscan, v_eobmax, v_nz_mask);
  }

  *eob_ptr = accumulate_eob256(v_eobmax);
}

static VPX_FORCE_INLINE __m256i quantize_b_32x32_16(
    const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *iscan, __m256i *v_quant,
    __m256i *v_dequant, __m256i *v_round, __m256i *v_zbin,
    __m256i *v_quant_shift, __m256i *v_eobmax) {
  const __m256i v_coeff = load_coefficients_avx2(coeff_ptr);
  const __m256i v_abs_coeff = _mm256_abs_epi16(v_coeff);
  const __m256i v_zbin_mask = _mm256_cmpgt_epi16(v_abs_coeff, *v_zbin);

  if (_mm256_movemask_epi8(v_zbin_mask) == 0) {
    _mm256_store_si256((__m256i *)qcoeff_ptr, _mm256_setzero_si256());
    _mm256_store_si256((__m256i *)dqcoeff_ptr, _mm256_setzero_si256());
#if CONFIG_VP9_HIGHBITDEPTH
    _mm256_store_si256((__m256i *)(qcoeff_ptr + 8), _mm256_setzero_si256());
    _mm256_store_si256((__m256i *)(dqcoeff_ptr + 8), _mm256_setzero_si256());
#endif
    return *v_eobmax;
  }
  {
    // tmp = v_zbin_mask ? (int64_t)abs_coeff + round : 0
    const __m256i v_tmp_rnd =
        _mm256_and_si256(_mm256_adds_epi16(v_abs_coeff, *v_round), v_zbin_mask);
    //  tmp32 = (int)(((((tmp * quant_ptr[rc != 0]) >> 16) + tmp) *
    //                 quant_shift_ptr[rc != 0]) >> 15);
    const __m256i v_tmp32_a = _mm256_mulhi_epi16(v_tmp_rnd, *v_quant);
    const __m256i v_tmp32_b = _mm256_add_epi16(v_tmp32_a, v_tmp_rnd);
    const __m256i v_tmp32_hi =
        _mm256_slli_epi16(_mm256_mulhi_epi16(v_tmp32_b, *v_quant_shift), 1);
    const __m256i v_tmp32_lo =
        _mm256_srli_epi16(_mm256_mullo_epi16(v_tmp32_b, *v_quant_shift), 15);
    const __m256i v_tmp32 = _mm256_or_si256(v_tmp32_hi, v_tmp32_lo);
    const __m256i v_qcoeff = _mm256_sign_epi16(v_tmp32, v_coeff);
    const __m256i v_sign_lo =
        _mm256_unpacklo_epi16(_mm256_setzero_si256(), v_coeff);
    const __m256i v_sign_hi =
        _mm256_unpackhi_epi16(_mm256_setzero_si256(), v_coeff);
    const __m256i low = _mm256_mullo_epi16(v_tmp32, *v_dequant);
    const __m256i high = _mm256_mulhi_epi16(v_tmp32, *v_dequant);
    const __m256i v_dqcoeff_lo = _mm256_sign_epi32(
        _mm256_srli_epi32(_mm256_unpacklo_epi16(low, high), 1), v_sign_lo);
    const __m256i v_dqcoeff_hi = _mm256_sign_epi32(
        _mm256_srli_epi32(_mm256_unpackhi_epi16(low, high), 1), v_sign_hi);
    const __m256i v_nz_mask =
        _mm256_cmpgt_epi16(v_tmp32, _mm256_setzero_si256());

    store_coefficients_avx2(v_qcoeff, qcoeff_ptr);

#if CONFIG_VP9_HIGHBITDEPTH
    _mm256_storeu_si256((__m256i *)(dqcoeff_ptr), v_dqcoeff_lo);
    _mm256_storeu_si256((__m256i *)(dqcoeff_ptr + 8), v_dqcoeff_hi);
#else
    store_coefficients_avx2(_mm256_packs_epi32(v_dqcoeff_lo, v_dqcoeff_hi),
                            dqcoeff_ptr);
#endif

    return get_max_lane_eob(iscan, *v_eobmax, v_nz_mask);
  }
}

void vpx_quantize_b_32x32_avx2(const tran_low_t *coeff_ptr,
                               const struct macroblock_plane *const mb_plane,
                               tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                               const int16_t *dequant_ptr, uint16_t *eob_ptr,
                               const struct ScanOrder *const scan_order) {
  __m256i v_zbin, v_round, v_quant, v_dequant, v_quant_shift;
  __m256i v_eobmax = _mm256_setzero_si256();
  intptr_t count;
  const int16_t *iscan = scan_order->iscan;

  load_b_values_avx2(mb_plane, &v_zbin, &v_round, &v_quant, dequant_ptr,
                     &v_dequant, &v_quant_shift, 1);

  // Do DC and first 15 AC.
  v_eobmax = quantize_b_32x32_16(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, iscan,
                                 &v_quant, &v_dequant, &v_round, &v_zbin,
                                 &v_quant_shift, &v_eobmax);

  v_round = _mm256_unpackhi_epi64(v_round, v_round);
  v_quant = _mm256_unpackhi_epi64(v_quant, v_quant);
  v_dequant = _mm256_unpackhi_epi64(v_dequant, v_dequant);
  v_quant_shift = _mm256_unpackhi_epi64(v_quant_shift, v_quant_shift);
  v_zbin = _mm256_unpackhi_epi64(v_zbin, v_zbin);

  for (count = (32 * 32) - 16; count > 0; count -= 16) {
    coeff_ptr += 16;
    qcoeff_ptr += 16;
    dqcoeff_ptr += 16;
    iscan += 16;
    v_eobmax = quantize_b_32x32_16(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, iscan,
                                   &v_quant, &v_dequant, &v_round, &v_zbin,
                                   &v_quant_shift, &v_eobmax);
  }

  *eob_ptr = accumulate_eob256(v_eobmax);
}
