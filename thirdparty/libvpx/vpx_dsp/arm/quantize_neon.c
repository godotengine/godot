/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"

static INLINE void calculate_dqcoeff_and_store(const int16x8_t qcoeff,
                                               const int16x8_t dequant,
                                               tran_low_t *dqcoeff_ptr) {
#if CONFIG_VP9_HIGHBITDEPTH
  const int32x4_t dqcoeff_0 =
      vmull_s16(vget_low_s16(qcoeff), vget_low_s16(dequant));
  const int32x4_t dqcoeff_1 =
      vmull_s16(vget_high_s16(qcoeff), vget_high_s16(dequant));

  vst1q_s32(dqcoeff_ptr, dqcoeff_0);
  vst1q_s32(dqcoeff_ptr + 4, dqcoeff_1);
#else
  vst1q_s16(dqcoeff_ptr, vmulq_s16(qcoeff, dequant));
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

static INLINE int16x8_t
quantize_b_neon(const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
                tran_low_t *dqcoeff_ptr, const int16x8_t zbin,
                const int16x8_t round, const int16x8_t quant,
                const int16x8_t quant_shift, const int16x8_t dequant) {
  // Load coeffs as 8 x 16-bit ints, take sign and abs values
  const int16x8_t coeff = load_tran_low_to_s16q(coeff_ptr);
  const int16x8_t coeff_sign = vshrq_n_s16(coeff, 15);
  const int16x8_t coeff_abs = vabsq_s16(coeff);

  // Calculate mask of elements outside the bin
  const int16x8_t zbin_mask = vreinterpretq_s16_u16(vcgeq_s16(coeff_abs, zbin));

  // Get the rounded values
  const int16x8_t rounded = vqaddq_s16(coeff_abs, round);

  // (round * quant * 2) >> 16 >> 1 == (round * quant) >> 16
  int16x8_t qcoeff = vshrq_n_s16(vqdmulhq_s16(rounded, quant), 1);

  qcoeff = vaddq_s16(qcoeff, rounded);

  // (qcoeff * quant_shift * 2) >> 16 >> 1 == (qcoeff * quant_shift) >> 16
  qcoeff = vshrq_n_s16(vqdmulhq_s16(qcoeff, quant_shift), 1);

  // Restore the sign bit.
  qcoeff = veorq_s16(qcoeff, coeff_sign);
  qcoeff = vsubq_s16(qcoeff, coeff_sign);

  // Only keep the relevant coeffs
  qcoeff = vandq_s16(qcoeff, zbin_mask);
  store_s16q_to_tran_low(qcoeff_ptr, qcoeff);

  calculate_dqcoeff_and_store(qcoeff, dequant, dqcoeff_ptr);

  return qcoeff;
}

void vpx_quantize_b_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                         const struct macroblock_plane *const mb_plane,
                         tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                         const int16_t *dequant_ptr, uint16_t *eob_ptr,
                         const struct ScanOrder *const scan_order) {
  const int16x8_t neg_one = vdupq_n_s16(-1);
  uint16x8_t eob_max;
  int16_t const *iscan = scan_order->iscan;

  // Only the first element of each vector is DC.
  int16x8_t zbin = vld1q_s16(mb_plane->zbin);
  int16x8_t round = vld1q_s16(mb_plane->round);
  int16x8_t quant = vld1q_s16(mb_plane->quant);
  int16x8_t quant_shift = vld1q_s16(mb_plane->quant_shift);
  int16x8_t dequant = vld1q_s16(dequant_ptr);

  // Process first 8 values which include a dc component.
  {
    const uint16x8_t v_iscan = vreinterpretq_u16_s16(vld1q_s16(iscan));

    const int16x8_t qcoeff =
        quantize_b_neon(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, zbin, round, quant,
                        quant_shift, dequant);

    // Set non-zero elements to -1 and use that to extract values for eob.
    eob_max = vandq_u16(vtstq_s16(qcoeff, neg_one), v_iscan);

    __builtin_prefetch(coeff_ptr + 64);
    coeff_ptr += 8;
    iscan += 8;
    qcoeff_ptr += 8;
    dqcoeff_ptr += 8;
  }

  n_coeffs -= 8;

  {
    zbin = vdupq_lane_s16(vget_low_s16(zbin), 1);
    round = vdupq_lane_s16(vget_low_s16(round), 1);
    quant = vdupq_lane_s16(vget_low_s16(quant), 1);
    quant_shift = vdupq_lane_s16(vget_low_s16(quant_shift), 1);
    dequant = vdupq_lane_s16(vget_low_s16(dequant), 1);

    do {
      const uint16x8_t v_iscan = vreinterpretq_u16_s16(vld1q_s16(iscan));

      const int16x8_t qcoeff =
          quantize_b_neon(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, zbin, round,
                          quant, quant_shift, dequant);

      // Set non-zero elements to -1 and use that to extract values for eob.
      eob_max =
          vmaxq_u16(eob_max, vandq_u16(vtstq_s16(qcoeff, neg_one), v_iscan));

      __builtin_prefetch(coeff_ptr + 64);
      coeff_ptr += 8;
      iscan += 8;
      qcoeff_ptr += 8;
      dqcoeff_ptr += 8;
      n_coeffs -= 8;
    } while (n_coeffs > 0);
  }

#if VPX_ARCH_AARCH64
  *eob_ptr = vmaxvq_u16(eob_max);
#else
  {
    const uint16x4_t eob_max_0 =
        vmax_u16(vget_low_u16(eob_max), vget_high_u16(eob_max));
    const uint16x4_t eob_max_1 = vpmax_u16(eob_max_0, eob_max_0);
    const uint16x4_t eob_max_2 = vpmax_u16(eob_max_1, eob_max_1);
    vst1_lane_u16(eob_ptr, eob_max_2, 0);
  }
#endif  // VPX_ARCH_AARCH64
}

static INLINE int32x4_t extract_sign_bit(int32x4_t a) {
  return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 31));
}

static INLINE void calculate_dqcoeff_and_store_32x32(const int16x8_t qcoeff,
                                                     const int16x8_t dequant,
                                                     tran_low_t *dqcoeff_ptr) {
  int32x4_t dqcoeff_0 = vmull_s16(vget_low_s16(qcoeff), vget_low_s16(dequant));
  int32x4_t dqcoeff_1 =
      vmull_s16(vget_high_s16(qcoeff), vget_high_s16(dequant));

  // Add 1 if negative to round towards zero because the C uses division.
  dqcoeff_0 = vaddq_s32(dqcoeff_0, extract_sign_bit(dqcoeff_0));
  dqcoeff_1 = vaddq_s32(dqcoeff_1, extract_sign_bit(dqcoeff_1));

#if CONFIG_VP9_HIGHBITDEPTH
  dqcoeff_0 = vshrq_n_s32(dqcoeff_0, 1);
  dqcoeff_1 = vshrq_n_s32(dqcoeff_1, 1);
  vst1q_s32(dqcoeff_ptr, dqcoeff_0);
  vst1q_s32(dqcoeff_ptr + 4, dqcoeff_1);
#else
  vst1q_s16(dqcoeff_ptr,
            vcombine_s16(vshrn_n_s32(dqcoeff_0, 1), vshrn_n_s32(dqcoeff_1, 1)));
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

static INLINE int16x8_t
quantize_b_32x32_neon(const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
                      tran_low_t *dqcoeff_ptr, const int16x8_t zbin,
                      const int16x8_t round, const int16x8_t quant,
                      const int16x8_t quant_shift, const int16x8_t dequant) {
  // Load coeffs as 8 x 16-bit ints, take sign and abs values
  const int16x8_t coeff = load_tran_low_to_s16q(coeff_ptr);
  const int16x8_t coeff_sign = vshrq_n_s16(coeff, 15);
  const int16x8_t coeff_abs = vabsq_s16(coeff);

  // Calculate mask of elements outside the bin
  const int16x8_t zbin_mask = vreinterpretq_s16_u16(vcgeq_s16(coeff_abs, zbin));

  // Get the rounded values
  const int16x8_t rounded = vqaddq_s16(coeff_abs, round);

  // (round * quant * 2) >> 16 >> 1 == (round * quant) >> 16
  int16x8_t qcoeff = vshrq_n_s16(vqdmulhq_s16(rounded, quant), 1);

  qcoeff = vaddq_s16(qcoeff, rounded);

  // (qcoeff * quant_shift * 2) >> 16 == (qcoeff * quant_shift) >> 15
  qcoeff = vqdmulhq_s16(qcoeff, quant_shift);

  // Restore the sign bit.
  qcoeff = veorq_s16(qcoeff, coeff_sign);
  qcoeff = vsubq_s16(qcoeff, coeff_sign);

  // Only keep the relevant coeffs
  qcoeff = vandq_s16(qcoeff, zbin_mask);
  store_s16q_to_tran_low(qcoeff_ptr, qcoeff);

  calculate_dqcoeff_and_store_32x32(qcoeff, dequant, dqcoeff_ptr);

  return qcoeff;
}

// Main difference is that zbin values are halved before comparison and dqcoeff
// values are divided by 2. zbin is rounded but dqcoeff is not.
void vpx_quantize_b_32x32_neon(const tran_low_t *coeff_ptr,
                               const struct macroblock_plane *mb_plane,
                               tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                               const int16_t *dequant_ptr, uint16_t *eob_ptr,
                               const struct ScanOrder *const scan_order) {
  const int16x8_t neg_one = vdupq_n_s16(-1);
  uint16x8_t eob_max;
  int i;
  const int16_t *iscan = scan_order->iscan;

  // Only the first element of each vector is DC.
  int16x8_t zbin = vrshrq_n_s16(vld1q_s16(mb_plane->zbin), 1);
  int16x8_t round = vrshrq_n_s16(vld1q_s16(mb_plane->round), 1);
  int16x8_t quant = vld1q_s16(mb_plane->quant);
  int16x8_t quant_shift = vld1q_s16(mb_plane->quant_shift);
  int16x8_t dequant = vld1q_s16(dequant_ptr);

  // Process first 8 values which include a dc component.
  {
    const uint16x8_t v_iscan = vreinterpretq_u16_s16(vld1q_s16(iscan));

    const int16x8_t qcoeff =
        quantize_b_32x32_neon(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, zbin, round,
                              quant, quant_shift, dequant);

    // Set non-zero elements to -1 and use that to extract values for eob.
    eob_max = vandq_u16(vtstq_s16(qcoeff, neg_one), v_iscan);

    __builtin_prefetch(coeff_ptr + 64);
    coeff_ptr += 8;
    iscan += 8;
    qcoeff_ptr += 8;
    dqcoeff_ptr += 8;
  }

  {
    zbin = vdupq_lane_s16(vget_low_s16(zbin), 1);
    round = vdupq_lane_s16(vget_low_s16(round), 1);
    quant = vdupq_lane_s16(vget_low_s16(quant), 1);
    quant_shift = vdupq_lane_s16(vget_low_s16(quant_shift), 1);
    dequant = vdupq_lane_s16(vget_low_s16(dequant), 1);

    for (i = 1; i < 32 * 32 / 8; ++i) {
      const uint16x8_t v_iscan = vreinterpretq_u16_s16(vld1q_s16(iscan));

      const int16x8_t qcoeff =
          quantize_b_32x32_neon(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, zbin, round,
                                quant, quant_shift, dequant);

      // Set non-zero elements to -1 and use that to extract values for eob.
      eob_max =
          vmaxq_u16(eob_max, vandq_u16(vtstq_s16(qcoeff, neg_one), v_iscan));

      __builtin_prefetch(coeff_ptr + 64);
      coeff_ptr += 8;
      iscan += 8;
      qcoeff_ptr += 8;
      dqcoeff_ptr += 8;
    }
  }

#if VPX_ARCH_AARCH64
  *eob_ptr = vmaxvq_u16(eob_max);
#else
  {
    const uint16x4_t eob_max_0 =
        vmax_u16(vget_low_u16(eob_max), vget_high_u16(eob_max));
    const uint16x4_t eob_max_1 = vpmax_u16(eob_max_0, eob_max_0);
    const uint16x4_t eob_max_2 = vpmax_u16(eob_max_1, eob_max_1);
    vst1_lane_u16(eob_ptr, eob_max_2, 0);
  }
#endif  // VPX_ARCH_AARCH64
}
