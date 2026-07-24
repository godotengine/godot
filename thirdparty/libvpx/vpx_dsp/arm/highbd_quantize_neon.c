/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"

static VPX_FORCE_INLINE void highbd_calculate_dqcoeff_and_store(
    const int32x4_t dqcoeff_0, const int32x4_t dqcoeff_1,
    tran_low_t *dqcoeff_ptr) {
  vst1q_s32(dqcoeff_ptr, dqcoeff_0);
  vst1q_s32(dqcoeff_ptr + 4, dqcoeff_1);
}

static VPX_FORCE_INLINE void highbd_quantize_8_neon(
    const int32x4_t coeff_0, const int32x4_t coeff_1, const int32x4_t zbin,
    const int32x4_t round, const int32x4_t quant, const int32x4_t quant_shift,
    int32x4_t *qcoeff_0, int32x4_t *qcoeff_1) {
  // Load coeffs as 2 vectors of 4 x 32-bit ints each, take sign and abs values
  const int32x4_t coeff_0_sign = vshrq_n_s32(coeff_0, 31);
  const int32x4_t coeff_1_sign = vshrq_n_s32(coeff_1, 31);
  const int32x4_t coeff_0_abs = vabsq_s32(coeff_0);
  const int32x4_t coeff_1_abs = vabsq_s32(coeff_1);

  // Calculate 2 masks of elements outside the bin
  const int32x4_t zbin_mask_0 =
      vreinterpretq_s32_u32(vcgeq_s32(coeff_0_abs, zbin));
  const int32x4_t zbin_mask_1 = vreinterpretq_s32_u32(
      vcgeq_s32(coeff_1_abs, vdupq_lane_s32(vget_low_s32(zbin), 1)));

  // Get the rounded values
  const int32x4_t rounded_0 = vaddq_s32(coeff_0_abs, round);
  const int32x4_t rounded_1 =
      vaddq_s32(coeff_1_abs, vdupq_lane_s32(vget_low_s32(round), 1));

  // (round * (quant << 15) * 2) >> 16 == (round * quant)
  int32x4_t qcoeff_tmp_0 = vqdmulhq_s32(rounded_0, quant);
  int32x4_t qcoeff_tmp_1 =
      vqdmulhq_s32(rounded_1, vdupq_lane_s32(vget_low_s32(quant), 1));

  // Add rounded values
  qcoeff_tmp_0 = vaddq_s32(qcoeff_tmp_0, rounded_0);
  qcoeff_tmp_1 = vaddq_s32(qcoeff_tmp_1, rounded_1);

  // (round * (quant_shift << 15) * 2) >> 16 == (round * quant_shift)
  qcoeff_tmp_0 = vqdmulhq_s32(qcoeff_tmp_0, quant_shift);
  qcoeff_tmp_1 =
      vqdmulhq_s32(qcoeff_tmp_1, vdupq_lane_s32(vget_low_s32(quant_shift), 1));

  // Restore the sign bit.
  qcoeff_tmp_0 = veorq_s32(qcoeff_tmp_0, coeff_0_sign);
  qcoeff_tmp_1 = veorq_s32(qcoeff_tmp_1, coeff_1_sign);
  qcoeff_tmp_0 = vsubq_s32(qcoeff_tmp_0, coeff_0_sign);
  qcoeff_tmp_1 = vsubq_s32(qcoeff_tmp_1, coeff_1_sign);

  // Only keep the relevant coeffs
  *qcoeff_0 = vandq_s32(qcoeff_tmp_0, zbin_mask_0);
  *qcoeff_1 = vandq_s32(qcoeff_tmp_1, zbin_mask_1);
}

static VPX_FORCE_INLINE int16x8_t
highbd_quantize_b_neon(const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
                       tran_low_t *dqcoeff_ptr, const int32x4_t zbin,
                       const int32x4_t round, const int32x4_t quant,
                       const int32x4_t quant_shift, const int32x4_t dequant) {
  int32x4_t qcoeff_0, qcoeff_1, dqcoeff_0, dqcoeff_1;

  // Load coeffs as 2 vectors of 4 x 32-bit ints each, take sign and abs values
  const int32x4_t coeff_0 = vld1q_s32(coeff_ptr);
  const int32x4_t coeff_1 = vld1q_s32(coeff_ptr + 4);
  highbd_quantize_8_neon(coeff_0, coeff_1, zbin, round, quant, quant_shift,
                         &qcoeff_0, &qcoeff_1);

  // Store the 32-bit qcoeffs
  vst1q_s32(qcoeff_ptr, qcoeff_0);
  vst1q_s32(qcoeff_ptr + 4, qcoeff_1);

  // Calculate and store the dqcoeffs
  dqcoeff_0 = vmulq_s32(qcoeff_0, dequant);
  dqcoeff_1 = vmulq_s32(qcoeff_1, vdupq_lane_s32(vget_low_s32(dequant), 1));

  highbd_calculate_dqcoeff_and_store(dqcoeff_0, dqcoeff_1, dqcoeff_ptr);

  return vcombine_s16(vmovn_s32(qcoeff_0), vmovn_s32(qcoeff_1));
}

void vpx_highbd_quantize_b_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                                const struct macroblock_plane *const mb_plane,
                                tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                                const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                const struct ScanOrder *const scan_order) {
  const int16x8_t neg_one = vdupq_n_s16(-1);
  uint16x8_t eob_max;
  const int16_t *iscan = scan_order->iscan;

  // Only the first element of each vector is DC.
  // High half has identical elements, but we can reconstruct it from the low
  // half by duplicating the 2nd element. So we only need to pass a 4x32-bit
  // vector
  int32x4_t zbin = vmovl_s16(vld1_s16(mb_plane->zbin));
  int32x4_t round = vmovl_s16(vld1_s16(mb_plane->round));
  // Extend the quant, quant_shift vectors to ones of 32-bit elements
  // scale to high-half, so we can use vqdmulhq_s32
  int32x4_t quant = vshlq_n_s32(vmovl_s16(vld1_s16(mb_plane->quant)), 15);
  int32x4_t quant_shift =
      vshlq_n_s32(vmovl_s16(vld1_s16(mb_plane->quant_shift)), 15);
  int32x4_t dequant = vmovl_s16(vld1_s16(dequant_ptr));

  // Process first 8 values which include a dc component.
  {
    const uint16x8_t v_iscan = vreinterpretq_u16_s16(vld1q_s16(iscan));

    const int16x8_t qcoeff =
        highbd_quantize_b_neon(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, zbin, round,
                               quant, quant_shift, dequant);

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
    zbin = vdupq_lane_s32(vget_low_s32(zbin), 1);
    round = vdupq_lane_s32(vget_low_s32(round), 1);
    quant = vdupq_lane_s32(vget_low_s32(quant), 1);
    quant_shift = vdupq_lane_s32(vget_low_s32(quant_shift), 1);
    dequant = vdupq_lane_s32(vget_low_s32(dequant), 1);

    do {
      const uint16x8_t v_iscan = vreinterpretq_u16_s16(vld1q_s16(iscan));

      const int16x8_t qcoeff =
          highbd_quantize_b_neon(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, zbin,
                                 round, quant, quant_shift, dequant);

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

static VPX_FORCE_INLINE int32x4_t extract_sign_bit(int32x4_t a) {
  return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 31));
}

static VPX_FORCE_INLINE void highbd_calculate_dqcoeff_and_store_32x32(
    int32x4_t dqcoeff_0, int32x4_t dqcoeff_1, tran_low_t *dqcoeff_ptr) {
  // Add 1 if negative to round towards zero because the C uses division.
  dqcoeff_0 = vaddq_s32(dqcoeff_0, extract_sign_bit(dqcoeff_0));
  dqcoeff_1 = vaddq_s32(dqcoeff_1, extract_sign_bit(dqcoeff_1));

  dqcoeff_0 = vshrq_n_s32(dqcoeff_0, 1);
  dqcoeff_1 = vshrq_n_s32(dqcoeff_1, 1);
  vst1q_s32(dqcoeff_ptr, dqcoeff_0);
  vst1q_s32(dqcoeff_ptr + 4, dqcoeff_1);
}

static VPX_FORCE_INLINE int16x8_t highbd_quantize_b_32x32_neon(
    const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int32x4_t zbin, const int32x4_t round,
    const int32x4_t quant, const int32x4_t quant_shift,
    const int32x4_t dequant) {
  int32x4_t qcoeff_0, qcoeff_1, dqcoeff_0, dqcoeff_1;

  // Load coeffs as 2 vectors of 4 x 32-bit ints each, take sign and abs values
  const int32x4_t coeff_0 = vld1q_s32(coeff_ptr);
  const int32x4_t coeff_1 = vld1q_s32(coeff_ptr + 4);
  highbd_quantize_8_neon(coeff_0, coeff_1, zbin, round, quant, quant_shift,
                         &qcoeff_0, &qcoeff_1);

  // Store the 32-bit qcoeffs
  vst1q_s32(qcoeff_ptr, qcoeff_0);
  vst1q_s32(qcoeff_ptr + 4, qcoeff_1);

  // Calculate and store the dqcoeffs
  dqcoeff_0 = vmulq_s32(qcoeff_0, dequant);
  dqcoeff_1 = vmulq_s32(qcoeff_1, vdupq_lane_s32(vget_low_s32(dequant), 1));

  highbd_calculate_dqcoeff_and_store_32x32(dqcoeff_0, dqcoeff_1, dqcoeff_ptr);

  return vcombine_s16(vmovn_s32(qcoeff_0), vmovn_s32(qcoeff_1));
}

void vpx_highbd_quantize_b_32x32_neon(
    const tran_low_t *coeff_ptr, const struct macroblock_plane *const mb_plane,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,
    uint16_t *eob_ptr, const struct ScanOrder *const scan_order) {
  const int16x8_t neg_one = vdupq_n_s16(-1);
  uint16x8_t eob_max;
  int i;
  const int16_t *iscan = scan_order->iscan;

  // Only the first element of each vector is DC.
  // High half has identical elements, but we can reconstruct it from the low
  // half by duplicating the 2nd element. So we only need to pass a 4x32-bit
  // vector
  int32x4_t zbin = vrshrq_n_s32(vmovl_s16(vld1_s16(mb_plane->zbin)), 1);
  int32x4_t round = vrshrq_n_s32(vmovl_s16(vld1_s16(mb_plane->round)), 1);
  // Extend the quant, quant_shift vectors to ones of 32-bit elements
  // scale to high-half, so we can use vqdmulhq_s32
  int32x4_t quant = vshlq_n_s32(vmovl_s16(vld1_s16(mb_plane->quant)), 15);
  int32x4_t quant_shift =
      vshlq_n_s32(vmovl_s16(vld1_s16(mb_plane->quant_shift)), 16);
  int32x4_t dequant = vmovl_s16(vld1_s16(dequant_ptr));

  // Process first 8 values which include a dc component.
  {
    const uint16x8_t v_iscan = vreinterpretq_u16_s16(vld1q_s16(iscan));

    const int16x8_t qcoeff =
        highbd_quantize_b_32x32_neon(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, zbin,
                                     round, quant, quant_shift, dequant);

    // Set non-zero elements to -1 and use that to extract values for eob.
    eob_max = vandq_u16(vtstq_s16(qcoeff, neg_one), v_iscan);

    __builtin_prefetch(coeff_ptr + 64);
    coeff_ptr += 8;
    iscan += 8;
    qcoeff_ptr += 8;
    dqcoeff_ptr += 8;
  }

  {
    zbin = vdupq_lane_s32(vget_low_s32(zbin), 1);
    round = vdupq_lane_s32(vget_low_s32(round), 1);
    quant = vdupq_lane_s32(vget_low_s32(quant), 1);
    quant_shift = vdupq_lane_s32(vget_low_s32(quant_shift), 1);
    dequant = vdupq_lane_s32(vget_low_s32(dequant), 1);

    for (i = 1; i < 32 * 32 / 8; ++i) {
      const uint16x8_t v_iscan = vreinterpretq_u16_s16(vld1q_s16(iscan));

      const int16x8_t qcoeff =
          highbd_quantize_b_32x32_neon(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, zbin,
                                       round, quant, quant_shift, dequant);

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
