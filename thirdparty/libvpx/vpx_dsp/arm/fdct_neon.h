/*
 *  Copyright (c) 2022 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_FDCT_NEON_H_
#define VPX_VPX_DSP_ARM_FDCT_NEON_H_

#include <arm_neon.h>

// fdct_round_shift((a +/- b) * c)
// Variant that performs fast vqrdmulh_s16 operation on half vector
// can be slightly less accurate, adequate for pass1
static INLINE void butterfly_one_coeff_s16_fast_half(const int16x4_t a,
                                                     const int16x4_t b,
                                                     const tran_coef_t constant,
                                                     int16x4_t *add,
                                                     int16x4_t *sub) {
  int16x4_t c = vdup_n_s16(2 * constant);
  *add = vqrdmulh_s16(vadd_s16(a, b), c);
  *sub = vqrdmulh_s16(vsub_s16(a, b), c);
}

// fdct_round_shift((a +/- b) * c)
// Variant that performs fast vqrdmulh_s16 operation on full vector
// can be slightly less accurate, adequate for pass1
static INLINE void butterfly_one_coeff_s16_fast(const int16x8_t a,
                                                const int16x8_t b,
                                                const tran_coef_t constant,
                                                int16x8_t *add,
                                                int16x8_t *sub) {
  int16x8_t c = vdupq_n_s16(2 * constant);
  *add = vqrdmulhq_s16(vaddq_s16(a, b), c);
  *sub = vqrdmulhq_s16(vsubq_s16(a, b), c);
}

// fdct_round_shift((a +/- b) * c)
// Variant that performs fast vqrdmulhq_s32 operation on full vector
// more accurate does 32-bit processing, takes 16-bit input values,
// returns full 32-bit values, high/low
static INLINE void butterfly_one_coeff_s16_s32_fast(
    const int16x8_t a, const int16x8_t b, const tran_coef_t constant,
    int32x4_t *add_lo, int32x4_t *add_hi, int32x4_t *sub_lo,
    int32x4_t *sub_hi) {
  int32x4_t c = vdupq_n_s32(constant << 17);
  const int16x4_t a_lo = vget_low_s16(a);
  const int16x4_t a_hi = vget_high_s16(a);
  const int16x4_t b_lo = vget_low_s16(b);
  const int16x4_t b_hi = vget_high_s16(b);
  *add_lo = vqrdmulhq_s32(vaddl_s16(a_lo, b_lo), c);
  *add_hi = vqrdmulhq_s32(vaddl_s16(a_hi, b_hi), c);
  *sub_lo = vqrdmulhq_s32(vsubl_s16(a_lo, b_lo), c);
  *sub_hi = vqrdmulhq_s32(vsubl_s16(a_hi, b_hi), c);
}

// fdct_round_shift((a +/- b) * c)
// Variant that performs fast vqrdmulhq_s32 operation on full vector
// more accurate does 32-bit processing, takes 16-bit input values,
// returns full 32-bit values, high/low
static INLINE void butterfly_one_coeff_s16_s32_fast_narrow(
    const int16x8_t a, const int16x8_t b, const tran_coef_t constant,
    int16x8_t *add, int16x8_t *sub) {
  int32x4_t add_lo, add_hi, sub_lo, sub_hi;
  butterfly_one_coeff_s16_s32_fast(a, b, constant, &add_lo, &add_hi, &sub_lo,
                                   &sub_hi);
  *add = vcombine_s16(vmovn_s32(add_lo), vmovn_s32(add_hi));
  *sub = vcombine_s16(vmovn_s32(sub_lo), vmovn_s32(sub_hi));
}

// fdct_round_shift((a +/- b) * c)
// Variant that performs fast vqrdmulhq_s32 operation on full vector
// more accurate does 32-bit processing, takes 16-bit input values,
// returns full 32-bit values, high/low
static INLINE void butterfly_one_coeff_s16_s32_fast_half(
    const int16x4_t a, const int16x4_t b, const tran_coef_t constant,
    int32x4_t *add, int32x4_t *sub) {
  int32x4_t c = vdupq_n_s32(constant << 17);
  *add = vqrdmulhq_s32(vaddl_s16(a, b), c);
  *sub = vqrdmulhq_s32(vsubl_s16(a, b), c);
}

// fdct_round_shift((a +/- b) * c)
// Variant that performs fast vqrdmulhq_s32 operation on half vector
// more accurate does 32-bit processing, takes 16-bit input values,
// returns narrowed down 16-bit values
static INLINE void butterfly_one_coeff_s16_s32_fast_narrow_half(
    const int16x4_t a, const int16x4_t b, const tran_coef_t constant,
    int16x4_t *add, int16x4_t *sub) {
  int32x4_t add32, sub32;
  butterfly_one_coeff_s16_s32_fast_half(a, b, constant, &add32, &sub32);
  *add = vmovn_s32(add32);
  *sub = vmovn_s32(sub32);
}

// fdct_round_shift((a +/- b) * c)
// Original Variant that performs normal implementation on full vector
// fully accurate does 32-bit processing, takes 16-bit values
static INLINE void butterfly_one_coeff_s16_s32(
    const int16x8_t a, const int16x8_t b, const tran_coef_t constant,
    int32x4_t *add_lo, int32x4_t *add_hi, int32x4_t *sub_lo,
    int32x4_t *sub_hi) {
  const int32x4_t a0 = vmull_n_s16(vget_low_s16(a), constant);
  const int32x4_t a1 = vmull_n_s16(vget_high_s16(a), constant);
  const int32x4_t sum0 = vmlal_n_s16(a0, vget_low_s16(b), constant);
  const int32x4_t sum1 = vmlal_n_s16(a1, vget_high_s16(b), constant);
  const int32x4_t diff0 = vmlsl_n_s16(a0, vget_low_s16(b), constant);
  const int32x4_t diff1 = vmlsl_n_s16(a1, vget_high_s16(b), constant);
  *add_lo = vrshrq_n_s32(sum0, DCT_CONST_BITS);
  *add_hi = vrshrq_n_s32(sum1, DCT_CONST_BITS);
  *sub_lo = vrshrq_n_s32(diff0, DCT_CONST_BITS);
  *sub_hi = vrshrq_n_s32(diff1, DCT_CONST_BITS);
}

// fdct_round_shift((a +/- b) * c)
// Original Variant that performs normal implementation on full vector
// fully accurate does 32-bit processing, takes 16-bit values
// returns narrowed down 16-bit values
static INLINE void butterfly_one_coeff_s16_s32_narrow(
    const int16x8_t a, const int16x8_t b, const tran_coef_t constant,
    int16x8_t *add, int16x8_t *sub) {
  int32x4_t add32_lo, add32_hi, sub32_lo, sub32_hi;
  butterfly_one_coeff_s16_s32(a, b, constant, &add32_lo, &add32_hi, &sub32_lo,
                              &sub32_hi);
  *add = vcombine_s16(vmovn_s32(add32_lo), vmovn_s32(add32_hi));
  *sub = vcombine_s16(vmovn_s32(sub32_lo), vmovn_s32(sub32_hi));
}

// fdct_round_shift((a +/- b) * c)
// Variant that performs fast vqrdmulhq_s32 operation on full vector
// more accurate does 32-bit processing, takes and returns 32-bit values,
// high/low
static INLINE void butterfly_one_coeff_s32_noround(
    const int32x4_t a_lo, const int32x4_t a_hi, const int32x4_t b_lo,
    const int32x4_t b_hi, const tran_coef_t constant, int32x4_t *add_lo,
    int32x4_t *add_hi, int32x4_t *sub_lo, int32x4_t *sub_hi) {
  const int32x4_t a1 = vmulq_n_s32(a_lo, constant);
  const int32x4_t a2 = vmulq_n_s32(a_hi, constant);
  const int32x4_t a3 = vmulq_n_s32(a_lo, constant);
  const int32x4_t a4 = vmulq_n_s32(a_hi, constant);
  *add_lo = vmlaq_n_s32(a1, b_lo, constant);
  *add_hi = vmlaq_n_s32(a2, b_hi, constant);
  *sub_lo = vmlsq_n_s32(a3, b_lo, constant);
  *sub_hi = vmlsq_n_s32(a4, b_hi, constant);
}

// fdct_round_shift((a +/- b) * c)
// Variant that performs fast vqrdmulhq_s32 operation on full vector
// more accurate does 32-bit processing, takes and returns 32-bit values,
// high/low
static INLINE void butterfly_one_coeff_s32_fast_half(const int32x4_t a,
                                                     const int32x4_t b,
                                                     const tran_coef_t constant,
                                                     int32x4_t *add,
                                                     int32x4_t *sub) {
  const int32x4_t c = vdupq_n_s32(constant << 17);
  *add = vqrdmulhq_s32(vaddq_s32(a, b), c);
  *sub = vqrdmulhq_s32(vsubq_s32(a, b), c);
}

// fdct_round_shift((a +/- b) * c)
// Variant that performs fast vqrdmulhq_s32 operation on full vector
// more accurate does 32-bit processing, takes and returns 32-bit values,
// high/low
static INLINE void butterfly_one_coeff_s32_fast(
    const int32x4_t a_lo, const int32x4_t a_hi, const int32x4_t b_lo,
    const int32x4_t b_hi, const tran_coef_t constant, int32x4_t *add_lo,
    int32x4_t *add_hi, int32x4_t *sub_lo, int32x4_t *sub_hi) {
  const int32x4_t c = vdupq_n_s32(constant << 17);
  *add_lo = vqrdmulhq_s32(vaddq_s32(a_lo, b_lo), c);
  *add_hi = vqrdmulhq_s32(vaddq_s32(a_hi, b_hi), c);
  *sub_lo = vqrdmulhq_s32(vsubq_s32(a_lo, b_lo), c);
  *sub_hi = vqrdmulhq_s32(vsubq_s32(a_hi, b_hi), c);
}

// fdct_round_shift((a +/- b) * c)
// Variant that performs normal implementation on full vector
// more accurate does 64-bit processing, takes and returns 32-bit values
// returns narrowed results
static INLINE void butterfly_one_coeff_s32_s64_narrow(
    const int32x4_t a_lo, const int32x4_t a_hi, const int32x4_t b_lo,
    const int32x4_t b_hi, const tran_coef_t constant, int32x4_t *add_lo,
    int32x4_t *add_hi, int32x4_t *sub_lo, int32x4_t *sub_hi) {
  // ac holds the following values:
  // ac: vget_low_s32(a_lo) * c, vget_high_s32(a_lo) * c,
  //     vget_low_s32(a_hi) * c, vget_high_s32(a_hi) * c
  int64x2_t ac[4];
  int64x2_t sum[4];
  int64x2_t diff[4];

  ac[0] = vmull_n_s32(vget_low_s32(a_lo), constant);
  ac[1] = vmull_n_s32(vget_high_s32(a_lo), constant);
  ac[2] = vmull_n_s32(vget_low_s32(a_hi), constant);
  ac[3] = vmull_n_s32(vget_high_s32(a_hi), constant);

  sum[0] = vmlal_n_s32(ac[0], vget_low_s32(b_lo), constant);
  sum[1] = vmlal_n_s32(ac[1], vget_high_s32(b_lo), constant);
  sum[2] = vmlal_n_s32(ac[2], vget_low_s32(b_hi), constant);
  sum[3] = vmlal_n_s32(ac[3], vget_high_s32(b_hi), constant);
  *add_lo = vcombine_s32(vrshrn_n_s64(sum[0], DCT_CONST_BITS),
                         vrshrn_n_s64(sum[1], DCT_CONST_BITS));
  *add_hi = vcombine_s32(vrshrn_n_s64(sum[2], DCT_CONST_BITS),
                         vrshrn_n_s64(sum[3], DCT_CONST_BITS));

  diff[0] = vmlsl_n_s32(ac[0], vget_low_s32(b_lo), constant);
  diff[1] = vmlsl_n_s32(ac[1], vget_high_s32(b_lo), constant);
  diff[2] = vmlsl_n_s32(ac[2], vget_low_s32(b_hi), constant);
  diff[3] = vmlsl_n_s32(ac[3], vget_high_s32(b_hi), constant);
  *sub_lo = vcombine_s32(vrshrn_n_s64(diff[0], DCT_CONST_BITS),
                         vrshrn_n_s64(diff[1], DCT_CONST_BITS));
  *sub_hi = vcombine_s32(vrshrn_n_s64(diff[2], DCT_CONST_BITS),
                         vrshrn_n_s64(diff[3], DCT_CONST_BITS));
}

// fdct_round_shift(a * c1 +/- b * c2)
// Variant that performs normal implementation on half vector
// more accurate does 64-bit processing, takes and returns 32-bit values
// returns narrowed results
static INLINE void butterfly_two_coeff_s32_s64_narrow_half(
    const int32x4_t a, const int32x4_t b, const tran_coef_t constant1,
    const tran_coef_t constant2, int32x4_t *add, int32x4_t *sub) {
  const int32x2_t a_lo = vget_low_s32(a);
  const int32x2_t a_hi = vget_high_s32(a);
  const int32x2_t b_lo = vget_low_s32(b);
  const int32x2_t b_hi = vget_high_s32(b);

  const int64x2_t axc0_64_lo = vmull_n_s32(a_lo, constant1);
  const int64x2_t axc0_64_hi = vmull_n_s32(a_hi, constant1);
  const int64x2_t axc1_64_lo = vmull_n_s32(a_lo, constant2);
  const int64x2_t axc1_64_hi = vmull_n_s32(a_hi, constant2);

  const int64x2_t sum_lo = vmlal_n_s32(axc0_64_lo, b_lo, constant2);
  const int64x2_t sum_hi = vmlal_n_s32(axc0_64_hi, b_hi, constant2);
  const int64x2_t diff_lo = vmlsl_n_s32(axc1_64_lo, b_lo, constant1);
  const int64x2_t diff_hi = vmlsl_n_s32(axc1_64_hi, b_hi, constant1);

  *add = vcombine_s32(vrshrn_n_s64(sum_lo, DCT_CONST_BITS),
                      vrshrn_n_s64(sum_hi, DCT_CONST_BITS));
  *sub = vcombine_s32(vrshrn_n_s64(diff_lo, DCT_CONST_BITS),
                      vrshrn_n_s64(diff_hi, DCT_CONST_BITS));
}

// fdct_round_shift(a * c1 +/- b * c2)
// Variant that performs normal implementation on full vector
// more accurate does 64-bit processing, takes and returns 64-bit values
// returns results without rounding
static INLINE void butterfly_two_coeff_s32_s64_noround(
    const int32x4_t a_lo, const int32x4_t a_hi, const int32x4_t b_lo,
    const int32x4_t b_hi, const tran_coef_t constant1,
    const tran_coef_t constant2, int64x2_t *add_lo /*[2]*/,
    int64x2_t *add_hi /*[2]*/, int64x2_t *sub_lo /*[2]*/,
    int64x2_t *sub_hi /*[2]*/) {
  // ac1/ac2 hold the following values:
  // ac1: vget_low_s32(a_lo) * c1, vget_high_s32(a_lo) * c1,
  //      vget_low_s32(a_hi) * c1, vget_high_s32(a_hi) * c1
  // ac2: vget_low_s32(a_lo) * c2, vget_high_s32(a_lo) * c2,
  //      vget_low_s32(a_hi) * c2, vget_high_s32(a_hi) * c2
  int64x2_t ac1[4];
  int64x2_t ac2[4];

  ac1[0] = vmull_n_s32(vget_low_s32(a_lo), constant1);
  ac1[1] = vmull_n_s32(vget_high_s32(a_lo), constant1);
  ac1[2] = vmull_n_s32(vget_low_s32(a_hi), constant1);
  ac1[3] = vmull_n_s32(vget_high_s32(a_hi), constant1);
  ac2[0] = vmull_n_s32(vget_low_s32(a_lo), constant2);
  ac2[1] = vmull_n_s32(vget_high_s32(a_lo), constant2);
  ac2[2] = vmull_n_s32(vget_low_s32(a_hi), constant2);
  ac2[3] = vmull_n_s32(vget_high_s32(a_hi), constant2);

  add_lo[0] = vmlal_n_s32(ac1[0], vget_low_s32(b_lo), constant2);
  add_lo[1] = vmlal_n_s32(ac1[1], vget_high_s32(b_lo), constant2);
  add_hi[0] = vmlal_n_s32(ac1[2], vget_low_s32(b_hi), constant2);
  add_hi[1] = vmlal_n_s32(ac1[3], vget_high_s32(b_hi), constant2);

  sub_lo[0] = vmlsl_n_s32(ac2[0], vget_low_s32(b_lo), constant1);
  sub_lo[1] = vmlsl_n_s32(ac2[1], vget_high_s32(b_lo), constant1);
  sub_hi[0] = vmlsl_n_s32(ac2[2], vget_low_s32(b_hi), constant1);
  sub_hi[1] = vmlsl_n_s32(ac2[3], vget_high_s32(b_hi), constant1);
}

// fdct_round_shift(a * c1 +/- b * c2)
// Variant that performs normal implementation on full vector
// more accurate does 64-bit processing, takes and returns 32-bit values
// returns narrowed results
static INLINE void butterfly_two_coeff_s32_s64_narrow(
    const int32x4_t a_lo, const int32x4_t a_hi, const int32x4_t b_lo,
    const int32x4_t b_hi, const tran_coef_t constant1,
    const tran_coef_t constant2, int32x4_t *add_lo, int32x4_t *add_hi,
    int32x4_t *sub_lo, int32x4_t *sub_hi) {
  // ac1/ac2 hold the following values:
  // ac1: vget_low_s32(a_lo) * c1, vget_high_s32(a_lo) * c1,
  //      vget_low_s32(a_hi) * c1, vget_high_s32(a_hi) * c1
  // ac2: vget_low_s32(a_lo) * c2, vget_high_s32(a_lo) * c2,
  //      vget_low_s32(a_hi) * c2, vget_high_s32(a_hi) * c2
  int64x2_t ac1[4];
  int64x2_t ac2[4];
  int64x2_t sum[4];
  int64x2_t diff[4];

  ac1[0] = vmull_n_s32(vget_low_s32(a_lo), constant1);
  ac1[1] = vmull_n_s32(vget_high_s32(a_lo), constant1);
  ac1[2] = vmull_n_s32(vget_low_s32(a_hi), constant1);
  ac1[3] = vmull_n_s32(vget_high_s32(a_hi), constant1);
  ac2[0] = vmull_n_s32(vget_low_s32(a_lo), constant2);
  ac2[1] = vmull_n_s32(vget_high_s32(a_lo), constant2);
  ac2[2] = vmull_n_s32(vget_low_s32(a_hi), constant2);
  ac2[3] = vmull_n_s32(vget_high_s32(a_hi), constant2);

  sum[0] = vmlal_n_s32(ac1[0], vget_low_s32(b_lo), constant2);
  sum[1] = vmlal_n_s32(ac1[1], vget_high_s32(b_lo), constant2);
  sum[2] = vmlal_n_s32(ac1[2], vget_low_s32(b_hi), constant2);
  sum[3] = vmlal_n_s32(ac1[3], vget_high_s32(b_hi), constant2);
  *add_lo = vcombine_s32(vrshrn_n_s64(sum[0], DCT_CONST_BITS),
                         vrshrn_n_s64(sum[1], DCT_CONST_BITS));
  *add_hi = vcombine_s32(vrshrn_n_s64(sum[2], DCT_CONST_BITS),
                         vrshrn_n_s64(sum[3], DCT_CONST_BITS));

  diff[0] = vmlsl_n_s32(ac2[0], vget_low_s32(b_lo), constant1);
  diff[1] = vmlsl_n_s32(ac2[1], vget_high_s32(b_lo), constant1);
  diff[2] = vmlsl_n_s32(ac2[2], vget_low_s32(b_hi), constant1);
  diff[3] = vmlsl_n_s32(ac2[3], vget_high_s32(b_hi), constant1);
  *sub_lo = vcombine_s32(vrshrn_n_s64(diff[0], DCT_CONST_BITS),
                         vrshrn_n_s64(diff[1], DCT_CONST_BITS));
  *sub_hi = vcombine_s32(vrshrn_n_s64(diff[2], DCT_CONST_BITS),
                         vrshrn_n_s64(diff[3], DCT_CONST_BITS));
}

// fdct_round_shift(a * c1 +/- b * c2)
// Original Variant that performs normal implementation on full vector
// more accurate does 32-bit processing, takes and returns 32-bit values
// returns narrowed results
static INLINE void butterfly_two_coeff_s16_s32_noround(
    const int16x4_t a_lo, const int16x4_t a_hi, const int16x4_t b_lo,
    const int16x4_t b_hi, const tran_coef_t constant1,
    const tran_coef_t constant2, int32x4_t *add_lo, int32x4_t *add_hi,
    int32x4_t *sub_lo, int32x4_t *sub_hi) {
  const int32x4_t a1 = vmull_n_s16(a_lo, constant1);
  const int32x4_t a2 = vmull_n_s16(a_hi, constant1);
  const int32x4_t a3 = vmull_n_s16(a_lo, constant2);
  const int32x4_t a4 = vmull_n_s16(a_hi, constant2);
  *add_lo = vmlal_n_s16(a1, b_lo, constant2);
  *add_hi = vmlal_n_s16(a2, b_hi, constant2);
  *sub_lo = vmlsl_n_s16(a3, b_lo, constant1);
  *sub_hi = vmlsl_n_s16(a4, b_hi, constant1);
}

// fdct_round_shift(a * c1 +/- b * c2)
// Original Variant that performs normal implementation on full vector
// more accurate does 32-bit processing, takes and returns 32-bit values
// returns narrowed results
static INLINE void butterfly_two_coeff_s32_noround(
    const int32x4_t a_lo, const int32x4_t a_hi, const int32x4_t b_lo,
    const int32x4_t b_hi, const tran_coef_t constant1,
    const tran_coef_t constant2, int32x4_t *add_lo, int32x4_t *add_hi,
    int32x4_t *sub_lo, int32x4_t *sub_hi) {
  const int32x4_t a1 = vmulq_n_s32(a_lo, constant1);
  const int32x4_t a2 = vmulq_n_s32(a_hi, constant1);
  const int32x4_t a3 = vmulq_n_s32(a_lo, constant2);
  const int32x4_t a4 = vmulq_n_s32(a_hi, constant2);
  *add_lo = vmlaq_n_s32(a1, b_lo, constant2);
  *add_hi = vmlaq_n_s32(a2, b_hi, constant2);
  *sub_lo = vmlsq_n_s32(a3, b_lo, constant1);
  *sub_hi = vmlsq_n_s32(a4, b_hi, constant1);
}

// fdct_round_shift(a * c1 +/- b * c2)
// Variant that performs normal implementation on half vector
// more accurate does 32-bit processing, takes and returns 16-bit values
// returns narrowed results
static INLINE void butterfly_two_coeff_half(const int16x4_t a,
                                            const int16x4_t b,
                                            const tran_coef_t constant1,
                                            const tran_coef_t constant2,
                                            int16x4_t *add, int16x4_t *sub) {
  const int32x4_t a1 = vmull_n_s16(a, constant1);
  const int32x4_t a2 = vmull_n_s16(a, constant2);
  const int32x4_t sum = vmlal_n_s16(a1, b, constant2);
  const int32x4_t diff = vmlsl_n_s16(a2, b, constant1);
  *add = vqrshrn_n_s32(sum, DCT_CONST_BITS);
  *sub = vqrshrn_n_s32(diff, DCT_CONST_BITS);
}

// fdct_round_shift(a * c1 +/- b * c2)
// Original Variant that performs normal implementation on full vector
// more accurate does 32-bit processing, takes and returns 16-bit values
// returns narrowed results
static INLINE void butterfly_two_coeff(const int16x8_t a, const int16x8_t b,
                                       const tran_coef_t constant1,
                                       const tran_coef_t constant2,
                                       int16x8_t *add, int16x8_t *sub) {
  const int32x4_t a1 = vmull_n_s16(vget_low_s16(a), constant1);
  const int32x4_t a2 = vmull_n_s16(vget_high_s16(a), constant1);
  const int32x4_t a3 = vmull_n_s16(vget_low_s16(a), constant2);
  const int32x4_t a4 = vmull_n_s16(vget_high_s16(a), constant2);
  const int32x4_t sum0 = vmlal_n_s16(a1, vget_low_s16(b), constant2);
  const int32x4_t sum1 = vmlal_n_s16(a2, vget_high_s16(b), constant2);
  const int32x4_t diff0 = vmlsl_n_s16(a3, vget_low_s16(b), constant1);
  const int32x4_t diff1 = vmlsl_n_s16(a4, vget_high_s16(b), constant1);
  const int16x4_t rounded0 = vqrshrn_n_s32(sum0, DCT_CONST_BITS);
  const int16x4_t rounded1 = vqrshrn_n_s32(sum1, DCT_CONST_BITS);
  const int16x4_t rounded2 = vqrshrn_n_s32(diff0, DCT_CONST_BITS);
  const int16x4_t rounded3 = vqrshrn_n_s32(diff1, DCT_CONST_BITS);
  *add = vcombine_s16(rounded0, rounded1);
  *sub = vcombine_s16(rounded2, rounded3);
}

// fdct_round_shift(a * c1 +/- b * c2)
// Original Variant that performs normal implementation on full vector
// more accurate does 32-bit processing, takes and returns 32-bit values
// returns narrowed results
static INLINE void butterfly_two_coeff_s32(
    const int32x4_t a_lo, const int32x4_t a_hi, const int32x4_t b_lo,
    const int32x4_t b_hi, const tran_coef_t constant1,
    const tran_coef_t constant2, int32x4_t *add_lo, int32x4_t *add_hi,
    int32x4_t *sub_lo, int32x4_t *sub_hi) {
  const int32x4_t a1 = vmulq_n_s32(a_lo, constant1);
  const int32x4_t a2 = vmulq_n_s32(a_hi, constant1);
  const int32x4_t a3 = vmulq_n_s32(a_lo, constant2);
  const int32x4_t a4 = vmulq_n_s32(a_hi, constant2);
  const int32x4_t sum0 = vmlaq_n_s32(a1, b_lo, constant2);
  const int32x4_t sum1 = vmlaq_n_s32(a2, b_hi, constant2);
  const int32x4_t diff0 = vmlsq_n_s32(a3, b_lo, constant1);
  const int32x4_t diff1 = vmlsq_n_s32(a4, b_hi, constant1);
  *add_lo = vrshrq_n_s32(sum0, DCT_CONST_BITS);
  *add_hi = vrshrq_n_s32(sum1, DCT_CONST_BITS);
  *sub_lo = vrshrq_n_s32(diff0, DCT_CONST_BITS);
  *sub_hi = vrshrq_n_s32(diff1, DCT_CONST_BITS);
}

// Add 1 if positive, 2 if negative, and shift by 2.
// In practice, add 1, then add the sign bit, then shift without rounding.
static INLINE int16x8_t add_round_shift_s16(const int16x8_t a) {
  const int16x8_t one = vdupq_n_s16(1);
  const uint16x8_t a_u16 = vreinterpretq_u16_s16(a);
  const uint16x8_t a_sign_u16 = vshrq_n_u16(a_u16, 15);
  const int16x8_t a_sign_s16 = vreinterpretq_s16_u16(a_sign_u16);
  return vshrq_n_s16(vaddq_s16(vaddq_s16(a, a_sign_s16), one), 2);
}

// Add 1 if positive, 2 if negative, and shift by 2.
// In practice, add 1, then add the sign bit, then shift and round,
// return narrowed results
static INLINE int16x8_t add_round_shift_s32_narrow(const int32x4_t a_lo,
                                                   const int32x4_t a_hi) {
  const int32x4_t one = vdupq_n_s32(1);
  const uint32x4_t a_lo_u32 = vreinterpretq_u32_s32(a_lo);
  const uint32x4_t a_lo_sign_u32 = vshrq_n_u32(a_lo_u32, 31);
  const int32x4_t a_lo_sign_s32 = vreinterpretq_s32_u32(a_lo_sign_u32);
  const int16x4_t b_lo =
      vshrn_n_s32(vqaddq_s32(vqaddq_s32(a_lo, a_lo_sign_s32), one), 2);
  const uint32x4_t a_hi_u32 = vreinterpretq_u32_s32(a_hi);
  const uint32x4_t a_hi_sign_u32 = vshrq_n_u32(a_hi_u32, 31);
  const int32x4_t a_hi_sign_s32 = vreinterpretq_s32_u32(a_hi_sign_u32);
  const int16x4_t b_hi =
      vshrn_n_s32(vqaddq_s32(vqaddq_s32(a_hi, a_hi_sign_s32), one), 2);
  return vcombine_s16(b_lo, b_hi);
}

// Add 1 if negative, and shift by 1.
// In practice, add the sign bit, then shift and round
static INLINE int32x4_t add_round_shift_half_s32(const int32x4_t a) {
  const uint32x4_t a_u32 = vreinterpretq_u32_s32(a);
  const uint32x4_t a_sign_u32 = vshrq_n_u32(a_u32, 31);
  const int32x4_t a_sign_s32 = vreinterpretq_s32_u32(a_sign_u32);
  return vshrq_n_s32(vaddq_s32(a, a_sign_s32), 1);
}

// Add 1 if positive, 2 if negative, and shift by 2.
// In practice, add 1, then add the sign bit, then shift without rounding.
static INLINE int32x4_t add_round_shift_s32(const int32x4_t a) {
  const int32x4_t one = vdupq_n_s32(1);
  const uint32x4_t a_u32 = vreinterpretq_u32_s32(a);
  const uint32x4_t a_sign_u32 = vshrq_n_u32(a_u32, 31);
  const int32x4_t a_sign_s32 = vreinterpretq_s32_u32(a_sign_u32);
  return vshrq_n_s32(vaddq_s32(vaddq_s32(a, a_sign_s32), one), 2);
}

// Add 2 if positive, 1 if negative, and shift by 2.
// In practice, subtract the sign bit, then shift with rounding.
static INLINE int16x8_t sub_round_shift_s16(const int16x8_t a) {
  const uint16x8_t a_u16 = vreinterpretq_u16_s16(a);
  const uint16x8_t a_sign_u16 = vshrq_n_u16(a_u16, 15);
  const int16x8_t a_sign_s16 = vreinterpretq_s16_u16(a_sign_u16);
  return vrshrq_n_s16(vsubq_s16(a, a_sign_s16), 2);
}

// Add 2 if positive, 1 if negative, and shift by 2.
// In practice, subtract the sign bit, then shift with rounding.
static INLINE int32x4_t sub_round_shift_s32(const int32x4_t a) {
  const uint32x4_t a_u32 = vreinterpretq_u32_s32(a);
  const uint32x4_t a_sign_u32 = vshrq_n_u32(a_u32, 31);
  const int32x4_t a_sign_s32 = vreinterpretq_s32_u32(a_sign_u32);
  return vrshrq_n_s32(vsubq_s32(a, a_sign_s32), 2);
}

static INLINE int32x4_t add_s64_round_narrow(const int64x2_t *a /*[2]*/,
                                             const int64x2_t *b /*[2]*/) {
  int64x2_t result[2];
  result[0] = vaddq_s64(a[0], b[0]);
  result[1] = vaddq_s64(a[1], b[1]);
  return vcombine_s32(vrshrn_n_s64(result[0], DCT_CONST_BITS),
                      vrshrn_n_s64(result[1], DCT_CONST_BITS));
}

static INLINE int32x4_t sub_s64_round_narrow(const int64x2_t *a /*[2]*/,
                                             const int64x2_t *b /*[2]*/) {
  int64x2_t result[2];
  result[0] = vsubq_s64(a[0], b[0]);
  result[1] = vsubq_s64(a[1], b[1]);
  return vcombine_s32(vrshrn_n_s64(result[0], DCT_CONST_BITS),
                      vrshrn_n_s64(result[1], DCT_CONST_BITS));
}

static INLINE int32x4_t add_s32_s64_narrow(const int32x4_t a,
                                           const int32x4_t b) {
  int64x2_t a64[2], b64[2], result[2];
  a64[0] = vmovl_s32(vget_low_s32(a));
  a64[1] = vmovl_s32(vget_high_s32(a));
  b64[0] = vmovl_s32(vget_low_s32(b));
  b64[1] = vmovl_s32(vget_high_s32(b));
  result[0] = vaddq_s64(a64[0], b64[0]);
  result[1] = vaddq_s64(a64[1], b64[1]);
  return vcombine_s32(vmovn_s64(result[0]), vmovn_s64(result[1]));
}

static INLINE int32x4_t sub_s32_s64_narrow(const int32x4_t a,
                                           const int32x4_t b) {
  int64x2_t a64[2], b64[2], result[2];
  a64[0] = vmovl_s32(vget_low_s32(a));
  a64[1] = vmovl_s32(vget_high_s32(a));
  b64[0] = vmovl_s32(vget_low_s32(b));
  b64[1] = vmovl_s32(vget_high_s32(b));
  result[0] = vsubq_s64(a64[0], b64[0]);
  result[1] = vsubq_s64(a64[1], b64[1]);
  return vcombine_s32(vmovn_s64(result[0]), vmovn_s64(result[1]));
}

#endif  // VPX_VPX_DSP_ARM_FDCT_NEON_H_
