/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/ppc/types_vsx.h"

// Negate 16-bit integers in a when the corresponding signed 16-bit
// integer in b is negative.
static INLINE int16x8_t vec_sign(int16x8_t a, int16x8_t b) {
  const int16x8_t mask = vec_sra(b, vec_shift_sign_s16);
  return vec_xor(vec_add(a, mask), mask);
}

// Sets the value of a 32-bit integers to 1 when the corresponding value in a is
// negative.
static INLINE int32x4_t vec_is_neg(int32x4_t a) {
  return vec_sr(a, vec_shift_sign_s32);
}

// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit
// integers, and return the high 16 bits of the intermediate integers.
// (a * b) >> 16
static INLINE int16x8_t vec_mulhi(int16x8_t a, int16x8_t b) {
  // madds does ((A * B) >>15) + C, we need >> 16, so we perform an extra right
  // shift.
  return vec_sra(vec_madds(a, b, vec_zeros_s16), vec_ones_u16);
}

// Quantization function used for 4x4, 8x8 and 16x16 blocks.
static INLINE int16x8_t quantize_coeff(int16x8_t coeff, int16x8_t coeff_abs,
                                       int16x8_t round, int16x8_t quant,
                                       int16x8_t quant_shift, bool16x8_t mask) {
  const int16x8_t rounded = vec_vaddshs(coeff_abs, round);
  int16x8_t qcoeff = vec_mulhi(rounded, quant);
  qcoeff = vec_add(qcoeff, rounded);
  qcoeff = vec_mulhi(qcoeff, quant_shift);
  qcoeff = vec_sign(qcoeff, coeff);
  return vec_and(qcoeff, mask);
}

// Quantization function used for 32x32 blocks.
static INLINE int16x8_t quantize_coeff_32(int16x8_t coeff, int16x8_t coeff_abs,
                                          int16x8_t round, int16x8_t quant,
                                          int16x8_t quant_shift,
                                          bool16x8_t mask) {
  const int16x8_t rounded = vec_vaddshs(coeff_abs, round);
  int16x8_t qcoeff = vec_mulhi(rounded, quant);
  qcoeff = vec_add(qcoeff, rounded);
  // 32x32 blocks require an extra multiplication by 2, this compensates for the
  // extra right shift added in vec_mulhi, as such vec_madds can be used
  // directly instead of vec_mulhi (((a * b) >> 15) >> 1) << 1 == (a * b >> 15)
  qcoeff = vec_madds(qcoeff, quant_shift, vec_zeros_s16);
  qcoeff = vec_sign(qcoeff, coeff);
  return vec_and(qcoeff, mask);
}

// DeQuantization function used for 32x32 blocks. Quantized coeff of 32x32
// blocks are twice as big as for other block sizes. As such, using
// vec_mladd results in overflow.
static INLINE int16x8_t dequantize_coeff_32(int16x8_t qcoeff,
                                            int16x8_t dequant) {
  int32x4_t dqcoeffe = vec_mule(qcoeff, dequant);
  int32x4_t dqcoeffo = vec_mulo(qcoeff, dequant);
  // Add 1 if negative to round towards zero because the C uses division.
  dqcoeffe = vec_add(dqcoeffe, vec_is_neg(dqcoeffe));
  dqcoeffo = vec_add(dqcoeffo, vec_is_neg(dqcoeffo));
  dqcoeffe = vec_sra(dqcoeffe, vec_ones_u32);
  dqcoeffo = vec_sra(dqcoeffo, vec_ones_u32);
  return (int16x8_t)vec_perm(dqcoeffe, dqcoeffo, vec_perm_odd_even_pack);
}

static INLINE int16x8_t nonzero_scanindex(int16x8_t qcoeff,
                                          const int16_t *iscan_ptr, int index) {
  int16x8_t scan = vec_vsx_ld(index, iscan_ptr);
  bool16x8_t zero_coeff = vec_cmpeq(qcoeff, vec_zeros_s16);
  return vec_andc(scan, zero_coeff);
}

// Compare packed 16-bit integers across a, and return the maximum value in
// every element. Returns a vector containing the biggest value across vector a.
static INLINE int16x8_t vec_max_across(int16x8_t a) {
  a = vec_max(a, vec_perm(a, a, vec_perm64));
  a = vec_max(a, vec_perm(a, a, vec_perm32));
  return vec_max(a, vec_perm(a, a, vec_perm16));
}

void vpx_quantize_b_vsx(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                        const int16_t *zbin_ptr, const int16_t *round_ptr,
                        const int16_t *quant_ptr,
                        const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
                        tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,
                        uint16_t *eob_ptr, const int16_t *scan_ptr,
                        const int16_t *iscan_ptr) {
  int16x8_t qcoeff0, qcoeff1, dqcoeff0, dqcoeff1, eob;
  bool16x8_t zero_mask0, zero_mask1;

  // First set of 8 coeff starts with DC + 7 AC
  int16x8_t zbin = vec_vsx_ld(0, zbin_ptr);
  int16x8_t round = vec_vsx_ld(0, round_ptr);
  int16x8_t quant = vec_vsx_ld(0, quant_ptr);
  int16x8_t dequant = vec_vsx_ld(0, dequant_ptr);
  int16x8_t quant_shift = vec_vsx_ld(0, quant_shift_ptr);

  int16x8_t coeff0 = vec_vsx_ld(0, coeff_ptr);
  int16x8_t coeff1 = vec_vsx_ld(16, coeff_ptr);

  int16x8_t coeff0_abs = vec_abs(coeff0);
  int16x8_t coeff1_abs = vec_abs(coeff1);

  zero_mask0 = vec_cmpge(coeff0_abs, zbin);
  zbin = vec_splat(zbin, 1);
  zero_mask1 = vec_cmpge(coeff1_abs, zbin);

  (void)scan_ptr;

  qcoeff0 =
      quantize_coeff(coeff0, coeff0_abs, round, quant, quant_shift, zero_mask0);
  vec_vsx_st(qcoeff0, 0, qcoeff_ptr);
  round = vec_splat(round, 1);
  quant = vec_splat(quant, 1);
  quant_shift = vec_splat(quant_shift, 1);
  qcoeff1 =
      quantize_coeff(coeff1, coeff1_abs, round, quant, quant_shift, zero_mask1);
  vec_vsx_st(qcoeff1, 16, qcoeff_ptr);

  dqcoeff0 = vec_mladd(qcoeff0, dequant, vec_zeros_s16);
  vec_vsx_st(dqcoeff0, 0, dqcoeff_ptr);
  dequant = vec_splat(dequant, 1);
  dqcoeff1 = vec_mladd(qcoeff1, dequant, vec_zeros_s16);
  vec_vsx_st(dqcoeff1, 16, dqcoeff_ptr);

  eob = vec_max(nonzero_scanindex(qcoeff0, iscan_ptr, 0),
                nonzero_scanindex(qcoeff1, iscan_ptr, 16));

  if (n_coeffs > 16) {
    int index = 16;
    int off0 = 32;
    int off1 = 48;
    int off2 = 64;
    do {
      int16x8_t coeff2, coeff2_abs, qcoeff2, dqcoeff2, eob2;
      bool16x8_t zero_mask2;
      coeff0 = vec_vsx_ld(off0, coeff_ptr);
      coeff1 = vec_vsx_ld(off1, coeff_ptr);
      coeff2 = vec_vsx_ld(off2, coeff_ptr);
      coeff0_abs = vec_abs(coeff0);
      coeff1_abs = vec_abs(coeff1);
      coeff2_abs = vec_abs(coeff2);
      zero_mask0 = vec_cmpge(coeff0_abs, zbin);
      zero_mask1 = vec_cmpge(coeff1_abs, zbin);
      zero_mask2 = vec_cmpge(coeff2_abs, zbin);
      qcoeff0 = quantize_coeff(coeff0, coeff0_abs, round, quant, quant_shift,
                               zero_mask0);
      qcoeff1 = quantize_coeff(coeff1, coeff1_abs, round, quant, quant_shift,
                               zero_mask1);
      qcoeff2 = quantize_coeff(coeff2, coeff2_abs, round, quant, quant_shift,
                               zero_mask2);
      vec_vsx_st(qcoeff0, off0, qcoeff_ptr);
      vec_vsx_st(qcoeff1, off1, qcoeff_ptr);
      vec_vsx_st(qcoeff2, off2, qcoeff_ptr);

      dqcoeff0 = vec_mladd(qcoeff0, dequant, vec_zeros_s16);
      dqcoeff1 = vec_mladd(qcoeff1, dequant, vec_zeros_s16);
      dqcoeff2 = vec_mladd(qcoeff2, dequant, vec_zeros_s16);

      vec_vsx_st(dqcoeff0, off0, dqcoeff_ptr);
      vec_vsx_st(dqcoeff1, off1, dqcoeff_ptr);
      vec_vsx_st(dqcoeff2, off2, dqcoeff_ptr);

      eob = vec_max(eob, nonzero_scanindex(qcoeff0, iscan_ptr, off0));
      eob2 = vec_max(nonzero_scanindex(qcoeff1, iscan_ptr, off1),
                     nonzero_scanindex(qcoeff2, iscan_ptr, off2));
      eob = vec_max(eob, eob2);

      index += 24;
      off0 += 48;
      off1 += 48;
      off2 += 48;
    } while (index < n_coeffs);
  }

  eob = vec_max_across(eob);
  *eob_ptr = eob[0];
}

void vpx_quantize_b_32x32_vsx(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                              const int16_t *zbin_ptr, const int16_t *round_ptr,
                              const int16_t *quant_ptr,
                              const int16_t *quant_shift_ptr,
                              tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                              const int16_t *dequant_ptr, uint16_t *eob_ptr,
                              const int16_t *scan_ptr,
                              const int16_t *iscan_ptr) {
  // In stage 1, we quantize 16 coeffs (DC + 15 AC)
  // In stage 2, we loop 42 times and quantize 24 coeffs per iteration
  // (32 * 32 - 16) / 24 = 42
  int num_itr = 42;
  // Offsets are in bytes, 16 coeffs = 32 bytes
  int off0 = 32;
  int off1 = 48;
  int off2 = 64;

  int16x8_t qcoeff0, qcoeff1, eob;
  bool16x8_t zero_mask0, zero_mask1;

  int16x8_t zbin = vec_vsx_ld(0, zbin_ptr);
  int16x8_t round = vec_vsx_ld(0, round_ptr);
  int16x8_t quant = vec_vsx_ld(0, quant_ptr);
  int16x8_t dequant = vec_vsx_ld(0, dequant_ptr);
  int16x8_t quant_shift = vec_vsx_ld(0, quant_shift_ptr);

  int16x8_t coeff0 = vec_vsx_ld(0, coeff_ptr);
  int16x8_t coeff1 = vec_vsx_ld(16, coeff_ptr);

  int16x8_t coeff0_abs = vec_abs(coeff0);
  int16x8_t coeff1_abs = vec_abs(coeff1);

  (void)scan_ptr;
  (void)n_coeffs;

  // 32x32 quantization requires that zbin and round be divided by 2
  zbin = vec_sra(vec_add(zbin, vec_ones_s16), vec_ones_u16);
  round = vec_sra(vec_add(round, vec_ones_s16), vec_ones_u16);

  zero_mask0 = vec_cmpge(coeff0_abs, zbin);
  zbin = vec_splat(zbin, 1);  // remove DC from zbin
  zero_mask1 = vec_cmpge(coeff1_abs, zbin);

  qcoeff0 = quantize_coeff_32(coeff0, coeff0_abs, round, quant, quant_shift,
                              zero_mask0);
  round = vec_splat(round, 1);              // remove DC from round
  quant = vec_splat(quant, 1);              // remove DC from quant
  quant_shift = vec_splat(quant_shift, 1);  // remove DC from quant_shift
  qcoeff1 = quantize_coeff_32(coeff1, coeff1_abs, round, quant, quant_shift,
                              zero_mask1);

  vec_vsx_st(qcoeff0, 0, qcoeff_ptr);
  vec_vsx_st(qcoeff1, 16, qcoeff_ptr);

  vec_vsx_st(dequantize_coeff_32(qcoeff0, dequant), 0, dqcoeff_ptr);
  dequant = vec_splat(dequant, 1);  // remove DC from dequant
  vec_vsx_st(dequantize_coeff_32(qcoeff1, dequant), 16, dqcoeff_ptr);

  eob = vec_max(nonzero_scanindex(qcoeff0, iscan_ptr, 0),
                nonzero_scanindex(qcoeff1, iscan_ptr, 16));

  do {
    int16x8_t coeff2, coeff2_abs, qcoeff2, eob2;
    bool16x8_t zero_mask2;

    coeff0 = vec_vsx_ld(off0, coeff_ptr);
    coeff1 = vec_vsx_ld(off1, coeff_ptr);
    coeff2 = vec_vsx_ld(off2, coeff_ptr);

    coeff0_abs = vec_abs(coeff0);
    coeff1_abs = vec_abs(coeff1);
    coeff2_abs = vec_abs(coeff2);

    zero_mask0 = vec_cmpge(coeff0_abs, zbin);
    zero_mask1 = vec_cmpge(coeff1_abs, zbin);
    zero_mask2 = vec_cmpge(coeff2_abs, zbin);

    qcoeff0 = quantize_coeff_32(coeff0, coeff0_abs, round, quant, quant_shift,
                                zero_mask0);
    qcoeff1 = quantize_coeff_32(coeff1, coeff1_abs, round, quant, quant_shift,
                                zero_mask1);
    qcoeff2 = quantize_coeff_32(coeff2, coeff2_abs, round, quant, quant_shift,
                                zero_mask2);

    vec_vsx_st(qcoeff0, off0, qcoeff_ptr);
    vec_vsx_st(qcoeff1, off1, qcoeff_ptr);
    vec_vsx_st(qcoeff2, off2, qcoeff_ptr);

    vec_vsx_st(dequantize_coeff_32(qcoeff0, dequant), off0, dqcoeff_ptr);
    vec_vsx_st(dequantize_coeff_32(qcoeff1, dequant), off1, dqcoeff_ptr);
    vec_vsx_st(dequantize_coeff_32(qcoeff2, dequant), off2, dqcoeff_ptr);

    eob = vec_max(eob, nonzero_scanindex(qcoeff0, iscan_ptr, off0));
    eob2 = vec_max(nonzero_scanindex(qcoeff1, iscan_ptr, off1),
                   nonzero_scanindex(qcoeff2, iscan_ptr, off2));
    eob = vec_max(eob, eob2);

    // 24 int16_t is 48 bytes
    off0 += 48;
    off1 += 48;
    off2 += 48;
    num_itr--;
  } while (num_itr != 0);

  eob = vec_max_across(eob);
  *eob_ptr = eob[0];
}
