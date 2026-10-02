/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"

#include "./vp9_rtcd.h"
#include "vpx_dsp/ppc/types_vsx.h"

// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit
// integers, and return the high 16 bits of the intermediate integers.
// (a * b) >> 16
// Note: Because this is done in 2 operations, a and b cannot both be UINT16_MIN
static INLINE int16x8_t vec_mulhi(int16x8_t a, int16x8_t b) {
  // madds does ((A * B) >> 15) + C, we need >> 16, so we perform an extra right
  // shift.
  return vec_sra(vec_madds(a, b, vec_zeros_s16), vec_ones_u16);
}

// Negate 16-bit integers in a when the corresponding signed 16-bit
// integer in b is negative.
static INLINE int16x8_t vec_sign(int16x8_t a, int16x8_t b) {
  const int16x8_t mask = vec_sra(b, vec_shift_sign_s16);
  return vec_xor(vec_add(a, mask), mask);
}

// Compare packed 16-bit integers across a, and return the maximum value in
// every element. Returns a vector containing the biggest value across vector a.
static INLINE int16x8_t vec_max_across(int16x8_t a) {
  a = vec_max(a, vec_perm(a, a, vec_perm64));
  a = vec_max(a, vec_perm(a, a, vec_perm32));
  return vec_max(a, vec_perm(a, a, vec_perm16));
}

void vp9_quantize_fp_vsx(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                         const int16_t *round_ptr, const int16_t *quant_ptr,
                         tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                         const int16_t *dequant_ptr, uint16_t *eob_ptr,
                         const int16_t *scan, const int16_t *iscan) {
  int16x8_t qcoeff0, qcoeff1, dqcoeff0, dqcoeff1, eob;
  bool16x8_t zero_coeff0, zero_coeff1;

  int16x8_t round = vec_vsx_ld(0, round_ptr);
  int16x8_t quant = vec_vsx_ld(0, quant_ptr);
  int16x8_t dequant = vec_vsx_ld(0, dequant_ptr);
  int16x8_t coeff0 = vec_vsx_ld(0, coeff_ptr);
  int16x8_t coeff1 = vec_vsx_ld(16, coeff_ptr);
  int16x8_t scan0 = vec_vsx_ld(0, iscan);
  int16x8_t scan1 = vec_vsx_ld(16, iscan);

  (void)scan;

  // First set of 8 coeff starts with DC + 7 AC
  qcoeff0 = vec_mulhi(vec_vaddshs(vec_abs(coeff0), round), quant);
  zero_coeff0 = vec_cmpeq(qcoeff0, vec_zeros_s16);
  qcoeff0 = vec_sign(qcoeff0, coeff0);
  vec_vsx_st(qcoeff0, 0, qcoeff_ptr);

  dqcoeff0 = vec_mladd(qcoeff0, dequant, vec_zeros_s16);
  vec_vsx_st(dqcoeff0, 0, dqcoeff_ptr);

  // Remove DC value from round and quant
  round = vec_splat(round, 1);
  quant = vec_splat(quant, 1);

  // Remove DC value from dequant
  dequant = vec_splat(dequant, 1);

  // Second set of 8 coeff starts with (all AC)
  qcoeff1 = vec_mulhi(vec_vaddshs(vec_abs(coeff1), round), quant);
  zero_coeff1 = vec_cmpeq(qcoeff1, vec_zeros_s16);
  qcoeff1 = vec_sign(qcoeff1, coeff1);
  vec_vsx_st(qcoeff1, 16, qcoeff_ptr);

  dqcoeff1 = vec_mladd(qcoeff1, dequant, vec_zeros_s16);
  vec_vsx_st(dqcoeff1, 16, dqcoeff_ptr);

  eob = vec_max(vec_or(scan0, zero_coeff0), vec_or(scan1, zero_coeff1));

  // We quantize 16 coeff up front (enough for a 4x4) and process 24 coeff per
  // loop iteration.
  // for 8x8: 16 + 2 x 24 = 64
  // for 16x16: 16 + 10 x 24 = 256
  if (n_coeffs > 16) {
    int16x8_t coeff2, qcoeff2, dqcoeff2, eob2, scan2;
    bool16x8_t zero_coeff2;

    int index = 16;
    int off0 = 32;
    int off1 = 48;
    int off2 = 64;

    do {
      coeff0 = vec_vsx_ld(off0, coeff_ptr);
      coeff1 = vec_vsx_ld(off1, coeff_ptr);
      coeff2 = vec_vsx_ld(off2, coeff_ptr);
      scan0 = vec_vsx_ld(off0, iscan);
      scan1 = vec_vsx_ld(off1, iscan);
      scan2 = vec_vsx_ld(off2, iscan);

      qcoeff0 = vec_mulhi(vec_vaddshs(vec_abs(coeff0), round), quant);
      zero_coeff0 = vec_cmpeq(qcoeff0, vec_zeros_s16);
      qcoeff0 = vec_sign(qcoeff0, coeff0);
      vec_vsx_st(qcoeff0, off0, qcoeff_ptr);
      dqcoeff0 = vec_mladd(qcoeff0, dequant, vec_zeros_s16);
      vec_vsx_st(dqcoeff0, off0, dqcoeff_ptr);

      qcoeff1 = vec_mulhi(vec_vaddshs(vec_abs(coeff1), round), quant);
      zero_coeff1 = vec_cmpeq(qcoeff1, vec_zeros_s16);
      qcoeff1 = vec_sign(qcoeff1, coeff1);
      vec_vsx_st(qcoeff1, off1, qcoeff_ptr);
      dqcoeff1 = vec_mladd(qcoeff1, dequant, vec_zeros_s16);
      vec_vsx_st(dqcoeff1, off1, dqcoeff_ptr);

      qcoeff2 = vec_mulhi(vec_vaddshs(vec_abs(coeff2), round), quant);
      zero_coeff2 = vec_cmpeq(qcoeff2, vec_zeros_s16);
      qcoeff2 = vec_sign(qcoeff2, coeff2);
      vec_vsx_st(qcoeff2, off2, qcoeff_ptr);
      dqcoeff2 = vec_mladd(qcoeff2, dequant, vec_zeros_s16);
      vec_vsx_st(dqcoeff2, off2, dqcoeff_ptr);

      eob = vec_max(eob, vec_or(scan0, zero_coeff0));
      eob2 = vec_max(vec_or(scan1, zero_coeff1), vec_or(scan2, zero_coeff2));
      eob = vec_max(eob, eob2);

      index += 24;
      off0 += 48;
      off1 += 48;
      off2 += 48;
    } while (index < n_coeffs);
  }

  eob = vec_max_across(eob);
  *eob_ptr = eob[0] + 1;
}

// Sets the value of a 32-bit integers to 1 when the corresponding value in a is
// negative.
static INLINE int32x4_t vec_is_neg(int32x4_t a) {
  return vec_sr(a, vec_shift_sign_s32);
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

void vp9_quantize_fp_32x32_vsx(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                               const int16_t *round_ptr,
                               const int16_t *quant_ptr, tran_low_t *qcoeff_ptr,
                               tran_low_t *dqcoeff_ptr,
                               const int16_t *dequant_ptr, uint16_t *eob_ptr,
                               const int16_t *scan, const int16_t *iscan) {
  // In stage 1, we quantize 16 coeffs (DC + 15 AC)
  // In stage 2, we loop 42 times and quantize 24 coeffs per iteration
  // (32 * 32 - 16) / 24 = 42
  int num_itr = 42;
  // Offsets are in bytes, 16 coeffs = 32 bytes
  int off0 = 32;
  int off1 = 48;
  int off2 = 64;

  int16x8_t qcoeff0, qcoeff1, dqcoeff0, dqcoeff1, eob;
  bool16x8_t mask0, mask1, zero_coeff0, zero_coeff1;

  int16x8_t round = vec_vsx_ld(0, round_ptr);
  int16x8_t quant = vec_vsx_ld(0, quant_ptr);
  int16x8_t dequant = vec_vsx_ld(0, dequant_ptr);
  int16x8_t coeff0 = vec_vsx_ld(0, coeff_ptr);
  int16x8_t coeff1 = vec_vsx_ld(16, coeff_ptr);
  int16x8_t scan0 = vec_vsx_ld(0, iscan);
  int16x8_t scan1 = vec_vsx_ld(16, iscan);
  int16x8_t thres = vec_sra(dequant, vec_splats((uint16_t)2));
  int16x8_t abs_coeff0 = vec_abs(coeff0);
  int16x8_t abs_coeff1 = vec_abs(coeff1);

  (void)scan;
  (void)n_coeffs;

  mask0 = vec_cmpge(abs_coeff0, thres);
  round = vec_sra(vec_add(round, vec_ones_s16), vec_ones_u16);
  // First set of 8 coeff starts with DC + 7 AC
  qcoeff0 = vec_madds(vec_vaddshs(abs_coeff0, round), quant, vec_zeros_s16);
  qcoeff0 = vec_and(qcoeff0, mask0);
  zero_coeff0 = vec_cmpeq(qcoeff0, vec_zeros_s16);
  qcoeff0 = vec_sign(qcoeff0, coeff0);
  vec_vsx_st(qcoeff0, 0, qcoeff_ptr);

  dqcoeff0 = dequantize_coeff_32(qcoeff0, dequant);
  vec_vsx_st(dqcoeff0, 0, dqcoeff_ptr);

  // Remove DC value from thres, round, quant and dequant
  thres = vec_splat(thres, 1);
  round = vec_splat(round, 1);
  quant = vec_splat(quant, 1);
  dequant = vec_splat(dequant, 1);

  mask1 = vec_cmpge(abs_coeff1, thres);

  // Second set of 8 coeff starts with (all AC)
  qcoeff1 =
      vec_madds(vec_vaddshs(vec_abs(coeff1), round), quant, vec_zeros_s16);
  qcoeff1 = vec_and(qcoeff1, mask1);
  zero_coeff1 = vec_cmpeq(qcoeff1, vec_zeros_s16);
  qcoeff1 = vec_sign(qcoeff1, coeff1);
  vec_vsx_st(qcoeff1, 16, qcoeff_ptr);

  dqcoeff1 = dequantize_coeff_32(qcoeff1, dequant);
  vec_vsx_st(dqcoeff1, 16, dqcoeff_ptr);

  eob = vec_max(vec_or(scan0, zero_coeff0), vec_or(scan1, zero_coeff1));

  do {
    int16x8_t coeff2, abs_coeff2, qcoeff2, dqcoeff2, eob2, scan2;
    bool16x8_t zero_coeff2, mask2;
    coeff0 = vec_vsx_ld(off0, coeff_ptr);
    coeff1 = vec_vsx_ld(off1, coeff_ptr);
    coeff2 = vec_vsx_ld(off2, coeff_ptr);
    scan0 = vec_vsx_ld(off0, iscan);
    scan1 = vec_vsx_ld(off1, iscan);
    scan2 = vec_vsx_ld(off2, iscan);

    abs_coeff0 = vec_abs(coeff0);
    abs_coeff1 = vec_abs(coeff1);
    abs_coeff2 = vec_abs(coeff2);

    qcoeff0 = vec_madds(vec_vaddshs(abs_coeff0, round), quant, vec_zeros_s16);
    qcoeff1 = vec_madds(vec_vaddshs(abs_coeff1, round), quant, vec_zeros_s16);
    qcoeff2 = vec_madds(vec_vaddshs(abs_coeff2, round), quant, vec_zeros_s16);

    mask0 = vec_cmpge(abs_coeff0, thres);
    mask1 = vec_cmpge(abs_coeff1, thres);
    mask2 = vec_cmpge(abs_coeff2, thres);

    qcoeff0 = vec_and(qcoeff0, mask0);
    qcoeff1 = vec_and(qcoeff1, mask1);
    qcoeff2 = vec_and(qcoeff2, mask2);

    zero_coeff0 = vec_cmpeq(qcoeff0, vec_zeros_s16);
    zero_coeff1 = vec_cmpeq(qcoeff1, vec_zeros_s16);
    zero_coeff2 = vec_cmpeq(qcoeff2, vec_zeros_s16);

    qcoeff0 = vec_sign(qcoeff0, coeff0);
    qcoeff1 = vec_sign(qcoeff1, coeff1);
    qcoeff2 = vec_sign(qcoeff2, coeff2);

    vec_vsx_st(qcoeff0, off0, qcoeff_ptr);
    vec_vsx_st(qcoeff1, off1, qcoeff_ptr);
    vec_vsx_st(qcoeff2, off2, qcoeff_ptr);

    dqcoeff0 = dequantize_coeff_32(qcoeff0, dequant);
    dqcoeff1 = dequantize_coeff_32(qcoeff1, dequant);
    dqcoeff2 = dequantize_coeff_32(qcoeff2, dequant);

    vec_vsx_st(dqcoeff0, off0, dqcoeff_ptr);
    vec_vsx_st(dqcoeff1, off1, dqcoeff_ptr);
    vec_vsx_st(dqcoeff2, off2, dqcoeff_ptr);

    eob = vec_max(eob, vec_or(scan0, zero_coeff0));
    eob2 = vec_max(vec_or(scan1, zero_coeff1), vec_or(scan2, zero_coeff2));
    eob = vec_max(eob, eob2);

    off0 += 48;
    off1 += 48;
    off2 += 48;
    num_itr--;
  } while (num_itr != 0);

  eob = vec_max_across(eob);
  *eob_ptr = eob[0] + 1;
}
