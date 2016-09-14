/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_DSP_INV_TXFM_H_
#define VPX_DSP_INV_TXFM_H_

#include <assert.h>

#include "./vpx_config.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_ports/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

static INLINE tran_high_t check_range(tran_high_t input) {
#if CONFIG_COEFFICIENT_RANGE_CHECKING
  // For valid VP9 input streams, intermediate stage coefficients should always
  // stay within the range of a signed 16 bit integer. Coefficients can go out
  // of this range for invalid/corrupt VP9 streams. However, strictly checking
  // this range for every intermediate coefficient can burdensome for a decoder,
  // therefore the following assertion is only enabled when configured with
  // --enable-coefficient-range-checking.
  assert(INT16_MIN <= input);
  assert(input <= INT16_MAX);
#endif  // CONFIG_COEFFICIENT_RANGE_CHECKING
  return input;
}

static INLINE tran_high_t dct_const_round_shift(tran_high_t input) {
  tran_high_t rv = ROUND_POWER_OF_TWO(input, DCT_CONST_BITS);
  return (tran_high_t)rv;
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE tran_high_t highbd_check_range(tran_high_t input,
                                             int bd) {
#if CONFIG_COEFFICIENT_RANGE_CHECKING
  // For valid highbitdepth VP9 streams, intermediate stage coefficients will
  // stay within the ranges:
  // - 8 bit: signed 16 bit integer
  // - 10 bit: signed 18 bit integer
  // - 12 bit: signed 20 bit integer
  const int32_t int_max = (1 << (7 + bd)) - 1;
  const int32_t int_min = -int_max - 1;
  assert(int_min <= input);
  assert(input <= int_max);
  (void) int_min;
#endif  // CONFIG_COEFFICIENT_RANGE_CHECKING
  (void) bd;
  return input;
}

static INLINE tran_high_t highbd_dct_const_round_shift(tran_high_t input) {
  tran_high_t rv = ROUND_POWER_OF_TWO(input, DCT_CONST_BITS);
  return (tran_high_t)rv;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

#if CONFIG_EMULATE_HARDWARE
// When CONFIG_EMULATE_HARDWARE is 1 the transform performs a
// non-normative method to handle overflows. A stream that causes
// overflows  in the inverse transform is considered invalid in VP9,
// and a hardware implementer is free to choose any reasonable
// method to handle overflows. However to aid in hardware
// verification they can use a specific implementation of the
// WRAPLOW() macro below that is identical to their intended
// hardware implementation (and also use configure options to trigger
// the C-implementation of the transform).
//
// The particular WRAPLOW implementation below performs strict
// overflow wrapping to match common hardware implementations.
// bd of 8 uses trans_low with 16bits, need to remove 16bits
// bd of 10 uses trans_low with 18bits, need to remove 14bits
// bd of 12 uses trans_low with 20bits, need to remove 12bits
// bd of x uses trans_low with 8+x bits, need to remove 24-x bits

#define WRAPLOW(x) ((((int32_t)check_range(x)) << 16) >> 16)
#if CONFIG_VP9_HIGHBITDEPTH
#define HIGHBD_WRAPLOW(x, bd) \
    ((((int32_t)highbd_check_range((x), bd)) << (24 - bd)) >> (24 - bd))
#endif  // CONFIG_VP9_HIGHBITDEPTH

#else   // CONFIG_EMULATE_HARDWARE

#define WRAPLOW(x) ((int32_t)check_range(x))
#if CONFIG_VP9_HIGHBITDEPTH
#define HIGHBD_WRAPLOW(x, bd) \
    ((int32_t)highbd_check_range((x), bd))
#endif  // CONFIG_VP9_HIGHBITDEPTH
#endif  // CONFIG_EMULATE_HARDWARE

void idct4_c(const tran_low_t *input, tran_low_t *output);
void idct8_c(const tran_low_t *input, tran_low_t *output);
void idct16_c(const tran_low_t *input, tran_low_t *output);
void idct32_c(const tran_low_t *input, tran_low_t *output);
void iadst4_c(const tran_low_t *input, tran_low_t *output);
void iadst8_c(const tran_low_t *input, tran_low_t *output);
void iadst16_c(const tran_low_t *input, tran_low_t *output);

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_idct4_c(const tran_low_t *input, tran_low_t *output, int bd);
void vpx_highbd_idct8_c(const tran_low_t *input, tran_low_t *output, int bd);
void vpx_highbd_idct16_c(const tran_low_t *input, tran_low_t *output, int bd);

void vpx_highbd_iadst4_c(const tran_low_t *input, tran_low_t *output, int bd);
void vpx_highbd_iadst8_c(const tran_low_t *input, tran_low_t *output, int bd);
void vpx_highbd_iadst16_c(const tran_low_t *input, tran_low_t *output, int bd);

static INLINE uint16_t highbd_clip_pixel_add(uint16_t dest, tran_high_t trans,
                                             int bd) {
  trans = HIGHBD_WRAPLOW(trans, bd);
  return clip_pixel_highbd(dest + (int)trans, bd);
}
#endif

static INLINE uint8_t clip_pixel_add(uint8_t dest, tran_high_t trans) {
  trans = WRAPLOW(trans);
  return clip_pixel(dest + (int)trans);
}
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_DSP_INV_TXFM_H_
