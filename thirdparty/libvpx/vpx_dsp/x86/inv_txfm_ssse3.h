/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_X86_INV_TXFM_SSSE3_H_
#define VPX_VPX_DSP_X86_INV_TXFM_SSSE3_H_

#include <tmmintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"
#include "vpx_dsp/x86/transpose_sse2.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

static INLINE void idct8x8_12_add_kernel_ssse3(__m128i *const io /* io[8] */) {
  const __m128i cp_28d_4d = dual_set_epi16(2 * cospi_28_64, 2 * cospi_4_64);
  const __m128i cp_n20d_12d = dual_set_epi16(-2 * cospi_20_64, 2 * cospi_12_64);
  const __m128i cp_8d_24d = dual_set_epi16(2 * cospi_8_64, 2 * cospi_24_64);
  const __m128i cp_16_16 = _mm_set1_epi16(cospi_16_64);
  const __m128i cp_16_n16 = pair_set_epi16(cospi_16_64, -cospi_16_64);
  const __m128i cospi_16_64d = _mm_set1_epi16((int16_t)(2 * cospi_16_64));
  const __m128i cospi_28_64d = _mm_set1_epi16((int16_t)(2 * cospi_28_64));
  const __m128i cospi_4_64d = _mm_set1_epi16((int16_t)(2 * cospi_4_64));
  const __m128i cospi_n20_64d = _mm_set1_epi16((int16_t)(-2 * cospi_20_64));
  const __m128i cospi_12_64d = _mm_set1_epi16((int16_t)(2 * cospi_12_64));
  const __m128i cospi_24_64d = _mm_set1_epi16((int16_t)(2 * cospi_24_64));
  const __m128i cospi_8_64d = _mm_set1_epi16((int16_t)(2 * cospi_8_64));
  __m128i step1[8], step2[8], tmp[4];

  // pass 1

  transpose_16bit_4x4(io, io);
  // io[0]: 00 10 20 30  01 11 21 31
  // io[1]: 02 12 22 32  03 13 23 33

  // stage 1
  tmp[0] = _mm_unpacklo_epi64(io[0], io[0]);
  tmp[1] = _mm_unpackhi_epi64(io[0], io[0]);
  tmp[2] = _mm_unpacklo_epi64(io[1], io[1]);
  tmp[3] = _mm_unpackhi_epi64(io[1], io[1]);
  step1[4] = _mm_mulhrs_epi16(tmp[1], cp_28d_4d);    // step1 4&7
  step1[5] = _mm_mulhrs_epi16(tmp[3], cp_n20d_12d);  // step1 5&6

  // stage 2
  step2[0] = _mm_mulhrs_epi16(tmp[0], cospi_16_64d);  // step2 0&1
  step2[2] = _mm_mulhrs_epi16(tmp[2], cp_8d_24d);     // step2 3&2
  step2[4] = _mm_add_epi16(step1[4], step1[5]);       // step2 4&7
  step2[5] = _mm_sub_epi16(step1[4], step1[5]);       // step2 5&6
  step2[6] = _mm_unpackhi_epi64(step2[5], step2[5]);  // step2 6

  // stage 3
  tmp[0] = _mm_unpacklo_epi16(step2[6], step2[5]);
  step1[5] = idct_calc_wraplow_sse2(cp_16_n16, cp_16_16, tmp[0]);  // step1 5&6
  tmp[0] = _mm_add_epi16(step2[0], step2[2]);                      // step1 0&1
  tmp[1] = _mm_sub_epi16(step2[0], step2[2]);                      // step1 3&2
  step1[2] = _mm_unpackhi_epi64(tmp[1], tmp[0]);                   // step1 2&1
  step1[3] = _mm_unpacklo_epi64(tmp[1], tmp[0]);                   // step1 3&0

  // stage 4
  tmp[0] = _mm_add_epi16(step1[3], step2[4]);  // output 3&0
  tmp[1] = _mm_add_epi16(step1[2], step1[5]);  // output 2&1
  tmp[2] = _mm_sub_epi16(step1[3], step2[4]);  // output 4&7
  tmp[3] = _mm_sub_epi16(step1[2], step1[5]);  // output 5&6

  // pass 2

  idct8x8_12_transpose_16bit_4x8(tmp, io);

  // stage 1
  step1[4] = _mm_mulhrs_epi16(io[1], cospi_28_64d);
  step1[7] = _mm_mulhrs_epi16(io[1], cospi_4_64d);
  step1[5] = _mm_mulhrs_epi16(io[3], cospi_n20_64d);
  step1[6] = _mm_mulhrs_epi16(io[3], cospi_12_64d);

  // stage 2
  step2[0] = _mm_mulhrs_epi16(io[0], cospi_16_64d);  // step2[1] = step2[0]
  step2[2] = _mm_mulhrs_epi16(io[2], cospi_24_64d);
  step2[3] = _mm_mulhrs_epi16(io[2], cospi_8_64d);
  step2[4] = _mm_add_epi16(step1[4], step1[5]);
  step2[5] = _mm_sub_epi16(step1[4], step1[5]);
  step2[6] = _mm_sub_epi16(step1[7], step1[6]);
  step2[7] = _mm_add_epi16(step1[7], step1[6]);

  // stage 3
  step1[0] = _mm_add_epi16(step2[0], step2[3]);
  step1[1] = _mm_add_epi16(step2[0], step2[2]);
  step1[2] = _mm_sub_epi16(step2[0], step2[2]);
  step1[3] = _mm_sub_epi16(step2[0], step2[3]);
  butterfly(step2[6], step2[5], cospi_16_64, cospi_16_64, &step1[5], &step1[6]);

  // stage 4
  io[0] = _mm_add_epi16(step1[0], step2[7]);
  io[1] = _mm_add_epi16(step1[1], step1[6]);
  io[2] = _mm_add_epi16(step1[2], step1[5]);
  io[3] = _mm_add_epi16(step1[3], step2[4]);
  io[4] = _mm_sub_epi16(step1[3], step2[4]);
  io[5] = _mm_sub_epi16(step1[2], step1[5]);
  io[6] = _mm_sub_epi16(step1[1], step1[6]);
  io[7] = _mm_sub_epi16(step1[0], step2[7]);
}

void idct32_135_8x32_ssse3(const __m128i *const in, __m128i *const out);

#endif  // VPX_VPX_DSP_X86_INV_TXFM_SSSE3_H_
