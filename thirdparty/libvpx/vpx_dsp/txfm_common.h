/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_TXFM_COMMON_H_
#define VPX_VPX_DSP_TXFM_COMMON_H_

#include "vpx_dsp/vpx_dsp_common.h"

// Constants and Macros used by all idct/dct functions
#define DCT_CONST_BITS 14
#define DCT_CONST_ROUNDING (1 << (DCT_CONST_BITS - 1))

#define UNIT_QUANT_SHIFT 2
#define UNIT_QUANT_FACTOR (1 << UNIT_QUANT_SHIFT)

// Constants:
//  for (int i = 1; i< 32; ++i)
//    printf("static const int cospi_%d_64 = %.0f;\n", i,
//           round(16384 * cos(i*M_PI/64)));
// Note: sin(k*Pi/64) = cos((32-k)*Pi/64)
static const tran_coef_t cospi_1_64 = 16364;
static const tran_coef_t cospi_2_64 = 16305;
static const tran_coef_t cospi_3_64 = 16207;
static const tran_coef_t cospi_4_64 = 16069;
static const tran_coef_t cospi_5_64 = 15893;
static const tran_coef_t cospi_6_64 = 15679;
static const tran_coef_t cospi_7_64 = 15426;
static const tran_coef_t cospi_8_64 = 15137;
static const tran_coef_t cospi_9_64 = 14811;
static const tran_coef_t cospi_10_64 = 14449;
static const tran_coef_t cospi_11_64 = 14053;
static const tran_coef_t cospi_12_64 = 13623;
static const tran_coef_t cospi_13_64 = 13160;
static const tran_coef_t cospi_14_64 = 12665;
static const tran_coef_t cospi_15_64 = 12140;
static const tran_coef_t cospi_16_64 = 11585;
static const tran_coef_t cospi_17_64 = 11003;
static const tran_coef_t cospi_18_64 = 10394;
static const tran_coef_t cospi_19_64 = 9760;
static const tran_coef_t cospi_20_64 = 9102;
static const tran_coef_t cospi_21_64 = 8423;
static const tran_coef_t cospi_22_64 = 7723;
static const tran_coef_t cospi_23_64 = 7005;
static const tran_coef_t cospi_24_64 = 6270;
static const tran_coef_t cospi_25_64 = 5520;
static const tran_coef_t cospi_26_64 = 4756;
static const tran_coef_t cospi_27_64 = 3981;
static const tran_coef_t cospi_28_64 = 3196;
static const tran_coef_t cospi_29_64 = 2404;
static const tran_coef_t cospi_30_64 = 1606;
static const tran_coef_t cospi_31_64 = 804;

//  16384 * sqrt(2) * sin(kPi/9) * 2 / 3
static const tran_coef_t sinpi_1_9 = 5283;
static const tran_coef_t sinpi_2_9 = 9929;
static const tran_coef_t sinpi_3_9 = 13377;
static const tran_coef_t sinpi_4_9 = 15212;

#endif  // VPX_VPX_DSP_TXFM_COMMON_H_
