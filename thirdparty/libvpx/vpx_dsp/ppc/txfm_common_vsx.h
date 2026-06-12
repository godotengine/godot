/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_PPC_TXFM_COMMON_VSX_H_
#define VPX_VPX_DSP_PPC_TXFM_COMMON_VSX_H_

#include "vpx_dsp/ppc/types_vsx.h"

static const int32x4_t vec_dct_const_rounding = { 8192, 8192, 8192, 8192 };

static const uint32x4_t vec_dct_const_bits = { 14, 14, 14, 14 };

static const uint16x8_t vec_dct_scale_log2 = { 2, 2, 2, 2, 2, 2, 2, 2 };

static const int16x8_t cospi1_v = { 16364, 16364, 16364, 16364,
                                    16364, 16364, 16364, 16364 };
static const int16x8_t cospi2_v = { 16305, 16305, 16305, 16305,
                                    16305, 16305, 16305, 16305 };
static const int16x8_t cospi3_v = { 16207, 16207, 16207, 16207,
                                    16207, 16207, 16207, 16207 };
static const int16x8_t cospi4_v = { 16069, 16069, 16069, 16069,
                                    16069, 16069, 16069, 16069 };
static const int16x8_t cospi4m_v = { -16069, -16069, -16069, -16069,
                                     -16069, -16069, -16069, -16069 };
static const int16x8_t cospi5_v = { 15893, 15893, 15893, 15893,
                                    15893, 15893, 15893, 15893 };
static const int16x8_t cospi6_v = { 15679, 15679, 15679, 15679,
                                    15679, 15679, 15679, 15679 };
static const int16x8_t cospi7_v = { 15426, 15426, 15426, 15426,
                                    15426, 15426, 15426, 15426 };
static const int16x8_t cospi8_v = { 15137, 15137, 15137, 15137,
                                    15137, 15137, 15137, 15137 };
static const int16x8_t cospi8m_v = { -15137, -15137, -15137, -15137,
                                     -15137, -15137, -15137, -15137 };
static const int16x8_t cospi9_v = { 14811, 14811, 14811, 14811,
                                    14811, 14811, 14811, 14811 };
static const int16x8_t cospi10_v = { 14449, 14449, 14449, 14449,
                                     14449, 14449, 14449, 14449 };
static const int16x8_t cospi11_v = { 14053, 14053, 14053, 14053,
                                     14053, 14053, 14053, 14053 };
static const int16x8_t cospi12_v = { 13623, 13623, 13623, 13623,
                                     13623, 13623, 13623, 13623 };
static const int16x8_t cospi13_v = { 13160, 13160, 13160, 13160,
                                     13160, 13160, 13160, 13160 };
static const int16x8_t cospi14_v = { 12665, 12665, 12665, 12665,
                                     12665, 12665, 12665, 12665 };
static const int16x8_t cospi15_v = { 12140, 12140, 12140, 12140,
                                     12140, 12140, 12140, 12140 };
static const int16x8_t cospi16_v = { 11585, 11585, 11585, 11585,
                                     11585, 11585, 11585, 11585 };
static const int16x8_t cospi17_v = { 11003, 11003, 11003, 11003,
                                     11003, 11003, 11003, 11003 };
static const int16x8_t cospi18_v = { 10394, 10394, 10394, 10394,
                                     10394, 10394, 10394, 10394 };
static const int16x8_t cospi19_v = { 9760, 9760, 9760, 9760,
                                     9760, 9760, 9760, 9760 };
static const int16x8_t cospi20_v = { 9102, 9102, 9102, 9102,
                                     9102, 9102, 9102, 9102 };
static const int16x8_t cospi20m_v = { -9102, -9102, -9102, -9102,
                                      -9102, -9102, -9102, -9102 };
static const int16x8_t cospi21_v = { 8423, 8423, 8423, 8423,
                                     8423, 8423, 8423, 8423 };
static const int16x8_t cospi22_v = { 7723, 7723, 7723, 7723,
                                     7723, 7723, 7723, 7723 };
static const int16x8_t cospi23_v = { 7005, 7005, 7005, 7005,
                                     7005, 7005, 7005, 7005 };
static const int16x8_t cospi24_v = { 6270, 6270, 6270, 6270,
                                     6270, 6270, 6270, 6270 };
static const int16x8_t cospi25_v = { 5520, 5520, 5520, 5520,
                                     5520, 5520, 5520, 5520 };
static const int16x8_t cospi26_v = { 4756, 4756, 4756, 4756,
                                     4756, 4756, 4756, 4756 };
static const int16x8_t cospi27_v = { 3981, 3981, 3981, 3981,
                                     3981, 3981, 3981, 3981 };
static const int16x8_t cospi28_v = { 3196, 3196, 3196, 3196,
                                     3196, 3196, 3196, 3196 };
static const int16x8_t cospi29_v = { 2404, 2404, 2404, 2404,
                                     2404, 2404, 2404, 2404 };
static const int16x8_t cospi30_v = { 1606, 1606, 1606, 1606,
                                     1606, 1606, 1606, 1606 };
static const int16x8_t cospi31_v = { 804, 804, 804, 804, 804, 804, 804, 804 };

#endif  // VPX_VPX_DSP_PPC_TXFM_COMMON_VSX_H_
