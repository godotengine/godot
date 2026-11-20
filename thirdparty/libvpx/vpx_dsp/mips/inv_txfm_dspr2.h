/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_MIPS_INV_TXFM_DSPR2_H_
#define VPX_VPX_DSP_MIPS_INV_TXFM_DSPR2_H_

#include <assert.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/inv_txfm.h"
#include "vpx_dsp/mips/common_dspr2.h"

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_DSPR2
#define DCT_CONST_ROUND_SHIFT_TWICE_COSPI_16_64(input)                         \
  ({                                                                           \
    int32_t tmp, out;                                                          \
    int dct_cost_rounding = DCT_CONST_ROUNDING;                                \
    int in = input;                                                            \
                                                                               \
    __asm__ __volatile__(/* out = dct_const_round_shift(dc *  cospi_16_64); */ \
                         "mtlo     %[dct_cost_rounding],   $ac1              " \
                         "                \n\t"                                \
                         "mthi     $zero,                  $ac1              " \
                         "                \n\t"                                \
                         "madd     $ac1,                   %[in],            " \
                         "%[cospi_16_64]  \n\t"                                \
                         "extp     %[tmp],                 $ac1,             " \
                         "31              \n\t"                                \
                                                                               \
                         /* out = dct_const_round_shift(out * cospi_16_64); */ \
                         "mtlo     %[dct_cost_rounding],   $ac2              " \
                         "                \n\t"                                \
                         "mthi     $zero,                  $ac2              " \
                         "                \n\t"                                \
                         "madd     $ac2,                   %[tmp],           " \
                         "%[cospi_16_64]  \n\t"                                \
                         "extp     %[out],                 $ac2,             " \
                         "31              \n\t"                                \
                                                                               \
                         : [tmp] "=&r"(tmp), [out] "=r"(out)                   \
                         : [in] "r"(in),                                       \
                           [dct_cost_rounding] "r"(dct_cost_rounding),         \
                           [cospi_16_64] "r"(cospi_16_64));                    \
    out;                                                                       \
  })

void vpx_idct32_cols_add_blk_dspr2(int16_t *input, uint8_t *dest, int stride);
void vpx_idct4_rows_dspr2(const int16_t *input, int16_t *output);
void vpx_idct4_columns_add_blk_dspr2(int16_t *input, uint8_t *dest, int stride);
void iadst4_dspr2(const int16_t *input, int16_t *output);
void idct8_rows_dspr2(const int16_t *input, int16_t *output, uint32_t no_rows);
void idct8_columns_add_blk_dspr2(int16_t *input, uint8_t *dest, int stride);
void iadst8_dspr2(const int16_t *input, int16_t *output);
void idct16_rows_dspr2(const int16_t *input, int16_t *output, uint32_t no_rows);
void idct16_cols_add_blk_dspr2(int16_t *input, uint8_t *dest, int stride);
void iadst16_dspr2(const int16_t *input, int16_t *output);

#endif  // #if HAVE_DSPR2
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_MIPS_INV_TXFM_DSPR2_H_
