/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/inv_txfm_dspr2.h"
#include "vpx_dsp/txfm_common.h"

#if HAVE_DSPR2
void idct8_rows_dspr2(const int16_t *input, int16_t *output, uint32_t no_rows) {
  int step1_0, step1_1, step1_2, step1_3, step1_4, step1_5, step1_6, step1_7;
  const int const_2_power_13 = 8192;
  int Temp0, Temp1, Temp2, Temp3, Temp4;
  int i;

  for (i = no_rows; i--;) {
    __asm__ __volatile__(
        /*
          temp_1 = (input[0] + input[4]) * cospi_16_64;
          step2_0 = dct_const_round_shift(temp_1);

          temp_2 = (input[0] - input[4]) * cospi_16_64;
          step2_1 = dct_const_round_shift(temp_2);
        */
        "lh       %[Temp0],             0(%[input])                     \n\t"
        "lh       %[Temp1],             8(%[input])                     \n\t"
        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"
        "add      %[Temp2],             %[Temp0],       %[Temp1]        \n\t"
        "madd     $ac0,                 %[Temp2],       %[cospi_16_64]  \n\t"
        "extp     %[Temp4],             $ac0,           31              \n\t"

        "sub      %[Temp3],             %[Temp0],       %[Temp1]        \n\t"
        "madd     $ac1,                 %[Temp3],       %[cospi_16_64]  \n\t"
        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"
        "extp     %[Temp2],             $ac1,           31              \n\t"

        /*
          temp_1 = input[2] * cospi_24_64 - input[6] * cospi_8_64;
          step2_2 = dct_const_round_shift(temp_1);
        */
        "lh       %[Temp0],             4(%[input])                     \n\t"
        "lh       %[Temp1],             12(%[input])                    \n\t"
        "madd     $ac0,                 %[Temp0],       %[cospi_24_64]  \n\t"
        "msub     $ac0,                 %[Temp1],       %[cospi_8_64]   \n\t"
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"
        "extp     %[Temp3],             $ac0,           31              \n\t"

        /*
          step1_1 = step2_1 + step2_2;
          step1_2 = step2_1 - step2_2;
        */
        "add      %[step1_1],           %[Temp2],       %[Temp3]        \n\t"
        "sub      %[step1_2],           %[Temp2],       %[Temp3]        \n\t"

        /*
          temp_2 = input[2] * cospi_8_64 + input[6] * cospi_24_64;
          step2_3 = dct_const_round_shift(temp_2);
        */
        "madd     $ac1,                 %[Temp0],       %[cospi_8_64]   \n\t"
        "madd     $ac1,                 %[Temp1],       %[cospi_24_64]  \n\t"
        "extp     %[Temp1],             $ac1,           31              \n\t"

        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"

        /*
          step1_0 = step2_0 + step2_3;
          step1_3 = step2_0 - step2_3;
        */
        "add      %[step1_0],           %[Temp4],       %[Temp1]        \n\t"
        "sub      %[step1_3],           %[Temp4],       %[Temp1]        \n\t"

        /*
          temp_1 = input[1] * cospi_28_64 - input[7] * cospi_4_64;
          step1_4 = dct_const_round_shift(temp_1);
        */
        "lh       %[Temp0],             2(%[input])                     \n\t"
        "madd     $ac0,                 %[Temp0],       %[cospi_28_64]  \n\t"
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"
        "lh       %[Temp1],             14(%[input])                    \n\t"
        "lh       %[Temp0],             2(%[input])                     \n\t"
        "msub     $ac0,                 %[Temp1],       %[cospi_4_64]   \n\t"
        "extp     %[step1_4],           $ac0,           31              \n\t"

        /*
          temp_2 = input[1] * cospi_4_64 + input[7] * cospi_28_64;
          step1_7 = dct_const_round_shift(temp_2);
        */
        "madd     $ac1,                 %[Temp0],       %[cospi_4_64]   \n\t"
        "madd     $ac1,                 %[Temp1],       %[cospi_28_64]  \n\t"
        "extp     %[step1_7],           $ac1,           31              \n\t"

        /*
          temp_1 = input[5] * cospi_12_64 - input[3] * cospi_20_64;
          step1_5 = dct_const_round_shift(temp_1);
        */
        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"
        "lh       %[Temp0],             10(%[input])                    \n\t"
        "madd     $ac0,                 %[Temp0],       %[cospi_12_64]  \n\t"
        "lh       %[Temp1],             6(%[input])                     \n\t"
        "msub     $ac0,                 %[Temp1],       %[cospi_20_64]  \n\t"
        "extp     %[step1_5],           $ac0,           31              \n\t"

        /*
          temp_2 = input[5] * cospi_20_64 + input[3] * cospi_12_64;
          step1_6 = dct_const_round_shift(temp_2);
        */
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"
        "lh       %[Temp0],             10(%[input])                    \n\t"
        "madd     $ac1,                 %[Temp0],       %[cospi_20_64]  \n\t"
        "lh       %[Temp1],             6(%[input])                     \n\t"
        "madd     $ac1,                 %[Temp1],       %[cospi_12_64]  \n\t"
        "extp     %[step1_6],           $ac1,           31              \n\t"

        /*
          temp_1 = (step1_7 - step1_6 - step1_4 + step1_5) * cospi_16_64;
          temp_2 = (step1_4 - step1_5 - step1_6 + step1_7) * cospi_16_64;
        */
        "sub      %[Temp0],             %[step1_7],     %[step1_6]      \n\t"
        "sub      %[Temp0],             %[Temp0],       %[step1_4]      \n\t"
        "add      %[Temp0],             %[Temp0],       %[step1_5]      \n\t"
        "sub      %[Temp1],             %[step1_4],     %[step1_5]      \n\t"
        "sub      %[Temp1],             %[Temp1],       %[step1_6]      \n\t"
        "add      %[Temp1],             %[Temp1],       %[step1_7]      \n\t"

        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"

        "madd     $ac0,                 %[Temp0],       %[cospi_16_64]  \n\t"
        "madd     $ac1,                 %[Temp1],       %[cospi_16_64]  \n\t"

        /*
          step1_4 = step1_4 + step1_5;
          step1_7 = step1_6 + step1_7;
        */
        "add      %[step1_4],           %[step1_4],     %[step1_5]      \n\t"
        "add      %[step1_7],           %[step1_7],     %[step1_6]      \n\t"

        "extp     %[step1_5],           $ac0,           31              \n\t"
        "extp     %[step1_6],           $ac1,           31              \n\t"

        "add      %[Temp0],             %[step1_0],     %[step1_7]      \n\t"
        "sh       %[Temp0],             0(%[output])                    \n\t"
        "add      %[Temp1],             %[step1_1],     %[step1_6]      \n\t"
        "sh       %[Temp1],             16(%[output])                   \n\t"
        "add      %[Temp0],             %[step1_2],     %[step1_5]      \n\t"
        "sh       %[Temp0],             32(%[output])                   \n\t"
        "add      %[Temp1],             %[step1_3],     %[step1_4]      \n\t"
        "sh       %[Temp1],             48(%[output])                   \n\t"

        "sub      %[Temp0],             %[step1_3],     %[step1_4]      \n\t"
        "sh       %[Temp0],             64(%[output])                   \n\t"
        "sub      %[Temp1],             %[step1_2],     %[step1_5]      \n\t"
        "sh       %[Temp1],             80(%[output])                   \n\t"
        "sub      %[Temp0],             %[step1_1],     %[step1_6]      \n\t"
        "sh       %[Temp0],             96(%[output])                   \n\t"
        "sub      %[Temp1],             %[step1_0],     %[step1_7]      \n\t"
        "sh       %[Temp1],             112(%[output])                  \n\t"

        : [step1_0] "=&r"(step1_0), [step1_1] "=&r"(step1_1),
          [step1_2] "=&r"(step1_2), [step1_3] "=&r"(step1_3),
          [step1_4] "=&r"(step1_4), [step1_5] "=&r"(step1_5),
          [step1_6] "=&r"(step1_6), [step1_7] "=&r"(step1_7),
          [Temp0] "=&r"(Temp0), [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2),
          [Temp3] "=&r"(Temp3), [Temp4] "=&r"(Temp4)
        : [const_2_power_13] "r"(const_2_power_13),
          [cospi_16_64] "r"(cospi_16_64), [cospi_28_64] "r"(cospi_28_64),
          [cospi_4_64] "r"(cospi_4_64), [cospi_12_64] "r"(cospi_12_64),
          [cospi_20_64] "r"(cospi_20_64), [cospi_8_64] "r"(cospi_8_64),
          [cospi_24_64] "r"(cospi_24_64), [output] "r"(output),
          [input] "r"(input));

    input += 8;
    output += 1;
  }
}

void idct8_columns_add_blk_dspr2(int16_t *input, uint8_t *dest, int stride) {
  int step1_0, step1_1, step1_2, step1_3, step1_4, step1_5, step1_6, step1_7;
  int Temp0, Temp1, Temp2, Temp3;
  int i;
  const int const_2_power_13 = 8192;
  const int const_255 = 255;
  uint8_t *dest_pix;

  for (i = 0; i < 8; ++i) {
    dest_pix = (dest + i);

    __asm__ __volatile__(
        /*
          temp_1 = (input[0] + input[4]) * cospi_16_64;
          step2_0 = dct_const_round_shift(temp_1);

          temp_2 = (input[0] - input[4]) * cospi_16_64;
          step2_1 = dct_const_round_shift(temp_2);
        */
        "lh       %[Temp0],             0(%[input])                     \n\t"
        "lh       %[Temp1],             8(%[input])                     \n\t"
        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"
        "add      %[Temp2],             %[Temp0],       %[Temp1]        \n\t"
        "madd     $ac0,                 %[Temp2],       %[cospi_16_64]  \n\t"
        "extp     %[step1_6],           $ac0,           31              \n\t"

        "sub      %[Temp3],             %[Temp0],       %[Temp1]        \n\t"
        "madd     $ac1,                 %[Temp3],       %[cospi_16_64]  \n\t"
        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"
        "extp     %[Temp2],             $ac1,           31              \n\t"

        /*
          temp_1 = input[2] * cospi_24_64 - input[6] * cospi_8_64;
          step2_2 = dct_const_round_shift(temp_1);
        */
        "lh       %[Temp0],             4(%[input])                     \n\t"
        "lh       %[Temp1],             12(%[input])                    \n\t"
        "madd     $ac0,                 %[Temp0],       %[cospi_24_64]  \n\t"
        "msub     $ac0,                 %[Temp1],       %[cospi_8_64]   \n\t"
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"
        "extp     %[Temp3],             $ac0,           31              \n\t"

        /*
          step1_1 = step2_1 + step2_2;
          step1_2 = step2_1 - step2_2;
        */
        "add      %[step1_1],           %[Temp2],       %[Temp3]        \n\t"
        "sub      %[step1_2],           %[Temp2],       %[Temp3]        \n\t"

        /*
          temp_2 = input[2] * cospi_8_64 + input[6] * cospi_24_64;
          step2_3 = dct_const_round_shift(temp_2);
        */
        "madd     $ac1,                 %[Temp0],       %[cospi_8_64]   \n\t"
        "madd     $ac1,                 %[Temp1],       %[cospi_24_64]  \n\t"
        "extp     %[Temp1],             $ac1,           31              \n\t"

        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"

        /*
          step1_0 = step2_0 + step2_3;
          step1_3 = step2_0 - step2_3;
        */
        "add      %[step1_0],           %[step1_6],     %[Temp1]        \n\t"
        "sub      %[step1_3],           %[step1_6],     %[Temp1]        \n\t"

        /*
          temp_1 = input[1] * cospi_28_64 - input[7] * cospi_4_64;
          step1_4 = dct_const_round_shift(temp_1);
        */
        "lh       %[Temp0],             2(%[input])                     \n\t"
        "madd     $ac0,                 %[Temp0],       %[cospi_28_64]  \n\t"
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"
        "lh       %[Temp1],             14(%[input])                    \n\t"
        "lh       %[Temp0],             2(%[input])                     \n\t"
        "msub     $ac0,                 %[Temp1],       %[cospi_4_64]   \n\t"
        "extp     %[step1_4],           $ac0,           31              \n\t"

        /*
          temp_2 = input[1] * cospi_4_64 + input[7] * cospi_28_64;
          step1_7 = dct_const_round_shift(temp_2);
        */
        "madd     $ac1,                 %[Temp0],       %[cospi_4_64]   \n\t"
        "madd     $ac1,                 %[Temp1],       %[cospi_28_64]  \n\t"
        "extp     %[step1_7],           $ac1,           31              \n\t"

        /*
          temp_1 = input[5] * cospi_12_64 - input[3] * cospi_20_64;
          step1_5 = dct_const_round_shift(temp_1);
        */
        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"
        "lh       %[Temp0],             10(%[input])                    \n\t"
        "madd     $ac0,                 %[Temp0],       %[cospi_12_64]  \n\t"
        "lh       %[Temp1],             6(%[input])                     \n\t"
        "msub     $ac0,                 %[Temp1],       %[cospi_20_64]  \n\t"
        "extp     %[step1_5],           $ac0,           31              \n\t"

        /*
          temp_2 = input[5] * cospi_20_64 + input[3] * cospi_12_64;
          step1_6 = dct_const_round_shift(temp_2);
        */
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"
        "lh       %[Temp0],             10(%[input])                    \n\t"
        "madd     $ac1,                 %[Temp0],       %[cospi_20_64]  \n\t"
        "lh       %[Temp1],             6(%[input])                     \n\t"
        "madd     $ac1,                 %[Temp1],       %[cospi_12_64]  \n\t"
        "extp     %[step1_6],           $ac1,           31              \n\t"

        /*
          temp_1 = (step1_7 - step1_6 - step1_4 + step1_5) * cospi_16_64;
          temp_2 = (step1_4 - step1_5 - step1_6 + step1_7) * cospi_16_64;
        */
        "sub      %[Temp0],             %[step1_7],     %[step1_6]      \n\t"
        "sub      %[Temp0],             %[Temp0],       %[step1_4]      \n\t"
        "add      %[Temp0],             %[Temp0],       %[step1_5]      \n\t"
        "sub      %[Temp1],             %[step1_4],     %[step1_5]      \n\t"
        "sub      %[Temp1],             %[Temp1],       %[step1_6]      \n\t"
        "add      %[Temp1],             %[Temp1],       %[step1_7]      \n\t"

        "mtlo     %[const_2_power_13],  $ac0                            \n\t"
        "mthi     $zero,                $ac0                            \n\t"
        "mtlo     %[const_2_power_13],  $ac1                            \n\t"
        "mthi     $zero,                $ac1                            \n\t"

        "madd     $ac0,                 %[Temp0],       %[cospi_16_64]  \n\t"
        "madd     $ac1,                 %[Temp1],       %[cospi_16_64]  \n\t"

        /*
          step1_4 = step1_4 + step1_5;
          step1_7 = step1_6 + step1_7;
        */
        "add      %[step1_4],           %[step1_4],     %[step1_5]      \n\t"
        "add      %[step1_7],           %[step1_7],     %[step1_6]      \n\t"

        "extp     %[step1_5],           $ac0,           31              \n\t"
        "extp     %[step1_6],           $ac1,           31              \n\t"

        /* add block */
        "lbu      %[Temp1],             0(%[dest_pix])                  \n\t"
        "add      %[Temp0],             %[step1_0],     %[step1_7]      \n\t"
        "addi     %[Temp0],             %[Temp0],       16              \n\t"
        "sra      %[Temp0],             %[Temp0],       5               \n\t"
        "add      %[Temp1],             %[Temp1],       %[Temp0]        \n\t"
        "add      %[Temp0],             %[step1_1],     %[step1_6]      \n\t"
        "slt      %[Temp2],             %[Temp1],       %[const_255]    \n\t"
        "slt      %[Temp3],             $zero,          %[Temp1]        \n\t"
        "movz     %[Temp1],             %[const_255],   %[Temp2]        \n\t"
        "movz     %[Temp1],             $zero,          %[Temp3]        \n\t"
        "sb       %[Temp1],             0(%[dest_pix])                  \n\t"
        "addu     %[dest_pix],          %[dest_pix],    %[stride]       \n\t"

        "lbu      %[Temp1],             0(%[dest_pix])                  \n\t"
        "addi     %[Temp0],             %[Temp0],       16              \n\t"
        "sra      %[Temp0],             %[Temp0],       5               \n\t"
        "add      %[Temp1],             %[Temp1],       %[Temp0]        \n\t"
        "add      %[Temp0],             %[step1_2],     %[step1_5]      \n\t"
        "slt      %[Temp2],             %[Temp1],       %[const_255]    \n\t"
        "slt      %[Temp3],             $zero,          %[Temp1]        \n\t"
        "movz     %[Temp1],             %[const_255],   %[Temp2]        \n\t"
        "movz     %[Temp1],             $zero,          %[Temp3]        \n\t"
        "sb       %[Temp1],             0(%[dest_pix])                  \n\t"
        "addu     %[dest_pix],          %[dest_pix],    %[stride]       \n\t"

        "lbu      %[Temp1],             0(%[dest_pix])                  \n\t"
        "addi     %[Temp0],             %[Temp0],       16              \n\t"
        "sra      %[Temp0],             %[Temp0],       5               \n\t"
        "add      %[Temp1],             %[Temp1],       %[Temp0]        \n\t"
        "add      %[Temp0],             %[step1_3],     %[step1_4]      \n\t"
        "slt      %[Temp2],             %[Temp1],       %[const_255]    \n\t"
        "slt      %[Temp3],             $zero,          %[Temp1]        \n\t"
        "movz     %[Temp1],             %[const_255],   %[Temp2]        \n\t"
        "movz     %[Temp1],             $zero,          %[Temp3]        \n\t"
        "sb       %[Temp1],             0(%[dest_pix])                  \n\t"
        "addu     %[dest_pix],          %[dest_pix],    %[stride]       \n\t"

        "lbu      %[Temp1],             0(%[dest_pix])                  \n\t"
        "addi     %[Temp0],             %[Temp0],       16              \n\t"
        "sra      %[Temp0],             %[Temp0],       5               \n\t"
        "add      %[Temp1],             %[Temp1],       %[Temp0]        \n\t"
        "sub      %[Temp0],             %[step1_3],     %[step1_4]      \n\t"
        "slt      %[Temp2],             %[Temp1],       %[const_255]    \n\t"
        "slt      %[Temp3],             $zero,          %[Temp1]        \n\t"
        "movz     %[Temp1],             %[const_255],   %[Temp2]        \n\t"
        "movz     %[Temp1],             $zero,          %[Temp3]        \n\t"
        "sb       %[Temp1],             0(%[dest_pix])                  \n\t"
        "addu     %[dest_pix],          %[dest_pix],    %[stride]       \n\t"

        "lbu      %[Temp1],             0(%[dest_pix])                  \n\t"
        "addi     %[Temp0],             %[Temp0],       16              \n\t"
        "sra      %[Temp0],             %[Temp0],       5               \n\t"
        "add      %[Temp1],             %[Temp1],       %[Temp0]        \n\t"
        "sub      %[Temp0],             %[step1_2],     %[step1_5]      \n\t"
        "slt      %[Temp2],             %[Temp1],       %[const_255]    \n\t"
        "slt      %[Temp3],             $zero,          %[Temp1]        \n\t"
        "movz     %[Temp1],             %[const_255],   %[Temp2]        \n\t"
        "movz     %[Temp1],             $zero,          %[Temp3]        \n\t"
        "sb       %[Temp1],             0(%[dest_pix])                  \n\t"
        "addu     %[dest_pix],          %[dest_pix],    %[stride]       \n\t"

        "lbu      %[Temp1],             0(%[dest_pix])                  \n\t"
        "addi     %[Temp0],             %[Temp0],       16              \n\t"
        "sra      %[Temp0],             %[Temp0],       5               \n\t"
        "add      %[Temp1],             %[Temp1],       %[Temp0]        \n\t"
        "sub      %[Temp0],             %[step1_1],     %[step1_6]      \n\t"
        "slt      %[Temp2],             %[Temp1],       %[const_255]    \n\t"
        "slt      %[Temp3],             $zero,          %[Temp1]        \n\t"
        "movz     %[Temp1],             %[const_255],   %[Temp2]        \n\t"
        "movz     %[Temp1],             $zero,          %[Temp3]        \n\t"
        "sb       %[Temp1],             0(%[dest_pix])                  \n\t"
        "addu     %[dest_pix],          %[dest_pix],    %[stride]       \n\t"

        "lbu      %[Temp1],             0(%[dest_pix])                  \n\t"
        "addi     %[Temp0],             %[Temp0],       16              \n\t"
        "sra      %[Temp0],             %[Temp0],       5               \n\t"
        "add      %[Temp1],             %[Temp1],       %[Temp0]        \n\t"
        "sub      %[Temp0],             %[step1_0],     %[step1_7]      \n\t"
        "slt      %[Temp2],             %[Temp1],       %[const_255]    \n\t"
        "slt      %[Temp3],             $zero,          %[Temp1]        \n\t"
        "movz     %[Temp1],             %[const_255],   %[Temp2]        \n\t"
        "movz     %[Temp1],             $zero,          %[Temp3]        \n\t"
        "sb       %[Temp1],             0(%[dest_pix])                  \n\t"
        "addu     %[dest_pix],          %[dest_pix],    %[stride]       \n\t"

        "lbu      %[Temp1],             0(%[dest_pix])                  \n\t"
        "addi     %[Temp0],             %[Temp0],       16              \n\t"
        "sra      %[Temp0],             %[Temp0],       5               \n\t"
        "add      %[Temp1],             %[Temp1],       %[Temp0]        \n\t"
        "slt      %[Temp2],             %[Temp1],       %[const_255]    \n\t"
        "slt      %[Temp3],             $zero,          %[Temp1]        \n\t"
        "movz     %[Temp1],             %[const_255],   %[Temp2]        \n\t"
        "movz     %[Temp1],             $zero,          %[Temp3]        \n\t"
        "sb       %[Temp1],             0(%[dest_pix])                  \n\t"

        : [step1_0] "=&r"(step1_0), [step1_1] "=&r"(step1_1),
          [step1_2] "=&r"(step1_2), [step1_3] "=&r"(step1_3),
          [step1_4] "=&r"(step1_4), [step1_5] "=&r"(step1_5),
          [step1_6] "=&r"(step1_6), [step1_7] "=&r"(step1_7),
          [Temp0] "=&r"(Temp0), [Temp1] "=&r"(Temp1), [Temp2] "=&r"(Temp2),
          [Temp3] "=&r"(Temp3), [dest_pix] "+r"(dest_pix)
        : [const_2_power_13] "r"(const_2_power_13), [const_255] "r"(const_255),
          [cospi_16_64] "r"(cospi_16_64), [cospi_28_64] "r"(cospi_28_64),
          [cospi_4_64] "r"(cospi_4_64), [cospi_12_64] "r"(cospi_12_64),
          [cospi_20_64] "r"(cospi_20_64), [cospi_8_64] "r"(cospi_8_64),
          [cospi_24_64] "r"(cospi_24_64), [input] "r"(input),
          [stride] "r"(stride));

    input += 8;
  }
}

void vpx_idct8x8_64_add_dspr2(const int16_t *input, uint8_t *dest, int stride) {
  DECLARE_ALIGNED(32, int16_t, out[8 * 8]);
  int16_t *outptr = out;
  uint32_t pos = 45;

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp    %[pos],    1    \n\t" : : [pos] "r"(pos));

  // First transform rows
  idct8_rows_dspr2(input, outptr, 8);

  // Then transform columns and add to dest
  idct8_columns_add_blk_dspr2(&out[0], dest, stride);
}

void vpx_idct8x8_12_add_dspr2(const int16_t *input, uint8_t *dest, int stride) {
  DECLARE_ALIGNED(32, int16_t, out[8 * 8]);
  int16_t *outptr = out;
  uint32_t pos = 45;

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp    %[pos],    1    \n\t" : : [pos] "r"(pos));

  // First transform rows
  idct8_rows_dspr2(input, outptr, 4);

  outptr += 4;

  __asm__ __volatile__(
      "sw  $zero,   0(%[outptr])  \n\t"
      "sw  $zero,   4(%[outptr])  \n\t"
      "sw  $zero,  16(%[outptr])  \n\t"
      "sw  $zero,  20(%[outptr])  \n\t"
      "sw  $zero,  32(%[outptr])  \n\t"
      "sw  $zero,  36(%[outptr])  \n\t"
      "sw  $zero,  48(%[outptr])  \n\t"
      "sw  $zero,  52(%[outptr])  \n\t"
      "sw  $zero,  64(%[outptr])  \n\t"
      "sw  $zero,  68(%[outptr])  \n\t"
      "sw  $zero,  80(%[outptr])  \n\t"
      "sw  $zero,  84(%[outptr])  \n\t"
      "sw  $zero,  96(%[outptr])  \n\t"
      "sw  $zero, 100(%[outptr])  \n\t"
      "sw  $zero, 112(%[outptr])  \n\t"
      "sw  $zero, 116(%[outptr])  \n\t"

      :
      : [outptr] "r"(outptr));

  // Then transform columns and add to dest
  idct8_columns_add_blk_dspr2(&out[0], dest, stride);
}

void vpx_idct8x8_1_add_dspr2(const int16_t *input, uint8_t *dest, int stride) {
  uint32_t pos = 45;
  int32_t out;
  int32_t r;
  int32_t a1, absa1;
  int32_t t1, t2, vector_a1, vector_1, vector_2;

  /* bit positon for extract from acc */
  __asm__ __volatile__("wrdsp      %[pos],     1           \n\t"

                       :
                       : [pos] "r"(pos));

  out = DCT_CONST_ROUND_SHIFT_TWICE_COSPI_16_64(input[0]);
  __asm__ __volatile__(
      "addi     %[out],     %[out],     16      \n\t"
      "sra      %[a1],      %[out],     5       \n\t"

      : [out] "+r"(out), [a1] "=r"(a1)
      :);

  if (a1 < 0) {
    /* use quad-byte
     * input and output memory are four byte aligned */
    __asm__ __volatile__(
        "abs        %[absa1],       %[a1]       \n\t"
        "replv.qb   %[vector_a1],   %[absa1]    \n\t"

        : [absa1] "=r"(absa1), [vector_a1] "=r"(vector_a1)
        : [a1] "r"(a1));

    for (r = 8; r--;) {
      __asm__ __volatile__(
          "lw           %[t1],          0(%[dest])                      \n\t"
          "lw           %[t2],          4(%[dest])                      \n\t"
          "subu_s.qb    %[vector_1],    %[t1],          %[vector_a1]    \n\t"
          "subu_s.qb    %[vector_2],    %[t2],          %[vector_a1]    \n\t"
          "sw           %[vector_1],    0(%[dest])                      \n\t"
          "sw           %[vector_2],    4(%[dest])                      \n\t"
          "add          %[dest],        %[dest],        %[stride]       \n\t"

          : [t1] "=&r"(t1), [t2] "=&r"(t2), [vector_1] "=&r"(vector_1),
            [vector_2] "=&r"(vector_2), [dest] "+&r"(dest)
          : [stride] "r"(stride), [vector_a1] "r"(vector_a1));
    }
  } else if (a1 > 255) {
    int32_t a11, a12, vector_a11, vector_a12;

    /* use quad-byte
     * input and output memory are four byte aligned */
    a11 = a1 >> 2;
    a12 = a1 - (a11 * 3);

    __asm__ __volatile__(
        "replv.qb      %[vector_a11],  %[a11]     \n\t"
        "replv.qb      %[vector_a12],  %[a12]     \n\t"

        : [vector_a11] "=&r"(vector_a11), [vector_a12] "=&r"(vector_a12)
        : [a11] "r"(a11), [a12] "r"(a12));

    for (r = 8; r--;) {
      __asm__ __volatile__(
          "lw             %[t1],          0(%[dest])                      \n\t"
          "lw             %[t2],          4(%[dest])                      \n\t"
          "addu_s.qb      %[vector_1],    %[t1],          %[vector_a11]   \n\t"
          "addu_s.qb      %[vector_2],    %[t2],          %[vector_a11]   \n\t"
          "addu_s.qb      %[vector_1],    %[vector_1],    %[vector_a11]   \n\t"
          "addu_s.qb      %[vector_2],    %[vector_2],    %[vector_a11]   \n\t"
          "addu_s.qb      %[vector_1],    %[vector_1],    %[vector_a11]   \n\t"
          "addu_s.qb      %[vector_2],    %[vector_2],    %[vector_a11]   \n\t"
          "addu_s.qb      %[vector_1],    %[vector_1],    %[vector_a12]   \n\t"
          "addu_s.qb      %[vector_2],    %[vector_2],    %[vector_a12]   \n\t"
          "sw             %[vector_1],    0(%[dest])                      \n\t"
          "sw             %[vector_2],    4(%[dest])                      \n\t"
          "add            %[dest],        %[dest],        %[stride]       \n\t"

          : [t1] "=&r"(t1), [t2] "=&r"(t2), [vector_1] "=&r"(vector_1),
            [vector_2] "=&r"(vector_2), [dest] "+r"(dest)
          : [stride] "r"(stride), [vector_a11] "r"(vector_a11),
            [vector_a12] "r"(vector_a12));
    }
  } else {
    /* use quad-byte
     * input and output memory are four byte aligned */
    __asm__ __volatile__("replv.qb   %[vector_a1],   %[a1]   \n\t"

                         : [vector_a1] "=r"(vector_a1)
                         : [a1] "r"(a1));

    for (r = 8; r--;) {
      __asm__ __volatile__(
          "lw           %[t1],          0(%[dest])                      \n\t"
          "lw           %[t2],          4(%[dest])                      \n\t"
          "addu_s.qb    %[vector_1],    %[t1],          %[vector_a1]    \n\t"
          "addu_s.qb    %[vector_2],    %[t2],          %[vector_a1]    \n\t"
          "sw           %[vector_1],    0(%[dest])                      \n\t"
          "sw           %[vector_2],    4(%[dest])                      \n\t"
          "add          %[dest],        %[dest],        %[stride]       \n\t"

          : [t1] "=&r"(t1), [t2] "=&r"(t2), [vector_1] "=&r"(vector_1),
            [vector_2] "=&r"(vector_2), [dest] "+r"(dest)
          : [stride] "r"(stride), [vector_a1] "r"(vector_a1));
    }
  }
}

void iadst8_dspr2(const int16_t *input, int16_t *output) {
  int s0, s1, s2, s3, s4, s5, s6, s7;
  int x0, x1, x2, x3, x4, x5, x6, x7;

  x0 = input[7];
  x1 = input[0];
  x2 = input[5];
  x3 = input[2];
  x4 = input[3];
  x5 = input[4];
  x6 = input[1];
  x7 = input[6];

  if (!(x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7)) {
    output[0] = output[1] = output[2] = output[3] = output[4] = output[5] =
        output[6] = output[7] = 0;
    return;
  }

  // stage 1
  s0 = cospi_2_64 * x0 + cospi_30_64 * x1;
  s1 = cospi_30_64 * x0 - cospi_2_64 * x1;
  s2 = cospi_10_64 * x2 + cospi_22_64 * x3;
  s3 = cospi_22_64 * x2 - cospi_10_64 * x3;
  s4 = cospi_18_64 * x4 + cospi_14_64 * x5;
  s5 = cospi_14_64 * x4 - cospi_18_64 * x5;
  s6 = cospi_26_64 * x6 + cospi_6_64 * x7;
  s7 = cospi_6_64 * x6 - cospi_26_64 * x7;

  x0 = ROUND_POWER_OF_TWO((s0 + s4), DCT_CONST_BITS);
  x1 = ROUND_POWER_OF_TWO((s1 + s5), DCT_CONST_BITS);
  x2 = ROUND_POWER_OF_TWO((s2 + s6), DCT_CONST_BITS);
  x3 = ROUND_POWER_OF_TWO((s3 + s7), DCT_CONST_BITS);
  x4 = ROUND_POWER_OF_TWO((s0 - s4), DCT_CONST_BITS);
  x5 = ROUND_POWER_OF_TWO((s1 - s5), DCT_CONST_BITS);
  x6 = ROUND_POWER_OF_TWO((s2 - s6), DCT_CONST_BITS);
  x7 = ROUND_POWER_OF_TWO((s3 - s7), DCT_CONST_BITS);

  // stage 2
  s0 = x0;
  s1 = x1;
  s2 = x2;
  s3 = x3;
  s4 = cospi_8_64 * x4 + cospi_24_64 * x5;
  s5 = cospi_24_64 * x4 - cospi_8_64 * x5;
  s6 = -cospi_24_64 * x6 + cospi_8_64 * x7;
  s7 = cospi_8_64 * x6 + cospi_24_64 * x7;

  x0 = s0 + s2;
  x1 = s1 + s3;
  x2 = s0 - s2;
  x3 = s1 - s3;
  x4 = ROUND_POWER_OF_TWO((s4 + s6), DCT_CONST_BITS);
  x5 = ROUND_POWER_OF_TWO((s5 + s7), DCT_CONST_BITS);
  x6 = ROUND_POWER_OF_TWO((s4 - s6), DCT_CONST_BITS);
  x7 = ROUND_POWER_OF_TWO((s5 - s7), DCT_CONST_BITS);

  // stage 3
  s2 = cospi_16_64 * (x2 + x3);
  s3 = cospi_16_64 * (x2 - x3);
  s6 = cospi_16_64 * (x6 + x7);
  s7 = cospi_16_64 * (x6 - x7);

  x2 = ROUND_POWER_OF_TWO((s2), DCT_CONST_BITS);
  x3 = ROUND_POWER_OF_TWO((s3), DCT_CONST_BITS);
  x6 = ROUND_POWER_OF_TWO((s6), DCT_CONST_BITS);
  x7 = ROUND_POWER_OF_TWO((s7), DCT_CONST_BITS);

  output[0] = x0;
  output[1] = -x4;
  output[2] = x6;
  output[3] = -x2;
  output[4] = x3;
  output[5] = -x7;
  output[6] = x5;
  output[7] = -x1;
}
#endif  // HAVE_DSPR2
