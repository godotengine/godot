/*
 * jcphuff-neon.c - prepare data for progressive Huffman encoding (Arm Neon)
 *
 * Copyright (C) 2020-2021, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2022, Matthieu Darbois.  All Rights Reserved.
 * Copyright (C) 2022, 2024, D. R. Commander.  All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#define JPEG_INTERNALS
#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jsimd.h"
#include "../../src/jdct.h"
#include "../../src/jsimddct.h"
#include "../jsimd.h"
#include "neon-compat.h"

#include <arm_neon.h>


/* Data preparation for encode_mcu_AC_first().
 *
 * The equivalent scalar C function (encode_mcu_AC_first_prepare()) can be
 * found in jcphuff.c.
 */

void jsimd_encode_mcu_AC_first_prepare_neon
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *values, size_t *zerobits)
{
  UJCOEF *values_ptr = values;
  UJCOEF *diff_values_ptr = values + DCTSIZE2;

  /* Rows of coefficients to zero (since they haven't been processed) */
  int i, rows_to_zero = 8;

  for (i = 0; i < Sl / 16; i++) {
    int16x8_t coefs1 = vld1q_dup_s16(block + jpeg_natural_order_start[0]);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[1], coefs1, 1);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[2], coefs1, 2);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[3], coefs1, 3);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[4], coefs1, 4);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[5], coefs1, 5);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[6], coefs1, 6);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[7], coefs1, 7);
    int16x8_t coefs2 = vld1q_dup_s16(block + jpeg_natural_order_start[8]);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[9], coefs2, 1);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[10], coefs2, 2);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[11], coefs2, 3);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[12], coefs2, 4);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[13], coefs2, 5);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[14], coefs2, 6);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[15], coefs2, 7);

    /* Isolate sign of coefficients. */
    uint16x8_t sign_coefs1 = vreinterpretq_u16_s16(vshrq_n_s16(coefs1, 15));
    uint16x8_t sign_coefs2 = vreinterpretq_u16_s16(vshrq_n_s16(coefs2, 15));
    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs1 = vreinterpretq_u16_s16(vabsq_s16(coefs1));
    uint16x8_t abs_coefs2 = vreinterpretq_u16_s16(vabsq_s16(coefs2));
    abs_coefs1 = vshlq_u16(abs_coefs1, vdupq_n_s16(-Al));
    abs_coefs2 = vshlq_u16(abs_coefs2, vdupq_n_s16(-Al));

    /* Compute diff values. */
    uint16x8_t diff1 = veorq_u16(abs_coefs1, sign_coefs1);
    uint16x8_t diff2 = veorq_u16(abs_coefs2, sign_coefs2);

    /* Store transformed coefficients and diff values. */
    vst1q_u16(values_ptr, abs_coefs1);
    vst1q_u16(values_ptr + DCTSIZE, abs_coefs2);
    vst1q_u16(diff_values_ptr, diff1);
    vst1q_u16(diff_values_ptr + DCTSIZE, diff2);
    values_ptr += 16;
    diff_values_ptr += 16;
    jpeg_natural_order_start += 16;
    rows_to_zero -= 2;
  }

  /* Same operation but for remaining partial vector */
  int remaining_coefs = Sl % 16;
  if (remaining_coefs > 8) {
    int16x8_t coefs1 = vld1q_dup_s16(block + jpeg_natural_order_start[0]);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[1], coefs1, 1);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[2], coefs1, 2);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[3], coefs1, 3);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[4], coefs1, 4);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[5], coefs1, 5);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[6], coefs1, 6);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[7], coefs1, 7);
    int16x8_t coefs2 = vdupq_n_s16(0);
    switch (remaining_coefs) {
    case 15:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[14], coefs2, 6);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 14:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[13], coefs2, 5);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 13:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[12], coefs2, 4);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 12:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[11], coefs2, 3);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 11:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[10], coefs2, 2);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 10:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[9], coefs2, 1);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 9:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[8], coefs2, 0);
      FALLTHROUGH               /*FALLTHROUGH*/
    default:
      break;
    }

    /* Isolate sign of coefficients. */
    uint16x8_t sign_coefs1 = vreinterpretq_u16_s16(vshrq_n_s16(coefs1, 15));
    uint16x8_t sign_coefs2 = vreinterpretq_u16_s16(vshrq_n_s16(coefs2, 15));
    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs1 = vreinterpretq_u16_s16(vabsq_s16(coefs1));
    uint16x8_t abs_coefs2 = vreinterpretq_u16_s16(vabsq_s16(coefs2));
    abs_coefs1 = vshlq_u16(abs_coefs1, vdupq_n_s16(-Al));
    abs_coefs2 = vshlq_u16(abs_coefs2, vdupq_n_s16(-Al));

    /* Compute diff values. */
    uint16x8_t diff1 = veorq_u16(abs_coefs1, sign_coefs1);
    uint16x8_t diff2 = veorq_u16(abs_coefs2, sign_coefs2);

    /* Store transformed coefficients and diff values. */
    vst1q_u16(values_ptr, abs_coefs1);
    vst1q_u16(values_ptr + DCTSIZE, abs_coefs2);
    vst1q_u16(diff_values_ptr, diff1);
    vst1q_u16(diff_values_ptr + DCTSIZE, diff2);
    values_ptr += 16;
    diff_values_ptr += 16;
    rows_to_zero -= 2;

  } else if (remaining_coefs > 0) {
    int16x8_t coefs = vdupq_n_s16(0);

    switch (remaining_coefs) {
    case 8:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[7], coefs, 7);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 7:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[6], coefs, 6);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 6:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[5], coefs, 5);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 5:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[4], coefs, 4);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 4:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[3], coefs, 3);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 3:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[2], coefs, 2);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 2:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[1], coefs, 1);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 1:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[0], coefs, 0);
      FALLTHROUGH               /*FALLTHROUGH*/
    default:
      break;
    }

    /* Isolate sign of coefficients. */
    uint16x8_t sign_coefs = vreinterpretq_u16_s16(vshrq_n_s16(coefs, 15));
    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs = vreinterpretq_u16_s16(vabsq_s16(coefs));
    abs_coefs = vshlq_u16(abs_coefs, vdupq_n_s16(-Al));

    /* Compute diff values. */
    uint16x8_t diff = veorq_u16(abs_coefs, sign_coefs);

    /* Store transformed coefficients and diff values. */
    vst1q_u16(values_ptr, abs_coefs);
    vst1q_u16(diff_values_ptr, diff);
    values_ptr += 8;
    diff_values_ptr += 8;
    rows_to_zero--;
  }

  /* Zero remaining memory in the values and diff_values blocks. */
  for (i = 0; i < rows_to_zero; i++) {
    vst1q_u16(values_ptr, vdupq_n_u16(0));
    vst1q_u16(diff_values_ptr, vdupq_n_u16(0));
    values_ptr += 8;
    diff_values_ptr += 8;
  }

  /* Construct zerobits bitmap.  A set bit means that the corresponding
   * coefficient != 0.
   */
  uint16x8_t row0 = vld1q_u16(values + 0 * DCTSIZE);
  uint16x8_t row1 = vld1q_u16(values + 1 * DCTSIZE);
  uint16x8_t row2 = vld1q_u16(values + 2 * DCTSIZE);
  uint16x8_t row3 = vld1q_u16(values + 3 * DCTSIZE);
  uint16x8_t row4 = vld1q_u16(values + 4 * DCTSIZE);
  uint16x8_t row5 = vld1q_u16(values + 5 * DCTSIZE);
  uint16x8_t row6 = vld1q_u16(values + 6 * DCTSIZE);
  uint16x8_t row7 = vld1q_u16(values + 7 * DCTSIZE);

  uint8x8_t row0_eq0 = vmovn_u16(vceqq_u16(row0, vdupq_n_u16(0)));
  uint8x8_t row1_eq0 = vmovn_u16(vceqq_u16(row1, vdupq_n_u16(0)));
  uint8x8_t row2_eq0 = vmovn_u16(vceqq_u16(row2, vdupq_n_u16(0)));
  uint8x8_t row3_eq0 = vmovn_u16(vceqq_u16(row3, vdupq_n_u16(0)));
  uint8x8_t row4_eq0 = vmovn_u16(vceqq_u16(row4, vdupq_n_u16(0)));
  uint8x8_t row5_eq0 = vmovn_u16(vceqq_u16(row5, vdupq_n_u16(0)));
  uint8x8_t row6_eq0 = vmovn_u16(vceqq_u16(row6, vdupq_n_u16(0)));
  uint8x8_t row7_eq0 = vmovn_u16(vceqq_u16(row7, vdupq_n_u16(0)));

  /* { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 } */
  const uint8x8_t bitmap_mask =
    vreinterpret_u8_u64(vmov_n_u64(0x8040201008040201));

  row0_eq0 = vand_u8(row0_eq0, bitmap_mask);
  row1_eq0 = vand_u8(row1_eq0, bitmap_mask);
  row2_eq0 = vand_u8(row2_eq0, bitmap_mask);
  row3_eq0 = vand_u8(row3_eq0, bitmap_mask);
  row4_eq0 = vand_u8(row4_eq0, bitmap_mask);
  row5_eq0 = vand_u8(row5_eq0, bitmap_mask);
  row6_eq0 = vand_u8(row6_eq0, bitmap_mask);
  row7_eq0 = vand_u8(row7_eq0, bitmap_mask);

  uint8x8_t bitmap_rows_01 = vpadd_u8(row0_eq0, row1_eq0);
  uint8x8_t bitmap_rows_23 = vpadd_u8(row2_eq0, row3_eq0);
  uint8x8_t bitmap_rows_45 = vpadd_u8(row4_eq0, row5_eq0);
  uint8x8_t bitmap_rows_67 = vpadd_u8(row6_eq0, row7_eq0);
  uint8x8_t bitmap_rows_0123 = vpadd_u8(bitmap_rows_01, bitmap_rows_23);
  uint8x8_t bitmap_rows_4567 = vpadd_u8(bitmap_rows_45, bitmap_rows_67);
  uint8x8_t bitmap_all = vpadd_u8(bitmap_rows_0123, bitmap_rows_4567);

#if defined(__aarch64__) || defined(_M_ARM64)
  /* Move bitmap to a 64-bit scalar register. */
  uint64_t bitmap = vget_lane_u64(vreinterpret_u64_u8(bitmap_all), 0);
  /* Store zerobits bitmap. */
  *zerobits = ~bitmap;
#else
  /* Move bitmap to two 32-bit scalar registers. */
  uint32_t bitmap0 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 0);
  uint32_t bitmap1 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 1);
  /* Store zerobits bitmap. */
  zerobits[0] = ~bitmap0;
  zerobits[1] = ~bitmap1;
#endif
}


/* Data preparation for encode_mcu_AC_refine().
 *
 * The equivalent scalar C function (encode_mcu_AC_refine_prepare()) can be
 * found in jcphuff.c.
 */

int jsimd_encode_mcu_AC_refine_prepare_neon
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *absvalues, size_t *bits)
{
  /* Temporary storage buffers for data used to compute the signbits bitmap and
   * the end-of-block (EOB) position
   */
  uint8_t coef_sign_bits[64];
  uint8_t coef_eq1_bits[64];

  UJCOEF *absvalues_ptr = absvalues;
  uint8_t *coef_sign_bits_ptr = coef_sign_bits;
  uint8_t *eq1_bits_ptr = coef_eq1_bits;

  /* Rows of coefficients to zero (since they haven't been processed) */
  int i, rows_to_zero = 8;

  for (i = 0; i < Sl / 16; i++) {
    int16x8_t coefs1 = vld1q_dup_s16(block + jpeg_natural_order_start[0]);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[1], coefs1, 1);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[2], coefs1, 2);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[3], coefs1, 3);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[4], coefs1, 4);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[5], coefs1, 5);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[6], coefs1, 6);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[7], coefs1, 7);
    int16x8_t coefs2 = vld1q_dup_s16(block + jpeg_natural_order_start[8]);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[9], coefs2, 1);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[10], coefs2, 2);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[11], coefs2, 3);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[12], coefs2, 4);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[13], coefs2, 5);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[14], coefs2, 6);
    coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[15], coefs2, 7);

    /* Compute and store data for signbits bitmap. */
    uint8x8_t sign_coefs1 =
      vmovn_u16(vreinterpretq_u16_s16(vshrq_n_s16(coefs1, 15)));
    uint8x8_t sign_coefs2 =
      vmovn_u16(vreinterpretq_u16_s16(vshrq_n_s16(coefs2, 15)));
    vst1_u8(coef_sign_bits_ptr, sign_coefs1);
    vst1_u8(coef_sign_bits_ptr + DCTSIZE, sign_coefs2);

    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs1 = vreinterpretq_u16_s16(vabsq_s16(coefs1));
    uint16x8_t abs_coefs2 = vreinterpretq_u16_s16(vabsq_s16(coefs2));
    abs_coefs1 = vshlq_u16(abs_coefs1, vdupq_n_s16(-Al));
    abs_coefs2 = vshlq_u16(abs_coefs2, vdupq_n_s16(-Al));
    vst1q_u16(absvalues_ptr, abs_coefs1);
    vst1q_u16(absvalues_ptr + DCTSIZE, abs_coefs2);

    /* Test whether transformed coefficient values == 1 (used to find EOB
     * position.)
     */
    uint8x8_t coefs_eq11 = vmovn_u16(vceqq_u16(abs_coefs1, vdupq_n_u16(1)));
    uint8x8_t coefs_eq12 = vmovn_u16(vceqq_u16(abs_coefs2, vdupq_n_u16(1)));
    vst1_u8(eq1_bits_ptr, coefs_eq11);
    vst1_u8(eq1_bits_ptr + DCTSIZE, coefs_eq12);

    absvalues_ptr += 16;
    coef_sign_bits_ptr += 16;
    eq1_bits_ptr += 16;
    jpeg_natural_order_start += 16;
    rows_to_zero -= 2;
  }

  /* Same operation but for remaining partial vector */
  int remaining_coefs = Sl % 16;
  if (remaining_coefs > 8) {
    int16x8_t coefs1 = vld1q_dup_s16(block + jpeg_natural_order_start[0]);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[1], coefs1, 1);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[2], coefs1, 2);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[3], coefs1, 3);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[4], coefs1, 4);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[5], coefs1, 5);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[6], coefs1, 6);
    coefs1 = vld1q_lane_s16(block + jpeg_natural_order_start[7], coefs1, 7);
    int16x8_t coefs2 = vdupq_n_s16(0);
    switch (remaining_coefs) {
    case 15:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[14], coefs2, 6);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 14:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[13], coefs2, 5);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 13:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[12], coefs2, 4);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 12:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[11], coefs2, 3);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 11:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[10], coefs2, 2);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 10:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[9], coefs2, 1);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 9:
      coefs2 = vld1q_lane_s16(block + jpeg_natural_order_start[8], coefs2, 0);
      FALLTHROUGH               /*FALLTHROUGH*/
    default:
      break;
    }

    /* Compute and store data for signbits bitmap. */
    uint8x8_t sign_coefs1 =
      vmovn_u16(vreinterpretq_u16_s16(vshrq_n_s16(coefs1, 15)));
    uint8x8_t sign_coefs2 =
      vmovn_u16(vreinterpretq_u16_s16(vshrq_n_s16(coefs2, 15)));
    vst1_u8(coef_sign_bits_ptr, sign_coefs1);
    vst1_u8(coef_sign_bits_ptr + DCTSIZE, sign_coefs2);

    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs1 = vreinterpretq_u16_s16(vabsq_s16(coefs1));
    uint16x8_t abs_coefs2 = vreinterpretq_u16_s16(vabsq_s16(coefs2));
    abs_coefs1 = vshlq_u16(abs_coefs1, vdupq_n_s16(-Al));
    abs_coefs2 = vshlq_u16(abs_coefs2, vdupq_n_s16(-Al));
    vst1q_u16(absvalues_ptr, abs_coefs1);
    vst1q_u16(absvalues_ptr + DCTSIZE, abs_coefs2);

    /* Test whether transformed coefficient values == 1 (used to find EOB
     * position.)
     */
    uint8x8_t coefs_eq11 = vmovn_u16(vceqq_u16(abs_coefs1, vdupq_n_u16(1)));
    uint8x8_t coefs_eq12 = vmovn_u16(vceqq_u16(abs_coefs2, vdupq_n_u16(1)));
    vst1_u8(eq1_bits_ptr, coefs_eq11);
    vst1_u8(eq1_bits_ptr + DCTSIZE, coefs_eq12);

    absvalues_ptr += 16;
    coef_sign_bits_ptr += 16;
    eq1_bits_ptr += 16;
    jpeg_natural_order_start += 16;
    rows_to_zero -= 2;

  } else if (remaining_coefs > 0) {
    int16x8_t coefs = vdupq_n_s16(0);

    switch (remaining_coefs) {
    case 8:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[7], coefs, 7);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 7:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[6], coefs, 6);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 6:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[5], coefs, 5);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 5:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[4], coefs, 4);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 4:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[3], coefs, 3);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 3:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[2], coefs, 2);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 2:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[1], coefs, 1);
      FALLTHROUGH               /*FALLTHROUGH*/
    case 1:
      coefs = vld1q_lane_s16(block + jpeg_natural_order_start[0], coefs, 0);
      FALLTHROUGH               /*FALLTHROUGH*/
    default:
      break;
    }

    /* Compute and store data for signbits bitmap. */
    uint8x8_t sign_coefs =
      vmovn_u16(vreinterpretq_u16_s16(vshrq_n_s16(coefs, 15)));
    vst1_u8(coef_sign_bits_ptr, sign_coefs);

    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs = vreinterpretq_u16_s16(vabsq_s16(coefs));
    abs_coefs = vshlq_u16(abs_coefs, vdupq_n_s16(-Al));
    vst1q_u16(absvalues_ptr, abs_coefs);

    /* Test whether transformed coefficient values == 1 (used to find EOB
     * position.)
     */
    uint8x8_t coefs_eq1 = vmovn_u16(vceqq_u16(abs_coefs, vdupq_n_u16(1)));
    vst1_u8(eq1_bits_ptr, coefs_eq1);

    absvalues_ptr += 8;
    coef_sign_bits_ptr += 8;
    eq1_bits_ptr += 8;
    rows_to_zero--;
  }

  /* Zero remaining memory in blocks. */
  for (i = 0; i < rows_to_zero; i++) {
    vst1q_u16(absvalues_ptr, vdupq_n_u16(0));
    vst1_u8(coef_sign_bits_ptr, vdup_n_u8(0));
    vst1_u8(eq1_bits_ptr, vdup_n_u8(0));
    absvalues_ptr += 8;
    coef_sign_bits_ptr += 8;
    eq1_bits_ptr += 8;
  }

  /* Construct zerobits bitmap. */
  uint16x8_t abs_row0 = vld1q_u16(absvalues + 0 * DCTSIZE);
  uint16x8_t abs_row1 = vld1q_u16(absvalues + 1 * DCTSIZE);
  uint16x8_t abs_row2 = vld1q_u16(absvalues + 2 * DCTSIZE);
  uint16x8_t abs_row3 = vld1q_u16(absvalues + 3 * DCTSIZE);
  uint16x8_t abs_row4 = vld1q_u16(absvalues + 4 * DCTSIZE);
  uint16x8_t abs_row5 = vld1q_u16(absvalues + 5 * DCTSIZE);
  uint16x8_t abs_row6 = vld1q_u16(absvalues + 6 * DCTSIZE);
  uint16x8_t abs_row7 = vld1q_u16(absvalues + 7 * DCTSIZE);

  uint8x8_t abs_row0_eq0 = vmovn_u16(vceqq_u16(abs_row0, vdupq_n_u16(0)));
  uint8x8_t abs_row1_eq0 = vmovn_u16(vceqq_u16(abs_row1, vdupq_n_u16(0)));
  uint8x8_t abs_row2_eq0 = vmovn_u16(vceqq_u16(abs_row2, vdupq_n_u16(0)));
  uint8x8_t abs_row3_eq0 = vmovn_u16(vceqq_u16(abs_row3, vdupq_n_u16(0)));
  uint8x8_t abs_row4_eq0 = vmovn_u16(vceqq_u16(abs_row4, vdupq_n_u16(0)));
  uint8x8_t abs_row5_eq0 = vmovn_u16(vceqq_u16(abs_row5, vdupq_n_u16(0)));
  uint8x8_t abs_row6_eq0 = vmovn_u16(vceqq_u16(abs_row6, vdupq_n_u16(0)));
  uint8x8_t abs_row7_eq0 = vmovn_u16(vceqq_u16(abs_row7, vdupq_n_u16(0)));

  /* { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 } */
  const uint8x8_t bitmap_mask =
    vreinterpret_u8_u64(vmov_n_u64(0x8040201008040201));

  abs_row0_eq0 = vand_u8(abs_row0_eq0, bitmap_mask);
  abs_row1_eq0 = vand_u8(abs_row1_eq0, bitmap_mask);
  abs_row2_eq0 = vand_u8(abs_row2_eq0, bitmap_mask);
  abs_row3_eq0 = vand_u8(abs_row3_eq0, bitmap_mask);
  abs_row4_eq0 = vand_u8(abs_row4_eq0, bitmap_mask);
  abs_row5_eq0 = vand_u8(abs_row5_eq0, bitmap_mask);
  abs_row6_eq0 = vand_u8(abs_row6_eq0, bitmap_mask);
  abs_row7_eq0 = vand_u8(abs_row7_eq0, bitmap_mask);

  uint8x8_t bitmap_rows_01 = vpadd_u8(abs_row0_eq0, abs_row1_eq0);
  uint8x8_t bitmap_rows_23 = vpadd_u8(abs_row2_eq0, abs_row3_eq0);
  uint8x8_t bitmap_rows_45 = vpadd_u8(abs_row4_eq0, abs_row5_eq0);
  uint8x8_t bitmap_rows_67 = vpadd_u8(abs_row6_eq0, abs_row7_eq0);
  uint8x8_t bitmap_rows_0123 = vpadd_u8(bitmap_rows_01, bitmap_rows_23);
  uint8x8_t bitmap_rows_4567 = vpadd_u8(bitmap_rows_45, bitmap_rows_67);
  uint8x8_t bitmap_all = vpadd_u8(bitmap_rows_0123, bitmap_rows_4567);

#if defined(__aarch64__) || defined(_M_ARM64)
  /* Move bitmap to a 64-bit scalar register. */
  uint64_t bitmap = vget_lane_u64(vreinterpret_u64_u8(bitmap_all), 0);
  /* Store zerobits bitmap. */
  bits[0] = ~bitmap;
#else
  /* Move bitmap to two 32-bit scalar registers. */
  uint32_t bitmap0 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 0);
  uint32_t bitmap1 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 1);
  /* Store zerobits bitmap. */
  bits[0] = ~bitmap0;
  bits[1] = ~bitmap1;
#endif

  /* Construct signbits bitmap. */
  uint8x8_t signbits_row0 = vld1_u8(coef_sign_bits + 0 * DCTSIZE);
  uint8x8_t signbits_row1 = vld1_u8(coef_sign_bits + 1 * DCTSIZE);
  uint8x8_t signbits_row2 = vld1_u8(coef_sign_bits + 2 * DCTSIZE);
  uint8x8_t signbits_row3 = vld1_u8(coef_sign_bits + 3 * DCTSIZE);
  uint8x8_t signbits_row4 = vld1_u8(coef_sign_bits + 4 * DCTSIZE);
  uint8x8_t signbits_row5 = vld1_u8(coef_sign_bits + 5 * DCTSIZE);
  uint8x8_t signbits_row6 = vld1_u8(coef_sign_bits + 6 * DCTSIZE);
  uint8x8_t signbits_row7 = vld1_u8(coef_sign_bits + 7 * DCTSIZE);

  signbits_row0 = vand_u8(signbits_row0, bitmap_mask);
  signbits_row1 = vand_u8(signbits_row1, bitmap_mask);
  signbits_row2 = vand_u8(signbits_row2, bitmap_mask);
  signbits_row3 = vand_u8(signbits_row3, bitmap_mask);
  signbits_row4 = vand_u8(signbits_row4, bitmap_mask);
  signbits_row5 = vand_u8(signbits_row5, bitmap_mask);
  signbits_row6 = vand_u8(signbits_row6, bitmap_mask);
  signbits_row7 = vand_u8(signbits_row7, bitmap_mask);

  bitmap_rows_01 = vpadd_u8(signbits_row0, signbits_row1);
  bitmap_rows_23 = vpadd_u8(signbits_row2, signbits_row3);
  bitmap_rows_45 = vpadd_u8(signbits_row4, signbits_row5);
  bitmap_rows_67 = vpadd_u8(signbits_row6, signbits_row7);
  bitmap_rows_0123 = vpadd_u8(bitmap_rows_01, bitmap_rows_23);
  bitmap_rows_4567 = vpadd_u8(bitmap_rows_45, bitmap_rows_67);
  bitmap_all = vpadd_u8(bitmap_rows_0123, bitmap_rows_4567);

#if defined(__aarch64__) || defined(_M_ARM64)
  /* Move bitmap to a 64-bit scalar register. */
  bitmap = vget_lane_u64(vreinterpret_u64_u8(bitmap_all), 0);
  /* Store signbits bitmap. */
  bits[1] = ~bitmap;
#else
  /* Move bitmap to two 32-bit scalar registers. */
  bitmap0 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 0);
  bitmap1 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 1);
  /* Store signbits bitmap. */
  bits[2] = ~bitmap0;
  bits[3] = ~bitmap1;
#endif

  /* Construct bitmap to find EOB position (the index of the last coefficient
   * equal to 1.)
   */
  uint8x8_t row0_eq1 = vld1_u8(coef_eq1_bits + 0 * DCTSIZE);
  uint8x8_t row1_eq1 = vld1_u8(coef_eq1_bits + 1 * DCTSIZE);
  uint8x8_t row2_eq1 = vld1_u8(coef_eq1_bits + 2 * DCTSIZE);
  uint8x8_t row3_eq1 = vld1_u8(coef_eq1_bits + 3 * DCTSIZE);
  uint8x8_t row4_eq1 = vld1_u8(coef_eq1_bits + 4 * DCTSIZE);
  uint8x8_t row5_eq1 = vld1_u8(coef_eq1_bits + 5 * DCTSIZE);
  uint8x8_t row6_eq1 = vld1_u8(coef_eq1_bits + 6 * DCTSIZE);
  uint8x8_t row7_eq1 = vld1_u8(coef_eq1_bits + 7 * DCTSIZE);

  row0_eq1 = vand_u8(row0_eq1, bitmap_mask);
  row1_eq1 = vand_u8(row1_eq1, bitmap_mask);
  row2_eq1 = vand_u8(row2_eq1, bitmap_mask);
  row3_eq1 = vand_u8(row3_eq1, bitmap_mask);
  row4_eq1 = vand_u8(row4_eq1, bitmap_mask);
  row5_eq1 = vand_u8(row5_eq1, bitmap_mask);
  row6_eq1 = vand_u8(row6_eq1, bitmap_mask);
  row7_eq1 = vand_u8(row7_eq1, bitmap_mask);

  bitmap_rows_01 = vpadd_u8(row0_eq1, row1_eq1);
  bitmap_rows_23 = vpadd_u8(row2_eq1, row3_eq1);
  bitmap_rows_45 = vpadd_u8(row4_eq1, row5_eq1);
  bitmap_rows_67 = vpadd_u8(row6_eq1, row7_eq1);
  bitmap_rows_0123 = vpadd_u8(bitmap_rows_01, bitmap_rows_23);
  bitmap_rows_4567 = vpadd_u8(bitmap_rows_45, bitmap_rows_67);
  bitmap_all = vpadd_u8(bitmap_rows_0123, bitmap_rows_4567);

#if defined(__aarch64__) || defined(_M_ARM64)
  /* Move bitmap to a 64-bit scalar register. */
  bitmap = vget_lane_u64(vreinterpret_u64_u8(bitmap_all), 0);

  /* Return EOB position. */
  if (bitmap == 0) {
    /* EOB position is defined to be 0 if all coefficients != 1. */
    return 0;
  } else {
    return 63 - BUILTIN_CLZLL(bitmap);
  }
#else
  /* Move bitmap to two 32-bit scalar registers. */
  bitmap0 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 0);
  bitmap1 = vget_lane_u32(vreinterpret_u32_u8(bitmap_all), 1);

  /* Return EOB position. */
  if (bitmap0 == 0 && bitmap1 == 0) {
    return 0;
  } else if (bitmap1 != 0) {
    return 63 - BUILTIN_CLZ(bitmap1);
  } else {
    return 31 - BUILTIN_CLZ(bitmap0);
  }
#endif
}
