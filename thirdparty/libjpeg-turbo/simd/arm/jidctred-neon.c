/*
 * jidctred-neon.c - reduced-size IDCT (Arm Neon)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2020, 2024, D. R. Commander.  All Rights Reserved.
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
#include "align.h"
#include "neon-compat.h"

#include <arm_neon.h>


#define CONST_BITS  13
#define PASS1_BITS  2

#define F_0_211  1730
#define F_0_509  4176
#define F_0_601  4926
#define F_0_720  5906
#define F_0_765  6270
#define F_0_850  6967
#define F_0_899  7373
#define F_1_061  8697
#define F_1_272  10426
#define F_1_451  11893
#define F_1_847  15137
#define F_2_172  17799
#define F_2_562  20995
#define F_3_624  29692


/* jsimd_idct_2x2_neon() is an inverse DCT function that produces reduced-size
 * 2x2 output from an 8x8 DCT block.  It uses the same calculations and
 * produces exactly the same output as IJG's original jpeg_idct_2x2() function
 * from jpeg-6b, which can be found in jidctred.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.720959822 =  5906 * 2^-13
 *    0.850430095 =  6967 * 2^-13
 *    1.272758580 = 10426 * 2^-13
 *    3.624509785 = 29692 * 2^-13
 *
 * See jidctred.c for further details of the 2x2 IDCT algorithm.  Where
 * possible, the variable names and comments here in jsimd_idct_2x2_neon()
 * match up with those in jpeg_idct_2x2().
 */

ALIGN(16) static const int16_t jsimd_idct_2x2_neon_consts[] = {
  -F_0_720, F_0_850, -F_1_272, F_3_624
};

void jsimd_idct_2x2_neon(void *dct_table, JCOEFPTR coef_block,
                         JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quantptr = dct_table;

  /* Load DCT coefficients. */
  int16x8_t row0 = vld1q_s16(coef_block + 0 * DCTSIZE);
  int16x8_t row1 = vld1q_s16(coef_block + 1 * DCTSIZE);
  int16x8_t row3 = vld1q_s16(coef_block + 3 * DCTSIZE);
  int16x8_t row5 = vld1q_s16(coef_block + 5 * DCTSIZE);
  int16x8_t row7 = vld1q_s16(coef_block + 7 * DCTSIZE);

  /* Load quantization table values. */
  int16x8_t quant_row0 = vld1q_s16(quantptr + 0 * DCTSIZE);
  int16x8_t quant_row1 = vld1q_s16(quantptr + 1 * DCTSIZE);
  int16x8_t quant_row3 = vld1q_s16(quantptr + 3 * DCTSIZE);
  int16x8_t quant_row5 = vld1q_s16(quantptr + 5 * DCTSIZE);
  int16x8_t quant_row7 = vld1q_s16(quantptr + 7 * DCTSIZE);

  /* Dequantize DCT coefficients. */
  row0 = vmulq_s16(row0, quant_row0);
  row1 = vmulq_s16(row1, quant_row1);
  row3 = vmulq_s16(row3, quant_row3);
  row5 = vmulq_s16(row5, quant_row5);
  row7 = vmulq_s16(row7, quant_row7);

  /* Load IDCT conversion constants. */
  const int16x4_t consts = vld1_s16(jsimd_idct_2x2_neon_consts);

  /* Pass 1: process columns from input, put results in vectors row0 and
   * row1.
   */

  /* Even part */
  int32x4_t tmp10_l = vshll_n_s16(vget_low_s16(row0), CONST_BITS + 2);
  int32x4_t tmp10_h = vshll_n_s16(vget_high_s16(row0), CONST_BITS + 2);

  /* Odd part */
  int32x4_t tmp0_l = vmull_lane_s16(vget_low_s16(row1), consts, 3);
  tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(row3), consts, 2);
  tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(row5), consts, 1);
  tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(row7), consts, 0);
  int32x4_t tmp0_h = vmull_lane_s16(vget_high_s16(row1), consts, 3);
  tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(row3), consts, 2);
  tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(row5), consts, 1);
  tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(row7), consts, 0);

  /* Final output stage: descale and narrow to 16-bit. */
  row0 = vcombine_s16(vrshrn_n_s32(vaddq_s32(tmp10_l, tmp0_l), CONST_BITS),
                      vrshrn_n_s32(vaddq_s32(tmp10_h, tmp0_h), CONST_BITS));
  row1 = vcombine_s16(vrshrn_n_s32(vsubq_s32(tmp10_l, tmp0_l), CONST_BITS),
                      vrshrn_n_s32(vsubq_s32(tmp10_h, tmp0_h), CONST_BITS));

  /* Transpose two rows, ready for second pass. */
  int16x8x2_t cols_0246_1357 = vtrnq_s16(row0, row1);
  int16x8_t cols_0246 = cols_0246_1357.val[0];
  int16x8_t cols_1357 = cols_0246_1357.val[1];
  /* Duplicate columns such that each is accessible in its own vector. */
  int32x4x2_t cols_1155_3377 = vtrnq_s32(vreinterpretq_s32_s16(cols_1357),
                                         vreinterpretq_s32_s16(cols_1357));
  int16x8_t cols_1155 = vreinterpretq_s16_s32(cols_1155_3377.val[0]);
  int16x8_t cols_3377 = vreinterpretq_s16_s32(cols_1155_3377.val[1]);

  /* Pass 2: process two rows, store to output array. */

  /* Even part: we're only interested in col0; the top half of tmp10 is "don't
   * care."
   */
  int32x4_t tmp10 = vshll_n_s16(vget_low_s16(cols_0246), CONST_BITS + 2);

  /* Odd part: we're only interested in the bottom half of tmp0. */
  int32x4_t tmp0 = vmull_lane_s16(vget_low_s16(cols_1155), consts, 3);
  tmp0 = vmlal_lane_s16(tmp0, vget_low_s16(cols_3377), consts, 2);
  tmp0 = vmlal_lane_s16(tmp0, vget_high_s16(cols_1155), consts, 1);
  tmp0 = vmlal_lane_s16(tmp0, vget_high_s16(cols_3377), consts, 0);

  /* Final output stage: descale and clamp to range [0-255]. */
  int16x8_t output_s16 = vcombine_s16(vaddhn_s32(tmp10, tmp0),
                                      vsubhn_s32(tmp10, tmp0));
  output_s16 = vrsraq_n_s16(vdupq_n_s16(CENTERJSAMPLE), output_s16,
                            CONST_BITS + PASS1_BITS + 3 + 2 - 16);
  /* Narrow to 8-bit and convert to unsigned. */
  uint8x8_t output_u8 = vqmovun_s16(output_s16);

  /* Store 2x2 block to memory. */
  vst1_lane_u8(output_buf[0] + output_col, output_u8, 0);
  vst1_lane_u8(output_buf[1] + output_col, output_u8, 1);
  vst1_lane_u8(output_buf[0] + output_col + 1, output_u8, 4);
  vst1_lane_u8(output_buf[1] + output_col + 1, output_u8, 5);
}


/* jsimd_idct_4x4_neon() is an inverse DCT function that produces reduced-size
 * 4x4 output from an 8x8 DCT block.  It uses the same calculations and
 * produces exactly the same output as IJG's original jpeg_idct_4x4() function
 * from jpeg-6b, which can be found in jidctred.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.211164243 =  1730 * 2^-13
 *    0.509795579 =  4176 * 2^-13
 *    0.601344887 =  4926 * 2^-13
 *    0.765366865 =  6270 * 2^-13
 *    0.899976223 =  7373 * 2^-13
 *    1.061594337 =  8697 * 2^-13
 *    1.451774981 = 11893 * 2^-13
 *    1.847759065 = 15137 * 2^-13
 *    2.172734803 = 17799 * 2^-13
 *    2.562915447 = 20995 * 2^-13
 *
 * See jidctred.c for further details of the 4x4 IDCT algorithm.  Where
 * possible, the variable names and comments here in jsimd_idct_4x4_neon()
 * match up with those in jpeg_idct_4x4().
 */

ALIGN(16) static const int16_t jsimd_idct_4x4_neon_consts[] = {
  F_1_847, -F_0_765, -F_0_211,  F_1_451,
 -F_2_172,  F_1_061, -F_0_509, -F_0_601,
  F_0_899,  F_2_562,        0,        0
};

void jsimd_idct_4x4_neon(void *dct_table, JCOEFPTR coef_block,
                         JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quantptr = dct_table;

  /* Load DCT coefficients. */
  int16x8_t row0  = vld1q_s16(coef_block + 0 * DCTSIZE);
  int16x8_t row1  = vld1q_s16(coef_block + 1 * DCTSIZE);
  int16x8_t row2  = vld1q_s16(coef_block + 2 * DCTSIZE);
  int16x8_t row3  = vld1q_s16(coef_block + 3 * DCTSIZE);
  int16x8_t row5  = vld1q_s16(coef_block + 5 * DCTSIZE);
  int16x8_t row6  = vld1q_s16(coef_block + 6 * DCTSIZE);
  int16x8_t row7  = vld1q_s16(coef_block + 7 * DCTSIZE);

  /* Load quantization table values for DC coefficients. */
  int16x8_t quant_row0 = vld1q_s16(quantptr + 0 * DCTSIZE);
  /* Dequantize DC coefficients. */
  row0 = vmulq_s16(row0, quant_row0);

  /* Construct bitmap to test if all AC coefficients are 0. */
  int16x8_t bitmap = vorrq_s16(row1, row2);
  bitmap = vorrq_s16(bitmap, row3);
  bitmap = vorrq_s16(bitmap, row5);
  bitmap = vorrq_s16(bitmap, row6);
  bitmap = vorrq_s16(bitmap, row7);

  int64_t left_ac_bitmap = vgetq_lane_s64(vreinterpretq_s64_s16(bitmap), 0);
  int64_t right_ac_bitmap = vgetq_lane_s64(vreinterpretq_s64_s16(bitmap), 1);

  /* Load constants for IDCT computation. */
#ifdef HAVE_VLD1_S16_X3
  const int16x4x3_t consts = vld1_s16_x3(jsimd_idct_4x4_neon_consts);
#else
  /* GCC does not currently support the intrinsic vld1_<type>_x3(). */
  const int16x4_t consts1 = vld1_s16(jsimd_idct_4x4_neon_consts);
  const int16x4_t consts2 = vld1_s16(jsimd_idct_4x4_neon_consts + 4);
  const int16x4_t consts3 = vld1_s16(jsimd_idct_4x4_neon_consts + 8);
  const int16x4x3_t consts = { { consts1, consts2, consts3 } };
#endif

  if (left_ac_bitmap == 0 && right_ac_bitmap == 0) {
    /* All AC coefficients are zero.
     * Compute DC values and duplicate into row vectors 0, 1, 2, and 3.
     */
    int16x8_t dcval = vshlq_n_s16(row0, PASS1_BITS);
    row0 = dcval;
    row1 = dcval;
    row2 = dcval;
    row3 = dcval;
  } else if (left_ac_bitmap == 0) {
    /* AC coefficients are zero for columns 0, 1, 2, and 3.
     * Compute DC values for these columns.
     */
    int16x4_t dcval = vshl_n_s16(vget_low_s16(row0), PASS1_BITS);

    /* Commence regular IDCT computation for columns 4, 5, 6, and 7. */

    /* Load quantization table. */
    int16x4_t quant_row1 = vld1_s16(quantptr + 1 * DCTSIZE + 4);
    int16x4_t quant_row2 = vld1_s16(quantptr + 2 * DCTSIZE + 4);
    int16x4_t quant_row3 = vld1_s16(quantptr + 3 * DCTSIZE + 4);
    int16x4_t quant_row5 = vld1_s16(quantptr + 5 * DCTSIZE + 4);
    int16x4_t quant_row6 = vld1_s16(quantptr + 6 * DCTSIZE + 4);
    int16x4_t quant_row7 = vld1_s16(quantptr + 7 * DCTSIZE + 4);

    /* Even part */
    int32x4_t tmp0 = vshll_n_s16(vget_high_s16(row0), CONST_BITS + 1);

    int16x4_t z2 = vmul_s16(vget_high_s16(row2), quant_row2);
    int16x4_t z3 = vmul_s16(vget_high_s16(row6), quant_row6);

    int32x4_t tmp2 = vmull_lane_s16(z2, consts.val[0], 0);
    tmp2 = vmlal_lane_s16(tmp2, z3, consts.val[0], 1);

    int32x4_t tmp10 = vaddq_s32(tmp0, tmp2);
    int32x4_t tmp12 = vsubq_s32(tmp0, tmp2);

    /* Odd part */
    int16x4_t z1 = vmul_s16(vget_high_s16(row7), quant_row7);
    z2 = vmul_s16(vget_high_s16(row5), quant_row5);
    z3 = vmul_s16(vget_high_s16(row3), quant_row3);
    int16x4_t z4 = vmul_s16(vget_high_s16(row1), quant_row1);

    tmp0 = vmull_lane_s16(z1, consts.val[0], 2);
    tmp0 = vmlal_lane_s16(tmp0, z2, consts.val[0], 3);
    tmp0 = vmlal_lane_s16(tmp0, z3, consts.val[1], 0);
    tmp0 = vmlal_lane_s16(tmp0, z4, consts.val[1], 1);

    tmp2 = vmull_lane_s16(z1, consts.val[1], 2);
    tmp2 = vmlal_lane_s16(tmp2, z2, consts.val[1], 3);
    tmp2 = vmlal_lane_s16(tmp2, z3, consts.val[2], 0);
    tmp2 = vmlal_lane_s16(tmp2, z4, consts.val[2], 1);

    /* Final output stage: descale and narrow to 16-bit. */
    row0 = vcombine_s16(dcval, vrshrn_n_s32(vaddq_s32(tmp10, tmp2),
                                            CONST_BITS - PASS1_BITS + 1));
    row3 = vcombine_s16(dcval, vrshrn_n_s32(vsubq_s32(tmp10, tmp2),
                                            CONST_BITS - PASS1_BITS + 1));
    row1 = vcombine_s16(dcval, vrshrn_n_s32(vaddq_s32(tmp12, tmp0),
                                            CONST_BITS - PASS1_BITS + 1));
    row2 = vcombine_s16(dcval, vrshrn_n_s32(vsubq_s32(tmp12, tmp0),
                                            CONST_BITS - PASS1_BITS + 1));
  } else if (right_ac_bitmap == 0) {
    /* AC coefficients are zero for columns 4, 5, 6, and 7.
     * Compute DC values for these columns.
     */
    int16x4_t dcval = vshl_n_s16(vget_high_s16(row0), PASS1_BITS);

    /* Commence regular IDCT computation for columns 0, 1, 2, and 3. */

    /* Load quantization table. */
    int16x4_t quant_row1 = vld1_s16(quantptr + 1 * DCTSIZE);
    int16x4_t quant_row2 = vld1_s16(quantptr + 2 * DCTSIZE);
    int16x4_t quant_row3 = vld1_s16(quantptr + 3 * DCTSIZE);
    int16x4_t quant_row5 = vld1_s16(quantptr + 5 * DCTSIZE);
    int16x4_t quant_row6 = vld1_s16(quantptr + 6 * DCTSIZE);
    int16x4_t quant_row7 = vld1_s16(quantptr + 7 * DCTSIZE);

    /* Even part */
    int32x4_t tmp0 = vshll_n_s16(vget_low_s16(row0), CONST_BITS + 1);

    int16x4_t z2 = vmul_s16(vget_low_s16(row2), quant_row2);
    int16x4_t z3 = vmul_s16(vget_low_s16(row6), quant_row6);

    int32x4_t tmp2 = vmull_lane_s16(z2, consts.val[0], 0);
    tmp2 = vmlal_lane_s16(tmp2, z3, consts.val[0], 1);

    int32x4_t tmp10 = vaddq_s32(tmp0, tmp2);
    int32x4_t tmp12 = vsubq_s32(tmp0, tmp2);

    /* Odd part */
    int16x4_t z1 = vmul_s16(vget_low_s16(row7), quant_row7);
    z2 = vmul_s16(vget_low_s16(row5), quant_row5);
    z3 = vmul_s16(vget_low_s16(row3), quant_row3);
    int16x4_t z4 = vmul_s16(vget_low_s16(row1), quant_row1);

    tmp0 = vmull_lane_s16(z1, consts.val[0], 2);
    tmp0 = vmlal_lane_s16(tmp0, z2, consts.val[0], 3);
    tmp0 = vmlal_lane_s16(tmp0, z3, consts.val[1], 0);
    tmp0 = vmlal_lane_s16(tmp0, z4, consts.val[1], 1);

    tmp2 = vmull_lane_s16(z1, consts.val[1], 2);
    tmp2 = vmlal_lane_s16(tmp2, z2, consts.val[1], 3);
    tmp2 = vmlal_lane_s16(tmp2, z3, consts.val[2], 0);
    tmp2 = vmlal_lane_s16(tmp2, z4, consts.val[2], 1);

    /* Final output stage: descale and narrow to 16-bit. */
    row0 = vcombine_s16(vrshrn_n_s32(vaddq_s32(tmp10, tmp2),
                                     CONST_BITS - PASS1_BITS + 1), dcval);
    row3 = vcombine_s16(vrshrn_n_s32(vsubq_s32(tmp10, tmp2),
                                     CONST_BITS - PASS1_BITS + 1), dcval);
    row1 = vcombine_s16(vrshrn_n_s32(vaddq_s32(tmp12, tmp0),
                                     CONST_BITS - PASS1_BITS + 1), dcval);
    row2 = vcombine_s16(vrshrn_n_s32(vsubq_s32(tmp12, tmp0),
                                     CONST_BITS - PASS1_BITS + 1), dcval);
  } else {
    /* All AC coefficients are non-zero; full IDCT calculation required. */
    int16x8_t quant_row1 = vld1q_s16(quantptr + 1 * DCTSIZE);
    int16x8_t quant_row2 = vld1q_s16(quantptr + 2 * DCTSIZE);
    int16x8_t quant_row3 = vld1q_s16(quantptr + 3 * DCTSIZE);
    int16x8_t quant_row5 = vld1q_s16(quantptr + 5 * DCTSIZE);
    int16x8_t quant_row6 = vld1q_s16(quantptr + 6 * DCTSIZE);
    int16x8_t quant_row7 = vld1q_s16(quantptr + 7 * DCTSIZE);

    /* Even part */
    int32x4_t tmp0_l = vshll_n_s16(vget_low_s16(row0), CONST_BITS + 1);
    int32x4_t tmp0_h = vshll_n_s16(vget_high_s16(row0), CONST_BITS + 1);

    int16x8_t z2 = vmulq_s16(row2, quant_row2);
    int16x8_t z3 = vmulq_s16(row6, quant_row6);

    int32x4_t tmp2_l = vmull_lane_s16(vget_low_s16(z2), consts.val[0], 0);
    int32x4_t tmp2_h = vmull_lane_s16(vget_high_s16(z2), consts.val[0], 0);
    tmp2_l = vmlal_lane_s16(tmp2_l, vget_low_s16(z3), consts.val[0], 1);
    tmp2_h = vmlal_lane_s16(tmp2_h, vget_high_s16(z3), consts.val[0], 1);

    int32x4_t tmp10_l = vaddq_s32(tmp0_l, tmp2_l);
    int32x4_t tmp10_h = vaddq_s32(tmp0_h, tmp2_h);
    int32x4_t tmp12_l = vsubq_s32(tmp0_l, tmp2_l);
    int32x4_t tmp12_h = vsubq_s32(tmp0_h, tmp2_h);

    /* Odd part */
    int16x8_t z1 = vmulq_s16(row7, quant_row7);
    z2 = vmulq_s16(row5, quant_row5);
    z3 = vmulq_s16(row3, quant_row3);
    int16x8_t z4 = vmulq_s16(row1, quant_row1);

    tmp0_l = vmull_lane_s16(vget_low_s16(z1), consts.val[0], 2);
    tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(z2), consts.val[0], 3);
    tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(z3), consts.val[1], 0);
    tmp0_l = vmlal_lane_s16(tmp0_l, vget_low_s16(z4), consts.val[1], 1);
    tmp0_h = vmull_lane_s16(vget_high_s16(z1), consts.val[0], 2);
    tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(z2), consts.val[0], 3);
    tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(z3), consts.val[1], 0);
    tmp0_h = vmlal_lane_s16(tmp0_h, vget_high_s16(z4), consts.val[1], 1);

    tmp2_l = vmull_lane_s16(vget_low_s16(z1), consts.val[1], 2);
    tmp2_l = vmlal_lane_s16(tmp2_l, vget_low_s16(z2), consts.val[1], 3);
    tmp2_l = vmlal_lane_s16(tmp2_l, vget_low_s16(z3), consts.val[2], 0);
    tmp2_l = vmlal_lane_s16(tmp2_l, vget_low_s16(z4), consts.val[2], 1);
    tmp2_h = vmull_lane_s16(vget_high_s16(z1), consts.val[1], 2);
    tmp2_h = vmlal_lane_s16(tmp2_h, vget_high_s16(z2), consts.val[1], 3);
    tmp2_h = vmlal_lane_s16(tmp2_h, vget_high_s16(z3), consts.val[2], 0);
    tmp2_h = vmlal_lane_s16(tmp2_h, vget_high_s16(z4), consts.val[2], 1);

    /* Final output stage: descale and narrow to 16-bit. */
    row0 = vcombine_s16(vrshrn_n_s32(vaddq_s32(tmp10_l, tmp2_l),
                                     CONST_BITS - PASS1_BITS + 1),
                        vrshrn_n_s32(vaddq_s32(tmp10_h, tmp2_h),
                                     CONST_BITS - PASS1_BITS + 1));
    row3 = vcombine_s16(vrshrn_n_s32(vsubq_s32(tmp10_l, tmp2_l),
                                     CONST_BITS - PASS1_BITS + 1),
                        vrshrn_n_s32(vsubq_s32(tmp10_h, tmp2_h),
                                     CONST_BITS - PASS1_BITS + 1));
    row1 = vcombine_s16(vrshrn_n_s32(vaddq_s32(tmp12_l, tmp0_l),
                                     CONST_BITS - PASS1_BITS + 1),
                        vrshrn_n_s32(vaddq_s32(tmp12_h, tmp0_h),
                                     CONST_BITS - PASS1_BITS + 1));
    row2 = vcombine_s16(vrshrn_n_s32(vsubq_s32(tmp12_l, tmp0_l),
                                     CONST_BITS - PASS1_BITS + 1),
                        vrshrn_n_s32(vsubq_s32(tmp12_h, tmp0_h),
                                     CONST_BITS - PASS1_BITS + 1));
  }

  /* Transpose 8x4 block to perform IDCT on rows in second pass. */
  int16x8x2_t row_01 = vtrnq_s16(row0, row1);
  int16x8x2_t row_23 = vtrnq_s16(row2, row3);

  int32x4x2_t cols_0426 = vtrnq_s32(vreinterpretq_s32_s16(row_01.val[0]),
                                    vreinterpretq_s32_s16(row_23.val[0]));
  int32x4x2_t cols_1537 = vtrnq_s32(vreinterpretq_s32_s16(row_01.val[1]),
                                    vreinterpretq_s32_s16(row_23.val[1]));

  int16x4_t col0 = vreinterpret_s16_s32(vget_low_s32(cols_0426.val[0]));
  int16x4_t col1 = vreinterpret_s16_s32(vget_low_s32(cols_1537.val[0]));
  int16x4_t col2 = vreinterpret_s16_s32(vget_low_s32(cols_0426.val[1]));
  int16x4_t col3 = vreinterpret_s16_s32(vget_low_s32(cols_1537.val[1]));
  int16x4_t col5 = vreinterpret_s16_s32(vget_high_s32(cols_1537.val[0]));
  int16x4_t col6 = vreinterpret_s16_s32(vget_high_s32(cols_0426.val[1]));
  int16x4_t col7 = vreinterpret_s16_s32(vget_high_s32(cols_1537.val[1]));

  /* Commence second pass of IDCT. */

  /* Even part */
  int32x4_t tmp0 = vshll_n_s16(col0, CONST_BITS + 1);
  int32x4_t tmp2 = vmull_lane_s16(col2, consts.val[0], 0);
  tmp2 = vmlal_lane_s16(tmp2, col6, consts.val[0], 1);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp0, tmp2);

  /* Odd part */
  tmp0 = vmull_lane_s16(col7, consts.val[0], 2);
  tmp0 = vmlal_lane_s16(tmp0, col5, consts.val[0], 3);
  tmp0 = vmlal_lane_s16(tmp0, col3, consts.val[1], 0);
  tmp0 = vmlal_lane_s16(tmp0, col1, consts.val[1], 1);

  tmp2 = vmull_lane_s16(col7, consts.val[1], 2);
  tmp2 = vmlal_lane_s16(tmp2, col5, consts.val[1], 3);
  tmp2 = vmlal_lane_s16(tmp2, col3, consts.val[2], 0);
  tmp2 = vmlal_lane_s16(tmp2, col1, consts.val[2], 1);

  /* Final output stage: descale and clamp to range [0-255]. */
  int16x8_t output_cols_02 = vcombine_s16(vaddhn_s32(tmp10, tmp2),
                                          vsubhn_s32(tmp12, tmp0));
  int16x8_t output_cols_13 = vcombine_s16(vaddhn_s32(tmp12, tmp0),
                                          vsubhn_s32(tmp10, tmp2));
  output_cols_02 = vrsraq_n_s16(vdupq_n_s16(CENTERJSAMPLE), output_cols_02,
                                CONST_BITS + PASS1_BITS + 3 + 1 - 16);
  output_cols_13 = vrsraq_n_s16(vdupq_n_s16(CENTERJSAMPLE), output_cols_13,
                                CONST_BITS + PASS1_BITS + 3 + 1 - 16);
  /* Narrow to 8-bit and convert to unsigned while zipping 8-bit elements.
   * An interleaving store completes the transpose.
   */
  uint8x8x2_t output_0123 = vzip_u8(vqmovun_s16(output_cols_02),
                                    vqmovun_s16(output_cols_13));
  uint16x4x2_t output_01_23 = { {
    vreinterpret_u16_u8(output_0123.val[0]),
    vreinterpret_u16_u8(output_0123.val[1])
  } };

  /* Store 4x4 block to memory. */
  JSAMPROW outptr0 = output_buf[0] + output_col;
  JSAMPROW outptr1 = output_buf[1] + output_col;
  JSAMPROW outptr2 = output_buf[2] + output_col;
  JSAMPROW outptr3 = output_buf[3] + output_col;
  vst2_lane_u16((uint16_t *)outptr0, output_01_23, 0);
  vst2_lane_u16((uint16_t *)outptr1, output_01_23, 1);
  vst2_lane_u16((uint16_t *)outptr2, output_01_23, 2);
  vst2_lane_u16((uint16_t *)outptr3, output_01_23, 3);
}
