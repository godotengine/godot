/*
 * jfdctint-neon.c - accurate integer FDCT (Arm Neon)
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


/* jsimd_fdct_islow_neon() performs a slower but more accurate forward DCT
 * (Discrete Cosine Transform) on one block of samples.  It uses the same
 * calculations and produces exactly the same output as IJG's original
 * jpeg_fdct_islow() function, which can be found in jfdctint.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.298631336 =  2446 * 2^-13
 *    0.390180644 =  3196 * 2^-13
 *    0.541196100 =  4433 * 2^-13
 *    0.765366865 =  6270 * 2^-13
 *    0.899976223 =  7373 * 2^-13
 *    1.175875602 =  9633 * 2^-13
 *    1.501321110 = 12299 * 2^-13
 *    1.847759065 = 15137 * 2^-13
 *    1.961570560 = 16069 * 2^-13
 *    2.053119869 = 16819 * 2^-13
 *    2.562915447 = 20995 * 2^-13
 *    3.072711026 = 25172 * 2^-13
 *
 * See jfdctint.c for further details of the DCT algorithm.  Where possible,
 * the variable names and comments here in jsimd_fdct_islow_neon() match up
 * with those in jpeg_fdct_islow().
 */

#define CONST_BITS  13
#define PASS1_BITS  2

#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS)

#define F_0_298  2446
#define F_0_390  3196
#define F_0_541  4433
#define F_0_765  6270
#define F_0_899  7373
#define F_1_175  9633
#define F_1_501  12299
#define F_1_847  15137
#define F_1_961  16069
#define F_2_053  16819
#define F_2_562  20995
#define F_3_072  25172


ALIGN(16) static const int16_t jsimd_fdct_islow_neon_consts[] = {
  F_0_298, -F_0_390,  F_0_541,  F_0_765,
 -F_0_899,  F_1_175,  F_1_501, -F_1_847,
 -F_1_961,  F_2_053, -F_2_562,  F_3_072
};

void jsimd_fdct_islow_neon(DCTELEM *data)
{
  /* Load DCT constants. */
#ifdef HAVE_VLD1_S16_X3
  const int16x4x3_t consts = vld1_s16_x3(jsimd_fdct_islow_neon_consts);
#else
  /* GCC does not currently support the intrinsic vld1_<type>_x3(). */
  const int16x4_t consts1 = vld1_s16(jsimd_fdct_islow_neon_consts);
  const int16x4_t consts2 = vld1_s16(jsimd_fdct_islow_neon_consts + 4);
  const int16x4_t consts3 = vld1_s16(jsimd_fdct_islow_neon_consts + 8);
  const int16x4x3_t consts = { { consts1, consts2, consts3 } };
#endif

  /* Load an 8x8 block of samples into Neon registers.  De-interleaving loads
   * are used, followed by vuzp to transpose the block such that we have a
   * column of samples per vector - allowing all rows to be processed at once.
   */
  int16x8x4_t s_rows_0123 = vld4q_s16(data);
  int16x8x4_t s_rows_4567 = vld4q_s16(data + 4 * DCTSIZE);

  int16x8x2_t cols_04 = vuzpq_s16(s_rows_0123.val[0], s_rows_4567.val[0]);
  int16x8x2_t cols_15 = vuzpq_s16(s_rows_0123.val[1], s_rows_4567.val[1]);
  int16x8x2_t cols_26 = vuzpq_s16(s_rows_0123.val[2], s_rows_4567.val[2]);
  int16x8x2_t cols_37 = vuzpq_s16(s_rows_0123.val[3], s_rows_4567.val[3]);

  int16x8_t col0 = cols_04.val[0];
  int16x8_t col1 = cols_15.val[0];
  int16x8_t col2 = cols_26.val[0];
  int16x8_t col3 = cols_37.val[0];
  int16x8_t col4 = cols_04.val[1];
  int16x8_t col5 = cols_15.val[1];
  int16x8_t col6 = cols_26.val[1];
  int16x8_t col7 = cols_37.val[1];

  /* Pass 1: process rows. */

  int16x8_t tmp0 = vaddq_s16(col0, col7);
  int16x8_t tmp7 = vsubq_s16(col0, col7);
  int16x8_t tmp1 = vaddq_s16(col1, col6);
  int16x8_t tmp6 = vsubq_s16(col1, col6);
  int16x8_t tmp2 = vaddq_s16(col2, col5);
  int16x8_t tmp5 = vsubq_s16(col2, col5);
  int16x8_t tmp3 = vaddq_s16(col3, col4);
  int16x8_t tmp4 = vsubq_s16(col3, col4);

  /* Even part */
  int16x8_t tmp10 = vaddq_s16(tmp0, tmp3);
  int16x8_t tmp13 = vsubq_s16(tmp0, tmp3);
  int16x8_t tmp11 = vaddq_s16(tmp1, tmp2);
  int16x8_t tmp12 = vsubq_s16(tmp1, tmp2);

  col0 = vshlq_n_s16(vaddq_s16(tmp10, tmp11), PASS1_BITS);
  col4 = vshlq_n_s16(vsubq_s16(tmp10, tmp11), PASS1_BITS);

  int16x8_t tmp12_add_tmp13 = vaddq_s16(tmp12, tmp13);
  int32x4_t z1_l =
    vmull_lane_s16(vget_low_s16(tmp12_add_tmp13), consts.val[0], 2);
  int32x4_t z1_h =
    vmull_lane_s16(vget_high_s16(tmp12_add_tmp13), consts.val[0], 2);

  int32x4_t col2_scaled_l =
    vmlal_lane_s16(z1_l, vget_low_s16(tmp13), consts.val[0], 3);
  int32x4_t col2_scaled_h =
    vmlal_lane_s16(z1_h, vget_high_s16(tmp13), consts.val[0], 3);
  col2 = vcombine_s16(vrshrn_n_s32(col2_scaled_l, DESCALE_P1),
                      vrshrn_n_s32(col2_scaled_h, DESCALE_P1));

  int32x4_t col6_scaled_l =
    vmlal_lane_s16(z1_l, vget_low_s16(tmp12), consts.val[1], 3);
  int32x4_t col6_scaled_h =
    vmlal_lane_s16(z1_h, vget_high_s16(tmp12), consts.val[1], 3);
  col6 = vcombine_s16(vrshrn_n_s32(col6_scaled_l, DESCALE_P1),
                      vrshrn_n_s32(col6_scaled_h, DESCALE_P1));

  /* Odd part */
  int16x8_t z1 = vaddq_s16(tmp4, tmp7);
  int16x8_t z2 = vaddq_s16(tmp5, tmp6);
  int16x8_t z3 = vaddq_s16(tmp4, tmp6);
  int16x8_t z4 = vaddq_s16(tmp5, tmp7);
  /* sqrt(2) * c3 */
  int32x4_t z5_l = vmull_lane_s16(vget_low_s16(z3), consts.val[1], 1);
  int32x4_t z5_h = vmull_lane_s16(vget_high_s16(z3), consts.val[1], 1);
  z5_l = vmlal_lane_s16(z5_l, vget_low_s16(z4), consts.val[1], 1);
  z5_h = vmlal_lane_s16(z5_h, vget_high_s16(z4), consts.val[1], 1);

  /* sqrt(2) * (-c1+c3+c5-c7) */
  int32x4_t tmp4_l = vmull_lane_s16(vget_low_s16(tmp4), consts.val[0], 0);
  int32x4_t tmp4_h = vmull_lane_s16(vget_high_s16(tmp4), consts.val[0], 0);
  /* sqrt(2) * ( c1+c3-c5+c7) */
  int32x4_t tmp5_l = vmull_lane_s16(vget_low_s16(tmp5), consts.val[2], 1);
  int32x4_t tmp5_h = vmull_lane_s16(vget_high_s16(tmp5), consts.val[2], 1);
  /* sqrt(2) * ( c1+c3+c5-c7) */
  int32x4_t tmp6_l = vmull_lane_s16(vget_low_s16(tmp6), consts.val[2], 3);
  int32x4_t tmp6_h = vmull_lane_s16(vget_high_s16(tmp6), consts.val[2], 3);
  /* sqrt(2) * ( c1+c3-c5-c7) */
  int32x4_t tmp7_l = vmull_lane_s16(vget_low_s16(tmp7), consts.val[1], 2);
  int32x4_t tmp7_h = vmull_lane_s16(vget_high_s16(tmp7), consts.val[1], 2);

  /* sqrt(2) * (c7-c3) */
  z1_l = vmull_lane_s16(vget_low_s16(z1), consts.val[1], 0);
  z1_h = vmull_lane_s16(vget_high_s16(z1), consts.val[1], 0);
  /* sqrt(2) * (-c1-c3) */
  int32x4_t z2_l = vmull_lane_s16(vget_low_s16(z2), consts.val[2], 2);
  int32x4_t z2_h = vmull_lane_s16(vget_high_s16(z2), consts.val[2], 2);
  /* sqrt(2) * (-c3-c5) */
  int32x4_t z3_l = vmull_lane_s16(vget_low_s16(z3), consts.val[2], 0);
  int32x4_t z3_h = vmull_lane_s16(vget_high_s16(z3), consts.val[2], 0);
  /* sqrt(2) * (c5-c3) */
  int32x4_t z4_l = vmull_lane_s16(vget_low_s16(z4), consts.val[0], 1);
  int32x4_t z4_h = vmull_lane_s16(vget_high_s16(z4), consts.val[0], 1);

  z3_l = vaddq_s32(z3_l, z5_l);
  z3_h = vaddq_s32(z3_h, z5_h);
  z4_l = vaddq_s32(z4_l, z5_l);
  z4_h = vaddq_s32(z4_h, z5_h);

  tmp4_l = vaddq_s32(tmp4_l, z1_l);
  tmp4_h = vaddq_s32(tmp4_h, z1_h);
  tmp4_l = vaddq_s32(tmp4_l, z3_l);
  tmp4_h = vaddq_s32(tmp4_h, z3_h);
  col7 = vcombine_s16(vrshrn_n_s32(tmp4_l, DESCALE_P1),
                      vrshrn_n_s32(tmp4_h, DESCALE_P1));

  tmp5_l = vaddq_s32(tmp5_l, z2_l);
  tmp5_h = vaddq_s32(tmp5_h, z2_h);
  tmp5_l = vaddq_s32(tmp5_l, z4_l);
  tmp5_h = vaddq_s32(tmp5_h, z4_h);
  col5 = vcombine_s16(vrshrn_n_s32(tmp5_l, DESCALE_P1),
                      vrshrn_n_s32(tmp5_h, DESCALE_P1));

  tmp6_l = vaddq_s32(tmp6_l, z2_l);
  tmp6_h = vaddq_s32(tmp6_h, z2_h);
  tmp6_l = vaddq_s32(tmp6_l, z3_l);
  tmp6_h = vaddq_s32(tmp6_h, z3_h);
  col3 = vcombine_s16(vrshrn_n_s32(tmp6_l, DESCALE_P1),
                      vrshrn_n_s32(tmp6_h, DESCALE_P1));

  tmp7_l = vaddq_s32(tmp7_l, z1_l);
  tmp7_h = vaddq_s32(tmp7_h, z1_h);
  tmp7_l = vaddq_s32(tmp7_l, z4_l);
  tmp7_h = vaddq_s32(tmp7_h, z4_h);
  col1 = vcombine_s16(vrshrn_n_s32(tmp7_l, DESCALE_P1),
                      vrshrn_n_s32(tmp7_h, DESCALE_P1));

  /* Transpose to work on columns in pass 2. */
  int16x8x2_t cols_01 = vtrnq_s16(col0, col1);
  int16x8x2_t cols_23 = vtrnq_s16(col2, col3);
  int16x8x2_t cols_45 = vtrnq_s16(col4, col5);
  int16x8x2_t cols_67 = vtrnq_s16(col6, col7);

  int32x4x2_t cols_0145_l = vtrnq_s32(vreinterpretq_s32_s16(cols_01.val[0]),
                                      vreinterpretq_s32_s16(cols_45.val[0]));
  int32x4x2_t cols_0145_h = vtrnq_s32(vreinterpretq_s32_s16(cols_01.val[1]),
                                      vreinterpretq_s32_s16(cols_45.val[1]));
  int32x4x2_t cols_2367_l = vtrnq_s32(vreinterpretq_s32_s16(cols_23.val[0]),
                                      vreinterpretq_s32_s16(cols_67.val[0]));
  int32x4x2_t cols_2367_h = vtrnq_s32(vreinterpretq_s32_s16(cols_23.val[1]),
                                      vreinterpretq_s32_s16(cols_67.val[1]));

  int32x4x2_t rows_04 = vzipq_s32(cols_0145_l.val[0], cols_2367_l.val[0]);
  int32x4x2_t rows_15 = vzipq_s32(cols_0145_h.val[0], cols_2367_h.val[0]);
  int32x4x2_t rows_26 = vzipq_s32(cols_0145_l.val[1], cols_2367_l.val[1]);
  int32x4x2_t rows_37 = vzipq_s32(cols_0145_h.val[1], cols_2367_h.val[1]);

  int16x8_t row0 = vreinterpretq_s16_s32(rows_04.val[0]);
  int16x8_t row1 = vreinterpretq_s16_s32(rows_15.val[0]);
  int16x8_t row2 = vreinterpretq_s16_s32(rows_26.val[0]);
  int16x8_t row3 = vreinterpretq_s16_s32(rows_37.val[0]);
  int16x8_t row4 = vreinterpretq_s16_s32(rows_04.val[1]);
  int16x8_t row5 = vreinterpretq_s16_s32(rows_15.val[1]);
  int16x8_t row6 = vreinterpretq_s16_s32(rows_26.val[1]);
  int16x8_t row7 = vreinterpretq_s16_s32(rows_37.val[1]);

  /* Pass 2: process columns. */

  tmp0 = vaddq_s16(row0, row7);
  tmp7 = vsubq_s16(row0, row7);
  tmp1 = vaddq_s16(row1, row6);
  tmp6 = vsubq_s16(row1, row6);
  tmp2 = vaddq_s16(row2, row5);
  tmp5 = vsubq_s16(row2, row5);
  tmp3 = vaddq_s16(row3, row4);
  tmp4 = vsubq_s16(row3, row4);

  /* Even part */
  tmp10 = vaddq_s16(tmp0, tmp3);
  tmp13 = vsubq_s16(tmp0, tmp3);
  tmp11 = vaddq_s16(tmp1, tmp2);
  tmp12 = vsubq_s16(tmp1, tmp2);

  row0 = vrshrq_n_s16(vaddq_s16(tmp10, tmp11), PASS1_BITS);
  row4 = vrshrq_n_s16(vsubq_s16(tmp10, tmp11), PASS1_BITS);

  tmp12_add_tmp13 = vaddq_s16(tmp12, tmp13);
  z1_l = vmull_lane_s16(vget_low_s16(tmp12_add_tmp13), consts.val[0], 2);
  z1_h = vmull_lane_s16(vget_high_s16(tmp12_add_tmp13), consts.val[0], 2);

  int32x4_t row2_scaled_l =
    vmlal_lane_s16(z1_l, vget_low_s16(tmp13), consts.val[0], 3);
  int32x4_t row2_scaled_h =
    vmlal_lane_s16(z1_h, vget_high_s16(tmp13), consts.val[0], 3);
  row2 = vcombine_s16(vrshrn_n_s32(row2_scaled_l, DESCALE_P2),
                      vrshrn_n_s32(row2_scaled_h, DESCALE_P2));

  int32x4_t row6_scaled_l =
    vmlal_lane_s16(z1_l, vget_low_s16(tmp12), consts.val[1], 3);
  int32x4_t row6_scaled_h =
    vmlal_lane_s16(z1_h, vget_high_s16(tmp12), consts.val[1], 3);
  row6 = vcombine_s16(vrshrn_n_s32(row6_scaled_l, DESCALE_P2),
                      vrshrn_n_s32(row6_scaled_h, DESCALE_P2));

  /* Odd part */
  z1 = vaddq_s16(tmp4, tmp7);
  z2 = vaddq_s16(tmp5, tmp6);
  z3 = vaddq_s16(tmp4, tmp6);
  z4 = vaddq_s16(tmp5, tmp7);
  /* sqrt(2) * c3 */
  z5_l = vmull_lane_s16(vget_low_s16(z3), consts.val[1], 1);
  z5_h = vmull_lane_s16(vget_high_s16(z3), consts.val[1], 1);
  z5_l = vmlal_lane_s16(z5_l, vget_low_s16(z4), consts.val[1], 1);
  z5_h = vmlal_lane_s16(z5_h, vget_high_s16(z4), consts.val[1], 1);

  /* sqrt(2) * (-c1+c3+c5-c7) */
  tmp4_l = vmull_lane_s16(vget_low_s16(tmp4), consts.val[0], 0);
  tmp4_h = vmull_lane_s16(vget_high_s16(tmp4), consts.val[0], 0);
  /* sqrt(2) * ( c1+c3-c5+c7) */
  tmp5_l = vmull_lane_s16(vget_low_s16(tmp5), consts.val[2], 1);
  tmp5_h = vmull_lane_s16(vget_high_s16(tmp5), consts.val[2], 1);
  /* sqrt(2) * ( c1+c3+c5-c7) */
  tmp6_l = vmull_lane_s16(vget_low_s16(tmp6), consts.val[2], 3);
  tmp6_h = vmull_lane_s16(vget_high_s16(tmp6), consts.val[2], 3);
  /* sqrt(2) * ( c1+c3-c5-c7) */
  tmp7_l = vmull_lane_s16(vget_low_s16(tmp7), consts.val[1], 2);
  tmp7_h = vmull_lane_s16(vget_high_s16(tmp7), consts.val[1], 2);

  /* sqrt(2) * (c7-c3) */
  z1_l = vmull_lane_s16(vget_low_s16(z1), consts.val[1], 0);
  z1_h = vmull_lane_s16(vget_high_s16(z1), consts.val[1], 0);
  /* sqrt(2) * (-c1-c3) */
  z2_l = vmull_lane_s16(vget_low_s16(z2), consts.val[2], 2);
  z2_h = vmull_lane_s16(vget_high_s16(z2), consts.val[2], 2);
  /* sqrt(2) * (-c3-c5) */
  z3_l = vmull_lane_s16(vget_low_s16(z3), consts.val[2], 0);
  z3_h = vmull_lane_s16(vget_high_s16(z3), consts.val[2], 0);
  /* sqrt(2) * (c5-c3) */
  z4_l = vmull_lane_s16(vget_low_s16(z4), consts.val[0], 1);
  z4_h = vmull_lane_s16(vget_high_s16(z4), consts.val[0], 1);

  z3_l = vaddq_s32(z3_l, z5_l);
  z3_h = vaddq_s32(z3_h, z5_h);
  z4_l = vaddq_s32(z4_l, z5_l);
  z4_h = vaddq_s32(z4_h, z5_h);

  tmp4_l = vaddq_s32(tmp4_l, z1_l);
  tmp4_h = vaddq_s32(tmp4_h, z1_h);
  tmp4_l = vaddq_s32(tmp4_l, z3_l);
  tmp4_h = vaddq_s32(tmp4_h, z3_h);
  row7 = vcombine_s16(vrshrn_n_s32(tmp4_l, DESCALE_P2),
                      vrshrn_n_s32(tmp4_h, DESCALE_P2));

  tmp5_l = vaddq_s32(tmp5_l, z2_l);
  tmp5_h = vaddq_s32(tmp5_h, z2_h);
  tmp5_l = vaddq_s32(tmp5_l, z4_l);
  tmp5_h = vaddq_s32(tmp5_h, z4_h);
  row5 = vcombine_s16(vrshrn_n_s32(tmp5_l, DESCALE_P2),
                      vrshrn_n_s32(tmp5_h, DESCALE_P2));

  tmp6_l = vaddq_s32(tmp6_l, z2_l);
  tmp6_h = vaddq_s32(tmp6_h, z2_h);
  tmp6_l = vaddq_s32(tmp6_l, z3_l);
  tmp6_h = vaddq_s32(tmp6_h, z3_h);
  row3 = vcombine_s16(vrshrn_n_s32(tmp6_l, DESCALE_P2),
                      vrshrn_n_s32(tmp6_h, DESCALE_P2));

  tmp7_l = vaddq_s32(tmp7_l, z1_l);
  tmp7_h = vaddq_s32(tmp7_h, z1_h);
  tmp7_l = vaddq_s32(tmp7_l, z4_l);
  tmp7_h = vaddq_s32(tmp7_h, z4_h);
  row1 = vcombine_s16(vrshrn_n_s32(tmp7_l, DESCALE_P2),
                      vrshrn_n_s32(tmp7_h, DESCALE_P2));

  vst1q_s16(data + 0 * DCTSIZE, row0);
  vst1q_s16(data + 1 * DCTSIZE, row1);
  vst1q_s16(data + 2 * DCTSIZE, row2);
  vst1q_s16(data + 3 * DCTSIZE, row3);
  vst1q_s16(data + 4 * DCTSIZE, row4);
  vst1q_s16(data + 5 * DCTSIZE, row5);
  vst1q_s16(data + 6 * DCTSIZE, row6);
  vst1q_s16(data + 7 * DCTSIZE, row7);
}
