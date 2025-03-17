/*
 * jidctint-neon.c - accurate integer IDCT (Arm Neon)
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

#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)

/* The computation of the inverse DCT requires the use of constants known at
 * compile time.  Scaled integer constants are used to avoid floating-point
 * arithmetic:
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
 */

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

#define F_1_175_MINUS_1_961  (F_1_175 - F_1_961)
#define F_1_175_MINUS_0_390  (F_1_175 - F_0_390)
#define F_0_541_MINUS_1_847  (F_0_541 - F_1_847)
#define F_3_072_MINUS_2_562  (F_3_072 - F_2_562)
#define F_0_298_MINUS_0_899  (F_0_298 - F_0_899)
#define F_1_501_MINUS_0_899  (F_1_501 - F_0_899)
#define F_2_053_MINUS_2_562  (F_2_053 - F_2_562)
#define F_0_541_PLUS_0_765   (F_0_541 + F_0_765)


ALIGN(16) static const int16_t jsimd_idct_islow_neon_consts[] = {
  F_0_899,             F_0_541,
  F_2_562,             F_0_298_MINUS_0_899,
  F_1_501_MINUS_0_899, F_2_053_MINUS_2_562,
  F_0_541_PLUS_0_765,  F_1_175,
  F_1_175_MINUS_0_390, F_0_541_MINUS_1_847,
  F_3_072_MINUS_2_562, F_1_175_MINUS_1_961,
  0, 0, 0, 0
};


/* Forward declaration of regular and sparse IDCT helper functions */

static INLINE void jsimd_idct_islow_pass1_regular(int16x4_t row0,
                                                  int16x4_t row1,
                                                  int16x4_t row2,
                                                  int16x4_t row3,
                                                  int16x4_t row4,
                                                  int16x4_t row5,
                                                  int16x4_t row6,
                                                  int16x4_t row7,
                                                  int16x4_t quant_row0,
                                                  int16x4_t quant_row1,
                                                  int16x4_t quant_row2,
                                                  int16x4_t quant_row3,
                                                  int16x4_t quant_row4,
                                                  int16x4_t quant_row5,
                                                  int16x4_t quant_row6,
                                                  int16x4_t quant_row7,
                                                  int16_t *workspace_1,
                                                  int16_t *workspace_2);

static INLINE void jsimd_idct_islow_pass1_sparse(int16x4_t row0,
                                                 int16x4_t row1,
                                                 int16x4_t row2,
                                                 int16x4_t row3,
                                                 int16x4_t quant_row0,
                                                 int16x4_t quant_row1,
                                                 int16x4_t quant_row2,
                                                 int16x4_t quant_row3,
                                                 int16_t *workspace_1,
                                                 int16_t *workspace_2);

static INLINE void jsimd_idct_islow_pass2_regular(int16_t *workspace,
                                                  JSAMPARRAY output_buf,
                                                  JDIMENSION output_col,
                                                  unsigned buf_offset);

static INLINE void jsimd_idct_islow_pass2_sparse(int16_t *workspace,
                                                 JSAMPARRAY output_buf,
                                                 JDIMENSION output_col,
                                                 unsigned buf_offset);


/* Perform dequantization and inverse DCT on one block of coefficients.  For
 * reference, the C implementation (jpeg_idct_slow()) can be found in
 * jidctint.c.
 *
 * Optimization techniques used for fast data access:
 *
 * In each pass, the inverse DCT is computed for the left and right 4x8 halves
 * of the DCT block.  This avoids spilling due to register pressure, and the
 * increased granularity allows for an optimized calculation depending on the
 * values of the DCT coefficients.  Between passes, intermediate data is stored
 * in 4x8 workspace buffers.
 *
 * Transposing the 8x8 DCT block after each pass can be achieved by transposing
 * each of the four 4x4 quadrants and swapping quadrants 1 and 2 (refer to the
 * diagram below.)  Swapping quadrants is cheap, since the second pass can just
 * swap the workspace buffer pointers.
 *
 *      +-------+-------+                   +-------+-------+
 *      |       |       |                   |       |       |
 *      |   0   |   1   |                   |   0   |   2   |
 *      |       |       |    transpose      |       |       |
 *      +-------+-------+     ------>       +-------+-------+
 *      |       |       |                   |       |       |
 *      |   2   |   3   |                   |   1   |   3   |
 *      |       |       |                   |       |       |
 *      +-------+-------+                   +-------+-------+
 *
 * Optimization techniques used to accelerate the inverse DCT calculation:
 *
 * In a DCT coefficient block, the coefficients are increasingly likely to be 0
 * as you move diagonally from top left to bottom right.  If whole rows of
 * coefficients are 0, then the inverse DCT calculation can be simplified.  On
 * the first pass of the inverse DCT, we test for three special cases before
 * defaulting to a full "regular" inverse DCT:
 *
 * 1) Coefficients in rows 4-7 are all zero.  In this case, we perform a
 *    "sparse" simplified inverse DCT on rows 0-3.
 * 2) AC coefficients (rows 1-7) are all zero.  In this case, the inverse DCT
 *    result is equal to the dequantized DC coefficients.
 * 3) AC and DC coefficients are all zero.  In this case, the inverse DCT
 *    result is all zero.  For the left 4x8 half, this is handled identically
 *    to Case 2 above.  For the right 4x8 half, we do no work and signal that
 *    the "sparse" algorithm is required for the second pass.
 *
 * In the second pass, only a single special case is tested: whether the AC and
 * DC coefficients were all zero in the right 4x8 block during the first pass
 * (refer to Case 3 above.)  If this is the case, then a "sparse" variant of
 * the second pass is performed for both the left and right halves of the DCT
 * block.  (The transposition after the first pass means that the right 4x8
 * block during the first pass becomes rows 4-7 during the second pass.)
 */

void jsimd_idct_islow_neon(void *dct_table, JCOEFPTR coef_block,
                           JSAMPARRAY output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quantptr = dct_table;

  int16_t workspace_l[8 * DCTSIZE / 2];
  int16_t workspace_r[8 * DCTSIZE / 2];

  /* Compute IDCT first pass on left 4x8 coefficient block. */

  /* Load DCT coefficients in left 4x8 block. */
  int16x4_t row0 = vld1_s16(coef_block + 0 * DCTSIZE);
  int16x4_t row1 = vld1_s16(coef_block + 1 * DCTSIZE);
  int16x4_t row2 = vld1_s16(coef_block + 2 * DCTSIZE);
  int16x4_t row3 = vld1_s16(coef_block + 3 * DCTSIZE);
  int16x4_t row4 = vld1_s16(coef_block + 4 * DCTSIZE);
  int16x4_t row5 = vld1_s16(coef_block + 5 * DCTSIZE);
  int16x4_t row6 = vld1_s16(coef_block + 6 * DCTSIZE);
  int16x4_t row7 = vld1_s16(coef_block + 7 * DCTSIZE);

  /* Load quantization table for left 4x8 block. */
  int16x4_t quant_row0 = vld1_s16(quantptr + 0 * DCTSIZE);
  int16x4_t quant_row1 = vld1_s16(quantptr + 1 * DCTSIZE);
  int16x4_t quant_row2 = vld1_s16(quantptr + 2 * DCTSIZE);
  int16x4_t quant_row3 = vld1_s16(quantptr + 3 * DCTSIZE);
  int16x4_t quant_row4 = vld1_s16(quantptr + 4 * DCTSIZE);
  int16x4_t quant_row5 = vld1_s16(quantptr + 5 * DCTSIZE);
  int16x4_t quant_row6 = vld1_s16(quantptr + 6 * DCTSIZE);
  int16x4_t quant_row7 = vld1_s16(quantptr + 7 * DCTSIZE);

  /* Construct bitmap to test if DCT coefficients in left 4x8 block are 0. */
  int16x4_t bitmap = vorr_s16(row7, row6);
  bitmap = vorr_s16(bitmap, row5);
  bitmap = vorr_s16(bitmap, row4);
  int64_t bitmap_rows_4567 = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);

  if (bitmap_rows_4567 == 0) {
    bitmap = vorr_s16(bitmap, row3);
    bitmap = vorr_s16(bitmap, row2);
    bitmap = vorr_s16(bitmap, row1);
    int64_t left_ac_bitmap = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);

    if (left_ac_bitmap == 0) {
      int16x4_t dcval = vshl_n_s16(vmul_s16(row0, quant_row0), PASS1_BITS);
      int16x4x4_t quadrant = { { dcval, dcval, dcval, dcval } };
      /* Store 4x4 blocks to workspace, transposing in the process. */
      vst4_s16(workspace_l, quadrant);
      vst4_s16(workspace_r, quadrant);
    } else {
      jsimd_idct_islow_pass1_sparse(row0, row1, row2, row3, quant_row0,
                                    quant_row1, quant_row2, quant_row3,
                                    workspace_l, workspace_r);
    }
  } else {
    jsimd_idct_islow_pass1_regular(row0, row1, row2, row3, row4, row5,
                                   row6, row7, quant_row0, quant_row1,
                                   quant_row2, quant_row3, quant_row4,
                                   quant_row5, quant_row6, quant_row7,
                                   workspace_l, workspace_r);
  }

  /* Compute IDCT first pass on right 4x8 coefficient block. */

  /* Load DCT coefficients in right 4x8 block. */
  row0 = vld1_s16(coef_block + 0 * DCTSIZE + 4);
  row1 = vld1_s16(coef_block + 1 * DCTSIZE + 4);
  row2 = vld1_s16(coef_block + 2 * DCTSIZE + 4);
  row3 = vld1_s16(coef_block + 3 * DCTSIZE + 4);
  row4 = vld1_s16(coef_block + 4 * DCTSIZE + 4);
  row5 = vld1_s16(coef_block + 5 * DCTSIZE + 4);
  row6 = vld1_s16(coef_block + 6 * DCTSIZE + 4);
  row7 = vld1_s16(coef_block + 7 * DCTSIZE + 4);

  /* Load quantization table for right 4x8 block. */
  quant_row0 = vld1_s16(quantptr + 0 * DCTSIZE + 4);
  quant_row1 = vld1_s16(quantptr + 1 * DCTSIZE + 4);
  quant_row2 = vld1_s16(quantptr + 2 * DCTSIZE + 4);
  quant_row3 = vld1_s16(quantptr + 3 * DCTSIZE + 4);
  quant_row4 = vld1_s16(quantptr + 4 * DCTSIZE + 4);
  quant_row5 = vld1_s16(quantptr + 5 * DCTSIZE + 4);
  quant_row6 = vld1_s16(quantptr + 6 * DCTSIZE + 4);
  quant_row7 = vld1_s16(quantptr + 7 * DCTSIZE + 4);

  /* Construct bitmap to test if DCT coefficients in right 4x8 block are 0. */
  bitmap = vorr_s16(row7, row6);
  bitmap = vorr_s16(bitmap, row5);
  bitmap = vorr_s16(bitmap, row4);
  bitmap_rows_4567 = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);
  bitmap = vorr_s16(bitmap, row3);
  bitmap = vorr_s16(bitmap, row2);
  bitmap = vorr_s16(bitmap, row1);
  int64_t right_ac_bitmap = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);

  /* If this remains non-zero, a "regular" second pass will be performed. */
  int64_t right_ac_dc_bitmap = 1;

  if (right_ac_bitmap == 0) {
    bitmap = vorr_s16(bitmap, row0);
    right_ac_dc_bitmap = vget_lane_s64(vreinterpret_s64_s16(bitmap), 0);

    if (right_ac_dc_bitmap != 0) {
      int16x4_t dcval = vshl_n_s16(vmul_s16(row0, quant_row0), PASS1_BITS);
      int16x4x4_t quadrant = { { dcval, dcval, dcval, dcval } };
      /* Store 4x4 blocks to workspace, transposing in the process. */
      vst4_s16(workspace_l + 4 * DCTSIZE / 2, quadrant);
      vst4_s16(workspace_r + 4 * DCTSIZE / 2, quadrant);
    }
  } else {
    if (bitmap_rows_4567 == 0) {
      jsimd_idct_islow_pass1_sparse(row0, row1, row2, row3, quant_row0,
                                    quant_row1, quant_row2, quant_row3,
                                    workspace_l + 4 * DCTSIZE / 2,
                                    workspace_r + 4 * DCTSIZE / 2);
    } else {
      jsimd_idct_islow_pass1_regular(row0, row1, row2, row3, row4, row5,
                                     row6, row7, quant_row0, quant_row1,
                                     quant_row2, quant_row3, quant_row4,
                                     quant_row5, quant_row6, quant_row7,
                                     workspace_l + 4 * DCTSIZE / 2,
                                     workspace_r + 4 * DCTSIZE / 2);
    }
  }

  /* Second pass: compute IDCT on rows in workspace. */

  /* If all coefficients in right 4x8 block are 0, use "sparse" second pass. */
  if (right_ac_dc_bitmap == 0) {
    jsimd_idct_islow_pass2_sparse(workspace_l, output_buf, output_col, 0);
    jsimd_idct_islow_pass2_sparse(workspace_r, output_buf, output_col, 4);
  } else {
    jsimd_idct_islow_pass2_regular(workspace_l, output_buf, output_col, 0);
    jsimd_idct_islow_pass2_regular(workspace_r, output_buf, output_col, 4);
  }
}


/* Perform dequantization and the first pass of the accurate inverse DCT on a
 * 4x8 block of coefficients.  (To process the full 8x8 DCT block, this
 * function-- or some other optimized variant-- needs to be called for both the
 * left and right 4x8 blocks.)
 *
 * This "regular" version assumes that no optimization can be made to the IDCT
 * calculation, since no useful set of AC coefficients is all 0.
 *
 * The original C implementation of the accurate IDCT (jpeg_idct_slow()) can be
 * found in jidctint.c.  Algorithmic changes made here are documented inline.
 */

static INLINE void jsimd_idct_islow_pass1_regular(int16x4_t row0,
                                                  int16x4_t row1,
                                                  int16x4_t row2,
                                                  int16x4_t row3,
                                                  int16x4_t row4,
                                                  int16x4_t row5,
                                                  int16x4_t row6,
                                                  int16x4_t row7,
                                                  int16x4_t quant_row0,
                                                  int16x4_t quant_row1,
                                                  int16x4_t quant_row2,
                                                  int16x4_t quant_row3,
                                                  int16x4_t quant_row4,
                                                  int16x4_t quant_row5,
                                                  int16x4_t quant_row6,
                                                  int16x4_t quant_row7,
                                                  int16_t *workspace_1,
                                                  int16_t *workspace_2)
{
  /* Load constants for IDCT computation. */
#ifdef HAVE_VLD1_S16_X3
  const int16x4x3_t consts = vld1_s16_x3(jsimd_idct_islow_neon_consts);
#else
  const int16x4_t consts1 = vld1_s16(jsimd_idct_islow_neon_consts);
  const int16x4_t consts2 = vld1_s16(jsimd_idct_islow_neon_consts + 4);
  const int16x4_t consts3 = vld1_s16(jsimd_idct_islow_neon_consts + 8);
  const int16x4x3_t consts = { { consts1, consts2, consts3 } };
#endif

  /* Even part */
  int16x4_t z2_s16 = vmul_s16(row2, quant_row2);
  int16x4_t z3_s16 = vmul_s16(row6, quant_row6);

  int32x4_t tmp2 = vmull_lane_s16(z2_s16, consts.val[0], 1);
  int32x4_t tmp3 = vmull_lane_s16(z2_s16, consts.val[1], 2);
  tmp2 = vmlal_lane_s16(tmp2, z3_s16, consts.val[2], 1);
  tmp3 = vmlal_lane_s16(tmp3, z3_s16, consts.val[0], 1);

  z2_s16 = vmul_s16(row0, quant_row0);
  z3_s16 = vmul_s16(row4, quant_row4);

  int32x4_t tmp0 = vshll_n_s16(vadd_s16(z2_s16, z3_s16), CONST_BITS);
  int32x4_t tmp1 = vshll_n_s16(vsub_s16(z2_s16, z3_s16), CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  /* Odd part */
  int16x4_t tmp0_s16 = vmul_s16(row7, quant_row7);
  int16x4_t tmp1_s16 = vmul_s16(row5, quant_row5);
  int16x4_t tmp2_s16 = vmul_s16(row3, quant_row3);
  int16x4_t tmp3_s16 = vmul_s16(row1, quant_row1);

  z3_s16 = vadd_s16(tmp0_s16, tmp2_s16);
  int16x4_t z4_s16 = vadd_s16(tmp1_s16, tmp3_s16);

  /* Implementation as per jpeg_idct_islow() in jidctint.c:
   *   z5 = (z3 + z4) * 1.175875602;
   *   z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
   *   z3 += z5;  z4 += z5;
   *
   * This implementation:
   *   z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
   *   z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);
   */

  int32x4_t z3 = vmull_lane_s16(z3_s16, consts.val[2], 3);
  int32x4_t z4 = vmull_lane_s16(z3_s16, consts.val[1], 3);
  z3 = vmlal_lane_s16(z3, z4_s16, consts.val[1], 3);
  z4 = vmlal_lane_s16(z4, z4_s16, consts.val[2], 0);

  /* Implementation as per jpeg_idct_islow() in jidctint.c:
   *   z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
   *   tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
   *   tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
   *   z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
   *   tmp0 += z1 + z3;  tmp1 += z2 + z4;
   *   tmp2 += z2 + z3;  tmp3 += z1 + z4;
   *
   * This implementation:
   *   tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
   *   tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
   *   tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
   *   tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
   *   tmp0 += z3;  tmp1 += z4;
   *   tmp2 += z3;  tmp3 += z4;
   */

  tmp0 = vmull_lane_s16(tmp0_s16, consts.val[0], 3);
  tmp1 = vmull_lane_s16(tmp1_s16, consts.val[1], 1);
  tmp2 = vmull_lane_s16(tmp2_s16, consts.val[2], 2);
  tmp3 = vmull_lane_s16(tmp3_s16, consts.val[1], 0);

  tmp0 = vmlsl_lane_s16(tmp0, tmp3_s16, consts.val[0], 0);
  tmp1 = vmlsl_lane_s16(tmp1, tmp2_s16, consts.val[0], 2);
  tmp2 = vmlsl_lane_s16(tmp2, tmp1_s16, consts.val[0], 2);
  tmp3 = vmlsl_lane_s16(tmp3, tmp0_s16, consts.val[0], 0);

  tmp0 = vaddq_s32(tmp0, z3);
  tmp1 = vaddq_s32(tmp1, z4);
  tmp2 = vaddq_s32(tmp2, z3);
  tmp3 = vaddq_s32(tmp3, z4);

  /* Final output stage: descale and narrow to 16-bit. */
  int16x4x4_t rows_0123 = { {
    vrshrn_n_s32(vaddq_s32(tmp10, tmp3), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp13, tmp0), DESCALE_P1)
  } };
  int16x4x4_t rows_4567 = { {
    vrshrn_n_s32(vsubq_s32(tmp13, tmp0), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp10, tmp3), DESCALE_P1)
  } };

  /* Store 4x4 blocks to the intermediate workspace, ready for the second pass.
   * (VST4 transposes the blocks.  We need to operate on rows in the next
   * pass.)
   */
  vst4_s16(workspace_1, rows_0123);
  vst4_s16(workspace_2, rows_4567);
}


/* Perform dequantization and the first pass of the accurate inverse DCT on a
 * 4x8 block of coefficients.
 *
 * This "sparse" version assumes that the AC coefficients in rows 4-7 are all
 * 0.  This simplifies the IDCT calculation, accelerating overall performance.
 */

static INLINE void jsimd_idct_islow_pass1_sparse(int16x4_t row0,
                                                 int16x4_t row1,
                                                 int16x4_t row2,
                                                 int16x4_t row3,
                                                 int16x4_t quant_row0,
                                                 int16x4_t quant_row1,
                                                 int16x4_t quant_row2,
                                                 int16x4_t quant_row3,
                                                 int16_t *workspace_1,
                                                 int16_t *workspace_2)
{
  /* Load constants for IDCT computation. */
#ifdef HAVE_VLD1_S16_X3
  const int16x4x3_t consts = vld1_s16_x3(jsimd_idct_islow_neon_consts);
#else
  const int16x4_t consts1 = vld1_s16(jsimd_idct_islow_neon_consts);
  const int16x4_t consts2 = vld1_s16(jsimd_idct_islow_neon_consts + 4);
  const int16x4_t consts3 = vld1_s16(jsimd_idct_islow_neon_consts + 8);
  const int16x4x3_t consts = { { consts1, consts2, consts3 } };
#endif

  /* Even part (z3 is all 0) */
  int16x4_t z2_s16 = vmul_s16(row2, quant_row2);

  int32x4_t tmp2 = vmull_lane_s16(z2_s16, consts.val[0], 1);
  int32x4_t tmp3 = vmull_lane_s16(z2_s16, consts.val[1], 2);

  z2_s16 = vmul_s16(row0, quant_row0);
  int32x4_t tmp0 = vshll_n_s16(z2_s16, CONST_BITS);
  int32x4_t tmp1 = vshll_n_s16(z2_s16, CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  /* Odd part (tmp0 and tmp1 are both all 0) */
  int16x4_t tmp2_s16 = vmul_s16(row3, quant_row3);
  int16x4_t tmp3_s16 = vmul_s16(row1, quant_row1);

  int16x4_t z3_s16 = tmp2_s16;
  int16x4_t z4_s16 = tmp3_s16;

  int32x4_t z3 = vmull_lane_s16(z3_s16, consts.val[2], 3);
  int32x4_t z4 = vmull_lane_s16(z3_s16, consts.val[1], 3);
  z3 = vmlal_lane_s16(z3, z4_s16, consts.val[1], 3);
  z4 = vmlal_lane_s16(z4, z4_s16, consts.val[2], 0);

  tmp0 = vmlsl_lane_s16(z3, tmp3_s16, consts.val[0], 0);
  tmp1 = vmlsl_lane_s16(z4, tmp2_s16, consts.val[0], 2);
  tmp2 = vmlal_lane_s16(z3, tmp2_s16, consts.val[2], 2);
  tmp3 = vmlal_lane_s16(z4, tmp3_s16, consts.val[1], 0);

  /* Final output stage: descale and narrow to 16-bit. */
  int16x4x4_t rows_0123 = { {
    vrshrn_n_s32(vaddq_s32(tmp10, tmp3), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrn_n_s32(vaddq_s32(tmp13, tmp0), DESCALE_P1)
  } };
  int16x4x4_t rows_4567 = { {
    vrshrn_n_s32(vsubq_s32(tmp13, tmp0), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrn_n_s32(vsubq_s32(tmp10, tmp3), DESCALE_P1)
  } };

  /* Store 4x4 blocks to the intermediate workspace, ready for the second pass.
   * (VST4 transposes the blocks.  We need to operate on rows in the next
   * pass.)
   */
  vst4_s16(workspace_1, rows_0123);
  vst4_s16(workspace_2, rows_4567);
}


/* Perform the second pass of the accurate inverse DCT on a 4x8 block of
 * coefficients.  (To process the full 8x8 DCT block, this function-- or some
 * other optimized variant-- needs to be called for both the right and left 4x8
 * blocks.)
 *
 * This "regular" version assumes that no optimization can be made to the IDCT
 * calculation, since no useful set of coefficient values are all 0 after the
 * first pass.
 *
 * Again, the original C implementation of the accurate IDCT (jpeg_idct_slow())
 * can be found in jidctint.c.  Algorithmic changes made here are documented
 * inline.
 */

static INLINE void jsimd_idct_islow_pass2_regular(int16_t *workspace,
                                                  JSAMPARRAY output_buf,
                                                  JDIMENSION output_col,
                                                  unsigned buf_offset)
{
  /* Load constants for IDCT computation. */
#ifdef HAVE_VLD1_S16_X3
  const int16x4x3_t consts = vld1_s16_x3(jsimd_idct_islow_neon_consts);
#else
  const int16x4_t consts1 = vld1_s16(jsimd_idct_islow_neon_consts);
  const int16x4_t consts2 = vld1_s16(jsimd_idct_islow_neon_consts + 4);
  const int16x4_t consts3 = vld1_s16(jsimd_idct_islow_neon_consts + 8);
  const int16x4x3_t consts = { { consts1, consts2, consts3 } };
#endif

  /* Even part */
  int16x4_t z2_s16 = vld1_s16(workspace + 2 * DCTSIZE / 2);
  int16x4_t z3_s16 = vld1_s16(workspace + 6 * DCTSIZE / 2);

  int32x4_t tmp2 = vmull_lane_s16(z2_s16, consts.val[0], 1);
  int32x4_t tmp3 = vmull_lane_s16(z2_s16, consts.val[1], 2);
  tmp2 = vmlal_lane_s16(tmp2, z3_s16, consts.val[2], 1);
  tmp3 = vmlal_lane_s16(tmp3, z3_s16, consts.val[0], 1);

  z2_s16 = vld1_s16(workspace + 0 * DCTSIZE / 2);
  z3_s16 = vld1_s16(workspace + 4 * DCTSIZE / 2);

  int32x4_t tmp0 = vshll_n_s16(vadd_s16(z2_s16, z3_s16), CONST_BITS);
  int32x4_t tmp1 = vshll_n_s16(vsub_s16(z2_s16, z3_s16), CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  /* Odd part */
  int16x4_t tmp0_s16 = vld1_s16(workspace + 7 * DCTSIZE / 2);
  int16x4_t tmp1_s16 = vld1_s16(workspace + 5 * DCTSIZE / 2);
  int16x4_t tmp2_s16 = vld1_s16(workspace + 3 * DCTSIZE / 2);
  int16x4_t tmp3_s16 = vld1_s16(workspace + 1 * DCTSIZE / 2);

  z3_s16 = vadd_s16(tmp0_s16, tmp2_s16);
  int16x4_t z4_s16 = vadd_s16(tmp1_s16, tmp3_s16);

  /* Implementation as per jpeg_idct_islow() in jidctint.c:
   *   z5 = (z3 + z4) * 1.175875602;
   *   z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
   *   z3 += z5;  z4 += z5;
   *
   * This implementation:
   *   z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
   *   z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);
   */

  int32x4_t z3 = vmull_lane_s16(z3_s16, consts.val[2], 3);
  int32x4_t z4 = vmull_lane_s16(z3_s16, consts.val[1], 3);
  z3 = vmlal_lane_s16(z3, z4_s16, consts.val[1], 3);
  z4 = vmlal_lane_s16(z4, z4_s16, consts.val[2], 0);

  /* Implementation as per jpeg_idct_islow() in jidctint.c:
   *   z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
   *   tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
   *   tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
   *   z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
   *   tmp0 += z1 + z3;  tmp1 += z2 + z4;
   *   tmp2 += z2 + z3;  tmp3 += z1 + z4;
   *
   * This implementation:
   *   tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
   *   tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
   *   tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
   *   tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
   *   tmp0 += z3;  tmp1 += z4;
   *   tmp2 += z3;  tmp3 += z4;
   */

  tmp0 = vmull_lane_s16(tmp0_s16, consts.val[0], 3);
  tmp1 = vmull_lane_s16(tmp1_s16, consts.val[1], 1);
  tmp2 = vmull_lane_s16(tmp2_s16, consts.val[2], 2);
  tmp3 = vmull_lane_s16(tmp3_s16, consts.val[1], 0);

  tmp0 = vmlsl_lane_s16(tmp0, tmp3_s16, consts.val[0], 0);
  tmp1 = vmlsl_lane_s16(tmp1, tmp2_s16, consts.val[0], 2);
  tmp2 = vmlsl_lane_s16(tmp2, tmp1_s16, consts.val[0], 2);
  tmp3 = vmlsl_lane_s16(tmp3, tmp0_s16, consts.val[0], 0);

  tmp0 = vaddq_s32(tmp0, z3);
  tmp1 = vaddq_s32(tmp1, z4);
  tmp2 = vaddq_s32(tmp2, z3);
  tmp3 = vaddq_s32(tmp3, z4);

  /* Final output stage: descale and narrow to 16-bit. */
  int16x8_t cols_02_s16 = vcombine_s16(vaddhn_s32(tmp10, tmp3),
                                       vaddhn_s32(tmp12, tmp1));
  int16x8_t cols_13_s16 = vcombine_s16(vaddhn_s32(tmp11, tmp2),
                                       vaddhn_s32(tmp13, tmp0));
  int16x8_t cols_46_s16 = vcombine_s16(vsubhn_s32(tmp13, tmp0),
                                       vsubhn_s32(tmp11, tmp2));
  int16x8_t cols_57_s16 = vcombine_s16(vsubhn_s32(tmp12, tmp1),
                                       vsubhn_s32(tmp10, tmp3));
  /* Descale and narrow to 8-bit. */
  int8x8_t cols_02_s8 = vqrshrn_n_s16(cols_02_s16, DESCALE_P2 - 16);
  int8x8_t cols_13_s8 = vqrshrn_n_s16(cols_13_s16, DESCALE_P2 - 16);
  int8x8_t cols_46_s8 = vqrshrn_n_s16(cols_46_s16, DESCALE_P2 - 16);
  int8x8_t cols_57_s8 = vqrshrn_n_s16(cols_57_s16, DESCALE_P2 - 16);
  /* Clamp to range [0-255]. */
  uint8x8_t cols_02_u8 = vadd_u8(vreinterpret_u8_s8(cols_02_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_13_u8 = vadd_u8(vreinterpret_u8_s8(cols_13_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_46_u8 = vadd_u8(vreinterpret_u8_s8(cols_46_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_57_u8 = vadd_u8(vreinterpret_u8_s8(cols_57_s8),
                                 vdup_n_u8(CENTERJSAMPLE));

  /* Transpose 4x8 block and store to memory.  (Zipping adjacent columns
   * together allows us to store 16-bit elements.)
   */
  uint8x8x2_t cols_01_23 = vzip_u8(cols_02_u8, cols_13_u8);
  uint8x8x2_t cols_45_67 = vzip_u8(cols_46_u8, cols_57_u8);
  uint16x4x4_t cols_01_23_45_67 = { {
    vreinterpret_u16_u8(cols_01_23.val[0]),
    vreinterpret_u16_u8(cols_01_23.val[1]),
    vreinterpret_u16_u8(cols_45_67.val[0]),
    vreinterpret_u16_u8(cols_45_67.val[1])
  } };

  JSAMPROW outptr0 = output_buf[buf_offset + 0] + output_col;
  JSAMPROW outptr1 = output_buf[buf_offset + 1] + output_col;
  JSAMPROW outptr2 = output_buf[buf_offset + 2] + output_col;
  JSAMPROW outptr3 = output_buf[buf_offset + 3] + output_col;
  /* VST4 of 16-bit elements completes the transpose. */
  vst4_lane_u16((uint16_t *)outptr0, cols_01_23_45_67, 0);
  vst4_lane_u16((uint16_t *)outptr1, cols_01_23_45_67, 1);
  vst4_lane_u16((uint16_t *)outptr2, cols_01_23_45_67, 2);
  vst4_lane_u16((uint16_t *)outptr3, cols_01_23_45_67, 3);
}


/* Performs the second pass of the accurate inverse DCT on a 4x8 block
 * of coefficients.
 *
 * This "sparse" version assumes that the coefficient values (after the first
 * pass) in rows 4-7 are all 0.  This simplifies the IDCT calculation,
 * accelerating overall performance.
 */

static INLINE void jsimd_idct_islow_pass2_sparse(int16_t *workspace,
                                                 JSAMPARRAY output_buf,
                                                 JDIMENSION output_col,
                                                 unsigned buf_offset)
{
  /* Load constants for IDCT computation. */
#ifdef HAVE_VLD1_S16_X3
  const int16x4x3_t consts = vld1_s16_x3(jsimd_idct_islow_neon_consts);
#else
  const int16x4_t consts1 = vld1_s16(jsimd_idct_islow_neon_consts);
  const int16x4_t consts2 = vld1_s16(jsimd_idct_islow_neon_consts + 4);
  const int16x4_t consts3 = vld1_s16(jsimd_idct_islow_neon_consts + 8);
  const int16x4x3_t consts = { { consts1, consts2, consts3 } };
#endif

  /* Even part (z3 is all 0) */
  int16x4_t z2_s16 = vld1_s16(workspace + 2 * DCTSIZE / 2);

  int32x4_t tmp2 = vmull_lane_s16(z2_s16, consts.val[0], 1);
  int32x4_t tmp3 = vmull_lane_s16(z2_s16, consts.val[1], 2);

  z2_s16 = vld1_s16(workspace + 0 * DCTSIZE / 2);
  int32x4_t tmp0 = vshll_n_s16(z2_s16, CONST_BITS);
  int32x4_t tmp1 = vshll_n_s16(z2_s16, CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  /* Odd part (tmp0 and tmp1 are both all 0) */
  int16x4_t tmp2_s16 = vld1_s16(workspace + 3 * DCTSIZE / 2);
  int16x4_t tmp3_s16 = vld1_s16(workspace + 1 * DCTSIZE / 2);

  int16x4_t z3_s16 = tmp2_s16;
  int16x4_t z4_s16 = tmp3_s16;

  int32x4_t z3 = vmull_lane_s16(z3_s16, consts.val[2], 3);
  z3 = vmlal_lane_s16(z3, z4_s16, consts.val[1], 3);
  int32x4_t z4 = vmull_lane_s16(z3_s16, consts.val[1], 3);
  z4 = vmlal_lane_s16(z4, z4_s16, consts.val[2], 0);

  tmp0 = vmlsl_lane_s16(z3, tmp3_s16, consts.val[0], 0);
  tmp1 = vmlsl_lane_s16(z4, tmp2_s16, consts.val[0], 2);
  tmp2 = vmlal_lane_s16(z3, tmp2_s16, consts.val[2], 2);
  tmp3 = vmlal_lane_s16(z4, tmp3_s16, consts.val[1], 0);

  /* Final output stage: descale and narrow to 16-bit. */
  int16x8_t cols_02_s16 = vcombine_s16(vaddhn_s32(tmp10, tmp3),
                                       vaddhn_s32(tmp12, tmp1));
  int16x8_t cols_13_s16 = vcombine_s16(vaddhn_s32(tmp11, tmp2),
                                       vaddhn_s32(tmp13, tmp0));
  int16x8_t cols_46_s16 = vcombine_s16(vsubhn_s32(tmp13, tmp0),
                                       vsubhn_s32(tmp11, tmp2));
  int16x8_t cols_57_s16 = vcombine_s16(vsubhn_s32(tmp12, tmp1),
                                       vsubhn_s32(tmp10, tmp3));
  /* Descale and narrow to 8-bit. */
  int8x8_t cols_02_s8 = vqrshrn_n_s16(cols_02_s16, DESCALE_P2 - 16);
  int8x8_t cols_13_s8 = vqrshrn_n_s16(cols_13_s16, DESCALE_P2 - 16);
  int8x8_t cols_46_s8 = vqrshrn_n_s16(cols_46_s16, DESCALE_P2 - 16);
  int8x8_t cols_57_s8 = vqrshrn_n_s16(cols_57_s16, DESCALE_P2 - 16);
  /* Clamp to range [0-255]. */
  uint8x8_t cols_02_u8 = vadd_u8(vreinterpret_u8_s8(cols_02_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_13_u8 = vadd_u8(vreinterpret_u8_s8(cols_13_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_46_u8 = vadd_u8(vreinterpret_u8_s8(cols_46_s8),
                                 vdup_n_u8(CENTERJSAMPLE));
  uint8x8_t cols_57_u8 = vadd_u8(vreinterpret_u8_s8(cols_57_s8),
                                 vdup_n_u8(CENTERJSAMPLE));

  /* Transpose 4x8 block and store to memory.  (Zipping adjacent columns
   * together allows us to store 16-bit elements.)
   */
  uint8x8x2_t cols_01_23 = vzip_u8(cols_02_u8, cols_13_u8);
  uint8x8x2_t cols_45_67 = vzip_u8(cols_46_u8, cols_57_u8);
  uint16x4x4_t cols_01_23_45_67 = { {
    vreinterpret_u16_u8(cols_01_23.val[0]),
    vreinterpret_u16_u8(cols_01_23.val[1]),
    vreinterpret_u16_u8(cols_45_67.val[0]),
    vreinterpret_u16_u8(cols_45_67.val[1])
  } };

  JSAMPROW outptr0 = output_buf[buf_offset + 0] + output_col;
  JSAMPROW outptr1 = output_buf[buf_offset + 1] + output_col;
  JSAMPROW outptr2 = output_buf[buf_offset + 2] + output_col;
  JSAMPROW outptr3 = output_buf[buf_offset + 3] + output_col;
  /* VST4 of 16-bit elements completes the transpose. */
  vst4_lane_u16((uint16_t *)outptr0, cols_01_23_45_67, 0);
  vst4_lane_u16((uint16_t *)outptr1, cols_01_23_45_67, 1);
  vst4_lane_u16((uint16_t *)outptr2, cols_01_23_45_67, 2);
  vst4_lane_u16((uint16_t *)outptr3, cols_01_23_45_67, 3);
}
