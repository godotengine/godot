/*
 * Armv7 Neon optimizations for libjpeg-turbo
 *
 * Copyright (C) 2009-2011, Nokia Corporation and/or its subsidiary(-ies).
 *                          All Rights Reserved.
 * Author:  Siarhei Siamashka <siarhei.siamashka@nokia.com>
 * Copyright (C) 2014, Siarhei Siamashka.  All Rights Reserved.
 * Copyright (C) 2014, Linaro Limited.  All Rights Reserved.
 * Copyright (C) 2015, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2015-2016, 2018, Matthieu Darbois.  All Rights Reserved.
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

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack, "", %progbits  /* mark stack as non-executable */
#endif

.text
.fpu neon
.arch armv7a
.object_arch armv4
.arm
.syntax unified


/*****************************************************************************/

/* Supplementary macro for setting function attributes */
.macro asm_function fname
#ifdef __APPLE__
    .private_extern _\fname
    .globl _\fname
_\fname:
#else
    .global \fname
#ifdef __ELF__
    .hidden \fname
    .type \fname, %function
#endif
\fname:
#endif
.endm


#define CENTERJSAMPLE  128

/*****************************************************************************/

/*
 * Perform dequantization and inverse DCT on one block of coefficients.
 *
 * GLOBAL(void)
 * jsimd_idct_islow_neon(void *dct_table, JCOEFPTR coef_block,
 *                       JSAMPARRAY output_buf, JDIMENSION output_col)
 */

#define FIX_0_298631336  (2446)
#define FIX_0_390180644  (3196)
#define FIX_0_541196100  (4433)
#define FIX_0_765366865  (6270)
#define FIX_0_899976223  (7373)
#define FIX_1_175875602  (9633)
#define FIX_1_501321110  (12299)
#define FIX_1_847759065  (15137)
#define FIX_1_961570560  (16069)
#define FIX_2_053119869  (16819)
#define FIX_2_562915447  (20995)
#define FIX_3_072711026  (25172)

#define FIX_1_175875602_MINUS_1_961570560  (FIX_1_175875602 - FIX_1_961570560)
#define FIX_1_175875602_MINUS_0_390180644  (FIX_1_175875602 - FIX_0_390180644)
#define FIX_0_541196100_MINUS_1_847759065  (FIX_0_541196100 - FIX_1_847759065)
#define FIX_3_072711026_MINUS_2_562915447  (FIX_3_072711026 - FIX_2_562915447)
#define FIX_0_298631336_MINUS_0_899976223  (FIX_0_298631336 - FIX_0_899976223)
#define FIX_1_501321110_MINUS_0_899976223  (FIX_1_501321110 - FIX_0_899976223)
#define FIX_2_053119869_MINUS_2_562915447  (FIX_2_053119869 - FIX_2_562915447)
#define FIX_0_541196100_PLUS_0_765366865   (FIX_0_541196100 + FIX_0_765366865)

/*
 * Reference SIMD-friendly 1-D ISLOW iDCT C implementation.
 * Uses some ideas from the comments in 'simd/jiss2int-64.asm'
 */
#define REF_1D_IDCT(xrow0, xrow1, xrow2, xrow3, xrow4, xrow5, xrow6, xrow7) { \
  DCTELEM row0, row1, row2, row3, row4, row5, row6, row7; \
  JLONG   q1, q2, q3, q4, q5, q6, q7; \
  JLONG   tmp11_plus_tmp2, tmp11_minus_tmp2; \
  \
  /* 1-D iDCT input data */ \
  row0 = xrow0; \
  row1 = xrow1; \
  row2 = xrow2; \
  row3 = xrow3; \
  row4 = xrow4; \
  row5 = xrow5; \
  row6 = xrow6; \
  row7 = xrow7; \
  \
  q5 = row7 + row3; \
  q4 = row5 + row1; \
  q6 = MULTIPLY(q5, FIX_1_175875602_MINUS_1_961570560) + \
       MULTIPLY(q4, FIX_1_175875602); \
  q7 = MULTIPLY(q5, FIX_1_175875602) + \
       MULTIPLY(q4, FIX_1_175875602_MINUS_0_390180644); \
  q2 = MULTIPLY(row2, FIX_0_541196100) + \
       MULTIPLY(row6, FIX_0_541196100_MINUS_1_847759065); \
  q4 = q6; \
  q3 = ((JLONG)row0 - (JLONG)row4) << 13; \
  q6 += MULTIPLY(row5, -FIX_2_562915447) + \
        MULTIPLY(row3, FIX_3_072711026_MINUS_2_562915447); \
  /* now we can use q1 (reloadable constants have been used up) */ \
  q1 = q3 + q2; \
  q4 += MULTIPLY(row7, FIX_0_298631336_MINUS_0_899976223) + \
        MULTIPLY(row1, -FIX_0_899976223); \
  q5 = q7; \
  q1 = q1 + q6; \
  q7 += MULTIPLY(row7, -FIX_0_899976223) + \
        MULTIPLY(row1, FIX_1_501321110_MINUS_0_899976223); \
  \
  /* (tmp11 + tmp2) has been calculated (out_row1 before descale) */ \
  tmp11_plus_tmp2 = q1; \
  row1 = 0; \
  \
  q1 = q1 - q6; \
  q5 += MULTIPLY(row5, FIX_2_053119869_MINUS_2_562915447) + \
        MULTIPLY(row3, -FIX_2_562915447); \
  q1 = q1 - q6; \
  q6 = MULTIPLY(row2, FIX_0_541196100_PLUS_0_765366865) + \
       MULTIPLY(row6, FIX_0_541196100); \
  q3 = q3 - q2; \
  \
  /* (tmp11 - tmp2) has been calculated (out_row6 before descale) */ \
  tmp11_minus_tmp2 = q1; \
  \
  q1 = ((JLONG)row0 + (JLONG)row4) << 13; \
  q2 = q1 + q6; \
  q1 = q1 - q6; \
  \
  /* pick up the results */ \
  tmp0  = q4; \
  tmp1  = q5; \
  tmp2  = (tmp11_plus_tmp2 - tmp11_minus_tmp2) / 2; \
  tmp3  = q7; \
  tmp10 = q2; \
  tmp11 = (tmp11_plus_tmp2 + tmp11_minus_tmp2) / 2; \
  tmp12 = q3; \
  tmp13 = q1; \
}

#define XFIX_0_899976223                    d0[0]
#define XFIX_0_541196100                    d0[1]
#define XFIX_2_562915447                    d0[2]
#define XFIX_0_298631336_MINUS_0_899976223  d0[3]
#define XFIX_1_501321110_MINUS_0_899976223  d1[0]
#define XFIX_2_053119869_MINUS_2_562915447  d1[1]
#define XFIX_0_541196100_PLUS_0_765366865   d1[2]
#define XFIX_1_175875602                    d1[3]
#define XFIX_1_175875602_MINUS_0_390180644  d2[0]
#define XFIX_0_541196100_MINUS_1_847759065  d2[1]
#define XFIX_3_072711026_MINUS_2_562915447  d2[2]
#define XFIX_1_175875602_MINUS_1_961570560  d2[3]

.balign 16
jsimd_idct_islow_neon_consts:
  .short FIX_0_899976223                    /* d0[0] */
  .short FIX_0_541196100                    /* d0[1] */
  .short FIX_2_562915447                    /* d0[2] */
  .short FIX_0_298631336_MINUS_0_899976223  /* d0[3] */
  .short FIX_1_501321110_MINUS_0_899976223  /* d1[0] */
  .short FIX_2_053119869_MINUS_2_562915447  /* d1[1] */
  .short FIX_0_541196100_PLUS_0_765366865   /* d1[2] */
  .short FIX_1_175875602                    /* d1[3] */
  /* reloadable constants */
  .short FIX_1_175875602_MINUS_0_390180644  /* d2[0] */
  .short FIX_0_541196100_MINUS_1_847759065  /* d2[1] */
  .short FIX_3_072711026_MINUS_2_562915447  /* d2[2] */
  .short FIX_1_175875602_MINUS_1_961570560  /* d2[3] */

asm_function jsimd_idct_islow_neon

    DCT_TABLE       .req r0
    COEF_BLOCK      .req r1
    OUTPUT_BUF      .req r2
    OUTPUT_COL      .req r3
    TMP1            .req r0
    TMP2            .req r1
    TMP3            .req r2
    TMP4            .req ip

    ROW0L           .req d16
    ROW0R           .req d17
    ROW1L           .req d18
    ROW1R           .req d19
    ROW2L           .req d20
    ROW2R           .req d21
    ROW3L           .req d22
    ROW3R           .req d23
    ROW4L           .req d24
    ROW4R           .req d25
    ROW5L           .req d26
    ROW5R           .req d27
    ROW6L           .req d28
    ROW6R           .req d29
    ROW7L           .req d30
    ROW7R           .req d31

    /* Load and dequantize coefficients into Neon registers
     * with the following allocation:
     *       0 1 2 3 | 4 5 6 7
     *      ---------+--------
     *   0 | d16     | d17     ( q8  )
     *   1 | d18     | d19     ( q9  )
     *   2 | d20     | d21     ( q10 )
     *   3 | d22     | d23     ( q11 )
     *   4 | d24     | d25     ( q12 )
     *   5 | d26     | d27     ( q13 )
     *   6 | d28     | d29     ( q14 )
     *   7 | d30     | d31     ( q15 )
     */
    adr             ip, jsimd_idct_islow_neon_consts
    vld1.16         {d16, d17, d18, d19}, [COEF_BLOCK, :128]!
    vld1.16         {d0, d1, d2, d3}, [DCT_TABLE, :128]!
    vld1.16         {d20, d21, d22, d23}, [COEF_BLOCK, :128]!
    vmul.s16        q8, q8, q0
    vld1.16         {d4, d5, d6, d7}, [DCT_TABLE, :128]!
    vmul.s16        q9, q9, q1
    vld1.16         {d24, d25, d26, d27}, [COEF_BLOCK, :128]!
    vmul.s16        q10, q10, q2
    vld1.16         {d0, d1, d2, d3}, [DCT_TABLE, :128]!
    vmul.s16        q11, q11, q3
    vld1.16         {d28, d29, d30, d31}, [COEF_BLOCK, :128]
    vmul.s16        q12, q12, q0
    vld1.16         {d4, d5, d6, d7}, [DCT_TABLE, :128]!
    vmul.s16        q14, q14, q2
    vmul.s16        q13, q13, q1
    vld1.16         {d0, d1, d2, d3}, [ip, :128]  /* load constants */
    add             ip, ip, #16
    vmul.s16        q15, q15, q3
    vpush           {d8 - d15}                    /* save Neon registers */
    /* 1-D IDCT, pass 1, left 4x8 half */
    vadd.s16        d4, ROW7L, ROW3L
    vadd.s16        d5, ROW5L, ROW1L
    vmull.s16       q6, d4, XFIX_1_175875602_MINUS_1_961570560
    vmlal.s16       q6, d5, XFIX_1_175875602
    vmull.s16       q7, d4, XFIX_1_175875602
      /* Check for the zero coefficients in the right 4x8 half */
      push            {r4, r5}
    vmlal.s16       q7, d5, XFIX_1_175875602_MINUS_0_390180644
    vsubl.s16       q3, ROW0L, ROW4L
      ldrd            r4, [COEF_BLOCK, #(-96 + 2 * (4 + 1 * 8))]
    vmull.s16       q2, ROW2L, XFIX_0_541196100
    vmlal.s16       q2, ROW6L, XFIX_0_541196100_MINUS_1_847759065
      orr             r0, r4, r5
    vmov            q4, q6
    vmlsl.s16       q6, ROW5L, XFIX_2_562915447
      ldrd            r4, [COEF_BLOCK, #(-96 + 2 * (4 + 2 * 8))]
    vmlal.s16       q6, ROW3L, XFIX_3_072711026_MINUS_2_562915447
    vshl.s32        q3, q3, #13
      orr             r0, r0, r4
    vmlsl.s16       q4, ROW1L, XFIX_0_899976223
      orr             r0, r0, r5
    vadd.s32        q1, q3, q2
      ldrd            r4, [COEF_BLOCK, #(-96 + 2 * (4 + 3 * 8))]
    vmov            q5, q7
    vadd.s32        q1, q1, q6
      orr             r0, r0, r4
    vmlsl.s16       q7, ROW7L, XFIX_0_899976223
      orr             r0, r0, r5
    vmlal.s16       q7, ROW1L, XFIX_1_501321110_MINUS_0_899976223
    vrshrn.s32      ROW1L, q1, #11
      ldrd            r4, [COEF_BLOCK, #(-96 + 2 * (4 + 4 * 8))]
    vsub.s32        q1, q1, q6
    vmlal.s16       q5, ROW5L, XFIX_2_053119869_MINUS_2_562915447
      orr             r0, r0, r4
    vmlsl.s16       q5, ROW3L, XFIX_2_562915447
      orr             r0, r0, r5
    vsub.s32        q1, q1, q6
    vmull.s16       q6, ROW2L, XFIX_0_541196100_PLUS_0_765366865
      ldrd            r4, [COEF_BLOCK, #(-96 + 2 * (4 + 5 * 8))]
    vmlal.s16       q6, ROW6L, XFIX_0_541196100
    vsub.s32        q3, q3, q2
      orr             r0, r0, r4
    vrshrn.s32      ROW6L, q1, #11
      orr             r0, r0, r5
    vadd.s32        q1, q3, q5
      ldrd            r4, [COEF_BLOCK, #(-96 + 2 * (4 + 6 * 8))]
    vsub.s32        q3, q3, q5
    vaddl.s16       q5, ROW0L, ROW4L
      orr             r0, r0, r4
    vrshrn.s32      ROW2L, q1, #11
      orr             r0, r0, r5
    vrshrn.s32      ROW5L, q3, #11
      ldrd            r4, [COEF_BLOCK, #(-96 + 2 * (4 + 7 * 8))]
    vshl.s32        q5, q5, #13
    vmlal.s16       q4, ROW7L, XFIX_0_298631336_MINUS_0_899976223
      orr             r0, r0, r4
    vadd.s32        q2, q5, q6
      orrs            r0, r0, r5
    vsub.s32        q1, q5, q6
    vadd.s32        q6, q2, q7
      ldrd            r4, [COEF_BLOCK, #(-96 + 2 * (4 + 0 * 8))]
    vsub.s32        q2, q2, q7
    vadd.s32        q5, q1, q4
      orr             r0, r4, r5
    vsub.s32        q3, q1, q4
      pop             {r4, r5}
    vrshrn.s32      ROW7L, q2, #11
    vrshrn.s32      ROW3L, q5, #11
    vrshrn.s32      ROW0L, q6, #11
    vrshrn.s32      ROW4L, q3, #11

      beq             3f  /* Go to do some special handling for the sparse
                             right 4x8 half */

    /* 1-D IDCT, pass 1, right 4x8 half */
    vld1.s16        {d2}, [ip, :64]  /* reload constants */
    vadd.s16        d10, ROW7R, ROW3R
    vadd.s16        d8, ROW5R, ROW1R
      /* Transpose left 4x8 half */
      vtrn.16         ROW6L, ROW7L
    vmull.s16       q6, d10, XFIX_1_175875602_MINUS_1_961570560
    vmlal.s16       q6, d8, XFIX_1_175875602
      vtrn.16         ROW2L, ROW3L
    vmull.s16       q7, d10, XFIX_1_175875602
    vmlal.s16       q7, d8, XFIX_1_175875602_MINUS_0_390180644
      vtrn.16         ROW0L, ROW1L
    vsubl.s16       q3, ROW0R, ROW4R
    vmull.s16       q2, ROW2R, XFIX_0_541196100
    vmlal.s16       q2, ROW6R, XFIX_0_541196100_MINUS_1_847759065
      vtrn.16         ROW4L, ROW5L
    vmov            q4, q6
    vmlsl.s16       q6, ROW5R, XFIX_2_562915447
    vmlal.s16       q6, ROW3R, XFIX_3_072711026_MINUS_2_562915447
      vtrn.32         ROW1L, ROW3L
    vshl.s32        q3, q3, #13
    vmlsl.s16       q4, ROW1R, XFIX_0_899976223
      vtrn.32         ROW4L, ROW6L
    vadd.s32        q1, q3, q2
    vmov            q5, q7
    vadd.s32        q1, q1, q6
      vtrn.32         ROW0L, ROW2L
    vmlsl.s16       q7, ROW7R, XFIX_0_899976223
    vmlal.s16       q7, ROW1R, XFIX_1_501321110_MINUS_0_899976223
    vrshrn.s32      ROW1R, q1, #11
      vtrn.32         ROW5L, ROW7L
    vsub.s32        q1, q1, q6
    vmlal.s16       q5, ROW5R, XFIX_2_053119869_MINUS_2_562915447
    vmlsl.s16       q5, ROW3R, XFIX_2_562915447
    vsub.s32        q1, q1, q6
    vmull.s16       q6, ROW2R, XFIX_0_541196100_PLUS_0_765366865
    vmlal.s16       q6, ROW6R, XFIX_0_541196100
    vsub.s32        q3, q3, q2
    vrshrn.s32      ROW6R, q1, #11
    vadd.s32        q1, q3, q5
    vsub.s32        q3, q3, q5
    vaddl.s16       q5, ROW0R, ROW4R
    vrshrn.s32      ROW2R, q1, #11
    vrshrn.s32      ROW5R, q3, #11
    vshl.s32        q5, q5, #13
    vmlal.s16       q4, ROW7R, XFIX_0_298631336_MINUS_0_899976223
    vadd.s32        q2, q5, q6
    vsub.s32        q1, q5, q6
    vadd.s32        q6, q2, q7
    vsub.s32        q2, q2, q7
    vadd.s32        q5, q1, q4
    vsub.s32        q3, q1, q4
    vrshrn.s32      ROW7R, q2, #11
    vrshrn.s32      ROW3R, q5, #11
    vrshrn.s32      ROW0R, q6, #11
    vrshrn.s32      ROW4R, q3, #11
    /* Transpose right 4x8 half */
    vtrn.16         ROW6R, ROW7R
    vtrn.16         ROW2R, ROW3R
    vtrn.16         ROW0R, ROW1R
    vtrn.16         ROW4R, ROW5R
    vtrn.32         ROW1R, ROW3R
    vtrn.32         ROW4R, ROW6R
    vtrn.32         ROW0R, ROW2R
    vtrn.32         ROW5R, ROW7R

1:  /* 1-D IDCT, pass 2 (normal variant), left 4x8 half */
    vld1.s16        {d2}, [ip, :64]               /* reload constants */
    vmull.s16       q6, ROW1R, XFIX_1_175875602   /* ROW5L <-> ROW1R */
    vmlal.s16       q6, ROW1L, XFIX_1_175875602
    vmlal.s16       q6, ROW3R, XFIX_1_175875602_MINUS_1_961570560  /* ROW7L <-> ROW3R */
    vmlal.s16       q6, ROW3L, XFIX_1_175875602_MINUS_1_961570560
    vmull.s16       q7, ROW3R, XFIX_1_175875602   /* ROW7L <-> ROW3R */
    vmlal.s16       q7, ROW3L, XFIX_1_175875602
    vmlal.s16       q7, ROW1R, XFIX_1_175875602_MINUS_0_390180644  /* ROW5L <-> ROW1R */
    vmlal.s16       q7, ROW1L, XFIX_1_175875602_MINUS_0_390180644
    vsubl.s16       q3, ROW0L, ROW0R              /* ROW4L <-> ROW0R */
    vmull.s16       q2, ROW2L, XFIX_0_541196100
    vmlal.s16       q2, ROW2R, XFIX_0_541196100_MINUS_1_847759065  /* ROW6L <-> ROW2R */
    vmov            q4, q6
    vmlsl.s16       q6, ROW1R, XFIX_2_562915447   /* ROW5L <-> ROW1R */
    vmlal.s16       q6, ROW3L, XFIX_3_072711026_MINUS_2_562915447
    vshl.s32        q3, q3, #13
    vmlsl.s16       q4, ROW1L, XFIX_0_899976223
    vadd.s32        q1, q3, q2
    vmov            q5, q7
    vadd.s32        q1, q1, q6
    vmlsl.s16       q7, ROW3R, XFIX_0_899976223   /* ROW7L <-> ROW3R */
    vmlal.s16       q7, ROW1L, XFIX_1_501321110_MINUS_0_899976223
    vshrn.s32       ROW1L, q1, #16
    vsub.s32        q1, q1, q6
    vmlal.s16       q5, ROW1R, XFIX_2_053119869_MINUS_2_562915447  /* ROW5L <-> ROW1R */
    vmlsl.s16       q5, ROW3L, XFIX_2_562915447
    vsub.s32        q1, q1, q6
    vmull.s16       q6, ROW2L, XFIX_0_541196100_PLUS_0_765366865
    vmlal.s16       q6, ROW2R, XFIX_0_541196100   /* ROW6L <-> ROW2R */
    vsub.s32        q3, q3, q2
    vshrn.s32       ROW2R, q1, #16                /* ROW6L <-> ROW2R */
    vadd.s32        q1, q3, q5
    vsub.s32        q3, q3, q5
    vaddl.s16       q5, ROW0L, ROW0R              /* ROW4L <-> ROW0R */
    vshrn.s32       ROW2L, q1, #16
    vshrn.s32       ROW1R, q3, #16                /* ROW5L <-> ROW1R */
    vshl.s32        q5, q5, #13
    vmlal.s16       q4, ROW3R, XFIX_0_298631336_MINUS_0_899976223  /* ROW7L <-> ROW3R */
    vadd.s32        q2, q5, q6
    vsub.s32        q1, q5, q6
    vadd.s32        q6, q2, q7
    vsub.s32        q2, q2, q7
    vadd.s32        q5, q1, q4
    vsub.s32        q3, q1, q4
    vshrn.s32       ROW3R, q2, #16                /* ROW7L <-> ROW3R */
    vshrn.s32       ROW3L, q5, #16
    vshrn.s32       ROW0L, q6, #16
    vshrn.s32       ROW0R, q3, #16                /* ROW4L <-> ROW0R */
    /* 1-D IDCT, pass 2, right 4x8 half */
    vld1.s16        {d2}, [ip, :64]               /* reload constants */
    vmull.s16       q6, ROW5R, XFIX_1_175875602
    vmlal.s16       q6, ROW5L, XFIX_1_175875602   /* ROW5L <-> ROW1R */
    vmlal.s16       q6, ROW7R, XFIX_1_175875602_MINUS_1_961570560
    vmlal.s16       q6, ROW7L, XFIX_1_175875602_MINUS_1_961570560  /* ROW7L <-> ROW3R */
    vmull.s16       q7, ROW7R, XFIX_1_175875602
    vmlal.s16       q7, ROW7L, XFIX_1_175875602   /* ROW7L <-> ROW3R */
    vmlal.s16       q7, ROW5R, XFIX_1_175875602_MINUS_0_390180644
    vmlal.s16       q7, ROW5L, XFIX_1_175875602_MINUS_0_390180644  /* ROW5L <-> ROW1R */
    vsubl.s16       q3, ROW4L, ROW4R              /* ROW4L <-> ROW0R */
    vmull.s16       q2, ROW6L, XFIX_0_541196100   /* ROW6L <-> ROW2R */
    vmlal.s16       q2, ROW6R, XFIX_0_541196100_MINUS_1_847759065
    vmov            q4, q6
    vmlsl.s16       q6, ROW5R, XFIX_2_562915447
    vmlal.s16       q6, ROW7L, XFIX_3_072711026_MINUS_2_562915447  /* ROW7L <-> ROW3R */
    vshl.s32        q3, q3, #13
    vmlsl.s16       q4, ROW5L, XFIX_0_899976223   /* ROW5L <-> ROW1R */
    vadd.s32        q1, q3, q2
    vmov            q5, q7
    vadd.s32        q1, q1, q6
    vmlsl.s16       q7, ROW7R, XFIX_0_899976223
    vmlal.s16       q7, ROW5L, XFIX_1_501321110_MINUS_0_899976223  /* ROW5L <-> ROW1R */
    vshrn.s32       ROW5L, q1, #16                /* ROW5L <-> ROW1R */
    vsub.s32        q1, q1, q6
    vmlal.s16       q5, ROW5R, XFIX_2_053119869_MINUS_2_562915447
    vmlsl.s16       q5, ROW7L, XFIX_2_562915447   /* ROW7L <-> ROW3R */
    vsub.s32        q1, q1, q6
    vmull.s16       q6, ROW6L, XFIX_0_541196100_PLUS_0_765366865  /* ROW6L <-> ROW2R */
    vmlal.s16       q6, ROW6R, XFIX_0_541196100
    vsub.s32        q3, q3, q2
    vshrn.s32       ROW6R, q1, #16
    vadd.s32        q1, q3, q5
    vsub.s32        q3, q3, q5
    vaddl.s16       q5, ROW4L, ROW4R              /* ROW4L <-> ROW0R */
    vshrn.s32       ROW6L, q1, #16                /* ROW6L <-> ROW2R */
    vshrn.s32       ROW5R, q3, #16
    vshl.s32        q5, q5, #13
    vmlal.s16       q4, ROW7R, XFIX_0_298631336_MINUS_0_899976223
    vadd.s32        q2, q5, q6
    vsub.s32        q1, q5, q6
    vadd.s32        q6, q2, q7
    vsub.s32        q2, q2, q7
    vadd.s32        q5, q1, q4
    vsub.s32        q3, q1, q4
    vshrn.s32       ROW7R, q2, #16
    vshrn.s32       ROW7L, q5, #16                /* ROW7L <-> ROW3R */
    vshrn.s32       ROW4L, q6, #16                /* ROW4L <-> ROW0R */
    vshrn.s32       ROW4R, q3, #16

2:  /* Descale to 8-bit and range limit */
    vqrshrn.s16     d16, q8, #2
    vqrshrn.s16     d17, q9, #2
    vqrshrn.s16     d18, q10, #2
    vqrshrn.s16     d19, q11, #2
    vpop            {d8 - d15}                    /* restore Neon registers */
    vqrshrn.s16     d20, q12, #2
      /* Transpose the final 8-bit samples and do signed->unsigned conversion */
      vtrn.16         q8, q9
    vqrshrn.s16     d21, q13, #2
    vqrshrn.s16     d22, q14, #2
      vmov.u8         q0, #(CENTERJSAMPLE)
    vqrshrn.s16     d23, q15, #2
      vtrn.8          d16, d17
      vtrn.8          d18, d19
      vadd.u8         q8, q8, q0
      vadd.u8         q9, q9, q0
      vtrn.16         q10, q11
        /* Store results to the output buffer */
        ldmia           OUTPUT_BUF!, {TMP1, TMP2}
        add             TMP1, TMP1, OUTPUT_COL
        add             TMP2, TMP2, OUTPUT_COL
        vst1.8          {d16}, [TMP1]
      vtrn.8          d20, d21
        vst1.8          {d17}, [TMP2]
        ldmia           OUTPUT_BUF!, {TMP1, TMP2}
        add             TMP1, TMP1, OUTPUT_COL
        add             TMP2, TMP2, OUTPUT_COL
        vst1.8          {d18}, [TMP1]
      vadd.u8         q10, q10, q0
        vst1.8          {d19}, [TMP2]
        ldmia           OUTPUT_BUF, {TMP1, TMP2, TMP3, TMP4}
        add             TMP1, TMP1, OUTPUT_COL
        add             TMP2, TMP2, OUTPUT_COL
        add             TMP3, TMP3, OUTPUT_COL
        add             TMP4, TMP4, OUTPUT_COL
      vtrn.8          d22, d23
        vst1.8          {d20}, [TMP1]
      vadd.u8         q11, q11, q0
        vst1.8          {d21}, [TMP2]
        vst1.8          {d22}, [TMP3]
        vst1.8          {d23}, [TMP4]
    bx              lr

3:  /* Left 4x8 half is done, right 4x8 half contains mostly zeros */

    /* Transpose left 4x8 half */
    vtrn.16         ROW6L, ROW7L
    vtrn.16         ROW2L, ROW3L
    vtrn.16         ROW0L, ROW1L
    vtrn.16         ROW4L, ROW5L
    vshl.s16        ROW0R, ROW0R, #2  /* PASS1_BITS */
    vtrn.32         ROW1L, ROW3L
    vtrn.32         ROW4L, ROW6L
    vtrn.32         ROW0L, ROW2L
    vtrn.32         ROW5L, ROW7L

    cmp             r0, #0
    beq             4f  /* Right 4x8 half has all zeros, go to 'sparse' second
                           pass */

    /* Only row 0 is non-zero for the right 4x8 half  */
    vdup.s16        ROW1R, ROW0R[1]
    vdup.s16        ROW2R, ROW0R[2]
    vdup.s16        ROW3R, ROW0R[3]
    vdup.s16        ROW4R, ROW0R[0]
    vdup.s16        ROW5R, ROW0R[1]
    vdup.s16        ROW6R, ROW0R[2]
    vdup.s16        ROW7R, ROW0R[3]
    vdup.s16        ROW0R, ROW0R[0]
    b               1b  /* Go to 'normal' second pass */

4:  /* 1-D IDCT, pass 2 (sparse variant with zero rows 4-7), left 4x8 half */
    vld1.s16        {d2}, [ip, :64]               /* reload constants */
    vmull.s16       q6, ROW1L, XFIX_1_175875602
    vmlal.s16       q6, ROW3L, XFIX_1_175875602_MINUS_1_961570560
    vmull.s16       q7, ROW3L, XFIX_1_175875602
    vmlal.s16       q7, ROW1L, XFIX_1_175875602_MINUS_0_390180644
    vmull.s16       q2, ROW2L, XFIX_0_541196100
    vshll.s16       q3, ROW0L, #13
    vmov            q4, q6
    vmlal.s16       q6, ROW3L, XFIX_3_072711026_MINUS_2_562915447
    vmlsl.s16       q4, ROW1L, XFIX_0_899976223
    vadd.s32        q1, q3, q2
    vmov            q5, q7
    vmlal.s16       q7, ROW1L, XFIX_1_501321110_MINUS_0_899976223
    vadd.s32        q1, q1, q6
    vadd.s32        q6, q6, q6
    vmlsl.s16       q5, ROW3L, XFIX_2_562915447
    vshrn.s32       ROW1L, q1, #16
    vsub.s32        q1, q1, q6
    vmull.s16       q6, ROW2L, XFIX_0_541196100_PLUS_0_765366865
    vsub.s32        q3, q3, q2
    vshrn.s32       ROW2R, q1, #16                /* ROW6L <-> ROW2R */
    vadd.s32        q1, q3, q5
    vsub.s32        q3, q3, q5
    vshll.s16       q5, ROW0L, #13
    vshrn.s32       ROW2L, q1, #16
    vshrn.s32       ROW1R, q3, #16                /* ROW5L <-> ROW1R */
    vadd.s32        q2, q5, q6
    vsub.s32        q1, q5, q6
    vadd.s32        q6, q2, q7
    vsub.s32        q2, q2, q7
    vadd.s32        q5, q1, q4
    vsub.s32        q3, q1, q4
    vshrn.s32       ROW3R, q2, #16                /* ROW7L <-> ROW3R */
    vshrn.s32       ROW3L, q5, #16
    vshrn.s32       ROW0L, q6, #16
    vshrn.s32       ROW0R, q3, #16                /* ROW4L <-> ROW0R */
    /* 1-D IDCT, pass 2 (sparse variant with zero rows 4-7), right 4x8 half */
    vld1.s16        {d2}, [ip, :64]               /* reload constants */
    vmull.s16       q6, ROW5L, XFIX_1_175875602
    vmlal.s16       q6, ROW7L, XFIX_1_175875602_MINUS_1_961570560
    vmull.s16       q7, ROW7L, XFIX_1_175875602
    vmlal.s16       q7, ROW5L, XFIX_1_175875602_MINUS_0_390180644
    vmull.s16       q2, ROW6L, XFIX_0_541196100
    vshll.s16       q3, ROW4L, #13
    vmov            q4, q6
    vmlal.s16       q6, ROW7L, XFIX_3_072711026_MINUS_2_562915447
    vmlsl.s16       q4, ROW5L, XFIX_0_899976223
    vadd.s32        q1, q3, q2
    vmov            q5, q7
    vmlal.s16       q7, ROW5L, XFIX_1_501321110_MINUS_0_899976223
    vadd.s32        q1, q1, q6
    vadd.s32        q6, q6, q6
    vmlsl.s16       q5, ROW7L, XFIX_2_562915447
    vshrn.s32       ROW5L, q1, #16                /* ROW5L <-> ROW1R */
    vsub.s32        q1, q1, q6
    vmull.s16       q6, ROW6L, XFIX_0_541196100_PLUS_0_765366865
    vsub.s32        q3, q3, q2
    vshrn.s32       ROW6R, q1, #16
    vadd.s32        q1, q3, q5
    vsub.s32        q3, q3, q5
    vshll.s16       q5, ROW4L, #13
    vshrn.s32       ROW6L, q1, #16                /* ROW6L <-> ROW2R */
    vshrn.s32       ROW5R, q3, #16
    vadd.s32        q2, q5, q6
    vsub.s32        q1, q5, q6
    vadd.s32        q6, q2, q7
    vsub.s32        q2, q2, q7
    vadd.s32        q5, q1, q4
    vsub.s32        q3, q1, q4
    vshrn.s32       ROW7R, q2, #16
    vshrn.s32       ROW7L, q5, #16                /* ROW7L <-> ROW3R */
    vshrn.s32       ROW4L, q6, #16                /* ROW4L <-> ROW0R */
    vshrn.s32       ROW4R, q3, #16
    b               2b                            /* Go to epilogue */

    .unreq          DCT_TABLE
    .unreq          COEF_BLOCK
    .unreq          OUTPUT_BUF
    .unreq          OUTPUT_COL
    .unreq          TMP1
    .unreq          TMP2
    .unreq          TMP3
    .unreq          TMP4

    .unreq          ROW0L
    .unreq          ROW0R
    .unreq          ROW1L
    .unreq          ROW1R
    .unreq          ROW2L
    .unreq          ROW2R
    .unreq          ROW3L
    .unreq          ROW3R
    .unreq          ROW4L
    .unreq          ROW4R
    .unreq          ROW5L
    .unreq          ROW5R
    .unreq          ROW6L
    .unreq          ROW6R
    .unreq          ROW7L
    .unreq          ROW7R


/*****************************************************************************/

/*
 * jsimd_idct_ifast_neon
 *
 * This function contains a fast, not so accurate integer implementation of
 * the inverse DCT (Discrete Cosine Transform). It uses the same calculations
 * and produces exactly the same output as IJG's original 'jpeg_idct_ifast'
 * function from jidctfst.c
 *
 * Normally 1-D AAN DCT needs 5 multiplications and 29 additions.
 * But in Arm Neon case some extra additions are required because VQDMULH
 * instruction can't handle the constants larger than 1. So the expressions
 * like "x * 1.082392200" have to be converted to "x * 0.082392200 + x",
 * which introduces an extra addition. Overall, there are 6 extra additions
 * per 1-D IDCT pass, totalling to 5 VQDMULH and 35 VADD/VSUB instructions.
 */

#define XFIX_1_082392200  d0[0]
#define XFIX_1_414213562  d0[1]
#define XFIX_1_847759065  d0[2]
#define XFIX_2_613125930  d0[3]

.balign 16
jsimd_idct_ifast_neon_consts:
  .short (277 * 128 - 256 * 128)  /* XFIX_1_082392200 */
  .short (362 * 128 - 256 * 128)  /* XFIX_1_414213562 */
  .short (473 * 128 - 256 * 128)  /* XFIX_1_847759065 */
  .short (669 * 128 - 512 * 128)  /* XFIX_2_613125930 */

asm_function jsimd_idct_ifast_neon

    DCT_TABLE       .req r0
    COEF_BLOCK      .req r1
    OUTPUT_BUF      .req r2
    OUTPUT_COL      .req r3
    TMP1            .req r0
    TMP2            .req r1
    TMP3            .req r2
    TMP4            .req ip

    /* Load and dequantize coefficients into Neon registers
     * with the following allocation:
     *       0 1 2 3 | 4 5 6 7
     *      ---------+--------
     *   0 | d16     | d17     ( q8  )
     *   1 | d18     | d19     ( q9  )
     *   2 | d20     | d21     ( q10 )
     *   3 | d22     | d23     ( q11 )
     *   4 | d24     | d25     ( q12 )
     *   5 | d26     | d27     ( q13 )
     *   6 | d28     | d29     ( q14 )
     *   7 | d30     | d31     ( q15 )
     */
    adr             ip, jsimd_idct_ifast_neon_consts
    vld1.16         {d16, d17, d18, d19}, [COEF_BLOCK, :128]!
    vld1.16         {d0, d1, d2, d3}, [DCT_TABLE, :128]!
    vld1.16         {d20, d21, d22, d23}, [COEF_BLOCK, :128]!
    vmul.s16        q8, q8, q0
    vld1.16         {d4, d5, d6, d7}, [DCT_TABLE, :128]!
    vmul.s16        q9, q9, q1
    vld1.16         {d24, d25, d26, d27}, [COEF_BLOCK, :128]!
    vmul.s16        q10, q10, q2
    vld1.16         {d0, d1, d2, d3}, [DCT_TABLE, :128]!
    vmul.s16        q11, q11, q3
    vld1.16         {d28, d29, d30, d31}, [COEF_BLOCK, :128]
    vmul.s16        q12, q12, q0
    vld1.16         {d4, d5, d6, d7}, [DCT_TABLE, :128]!
    vmul.s16        q14, q14, q2
    vmul.s16        q13, q13, q1
    vld1.16         {d0}, [ip, :64]  /* load constants */
    vmul.s16        q15, q15, q3
    vpush           {d8 - d13}       /* save Neon registers */
    /* 1-D IDCT, pass 1 */
    vsub.s16        q2, q10, q14
    vadd.s16        q14, q10, q14
    vsub.s16        q1, q11, q13
    vadd.s16        q13, q11, q13
    vsub.s16        q5, q9, q15
    vadd.s16        q15, q9, q15
    vqdmulh.s16     q4, q2, XFIX_1_414213562
    vqdmulh.s16     q6, q1, XFIX_2_613125930
    vadd.s16        q3, q1, q1
    vsub.s16        q1, q5, q1
    vadd.s16        q10, q2, q4
    vqdmulh.s16     q4, q1, XFIX_1_847759065
    vsub.s16        q2, q15, q13
    vadd.s16        q3, q3, q6
    vqdmulh.s16     q6, q2, XFIX_1_414213562
    vadd.s16        q1, q1, q4
    vqdmulh.s16     q4, q5, XFIX_1_082392200
    vsub.s16        q10, q10, q14
    vadd.s16        q2, q2, q6
    vsub.s16        q6, q8, q12
    vadd.s16        q12, q8, q12
    vadd.s16        q9, q5, q4
    vadd.s16        q5, q6, q10
    vsub.s16        q10, q6, q10
    vadd.s16        q6, q15, q13
    vadd.s16        q8, q12, q14
    vsub.s16        q3, q6, q3
    vsub.s16        q12, q12, q14
    vsub.s16        q3, q3, q1
    vsub.s16        q1, q9, q1
    vadd.s16        q2, q3, q2
    vsub.s16        q15, q8, q6
    vadd.s16        q1, q1, q2
    vadd.s16        q8, q8, q6
    vadd.s16        q14, q5, q3
    vsub.s16        q9, q5, q3
    vsub.s16        q13, q10, q2
    vadd.s16        q10, q10, q2
      /* Transpose */
      vtrn.16         q8, q9
    vsub.s16        q11, q12, q1
      vtrn.16         q14, q15
    vadd.s16        q12, q12, q1
      vtrn.16         q10, q11
      vtrn.16         q12, q13
      vtrn.32         q9, q11
      vtrn.32         q12, q14
      vtrn.32         q8, q10
      vtrn.32         q13, q15
      vswp            d28, d21
      vswp            d26, d19
    /* 1-D IDCT, pass 2 */
    vsub.s16        q2, q10, q14
      vswp            d30, d23
    vadd.s16        q14, q10, q14
      vswp            d24, d17
    vsub.s16        q1, q11, q13
    vadd.s16        q13, q11, q13
    vsub.s16        q5, q9, q15
    vadd.s16        q15, q9, q15
    vqdmulh.s16     q4, q2, XFIX_1_414213562
    vqdmulh.s16     q6, q1, XFIX_2_613125930
    vadd.s16        q3, q1, q1
    vsub.s16        q1, q5, q1
    vadd.s16        q10, q2, q4
    vqdmulh.s16     q4, q1, XFIX_1_847759065
    vsub.s16        q2, q15, q13
    vadd.s16        q3, q3, q6
    vqdmulh.s16     q6, q2, XFIX_1_414213562
    vadd.s16        q1, q1, q4
    vqdmulh.s16     q4, q5, XFIX_1_082392200
    vsub.s16        q10, q10, q14
    vadd.s16        q2, q2, q6
    vsub.s16        q6, q8, q12
    vadd.s16        q12, q8, q12
    vadd.s16        q9, q5, q4
    vadd.s16        q5, q6, q10
    vsub.s16        q10, q6, q10
    vadd.s16        q6, q15, q13
    vadd.s16        q8, q12, q14
    vsub.s16        q3, q6, q3
    vsub.s16        q12, q12, q14
    vsub.s16        q3, q3, q1
    vsub.s16        q1, q9, q1
    vadd.s16        q2, q3, q2
    vsub.s16        q15, q8, q6
    vadd.s16        q1, q1, q2
    vadd.s16        q8, q8, q6
    vadd.s16        q14, q5, q3
    vsub.s16        q9, q5, q3
    vsub.s16        q13, q10, q2
    vpop            {d8 - d13}    /* restore Neon registers */
    vadd.s16        q10, q10, q2
    vsub.s16        q11, q12, q1
    vadd.s16        q12, q12, q1
    /* Descale to 8-bit and range limit */
    vmov.u8         q0, #0x80
    vqshrn.s16      d16, q8, #5
    vqshrn.s16      d17, q9, #5
    vqshrn.s16      d18, q10, #5
    vqshrn.s16      d19, q11, #5
    vqshrn.s16      d20, q12, #5
    vqshrn.s16      d21, q13, #5
    vqshrn.s16      d22, q14, #5
    vqshrn.s16      d23, q15, #5
    vadd.u8         q8, q8, q0
    vadd.u8         q9, q9, q0
    vadd.u8         q10, q10, q0
    vadd.u8         q11, q11, q0
    /* Transpose the final 8-bit samples */
    vtrn.16         q8, q9
    vtrn.16         q10, q11
    vtrn.32         q8, q10
    vtrn.32         q9, q11
    vtrn.8          d16, d17
    vtrn.8          d18, d19
      /* Store results to the output buffer */
      ldmia           OUTPUT_BUF!, {TMP1, TMP2}
      add             TMP1, TMP1, OUTPUT_COL
      add             TMP2, TMP2, OUTPUT_COL
      vst1.8          {d16}, [TMP1]
      vst1.8          {d17}, [TMP2]
      ldmia           OUTPUT_BUF!, {TMP1, TMP2}
      add             TMP1, TMP1, OUTPUT_COL
      add             TMP2, TMP2, OUTPUT_COL
      vst1.8          {d18}, [TMP1]
    vtrn.8          d20, d21
      vst1.8          {d19}, [TMP2]
      ldmia           OUTPUT_BUF, {TMP1, TMP2, TMP3, TMP4}
      add             TMP1, TMP1, OUTPUT_COL
      add             TMP2, TMP2, OUTPUT_COL
      add             TMP3, TMP3, OUTPUT_COL
      add             TMP4, TMP4, OUTPUT_COL
      vst1.8          {d20}, [TMP1]
    vtrn.8          d22, d23
      vst1.8          {d21}, [TMP2]
      vst1.8          {d22}, [TMP3]
      vst1.8          {d23}, [TMP4]
    bx              lr

    .unreq          DCT_TABLE
    .unreq          COEF_BLOCK
    .unreq          OUTPUT_BUF
    .unreq          OUTPUT_COL
    .unreq          TMP1
    .unreq          TMP2
    .unreq          TMP3
    .unreq          TMP4


/*****************************************************************************/

/*
 * jsimd_extrgb_ycc_convert_neon
 * jsimd_extbgr_ycc_convert_neon
 * jsimd_extrgbx_ycc_convert_neon
 * jsimd_extbgrx_ycc_convert_neon
 * jsimd_extxbgr_ycc_convert_neon
 * jsimd_extxrgb_ycc_convert_neon
 *
 * Colorspace conversion RGB -> YCbCr
 */

.macro do_store size
  .if \size == 8
    vst1.8          {d20}, [Y]!
    vst1.8          {d21}, [U]!
    vst1.8          {d22}, [V]!
  .elseif \size == 4
    vst1.8          {d20[0]}, [Y]!
    vst1.8          {d20[1]}, [Y]!
    vst1.8          {d20[2]}, [Y]!
    vst1.8          {d20[3]}, [Y]!
    vst1.8          {d21[0]}, [U]!
    vst1.8          {d21[1]}, [U]!
    vst1.8          {d21[2]}, [U]!
    vst1.8          {d21[3]}, [U]!
    vst1.8          {d22[0]}, [V]!
    vst1.8          {d22[1]}, [V]!
    vst1.8          {d22[2]}, [V]!
    vst1.8          {d22[3]}, [V]!
  .elseif \size == 2
    vst1.8          {d20[4]}, [Y]!
    vst1.8          {d20[5]}, [Y]!
    vst1.8          {d21[4]}, [U]!
    vst1.8          {d21[5]}, [U]!
    vst1.8          {d22[4]}, [V]!
    vst1.8          {d22[5]}, [V]!
  .elseif \size == 1
    vst1.8          {d20[6]}, [Y]!
    vst1.8          {d21[6]}, [U]!
    vst1.8          {d22[6]}, [V]!
  .else
    .error unsupported macroblock size
  .endif
.endm

.macro do_load bpp, size
  .if \bpp == 24
    .if \size == 8
      vld3.8        {d10, d11, d12}, [RGB]!
      pld           [RGB, #128]
    .elseif \size == 4
      vld3.8        {d10[0], d11[0], d12[0]}, [RGB]!
      vld3.8        {d10[1], d11[1], d12[1]}, [RGB]!
      vld3.8        {d10[2], d11[2], d12[2]}, [RGB]!
      vld3.8        {d10[3], d11[3], d12[3]}, [RGB]!
    .elseif \size == 2
      vld3.8        {d10[4], d11[4], d12[4]}, [RGB]!
      vld3.8        {d10[5], d11[5], d12[5]}, [RGB]!
    .elseif \size == 1
      vld3.8        {d10[6], d11[6], d12[6]}, [RGB]!
    .else
      .error unsupported macroblock size
    .endif
  .elseif \bpp == 32
    .if \size == 8
      vld4.8        {d10, d11, d12, d13}, [RGB]!
      pld           [RGB, #128]
    .elseif \size == 4
      vld4.8        {d10[0], d11[0], d12[0], d13[0]}, [RGB]!
      vld4.8        {d10[1], d11[1], d12[1], d13[1]}, [RGB]!
      vld4.8        {d10[2], d11[2], d12[2], d13[2]}, [RGB]!
      vld4.8        {d10[3], d11[3], d12[3], d13[3]}, [RGB]!
    .elseif \size == 2
      vld4.8        {d10[4], d11[4], d12[4], d13[4]}, [RGB]!
      vld4.8        {d10[5], d11[5], d12[5], d13[5]}, [RGB]!
    .elseif \size == 1
      vld4.8        {d10[6], d11[6], d12[6], d13[6]}, [RGB]!
    .else
      .error unsupported macroblock size
    .endif
  .else
    .error unsupported bpp
  .endif
.endm

.macro generate_jsimd_rgb_ycc_convert_neon colorid, bpp, r_offs, g_offs, b_offs

/*
 * 2-stage pipelined RGB->YCbCr conversion
 */

.macro do_rgb_to_yuv_stage1
    vmovl.u8        q2, d1\r_offs  /* r = { d4, d5 } */
    vmovl.u8        q3, d1\g_offs  /* g = { d6, d7 } */
    vmovl.u8        q4, d1\b_offs  /* b = { d8, d9 } */
    vmull.u16       q7, d4, d0[0]
    vmlal.u16       q7, d6, d0[1]
    vmlal.u16       q7, d8, d0[2]
    vmull.u16       q8, d5, d0[0]
    vmlal.u16       q8, d7, d0[1]
    vmlal.u16       q8, d9, d0[2]
    vrev64.32       q9, q1
    vrev64.32       q13, q1
    vmlsl.u16       q9, d4, d0[3]
    vmlsl.u16       q9, d6, d1[0]
    vmlal.u16       q9, d8, d1[1]
    vmlsl.u16       q13, d5, d0[3]
    vmlsl.u16       q13, d7, d1[0]
    vmlal.u16       q13, d9, d1[1]
    vrev64.32       q14, q1
    vrev64.32       q15, q1
    vmlal.u16       q14, d4, d1[1]
    vmlsl.u16       q14, d6, d1[2]
    vmlsl.u16       q14, d8, d1[3]
    vmlal.u16       q15, d5, d1[1]
    vmlsl.u16       q15, d7, d1[2]
    vmlsl.u16       q15, d9, d1[3]
.endm

.macro do_rgb_to_yuv_stage2
    vrshrn.u32      d20, q7, #16
    vrshrn.u32      d21, q8, #16
    vshrn.u32       d22, q9, #16
    vshrn.u32       d23, q13, #16
    vshrn.u32       d24, q14, #16
    vshrn.u32       d25, q15, #16
    vmovn.u16       d20, q10       /* d20 = y */
    vmovn.u16       d21, q11       /* d21 = u */
    vmovn.u16       d22, q12       /* d22 = v */
.endm

.macro do_rgb_to_yuv
    do_rgb_to_yuv_stage1
    do_rgb_to_yuv_stage2
.endm

.macro do_rgb_to_yuv_stage2_store_load_stage1
      vrshrn.u32      d20, q7, #16
      vrshrn.u32      d21, q8, #16
      vshrn.u32       d22, q9, #16
    vrev64.32       q9, q1
      vshrn.u32       d23, q13, #16
    vrev64.32       q13, q1
      vshrn.u32       d24, q14, #16
      vshrn.u32       d25, q15, #16
    do_load         \bpp, 8
      vmovn.u16       d20, q10     /* d20 = y */
    vmovl.u8        q2, d1\r_offs  /* r = { d4, d5 } */
      vmovn.u16       d21, q11     /* d21 = u */
    vmovl.u8        q3, d1\g_offs  /* g = { d6, d7 } */
      vmovn.u16       d22, q12     /* d22 = v */
    vmovl.u8        q4, d1\b_offs  /* b = { d8, d9 } */
    vmull.u16       q7, d4, d0[0]
    vmlal.u16       q7, d6, d0[1]
    vmlal.u16       q7, d8, d0[2]
      vst1.8          {d20}, [Y]!
    vmull.u16       q8, d5, d0[0]
    vmlal.u16       q8, d7, d0[1]
    vmlal.u16       q8, d9, d0[2]
    vmlsl.u16       q9, d4, d0[3]
    vmlsl.u16       q9, d6, d1[0]
    vmlal.u16       q9, d8, d1[1]
      vst1.8          {d21}, [U]!
    vmlsl.u16       q13, d5, d0[3]
    vmlsl.u16       q13, d7, d1[0]
    vmlal.u16       q13, d9, d1[1]
    vrev64.32       q14, q1
    vrev64.32       q15, q1
    vmlal.u16       q14, d4, d1[1]
    vmlsl.u16       q14, d6, d1[2]
    vmlsl.u16       q14, d8, d1[3]
      vst1.8          {d22}, [V]!
    vmlal.u16       q15, d5, d1[1]
    vmlsl.u16       q15, d7, d1[2]
    vmlsl.u16       q15, d9, d1[3]
.endm

.balign 16
jsimd_\colorid\()_ycc_neon_consts:
  .short 19595, 38470, 7471,  11059
  .short 21709, 32768, 27439, 5329
  .short 32767, 128,   32767, 128
  .short 32767, 128,   32767, 128

asm_function jsimd_\colorid\()_ycc_convert_neon
    OUTPUT_WIDTH    .req r0
    INPUT_BUF       .req r1
    OUTPUT_BUF      .req r2
    OUTPUT_ROW      .req r3
    NUM_ROWS        .req r4

    OUTPUT_BUF0     .req r5
    OUTPUT_BUF1     .req r6
    OUTPUT_BUF2     .req OUTPUT_BUF

    RGB             .req r7
    Y               .req r8
    U               .req r9
    V               .req r10
    N               .req ip

    /* Load constants to d0, d1, d2, d3 */
    adr             ip, jsimd_\colorid\()_ycc_neon_consts
    vld1.16         {d0, d1, d2, d3}, [ip, :128]

    /* Save Arm registers and handle input arguments */
    push            {r4, r5, r6, r7, r8, r9, r10, lr}
    ldr             NUM_ROWS, [sp, #(4 * 8)]
    ldr             OUTPUT_BUF0, [OUTPUT_BUF]
    ldr             OUTPUT_BUF1, [OUTPUT_BUF, #4]
    ldr             OUTPUT_BUF2, [OUTPUT_BUF, #8]
    .unreq          OUTPUT_BUF

    /* Save Neon registers */
    vpush           {d8 - d15}

    /* Outer loop over scanlines */
    cmp             NUM_ROWS, #1
    blt             9f
0:
    ldr             Y, [OUTPUT_BUF0, OUTPUT_ROW, lsl #2]
    ldr             U, [OUTPUT_BUF1, OUTPUT_ROW, lsl #2]
    mov             N, OUTPUT_WIDTH
    ldr             V, [OUTPUT_BUF2, OUTPUT_ROW, lsl #2]
    add             OUTPUT_ROW, OUTPUT_ROW, #1
    ldr             RGB, [INPUT_BUF], #4

    /* Inner loop over pixels */
    subs            N, N, #8
    blt             3f
    do_load         \bpp, 8
    do_rgb_to_yuv_stage1
    subs            N, N, #8
    blt             2f
1:
    do_rgb_to_yuv_stage2_store_load_stage1
    subs            N, N, #8
    bge             1b
2:
    do_rgb_to_yuv_stage2
    do_store        8
    tst             N, #7
    beq             8f
3:
    tst             N, #4
    beq             3f
    do_load         \bpp, 4
3:
    tst             N, #2
    beq             4f
    do_load         \bpp, 2
4:
    tst             N, #1
    beq             5f
    do_load         \bpp, 1
5:
    do_rgb_to_yuv
    tst             N, #4
    beq             6f
    do_store        4
6:
    tst             N, #2
    beq             7f
    do_store        2
7:
    tst             N, #1
    beq             8f
    do_store        1
8:
    subs            NUM_ROWS, NUM_ROWS, #1
    bgt             0b
9:
    /* Restore all registers and return */
    vpop            {d8 - d15}
    pop             {r4, r5, r6, r7, r8, r9, r10, pc}

    .unreq          OUTPUT_WIDTH
    .unreq          OUTPUT_ROW
    .unreq          INPUT_BUF
    .unreq          NUM_ROWS
    .unreq          OUTPUT_BUF0
    .unreq          OUTPUT_BUF1
    .unreq          OUTPUT_BUF2
    .unreq          RGB
    .unreq          Y
    .unreq          U
    .unreq          V
    .unreq          N

.purgem do_rgb_to_yuv
.purgem do_rgb_to_yuv_stage1
.purgem do_rgb_to_yuv_stage2
.purgem do_rgb_to_yuv_stage2_store_load_stage1

.endm

/*--------------------------------- id ----- bpp R  G  B */
generate_jsimd_rgb_ycc_convert_neon extrgb,  24, 0, 1, 2
generate_jsimd_rgb_ycc_convert_neon extbgr,  24, 2, 1, 0
generate_jsimd_rgb_ycc_convert_neon extrgbx, 32, 0, 1, 2
generate_jsimd_rgb_ycc_convert_neon extbgrx, 32, 2, 1, 0
generate_jsimd_rgb_ycc_convert_neon extxbgr, 32, 3, 2, 1
generate_jsimd_rgb_ycc_convert_neon extxrgb, 32, 1, 2, 3

.purgem do_load
.purgem do_store
