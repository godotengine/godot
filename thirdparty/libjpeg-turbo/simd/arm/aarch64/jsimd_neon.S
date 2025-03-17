/*
 * Armv8 Neon optimizations for libjpeg-turbo
 *
 * Copyright (C) 2009-2011, Nokia Corporation and/or its subsidiary(-ies).
 *                          All Rights Reserved.
 * Author:  Siarhei Siamashka <siarhei.siamashka@nokia.com>
 * Copyright (C) 2013-2014, Linaro Limited.  All Rights Reserved.
 * Author:  Ragesh Radhakrishnan <ragesh.r@linaro.org>
 * Copyright (C) 2014-2016, 2020, D. R. Commander.  All Rights Reserved.
 * Copyright (C) 2015-2016, 2018, Matthieu Darbois.  All Rights Reserved.
 * Copyright (C) 2016, Siarhei Siamashka.  All Rights Reserved.
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

#if defined(__APPLE__)
.section __DATA, __const
#elif defined(_WIN32)
.section .rdata
#else
.section .rodata, "a", %progbits
#endif

/* Constants for jsimd_idct_islow_neon() */

#define F_0_298   2446  /* FIX(0.298631336) */
#define F_0_390   3196  /* FIX(0.390180644) */
#define F_0_541   4433  /* FIX(0.541196100) */
#define F_0_765   6270  /* FIX(0.765366865) */
#define F_0_899   7373  /* FIX(0.899976223) */
#define F_1_175   9633  /* FIX(1.175875602) */
#define F_1_501  12299  /* FIX(1.501321110) */
#define F_1_847  15137  /* FIX(1.847759065) */
#define F_1_961  16069  /* FIX(1.961570560) */
#define F_2_053  16819  /* FIX(2.053119869) */
#define F_2_562  20995  /* FIX(2.562915447) */
#define F_3_072  25172  /* FIX(3.072711026) */

.balign 16
Ljsimd_idct_islow_neon_consts:
  .short F_0_298
  .short -F_0_390
  .short F_0_541
  .short F_0_765
  .short - F_0_899
  .short F_1_175
  .short F_1_501
  .short - F_1_847
  .short - F_1_961
  .short F_2_053
  .short - F_2_562
  .short F_3_072
  .short 0          /* padding */
  .short 0
  .short 0
  .short 0

#undef F_0_298
#undef F_0_390
#undef F_0_541
#undef F_0_765
#undef F_0_899
#undef F_1_175
#undef F_1_501
#undef F_1_847
#undef F_1_961
#undef F_2_053
#undef F_2_562
#undef F_3_072

/* Constants for jsimd_ycc_*_neon() */

.balign 16
Ljsimd_ycc_rgb_neon_consts:
  .short 0,      0,     0,      0
  .short 22971, -11277, -23401, 29033
  .short -128,  -128,   -128,   -128
  .short -128,  -128,   -128,   -128

/* Constants for jsimd_*_ycc_neon() */

.balign 16
Ljsimd_rgb_ycc_neon_consts:
  .short 19595, 38470, 7471, 11059
  .short 21709, 32768, 27439, 5329
  .short 32767, 128, 32767, 128
  .short 32767, 128, 32767, 128

/* Constants for jsimd_fdct_islow_neon() */

#define F_0_298   2446  /* FIX(0.298631336) */
#define F_0_390   3196  /* FIX(0.390180644) */
#define F_0_541   4433  /* FIX(0.541196100) */
#define F_0_765   6270  /* FIX(0.765366865) */
#define F_0_899   7373  /* FIX(0.899976223) */
#define F_1_175   9633  /* FIX(1.175875602) */
#define F_1_501  12299  /* FIX(1.501321110) */
#define F_1_847  15137  /* FIX(1.847759065) */
#define F_1_961  16069  /* FIX(1.961570560) */
#define F_2_053  16819  /* FIX(2.053119869) */
#define F_2_562  20995  /* FIX(2.562915447) */
#define F_3_072  25172  /* FIX(3.072711026) */

.balign 16
Ljsimd_fdct_islow_neon_consts:
  .short F_0_298
  .short -F_0_390
  .short F_0_541
  .short F_0_765
  .short - F_0_899
  .short F_1_175
  .short F_1_501
  .short - F_1_847
  .short - F_1_961
  .short F_2_053
  .short - F_2_562
  .short F_3_072
  .short 0          /* padding */
  .short 0
  .short 0
  .short 0

#undef F_0_298
#undef F_0_390
#undef F_0_541
#undef F_0_765
#undef F_0_899
#undef F_1_175
#undef F_1_501
#undef F_1_847
#undef F_1_961
#undef F_2_053
#undef F_2_562
#undef F_3_072

/* Constants for jsimd_huff_encode_one_block_neon() */

.balign 16
Ljsimd_huff_encode_one_block_neon_consts:
    .byte 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, \
          0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    .byte    0,   1,   2,   3,  16,  17,  32,  33, \
            18,  19,   4,   5,   6,   7,  20,  21  /* L0 => L3 : 4 lines OK */
    .byte   34,  35,  48,  49, 255, 255,  50,  51, \
            36,  37,  22,  23,   8,   9,  10,  11  /* L0 => L3 : 4 lines OK */
    .byte    8,   9,  22,  23,  36,  37,  50,  51, \
           255, 255, 255, 255, 255, 255,  52,  53  /* L1 => L4 : 4 lines OK */
    .byte   54,  55,  40,  41,  26,  27,  12,  13, \
            14,  15,  28,  29,  42,  43,  56,  57  /* L0 => L3 : 4 lines OK */
    .byte    6,   7,  20,  21,  34,  35,  48,  49, \
            50,  51,  36,  37,  22,  23,   8,   9  /* L4 => L7 : 4 lines OK */
    .byte   42,  43,  28,  29,  14,  15,  30,  31, \
            44,  45,  58,  59, 255, 255, 255, 255  /* L1 => L4 : 4 lines OK */
    .byte  255, 255, 255, 255,  56,  57,  42,  43, \
            28,  29,  14,  15,  30,  31,  44,  45  /* L3 => L6 : 4 lines OK */
    .byte   26,  27,  40,  41,  42,  43,  28,  29, \
            14,  15,  30,  31,  44,  45,  46,  47  /* L5 => L7 : 3 lines OK */
    .byte  255, 255, 255, 255,   0,   1, 255, 255, \
           255, 255, 255, 255, 255, 255, 255, 255  /* L4 : 1 lines OK */
    .byte  255, 255, 255, 255, 255, 255, 255, 255, \
             0,   1,  16,  17,   2,   3, 255, 255  /* L5 => L6 : 2 lines OK */
    .byte  255, 255, 255, 255, 255, 255, 255, 255, \
           255, 255, 255, 255,   8,   9,  22,  23  /* L5 => L6 : 2 lines OK */
    .byte    4,   5,   6,   7, 255, 255, 255, 255, \
           255, 255, 255, 255, 255, 255, 255, 255  /* L7 : 1 line OK */

.text


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

/* Get symbol location */
.macro get_symbol_loc reg, symbol
#ifdef __APPLE__
    adrp            \reg, \symbol@PAGE
    add             \reg, \reg, \symbol@PAGEOFF
#else
    adrp            \reg, \symbol
    add             \reg, \reg, :lo12:\symbol
#endif
.endm

.macro transpose_8x8 l0, l1, l2, l3, l4, l5, l6, l7, t0, t1, t2, t3
    trn1            \t0\().8h, \l0\().8h, \l1\().8h
    trn1            \t1\().8h, \l2\().8h, \l3\().8h
    trn1            \t2\().8h, \l4\().8h, \l5\().8h
    trn1            \t3\().8h, \l6\().8h, \l7\().8h
    trn2            \l1\().8h, \l0\().8h, \l1\().8h
    trn2            \l3\().8h, \l2\().8h, \l3\().8h
    trn2            \l5\().8h, \l4\().8h, \l5\().8h
    trn2            \l7\().8h, \l6\().8h, \l7\().8h

    trn1            \l4\().4s, \t2\().4s, \t3\().4s
    trn2            \t3\().4s, \t2\().4s, \t3\().4s
    trn1            \t2\().4s, \t0\().4s, \t1\().4s
    trn2            \l2\().4s, \t0\().4s, \t1\().4s
    trn1            \t0\().4s, \l1\().4s, \l3\().4s
    trn2            \l3\().4s, \l1\().4s, \l3\().4s
    trn2            \t1\().4s, \l5\().4s, \l7\().4s
    trn1            \l5\().4s, \l5\().4s, \l7\().4s

    trn2            \l6\().2d, \l2\().2d, \t3\().2d
    trn1            \l0\().2d, \t2\().2d, \l4\().2d
    trn1            \l1\().2d, \t0\().2d, \l5\().2d
    trn2            \l7\().2d, \l3\().2d, \t1\().2d
    trn1            \l2\().2d, \l2\().2d, \t3\().2d
    trn2            \l4\().2d, \t2\().2d, \l4\().2d
    trn1            \l3\().2d, \l3\().2d, \t1\().2d
    trn2            \l5\().2d, \t0\().2d, \l5\().2d
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

#define CONST_BITS  13
#define PASS1_BITS  2

#define XFIX_P_0_298  v0.h[0]
#define XFIX_N_0_390  v0.h[1]
#define XFIX_P_0_541  v0.h[2]
#define XFIX_P_0_765  v0.h[3]
#define XFIX_N_0_899  v0.h[4]
#define XFIX_P_1_175  v0.h[5]
#define XFIX_P_1_501  v0.h[6]
#define XFIX_N_1_847  v0.h[7]
#define XFIX_N_1_961  v1.h[0]
#define XFIX_P_2_053  v1.h[1]
#define XFIX_N_2_562  v1.h[2]
#define XFIX_P_3_072  v1.h[3]

asm_function jsimd_idct_islow_neon
    DCT_TABLE       .req x0
    COEF_BLOCK      .req x1
    OUTPUT_BUF      .req x2
    OUTPUT_COL      .req x3
    TMP1            .req x0
    TMP2            .req x1
    TMP3            .req x9
    TMP4            .req x10
    TMP5            .req x11
    TMP6            .req x12
    TMP7            .req x13
    TMP8            .req x14

    /* OUTPUT_COL is a JDIMENSION (unsigned int) argument, so the ABI doesn't
       guarantee that the upper (unused) 32 bits of x3 are valid.  This
       instruction ensures that those bits are set to zero. */
    uxtw x3, w3

    sub             sp, sp, #64
    get_symbol_loc  x15, Ljsimd_idct_islow_neon_consts
    mov             x10, sp
    st1             {v8.8b, v9.8b, v10.8b, v11.8b}, [x10], #32
    st1             {v12.8b, v13.8b, v14.8b, v15.8b}, [x10], #32
    ld1             {v0.8h, v1.8h}, [x15]
    ld1             {v2.8h, v3.8h, v4.8h, v5.8h}, [COEF_BLOCK], #64
    ld1             {v18.8h, v19.8h, v20.8h, v21.8h}, [DCT_TABLE], #64
    ld1             {v6.8h, v7.8h, v8.8h, v9.8h}, [COEF_BLOCK], #64
    ld1             {v22.8h, v23.8h, v24.8h, v25.8h}, [DCT_TABLE], #64

    cmeq            v16.8h, v3.8h, #0
    cmeq            v26.8h, v4.8h, #0
    cmeq            v27.8h, v5.8h, #0
    cmeq            v28.8h, v6.8h, #0
    cmeq            v29.8h, v7.8h, #0
    cmeq            v30.8h, v8.8h, #0
    cmeq            v31.8h, v9.8h, #0

    and             v10.16b, v16.16b, v26.16b
    and             v11.16b, v27.16b, v28.16b
    and             v12.16b, v29.16b, v30.16b
    and             v13.16b, v31.16b, v10.16b
    and             v14.16b, v11.16b, v12.16b
    mul             v2.8h, v2.8h, v18.8h
    and             v15.16b, v13.16b, v14.16b
    shl             v10.8h, v2.8h, #(PASS1_BITS)
    sqxtn           v16.8b, v15.8h
    mov             TMP1, v16.d[0]
    mvn             TMP2, TMP1

    cbnz            TMP2, 2f
    /* case all AC coeffs are zeros */
    dup             v2.2d, v10.d[0]
    dup             v6.2d, v10.d[1]
    mov             v3.16b, v2.16b
    mov             v7.16b, v6.16b
    mov             v4.16b, v2.16b
    mov             v8.16b, v6.16b
    mov             v5.16b, v2.16b
    mov             v9.16b, v6.16b
1:
    /* for this transpose, we should organise data like this:
     * 00, 01, 02, 03, 40, 41, 42, 43
     * 10, 11, 12, 13, 50, 51, 52, 53
     * 20, 21, 22, 23, 60, 61, 62, 63
     * 30, 31, 32, 33, 70, 71, 72, 73
     * 04, 05, 06, 07, 44, 45, 46, 47
     * 14, 15, 16, 17, 54, 55, 56, 57
     * 24, 25, 26, 27, 64, 65, 66, 67
     * 34, 35, 36, 37, 74, 75, 76, 77
     */
    trn1            v28.8h, v2.8h, v3.8h
    trn1            v29.8h, v4.8h, v5.8h
    trn1            v30.8h, v6.8h, v7.8h
    trn1            v31.8h, v8.8h, v9.8h
    trn2            v16.8h, v2.8h, v3.8h
    trn2            v17.8h, v4.8h, v5.8h
    trn2            v18.8h, v6.8h, v7.8h
    trn2            v19.8h, v8.8h, v9.8h
    trn1            v2.4s, v28.4s, v29.4s
    trn1            v6.4s, v30.4s, v31.4s
    trn1            v3.4s, v16.4s, v17.4s
    trn1            v7.4s, v18.4s, v19.4s
    trn2            v4.4s, v28.4s, v29.4s
    trn2            v8.4s, v30.4s, v31.4s
    trn2            v5.4s, v16.4s, v17.4s
    trn2            v9.4s, v18.4s, v19.4s
    /* Even part: reverse the even part of the forward DCT. */
    add             v18.8h, v4.8h, v8.8h           /* z2 + z3 = DEQUANTIZE(inptr[DCTSIZE*2], quantptr[DCTSIZE*2]) + DEQUANTIZE(inptr[DCTSIZE*6], quantptr[DCTSIZE*6]) */
    add             v22.8h, v2.8h, v6.8h           /* z2 + z3 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) + DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]) */
    smull2          v19.4s, v18.8h, XFIX_P_0_541   /* z1h z1 = MULTIPLY(z2 + z3, FIX_0_541196100); */
    sub             v26.8h, v2.8h, v6.8h           /* z2 - z3 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) - DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]) */
    smull           v18.4s, v18.4h, XFIX_P_0_541   /* z1l z1 = MULTIPLY(z2 + z3, FIX_0_541196100); */
    sshll2          v23.4s, v22.8h, #(CONST_BITS)  /* tmp0h tmp0 = LEFT_SHIFT(z2 + z3, CONST_BITS); */
    mov             v21.16b, v19.16b               /* tmp3 = z1 */
    mov             v20.16b, v18.16b               /* tmp3 = z1 */
    smlal2          v19.4s, v8.8h, XFIX_N_1_847    /* tmp2h tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065); */
    smlal           v18.4s, v8.4h, XFIX_N_1_847    /* tmp2l tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065); */
    sshll2          v27.4s, v26.8h, #(CONST_BITS)  /* tmp1h tmp1 = LEFT_SHIFT(z2 - z3, CONST_BITS); */
    smlal2          v21.4s, v4.8h, XFIX_P_0_765    /* tmp3h tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865); */
    smlal           v20.4s, v4.4h, XFIX_P_0_765    /* tmp3l tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865); */
    sshll           v22.4s, v22.4h, #(CONST_BITS)  /* tmp0l tmp0 = LEFT_SHIFT(z2 + z3, CONST_BITS); */
    sshll           v26.4s, v26.4h, #(CONST_BITS)  /* tmp1l tmp1 = LEFT_SHIFT(z2 - z3, CONST_BITS); */
    add             v2.4s, v22.4s, v20.4s          /* tmp10l tmp10 = tmp0 + tmp3; */
    sub             v6.4s, v22.4s, v20.4s          /* tmp13l tmp13 = tmp0 - tmp3; */
    add             v8.4s, v26.4s, v18.4s          /* tmp11l tmp11 = tmp1 + tmp2; */
    sub             v4.4s, v26.4s, v18.4s          /* tmp12l tmp12 = tmp1 - tmp2; */
    add             v28.4s, v23.4s, v21.4s         /* tmp10h tmp10 = tmp0 + tmp3; */
    sub             v31.4s, v23.4s, v21.4s         /* tmp13h tmp13 = tmp0 - tmp3; */
    add             v29.4s, v27.4s, v19.4s         /* tmp11h tmp11 = tmp1 + tmp2; */
    sub             v30.4s, v27.4s, v19.4s         /* tmp12h tmp12 = tmp1 - tmp2; */

    /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

    add             v22.8h, v9.8h, v5.8h    /* z3 = tmp0 + tmp2 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]) + DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]) */
    add             v24.8h, v7.8h, v3.8h    /* z4 = tmp1 + tmp3 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]) + DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]) */
    add             v18.8h, v9.8h, v3.8h    /* z1 = tmp0 + tmp3 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]) + DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]) */
    add             v20.8h, v7.8h, v5.8h    /* z2 = tmp1 + tmp2 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]) + DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]) */
    add             v26.8h, v22.8h, v24.8h  /* z5 = z3 + z4 */

    smull2          v11.4s, v9.8h, XFIX_P_0_298   /* tmp0 = MULTIPLY(tmp0, FIX_0_298631336) */
    smull2          v13.4s, v7.8h, XFIX_P_2_053   /* tmp1 = MULTIPLY(tmp1, FIX_2_053119869) */
    smull2          v15.4s, v5.8h, XFIX_P_3_072   /* tmp2 = MULTIPLY(tmp2, FIX_3_072711026) */
    smull2          v17.4s, v3.8h, XFIX_P_1_501   /* tmp3 = MULTIPLY(tmp3, FIX_1_501321110) */
    smull2          v27.4s, v26.8h, XFIX_P_1_175  /* z5h z5 = MULTIPLY(z3 + z4, FIX_1_175875602) */
    smull2          v23.4s, v22.8h, XFIX_N_1_961  /* z3 = MULTIPLY(z3, -FIX_1_961570560) */
    smull2          v25.4s, v24.8h, XFIX_N_0_390  /* z4 = MULTIPLY(z4, -FIX_0_390180644) */
    smull2          v19.4s, v18.8h, XFIX_N_0_899  /* z1 = MULTIPLY(z1, -FIX_0_899976223) */
    smull2          v21.4s, v20.8h, XFIX_N_2_562  /* z2 = MULTIPLY(z2, -FIX_2_562915447) */

    smull           v10.4s, v9.4h, XFIX_P_0_298   /* tmp0 = MULTIPLY(tmp0, FIX_0_298631336) */
    smull           v12.4s, v7.4h, XFIX_P_2_053   /* tmp1 = MULTIPLY(tmp1, FIX_2_053119869) */
    smull           v14.4s, v5.4h, XFIX_P_3_072   /* tmp2 = MULTIPLY(tmp2, FIX_3_072711026) */
    smull           v16.4s, v3.4h, XFIX_P_1_501   /* tmp3 = MULTIPLY(tmp3, FIX_1_501321110) */
    smull           v26.4s, v26.4h, XFIX_P_1_175  /* z5l z5 = MULTIPLY(z3 + z4, FIX_1_175875602) */
    smull           v22.4s, v22.4h, XFIX_N_1_961  /* z3 = MULTIPLY(z3, -FIX_1_961570560) */
    smull           v24.4s, v24.4h, XFIX_N_0_390  /* z4 = MULTIPLY(z4, -FIX_0_390180644) */
    smull           v18.4s, v18.4h, XFIX_N_0_899  /* z1 = MULTIPLY(z1, -FIX_0_899976223) */
    smull           v20.4s, v20.4h, XFIX_N_2_562  /* z2 = MULTIPLY(z2, -FIX_2_562915447) */

    add             v23.4s, v23.4s, v27.4s  /* z3 += z5 */
    add             v22.4s, v22.4s, v26.4s  /* z3 += z5 */
    add             v25.4s, v25.4s, v27.4s  /* z4 += z5 */
    add             v24.4s, v24.4s, v26.4s  /* z4 += z5 */

    add             v11.4s, v11.4s, v19.4s  /* tmp0 += z1 */
    add             v10.4s, v10.4s, v18.4s  /* tmp0 += z1 */
    add             v13.4s, v13.4s, v21.4s  /* tmp1 += z2 */
    add             v12.4s, v12.4s, v20.4s  /* tmp1 += z2 */
    add             v15.4s, v15.4s, v21.4s  /* tmp2 += z2 */
    add             v14.4s, v14.4s, v20.4s  /* tmp2 += z2 */
    add             v17.4s, v17.4s, v19.4s  /* tmp3 += z1 */
    add             v16.4s, v16.4s, v18.4s  /* tmp3 += z1 */

    add             v11.4s, v11.4s, v23.4s  /* tmp0 += z3 */
    add             v10.4s, v10.4s, v22.4s  /* tmp0 += z3 */
    add             v13.4s, v13.4s, v25.4s  /* tmp1 += z4 */
    add             v12.4s, v12.4s, v24.4s  /* tmp1 += z4 */
    add             v17.4s, v17.4s, v25.4s  /* tmp3 += z4 */
    add             v16.4s, v16.4s, v24.4s  /* tmp3 += z4 */
    add             v15.4s, v15.4s, v23.4s  /* tmp2 += z3 */
    add             v14.4s, v14.4s, v22.4s  /* tmp2 += z3 */

    /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

    add             v18.4s, v2.4s, v16.4s   /* tmp10 + tmp3 */
    add             v19.4s, v28.4s, v17.4s  /* tmp10 + tmp3 */
    sub             v20.4s, v2.4s, v16.4s   /* tmp10 - tmp3 */
    sub             v21.4s, v28.4s, v17.4s  /* tmp10 - tmp3 */
    add             v22.4s, v8.4s, v14.4s   /* tmp11 + tmp2 */
    add             v23.4s, v29.4s, v15.4s  /* tmp11 + tmp2 */
    sub             v24.4s, v8.4s, v14.4s   /* tmp11 - tmp2 */
    sub             v25.4s, v29.4s, v15.4s  /* tmp11 - tmp2 */
    add             v26.4s, v4.4s, v12.4s   /* tmp12 + tmp1 */
    add             v27.4s, v30.4s, v13.4s  /* tmp12 + tmp1 */
    sub             v28.4s, v4.4s, v12.4s   /* tmp12 - tmp1 */
    sub             v29.4s, v30.4s, v13.4s  /* tmp12 - tmp1 */
    add             v14.4s, v6.4s, v10.4s   /* tmp13 + tmp0 */
    add             v15.4s, v31.4s, v11.4s  /* tmp13 + tmp0 */
    sub             v16.4s, v6.4s, v10.4s   /* tmp13 - tmp0 */
    sub             v17.4s, v31.4s, v11.4s  /* tmp13 - tmp0 */

    shrn            v2.4h, v18.4s, #16  /* wsptr[DCTSIZE*0] = (int)DESCALE(tmp10 + tmp3, CONST_BITS+PASS1_BITS+3) */
    shrn            v9.4h, v20.4s, #16  /* wsptr[DCTSIZE*7] = (int)DESCALE(tmp10 - tmp3, CONST_BITS+PASS1_BITS+3) */
    shrn            v3.4h, v22.4s, #16  /* wsptr[DCTSIZE*1] = (int)DESCALE(tmp11 + tmp2, CONST_BITS+PASS1_BITS+3) */
    shrn            v8.4h, v24.4s, #16  /* wsptr[DCTSIZE*6] = (int)DESCALE(tmp11 - tmp2, CONST_BITS+PASS1_BITS+3) */
    shrn            v4.4h, v26.4s, #16  /* wsptr[DCTSIZE*2] = (int)DESCALE(tmp12 + tmp1, CONST_BITS+PASS1_BITS+3) */
    shrn            v7.4h, v28.4s, #16  /* wsptr[DCTSIZE*5] = (int)DESCALE(tmp12 - tmp1, CONST_BITS+PASS1_BITS+3) */
    shrn            v5.4h, v14.4s, #16  /* wsptr[DCTSIZE*3] = (int)DESCALE(tmp13 + tmp0, CONST_BITS+PASS1_BITS+3) */
    shrn            v6.4h, v16.4s, #16  /* wsptr[DCTSIZE*4] = (int)DESCALE(tmp13 - tmp0, CONST_BITS+PASS1_BITS+3) */
    shrn2           v2.8h, v19.4s, #16  /* wsptr[DCTSIZE*0] = (int)DESCALE(tmp10 + tmp3, CONST_BITS+PASS1_BITS+3) */
    shrn2           v9.8h, v21.4s, #16  /* wsptr[DCTSIZE*7] = (int)DESCALE(tmp10 - tmp3, CONST_BITS+PASS1_BITS+3) */
    shrn2           v3.8h, v23.4s, #16  /* wsptr[DCTSIZE*1] = (int)DESCALE(tmp11 + tmp2, CONST_BITS+PASS1_BITS+3) */
    shrn2           v8.8h, v25.4s, #16  /* wsptr[DCTSIZE*6] = (int)DESCALE(tmp11 - tmp2, CONST_BITS+PASS1_BITS+3) */
    shrn2           v4.8h, v27.4s, #16  /* wsptr[DCTSIZE*2] = (int)DESCALE(tmp12 + tmp1, CONST_BITS+PASS1_BITS+3) */
    shrn2           v7.8h, v29.4s, #16  /* wsptr[DCTSIZE*5] = (int)DESCALE(tmp12 - tmp1, CONST_BITS+PASS1_BITS+3) */
    shrn2           v5.8h, v15.4s, #16  /* wsptr[DCTSIZE*3] = (int)DESCALE(tmp13 + tmp0, CONST_BITS+PASS1_BITS+3) */
    shrn2           v6.8h, v17.4s, #16  /* wsptr[DCTSIZE*4] = (int)DESCALE(tmp13 - tmp0, CONST_BITS+PASS1_BITS+3) */
    movi            v0.16b, #(CENTERJSAMPLE)
    /* Prepare pointers (dual-issue with Neon instructions) */
      ldp             TMP1, TMP2, [OUTPUT_BUF], 16
    sqrshrn         v28.8b, v2.8h, #(CONST_BITS + PASS1_BITS + 3 - 16)
      ldp             TMP3, TMP4, [OUTPUT_BUF], 16
    sqrshrn         v29.8b, v3.8h, #(CONST_BITS + PASS1_BITS + 3 - 16)
      add             TMP1, TMP1, OUTPUT_COL
    sqrshrn         v30.8b, v4.8h, #(CONST_BITS + PASS1_BITS + 3 - 16)
      add             TMP2, TMP2, OUTPUT_COL
    sqrshrn         v31.8b, v5.8h, #(CONST_BITS + PASS1_BITS + 3 - 16)
      add             TMP3, TMP3, OUTPUT_COL
    sqrshrn2        v28.16b, v6.8h, #(CONST_BITS + PASS1_BITS + 3 - 16)
      add             TMP4, TMP4, OUTPUT_COL
    sqrshrn2        v29.16b, v7.8h, #(CONST_BITS + PASS1_BITS + 3 - 16)
      ldp             TMP5, TMP6, [OUTPUT_BUF], 16
    sqrshrn2        v30.16b, v8.8h, #(CONST_BITS + PASS1_BITS + 3 - 16)
      ldp             TMP7, TMP8, [OUTPUT_BUF], 16
    sqrshrn2        v31.16b, v9.8h, #(CONST_BITS + PASS1_BITS + 3 - 16)
      add             TMP5, TMP5, OUTPUT_COL
    add             v16.16b, v28.16b, v0.16b
      add             TMP6, TMP6, OUTPUT_COL
    add             v18.16b, v29.16b, v0.16b
      add             TMP7, TMP7, OUTPUT_COL
    add             v20.16b, v30.16b, v0.16b
      add             TMP8, TMP8, OUTPUT_COL
    add             v22.16b, v31.16b, v0.16b

    /* Transpose the final 8-bit samples */
    trn1            v28.16b, v16.16b, v18.16b
    trn1            v30.16b, v20.16b, v22.16b
    trn2            v29.16b, v16.16b, v18.16b
    trn2            v31.16b, v20.16b, v22.16b

    trn1            v16.8h, v28.8h, v30.8h
    trn2            v18.8h, v28.8h, v30.8h
    trn1            v20.8h, v29.8h, v31.8h
    trn2            v22.8h, v29.8h, v31.8h

    uzp1            v28.4s, v16.4s, v18.4s
    uzp2            v30.4s, v16.4s, v18.4s
    uzp1            v29.4s, v20.4s, v22.4s
    uzp2            v31.4s, v20.4s, v22.4s

    /* Store results to the output buffer */
    st1             {v28.d}[0], [TMP1]
    st1             {v29.d}[0], [TMP2]
    st1             {v28.d}[1], [TMP3]
    st1             {v29.d}[1], [TMP4]
    st1             {v30.d}[0], [TMP5]
    st1             {v31.d}[0], [TMP6]
    st1             {v30.d}[1], [TMP7]
    st1             {v31.d}[1], [TMP8]
    ld1             {v8.8b, v9.8b, v10.8b, v11.8b}, [sp], #32
    ld1             {v12.8b, v13.8b, v14.8b, v15.8b}, [sp], #32
    blr             x30

.balign 16
2:
    mul             v3.8h, v3.8h, v19.8h
    mul             v4.8h, v4.8h, v20.8h
    mul             v5.8h, v5.8h, v21.8h
    add             TMP4, xzr, TMP2, LSL #32
    mul             v6.8h, v6.8h, v22.8h
    mul             v7.8h, v7.8h, v23.8h
    adds            TMP3, xzr, TMP2, LSR #32
    mul             v8.8h, v8.8h, v24.8h
    mul             v9.8h, v9.8h, v25.8h
    b.ne            3f
    /* Right AC coef is zero */
    dup             v15.2d, v10.d[1]
    /* Even part: reverse the even part of the forward DCT. */
    add             v18.4h, v4.4h, v8.4h           /* z2 + z3 = DEQUANTIZE(inptr[DCTSIZE*2], quantptr[DCTSIZE*2]) + DEQUANTIZE(inptr[DCTSIZE*6], quantptr[DCTSIZE*6]) */
    add             v22.4h, v2.4h, v6.4h           /* z2 + z3 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) + DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]) */
    sub             v26.4h, v2.4h, v6.4h           /* z2 - z3 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) - DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]) */
    smull           v18.4s, v18.4h, XFIX_P_0_541   /* z1l z1 = MULTIPLY(z2 + z3, FIX_0_541196100); */
    sshll           v22.4s, v22.4h, #(CONST_BITS)  /* tmp0l tmp0 = LEFT_SHIFT(z2 + z3, CONST_BITS); */
    mov             v20.16b, v18.16b               /* tmp3 = z1 */
    sshll           v26.4s, v26.4h, #(CONST_BITS)  /* tmp1l tmp1 = LEFT_SHIFT(z2 - z3, CONST_BITS); */
    smlal           v18.4s, v8.4h, XFIX_N_1_847    /* tmp2l tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065); */
    smlal           v20.4s, v4.4h, XFIX_P_0_765    /* tmp3l tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865); */
    add             v2.4s, v22.4s, v20.4s          /* tmp10l tmp10 = tmp0 + tmp3; */
    sub             v6.4s, v22.4s, v20.4s          /* tmp13l tmp13 = tmp0 - tmp3; */
    add             v8.4s, v26.4s, v18.4s          /* tmp11l tmp11 = tmp1 + tmp2; */
    sub             v4.4s, v26.4s, v18.4s          /* tmp12l tmp12 = tmp1 - tmp2; */

    /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

    add             v22.4h, v9.4h, v5.4h    /* z3 = tmp0 + tmp2 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]) + DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]) */
    add             v24.4h, v7.4h, v3.4h    /* z4 = tmp1 + tmp3 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]) + DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]) */
    add             v18.4h, v9.4h, v3.4h    /* z1 = tmp0 + tmp3 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]) + DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]) */
    add             v20.4h, v7.4h, v5.4h    /* z2 = tmp1 + tmp2 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]) + DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]) */
    add             v26.4h, v22.4h, v24.4h  /* z5 = z3 + z4 */

    smull           v10.4s, v9.4h, XFIX_P_0_298   /* tmp0 = MULTIPLY(tmp0, FIX_0_298631336) */
    smull           v12.4s, v7.4h, XFIX_P_2_053   /* tmp1 = MULTIPLY(tmp1, FIX_2_053119869) */
    smull           v14.4s, v5.4h, XFIX_P_3_072   /* tmp2 = MULTIPLY(tmp2, FIX_3_072711026) */
    smull           v16.4s, v3.4h, XFIX_P_1_501   /* tmp3 = MULTIPLY(tmp3, FIX_1_501321110) */
    smull           v26.4s, v26.4h, XFIX_P_1_175  /* z5l z5 = MULTIPLY(z3 + z4, FIX_1_175875602) */
    smull           v22.4s, v22.4h, XFIX_N_1_961  /* z3 = MULTIPLY(z3, -FIX_1_961570560) */
    smull           v24.4s, v24.4h, XFIX_N_0_390  /* z4 = MULTIPLY(z4, -FIX_0_390180644) */
    smull           v18.4s, v18.4h, XFIX_N_0_899  /* z1 = MULTIPLY(z1, -FIX_0_899976223) */
    smull           v20.4s, v20.4h, XFIX_N_2_562  /* z2 = MULTIPLY(z2, -FIX_2_562915447) */

    add             v22.4s, v22.4s, v26.4s  /* z3 += z5 */
    add             v24.4s, v24.4s, v26.4s  /* z4 += z5 */

    add             v10.4s, v10.4s, v18.4s  /* tmp0 += z1 */
    add             v12.4s, v12.4s, v20.4s  /* tmp1 += z2 */
    add             v14.4s, v14.4s, v20.4s  /* tmp2 += z2 */
    add             v16.4s, v16.4s, v18.4s  /* tmp3 += z1 */

    add             v10.4s, v10.4s, v22.4s  /* tmp0 += z3 */
    add             v12.4s, v12.4s, v24.4s  /* tmp1 += z4 */
    add             v16.4s, v16.4s, v24.4s  /* tmp3 += z4 */
    add             v14.4s, v14.4s, v22.4s  /* tmp2 += z3 */

    /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

    add             v18.4s, v2.4s, v16.4s  /* tmp10 + tmp3 */
    sub             v20.4s, v2.4s, v16.4s  /* tmp10 - tmp3 */
    add             v22.4s, v8.4s, v14.4s  /* tmp11 + tmp2 */
    sub             v24.4s, v8.4s, v14.4s  /* tmp11 - tmp2 */
    add             v26.4s, v4.4s, v12.4s  /* tmp12 + tmp1 */
    sub             v28.4s, v4.4s, v12.4s  /* tmp12 - tmp1 */
    add             v14.4s, v6.4s, v10.4s  /* tmp13 + tmp0 */
    sub             v16.4s, v6.4s, v10.4s  /* tmp13 - tmp0 */

    rshrn           v2.4h, v18.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*0] = (int)DESCALE(tmp10 + tmp3, CONST_BITS-PASS1_BITS) */
    rshrn           v3.4h, v22.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*1] = (int)DESCALE(tmp11 + tmp2, CONST_BITS-PASS1_BITS) */
    rshrn           v4.4h, v26.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*2] = (int)DESCALE(tmp12 + tmp1, CONST_BITS-PASS1_BITS) */
    rshrn           v5.4h, v14.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*3] = (int)DESCALE(tmp13 + tmp0, CONST_BITS-PASS1_BITS) */
    rshrn2          v2.8h, v16.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*4] = (int)DESCALE(tmp13 - tmp0, CONST_BITS-PASS1_BITS) */
    rshrn2          v3.8h, v28.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*5] = (int)DESCALE(tmp12 - tmp1, CONST_BITS-PASS1_BITS) */
    rshrn2          v4.8h, v24.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*6] = (int)DESCALE(tmp11 - tmp2, CONST_BITS-PASS1_BITS) */
    rshrn2          v5.8h, v20.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*7] = (int)DESCALE(tmp10 - tmp3, CONST_BITS-PASS1_BITS) */
    mov             v6.16b, v15.16b
    mov             v7.16b, v15.16b
    mov             v8.16b, v15.16b
    mov             v9.16b, v15.16b
    b               1b

.balign 16
3:
    cbnz            TMP4, 4f
    /* Left AC coef is zero */
    dup             v14.2d, v10.d[0]
    /* Even part: reverse the even part of the forward DCT. */
    add             v18.8h, v4.8h, v8.8h           /* z2 + z3 = DEQUANTIZE(inptr[DCTSIZE*2], quantptr[DCTSIZE*2]) + DEQUANTIZE(inptr[DCTSIZE*6], quantptr[DCTSIZE*6]) */
    add             v22.8h, v2.8h, v6.8h           /* z2 + z3 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) + DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]) */
    smull2          v19.4s, v18.8h, XFIX_P_0_541   /* z1h z1 = MULTIPLY(z2 + z3, FIX_0_541196100); */
    sub             v26.8h, v2.8h, v6.8h           /* z2 - z3 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) - DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]) */
    sshll2          v23.4s, v22.8h, #(CONST_BITS)  /* tmp0h tmp0 = LEFT_SHIFT(z2 + z3, CONST_BITS); */
    mov             v21.16b, v19.16b               /* tmp3 = z1 */
    smlal2          v19.4s, v8.8h, XFIX_N_1_847    /* tmp2h tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065); */
    sshll2          v27.4s, v26.8h, #(CONST_BITS)  /* tmp1h tmp1 = LEFT_SHIFT(z2 - z3, CONST_BITS); */
    smlal2          v21.4s, v4.8h, XFIX_P_0_765    /* tmp3h tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865); */
    add             v28.4s, v23.4s, v21.4s         /* tmp10h tmp10 = tmp0 + tmp3; */
    sub             v31.4s, v23.4s, v21.4s         /* tmp13h tmp13 = tmp0 - tmp3; */
    add             v29.4s, v27.4s, v19.4s         /* tmp11h tmp11 = tmp1 + tmp2; */
    sub             v30.4s, v27.4s, v19.4s         /* tmp12h tmp12 = tmp1 - tmp2; */

    /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

    add             v22.8h, v9.8h, v5.8h    /* z3 = tmp0 + tmp2 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]) + DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]) */
    add             v24.8h, v7.8h, v3.8h    /* z4 = tmp1 + tmp3 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]) + DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]) */
    add             v18.8h, v9.8h, v3.8h    /* z1 = tmp0 + tmp3 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]) + DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]) */
    add             v20.8h, v7.8h, v5.8h    /* z2 = tmp1 + tmp2 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]) + DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]) */
    add             v26.8h, v22.8h, v24.8h  /* z5 = z3 + z4 */

    smull2          v11.4s, v9.8h, XFIX_P_0_298   /* tmp0 = MULTIPLY(tmp0, FIX_0_298631336) */
    smull2          v13.4s, v7.8h, XFIX_P_2_053   /* tmp1 = MULTIPLY(tmp1, FIX_2_053119869) */
    smull2          v15.4s, v5.8h, XFIX_P_3_072   /* tmp2 = MULTIPLY(tmp2, FIX_3_072711026) */
    smull2          v17.4s, v3.8h, XFIX_P_1_501   /* tmp3 = MULTIPLY(tmp3, FIX_1_501321110) */
    smull2          v27.4s, v26.8h, XFIX_P_1_175  /* z5h z5 = MULTIPLY(z3 + z4, FIX_1_175875602) */
    smull2          v23.4s, v22.8h, XFIX_N_1_961  /* z3 = MULTIPLY(z3, -FIX_1_961570560) */
    smull2          v25.4s, v24.8h, XFIX_N_0_390  /* z4 = MULTIPLY(z4, -FIX_0_390180644) */
    smull2          v19.4s, v18.8h, XFIX_N_0_899  /* z1 = MULTIPLY(z1, -FIX_0_899976223) */
    smull2          v21.4s, v20.8h, XFIX_N_2_562  /* z2 = MULTIPLY(z2, -FIX_2_562915447) */

    add             v23.4s, v23.4s, v27.4s  /* z3 += z5 */
    add             v22.4s, v22.4s, v26.4s  /* z3 += z5 */
    add             v25.4s, v25.4s, v27.4s  /* z4 += z5 */
    add             v24.4s, v24.4s, v26.4s  /* z4 += z5 */

    add             v11.4s, v11.4s, v19.4s  /* tmp0 += z1 */
    add             v13.4s, v13.4s, v21.4s  /* tmp1 += z2 */
    add             v15.4s, v15.4s, v21.4s  /* tmp2 += z2 */
    add             v17.4s, v17.4s, v19.4s  /* tmp3 += z1 */

    add             v11.4s, v11.4s, v23.4s  /* tmp0 += z3 */
    add             v13.4s, v13.4s, v25.4s  /* tmp1 += z4 */
    add             v17.4s, v17.4s, v25.4s  /* tmp3 += z4 */
    add             v15.4s, v15.4s, v23.4s  /* tmp2 += z3 */

    /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

    add             v19.4s, v28.4s, v17.4s  /* tmp10 + tmp3 */
    sub             v21.4s, v28.4s, v17.4s  /* tmp10 - tmp3 */
    add             v23.4s, v29.4s, v15.4s  /* tmp11 + tmp2 */
    sub             v25.4s, v29.4s, v15.4s  /* tmp11 - tmp2 */
    add             v27.4s, v30.4s, v13.4s  /* tmp12 + tmp1 */
    sub             v29.4s, v30.4s, v13.4s  /* tmp12 - tmp1 */
    add             v15.4s, v31.4s, v11.4s  /* tmp13 + tmp0 */
    sub             v17.4s, v31.4s, v11.4s  /* tmp13 - tmp0 */

    mov             v2.16b, v14.16b
    mov             v3.16b, v14.16b
    mov             v4.16b, v14.16b
    mov             v5.16b, v14.16b
    rshrn           v6.4h, v19.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*0] = (int)DESCALE(tmp10 + tmp3, CONST_BITS-PASS1_BITS) */
    rshrn           v7.4h, v23.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*1] = (int)DESCALE(tmp11 + tmp2, CONST_BITS-PASS1_BITS) */
    rshrn           v8.4h, v27.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*2] = (int)DESCALE(tmp12 + tmp1, CONST_BITS-PASS1_BITS) */
    rshrn           v9.4h, v15.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*3] = (int)DESCALE(tmp13 + tmp0, CONST_BITS-PASS1_BITS) */
    rshrn2          v6.8h, v17.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*4] = (int)DESCALE(tmp13 - tmp0, CONST_BITS-PASS1_BITS) */
    rshrn2          v7.8h, v29.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*5] = (int)DESCALE(tmp12 - tmp1, CONST_BITS-PASS1_BITS) */
    rshrn2          v8.8h, v25.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*6] = (int)DESCALE(tmp11 - tmp2, CONST_BITS-PASS1_BITS) */
    rshrn2          v9.8h, v21.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*7] = (int)DESCALE(tmp10 - tmp3, CONST_BITS-PASS1_BITS) */
    b               1b

.balign 16
4:
    /* "No" AC coef is zero */
    /* Even part: reverse the even part of the forward DCT. */
    add             v18.8h, v4.8h, v8.8h           /* z2 + z3 = DEQUANTIZE(inptr[DCTSIZE*2], quantptr[DCTSIZE*2]) + DEQUANTIZE(inptr[DCTSIZE*6], quantptr[DCTSIZE*6]) */
    add             v22.8h, v2.8h, v6.8h           /* z2 + z3 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) + DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]) */
    smull2          v19.4s, v18.8h, XFIX_P_0_541   /* z1h z1 = MULTIPLY(z2 + z3, FIX_0_541196100); */
    sub             v26.8h, v2.8h, v6.8h           /* z2 - z3 = DEQUANTIZE(inptr[DCTSIZE*0], quantptr[DCTSIZE*0]) - DEQUANTIZE(inptr[DCTSIZE*4], quantptr[DCTSIZE*4]) */
    smull           v18.4s, v18.4h, XFIX_P_0_541   /* z1l z1 = MULTIPLY(z2 + z3, FIX_0_541196100); */
    sshll2          v23.4s, v22.8h, #(CONST_BITS)  /* tmp0h tmp0 = LEFT_SHIFT(z2 + z3, CONST_BITS); */
    mov             v21.16b, v19.16b               /* tmp3 = z1 */
    mov             v20.16b, v18.16b               /* tmp3 = z1 */
    smlal2          v19.4s, v8.8h, XFIX_N_1_847    /* tmp2h tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065); */
    smlal           v18.4s, v8.4h, XFIX_N_1_847    /* tmp2l tmp2 = z1 + MULTIPLY(z3, -FIX_1_847759065); */
    sshll2          v27.4s, v26.8h, #(CONST_BITS)  /* tmp1h tmp1 = LEFT_SHIFT(z2 - z3, CONST_BITS); */
    smlal2          v21.4s, v4.8h, XFIX_P_0_765    /* tmp3h tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865); */
    smlal           v20.4s, v4.4h, XFIX_P_0_765    /* tmp3l tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865); */
    sshll           v22.4s, v22.4h, #(CONST_BITS)  /* tmp0l tmp0 = LEFT_SHIFT(z2 + z3, CONST_BITS); */
    sshll           v26.4s, v26.4h, #(CONST_BITS)  /* tmp1l tmp1 = LEFT_SHIFT(z2 - z3, CONST_BITS); */
    add             v2.4s, v22.4s, v20.4s          /* tmp10l tmp10 = tmp0 + tmp3; */
    sub             v6.4s, v22.4s, v20.4s          /* tmp13l tmp13 = tmp0 - tmp3; */
    add             v8.4s, v26.4s, v18.4s          /* tmp11l tmp11 = tmp1 + tmp2; */
    sub             v4.4s, v26.4s, v18.4s          /* tmp12l tmp12 = tmp1 - tmp2; */
    add             v28.4s, v23.4s, v21.4s         /* tmp10h tmp10 = tmp0 + tmp3; */
    sub             v31.4s, v23.4s, v21.4s         /* tmp13h tmp13 = tmp0 - tmp3; */
    add             v29.4s, v27.4s, v19.4s         /* tmp11h tmp11 = tmp1 + tmp2; */
    sub             v30.4s, v27.4s, v19.4s         /* tmp12h tmp12 = tmp1 - tmp2; */

    /* Odd part per figure 8; the matrix is unitary and hence its
     * transpose is its inverse.  i0..i3 are y7,y5,y3,y1 respectively.
     */

    add             v22.8h, v9.8h, v5.8h    /* z3 = tmp0 + tmp2 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]) + DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]) */
    add             v24.8h, v7.8h, v3.8h    /* z4 = tmp1 + tmp3 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]) + DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]) */
    add             v18.8h, v9.8h, v3.8h    /* z1 = tmp0 + tmp3 = DEQUANTIZE(inptr[DCTSIZE*7], quantptr[DCTSIZE*7]) + DEQUANTIZE(inptr[DCTSIZE*1], quantptr[DCTSIZE*1]) */
    add             v20.8h, v7.8h, v5.8h    /* z2 = tmp1 + tmp2 = DEQUANTIZE(inptr[DCTSIZE*5], quantptr[DCTSIZE*5]) + DEQUANTIZE(inptr[DCTSIZE*3], quantptr[DCTSIZE*3]) */
    add             v26.8h, v22.8h, v24.8h  /* z5 = z3 + z4 */

    smull2          v11.4s, v9.8h, XFIX_P_0_298   /* tmp0 = MULTIPLY(tmp0, FIX_0_298631336) */
    smull2          v13.4s, v7.8h, XFIX_P_2_053   /* tmp1 = MULTIPLY(tmp1, FIX_2_053119869) */
    smull2          v15.4s, v5.8h, XFIX_P_3_072   /* tmp2 = MULTIPLY(tmp2, FIX_3_072711026) */
    smull2          v17.4s, v3.8h, XFIX_P_1_501   /* tmp3 = MULTIPLY(tmp3, FIX_1_501321110) */
    smull2          v27.4s, v26.8h, XFIX_P_1_175  /* z5h z5 = MULTIPLY(z3 + z4, FIX_1_175875602) */
    smull2          v23.4s, v22.8h, XFIX_N_1_961  /* z3 = MULTIPLY(z3, -FIX_1_961570560) */
    smull2          v25.4s, v24.8h, XFIX_N_0_390  /* z4 = MULTIPLY(z4, -FIX_0_390180644) */
    smull2          v19.4s, v18.8h, XFIX_N_0_899  /* z1 = MULTIPLY(z1, -FIX_0_899976223) */
    smull2          v21.4s, v20.8h, XFIX_N_2_562  /* z2 = MULTIPLY(z2, -FIX_2_562915447) */

    smull           v10.4s, v9.4h, XFIX_P_0_298   /* tmp0 = MULTIPLY(tmp0, FIX_0_298631336) */
    smull           v12.4s, v7.4h, XFIX_P_2_053   /* tmp1 = MULTIPLY(tmp1, FIX_2_053119869) */
    smull           v14.4s, v5.4h, XFIX_P_3_072   /* tmp2 = MULTIPLY(tmp2, FIX_3_072711026) */
    smull           v16.4s, v3.4h, XFIX_P_1_501   /* tmp3 = MULTIPLY(tmp3, FIX_1_501321110) */
    smull           v26.4s, v26.4h, XFIX_P_1_175  /* z5l z5 = MULTIPLY(z3 + z4, FIX_1_175875602) */
    smull           v22.4s, v22.4h, XFIX_N_1_961  /* z3 = MULTIPLY(z3, -FIX_1_961570560) */
    smull           v24.4s, v24.4h, XFIX_N_0_390  /* z4 = MULTIPLY(z4, -FIX_0_390180644) */
    smull           v18.4s, v18.4h, XFIX_N_0_899  /* z1 = MULTIPLY(z1, -FIX_0_899976223) */
    smull           v20.4s, v20.4h, XFIX_N_2_562  /* z2 = MULTIPLY(z2, -FIX_2_562915447) */

    add             v23.4s, v23.4s, v27.4s  /* z3 += z5 */
    add             v22.4s, v22.4s, v26.4s  /* z3 += z5 */
    add             v25.4s, v25.4s, v27.4s  /* z4 += z5 */
    add             v24.4s, v24.4s, v26.4s  /* z4 += z5 */

    add             v11.4s, v11.4s, v19.4s  /* tmp0 += z1 */
    add             v10.4s, v10.4s, v18.4s  /* tmp0 += z1 */
    add             v13.4s, v13.4s, v21.4s  /* tmp1 += z2 */
    add             v12.4s, v12.4s, v20.4s  /* tmp1 += z2 */
    add             v15.4s, v15.4s, v21.4s  /* tmp2 += z2 */
    add             v14.4s, v14.4s, v20.4s  /* tmp2 += z2 */
    add             v17.4s, v17.4s, v19.4s  /* tmp3 += z1 */
    add             v16.4s, v16.4s, v18.4s  /* tmp3 += z1 */

    add             v11.4s, v11.4s, v23.4s  /* tmp0 += z3 */
    add             v10.4s, v10.4s, v22.4s  /* tmp0 += z3 */
    add             v13.4s, v13.4s, v25.4s  /* tmp1 += z4 */
    add             v12.4s, v12.4s, v24.4s  /* tmp1 += z4 */
    add             v17.4s, v17.4s, v25.4s  /* tmp3 += z4 */
    add             v16.4s, v16.4s, v24.4s  /* tmp3 += z4 */
    add             v15.4s, v15.4s, v23.4s  /* tmp2 += z3 */
    add             v14.4s, v14.4s, v22.4s  /* tmp2 += z3 */

    /* Final output stage: inputs are tmp10..tmp13, tmp0..tmp3 */

    add             v18.4s, v2.4s, v16.4s   /* tmp10 + tmp3 */
    add             v19.4s, v28.4s, v17.4s  /* tmp10 + tmp3 */
    sub             v20.4s, v2.4s, v16.4s   /* tmp10 - tmp3 */
    sub             v21.4s, v28.4s, v17.4s  /* tmp10 - tmp3 */
    add             v22.4s, v8.4s, v14.4s   /* tmp11 + tmp2 */
    add             v23.4s, v29.4s, v15.4s  /* tmp11 + tmp2 */
    sub             v24.4s, v8.4s, v14.4s   /* tmp11 - tmp2 */
    sub             v25.4s, v29.4s, v15.4s  /* tmp11 - tmp2 */
    add             v26.4s, v4.4s, v12.4s   /* tmp12 + tmp1 */
    add             v27.4s, v30.4s, v13.4s  /* tmp12 + tmp1 */
    sub             v28.4s, v4.4s, v12.4s   /* tmp12 - tmp1 */
    sub             v29.4s, v30.4s, v13.4s  /* tmp12 - tmp1 */
    add             v14.4s, v6.4s, v10.4s   /* tmp13 + tmp0 */
    add             v15.4s, v31.4s, v11.4s  /* tmp13 + tmp0 */
    sub             v16.4s, v6.4s, v10.4s   /* tmp13 - tmp0 */
    sub             v17.4s, v31.4s, v11.4s  /* tmp13 - tmp0 */

    rshrn           v2.4h, v18.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*0] = (int)DESCALE(tmp10 + tmp3, CONST_BITS-PASS1_BITS) */
    rshrn           v3.4h, v22.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*1] = (int)DESCALE(tmp11 + tmp2, CONST_BITS-PASS1_BITS) */
    rshrn           v4.4h, v26.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*2] = (int)DESCALE(tmp12 + tmp1, CONST_BITS-PASS1_BITS) */
    rshrn           v5.4h, v14.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*3] = (int)DESCALE(tmp13 + tmp0, CONST_BITS-PASS1_BITS) */
    rshrn           v6.4h, v19.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*0] = (int)DESCALE(tmp10 + tmp3, CONST_BITS-PASS1_BITS) */
    rshrn           v7.4h, v23.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*1] = (int)DESCALE(tmp11 + tmp2, CONST_BITS-PASS1_BITS) */
    rshrn           v8.4h, v27.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*2] = (int)DESCALE(tmp12 + tmp1, CONST_BITS-PASS1_BITS) */
    rshrn           v9.4h, v15.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*3] = (int)DESCALE(tmp13 + tmp0, CONST_BITS-PASS1_BITS) */
    rshrn2          v2.8h, v16.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*4] = (int)DESCALE(tmp13 - tmp0, CONST_BITS-PASS1_BITS) */
    rshrn2          v3.8h, v28.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*5] = (int)DESCALE(tmp12 - tmp1, CONST_BITS-PASS1_BITS) */
    rshrn2          v4.8h, v24.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*6] = (int)DESCALE(tmp11 - tmp2, CONST_BITS-PASS1_BITS) */
    rshrn2          v5.8h, v20.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*7] = (int)DESCALE(tmp10 - tmp3, CONST_BITS-PASS1_BITS) */
    rshrn2          v6.8h, v17.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*4] = (int)DESCALE(tmp13 - tmp0, CONST_BITS-PASS1_BITS) */
    rshrn2          v7.8h, v29.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*5] = (int)DESCALE(tmp12 - tmp1, CONST_BITS-PASS1_BITS) */
    rshrn2          v8.8h, v25.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*6] = (int)DESCALE(tmp11 - tmp2, CONST_BITS-PASS1_BITS) */
    rshrn2          v9.8h, v21.4s, #(CONST_BITS - PASS1_BITS)  /* wsptr[DCTSIZE*7] = (int)DESCALE(tmp10 - tmp3, CONST_BITS-PASS1_BITS) */
    b               1b

    .unreq          DCT_TABLE
    .unreq          COEF_BLOCK
    .unreq          OUTPUT_BUF
    .unreq          OUTPUT_COL
    .unreq          TMP1
    .unreq          TMP2
    .unreq          TMP3
    .unreq          TMP4
    .unreq          TMP5
    .unreq          TMP6
    .unreq          TMP7
    .unreq          TMP8

#undef CENTERJSAMPLE
#undef CONST_BITS
#undef PASS1_BITS
#undef XFIX_P_0_298
#undef XFIX_N_0_390
#undef XFIX_P_0_541
#undef XFIX_P_0_765
#undef XFIX_N_0_899
#undef XFIX_P_1_175
#undef XFIX_P_1_501
#undef XFIX_N_1_847
#undef XFIX_N_1_961
#undef XFIX_P_2_053
#undef XFIX_N_2_562
#undef XFIX_P_3_072


/*****************************************************************************/

/*
 * jsimd_ycc_extrgb_convert_neon
 * jsimd_ycc_extbgr_convert_neon
 * jsimd_ycc_extrgbx_convert_neon
 * jsimd_ycc_extbgrx_convert_neon
 * jsimd_ycc_extxbgr_convert_neon
 * jsimd_ycc_extxrgb_convert_neon
 *
 * Colorspace conversion YCbCr -> RGB
 */

.macro do_load size
  .if \size == 8
    ld1             {v4.8b}, [U], 8
    ld1             {v5.8b}, [V], 8
    ld1             {v0.8b}, [Y], 8
    prfm            pldl1keep, [U, #64]
    prfm            pldl1keep, [V, #64]
    prfm            pldl1keep, [Y, #64]
  .elseif \size == 4
    ld1             {v4.b}[0], [U], 1
    ld1             {v4.b}[1], [U], 1
    ld1             {v4.b}[2], [U], 1
    ld1             {v4.b}[3], [U], 1
    ld1             {v5.b}[0], [V], 1
    ld1             {v5.b}[1], [V], 1
    ld1             {v5.b}[2], [V], 1
    ld1             {v5.b}[3], [V], 1
    ld1             {v0.b}[0], [Y], 1
    ld1             {v0.b}[1], [Y], 1
    ld1             {v0.b}[2], [Y], 1
    ld1             {v0.b}[3], [Y], 1
  .elseif \size == 2
    ld1             {v4.b}[4], [U], 1
    ld1             {v4.b}[5], [U], 1
    ld1             {v5.b}[4], [V], 1
    ld1             {v5.b}[5], [V], 1
    ld1             {v0.b}[4], [Y], 1
    ld1             {v0.b}[5], [Y], 1
  .elseif \size == 1
    ld1             {v4.b}[6], [U], 1
    ld1             {v5.b}[6], [V], 1
    ld1             {v0.b}[6], [Y], 1
  .else
    .error unsupported macroblock size
  .endif
.endm

.macro do_store bpp, size, fast_st3
  .if \bpp == 24
    .if \size == 8
      .if \fast_st3 == 1
        st3         {v10.8b, v11.8b, v12.8b}, [RGB], 24
      .else
        st1         {v10.b}[0], [RGB], #1
        st1         {v11.b}[0], [RGB], #1
        st1         {v12.b}[0], [RGB], #1

        st1         {v10.b}[1], [RGB], #1
        st1         {v11.b}[1], [RGB], #1
        st1         {v12.b}[1], [RGB], #1

        st1         {v10.b}[2], [RGB], #1
        st1         {v11.b}[2], [RGB], #1
        st1         {v12.b}[2], [RGB], #1

        st1         {v10.b}[3], [RGB], #1
        st1         {v11.b}[3], [RGB], #1
        st1         {v12.b}[3], [RGB], #1

        st1         {v10.b}[4], [RGB], #1
        st1         {v11.b}[4], [RGB], #1
        st1         {v12.b}[4], [RGB], #1

        st1         {v10.b}[5], [RGB], #1
        st1         {v11.b}[5], [RGB], #1
        st1         {v12.b}[5], [RGB], #1

        st1         {v10.b}[6], [RGB], #1
        st1         {v11.b}[6], [RGB], #1
        st1         {v12.b}[6], [RGB], #1

        st1         {v10.b}[7], [RGB], #1
        st1         {v11.b}[7], [RGB], #1
        st1         {v12.b}[7], [RGB], #1
      .endif
    .elseif \size == 4
      st3           {v10.b, v11.b, v12.b}[0], [RGB], 3
      st3           {v10.b, v11.b, v12.b}[1], [RGB], 3
      st3           {v10.b, v11.b, v12.b}[2], [RGB], 3
      st3           {v10.b, v11.b, v12.b}[3], [RGB], 3
    .elseif \size == 2
      st3           {v10.b, v11.b, v12.b}[4], [RGB], 3
      st3           {v10.b, v11.b, v12.b}[5], [RGB], 3
    .elseif \size == 1
      st3           {v10.b, v11.b, v12.b}[6], [RGB], 3
    .else
     .error unsupported macroblock size
    .endif
  .elseif \bpp == 32
    .if \size == 8
      st4           {v10.8b, v11.8b, v12.8b, v13.8b}, [RGB], 32
    .elseif \size == 4
      st4           {v10.b, v11.b, v12.b, v13.b}[0], [RGB], 4
      st4           {v10.b, v11.b, v12.b, v13.b}[1], [RGB], 4
      st4           {v10.b, v11.b, v12.b, v13.b}[2], [RGB], 4
      st4           {v10.b, v11.b, v12.b, v13.b}[3], [RGB], 4
    .elseif \size == 2
      st4           {v10.b, v11.b, v12.b, v13.b}[4], [RGB], 4
      st4           {v10.b, v11.b, v12.b, v13.b}[5], [RGB], 4
    .elseif \size == 1
      st4           {v10.b, v11.b, v12.b, v13.b}[6], [RGB], 4
    .else
      .error unsupported macroblock size
    .endif
  .elseif \bpp == 16
    .if \size == 8
      st1           {v25.8h}, [RGB], 16
    .elseif \size == 4
      st1           {v25.4h}, [RGB], 8
    .elseif \size == 2
      st1           {v25.h}[4], [RGB], 2
      st1           {v25.h}[5], [RGB], 2
    .elseif \size == 1
      st1           {v25.h}[6], [RGB], 2
    .else
      .error unsupported macroblock size
    .endif
  .else
    .error unsupported bpp
  .endif
.endm

.macro generate_jsimd_ycc_rgb_convert_neon colorid, bpp, r_offs, rsize, \
                                           g_offs, gsize, b_offs, bsize, \
                                           defsize, fast_st3

/*
 * 2-stage pipelined YCbCr->RGB conversion
 */

.macro do_yuv_to_rgb_stage1
    uaddw           v6.8h, v2.8h, v4.8b     /* q3 = u - 128 */
    uaddw           v8.8h, v2.8h, v5.8b     /* q2 = v - 128 */
    smull           v20.4s, v6.4h, v1.h[1]  /* multiply by -11277 */
    smlal           v20.4s, v8.4h, v1.h[2]  /* multiply by -23401 */
    smull2          v22.4s, v6.8h, v1.h[1]  /* multiply by -11277 */
    smlal2          v22.4s, v8.8h, v1.h[2]  /* multiply by -23401 */
    smull           v24.4s, v8.4h, v1.h[0]  /* multiply by 22971 */
    smull2          v26.4s, v8.8h, v1.h[0]  /* multiply by 22971 */
    smull           v28.4s, v6.4h, v1.h[3]  /* multiply by 29033 */
    smull2          v30.4s, v6.8h, v1.h[3]  /* multiply by 29033 */
.endm

.macro do_yuv_to_rgb_stage2
    rshrn           v20.4h, v20.4s, #15
    rshrn2          v20.8h, v22.4s, #15
    rshrn           v24.4h, v24.4s, #14
    rshrn2          v24.8h, v26.4s, #14
    rshrn           v28.4h, v28.4s, #14
    rshrn2          v28.8h, v30.4s, #14
    uaddw           v20.8h, v20.8h, v0.8b
    uaddw           v24.8h, v24.8h, v0.8b
    uaddw           v28.8h, v28.8h, v0.8b
  .if \bpp != 16
    sqxtun          v1\g_offs\defsize, v20.8h
    sqxtun          v1\r_offs\defsize, v24.8h
    sqxtun          v1\b_offs\defsize, v28.8h
  .else
    sqshlu          v21.8h, v20.8h, #8
    sqshlu          v25.8h, v24.8h, #8
    sqshlu          v29.8h, v28.8h, #8
    sri             v25.8h, v21.8h, #5
    sri             v25.8h, v29.8h, #11
  .endif
.endm

.macro do_yuv_to_rgb_stage2_store_load_stage1 fast_st3
    rshrn           v20.4h, v20.4s, #15
    rshrn           v24.4h, v24.4s, #14
    rshrn           v28.4h, v28.4s, #14
    ld1             {v4.8b}, [U], 8
    rshrn2          v20.8h, v22.4s, #15
    rshrn2          v24.8h, v26.4s, #14
    rshrn2          v28.8h, v30.4s, #14
    ld1             {v5.8b}, [V], 8
    uaddw           v20.8h, v20.8h, v0.8b
    uaddw           v24.8h, v24.8h, v0.8b
    uaddw           v28.8h, v28.8h, v0.8b
  .if \bpp != 16  /**************** rgb24/rgb32 ******************************/
    sqxtun          v1\g_offs\defsize, v20.8h
    ld1             {v0.8b}, [Y], 8
    sqxtun          v1\r_offs\defsize, v24.8h
    prfm            pldl1keep, [U, #64]
    prfm            pldl1keep, [V, #64]
    prfm            pldl1keep, [Y, #64]
    sqxtun          v1\b_offs\defsize, v28.8h
    uaddw           v6.8h, v2.8h, v4.8b     /* v6.16b = u - 128 */
    uaddw           v8.8h, v2.8h, v5.8b     /* q2 = v - 128 */
    smull           v20.4s, v6.4h, v1.h[1]  /* multiply by -11277 */
    smlal           v20.4s, v8.4h, v1.h[2]  /* multiply by -23401 */
    smull2          v22.4s, v6.8h, v1.h[1]  /* multiply by -11277 */
    smlal2          v22.4s, v8.8h, v1.h[2]  /* multiply by -23401 */
    smull           v24.4s, v8.4h, v1.h[0]  /* multiply by 22971 */
    smull2          v26.4s, v8.8h, v1.h[0]  /* multiply by 22971 */
  .else  /**************************** rgb565 ********************************/
    sqshlu          v21.8h, v20.8h, #8
    sqshlu          v25.8h, v24.8h, #8
    sqshlu          v29.8h, v28.8h, #8
    uaddw           v6.8h, v2.8h, v4.8b     /* v6.16b = u - 128 */
    uaddw           v8.8h, v2.8h, v5.8b     /* q2 = v - 128 */
    ld1             {v0.8b}, [Y], 8
    smull           v20.4s, v6.4h, v1.h[1]  /* multiply by -11277 */
    smlal           v20.4s, v8.4h, v1.h[2]  /* multiply by -23401 */
    smull2          v22.4s, v6.8h, v1.h[1]  /* multiply by -11277 */
    smlal2          v22.4s, v8.8h, v1.h[2]  /* multiply by -23401 */
    sri             v25.8h, v21.8h, #5
    smull           v24.4s, v8.4h, v1.h[0]  /* multiply by 22971 */
    smull2          v26.4s, v8.8h, v1.h[0]  /* multiply by 22971 */
    prfm            pldl1keep, [U, #64]
    prfm            pldl1keep, [V, #64]
    prfm            pldl1keep, [Y, #64]
    sri             v25.8h, v29.8h, #11
  .endif
    do_store        \bpp, 8, \fast_st3
    smull           v28.4s, v6.4h, v1.h[3]  /* multiply by 29033 */
    smull2          v30.4s, v6.8h, v1.h[3]  /* multiply by 29033 */
.endm

.macro do_yuv_to_rgb
    do_yuv_to_rgb_stage1
    do_yuv_to_rgb_stage2
.endm

.if \fast_st3 == 1
asm_function jsimd_ycc_\colorid\()_convert_neon
.else
asm_function jsimd_ycc_\colorid\()_convert_neon_slowst3
.endif
    OUTPUT_WIDTH    .req w0
    INPUT_BUF       .req x1
    INPUT_ROW       .req w2
    OUTPUT_BUF      .req x3
    NUM_ROWS        .req w4

    INPUT_BUF0      .req x5
    INPUT_BUF1      .req x6
    INPUT_BUF2      .req x1

    RGB             .req x7
    Y               .req x9
    U               .req x10
    V               .req x11
    N               .req w15

    sub             sp, sp, 64
    mov             x9, sp

    /* Load constants to d1, d2, d3 (v0.4h is just used for padding) */
    get_symbol_loc  x15, Ljsimd_ycc_rgb_neon_consts

    /* Save Neon registers */
    st1             {v8.8b, v9.8b, v10.8b, v11.8b}, [x9], 32
    st1             {v12.8b, v13.8b, v14.8b, v15.8b}, [x9], 32
    ld1             {v0.4h, v1.4h}, [x15], 16
    ld1             {v2.8h}, [x15]

    ldr             INPUT_BUF0, [INPUT_BUF]
    ldr             INPUT_BUF1, [INPUT_BUF, #8]
    ldr             INPUT_BUF2, [INPUT_BUF, #16]
    .unreq          INPUT_BUF

    /* Initially set v10, v11.4h, v12.8b, d13 to 0xFF */
    movi            v10.16b, #255
    movi            v13.16b, #255

    /* Outer loop over scanlines */
    cmp             NUM_ROWS, #1
    b.lt            9f
0:
    ldr             Y, [INPUT_BUF0, INPUT_ROW, uxtw #3]
    ldr             U, [INPUT_BUF1, INPUT_ROW, uxtw #3]
    mov             N, OUTPUT_WIDTH
    ldr             V, [INPUT_BUF2, INPUT_ROW, uxtw #3]
    add             INPUT_ROW, INPUT_ROW, #1
    ldr             RGB, [OUTPUT_BUF], #8

    /* Inner loop over pixels */
    subs            N, N, #8
    b.lt            3f
    do_load         8
    do_yuv_to_rgb_stage1
    subs            N, N, #8
    b.lt            2f
1:
    do_yuv_to_rgb_stage2_store_load_stage1 \fast_st3
    subs            N, N, #8
    b.ge            1b
2:
    do_yuv_to_rgb_stage2
    do_store        \bpp, 8, \fast_st3
    tst             N, #7
    b.eq            8f
3:
    tst             N, #4
    b.eq            3f
    do_load         4
3:
    tst             N, #2
    b.eq            4f
    do_load         2
4:
    tst             N, #1
    b.eq            5f
    do_load         1
5:
    do_yuv_to_rgb
    tst             N, #4
    b.eq            6f
    do_store        \bpp, 4, \fast_st3
6:
    tst             N, #2
    b.eq            7f
    do_store        \bpp, 2, \fast_st3
7:
    tst             N, #1
    b.eq            8f
    do_store        \bpp, 1, \fast_st3
8:
    subs            NUM_ROWS, NUM_ROWS, #1
    b.gt            0b
9:
    /* Restore all registers and return */
    ld1             {v8.8b, v9.8b, v10.8b, v11.8b}, [sp], 32
    ld1             {v12.8b, v13.8b, v14.8b, v15.8b}, [sp], 32
    br              x30
    .unreq          OUTPUT_WIDTH
    .unreq          INPUT_ROW
    .unreq          OUTPUT_BUF
    .unreq          NUM_ROWS
    .unreq          INPUT_BUF0
    .unreq          INPUT_BUF1
    .unreq          INPUT_BUF2
    .unreq          RGB
    .unreq          Y
    .unreq          U
    .unreq          V
    .unreq          N

.purgem do_yuv_to_rgb
.purgem do_yuv_to_rgb_stage1
.purgem do_yuv_to_rgb_stage2
.purgem do_yuv_to_rgb_stage2_store_load_stage1

.endm

/*--------------------------------- id ----- bpp R  rsize G  gsize B  bsize defsize fast_st3*/
generate_jsimd_ycc_rgb_convert_neon extrgb,  24, 0, .4h,  1, .4h,  2, .4h,  .8b,    1
generate_jsimd_ycc_rgb_convert_neon extbgr,  24, 2, .4h,  1, .4h,  0, .4h,  .8b,    1
generate_jsimd_ycc_rgb_convert_neon extrgbx, 32, 0, .4h,  1, .4h,  2, .4h,  .8b,    1
generate_jsimd_ycc_rgb_convert_neon extbgrx, 32, 2, .4h,  1, .4h,  0, .4h,  .8b,    1
generate_jsimd_ycc_rgb_convert_neon extxbgr, 32, 3, .4h,  2, .4h,  1, .4h,  .8b,    1
generate_jsimd_ycc_rgb_convert_neon extxrgb, 32, 1, .4h,  2, .4h,  3, .4h,  .8b,    1
generate_jsimd_ycc_rgb_convert_neon rgb565,  16, 0, .4h,  0, .4h,  0, .4h,  .8b,    1

generate_jsimd_ycc_rgb_convert_neon extrgb,  24, 0, .4h,  1, .4h,  2, .4h,  .8b,    0
generate_jsimd_ycc_rgb_convert_neon extbgr,  24, 2, .4h,  1, .4h,  0, .4h,  .8b,    0

.purgem do_load
.purgem do_store


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
    st1             {v20.8b}, [Y], #8
    st1             {v21.8b}, [U], #8
    st1             {v22.8b}, [V], #8
  .elseif \size == 4
    st1             {v20.b}[0], [Y], #1
    st1             {v20.b}[1], [Y], #1
    st1             {v20.b}[2], [Y], #1
    st1             {v20.b}[3], [Y], #1
    st1             {v21.b}[0], [U], #1
    st1             {v21.b}[1], [U], #1
    st1             {v21.b}[2], [U], #1
    st1             {v21.b}[3], [U], #1
    st1             {v22.b}[0], [V], #1
    st1             {v22.b}[1], [V], #1
    st1             {v22.b}[2], [V], #1
    st1             {v22.b}[3], [V], #1
  .elseif \size == 2
    st1             {v20.b}[4], [Y], #1
    st1             {v20.b}[5], [Y], #1
    st1             {v21.b}[4], [U], #1
    st1             {v21.b}[5], [U], #1
    st1             {v22.b}[4], [V], #1
    st1             {v22.b}[5], [V], #1
  .elseif \size == 1
    st1             {v20.b}[6], [Y], #1
    st1             {v21.b}[6], [U], #1
    st1             {v22.b}[6], [V], #1
  .else
    .error unsupported macroblock size
  .endif
.endm

.macro do_load bpp, size, fast_ld3
  .if \bpp == 24
    .if \size == 8
      .if \fast_ld3 == 1
        ld3         {v10.8b, v11.8b, v12.8b}, [RGB], #24
      .else
        ld1         {v10.b}[0], [RGB], #1
        ld1         {v11.b}[0], [RGB], #1
        ld1         {v12.b}[0], [RGB], #1

        ld1         {v10.b}[1], [RGB], #1
        ld1         {v11.b}[1], [RGB], #1
        ld1         {v12.b}[1], [RGB], #1

        ld1         {v10.b}[2], [RGB], #1
        ld1         {v11.b}[2], [RGB], #1
        ld1         {v12.b}[2], [RGB], #1

        ld1         {v10.b}[3], [RGB], #1
        ld1         {v11.b}[3], [RGB], #1
        ld1         {v12.b}[3], [RGB], #1

        ld1         {v10.b}[4], [RGB], #1
        ld1         {v11.b}[4], [RGB], #1
        ld1         {v12.b}[4], [RGB], #1

        ld1         {v10.b}[5], [RGB], #1
        ld1         {v11.b}[5], [RGB], #1
        ld1         {v12.b}[5], [RGB], #1

        ld1         {v10.b}[6], [RGB], #1
        ld1         {v11.b}[6], [RGB], #1
        ld1         {v12.b}[6], [RGB], #1

        ld1         {v10.b}[7], [RGB], #1
        ld1         {v11.b}[7], [RGB], #1
        ld1         {v12.b}[7], [RGB], #1
      .endif
      prfm          pldl1keep, [RGB, #128]
    .elseif \size == 4
      ld3           {v10.b, v11.b, v12.b}[0], [RGB], #3
      ld3           {v10.b, v11.b, v12.b}[1], [RGB], #3
      ld3           {v10.b, v11.b, v12.b}[2], [RGB], #3
      ld3           {v10.b, v11.b, v12.b}[3], [RGB], #3
    .elseif \size == 2
      ld3           {v10.b, v11.b, v12.b}[4], [RGB], #3
      ld3           {v10.b, v11.b, v12.b}[5], [RGB], #3
    .elseif \size == 1
      ld3           {v10.b, v11.b, v12.b}[6], [RGB], #3
    .else
      .error unsupported macroblock size
    .endif
  .elseif \bpp == 32
    .if \size == 8
      ld4           {v10.8b, v11.8b, v12.8b, v13.8b}, [RGB], #32
      prfm          pldl1keep, [RGB, #128]
    .elseif \size == 4
      ld4           {v10.b, v11.b, v12.b, v13.b}[0], [RGB], #4
      ld4           {v10.b, v11.b, v12.b, v13.b}[1], [RGB], #4
      ld4           {v10.b, v11.b, v12.b, v13.b}[2], [RGB], #4
      ld4           {v10.b, v11.b, v12.b, v13.b}[3], [RGB], #4
    .elseif \size == 2
      ld4           {v10.b, v11.b, v12.b, v13.b}[4], [RGB], #4
      ld4           {v10.b, v11.b, v12.b, v13.b}[5], [RGB], #4
    .elseif \size == 1
      ld4           {v10.b, v11.b, v12.b, v13.b}[6], [RGB], #4
    .else
      .error unsupported macroblock size
    .endif
  .else
    .error unsupported bpp
  .endif
.endm

.macro generate_jsimd_rgb_ycc_convert_neon colorid, bpp, r_offs, g_offs, \
                                           b_offs, fast_ld3

/*
 * 2-stage pipelined RGB->YCbCr conversion
 */

.macro do_rgb_to_yuv_stage1
    ushll           v4.8h, v1\r_offs\().8b, #0  /* r = v4 */
    ushll           v6.8h, v1\g_offs\().8b, #0  /* g = v6 */
    ushll           v8.8h, v1\b_offs\().8b, #0  /* b = v8 */
    rev64           v18.4s, v1.4s
    rev64           v26.4s, v1.4s
    rev64           v28.4s, v1.4s
    rev64           v30.4s, v1.4s
    umull           v14.4s, v4.4h, v0.h[0]
    umull2          v16.4s, v4.8h, v0.h[0]
    umlsl           v18.4s, v4.4h, v0.h[3]
    umlsl2          v26.4s, v4.8h, v0.h[3]
    umlal           v28.4s, v4.4h, v0.h[5]
    umlal2          v30.4s, v4.8h, v0.h[5]
    umlal           v14.4s, v6.4h, v0.h[1]
    umlal2          v16.4s, v6.8h, v0.h[1]
    umlsl           v18.4s, v6.4h, v0.h[4]
    umlsl2          v26.4s, v6.8h, v0.h[4]
    umlsl           v28.4s, v6.4h, v0.h[6]
    umlsl2          v30.4s, v6.8h, v0.h[6]
    umlal           v14.4s, v8.4h, v0.h[2]
    umlal2          v16.4s, v8.8h, v0.h[2]
    umlal           v18.4s, v8.4h, v0.h[5]
    umlal2          v26.4s, v8.8h, v0.h[5]
    umlsl           v28.4s, v8.4h, v0.h[7]
    umlsl2          v30.4s, v8.8h, v0.h[7]
.endm

.macro do_rgb_to_yuv_stage2
    rshrn           v20.4h, v14.4s, #16
    shrn            v22.4h, v18.4s, #16
    shrn            v24.4h, v28.4s, #16
    rshrn2          v20.8h, v16.4s, #16
    shrn2           v22.8h, v26.4s, #16
    shrn2           v24.8h, v30.4s, #16
    xtn             v20.8b, v20.8h       /* v20 = y */
    xtn             v21.8b, v22.8h       /* v21 = u */
    xtn             v22.8b, v24.8h       /* v22 = v */
.endm

.macro do_rgb_to_yuv
    do_rgb_to_yuv_stage1
    do_rgb_to_yuv_stage2
.endm

/* TODO: expand macros and interleave instructions if some in-order
 *       AArch64 processor actually can dual-issue LOAD/STORE with ALU */
.macro do_rgb_to_yuv_stage2_store_load_stage1 fast_ld3
    do_rgb_to_yuv_stage2
    do_load         \bpp, 8, \fast_ld3
    st1             {v20.8b}, [Y], #8
    st1             {v21.8b}, [U], #8
    st1             {v22.8b}, [V], #8
    do_rgb_to_yuv_stage1
.endm

.if \fast_ld3 == 1
asm_function jsimd_\colorid\()_ycc_convert_neon
.else
asm_function jsimd_\colorid\()_ycc_convert_neon_slowld3
.endif
    OUTPUT_WIDTH    .req w0
    INPUT_BUF       .req x1
    OUTPUT_BUF      .req x2
    OUTPUT_ROW      .req w3
    NUM_ROWS        .req w4

    OUTPUT_BUF0     .req x5
    OUTPUT_BUF1     .req x6
    OUTPUT_BUF2     .req x2  /* OUTPUT_BUF */

    RGB             .req x7
    Y               .req x9
    U               .req x10
    V               .req x11
    N               .req w12

    /* Load constants to d0, d1, d2, d3 */
    get_symbol_loc  x13, Ljsimd_rgb_ycc_neon_consts
    ld1             {v0.8h, v1.8h}, [x13]

    ldr             OUTPUT_BUF0, [OUTPUT_BUF]
    ldr             OUTPUT_BUF1, [OUTPUT_BUF, #8]
    ldr             OUTPUT_BUF2, [OUTPUT_BUF, #16]
    .unreq          OUTPUT_BUF

    /* Save Neon registers */
    sub             sp, sp, #64
    mov             x9, sp
    st1             {v8.8b, v9.8b, v10.8b, v11.8b}, [x9], 32
    st1             {v12.8b, v13.8b, v14.8b, v15.8b}, [x9], 32

    /* Outer loop over scanlines */
    cmp             NUM_ROWS, #1
    b.lt            9f
0:
    ldr             Y, [OUTPUT_BUF0, OUTPUT_ROW, uxtw #3]
    ldr             U, [OUTPUT_BUF1, OUTPUT_ROW, uxtw #3]
    mov             N, OUTPUT_WIDTH
    ldr             V, [OUTPUT_BUF2, OUTPUT_ROW, uxtw #3]
    add             OUTPUT_ROW, OUTPUT_ROW, #1
    ldr             RGB, [INPUT_BUF], #8

    /* Inner loop over pixels */
    subs            N, N, #8
    b.lt            3f
    do_load         \bpp, 8, \fast_ld3
    do_rgb_to_yuv_stage1
    subs            N, N, #8
    b.lt            2f
1:
    do_rgb_to_yuv_stage2_store_load_stage1 \fast_ld3
    subs            N, N, #8
    b.ge            1b
2:
    do_rgb_to_yuv_stage2
    do_store        8
    tst             N, #7
    b.eq            8f
3:
    tbz             N, #2, 3f
    do_load         \bpp, 4, \fast_ld3
3:
    tbz             N, #1, 4f
    do_load         \bpp, 2, \fast_ld3
4:
    tbz             N, #0, 5f
    do_load         \bpp, 1, \fast_ld3
5:
    do_rgb_to_yuv
    tbz             N, #2, 6f
    do_store        4
6:
    tbz             N, #1, 7f
    do_store        2
7:
    tbz             N, #0, 8f
    do_store        1
8:
    subs            NUM_ROWS, NUM_ROWS, #1
    b.gt            0b
9:
    /* Restore all registers and return */
    ld1             {v8.8b, v9.8b, v10.8b, v11.8b}, [sp], 32
    ld1             {v12.8b, v13.8b, v14.8b, v15.8b}, [sp], 32
    br              x30

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

/*--------------------------------- id ----- bpp R  G  B  Fast LD3 */
generate_jsimd_rgb_ycc_convert_neon extrgb,  24, 0, 1, 2, 1
generate_jsimd_rgb_ycc_convert_neon extbgr,  24, 2, 1, 0, 1
generate_jsimd_rgb_ycc_convert_neon extrgbx, 32, 0, 1, 2, 1
generate_jsimd_rgb_ycc_convert_neon extbgrx, 32, 2, 1, 0, 1
generate_jsimd_rgb_ycc_convert_neon extxbgr, 32, 3, 2, 1, 1
generate_jsimd_rgb_ycc_convert_neon extxrgb, 32, 1, 2, 3, 1

generate_jsimd_rgb_ycc_convert_neon extrgb,  24, 0, 1, 2, 0
generate_jsimd_rgb_ycc_convert_neon extbgr,  24, 2, 1, 0, 0

.purgem do_load
.purgem do_store


/*****************************************************************************/

/*
 * jsimd_fdct_islow_neon
 *
 * This file contains a slower but more accurate integer implementation of the
 * forward DCT (Discrete Cosine Transform). The following code is based
 * directly on the IJG''s original jfdctint.c; see the jfdctint.c for
 * more details.
 *
 * TODO: can be combined with 'jsimd_convsamp_neon' to get
 *       rid of a bunch of VLD1.16 instructions
 */

#define CONST_BITS  13
#define PASS1_BITS  2

#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS)

#define XFIX_P_0_298  v0.h[0]
#define XFIX_N_0_390  v0.h[1]
#define XFIX_P_0_541  v0.h[2]
#define XFIX_P_0_765  v0.h[3]
#define XFIX_N_0_899  v0.h[4]
#define XFIX_P_1_175  v0.h[5]
#define XFIX_P_1_501  v0.h[6]
#define XFIX_N_1_847  v0.h[7]
#define XFIX_N_1_961  v1.h[0]
#define XFIX_P_2_053  v1.h[1]
#define XFIX_N_2_562  v1.h[2]
#define XFIX_P_3_072  v1.h[3]

asm_function jsimd_fdct_islow_neon

    DATA            .req x0
    TMP             .req x9

    /* Load constants */
    get_symbol_loc  TMP, Ljsimd_fdct_islow_neon_consts
    ld1             {v0.8h, v1.8h}, [TMP]

    /* Save Neon registers */
    sub             sp, sp, #64
    mov             x10, sp
    st1             {v8.8b, v9.8b, v10.8b, v11.8b}, [x10], 32
    st1             {v12.8b, v13.8b, v14.8b, v15.8b}, [x10], 32

    /* Load all DATA into Neon registers with the following allocation:
     *       0 1 2 3 | 4 5 6 7
     *      ---------+--------
     *   0 | d16     | d17    | v16.8h
     *   1 | d18     | d19    | v17.8h
     *   2 | d20     | d21    | v18.8h
     *   3 | d22     | d23    | v19.8h
     *   4 | d24     | d25    | v20.8h
     *   5 | d26     | d27    | v21.8h
     *   6 | d28     | d29    | v22.8h
     *   7 | d30     | d31    | v23.8h
     */

    ld1             {v16.8h, v17.8h, v18.8h, v19.8h}, [DATA], 64
    ld1             {v20.8h, v21.8h, v22.8h, v23.8h}, [DATA]
    sub             DATA, DATA, #64

    /* Transpose */
    transpose_8x8   v16, v17, v18, v19, v20, v21, v22, v23, v31, v2, v3, v4
    /* 1-D FDCT */
    add             v24.8h, v16.8h, v23.8h  /* tmp0 = dataptr[0] + dataptr[7]; */
    sub             v31.8h, v16.8h, v23.8h  /* tmp7 = dataptr[0] - dataptr[7]; */
    add             v25.8h, v17.8h, v22.8h  /* tmp1 = dataptr[1] + dataptr[6]; */
    sub             v30.8h, v17.8h, v22.8h  /* tmp6 = dataptr[1] - dataptr[6]; */
    add             v26.8h, v18.8h, v21.8h  /* tmp2 = dataptr[2] + dataptr[5]; */
    sub             v29.8h, v18.8h, v21.8h  /* tmp5 = dataptr[2] - dataptr[5]; */
    add             v27.8h, v19.8h, v20.8h  /* tmp3 = dataptr[3] + dataptr[4]; */
    sub             v28.8h, v19.8h, v20.8h  /* tmp4 = dataptr[3] - dataptr[4]; */

    /* even part */

    add             v8.8h, v24.8h, v27.8h   /* tmp10 = tmp0 + tmp3; */
    sub             v9.8h, v24.8h, v27.8h   /* tmp13 = tmp0 - tmp3; */
    add             v10.8h, v25.8h, v26.8h  /* tmp11 = tmp1 + tmp2; */
    sub             v11.8h, v25.8h, v26.8h  /* tmp12 = tmp1 - tmp2; */

    add             v16.8h, v8.8h, v10.8h  /* tmp10 + tmp11 */
    sub             v20.8h, v8.8h, v10.8h  /* tmp10 - tmp11 */

    add             v18.8h, v11.8h, v9.8h  /* tmp12 + tmp13 */

    shl             v16.8h, v16.8h, #PASS1_BITS  /* dataptr[0] = (DCTELEM)LEFT_SHIFT(tmp10 + tmp11, PASS1_BITS); */
    shl             v20.8h, v20.8h, #PASS1_BITS  /* dataptr[4] = (DCTELEM)LEFT_SHIFT(tmp10 - tmp11, PASS1_BITS); */

    smull2          v24.4s, v18.8h, XFIX_P_0_541  /* z1 hi = MULTIPLY(tmp12 + tmp13, XFIX_P_0_541); */
    smull           v18.4s, v18.4h, XFIX_P_0_541  /* z1 lo = MULTIPLY(tmp12 + tmp13, XFIX_P_0_541); */
    mov             v22.16b, v18.16b
    mov             v25.16b, v24.16b

    smlal           v18.4s, v9.4h, XFIX_P_0_765   /* lo z1 + MULTIPLY(tmp13, XFIX_P_0_765) */
    smlal2          v24.4s, v9.8h, XFIX_P_0_765   /* hi z1 + MULTIPLY(tmp13, XFIX_P_0_765) */
    smlal           v22.4s, v11.4h, XFIX_N_1_847  /* lo z1 + MULTIPLY(tmp12, XFIX_N_1_847) */
    smlal2          v25.4s, v11.8h, XFIX_N_1_847  /* hi z1 + MULTIPLY(tmp12, XFIX_N_1_847) */

    rshrn           v18.4h, v18.4s, #DESCALE_P1
    rshrn           v22.4h, v22.4s, #DESCALE_P1
    rshrn2          v18.8h, v24.4s, #DESCALE_P1  /* dataptr[2] = (DCTELEM)DESCALE(z1 + MULTIPLY(tmp13, XFIX_P_0_765), CONST_BITS-PASS1_BITS); */
    rshrn2          v22.8h, v25.4s, #DESCALE_P1  /* dataptr[6] = (DCTELEM)DESCALE(z1 + MULTIPLY(tmp12, XFIX_N_1_847), CONST_BITS-PASS1_BITS); */

    /* Odd part */

    add             v8.8h, v28.8h, v31.8h        /* z1 = tmp4 + tmp7; */
    add             v9.8h, v29.8h, v30.8h        /* z2 = tmp5 + tmp6; */
    add             v10.8h, v28.8h, v30.8h       /* z3 = tmp4 + tmp6; */
    add             v11.8h, v29.8h, v31.8h       /* z4 = tmp5 + tmp7; */
    smull           v4.4s, v10.4h, XFIX_P_1_175  /* z5 lo = z3 lo * XFIX_P_1_175 */
    smull2          v5.4s, v10.8h, XFIX_P_1_175
    smlal           v4.4s, v11.4h, XFIX_P_1_175  /* z5 = MULTIPLY(z3 + z4, FIX_1_175875602); */
    smlal2          v5.4s, v11.8h, XFIX_P_1_175

    smull2          v24.4s, v28.8h, XFIX_P_0_298
    smull2          v25.4s, v29.8h, XFIX_P_2_053
    smull2          v26.4s, v30.8h, XFIX_P_3_072
    smull2          v27.4s, v31.8h, XFIX_P_1_501
    smull           v28.4s, v28.4h, XFIX_P_0_298  /* tmp4 = MULTIPLY(tmp4, FIX_0_298631336); */
    smull           v29.4s, v29.4h, XFIX_P_2_053  /* tmp5 = MULTIPLY(tmp5, FIX_2_053119869); */
    smull           v30.4s, v30.4h, XFIX_P_3_072  /* tmp6 = MULTIPLY(tmp6, FIX_3_072711026); */
    smull           v31.4s, v31.4h, XFIX_P_1_501  /* tmp7 = MULTIPLY(tmp7, FIX_1_501321110); */

    smull2          v12.4s, v8.8h, XFIX_N_0_899
    smull2          v13.4s, v9.8h, XFIX_N_2_562
    smull2          v14.4s, v10.8h, XFIX_N_1_961
    smull2          v15.4s, v11.8h, XFIX_N_0_390
    smull           v8.4s, v8.4h, XFIX_N_0_899    /* z1 = MULTIPLY(z1, -FIX_0_899976223); */
    smull           v9.4s, v9.4h, XFIX_N_2_562    /* z2 = MULTIPLY(z2, -FIX_2_562915447); */
    smull           v10.4s, v10.4h, XFIX_N_1_961  /* z3 = MULTIPLY(z3, -FIX_1_961570560); */
    smull           v11.4s, v11.4h, XFIX_N_0_390  /* z4 = MULTIPLY(z4, -FIX_0_390180644); */

    add             v10.4s, v10.4s, v4.4s  /* z3 += z5 */
    add             v14.4s, v14.4s, v5.4s
    add             v11.4s, v11.4s, v4.4s  /* z4 += z5 */
    add             v15.4s, v15.4s, v5.4s

    add             v28.4s, v28.4s, v8.4s   /* tmp4 += z1 */
    add             v24.4s, v24.4s, v12.4s
    add             v29.4s, v29.4s, v9.4s   /* tmp5 += z2 */
    add             v25.4s, v25.4s, v13.4s
    add             v30.4s, v30.4s, v10.4s  /* tmp6 += z3 */
    add             v26.4s, v26.4s, v14.4s
    add             v31.4s, v31.4s, v11.4s  /* tmp7 += z4 */
    add             v27.4s, v27.4s, v15.4s

    add             v28.4s, v28.4s, v10.4s  /* tmp4 += z3 */
    add             v24.4s, v24.4s, v14.4s
    add             v29.4s, v29.4s, v11.4s  /* tmp5 += z4 */
    add             v25.4s, v25.4s, v15.4s
    add             v30.4s, v30.4s, v9.4s   /* tmp6 += z2 */
    add             v26.4s, v26.4s, v13.4s
    add             v31.4s, v31.4s, v8.4s   /* tmp7 += z1 */
    add             v27.4s, v27.4s, v12.4s

    rshrn           v23.4h, v28.4s, #DESCALE_P1
    rshrn           v21.4h, v29.4s, #DESCALE_P1
    rshrn           v19.4h, v30.4s, #DESCALE_P1
    rshrn           v17.4h, v31.4s, #DESCALE_P1
    rshrn2          v23.8h, v24.4s, #DESCALE_P1  /* dataptr[7] = (DCTELEM)DESCALE(tmp4 + z1 + z3, CONST_BITS-PASS1_BITS); */
    rshrn2          v21.8h, v25.4s, #DESCALE_P1  /* dataptr[5] = (DCTELEM)DESCALE(tmp5 + z2 + z4, CONST_BITS-PASS1_BITS); */
    rshrn2          v19.8h, v26.4s, #DESCALE_P1  /* dataptr[3] = (DCTELEM)DESCALE(tmp6 + z2 + z3, CONST_BITS-PASS1_BITS); */
    rshrn2          v17.8h, v27.4s, #DESCALE_P1  /* dataptr[1] = (DCTELEM)DESCALE(tmp7 + z1 + z4, CONST_BITS-PASS1_BITS); */

    /* Transpose */
    transpose_8x8   v16, v17, v18, v19, v20, v21, v22, v23, v31, v2, v3, v4

    /* 1-D FDCT */
    add             v24.8h, v16.8h, v23.8h  /* tmp0 = dataptr[0] + dataptr[7]; */
    sub             v31.8h, v16.8h, v23.8h  /* tmp7 = dataptr[0] - dataptr[7]; */
    add             v25.8h, v17.8h, v22.8h  /* tmp1 = dataptr[1] + dataptr[6]; */
    sub             v30.8h, v17.8h, v22.8h  /* tmp6 = dataptr[1] - dataptr[6]; */
    add             v26.8h, v18.8h, v21.8h  /* tmp2 = dataptr[2] + dataptr[5]; */
    sub             v29.8h, v18.8h, v21.8h  /* tmp5 = dataptr[2] - dataptr[5]; */
    add             v27.8h, v19.8h, v20.8h  /* tmp3 = dataptr[3] + dataptr[4]; */
    sub             v28.8h, v19.8h, v20.8h  /* tmp4 = dataptr[3] - dataptr[4]; */

    /* even part */
    add             v8.8h, v24.8h, v27.8h   /* tmp10 = tmp0 + tmp3; */
    sub             v9.8h, v24.8h, v27.8h   /* tmp13 = tmp0 - tmp3; */
    add             v10.8h, v25.8h, v26.8h  /* tmp11 = tmp1 + tmp2; */
    sub             v11.8h, v25.8h, v26.8h  /* tmp12 = tmp1 - tmp2; */

    add             v16.8h, v8.8h, v10.8h  /* tmp10 + tmp11 */
    sub             v20.8h, v8.8h, v10.8h  /* tmp10 - tmp11 */

    add             v18.8h, v11.8h, v9.8h  /* tmp12 + tmp13 */

    srshr           v16.8h, v16.8h, #PASS1_BITS  /* dataptr[0] = (DCTELEM)DESCALE(tmp10 + tmp11, PASS1_BITS); */
    srshr           v20.8h, v20.8h, #PASS1_BITS  /* dataptr[4] = (DCTELEM)DESCALE(tmp10 - tmp11, PASS1_BITS); */

    smull2          v24.4s, v18.8h, XFIX_P_0_541  /* z1 hi = MULTIPLY(tmp12 + tmp13, XFIX_P_0_541); */
    smull           v18.4s, v18.4h, XFIX_P_0_541  /* z1 lo = MULTIPLY(tmp12 + tmp13, XFIX_P_0_541); */
    mov             v22.16b, v18.16b
    mov             v25.16b, v24.16b

    smlal           v18.4s, v9.4h, XFIX_P_0_765   /* lo z1 + MULTIPLY(tmp13, XFIX_P_0_765) */
    smlal2          v24.4s, v9.8h, XFIX_P_0_765   /* hi z1 + MULTIPLY(tmp13, XFIX_P_0_765) */
    smlal           v22.4s, v11.4h, XFIX_N_1_847  /* lo z1 + MULTIPLY(tmp12, XFIX_N_1_847) */
    smlal2          v25.4s, v11.8h, XFIX_N_1_847  /* hi z1 + MULTIPLY(tmp12, XFIX_N_1_847) */

    rshrn           v18.4h, v18.4s, #DESCALE_P2
    rshrn           v22.4h, v22.4s, #DESCALE_P2
    rshrn2          v18.8h, v24.4s, #DESCALE_P2  /* dataptr[2] = (DCTELEM)DESCALE(z1 + MULTIPLY(tmp13, XFIX_P_0_765), CONST_BITS-PASS1_BITS); */
    rshrn2          v22.8h, v25.4s, #DESCALE_P2  /* dataptr[6] = (DCTELEM)DESCALE(z1 + MULTIPLY(tmp12, XFIX_N_1_847), CONST_BITS-PASS1_BITS); */

    /* Odd part */
    add             v8.8h, v28.8h, v31.8h   /* z1 = tmp4 + tmp7; */
    add             v9.8h, v29.8h, v30.8h   /* z2 = tmp5 + tmp6; */
    add             v10.8h, v28.8h, v30.8h  /* z3 = tmp4 + tmp6; */
    add             v11.8h, v29.8h, v31.8h  /* z4 = tmp5 + tmp7; */

    smull           v4.4s, v10.4h, XFIX_P_1_175  /* z5 lo = z3 lo * XFIX_P_1_175 */
    smull2          v5.4s, v10.8h, XFIX_P_1_175
    smlal           v4.4s, v11.4h, XFIX_P_1_175  /* z5 = MULTIPLY(z3 + z4, FIX_1_175875602); */
    smlal2          v5.4s, v11.8h, XFIX_P_1_175

    smull2          v24.4s, v28.8h, XFIX_P_0_298
    smull2          v25.4s, v29.8h, XFIX_P_2_053
    smull2          v26.4s, v30.8h, XFIX_P_3_072
    smull2          v27.4s, v31.8h, XFIX_P_1_501
    smull           v28.4s, v28.4h, XFIX_P_0_298  /* tmp4 = MULTIPLY(tmp4, FIX_0_298631336); */
    smull           v29.4s, v29.4h, XFIX_P_2_053  /* tmp5 = MULTIPLY(tmp5, FIX_2_053119869); */
    smull           v30.4s, v30.4h, XFIX_P_3_072  /* tmp6 = MULTIPLY(tmp6, FIX_3_072711026); */
    smull           v31.4s, v31.4h, XFIX_P_1_501  /* tmp7 = MULTIPLY(tmp7, FIX_1_501321110); */

    smull2          v12.4s, v8.8h, XFIX_N_0_899
    smull2          v13.4s, v9.8h, XFIX_N_2_562
    smull2          v14.4s, v10.8h, XFIX_N_1_961
    smull2          v15.4s, v11.8h, XFIX_N_0_390
    smull           v8.4s, v8.4h, XFIX_N_0_899    /* z1 = MULTIPLY(z1, -FIX_0_899976223); */
    smull           v9.4s, v9.4h, XFIX_N_2_562    /* z2 = MULTIPLY(z2, -FIX_2_562915447); */
    smull           v10.4s, v10.4h, XFIX_N_1_961  /* z3 = MULTIPLY(z3, -FIX_1_961570560); */
    smull           v11.4s, v11.4h, XFIX_N_0_390  /* z4 = MULTIPLY(z4, -FIX_0_390180644); */

    add             v10.4s, v10.4s, v4.4s
    add             v14.4s, v14.4s, v5.4s
    add             v11.4s, v11.4s, v4.4s
    add             v15.4s, v15.4s, v5.4s

    add             v28.4s, v28.4s, v8.4s   /* tmp4 += z1 */
    add             v24.4s, v24.4s, v12.4s
    add             v29.4s, v29.4s, v9.4s   /* tmp5 += z2 */
    add             v25.4s, v25.4s, v13.4s
    add             v30.4s, v30.4s, v10.4s  /* tmp6 += z3 */
    add             v26.4s, v26.4s, v14.4s
    add             v31.4s, v31.4s, v11.4s  /* tmp7 += z4 */
    add             v27.4s, v27.4s, v15.4s

    add             v28.4s, v28.4s, v10.4s  /* tmp4 += z3 */
    add             v24.4s, v24.4s, v14.4s
    add             v29.4s, v29.4s, v11.4s  /* tmp5 += z4 */
    add             v25.4s, v25.4s, v15.4s
    add             v30.4s, v30.4s, v9.4s   /* tmp6 += z2 */
    add             v26.4s, v26.4s, v13.4s
    add             v31.4s, v31.4s, v8.4s   /* tmp7 += z1 */
    add             v27.4s, v27.4s, v12.4s

    rshrn           v23.4h, v28.4s, #DESCALE_P2
    rshrn           v21.4h, v29.4s, #DESCALE_P2
    rshrn           v19.4h, v30.4s, #DESCALE_P2
    rshrn           v17.4h, v31.4s, #DESCALE_P2
    rshrn2          v23.8h, v24.4s, #DESCALE_P2  /* dataptr[7] = (DCTELEM)DESCALE(tmp4 + z1 + z3, CONST_BITS-PASS1_BITS); */
    rshrn2          v21.8h, v25.4s, #DESCALE_P2  /* dataptr[5] = (DCTELEM)DESCALE(tmp5 + z2 + z4, CONST_BITS-PASS1_BITS); */
    rshrn2          v19.8h, v26.4s, #DESCALE_P2  /* dataptr[3] = (DCTELEM)DESCALE(tmp6 + z2 + z3, CONST_BITS-PASS1_BITS); */
    rshrn2          v17.8h, v27.4s, #DESCALE_P2  /* dataptr[1] = (DCTELEM)DESCALE(tmp7 + z1 + z4, CONST_BITS-PASS1_BITS); */

    /* store results */
    st1             {v16.8h, v17.8h, v18.8h, v19.8h}, [DATA], 64
    st1             {v20.8h, v21.8h, v22.8h, v23.8h}, [DATA]

    /* Restore Neon registers */
    ld1             {v8.8b, v9.8b, v10.8b, v11.8b}, [sp], 32
    ld1             {v12.8b, v13.8b, v14.8b, v15.8b}, [sp], 32

    br              x30

    .unreq          DATA
    .unreq          TMP

#undef XFIX_P_0_298
#undef XFIX_N_0_390
#undef XFIX_P_0_541
#undef XFIX_P_0_765
#undef XFIX_N_0_899
#undef XFIX_P_1_175
#undef XFIX_P_1_501
#undef XFIX_N_1_847
#undef XFIX_N_1_961
#undef XFIX_P_2_053
#undef XFIX_N_2_562
#undef XFIX_P_3_072


/*****************************************************************************/

/*
 * GLOBAL(JOCTET *)
 * jsimd_huff_encode_one_block(working_state *state, JOCTET *buffer,
 *                             JCOEFPTR block, int last_dc_val,
 *                             c_derived_tbl *dctbl, c_derived_tbl *actbl)
 *
 */

    BUFFER          .req x1
    PUT_BUFFER      .req x6
    PUT_BITS        .req x7
    PUT_BITSw       .req w7

.macro emit_byte
    sub             PUT_BITS, PUT_BITS, #0x8
    lsr             x19, PUT_BUFFER, PUT_BITS
    uxtb            w19, w19
    strb            w19, [BUFFER, #1]!
    cmp             w19, #0xff
    b.ne            14f
    strb            wzr, [BUFFER, #1]!
14:
.endm
.macro put_bits CODE, SIZE
    lsl             PUT_BUFFER, PUT_BUFFER, \SIZE
    add             PUT_BITS, PUT_BITS, \SIZE
    orr             PUT_BUFFER, PUT_BUFFER, \CODE
.endm
.macro checkbuf31
    cmp             PUT_BITS, #0x20
    b.lt            31f
    emit_byte
    emit_byte
    emit_byte
    emit_byte
31:
.endm
.macro checkbuf47
    cmp             PUT_BITS, #0x30
    b.lt            47f
    emit_byte
    emit_byte
    emit_byte
    emit_byte
    emit_byte
    emit_byte
47:
.endm

.macro generate_jsimd_huff_encode_one_block fast_tbl

.if \fast_tbl == 1
asm_function jsimd_huff_encode_one_block_neon
.else
asm_function jsimd_huff_encode_one_block_neon_slowtbl
.endif
    sub             sp, sp, 272
    sub             BUFFER, BUFFER, #0x1    /* BUFFER=buffer-- */
    /* Save Arm registers */
    stp             x19, x20, [sp]
    get_symbol_loc  x15, Ljsimd_huff_encode_one_block_neon_consts
    ldr             PUT_BUFFER, [x0, #0x10]
    ldr             PUT_BITSw, [x0, #0x18]
    ldrsh           w12, [x2]               /* load DC coeff in w12 */
    /* prepare data */
.if \fast_tbl == 1
    ld1             {v23.16b}, [x15], #16
    ld1             {v0.16b, v1.16b, v2.16b, v3.16b}, [x15], #64
    ld1             {v4.16b, v5.16b, v6.16b, v7.16b}, [x15], #64
    ld1             {v16.16b, v17.16b, v18.16b, v19.16b}, [x15], #64
    ld1             {v24.16b, v25.16b, v26.16b, v27.16b}, [x2], #64
    ld1             {v28.16b, v29.16b, v30.16b, v31.16b}, [x2], #64
    sub             w12, w12, w3      /* last_dc_val, not used afterwards */
    /* ZigZag 8x8 */
    tbl             v0.16b, {v24.16b, v25.16b, v26.16b, v27.16b}, v0.16b
    tbl             v1.16b, {v24.16b, v25.16b, v26.16b, v27.16b}, v1.16b
    tbl             v2.16b, {v25.16b, v26.16b, v27.16b, v28.16b}, v2.16b
    tbl             v3.16b, {v24.16b, v25.16b, v26.16b, v27.16b}, v3.16b
    tbl             v4.16b, {v28.16b, v29.16b, v30.16b, v31.16b}, v4.16b
    tbl             v5.16b, {v25.16b, v26.16b, v27.16b, v28.16b}, v5.16b
    tbl             v6.16b, {v27.16b, v28.16b, v29.16b, v30.16b}, v6.16b
    tbl             v7.16b, {v29.16b, v30.16b, v31.16b}, v7.16b
    ins             v0.h[0], w12
    tbx             v1.16b, {v28.16b}, v16.16b
    tbx             v2.16b, {v29.16b, v30.16b}, v17.16b
    tbx             v5.16b, {v29.16b, v30.16b}, v18.16b
    tbx             v6.16b, {v31.16b}, v19.16b
.else
      add             x13, x2, #0x22
      sub             w12, w12, w3    /* last_dc_val, not used afterwards */
    ld1             {v23.16b}, [x15]
      add             x14, x2, #0x18
      add             x3, x2, #0x36
    ins             v0.h[0], w12
      add             x9, x2, #0x2
    ld1             {v1.h}[0], [x13]
      add             x15, x2, #0x30
    ld1             {v2.h}[0], [x14]
      add             x19, x2, #0x26
    ld1             {v3.h}[0], [x3]
      add             x20, x2, #0x28
    ld1             {v0.h}[1], [x9]
      add             x12, x2, #0x10
    ld1             {v1.h}[1], [x15]
      add             x13, x2, #0x40
    ld1             {v2.h}[1], [x19]
      add             x14, x2, #0x34
    ld1             {v3.h}[1], [x20]
      add             x3, x2, #0x1a
    ld1             {v0.h}[2], [x12]
      add             x9, x2, #0x20
    ld1             {v1.h}[2], [x13]
      add             x15, x2, #0x32
    ld1             {v2.h}[2], [x14]
      add             x19, x2, #0x42
    ld1             {v3.h}[2], [x3]
      add             x20, x2, #0xc
    ld1             {v0.h}[3], [x9]
      add             x12, x2, #0x12
    ld1             {v1.h}[3], [x15]
      add             x13, x2, #0x24
    ld1             {v2.h}[3], [x19]
      add             x14, x2, #0x50
    ld1             {v3.h}[3], [x20]
      add             x3, x2, #0xe
    ld1             {v0.h}[4], [x12]
      add             x9, x2, #0x4
    ld1             {v1.h}[4], [x13]
      add             x15, x2, #0x16
    ld1             {v2.h}[4], [x14]
      add             x19, x2, #0x60
    ld1             {v3.h}[4], [x3]
      add             x20, x2, #0x1c
    ld1             {v0.h}[5], [x9]
      add             x12, x2, #0x6
    ld1             {v1.h}[5], [x15]
      add             x13, x2, #0x8
    ld1             {v2.h}[5], [x19]
      add             x14, x2, #0x52
    ld1             {v3.h}[5], [x20]
      add             x3, x2, #0x2a
    ld1             {v0.h}[6], [x12]
      add             x9, x2, #0x14
    ld1             {v1.h}[6], [x13]
      add             x15, x2, #0xa
    ld1             {v2.h}[6], [x14]
      add             x19, x2, #0x44
    ld1             {v3.h}[6], [x3]
      add             x20, x2, #0x38
    ld1             {v0.h}[7], [x9]
      add             x12, x2, #0x46
    ld1             {v1.h}[7], [x15]
      add             x13, x2, #0x3a
    ld1             {v2.h}[7], [x19]
      add             x14, x2, #0x74
    ld1             {v3.h}[7], [x20]
      add             x3, x2, #0x6a
    ld1             {v4.h}[0], [x12]
      add             x9, x2, #0x54
    ld1             {v5.h}[0], [x13]
      add             x15, x2, #0x2c
    ld1             {v6.h}[0], [x14]
      add             x19, x2, #0x76
    ld1             {v7.h}[0], [x3]
      add             x20, x2, #0x78
    ld1             {v4.h}[1], [x9]
      add             x12, x2, #0x62
    ld1             {v5.h}[1], [x15]
      add             x13, x2, #0x1e
    ld1             {v6.h}[1], [x19]
      add             x14, x2, #0x68
    ld1             {v7.h}[1], [x20]
      add             x3, x2, #0x7a
    ld1             {v4.h}[2], [x12]
      add             x9, x2, #0x70
    ld1             {v5.h}[2], [x13]
      add             x15, x2, #0x2e
    ld1             {v6.h}[2], [x14]
      add             x19, x2, #0x5a
    ld1             {v7.h}[2], [x3]
      add             x20, x2, #0x6c
    ld1             {v4.h}[3], [x9]
      add             x12, x2, #0x72
    ld1             {v5.h}[3], [x15]
      add             x13, x2, #0x3c
    ld1             {v6.h}[3], [x19]
      add             x14, x2, #0x4c
    ld1             {v7.h}[3], [x20]
      add             x3, x2, #0x5e
    ld1             {v4.h}[4], [x12]
      add             x9, x2, #0x64
    ld1             {v5.h}[4], [x13]
      add             x15, x2, #0x4a
    ld1             {v6.h}[4], [x14]
      add             x19, x2, #0x3e
    ld1             {v7.h}[4], [x3]
      add             x20, x2, #0x6e
    ld1             {v4.h}[5], [x9]
      add             x12, x2, #0x56
    ld1             {v5.h}[5], [x15]
      add             x13, x2, #0x58
    ld1             {v6.h}[5], [x19]
      add             x14, x2, #0x4e
    ld1             {v7.h}[5], [x20]
      add             x3, x2, #0x7c
    ld1             {v4.h}[6], [x12]
      add             x9, x2, #0x48
    ld1             {v5.h}[6], [x13]
      add             x15, x2, #0x66
    ld1             {v6.h}[6], [x14]
      add             x19, x2, #0x5c
    ld1             {v7.h}[6], [x3]
      add             x20, x2, #0x7e
    ld1             {v4.h}[7], [x9]
    ld1             {v5.h}[7], [x15]
    ld1             {v6.h}[7], [x19]
    ld1             {v7.h}[7], [x20]
.endif
    cmlt            v24.8h, v0.8h, #0
    cmlt            v25.8h, v1.8h, #0
    cmlt            v26.8h, v2.8h, #0
    cmlt            v27.8h, v3.8h, #0
    cmlt            v28.8h, v4.8h, #0
    cmlt            v29.8h, v5.8h, #0
    cmlt            v30.8h, v6.8h, #0
    cmlt            v31.8h, v7.8h, #0
    abs             v0.8h, v0.8h
    abs             v1.8h, v1.8h
    abs             v2.8h, v2.8h
    abs             v3.8h, v3.8h
    abs             v4.8h, v4.8h
    abs             v5.8h, v5.8h
    abs             v6.8h, v6.8h
    abs             v7.8h, v7.8h
    eor             v24.16b, v24.16b, v0.16b
    eor             v25.16b, v25.16b, v1.16b
    eor             v26.16b, v26.16b, v2.16b
    eor             v27.16b, v27.16b, v3.16b
    eor             v28.16b, v28.16b, v4.16b
    eor             v29.16b, v29.16b, v5.16b
    eor             v30.16b, v30.16b, v6.16b
    eor             v31.16b, v31.16b, v7.16b
    cmeq            v16.8h, v0.8h, #0
    cmeq            v17.8h, v1.8h, #0
    cmeq            v18.8h, v2.8h, #0
    cmeq            v19.8h, v3.8h, #0
    cmeq            v20.8h, v4.8h, #0
    cmeq            v21.8h, v5.8h, #0
    cmeq            v22.8h, v6.8h, #0
    xtn             v16.8b, v16.8h
    xtn             v18.8b, v18.8h
    xtn             v20.8b, v20.8h
    xtn             v22.8b, v22.8h
      umov            w14, v0.h[0]
    xtn2            v16.16b, v17.8h
      umov            w13, v24.h[0]
    xtn2            v18.16b, v19.8h
      clz             w14, w14
    xtn2            v20.16b, v21.8h
      lsl             w13, w13, w14
    cmeq            v17.8h, v7.8h, #0
      sub             w12, w14, #32
    xtn2            v22.16b, v17.8h
      lsr             w13, w13, w14
    and             v16.16b, v16.16b, v23.16b
      neg             w12, w12
    and             v18.16b, v18.16b, v23.16b
      add             x3, x4, #0x400           /* r1 = dctbl->ehufsi */
    and             v20.16b, v20.16b, v23.16b
      add             x15, sp, #0x90           /* x15 = t2 */
    and             v22.16b, v22.16b, v23.16b
      ldr             w10, [x4, x12, lsl #2]
    addp            v16.16b, v16.16b, v18.16b
      ldrb            w11, [x3, x12]
    addp            v20.16b, v20.16b, v22.16b
      checkbuf47
    addp            v16.16b, v16.16b, v20.16b
      put_bits        x10, x11
    addp            v16.16b, v16.16b, v18.16b
      checkbuf47
    umov            x9, v16.D[0]
      put_bits        x13, x12
    cnt             v17.8b, v16.8b
      mvn             x9, x9
    addv            B18, v17.8b
      add             x4, x5, #0x400   /* x4 = actbl->ehufsi */
    umov            w12, v18.b[0]
      lsr             x9, x9, #0x1     /* clear AC coeff */
    ldr             w13, [x5, #0x3c0]  /* x13 = actbl->ehufco[0xf0] */
    rbit            x9, x9             /* x9 = index0 */
    ldrb            w14, [x4, #0xf0]   /* x14 = actbl->ehufsi[0xf0] */
    cmp             w12, #(64-8)
    add             x11, sp, #16
    b.lt            4f
    cbz             x9, 6f
    st1             {v0.8h, v1.8h, v2.8h, v3.8h}, [x11], #64
    st1             {v4.8h, v5.8h, v6.8h, v7.8h}, [x11], #64
    st1             {v24.8h, v25.8h, v26.8h, v27.8h}, [x11], #64
    st1             {v28.8h, v29.8h, v30.8h, v31.8h}, [x11], #64
1:
    clz             x2, x9
    add             x15, x15, x2, lsl #1
    lsl             x9, x9, x2
    ldrh            w20, [x15, #-126]
2:
    cmp             x2, #0x10
    b.lt            3f
    sub             x2, x2, #0x10
    checkbuf47
    put_bits        x13, x14
    b               2b
3:
    clz             w20, w20
    ldrh            w3, [x15, #2]!
    sub             w11, w20, #32
    lsl             w3, w3, w20
    neg             w11, w11
    lsr             w3, w3, w20
    add             x2, x11, x2, lsl #4
    lsl             x9, x9, #0x1
    ldr             w12, [x5, x2, lsl #2]
    ldrb            w10, [x4, x2]
    checkbuf31
    put_bits        x12, x10
    put_bits        x3, x11
    cbnz            x9, 1b
    b               6f
4:
    movi            v21.8h, #0x0010
    clz             v0.8h, v0.8h
    clz             v1.8h, v1.8h
    clz             v2.8h, v2.8h
    clz             v3.8h, v3.8h
    clz             v4.8h, v4.8h
    clz             v5.8h, v5.8h
    clz             v6.8h, v6.8h
    clz             v7.8h, v7.8h
    ushl            v24.8h, v24.8h, v0.8h
    ushl            v25.8h, v25.8h, v1.8h
    ushl            v26.8h, v26.8h, v2.8h
    ushl            v27.8h, v27.8h, v3.8h
    ushl            v28.8h, v28.8h, v4.8h
    ushl            v29.8h, v29.8h, v5.8h
    ushl            v30.8h, v30.8h, v6.8h
    ushl            v31.8h, v31.8h, v7.8h
    neg             v0.8h, v0.8h
    neg             v1.8h, v1.8h
    neg             v2.8h, v2.8h
    neg             v3.8h, v3.8h
    neg             v4.8h, v4.8h
    neg             v5.8h, v5.8h
    neg             v6.8h, v6.8h
    neg             v7.8h, v7.8h
    ushl            v24.8h, v24.8h, v0.8h
    ushl            v25.8h, v25.8h, v1.8h
    ushl            v26.8h, v26.8h, v2.8h
    ushl            v27.8h, v27.8h, v3.8h
    ushl            v28.8h, v28.8h, v4.8h
    ushl            v29.8h, v29.8h, v5.8h
    ushl            v30.8h, v30.8h, v6.8h
    ushl            v31.8h, v31.8h, v7.8h
    add             v0.8h, v21.8h, v0.8h
    add             v1.8h, v21.8h, v1.8h
    add             v2.8h, v21.8h, v2.8h
    add             v3.8h, v21.8h, v3.8h
    add             v4.8h, v21.8h, v4.8h
    add             v5.8h, v21.8h, v5.8h
    add             v6.8h, v21.8h, v6.8h
    add             v7.8h, v21.8h, v7.8h
    st1             {v0.8h, v1.8h, v2.8h, v3.8h}, [x11], #64
    st1             {v4.8h, v5.8h, v6.8h, v7.8h}, [x11], #64
    st1             {v24.8h, v25.8h, v26.8h, v27.8h}, [x11], #64
    st1             {v28.8h, v29.8h, v30.8h, v31.8h}, [x11], #64
1:
    clz             x2, x9
    add             x15, x15, x2, lsl #1
    lsl             x9, x9, x2
    ldrh            w11, [x15, #-126]
2:
    cmp             x2, #0x10
    b.lt            3f
    sub             x2, x2, #0x10
    checkbuf47
    put_bits        x13, x14
    b               2b
3:
    ldrh            w3, [x15, #2]!
    add             x2, x11, x2, lsl #4
    lsl             x9, x9, #0x1
    ldr             w12, [x5, x2, lsl #2]
    ldrb            w10, [x4, x2]
    checkbuf31
    put_bits        x12, x10
    put_bits        x3, x11
    cbnz            x9, 1b
6:
    add             x13, sp, #0x10e
    cmp             x15, x13
    b.hs            1f
    ldr             w12, [x5]
    ldrb            w14, [x4]
    checkbuf47
    put_bits        x12, x14
1:
    str             PUT_BUFFER, [x0, #0x10]
    str             PUT_BITSw, [x0, #0x18]
    ldp             x19, x20, [sp], 16
    add             x0, BUFFER, #0x1
    add             sp, sp, 256
    br              x30

.endm

generate_jsimd_huff_encode_one_block 1
generate_jsimd_huff_encode_one_block 0

    .unreq          BUFFER
    .unreq          PUT_BUFFER
    .unreq          PUT_BITS
    .unreq          PUT_BITSw

.purgem emit_byte
.purgem put_bits
.purgem checkbuf31
.purgem checkbuf47
