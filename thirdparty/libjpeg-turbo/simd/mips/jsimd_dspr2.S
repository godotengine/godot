/*
 * MIPS DSPr2 optimizations for libjpeg-turbo
 *
 * Copyright (C) 2013-2014, MIPS Technologies, Inc., California.
 *                          All Rights Reserved.
 * Authors:  Teodora Novkovic <teodora.novkovic@imgtec.com>
 *           Darko Laus       <darko.laus@imgtec.com>
 * Copyright (C) 2015, D. R. Commander.  All Rights Reserved.
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

#include "jsimd_dspr2_asm.h"


/*****************************************************************************/
LEAF_DSPR2(jsimd_c_null_convert_dspr2)
/*
 * a0     = cinfo->image_width
 * a1     = input_buf
 * a2     = output_buf
 * a3     = output_row
 * 16(sp) = num_rows
 * 20(sp) = cinfo->num_components
 *
 * Null conversion for compression
 */
    SAVE_REGS_ON_STACK 8, s0, s1

    lw          t9, 24(sp)      /* t9 = num_rows */
    lw          s0, 28(sp)      /* s0 = cinfo->num_components */
    andi        t0, a0, 3       /* t0 = cinfo->image_width & 3 */
    beqz        t0, 4f          /* no residual */
     nop
0:
    addiu       t9, t9, -1
    bltz        t9, 7f
     li         t1, 0
1:
    sll         t3, t1, 2
    lwx         t5, t3(a2)      /* t5 = outptr = output_buf[ci] */
    lw          t2, 0(a1)       /* t2 = inptr = *input_buf */
    sll         t4, a3, 2
    lwx         t5, t4(t5)      /* t5 = outptr = output_buf[ci][output_row] */
    addu        t2, t2, t1
    addu        s1, t5, a0
    addu        t6, t5, t0
2:
    lbu         t3, 0(t2)
    addiu       t5, t5, 1
    sb          t3, -1(t5)
    bne         t6, t5, 2b
     addu       t2, t2, s0
3:
    lbu         t3, 0(t2)
    addu        t4, t2, s0
    addu        t7, t4, s0
    addu        t8, t7, s0
    addu        t2, t8, s0
    lbu         t4, 0(t4)
    lbu         t7, 0(t7)
    lbu         t8, 0(t8)
    addiu       t5, t5, 4
    sb          t3, -4(t5)
    sb          t4, -3(t5)
    sb          t7, -2(t5)
    bne         s1, t5, 3b
     sb         t8, -1(t5)
    addiu       t1, t1, 1
    bne         t1, s0, 1b
     nop
    addiu       a1, a1, 4
    bgez        t9, 0b
     addiu      a3, a3, 1
    b           7f
     nop
4:
    addiu       t9, t9, -1
    bltz        t9, 7f
     li         t1, 0
5:
    sll         t3, t1, 2
    lwx         t5, t3(a2)      /* t5 = outptr = output_buf[ci] */
    lw          t2, 0(a1)       /* t2 = inptr = *input_buf */
    sll         t4, a3, 2
    lwx         t5, t4(t5)      /* t5 = outptr = output_buf[ci][output_row] */
    addu        t2, t2, t1
    addu        s1, t5, a0
    addu        t6, t5, t0
6:
    lbu         t3, 0(t2)
    addu        t4, t2, s0
    addu        t7, t4, s0
    addu        t8, t7, s0
    addu        t2, t8, s0
    lbu         t4, 0(t4)
    lbu         t7, 0(t7)
    lbu         t8, 0(t8)
    addiu       t5, t5, 4
    sb          t3, -4(t5)
    sb          t4, -3(t5)
    sb          t7, -2(t5)
    bne         s1, t5, 6b
     sb         t8, -1(t5)
    addiu       t1, t1, 1
    bne         t1, s0, 5b
     nop
    addiu       a1, a1, 4
    bgez        t9, 4b
     addiu      a3, a3, 1
7:
    RESTORE_REGS_FROM_STACK 8, s0, s1

    j           ra
     nop

END(jsimd_c_null_convert_dspr2)


/*****************************************************************************/
/*
 * jsimd_extrgb_ycc_convert_dspr2
 * jsimd_extbgr_ycc_convert_dspr2
 * jsimd_extrgbx_ycc_convert_dspr2
 * jsimd_extbgrx_ycc_convert_dspr2
 * jsimd_extxbgr_ycc_convert_dspr2
 * jsimd_extxrgb_ycc_convert_dspr2
 *
 * Colorspace conversion RGB -> YCbCr
 */

.macro GENERATE_JSIMD_RGB_YCC_CONVERT_DSPR2  colorid, pixel_size, \
                                             r_offs, g_offs, b_offs

.macro DO_RGB_TO_YCC  r, g, b, inptr
    lbu         \r, \r_offs(\inptr)
    lbu         \g, \g_offs(\inptr)
    lbu         \b, \b_offs(\inptr)
    addiu       \inptr, \pixel_size
.endm

LEAF_DSPR2(jsimd_\colorid\()_ycc_convert_dspr2)
/*
 * a0     = cinfo->image_width
 * a1     = input_buf
 * a2     = output_buf
 * a3     = output_row
 * 16(sp) = num_rows
 */
    SAVE_REGS_ON_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    lw          t7, 48(sp)      /* t7 = num_rows */
    li          s0, 0x4c8b      /* FIX(0.29900) */
    li          s1, 0x9646      /* FIX(0.58700) */
    li          s2, 0x1d2f      /* FIX(0.11400) */
    li          s3, 0xffffd4cd  /* -FIX(0.16874) */
    li          s4, 0xffffab33  /* -FIX(0.33126) */
    li          s5, 0x8000      /* FIX(0.50000) */
    li          s6, 0xffff94d1  /* -FIX(0.41869) */
    li          s7, 0xffffeb2f  /* -FIX(0.08131) */
    li          t8, 0x807fff    /* CBCR_OFFSET + ONE_HALF-1 */

0:
    addiu       t7, -1          /* --num_rows */
    lw          t6, 0(a1)       /* t6 = input_buf[0] */
    lw          t0, 0(a2)
    lw          t1, 4(a2)
    lw          t2, 8(a2)
    sll         t3, a3, 2
    lwx         t0, t3(t0)      /* t0 = output_buf[0][output_row] */
    lwx         t1, t3(t1)      /* t1 = output_buf[1][output_row] */
    lwx         t2, t3(t2)      /* t2 = output_buf[2][output_row] */

    addu        t9, t2, a0      /* t9 = end address */
    addiu       a3, 1

1:
    DO_RGB_TO_YCC t3, t4, t5, t6

    mtlo        s5, $ac0
    mtlo        t8, $ac1
    mtlo        t8, $ac2
    maddu       $ac0, s2, t5
    maddu       $ac1, s5, t5
    maddu       $ac2, s5, t3
    maddu       $ac0, s0, t3
    maddu       $ac1, s3, t3
    maddu       $ac2, s6, t4
    maddu       $ac0, s1, t4
    maddu       $ac1, s4, t4
    maddu       $ac2, s7, t5
    extr.w      t3, $ac0, 16
    extr.w      t4, $ac1, 16
    extr.w      t5, $ac2, 16
    sb          t3, 0(t0)
    sb          t4, 0(t1)
    sb          t5, 0(t2)
    addiu       t0, 1
    addiu       t2, 1
    bne         t2, t9, 1b
     addiu      t1, 1
    bgtz        t7, 0b
     addiu      a1, 4

    RESTORE_REGS_FROM_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    j           ra
     nop
END(jsimd_\colorid\()_ycc_convert_dspr2)

.purgem DO_RGB_TO_YCC

.endm

/*-------------------------------------id -- pix R  G  B */
GENERATE_JSIMD_RGB_YCC_CONVERT_DSPR2 extrgb,  3, 0, 1, 2
GENERATE_JSIMD_RGB_YCC_CONVERT_DSPR2 extbgr,  3, 2, 1, 0
GENERATE_JSIMD_RGB_YCC_CONVERT_DSPR2 extrgbx, 4, 0, 1, 2
GENERATE_JSIMD_RGB_YCC_CONVERT_DSPR2 extbgrx, 4, 2, 1, 0
GENERATE_JSIMD_RGB_YCC_CONVERT_DSPR2 extxbgr, 4, 3, 2, 1
GENERATE_JSIMD_RGB_YCC_CONVERT_DSPR2 extxrgb, 4, 1, 2, 3


/*****************************************************************************/
/*
 * jsimd_ycc_extrgb_convert_dspr2
 * jsimd_ycc_extbgr_convert_dspr2
 * jsimd_ycc_extrgbx_convert_dspr2
 * jsimd_ycc_extbgrx_convert_dspr2
 * jsimd_ycc_extxbgr_convert_dspr2
 * jsimd_ycc_extxrgb_convert_dspr2
 *
 * Colorspace conversion YCbCr -> RGB
 */

.macro GENERATE_JSIMD_YCC_RGB_CONVERT_DSPR2  colorid, pixel_size, \
                                             r_offs, g_offs, b_offs, a_offs

.macro STORE_YCC_TO_RGB  scratch0 scratch1 scratch2 outptr
    sb          \scratch0, \r_offs(\outptr)
    sb          \scratch1, \g_offs(\outptr)
    sb          \scratch2, \b_offs(\outptr)
.if (\pixel_size == 4)
    li          t0, 0xFF
    sb          t0, \a_offs(\outptr)
.endif
    addiu       \outptr, \pixel_size
.endm

LEAF_DSPR2(jsimd_ycc_\colorid\()_convert_dspr2)
/*
 * a0     = cinfo->image_width
 * a1     = input_buf
 * a2     = input_row
 * a3     = output_buf
 * 16(sp) = num_rows
 */
    SAVE_REGS_ON_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    lw          s1, 48(sp)
    li          t3, 0x8000
    li          t4, 0x166e9     /* FIX(1.40200) */
    li          t5, 0x1c5a2     /* FIX(1.77200) */
    li          t6, 0xffff492e  /* -FIX(0.71414) */
    li          t7, 0xffffa7e6  /* -FIX(0.34414) */
    repl.ph     t8, 128

0:
    lw          s0, 0(a3)
    lw          t0, 0(a1)
    lw          t1, 4(a1)
    lw          t2, 8(a1)
    sll         s5, a2, 2
    addiu       s1, -1
    lwx         s2, s5(t0)
    lwx         s3, s5(t1)
    lwx         s4, s5(t2)
    addu        t9, s2, a0
    addiu       a2, 1

1:
    lbu         s7, 0(s4)       /* cr */
    lbu         s6, 0(s3)       /* cb */
    lbu         s5, 0(s2)       /* y */
    addiu       s2, 1
    addiu       s4, 1
    addiu       s7, -128
    addiu       s6, -128
    mul         t2, t7, s6
    mul         t0, t6, s7      /* Crgtab[cr] */
    sll         s7, 15
    mulq_rs.w   t1, t4, s7      /* Crrtab[cr] */
    sll         s6, 15
    addu        t2, t3          /* Cbgtab[cb] */
    addu        t2, t0

    mulq_rs.w   t0, t5, s6      /* Cbbtab[cb] */
    sra         t2, 16
    addu        t1, s5
    addu        t2, s5          /* add y */
    ins         t2, t1, 16, 16
    subu.ph     t2, t2, t8
    addu        t0, s5
    shll_s.ph   t2, t2, 8
    subu        t0, 128
    shra.ph     t2, t2, 8
    shll_s.w    t0, t0, 24
    addu.ph     t2, t2, t8      /* clip & store */
    sra         t0, t0, 24
    sra         t1, t2, 16
    addiu       t0, 128

    STORE_YCC_TO_RGB t1, t2, t0, s0

    bne         s2, t9, 1b
     addiu      s3, 1
    bgtz        s1, 0b
     addiu      a3, 4

    RESTORE_REGS_FROM_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    j           ra
     nop
END(jsimd_ycc_\colorid\()_convert_dspr2)

.purgem STORE_YCC_TO_RGB

.endm

/*-------------------------------------id -- pix R  G  B  A */
GENERATE_JSIMD_YCC_RGB_CONVERT_DSPR2 extrgb,  3, 0, 1, 2, 3
GENERATE_JSIMD_YCC_RGB_CONVERT_DSPR2 extbgr,  3, 2, 1, 0, 3
GENERATE_JSIMD_YCC_RGB_CONVERT_DSPR2 extrgbx, 4, 0, 1, 2, 3
GENERATE_JSIMD_YCC_RGB_CONVERT_DSPR2 extbgrx, 4, 2, 1, 0, 3
GENERATE_JSIMD_YCC_RGB_CONVERT_DSPR2 extxbgr, 4, 3, 2, 1, 0
GENERATE_JSIMD_YCC_RGB_CONVERT_DSPR2 extxrgb, 4, 1, 2, 3, 0


/*****************************************************************************/
/*
 * jsimd_extrgb_gray_convert_dspr2
 * jsimd_extbgr_gray_convert_dspr2
 * jsimd_extrgbx_gray_convert_dspr2
 * jsimd_extbgrx_gray_convert_dspr2
 * jsimd_extxbgr_gray_convert_dspr2
 * jsimd_extxrgb_gray_convert_dspr2
 *
 * Colorspace conversion RGB -> GRAY
 */

.macro GENERATE_JSIMD_RGB_GRAY_CONVERT_DSPR2  colorid, pixel_size, \
                                              r_offs, g_offs, b_offs

.macro DO_RGB_TO_GRAY  r, g, b, inptr
    lbu         \r, \r_offs(\inptr)
    lbu         \g, \g_offs(\inptr)
    lbu         \b, \b_offs(\inptr)
    addiu       \inptr, \pixel_size
.endm

LEAF_DSPR2(jsimd_\colorid\()_gray_convert_dspr2)
/*
 * a0     = cinfo->image_width
 * a1     = input_buf
 * a2     = output_buf
 * a3     = output_row
 * 16(sp) = num_rows
 */
    SAVE_REGS_ON_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    li          s0, 0x4c8b      /* s0 = FIX(0.29900) */
    li          s1, 0x9646      /* s1 = FIX(0.58700) */
    li          s2, 0x1d2f      /* s2 = FIX(0.11400) */
    li          s7, 0x8000      /* s7 = FIX(0.50000) */
    lw          s6, 48(sp)
    andi        t7, a0, 3

0:
    addiu       s6, -1          /* s6 = num_rows */
    lw          t0, 0(a1)
    lw          t1, 0(a2)
    sll         t3, a3, 2
    lwx         t1, t3(t1)
    addiu       a3, 1
    addu        t9, t1, a0
    subu        t8, t9, t7
    beq         t1, t8, 2f
     nop

1:
    DO_RGB_TO_GRAY t3, t4, t5, t0
    DO_RGB_TO_GRAY s3, s4, s5, t0

    mtlo        s7, $ac0
    maddu       $ac0, s2, t5
    maddu       $ac0, s1, t4
    maddu       $ac0, s0, t3
    mtlo        s7, $ac1
    maddu       $ac1, s2, s5
    maddu       $ac1, s1, s4
    maddu       $ac1, s0, s3
    extr.w      t6, $ac0, 16

    DO_RGB_TO_GRAY t3, t4, t5, t0
    DO_RGB_TO_GRAY s3, s4, s5, t0

    mtlo        s7, $ac0
    maddu       $ac0, s2, t5
    maddu       $ac0, s1, t4
    extr.w      t2, $ac1, 16
    maddu       $ac0, s0, t3
    mtlo        s7, $ac1
    maddu       $ac1, s2, s5
    maddu       $ac1, s1, s4
    maddu       $ac1, s0, s3
    extr.w      t5, $ac0, 16
    sb          t6, 0(t1)
    sb          t2, 1(t1)
    extr.w      t3, $ac1, 16
    addiu       t1, 4
    sb          t5, -2(t1)
    sb          t3, -1(t1)
    bne         t1, t8, 1b
     nop

2:
    beqz        t7, 4f
     nop

3:
    DO_RGB_TO_GRAY t3, t4, t5, t0

    mtlo        s7, $ac0
    maddu       $ac0, s2, t5
    maddu       $ac0, s1, t4
    maddu       $ac0, s0, t3
    extr.w      t6, $ac0, 16
    sb          t6, 0(t1)
    addiu       t1, 1
    bne         t1, t9, 3b
     nop

4:
    bgtz        s6, 0b
     addiu      a1, 4

    RESTORE_REGS_FROM_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    j           ra
     nop
END(jsimd_\colorid\()_gray_convert_dspr2)

.purgem DO_RGB_TO_GRAY

.endm

/*-------------------------------------id --  pix R  G  B */
GENERATE_JSIMD_RGB_GRAY_CONVERT_DSPR2 extrgb,  3, 0, 1, 2
GENERATE_JSIMD_RGB_GRAY_CONVERT_DSPR2 extbgr,  3, 2, 1, 0
GENERATE_JSIMD_RGB_GRAY_CONVERT_DSPR2 extrgbx, 4, 0, 1, 2
GENERATE_JSIMD_RGB_GRAY_CONVERT_DSPR2 extbgrx, 4, 2, 1, 0
GENERATE_JSIMD_RGB_GRAY_CONVERT_DSPR2 extxbgr, 4, 3, 2, 1
GENERATE_JSIMD_RGB_GRAY_CONVERT_DSPR2 extxrgb, 4, 1, 2, 3


/*****************************************************************************/
/*
 * jsimd_h2v2_merged_upsample_dspr2
 * jsimd_h2v2_extrgb_merged_upsample_dspr2
 * jsimd_h2v2_extrgbx_merged_upsample_dspr2
 * jsimd_h2v2_extbgr_merged_upsample_dspr2
 * jsimd_h2v2_extbgrx_merged_upsample_dspr2
 * jsimd_h2v2_extxbgr_merged_upsample_dspr2
 * jsimd_h2v2_extxrgb_merged_upsample_dspr2
 *
 * Merged h2v2 upsample routines
 */
.macro GENERATE_H2V2_MERGED_UPSAMPLE_DSPR2  colorid, pixel_size, \
                                            r1_offs, g1_offs, \
                                            b1_offs, a1_offs, \
                                            r2_offs, g2_offs, \
                                            b2_offs, a2_offs

.macro STORE_H2V2_2_PIXELS  scratch0 scratch1 scratch2 scratch3 scratch4 \
                            scratch5 outptr
    sb          \scratch0, \r1_offs(\outptr)
    sb          \scratch1, \g1_offs(\outptr)
    sb          \scratch2, \b1_offs(\outptr)
    sb          \scratch3, \r2_offs(\outptr)
    sb          \scratch4, \g2_offs(\outptr)
    sb          \scratch5, \b2_offs(\outptr)
.if (\pixel_size == 8)
    li          \scratch0, 0xFF
    sb          \scratch0, \a1_offs(\outptr)
    sb          \scratch0, \a2_offs(\outptr)
.endif
    addiu       \outptr, \pixel_size
.endm

.macro STORE_H2V2_1_PIXEL  scratch0 scratch1 scratch2 outptr
    sb          \scratch0, \r1_offs(\outptr)
    sb          \scratch1, \g1_offs(\outptr)
    sb          \scratch2, \b1_offs(\outptr)

.if (\pixel_size == 8)
    li          t0, 0xFF
    sb          t0, \a1_offs(\outptr)
.endif
.endm

LEAF_DSPR2(jsimd_h2v2_\colorid\()_merged_upsample_dspr2)
/*
 * a0     = cinfo->output_width
 * a1     = input_buf
 * a2     = in_row_group_ctr
 * a3     = output_buf
 * 16(sp) = cinfo->sample_range_limit
 */
    SAVE_REGS_ON_STACK 40, s0, s1, s2, s3, s4, s5, s6, s7, ra

    lw          t9, 56(sp)      /* cinfo->sample_range_limit */
    lw          v0, 0(a1)
    lw          v1, 4(a1)
    lw          t0, 8(a1)
    sll         t1, a2, 3
    addiu       t2, t1, 4
    sll         t3, a2, 2
    lw          t4, 0(a3)       /* t4 = output_buf[0] */
    lwx         t1, t1(v0)      /* t1 = input_buf[0][in_row_group_ctr*2] */
    lwx         t2, t2(v0)      /* t2 = input_buf[0][in_row_group_ctr*2 + 1] */
    lwx         t5, t3(v1)      /* t5 = input_buf[1][in_row_group_ctr] */
    lwx         t6, t3(t0)      /* t6 = input_buf[2][in_row_group_ctr] */
    lw          t7, 4(a3)       /* t7 = output_buf[1] */
    li          s1, 0xe6ea
    addiu       t8, s1, 0x7fff    /* t8 = 0x166e9 [FIX(1.40200)] */
    addiu       s0, t8, 0x5eb9    /* s0 = 0x1c5a2 [FIX(1.77200)] */
    addiu       s1, zero, 0xa7e6  /* s4 = 0xffffa7e6 [-FIX(0.34414)] */
    xori        s2, s1, 0xeec8    /* s3 = 0xffff492e [-FIX(0.71414)] */
    srl         t3, a0, 1
    blez        t3, 2f
     addu       t0, t5, t3      /* t0 = end address */
 1:
    lbu         t3, 0(t5)
    lbu         s3, 0(t6)
    addiu       t5, t5, 1
    addiu       t3, t3, -128    /* (cb - 128) */
    addiu       s3, s3, -128    /* (cr - 128) */
    mult        $ac1, s1, t3
    madd        $ac1, s2, s3
    sll         s3, s3, 15
    sll         t3, t3, 15
    mulq_rs.w   s4, t8, s3      /* s4 = (C1 * cr + ONE_HALF)>> SCALEBITS */
    extr_r.w    s5, $ac1, 16
    mulq_rs.w   s6, s0, t3      /* s6 = (C2 * cb + ONE_HALF)>> SCALEBITS */
    lbu         v0, 0(t1)
    addiu       t6, t6, 1
    addiu       t1, t1, 2
    addu        t3, v0, s4      /* y+cred */
    addu        s3, v0, s5      /* y+cgreen */
    addu        v1, v0, s6      /* y+cblue */
    addu        t3, t9, t3      /* y+cred */
    addu        s3, t9, s3      /* y+cgreen */
    addu        v1, t9, v1      /* y+cblue */
    lbu         AT, 0(t3)
    lbu         s7, 0(s3)
    lbu         ra, 0(v1)
    lbu         v0, -1(t1)
    addu        t3, v0, s4      /* y+cred */
    addu        s3, v0, s5      /* y+cgreen */
    addu        v1, v0, s6      /* y+cblue */
    addu        t3, t9, t3      /* y+cred */
    addu        s3, t9, s3      /* y+cgreen */
    addu        v1, t9, v1      /* y+cblue */
    lbu         t3, 0(t3)
    lbu         s3, 0(s3)
    lbu         v1, 0(v1)
    lbu         v0, 0(t2)

    STORE_H2V2_2_PIXELS AT, s7, ra, t3, s3, v1, t4

    addu        t3, v0, s4      /* y+cred */
    addu        s3, v0, s5      /* y+cgreen */
    addu        v1, v0, s6      /* y+cblue */
    addu        t3, t9, t3      /* y+cred */
    addu        s3, t9, s3      /* y+cgreen */
    addu        v1, t9, v1      /* y+cblue */
    lbu         AT, 0(t3)
    lbu         s7, 0(s3)
    lbu         ra, 0(v1)
    lbu         v0, 1(t2)
    addiu       t2, t2, 2
    addu        t3, v0, s4      /* y+cred */
    addu        s3, v0, s5      /* y+cgreen */
    addu        v1, v0, s6      /* y+cblue */
    addu        t3, t9, t3      /* y+cred */
    addu        s3, t9, s3      /* y+cgreen */
    addu        v1, t9, v1      /* y+cblue */
    lbu         t3, 0(t3)
    lbu         s3, 0(s3)
    lbu         v1, 0(v1)

    STORE_H2V2_2_PIXELS AT, s7, ra, t3, s3, v1, t7

    bne         t0, t5, 1b
     nop
2:
    andi        t0, a0, 1
    beqz        t0, 4f
     lbu        t3, 0(t5)
    lbu         s3, 0(t6)
    addiu       t3, t3, -128    /* (cb - 128) */
    addiu       s3, s3, -128    /* (cr - 128) */
    mult        $ac1, s1, t3
    madd        $ac1, s2, s3
    sll         s3, s3, 15
    sll         t3, t3, 15
    lbu         v0, 0(t1)
    extr_r.w    s5, $ac1, 16
    mulq_rs.w   s4, t8, s3      /* s4 = (C1 * cr + ONE_HALF)>> SCALEBITS */
    mulq_rs.w   s6, s0, t3      /* s6 = (C2 * cb + ONE_HALF)>> SCALEBITS */
    addu        t3, v0, s4      /* y+cred */
    addu        s3, v0, s5      /* y+cgreen */
    addu        v1, v0, s6      /* y+cblue */
    addu        t3, t9, t3      /* y+cred */
    addu        s3, t9, s3      /* y+cgreen */
    addu        v1, t9, v1      /* y+cblue */
    lbu         t3, 0(t3)
    lbu         s3, 0(s3)
    lbu         v1, 0(v1)
    lbu         v0, 0(t2)

    STORE_H2V2_1_PIXEL t3, s3, v1, t4

    addu        t3, v0, s4      /* y+cred */
    addu        s3, v0, s5      /* y+cgreen */
    addu        v1, v0, s6      /* y+cblue */
    addu        t3, t9, t3      /* y+cred */
    addu        s3, t9, s3      /* y+cgreen */
    addu        v1, t9, v1      /* y+cblue */
    lbu         t3, 0(t3)
    lbu         s3, 0(s3)
    lbu         v1, 0(v1)

    STORE_H2V2_1_PIXEL t3, s3, v1, t7
4:
    RESTORE_REGS_FROM_STACK 40, s0, s1, s2, s3, s4, s5, s6, s7, ra

    j           ra
     nop

END(jsimd_h2v2_\colorid\()_merged_upsample_dspr2)

.purgem STORE_H2V2_1_PIXEL
.purgem STORE_H2V2_2_PIXELS
.endm

/*------------------------------------id -- pix R1 G1 B1 A1 R2 G2 B2 A2 */
GENERATE_H2V2_MERGED_UPSAMPLE_DSPR2 extrgb,  6, 0, 1, 2, 6, 3, 4, 5, 6
GENERATE_H2V2_MERGED_UPSAMPLE_DSPR2 extbgr,  6, 2, 1, 0, 3, 5, 4, 3, 6
GENERATE_H2V2_MERGED_UPSAMPLE_DSPR2 extrgbx, 8, 0, 1, 2, 3, 4, 5, 6, 7
GENERATE_H2V2_MERGED_UPSAMPLE_DSPR2 extbgrx, 8, 2, 1, 0, 3, 6, 5, 4, 7
GENERATE_H2V2_MERGED_UPSAMPLE_DSPR2 extxbgr, 8, 3, 2, 1, 0, 7, 6, 5, 4
GENERATE_H2V2_MERGED_UPSAMPLE_DSPR2 extxrgb, 8, 1, 2, 3, 0, 5, 6, 7, 4


/*****************************************************************************/
/*
 * jsimd_h2v1_merged_upsample_dspr2
 * jsimd_h2v1_extrgb_merged_upsample_dspr2
 * jsimd_h2v1_extrgbx_merged_upsample_dspr2
 * jsimd_h2v1_extbgr_merged_upsample_dspr2
 * jsimd_h2v1_extbgrx_merged_upsample_dspr2
 * jsimd_h2v1_extxbgr_merged_upsample_dspr2
 * jsimd_h2v1_extxrgb_merged_upsample_dspr2
 *
 * Merged h2v1 upsample routines
 */

.macro GENERATE_H2V1_MERGED_UPSAMPLE_DSPR2  colorid, pixel_size, \
                                            r1_offs, g1_offs, \
                                            b1_offs, a1_offs, \
                                            r2_offs, g2_offs, \
                                            b2_offs, a2_offs

.macro STORE_H2V1_2_PIXELS  scratch0 scratch1 scratch2 scratch3 scratch4 \
                            scratch5 outptr
    sb          \scratch0, \r1_offs(\outptr)
    sb          \scratch1, \g1_offs(\outptr)
    sb          \scratch2, \b1_offs(\outptr)
    sb          \scratch3, \r2_offs(\outptr)
    sb          \scratch4, \g2_offs(\outptr)
    sb          \scratch5, \b2_offs(\outptr)
.if (\pixel_size == 8)
    li          t0, 0xFF
    sb          t0, \a1_offs(\outptr)
    sb          t0, \a2_offs(\outptr)
.endif
    addiu       \outptr, \pixel_size
.endm

.macro STORE_H2V1_1_PIXEL  scratch0 scratch1 scratch2 outptr
    sb          \scratch0, \r1_offs(\outptr)
    sb          \scratch1, \g1_offs(\outptr)
    sb          \scratch2, \b1_offs(\outptr)
.if (\pixel_size == 8)
    li          t0, 0xFF
    sb          t0, \a1_offs(\outptr)
.endif
.endm

LEAF_DSPR2(jsimd_h2v1_\colorid\()_merged_upsample_dspr2)
/*
 * a0     = cinfo->output_width
 * a1     = input_buf
 * a2     = in_row_group_ctr
 * a3     = output_buf
 * 16(sp) = range_limit
 */
    SAVE_REGS_ON_STACK 40, s0, s1, s2, s3, s4, s5, s6, s7, ra

    li          t0, 0xe6ea
    lw          t1, 0(a1)         /* t1 = input_buf[0] */
    lw          t2, 4(a1)         /* t2 = input_buf[1] */
    lw          t3, 8(a1)         /* t3 = input_buf[2] */
    lw          t8, 56(sp)        /* t8 = range_limit */
    addiu       s1, t0, 0x7fff    /* s1 = 0x166e9 [FIX(1.40200)] */
    addiu       s2, s1, 0x5eb9    /* s2 = 0x1c5a2 [FIX(1.77200)] */
    addiu       s0, t0, 0x9916    /* s0 = 0x8000 */
    addiu       s4, zero, 0xa7e6  /* s4 = 0xffffa7e6 [-FIX(0.34414)] */
    xori        s3, s4, 0xeec8    /* s3 = 0xffff492e [-FIX(0.71414)] */
    srl         t0, a0, 1
    sll         t4, a2, 2
    lwx         s5, t4(t1)      /* s5 = inptr0 */
    lwx         s6, t4(t2)      /* s6 = inptr1 */
    lwx         s7, t4(t3)      /* s7 = inptr2 */
    lw          t7, 0(a3)       /* t7 = outptr */
    blez        t0, 2f
     addu       t9, s6, t0      /* t9 = end address */
1:
    lbu         t2, 0(s6)       /* t2 = cb */
    lbu         t0, 0(s7)       /* t0 = cr */
    lbu         t1, 0(s5)       /* t1 = y */
    addiu       t2, t2, -128    /* t2 = cb - 128 */
    addiu       t0, t0, -128    /* t0 = cr - 128 */
    mult        $ac1, s4, t2
    madd        $ac1, s3, t0
    sll         t0, t0, 15
    sll         t2, t2, 15
    mulq_rs.w   t0, s1, t0      /* t0 = (C1*cr + ONE_HALF)>> SCALEBITS */
    extr_r.w    t5, $ac1, 16
    mulq_rs.w   t6, s2, t2      /* t6 = (C2*cb + ONE_HALF)>> SCALEBITS */
    addiu       s7, s7, 1
    addiu       s6, s6, 1
    addu        t2, t1, t0      /* t2 = y + cred */
    addu        t3, t1, t5      /* t3 = y + cgreen */
    addu        t4, t1, t6      /* t4 = y + cblue */
    addu        t2, t8, t2
    addu        t3, t8, t3
    addu        t4, t8, t4
    lbu         t1, 1(s5)
    lbu         v0, 0(t2)
    lbu         v1, 0(t3)
    lbu         ra, 0(t4)
    addu        t2, t1, t0
    addu        t3, t1, t5
    addu        t4, t1, t6
    addu        t2, t8, t2
    addu        t3, t8, t3
    addu        t4, t8, t4
    lbu         t2, 0(t2)
    lbu         t3, 0(t3)
    lbu         t4, 0(t4)

    STORE_H2V1_2_PIXELS v0, v1, ra, t2, t3, t4, t7

    bne         t9, s6, 1b
     addiu      s5, s5, 2
2:
    andi        t0, a0, 1
    beqz        t0, 4f
     nop
3:
    lbu         t2, 0(s6)
    lbu         t0, 0(s7)
    lbu         t1, 0(s5)
    addiu       t2, t2, -128    /* (cb - 128) */
    addiu       t0, t0, -128    /* (cr - 128) */
    mul         t3, s4, t2
    mul         t4, s3, t0
    sll         t0, t0, 15
    sll         t2, t2, 15
    mulq_rs.w   t0, s1, t0      /* (C1*cr + ONE_HALF)>> SCALEBITS */
    mulq_rs.w   t6, s2, t2      /* (C2*cb + ONE_HALF)>> SCALEBITS */
    addu        t3, t3, s0
    addu        t3, t4, t3
    sra         t5, t3, 16      /* (C4*cb + ONE_HALF + C3*cr)>> SCALEBITS */
    addu        t2, t1, t0      /* y + cred */
    addu        t3, t1, t5      /* y + cgreen */
    addu        t4, t1, t6      /* y + cblue */
    addu        t2, t8, t2
    addu        t3, t8, t3
    addu        t4, t8, t4
    lbu         t2, 0(t2)
    lbu         t3, 0(t3)
    lbu         t4, 0(t4)

    STORE_H2V1_1_PIXEL t2, t3, t4, t7
4:
    RESTORE_REGS_FROM_STACK 40, s0, s1, s2, s3, s4, s5, s6, s7, ra

    j           ra
     nop

END(jsimd_h2v1_\colorid\()_merged_upsample_dspr2)

.purgem STORE_H2V1_1_PIXEL
.purgem STORE_H2V1_2_PIXELS
.endm

/*------------------------------------id -- pix R1 G1 B1 A1 R2 G2 B2 A2 */
GENERATE_H2V1_MERGED_UPSAMPLE_DSPR2 extrgb,  6, 0, 1, 2, 6, 3, 4, 5, 6
GENERATE_H2V1_MERGED_UPSAMPLE_DSPR2 extbgr,  6, 2, 1, 0, 3, 5, 4, 3, 6
GENERATE_H2V1_MERGED_UPSAMPLE_DSPR2 extrgbx, 8, 0, 1, 2, 3, 4, 5, 6, 7
GENERATE_H2V1_MERGED_UPSAMPLE_DSPR2 extbgrx, 8, 2, 1, 0, 3, 6, 5, 4, 7
GENERATE_H2V1_MERGED_UPSAMPLE_DSPR2 extxbgr, 8, 3, 2, 1, 0, 7, 6, 5, 4
GENERATE_H2V1_MERGED_UPSAMPLE_DSPR2 extxrgb, 8, 1, 2, 3, 0, 5, 6, 7, 4


/*****************************************************************************/
/*
 * jsimd_h2v2_fancy_upsample_dspr2
 *
 * Fancy processing for the common case of 2:1 horizontal and 2:1 vertical.
 */
LEAF_DSPR2(jsimd_h2v2_fancy_upsample_dspr2)
/*
 * a0 = cinfo->max_v_samp_factor
 * a1 = downsampled_width
 * a2 = input_data
 * a3 = output_data_ptr
 */
    SAVE_REGS_ON_STACK 24, s0, s1, s2, s3, s4, s5

    li            s4, 0
    lw            s2, 0(a3)       /* s2 = *output_data_ptr */
0:
    li            t9, 2
    lw            s1, -4(a2)      /* s1 = inptr1 */

1:
    lw            s0, 0(a2)       /* s0 = inptr0 */
    lwx           s3, s4(s2)
    addiu         s5, a1, -2      /* s5 = downsampled_width - 2 */
    srl           t4, s5, 1
    sll           t4, t4, 1
    lbu           t0, 0(s0)
    lbu           t1, 1(s0)
    lbu           t2, 0(s1)
    lbu           t3, 1(s1)
    addiu         s0, 2
    addiu         s1, 2
    addu          t8, s0, t4      /* t8 = end address */
    andi          s5, s5, 1       /* s5 = residual */
    sll           t4, t0, 1
    sll           t6, t1, 1
    addu          t0, t0, t4      /* t0 = (*inptr0++) * 3 */
    addu          t1, t1, t6      /* t1 = (*inptr0++) * 3 */
    addu          t7, t0, t2      /* t7 = thiscolsum */
    addu          t6, t1, t3      /* t5 = nextcolsum */
    sll           t0, t7, 2       /* t0 = thiscolsum * 4 */
    subu          t1, t0, t7      /* t1 = thiscolsum * 3 */
    shra_r.w      t0, t0, 4
    addiu         t1, 7
    addu          t1, t1, t6
    srl           t1, t1, 4
    sb            t0, 0(s3)
    sb            t1, 1(s3)
    beq           t8, s0, 22f     /* skip to final iteration if width == 3 */
     addiu        s3, 2
2:
    lh            t0, 0(s0)       /* t0 = A3|A2 */
    lh            t2, 0(s1)       /* t2 = B3|B2 */
    addiu         s0, 2
    addiu         s1, 2
    preceu.ph.qbr t0, t0          /* t0 = 0|A3|0|A2 */
    preceu.ph.qbr t2, t2          /* t2 = 0|B3|0|B2 */
    shll.ph       t1, t0, 1
    sll           t3, t6, 1
    addu.ph       t0, t1, t0      /* t0 = A3*3|A2*3 */
    addu          t3, t3, t6      /* t3 = this * 3 */
    addu.ph       t0, t0, t2      /* t0 = next2|next1 */
    addu          t1, t3, t7
    andi          t7, t0, 0xFFFF  /* t7 = next1 */
    sll           t2, t7, 1
    addu          t2, t7, t2      /* t2 = next1*3 */
    addu          t4, t2, t6
    srl           t6, t0, 16      /* t6 = next2 */
    shra_r.w      t1, t1, 4       /* t1 = (this*3 + last + 8) >> 4 */
    addu          t0, t3, t7
    addiu         t0, 7
    srl           t0, t0, 4       /* t0 = (this*3 + next1 + 7) >> 4 */
    shra_r.w      t4, t4, 4       /* t3 = (next1*3 + this + 8) >> 4 */
    addu          t2, t2, t6
    addiu         t2, 7
    srl           t2, t2, 4       /* t2 = (next1*3 + next2 + 7) >> 4 */
    sb            t1, 0(s3)
    sb            t0, 1(s3)
    sb            t4, 2(s3)
    sb            t2, 3(s3)
    bne           t8, s0, 2b
     addiu        s3, 4
22:
    beqz          s5, 4f
     addu         t8, s0, s5
3:
    lbu           t0, 0(s0)
    lbu           t2, 0(s1)
    addiu         s0, 1
    addiu         s1, 1
    sll           t3, t6, 1
    sll           t1, t0, 1
    addu          t1, t0, t1      /* t1 = inptr0 * 3 */
    addu          t3, t3, t6      /* t3 = thiscolsum * 3 */
    addu          t5, t1, t2
    addu          t1, t3, t7
    shra_r.w      t1, t1, 4
    addu          t0, t3, t5
    addiu         t0, 7
    srl           t0, t0, 4
    sb            t1, 0(s3)
    sb            t0, 1(s3)
    addiu         s3, 2
    move          t7, t6
    bne           t8, s0, 3b
     move         t6, t5
4:
    sll           t0, t6, 2       /* t0 = thiscolsum * 4 */
    subu          t1, t0, t6      /* t1 = thiscolsum * 3 */
    addu          t1, t1, t7
    addiu         s4, 4
    shra_r.w      t1, t1, 4
    addiu         t0, 7
    srl           t0, t0, 4
    sb            t1, 0(s3)
    sb            t0, 1(s3)
    addiu         t9, -1
    addiu         s3, 2
    bnez          t9, 1b
     lw           s1, 4(a2)
    srl           t0, s4, 2
    subu          t0, a0, t0
    bgtz          t0, 0b
     addiu        a2, 4

    RESTORE_REGS_FROM_STACK 24, s0, s1, s2, s3, s4, s5

    j             ra
     nop
END(jsimd_h2v2_fancy_upsample_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_h2v1_fancy_upsample_dspr2)
/*
 * a0 = cinfo->max_v_samp_factor
 * a1 = downsampled_width
 * a2 = input_data
 * a3 = output_data_ptr
 */
    SAVE_REGS_ON_STACK 16, s0, s1, s2, s3

    .set at

    beqz          a0, 3f
     sll          t0, a0, 2
    lw            s1, 0(a3)
    li            s3, 0x10001
    addu          s0, s1, t0
0:
    addiu         t8, a1, -2
    srl           t9, t8, 2
    lw            t7, 0(a2)
    lw            s2, 0(s1)
    lbu           t0, 0(t7)
    lbu           t1, 1(t7)       /* t1 = inptr[1] */
    sll           t2, t0, 1
    addu          t2, t2, t0      /* t2 = invalue*3 */
    addu          t2, t2, t1
    shra_r.w      t2, t2, 2
    sb            t0, 0(s2)
    sb            t2, 1(s2)
    beqz          t9, 11f
     addiu        s2, 2
1:
    ulw           t0, 0(t7)       /* t0 = |P3|P2|P1|P0| */
    ulw           t1, 1(t7)
    ulh           t2, 4(t7)       /* t2 = |0|0|P5|P4| */
    preceu.ph.qbl t3, t0          /* t3 = |0|P3|0|P2| */
    preceu.ph.qbr t0, t0          /* t0 = |0|P1|0|P0| */
    preceu.ph.qbr t2, t2          /* t2 = |0|P5|0|P4| */
    preceu.ph.qbl t4, t1          /* t4 = |0|P4|0|P3| */
    preceu.ph.qbr t1, t1          /* t1 = |0|P2|0|P1| */
    shll.ph       t5, t4, 1
    shll.ph       t6, t1, 1
    addu.ph       t5, t5, t4      /* t5 = |P4*3|P3*3| */
    addu.ph       t6, t6, t1      /* t6 = |P2*3|P1*3| */
    addu.ph       t4, t3, s3
    addu.ph       t0, t0, s3
    addu.ph       t4, t4, t5
    addu.ph       t0, t0, t6
    shrl.ph       t4, t4, 2       /* t4 = |0|P3|0|P2| */
    shrl.ph       t0, t0, 2       /* t0 = |0|P1|0|P0| */
    addu.ph       t2, t2, t5
    addu.ph       t3, t3, t6
    shra_r.ph     t2, t2, 2       /* t2 = |0|P5|0|P4| */
    shra_r.ph     t3, t3, 2       /* t3 = |0|P3|0|P2| */
    shll.ph       t2, t2, 8
    shll.ph       t3, t3, 8
    or            t2, t4, t2
    or            t3, t3, t0
    addiu         t9, -1
    usw           t3, 0(s2)
    usw           t2, 4(s2)
    addiu         s2, 8
    bgtz          t9, 1b
     addiu        t7, 4
11:
    andi          t8, 3
    beqz          t8, 22f
     addiu        t7, 1

2:
    lbu           t0, 0(t7)
    addiu         t7, 1
    sll           t1, t0, 1
    addu          t2, t0, t1      /* t2 = invalue */
    lbu           t3, -2(t7)
    lbu           t4, 0(t7)
    addiu         t3, 1
    addiu         t4, 2
    addu          t3, t3, t2
    addu          t4, t4, t2
    srl           t3, 2
    srl           t4, 2
    sb            t3, 0(s2)
    sb            t4, 1(s2)
    addiu         t8, -1
    bgtz          t8, 2b
     addiu        s2, 2

22:
    lbu           t0, 0(t7)
    lbu           t2, -1(t7)
    sll           t1, t0, 1
    addu          t1, t1, t0      /* t1 = invalue * 3 */
    addu          t1, t1, t2
    addiu         t1, 1
    srl           t1, t1, 2
    sb            t1, 0(s2)
    sb            t0, 1(s2)
    addiu         s1, 4
    bne           s1, s0, 0b
     addiu        a2, 4
3:
    RESTORE_REGS_FROM_STACK 16, s0, s1, s2, s3

    j             ra
     nop
END(jsimd_h2v1_fancy_upsample_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_h2v1_downsample_dspr2)
/*
 * a0     = cinfo->image_width
 * a1     = cinfo->max_v_samp_factor
 * a2     = compptr->v_samp_factor
 * a3     = compptr->width_in_blocks
 * 16(sp) = input_data
 * 20(sp) = output_data
 */
    .set at

    SAVE_REGS_ON_STACK 24, s0, s1, s2, s3, s4

    beqz        a2, 7f
     lw         s1, 44(sp)      /* s1 = output_data */
    lw          s0, 40(sp)      /* s0 = input_data */
    srl         s2, a0, 2
    andi        t9, a0, 2
    srl         t7, t9, 1
    addu        s2, t7, s2
    sll         t0, a3, 3       /* t0 = width_in_blocks*DCT */
    srl         t7, t0, 1
    subu        s2, t7, s2
0:
    andi        t6, a0, 1       /* t6 = temp_index */
    addiu       t6, -1
    lw          t4, 0(s1)       /* t4 = outptr */
    lw          t5, 0(s0)       /* t5 = inptr0 */
    li          s3, 0           /* s3 = bias */
    srl         t7, a0, 1       /* t7 = image_width1 */
    srl         s4, t7, 2
    andi        t8, t7, 3
1:
    ulhu        t0, 0(t5)
    ulhu        t1, 2(t5)
    ulhu        t2, 4(t5)
    ulhu        t3, 6(t5)
    raddu.w.qb  t0, t0
    raddu.w.qb  t1, t1
    raddu.w.qb  t2, t2
    raddu.w.qb  t3, t3
    shra.ph     t0, t0, 1
    shra_r.ph   t1, t1, 1
    shra.ph     t2, t2, 1
    shra_r.ph   t3, t3, 1
    sb          t0, 0(t4)
    sb          t1, 1(t4)
    sb          t2, 2(t4)
    sb          t3, 3(t4)
    addiu       s4, -1
    addiu       t4, 4
    bgtz        s4, 1b
     addiu      t5, 8
    beqz        t8, 3f
     addu       s4, t4, t8
2:
    ulhu        t0, 0(t5)
    raddu.w.qb  t0, t0
    addqh.w     t0, t0, s3
    xori        s3, s3, 1
    sb          t0, 0(t4)
    addiu       t4, 1
    bne         t4, s4, 2b
     addiu      t5, 2
3:
    lbux        t1, t6(t5)
    sll         t1, 1
    addqh.w     t2, t1, s3      /* t2 = pixval1 */
    xori        s3, s3, 1
    addqh.w     t3, t1, s3      /* t3 = pixval2 */
    blez        s2, 5f
     append     t3, t2,  8
    addu        t5, t4, s2      /* t5 = loop_end2 */
4:
    ush         t3, 0(t4)
    addiu       s2, -1
    bgtz        s2, 4b
     addiu      t4,  2
5:
    beqz        t9, 6f
     nop
    sb          t2, 0(t4)
6:
    addiu       s1, 4
    addiu       a2, -1
    bnez        a2, 0b
     addiu      s0, 4
7:
    RESTORE_REGS_FROM_STACK 24, s0, s1, s2, s3, s4

    j           ra
    nop
END(jsimd_h2v1_downsample_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_h2v2_downsample_dspr2)
/*
 * a0     = cinfo->image_width
 * a1     = cinfo->max_v_samp_factor
 * a2     = compptr->v_samp_factor
 * a3     = compptr->width_in_blocks
 * 16(sp) = input_data
 * 20(sp) = output_data
 */
    .set at

    SAVE_REGS_ON_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    beqz        a2, 8f
     lw         s1, 52(sp)      /* s1 = output_data */
    lw          s0, 48(sp)      /* s0 = input_data */

    andi        t6, a0, 1       /* t6 = temp_index */
    addiu       t6, -1
    srl         t7, a0, 1       /* t7 = image_width1 */
    srl         s4, t7, 2
    andi        t8, t7, 3
    andi        t9, a0, 2
    srl         s2, a0, 2
    srl         t7, t9, 1
    addu        s2, t7, s2
    sll         t0, a3, 3       /* s2 = width_in_blocks*DCT */
    srl         t7, t0, 1
    subu        s2, t7, s2
0:
    lw          t4, 0(s1)       /* t4 = outptr */
    lw          t5, 0(s0)       /* t5 = inptr0 */
    lw          s7, 4(s0)       /* s7 = inptr1 */
    li          s6, 1           /* s6 = bias */
2:
    ulw         t0, 0(t5)       /* t0 = |P3|P2|P1|P0| */
    ulw         t1, 0(s7)       /* t1 = |Q3|Q2|Q1|Q0| */
    ulw         t2, 4(t5)
    ulw         t3, 4(s7)
    precrq.ph.w t7, t0, t1      /* t2 = |P3|P2|Q3|Q2| */
    ins         t0, t1, 16, 16  /* t0 = |Q1|Q0|P1|P0| */
    raddu.w.qb  t1, t7
    raddu.w.qb  t0, t0
    shra_r.w    t1, t1, 2
    addiu       t0, 1
    srl         t0, 2
    precrq.ph.w t7, t2, t3
    ins         t2, t3, 16, 16
    raddu.w.qb  t7, t7
    raddu.w.qb  t2, t2
    shra_r.w    t7, t7, 2
    addiu       t2, 1
    srl         t2, 2
    sb          t0, 0(t4)
    sb          t1, 1(t4)
    sb          t2, 2(t4)
    sb          t7, 3(t4)
    addiu       t4, 4
    addiu       t5, 8
    addiu       s4, s4, -1
    bgtz        s4, 2b
     addiu      s7, 8
    beqz        t8, 4f
     addu       t8, t4, t8
3:
    ulhu        t0, 0(t5)
    ulhu        t1, 0(s7)
    ins         t0, t1, 16, 16
    raddu.w.qb  t0, t0
    addu        t0, t0, s6
    srl         t0, 2
    xori        s6, s6, 3
    sb          t0, 0(t4)
    addiu       t5, 2
    addiu       t4, 1
    bne         t8, t4, 3b
     addiu      s7, 2
4:
    lbux        t1, t6(t5)
    sll         t1, 1
    lbux        t0, t6(s7)
    sll         t0, 1
    addu        t1, t1, t0
    addu        t3, t1, s6
    srl         t0, t3, 2       /* t2 = pixval1 */
    xori        s6, s6, 3
    addu        t2, t1, s6
    srl         t1, t2, 2       /* t3 = pixval2 */
    blez        s2, 6f
     append     t1, t0, 8
5:
    ush         t1, 0(t4)
    addiu       s2, -1
    bgtz        s2, 5b
     addiu      t4, 2
6:
    beqz        t9, 7f
     nop
    sb          t0, 0(t4)
7:
    addiu       s1, 4
    addiu       a2, -1
    bnez        a2, 0b
     addiu      s0, 8
8:
    RESTORE_REGS_FROM_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    j           ra
     nop
END(jsimd_h2v2_downsample_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_h2v2_smooth_downsample_dspr2)
/*
 * a0     = input_data
 * a1     = output_data
 * a2     = compptr->v_samp_factor
 * a3     = cinfo->max_v_samp_factor
 * 16(sp) = cinfo->smoothing_factor
 * 20(sp) = compptr->width_in_blocks
 * 24(sp) = cinfo->image_width
 */
    .set at

    SAVE_REGS_ON_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    lw          s7, 52(sp)      /* compptr->width_in_blocks */
    lw          s0, 56(sp)      /* cinfo->image_width */
    lw          s6, 48(sp)      /* cinfo->smoothing_factor */
    sll         s7, 3           /* output_cols = width_in_blocks * DCTSIZE */
    sll         v0, s7, 1
    subu        v0, v0, s0
    blez        v0, 2f
    move        v1, zero
    addiu       t0, a3, 2       /* t0 = cinfo->max_v_samp_factor + 2 */
0:
    addiu       t1, a0, -4
    sll         t2, v1, 2
    lwx         t1, t2(t1)
    move        t3, v0
    addu        t1, t1, s0
    lbu         t2, -1(t1)
1:
    addiu       t3, t3, -1
    sb          t2, 0(t1)
    bgtz        t3, 1b
    addiu       t1, t1, 1
    addiu       v1, v1, 1
    bne         v1, t0, 0b
    nop
2:
    li          v0, 80
    mul         v0, s6, v0
    li          v1, 16384
    move        t4, zero
    move        t5, zero
    subu        t6, v1, v0      /* t6 = 16384 - tmp_smoot_f * 80 */
    sll         t7, s6, 4       /* t7 = tmp_smoot_f * 16 */
3:
/* Special case for first column: pretend column -1 is same as column 0 */
    sll         v0, t4, 2
    lwx         t8, v0(a1)      /*  outptr = output_data[outrow] */
    sll         v1, t5, 2
    addiu       t9, v1, 4
    addiu       s0, v1, -4
    addiu       s1, v1, 8
    lwx         s2, v1(a0)      /* inptr0 = input_data[inrow] */
    lwx         t9, t9(a0)      /* inptr1 = input_data[inrow+1] */
    lwx         s0, s0(a0)      /* above_ptr = input_data[inrow-1] */
    lwx         s1, s1(a0)      /* below_ptr = input_data[inrow+2] */
    lh          v0, 0(s2)
    lh          v1, 0(t9)
    lh          t0, 0(s0)
    lh          t1, 0(s1)
    ins         v0, v1, 16, 16
    ins         t0, t1, 16, 16
    raddu.w.qb  t2, v0
    raddu.w.qb  s3, t0
    lbu         v0, 0(s2)
    lbu         v1, 2(s2)
    lbu         t0, 0(t9)
    lbu         t1, 2(t9)
    addu        v0, v0, v1
    mult        $ac1, t2, t6
    addu        t0, t0, t1
    lbu         t2, 2(s0)
    addu        t0, t0, v0
    lbu         t3, 2(s1)
    addu        s3, t0, s3
    lbu         v0, 0(s0)
    lbu         t0, 0(s1)
    sll         s3, s3, 1
    addu        v0, v0, t2
    addu        t0, t0, t3
    addu        t0, t0, v0
    addu        s3, t0, s3
    madd        $ac1, s3, t7
    extr_r.w    v0, $ac1, 16
    addiu       t8, t8, 1
    addiu       s2, s2, 2
    addiu       t9, t9, 2
    addiu       s0, s0, 2
    addiu       s1, s1, 2
    sb          v0, -1(t8)
    addiu       s4, s7, -2
    and         s4, s4, 3
    addu        s5, s4, t8      /* end address */
4:
    lh          v0, 0(s2)
    lh          v1, 0(t9)
    lh          t0, 0(s0)
    lh          t1, 0(s1)
    ins         v0, v1, 16, 16
    ins         t0, t1, 16, 16
    raddu.w.qb  t2, v0
    raddu.w.qb  s3, t0
    lbu         v0, -1(s2)
    lbu         v1, 2(s2)
    lbu         t0, -1(t9)
    lbu         t1, 2(t9)
    addu        v0, v0, v1
    mult        $ac1, t2, t6
    addu        t0, t0, t1
    lbu         t2, 2(s0)
    addu        t0, t0, v0
    lbu         t3, 2(s1)
    addu        s3, t0, s3
    lbu         v0, -1(s0)
    lbu         t0, -1(s1)
    sll         s3, s3, 1
    addu        v0, v0, t2
    addu        t0, t0, t3
    addu        t0, t0, v0
    addu        s3, t0, s3
    madd        $ac1, s3, t7
    extr_r.w    t2, $ac1, 16
    addiu       t8, t8, 1
    addiu       s2, s2, 2
    addiu       t9, t9, 2
    addiu       s0, s0, 2
    sb          t2, -1(t8)
    bne         s5, t8, 4b
    addiu       s1, s1, 2
    addiu       s5, s7, -2
    subu        s5, s5, s4
    addu        s5, s5, t8      /* end address */
5:
    lh          v0, 0(s2)
    lh          v1, 0(t9)
    lh          t0, 0(s0)
    lh          t1, 0(s1)
    ins         v0, v1, 16, 16
    ins         t0, t1, 16, 16
    raddu.w.qb  t2, v0
    raddu.w.qb  s3, t0
    lbu         v0, -1(s2)
    lbu         v1, 2(s2)
    lbu         t0, -1(t9)
    lbu         t1, 2(t9)
    addu        v0, v0, v1
    mult        $ac1, t2, t6
    addu        t0, t0, t1
    lbu         t2, 2(s0)
    addu        t0, t0, v0
    lbu         t3, 2(s1)
    addu        s3, t0, s3
    lbu         v0, -1(s0)
    lbu         t0, -1(s1)
    sll         s3, s3, 1
    addu        v0, v0, t2
    addu        t0, t0, t3
    lh          v1, 2(t9)
    addu        t0, t0, v0
    lh          v0, 2(s2)
    addu        s3, t0, s3
    lh          t0, 2(s0)
    lh          t1, 2(s1)
    madd        $ac1, s3, t7
    extr_r.w    t2, $ac1, 16
    ins         t0, t1, 16, 16
    ins         v0, v1, 16, 16
    raddu.w.qb  s3, t0
    lbu         v1, 4(s2)
    lbu         t0, 1(t9)
    lbu         t1, 4(t9)
    sb          t2, 0(t8)
    raddu.w.qb  t3, v0
    lbu         v0, 1(s2)
    addu        t0, t0, t1
    mult        $ac1, t3, t6
    addu        v0, v0, v1
    lbu         t2, 4(s0)
    addu        t0, t0, v0
    lbu         v0, 1(s0)
    addu        s3, t0, s3
    lbu         t0, 1(s1)
    lbu         t3, 4(s1)
    addu        v0, v0, t2
    sll         s3, s3, 1
    addu        t0, t0, t3
    lh          v1, 4(t9)
    addu        t0, t0, v0
    lh          v0, 4(s2)
    addu        s3, t0, s3
    lh          t0, 4(s0)
    lh          t1, 4(s1)
    madd        $ac1, s3, t7
    extr_r.w    t2, $ac1, 16
    ins         t0, t1, 16, 16
    ins         v0, v1, 16, 16
    raddu.w.qb  s3, t0
    lbu         v1, 6(s2)
    lbu         t0, 3(t9)
    lbu         t1, 6(t9)
    sb          t2, 1(t8)
    raddu.w.qb  t3, v0
    lbu         v0, 3(s2)
    addu        t0, t0, t1
    mult        $ac1, t3, t6
    addu        v0, v0, v1
    lbu         t2, 6(s0)
    addu        t0, t0, v0
    lbu         v0, 3(s0)
    addu        s3, t0, s3
    lbu         t0, 3(s1)
    lbu         t3, 6(s1)
    addu        v0, v0, t2
    sll         s3, s3, 1
    addu        t0, t0, t3
    lh          v1, 6(t9)
    addu        t0, t0, v0
    lh          v0, 6(s2)
    addu        s3, t0, s3
    lh          t0, 6(s0)
    lh          t1, 6(s1)
    madd        $ac1, s3, t7
    extr_r.w    t3, $ac1, 16
    ins         t0, t1, 16, 16
    ins         v0, v1, 16, 16
    raddu.w.qb  s3, t0
    lbu         v1, 8(s2)
    lbu         t0, 5(t9)
    lbu         t1, 8(t9)
    sb          t3, 2(t8)
    raddu.w.qb  t2, v0
    lbu         v0, 5(s2)
    addu        t0, t0, t1
    mult        $ac1, t2, t6
    addu        v0, v0, v1
    lbu         t2, 8(s0)
    addu        t0, t0, v0
    lbu         v0, 5(s0)
    addu        s3, t0, s3
    lbu         t0, 5(s1)
    lbu         t3, 8(s1)
    addu        v0, v0, t2
    sll         s3, s3, 1
    addu        t0, t0, t3
    addiu       t8, t8, 4
    addu        t0, t0, v0
    addiu       s2, s2, 8
    addu        s3, t0, s3
    addiu       t9, t9, 8
    madd        $ac1, s3, t7
    extr_r.w    t1, $ac1, 16
    addiu       s0, s0, 8
    addiu       s1, s1, 8
    bne         s5, t8, 5b
    sb          t1, -1(t8)
/* Special case for last column */
    lh          v0, 0(s2)
    lh          v1, 0(t9)
    lh          t0, 0(s0)
    lh          t1, 0(s1)
    ins         v0, v1, 16, 16
    ins         t0, t1, 16, 16
    raddu.w.qb  t2, v0
    raddu.w.qb  s3, t0
    lbu         v0, -1(s2)
    lbu         v1, 1(s2)
    lbu         t0, -1(t9)
    lbu         t1, 1(t9)
    addu        v0, v0, v1
    mult        $ac1, t2, t6
    addu        t0, t0, t1
    lbu         t2, 1(s0)
    addu        t0, t0, v0
    lbu         t3, 1(s1)
    addu        s3, t0, s3
    lbu         v0, -1(s0)
    lbu         t0, -1(s1)
    sll         s3, s3, 1
    addu        v0, v0, t2
    addu        t0, t0, t3
    addu        t0, t0, v0
    addu        s3, t0, s3
    madd        $ac1, s3, t7
    extr_r.w    t0, $ac1, 16
    addiu       t5, t5, 2
    sb          t0, 0(t8)
    addiu       t4, t4, 1
    bne         t4, a2, 3b
    addiu       t5, t5, 2

    RESTORE_REGS_FROM_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    j           ra
     nop

END(jsimd_h2v2_smooth_downsample_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_int_upsample_dspr2)
/*
 * a0     = upsample->h_expand[compptr->component_index]
 * a1     = upsample->v_expand[compptr->component_index]
 * a2     = input_data
 * a3     = output_data_ptr
 * 16(sp) = cinfo->output_width
 * 20(sp) = cinfo->max_v_samp_factor
 */
    .set at

    SAVE_REGS_ON_STACK 16, s0, s1, s2, s3

    lw          s0, 0(a3)       /* s0 = output_data */
    lw          s1, 32(sp)      /* s1 = cinfo->output_width */
    lw          s2, 36(sp)      /* s2 = cinfo->max_v_samp_factor */
    li          t6, 0           /* t6 = inrow */
    beqz        s2, 10f
     li         s3, 0           /* s3 = outrow */
0:
    addu        t0, a2, t6
    addu        t7, s0, s3
    lw          t3, 0(t0)       /* t3 = inptr */
    lw          t8, 0(t7)       /* t8 = outptr */
    beqz        s1, 4f
     addu       t5, t8, s1      /* t5 = outend */
1:
    lb          t2, 0(t3)       /* t2 = invalue = *inptr++ */
    addiu       t3, 1
    beqz        a0, 3f
     move       t0, a0          /* t0 = h_expand */
2:
    sb          t2, 0(t8)
    addiu       t0, -1
    bgtz        t0, 2b
     addiu      t8, 1
3:
    bgt         t5, t8, 1b
     nop
4:
    addiu       t9, a1, -1      /* t9 = v_expand - 1 */
    blez        t9, 9f
     nop
5:
    lw          t3, 0(s0)
    lw          t4, 4(s0)
    subu        t0, s1, 0xF
    blez        t0, 7f
     addu       t5, t3, s1      /* t5 = end address */
    andi        t7, s1, 0xF     /* t7 = residual */
    subu        t8, t5, t7
6:
    ulw         t0, 0(t3)
    ulw         t1, 4(t3)
    ulw         t2, 8(t3)
    usw         t0, 0(t4)
    ulw         t0, 12(t3)
    usw         t1, 4(t4)
    usw         t2, 8(t4)
    usw         t0, 12(t4)
    addiu       t3, 16
    bne         t3, t8, 6b
     addiu      t4, 16
    beqz        t7, 8f
     nop
7:
    lbu         t0, 0(t3)
    sb          t0, 0(t4)
    addiu       t3, 1
    bne         t3, t5, 7b
     addiu      t4, 1
8:
    addiu       t9, -1
    bgtz        t9, 5b
     addiu      s0, 8
9:
    addu        s3, s3, a1
    bne         s3, s2, 0b
     addiu      t6, 1
10:
    RESTORE_REGS_FROM_STACK 16, s0, s1, s2, s3

    j           ra
     nop
END(jsimd_int_upsample_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_h2v1_upsample_dspr2)
/*
 * a0 = cinfo->max_v_samp_factor
 * a1 = cinfo->output_width
 * a2 = input_data
 * a3 = output_data_ptr
 */
    lw          t7, 0(a3)       /* t7 = output_data */
    andi        t8, a1, 0xf     /* t8 = residual */
    sll         t0, a0, 2
    blez        a0, 4f
     addu       t9, t7, t0      /* t9 = output_data end address */
0:
    lw          t5, 0(t7)       /* t5 = outptr */
    lw          t6, 0(a2)       /* t6 = inptr */
    addu        t3, t5, a1      /* t3 = outptr + output_width (end address) */
    subu        t3, t8          /* t3 = end address - residual */
    beq         t5, t3, 2f
     move       t4, t8
1:
    ulw         t0, 0(t6)       /* t0 = |P3|P2|P1|P0| */
    ulw         t2, 4(t6)       /* t2 = |P7|P6|P5|P4| */
    srl         t1, t0, 16      /* t1 = |X|X|P3|P2| */
    ins         t0, t0, 16, 16  /* t0 = |P1|P0|P1|P0| */
    ins         t1, t1, 16, 16  /* t1 = |P3|P2|P3|P2| */
    ins         t0, t0, 8, 16   /* t0 = |P1|P1|P0|P0| */
    ins         t1, t1, 8, 16   /* t1 = |P3|P3|P2|P2| */
    usw         t0, 0(t5)
    usw         t1, 4(t5)
    srl         t0, t2, 16      /* t0 = |X|X|P7|P6| */
    ins         t2, t2, 16, 16  /* t2 = |P5|P4|P5|P4| */
    ins         t0, t0, 16, 16  /* t0 = |P7|P6|P7|P6| */
    ins         t2, t2, 8, 16   /* t2 = |P5|P5|P4|P4| */
    ins         t0, t0, 8, 16   /* t0 = |P7|P7|P6|P6| */
    usw         t2, 8(t5)
    usw         t0, 12(t5)
    addiu       t5, 16
    bne         t5, t3, 1b
     addiu      t6, 8
    beqz        t8, 3f
     move       t4, t8
2:
    lbu         t1, 0(t6)
    sb          t1, 0(t5)
    sb          t1, 1(t5)
    addiu       t4, -2
    addiu       t6, 1
    bgtz        t4, 2b
     addiu      t5, 2
3:
    addiu       t7, 4
    bne         t9, t7, 0b
     addiu      a2, 4
4:
    j           ra
     nop
END(jsimd_h2v1_upsample_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_h2v2_upsample_dspr2)
/*
 * a0 = cinfo->max_v_samp_factor
 * a1 = cinfo->output_width
 * a2 = input_data
 * a3 = output_data_ptr
 */
    lw          t7, 0(a3)
    blez        a0, 7f
     andi       t9, a1, 0xf     /* t9 = residual */
0:
    lw          t6, 0(a2)       /* t6 = inptr */
    lw          t5, 0(t7)       /* t5 = outptr */
    addu        t8, t5, a1      /* t8 = outptr end address */
    subu        t8, t9          /* t8 = end address - residual */
    beq         t5, t8, 2f
     move       t4, t9
1:
    ulw         t0, 0(t6)
    srl         t1, t0, 16
    ins         t0, t0, 16, 16
    ins         t0, t0, 8, 16
    ins         t1, t1, 16, 16
    ins         t1, t1, 8, 16
    ulw         t2, 4(t6)
    usw         t0, 0(t5)
    usw         t1, 4(t5)
    srl         t3, t2, 16
    ins         t2, t2, 16, 16
    ins         t2, t2, 8, 16
    ins         t3, t3, 16, 16
    ins         t3, t3, 8, 16
    usw         t2, 8(t5)
    usw         t3, 12(t5)
    addiu       t5, 16
    bne         t5, t8, 1b
     addiu      t6, 8
    beqz        t9, 3f
     move       t4, t9
2:
    lbu         t0, 0(t6)
    sb          t0, 0(t5)
    sb          t0, 1(t5)
    addiu       t4, -2
    addiu       t6, 1
    bgtz        t4, 2b
     addiu      t5, 2
3:
    lw          t6, 0(t7)       /* t6 = outptr[0] */
    lw          t5, 4(t7)       /* t5 = outptr[1] */
    addu        t4, t6, a1      /* t4 = new end address */
    beq         a1, t9, 5f
     subu       t8, t4, t9
4:
    ulw         t0, 0(t6)
    ulw         t1, 4(t6)
    ulw         t2, 8(t6)
    usw         t0, 0(t5)
    ulw         t0, 12(t6)
    usw         t1, 4(t5)
    usw         t2, 8(t5)
    usw         t0, 12(t5)
    addiu       t6, 16
    bne         t6, t8, 4b
     addiu      t5, 16
    beqz        t9, 6f
     nop
5:
    lbu         t0, 0(t6)
    sb          t0, 0(t5)
    addiu       t6, 1
    bne         t6, t4, 5b
     addiu      t5, 1
6:
    addiu       t7, 8
    addiu       a0, -2
    bgtz        a0, 0b
     addiu      a2, 4
7:
    j           ra
     nop
END(jsimd_h2v2_upsample_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_idct_islow_dspr2)
/*
 * a0 = coef_block
 * a1 = compptr->dcttable
 * a2 = output
 * a3 = range_limit
 */
    SAVE_REGS_ON_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    addiu       sp, sp, -256
    move        v0, sp
    addiu       v1, zero, 8     /* v1 = DCTSIZE = 8 */
1:
    lh          s4, 32(a0)      /* s4 = inptr[16] */
    lh          s5, 64(a0)      /* s5 = inptr[32] */
    lh          s6, 96(a0)      /* s6 = inptr[48] */
    lh          t1, 112(a0)     /* t1 = inptr[56] */
    lh          t7, 16(a0)      /* t7 = inptr[8] */
    lh          t5, 80(a0)      /* t5 = inptr[40] */
    lh          t3, 48(a0)      /* t3 = inptr[24] */
    or          s4, s4, t1
    or          s4, s4, t3
    or          s4, s4, t5
    or          s4, s4, t7
    or          s4, s4, s5
    or          s4, s4, s6
    bnez        s4, 2f
     addiu      v1, v1, -1
    lh          s5, 0(a1)       /* quantptr[DCTSIZE*0] */
    lh          s6, 0(a0)       /* inptr[DCTSIZE*0] */
    mul         s5, s5, s6      /* DEQUANTIZE(inptr[0], quantptr[0]) */
    sll         s5, s5, 2
    sw          s5, 0(v0)
    sw          s5, 32(v0)
    sw          s5, 64(v0)
    sw          s5, 96(v0)
    sw          s5, 128(v0)
    sw          s5, 160(v0)
    sw          s5, 192(v0)
    b           3f
     sw         s5, 224(v0)
2:
    lh          t0, 112(a1)
    lh          t2, 48(a1)
    lh          t4, 80(a1)
    lh          t6, 16(a1)
    mul         t0, t0, t1      /* DEQUANTIZE(inptr[DCTSIZE*7],
                                              quantptr[DCTSIZE*7]) */
    mul         t1, t2, t3      /* DEQUANTIZE(inptr[DCTSIZE*3],
                                              quantptr[DCTSIZE*3]) */
    mul         t2, t4, t5      /* DEQUANTIZE(inptr[DCTSIZE*5],
                                              quantptr[DCTSIZE*5]) */
    mul         t3, t6, t7      /* DEQUANTIZE(inptr[DCTSIZE*1],
                                              quantptr[DCTSIZE*1]) */
    lh          t4, 32(a1)
    lh          t5, 32(a0)
    lh          t6, 96(a1)
    lh          t7, 96(a0)
    addu        s0, t0, t1       /* z3 = tmp0 + tmp2 */
    addu        s1, t1, t2       /* z2 = tmp1 + tmp2 */
    addu        s2, t2, t3       /* z4 = tmp1 + tmp3 */
    addu        s3, s0, s2       /* z3 + z4 */
    addiu       t9, zero, 9633   /* FIX_1_175875602 */
    mul         s3, s3, t9       /* z5 = MULTIPLY(z3 + z4, FIX_1_175875602) */
    addu        t8, t0, t3       /* z1 = tmp0 + tmp3 */
    addiu       t9, zero, 2446   /* FIX_0_298631336 */
    mul         t0, t0, t9       /* tmp0 = MULTIPLY(tmp0, FIX_0_298631336) */
    addiu       t9, zero, 16819  /* FIX_2_053119869 */
    mul         t2, t2, t9       /* tmp1 = MULTIPLY(tmp1, FIX_2_053119869) */
    addiu       t9, zero, 25172  /* FIX_3_072711026 */
    mul         t1, t1, t9       /* tmp2 = MULTIPLY(tmp2, FIX_3_072711026) */
    addiu       t9, zero, 12299  /* FIX_1_501321110 */
    mul         t3, t3, t9       /* tmp3 = MULTIPLY(tmp3, FIX_1_501321110) */
    addiu       t9, zero, 16069  /* FIX_1_961570560 */
    mul         s0, s0, t9       /* -z3 = MULTIPLY(z3, FIX_1_961570560) */
    addiu       t9, zero, 3196   /* FIX_0_390180644 */
    mul         s2, s2, t9       /* -z4 = MULTIPLY(z4, FIX_0_390180644) */
    addiu       t9, zero, 7373   /* FIX_0_899976223 */
    mul         t8, t8, t9       /* -z1 = MULTIPLY(z1, FIX_0_899976223) */
    addiu       t9, zero, 20995  /* FIX_2_562915447 */
    mul         s1, s1, t9       /* -z2 = MULTIPLY(z2, FIX_2_562915447) */
    subu        s0, s3, s0       /* z3 += z5 */
    addu        t0, t0, s0       /* tmp0 += z3 */
    addu        t1, t1, s0       /* tmp2 += z3 */
    subu        s2, s3, s2       /* z4 += z5 */
    addu        t2, t2, s2       /* tmp1 += z4 */
    addu        t3, t3, s2       /* tmp3 += z4 */
    subu        t0, t0, t8       /* tmp0 += z1 */
    subu        t1, t1, s1       /* tmp2 += z2 */
    subu        t2, t2, s1       /* tmp1 += z2 */
    subu        t3, t3, t8       /* tmp3 += z1 */
    mul         s0, t4, t5       /* DEQUANTIZE(inptr[DCTSIZE*2],
                                               quantptr[DCTSIZE*2]) */
    addiu       t9, zero, 6270   /* FIX_0_765366865 */
    mul         s1, t6, t7       /* DEQUANTIZE(inptr[DCTSIZE*6],
                                               quantptr[DCTSIZE*6]) */
    lh          t4, 0(a1)
    lh          t5, 0(a0)
    lh          t6, 64(a1)
    lh          t7, 64(a0)
    mul         s2, t9, s0       /* MULTIPLY(z2, FIX_0_765366865) */
    mul         t5, t4, t5       /* DEQUANTIZE(inptr[DCTSIZE*0],
                                               quantptr[DCTSIZE*0]) */
    mul         t6, t6, t7       /* DEQUANTIZE(inptr[DCTSIZE*4],
                                               quantptr[DCTSIZE*4]) */
    addiu       t9, zero, 4433   /* FIX_0_541196100 */
    addu        s3, s0, s1       /* z2 + z3 */
    mul         s3, s3, t9       /* z1 = MULTIPLY(z2 + z3, FIX_0_541196100) */
    addiu       t9, zero, 15137  /* FIX_1_847759065 */
    mul         t8, s1, t9       /* MULTIPLY(z3, FIX_1_847759065) */
    addu        t4, t5, t6
    subu        t5, t5, t6
    sll         t4, t4, 13      /* tmp0 = (z2 + z3) << CONST_BITS */
    sll         t5, t5, 13      /* tmp1 = (z2 - z3) << CONST_BITS */
    addu        t7, s3, s2      /* tmp3 = z1 + MULTIPLY(z2, FIX_0_765366865) */
    subu        t6, s3, t8      /* tmp2 =
                                     z1 + MULTIPLY(z3, -FIX_1_847759065) */
    addu        s0, t4, t7
    subu        s1, t4, t7
    addu        s2, t5, t6
    subu        s3, t5, t6
    addu        t4, s0, t3
    subu        s0, s0, t3
    addu        t3, s2, t1
    subu        s2, s2, t1
    addu        t1, s3, t2
    subu        s3, s3, t2
    addu        t2, s1, t0
    subu        s1, s1, t0
    shra_r.w    t4, t4, 11
    shra_r.w    t3, t3, 11
    shra_r.w    t1, t1, 11
    shra_r.w    t2, t2, 11
    shra_r.w    s1, s1, 11
    shra_r.w    s3, s3, 11
    shra_r.w    s2, s2, 11
    shra_r.w    s0, s0, 11
    sw          t4, 0(v0)
    sw          t3, 32(v0)
    sw          t1, 64(v0)
    sw          t2, 96(v0)
    sw          s1, 128(v0)
    sw          s3, 160(v0)
    sw          s2, 192(v0)
    sw          s0, 224(v0)
3:
    addiu       a1, a1, 2
    addiu       a0, a0, 2
    bgtz        v1, 1b
     addiu      v0, v0, 4
    move        v0, sp
    addiu       v1, zero, 8
4:
    lw          t0, 8(v0)       /* z2 = (JLONG)wsptr[2] */
    lw          t1, 24(v0)      /* z3 = (JLONG)wsptr[6] */
    lw          t2, 0(v0)       /* (JLONG)wsptr[0] */
    lw          t3, 16(v0)      /* (JLONG)wsptr[4] */
    lw          s4, 4(v0)       /* (JLONG)wsptr[1] */
    lw          s5, 12(v0)      /* (JLONG)wsptr[3] */
    lw          s6, 20(v0)      /* (JLONG)wsptr[5] */
    lw          s7, 28(v0)      /* (JLONG)wsptr[7] */
    or          s4, s4, t0
    or          s4, s4, t1
    or          s4, s4, t3
    or          s4, s4, s7
    or          s4, s4, s5
    or          s4, s4, s6
    bnez        s4, 5f
     addiu      v1, v1, -1
    shra_r.w    s5, t2, 5
    andi        s5, s5, 0x3ff
    lbux        s5, s5(a3)
    lw          s1, 0(a2)
    replv.qb    s5, s5
    usw         s5, 0(s1)
    usw         s5, 4(s1)
    b           6f
     nop
5:
    addu        t4, t0, t1       /* z2 + z3 */
    addiu       t8, zero, 4433   /* FIX_0_541196100 */
    mul         t5, t4, t8       /* z1 = MULTIPLY(z2 + z3, FIX_0_541196100) */
    addiu       t8, zero, 15137  /* FIX_1_847759065 */
    mul         t1, t1, t8       /* MULTIPLY(z3, FIX_1_847759065) */
    addiu       t8, zero, 6270   /* FIX_0_765366865 */
    mul         t0, t0, t8       /* MULTIPLY(z2, FIX_0_765366865) */
    addu        t4, t2, t3       /* (JLONG)wsptr[0] + (JLONG)wsptr[4] */
    subu        t2, t2, t3       /* (JLONG)wsptr[0] - (JLONG)wsptr[4] */
    sll         t4, t4, 13       /* tmp0 =
                                      (wsptr[0] + wsptr[4]) << CONST_BITS */
    sll         t2, t2, 13       /* tmp1 =
                                      (wsptr[0] - wsptr[4]) << CONST_BITS */
    subu        t1, t5, t1       /* tmp2 =
                                      z1 + MULTIPLY(z3, -FIX_1_847759065) */
    subu        t3, t2, t1       /* tmp12 = tmp1 - tmp2 */
    addu        t2, t2, t1       /* tmp11 = tmp1 + tmp2 */
    addu        t5, t5, t0       /* tmp3 =
                                      z1 + MULTIPLY(z2, FIX_0_765366865) */
    subu        t1, t4, t5       /* tmp13 = tmp0 - tmp3 */
    addu        t0, t4, t5       /* tmp10 = tmp0 + tmp3 */
    lw          t4, 28(v0)       /* tmp0 = (JLONG)wsptr[7] */
    lw          t6, 12(v0)       /* tmp2 = (JLONG)wsptr[3] */
    lw          t5, 20(v0)       /* tmp1 = (JLONG)wsptr[5] */
    lw          t7, 4(v0)        /* tmp3 = (JLONG)wsptr[1] */
    addu        s0, t4, t6       /* z3 = tmp0 + tmp2 */
    addiu       t8, zero, 9633   /* FIX_1_175875602 */
    addu        s1, t5, t7       /* z4 = tmp1 + tmp3 */
    addu        s2, s0, s1       /* z3 + z4 */
    mul         s2, s2, t8       /* z5 = MULTIPLY(z3 + z4, FIX_1_175875602) */
    addu        s3, t4, t7       /* z1 = tmp0 + tmp3 */
    addu        t9, t5, t6       /* z2 = tmp1 + tmp2 */
    addiu       t8, zero, 16069  /* FIX_1_961570560 */
    mul         s0, s0, t8       /* -z3 = MULTIPLY(z3, FIX_1_961570560) */
    addiu       t8, zero, 3196   /* FIX_0_390180644 */
    mul         s1, s1, t8       /* -z4 = MULTIPLY(z4, FIX_0_390180644) */
    addiu       t8, zero, 2446   /* FIX_0_298631336 */
    mul         t4, t4, t8       /* tmp0 = MULTIPLY(tmp0, FIX_0_298631336) */
    addiu       t8, zero, 7373   /* FIX_0_899976223 */
    mul         s3, s3, t8       /* -z1 = MULTIPLY(z1, FIX_0_899976223) */
    addiu       t8, zero, 16819  /* FIX_2_053119869 */
    mul         t5, t5, t8       /* tmp1 = MULTIPLY(tmp1, FIX_2_053119869) */
    addiu       t8, zero, 20995  /* FIX_2_562915447 */
    mul         t9, t9, t8       /* -z2 = MULTIPLY(z2, FIX_2_562915447) */
    addiu       t8, zero, 25172  /* FIX_3_072711026 */
    mul         t6, t6, t8       /* tmp2 = MULTIPLY(tmp2, FIX_3_072711026) */
    addiu       t8, zero, 12299  /* FIX_1_501321110 */
    mul         t7, t7, t8       /* tmp3 = MULTIPLY(tmp3, FIX_1_501321110) */
    subu        s0, s2, s0       /* z3 += z5 */
    subu        s1, s2, s1       /* z4 += z5 */
    addu        t4, t4, s0
    subu        t4, t4, s3      /* tmp0 */
    addu        t5, t5, s1
    subu        t5, t5, t9      /* tmp1 */
    addu        t6, t6, s0
    subu        t6, t6, t9      /* tmp2 */
    addu        t7, t7, s1
    subu        t7, t7, s3      /* tmp3 */
    addu        s0, t0, t7
    subu        t0, t0, t7
    addu        t7, t2, t6
    subu        t2, t2, t6
    addu        t6, t3, t5
    subu        t3, t3, t5
    addu        t5, t1, t4
    subu        t1, t1, t4
    shra_r.w    s0, s0, 18
    shra_r.w    t7, t7, 18
    shra_r.w    t6, t6, 18
    shra_r.w    t5, t5, 18
    shra_r.w    t1, t1, 18
    shra_r.w    t3, t3, 18
    shra_r.w    t2, t2, 18
    shra_r.w    t0, t0, 18
    andi        s0, s0, 0x3ff
    andi        t7, t7, 0x3ff
    andi        t6, t6, 0x3ff
    andi        t5, t5, 0x3ff
    andi        t1, t1, 0x3ff
    andi        t3, t3, 0x3ff
    andi        t2, t2, 0x3ff
    andi        t0, t0, 0x3ff
    lw          s1, 0(a2)
    lbux        s0, s0(a3)
    lbux        t7, t7(a3)
    lbux        t6, t6(a3)
    lbux        t5, t5(a3)
    lbux        t1, t1(a3)
    lbux        t3, t3(a3)
    lbux        t2, t2(a3)
    lbux        t0, t0(a3)
    sb          s0, 0(s1)
    sb          t7, 1(s1)
    sb          t6, 2(s1)
    sb          t5, 3(s1)
    sb          t1, 4(s1)
    sb          t3, 5(s1)
    sb          t2, 6(s1)
    sb          t0, 7(s1)
6:
    addiu       v0, v0, 32
    bgtz        v1, 4b
     addiu      a2, a2, 4
    addiu       sp, sp, 256

    RESTORE_REGS_FROM_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    j           ra
     nop

END(jsimd_idct_islow_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_idct_ifast_cols_dspr2)
/*
 * a0 = inptr
 * a1 = quantptr
 * a2 = wsptr
 * a3 = mips_idct_ifast_coefs
 */
    SAVE_REGS_ON_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    addiu         t9, a0, 16      /* end address */
    or            AT, a3, zero

0:
    lw            s0, 0(a1)       /* quantptr[DCTSIZE*0] */
    lw            t0, 0(a0)       /* inptr[DCTSIZE*0] */
    lw            t1, 16(a0)      /* inptr[DCTSIZE*1] */
    muleq_s.w.phl v0, t0, s0      /* tmp0 ... */
    lw            t2, 32(a0)      /* inptr[DCTSIZE*2] */
    lw            t3, 48(a0)      /* inptr[DCTSIZE*3] */
    lw            t4, 64(a0)      /* inptr[DCTSIZE*4] */
    lw            t5, 80(a0)      /* inptr[DCTSIZE*5] */
    muleq_s.w.phr t0, t0, s0      /* ... tmp0 ... */
    lw            t6, 96(a0)      /* inptr[DCTSIZE*6] */
    lw            t7, 112(a0)     /* inptr[DCTSIZE*7] */
    or            s4, t1, t2
    or            s5, t3, t4
    bnez          s4, 1f
     ins          t0, v0, 16, 16  /* ... tmp0 */
    bnez          s5, 1f
     or           s6, t5, t6
    or            s6, s6, t7
    bnez          s6, 1f
     sw           t0, 0(a2)       /* wsptr[DCTSIZE*0] */
    sw            t0, 16(a2)      /* wsptr[DCTSIZE*1] */
    sw            t0, 32(a2)      /* wsptr[DCTSIZE*2] */
    sw            t0, 48(a2)      /* wsptr[DCTSIZE*3] */
    sw            t0, 64(a2)      /* wsptr[DCTSIZE*4] */
    sw            t0, 80(a2)      /* wsptr[DCTSIZE*5] */
    sw            t0, 96(a2)      /* wsptr[DCTSIZE*6] */
    sw            t0, 112(a2)     /* wsptr[DCTSIZE*7] */
    addiu         a0, a0, 4
    b             2f
     addiu        a1, a1, 4

1:
    lw            s1, 32(a1)      /* quantptr[DCTSIZE*2] */
    lw            s2, 64(a1)      /* quantptr[DCTSIZE*4] */
    muleq_s.w.phl v0, t2, s1      /* tmp1 ... */
    muleq_s.w.phr t2, t2, s1      /* ... tmp1 ... */
    lw            s0, 16(a1)      /* quantptr[DCTSIZE*1] */
    lw            s1, 48(a1)      /* quantptr[DCTSIZE*3] */
    lw            s3, 96(a1)      /* quantptr[DCTSIZE*6] */
    muleq_s.w.phl v1, t4, s2      /* tmp2 ... */
    muleq_s.w.phr t4, t4, s2      /* ... tmp2 ... */
    lw            s2, 80(a1)      /* quantptr[DCTSIZE*5] */
    lw            t8, 4(AT)       /* FIX(1.414213562) */
    ins           t2, v0, 16, 16  /* ... tmp1 */
    muleq_s.w.phl v0, t6, s3      /* tmp3 ... */
    muleq_s.w.phr t6, t6, s3      /* ... tmp3 ... */
    ins           t4, v1, 16, 16  /* ... tmp2 */
    addq.ph       s4, t0, t4      /* tmp10 */
    subq.ph       s5, t0, t4      /* tmp11 */
    ins           t6, v0, 16, 16  /* ... tmp3 */
    subq.ph       s6, t2, t6      /* tmp12 ... */
    addq.ph       s7, t2, t6      /* tmp13 */
    mulq_s.ph     s6, s6, t8      /* ... tmp12 ... */
    addq.ph       t0, s4, s7      /* tmp0 */
    subq.ph       t6, s4, s7      /* tmp3 */
    muleq_s.w.phl v0, t1, s0      /* tmp4 ... */
    muleq_s.w.phr t1, t1, s0      /* ... tmp4 ... */
    shll_s.ph     s6, s6, 1       /* x2 */
    lw            s3, 112(a1)     /* quantptr[DCTSIZE*7] */
    subq.ph       s6, s6, s7      /* ... tmp12 */
    muleq_s.w.phl v1, t7, s3      /* tmp7 ... */
    muleq_s.w.phr t7, t7, s3      /* ... tmp7 ... */
    ins           t1, v0, 16, 16  /* ... tmp4 */
    addq.ph       t2, s5, s6      /* tmp1 */
    subq.ph       t4, s5, s6      /* tmp2 */
    muleq_s.w.phl v0, t5, s2      /* tmp6 ... */
    muleq_s.w.phr t5, t5, s2      /* ... tmp6 ... */
    ins           t7, v1, 16, 16  /* ... tmp7 */
    addq.ph       s5, t1, t7      /* z11 */
    subq.ph       s6, t1, t7      /* z12 */
    muleq_s.w.phl v1, t3, s1      /* tmp5 ... */
    muleq_s.w.phr t3, t3, s1      /* ... tmp5 ... */
    ins           t5, v0, 16, 16  /* ... tmp6 */
    ins           t3, v1, 16, 16  /* ... tmp5 */
    addq.ph       s7, t5, t3      /* z13 */
    subq.ph       v0, t5, t3      /* z10 */
    addq.ph       t7, s5, s7      /* tmp7 */
    subq.ph       s5, s5, s7      /* tmp11 ... */
    addq.ph       v1, v0, s6      /* z5 ... */
    mulq_s.ph     s5, s5, t8      /* ... tmp11 */
    lw            t8, 8(AT)       /* FIX(1.847759065) */
    lw            s4, 0(AT)       /* FIX(1.082392200) */
    addq.ph       s0, t0, t7
    subq.ph       s1, t0, t7
    mulq_s.ph     v1, v1, t8      /* ... z5 */
    shll_s.ph     s5, s5, 1       /* x2 */
    lw            t8, 12(AT)      /* FIX(-2.613125930) */
    sw            s0, 0(a2)       /* wsptr[DCTSIZE*0] */
    shll_s.ph     v0, v0, 1       /* x4 */
    mulq_s.ph     v0, v0, t8      /* tmp12 ... */
    mulq_s.ph     s4, s6, s4      /* tmp10 ... */
    shll_s.ph     v1, v1, 1       /* x2 */
    addiu         a0, a0, 4
    addiu         a1, a1, 4
    sw            s1, 112(a2)     /* wsptr[DCTSIZE*7] */
    shll_s.ph     s6, v0, 1       /* x4 */
    shll_s.ph     s4, s4, 1       /* x2 */
    addq.ph       s6, s6, v1      /* ... tmp12 */
    subq.ph       t5, s6, t7      /* tmp6 */
    subq.ph       s4, s4, v1      /* ... tmp10 */
    subq.ph       t3, s5, t5      /* tmp5 */
    addq.ph       s2, t2, t5
    addq.ph       t1, s4, t3      /* tmp4 */
    subq.ph       s3, t2, t5
    sw            s2, 16(a2)      /* wsptr[DCTSIZE*1] */
    sw            s3, 96(a2)      /* wsptr[DCTSIZE*6] */
    addq.ph       v0, t4, t3
    subq.ph       v1, t4, t3
    sw            v0, 32(a2)      /* wsptr[DCTSIZE*2] */
    sw            v1, 80(a2)      /* wsptr[DCTSIZE*5] */
    addq.ph       v0, t6, t1
    subq.ph       v1, t6, t1
    sw            v0, 64(a2)      /* wsptr[DCTSIZE*4] */
    sw            v1, 48(a2)      /* wsptr[DCTSIZE*3] */

2:
    bne           a0, t9, 0b
     addiu        a2, a2, 4

    RESTORE_REGS_FROM_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    j             ra
     nop

END(jsimd_idct_ifast_cols_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_idct_ifast_rows_dspr2)
/*
 * a0 = wsptr
 * a1 = output_buf
 * a2 = output_col
 * a3 = mips_idct_ifast_coefs
 */
    SAVE_REGS_ON_STACK 40, s0, s1, s2, s3, s4, s5, s6, s7, s8, a3

    addiu         t9, a0, 128     /* end address */
    lui           s8, 0x8080
    ori           s8, s8, 0x8080

0:
    lw            AT, 36(sp)      /* restore $a3 (mips_idct_ifast_coefs) */
    lw            t0, 0(a0)       /* wsptr[DCTSIZE*0+0/1]  b a */
    lw            s0, 16(a0)      /* wsptr[DCTSIZE*1+0/1]  B A */
    lw            t2, 4(a0)       /* wsptr[DCTSIZE*0+2/3]  d c */
    lw            s2, 20(a0)      /* wsptr[DCTSIZE*1+2/3]  D C */
    lw            t4, 8(a0)       /* wsptr[DCTSIZE*0+4/5]  f e */
    lw            s4, 24(a0)      /* wsptr[DCTSIZE*1+4/5]  F E */
    lw            t6, 12(a0)      /* wsptr[DCTSIZE*0+6/7]  h g */
    lw            s6, 28(a0)      /* wsptr[DCTSIZE*1+6/7]  H G */
    precrq.ph.w   t1, s0, t0      /* B b */
    ins           t0, s0, 16, 16  /* A a */
    bnez          t1, 1f
     or           s0, t2, s2
    bnez          s0, 1f
     or           s0, t4, s4
    bnez          s0, 1f
     or           s0, t6, s6
    bnez          s0, 1f
     shll_s.ph    s0, t0, 2       /* A a */
    lw            a3, 0(a1)
    lw            AT, 4(a1)
    precrq.ph.w   t0, s0, s0      /* A A */
    ins           s0, s0, 16, 16  /* a a */
    addu          a3, a3, a2
    addu          AT, AT, a2
    precrq.qb.ph  t0, t0, t0      /* A A A A */
    precrq.qb.ph  s0, s0, s0      /* a a a a */
    addu.qb       s0, s0, s8
    addu.qb       t0, t0, s8
    sw            s0, 0(a3)
    sw            s0, 4(a3)
    sw            t0, 0(AT)
    sw            t0, 4(AT)
    addiu         a0, a0, 32
    bne           a0, t9, 0b
     addiu        a1, a1, 8
    b             2f
     nop

1:
    precrq.ph.w   t3, s2, t2
    ins           t2, s2, 16, 16
    precrq.ph.w   t5, s4, t4
    ins           t4, s4, 16, 16
    precrq.ph.w   t7, s6, t6
    ins           t6, s6, 16, 16
    lw            t8, 4(AT)       /* FIX(1.414213562) */
    addq.ph       s4, t0, t4      /* tmp10 */
    subq.ph       s5, t0, t4      /* tmp11 */
    subq.ph       s6, t2, t6      /* tmp12 ... */
    addq.ph       s7, t2, t6      /* tmp13 */
    mulq_s.ph     s6, s6, t8      /* ... tmp12 ... */
    addq.ph       t0, s4, s7      /* tmp0 */
    subq.ph       t6, s4, s7      /* tmp3 */
    shll_s.ph     s6, s6, 1       /* x2 */
    subq.ph       s6, s6, s7      /* ... tmp12 */
    addq.ph       t2, s5, s6      /* tmp1 */
    subq.ph       t4, s5, s6      /* tmp2 */
    addq.ph       s5, t1, t7      /* z11 */
    subq.ph       s6, t1, t7      /* z12 */
    addq.ph       s7, t5, t3      /* z13 */
    subq.ph       v0, t5, t3      /* z10 */
    addq.ph       t7, s5, s7      /* tmp7 */
    subq.ph       s5, s5, s7      /* tmp11 ... */
    addq.ph       v1, v0, s6      /* z5 ... */
    mulq_s.ph     s5, s5, t8      /* ... tmp11 */
    lw            t8, 8(AT)       /* FIX(1.847759065) */
    lw            s4, 0(AT)       /* FIX(1.082392200) */
    addq.ph       s0, t0, t7      /* tmp0 + tmp7 */
    subq.ph       s7, t0, t7      /* tmp0 - tmp7 */
    mulq_s.ph     v1, v1, t8      /* ... z5 */
    lw            a3, 0(a1)
    lw            t8, 12(AT)      /* FIX(-2.613125930) */
    shll_s.ph     s5, s5, 1       /* x2 */
    addu          a3, a3, a2
    shll_s.ph     v0, v0, 1       /* x4 */
    mulq_s.ph     v0, v0, t8      /* tmp12 ... */
    mulq_s.ph     s4, s6, s4      /* tmp10 ... */
    shll_s.ph     v1, v1, 1       /* x2 */
    addiu         a0, a0, 32
    addiu         a1, a1, 8
    shll_s.ph     s6, v0, 1       /* x4 */
    shll_s.ph     s4, s4, 1       /* x2 */
    addq.ph       s6, s6, v1      /* ... tmp12 */
    shll_s.ph     s0, s0, 2
    subq.ph       t5, s6, t7      /* tmp6 */
    subq.ph       s4, s4, v1      /* ... tmp10 */
    subq.ph       t3, s5, t5      /* tmp5 */
    shll_s.ph     s7, s7, 2
    addq.ph       t1, s4, t3      /* tmp4 */
    addq.ph       s1, t2, t5      /* tmp1 + tmp6 */
    subq.ph       s6, t2, t5      /* tmp1 - tmp6 */
    addq.ph       s2, t4, t3      /* tmp2 + tmp5 */
    subq.ph       s5, t4, t3      /* tmp2 - tmp5 */
    addq.ph       s4, t6, t1      /* tmp3 + tmp4 */
    subq.ph       s3, t6, t1      /* tmp3 - tmp4 */
    shll_s.ph     s1, s1, 2
    shll_s.ph     s2, s2, 2
    shll_s.ph     s3, s3, 2
    shll_s.ph     s4, s4, 2
    shll_s.ph     s5, s5, 2
    shll_s.ph     s6, s6, 2
    precrq.ph.w   t0, s1, s0      /* B A */
    ins           s0, s1, 16, 16  /* b a */
    precrq.ph.w   t2, s3, s2      /* D C */
    ins           s2, s3, 16, 16  /* d c */
    precrq.ph.w   t4, s5, s4      /* F E */
    ins           s4, s5, 16, 16  /* f e */
    precrq.ph.w   t6, s7, s6      /* H G */
    ins           s6, s7, 16, 16  /* h g */
    precrq.qb.ph  t0, t2, t0      /* D C B A */
    precrq.qb.ph  s0, s2, s0      /* d c b a */
    precrq.qb.ph  t4, t6, t4      /* H G F E */
    precrq.qb.ph  s4, s6, s4      /* h g f e */
    addu.qb       s0, s0, s8
    addu.qb       s4, s4, s8
    sw            s0, 0(a3)       /* outptr[0/1/2/3]       d c b a */
    sw            s4, 4(a3)       /* outptr[4/5/6/7]       h g f e */
    lw            a3, -4(a1)
    addu.qb       t0, t0, s8
    addu          a3, a3, a2
    addu.qb       t4, t4, s8
    sw            t0, 0(a3)       /* outptr[0/1/2/3]       D C B A */
    bne           a0, t9, 0b
     sw           t4, 4(a3)       /* outptr[4/5/6/7]       H G F E */

2:

    RESTORE_REGS_FROM_STACK 40, s0, s1, s2, s3, s4, s5, s6, s7, s8, a3

    j             ra
     nop

END(jsimd_idct_ifast_rows_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_fdct_islow_dspr2)
/*
 * a0 = data
 */
    SAVE_REGS_ON_STACK 40, s0, s1, s2, s3, s4, s5, s6, s7, s8

    lui         t0, 6437
    ori         t0, 2260
    lui         t1, 9633
    ori         t1, 11363
    lui         t2, 0xd39e
    ori         t2, 0xe6dc
    lui         t3, 0xf72d
    ori         t3, 9633
    lui         t4, 2261
    ori         t4, 9633
    lui         t5, 0xd39e
    ori         t5, 6437
    lui         t6, 9633
    ori         t6, 0xd39d
    lui         t7, 0xe6dc
    ori         t7, 2260
    lui         t8, 4433
    ori         t8, 10703
    lui         t9, 0xd630
    ori         t9, 4433
    li          s8, 8
    move        a1, a0
1:
    lw          s0, 0(a1)       /* tmp0 = 1|0 */
    lw          s1, 4(a1)       /* tmp1 = 3|2 */
    lw          s2, 8(a1)       /* tmp2 = 5|4 */
    lw          s3, 12(a1)      /* tmp3 = 7|6 */
    packrl.ph   s1, s1, s1      /* tmp1 = 2|3 */
    packrl.ph   s3, s3, s3      /* tmp3 = 6|7 */
    subq.ph     s7, s1, s2      /* tmp7 = 2-5|3-4 = t5|t4 */
    subq.ph     s5, s0, s3      /* tmp5 = 1-6|0-7 = t6|t7 */
    mult        $0, $0          /* ac0  = 0 */
    dpa.w.ph    $ac0, s7, t0    /* ac0 += t5*  6437 + t4*  2260 */
    dpa.w.ph    $ac0, s5, t1    /* ac0 += t6*  9633 + t7* 11363 */
    mult        $ac1, $0, $0    /* ac1  = 0 */
    dpa.w.ph    $ac1, s7, t2    /* ac1 += t5*-11362 + t4* -6436 */
    dpa.w.ph    $ac1, s5, t3    /* ac1 += t6* -2259 + t7*  9633 */
    mult        $ac2, $0, $0    /* ac2  = 0 */
    dpa.w.ph    $ac2, s7, t4    /* ac2 += t5*  2261 + t4*  9633 */
    dpa.w.ph    $ac2, s5, t5    /* ac2 += t6*-11362 + t7*  6437 */
    mult        $ac3, $0, $0    /* ac3  = 0 */
    dpa.w.ph    $ac3, s7, t6    /* ac3 += t5*  9633 + t4*-11363 */
    dpa.w.ph    $ac3, s5, t7    /* ac3 += t6* -6436 + t7*  2260 */
    addq.ph     s6, s1, s2      /* tmp6 = 2+5|3+4 = t2|t3 */
    addq.ph     s4, s0, s3      /* tmp4 = 1+6|0+7 = t1|t0 */
    extr_r.w    s0, $ac0, 11    /* tmp0 = (ac0 + 1024) >> 11 */
    extr_r.w    s1, $ac1, 11    /* tmp1 = (ac1 + 1024) >> 11 */
    extr_r.w    s2, $ac2, 11    /* tmp2 = (ac2 + 1024) >> 11 */
    extr_r.w    s3, $ac3, 11    /* tmp3 = (ac3 + 1024) >> 11 */
    addq.ph     s5, s4, s6      /* tmp5 = t1+t2|t0+t3 = t11|t10 */
    subq.ph     s7, s4, s6      /* tmp7 = t1-t2|t0-t3 = t12|t13 */
    sh          s0, 2(a1)
    sh          s1, 6(a1)
    sh          s2, 10(a1)
    sh          s3, 14(a1)
    mult        $0, $0          /* ac0  = 0 */
    dpa.w.ph    $ac0, s7, t8    /* ac0 += t12*  4433 + t13* 10703 */
    mult        $ac1, $0, $0    /* ac1  = 0 */
    dpa.w.ph    $ac1, s7, t9    /* ac1 += t12*-10704 + t13*  4433 */
    sra         s4, s5, 16      /* tmp4 = t11 */
    addiu       a1, a1, 16
    addiu       s8, s8, -1
    extr_r.w    s0, $ac0, 11    /* tmp0 = (ac0 + 1024) >> 11 */
    extr_r.w    s1, $ac1, 11    /* tmp1 = (ac1 + 1024) >> 11 */
    addu        s2, s5, s4      /* tmp2 = t10 + t11 */
    subu        s3, s5, s4      /* tmp3 = t10 - t11 */
    sll         s2, s2, 2       /* tmp2 = (t10 + t11) << 2 */
    sll         s3, s3, 2       /* tmp3 = (t10 - t11) << 2 */
    sh          s2, -16(a1)
    sh          s3, -8(a1)
    sh          s0, -12(a1)
    bgtz        s8, 1b
     sh         s1, -4(a1)
    li          t0, 2260
    li          t1, 11363
    li          t2, 9633
    li          t3, 6436
    li          t4, 6437
    li          t5, 2261
    li          t6, 11362
    li          t7, 2259
    li          t8, 4433
    li          t9, 10703
    li          a1, 10704
    li          s8, 8

2:
    lh          a2, 0(a0)       /* 0 */
    lh          a3, 16(a0)      /* 8 */
    lh          v0, 32(a0)      /* 16 */
    lh          v1, 48(a0)      /* 24 */
    lh          s4, 64(a0)      /* 32 */
    lh          s5, 80(a0)      /* 40 */
    lh          s6, 96(a0)      /* 48 */
    lh          s7, 112(a0)     /* 56 */
    addu        s2, v0, s5      /* tmp2 = 16 + 40 */
    subu        s5, v0, s5      /* tmp5 = 16 - 40 */
    addu        s3, v1, s4      /* tmp3 = 24 + 32 */
    subu        s4, v1, s4      /* tmp4 = 24 - 32 */
    addu        s0, a2, s7      /* tmp0 =  0 + 56 */
    subu        s7, a2, s7      /* tmp7 =  0 - 56 */
    addu        s1, a3, s6      /* tmp1 =  8 + 48 */
    subu        s6, a3, s6      /* tmp6 =  8 - 48 */
    addu        a2, s0, s3      /* tmp10 = tmp0 + tmp3 */
    subu        v1, s0, s3      /* tmp13 = tmp0 - tmp3 */
    addu        a3, s1, s2      /* tmp11 = tmp1 + tmp2 */
    subu        v0, s1, s2      /* tmp12 = tmp1 - tmp2 */
    mult        s7, t1          /* ac0  = tmp7 * c1 */
    madd        s4, t0          /* ac0 += tmp4 * c0 */
    madd        s5, t4          /* ac0 += tmp5 * c4 */
    madd        s6, t2          /* ac0 += tmp6 * c2 */
    mult        $ac1, s7, t2    /* ac1  = tmp7 * c2 */
    msub        $ac1, s4, t3    /* ac1 -= tmp4 * c3 */
    msub        $ac1, s5, t6    /* ac1 -= tmp5 * c6 */
    msub        $ac1, s6, t7    /* ac1 -= tmp6 * c7 */
    mult        $ac2, s7, t4    /* ac2  = tmp7 * c4 */
    madd        $ac2, s4, t2    /* ac2 += tmp4 * c2 */
    madd        $ac2, s5, t5    /* ac2 += tmp5 * c5 */
    msub        $ac2, s6, t6    /* ac2 -= tmp6 * c6 */
    mult        $ac3, s7, t0    /* ac3  = tmp7 * c0 */
    msub        $ac3, s4, t1    /* ac3 -= tmp4 * c1 */
    madd        $ac3, s5, t2    /* ac3 += tmp5 * c2 */
    msub        $ac3, s6, t3    /* ac3 -= tmp6 * c3 */
    extr_r.w    s0, $ac0, 15    /* tmp0 = (ac0 + 16384) >> 15 */
    extr_r.w    s1, $ac1, 15    /* tmp1 = (ac1 + 16384) >> 15 */
    extr_r.w    s2, $ac2, 15    /* tmp2 = (ac2 + 16384) >> 15 */
    extr_r.w    s3, $ac3, 15    /* tmp3 = (ac3 + 16384) >> 15 */
    addiu       s8, s8, -1
    addu        s4, a2, a3      /* tmp4 = tmp10 + tmp11 */
    subu        s5, a2, a3      /* tmp5 = tmp10 - tmp11 */
    sh          s0, 16(a0)
    sh          s1, 48(a0)
    sh          s2, 80(a0)
    sh          s3, 112(a0)
    mult        v0, t8          /* ac0  = tmp12 * c8 */
    madd        v1, t9          /* ac0 += tmp13 * c9 */
    mult        $ac1, v1, t8    /* ac1  = tmp13 * c8 */
    msub        $ac1, v0, a1    /* ac1 -= tmp12 * c10 */
    addiu       a0, a0, 2
    extr_r.w    s6, $ac0, 15    /* tmp6 = (ac0 + 16384) >> 15 */
    extr_r.w    s7, $ac1, 15    /* tmp7 = (ac1 + 16384) >> 15 */
    shra_r.w    s4, s4, 2       /* tmp4 = (tmp4 + 2) >> 2 */
    shra_r.w    s5, s5, 2       /* tmp5 = (tmp5 + 2) >> 2 */
    sh          s4, -2(a0)
    sh          s5, 62(a0)
    sh          s6, 30(a0)
    bgtz        s8, 2b
     sh         s7, 94(a0)

    RESTORE_REGS_FROM_STACK 40, s0, s1, s2, s3, s4, s5, s6, s7, s8

    jr          ra
     nop

END(jsimd_fdct_islow_dspr2)


/**************************************************************************/
LEAF_DSPR2(jsimd_fdct_ifast_dspr2)
/*
 * a0 = data
 */
    .set at

    SAVE_REGS_ON_STACK 8, s0, s1

    li          a1, 0x014e014e  /* FIX_1_306562965 (334 << 16) |
                                                   (334 & 0xffff) */
    li          a2, 0x008b008b  /* FIX_0_541196100 (139 << 16) |
                                                   (139 & 0xffff) */
    li          a3, 0x00620062  /* FIX_0_382683433 (98 << 16) |
                                                   (98 & 0xffff) */
    li          s1, 0x00b500b5  /* FIX_0_707106781 (181 << 16) |
                                                   (181 & 0xffff) */

    move        v0, a0
    addiu       v1, v0, 128     /* end address */

0:
    lw          t0, 0(v0)       /* tmp0 = 1|0 */
    lw          t1, 4(v0)       /* tmp1 = 3|2 */
    lw          t2, 8(v0)       /* tmp2 = 5|4 */
    lw          t3, 12(v0)      /* tmp3 = 7|6 */
    packrl.ph   t1, t1, t1      /* tmp1 = 2|3 */
    packrl.ph   t3, t3, t3      /* tmp3 = 6|7 */
    subq.ph     t7, t1, t2      /* tmp7 = 2-5|3-4 = t5|t4 */
    subq.ph     t5, t0, t3      /* tmp5 = 1-6|0-7 = t6|t7 */
    addq.ph     t6, t1, t2      /* tmp6 = 2+5|3+4 = t2|t3 */
    addq.ph     t4, t0, t3      /* tmp4 = 1+6|0+7 = t1|t0 */
    addq.ph     t8, t4, t6      /* tmp5 = t1+t2|t0+t3 = t11|t10 */
    subq.ph     t9, t4, t6      /* tmp7 = t1-t2|t0-t3 = t12|t13 */
    sra         t4, t8, 16      /* tmp4 = t11 */
    mult        $0, $0          /* ac0  = 0 */
    dpa.w.ph    $ac0, t9, s1
    mult        $ac1, $0, $0    /* ac1  = 0 */
    dpa.w.ph    $ac1, t7, a3    /* ac1 += t4*98 + t5*98 */
    dpsx.w.ph   $ac1, t5, a3    /* ac1 += t6*98 + t7*98 */
    mult        $ac2, $0, $0    /* ac2  = 0 */
    dpa.w.ph    $ac2, t7, a2    /* ac2 += t4*139 + t5*139 */
    mult        $ac3, $0, $0    /* ac3  = 0 */
    dpa.w.ph    $ac3, t5, a1    /* ac3 += t6*334 + t7*334 */
    precrq.ph.w t0, t5, t7      /* t0 = t5|t6 */
    addq.ph     t2, t8, t4      /* tmp2 = t10 + t11 */
    subq.ph     t3, t8, t4      /* tmp3 = t10 - t11 */
    extr.w      t4, $ac0, 8
    mult        $0, $0          /* ac0  = 0 */
    dpa.w.ph    $ac0, t0, s1    /* ac0 += t5*181 + t6*181 */
    extr.w      t0, $ac1, 8     /* t0 = z5 */
    extr.w      t1, $ac2, 8     /* t1 = MULTIPLY(tmp10, 139) */
    extr.w      t7, $ac3, 8     /* t2 = MULTIPLY(tmp12, 334) */
    extr.w      t8, $ac0, 8     /* t8 = z3 = MULTIPLY(tmp11, 181) */
    add         t6, t1, t0      /* t6 = z2 */
    add         t7, t7, t0      /* t7 = z4 */
    subq.ph     t0, t5, t8      /* t0 = z13 = tmp7 - z3 */
    addq.ph     t8, t5, t8      /* t9 = z11 = tmp7 + z3 */
    addq.ph     t1, t0, t6      /* t1 = z13 + z2 */
    subq.ph     t6, t0, t6      /* t6 = z13 - z2 */
    addq.ph     t0, t8, t7      /* t0 = z11 + z4 */
    subq.ph     t7, t8, t7      /* t7 = z11 - z4 */
    addq.ph     t5, t4, t9
    subq.ph     t4, t9, t4
    sh          t2, 0(v0)
    sh          t5, 4(v0)
    sh          t3, 8(v0)
    sh          t4, 12(v0)
    sh          t1, 10(v0)
    sh          t6, 6(v0)
    sh          t0, 2(v0)
    sh          t7, 14(v0)
    addiu       v0, 16
    bne         v1, v0, 0b
     nop
    move        v0, a0
    addiu       v1, v0, 16

1:
    lh          t0, 0(v0)       /* 0 */
    lh          t1, 16(v0)      /* 8 */
    lh          t2, 32(v0)      /* 16 */
    lh          t3, 48(v0)      /* 24 */
    lh          t4, 64(v0)      /* 32 */
    lh          t5, 80(v0)      /* 40 */
    lh          t6, 96(v0)      /* 48 */
    lh          t7, 112(v0)     /* 56 */
    add         t8, t0, t7      /* t8 = tmp0 */
    sub         t7, t0, t7      /* t7 = tmp7 */
    add         t0, t1, t6      /* t0 = tmp1 */
    sub         t1, t1, t6      /* t1 = tmp6 */
    add         t6, t2, t5      /* t6 = tmp2 */
    sub         t5, t2, t5      /* t5 = tmp5 */
    add         t2, t3, t4      /* t2 = tmp3 */
    sub         t3, t3, t4      /* t3 = tmp4 */
    add         t4, t8, t2      /* t4 = tmp10 = tmp0 + tmp3 */
    sub         t8, t8, t2      /* t8 = tmp13 = tmp0 - tmp3 */
    sub         s0, t0, t6      /* s0 = tmp12 = tmp1 - tmp2 */
    ins         t8, s0, 16, 16  /* t8 = tmp12|tmp13 */
    add         t2, t0, t6      /* t2 = tmp11 = tmp1 + tmp2 */
    mult        $0, $0          /* ac0  = 0 */
    dpa.w.ph    $ac0, t8, s1    /* ac0 += t12*181 + t13*181 */
    add         s0, t4, t2      /* t8 = tmp10+tmp11 */
    sub         t4, t4, t2      /* t4 = tmp10-tmp11 */
    sh          s0, 0(v0)
    sh          t4, 64(v0)
    extr.w      t2, $ac0, 8     /* z1 = MULTIPLY(tmp12+tmp13,
                                                 FIX_0_707106781) */
    addq.ph     t4, t8, t2      /* t9 = tmp13 + z1 */
    subq.ph     t8, t8, t2      /* t2 = tmp13 - z1 */
    sh          t4, 32(v0)
    sh          t8, 96(v0)
    add         t3, t3, t5      /* t3 = tmp10 = tmp4 + tmp5 */
    add         t0, t5, t1      /* t0 = tmp11 = tmp5 + tmp6 */
    add         t1, t1, t7      /* t1 = tmp12 = tmp6 + tmp7 */
    andi        t4, a1, 0xffff
    mul         s0, t1, t4
    sra         s0, s0, 8       /* s0 = z4 =
                                     MULTIPLY(tmp12, FIX_1_306562965) */
    ins         t1, t3, 16, 16  /* t1 = tmp10|tmp12 */
    mult        $0, $0          /* ac0  = 0 */
    mulsa.w.ph  $ac0, t1, a3    /* ac0 += t10*98 - t12*98 */
    extr.w      t8, $ac0, 8     /* z5 = MULTIPLY(tmp10-tmp12,
                                                 FIX_0_382683433) */
    add         t2, t7, t8      /* t2 = tmp7 + z5 */
    sub         t7, t7, t8      /* t7 = tmp7 - z5 */
    andi        t4, a2, 0xffff
    mul         t8, t3, t4
    sra         t8, t8, 8       /* t8 = z2 =
                                     MULTIPLY(tmp10, FIX_0_541196100) */
    andi        t4, s1, 0xffff
    mul         t6, t0, t4
    sra         t6, t6, 8       /* t6 = z3 =
                                     MULTIPLY(tmp11, FIX_0_707106781) */
    add         t0, t6, t8      /* t0 = z3 + z2 */
    sub         t1, t6, t8      /* t1 = z3 - z2 */
    add         t3, t6, s0      /* t3 = z3 + z4 */
    sub         t4, t6, s0      /* t4 = z3 - z4 */
    sub         t5, t2, t1      /* t5 = dataptr[5] */
    sub         t6, t7, t0      /* t6 = dataptr[3] */
    add         t3, t2, t3      /* t3 = dataptr[1] */
    add         t4, t7, t4      /* t4 = dataptr[7] */
    sh          t5, 80(v0)
    sh          t6, 48(v0)
    sh          t3, 16(v0)
    sh          t4, 112(v0)
    addiu       v0, 2
    bne         v0, v1, 1b
     nop

    RESTORE_REGS_FROM_STACK 8, s0, s1

    j           ra
     nop
END(jsimd_fdct_ifast_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_quantize_dspr2)
/*
 * a0 = coef_block
 * a1 = divisors
 * a2 = workspace
 */
    .set at

    SAVE_REGS_ON_STACK 16, s0, s1, s2

    addiu       v0, a2, 124     /* v0 = workspace_end */
    lh          t0, 0(a2)
    lh          t1, 0(a1)
    lh          t2, 128(a1)
    sra         t3, t0, 15
    sll         t3, t3, 1
    addiu       t3, t3, 1
    mul         t0, t0, t3
    lh          t4, 384(a1)
    lh          t5, 130(a1)
    lh          t6, 2(a2)
    lh          t7, 2(a1)
    lh          t8, 386(a1)

1:
    andi        t1, 0xffff
    add         t9, t0, t2
    andi        t9, 0xffff
    mul         v1, t9, t1
    sra         s0, t6, 15
    sll         s0, s0, 1
    addiu       s0, s0, 1
    addiu       t9, t4, 16
    srav        v1, v1, t9
    mul         v1, v1, t3
    mul         t6, t6, s0
    andi        t7, 0xffff
    addiu       a2, a2, 4
    addiu       a1, a1, 4
    add         s1, t6, t5
    andi        s1, 0xffff
    sh          v1, 0(a0)

    mul         s2, s1, t7
    addiu       s1, t8, 16
    srav        s2, s2, s1
    mul         s2, s2, s0
    lh          t0, 0(a2)
    lh          t1, 0(a1)
    sra         t3, t0, 15
    sll         t3, t3, 1
    addiu       t3, t3, 1
    mul         t0, t0, t3
    lh          t2, 128(a1)
    lh          t4, 384(a1)
    lh          t5, 130(a1)
    lh          t8, 386(a1)
    lh          t6, 2(a2)
    lh          t7, 2(a1)
    sh          s2, 2(a0)
    lh          t0, 0(a2)
    sra         t3, t0, 15
    sll         t3, t3, 1
    addiu       t3, t3, 1
    mul         t0, t0, t3
    bne         a2, v0, 1b
     addiu      a0, a0, 4

    andi        t1, 0xffff
    add         t9, t0, t2
    andi        t9, 0xffff
    mul         v1, t9, t1
    sra         s0, t6, 15
    sll         s0, s0, 1
    addiu       s0, s0, 1
    addiu       t9, t4, 16
    srav        v1, v1, t9
    mul         v1, v1, t3
    mul         t6, t6, s0
    andi        t7, 0xffff
    sh          v1, 0(a0)
    add         s1, t6, t5
    andi        s1, 0xffff
    mul         s2, s1, t7
    addiu       s1, t8, 16
    addiu       a2, a2, 4
    addiu       a1, a1, 4
    srav        s2, s2, s1
    mul         s2, s2, s0
    sh          s2, 2(a0)

    RESTORE_REGS_FROM_STACK 16, s0, s1, s2

    j           ra
     nop

END(jsimd_quantize_dspr2)


#ifndef __mips_soft_float

/*****************************************************************************/
LEAF_DSPR2(jsimd_quantize_float_dspr2)
/*
 * a0 = coef_block
 * a1 = divisors
 * a2 = workspace
 */
    .set at

    li          t1, 0x46800100  /* integer representation 16384.5 */
    mtc1        t1, f0
    li          t0, 63
0:
    lwc1        f2, 0(a2)
    lwc1        f10, 0(a1)
    lwc1        f4, 4(a2)
    lwc1        f12, 4(a1)
    lwc1        f6, 8(a2)
    lwc1        f14, 8(a1)
    lwc1        f8, 12(a2)
    lwc1        f16, 12(a1)
    madd.s      f2, f0, f2, f10
    madd.s      f4, f0, f4, f12
    madd.s      f6, f0, f6, f14
    madd.s      f8, f0, f8, f16
    lwc1        f10, 16(a1)
    lwc1        f12, 20(a1)
    trunc.w.s   f2, f2
    trunc.w.s   f4, f4
    trunc.w.s   f6, f6
    trunc.w.s   f8, f8
    lwc1        f14, 24(a1)
    lwc1        f16, 28(a1)
    mfc1        t1, f2
    mfc1        t2, f4
    mfc1        t3, f6
    mfc1        t4, f8
    lwc1        f2, 16(a2)
    lwc1        f4, 20(a2)
    lwc1        f6, 24(a2)
    lwc1        f8, 28(a2)
    madd.s      f2, f0, f2, f10
    madd.s      f4, f0, f4, f12
    madd.s      f6, f0, f6, f14
    madd.s      f8, f0, f8, f16
    addiu       t1, t1, -16384
    addiu       t2, t2, -16384
    addiu       t3, t3, -16384
    addiu       t4, t4, -16384
    trunc.w.s   f2, f2
    trunc.w.s   f4, f4
    trunc.w.s   f6, f6
    trunc.w.s   f8, f8
    sh          t1, 0(a0)
    sh          t2, 2(a0)
    sh          t3, 4(a0)
    sh          t4, 6(a0)
    mfc1        t1, f2
    mfc1        t2, f4
    mfc1        t3, f6
    mfc1        t4, f8
    addiu       t0, t0, -8
    addiu       a2, a2, 32
    addiu       a1, a1, 32
    addiu       t1, t1, -16384
    addiu       t2, t2, -16384
    addiu       t3, t3, -16384
    addiu       t4, t4, -16384
    sh          t1, 8(a0)
    sh          t2, 10(a0)
    sh          t3, 12(a0)
    sh          t4, 14(a0)
    bgez        t0, 0b
     addiu      a0, a0, 16

    j           ra
     nop

END(jsimd_quantize_float_dspr2)

#endif


/*****************************************************************************/
LEAF_DSPR2(jsimd_idct_2x2_dspr2)
/*
 * a0 = compptr->dct_table
 * a1 = coef_block
 * a2 = output_buf
 * a3 = output_col
 */
    .set at

    SAVE_REGS_ON_STACK 24, s0, s1, s2, s3, s4, s5

    addiu       sp, sp, -40
    move        v0, sp
    addiu       s2, zero, 29692
    addiu       s3, zero, -10426
    addiu       s4, zero, 6967
    addiu       s5, zero, -5906
    lh          t0, 0(a1)       /* t0 = inptr[DCTSIZE*0] */
    lh          t5, 0(a0)       /* t5 = quantptr[DCTSIZE*0] */
    lh          t1, 48(a1)      /* t1 = inptr[DCTSIZE*3] */
    lh          t6, 48(a0)      /* t6 = quantptr[DCTSIZE*3] */
    mul         t4, t5, t0
    lh          t0, 16(a1)      /* t0 = inptr[DCTSIZE*1] */
    lh          t5, 16(a0)      /* t5 = quantptr[DCTSIZE*1] */
    mul         t6, t6, t1
    mul         t5, t5, t0
    lh          t2, 80(a1)      /* t2 = inptr[DCTSIZE*5] */
    lh          t7, 80(a0)      /* t7 = quantptr[DCTSIZE*5] */
    lh          t3, 112(a1)     /* t3 = inptr[DCTSIZE*7] */
    lh          t8, 112(a0)     /* t8 = quantptr[DCTSIZE*7] */
    mul         t7, t7, t2
    mult        zero, zero
    mul         t8, t8, t3
    li          s0, 0x73FCD746  /* s0 = (29692 << 16) | (-10426 & 0xffff) */
    li          s1, 0x1B37E8EE  /* s1 = (6967 << 16) | (-5906 & 0xffff) */
    ins         t6, t5, 16, 16  /* t6 = t5|t6 */
    sll         t4, t4, 15
    dpa.w.ph    $ac0, t6, s0
    lh          t1, 2(a1)
    lh          t6, 2(a0)
    ins         t8, t7, 16, 16  /* t8 = t7|t8 */
    dpa.w.ph    $ac0, t8, s1
    mflo        t0, $ac0
    mul         t5, t6, t1
    lh          t1, 18(a1)
    lh          t6, 18(a0)
    lh          t2, 50(a1)
    lh          t7, 50(a0)
    mul         t6, t6, t1
    subu        t8, t4, t0
    mul         t7, t7, t2
    addu        t0, t4, t0
    shra_r.w    t0, t0, 13
    lh          t1, 82(a1)
    lh          t2, 82(a0)
    lh          t3, 114(a1)
    lh          t4, 114(a0)
    shra_r.w    t8, t8, 13
    mul         t1, t1, t2
    mul         t3, t3, t4
    sw          t0, 0(v0)
    sw          t8, 20(v0)
    sll         t4, t5, 15
    ins         t7, t6, 16, 16
    mult        zero, zero
    dpa.w.ph    $ac0, t7, s0
    ins         t3, t1, 16, 16
    lh          t1, 6(a1)
    lh          t6, 6(a0)
    dpa.w.ph    $ac0, t3, s1
    mflo        t0, $ac0
    mul         t5, t6, t1
    lh          t1, 22(a1)
    lh          t6, 22(a0)
    lh          t2, 54(a1)
    lh          t7, 54(a0)
    mul         t6, t6, t1
    subu        t8, t4, t0
    mul         t7, t7, t2
    addu        t0, t4, t0
    shra_r.w    t0, t0, 13
    lh          t1, 86(a1)
    lh          t2, 86(a0)
    lh          t3, 118(a1)
    lh          t4, 118(a0)
    shra_r.w    t8, t8, 13
    mul         t1, t1, t2
    mul         t3, t3, t4
    sw          t0, 4(v0)
    sw          t8, 24(v0)
    sll         t4, t5, 15
    ins         t7, t6, 16, 16
    mult        zero, zero
    dpa.w.ph    $ac0, t7, s0
    ins         t3, t1, 16, 16
    lh          t1, 10(a1)
    lh          t6, 10(a0)
    dpa.w.ph    $ac0, t3, s1
    mflo        t0, $ac0
    mul         t5, t6, t1
    lh          t1, 26(a1)
    lh          t6, 26(a0)
    lh          t2, 58(a1)
    lh          t7, 58(a0)
    mul         t6, t6, t1
    subu        t8, t4, t0
    mul         t7, t7, t2
    addu        t0, t4, t0
    shra_r.w    t0, t0, 13
    lh          t1, 90(a1)
    lh          t2, 90(a0)
    lh          t3, 122(a1)
    lh          t4, 122(a0)
    shra_r.w    t8, t8, 13
    mul         t1, t1, t2
    mul         t3, t3, t4
    sw          t0, 8(v0)
    sw          t8, 28(v0)
    sll         t4, t5, 15
    ins         t7, t6, 16, 16
    mult        zero, zero
    dpa.w.ph    $ac0, t7, s0
    ins         t3, t1, 16, 16
    lh          t1, 14(a1)
    lh          t6, 14(a0)
    dpa.w.ph    $ac0, t3, s1
    mflo        t0, $ac0
    mul         t5, t6, t1
    lh          t1, 30(a1)
    lh          t6, 30(a0)
    lh          t2, 62(a1)
    lh          t7, 62(a0)
    mul         t6, t6, t1
    subu        t8, t4, t0
    mul         t7, t7, t2
    addu        t0, t4, t0
    shra_r.w    t0, t0, 13
    lh          t1, 94(a1)
    lh          t2, 94(a0)
    lh          t3, 126(a1)
    lh          t4, 126(a0)
    shra_r.w    t8, t8, 13
    mul         t1, t1, t2
    mul         t3, t3, t4
    sw          t0, 12(v0)
    sw          t8, 32(v0)
    sll         t4, t5, 15
    ins         t7, t6, 16, 16
    mult        zero, zero
    dpa.w.ph    $ac0, t7, s0
    ins         t3, t1, 16, 16
    dpa.w.ph    $ac0, t3, s1
    mflo        t0, $ac0
    lw          t9, 0(a2)
    lw          t3, 0(v0)
    lw          t7, 4(v0)
    lw          t1, 8(v0)
    addu        t9, t9, a3
    sll         t3, t3, 15
    subu        t8, t4, t0
    addu        t0, t4, t0
    shra_r.w    t0, t0, 13
    shra_r.w    t8, t8, 13
    sw          t0, 16(v0)
    sw          t8, 36(v0)
    lw          t5, 12(v0)
    lw          t6, 16(v0)
    mult        t7, s2
    madd        t1, s3
    madd        t5, s4
    madd        t6, s5
    lw          t5, 24(v0)
    lw          t7, 28(v0)
    mflo        t0, $ac0
    lw          t8, 32(v0)
    lw          t2, 36(v0)
    mult        $ac1, t5, s2
    madd        $ac1, t7, s3
    madd        $ac1, t8, s4
    madd        $ac1, t2, s5
    addu        t1, t3, t0
    subu        t6, t3, t0
    shra_r.w    t1, t1, 20
    shra_r.w    t6, t6, 20
    mflo        t4, $ac1
    shll_s.w    t1, t1, 24
    shll_s.w    t6, t6, 24
    sra         t1, t1, 24
    sra         t6, t6, 24
    addiu       t1, t1, 128
    addiu       t6, t6, 128
    lw          t0, 20(v0)
    sb          t1, 0(t9)
    sb          t6, 1(t9)
    sll         t0, t0, 15
    lw          t9, 4(a2)
    addu        t1, t0, t4
    subu        t6, t0, t4
    addu        t9, t9, a3
    shra_r.w    t1, t1, 20
    shra_r.w    t6, t6, 20
    shll_s.w    t1, t1, 24
    shll_s.w    t6, t6, 24
    sra         t1, t1, 24
    sra         t6, t6, 24
    addiu       t1, t1, 128
    addiu       t6, t6, 128
    sb          t1, 0(t9)
    sb          t6, 1(t9)
    addiu       sp, sp, 40

    RESTORE_REGS_FROM_STACK 24, s0, s1, s2, s3, s4, s5

    j           ra
     nop

END(jsimd_idct_2x2_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_idct_4x4_dspr2)
/*
 * a0     = compptr->dct_table
 * a1     = coef_block
 * a2     = output_buf
 * a3     = output_col
 * 16(sp) = workspace[DCTSIZE*4]  (buffers data between passes)
 */
    .set at

    SAVE_REGS_ON_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    lw          v1, 48(sp)
    move        t0, a1
    move        t1, v1
    li          t9, 4
    li          s0, 0x2e75f93e
    li          s1, 0x21f9ba79
    li          s2, 0xecc2efb0
    li          s3, 0x52031ccd

0:
    lh          s6, 32(t0)      /* inptr[DCTSIZE*2] */
    lh          t6, 32(a0)      /* quantptr[DCTSIZE*2] */
    lh          s7, 96(t0)      /* inptr[DCTSIZE*6] */
    lh          t7, 96(a0)      /* quantptr[DCTSIZE*6] */
    mul         t6, s6, t6      /* z2 = (inptr[DCTSIZE*2] *
                                         quantptr[DCTSIZE*2]) */
    lh          s4, 0(t0)       /* inptr[DCTSIZE*0] */
    mul         t7, s7, t7      /* z3 = (inptr[DCTSIZE*6] *
                                         quantptr[DCTSIZE*6]) */
    lh          s5, 0(a0)       /* quantptr[0] */
    li          s6, 15137
    li          s7, 6270
    mul         t2, s4, s5      /* tmp0 = (inptr[0] * quantptr[0]) */
    mul         t6, s6, t6      /* z2 = (inptr[DCTSIZE*2] *
                                         quantptr[DCTSIZE*2]) */
    lh          t5, 112(t0)     /* inptr[DCTSIZE*7] */
    mul         t7, s7, t7      /* z3 = (inptr[DCTSIZE*6] *
                                         quantptr[DCTSIZE*6]) */
    lh          s4, 112(a0)     /* quantptr[DCTSIZE*7] */
    lh          v0, 80(t0)      /* inptr[DCTSIZE*5] */
    lh          s5, 80(a0)      /* quantptr[DCTSIZE*5] */
    lh          s6, 48(a0)      /* quantptr[DCTSIZE*3] */
    sll         t2, t2, 14      /* tmp0 <<= (CONST_BITS+1) */
    lh          s7, 16(a0)      /* quantptr[DCTSIZE*1] */
    lh          t8, 16(t0)      /* inptr[DCTSIZE*1] */
    subu        t6, t6, t7      /* tmp2 =
                                     MULTIPLY(z2, t5) - MULTIPLY(z3, t6) */
    lh          t7, 48(t0)      /* inptr[DCTSIZE*3] */
    mul         t5, s4, t5      /* z1 = (inptr[DCTSIZE*7] *
                                         quantptr[DCTSIZE*7]) */
    mul         v0, s5, v0      /* z2 = (inptr[DCTSIZE*5] *
                                         quantptr[DCTSIZE*5]) */
    mul         t7, s6, t7      /* z3 = (inptr[DCTSIZE*3] *
                                         quantptr[DCTSIZE*3]) */
    mul         t8, s7, t8      /* z4 = (inptr[DCTSIZE*1] *
                                         quantptr[DCTSIZE*1]) */
    addu        t3, t2, t6      /* tmp10 = tmp0 + z2 */
    subu        t4, t2, t6      /* tmp10 = tmp0 - z2 */
    mult        $ac0, zero, zero
    mult        $ac1, zero, zero
    ins         t5, v0, 16, 16
    ins         t7, t8, 16, 16
    addiu       t9, t9, -1
    dpa.w.ph    $ac0, t5, s0
    dpa.w.ph    $ac0, t7, s1
    dpa.w.ph    $ac1, t5, s2
    dpa.w.ph    $ac1, t7, s3
    mflo        s4, $ac0
    mflo        s5, $ac1
    addiu       a0, a0, 2
    addiu       t1, t1, 4
    addiu       t0, t0, 2
    addu        t6, t4, s4
    subu        t5, t4, s4
    addu        s6, t3, s5
    subu        s7, t3, s5
    shra_r.w    t6, t6, 12      /* DESCALE(tmp12 + temp1, 12) */
    shra_r.w    t5, t5, 12      /* DESCALE(tmp12 - temp1, 12) */
    shra_r.w    s6, s6, 12      /* DESCALE(tmp10 + temp2, 12) */
    shra_r.w    s7, s7, 12      /* DESCALE(tmp10 - temp2, 12) */
    sw          t6, 28(t1)
    sw          t5, 60(t1)
    sw          s6, -4(t1)
    bgtz        t9, 0b
     sw         s7, 92(t1)
    /* second loop three pass */
    li          t9, 3
1:
    lh          s6, 34(t0)      /* inptr[DCTSIZE*2] */
    lh          t6, 34(a0)      /* quantptr[DCTSIZE*2] */
    lh          s7, 98(t0)      /* inptr[DCTSIZE*6] */
    lh          t7, 98(a0)      /* quantptr[DCTSIZE*6] */
    mul         t6, s6, t6      /* z2 = (inptr[DCTSIZE*2] *
                                         quantptr[DCTSIZE*2]) */
    lh          s4, 2(t0)       /* inptr[DCTSIZE*0] */
    mul         t7, s7, t7      /* z3 = (inptr[DCTSIZE*6] *
                                         quantptr[DCTSIZE*6]) */
    lh          s5, 2(a0)       /* quantptr[DCTSIZE*0] */
    li          s6, 15137
    li          s7, 6270
    mul         t2, s4, s5      /* tmp0 = (inptr[0] * quantptr[0]) */
    mul         v0, s6, t6      /* z2 = (inptr[DCTSIZE*2] *
                                         quantptr[DCTSIZE*2]) */
    lh          t5, 114(t0)     /* inptr[DCTSIZE*7] */
    mul         t7, s7, t7      /* z3 = (inptr[DCTSIZE*6] *
                                         quantptr[DCTSIZE*6]) */
    lh          s4, 114(a0)     /* quantptr[DCTSIZE*7] */
    lh          s5, 82(a0)      /* quantptr[DCTSIZE*5] */
    lh          t6, 82(t0)      /* inptr[DCTSIZE*5] */
    sll         t2, t2, 14      /* tmp0 <<= (CONST_BITS+1) */
    lh          s6, 50(a0)      /* quantptr[DCTSIZE*3] */
    lh          t8, 18(t0)      /* inptr[DCTSIZE*1] */
    subu        v0, v0, t7      /* tmp2 =
                                     MULTIPLY(z2, t5) - MULTIPLY(z3, t6) */
    lh          t7, 50(t0)      /* inptr[DCTSIZE*3] */
    lh          s7, 18(a0)      /* quantptr[DCTSIZE*1] */
    mul         t5, s4, t5      /* z1 = (inptr[DCTSIZE*7] *
                                         quantptr[DCTSIZE*7]) */
    mul         t6, s5, t6      /* z2 = (inptr[DCTSIZE*5] *
                                         quantptr[DCTSIZE*5]) */
    mul         t7, s6, t7      /* z3 = (inptr[DCTSIZE*3] *
                                         quantptr[DCTSIZE*3]) */
    mul         t8, s7, t8      /* z4 = (inptr[DCTSIZE*1] *
                                         quantptr[DCTSIZE*1]) */
    addu        t3, t2, v0      /* tmp10 = tmp0 + z2 */
    subu        t4, t2, v0      /* tmp10 = tmp0 - z2 */
    mult        $ac0, zero, zero
    mult        $ac1, zero, zero
    ins         t5, t6, 16, 16
    ins         t7, t8, 16, 16
    dpa.w.ph    $ac0, t5, s0
    dpa.w.ph    $ac0, t7, s1
    dpa.w.ph    $ac1, t5, s2
    dpa.w.ph    $ac1, t7, s3
    mflo        t5, $ac0
    mflo        t6, $ac1
    addiu       t9, t9, -1
    addiu       t0, t0, 2
    addiu       a0, a0, 2
    addiu       t1, t1, 4
    addu        s5, t4, t5
    subu        s4, t4, t5
    addu        s6, t3, t6
    subu        s7, t3, t6
    shra_r.w    s5, s5, 12      /* DESCALE(tmp12 + temp1, 12) */
    shra_r.w    s4, s4, 12      /* DESCALE(tmp12 - temp1, 12) */
    shra_r.w    s6, s6, 12      /* DESCALE(tmp10 + temp2, 12) */
    shra_r.w    s7, s7, 12      /* DESCALE(tmp10 - temp2, 12) */
    sw          s5, 32(t1)
    sw          s4, 64(t1)
    sw          s6, 0(t1)
    bgtz        t9, 1b
     sw         s7, 96(t1)
    move        t1, v1
    li          s4, 15137
    lw          s6, 8(t1)       /* wsptr[2] */
    li          s5, 6270
    lw          s7, 24(t1)      /* wsptr[6] */
    mul         s4, s4, s6      /* MULTIPLY((JLONG)wsptr[2],
                                            FIX_1_847759065) */
    lw          t2, 0(t1)       /* wsptr[0] */
    mul         s5, s5, s7      /* MULTIPLY((JLONG)wsptr[6],
                                            -FIX_0_765366865) */
    lh          t5, 28(t1)      /* wsptr[7] */
    lh          t6, 20(t1)      /* wsptr[5] */
    lh          t7, 12(t1)      /* wsptr[3] */
    lh          t8, 4(t1)       /* wsptr[1] */
    ins         t5, t6, 16, 16
    ins         t7, t8, 16, 16
    mult        $ac0, zero, zero
    dpa.w.ph    $ac0, t5, s0
    dpa.w.ph    $ac0, t7, s1
    mult        $ac1, zero, zero
    dpa.w.ph    $ac1, t5, s2
    dpa.w.ph    $ac1, t7, s3
    sll         t2, t2, 14      /* tmp0 =
                                     ((JLONG)wsptr[0]) << (CONST_BITS+1) */
    mflo        s6, $ac0
    /* MULTIPLY(wsptr[2], FIX_1_847759065) +
       MULTIPLY(wsptr[6], -FIX_0_765366865) */
    subu        s4, s4, s5
    addu        t3, t2, s4      /* tmp10 = tmp0 + z2 */
    mflo        s7, $ac1
    subu        t4, t2, s4      /* tmp10 = tmp0 - z2 */
    addu        t7, t4, s6
    subu        t8, t4, s6
    addu        t5, t3, s7
    subu        t6, t3, s7
    shra_r.w    t5, t5, 19      /* DESCALE(tmp10 + temp2, 19) */
    shra_r.w    t6, t6, 19      /* DESCALE(tmp10 - temp2, 19) */
    shra_r.w    t7, t7, 19      /* DESCALE(tmp12 + temp1, 19) */
    shra_r.w    t8, t8, 19      /* DESCALE(tmp12 - temp1, 19) */
    sll         s4, t9, 2
    lw          v0, 0(a2)       /* output_buf[ctr] */
    shll_s.w    t5, t5, 24
    shll_s.w    t6, t6, 24
    shll_s.w    t7, t7, 24
    shll_s.w    t8, t8, 24
    sra         t5, t5, 24
    sra         t6, t6, 24
    sra         t7, t7, 24
    sra         t8, t8, 24
    addu        v0, v0, a3      /* outptr = output_buf[ctr] + output_col */
    addiu       t5, t5, 128
    addiu       t6, t6, 128
    addiu       t7, t7, 128
    addiu       t8, t8, 128
    sb          t5, 0(v0)
    sb          t7, 1(v0)
    sb          t8, 2(v0)
    sb          t6, 3(v0)
    /* 2 */
    li          s4, 15137
    lw          s6, 40(t1)      /* wsptr[2] */
    li          s5, 6270
    lw          s7, 56(t1)      /* wsptr[6] */
    mul         s4, s4, s6      /* MULTIPLY((JLONG)wsptr[2],
                                            FIX_1_847759065) */
    lw          t2, 32(t1)      /* wsptr[0] */
    mul         s5, s5, s7      /* MULTIPLY((JLONG)wsptr[6],
                                            -FIX_0_765366865) */
    lh          t5, 60(t1)      /* wsptr[7] */
    lh          t6, 52(t1)      /* wsptr[5] */
    lh          t7, 44(t1)      /* wsptr[3] */
    lh          t8, 36(t1)      /* wsptr[1] */
    ins         t5, t6, 16, 16
    ins         t7, t8, 16, 16
    mult        $ac0, zero, zero
    dpa.w.ph    $ac0, t5, s0
    dpa.w.ph    $ac0, t7, s1
    mult        $ac1, zero, zero
    dpa.w.ph    $ac1, t5, s2
    dpa.w.ph    $ac1, t7, s3
    sll         t2, t2, 14      /* tmp0 =
                                     ((JLONG)wsptr[0]) << (CONST_BITS+1) */
    mflo        s6, $ac0
    /* MULTIPLY(wsptr[2], FIX_1_847759065) +
       MULTIPLY(wsptr[6], -FIX_0_765366865) */
    subu        s4, s4, s5
    addu        t3, t2, s4      /* tmp10 = tmp0 + z2 */
    mflo        s7, $ac1
    subu        t4, t2, s4      /* tmp10 = tmp0 - z2 */
    addu        t7, t4, s6
    subu        t8, t4, s6
    addu        t5, t3, s7
    subu        t6, t3, s7
    shra_r.w    t5, t5, 19      /* DESCALE(tmp10 + temp2,
                                           CONST_BITS-PASS1_BITS+1) */
    shra_r.w    t6, t6, 19      /* DESCALE(tmp10 - temp2,
                                           CONST_BITS-PASS1_BITS+1) */
    shra_r.w    t7, t7, 19      /* DESCALE(tmp12 + temp1,
                                           CONST_BITS-PASS1_BITS+1) */
    shra_r.w    t8, t8, 19      /* DESCALE(tmp12 - temp1,
                                           CONST_BITS-PASS1_BITS+1) */
    sll         s4, t9, 2
    lw          v0, 4(a2)       /* output_buf[ctr] */
    shll_s.w    t5, t5, 24
    shll_s.w    t6, t6, 24
    shll_s.w    t7, t7, 24
    shll_s.w    t8, t8, 24
    sra         t5, t5, 24
    sra         t6, t6, 24
    sra         t7, t7, 24
    sra         t8, t8, 24
    addu        v0, v0, a3      /* outptr = output_buf[ctr] + output_col */
    addiu       t5, t5, 128
    addiu       t6, t6, 128
    addiu       t7, t7, 128
    addiu       t8, t8, 128
    sb          t5, 0(v0)
    sb          t7, 1(v0)
    sb          t8, 2(v0)
    sb          t6, 3(v0)
    /* 3 */
    li          s4, 15137
    lw          s6, 72(t1)      /* wsptr[2] */
    li          s5, 6270
    lw          s7, 88(t1)      /* wsptr[6] */
    mul         s4, s4, s6      /* MULTIPLY((JLONG)wsptr[2],
                                            FIX_1_847759065) */
    lw          t2, 64(t1)      /* wsptr[0] */
    mul         s5, s5, s7      /* MULTIPLY((JLONG)wsptr[6],
                                            -FIX_0_765366865) */
    lh          t5, 92(t1)      /* wsptr[7] */
    lh          t6, 84(t1)      /* wsptr[5] */
    lh          t7, 76(t1)      /* wsptr[3] */
    lh          t8, 68(t1)      /* wsptr[1] */
    ins         t5, t6, 16, 16
    ins         t7, t8, 16, 16
    mult        $ac0, zero, zero
    dpa.w.ph    $ac0, t5, s0
    dpa.w.ph    $ac0, t7, s1
    mult        $ac1, zero, zero
    dpa.w.ph    $ac1, t5, s2
    dpa.w.ph    $ac1, t7, s3
    sll         t2, t2, 14      /* tmp0 =
                                     ((JLONG)wsptr[0]) << (CONST_BITS+1) */
    mflo        s6, $ac0
    /* MULTIPLY(wsptr[2], FIX_1_847759065) +
       MULTIPLY(wsptr[6], -FIX_0_765366865) */
    subu        s4, s4, s5
    addu        t3, t2, s4      /* tmp10 = tmp0 + z2 */
    mflo        s7, $ac1
    subu        t4, t2, s4      /* tmp10 = tmp0 - z2 */
    addu        t7, t4, s6
    subu        t8, t4, s6
    addu        t5, t3, s7
    subu        t6, t3, s7
    shra_r.w    t5, t5, 19      /* DESCALE(tmp10 + temp2, 19) */
    shra_r.w    t6, t6, 19      /* DESCALE(tmp10 - temp2, 19) */
    shra_r.w    t7, t7, 19      /* DESCALE(tmp12 + temp1, 19) */
    shra_r.w    t8, t8, 19      /* DESCALE(tmp12 - temp1, 19) */
    sll         s4, t9, 2
    lw          v0, 8(a2)       /* output_buf[ctr] */
    shll_s.w    t5, t5, 24
    shll_s.w    t6, t6, 24
    shll_s.w    t7, t7, 24
    shll_s.w    t8, t8, 24
    sra         t5, t5, 24
    sra         t6, t6, 24
    sra         t7, t7, 24
    sra         t8, t8, 24
    addu        v0, v0, a3      /* outptr = output_buf[ctr] + output_col */
    addiu       t5, t5, 128
    addiu       t6, t6, 128
    addiu       t7, t7, 128
    addiu       t8, t8, 128
    sb          t5, 0(v0)
    sb          t7, 1(v0)
    sb          t8, 2(v0)
    sb          t6, 3(v0)
    li          s4, 15137
    lw          s6, 104(t1)     /* wsptr[2] */
    li          s5, 6270
    lw          s7, 120(t1)     /* wsptr[6] */
    mul         s4, s4, s6      /* MULTIPLY((JLONG)wsptr[2],
                                            FIX_1_847759065) */
    lw          t2, 96(t1)      /* wsptr[0] */
    mul         s5, s5, s7      /* MULTIPLY((JLONG)wsptr[6],
                                            -FIX_0_765366865) */
    lh          t5, 124(t1)     /* wsptr[7] */
    lh          t6, 116(t1)     /* wsptr[5] */
    lh          t7, 108(t1)     /* wsptr[3] */
    lh          t8, 100(t1)     /* wsptr[1] */
    ins         t5, t6, 16, 16
    ins         t7, t8, 16, 16
    mult        $ac0, zero, zero
    dpa.w.ph    $ac0, t5, s0
    dpa.w.ph    $ac0, t7, s1
    mult        $ac1, zero, zero
    dpa.w.ph    $ac1, t5, s2
    dpa.w.ph    $ac1, t7, s3
    sll         t2, t2, 14      /* tmp0 =
                                     ((JLONG)wsptr[0]) << (CONST_BITS+1) */
    mflo        s6, $ac0
    /* MULTIPLY(wsptr[2], FIX_1_847759065) +
       MULTIPLY(wsptr[6], -FIX_0_765366865) */
    subu        s4, s4, s5
    addu        t3, t2, s4      /* tmp10 = tmp0 + z2; */
    mflo        s7, $ac1
    subu        t4, t2, s4      /* tmp10 = tmp0 - z2; */
    addu        t7, t4, s6
    subu        t8, t4, s6
    addu        t5, t3, s7
    subu        t6, t3, s7
    shra_r.w    t5, t5, 19      /* DESCALE(tmp10 + temp2, 19) */
    shra_r.w    t6, t6, 19      /* DESCALE(tmp10 - temp2, 19) */
    shra_r.w    t7, t7, 19      /* DESCALE(tmp12 + temp1, 19) */
    shra_r.w    t8, t8, 19      /* DESCALE(tmp12 - temp1, 19) */
    sll         s4, t9, 2
    lw          v0, 12(a2)      /* output_buf[ctr] */
    shll_s.w    t5, t5, 24
    shll_s.w    t6, t6, 24
    shll_s.w    t7, t7, 24
    shll_s.w    t8, t8, 24
    sra         t5, t5, 24
    sra         t6, t6, 24
    sra         t7, t7, 24
    sra         t8, t8, 24
    addu        v0, v0, a3      /* outptr = output_buf[ctr] + output_col */
    addiu       t5, t5, 128
    addiu       t6, t6, 128
    addiu       t7, t7, 128
    addiu       t8, t8, 128
    sb          t5, 0(v0)
    sb          t7, 1(v0)
    sb          t8, 2(v0)
    sb          t6, 3(v0)

    RESTORE_REGS_FROM_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    j           ra
     nop
END(jsimd_idct_4x4_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_idct_6x6_dspr2)
/*
 * a0 = compptr->dct_table
 * a1 = coef_block
 * a2 = output_buf
 * a3 = output_col
 */
    .set at

    SAVE_REGS_ON_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    addiu       sp, sp, -144
    move        v0, sp
    addiu       v1, v0, 24
    addiu       t9, zero, 5793
    addiu       s0, zero, 10033
    addiu       s1, zero, 2998

1:
    lh          s2, 0(a0)       /* q0 = quantptr[ 0] */
    lh          s3, 32(a0)      /* q1 = quantptr[16] */
    lh          s4, 64(a0)      /* q2 = quantptr[32] */
    lh          t2, 64(a1)      /* tmp2 = inptr[32] */
    lh          t1, 32(a1)      /* tmp1 = inptr[16] */
    lh          t0, 0(a1)       /* tmp0 = inptr[ 0] */
    mul         t2, t2, s4      /* tmp2 = tmp2 * q2 */
    mul         t1, t1, s3      /* tmp1 = tmp1 * q1 */
    mul         t0, t0, s2      /* tmp0 = tmp0 * q0 */
    lh          t6, 16(a1)      /* z1 = inptr[ 8] */
    lh          t8, 80(a1)      /* z3 = inptr[40] */
    lh          t7, 48(a1)      /* z2 = inptr[24] */
    lh          s2, 16(a0)      /* q0 = quantptr[ 8] */
    lh          s4, 80(a0)      /* q2 = quantptr[40] */
    lh          s3, 48(a0)      /* q1 = quantptr[24] */
    mul         t2, t2, t9      /* tmp2 = tmp2 * 5793 */
    mul         t1, t1, s0      /* tmp1 = tmp1 * 10033 */
    sll         t0, t0, 13      /* tmp0 = tmp0 << 13 */
    mul         t6, t6, s2      /* z1 = z1 * q0 */
    mul         t8, t8, s4      /* z3 = z3 * q2 */
    mul         t7, t7, s3      /* z2 = z2 * q1 */
    addu        t3, t0, t2      /* tmp10 = tmp0 + tmp2 */
    sll         t2, t2, 1       /* tmp2 = tmp2 << 2 */
    subu        t4, t0, t2      /* tmp11 = tmp0 - tmp2; */
    subu        t5, t3, t1      /* tmp12 = tmp10 - tmp1 */
    addu        t3, t3, t1      /* tmp10 = tmp10 + tmp1 */
    addu        t1, t6, t8      /* tmp1 = z1 + z3 */
    mul         t1, t1, s1      /* tmp1 = tmp1 * 2998 */
    shra_r.w    t4, t4, 11      /* tmp11 = (tmp11 + 1024) >> 11 */
    subu        t2, t6, t8      /* tmp2 = z1 - z3 */
    subu        t2, t2, t7      /* tmp2 = tmp2 - z2 */
    sll         t2, t2, 2       /* tmp2 = tmp2 << 2 */
    addu        t0, t6, t7      /* tmp0 = z1 + z2 */
    sll         t0, t0, 13      /* tmp0 = tmp0 << 13 */
    subu        s2, t8, t7      /* q0 = z3 - z2 */
    sll         s2, s2, 13      /* q0 = q0 << 13 */
    addu        t0, t0, t1      /* tmp0 = tmp0 + tmp1 */
    addu        t1, s2, t1      /* tmp1 = q0 + tmp1 */
    addu        s2, t4, t2      /* q0 = tmp11 + tmp2 */
    subu        s3, t4, t2      /* q1 = tmp11 - tmp2 */
    addu        t6, t3, t0      /* z1 = tmp10 + tmp0 */
    subu        t7, t3, t0      /* z2 = tmp10 - tmp0 */
    addu        t4, t5, t1      /* tmp11 = tmp12 + tmp1 */
    subu        t5, t5, t1      /* tmp12 = tmp12 - tmp1 */
    shra_r.w    t6, t6, 11      /* z1 = (z1 + 1024) >> 11 */
    shra_r.w    t7, t7, 11      /* z2 = (z2 + 1024) >> 11 */
    shra_r.w    t4, t4, 11      /* tmp11 = (tmp11 + 1024) >> 11 */
    shra_r.w    t5, t5, 11      /* tmp12 = (tmp12 + 1024) >> 11 */
    sw          s2, 24(v0)
    sw          s3, 96(v0)
    sw          t6, 0(v0)
    sw          t7, 120(v0)
    sw          t4, 48(v0)
    sw          t5, 72(v0)
    addiu       v0, v0, 4
    addiu       a1, a1, 2
    bne         v0, v1, 1b
     addiu      a0, a0, 2

    /* Pass 2: process 6 rows from work array, store into output array. */
    move        v0, sp
    addiu       v1, v0, 144

2:
    lw          t0, 0(v0)
    lw          t2, 16(v0)
    lw          s5, 0(a2)
    addiu       t0, t0, 16
    sll         t0, t0, 13
    mul         t3, t2, t9
    lw          t6, 4(v0)
    lw          t8, 20(v0)
    lw          t7, 12(v0)
    addu        s5, s5, a3
    addu        s6, t6, t8
    mul         s6, s6, s1
    addu        t1, t0, t3
    subu        t4, t0, t3
    subu        t4, t4, t3
    lw          t3, 8(v0)
    mul         t0, t3, s0
    addu        s7, t6, t7
    sll         s7, s7, 13
    addu        s7, s6, s7
    subu        t2, t8, t7
    sll         t2, t2, 13
    addu        t2, s6, t2
    subu        s6, t6, t7
    subu        s6, s6, t8
    sll         s6, s6, 13
    addu        t3, t1, t0
    subu        t5, t1, t0
    addu        t6, t3, s7
    subu        t3, t3, s7
    addu        t7, t4, s6
    subu        t4, t4, s6
    addu        t8, t5, t2
    subu        t5, t5, t2
    shll_s.w    t6, t6, 6
    shll_s.w    t3, t3, 6
    shll_s.w    t7, t7, 6
    shll_s.w    t4, t4, 6
    shll_s.w    t8, t8, 6
    shll_s.w    t5, t5, 6
    sra         t6, t6, 24
    addiu       t6, t6, 128
    sra         t3, t3, 24
    addiu       t3, t3, 128
    sb          t6, 0(s5)
    sra         t7, t7, 24
    addiu       t7, t7, 128
    sb          t3, 5(s5)
    sra         t4, t4, 24
    addiu       t4, t4, 128
    sb          t7, 1(s5)
    sra         t8, t8, 24
    addiu       t8, t8, 128
    sb          t4, 4(s5)
    addiu       v0, v0, 24
    sra         t5, t5, 24
    addiu       t5, t5, 128
    sb          t8, 2(s5)
    addiu       a2, a2,  4
    bne         v0, v1, 2b
     sb         t5, 3(s5)

    addiu       sp, sp, 144

    RESTORE_REGS_FROM_STACK 32, s0, s1, s2, s3, s4, s5, s6, s7

    j           ra
     nop

END(jsimd_idct_6x6_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_idct_12x12_pass1_dspr2)
/*
 * a0 = compptr->dct_table
 * a1 = coef_block
 * a2 = workspace
 */
    SAVE_REGS_ON_STACK 16, s0, s1, s2, s3

    li          a3, 8

1:
    /* odd part */
    lh          t0, 48(a1)
    lh          t1, 48(a0)
    lh          t2, 16(a1)
    lh          t3, 16(a0)
    lh          t4, 80(a1)
    lh          t5, 80(a0)
    lh          t6, 112(a1)
    lh          t7, 112(a0)
    mul         t0, t0, t1      /* z2 */
    mul         t1, t2, t3      /* z1 */
    mul         t2, t4, t5      /* z3 */
    mul         t3, t6, t7      /* z4 */
    li          t4, 10703       /* FIX(1.306562965) */
    li          t5, 4433        /* FIX_0_541196100 */
    li          t6, 7053        /* FIX(0.860918669) */
    mul         t4, t0, t4      /* tmp11 */
    mul         t5, t0, t5      /* -tmp14 */
    addu        t7, t1, t2      /* tmp10 */
    addu        t8, t7, t3      /* tmp10 + z4 */
    mul         t6, t6, t8      /* tmp15 */
    li          t8, 2139        /* FIX(0.261052384) */
    mul         t8, t7, t8      /* MULTIPLY(tmp10, FIX(0.261052384)) */
    li          t7, 2295        /* FIX(0.280143716) */
    mul         t7, t1, t7      /* MULTIPLY(z1, FIX(0.280143716)) */
    addu        t9, t2, t3      /* z3 + z4 */
    li          s0, 8565        /* FIX(1.045510580) */
    mul         t9, t9, s0      /* -tmp13 */
    li          s0, 12112       /* FIX(1.478575242) */
    mul         s0, t2, s0      /* MULTIPLY(z3, FIX(1.478575242) */
    li          s1, 12998       /* FIX(1.586706681) */
    mul         s1, t3, s1      /* MULTIPLY(z4, FIX(1.586706681)) */
    li          s2, 5540        /* FIX(0.676326758) */
    mul         s2, t1, s2      /* MULTIPLY(z1, FIX(0.676326758)) */
    li          s3, 16244       /* FIX(1.982889723) */
    mul         s3, t3, s3      /* MULTIPLY(z4, FIX(1.982889723)) */
    subu        t1, t1, t3      /* z1-=z4 */
    subu        t0, t0, t2      /* z2-=z3 */
    addu        t2, t0, t1      /* z1+z2 */
    li          t3, 4433        /* FIX_0_541196100 */
    mul         t2, t2, t3      /* z3 */
    li          t3, 6270        /* FIX_0_765366865 */
    mul         t1, t1, t3      /* MULTIPLY(z1, FIX_0_765366865) */
    li          t3, 15137       /* FIX_0_765366865 */
    mul         t0, t0, t3      /* MULTIPLY(z2, FIX_1_847759065) */
    addu        t8, t6, t8      /* tmp12 */
    addu        t3, t8, t4      /* tmp12 + tmp11 */
    addu        t3, t3, t7      /* tmp10 */
    subu        t8, t8, t9      /* tmp12 + tmp13 */
    addu        s0, t5, s0
    subu        t8, t8, s0      /* tmp12 */
    subu        t9, t6, t9
    subu        s1, s1, t4
    addu        t9, t9, s1      /* tmp13 */
    subu        t6, t6, t5
    subu        t6, t6, s2
    subu        t6, t6, s3      /* tmp15 */
    /* even part start */
    lh          t4, 64(a1)
    lh          t5, 64(a0)
    lh          t7, 32(a1)
    lh          s0, 32(a0)
    lh          s1, 0(a1)
    lh          s2, 0(a0)
    lh          s3, 96(a1)
    lh          v0, 96(a0)
    mul         t4, t4, t5      /* DEQUANTIZE(inptr[DCTSIZE*4],
                                              quantptr[DCTSIZE*4]) */
    mul         t5, t7, s0      /* DEQUANTIZE(inptr[DCTSIZE*2],
                                              quantptr[DCTSIZE*2]) */
    mul         t7, s1, s2      /* DEQUANTIZE(inptr[DCTSIZE*0],
                                              quantptr[DCTSIZE*0]) */
    mul         s0, s3, v0      /* DEQUANTIZE(inptr[DCTSIZE*6],
                                              quantptr[DCTSIZE*6]) */
    /* odd part end */
    addu        t1, t2, t1      /* tmp11 */
    subu        t0, t2, t0      /* tmp14 */
    /* update counter and pointers */
    addiu       a3, a3, -1
    addiu       a0, a0, 2
    addiu       a1, a1, 2
    /* even part rest */
    li          s1, 10033
    li          s2, 11190
    mul         t4, t4, s1      /* z4 */
    mul         s1, t5, s2      /* z4 */
    sll         t5, t5, 13      /* z1 */
    sll         t7, t7, 13
    addiu       t7, t7, 1024    /* z3 */
    sll         s0, s0, 13      /* z2 */
    addu        s2, t7, t4      /* tmp10 */
    subu        t4, t7, t4      /* tmp11 */
    subu        s3, t5, s0      /* tmp12 */
    addu        t2, t7, s3      /* tmp21 */
    subu        s3, t7, s3      /* tmp24 */
    addu        t7, s1, s0      /* tmp12 */
    addu        v0, s2, t7      /* tmp20 */
    subu        s2, s2, t7      /* tmp25 */
    subu        s1, s1, t5      /* z4 - z1 */
    subu        s1, s1, s0      /* tmp12 */
    addu        s0, t4, s1      /* tmp22 */
    subu        t4, t4, s1      /* tmp23 */
    /* final output stage */
    addu        t5, v0, t3
    subu        v0, v0, t3
    addu        t3, t2, t1
    subu        t2, t2, t1
    addu        t1, s0, t8
    subu        s0, s0, t8
    addu        t8, t4, t9
    subu        t4, t4, t9
    addu        t9, s3, t0
    subu        s3, s3, t0
    addu        t0, s2, t6
    subu        s2, s2, t6
    sra         t5, t5, 11
    sra         t3, t3, 11
    sra         t1, t1, 11
    sra         t8, t8, 11
    sra         t9, t9, 11
    sra         t0, t0, 11
    sra         s2, s2, 11
    sra         s3, s3, 11
    sra         t4, t4, 11
    sra         s0, s0, 11
    sra         t2, t2, 11
    sra         v0, v0, 11
    sw          t5, 0(a2)
    sw          t3, 32(a2)
    sw          t1, 64(a2)
    sw          t8, 96(a2)
    sw          t9, 128(a2)
    sw          t0, 160(a2)
    sw          s2, 192(a2)
    sw          s3, 224(a2)
    sw          t4, 256(a2)
    sw          s0, 288(a2)
    sw          t2, 320(a2)
    sw          v0, 352(a2)
    bgtz        a3, 1b
     addiu      a2, a2, 4

    RESTORE_REGS_FROM_STACK 16, s0, s1, s2, s3

    j           ra
     nop

END(jsimd_idct_12x12_pass1_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_idct_12x12_pass2_dspr2)
/*
 * a0 = workspace
 * a1 = output
 */
    SAVE_REGS_ON_STACK 16, s0, s1, s2, s3

    li          a3, 12

1:
    /* Odd part */
    lw          t0, 12(a0)
    lw          t1, 4(a0)
    lw          t2, 20(a0)
    lw          t3, 28(a0)
    li          t4, 10703       /* FIX(1.306562965) */
    li          t5, 4433        /* FIX_0_541196100 */
    mul         t4, t0, t4      /* tmp11 */
    mul         t5, t0, t5      /* -tmp14 */
    addu        t6, t1, t2      /* tmp10 */
    li          t7, 2139        /* FIX(0.261052384) */
    mul         t7, t6, t7      /* MULTIPLY(tmp10, FIX(0.261052384)) */
    addu        t6, t6, t3      /* tmp10 + z4 */
    li          t8, 7053        /* FIX(0.860918669) */
    mul         t6, t6, t8      /* tmp15 */
    li          t8, 2295        /* FIX(0.280143716) */
    mul         t8, t1, t8      /* MULTIPLY(z1, FIX(0.280143716)) */
    addu        t9, t2, t3      /* z3 + z4 */
    li          s0, 8565        /* FIX(1.045510580) */
    mul         t9, t9, s0      /* -tmp13 */
    li          s0, 12112       /* FIX(1.478575242) */
    mul         s0, t2, s0      /* MULTIPLY(z3, FIX(1.478575242)) */
    li          s1, 12998       /* FIX(1.586706681) */
    mul         s1, t3, s1      /* MULTIPLY(z4, FIX(1.586706681)) */
    li          s2, 5540        /* FIX(0.676326758) */
    mul         s2, t1, s2      /* MULTIPLY(z1, FIX(0.676326758)) */
    li          s3, 16244       /* FIX(1.982889723) */
    mul         s3, t3, s3      /* MULTIPLY(z4, FIX(1.982889723)) */
    subu        t1, t1, t3      /* z1 -= z4 */
    subu        t0, t0, t2      /* z2 -= z3 */
    addu        t2, t1, t0      /* z1 + z2 */
    li          t3, 4433        /* FIX_0_541196100 */
    mul         t2, t2, t3      /* z3 */
    li          t3, 6270        /* FIX_0_765366865 */
    mul         t1, t1, t3      /* MULTIPLY(z1, FIX_0_765366865) */
    li          t3, 15137       /* FIX_1_847759065 */
    mul         t0, t0, t3      /* MULTIPLY(z2, FIX_1_847759065) */
    addu        t3, t6, t7      /* tmp12 */
    addu        t7, t3, t4
    addu        t7, t7, t8      /* tmp10 */
    subu        t3, t3, t9
    subu        t3, t3, t5
    subu        t3, t3, s0      /* tmp12 */
    subu        t9, t6, t9
    subu        t9, t9, t4
    addu        t9, t9, s1      /* tmp13 */
    subu        t6, t6, t5
    subu        t6, t6, s2
    subu        t6, t6, s3      /* tmp15 */
    addu        t1, t2, t1      /* tmp11 */
    subu        t0, t2, t0      /* tmp14 */
    /* even part */
    lw          t2, 16(a0)      /* z4 */
    lw          t4, 8(a0)       /* z1 */
    lw          t5, 0(a0)       /* z3 */
    lw          t8, 24(a0)      /* z2 */
    li          s0, 10033       /* FIX(1.224744871) */
    li          s1, 11190       /* FIX(1.366025404) */
    mul         t2, t2, s0      /* z4 */
    mul         s0, t4, s1      /* z4 */
    addiu       t5, t5, 0x10
    sll         t5, t5, 13      /* z3 */
    sll         t4, t4, 13      /* z1 */
    sll         t8, t8, 13      /* z2 */
    subu        s1, t4, t8      /* tmp12 */
    addu        s2, t5, t2      /* tmp10 */
    subu        t2, t5, t2      /* tmp11 */
    addu        s3, t5, s1      /* tmp21 */
    subu        s1, t5, s1      /* tmp24 */
    addu        t5, s0, t8      /* tmp12 */
    addu        v0, s2, t5      /* tmp20 */
    subu        t5, s2, t5      /* tmp25 */
    subu        t4, s0, t4
    subu        t4, t4, t8      /* tmp12 */
    addu        t8, t2, t4      /* tmp22 */
    subu        t2, t2, t4      /* tmp23 */
    /* increment counter and pointers */
    addiu       a3, a3, -1
    addiu       a0, a0, 32
    /* Final stage */
    addu        t4, v0, t7
    subu        v0, v0, t7
    addu        t7, s3, t1
    subu        s3, s3, t1
    addu        t1, t8, t3
    subu        t8, t8, t3
    addu        t3, t2, t9
    subu        t2, t2, t9
    addu        t9, s1, t0
    subu        s1, s1, t0
    addu        t0, t5, t6
    subu        t5, t5, t6
    sll         t4, t4, 4
    sll         t7, t7, 4
    sll         t1, t1, 4
    sll         t3, t3, 4
    sll         t9, t9, 4
    sll         t0, t0, 4
    sll         t5, t5, 4
    sll         s1, s1, 4
    sll         t2, t2, 4
    sll         t8, t8, 4
    sll         s3, s3, 4
    sll         v0, v0, 4
    shll_s.w    t4, t4, 2
    shll_s.w    t7, t7, 2
    shll_s.w    t1, t1, 2
    shll_s.w    t3, t3, 2
    shll_s.w    t9, t9, 2
    shll_s.w    t0, t0, 2
    shll_s.w    t5, t5, 2
    shll_s.w    s1, s1, 2
    shll_s.w    t2, t2, 2
    shll_s.w    t8, t8, 2
    shll_s.w    s3, s3, 2
    shll_s.w    v0, v0, 2
    srl         t4, t4, 24
    srl         t7, t7, 24
    srl         t1, t1, 24
    srl         t3, t3, 24
    srl         t9, t9, 24
    srl         t0, t0, 24
    srl         t5, t5, 24
    srl         s1, s1, 24
    srl         t2, t2, 24
    srl         t8, t8, 24
    srl         s3, s3, 24
    srl         v0, v0, 24
    lw          t6, 0(a1)
    addiu       t4, t4, 0x80
    addiu       t7, t7, 0x80
    addiu       t1, t1, 0x80
    addiu       t3, t3, 0x80
    addiu       t9, t9, 0x80
    addiu       t0, t0, 0x80
    addiu       t5, t5, 0x80
    addiu       s1, s1, 0x80
    addiu       t2, t2, 0x80
    addiu       t8, t8, 0x80
    addiu       s3, s3, 0x80
    addiu       v0, v0, 0x80
    sb          t4, 0(t6)
    sb          t7, 1(t6)
    sb          t1, 2(t6)
    sb          t3, 3(t6)
    sb          t9, 4(t6)
    sb          t0, 5(t6)
    sb          t5, 6(t6)
    sb          s1, 7(t6)
    sb          t2, 8(t6)
    sb          t8, 9(t6)
    sb          s3, 10(t6)
    sb          v0, 11(t6)
    bgtz        a3, 1b
     addiu      a1, a1, 4

    RESTORE_REGS_FROM_STACK 16, s0, s1, s2, s3

    jr          ra
     nop

END(jsimd_idct_12x12_pass2_dspr2)


/*****************************************************************************/
LEAF_DSPR2(jsimd_convsamp_dspr2)
/*
 * a0 = sample_data
 * a1 = start_col
 * a2 = workspace
 */
    lw            t0, 0(a0)
    li            t7, 0xff80ff80
    addu          t0, t0, a1
    ulw           t1, 0(t0)
    ulw           t2, 4(t0)
    preceu.ph.qbr t3, t1
    preceu.ph.qbl t4, t1
    lw            t0, 4(a0)
    preceu.ph.qbr t5, t2
    preceu.ph.qbl t6, t2
    addu          t0, t0, a1
    addu.ph       t3, t3, t7
    addu.ph       t4, t4, t7
    ulw           t1, 0(t0)
    ulw           t2, 4(t0)
    addu.ph       t5, t5, t7
    addu.ph       t6, t6, t7
    usw           t3, 0(a2)
    usw           t4, 4(a2)
    preceu.ph.qbr t3, t1
    preceu.ph.qbl t4, t1
    usw           t5, 8(a2)
    usw           t6, 12(a2)

    lw            t0, 8(a0)
    preceu.ph.qbr t5, t2
    preceu.ph.qbl t6, t2
    addu          t0, t0, a1
    addu.ph       t3, t3, t7
    addu.ph       t4, t4, t7
    ulw           t1, 0(t0)
    ulw           t2, 4(t0)
    addu.ph       t5, t5, t7
    addu.ph       t6, t6, t7
    usw           t3, 16(a2)
    usw           t4, 20(a2)
    preceu.ph.qbr t3, t1
    preceu.ph.qbl t4, t1
    usw           t5, 24(a2)
    usw           t6, 28(a2)

    lw            t0, 12(a0)
    preceu.ph.qbr t5, t2
    preceu.ph.qbl t6, t2
    addu          t0, t0, a1
    addu.ph       t3, t3, t7
    addu.ph       t4, t4, t7
    ulw           t1, 0(t0)
    ulw           t2, 4(t0)
    addu.ph       t5, t5, t7
    addu.ph       t6, t6, t7
    usw           t3, 32(a2)
    usw           t4, 36(a2)
    preceu.ph.qbr t3, t1
    preceu.ph.qbl t4, t1
    usw           t5, 40(a2)
    usw           t6, 44(a2)

    lw            t0, 16(a0)
    preceu.ph.qbr t5, t2
    preceu.ph.qbl t6, t2
    addu          t0, t0, a1
    addu.ph       t3, t3, t7
    addu.ph       t4, t4, t7
    ulw           t1, 0(t0)
    ulw           t2, 4(t0)
    addu.ph       t5, t5, t7
    addu.ph       t6, t6, t7
    usw           t3, 48(a2)
    usw           t4, 52(a2)
    preceu.ph.qbr t3, t1
    preceu.ph.qbl t4, t1
    usw           t5, 56(a2)
    usw           t6, 60(a2)

    lw            t0, 20(a0)
    preceu.ph.qbr t5, t2
    preceu.ph.qbl t6, t2
    addu          t0, t0, a1
    addu.ph       t3, t3, t7
    addu.ph       t4, t4, t7
    ulw           t1, 0(t0)
    ulw           t2, 4(t0)
    addu.ph       t5, t5, t7
    addu.ph       t6, t6, t7
    usw           t3, 64(a2)
    usw           t4, 68(a2)
    preceu.ph.qbr t3, t1
    preceu.ph.qbl t4, t1
    usw           t5, 72(a2)
    usw           t6, 76(a2)

    lw            t0, 24(a0)
    preceu.ph.qbr t5, t2
    preceu.ph.qbl t6, t2
    addu          t0, t0, a1
    addu.ph       t3, t3, t7
    addu.ph       t4, t4, t7
    ulw           t1, 0(t0)
    ulw           t2, 4(t0)
    addu.ph       t5, t5, t7
    addu.ph       t6, t6, t7
    usw           t3, 80(a2)
    usw           t4, 84(a2)
    preceu.ph.qbr t3, t1
    preceu.ph.qbl t4, t1
    usw           t5, 88(a2)
    usw           t6, 92(a2)

    lw            t0, 28(a0)
    preceu.ph.qbr t5, t2
    preceu.ph.qbl t6, t2
    addu          t0, t0, a1
    addu.ph       t3, t3, t7
    addu.ph       t4, t4, t7
    ulw           t1, 0(t0)
    ulw           t2, 4(t0)
    addu.ph       t5, t5, t7
    addu.ph       t6, t6, t7
    usw           t3, 96(a2)
    usw           t4, 100(a2)
    preceu.ph.qbr t3, t1
    preceu.ph.qbl t4, t1
    usw           t5, 104(a2)
    usw           t6, 108(a2)
    preceu.ph.qbr t5, t2
    preceu.ph.qbl t6, t2
    addu.ph       t3, t3, t7
    addu.ph       t4, t4, t7
    addu.ph       t5, t5, t7
    addu.ph       t6, t6, t7
    usw           t3, 112(a2)
    usw           t4, 116(a2)
    usw           t5, 120(a2)
    usw           t6, 124(a2)

    j             ra
     nop

END(jsimd_convsamp_dspr2)


#ifndef __mips_soft_float

/*****************************************************************************/
LEAF_DSPR2(jsimd_convsamp_float_dspr2)
/*
 * a0 = sample_data
 * a1 = start_col
 * a2 = workspace
 */
    .set at

    lw          t0, 0(a0)
    addu        t0, t0, a1
    lbu         t1, 0(t0)
    lbu         t2, 1(t0)
    lbu         t3, 2(t0)
    lbu         t4, 3(t0)
    lbu         t5, 4(t0)
    lbu         t6, 5(t0)
    lbu         t7, 6(t0)
    lbu         t8, 7(t0)
    addiu       t1, t1, -128
    addiu       t2, t2, -128
    addiu       t3, t3, -128
    addiu       t4, t4, -128
    addiu       t5, t5, -128
    addiu       t6, t6, -128
    addiu       t7, t7, -128
    addiu       t8, t8, -128
    mtc1        t1, f2
    mtc1        t2, f4
    mtc1        t3, f6
    mtc1        t4, f8
    mtc1        t5, f10
    mtc1        t6, f12
    mtc1        t7, f14
    mtc1        t8, f16
    cvt.s.w     f2, f2
    cvt.s.w     f4, f4
    cvt.s.w     f6, f6
    cvt.s.w     f8, f8
    cvt.s.w     f10, f10
    cvt.s.w     f12, f12
    cvt.s.w     f14, f14
    cvt.s.w     f16, f16
    lw          t0, 4(a0)
    swc1        f2, 0(a2)
    swc1        f4, 4(a2)
    swc1        f6, 8(a2)
    addu        t0, t0, a1
    swc1        f8, 12(a2)
    swc1        f10, 16(a2)
    swc1        f12, 20(a2)
    swc1        f14, 24(a2)
    swc1        f16, 28(a2)
    /* elemr 1 */
    lbu         t1, 0(t0)
    lbu         t2, 1(t0)
    lbu         t3, 2(t0)
    lbu         t4, 3(t0)
    lbu         t5, 4(t0)
    lbu         t6, 5(t0)
    lbu         t7, 6(t0)
    lbu         t8, 7(t0)
    addiu       t1, t1, -128
    addiu       t2, t2, -128
    addiu       t3, t3, -128
    addiu       t4, t4, -128
    addiu       t5, t5, -128
    addiu       t6, t6, -128
    addiu       t7, t7, -128
    addiu       t8, t8, -128
    mtc1        t1, f2
    mtc1        t2, f4
    mtc1        t3, f6
    mtc1        t4, f8
    mtc1        t5, f10
    mtc1        t6, f12
    mtc1        t7, f14
    mtc1        t8, f16
    cvt.s.w     f2, f2
    cvt.s.w     f4, f4
    cvt.s.w     f6, f6
    cvt.s.w     f8, f8
    cvt.s.w     f10, f10
    cvt.s.w     f12, f12
    cvt.s.w     f14, f14
    cvt.s.w     f16, f16
    lw          t0, 8(a0)
    swc1        f2, 32(a2)
    swc1        f4, 36(a2)
    swc1        f6, 40(a2)
    addu        t0, t0, a1
    swc1        f8, 44(a2)
    swc1        f10, 48(a2)
    swc1        f12, 52(a2)
    swc1        f14, 56(a2)
    swc1        f16, 60(a2)
    /* elemr 2 */
    lbu         t1, 0(t0)
    lbu         t2, 1(t0)
    lbu         t3, 2(t0)
    lbu         t4, 3(t0)
    lbu         t5, 4(t0)
    lbu         t6, 5(t0)
    lbu         t7, 6(t0)
    lbu         t8, 7(t0)
    addiu       t1, t1, -128
    addiu       t2, t2, -128
    addiu       t3, t3, -128
    addiu       t4, t4, -128
    addiu       t5, t5, -128
    addiu       t6, t6, -128
    addiu       t7, t7, -128
    addiu       t8, t8, -128
    mtc1        t1, f2
    mtc1        t2, f4
    mtc1        t3, f6
    mtc1        t4, f8
    mtc1        t5, f10
    mtc1        t6, f12
    mtc1        t7, f14
    mtc1        t8, f16
    cvt.s.w     f2, f2
    cvt.s.w     f4, f4
    cvt.s.w     f6, f6
    cvt.s.w     f8, f8
    cvt.s.w     f10, f10
    cvt.s.w     f12, f12
    cvt.s.w     f14, f14
    cvt.s.w     f16, f16
    lw          t0, 12(a0)
    swc1        f2, 64(a2)
    swc1        f4, 68(a2)
    swc1        f6, 72(a2)
    addu        t0, t0, a1
    swc1        f8, 76(a2)
    swc1        f10, 80(a2)
    swc1        f12, 84(a2)
    swc1        f14, 88(a2)
    swc1        f16, 92(a2)
    /*  elemr 3 */
    lbu         t1, 0(t0)
    lbu         t2, 1(t0)
    lbu         t3, 2(t0)
    lbu         t4, 3(t0)
    lbu         t5, 4(t0)
    lbu         t6, 5(t0)
    lbu         t7, 6(t0)
    lbu         t8, 7(t0)
    addiu       t1, t1, -128
    addiu       t2, t2, -128
    addiu       t3, t3, -128
    addiu       t4, t4, -128
    addiu       t5, t5, -128
    addiu       t6, t6, -128
    addiu       t7, t7, -128
    addiu       t8, t8, -128
    mtc1        t1, f2
    mtc1        t2, f4
    mtc1        t3, f6
    mtc1        t4, f8
    mtc1        t5, f10
    mtc1        t6, f12
    mtc1        t7, f14
    mtc1        t8, f16
    cvt.s.w     f2, f2
    cvt.s.w     f4, f4
    cvt.s.w     f6, f6
    cvt.s.w     f8, f8
    cvt.s.w     f10, f10
    cvt.s.w     f12, f12
    cvt.s.w     f14, f14
    cvt.s.w     f16, f16
    lw          t0, 16(a0)
    swc1        f2, 96(a2)
    swc1        f4, 100(a2)
    swc1        f6, 104(a2)
    addu        t0, t0, a1
    swc1        f8, 108(a2)
    swc1        f10, 112(a2)
    swc1        f12, 116(a2)
    swc1        f14, 120(a2)
    swc1        f16, 124(a2)
    /* elemr 4 */
    lbu         t1, 0(t0)
    lbu         t2, 1(t0)
    lbu         t3, 2(t0)
    lbu         t4, 3(t0)
    lbu         t5, 4(t0)
    lbu         t6, 5(t0)
    lbu         t7, 6(t0)
    lbu         t8, 7(t0)
    addiu       t1, t1, -128
    addiu       t2, t2, -128
    addiu       t3, t3, -128
    addiu       t4, t4, -128
    addiu       t5, t5, -128
    addiu       t6, t6, -128
    addiu       t7, t7, -128
    addiu       t8, t8, -128
    mtc1        t1, f2
    mtc1        t2, f4
    mtc1        t3, f6
    mtc1        t4, f8
    mtc1        t5, f10
    mtc1        t6, f12
    mtc1        t7, f14
    mtc1        t8, f16
    cvt.s.w     f2, f2
    cvt.s.w     f4, f4
    cvt.s.w     f6, f6
    cvt.s.w     f8, f8
    cvt.s.w     f10, f10
    cvt.s.w     f12, f12
    cvt.s.w     f14, f14
    cvt.s.w     f16, f16
    lw          t0, 20(a0)
    swc1        f2, 128(a2)
    swc1        f4, 132(a2)
    swc1        f6, 136(a2)
    addu        t0, t0, a1
    swc1        f8, 140(a2)
    swc1        f10, 144(a2)
    swc1        f12, 148(a2)
    swc1        f14, 152(a2)
    swc1        f16, 156(a2)
    /* elemr 5 */
    lbu         t1, 0(t0)
    lbu         t2, 1(t0)
    lbu         t3, 2(t0)
    lbu         t4, 3(t0)
    lbu         t5, 4(t0)
    lbu         t6, 5(t0)
    lbu         t7, 6(t0)
    lbu         t8, 7(t0)
    addiu       t1, t1, -128
    addiu       t2, t2, -128
    addiu       t3, t3, -128
    addiu       t4, t4, -128
    addiu       t5, t5, -128
    addiu       t6, t6, -128
    addiu       t7, t7, -128
    addiu       t8, t8, -128
    mtc1        t1, f2
    mtc1        t2, f4
    mtc1        t3, f6
    mtc1        t4, f8
    mtc1        t5, f10
    mtc1        t6, f12
    mtc1        t7, f14
    mtc1        t8, f16
    cvt.s.w     f2, f2
    cvt.s.w     f4, f4
    cvt.s.w     f6, f6
    cvt.s.w     f8, f8
    cvt.s.w     f10, f10
    cvt.s.w     f12, f12
    cvt.s.w     f14, f14
    cvt.s.w     f16, f16
    lw          t0, 24(a0)
    swc1        f2, 160(a2)
    swc1        f4, 164(a2)
    swc1        f6, 168(a2)
    addu        t0, t0, a1
    swc1        f8, 172(a2)
    swc1        f10, 176(a2)
    swc1        f12, 180(a2)
    swc1        f14, 184(a2)
    swc1        f16, 188(a2)
    /* elemr 6 */
    lbu         t1, 0(t0)
    lbu         t2, 1(t0)
    lbu         t3, 2(t0)
    lbu         t4, 3(t0)
    lbu         t5, 4(t0)
    lbu         t6, 5(t0)
    lbu         t7, 6(t0)
    lbu         t8, 7(t0)
    addiu       t1, t1, -128
    addiu       t2, t2, -128
    addiu       t3, t3, -128
    addiu       t4, t4, -128
    addiu       t5, t5, -128
    addiu       t6, t6, -128
    addiu       t7, t7, -128
    addiu       t8, t8, -128
    mtc1        t1, f2
    mtc1        t2, f4
    mtc1        t3, f6
    mtc1        t4, f8
    mtc1        t5, f10
    mtc1        t6, f12
    mtc1        t7, f14
    mtc1        t8, f16
    cvt.s.w     f2, f2
    cvt.s.w     f4, f4
    cvt.s.w     f6, f6
    cvt.s.w     f8, f8
    cvt.s.w     f10, f10
    cvt.s.w     f12, f12
    cvt.s.w     f14, f14
    cvt.s.w     f16, f16
    lw          t0, 28(a0)
    swc1        f2, 192(a2)
    swc1        f4, 196(a2)
    swc1        f6, 200(a2)
    addu        t0, t0, a1
    swc1        f8, 204(a2)
    swc1        f10, 208(a2)
    swc1        f12, 212(a2)
    swc1        f14, 216(a2)
    swc1        f16, 220(a2)
    /* elemr 7 */
    lbu         t1, 0(t0)
    lbu         t2, 1(t0)
    lbu         t3, 2(t0)
    lbu         t4, 3(t0)
    lbu         t5, 4(t0)
    lbu         t6, 5(t0)
    lbu         t7, 6(t0)
    lbu         t8, 7(t0)
    addiu       t1, t1, -128
    addiu       t2, t2, -128
    addiu       t3, t3, -128
    addiu       t4, t4, -128
    addiu       t5, t5, -128
    addiu       t6, t6, -128
    addiu       t7, t7, -128
    addiu       t8, t8, -128
    mtc1        t1, f2
    mtc1        t2, f4
    mtc1        t3, f6
    mtc1        t4, f8
    mtc1        t5, f10
    mtc1        t6, f12
    mtc1        t7, f14
    mtc1        t8, f16
    cvt.s.w     f2, f2
    cvt.s.w     f4, f4
    cvt.s.w     f6, f6
    cvt.s.w     f8, f8
    cvt.s.w     f10, f10
    cvt.s.w     f12, f12
    cvt.s.w     f14, f14
    cvt.s.w     f16, f16
    swc1        f2, 224(a2)
    swc1        f4, 228(a2)
    swc1        f6, 232(a2)
    swc1        f8, 236(a2)
    swc1        f10, 240(a2)
    swc1        f12, 244(a2)
    swc1        f14, 248(a2)
    swc1        f16, 252(a2)

    j           ra
     nop

END(jsimd_convsamp_float_dspr2)

#endif

/*****************************************************************************/
