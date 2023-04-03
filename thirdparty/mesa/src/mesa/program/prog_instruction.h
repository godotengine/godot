/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */


/**
 * \file prog_instruction.h
 *
 * Vertex/fragment program instruction datatypes and constants.
 *
 * \author Brian Paul
 * \author Keith Whitwell
 * \author Ian Romanick <idr@us.ibm.com>
 */


#ifndef PROG_INSTRUCTION_H
#define PROG_INSTRUCTION_H


#include "util/glheader.h"


/**
 * Swizzle indexes.
 * Do not change!
 */
/*@{*/
#define SWIZZLE_X    0
#define SWIZZLE_Y    1
#define SWIZZLE_Z    2
#define SWIZZLE_W    3
#define SWIZZLE_ZERO 4   /**< For SWZ instruction only */
#define SWIZZLE_ONE  5   /**< For SWZ instruction only */
#define SWIZZLE_NIL  7   /**< used during shader code gen (undefined value) */
/*@}*/

#define MAKE_SWIZZLE4(a,b,c,d) (((a)<<0) | ((b)<<3) | ((c)<<6) | ((d)<<9))
#define SWIZZLE_NOOP           MAKE_SWIZZLE4(0,1,2,3)
#define GET_SWZ(swz, idx)      (((swz) >> ((idx)*3)) & 0x7)
#define GET_BIT(msk, idx)      (((msk) >> (idx)) & 0x1)
/** Determine if swz contains SWIZZLE_ZERO/ONE/NIL for any components. */
#define HAS_EXTENDED_SWIZZLE(swz) (swz & 0x924)

#define SWIZZLE_XYZW MAKE_SWIZZLE4(SWIZZLE_X, SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_W)
#define SWIZZLE_XXXX MAKE_SWIZZLE4(SWIZZLE_X, SWIZZLE_X, SWIZZLE_X, SWIZZLE_X)
#define SWIZZLE_YYYY MAKE_SWIZZLE4(SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y, SWIZZLE_Y)
#define SWIZZLE_ZZZZ MAKE_SWIZZLE4(SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z, SWIZZLE_Z)
#define SWIZZLE_WWWW MAKE_SWIZZLE4(SWIZZLE_W, SWIZZLE_W, SWIZZLE_W, SWIZZLE_W)


/**
 * Writemask values, 1 bit per component.
 */
/*@{*/
#define WRITEMASK_X     0x1
#define WRITEMASK_Y     0x2
#define WRITEMASK_XY    0x3
#define WRITEMASK_Z     0x4
#define WRITEMASK_XZ    0x5
#define WRITEMASK_YZ    0x6
#define WRITEMASK_XYZ   0x7
#define WRITEMASK_W     0x8
#define WRITEMASK_XW    0x9
#define WRITEMASK_YW    0xa
#define WRITEMASK_XYW   0xb
#define WRITEMASK_ZW    0xc
#define WRITEMASK_XZW   0xd
#define WRITEMASK_YZW   0xe
#define WRITEMASK_XYZW  0xf
/*@}*/


/**
 * Per-component negation masks
 */
/*@{*/
#define NEGATE_X    0x1
#define NEGATE_Y    0x2
#define NEGATE_Z    0x4
#define NEGATE_W    0x8
#define NEGATE_XYZ  0x7
#define NEGATE_XYZW 0xf
#define NEGATE_NONE 0x0
/*@}*/


/**
 * Program instruction opcodes for vertex, fragment and geometry programs.
 */
enum prog_opcode {
                     /* ARB_vp   ARB_fp   NV_vp   NV_fp     GLSL */
                     /*------------------------------------------*/
   OPCODE_NOP = 0,   /*                                      X   */
   OPCODE_ABS,       /*   X        X       1.1               X   */
   OPCODE_ADD,       /*   X        X       X       X         X   */
   OPCODE_ARL,       /*   X                X                 X   */
   OPCODE_BGNLOOP,   /*                                     opt  */
   OPCODE_BGNSUB,    /*                                     opt  */
   OPCODE_BRK,       /*                    2                opt  */
   OPCODE_CAL,       /*                    2       2        opt  */
   OPCODE_CMP,       /*            X                         X   */
   OPCODE_CONT,      /*                                     opt  */
   OPCODE_COS,       /*            X       2       X         X   */
   OPCODE_DDX,       /*                            X         X   */
   OPCODE_DDY,       /*                            X         X   */
   OPCODE_DP2,       /*                            2         X   */
   OPCODE_DP3,       /*   X        X       X       X         X   */
   OPCODE_DP4,       /*   X        X       X       X         X   */
   OPCODE_DPH,       /*   X        X       1.1                   */
   OPCODE_DST,       /*   X        X       X       X             */
   OPCODE_ELSE,      /*                                     opt  */
   OPCODE_END,       /*   X        X       X       X        opt  */
   OPCODE_ENDIF,     /*                                     opt  */
   OPCODE_ENDLOOP,   /*                                     opt  */
   OPCODE_ENDSUB,    /*                                     opt  */
   OPCODE_EX2,       /*   X        X       2       X         X   */
   OPCODE_EXP,       /*   X                X                     */
   OPCODE_FLR,       /*   X        X       2       X         X   */
   OPCODE_FRC,       /*   X        X       2       X         X   */
   OPCODE_IF,        /*                                     opt  */
   OPCODE_KIL,       /*            X                         X   */
   OPCODE_LG2,       /*   X        X       2       X         X   */
   OPCODE_LIT,       /*   X        X       X       X             */
   OPCODE_LOG,       /*   X                X                     */
   OPCODE_LRP,       /*            X               X             */
   OPCODE_MAD,       /*   X        X       X       X         X   */
   OPCODE_MAX,       /*   X        X       X       X         X   */
   OPCODE_MIN,       /*   X        X       X       X         X   */
   OPCODE_MOV,       /*   X        X       X       X         X   */
   OPCODE_MUL,       /*   X        X       X       X         X   */
   OPCODE_NOISE1,    /*                                      X   */
   OPCODE_NOISE2,    /*                                      X   */
   OPCODE_NOISE3,    /*                                      X   */
   OPCODE_NOISE4,    /*                                      X   */
   OPCODE_POW,       /*   X        X               X         X   */
   OPCODE_RCP,       /*   X        X       X       X         X   */
   OPCODE_RET,       /*                    2       2        opt  */
   OPCODE_RSQ,       /*   X        X       X       X         X   */
   OPCODE_SCS,       /*            X                         X   */
   OPCODE_SGE,       /*   X        X       X       X         X   */
   OPCODE_SIN,       /*            X       2       X         X   */
   OPCODE_SLT,       /*   X        X       X       X         X   */
   OPCODE_SSG,       /*                    2                 X   */
   OPCODE_SUB,       /*   X        X       1.1     X         X   */
   OPCODE_SWZ,       /*   X        X                         X   */
   OPCODE_TEX,       /*            X       3       X         X   */
   OPCODE_TXB,       /*            X       3                 X   */
   OPCODE_TXD,       /*                            X         X   */
   OPCODE_TXL,       /*                    3       2         X   */
   OPCODE_TXP,       /*            X                         X   */
   OPCODE_TRUNC,     /*                                      X   */
   OPCODE_XPD,       /*   X        X                             */
   MAX_OPCODE
};


/**
 * Number of bits for the src/dst register Index field.
 * This limits the size of temp/uniform register files.
 */
#define INST_INDEX_BITS 12


/**
 * Instruction source register.
 */
struct prog_src_register
{
   GLuint File:4;	/**< One of the PROGRAM_* register file values. */
   GLint Index:(INST_INDEX_BITS+1); /**< Extra bit here for sign bit.
                                     * May be negative for relative addressing.
                                     */
   GLuint Swizzle:12;
   GLuint RelAddr:1;

   /**
    * Negation.
    * This will either be NEGATE_NONE or NEGATE_XYZW, except for the SWZ
    * instruction which allows per-component negation.
    */
   GLuint Negate:4;
};


/**
 * Instruction destination register.
 */
struct prog_dst_register
{
   GLuint File:4;      /**< One of the PROGRAM_* register file values */
   GLuint Index:INST_INDEX_BITS;  /**< Unsigned, never negative */
   GLuint WriteMask:4;
   GLuint RelAddr:1;
};


/**
 * Vertex/fragment program instruction.
 */
struct prog_instruction
{
   enum prog_opcode Opcode;
   struct prog_src_register SrcReg[3];
   struct prog_dst_register DstReg;

   /**
    * Saturate each value of the vectored result to the range [0,1].
    *
    * \since
    * ARB_fragment_program
    */
   GLuint Saturate:1;

   /**
    * \name Extra fields for TEX, TXB, TXD, TXL, TXP instructions.
    */
   /*@{*/
   /** Source texture unit. */
   GLuint TexSrcUnit:5;

   /** Source texture target, one of TEXTURE_{1D,2D,3D,CUBE,RECT}_INDEX */
   GLuint TexSrcTarget:4;

   /** True if tex instruction should do shadow comparison */
   GLuint TexShadow:1;
   /*@}*/

   /**
    * For BRA and CAL instructions, the location to jump to.
    * For BGNLOOP, points to ENDLOOP (and vice-versa).
    * For BRK, points to ENDLOOP
    * For IF, points to ELSE or ENDIF.
    * For ELSE, points to ENDIF.
    */
   GLint BranchTarget;
};


#ifdef __cplusplus
extern "C" {
#endif

struct gl_program;

extern void
_mesa_init_instructions(struct prog_instruction *inst, GLuint count);

extern struct prog_instruction *
_mesa_copy_instructions(struct prog_instruction *dest,
                        const struct prog_instruction *src, GLuint n);

extern GLuint
_mesa_num_inst_src_regs(enum prog_opcode opcode);

extern GLuint
_mesa_num_inst_dst_regs(enum prog_opcode opcode);

extern const char *
_mesa_opcode_string(enum prog_opcode opcode);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROG_INSTRUCTION_H */
