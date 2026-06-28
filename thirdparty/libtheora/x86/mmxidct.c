/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009,2025           *
 * by the Xiph.Org Foundation and contributors                      *
 * https://www.xiph.org/                                            *
 *                                                                  *
 ********************************************************************

  function:

 ********************************************************************/

/*MMX acceleration of Theora's iDCT.
  Originally written by Rudolf Marek, based on code from On2's VP3.*/
#include "x86int.h"
#include "../dct.h"

#if defined(OC_X86_ASM)

/*These are offsets into the table of constants below.*/
/*7 rows of cosines, in order: pi/16 * (1 ... 7).*/
#define OC_COSINE_OFFSET (0)
/*A row of 8's.*/
#define OC_EIGHT_OFFSET  (56)



/*38 cycles*/
#define OC_IDCT_BEGIN(_y,_x) \
  "#OC_IDCT_BEGIN\n\t" \
  "movq "OC_I(3,_x)",%%mm2\n\t" \
  "movq "OC_MEM_OFFS(0x30,c)",%%mm6\n\t" \
  "movq %%mm2,%%mm4\n\t" \
  "movq "OC_J(5,_x)",%%mm7\n\t" \
  "pmulhw %%mm6,%%mm4\n\t" \
  "movq "OC_MEM_OFFS(0x50,c)",%%mm1\n\t" \
  "pmulhw %%mm7,%%mm6\n\t" \
  "movq %%mm1,%%mm5\n\t" \
  "pmulhw %%mm2,%%mm1\n\t" \
  "movq "OC_I(1,_x)",%%mm3\n\t" \
  "pmulhw %%mm7,%%mm5\n\t" \
  "movq "OC_MEM_OFFS(0x10,c)",%%mm0\n\t" \
  "paddw %%mm2,%%mm4\n\t" \
  "paddw %%mm7,%%mm6\n\t" \
  "paddw %%mm1,%%mm2\n\t" \
  "movq "OC_J(7,_x)",%%mm1\n\t" \
  "paddw %%mm5,%%mm7\n\t" \
  "movq %%mm0,%%mm5\n\t" \
  "pmulhw %%mm3,%%mm0\n\t" \
  "paddw %%mm7,%%mm4\n\t" \
  "pmulhw %%mm1,%%mm5\n\t" \
  "movq "OC_MEM_OFFS(0x70,c)",%%mm7\n\t" \
  "psubw %%mm2,%%mm6\n\t" \
  "paddw %%mm3,%%mm0\n\t" \
  "pmulhw %%mm7,%%mm3\n\t" \
  "movq "OC_I(2,_x)",%%mm2\n\t" \
  "pmulhw %%mm1,%%mm7\n\t" \
  "paddw %%mm1,%%mm5\n\t" \
  "movq %%mm2,%%mm1\n\t" \
  "pmulhw "OC_MEM_OFFS(0x20,c)",%%mm2\n\t" \
  "psubw %%mm5,%%mm3\n\t" \
  "movq "OC_J(6,_x)",%%mm5\n\t" \
  "paddw %%mm7,%%mm0\n\t" \
  "movq %%mm5,%%mm7\n\t" \
  "psubw %%mm4,%%mm0\n\t" \
  "pmulhw "OC_MEM_OFFS(0x20,c)",%%mm5\n\t" \
  "paddw %%mm1,%%mm2\n\t" \
  "pmulhw "OC_MEM_OFFS(0x60,c)",%%mm1\n\t" \
  "paddw %%mm4,%%mm4\n\t" \
  "paddw %%mm0,%%mm4\n\t" \
  "psubw %%mm6,%%mm3\n\t" \
  "paddw %%mm7,%%mm5\n\t" \
  "paddw %%mm6,%%mm6\n\t" \
  "pmulhw "OC_MEM_OFFS(0x60,c)",%%mm7\n\t" \
  "paddw %%mm3,%%mm6\n\t" \
  "movq %%mm4,"OC_I(1,_y)"\n\t" \
  "psubw %%mm5,%%mm1\n\t" \
  "movq "OC_MEM_OFFS(0x40,c)",%%mm4\n\t" \
  "movq %%mm3,%%mm5\n\t" \
  "pmulhw %%mm4,%%mm3\n\t" \
  "paddw %%mm2,%%mm7\n\t" \
  "movq %%mm6,"OC_I(2,_y)"\n\t" \
  "movq %%mm0,%%mm2\n\t" \
  "movq "OC_I(0,_x)",%%mm6\n\t" \
  "pmulhw %%mm4,%%mm0\n\t" \
  "paddw %%mm3,%%mm5\n\t" \
  "movq "OC_J(4,_x)",%%mm3\n\t" \
  "psubw %%mm1,%%mm5\n\t" \
  "paddw %%mm0,%%mm2\n\t" \
  "psubw %%mm3,%%mm6\n\t" \
  "movq %%mm6,%%mm0\n\t" \
  "pmulhw %%mm4,%%mm6\n\t" \
  "paddw %%mm3,%%mm3\n\t" \
  "paddw %%mm1,%%mm1\n\t" \
  "paddw %%mm0,%%mm3\n\t" \
  "paddw %%mm5,%%mm1\n\t" \
  "pmulhw %%mm3,%%mm4\n\t" \
  "paddw %%mm0,%%mm6\n\t" \
  "psubw %%mm2,%%mm6\n\t" \
  "paddw %%mm2,%%mm2\n\t" \
  "movq "OC_I(1,_y)",%%mm0\n\t" \
  "paddw %%mm6,%%mm2\n\t" \
  "paddw %%mm3,%%mm4\n\t" \
  "psubw %%mm1,%%mm2\n\t" \
  "#end OC_IDCT_BEGIN\n\t" \

/*38+8=46 cycles.*/
#define OC_ROW_IDCT(_y,_x) \
  "#OC_ROW_IDCT\n" \
  OC_IDCT_BEGIN(_y,_x) \
  /*r3=D'*/ \
  "movq "OC_I(2,_y)",%%mm3\n\t" \
  /*r4=E'=E-G*/ \
  "psubw %%mm7,%%mm4\n\t" \
  /*r1=H'+H'*/ \
  "paddw %%mm1,%%mm1\n\t" \
  /*r7=G+G*/ \
  "paddw %%mm7,%%mm7\n\t" \
  /*r1=R1=A''+H'*/ \
  "paddw %%mm2,%%mm1\n\t" \
  /*r7=G'=E+G*/ \
  "paddw %%mm4,%%mm7\n\t" \
  /*r4=R4=E'-D'*/ \
  "psubw %%mm3,%%mm4\n\t" \
  "paddw %%mm3,%%mm3\n\t" \
  /*r6=R6=F'-B''*/ \
  "psubw %%mm5,%%mm6\n\t" \
  "paddw %%mm5,%%mm5\n\t" \
  /*r3=R3=E'+D'*/ \
  "paddw %%mm4,%%mm3\n\t" \
  /*r5=R5=F'+B''*/ \
  "paddw %%mm6,%%mm5\n\t" \
  /*r7=R7=G'-C'*/ \
  "psubw %%mm0,%%mm7\n\t" \
  "paddw %%mm0,%%mm0\n\t" \
  /*Save R1.*/ \
  "movq %%mm1,"OC_I(1,_y)"\n\t" \
  /*r0=R0=G.+C.*/ \
  "paddw %%mm7,%%mm0\n\t" \
  "#end OC_ROW_IDCT\n\t" \

/*The following macro does two 4x4 transposes in place.
  At entry, we assume:
    r0 = a3 a2 a1 a0
  I(1) = b3 b2 b1 b0
    r2 = c3 c2 c1 c0
    r3 = d3 d2 d1 d0

    r4 = e3 e2 e1 e0
    r5 = f3 f2 f1 f0
    r6 = g3 g2 g1 g0
    r7 = h3 h2 h1 h0

  At exit, we have:
  I(0) = d0 c0 b0 a0
  I(1) = d1 c1 b1 a1
  I(2) = d2 c2 b2 a2
  I(3) = d3 c3 b3 a3

  J(4) = h0 g0 f0 e0
  J(5) = h1 g1 f1 e1
  J(6) = h2 g2 f2 e2
  J(7) = h3 g3 f3 e3

  I(0) I(1) I(2) I(3) is the transpose of r0 I(1) r2 r3.
  J(4) J(5) J(6) J(7) is the transpose of r4  r5  r6 r7.

  Since r1 is free at entry, we calculate the Js first.*/
/*19 cycles.*/
#define OC_TRANSPOSE(_y) \
  "#OC_TRANSPOSE\n\t" \
  "movq %%mm4,%%mm1\n\t" \
  "punpcklwd %%mm5,%%mm4\n\t" \
  "movq %%mm0,"OC_I(0,_y)"\n\t" \
  "punpckhwd %%mm5,%%mm1\n\t" \
  "movq %%mm6,%%mm0\n\t" \
  "punpcklwd %%mm7,%%mm6\n\t" \
  "movq %%mm4,%%mm5\n\t" \
  "punpckldq %%mm6,%%mm4\n\t" \
  "punpckhdq %%mm6,%%mm5\n\t" \
  "movq %%mm1,%%mm6\n\t" \
  "movq %%mm4,"OC_J(4,_y)"\n\t" \
  "punpckhwd %%mm7,%%mm0\n\t" \
  "movq %%mm5,"OC_J(5,_y)"\n\t" \
  "punpckhdq %%mm0,%%mm6\n\t" \
  "movq "OC_I(0,_y)",%%mm4\n\t" \
  "punpckldq %%mm0,%%mm1\n\t" \
  "movq "OC_I(1,_y)",%%mm5\n\t" \
  "movq %%mm4,%%mm0\n\t" \
  "movq %%mm6,"OC_J(7,_y)"\n\t" \
  "punpcklwd %%mm5,%%mm0\n\t" \
  "movq %%mm1,"OC_J(6,_y)"\n\t" \
  "punpckhwd %%mm5,%%mm4\n\t" \
  "movq %%mm2,%%mm5\n\t" \
  "punpcklwd %%mm3,%%mm2\n\t" \
  "movq %%mm0,%%mm1\n\t" \
  "punpckldq %%mm2,%%mm0\n\t" \
  "punpckhdq %%mm2,%%mm1\n\t" \
  "movq %%mm4,%%mm2\n\t" \
  "movq %%mm0,"OC_I(0,_y)"\n\t" \
  "punpckhwd %%mm3,%%mm5\n\t" \
  "movq %%mm1,"OC_I(1,_y)"\n\t" \
  "punpckhdq %%mm5,%%mm4\n\t" \
  "punpckldq %%mm5,%%mm2\n\t" \
  "movq %%mm4,"OC_I(3,_y)"\n\t" \
  "movq %%mm2,"OC_I(2,_y)"\n\t" \
  "#end OC_TRANSPOSE\n\t" \

/*38+19=57 cycles.*/
#define OC_COLUMN_IDCT(_y) \
  "#OC_COLUMN_IDCT\n" \
  OC_IDCT_BEGIN(_y,_y) \
  "paddw "OC_MEM_OFFS(0x00,c)",%%mm2\n\t" \
  /*r1=H'+H'*/ \
  "paddw %%mm1,%%mm1\n\t" \
  /*r1=R1=A''+H'*/ \
  "paddw %%mm2,%%mm1\n\t" \
  /*r2=NR2*/ \
  "psraw $4,%%mm2\n\t" \
  /*r4=E'=E-G*/ \
  "psubw %%mm7,%%mm4\n\t" \
  /*r1=NR1*/ \
  "psraw $4,%%mm1\n\t" \
  /*r3=D'*/ \
  "movq "OC_I(2,_y)",%%mm3\n\t" \
  /*r7=G+G*/ \
  "paddw %%mm7,%%mm7\n\t" \
  /*Store NR2 at I(2).*/ \
  "movq %%mm2,"OC_I(2,_y)"\n\t" \
  /*r7=G'=E+G*/ \
  "paddw %%mm4,%%mm7\n\t" \
  /*Store NR1 at I(1).*/ \
  "movq %%mm1,"OC_I(1,_y)"\n\t" \
  /*r4=R4=E'-D'*/ \
  "psubw %%mm3,%%mm4\n\t" \
  "paddw "OC_MEM_OFFS(0x00,c)",%%mm4\n\t" \
  /*r3=D'+D'*/ \
  "paddw %%mm3,%%mm3\n\t" \
  /*r3=R3=E'+D'*/ \
  "paddw %%mm4,%%mm3\n\t" \
  /*r4=NR4*/ \
  "psraw $4,%%mm4\n\t" \
  /*r6=R6=F'-B''*/ \
  "psubw %%mm5,%%mm6\n\t" \
  /*r3=NR3*/ \
  "psraw $4,%%mm3\n\t" \
  "paddw "OC_MEM_OFFS(0x00,c)",%%mm6\n\t" \
  /*r5=B''+B''*/ \
  "paddw %%mm5,%%mm5\n\t" \
  /*r5=R5=F'+B''*/ \
  "paddw %%mm6,%%mm5\n\t" \
  /*r6=NR6*/ \
  "psraw $4,%%mm6\n\t" \
  /*Store NR4 at J(4).*/ \
  "movq %%mm4,"OC_J(4,_y)"\n\t" \
  /*r5=NR5*/ \
  "psraw $4,%%mm5\n\t" \
  /*Store NR3 at I(3).*/ \
  "movq %%mm3,"OC_I(3,_y)"\n\t" \
  /*r7=R7=G'-C'*/ \
  "psubw %%mm0,%%mm7\n\t" \
  "paddw "OC_MEM_OFFS(0x00,c)",%%mm7\n\t" \
  /*r0=C'+C'*/ \
  "paddw %%mm0,%%mm0\n\t" \
  /*r0=R0=G'+C'*/ \
  "paddw %%mm7,%%mm0\n\t" \
  /*r7=NR7*/ \
  "psraw $4,%%mm7\n\t" \
  /*Store NR6 at J(6).*/ \
  "movq %%mm6,"OC_J(6,_y)"\n\t" \
  /*r0=NR0*/ \
  "psraw $4,%%mm0\n\t" \
  /*Store NR5 at J(5).*/ \
  "movq %%mm5,"OC_J(5,_y)"\n\t" \
  /*Store NR7 at J(7).*/ \
  "movq %%mm7,"OC_J(7,_y)"\n\t" \
  /*Store NR0 at I(0).*/ \
  "movq %%mm0,"OC_I(0,_y)"\n\t" \
  "#end OC_COLUMN_IDCT\n\t" \

static void oc_idct8x8_slow_mmx(ogg_int16_t _y[64],ogg_int16_t _x[64]){
  int i;
  /*This routine accepts an 8x8 matrix, but in partially transposed form.
    Every 4x4 block is transposed.*/
  __asm__ __volatile__(
#define OC_I(_k,_y)   OC_MEM_OFFS((_k)*16,_y)
#define OC_J(_k,_y)   OC_MEM_OFFS(((_k)-4)*16+8,_y)
    OC_ROW_IDCT(y,x)
    OC_TRANSPOSE(y)
#undef  OC_I
#undef  OC_J
#define OC_I(_k,_y)   OC_MEM_OFFS((_k)*16+64,_y)
#define OC_J(_k,_y)   OC_MEM_OFFS(((_k)-4)*16+72,_y)
    OC_ROW_IDCT(y,x)
    OC_TRANSPOSE(y)
#undef  OC_I
#undef  OC_J
#define OC_I(_k,_y)   OC_MEM_OFFS((_k)*16,_y)
#define OC_J(_k,_y)   OC_I(_k,_y)
    OC_COLUMN_IDCT(y)
#undef  OC_I
#undef  OC_J
#define OC_I(_k,_y)   OC_MEM_OFFS((_k)*16+8,_y)
#define OC_J(_k,_y)   OC_I(_k,_y)
    OC_COLUMN_IDCT(y)
#undef  OC_I
#undef  OC_J
    :[y]"=m"OC_ARRAY_OPERAND(ogg_int16_t,_y,64)
    :[x]"m"OC_CONST_ARRAY_OPERAND(ogg_int16_t,_x,64),
     [c]"m"OC_CONST_ARRAY_OPERAND(ogg_int16_t,OC_IDCT_CONSTS,64)
  );
  __asm__ __volatile__("pxor %%mm0,%%mm0\n\t"::);
  for(i=0;i<4;i++){
    __asm__ __volatile__(
      "movq %%mm0,"OC_MEM_OFFS(0x00,x)"\n\t"
      "movq %%mm0,"OC_MEM_OFFS(0x08,x)"\n\t"
      "movq %%mm0,"OC_MEM_OFFS(0x10,x)"\n\t"
      "movq %%mm0,"OC_MEM_OFFS(0x18,x)"\n\t"
      :[x]"=m"OC_ARRAY_OPERAND(ogg_int16_t,_x+16*i,16)
    );
  }
}

/*25 cycles.*/
#define OC_IDCT_BEGIN_10(_y,_x) \
 "#OC_IDCT_BEGIN_10\n\t" \
 "movq "OC_I(3,_x)",%%mm2\n\t" \
 "nop\n\t" \
 "movq "OC_MEM_OFFS(0x30,c)",%%mm6\n\t" \
 "movq %%mm2,%%mm4\n\t" \
 "movq "OC_MEM_OFFS(0x50,c)",%%mm1\n\t" \
 "pmulhw %%mm6,%%mm4\n\t" \
 "movq "OC_I(1,_x)",%%mm3\n\t" \
 "pmulhw %%mm2,%%mm1\n\t" \
 "movq "OC_MEM_OFFS(0x10,c)",%%mm0\n\t" \
 "paddw %%mm2,%%mm4\n\t" \
 "pxor %%mm6,%%mm6\n\t" \
 "paddw %%mm1,%%mm2\n\t" \
 "movq "OC_I(2,_x)",%%mm5\n\t" \
 "pmulhw %%mm3,%%mm0\n\t" \
 "movq %%mm5,%%mm1\n\t" \
 "paddw %%mm3,%%mm0\n\t" \
 "pmulhw "OC_MEM_OFFS(0x70,c)",%%mm3\n\t" \
 "psubw %%mm2,%%mm6\n\t" \
 "pmulhw "OC_MEM_OFFS(0x20,c)",%%mm5\n\t" \
 "psubw %%mm4,%%mm0\n\t" \
 "movq "OC_I(2,_x)",%%mm7\n\t" \
 "paddw %%mm4,%%mm4\n\t" \
 "paddw %%mm5,%%mm7\n\t" \
 "paddw %%mm0,%%mm4\n\t" \
 "pmulhw "OC_MEM_OFFS(0x60,c)",%%mm1\n\t" \
 "psubw %%mm6,%%mm3\n\t" \
 "movq %%mm4,"OC_I(1,_y)"\n\t" \
 "paddw %%mm6,%%mm6\n\t" \
 "movq "OC_MEM_OFFS(0x40,c)",%%mm4\n\t" \
 "paddw %%mm3,%%mm6\n\t" \
 "movq %%mm3,%%mm5\n\t" \
 "pmulhw %%mm4,%%mm3\n\t" \
 "movq %%mm6,"OC_I(2,_y)"\n\t" \
 "movq %%mm0,%%mm2\n\t" \
 "movq "OC_I(0,_x)",%%mm6\n\t" \
 "pmulhw %%mm4,%%mm0\n\t" \
 "paddw %%mm3,%%mm5\n\t" \
 "paddw %%mm0,%%mm2\n\t" \
 "psubw %%mm1,%%mm5\n\t" \
 "pmulhw %%mm4,%%mm6\n\t" \
 "paddw "OC_I(0,_x)",%%mm6\n\t" \
 "paddw %%mm1,%%mm1\n\t" \
 "movq %%mm6,%%mm4\n\t" \
 "paddw %%mm5,%%mm1\n\t" \
 "psubw %%mm2,%%mm6\n\t" \
 "paddw %%mm2,%%mm2\n\t" \
 "movq "OC_I(1,_y)",%%mm0\n\t" \
 "paddw %%mm6,%%mm2\n\t" \
 "psubw %%mm1,%%mm2\n\t" \
 "nop\n\t" \
 "#end OC_IDCT_BEGIN_10\n\t" \

/*25+8=33 cycles.*/
#define OC_ROW_IDCT_10(_y,_x) \
 "#OC_ROW_IDCT_10\n\t" \
 OC_IDCT_BEGIN_10(_y,_x) \
 /*r3=D'*/ \
 "movq "OC_I(2,_y)",%%mm3\n\t" \
 /*r4=E'=E-G*/ \
 "psubw %%mm7,%%mm4\n\t" \
 /*r1=H'+H'*/ \
 "paddw %%mm1,%%mm1\n\t" \
 /*r7=G+G*/ \
 "paddw %%mm7,%%mm7\n\t" \
 /*r1=R1=A''+H'*/ \
 "paddw %%mm2,%%mm1\n\t" \
 /*r7=G'=E+G*/ \
 "paddw %%mm4,%%mm7\n\t" \
 /*r4=R4=E'-D'*/ \
 "psubw %%mm3,%%mm4\n\t" \
 "paddw %%mm3,%%mm3\n\t" \
 /*r6=R6=F'-B''*/ \
 "psubw %%mm5,%%mm6\n\t" \
 "paddw %%mm5,%%mm5\n\t" \
 /*r3=R3=E'+D'*/ \
 "paddw %%mm4,%%mm3\n\t" \
 /*r5=R5=F'+B''*/ \
 "paddw %%mm6,%%mm5\n\t" \
 /*r7=R7=G'-C'*/ \
 "psubw %%mm0,%%mm7\n\t" \
 "paddw %%mm0,%%mm0\n\t" \
 /*Save R1.*/ \
 "movq %%mm1,"OC_I(1,_y)"\n\t" \
 /*r0=R0=G'+C'*/ \
 "paddw %%mm7,%%mm0\n\t" \
 "#end OC_ROW_IDCT_10\n\t" \

/*25+19=44 cycles'*/
#define OC_COLUMN_IDCT_10(_y) \
 "#OC_COLUMN_IDCT_10\n\t" \
 OC_IDCT_BEGIN_10(_y,_y) \
 "paddw "OC_MEM_OFFS(0x00,c)",%%mm2\n\t" \
 /*r1=H'+H'*/ \
 "paddw %%mm1,%%mm1\n\t" \
 /*r1=R1=A''+H'*/ \
 "paddw %%mm2,%%mm1\n\t" \
 /*r2=NR2*/ \
 "psraw $4,%%mm2\n\t" \
 /*r4=E'=E-G*/ \
 "psubw %%mm7,%%mm4\n\t" \
 /*r1=NR1*/ \
 "psraw $4,%%mm1\n\t" \
 /*r3=D'*/ \
 "movq "OC_I(2,_y)",%%mm3\n\t" \
 /*r7=G+G*/ \
 "paddw %%mm7,%%mm7\n\t" \
 /*Store NR2 at I(2).*/ \
 "movq %%mm2,"OC_I(2,_y)"\n\t" \
 /*r7=G'=E+G*/ \
 "paddw %%mm4,%%mm7\n\t" \
 /*Store NR1 at I(1).*/ \
 "movq %%mm1,"OC_I(1,_y)"\n\t" \
 /*r4=R4=E'-D'*/ \
 "psubw %%mm3,%%mm4\n\t" \
 "paddw "OC_MEM_OFFS(0x00,c)",%%mm4\n\t" \
 /*r3=D'+D'*/ \
 "paddw %%mm3,%%mm3\n\t" \
 /*r3=R3=E'+D'*/ \
 "paddw %%mm4,%%mm3\n\t" \
 /*r4=NR4*/ \
 "psraw $4,%%mm4\n\t" \
 /*r6=R6=F'-B''*/ \
 "psubw %%mm5,%%mm6\n\t" \
 /*r3=NR3*/ \
 "psraw $4,%%mm3\n\t" \
 "paddw "OC_MEM_OFFS(0x00,c)",%%mm6\n\t" \
 /*r5=B''+B''*/ \
 "paddw %%mm5,%%mm5\n\t" \
 /*r5=R5=F'+B''*/ \
 "paddw %%mm6,%%mm5\n\t" \
 /*r6=NR6*/ \
 "psraw $4,%%mm6\n\t" \
 /*Store NR4 at J(4).*/ \
 "movq %%mm4,"OC_J(4,_y)"\n\t" \
 /*r5=NR5*/ \
 "psraw $4,%%mm5\n\t" \
 /*Store NR3 at I(3).*/ \
 "movq %%mm3,"OC_I(3,_y)"\n\t" \
 /*r7=R7=G'-C'*/ \
 "psubw %%mm0,%%mm7\n\t" \
 "paddw "OC_MEM_OFFS(0x00,c)",%%mm7\n\t" \
 /*r0=C'+C'*/ \
 "paddw %%mm0,%%mm0\n\t" \
 /*r0=R0=G'+C'*/ \
 "paddw %%mm7,%%mm0\n\t" \
 /*r7=NR7*/ \
 "psraw $4,%%mm7\n\t" \
 /*Store NR6 at J(6).*/ \
 "movq %%mm6,"OC_J(6,_y)"\n\t" \
 /*r0=NR0*/ \
 "psraw $4,%%mm0\n\t" \
 /*Store NR5 at J(5).*/ \
 "movq %%mm5,"OC_J(5,_y)"\n\t" \
 /*Store NR7 at J(7).*/ \
 "movq %%mm7,"OC_J(7,_y)"\n\t" \
 /*Store NR0 at I(0).*/ \
 "movq %%mm0,"OC_I(0,_y)"\n\t" \
 "#end OC_COLUMN_IDCT_10\n\t" \

static void oc_idct8x8_10_mmx(ogg_int16_t _y[64],ogg_int16_t _x[64]){
  __asm__ __volatile__(
#define OC_I(_k,_y) OC_MEM_OFFS((_k)*16,_y)
#define OC_J(_k,_y) OC_MEM_OFFS(((_k)-4)*16+8,_y)
    /*Done with dequant, descramble, and partial transpose.
      Now do the iDCT itself.*/
    OC_ROW_IDCT_10(y,x)
    OC_TRANSPOSE(y)
#undef  OC_I
#undef  OC_J
#define OC_I(_k,_y) OC_MEM_OFFS((_k)*16,_y)
#define OC_J(_k,_y) OC_I(_k,_y)
    OC_COLUMN_IDCT_10(y)
#undef  OC_I
#undef  OC_J
#define OC_I(_k,_y) OC_MEM_OFFS((_k)*16+8,_y)
#define OC_J(_k,_y) OC_I(_k,_y)
    OC_COLUMN_IDCT_10(y)
#undef  OC_I
#undef  OC_J
    :[y]"=m"OC_ARRAY_OPERAND(ogg_int16_t,_y,64)
    :[x]"m"OC_CONST_ARRAY_OPERAND(ogg_int16_t,_x,64),
     [c]"m"OC_CONST_ARRAY_OPERAND(ogg_int16_t,OC_IDCT_CONSTS,64)
  );
  __asm__ __volatile__(
    "pxor %%mm0,%%mm0\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x00,x)"\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x10,x)"\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x20,x)"\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x30,x)"\n\t"
    :[x]"+m"OC_ARRAY_OPERAND(ogg_int16_t,_x,28)
  );
}

/*Performs an inverse 8x8 Type-II DCT transform.
  The input is assumed to be scaled by a factor of 4 relative to orthonormal
   version of the transform.*/
void oc_idct8x8_mmx(ogg_int16_t _y[64],ogg_int16_t _x[64],int _last_zzi){
  /*_last_zzi is subtly different from an actual count of the number of
     coefficients we decoded for this block.
    It contains the value of zzi BEFORE the final token in the block was
     decoded.
    In most cases this is an EOB token (the continuation of an EOB run from a
     previous block counts), and so this is the same as the coefficient count.
    However, in the case that the last token was NOT an EOB token, but filled
     the block up with exactly 64 coefficients, _last_zzi will be less than 64.
    Provided the last token was not a pure zero run, the minimum value it can
     be is 46, and so that doesn't affect any of the cases in this routine.
    However, if the last token WAS a pure zero run of length 63, then _last_zzi
     will be 1 while the number of coefficients decoded is 64.
    Thus, we will trigger the following special case, where the real
     coefficient count would not.
    Note also that a zero run of length 64 will give _last_zzi a value of 0,
     but we still process the DC coefficient, which might have a non-zero value
     due to DC prediction.
    Although convoluted, this is arguably the correct behavior: it allows us to
     use a smaller transform when the block ends with a long zero run instead
     of a normal EOB token.
    It could be smarter... multiple separate zero runs at the end of a block
     will fool it, but an encoder that generates these really deserves what it
     gets.
    Needless to say we inherited this approach from VP3.*/
  /*Then perform the iDCT.*/
  if(_last_zzi<=10)oc_idct8x8_10_mmx(_y,_x);
  else oc_idct8x8_slow_mmx(_y,_x);
}

#endif
