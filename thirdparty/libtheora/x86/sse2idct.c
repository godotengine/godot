/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************

  function:
    last mod: $Id: mmxidct.c 16503 2009-08-22 18:14:02Z giles $

 ********************************************************************/

/*SSE2 acceleration of Theora's iDCT.*/
#include "x86int.h"
#include "sse2trans.h"
#include "../dct.h"

#if defined(OC_X86_ASM)

/*A table of constants used by the MMX routines.*/
const unsigned short __attribute__((aligned(16),used)) OC_IDCT_CONSTS[64]={
        8,      8,      8,      8,      8,      8,      8,      8,
  OC_C1S7,OC_C1S7,OC_C1S7,OC_C1S7,OC_C1S7,OC_C1S7,OC_C1S7,OC_C1S7,
  OC_C2S6,OC_C2S6,OC_C2S6,OC_C2S6,OC_C2S6,OC_C2S6,OC_C2S6,OC_C2S6,
  OC_C3S5,OC_C3S5,OC_C3S5,OC_C3S5,OC_C3S5,OC_C3S5,OC_C3S5,OC_C3S5,
  OC_C4S4,OC_C4S4,OC_C4S4,OC_C4S4,OC_C4S4,OC_C4S4,OC_C4S4,OC_C4S4,
  OC_C5S3,OC_C5S3,OC_C5S3,OC_C5S3,OC_C5S3,OC_C5S3,OC_C5S3,OC_C5S3,
  OC_C6S2,OC_C6S2,OC_C6S2,OC_C6S2,OC_C6S2,OC_C6S2,OC_C6S2,OC_C6S2,
  OC_C7S1,OC_C7S1,OC_C7S1,OC_C7S1,OC_C7S1,OC_C7S1,OC_C7S1,OC_C7S1
};


/*Performs the first three stages of the iDCT.
  xmm2, xmm6, xmm3, and xmm5 must contain the corresponding rows of the input
   (accessed in that order).
  The remaining rows must be in _x at their corresponding locations.
  On output, xmm7 down to xmm4 contain rows 0 through 3, and xmm0 up to xmm3
   contain rows 4 through 7.*/
#define OC_IDCT_8x8_ABC(_x) \
  "#OC_IDCT_8x8_ABC\n\t" \
  /*Stage 1:*/ \
  /*2-3 rotation by 6pi/16. \
    xmm4=xmm7=C6, xmm0=xmm1=C2, xmm2=X2, xmm6=X6.*/ \
  "movdqa "OC_MEM_OFFS(0x20,c)",%%xmm1\n\t" \
  "movdqa "OC_MEM_OFFS(0x60,c)",%%xmm4\n\t" \
  "movdqa %%xmm1,%%xmm0\n\t" \
  "pmulhw %%xmm2,%%xmm1\n\t" \
  "movdqa %%xmm4,%%xmm7\n\t" \
  "pmulhw %%xmm6,%%xmm0\n\t" \
  "pmulhw %%xmm2,%%xmm7\n\t" \
  "pmulhw %%xmm6,%%xmm4\n\t" \
  "paddw %%xmm6,%%xmm0\n\t" \
  "movdqa "OC_MEM_OFFS(0x30,c)",%%xmm6\n\t" \
  "paddw %%xmm1,%%xmm2\n\t" \
  "psubw %%xmm0,%%xmm7\n\t" \
  "movdqa %%xmm7,"OC_MEM_OFFS(0x00,buf)"\n\t" \
  "paddw %%xmm4,%%xmm2\n\t" \
  "movdqa "OC_MEM_OFFS(0x50,c)",%%xmm4\n\t" \
  "movdqa %%xmm2,"OC_MEM_OFFS(0x10,buf)"\n\t" \
  /*5-6 rotation by 3pi/16. \
    xmm4=xmm2=C5, xmm1=xmm6=C3, xmm3=X3, xmm5=X5.*/ \
  "movdqa %%xmm4,%%xmm2\n\t" \
  "movdqa %%xmm6,%%xmm1\n\t" \
  "pmulhw %%xmm3,%%xmm4\n\t" \
  "pmulhw %%xmm5,%%xmm1\n\t" \
  "pmulhw %%xmm3,%%xmm6\n\t" \
  "pmulhw %%xmm5,%%xmm2\n\t" \
  "paddw %%xmm3,%%xmm4\n\t" \
  "paddw %%xmm5,%%xmm3\n\t" \
  "paddw %%xmm6,%%xmm3\n\t" \
  "movdqa "OC_MEM_OFFS(0x70,_x)",%%xmm6\n\t" \
  "paddw %%xmm5,%%xmm1\n\t" \
  "movdqa "OC_MEM_OFFS(0x10,_x)",%%xmm5\n\t" \
  "paddw %%xmm3,%%xmm2\n\t" \
  "movdqa "OC_MEM_OFFS(0x70,c)",%%xmm3\n\t" \
  "psubw %%xmm4,%%xmm1\n\t" \
  "movdqa "OC_MEM_OFFS(0x10,c)",%%xmm4\n\t" \
  /*4-7 rotation by 7pi/16. \
    xmm4=xmm7=C1, xmm3=xmm0=C7, xmm5=X1, xmm6=X7.*/ \
  "movdqa %%xmm3,%%xmm0\n\t" \
  "movdqa %%xmm4,%%xmm7\n\t" \
  "pmulhw %%xmm5,%%xmm3\n\t" \
  "pmulhw %%xmm5,%%xmm7\n\t" \
  "pmulhw %%xmm6,%%xmm4\n\t" \
  "pmulhw %%xmm6,%%xmm0\n\t" \
  "paddw %%xmm6,%%xmm4\n\t" \
  "movdqa "OC_MEM_OFFS(0x40,_x)",%%xmm6\n\t" \
  "paddw %%xmm5,%%xmm7\n\t" \
  "psubw %%xmm4,%%xmm3\n\t" \
  "movdqa "OC_MEM_OFFS(0x40,c)",%%xmm4\n\t" \
  "paddw %%xmm7,%%xmm0\n\t" \
  "movdqa "OC_MEM_OFFS(0x00,_x)",%%xmm7\n\t" \
  /*0-1 butterfly. \
    xmm4=xmm5=C4, xmm7=X0, xmm6=X4.*/ \
  "paddw %%xmm7,%%xmm6\n\t" \
  "movdqa %%xmm4,%%xmm5\n\t" \
  "pmulhw %%xmm6,%%xmm4\n\t" \
  "paddw %%xmm7,%%xmm7\n\t" \
  "psubw %%xmm6,%%xmm7\n\t" \
  "paddw %%xmm6,%%xmm4\n\t" \
  /*Stage 2:*/ \
  /*4-5 butterfly: xmm3=t[4], xmm1=t[5] \
    7-6 butterfly: xmm2=t[6], xmm0=t[7]*/ \
  "movdqa %%xmm3,%%xmm6\n\t" \
  "paddw %%xmm1,%%xmm3\n\t" \
  "psubw %%xmm1,%%xmm6\n\t" \
  "movdqa %%xmm5,%%xmm1\n\t" \
  "pmulhw %%xmm7,%%xmm5\n\t" \
  "paddw %%xmm7,%%xmm5\n\t" \
  "movdqa %%xmm0,%%xmm7\n\t" \
  "paddw %%xmm2,%%xmm0\n\t" \
  "psubw %%xmm2,%%xmm7\n\t" \
  "movdqa %%xmm1,%%xmm2\n\t" \
  "pmulhw %%xmm6,%%xmm1\n\t" \
  "pmulhw %%xmm7,%%xmm2\n\t" \
  "paddw %%xmm6,%%xmm1\n\t" \
  "movdqa "OC_MEM_OFFS(0x00,buf)",%%xmm6\n\t" \
  "paddw %%xmm7,%%xmm2\n\t" \
  "movdqa "OC_MEM_OFFS(0x10,buf)",%%xmm7\n\t" \
  /*Stage 3: \
    6-5 butterfly: xmm1=t[5], xmm2=t[6] -> xmm1=t[6]+t[5], xmm2=t[6]-t[5] \
    0-3 butterfly: xmm4=t[0], xmm7=t[3] -> xmm7=t[0]+t[3], xmm4=t[0]-t[3] \
    1-2 butterfly: xmm5=t[1], xmm6=t[2] -> xmm6=t[1]+t[2], xmm5=t[1]-t[2]*/ \
  "paddw %%xmm2,%%xmm1\n\t" \
  "paddw %%xmm5,%%xmm6\n\t" \
  "paddw %%xmm4,%%xmm7\n\t" \
  "paddw %%xmm2,%%xmm2\n\t" \
  "paddw %%xmm4,%%xmm4\n\t" \
  "paddw %%xmm5,%%xmm5\n\t" \
  "psubw %%xmm1,%%xmm2\n\t" \
  "psubw %%xmm7,%%xmm4\n\t" \
  "psubw %%xmm6,%%xmm5\n\t" \

/*Performs the last stage of the iDCT.
  On input, xmm7 down to xmm4 contain rows 0 through 3, and xmm0 up to xmm3
   contain rows 4 through 7.
  On output, xmm0 through xmm7 contain the corresponding rows.*/
#define OC_IDCT_8x8_D \
  "#OC_IDCT_8x8_D\n\t" \
  /*Stage 4: \
    0-7 butterfly: xmm7=t[0], xmm0=t[7] -> xmm0=t[0]+t[7], xmm7=t[0]-t[7] \
    1-6 butterfly: xmm6=t[1], xmm1=t[6] -> xmm1=t[1]+t[6], xmm6=t[1]-t[6] \
    2-5 butterfly: xmm5=t[2], xmm2=t[5] -> xmm2=t[2]+t[5], xmm5=t[2]-t[5] \
    3-4 butterfly: xmm4=t[3], xmm3=t[4] -> xmm3=t[3]+t[4], xmm4=t[3]-t[4]*/ \
  "psubw %%xmm0,%%xmm7\n\t" \
  "psubw %%xmm1,%%xmm6\n\t" \
  "psubw %%xmm2,%%xmm5\n\t" \
  "psubw %%xmm3,%%xmm4\n\t" \
  "paddw %%xmm0,%%xmm0\n\t" \
  "paddw %%xmm1,%%xmm1\n\t" \
  "paddw %%xmm2,%%xmm2\n\t" \
  "paddw %%xmm3,%%xmm3\n\t" \
  "paddw %%xmm7,%%xmm0\n\t" \
  "paddw %%xmm6,%%xmm1\n\t" \
  "paddw %%xmm5,%%xmm2\n\t" \
  "paddw %%xmm4,%%xmm3\n\t" \

/*Performs the last stage of the iDCT.
  On input, xmm7 down to xmm4 contain rows 0 through 3, and xmm0 up to xmm3
   contain rows 4 through 7.
  On output, xmm0 through xmm7 contain the corresponding rows.*/
#define OC_IDCT_8x8_D_STORE \
  "#OC_IDCT_8x8_D_STORE\n\t" \
  /*Stage 4: \
    0-7 butterfly: xmm7=t[0], xmm0=t[7] -> xmm0=t[0]+t[7], xmm7=t[0]-t[7] \
    1-6 butterfly: xmm6=t[1], xmm1=t[6] -> xmm1=t[1]+t[6], xmm6=t[1]-t[6] \
    2-5 butterfly: xmm5=t[2], xmm2=t[5] -> xmm2=t[2]+t[5], xmm5=t[2]-t[5] \
    3-4 butterfly: xmm4=t[3], xmm3=t[4] -> xmm3=t[3]+t[4], xmm4=t[3]-t[4]*/ \
  "psubw %%xmm3,%%xmm4\n\t" \
  "movdqa %%xmm4,"OC_MEM_OFFS(0x40,y)"\n\t" \
  "movdqa "OC_MEM_OFFS(0x00,c)",%%xmm4\n\t" \
  "psubw %%xmm0,%%xmm7\n\t" \
  "psubw %%xmm1,%%xmm6\n\t" \
  "psubw %%xmm2,%%xmm5\n\t" \
  "paddw %%xmm4,%%xmm7\n\t" \
  "paddw %%xmm4,%%xmm6\n\t" \
  "paddw %%xmm4,%%xmm5\n\t" \
  "paddw "OC_MEM_OFFS(0x40,y)",%%xmm4\n\t" \
  "paddw %%xmm0,%%xmm0\n\t" \
  "paddw %%xmm1,%%xmm1\n\t" \
  "paddw %%xmm2,%%xmm2\n\t" \
  "paddw %%xmm3,%%xmm3\n\t" \
  "paddw %%xmm7,%%xmm0\n\t" \
  "paddw %%xmm6,%%xmm1\n\t" \
  "psraw $4,%%xmm0\n\t" \
  "paddw %%xmm5,%%xmm2\n\t" \
  "movdqa %%xmm0,"OC_MEM_OFFS(0x00,y)"\n\t" \
  "psraw $4,%%xmm1\n\t" \
  "paddw %%xmm4,%%xmm3\n\t" \
  "movdqa %%xmm1,"OC_MEM_OFFS(0x10,y)"\n\t" \
  "psraw $4,%%xmm2\n\t" \
  "movdqa %%xmm2,"OC_MEM_OFFS(0x20,y)"\n\t" \
  "psraw $4,%%xmm3\n\t" \
  "movdqa %%xmm3,"OC_MEM_OFFS(0x30,y)"\n\t" \
  "psraw $4,%%xmm4\n\t" \
  "movdqa %%xmm4,"OC_MEM_OFFS(0x40,y)"\n\t" \
  "psraw $4,%%xmm5\n\t" \
  "movdqa %%xmm5,"OC_MEM_OFFS(0x50,y)"\n\t" \
  "psraw $4,%%xmm6\n\t" \
  "movdqa %%xmm6,"OC_MEM_OFFS(0x60,y)"\n\t" \
  "psraw $4,%%xmm7\n\t" \
  "movdqa %%xmm7,"OC_MEM_OFFS(0x70,y)"\n\t" \

static void oc_idct8x8_slow_sse2(ogg_int16_t _y[64],ogg_int16_t _x[64]){
  OC_ALIGN16(ogg_int16_t buf[16]);
  int i;
  /*This routine accepts an 8x8 matrix pre-transposed.*/
  __asm__ __volatile__(
    /*Load rows 2, 3, 5, and 6 for the first stage of the iDCT.*/
    "movdqa "OC_MEM_OFFS(0x20,x)",%%xmm2\n\t"
    "movdqa "OC_MEM_OFFS(0x60,x)",%%xmm6\n\t"
    "movdqa "OC_MEM_OFFS(0x30,x)",%%xmm3\n\t"
    "movdqa "OC_MEM_OFFS(0x50,x)",%%xmm5\n\t"
    OC_IDCT_8x8_ABC(x)
    OC_IDCT_8x8_D
    OC_TRANSPOSE_8x8
    /*Clear out rows 0, 1, 4, and 7 for the first stage of the iDCT.*/
    "movdqa %%xmm7,"OC_MEM_OFFS(0x70,y)"\n\t"
    "movdqa %%xmm4,"OC_MEM_OFFS(0x40,y)"\n\t"
    "movdqa %%xmm1,"OC_MEM_OFFS(0x10,y)"\n\t"
    "movdqa %%xmm0,"OC_MEM_OFFS(0x00,y)"\n\t"
    OC_IDCT_8x8_ABC(y)
    OC_IDCT_8x8_D_STORE
    :[buf]"=m"(OC_ARRAY_OPERAND(ogg_int16_t,buf,16)),
     [y]"=m"(OC_ARRAY_OPERAND(ogg_int16_t,_y,64))
    :[x]"m"(OC_CONST_ARRAY_OPERAND(ogg_int16_t,_x,64)),
     [c]"m"(OC_CONST_ARRAY_OPERAND(ogg_int16_t,OC_IDCT_CONSTS,128))
  );
  __asm__ __volatile__("pxor %%xmm0,%%xmm0\n\t"::);
  /*Clear input data for next block (decoder only).*/
  for(i=0;i<2;i++){
    __asm__ __volatile__(
      "movdqa %%xmm0,"OC_MEM_OFFS(0x00,x)"\n\t"
      "movdqa %%xmm0,"OC_MEM_OFFS(0x10,x)"\n\t"
      "movdqa %%xmm0,"OC_MEM_OFFS(0x20,x)"\n\t"
      "movdqa %%xmm0,"OC_MEM_OFFS(0x30,x)"\n\t"
      :[x]"=m"(OC_ARRAY_OPERAND(ogg_int16_t,_x+i*32,32))
    );
  }
}

/*For the first step of the 10-coefficient version of the 8x8 iDCT, we only
   need to work with four columns at a time.
  Doing this in MMX is faster on processors with a 64-bit data path.*/
#define OC_IDCT_8x8_10_MMX \
  "#OC_IDCT_8x8_10_MMX\n\t" \
  /*Stage 1:*/ \
  /*2-3 rotation by 6pi/16. \
    mm7=C6, mm6=C2, mm2=X2, X6=0.*/ \
  "movq "OC_MEM_OFFS(0x60,c)",%%mm7\n\t" \
  "movq "OC_MEM_OFFS(0x20,c)",%%mm6\n\t" \
  "pmulhw %%mm2,%%mm6\n\t" \
  "pmulhw %%mm2,%%mm7\n\t" \
  "movq "OC_MEM_OFFS(0x50,c)",%%mm5\n\t" \
  "paddw %%mm6,%%mm2\n\t" \
  "movq %%mm2,"OC_MEM_OFFS(0x10,buf)"\n\t" \
  "movq "OC_MEM_OFFS(0x30,c)",%%mm2\n\t" \
  "movq %%mm7,"OC_MEM_OFFS(0x00,buf)"\n\t" \
  /*5-6 rotation by 3pi/16. \
    mm5=C5, mm2=C3, mm3=X3, X5=0.*/ \
  "pmulhw %%mm3,%%mm5\n\t" \
  "pmulhw %%mm3,%%mm2\n\t" \
  "movq "OC_MEM_OFFS(0x10,c)",%%mm7\n\t" \
  "paddw %%mm3,%%mm5\n\t" \
  "paddw %%mm3,%%mm2\n\t" \
  "movq "OC_MEM_OFFS(0x70,c)",%%mm3\n\t" \
  /*4-7 rotation by 7pi/16. \
    mm7=C1, mm3=C7, mm1=X1, X7=0.*/ \
  "pmulhw %%mm1,%%mm3\n\t" \
  "pmulhw %%mm1,%%mm7\n\t" \
  "movq "OC_MEM_OFFS(0x40,c)",%%mm4\n\t" \
  "movq %%mm3,%%mm6\n\t" \
  "paddw %%mm1,%%mm7\n\t" \
  /*0-1 butterfly. \
    mm4=C4, mm0=X0, X4=0.*/ \
  /*Stage 2:*/ \
  /*4-5 butterfly: mm3=t[4], mm5=t[5] \
    7-6 butterfly: mm2=t[6], mm7=t[7]*/ \
  "psubw %%mm5,%%mm3\n\t" \
  "paddw %%mm5,%%mm6\n\t" \
  "movq %%mm4,%%mm1\n\t" \
  "pmulhw %%mm0,%%mm4\n\t" \
  "paddw %%mm0,%%mm4\n\t" \
  "movq %%mm7,%%mm0\n\t" \
  "movq %%mm4,%%mm5\n\t" \
  "paddw %%mm2,%%mm0\n\t" \
  "psubw %%mm2,%%mm7\n\t" \
  "movq %%mm1,%%mm2\n\t" \
  "pmulhw %%mm6,%%mm1\n\t" \
  "pmulhw %%mm7,%%mm2\n\t" \
  "paddw %%mm6,%%mm1\n\t" \
  "movq "OC_MEM_OFFS(0x00,buf)",%%mm6\n\t" \
  "paddw %%mm7,%%mm2\n\t" \
  "movq "OC_MEM_OFFS(0x10,buf)",%%mm7\n\t" \
  /*Stage 3: \
    6-5 butterfly: mm1=t[5], mm2=t[6] -> mm1=t[6]+t[5], mm2=t[6]-t[5] \
    0-3 butterfly: mm4=t[0], mm7=t[3] -> mm7=t[0]+t[3], mm4=t[0]-t[3] \
    1-2 butterfly: mm5=t[1], mm6=t[2] -> mm6=t[1]+t[2], mm5=t[1]-t[2]*/ \
  "paddw %%mm2,%%mm1\n\t" \
  "paddw %%mm5,%%mm6\n\t" \
  "paddw %%mm4,%%mm7\n\t" \
  "paddw %%mm2,%%mm2\n\t" \
  "paddw %%mm4,%%mm4\n\t" \
  "paddw %%mm5,%%mm5\n\t" \
  "psubw %%mm1,%%mm2\n\t" \
  "psubw %%mm7,%%mm4\n\t" \
  "psubw %%mm6,%%mm5\n\t" \
  /*Stage 4: \
    0-7 butterfly: mm7=t[0], mm0=t[7] -> mm0=t[0]+t[7], mm7=t[0]-t[7] \
    1-6 butterfly: mm6=t[1], mm1=t[6] -> mm1=t[1]+t[6], mm6=t[1]-t[6] \
    2-5 butterfly: mm5=t[2], mm2=t[5] -> mm2=t[2]+t[5], mm5=t[2]-t[5] \
    3-4 butterfly: mm4=t[3], mm3=t[4] -> mm3=t[3]+t[4], mm4=t[3]-t[4]*/ \
  "psubw %%mm0,%%mm7\n\t" \
  "psubw %%mm1,%%mm6\n\t" \
  "psubw %%mm2,%%mm5\n\t" \
  "psubw %%mm3,%%mm4\n\t" \
  "paddw %%mm0,%%mm0\n\t" \
  "paddw %%mm1,%%mm1\n\t" \
  "paddw %%mm2,%%mm2\n\t" \
  "paddw %%mm3,%%mm3\n\t" \
  "paddw %%mm7,%%mm0\n\t" \
  "paddw %%mm6,%%mm1\n\t" \
  "paddw %%mm5,%%mm2\n\t" \
  "paddw %%mm4,%%mm3\n\t" \

#define OC_IDCT_8x8_10_ABC \
  "#OC_IDCT_8x8_10_ABC\n\t" \
  /*Stage 1:*/ \
  /*2-3 rotation by 6pi/16. \
    xmm7=C6, xmm6=C2, xmm2=X2, X6=0.*/ \
  "movdqa "OC_MEM_OFFS(0x60,c)",%%xmm7\n\t" \
  "movdqa "OC_MEM_OFFS(0x20,c)",%%xmm6\n\t" \
  "pmulhw %%xmm2,%%xmm6\n\t" \
  "pmulhw %%xmm2,%%xmm7\n\t" \
  "movdqa "OC_MEM_OFFS(0x50,c)",%%xmm5\n\t" \
  "paddw %%xmm6,%%xmm2\n\t" \
  "movdqa %%xmm2,"OC_MEM_OFFS(0x10,buf)"\n\t" \
  "movdqa "OC_MEM_OFFS(0x30,c)",%%xmm2\n\t" \
  "movdqa %%xmm7,"OC_MEM_OFFS(0x00,buf)"\n\t" \
  /*5-6 rotation by 3pi/16. \
    xmm5=C5, xmm2=C3, xmm3=X3, X5=0.*/ \
  "pmulhw %%xmm3,%%xmm5\n\t" \
  "pmulhw %%xmm3,%%xmm2\n\t" \
  "movdqa "OC_MEM_OFFS(0x10,c)",%%xmm7\n\t" \
  "paddw %%xmm3,%%xmm5\n\t" \
  "paddw %%xmm3,%%xmm2\n\t" \
  "movdqa "OC_MEM_OFFS(0x70,c)",%%xmm3\n\t" \
  /*4-7 rotation by 7pi/16. \
    xmm7=C1, xmm3=C7, xmm1=X1, X7=0.*/ \
  "pmulhw %%xmm1,%%xmm3\n\t" \
  "pmulhw %%xmm1,%%xmm7\n\t" \
  "movdqa "OC_MEM_OFFS(0x40,c)",%%xmm4\n\t" \
  "movdqa %%xmm3,%%xmm6\n\t" \
  "paddw %%xmm1,%%xmm7\n\t" \
  /*0-1 butterfly. \
    xmm4=C4, xmm0=X0, X4=0.*/ \
  /*Stage 2:*/ \
  /*4-5 butterfly: xmm3=t[4], xmm5=t[5] \
    7-6 butterfly: xmm2=t[6], xmm7=t[7]*/ \
  "psubw %%xmm5,%%xmm3\n\t" \
  "paddw %%xmm5,%%xmm6\n\t" \
  "movdqa %%xmm4,%%xmm1\n\t" \
  "pmulhw %%xmm0,%%xmm4\n\t" \
  "paddw %%xmm0,%%xmm4\n\t" \
  "movdqa %%xmm7,%%xmm0\n\t" \
  "movdqa %%xmm4,%%xmm5\n\t" \
  "paddw %%xmm2,%%xmm0\n\t" \
  "psubw %%xmm2,%%xmm7\n\t" \
  "movdqa %%xmm1,%%xmm2\n\t" \
  "pmulhw %%xmm6,%%xmm1\n\t" \
  "pmulhw %%xmm7,%%xmm2\n\t" \
  "paddw %%xmm6,%%xmm1\n\t" \
  "movdqa "OC_MEM_OFFS(0x00,buf)",%%xmm6\n\t" \
  "paddw %%xmm7,%%xmm2\n\t" \
  "movdqa "OC_MEM_OFFS(0x10,buf)",%%xmm7\n\t" \
  /*Stage 3: \
    6-5 butterfly: xmm1=t[5], xmm2=t[6] -> xmm1=t[6]+t[5], xmm2=t[6]-t[5] \
    0-3 butterfly: xmm4=t[0], xmm7=t[3] -> xmm7=t[0]+t[3], xmm4=t[0]-t[3] \
    1-2 butterfly: xmm5=t[1], xmm6=t[2] -> xmm6=t[1]+t[2], xmm5=t[1]-t[2]*/ \
  "paddw %%xmm2,%%xmm1\n\t" \
  "paddw %%xmm5,%%xmm6\n\t" \
  "paddw %%xmm4,%%xmm7\n\t" \
  "paddw %%xmm2,%%xmm2\n\t" \
  "paddw %%xmm4,%%xmm4\n\t" \
  "paddw %%xmm5,%%xmm5\n\t" \
  "psubw %%xmm1,%%xmm2\n\t" \
  "psubw %%xmm7,%%xmm4\n\t" \
  "psubw %%xmm6,%%xmm5\n\t" \

static void oc_idct8x8_10_sse2(ogg_int16_t _y[64],ogg_int16_t _x[64]){
  OC_ALIGN16(ogg_int16_t buf[16]);
  /*This routine accepts an 8x8 matrix pre-transposed.*/
  __asm__ __volatile__(
    "movq "OC_MEM_OFFS(0x20,x)",%%mm2\n\t"
    "movq "OC_MEM_OFFS(0x30,x)",%%mm3\n\t"
    "movq "OC_MEM_OFFS(0x10,x)",%%mm1\n\t"
    "movq "OC_MEM_OFFS(0x00,x)",%%mm0\n\t"
    OC_IDCT_8x8_10_MMX
    OC_TRANSPOSE_8x4_MMX2SSE
    OC_IDCT_8x8_10_ABC
    OC_IDCT_8x8_D_STORE
    :[buf]"=m"(OC_ARRAY_OPERAND(short,buf,16)),
     [y]"=m"(OC_ARRAY_OPERAND(ogg_int16_t,_y,64))
    :[x]"m"OC_CONST_ARRAY_OPERAND(ogg_int16_t,_x,64),
     [c]"m"(OC_CONST_ARRAY_OPERAND(ogg_int16_t,OC_IDCT_CONSTS,128))
  );
  /*Clear input data for next block (decoder only).*/
  __asm__ __volatile__(
    "pxor %%mm0,%%mm0\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x00,x)"\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x10,x)"\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x20,x)"\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x30,x)"\n\t"
    :[x]"+m"(OC_ARRAY_OPERAND(ogg_int16_t,_x,28))
  );
}

/*Performs an inverse 8x8 Type-II DCT transform.
  The input is assumed to be scaled by a factor of 4 relative to orthonormal
   version of the transform.*/
void oc_idct8x8_sse2(ogg_int16_t _y[64],ogg_int16_t _x[64],int _last_zzi){
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
  if(_last_zzi<=10)oc_idct8x8_10_sse2(_y,_x);
  else oc_idct8x8_slow_sse2(_y,_x);
}

#endif
