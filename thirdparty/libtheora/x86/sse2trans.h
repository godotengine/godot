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
    last mod: $Id: sse2trans.h 15675 2009-02-06 09:43:27Z tterribe $

 ********************************************************************/

#if !defined(_x86_sse2trans_H)
# define _x86_sse2trans_H (1)
# include "x86int.h"

# if defined(OC_X86_64_ASM)
/*On x86-64 we can transpose in-place without spilling registers.
  By clever choices of the order to apply the butterflies and the order of
   their outputs, we can take the rows in order and output the columns in order
   without any extra operations and using just one temporary register.*/
#  define OC_TRANSPOSE_8x8 \
 "#OC_TRANSPOSE_8x8\n\t" \
 "movdqa %%xmm4,%%xmm8\n\t" \
 /*xmm4 = f3 e3 f2 e2 f1 e1 f0 e0*/ \
 "punpcklwd %%xmm5,%%xmm4\n\t" \
 /*xmm8 = f7 e7 f6 e6 f5 e5 f4 e4*/ \
 "punpckhwd %%xmm5,%%xmm8\n\t" \
 /*xmm5 is free.*/ \
 "movdqa %%xmm0,%%xmm5\n\t" \
 /*xmm0 = b3 a3 b2 a2 b1 a1 b0 a0*/ \
 "punpcklwd %%xmm1,%%xmm0\n\t" \
 /*xmm5 = b7 a7 b6 a6 b5 a5 b4 a4*/ \
 "punpckhwd %%xmm1,%%xmm5\n\t" \
 /*xmm1 is free.*/ \
 "movdqa %%xmm6,%%xmm1\n\t" \
 /*xmm6 = h3 g3 h2 g2 h1 g1 h0 g0*/ \
 "punpcklwd %%xmm7,%%xmm6\n\t" \
 /*xmm1 = h7 g7 h6 g6 h5 g5 h4 g4*/ \
 "punpckhwd %%xmm7,%%xmm1\n\t" \
 /*xmm7 is free.*/ \
 "movdqa %%xmm2,%%xmm7\n\t" \
 /*xmm2 = d7 c7 d6 c6 d5 c5 d4 c4*/ \
 "punpckhwd %%xmm3,%%xmm2\n\t" \
 /*xmm7 = d3 c3 d2 c2 d1 c1 d0 c0*/ \
 "punpcklwd %%xmm3,%%xmm7\n\t" \
 /*xmm3 is free.*/ \
 "movdqa %%xmm0,%%xmm3\n\t" \
 /*xmm0 = d1 c1 b1 a1 d0 c0 b0 a0*/ \
 "punpckldq %%xmm7,%%xmm0\n\t" \
 /*xmm3 = d3 c3 b3 a3 d2 c2 b2 a2*/ \
 "punpckhdq %%xmm7,%%xmm3\n\t" \
 /*xmm7 is free.*/ \
 "movdqa %%xmm5,%%xmm7\n\t" \
 /*xmm5 = d5 c5 b5 a5 d4 c4 b4 a4*/ \
 "punpckldq %%xmm2,%%xmm5\n\t" \
 /*xmm7 = d7 c7 b7 a7 d6 c6 b6 a6*/ \
 "punpckhdq %%xmm2,%%xmm7\n\t" \
 /*xmm2 is free.*/ \
 "movdqa %%xmm4,%%xmm2\n\t" \
 /*xmm4 = h3 g3 f3 e3 h2 g2 f2 e2*/ \
 "punpckhdq %%xmm6,%%xmm4\n\t" \
 /*xmm2 = h1 g1 f1 e1 h0 g0 f0 e0*/ \
 "punpckldq %%xmm6,%%xmm2\n\t" \
 /*xmm6 is free.*/ \
 "movdqa %%xmm8,%%xmm6\n\t" \
 /*xmm6 = h5 g5 f5 e5 h4 g4 f4 e4*/ \
 "punpckldq %%xmm1,%%xmm6\n\t" \
 /*xmm8 = h7 g7 f7 e7 h6 g6 f6 e6*/ \
 "punpckhdq %%xmm1,%%xmm8\n\t" \
 /*xmm1 is free.*/ \
 "movdqa %%xmm0,%%xmm1\n\t" \
 /*xmm0 = h0 g0 f0 e0 d0 c0 b0 a0*/ \
 "punpcklqdq %%xmm2,%%xmm0\n\t" \
 /*xmm1 = h1 g1 f1 e1 d1 c1 b1 a1*/ \
 "punpckhqdq %%xmm2,%%xmm1\n\t" \
 /*xmm2 is free.*/ \
 "movdqa %%xmm3,%%xmm2\n\t" \
 /*xmm3 = h3 g3 f3 e3 d3 c3 b3 a3*/ \
 "punpckhqdq %%xmm4,%%xmm3\n\t" \
 /*xmm2 = h2 g2 f2 e2 d2 c2 b2 a2*/ \
 "punpcklqdq %%xmm4,%%xmm2\n\t" \
 /*xmm4 is free.*/ \
 "movdqa %%xmm5,%%xmm4\n\t" \
 /*xmm5 = h5 g5 f5 e5 d5 c5 b5 a5*/ \
 "punpckhqdq %%xmm6,%%xmm5\n\t" \
 /*xmm4 = h4 g4 f4 e4 d4 c4 b4 a4*/ \
 "punpcklqdq %%xmm6,%%xmm4\n\t" \
 /*xmm6 is free.*/ \
 "movdqa %%xmm7,%%xmm6\n\t" \
 /*xmm7 = h7 g7 f7 e7 d7 c7 b7 a7*/ \
 "punpckhqdq %%xmm8,%%xmm7\n\t" \
 /*xmm6 = h6 g6 f6 e6 d6 c6 b6 a6*/ \
 "punpcklqdq %%xmm8,%%xmm6\n\t" \
 /*xmm8 is free.*/ \

# else
/*Otherwise, we need to spill some values to %[buf] temporarily.
  Again, the butterflies are carefully arranged to get the columns to come out
   in order, minimizing register spills and maximizing the delay between a load
   and when the value loaded is actually used.*/
#  define OC_TRANSPOSE_8x8 \
 "#OC_TRANSPOSE_8x8\n\t" \
 /*buf[0] = a7 a6 a5 a4 a3 a2 a1 a0*/ \
 "movdqa %%xmm0,"OC_MEM_OFFS(0x00,buf)"\n\t" \
 /*xmm0 is free.*/ \
 "movdqa %%xmm2,%%xmm0\n\t" \
 /*xmm2 = d7 c7 d6 c6 d5 c5 d4 c4*/ \
 "punpckhwd %%xmm3,%%xmm2\n\t" \
 /*xmm0 = d3 c3 d2 c2 d1 c1 d0 c0*/ \
 "punpcklwd %%xmm3,%%xmm0\n\t" \
 /*xmm3 = a7 a6 a5 a4 a3 a2 a1 a0*/ \
 "movdqa "OC_MEM_OFFS(0x00,buf)",%%xmm3\n\t" \
 /*buf[1] = d7 c7 d6 c6 d5 c5 d4 c4*/ \
 "movdqa %%xmm2,"OC_MEM_OFFS(0x10,buf)"\n\t" \
 /*xmm2 is free.*/ \
 "movdqa %%xmm6,%%xmm2\n\t" \
 /*xmm6 = h3 g3 h2 g2 h1 g1 h0 g0*/ \
 "punpcklwd %%xmm7,%%xmm6\n\t" \
 /*xmm2 = h7 g7 h6 g6 h5 g5 h4 g4*/ \
 "punpckhwd %%xmm7,%%xmm2\n\t" \
 /*xmm7 is free.*/ \
 "movdqa %%xmm4,%%xmm7\n\t" \
 /*xmm4 = f3 e3 f2 e2 f1 e1 f0 e0*/ \
 "punpcklwd %%xmm5,%%xmm4\n\t" \
 /*xmm7 = f7 e7 f6 e6 f5 e5 f4 e4*/ \
 "punpckhwd %%xmm5,%%xmm7\n\t" \
 /*xmm5 is free.*/ \
 "movdqa %%xmm3,%%xmm5\n\t" \
 /*xmm3 = b3 a3 b2 a2 b1 a1 b0 a0*/ \
 "punpcklwd %%xmm1,%%xmm3\n\t" \
 /*xmm5 = b7 a7 b6 a6 b5 a5 b4 a4*/ \
 "punpckhwd %%xmm1,%%xmm5\n\t" \
 /*xmm1 is free.*/ \
 "movdqa %%xmm7,%%xmm1\n\t" \
 /*xmm7 = h5 g5 f5 e5 h4 g4 f4 e4*/ \
 "punpckldq %%xmm2,%%xmm7\n\t" \
 /*xmm1 = h7 g7 f7 e7 h6 g6 f6 e6*/ \
 "punpckhdq %%xmm2,%%xmm1\n\t" \
 /*xmm2 = d7 c7 d6 c6 d5 c5 d4 c4*/ \
 "movdqa "OC_MEM_OFFS(0x10,buf)",%%xmm2\n\t" \
 /*buf[0] = h7 g7 f7 e7 h6 g6 f6 e6*/ \
 "movdqa %%xmm1,"OC_MEM_OFFS(0x00,buf)"\n\t" \
 /*xmm1 is free.*/ \
 "movdqa %%xmm3,%%xmm1\n\t" \
 /*xmm3 = d3 c3 b3 a3 d2 c2 b2 a2*/ \
 "punpckhdq %%xmm0,%%xmm3\n\t" \
 /*xmm1 = d1 c1 b1 a1 d0 c0 b0 a0*/ \
 "punpckldq %%xmm0,%%xmm1\n\t" \
 /*xmm0 is free.*/ \
 "movdqa %%xmm4,%%xmm0\n\t" \
 /*xmm4 = h3 g3 f3 e3 h2 g2 f2 e2*/ \
 "punpckhdq %%xmm6,%%xmm4\n\t" \
 /*xmm0 = h1 g1 f1 e1 h0 g0 f0 e0*/ \
 "punpckldq %%xmm6,%%xmm0\n\t" \
 /*xmm6 is free.*/ \
 "movdqa %%xmm5,%%xmm6\n\t" \
 /*xmm5 = d5 c5 b5 a5 d4 c4 b4 a4*/ \
 "punpckldq %%xmm2,%%xmm5\n\t" \
 /*xmm6 = d7 c7 b7 a7 d6 c6 b6 a6*/ \
 "punpckhdq %%xmm2,%%xmm6\n\t" \
 /*xmm2 is free.*/ \
 "movdqa %%xmm1,%%xmm2\n\t" \
 /*xmm1 = h1 g1 f1 e1 d1 c1 b1 a1*/ \
 "punpckhqdq %%xmm0,%%xmm1\n\t" \
 /*xmm2 = h0 g0 f0 e0 d0 c0 b0 a0*/ \
 "punpcklqdq %%xmm0,%%xmm2\n\t" \
 /*xmm0 = h7 g7 f7 e7 h6 g6 f6 e6*/ \
 "movdqa "OC_MEM_OFFS(0x00,buf)",%%xmm0\n\t" \
 /*buf[1] = h0 g0 f0 e0 d0 c0 b0 a0*/ \
 "movdqa %%xmm2,"OC_MEM_OFFS(0x10,buf)"\n\t" \
 /*xmm2 is free.*/ \
 "movdqa %%xmm3,%%xmm2\n\t" \
 /*xmm3 = h3 g3 f3 e3 d3 c3 b3 a3*/ \
 "punpckhqdq %%xmm4,%%xmm3\n\t" \
 /*xmm2 = h2 g2 f2 e2 d2 c2 b2 a2*/ \
 "punpcklqdq %%xmm4,%%xmm2\n\t" \
 /*xmm4 is free.*/ \
 "movdqa %%xmm5,%%xmm4\n\t" \
 /*xmm5 = h5 g5 f5 e5 d5 c5 b5 a5*/ \
 "punpckhqdq %%xmm7,%%xmm5\n\t" \
 /*xmm4 = h4 g4 f4 e4 d4 c4 b4 a4*/ \
 "punpcklqdq %%xmm7,%%xmm4\n\t" \
 /*xmm7 is free.*/ \
 "movdqa %%xmm6,%%xmm7\n\t" \
 /*xmm6 = h6 g6 f6 e6 d6 c6 b6 a6*/ \
 "punpcklqdq %%xmm0,%%xmm6\n\t" \
 /*xmm7 = h7 g7 f7 e7 d7 c7 b7 a7*/ \
 "punpckhqdq %%xmm0,%%xmm7\n\t" \
 /*xmm0 = h0 g0 f0 e0 d0 c0 b0 a0*/ \
 "movdqa "OC_MEM_OFFS(0x10,buf)",%%xmm0\n\t" \

# endif

/*Transpose 4 values in each of 8 MMX registers into 8 values in the first
   four SSE registers.
  No need to be clever here; we have plenty of room.*/
#  define OC_TRANSPOSE_8x4_MMX2SSE \
 "#OC_TRANSPOSE_8x4_MMX2SSE\n\t" \
 "movq2dq %%mm0,%%xmm0\n\t" \
 "movq2dq %%mm1,%%xmm1\n\t" \
 /*xmmA = b3 a3 b2 a2 b1 a1 b0 a0*/ \
 "punpcklwd %%xmm1,%%xmm0\n\t" \
 "movq2dq %%mm2,%%xmm3\n\t" \
 "movq2dq %%mm3,%%xmm2\n\t" \
 /*xmmC = d3 c3 d2 c2 d1 c1 d0 c0*/ \
 "punpcklwd %%xmm2,%%xmm3\n\t" \
 "movq2dq %%mm4,%%xmm4\n\t" \
 "movq2dq %%mm5,%%xmm5\n\t" \
 /*xmmE = f3 e3 f2 e2 f1 e1 f0 e0*/ \
 "punpcklwd %%xmm5,%%xmm4\n\t" \
 "movq2dq %%mm6,%%xmm7\n\t" \
 "movq2dq %%mm7,%%xmm6\n\t" \
 /*xmmG = h3 g3 h2 g2 h1 g1 h0 g0*/ \
 "punpcklwd %%xmm6,%%xmm7\n\t" \
 "movdqa %%xmm0,%%xmm2\n\t" \
 /*xmm0 = d1 c1 b1 a1 d0 c0 b0 a0*/ \
 "punpckldq %%xmm3,%%xmm0\n\t" \
 /*xmm2 = d3 c3 b3 a3 d2 c2 b2 a2*/ \
 "punpckhdq %%xmm3,%%xmm2\n\t" \
 "movdqa %%xmm4,%%xmm5\n\t" \
 /*xmm4 = h1 g1 f1 e1 h0 g0 f0 e0*/ \
 "punpckldq %%xmm7,%%xmm4\n\t" \
 /*xmm3 = h3 g3 f3 e3 h2 g2 f2 e2*/ \
 "punpckhdq %%xmm7,%%xmm5\n\t" \
 "movdqa %%xmm0,%%xmm1\n\t" \
 /*xmm0 = h0 g0 f0 e0 d0 c0 b0 a0*/ \
 "punpcklqdq %%xmm4,%%xmm0\n\t" \
 /*xmm1 = h1 g1 f1 e1 d1 c1 b1 a1*/ \
 "punpckhqdq %%xmm4,%%xmm1\n\t" \
 "movdqa %%xmm2,%%xmm3\n\t" \
 /*xmm2 = h2 g2 f2 e2 d2 c2 b2 a2*/ \
 "punpcklqdq %%xmm5,%%xmm2\n\t" \
 /*xmm3 = h3 g3 f3 e3 d3 c3 b3 a3*/ \
 "punpckhqdq %%xmm5,%%xmm3\n\t" \

#endif
