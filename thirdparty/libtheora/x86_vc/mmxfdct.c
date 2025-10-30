/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 1999-2006                *
 * by the Xiph.Org Foundation https://www.xiph.org/                 *
 *                                                                  *
 ********************************************************************/ 
 /*MMX fDCT implementation for x86_32*/
/*$Id: fdct_ses2.c 14579 2008-03-12 06:42:40Z xiphmont $*/
#include "x86enc.h"
#include "x86zigzag.h"

#if defined(OC_X86_ASM)

#define OC_FDCT_STAGE1_8x4  __asm{ \
  /*Stage 1:*/ \
  /*mm0=t7'=t0-t7*/ \
  __asm  psubw mm0,mm7 \
  __asm  paddw mm7,mm7 \
  /*mm1=t6'=t1-t6*/ \
  __asm  psubw mm1, mm6 \
  __asm  paddw mm6,mm6 \
  /*mm2=t5'=t2-t5*/ \
  __asm  psubw mm2,mm5 \
  __asm  paddw mm5,mm5 \
  /*mm3=t4'=t3-t4*/ \
  __asm  psubw mm3,mm4 \
  __asm  paddw mm4,mm4 \
  /*mm7=t0'=t0+t7*/ \
  __asm  paddw mm7,mm0 \
  /*mm6=t1'=t1+t6*/  \
  __asm  paddw mm6,mm1 \
  /*mm5=t2'=t2+t5*/ \
  __asm  paddw mm5,mm2 \
  /*mm4=t3'=t3+t4*/ \
  __asm  paddw mm4,mm3\
}

#define OC_FDCT8x4(_r0,_r1,_r2,_r3,_r4,_r5,_r6,_r7) __asm{ \
  /*Stage 2:*/ \
  /*mm7=t3''=t0'-t3'*/ \
  __asm  psubw mm7,mm4 \
  __asm  paddw mm4,mm4 \
  /*mm6=t2''=t1'-t2'*/ \
  __asm  psubw mm6,mm5 \
  __asm  movq [Y+_r6],mm7 \
  __asm  paddw mm5,mm5 \
  /*mm1=t5''=t6'-t5'*/ \
  __asm  psubw mm1,mm2 \
  __asm  movq [Y+_r2],mm6 \
  /*mm4=t0''=t0'+t3'*/ \
  __asm  paddw mm4,mm7 \
  __asm  paddw mm2,mm2 \
  /*mm5=t1''=t1'+t2'*/ \
  __asm  movq [Y+_r0],mm4 \
  __asm  paddw mm5,mm6 \
  /*mm2=t6''=t6'+t5'*/ \
  __asm  paddw mm2,mm1 \
  __asm  movq [Y+_r4],mm5 \
  /*mm0=t7', mm1=t5'', mm2=t6'', mm3=t4'.*/ \
  /*mm4, mm5, mm6, mm7 are free.*/ \
  /*Stage 3:*/ \
  /*mm6={2}x4, mm7={27146,0xB500>>1}x2*/ \
  __asm  mov A,0x5A806A0A \
  __asm  pcmpeqb mm6,mm6 \
  __asm  movd mm7,A \
  __asm  psrlw mm6,15 \
  __asm  punpckldq mm7,mm7 \
  __asm  paddw mm6,mm6 \
  /*mm0=0, m2={-1}x4 \
    mm5:mm4=t5''*27146+0xB500*/ \
  __asm  movq mm4,mm1 \
  __asm  movq mm5,mm1 \
  __asm  punpcklwd mm4,mm6 \
  __asm  movq [Y+_r3],mm2 \
  __asm  pmaddwd mm4,mm7 \
  __asm  movq [Y+_r7],mm0 \
  __asm  punpckhwd mm5,mm6 \
  __asm  pxor mm0,mm0 \
  __asm  pmaddwd mm5,mm7 \
  __asm  pcmpeqb mm2,mm2 \
  /*mm2=t6'', mm1=t5''+(t5''!=0) \
    mm4=(t5''*27146+0xB500>>16)*/ \
  __asm  pcmpeqw mm0,mm1 \
  __asm  psrad mm4,16 \
  __asm  psubw mm0,mm2 \
  __asm  movq mm2, [Y+_r3] \
  __asm  psrad mm5,16 \
  __asm  paddw mm1,mm0 \
  __asm  packssdw mm4,mm5 \
  /*mm4=s=(t5''*27146+0xB500>>16)+t5''+(t5''!=0)>>1*/ \
  __asm  paddw mm4,mm1 \
  __asm  movq mm0, [Y+_r7] \
  __asm  psraw mm4,1 \
  __asm  movq mm1,mm3 \
  /*mm3=t4''=t4'+s*/ \
  __asm  paddw mm3,mm4 \
  /*mm1=t5'''=t4'-s*/ \
  __asm  psubw mm1,mm4 \
  /*mm1=0, mm3={-1}x4 \
    mm5:mm4=t6''*27146+0xB500*/ \
  __asm  movq mm4,mm2 \
  __asm  movq mm5,mm2 \
  __asm  punpcklwd mm4,mm6 \
  __asm  movq [Y+_r5],mm1 \
  __asm  pmaddwd mm4,mm7 \
  __asm  movq [Y+_r1],mm3 \
  __asm  punpckhwd mm5,mm6 \
  __asm  pxor mm1,mm1 \
  __asm  pmaddwd mm5,mm7 \
  __asm  pcmpeqb mm3,mm3 \
  /*mm2=t6''+(t6''!=0), mm4=(t6''*27146+0xB500>>16)*/ \
  __asm  psrad mm4,16 \
  __asm  pcmpeqw mm1,mm2 \
  __asm  psrad mm5,16 \
  __asm  psubw mm1,mm3 \
  __asm  packssdw mm4,mm5 \
  __asm  paddw mm2,mm1 \
  /*mm1=t1'' \
    mm4=s=(t6''*27146+0xB500>>16)+t6''+(t6''!=0)>>1*/ \
  __asm  paddw mm4,mm2 \
  __asm  movq mm1,[Y+_r4] \
  __asm  psraw mm4,1 \
  __asm  movq mm2,mm0 \
  /*mm7={54491-0x7FFF,0x7FFF}x2 \
    mm0=t7''=t7'+s*/ \
  __asm  paddw mm0,mm4 \
  /*mm2=t6'''=t7'-s*/ \
  __asm  psubw mm2,mm4 \
  /*Stage 4:*/ \
  /*mm0=0, mm2=t0'' \
    mm5:mm4=t1''*27146+0xB500*/ \
  __asm  movq mm4,mm1 \
  __asm  movq mm5,mm1 \
  __asm  punpcklwd mm4,mm6 \
  __asm  movq [Y+_r3],mm2 \
  __asm  pmaddwd mm4,mm7 \
  __asm  movq mm2,[Y+_r0] \
  __asm  punpckhwd mm5,mm6 \
  __asm  movq [Y+_r7],mm0 \
  __asm  pmaddwd mm5,mm7 \
  __asm  pxor mm0,mm0 \
  /*mm7={27146,0x4000>>1}x2 \
    mm0=s=(t1''*27146+0xB500>>16)+t1''+(t1''!=0)*/ \
  __asm  psrad mm4,16 \
  __asm  mov A,0x20006A0A \
  __asm  pcmpeqw mm0,mm1 \
  __asm  movd mm7,A \
  __asm  psrad mm5,16 \
  __asm  psubw mm0,mm3 \
  __asm  packssdw mm4,mm5 \
  __asm  paddw mm0,mm1 \
  __asm  punpckldq mm7,mm7 \
  __asm  paddw mm0,mm4 \
  /*mm6={0x00000E3D}x2 \
    mm1=-(t0''==0), mm5:mm4=t0''*27146+0x4000*/ \
  __asm  movq mm4,mm2 \
  __asm  movq mm5,mm2 \
  __asm  punpcklwd mm4,mm6 \
  __asm  mov A,0x0E3D \
  __asm  pmaddwd mm4,mm7 \
  __asm  punpckhwd mm5,mm6 \
  __asm  movd mm6,A \
  __asm  pmaddwd mm5,mm7 \
  __asm  pxor mm1,mm1 \
  __asm  punpckldq mm6,mm6 \
  __asm  pcmpeqw mm1,mm2 \
  /*mm4=r=(t0''*27146+0x4000>>16)+t0''+(t0''!=0)*/ \
  __asm  psrad mm4,16 \
  __asm  psubw mm1,mm3 \
  __asm  psrad mm5,16 \
  __asm  paddw mm2,mm1 \
  __asm  packssdw mm4,mm5 \
  __asm  movq mm1,[Y+_r5] \
  __asm  paddw mm4,mm2 \
  /*mm2=t6'', mm0=_y[0]=u=r+s>>1 \
    The naive implementation could cause overflow, so we use \
     u=(r&s)+((r^s)>>1).*/ \
  __asm  movq mm2,[Y+_r3] \
  __asm  movq mm7,mm0 \
  __asm  pxor mm0,mm4 \
  __asm  pand mm7,mm4 \
  __asm  psraw mm0,1 \
  __asm  mov A,0x7FFF54DC \
  __asm  paddw mm0,mm7 \
  __asm  movd mm7,A \
  /*mm7={54491-0x7FFF,0x7FFF}x2 \
    mm4=_y[4]=v=r-u*/ \
  __asm  psubw mm4,mm0 \
  __asm  punpckldq mm7,mm7 \
  __asm  movq [Y+_r4],mm4 \
  /*mm0=0, mm7={36410}x4 \
    mm1=(t5'''!=0), mm5:mm4=54491*t5'''+0x0E3D*/ \
  __asm  movq mm4,mm1 \
  __asm  movq mm5,mm1 \
  __asm  punpcklwd mm4,mm1 \
  __asm  mov A,0x8E3A8E3A \
  __asm  pmaddwd mm4,mm7 \
  __asm  movq [Y+_r0],mm0 \
  __asm  punpckhwd mm5,mm1 \
  __asm  pxor mm0,mm0 \
  __asm  pmaddwd mm5,mm7 \
  __asm  pcmpeqw mm1,mm0 \
  __asm  movd mm7,A \
  __asm  psubw mm1,mm3 \
  __asm  punpckldq mm7,mm7 \
  __asm  paddd mm4,mm6 \
  __asm  paddd mm5,mm6 \
  /*mm0=0 \
    mm3:mm1=36410*t6'''+((t5'''!=0)<<16)*/ \
  __asm  movq mm6,mm2 \
  __asm  movq mm3,mm2 \
  __asm  pmulhw mm6,mm7 \
  __asm  paddw mm1,mm2 \
  __asm  pmullw mm3,mm7 \
  __asm  pxor mm0,mm0 \
  __asm  paddw mm6,mm1 \
  __asm  movq mm1,mm3 \
  __asm  punpckhwd mm3,mm6 \
  __asm  punpcklwd mm1,mm6 \
  /*mm3={-1}x4, mm6={1}x4 \
    mm4=_y[5]=u=(54491*t5'''+36410*t6'''+0x0E3D>>16)+(t5'''!=0)*/ \
  __asm  paddd mm5,mm3 \
  __asm  paddd mm4,mm1 \
  __asm  psrad mm5,16 \
  __asm  pxor mm6,mm6 \
  __asm  psrad mm4,16 \
  __asm  pcmpeqb mm3,mm3 \
  __asm  packssdw mm4,mm5 \
  __asm  psubw mm6,mm3 \
  /*mm1=t7'', mm7={26568,0x3400}x2 \
    mm2=s=t6'''-(36410*u>>16)*/ \
  __asm  movq mm1,mm4 \
  __asm  mov A,0x340067C8 \
  __asm  pmulhw mm4,mm7 \
  __asm  movd mm7,A \
  __asm  movq [Y+_r5],mm1 \
  __asm  punpckldq mm7,mm7 \
  __asm  paddw mm4,mm1 \
  __asm  movq mm1,[Y+_r7] \
  __asm  psubw mm2,mm4 \
  /*mm6={0x00007B1B}x2 \
    mm0=(s!=0), mm5:mm4=s*26568+0x3400*/ \
  __asm  movq mm4,mm2 \
  __asm  movq mm5,mm2 \
  __asm  punpcklwd mm4,mm6 \
  __asm  pcmpeqw mm0,mm2 \
  __asm  pmaddwd mm4,mm7 \
  __asm  mov A,0x7B1B \
  __asm  punpckhwd mm5,mm6 \
  __asm  movd mm6,A \
  __asm  pmaddwd mm5,mm7 \
  __asm  psubw mm0,mm3 \
  __asm  punpckldq mm6,mm6 \
  /*mm7={64277-0x7FFF,0x7FFF}x2 \
    mm2=_y[3]=v=(s*26568+0x3400>>17)+s+(s!=0)*/ \
  __asm  psrad mm4,17 \
  __asm  paddw mm2,mm0 \
  __asm  psrad mm5,17 \
  __asm  mov A,0x7FFF7B16 \
  __asm  packssdw mm4,mm5 \
  __asm  movd mm7,A \
  __asm  paddw mm2,mm4 \
  __asm  punpckldq mm7,mm7 \
  /*mm0=0, mm7={12785}x4 \
    mm1=(t7''!=0), mm2=t4'', mm5:mm4=64277*t7''+0x7B1B*/ \
  __asm  movq mm4,mm1 \
  __asm  movq mm5,mm1 \
  __asm  movq [Y+_r3],mm2 \
  __asm  punpcklwd mm4,mm1 \
  __asm  movq mm2,[Y+_r1] \
  __asm  pmaddwd mm4,mm7 \
  __asm  mov A,0x31F131F1 \
  __asm  punpckhwd mm5,mm1 \
  __asm  pxor mm0,mm0 \
  __asm  pmaddwd mm5,mm7 \
  __asm  pcmpeqw mm1,mm0 \
  __asm  movd mm7,A \
  __asm  psubw mm1,mm3 \
  __asm  punpckldq mm7,mm7 \
  __asm  paddd mm4,mm6 \
  __asm  paddd mm5,mm6 \
  /*mm3:mm1=12785*t4'''+((t7''!=0)<<16)*/ \
  __asm  movq mm6,mm2 \
  __asm  movq mm3,mm2 \
  __asm  pmulhw mm6,mm7 \
  __asm  pmullw mm3,mm7 \
  __asm  paddw mm6,mm1 \
  __asm  movq mm1,mm3 \
  __asm  punpckhwd mm3,mm6 \
  __asm  punpcklwd mm1,mm6 \
  /*mm3={-1}x4, mm6={1}x4 \
    mm4=_y[1]=u=(12785*t4'''+64277*t7''+0x7B1B>>16)+(t7''!=0)*/ \
  __asm  paddd mm5,mm3 \
  __asm  paddd mm4,mm1 \
  __asm  psrad mm5,16 \
  __asm  pxor mm6,mm6 \
  __asm  psrad mm4,16 \
  __asm  pcmpeqb mm3,mm3 \
  __asm  packssdw mm4,mm5 \
  __asm  psubw mm6,mm3 \
  /*mm1=t3'', mm7={20539,0x3000}x2 \
    mm4=s=(12785*u>>16)-t4''*/ \
  __asm  movq [Y+_r1],mm4 \
  __asm  pmulhw mm4,mm7 \
  __asm  mov A,0x3000503B \
  __asm  movq mm1,[Y+_r6] \
  __asm  movd mm7,A \
  __asm  psubw mm4,mm2 \
  __asm  punpckldq mm7,mm7 \
  /*mm6={0x00006CB7}x2 \
    mm0=(s!=0), mm5:mm4=s*20539+0x3000*/ \
  __asm  movq mm5,mm4 \
  __asm  movq mm2,mm4 \
  __asm  punpcklwd mm4,mm6 \
  __asm  pcmpeqw mm0,mm2 \
  __asm  pmaddwd mm4,mm7 \
  __asm  mov A,0x6CB7 \
  __asm  punpckhwd mm5,mm6 \
  __asm  movd mm6,A \
  __asm  pmaddwd mm5,mm7 \
  __asm  psubw mm0,mm3 \
  __asm  punpckldq mm6,mm6 \
  /*mm7={60547-0x7FFF,0x7FFF}x2 \
    mm2=_y[7]=v=(s*20539+0x3000>>20)+s+(s!=0)*/ \
  __asm  psrad mm4,20 \
  __asm  paddw mm2,mm0 \
  __asm  psrad mm5,20 \
  __asm  mov A,0x7FFF6C84 \
  __asm  packssdw mm4,mm5 \
  __asm  movd mm7,A \
  __asm  paddw mm2,mm4 \
  __asm  punpckldq mm7,mm7 \
  /*mm0=0, mm7={25080}x4 \
    mm2=t2'', mm5:mm4=60547*t3''+0x6CB7*/ \
  __asm  movq mm4,mm1 \
  __asm  movq mm5,mm1 \
  __asm  movq [Y+_r7],mm2 \
  __asm  punpcklwd mm4,mm1 \
  __asm  movq mm2,[Y+_r2] \
  __asm  pmaddwd mm4,mm7 \
  __asm  mov A,0x61F861F8 \
  __asm  punpckhwd mm5,mm1 \
  __asm  pxor mm0,mm0 \
  __asm  pmaddwd mm5,mm7 \
  __asm  movd mm7,A \
  __asm  pcmpeqw mm1,mm0 \
  __asm  psubw mm1,mm3 \
  __asm  punpckldq mm7,mm7 \
  __asm  paddd mm4,mm6 \
  __asm  paddd mm5,mm6 \
  /*mm3:mm1=25080*t2''+((t3''!=0)<<16)*/ \
  __asm  movq mm6,mm2 \
  __asm  movq mm3,mm2 \
  __asm  pmulhw mm6,mm7 \
  __asm  pmullw mm3,mm7 \
  __asm  paddw mm6,mm1 \
  __asm  movq mm1,mm3 \
  __asm  punpckhwd mm3,mm6 \
  __asm  punpcklwd mm1,mm6 \
  /*mm1={-1}x4 \
    mm4=u=(25080*t2''+60547*t3''+0x6CB7>>16)+(t3''!=0)*/ \
  __asm  paddd mm5,mm3 \
  __asm  paddd mm4,mm1 \
  __asm  psrad mm5,16 \
  __asm  mov A,0x28005460 \
  __asm  psrad mm4,16 \
  __asm  pcmpeqb mm1,mm1 \
  __asm  packssdw mm4,mm5 \
  /*mm5={1}x4, mm6=_y[2]=u, mm7={21600,0x2800}x2 \
    mm4=s=(25080*u>>16)-t2''*/ \
  __asm  movq mm6,mm4 \
  __asm  pmulhw mm4,mm7 \
  __asm  pxor mm5,mm5 \
  __asm  movd mm7,A \
  __asm  psubw mm5,mm1 \
  __asm  punpckldq mm7,mm7 \
  __asm  psubw mm4,mm2 \
  /*mm2=s+(s!=0) \
    mm4:mm3=s*21600+0x2800*/ \
  __asm  movq mm3,mm4 \
  __asm  movq mm2,mm4 \
  __asm  punpckhwd mm4,mm5 \
  __asm  pcmpeqw mm0,mm2 \
  __asm  pmaddwd mm4,mm7 \
  __asm  psubw mm0,mm1 \
  __asm  punpcklwd mm3,mm5 \
  __asm  paddw mm2,mm0 \
  __asm  pmaddwd mm3,mm7 \
  /*mm0=_y[4], mm1=_y[7], mm4=_y[0], mm5=_y[5] \
    mm3=_y[6]=v=(s*21600+0x2800>>18)+s+(s!=0)*/ \
  __asm  movq mm0,[Y+_r4] \
  __asm  psrad mm4,18 \
  __asm  movq mm5,[Y+_r5] \
  __asm  psrad mm3,18 \
  __asm  movq mm1,[Y+_r7] \
  __asm  packssdw mm3,mm4 \
  __asm  movq mm4,[Y+_r0] \
  __asm  paddw mm3,mm2 \
}

/*On input, mm4=_y[0], mm6=_y[2], mm0=_y[4], mm5=_y[5], mm3=_y[6], mm1=_y[7].
  On output, {_y[4],mm1,mm2,mm3} contains the transpose of _y[4...7] and
   {mm4,mm5,mm6,mm7} contains the transpose of _y[0...3].*/
#define OC_TRANSPOSE8x4(_r0,_r1,_r2,_r3,_r4,_r5,_r6,_r7) __asm{ \
  /*First 4x4 transpose:*/ \
  /*mm0 = e3 e2 e1 e0 \
    mm5 = f3 f2 f1 f0 \
    mm3 = g3 g2 g1 g0 \
    mm1 = h3 h2 h1 h0*/ \
  __asm  movq mm2,mm0 \
  __asm  punpcklwd mm0,mm5 \
  __asm  punpckhwd mm2,mm5 \
  __asm  movq mm5,mm3 \
  __asm  punpcklwd mm3,mm1 \
  __asm  punpckhwd mm5,mm1 \
  /*mm0 = f1 e1 f0 e0 \
    mm2 = f3 e3 f2 e2 \
    mm3 = h1 g1 h0 g0 \
    mm5 = h3 g3 h2 g2*/ \
  __asm  movq mm1,mm0 \
  __asm  punpckldq mm0,mm3 \
  __asm  movq [Y+_r4],mm0 \
  __asm  punpckhdq mm1,mm3 \
  __asm  movq mm0,[Y+_r1] \
  __asm  movq mm3,mm2 \
  __asm  punpckldq mm2,mm5 \
  __asm  punpckhdq mm3,mm5 \
  __asm  movq mm5,[Y+_r3] \
  /*_y[4] = h0 g0 f0 e0 \
   mm1  = h1 g1 f1 e1 \
   mm2  = h2 g2 f2 e2 \
   mm3  = h3 g3 f3 e3*/ \
  /*Second 4x4 transpose:*/ \
  /*mm4 = a3 a2 a1 a0 \
    mm0 = b3 b2 b1 b0 \
    mm6 = c3 c2 c1 c0 \
    mm5 = d3 d2 d1 d0*/ \
  __asm  movq mm7,mm4 \
  __asm  punpcklwd mm4,mm0 \
  __asm  punpckhwd mm7,mm0 \
  __asm  movq mm0,mm6 \
  __asm  punpcklwd mm6,mm5 \
  __asm  punpckhwd mm0,mm5 \
  /*mm4 = b1 a1 b0 a0 \
    mm7 = b3 a3 b2 a2 \
    mm6 = d1 c1 d0 c0 \
    mm0 = d3 c3 d2 c2*/ \
  __asm  movq mm5,mm4 \
  __asm  punpckldq mm4,mm6 \
  __asm  punpckhdq mm5,mm6 \
  __asm  movq mm6,mm7 \
  __asm  punpckhdq mm7,mm0 \
  __asm  punpckldq mm6,mm0 \
  /*mm4 = d0 c0 b0 a0 \
    mm5 = d1 c1 b1 a1 \
    mm6 = d2 c2 b2 a2 \
    mm7 = d3 c3 b3 a3*/ \
}

/*MMX implementation of the fDCT.*/
void oc_enc_fdct8x8_mmxext(ogg_int16_t _y[64],const ogg_int16_t _x[64]){
  OC_ALIGN8(ogg_int16_t buf[64]);
  ogg_int16_t *bufp;
  bufp=buf;
  __asm{
#define X edx
#define Y eax
#define A ecx
#define BUF esi
    /*Add two extra bits of working precision to improve accuracy; any more and
       we could overflow.*/
    /*We also add biases to correct for some systematic error that remains in
       the full fDCT->iDCT round trip.*/
    mov X, _x
    mov Y, _y
	mov BUF, bufp
    movq mm0,[0x00+X]
    movq mm1,[0x10+X]
    movq mm2,[0x20+X]
    movq mm3,[0x30+X]
    pcmpeqb mm4,mm4
    pxor mm7,mm7
    movq mm5,mm0
    psllw mm0,2
    pcmpeqw mm5,mm7
    movq mm7,[0x70+X]
    psllw mm1,2
    psubw mm5,mm4
    psllw mm2,2
    mov A,1
    pslld mm5,16
    movd mm6,A
    psllq mm5,16
    mov A,0x10001
    psllw mm3,2
    movd mm4,A
    punpckhwd mm5,mm6
    psubw mm1,mm6
    movq mm6,[0x60+X]
    paddw mm0,mm5
    movq mm5,[0x50+X]
    paddw mm0,mm4
    movq mm4,[0x40+X]
    /*We inline stage1 of the transform here so we can get better instruction
       scheduling with the shifts.*/
    /*mm0=t7'=t0-t7*/
    psllw mm7,2
    psubw mm0,mm7
    psllw mm6,2
    paddw mm7,mm7
    /*mm1=t6'=t1-t6*/
    psllw mm5,2
    psubw mm1,mm6
    psllw mm4,2
    paddw mm6,mm6
    /*mm2=t5'=t2-t5*/
    psubw mm2,mm5
    paddw mm5,mm5
    /*mm3=t4'=t3-t4*/
    psubw mm3,mm4
    paddw mm4,mm4
    /*mm7=t0'=t0+t7*/
    paddw mm7,mm0
    /*mm6=t1'=t1+t6*/
    paddw mm6,mm1
    /*mm5=t2'=t2+t5*/
    paddw mm5,mm2
    /*mm4=t3'=t3+t4*/
    paddw mm4,mm3
    OC_FDCT8x4(0x00,0x10,0x20,0x30,0x40,0x50,0x60,0x70)
    OC_TRANSPOSE8x4(0x00,0x10,0x20,0x30,0x40,0x50,0x60,0x70)
    /*Swap out this 8x4 block for the next one.*/
    movq mm0,[0x08+X]
    movq [0x30+Y],mm7
    movq mm7,[0x78+X]
    movq [0x50+Y],mm1
    movq mm1,[0x18+X]
    movq [0x20+Y],mm6
    movq mm6,[0x68+X]
    movq [0x60+Y],mm2
    movq mm2,[0x28+X]
    movq [0x10+Y],mm5
    movq mm5,[0x58+X]
    movq [0x70+Y],mm3
    movq mm3,[0x38+X]
    /*And increase its working precision, too.*/
    psllw mm0,2
    movq [0x00+Y],mm4
    psllw mm7,2
    movq mm4,[0x48+X]
    /*We inline stage1 of the transform here so we can get better instruction
       scheduling with the shifts.*/
    /*mm0=t7'=t0-t7*/
    psubw mm0,mm7
    psllw mm1,2
    paddw mm7,mm7
    psllw mm6,2
    /*mm1=t6'=t1-t6*/
    psubw mm1,mm6
    psllw mm2,2
    paddw mm6,mm6
    psllw mm5,2
    /*mm2=t5'=t2-t5*/
    psubw mm2,mm5
    psllw mm3,2
    paddw mm5,mm5
    psllw mm4,2
    /*mm3=t4'=t3-t4*/
    psubw mm3,mm4
    paddw mm4,mm4
    /*mm7=t0'=t0+t7*/
    paddw mm7,mm0
    /*mm6=t1'=t1+t6*/
    paddw mm6,mm1
    /*mm5=t2'=t2+t5*/
    paddw mm5,mm2
    /*mm4=t3'=t3+t4*/
    paddw mm4,mm3
    OC_FDCT8x4(0x08,0x18,0x28,0x38,0x48,0x58,0x68,0x78)
    OC_TRANSPOSE8x4(0x08,0x18,0x28,0x38,0x48,0x58,0x68,0x78)
    /*Here the first 4x4 block of output from the last transpose is the second
       4x4 block of input for the next transform.
      We have cleverly arranged that it already be in the appropriate place,
       so we only have to do half the stores and loads.*/
    movq mm0,[0x00+Y]
    movq [0x58+Y],mm1
    movq mm1,[0x10+Y]
    movq [0x68+Y],mm2
    movq mm2,[0x20+Y]
    movq [0x78+Y],mm3
    movq mm3,[0x30+Y]
    OC_FDCT_STAGE1_8x4
    OC_FDCT8x4(0x00,0x10,0x20,0x30,0x08,0x18,0x28,0x38)
    /*mm0={-2}x4*/
    pcmpeqw mm2,mm2
    paddw mm2,mm2
    /*Round and store the results (no transpose).*/
    movq mm7,[Y+0x10]
    psubw mm4,mm2
    psubw mm6,mm2
    psraw mm4,2
    psubw mm0,mm2
    movq [BUF+0x00],mm4
    movq mm4,[Y+0x30]
    psraw mm6,2
    psubw mm5,mm2
    movq [BUF+0x20],mm6
    psraw mm0,2
    psubw mm3,mm2
    movq [BUF+0x40],mm0
    psraw mm5,2
    psubw mm1,mm2
    movq [BUF+0x50],mm5
    psraw mm3,2
    psubw mm7,mm2
    movq [BUF+0x60],mm3
    psraw mm1,2
    psubw mm4,mm2
    movq [BUF+0x70],mm1
    psraw mm7,2
    movq [BUF+0x10],mm7
    psraw mm4,2
    movq [BUF+0x30],mm4
    /*Load the next block.*/
    movq mm0,[0x40+Y]
    movq mm7,[0x78+Y]
    movq mm1,[0x50+Y]
    movq mm6,[0x68+Y]
    movq mm2,[0x60+Y]
    movq mm5,[0x58+Y]
    movq mm3,[0x70+Y]
    movq mm4,[0x48+Y]
    OC_FDCT_STAGE1_8x4
    OC_FDCT8x4(0x40,0x50,0x60,0x70,0x48,0x58,0x68,0x78)
    /*mm0={-2}x4*/
    pcmpeqw mm2,mm2
    paddw mm2,mm2
    /*Round and store the results (no transpose).*/
    movq mm7,[Y+0x50]
    psubw mm4,mm2
    psubw mm6,mm2
    psraw mm4,2
    psubw mm0,mm2
    movq [BUF+0x08],mm4
    movq mm4,[Y+0x70]
    psraw mm6,2
    psubw mm5,mm2
    movq [BUF+0x28],mm6
    psraw mm0,2
    psubw mm3,mm2
    movq [BUF+0x48],mm0
    psraw mm5,2
    psubw mm1,mm2
    movq [BUF+0x58],mm5
    psraw mm3,2
    psubw mm7,mm2
    movq [BUF+0x68],mm3
    psraw mm1,2
    psubw mm4,mm2
    movq [BUF+0x78],mm1
    psraw mm7,2
    movq [BUF+0x18],mm7
    psraw mm4,2
    movq [BUF+0x38],mm4
#define OC_ZZ_LOAD_ROW_LO(_row,_reg) \
    __asm movq _reg,[BUF+16*(_row)] \

#define OC_ZZ_LOAD_ROW_HI(_row,_reg) \
    __asm movq _reg,[BUF+16*(_row)+8] \

    OC_TRANSPOSE_ZIG_ZAG_MMXEXT
#undef OC_ZZ_LOAD_ROW_LO
#undef OC_ZZ_LOAD_ROW_HI
#undef X
#undef Y
#undef A
#undef BUF
  }
}

#endif
