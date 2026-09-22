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
/*SSE2 fDCT implementation for x86_64.*/
/*$Id: fdct_ses2.c 14579 2008-03-12 06:42:40Z xiphmont $*/
#include <stddef.h>
#include "x86enc.h"
#include "x86zigzag.h"
#include "sse2trans.h"

#if defined(OC_X86_64_ASM)

# define OC_FDCT_8x8 \
 /*Note: xmm15={0}x8 and xmm14={-1}x8.*/ \
 "#OC_FDCT_8x8\n\t" \
 /*Stage 1:*/ \
 "movdqa %%xmm0,%%xmm11\n\t" \
 "movdqa %%xmm1,%%xmm10\n\t" \
 "movdqa %%xmm2,%%xmm9\n\t" \
 "movdqa %%xmm3,%%xmm8\n\t" \
 /*xmm11=t7'=t0-t7*/ \
 "psubw %%xmm7,%%xmm11\n\t" \
 /*xmm10=t6'=t1-t6*/ \
 "psubw %%xmm6,%%xmm10\n\t" \
 /*xmm9=t5'=t2-t5*/ \
 "psubw %%xmm5,%%xmm9\n\t" \
 /*xmm8=t4'=t3-t4*/ \
 "psubw %%xmm4,%%xmm8\n\t" \
 /*xmm0=t0'=t0+t7*/ \
 "paddw %%xmm7,%%xmm0\n\t" \
 /*xmm1=t1'=t1+t6*/ \
 "paddw %%xmm6,%%xmm1\n\t" \
 /*xmm5=t2'=t2+t5*/ \
 "paddw %%xmm2,%%xmm5\n\t" \
 /*xmm4=t3'=t3+t4*/ \
 "paddw %%xmm3,%%xmm4\n\t" \
 /*xmm2,3,6,7 are now free.*/ \
 /*Stage 2:*/ \
 "movdqa %%xmm0,%%xmm3\n\t" \
 "mov $0x5A806A0A,%[a]\n\t" \
 "movdqa %%xmm1,%%xmm2\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "movdqa %%xmm10,%%xmm6\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 /*xmm2=t2''=t1'-t2'*/ \
 "psubw %%xmm5,%%xmm2\n\t" \
 "pxor %%xmm12,%%xmm12\n\t" \
 /*xmm3=t3''=t0'-t3'*/ \
 "psubw %%xmm4,%%xmm3\n\t" \
 "psubw %%xmm14,%%xmm12\n\t" \
 /*xmm10=t5''=t6'-t5'*/ \
 "psubw %%xmm9,%%xmm10\n\t" \
 "paddw %%xmm12,%%xmm12\n\t" \
 /*xmm4=t0''=t0'+t3'*/ \
 "paddw %%xmm0,%%xmm4\n\t" \
 /*xmm1=t1''=t1'+t2'*/ \
 "paddw %%xmm5,%%xmm1\n\t" \
 /*xmm6=t6''=t6'+t5'*/ \
 "paddw %%xmm9,%%xmm6\n\t" \
 /*xmm0,xmm5,xmm9 are now free.*/ \
 /*Stage 3:*/ \
 /*xmm10:xmm5=t5''*27146+0xB500 \
   xmm0=t5''*/ \
 "movdqa %%xmm10,%%xmm5\n\t" \
 "movdqa %%xmm10,%%xmm0\n\t" \
 "punpckhwd %%xmm12,%%xmm10\n\t" \
 "pmaddwd %%xmm13,%%xmm10\n\t" \
 "punpcklwd %%xmm12,%%xmm5\n\t" \
 "pmaddwd %%xmm13,%%xmm5\n\t" \
 /*xmm5=(t5''*27146+0xB500>>16)+t5''*/ \
 "psrad $16,%%xmm10\n\t" \
 "psrad $16,%%xmm5\n\t" \
 "packssdw %%xmm10,%%xmm5\n\t" \
 "paddw %%xmm0,%%xmm5\n\t" \
 /*xmm0=s=(t5''*27146+0xB500>>16)+t5''+(t5''!=0)>>1*/ \
 "pcmpeqw %%xmm15,%%xmm0\n\t" \
 "psubw %%xmm14,%%xmm0\n\t" \
 "paddw %%xmm5,%%xmm0\n\t" \
 "movdqa %%xmm8,%%xmm5\n\t" \
 "psraw $1,%%xmm0\n\t" \
 /*xmm5=t5'''=t4'-s*/ \
 "psubw %%xmm0,%%xmm5\n\t" \
 /*xmm8=t4''=t4'+s*/ \
 "paddw %%xmm0,%%xmm8\n\t" \
 /*xmm0,xmm7,xmm9,xmm10 are free.*/ \
 /*xmm7:xmm9=t6''*27146+0xB500*/ \
 "movdqa %%xmm6,%%xmm7\n\t" \
 "movdqa %%xmm6,%%xmm9\n\t" \
 "punpckhwd %%xmm12,%%xmm7\n\t" \
 "pmaddwd %%xmm13,%%xmm7\n\t" \
 "punpcklwd %%xmm12,%%xmm9\n\t" \
 "pmaddwd %%xmm13,%%xmm9\n\t" \
 /*xmm9=(t6''*27146+0xB500>>16)+t6''*/ \
 "psrad $16,%%xmm7\n\t" \
 "psrad $16,%%xmm9\n\t" \
 "packssdw %%xmm7,%%xmm9\n\t" \
 "paddw %%xmm6,%%xmm9\n\t" \
 /*xmm9=s=(t6''*27146+0xB500>>16)+t6''+(t6''!=0)>>1*/ \
 "pcmpeqw %%xmm15,%%xmm6\n\t" \
 "psubw %%xmm14,%%xmm6\n\t" \
 "paddw %%xmm6,%%xmm9\n\t" \
 "movdqa %%xmm11,%%xmm7\n\t" \
 "psraw $1,%%xmm9\n\t" \
 /*xmm7=t6'''=t7'-s*/ \
 "psubw %%xmm9,%%xmm7\n\t" \
 /*xmm9=t7''=t7'+s*/ \
 "paddw %%xmm11,%%xmm9\n\t" \
 /*xmm0,xmm6,xmm10,xmm11 are free.*/ \
 /*Stage 4:*/ \
 /*xmm10:xmm0=t1''*27146+0xB500*/ \
 "movdqa %%xmm1,%%xmm0\n\t" \
 "movdqa %%xmm1,%%xmm10\n\t" \
 "punpcklwd %%xmm12,%%xmm0\n\t" \
 "pmaddwd %%xmm13,%%xmm0\n\t" \
 "punpckhwd %%xmm12,%%xmm10\n\t" \
 "pmaddwd %%xmm13,%%xmm10\n\t" \
 /*xmm0=(t1''*27146+0xB500>>16)+t1''*/ \
 "psrad $16,%%xmm0\n\t" \
 "psrad $16,%%xmm10\n\t" \
 "mov $0x20006A0A,%[a]\n\t" \
 "packssdw %%xmm10,%%xmm0\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "paddw %%xmm1,%%xmm0\n\t" \
 /*xmm0=s=(t1''*27146+0xB500>>16)+t1''+(t1''!=0)*/ \
 "pcmpeqw %%xmm15,%%xmm1\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 "psubw %%xmm14,%%xmm1\n\t" \
 "paddw %%xmm1,%%xmm0\n\t" \
 /*xmm10:xmm4=t0''*27146+0x4000*/ \
 "movdqa %%xmm4,%%xmm1\n\t" \
 "movdqa %%xmm4,%%xmm10\n\t" \
 "punpcklwd %%xmm12,%%xmm4\n\t" \
 "pmaddwd %%xmm13,%%xmm4\n\t" \
 "punpckhwd %%xmm12,%%xmm10\n\t" \
 "pmaddwd %%xmm13,%%xmm10\n\t" \
 /*xmm4=(t0''*27146+0x4000>>16)+t0''*/ \
 "psrad $16,%%xmm4\n\t" \
 "psrad $16,%%xmm10\n\t" \
 "mov $0x6CB7,%[a]\n\t" \
 "packssdw %%xmm10,%%xmm4\n\t" \
 "movd %[a],%%xmm12\n\t" \
 "paddw %%xmm1,%%xmm4\n\t" \
 /*xmm4=r=(t0''*27146+0x4000>>16)+t0''+(t0''!=0)*/ \
 "pcmpeqw %%xmm15,%%xmm1\n\t" \
 "pshufd $00,%%xmm12,%%xmm12\n\t" \
 "psubw %%xmm14,%%xmm1\n\t" \
 "mov $0x7FFF6C84,%[a]\n\t" \
 "paddw %%xmm1,%%xmm4\n\t" \
 /*xmm0=_y[0]=u=r+s>>1 \
   The naive implementation could cause overflow, so we use \
    u=(r&s)+((r^s)>>1).*/ \
 "movdqa %%xmm0,%%xmm6\n\t" \
 "pxor %%xmm4,%%xmm0\n\t" \
 "pand %%xmm4,%%xmm6\n\t" \
 "psraw $1,%%xmm0\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "paddw %%xmm6,%%xmm0\n\t" \
 /*xmm4=_y[4]=v=r-u*/ \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 "psubw %%xmm0,%%xmm4\n\t" \
 /*xmm1,xmm6,xmm10,xmm11 are free.*/ \
 /*xmm6:xmm10=60547*t3''+0x6CB7*/ \
 "movdqa %%xmm3,%%xmm10\n\t" \
 "movdqa %%xmm3,%%xmm6\n\t" \
 "punpcklwd %%xmm3,%%xmm10\n\t" \
 "pmaddwd %%xmm13,%%xmm10\n\t" \
 "mov $0x61F861F8,%[a]\n\t" \
 "punpckhwd %%xmm3,%%xmm6\n\t" \
 "pmaddwd %%xmm13,%%xmm6\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "paddd %%xmm12,%%xmm10\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 "paddd %%xmm12,%%xmm6\n\t" \
 /*xmm1:xmm2=25080*t2'' \
   xmm12=t2''*/ \
 "movdqa %%xmm2,%%xmm11\n\t" \
 "movdqa %%xmm2,%%xmm12\n\t" \
 "pmullw %%xmm13,%%xmm2\n\t" \
 "pmulhw %%xmm13,%%xmm11\n\t" \
 "movdqa %%xmm2,%%xmm1\n\t" \
 "punpcklwd %%xmm11,%%xmm2\n\t" \
 "punpckhwd %%xmm11,%%xmm1\n\t" \
 /*xmm10=u=(25080*t2''+60547*t3''+0x6CB7>>16)+(t3''!=0)*/ \
 "paddd %%xmm2,%%xmm10\n\t" \
 "paddd %%xmm1,%%xmm6\n\t" \
 "psrad $16,%%xmm10\n\t" \
 "pcmpeqw %%xmm15,%%xmm3\n\t" \
 "psrad $16,%%xmm6\n\t" \
 "psubw %%xmm14,%%xmm3\n\t" \
 "packssdw %%xmm6,%%xmm10\n\t" \
 "paddw %%xmm3,%%xmm10\n\t" \
 /*xmm2=_y[2]=u \
   xmm10=s=(25080*u>>16)-t2''*/ \
 "movdqa %%xmm10,%%xmm2\n\t" \
 "pmulhw %%xmm13,%%xmm10\n\t" \
 "psubw %%xmm12,%%xmm10\n\t" \
 /*xmm1:xmm6=s*21600+0x2800*/ \
 "pxor %%xmm12,%%xmm12\n\t" \
 "psubw %%xmm14,%%xmm12\n\t" \
 "mov $0x28005460,%[a]\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 "movdqa %%xmm10,%%xmm6\n\t" \
 "movdqa %%xmm10,%%xmm1\n\t" \
 "punpcklwd %%xmm12,%%xmm6\n\t" \
 "pmaddwd %%xmm13,%%xmm6\n\t" \
 "mov $0x0E3D,%[a]\n\t" \
 "punpckhwd %%xmm12,%%xmm1\n\t" \
 "pmaddwd %%xmm13,%%xmm1\n\t" \
 /*xmm6=(s*21600+0x2800>>18)+s*/ \
 "psrad $18,%%xmm6\n\t" \
 "psrad $18,%%xmm1\n\t" \
 "movd %[a],%%xmm12\n\t" \
 "packssdw %%xmm1,%%xmm6\n\t" \
 "pshufd $00,%%xmm12,%%xmm12\n\t" \
 "paddw %%xmm10,%%xmm6\n\t" \
 /*xmm6=_y[6]=v=(s*21600+0x2800>>18)+s+(s!=0)*/ \
 "mov $0x7FFF54DC,%[a]\n\t" \
 "pcmpeqw %%xmm15,%%xmm10\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "psubw %%xmm14,%%xmm10\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 "paddw %%xmm10,%%xmm6\n\t " \
 /*xmm1,xmm3,xmm10,xmm11 are free.*/ \
 /*xmm11:xmm10=54491*t5'''+0x0E3D*/ \
 "movdqa %%xmm5,%%xmm10\n\t" \
 "movdqa %%xmm5,%%xmm11\n\t" \
 "punpcklwd %%xmm5,%%xmm10\n\t" \
 "pmaddwd %%xmm13,%%xmm10\n\t" \
 "mov $0x8E3A8E3A,%[a]\n\t" \
 "punpckhwd %%xmm5,%%xmm11\n\t" \
 "pmaddwd %%xmm13,%%xmm11\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "paddd %%xmm12,%%xmm10\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 "paddd %%xmm12,%%xmm11\n\t" \
 /*xmm7:xmm12=36410*t6''' \
   xmm1=t6'''*/ \
 "movdqa %%xmm7,%%xmm3\n\t" \
 "movdqa %%xmm7,%%xmm1\n\t" \
 "pmulhw %%xmm13,%%xmm3\n\t" \
 "pmullw %%xmm13,%%xmm7\n\t" \
 "paddw %%xmm1,%%xmm3\n\t" \
 "movdqa %%xmm7,%%xmm12\n\t" \
 "punpckhwd %%xmm3,%%xmm7\n\t" \
 "punpcklwd %%xmm3,%%xmm12\n\t" \
 /*xmm10=u=(54491*t5'''+36410*t6'''+0x0E3D>>16)+(t5'''!=0)*/ \
 "paddd %%xmm12,%%xmm10\n\t" \
 "paddd %%xmm7,%%xmm11\n\t" \
 "psrad $16,%%xmm10\n\t" \
 "pcmpeqw %%xmm15,%%xmm5\n\t" \
 "psrad $16,%%xmm11\n\t" \
 "psubw %%xmm14,%%xmm5\n\t" \
 "packssdw %%xmm11,%%xmm10\n\t" \
 "pxor %%xmm12,%%xmm12\n\t" \
 "paddw %%xmm5,%%xmm10\n\t" \
 /*xmm5=_y[5]=u \
   xmm1=s=t6'''-(36410*u>>16)*/ \
 "psubw %%xmm14,%%xmm12\n\t" \
 "movdqa %%xmm10,%%xmm5\n\t" \
 "mov $0x340067C8,%[a]\n\t" \
 "pmulhw %%xmm13,%%xmm10\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "paddw %%xmm5,%%xmm10\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 "psubw %%xmm10,%%xmm1\n\t" \
 /*xmm11:xmm3=s*26568+0x3400*/ \
 "movdqa %%xmm1,%%xmm3\n\t" \
 "movdqa %%xmm1,%%xmm11\n\t" \
 "punpcklwd %%xmm12,%%xmm3\n\t" \
 "pmaddwd %%xmm13,%%xmm3\n\t" \
 "mov $0x7B1B,%[a]\n\t" \
 "punpckhwd %%xmm12,%%xmm11\n\t" \
 "pmaddwd %%xmm13,%%xmm11\n\t" \
 /*xmm3=(s*26568+0x3400>>17)+s*/ \
 "psrad $17,%%xmm3\n\t" \
 "psrad $17,%%xmm11\n\t" \
 "movd %[a],%%xmm12\n\t" \
 "packssdw %%xmm11,%%xmm3\n\t" \
 "pshufd $00,%%xmm12,%%xmm12\n\t" \
 "paddw %%xmm1,%%xmm3\n\t" \
 /*xmm3=_y[3]=v=(s*26568+0x3400>>17)+s+(s!=0)*/ \
 "mov $0x7FFF7B16,%[a]\n\t" \
 "pcmpeqw %%xmm15,%%xmm1\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "psubw %%xmm14,%%xmm1\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 "paddw %%xmm1,%%xmm3\n\t " \
 /*xmm1,xmm7,xmm10,xmm11 are free.*/ \
 /*xmm11:xmm10=64277*t7''+0x7B1B*/ \
 "movdqa %%xmm9,%%xmm10\n\t" \
 "movdqa %%xmm9,%%xmm11\n\t" \
 "punpcklwd %%xmm9,%%xmm10\n\t" \
 "pmaddwd %%xmm13,%%xmm10\n\t" \
 "mov $0x31F131F1,%[a]\n\t" \
 "punpckhwd %%xmm9,%%xmm11\n\t" \
 "pmaddwd %%xmm13,%%xmm11\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "paddd %%xmm12,%%xmm10\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 "paddd %%xmm12,%%xmm11\n\t" \
 /*xmm12:xmm7=12785*t4''*/ \
 "movdqa %%xmm8,%%xmm7\n\t" \
 "movdqa %%xmm8,%%xmm1\n\t" \
 "pmullw %%xmm13,%%xmm7\n\t" \
 "pmulhw %%xmm13,%%xmm1\n\t" \
 "movdqa %%xmm7,%%xmm12\n\t" \
 "punpcklwd %%xmm1,%%xmm7\n\t" \
 "punpckhwd %%xmm1,%%xmm12\n\t" \
 /*xmm10=u=(12785*t4''+64277*t7''+0x7B1B>>16)+(t7''!=0)*/ \
 "paddd %%xmm7,%%xmm10\n\t" \
 "paddd %%xmm12,%%xmm11\n\t" \
 "psrad $16,%%xmm10\n\t" \
 "pcmpeqw %%xmm15,%%xmm9\n\t" \
 "psrad $16,%%xmm11\n\t" \
 "psubw %%xmm14,%%xmm9\n\t" \
 "packssdw %%xmm11,%%xmm10\n\t" \
 "pxor %%xmm12,%%xmm12\n\t" \
 "paddw %%xmm9,%%xmm10\n\t" \
 /*xmm1=_y[1]=u \
   xmm10=s=(12785*u>>16)-t4''*/ \
 "psubw %%xmm14,%%xmm12\n\t" \
 "movdqa %%xmm10,%%xmm1\n\t" \
 "mov $0x3000503B,%[a]\n\t" \
 "pmulhw %%xmm13,%%xmm10\n\t" \
 "movd %[a],%%xmm13\n\t" \
 "psubw %%xmm8,%%xmm10\n\t" \
 "pshufd $00,%%xmm13,%%xmm13\n\t" \
 /*xmm8:xmm7=s*20539+0x3000*/ \
 "movdqa %%xmm10,%%xmm7\n\t" \
 "movdqa %%xmm10,%%xmm8\n\t" \
 "punpcklwd %%xmm12,%%xmm7\n\t" \
 "pmaddwd %%xmm13,%%xmm7\n\t" \
 "punpckhwd %%xmm12,%%xmm8\n\t" \
 "pmaddwd %%xmm13,%%xmm8\n\t" \
 /*xmm7=(s*20539+0x3000>>20)+s*/ \
 "psrad $20,%%xmm7\n\t" \
 "psrad $20,%%xmm8\n\t" \
 "packssdw %%xmm8,%%xmm7\n\t" \
 "paddw %%xmm10,%%xmm7\n\t" \
 /*xmm7=_y[7]=v=(s*20539+0x3000>>20)+s+(s!=0)*/ \
 "pcmpeqw %%xmm15,%%xmm10\n\t" \
 "psubw %%xmm14,%%xmm10\n\t" \
 "paddw %%xmm10,%%xmm7\n\t " \

/*SSE2 implementation of the fDCT for x86-64 only.
  Because of the 8 extra XMM registers on x86-64, this version can operate
   without any temporary stack access at all.*/
void oc_enc_fdct8x8_x86_64sse2(ogg_int16_t _y[64],const ogg_int16_t _x[64]){
  ptrdiff_t a;
  __asm__ __volatile__(
    /*Load the input.*/
    "movdqa 0x00(%[x]),%%xmm0\n\t"
    "movdqa 0x10(%[x]),%%xmm1\n\t"
    "movdqa 0x20(%[x]),%%xmm2\n\t"
    "movdqa 0x30(%[x]),%%xmm3\n\t"
    "movdqa 0x40(%[x]),%%xmm4\n\t"
    "movdqa 0x50(%[x]),%%xmm5\n\t"
    "movdqa 0x60(%[x]),%%xmm6\n\t"
    "movdqa 0x70(%[x]),%%xmm7\n\t"
    /*Add two extra bits of working precision to improve accuracy; any more and
       we could overflow.*/
    /*We also add a few biases to correct for some systematic error that
       remains in the full fDCT->iDCT round trip.*/
    /*xmm15={0}x8*/
    "pxor %%xmm15,%%xmm15\n\t"
    /*xmm14={-1}x8*/
    "pcmpeqb %%xmm14,%%xmm14\n\t"
    "psllw $2,%%xmm0\n\t"
    /*xmm8=xmm0*/
    "movdqa %%xmm0,%%xmm8\n\t"
    "psllw $2,%%xmm1\n\t"
    /*xmm8={_x[7...0]==0}*/
    "pcmpeqw %%xmm15,%%xmm8\n\t"
    "psllw $2,%%xmm2\n\t"
    /*xmm8={_x[7...0]!=0}*/
    "psubw %%xmm14,%%xmm8\n\t"
    "psllw $2,%%xmm3\n\t"
    /*%[a]=1*/
    "mov $1,%[a]\n\t"
    /*xmm8={_x[6]!=0,0,_x[4]!=0,0,_x[2]!=0,0,_x[0]!=0,0}*/
    "pslld $16,%%xmm8\n\t"
    "psllw $2,%%xmm4\n\t"
    /*xmm9={0,0,0,0,0,0,0,1}*/
    "movd %[a],%%xmm9\n\t"
    /*xmm8={0,0,_x[2]!=0,0,_x[0]!=0,0}*/
    "pshufhw $0x00,%%xmm8,%%xmm8\n\t"
    "psllw $2,%%xmm5\n\t"
    /*%[a]={1}x2*/
    "mov $0x10001,%[a]\n\t"
    /*xmm8={0,0,0,0,0,0,0,_x[0]!=0}*/
    "pshuflw $0x01,%%xmm8,%%xmm8\n\t"
    "psllw $2,%%xmm6\n\t"
    /*xmm10={0,0,0,0,0,0,1,1}*/
    "movd %[a],%%xmm10\n\t"
    /*xmm0=_x[7...0]+{0,0,0,0,0,0,0,_x[0]!=0}*/
    "paddw %%xmm8,%%xmm0\n\t"
    "psllw $2,%%xmm7\n\t"
    /*xmm0=_x[7...0]+{0,0,0,0,0,0,1,(_x[0]!=0)+1}*/
    "paddw %%xmm10,%%xmm0\n\t"
    /*xmm1=_x[15...8]-{0,0,0,0,0,0,0,1}*/
    "psubw %%xmm9,%%xmm1\n\t"
    /*Transform columns.*/
    OC_FDCT_8x8
    /*Transform rows.*/
    OC_TRANSPOSE_8x8
    OC_FDCT_8x8
    /*xmm14={-2,-2,-2,-2,-2,-2,-2,-2}*/
    "paddw %%xmm14,%%xmm14\n\t"
    "psubw %%xmm14,%%xmm0\n\t"
    "psubw %%xmm14,%%xmm1\n\t"
    "psraw $2,%%xmm0\n\t"
    "psubw %%xmm14,%%xmm2\n\t"
    "psraw $2,%%xmm1\n\t"
    "psubw %%xmm14,%%xmm3\n\t"
    "psraw $2,%%xmm2\n\t"
    "psubw %%xmm14,%%xmm4\n\t"
    "psraw $2,%%xmm3\n\t"
    "psubw %%xmm14,%%xmm5\n\t"
    "psraw $2,%%xmm4\n\t"
    "psubw %%xmm14,%%xmm6\n\t"
    "psraw $2,%%xmm5\n\t"
    "psubw %%xmm14,%%xmm7\n\t"
    "psraw $2,%%xmm6\n\t"
    "psraw $2,%%xmm7\n\t"
    /*Transpose, zig-zag, and store the result.*/
    /*We could probably do better using SSSE3's palignr, but re-using MMXEXT
       version will do for now.*/
#define OC_ZZ_LOAD_ROW_LO(_row,_reg) \
    "movdq2q %%xmm"#_row","_reg"\n\t" \

#define OC_ZZ_LOAD_ROW_HI(_row,_reg) \
    "punpckhqdq %%xmm"#_row",%%xmm"#_row"\n\t" \
    "movdq2q %%xmm"#_row","_reg"\n\t" \

    OC_TRANSPOSE_ZIG_ZAG_MMXEXT
#undef OC_ZZ_LOAD_ROW_LO
#undef OC_ZZ_LOAD_ROW_HI
    :[a]"=&r"(a)
    :[y]"r"(_y),[x]"r"(_x)
    :"memory"
  );
}
#endif
