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

# define OC_FDCT_STAGE1_8x4 \
 "#OC_FDCT_STAGE1_8x4\n\t" \
 /*Stage 1:*/ \
 /*mm0=t7'=t0-t7*/ \
 "psubw %%mm7,%%mm0\n\t" \
 "paddw %%mm7,%%mm7\n\t" \
 /*mm1=t6'=t1-t6*/ \
 "psubw %%mm6,%%mm1\n\t" \
 "paddw %%mm6,%%mm6\n\t" \
 /*mm2=t5'=t2-t5*/ \
 "psubw %%mm5,%%mm2\n\t" \
 "paddw %%mm5,%%mm5\n\t" \
 /*mm3=t4'=t3-t4*/ \
 "psubw %%mm4,%%mm3\n\t" \
 "paddw %%mm4,%%mm4\n\t" \
 /*mm7=t0'=t0+t7*/ \
 "paddw %%mm0,%%mm7\n\t" \
 /*mm6=t1'=t1+t6*/ \
 "paddw %%mm1,%%mm6\n\t" \
 /*mm5=t2'=t2+t5*/ \
 "paddw %%mm2,%%mm5\n\t" \
 /*mm4=t3'=t3+t4*/ \
 "paddw %%mm3,%%mm4\n\t" \

# define OC_FDCT8x4(_r0,_r1,_r2,_r3,_r4,_r5,_r6,_r7) \
 "#OC_FDCT8x4\n\t" \
 /*Stage 2:*/ \
 /*mm7=t3''=t0'-t3'*/ \
 "psubw %%mm4,%%mm7\n\t" \
 "paddw %%mm4,%%mm4\n\t" \
 /*mm6=t2''=t1'-t2'*/ \
 "psubw %%mm5,%%mm6\n\t" \
 "movq %%mm7,"_r6"(%[y])\n\t" \
 "paddw %%mm5,%%mm5\n\t" \
 /*mm1=t5''=t6'-t5'*/ \
 "psubw %%mm2,%%mm1\n\t" \
 "movq %%mm6,"_r2"(%[y])\n\t" \
 /*mm4=t0''=t0'+t3'*/ \
 "paddw %%mm7,%%mm4\n\t" \
 "paddw %%mm2,%%mm2\n\t" \
 /*mm5=t1''=t1'+t2'*/ \
 "movq %%mm4,"_r0"(%[y])\n\t" \
 "paddw %%mm6,%%mm5\n\t" \
 /*mm2=t6''=t6'+t5'*/ \
 "paddw %%mm1,%%mm2\n\t" \
 "movq %%mm5,"_r4"(%[y])\n\t" \
 /*mm0=t7', mm1=t5'', mm2=t6'', mm3=t4'.*/ \
 /*mm4, mm5, mm6, mm7 are free.*/ \
 /*Stage 3:*/ \
 /*mm6={2}x4, mm7={27146,0xB500>>1}x2*/ \
 "mov $0x5A806A0A,%[a]\n\t" \
 "pcmpeqb %%mm6,%%mm6\n\t" \
 "movd %[a],%%mm7\n\t" \
 "psrlw $15,%%mm6\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 "paddw %%mm6,%%mm6\n\t" \
 /*mm0=0, m2={-1}x4 \
   mm5:mm4=t5''*27146+0xB500*/ \
 "movq %%mm1,%%mm4\n\t" \
 "movq %%mm1,%%mm5\n\t" \
 "punpcklwd %%mm6,%%mm4\n\t" \
 "movq %%mm2,"_r3"(%[y])\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "movq %%mm0,"_r7"(%[y])\n\t" \
 "punpckhwd %%mm6,%%mm5\n\t" \
 "pxor %%mm0,%%mm0\n\t" \
 "pmaddwd %%mm7,%%mm5\n\t" \
 "pcmpeqb %%mm2,%%mm2\n\t" \
 /*mm2=t6'', mm1=t5''+(t5''!=0) \
   mm4=(t5''*27146+0xB500>>16)*/ \
 "pcmpeqw %%mm1,%%mm0\n\t" \
 "psrad $16,%%mm4\n\t" \
 "psubw %%mm2,%%mm0\n\t" \
 "movq "_r3"(%[y]),%%mm2\n\t" \
 "psrad $16,%%mm5\n\t" \
 "paddw %%mm0,%%mm1\n\t" \
 "packssdw %%mm5,%%mm4\n\t" \
 /*mm4=s=(t5''*27146+0xB500>>16)+t5''+(t5''!=0)>>1*/ \
 "paddw %%mm1,%%mm4\n\t" \
 "movq "_r7"(%[y]),%%mm0\n\t" \
 "psraw $1,%%mm4\n\t" \
 "movq %%mm3,%%mm1\n\t" \
 /*mm3=t4''=t4'+s*/ \
 "paddw %%mm4,%%mm3\n\t" \
 /*mm1=t5'''=t4'-s*/ \
 "psubw %%mm4,%%mm1\n\t" \
 /*mm1=0, mm3={-1}x4 \
   mm5:mm4=t6''*27146+0xB500*/ \
 "movq %%mm2,%%mm4\n\t" \
 "movq %%mm2,%%mm5\n\t" \
 "punpcklwd %%mm6,%%mm4\n\t" \
 "movq %%mm1,"_r5"(%[y])\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "movq %%mm3,"_r1"(%[y])\n\t" \
 "punpckhwd %%mm6,%%mm5\n\t" \
 "pxor %%mm1,%%mm1\n\t" \
 "pmaddwd %%mm7,%%mm5\n\t" \
 "pcmpeqb %%mm3,%%mm3\n\t" \
 /*mm2=t6''+(t6''!=0), mm4=(t6''*27146+0xB500>>16)*/ \
 "psrad $16,%%mm4\n\t" \
 "pcmpeqw %%mm2,%%mm1\n\t" \
 "psrad $16,%%mm5\n\t" \
 "psubw %%mm3,%%mm1\n\t" \
 "packssdw %%mm5,%%mm4\n\t" \
 "paddw %%mm1,%%mm2\n\t" \
 /*mm1=t1'' \
   mm4=s=(t6''*27146+0xB500>>16)+t6''+(t6''!=0)>>1*/ \
 "paddw %%mm2,%%mm4\n\t" \
 "movq "_r4"(%[y]),%%mm1\n\t" \
 "psraw $1,%%mm4\n\t" \
 "movq %%mm0,%%mm2\n\t" \
 /*mm7={54491-0x7FFF,0x7FFF}x2 \
   mm0=t7''=t7'+s*/ \
 "paddw %%mm4,%%mm0\n\t" \
 /*mm2=t6'''=t7'-s*/ \
 "psubw %%mm4,%%mm2\n\t" \
 /*Stage 4:*/ \
 /*mm0=0, mm2=t0'' \
   mm5:mm4=t1''*27146+0xB500*/ \
 "movq %%mm1,%%mm4\n\t" \
 "movq %%mm1,%%mm5\n\t" \
 "punpcklwd %%mm6,%%mm4\n\t" \
 "movq %%mm2,"_r3"(%[y])\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "movq "_r0"(%[y]),%%mm2\n\t" \
 "punpckhwd %%mm6,%%mm5\n\t" \
 "movq %%mm0,"_r7"(%[y])\n\t" \
 "pmaddwd %%mm7,%%mm5\n\t" \
 "pxor %%mm0,%%mm0\n\t" \
 /*mm7={27146,0x4000>>1}x2 \
   mm0=s=(t1''*27146+0xB500>>16)+t1''+(t1''!=0)*/ \
 "psrad $16,%%mm4\n\t" \
 "mov $0x20006A0A,%[a]\n\t" \
 "pcmpeqw %%mm1,%%mm0\n\t" \
 "movd %[a],%%mm7\n\t" \
 "psrad $16,%%mm5\n\t" \
 "psubw %%mm3,%%mm0\n\t" \
 "packssdw %%mm5,%%mm4\n\t" \
 "paddw %%mm1,%%mm0\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 "paddw %%mm4,%%mm0\n\t" \
 /*mm6={0x00000E3D}x2 \
   mm1=-(t0''==0), mm5:mm4=t0''*27146+0x4000*/ \
 "movq %%mm2,%%mm4\n\t" \
 "movq %%mm2,%%mm5\n\t" \
 "punpcklwd %%mm6,%%mm4\n\t" \
 "mov $0x0E3D,%[a]\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "punpckhwd %%mm6,%%mm5\n\t" \
 "movd %[a],%%mm6\n\t" \
 "pmaddwd %%mm7,%%mm5\n\t" \
 "pxor %%mm1,%%mm1\n\t" \
 "punpckldq %%mm6,%%mm6\n\t" \
 "pcmpeqw %%mm2,%%mm1\n\t" \
 /*mm4=r=(t0''*27146+0x4000>>16)+t0''+(t0''!=0)*/ \
 "psrad $16,%%mm4\n\t" \
 "psubw %%mm3,%%mm1\n\t" \
 "psrad $16,%%mm5\n\t" \
 "paddw %%mm1,%%mm2\n\t" \
 "packssdw %%mm5,%%mm4\n\t" \
 "movq "_r5"(%[y]),%%mm1\n\t" \
 "paddw %%mm2,%%mm4\n\t" \
 /*mm2=t6'', mm0=_y[0]=u=r+s>>1 \
   The naive implementation could cause overflow, so we use \
    u=(r&s)+((r^s)>>1).*/ \
 "movq "_r3"(%[y]),%%mm2\n\t" \
 "movq %%mm0,%%mm7\n\t" \
 "pxor %%mm4,%%mm0\n\t" \
 "pand %%mm4,%%mm7\n\t" \
 "psraw $1,%%mm0\n\t" \
 "mov $0x7FFF54DC,%[a]\n\t" \
 "paddw %%mm7,%%mm0\n\t" \
 "movd %[a],%%mm7\n\t" \
 /*mm7={54491-0x7FFF,0x7FFF}x2 \
   mm4=_y[4]=v=r-u*/ \
 "psubw %%mm0,%%mm4\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 "movq %%mm4,"_r4"(%[y])\n\t" \
 /*mm0=0, mm7={36410}x4 \
   mm1=(t5'''!=0), mm5:mm4=54491*t5'''+0x0E3D*/ \
 "movq %%mm1,%%mm4\n\t" \
 "movq %%mm1,%%mm5\n\t" \
 "punpcklwd %%mm1,%%mm4\n\t" \
 "mov $0x8E3A8E3A,%[a]\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "movq %%mm0,"_r0"(%[y])\n\t" \
 "punpckhwd %%mm1,%%mm5\n\t" \
 "pxor %%mm0,%%mm0\n\t" \
 "pmaddwd %%mm7,%%mm5\n\t" \
 "pcmpeqw %%mm0,%%mm1\n\t" \
 "movd %[a],%%mm7\n\t" \
 "psubw %%mm3,%%mm1\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 "paddd %%mm6,%%mm4\n\t" \
 "paddd %%mm6,%%mm5\n\t" \
 /*mm0=0 \
   mm3:mm1=36410*t6'''+((t5'''!=0)<<16)*/ \
 "movq %%mm2,%%mm6\n\t" \
 "movq %%mm2,%%mm3\n\t" \
 "pmulhw %%mm7,%%mm6\n\t" \
 "paddw %%mm2,%%mm1\n\t" \
 "pmullw %%mm7,%%mm3\n\t" \
 "pxor %%mm0,%%mm0\n\t" \
 "paddw %%mm1,%%mm6\n\t" \
 "movq %%mm3,%%mm1\n\t" \
 "punpckhwd %%mm6,%%mm3\n\t" \
 "punpcklwd %%mm6,%%mm1\n\t" \
 /*mm3={-1}x4, mm6={1}x4 \
   mm4=_y[5]=u=(54491*t5'''+36410*t6'''+0x0E3D>>16)+(t5'''!=0)*/ \
 "paddd %%mm3,%%mm5\n\t" \
 "paddd %%mm1,%%mm4\n\t" \
 "psrad $16,%%mm5\n\t" \
 "pxor %%mm6,%%mm6\n\t" \
 "psrad $16,%%mm4\n\t" \
 "pcmpeqb %%mm3,%%mm3\n\t" \
 "packssdw %%mm5,%%mm4\n\t" \
 "psubw %%mm3,%%mm6\n\t" \
 /*mm1=t7'', mm7={26568,0x3400}x2 \
   mm2=s=t6'''-(36410*u>>16)*/ \
 "movq %%mm4,%%mm1\n\t" \
 "mov $0x340067C8,%[a]\n\t" \
 "pmulhw %%mm7,%%mm4\n\t" \
 "movd %[a],%%mm7\n\t" \
 "movq %%mm1,"_r5"(%[y])\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 "paddw %%mm1,%%mm4\n\t" \
 "movq "_r7"(%[y]),%%mm1\n\t" \
 "psubw %%mm4,%%mm2\n\t" \
 /*mm6={0x00007B1B}x2 \
   mm0=(s!=0), mm5:mm4=s*26568+0x3400*/ \
 "movq %%mm2,%%mm4\n\t" \
 "movq %%mm2,%%mm5\n\t" \
 "punpcklwd %%mm6,%%mm4\n\t" \
 "pcmpeqw %%mm2,%%mm0\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "mov $0x7B1B,%[a]\n\t" \
 "punpckhwd %%mm6,%%mm5\n\t" \
 "movd %[a],%%mm6\n\t" \
 "pmaddwd %%mm7,%%mm5\n\t" \
 "psubw %%mm3,%%mm0\n\t" \
 "punpckldq %%mm6,%%mm6\n\t" \
 /*mm7={64277-0x7FFF,0x7FFF}x2 \
   mm2=_y[3]=v=(s*26568+0x3400>>17)+s+(s!=0)*/ \
 "psrad $17,%%mm4\n\t" \
 "paddw %%mm0,%%mm2\n\t" \
 "psrad $17,%%mm5\n\t" \
 "mov $0x7FFF7B16,%[a]\n\t" \
 "packssdw %%mm5,%%mm4\n\t" \
 "movd %[a],%%mm7\n\t" \
 "paddw %%mm4,%%mm2\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 /*mm0=0, mm7={12785}x4 \
   mm1=(t7''!=0), mm2=t4'', mm5:mm4=64277*t7''+0x7B1B*/ \
 "movq %%mm1,%%mm4\n\t" \
 "movq %%mm1,%%mm5\n\t" \
 "movq %%mm2,"_r3"(%[y])\n\t" \
 "punpcklwd %%mm1,%%mm4\n\t" \
 "movq "_r1"(%[y]),%%mm2\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "mov $0x31F131F1,%[a]\n\t" \
 "punpckhwd %%mm1,%%mm5\n\t" \
 "pxor %%mm0,%%mm0\n\t" \
 "pmaddwd %%mm7,%%mm5\n\t" \
 "pcmpeqw %%mm0,%%mm1\n\t" \
 "movd %[a],%%mm7\n\t" \
 "psubw %%mm3,%%mm1\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 "paddd %%mm6,%%mm4\n\t" \
 "paddd %%mm6,%%mm5\n\t" \
 /*mm3:mm1=12785*t4'''+((t7''!=0)<<16)*/ \
 "movq %%mm2,%%mm6\n\t" \
 "movq %%mm2,%%mm3\n\t" \
 "pmulhw %%mm7,%%mm6\n\t" \
 "pmullw %%mm7,%%mm3\n\t" \
 "paddw %%mm1,%%mm6\n\t" \
 "movq %%mm3,%%mm1\n\t" \
 "punpckhwd %%mm6,%%mm3\n\t" \
 "punpcklwd %%mm6,%%mm1\n\t" \
 /*mm3={-1}x4, mm6={1}x4 \
   mm4=_y[1]=u=(12785*t4'''+64277*t7''+0x7B1B>>16)+(t7''!=0)*/ \
 "paddd %%mm3,%%mm5\n\t" \
 "paddd %%mm1,%%mm4\n\t" \
 "psrad $16,%%mm5\n\t" \
 "pxor %%mm6,%%mm6\n\t" \
 "psrad $16,%%mm4\n\t" \
 "pcmpeqb %%mm3,%%mm3\n\t" \
 "packssdw %%mm5,%%mm4\n\t" \
 "psubw %%mm3,%%mm6\n\t" \
 /*mm1=t3'', mm7={20539,0x3000}x2 \
   mm4=s=(12785*u>>16)-t4''*/ \
 "movq %%mm4,"_r1"(%[y])\n\t" \
 "pmulhw %%mm7,%%mm4\n\t" \
 "mov $0x3000503B,%[a]\n\t" \
 "movq "_r6"(%[y]),%%mm1\n\t" \
 "movd %[a],%%mm7\n\t" \
 "psubw %%mm2,%%mm4\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 /*mm6={0x00006CB7}x2 \
   mm0=(s!=0), mm5:mm4=s*20539+0x3000*/ \
 "movq %%mm4,%%mm5\n\t" \
 "movq %%mm4,%%mm2\n\t" \
 "punpcklwd %%mm6,%%mm4\n\t" \
 "pcmpeqw %%mm2,%%mm0\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "mov $0x6CB7,%[a]\n\t" \
 "punpckhwd %%mm6,%%mm5\n\t" \
 "movd %[a],%%mm6\n\t" \
 "pmaddwd %%mm7,%%mm5\n\t" \
 "psubw %%mm3,%%mm0\n\t" \
 "punpckldq %%mm6,%%mm6\n\t" \
 /*mm7={60547-0x7FFF,0x7FFF}x2 \
   mm2=_y[7]=v=(s*20539+0x3000>>20)+s+(s!=0)*/ \
 "psrad $20,%%mm4\n\t" \
 "paddw %%mm0,%%mm2\n\t" \
 "psrad $20,%%mm5\n\t" \
 "mov $0x7FFF6C84,%[a]\n\t" \
 "packssdw %%mm5,%%mm4\n\t" \
 "movd %[a],%%mm7\n\t" \
 "paddw %%mm4,%%mm2\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 /*mm0=0, mm7={25080}x4 \
   mm2=t2'', mm5:mm4=60547*t3''+0x6CB7*/ \
 "movq %%mm1,%%mm4\n\t" \
 "movq %%mm1,%%mm5\n\t" \
 "movq %%mm2,"_r7"(%[y])\n\t" \
 "punpcklwd %%mm1,%%mm4\n\t" \
 "movq "_r2"(%[y]),%%mm2\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "mov $0x61F861F8,%[a]\n\t" \
 "punpckhwd %%mm1,%%mm5\n\t" \
 "pxor %%mm0,%%mm0\n\t" \
 "pmaddwd %%mm7,%%mm5\n\t" \
 "movd %[a],%%mm7\n\t" \
 "pcmpeqw %%mm0,%%mm1\n\t" \
 "psubw %%mm3,%%mm1\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 "paddd %%mm6,%%mm4\n\t" \
 "paddd %%mm6,%%mm5\n\t" \
 /*mm3:mm1=25080*t2''+((t3''!=0)<<16)*/ \
 "movq %%mm2,%%mm6\n\t" \
 "movq %%mm2,%%mm3\n\t" \
 "pmulhw %%mm7,%%mm6\n\t" \
 "pmullw %%mm7,%%mm3\n\t" \
 "paddw %%mm1,%%mm6\n\t" \
 "movq %%mm3,%%mm1\n\t" \
 "punpckhwd %%mm6,%%mm3\n\t" \
 "punpcklwd %%mm6,%%mm1\n\t" \
 /*mm1={-1}x4 \
   mm4=u=(25080*t2''+60547*t3''+0x6CB7>>16)+(t3''!=0)*/ \
 "paddd %%mm3,%%mm5\n\t" \
 "paddd %%mm1,%%mm4\n\t" \
 "psrad $16,%%mm5\n\t" \
 "mov $0x28005460,%[a]\n\t" \
 "psrad $16,%%mm4\n\t" \
 "pcmpeqb %%mm1,%%mm1\n\t" \
 "packssdw %%mm5,%%mm4\n\t" \
 /*mm5={1}x4, mm6=_y[2]=u, mm7={21600,0x2800}x2 \
   mm4=s=(25080*u>>16)-t2''*/ \
 "movq %%mm4,%%mm6\n\t" \
 "pmulhw %%mm7,%%mm4\n\t" \
 "pxor %%mm5,%%mm5\n\t" \
 "movd %[a],%%mm7\n\t" \
 "psubw %%mm1,%%mm5\n\t" \
 "punpckldq %%mm7,%%mm7\n\t" \
 "psubw %%mm2,%%mm4\n\t" \
 /*mm2=s+(s!=0) \
   mm4:mm3=s*21600+0x2800*/ \
 "movq %%mm4,%%mm3\n\t" \
 "movq %%mm4,%%mm2\n\t" \
 "punpckhwd %%mm5,%%mm4\n\t" \
 "pcmpeqw %%mm2,%%mm0\n\t" \
 "pmaddwd %%mm7,%%mm4\n\t" \
 "psubw %%mm1,%%mm0\n\t" \
 "punpcklwd %%mm5,%%mm3\n\t" \
 "paddw %%mm0,%%mm2\n\t" \
 "pmaddwd %%mm7,%%mm3\n\t" \
 /*mm0=_y[4], mm1=_y[7], mm4=_y[0], mm5=_y[5] \
   mm3=_y[6]=v=(s*21600+0x2800>>18)+s+(s!=0)*/ \
 "movq "_r4"(%[y]),%%mm0\n\t" \
 "psrad $18,%%mm4\n\t" \
 "movq "_r5"(%[y]),%%mm5\n\t" \
 "psrad $18,%%mm3\n\t" \
 "movq "_r7"(%[y]),%%mm1\n\t" \
 "packssdw %%mm4,%%mm3\n\t" \
 "movq "_r0"(%[y]),%%mm4\n\t" \
 "paddw %%mm2,%%mm3\n\t" \

/*On input, mm4=_y[0], mm6=_y[2], mm0=_y[4], mm5=_y[5], mm3=_y[6], mm1=_y[7].
  On output, {_y[4],mm1,mm2,mm3} contains the transpose of _y[4...7] and
   {mm4,mm5,mm6,mm7} contains the transpose of _y[0...3].*/
# define OC_TRANSPOSE8x4(_r0,_r1,_r2,_r3,_r4,_r5,_r6,_r7) \
 "#OC_TRANSPOSE8x4\n\t" \
 /*First 4x4 transpose:*/ \
 /*mm0 = e3 e2 e1 e0 \
   mm5 = f3 f2 f1 f0 \
   mm3 = g3 g2 g1 g0 \
   mm1 = h3 h2 h1 h0*/ \
 "movq %%mm0,%%mm2\n\t" \
 "punpcklwd %%mm5,%%mm0\n\t" \
 "punpckhwd %%mm5,%%mm2\n\t" \
 "movq %%mm3,%%mm5\n\t" \
 "punpcklwd %%mm1,%%mm3\n\t" \
 "punpckhwd %%mm1,%%mm5\n\t" \
 /*mm0 = f1 e1 f0 e0 \
   mm2 = f3 e3 f2 e2 \
   mm3 = h1 g1 h0 g0 \
   mm5 = h3 g3 h2 g2*/ \
 "movq %%mm0,%%mm1\n\t" \
 "punpckldq %%mm3,%%mm0\n\t" \
 "movq %%mm0,"_r4"(%[y])\n\t" \
 "punpckhdq %%mm3,%%mm1\n\t" \
 "movq "_r1"(%[y]),%%mm0\n\t" \
 "movq %%mm2,%%mm3\n\t" \
 "punpckldq %%mm5,%%mm2\n\t" \
 "punpckhdq %%mm5,%%mm3\n\t" \
 "movq "_r3"(%[y]),%%mm5\n\t" \
 /*_y[4] = h0 g0 f0 e0 \
    mm1  = h1 g1 f1 e1 \
    mm2  = h2 g2 f2 e2 \
    mm3  = h3 g3 f3 e3*/ \
 /*Second 4x4 transpose:*/ \
 /*mm4 = a3 a2 a1 a0 \
   mm0 = b3 b2 b1 b0 \
   mm6 = c3 c2 c1 c0 \
   mm5 = d3 d2 d1 d0*/ \
 "movq %%mm4,%%mm7\n\t" \
 "punpcklwd %%mm0,%%mm4\n\t" \
 "punpckhwd %%mm0,%%mm7\n\t" \
 "movq %%mm6,%%mm0\n\t" \
 "punpcklwd %%mm5,%%mm6\n\t" \
 "punpckhwd %%mm5,%%mm0\n\t" \
 /*mm4 = b1 a1 b0 a0 \
   mm7 = b3 a3 b2 a2 \
   mm6 = d1 c1 d0 c0 \
   mm0 = d3 c3 d2 c2*/ \
 "movq %%mm4,%%mm5\n\t" \
 "punpckldq %%mm6,%%mm4\n\t" \
 "punpckhdq %%mm6,%%mm5\n\t" \
 "movq %%mm7,%%mm6\n\t" \
 "punpckhdq %%mm0,%%mm7\n\t" \
 "punpckldq %%mm0,%%mm6\n\t" \
 /*mm4 = d0 c0 b0 a0 \
   mm5 = d1 c1 b1 a1 \
   mm6 = d2 c2 b2 a2 \
   mm7 = d3 c3 b3 a3*/ \

/*MMX implementation of the fDCT.*/
void oc_enc_fdct8x8_mmxext(ogg_int16_t _y[64],const ogg_int16_t _x[64]){
  OC_ALIGN8(ogg_int16_t buf[64]);
  ptrdiff_t   a;
  __asm__ __volatile__(
    /*Add two extra bits of working precision to improve accuracy; any more and
       we could overflow.*/
    /*We also add biases to correct for some systematic error that remains in
       the full fDCT->iDCT round trip.*/
    "movq 0x00(%[x]),%%mm0\n\t"
    "movq 0x10(%[x]),%%mm1\n\t"
    "movq 0x20(%[x]),%%mm2\n\t"
    "movq 0x30(%[x]),%%mm3\n\t"
    "pcmpeqb %%mm4,%%mm4\n\t"
    "pxor %%mm7,%%mm7\n\t"
    "movq %%mm0,%%mm5\n\t"
    "psllw $2,%%mm0\n\t"
    "pcmpeqw %%mm7,%%mm5\n\t"
    "movq 0x70(%[x]),%%mm7\n\t"
    "psllw $2,%%mm1\n\t"
    "psubw %%mm4,%%mm5\n\t"
    "psllw $2,%%mm2\n\t"
    "mov $1,%[a]\n\t"
    "pslld $16,%%mm5\n\t"
    "movd %[a],%%mm6\n\t"
    "psllq $16,%%mm5\n\t"
    "mov $0x10001,%[a]\n\t"
    "psllw $2,%%mm3\n\t"
    "movd %[a],%%mm4\n\t"
    "punpckhwd %%mm6,%%mm5\n\t"
    "psubw %%mm6,%%mm1\n\t"
    "movq 0x60(%[x]),%%mm6\n\t"
    "paddw %%mm5,%%mm0\n\t"
    "movq 0x50(%[x]),%%mm5\n\t"
    "paddw %%mm4,%%mm0\n\t"
    "movq 0x40(%[x]),%%mm4\n\t"
    /*We inline stage1 of the transform here so we can get better instruction
       scheduling with the shifts.*/
    /*mm0=t7'=t0-t7*/
    "psllw $2,%%mm7\n\t"
    "psubw %%mm7,%%mm0\n\t"
    "psllw $2,%%mm6\n\t"
    "paddw %%mm7,%%mm7\n\t"
    /*mm1=t6'=t1-t6*/
    "psllw $2,%%mm5\n\t"
    "psubw %%mm6,%%mm1\n\t"
    "psllw $2,%%mm4\n\t"
    "paddw %%mm6,%%mm6\n\t"
    /*mm2=t5'=t2-t5*/
    "psubw %%mm5,%%mm2\n\t"
    "paddw %%mm5,%%mm5\n\t"
    /*mm3=t4'=t3-t4*/
    "psubw %%mm4,%%mm3\n\t"
    "paddw %%mm4,%%mm4\n\t"
    /*mm7=t0'=t0+t7*/
    "paddw %%mm0,%%mm7\n\t"
    /*mm6=t1'=t1+t6*/
    "paddw %%mm1,%%mm6\n\t"
    /*mm5=t2'=t2+t5*/
    "paddw %%mm2,%%mm5\n\t"
    /*mm4=t3'=t3+t4*/
    "paddw %%mm3,%%mm4\n\t"
    OC_FDCT8x4("0x00","0x10","0x20","0x30","0x40","0x50","0x60","0x70")
    OC_TRANSPOSE8x4("0x00","0x10","0x20","0x30","0x40","0x50","0x60","0x70")
    /*Swap out this 8x4 block for the next one.*/
    "movq 0x08(%[x]),%%mm0\n\t"
    "movq %%mm7,0x30(%[y])\n\t"
    "movq 0x78(%[x]),%%mm7\n\t"
    "movq %%mm1,0x50(%[y])\n\t"
    "movq 0x18(%[x]),%%mm1\n\t"
    "movq %%mm6,0x20(%[y])\n\t"
    "movq 0x68(%[x]),%%mm6\n\t"
    "movq %%mm2,0x60(%[y])\n\t"
    "movq 0x28(%[x]),%%mm2\n\t"
    "movq %%mm5,0x10(%[y])\n\t"
    "movq 0x58(%[x]),%%mm5\n\t"
    "movq %%mm3,0x70(%[y])\n\t"
    "movq 0x38(%[x]),%%mm3\n\t"
    /*And increase its working precision, too.*/
    "psllw $2,%%mm0\n\t"
    "movq %%mm4,0x00(%[y])\n\t"
    "psllw $2,%%mm7\n\t"
    "movq 0x48(%[x]),%%mm4\n\t"
    /*We inline stage1 of the transform here so we can get better instruction
       scheduling with the shifts.*/
    /*mm0=t7'=t0-t7*/
    "psubw %%mm7,%%mm0\n\t"
    "psllw $2,%%mm1\n\t"
    "paddw %%mm7,%%mm7\n\t"
    "psllw $2,%%mm6\n\t"
    /*mm1=t6'=t1-t6*/
    "psubw %%mm6,%%mm1\n\t"
    "psllw $2,%%mm2\n\t"
    "paddw %%mm6,%%mm6\n\t"
    "psllw $2,%%mm5\n\t"
    /*mm2=t5'=t2-t5*/
    "psubw %%mm5,%%mm2\n\t"
    "psllw $2,%%mm3\n\t"
    "paddw %%mm5,%%mm5\n\t"
    "psllw $2,%%mm4\n\t"
    /*mm3=t4'=t3-t4*/
    "psubw %%mm4,%%mm3\n\t"
    "paddw %%mm4,%%mm4\n\t"
    /*mm7=t0'=t0+t7*/
    "paddw %%mm0,%%mm7\n\t"
    /*mm6=t1'=t1+t6*/
    "paddw %%mm1,%%mm6\n\t"
    /*mm5=t2'=t2+t5*/
    "paddw %%mm2,%%mm5\n\t"
    /*mm4=t3'=t3+t4*/
    "paddw %%mm3,%%mm4\n\t"
    OC_FDCT8x4("0x08","0x18","0x28","0x38","0x48","0x58","0x68","0x78")
    OC_TRANSPOSE8x4("0x08","0x18","0x28","0x38","0x48","0x58","0x68","0x78")
    /*Here the first 4x4 block of output from the last transpose is the second
       4x4 block of input for the next transform.
      We have cleverly arranged that it already be in the appropriate place,
       so we only have to do half the stores and loads.*/
    "movq 0x00(%[y]),%%mm0\n\t"
    "movq %%mm1,0x58(%[y])\n\t"
    "movq 0x10(%[y]),%%mm1\n\t"
    "movq %%mm2,0x68(%[y])\n\t"
    "movq 0x20(%[y]),%%mm2\n\t"
    "movq %%mm3,0x78(%[y])\n\t"
    "movq 0x30(%[y]),%%mm3\n\t"
    OC_FDCT_STAGE1_8x4
    OC_FDCT8x4("0x00","0x10","0x20","0x30","0x08","0x18","0x28","0x38")
    /*mm2={-2}x4*/
    "pcmpeqw %%mm2,%%mm2\n\t"
    "paddw %%mm2,%%mm2\n\t"
    /*Round and store the results (no transpose).*/
    "movq 0x10(%[y]),%%mm7\n\t"
    "psubw %%mm2,%%mm4\n\t"
    "psubw %%mm2,%%mm6\n\t"
    "psraw $2,%%mm4\n\t"
    "psubw %%mm2,%%mm0\n\t"
    "movq %%mm4,"OC_MEM_OFFS(0x00,buf)"\n\t"
    "movq 0x30(%[y]),%%mm4\n\t"
    "psraw $2,%%mm6\n\t"
    "psubw %%mm2,%%mm5\n\t"
    "movq %%mm6,"OC_MEM_OFFS(0x20,buf)"\n\t"
    "psraw $2,%%mm0\n\t"
    "psubw %%mm2,%%mm3\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x40,buf)"\n\t"
    "psraw $2,%%mm5\n\t"
    "psubw %%mm2,%%mm1\n\t"
    "movq %%mm5,"OC_MEM_OFFS(0x50,buf)"\n\t"
    "psraw $2,%%mm3\n\t"
    "psubw %%mm2,%%mm7\n\t"
    "movq %%mm3,"OC_MEM_OFFS(0x60,buf)"\n\t"
    "psraw $2,%%mm1\n\t"
    "psubw %%mm2,%%mm4\n\t"
    "movq %%mm1,"OC_MEM_OFFS(0x70,buf)"\n\t"
    "psraw $2,%%mm7\n\t"
    "movq %%mm7,"OC_MEM_OFFS(0x10,buf)"\n\t"
    "psraw $2,%%mm4\n\t"
    "movq %%mm4,"OC_MEM_OFFS(0x30,buf)"\n\t"
    /*Load the next block.*/
    "movq 0x40(%[y]),%%mm0\n\t"
    "movq 0x78(%[y]),%%mm7\n\t"
    "movq 0x50(%[y]),%%mm1\n\t"
    "movq 0x68(%[y]),%%mm6\n\t"
    "movq 0x60(%[y]),%%mm2\n\t"
    "movq 0x58(%[y]),%%mm5\n\t"
    "movq 0x70(%[y]),%%mm3\n\t"
    "movq 0x48(%[y]),%%mm4\n\t"
    OC_FDCT_STAGE1_8x4
    OC_FDCT8x4("0x40","0x50","0x60","0x70","0x48","0x58","0x68","0x78")
    /*mm2={-2}x4*/
    "pcmpeqw %%mm2,%%mm2\n\t"
    "paddw %%mm2,%%mm2\n\t"
    /*Round and store the results (no transpose).*/
    "movq 0x50(%[y]),%%mm7\n\t"
    "psubw %%mm2,%%mm4\n\t"
    "psubw %%mm2,%%mm6\n\t"
    "psraw $2,%%mm4\n\t"
    "psubw %%mm2,%%mm0\n\t"
    "movq %%mm4,"OC_MEM_OFFS(0x08,buf)"\n\t"
    "movq 0x70(%[y]),%%mm4\n\t"
    "psraw $2,%%mm6\n\t"
    "psubw %%mm2,%%mm5\n\t"
    "movq %%mm6,"OC_MEM_OFFS(0x28,buf)"\n\t"
    "psraw $2,%%mm0\n\t"
    "psubw %%mm2,%%mm3\n\t"
    "movq %%mm0,"OC_MEM_OFFS(0x48,buf)"\n\t"
    "psraw $2,%%mm5\n\t"
    "psubw %%mm2,%%mm1\n\t"
    "movq %%mm5,"OC_MEM_OFFS(0x58,buf)"\n\t"
    "psraw $2,%%mm3\n\t"
    "psubw %%mm2,%%mm7\n\t"
    "movq %%mm3,"OC_MEM_OFFS(0x68,buf)"\n\t"
    "psraw $2,%%mm1\n\t"
    "psubw %%mm2,%%mm4\n\t"
    "movq %%mm1,"OC_MEM_OFFS(0x78,buf)"\n\t"
    "psraw $2,%%mm7\n\t"
    "movq %%mm7,"OC_MEM_OFFS(0x18,buf)"\n\t"
    "psraw $2,%%mm4\n\t"
    "movq %%mm4,"OC_MEM_OFFS(0x38,buf)"\n\t"
    /*Final transpose and zig-zag.*/
#define OC_ZZ_LOAD_ROW_LO(_row,_reg) \
    "movq "OC_MEM_OFFS(16*_row,buf)","_reg"\n\t" \

#define OC_ZZ_LOAD_ROW_HI(_row,_reg) \
    "movq "OC_MEM_OFFS(16*_row+8,buf)","_reg"\n\t" \

    OC_TRANSPOSE_ZIG_ZAG_MMXEXT
#undef OC_ZZ_LOAD_ROW_LO
#undef OC_ZZ_LOAD_ROW_HI
    :[a]"=&r"(a),[buf]"=m"(OC_ARRAY_OPERAND(ogg_int16_t,buf,64))
    :[y]"r"(_y),[x]"r"(_x)
    :"memory"
  );
}

#endif
