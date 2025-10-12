/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors                      *
 * https://www.xiph.org/                                            *
 *                                                                  *
 ********************************************************************

  function:

 ********************************************************************/

#include "x86enc.h"

#if defined(OC_X86_ASM)



/*The default enquant table is not quite suitable for SIMD purposes.
  First, the m and l parameters need to be separated so that an entire row full
   of m's or l's can be loaded at a time.
  Second, x86 SIMD has no element-wise arithmetic right-shift, so we have to
   emulate one with a multiply.
  Therefore we translate the shift count into a scale factor.*/
void oc_enc_enquant_table_init_x86(void *_enquant,
 const ogg_uint16_t _dequant[64]){
  ogg_int16_t *m;
  ogg_int16_t *l;
  int          zzi;
  m=(ogg_int16_t *)_enquant;
  l=m+64;
  for(zzi=0;zzi<64;zzi++){
    oc_iquant q;
    oc_iquant_init(&q,_dequant[zzi]);
    m[zzi]=q.m;
    /*q.l must be at least 2 for this to work; fortunately, once all the scale
       factors are baked in, the minimum quantizer is much larger than that.*/
    l[zzi]=1<<16-q.l;
  }
}

void oc_enc_enquant_table_fixup_x86(void *_enquant[3][3][2],int _nqis){
  int pli;
  int qii;
  int qti;
  for(pli=0;pli<3;pli++)for(qii=1;qii<_nqis;qii++)for(qti=0;qti<2;qti++){
    ((ogg_int16_t *)_enquant[pli][qii][qti])[0]=
     ((ogg_int16_t *)_enquant[pli][0][qti])[0];
    ((ogg_int16_t *)_enquant[pli][qii][qti])[64]=
     ((ogg_int16_t *)_enquant[pli][0][qti])[64];
  }
}

int oc_enc_quantize_sse2(ogg_int16_t _qdct[64],const ogg_int16_t _dct[64],
 const ogg_uint16_t _dequant[64],const void *_enquant){
  ptrdiff_t r;
  __asm__ __volatile__(
    "xor %[r],%[r]\n\t"
    /*Loop through two rows at a time.*/
    ".p2align 4\n\t"
    "0:\n\t"
    /*Load the first two rows of the data and the quant matrices.*/
    "movdqa 0x00(%[dct],%[r]),%%xmm0\n\t"
    "movdqa 0x10(%[dct],%[r]),%%xmm1\n\t"
    "movdqa 0x00(%[dq],%[r]),%%xmm2\n\t"
    "movdqa 0x10(%[dq],%[r]),%%xmm3\n\t"
    "movdqa 0x00(%[q],%[r]),%%xmm4\n\t"
    "movdqa 0x10(%[q],%[r]),%%xmm5\n\t"
    /*Double the input and propagate its sign to the rounding factor.
      Using SSSE3's psignw would help here, but we need the mask later anyway.*/
    "movdqa %%xmm0,%%xmm6\n\t"
    "psraw $15,%%xmm0\n\t"
    "movdqa %%xmm1,%%xmm7\n\t"
    "paddw %%xmm6,%%xmm6\n\t"
    "psraw $15,%%xmm1\n\t"
    "paddw %%xmm7,%%xmm7\n\t"
    "paddw %%xmm0,%%xmm2\n\t"
    "paddw %%xmm1,%%xmm3\n\t"
    "pxor %%xmm0,%%xmm2\n\t"
    "pxor %%xmm1,%%xmm3\n\t"
    /*Add the rounding factor and perform the first multiply.*/
    "paddw %%xmm2,%%xmm6\n\t"
    "paddw %%xmm3,%%xmm7\n\t"
    "pmulhw %%xmm6,%%xmm4\n\t"
    "pmulhw %%xmm7,%%xmm5\n\t"
    "movdqa 0x80(%[q],%[r]),%%xmm2\n\t"
    "movdqa 0x90(%[q],%[r]),%%xmm3\n\t"
    "paddw %%xmm4,%%xmm6\n\t"
    "paddw %%xmm5,%%xmm7\n\t"
    /*Emulate an element-wise right-shift via a second multiply.*/
    "pmulhw %%xmm2,%%xmm6\n\t"
    "pmulhw %%xmm3,%%xmm7\n\t"
    "add $32,%[r]\n\t"
    "cmp $96,%[r]\n\t"
    /*Correct for the sign.*/
    "psubw %%xmm0,%%xmm6\n\t"
    "psubw %%xmm1,%%xmm7\n\t"
    /*Save the result.*/
    "movdqa %%xmm6,-0x20(%[qdct],%[r])\n\t"
    "movdqa %%xmm7,-0x10(%[qdct],%[r])\n\t"
    "jle 0b\n\t"
    /*Now find the location of the last non-zero value.*/
    "movdqa 0x50(%[qdct]),%%xmm5\n\t"
    "movdqa 0x40(%[qdct]),%%xmm4\n\t"
    "packsswb %%xmm7,%%xmm6\n\t"
    "packsswb %%xmm5,%%xmm4\n\t"
    "pxor %%xmm0,%%xmm0\n\t"
    "mov $-1,%k[dq]\n\t"
    "pcmpeqb %%xmm0,%%xmm6\n\t"
    "pcmpeqb %%xmm0,%%xmm4\n\t"
    "pmovmskb %%xmm6,%k[q]\n\t"
    "pmovmskb %%xmm4,%k[r]\n\t"
    "shl $16,%k[q]\n\t"
    "or %k[r],%k[q]\n\t"
    "mov $32,%[r]\n\t"
    /*We have to use xor here instead of not in order to set the flags.*/
    "xor %k[dq],%k[q]\n\t"
    "jnz 1f\n\t"
    "movdqa 0x30(%[qdct]),%%xmm7\n\t"
    "movdqa 0x20(%[qdct]),%%xmm6\n\t"
    "movdqa 0x10(%[qdct]),%%xmm5\n\t"
    "movdqa 0x00(%[qdct]),%%xmm4\n\t"
    "packsswb %%xmm7,%%xmm6\n\t"
    "packsswb %%xmm5,%%xmm4\n\t"
    "pcmpeqb %%xmm0,%%xmm6\n\t"
    "pcmpeqb %%xmm0,%%xmm4\n\t"
    "pmovmskb %%xmm6,%k[q]\n\t"
    "pmovmskb %%xmm4,%k[r]\n\t"
    "shl $16,%k[q]\n\t"
    "or %k[r],%k[q]\n\t"
    "xor %[r],%[r]\n\t"
    "not %k[q]\n\t"
    "or $1,%k[q]\n\t"
    "1:\n\t"
    "bsr %k[q],%k[q]\n\t"
    "add %k[q],%k[r]\n\t"
    :[r]"=&a"(r),[q]"+r"(_enquant),[dq]"+r"(_dequant)
    :[dct]"r"(_dct),[qdct]"r"(_qdct)
    :"cc","memory"
  );
  return (int)r;
}

#endif
