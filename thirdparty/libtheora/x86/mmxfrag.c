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
    last mod: $Id$

 ********************************************************************/

/*MMX acceleration of fragment reconstruction for motion compensation.
  Originally written by Rudolf Marek.
  Additional optimization by Nils Pipenbrinck.
  Note: Loops are unrolled for best performance.
  The iteration each instruction belongs to is marked in the comments as #i.*/
#include <stddef.h>
#include "x86int.h"

#if defined(OC_X86_ASM)

/*Copies an 8x8 block of pixels from _src to _dst, assuming _ystride bytes
   between rows.*/
# define OC_FRAG_COPY_MMX(_dst,_src,_ystride) \
  do{ \
    const unsigned char *src; \
    unsigned char       *dst; \
    ptrdiff_t            ystride3; \
    src=(_src); \
    dst=(_dst); \
    __asm__ __volatile__( \
      /*src+0*ystride*/ \
      "movq (%[src]),%%mm0\n\t" \
      /*src+1*ystride*/ \
      "movq (%[src],%[ystride]),%%mm1\n\t" \
      /*ystride3=ystride*3*/ \
      "lea (%[ystride],%[ystride],2),%[ystride3]\n\t" \
      /*src+2*ystride*/ \
      "movq (%[src],%[ystride],2),%%mm2\n\t" \
      /*src+3*ystride*/ \
      "movq (%[src],%[ystride3]),%%mm3\n\t" \
      /*dst+0*ystride*/ \
      "movq %%mm0,(%[dst])\n\t" \
      /*dst+1*ystride*/ \
      "movq %%mm1,(%[dst],%[ystride])\n\t" \
      /*Pointer to next 4.*/ \
      "lea (%[src],%[ystride],4),%[src]\n\t" \
      /*dst+2*ystride*/ \
      "movq %%mm2,(%[dst],%[ystride],2)\n\t" \
      /*dst+3*ystride*/ \
      "movq %%mm3,(%[dst],%[ystride3])\n\t" \
      /*Pointer to next 4.*/ \
      "lea (%[dst],%[ystride],4),%[dst]\n\t" \
      /*src+0*ystride*/ \
      "movq (%[src]),%%mm0\n\t" \
      /*src+1*ystride*/ \
      "movq (%[src],%[ystride]),%%mm1\n\t" \
      /*src+2*ystride*/ \
      "movq (%[src],%[ystride],2),%%mm2\n\t" \
      /*src+3*ystride*/ \
      "movq (%[src],%[ystride3]),%%mm3\n\t" \
      /*dst+0*ystride*/ \
      "movq %%mm0,(%[dst])\n\t" \
      /*dst+1*ystride*/ \
      "movq %%mm1,(%[dst],%[ystride])\n\t" \
      /*dst+2*ystride*/ \
      "movq %%mm2,(%[dst],%[ystride],2)\n\t" \
      /*dst+3*ystride*/ \
      "movq %%mm3,(%[dst],%[ystride3])\n\t" \
      :[dst]"+r"(dst),[src]"+r"(src),[ystride3]"=&r"(ystride3) \
      :[ystride]"r"((ptrdiff_t)(_ystride)) \
      :"memory" \
    ); \
  } \
  while(0)

/*Copies an 8x8 block of pixels from _src to _dst, assuming _ystride bytes
   between rows.*/
void oc_frag_copy_mmx(unsigned char *_dst,
 const unsigned char *_src,int _ystride){
  OC_FRAG_COPY_MMX(_dst,_src,_ystride);
}

/*Copies the fragments specified by the lists of fragment indices from one
   frame to another.
  _dst_frame:     The reference frame to copy to.
  _src_frame:     The reference frame to copy from.
  _ystride:       The row stride of the reference frames.
  _fragis:        A pointer to a list of fragment indices.
  _nfragis:       The number of fragment indices to copy.
  _frag_buf_offs: The offsets of fragments in the reference frames.*/
void oc_frag_copy_list_mmx(unsigned char *_dst_frame,
 const unsigned char *_src_frame,int _ystride,
 const ptrdiff_t *_fragis,ptrdiff_t _nfragis,const ptrdiff_t *_frag_buf_offs){
  ptrdiff_t fragii;
  for(fragii=0;fragii<_nfragis;fragii++){
    ptrdiff_t frag_buf_off;
    frag_buf_off=_frag_buf_offs[_fragis[fragii]];
    OC_FRAG_COPY_MMX(_dst_frame+frag_buf_off,
     _src_frame+frag_buf_off,_ystride);
  }
}


void oc_frag_recon_intra_mmx(unsigned char *_dst,int _ystride,
 const ogg_int16_t *_residue){
  __asm__ __volatile__(
    /*Set mm0 to 0xFFFFFFFFFFFFFFFF.*/
    "pcmpeqw %%mm0,%%mm0\n\t"
    /*#0 Load low residue.*/
    "movq 0*8(%[residue]),%%mm1\n\t"
    /*#0 Load high residue.*/
    "movq 1*8(%[residue]),%%mm2\n\t"
    /*Set mm0 to 0x8000800080008000.*/
    "psllw $15,%%mm0\n\t"
    /*#1 Load low residue.*/
    "movq 2*8(%[residue]),%%mm3\n\t"
    /*#1 Load high residue.*/
    "movq 3*8(%[residue]),%%mm4\n\t"
    /*Set mm0 to 0x0080008000800080.*/
    "psrlw $8,%%mm0\n\t"
    /*#2 Load low residue.*/
    "movq 4*8(%[residue]),%%mm5\n\t"
    /*#2 Load high residue.*/
    "movq 5*8(%[residue]),%%mm6\n\t"
    /*#0 Bias low  residue.*/
    "paddsw %%mm0,%%mm1\n\t"
    /*#0 Bias high residue.*/
    "paddsw %%mm0,%%mm2\n\t"
    /*#0 Pack to byte.*/
    "packuswb %%mm2,%%mm1\n\t"
    /*#1 Bias low  residue.*/
    "paddsw %%mm0,%%mm3\n\t"
    /*#1 Bias high residue.*/
    "paddsw %%mm0,%%mm4\n\t"
    /*#1 Pack to byte.*/
    "packuswb %%mm4,%%mm3\n\t"
    /*#2 Bias low  residue.*/
    "paddsw %%mm0,%%mm5\n\t"
    /*#2 Bias high residue.*/
    "paddsw %%mm0,%%mm6\n\t"
    /*#2 Pack to byte.*/
    "packuswb %%mm6,%%mm5\n\t"
    /*#0 Write row.*/
    "movq %%mm1,(%[dst])\n\t"
    /*#1 Write row.*/
    "movq %%mm3,(%[dst],%[ystride])\n\t"
    /*#2 Write row.*/
    "movq %%mm5,(%[dst],%[ystride],2)\n\t"
    /*#3 Load low residue.*/
    "movq 6*8(%[residue]),%%mm1\n\t"
    /*#3 Load high residue.*/
    "movq 7*8(%[residue]),%%mm2\n\t"
    /*#4 Load high residue.*/
    "movq 8*8(%[residue]),%%mm3\n\t"
    /*#4 Load high residue.*/
    "movq 9*8(%[residue]),%%mm4\n\t"
    /*#5 Load high residue.*/
    "movq 10*8(%[residue]),%%mm5\n\t"
    /*#5 Load high residue.*/
    "movq 11*8(%[residue]),%%mm6\n\t"
    /*#3 Bias low  residue.*/
    "paddsw %%mm0,%%mm1\n\t"
    /*#3 Bias high residue.*/
    "paddsw %%mm0,%%mm2\n\t"
    /*#3 Pack to byte.*/
    "packuswb %%mm2,%%mm1\n\t"
    /*#4 Bias low  residue.*/
    "paddsw %%mm0,%%mm3\n\t"
    /*#4 Bias high residue.*/
    "paddsw %%mm0,%%mm4\n\t"
    /*#4 Pack to byte.*/
    "packuswb %%mm4,%%mm3\n\t"
    /*#5 Bias low  residue.*/
    "paddsw %%mm0,%%mm5\n\t"
    /*#5 Bias high residue.*/
    "paddsw %%mm0,%%mm6\n\t"
    /*#5 Pack to byte.*/
    "packuswb %%mm6,%%mm5\n\t"
    /*#3 Write row.*/
    "movq %%mm1,(%[dst],%[ystride3])\n\t"
    /*#4 Write row.*/
    "movq %%mm3,(%[dst4])\n\t"
    /*#5 Write row.*/
    "movq %%mm5,(%[dst4],%[ystride])\n\t"
    /*#6 Load low residue.*/
    "movq 12*8(%[residue]),%%mm1\n\t"
    /*#6 Load high residue.*/
    "movq 13*8(%[residue]),%%mm2\n\t"
    /*#7 Load low residue.*/
    "movq 14*8(%[residue]),%%mm3\n\t"
    /*#7 Load high residue.*/
    "movq 15*8(%[residue]),%%mm4\n\t"
    /*#6 Bias low  residue.*/
    "paddsw %%mm0,%%mm1\n\t"
    /*#6 Bias high residue.*/
    "paddsw %%mm0,%%mm2\n\t"
    /*#6 Pack to byte.*/
    "packuswb %%mm2,%%mm1\n\t"
    /*#7 Bias low  residue.*/
    "paddsw %%mm0,%%mm3\n\t"
    /*#7 Bias high residue.*/
    "paddsw %%mm0,%%mm4\n\t"
    /*#7 Pack to byte.*/
    "packuswb %%mm4,%%mm3\n\t"
    /*#6 Write row.*/
    "movq %%mm1,(%[dst4],%[ystride],2)\n\t"
    /*#7 Write row.*/
    "movq %%mm3,(%[dst4],%[ystride3])\n\t"
    :
    :[residue]"r"(_residue),
     [dst]"r"(_dst),
     [dst4]"r"(_dst+(_ystride<<2)),
     [ystride]"r"((ptrdiff_t)_ystride),
     [ystride3]"r"((ptrdiff_t)_ystride*3)
    :"memory"
  );
}

void oc_frag_recon_inter_mmx(unsigned char *_dst,const unsigned char *_src,
 int _ystride,const ogg_int16_t *_residue){
  int i;
  /*Zero mm0.*/
  __asm__ __volatile__("pxor %%mm0,%%mm0\n\t"::);
  for(i=4;i-->0;){
    __asm__ __volatile__(
      /*#0 Load source.*/
      "movq (%[src]),%%mm3\n\t"
      /*#1 Load source.*/
      "movq (%[src],%[ystride]),%%mm7\n\t"
      /*#0 Get copy of src.*/
      "movq %%mm3,%%mm4\n\t"
      /*#0 Expand high source.*/
      "punpckhbw %%mm0,%%mm4\n\t"
      /*#0 Expand low  source.*/
      "punpcklbw %%mm0,%%mm3\n\t"
      /*#0 Add residue high.*/
      "paddsw 8(%[residue]),%%mm4\n\t"
      /*#1 Get copy of src.*/
      "movq %%mm7,%%mm2\n\t"
      /*#0 Add residue low.*/
      "paddsw (%[residue]), %%mm3\n\t"
      /*#1 Expand high source.*/
      "punpckhbw %%mm0,%%mm2\n\t"
      /*#0 Pack final row pixels.*/
      "packuswb %%mm4,%%mm3\n\t"
      /*#1 Expand low  source.*/
      "punpcklbw %%mm0,%%mm7\n\t"
      /*#1 Add residue low.*/
      "paddsw 16(%[residue]),%%mm7\n\t"
      /*#1 Add residue high.*/
      "paddsw 24(%[residue]),%%mm2\n\t"
      /*Advance residue.*/
      "lea 32(%[residue]),%[residue]\n\t"
      /*#1 Pack final row pixels.*/
      "packuswb %%mm2,%%mm7\n\t"
      /*Advance src.*/
      "lea (%[src],%[ystride],2),%[src]\n\t"
      /*#0 Write row.*/
      "movq %%mm3,(%[dst])\n\t"
      /*#1 Write row.*/
      "movq %%mm7,(%[dst],%[ystride])\n\t"
      /*Advance dst.*/
      "lea (%[dst],%[ystride],2),%[dst]\n\t"
      :[residue]"+r"(_residue),[dst]"+r"(_dst),[src]"+r"(_src)
      :[ystride]"r"((ptrdiff_t)_ystride)
      :"memory"
    );
  }
}

void oc_frag_recon_inter2_mmx(unsigned char *_dst,const unsigned char *_src1,
 const unsigned char *_src2,int _ystride,const ogg_int16_t *_residue){
  int i;
  /*Zero mm7.*/
  __asm__ __volatile__("pxor %%mm7,%%mm7\n\t"::);
  for(i=4;i-->0;){
    __asm__ __volatile__(
      /*#0 Load src1.*/
      "movq (%[src1]),%%mm0\n\t"
      /*#0 Load src2.*/
      "movq (%[src2]),%%mm2\n\t"
      /*#0 Copy src1.*/
      "movq %%mm0,%%mm1\n\t"
      /*#0 Copy src2.*/
      "movq %%mm2,%%mm3\n\t"
      /*#1 Load src1.*/
      "movq (%[src1],%[ystride]),%%mm4\n\t"
      /*#0 Unpack lower src1.*/
      "punpcklbw %%mm7,%%mm0\n\t"
      /*#1 Load src2.*/
      "movq (%[src2],%[ystride]),%%mm5\n\t"
      /*#0 Unpack higher src1.*/
      "punpckhbw %%mm7,%%mm1\n\t"
      /*#0 Unpack lower src2.*/
      "punpcklbw %%mm7,%%mm2\n\t"
      /*#0 Unpack higher src2.*/
      "punpckhbw %%mm7,%%mm3\n\t"
      /*Advance src1 ptr.*/
      "lea (%[src1],%[ystride],2),%[src1]\n\t"
      /*Advance src2 ptr.*/
      "lea (%[src2],%[ystride],2),%[src2]\n\t"
      /*#0 Lower src1+src2.*/
      "paddsw %%mm2,%%mm0\n\t"
      /*#0 Higher src1+src2.*/
      "paddsw %%mm3,%%mm1\n\t"
      /*#1 Copy src1.*/
      "movq %%mm4,%%mm2\n\t"
      /*#0 Build lo average.*/
      "psraw $1,%%mm0\n\t"
      /*#1 Copy src2.*/
      "movq %%mm5,%%mm3\n\t"
      /*#1 Unpack lower src1.*/
      "punpcklbw %%mm7,%%mm4\n\t"
      /*#0 Build hi average.*/
      "psraw $1,%%mm1\n\t"
      /*#1 Unpack higher src1.*/
      "punpckhbw %%mm7,%%mm2\n\t"
      /*#0 low+=residue.*/
      "paddsw (%[residue]),%%mm0\n\t"
      /*#1 Unpack lower src2.*/
      "punpcklbw %%mm7,%%mm5\n\t"
      /*#0 high+=residue.*/
      "paddsw 8(%[residue]),%%mm1\n\t"
      /*#1 Unpack higher src2.*/
      "punpckhbw %%mm7,%%mm3\n\t"
      /*#1 Lower src1+src2.*/
      "paddsw %%mm4,%%mm5\n\t"
      /*#0 Pack and saturate.*/
      "packuswb %%mm1,%%mm0\n\t"
      /*#1 Higher src1+src2.*/
      "paddsw %%mm2,%%mm3\n\t"
      /*#0 Write row.*/
      "movq %%mm0,(%[dst])\n\t"
      /*#1 Build lo average.*/
      "psraw $1,%%mm5\n\t"
      /*#1 Build hi average.*/
      "psraw $1,%%mm3\n\t"
      /*#1 low+=residue.*/
      "paddsw 16(%[residue]),%%mm5\n\t"
      /*#1 high+=residue.*/
      "paddsw 24(%[residue]),%%mm3\n\t"
      /*#1 Pack and saturate.*/
      "packuswb  %%mm3,%%mm5\n\t"
      /*#1 Write row ptr.*/
      "movq %%mm5,(%[dst],%[ystride])\n\t"
      /*Advance residue ptr.*/
      "add $32,%[residue]\n\t"
      /*Advance dest ptr.*/
      "lea (%[dst],%[ystride],2),%[dst]\n\t"
     :[dst]"+r"(_dst),[residue]"+r"(_residue),
      [src1]"+r"(_src1),[src2]"+r"(_src2)
     :[ystride]"r"((ptrdiff_t)_ystride)
     :"memory"
    );
  }
}

void oc_restore_fpu_mmx(void){
  __asm__ __volatile__("emms\n\t");
}
#endif
