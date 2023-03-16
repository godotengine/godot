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
    src=(_src); \
    dst=(_dst); \
    __asm  mov SRC,src \
    __asm  mov DST,dst \
    __asm  mov YSTRIDE,_ystride \
    /*src+0*ystride*/ \
    __asm  movq mm0,[SRC] \
    /*src+1*ystride*/ \
    __asm  movq mm1,[SRC+YSTRIDE] \
    /*ystride3=ystride*3*/ \
    __asm  lea YSTRIDE3,[YSTRIDE+YSTRIDE*2] \
    /*src+2*ystride*/ \
    __asm  movq mm2,[SRC+YSTRIDE*2] \
    /*src+3*ystride*/ \
    __asm  movq mm3,[SRC+YSTRIDE3] \
    /*dst+0*ystride*/ \
    __asm  movq [DST],mm0 \
    /*dst+1*ystride*/ \
    __asm  movq [DST+YSTRIDE],mm1 \
    /*Pointer to next 4.*/ \
    __asm  lea SRC,[SRC+YSTRIDE*4] \
    /*dst+2*ystride*/ \
    __asm  movq [DST+YSTRIDE*2],mm2 \
    /*dst+3*ystride*/ \
    __asm  movq [DST+YSTRIDE3],mm3 \
    /*Pointer to next 4.*/ \
    __asm  lea DST,[DST+YSTRIDE*4] \
    /*src+0*ystride*/ \
    __asm  movq mm0,[SRC] \
    /*src+1*ystride*/ \
    __asm  movq mm1,[SRC+YSTRIDE] \
    /*src+2*ystride*/ \
    __asm  movq mm2,[SRC+YSTRIDE*2] \
    /*src+3*ystride*/ \
    __asm  movq mm3,[SRC+YSTRIDE3] \
    /*dst+0*ystride*/ \
    __asm  movq [DST],mm0 \
    /*dst+1*ystride*/ \
    __asm  movq [DST+YSTRIDE],mm1 \
    /*dst+2*ystride*/ \
    __asm  movq [DST+YSTRIDE*2],mm2 \
    /*dst+3*ystride*/ \
    __asm  movq [DST+YSTRIDE3],mm3 \
  } \
  while(0)

/*Copies an 8x8 block of pixels from _src to _dst, assuming _ystride bytes
   between rows.*/
void oc_frag_copy_mmx(unsigned char *_dst,
 const unsigned char *_src,int _ystride){
#define SRC edx
#define DST eax
#define YSTRIDE ecx
#define YSTRIDE3 esi
  OC_FRAG_COPY_MMX(_dst,_src,_ystride);
#undef SRC
#undef DST
#undef YSTRIDE
#undef YSTRIDE3
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
#define SRC edx
#define DST eax
#define YSTRIDE ecx
#define YSTRIDE3 edi
    OC_FRAG_COPY_MMX(_dst_frame+frag_buf_off,
     _src_frame+frag_buf_off,_ystride);
#undef SRC
#undef DST
#undef YSTRIDE
#undef YSTRIDE3
  }
}

void oc_frag_recon_intra_mmx(unsigned char *_dst,int _ystride,
 const ogg_int16_t *_residue){
  __asm{
#define DST edx
#define DST4 esi
#define YSTRIDE eax
#define YSTRIDE3 edi
#define RESIDUE ecx
    mov DST,_dst
    mov YSTRIDE,_ystride
    mov RESIDUE,_residue
    lea DST4,[DST+YSTRIDE*4]
    lea YSTRIDE3,[YSTRIDE+YSTRIDE*2]
    /*Set mm0 to 0xFFFFFFFFFFFFFFFF.*/
    pcmpeqw mm0,mm0
    /*#0 Load low residue.*/
    movq mm1,[0*8+RESIDUE]
    /*#0 Load high residue.*/
    movq mm2,[1*8+RESIDUE]
    /*Set mm0 to 0x8000800080008000.*/
    psllw mm0,15
    /*#1 Load low residue.*/
    movq mm3,[2*8+RESIDUE]
    /*#1 Load high residue.*/
    movq mm4,[3*8+RESIDUE]
    /*Set mm0 to 0x0080008000800080.*/
    psrlw mm0,8
    /*#2 Load low residue.*/
    movq mm5,[4*8+RESIDUE]
    /*#2 Load high residue.*/
    movq mm6,[5*8+RESIDUE]
    /*#0 Bias low  residue.*/
    paddsw mm1,mm0
    /*#0 Bias high residue.*/
    paddsw mm2,mm0
    /*#0 Pack to byte.*/
    packuswb mm1,mm2
    /*#1 Bias low  residue.*/
    paddsw mm3,mm0
    /*#1 Bias high residue.*/
    paddsw mm4,mm0
    /*#1 Pack to byte.*/
    packuswb mm3,mm4
    /*#2 Bias low  residue.*/
    paddsw mm5,mm0
    /*#2 Bias high residue.*/
    paddsw mm6,mm0
    /*#2 Pack to byte.*/
    packuswb mm5,mm6
    /*#0 Write row.*/
    movq [DST],mm1
    /*#1 Write row.*/
    movq [DST+YSTRIDE],mm3
    /*#2 Write row.*/
    movq [DST+YSTRIDE*2],mm5
    /*#3 Load low residue.*/
    movq mm1,[6*8+RESIDUE]
    /*#3 Load high residue.*/
    movq mm2,[7*8+RESIDUE]
    /*#4 Load high residue.*/
    movq mm3,[8*8+RESIDUE]
    /*#4 Load high residue.*/
    movq mm4,[9*8+RESIDUE]
    /*#5 Load high residue.*/
    movq mm5,[10*8+RESIDUE]
    /*#5 Load high residue.*/
    movq mm6,[11*8+RESIDUE]
    /*#3 Bias low  residue.*/
    paddsw mm1,mm0
    /*#3 Bias high residue.*/
    paddsw mm2,mm0
    /*#3 Pack to byte.*/
    packuswb mm1,mm2
    /*#4 Bias low  residue.*/
    paddsw mm3,mm0
    /*#4 Bias high residue.*/
    paddsw mm4,mm0
    /*#4 Pack to byte.*/
    packuswb mm3,mm4
    /*#5 Bias low  residue.*/
    paddsw mm5,mm0
    /*#5 Bias high residue.*/
    paddsw mm6,mm0
    /*#5 Pack to byte.*/
    packuswb mm5,mm6
    /*#3 Write row.*/
    movq [DST+YSTRIDE3],mm1
    /*#4 Write row.*/
    movq [DST4],mm3
    /*#5 Write row.*/
    movq [DST4+YSTRIDE],mm5
    /*#6 Load low residue.*/
    movq mm1,[12*8+RESIDUE]
    /*#6 Load high residue.*/
    movq mm2,[13*8+RESIDUE]
    /*#7 Load low residue.*/
    movq mm3,[14*8+RESIDUE]
    /*#7 Load high residue.*/
    movq mm4,[15*8+RESIDUE]
    /*#6 Bias low  residue.*/
    paddsw mm1,mm0
    /*#6 Bias high residue.*/
    paddsw mm2,mm0
    /*#6 Pack to byte.*/
    packuswb mm1,mm2
    /*#7 Bias low  residue.*/
    paddsw mm3,mm0
    /*#7 Bias high residue.*/
    paddsw mm4,mm0
    /*#7 Pack to byte.*/
    packuswb mm3,mm4
    /*#6 Write row.*/
    movq [DST4+YSTRIDE*2],mm1
    /*#7 Write row.*/
    movq [DST4+YSTRIDE3],mm3
#undef DST
#undef DST4
#undef YSTRIDE
#undef YSTRIDE3
#undef RESIDUE
  }
}

void oc_frag_recon_inter_mmx(unsigned char *_dst,const unsigned char *_src,
 int _ystride,const ogg_int16_t *_residue){
  int i;
  /*Zero mm0.*/
  __asm pxor mm0,mm0;
  for(i=4;i-->0;){
    __asm{
#define DST edx
#define SRC ecx
#define YSTRIDE edi
#define RESIDUE eax
      mov DST,_dst
      mov SRC,_src
      mov YSTRIDE,_ystride
      mov RESIDUE,_residue
      /*#0 Load source.*/
      movq mm3,[SRC]
      /*#1 Load source.*/
      movq mm7,[SRC+YSTRIDE]
      /*#0 Get copy of src.*/
      movq mm4,mm3
      /*#0 Expand high source.*/
      punpckhbw mm4,mm0
      /*#0 Expand low  source.*/
      punpcklbw mm3,mm0
      /*#0 Add residue high.*/
      paddsw mm4,[8+RESIDUE]
      /*#1 Get copy of src.*/
      movq mm2,mm7
      /*#0 Add residue low.*/
      paddsw  mm3,[RESIDUE]
      /*#1 Expand high source.*/
      punpckhbw mm2,mm0
      /*#0 Pack final row pixels.*/
      packuswb mm3,mm4
      /*#1 Expand low  source.*/
      punpcklbw mm7,mm0
      /*#1 Add residue low.*/
      paddsw mm7,[16+RESIDUE]
      /*#1 Add residue high.*/
      paddsw mm2,[24+RESIDUE]
      /*Advance residue.*/
      lea RESIDUE,[32+RESIDUE]
      /*#1 Pack final row pixels.*/
      packuswb mm7,mm2
      /*Advance src.*/
      lea SRC,[SRC+YSTRIDE*2]
      /*#0 Write row.*/
      movq [DST],mm3
      /*#1 Write row.*/
      movq [DST+YSTRIDE],mm7
      /*Advance dst.*/
      lea DST,[DST+YSTRIDE*2]
      mov _residue,RESIDUE
      mov _dst,DST
      mov _src,SRC
#undef DST
#undef SRC
#undef YSTRIDE
#undef RESIDUE
    }
  }
}

void oc_frag_recon_inter2_mmx(unsigned char *_dst,const unsigned char *_src1,
 const unsigned char *_src2,int _ystride,const ogg_int16_t *_residue){
  int i;
  /*Zero mm7.*/
  __asm pxor mm7,mm7;
  for(i=4;i-->0;){
    __asm{
#define SRC1 ecx
#define SRC2 edi
#define YSTRIDE esi
#define RESIDUE edx
#define DST eax
      mov YSTRIDE,_ystride
      mov DST,_dst
      mov RESIDUE,_residue
      mov SRC1,_src1
      mov SRC2,_src2
      /*#0 Load src1.*/
      movq mm0,[SRC1]
      /*#0 Load src2.*/
      movq mm2,[SRC2]
      /*#0 Copy src1.*/
      movq mm1,mm0
      /*#0 Copy src2.*/
      movq mm3,mm2
      /*#1 Load src1.*/
      movq mm4,[SRC1+YSTRIDE]
      /*#0 Unpack lower src1.*/
      punpcklbw mm0,mm7
      /*#1 Load src2.*/
      movq mm5,[SRC2+YSTRIDE]
      /*#0 Unpack higher src1.*/
      punpckhbw mm1,mm7
      /*#0 Unpack lower src2.*/
      punpcklbw mm2,mm7
      /*#0 Unpack higher src2.*/
      punpckhbw mm3,mm7
      /*Advance src1 ptr.*/
      lea SRC1,[SRC1+YSTRIDE*2]
      /*Advance src2 ptr.*/
      lea SRC2,[SRC2+YSTRIDE*2]
      /*#0 Lower src1+src2.*/
      paddsw mm0,mm2
      /*#0 Higher src1+src2.*/
      paddsw mm1,mm3
      /*#1 Copy src1.*/
      movq mm2,mm4
      /*#0 Build lo average.*/
      psraw mm0,1
      /*#1 Copy src2.*/
      movq mm3,mm5
      /*#1 Unpack lower src1.*/
      punpcklbw mm4,mm7
      /*#0 Build hi average.*/
      psraw mm1,1
      /*#1 Unpack higher src1.*/
      punpckhbw mm2,mm7
      /*#0 low+=residue.*/
      paddsw mm0,[RESIDUE]
      /*#1 Unpack lower src2.*/
      punpcklbw mm5,mm7
      /*#0 high+=residue.*/
      paddsw mm1,[8+RESIDUE]
      /*#1 Unpack higher src2.*/
      punpckhbw mm3,mm7
      /*#1 Lower src1+src2.*/
      paddsw mm5,mm4
      /*#0 Pack and saturate.*/
      packuswb mm0,mm1
      /*#1 Higher src1+src2.*/
      paddsw mm3,mm2
      /*#0 Write row.*/
      movq [DST],mm0
      /*#1 Build lo average.*/
      psraw mm5,1
      /*#1 Build hi average.*/
      psraw mm3,1
      /*#1 low+=residue.*/
      paddsw mm5,[16+RESIDUE]
      /*#1 high+=residue.*/
      paddsw mm3,[24+RESIDUE]
      /*#1 Pack and saturate.*/
      packuswb  mm5,mm3
      /*#1 Write row ptr.*/
      movq [DST+YSTRIDE],mm5
      /*Advance residue ptr.*/
      add RESIDUE,32
      /*Advance dest ptr.*/
      lea DST,[DST+YSTRIDE*2]
      mov _dst,DST
      mov _residue,RESIDUE
      mov _src1,SRC1
      mov _src2,SRC2
#undef SRC1
#undef SRC2
#undef YSTRIDE
#undef RESIDUE
#undef DST
    }
  }
}

void oc_restore_fpu_mmx(void){
  __asm emms;
}

#endif
