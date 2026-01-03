/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation https://www.xiph.org/                 *
 *                                                                  *
 ********************************************************************

  function:

 ********************************************************************/
#include <stddef.h>
#include "x86enc.h"

#if defined(OC_X86_ASM)

unsigned oc_enc_frag_sad_mmxext(const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  ptrdiff_t ret;
  __asm{
#define SRC esi
#define REF edx
#define YSTRIDE ecx
#define YSTRIDE3 edi
    mov YSTRIDE,_ystride
    mov SRC,_src
    mov REF,_ref
    /*Load the first 4 rows of each block.*/
    movq mm0,[SRC]
    movq mm1,[REF]
    movq mm2,[SRC][YSTRIDE]
    movq mm3,[REF][YSTRIDE]
    lea YSTRIDE3,[YSTRIDE+YSTRIDE*2]
    movq mm4,[SRC+YSTRIDE*2]
    movq mm5,[REF+YSTRIDE*2]
    movq mm6,[SRC+YSTRIDE3]
    movq mm7,[REF+YSTRIDE3]
    /*Compute their SADs and add them in mm0*/
    psadbw mm0,mm1
    psadbw mm2,mm3
    lea SRC,[SRC+YSTRIDE*4]
    paddw mm0,mm2
    lea REF,[REF+YSTRIDE*4]
    /*Load the next 3 rows as registers become available.*/
    movq mm2,[SRC]
    movq mm3,[REF]
    psadbw mm4,mm5
    psadbw mm6,mm7
    paddw mm0,mm4
    movq mm5,[REF+YSTRIDE]
    movq mm4,[SRC+YSTRIDE]
    paddw mm0,mm6
    movq mm7,[REF+YSTRIDE*2]
    movq mm6,[SRC+YSTRIDE*2]
    /*Start adding their SADs to mm0*/
    psadbw mm2,mm3
    psadbw mm4,mm5
    paddw mm0,mm2
    psadbw mm6,mm7
    /*Load last row as registers become available.*/
    movq mm2,[SRC+YSTRIDE3]
    movq mm3,[REF+YSTRIDE3]
    /*And finish adding up their SADs.*/
    paddw mm0,mm4
    psadbw mm2,mm3
    paddw mm0,mm6
    paddw mm0,mm2
    movd [ret],mm0
#undef SRC
#undef REF
#undef YSTRIDE
#undef YSTRIDE3
  }
  return (unsigned)ret;
}

unsigned oc_enc_frag_sad_thresh_mmxext(const unsigned char *_src,
 const unsigned char *_ref,int _ystride,unsigned _thresh){
  /*Early termination is for suckers.*/
  return oc_enc_frag_sad_mmxext(_src,_ref,_ystride);
}

#define OC_SAD2_LOOP __asm{ \
  /*We want to compute (mm0+mm1>>1) on unsigned bytes without overflow, but \
     pavgb computes (mm0+mm1+1>>1). \
   The latter is exactly 1 too large when the low bit of two corresponding \
    bytes is only set in one of them. \
   Therefore we pxor the operands, pand to mask out the low bits, and psubb to \
    correct the output of pavgb.*/ \
  __asm  movq mm6,mm0 \
  __asm  lea REF1,[REF1+YSTRIDE*2] \
  __asm  pxor mm0,mm1 \
  __asm  pavgb mm6,mm1 \
  __asm  lea REF2,[REF2+YSTRIDE*2] \
  __asm  movq mm1,mm2 \
  __asm  pand mm0,mm7 \
  __asm  pavgb mm2,mm3 \
  __asm  pxor mm1,mm3 \
  __asm  movq mm3,[REF2+YSTRIDE] \
  __asm  psubb mm6,mm0 \
  __asm  movq mm0,[REF1] \
  __asm  pand mm1,mm7 \
  __asm  psadbw mm4,mm6 \
  __asm  movd mm6,RET \
  __asm  psubb mm2,mm1 \
  __asm  movq mm1,[REF2] \
  __asm  lea SRC,[SRC+YSTRIDE*2] \
  __asm  psadbw mm5,mm2 \
  __asm  movq mm2,[REF1+YSTRIDE] \
  __asm  paddw mm5,mm4 \
  __asm  movq mm4,[SRC] \
  __asm  paddw mm6,mm5 \
  __asm  movq mm5,[SRC+YSTRIDE] \
  __asm  movd RET,mm6 \
}

/*Same as above, but does not pre-load the next two rows.*/
#define OC_SAD2_TAIL __asm{ \
  __asm  movq mm6,mm0 \
  __asm  pavgb mm0,mm1 \
  __asm  pxor mm6,mm1 \
  __asm  movq mm1,mm2 \
  __asm  pand mm6,mm7 \
  __asm  pavgb mm2,mm3 \
  __asm  pxor mm1,mm3 \
  __asm  psubb mm0,mm6 \
  __asm  pand mm1,mm7 \
  __asm  psadbw mm4,mm0 \
  __asm  psubb mm2,mm1 \
  __asm  movd mm6,RET \
  __asm  psadbw mm5,mm2 \
  __asm  paddw mm5,mm4 \
  __asm  paddw mm6,mm5 \
  __asm  movd RET,mm6 \
}

unsigned oc_enc_frag_sad2_thresh_mmxext(const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride,
 unsigned _thresh){
  ptrdiff_t ret;
  __asm{
#define REF1 ecx
#define REF2 edi
#define YSTRIDE esi
#define SRC edx
#define RET eax
    mov YSTRIDE,_ystride
    mov SRC,_src
    mov REF1,_ref1
    mov REF2,_ref2
    movq mm0,[REF1]
    movq mm1,[REF2]
    movq mm2,[REF1+YSTRIDE]
    movq mm3,[REF2+YSTRIDE]
    xor RET,RET
    movq mm4,[SRC]
    pxor mm7,mm7
    pcmpeqb mm6,mm6
    movq mm5,[SRC+YSTRIDE]
    psubb mm7,mm6
    OC_SAD2_LOOP
    OC_SAD2_LOOP
    OC_SAD2_LOOP
    OC_SAD2_TAIL
    mov [ret],RET
#undef REF1
#undef REF2
#undef YSTRIDE
#undef SRC
#undef RET
  }
  return (unsigned)ret;
}

/*Load an 8x4 array of pixel values from %[src] and %[ref] and compute their
  16-bit difference in mm0...mm7.*/
#define OC_LOAD_SUB_8x4(_off) __asm{ \
  __asm  movd mm0,[_off+SRC] \
  __asm  movd mm4,[_off+REF] \
  __asm  movd mm1,[_off+SRC+SRC_YSTRIDE] \
  __asm  lea SRC,[SRC+SRC_YSTRIDE*2] \
  __asm  movd mm5,[_off+REF+REF_YSTRIDE] \
  __asm  lea REF,[REF+REF_YSTRIDE*2] \
  __asm  movd mm2,[_off+SRC] \
  __asm  movd mm7,[_off+REF] \
  __asm  movd mm3,[_off+SRC+SRC_YSTRIDE] \
  __asm  movd mm6,[_off+REF+REF_YSTRIDE] \
  __asm  punpcklbw mm0,mm4 \
  __asm  lea SRC,[SRC+SRC_YSTRIDE*2] \
  __asm  punpcklbw mm4,mm4 \
  __asm  lea REF,[REF+REF_YSTRIDE*2] \
  __asm  psubw mm0,mm4 \
  __asm  movd mm4,[_off+SRC] \
  __asm  movq [_off*2+BUF],mm0 \
  __asm  movd mm0,[_off+REF] \
  __asm  punpcklbw mm1,mm5 \
  __asm  punpcklbw mm5,mm5 \
  __asm  psubw mm1,mm5 \
  __asm  movd mm5,[_off+SRC+SRC_YSTRIDE] \
  __asm  punpcklbw mm2,mm7 \
  __asm  punpcklbw mm7,mm7 \
  __asm  psubw mm2,mm7 \
  __asm  movd mm7,[_off+REF+REF_YSTRIDE] \
  __asm  punpcklbw mm3,mm6 \
  __asm  lea SRC,[SRC+SRC_YSTRIDE*2] \
  __asm  punpcklbw mm6,mm6 \
  __asm  psubw mm3,mm6 \
  __asm  movd mm6,[_off+SRC] \
  __asm  punpcklbw mm4,mm0 \
  __asm  lea REF,[REF+REF_YSTRIDE*2] \
  __asm  punpcklbw mm0,mm0 \
  __asm  lea SRC,[SRC+SRC_YSTRIDE*2] \
  __asm  psubw mm4,mm0 \
  __asm  movd mm0,[_off+REF] \
  __asm  punpcklbw mm5,mm7 \
  __asm  neg SRC_YSTRIDE \
  __asm  punpcklbw mm7,mm7 \
  __asm  psubw mm5,mm7 \
  __asm  movd mm7,[_off+SRC+SRC_YSTRIDE] \
  __asm  punpcklbw mm6,mm0 \
  __asm  lea REF,[REF+REF_YSTRIDE*2] \
  __asm  punpcklbw mm0,mm0 \
  __asm  neg REF_YSTRIDE \
  __asm  psubw mm6,mm0 \
  __asm  movd mm0,[_off+REF+REF_YSTRIDE] \
  __asm  lea SRC,[SRC+SRC_YSTRIDE*8] \
  __asm  punpcklbw mm7,mm0 \
  __asm  neg SRC_YSTRIDE \
  __asm  punpcklbw mm0,mm0 \
  __asm  lea REF,[REF+REF_YSTRIDE*8] \
  __asm  psubw mm7,mm0 \
  __asm  neg REF_YSTRIDE \
  __asm  movq mm0,[_off*2+BUF] \
}

/*Load an 8x4 array of pixel values from %[src] into %%mm0...%%mm7.*/
#define OC_LOAD_8x4(_off) __asm{ \
  __asm  movd mm0,[_off+SRC] \
  __asm  movd mm1,[_off+SRC+YSTRIDE] \
  __asm  movd mm2,[_off+SRC+YSTRIDE*2] \
  __asm  pxor mm7,mm7 \
  __asm  movd mm3,[_off+SRC+YSTRIDE3] \
  __asm  punpcklbw mm0,mm7 \
  __asm  movd mm4,[_off+SRC4] \
  __asm  punpcklbw mm1,mm7 \
  __asm  movd mm5,[_off+SRC4+YSTRIDE] \
  __asm  punpcklbw mm2,mm7 \
  __asm  movd mm6,[_off+SRC4+YSTRIDE*2] \
  __asm  punpcklbw mm3,mm7 \
  __asm  movd mm7,[_off+SRC4+YSTRIDE3] \
  __asm  punpcklbw mm4,mm4 \
  __asm  punpcklbw mm5,mm5 \
  __asm  psrlw mm4,8 \
  __asm  psrlw mm5,8 \
  __asm  punpcklbw mm6,mm6 \
  __asm  punpcklbw mm7,mm7 \
  __asm  psrlw mm6,8 \
  __asm  psrlw mm7,8 \
}

/*Performs the first two stages of an 8-point 1-D Hadamard transform.
  The transform is performed in place, except that outputs 0-3 are swapped with
   outputs 4-7.
  Outputs 2, 3, 6, and 7 from the second stage are negated (which allows us to
   perform this stage in place with no temporary registers).*/
#define OC_HADAMARD_AB_8x4 __asm{ \
  /*Stage A: \
    Outputs 0-3 are swapped with 4-7 here.*/ \
  __asm  paddw mm5,mm1 \
  __asm  paddw mm6,mm2 \
  __asm  paddw mm1,mm1 \
  __asm  paddw mm2,mm2 \
  __asm  psubw mm1,mm5 \
  __asm  psubw mm2,mm6 \
  __asm  paddw mm7,mm3 \
  __asm  paddw mm4,mm0 \
  __asm  paddw mm3,mm3 \
  __asm  paddw mm0,mm0 \
  __asm  psubw mm3,mm7 \
  __asm  psubw mm0,mm4 \
   /*Stage B:*/ \
  __asm  paddw mm0,mm2 \
  __asm  paddw mm1,mm3 \
  __asm  paddw mm4,mm6 \
  __asm  paddw mm5,mm7 \
  __asm  paddw mm2,mm2 \
  __asm  paddw mm3,mm3 \
  __asm  paddw mm6,mm6 \
  __asm  paddw mm7,mm7 \
  __asm  psubw mm2,mm0 \
  __asm  psubw mm3,mm1 \
  __asm  psubw mm6,mm4 \
  __asm  psubw mm7,mm5 \
}

/*Performs the last stage of an 8-point 1-D Hadamard transform in place.
  Outputs 1, 3, 5, and 7 are negated (which allows us to perform this stage in
   place with no temporary registers).*/
#define OC_HADAMARD_C_8x4 __asm{ \
  /*Stage C:*/ \
  __asm  paddw mm0,mm1 \
  __asm  paddw mm2,mm3 \
  __asm  paddw mm4,mm5 \
  __asm  paddw mm6,mm7 \
  __asm  paddw mm1,mm1 \
  __asm  paddw mm3,mm3 \
  __asm  paddw mm5,mm5 \
  __asm  paddw mm7,mm7 \
  __asm  psubw mm1,mm0 \
  __asm  psubw mm3,mm2 \
  __asm  psubw mm5,mm4 \
  __asm  psubw mm7,mm6 \
}

/*Performs an 8-point 1-D Hadamard transform.
  The transform is performed in place, except that outputs 0-3 are swapped with
   outputs 4-7.
  Outputs 1, 2, 5 and 6 are negated (which allows us to perform the transform
   in place with no temporary registers).*/
#define OC_HADAMARD_8x4 __asm{ \
  OC_HADAMARD_AB_8x4 \
  OC_HADAMARD_C_8x4 \
}

/*Performs the first part of the final stage of the Hadamard transform and
   summing of absolute values.
  At the end of this part, mm1 will contain the DC coefficient of the
   transform.*/
#define OC_HADAMARD_C_ABS_ACCUM_A_8x4(_r6,_r7) __asm{ \
  /*We use the fact that \
      (abs(a+b)+abs(a-b))/2=max(abs(a),abs(b)) \
     to merge the final butterfly with the abs and the first stage of \
     accumulation. \
    Thus we can avoid using pabsw, which is not available until SSSE3. \
    Emulating pabsw takes 3 instructions, so the straightforward MMXEXT \
     implementation would be (3+3)*8+7=55 instructions (+4 for spilling \
     registers). \
    Even with pabsw, it would be (3+1)*8+7=39 instructions (with no spills). \
    This implementation is only 26 (+4 for spilling registers).*/ \
  __asm  movq [_r7+BUF],mm7 \
  __asm  movq [_r6+BUF],mm6 \
  /*mm7={0x7FFF}x4 \
    mm0=max(abs(mm0),abs(mm1))-0x7FFF*/ \
  __asm  pcmpeqb mm7,mm7 \
  __asm  movq mm6,mm0 \
  __asm  psrlw mm7,1 \
  __asm  paddw mm6,mm1 \
  __asm  pmaxsw mm0,mm1 \
  __asm  paddsw mm6,mm7 \
  __asm  psubw mm0,mm6 \
  /*mm2=max(abs(mm2),abs(mm3))-0x7FFF \
    mm4=max(abs(mm4),abs(mm5))-0x7FFF*/ \
  __asm  movq mm6,mm2 \
  __asm  movq mm1,mm4 \
  __asm  pmaxsw mm2,mm3 \
  __asm  pmaxsw mm4,mm5 \
  __asm  paddw mm6,mm3 \
  __asm  paddw mm1,mm5 \
  __asm  movq mm3,[_r7+BUF] \
}

/*Performs the second part of the final stage of the Hadamard transform and
   summing of absolute values.*/
#define OC_HADAMARD_C_ABS_ACCUM_B_8x4(_r6,_r7) __asm{ \
  __asm  paddsw mm6,mm7 \
  __asm  movq mm5,[_r6+BUF] \
  __asm  paddsw mm1,mm7 \
  __asm  psubw mm2,mm6 \
  __asm  psubw mm4,mm1 \
  /*mm7={1}x4 (needed for the horizontal add that follows) \
    mm0+=mm2+mm4+max(abs(mm3),abs(mm5))-0x7FFF*/ \
  __asm  movq mm6,mm3 \
  __asm  pmaxsw mm3,mm5 \
  __asm  paddw mm0,mm2 \
  __asm  paddw mm6,mm5 \
  __asm  paddw mm0,mm4 \
  __asm  paddsw mm6,mm7 \
  __asm  paddw mm0,mm3 \
  __asm  psrlw mm7,14 \
  __asm  psubw mm0,mm6 \
}

/*Performs the last stage of an 8-point 1-D Hadamard transform, takes the
   absolute value of each component, and accumulates everything into mm0.
  This is the only portion of SATD which requires MMXEXT (we could use plain
   MMX, but it takes 4 instructions and an extra register to work around the
   lack of a pmaxsw, which is a pretty serious penalty).*/
#define OC_HADAMARD_C_ABS_ACCUM_8x4(_r6,_r7) __asm{ \
  OC_HADAMARD_C_ABS_ACCUM_A_8x4(_r6,_r7) \
  OC_HADAMARD_C_ABS_ACCUM_B_8x4(_r6,_r7) \
}

/*Performs an 8-point 1-D Hadamard transform, takes the absolute value of each
   component, and accumulates everything into mm0.
  Note that mm0 will have an extra 4 added to each column, and that after
   removing this value, the remainder will be half the conventional value.*/
#define OC_HADAMARD_ABS_ACCUM_8x4(_r6,_r7) __asm{ \
  OC_HADAMARD_AB_8x4 \
  OC_HADAMARD_C_ABS_ACCUM_8x4(_r6,_r7) \
}

/*Performs two 4x4 transposes (mostly) in place.
  On input, {mm0,mm1,mm2,mm3} contains rows {e,f,g,h}, and {mm4,mm5,mm6,mm7}
   contains rows {a,b,c,d}.
  On output, {0x40,0x50,0x60,0x70}+_off+BUF contains {e,f,g,h}^T, and
   {mm4,mm5,mm6,mm7} contains the transposed rows {a,b,c,d}^T.*/
#define OC_TRANSPOSE_4x4x2(_off) __asm{ \
  /*First 4x4 transpose:*/ \
  __asm  movq [0x10+_off+BUF],mm5 \
  /*mm0 = e3 e2 e1 e0 \
    mm1 = f3 f2 f1 f0 \
    mm2 = g3 g2 g1 g0 \
    mm3 = h3 h2 h1 h0*/ \
  __asm  movq mm5,mm2 \
  __asm  punpcklwd mm2,mm3 \
  __asm  punpckhwd mm5,mm3 \
  __asm  movq mm3,mm0 \
  __asm  punpcklwd mm0,mm1 \
  __asm  punpckhwd mm3,mm1 \
  /*mm0 = f1 e1 f0 e0 \
    mm3 = f3 e3 f2 e2 \
    mm2 = h1 g1 h0 g0 \
    mm5 = h3 g3 h2 g2*/ \
  __asm  movq mm1,mm0 \
  __asm  punpckldq mm0,mm2 \
  __asm  punpckhdq mm1,mm2 \
  __asm  movq mm2,mm3 \
  __asm  punpckhdq mm3,mm5 \
  __asm  movq [0x40+_off+BUF],mm0 \
  __asm  punpckldq mm2,mm5 \
  /*mm0 = h0 g0 f0 e0 \
    mm1 = h1 g1 f1 e1 \
    mm2 = h2 g2 f2 e2 \
    mm3 = h3 g3 f3 e3*/ \
  __asm  movq mm5,[0x10+_off+BUF] \
  /*Second 4x4 transpose:*/ \
  /*mm4 = a3 a2 a1 a0 \
    mm5 = b3 b2 b1 b0 \
    mm6 = c3 c2 c1 c0 \
    mm7 = d3 d2 d1 d0*/ \
  __asm  movq mm0,mm6 \
  __asm  punpcklwd mm6,mm7 \
  __asm  movq [0x50+_off+BUF],mm1 \
  __asm  punpckhwd mm0,mm7 \
  __asm  movq mm7,mm4 \
  __asm  punpcklwd mm4,mm5 \
  __asm  movq [0x60+_off+BUF],mm2 \
  __asm  punpckhwd mm7,mm5 \
  /*mm4 = b1 a1 b0 a0 \
    mm7 = b3 a3 b2 a2 \
    mm6 = d1 c1 d0 c0 \
    mm0 = d3 c3 d2 c2*/ \
  __asm  movq mm5,mm4 \
  __asm  punpckldq mm4,mm6 \
  __asm  movq [0x70+_off+BUF],mm3 \
  __asm  punpckhdq mm5,mm6 \
  __asm  movq mm6,mm7 \
  __asm  punpckhdq mm7,mm0 \
  __asm  punpckldq mm6,mm0 \
  /*mm4 = d0 c0 b0 a0 \
    mm5 = d1 c1 b1 a1 \
    mm6 = d2 c2 b2 a2 \
    mm7 = d3 c3 b3 a3*/ \
}

static unsigned oc_int_frag_satd_mmxext(int *_dc,
 const unsigned char *_src,int _src_ystride,
 const unsigned char *_ref,int _ref_ystride){
  OC_ALIGN8(ogg_int16_t buf[64]);
  ogg_int16_t *bufp;
  unsigned     ret;
  unsigned     ret2;
  int          dc;
  bufp=buf;
  __asm{
#define SRC esi
#define REF eax
#define SRC_YSTRIDE ecx
#define REF_YSTRIDE edx
#define BUF edi
#define RET edx
#define RET2 ecx
#define DC eax
#define DC_WORD ax
    mov SRC,_src
    mov SRC_YSTRIDE,_src_ystride
    mov REF,_ref
    mov REF_YSTRIDE,_ref_ystride
    mov BUF,bufp
    OC_LOAD_SUB_8x4(0x00)
    OC_HADAMARD_8x4
    OC_TRANSPOSE_4x4x2(0x00)
    /*Finish swapping out this 8x4 block to make room for the next one.
      mm0...mm3 have been swapped out already.*/
    movq [0x00+BUF],mm4
    movq [0x10+BUF],mm5
    movq [0x20+BUF],mm6
    movq [0x30+BUF],mm7
    OC_LOAD_SUB_8x4(0x04)
    OC_HADAMARD_8x4
    OC_TRANSPOSE_4x4x2(0x08)
    /*Here the first 4x4 block of output from the last transpose is the second
       4x4 block of input for the next transform.
      We have cleverly arranged that it already be in the appropriate place, so
       we only have to do half the loads.*/
    movq mm1,[0x10+BUF]
    movq mm2,[0x20+BUF]
    movq mm3,[0x30+BUF]
    movq mm0,[0x00+BUF]
    /*We split out the stages here so we can save the DC coefficient in the
       middle.*/
    OC_HADAMARD_AB_8x4
    OC_HADAMARD_C_ABS_ACCUM_A_8x4(0x28,0x38)
    movd DC,mm1
    OC_HADAMARD_C_ABS_ACCUM_B_8x4(0x28,0x38)
    /*Up to this point, everything fit in 16 bits (8 input + 1 for the
       difference + 2*3 for the two 8-point 1-D Hadamards - 1 for the abs - 1
       for the factor of two we dropped + 3 for the vertical accumulation).
      Now we finally have to promote things to dwords.
      We break this part out of OC_HADAMARD_ABS_ACCUM_8x4 to hide the long
       latency of pmaddwd by starting the next series of loads now.*/
    pmaddwd mm0,mm7
    movq mm1,[0x50+BUF]
    movq mm5,[0x58+BUF]
    movq mm4,mm0
    movq mm2,[0x60+BUF]
    punpckhdq mm0,mm0
    movq mm6,[0x68+BUF]
    paddd mm4,mm0
    movq mm3,[0x70+BUF]
    movd RET2,mm4
    movq mm7,[0x78+BUF]
    movq mm0,[0x40+BUF]
    movq mm4,[0x48+BUF]
    OC_HADAMARD_ABS_ACCUM_8x4(0x68,0x78)
    pmaddwd mm0,mm7
    /*Subtract abs(dc) from 2*ret2.*/
    movsx DC,DC_WORD
    cdq
    lea RET2,[RET+RET2*2]
    movq mm4,mm0
    punpckhdq mm0,mm0
    xor RET,DC
    paddd mm4,mm0
    /*The sums produced by OC_HADAMARD_ABS_ACCUM_8x4 each have an extra 4
       added to them, a factor of two removed, and the DC value included;
       correct the final sum here.*/
    sub RET2,RET
    movd RET,mm4
    lea RET,[RET2+RET*2-64]
    mov ret,RET
    mov dc,DC
#undef SRC
#undef REF
#undef SRC_YSTRIDE
#undef REF_YSTRIDE
#undef BUF
#undef RET
#undef RET2
#undef DC
#undef DC_WORD
  }
  *_dc=dc;
  return ret;
}

unsigned oc_enc_frag_satd_mmxext(int *_dc,const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  return oc_int_frag_satd_mmxext(_dc,_src,_ystride,_ref,_ystride);
}


/*Our internal implementation of frag_copy2 takes an extra stride parameter so
   we can share code with oc_enc_frag_satd2_mmxext().*/
static void oc_int_frag_copy2_mmxext(unsigned char *_dst,int _dst_ystride,
 const unsigned char *_src1,const unsigned char *_src2,int _src_ystride){
  __asm{
    /*Load the first 3 rows.*/
#define DST_YSTRIDE edi
#define SRC_YSTRIDE esi
#define DST eax
#define SRC1 edx
#define SRC2 ecx
    mov DST_YSTRIDE,_dst_ystride
    mov SRC_YSTRIDE,_src_ystride
    mov DST,_dst
    mov SRC1,_src1
    mov SRC2,_src2
    movq mm0,[SRC1]
    movq mm1,[SRC2]
    movq mm2,[SRC1+SRC_YSTRIDE]
    lea SRC1,[SRC1+SRC_YSTRIDE*2]
    movq mm3,[SRC2+SRC_YSTRIDE]
    lea SRC2,[SRC2+SRC_YSTRIDE*2]
    pxor mm7,mm7
    movq mm4,[SRC1]
    pcmpeqb mm6,mm6
    movq mm5,[SRC2]
    /*mm7={1}x8.*/
    psubb mm7,mm6
    /*Start averaging mm0 and mm1 into mm6.*/
    movq mm6,mm0
    pxor mm0,mm1
    pavgb mm6,mm1
    /*mm1 is free, start averaging mm3 into mm2 using mm1.*/
    movq mm1,mm2
    pand mm0,mm7
    pavgb mm2,mm3
    pxor mm1,mm3
    /*mm3 is free.*/
    psubb mm6,mm0
    /*mm0 is free, start loading the next row.*/
    movq mm0,[SRC1+SRC_YSTRIDE]
    /*Start averaging mm5 and mm4 using mm3.*/
    movq mm3,mm4
    /*mm6 [row 0] is done; write it out.*/
    movq [DST],mm6
    pand mm1,mm7
    pavgb mm4,mm5
    psubb mm2,mm1
    /*mm1 is free, continue loading the next row.*/
    movq mm1,[SRC2+SRC_YSTRIDE]
    pxor mm3,mm5
    lea SRC1,[SRC1+SRC_YSTRIDE*2]
    /*mm2 [row 1] is done; write it out.*/
    movq [DST+DST_YSTRIDE],mm2
    pand mm3,mm7
    /*Start loading the next row.*/
    movq mm2,[SRC1]
    lea DST,[DST+DST_YSTRIDE*2]
    psubb mm4,mm3
    lea SRC2,[SRC2+SRC_YSTRIDE*2]
    /*mm4 [row 2] is done; write it out.*/
    movq [DST],mm4
    /*Continue loading the next row.*/
    movq mm3,[SRC2]
    /*Start averaging mm0 and mm1 into mm6.*/
    movq mm6,mm0
    pxor mm0,mm1
    /*Start loading the next row.*/
    movq mm4,[SRC1+SRC_YSTRIDE]
    pavgb mm6,mm1
    /*mm1 is free; start averaging mm3 into mm2 using mm1.*/
    movq mm1,mm2
    pand mm0,mm7
    /*Continue loading the next row.*/
    movq mm5,[SRC2+SRC_YSTRIDE]
    pavgb mm2,mm3
    lea SRC1,[SRC1+SRC_YSTRIDE*2]
    pxor mm1,mm3
    /*mm3 is free.*/
    psubb mm6,mm0
    /*mm0 is free, start loading the next row.*/
    movq mm0,[SRC1]
    /*Start averaging mm5 into mm4 using mm3.*/
    movq mm3,mm4
    /*mm6 [row 3] is done; write it out.*/
    movq [DST+DST_YSTRIDE],mm6
    pand mm1,mm7
    lea SRC2,[SRC2+SRC_YSTRIDE*2]
    pavgb mm4,mm5
    lea DST,[DST+DST_YSTRIDE*2]
    psubb mm2,mm1
    /*mm1 is free; continue loading the next row.*/
    movq mm1,[SRC2]
    pxor mm3,mm5
    /*mm2 [row 4] is done; write it out.*/
    movq [DST],mm2
    pand mm3,mm7
    /*Start loading the next row.*/
    movq mm2,[SRC1+SRC_YSTRIDE]
    psubb mm4,mm3
    /*Start averaging mm0 and mm1 into mm6.*/
    movq mm6,mm0
    /*Continue loading the next row.*/
    movq mm3,[SRC2+SRC_YSTRIDE]
    /*mm4 [row 5] is done; write it out.*/
    movq [DST+DST_YSTRIDE],mm4
    pxor mm0,mm1
    pavgb mm6,mm1
    /*mm4 is free; start averaging mm3 into mm2 using mm4.*/
    movq mm4,mm2
    pand mm0,mm7
    pavgb mm2,mm3
    pxor mm4,mm3
    lea DST,[DST+DST_YSTRIDE*2]
    psubb mm6,mm0
    pand mm4,mm7
    /*mm6 [row 6] is done, write it out.*/
    movq [DST],mm6
    psubb mm2,mm4
    /*mm2 [row 7] is done, write it out.*/
    movq [DST+DST_YSTRIDE],mm2
#undef SRC1
#undef SRC2
#undef SRC_YSTRIDE
#undef DST_YSTRIDE
#undef DST
  }
}

unsigned oc_enc_frag_satd2_mmxext(int *_dc,const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride){
  OC_ALIGN8(unsigned char ref[64]);
  oc_int_frag_copy2_mmxext(ref,8,_ref1,_ref2,_ystride);
  return oc_int_frag_satd_mmxext(_dc,_src,_ystride,ref,8);
}

unsigned oc_enc_frag_intra_satd_mmxext(int *_dc,const unsigned char *_src,
 int _ystride){
  OC_ALIGN8(ogg_int16_t buf[64]);
  ogg_int16_t *bufp;
  unsigned     ret1;
  unsigned     ret2;
  int          dc;
  bufp=buf;
  __asm{
#define SRC eax
#define SRC4 esi
#define BUF edi
#define YSTRIDE edx
#define YSTRIDE3 ecx
#define RET eax
#define RET2 ecx
#define DC edx
#define DC_WORD dx
    mov SRC,_src
    mov BUF,bufp
    mov YSTRIDE,_ystride
    /* src4 = src+4*ystride */
    lea SRC4,[SRC+YSTRIDE*4]
    /* ystride3 = 3*ystride */
    lea YSTRIDE3,[YSTRIDE+YSTRIDE*2]
    OC_LOAD_8x4(0x00)
    OC_HADAMARD_8x4
    OC_TRANSPOSE_4x4x2(0x00)
    /*Finish swapping out this 8x4 block to make room for the next one.
      mm0...mm3 have been swapped out already.*/
    movq [0x00+BUF],mm4
    movq [0x10+BUF],mm5
    movq [0x20+BUF],mm6
    movq [0x30+BUF],mm7
    OC_LOAD_8x4(0x04)
    OC_HADAMARD_8x4
    OC_TRANSPOSE_4x4x2(0x08)
    /*Here the first 4x4 block of output from the last transpose is the second
      4x4 block of input for the next transform.
      We have cleverly arranged that it already be in the appropriate place, so
      we only have to do half the loads.*/
    movq mm1,[0x10+BUF]
    movq mm2,[0x20+BUF]
    movq mm3,[0x30+BUF]
    movq mm0,[0x00+BUF]
    /*We split out the stages here so we can save the DC coefficient in the
      middle.*/
    OC_HADAMARD_AB_8x4
    OC_HADAMARD_C_ABS_ACCUM_A_8x4(0x28,0x38)
    movd DC,mm1
    OC_HADAMARD_C_ABS_ACCUM_B_8x4(0x28,0x38)
    /*Up to this point, everything fit in 16 bits (8 input + 1 for the
      difference + 2*3 for the two 8-point 1-D Hadamards - 1 for the abs - 1
      for the factor of two we dropped + 3 for the vertical accumulation).
      Now we finally have to promote things to dwords.
      We break this part out of OC_HADAMARD_ABS_ACCUM_8x4 to hide the long
      latency of pmaddwd by starting the next series of loads now.*/
    pmaddwd mm0,mm7
    movq mm1,[0x50+BUF]
    movq mm5,[0x58+BUF]
    movq mm2,[0x60+BUF]
    movq mm4,mm0
    movq mm6,[0x68+BUF]
    punpckhdq mm0,mm0
    movq mm3,[0x70+BUF]
    paddd mm4,mm0
    movq mm7,[0x78+BUF]
    movd RET,mm4
    movq mm0,[0x40+BUF]
    movq mm4,[0x48+BUF]
    OC_HADAMARD_ABS_ACCUM_8x4(0x68,0x78)
    pmaddwd mm0,mm7
    /*We assume that the DC coefficient is always positive (which is true,
    because the input to the INTRA transform was not a difference).*/
    movzx DC,DC_WORD
    add RET,RET
    sub RET,DC
    movq mm4,mm0
    punpckhdq mm0,mm0
    paddd mm4,mm0
    movd RET2,mm4
    lea RET,[-64+RET+RET2*2]
    mov [dc],DC
    mov [ret1],RET
#undef SRC
#undef SRC4
#undef BUF
#undef YSTRIDE
#undef YSTRIDE3
#undef RET
#undef RET2
#undef DC
#undef DC_WORD
  }
  *_dc=dc;
  return ret1;
}

void oc_enc_frag_sub_mmx(ogg_int16_t _residue[64],
 const unsigned char *_src, const unsigned char *_ref,int _ystride){
  int i;
  __asm  pxor mm7,mm7
  for(i=4;i-->0;){
    __asm{
#define SRC edx
#define YSTRIDE esi
#define RESIDUE eax
#define REF ecx
      mov YSTRIDE,_ystride
      mov RESIDUE,_residue
      mov SRC,_src
      mov REF,_ref
      /*mm0=[src]*/
      movq mm0,[SRC]
      /*mm1=[ref]*/
      movq mm1,[REF]
      /*mm4=[src+ystride]*/
      movq mm4,[SRC+YSTRIDE]
      /*mm5=[ref+ystride]*/
      movq mm5,[REF+YSTRIDE]
      /*Compute [src]-[ref].*/
      movq mm2,mm0
      punpcklbw mm0,mm7
      movq mm3,mm1
      punpckhbw mm2,mm7
      punpcklbw mm1,mm7
      punpckhbw mm3,mm7
      psubw mm0,mm1
      psubw mm2,mm3
      /*Compute [src+ystride]-[ref+ystride].*/
      movq mm1,mm4
      punpcklbw mm4,mm7
      movq mm3,mm5
      punpckhbw mm1,mm7
      lea SRC,[SRC+YSTRIDE*2]
      punpcklbw mm5,mm7
      lea REF,[REF+YSTRIDE*2]
      punpckhbw mm3,mm7
      psubw mm4,mm5
      psubw mm1,mm3
      /*Write the answer out.*/
      movq [RESIDUE+0x00],mm0
      movq [RESIDUE+0x08],mm2
      movq [RESIDUE+0x10],mm4
      movq [RESIDUE+0x18],mm1
      lea RESIDUE,[RESIDUE+0x20]
      mov _residue,RESIDUE
      mov _src,SRC
      mov _ref,REF
#undef SRC
#undef YSTRIDE
#undef RESIDUE
#undef REF
    }
  }
}

void oc_enc_frag_sub_128_mmx(ogg_int16_t _residue[64],
 const unsigned char *_src,int _ystride){
   __asm{
#define YSTRIDE edx
#define YSTRIDE3 edi
#define RESIDUE ecx
#define SRC eax
    mov YSTRIDE,_ystride
    mov RESIDUE,_residue
    mov SRC,_src
    /*mm0=[src]*/
    movq mm0,[SRC]
    /*mm1=[src+ystride]*/
    movq mm1,[SRC+YSTRIDE]
    /*mm6={-1}x4*/
    pcmpeqw mm6,mm6
    /*mm2=[src+2*ystride]*/
    movq mm2,[SRC+YSTRIDE*2]
    /*[ystride3]=3*[ystride]*/
    lea YSTRIDE3,[YSTRIDE+YSTRIDE*2]
    /*mm6={1}x4*/
    psllw mm6,15
    /*mm3=[src+3*ystride]*/
    movq mm3,[SRC+YSTRIDE3]
    /*mm6={128}x4*/
    psrlw mm6,8
    /*mm7=0*/ 
    pxor mm7,mm7
    /*[src]=[src]+4*[ystride]*/
    lea SRC,[SRC+YSTRIDE*4]
    /*Compute [src]-128 and [src+ystride]-128*/
    movq mm4,mm0
    punpcklbw mm0,mm7
    movq mm5,mm1
    punpckhbw mm4,mm7
    psubw mm0,mm6
    punpcklbw mm1,mm7
    psubw mm4,mm6
    punpckhbw mm5,mm7
    psubw mm1,mm6
    psubw mm5,mm6
    /*Write the answer out.*/
    movq [RESIDUE+0x00],mm0
    movq [RESIDUE+0x08],mm4
    movq [RESIDUE+0x10],mm1
    movq [RESIDUE+0x18],mm5
    /*mm0=[src+4*ystride]*/
    movq mm0,[SRC]
    /*mm1=[src+5*ystride]*/
    movq mm1,[SRC+YSTRIDE]
    /*Compute [src+2*ystride]-128 and [src+3*ystride]-128*/
    movq mm4,mm2
    punpcklbw mm2,mm7
    movq mm5,mm3
    punpckhbw mm4,mm7
    psubw mm2,mm6
    punpcklbw mm3,mm7
    psubw mm4,mm6
    punpckhbw mm5,mm7
    psubw mm3,mm6
    psubw mm5,mm6
    /*Write the answer out.*/
    movq [RESIDUE+0x20],mm2
    movq [RESIDUE+0x28],mm4
    movq [RESIDUE+0x30],mm3
    movq [RESIDUE+0x38],mm5
    /*Compute [src+6*ystride]-128 and [src+7*ystride]-128*/
    movq mm2,[SRC+YSTRIDE*2]
    movq mm3,[SRC+YSTRIDE3]
    movq mm4,mm0
    punpcklbw mm0,mm7
    movq mm5,mm1
    punpckhbw mm4,mm7
    psubw mm0,mm6
    punpcklbw mm1,mm7
    psubw mm4,mm6
    punpckhbw mm5,mm7
    psubw mm1,mm6
    psubw mm5,mm6
    /*Write the answer out.*/
    movq [RESIDUE+0x40],mm0
    movq [RESIDUE+0x48],mm4
    movq [RESIDUE+0x50],mm1
    movq [RESIDUE+0x58],mm5
    /*Compute [src+6*ystride]-128 and [src+7*ystride]-128*/
    movq mm4,mm2
    punpcklbw mm2,mm7
    movq mm5,mm3
    punpckhbw mm4,mm7
    psubw mm2,mm6
    punpcklbw mm3,mm7
    psubw mm4,mm6
    punpckhbw mm5,mm7
    psubw mm3,mm6
    psubw mm5,mm6
    /*Write the answer out.*/
    movq [RESIDUE+0x60],mm2
    movq [RESIDUE+0x68],mm4
    movq [RESIDUE+0x70],mm3
    movq [RESIDUE+0x78],mm5
#undef YSTRIDE
#undef YSTRIDE3
#undef RESIDUE
#undef SRC
  }
}

void oc_enc_frag_copy2_mmxext(unsigned char *_dst,
 const unsigned char *_src1,const unsigned char *_src2,int _ystride){
  oc_int_frag_copy2_mmxext(_dst,_ystride,_src1,_src2,_ystride);
}

#endif
