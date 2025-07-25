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
  ptrdiff_t ystride3;
  ptrdiff_t ret;
  __asm__ __volatile__(
    /*Load the first 4 rows of each block.*/
    "movq (%[src]),%%mm0\n\t"
    "movq (%[ref]),%%mm1\n\t"
    "movq (%[src],%[ystride]),%%mm2\n\t"
    "movq (%[ref],%[ystride]),%%mm3\n\t"
    "lea (%[ystride],%[ystride],2),%[ystride3]\n\t"
    "movq (%[src],%[ystride],2),%%mm4\n\t"
    "movq (%[ref],%[ystride],2),%%mm5\n\t"
    "movq (%[src],%[ystride3]),%%mm6\n\t"
    "movq (%[ref],%[ystride3]),%%mm7\n\t"
    /*Compute their SADs and add them in %%mm0*/
    "psadbw %%mm1,%%mm0\n\t"
    "psadbw %%mm3,%%mm2\n\t"
    "lea (%[src],%[ystride],4),%[src]\n\t"
    "paddw %%mm2,%%mm0\n\t"
    "lea (%[ref],%[ystride],4),%[ref]\n\t"
    /*Load the next 3 rows as registers become available.*/
    "movq (%[src]),%%mm2\n\t"
    "movq (%[ref]),%%mm3\n\t"
    "psadbw %%mm5,%%mm4\n\t"
    "psadbw %%mm7,%%mm6\n\t"
    "paddw %%mm4,%%mm0\n\t"
    "movq (%[ref],%[ystride]),%%mm5\n\t"
    "movq (%[src],%[ystride]),%%mm4\n\t"
    "paddw %%mm6,%%mm0\n\t"
    "movq (%[ref],%[ystride],2),%%mm7\n\t"
    "movq (%[src],%[ystride],2),%%mm6\n\t"
    /*Start adding their SADs to %%mm0*/
    "psadbw %%mm3,%%mm2\n\t"
    "psadbw %%mm5,%%mm4\n\t"
    "paddw %%mm2,%%mm0\n\t"
    "psadbw %%mm7,%%mm6\n\t"
    /*Load last row as registers become available.*/
    "movq (%[src],%[ystride3]),%%mm2\n\t"
    "movq (%[ref],%[ystride3]),%%mm3\n\t"
    /*And finish adding up their SADs.*/
    "paddw %%mm4,%%mm0\n\t"
    "psadbw %%mm3,%%mm2\n\t"
    "paddw %%mm6,%%mm0\n\t"
    "paddw %%mm2,%%mm0\n\t"
    "movd %%mm0,%[ret]\n\t"
    :[ret]"=a"(ret),[src]"+r"(_src),[ref]"+r"(_ref),[ystride3]"=&r"(ystride3)
    :[ystride]"r"((ptrdiff_t)_ystride)
  );
  return (unsigned)ret;
}

unsigned oc_enc_frag_sad_thresh_mmxext(const unsigned char *_src,
 const unsigned char *_ref,int _ystride,unsigned _thresh){
  /*Early termination is for suckers.*/
  return oc_enc_frag_sad_mmxext(_src,_ref,_ystride);
}

/*Assumes the first two rows of %[ref1] and %[ref2] are in %%mm0...%%mm3, the
   first two rows of %[src] are in %%mm4,%%mm5, and {1}x8 is in %%mm7.
  We pre-load the next two rows of data as registers become available.*/
#define OC_SAD2_LOOP \
 "#OC_SAD2_LOOP\n\t" \
 /*We want to compute (%%mm0+%%mm1>>1) on unsigned bytes without overflow, but \
    pavgb computes (%%mm0+%%mm1+1>>1). \
   The latter is exactly 1 too large when the low bit of two corresponding \
    bytes is only set in one of them. \
   Therefore we pxor the operands, pand to mask out the low bits, and psubb to \
    correct the output of pavgb. \
   TODO: This should be rewritten to compute ~pavgb(~a,~b) instead, which \
    schedules better; currently, however, this function is unused.*/ \
 "movq %%mm0,%%mm6\n\t" \
 "lea (%[ref1],%[ystride],2),%[ref1]\n\t" \
 "pxor %%mm1,%%mm0\n\t" \
 "pavgb %%mm1,%%mm6\n\t" \
 "lea (%[ref2],%[ystride],2),%[ref2]\n\t" \
 "movq %%mm2,%%mm1\n\t" \
 "pand %%mm7,%%mm0\n\t" \
 "pavgb %%mm3,%%mm2\n\t" \
 "pxor %%mm3,%%mm1\n\t" \
 "movq (%[ref2],%[ystride]),%%mm3\n\t" \
 "psubb %%mm0,%%mm6\n\t" \
 "movq (%[ref1]),%%mm0\n\t" \
 "pand %%mm7,%%mm1\n\t" \
 "psadbw %%mm6,%%mm4\n\t" \
 "movd %[ret],%%mm6\n\t" \
 "psubb %%mm1,%%mm2\n\t" \
 "movq (%[ref2]),%%mm1\n\t" \
 "lea (%[src],%[ystride],2),%[src]\n\t" \
 "psadbw %%mm2,%%mm5\n\t" \
 "movq (%[ref1],%[ystride]),%%mm2\n\t" \
 "paddw %%mm4,%%mm5\n\t" \
 "movq (%[src]),%%mm4\n\t" \
 "paddw %%mm5,%%mm6\n\t" \
 "movq (%[src],%[ystride]),%%mm5\n\t" \
 "movd %%mm6,%[ret]\n\t" \

/*Same as above, but does not pre-load the next two rows.*/
#define OC_SAD2_TAIL \
 "#OC_SAD2_TAIL\n\t" \
 "movq %%mm0,%%mm6\n\t" \
 "pavgb %%mm1,%%mm0\n\t" \
 "pxor %%mm1,%%mm6\n\t" \
 "movq %%mm2,%%mm1\n\t" \
 "pand %%mm7,%%mm6\n\t" \
 "pavgb %%mm3,%%mm2\n\t" \
 "pxor %%mm3,%%mm1\n\t" \
 "psubb %%mm6,%%mm0\n\t" \
 "pand %%mm7,%%mm1\n\t" \
 "psadbw %%mm0,%%mm4\n\t" \
 "psubb %%mm1,%%mm2\n\t" \
 "movd %[ret],%%mm6\n\t" \
 "psadbw %%mm2,%%mm5\n\t" \
 "paddw %%mm4,%%mm5\n\t" \
 "paddw %%mm5,%%mm6\n\t" \
 "movd %%mm6,%[ret]\n\t" \

unsigned oc_enc_frag_sad2_thresh_mmxext(const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride,
 unsigned _thresh){
  ptrdiff_t ret;
  __asm__ __volatile__(
    "movq (%[ref1]),%%mm0\n\t"
    "movq (%[ref2]),%%mm1\n\t"
    "movq (%[ref1],%[ystride]),%%mm2\n\t"
    "movq (%[ref2],%[ystride]),%%mm3\n\t"
    "xor %[ret],%[ret]\n\t"
    "movq (%[src]),%%mm4\n\t"
    "pxor %%mm7,%%mm7\n\t"
    "pcmpeqb %%mm6,%%mm6\n\t"
    "movq (%[src],%[ystride]),%%mm5\n\t"
    "psubb %%mm6,%%mm7\n\t"
    OC_SAD2_LOOP
    OC_SAD2_LOOP
    OC_SAD2_LOOP
    OC_SAD2_TAIL
    :[ret]"=&a"(ret),[src]"+r"(_src),[ref1]"+r"(_ref1),[ref2]"+r"(_ref2)
    :[ystride]"r"((ptrdiff_t)_ystride)
  );
  return (unsigned)ret;
}

/*Load an 8x4 array of pixel values from %[src] and %[ref] and compute their
   16-bit difference in %%mm0...%%mm7.*/
#define OC_LOAD_SUB_8x4(_off) \
 "#OC_LOAD_SUB_8x4\n\t" \
 "movd "#_off"(%[src]),%%mm0\n\t" \
 "movd "#_off"(%[ref]),%%mm4\n\t" \
 "movd "#_off"(%[src],%[src_ystride]),%%mm1\n\t" \
 "lea (%[src],%[src_ystride],2),%[src]\n\t" \
 "movd "#_off"(%[ref],%[ref_ystride]),%%mm5\n\t" \
 "lea (%[ref],%[ref_ystride],2),%[ref]\n\t" \
 "movd "#_off"(%[src]),%%mm2\n\t" \
 "movd "#_off"(%[ref]),%%mm7\n\t" \
 "movd "#_off"(%[src],%[src_ystride]),%%mm3\n\t" \
 "movd "#_off"(%[ref],%[ref_ystride]),%%mm6\n\t" \
 "punpcklbw %%mm4,%%mm0\n\t" \
 "lea (%[src],%[src_ystride],2),%[src]\n\t" \
 "punpcklbw %%mm4,%%mm4\n\t" \
 "lea (%[ref],%[ref_ystride],2),%[ref]\n\t" \
 "psubw %%mm4,%%mm0\n\t" \
 "movd "#_off"(%[src]),%%mm4\n\t" \
 "movq %%mm0,"OC_MEM_OFFS(_off*2,buf)"\n\t" \
 "movd "#_off"(%[ref]),%%mm0\n\t" \
 "punpcklbw %%mm5,%%mm1\n\t" \
 "punpcklbw %%mm5,%%mm5\n\t" \
 "psubw %%mm5,%%mm1\n\t" \
 "movd "#_off"(%[src],%[src_ystride]),%%mm5\n\t" \
 "punpcklbw %%mm7,%%mm2\n\t" \
 "punpcklbw %%mm7,%%mm7\n\t" \
 "psubw %%mm7,%%mm2\n\t" \
 "movd "#_off"(%[ref],%[ref_ystride]),%%mm7\n\t" \
 "punpcklbw %%mm6,%%mm3\n\t" \
 "lea (%[src],%[src_ystride],2),%[src]\n\t" \
 "punpcklbw %%mm6,%%mm6\n\t" \
 "psubw %%mm6,%%mm3\n\t" \
 "movd "#_off"(%[src]),%%mm6\n\t" \
 "punpcklbw %%mm0,%%mm4\n\t" \
 "lea (%[ref],%[ref_ystride],2),%[ref]\n\t" \
 "punpcklbw %%mm0,%%mm0\n\t" \
 "lea (%[src],%[src_ystride],2),%[src]\n\t" \
 "psubw %%mm0,%%mm4\n\t" \
 "movd "#_off"(%[ref]),%%mm0\n\t" \
 "punpcklbw %%mm7,%%mm5\n\t" \
 "neg %[src_ystride]\n\t" \
 "punpcklbw %%mm7,%%mm7\n\t" \
 "psubw %%mm7,%%mm5\n\t" \
 "movd "#_off"(%[src],%[src_ystride]),%%mm7\n\t" \
 "punpcklbw %%mm0,%%mm6\n\t" \
 "lea (%[ref],%[ref_ystride],2),%[ref]\n\t" \
 "punpcklbw %%mm0,%%mm0\n\t" \
 "neg %[ref_ystride]\n\t" \
 "psubw %%mm0,%%mm6\n\t" \
 "movd "#_off"(%[ref],%[ref_ystride]),%%mm0\n\t" \
 "lea (%[src],%[src_ystride],8),%[src]\n\t" \
 "punpcklbw %%mm0,%%mm7\n\t" \
 "neg %[src_ystride]\n\t" \
 "punpcklbw %%mm0,%%mm0\n\t" \
 "lea (%[ref],%[ref_ystride],8),%[ref]\n\t" \
 "psubw %%mm0,%%mm7\n\t" \
 "neg %[ref_ystride]\n\t" \
 "movq "OC_MEM_OFFS(_off*2,buf)",%%mm0\n\t" \

/*Load an 8x4 array of pixel values from %[src] into %%mm0...%%mm7.*/
#define OC_LOAD_8x4(_off) \
 "#OC_LOAD_8x4\n\t" \
 "movd "#_off"(%[src]),%%mm0\n\t" \
 "movd "#_off"(%[src],%[ystride]),%%mm1\n\t" \
 "movd "#_off"(%[src],%[ystride],2),%%mm2\n\t" \
 "pxor %%mm7,%%mm7\n\t" \
 "movd "#_off"(%[src],%[ystride3]),%%mm3\n\t" \
 "punpcklbw %%mm7,%%mm0\n\t" \
 "movd "#_off"(%[src4]),%%mm4\n\t" \
 "punpcklbw %%mm7,%%mm1\n\t" \
 "movd "#_off"(%[src4],%[ystride]),%%mm5\n\t" \
 "punpcklbw %%mm7,%%mm2\n\t" \
 "movd "#_off"(%[src4],%[ystride],2),%%mm6\n\t" \
 "punpcklbw %%mm7,%%mm3\n\t" \
 "movd "#_off"(%[src4],%[ystride3]),%%mm7\n\t" \
 "punpcklbw %%mm4,%%mm4\n\t" \
 "punpcklbw %%mm5,%%mm5\n\t" \
 "psrlw $8,%%mm4\n\t" \
 "psrlw $8,%%mm5\n\t" \
 "punpcklbw %%mm6,%%mm6\n\t" \
 "punpcklbw %%mm7,%%mm7\n\t" \
 "psrlw $8,%%mm6\n\t" \
 "psrlw $8,%%mm7\n\t" \

/*Performs the first two stages of an 8-point 1-D Hadamard transform.
  The transform is performed in place, except that outputs 0-3 are swapped with
   outputs 4-7.
  Outputs 2, 3, 6, and 7 from the second stage are negated (which allows us to
   perform this stage in place with no temporary registers).*/
#define OC_HADAMARD_AB_8x4 \
 "#OC_HADAMARD_AB_8x4\n\t" \
 /*Stage A: \
   Outputs 0-3 are swapped with 4-7 here.*/ \
 "paddw %%mm1,%%mm5\n\t" \
 "paddw %%mm2,%%mm6\n\t" \
 "paddw %%mm1,%%mm1\n\t" \
 "paddw %%mm2,%%mm2\n\t" \
 "psubw %%mm5,%%mm1\n\t" \
 "psubw %%mm6,%%mm2\n\t" \
 "paddw %%mm3,%%mm7\n\t" \
 "paddw %%mm0,%%mm4\n\t" \
 "paddw %%mm3,%%mm3\n\t" \
 "paddw %%mm0,%%mm0\n\t" \
 "psubw %%mm7,%%mm3\n\t" \
 "psubw %%mm4,%%mm0\n\t" \
 /*Stage B:*/ \
 "paddw %%mm2,%%mm0\n\t" \
 "paddw %%mm3,%%mm1\n\t" \
 "paddw %%mm6,%%mm4\n\t" \
 "paddw %%mm7,%%mm5\n\t" \
 "paddw %%mm2,%%mm2\n\t" \
 "paddw %%mm3,%%mm3\n\t" \
 "paddw %%mm6,%%mm6\n\t" \
 "paddw %%mm7,%%mm7\n\t" \
 "psubw %%mm0,%%mm2\n\t" \
 "psubw %%mm1,%%mm3\n\t" \
 "psubw %%mm4,%%mm6\n\t" \
 "psubw %%mm5,%%mm7\n\t" \

/*Performs the last stage of an 8-point 1-D Hadamard transform in place.
  Outputs 1, 3, 5, and 7 are negated (which allows us to perform this stage in
   place with no temporary registers).*/
#define OC_HADAMARD_C_8x4 \
 "#OC_HADAMARD_C_8x4\n\t" \
 /*Stage C:*/ \
 "paddw %%mm1,%%mm0\n\t" \
 "paddw %%mm3,%%mm2\n\t" \
 "paddw %%mm5,%%mm4\n\t" \
 "paddw %%mm7,%%mm6\n\t" \
 "paddw %%mm1,%%mm1\n\t" \
 "paddw %%mm3,%%mm3\n\t" \
 "paddw %%mm5,%%mm5\n\t" \
 "paddw %%mm7,%%mm7\n\t" \
 "psubw %%mm0,%%mm1\n\t" \
 "psubw %%mm2,%%mm3\n\t" \
 "psubw %%mm4,%%mm5\n\t" \
 "psubw %%mm6,%%mm7\n\t" \

/*Performs an 8-point 1-D Hadamard transform.
  The transform is performed in place, except that outputs 0-3 are swapped with
   outputs 4-7.
  Outputs 1, 2, 5 and 6 are negated (which allows us to perform the transform
   in place with no temporary registers).*/
#define OC_HADAMARD_8x4 \
 OC_HADAMARD_AB_8x4 \
 OC_HADAMARD_C_8x4 \

/*Performs the first part of the final stage of the Hadamard transform and
   summing of absolute values.
  At the end of this part, %%mm1 will contain the DC coefficient of the
   transform.*/
#define OC_HADAMARD_C_ABS_ACCUM_A_8x4(_r6,_r7) \
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
 "#OC_HADAMARD_C_ABS_ACCUM_A_8x4\n\t" \
 "movq %%mm7,"OC_MEM_OFFS(_r7,buf)"\n\t" \
 "movq %%mm6,"OC_MEM_OFFS(_r6,buf)"\n\t" \
 /*mm7={0x7FFF}x4 \
   mm0=max(abs(mm0),abs(mm1))-0x7FFF*/ \
 "pcmpeqb %%mm7,%%mm7\n\t" \
 "movq %%mm0,%%mm6\n\t" \
 "psrlw $1,%%mm7\n\t" \
 "paddw %%mm1,%%mm6\n\t" \
 "pmaxsw %%mm1,%%mm0\n\t" \
 "paddsw %%mm7,%%mm6\n\t" \
 "psubw %%mm6,%%mm0\n\t" \
 /*mm2=max(abs(mm2),abs(mm3))-0x7FFF \
   mm4=max(abs(mm4),abs(mm5))-0x7FFF*/ \
 "movq %%mm2,%%mm6\n\t" \
 "movq %%mm4,%%mm1\n\t" \
 "pmaxsw %%mm3,%%mm2\n\t" \
 "pmaxsw %%mm5,%%mm4\n\t" \
 "paddw %%mm3,%%mm6\n\t" \
 "paddw %%mm5,%%mm1\n\t" \
 "movq "OC_MEM_OFFS(_r7,buf)",%%mm3\n\t" \

/*Performs the second part of the final stage of the Hadamard transform and
   summing of absolute values.*/
#define OC_HADAMARD_C_ABS_ACCUM_B_8x4(_r6,_r7) \
 "#OC_HADAMARD_C_ABS_ACCUM_B_8x4\n\t" \
 "paddsw %%mm7,%%mm6\n\t" \
 "movq "OC_MEM_OFFS(_r6,buf)",%%mm5\n\t" \
 "paddsw %%mm7,%%mm1\n\t" \
 "psubw %%mm6,%%mm2\n\t" \
 "psubw %%mm1,%%mm4\n\t" \
 /*mm7={1}x4 (needed for the horizontal add that follows) \
   mm0+=mm2+mm4+max(abs(mm3),abs(mm5))-0x7FFF*/ \
 "movq %%mm3,%%mm6\n\t" \
 "pmaxsw %%mm5,%%mm3\n\t" \
 "paddw %%mm2,%%mm0\n\t" \
 "paddw %%mm5,%%mm6\n\t" \
 "paddw %%mm4,%%mm0\n\t" \
 "paddsw %%mm7,%%mm6\n\t" \
 "paddw %%mm3,%%mm0\n\t" \
 "psrlw $14,%%mm7\n\t" \
 "psubw %%mm6,%%mm0\n\t" \

/*Performs the last stage of an 8-point 1-D Hadamard transform, takes the
   absolute value of each component, and accumulates everything into mm0.
  This is the only portion of SATD which requires MMXEXT (we could use plain
   MMX, but it takes 4 instructions and an extra register to work around the
   lack of a pmaxsw, which is a pretty serious penalty).*/
#define OC_HADAMARD_C_ABS_ACCUM_8x4(_r6,_r7) \
 OC_HADAMARD_C_ABS_ACCUM_A_8x4(_r6,_r7) \
 OC_HADAMARD_C_ABS_ACCUM_B_8x4(_r6,_r7) \

/*Performs an 8-point 1-D Hadamard transform, takes the absolute value of each
   component, and accumulates everything into mm0.
  Note that mm0 will have an extra 4 added to each column, and that after
   removing this value, the remainder will be half the conventional value.*/
#define OC_HADAMARD_ABS_ACCUM_8x4(_r6,_r7) \
 OC_HADAMARD_AB_8x4 \
 OC_HADAMARD_C_ABS_ACCUM_8x4(_r6,_r7)

/*Performs two 4x4 transposes (mostly) in place.
  On input, {mm0,mm1,mm2,mm3} contains rows {e,f,g,h}, and {mm4,mm5,mm6,mm7}
   contains rows {a,b,c,d}.
  On output, {0x40,0x50,0x60,0x70}+_off(%[buf]) contains {e,f,g,h}^T, and
   {mm4,mm5,mm6,mm7} contains the transposed rows {a,b,c,d}^T.*/
#define OC_TRANSPOSE_4x4x2(_off) \
 "#OC_TRANSPOSE_4x4x2\n\t" \
 /*First 4x4 transpose:*/ \
 "movq %%mm5,"OC_MEM_OFFS(0x10+(_off),buf)"\n\t" \
 /*mm0 = e3 e2 e1 e0 \
   mm1 = f3 f2 f1 f0 \
   mm2 = g3 g2 g1 g0 \
   mm3 = h3 h2 h1 h0*/ \
 "movq %%mm2,%%mm5\n\t" \
 "punpcklwd %%mm3,%%mm2\n\t" \
 "punpckhwd %%mm3,%%mm5\n\t" \
 "movq %%mm0,%%mm3\n\t" \
 "punpcklwd %%mm1,%%mm0\n\t" \
 "punpckhwd %%mm1,%%mm3\n\t" \
 /*mm0 = f1 e1 f0 e0 \
   mm3 = f3 e3 f2 e2 \
   mm2 = h1 g1 h0 g0 \
   mm5 = h3 g3 h2 g2*/ \
 "movq %%mm0,%%mm1\n\t" \
 "punpckldq %%mm2,%%mm0\n\t" \
 "punpckhdq %%mm2,%%mm1\n\t" \
 "movq %%mm3,%%mm2\n\t" \
 "punpckhdq %%mm5,%%mm3\n\t" \
 "movq %%mm0,"OC_MEM_OFFS(0x40+(_off),buf)"\n\t" \
 "punpckldq %%mm5,%%mm2\n\t" \
 /*mm0 = h0 g0 f0 e0 \
   mm1 = h1 g1 f1 e1 \
   mm2 = h2 g2 f2 e2 \
   mm3 = h3 g3 f3 e3*/ \
 "movq "OC_MEM_OFFS(0x10+(_off),buf)",%%mm5\n\t" \
 /*Second 4x4 transpose:*/ \
 /*mm4 = a3 a2 a1 a0 \
   mm5 = b3 b2 b1 b0 \
   mm6 = c3 c2 c1 c0 \
   mm7 = d3 d2 d1 d0*/ \
 "movq %%mm6,%%mm0\n\t" \
 "punpcklwd %%mm7,%%mm6\n\t" \
 "movq %%mm1,"OC_MEM_OFFS(0x50+(_off),buf)"\n\t" \
 "punpckhwd %%mm7,%%mm0\n\t" \
 "movq %%mm4,%%mm7\n\t" \
 "punpcklwd %%mm5,%%mm4\n\t" \
 "movq %%mm2,"OC_MEM_OFFS(0x60+(_off),buf)"\n\t" \
 "punpckhwd %%mm5,%%mm7\n\t" \
 /*mm4 = b1 a1 b0 a0 \
   mm7 = b3 a3 b2 a2 \
   mm6 = d1 c1 d0 c0 \
   mm0 = d3 c3 d2 c2*/ \
 "movq %%mm4,%%mm5\n\t" \
 "punpckldq %%mm6,%%mm4\n\t" \
 "movq %%mm3,"OC_MEM_OFFS(0x70+(_off),buf)"\n\t" \
 "punpckhdq %%mm6,%%mm5\n\t" \
 "movq %%mm7,%%mm6\n\t" \
 "punpckhdq %%mm0,%%mm7\n\t" \
 "punpckldq %%mm0,%%mm6\n\t" \
 /*mm4 = d0 c0 b0 a0 \
   mm5 = d1 c1 b1 a1 \
   mm6 = d2 c2 b2 a2 \
   mm7 = d3 c3 b3 a3*/ \

static unsigned oc_int_frag_satd_mmxext(int *_dc,
 const unsigned char *_src,int _src_ystride,
 const unsigned char *_ref,int _ref_ystride){
  OC_ALIGN8(ogg_int16_t buf[64]);
  unsigned ret;
  unsigned ret2;
  int      dc;
  __asm__ __volatile__(
    OC_LOAD_SUB_8x4(0x00)
    OC_HADAMARD_8x4
    OC_TRANSPOSE_4x4x2(0x00)
    /*Finish swapping out this 8x4 block to make room for the next one.
      mm0...mm3 have been swapped out already.*/
    "movq %%mm4,"OC_MEM_OFFS(0x00,buf)"\n\t"
    "movq %%mm5,"OC_MEM_OFFS(0x10,buf)"\n\t"
    "movq %%mm6,"OC_MEM_OFFS(0x20,buf)"\n\t"
    "movq %%mm7,"OC_MEM_OFFS(0x30,buf)"\n\t"
    OC_LOAD_SUB_8x4(0x04)
    OC_HADAMARD_8x4
    OC_TRANSPOSE_4x4x2(0x08)
    /*Here the first 4x4 block of output from the last transpose is the second
       4x4 block of input for the next transform.
      We have cleverly arranged that it already be in the appropriate place, so
       we only have to do half the loads.*/
    "movq "OC_MEM_OFFS(0x10,buf)",%%mm1\n\t"
    "movq "OC_MEM_OFFS(0x20,buf)",%%mm2\n\t"
    "movq "OC_MEM_OFFS(0x30,buf)",%%mm3\n\t"
    "movq "OC_MEM_OFFS(0x00,buf)",%%mm0\n\t"
    /*We split out the stages here so we can save the DC coefficient in the
       middle.*/
    OC_HADAMARD_AB_8x4
    OC_HADAMARD_C_ABS_ACCUM_A_8x4(0x28,0x38)
    "movd %%mm1,%[dc]\n\t"
    OC_HADAMARD_C_ABS_ACCUM_B_8x4(0x28,0x38)
    /*Up to this point, everything fit in 16 bits (8 input + 1 for the
       difference + 2*3 for the two 8-point 1-D Hadamards - 1 for the abs - 1
       for the factor of two we dropped + 3 for the vertical accumulation).
      Now we finally have to promote things to dwords.
      We break this part out of OC_HADAMARD_ABS_ACCUM_8x4 to hide the long
       latency of pmaddwd by starting the next series of loads now.*/
    "pmaddwd %%mm7,%%mm0\n\t"
    "movq "OC_MEM_OFFS(0x50,buf)",%%mm1\n\t"
    "movq "OC_MEM_OFFS(0x58,buf)",%%mm5\n\t"
    "movq %%mm0,%%mm4\n\t"
    "movq "OC_MEM_OFFS(0x60,buf)",%%mm2\n\t"
    "punpckhdq %%mm0,%%mm0\n\t"
    "movq "OC_MEM_OFFS(0x68,buf)",%%mm6\n\t"
    "paddd %%mm0,%%mm4\n\t"
    "movq "OC_MEM_OFFS(0x70,buf)",%%mm3\n\t"
    "movd %%mm4,%[ret2]\n\t"
    "movq "OC_MEM_OFFS(0x78,buf)",%%mm7\n\t"
    "movq "OC_MEM_OFFS(0x40,buf)",%%mm0\n\t"
    "movq "OC_MEM_OFFS(0x48,buf)",%%mm4\n\t"
    OC_HADAMARD_ABS_ACCUM_8x4(0x68,0x78)
    "pmaddwd %%mm7,%%mm0\n\t"
    /*Subtract abs(dc) from 2*ret2.*/
    "movsx %w[dc],%[dc]\n\t"
    "cdq\n\t"
    "lea (%[ret],%[ret2],2),%[ret2]\n\t"
    "movq %%mm0,%%mm4\n\t"
    "punpckhdq %%mm0,%%mm0\n\t"
    "xor %[dc],%[ret]\n\t"
    "paddd %%mm0,%%mm4\n\t"
    /*The sums produced by OC_HADAMARD_ABS_ACCUM_8x4 each have an extra 4
       added to them, a factor of two removed, and the DC value included;
       correct the final sum here.*/
    "sub %[ret],%[ret2]\n\t"
    "movd %%mm4,%[ret]\n\t"
    "lea -64(%[ret2],%[ret],2),%[ret]\n\t"
    /*Although it looks like we're using 8 registers here, gcc can alias %[ret]
       and %[ret2] with some of the inputs, since for once we don't write to
       them until after we're done using everything but %[buf].*/
    /*Note that _src_ystride and _ref_ystride must be given non-overlapping
       constraints, otherwise if gcc can prove they're equal it will allocate
       them to the same register (which is bad); _src and _ref face a similar
       problem, though those are never actually the same.*/
    :[ret]"=d"(ret),[ret2]"=r"(ret2),[dc]"=a"(dc),
     [buf]"=m"(OC_ARRAY_OPERAND(ogg_int16_t,buf,64))
    :[src]"r"(_src),[src_ystride]"c"((ptrdiff_t)_src_ystride),
     [ref]"r"(_ref),[ref_ystride]"d"((ptrdiff_t)_ref_ystride)
    /*We have to use neg, so we actually clobber the condition codes for once
       (not to mention cmp, sub, and add).*/
    :"cc"
  );
  *_dc=dc;
  return ret;
}

unsigned oc_enc_frag_satd_mmxext(int *_dc,const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  return oc_int_frag_satd_mmxext(_dc,_src,_ystride,_ref,_ystride);
}

/*Our internal implementation of frag_copy2 takes an extra stride parameter so
   we can share code with oc_enc_frag_satd2_mmxext().*/
void oc_int_frag_copy2_mmxext(unsigned char *_dst,int _dst_ystride,
 const unsigned char *_src1,const unsigned char *_src2,int _src_ystride){
  __asm__ __volatile__(
    /*Load the first 3 rows.*/
    "movq (%[src1]),%%mm0\n\t"
    "movq (%[src2]),%%mm1\n\t"
    "movq (%[src1],%[src_ystride]),%%mm2\n\t"
    "lea (%[src1],%[src_ystride],2),%[src1]\n\t"
    "movq (%[src2],%[src_ystride]),%%mm3\n\t"
    "lea (%[src2],%[src_ystride],2),%[src2]\n\t"
    "pxor %%mm7,%%mm7\n\t"
    "movq (%[src1]),%%mm4\n\t"
    "pcmpeqb %%mm6,%%mm6\n\t"
    "movq (%[src2]),%%mm5\n\t"
    /*mm7={1}x8.*/
    "psubb %%mm6,%%mm7\n\t"
    /*Start averaging %%mm0 and %%mm1 into %%mm6.*/
    "movq %%mm0,%%mm6\n\t"
    "pxor %%mm1,%%mm0\n\t"
    "pavgb %%mm1,%%mm6\n\t"
    /*%%mm1 is free, start averaging %%mm3 into %%mm2 using %%mm1.*/
    "movq %%mm2,%%mm1\n\t"
    "pand %%mm7,%%mm0\n\t"
    "pavgb %%mm3,%%mm2\n\t"
    "pxor %%mm3,%%mm1\n\t"
    /*%%mm3 is free.*/
    "psubb %%mm0,%%mm6\n\t"
    /*%%mm0 is free, start loading the next row.*/
    "movq (%[src1],%[src_ystride]),%%mm0\n\t"
    /*Start averaging %%mm5 and %%mm4 using %%mm3.*/
    "movq %%mm4,%%mm3\n\t"
    /*%%mm6 (row 0) is done; write it out.*/
    "movq %%mm6,(%[dst])\n\t"
    "pand %%mm7,%%mm1\n\t"
    "pavgb %%mm5,%%mm4\n\t"
    "psubb %%mm1,%%mm2\n\t"
    /*%%mm1 is free, continue loading the next row.*/
    "movq (%[src2],%[src_ystride]),%%mm1\n\t"
    "pxor %%mm5,%%mm3\n\t"
    "lea (%[src1],%[src_ystride],2),%[src1]\n\t"
    /*%%mm2 (row 1) is done; write it out.*/
    "movq %%mm2,(%[dst],%[dst_ystride])\n\t"
    "pand %%mm7,%%mm3\n\t"
    /*Start loading the next row.*/
    "movq (%[src1]),%%mm2\n\t"
    "lea (%[dst],%[dst_ystride],2),%[dst]\n\t"
    "psubb %%mm3,%%mm4\n\t"
    "lea (%[src2],%[src_ystride],2),%[src2]\n\t"
    /*%%mm4 (row 2) is done; write it out.*/
    "movq %%mm4,(%[dst])\n\t"
    /*Continue loading the next row.*/
    "movq (%[src2]),%%mm3\n\t"
    /*Start averaging %%mm0 and %%mm1 into %%mm6.*/
    "movq %%mm0,%%mm6\n\t"
    "pxor %%mm1,%%mm0\n\t"
    /*Start loading the next row.*/
    "movq (%[src1],%[src_ystride]),%%mm4\n\t"
    "pavgb %%mm1,%%mm6\n\t"
    /*%%mm1 is free; start averaging %%mm3 into %%mm2 using %%mm1.*/
    "movq %%mm2,%%mm1\n\t"
    "pand %%mm7,%%mm0\n\t"
    /*Continue loading the next row.*/
    "movq (%[src2],%[src_ystride]),%%mm5\n\t"
    "pavgb %%mm3,%%mm2\n\t"
    "lea (%[src1],%[src_ystride],2),%[src1]\n\t"
    "pxor %%mm3,%%mm1\n\t"
    /*%%mm3 is free.*/
    "psubb %%mm0,%%mm6\n\t"
    /*%%mm0 is free, start loading the next row.*/
    "movq (%[src1]),%%mm0\n\t"
    /*Start averaging %%mm5 into %%mm4 using %%mm3.*/
    "movq %%mm4,%%mm3\n\t"
    /*%%mm6 (row 3) is done; write it out.*/
    "movq %%mm6,(%[dst],%[dst_ystride])\n\t"
    "pand %%mm7,%%mm1\n\t"
    "lea (%[src2],%[src_ystride],2),%[src2]\n\t"
    "pavgb %%mm5,%%mm4\n\t"
    "lea (%[dst],%[dst_ystride],2),%[dst]\n\t"
    "psubb %%mm1,%%mm2\n\t"
    /*%%mm1 is free; continue loading the next row.*/
    "movq (%[src2]),%%mm1\n\t"
    "pxor %%mm5,%%mm3\n\t"
    /*%%mm2 (row 4) is done; write it out.*/
    "movq %%mm2,(%[dst])\n\t"
    "pand %%mm7,%%mm3\n\t"
    /*Start loading the next row.*/
    "movq (%[src1],%[src_ystride]),%%mm2\n\t"
    "psubb %%mm3,%%mm4\n\t"
    /*Start averaging %%mm0 and %%mm1 into %%mm6.*/
    "movq %%mm0,%%mm6\n\t"
    /*Continue loading the next row.*/
    "movq (%[src2],%[src_ystride]),%%mm3\n\t"
    /*%%mm4 (row 5) is done; write it out.*/
    "movq %%mm4,(%[dst],%[dst_ystride])\n\t"
    "pxor %%mm1,%%mm0\n\t"
    "pavgb %%mm1,%%mm6\n\t"
    /*%%mm4 is free; start averaging %%mm3 into %%mm2 using %%mm4.*/
    "movq %%mm2,%%mm4\n\t"
    "pand %%mm7,%%mm0\n\t"
    "pavgb %%mm3,%%mm2\n\t"
    "pxor %%mm3,%%mm4\n\t"
    "lea (%[dst],%[dst_ystride],2),%[dst]\n\t"
    "psubb %%mm0,%%mm6\n\t"
    "pand %%mm7,%%mm4\n\t"
    /*%%mm6 (row 6) is done, write it out.*/
    "movq %%mm6,(%[dst])\n\t"
    "psubb %%mm4,%%mm2\n\t"
    /*%%mm2 (row 7) is done, write it out.*/
    "movq %%mm2,(%[dst],%[dst_ystride])\n\t"
    :[dst]"+r"(_dst),[src1]"+r"(_src1),[src2]"+r"(_src2)
    :[dst_ystride]"r"((ptrdiff_t)_dst_ystride),
     [src_ystride]"r"((ptrdiff_t)_src_ystride)
    :"memory"
  );
}

unsigned oc_enc_frag_satd2_mmxext(int *_dc,const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride){
  OC_ALIGN8(unsigned char ref[64]);
  oc_int_frag_copy2_mmxext(ref,8,_ref1,_ref2,_ystride);
  return oc_int_frag_satd_mmxext(_dc,_src,_ystride,ref,8);
}

unsigned oc_enc_frag_intra_satd_mmxext(int *_dc,
 const unsigned char *_src,int _ystride){
  OC_ALIGN8(ogg_int16_t buf[64]);
  unsigned ret;
  unsigned ret2;
  int      dc;
  __asm__ __volatile__(
    OC_LOAD_8x4(0x00)
    OC_HADAMARD_8x4
    OC_TRANSPOSE_4x4x2(0x00)
    /*Finish swapping out this 8x4 block to make room for the next one.
      mm0...mm3 have been swapped out already.*/
    "movq %%mm4,"OC_MEM_OFFS(0x00,buf)"\n\t"
    "movq %%mm5,"OC_MEM_OFFS(0x10,buf)"\n\t"
    "movq %%mm6,"OC_MEM_OFFS(0x20,buf)"\n\t"
    "movq %%mm7,"OC_MEM_OFFS(0x30,buf)"\n\t"
    OC_LOAD_8x4(0x04)
    OC_HADAMARD_8x4
    OC_TRANSPOSE_4x4x2(0x08)
    /*Here the first 4x4 block of output from the last transpose is the second
       4x4 block of input for the next transform.
      We have cleverly arranged that it already be in the appropriate place, so
       we only have to do half the loads.*/
    "movq "OC_MEM_OFFS(0x10,buf)",%%mm1\n\t"
    "movq "OC_MEM_OFFS(0x20,buf)",%%mm2\n\t"
    "movq "OC_MEM_OFFS(0x30,buf)",%%mm3\n\t"
    "movq "OC_MEM_OFFS(0x00,buf)",%%mm0\n\t"
    /*We split out the stages here so we can save the DC coefficient in the
       middle.*/
    OC_HADAMARD_AB_8x4
    OC_HADAMARD_C_ABS_ACCUM_A_8x4(0x28,0x38)
    "movd %%mm1,%[dc]\n\t"
    OC_HADAMARD_C_ABS_ACCUM_B_8x4(0x28,0x38)
    /*Up to this point, everything fit in 16 bits (8 input + 1 for the
       difference + 2*3 for the two 8-point 1-D Hadamards - 1 for the abs - 1
       for the factor of two we dropped + 3 for the vertical accumulation).
      Now we finally have to promote things to dwords.
      We break this part out of OC_HADAMARD_ABS_ACCUM_8x4 to hide the long
       latency of pmaddwd by starting the next series of loads now.*/
    "pmaddwd %%mm7,%%mm0\n\t"
    "movq "OC_MEM_OFFS(0x50,buf)",%%mm1\n\t"
    "movq "OC_MEM_OFFS(0x58,buf)",%%mm5\n\t"
    "movq "OC_MEM_OFFS(0x60,buf)",%%mm2\n\t"
    "movq %%mm0,%%mm4\n\t"
    "movq "OC_MEM_OFFS(0x68,buf)",%%mm6\n\t"
    "punpckhdq %%mm0,%%mm0\n\t"
    "movq "OC_MEM_OFFS(0x70,buf)",%%mm3\n\t"
    "paddd %%mm0,%%mm4\n\t"
    "movq "OC_MEM_OFFS(0x78,buf)",%%mm7\n\t"
    "movd %%mm4,%[ret]\n\t"
    "movq "OC_MEM_OFFS(0x40,buf)",%%mm0\n\t"
    "movq "OC_MEM_OFFS(0x48,buf)",%%mm4\n\t"
    OC_HADAMARD_ABS_ACCUM_8x4(0x68,0x78)
    "pmaddwd %%mm7,%%mm0\n\t"
    /*We assume that the DC coefficient is always positive (which is true,
       because the input to the INTRA transform was not a difference).*/
    "movzx %w[dc],%[dc]\n\t"
    "add %[ret],%[ret]\n\t"
    "sub %[dc],%[ret]\n\t"
    "movq %%mm0,%%mm4\n\t"
    "punpckhdq %%mm0,%%mm0\n\t"
    "paddd %%mm0,%%mm4\n\t"
    "movd %%mm4,%[ret2]\n\t"
    "lea -64(%[ret],%[ret2],2),%[ret]\n\t"
    /*Although it looks like we're using 8 registers here, gcc can alias %[ret]
       and %[ret2] with some of the inputs, since for once we don't write to
       them until after we're done using everything but %[buf] (which is also
       listed as an output to ensure gcc _doesn't_ alias them against it).*/
    :[ret]"=a"(ret),[ret2]"=r"(ret2),[dc]"=r"(dc),
     [buf]"=m"(OC_ARRAY_OPERAND(ogg_int16_t,buf,64))
    :[src]"r"(_src),[src4]"r"(_src+4*_ystride),
     [ystride]"r"((ptrdiff_t)_ystride),[ystride3]"r"((ptrdiff_t)3*_ystride)
    /*We have to use sub, so we actually clobber the condition codes for once
       (not to mention add).*/
    :"cc"
  );
  *_dc=dc;
  return ret;
}

void oc_enc_frag_sub_mmx(ogg_int16_t _residue[64],
 const unsigned char *_src,const unsigned char *_ref,int _ystride){
  int i;
  __asm__ __volatile__("pxor %%mm7,%%mm7\n\t"::);
  for(i=4;i-->0;){
    __asm__ __volatile__(
      /*mm0=[src]*/
      "movq (%[src]),%%mm0\n\t"
      /*mm1=[ref]*/
      "movq (%[ref]),%%mm1\n\t"
      /*mm4=[src+ystride]*/
      "movq (%[src],%[ystride]),%%mm4\n\t"
      /*mm5=[ref+ystride]*/
      "movq (%[ref],%[ystride]),%%mm5\n\t"
      /*Compute [src]-[ref].*/
      "movq %%mm0,%%mm2\n\t"
      "punpcklbw %%mm7,%%mm0\n\t"
      "movq %%mm1,%%mm3\n\t"
      "punpckhbw %%mm7,%%mm2\n\t"
      "punpcklbw %%mm7,%%mm1\n\t"
      "punpckhbw %%mm7,%%mm3\n\t"
      "psubw %%mm1,%%mm0\n\t"
      "psubw %%mm3,%%mm2\n\t"
      /*Compute [src+ystride]-[ref+ystride].*/
      "movq %%mm4,%%mm1\n\t"
      "punpcklbw %%mm7,%%mm4\n\t"
      "movq %%mm5,%%mm3\n\t"
      "punpckhbw %%mm7,%%mm1\n\t"
      "lea (%[src],%[ystride],2),%[src]\n\t"
      "punpcklbw %%mm7,%%mm5\n\t"
      "lea (%[ref],%[ystride],2),%[ref]\n\t"
      "punpckhbw %%mm7,%%mm3\n\t"
      "psubw %%mm5,%%mm4\n\t"
      "psubw %%mm3,%%mm1\n\t"
      /*Write the answer out.*/
      "movq %%mm0,0x00(%[residue])\n\t"
      "movq %%mm2,0x08(%[residue])\n\t"
      "movq %%mm4,0x10(%[residue])\n\t"
      "movq %%mm1,0x18(%[residue])\n\t"
      "lea 0x20(%[residue]),%[residue]\n\t"
      :[residue]"+r"(_residue),[src]"+r"(_src),[ref]"+r"(_ref)
      :[ystride]"r"((ptrdiff_t)_ystride)
      :"memory"
    );
  }
}

void oc_enc_frag_sub_128_mmx(ogg_int16_t _residue[64],
 const unsigned char *_src,int _ystride){
  ptrdiff_t ystride3;
  __asm__ __volatile__(
    /*mm0=[src]*/
    "movq (%[src]),%%mm0\n\t"
    /*mm1=[src+ystride]*/
    "movq (%[src],%[ystride]),%%mm1\n\t"
    /*mm6={-1}x4*/
    "pcmpeqw %%mm6,%%mm6\n\t"
    /*mm2=[src+2*ystride]*/
    "movq (%[src],%[ystride],2),%%mm2\n\t"
    /*[ystride3]=3*[ystride]*/
    "lea (%[ystride],%[ystride],2),%[ystride3]\n\t"
    /*mm6={1}x4*/
    "psllw $15,%%mm6\n\t"
    /*mm3=[src+3*ystride]*/
    "movq (%[src],%[ystride3]),%%mm3\n\t"
    /*mm6={128}x4*/
    "psrlw $8,%%mm6\n\t"
    /*mm7=0*/
    "pxor %%mm7,%%mm7\n\t"
    /*[src]=[src]+4*[ystride]*/
    "lea (%[src],%[ystride],4),%[src]\n\t"
    /*Compute [src]-128 and [src+ystride]-128*/
    "movq %%mm0,%%mm4\n\t"
    "punpcklbw %%mm7,%%mm0\n\t"
    "movq %%mm1,%%mm5\n\t"
    "punpckhbw %%mm7,%%mm4\n\t"
    "psubw %%mm6,%%mm0\n\t"
    "punpcklbw %%mm7,%%mm1\n\t"
    "psubw %%mm6,%%mm4\n\t"
    "punpckhbw %%mm7,%%mm5\n\t"
    "psubw %%mm6,%%mm1\n\t"
    "psubw %%mm6,%%mm5\n\t"
    /*Write the answer out.*/
    "movq %%mm0,0x00(%[residue])\n\t"
    "movq %%mm4,0x08(%[residue])\n\t"
    "movq %%mm1,0x10(%[residue])\n\t"
    "movq %%mm5,0x18(%[residue])\n\t"
    /*mm0=[src+4*ystride]*/
    "movq (%[src]),%%mm0\n\t"
    /*mm1=[src+5*ystride]*/
    "movq (%[src],%[ystride]),%%mm1\n\t"
    /*Compute [src+2*ystride]-128 and [src+3*ystride]-128*/
    "movq %%mm2,%%mm4\n\t"
    "punpcklbw %%mm7,%%mm2\n\t"
    "movq %%mm3,%%mm5\n\t"
    "punpckhbw %%mm7,%%mm4\n\t"
    "psubw %%mm6,%%mm2\n\t"
    "punpcklbw %%mm7,%%mm3\n\t"
    "psubw %%mm6,%%mm4\n\t"
    "punpckhbw %%mm7,%%mm5\n\t"
    "psubw %%mm6,%%mm3\n\t"
    "psubw %%mm6,%%mm5\n\t"
    /*Write the answer out.*/
    "movq %%mm2,0x20(%[residue])\n\t"
    "movq %%mm4,0x28(%[residue])\n\t"
    "movq %%mm3,0x30(%[residue])\n\t"
    "movq %%mm5,0x38(%[residue])\n\t"
    /*mm2=[src+6*ystride]*/
    "movq (%[src],%[ystride],2),%%mm2\n\t"
    /*mm3=[src+7*ystride]*/
    "movq (%[src],%[ystride3]),%%mm3\n\t"
    /*Compute [src+4*ystride]-128 and [src+5*ystride]-128*/
    "movq %%mm0,%%mm4\n\t"
    "punpcklbw %%mm7,%%mm0\n\t"
    "movq %%mm1,%%mm5\n\t"
    "punpckhbw %%mm7,%%mm4\n\t"
    "psubw %%mm6,%%mm0\n\t"
    "punpcklbw %%mm7,%%mm1\n\t"
    "psubw %%mm6,%%mm4\n\t"
    "punpckhbw %%mm7,%%mm5\n\t"
    "psubw %%mm6,%%mm1\n\t"
    "psubw %%mm6,%%mm5\n\t"
    /*Write the answer out.*/
    "movq %%mm0,0x40(%[residue])\n\t"
    "movq %%mm4,0x48(%[residue])\n\t"
    "movq %%mm1,0x50(%[residue])\n\t"
    "movq %%mm5,0x58(%[residue])\n\t"
    /*Compute [src+6*ystride]-128 and [src+7*ystride]-128*/
    "movq %%mm2,%%mm4\n\t"
    "punpcklbw %%mm7,%%mm2\n\t"
    "movq %%mm3,%%mm5\n\t"
    "punpckhbw %%mm7,%%mm4\n\t"
    "psubw %%mm6,%%mm2\n\t"
    "punpcklbw %%mm7,%%mm3\n\t"
    "psubw %%mm6,%%mm4\n\t"
    "punpckhbw %%mm7,%%mm5\n\t"
    "psubw %%mm6,%%mm3\n\t"
    "psubw %%mm6,%%mm5\n\t"
    /*Write the answer out.*/
    "movq %%mm2,0x60(%[residue])\n\t"
    "movq %%mm4,0x68(%[residue])\n\t"
    "movq %%mm3,0x70(%[residue])\n\t"
    "movq %%mm5,0x78(%[residue])\n\t"
    :[src]"+r"(_src),[ystride3]"=&r"(ystride3)
    :[residue]"r"(_residue),[ystride]"r"((ptrdiff_t)_ystride)
    :"memory"
  );
}

void oc_enc_frag_copy2_mmxext(unsigned char *_dst,
 const unsigned char *_src1,const unsigned char *_src2,int _ystride){
  oc_int_frag_copy2_mmxext(_dst,_ystride,_src1,_src2,_ystride);
}

#endif
