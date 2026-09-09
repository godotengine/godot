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
#include "sse2trans.h"

#if defined(OC_X86_ASM)

/*Load a 4x8 array of pixels values from %[src] and %[ref] and compute their
   16-bit differences.
  On output, these are stored in _m0, xmm1, xmm2, and xmm3.
  xmm4 and xmm5 are clobbered.*/
#define OC_LOAD_SUB_4x8(_m0) \
 "#OC_LOAD_SUB_4x8\n\t" \
 /*Load the first three rows.*/ \
 "movq (%[src]),"_m0"\n\t" \
 "movq (%[ref]),%%xmm4\n\t" \
 "movq (%[src],%[ystride]),%%xmm1\n\t" \
 "movq (%[ref],%[ystride]),%%xmm3\n\t" \
 "movq (%[src],%[ystride],2),%%xmm2\n\t" \
 "movq (%[ref],%[ystride],2),%%xmm5\n\t" \
 /*Unpack and subtract.*/ \
 "punpcklbw %%xmm4,"_m0"\n\t" \
 "punpcklbw %%xmm4,%%xmm4\n\t" \
 "punpcklbw %%xmm3,%%xmm1\n\t" \
 "punpcklbw %%xmm3,%%xmm3\n\t" \
 "psubw %%xmm4,"_m0"\n\t" \
 "psubw %%xmm3,%%xmm1\n\t" \
 /*Load the last row.*/ \
 "movq (%[src],%[ystride3]),%%xmm3\n\t" \
 "movq (%[ref],%[ystride3]),%%xmm4\n\t" \
 /*Unpack, subtract, and advance the pointers.*/ \
 "punpcklbw %%xmm5,%%xmm2\n\t" \
 "punpcklbw %%xmm5,%%xmm5\n\t" \
 "lea (%[src],%[ystride],4),%[src]\n\t" \
 "psubw %%xmm5,%%xmm2\n\t" \
 "punpcklbw %%xmm4,%%xmm3\n\t" \
 "punpcklbw %%xmm4,%%xmm4\n\t" \
 "lea (%[ref],%[ystride],4),%[ref]\n\t" \
 "psubw %%xmm4,%%xmm3\n\t" \

/*Square and accumulate four rows of differences in _m0, xmm1, xmm2, and xmm3.
  On output, xmm0 contains the sum of two of the rows, and the other two are
   added to xmm7.*/
#define OC_SSD_4x8(_m0) \
 "pmaddwd "_m0","_m0"\n\t" \
 "pmaddwd %%xmm1,%%xmm1\n\t" \
 "pmaddwd %%xmm2,%%xmm2\n\t" \
 "pmaddwd %%xmm3,%%xmm3\n\t" \
 "paddd %%xmm1,"_m0"\n\t" \
 "paddd %%xmm3,%%xmm2\n\t" \
 "paddd %%xmm2,%%xmm7\n\t" \

unsigned oc_enc_frag_ssd_sse2(const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  unsigned ret;
  __asm__ __volatile__(
    OC_LOAD_SUB_4x8("%%xmm7")
    OC_SSD_4x8("%%xmm7")
    OC_LOAD_SUB_4x8("%%xmm0")
    OC_SSD_4x8("%%xmm0")
    "paddd %%xmm0,%%xmm7\n\t"
    "movdqa %%xmm7,%%xmm6\n\t"
    "punpckhqdq %%xmm7,%%xmm7\n\t"
    "paddd %%xmm6,%%xmm7\n\t"
    "pshufd $1,%%xmm7,%%xmm6\n\t"
    "paddd %%xmm6,%%xmm7\n\t"
    "movd %%xmm7,%[ret]\n\t"
    :[ret]"=a"(ret)
    :[src]"r"(_src),[ref]"r"(_ref),[ystride]"r"((ptrdiff_t)_ystride),
     [ystride3]"r"((ptrdiff_t)_ystride*3)
  );
  return ret;
}

static const unsigned char __attribute__((aligned(16))) OC_MASK_CONSTS[8]={
  0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80
};

/*Load a 2x8 array of pixels values from %[src] and %[ref] and compute their
   horizontal sums as well as their 16-bit differences subject to a mask.
  %%xmm5 must contain OC_MASK_CONSTS[0...7] and %%xmm6 must contain 0.*/
#define OC_LOAD_SUB_MASK_2x8 \
 "#OC_LOAD_SUB_MASK_2x8\n\t" \
 /*Start the loads and expand the next 8 bits of the mask.*/ \
 "shl $8,%[m]\n\t" \
 "movq (%[src]),%%xmm0\n\t" \
 "mov %h[m],%b[m]\n\t" \
 "movq (%[ref]),%%xmm2\n\t" \
 "movd %[m],%%xmm4\n\t" \
 "shr $8,%[m]\n\t" \
 "pshuflw $0x00,%%xmm4,%%xmm4\n\t" \
 "mov %h[m],%b[m]\n\t" \
 "pand %%xmm6,%%xmm4\n\t" \
 "pcmpeqb %%xmm6,%%xmm4\n\t" \
 /*Perform the masking.*/ \
 "pand %%xmm4,%%xmm0\n\t" \
 "pand %%xmm4,%%xmm2\n\t" \
 /*Finish the loads while unpacking the first set of rows, and expand the next
    8 bits of the mask.*/ \
 "movd %[m],%%xmm4\n\t" \
 "movq (%[src],%[ystride]),%%xmm1\n\t" \
 "pshuflw $0x00,%%xmm4,%%xmm4\n\t" \
 "movq (%[ref],%[ystride]),%%xmm3\n\t" \
 "pand %%xmm6,%%xmm4\n\t" \
 "punpcklbw %%xmm2,%%xmm0\n\t" \
 "pcmpeqb %%xmm6,%%xmm4\n\t" \
 "punpcklbw %%xmm2,%%xmm2\n\t" \
 /*Mask and unpack the second set of rows.*/ \
 "pand %%xmm4,%%xmm1\n\t" \
 "pand %%xmm4,%%xmm3\n\t" \
 "punpcklbw %%xmm3,%%xmm1\n\t" \
 "punpcklbw %%xmm3,%%xmm3\n\t" \
 "psubw %%xmm2,%%xmm0\n\t" \
 "psubw %%xmm3,%%xmm1\n\t" \

unsigned oc_enc_frag_border_ssd_sse2(const unsigned char *_src,
 const unsigned char *_ref,int _ystride,ogg_int64_t _mask){
  ptrdiff_t ystride;
  unsigned  ret;
  int       i;
  ystride=_ystride;
  __asm__ __volatile__(
    "pxor %%xmm7,%%xmm7\n\t"
    "movq %[c],%%xmm6\n\t"
    :
    :[c]"m"(OC_CONST_ARRAY_OPERAND(unsigned char,OC_MASK_CONSTS,8))
  );
  for(i=0;i<4;i++){
    unsigned m;
    m=_mask&0xFFFF;
    _mask>>=16;
    if(m){
      __asm__ __volatile__(
        OC_LOAD_SUB_MASK_2x8
        "pmaddwd %%xmm0,%%xmm0\n\t"
        "pmaddwd %%xmm1,%%xmm1\n\t"
        "paddd %%xmm0,%%xmm7\n\t"
        "paddd %%xmm1,%%xmm7\n\t"
        :[src]"+r"(_src),[ref]"+r"(_ref),[ystride]"+r"(ystride),[m]"+Q"(m)
      );
    }
    _src+=2*ystride;
    _ref+=2*ystride;
  }
  __asm__ __volatile__(
    "movdqa %%xmm7,%%xmm6\n\t"
    "punpckhqdq %%xmm7,%%xmm7\n\t"
    "paddd %%xmm6,%%xmm7\n\t"
    "pshufd $1,%%xmm7,%%xmm6\n\t"
    "paddd %%xmm6,%%xmm7\n\t"
    "movd %%xmm7,%[ret]\n\t"
    :[ret]"=a"(ret)
  );
  return ret;
}


/*Load an 8x8 array of pixel values from %[src] and %[ref] and compute their
   16-bit difference in %%xmm0...%%xmm7.*/
#define OC_LOAD_SUB_8x8 \
 "#OC_LOAD_SUB_8x8\n\t" \
 "movq (%[src]),%%xmm0\n\t" \
 "movq (%[ref]),%%xmm4\n\t" \
 "movq (%[src],%[src_ystride]),%%xmm1\n\t" \
 "lea (%[src],%[src_ystride],2),%[src]\n\t" \
 "movq (%[ref],%[ref_ystride]),%%xmm5\n\t" \
 "lea (%[ref],%[ref_ystride],2),%[ref]\n\t" \
 "movq (%[src]),%%xmm2\n\t" \
 "movq (%[ref]),%%xmm7\n\t" \
 "movq (%[src],%[src_ystride]),%%xmm3\n\t" \
 "movq (%[ref],%[ref_ystride]),%%xmm6\n\t" \
 "punpcklbw %%xmm4,%%xmm0\n\t" \
 "lea (%[src],%[src_ystride],2),%[src]\n\t" \
 "punpcklbw %%xmm4,%%xmm4\n\t" \
 "lea (%[ref],%[ref_ystride],2),%[ref]\n\t" \
 "psubw %%xmm4,%%xmm0\n\t" \
 "movq (%[src]),%%xmm4\n\t" \
 "movdqa %%xmm0,"OC_MEM_OFFS(0x00,buf)"\n\t" \
 "movq (%[ref]),%%xmm0\n\t" \
 "punpcklbw %%xmm5,%%xmm1\n\t" \
 "punpcklbw %%xmm5,%%xmm5\n\t" \
 "psubw %%xmm5,%%xmm1\n\t" \
 "movq (%[src],%[src_ystride]),%%xmm5\n\t" \
 "punpcklbw %%xmm7,%%xmm2\n\t" \
 "punpcklbw %%xmm7,%%xmm7\n\t" \
 "psubw %%xmm7,%%xmm2\n\t" \
 "movq (%[ref],%[ref_ystride]),%%xmm7\n\t" \
 "punpcklbw %%xmm6,%%xmm3\n\t" \
 "lea (%[src],%[src_ystride],2),%[src]\n\t" \
 "punpcklbw %%xmm6,%%xmm6\n\t" \
 "psubw %%xmm6,%%xmm3\n\t" \
 "movq (%[src]),%%xmm6\n\t" \
 "punpcklbw %%xmm0,%%xmm4\n\t" \
 "lea (%[ref],%[ref_ystride],2),%[ref]\n\t" \
 "punpcklbw %%xmm0,%%xmm0\n\t" \
 "lea (%[src],%[src_ystride],2),%[src]\n\t" \
 "psubw %%xmm0,%%xmm4\n\t" \
 "movq (%[ref]),%%xmm0\n\t" \
 "punpcklbw %%xmm7,%%xmm5\n\t" \
 "neg %[src_ystride]\n\t" \
 "punpcklbw %%xmm7,%%xmm7\n\t" \
 "psubw %%xmm7,%%xmm5\n\t" \
 "movq (%[src],%[src_ystride]),%%xmm7\n\t" \
 "punpcklbw %%xmm0,%%xmm6\n\t" \
 "lea (%[ref],%[ref_ystride],2),%[ref]\n\t" \
 "punpcklbw %%xmm0,%%xmm0\n\t" \
 "neg %[ref_ystride]\n\t" \
 "psubw %%xmm0,%%xmm6\n\t" \
 "movq (%[ref],%[ref_ystride]),%%xmm0\n\t" \
 "punpcklbw %%xmm0,%%xmm7\n\t" \
 "punpcklbw %%xmm0,%%xmm0\n\t" \
 "psubw %%xmm0,%%xmm7\n\t" \
 "movdqa "OC_MEM_OFFS(0x00,buf)",%%xmm0\n\t" \

/*Load an 8x8 array of pixel values from %[src] into %%xmm0...%%xmm7.*/
#define OC_LOAD_8x8 \
 "#OC_LOAD_8x8\n\t" \
 "movq (%[src]),%%xmm0\n\t" \
 "movq (%[src],%[ystride]),%%xmm1\n\t" \
 "movq (%[src],%[ystride],2),%%xmm2\n\t" \
 "pxor %%xmm7,%%xmm7\n\t" \
 "movq (%[src],%[ystride3]),%%xmm3\n\t" \
 "punpcklbw %%xmm7,%%xmm0\n\t" \
 "movq (%[src4]),%%xmm4\n\t" \
 "punpcklbw %%xmm7,%%xmm1\n\t" \
 "movq (%[src4],%[ystride]),%%xmm5\n\t" \
 "punpcklbw %%xmm7,%%xmm2\n\t" \
 "movq (%[src4],%[ystride],2),%%xmm6\n\t" \
 "punpcklbw %%xmm7,%%xmm3\n\t" \
 "movq (%[src4],%[ystride3]),%%xmm7\n\t" \
 "punpcklbw %%xmm4,%%xmm4\n\t" \
 "punpcklbw %%xmm5,%%xmm5\n\t" \
 "psrlw $8,%%xmm4\n\t" \
 "psrlw $8,%%xmm5\n\t" \
 "punpcklbw %%xmm6,%%xmm6\n\t" \
 "punpcklbw %%xmm7,%%xmm7\n\t" \
 "psrlw $8,%%xmm6\n\t" \
 "psrlw $8,%%xmm7\n\t" \

/*Performs the first two stages of an 8-point 1-D Hadamard transform in place.
  Outputs 1, 3, 4, and 5 from the second stage are negated (which allows us to
   perform this stage in place with no temporary registers).*/
#define OC_HADAMARD_AB_8x8 \
 "#OC_HADAMARD_AB_8x8\n\t" \
 /*Stage A:*/ \
 "paddw %%xmm5,%%xmm1\n\t" \
 "paddw %%xmm6,%%xmm2\n\t" \
 "paddw %%xmm5,%%xmm5\n\t" \
 "paddw %%xmm6,%%xmm6\n\t" \
 "psubw %%xmm1,%%xmm5\n\t" \
 "psubw %%xmm2,%%xmm6\n\t" \
 "paddw %%xmm7,%%xmm3\n\t" \
 "paddw %%xmm4,%%xmm0\n\t" \
 "paddw %%xmm7,%%xmm7\n\t" \
 "paddw %%xmm4,%%xmm4\n\t" \
 "psubw %%xmm3,%%xmm7\n\t" \
 "psubw %%xmm0,%%xmm4\n\t" \
 /*Stage B:*/ \
 "paddw %%xmm2,%%xmm0\n\t" \
 "paddw %%xmm3,%%xmm1\n\t" \
 "paddw %%xmm6,%%xmm4\n\t" \
 "paddw %%xmm7,%%xmm5\n\t" \
 "paddw %%xmm2,%%xmm2\n\t" \
 "paddw %%xmm3,%%xmm3\n\t" \
 "paddw %%xmm6,%%xmm6\n\t" \
 "paddw %%xmm7,%%xmm7\n\t" \
 "psubw %%xmm0,%%xmm2\n\t" \
 "psubw %%xmm1,%%xmm3\n\t" \
 "psubw %%xmm4,%%xmm6\n\t" \
 "psubw %%xmm5,%%xmm7\n\t" \

/*Performs the last stage of an 8-point 1-D Hadamard transform in place.
  Outputs 1, 3, 5, and 7 are negated (which allows us to perform this stage in
   place with no temporary registers).*/
#define OC_HADAMARD_C_8x8 \
 "#OC_HADAMARD_C_8x8\n\t" \
 /*Stage C:*/ \
 "paddw %%xmm1,%%xmm0\n\t" \
 "paddw %%xmm3,%%xmm2\n\t" \
 "paddw %%xmm5,%%xmm4\n\t" \
 "paddw %%xmm7,%%xmm6\n\t" \
 "paddw %%xmm1,%%xmm1\n\t" \
 "paddw %%xmm3,%%xmm3\n\t" \
 "paddw %%xmm5,%%xmm5\n\t" \
 "paddw %%xmm7,%%xmm7\n\t" \
 "psubw %%xmm0,%%xmm1\n\t" \
 "psubw %%xmm2,%%xmm3\n\t" \
 "psubw %%xmm4,%%xmm5\n\t" \
 "psubw %%xmm6,%%xmm7\n\t" \

/*Performs an 8-point 1-D Hadamard transform in place.
  Outputs 1, 2, 4, and 7 are negated (which allows us to perform the transform
   in place with no temporary registers).*/
#define OC_HADAMARD_8x8 \
 OC_HADAMARD_AB_8x8 \
 OC_HADAMARD_C_8x8 \

/*Performs the first part of the final stage of the Hadamard transform and
   summing of absolute values.
  At the end of this part, %%xmm1 will contain the DC coefficient of the
   transform.*/
#define OC_HADAMARD_C_ABS_ACCUM_A_8x8 \
 /*We use the fact that \
     (abs(a+b)+abs(a-b))/2=max(abs(a),abs(b)) \
    to merge the final butterfly with the abs and the first stage of \
    accumulation. \
   Thus we can avoid using pabsw, which is not available until SSSE3. \
   Emulating pabsw takes 3 instructions, so the straightforward SSE2 \
    implementation would be (3+3)*8+7=55 instructions (+4 for spilling \
    registers). \
   Even with pabsw, it would be (3+1)*8+7=39 instructions (with no spills). \
   This implementation is only 26 (+4 for spilling registers).*/ \
 "#OC_HADAMARD_C_ABS_ACCUM_A_8x8\n\t" \
 "movdqa %%xmm7,"OC_MEM_OFFS(0x10,buf)"\n\t" \
 "movdqa %%xmm6,"OC_MEM_OFFS(0x00,buf)"\n\t" \
 /*xmm7={0x7FFF}x4 \
   xmm4=max(abs(xmm4),abs(xmm5))-0x7FFF*/ \
 "pcmpeqb %%xmm7,%%xmm7\n\t" \
 "movdqa %%xmm4,%%xmm6\n\t" \
 "psrlw $1,%%xmm7\n\t" \
 "paddw %%xmm5,%%xmm6\n\t" \
 "pmaxsw %%xmm5,%%xmm4\n\t" \
 "paddsw %%xmm7,%%xmm6\n\t" \
 "psubw %%xmm6,%%xmm4\n\t" \
 /*xmm2=max(abs(xmm2),abs(xmm3))-0x7FFF \
   xmm0=max(abs(xmm0),abs(xmm1))-0x7FFF*/ \
 "movdqa %%xmm2,%%xmm6\n\t" \
 "movdqa %%xmm0,%%xmm5\n\t" \
 "pmaxsw %%xmm3,%%xmm2\n\t" \
 "pmaxsw %%xmm1,%%xmm0\n\t" \
 "paddw %%xmm3,%%xmm6\n\t" \
 "movdqa "OC_MEM_OFFS(0x10,buf)",%%xmm3\n\t" \
 "paddw %%xmm5,%%xmm1\n\t" \
 "movdqa "OC_MEM_OFFS(0x00,buf)",%%xmm5\n\t" \

/*Performs the second part of the final stage of the Hadamard transform and
   summing of absolute values.*/
#define OC_HADAMARD_C_ABS_ACCUM_B_8x8 \
 "#OC_HADAMARD_C_ABS_ACCUM_B_8x8\n\t" \
 "paddsw %%xmm7,%%xmm6\n\t" \
 "paddsw %%xmm7,%%xmm1\n\t" \
 "psubw %%xmm6,%%xmm2\n\t" \
 "psubw %%xmm1,%%xmm0\n\t" \
 /*xmm7={1}x4 (needed for the horizontal add that follows) \
   xmm0+=xmm2+xmm4+max(abs(xmm3),abs(xmm5))-0x7FFF*/ \
 "movdqa %%xmm3,%%xmm6\n\t" \
 "pmaxsw %%xmm5,%%xmm3\n\t" \
 "paddw %%xmm2,%%xmm0\n\t" \
 "paddw %%xmm5,%%xmm6\n\t" \
 "paddw %%xmm4,%%xmm0\n\t" \
 "paddsw %%xmm7,%%xmm6\n\t" \
 "paddw %%xmm3,%%xmm0\n\t" \
 "psrlw $14,%%xmm7\n\t" \
 "psubw %%xmm6,%%xmm0\n\t" \

/*Performs the last stage of an 8-point 1-D Hadamard transform, takes the
   absolute value of each component, and accumulates everything into xmm0.*/
#define OC_HADAMARD_C_ABS_ACCUM_8x8 \
 OC_HADAMARD_C_ABS_ACCUM_A_8x8 \
 OC_HADAMARD_C_ABS_ACCUM_B_8x8 \

/*Performs an 8-point 1-D Hadamard transform, takes the absolute value of each
   component, and accumulates everything into xmm0.
  Note that xmm0 will have an extra 4 added to each column, and that after
   removing this value, the remainder will be half the conventional value.*/
#define OC_HADAMARD_ABS_ACCUM_8x8 \
 OC_HADAMARD_AB_8x8 \
 OC_HADAMARD_C_ABS_ACCUM_8x8

static unsigned oc_int_frag_satd_sse2(int *_dc,
 const unsigned char *_src,int _src_ystride,
 const unsigned char *_ref,int _ref_ystride){
  OC_ALIGN16(ogg_int16_t buf[16]);
  unsigned ret;
  unsigned ret2;
  int      dc;
  __asm__ __volatile__(
    OC_LOAD_SUB_8x8
    OC_HADAMARD_8x8
    OC_TRANSPOSE_8x8
    /*We split out the stages here so we can save the DC coefficient in the
       middle.*/
    OC_HADAMARD_AB_8x8
    OC_HADAMARD_C_ABS_ACCUM_A_8x8
    "movd %%xmm1,%[dc]\n\t"
    OC_HADAMARD_C_ABS_ACCUM_B_8x8
    /*Up to this point, everything fit in 16 bits (8 input + 1 for the
       difference + 2*3 for the two 8-point 1-D Hadamards - 1 for the abs - 1
       for the factor of two we dropped + 3 for the vertical accumulation).
      Now we finally have to promote things to dwords.
      We break this part out of OC_HADAMARD_ABS_ACCUM_8x8 to hide the long
       latency of pmaddwd by starting to compute abs(dc) here.*/
    "pmaddwd %%xmm7,%%xmm0\n\t"
    "movsx %w[dc],%[dc]\n\t"
    "cdq\n\t"
    "movdqa %%xmm0,%%xmm1\n\t"
    "punpckhqdq %%xmm0,%%xmm0\n\t"
    "paddd %%xmm1,%%xmm0\n\t"
    "pshuflw $0xE,%%xmm0,%%xmm1\n\t"
    "paddd %%xmm1,%%xmm0\n\t"
    "movd %%xmm0,%[ret]\n\t"
    /*The sums produced by OC_HADAMARD_ABS_ACCUM_8x8 each have an extra 4
       added to them, a factor of two removed, and the DC value included;
       correct the final sum here.*/
    "lea -64(%[ret2],%[ret],2),%[ret]\n\t"
    "xor %[dc],%[ret2]\n\t"
    "sub %[ret2],%[ret]\n\t"
    /*Although it looks like we're using 7 registers here, gcc can alias %[ret]
       and %[dc] with some of the inputs, since for once we don't write to
       them until after we're done using everything but %[buf].*/
    /*Note that _src_ystride and _ref_ystride must be given non-overlapping
       constraints, otherwise if gcc can prove they're equal it will allocate
       them to the same register (which is bad); _src and _ref face a similar
       problem.
      All four are destructively modified, but if we list them as output
       constraints, gcc can't alias them with other outputs.*/
    :[ret]"=r"(ret),[ret2]"=d"(ret2),[dc]"=a"(dc),
     [buf]"=m"(OC_ARRAY_OPERAND(ogg_int16_t,buf,16))
    :[src]"S"(_src),[src_ystride]"c"((ptrdiff_t)_src_ystride),
     [ref]"a"(_ref),[ref_ystride]"d"((ptrdiff_t)_ref_ystride)
    /*We have to use neg, so we actually clobber the condition codes for once
       (not to mention sub, and add).*/
    :"cc"
  );
  *_dc=dc;
  return ret;
}

unsigned oc_enc_frag_satd_sse2(int *_dc,const unsigned char *_src,
 const unsigned char *_ref,int _ystride){
  return oc_int_frag_satd_sse2(_dc,_src,_ystride,_ref,_ystride);
}

unsigned oc_enc_frag_satd2_sse2(int *_dc,const unsigned char *_src,
 const unsigned char *_ref1,const unsigned char *_ref2,int _ystride){
  OC_ALIGN8(unsigned char ref[64]);
  oc_int_frag_copy2_mmxext(ref,8,_ref1,_ref2,_ystride);
  return oc_int_frag_satd_sse2(_dc,_src,_ystride,ref,8);
}

unsigned oc_enc_frag_intra_satd_sse2(int *_dc,
 const unsigned char *_src,int _ystride){
  OC_ALIGN16(ogg_int16_t buf[16]);
  unsigned ret;
  int      dc;
  __asm__ __volatile__(
    OC_LOAD_8x8
    OC_HADAMARD_8x8
    OC_TRANSPOSE_8x8
    /*We split out the stages here so we can save the DC coefficient in the
       middle.*/
    OC_HADAMARD_AB_8x8
    OC_HADAMARD_C_ABS_ACCUM_A_8x8
    "movd %%xmm1,%[dc]\n\t"
    OC_HADAMARD_C_ABS_ACCUM_B_8x8
    /*Up to this point, everything fit in 16 bits (8 input + 1 for the
       difference + 2*3 for the two 8-point 1-D Hadamards - 1 for the abs - 1
       for the factor of two we dropped + 3 for the vertical accumulation).
      Now we finally have to promote things to dwords.*/
    "pmaddwd %%xmm7,%%xmm0\n\t"
    /*We assume that the DC coefficient is always positive (which is true,
       because the input to the INTRA transform was not a difference).*/
    "movzx %w[dc],%[dc]\n\t"
    "movdqa %%xmm0,%%xmm1\n\t"
    "punpckhqdq %%xmm0,%%xmm0\n\t"
    "paddd %%xmm1,%%xmm0\n\t"
    "pshuflw $0xE,%%xmm0,%%xmm1\n\t"
    "paddd %%xmm1,%%xmm0\n\t"
    "movd %%xmm0,%[ret]\n\t"
    "lea -64(%[ret],%[ret]),%[ret]\n\t"
    "sub %[dc],%[ret]\n\t"
    /*Although it looks like we're using 7 registers here, gcc can alias %[ret]
       and %[dc] with some of the inputs, since for once we don't write to
       them until after we're done using everything but %[buf].*/
    :[ret]"=a"(ret),[dc]"=r"(dc),
     [buf]"=m"(OC_ARRAY_OPERAND(ogg_int16_t,buf,16))
    :[src]"r"(_src),[src4]"r"(_src+4*_ystride),
     [ystride]"r"((ptrdiff_t)_ystride),[ystride3]"r"((ptrdiff_t)3*_ystride)
    /*We have to use sub, so we actually clobber the condition codes for once.*/
    :"cc"
  );
  *_dc=dc;
  return ret;
}

#endif
