#if !defined(_x86_mmxfrag_H)
# define _x86_mmxfrag_H (1)
# include <stddef.h>
# include "x86int.h"

#if defined(OC_X86_ASM)

/*Copies an 8x8 block of pixels from _src to _dst, assuming _ystride bytes
   between rows.*/
#define OC_FRAG_COPY_MMX(_dst,_src,_ystride) \
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

# endif
#endif
