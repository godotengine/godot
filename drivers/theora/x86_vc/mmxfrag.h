#if !defined(_x86_vc_mmxfrag_H)
# define _x86_vc_mmxfrag_H (1)
# include <stddef.h>
# include "x86int.h"

#if defined(OC_X86_ASM)

/*Copies an 8x8 block of pixels from _src to _dst, assuming _ystride bytes
   between rows.*/
#define OC_FRAG_COPY_MMX(_dst,_src,_ystride) \
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

# endif
#endif
