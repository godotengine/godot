#if !defined(_x86_vc_mmxloop_H)
# define _x86_vc_mmxloop_H (1)
# include <stddef.h>
# include "x86int.h"

#if defined(OC_X86_ASM)

/*On entry, mm0={a0,...,a7}, mm1={b0,...,b7}, mm2={c0,...,c7}, mm3={d0,...d7}.
  On exit, mm1={b0+lflim(R_0,L),...,b7+lflim(R_7,L)} and
   mm2={c0-lflim(R_0,L),...,c7-lflim(R_7,L)}; mm0 and mm3 are clobbered.*/
#define OC_LOOP_FILTER8_MMX __asm{ \
  /*mm7=0*/ \
  __asm pxor mm7,mm7 \
  /*mm6:mm0={a0,...,a7}*/ \
  __asm movq mm6,mm0 \
  __asm punpcklbw mm0,mm7 \
  __asm punpckhbw mm6,mm7 \
  /*mm3:mm5={d0,...,d7}*/ \
  __asm movq mm5,mm3 \
  __asm punpcklbw mm3,mm7 \
  __asm punpckhbw mm5,mm7 \
  /*mm6:mm0={a0-d0,...,a7-d7}*/ \
  __asm psubw mm0,mm3 \
  __asm psubw mm6,mm5 \
  /*mm3:mm1={b0,...,b7}*/ \
  __asm movq mm3,mm1 \
  __asm punpcklbw mm1,mm7 \
  __asm movq mm4,mm2 \
  __asm punpckhbw mm3,mm7 \
  /*mm5:mm4={c0,...,c7}*/ \
  __asm movq mm5,mm2 \
  __asm punpcklbw mm4,mm7 \
  __asm punpckhbw mm5,mm7 \
  /*mm7={3}x4 \
    mm5:mm4={c0-b0,...,c7-b7}*/ \
  __asm pcmpeqw mm7,mm7 \
  __asm psubw mm4,mm1 \
  __asm psrlw mm7,14 \
  __asm psubw mm5,mm3 \
  /*Scale by 3.*/ \
  __asm pmullw mm4,mm7 \
  __asm pmullw mm5,mm7 \
  /*mm7={4}x4 \
    mm5:mm4=f={a0-d0+3*(c0-b0),...,a7-d7+3*(c7-b7)}*/ \
  __asm psrlw mm7,1 \
  __asm paddw mm4,mm0 \
  __asm psllw mm7,2 \
  __asm movq mm0,[LL] \
  __asm paddw mm5,mm6 \
  /*R_i has the range [-127,128], so we compute -R_i instead. \
    mm4=-R_i=-(f+4>>3)=0xFF^(f-4>>3)*/ \
  __asm psubw mm4,mm7 \
  __asm psubw mm5,mm7 \
  __asm psraw mm4,3 \
  __asm psraw mm5,3 \
  __asm pcmpeqb mm7,mm7 \
  __asm packsswb mm4,mm5 \
  __asm pxor mm6,mm6 \
  __asm pxor mm4,mm7 \
  __asm packuswb mm1,mm3 \
  /*Now compute lflim of -mm4 cf. Section 7.10 of the sepc.*/ \
  /*There's no unsigned byte+signed byte with unsigned saturation op code, so \
     we have to split things by sign (the other option is to work in 16 bits, \
     but working in 8 bits gives much better parallelism). \
    We compute abs(R_i), but save a mask of which terms were negative in mm6. \
    Then we compute mm4=abs(lflim(R_i,L))=min(abs(R_i),max(2*L-abs(R_i),0)). \
    Finally, we split mm4 into positive and negative pieces using the mask in \
     mm6, and add and subtract them as appropriate.*/ \
  /*mm4=abs(-R_i)*/ \
  /*mm7=255-2*L*/ \
  __asm pcmpgtb mm6,mm4 \
  __asm psubb mm7,mm0 \
  __asm pxor mm4,mm6 \
  __asm psubb mm7,mm0 \
  __asm psubb mm4,mm6 \
  /*mm7=255-max(2*L-abs(R_i),0)*/ \
  __asm paddusb mm7,mm4 \
  /*mm4=min(abs(R_i),max(2*L-abs(R_i),0))*/ \
  __asm paddusb mm4,mm7 \
  __asm psubusb mm4,mm7 \
  /*Now split mm4 by the original sign of -R_i.*/ \
  __asm movq mm5,mm4 \
  __asm pand mm4,mm6 \
  __asm pandn mm6,mm5 \
  /*mm1={b0+lflim(R_0,L),...,b7+lflim(R_7,L)}*/ \
  /*mm2={c0-lflim(R_0,L),...,c7-lflim(R_7,L)}*/ \
  __asm paddusb mm1,mm4 \
  __asm psubusb mm2,mm4 \
  __asm psubusb mm1,mm6 \
  __asm paddusb mm2,mm6 \
}

#define OC_LOOP_FILTER_V_MMX(_pix,_ystride,_ll) \
  do{ \
    /*Used local variable pix__ in order to fix compilation errors like: \
       "error C2425: 'SHL' : non-constant expression in 'second operand'".*/ \
    unsigned char *pix__; \
    unsigned char *ll__; \
    ll__=(_ll); \
    pix__=(_pix); \
    __asm mov YSTRIDE,_ystride \
    __asm mov LL,ll__ \
    __asm mov PIX,pix__ \
    __asm sub PIX,YSTRIDE \
    __asm sub PIX,YSTRIDE \
    /*mm0={a0,...,a7}*/ \
    __asm movq mm0,[PIX] \
    /*ystride3=_ystride*3*/ \
    __asm lea YSTRIDE3,[YSTRIDE+YSTRIDE*2] \
    /*mm3={d0,...,d7}*/ \
    __asm movq mm3,[PIX+YSTRIDE3] \
    /*mm1={b0,...,b7}*/ \
    __asm movq mm1,[PIX+YSTRIDE] \
    /*mm2={c0,...,c7}*/ \
    __asm movq mm2,[PIX+YSTRIDE*2] \
    OC_LOOP_FILTER8_MMX \
    /*Write it back out.*/ \
    __asm movq [PIX+YSTRIDE],mm1 \
    __asm movq [PIX+YSTRIDE*2],mm2 \
  } \
  while(0)

#define OC_LOOP_FILTER_H_MMX(_pix,_ystride,_ll) \
  do{ \
    /*Used local variable ll__ in order to fix compilation errors like: \
       "error C2443: operand size conflict".*/ \
    unsigned char *ll__; \
    unsigned char *pix__; \
    ll__=(_ll); \
    pix__=(_pix)-2; \
    __asm mov PIX,pix__ \
    __asm mov YSTRIDE,_ystride \
    __asm mov LL,ll__ \
    /*x x x x d0 c0 b0 a0*/ \
    __asm movd mm0,[PIX] \
    /*x x x x d1 c1 b1 a1*/ \
    __asm movd mm1,[PIX+YSTRIDE] \
    /*ystride3=_ystride*3*/ \
    __asm lea YSTRIDE3,[YSTRIDE+YSTRIDE*2] \
    /*x x x x d2 c2 b2 a2*/ \
    __asm movd mm2,[PIX+YSTRIDE*2] \
    /*x x x x d3 c3 b3 a3*/ \
    __asm lea D,[PIX+YSTRIDE*4] \
    __asm movd mm3,[PIX+YSTRIDE3] \
    /*x x x x d4 c4 b4 a4*/ \
    __asm movd mm4,[D] \
    /*x x x x d5 c5 b5 a5*/ \
    __asm movd mm5,[D+YSTRIDE] \
    /*x x x x d6 c6 b6 a6*/ \
    __asm movd mm6,[D+YSTRIDE*2] \
    /*x x x x d7 c7 b7 a7*/ \
    __asm movd mm7,[D+YSTRIDE3] \
    /*mm0=d1 d0 c1 c0 b1 b0 a1 a0*/ \
    __asm punpcklbw mm0,mm1 \
    /*mm2=d3 d2 c3 c2 b3 b2 a3 a2*/ \
    __asm punpcklbw mm2,mm3 \
    /*mm3=d1 d0 c1 c0 b1 b0 a1 a0*/ \
    __asm movq mm3,mm0 \
    /*mm0=b3 b2 b1 b0 a3 a2 a1 a0*/ \
    __asm punpcklwd mm0,mm2 \
    /*mm3=d3 d2 d1 d0 c3 c2 c1 c0*/ \
    __asm punpckhwd mm3,mm2 \
    /*mm1=b3 b2 b1 b0 a3 a2 a1 a0*/ \
    __asm movq mm1,mm0 \
    /*mm4=d5 d4 c5 c4 b5 b4 a5 a4*/ \
    __asm punpcklbw mm4,mm5 \
    /*mm6=d7 d6 c7 c6 b7 b6 a7 a6*/ \
    __asm punpcklbw mm6,mm7 \
    /*mm5=d5 d4 c5 c4 b5 b4 a5 a4*/ \
    __asm movq mm5,mm4 \
    /*mm4=b7 b6 b5 b4 a7 a6 a5 a4*/ \
    __asm punpcklwd mm4,mm6 \
    /*mm5=d7 d6 d5 d4 c7 c6 c5 c4*/ \
    __asm punpckhwd mm5,mm6 \
    /*mm2=d3 d2 d1 d0 c3 c2 c1 c0*/ \
    __asm movq mm2,mm3 \
    /*mm0=a7 a6 a5 a4 a3 a2 a1 a0*/ \
    __asm punpckldq mm0,mm4 \
    /*mm1=b7 b6 b5 b4 b3 b2 b1 b0*/ \
    __asm punpckhdq mm1,mm4 \
    /*mm2=c7 c6 c5 c4 c3 c2 c1 c0*/ \
    __asm punpckldq mm2,mm5 \
    /*mm3=d7 d6 d5 d4 d3 d2 d1 d0*/ \
    __asm punpckhdq mm3,mm5 \
    OC_LOOP_FILTER8_MMX \
    /*mm2={b0+R_0'',...,b7+R_7''}*/ \
    __asm movq mm0,mm1 \
    /*mm1={b0+R_0'',c0-R_0'',...,b3+R_3'',c3-R_3''}*/ \
    __asm punpcklbw mm1,mm2 \
    /*mm2={b4+R_4'',c4-R_4'',...,b7+R_7'',c7-R_7''}*/ \
    __asm punpckhbw mm0,mm2 \
    /*[d]=c1 b1 c0 b0*/ \
    __asm movd D,mm1 \
    __asm mov [PIX+1],D_WORD \
    __asm psrlq mm1,32 \
    __asm shr D,16 \
    __asm mov [PIX+YSTRIDE+1],D_WORD \
    /*[d]=c3 b3 c2 b2*/ \
    __asm movd D,mm1 \
    __asm mov [PIX+YSTRIDE*2+1],D_WORD \
    __asm shr D,16 \
    __asm mov [PIX+YSTRIDE3+1],D_WORD \
    __asm lea PIX,[PIX+YSTRIDE*4] \
    /*[d]=c5 b5 c4 b4*/ \
    __asm movd D,mm0 \
    __asm mov [PIX+1],D_WORD \
    __asm psrlq mm0,32 \
    __asm shr D,16 \
    __asm mov [PIX+YSTRIDE+1],D_WORD \
    /*[d]=c7 b7 c6 b6*/ \
    __asm movd D,mm0 \
    __asm mov [PIX+YSTRIDE*2+1],D_WORD \
    __asm shr D,16 \
    __asm mov [PIX+YSTRIDE3+1],D_WORD \
  } \
  while(0)

# endif
#endif
