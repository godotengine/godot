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

 CPU capability detection for x86 processors.
  Originally written by Rudolf Marek.

 function:

 ********************************************************************/

#include "x86cpu.h"

#if !defined(OC_X86_ASM)
ogg_uint32_t oc_cpu_flags_get(void){
  return 0;
}
#else
# if defined(__amd64__)||defined(__x86_64__)
/*On x86-64, gcc seems to be able to figure out how to save %rbx for us when
   compiling with -fPIC.*/
#  define cpuid(_op,_eax,_ebx,_ecx,_edx) \
  __asm__ __volatile__( \
   "cpuid\n\t" \
   :[eax]"=a"(_eax),[ebx]"=b"(_ebx),[ecx]"=c"(_ecx),[edx]"=d"(_edx) \
   :"a"(_op) \
   :"cc" \
  )
# else
/*On x86-32, not so much.*/
#  define cpuid(_op,_eax,_ebx,_ecx,_edx) \
  __asm__ __volatile__( \
   "xchgl %%ebx,%[ebx]\n\t" \
   "cpuid\n\t" \
   "xchgl %%ebx,%[ebx]\n\t" \
   :[eax]"=a"(_eax),[ebx]"=r"(_ebx),[ecx]"=c"(_ecx),[edx]"=d"(_edx) \
   :"a"(_op) \
   :"cc" \
  )
# endif

static ogg_uint32_t oc_parse_intel_flags(ogg_uint32_t _edx,ogg_uint32_t _ecx){
  ogg_uint32_t flags;
  /*If there isn't even MMX, give up.*/
  if(!(_edx&0x00800000))return 0;
  flags=OC_CPU_X86_MMX;
  if(_edx&0x02000000)flags|=OC_CPU_X86_MMXEXT|OC_CPU_X86_SSE;
  if(_edx&0x04000000)flags|=OC_CPU_X86_SSE2;
  if(_ecx&0x00000001)flags|=OC_CPU_X86_PNI;
  if(_ecx&0x00000100)flags|=OC_CPU_X86_SSSE3;
  if(_ecx&0x00080000)flags|=OC_CPU_X86_SSE4_1;
  if(_ecx&0x00100000)flags|=OC_CPU_X86_SSE4_2;
  return flags;
}

static ogg_uint32_t oc_parse_amd_flags(ogg_uint32_t _edx,ogg_uint32_t _ecx){
  ogg_uint32_t flags;
  /*If there isn't even MMX, give up.*/
  if(!(_edx&0x00800000))return 0;
  flags=OC_CPU_X86_MMX;
  if(_edx&0x00400000)flags|=OC_CPU_X86_MMXEXT;
  if(_edx&0x80000000)flags|=OC_CPU_X86_3DNOW;
  if(_edx&0x40000000)flags|=OC_CPU_X86_3DNOWEXT;
  if(_ecx&0x00000040)flags|=OC_CPU_X86_SSE4A;
  if(_ecx&0x00000800)flags|=OC_CPU_X86_SSE5;
  return flags;
}

ogg_uint32_t oc_cpu_flags_get(void){
  ogg_uint32_t flags;
  ogg_uint32_t eax;
  ogg_uint32_t ebx;
  ogg_uint32_t ecx;
  ogg_uint32_t edx;
# if !defined(__amd64__)&&!defined(__x86_64__)
  /*Not all x86-32 chips support cpuid, so we have to check.*/
  __asm__ __volatile__(
   "pushfl\n\t"
   "pushfl\n\t"
   "popl %[a]\n\t"
   "movl %[a],%[b]\n\t"
   "xorl $0x200000,%[a]\n\t"
   "pushl %[a]\n\t"
   "popfl\n\t"
   "pushfl\n\t"
   "popl %[a]\n\t"
   "popfl\n\t"
   :[a]"=r"(eax),[b]"=r"(ebx)
   :
   :"cc"
  );
  /*No cpuid.*/
  if(eax==ebx)return 0;
# endif
  cpuid(0,eax,ebx,ecx,edx);
  /*         l e t n          I e n i          u n e G*/
  if(ecx==0x6C65746E&&edx==0x49656E69&&ebx==0x756E6547||
   /*      6 8 x M          T e n i          u n e G*/
   ecx==0x3638784D&&edx==0x54656E69&&ebx==0x756E6547){
    int family;
    int model;
    /*Intel, Transmeta (tested with Crusoe TM5800):*/
    cpuid(1,eax,ebx,ecx,edx);
    flags=oc_parse_intel_flags(edx,ecx);
    family=(eax>>8)&0xF;
    model=(eax>>4)&0xF;
    /*The SSE unit on the Pentium M and Core Duo is much slower than the MMX
       unit, so don't use it.*/
    if(family==6&&(model==9||model==13||model==14)){
      flags&=~(OC_CPU_X86_SSE2|OC_CPU_X86_PNI);
    }
  }
  /*              D M A c          i t n e          h t u A*/
  else if(ecx==0x444D4163&&edx==0x69746E65&&ebx==0x68747541||
   /*      C S N            y b   e          d o e G*/
   ecx==0x43534e20&&edx==0x79622065&&ebx==0x646f6547){
    /*AMD, Geode:*/
    cpuid(0x80000000,eax,ebx,ecx,edx);
    if(eax<0x80000001)flags=0;
    else{
      cpuid(0x80000001,eax,ebx,ecx,edx);
      flags=oc_parse_amd_flags(edx,ecx);
    }
    /*Also check for SSE.*/
    cpuid(1,eax,ebx,ecx,edx);
    flags|=oc_parse_intel_flags(edx,ecx);
  }
  /*Technically some VIA chips can be configured in the BIOS to return any
     string here the user wants.
    There is a special detection method that can be used to identify such
     processors, but in my opinion, if the user really wants to change it, they
     deserve what they get.*/
  /*              s l u a          H r u a          t n e C*/
  else if(ecx==0x736C7561&&edx==0x48727561&&ebx==0x746E6543){
    /*VIA:*/
    /*I only have documentation for the C7 (Esther) and Isaiah (forthcoming)
       chips (thanks to the engineers from Centaur Technology who provided it).
      These chips support Intel-like cpuid info.
      The C3-2 (Nehemiah) cores appear to, as well.*/
    cpuid(1,eax,ebx,ecx,edx);
    flags=oc_parse_intel_flags(edx,ecx);
    if(eax>=0x80000001){
      /*The (non-Nehemiah) C3 processors support AMD-like cpuid info.
        We need to check this even if the Intel test succeeds to pick up 3DNow!
         support on these processors.
        Unlike actual AMD processors, we cannot _rely_ on this info, since
         some cores (e.g., the 693 stepping of the Nehemiah) claim to support
         this function, yet return edx=0, despite the Intel test indicating
         MMX support.
        Therefore the features detected here are strictly added to those
         detected by the Intel test.*/
      /*TODO: How about earlier chips?*/
      cpuid(0x80000001,eax,ebx,ecx,edx);
      /*Note: As of the C7, this function returns Intel-style extended feature
         flags, not AMD-style.
        Currently, this only defines bits 11, 20, and 29 (0x20100800), which
         do not conflict with any of the AMD flags we inspect.
        For the remaining bits, Intel tells us, "Do not count on their value",
         but VIA assures us that they will all be zero (at least on the C7 and
         Isaiah chips).
        In the (unlikely) event a future processor uses bits 18, 19, 30, or 31
         (0xC0C00000) for something else, we will have to add code to detect
         the model to decide when it is appropriate to inspect them.*/
      flags|=oc_parse_amd_flags(edx,ecx);
    }
  }
  else{
    /*Implement me.*/
    flags=0;
  }
  return flags;
}
#endif
