/*Copyright (c) 2013, Xiph.Org Foundation and contributors.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.*/

#ifndef KISS_FFT_ARMv4_H
#define KISS_FFT_ARMv4_H

#if !defined(KISS_FFT_GUTS_H)
#error "This file should only be included from _kiss_fft_guts.h"
#endif

#ifdef FIXED_POINT

#undef C_MUL
#define C_MUL(m,a,b) \
    do{ \
       int br__; \
       int bi__; \
       int tt__; \
        __asm__ __volatile__( \
            "#C_MUL\n\t" \
            "ldrsh %[br], [%[bp], #0]\n\t" \
            "ldm %[ap], {r0,r1}\n\t" \
            "ldrsh %[bi], [%[bp], #2]\n\t" \
            "smull %[tt], %[mi], r1, %[br]\n\t" \
            "smlal %[tt], %[mi], r0, %[bi]\n\t" \
            "rsb %[bi], %[bi], #0\n\t" \
            "smull %[br], %[mr], r0, %[br]\n\t" \
            "mov %[tt], %[tt], lsr #15\n\t" \
            "smlal %[br], %[mr], r1, %[bi]\n\t" \
            "orr %[mi], %[tt], %[mi], lsl #17\n\t" \
            "mov %[br], %[br], lsr #15\n\t" \
            "orr %[mr], %[br], %[mr], lsl #17\n\t" \
            : [mr]"=r"((m).r), [mi]"=r"((m).i), \
              [br]"=&r"(br__), [bi]"=r"(bi__), [tt]"=r"(tt__) \
            : [ap]"r"(&(a)), [bp]"r"(&(b)) \
            : "r0", "r1" \
        ); \
    } \
    while(0)

#undef C_MUL4
#define C_MUL4(m,a,b) \
    do{ \
       int br__; \
       int bi__; \
       int tt__; \
        __asm__ __volatile__( \
            "#C_MUL4\n\t" \
            "ldrsh %[br], [%[bp], #0]\n\t" \
            "ldm %[ap], {r0,r1}\n\t" \
            "ldrsh %[bi], [%[bp], #2]\n\t" \
            "smull %[tt], %[mi], r1, %[br]\n\t" \
            "smlal %[tt], %[mi], r0, %[bi]\n\t" \
            "rsb %[bi], %[bi], #0\n\t" \
            "smull %[br], %[mr], r0, %[br]\n\t" \
            "mov %[tt], %[tt], lsr #17\n\t" \
            "smlal %[br], %[mr], r1, %[bi]\n\t" \
            "orr %[mi], %[tt], %[mi], lsl #15\n\t" \
            "mov %[br], %[br], lsr #17\n\t" \
            "orr %[mr], %[br], %[mr], lsl #15\n\t" \
            : [mr]"=r"((m).r), [mi]"=r"((m).i), \
              [br]"=&r"(br__), [bi]"=r"(bi__), [tt]"=r"(tt__) \
            : [ap]"r"(&(a)), [bp]"r"(&(b)) \
            : "r0", "r1" \
        ); \
    } \
    while(0)

#undef C_MULC
#define C_MULC(m,a,b) \
    do{ \
       int br__; \
       int bi__; \
       int tt__; \
        __asm__ __volatile__( \
            "#C_MULC\n\t" \
            "ldrsh %[br], [%[bp], #0]\n\t" \
            "ldm %[ap], {r0,r1}\n\t" \
            "ldrsh %[bi], [%[bp], #2]\n\t" \
            "smull %[tt], %[mr], r0, %[br]\n\t" \
            "smlal %[tt], %[mr], r1, %[bi]\n\t" \
            "rsb %[bi], %[bi], #0\n\t" \
            "smull %[br], %[mi], r1, %[br]\n\t" \
            "mov %[tt], %[tt], lsr #15\n\t" \
            "smlal %[br], %[mi], r0, %[bi]\n\t" \
            "orr %[mr], %[tt], %[mr], lsl #17\n\t" \
            "mov %[br], %[br], lsr #15\n\t" \
            "orr %[mi], %[br], %[mi], lsl #17\n\t" \
            : [mr]"=r"((m).r), [mi]"=r"((m).i), \
              [br]"=&r"(br__), [bi]"=r"(bi__), [tt]"=r"(tt__) \
            : [ap]"r"(&(a)), [bp]"r"(&(b)) \
            : "r0", "r1" \
        ); \
    } \
    while(0)

#endif /* FIXED_POINT */

#endif /* KISS_FFT_ARMv4_H */
