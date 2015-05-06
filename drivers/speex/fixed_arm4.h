/* Copyright (C) 2004 Jean-Marc Valin */
/**
   @file fixed_arm4.h
   @brief ARM4 fixed-point operations
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef FIXED_ARM4_H
#define FIXED_ARM4_H

#undef MULT16_32_Q14
static inline spx_word32_t MULT16_32_Q14(spx_word16_t x, spx_word32_t y) {
  int res;
  int dummy;
  asm (
        "smull  %0,%1,%2,%3 \n\t"
        "mov %0, %0, lsr #14 \n\t"
        "add %0, %0, %1, lsl #18 \n\t"
   : "=&r"(res), "=&r" (dummy)
   : "r"(y),"r"((int)x));
  return(res);
}

#undef MULT16_32_Q15
static inline spx_word32_t MULT16_32_Q15(spx_word16_t x, spx_word32_t y) {
  int res;
  int dummy;
  asm (
        "smull  %0,%1,%2,%3 \n\t"
        "mov %0, %0, lsr #15 \n\t"
        "add %0, %0, %1, lsl #17 \n\t"
   : "=&r"(res), "=&r" (dummy)
   : "r"(y),"r"((int)x));
  return(res);
}

#undef DIV32_16
static inline short DIV32_16(int a, int b)
{
   int res=0;
   int dead1, dead2, dead3, dead4, dead5;
   __asm__ __volatile__ (
         "\teor %5, %0, %1\n"
         "\tmovs %4, %0\n"
         "\trsbmi %0, %0, #0 \n"
         "\tmovs %4, %1\n"
         "\trsbmi %1, %1, #0 \n"
         "\tmov %4, #1\n"

         "\tsubs %3, %0, %1, asl #14 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #14 \n"

         "\tsubs %3, %0, %1, asl #13 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #13 \n"

         "\tsubs %3, %0, %1, asl #12 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #12 \n"

         "\tsubs %3, %0, %1, asl #11 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #11 \n"

         "\tsubs %3, %0, %1, asl #10 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #10 \n"

         "\tsubs %3, %0, %1, asl #9 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #9 \n"

         "\tsubs %3, %0, %1, asl #8 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #8 \n"

         "\tsubs %3, %0, %1, asl #7 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #7 \n"

         "\tsubs %3, %0, %1, asl #6 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #6 \n"
         
         "\tsubs %3, %0, %1, asl #5 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #5 \n"

         "\tsubs %3, %0, %1, asl #4 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #4 \n"

         "\tsubs %3, %0, %1, asl #3 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #3 \n"

         "\tsubs %3, %0, %1, asl #2 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #2 \n"

         "\tsubs %3, %0, %1, asl #1 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4, asl #1 \n"

         "\tsubs %3, %0, %1 \n"
         "\tmovpl %0, %3 \n"
         "\torrpl %2, %2, %4 \n"

         "\tmovs %5, %5, lsr #31 \n"
         "\trsbne %2, %2, #0 \n"
   : "=r" (dead1), "=r" (dead2), "=r" (res),
   "=r" (dead3), "=r" (dead4), "=r" (dead5)
   : "0" (a), "1" (b), "2" (res)
   : "cc"
                        );
   return res;
}


#endif
