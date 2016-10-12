/* Copyright (C) 2005 Analog Devices
   Author: Jean-Marc Valin */
/**
   @file fixed_bfin.h
   @brief Blackfin fixed-point operations
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

#ifndef FIXED_BFIN_H
#define FIXED_BFIN_H

#undef PDIV32_16
static inline spx_word16_t PDIV32_16(spx_word32_t a, spx_word16_t b)
{
   spx_word32_t res, bb;
   bb = b;
   a += b>>1;
   __asm__  (
         "P0 = 15;\n\t"
         "R0 = %1;\n\t"
         "R1 = %2;\n\t"
         //"R0 = R0 + R1;\n\t"
         "R0 <<= 1;\n\t"
         "DIVS (R0, R1);\n\t"
         "LOOP divide%= LC0 = P0;\n\t"
         "LOOP_BEGIN divide%=;\n\t"
            "DIVQ (R0, R1);\n\t"
         "LOOP_END divide%=;\n\t"
         "R0 = R0.L;\n\t"
         "%0 = R0;\n\t"
   : "=m" (res)
   : "m" (a), "m" (bb)
   : "P0", "R0", "R1", "cc");
   return res;
}

#undef DIV32_16
static inline spx_word16_t DIV32_16(spx_word32_t a, spx_word16_t b)
{
   spx_word32_t res, bb;
   bb = b;
   /* Make the roundinf consistent with the C version 
      (do we need to do that?)*/
   if (a<0) 
      a += (b-1);
   __asm__  (
         "P0 = 15;\n\t"
         "R0 = %1;\n\t"
         "R1 = %2;\n\t"
         "R0 <<= 1;\n\t"
         "DIVS (R0, R1);\n\t"
         "LOOP divide%= LC0 = P0;\n\t"
         "LOOP_BEGIN divide%=;\n\t"
            "DIVQ (R0, R1);\n\t"
         "LOOP_END divide%=;\n\t"
         "R0 = R0.L;\n\t"
         "%0 = R0;\n\t"
   : "=m" (res)
   : "m" (a), "m" (bb)
   : "P0", "R0", "R1", "cc");
   return res;
}

#undef MAX16
static inline spx_word16_t MAX16(spx_word16_t a, spx_word16_t b)
{
   spx_word32_t res;
   __asm__  (
         "%1 = %1.L (X);\n\t"
         "%2 = %2.L (X);\n\t"
         "%0 = MAX(%1,%2);"
   : "=d" (res)
   : "%d" (a), "d" (b)
   );
   return res;
}

#undef MULT16_32_Q15
static inline spx_word32_t MULT16_32_Q15(spx_word16_t a, spx_word32_t b)
{
   spx_word32_t res;
   __asm__
   (
         "A1 = %2.L*%1.L (M);\n\t"
         "A1 = A1 >>> 15;\n\t"
         "%0 = (A1 += %2.L*%1.H) ;\n\t"
   : "=&W" (res), "=&d" (b)
   : "d" (a), "1" (b)
   : "A1"
   );
   return res;
}

#undef MAC16_32_Q15
static inline spx_word32_t MAC16_32_Q15(spx_word32_t c, spx_word16_t a, spx_word32_t b)
{
   spx_word32_t res;
   __asm__
         (
         "A1 = %2.L*%1.L (M);\n\t"
         "A1 = A1 >>> 15;\n\t"
         "%0 = (A1 += %2.L*%1.H);\n\t"
         "%0 = %0 + %4;\n\t"
   : "=&W" (res), "=&d" (b)
   : "d" (a), "1" (b), "d" (c)
   : "A1"
         );
   return res;
}

#undef MULT16_32_Q14
static inline spx_word32_t MULT16_32_Q14(spx_word16_t a, spx_word32_t b)
{
   spx_word32_t res;
   __asm__
         (
         "%2 <<= 1;\n\t"
         "A1 = %1.L*%2.L (M);\n\t"
         "A1 = A1 >>> 15;\n\t"
         "%0 = (A1 += %1.L*%2.H);\n\t"
   : "=W" (res), "=d" (a), "=d" (b)
   : "1" (a), "2" (b)
   : "A1"
         );
   return res;
}

#undef MAC16_32_Q14
static inline spx_word32_t MAC16_32_Q14(spx_word32_t c, spx_word16_t a, spx_word32_t b)
{
   spx_word32_t res;
   __asm__
         (
         "%1 <<= 1;\n\t"
         "A1 = %2.L*%1.L (M);\n\t"
         "A1 = A1 >>> 15;\n\t"
         "%0 = (A1 += %2.L*%1.H);\n\t"
         "%0 = %0 + %4;\n\t"
   : "=&W" (res), "=&d" (b)
   : "d" (a), "1" (b), "d" (c)
   : "A1"
         );
   return res;
}

#endif
