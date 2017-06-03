/* Copyright (C) 2007-2009 Xiph.Org Foundation
   Copyright (C) 2003-2008 Jean-Marc Valin
   Copyright (C) 2007-2008 CSIRO */
/**
   @file fixed_generic.h
   @brief Generic fixed-point operations
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

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CELT_FIXED_GENERIC_MIPSR1_H
#define CELT_FIXED_GENERIC_MIPSR1_H

#undef MULT16_32_Q15_ADD
static inline int MULT16_32_Q15_ADD(int a, int b, int c, int d) {
    int m;
    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a), "r" ((int)b));
    asm volatile("madd $ac1, %0, %1" : : "r" ((int)c), "r" ((int)d));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m): "i" (15));
    return m;
}

#undef MULT16_32_Q15_SUB
static inline int MULT16_32_Q15_SUB(int a, int b, int c, int d) {
    int m;
    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a), "r" ((int)b));
    asm volatile("msub $ac1, %0, %1" : : "r" ((int)c), "r" ((int)d));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m): "i" (15));
    return m;
}

#undef MULT16_16_Q15_ADD
static inline int MULT16_16_Q15_ADD(int a, int b, int c, int d) {
    int m;
    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a), "r" ((int)b));
    asm volatile("madd $ac1, %0, %1" : : "r" ((int)c), "r" ((int)d));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m): "i" (15));
    return m;
}

#undef MULT16_16_Q15_SUB
static inline int MULT16_16_Q15_SUB(int a, int b, int c, int d) {
    int m;
    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a), "r" ((int)b));
    asm volatile("msub $ac1, %0, %1" : : "r" ((int)c), "r" ((int)d));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m): "i" (15));
    return m;
}


#undef MULT16_32_Q16
static inline int MULT16_32_Q16(int a, int b)
{
    int c;
    asm volatile("MULT $ac1,%0, %1" : : "r" (a), "r" (b));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (c): "i" (16));
    return c;
}

#undef MULT16_32_P16
static inline int MULT16_32_P16(int a, int b)
{
    int c;
    asm volatile("MULT $ac1, %0, %1" : : "r" (a), "r" (b));
    asm volatile("EXTR_R.W %0,$ac1, %1" : "=r" (c): "i" (16));
    return c;
}

#undef MULT16_32_Q15
static inline int MULT16_32_Q15(int a, int b)
{
    int c;
    asm volatile("MULT $ac1, %0, %1" : : "r" (a), "r" (b));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (c): "i" (15));
    return c;
}

#undef MULT32_32_Q31
static inline int MULT32_32_Q31(int a, int b)
{
    int r;
    asm volatile("MULT $ac1, %0, %1" : : "r" (a), "r" (b));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (r): "i" (31));
    return r;
}

#undef PSHR32
static inline int PSHR32(int a, int shift)
{
    int r;
    asm volatile ("SHRAV_R.W %0, %1, %2" :"=r" (r): "r" (a), "r" (shift));
    return r;
}

#undef MULT16_16_P15
static inline int MULT16_16_P15(int a, int b)
{
    int r;
    asm volatile ("mul %0, %1, %2" :"=r" (r): "r" (a), "r" (b));
    asm volatile ("SHRA_R.W %0, %1, %2" : "+r" (r):  "0" (r), "i"(15));
    return r;
}

#endif /* CELT_FIXED_GENERIC_MIPSR1_H */
