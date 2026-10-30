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

#if defined (__mips_dsp) && __mips == 32

typedef short v2i16 __attribute__((vector_size(4)));
typedef char  v2i8  __attribute__((vector_size(4)));

#undef MULT16_32_Q16
static inline int MULT16_32_Q16(int a, int b)
{
    long long acc = __builtin_mips_mult(a, b);
    return __builtin_mips_extr_w(acc, 16);
}

#undef MULT16_32_P16
static inline int MULT16_32_P16(int a, int b)
{
    long long acc = __builtin_mips_mult(a, b);
    return __builtin_mips_extr_r_w(acc, 16);
}

#undef MULT16_32_Q15
static inline int MULT16_32_Q15(int a, int b)
{
    long long acc = __builtin_mips_mult(a, b);
    return __builtin_mips_extr_w(acc, 15);
}

#undef MULT32_32_Q31
static inline int MULT32_32_Q31(int a, int b)
{
    long long acc = __builtin_mips_mult(a, b);
    return __builtin_mips_extr_w(acc, 31);
}

#undef PSHR32
static inline int PSHR32(int a, int shift)
{
    return __builtin_mips_shra_r_w(a, shift);
}

#undef MULT16_16_P15
static inline int MULT16_16_P15(int a, int b)
{
    int r = a * b;
    return __builtin_mips_shra_r_w(r, 15);
}

#define OVERRIDE_CELT_MAXABS16
static OPUS_INLINE opus_val32 celt_maxabs16(const opus_val16 *x, int len)
{
   int i;
   v2i16 v2max = (v2i16){ 0, 0 };
   v2i16 x01, x23;
   const v2i16 *x2;
   opus_val16 maxlo, maxhi;
   int loops;

   if ((long)x & 2 && len > 0) {
      v2max = (v2i16){ 0, ABS16(*x) };
      x++;
      len--;
   }
   x2 = __builtin_assume_aligned(x, 4);
   loops = len / 4;

   for (i = 0; i < loops; i++)
   {
       x01 = *x2++;
       x23 = *x2++;
       x01 = __builtin_mips_absq_s_ph(x01);
       x23 = __builtin_mips_absq_s_ph(x23);
       __builtin_mips_cmp_lt_ph(v2max, x01);
       v2max = __builtin_mips_pick_ph(x01, v2max);
       __builtin_mips_cmp_lt_ph(v2max, x23);
       v2max = __builtin_mips_pick_ph(x23, v2max);
   }

   switch (len & 3) {
   case 3:
       x01 = __builtin_mips_absq_s_ph(*x2);
       __builtin_mips_cmp_lt_ph(v2max, x01);
       v2max = __builtin_mips_pick_ph(x01, v2max);
       maxlo = EXTRACT16((opus_val32)v2max);
       maxhi = EXTRACT16((opus_val32)v2max >> 16);
       maxlo = MAX16(MAX16(maxlo, maxhi), ABS16(x[len - 1]));
       break;
   case 2:
       x01 = __builtin_mips_absq_s_ph(*x2);
       __builtin_mips_cmp_lt_ph(v2max, x01);
       v2max = __builtin_mips_pick_ph(x01, v2max);
       maxlo = EXTRACT16((opus_val32)v2max);
       maxhi = EXTRACT16((opus_val32)v2max >> 16);
       maxlo = MAX16(maxlo, maxhi);
       break;
   case 1:
       maxlo = EXTRACT16((opus_val32)v2max);
       maxhi = EXTRACT16((opus_val32)v2max >> 16);
       return MAX16(MAX16(maxlo, maxhi), ABS16(x[len - 1]));
       break;
   case 0:
       maxlo = EXTRACT16((opus_val32)v2max);
       maxhi = EXTRACT16((opus_val32)v2max >> 16);
       maxlo = MAX16(maxlo, maxhi);
       break;
   default:
       __builtin_unreachable();
   }
   /* C version might return 0x8000, this one can't
    * because abs is saturated here. Since result
    * used only for determine dynamic range
    * in ilog2-like context it's worth to add 1
    * for proper magnitude whether saturated
    */
   return (opus_val32)maxlo + 1;
}

#elif __mips == 32

#undef MULT16_32_Q16
#define MULT16_32_Q16(a,b) ((opus_val32)SHR((opus_int64)(SHL32((a), 16))*(b),32))

#endif

#endif /* CELT_FIXED_GENERIC_MIPSR1_H */
