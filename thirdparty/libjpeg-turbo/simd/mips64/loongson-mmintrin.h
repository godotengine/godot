/*
 * Loongson MMI optimizations for libjpeg-turbo
 *
 * Copyright (C) 2016-2018, Loongson Technology Corporation Limited, BeiJing.
 *                          All Rights Reserved.
 * Copyright (C) 2019, D. R. Commander.  All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef __LOONGSON_MMINTRIN_H__
#define __LOONGSON_MMINTRIN_H__

#include <stdint.h>


#define FUNCTION_ATTRIBS \
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))


/* Vectors are stored in 64-bit floating-point registers. */
typedef double __m64;

/* Having a 32-bit datatype allows us to use 32-bit loads in places like
   load8888. */
typedef float __m32;


/********** Set Operations **********/

extern __inline __m64 FUNCTION_ATTRIBS
_mm_setzero_si64(void)
{
  return 0.0;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_set_pi8(uint8_t __b7, uint8_t __b6, uint8_t __b5, uint8_t __b4,
            uint8_t __b3, uint8_t __b2, uint8_t __b1, uint8_t __b0)
{
  __m64 ret;
  uint32_t lo = ((uint32_t)__b6 << 24) |
                ((uint32_t)__b4 << 16) |
                ((uint32_t)__b2 << 8) |
                (uint32_t)__b0;
  uint32_t hi = ((uint32_t)__b7 << 24) |
                ((uint32_t)__b5 << 16) |
                ((uint32_t)__b3 << 8) |
                (uint32_t)__b1;

  asm("mtc1      %1, %0\n\t"
      "mtc1      %2, $f0\n\t"
      "punpcklbh %0, %0, $f0\n\t"
      : "=f" (ret)
      : "r" (lo), "r" (hi)
      : "$f0"
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_set_pi16(uint16_t __h3, uint16_t __h2, uint16_t __h1, uint16_t __h0)
{
  __m64 ret;
  uint32_t lo = ((uint32_t)__h2 << 16) | (uint32_t)__h0;
  uint32_t hi = ((uint32_t)__h3 << 16) | (uint32_t)__h1;

  asm("mtc1      %1, %0\n\t"
      "mtc1      %2, $f0\n\t"
      "punpcklhw %0, %0, $f0\n\t"
      : "=f" (ret)
      : "r" (lo), "r" (hi)
      : "$f0"
     );

  return ret;
}

#define _MM_SHUFFLE(fp3, fp2, fp1, fp0) \
  (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0))

extern __inline __m64 FUNCTION_ATTRIBS
_mm_set_pi32(uint32_t __i1, uint32_t __i0)
{
  if (__builtin_constant_p(__i1) && __builtin_constant_p(__i0)) {
    uint64_t val = ((uint64_t)__i1 << 32) |
                   ((uint64_t)__i0 <<  0);

    return *(__m64 *)&val;
  } else if (__i1 == __i0) {
    uint64_t imm = _MM_SHUFFLE(1, 0, 1, 0);
    __m64 ret;

    asm("pshufh %0, %1, %2\n\t"
        : "=f" (ret)
        : "f" (*(__m32 *)&__i1), "f" (*(__m64 *)&imm)
       );

    return ret;
  } else {
    uint64_t val = ((uint64_t)__i1 << 32) |
                   ((uint64_t)__i0 <<  0);

    return *(__m64 *)&val;
  }
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_set1_pi8(uint8_t __b0)
{
  __m64 ret;

  asm("sll    $8, %1, 8\n\t"
      "or     %1, %1, $8\n\t"
      "mtc1   %1, %0\n\t"
      "mtc1   $0, $f0\n\t"
      "pshufh %0, %0, $f0\n\t"
      : "=f" (ret)
      : "r" (__b0)
      : "$8", "$f0"
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_set1_pi16(uint16_t __h0)
{
  __m64 ret;

  asm("mtc1   %1, %0\n\t"
      "mtc1   $0, $f0\n\t"
      "pshufh %0, %0, $f0\n\t"
      : "=f" (ret)
      : "r" (__h0)
      : "$8", "$f0"
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_set1_pi32(unsigned __i0)
{
  return _mm_set_pi32(__i0, __i0);
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_setr_pi8(uint8_t __h0, uint8_t __h1, uint8_t __h2, uint8_t __h3,
             uint8_t __h4, uint8_t __h5, uint8_t __h6, uint8_t __h7)
{
  return _mm_set_pi8(__h7, __h6, __h5, __h4,
                     __h3, __h2, __h1, __h0);
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_setr_pi16(uint16_t __w0, uint16_t __w1, uint16_t __w2, uint16_t __w3)
{
  return _mm_set_pi16(__w3, __w2, __w1, __w0);
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_setr_pi32(uint32_t __i0, uint32_t __i1)
{
  return _mm_set_pi32(__i1, __i0);
}


/********** Arithmetic Operations **********/

extern __inline __m64 FUNCTION_ATTRIBS
_mm_add_pi8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("paddb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_add_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("paddh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_add_pi32(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("paddw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_add_si64(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("paddd %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_adds_pi8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("paddsb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_adds_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("paddsh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}


extern __inline __m64 FUNCTION_ATTRIBS
_mm_adds_pu8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("paddusb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_adds_pu16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("paddush %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_avg_pu8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pavgb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_avg_pu16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pavgh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_madd_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pmaddhw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_max_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pmaxsh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_max_pu8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pmaxub %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_min_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pminsh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_min_pu8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pminub %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline int FUNCTION_ATTRIBS
_mm_movemask_pi8(__m64 __m1)
{
  int ret;

  asm("pmovmskb %0, %1\n\t"
      : "=r" (ret)
      : "y" (__m1)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_mulhi_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pmulhh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_mulhi_pu16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pmulhuh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_mullo_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pmullh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_mul_pu32(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pmuluw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_sad_pu8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("psadbh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}


extern __inline __m64 FUNCTION_ATTRIBS
_mm_asub_pu8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pasubub %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_biadd_pu8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("biadd %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_sub_pi8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("psubb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_sub_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("psubh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_sub_pi32(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("psubw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_sub_si64(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("psubd %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_subs_pi8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("psubsb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_subs_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("psubsh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}


extern __inline __m64 FUNCTION_ATTRIBS
_mm_subs_pu8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("psubusb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_subs_pu16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("psubush %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}


/********** Logical Operations **********/

extern __inline __m64 FUNCTION_ATTRIBS
_mm_and_si64(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("and %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_andnot_si64(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("andn %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}


extern __inline __m64 FUNCTION_ATTRIBS
_mm_or_si32(__m32 __m1, __m32 __m2)
{
  __m32 ret;

  asm("or %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_or_si64(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("or %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_xor_si64(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("xor %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}


/********** Shift Operations **********/

extern __inline __m64 FUNCTION_ATTRIBS
_mm_slli_pi16(__m64 __m, int64_t __count)
{
  __m64 ret;

  asm("psllh  %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__count)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_slli_pi32(__m64 __m, int64_t __count)
{
  __m64 ret;

  asm("psllw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__count)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_slli_si64(__m64 __m, int64_t __count)
{
  __m64 ret;

  asm("dsll  %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__count)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_srli_pi16(__m64 __m, int64_t __count)
{
  __m64 ret;

  asm("psrlh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__count)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_srli_pi32(__m64 __m, int64_t __count)
{
  __m64 ret;

  asm("psrlw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__count)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_srli_si64(__m64 __m, int64_t __count)
{
  __m64 ret;

  asm("dsrl  %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__count)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_srai_pi16(__m64 __m, int64_t __count)
{
  __m64 ret;

  asm("psrah %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__count)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_srai_pi32(__m64 __m, int64_t __count)
{
  __m64 ret;

  asm("psraw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__count)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_srai_si64(__m64 __m, int64_t __count)
{
  __m64 ret;

  asm("dsra %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__count)
     );

  return ret;
}


/********** Conversion Intrinsics **********/

extern __inline __m64 FUNCTION_ATTRIBS
to_m64(uint64_t x)
{
  return *(__m64 *)&x;
}

extern __inline uint64_t FUNCTION_ATTRIBS
to_uint64(__m64 x)
{
  return *(uint64_t *)&x;
}


/********** Comparison Intrinsics **********/

extern __inline __m64 FUNCTION_ATTRIBS
_mm_cmpeq_pi8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pcmpeqb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_cmpeq_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pcmpeqh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_cmpeq_pi32(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pcmpeqw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_cmpgt_pi8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pcmpgtb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_cmpgt_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pcmpgth %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_cmpgt_pi32(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pcmpgtw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_cmplt_pi8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pcmpltb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_cmplt_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pcmplth %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_cmplt_pi32(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("pcmpltw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}


/********** Miscellaneous Operations **********/

extern __inline __m64 FUNCTION_ATTRIBS
_mm_packs_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("packsshb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_packs_pi32(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("packsswh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_packs_pi32_f(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("packsswh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_packs_pu16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("packushb %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_extract_pi16(__m64 __m, int64_t __pos)
{
  __m64 ret;

  asm("pextrh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__pos)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_insert_pi16(__m64 __m1, __m64 __m2, int64_t __pos)
{
  __m64 ret;

  switch (__pos) {
  case 0:

    asm("pinsrh_0 %0, %1, %2\n\t"
        : "=f" (ret)
        : "f" (__m1), "f" (__m2), "i" (__pos)
       );

    break;

  case 1:

    asm("pinsrh_1 %0, %1, %2\n\t"
        : "=f" (ret)
        : "f" (__m1), "f" (__m2), "i" (__pos)
       );

    break;
  case 2:

    asm("pinsrh_2 %0, %1, %2\n\t"
        : "=f" (ret)
        : "f" (__m1), "f" (__m2), "i" (__pos)
       );

    break;

  case 3:

    asm("pinsrh_3 %0, %1, %2\n\t"
        : "=f" (ret)
        : "f" (__m1), "f" (__m2), "i" (__pos)
       );

    break;
  }

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_shuffle_pi16(__m64 __m, int64_t __n)
{
  __m64 ret;

  asm("pshufh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m), "f" (*(__m64 *)&__n)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpackhi_pi8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpckhbh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpackhi_pi8_f(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpckhbh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpackhi_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpckhhw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpackhi_pi16_f(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpckhhw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpackhi_pi32(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpckhwd %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpacklo_pi8(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpcklbh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

/* Since punpcklbh cares about the high 32-bits, we use the __m64 datatype,
   which preserves the data. */

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpacklo_pi8_f64(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpcklbh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

/* Since punpcklbh doesn't care about the high 32-bits, we use the __m32,
   datatype, which allows load8888 to use 32-bit loads. */

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpacklo_pi8_f(__m32 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpcklbh %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpacklo_pi16(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpcklhw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpacklo_pi16_f(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpcklhw %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpacklo_pi32(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpcklwd %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}


extern __inline __m64 FUNCTION_ATTRIBS
_mm_unpacklo_pi32_f(__m64 __m1, __m64 __m2)
{
  __m64 ret;

  asm("punpcklwd %0, %1, %2\n\t"
      : "=f" (ret)
      : "f" (__m1), "f" (__m2)
     );

  return ret;
}

extern __inline void FUNCTION_ATTRIBS
_mm_store_pi32(__m32 *dest, __m64 src)
{
  src = _mm_packs_pu16(src, _mm_setzero_si64());

  asm("swc1 %1, %0\n\t"
      : "=m" (*dest)
      : "f" (src)
      : "memory"
     );
}

extern __inline void FUNCTION_ATTRIBS
_mm_store_si64(__m64 *dest, __m64 src)
{
  asm("sdc1 %1, %0 \n\t"
      : "=m" (*dest)
      : "f" (src)
      : "memory"
     );
}

extern __inline void FUNCTION_ATTRIBS
_mm_storeu_si64(__m64 *dest, __m64 src)
{
  asm("gssdlc1 %1, 7(%0) \n\t"
      "gssdrc1 %1, 0(%0) \n\t"
      :
      : "r" (dest), "f" (src)
      : "memory"
     );
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_load_si32(const __m32 *src)
{
  __m32 ret;

  asm("lwc1 %0, %1\n\t"
      : "=f" (ret)
      : "m" (*src)
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_load_si64(const __m64 *src)
{
  __m64 ret;

  asm("ldc1 %0, %1\n\t"
      : "=f" (ret)
      : "m" (*src)
      : "memory"
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_loadu_si64(const __m64 *src)
{
  __m64 ret;

  asm("gsldlc1 %0,  7(%1)\n\t"
      "gsldrc1 %0,  0(%1)\n\t"
      : "=f" (ret)
      : "r" (src)
      : "memory"
     );

  return ret;
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_loadlo_pi8(const uint32_t *src)
{
  return _mm_unpacklo_pi8_f(*(__m32 *)src, _mm_setzero_si64());
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_loadlo_pi8_f(__m64 src)
{
  return _mm_unpacklo_pi8_f64(src, _mm_setzero_si64());
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_loadhi_pi8_f(__m64 src)
{
  return _mm_unpackhi_pi8_f(src, _mm_setzero_si64());
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_loadlo_pi16(__m64 src)
{
  return _mm_unpacklo_pi16(src, _mm_setzero_si64());
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_loadlo_pi16_f(__m64 src)
{
  return _mm_unpacklo_pi16_f(_mm_setzero_si64(), src);
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_loadhi_pi16(__m64 src)
{
  return _mm_unpackhi_pi16(src, _mm_setzero_si64());
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_loadhi_pi16_f(__m64 src)
{
  return _mm_unpackhi_pi16_f(_mm_setzero_si64(), src);
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_expand_alpha(__m64 pixel)
{
  return _mm_shuffle_pi16(pixel, _MM_SHUFFLE(3, 3, 3, 3));
}

extern __inline __m64 FUNCTION_ATTRIBS
_mm_expand_alpha_rev(__m64 pixel)
{
  return _mm_shuffle_pi16(pixel, _MM_SHUFFLE(0, 0, 0, 0));
}

#endif  /* __LOONGSON_MMINTRIN_H__ */
