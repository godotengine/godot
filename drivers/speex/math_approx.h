/* Copyright (C) 2002 Jean-Marc Valin */
/**
   @file math_approx.h
   @brief Various math approximation functions for Speex
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

#ifndef MATH_APPROX_H
#define MATH_APPROX_H

#include "arch.h"
#include "os_support.h"

#ifndef FIXED_POINT

#define spx_sqrt sqrt
#define spx_acos acos
#define spx_exp exp
#define spx_cos_norm(x) (cos((.5f*M_PI)*(x)))
#define spx_atan atan

/** Generate a pseudo-random number */
static SPEEX_INLINE spx_word16_t speex_rand(spx_word16_t std, spx_int32_t *seed)
{
   const unsigned int jflone = 0x3f800000;
   const unsigned int jflmsk = 0x007fffff;
   union {int i; float f;} ran;
   *seed = 1664525 * *seed + 1013904223;
   ran.i = jflone | (jflmsk & *seed);
   ran.f -= 1.5;
   return 3.4642*std*ran.f;
}


#endif


static SPEEX_INLINE spx_int16_t spx_ilog2(spx_uint32_t x)
{
   int r=0;
   if (x>=(spx_int32_t)65536)
   {
      x >>= 16;
      r += 16;
   }
   if (x>=256)
   {
      x >>= 8;
      r += 8;
   }
   if (x>=16)
   {
      x >>= 4;
      r += 4;
   }
   if (x>=4)
   {
      x >>= 2;
      r += 2;
   }
   if (x>=2)
   {
      r += 1;
   }
   return r;
}

static SPEEX_INLINE spx_int16_t spx_ilog4(spx_uint32_t x)
{
   int r=0;
   if (x>=(spx_int32_t)65536)
   {
      x >>= 16;
      r += 8;
   }
   if (x>=256)
   {
      x >>= 8;
      r += 4;
   }
   if (x>=16)
   {
      x >>= 4;
      r += 2;
   }
   if (x>=4)
   {
      r += 1;
   }
   return r;
}

#ifdef FIXED_POINT

/** Generate a pseudo-random number */
static SPEEX_INLINE spx_word16_t speex_rand(spx_word16_t std, spx_int32_t *seed)
{
   spx_word32_t res;
   *seed = 1664525 * *seed + 1013904223;
   res = MULT16_16(EXTRACT16(SHR32(*seed,16)),std);
   return EXTRACT16(PSHR32(SUB32(res, SHR32(res, 3)),14));
}

/* sqrt(x) ~= 0.22178 + 1.29227*x - 0.77070*x^2 + 0.25723*x^3 (for .25 < x < 1) */
/*#define C0 3634
#define C1 21173
#define C2 -12627
#define C3 4215*/

/* sqrt(x) ~= 0.22178 + 1.29227*x - 0.77070*x^2 + 0.25659*x^3 (for .25 < x < 1) */
#define C0 3634
#define C1 21173
#define C2 -12627
#define C3 4204

static SPEEX_INLINE spx_word16_t spx_sqrt(spx_word32_t x)
{
   int k;
   spx_word32_t rt;
   k = spx_ilog4(x)-6;
   x = VSHR32(x, (k<<1));
   rt = ADD16(C0, MULT16_16_Q14(x, ADD16(C1, MULT16_16_Q14(x, ADD16(C2, MULT16_16_Q14(x, (C3)))))));
   rt = VSHR32(rt,7-k);
   return rt;
}

/* log(x) ~= -2.18151 + 4.20592*x - 2.88938*x^2 + 0.86535*x^3 (for .5 < x < 1) */


#define A1 16469
#define A2 2242
#define A3 1486

static SPEEX_INLINE spx_word16_t spx_acos(spx_word16_t x)
{
   int s=0;
   spx_word16_t ret;
   spx_word16_t sq;
   if (x<0)
   {
      s=1;
      x = NEG16(x);
   }
   x = SUB16(16384,x);
   
   x = x >> 1;
   sq = MULT16_16_Q13(x, ADD16(A1, MULT16_16_Q13(x, ADD16(A2, MULT16_16_Q13(x, (A3))))));
   ret = spx_sqrt(SHL32(EXTEND32(sq),13));
   
   /*ret = spx_sqrt(67108864*(-1.6129e-04 + 2.0104e+00*f + 2.7373e-01*f*f + 1.8136e-01*f*f*f));*/
   if (s)
      ret = SUB16(25736,ret);
   return ret;
}


#define K1 8192
#define K2 -4096
#define K3 340
#define K4 -10

static SPEEX_INLINE spx_word16_t spx_cos(spx_word16_t x)
{
   spx_word16_t x2;

   if (x<12868)
   {
      x2 = MULT16_16_P13(x,x);
      return ADD32(K1, MULT16_16_P13(x2, ADD32(K2, MULT16_16_P13(x2, ADD32(K3, MULT16_16_P13(K4, x2))))));
   } else {
      x = SUB16(25736,x);
      x2 = MULT16_16_P13(x,x);
      return SUB32(-K1, MULT16_16_P13(x2, ADD32(K2, MULT16_16_P13(x2, ADD32(K3, MULT16_16_P13(K4, x2))))));
   }
}

#define L1 32767
#define L2 -7651
#define L3 8277
#define L4 -626

static SPEEX_INLINE spx_word16_t _spx_cos_pi_2(spx_word16_t x)
{
   spx_word16_t x2;
   
   x2 = MULT16_16_P15(x,x);
   return ADD16(1,MIN16(32766,ADD32(SUB16(L1,x2), MULT16_16_P15(x2, ADD32(L2, MULT16_16_P15(x2, ADD32(L3, MULT16_16_P15(L4, x2))))))));
}

static SPEEX_INLINE spx_word16_t spx_cos_norm(spx_word32_t x)
{
   x = x&0x0001ffff;
   if (x>SHL32(EXTEND32(1), 16))
      x = SUB32(SHL32(EXTEND32(1), 17),x);
   if (x&0x00007fff)
   {
      if (x<SHL32(EXTEND32(1), 15))
      {
         return _spx_cos_pi_2(EXTRACT16(x));
      } else {
         return NEG32(_spx_cos_pi_2(EXTRACT16(65536-x)));
      }
   } else {
      if (x&0x0000ffff)
         return 0;
      else if (x&0x0001ffff)
         return -32767;
      else
         return 32767;
   }
}

/*
 K0 = 1
 K1 = log(2)
 K2 = 3-4*log(2)
 K3 = 3*log(2) - 2
*/
#define D0 16384
#define D1 11356
#define D2 3726
#define D3 1301
/* Input in Q11 format, output in Q16 */
static SPEEX_INLINE spx_word32_t spx_exp2(spx_word16_t x)
{
   int integer;
   spx_word16_t frac;
   integer = SHR16(x,11);
   if (integer>14)
      return 0x7fffffff;
   else if (integer < -15)
      return 0;
   frac = SHL16(x-SHL16(integer,11),3);
   frac = ADD16(D0, MULT16_16_Q14(frac, ADD16(D1, MULT16_16_Q14(frac, ADD16(D2 , MULT16_16_Q14(D3,frac))))));
   return VSHR32(EXTEND32(frac), -integer-2);
}

/* Input in Q11 format, output in Q16 */
static SPEEX_INLINE spx_word32_t spx_exp(spx_word16_t x)
{
   if (x>21290)
      return 0x7fffffff;
   else if (x<-21290)
      return 0;
   else
      return spx_exp2(MULT16_16_P14(23637,x));
}
#define M1 32767
#define M2 -21
#define M3 -11943
#define M4 4936

static SPEEX_INLINE spx_word16_t spx_atan01(spx_word16_t x)
{
   return MULT16_16_P15(x, ADD32(M1, MULT16_16_P15(x, ADD32(M2, MULT16_16_P15(x, ADD32(M3, MULT16_16_P15(M4, x)))))));
}

#undef M1
#undef M2
#undef M3
#undef M4

/* Input in Q15, output in Q14 */
static SPEEX_INLINE spx_word16_t spx_atan(spx_word32_t x)
{
   if (x <= 32767)
   {
      return SHR16(spx_atan01(x),1);
   } else {
      int e = spx_ilog2(x);
      if (e>=29)
         return 25736;
      x = DIV32_16(SHL32(EXTEND32(32767),29-e), EXTRACT16(SHR32(x, e-14)));
      return SUB16(25736, SHR16(spx_atan01(x),1));
   }
}
#else

#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

#define C1 0.9999932946f
#define C2 -0.4999124376f
#define C3 0.0414877472f
#define C4 -0.0012712095f


#define SPX_PI_2 1.5707963268
static SPEEX_INLINE spx_word16_t spx_cos(spx_word16_t x)
{
   if (x<SPX_PI_2)
   {
      x *= x;
      return C1 + x*(C2+x*(C3+C4*x));
   } else {
      x = M_PI-x;
      x *= x;
      return NEG16(C1 + x*(C2+x*(C3+C4*x)));
   }
}

#endif


#endif
