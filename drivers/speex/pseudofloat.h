/* Copyright (C) 2005 Jean-Marc Valin */
/**
   @file pseudofloat.h
   @brief Pseudo-floating point
 * This header file provides a lightweight floating point type for
 * use on fixed-point platforms when a large dynamic range is 
 * required. The new type is not compatible with the 32-bit IEEE format,
 * it is not even remotely as accurate as 32-bit floats, and is not
 * even guaranteed to produce even remotely correct results for code
 * other than Speex. It makes all kinds of shortcuts that are acceptable
 * for Speex, but may not be acceptable for your application. You're
 * quite welcome to reuse this code and improve it, but don't assume
 * it works out of the box. Most likely, it doesn't.
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

#ifndef PSEUDOFLOAT_H
#define PSEUDOFLOAT_H

#include "arch.h"
#include "os_support.h"
#include "math_approx.h"
#include <math.h>

#ifdef FIXED_POINT

typedef struct {
   spx_int16_t m;
   spx_int16_t e;
} spx_float_t;

static const spx_float_t FLOAT_ZERO = {0,0};
static const spx_float_t FLOAT_ONE = {16384,-14};
static const spx_float_t FLOAT_HALF = {16384,-15};

#define MIN(a,b) ((a)<(b)?(a):(b))
static inline spx_float_t PSEUDOFLOAT(spx_int32_t x)
{
   int e=0;
   int sign=0;
   if (x<0)
   {
      sign = 1;
      x = -x;
   }
   if (x==0)
   {
      spx_float_t r = {0,0};
      return r;
   }
   e = spx_ilog2(ABS32(x))-14;
   x = VSHR32(x, e);
   if (sign)
   {
      spx_float_t r;
      r.m = -x;
      r.e = e;
      return r;
   }
   else      
   {
      spx_float_t r;
      r.m = x;
      r.e = e;
      return r;
   }
}


static inline spx_float_t FLOAT_ADD(spx_float_t a, spx_float_t b)
{
   spx_float_t r;
   if (a.m==0)
      return b;
   else if (b.m==0)
      return a;
   if ((a).e > (b).e) 
   {
      r.m = ((a).m>>1) + ((b).m>>MIN(15,(a).e-(b).e+1));
      r.e = (a).e+1;
   }
   else 
   {
      r.m = ((b).m>>1) + ((a).m>>MIN(15,(b).e-(a).e+1));
      r.e = (b).e+1;
   }
   if (r.m>0)
   {
      if (r.m<16384)
      {
         r.m<<=1;
         r.e-=1;
      }
   } else {
      if (r.m>-16384)
      {
         r.m<<=1;
         r.e-=1;
      }
   }
   /*printf ("%f + %f = %f\n", REALFLOAT(a), REALFLOAT(b), REALFLOAT(r));*/
   return r;
}

static inline spx_float_t FLOAT_SUB(spx_float_t a, spx_float_t b)
{
   spx_float_t r;
   if (a.m==0)
      return b;
   else if (b.m==0)
      return a;
   if ((a).e > (b).e)
   {
      r.m = ((a).m>>1) - ((b).m>>MIN(15,(a).e-(b).e+1));
      r.e = (a).e+1;
   }
   else 
   {
      r.m = ((a).m>>MIN(15,(b).e-(a).e+1)) - ((b).m>>1);
      r.e = (b).e+1;
   }
   if (r.m>0)
   {
      if (r.m<16384)
      {
         r.m<<=1;
         r.e-=1;
      }
   } else {
      if (r.m>-16384)
      {
         r.m<<=1;
         r.e-=1;
      }
   }
   /*printf ("%f + %f = %f\n", REALFLOAT(a), REALFLOAT(b), REALFLOAT(r));*/
   return r;
}

static inline int FLOAT_LT(spx_float_t a, spx_float_t b)
{
   if (a.m==0)
      return b.m>0;
   else if (b.m==0)
      return a.m<0;   
   if ((a).e > (b).e)
      return ((a).m>>1) < ((b).m>>MIN(15,(a).e-(b).e+1));
   else 
      return ((b).m>>1) > ((a).m>>MIN(15,(b).e-(a).e+1));

}

static inline int FLOAT_GT(spx_float_t a, spx_float_t b)
{
   return FLOAT_LT(b,a);
}

static inline spx_float_t FLOAT_MULT(spx_float_t a, spx_float_t b)
{
   spx_float_t r;
   r.m = (spx_int16_t)((spx_int32_t)(a).m*(b).m>>15);
   r.e = (a).e+(b).e+15;
   if (r.m>0)
   {
      if (r.m<16384)
      {
         r.m<<=1;
         r.e-=1;
      }
   } else {
      if (r.m>-16384)
      {
         r.m<<=1;
         r.e-=1;
      }
   }
   /*printf ("%f * %f = %f\n", REALFLOAT(a), REALFLOAT(b), REALFLOAT(r));*/
   return r;   
}

static inline spx_float_t FLOAT_AMULT(spx_float_t a, spx_float_t b)
{
   spx_float_t r;
   r.m = (spx_int16_t)((spx_int32_t)(a).m*(b).m>>15);
   r.e = (a).e+(b).e+15;
   return r;   
}


static inline spx_float_t FLOAT_SHL(spx_float_t a, int b)
{
   spx_float_t r;
   r.m = a.m;
   r.e = a.e+b;
   return r;
}

static inline spx_int16_t FLOAT_EXTRACT16(spx_float_t a)
{
   if (a.e<0)
      return EXTRACT16((EXTEND32(a.m)+(EXTEND32(1)<<(-a.e-1)))>>-a.e);
   else
      return a.m<<a.e;
}

static inline spx_int32_t FLOAT_EXTRACT32(spx_float_t a)
{
   if (a.e<0)
      return (EXTEND32(a.m)+(EXTEND32(1)<<(-a.e-1)))>>-a.e;
   else
      return EXTEND32(a.m)<<a.e;
}

static inline spx_int32_t FLOAT_MUL32(spx_float_t a, spx_word32_t b)
{
   return VSHR32(MULT16_32_Q15(a.m, b),-a.e-15);
}

static inline spx_float_t FLOAT_MUL32U(spx_word32_t a, spx_word32_t b)
{
   int e1, e2;
   spx_float_t r;
   if (a==0 || b==0)
   {
      return FLOAT_ZERO;
   }
   e1 = spx_ilog2(ABS32(a));
   a = VSHR32(a, e1-14);
   e2 = spx_ilog2(ABS32(b));
   b = VSHR32(b, e2-14);
   r.m = MULT16_16_Q15(a,b);
   r.e = e1+e2-13;
   return r;
}

/* Do NOT attempt to divide by a negative number */
static inline spx_float_t FLOAT_DIV32_FLOAT(spx_word32_t a, spx_float_t b)
{
   int e=0;
   spx_float_t r;
   if (a==0)
   {
      return FLOAT_ZERO;
   }
   e = spx_ilog2(ABS32(a))-spx_ilog2(b.m-1)-15;
   a = VSHR32(a, e);
   if (ABS32(a)>=SHL32(EXTEND32(b.m-1),15))
   {
      a >>= 1;
      e++;
   }
   r.m = DIV32_16(a,b.m);
   r.e = e-b.e;
   return r;
}


/* Do NOT attempt to divide by a negative number */
static inline spx_float_t FLOAT_DIV32(spx_word32_t a, spx_word32_t b)
{
   int e0=0,e=0;
   spx_float_t r;
   if (a==0)
   {
      return FLOAT_ZERO;
   }
   if (b>32767)
   {
      e0 = spx_ilog2(b)-14;
      b = VSHR32(b, e0);
      e0 = -e0;
   }
   e = spx_ilog2(ABS32(a))-spx_ilog2(b-1)-15;
   a = VSHR32(a, e);
   if (ABS32(a)>=SHL32(EXTEND32(b-1),15))
   {
      a >>= 1;
      e++;
   }
   e += e0;
   r.m = DIV32_16(a,b);
   r.e = e;
   return r;
}

/* Do NOT attempt to divide by a negative number */
static inline spx_float_t FLOAT_DIVU(spx_float_t a, spx_float_t b)
{
   int e=0;
   spx_int32_t num;
   spx_float_t r;
   if (b.m<=0)
   {
      speex_warning_int("Attempted to divide by", b.m);
      return FLOAT_ONE;
   }
   num = a.m;
   a.m = ABS16(a.m);
   while (a.m >= b.m)
   {
      e++;
      a.m >>= 1;
   }
   num = num << (15-e);
   r.m = DIV32_16(num,b.m);
   r.e = a.e-b.e-15+e;
   return r;
}

static inline spx_float_t FLOAT_SQRT(spx_float_t a)
{
   spx_float_t r;
   spx_int32_t m;
   m = SHL32(EXTEND32(a.m), 14);
   r.e = a.e - 14;
   if (r.e & 1)
   {
      r.e -= 1;
      m <<= 1;
   }
   r.e >>= 1;
   r.m = spx_sqrt(m);
   return r;
}

#else

#define spx_float_t float
#define FLOAT_ZERO 0.f
#define FLOAT_ONE 1.f
#define FLOAT_HALF 0.5f
#define PSEUDOFLOAT(x) (x)
#define FLOAT_MULT(a,b) ((a)*(b))
#define FLOAT_AMULT(a,b) ((a)*(b))
#define FLOAT_MUL32(a,b) ((a)*(b))
#define FLOAT_DIV32(a,b) ((a)/(b))
#define FLOAT_EXTRACT16(a) (a)
#define FLOAT_EXTRACT32(a) (a)
#define FLOAT_ADD(a,b) ((a)+(b))
#define FLOAT_SUB(a,b) ((a)-(b))
#define REALFLOAT(x) (x)
#define FLOAT_DIV32_FLOAT(a,b) ((a)/(b))
#define FLOAT_MUL32U(a,b) ((a)*(b))
#define FLOAT_SHL(a,b) (a)
#define FLOAT_LT(a,b) ((a)<(b))
#define FLOAT_GT(a,b) ((a)>(b))
#define FLOAT_DIVU(a,b) ((a)/(b))
#define FLOAT_SQRT(a) (spx_sqrt(a))

#endif

#endif
