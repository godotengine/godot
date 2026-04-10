/* Copyright (c) 2002-2008 Jean-Marc Valin
   Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2024 Arm Limited
   Written by Jean-Marc Valin, and Yunho Huh */
/**
   @file mathops.h
   @brief Various math functions
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

#ifndef MATHOPS_H
#define MATHOPS_H

#include "arch.h"
#include "entcode.h"
#include "os_support.h"


#if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
#include "arm/mathops_arm.h"
#endif

#define PI 3.1415926535897931

/* Multiplies two 16-bit fractional values. Bit-exactness of this macro is important */
#define FRAC_MUL16(a,b) ((16384+((opus_int32)(opus_int16)(a)*(opus_int16)(b)))>>15)

unsigned isqrt32(opus_uint32 _val);

/* CELT doesn't need it for fixed-point, by analysis.c does. */
#if !defined(FIXED_POINT) || defined(ANALYSIS_C)
#define cA 0.43157974f
#define cB 0.67848403f
#define cC 0.08595542f
#define cE ((float)PI/2)
static OPUS_INLINE float fast_atan2f(float y, float x) {
   float x2, y2;
   x2 = x*x;
   y2 = y*y;
   /* For very small values, we don't care about the answer, so
      we can just return 0. */
   if (x2 + y2 < 1e-18f)
   {
      return 0;
   }
   if(x2<y2){
      float den = (y2 + cB*x2) * (y2 + cC*x2);
      return -x*y*(y2 + cA*x2) / den + (y<0 ? -cE : cE);
   }else{
      float den = (x2 + cB*y2) * (x2 + cC*y2);
      return  x*y*(x2 + cA*y2) / den + (y<0 ? -cE : cE) - (x*y<0 ? -cE : cE);
   }
}
#undef cA
#undef cB
#undef cC
#undef cE
#endif


#ifndef OVERRIDE_CELT_MAXABS16
static OPUS_INLINE opus_val32 celt_maxabs16(const opus_val16 *x, int len)
{
   int i;
   opus_val16 maxval = 0;
   opus_val16 minval = 0;
   for (i=0;i<len;i++)
   {
      maxval = MAX16(maxval, x[i]);
      minval = MIN16(minval, x[i]);
   }
   return MAX32(EXTEND32(maxval),-EXTEND32(minval));
}
#endif

#if defined(ENABLE_RES24) && defined(FIXED_POINT)
static OPUS_INLINE opus_res celt_maxabs_res(const opus_res *x, int len)
{
   int i;
   opus_res maxval = 0;
   opus_res minval = 0;
   for (i=0;i<len;i++)
   {
      maxval = MAX32(maxval, x[i]);
      minval = MIN32(minval, x[i]);
   }
   /* opus_res should never reach such amplitude, so we should be safe. */
   celt_sig_assert(minval != -2147483648);
   return MAX32(maxval,-minval);
}
#else
#define celt_maxabs_res celt_maxabs16
#endif


#ifndef OVERRIDE_CELT_MAXABS32
#ifdef FIXED_POINT
static OPUS_INLINE opus_val32 celt_maxabs32(const opus_val32 *x, int len)
{
   int i;
   opus_val32 maxval = 0;
   opus_val32 minval = 0;
   for (i=0;i<len;i++)
   {
      maxval = MAX32(maxval, x[i]);
      minval = MIN32(minval, x[i]);
   }
   return MAX32(maxval, -minval);
}
#else
#define celt_maxabs32(x,len) celt_maxabs16(x,len)
#endif
#endif

#ifndef FIXED_POINT
/* Calculates the arctangent of x using a Remez approximation of order 15,
 * incorporating only odd-powered terms. */
static OPUS_INLINE float celt_atan_norm(float x)
{
   #define ATAN2_2_OVER_PI 0.636619772367581f
   float x_sq = x * x;

   /* Polynomial coefficients approximated in the [0, 1] range.
    * Lolremez command: lolremez --degree 6 --range "0:1"
    *                   "(atan(sqrt(x))-sqrt(x))/(x*sqrt(x))" "1/(sqrt(x)*x)"
    * Please note that ATAN2_COEFF_A01 is fixed to 1.0f. */
   #define ATAN2_COEFF_A03 -3.3331659436225891113281250000e-01f
   #define ATAN2_COEFF_A05 1.99627041816711425781250000000e-01f
   #define ATAN2_COEFF_A07 -1.3976582884788513183593750000e-01f
   #define ATAN2_COEFF_A09 9.79423448443412780761718750000e-02f
   #define ATAN2_COEFF_A11 -5.7773590087890625000000000000e-02f
   #define ATAN2_COEFF_A13 2.30401363223791122436523437500e-02f
   #define ATAN2_COEFF_A15 -4.3554059229791164398193359375e-03f
   return ATAN2_2_OVER_PI * (x + x * x_sq * (ATAN2_COEFF_A03
                + x_sq * (ATAN2_COEFF_A05
                + x_sq * (ATAN2_COEFF_A07
                + x_sq * (ATAN2_COEFF_A09
                + x_sq * (ATAN2_COEFF_A11
                + x_sq * (ATAN2_COEFF_A13
                + x_sq * (ATAN2_COEFF_A15))))))));
}

/* Calculates the arctangent of y/x, returning an approximate value in radians.
 * Please refer to the linked wiki page (https://en.wikipedia.org/wiki/Atan2)
 * to learn how atan2 results are computed. */
static OPUS_INLINE float celt_atan2p_norm(float y, float x)
{
   celt_sig_assert(x>=0 && y>=0);

   /* For very small values, we don't care about the answer. */
   if ((x*x + y*y) < 1e-18f)
   {
      return 0;
   }

   if (y < x)
   {
      return celt_atan_norm(y / x);
   } else {
      return 1.f - celt_atan_norm(x / y);
   }
}
#endif

#if !defined(FIXED_POINT) || defined(ENABLE_QEXT)
/* Computes estimated cosine values for (PI/2 * x) using only terms with even
 * exponents. */
static OPUS_INLINE float celt_cos_norm2(float x)
{
   float x_norm_sq;
   int output_sign;
   /* Restrict x to [-1, 3]. */
   x -= 4*floor(.25*(x+1));
   /* Negative sign for [1, 3]. */
   output_sign = 1 - 2*(x>1);
   /* Restrict to [-1, 1]. */
   x -= 2*(x>1);

   /* The cosine function, cos(x), has a Taylor series representation consisting
    * exclusively of even-powered polynomial terms. */
   x_norm_sq = x * x;

   /* Polynomial coefficients approximated in the [0, 1] range using only terms
    * with even exponents.
    * Lolremez command: lolremez --degree 4 --range 0:1 "cos(sqrt(x)*pi*0.5)" */
   #define COS_COEFF_A0 9.999999403953552246093750000000e-01f
   #define COS_COEFF_A2 -1.233698248863220214843750000000000f
   #define COS_COEFF_A4 2.536507546901702880859375000000e-01f
   #define COS_COEFF_A6 -2.08106283098459243774414062500e-02f
   #define COS_COEFF_A8 8.581906440667808055877685546875e-04f
   return output_sign * (COS_COEFF_A0 + x_norm_sq * (COS_COEFF_A2 +
                               x_norm_sq * (COS_COEFF_A4 +
                               x_norm_sq * (COS_COEFF_A6 +
                               x_norm_sq * (COS_COEFF_A8)))));
}

#endif

#ifndef FIXED_POINT

#define celt_sqrt(x) ((float)sqrt(x))
#define celt_sqrt32(x) ((float)sqrt(x))
#define celt_rsqrt(x) (1.f/celt_sqrt(x))
#define celt_rsqrt_norm(x) (celt_rsqrt(x))
#define celt_rsqrt_norm32(x) (celt_rsqrt(x))
#define celt_cos_norm(x) ((float)cos((.5f*PI)*(x)))
#define celt_rcp(x) (1.f/(x))
#define celt_div(a,b) ((a)/(b))
#define frac_div32(a,b) ((float)(a)/(b))
#define frac_div32_q29(a,b) frac_div32(a,b)

#ifdef FLOAT_APPROX
/* Calculates the base-2 logarithm (log2(x)) of a number. It is designed for
 * systems using radix-2 floating-point representation, with the exponent
 * located at bits 23 to 30 and an offset of 127. Note that special cases like
 * denormalized numbers, positive/negative infinity, and NaN are not handled.
 * log2(x) = log2(x^exponent * mantissa)
 *         = exponent + log2(mantissa) */

/* Log2 x normalization single precision coefficients calculated by
 * 1 / (1 + 0.125 * index).
 * Coefficients in Double Precision
 * double log2_x_norm_coeff[8] = {
 *    1.0000000000000000000, 8.888888888888888e-01,
 *    8.000000000000000e-01, 7.272727272727273e-01,
 *    6.666666666666666e-01, 6.153846153846154e-01,
 *    5.714285714285714e-01, 5.333333333333333e-01} */
static const float log2_x_norm_coeff[8] = {
   1.000000000000000000000000000f, 8.88888895511627197265625e-01f,
   8.00000000000000000000000e-01f, 7.27272748947143554687500e-01f,
   6.66666686534881591796875e-01f, 6.15384638309478759765625e-01f,
   5.71428596973419189453125e-01f, 5.33333361148834228515625e-01f};

/* Log2 y normalization single precision coefficients calculated by
 * log2(1 + 0.125 * index).
 * Coefficients in Double Precision
 * double log2_y_norm_coeff[8] = {
 *    0.0000000000000000000, 1.699250014423124e-01,
 *    3.219280948873623e-01, 4.594316186372973e-01,
 *    5.849625007211562e-01, 7.004397181410922e-01,
 *    8.073549220576041e-01, 9.068905956085185e-01}; */
static const float log2_y_norm_coeff[8] = {
   0.0000000000000000000000000000f, 1.699250042438507080078125e-01f,
   3.219280838966369628906250e-01f, 4.594316184520721435546875e-01f,
   5.849624872207641601562500e-01f, 7.004396915435791015625000e-01f,
   8.073549270629882812500000e-01f, 9.068905711174011230468750e-01f};

static OPUS_INLINE float celt_log2(float x)
{
   opus_int32 integer;
   opus_int32 range_idx;
   union {
      float f;
      opus_uint32 i;
   } in;
   in.f = x;
   integer = (opus_int32)(in.i>>23)-127;
   in.i = (opus_int32)in.i - (opus_int32)((opus_uint32)integer<<23);

   /* Normalize the mantissa range from [1, 2] to [1,1.125], and then shift x
    * by 1.0625 to [-0.0625, 0.0625]. */
   range_idx = (in.i >> 20) & 0x7;
   in.f = in.f * log2_x_norm_coeff[range_idx] - 1.0625f;

   /* Polynomial coefficients approximated in the [1, 1.125] range.
    * Lolremez command: lolremez --degree 4 --range -0.0625:0.0625
    *                   "log(x+1.0625)/log(2)"
    * Coefficients in Double Precision
    * A0: 8.7462840624502679e-2    A1: 1.3578296070972002
    * A2: -6.3897703690210047e-1   A3: 4.0197125617419959e-1
    * A4: -2.8415445877832832e-1 */
   #define LOG2_COEFF_A0 8.74628424644470214843750000e-02f
   #define LOG2_COEFF_A1 1.357829570770263671875000000000f
   #define LOG2_COEFF_A2 -6.3897705078125000000000000e-01f
   #define LOG2_COEFF_A3 4.01971250772476196289062500e-01f
   #define LOG2_COEFF_A4 -2.8415444493293762207031250e-01f
   in.f = LOG2_COEFF_A0 + in.f * (LOG2_COEFF_A1
               + in.f * (LOG2_COEFF_A2
               + in.f * (LOG2_COEFF_A3
               + in.f * (LOG2_COEFF_A4))));
   return integer + in.f + log2_y_norm_coeff[range_idx];
}

/* Calculates an approximation of 2^x. The approximation was achieved by
 * employing a base-2 exponential function and utilizing a Remez approximation
 * of order 5, ensuring a controlled relative error.
 * exp2(x) = exp2(integer + fraction)
 *         = exp2(integer) * exp2(fraction) */
static OPUS_INLINE float celt_exp2(float x)
{
   opus_int32 integer;
   float frac;
   union {
      float f;
      opus_uint32 i;
   } res;
   integer = (int)floor(x);
   if (integer < -50)
      return 0;
   frac = x-integer;

   /* Polynomial coefficients approximated in the [0, 1] range.
    * Lolremez command: lolremez --degree 5 --range 0:1
    *                   "exp(x*0.693147180559945)" "exp(x*0.693147180559945)"
    * NOTE: log(2) ~ 0.693147180559945 */
   #define EXP2_COEFF_A0 9.999999403953552246093750000000e-01f
   #define EXP2_COEFF_A1 6.931530833244323730468750000000e-01f
   #define EXP2_COEFF_A2 2.401536107063293457031250000000e-01f
   #define EXP2_COEFF_A3 5.582631751894950866699218750000e-02f
   #define EXP2_COEFF_A4 8.989339694380760192871093750000e-03f
   #define EXP2_COEFF_A5 1.877576694823801517486572265625e-03f
   res.f = EXP2_COEFF_A0 + frac * (EXP2_COEFF_A1
               + frac * (EXP2_COEFF_A2
               + frac * (EXP2_COEFF_A3
               + frac * (EXP2_COEFF_A4
               + frac * (EXP2_COEFF_A5)))));
   res.i = (opus_uint32)((opus_int32)res.i + (opus_int32)((opus_uint32)integer<<23)) & 0x7fffffff;
   return res.f;
}

#else
#define celt_log2(x) ((float)(1.442695040888963387*log(x)))
#define celt_exp2(x) ((float)exp(0.6931471805599453094*(x)))
#endif

#define celt_exp2_db celt_exp2
#define celt_log2_db celt_log2

#define celt_sin(x) celt_cos_norm2((0.5f*PI) * (x) - 1.0f)
#define celt_log(x) (celt_log2(x) * 0.6931471805599453f)
#define celt_exp(x) (celt_exp2((x) * 1.4426950408889634f))

#endif

#ifdef FIXED_POINT

#include "os_support.h"

#ifndef OVERRIDE_CELT_ILOG2
/** Integer log in base2. Undefined for zero and negative numbers */
static OPUS_INLINE opus_int16 celt_ilog2(opus_int32 x)
{
   celt_sig_assert(x>0);
   return EC_ILOG(x)-1;
}
#endif


/** Integer log in base2. Defined for zero, but not for negative numbers */
static OPUS_INLINE opus_int16 celt_zlog2(opus_val32 x)
{
   return x <= 0 ? 0 : celt_ilog2(x);
}

opus_val16 celt_rsqrt_norm(opus_val32 x);

opus_val32 celt_rsqrt_norm32(opus_val32 x);

opus_val32 celt_sqrt(opus_val32 x);

opus_val32 celt_sqrt32(opus_val32 x);

opus_val16 celt_cos_norm(opus_val32 x);

opus_val32 celt_cos_norm32(opus_val32 x);

/** Base-2 logarithm approximation (log2(x)). (Q14 input, Q10 output) */
static OPUS_INLINE opus_val16 celt_log2(opus_val32 x)
{
   int i;
   opus_val16 n, frac;
   /* -0.41509302963303146, 0.9609890551383969, -0.31836011537636605,
       0.15530808010959576, -0.08556153059057618 */
   static const opus_val16 C[5] = {-6801+(1<<(13-10)), 15746, -5217, 2545, -1401};
   if (x==0)
      return -32767;
   i = celt_ilog2(x);
   n = VSHR32(x,i-15)-32768-16384;
   frac = ADD16(C[0], MULT16_16_Q15(n, ADD16(C[1], MULT16_16_Q15(n, ADD16(C[2], MULT16_16_Q15(n, ADD16(C[3], MULT16_16_Q15(n, C[4]))))))));
   return SHL32(i-13,10)+SHR32(frac,14-10);
}

/*
 K0 = 1
 K1 = log(2)
 K2 = 3-4*log(2)
 K3 = 3*log(2) - 2
*/
#define D0 16383
#define D1 22804
#define D2 14819
#define D3 10204

static OPUS_INLINE opus_val32 celt_exp2_frac(opus_val16 x)
{
   opus_val16 frac;
   frac = SHL16(x, 4);
   return ADD16(D0, MULT16_16_Q15(frac, ADD16(D1, MULT16_16_Q15(frac, ADD16(D2 , MULT16_16_Q15(D3,frac))))));
}

#undef D0
#undef D1
#undef D2
#undef D3

/** Base-2 exponential approximation (2^x). (Q10 input, Q16 output) */
static OPUS_INLINE opus_val32 celt_exp2(opus_val16 x)
{
   int integer;
   opus_val16 frac;
   integer = SHR16(x,10);
   if (integer>14)
      return 0x7f000000;
   else if (integer < -15)
      return 0;
   frac = celt_exp2_frac(x-SHL16(integer,10));
   return VSHR32(EXTEND32(frac), -integer-2);
}

#ifdef ENABLE_QEXT

/* Calculates the base-2 logarithm of a Q14 input value. The result is returned
 * in Q(DB_SHIFT). If the input value is 0, the function will output -32.0f. */
static OPUS_INLINE opus_val32 celt_log2_db(opus_val32 x) {
   /* Q30 */
   static const opus_val32 log2_x_norm_coeff[8] = {
      1073741824, 954437184, 858993472, 780903168,
      715827904,  660764224, 613566784, 572662336};
   /* Q24 */
   static const opus_val32 log2_y_norm_coeff[8] = {
      0,       2850868,  5401057,  7707983,
      9814042, 11751428, 13545168, 15215099};
   static const opus_val32 LOG2_COEFF_A0 = 1467383;     /* Q24 */
   static const opus_val32 LOG2_COEFF_A1 = 182244800;   /* Q27 */
   static const opus_val32 LOG2_COEFF_A2 = -21440512;   /* Q25 */
   static const opus_val32 LOG2_COEFF_A3 = 107903336;   /* Q28 */
   static const opus_val32 LOG2_COEFF_A4 = -610217024;  /* Q31 */

   opus_int32 integer, norm_coeff_idx, tmp;
   opus_val32 mantissa;
   if (x==0) {
      return -536870912; /* -32.0f */
   }
   integer =  SUB32(celt_ilog2(x), 14);  /* Q0 */
   mantissa = VSHR32(x, integer + 14 - 29);  /* Q29 */
   norm_coeff_idx = SHR32(mantissa, 29 - 3) & 0x7;
   /* mantissa is in Q28 (29 + Q_NORM_CONST - 31 where Q_NORM_CONST is Q30)
    * 285212672 (Q28) is 1.0625f. */
   mantissa = SUB32(MULT32_32_Q31(mantissa, log2_x_norm_coeff[norm_coeff_idx]),
                    285212672);

   /* q_a3(Q28): q_mantissa + q_a4 - 31
    * q_a2(Q25): q_mantissa + q_a3 - 31
    * q_a1(Q27): q_mantissa + q_a2 - 31 + 5
    * q_a0(Q24): q_mantissa + q_a1 - 31
    * where  q_mantissa is Q28 */
   /* Split evaluation in steps to avoid exploding macro expansion. */
   tmp = MULT32_32_Q31(mantissa, LOG2_COEFF_A4);
   tmp = MULT32_32_Q31(mantissa, ADD32(LOG2_COEFF_A3, tmp));
   tmp = SHL32(MULT32_32_Q31(mantissa, ADD32(LOG2_COEFF_A2, tmp)), 5 /* SHL32 for LOG2_COEFF_A1 */);
   tmp = MULT32_32_Q31(mantissa, ADD32(LOG2_COEFF_A1, tmp));
   return ADD32(log2_y_norm_coeff[norm_coeff_idx],
          ADD32(SHL32(integer, DB_SHIFT),
          ADD32(LOG2_COEFF_A0, tmp)));
}

/* Calculates exp2 for Q28 within a specific range (0 to 1.0) using fixed-point
 * arithmetic. The input number must be adjusted for Q DB_SHIFT. */
static OPUS_INLINE opus_val32 celt_exp2_db_frac(opus_val32 x)
{
   /* Approximation constants. */
   static const opus_int32 EXP2_COEFF_A0 = 268435440;   /* Q28 */
   static const opus_int32 EXP2_COEFF_A1 = 744267456;   /* Q30 */
   static const opus_int32 EXP2_COEFF_A2 = 1031451904;  /* Q32 */
   static const opus_int32 EXP2_COEFF_A3 = 959088832;   /* Q34 */
   static const opus_int32 EXP2_COEFF_A4 = 617742720;   /* Q36 */
   static const opus_int32 EXP2_COEFF_A5 = 516104352;   /* Q38 */
   opus_int32 tmp;
   /* Converts input value from Q24 to Q29. */
   opus_val32 x_q29 = SHL32(x, 29 - 24);
   /* Split evaluation in steps to avoid exploding macro expansion. */
   tmp = ADD32(EXP2_COEFF_A4, MULT32_32_Q31(x_q29, EXP2_COEFF_A5));
   tmp = ADD32(EXP2_COEFF_A3, MULT32_32_Q31(x_q29, tmp));
   tmp = ADD32(EXP2_COEFF_A2, MULT32_32_Q31(x_q29, tmp));
   tmp = ADD32(EXP2_COEFF_A1, MULT32_32_Q31(x_q29, tmp));
   return ADD32(EXP2_COEFF_A0, MULT32_32_Q31(x_q29, tmp));
}

/* Calculates exp2 for Q16 using fixed-point arithmetic. The input number must
 * be adjusted for Q DB_SHIFT. */
static OPUS_INLINE opus_val32 celt_exp2_db(opus_val32 x)
{
   int integer;
   opus_val32 frac;
   integer = SHR32(x,DB_SHIFT);
   if (integer>14)
      return 0x7f000000;
   else if (integer <= -17)
      return 0;
   frac = celt_exp2_db_frac(x-SHL32(integer, DB_SHIFT));  /* Q28 */
   return VSHR32(frac, -integer + 28 - 16);  /* Q16 */
}
#else

#define celt_log2_db(x) SHL32(EXTEND32(celt_log2(x)), DB_SHIFT-10)
#define celt_exp2_db_frac(x) SHL32(celt_exp2_frac(PSHR32(x, DB_SHIFT-10)), 14)
#define celt_exp2_db(x) celt_exp2(PSHR32(x, DB_SHIFT-10))

#endif


opus_val32 celt_rcp(opus_val32 x);
opus_val32 celt_rcp_norm32(opus_val32 x);

#define celt_div(a,b) MULT32_32_Q31((opus_val32)(a),celt_rcp(b))

opus_val32 frac_div32_q29(opus_val32 a, opus_val32 b);
opus_val32 frac_div32(opus_val32 a, opus_val32 b);

/* Computes atan(x) multiplied by 2/PI. The input value (x) should be within the
 * range of -1 to 1 and represented in Q30 format. The function will return the
 * result in Q30 format. */
static OPUS_INLINE opus_val32 celt_atan_norm(opus_val32 x)
{
   /* Approximation constants. */
   static const opus_int32 ATAN_2_OVER_PI = 1367130551;   /* Q31 */
   static const opus_int32 ATAN_COEFF_A03 = -715791936;   /* Q31 */
   static const opus_int32 ATAN_COEFF_A05 = 857391616;    /* Q32 */
   static const opus_int32 ATAN_COEFF_A07 = -1200579328;  /* Q33 */
   static const opus_int32 ATAN_COEFF_A09 = 1682636672;   /* Q34 */
   static const opus_int32 ATAN_COEFF_A11 = -1985085440;  /* Q35 */
   static const opus_int32 ATAN_COEFF_A13 = 1583306112;   /* Q36 */
   static const opus_int32 ATAN_COEFF_A15 = -598602432;   /* Q37 */
   opus_int32 x_sq_q30;
   opus_int32 x_q31;
   opus_int32 tmp;
   /* The expected x is in the range of [-1.0f, 1.0f] */
   celt_sig_assert((x <= 1073741824) && (x >= -1073741824));

   /* If x = 1.0f, returns 0.5f */
   if (x == 1073741824)
   {
      return 536870912; /* 0.5f (Q30) */
   }
   /* If x = 1.0f, returns 0.5f */
   if (x == -1073741824)
   {
      return -536870912; /* -0.5f (Q30) */
   }
   x_q31 = SHL32(x, 1);
   x_sq_q30 = MULT32_32_Q31(x_q31, x);
   /* Split evaluation in steps to avoid exploding macro expansion. */
   tmp = MULT32_32_Q31(x_sq_q30, ATAN_COEFF_A15);
   tmp = MULT32_32_Q31(x_sq_q30, ADD32(ATAN_COEFF_A13, tmp));
   tmp = MULT32_32_Q31(x_sq_q30, ADD32(ATAN_COEFF_A11, tmp));
   tmp = MULT32_32_Q31(x_sq_q30, ADD32(ATAN_COEFF_A09, tmp));
   tmp = MULT32_32_Q31(x_sq_q30, ADD32(ATAN_COEFF_A07, tmp));
   tmp = MULT32_32_Q31(x_sq_q30, ADD32(ATAN_COEFF_A05, tmp));
   tmp = MULT32_32_Q31(x_sq_q30, ADD32(ATAN_COEFF_A03, tmp));
   tmp = ADD32(x, MULT32_32_Q31(x_q31, tmp));
   return MULT32_32_Q31(ATAN_2_OVER_PI, tmp);
}

/* Calculates the arctangent of y/x, multiplies the result by 2/pi, and returns
 * the value in Q30 format. Both input values (x and y) must be within the range
 * of 0 to 1 and represented in Q30 format. Inputs must be zero or greater, and
 * at least one input must be non-zero. */
static OPUS_INLINE opus_val32 celt_atan2p_norm(opus_val32 y, opus_val32 x)
{
   celt_sig_assert(x>=0 && y>=0);
   if (y==0 && x==0) {
      return 0;
   } else if (y < x) {
      return celt_atan_norm(SHR32(frac_div32(y, x), 1));
   } else {
      celt_sig_assert(y > 0);
      return 1073741824 /* 1.0f Q30 */ -
             celt_atan_norm(SHR32(frac_div32(x, y), 1));
   }
}

#define M1 32767
#define M2 -21
#define M3 -11943
#define M4 4936

/* Atan approximation using a 4th order polynomial. Input is in Q15 format
   and normalized by pi/4. Output is in Q15 format */
static OPUS_INLINE opus_val16 celt_atan01(opus_val16 x)
{
   return MULT16_16_P15(x, ADD32(M1, MULT16_16_P15(x, ADD32(M2, MULT16_16_P15(x, ADD32(M3, MULT16_16_P15(M4, x)))))));
}

#undef M1
#undef M2
#undef M3
#undef M4

/* atan2() approximation valid for positive input values */
static OPUS_INLINE opus_val16 celt_atan2p(opus_val16 y, opus_val16 x)
{
   if (x==0 && y==0) {
      return 0;
   } else if (y < x)
   {
      opus_val32 arg;
      arg = celt_div(SHL32(EXTEND32(y),15),x);
      if (arg >= 32767)
         arg = 32767;
      return SHR16(celt_atan01(EXTRACT16(arg)),1);
   } else {
      opus_val32 arg;
      arg = celt_div(SHL32(EXTEND32(x),15),y);
      if (arg >= 32767)
         arg = 32767;
      return 25736-SHR16(celt_atan01(EXTRACT16(arg)),1);
   }
}

#endif /* FIXED_POINT */

#ifndef DISABLE_FLOAT_API

void celt_float2int16_c(const float * OPUS_RESTRICT in, short * OPUS_RESTRICT out, int cnt);

#ifndef OVERRIDE_FLOAT2INT16
#define celt_float2int16(in, out, cnt, arch) ((void)(arch), celt_float2int16_c(in, out, cnt))
#endif

int opus_limit2_checkwithin1_c(float *samples, int cnt);

#ifndef OVERRIDE_LIMIT2_CHECKWITHIN1
#define opus_limit2_checkwithin1(samples, cnt, arch) ((void)(arch), opus_limit2_checkwithin1_c(samples, cnt))
#endif

#endif /* DISABLE_FLOAT_API */

#endif /* MATHOPS_H */
