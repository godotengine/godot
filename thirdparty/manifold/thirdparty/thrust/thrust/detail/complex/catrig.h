/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Adapted from FreeBSD by Filipe Maia <filipe.c.maia@gmail.com>:
 *    freebsd/lib/msun/src/catrig.c
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>
#include <cfloat>
#include <cmath>

THRUST_NAMESPACE_BEGIN
namespace detail{
namespace complex{

using thrust::complex;

__host__ __device__
inline void raise_inexact(){
  const volatile float tiny = 7.888609052210118054117286e-31; /* 0x1p-100; */
  // needs the volatile to prevent compiler from ignoring it
  volatile float junk = 1 + tiny;
  (void)junk;
}

__host__ __device__ inline complex<double> clog_for_large_values(complex<double> z);

/*
 * Testing indicates that all these functions are accurate up to 4 ULP.
 * The functions casin(h) and cacos(h) are about 2.5 times slower than asinh.
 * The functions catan(h) are a little under 2 times slower than atanh.
 *
 * The code for casinh, casin, cacos, and cacosh comes first.  The code is
 * rather complicated, and the four functions are highly interdependent.
 *
 * The code for catanh and catan comes at the end.  It is much simpler than
 * the other functions, and the code for these can be disconnected from the
 * rest of the code.
 */

/*
 *			================================
 *			| casinh, casin, cacos, cacosh |
 *			================================
 */

/*
 * The algorithm is very close to that in "Implementing the complex arcsine
 * and arccosine functions using exception handling" by T. E. Hull, Thomas F.
 * Fairgrieve, and Ping Tak Peter Tang, published in ACM Transactions on
 * Mathematical Software, Volume 23 Issue 3, 1997, Pages 299-335,
 * http://dl.acm.org/citation.cfm?id=275324.
 *
 * Throughout we use the convention z = x + I*y.
 *
 * casinh(z) = sign(x)*log(A+sqrt(A*A-1)) + I*asin(B)
 * where
 * A = (|z+I| + |z-I|) / 2
 * B = (|z+I| - |z-I|) / 2 = y/A
 *
 * These formulas become numerically unstable:
 *   (a) for Re(casinh(z)) when z is close to the line segment [-I, I] (that
 *       is, Re(casinh(z)) is close to 0);
 *   (b) for Im(casinh(z)) when z is close to either of the intervals
 *       [I, I*infinity) or (-I*infinity, -I] (that is, |Im(casinh(z))| is
 *       close to PI/2).
 *
 * These numerical problems are overcome by defining
 * f(a, b) = (hypot(a, b) - b) / 2 = a*a / (hypot(a, b) + b) / 2
 * Then if A < A_crossover, we use
 *   log(A + sqrt(A*A-1)) = log1p((A-1) + sqrt((A-1)*(A+1)))
 *   A-1 = f(x, 1+y) + f(x, 1-y)
 * and if B > B_crossover, we use
 *   asin(B) = atan2(y, sqrt(A*A - y*y)) = atan2(y, sqrt((A+y)*(A-y)))
 *   A-y = f(x, y+1) + f(x, y-1)
 * where without loss of generality we have assumed that x and y are
 * non-negative.
 *
 * Much of the difficulty comes because the intermediate computations may
 * produce overflows or underflows.  This is dealt with in the paper by Hull
 * et al by using exception handling.  We do this by detecting when
 * computations risk underflow or overflow.  The hardest part is handling the
 * underflows when computing f(a, b).
 *
 * Note that the function f(a, b) does not appear explicitly in the paper by
 * Hull et al, but the idea may be found on pages 308 and 309.  Introducing the
 * function f(a, b) allows us to concentrate many of the clever tricks in this
 * paper into one function.
 */

/*
 * Function f(a, b, hypot_a_b) = (hypot(a, b) - b) / 2.
 * Pass hypot(a, b) as the third argument.
 */
__host__ __device__
inline double
f(double a, double b, double hypot_a_b)
{
  if (b < 0)
    return ((hypot_a_b - b) / 2);
  if (b == 0)
    return (a / 2);
  return (a * a / (hypot_a_b + b) / 2);
}

/*
 * All the hard work is contained in this function.
 * x and y are assumed positive or zero, and less than RECIP_EPSILON.
 * Upon return:
 * rx = Re(casinh(z)) = -Im(cacos(y + I*x)).
 * B_is_usable is set to 1 if the value of B is usable.
 * If B_is_usable is set to 0, sqrt_A2my2 = sqrt(A*A - y*y), and new_y = y.
 * If returning sqrt_A2my2 has potential to result in an underflow, it is
 * rescaled, and new_y is similarly rescaled.
 */
__host__ __device__
inline void
do_hard_work(double x, double y, double *rx, int *B_is_usable, double *B,
			double *sqrt_A2my2, double *new_y)
{
  double R, S, A; /* A, B, R, and S are as in Hull et al. */
  double Am1, Amy; /* A-1, A-y. */
  const double A_crossover = 10; /* Hull et al suggest 1.5, but 10 works better */
  const double FOUR_SQRT_MIN = 5.966672584960165394632772e-154; /* =0x1p-509; >= 4 * sqrt(DBL_MIN) */
  const double B_crossover = 0.6417; /* suggested by Hull et al */

  R = hypot(x, y + 1);		/* |z+I| */
  S = hypot(x, y - 1);		/* |z-I| */

  /* A = (|z+I| + |z-I|) / 2 */
  A = (R + S) / 2;
  /*
   * Mathematically A >= 1.  There is a small chance that this will not
   * be so because of rounding errors.  So we will make certain it is
   * so.
   */
  if (A < 1)
    A = 1;

  if (A < A_crossover) {
    /*
     * Am1 = fp + fm, where fp = f(x, 1+y), and fm = f(x, 1-y).
     * rx = log1p(Am1 + sqrt(Am1*(A+1)))
     */
    if (y == 1 && x < DBL_EPSILON * DBL_EPSILON / 128) {
      /*
       * fp is of order x^2, and fm = x/2.
       * A = 1 (inexactly).
       */
      *rx = sqrt(x);
    } else if (x >= DBL_EPSILON * fabs(y - 1)) {
      /*
       * Underflow will not occur because
       * x >= DBL_EPSILON^2/128 >= FOUR_SQRT_MIN
       */
      Am1 = f(x, 1 + y, R) + f(x, 1 - y, S);
      *rx = log1p(Am1 + sqrt(Am1 * (A + 1)));
    } else if (y < 1) {
      /*
       * fp = x*x/(1+y)/4, fm = x*x/(1-y)/4, and
       * A = 1 (inexactly).
       */
      *rx = x / sqrt((1 - y) * (1 + y));
    } else {		/* if (y > 1) */
      /*
       * A-1 = y-1 (inexactly).
       */
      *rx = log1p((y - 1) + sqrt((y - 1) * (y + 1)));
    }
  } else {
    *rx = log(A + sqrt(A * A - 1));
  }

  *new_y = y;

  if (y < FOUR_SQRT_MIN) {
    /*
     * Avoid a possible underflow caused by y/A.  For casinh this
     * would be legitimate, but will be picked up by invoking atan2
     * later on.  For cacos this would not be legitimate.
     */
    *B_is_usable = 0;
    *sqrt_A2my2 = A * (2 / DBL_EPSILON);
    *new_y = y * (2 / DBL_EPSILON);
    return;
  }

  /* B = (|z+I| - |z-I|) / 2 = y/A */
  *B = y / A;
  *B_is_usable = 1;

  if (*B > B_crossover) {
    *B_is_usable = 0;
    /*
     * Amy = fp + fm, where fp = f(x, y+1), and fm = f(x, y-1).
     * sqrt_A2my2 = sqrt(Amy*(A+y))
     */
    if (y == 1 && x < DBL_EPSILON / 128) {
      /*
       * fp is of order x^2, and fm = x/2.
       * A = 1 (inexactly).
       */
      *sqrt_A2my2 = sqrt(x) * sqrt((A + y) / 2);
    } else if (x >= DBL_EPSILON * fabs(y - 1)) {
      /*
       * Underflow will not occur because
       * x >= DBL_EPSILON/128 >= FOUR_SQRT_MIN
       * and
       * x >= DBL_EPSILON^2 >= FOUR_SQRT_MIN
       */
      Amy = f(x, y + 1, R) + f(x, y - 1, S);
      *sqrt_A2my2 = sqrt(Amy * (A + y));
    } else if (y > 1) {
      /*
       * fp = x*x/(y+1)/4, fm = x*x/(y-1)/4, and
       * A = y (inexactly).
       *
       * y < RECIP_EPSILON.  So the following
       * scaling should avoid any underflow problems.
       */
      *sqrt_A2my2 = x * (4 / DBL_EPSILON / DBL_EPSILON) * y /
	sqrt((y + 1) * (y - 1));
      *new_y = y * (4 / DBL_EPSILON / DBL_EPSILON);
    } else {		/* if (y < 1) */
      /*
       * fm = 1-y >= DBL_EPSILON, fp is of order x^2, and
       * A = 1 (inexactly).
       */
      *sqrt_A2my2 = sqrt((1 - y) * (1 + y));
    }
  }
}

/*
 * casinh(z) = z + O(z^3)   as z -> 0
 *
 * casinh(z) = sign(x)*clog(sign(x)*z) + O(1/z^2)   as z -> infinity
 * The above formula works for the imaginary part as well, because
 * Im(casinh(z)) = sign(x)*atan2(sign(x)*y, fabs(x)) + O(y/z^3)
 *    as z -> infinity, uniformly in y
 */
__host__ __device__ inline
complex<double> casinh(complex<double> z)
{
  double x, y, ax, ay, rx, ry, B, sqrt_A2my2, new_y;
  int B_is_usable;
  complex<double> w;
  const double RECIP_EPSILON = 1.0 / DBL_EPSILON;
  const double m_ln2 = 6.9314718055994531e-1; /*  0x162e42fefa39ef.0p-53 */
  x = z.real();
  y = z.imag();
  ax = fabs(x);
  ay = fabs(y);

  if (isnan(x) || isnan(y)) {
    /* casinh(+-Inf + I*NaN) = +-Inf + I*NaN */
    if (isinf(x))
      return (complex<double>(x, y + y));
    /* casinh(NaN + I*+-Inf) = opt(+-)Inf + I*NaN */
    if (isinf(y))
      return (complex<double>(y, x + x));
    /* casinh(NaN + I*0) = NaN + I*0 */
    if (y == 0)
      return (complex<double>(x + x, y));
    /*
     * All other cases involving NaN return NaN + I*NaN.
     * C99 leaves it optional whether to raise invalid if one of
     * the arguments is not NaN, so we opt not to raise it.
     */
    return (complex<double>(x + 0.0 + (y + 0.0), x + 0.0 + (y + 0.0)));
  }

  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    /* clog...() will raise inexact unless x or y is infinite. */
    if (signbit(x) == 0)
      w = clog_for_large_values(z) + m_ln2;
    else
      w = clog_for_large_values(-z) + m_ln2;
    return (complex<double>(copysign(w.real(), x), copysign(w.imag(), y)));
  }

  /* Avoid spuriously raising inexact for z = 0. */
  if (x == 0 && y == 0)
    return (z);

  /* All remaining cases are inexact. */
  raise_inexact();

  const double SQRT_6_EPSILON = 3.6500241499888571e-8; /*  0x13988e1409212e.0p-77 */
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
    return (z);

  do_hard_work(ax, ay, &rx, &B_is_usable, &B, &sqrt_A2my2, &new_y);
  if (B_is_usable)
    ry = asin(B);
  else
    ry = atan2(new_y, sqrt_A2my2);
  return (complex<double>(copysign(rx, x), copysign(ry, y)));
}

/*
 * casin(z) = reverse(casinh(reverse(z)))
 * where reverse(x + I*y) = y + I*x = I*conj(z).
 */
__host__ __device__ inline
complex<double> casin(complex<double> z)
{
  complex<double> w = casinh(complex<double>(z.imag(), z.real()));

  return (complex<double>(w.imag(), w.real()));
}

/*
 * cacos(z) = PI/2 - casin(z)
 * but do the computation carefully so cacos(z) is accurate when z is
 * close to 1.
 *
 * cacos(z) = PI/2 - z + O(z^3)   as z -> 0
 *
 * cacos(z) = -sign(y)*I*clog(z) + O(1/z^2)   as z -> infinity
 * The above formula works for the real part as well, because
 * Re(cacos(z)) = atan2(fabs(y), x) + O(y/z^3)
 *    as z -> infinity, uniformly in y
 */
__host__ __device__ inline
complex<double> cacos(complex<double> z)
{
  double x, y, ax, ay, rx, ry, B, sqrt_A2mx2, new_x;
  int sx, sy;
  int B_is_usable;
  complex<double> w;
  const double pio2_hi = 1.5707963267948966e0; /*  0x1921fb54442d18.0p-52 */
  const volatile double pio2_lo = 6.1232339957367659e-17;	/*  0x11a62633145c07.0p-106 */
  const double m_ln2 = 6.9314718055994531e-1; /*  0x162e42fefa39ef.0p-53 */

  x = z.real();
  y = z.imag();
  sx = signbit(x);
  sy = signbit(y);
  ax = fabs(x);
  ay = fabs(y);

  if (isnan(x) || isnan(y)) {
    /* cacos(+-Inf + I*NaN) = NaN + I*opt(-)Inf */
    if (isinf(x))
      return (complex<double>(y + y, -infinity<double>()));
    /* cacos(NaN + I*+-Inf) = NaN + I*-+Inf */
    if (isinf(y))
      return (complex<double>(x + x, -y));
    /* cacos(0 + I*NaN) = PI/2 + I*NaN with inexact */
    if (x == 0)
      return (complex<double>(pio2_hi + pio2_lo, y + y));
    /*
     * All other cases involving NaN return NaN + I*NaN.
     * C99 leaves it optional whether to raise invalid if one of
     * the arguments is not NaN, so we opt not to raise it.
     */
    return (complex<double>(x + 0.0 + (y + 0), x + 0.0 + (y + 0)));
  }

  const double RECIP_EPSILON = 1.0 / DBL_EPSILON;
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    /* clog...() will raise inexact unless x or y is infinite. */
    w = clog_for_large_values(z);
    rx = fabs(w.imag());
    ry = w.real() + m_ln2;
    if (sy == 0)
      ry = -ry;
    return (complex<double>(rx, ry));
  }

  /* Avoid spuriously raising inexact for z = 1. */
  if (x == 1.0 && y == 0.0)
    return (complex<double>(0, -y));

  /* All remaining cases are inexact. */
  raise_inexact();

  const double SQRT_6_EPSILON = 3.6500241499888571e-8; /*  0x13988e1409212e.0p-77 */
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
    return (complex<double>(pio2_hi - (x - pio2_lo), -y));

  do_hard_work(ay, ax, &ry, &B_is_usable, &B, &sqrt_A2mx2, &new_x);
  if (B_is_usable) {
    if (sx == 0)
      rx = acos(B);
    else
      rx = acos(-B);
  } else {
    if (sx == 0)
      rx = atan2(sqrt_A2mx2, new_x);
    else
      rx = atan2(sqrt_A2mx2, -new_x);
  }
  if (sy == 0)
    ry = -ry;
  return (complex<double>(rx, ry));
}

/*
 * cacosh(z) = I*cacos(z) or -I*cacos(z)
 * where the sign is chosen so Re(cacosh(z)) >= 0.
 */
__host__ __device__ inline
complex<double> cacosh(complex<double> z)
{
  complex<double> w;
  double rx, ry;

  w = cacos(z);
  rx = w.real();
  ry = w.imag();
  /* cacosh(NaN + I*NaN) = NaN + I*NaN */
  if (isnan(rx) && isnan(ry))
    return (complex<double>(ry, rx));
  /* cacosh(NaN + I*+-Inf) = +Inf + I*NaN */
  /* cacosh(+-Inf + I*NaN) = +Inf + I*NaN */
  if (isnan(rx))
    return (complex<double>(fabs(ry), rx));
  /* cacosh(0 + I*NaN) = NaN + I*NaN */
  if (isnan(ry))
    return (complex<double>(ry, ry));
  return (complex<double>(fabs(ry), copysign(rx, z.imag())));
}

/*
 * Optimized version of clog() for |z| finite and larger than ~RECIP_EPSILON.
 */
__host__ __device__ inline
complex<double> clog_for_large_values(complex<double> z)
{
  double x, y;
  double ax, ay, t;
  const double m_e = 2.7182818284590452e0; /*  0x15bf0a8b145769.0p-51 */

  x = z.real();
  y = z.imag();
  ax = fabs(x);
  ay = fabs(y);
  if (ax < ay) {
    t = ax;
    ax = ay;
    ay = t;
  }

  /*
   * Avoid overflow in hypot() when x and y are both very large.
   * Divide x and y by E, and then add 1 to the logarithm.  This depends
   * on E being larger than sqrt(2).
   * Dividing by E causes an insignificant loss of accuracy; however
   * this method is still poor since it is uneccessarily slow.
   */
  if (ax > DBL_MAX / 2)
    return (complex<double>(log(hypot(x / m_e, y / m_e)) + 1, atan2(y, x)));

  /*
   * Avoid overflow when x or y is large.  Avoid underflow when x or
   * y is small.
   */
  const double QUARTER_SQRT_MAX = 5.966672584960165394632772e-154; /* = 0x1p509; <= sqrt(DBL_MAX) / 4 */
  const double SQRT_MIN =	1.491668146240041348658193e-154; /* = 0x1p-511; >= sqrt(DBL_MIN) */
  if (ax > QUARTER_SQRT_MAX || ay < SQRT_MIN)
    return (complex<double>(log(hypot(x, y)), atan2(y, x)));

  return (complex<double>(log(ax * ax + ay * ay) / 2, atan2(y, x)));
}

/*
 *				=================
 *				| catanh, catan |
 *				=================
 */

/*
   * sum_squares(x,y) = x*x + y*y (or just x*x if y*y would underflow).
   * Assumes x*x and y*y will not overflow.
   * Assumes x and y are finite.
   * Assumes y is non-negative.
   * Assumes fabs(x) >= DBL_EPSILON.
   */
__host__ __device__
inline double sum_squares(double x, double y)
{
  const double SQRT_MIN =	1.491668146240041348658193e-154; /* = 0x1p-511; >= sqrt(DBL_MIN) */
  /* Avoid underflow when y is small. */
  if (y < SQRT_MIN)
    return (x * x);

  return (x * x + y * y);
}

/*
 * real_part_reciprocal(x, y) = Re(1/(x+I*y)) = x/(x*x + y*y).
 * Assumes x and y are not NaN, and one of x and y is larger than
 * RECIP_EPSILON.  We avoid unwarranted underflow.  It is important to not use
 * the code creal(1/z), because the imaginary part may produce an unwanted
 * underflow.
 * This is only called in a context where inexact is always raised before
 * the call, so no effort is made to avoid or force inexact.
 */
__host__ __device__
inline double real_part_reciprocal(double x, double y)
{
  double scale;
  uint32_t hx, hy;
  int32_t ix, iy;

  /*
   * This code is inspired by the C99 document n1124.pdf, Section G.5.1,
   * example 2.
   */
  get_high_word(hx, x);
  ix = hx & 0x7ff00000;
  get_high_word(hy, y);
  iy = hy & 0x7ff00000;
  //#define	BIAS	(DBL_MAX_EXP - 1)
  const int BIAS = DBL_MAX_EXP - 1;
  /* XXX more guard digits are useful iff there is extra precision. */
  //#define	CUTOFF	(DBL_MANT_DIG / 2 + 1)	/* just half or 1 guard digit */
  const int CUTOFF = (DBL_MANT_DIG / 2 + 1);
  if (ix - iy >= CUTOFF << 20 || isinf(x))
    return (1 / x);		/* +-Inf -> +-0 is special */
  if (iy - ix >= CUTOFF << 20)
    return (x / y / y);	/* should avoid double div, but hard */
  if (ix <= (BIAS + DBL_MAX_EXP / 2 - CUTOFF) << 20)
    return (x / (x * x + y * y));
  scale = 1;
  set_high_word(scale, 0x7ff00000 - ix);	/* 2**(1-ilogb(x)) */
  x *= scale;
  y *= scale;
  return (x / (x * x + y * y) * scale);
}


/*
 * catanh(z) = log((1+z)/(1-z)) / 2
 *           = log1p(4*x / |z-1|^2) / 4
 *             + I * atan2(2*y, (1-x)*(1+x)-y*y) / 2
 *
 * catanh(z) = z + O(z^3)   as z -> 0
 *
 * catanh(z) = 1/z + sign(y)*I*PI/2 + O(1/z^3)   as z -> infinity
 * The above formula works for the real part as well, because
 * Re(catanh(z)) = x/|z|^2 + O(x/z^4)
 *    as z -> infinity, uniformly in x
 */
#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
__host__ __device__ inline
complex<double> catanh(complex<double> z)
{
  double x, y, ax, ay, rx, ry;
  const volatile double pio2_lo = 6.1232339957367659e-17; /*  0x11a62633145c07.0p-106 */
  const double pio2_hi = 1.5707963267948966e0;/*  0x1921fb54442d18.0p-52 */


  x = z.real();
  y = z.imag();
  ax = fabs(x);
  ay = fabs(y);

  /* This helps handle many cases. */
  if (y == 0 && ax <= 1)
    return (complex<double>(atanh(x), y));

  /* To ensure the same accuracy as atan(), and to filter out z = 0. */
  if (x == 0)
    return (complex<double>(x, atan(y)));

  if (isnan(x) || isnan(y)) {
    /* catanh(+-Inf + I*NaN) = +-0 + I*NaN */
    if (isinf(x))
      return (complex<double>(copysign(0.0, x), y + y));
    /* catanh(NaN + I*+-Inf) = sign(NaN)0 + I*+-PI/2 */
    if (isinf(y))
      return (complex<double>(copysign(0.0, x),
			      copysign(pio2_hi + pio2_lo, y)));
    /*
     * All other cases involving NaN return NaN + I*NaN.
     * C99 leaves it optional whether to raise invalid if one of
     * the arguments is not NaN, so we opt not to raise it.
     */
    return (complex<double>(x + 0.0 + (y + 0), x + 0.0 + (y + 0)));
  }

  const double RECIP_EPSILON = 1.0 / DBL_EPSILON;
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON)
    return (complex<double>(real_part_reciprocal(x, y),
			    copysign(pio2_hi + pio2_lo, y)));

  const double SQRT_3_EPSILON = 2.5809568279517849e-8; /*  0x1bb67ae8584caa.0p-78 */
  if (ax < SQRT_3_EPSILON / 2 && ay < SQRT_3_EPSILON / 2) {
    /*
     * z = 0 was filtered out above.  All other cases must raise
     * inexact, but this is the only only that needs to do it
     * explicitly.
     */
    raise_inexact();
    return (z);
  }

  const double m_ln2 = 6.9314718055994531e-1; /*  0x162e42fefa39ef.0p-53 */
  if (ax == 1 && ay < DBL_EPSILON)
    rx = (m_ln2 - log(ay)) / 2;
  else
    rx = log1p(4 * ax / sum_squares(ax - 1, ay)) / 4;

  if (ax == 1)
    ry = atan2(2.0, -ay) / 2;
  else if (ay < DBL_EPSILON)
    ry = atan2(2 * ay, (1 - ax) * (1 + ax)) / 2;
  else
    ry = atan2(2 * ay, (1 - ax) * (1 + ax) - ay * ay) / 2;

  return (complex<double>(copysign(rx, x), copysign(ry, y)));
}

/*
 * catan(z) = reverse(catanh(reverse(z)))
 * where reverse(x + I*y) = y + I*x = I*conj(z).
 */
__host__ __device__ inline
complex<double>catan(complex<double> z)
{
  complex<double> w = catanh(complex<double>(z.imag(), z.real()));
  return (complex<double>(w.imag(), w.real()));
}

#endif

} // namespace complex

} // namespace detail


template <typename ValueType>
__host__ __device__
inline complex<ValueType> acos(const complex<ValueType>& z){
  const complex<ValueType> ret = thrust::asin(z);
  const ValueType pi = ValueType(3.14159265358979323846);
  return complex<ValueType>(pi/2 - ret.real(),-ret.imag());
}


template <typename ValueType>
__host__ __device__
inline complex<ValueType> asin(const complex<ValueType>& z){
  const complex<ValueType> i(0,1);
  return -i*asinh(i*z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> atan(const complex<ValueType>& z){
  const complex<ValueType> i(0,1);
  return -i*thrust::atanh(i*z);
}


template <typename ValueType>
__host__ __device__
inline complex<ValueType> acosh(const complex<ValueType>& z){
  thrust::complex<ValueType> ret((z.real() - z.imag()) * (z.real() + z.imag()) - ValueType(1.0),
				 ValueType(2.0) * z.real() * z.imag());
  ret = thrust::sqrt(ret);
  if (z.real() < ValueType(0.0)){
    ret = -ret;
  }
  ret += z;
  ret = thrust::log(ret);
  if (ret.real() < ValueType(0.0)){
    ret = -ret;
  }
  return ret;
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> asinh(const complex<ValueType>& z){
  return thrust::log(thrust::sqrt(z*z+ValueType(1))+z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> atanh(const complex<ValueType>& z){
  ValueType imag2 = z.imag() *  z.imag();
  ValueType n = ValueType(1.0) + z.real();
  n = imag2 + n * n;

  ValueType d = ValueType(1.0) - z.real();
  d = imag2 + d * d;
  complex<ValueType> ret(ValueType(0.25) * (std::log(n) - std::log(d)),0);

  d = ValueType(1.0) -  z.real() * z.real() - imag2;

  ret.imag(ValueType(0.5) * std::atan2(ValueType(2.0) * z.imag(), d));
  return ret;
}

template <>
__host__ __device__
inline complex<double> acos(const complex<double>& z){
  return detail::complex::cacos(z);
}

template <>
__host__ __device__
inline complex<double> asin(const complex<double>& z){
  return detail::complex::casin(z);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
template <>
__host__ __device__
inline complex<double> atan(const complex<double>& z){
  return detail::complex::catan(z);
}
#endif

template <>
__host__ __device__
inline complex<double> acosh(const complex<double>& z){
  return detail::complex::cacosh(z);
}


template <>
__host__ __device__
inline complex<double> asinh(const complex<double>& z){
  return detail::complex::casinh(z);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
template <>
__host__ __device__
inline complex<double> atanh(const complex<double>& z){
  return detail::complex::catanh(z);
}
#endif

THRUST_NAMESPACE_END
