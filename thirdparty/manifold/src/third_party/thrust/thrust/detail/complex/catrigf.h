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

#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>
#include <thrust/detail/config.h>
#include <cfloat>
#include <cmath>

THRUST_NAMESPACE_BEGIN
namespace detail{
namespace complex{		      	

using thrust::complex;
  
__host__ __device__ inline
      complex<float> clog_for_large_values(complex<float> z);

/*
 * The algorithm is very close to that in "Implementing the complex arcsine
 * and arccosine functions using exception handling" by T. E. Hull, Thomas F.
 * Fairgrieve, and Ping Tak Peter Tang, published in ACM Transactions on
 * Mathematical Software, Volume 23 Issue 3, 1997, Pages 299-335,
 * http://dl.acm.org/citation.cfm?id=275324.
 *
 * See catrig.c for complete comments.
 *
 * XXX comments were removed automatically, and even short ones on the right
 * of statements were removed (all of them), contrary to normal style.  Only
 * a few comments on the right of declarations remain.
 */

__host__ __device__
inline float
f(float a, float b, float hypot_a_b)
{
  if (b < 0.0f)
    return ((hypot_a_b - b) / 2.0f);
  if (b == 0.0f)
    return (a / 2.0f);
  return (a * a / (hypot_a_b + b) / 2.0f);
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
do_hard_work(float x, float y, float *rx, int *B_is_usable, float *B,
	     float *sqrt_A2my2, float *new_y)
{
  float R, S, A; /* A, B, R, and S are as in Hull et al. */
  float Am1, Amy; /* A-1, A-y. */
  const float A_crossover = 10; /* Hull et al suggest 1.5, but 10 works better */
  const float FOUR_SQRT_MIN = 4.336808689942017736029811e-19f;; /* =0x1p-61; >= 4 * sqrt(FLT_MIN) */
  const float B_crossover = 0.6417f; /* suggested by Hull et al */
  R = hypotf(x, y + 1);
  S = hypotf(x, y - 1);

  A = (R + S) / 2;
  if (A < 1)
    A = 1;

  if (A < A_crossover) {
    if (y == 1 && x < FLT_EPSILON * FLT_EPSILON / 128) {
      *rx = sqrtf(x);
    } else if (x >= FLT_EPSILON * fabsf(y - 1)) {
      Am1 = f(x, 1 + y, R) + f(x, 1 - y, S);
      *rx = log1pf(Am1 + sqrtf(Am1 * (A + 1)));
    } else if (y < 1) {
      *rx = x / sqrtf((1 - y) * (1 + y));
    } else {
      *rx = log1pf((y - 1) + sqrtf((y - 1) * (y + 1)));
    }
  } else {
    *rx = logf(A + sqrtf(A * A - 1));
  }

  *new_y = y;

  if (y < FOUR_SQRT_MIN) {
    *B_is_usable = 0;
    *sqrt_A2my2 = A * (2 / FLT_EPSILON);
    *new_y = y * (2 / FLT_EPSILON);
    return;
  }

  *B = y / A;
  *B_is_usable = 1;

  if (*B > B_crossover) {
    *B_is_usable = 0;
    if (y == 1 && x < FLT_EPSILON / 128) {
      *sqrt_A2my2 = sqrtf(x) * sqrtf((A + y) / 2);
    } else if (x >= FLT_EPSILON * fabsf(y - 1)) {
      Amy = f(x, y + 1, R) + f(x, y - 1, S);
      *sqrt_A2my2 = sqrtf(Amy * (A + y));
    } else if (y > 1) {
      *sqrt_A2my2 = x * (4 / FLT_EPSILON / FLT_EPSILON) * y /
	sqrtf((y + 1) * (y - 1));
      *new_y = y * (4 / FLT_EPSILON / FLT_EPSILON);
    } else {
      *sqrt_A2my2 = sqrtf((1 - y) * (1 + y));
    }
  }

}

__host__ __device__ inline
complex<float>
casinhf(complex<float> z)
{
  float x, y, ax, ay, rx, ry, B, sqrt_A2my2, new_y;
  int B_is_usable;
  complex<float> w;
  const float RECIP_EPSILON = 1.0 / FLT_EPSILON;
  const float m_ln2 = 6.9314718055994531e-1f; /*  0x162e42fefa39ef.0p-53 */
  x = z.real();
  y = z.imag();
  ax = fabsf(x);
  ay = fabsf(y);

  if (isnan(x) || isnan(y)) {
    if (isinf(x))
      return (complex<float>(x, y + y));
    if (isinf(y))
      return (complex<float>(y, x + x));
    if (y == 0)
      return (complex<float>(x + x, y));
    return (complex<float>(x + 0.0f + (y + 0), x + 0.0f + (y + 0)));
  }

  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    if (signbit(x) == 0)
      w = clog_for_large_values(z) + m_ln2;
    else
      w = clog_for_large_values(-z) + m_ln2;
    return (complex<float>(copysignf(w.real(), x),
			   copysignf(w.imag(), y)));
  }

  if (x == 0 && y == 0)
    return (z);

  raise_inexact();

  const float SQRT_6_EPSILON = 8.4572793338e-4f;	/*  0xddb3d7.0p-34 */
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
    return (z);

  do_hard_work(ax, ay, &rx, &B_is_usable, &B, &sqrt_A2my2, &new_y);
  if (B_is_usable)
    ry = asinf(B);
  else
    ry = atan2f(new_y, sqrt_A2my2);
  return (complex<float>(copysignf(rx, x), copysignf(ry, y)));
}

__host__ __device__ inline
complex<float> casinf(complex<float> z)
{
  complex<float> w = casinhf(complex<float>(z.imag(), z.real()));

  return (complex<float>(w.imag(), w.real()));
}

__host__ __device__ inline
complex<float> cacosf(complex<float> z)
{
  float x, y, ax, ay, rx, ry, B, sqrt_A2mx2, new_x;
  int sx, sy;
  int B_is_usable;
  complex<float> w;
  const float pio2_hi = 1.5707963267948966e0f; /*  0x1921fb54442d18.0p-52 */
  const volatile float pio2_lo = 6.1232339957367659e-17f;	/*  0x11a62633145c07.0p-106 */
  const float m_ln2 = 6.9314718055994531e-1f; /*  0x162e42fefa39ef.0p-53 */

  x = z.real();
  y = z.imag();
  sx = signbit(x);
  sy = signbit(y);
  ax = fabsf(x);
  ay = fabsf(y);

  if (isnan(x) || isnan(y)) {
    if (isinf(x))
      return (complex<float>(y + y, -infinity<float>()));
    if (isinf(y))
      return (complex<float>(x + x, -y));
    if (x == 0)
      return (complex<float>(pio2_hi + pio2_lo, y + y));
    return (complex<float>(x + 0.0f + (y + 0), x + 0.0f + (y + 0)));
  }

  const float RECIP_EPSILON = 1.0 / FLT_EPSILON;
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    w = clog_for_large_values(z);
    rx = fabsf(w.imag());
    ry = w.real() + m_ln2;
    if (sy == 0)
      ry = -ry;
    return (complex<float>(rx, ry));
  }

  if (x == 1 && y == 0)
    return (complex<float>(0, -y));

  raise_inexact();

  const float SQRT_6_EPSILON = 8.4572793338e-4f;	/*  0xddb3d7.0p-34 */
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
    return (complex<float>(pio2_hi - (x - pio2_lo), -y));

  do_hard_work(ay, ax, &ry, &B_is_usable, &B, &sqrt_A2mx2, &new_x);
  if (B_is_usable) {
    if (sx == 0)
      rx = acosf(B);
    else
      rx = acosf(-B);
  } else {
    if (sx == 0)
      rx = atan2f(sqrt_A2mx2, new_x);
    else
      rx = atan2f(sqrt_A2mx2, -new_x);
  }
  if (sy == 0)
    ry = -ry;
  return (complex<float>(rx, ry));
}

__host__ __device__ inline
complex<float> cacoshf(complex<float> z)
{
  complex<float> w;
  float rx, ry;

  w = cacosf(z);
  rx = w.real();
  ry = w.imag();
  /* cacosh(NaN + I*NaN) = NaN + I*NaN */
  if (isnan(rx) && isnan(ry))
    return (complex<float>(ry, rx));
  /* cacosh(NaN + I*+-Inf) = +Inf + I*NaN */
  /* cacosh(+-Inf + I*NaN) = +Inf + I*NaN */
  if (isnan(rx))
    return (complex<float>(fabsf(ry), rx));
  /* cacosh(0 + I*NaN) = NaN + I*NaN */
  if (isnan(ry))
    return (complex<float>(ry, ry));
  return (complex<float>(fabsf(ry), copysignf(rx, z.imag())));
}

  /*
   * Optimized version of clog() for |z| finite and larger than ~RECIP_EPSILON.
   */
__host__ __device__ inline
complex<float> clog_for_large_values(complex<float> z)
{
  float x, y;
  float ax, ay, t;
  const float m_e = 2.7182818284590452e0f; /*  0x15bf0a8b145769.0p-51 */

  x = z.real();
  y = z.imag();
  ax = fabsf(x);
  ay = fabsf(y);
  if (ax < ay) {
    t = ax;
    ax = ay;
    ay = t;
  }

  if (ax > FLT_MAX / 2)
    return (complex<float>(logf(hypotf(x / m_e, y / m_e)) + 1,
			   atan2f(y, x)));

  const float QUARTER_SQRT_MAX = 2.3058430092136939520000000e+18f; /* = 0x1p61; <= sqrt(FLT_MAX) / 4 */
  const float SQRT_MIN =	1.084202172485504434007453e-19f; /* 0x1p-63; >= sqrt(FLT_MIN) */
  if (ax > QUARTER_SQRT_MAX || ay < SQRT_MIN)
    return (complex<float>(logf(hypotf(x, y)), atan2f(y, x)));

  return (complex<float>(logf(ax * ax + ay * ay) / 2, atan2f(y, x)));
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
 * Assumes fabsf(x) >= FLT_EPSILON.
 */
__host__ __device__
inline float sum_squares(float x, float y)
{
  const float SQRT_MIN =	1.084202172485504434007453e-19f; /* 0x1p-63; >= sqrt(FLT_MIN) */
  /* Avoid underflow when y is small. */
  if (y < SQRT_MIN)
    return (x * x);

  return (x * x + y * y);
}

__host__ __device__
inline float real_part_reciprocal(float x, float y)
{
  float scale;
  uint32_t hx, hy;
  int32_t ix, iy;

  get_float_word(hx, x);
  ix = hx & 0x7f800000;
  get_float_word(hy, y);
  iy = hy & 0x7f800000;
  //#define	BIAS	(FLT_MAX_EXP - 1)
  const int BIAS = FLT_MAX_EXP - 1;
  //#define	CUTOFF	(FLT_MANT_DIG / 2 + 1)
  const int CUTOFF = (FLT_MANT_DIG / 2 + 1);
  if (ix - iy >= CUTOFF << 23 || isinf(x))
    return (1 / x);
  if (iy - ix >= CUTOFF << 23)
    return (x / y / y);
  if (ix <= (BIAS + FLT_MAX_EXP / 2 - CUTOFF) << 23)
    return (x / (x * x + y * y));
  set_float_word(scale, 0x7f800000 - ix);
  x *= scale;
  y *= scale;
  return (x / (x * x + y * y) * scale);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
__host__ __device__ inline
complex<float> catanhf(complex<float> z)
{
  float x, y, ax, ay, rx, ry;
  const volatile float pio2_lo = 6.1232339957367659e-17f; /*  0x11a62633145c07.0p-106 */
  const float pio2_hi = 1.5707963267948966e0f;/*  0x1921fb54442d18.0p-52 */


  x = z.real();
  y = z.imag();
  ax = fabsf(x);
  ay = fabsf(y);


  if (y == 0 && ax <= 1)
    return (complex<float>(atanhf(x), y));

  if (x == 0)
    return (complex<float>(x, atanf(y)));

  if (isnan(x) || isnan(y)) {
    if (isinf(x))
      return (complex<float>(copysignf(0, x), y + y));
    if (isinf(y))
      return (complex<float>(copysignf(0, x),
			     copysignf(pio2_hi + pio2_lo, y)));
    return (complex<float>(x + 0.0f + (y + 0.0f), x + 0.0f + (y + 0.0f)));
  }

  const float RECIP_EPSILON = 1.0f / FLT_EPSILON;
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON)
    return (complex<float>(real_part_reciprocal(x, y),
			   copysignf(pio2_hi + pio2_lo, y)));

  const float SQRT_3_EPSILON = 5.9801995673e-4f; /*  0x9cc471.0p-34 */
  if (ax < SQRT_3_EPSILON / 2 && ay < SQRT_3_EPSILON / 2) {
    raise_inexact();
    return (z);
  }

  const float m_ln2 = 6.9314718056e-1f; /*  0xb17218.0p-24 */
  if (ax == 1 && ay < FLT_EPSILON)
    rx = (m_ln2 - logf(ay)) / 2;
  else
    rx = log1pf(4 * ax / sum_squares(ax - 1, ay)) / 4;

  if (ax == 1)
    ry = atan2f(2, -ay) / 2;
  else if (ay < FLT_EPSILON)
    ry = atan2f(2 * ay, (1 - ax) * (1 + ax)) / 2;
  else
    ry = atan2f(2 * ay, (1 - ax) * (1 + ax) - ay * ay) / 2;

  return (complex<float>(copysignf(rx, x), copysignf(ry, y)));
}

__host__ __device__ inline
complex<float>catanf(complex<float> z){
  complex<float> w = catanhf(complex<float>(z.imag(), z.real()));
  return (complex<float>(w.imag(), w.real()));
}
#endif

} // namespace complex

} // namespace detail


template <>
__host__ __device__
inline complex<float> acos(const complex<float>& z){
  return detail::complex::cacosf(z);
}

template <>
__host__ __device__
inline complex<float> asin(const complex<float>& z){
  return detail::complex::casinf(z);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
template <>
__host__ __device__
inline complex<float> atan(const complex<float>& z){
  return detail::complex::catanf(z);
}
#endif

template <>
__host__ __device__
inline complex<float> acosh(const complex<float>& z){
  return detail::complex::cacoshf(z);
}


template <>
__host__ __device__
inline complex<float> asinh(const complex<float>& z){
  return detail::complex::casinhf(z);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
template <>
__host__ __device__
inline complex<float> atanh(const complex<float>& z){
  return detail::complex::catanhf(z);
}
#endif

THRUST_NAMESPACE_END
