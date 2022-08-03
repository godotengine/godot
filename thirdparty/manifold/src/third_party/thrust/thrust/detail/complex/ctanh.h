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
 * Copyright (c) 2011 David Schultz
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Adapted from FreeBSD by Filipe Maia <filipe.c.maia@gmail.com>:
 *    freebsd/lib/msun/src/s_ctanh.c
 */

/*
 * Hyperbolic tangent of a complex argument z = x + i y.
 *
 * The algorithm is from:
 *
 *   W. Kahan.  Branch Cuts for Complex Elementary Functions or Much
 *   Ado About Nothing's Sign Bit.  In The State of the Art in
 *   Numerical Analysis, pp. 165 ff.  Iserles and Powell, eds., 1987.
 *
 * Method:
 *
 *   Let t    = tan(x)
 *       beta = 1/cos^2(y)
 *       s    = sinh(x)
 *       rho  = cosh(x)
 *
 *   We have:
 *
 *   tanh(z) = sinh(z) / cosh(z)
 *
 *             sinh(x) cos(y) + i cosh(x) sin(y)
 *           = ---------------------------------
 *             cosh(x) cos(y) + i sinh(x) sin(y)
 *
 *             cosh(x) sinh(x) / cos^2(y) + i tan(y)
 *           = -------------------------------------
 *                    1 + sinh^2(x) / cos^2(y)
 *
 *             beta rho s + i t
 *           = ----------------
 *               1 + beta s^2
 *
 * Modifications:
 *
 *   I omitted the original algorithm's handling of overflow in tan(x) after
 *   verifying with nearpi.c that this can't happen in IEEE single or double
 *   precision.  I also handle large x differently.
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>
#include <cmath>

THRUST_NAMESPACE_BEGIN
namespace detail{
namespace complex{		      	

using thrust::complex;

__host__ __device__ inline
complex<double> ctanh(const complex<double>& z){
  double x, y;
  double t, beta, s, rho, denom;
  uint32_t hx, ix, lx;

  x = z.real();
  y = z.imag();

  extract_words(hx, lx, x);
  ix = hx & 0x7fffffff;

  /*
   * ctanh(NaN + i 0) = NaN + i 0
   *
   * ctanh(NaN + i y) = NaN + i NaN		for y != 0
   *
   * The imaginary part has the sign of x*sin(2*y), but there's no
   * special effort to get this right.
   *
   * ctanh(+-Inf +- i Inf) = +-1 +- 0
   *
   * ctanh(+-Inf + i y) = +-1 + 0 sin(2y)		for y finite
   *
   * The imaginary part of the sign is unspecified.  This special
   * case is only needed to avoid a spurious invalid exception when
   * y is infinite.
   */
  if (ix >= 0x7ff00000) {
    if ((ix & 0xfffff) | lx)	/* x is NaN */
      return (complex<double>(x, (y == 0 ? y : x * y)));
    set_high_word(x, hx - 0x40000000);	/* x = copysign(1, x) */
    return (complex<double>(x, copysign(0.0, isinf(y) ? y : sin(y) * cos(y))));
  }

  /*
   * ctanh(x + i NAN) = NaN + i NaN
   * ctanh(x +- i Inf) = NaN + i NaN
   */
  if (!isfinite(y))
    return (complex<double>(y - y, y - y));

  /*
   * ctanh(+-huge + i +-y) ~= +-1 +- i 2sin(2y)/exp(2x), using the
   * approximation sinh^2(huge) ~= exp(2*huge) / 4.
   * We use a modified formula to avoid spurious overflow.
   */
  if (ix >= 0x40360000) {	/* x >= 22 */
    double exp_mx = exp(-fabs(x));
    return (complex<double>(copysign(1.0, x),
			    4.0 * sin(y) * cos(y) * exp_mx * exp_mx));
  }

  /* Kahan's algorithm */
  t = tan(y);
  beta = 1.0 + t * t;	/* = 1 / cos^2(y) */
  s = sinh(x);
  rho = sqrt(1.0 + s * s);	/* = cosh(x) */
  denom = 1.0 + beta * s * s;
  return (complex<double>((beta * rho * s) / denom, t / denom));
}

__host__ __device__ inline
complex<double> ctan(complex<double> z){
  /* ctan(z) = -I * ctanh(I * z) */
  z = ctanh(complex<double>(-z.imag(), z.real()));
  return (complex<double>(z.imag(), -z.real()));
}

} // namespace complex

} // namespace detail


template <typename ValueType>
__host__ __device__
inline complex<ValueType> tan(const complex<ValueType>& z){
  return sin(z)/cos(z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> tanh(const complex<ValueType>& z){
  // This implementation seems better than the simple sin/cos
  return (thrust::exp(ValueType(2)*z)-ValueType(1))/
    (thrust::exp(ValueType(2)*z)+ValueType(1));
}

template <>
__host__ __device__
inline complex<double> tan(const complex<double>& z){
  return detail::complex::ctan(z);
}
  
template <>
__host__ __device__
inline complex<double> tanh(const complex<double>& z){
  return detail::complex::ctanh(z);
}
  
THRUST_NAMESPACE_END
