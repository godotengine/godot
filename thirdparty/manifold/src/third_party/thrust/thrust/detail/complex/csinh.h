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
 * Copyright (c) 2005 Bruce D. Evans and Steven G. Kargl
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

/* adapted from FreeBSD:
 *    lib/msun/src/s_csinh.c
 */


#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>

THRUST_NAMESPACE_BEGIN
namespace detail{
namespace complex{		      	

using thrust::complex;

__host__ __device__ inline
complex<double> csinh(const complex<double>& z){
  double x, y, h;
  uint32_t hx, hy, ix, iy, lx, ly;
  const double huge = 8.98846567431157953864652595395e+307; // 0x1p1023;

  x = z.real();
  y = z.imag();

  extract_words(hx, lx, x);
  extract_words(hy, ly, y);

  ix = 0x7fffffff & hx;
  iy = 0x7fffffff & hy;

  /* Handle the nearly-non-exceptional cases where x and y are finite. */
  if (ix < 0x7ff00000 && iy < 0x7ff00000) {
    if ((iy | ly) == 0)
      return (complex<double>(sinh(x), y));
    if (ix < 0x40360000)	/* small x: normal case */
      return (complex<double>(sinh(x) * cos(y), cosh(x) * sin(y)));

    /* |x| >= 22, so cosh(x) ~= exp(|x|) */
    if (ix < 0x40862e42) {
      /* x < 710: exp(|x|) won't overflow */
      h = exp(fabs(x)) * 0.5;
      return (complex<double>(copysign(h, x) * cos(y), h * sin(y)));
    } else if (ix < 0x4096bbaa) {
      /* x < 1455: scale to avoid overflow */
      complex<double> z_ = ldexp_cexp(complex<double>(fabs(x), y), -1);
      return (complex<double>(z_.real() * copysign(1.0, x), z_.imag()));
    } else {
      /* x >= 1455: the result always overflows */
      h = huge * x;
      return (complex<double>(h * cos(y), h * h * sin(y)));
    }
  }

  /*
   * sinh(+-0 +- I Inf) = sign(d(+-0, dNaN))0 + I dNaN.
   * The sign of 0 in the result is unspecified.  Choice = normally
   * the same as dNaN.  Raise the invalid floating-point exception.
   *
   * sinh(+-0 +- I NaN) = sign(d(+-0, NaN))0 + I d(NaN).
   * The sign of 0 in the result is unspecified.  Choice = normally
   * the same as d(NaN).
   */
  if ((ix | lx) == 0 && iy >= 0x7ff00000)
    return (complex<double>(copysign(0.0, x * (y - y)), y - y));

  /*
   * sinh(+-Inf +- I 0) = +-Inf + I +-0.
   *
   * sinh(NaN +- I 0)   = d(NaN) + I +-0.
   */
  if ((iy | ly) == 0 && ix >= 0x7ff00000) {
    if (((hx & 0xfffff) | lx) == 0)
      return (complex<double>(x, y));
    return (complex<double>(x, copysign(0.0, y)));
  }

  /*
   * sinh(x +- I Inf) = dNaN + I dNaN.
   * Raise the invalid floating-point exception for finite nonzero x.
   *
   * sinh(x + I NaN) = d(NaN) + I d(NaN).
   * Optionally raises the invalid floating-point exception for finite
   * nonzero x.  Choice = don't raise (except for signaling NaNs).
   */
  if (ix < 0x7ff00000 && iy >= 0x7ff00000)
    return (complex<double>(y - y, x * (y - y)));

  /*
   * sinh(+-Inf + I NaN)  = +-Inf + I d(NaN).
   * The sign of Inf in the result is unspecified.  Choice = normally
   * the same as d(NaN).
   *
   * sinh(+-Inf +- I Inf) = +Inf + I dNaN.
   * The sign of Inf in the result is unspecified.  Choice = always +.
   * Raise the invalid floating-point exception.
   *
   * sinh(+-Inf + I y)   = +-Inf cos(y) + I Inf sin(y)
   */
  if (ix >= 0x7ff00000 && ((hx & 0xfffff) | lx) == 0) {
    if (iy >= 0x7ff00000)
      return (complex<double>(x * x, x * (y - y)));
    return (complex<double>(x * cos(y), infinity<double>() * sin(y)));
  }

  /*
   * sinh(NaN + I NaN)  = d(NaN) + I d(NaN).
   *
   * sinh(NaN +- I Inf) = d(NaN) + I d(NaN).
   * Optionally raises the invalid floating-point exception.
   * Choice = raise.
   *
   * sinh(NaN + I y)    = d(NaN) + I d(NaN).
   * Optionally raises the invalid floating-point exception for finite
   * nonzero y.  Choice = don't raise (except for signaling NaNs).
   */
  return (complex<double>((x * x) * (y - y), (x + x) * (y - y)));
}

__host__ __device__ inline
complex<double> csin(complex<double> z){
  /* csin(z) = -I * csinh(I * z) */
  z = csinh(complex<double>(-z.imag(), z.real()));
  return (complex<double>(z.imag(), -z.real()));
}

} // namespace complex

} // namespace detail

template <typename ValueType>
__host__ __device__
inline complex<ValueType> sin(const complex<ValueType>& z){
  const ValueType re = z.real();
  const ValueType im = z.imag();
  return complex<ValueType>(std::sin(re) * std::cosh(im), 
			    std::cos(re) * std::sinh(im));
}


template <typename ValueType>
__host__ __device__
inline complex<ValueType> sinh(const complex<ValueType>& z){
  const ValueType re = z.real();
  const ValueType im = z.imag();
  return complex<ValueType>(std::sinh(re) * std::cos(im), 
			    std::cosh(re) * std::sin(im));
}

template <>
__host__ __device__
inline complex<double> sin(const complex<double>& z){
  return detail::complex::csin(z);
}

template <>
__host__ __device__
inline complex<double> sinh(const complex<double>& z){
  return detail::complex::csinh(z);
}

THRUST_NAMESPACE_END
