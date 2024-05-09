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
 * Copyright (c) 2007 David Schultz <das@FreeBSD.ORG>
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
 *    freebsd/lib/msun/src/s_csqrt.c
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
complex<double> csqrt(const complex<double>& z){
  complex<double> result;
  double a, b;
  double t;
  int scale;

  /* We risk spurious overflow for components >= DBL_MAX / (1 + sqrt(2)). */
  const double THRESH = 7.446288774449766337959726e+307;

  a = z.real();
  b = z.imag();

  /* Handle special cases. */
  if (z == 0.0)
    return (complex<double>(0.0, b));
  if (isinf(b))
    return (complex<double>(infinity<double>(), b));
  if (isnan(a)) {
    t = (b - b) / (b - b);	/* raise invalid if b is not a NaN */
    return (complex<double>(a, t));	/* return NaN + NaN i */
  }
  if (isinf(a)) {
    /*
     * csqrt(inf + NaN i)  = inf +  NaN i
     * csqrt(inf + y i)    = inf +  0 i
     * csqrt(-inf + NaN i) = NaN +- inf i
     * csqrt(-inf + y i)   = 0   +  inf i
     */
    if (signbit(a))
      return (complex<double>(fabs(b - b), copysign(a, b)));
    else
      return (complex<double>(a, copysign(b - b, b)));
  }
  /*
   * The remaining special case (b is NaN) is handled just fine by
   * the normal code path below.
   */

  // DBL_MIN*2
  const double low_thresh = 4.450147717014402766180465e-308;
  scale = 0;

  if (fabs(a) >= THRESH || fabs(b) >= THRESH) {
    /* Scale to avoid overflow. */
    a *= 0.25;
    b *= 0.25;
    scale = 1;
  }else if (fabs(a) <= low_thresh && fabs(b) <= low_thresh) {
    /* Scale to avoid underflow. */
    a *= 4.0;
    b *= 4.0;
    scale = 2;
  }
	

  /* Algorithm 312, CACM vol 10, Oct 1967. */
  if (a >= 0.0) {
    t = sqrt((a + hypot(a, b)) * 0.5);
    result = complex<double>(t, b / (2 * t));
  } else {
    t = sqrt((-a + hypot(a, b)) * 0.5);
    result = complex<double>(fabs(b) / (2 * t), copysign(t, b));
  }

  /* Rescale. */
  if (scale == 1)
    return (result * 2.0);
  else if (scale == 2)
    return (result * 0.5);
  else
    return (result);
}
      
} // namespace complex

} // namespace detail

template <typename ValueType>
__host__ __device__
inline complex<ValueType> sqrt(const complex<ValueType>& z){
  return thrust::polar(std::sqrt(thrust::abs(z)),thrust::arg(z)/ValueType(2));
}

template <>
__host__ __device__
inline complex<double> sqrt(const complex<double>& z){
  return detail::complex::csqrt(z);
}

THRUST_NAMESPACE_END
