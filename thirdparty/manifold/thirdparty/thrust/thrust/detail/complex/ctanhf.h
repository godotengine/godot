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
 * Adapted from FreeBSD by Filipe Maia, filipe.c.maia@gmail.com:
 *    freebsd/lib/msun/src/s_ctanhf.c
 */

/*
 * Hyperbolic tangent of a complex argument z.  See ctanh.c for details.
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
complex<float> ctanhf(const complex<float>& z){
  float x, y;
  float t, beta, s, rho, denom;
  uint32_t hx, ix;

  x = z.real();
  y = z.imag();

  get_float_word(hx, x);
  ix = hx & 0x7fffffff;

  if (ix >= 0x7f800000) {
    if (ix & 0x7fffff)
      return (complex<float>(x, (y == 0.0f ? y : x * y)));
    set_float_word(x, hx - 0x40000000);
    return (complex<float>(x,
			   copysignf(0, isinf(y) ? y : sinf(y) * cosf(y))));
  }

  if (!isfinite(y))
    return (complex<float>(y - y, y - y));

  if (ix >= 0x41300000) {	/* x >= 11 */
    float exp_mx = expf(-fabsf(x));
    return (complex<float>(copysignf(1.0f, x),
			   4.0f * sinf(y) * cosf(y) * exp_mx * exp_mx));
  }

  t = tanf(y);
  beta = 1.0f + t * t;
  s = sinhf(x);
  rho = sqrtf(1.0f + s * s);
  denom = 1.0f + beta * s * s;
  return (complex<float>((beta * rho * s) / denom, t / denom));
}

  __host__ __device__ inline
  complex<float> ctanf(complex<float> z){
    z = ctanhf(complex<float>(-z.imag(), z.real()));
    return (complex<float>(z.imag(), -z.real()));
  }

} // namespace complex

} // namespace detail

template <>
__host__ __device__
inline complex<float> tan(const complex<float>& z){
  return detail::complex::ctanf(z);
}

template <>
__host__ __device__
inline complex<float> tanh(const complex<float>& z){
  return detail::complex::ctanhf(z);
}

THRUST_NAMESPACE_END
