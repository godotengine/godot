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
 *    lib/msun/src/s_csinhf.c
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
complex<float> csinhf(const complex<float>& z){

  float x, y, h;
  uint32_t hx, hy, ix, iy;

  const float huge = 1.70141183460469231731687303716e+38; //0x1p127;

  x = z.real();
  y = z.imag();

  get_float_word(hx, x);
  get_float_word(hy, y);

  ix = 0x7fffffff & hx;
  iy = 0x7fffffff & hy;

  if (ix < 0x7f800000 && iy < 0x7f800000) {
    if (iy == 0)
      return (complex<float>(sinhf(x), y));
    if (ix < 0x41100000)	/* small x: normal case */
      return (complex<float>(sinhf(x) * cosf(y), coshf(x) * sinf(y)));

    /* |x| >= 9, so cosh(x) ~= exp(|x|) */
    if (ix < 0x42b17218) {
      /* x < 88.7: expf(|x|) won't overflow */
      h = expf(fabsf(x)) * 0.5f;
      return (complex<float>(copysignf(h, x) * cosf(y), h * sinf(y)));
    } else if (ix < 0x4340b1e7) {
      /* x < 192.7: scale to avoid overflow */
      complex<float> z_ = ldexp_cexpf(complex<float>(fabsf(x), y), -1);
      return (complex<float>(z_.real() * copysignf(1.0f, x), z_.imag()));
    } else {
      /* x >= 192.7: the result always overflows */
      h = huge * x;
      return (complex<float>(h * cosf(y), h * h * sinf(y)));
    }
  }

  if (ix == 0 && iy >= 0x7f800000)
    return (complex<float>(copysignf(0, x * (y - y)), y - y));

  if (iy == 0 && ix >= 0x7f800000) {
    if ((hx & 0x7fffff) == 0)
      return (complex<float>(x, y));
    return (complex<float>(x, copysignf(0.0f, y)));
  }

  if (ix < 0x7f800000 && iy >= 0x7f800000)
    return (complex<float>(y - y, x * (y - y)));

  if (ix >= 0x7f800000 && (hx & 0x7fffff) == 0) {
    if (iy >= 0x7f800000)
      return (complex<float>(x * x, x * (y - y)));
    return (complex<float>(x * cosf(y), infinity<float>() * sinf(y)));
  }

  return (complex<float>((x * x) * (y - y), (x + x) * (y - y)));
}

__host__ __device__ inline
complex<float> csinf(complex<float> z){
  z = csinhf(complex<float>(-z.imag(), z.real()));
  return (complex<float>(z.imag(), -z.real()));
}
      
} // namespace complex

} // namespace detail
  
template <>
__host__ __device__
inline complex<float> sin(const complex<float>& z){
  return detail::complex::csinf(z);
}

template <>
__host__ __device__
inline complex<float> sinh(const complex<float>& z){
  return detail::complex::csinhf(z);
}

THRUST_NAMESPACE_END
