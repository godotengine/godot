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
complex<float> csqrtf(const complex<float>& z){
  float a = z.real(), b = z.imag();
  float t;
  int scale;
  complex<float> result;

  /* We risk spurious overflow for components >= FLT_MAX / (1 + sqrt(2)). */
  const float THRESH = 1.40949553037932e+38f;

  /* Handle special cases. */
  if (z == 0.0f)
    return (complex<float>(0, b));
  if (isinf(b))
    return (complex<float>(infinity<float>(), b));
  if (isnan(a)) {
    t = (b - b) / (b - b);	/* raise invalid if b is not a NaN */
    return (complex<float>(a, t));	/* return NaN + NaN i */
  }
  if (isinf(a)) {
    /*
     * csqrtf(inf + NaN i)  = inf +  NaN i
     * csqrtf(inf + y i)    = inf +  0 i
     * csqrtf(-inf + NaN i) = NaN +- inf i
     * csqrtf(-inf + y i)   = 0   +  inf i
     */
    if (signbit(a))
      return (complex<float>(fabsf(b - b), copysignf(a, b)));
    else
      return (complex<float>(a, copysignf(b - b, b)));
  }
  /*
   * The remaining special case (b is NaN) is handled just fine by
   * the normal code path below.
   */

  /* 
   * Unlike in the FreeBSD code we'll avoid using double precision as
   * not all hardware supports it.
   */

  // FLT_MIN*2
  const float low_thresh = 2.35098870164458e-38f;
  scale = 0;

  if (fabsf(a) >= THRESH || fabsf(b) >= THRESH) {
    /* Scale to avoid overflow. */
    a *= 0.25f;
    b *= 0.25f;
    scale = 1;
  }else if (fabsf(a) <= low_thresh && fabsf(b) <= low_thresh) {
    /* Scale to avoid underflow. */
    a *= 4.f;
    b *= 4.f;
    scale = 2;
  }

  /* Algorithm 312, CACM vol 10, Oct 1967. */
  if (a >= 0.0f) {
    t = sqrtf((a + hypotf(a, b)) * 0.5f);
    result = complex<float>(t, b / (2.0f * t));
  } else {
    t = sqrtf((-a + hypotf(a, b)) * 0.5f);
    result = complex<float>(fabsf(b) / (2.0f * t), copysignf(t, b));
  }

  /* Rescale. */
  if (scale == 1)
    return (result * 2.0f);
  else if (scale == 2)
    return (result * 0.5f);
  else
    return (result);
}      

} // namespace complex

} // namespace detail

template <>
__host__ __device__
inline complex<float> sqrt(const complex<float>& z){
  return detail::complex::csqrtf(z);
}

THRUST_NAMESPACE_END
