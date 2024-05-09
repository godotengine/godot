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
 * Copyright (c) 2011 David Schultz <das@FreeBSD.ORG>
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

/* adapted from FreeBSD:
 *    lib/msun/src/s_cexp.c
 *    lib/msun/src/k_exp.c
 *
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>

THRUST_NAMESPACE_BEGIN
namespace detail{
namespace complex{		      	
/*
 * Compute exp(x), scaled to avoid spurious overflow.  An exponent is
 * returned separately in 'expt'.
 *
 * Input:  ln(DBL_MAX) <= x < ln(2 * DBL_MAX / DBL_MIN_DENORM) ~= 1454.91
 * Output: 2**1023 <= y < 2**1024
 */
__host__ __device__ inline
	double frexp_exp(double x, int *expt){
  const uint32_t k = 1799;		/* constant for reduction */
  const double kln2 =  1246.97177782734161156;	/* k * ln2 */
	
  double exp_x;
  uint32_t hx;
	
  /*
   * We use exp(x) = exp(x - kln2) * 2**k, carefully chosen to
   * minimize |exp(kln2) - 2**k|.  We also scale the exponent of
   * exp_x to MAX_EXP so that the result can be multiplied by
   * a tiny number without losing accuracy due to denormalization.
   */
  exp_x = exp(x - kln2);
  get_high_word(hx, exp_x);
  *expt = (hx >> 20) - (0x3ff + 1023) + k;
  set_high_word(exp_x, (hx & 0xfffff) | ((0x3ff + 1023) << 20));
  return (exp_x);
}
      
      
__host__ __device__ inline
complex<double>	ldexp_cexp(complex<double> z, int expt){
  double x, y, exp_x, scale1, scale2;
  int ex_expt, half_expt;
	
  x = z.real();
  y = z.imag();
  exp_x = frexp_exp(x, &ex_expt);
  expt += ex_expt;
	
  /*
   * Arrange so that scale1 * scale2 == 2**expt.  We use this to
   * compensate for scalbn being horrendously slow.
   */
  half_expt = expt / 2;
  insert_words(scale1, (0x3ff + half_expt) << 20, 0);
  half_expt = expt - half_expt;
  insert_words(scale2, (0x3ff + half_expt) << 20, 0);
	
  return (complex<double>(cos(y) * exp_x * scale1 * scale2,
			  sin(y) * exp_x * scale1 * scale2));
}
	

__host__ __device__ inline
complex<double> cexp(const complex<double>& z){
  double x, y, exp_x;
  uint32_t hx, hy, lx, ly;

  const uint32_t
    exp_ovfl  = 0x40862e42,			/* high bits of MAX_EXP * ln2 ~= 710 */
    cexp_ovfl = 0x4096b8e4;			/* (MAX_EXP - MIN_DENORM_EXP) * ln2 */

	  
  x = z.real();
  y = z.imag();
	  
  extract_words(hy, ly, y);
  hy &= 0x7fffffff;
	  
  /* cexp(x + I 0) = exp(x) + I 0 */
  if ((hy | ly) == 0)
    return (complex<double>(exp(x), y));
  extract_words(hx, lx, x);
  /* cexp(0 + I y) = cos(y) + I sin(y) */
  if (((hx & 0x7fffffff) | lx) == 0)
    return (complex<double>(cos(y), sin(y)));
	  
  if (hy >= 0x7ff00000) {
    if (lx != 0 || (hx & 0x7fffffff) != 0x7ff00000) {
      /* cexp(finite|NaN +- I Inf|NaN) = NaN + I NaN */
      return (complex<double>(y - y, y - y));
    } else if (hx & 0x80000000) {
      /* cexp(-Inf +- I Inf|NaN) = 0 + I 0 */
      return (complex<double>(0.0, 0.0));
    } else {
      /* cexp(+Inf +- I Inf|NaN) = Inf + I NaN */
      return (complex<double>(x, y - y));
    }
  }
	  
  if (hx >= exp_ovfl && hx <= cexp_ovfl) {
    /*
     * x is between 709.7 and 1454.3, so we must scale to avoid
     * overflow in exp(x).
     */
    return (ldexp_cexp(z, 0));
  } else {
    /*
     * Cases covered here:
     *  -  x < exp_ovfl and exp(x) won't overflow (common case)
     *  -  x > cexp_ovfl, so exp(x) * s overflows for all s > 0
     *  -  x = +-Inf (generated by exp())
     *  -  x = NaN (spurious inexact exception from y)
     */
    exp_x = std::exp(x);
    return (complex<double>(exp_x * cos(y), exp_x * sin(y)));
  }
}
	
} // namespace complex
 
} // namespace detail

template <typename ValueType>
__host__ __device__
inline complex<ValueType> exp(const complex<ValueType>& z){    
  return polar(std::exp(z.real()),z.imag());
}

template <>
__host__ __device__
inline complex<double> exp(const complex<double>& z){    
  return detail::complex::cexp(z);
}

THRUST_NAMESPACE_END
