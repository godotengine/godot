/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

/* adapted from FreeBSDs msun:*/


#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>

THRUST_NAMESPACE_BEGIN
namespace detail{
namespace complex{

using thrust::complex;

/* round down to 18 = 54/3 bits */
__host__ __device__ inline
double trim(double x){
  uint32_t hi;
  get_high_word(hi, x);
  insert_words(x, hi &0xfffffff8, 0);
  return x;
}


__host__ __device__ inline
complex<double> clog(const complex<double>& z){

  // Adapted from FreeBSDs msun
  double x, y;
  double ax, ay;
  double x0, y0, x1, y1, x2, y2, t, hm1;
  double val[12];
  int i, sorted;
  const double e = 2.7182818284590452354;

  x = z.real();
  y = z.imag();

  /* Handle NaNs using the general formula to mix them right. */
  if (x != x || y != y){
    return (complex<double>(std::log(norm(z)), std::atan2(y, x)));
  }

  ax = std::abs(x);
  ay = std::abs(y);
  if (ax < ay) {
    t = ax;
    ax = ay;
    ay = t;
  }

  /*
   * To avoid unnecessary overflow, if x and y are very large, divide x
   * and y by M_E, and then add 1 to the logarithm.  This depends on
   * M_E being larger than sqrt(2).
   * There is a potential loss of accuracy caused by dividing by M_E,
   * but this case should happen extremely rarely.
   */
  //    if (ay > 5e307){
  // For high values of ay -> hypotf(DBL_MAX,ay) = inf
  // We expect that for values at or below ay = 5e307 this should not happen
  if (ay > 5e307){
    return (complex<double>(std::log(hypot(x / e, y / e)) + 1.0, std::atan2(y, x)));
  }
  if (ax == 1.) {
    if (ay < 1e-150){
      return (complex<double>((ay * 0.5) * ay, std::atan2(y, x)));
    }
    return (complex<double>(log1p(ay * ay) * 0.5, std::atan2(y, x)));
  }

  /*
   * Because atan2 and hypot conform to C99, this also covers all the
   * edge cases when x or y are 0 or infinite.
   */
  if (ax < 1e-50 || ay < 1e-50 || ax > 1e50 || ay > 1e50){
    return (complex<double>(std::log(hypot(x, y)), std::atan2(y, x)));
  }

  /*
   * From this point on, we don't need to worry about underflow or
   * overflow in calculating ax*ax or ay*ay.
   */

  /* Some easy cases. */

  if (ax >= 1.0){
    return (complex<double>(log1p((ax-1)*(ax+1) + ay*ay) * 0.5, atan2(y, x)));
  }

  if (ax*ax + ay*ay <= 0.7){
    return (complex<double>(std::log(ax*ax + ay*ay) * 0.5, std::atan2(y, x)));
  }

  /*
   * Take extra care so that ULP of real part is small if hypot(x,y) is
   * moderately close to 1.
   */


  x0 = trim(ax);
  ax = ax-x0;
  x1 = trim(ax);
  x2 = ax-x1;
  y0 = trim(ay);
  ay = ay-y0;
  y1 = trim(ay);
  y2 = ay-y1;

  val[0] = x0*x0;
  val[1] = y0*y0;
  val[2] = 2*x0*x1;
  val[3] = 2*y0*y1;
  val[4] = x1*x1;
  val[5] = y1*y1;
  val[6] = 2*x0*x2;
  val[7] = 2*y0*y2;
  val[8] = 2*x1*x2;
  val[9] = 2*y1*y2;
  val[10] = x2*x2;
  val[11] = y2*y2;

  /* Bubble sort. */

  do {
    sorted = 1;
    for (i=0;i<11;i++) {
      if (val[i] < val[i+1]) {
	sorted = 0;
	t = val[i];
	val[i] = val[i+1];
	val[i+1] = t;
      }
    }
  } while (!sorted);

  hm1 = -1;
  for (i=0;i<12;i++){
    hm1 += val[i];
  }
  return (complex<double>(0.5 * log1p(hm1), atan2(y, x)));
}

} // namespace complex

} // namespace detail

template <typename ValueType>
__host__ __device__
inline complex<ValueType> log(const complex<ValueType>& z){
  return complex<ValueType>(std::log(thrust::abs(z)),thrust::arg(z));
}

template <>
__host__ __device__
inline complex<double> log(const complex<double>& z){
  return detail::complex::clog(z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> log10(const complex<ValueType>& z){
  // Using the explicit literal prevents compile time warnings in
  // devices that don't support doubles
  return thrust::log(z)/ValueType(2.30258509299404568402);
}

THRUST_NAMESPACE_END

