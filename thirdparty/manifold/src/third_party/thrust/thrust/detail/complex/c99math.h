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
#pragma once

#include <thrust/detail/config.h>

#include <math.h>
#include <cmath>
#include <thrust/detail/complex/math_private.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace complex
{

// Define basic arithmetic functions so we can use them without explicit scope
// keeping the code as close as possible to FreeBSDs for ease of maintenance.
// It also provides an easy way to support compilers with missing C99 functions.
// When possible, just use the names in the global scope.
// Some platforms define these as macros, others as free functions.
// Avoid using the std:: form of these as nvcc may treat std::foo() as __host__ functions.

using ::log;
using ::acos;
using ::asin;
using ::sqrt;
using ::sinh;
using ::tan;
using ::cos;
using ::sin;
using ::exp;
using ::cosh;
using ::atan;

template <typename T>
inline __host__ __device__ T infinity();

template <>
inline __host__ __device__ float infinity<float>()
{
  float res;
  set_float_word(res, 0x7f800000);
  return res;
}


template <>
inline __host__ __device__ double infinity<double>()
{
  double res;
  insert_words(res, 0x7ff00000,0);
  return res;
}

#if defined _MSC_VER
__host__ __device__ inline int isinf(float x){
  return std::abs(x) == infinity<float>();
}

__host__ __device__ inline int isinf(double x){
  return std::abs(x) == infinity<double>();
}

__host__ __device__ inline int isnan(float x){
  return x != x;
}

__host__ __device__ inline int isnan(double x){
  return x != x;
}

__host__ __device__ inline int signbit(float x){
  return ((*((uint32_t *)&x)) & 0x80000000) != 0 ? 1 : 0;
}

__host__ __device__ inline int signbit(double x){
  return ((*((uint64_t *)&x)) & 0x8000000000000000) != 0ull ? 1 : 0;
}

__host__ __device__ inline int isfinite(float x){
  return !isnan(x) && !isinf(x);
}

__host__ __device__ inline int isfinite(double x){
  return !isnan(x) && !isinf(x);
}

#else

#  if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__)) && !defined(_NVHPC_CUDA)
// NVCC implements at least some signature of these as functions not macros.
using ::isinf;
using ::isnan;
using ::signbit;
using ::isfinite;
#  else
// Some compilers do not provide these in the global scope, because they are
// supposed to be macros. The versions in `std` are supposed to be functions.
// Since we're not compiling with nvcc, it's safe to use the functions in std::
using std::isinf;
using std::isnan;
using std::signbit;
using std::isfinite;
#  endif // __CUDACC__
#endif // _MSC_VER

using ::atanh;

#if defined _MSC_VER

__host__ __device__ inline double copysign(double x, double y){
  uint32_t hx,hy;
  get_high_word(hx,x);
  get_high_word(hy,y);
  set_high_word(x,(hx&0x7fffffff)|(hy&0x80000000));
  return x;
}

__host__ __device__ inline float copysignf(float x, float y){
  uint32_t ix,iy;
  get_float_word(ix,x);
  get_float_word(iy,y);
  set_float_word(x,(ix&0x7fffffff)|(iy&0x80000000));
  return x;
}



#if !defined(__CUDACC__) && !defined(_NVHPC_CUDA)

// Simple approximation to log1p as Visual Studio is lacking one
inline double log1p(double x){
  double u = 1.0+x;
  if(u == 1.0){
    return x;
  }else{
    if(u > 2.0){
      // Use normal log for large arguments
      return log(u);
    }else{
      return log(u)*(x/(u-1.0));
    }
  }
}

inline float log1pf(float x){
  float u = 1.0f+x;
  if(u == 1.0f){
    return x;
  }else{
    if(u > 2.0f){
      // Use normal log for large arguments
      return logf(u);
    }else{
      return logf(u)*(x/(u-1.0f));
    }
  }
}

#if _MSV_VER <= 1500
#include <complex>

inline float hypotf(float x, float y){
	return abs(std::complex<float>(x,y));
}

inline double hypot(double x, double y){
	return _hypot(x,y);
}

#endif // _MSC_VER <= 1500

#endif // __CUDACC__

#endif // _MSC_VER

} // namespace complex

} // namespace detail

THRUST_NAMESPACE_END

