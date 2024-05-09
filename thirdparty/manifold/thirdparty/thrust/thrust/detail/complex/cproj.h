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

#include <thrust/complex.h>
#include <thrust/detail/complex/math_private.h>
#include <cmath>

THRUST_NAMESPACE_BEGIN
namespace detail{
namespace complex{	 
__host__ __device__
inline complex<float> cprojf(const complex<float>& z){
  if(!isinf(z.real()) && !isinf(z.imag())){
    return z;
  }else{
    // std::numeric_limits<T>::infinity() doesn't run on the GPU
    return complex<float>(infinity<float>(), copysignf(0.0, z.imag()));
  }
}
  
__host__ __device__
inline complex<double> cproj(const complex<double>& z){
  if(!isinf(z.real()) && !isinf(z.imag())){
    return z;
  }else{
    // std::numeric_limits<T>::infinity() doesn't run on the GPU
    return complex<double>(infinity<double>(), copysign(0.0, z.imag()));
  }
}

}
 
}

template <typename T>
__host__ __device__
inline thrust::complex<T> proj(const thrust::complex<T>& z){
  return detail::complex::cproj(z);
}
  

template <>
__host__ __device__
inline thrust::complex<double> proj(const thrust::complex<double>& z){
  return detail::complex::cproj(z);
}
  
template <>
__host__ __device__
inline thrust::complex<float> proj(const thrust::complex<float>& z){
  return detail::complex::cprojf(z);
}

THRUST_NAMESPACE_END
