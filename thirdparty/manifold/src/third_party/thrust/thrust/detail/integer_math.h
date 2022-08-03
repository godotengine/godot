/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
#include <limits>

#if THRUST_CPP_DIALECT >= 2011
  #include <thrust/detail/type_deduction.h>
#endif

THRUST_NAMESPACE_BEGIN

namespace detail
{

template <typename Integer>
__host__ __device__ __thrust_forceinline__
Integer clz(Integer x)
{
  Integer result;
  if (THRUST_IS_DEVICE_CODE) {
    #if THRUST_INCLUDE_DEVICE_CODE
      result = ::__clz(x);
    #endif
  } else {
    #if THRUST_INCLUDE_HOST_CODE
      int num_bits = 8 * sizeof(Integer);
      int num_bits_minus_one = num_bits - 1;
      result = num_bits;
      for (int i = num_bits_minus_one; i >= 0; --i)
      {
        if ((Integer(1) << i) & x)
        {
          result = num_bits_minus_one - i;
          break;
        }
      }
    #endif
  }
  return result;
}

template <typename Integer>
__host__ __device__ __thrust_forceinline__
bool is_power_of_2(Integer x)
{
  return 0 == (x & (x - 1));
}

template <typename Integer>
__host__ __device__ __thrust_forceinline__
bool is_odd(Integer x)
{
  return 1 & x;
}

template <typename Integer>
__host__ __device__ __thrust_forceinline__
Integer log2(Integer x)
{
  Integer num_bits = 8 * sizeof(Integer);
  Integer num_bits_minus_one = num_bits - 1;

  return num_bits_minus_one - clz(x);
}


template <typename Integer>
__host__ __device__ __thrust_forceinline__
Integer log2_ri(Integer x)
{
  Integer result = log2(x);

  // This is where we round up to the nearest log.
  if (!is_power_of_2(x))
    ++result;

  return result;
}

// x/y rounding towards +infinity for integers
// Used to determine # of blocks/warps etc.
template <typename Integer0, typename Integer1>
__host__ __device__ __thrust_forceinline__
#if THRUST_CPP_DIALECT >= 2011
// FIXME: Should use common_type.
auto divide_ri(Integer0 const x, Integer1 const y)
THRUST_DECLTYPE_RETURNS((x + (y - 1)) / y)
#else
// FIXME: Should use common_type.
Integer0 divide_ri(Integer0 const x, Integer1 const y)
{
  return (x + (y - 1)) / y;
}
#endif

// x/y rounding towards zero for integers.
// Used to determine # of blocks/warps etc.
template <typename Integer0, typename Integer1>
__host__ __device__ __thrust_forceinline__
#if THRUST_CPP_DIALECT >= 2011
auto divide_rz(Integer0 const x, Integer1 const y)
THRUST_DECLTYPE_RETURNS(x / y)
#else
// FIXME: Should use common_type.
Integer0 divide_rz(Integer0 const x, Integer1 const y)
{
  return x / y;
}
#endif

// Round x towards infinity to the next multiple of y.
template <typename Integer0, typename Integer1>
__host__ __device__ __thrust_forceinline__
#if THRUST_CPP_DIALECT >= 2011
auto round_i(Integer0 const x, Integer1 const y)
THRUST_DECLTYPE_RETURNS(y * divide_ri(x, y))
#else
Integer0 round_i(Integer0 const x, Integer1 const y)
{
  return y * divide_ri(x, y);
}
#endif

// Round x towards 0 to the next multiple of y.
template <typename Integer0, typename Integer1>
__host__ __device__ __thrust_forceinline__
#if THRUST_CPP_DIALECT >= 2011
auto round_z(Integer0 const x, Integer1 const y)
THRUST_DECLTYPE_RETURNS(y * divide_rz(x, y))
#else
Integer0 round_z(Integer0 const x, Integer1 const y)
{
  return y * divide_rz(x, y);
}
#endif

} // end detail

THRUST_NAMESPACE_END
