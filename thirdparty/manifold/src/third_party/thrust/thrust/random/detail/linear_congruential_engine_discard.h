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

#include <thrust/detail/cstdint.h>
#include <thrust/random/detail/mod.h>

THRUST_NAMESPACE_BEGIN

namespace random
{

namespace detail
{


template<typename UIntType, UIntType a, unsigned long long c, UIntType m>
  struct linear_congruential_engine_discard_implementation
{
  __host__ __device__
  static void discard(UIntType &state, unsigned long long z)
  {
    for(; z > 0; --z)
    {
      state = detail::mod<UIntType,a,c,m>(state);
    }
  }
}; // end linear_congruential_engine_discard


// specialize for small integers and c == 0
// XXX figure out a robust implemenation of this for any unsigned integer type later
template<thrust::detail::uint32_t a, thrust::detail::uint32_t m>
  struct linear_congruential_engine_discard_implementation<thrust::detail::uint32_t,a,0,m>
{
  __host__ __device__
  static void discard(thrust::detail::uint32_t &state, unsigned long long z)
  {
    const thrust::detail::uint32_t modulus = m;

    // XXX we need to use unsigned long long here or we will encounter overflow in the
    //     multiplies below
    //     figure out a robust implementation of this later
    unsigned long long multiplier = a;
    unsigned long long multiplier_to_z = 1;
    
    // see http://en.wikipedia.org/wiki/Modular_exponentiation
    while(z > 0)
    {
      if(z & 1)
      {
        // multiply in this bit's contribution while using modulus to keep result small
        multiplier_to_z = (multiplier_to_z * multiplier) % modulus;
      }

      // move to the next bit of the exponent, square (and mod) the base accordingly
      z >>= 1;
      multiplier = (multiplier * multiplier) % modulus;
    }

    state = static_cast<thrust::detail::uint32_t>((multiplier_to_z * state) % modulus);
  }
}; // end linear_congruential_engine_discard


struct linear_congruential_engine_discard
{
  template<typename LinearCongruentialEngine>
  __host__ __device__
  static void discard(LinearCongruentialEngine &lcg, unsigned long long z)
  {
    typedef typename LinearCongruentialEngine::result_type result_type;
    const result_type c = LinearCongruentialEngine::increment;
    const result_type a = LinearCongruentialEngine::multiplier;
    const result_type m = LinearCongruentialEngine::modulus;
    
    // XXX WAR unused variable warnings
    (void) c;
    (void) a;
    (void) m;

    linear_congruential_engine_discard_implementation<result_type,a,c,m>::discard(lcg.m_x, z);
  }
}; // end linear_congruential_engine_discard


} // end detail

} // end random

THRUST_NAMESPACE_END

