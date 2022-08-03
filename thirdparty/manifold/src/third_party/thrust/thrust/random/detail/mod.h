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

THRUST_NAMESPACE_BEGIN

namespace random
{

namespace detail
{

template<typename T, T a, T c, T m, bool = (m == 0)>
  struct static_mod
{
  static const T q = m / a;
  static const T r = m % a;

  __host__ __device__
  T operator()(T x) const
  {
    THRUST_IF_CONSTEXPR(a == 1)
    {
      x %= m;
    }
    else
    {
      T t1 = a * (x % q);
      T t2 = r * (x / q);
      if(t1 >= t2)
      {
        x = t1 - t2;
      }
      else
      {
        x = m - t2 + t1;
      }
    }

    THRUST_IF_CONSTEXPR(c != 0)
    {
      const T d = m - x;
      if(d > c)
      {
        x += c;
      }
      else
      {
        x = c - d;
      }
    }

    return x;
  }
}; // end static_mod


// Rely on machine overflow handling
template<typename T, T a, T c, T m>
  struct static_mod<T,a,c,m,true>
{
  __host__ __device__
  T operator()(T x) const
  {
    return a * x + c;
  }
}; // end static_mod

template<typename T, T a, T c, T m>
__host__ __device__
  T mod(T x)
{
  static_mod<T,a,c,m> f;
  return f(x);
} // end static_mod

} // end detail

} // end random

THRUST_NAMESPACE_END

