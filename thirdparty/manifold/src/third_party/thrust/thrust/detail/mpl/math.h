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


/*! \file math.h
 *  \brief Math-related metaprogramming functionality.
 */


#pragma once

#include <thrust/detail/config.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

namespace mpl
{

namespace math
{

namespace detail
{

// compute the log base-2 of an integer at compile time
template <unsigned int N, unsigned int Cur>
struct log2
{
    static const unsigned int value = log2<N / 2,Cur+1>::value;
};

template <unsigned int Cur>
struct log2<1, Cur>
{
    static const unsigned int value = Cur;
};

template <unsigned int Cur>
struct log2<0, Cur>
{
    // undefined
};

} // end namespace detail


template <unsigned int N>
struct log2
{
    static const unsigned int value = detail::log2<N,0>::value;
};


template <typename T, T lhs, T rhs>
struct min
{
  static const T value = (lhs < rhs) ? lhs : rhs;
};


template <typename T, T lhs, T rhs>
struct max
{
  static const T value = (!(lhs < rhs)) ? lhs : rhs;
};


template<typename result_type, result_type x, result_type y>
  struct mul
{
  static const result_type value = x * y;
};


template<typename result_type, result_type x, result_type y>
  struct mod
{
  static const result_type value = x % y;
};


template<typename result_type, result_type x, result_type y>
  struct div
{
  static const result_type value = x / y;
};


template<typename result_type, result_type x, result_type y>
  struct geq
{
  static const bool value = x >= y;
};


template<typename result_type, result_type x, result_type y>
  struct lt
{
  static const bool value = x < y;
};


template<typename result_type, result_type x, result_type y>
  struct gt
{
  static const bool value = x > y;
};


template<bool x, bool y>
  struct or_
{
  static const bool value = (x || y);
};


template<typename result_type, result_type x, result_type y>
  struct bit_and
{
  static const result_type value = x & y;
};


template<typename result_type, result_type x, result_type y>
  struct plus
{
  static const result_type value = x + y;
};


template<typename result_type, result_type x, result_type y>
  struct minus
{
  static const result_type value = x - y;
};


template<typename result_type, result_type x, result_type y>
  struct equal
{
  static const bool value = x == y;
};


template<typename result_type, result_type x>
  struct is_odd
{
  static const bool value = x & 1;
};


} // end namespace math

} // end namespace mpl

} // end namespace detail

THRUST_NAMESPACE_END

