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
#include <limits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template<typename T>
  class integer_traits
{
  public:
    static constexpr bool is_integral = false;
};

template<typename T, T min_val, T max_val>
  class integer_traits_base
{
  public:
    static constexpr bool is_integral = true;
    static constexpr T const_min = min_val;
    static constexpr T const_max = max_val;
};


template<>
  class integer_traits<bool>
    : public std::numeric_limits<bool>,
      public integer_traits_base<bool, false, true>
{};


template<>
  class integer_traits<char>
    : public std::numeric_limits<char>,
      public integer_traits_base<char, CHAR_MIN, CHAR_MAX>
{};


template<>
  class integer_traits<signed char>
    : public std::numeric_limits<signed char>,
      public integer_traits_base<signed char, SCHAR_MIN, SCHAR_MAX>
{};


template<>
  class integer_traits<unsigned char>
    : public std::numeric_limits<unsigned char>,
      public integer_traits_base<unsigned char, 0, UCHAR_MAX>
{};


template<>
  class integer_traits<short>
    : public std::numeric_limits<short>,
      public integer_traits_base<short, SHRT_MIN, SHRT_MAX>
{};


template<>
  class integer_traits<unsigned short>
    : public std::numeric_limits<unsigned short>,
      public integer_traits_base<unsigned short, 0, USHRT_MAX>
{};


template<>
  class integer_traits<int>
    : public std::numeric_limits<int>,
      public integer_traits_base<int, INT_MIN, INT_MAX>
{};


template<>
  class integer_traits<unsigned int>
    : public std::numeric_limits<unsigned int>,
      public integer_traits_base<unsigned int, 0, UINT_MAX>
{};


template<>
  class integer_traits<long>
    : public std::numeric_limits<long>,
      public integer_traits_base<long, LONG_MIN, LONG_MAX>
{};


template<>
  class integer_traits<unsigned long>
    : public std::numeric_limits<unsigned long>,
      public integer_traits_base<unsigned long, 0, ULONG_MAX>
{};


template<>
  class integer_traits<long long>
    : public std::numeric_limits<long long>,
      public integer_traits_base<long long, LLONG_MIN, LLONG_MAX>
{};


template<>
  class integer_traits<unsigned long long>
    : public std::numeric_limits<unsigned long long>,
      public integer_traits_base<unsigned long long, 0, ULLONG_MAX>
{};

} // end detail

THRUST_NAMESPACE_END
