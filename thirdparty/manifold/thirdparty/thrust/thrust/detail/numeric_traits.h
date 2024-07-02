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
#include <thrust/detail/type_traits.h>
#include <limits>

//#include <stdint.h> // for intmax_t (not provided on MSVS 2005)

THRUST_NAMESPACE_BEGIN

namespace detail
{

// XXX good enough for the platforms we care about
typedef long long intmax_t;

template<typename Number>
  struct is_signed
    : integral_constant<bool, std::numeric_limits<Number>::is_signed>
{}; // end is_signed


template<typename T>
  struct num_digits
    : eval_if<
        std::numeric_limits<T>::is_specialized,
        integral_constant<
          int,
          std::numeric_limits<T>::digits
        >,
        integral_constant<
          int,
          sizeof(T) * std::numeric_limits<unsigned char>::digits - (is_signed<T>::value ? 1 : 0)  
        >
      >::type
{}; // end num_digits


template<typename Integer>
  struct integer_difference
    //: eval_if<
    //    sizeof(Integer) >= sizeof(intmax_t),
    //    eval_if<
    //      is_signed<Integer>::value,
    //      identity_<Integer>,
    //      identity_<intmax_t>
    //    >,
    //    eval_if<
    //      sizeof(Integer) < sizeof(std::ptrdiff_t),
    //      identity_<std::ptrdiff_t>,
    //      identity_<intmax_t>
    //    >
    //  >
{
  private:
    // XXX workaround a pedantic warning in old versions of g++
    //     which complains about &&ing with a constant value
    template<bool x, bool y>
      struct and_
    {
      static const bool value = false;
    };

    template<bool y>
      struct and_<true,y>
    {
      static const bool value = y;
    };

  public:
    typedef typename
      eval_if<
        and_<
          std::numeric_limits<Integer>::is_signed,
          // digits is the number of no-sign bits
          (!std::numeric_limits<Integer>::is_bounded || (int(std::numeric_limits<Integer>::digits) + 1 >= num_digits<intmax_t>::value))
        >::value,
        identity_<Integer>,
        eval_if<
          int(std::numeric_limits<Integer>::digits) + 1 < num_digits<signed int>::value,
          identity_<signed int>,
          eval_if<
            int(std::numeric_limits<Integer>::digits) + 1 < num_digits<signed long>::value,
            identity_<signed long>,
            identity_<intmax_t>
          >
        >
      >::type type;
}; // end integer_difference


template<typename Number>
  struct numeric_difference
    : eval_if<
      is_integral<Number>::value,
      integer_difference<Number>,
      identity_<Number>
    >
{}; // end numeric_difference


template<typename Number>
__host__ __device__
typename numeric_difference<Number>::type
numeric_distance(Number x, Number y)
{
  typedef typename numeric_difference<Number>::type difference_type;
  return difference_type(y) - difference_type(x);
} // end numeric_distance

} // end detail

THRUST_NAMESPACE_END
