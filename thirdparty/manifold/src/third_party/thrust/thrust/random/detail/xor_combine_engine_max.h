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
#include <thrust/detail/mpl/math.h>
#include <limits>
#include <cstddef>

THRUST_NAMESPACE_BEGIN

namespace random
{

namespace detail
{


namespace math = thrust::detail::mpl::math;


namespace detail
{

// two cases for this function avoids compile-time warnings of overflow
template<typename UIntType, UIntType w,
         UIntType lhs, UIntType rhs,
         bool shift_will_overflow>
  struct lshift_w
{
  static const UIntType value = 0;
};


template<typename UIntType, UIntType w,
         UIntType lhs, UIntType rhs>
  struct lshift_w<UIntType,w,lhs,rhs,false>
{
  static const UIntType value = lhs << rhs;
};

} // end detail


template<typename UIntType, UIntType w,
         UIntType lhs, UIntType rhs>
  struct lshift_w
{
  static const bool shift_will_overflow = rhs >= w;

  static const UIntType value = detail::lshift_w<UIntType, w, lhs, rhs, shift_will_overflow>::value;
};


template<typename UIntType, UIntType lhs, UIntType rhs>
  struct lshift
    : lshift_w<UIntType, std::numeric_limits<UIntType>::digits, lhs, rhs>
{};


template<typename UIntType, int p>
  struct two_to_the_power
    : lshift<UIntType, 1, p>
{};


template<typename result_type, result_type a, result_type b, int d>
  class xor_combine_engine_max_aux_constants
{
  public:
    static const result_type two_to_the_d = two_to_the_power<result_type, d>::value;
    static const result_type c = lshift<result_type, a, d>::value;

    static const result_type t =
      math::max<
        result_type,
        c,
        b
      >::value;

    static const result_type u =
      math::min<
        result_type,
        c,
        b
      >::value;

    static const result_type p            = math::log2<u>::value;
    static const result_type two_to_the_p = two_to_the_power<result_type, p>::value;

    static const result_type k = math::div<result_type, t, two_to_the_p>::value;
};


template<typename result_type, result_type, result_type, int> struct xor_combine_engine_max_aux;


template<typename result_type, result_type a, result_type b, int d>
  struct xor_combine_engine_max_aux_case4
{
  typedef xor_combine_engine_max_aux_constants<result_type,a,b,d> constants;

  static const result_type k_plus_1_times_two_to_the_p =
    lshift<
      result_type,
      math::plus<result_type,constants::k,1>::value,
      constants::p
    >::value;

  static const result_type M =
    xor_combine_engine_max_aux<
      result_type,
      math::div<
        result_type,
        math::mod<
          result_type,
          constants::u,
          constants::two_to_the_p
        >::value,
        constants::two_to_the_p
      >::value,
      math::mod<
        result_type,
        constants::t,
        constants::two_to_the_p
      >::value,
      d
    >::value;

  static const result_type value = math::plus<result_type, k_plus_1_times_two_to_the_p, M>::value;
};


template<typename result_type, result_type a, result_type b, int d>
  struct xor_combine_engine_max_aux_case3
{
  typedef xor_combine_engine_max_aux_constants<result_type,a,b,d> constants;

  static const result_type k_plus_1_times_two_to_the_p =
    lshift<
      result_type,
      math::plus<result_type,constants::k,1>::value,
      constants::p
    >::value;

  static const result_type M =
    xor_combine_engine_max_aux<
      result_type,
      math::div<
        result_type,
        math::mod<
          result_type,
          constants::t,
          constants::two_to_the_p
        >::value,
        constants::two_to_the_p
      >::value,
      math::mod<
        result_type,
        constants::u,
        constants::two_to_the_p
      >::value,
      d
    >::value;

  static const result_type value = math::plus<result_type, k_plus_1_times_two_to_the_p, M>::value;
};



template<typename result_type, result_type a, result_type b, int d>
  struct xor_combine_engine_max_aux_case2
{
  typedef xor_combine_engine_max_aux_constants<result_type,a,b,d> constants;

  static const result_type k_plus_1_times_two_to_the_p =
    lshift<
      result_type,
      math::plus<result_type,constants::k,1>::value,
      constants::p
    >::value;

  static const result_type value =
    math::minus<
      result_type,
      k_plus_1_times_two_to_the_p,
      1
    >::value;
};


template<typename result_type, result_type a, result_type b, int d>
  struct xor_combine_engine_max_aux_case1
{
  static const result_type c     = lshift<result_type, a, d>::value;

  static const result_type value = math::plus<result_type,c,b>::value;
};


template<typename result_type, result_type a, result_type b, int d>
  struct xor_combine_engine_max_aux_2
{
  typedef xor_combine_engine_max_aux_constants<result_type,a,b,d> constants;

  static const result_type value = 
    thrust::detail::eval_if<
      // if k is odd...
      math::is_odd<result_type, constants::k>::value,
      thrust::detail::identity_<
        thrust::detail::integral_constant<
          result_type,
          xor_combine_engine_max_aux_case2<result_type,a,b,d>::value
        >
      >,
      thrust::detail::eval_if<
        // otherwise if a * 2^3 >= b, then case 3
        a * constants::two_to_the_d >= b,
        thrust::detail::identity_<
          thrust::detail::integral_constant<
            result_type,
            xor_combine_engine_max_aux_case3<result_type,a,b,d>::value
          >
        >,
        // otherwise, case 4
        thrust::detail::identity_<
          thrust::detail::integral_constant<
            result_type,
            xor_combine_engine_max_aux_case4<result_type,a,b,d>::value
          >
        >
      >
    >::type::value;
};


template<typename result_type,
         result_type a,
         result_type b,
         int d,
         bool use_case1 = (a == 0) || (b < two_to_the_power<result_type,d>::value)>
  struct xor_combine_engine_max_aux_1
    : xor_combine_engine_max_aux_case1<result_type,a,b,d>
{};


template<typename result_type,
         result_type a,
         result_type b,
         int d>
  struct xor_combine_engine_max_aux_1<result_type,a,b,d,false>
    : xor_combine_engine_max_aux_2<result_type,a,b,d>
{};


template<typename result_type,
         result_type a,
         result_type b,
         int d>
  struct xor_combine_engine_max_aux
    : xor_combine_engine_max_aux_1<result_type,a,b,d>
{};


template<typename Engine1, size_t s1, typename Engine2, size_t s2, typename result_type>
  struct xor_combine_engine_max
{
  static const size_t w = std::numeric_limits<result_type>::digits;

  static const result_type m1 =
    math::min<
      result_type,
      result_type(Engine1::max - Engine1::min),
      two_to_the_power<result_type, w-s1>::value - 1 
    >::value;

  static const result_type m2 =
    math::min<
      result_type,
      result_type(Engine2::max - Engine2::min),
      two_to_the_power<result_type, w-s2>::value - 1
    >::value;

  static const result_type s = s1 - s2;

  static const result_type M =
    xor_combine_engine_max_aux<
      result_type,
      m1,
      m2,
      s
    >::value;

  // the value is M(m1,m2,s) lshift_w s2
  static const result_type value =
    lshift_w<
      result_type,
      w,
      M,
      s2
    >::value;
}; // end xor_combine_engine_max

} // end detail

} // end random

THRUST_NAMESPACE_END

