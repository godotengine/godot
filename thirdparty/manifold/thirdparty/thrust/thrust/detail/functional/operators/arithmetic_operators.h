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
#include <thrust/detail/functional/actor.h>
#include <thrust/detail/functional/composite.h>
#include <thrust/detail/functional/operators/operator_adaptors.h>
#include <thrust/functional.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace functional
{

template<typename Eval>
__host__ __device__
actor<
  composite<
    transparent_unary_operator<thrust::negate<>>,
    actor<Eval>
  >
>
__host__ __device__
operator-(const actor<Eval> &_1)
{
  return compose(transparent_unary_operator<thrust::negate<>>(), _1);
} // end operator-()

// there's no standard unary_plus functional, so roll an ad hoc one here
struct unary_plus
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1>
  __host__ __device__
  constexpr auto operator()(T1&& t1) const
  noexcept(noexcept(+THRUST_FWD(t1)))
  THRUST_TRAILING_RETURN(decltype(+THRUST_FWD(t1)))
  {
    return +THRUST_FWD(t1);
  }
};

template<typename Eval>
__host__ __device__
actor<
  composite<
    transparent_unary_operator<unary_plus>,
    actor<Eval>
  >
>
operator+(const actor<Eval> &_1)
{
  return compose(transparent_unary_operator<unary_plus>(), _1);
} // end operator+()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::plus<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator+(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::plus<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator+()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::plus<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator+(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::plus<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator+()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::plus<>>,
    actor<T1>,
    actor<T2>
  >
>
operator+(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::plus<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator+()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::minus<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator-(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::minus<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator-()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::minus<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator-(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::minus<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator-()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::minus<>>,
    actor<T1>,
    actor<T2>
  >
>
operator-(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::minus<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator-()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::multiplies<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator*(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::multiplies<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator*()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::multiplies<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator*(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::multiplies<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator*()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::multiplies<>>,
    actor<T1>,
    actor<T2>
  >
>
operator*(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::multiplies<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator*()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::divides<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator/(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::divides<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator/()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::divides<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator/(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::divides<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator/()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::divides<>>,
    actor<T1>,
    actor<T2>
  >
>
operator/(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::divides<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator/()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::modulus<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator%(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::modulus<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator%()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::modulus<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator%(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::modulus<void>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator%()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::modulus<>>,
    actor<T1>,
    actor<T2>
  >
>
operator%(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::modulus<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator%()

// there's no standard prefix_increment functional, so roll an ad hoc one here
struct prefix_increment
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1>
  __host__ __device__
  constexpr auto operator()(T1&& t1) const
  noexcept(noexcept(++THRUST_FWD(t1)))
  THRUST_TRAILING_RETURN(decltype(++THRUST_FWD(t1)))
  {
    return ++THRUST_FWD(t1);
  }
}; // end prefix_increment

template<typename Eval>
__host__ __device__
actor<
  composite<
    transparent_unary_operator<prefix_increment>,
    actor<Eval>
  >
>
operator++(const actor<Eval> &_1)
{
  return compose(transparent_unary_operator<prefix_increment>(), _1);
} // end operator++()


// there's no standard postfix_increment functional, so roll an ad hoc one here
struct postfix_increment
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1>
  __host__ __device__
  constexpr auto operator()(T1&& t1) const
  noexcept(noexcept(THRUST_FWD(t1)++))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1)++))
  {
    return THRUST_FWD(t1)++;
  }
}; // end postfix_increment

template<typename Eval>
__host__ __device__
actor<
  composite<
    transparent_unary_operator<postfix_increment>,
    actor<Eval>
  >
>
operator++(const actor<Eval> &_1, int)
{
  return compose(transparent_unary_operator<postfix_increment>(), _1);
} // end operator++()


// there's no standard prefix_decrement functional, so roll an ad hoc one here
struct prefix_decrement
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1>
  __host__ __device__
  constexpr auto operator()(T1&& t1) const
  noexcept(noexcept(--THRUST_FWD(t1)))
  THRUST_TRAILING_RETURN(decltype(--THRUST_FWD(t1)))
  {
    return --THRUST_FWD(t1);
  }
}; // end prefix_decrement

template<typename Eval>
__host__ __device__
actor<
  composite<
    transparent_unary_operator<prefix_decrement>,
    actor<Eval>
  >
>
operator--(const actor<Eval> &_1)
{
  return compose(transparent_unary_operator<prefix_decrement>(), _1);
} // end operator--()


// there's no standard postfix_decrement functional, so roll an ad hoc one here
struct postfix_decrement
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1>
  __host__ __device__
  constexpr auto operator()(T1&& t1) const
  noexcept(noexcept(THRUST_FWD(t1)--))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1)--))
  {
    return THRUST_FWD(t1)--;
  }
}; // end prefix_increment

template<typename Eval>
__host__ __device__
actor<
  composite<
    transparent_unary_operator<postfix_decrement>,
    actor<Eval>
  >
>
operator--(const actor<Eval> &_1, int)
{
  return compose(transparent_unary_operator<postfix_decrement>(), _1);
} // end operator--()

} // end functional
} // end detail
THRUST_NAMESPACE_END

