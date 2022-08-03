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

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_and<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator&(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_and<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator&()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_and<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator&(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_and<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator&()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_and<>>,
    actor<T1>,
    actor<T2>
  >
>
operator&(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_and<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator&()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_or<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator|(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_or<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator|()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_or<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator|(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_or<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator|()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_or<>>,
    actor<T1>,
    actor<T2>
  >
>
operator|(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_or<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator|()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_xor<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator^(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_xor<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator^()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_xor<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator^(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_xor<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator^()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_xor<>>,
    actor<T1>,
    actor<T2>
  >
>
operator^(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_xor<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator^()


// there's no standard bit_not functional, so roll an ad hoc one here
struct bit_not
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1>
  __host__ __device__
  constexpr auto operator()(T1&& t1) const
  noexcept(noexcept(~THRUST_FWD(t1)))
  THRUST_TRAILING_RETURN(decltype(~THRUST_FWD(t1)))
  {
    return ~THRUST_FWD(t1);
  }
}; // end prefix_increment

template<typename Eval>
__host__ __device__
actor<
  composite<
    transparent_unary_operator<bit_not>,
    actor<Eval>
  >
>
__host__ __device__
operator~(const actor<Eval> &_1)
{
  return compose(transparent_unary_operator<bit_not>(), _1);
} // end operator~()

// there's no standard bit_lshift functional, so roll an ad hoc one here
struct bit_lshift
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) << THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) << THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) << THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_lshift>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator<<(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_lshift>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<<()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_lshift>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator<<(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_lshift>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<<()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_lshift>,
    actor<T1>,
    actor<T2>
  >
>
operator<<(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_lshift>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<<()

// there's no standard bit_rshift functional, so roll an ad hoc one here
struct bit_rshift
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) >> THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) >> THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) >> THRUST_FWD(t2);
  }
};


template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_rshift>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator>>(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_rshift>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>>()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_rshift>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator>>(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_rshift>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>>()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_rshift>,
    actor<T1>,
    actor<T2>
  >
>
operator>>(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_rshift>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>>()

} // end functional
} // end detail
THRUST_NAMESPACE_END

