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
    transparent_binary_operator<thrust::equal_to<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator==(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::equal_to<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator==()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::equal_to<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator==(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::equal_to<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator==()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::equal_to<>>,
    actor<T1>,
    actor<T2>
  >
>
operator==(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::equal_to<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator==()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::not_equal_to<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator!=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::not_equal_to<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator!=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::not_equal_to<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator!=(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::not_equal_to<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator!=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::not_equal_to<>>,
    actor<T1>,
    actor<T2>
  >
>
operator!=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::not_equal_to<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator!=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::greater<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator>(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::greater<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::greater<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator>(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::greater<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::greater<>>,
    actor<T1>,
    actor<T2>
  >
>
operator>(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::greater<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::less<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator<(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::less<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::less<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator<(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::less<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::less<>>,
    actor<T1>,
    actor<T2>
  >
>
operator<(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::less<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::greater_equal<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator>=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::greater_equal<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::greater_equal<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator>=(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::greater_equal<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::greater_equal<>>,
    actor<T1>,
    actor<T2>
  >
>
operator>=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::greater_equal<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::less_equal<>>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator<=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<thrust::less_equal<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::less_equal<>>,
    typename as_actor<T1>::type,
    actor<T2>
  >
>
operator<=(const T1 &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::less_equal<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<thrust::less_equal<>>,
    actor<T1>,
    actor<T2>
  >
>
operator<=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<thrust::less_equal<>>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<=()

} // end functional
} // end detail
THRUST_NAMESPACE_END

