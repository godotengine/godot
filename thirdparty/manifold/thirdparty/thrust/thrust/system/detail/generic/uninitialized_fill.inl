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
#include <thrust/system/detail/generic/uninitialized_fill.h>
#include <thrust/fill.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T>
__host__ __device__
  void uninitialized_fill(thrust::execution_policy<DerivedPolicy> &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          const T &x,
                          thrust::detail::true_type) // has_trivial_copy_constructor
{
  thrust::fill(exec, first, last, x);
} // end uninitialized_fill()

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T>
__host__ __device__
  void uninitialized_fill(thrust::execution_policy<DerivedPolicy> &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          const T &x,
                          thrust::detail::false_type) // has_trivial_copy_constructor
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  thrust::for_each(exec, first, last, thrust::detail::uninitialized_fill_functor<ValueType>(x));
} // end uninitialized_fill()

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Size,
         typename T>
__host__ __device__
  ForwardIterator uninitialized_fill_n(thrust::execution_policy<DerivedPolicy> &exec,
                                       ForwardIterator first,
                                       Size n,
                                       const T &x,
                                       thrust::detail::true_type) // has_trivial_copy_constructor
{
  return thrust::fill_n(exec, first, n, x);
} // end uninitialized_fill()

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Size,
         typename T>
__host__ __device__
  ForwardIterator uninitialized_fill_n(thrust::execution_policy<DerivedPolicy> &exec,
                                       ForwardIterator first,
                                       Size n,
                                       const T &x,
                                       thrust::detail::false_type) // has_trivial_copy_constructor
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  return thrust::for_each_n(exec, first, n, thrust::detail::uninitialized_fill_functor<ValueType>(x));
} // end uninitialized_fill()

} // end detail

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename T>
__host__ __device__
  void uninitialized_fill(thrust::execution_policy<DerivedPolicy> &exec,
                          ForwardIterator first,
                          ForwardIterator last,
                          const T &x)
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  typedef thrust::detail::has_trivial_copy_constructor<ValueType> ValueTypeHasTrivialCopyConstructor;

  thrust::system::detail::generic::detail::uninitialized_fill(exec, first, last, x,
    ValueTypeHasTrivialCopyConstructor());
} // end uninitialized_fill()

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Size,
         typename T>
__host__ __device__
  ForwardIterator uninitialized_fill_n(thrust::execution_policy<DerivedPolicy> &exec,
                                       ForwardIterator first,
                                       Size n,
                                       const T &x)
{
  typedef typename iterator_traits<ForwardIterator>::value_type ValueType;

  typedef thrust::detail::has_trivial_copy_constructor<ValueType> ValueTypeHasTrivialCopyConstructor;

  return thrust::system::detail::generic::detail::uninitialized_fill_n(exec, first, n, x,
    ValueTypeHasTrivialCopyConstructor());
} // end uninitialized_fill()

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

