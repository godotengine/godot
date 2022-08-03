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

#include <thrust/uninitialized_copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/uninitialized_copy.h>
#include <thrust/system/detail/adl/uninitialized_copy.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                     InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result)
{
  using thrust::system::detail::generic::uninitialized_copy;
  return uninitialized_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result);
} // end uninitialized_copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Size, typename ForwardIterator>
__host__ __device__
  ForwardIterator uninitialized_copy_n(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       InputIterator first,
                                       Size n,
                                       ForwardIterator result)
{
  using thrust::system::detail::generic::uninitialized_copy_n;
  return uninitialized_copy_n(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, n, result);
} // end uninitialized_copy_n()


template<typename InputIterator,
         typename ForwardIterator>
  ForwardIterator uninitialized_copy(InputIterator first,
                                     InputIterator last,
                                     ForwardIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type   System1;
  typedef typename thrust::iterator_system<ForwardIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::uninitialized_copy(select_system(system1,system2), first, last, result);
} // end uninitialized_copy()


template<typename InputIterator,
         typename Size,
         typename ForwardIterator>
  ForwardIterator uninitialized_copy_n(InputIterator first,
                                       Size n,
                                       ForwardIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type   System1;
  typedef typename thrust::iterator_system<ForwardIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::uninitialized_copy_n(select_system(system1,system2), first, n, result);
} // end uninitialized_copy_n()


THRUST_NAMESPACE_END


