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
#include <thrust/reverse.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/reverse.h>
#include <thrust/system/detail/adl/reverse.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename BidirectionalIterator>
__host__ __device__
  void reverse(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               BidirectionalIterator first,
               BidirectionalIterator last)
{
  using thrust::system::detail::generic::reverse;
  return reverse(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end reverse()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename BidirectionalIterator, typename OutputIterator>
__host__ __device__
  OutputIterator reverse_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                              BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result)
{
  using thrust::system::detail::generic::reverse_copy;
  return reverse_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result);
} // end reverse_copy()


template<typename BidirectionalIterator>
  void reverse(BidirectionalIterator first,
               BidirectionalIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<BidirectionalIterator>::type System;

  System system;

  return thrust::reverse(select_system(system), first, last);
} // end reverse()


template<typename BidirectionalIterator,
         typename OutputIterator>
  OutputIterator reverse_copy(BidirectionalIterator first,
                              BidirectionalIterator last,
                              OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<BidirectionalIterator>::type System1;
  typedef typename thrust::iterator_system<OutputIterator>::type        System2;

  System1 system1;
  System2 system2;

  return thrust::reverse_copy(select_system(system1,system2), first, last, result);
} // end reverse_copy()


THRUST_NAMESPACE_END

