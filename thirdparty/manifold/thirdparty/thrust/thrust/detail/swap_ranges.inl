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

#include <thrust/swap.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/swap_ranges.h>
#include <thrust/system/detail/adl/swap_ranges.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2>
__host__ __device__
  ForwardIterator2 swap_ranges(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2)
{
  using thrust::system::detail::generic::swap_ranges;
  return swap_ranges(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2);
} // end swap_ranges()


template<typename ForwardIterator1,
         typename ForwardIterator2>
  ForwardIterator2 swap_ranges(ForwardIterator1 first1,
                               ForwardIterator1 last1,
                               ForwardIterator2 first2)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator1>::type System1;
  typedef typename thrust::iterator_system<ForwardIterator2>::type System2;

  System1 system1;
  System2 system2;

  return thrust::swap_ranges(select_system(system1,system2), first1, last1, first2);
} // end swap_ranges()


THRUST_NAMESPACE_END

