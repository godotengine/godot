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
#include <thrust/extrema.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/extrema.h>
#include <thrust/system/detail/adl/extrema.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator min_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last)
{
  using thrust::system::detail::generic::min_element;
  return min_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end min_element()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator min_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  using thrust::system::detail::generic::min_element;
  return min_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end min_element()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
ForwardIterator max_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last)
{
  using thrust::system::detail::generic::max_element;
  return max_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end max_element()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
ForwardIterator max_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  using thrust::system::detail::generic::max_element;
  return max_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end max_element()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last)
{
  using thrust::system::detail::generic::minmax_element;
  return minmax_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end minmax_element()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
__host__ __device__
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  using thrust::system::detail::generic::minmax_element;
  return minmax_element(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end minmax_element()


template <typename ForwardIterator>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::min_element(select_system(system), first, last);
} // end min_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::min_element(select_system(system), first, last, comp);
} // end min_element()


template <typename ForwardIterator>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::max_element(select_system(system), first, last);
} // end max_element()


template <typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(ForwardIterator first, ForwardIterator last,
                            BinaryPredicate comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::max_element(select_system(system), first, last, comp);
} // end max_element()


template <typename ForwardIterator>
thrust::pair<ForwardIterator,ForwardIterator>
minmax_element(ForwardIterator first, ForwardIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::minmax_element(select_system(system), first, last);
} // end minmax_element()


template <typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator>
minmax_element(ForwardIterator first, ForwardIterator last, BinaryPredicate comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::minmax_element(select_system(system), first, last, comp);
} // end minmax_element()

THRUST_NAMESPACE_END
