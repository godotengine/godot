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
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/adjacent_difference.h>
#include <thrust/system/detail/adl/adjacent_difference.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator adjacent_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   InputIterator first, InputIterator last,
                                   OutputIterator result)
{
  using thrust::system::detail::generic::adjacent_difference;

  return adjacent_difference(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result);
} // end adjacent_difference()


__thrust_exec_check_disable__
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
__host__ __device__
OutputIterator adjacent_difference(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
  using thrust::system::detail::generic::adjacent_difference;

  return adjacent_difference(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, binary_op);
} // end adjacent_difference()


template <typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::adjacent_difference(select_system(system1, system2), first, last, result);
} // end adjacent_difference()


template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::adjacent_difference(select_system(system1, system2), first, last, result, binary_op);
} // end adjacent_difference()


THRUST_NAMESPACE_END
