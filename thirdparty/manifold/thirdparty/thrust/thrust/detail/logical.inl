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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/logical.h>
#include <thrust/system/detail/adl/logical.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
bool all_of(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::all_of;
  return all_of(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end all_of()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
bool any_of(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::any_of;
  return any_of(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end any_of()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
bool none_of(const thrust::detail::execution_policy_base<DerivedPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::none_of;
  return none_of(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end none_of()


template<typename InputIterator, typename Predicate>
bool all_of(InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::all_of(select_system(system), first, last, pred);
}


template<typename InputIterator, typename Predicate>
bool any_of(InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::any_of(select_system(system), first, last, pred);
}


template<typename InputIterator, typename Predicate>
bool none_of(InputIterator first, InputIterator last, Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::none_of(select_system(system), first, last, pred);
}

THRUST_NAMESPACE_END
