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
#include <thrust/system/detail/generic/find.h>
#include <thrust/system/detail/adl/find.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename T>
__host__ __device__
InputIterator find(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   InputIterator first,
                   InputIterator last,
                   const T& value)
{
  using thrust::system::detail::generic::find;
  return find(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, value);
} // end find()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
InputIterator find_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  using thrust::system::detail::generic::find_if;
  return find_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end find_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
InputIterator find_if_not(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          Predicate pred)
{
  using thrust::system::detail::generic::find_if_not;
  return find_if_not(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end find_if_not()


template <typename InputIterator, typename T>
InputIterator find(InputIterator first,
                   InputIterator last,
                   const T& value)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::find(select_system(system), first, last, value);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if(InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::find_if(select_system(system), first, last, pred);
}

template <typename InputIterator, typename Predicate>
InputIterator find_if_not(InputIterator first,
                          InputIterator last,
                          Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::find_if_not(select_system(system), first, last, pred);
}

THRUST_NAMESPACE_END
