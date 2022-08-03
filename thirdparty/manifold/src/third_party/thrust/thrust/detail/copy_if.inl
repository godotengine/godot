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
#include <thrust/detail/copy_if.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/copy_if.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/adl/copy_if.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
  OutputIterator copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                         InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         Predicate pred)
{
  using thrust::system::detail::generic::copy_if;
  return copy_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, pred);
} // end copy_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
  OutputIterator copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                         InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred)
{
  using thrust::system::detail::generic::copy_if;
  return copy_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, stencil, result, pred);
} // end copy_if()


template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::copy_if(select_system(system1,system2), first, last, result, pred);
} // end copy_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type System1;
  typedef typename thrust::iterator_system<InputIterator2>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::copy_if(select_system(system1,system2,system3), first, last, stencil, result, pred);
} // end copy_if()

THRUST_NAMESPACE_END
