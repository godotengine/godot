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

#include <thrust/scatter.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/scatter.h>
#include <thrust/system/detail/adl/scatter.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
__host__ __device__
  void scatter(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  using thrust::system::detail::generic::scatter;
  return scatter(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, map, output);
} // end scatter()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
__host__ __device__
  void scatter_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output)
{
  using thrust::system::detail::generic::scatter_if;
  return scatter_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, map, stencil, output);
} // end scatter_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
__host__ __device__
  void scatter_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred)
{
  using thrust::system::detail::generic::scatter_if;
  return scatter_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, map, stencil, output, pred);
} // end scatter_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type       System1;
  typedef typename thrust::iterator_system<InputIterator2>::type       System2;
  typedef typename thrust::iterator_system<RandomAccessIterator>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::scatter(select_system(system1,system2,system3), first, last, map, output);
} // end scatter()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
  void scatter_if(InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type       System1;
  typedef typename thrust::iterator_system<InputIterator2>::type       System2;
  typedef typename thrust::iterator_system<InputIterator3>::type       System3;
  typedef typename thrust::iterator_system<RandomAccessIterator>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::scatter_if(select_system(system1,system2,system3,system4), first, last, map, stencil, output);
} // end scatter_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator,
         typename Predicate>
  void scatter_if(InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output,
                  Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type       System1;
  typedef typename thrust::iterator_system<InputIterator2>::type       System2;
  typedef typename thrust::iterator_system<InputIterator3>::type       System3;
  typedef typename thrust::iterator_system<RandomAccessIterator>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::scatter_if(select_system(system1,system2,system3,system4), first, last, map, stencil, output, pred);
} // end scatter_if()

THRUST_NAMESPACE_END

