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

#include <thrust/gather.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/gather.h>
#include <thrust/system/detail/adl/gather.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator gather(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        InputIterator                                               map_first,
                        InputIterator                                               map_last,
                        RandomAccessIterator                                        input_first,
                        OutputIterator                                              result)
{
  using thrust::system::detail::generic::gather;
  return gather(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), map_first, map_last, input_first, result);
} // end gather()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator gather_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator1                                              map_first,
                           InputIterator1                                              map_last,
                           InputIterator2                                              stencil,
                           RandomAccessIterator                                        input_first,
                           OutputIterator                                              result)
{
  using thrust::system::detail::generic::gather_if;
  return gather_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), map_first, map_last, stencil, input_first, result);
} // end gather_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
  OutputIterator gather_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator1                                              map_first,
                           InputIterator1                                              map_last,
                           InputIterator2                                              stencil,
                           RandomAccessIterator                                        input_first,
                           OutputIterator                                              result,
                           Predicate                                                   pred)
{
  using thrust::system::detail::generic::gather_if;
  return gather_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), map_first, map_last, stencil, input_first, result, pred);
} // end gather_if()


template<typename InputIterator,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator gather(InputIterator        map_first,
                        InputIterator        map_last,
                        RandomAccessIterator input_first,
                        OutputIterator       result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type        System1;
  typedef typename thrust::iterator_system<RandomAccessIterator>::type System2;
  typedef typename thrust::iterator_system<OutputIterator>::type       System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::gather(select_system(system1,system2,system3), map_first, map_last, input_first, result);
} // end gather()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator>
  OutputIterator gather_if(InputIterator1       map_first,
                           InputIterator1       map_last,
                           InputIterator2       stencil,
                           RandomAccessIterator input_first,
                           OutputIterator       result)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type       System1;
  typedef typename thrust::iterator_system<InputIterator2>::type       System2;
  typedef typename thrust::iterator_system<RandomAccessIterator>::type System3;
  typedef typename thrust::iterator_system<OutputIterator>::type       System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::gather_if(select_system(system1,system2,system3,system4), map_first, map_last, stencil, input_first, result);
} // end gather_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator gather_if(InputIterator1       map_first,
                           InputIterator1       map_last,
                           InputIterator2       stencil,
                           RandomAccessIterator input_first,
                           OutputIterator       result,
                           Predicate            pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type       System1;
  typedef typename thrust::iterator_system<InputIterator2>::type       System2;
  typedef typename thrust::iterator_system<RandomAccessIterator>::type System3;
  typedef typename thrust::iterator_system<OutputIterator>::type       System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::gather_if(select_system(system1,system2,system3,system4), map_first, map_last, stencil, input_first, result, pred);
} // end gather_if()

THRUST_NAMESPACE_END
