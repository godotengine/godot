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
#include <thrust/partition.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/partition.h>
#include <thrust/system/detail/adl/partition.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  using thrust::system::detail::generic::partition;
  return partition(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end partition()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  using thrust::system::detail::generic::partition;
  return partition(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, stencil, pred);
} // end partition()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  using thrust::system::detail::generic::partition_copy;
  return partition_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, out_true, out_false, pred);
} // end partition_copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   InputIterator1 first,
                   InputIterator1 last,
                   InputIterator2 stencil,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  using thrust::system::detail::generic::partition_copy;
  return partition_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, stencil, out_true, out_false, pred);
} // end partition_copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  using thrust::system::detail::generic::stable_partition;
  return stable_partition(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end stable_partition()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred)
{
  using thrust::system::detail::generic::stable_partition;
  return stable_partition(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, stencil, pred);
} // end stable_partition()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  using thrust::system::detail::generic::stable_partition_copy;
  return stable_partition_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, out_true, out_false, pred);
} // end stable_partition_copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  using thrust::system::detail::generic::stable_partition_copy;
  return stable_partition_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, stencil, out_true, out_false, pred);
} // end stable_partition_copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename Predicate>
__host__ __device__
  ForwardIterator partition_point(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred)
{
  using thrust::system::detail::generic::partition_point;
  return partition_point(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end partition_point()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
  bool is_partitioned(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  using thrust::system::detail::generic::is_partitioned;
  return is_partitioned(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, pred);
} // end is_partitioned()


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::partition(select_system(system), first, last, pred);
} // end partition()


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator partition(ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System1;
  typedef typename thrust::iterator_system<InputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::partition(select_system(system1,system2), first, last, stencil, pred);
} // end partition()


template<typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::stable_partition(select_system(system), first, last, pred);
} // end stable_partition()


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator stable_partition(ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System1;
  typedef typename thrust::iterator_system<InputIterator>::type   System2;

  System1 system1;
  System2 system2;

  return thrust::stable_partition(select_system(system1,system2), first, last, stencil, pred);
} // end stable_partition()


template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type   System1;
  typedef typename thrust::iterator_system<OutputIterator1>::type System2;
  typedef typename thrust::iterator_system<OutputIterator2>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::partition_copy(select_system(system1,system2,system3), first, last, out_true, out_false, pred);
} // end partition_copy()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(InputIterator1 first,
                   InputIterator1 last,
                   InputIterator2 stencil,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator1>::type  System2;
  typedef typename thrust::iterator_system<OutputIterator1>::type System3;
  typedef typename thrust::iterator_system<OutputIterator2>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::partition_copy(select_system(system1,system2,system3,system4), first, last, stencil, out_true, out_false, pred);
} // end partition_copy()


template<typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type   System1;
  typedef typename thrust::iterator_system<OutputIterator1>::type System2;
  typedef typename thrust::iterator_system<OutputIterator2>::type System3;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::stable_partition_copy(select_system(system1,system2,system3), first, last, out_true, out_false, pred);
} // end stable_partition_copy()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type   System1;
  typedef typename thrust::iterator_system<InputIterator2>::type   System2;
  typedef typename thrust::iterator_system<OutputIterator1>::type  System3;
  typedef typename thrust::iterator_system<OutputIterator2>::type  System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::stable_partition_copy(select_system(system1,system2,system3,system4), first, last, stencil, out_true, out_false, pred);
} // end stable_partition_copy()


template<typename ForwardIterator, typename Predicate>
  ForwardIterator partition_point(ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::partition_point(select_system(system), first, last, pred);
} // end partition_point()


template<typename InputIterator, typename Predicate>
  bool is_partitioned(InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::is_partitioned(select_system(system), first, last, pred);
} // end is_partitioned()

THRUST_NAMESPACE_END
