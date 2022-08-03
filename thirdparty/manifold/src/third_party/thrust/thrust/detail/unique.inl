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
#include <thrust/unique.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/unique.h>
#include <thrust/system/detail/generic/unique_by_key.h>
#include <thrust/system/detail/adl/unique.h>
#include <thrust/system/detail/adl/unique_by_key.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator>
__host__ __device__
ForwardIterator unique(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       ForwardIterator first,
                       ForwardIterator last)
{
  using thrust::system::detail::generic::unique;
  return unique(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end unique()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
__host__ __device__
ForwardIterator unique(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       ForwardIterator first,
                       ForwardIterator last,
                       BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::unique;
  return unique(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, binary_pred);
} // end unique()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
OutputIterator unique_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator output)
{
  using thrust::system::detail::generic::unique_copy;
  return unique_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, output);
} // end unique_copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
OutputIterator unique_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                           InputIterator first,
                           InputIterator last,
                           OutputIterator output,
                           BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::unique_copy;
  return unique_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, output, binary_pred);
} // end unique_copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2>
__host__ __device__
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator1 keys_first,
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first)
{
  using thrust::system::detail::generic::unique_by_key;
  return unique_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first);
} // end unique_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<ForwardIterator1,ForwardIterator2>
  unique_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator1 keys_first,
                ForwardIterator1 keys_last,
                ForwardIterator2 values_first,
                BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::unique_by_key;
  return unique_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, binary_pred);
} // end unique_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_by_key_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                     InputIterator1 keys_first,
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output)
{
  using thrust::system::detail::generic::unique_by_key_copy;
  return unique_by_key_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, keys_output, values_output);
} // end unique_by_key_copy()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
  unique_by_key_copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                     InputIterator1 keys_first,
                     InputIterator1 keys_last,
                     InputIterator2 values_first,
                     OutputIterator1 keys_output,
                     OutputIterator2 values_output,
                     BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::unique_by_key_copy;
  return unique_by_key_copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
} // end unique_by_key_copy()


template<typename ForwardIterator>
  ForwardIterator unique(ForwardIterator first,
                         ForwardIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::unique(select_system(system), first, last);
} // end unique()


template<typename ForwardIterator,
         typename BinaryPredicate>
  ForwardIterator unique(ForwardIterator first,
                         ForwardIterator last,
                         BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::unique(select_system(system), first, last, binary_pred);
} // end unique()


template<typename InputIterator,
         typename OutputIterator>
  OutputIterator unique_copy(InputIterator first,
                             InputIterator last,
                             OutputIterator output)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::unique_copy(select_system(system1,system2), first, last, output);
} // end unique_copy()


template<typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
  OutputIterator unique_copy(InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type  System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::unique_copy(select_system(system1,system2), first, last, output, binary_pred);
} // end unique_copy()


template<typename ForwardIterator1,
         typename ForwardIterator2>
  thrust::pair<ForwardIterator1,ForwardIterator2>
    unique_by_key(ForwardIterator1 keys_first,
                  ForwardIterator1 keys_last,
                  ForwardIterator2 values_first)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator1>::type System1;
  typedef typename thrust::iterator_system<ForwardIterator2>::type System2;

  System1 system1;
  System2 system2;

  return thrust::unique_by_key(select_system(system1,system2), keys_first, keys_last, values_first);
} // end unique_by_key()


template<typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
  thrust::pair<ForwardIterator1,ForwardIterator2>
    unique_by_key(ForwardIterator1 keys_first,
                  ForwardIterator1 keys_last,
                  ForwardIterator2 values_first,
                  BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator1>::type System1;
  typedef typename thrust::iterator_system<ForwardIterator2>::type System2;

  System1 system1;
  System2 system2;

  return thrust::unique_by_key(select_system(system1,system2), keys_first, keys_last, values_first, binary_pred);
} // end unique_by_key()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(InputIterator1 keys_first,
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_output,
                       OutputIterator2 values_output)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<OutputIterator1>::type System3;
  typedef typename thrust::iterator_system<OutputIterator2>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::unique_by_key_copy(select_system(system1,system2,system3,system4), keys_first, keys_last, values_first, keys_output, values_output);
} // end unique_by_key_copy()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(InputIterator1 keys_first,
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_output,
                       OutputIterator2 values_output,
                       BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator1>::type  System1;
  typedef typename thrust::iterator_system<InputIterator2>::type  System2;
  typedef typename thrust::iterator_system<OutputIterator1>::type System3;
  typedef typename thrust::iterator_system<OutputIterator2>::type System4;

  System1 system1;
  System2 system2;
  System3 system3;
  System4 system4;

  return thrust::unique_by_key_copy(select_system(system1,system2,system3,system4), keys_first, keys_last, values_first, keys_output, values_output, binary_pred);
} // end unique_by_key_copy()

__thrust_exec_check_disable__
template <typename DerivedPolicy,
          typename ForwardIterator,
          typename BinaryPredicate>
__host__ __device__
    typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last,
                 BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::unique_count;
  return unique_count(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, binary_pred);
} // end unique_count()

__thrust_exec_check_disable__
template <typename DerivedPolicy,
          typename ForwardIterator>
__host__ __device__
    typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last)
{
  using thrust::system::detail::generic::unique_count;
  return unique_count(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end unique_count()

__thrust_exec_check_disable__
template <typename ForwardIterator,
          typename BinaryPredicate>
__host__ __device__
    typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(ForwardIterator first,
                 ForwardIterator last,
                 BinaryPredicate binary_pred)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::unique_count(select_system(system), first, last, binary_pred);
} // end unique_count()

__thrust_exec_check_disable__
template <typename ForwardIterator>
__host__ __device__
    typename thrust::iterator_traits<ForwardIterator>::difference_type
    unique_count(ForwardIterator first,
                 ForwardIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::unique_count(select_system(system), first, last);
} // end unique_count()

THRUST_NAMESPACE_END

