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
#include <thrust/sort.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/sort.h>
#include <thrust/system/detail/adl/sort.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename RandomAccessIterator>
__host__ __device__
  void sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            RandomAccessIterator first,
            RandomAccessIterator last)
{
  using thrust::system::detail::generic::sort;
  return sort(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end sort()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
  void sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::sort;
  return sort(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end sort()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename RandomAccessIterator>
__host__ __device__
  void stable_sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator last)
{
  using thrust::system::detail::generic::stable_sort;
  return stable_sort(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end stable_sort()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
  void stable_sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::stable_sort;
  return stable_sort(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end stable_sort()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
  void sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first)
{
  using thrust::system::detail::generic::sort_by_key;
  return sort_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first);
} // end sort_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
  void sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::sort_by_key;
  return sort_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, comp);
} // end sort_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
  void stable_sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first)
{
  using thrust::system::detail::generic::stable_sort_by_key;
  return stable_sort_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first);
} // end stable_sort_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
  void stable_sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::stable_sort_by_key;
  return stable_sort_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys_first, keys_last, values_first, comp);
} // end stable_sort_by_key()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  bool is_sorted(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last)
{
  using thrust::system::detail::generic::is_sorted;
  return is_sorted(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end is_sorted()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename Compare>
__host__ __device__
  bool is_sorted(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last,
                 Compare comp)
{
  using thrust::system::detail::generic::is_sorted;
  return is_sorted(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end is_sorted()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator is_sorted_until(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last)
{
  using thrust::system::detail::generic::is_sorted_until;
  return is_sorted_until(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last);
} // end is_sorted_until()


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename Compare>
__host__ __device__
  ForwardIterator is_sorted_until(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp)
{
  using thrust::system::detail::generic::is_sorted_until;
  return is_sorted_until(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, comp);
} // end is_sorted_until()


///////////////
// Key Sorts //
///////////////

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomAccessIterator>::type System;

  System system;

  return thrust::sort(select_system(system), first, last);
} // end sort()


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  __host__ __device__
  void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomAccessIterator>::type System;

  System system;

  return thrust::sort(select_system(system), first, last, comp);
} // end sort()


template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomAccessIterator>::type System;

  System system;

  return thrust::stable_sort(select_system(system), first, last);
} // end stable_sort()


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomAccessIterator>::type System;

  System system;

  return thrust::stable_sort(select_system(system), first, last, comp);
} // end stable_sort()



/////////////////////
// Key-Value Sorts //
/////////////////////

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomAccessIterator1>::type System1;
  typedef typename thrust::iterator_system<RandomAccessIterator2>::type System2;

  System1 system1;
  System2 system2;

  return thrust::sort_by_key(select_system(system1,system2), keys_first, keys_last, values_first);
} // end sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomAccessIterator1>::type System1;
  typedef typename thrust::iterator_system<RandomAccessIterator2>::type System2;

  System1 system1;
  System2 system2;

  return thrust::sort_by_key(select_system(system1,system2), keys_first, keys_last, values_first, comp);
} // end sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomAccessIterator1>::type System1;
  typedef typename thrust::iterator_system<RandomAccessIterator2>::type System2;

  System1 system1;
  System2 system2;

  return thrust::stable_sort_by_key(select_system(system1,system2), keys_first, keys_last, values_first);
} // end stable_sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomAccessIterator1>::type System1;
  typedef typename thrust::iterator_system<RandomAccessIterator2>::type System2;

  System1 system1;
  System2 system2;

  return thrust::stable_sort_by_key(select_system(system1,system2), keys_first, keys_last, values_first, comp);
} // end stable_sort_by_key()


template<typename ForwardIterator>
  bool is_sorted(ForwardIterator first,
                 ForwardIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::is_sorted(select_system(system), first, last);
} // end is_sorted()


template<typename ForwardIterator,
         typename Compare>
  bool is_sorted(ForwardIterator first,
                 ForwardIterator last,
                 Compare comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::is_sorted(select_system(system), first, last, comp);
} // end is_sorted()


template<typename ForwardIterator>
  ForwardIterator is_sorted_until(ForwardIterator first,
                                  ForwardIterator last)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::is_sorted_until(select_system(system), first, last);
} // end is_sorted_until()


template<typename ForwardIterator,
         typename Compare>
  ForwardIterator is_sorted_until(ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::is_sorted_until(select_system(system), first, last, comp);
} // end is_sorted_until()


THRUST_NAMESPACE_END

