/*
 *  Copyright 2008-2021 NVIDIA Corporation
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
#include <thrust/detail/temporary_array.h>
#include <thrust/merge.h>
#include <thrust/system/detail/sequential/insertion_sort.h>
#include <thrust/detail/minmax.h>

#include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{
namespace stable_merge_sort_detail
{


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void inplace_merge(sequential::execution_policy<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator middle,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  thrust::detail::temporary_array<value_type, DerivedPolicy> a(exec, first, middle);
  thrust::detail::temporary_array<value_type, DerivedPolicy> b(exec, middle, last);

  thrust::merge(exec, a.begin(), a.end(), b.begin(), b.end(), first, comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void inplace_merge_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                          RandomAccessIterator1 first1,
                          RandomAccessIterator1 middle1,
                          RandomAccessIterator1 last1,
                          RandomAccessIterator2 first2,
                          StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  RandomAccessIterator2 middle2 = first2 + (middle1 - first1);
  RandomAccessIterator2 last2   = first2 + (last1   - first1);

  thrust::detail::temporary_array<value_type1, DerivedPolicy> lhs1(exec, first1, middle1);
  thrust::detail::temporary_array<value_type1, DerivedPolicy> rhs1(exec, middle1, last1);
  thrust::detail::temporary_array<value_type2, DerivedPolicy> lhs2(exec, first2, middle2);
  thrust::detail::temporary_array<value_type2, DerivedPolicy> rhs2(exec, middle2, last2);

  thrust::merge_by_key(exec,
                       lhs1.begin(), lhs1.end(),
                       rhs1.begin(), rhs1.end(),
                       lhs2.begin(), rhs2.begin(),
                       first1, first2,
                       comp);
}


template<typename RandomAccessIterator,
         typename Size,
         typename StrictWeakOrdering>
__host__ __device__
void insertion_sort_each(RandomAccessIterator first,
                         RandomAccessIterator last,
                         Size partition_size,
                         StrictWeakOrdering comp)
{
  if(partition_size > 1)
  {
    for(; first < last; first += partition_size)
    {
      RandomAccessIterator partition_last = (thrust::min)(last, first + partition_size);

      thrust::system::detail::sequential::insertion_sort(first, partition_last, comp);
    } // end for
  } // end if
} // end insertion_sort_each()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Size,
         typename StrictWeakOrdering>
__host__ __device__
void insertion_sort_each_by_key(RandomAccessIterator1 keys_first,
                                RandomAccessIterator1 keys_last,
                                RandomAccessIterator2 values_first,
                                Size partition_size,
                                StrictWeakOrdering comp)
{
  if(partition_size > 1)
  {
    for(; keys_first < keys_last; keys_first += partition_size, values_first += partition_size)
    {
      RandomAccessIterator1 keys_partition_last = (thrust::min)(keys_last, keys_first + partition_size);

      thrust::system::detail::sequential::insertion_sort_by_key(keys_first, keys_partition_last, values_first, comp);
    } // end for
  } // end if
} // end insertion_sort_each()


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename Size,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void merge_adjacent_partitions(sequential::execution_policy<DerivedPolicy> &exec,
                               RandomAccessIterator1 first,
                               RandomAccessIterator1 last,
                               Size partition_size,
                               RandomAccessIterator2 result,
                               StrictWeakOrdering comp)
{
  for(; first < last; first += 2 * partition_size, result += 2 * partition_size)
  {
    RandomAccessIterator1 interval_middle = (thrust::min)(last, first + partition_size);
    RandomAccessIterator1 interval_last   = (thrust::min)(last, interval_middle + partition_size);

    thrust::merge(exec,
                  first, interval_middle,
                  interval_middle, interval_last,
                  result,
                  comp);
  } // end for
} // end merge_adjacent_partitions()


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Size,
         typename RandomAccessIterator3,
         typename RandomAccessIterator4,
         typename StrictWeakOrdering>
__host__ __device__
void merge_adjacent_partitions_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                                      RandomAccessIterator1 keys_first,
                                      RandomAccessIterator1 keys_last,
                                      RandomAccessIterator2 values_first,
                                      Size partition_size,
                                      RandomAccessIterator3 keys_result,
                                      RandomAccessIterator4 values_result,
                                      StrictWeakOrdering comp)
{
  Size stride = 2 * partition_size;

  for(;
      keys_first < keys_last;
      keys_first += stride, values_first += stride, keys_result += stride, values_result += stride)
  {
    RandomAccessIterator1 keys_interval_middle = (thrust::min)(keys_last, keys_first + partition_size);
    RandomAccessIterator1 keys_interval_last   = (thrust::min)(keys_last, keys_interval_middle + partition_size);

    RandomAccessIterator2 values_first2 = values_first + (keys_interval_middle - keys_first);

    thrust::merge_by_key(exec,
                         keys_first, keys_interval_middle,
                         keys_interval_middle, keys_interval_last,
                         values_first,
                         values_first2,
                         keys_result,
                         values_result,
                         comp);
  } // end for
} // end merge_adjacent_partitions()


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void iterative_stable_merge_sort(sequential::execution_policy<DerivedPolicy> &exec,
                                 RandomAccessIterator first,
                                 RandomAccessIterator last,
                                 StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference_type;

  difference_type n = last - first;

  thrust::detail::temporary_array<value_type, DerivedPolicy> temp(exec, n);

  // insertion sort each 32 element partition
  difference_type partition_size = 32;
  insertion_sort_each(first, last, partition_size, comp);

  // ping indicates whether or not the latest data is in the source range [first, last)
  bool ping = true;

  // merge adjacent partitions until the partition size covers the entire range
  for(;
      partition_size < n;
      partition_size *= 2, ping = !ping)
  {
    if(ping)
    {
      merge_adjacent_partitions(exec, first, last, partition_size, temp.begin(), comp);
    } // end if
    else
    {
      merge_adjacent_partitions(exec, temp.begin(), temp.end(), partition_size, first, comp);
    } // end else
  } // end for m

  if(!ping)
  {
    thrust::copy(exec, temp.begin(), temp.end(), first);
  } // end if
} // end iterative_stable_merge_sort()


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void iterative_stable_merge_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                                        RandomAccessIterator1 keys_first,
                                        RandomAccessIterator1 keys_last,
                                        RandomAccessIterator2 values_first,
                                        StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type      value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type      value_type2;
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type difference_type;

  difference_type n = keys_last - keys_first;

  thrust::detail::temporary_array<value_type1, DerivedPolicy> keys_temp(exec, n);
  thrust::detail::temporary_array<value_type2, DerivedPolicy> values_temp(exec, n);

  // insertion sort each 32 element partition
  difference_type partition_size = 32;
  insertion_sort_each_by_key(keys_first, keys_last, values_first, partition_size, comp);

  // ping indicates whether or not the latest data is in the source range [first, last)
  bool ping = true;

  // merge adjacent partitions until the partition size covers the entire range
  for(;
      partition_size < n;
      partition_size *= 2, ping = !ping)
  {
    if(ping)
    {
      merge_adjacent_partitions_by_key(exec, keys_first, keys_last, values_first, partition_size, keys_temp.begin(), values_temp.begin(), comp);
    } // end if
    else
    {
      merge_adjacent_partitions_by_key(exec, keys_temp.begin(), keys_temp.end(), values_temp.begin(), partition_size, keys_first, values_first, comp);
    } // end else
  } // end for m

  if(!ping)
  {
    thrust::copy(exec, keys_temp.begin(), keys_temp.end(), keys_first);
    thrust::copy(exec, values_temp.begin(), values_temp.end(), values_first);
  } // end if
} // end iterative_stable_merge_sort()


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void recursive_stable_merge_sort(sequential::execution_policy<DerivedPolicy> &exec,
                                 RandomAccessIterator first,
                                 RandomAccessIterator last,
                                 StrictWeakOrdering comp)
{
  if(last - first <= 32)
  {
    thrust::system::detail::sequential::insertion_sort(first, last, comp);
  } // end if
  else
  {
    RandomAccessIterator middle = first + (last - first) / 2;

    stable_merge_sort_detail::recursive_stable_merge_sort(exec, first, middle, comp);
    stable_merge_sort_detail::recursive_stable_merge_sort(exec, middle,  last, comp);
    stable_merge_sort_detail::inplace_merge(exec, first, middle, last, comp);
  } // end else
} // end recursive_stable_merge_sort()


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void recursive_stable_merge_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                                        RandomAccessIterator1 first1,
                                        RandomAccessIterator1 last1,
                                        RandomAccessIterator2 first2,
                                        StrictWeakOrdering comp)
{
  if(last1 - first1 <= 32)
  {
    thrust::system::detail::sequential::insertion_sort_by_key(first1, last1, first2, comp);
  } // end if
  else
  {
    RandomAccessIterator1 middle1 = first1 + (last1 - first1) / 2;
    RandomAccessIterator2 middle2 = first2 + (last1 - first1) / 2;

    stable_merge_sort_detail::recursive_stable_merge_sort_by_key(exec, first1, middle1, first2,  comp);
    stable_merge_sort_detail::recursive_stable_merge_sort_by_key(exec, middle1,  last1, middle2, comp);
    stable_merge_sort_detail::inplace_merge_by_key(exec, first1, middle1, last1, first2, comp);
  } // end else
} // end recursive_stable_merge_sort_by_key()


} // end namespace stable_merge_sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void stable_merge_sort(sequential::execution_policy<DerivedPolicy> &exec,
                       RandomAccessIterator first,
                       RandomAccessIterator last,
                       StrictWeakOrdering comp)
{
  NV_IF_TARGET(NV_IS_DEVICE, (
    // avoid recursion in CUDA threads
    stable_merge_sort_detail::iterative_stable_merge_sort(exec, first, last, comp);
  ), (
    stable_merge_sort_detail::recursive_stable_merge_sort(exec, first, last, comp);
  ));
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void stable_merge_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                              RandomAccessIterator1 first1,
                              RandomAccessIterator1 last1,
                              RandomAccessIterator2 first2,
                              StrictWeakOrdering comp)
{
  NV_IF_TARGET(NV_IS_DEVICE, (
    // avoid recursion in CUDA threads
    stable_merge_sort_detail::iterative_stable_merge_sort_by_key(exec, first1, last1, first2, comp);
  ), (
    stable_merge_sort_detail::recursive_stable_merge_sort_by_key(exec, first1, last1, first2, comp);
  ));
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

