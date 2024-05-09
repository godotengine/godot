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
#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
  void sort(thrust::execution_policy<DerivedPolicy> &exec,
            RandomAccessIterator first,
            RandomAccessIterator last);


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
  void sort(thrust::execution_policy<DerivedPolicy> &exec,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp);


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
  void sort_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first);


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
  void sort_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp);


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
  void stable_sort(thrust::execution_policy<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator last);


// XXX it is an error to call this function; it has no implementation
template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
  void stable_sort(thrust::execution_policy<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp);


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
  void stable_sort_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first);


// XXX it is an error to call this function; it has no implementation
template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
  void stable_sort_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first,
                          StrictWeakOrdering comp);


template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  bool is_sorted(thrust::execution_policy<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last);


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Compare>
__host__ __device__
  bool is_sorted(thrust::execution_policy<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last,
                 Compare comp);


template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator is_sorted_until(thrust::execution_policy<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last);


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Compare>
__host__ __device__
  ForwardIterator is_sorted_until(thrust::execution_policy<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp);


} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/sort.inl>

