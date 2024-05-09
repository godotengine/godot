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
#include <thrust/system/detail/generic/sort.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/find.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/detail/internal_functional.h>

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
            RandomAccessIterator last)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type; 
  thrust::sort(exec, first, last, thrust::less<value_type>());
} // end sort()


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
  void sort(thrust::execution_policy<DerivedPolicy> &exec,
            RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
  // implement with stable_sort
  thrust::stable_sort(exec, first, last, comp);
} // end sort()


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
  void sort_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type;
  thrust::sort_by_key(exec, keys_first, keys_last, values_first, thrust::less<value_type>());
} // end sort_by_key()


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
  void sort_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                   RandomAccessIterator1 keys_first,
                   RandomAccessIterator1 keys_last,
                   RandomAccessIterator2 values_first,
                   StrictWeakOrdering comp)
{
  // implement with stable_sort_by_key
  thrust::stable_sort_by_key(exec, keys_first, keys_last, values_first, comp);
} // end sort_by_key()


template<typename DerivedPolicy,
         typename RandomAccessIterator>
__host__ __device__
  void stable_sort(thrust::execution_policy<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator last)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;
  thrust::stable_sort(exec, first, last, thrust::less<value_type>());
} // end stable_sort()


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ __device__
  void stable_sort_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                          RandomAccessIterator1 keys_first,
                          RandomAccessIterator1 keys_last,
                          RandomAccessIterator2 values_first)
{
  typedef typename iterator_value<RandomAccessIterator1>::type value_type;
  thrust::stable_sort_by_key(exec, keys_first, keys_last, values_first, thrust::less<value_type>());
} // end stable_sort_by_key()


template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  bool is_sorted(thrust::execution_policy<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last)
{
  return thrust::is_sorted_until(exec, first, last) == last;
} // end is_sorted()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Compare>
__host__ __device__
  bool is_sorted(thrust::execution_policy<DerivedPolicy> &exec,
                 ForwardIterator first,
                 ForwardIterator last,
                 Compare comp)
{
  return thrust::is_sorted_until(exec, first, last, comp) == last;
} // end is_sorted()


template<typename DerivedPolicy, typename ForwardIterator>
__host__ __device__
  ForwardIterator is_sorted_until(thrust::execution_policy<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last)
{
  typedef typename thrust::iterator_value<ForwardIterator>::type InputType;

  return thrust::is_sorted_until(exec, first, last, thrust::less<InputType>());
} // end is_sorted_until()


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Compare>
__host__ __device__
  ForwardIterator is_sorted_until(thrust::execution_policy<DerivedPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Compare comp)
{
  if(thrust::distance(first,last) < 2) return last;

  typedef thrust::tuple<ForwardIterator,ForwardIterator> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple>            ZipIterator;

  ForwardIterator first_plus_one = first;
  thrust::advance(first_plus_one, 1);

  ZipIterator zipped_first = thrust::make_zip_iterator(thrust::make_tuple(first_plus_one, first));
  ZipIterator zipped_last  = thrust::make_zip_iterator(thrust::make_tuple(last, first));

  return thrust::get<0>(thrust::find_if(exec, zipped_first, zipped_last, thrust::detail::tuple_binary_predicate<Compare>(comp)).get_iterator_tuple());
} // end is_sorted_until()


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
  void stable_sort(thrust::execution_policy<DerivedPolicy> &,
                   RandomAccessIterator,
                   RandomAccessIterator,
                   StrictWeakOrdering)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<RandomAccessIterator, false>::value)
  , "unimplemented for this system"
  );
} // end stable_sort()


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
  void stable_sort_by_key(thrust::execution_policy<DerivedPolicy> &,
                          RandomAccessIterator1,
                          RandomAccessIterator1,
                          RandomAccessIterator2,
                          StrictWeakOrdering)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<RandomAccessIterator1, false>::value)
  , "unimplemented for this system"
  );
} // end stable_sort_by_key()


} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

