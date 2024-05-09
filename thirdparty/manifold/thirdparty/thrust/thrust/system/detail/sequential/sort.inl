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

#include <thrust/reverse.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/sequential/stable_merge_sort.h>
#include <thrust/system/detail/sequential/stable_primitive_sort.h>

#include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{
namespace sort_detail
{


////////////////////
// Primitive Sort //
////////////////////


template<typename KeyType, typename Compare>
struct needs_reverse
  : thrust::detail::integral_constant<
      bool,
      thrust::detail::is_same<Compare, typename thrust::greater<KeyType> >::value
    >
{};


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort(sequential::execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering,
                 thrust::detail::true_type)
{
  thrust::system::detail::sequential::stable_primitive_sort(exec, first, last);

  // if comp is greater<T> then reverse the keys
  typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;

  if(needs_reverse<KeyType,StrictWeakOrdering>::value)
  {
    thrust::reverse(exec, first, last);
  }
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator1 first1,
                        RandomAccessIterator1 last1,
                        RandomAccessIterator2 first2,
                        StrictWeakOrdering,
                        thrust::detail::true_type)
{
  // if comp is greater<T> then reverse the keys and values
  typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type KeyType;

  // note, we also have to reverse the (unordered) input to preserve stability
  if(needs_reverse<KeyType,StrictWeakOrdering>::value)
  {
    thrust::reverse(exec, first1,  last1);
    thrust::reverse(exec, first2, first2 + (last1 - first1));
  }

  thrust::system::detail::sequential::stable_primitive_sort_by_key(exec, first1, last1, first2);

  if(needs_reverse<KeyType,StrictWeakOrdering>::value)
  {
    thrust::reverse(exec, first1,  last1);
    thrust::reverse(exec, first2, first2 + (last1 - first1));
  }
}


////////////////
// Merge Sort //
////////////////


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort(sequential::execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp,
                 thrust::detail::false_type)
{
  thrust::system::detail::sequential::stable_merge_sort(exec, first, last, comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator1 first1,
                        RandomAccessIterator1 last1,
                        RandomAccessIterator2 first2,
                        StrictWeakOrdering comp,
                        thrust::detail::false_type)
{
  thrust::system::detail::sequential::stable_merge_sort_by_key(exec, first1, last1, first2, comp);
}


template<typename KeyType, typename Compare>
struct use_primitive_sort
  : thrust::detail::and_<
      thrust::detail::is_arithmetic<KeyType>,
      thrust::detail::or_<
        thrust::detail::is_same<Compare, thrust::less<KeyType> >,
        thrust::detail::is_same<Compare, thrust::greater<KeyType> >
      >
    >
{};


} // end namespace sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort(sequential::execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{

  // the compilation time of stable_primitive_sort is too expensive to use within a single CUDA thread
  NV_IF_TARGET(NV_IS_HOST, (
    using KeyType = thrust::iterator_value_t<RandomAccessIterator>;
    sort_detail::use_primitive_sort<KeyType, StrictWeakOrdering> use_primitive_sort;
    sort_detail::stable_sort(exec, first, last, comp, use_primitive_sort);
  ), ( // NV_IS_DEVICE:
    thrust::detail::false_type use_primitive_sort;
    sort_detail::stable_sort(exec, first, last, comp, use_primitive_sort);
  ));
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
__host__ __device__
void stable_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator1 first1,
                        RandomAccessIterator1 last1,
                        RandomAccessIterator2 first2,
                        StrictWeakOrdering comp)
{

  // the compilation time of stable_primitive_sort_by_key is too expensive to use within a single CUDA thread
  NV_IF_TARGET(NV_IS_HOST, (
    using KeyType = thrust::iterator_value_t<RandomAccessIterator1>;
    sort_detail::use_primitive_sort<KeyType, StrictWeakOrdering> use_primitive_sort;
    sort_detail::stable_sort_by_key(exec, first1, last1, first2, comp, use_primitive_sort);
  ), ( // NV_IS_DEVICE:
    thrust::detail::false_type use_primitive_sort;
    sort_detail::stable_sort_by_key(exec, first1, last1, first2, comp, use_primitive_sort);
  ));
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

