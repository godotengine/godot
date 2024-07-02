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
#include <thrust/detail/static_assert.h>
#include <thrust/system/detail/generic/merge.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/detail/internal_functional.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
__host__ __device__
  OutputIterator merge(thrust::execution_policy<DerivedPolicy> &,
                       InputIterator1,
                       InputIterator1,
                       InputIterator2,
                       InputIterator2,
                       OutputIterator result,
                       StrictWeakOrdering)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<InputIterator1, false>::value)
  , "unimplemented for this system"
  );
  return result;
} // end merge()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator merge(thrust::execution_policy<DerivedPolicy> &exec,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result)
{
  typedef typename thrust::iterator_value<InputIterator1>::type value_type;
  return thrust::merge(exec,first1,last1,first2,last2,result,thrust::less<value_type>());
} // end merge()


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2, typename Compare>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result,
                 Compare comp)
{
  typedef thrust::tuple<InputIterator1, InputIterator3>   iterator_tuple1;
  typedef thrust::tuple<InputIterator2, InputIterator4>   iterator_tuple2;
  typedef thrust::tuple<OutputIterator1, OutputIterator2> iterator_tuple3;

  typedef thrust::zip_iterator<iterator_tuple1> zip_iterator1;
  typedef thrust::zip_iterator<iterator_tuple2> zip_iterator2;
  typedef thrust::zip_iterator<iterator_tuple3> zip_iterator3;

  zip_iterator1 zipped_first1 = thrust::make_zip_iterator(thrust::make_tuple(keys_first1, values_first1));
  zip_iterator1 zipped_last1  = thrust::make_zip_iterator(thrust::make_tuple(keys_last1, values_first1));

  zip_iterator2 zipped_first2 = thrust::make_zip_iterator(thrust::make_tuple(keys_first2, values_first2));
  zip_iterator2 zipped_last2  = thrust::make_zip_iterator(thrust::make_tuple(keys_last2, values_first2));

  zip_iterator3 zipped_result = thrust::make_zip_iterator(thrust::make_tuple(keys_result, values_result));

  thrust::detail::compare_first<Compare> comp_first(comp);

  iterator_tuple3 result = thrust::merge(exec, zipped_first1, zipped_last1, zipped_first2, zipped_last2, zipped_result, comp_first).get_iterator_tuple();

  return thrust::make_pair(thrust::get<0>(result), thrust::get<1>(result));
} // end merge_by_key()


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result)
{
  typedef typename thrust::iterator_value<InputIterator1>::type value_type;
  return thrust::merge_by_key(exec, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result, thrust::less<value_type>());
} // end merge_by_key()


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

