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


// XXX calling this function is an error; there is no implementation
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
__host__ __device__
  OutputIterator merge(thrust::execution_policy<DerivedPolicy> &exec,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       InputIterator2 last2,
                       OutputIterator result,
                       StrictWeakOrdering comp);


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
                       OutputIterator result);


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2, typename Compare>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result,
                 Compare comp);


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename InputIterator3, typename InputIterator4, typename OutputIterator1, typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    merge_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                 InputIterator1 keys_first1, InputIterator1 keys_last1,
                 InputIterator2 keys_first2, InputIterator2 keys_last2,
                 InputIterator3 values_first1, InputIterator4 values_first2,
                 OutputIterator1 keys_result,
                 OutputIterator2 values_result);


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/merge.inl>

