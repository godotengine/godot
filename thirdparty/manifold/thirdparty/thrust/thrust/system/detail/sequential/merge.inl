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
#include <thrust/system/detail/sequential/merge.h>
#include <thrust/detail/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/function.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename StrictWeakOrdering>
__host__ __device__
OutputIterator merge(sequential::execution_policy<DerivedPolicy> &exec,
                     InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 first2,
                     InputIterator2 last2,
                     OutputIterator result,
                     StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(first1 != last1 && first2 != last2)
  {
    if(wrapped_comp(*first2, *first1))
    {
      *result = *first2;
      ++first2;
    } // end if
    else
    {
      *result = *first1;
      ++first1;
    } // end else

    ++result;
  } // end while

  return thrust::copy(exec, first2, last2, thrust::copy(exec, first1, last1, result));
} // end merge()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename InputIterator4,
         typename OutputIterator1,
         typename OutputIterator2,
         typename StrictWeakOrdering>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
  merge_by_key(sequential::execution_policy<DerivedPolicy> &,
               InputIterator1 keys_first1,
               InputIterator1 keys_last1,
               InputIterator2 keys_first2,
               InputIterator2 keys_last2,
               InputIterator3 values_first1,
               InputIterator4 values_first2,
               OutputIterator1 keys_result,
               OutputIterator2 values_result,
               StrictWeakOrdering comp)
{
  // wrap comp
  thrust::detail::wrapped_function<
    StrictWeakOrdering,
    bool
  > wrapped_comp(comp);

  while(keys_first1 != keys_last1 && keys_first2 != keys_last2)
  {
    if(!wrapped_comp(*keys_first2, *keys_first1))
    {
      // *keys_first1 <= *keys_first2
      *keys_result   = *keys_first1;
      *values_result = *values_first1;
      ++keys_first1;
      ++values_first1;
    }
    else
    {
      // *keys_first1 > keys_first2
      *keys_result   = *keys_first2;
      *values_result = *values_first2;
      ++keys_first2;
      ++values_first2;
    }

    ++keys_result;
    ++values_result;
  }

  while(keys_first1 != keys_last1)
  {
    *keys_result   = *keys_first1;
    *values_result = *values_first1;
    ++keys_first1;
    ++values_first1;
    ++keys_result;
    ++values_result;
  }

  while(keys_first2 != keys_last2)
  {
    *keys_result   = *keys_first2;
    *values_result = *values_first2;
    ++keys_first2;
    ++values_first2;
    ++keys_result;
    ++values_result;
  }

  return thrust::make_pair(keys_result, values_result);
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

