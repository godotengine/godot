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


/*! \file scan_by_key.h
 *  \brief Sequential implementation of scan_by_key functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/function.h>
#include <thrust/system/detail/sequential/execution_policy.h>

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
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
  OutputIterator inclusive_scan_by_key(sequential::execution_policy<DerivedPolicy> &,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       BinaryFunction binary_op)
{
  using KeyType = typename thrust::iterator_traits<InputIterator1>::value_type;
  using ValueType = typename thrust::iterator_traits<InputIterator2>::value_type;

  // wrap binary_op
  thrust::detail::wrapped_function<
    BinaryFunction,
    ValueType
  > wrapped_binary_op(binary_op);

  if(first1 != last1)
  {
    KeyType   prev_key   = *first1;
    ValueType prev_value = *first2;

    *result = prev_value;

    for(++first1, ++first2, ++result;
        first1 != last1;
        ++first1, ++first2, ++result)
    {
      KeyType key = *first1;

      if(binary_pred(prev_key, key))
        *result = prev_value = wrapped_binary_op(prev_value,*first2);
      else
        *result = prev_value = *first2;

      prev_key = key;
    }
  }

  return result;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename BinaryFunction>
__host__ __device__
  OutputIterator exclusive_scan_by_key(sequential::execution_policy<DerivedPolicy> &,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       BinaryFunction binary_op)
{
  using KeyType = typename thrust::iterator_traits<InputIterator1>::value_type;
  using ValueType = T;

  if(first1 != last1)
  {
    KeyType   temp_key   = *first1;
    ValueType temp_value = *first2;

    ValueType next = init;

    // first one is init
    *result = next;

    next = binary_op(next, temp_value);

    for(++first1, ++first2, ++result;
        first1 != last1;
        ++first1, ++first2, ++result)
    {
      KeyType key = *first1;

      // use temp to permit in-place scans
      temp_value = *first2;

      if (!binary_pred(temp_key, key))
        next = init;  // reset sum

      *result = next;  
      next = binary_op(next, temp_value);

      temp_key = key;
    }
  }

  return result;
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

