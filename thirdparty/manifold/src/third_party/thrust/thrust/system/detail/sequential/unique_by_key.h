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


/*! \file unique_by_key.h
 *  \brief Sequential implementations of unique_by_key algorithms.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/sequential/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>

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
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(sequential::execution_policy<DerivedPolicy> &,
                       InputIterator1 keys_first, 
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_output,
                       OutputIterator2 values_output,
                       BinaryPredicate binary_pred)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type  InputKeyType;
  typedef typename thrust::iterator_traits<OutputIterator2>::value_type OutputValueType;

  if(keys_first != keys_last)
  {
    InputKeyType    temp_key   = *keys_first;
    OutputValueType temp_value = *values_first;

    for(++keys_first, ++values_first;
        keys_first != keys_last;
        ++keys_first, ++values_first)
    {
      InputKeyType    key   = *keys_first;
      OutputValueType value = *values_first;

      if(!binary_pred(temp_key, key))
      {
        *keys_output   = temp_key;
        *values_output = temp_value;

        ++keys_output;
        ++values_output;

        temp_key   = key;
        temp_value = value;
      }
    }

    *keys_output   = temp_key;
    *values_output = temp_value;

    ++keys_output;
    ++values_output;
  }

  return thrust::make_pair(keys_output, values_output);
} // end unique_by_key_copy()


template<typename DerivedPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<ForwardIterator1,ForwardIterator2>
    unique_by_key(sequential::execution_policy<DerivedPolicy> &exec,
                  ForwardIterator1 keys_first, 
                  ForwardIterator1 keys_last,
                  ForwardIterator2 values_first,
                  BinaryPredicate binary_pred)
{
  // sequential unique_by_key_copy() permits in-situ operation
  return sequential::unique_by_key_copy(exec, keys_first, keys_last, values_first, keys_first, values_first, binary_pred);
} // end unique_by_key()


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

