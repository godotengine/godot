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
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2>
__host__ __device__
  thrust::pair<ForwardIterator1,ForwardIterator2>
    unique_by_key(thrust::execution_policy<ExecutionPolicy> &exec,
                  ForwardIterator1 keys_first, 
                  ForwardIterator1 keys_last,
                  ForwardIterator2 values_first);


template<typename ExecutionPolicy,
         typename ForwardIterator1,
         typename ForwardIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<ForwardIterator1,ForwardIterator2>
    unique_by_key(thrust::execution_policy<ExecutionPolicy> &exec,
                  ForwardIterator1 keys_first, 
                  ForwardIterator1 keys_last,
                  ForwardIterator2 values_first,
                  BinaryPredicate binary_pred);


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(thrust::execution_policy<ExecutionPolicy> &exec,
                       InputIterator1 keys_first, 
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_output,
                       OutputIterator2 values_output);


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    unique_by_key_copy(thrust::execution_policy<ExecutionPolicy> &exec,
                       InputIterator1 keys_first, 
                       InputIterator1 keys_last,
                       InputIterator2 values_first,
                       OutputIterator1 keys_output,
                       OutputIterator2 values_output,
                       BinaryPredicate binary_pred);


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/unique_by_key.inl>

