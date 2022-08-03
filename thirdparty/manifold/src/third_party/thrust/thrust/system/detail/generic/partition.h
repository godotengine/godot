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


/*! \file partition.h
 *  \brief Generic implementations of partition functions.
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


template<typename ExecutionPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(thrust::execution_policy<ExecutionPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred);

template<typename ExecutionPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator stable_partition(thrust::execution_policy<ExecutionPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred);


template<typename ExecutionPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(thrust::execution_policy<ExecutionPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred);


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(thrust::execution_policy<ExecutionPolicy> &exec,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred);


template<typename ExecutionPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(thrust::execution_policy<ExecutionPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred);


template<typename ExecutionPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition(thrust::execution_policy<ExecutionPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred);


template<typename ExecutionPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(thrust::execution_policy<ExecutionPolicy> &exec,
                   InputIterator first,
                   InputIterator last,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred);


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
__host__ __device__
  thrust::pair<OutputIterator1,OutputIterator2>
    partition_copy(thrust::execution_policy<ExecutionPolicy> &exec,
                   InputIterator1 first,
                   InputIterator1 last,
                   InputIterator2 stencil,
                   OutputIterator1 out_true,
                   OutputIterator2 out_false,
                   Predicate pred);


template<typename ExecutionPolicy,
         typename ForwardIterator,
         typename Predicate>
__host__ __device__
  ForwardIterator partition_point(thrust::execution_policy<ExecutionPolicy> &exec,
                                  ForwardIterator first,
                                  ForwardIterator last,
                                  Predicate pred);


template<typename ExecutionPolicy,
         typename InputIterator,
         typename Predicate>
__host__ __device__
  bool is_partitioned(thrust::execution_policy<ExecutionPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred);


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/partition.inl>

