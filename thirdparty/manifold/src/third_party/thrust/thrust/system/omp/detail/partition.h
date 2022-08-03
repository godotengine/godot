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


/*! \file reduce.h
 *  \brief OpenMP implementation of reduce algorithms.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename Predicate>
  ForwardIterator stable_partition(execution_policy<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   Predicate pred);

template<typename DerivedPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator stable_partition(execution_policy<DerivedPolicy> &exec,
                                   ForwardIterator first,
                                   ForwardIterator last,
                                   InputIterator stencil,
                                   Predicate pred);

template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(execution_policy<DerivedPolicy> &exec,
                          InputIterator first,
                          InputIterator last,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred);

template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename Predicate>
  thrust::pair<OutputIterator1,OutputIterator2>
    stable_partition_copy(execution_policy<DerivedPolicy> &exec,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator1 out_true,
                          OutputIterator2 out_false,
                          Predicate pred);


} // end namespace detail
} // end namespace omp
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/omp/detail/partition.inl>

