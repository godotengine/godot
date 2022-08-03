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
 *  \brief Generic implementations of key-value scans.
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


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator inclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result);


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator inclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred);


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryPredicate,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator inclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op);


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>
__host__ __device__
  OutputIterator exclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result);


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T>
__host__ __device__
  OutputIterator exclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init);


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate>
__host__ __device__
  OutputIterator exclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred);


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename T,
         typename BinaryPredicate,
         typename AssociativeOperator>
__host__ __device__
  OutputIterator exclusive_scan_by_key(thrust::execution_policy<DerivedPolicy> &exec,
                                       InputIterator1 first1,
                                       InputIterator1 last1,
                                       InputIterator2 first2,
                                       OutputIterator result,
                                       T init,
                                       BinaryPredicate binary_pred,
                                       AssociativeOperator binary_op);

} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/scan_by_key.inl>

