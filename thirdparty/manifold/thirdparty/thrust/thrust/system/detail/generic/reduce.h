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
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename InputIterator>
__host__ __device__
  typename thrust::iterator_traits<InputIterator>::value_type
    reduce(thrust::execution_policy<DerivedPolicy> &exec, InputIterator first, InputIterator last);


template<typename DerivedPolicy, typename InputIterator, typename T>
__host__ __device__
  T reduce(thrust::execution_policy<DerivedPolicy> &exec, InputIterator first, InputIterator last, T init);


template<typename DerivedPolicy,
         typename InputIterator,
         typename T,
         typename BinaryFunction>
__host__ __device__
  T reduce(thrust::execution_policy<DerivedPolicy> &exec, InputIterator first, InputIterator last, T init, BinaryFunction binary_op);


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/reduce.inl>

