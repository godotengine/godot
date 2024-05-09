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


/*! \file adjacent_difference.h
 *  \brief Generic implementation of adjacent_difference.
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


template<typename DerivedPolicy, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator adjacent_difference(thrust::execution_policy<DerivedPolicy> &exec,
                                   InputIterator first, InputIterator last,
                                   OutputIterator result);


template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
__host__ __device__
OutputIterator adjacent_difference(thrust::execution_policy<DerivedPolicy> &exec,
                                   InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op);


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/adjacent_difference.inl>

