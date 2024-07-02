/*
 *  Copyright 2008-2020 NVIDIA Corporation
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

/*! \file shuffle.h
 *  \brief Generic implementations of shuffle functions.
 */

#pragma once


#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system {
namespace detail {
namespace generic {

template <typename ExecutionPolicy, typename RandomIterator, typename URBG>
__host__ __device__ void shuffle(
    thrust::execution_policy<ExecutionPolicy>& exec, RandomIterator first,
    RandomIterator last, URBG&& g);

template <typename ExecutionPolicy, typename RandomIterator,
          typename OutputIterator, typename URBG>
__host__ __device__ void shuffle_copy(
    thrust::execution_policy<ExecutionPolicy>& exec, RandomIterator first,
    RandomIterator last, OutputIterator result, URBG&& g);

}  // end namespace generic
}  // end namespace detail
}  // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/shuffle.inl>

#endif
