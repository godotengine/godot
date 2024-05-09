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
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template<typename System, typename Allocator, typename InputIterator, typename Pointer>
__host__ __device__
  Pointer copy_construct_range(thrust::execution_policy<System> &from_system,
                               Allocator &a,
                               InputIterator first,
                               InputIterator last,
                               Pointer result);

template<typename System, typename Allocator, typename InputIterator, typename Size, typename Pointer>
__host__ __device__
  Pointer copy_construct_range_n(thrust::execution_policy<System> &from_system,
                                 Allocator &a,
                                 InputIterator first,
                                 Size n,
                                 Pointer result);

} // end detail
THRUST_NAMESPACE_END

#include <thrust/detail/allocator/copy_construct_range.inl>

