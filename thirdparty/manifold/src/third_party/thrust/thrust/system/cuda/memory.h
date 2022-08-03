/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file thrust/system/cuda/memory.h
 *  \brief Managing memory associated with Thrust's CUDA system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/mr/allocator.h>
#include <ostream>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{

/*! Allocates an area of memory available to Thrust's <tt>cuda</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>cuda::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>cuda::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>cuda::pointer<void></tt> returned by this function must be
 *        deallocated with \p cuda::free.
 *  \see cuda::free
 *  \see std::malloc
 */
inline __host__ __device__ pointer<void> malloc(std::size_t n);

/*! Allocates a typed area of memory available to Thrust's <tt>cuda</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>cuda::pointer<T></tt> pointing to the beginning of the newly
 *          allocated elements. A null <tt>cuda::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>cuda::pointer<T></tt> returned by this function must be
 *        deallocated with \p cuda::free.
 *  \see cuda::free
 *  \see std::malloc
 */
template <typename T>
inline __host__ __device__ pointer<T> malloc(std::size_t n);

/*! Deallocates an area of memory previously allocated by <tt>cuda::malloc</tt>.
 *  \param ptr A <tt>cuda::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>cuda::malloc</tt>.
 *  \see cuda::malloc
 *  \see std::free
 */
inline __host__ __device__ void free(pointer<void> ptr);

/*! \p cuda::allocator is the default allocator used by the \p cuda system's
 *  containers such as <tt>cuda::vector</tt> if no user-specified allocator is
 *  provided. \p cuda::allocator allocates (deallocates) storage with \p
 *  cuda::malloc (\p cuda::free).
 */
template<typename T>
using allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::cuda::memory_resource
>;

/*! \p cuda::universal_allocator allocates memory that can be used by the \p cuda
 *  system and host systems.
 */
template<typename T>
using universal_allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::cuda::universal_memory_resource
>;

} // namespace cuda_cub

namespace system { namespace cuda
{
using thrust::cuda_cub::malloc;
using thrust::cuda_cub::free;
using thrust::cuda_cub::allocator;
using thrust::cuda_cub::universal_allocator;
}} // namespace system::cuda

/*! \namespace thrust::cuda
 *  \brief \p thrust::cuda is a top-level alias for \p thrust::system::cuda.
 */
namespace cuda
{
using thrust::cuda_cub::malloc;
using thrust::cuda_cub::free;
using thrust::cuda_cub::allocator;
using thrust::cuda_cub::universal_allocator;
} // namespace cuda

THRUST_NAMESPACE_END

#include <thrust/system/cuda/detail/memory.inl>

