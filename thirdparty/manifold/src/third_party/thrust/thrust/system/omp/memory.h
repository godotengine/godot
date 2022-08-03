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

/*! \file thrust/system/omp/memory.h
 *  \brief Managing memory associated with Thrust's OpenMP system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/omp/memory_resource.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/mr/allocator.h>
#include <ostream>

THRUST_NAMESPACE_BEGIN
namespace system { namespace omp
{

/*! Allocates an area of memory available to Thrust's <tt>omp</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>omp::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>omp::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>omp::pointer<void></tt> returned by this function must be
 *        deallocated with \p omp::free.
 *  \see omp::free
 *  \see std::malloc
 */
inline pointer<void> malloc(std::size_t n);

/*! Allocates a typed area of memory available to Thrust's <tt>omp</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>omp::pointer<T></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>omp::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>omp::pointer<T></tt> returned by this function must be
 *        deallocated with \p omp::free.
 *  \see omp::free
 *  \see std::malloc
 */
template<typename T>
inline pointer<T> malloc(std::size_t n);

/*! Deallocates an area of memory previously allocated by <tt>omp::malloc</tt>.
 *  \param ptr A <tt>omp::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>omp::malloc</tt>.
 *  \see omp::malloc
 *  \see std::free
 */
inline void free(pointer<void> ptr);

/*! \p omp::allocator is the default allocator used by the \p omp system's
 *  containers such as <tt>omp::vector</tt> if no user-specified allocator is
 *  provided. \p omp::allocator allocates (deallocates) storage with \p
 *  omp::malloc (\p omp::free).
 */
template<typename T>
using allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::omp::memory_resource
>;

/*! \p omp::universal_allocator allocates memory that can be used by the \p omp
 *  system and host systems.
 */
template<typename T>
using universal_allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::omp::universal_memory_resource
>;

}} // namespace system::omp

/*! \namespace thrust::omp
 *  \brief \p thrust::omp is a top-level alias for thrust::system::omp.
 */
namespace omp
{
using thrust::system::omp::malloc;
using thrust::system::omp::free;
using thrust::system::omp::allocator;
using thrust::system::omp::universal_allocator;
} // namespace omp

THRUST_NAMESPACE_END

#include <thrust/system/omp/detail/memory.inl>

