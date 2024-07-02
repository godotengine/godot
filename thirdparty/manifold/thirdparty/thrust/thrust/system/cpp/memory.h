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

/*! \file thrust/system/cpp/memory.h
 *  \brief Managing memory associated with Thrust's Standard C++ system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cpp/memory_resource.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/mr/allocator.h>
#include <ostream>

THRUST_NAMESPACE_BEGIN
namespace system { namespace cpp
{

/*! Allocates an area of memory available to Thrust's <tt>cpp</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>cpp::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>cpp::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>cpp::pointer<void></tt> returned by this function must be
 *        deallocated with \p cpp::free.
 *  \see cpp::free
 *  \see std::malloc
 */
inline pointer<void> malloc(std::size_t n);

/*! Allocates a typed area of memory available to Thrust's <tt>cpp</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>cpp::pointer<T></tt> pointing to the beginning of the newly
 *          allocated elements. A null <tt>cpp::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>cpp::pointer<T></tt> returned by this function must be
 *        deallocated with \p cpp::free.
 *  \see cpp::free
 *  \see std::malloc
 */
template<typename T>
inline pointer<T> malloc(std::size_t n);

/*! Deallocates an area of memory previously allocated by <tt>cpp::malloc</tt>.
 *  \param ptr A <tt>cpp::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>cpp::malloc</tt>.
 *  \see cpp::malloc
 *  \see std::free
 */
inline void free(pointer<void> ptr);

/*! \p cpp::allocator is the default allocator used by the \p cpp system's
 *  containers such as <tt>cpp::vector</tt> if no user-specified allocator is
 *  provided. \p cpp::allocator allocates (deallocates) storage with \p
 *  cpp::malloc (\p cpp::free).
 */
template<typename T>
using allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::cpp::memory_resource
>;

/*! \p cpp::universal_allocator allocates memory that can be used by the \p cpp
 *  system and host systems.
 */
template<typename T>
using universal_allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::cpp::universal_memory_resource
>;

}} // namespace system::cpp

/*! \namespace thrust::cpp
 *  \brief \p thrust::cpp is a top-level alias for thrust::system::cpp.
 */
namespace cpp
{
using thrust::system::cpp::malloc;
using thrust::system::cpp::free;
using thrust::system::cpp::allocator;
} // namespace cpp

THRUST_NAMESPACE_END

#include <thrust/system/cpp/detail/memory.inl>

