/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in ctbbliance with the License.
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

/*! \file thrust/system/tbb/memory.h
 *  \brief Managing memory associated with Thrust's TBB system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/tbb/memory_resource.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/mr/allocator.h>
#include <ostream>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{

/*! Allocates an area of memory available to Thrust's <tt>tbb</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>tbb::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>tbb::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>tbb::pointer<void></tt> returned by this function must be
 *        deallocated with \p tbb::free.
 *  \see tbb::free
 *  \see std::malloc
 */
inline pointer<void> malloc(std::size_t n);

/*! Allocates a typed area of memory available to Thrust's <tt>tbb</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>tbb::pointer<T></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>tbb::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>tbb::pointer<T></tt> returned by this function must be
 *        deallocated with \p tbb::free.
 *  \see tbb::free
 *  \see std::malloc
 */
template<typename T>
inline pointer<T> malloc(std::size_t n);

/*! Deallocates an area of memory previously allocated by <tt>tbb::malloc</tt>.
 *  \param ptr A <tt>tbb::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>tbb::malloc</tt>.
 *  \see tbb::malloc
 *  \see std::free
 */
inline void free(pointer<void> ptr);

/*! \p tbb::allocator is the default allocator used by the \p tbb system's
 *  containers such as <tt>tbb::vector</tt> if no user-specified allocator is
 *  provided. \p tbb::allocator allocates (deallocates) storage with \p
 *  tbb::malloc (\p tbb::free).
 */
template<typename T>
using allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::tbb::memory_resource
>;

/*! \p tbb::universal_allocator allocates memory that can be used by the \p tbb
 *  system and host systems.
 */
template<typename T>
using universal_allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::tbb::universal_memory_resource
>;

}} // namespace system::tbb

/*! \namespace thrust::tbb
 *  \brief \p thrust::tbb is a top-level alias for thrust::system::tbb.
 */
namespace tbb
{
using thrust::system::tbb::malloc;
using thrust::system::tbb::free;
using thrust::system::tbb::allocator;
using thrust::system::tbb::universal_allocator;
} // namsespace tbb

THRUST_NAMESPACE_END

#include <thrust/system/tbb/detail/memory.inl>

