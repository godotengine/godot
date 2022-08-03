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

/*! \file thrust/system/tbb/vector.h
 *  \brief A dynamically-sizable array of elements which reside in memory available to
 *         Thrust's TBB system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/tbb/memory.h>
#include <thrust/detail/vector_base.h>
#include <vector>

THRUST_NAMESPACE_BEGIN
namespace system { namespace tbb
{

/*! \p tbb::vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p tbb::vector may vary dynamically; memory management is
 *  automatic. The elements contained in a \p tbb::vector reside in memory
 *  accessible by the \p tbb system.
 *
 *  \tparam T The element type of the \p tbb::vector.
 *  \tparam Allocator The allocator type of the \p tbb::vector.
 *          Defaults to \p tbb::allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p tbb::vector.
 *  \see device_vector
 *  \see universal_vector
 */
template <typename T, typename Allocator = thrust::system::tbb::allocator<T>>
using vector = thrust::detail::vector_base<T, Allocator>;

/*! \p tbb::universal_vector is a container that supports random access to
 *  elements, constant time removal of elements at the end, and linear time
 *  insertion and removal of elements at the beginning or in the middle. The
 *  number of elements in a \p tbb::universal_vector may vary dynamically;
 *  memory management is automatic. The elements contained in a
 *  \p tbb::universal_vector reside in memory accessible by the \p tbb system
 *  and host systems.
 *
 *  \tparam T The element type of the \p tbb::universal_vector.
 *  \tparam Allocator The allocator type of the \p tbb::universal_vector.
 *          Defaults to \p tbb::universal_allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p tbb::universal_vector
 *  \see device_vector
 *  \see universal_vector
 */
template <typename T, typename Allocator = thrust::system::tbb::universal_allocator<T>>
using universal_vector = thrust::detail::vector_base<T, Allocator>;

}} // namespace system::tbb

namespace tbb
{
using thrust::system::tbb::vector;
using thrust::system::tbb::universal_vector;
}

THRUST_NAMESPACE_END
