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

/*! \file thrust/system/cpp/vector.h
 *  \brief A dynamically-sizable array of elements which reside in memory available to
 *         Thrust's standard C++ system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/detail/vector_base.h>
#include <vector>

THRUST_NAMESPACE_BEGIN
namespace system { namespace cpp
{

/*! \p cpp::vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p cpp::vector may vary dynamically; memory management is
 *  automatic. The elements contained in a \p cpp::vector reside in memory
 *  accessible by the \p cpp system.
 *
 *  \tparam T The element type of the \p cpp::vector.
 *  \tparam Allocator The allocator type of the \p cpp::vector.
 *          Defaults to \p cpp::allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p cpp::vector.
 *  \see device_vector
 *  \see universal_vector
 */
template <typename T, typename Allocator = thrust::system::cpp::allocator<T>>
using vector = thrust::detail::vector_base<T, Allocator>;

/*! \p cpp::universal_vector is a container that supports random access to
 *  elements, constant time removal of elements at the end, and linear time
 *  insertion and removal of elements at the beginning or in the middle. The
 *  number of elements in a \p cpp::universal_vector may vary dynamically;
 *  memory management is automatic. The elements contained in a
 *  \p cpp::universal_vector reside in memory accessible by the \p cpp system
 *  and host systems.
 *
 *  \tparam T The element type of the \p cpp::universal_vector.
 *  \tparam Allocator The allocator type of the \p cpp::universal_vector.
 *          Defaults to \p cpp::universal_allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p cpp::universal_vector
 *  \see device_vector
 *  \see universal_vector
 */
template <typename T, typename Allocator = thrust::system::cpp::universal_allocator<T>>
using universal_vector = thrust::detail::vector_base<T, Allocator>;

}} // namespace system::cpp

namespace cpp
{
using thrust::system::cpp::vector;
using thrust::system::cpp::universal_vector;
}

THRUST_NAMESPACE_END
