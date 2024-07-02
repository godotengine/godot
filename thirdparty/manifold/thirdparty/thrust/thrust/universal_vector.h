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

/*! \file
 *  \brief A dynamically-sizable array of elements which resides in memory
 *         accessible to both hosts and devices.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/universal_allocator.h>

// #include the device system's vector header
#define __THRUST_DEVICE_SYSTEM_VECTOR_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/vector.h>
#include __THRUST_DEVICE_SYSTEM_VECTOR_HEADER
#undef __THRUST_DEVICE_SYSTEM_VECTOR_HEADER

THRUST_NAMESPACE_BEGIN

/*! \addtogroup containers Containers
 *  \{
 */

/*! A \p universal_vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p universal_vector may vary dynamically; memory management is
 *  automatic. The memory associated with a \p universal_vector resides in memory
 *  accessible to hosts and devices.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p universal_vector.
 *  \see device_vector
 */
using thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::universal_vector;

/*! \} // containers
 */

THRUST_NAMESPACE_END
