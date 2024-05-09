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

/*! \file thrust/system/cpp/memory.h
 *  \brief Managing memory associated with Thrust's TBB system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <type_traits>
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/reference.h>

THRUST_NAMESPACE_BEGIN
namespace system { namespace cpp
{

/*! \p cpp::pointer stores a pointer to an object allocated in memory accessible
 *  by the \p cpp system. This type provides type safety when dispatching
 *  algorithms on ranges resident in \p cpp memory.
 *
 *  \p cpp::pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p cpp::pointer can be created with the function \p cpp::malloc, or by
 *  explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p cpp::pointer may be obtained by eiter its
 *  <tt>get</tt> member function or the \p raw_pointer_cast function.
 *
 *  \note \p cpp::pointer is not a "smart" pointer; it is the programmer's
 *        responsibility to deallocate memory pointed to by \p cpp::pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see cpp::malloc
 *  \see cpp::free
 *  \see raw_pointer_cast
 */
template <typename T>
using pointer = thrust::pointer<
  T,
  thrust::system::cpp::tag,
  thrust::tagged_reference<T, thrust::system::cpp::tag>
>;

/*! \p cpp::universal_pointer stores a pointer to an object allocated in memory
 * accessible by the \p cpp system and host systems.
 *
 *  \p cpp::universal_pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p cpp::universal_pointer can be created with \p cpp::universal_allocator
 *  or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p cpp::universal_pointer may be obtained
 *  by eiter its <tt>get</tt> member function or the \p raw_pointer_cast
 *  function.
 *
 *  \note \p cpp::universal_pointer is not a "smart" pointer; it is the
 *        programmer's responsibility to deallocate memory pointed to by
 *        \p cpp::universal_pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see cpp::universal_allocator
 *  \see raw_pointer_cast
 */
template <typename T>
using universal_pointer = thrust::pointer<
  T,
  thrust::system::cpp::tag,
  typename std::add_lvalue_reference<T>::type
>;

/*! \p reference is a wrapped reference to an object stored in memory available
 *  to the \p cpp system. \p reference is the type of the result of
 *  dereferencing a \p cpp::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 */
template <typename T>
using reference = thrust::reference<T, thrust::system::cpp::tag>;

}} // namespace system::cpp

/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace thrust::cpp
 *  \brief \p thrust::cpp is a top-level alias for \p thrust::system::cpp. */
namespace cpp
{
using thrust::system::cpp::pointer;
using thrust::system::cpp::universal_pointer;
using thrust::system::cpp::reference;
} // namespace cpp

THRUST_NAMESPACE_END

