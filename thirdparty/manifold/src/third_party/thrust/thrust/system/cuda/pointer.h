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

/*! \file thrust/system/cuda/memory.h
 *  \brief Managing memory associated with Thrust's Standard C++ system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <type_traits>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/reference.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{

/*! \p cuda::pointer stores a pointer to an object allocated in memory
 *  accessible by the \p cuda system. This type provides type safety when
 *  dispatching algorithms on ranges resident in \p cuda memory.
 *
 *  \p cuda::pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p cuda::pointer can be created with the function \p cuda::malloc, or by
 *  explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p cuda::pointer may be obtained by eiter
 *  its <tt>get</tt> member function or the \p raw_pointer_cast function.
 *
 *  \note \p cuda::pointer is not a "smart" pointer; it is the programmer's
 *        responsibility to deallocate memory pointed to by \p cuda::pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see cuda::malloc
 *  \see cuda::free
 *  \see raw_pointer_cast
 */
template <typename T>
using pointer = thrust::pointer<
  T,
  thrust::cuda_cub::tag,
  thrust::tagged_reference<T, thrust::cuda_cub::tag>
>;

/*! \p cuda::universal_pointer stores a pointer to an object allocated in
 *  memory accessible by the \p cuda system and host systems.
 *
 *  \p cuda::universal_pointer has pointer semantics: it may be dereferenced
 *  and manipulated with pointer arithmetic.
 *
 *  \p cuda::universal_pointer can be created with \p cuda::universal_allocator
 *  or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p cuda::universal_pointer may be
 *  obtained by eiter its <tt>get</tt> member function or the \p
 *  raw_pointer_cast function.
 *
 *  \note \p cuda::universal_pointer is not a "smart" pointer; it is the
 *        programmer's responsibility to deallocate memory pointed to by
 *        \p cuda::universal_pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see cuda::universal_allocator
 *  \see raw_pointer_cast
 */
template <typename T>
using universal_pointer = thrust::pointer<
  T,
  thrust::cuda_cub::tag,
  typename std::add_lvalue_reference<T>::type
>;

/*! \p cuda::reference is a wrapped reference to an object stored in memory
 *  accessible by the \p cuda system. \p cuda::reference is the type of the
 *  result of dereferencing a \p cuda::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 *
 *  \see cuda::pointer
 */
template <typename T>
using reference = thrust::tagged_reference<T, thrust::cuda_cub::tag>;

} // namespace cuda_cub

/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace thrust::system::cuda
 *  \brief \p thrust::system::cuda is the namespace containing functionality
 *  for allocating, manipulating, and deallocating memory available to Thrust's
 *  CUDA backend system. The identifiers are provided in a separate namespace
 *  underneath \p thrust::system for import convenience but are also
 *  aliased in the top-level <tt>thrust::cuda</tt> namespace for easy access.
 *
 */
namespace system { namespace cuda
{
using thrust::cuda_cub::pointer;
using thrust::cuda_cub::universal_pointer;
using thrust::cuda_cub::reference;
}} // namespace system::cuda
/*! \}
 */

/*! \namespace thrust::cuda
 *  \brief \p thrust::cuda is a top-level alias for \p thrust::system::cuda.
 */
namespace cuda
{
using thrust::cuda_cub::pointer;
using thrust::cuda_cub::universal_pointer;
using thrust::cuda_cub::reference;
} // namespace cuda

THRUST_NAMESPACE_END

