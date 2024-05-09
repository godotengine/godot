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

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/execute_with_allocator_fwd.h>
#include <thrust/pair.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/integer_math.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template <
    typename T
  , typename Allocator
  , template <typename> class BaseSystem
>
__host__
thrust::pair<T*, std::ptrdiff_t>
get_temporary_buffer(
    thrust::detail::execute_with_allocator<Allocator, BaseSystem>& system
  , std::ptrdiff_t n
    )
{
  typedef typename thrust::detail::remove_reference<Allocator>::type naked_allocator;
  typedef typename thrust::detail::allocator_traits<naked_allocator> alloc_traits;
  typedef typename alloc_traits::void_pointer                        void_pointer;
  typedef typename alloc_traits::size_type                           size_type;
  typedef typename alloc_traits::value_type                          value_type;

  // How many elements of type value_type do we need to accommodate n elements
  // of type T?
  size_type num_elements = divide_ri(sizeof(T) * n, sizeof(value_type));

  void_pointer ptr = alloc_traits::allocate(system.get_allocator(), num_elements);

  // Return the pointer and the number of elements of type T allocated.
  return thrust::make_pair(thrust::reinterpret_pointer_cast<T*>(ptr),n);
}

template <
    typename Pointer
  , typename Allocator
  , template <typename> class BaseSystem
>
__host__
void
return_temporary_buffer(
    thrust::detail::execute_with_allocator<Allocator, BaseSystem>& system
  , Pointer p
  , std::ptrdiff_t n
    )
{
  typedef typename thrust::detail::remove_reference<Allocator>::type naked_allocator;
  typedef typename thrust::detail::allocator_traits<naked_allocator> alloc_traits;
  typedef typename alloc_traits::pointer                             pointer;
  typedef typename alloc_traits::size_type                           size_type;
  typedef typename alloc_traits::value_type                          value_type;
  typedef typename thrust::detail::pointer_traits<Pointer>::element_type T;

  size_type num_elements = divide_ri(sizeof(T) * n, sizeof(value_type));

  pointer to_ptr = thrust::reinterpret_pointer_cast<pointer>(p);
  alloc_traits::deallocate(system.get_allocator(), to_ptr, num_elements);
}

#if THRUST_CPP_DIALECT >= 2011

template <
    typename T,
    template <typename> class BaseSystem,
    typename Allocator,
    typename ...Dependencies
>
__host__
thrust::pair<T*, std::ptrdiff_t>
get_temporary_buffer(
    thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>& system,
    std::ptrdiff_t n
    )
{
  typedef typename thrust::detail::remove_reference<Allocator>::type naked_allocator;
  typedef typename thrust::detail::allocator_traits<naked_allocator> alloc_traits;
  typedef typename alloc_traits::void_pointer                        void_pointer;
  typedef typename alloc_traits::size_type                           size_type;
  typedef typename alloc_traits::value_type                          value_type;

  // How many elements of type value_type do we need to accommodate n elements
  // of type T?
  size_type num_elements = divide_ri(sizeof(T) * n, sizeof(value_type));

  void_pointer ptr = alloc_traits::allocate(system.get_allocator(), num_elements);

  // Return the pointer and the number of elements of type T allocated.
  return thrust::make_pair(thrust::reinterpret_pointer_cast<T*>(ptr),n);
}

template <
    typename Pointer,
    template <typename> class BaseSystem,
    typename Allocator,
    typename ...Dependencies
>
__host__
void
return_temporary_buffer(
    thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>& system,
    Pointer p,
    std::ptrdiff_t n
    )
{
  typedef typename thrust::detail::remove_reference<Allocator>::type naked_allocator;
  typedef typename thrust::detail::allocator_traits<naked_allocator> alloc_traits;
  typedef typename alloc_traits::pointer                             pointer;
  typedef typename alloc_traits::size_type                           size_type;
  typedef typename alloc_traits::value_type                          value_type;
  typedef typename thrust::detail::pointer_traits<Pointer>::element_type T;

  size_type num_elements = divide_ri(sizeof(T) * n, sizeof(value_type));

  pointer to_ptr = thrust::reinterpret_pointer_cast<pointer>(p);
  alloc_traits::deallocate(system.get_allocator(), to_ptr, num_elements);
}

#endif

} // namespace detail

THRUST_NAMESPACE_END
