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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/allocator/tagged_allocator.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/pair.h>
#include <thrust/memory.h>
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{


// XXX the pointer parameter given to tagged_allocator should be related to
//     the type of the expression get_temporary_buffer(system, n).first
//     without decltype, compromise on pointer<T,System>
template<typename T, typename System>
  class temporary_allocator
    : public thrust::detail::tagged_allocator<
               T, System, thrust::pointer<T,System>
             >
{
  private:
    typedef thrust::detail::tagged_allocator<
      T, System, thrust::pointer<T,System>
    > super_t;

    System &m_system;

  public:
    typedef typename super_t::pointer   pointer;
    typedef typename super_t::size_type size_type;

    inline __host__ __device__
    temporary_allocator(const temporary_allocator &other) :
      super_t(),
      m_system(other.m_system)
    {}

    inline __host__ __device__
    explicit temporary_allocator(thrust::execution_policy<System> &system) :
      super_t(),
      m_system(thrust::detail::derived_cast(system))
    {}

    __host__ __device__
    pointer allocate(size_type cnt);

    __host__ __device__
    void deallocate(pointer p, size_type n);

    __host__ __device__
    inline System &system()
    {
      return m_system;
    } // end system()

  private:
    typedef thrust::pair<pointer, size_type> pointer_and_size;
}; // end temporary_allocator


} // end detail
THRUST_NAMESPACE_END

#include <thrust/detail/allocator/temporary_allocator.inl>

