/*
 *  Copyright 2018 NVIDIA Corporation
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
#include <thrust/detail/alignment.h>

#if THRUST_CPP_DIALECT >= 2011
  #include <type_traits>
#endif

THRUST_NAMESPACE_BEGIN

namespace mr
{

template<typename T, class MR>
class allocator;

}

namespace detail
{

template<template <typename> class ExecutionPolicyCRTPBase>
struct allocator_aware_execution_policy
{
  template<typename MemoryResource>
  struct execute_with_memory_resource_type
  {
    typedef thrust::detail::execute_with_allocator<
      thrust::mr::allocator<
        thrust::detail::max_align_t,
        MemoryResource
      >,
      ExecutionPolicyCRTPBase
    > type;
  };

  template<typename Allocator>
  struct execute_with_allocator_type
  {
      typedef thrust::detail::execute_with_allocator<
        Allocator,
        ExecutionPolicyCRTPBase
      > type;
  };

  template<typename MemoryResource>
    typename execute_with_memory_resource_type<MemoryResource>::type
      operator()(MemoryResource * mem_res) const
  {
    return typename execute_with_memory_resource_type<MemoryResource>::type(mem_res);
  }

  template<typename Allocator>
    typename execute_with_allocator_type<Allocator&>::type
      operator()(Allocator &alloc) const
  {
    return typename execute_with_allocator_type<Allocator&>::type(alloc);
  }

  template<typename Allocator>
    typename execute_with_allocator_type<Allocator>::type
      operator()(const Allocator &alloc) const
  {
    return typename execute_with_allocator_type<Allocator>::type(alloc);
  }

#if THRUST_CPP_DIALECT >= 2011
  // just the rvalue overload
  // perfect forwarding doesn't help, because a const reference has to be turned
  // into a value by copying for the purpose of storing it in execute_with_allocator
  template<typename Allocator,
      typename std::enable_if<!std::is_lvalue_reference<Allocator>::value>::type * = nullptr>
    typename execute_with_allocator_type<Allocator>::type
      operator()(Allocator &&alloc) const
  {
    return typename execute_with_allocator_type<Allocator>::type(std::move(alloc));
  }
#endif
};

} // end namespace detail

THRUST_NAMESPACE_END
