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
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/for_each.h>
#include <thrust/uninitialized_fill.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace allocator_traits_detail
{


template<typename Allocator>
  struct construct1_via_allocator
{
  Allocator &a;

  __host__ __device__
  construct1_via_allocator(Allocator &a)
    : a(a)
  {}

  template<typename T>
  inline __host__ __device__
  void operator()(T &x)
  {
    allocator_traits<Allocator>::construct(a, &x);
  }
};


// we need to construct T via the allocator if...
template<typename Allocator, typename T>
  struct needs_default_construct_via_allocator
    : thrust::detail::or_<
        has_member_construct1<Allocator,T>,               // if the Allocator does something interesting
        thrust::detail::not_<has_trivial_constructor<T> > // or if T's default constructor does something interesting
      >
{};


// we know that std::allocator::construct's only effect is to call T's
// default constructor, so we needn't use it for default construction
// unless T's constructor does something interesting
template<typename U, typename T>
  struct needs_default_construct_via_allocator<std::allocator<U>, T>
    : thrust::detail::not_<has_trivial_constructor<T> >
{};


template<typename Allocator, typename Pointer, typename Size>
__host__ __device__
  typename enable_if<
    needs_default_construct_via_allocator<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value
  >::type
    default_construct_range(Allocator &a, Pointer p, Size n)
{
  thrust::for_each_n(allocator_system<Allocator>::get(a), p, n, construct1_via_allocator<Allocator>(a));
}


template<typename Allocator, typename Pointer, typename Size>
__host__ __device__
  typename disable_if<
    needs_default_construct_via_allocator<
      Allocator,
      typename pointer_element<Pointer>::type
    >::value
  >::type
    default_construct_range(Allocator &a, Pointer p, Size n)
{
  thrust::uninitialized_fill_n(allocator_system<Allocator>::get(a), p, n, typename pointer_element<Pointer>::type());
}


} // end allocator_traits_detail


template<typename Allocator, typename Pointer, typename Size>
__host__ __device__
  void default_construct_range(Allocator &a, Pointer p, Size n)
{
  return allocator_traits_detail::default_construct_range(a,p,n);
}


} // end detail
THRUST_NAMESPACE_END

