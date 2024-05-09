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

// allocator_traits::rebind_alloc and allocator::rebind_traits are from libc++,
// dual licensed under the MIT and the University of Illinois Open Source
// Licenses.

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/detail/type_traits/has_member_function.h>
#include <thrust/detail/type_traits.h>

#include <thrust/detail/memory_wrapper.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{


// forward declaration for has_member_system
template<typename Alloc> struct allocator_system;


namespace allocator_traits_detail
{

__THRUST_DEFINE_HAS_NESTED_TYPE(has_value_type, value_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_pointer, pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_const_pointer, const_pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_reference, reference)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_const_reference, const_reference)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_void_pointer, void_pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_const_void_pointer, const_void_pointer)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_difference_type, difference_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_size_type, size_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_propagate_on_container_copy_assignment, propagate_on_container_copy_assignment)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_propagate_on_container_move_assignment, propagate_on_container_move_assignment)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_propagate_on_container_swap, propagate_on_container_swap)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_system_type, system_type)
__THRUST_DEFINE_HAS_NESTED_TYPE(has_is_always_equal, is_always_equal)
__THRUST_DEFINE_HAS_MEMBER_FUNCTION(has_member_system_impl, system)

template<typename Alloc, typename U>
  struct has_rebind
{
  typedef char yes_type;
  typedef int  no_type;

  template<typename S>
  static yes_type test(typename S::template rebind<U>::other*);
  template<typename S>
  static no_type  test(...);

  static bool const value = sizeof(test<U>(0)) == sizeof(yes_type);

  typedef thrust::detail::integral_constant<bool, value> type;
};

// The following fields of std::allocator have been deprecated (since C++17).
// There's no way to detect it other than explicit specialization.
#if THRUST_CPP_DIALECT >= 2017
#define THRUST_SPECIALIZE_DEPRECATED(trait_name)                               \
template <typename T>                                                          \
struct trait_name<std::allocator<T>> : false_type {};

THRUST_SPECIALIZE_DEPRECATED(has_is_always_equal)
THRUST_SPECIALIZE_DEPRECATED(has_pointer)
THRUST_SPECIALIZE_DEPRECATED(has_const_pointer)
THRUST_SPECIALIZE_DEPRECATED(has_reference)
THRUST_SPECIALIZE_DEPRECATED(has_const_reference)

#undef THRUST_SPECIALIZE_DEPRECATED

template<typename T, typename U>
struct has_rebind<std::allocator<T>, U> : false_type {};
#endif

template<typename T>
  struct nested_pointer
{
  typedef typename T::pointer type;
};

template<typename T>
  struct nested_const_pointer
{
  typedef typename T::const_pointer type;
};

template<typename T>
  struct nested_reference
{
  typedef typename T::reference type;
};

template<typename T>
  struct nested_const_reference
{
  typedef typename T::const_reference type;
};

template<typename T>
  struct nested_void_pointer
{
  typedef typename T::void_pointer type;
};

template<typename T>
  struct nested_const_void_pointer
{
  typedef typename T::const_void_pointer type;
};

template<typename T>
  struct nested_difference_type
{
  typedef typename T::difference_type type;
};

template<typename T>
  struct nested_size_type
{
  typedef typename T::size_type type;
};

template<typename T>
  struct nested_propagate_on_container_copy_assignment
{
  typedef typename T::propagate_on_container_copy_assignment type;
};

template<typename T>
  struct nested_propagate_on_container_move_assignment
{
  typedef typename T::propagate_on_container_move_assignment type;
};

template<typename T>
  struct nested_propagate_on_container_swap
{
  typedef typename T::propagate_on_container_swap type;
};

template<typename T>
  struct nested_is_always_equal
{
  typedef typename T::is_always_equal type;
};

template<typename T>
  struct nested_system_type
{
  typedef typename T::system_type type;
};

template<typename Alloc>
  struct has_member_system
{
  typedef typename allocator_system<Alloc>::type system_type;

  typedef typename has_member_system_impl<Alloc, system_type&(void)>::type type;
  static const bool value = type::value;
};

template<class Alloc, class U, bool = has_rebind<Alloc, U>::value>
  struct rebind_alloc
{
    typedef typename Alloc::template rebind<U>::other type;
};

#if THRUST_CPP_DIALECT >= 2011
template<template<typename, typename...> class Alloc,
         typename T, typename... Args, typename U>
  struct rebind_alloc<Alloc<T, Args...>, U, true>
{
    typedef typename Alloc<T, Args...>::template rebind<U>::other type;
};

template<template<typename, typename...> class Alloc,
         typename T, typename... Args, typename U>
  struct rebind_alloc<Alloc<T, Args...>, U, false>
{
    typedef Alloc<U, Args...> type;
};
#else // C++03
template <template <typename> class Alloc, typename T, typename U>
  struct rebind_alloc<Alloc<T>, U, true>
{
    typedef typename Alloc<T>::template rebind<U>::other type;
};

template <template <typename> class Alloc, typename T, typename U>
  struct rebind_alloc<Alloc<T>, U, false>
{
    typedef Alloc<U> type;
};

template<template<typename, typename> class Alloc,
         typename T, typename A0, typename U>
  struct rebind_alloc<Alloc<T, A0>, U, true>
{
    typedef typename Alloc<T, A0>::template rebind<U>::other type;
};

template<template<typename, typename> class Alloc,
         typename T, typename A0, typename U>
  struct rebind_alloc<Alloc<T, A0>, U, false>
{
    typedef Alloc<U, A0> type;
};

template<template<typename, typename, typename> class Alloc,
         typename T, typename A0, typename A1, typename U>
  struct rebind_alloc<Alloc<T, A0, A1>, U, true>
{
    typedef typename Alloc<T, A0, A1>::template rebind<U>::other type;
};

template<template<typename, typename, typename> class Alloc,
         typename T, typename A0, typename A1, typename U>
  struct rebind_alloc<Alloc<T, A0, A1>, U, false>
{
    typedef Alloc<U, A0, A1> type;
};

template<template<typename, typename, typename, typename> class Alloc,
         typename T, typename A0, typename A1, typename A2, typename U>
  struct rebind_alloc<Alloc<T, A0, A1, A2>, U, true>
{
    typedef typename Alloc<T, A0, A1, A2>::template rebind<U>::other type;
};

template<template<typename, typename, typename, typename> class Alloc,
         typename T, typename A0, typename A1, typename A2, typename U>
  struct rebind_alloc<Alloc<T, A0, A1, A2>, U, false>
{
    typedef Alloc<U, A0, A1, A2> type;
};
#endif

} // end allocator_traits_detail


template<typename Alloc>
  struct allocator_traits
{
  typedef Alloc allocator_type;

  typedef typename allocator_type::value_type value_type;

  typedef typename eval_if<
    allocator_traits_detail::has_pointer<allocator_type>::value,
    allocator_traits_detail::nested_pointer<allocator_type>,
    identity_<value_type*>
  >::type pointer;

  private:
    template<typename T>
      struct rebind_pointer
    {
      typedef typename pointer_traits<pointer>::template rebind<T>::other type;
    };

  public:

  typedef typename eval_if<
    allocator_traits_detail::has_const_pointer<allocator_type>::value,
    allocator_traits_detail::nested_const_pointer<allocator_type>,
    rebind_pointer<const value_type>
  >::type const_pointer;

  typedef typename eval_if<
    allocator_traits_detail::has_void_pointer<allocator_type>::value,
    allocator_traits_detail::nested_void_pointer<allocator_type>,
    rebind_pointer<void>
  >::type void_pointer;

  typedef typename eval_if<
    allocator_traits_detail::has_const_void_pointer<allocator_type>::value,
    allocator_traits_detail::nested_const_void_pointer<allocator_type>,
    rebind_pointer<const void>
  >::type const_void_pointer;

  typedef typename eval_if<
    allocator_traits_detail::has_difference_type<allocator_type>::value,
    allocator_traits_detail::nested_difference_type<allocator_type>,
    pointer_difference<pointer>
  >::type difference_type;

  typedef typename eval_if<
    allocator_traits_detail::has_size_type<allocator_type>::value,
    allocator_traits_detail::nested_size_type<allocator_type>,
    make_unsigned<difference_type>
  >::type size_type;

  typedef typename eval_if<
    allocator_traits_detail::has_propagate_on_container_copy_assignment<allocator_type>::value,
    allocator_traits_detail::nested_propagate_on_container_copy_assignment<allocator_type>,
    identity_<false_type>
  >::type propagate_on_container_copy_assignment;

  typedef typename eval_if<
    allocator_traits_detail::has_propagate_on_container_move_assignment<allocator_type>::value,
    allocator_traits_detail::nested_propagate_on_container_move_assignment<allocator_type>,
    identity_<false_type>
  >::type propagate_on_container_move_assignment;

  typedef typename eval_if<
    allocator_traits_detail::has_propagate_on_container_swap<allocator_type>::value,
    allocator_traits_detail::nested_propagate_on_container_swap<allocator_type>,
    identity_<false_type>
  >::type propagate_on_container_swap;

  typedef typename eval_if<
    allocator_traits_detail::has_is_always_equal<allocator_type>::value,
    allocator_traits_detail::nested_is_always_equal<allocator_type>,
    is_empty<allocator_type>
  >::type is_always_equal;

  typedef typename eval_if<
    allocator_traits_detail::has_system_type<allocator_type>::value,
    allocator_traits_detail::nested_system_type<allocator_type>,
    thrust::iterator_system<pointer>
  >::type system_type;

  // XXX rebind and rebind_traits are alias templates
  //     and so are omitted while c++11 is unavailable

#if THRUST_CPP_DIALECT >= 2011
  template <typename U>
  using rebind_alloc =
    typename allocator_traits_detail::rebind_alloc<allocator_type, U>::type;

  template <typename U>
  using rebind_traits = allocator_traits<rebind_alloc<U>>;

  // We define this nested type alias for compatibility with the C++03-style
  // rebind_* mechanisms.
  using other = allocator_traits;
#else
  template <typename U>
  struct rebind_alloc
  {
    typedef typename
      allocator_traits_detail::rebind_alloc<allocator_type, U>::type other;
  };
  template <typename U>
  struct rebind_traits
  {
    typedef allocator_traits<typename rebind_alloc<U>::other> other;
  };
#endif

  // Deprecated std::allocator typedefs that we need:
  typedef typename thrust::detail::pointer_traits<pointer>::reference reference;
  typedef typename thrust::detail::pointer_traits<const_pointer>::reference const_reference;

  inline __host__ __device__
  static pointer allocate(allocator_type &a, size_type n);

  inline __host__ __device__
  static pointer allocate(allocator_type &a, size_type n, const_void_pointer hint);

  inline __host__ __device__
  static void deallocate(allocator_type &a, pointer p, size_type n);

  // XXX should probably change T* to pointer below and then relax later

  template<typename T>
  inline __host__ __device__ static void construct(allocator_type &a, T *p);
  
  template<typename T, typename Arg1>
  inline __host__ __device__ static void construct(allocator_type &a, T *p, const Arg1 &arg1);

#if THRUST_CPP_DIALECT >= 2011
  template<typename T, typename... Args>
  inline __host__ __device__ static void construct(allocator_type &a, T *p, Args&&... args);
#endif

  template<typename T>
  inline __host__ __device__ static void destroy(allocator_type &a, T *p);

  inline __host__ __device__
  static size_type max_size(const allocator_type &a);
}; // end allocator_traits


// we consider a type an allocator if T::value_type exists
// it doesn't make much sense (containers, which are not allocators, will fulfill this requirement),
// but allocator_traits is specified to work for any type with that nested typedef
template<typename T>
  struct is_allocator
    : allocator_traits_detail::has_value_type<T>
{};


// XXX consider moving this non-standard functionality inside allocator_traits
template<typename Alloc>
  struct allocator_system
{
  // the type of the allocator's system
  typedef typename eval_if<
    allocator_traits_detail::has_system_type<Alloc>::value,
    allocator_traits_detail::nested_system_type<Alloc>,
    thrust::iterator_system<
      typename allocator_traits<Alloc>::pointer
    >
  >::type type;

  // the type that get returns
  typedef typename eval_if<
    allocator_traits_detail::has_member_system<Alloc>::value, // if Alloc.system() exists
    add_reference<type>,                                      // then get() needs to return a reference
    identity_<type>                                           // else get() needs to return a value
  >::type get_result_type;

  __host__ __device__
  inline static get_result_type get(Alloc &a);
};


} // end detail
THRUST_NAMESPACE_END

#include <thrust/detail/allocator/allocator_traits.inl>

