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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/is_metafunction_defined.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/iterator/iterator_traits.h>
#include <cstddef>
#include <type_traits>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template<typename Ptr> struct pointer_element;

template<template<typename> class Ptr, typename Arg>
  struct pointer_element<Ptr<Arg> >
{
  typedef Arg type;
};

template<template<typename,typename> class Ptr, typename Arg1, typename Arg2>
  struct pointer_element<Ptr<Arg1,Arg2> >
{
  typedef Arg1 type;
};

template<template<typename,typename,typename> class Ptr, typename Arg1, typename Arg2, typename Arg3>
  struct pointer_element<Ptr<Arg1,Arg2,Arg3> >
{
  typedef Arg1 type;
};

template<template<typename,typename,typename,typename> class Ptr, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  struct pointer_element<Ptr<Arg1,Arg2,Arg3,Arg4> >
{
  typedef Arg1 type;
};

template<template<typename,typename,typename,typename,typename> class Ptr, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  struct pointer_element<Ptr<Arg1,Arg2,Arg3,Arg4,Arg5> >
{
  typedef Arg1 type;
};

template<typename T>
  struct pointer_element<T*>
{
  typedef T type;
};

template<typename Ptr>
  struct pointer_difference
{
  typedef typename Ptr::difference_type type;
};

template<typename T>
  struct pointer_difference<T*>
{
  typedef std::ptrdiff_t type;
};

template<typename Ptr, typename T> struct rebind_pointer;

template<typename T, typename U>
  struct rebind_pointer<T*,U>
{
  using type = U*;
};

// Rebind generic fancy pointers.
template<template<typename, typename...> class Ptr, typename OldT, typename... Tail, typename T>
  struct rebind_pointer<Ptr<OldT,Tail...>,T>
{
  using type = Ptr<T,Tail...>;
};

// Rebind `thrust::pointer`-like things with `thrust::reference`-like references.
template<template<typename, typename, typename, typename...> class Ptr, typename OldT, typename Tag,
         template<typename...> class Ref, typename... RefTail,
         typename... PtrTail, typename T>
  struct rebind_pointer<Ptr<OldT,Tag,Ref<OldT,RefTail...>,PtrTail...>,T>
{
//  static_assert(std::is_same<OldT, Tag>::value, "0");
  using type = Ptr<T,Tag,Ref<T,RefTail...>,PtrTail...>;
};

// Rebind `thrust::pointer`-like things with `thrust::reference`-like references
// and templated derived types.
template<template<typename, typename, typename, typename...> class Ptr, typename OldT, typename Tag,
         template<typename...> class Ref, typename... RefTail,
         template<typename...> class DerivedPtr, typename... DerivedPtrTail,
         typename T>
  struct rebind_pointer<Ptr<OldT,Tag,Ref<OldT,RefTail...>,DerivedPtr<OldT,DerivedPtrTail...>>,T>
{
//  static_assert(std::is_same<OldT, Tag>::value, "1");
  using type = Ptr<T,Tag,Ref<T,RefTail...>,DerivedPtr<T,DerivedPtrTail...>>;
};

// Rebind `thrust::pointer`-like things with native reference types.
template<template<typename, typename, typename, typename...> class Ptr, typename OldT, typename Tag,
         typename... PtrTail, typename T>
  struct rebind_pointer<Ptr<OldT,Tag,typename std::add_lvalue_reference<OldT>::type,PtrTail...>,T>
{
//  static_assert(std::is_same<OldT, Tag>::value, "2");
  using type = Ptr<T,Tag,typename std::add_lvalue_reference<T>::type,PtrTail...>;
};

// Rebind `thrust::pointer`-like things with native reference types and templated
// derived types.
template<template<typename, typename, typename, typename...> class Ptr, typename OldT, typename Tag,
         template<typename...> class DerivedPtr, typename... DerivedPtrTail,
         typename T>
  struct rebind_pointer<Ptr<OldT,Tag,typename std::add_lvalue_reference<OldT>::type,DerivedPtr<OldT,DerivedPtrTail...>>,T>
{
//  static_assert(std::is_same<OldT, Tag>::value, "3");
  using type = Ptr<T,Tag,typename std::add_lvalue_reference<T>::type,DerivedPtr<T,DerivedPtrTail...>>;
};

__THRUST_DEFINE_HAS_NESTED_TYPE(has_raw_pointer, raw_pointer)

namespace pointer_traits_detail
{

template<typename Ptr, typename Enable = void> struct pointer_raw_pointer_impl {};

template<typename T>
  struct pointer_raw_pointer_impl<T*>
{
  typedef T* type;
};

template<typename Ptr>
  struct pointer_raw_pointer_impl<Ptr, typename enable_if<has_raw_pointer<Ptr>::value>::type>
{
  typedef typename Ptr::raw_pointer type;
};

} // end pointer_traits_detail

template<typename T>
  struct pointer_raw_pointer
    : pointer_traits_detail::pointer_raw_pointer_impl<T>
{};

namespace pointer_traits_detail
{

template<typename Void>
  struct capture_address
{
  template<typename T>
  __host__ __device__
  capture_address(T &r)
    : m_addr(&r)
  {}

  inline __host__ __device__
  Void *operator&() const
  {
    return m_addr;
  }

  Void *m_addr;
};

// metafunction to compute the type of pointer_to's parameter below
template<typename T>
  struct pointer_to_param
    : thrust::detail::eval_if<
        thrust::detail::is_void<T>::value,
        thrust::detail::identity_<capture_address<T> >,
        thrust::detail::add_reference<T>
      >
{};

}

template<typename Ptr>
  struct pointer_traits
{
  typedef Ptr                                    pointer;
  typedef typename Ptr::reference                reference;
  typedef typename pointer_element<Ptr>::type    element_type;
  typedef typename pointer_difference<Ptr>::type difference_type;

  template<typename U>
    struct rebind
  {
    typedef typename rebind_pointer<Ptr,U>::type other;
  };

  __host__ __device__
  inline static pointer pointer_to(typename pointer_traits_detail::pointer_to_param<element_type>::type r)
  {
    // XXX this is supposed to be pointer::pointer_to(&r); (i.e., call a static member function of pointer called pointer_to)
    //     assume that pointer has a constructor from raw pointer instead

    return pointer(&r);
  }

  // thrust additions follow
  typedef typename pointer_raw_pointer<Ptr>::type raw_pointer;

  __host__ __device__
  inline static raw_pointer get(pointer ptr)
  {
    return ptr.get();
  }
};

template<typename T>
  struct pointer_traits<T*>
{
  typedef T*                                    pointer;
  typedef T&                                    reference;
  typedef T                                     element_type;
  typedef typename pointer_difference<T*>::type difference_type;

  template<typename U>
    struct rebind
  {
    typedef U* other;
  };

  __host__ __device__
  inline static pointer pointer_to(typename pointer_traits_detail::pointer_to_param<element_type>::type r)
  {
    return &r;
  }

  // thrust additions follow
  typedef typename pointer_raw_pointer<T*>::type raw_pointer;

  __host__ __device__
  inline static raw_pointer get(pointer ptr)
  {
    return ptr;
  }
};

template<>
  struct pointer_traits<void*>
{
  typedef void*                                    pointer;
  typedef void                                     reference;
  typedef void                                     element_type;
  typedef pointer_difference<void*>::type          difference_type;

  template<typename U>
    struct rebind
  {
    typedef U* other;
  };

  __host__ __device__
  inline static pointer pointer_to(pointer_traits_detail::pointer_to_param<element_type>::type r)
  {
    return &r;
  }

  // thrust additions follow
  typedef pointer_raw_pointer<void*>::type raw_pointer;

  __host__ __device__
  inline static raw_pointer get(pointer ptr)
  {
    return ptr;
  }
};

template<>
  struct pointer_traits<const void*>
{
  typedef const void*                           pointer;
  typedef const void                            reference;
  typedef const void                            element_type;
  typedef pointer_difference<const void*>::type difference_type;

  template<typename U>
    struct rebind
  {
    typedef U* other;
  };

  __host__ __device__
  inline static pointer pointer_to(pointer_traits_detail::pointer_to_param<element_type>::type r)
  {
    return &r;
  }

  // thrust additions follow
  typedef pointer_raw_pointer<const void*>::type raw_pointer;

  __host__ __device__
  inline static raw_pointer get(pointer ptr)
  {
    return ptr;
  }
};

template<typename FromPtr, typename ToPtr>
  struct is_pointer_system_convertible
    : thrust::detail::is_convertible<
        typename iterator_system<FromPtr>::type,
        typename iterator_system<ToPtr>::type
      >
{};

template<typename FromPtr, typename ToPtr>
  struct is_pointer_convertible
    : thrust::detail::and_<
        thrust::detail::is_convertible<
          typename pointer_element<FromPtr>::type *,
          typename pointer_element<ToPtr>::type *
        >,
        is_pointer_system_convertible<FromPtr, ToPtr>
      >
{};

template<typename FromPtr, typename ToPtr>
  struct is_void_pointer_system_convertible
    : thrust::detail::and_<
        thrust::detail::is_same<
          typename pointer_element<FromPtr>::type,
          void
        >,
        is_pointer_system_convertible<FromPtr, ToPtr>
      >
{};

// this could be a lot better, but for our purposes, it's probably
// sufficient just to check if pointer_raw_pointer<T> has meaning
template<typename T>
  struct is_thrust_pointer
    : is_metafunction_defined<pointer_raw_pointer<T> >
{};

// avoid inspecting traits of the arguments if they aren't known to be pointers
template<typename FromPtr, typename ToPtr>
  struct lazy_is_pointer_convertible
    : thrust::detail::eval_if<
        is_thrust_pointer<FromPtr>::value && is_thrust_pointer<ToPtr>::value,
        is_pointer_convertible<FromPtr,ToPtr>,
        thrust::detail::identity_<thrust::detail::false_type>
      >
{};

template<typename FromPtr, typename ToPtr>
  struct lazy_is_void_pointer_system_convertible
    : thrust::detail::eval_if<
        is_thrust_pointer<FromPtr>::value && is_thrust_pointer<ToPtr>::value,
        is_void_pointer_system_convertible<FromPtr,ToPtr>,
        thrust::detail::identity_<thrust::detail::false_type>
      >
{};

template<typename FromPtr, typename ToPtr, typename T = void>
  struct enable_if_pointer_is_convertible
    : thrust::detail::enable_if<
        lazy_is_pointer_convertible<FromPtr,ToPtr>::type::value,
        T
      >
{};

template<typename FromPtr, typename ToPtr, typename T = void>
  struct enable_if_void_pointer_is_system_convertible
    : thrust::detail::enable_if<
        lazy_is_void_pointer_system_convertible<FromPtr,ToPtr>::type::value,
        T
      >
{};


} // end detail
THRUST_NAMESPACE_END

