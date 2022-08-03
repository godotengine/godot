// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

// TODO: These need to be turned into proper Thrust algorithms (dispatch layer,
// backends, etc).

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/addressof.h>

#include <utility>
#include <new>
#include <thrust/detail/memory_wrapper.h>

THRUST_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__
void destroy_at(T* location)
{
  location->~T();
}

template <typename Allocator, typename T>
__host__ __device__
void destroy_at(Allocator const& alloc, T* location)
{
  typedef typename detail::allocator_traits<
    typename detail::remove_cv<
      typename detail::remove_reference<Allocator>::type
    >::type
  >::template rebind_traits<T>::other traits;

  typename traits::allocator_type alloc_T(alloc);

  traits::destroy(alloc_T, location);
}

template <typename ForwardIt>
__host__ __device__
ForwardIt destroy(ForwardIt first, ForwardIt last)
{
  for (; first != last; ++first)
    destroy_at(addressof(*first));

  return first;
}

template <typename Allocator, typename ForwardIt>
__host__ __device__
ForwardIt destroy(Allocator const& alloc, ForwardIt first, ForwardIt last)
{
  typedef typename iterator_traits<ForwardIt>::value_type T;
  typedef typename detail::allocator_traits<
    typename detail::remove_cv<
      typename detail::remove_reference<Allocator>::type
    >::type
  >::template rebind_traits<T>::other traits;

  typename traits::allocator_type alloc_T(alloc);

  for (; first != last; ++first)
    destroy_at(alloc_T, addressof(*first));

  return first;
}

template <typename ForwardIt, typename Size>
__host__ __device__
ForwardIt destroy_n(ForwardIt first, Size n)
{
  for (; n > 0; (void) ++first, --n)
    destroy_at(addressof(*first));

  return first;
}

template <typename Allocator, typename ForwardIt, typename Size>
__host__ __device__
ForwardIt destroy_n(Allocator const& alloc, ForwardIt first, Size n)
{
  typedef typename iterator_traits<ForwardIt>::value_type T;
  typedef typename detail::allocator_traits<
    typename detail::remove_cv<
      typename detail::remove_reference<Allocator>::type
    >::type
  >::template rebind_traits<T>::other traits;

  typename traits::allocator_type alloc_T(alloc);

  for (; n > 0; (void) ++first, --n)
    destroy_at(alloc_T, addressof(*first));

  return first;
}

#if THRUST_CPP_DIALECT >= 2011
template <typename ForwardIt, typename... Args>
__host__ __device__
void uninitialized_construct(
  ForwardIt first, ForwardIt last, Args const&... args
)
{
  using T = typename iterator_traits<ForwardIt>::value_type;

  ForwardIt current = first;
  #if !__CUDA_ARCH__ // No exceptions in CUDA.
  try {
  #endif
    for (; current != last; ++current)
      ::new (static_cast<void*>(addressof(*current))) T(args...);
  #if !__CUDA_ARCH__ // No exceptions in CUDA.
  } catch (...) {
    destroy(first, current);
    throw;
  }
  #endif
}

template <typename Allocator, typename ForwardIt, typename... Args>
void uninitialized_construct_with_allocator(
  Allocator const& alloc, ForwardIt first, ForwardIt last, Args const&... args
)
{
  using T = typename iterator_traits<ForwardIt>::value_type;
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >::template rebind_traits<T>;

  typename traits::allocator_type alloc_T(alloc);

  ForwardIt current = first;
  #if !__CUDA_ARCH__ // No exceptions in CUDA.
  try {
  #endif
    for (; current != last; ++current)
      traits::construct(alloc_T, addressof(*current), args...);
  #if !__CUDA_ARCH__ // No exceptions in CUDA.
  } catch (...) {
    destroy(alloc_T, first, current);
    throw;
  }
  #endif
}

template <typename ForwardIt, typename Size, typename... Args>
void uninitialized_construct_n(
  ForwardIt first, Size n, Args const&... args
)
{
  using T = typename iterator_traits<ForwardIt>::value_type;

  ForwardIt current = first;
  #if !__CUDA_ARCH__ // No exceptions in CUDA.
  try {
  #endif
    for (; n > 0; (void) ++current, --n)
      ::new (static_cast<void*>(addressof(*current))) T(args...);
  #if !__CUDA_ARCH__ // No exceptions in CUDA.
  } catch (...) {
    destroy(first, current);
    throw;
  }
  #endif
}

template <typename Allocator, typename ForwardIt, typename Size, typename... Args>
void uninitialized_construct_n_with_allocator(
  Allocator const& alloc, ForwardIt first, Size n, Args const&... args
)
{
  using T = typename iterator_traits<ForwardIt>::value_type;
  using traits = typename detail::allocator_traits<
    typename std::remove_cv<
      typename std::remove_reference<Allocator>::type
    >::type
  >::template rebind_traits<T>;

  typename traits::allocator_type alloc_T(alloc);

  ForwardIt current = first;
  #if !__CUDA_ARCH__ // No exceptions in CUDA.
  try {
  #endif
    for (; n > 0; (void) ++current, --n)
      traits::construct(alloc_T, addressof(*current), args...);
  #if !__CUDA_ARCH__ // No exceptions in CUDA.
  } catch (...) {
    destroy(alloc_T, first, current);
    throw;
  }
  #endif
}
#endif

///////////////////////////////////////////////////////////////////////////////

THRUST_NAMESPACE_END
