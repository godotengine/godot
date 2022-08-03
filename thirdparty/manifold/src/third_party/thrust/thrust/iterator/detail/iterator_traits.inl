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

#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/detail/iterator_category_to_traversal.h>
#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/void_t.h>

THRUST_NAMESPACE_BEGIN

template<typename Iterator>
  struct iterator_value
{
  typedef typename thrust::iterator_traits<Iterator>::value_type type;
}; // end iterator_value

template <typename Iterator>
using iterator_value_t = typename iterator_value<Iterator>::type;

template<typename Iterator>
  struct iterator_pointer
{
  typedef typename thrust::iterator_traits<Iterator>::pointer type;
}; // end iterator_pointer

template <typename Iterator>
using iterator_pointer_t = typename iterator_pointer<Iterator>::type;

template<typename Iterator>
  struct iterator_reference
{
  typedef typename iterator_traits<Iterator>::reference type;
}; // end iterator_reference

template <typename Iterator>
using iterator_reference_t = typename iterator_reference<Iterator>::type;

template<typename Iterator>
  struct iterator_difference
{
  typedef typename thrust::iterator_traits<Iterator>::difference_type type;
}; // end iterator_difference

template <typename Iterator>
using iterator_difference_t = typename iterator_difference<Iterator>::type;

namespace detail
{

template <typename Iterator, typename = void>
struct iterator_system_impl {};

template <typename Iterator>
struct iterator_system_impl<
  Iterator
, typename voider<
    typename iterator_traits<Iterator>::iterator_category
  >::type
>
  : detail::iterator_category_to_system<
      typename iterator_traits<Iterator>::iterator_category
    >
{};

} // namespace detail

template <typename Iterator>
struct iterator_system : detail::iterator_system_impl<Iterator> {};

// specialize iterator_system for void *, which has no category
template<>
  struct iterator_system<void *>
{
  typedef thrust::iterator_system<int*>::type type;
}; // end iterator_system<void*>

template<>
  struct iterator_system<const void *>
{
  typedef thrust::iterator_system<const int*>::type type;
}; // end iterator_system<void*>

template <typename Iterator>
using iterator_system_t = typename iterator_system<Iterator>::type;

template <typename Iterator>
  struct iterator_traversal
    : detail::iterator_category_to_traversal<
        typename thrust::iterator_traits<Iterator>::iterator_category
      >
{
}; // end iterator_traversal

namespace detail
{

template <typename T>
  struct is_iterator_traversal
    : thrust::detail::is_convertible<T, incrementable_traversal_tag>
{
}; // end is_iterator_traversal


template<typename T>
  struct is_iterator_system
    : detail::or_<
        detail::is_convertible<T, any_system_tag>,
        detail::or_<
          detail::is_convertible<T, host_system_tag>,
          detail::is_convertible<T, device_system_tag>
        >
      >
{
}; // end is_iterator_system


} // end namespace detail
THRUST_NAMESPACE_END

