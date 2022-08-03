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


/*! \file thrust/iterator/iterator_categories.h
 *  \brief Types for reasoning about the categories of iterators
 */

/*
 * (C) Copyright Jeremy Siek 2002.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */


#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/iterator_category_with_system_and_traversal.h>
#include <thrust/iterator/detail/iterator_traversal_tags.h>
#include <thrust/iterator/detail/device_system_tag.h>

// #include this for stl's iterator tags
#include <iterator>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \addtogroup iterator_tags Iterator Tags
 *  \ingroup iterators
 *  \addtogroup iterator_tag_classes Iterator Tag Classes
 *  \ingroup iterator_tags
 *  \{
 */

/*! \p input_device_iterator_tag is an empty class: it has no member functions,
 *  member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Input Device Iterator concept within the C++ type
 *  system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags  iterator_traits,
 *  output_device_iterator_tag, forward_device_iterator_tag,
 *  bidirectional_device_iterator_tag, random_access_device_iterator_tag,
 *  input_host_iterator_tag, output_host_iterator_tag, forward_host_iterator_tag,
 *  bidirectional_host_iterator_tag, random_access_host_iterator_tag
 */
struct input_device_iterator_tag
  : thrust::detail::iterator_category_with_system_and_traversal<
      std::input_iterator_tag,
      thrust::device_system_tag,
      thrust::single_pass_traversal_tag
    >
{};

/*! \p output_device_iterator_tag is an empty class: it has no member functions,
 *  member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Output Device Iterator concept within the C++ type
 *  system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags  iterator_traits,
 *  input_device_iterator_tag, forward_device_iterator_tag,
 *  bidirectional_device_iterator_tag, random_access_device_iterator_tag,
 *  input_host_iterator_tag, output_host_iterator_tag, forward_host_iterator_tag,
 *  bidirectional_host_iterator_tag, random_access_host_iterator_tag
 */
struct output_device_iterator_tag
  : thrust::detail::iterator_category_with_system_and_traversal<
      std::output_iterator_tag,
      thrust::device_system_tag,
      thrust::single_pass_traversal_tag
    >
{};

/*! \p forward_device_iterator_tag is an empty class: it has no member functions,
 *  member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Forward Device Iterator concept within the C++ type
 *  system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags  iterator_traits,
 *  input_device_iterator_tag, output_device_iterator_tag,
 *  bidirectional_device_iterator_tag, random_access_device_iterator_tag,
 *  input_host_iterator_tag, output_host_iterator_tag, forward_host_iterator_tag,
 *  bidirectional_host_iterator_tag, random_access_host_iterator_tag
 */
struct forward_device_iterator_tag
  : thrust::detail::iterator_category_with_system_and_traversal<
      std::forward_iterator_tag,
      thrust::device_system_tag,
      thrust::forward_traversal_tag
    >
{};

/*! \p bidirectional_device_iterator_tag is an empty class: it has no member
 *  functions, member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Bidirectional Device Iterator concept within the C++
 *  type system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags 
 *  iterator_traits, input_device_iterator_tag, output_device_iterator_tag,
 *  forward_device_iterator_tag, random_access_device_iterator_tag,
 *  input_host_iterator_tag, output_host_iterator_tag, forward_host_iterator_tag,
 *  bidirectional_host_iterator_tag, random_access_host_iterator_tag
 */
struct bidirectional_device_iterator_tag
  : thrust::detail::iterator_category_with_system_and_traversal<
      std::bidirectional_iterator_tag,
      thrust::device_system_tag,
      thrust::bidirectional_traversal_tag
    >
{};

/*! \p random_access_device_iterator_tag is an empty class: it has no member
 *  functions, member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Random Access Device Iterator concept within the C++
 *  type system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags 
 *  iterator_traits, input_device_iterator_tag, output_device_iterator_tag,
 *  forward_device_iterator_tag, bidirectional_device_iterator_tag,
 *  input_host_iterator_tag, output_host_iterator_tag, forward_host_iterator_tag,
 *  bidirectional_host_iterator_tag, random_access_host_iterator_tag
 */
struct random_access_device_iterator_tag
  : thrust::detail::iterator_category_with_system_and_traversal<
      std::random_access_iterator_tag,
      thrust::device_system_tag,
      thrust::random_access_traversal_tag
    >
{};

/*! \p input_host_iterator_tag is an empty class: it has no member
 *  functions, member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Input Host Iterator concept within the C++
 *  type system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags 
 *  iterator_traits, input_device_iterator_tag, output_device_iterator_tag,
 *  forward_device_iterator_tag, bidirectional_device_iterator_tag,
 *  random_access_device_iterator_tag,
 *  output_host_iterator_tag, forward_host_iterator_tag,
 *  bidirectional_host_iterator_tag, random_access_host_iterator_tag
 */
typedef std::input_iterator_tag input_host_iterator_tag;

/*! \p output_host_iterator_tag is an empty class: it has no member
 *  functions, member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Output Host Iterator concept within the C++
 *  type system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags 
 *  iterator_traits, input_device_iterator_tag, output_device_iterator_tag,
 *  forward_device_iterator_tag, bidirectional_device_iterator_tag,
 *  random_access_device_iterator_tag,
 *  input_host_iterator_tag, forward_host_iterator_tag,
 *  bidirectional_host_iterator_tag, random_access_host_iterator_tag
 */
typedef std::output_iterator_tag output_host_iterator_tag;

/*! \p forward_host_iterator_tag is an empty class: it has no member
 *  functions, member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Forward Host Iterator concept within the C++
 *  type system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags 
 *  iterator_traits, input_device_iterator_tag, output_device_iterator_tag,
 *  forward_device_iterator_tag, bidirectional_device_iterator_tag,
 *  random_access_device_iterator_tag,
 *  input_host_iterator_tag, output_host_iterator_tag,
 *  bidirectional_host_iterator_tag, random_access_host_iterator_tag
 */
typedef std::forward_iterator_tag forward_host_iterator_tag;

/*! \p bidirectional_host_iterator_tag is an empty class: it has no member
 *  functions, member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Forward Host Iterator concept within the C++
 *  type system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags 
 *  iterator_traits, input_device_iterator_tag, output_device_iterator_tag,
 *  forward_device_iterator_tag, bidirectional_device_iterator_tag,
 *  random_access_device_iterator_tag,
 *  input_host_iterator_tag, output_host_iterator_tag,
 *  forward_host_iterator_tag, random_access_host_iterator_tag
 */
typedef std::bidirectional_iterator_tag bidirectional_host_iterator_tag;

/*! \p random_access_host_iterator_tag is an empty class: it has no member
 *  functions, member variables, or nested types. It is used solely as a "tag": a
 *  representation of the Forward Host Iterator concept within the C++
 *  type system.
 *
 *  \see https://en.cppreference.com/w/cpp/iterator/iterator_tags 
 *  iterator_traits, input_device_iterator_tag, output_device_iterator_tag,
 *  forward_device_iterator_tag, bidirectional_device_iterator_tag,
 *  random_access_device_iterator_tag,
 *  input_host_iterator_tag, output_host_iterator_tag,
 *  forward_host_iterator_tag, bidirectional_host_iterator_tag
 */
typedef std::random_access_iterator_tag random_access_host_iterator_tag;

/*! \} // end iterator_tag_classes
 */

THRUST_NAMESPACE_END

#include <thrust/iterator/detail/universal_categories.h>

