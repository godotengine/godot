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


/*! \file thrust/iterator/iterator_traits.h
 *  \brief Traits and metafunctions for reasoning about the traits of iterators
 */

/*
 * (C) Copyright David Abrahams 2003.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/type_traits/void_t.h>

#include <iterator>

THRUST_NAMESPACE_BEGIN

/*! \p iterator_traits is a type trait class that provides a uniform
 *  interface for querying the properties of iterators at compile-time.
 */
template <typename T>
struct iterator_traits : std::iterator_traits<T> {};

template<typename Iterator> struct iterator_value;

template<typename Iterator> struct iterator_pointer;

template<typename Iterator> struct iterator_reference;

template<typename Iterator> struct iterator_difference;

template<typename Iterator> struct iterator_traversal;

template<typename Iterator> struct iterator_system;

THRUST_NAMESPACE_END

#include <thrust/iterator/detail/iterator_traversal_tags.h>
#include <thrust/iterator/detail/host_system_tag.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/iterator/detail/iterator_traits.inl>

