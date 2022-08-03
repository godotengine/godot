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

THRUST_NAMESPACE_BEGIN
namespace detail
{


template<typename Category, typename System, typename Traversal>
  struct iterator_category_with_system_and_traversal
    : Category
{
}; // end iterator_category_with_system_and_traversal


// specialize iterator_category_to_system for iterator_category_with_system_and_traversal
template<typename Category> struct iterator_category_to_system;

template<typename Category, typename System, typename Traversal>
  struct iterator_category_to_system<iterator_category_with_system_and_traversal<Category,System,Traversal> >
{
  typedef System type;
}; // end iterator_category_to_system


// specialize iterator_category_to_traversal for iterator_category_with_system_and_traversal
template<typename Category> struct iterator_category_to_traversal;

template<typename Category, typename System, typename Traversal>
  struct iterator_category_to_traversal<iterator_category_with_system_and_traversal<Category,System,Traversal> >
{
  typedef Traversal type;
}; // end iterator_category_to_traversal



} // end detail
THRUST_NAMESPACE_END

