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
#include <thrust/iterator/detail/iterator_traversal_tags.h>
#include <thrust/iterator/detail/iterator_category_to_system.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

// forward declarations
template <typename> struct is_iterator_system;
template <typename> struct is_iterator_traversal;

template <typename Category>
  struct host_system_category_to_traversal
    : eval_if<
        is_convertible<Category, random_access_host_iterator_tag>::value,
        detail::identity_<random_access_traversal_tag>,
        eval_if<
          is_convertible<Category, bidirectional_host_iterator_tag>::value,
          detail::identity_<bidirectional_traversal_tag>,
          eval_if<
            is_convertible<Category, forward_host_iterator_tag>::value,
            detail::identity_<forward_traversal_tag>,
            eval_if<
              is_convertible<Category, input_host_iterator_tag>::value,
              detail::identity_<single_pass_traversal_tag>,
              eval_if<
                is_convertible<Category, output_host_iterator_tag>::value,
                detail::identity_<incrementable_traversal_tag>,
                detail::identity_<void>
              >
            >
          >
        >
      >
{
}; // end host_system_category_to_traversal



template <typename Category>
  struct device_system_category_to_traversal
    : eval_if<
        is_convertible<Category, random_access_device_iterator_tag>::value,
        detail::identity_<random_access_traversal_tag>,
        eval_if<
          is_convertible<Category, bidirectional_device_iterator_tag>::value,
          detail::identity_<bidirectional_traversal_tag>,
          eval_if<
            is_convertible<Category, forward_device_iterator_tag>::value,
            detail::identity_<forward_traversal_tag>,
            eval_if<
              is_convertible<Category, input_device_iterator_tag>::value,
              detail::identity_<single_pass_traversal_tag>,
              eval_if<
                is_convertible<Category, output_device_iterator_tag>::value,
                detail::identity_<incrementable_traversal_tag>,
                detail::identity_<void>
              >
            >
          >
        >
      >
{
}; // end device_system_category_to_traversal


template<typename Category>
  struct category_to_traversal
      // check for host system
    : eval_if<
        or_<
          is_convertible<Category, thrust::input_host_iterator_tag>,
          is_convertible<Category, thrust::output_host_iterator_tag>
        >::value,

        host_system_category_to_traversal<Category>,

        // check for device system
        eval_if<
          or_<
            is_convertible<Category, thrust::input_device_iterator_tag>,
            is_convertible<Category, thrust::output_device_iterator_tag>
          >::value,

          device_system_category_to_traversal<Category>,

          // unknown category
          detail::identity_<void>
        >
      >
{};


template <typename CategoryOrTraversal>
  struct iterator_category_to_traversal
    : eval_if<
        is_iterator_traversal<CategoryOrTraversal>::value,
        detail::identity_<CategoryOrTraversal>,
        category_to_traversal<CategoryOrTraversal>
      >
{
}; // end iterator_category_to_traversal


} // end detail

THRUST_NAMESPACE_END

