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
#include <thrust/iterator/detail/host_system_tag.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

// forward declaration
template <typename> struct is_iterator_system;

template <typename> struct device_iterator_category_to_backend_system;

// XXX this should work entirely differently
// we should just specialize this metafunction for iterator_category_with_system_and_traversal
template<typename Category>
  struct iterator_category_to_system
    // convertible to host iterator?
    : eval_if<
        or_<
          is_convertible<Category, thrust::input_host_iterator_tag>,
          is_convertible<Category, thrust::output_host_iterator_tag>
        >::value,

        detail::identity_<thrust::host_system_tag>,
        
        // convertible to device iterator?
        eval_if<
          or_<
            is_convertible<Category, thrust::input_device_iterator_tag>,
            is_convertible<Category, thrust::output_device_iterator_tag>
          >::value,

          detail::identity_<thrust::device_system_tag>,

          // unknown system
          detail::identity_<void>
        > // if device
      > // if host
{
}; // end iterator_category_to_system


template<typename CategoryOrTraversal>
  struct iterator_category_or_traversal_to_system
    : eval_if<
        is_iterator_system<CategoryOrTraversal>::value,
        detail::identity_<CategoryOrTraversal>,
        iterator_category_to_system<CategoryOrTraversal>
      >
{
}; // end iterator_category_or_traversal_to_system

} // end detail
THRUST_NAMESPACE_END

