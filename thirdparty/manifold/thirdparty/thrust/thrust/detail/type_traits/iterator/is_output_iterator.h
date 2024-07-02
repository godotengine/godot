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
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/is_metafunction_defined.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/any_assign.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{


template<typename T>
  struct is_void_like
    : thrust::detail::or_<
        thrust::detail::is_void<T>,
        thrust::detail::is_same<T,thrust::detail::any_assign>
      >
{}; // end is_void_like


template<typename T>
  struct lazy_is_void_like
    : is_void_like<typename T::type>
{}; // end lazy_is_void_like


// XXX this meta function should first check that T is actually an iterator
//
//     if thrust::iterator_value<T> is defined and thrust::iterator_value<T>::type == void
//       return false
//     else
//       return true
template<typename T>
  struct is_output_iterator
    : eval_if<
        is_metafunction_defined<thrust::iterator_value<T> >::value,
        lazy_is_void_like<thrust::iterator_value<T> >,
        thrust::detail::true_type
      >::type
{
}; // end is_output_iterator

} // end detail

THRUST_NAMESPACE_END

