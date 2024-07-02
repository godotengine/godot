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
#include <thrust/iterator/detail/host_system_tag.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/detail/iterator_traversal_tags.h>
#include <thrust/iterator/detail/is_iterator_category.h>
#include <thrust/iterator/detail/iterator_category_with_system_and_traversal.h>
#include <thrust/iterator/detail/iterator_category_to_traversal.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{


// adapted from http://www.boost.org/doc/libs/1_37_0/libs/iterator/doc/iterator_facade.html#iterator-category
//
// in our implementation, R need not be a reference type to result in a category
// derived from forward_XXX_iterator_tag
//
// iterator-category(T,V,R) :=
//   if(T is convertible to input_host_iterator_tag
//      || T is convertible to output_host_iterator_tag
//      || T is convertible to input_device_iterator_tag
//      || T is convertible to output_device_iterator_tag
//   )
//     return T
//
//   else if (T is not convertible to incrementable_traversal_tag)
//     the program is ill-formed
//
//   else return a type X satisfying the following two constraints:
//
//     1. X is convertible to X1, and not to any more-derived
//        type, where X1 is defined by:
//
//        if (T is convertible to forward_traversal_tag)
//        {
//          if (T is convertible to random_access_traversal_tag)
//            X1 = random_access_host_iterator_tag
//          else if (T is convertible to bidirectional_traversal_tag)
//            X1 = bidirectional_host_iterator_tag
//          else
//            X1 = forward_host_iterator_tag
//        }
//        else
//        {
//          if (T is convertible to single_pass_traversal_tag
//              && R is convertible to V)
//            X1 = input_host_iterator_tag
//          else
//            X1 = T
//        }
//
//     2. category-to-traversal(X) is convertible to the most
//        derived traversal tag type to which X is also convertible,
//        and not to any more-derived traversal tag type.


template<typename System, typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category;


// Thrust's implementation of iterator_facade_default_category is slightly
// different from Boost's equivalent.
// Thrust does not check is_convertible<Reference, ValueParam> because Reference
// may not be a complete type at this point, and implementations of is_convertible
// typically require that both types be complete.
// Instead, it simply assumes that if is_convertible<Traversal, single_pass_traversal_tag>,
// then the category is input_iterator_tag


// this is the function for standard system iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_std :
    thrust::detail::eval_if<
      thrust::detail::is_convertible<Traversal, thrust::forward_traversal_tag>::value,
      thrust::detail::eval_if<
        thrust::detail::is_convertible<Traversal, thrust::random_access_traversal_tag>::value,
        thrust::detail::identity_<std::random_access_iterator_tag>,
        thrust::detail::eval_if<
          thrust::detail::is_convertible<Traversal, thrust::bidirectional_traversal_tag>::value,
          thrust::detail::identity_<std::bidirectional_iterator_tag>,
          thrust::detail::identity_<std::forward_iterator_tag>
        >
      >,
      thrust::detail::eval_if< // XXX note we differ from Boost here
        thrust::detail::is_convertible<Traversal, thrust::single_pass_traversal_tag>::value,
        thrust::detail::identity_<std::input_iterator_tag>,
        thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_std


// this is the function for host system iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_host :
    thrust::detail::eval_if<
      thrust::detail::is_convertible<Traversal, thrust::forward_traversal_tag>::value,
      thrust::detail::eval_if<
        thrust::detail::is_convertible<Traversal, thrust::random_access_traversal_tag>::value,
        thrust::detail::identity_<thrust::random_access_host_iterator_tag>,
        thrust::detail::eval_if<
          thrust::detail::is_convertible<Traversal, thrust::bidirectional_traversal_tag>::value,
          thrust::detail::identity_<thrust::bidirectional_host_iterator_tag>,
          thrust::detail::identity_<thrust::forward_host_iterator_tag>
        >
      >,
      thrust::detail::eval_if< // XXX note we differ from Boost here
        thrust::detail::is_convertible<Traversal, thrust::single_pass_traversal_tag>::value,
        thrust::detail::identity_<thrust::input_host_iterator_tag>,
        thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_host


// this is the function for device system iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_device :
    thrust::detail::eval_if<
      thrust::detail::is_convertible<Traversal, thrust::forward_traversal_tag>::value,
      thrust::detail::eval_if<
        thrust::detail::is_convertible<Traversal, thrust::random_access_traversal_tag>::value,
        thrust::detail::identity_<thrust::random_access_device_iterator_tag>,
        thrust::detail::eval_if<
          thrust::detail::is_convertible<Traversal, thrust::bidirectional_traversal_tag>::value,
          thrust::detail::identity_<thrust::bidirectional_device_iterator_tag>,
          thrust::detail::identity_<thrust::forward_device_iterator_tag>
        >
      >,
      thrust::detail::eval_if<
        thrust::detail::is_convertible<Traversal, thrust::single_pass_traversal_tag>::value, // XXX note we differ from Boost here
        thrust::detail::identity_<thrust::input_device_iterator_tag>,
        thrust::detail::identity_<Traversal>
      >
    >
{
}; // end iterator_facade_default_category_device


// this is the function for any system iterators
template<typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category_any
{
  typedef thrust::detail::iterator_category_with_system_and_traversal<
    typename iterator_facade_default_category_std<Traversal, ValueParam, Reference>::type,
    thrust::any_system_tag,
    Traversal
  > type;
}; // end iterator_facade_default_category_any


template<typename System, typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_default_category
      // check for any system
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<System, thrust::any_system_tag>::value,
        iterator_facade_default_category_any<Traversal, ValueParam, Reference>,

        // check for host system
        thrust::detail::eval_if<
          thrust::detail::is_convertible<System, thrust::host_system_tag>::value,
          iterator_facade_default_category_host<Traversal, ValueParam, Reference>,

          // check for device system
          thrust::detail::eval_if<
            thrust::detail::is_convertible<System, thrust::device_system_tag>::value,
            iterator_facade_default_category_device<Traversal, ValueParam, Reference>,

            // if we don't recognize the system, get a standard iterator category
            // and combine it with System & Traversal
            thrust::detail::identity_<
              thrust::detail::iterator_category_with_system_and_traversal<
                typename iterator_facade_default_category_std<Traversal, ValueParam, Reference>::type,
                System,
                Traversal
              >
            >
          >
        >
      >
{};


template<typename System, typename Traversal, typename ValueParam, typename Reference>
  struct iterator_facade_category_impl
{
  typedef typename iterator_facade_default_category<
    System,Traversal,ValueParam,Reference
  >::type category;

  // we must be able to deduce both Traversal & System from category
  // otherwise, munge them all together
  typedef typename thrust::detail::eval_if<
    thrust::detail::and_<
      thrust::detail::is_same<
        Traversal,
        typename thrust::detail::iterator_category_to_traversal<category>::type
      >,
      thrust::detail::is_same<
        System,
        typename thrust::detail::iterator_category_to_system<category>::type
      >
    >::value,
    thrust::detail::identity_<category>,
    thrust::detail::identity_<thrust::detail::iterator_category_with_system_and_traversal<category,System,Traversal> >
  >::type type;
}; // end iterator_facade_category_impl


template<typename CategoryOrSystem,
         typename CategoryOrTraversal,
         typename ValueParam,
         typename Reference>
  struct iterator_facade_category
{
  typedef typename
  thrust::detail::eval_if<
    thrust::detail::is_iterator_category<CategoryOrTraversal>::value,
    thrust::detail::identity_<CategoryOrTraversal>, // categories are fine as-is
    iterator_facade_category_impl<CategoryOrSystem, CategoryOrTraversal, ValueParam, Reference>
  >::type type;
}; // end iterator_facade_category


} // end detail
THRUST_NAMESPACE_END

