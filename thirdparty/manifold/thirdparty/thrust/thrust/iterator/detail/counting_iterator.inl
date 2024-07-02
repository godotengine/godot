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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/numeric_traits.h>
#include <thrust/detail/type_traits.h>
#include <cstddef>

THRUST_NAMESPACE_BEGIN

// forward declaration of counting_iterator
template <typename Incrementable, typename System, typename Traversal, typename Difference>
  class counting_iterator;

namespace detail
{

template <typename Incrementable, typename System, typename Traversal, typename Difference>
  struct counting_iterator_base
{
  typedef typename thrust::detail::eval_if<
    // use any_system_tag if we are given use_default
    thrust::detail::is_same<System,use_default>::value,
    thrust::detail::identity_<thrust::any_system_tag>,
    thrust::detail::identity_<System>
  >::type system;

  typedef typename thrust::detail::ia_dflt_help<
      Traversal,
      thrust::detail::eval_if<
          thrust::detail::is_numeric<Incrementable>::value,
          thrust::detail::identity_<random_access_traversal_tag>,
          thrust::iterator_traversal<Incrementable>
      >
  >::type traversal;

  // unlike Boost, we explicitly use std::ptrdiff_t as the difference type
  // for floating point counting_iterators
  typedef typename thrust::detail::ia_dflt_help<
    Difference,
    thrust::detail::eval_if<
      thrust::detail::is_numeric<Incrementable>::value,
        thrust::detail::eval_if<
          thrust::detail::is_integral<Incrementable>::value,
          thrust::detail::numeric_difference<Incrementable>,
          thrust::detail::identity_<std::ptrdiff_t>
        >,
      thrust::iterator_difference<Incrementable>
    >
  >::type difference;

  // our implementation departs from Boost's in that counting_iterator::dereference
  // returns a copy of its counter, rather than a reference to it. returning a reference
  // to the internal state of an iterator causes subtle bugs (consider the temporary
  // iterator created in the expression *(iter + i)) and has no compelling use case
  typedef thrust::iterator_adaptor<
    counting_iterator<Incrementable, System, Traversal, Difference>, // self
    Incrementable,                                                  // Base
    Incrementable,                                                  // XXX we may need to pass const here as Boost does
    system,
    traversal,
    Incrementable,
    difference
  > type;
}; // end counting_iterator_base


template<typename Difference, typename Incrementable1, typename Incrementable2>
  struct iterator_distance
{
  __host__ __device__
  static Difference distance(Incrementable1 x, Incrementable2 y)
  {
    return y - x;
  }
};


template<typename Difference, typename Incrementable1, typename Incrementable2>
  struct number_distance
{
  __host__ __device__
  static Difference distance(Incrementable1 x, Incrementable2 y)
  {
      return static_cast<Difference>(numeric_distance(x,y));
  }
};


template<typename Difference, typename Incrementable1, typename Incrementable2, typename Enable = void>
  struct counting_iterator_equal
{
  __host__ __device__
  static bool equal(Incrementable1 x, Incrementable2 y)
  {
    return x == y;
  }
};


// specialization for floating point equality
template<typename Difference, typename Incrementable1, typename Incrementable2>
  struct counting_iterator_equal<
    Difference,
    Incrementable1,
    Incrementable2,
    typename thrust::detail::enable_if<
      thrust::detail::is_floating_point<Incrementable1>::value ||
      thrust::detail::is_floating_point<Incrementable2>::value
    >::type
  >
{
  __host__ __device__
  static bool equal(Incrementable1 x, Incrementable2 y)
  {
    typedef number_distance<Difference,Incrementable1,Incrementable2> d;
    return d::distance(x,y) == 0;
  }
};


} // end detail
THRUST_NAMESPACE_END

