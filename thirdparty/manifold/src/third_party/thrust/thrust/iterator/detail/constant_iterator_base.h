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
#include <thrust/iterator/iterator_adaptor.h>

THRUST_NAMESPACE_BEGIN

// forward declaration of constant_iterator
template<typename,typename,typename> class constant_iterator;

namespace detail
{

template<typename Value,
         typename Incrementable,
         typename System>
  struct constant_iterator_base
{
  typedef Value              value_type;

  // the reference type is the same as the value_type.
  // we wish to avoid returning a reference to the internal state
  // of the constant_iterator, which is prone to subtle bugs.
  // consider the temporary iterator created in the expression
  // *(iter + i)
  typedef value_type         reference;

  // the incrementable type is int unless otherwise specified
  typedef typename thrust::detail::ia_dflt_help<
    Incrementable,
    thrust::detail::identity_<thrust::detail::intmax_t>
  >::type incrementable;

  typedef typename thrust::counting_iterator<
    incrementable,
    System,
    thrust::random_access_traversal_tag
  > base_iterator;

  typedef typename thrust::iterator_adaptor<
    constant_iterator<Value, Incrementable, System>,
    base_iterator,
    value_type, // XXX we may need to pass const value_type here as boost counting_iterator does
    typename thrust::iterator_system<base_iterator>::type,
    typename thrust::iterator_traversal<base_iterator>::type,
    reference
  > type;
}; // end constant_iterator_base

} // end detail
  
THRUST_NAMESPACE_END

