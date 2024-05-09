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

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

template <typename> class reverse_iterator;

namespace detail
{

template<typename BidirectionalIterator>
  struct reverse_iterator_base
{
  typedef thrust::iterator_adaptor<
    thrust::reverse_iterator<BidirectionalIterator>,
    BidirectionalIterator
  > type;
}; // end reverse_iterator_base

} // end detail

THRUST_NAMESPACE_END

