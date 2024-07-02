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
#include <thrust/advance.h>
#include <thrust/system/detail/generic/advance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/detail/type_traits/pointer_traits.h>

THRUST_NAMESPACE_BEGIN

__THRUST_DEFINE_HAS_NESTED_TYPE(has_difference_type, difference_type)

template <typename InputIterator, typename Distance>
__host__ __device__
void advance(InputIterator& i, Distance n)
{
  thrust::system::detail::generic::advance(i, n);
}

template <typename InputIterator>
__host__ __device__
InputIterator next(
  InputIterator i
, typename iterator_traits<InputIterator>::difference_type n = 1
)
{
  thrust::system::detail::generic::advance(i, n);
  return i;
}

template <typename BidirectionalIterator>
__host__ __device__
BidirectionalIterator prev(
  BidirectionalIterator i
, typename iterator_traits<BidirectionalIterator>::difference_type n = 1
)
{
  thrust::system::detail::generic::advance(i, -n);
  return i;
}

template <typename BidirectionalIterator>
__host__ __device__
typename detail::disable_if<
  has_difference_type<iterator_traits<BidirectionalIterator> >::value
, BidirectionalIterator
>::type prev(
  BidirectionalIterator i
, typename detail::pointer_traits<BidirectionalIterator>::difference_type n = 1
)
{
  thrust::system::detail::generic::advance(i, -n);
  return i;
}

THRUST_NAMESPACE_END
