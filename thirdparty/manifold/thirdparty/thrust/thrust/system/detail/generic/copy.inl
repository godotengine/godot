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
#include <thrust/system/detail/generic/copy.h>
#include <thrust/functional.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/detail/minimum_system.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
__host__ __device__
  OutputIterator copy(thrust::execution_policy<DerivedPolicy> &exec,
                      InputIterator                            first,
                      InputIterator                            last,
                      OutputIterator                           result)
{
  typedef typename thrust::iterator_value<InputIterator>::type T;
  return thrust::transform(exec, first, last, result, thrust::identity<T>());
} // end copy()


template<typename DerivedPolicy,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
__host__ __device__
  OutputIterator copy_n(thrust::execution_policy<DerivedPolicy> &exec,
                        InputIterator                            first,
                        Size                                     n,
                        OutputIterator                           result)
{
  typedef typename thrust::iterator_value<InputIterator>::type value_type;
  typedef thrust::identity<value_type>                         xfrm_type;

  typedef thrust::detail::unary_transform_functor<xfrm_type> functor_type;

  typedef thrust::tuple<InputIterator,OutputIterator> iterator_tuple;
  typedef thrust::zip_iterator<iterator_tuple>        zip_iter;

  zip_iter zipped = thrust::make_zip_iterator(thrust::make_tuple(first,result));

  return thrust::get<1>(thrust::for_each_n(exec, zipped, n, functor_type(xfrm_type())).get_iterator_tuple());
} // end copy_n()


} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

