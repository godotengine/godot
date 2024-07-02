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
#include <thrust/system/detail/generic/tabulate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/transform.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename UnaryOperation>
__host__ __device__
  void tabulate(thrust::execution_policy<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op)
{
  typedef typename iterator_difference<ForwardIterator>::type difference_type;

  // by default, counting_iterator uses a 64b difference_type on 32b platforms to avoid overflowing its counter.
  // this causes problems when a zip_iterator is created in transform's implementation -- ForwardIterator is
  // incremented by a 64b difference_type and some compilers warn
  // to avoid this, specify the counting_iterator's difference_type to be the same as ForwardIterator's.
  thrust::counting_iterator<difference_type, thrust::use_default, thrust::use_default, difference_type> iter(0);

  thrust::transform(exec, iter, iter + thrust::distance(first, last), first, unary_op);
} // end tabulate()


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END


