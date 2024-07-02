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
#include <thrust/system/detail/generic/inner_product.h>
#include <thrust/functional.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/transform_reduce.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputType>
__host__ __device__
OutputType inner_product(thrust::execution_policy<DerivedPolicy> &exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init)
{
  thrust::plus<OutputType>       binary_op1;
  thrust::multiplies<OutputType> binary_op2;
  return thrust::inner_product(exec, first1, last1, first2, init, binary_op1, binary_op2);
} // end inner_product()


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
__host__ __device__
OutputType inner_product(thrust::execution_policy<DerivedPolicy> &exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init,
                         BinaryFunction1 binary_op1,
                         BinaryFunction2 binary_op2)
{
  typedef thrust::zip_iterator<thrust::tuple<InputIterator1,InputIterator2> > ZipIter;

  ZipIter first = thrust::make_zip_iterator(thrust::make_tuple(first1,first2));

  // only the first iterator in the tuple is relevant for the purposes of last
  ZipIter last  = thrust::make_zip_iterator(thrust::make_tuple(last1, first2));

  return thrust::transform_reduce(exec, first, last, thrust::detail::zipped_binary_op<OutputType,BinaryFunction2>(binary_op2), init, binary_op1);
} // end inner_product()


} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

