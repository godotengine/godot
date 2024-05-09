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
#include <thrust/system/detail/generic/adjacent_difference.h>
#include <thrust/adjacent_difference.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/transform.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename InputIterator, typename OutputIterator>
__host__ __device__
OutputIterator adjacent_difference(thrust::execution_policy<DerivedPolicy> &exec,
                                   InputIterator first, InputIterator last,
                                   OutputIterator result)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  thrust::minus<InputType> binary_op;

  return thrust::adjacent_difference(exec, first, last, result, binary_op);
} // end adjacent_difference()


template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
__host__ __device__
OutputIterator adjacent_difference(thrust::execution_policy<DerivedPolicy> &exec,
                                   InputIterator first, InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

  if(first == last)
  {
    // empty range, nothing to do
    return result;
  }
  else
  {
    // an in-place operation is requested, copy the input and call the entry point
    // XXX a special-purpose kernel would be faster here since
    // only block boundaries need to be copied
    thrust::detail::temporary_array<InputType, DerivedPolicy> input_copy(exec, first, last);

    *result = *first;
    thrust::transform(exec, input_copy.begin() + 1, input_copy.end(), input_copy.begin(), result + 1, binary_op);
  }

  return result + (last - first);
}


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

