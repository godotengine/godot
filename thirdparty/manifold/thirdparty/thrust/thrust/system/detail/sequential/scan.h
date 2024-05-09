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


/*! \file scan.h
 *  \brief Sequential implementations of scan functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/sequential/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/detail/function.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryFunction>
__host__ __device__
  OutputIterator inclusive_scan(sequential::execution_policy<DerivedPolicy> &,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                BinaryFunction binary_op)
{
  using namespace thrust::detail;

  // Use the input iterator's value type per https://wg21.link/P0571
  using ValueType = typename thrust::iterator_value<InputIterator>::type;

  // wrap binary_op
  thrust::detail::wrapped_function<
    BinaryFunction,
    ValueType
  > wrapped_binary_op(binary_op);

  if(first != last)
  {
    ValueType sum = *first;

    *result = *first;

    for(++first, ++result; first != last; ++first, ++result)
      *result = sum = wrapped_binary_op(sum,*first);
  }

  return result;
}


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename InitialValueType,
         typename BinaryFunction>
__host__ __device__
  OutputIterator exclusive_scan(sequential::execution_policy<DerivedPolicy> &,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                InitialValueType init,
                                BinaryFunction binary_op)
{
  using namespace thrust::detail;

  // Use the initial value type per https://wg21.link/P0571
  using ValueType = InitialValueType;

  if(first != last)
  {
    ValueType tmp = *first;  // temporary value allows in-situ scan
    ValueType sum = init;

    *result = sum;
    sum = binary_op(sum, tmp);

    for(++first, ++result; first != last; ++first, ++result)
    {
      tmp = *first;
      *result = sum;
      sum = binary_op(sum, tmp);
    }
  }

  return result;
} 


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

