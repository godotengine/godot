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


/*! \file adjacent_difference.h
 *  \brief Sequential implementation of adjacent_difference.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/sequential/execution_policy.h>

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
OutputIterator adjacent_difference(sequential::execution_policy<DerivedPolicy> &,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result,
                                   BinaryFunction binary_op)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;

  if(first == last)
    return result;

  InputType curr = *first;

  *result = curr;

  while(++first != last)
  {
    InputType next = *first;
    *(++result) = binary_op(next, curr);
    curr = next;
  }

  return ++result;
}


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

