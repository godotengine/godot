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
#include <thrust/system/detail/generic/equal.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/mismatch.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2>
__host__ __device__
bool equal(thrust::execution_policy<DerivedPolicy> &exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  typedef typename thrust::iterator_traits<InputIterator1>::value_type InputType1;

  return thrust::equal(exec, first1, last1, first2, thrust::detail::equal_to<InputType1>());
}


// the == below could be a __host__ function in the case of std::vector::iterator::operator==
// we make this exception for equal and use __thrust_exec_check_disable__ because it is used in vector's implementation
__thrust_exec_check_disable__
template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
__host__ __device__
bool equal(thrust::execution_policy<DerivedPolicy> &exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, BinaryPredicate binary_pred)
{
  return thrust::mismatch(exec, first1, last1, first2, binary_pred).first == last1;
}


} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

