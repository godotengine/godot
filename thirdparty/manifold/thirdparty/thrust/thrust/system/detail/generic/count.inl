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
#include <thrust/system/detail/generic/count.h>
#include <thrust/transform_reduce.h>
#include <thrust/detail/internal_functional.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template <typename InputType, typename Predicate, typename CountType>
struct count_if_transform
{
  __host__ __device__
  count_if_transform(Predicate _pred) : pred(_pred){}

  __thrust_exec_check_disable__
  __host__ __device__
  CountType operator()(const InputType& val)
  {
    if(pred(val))
      return 1;
    else
      return 0;
  } // end operator()

  Predicate pred;
}; // end count_if_transform


template <typename DerivedPolicy, typename InputIterator, typename EqualityComparable>
__host__ __device__
typename thrust::iterator_traits<InputIterator>::difference_type
count(thrust::execution_policy<DerivedPolicy> &exec, InputIterator first, InputIterator last, const EqualityComparable& value)
{
  using thrust::placeholders::_1;

  return thrust::count_if(exec, first, last, _1 == value);
} // end count()


template <typename DerivedPolicy, typename InputIterator, typename Predicate>
__host__ __device__
typename thrust::iterator_traits<InputIterator>::difference_type
count_if(thrust::execution_policy<DerivedPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  typedef typename thrust::iterator_traits<InputIterator>::value_type InputType;
  typedef typename thrust::iterator_traits<InputIterator>::difference_type CountType;

  thrust::system::detail::generic::count_if_transform<InputType, Predicate, CountType> unary_op(pred);
  thrust::plus<CountType> binary_op;
  return thrust::transform_reduce(exec, first, last, unary_op, CountType(0), binary_op);
} // end count_if()


} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

