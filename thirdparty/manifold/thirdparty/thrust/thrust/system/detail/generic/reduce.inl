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

#include <thrust/reduce.h>
#include <thrust/system/detail/generic/reduce.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/detail/static_assert.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy, typename InputIterator>
__host__ __device__
  typename thrust::iterator_traits<InputIterator>::value_type
    reduce(thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last)
{
  typedef typename thrust::iterator_value<InputIterator>::type InputType;

  // use InputType(0) as init by default
  return thrust::reduce(exec, first, last, InputType(0));
} // end reduce()


template<typename ExecutionPolicy, typename InputIterator, typename T>
__host__ __device__
  T reduce(thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last, T init)
{
  // use plus<T> by default
  return thrust::reduce(exec, first, last, init, thrust::plus<T>());
} // end reduce()


template<typename ExecutionPolicy,
         typename RandomAccessIterator,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
  OutputType reduce(thrust::execution_policy<ExecutionPolicy> &,
                    RandomAccessIterator,
                    RandomAccessIterator,
                    OutputType,
                    BinaryFunction)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<RandomAccessIterator, false>::value)
  , "unimplemented for this system"
  );
  return OutputType();
} // end reduce()


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

