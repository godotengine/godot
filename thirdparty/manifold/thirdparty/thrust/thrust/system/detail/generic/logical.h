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
#include <thrust/system/detail/generic/tag.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/find.h>
#include <thrust/logical.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename ExecutionPolicy, typename InputIterator, typename Predicate>
__host__ __device__
bool all_of(thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::find_if(exec, first, last, thrust::detail::not1(pred)) == last;
}


template<typename ExecutionPolicy, typename InputIterator, typename Predicate>
__host__ __device__
bool any_of(thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  return thrust::find_if(exec, first, last, pred) != last;
}


template<typename ExecutionPolicy, typename InputIterator, typename Predicate>
__host__ __device__
bool none_of(thrust::execution_policy<ExecutionPolicy> &exec, InputIterator first, InputIterator last, Predicate pred)
{
  return !thrust::any_of(exec, first, last, pred);
}


} // end generic
} // end detail
} // end system
THRUST_NAMESPACE_END

