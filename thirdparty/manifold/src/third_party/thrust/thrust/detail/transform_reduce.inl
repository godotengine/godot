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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/transform_reduce.h>
#include <thrust/system/detail/adl/transform_reduce.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename UnaryFunction,
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
  OutputType transform_reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  using thrust::system::detail::generic::transform_reduce;
  return transform_reduce(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, unary_op, init, binary_op);
} // end transform_reduce()


template<typename InputIterator,
         typename UnaryFunction,
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::transform_reduce(select_system(system), first, last, unary_op, init, binary_op);
} // end transform_reduce()


THRUST_NAMESPACE_END

