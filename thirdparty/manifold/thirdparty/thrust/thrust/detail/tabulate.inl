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
#include <thrust/tabulate.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/tabulate.h>
#include <thrust/system/detail/adl/tabulate.h>

THRUST_NAMESPACE_BEGIN


__thrust_exec_check_disable__
template<typename DerivedPolicy, typename ForwardIterator, typename UnaryOperation>
__host__ __device__
  void tabulate(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op)
{
  using thrust::system::detail::generic::tabulate;
  return tabulate(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, unary_op);
} // end tabulate()


template<typename ForwardIterator, typename UnaryOperation>
  void tabulate(ForwardIterator first,
                ForwardIterator last,
                UnaryOperation unary_op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<ForwardIterator>::type System;

  System system;

  return thrust::tabulate(select_system(system), first, last, unary_op);
} // end tabulate()


THRUST_NAMESPACE_END

