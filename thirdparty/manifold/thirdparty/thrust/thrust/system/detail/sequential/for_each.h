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


/*! \file for_each.h
 *  \brief Sequential implementations of for_each functions.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/function.h>
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
         typename UnaryFunction>
__host__ __device__
InputIterator for_each(sequential::execution_policy<DerivedPolicy> &,
                       InputIterator first,
                       InputIterator last,
                       UnaryFunction f)
{
  // wrap f
  thrust::detail::wrapped_function<
    UnaryFunction,
    void
  > wrapped_f(f);

  for(; first != last; ++first)
  {
    wrapped_f(*first);
  }

  return first;
} // end for_each()


template<typename DerivedPolicy,
         typename InputIterator,
         typename Size,
         typename UnaryFunction>
__host__ __device__
InputIterator for_each_n(sequential::execution_policy<DerivedPolicy> &,
                         InputIterator first,
                         Size n,
                         UnaryFunction f)
{
  // wrap f
  thrust::detail::wrapped_function<
    UnaryFunction,
    void
  > wrapped_f(f);

  for(Size i = 0; i != n; i++)
  {
    // we can dereference an OutputIterator if f does not
    // try to use the reference for anything besides assignment
    wrapped_f(*first);
    ++first;
  }

  return first;
} // end for_each_n()


} // end namespace sequential
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

