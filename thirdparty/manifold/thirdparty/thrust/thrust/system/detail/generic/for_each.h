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
 *  \brief Generic implementation of for_each & for_each_n.
 *         It is an error to call these functions; they have no implementation.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/system/detail/generic/tag.h>
#include <thrust/detail/static_assert.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{


template<typename DerivedPolicy,
         typename InputIterator,
         typename UnaryFunction>
__host__ __device__
InputIterator for_each(thrust::execution_policy<DerivedPolicy> &,
                       InputIterator first,
                       InputIterator ,
                       UnaryFunction )
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<InputIterator, false>::value)
  , "unimplemented for this system"
  );
  return first;
} // end for_each()


template<typename DerivedPolicy,
         typename InputIterator,
         typename Size,
         typename UnaryFunction>
__host__ __device__
InputIterator for_each_n(thrust::execution_policy<DerivedPolicy> &,
                         InputIterator first,
                         Size ,
                         UnaryFunction )
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<InputIterator, false>::value)
  , "unimplemented for this system"
  );
  return first;
} // end for_each_n()


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

