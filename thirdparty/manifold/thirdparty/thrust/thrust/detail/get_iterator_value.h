#pragma once
/*
 *  Copyright 2008-2016 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/execution_policy.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/system/detail/generic/memory.h> // for get_value()

THRUST_NAMESPACE_BEGIN

namespace detail {

// get_iterator_value specialization on iterators
// --------------------------------------------------
// it is okay to dereference iterator in the usual way
template<typename DerivedPolicy, typename Iterator>
__host__ __device__
typename thrust::iterator_traits<Iterator>::value_type
get_iterator_value(thrust::execution_policy<DerivedPolicy> &, Iterator it)
{
  return *it;
} // get_iterator_value(exec,Iterator);

// get_iterator_value specialization on pointer
// ----------------------------------------------
// we can't just dereference a pointer in the usual way, because
// it may point to a location in the device memory. 
// we use get_value(exec,pointer*) function
// to perform a dereferencing consistent with the execution policy
template<typename DerivedPolicy, typename Pointer>
__host__ __device__
typename thrust::detail::pointer_traits<Pointer*>::element_type 
get_iterator_value(thrust::execution_policy<DerivedPolicy> &exec, Pointer* ptr)
{
  return get_value(derived_cast(exec),ptr);
} // get_iterator_value(exec,Pointer*)

} // namespace detail

THRUST_NAMESPACE_END
