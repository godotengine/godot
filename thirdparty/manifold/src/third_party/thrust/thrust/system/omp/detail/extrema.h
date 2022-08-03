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
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/detail/generic/extrema.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first, 
                            ForwardIterator last,
                            BinaryPredicate comp)
{
  // omp prefers generic::max_element to cpp::max_element
  return thrust::system::detail::generic::max_element(exec, first, last, comp);
} // end max_element()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(execution_policy<DerivedPolicy> &exec,
                            ForwardIterator first, 
                            ForwardIterator last,
                            BinaryPredicate comp)
{
  // omp prefers generic::min_element to cpp::min_element
  return thrust::system::detail::generic::min_element(exec, first, last, comp);
} // end min_element()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
thrust::pair<ForwardIterator,ForwardIterator> minmax_element(execution_policy<DerivedPolicy> &exec,
                                                             ForwardIterator first, 
                                                             ForwardIterator last,
                                                             BinaryPredicate comp)
{
  // omp prefers generic::minmax_element to cpp::minmax_element
  return thrust::system::detail::generic::minmax_element(exec, first, last, comp);
} // end minmax_element()

} // end detail
} // end omp
} // end system
THRUST_NAMESPACE_END


