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
#include <thrust/system/omp/detail/reduce_by_key.h>
#include <thrust/system/detail/generic/reduce_by_key.h>
#include <thrust/distance.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
    reduce_by_key(execution_policy<DerivedPolicy> &exec,
                  InputIterator1 keys_first,
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output,
                  BinaryPredicate binary_pred,
                  BinaryFunction binary_op)
{
  // omp prefers generic::reduce_by_key to cpp::reduce_by_key
  return thrust::system::detail::generic::reduce_by_key(exec, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
} // end reduce_by_key()


} // end detail
} // end omp
} // end system
THRUST_NAMESPACE_END

