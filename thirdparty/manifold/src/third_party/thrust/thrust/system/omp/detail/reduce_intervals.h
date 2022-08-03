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


/*! \file reduce_intervals.h
 *  \brief OpenMP implementations of reduce_intervals algorithms.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/omp/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void reduce_intervals(execution_policy<DerivedPolicy> &exec,
                      InputIterator input,
                      OutputIterator output,
                      BinaryFunction binary_op,
                      Decomposition decomp);

} // end namespace detail
} // end namespace omp
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/omp/detail/reduce_intervals.inl>

