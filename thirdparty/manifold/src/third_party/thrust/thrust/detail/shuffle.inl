/*
 *  Copyright 2008-2020 NVIDIA Corporation
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
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/iterator/iterator_traits.h>
#include <thrust/shuffle.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/shuffle.h>

THRUST_NAMESPACE_BEGIN

__thrust_exec_check_disable__
template <typename DerivedPolicy, typename RandomIterator, typename URBG>
__host__ __device__ void shuffle(
    const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
    RandomIterator first, RandomIterator last, URBG&& g) {
  using thrust::system::detail::generic::shuffle;
  return shuffle(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      first, last, g);
}

template <typename RandomIterator, typename URBG>
__host__ __device__ void shuffle(RandomIterator first, RandomIterator last,
                                 URBG&& g) {
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomIterator>::type System;
  System system;

  return thrust::shuffle(select_system(system), first, last, g);
}

__thrust_exec_check_disable__
template <typename DerivedPolicy, typename RandomIterator,
          typename OutputIterator, typename URBG>
__host__ __device__ void shuffle_copy(
    const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
    RandomIterator first, RandomIterator last, OutputIterator result,
    URBG&& g) {
  using thrust::system::detail::generic::shuffle_copy;
  return shuffle_copy(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      first, last, result, g);
}

template <typename RandomIterator, typename OutputIterator, typename URBG>
__host__ __device__ void shuffle_copy(RandomIterator first, RandomIterator last,
                                      OutputIterator result, URBG&& g) {
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<RandomIterator>::type System1;
  typedef typename thrust::iterator_system<OutputIterator>::type System2;

  System1 system1;
  System2 system2;

  return thrust::shuffle_copy(select_system(system1, system2), first, last,
                              result, g);
}

THRUST_NAMESPACE_END

#endif
