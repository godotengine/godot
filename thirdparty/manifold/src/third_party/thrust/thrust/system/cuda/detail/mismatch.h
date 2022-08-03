/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
#include <thrust/system/cuda/config.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/distance.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

template <class Derived,
          class InputIt1,
          class InputIt2,
          class BinaryPred>
pair<InputIt1, InputIt2> __host__ __device__
mismatch(execution_policy<Derived>& policy,
         InputIt1                   first1,
         InputIt1                   last1,
         InputIt2                   first2,
         BinaryPred                 binary_pred);

template <class Derived,
          class InputIt1,
          class InputIt2>
pair<InputIt1, InputIt2> __host__ __device__
mismatch(execution_policy<Derived>& policy,
         InputIt1                   first1,
         InputIt1                   last1,
         InputIt2                   first2);
} // namespace cuda_
THRUST_NAMESPACE_END

#include <thrust/system/cuda/detail/find.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

template <class Derived,
          class InputIt1,
          class InputIt2,
          class BinaryPred>
pair<InputIt1, InputIt2> __host__ __device__
mismatch(execution_policy<Derived>& policy,
         InputIt1                   first1,
         InputIt1                   last1,
         InputIt2                   first2,
         BinaryPred                 binary_pred)
{
  typedef transform_pair_of_input_iterators_t<bool,
                                              InputIt1,
                                              InputIt2,
                                              BinaryPred>
      transform_t;

  transform_t transform_first = transform_t(first1, first2, binary_pred);

  transform_t result = cuda_cub::find_if_not(policy,
                                          transform_first,
                                          transform_first + thrust::distance(first1, last1),
                                          identity());

  return thrust::make_pair(first1 + thrust::distance(transform_first,result),
                           first2 + thrust::distance(transform_first,result));
}

template <class Derived,
          class InputIt1,
          class InputIt2>
pair<InputIt1, InputIt2> __host__ __device__
mismatch(execution_policy<Derived>& policy,
         InputIt1                   first1,
         InputIt1                   last1,
         InputIt2                   first2)
{
  typedef typename thrust::iterator_value<InputIt1>::type InputType1;
  return cuda_cub::mismatch(policy,
                         first1,
                         last1,
                         first2,
                         equal_to<InputType1>());
}



} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
