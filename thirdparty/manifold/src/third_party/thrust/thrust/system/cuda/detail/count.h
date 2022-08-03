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

#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/reduce.h>
#include <thrust/distance.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

template <class Derived,
          class InputIt,
          class UnaryPred>
typename iterator_traits<InputIt>::difference_type __host__ __device__
count_if(execution_policy<Derived> &policy,
         InputIt                    first,
         InputIt                    last,
         UnaryPred                  unary_pred)
{
  typedef typename iterator_traits<InputIt>::difference_type size_type;
  typedef transform_input_iterator_t<size_type,
                                     InputIt,
                                     UnaryPred>
      flag_iterator_t;

  return cuda_cub::reduce_n(policy,
                            flag_iterator_t(first, unary_pred),
                            thrust::distance(first, last),
                            size_type(0),
                            plus<size_type>());
}

template <class Derived,
          class InputIt,
          class Value>
typename iterator_traits<InputIt>::difference_type __host__ __device__
count(execution_policy<Derived> &policy,
      InputIt                    first,
      InputIt                    last,
      Value const &              value)
{
  return cuda_cub::count_if(policy,
                            first,
                            last,
                            thrust::detail::equal_to_value<Value>(value));
}

} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
