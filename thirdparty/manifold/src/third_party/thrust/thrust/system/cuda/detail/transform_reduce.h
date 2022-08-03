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
#include <iterator>
#include <thrust/system/cuda/detail/reduce.h>
#include <thrust/distance.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

template <class Derived,
          class InputIt,
          class TransformOp,
          class T,
          class ReduceOp>
T __host__ __device__
transform_reduce(execution_policy<Derived> &policy,
                 InputIt                    first,
                 InputIt                    last,
                 TransformOp                transform_op,
                 T                          init,
                 ReduceOp                   reduce_op)
{
  typedef typename iterator_traits<InputIt>::difference_type size_type;
  size_type num_items = static_cast<size_type>(thrust::distance(first, last));
  typedef transform_input_iterator_t<T,
                                     InputIt,
                                     TransformOp>
      transformed_iterator_t;

  return cuda_cub::reduce_n(policy,
                            transformed_iterator_t(first, transform_op),
                            num_items,
                            init,
                            reduce_op);
}

}    // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
