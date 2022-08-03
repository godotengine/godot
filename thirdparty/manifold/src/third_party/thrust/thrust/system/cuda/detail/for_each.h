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
#include <thrust/system/cuda/config.h>

#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/parallel_for.h>
#include <thrust/detail/function.h>
#include <thrust/distance.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub {

  // for_each functor
  template <class Input, class UnaryOp>
  struct for_each_f
  {
    Input input;
    UnaryOp op;

    THRUST_FUNCTION
    for_each_f(Input input, UnaryOp op)
        : input(input), op(op) {}

    template <class Size>
    THRUST_DEVICE_FUNCTION void operator()(Size idx)
    {
      op(raw_reference_cast(input[idx]));
    }
  };

  //-------------------------
  // Thrust API entry points
  //-------------------------

  // for_each_n
  template <class Derived,
            class Input,
            class Size,
            class UnaryOp>
  Input THRUST_FUNCTION
  for_each_n(execution_policy<Derived> &policy,
             Input                      first,
             Size                       count,
             UnaryOp                    op)
  {
    typedef thrust::detail::wrapped_function<UnaryOp, void> wrapped_t;
    wrapped_t wrapped_op(op);

    cuda_cub::parallel_for(policy,
                           for_each_f<Input, wrapped_t>(first, wrapped_op),
                           count);

    cuda_cub::throw_on_error(
      cuda_cub::synchronize_optional(policy)
    , "for_each: failed to synchronize"
    );

    return first + count;
  }

  // for_each
  template <class Derived,
            class Input,
            class UnaryOp>
  Input THRUST_FUNCTION
  for_each(execution_policy<Derived> &policy,
           Input                      first,
           Input                      last,
           UnaryOp                    op)
  {
    typedef typename iterator_traits<Input>::difference_type size_type;
    size_type count = static_cast<size_type>(thrust::distance(first,last));
    return cuda_cub::for_each_n(policy, first,  count, op);
  }
}    // namespace cuda_cub

THRUST_NAMESPACE_END
#endif
