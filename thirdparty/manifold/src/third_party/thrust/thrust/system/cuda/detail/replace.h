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
#include <thrust/system/cuda/detail/transform.h>
#include <thrust/detail/internal_functional.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

  namespace __replace
  {
    template<class T>
    struct constant_f
    {
      T value;

      THRUST_FUNCTION
      constant_f(T const &x) : value(x) {}

      template<class U>
      THRUST_DEVICE_FUNCTION
      T operator()(U const &)  const
      {
        return value;
      }
    }; // struct constant_f

    template<class Predicate, class NewType, class OutputType>
    struct new_value_if_f
    {
      Predicate pred;
      NewType new_value;

      THRUST_FUNCTION
      new_value_if_f(Predicate pred_, NewType new_value_)
          : pred(pred_), new_value(new_value_) {}

      template<class T>
      OutputType THRUST_DEVICE_FUNCTION
      operator()(T const &x)
      {
        return pred(x) ? new_value : x;
      }

      template<class T, class P>
      OutputType THRUST_DEVICE_FUNCTION
      operator()(T const &x, P const& y)
      {
        return pred(y) ? new_value : x;
      }
    }; // struct new_value_if_f

  } // namespace __replace

template <class Derived,
          class Iterator,
          class T>
void __host__ __device__
replace(execution_policy<Derived> &policy,
        Iterator                   first,
        Iterator                   last,
        T const &                  old_value,
        T const &                  new_value)
{
  using thrust::placeholders::_1;

  cuda_cub::transform_if(policy,
                      first,
                      last,
                      first,
                      __replace::constant_f<T>(new_value),
                      _1 == old_value);
}

template <class Derived,
          class Iterator,
          class Predicate,
          class T>
void __host__ __device__
replace_if(execution_policy<Derived> &policy,
           Iterator                   first,
           Iterator                   last,
           Predicate                  pred,
           T const &                  new_value)
{
  cuda_cub::transform_if(policy,
                      first,
                      last,
                      first,
                      __replace::constant_f<T>(new_value),
                      pred);
}

template <class Derived,
          class Iterator,
          class StencilIt,
          class Predicate,
          class T>
void __host__ __device__
replace_if(execution_policy<Derived> &policy,
           Iterator                   first,
           Iterator                   last,
           StencilIt                  stencil,
           Predicate                  pred,
           T const &                  new_value)
{
  cuda_cub::transform_if(policy,
                      first,
                      last,
                      stencil,
                      first,
                      __replace::constant_f<T>(new_value),
                      pred);
}

template <class Derived,
          class InputIt,
          class OutputIt,
          class Predicate,
          class T>
OutputIt __host__ __device__
replace_copy_if(execution_policy<Derived> &policy,
                InputIt                    first,
                InputIt                    last,
                OutputIt                   result,
                Predicate                  predicate,
                T const &                  new_value)
{
  typedef typename iterator_traits<OutputIt>::value_type output_type;
  typedef __replace::new_value_if_f<Predicate, T, output_type> new_value_if_t;
  return cuda_cub::transform(policy,
                             first,
                             last,
                             result,
                             new_value_if_t(predicate, new_value));
}

template <class Derived,
          class InputIt,
          class StencilIt,
          class OutputIt,
          class Predicate,
          class T>
OutputIt __host__ __device__
replace_copy_if(execution_policy<Derived> &policy,
                InputIt                    first,
                InputIt                    last,
                StencilIt                  stencil,
                OutputIt                   result,
                Predicate                  predicate,
                T const &                  new_value)
{
  typedef typename iterator_traits<OutputIt>::value_type output_type;
  typedef __replace::new_value_if_f<Predicate, T, output_type> new_value_if_t;
  return cuda_cub::transform(policy,
                           first,
                           last,
                           stencil,
                           result,
                           new_value_if_t(predicate, new_value));
}

template <class Derived,
          class InputIt,
          class OutputIt,
          class T>
OutputIt __host__ __device__
replace_copy(execution_policy<Derived> &policy,
             InputIt                    first,
             InputIt                    last,
             OutputIt                   result,
             T const &                  old_value,
             T const &                  new_value)
{
  return cuda_cub::replace_copy_if(policy,
                                   first,
                                   last,
                                   result,
                                   thrust::detail::equal_to_value<T>(old_value),
                                   new_value);
}

}    // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
