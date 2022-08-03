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

#include <thrust/detail/seq.h>
#include <thrust/system/cuda/detail/par.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

template <int PAR>
struct has_par : thrust::detail::true_type {};

template <>
struct has_par<0> : thrust::detail::false_type {};

template<class Policy>
struct cvt_to_seq_impl
{
  typedef thrust::detail::seq_t seq_t;

  static seq_t __host__ __device__
  doit(Policy&)
  {
    return seq_t();
  }
};    // cvt_to_seq_impl

#if 0
template <class Allocator>
struct cvt_to_seq_impl<
    thrust::detail::execute_with_allocator<Allocator,
                                           execute_on_stream_base> >
{
  typedef thrust::detail::execute_with_allocator<Allocator,
                                                 execute_on_stream_base>
      Policy;
  typedef thrust::detail::execute_with_allocator<
      Allocator,
      thrust::system::detail::sequential::execution_policy>
      seq_t;


  static seq_t __host__ __device__
  doit(Policy& policy)
  {
    return seq_t(policy.m_alloc);
  }
};    // specialization of struct cvt_to_seq_impl
#endif

template <class Policy>
typename cvt_to_seq_impl<Policy>::seq_t __host__ __device__
cvt_to_seq(Policy& policy)
{
  return cvt_to_seq_impl<Policy>::doit(policy);
}

#if __THRUST_HAS_CUDART__
#define THRUST_CUDART_DISPATCH par
#else
#define THRUST_CUDART_DISPATCH seq
#endif

} // namespace cuda_
THRUST_NAMESPACE_END
