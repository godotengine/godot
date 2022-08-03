/******************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION.  All rights reserved.
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
#include <thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/util.h>

#include <thrust/detail/allocator_aware_execution_policy.h>

#if THRUST_CPP_DIALECT >= 2011
#  include <thrust/detail/dependencies_aware_execution_policy.h>
#endif


THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

template <class Derived>
struct execute_on_stream_base : execution_policy<Derived>
{
private:
  cudaStream_t stream;

public:
  __host__ __device__
  execute_on_stream_base(cudaStream_t stream_ = default_stream())
      : stream(stream_){}

  THRUST_RUNTIME_FUNCTION
  Derived
  on(cudaStream_t const &s) const
  {
    Derived result = derived_cast(*this);
    result.stream  = s;
    return result;
  }

private:
  friend __host__ __device__
  cudaStream_t
  get_stream(const execute_on_stream_base &exec)
  {
    return exec.stream;
  }
};

template <class Derived>
struct execute_on_stream_nosync_base : execution_policy<Derived>
{
private:
  cudaStream_t stream;

public:
  __host__ __device__
  execute_on_stream_nosync_base(cudaStream_t stream_ = default_stream())
      : stream(stream_){}

  THRUST_RUNTIME_FUNCTION
  Derived
  on(cudaStream_t const &s) const
  {
    Derived result = derived_cast(*this);
    result.stream  = s;
    return result;
  }

private:
  friend __host__ __device__
  cudaStream_t
  get_stream(const execute_on_stream_nosync_base &exec)
  {
    return exec.stream;
  }

  friend __host__ __device__
  bool
  must_perform_optional_stream_synchronization(const execute_on_stream_nosync_base &)
  {
    return false;
  }
};

struct execute_on_stream : execute_on_stream_base<execute_on_stream>
{
  typedef execute_on_stream_base<execute_on_stream> base_t;

  __host__ __device__
  execute_on_stream() : base_t(){};
  __host__ __device__
  execute_on_stream(cudaStream_t stream) 
  : base_t(stream){};
};

struct execute_on_stream_nosync : execute_on_stream_nosync_base<execute_on_stream_nosync>
{
  typedef execute_on_stream_nosync_base<execute_on_stream_nosync> base_t;

  __host__ __device__
  execute_on_stream_nosync() : base_t(){};
  __host__ __device__
  execute_on_stream_nosync(cudaStream_t stream) 
  : base_t(stream){};
};


struct par_t : execution_policy<par_t>,
  thrust::detail::allocator_aware_execution_policy<
    execute_on_stream_base>
#if THRUST_CPP_DIALECT >= 2011
, thrust::detail::dependencies_aware_execution_policy<
    execute_on_stream_base>
#endif
{
  typedef execution_policy<par_t> base_t;

  __host__ __device__
  constexpr par_t() : base_t() {}

  typedef execute_on_stream stream_attachment_type;

  THRUST_RUNTIME_FUNCTION
  stream_attachment_type
  on(cudaStream_t const &stream) const
  {
    return execute_on_stream(stream);
  }
};

struct par_nosync_t : execution_policy<par_nosync_t>,
  thrust::detail::allocator_aware_execution_policy<
    execute_on_stream_nosync_base>
#if THRUST_CPP_DIALECT >= 2011
, thrust::detail::dependencies_aware_execution_policy<
    execute_on_stream_nosync_base>
#endif
{
  typedef execution_policy<par_nosync_t> base_t;

  __host__ __device__
  constexpr par_nosync_t() : base_t() {}

  typedef execute_on_stream_nosync stream_attachment_type;

  THRUST_RUNTIME_FUNCTION
  stream_attachment_type
  on(cudaStream_t const &stream) const
  {
    return execute_on_stream_nosync(stream);
  }

private:
  //this function is defined to allow non-blocking calls on the default_stream() with thrust::cuda::par_nosync
  //without explicitly using thrust::cuda::par_nosync.on(default_stream())
  friend __host__ __device__
  bool
  must_perform_optional_stream_synchronization(const par_nosync_t &)
  {
    return false;
  }
};

THRUST_INLINE_CONSTANT par_t par;

/*! \p thrust::cuda::par_nosync is a parallel execution policy targeting Thrust's CUDA device backend.
 *  Similar to \p thrust::cuda::par it allows execution of Thrust algorithms in a specific CUDA stream.
 *
 *  \p thrust::cuda::par_nosync indicates that an algorithm is free to avoid any synchronization of the 
 *  associated stream that is not strictly required for correctness. Additionally, algorithms may return
 *  before the corresponding kernels are completed, similar to asynchronous kernel launches via <<< >>> syntax.
 *  The user must take care to perform explicit synchronization if necessary.
 *  
 *  The following code snippet demonstrates how to use \p thrust::cuda::par_nosync :
 *
 *  \code
 *    #include <thrust/device_vector.h>
 *    #include <thrust/for_each.h>
 *    #include <thrust/execution_policy.h>
 *
 *    struct IncFunctor{
 *        __host__ __device__
 *        void operator()(std::size_t& x){ x = x + 1; };
 *    };
 *
 *    int main(){
 *        std::size_t N = 1000000;
 *        thrust::device_vector<std::size_t> d_vec(N);
 *
 *        cudaStream_t stream;
 *        cudaStreamCreate(&stream);
 *        auto nosync_policy = thrust::cuda::par_nosync.on(stream);
 *
 *        thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});
 *        thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});
 *        thrust::for_each(nosync_policy, d_vec.begin(), d_vec.end(), IncFunctor{});
 *
 *        //for_each may return before completion. Could do other cpu work in the meantime
 *        // ...
 *
 *        //Wait for the completion of all for_each kernels
 *        cudaStreamSynchronize(stream);
 *
 *        std::size_t x = thrust::reduce(nosync_policy, d_vec.begin(), d_vec.end());
 *        //Currently, this synchronization is not necessary. reduce will still perform
 *        //implicit synchronization to transfer the reduced value to the host to return it.
 *        cudaStreamSynchronize(stream);
 *        cudaStreamDestroy(stream);
 *    }
 *  \endcode
 *
 */
THRUST_INLINE_CONSTANT par_nosync_t par_nosync;
}    // namespace cuda_

namespace system {
namespace cuda {
  using thrust::cuda_cub::par;
  using thrust::cuda_cub::par_nosync;
  namespace detail {
    using thrust::cuda_cub::par_t;
    using thrust::cuda_cub::par_nosync_t;
  }
} // namesapce cuda
} // namespace system

namespace cuda {
using thrust::cuda_cub::par;
using thrust::cuda_cub::par_nosync;
} // namespace cuda

THRUST_NAMESPACE_END

