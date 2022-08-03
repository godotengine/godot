/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditionu and the following disclaimer.
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

// XXX
// this file must not be included on its own, ever,
// but must be part of include in thrust/system/cuda/detail/copy.h

#include <thrust/detail/config.h>

#include <thrust/system/cuda/config.h>

#include <thrust/distance.h>
#include <thrust/advance.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/cuda/detail/uninitialized_copy.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

namespace __copy {


  template <class H,
            class D,
            class T,
            class Size>
  THRUST_HOST_FUNCTION void
  trivial_device_copy(thrust::cpp::execution_policy<H>&      ,
                      thrust::cuda_cub::execution_policy<D>& device_s,
                      T*                                     dst,
                      T const*                               src,
                      Size                                   count)
  {
    cudaError status;
    status = cuda_cub::trivial_copy_to_device(dst,
                                              src,
                                              count,
                                              cuda_cub::stream(device_s));
    cuda_cub::throw_on_error(status, "__copy::trivial_device_copy H->D: failed");
  }

  template <class D,
            class H,
            class T,
            class Size>
  THRUST_HOST_FUNCTION void
  trivial_device_copy(thrust::cuda_cub::execution_policy<D>& device_s,
                      thrust::cpp::execution_policy<H>&      ,
                      T*                                     dst,
                      T const*                               src,
                      Size                                   count)
  {
    cudaError status;
    status = cuda_cub::trivial_copy_from_device(dst,
                                                src,
                                                count,
                                                cuda_cub::stream(device_s));
    cuda_cub::throw_on_error(status, "trivial_device_copy D->H failed");
  }

  template <class System1,
            class System2,
            class InputIt,
            class Size,
            class OutputIt>
  OutputIt __host__
  cross_system_copy_n(thrust::execution_policy<System1>& sys1,
                      thrust::execution_policy<System2>& sys2,
                      InputIt                            begin,
                      Size                               n,
                      OutputIt                           result,
                      thrust::detail::true_type)    // trivial copy

  {
    typedef typename iterator_traits<InputIt>::value_type InputTy;
    if (n > 0) {
      trivial_device_copy(derived_cast(sys1),
                          derived_cast(sys2),
                          reinterpret_cast<InputTy*>(thrust::raw_pointer_cast(&*result)),
                          reinterpret_cast<InputTy const*>(thrust::raw_pointer_cast(&*begin)),
                          n);
    }

    return result + n;
  }

  // non-trivial H->D copy
  template <class H,
            class D,
            class InputIt,
            class Size,
            class OutputIt>
  OutputIt __host__
  cross_system_copy_n(thrust::cpp::execution_policy<H>&      host_s,
                      thrust::cuda_cub::execution_policy<D>& device_s,
                      InputIt                                first,
                      Size                                   num_items,
                      OutputIt                               result,
                      thrust::detail::false_type)    // non-trivial copy
  {
    // get type of the input data
    typedef typename thrust::iterator_value<InputIt>::type InputTy;

    // copy input data into host temp storage
    InputIt last = first;
    thrust::advance(last, num_items);
    thrust::detail::temporary_array<InputTy, H> temp(host_s, num_items);

    for (Size idx = 0; idx != num_items; idx++)
    {
      ::new (static_cast<void*>(temp.data().get()+idx)) InputTy(*first);
      ++first;
    }

    // allocate device temporary storage
    thrust::detail::temporary_array<InputTy, D> d_in_ptr(device_s, num_items);

    // trivial copy data from host to device
    cudaError status = cuda_cub::trivial_copy_to_device(d_in_ptr.data().get(),
                                                        temp.data().get(),
                                                        num_items,
                                                        cuda_cub::stream(device_s));
    cuda_cub::throw_on_error(status, "__copy:: H->D: failed");


    // device->device copy
    OutputIt ret = cuda_cub::copy_n(device_s, d_in_ptr.data(), num_items, result);

    return ret;
  }

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
  // non-trivial copy D->H, only supported with NVCC compiler
  // because copy ctor must have  __device__ annotations, which is nvcc-only
  // feature
  template <class D,
            class H,
            class InputIt,
            class Size,
            class OutputIt>
  OutputIt __host__
  cross_system_copy_n(thrust::cuda_cub::execution_policy<D>& device_s,
                      thrust::cpp::execution_policy<H>&   host_s,
                      InputIt                             first,
                      Size                                num_items,
                      OutputIt                            result,
                      thrust::detail::false_type)    // non-trivial copy

  {
    // get type of the input data
    typedef typename thrust::iterator_value<InputIt>::type InputTy;

    // allocate device temp storage 
    thrust::detail::temporary_array<InputTy, D> d_in_ptr(device_s, num_items);

    // uninitialize copy into temp device storage
    cuda_cub::uninitialized_copy_n(device_s, first, num_items, d_in_ptr.data());

    // allocate host temp storage
    thrust::detail::temporary_array<InputTy, H> temp(host_s, num_items);

    // trivial copy from device to host
    cudaError status;
    status = cuda_cub::trivial_copy_from_device(temp.data().get(),
                                                d_in_ptr.data().get(),
                                                num_items,
                                                cuda_cub::stream(device_s));
    cuda_cub::throw_on_error(status, "__copy:: D->H: failed");

    // host->host copy
    OutputIt ret = thrust::copy_n(host_s, temp.data(), num_items, result);

    return ret;
  }
#endif

  template <class System1,
            class System2,
            class InputIt,
            class Size,
            class OutputIt>
  OutputIt __host__
  cross_system_copy_n(cross_system<System1, System2> systems,
                      InputIt  begin,
                      Size     n,
                      OutputIt result)
  {
    return cross_system_copy_n(
        derived_cast(systems.sys1),
        derived_cast(systems.sys2),
        begin,
        n,
        result,
        typename is_indirectly_trivially_relocatable_to<InputIt, OutputIt>::type());
  }

  template <class System1,
            class System2,
            class InputIterator,
            class OutputIterator>
  OutputIterator __host__
  cross_system_copy(cross_system<System1, System2> systems,
                    InputIterator  begin,
                    InputIterator  end,
                    OutputIterator result)
  {
    return cross_system_copy_n(systems,
                               begin,
                               thrust::distance(begin, end),
                               result);
  }

}    // namespace __copy

} // namespace cuda_cub
THRUST_NAMESPACE_END
