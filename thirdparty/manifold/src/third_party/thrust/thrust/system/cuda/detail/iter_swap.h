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

#pragma once

#include <thrust/detail/config.h>

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC

#include <thrust/system/cuda/config.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/swap.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {


template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
inline __host__ __device__
void iter_swap(thrust::cuda::execution_policy<DerivedPolicy> &, Pointer1 a, Pointer2 b)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(Pointer1 a, Pointer2 b)
    {
      thrust::swap_ranges(a, a + 1, b);
    }

    __device__ inline static void device_path(Pointer1 a, Pointer2 b)
    {
      using thrust::swap;
      swap(*thrust::raw_pointer_cast(a),
           *thrust::raw_pointer_cast(b));
    }
  };

  if (THRUST_IS_HOST_CODE) {
    #if THRUST_INCLUDE_HOST_CODE
      war_nvbugs_881631::host_path(a, b);
    #endif
  } else {
    #if THRUST_INCLUDE_DEVICE_CODE
      war_nvbugs_881631::device_path(a, b);
    #endif
  }
} // end iter_swap()


} // end cuda_cub
THRUST_NAMESPACE_END
#endif
