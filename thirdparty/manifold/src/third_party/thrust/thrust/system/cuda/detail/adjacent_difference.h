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
#include <thrust/detail/cstdint.h>
#include <thrust/detail/minmax.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/type_traits.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/config.h>
#include <thrust/system/cuda/detail/dispatch.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/remove_cvref.h>

#include <cub/device/device_adjacent_difference.cuh>
#include <cub/device/device_select.cuh>
#include <cub/util_math.cuh>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
__host__ __device__ OutputIterator
adjacent_difference(
    const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    InputIterator                                               first,
    InputIterator                                               last,
    OutputIterator                                              result,
    BinaryFunction                                              binary_op);

namespace cuda_cub {

namespace __adjacent_difference {

  template <bool MayAlias,
            class InputIt,
            class OutputIt,
            class BinaryOp>
  cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *d_temp_storage,
            size_t &temp_storage_bytes,
            InputIt first,
            OutputIt result,
            BinaryOp binary_op,
            std::size_t num_items,
            cudaStream_t stream,
            bool debug_sync)
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    constexpr bool may_alias = MayAlias;
    constexpr bool read_left = true;

    using Dispatch32 = cub::DispatchAdjacentDifference<InputIt,
                                                       OutputIt,
                                                       BinaryOp,
                                                       thrust::detail::int32_t,
                                                       may_alias,
                                                       read_left>;
    using Dispatch64 = cub::DispatchAdjacentDifference<InputIt,
                                                       OutputIt,
                                                       BinaryOp,
                                                       thrust::detail::int64_t,
                                                       may_alias,
                                                       read_left>;

    cudaError_t status;
    THRUST_INDEX_TYPE_DISPATCH2(status,
                                Dispatch32::Dispatch,
                                Dispatch64::Dispatch,
                                num_items,
                                (d_temp_storage,
                                 temp_storage_bytes,
                                 first,
                                 result,
                                 num_items_fixed,
                                 binary_op,
                                 stream,
                                 debug_sync));
    return status;
  }

  template <class InputIt,
            class OutputIt,
            class BinaryOp>
  cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *d_temp_storage,
            size_t &temp_storage_bytes,
            InputIt first,
            OutputIt result,
            BinaryOp binary_op,
            std::size_t num_items,
            cudaStream_t stream,
            bool debug_sync,
            thrust::detail::integral_constant<bool, false> /* comparable */)
  {
    constexpr bool may_alias = true;
    return doit_step<may_alias>(d_temp_storage,
                                temp_storage_bytes,
                                first,
                                result,
                                binary_op,
                                num_items,
                                stream,
                                debug_sync);
  }

  template <class InputIt,
            class OutputIt,
            class BinaryOp>
  cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *d_temp_storage,
            size_t &temp_storage_bytes,
            InputIt first,
            OutputIt result,
            BinaryOp binary_op,
            std::size_t num_items,
            cudaStream_t stream,
            bool debug_sync,
            thrust::detail::integral_constant<bool, true> /* comparable */)
  {
    // The documentation states that pointers might be equal but can't alias in
    // any other way. That is, the distance should be equal to zero or exceed
    // `num_items`. In the latter case, we use an optimized version.
    if (first != result)
    {
      constexpr bool may_alias = false;
      return doit_step<may_alias>(d_temp_storage,
                                  temp_storage_bytes,
                                  first,
                                  result,
                                  binary_op,
                                  num_items,
                                  stream,
                                  debug_sync);
    }

    constexpr bool may_alias = true;
    return doit_step<may_alias>(d_temp_storage,
                                temp_storage_bytes,
                                first,
                                result,
                                binary_op,
                                num_items,
                                stream,
                                debug_sync);
  }

  template <typename Derived,
            typename InputIt,
            typename OutputIt,
            typename BinaryOp>
  OutputIt THRUST_RUNTIME_FUNCTION
  adjacent_difference(execution_policy<Derived>& policy,
                      InputIt                    first,
                      InputIt                    last,
                      OutputIt                   result,
                      BinaryOp                   binary_op)
  {
    const auto num_items =
      static_cast<std::size_t>(thrust::distance(first, last));
    std::size_t storage_size = 0;
    cudaStream_t stream = cuda_cub::stream(policy);
    const bool debug_sync = THRUST_DEBUG_SYNC_FLAG;

    using UnwrapInputIt = thrust::detail::try_unwrap_contiguous_iterator_return_t<InputIt>;
    using UnwrapOutputIt = thrust::detail::try_unwrap_contiguous_iterator_return_t<OutputIt>;

    using InputValueT = thrust::iterator_value_t<UnwrapInputIt>;
    using OutputValueT = thrust::iterator_value_t<UnwrapOutputIt>;

    constexpr bool can_compare_iterators =
      std::is_pointer<UnwrapInputIt>::value &&
      std::is_pointer<UnwrapOutputIt>::value &&
      std::is_same<InputValueT, OutputValueT>::value;

    auto first_unwrap = thrust::detail::try_unwrap_contiguous_iterator(first);
    auto result_unwrap = thrust::detail::try_unwrap_contiguous_iterator(result);

    thrust::detail::integral_constant<bool, can_compare_iterators> comparable;

    cudaError_t status = doit_step(nullptr,
                                   storage_size,
                                   first_unwrap,
                                   result_unwrap,
                                   binary_op,
                                   num_items,
                                   stream,
                                   debug_sync,
                                   comparable);
    cuda_cub::throw_on_error(status, "adjacent_difference failed on 1st step");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);

    status = doit_step(static_cast<void *>(tmp.data().get()),
                       storage_size,
                       first_unwrap,
                       result_unwrap,
                       binary_op,
                       num_items,
                       stream,
                       debug_sync,
                       comparable);
    cuda_cub::throw_on_error(status, "adjacent_difference failed on 2nd step");

    status = cuda_cub::synchronize_optional(policy);
    cuda_cub::throw_on_error(status, "adjacent_difference failed to synchronize");

    return result + num_items;
  }

}    // namespace __adjacent_difference

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class OutputIt,
          class BinaryOp>
OutputIt __host__ __device__
adjacent_difference(execution_policy<Derived> &policy,
                    InputIt                    first,
                    InputIt                    last,
                    OutputIt                   result,
                    BinaryOp                   binary_op)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __adjacent_difference::adjacent_difference(policy,
        first,
        last,
        result,
        binary_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::adjacent_difference(cvt_to_seq(derived_cast(policy)),
                                      first,
                                      last,
                                      result,
                                      binary_op);
#endif
  }

  return ret;
}

template <class Derived,
          class InputIt,
          class OutputIt>
OutputIt __host__ __device__
adjacent_difference(execution_policy<Derived> &policy,
                    InputIt                    first,
                    InputIt                    last,
                    OutputIt                   result)
{
  typedef typename iterator_traits<InputIt>::value_type input_type;
  return cuda_cub::adjacent_difference(policy,
                                       first,
                                       last,
                                       result,
                                       minus<input_type>());
}


} // namespace cuda_cub
THRUST_NAMESPACE_END

//
#include <thrust/memory.h>
#include <thrust/adjacent_difference.h>
#endif

