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

#include <thrust/distance.h>
#include <thrust/functional.h>

#include <thrust/type_traits/is_contiguous_iterator.h>

#include <thrust/iterator/iterator_traits.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>

#include <thrust/system/cuda/config.h>
#include <thrust/system/cuda/detail/dispatch.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/system/cuda/detail/util.h>

#include <cub/device/dispatch/dispatch_scan_by_key.cuh>
#include <cub/util_type.cuh>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace detail
{

__thrust_exec_check_disable__
template <typename Derived,
          typename KeysInIt,
          typename ValuesInIt,
          typename ValuesOutIt,
          typename EqualityOpT,
          typename ScanOpT,
          typename SizeT>
__host__ __device__
ValuesOutIt inclusive_scan_by_key_n(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  KeysInIt keys,
  ValuesInIt values,
  ValuesOutIt result,
  SizeT num_items,
  EqualityOpT equality_op,
  ScanOpT scan_op)
{
  if (num_items == 0)
  {
    return result;
  }

  // Convert to raw pointers if possible:
  using KeysInUnwrapIt =
    thrust::detail::try_unwrap_contiguous_iterator_return_t<KeysInIt>;
  using ValuesInUnwrapIt =
    thrust::detail::try_unwrap_contiguous_iterator_return_t<ValuesInIt>;
  using ValuesOutUnwrapIt =
    thrust::detail::try_unwrap_contiguous_iterator_return_t<ValuesOutIt>;

  auto keys_unwrap = thrust::detail::try_unwrap_contiguous_iterator(keys);
  auto values_unwrap = thrust::detail::try_unwrap_contiguous_iterator(values);
  auto result_unwrap = thrust::detail::try_unwrap_contiguous_iterator(result);

  using Dispatch32 = cub::DispatchScanByKey<KeysInUnwrapIt,
                                            ValuesInUnwrapIt,
                                            ValuesOutUnwrapIt,
                                            EqualityOpT,
                                            ScanOpT,
                                            cub::NullType,
                                            thrust::detail::int32_t>;
  using Dispatch64 = cub::DispatchScanByKey<KeysInUnwrapIt,
                                            ValuesInUnwrapIt,
                                            ValuesOutUnwrapIt,
                                            EqualityOpT,
                                            ScanOpT,
                                            cub::NullType,
                                            thrust::detail::int64_t>;

  cudaStream_t stream = thrust::cuda_cub::stream(policy);
  cudaError_t status{};

  // Determine temporary storage requirements:
  std::size_t tmp_size = 0;
  {
    THRUST_INDEX_TYPE_DISPATCH2(status,
                                Dispatch32::Dispatch,
                                Dispatch64::Dispatch,
                                num_items,
                                (nullptr,
                                 tmp_size,
                                 keys_unwrap,
                                 values_unwrap,
                                 result_unwrap,
                                 equality_op,
                                 scan_op,
                                 cub::NullType{},
                                 num_items_fixed,
                                 stream,
                                 THRUST_DEBUG_SYNC_FLAG));
    thrust::cuda_cub::throw_on_error(status,
                                     "after determining tmp storage "
                                     "requirements for inclusive_scan_by_key");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived> tmp{
      policy,
      tmp_size};

    THRUST_INDEX_TYPE_DISPATCH2(status,
                                Dispatch32::Dispatch,
                                Dispatch64::Dispatch,
                                num_items,
                                (tmp.data().get(),
                                 tmp_size,
                                 keys_unwrap,
                                 values_unwrap,
                                 result_unwrap,
                                 equality_op,
                                 scan_op,
                                 cub::NullType{},
                                 num_items_fixed,
                                 stream,
                                 THRUST_DEBUG_SYNC_FLAG));

    thrust::cuda_cub::throw_on_error(
      status, "after dispatching inclusive_scan_by_key kernel");

    thrust::cuda_cub::throw_on_error(
      thrust::cuda_cub::synchronize_optional(policy),
      "inclusive_scan_by_key failed to synchronize");
  }

  return result + num_items;
}

__thrust_exec_check_disable__
template <typename Derived,
          typename KeysInIt,
          typename ValuesInIt,
          typename ValuesOutIt,
          typename InitValueT,
          typename EqualityOpT,
          typename ScanOpT,
          typename SizeT>
__host__ __device__
ValuesOutIt exclusive_scan_by_key_n(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  KeysInIt keys,
  ValuesInIt values,
  ValuesOutIt result,
  SizeT num_items,
  InitValueT init_value,
  EqualityOpT equality_op,
  ScanOpT scan_op)
{

  if (num_items == 0)
  {
    return result;
  }

  // Convert to raw pointers if possible:
  using KeysInUnwrapIt =
    thrust::detail::try_unwrap_contiguous_iterator_return_t<KeysInIt>;
  using ValuesInUnwrapIt =
    thrust::detail::try_unwrap_contiguous_iterator_return_t<ValuesInIt>;
  using ValuesOutUnwrapIt =
    thrust::detail::try_unwrap_contiguous_iterator_return_t<ValuesOutIt>;

  auto keys_unwrap = thrust::detail::try_unwrap_contiguous_iterator(keys);
  auto values_unwrap = thrust::detail::try_unwrap_contiguous_iterator(values);
  auto result_unwrap = thrust::detail::try_unwrap_contiguous_iterator(result);

  using Dispatch32 = cub::DispatchScanByKey<KeysInUnwrapIt,
                                            ValuesInUnwrapIt,
                                            ValuesOutUnwrapIt,
                                            EqualityOpT,
                                            ScanOpT,
                                            InitValueT,
                                            thrust::detail::int32_t>;
  using Dispatch64 = cub::DispatchScanByKey<KeysInUnwrapIt,
                                            ValuesInUnwrapIt,
                                            ValuesOutUnwrapIt,
                                            EqualityOpT,
                                            ScanOpT,
                                            InitValueT,
                                            thrust::detail::int64_t>;

  cudaStream_t stream = thrust::cuda_cub::stream(policy);
  cudaError_t status{};

  // Determine temporary storage requirements:
  std::size_t tmp_size = 0;
  {
    THRUST_INDEX_TYPE_DISPATCH2(status,
                                Dispatch32::Dispatch,
                                Dispatch64::Dispatch,
                                num_items,
                                (nullptr,
                                 tmp_size,
                                 keys_unwrap,
                                 values_unwrap,
                                 result_unwrap,
                                 equality_op,
                                 scan_op,
                                 init_value,
                                 num_items_fixed,
                                 stream,
                                 THRUST_DEBUG_SYNC_FLAG));
    thrust::cuda_cub::throw_on_error(status,
                                     "after determining tmp storage "
                                     "requirements for exclusive_scan_by_key");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived> tmp{
      policy,
      tmp_size};

    THRUST_INDEX_TYPE_DISPATCH2(status,
                                Dispatch32::Dispatch,
                                Dispatch64::Dispatch,
                                num_items,
                                (tmp.data().get(),
                                 tmp_size,
                                 keys_unwrap,
                                 values_unwrap,
                                 result_unwrap,
                                 equality_op,
                                 scan_op,
                                 init_value,
                                 num_items_fixed,
                                 stream,
                                 THRUST_DEBUG_SYNC_FLAG));

    thrust::cuda_cub::throw_on_error(
      status, "after dispatching exclusive_scan_by_key kernel");

    thrust::cuda_cub::throw_on_error(
      thrust::cuda_cub::synchronize_optional(policy),
      "exclusive_scan_by_key failed to synchronize");
  }

  return result + num_items;
}


} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

//---------------------------
//   Inclusive scan
//---------------------------

__thrust_exec_check_disable__
template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class BinaryPred,
          class ScanOp>
ValOutputIt __host__ __device__
inclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      BinaryPred                 binary_pred,
                      ScanOp                     scan_op)
{
  ValOutputIt ret = value_result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = thrust::cuda_cub::detail::inclusive_scan_by_key_n(
      policy,
      key_first,
      value_first,
      value_result,
      thrust::distance(key_first, key_last),
      binary_pred,
      scan_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::inclusive_scan_by_key(cvt_to_seq(derived_cast(policy)),
                                        key_first,
                                        key_last,
                                        value_first,
                                        value_result,
                                        binary_pred,
                                        scan_op);
#endif
  }
  return ret;
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class BinaryPred>
ValOutputIt __host__ __device__
inclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      BinaryPred                 binary_pred)
{
  return cuda_cub::inclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         binary_pred,
                                         thrust::plus<>());
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt>
ValOutputIt __host__ __device__
inclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result)
{
  return cuda_cub::inclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         thrust::equal_to<>());
}


//---------------------------
//   Exclusive scan
//---------------------------

__thrust_exec_check_disable__
template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class Init,
          class BinaryPred,
          class ScanOp>
ValOutputIt __host__ __device__
exclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      Init                       init,
                      BinaryPred                 binary_pred,
                      ScanOp                     scan_op)
{
  ValOutputIt ret = value_result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = thrust::cuda_cub::detail::exclusive_scan_by_key_n(
      policy,
      key_first,
      value_first,
      value_result,
      thrust::distance(key_first, key_last),
      init,
      binary_pred,
      scan_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::exclusive_scan_by_key(cvt_to_seq(derived_cast(policy)),
                                        key_first,
                                        key_last,
                                        value_first,
                                        value_result,
                                        init,
                                        binary_pred,
                                        scan_op);
#endif
  }
  return ret;
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class Init,
          class BinaryPred>
ValOutputIt __host__ __device__
exclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      Init                       init,
                      BinaryPred                 binary_pred)
{
  return cuda_cub::exclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         init,
                                         binary_pred,
                                         thrust::plus<>());
}

template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt,
          class Init>
ValOutputIt __host__ __device__
exclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result,
                      Init                       init)
{
  return cuda_cub::exclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         init,
                                         thrust::equal_to<>());
}


template <class Derived,
          class KeyInputIt,
          class ValInputIt,
          class ValOutputIt>
ValOutputIt __host__ __device__
exclusive_scan_by_key(execution_policy<Derived> &policy,
                      KeyInputIt                 key_first,
                      KeyInputIt                 key_last,
                      ValInputIt                 value_first,
                      ValOutputIt                value_result)
{
  using value_type = typename thrust::iterator_traits<ValInputIt>::value_type;
  return cuda_cub::exclusive_scan_by_key(policy,
                                         key_first,
                                         key_last,
                                         value_first,
                                         value_result,
                                         value_type{});
}


}    // namespace cuda_cub
THRUST_NAMESPACE_END

#include <thrust/scan.h>

#endif // NVCC
