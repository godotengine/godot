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
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/config.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/core/util.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_merge_sort.cuh>

#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/detail/trivial_sequence.h>
#include <thrust/detail/integer_math.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/distance.h>
#include <thrust/sequence.h>
#include <thrust/detail/alignment.h>
#include <thrust/type_traits/is_contiguous_iterator.h>


THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

namespace __merge_sort {

  template <class KeysIt,
            class ItemsIt,
            class Size,
            class CompareOp>
  THRUST_RUNTIME_FUNCTION cudaError_t
  doit_step(void*        d_temp_storage,
            size_t&      temp_storage_bytes,
            KeysIt       keys,
            ItemsIt      ,
            Size         keys_count,
            CompareOp    compare_op,
            cudaStream_t stream,
            bool         debug_sync,
            thrust::detail::integral_constant<bool, false> /* sort_keys */)
  {
    using ItemsInputIt = cub::NullType *;
    ItemsInputIt items = nullptr;

    using DispatchMergeSortT = cub::DispatchMergeSort<KeysIt,
                                                      ItemsInputIt,
                                                      KeysIt,
                                                      ItemsInputIt,
                                                      Size,
                                                      CompareOp>;


    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        keys,
                                        items,
                                        keys,
                                        items,
                                        keys_count,
                                        compare_op,
                                        stream,
                                        debug_sync);
  }

  template <class KeysIt,
            class ItemsIt,
            class Size,
            class CompareOp>
  THRUST_RUNTIME_FUNCTION cudaError_t
  doit_step(void *d_temp_storage,
            size_t &temp_storage_bytes,
            KeysIt keys,
            ItemsIt items,
            Size keys_count,
            CompareOp compare_op,
            cudaStream_t stream,
            bool debug_sync,
            thrust::detail::integral_constant<bool, true> /* sort_items */)
  {
    using DispatchMergeSortT =
      cub::DispatchMergeSort<KeysIt, ItemsIt, KeysIt, ItemsIt, Size, CompareOp>;

    return DispatchMergeSortT::Dispatch(d_temp_storage,
                                        temp_storage_bytes,
                                        keys,
                                        items,
                                        keys,
                                        items,
                                        keys_count,
                                        compare_op,
                                        stream,
                                        debug_sync);
  }

  template <class SORT_ITEMS,
            class /* STABLE */,
            class KeysIt,
            class ItemsIt,
            class Size,
            class CompareOp>
  THRUST_RUNTIME_FUNCTION cudaError_t
  doit_step(void *d_temp_storage,
            size_t &temp_storage_bytes,
            KeysIt keys,
            ItemsIt items,
            Size keys_count,
            CompareOp compare_op,
            cudaStream_t stream,
            bool debug_sync)
  {
    if (keys_count == 0)
    {
      return cudaSuccess;
    }

    thrust::detail::integral_constant<bool, SORT_ITEMS::value> sort_items{};

    return doit_step(d_temp_storage,
                     temp_storage_bytes,
                     keys,
                     items,
                     keys_count,
                     compare_op,
                     stream,
                     debug_sync,
                     sort_items);
  }

  template <typename SORT_ITEMS,
            typename STABLE,
            typename Derived,
            typename KeysIt,
            typename ItemsIt,
            typename CompareOp>
  THRUST_RUNTIME_FUNCTION
  void merge_sort(execution_policy<Derived>& policy,
                  KeysIt                     keys_first,
                  KeysIt                     keys_last,
                  ItemsIt                    items_first,
                  CompareOp                  compare_op)

  {
    typedef typename iterator_traits<KeysIt>::difference_type size_type;

    size_type count = static_cast<size_type>(thrust::distance(keys_first, keys_last));

    size_t       storage_size = 0;
    cudaStream_t stream       = cuda_cub::stream(policy);
    bool         debug_sync   = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    status = doit_step<SORT_ITEMS, STABLE>(NULL,
                                           storage_size,
                                           keys_first,
                                           items_first,
                                           count,
                                           compare_op,
                                           stream,
                                           debug_sync);
    cuda_cub::throw_on_error(status, "merge_sort: failed on 1st step");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = doit_step<SORT_ITEMS, STABLE>(ptr,
                                           storage_size,
                                           keys_first,
                                           items_first,
                                           count,
                                           compare_op,
                                           stream,
                                           debug_sync);
    cuda_cub::throw_on_error(status, "merge_sort: failed on 2nd step");

    status = cuda_cub::synchronize_optional(policy);
    cuda_cub::throw_on_error(status, "merge_sort: failed to synchronize");
  }
}    // namespace __merge_sort

namespace __radix_sort {

  template <class SORT_ITEMS, class Comparator>
  struct dispatch;

  // sort keys in ascending order
  template <class K>
  struct dispatch<thrust::detail::false_type, thrust::less<K> >
  {
    template <class Key, class Item, class Size>
    THRUST_RUNTIME_FUNCTION static cudaError_t
    doit(void*                    d_temp_storage,
         size_t&                  temp_storage_bytes,
         cub::DoubleBuffer<Key>&  keys_buffer,
         cub::DoubleBuffer<Item>& /*items_buffer*/,
         Size                     count,
         cudaStream_t             stream,
         bool                     debug_sync)
    {
      return cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                            temp_storage_bytes,
                                            keys_buffer,
                                            static_cast<int>(count),
                                            0,
                                            static_cast<int>(sizeof(Key) * 8),
                                            stream,
                                            debug_sync);
    }
  }; // struct dispatch -- sort keys in ascending order;

  // sort keys in descending order
  template <class K>
  struct dispatch<thrust::detail::false_type, thrust::greater<K> >
  {
    template <class Key, class Item, class Size>
    THRUST_RUNTIME_FUNCTION static cudaError_t
    doit(void*                    d_temp_storage,
         size_t&                  temp_storage_bytes,
         cub::DoubleBuffer<Key>&  keys_buffer,
         cub::DoubleBuffer<Item>& /*items_buffer*/,
         Size                     count,
         cudaStream_t             stream,
         bool                     debug_sync)
    {
      return cub::DeviceRadixSort::SortKeysDescending(d_temp_storage,
                                                      temp_storage_bytes,
                                                      keys_buffer,
                                                      static_cast<int>(count),
                                                      0,
                                                      static_cast<int>(sizeof(Key) * 8),
                                                      stream,
                                                      debug_sync);
    }
  }; // struct dispatch -- sort keys in descending order;

  // sort pairs in ascending order
  template <class K>
  struct dispatch<thrust::detail::true_type, thrust::less<K> >
  {
    template <class Key, class Item, class Size>
    THRUST_RUNTIME_FUNCTION static cudaError_t
    doit(void*                    d_temp_storage,
         size_t&                  temp_storage_bytes,
         cub::DoubleBuffer<Key>&  keys_buffer,
         cub::DoubleBuffer<Item>& items_buffer,
         Size                     count,
         cudaStream_t             stream,
         bool                     debug_sync)
    {
      return cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                             temp_storage_bytes,
                                             keys_buffer,
                                             items_buffer,
                                             static_cast<int>(count),
                                             0,
                                             static_cast<int>(sizeof(Key) * 8),
                                             stream,
                                             debug_sync);
    }
  }; // struct dispatch -- sort pairs in ascending order;

  // sort pairs in descending order
  template <class K>
  struct dispatch<thrust::detail::true_type, thrust::greater<K> >
  {
    template <class Key, class Item, class Size>
    THRUST_RUNTIME_FUNCTION static cudaError_t
    doit(void*                    d_temp_storage,
         size_t&                  temp_storage_bytes,
         cub::DoubleBuffer<Key>&  keys_buffer,
         cub::DoubleBuffer<Item>& items_buffer,
         Size                     count,
         cudaStream_t             stream,
         bool                     debug_sync)
    {
      return cub::DeviceRadixSort::SortPairsDescending(d_temp_storage,
                                                       temp_storage_bytes,
                                                       keys_buffer,
                                                       items_buffer,
                                                       static_cast<int>(count),
                                                       0,
                                                       static_cast<int>(sizeof(Key) * 8),
                                                       stream,
                                                       debug_sync);
    }
  }; // struct dispatch -- sort pairs in descending order;

  template <typename SORT_ITEMS,
            typename Derived,
            typename Key,
            typename Item,
            typename Size,
            typename CompareOp>
  THRUST_RUNTIME_FUNCTION
  void radix_sort(execution_policy<Derived>& policy,
                  Key*                       keys,
                  Item*                      items,
                  Size                       count,
                  CompareOp)
  {
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    cub::DoubleBuffer<Key>  keys_buffer(keys, NULL);
    cub::DoubleBuffer<Item> items_buffer(items, NULL);

    Size keys_count = count;
    Size items_count = SORT_ITEMS::value ? count : 0;

    cudaError_t status;

    status = dispatch<SORT_ITEMS, CompareOp>::doit(NULL,
                                                   temp_storage_bytes,
                                                   keys_buffer,
                                                   items_buffer,
                                                   keys_count,
                                                   stream,
                                                   debug_sync);
    cuda_cub::throw_on_error(status, "radix_sort: failed on 1st step");

    size_t keys_temp_storage  = core::align_to(sizeof(Key) * keys_count, 128);
    size_t items_temp_storage = core::align_to(sizeof(Item) * items_count, 128);

    size_t storage_size = keys_temp_storage
                        + items_temp_storage
                        + temp_storage_bytes;

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);

    keys_buffer.d_buffers[1]  = thrust::detail::aligned_reinterpret_cast<Key*>(
      tmp.data().get()
    );
    items_buffer.d_buffers[1] = thrust::detail::aligned_reinterpret_cast<Item*>(
      tmp.data().get() + keys_temp_storage
    );
    void *ptr = static_cast<void*>(
      tmp.data().get() + keys_temp_storage + items_temp_storage
    );

    status = dispatch<SORT_ITEMS, CompareOp>::doit(ptr,
                                                   temp_storage_bytes,
                                                   keys_buffer,
                                                   items_buffer,
                                                   keys_count,
                                                   stream,
                                                   debug_sync);
    cuda_cub::throw_on_error(status, "radix_sort: failed on 2nd step");

    if (keys_buffer.selector != 0)
    {
      Key* temp_ptr = reinterpret_cast<Key*>(keys_buffer.d_buffers[1]);
      cuda_cub::copy_n(policy, temp_ptr, keys_count, keys);
    }
    THRUST_IF_CONSTEXPR(SORT_ITEMS::value)
    {
      if (items_buffer.selector != 0)
      {
        Item *temp_ptr = reinterpret_cast<Item *>(items_buffer.d_buffers[1]);
        cuda_cub::copy_n(policy, temp_ptr, items_count, items);
      }
    }
  }
}    // __radix_sort

//---------------------------------------------------------------------
// Smart sort picks at compile-time whether to dispatch radix or merge sort
//---------------------------------------------------------------------

namespace __smart_sort {

  template <class Key, class CompareOp>
  struct can_use_primitive_sort
      : thrust::detail::and_<
            thrust::detail::is_arithmetic<Key>,
            thrust::detail::or_<
                thrust::detail::is_same<CompareOp, thrust::less<Key> >,
                thrust::detail::is_same<CompareOp, thrust::greater<Key> > > > {};

  template <class Iterator, class CompareOp>
  struct enable_if_primitive_sort
      : thrust::detail::enable_if<
            can_use_primitive_sort<typename iterator_value<Iterator>::type,
                                   CompareOp>::value> {};

  template <class Iterator, class CompareOp>
  struct enable_if_comparison_sort
      : thrust::detail::disable_if<
            can_use_primitive_sort<typename iterator_value<Iterator>::type,
                                   CompareOp>::value> {};


  template <class SORT_ITEMS,
            class STABLE,
            class Policy,
            class KeysIt,
            class ItemsIt,
            class CompareOp>
  THRUST_RUNTIME_FUNCTION typename enable_if_comparison_sort<KeysIt, CompareOp>::type
  smart_sort(Policy&   policy,
             KeysIt    keys_first,
             KeysIt    keys_last,
             ItemsIt   items_first,
             CompareOp compare_op)
  {
    __merge_sort::merge_sort<SORT_ITEMS, STABLE>(policy,
                                                 keys_first,
                                                 keys_last,
                                                 items_first,
                                                 compare_op);

  }

  template <class SORT_ITEMS,
            class STABLE,
            class Policy,
            class KeysIt,
            class ItemsIt,
            class CompareOp>
  THRUST_RUNTIME_FUNCTION typename enable_if_primitive_sort<KeysIt, CompareOp>::type
  smart_sort(execution_policy<Policy>& policy,
             KeysIt                    keys_first,
             KeysIt                    keys_last,
             ItemsIt                   items_first,
             CompareOp                 compare_op)
  {
    // ensure sequences have trivial iterators
    thrust::detail::trivial_sequence<KeysIt, Policy>
        keys(policy, keys_first, keys_last);

    if (SORT_ITEMS::value)
    {
      thrust::detail::trivial_sequence<ItemsIt, Policy>
          values(policy, items_first, items_first + (keys_last - keys_first));

      __radix_sort::radix_sort<SORT_ITEMS>(
          policy,
          thrust::raw_pointer_cast(&*keys.begin()),
          thrust::raw_pointer_cast(&*values.begin()),
          keys_last - keys_first,
          compare_op);

      if (!is_contiguous_iterator<ItemsIt>::value)
      {
        cuda_cub::copy(policy, values.begin(), values.end(), items_first);
      }
    }
    else
    {
      __radix_sort::radix_sort<SORT_ITEMS>(
          policy,
          thrust::raw_pointer_cast(&*keys.begin()),
          thrust::raw_pointer_cast(&*keys.begin()),
          keys_last - keys_first,
          compare_op);
    }

    // copy results back, if necessary
    if (!is_contiguous_iterator<KeysIt>::value)
    {
      cuda_cub::copy(policy, keys.begin(), keys.end(), keys_first);
    }

    cuda_cub::throw_on_error(
      cuda_cub::synchronize_optional(policy),
      "smart_sort: failed to synchronize");
  }
}    // namespace __smart_sort


//-------------------------
// Thrust API entry points
//-------------------------


__thrust_exec_check_disable__
template <class Derived, class ItemsIt, class CompareOp>
void __host__ __device__
sort(execution_policy<Derived>& policy,
     ItemsIt                    first,
     ItemsIt                    last,
     CompareOp                  compare_op)
{
  if (__THRUST_HAS_CUDART__)
  {
    typedef typename thrust::iterator_value<ItemsIt>::type item_type;
    __smart_sort::smart_sort<thrust::detail::false_type, thrust::detail::false_type>(
        policy, first, last, (item_type*)NULL, compare_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    thrust::sort(cvt_to_seq(derived_cast(policy)), first, last, compare_op);
#endif
  }
}

__thrust_exec_check_disable__
template <class Derived, class ItemsIt, class CompareOp>
void __host__ __device__
stable_sort(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last,
            CompareOp                  compare_op)
{
  if (__THRUST_HAS_CUDART__)
  {
    typedef typename thrust::iterator_value<ItemsIt>::type item_type;
    __smart_sort::smart_sort<thrust::detail::false_type, thrust::detail::true_type>(
        policy, first, last, (item_type*)NULL, compare_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    thrust::stable_sort(cvt_to_seq(derived_cast(policy)), first, last, compare_op);
#endif
  }
}

__thrust_exec_check_disable__
template <class Derived, class KeysIt, class ValuesIt, class CompareOp>
void __host__ __device__
sort_by_key(execution_policy<Derived>& policy,
            KeysIt                     keys_first,
            KeysIt                     keys_last,
            ValuesIt                   values,
            CompareOp                  compare_op)
{
  if (__THRUST_HAS_CUDART__)
  {
    __smart_sort::smart_sort<thrust::detail::true_type, thrust::detail::false_type>(
        policy, keys_first, keys_last, values, compare_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    thrust::sort_by_key(
        cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values, compare_op);
#endif
  }
}

__thrust_exec_check_disable__
template <class Derived,
          class KeysIt,
          class ValuesIt,
          class CompareOp>
void __host__ __device__
stable_sort_by_key(execution_policy<Derived> &policy,
            KeysIt                     keys_first,
            KeysIt                     keys_last,
            ValuesIt                   values,
            CompareOp                  compare_op)
{
  if (__THRUST_HAS_CUDART__)
  {
    __smart_sort::smart_sort<thrust::detail::true_type, thrust::detail::true_type>(
        policy, keys_first, keys_last, values, compare_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    thrust::stable_sort_by_key(
        cvt_to_seq(derived_cast(policy)), keys_first, keys_last, values, compare_op);
#endif
  }
}

// API with default comparator

template <class Derived, class ItemsIt>
void __host__ __device__
sort(execution_policy<Derived>& policy,
     ItemsIt                    first,
     ItemsIt                    last)
{
  typedef typename thrust::iterator_value<ItemsIt>::type item_type;
  cuda_cub::sort(policy, first, last, less<item_type>());
}

template <class Derived, class ItemsIt>
void __host__ __device__
stable_sort(execution_policy<Derived>& policy,
            ItemsIt                    first,
            ItemsIt                    last)
{
  typedef typename thrust::iterator_value<ItemsIt>::type item_type;
  cuda_cub::stable_sort(policy, first, last, less<item_type>());
}

template <class Derived, class KeysIt, class ValuesIt>
void __host__ __device__
sort_by_key(execution_policy<Derived>& policy,
            KeysIt                     keys_first,
            KeysIt                     keys_last,
            ValuesIt                   values)
{
  typedef typename thrust::iterator_value<KeysIt>::type key_type;
  cuda_cub::sort_by_key(policy, keys_first, keys_last, values, less<key_type>());
}

template <class Derived, class KeysIt, class ValuesIt>
void __host__ __device__
stable_sort_by_key(
    execution_policy<Derived>& policy, KeysIt keys_first, KeysIt keys_last, ValuesIt values)
{
  typedef typename thrust::iterator_value<KeysIt>::type key_type;
  cuda_cub::stable_sort_by_key(policy, keys_first, keys_last, values, less<key_type>());
}


}    // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
