/******************************************************************************
j * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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

#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/core/util.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/merge.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/distance.h>


THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

namespace __merge {

  template <class KeysIt1,
            class KeysIt2,
            class Size,
            class BinaryPred>
  Size THRUST_DEVICE_FUNCTION
  merge_path(KeysIt1    keys1,
             KeysIt2    keys2,
             Size       keys1_count,
             Size       keys2_count,
             Size       diag,
             BinaryPred binary_pred)
  {
    typedef typename iterator_traits<KeysIt1>::value_type key1_type;
    typedef typename iterator_traits<KeysIt2>::value_type key2_type;

    Size keys1_begin = thrust::max<Size>(0, diag - keys2_count);
    Size keys1_end   = thrust::min<Size>(diag, keys1_count);

    while (keys1_begin < keys1_end)
    {
      Size mid = (keys1_begin + keys1_end) >> 1;
      key1_type key1 = keys1[mid];
      key2_type key2 = keys2[diag - 1 - mid];
      bool pred = binary_pred(key2, key1);
      if (pred)
      {
        keys1_end = mid;
      }
      else
      {
        keys1_begin = mid+1;
      }
    }
    return keys1_begin;
  }

  template <class It, class T2, class CompareOp, int ITEMS_PER_THREAD>
  THRUST_DEVICE_FUNCTION void
  serial_merge(It  keys_shared,
               int keys1_beg,
               int keys2_beg,
               int keys1_count,
               int keys2_count,
               T2 (&output)[ITEMS_PER_THREAD],
               int (&indices)[ITEMS_PER_THREAD],
               CompareOp compare_op)
  {
    int keys1_end = keys1_beg + keys1_count;
    int keys2_end = keys2_beg + keys2_count;

    typedef typename iterator_value<It>::type key_type;

    key_type key1 = keys_shared[keys1_beg];
    key_type key2 = keys_shared[keys2_beg];


#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      bool p = (keys2_beg < keys2_end) &&
               ((keys1_beg >= keys1_end) ||
                compare_op(key2,key1));

      output[ITEM]  = p ? key2 : key1;
      indices[ITEM] = p ? keys2_beg++ : keys1_beg++;

      if (p)
      {
        key2 = keys_shared[keys2_beg];
      }
      else
      {
        key1 = keys_shared[keys1_beg];
      }
    }
  }

  template <int                      _BLOCK_THREADS,
            int                      _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm  _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier   _LOAD_MODIFIER    = cub::LOAD_LDG,
            cub::BlockStoreAlgorithm _STORE_ALGORITHM  = cub::BLOCK_STORE_DIRECT>
  struct PtxPolicy
  {
    enum
    {
      BLOCK_THREADS      = _BLOCK_THREADS,
      ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
      ITEMS_PER_TILE     = _BLOCK_THREADS * _ITEMS_PER_THREAD,
    };

    static const cub::BlockLoadAlgorithm  LOAD_ALGORITHM  = _LOAD_ALGORITHM;
    static const cub::CacheLoadModifier   LOAD_MODIFIER   = _LOAD_MODIFIER;
    static const cub::BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
  };    // PtxPolicy

  template <class KeysIt1,
            class KeysIt2,
            class Size,
            class CompareOp>
  struct PartitionAgent
  {
    template <class Arch>
    struct PtxPlan : PtxPolicy<256> {};

    typedef core::specialize_plan<PtxPlan> ptx_plan;

    THRUST_AGENT_ENTRY(KeysIt1   keys1,
                       KeysIt2   keys2,
                       Size      keys1_count,
                       Size      keys2_count,
                       Size      num_partitions,
                       Size*     merge_partitions,
                       CompareOp compare_op,
                       int       items_per_tile,
                       char*     /*shmem*/)
    {
      Size partition_idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (partition_idx < num_partitions)
      {
        Size partition_at = (thrust::min)(partition_idx * items_per_tile,
                                        keys1_count + keys2_count);
        Size partition_diag = merge_path(keys1,
                                         keys2,
                                         keys1_count,
                                         keys2_count,
                                         partition_at,
                                         compare_op);
        merge_partitions[partition_idx] = partition_diag;
      }
    }
  };    // struct PartitionAgent


  template <class Arch, class TSize>
  struct Tuning;

  namespace mpl = thrust::detail::mpl::math;

  template<int NOMINAL_4B_ITEMS_PER_THREAD, size_t INPUT_SIZE>
  struct items_per_thread
  {
    enum
    {
      ITEMS_PER_THREAD =
          mpl::min<
              int,
              NOMINAL_4B_ITEMS_PER_THREAD,
              mpl::max<
                  int,
                  1,
                  static_cast<int>(NOMINAL_4B_ITEMS_PER_THREAD * 4 / INPUT_SIZE)>::value>::value,
      value = mpl::is_odd<int, ITEMS_PER_THREAD>::value
                  ? ITEMS_PER_THREAD
                  : ITEMS_PER_THREAD + 1
    };
  };

  template<class TSize>
  struct Tuning<sm30,TSize>
  {
    const static int INPUT_SIZE = TSize::value;
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 7,
      ITEMS_PER_THREAD            = items_per_thread<NOMINAL_4B_ITEMS_PER_THREAD,
                                          INPUT_SIZE>::value
    };

    typedef PtxPolicy<128,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_DEFAULT,
                      cub::BLOCK_STORE_WARP_TRANSPOSE>
        type;
  };    // Tuning sm300



  template<class TSize>
  struct Tuning<sm60,TSize> : Tuning<sm30,TSize>
  {
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 15,
      ITEMS_PER_THREAD            = items_per_thread<NOMINAL_4B_ITEMS_PER_THREAD,
                                          Tuning::INPUT_SIZE>::value
    };


    typedef PtxPolicy<512,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_DEFAULT,
                      cub::BLOCK_STORE_WARP_TRANSPOSE>
        type;
  };    // Tuning sm52

  template<class TSize>
  struct Tuning<sm52,TSize> : Tuning<sm30,TSize>
  {
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 13,
      ITEMS_PER_THREAD            = items_per_thread<NOMINAL_4B_ITEMS_PER_THREAD,
                                          Tuning::INPUT_SIZE>::value
    };

    typedef PtxPolicy<512,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_LDG,
                      cub::BLOCK_STORE_WARP_TRANSPOSE>
        type;
  };    // Tuning sm52

  template<class TSize>
  struct Tuning<sm35,TSize> : Tuning<sm30,TSize>
  {
    const static int INPUT_SIZE = TSize::value;
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 11,
      ITEMS_PER_THREAD            = items_per_thread<NOMINAL_4B_ITEMS_PER_THREAD,
                                          Tuning::INPUT_SIZE>::value
    };


    typedef PtxPolicy<256,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_LDG,
                      cub::BLOCK_STORE_WARP_TRANSPOSE>
        type;
  };    // Tuning sm350


  template<size_t VALUE>
  struct integer_constant : thrust::detail::integral_constant<size_t, VALUE> {};

  template <class KeysIt1,
            class KeysIt2,
            class ItemsIt1,
            class ItemsIt2,
            class Size,
            class KeysOutputIt,
            class ItemsOutputIt,
            class CompareOp,
            class MERGE_ITEMS>
  struct MergeAgent
  {
    typedef typename iterator_traits<KeysIt1>::value_type  key1_type;
    typedef typename iterator_traits<KeysIt2>::value_type  key2_type;
    typedef typename iterator_traits<ItemsIt1>::value_type item1_type;
    typedef typename iterator_traits<ItemsIt2>::value_type item2_type;

    typedef key1_type  key_type;
    typedef item1_type item_type;

    typedef typename thrust::detail::conditional<
        MERGE_ITEMS::value,
        integer_constant<sizeof(key_type) + sizeof(item_type)>,
        integer_constant<sizeof(key_type)> >::type tuning_type;


    template <class Arch>
    struct PtxPlan : Tuning<Arch, tuning_type>::type
    {
      typedef Tuning<Arch,tuning_type> tuning;

      typedef typename core::LoadIterator<PtxPlan, KeysIt1>::type  KeysLoadIt1;
      typedef typename core::LoadIterator<PtxPlan, KeysIt2>::type  KeysLoadIt2;
      typedef typename core::LoadIterator<PtxPlan, ItemsIt1>::type ItemsLoadIt1;
      typedef typename core::LoadIterator<PtxPlan, ItemsIt2>::type ItemsLoadIt2;

      typedef typename core::BlockLoad<PtxPlan, KeysLoadIt1>::type  BlockLoadKeys1;
      typedef typename core::BlockLoad<PtxPlan, KeysLoadIt2>::type  BlockLoadKeys2;
      typedef typename core::BlockLoad<PtxPlan, ItemsLoadIt1>::type BlockLoadItems1;
      typedef typename core::BlockLoad<PtxPlan, ItemsLoadIt2>::type BlockLoadItems2;

      typedef typename core::BlockStore<PtxPlan,
                                        KeysOutputIt,
                                        key_type>::type BlockStoreKeys;
      typedef typename core::BlockStore<PtxPlan,
                                        ItemsOutputIt,
                                        item_type>::type BlockStoreItems;

      // gather required temporary storage in a union
      //
      union TempStorage
      {
        typename BlockLoadKeys1::TempStorage  load_keys1;
        typename BlockLoadKeys2::TempStorage  load_keys2;
        typename BlockLoadItems1::TempStorage load_items1;
        typename BlockLoadItems2::TempStorage load_items2;
        typename BlockStoreKeys::TempStorage  store_keys;
        typename BlockStoreItems::TempStorage store_items;

        core::uninitialized_array<item_type, PtxPlan::ITEMS_PER_TILE + 1> items_shared;
        core::uninitialized_array<key_type, PtxPlan::ITEMS_PER_TILE + 1>  keys_shared;
      };    // union TempStorage
    };    // struct PtxPlan

    typedef typename core::specialize_plan_msvc10_war<PtxPlan>::type::type ptx_plan;

    typedef typename ptx_plan::KeysLoadIt1     KeysLoadIt1;
    typedef typename ptx_plan::KeysLoadIt2     KeysLoadIt2;
    typedef typename ptx_plan::ItemsLoadIt1    ItemsLoadIt1;
    typedef typename ptx_plan::ItemsLoadIt2    ItemsLoadIt2;
    typedef typename ptx_plan::BlockLoadKeys1  BlockLoadKeys1;
    typedef typename ptx_plan::BlockLoadKeys2  BlockLoadKeys2;
    typedef typename ptx_plan::BlockLoadItems1 BlockLoadItems1;
    typedef typename ptx_plan::BlockLoadItems2 BlockLoadItems2;
    typedef typename ptx_plan::BlockStoreKeys  BlockStoreKeys;
    typedef typename ptx_plan::BlockStoreItems BlockStoreItems;
    typedef typename ptx_plan::TempStorage     TempStorage;

    enum
    {
      ITEMS_PER_THREAD = ptx_plan::ITEMS_PER_THREAD,
      BLOCK_THREADS    = ptx_plan::BLOCK_THREADS,
      ITEMS_PER_TILE   = ptx_plan::ITEMS_PER_TILE
    };

    struct impl
    {
      //---------------------------------------------------------------------
      // Per thread data
      //---------------------------------------------------------------------

      TempStorage&  storage;
      KeysLoadIt1   keys1_in;
      KeysLoadIt2   keys2_in;
      ItemsLoadIt1  items1_in;
      ItemsLoadIt2  items2_in;
      Size          keys1_count;
      Size          keys2_count;
      KeysOutputIt  keys_out;
      ItemsOutputIt items_out;
      CompareOp     compare_op;
      Size*         merge_partitions;

      //---------------------------------------------------------------------
      // Utility functions
      //---------------------------------------------------------------------

      template <bool IS_FULL_TILE, class T, class It1, class It2>
      THRUST_DEVICE_FUNCTION void
      gmem_to_reg(T (&output)[ITEMS_PER_THREAD],
                  It1 input1,
                  It2 input2,
                  int count1,
                  int count2)
      {
        if (IS_FULL_TILE)
        {
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            int idx = BLOCK_THREADS * ITEM + threadIdx.x;
            if (idx < count1)
              output[ITEM] = input1[idx];
            else
              output[ITEM] = input2[idx - count1];
          }
        }
        else
        {
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            int idx = BLOCK_THREADS * ITEM + threadIdx.x;
            if (idx < count1 + count2)
            {
              if (idx < count1)
                output[ITEM] = input1[idx];
              else
                output[ITEM] = input2[idx - count1];
            }
          }
        }
      }

      template <class T, class It>
      THRUST_DEVICE_FUNCTION void
      reg_to_shared(It output,
                    T (&input)[ITEMS_PER_THREAD])
      {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int idx = BLOCK_THREADS * ITEM + threadIdx.x;
          output[idx] = input[ITEM];
        }
      }

      //---------------------------------------------------------------------
      // Tile processing
      //---------------------------------------------------------------------

      template <bool IS_FULL_TILE>
      void THRUST_DEVICE_FUNCTION
      consume_tile(Size tile_idx,
                   Size tile_base,
                   int  num_remaining)
      {
        using core::sync_threadblock;
        using core::uninitialized_array;

        Size partition_beg = merge_partitions[tile_idx + 0];
        Size partition_end = merge_partitions[tile_idx + 1];

        Size diag0 = ITEMS_PER_TILE * tile_idx;
        Size diag1 = (thrust::min)(keys1_count + keys2_count, diag0 + ITEMS_PER_TILE);

        // compute bounding box for keys1 & keys2
        //
        Size keys1_beg = partition_beg;
        Size keys1_end = partition_end;
        Size keys2_beg = diag0 - keys1_beg;
        Size keys2_end = diag1 - keys1_end;

        // number of keys per tile
        //
        int num_keys1 = static_cast<int>(keys1_end - keys1_beg);
        int num_keys2 = static_cast<int>(keys2_end - keys2_beg);

        key_type keys_loc[ITEMS_PER_THREAD];
        gmem_to_reg<IS_FULL_TILE>(keys_loc,
                                  keys1_in + keys1_beg,
                                  keys2_in + keys2_beg,
                                  num_keys1,
                                  num_keys2);
        reg_to_shared(&storage.keys_shared[0], keys_loc);

        sync_threadblock();

        // use binary search in shared memory
        // to find merge path for each of thread
        // we can use int type here, because the number of
        // items in shared memory is limited
        //
        int diag0_loc = min<int>(num_keys1 + num_keys2,
                                 ITEMS_PER_THREAD * threadIdx.x);

        int keys1_beg_loc = merge_path(&storage.keys_shared[0],
                                       &storage.keys_shared[num_keys1],
                                       num_keys1,
                                       num_keys2,
                                       diag0_loc,
                                       compare_op);
        int keys1_end_loc = num_keys1;
        int keys2_beg_loc = diag0_loc - keys1_beg_loc;
        int keys2_end_loc = num_keys2;

        int num_keys1_loc = keys1_end_loc - keys1_beg_loc;
        int num_keys2_loc = keys2_end_loc - keys2_beg_loc;

        // perform serial merge
        //
        int indices[ITEMS_PER_THREAD];

        serial_merge(&storage.keys_shared[0],
                     keys1_beg_loc,
                     keys2_beg_loc + num_keys1,
                     num_keys1_loc,
                     num_keys2_loc,
                     keys_loc,
                     indices,
                     compare_op);

        sync_threadblock();

        // write keys
        //
        if (IS_FULL_TILE)
        {
          BlockStoreKeys(storage.store_keys)
              .Store(keys_out + tile_base, keys_loc);
        }
        else
        {
          BlockStoreKeys(storage.store_keys)
              .Store(keys_out + tile_base, keys_loc, num_remaining);
        }

        // if items are provided, merge them
        if (MERGE_ITEMS::value)
        {
          item_type items_loc[ITEMS_PER_THREAD];
          gmem_to_reg<IS_FULL_TILE>(items_loc,
                                    items1_in + keys1_beg,
                                    items2_in + keys2_beg,
                                    num_keys1,
                                    num_keys2);

          sync_threadblock();

          reg_to_shared(&storage.items_shared[0], items_loc);

          sync_threadblock();

          // gather items from shared mem
          //
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            items_loc[ITEM] = storage.items_shared[indices[ITEM]];
          }

          sync_threadblock();

          // write form reg to gmem
          //
          if (IS_FULL_TILE)
          {
            BlockStoreItems(storage.store_items)
                .Store(items_out + tile_base, items_loc);
          }
          else
          {
            BlockStoreItems(storage.store_items)
                .Store(items_out + tile_base, items_loc, num_remaining);
          }
        }
      }

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION
      impl(TempStorage&  storage_,
           KeysLoadIt1   keys1_in_,
           KeysLoadIt2   keys2_in_,
           ItemsLoadIt1  items1_in_,
           ItemsLoadIt2  items2_in_,
           Size          keys1_count_,
           Size          keys2_count_,
           KeysOutputIt  keys_out_,
           ItemsOutputIt items_out_,
           CompareOp     compare_op_,
           Size*         merge_partitions_)
          : storage(storage_),
            keys1_in(keys1_in_),
            keys2_in(keys2_in_),
            items1_in(items1_in_),
            items2_in(items2_in_),
            keys1_count(keys1_count_),
            keys2_count(keys2_count_),
            keys_out(keys_out_),
            items_out(items_out_),
            compare_op(compare_op_),
            merge_partitions(merge_partitions_)
      {
        // XXX with 8.5 chaging type to Size (or long long) results in error!
        int  tile_idx      = blockIdx.x;
        Size  tile_base     = tile_idx * ITEMS_PER_TILE;
        int  items_in_tile = static_cast<int>(
            min<Size>(ITEMS_PER_TILE,
                      keys1_count + keys2_count - tile_base));
        if (items_in_tile == ITEMS_PER_TILE)
        {
          // full tile
          consume_tile<true>(tile_idx,
                             tile_base,
                             ITEMS_PER_TILE);
        }
        else
        {
          // partial tile
          consume_tile<false>(tile_idx,
                              tile_base,
                              items_in_tile);
        }
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    THRUST_AGENT_ENTRY(KeysIt1       keys1_in,
                       KeysIt2       keys2_in,
                       ItemsIt1      items1_in,
                       ItemsIt2      items2_in,
                       Size          keys1_count,
                       Size          keys2_count,
                       KeysOutputIt  keys_out,
                       ItemsOutputIt items_out,
                       CompareOp     compare_op,
                       Size*         merge_partitions,
                       char*         shmem)
    {
      TempStorage& storage = *reinterpret_cast<TempStorage*>(shmem);

      impl(storage,
           core::make_load_iterator(ptx_plan(), keys1_in),
           core::make_load_iterator(ptx_plan(), keys2_in),
           core::make_load_iterator(ptx_plan(), items1_in),
           core::make_load_iterator(ptx_plan(), items2_in),
           keys1_count,
           keys2_count,
           keys_out,
           items_out,
           compare_op,
           merge_partitions);
    }
  };    // struct MergeAgent;

  //---------------------------------------------------------------------
  // Two-step internal API
  //---------------------------------------------------------------------

  template <class MERGE_ITEMS,
            class KeysIt1,
            class KeysIt2,
            class ItemsIt1,
            class ItemsIt2,
            class Size,
            class KeysOutputIt,
            class ItemsOutputIt,
            class CompareOp>
  cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void*         d_temp_storage,
            size_t&       temp_storage_bytes,
            KeysIt1       keys1,
            KeysIt2       keys2,
            ItemsIt1      items1,
            ItemsIt2      items2,
            Size          num_keys1,
            Size          num_keys2,
            KeysOutputIt  keys_result,
            ItemsOutputIt items_result,
            CompareOp     compare_op,
            cudaStream_t  stream,
            bool          debug_sync)
  {
    if (num_keys1 + num_keys2 == 0)
      return cudaErrorNotSupported;

    using core::AgentPlan;
    using core::get_agent_plan;
    typedef core::AgentLauncher<
        MergeAgent<KeysIt1,
                   KeysIt2,
                   ItemsIt1,
                   ItemsIt2,
                   Size,
                   KeysOutputIt,
                   ItemsOutputIt,
                   CompareOp,
                   MERGE_ITEMS> >
        merge_agent;

    typedef core::AgentLauncher<
        PartitionAgent<KeysIt1,
                       KeysIt2,
                       Size,
                       CompareOp> >
        partition_agent;

    cudaError_t status = cudaSuccess;

    AgentPlan partition_plan = partition_agent::get_plan();
    AgentPlan merge_plan     = merge_agent::get_plan(stream);

    int  tile_size = merge_plan.items_per_tile;
    Size num_tiles = (num_keys1 + num_keys2 + tile_size - 1) / tile_size;

    size_t temp_storage1 = (1 + num_tiles) * sizeof(Size);
    size_t temp_storage2 = core::vshmem_size(merge_plan.shared_memory_size,
                                             num_tiles);

    void*  allocations[2]      = {NULL, NULL};
    size_t allocation_sizes[2] = {temp_storage1, temp_storage2};

    status = core::alias_storage(d_temp_storage,
                                 temp_storage_bytes,
                                 allocations,
                                 allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    if (d_temp_storage == NULL)
    {
      return status;
    }

    // partition data into work balanced tiles
    Size* merge_partitions = (Size*)allocations[0];
    char* vshmem_ptr       = temp_storage2 > 0 ? (char*)allocations[1] : NULL;

    {
      Size num_partitions = num_tiles + 1;

      partition_agent(partition_plan, num_partitions, stream, "partition agent", debug_sync)
          .launch(keys1,
                  keys2,
                  num_keys1,
                  num_keys2,
                  num_partitions,
                  merge_partitions,
                  compare_op,
                  merge_plan.items_per_tile);
      CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
    }

    merge_agent(merge_plan, num_keys1 + num_keys2, stream, vshmem_ptr, "merge agent", debug_sync)
        .launch(keys1,
                keys2,
                items1,
                items2,
                num_keys1,
                num_keys2,
                keys_result,
                items_result,
                compare_op,
                merge_partitions);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    return status;
  }

  template <typename MERGE_ITEMS,
            typename Derived,
            typename KeysIt1,
            typename KeysIt2,
            typename ItemsIt1,
            typename ItemsIt2,
            typename KeysOutputIt,
            typename ItemsOutputIt,
            typename CompareOp>
  THRUST_RUNTIME_FUNCTION
  pair<KeysOutputIt, ItemsOutputIt>
  merge(execution_policy<Derived>& policy,
        KeysIt1                    keys1_first,
        KeysIt1                    keys1_last,
        KeysIt2                    keys2_first,
        KeysIt2                    keys2_last,
        ItemsIt1                   items1_first,
        ItemsIt2                   items2_first,
        KeysOutputIt               keys_result,
        ItemsOutputIt              items_result,
        CompareOp                  compare_op)
  {
    typedef typename iterator_traits<KeysIt1>::difference_type size_type;

    size_type num_keys1
      = static_cast<size_type>(thrust::distance(keys1_first, keys1_last));
    size_type num_keys2
      = static_cast<size_type>(thrust::distance(keys2_first, keys2_last));

    size_type const count = num_keys1 + num_keys2;

    if (count == 0)
      return thrust::make_pair(keys_result, items_result);

    size_t       storage_size = 0;
    cudaStream_t stream       = cuda_cub::stream(policy);
    bool         debug_sync   = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    status = doit_step<MERGE_ITEMS>(NULL,
                                    storage_size,
                                    keys1_first,
                                    keys2_first,
                                    items1_first,
                                    items2_first,
                                    num_keys1,
                                    num_keys2,
                                    keys_result,
                                    items_result,
                                    compare_op,
                                    stream,
                                    debug_sync);
    cuda_cub::throw_on_error(status, "merge: failed on 1st step");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = doit_step<MERGE_ITEMS>(ptr,
                                    storage_size,
                                    keys1_first,
                                    keys2_first,
                                    items1_first,
                                    items2_first,
                                    num_keys1,
                                    num_keys2,
                                    keys_result,
                                    items_result,
                                    compare_op,
                                    stream,
                                    debug_sync);
    cuda_cub::throw_on_error(status, "merge: failed on 2nd step");

    status = cuda_cub::synchronize_optional(policy);
    cuda_cub::throw_on_error(status, "merge: failed to synchronize");

    return thrust::make_pair(keys_result + count, items_result + count);
  }
}    // namespace __merge


//-------------------------
// Thrust API entry points
//-------------------------


__thrust_exec_check_disable__
template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ResultIt,
          class CompareOp>
ResultIt __host__ __device__
merge(execution_policy<Derived>& policy,
      KeysIt1                    keys1_first,
      KeysIt1                    keys1_last,
      KeysIt2                    keys2_first,
      KeysIt2                    keys2_last,
      ResultIt                   result,
      CompareOp                  compare_op)

{
  ResultIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    typedef typename thrust::iterator_value<KeysIt1>::type keys_type;
    //
    keys_type* null_ = NULL;
    //
    ret = __merge::merge<thrust::detail::false_type>(policy,
                                                     keys1_first,
                                                     keys1_last,
                                                     keys2_first,
                                                     keys2_last,
                                                     null_,
                                                     null_,
                                                     result,
                                                     null_,
                                                     compare_op)
              .first;
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::merge(cvt_to_seq(derived_cast(policy)),
                        keys1_first,
                        keys1_last,
                        keys2_first,
                        keys2_last,
                        result,
                        compare_op);
#endif
  }
  return ret;
}

template <class Derived, class KeysIt1, class KeysIt2, class ResultIt>
ResultIt __host__ __device__
merge(execution_policy<Derived>& policy,
      KeysIt1                    keys1_first,
      KeysIt1                    keys1_last,
      KeysIt2                    keys2_first,
      KeysIt2                    keys2_last,
      ResultIt                   result)
{
  typedef typename thrust::iterator_value<KeysIt1>::type keys_type;
  return cuda_cub::merge(policy,
                         keys1_first,
                         keys1_last,
                         keys2_first,
                         keys2_last,
                         result,
                         less<keys_type>());
}

__thrust_exec_check_disable__
template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp>
pair<KeysOutputIt, ItemsOutputIt> __host__ __device__
merge_by_key(execution_policy<Derived> &policy,
             KeysIt1                    keys1_first,
             KeysIt1                    keys1_last,
             KeysIt2                    keys2_first,
             KeysIt2                    keys2_last,
             ItemsIt1                   items1_first,
             ItemsIt2                   items2_first,
             KeysOutputIt               keys_result,
             ItemsOutputIt              items_result,
             CompareOp                  compare_op)
{
  pair<KeysOutputIt, ItemsOutputIt> ret = thrust::make_pair(keys_result, items_result);
  if (__THRUST_HAS_CUDART__)
  {
    return __merge::merge<thrust::detail::true_type>(policy,
                                                     keys1_first,
                                                     keys1_last,
                                                     keys2_first,
                                                     keys2_last,
                                                     items1_first,
                                                     items2_first,
                                                     keys_result,
                                                     items_result,
                                                     compare_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::merge_by_key(cvt_to_seq(derived_cast(policy)),
                               keys1_first,
                               keys1_last,
                               keys2_first,
                               keys2_last,
                               items1_first,
                               items2_first,
                               keys_result,
                               items_result,
                               compare_op);
#endif
  }
  return ret;
}

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt>
pair<KeysOutputIt, ItemsOutputIt> __host__ __device__
merge_by_key(execution_policy<Derived> &policy,
             KeysIt1                    keys1_first,
             KeysIt1                    keys1_last,
             KeysIt2                    keys2_first,
             KeysIt2                    keys2_last,
             ItemsIt1                   items1_first,
             ItemsIt2                   items2_first,
             KeysOutputIt               keys_result,
             ItemsOutputIt              items_result)
{
  typedef typename thrust::iterator_value<KeysIt1>::type keys_type;
  return cuda_cub::merge_by_key(policy,
                                keys1_first,
                                keys1_last,
                                keys2_first,
                                keys2_last,
                                items1_first,
                                items2_first,
                                keys_result,
                                items_result,
                                thrust::less<keys_type>());
}


}    // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
