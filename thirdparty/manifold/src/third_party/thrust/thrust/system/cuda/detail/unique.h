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
#include <thrust/system/cuda/config.h>

#include <cub/device/device_select.cuh>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/get_value.h>
#include <thrust/functional.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>
#include <thrust/advance.h>
#include <thrust/distance.h>

#include <cub/util_math.cuh>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy,
          typename ForwardIterator,
          typename BinaryPredicate>
__host__ __device__ ForwardIterator
unique(
    const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    ForwardIterator                                             first,
    ForwardIterator                                             last,
    BinaryPredicate                                             binary_pred);

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryPredicate>
__host__ __device__ OutputIterator
unique_copy(
    const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    InputIterator                                               first,
    InputIterator                                               last,
    OutputIterator                                              result,
    BinaryPredicate                                             binary_pred);

template <typename DerivedPolicy,
          typename ForwardIterator,
          typename BinaryPredicate>
__host__ __device__ typename thrust::iterator_traits<ForwardIterator>::difference_type
unique_count(
    const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    ForwardIterator                                             first,
    ForwardIterator                                             last,
    BinaryPredicate                                             binary_pred);

namespace cuda_cub {

// XXX  it should be possible to unify unique & unique_by_key into a single
//      agent with various specializations, similar to what is done
//      with partition
namespace __unique {

  template <int                     _BLOCK_THREADS,
            int                     _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier  _LOAD_MODIFIER    = cub::LOAD_LDG,
            cub::BlockScanAlgorithm _SCAN_ALGORITHM   = cub::BLOCK_SCAN_WARP_SCANS>
  struct PtxPolicy
  {
    enum
    {
      BLOCK_THREADS    = _BLOCK_THREADS,
      ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
      ITEMS_PER_TILE   = _BLOCK_THREADS * _ITEMS_PER_THREAD,
    };
    static const cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
    static const cub::CacheLoadModifier  LOAD_MODIFIER  = _LOAD_MODIFIER;
    static const cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
  };    // struct PtxPolicy

  template<class,class>
  struct Tuning;

  namespace mpl = thrust::detail::mpl::math;

  template<class T, int NOMINAL_4B_ITEMS_PER_THREAD>
  struct items_per_thread
  {
    enum
    {
      value = mpl::min<
          int,
          NOMINAL_4B_ITEMS_PER_THREAD,
          mpl::max<int,
                   1,
                   static_cast<int>(NOMINAL_4B_ITEMS_PER_THREAD * 4 /
                    sizeof(T))>::value>::value
    };
  };

  template<class T>
  struct Tuning<sm52,T>
  {
    const static int INPUT_SIZE = sizeof(T);
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 11,
      //
      ITEMS_PER_THREAD = items_per_thread<T,
                                          NOMINAL_4B_ITEMS_PER_THREAD>::value
    };

    typedef PtxPolicy<64,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_LDG,
                      cub::BLOCK_SCAN_WARP_SCANS>
        type;
  };    // Tuning for sm52


  template <class T>
  struct Tuning<sm35, T>
  {
    const static int INPUT_SIZE = sizeof(T);
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 9,
      //
      ITEMS_PER_THREAD = items_per_thread<T,
                                          NOMINAL_4B_ITEMS_PER_THREAD>::value
    };

    typedef PtxPolicy<128,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_LDG,
                      cub::BLOCK_SCAN_WARP_SCANS>
        type;
  };    // Tuning for sm35

  template<class T>
  struct Tuning<sm30,T>
  {
    const static int INPUT_SIZE = sizeof(T);
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 7,
      //
      ITEMS_PER_THREAD = items_per_thread<T,
                                          NOMINAL_4B_ITEMS_PER_THREAD>::value
    };

    typedef PtxPolicy<128,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_DEFAULT,
                      cub::BLOCK_SCAN_WARP_SCANS>
        type;
  };    // Tuning for sm30

  template <class ItemsIt,
            class ItemsOutputIt,
            class BinaryPred,
            class Size,
            class NumSelectedOutIt>
  struct UniqueAgent
  {
    typedef typename iterator_traits<ItemsIt>::value_type item_type;

    typedef cub::ScanTileState<Size> ScanTileState;

    template <class Arch>
    struct PtxPlan : Tuning<Arch, item_type>::type
    {
      typedef Tuning<Arch, item_type> tuning;

      typedef typename core::LoadIterator<PtxPlan, ItemsIt>::type ItemsLoadIt;

      typedef typename core::BlockLoad<PtxPlan, ItemsLoadIt>::type BlockLoadItems;

      typedef cub::BlockDiscontinuity<item_type,
                                      PtxPlan::BLOCK_THREADS,
                                      1,
                                      1,
                                      Arch::ver>
          BlockDiscontinuityItems;

      typedef cub::TilePrefixCallbackOp<Size,
                                        cub::Sum,
                                        ScanTileState,
                                        Arch::ver>
          TilePrefixCallback;
      typedef cub::BlockScan<Size,
                             PtxPlan::BLOCK_THREADS,
                             PtxPlan::SCAN_ALGORITHM,
                             1,
                             1,
                             Arch::ver>
          BlockScan;

      typedef core::uninitialized_array<item_type, PtxPlan::ITEMS_PER_TILE>
          shared_items_t;

      union TempStorage
      {
        struct ScanStorage
        {
          typename BlockScan::TempStorage               scan;
          typename TilePrefixCallback::TempStorage      prefix;
          typename BlockDiscontinuityItems::TempStorage discontinuity;
        } scan_storage;

        typename BlockLoadItems::TempStorage  load_items;
        shared_items_t shared_items;

      };    // union TempStorage
    };      // struct PtxPlan

    typedef typename core::specialize_plan_msvc10_war<PtxPlan>::type::type ptx_plan;

    typedef typename ptx_plan::ItemsLoadIt             ItemsLoadIt;
    typedef typename ptx_plan::BlockLoadItems          BlockLoadItems;
    typedef typename ptx_plan::BlockDiscontinuityItems BlockDiscontinuityItems;
    typedef typename ptx_plan::TilePrefixCallback      TilePrefixCallback;
    typedef typename ptx_plan::BlockScan               BlockScan;
    typedef typename ptx_plan::shared_items_t          shared_items_t;
    typedef typename ptx_plan::TempStorage             TempStorage;

    enum
    {
      BLOCK_THREADS    = ptx_plan::BLOCK_THREADS,
      ITEMS_PER_THREAD = ptx_plan::ITEMS_PER_THREAD,
      ITEMS_PER_TILE   = ptx_plan::ITEMS_PER_TILE
    };

    struct impl
    {
      //---------------------------------------------------------------------
      // Per-thread fields
      //---------------------------------------------------------------------

      TempStorage &                      temp_storage;
      ScanTileState &                    tile_state;
      ItemsLoadIt                        items_in;
      ItemsOutputIt                      items_out;
      cub::InequalityWrapper<BinaryPred> predicate;
      Size                               num_items;

      //---------------------------------------------------------------------
      // Utility functions
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION
      shared_items_t &get_shared()
      {
        return temp_storage.shared_items;
      }

      void THRUST_DEVICE_FUNCTION
      scatter(item_type (&items)[ITEMS_PER_THREAD],
              Size (&selection_flags)[ITEMS_PER_THREAD],
              Size (&selection_indices)[ITEMS_PER_THREAD],
              int  /*num_tile_items*/,
              int  num_tile_selections,
              Size num_selections_prefix,
              Size /*num_selections*/)
      {
        using core::sync_threadblock;

#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int local_scatter_offset = selection_indices[ITEM] -
                                     num_selections_prefix;
          if (selection_flags[ITEM])
          {
            get_shared()[local_scatter_offset] = items[ITEM];
          }
        }

        sync_threadblock();

        for (int item = threadIdx.x;
             item < num_tile_selections;
             item += BLOCK_THREADS)
        {
          items_out[num_selections_prefix + item] = get_shared()[item];
        }

        sync_threadblock();
      }

      //---------------------------------------------------------------------
      // Tile processing
      //---------------------------------------------------------------------

      template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
      Size THRUST_DEVICE_FUNCTION
      consume_tile_impl(int  num_tile_items,
                        int  tile_idx,
                        Size tile_base)
      {
        using core::sync_threadblock;
        using core::uninitialized_array;

        item_type items_loc[ITEMS_PER_THREAD];
        Size      selection_flags[ITEMS_PER_THREAD];
        Size      selection_idx[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
          BlockLoadItems(temp_storage.load_items)
              .Load(items_in + tile_base,
                    items_loc,
                    num_tile_items,
                    *(items_in + tile_base));
        }
        else
        {
          BlockLoadItems(temp_storage.load_items)
              .Load(items_in + tile_base, items_loc);
        }


        sync_threadblock();

        if (IS_FIRST_TILE)
        {
          BlockDiscontinuityItems(temp_storage.scan_storage.discontinuity)
              .FlagHeads(selection_flags, items_loc, predicate);
        }
        else
        {
          item_type tile_predecessor = items_in[tile_base - 1];
          BlockDiscontinuityItems(temp_storage.scan_storage.discontinuity)
              .FlagHeads(selection_flags, items_loc, predicate, tile_predecessor);
        }

#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          // Set selection_flags for out-of-bounds items
          if ((IS_LAST_TILE) &&
              (Size(threadIdx.x * ITEMS_PER_THREAD) + ITEM >= num_tile_items))
            selection_flags[ITEM] = 1;
        }

        sync_threadblock();

        Size num_tile_selections   = 0;
        Size num_selections        = 0;
        Size num_selections_prefix = 0;
        if (IS_FIRST_TILE)
        {
          BlockScan(temp_storage.scan_storage.scan)
              .ExclusiveSum(selection_flags,
                            selection_idx,
                            num_tile_selections);

          if (threadIdx.x == 0)
          {
            // Update tile status if this is not the last tile
            if (!IS_LAST_TILE)
              tile_state.SetInclusive(0, num_tile_selections);
          }

          // Do not count any out-of-bounds selections
          if (IS_LAST_TILE)
          {
            int num_discount = ITEMS_PER_TILE - num_tile_items;
            num_tile_selections -= num_discount;
          }
          num_selections = num_tile_selections;
        }
        else
        {
          TilePrefixCallback prefix_cb(tile_state,
                                       temp_storage.scan_storage.prefix,
                                       cub::Sum(),
                                       tile_idx);
          BlockScan(temp_storage.scan_storage.scan)
              .ExclusiveSum(selection_flags,
                            selection_idx,
                            prefix_cb);

          num_selections        = prefix_cb.GetInclusivePrefix();
          num_tile_selections   = prefix_cb.GetBlockAggregate();
          num_selections_prefix = prefix_cb.GetExclusivePrefix();

          if (IS_LAST_TILE)
          {
            int num_discount = ITEMS_PER_TILE - num_tile_items;
            num_tile_selections -= num_discount;
            num_selections -= num_discount;
          }
        }

        sync_threadblock();

        scatter(items_loc,
                selection_flags,
                selection_idx,
                num_tile_items,
                num_tile_selections,
                num_selections_prefix,
                num_selections);

        return num_selections;
      }


      template <bool IS_LAST_TILE>
      Size THRUST_DEVICE_FUNCTION
      consume_tile(int  num_tile_items,
                   int  tile_idx,
                   Size tile_base)
      {
        if (tile_idx == 0)
        {
          return consume_tile_impl<IS_LAST_TILE, true>(num_tile_items,
                                                       tile_idx,
                                                       tile_base);
        }
        else
        {
          return consume_tile_impl<IS_LAST_TILE, false>(num_tile_items,
                                                        tile_idx,
                                                        tile_base);
        }
      }

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION
      impl(TempStorage &    temp_storage_,
           ScanTileState &  tile_state_,
           ItemsLoadIt      items_in_,
           ItemsOutputIt    items_out_,
           BinaryPred       binary_pred_,
           Size             num_items_,
           int              num_tiles,
           NumSelectedOutIt num_selected_out)
          : temp_storage(temp_storage_),
            tile_state(tile_state_),
            items_in(items_in_),
            items_out(items_out_),
            predicate(binary_pred_),
            num_items(num_items_)
      {
        int  tile_idx  = blockIdx.x;
        Size tile_base = tile_idx * ITEMS_PER_TILE;

        if (tile_idx < num_tiles - 1)
        {
          consume_tile<false>(ITEMS_PER_TILE,
                              tile_idx,
                              tile_base);
        }
        else
        {
          int  num_remaining  = static_cast<int>(num_items - tile_base);
          Size num_selections = consume_tile<true>(num_remaining,
                                                   tile_idx,
                                                   tile_base);
          if (threadIdx.x == 0)
          {
            *num_selected_out = num_selections;
          }
        }
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    THRUST_AGENT_ENTRY(ItemsIt          items_in,
                       ItemsOutputIt    items_out,
                       BinaryPred       binary_pred,
                       NumSelectedOutIt num_selected_out,
                       Size             num_items,
                       ScanTileState    tile_state,
                       int              num_tiles,
                       char *           shmem)
    {
      TempStorage &storage = *reinterpret_cast<TempStorage *>(shmem);

      impl(storage,
           tile_state,
           core::make_load_iterator(ptx_plan(), items_in),
           items_out,
           binary_pred,
           num_items,
           num_tiles,
           num_selected_out);
    }
  };    // struct UniqueAgent

  template <class ScanTileState,
            class NumSelectedIt,
            class Size>
  struct InitAgent
  {
    template <class Arch>
    struct PtxPlan : PtxPolicy<128> {};
    typedef core::specialize_plan<PtxPlan> ptx_plan;

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    THRUST_AGENT_ENTRY(ScanTileState tile_state,
                       Size          num_tiles,
                       NumSelectedIt num_selected_out,
                       char * /*shmem*/)
    {
      tile_state.InitializeStatus(num_tiles);
      if (blockIdx.x == 0 && threadIdx.x == 0)
        *num_selected_out = 0;
    }

  }; // struct InitAgent

  template <class ItemsInputIt,
            class ItemsOutputIt,
            class BinaryPred,
            class Size,
            class NumSelectedOutIt>
  static cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *           d_temp_storage,
            size_t &         temp_storage_bytes,
            ItemsInputIt     items_in,
            ItemsOutputIt    items_out,
            BinaryPred       binary_pred,
            NumSelectedOutIt num_selected_out,
            Size             num_items,
            cudaStream_t     stream,
            bool             debug_sync)
  {
    using core::AgentLauncher;
    using core::AgentPlan;
    using core::get_agent_plan;

    typedef AgentLauncher<
        UniqueAgent<ItemsInputIt,
                    ItemsOutputIt,
                    BinaryPred,
                    Size,
                    NumSelectedOutIt> >
        unique_agent;

    typedef typename unique_agent::ScanTileState ScanTileState;

    typedef AgentLauncher<
        InitAgent<ScanTileState, NumSelectedOutIt, Size> >
        init_agent;

    using core::get_plan;
    typename get_plan<init_agent>::type   init_plan   = init_agent::get_plan();
    typename get_plan<unique_agent>::type unique_plan = unique_agent::get_plan(stream);


    int tile_size = unique_plan.items_per_tile;
    size_t num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

    size_t vshmem_size = core::vshmem_size(unique_plan.shared_memory_size,
                                           num_tiles);

    cudaError_t status = cudaSuccess;
    size_t      allocation_sizes[2] = {0, vshmem_size};
    status = ScanTileState::AllocationSize(static_cast<int>(num_tiles), allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    void *allocations[2] = {NULL, NULL};
    //
    status = cub::AliasTemporaries(d_temp_storage,
                                   temp_storage_bytes,
                                   allocations,
                                   allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    if (d_temp_storage == NULL)
    {
      return status;
    }

    ScanTileState tile_status;
    status =  tile_status.Init(static_cast<int>(num_tiles), allocations[0], allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    num_tiles = max<size_t>(1,num_tiles);
    init_agent ia(init_plan, num_tiles, stream, "unique_by_key::init_agent", debug_sync);
    ia.launch(tile_status, num_tiles, num_selected_out);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    if (num_items == 0) { return status; }

    char *vshmem_ptr = vshmem_size > 0 ? (char *)allocations[1] : NULL;

    unique_agent ua(unique_plan, num_items, stream, vshmem_ptr, "unique_by_key::unique_agent", debug_sync);
    ua.launch(items_in,
              items_out,
              binary_pred,
              num_selected_out,
              num_items,
              tile_status,
              num_tiles);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
    return status;
  }

  template <typename Derived,
            typename ItemsInputIt,
            typename ItemsOutputIt,
            typename BinaryPred>
  THRUST_RUNTIME_FUNCTION
  ItemsOutputIt unique(execution_policy<Derived>& policy,
                       ItemsInputIt               items_first,
                       ItemsInputIt               items_last,
                       ItemsOutputIt              items_result,
                       BinaryPred                 binary_pred)
  {
    //  typedef typename iterator_traits<ItemsInputIt>::difference_type size_type;
    typedef int size_type;

    size_type    num_items          = static_cast<size_type>(thrust::distance(items_first, items_last));
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    status = doit_step(NULL,
                       temp_storage_bytes,
                       items_first,
                       items_result,
                       binary_pred,
                       reinterpret_cast<size_type*>(NULL),
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "unique: failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(size_type), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;
    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "unique: failed on 1st step");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "unique: failed on 2nd step");

    size_type* d_num_selected_out
      = thrust::detail::aligned_reinterpret_cast<size_type*>(allocations[0]);

    status = doit_step(allocations[1],
                       temp_storage_bytes,
                       items_first,
                       items_result,
                       binary_pred,
                       d_num_selected_out,
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "unique: failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "unique: failed to synchronize");

    size_type num_selected = get_value(policy, d_num_selected_out);

    return items_result + num_selected;
  }
}    // namespace __unique

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class OutputIt,
          class BinaryPred>
OutputIt __host__ __device__
unique_copy(execution_policy<Derived> &policy,
            InputIt                    first,
            InputIt                    last,
            OutputIt                   result,
            BinaryPred                 binary_pred)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __unique::unique(policy,
                           first,
                           last,
                           result,
                           binary_pred);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::unique_copy(cvt_to_seq(derived_cast(policy)),
                              first,
                              last,
                              result,
                              binary_pred);
#endif
  }
  return ret;
}

template <class Derived,
          class InputIt,
          class OutputIt>
OutputIt __host__ __device__
unique_copy(execution_policy<Derived> &policy,
            InputIt                    first,
            InputIt                    last,
            OutputIt                   result)
{
  typedef typename iterator_traits<InputIt>::value_type input_type;
  return cuda_cub::unique_copy(policy, first, last, result, equal_to<input_type>());
}



__thrust_exec_check_disable__
template <class Derived,
          class ForwardIt,
          class BinaryPred>
ForwardIt __host__ __device__
unique(execution_policy<Derived> &policy,
       ForwardIt                  first,
       ForwardIt                  last,
       BinaryPred                 binary_pred)
{
  ForwardIt ret = first;
  if (__THRUST_HAS_CUDART__)
  {
    ret = cuda_cub::unique_copy(policy, first, last, first, binary_pred);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::unique(cvt_to_seq(derived_cast(policy)),
                         first,
                         last,
                         binary_pred);
#endif
  }
  return ret;
}

template <class Derived,
          class ForwardIt>
ForwardIt __host__ __device__
unique(execution_policy<Derived> &policy,
       ForwardIt                  first,
       ForwardIt                  last)
{
  typedef typename iterator_traits<ForwardIt>::value_type input_type;
  return cuda_cub::unique(policy, first, last, equal_to<input_type>());
}


template <typename BinaryPred>
struct zip_adj_not_predicate {
  template <typename TupleType>
  bool __host__ __device__ operator()(TupleType&& tuple) {
      return !binary_pred(thrust::get<0>(tuple), thrust::get<1>(tuple));
  }
  
  BinaryPred binary_pred;
};


__thrust_exec_check_disable__
template <class Derived,
          class ForwardIt,
          class BinaryPred>
typename thrust::iterator_traits<ForwardIt>::difference_type
__host__ __device__
unique_count(execution_policy<Derived> &policy,
       ForwardIt                  first,
       ForwardIt                  last,
       BinaryPred                 binary_pred)
{
  if (first == last) {
    return 0;
  }
  auto size = thrust::distance(first, last);
  auto it = thrust::make_zip_iterator(thrust::make_tuple(first, thrust::next(first)));
  return 1 + thrust::count_if(policy, it, thrust::next(it, size - 1), zip_adj_not_predicate<BinaryPred>{binary_pred});
}

}    // namespace cuda_cub
THRUST_NAMESPACE_END

//
#include <thrust/memory.h>
#include <thrust/unique.h>
#endif
