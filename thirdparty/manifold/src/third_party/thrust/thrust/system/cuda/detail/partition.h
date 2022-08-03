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

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/util.h>
#include <thrust/system/cuda/detail/reverse.h>
#include <thrust/system/cuda/detail/find.h>
#include <thrust/system/cuda/detail/uninitialized_copy.h>
#include <cub/device/device_partition.cuh>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/partition.h>
#include <thrust/pair.h>
#include <thrust/distance.h>

#include <cub/util_math.cuh>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

namespace __partition {

  template <int                     _BLOCK_THREADS,
            int                     _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier  _LOAD_MODIFIER    = cub::LOAD_LDG,
            cub::BlockScanAlgorithm _SCAN_ALGORITHM   = cub::BLOCK_SCAN_WARP_SCANS>
  struct PtxPolicy
  {
    enum
    {
      BLOCK_THREADS      = _BLOCK_THREADS,
      ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
      ITEMS_PER_TILE     = _BLOCK_THREADS * _ITEMS_PER_THREAD
    };
    static const cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
    static const cub::CacheLoadModifier  LOAD_MODIFIER  = _LOAD_MODIFIER;
    static const cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
  };    // struct PtxPolicy

  template<class, class>
  struct Tuning;

  template<class T>
  struct Tuning<sm35, T>
  {
    const static int INPUT_SIZE = sizeof(T);

    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 10,
      ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
    };

    typedef PtxPolicy<128,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_LDG,
                      cub::BLOCK_SCAN_WARP_SCANS>
        type;
  };    // Tuning<350>

  template<class T>
  struct Tuning<sm30, T>
  {
    const static int INPUT_SIZE = sizeof(T);

    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 7,
      ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(3, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
    };

    typedef PtxPolicy<128,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_DEFAULT,
                      cub::BLOCK_SCAN_WARP_SCANS>
        type;
  };    // Tuning<300>

  template<int T>
  struct __tag{};


  struct no_stencil_tag_    {};
  struct single_output_tag_
  {
    template<class T>
    THRUST_DEVICE_FUNCTION T const& operator=(T const& t) const { return t; }
  };

  typedef no_stencil_tag_* no_stencil_tag;
  typedef single_output_tag_* single_output_tag;;

  template <class ItemsIt,
            class StencilIt,
            class SelectedOutIt,
            class RejectedOutIt,
            class Predicate,
            class Size,
            class NumSelectedOutIt>
  struct PartitionAgent
  {
    typedef typename iterator_traits<ItemsIt>::value_type   item_type;
    typedef typename iterator_traits<StencilIt>::value_type stencil_type;


    typedef cub::ScanTileState<Size> ScanTileState;

    template <class Arch>
    struct PtxPlan : Tuning<Arch, item_type>::type
    {
      typedef Tuning<Arch,item_type> tuning;

      typedef typename core::LoadIterator<PtxPlan, ItemsIt>::type   ItemsLoadIt;
      typedef typename core::LoadIterator<PtxPlan, StencilIt>::type StencilLoadIt;

      typedef typename core::BlockLoad<PtxPlan, ItemsLoadIt>::type   BlockLoadItems;
      typedef typename core::BlockLoad<PtxPlan, StencilLoadIt>::type BlockLoadStencil;

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


      union TempStorage
      {
        struct ScanStorage
        {
          typename BlockScan::TempStorage          scan;
          typename TilePrefixCallback::TempStorage prefix;
        } scan_storage;

        typename BlockLoadItems::TempStorage   load_items;
        typename BlockLoadStencil::TempStorage load_stencil;

        core::uninitialized_array<item_type, PtxPlan::ITEMS_PER_TILE> raw_exchange;
      };    // union TempStorage
    };    // struct PtxPlan
    typedef typename core::specialize_plan_msvc10_war<PtxPlan>::type::type ptx_plan;

    typedef typename ptx_plan::ItemsLoadIt        ItemsLoadIt;
    typedef typename ptx_plan::StencilLoadIt      StencilLoadIt;
    typedef typename ptx_plan::BlockLoadItems     BlockLoadItems;
    typedef typename ptx_plan::BlockLoadStencil   BlockLoadStencil;
    typedef typename ptx_plan::TilePrefixCallback TilePrefixCallback;
    typedef typename ptx_plan::BlockScan          BlockScan;
    typedef typename ptx_plan::TempStorage        TempStorage;

    enum
    {
      SINGLE_OUTPUT    = thrust::detail::is_same<RejectedOutIt, single_output_tag>::value,
      USE_STENCIL      = !thrust::detail::is_same<StencilIt, no_stencil_tag>::value,
      BLOCK_THREADS    = ptx_plan::BLOCK_THREADS,
      ITEMS_PER_THREAD = ptx_plan::ITEMS_PER_THREAD,
      ITEMS_PER_TILE   = ptx_plan::ITEMS_PER_TILE
    };


    struct impl
    {
      //---------------------------------------------------------------------
      // Per-thread fields
      //---------------------------------------------------------------------

      TempStorage &  temp_storage;
      ScanTileState &tile_state;
      ItemsLoadIt    items_glob;
      StencilLoadIt  stencil_glob;
      SelectedOutIt  selected_out_glob;
      RejectedOutIt  rejected_out_glob;
      Predicate      predicate;
      Size           num_items;

      //---------------------------------------------------------------------
      // Utilities
      //---------------------------------------------------------------------

      template <bool IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION void
      scatter(item_type (&items)[ITEMS_PER_THREAD],
              Size (&selection_flags)[ITEMS_PER_THREAD],
              Size (&selection_indices)[ITEMS_PER_THREAD],
              int  num_tile_items,
              int  num_tile_selections,
              Size num_selections_prefix,
              Size num_rejected_prefix,
              Size /*num_selections*/)
      {
        int tile_num_rejections = num_tile_items - num_tile_selections;

        // Scatter items to shared memory (rejections first)
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int item_idx             = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
          int local_selection_idx  = selection_indices[ITEM] - num_selections_prefix;
          int local_rejection_idx  = item_idx - local_selection_idx;
          int local_scatter_offset = (selection_flags[ITEM])
                                         ? tile_num_rejections + local_selection_idx
                                         : local_rejection_idx;

          temp_storage.raw_exchange[local_scatter_offset] = items[ITEM];
        }

        core::sync_threadblock();

        // Gather items from shared memory and scatter to global
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int  item_idx       = (ITEM * BLOCK_THREADS) + threadIdx.x;
          int  rejection_idx  = item_idx;
          int  selection_idx  = item_idx - tile_num_rejections;
          Size scatter_offset = (item_idx < tile_num_rejections)
                                    ? num_items -
                                          num_rejected_prefix - rejection_idx - 1
                                    : num_selections_prefix + selection_idx;

          item_type item = temp_storage.raw_exchange[item_idx];

          if (!IS_LAST_TILE || (item_idx < num_tile_items))
          {
            if (SINGLE_OUTPUT || item_idx >= tile_num_rejections)
            {
              selected_out_glob[scatter_offset] = item;
            }
            else    // if !SINGLE_OUTPUT, scatter rejected items separately
            {
              rejected_out_glob[num_items - scatter_offset - 1] = item;
            }
          }
        }
      }    // func scatter

      //------------------------------------------
      // specialize predicate on different types
      //------------------------------------------

      enum ItemStencil
      {
        ITEM,
        STENCIL
      };

      template <bool TAG, class T>
      struct wrap_value
      {
        T const &              x;
        THRUST_DEVICE_FUNCTION wrap_value(T const &x) : x(x) {}

        THRUST_DEVICE_FUNCTION T const &operator()() const { return x; };
      };    // struct wrap_type

      //------- item

      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<ITEM, item_type> const &x,
                        __tag<false /* USE_STENCIL */>)
      {
        return predicate(x());
      }

      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<ITEM, item_type> const &,
                        __tag<true>)
      {
        return false;
      }

      //-------- stencil

      template <class T>
      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<STENCIL, T> const &x,
                        __tag<true>)
      {
        return predicate(x());
      }

      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<STENCIL, no_stencil_tag_> const &,
                        __tag<true>)
      {
        return false;
      }


      THRUST_DEVICE_FUNCTION bool
      predicate_wrapper(wrap_value<STENCIL, stencil_type> const &,
                        __tag<false>)
      {
        return false;
      }

      template <bool IS_LAST_TILE, ItemStencil TYPE, class T>
      THRUST_DEVICE_FUNCTION void
      compute_selection_flags(int num_tile_items,
                              T (&values)[ITEMS_PER_THREAD],
                              Size (&selection_flags)[ITEMS_PER_THREAD])
      {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          // Out-of-bounds items are selection_flags
          selection_flags[ITEM] = 1;

          if (!IS_LAST_TILE ||
              (Size(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
          {
            selection_flags[ITEM] =
                predicate_wrapper(wrap_value<TYPE, T>(values[ITEM]),
                                  __tag<USE_STENCIL>());
          }
        }
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
        item_type items_loc[ITEMS_PER_THREAD];
        Size      selection_flags[ITEMS_PER_THREAD];
        Size      selection_idx[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
          BlockLoadItems(temp_storage.load_items)
              .Load(items_glob + tile_base, items_loc, num_tile_items);
        }
        else
        {
          BlockLoadItems(temp_storage.load_items)
              .Load(items_glob + tile_base, items_loc);
        }

        core::sync_threadblock();

        if (USE_STENCIL)
        {
          stencil_type stencil_loc[ITEMS_PER_THREAD];

          if (IS_LAST_TILE)
          {
            BlockLoadStencil(temp_storage.load_stencil)
                .Load(stencil_glob + tile_base, stencil_loc, num_tile_items);
          }
          else
          {
            BlockLoadStencil(temp_storage.load_stencil)
                .Load(stencil_glob + tile_base, stencil_loc);
          }

          compute_selection_flags<IS_LAST_TILE, STENCIL>(num_tile_items,
                                                         stencil_loc,
                                                         selection_flags);
        }
        else /* Use predicate on items rather then stencil */
        {
          compute_selection_flags<IS_LAST_TILE, ITEM>(num_tile_items,
                                                      items_loc,
                                                      selection_flags);
        }

        core::sync_threadblock();

        Size num_tile_selections   = 0;
        Size num_selections        = 0;
        Size num_selections_prefix = 0;
        Size num_rejected_prefix   = 0;
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
          num_rejected_prefix   = tile_base - num_selections_prefix;

          if (IS_LAST_TILE)
          {
            int num_discount = ITEMS_PER_TILE - num_tile_items;
            num_tile_selections -= num_discount;
            num_selections -= num_discount;
          }
        }

        core::sync_threadblock();

        scatter<IS_LAST_TILE>(items_loc,
                              selection_flags,
                              selection_idx,
                              num_tile_items,
                              num_tile_selections,
                              num_selections_prefix,
                              num_rejected_prefix,
                              num_selections);


        return num_selections;
      }


      template <bool         IS_LAST_TILE>
      THRUST_DEVICE_FUNCTION Size
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
           ItemsLoadIt      items_glob_,
           StencilLoadIt    stencil_glob_,
           SelectedOutIt    selected_out_glob_,
           RejectedOutIt    rejected_out_glob_,
           Predicate        predicate_,
           Size             num_items_,
           int              num_tiles,
           NumSelectedOutIt num_selected_out)
          : temp_storage(temp_storage_),
            tile_state(tile_state_),
            items_glob(items_glob_),
            stencil_glob(stencil_glob_),
            selected_out_glob(selected_out_glob_),
            rejected_out_glob(rejected_out_glob_),
            predicate(predicate_),
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
      }    //
    };     //struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    THRUST_AGENT_ENTRY(ItemsIt          items,
                       StencilIt        stencil,
                       SelectedOutIt    selected_out,
                       RejectedOutIt    rejected_out,
                       Predicate        predicate,
                       Size             num_items,
                       NumSelectedOutIt num_selected_out,
                       ScanTileState    tile_state,
                       int              num_tiles,
                       char *           shmem)
    {
      TempStorage &storage = *reinterpret_cast<TempStorage *>(shmem);

      impl(storage,
           tile_state,
           core::make_load_iterator(ptx_plan(), items),
           core::make_load_iterator(ptx_plan(), stencil),
           selected_out,
           rejected_out,
           predicate,
           num_items,
           num_tiles,
           num_selected_out);
    }
  };       // struct PartitionAgent

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
                       char *        /*shmem*/)
    {
      tile_state.InitializeStatus(num_tiles);
      if (blockIdx.x == 0 && threadIdx.x == 0)
        *num_selected_out = 0;
    }

  }; // struct InitAgent

  template <class ItemsIt,
            class StencilIt,
            class SelectedOutIt,
            class RejectedOutIt,
            class Predicate,
            class Size,
            class NumSelectedOutIt>
  static cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *           d_temp_storage,
            size_t &         temp_storage_bytes,
            ItemsIt          items,
            StencilIt        stencil,
            SelectedOutIt    selected_out,
            RejectedOutIt    rejected_out,
            Predicate        predicate,
            NumSelectedOutIt num_selected_out,
            Size             num_items,
            cudaStream_t     stream,
            bool             debug_sync)
  {
    using core::AgentLauncher;
    using core::AgentPlan;
    using core::get_agent_plan;

    typedef AgentLauncher<
        PartitionAgent<ItemsIt,
                       StencilIt,
                       SelectedOutIt,
                       RejectedOutIt,
                       Predicate,
                       Size,
                       NumSelectedOutIt> >
        partition_agent;

    typedef typename partition_agent::ScanTileState ScanTileState;

    typedef AgentLauncher<
        InitAgent<ScanTileState, NumSelectedOutIt, Size> >
        init_agent;


    using core::get_plan;
    typename get_plan<init_agent>::type      init_plan      = init_agent::get_plan();
    typename get_plan<partition_agent>::type partition_plan = partition_agent::get_plan(stream);

    int tile_size = partition_plan.items_per_tile;
    size_t num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

    size_t vshmem_storage = core::vshmem_size(partition_plan.shared_memory_size,
                                              num_tiles);

    cudaError_t status = cudaSuccess;
    if (num_items == 0)
      return status;

    size_t allocation_sizes[2] = {0, vshmem_storage};
    status = ScanTileState::AllocationSize(static_cast<int>(num_tiles), allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);


    void* allocations[2] = {NULL, NULL};
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
    status = tile_status.Init(static_cast<int>(num_tiles), allocations[0], allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    init_agent ia(init_plan, num_tiles, stream, "partition::init_agent", debug_sync);

    char *vshmem_ptr = vshmem_storage > 0 ? (char *)allocations[1] : NULL;

    partition_agent pa(partition_plan, num_items, stream, vshmem_ptr, "partition::partition_agent", debug_sync);

    ia.launch(tile_status, num_tiles, num_selected_out);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    pa.launch(items,
              stencil,
              selected_out,
              rejected_out,
              predicate,
              num_items,
              num_selected_out,
              tile_status,
              num_tiles);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
    return status;

  }

  template <typename Derived,
            typename InputIt,
            typename StencilIt,
            typename SelectedOutIt,
            typename RejectedOutIt,
            typename Predicate>
  THRUST_RUNTIME_FUNCTION
  pair<SelectedOutIt, RejectedOutIt>
  partition(execution_policy<Derived>& policy,
            InputIt                    first,
            InputIt                    last,
            StencilIt                  stencil,
            SelectedOutIt              selected_result,
            RejectedOutIt              rejected_result,
            Predicate                  predicate)
  {
    typedef typename iterator_traits<InputIt>::difference_type size_type;

    size_type    num_items          = static_cast<size_type>(thrust::distance(first, last));
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    status = doit_step(NULL,
                       temp_storage_bytes,
                       first,
                       stencil,
                       selected_result,
                       rejected_result,
                       predicate,
                       reinterpret_cast<size_type*>(NULL),
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "partition failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(size_type), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;

    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "partition failed on 1st alias_storage");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "partition failed on 2nd alias_storage");

    size_type* d_num_selected_out
      = thrust::detail::aligned_reinterpret_cast<size_type*>(allocations[0]);

    status = doit_step(allocations[1],
                       temp_storage_bytes,
                       first,
                       stencil,
                       selected_result,
                       rejected_result,
                       predicate,
                       d_num_selected_out,
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "partition failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "partition failed to synchronize");

    size_type num_selected = 0;
    if (num_items > 0)
    {
      num_selected = get_value(policy, d_num_selected_out);
    }

    return thrust::make_pair(selected_result + num_selected,
                             rejected_result + num_items - num_selected);
  }

  template <typename Derived,
            typename Iterator,
            typename StencilIt,
            typename Predicate>
  THRUST_RUNTIME_FUNCTION
  Iterator partition_inplace(execution_policy<Derived>& policy,
                             Iterator                   first,
                             Iterator                   last,
                             StencilIt                  stencil,
                             Predicate                  predicate)
  {
    typedef typename iterator_traits<Iterator>::difference_type size_type;
    typedef typename iterator_traits<Iterator>::value_type      value_type;

    size_type num_items = thrust::distance(first, last);

    // Allocate temporary storage.
    thrust::detail::temporary_array<value_type, Derived> tmp(policy, num_items);

    cuda_cub::uninitialized_copy(policy, first, last, tmp.begin());

    pair<Iterator, single_output_tag> result =
        partition(policy,
                  tmp.data().get(),
                  tmp.data().get() + num_items,
                  stencil,
                  first,
                  single_output_tag(),
                  predicate);

    size_type num_selected = result.first - first;

    return first + num_selected;
  }
}    // namespace __partition

///// copy

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class StencilIt,
          class SelectedOutIt,
          class RejectedOutIt,
          class Predicate>
pair<SelectedOutIt, RejectedOutIt> __host__ __device__
partition_copy(execution_policy<Derived> &policy,
               InputIt                    first,
               InputIt                    last,
               StencilIt                  stencil,
               SelectedOutIt              selected_result,
               RejectedOutIt              rejected_result,
               Predicate                  predicate)
{
  pair<SelectedOutIt, RejectedOutIt> ret = thrust::make_pair(selected_result, rejected_result);
  if (__THRUST_HAS_CUDART__)
  {
    ret = __partition::partition(policy,
                            first,
                            last,
                            stencil,
                            selected_result,
                            rejected_result,
                            predicate);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::partition_copy(cvt_to_seq(derived_cast(policy)),
                                 first,
                                 last,
                                 stencil,
                                 selected_result,
                                 rejected_result,
                                 predicate);
#endif
  }
  return ret;
}

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class SelectedOutIt,
          class RejectedOutIt,
          class Predicate>
pair<SelectedOutIt, RejectedOutIt> __host__ __device__
partition_copy(execution_policy<Derived> &policy,
               InputIt                    first,
               InputIt                    last,
               SelectedOutIt              selected_result,
               RejectedOutIt              rejected_result,
               Predicate                  predicate)
{
  pair<SelectedOutIt, RejectedOutIt> ret = thrust::make_pair(selected_result, rejected_result);
  if (__THRUST_HAS_CUDART__)
  {
    ret = __partition::partition(policy,
                                 first,
                                 last,
                                 __partition::no_stencil_tag(),
                                 selected_result,
                                 rejected_result,
                                 predicate);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::partition_copy(cvt_to_seq(derived_cast(policy)),
                                 first,
                                 last,
                                 selected_result,
                                 rejected_result,
                                 predicate);
#endif
  }
  return ret;
}

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class SelectedOutIt,
          class RejectedOutIt,
          class Predicate>
pair<SelectedOutIt, RejectedOutIt> __host__ __device__
stable_partition_copy(execution_policy<Derived> &policy,
                      InputIt                    first,
                      InputIt                    last,
                      SelectedOutIt              selected_result,
                      RejectedOutIt              rejected_result,
                      Predicate                  predicate)
{
  pair<SelectedOutIt, RejectedOutIt> ret = thrust::make_pair(selected_result, rejected_result);
  if (__THRUST_HAS_CUDART__)
  {
    ret = __partition::partition(policy,
                                 first,
                                 last,
                                 __partition::no_stencil_tag(),
                                 selected_result,
                                 rejected_result,
                                 predicate);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::stable_partition_copy(cvt_to_seq(derived_cast(policy)),
                                        first,
                                        last,
                                        selected_result,
                                        rejected_result,
                                        predicate);
#endif
  }
  return ret;
}

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class StencilIt,
          class SelectedOutIt,
          class RejectedOutIt,
          class Predicate>
pair<SelectedOutIt, RejectedOutIt> __host__ __device__
stable_partition_copy(execution_policy<Derived> &policy,
                      InputIt                    first,
                      InputIt                    last,
                      StencilIt                  stencil,
                      SelectedOutIt              selected_result,
                      RejectedOutIt              rejected_result,
                      Predicate                  predicate)
{
  pair<SelectedOutIt, RejectedOutIt> ret = thrust::make_pair(selected_result, rejected_result);
  if (__THRUST_HAS_CUDART__)
  {
    ret = __partition::partition(policy,
                                 first,
                                 last,
                                 stencil,
                                 selected_result,
                                 rejected_result,
                                 predicate);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::stable_partition_copy(cvt_to_seq(derived_cast(policy)),
                                        first,
                                        last,
                                        stencil,
                                        selected_result,
                                        rejected_result,
                                        predicate);
#endif
  }
  return ret;
}

/// inplace

__thrust_exec_check_disable__
template <class Derived,
          class Iterator,
          class StencilIt,
          class Predicate>
Iterator __host__ __device__
partition(execution_policy<Derived> &policy,
          Iterator                   first,
          Iterator                   last,
          StencilIt                  stencil,
          Predicate                  predicate)
{
  Iterator ret = first;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __partition::partition_inplace(policy, first, last, stencil, predicate);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::partition(cvt_to_seq(derived_cast(policy)),
                            first,
                            last,
                            stencil,
                            predicate);
#endif
  }
  return ret;
}

__thrust_exec_check_disable__
template <class Derived,
          class Iterator,
          class Predicate>
Iterator __host__ __device__
partition(execution_policy<Derived> &policy,
          Iterator                   first,
          Iterator                   last,
          Predicate                  predicate)
{
  Iterator ret = first;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __partition::partition_inplace(policy,
                                         first,
                                         last,
                                         __partition::no_stencil_tag(),
                                         predicate);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::partition(cvt_to_seq(derived_cast(policy)),
                            first,
                            last,
                            predicate);
#endif
  }
  return ret;
}

__thrust_exec_check_disable__
template <class Derived,
          class Iterator,
          class StencilIt,
          class Predicate>
Iterator __host__ __device__
stable_partition(execution_policy<Derived> &policy,
                 Iterator                   first,
                 Iterator                   last,
                 StencilIt                  stencil,
                 Predicate                  predicate)
{
  Iterator result = first;
  if (__THRUST_HAS_CUDART__)
  {
    result = __partition::partition_inplace(policy,
                                    first,
                                    last,
                                    stencil,
                                    predicate);

    // partition returns rejected values in reverese order
    // so reverse the rejected elements to make it stable
    cuda_cub::reverse(policy, result, last);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    result = thrust::stable_partition(cvt_to_seq(derived_cast(policy)),
                                      first,
                                      last,
                                      stencil,
                                      predicate);
#endif
  }
  return result;
}

__thrust_exec_check_disable__
template <class Derived,
          class Iterator,
          class Predicate>
Iterator __host__ __device__
stable_partition(execution_policy<Derived> &policy,
                 Iterator                   first,
                 Iterator                   last,
                 Predicate                  predicate)
{
  Iterator result = first;
  if (__THRUST_HAS_CUDART__)
  {
    result = __partition::partition_inplace(policy,
                                       first,
                                       last,
                                       __partition::no_stencil_tag(),
                                       predicate);

    // partition returns rejected values in reverese order
    // so reverse the rejected elements to make it stable
    cuda_cub::reverse(policy, result, last);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    result = thrust::stable_partition(cvt_to_seq(derived_cast(policy)),
                                      first,
                                      last,
                                      predicate);
#endif
  }
  return result;
}

template <class Derived,
          class ItemsIt,
          class Predicate>
bool __host__ __device__
is_partitioned(execution_policy<Derived> &policy,
               ItemsIt                    first,
               ItemsIt                    last,
               Predicate                  predicate)
{
  ItemsIt boundary = cuda_cub::find_if_not(policy, first, last, predicate);
  ItemsIt end      = cuda_cub::find_if(policy,boundary,last,predicate);
  return end == last;
}


}    // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
