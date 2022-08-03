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
#include <cub/device/device_select.cuh>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/core/util.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/detail/function.h>
#include <thrust/distance.h>
#include <thrust/detail/alignment.h>

#include <cub/util_math.cuh>

THRUST_NAMESPACE_BEGIN
// XXX declare generic copy_if interface
// to avoid circulular dependency from thrust/copy.h
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
__host__ __device__
    OutputIterator
    copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            InputIterator                                               first,
            InputIterator                                               last,
            OutputIterator                                              result,
            Predicate                                                   pred);

template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
__host__ __device__
    OutputIterator
    copy_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
            InputIterator1                                              first,
            InputIterator1                                              last,
            InputIterator2                                              stencil,
            OutputIterator                                              result,
            Predicate                                                   pred);

namespace cuda_cub {

namespace __copy_if {

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
      ITEMS_PER_TILE     = _BLOCK_THREADS * _ITEMS_PER_THREAD,
    };
    static const cub::BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;
    static const cub::CacheLoadModifier  LOAD_MODIFIER  = _LOAD_MODIFIER;
    static const cub::BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
  };    // struct PtxPolicy

  template<class, class>
  struct Tuning;

  template<class T>
  struct Tuning<sm52, T>
  {
    const static int INPUT_SIZE = sizeof(T);

    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 9,
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

  struct no_stencil_tag_    {};
  typedef no_stencil_tag_* no_stencil_tag;
  template <class ItemsIt,
            class StencilIt,
            class OutputIt,
            class Predicate,
            class Size,
            class NumSelectedOutputIt>
  struct CopyIfAgent
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

      TempStorage &  storage;
      ScanTileState &tile_state;
      ItemsLoadIt    items_load_it;
      StencilLoadIt  stencil_load_it;
      OutputIt       output_it;
      Predicate      predicate;
      Size           num_items;

      //------------------------------------------
      // scatter results to memory
      //------------------------------------------

      THRUST_DEVICE_FUNCTION void
      scatter(item_type (&items)[ITEMS_PER_THREAD],
              Size (&selection_flags)[ITEMS_PER_THREAD],
              Size (&selection_indices)[ITEMS_PER_THREAD],
              int  num_tile_selections,
              Size num_selections_prefix)
      {
        using core::sync_threadblock;

#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          int local_scatter_offset = selection_indices[ITEM] -
                                     num_selections_prefix;
          if (selection_flags[ITEM])
          {
            new (&storage.raw_exchange[local_scatter_offset]) item_type(items[ITEM]);
          }
        }

        sync_threadblock();

        for (int item = threadIdx.x;
             item < num_tile_selections;
             item += BLOCK_THREADS)
        {
          output_it[num_selections_prefix + item] = storage.raw_exchange[item];
        }
      }    // func scatter

      //------------------------------------------
      // specialize predicate on different types
      //------------------------------------------

      template <int T>
      struct __tag {};

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

      //------------------------------------------
      // consume tiles
      //------------------------------------------

      template <bool IS_LAST_TILE, bool IS_FIRST_TILE>
      Size THRUST_DEVICE_FUNCTION
      consume_tile_impl(int  num_tile_items,
                        int  tile_idx,
                        Size tile_base)
      {
        item_type items_loc[ITEMS_PER_THREAD];
        Size      selection_flags[ITEMS_PER_THREAD];
        Size      selection_idx[ITEMS_PER_THREAD];

        if (IS_LAST_TILE) {
          BlockLoadItems(storage.load_items)
              .Load(items_load_it + tile_base,
                    items_loc,
                    num_tile_items);
        }
        else
        {
          BlockLoadItems(storage.load_items)
              .Load(items_load_it + tile_base,
                    items_loc);
        }

        core::sync_threadblock();

        if (USE_STENCIL)
        {
          stencil_type stencil_loc[ITEMS_PER_THREAD];

          if (IS_LAST_TILE)
          {
            BlockLoadStencil(storage.load_stencil)
                .Load(stencil_load_it + tile_base,
                      stencil_loc,
                      num_tile_items);
          }
          else
          {
            BlockLoadStencil(storage.load_stencil)
                .Load(stencil_load_it + tile_base,
                      stencil_loc);
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
        if (IS_FIRST_TILE)
        {
          BlockScan(storage.scan_storage.scan)
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
                                       storage.scan_storage.prefix,
                                       cub::Sum(),
                                       tile_idx);
          BlockScan(storage.scan_storage.scan)
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

        core::sync_threadblock();

        scatter(items_loc,
                selection_flags,
                selection_idx,
                num_tile_selections,
                num_selections_prefix);


        return num_selections;
      }    // func consume_tile_impl

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
      }    // func consume_tile

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION impl(TempStorage &       storage_,
                                  ScanTileState &     tile_state_,
                                  ItemsIt             items_it,
                                  StencilIt           stencil_it,
                                  OutputIt            output_it_,
                                  Predicate           predicate_,
                                  Size                num_items_,
                                  int                 num_tiles,
                                  NumSelectedOutputIt num_selected_out)
          : storage(storage_),
            tile_state(tile_state_),
            items_load_it(core::make_load_iterator(ptx_plan(), items_it)),
            stencil_load_it(core::make_load_iterator(ptx_plan(), stencil_it)),
            output_it(output_it_),
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
      }    // ctor impl
    };

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    THRUST_AGENT_ENTRY(ItemsIt             items_it,
                       StencilIt           stencil_it,
                       OutputIt            output_it,
                       Predicate           predicate,
                       Size                num_items,
                       NumSelectedOutputIt num_selected_out,
                       ScanTileState       tile_state,
                       int                 num_tiles,
                       char *              shmem)
    {
      TempStorage &storage = *reinterpret_cast<TempStorage *>(shmem);

      impl(storage,
           tile_state,
           items_it,
           stencil_it,
           output_it,
           predicate,
           num_items,
           num_tiles,
           num_selected_out);
    }
  };    // struct CopyIfAgent

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
  };    // struct InitAgent

  template <class ItemsIt,
            class StencilIt,
            class OutputIt,
            class Predicate,
            class Size,
            class NumSelectedOutIt>
  static cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *           d_temp_storage,
            size_t &         temp_storage_bytes,
            ItemsIt          items,
            StencilIt        stencil,
            OutputIt         output_it,
            Predicate        predicate,
            NumSelectedOutIt num_selected_out,
            Size             num_items,
            cudaStream_t     stream,
            bool             debug_sync)
  {
    if (num_items == 0)
      return cudaSuccess;

    using core::AgentLauncher;
    using core::AgentPlan;
    using core::get_agent_plan;

    typedef AgentLauncher<
        CopyIfAgent<ItemsIt,
                    StencilIt,
                    OutputIt,
                    Predicate,
                    Size,
                    NumSelectedOutIt> >
        copy_if_agent;

    typedef typename copy_if_agent::ScanTileState ScanTileState;

    typedef AgentLauncher<
        InitAgent<ScanTileState, NumSelectedOutIt, Size> >
        init_agent;


    using core::get_plan;
    typename get_plan<init_agent>::type    init_plan    = init_agent::get_plan();
    typename get_plan<copy_if_agent>::type copy_if_plan = copy_if_agent::get_plan(stream);

    int tile_size = copy_if_plan.items_per_tile;
    size_t num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

    size_t vshmem_size = core::vshmem_size(copy_if_plan.shared_memory_size,
                                           num_tiles);

    cudaError_t status = cudaSuccess;
    if (num_items == 0)
      return status;

    size_t allocation_sizes[2] = {0, vshmem_size};
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

    init_agent ia(init_plan, num_tiles, stream, "copy_if::init_agent", debug_sync);

    char *vshmem_ptr = vshmem_size > 0 ? (char*)allocations[1] : NULL;

    copy_if_agent pa(copy_if_plan, num_items, stream, vshmem_ptr, "copy_if::partition_agent", debug_sync);

    ia.launch(tile_status, num_tiles, num_selected_out);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    pa.launch(items,
              stencil,
              output_it,
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
            typename OutputIt,
            typename Predicate>
  THRUST_RUNTIME_FUNCTION
  OutputIt copy_if(execution_policy<Derived>& policy,
                   InputIt                    first,
                   InputIt                    last,
                   StencilIt                  stencil,
                   OutputIt                   output,
                   Predicate                  predicate)
  {
    typedef int size_type;

    size_type    num_items          = static_cast<size_type>(thrust::distance(first, last));
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    if (num_items == 0)
      return output;

    cudaError_t status;
    status = doit_step(NULL,
                       temp_storage_bytes,
                       first,
                       stencil,
                       output,
                       predicate,
                       reinterpret_cast<size_type*>(NULL),
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "copy_if failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(size_type), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;

    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "copy_if failed on 1st alias_storage");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "copy_if failed on 2nd alias_storage");

    size_type* d_num_selected_out
      = thrust::detail::aligned_reinterpret_cast<size_type*>(allocations[0]);

    status = doit_step(allocations[1],
                       temp_storage_bytes,
                       first,
                       stencil,
                       output,
                       predicate,
                       d_num_selected_out,
                       num_items,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "copy_if failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "copy_if failed to synchronize");

    size_type num_selected = get_value(policy, d_num_selected_out);

    return output + num_selected;
  }

}    // namespace __copy_if

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class InputIterator,
          class OutputIterator,
          class Predicate>
OutputIterator __host__ __device__
copy_if(execution_policy<Derived> &policy,
        InputIterator              first,
        InputIterator              last,
        OutputIterator             result,
        Predicate                  pred)
{
  OutputIterator ret = result;

  if (__THRUST_HAS_CUDART__)
  {
    ret = __copy_if::copy_if(policy,
                             first,
                             last,
                             __copy_if::no_stencil_tag(),
                             result,
                             pred);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::copy_if(cvt_to_seq(derived_cast(policy)),
                          first,
                          last,
                          result,
                          pred);
#endif
  }
  return ret;
} // func copy_if

__thrust_exec_check_disable__
template <class Derived,
          class InputIterator,
          class StencilIterator,
          class OutputIterator,
          class Predicate>
OutputIterator __host__ __device__
copy_if(execution_policy<Derived> &policy,
        InputIterator              first,
        InputIterator              last,
        StencilIterator            stencil,
        OutputIterator             result,
        Predicate                  pred)
{
  OutputIterator ret = result;

  if (__THRUST_HAS_CUDART__)
  {
    ret = __copy_if::copy_if(policy,
                             first,
                             last,
                             stencil,
                             result,
                             pred);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::copy_if(cvt_to_seq(derived_cast(policy)),
                          first,
                          last,
                          stencil,
                          result,
                          pred);
#endif
  }
  return ret;
}    // func copy_if

}    // namespace cuda_cub
THRUST_NAMESPACE_END

#include <thrust/copy.h>
#endif
