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
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <cub/device/device_reduce.cuh>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/system/cuda/detail/get_value.h>
#include <thrust/system/cuda/detail/dispatch.h>
#include <thrust/system/cuda/detail/make_unsigned_special.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>
#include <thrust/detail/alignment.h>

#include <cub/util_math.cuh>

THRUST_NAMESPACE_BEGIN

// forward declare generic reduce
// to circumvent circular dependency
template <typename DerivedPolicy,
          typename InputIterator,
          typename T,
          typename BinaryFunction>
T __host__ __device__
reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
       InputIterator                                               first,
       InputIterator                                               last,
       T                                                           init,
       BinaryFunction                                              binary_op);

namespace cuda_cub {

namespace __reduce {

  template<bool>
  struct is_true : thrust::detail::false_type {};
  template<>
  struct is_true<true> : thrust::detail::true_type {};

  template <int                       _BLOCK_THREADS,
            int                       _ITEMS_PER_THREAD   = 1,
            int                       _VECTOR_LOAD_LENGTH = 1,
            cub::BlockReduceAlgorithm _BLOCK_ALGORITHM    = cub::BLOCK_REDUCE_RAKING,
            cub::CacheLoadModifier    _LOAD_MODIFIER      = cub::LOAD_DEFAULT,
            cub::GridMappingStrategy  _GRID_MAPPING       = cub::GRID_MAPPING_DYNAMIC>
  struct PtxPolicy
  {
    enum
    {
      BLOCK_THREADS      = _BLOCK_THREADS,
      ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
      VECTOR_LOAD_LENGTH = _VECTOR_LOAD_LENGTH,
      ITEMS_PER_TILE     = _BLOCK_THREADS * _ITEMS_PER_THREAD
    };

    static const cub::BlockReduceAlgorithm BLOCK_ALGORITHM = _BLOCK_ALGORITHM;
    static const cub::CacheLoadModifier    LOAD_MODIFIER   = _LOAD_MODIFIER;
    static const cub::GridMappingStrategy  GRID_MAPPING    = _GRID_MAPPING;
  }; // struct PtxPolicy

  template<class,class>
  struct Tuning;

  template <class T>
  struct Tuning<sm30, T>
  {
    enum
    {
      // Relative size of T type to a 4-byte word
      SCALE_FACTOR_4B = (sizeof(T) + 3) / 4,
      // Relative size of T type to a 1-byte word
      SCALE_FACTOR_1B = sizeof(T),
    };

    typedef PtxPolicy<256,
                      CUB_MAX(1, 20 / SCALE_FACTOR_4B),
                      2,
                      cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                      cub::LOAD_DEFAULT,
                      cub::GRID_MAPPING_RAKE>
        type;
  }; // Tuning sm30

  template <class T>
  struct Tuning<sm35, T> : Tuning<sm30,T>
  {
    // ReducePolicy1B (GTX Titan: 228.7 GB/s @ 192M 1B items)
    typedef PtxPolicy<128,
                      CUB_MAX(1, 24 / Tuning::SCALE_FACTOR_1B),
                      4,
                      cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                      cub::LOAD_LDG,
                      cub::GRID_MAPPING_DYNAMIC>
        ReducePolicy1B;

    // ReducePolicy4B types (GTX Titan: 255.1 GB/s @ 48M 4B items)
    typedef PtxPolicy<256,
                      CUB_MAX(1, 20 / Tuning::SCALE_FACTOR_4B),
                      4,
                      cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                      cub::LOAD_LDG,
                      cub::GRID_MAPPING_DYNAMIC>
        ReducePolicy4B;

    typedef typename thrust::detail::conditional<(sizeof(T) < 4),
                                                 ReducePolicy1B,
                                                 ReducePolicy4B>::type type;
  };    // Tuning sm35

  template <class InputIt,
            class OutputIt,
            class T,
            class Size,
            class ReductionOp>
  struct ReduceAgent
  {
    typedef typename detail::make_unsigned_special<Size>::type UnsignedSize;

    template<class Arch>
    struct PtxPlan : Tuning<Arch,T>::type
    {
      // we need this type definition to indicate "specialize_plan" metafunction
      // that this PtxPlan may have specializations for different Arch
      // via Tuning<Arch,T> type.
      //
      typedef Tuning<Arch,T> tuning;

      typedef typename cub::CubVector<T, PtxPlan::VECTOR_LOAD_LENGTH> Vector;
      typedef typename core::LoadIterator<PtxPlan, InputIt>::type     LoadIt;
      typedef cub::BlockReduce<T,
                               PtxPlan::BLOCK_THREADS,
                               PtxPlan::BLOCK_ALGORITHM,
                               1,
                               1,
                               Arch::ver>
          BlockReduce;

      typedef cub::CacheModifiedInputIterator<PtxPlan::LOAD_MODIFIER,
                                              Vector,
                                              Size>
          VectorLoadIt;

      struct TempStorage
      {
        typename BlockReduce::TempStorage reduce;
        //
        Size dequeue_offset;
      };    // struct TempStorage


    }; // struct PtxPlan

    // Reduction need additional information which is not covered in
    // default core::AgentPlan. We thus inherit from core::AgentPlan
    // and add additional member fields that are needed.
    // Other algorithms, e.g. merge, may not need additional information,
    // and may use AgentPlan directly, instead of defining their own Plan type.
    //
    struct Plan : core::AgentPlan
    {
      cub::GridMappingStrategy grid_mapping;

      template <class P>
      THRUST_RUNTIME_FUNCTION
          Plan(P) : core::AgentPlan(P()),
                    grid_mapping(P::GRID_MAPPING)
      {
      }
    };

    // this specialized PtxPlan for a device-compiled Arch
    // ptx_plan type *must* only be used from device code
    // Its use from host code will result in *undefined behaviour*
    //
    typedef typename core::specialize_plan_msvc10_war<PtxPlan>::type::type ptx_plan;

    typedef typename ptx_plan::TempStorage  TempStorage;
    typedef typename ptx_plan::Vector       Vector;
    typedef typename ptx_plan::LoadIt       LoadIt;
    typedef typename ptx_plan::BlockReduce  BlockReduce;
    typedef typename ptx_plan::VectorLoadIt VectorLoadIt;

    enum
    {
      ITEMS_PER_THREAD   = ptx_plan::ITEMS_PER_THREAD,
      BLOCK_THREADS      = ptx_plan::BLOCK_THREADS,
      ITEMS_PER_TILE     = ptx_plan::ITEMS_PER_TILE,
      VECTOR_LOAD_LENGTH = ptx_plan::VECTOR_LOAD_LENGTH,

      ATTEMPT_VECTORIZATION = (VECTOR_LOAD_LENGTH > 1) &&
                              (ITEMS_PER_THREAD % VECTOR_LOAD_LENGTH == 0) &&
                              thrust::detail::is_pointer<InputIt>::value &&
                              thrust::detail::is_arithmetic<
                                  typename thrust::detail::remove_cv<T> >::value
    };

    struct impl
    {
      //---------------------------------------------------------------------
      // Per thread data
      //---------------------------------------------------------------------

      TempStorage &storage;
      InputIt      input_it;
      LoadIt       load_it;
      ReductionOp  reduction_op;

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION impl(TempStorage &storage_,
                                  InputIt      input_it_,
                                  ReductionOp  reduction_op_)
          : storage(storage_),
            input_it(input_it_),
            load_it(core::make_load_iterator(ptx_plan(), input_it)),
            reduction_op(reduction_op_) {}

      //---------------------------------------------------------------------
      // Utility
      //---------------------------------------------------------------------


      // Whether or not the input is aligned with the vector type
      // (specialized for types we can vectorize)
      //
      template <class Iterator>
      static THRUST_DEVICE_FUNCTION bool
      is_aligned(Iterator d_in,
                 thrust::detail::true_type /* can_vectorize */)
      {
        return (size_t(d_in) & (sizeof(Vector) - 1)) == 0;
      }

      // Whether or not the input is aligned with the vector type
      // (specialized for types we cannot vectorize)
      //
      template <class Iterator>
      static THRUST_DEVICE_FUNCTION bool
      is_aligned(Iterator,
                 thrust::detail::false_type /* can_vectorize */)
      {
        return false;
      }

      //---------------------------------------------------------------------
      // Tile processing
      //---------------------------------------------------------------------

      // Consume a full tile of input (non-vectorized)
      //
      template <int IS_FIRST_TILE>
      THRUST_DEVICE_FUNCTION void
      consume_tile(T &  thread_aggregate,
                   Size block_offset,
                   int  /*valid_items*/,
                   thrust::detail::true_type /* is_full_tile */,
                   thrust::detail::false_type /* can_vectorize */)
      {
        T items[ITEMS_PER_THREAD];

        // Load items in striped fashion
        cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x,
                                              load_it + block_offset,
                                              items);

        // Reduce items within each thread stripe
        thread_aggregate =
            (IS_FIRST_TILE) ? cub::internal::ThreadReduce(items, reduction_op)
                            : cub::internal::ThreadReduce(items, reduction_op,
                                                          thread_aggregate);
      }

      // Consume a full tile of input (vectorized)
      //
      template <int IS_FIRST_TILE>
      THRUST_DEVICE_FUNCTION void
      consume_tile(T &  thread_aggregate,
                   Size block_offset,
                   int  /*valid_items*/,
                   thrust::detail::true_type /* is_full_tile */,
                   thrust::detail::true_type /* can_vectorize */)
      {
        // Alias items as an array of VectorT and load it in striped fashion
        enum
        {
          WORDS = ITEMS_PER_THREAD / VECTOR_LOAD_LENGTH
        };

        T items[ITEMS_PER_THREAD];

        Vector *vec_items = reinterpret_cast<Vector *>(items);

        // Vector Input iterator wrapper type (for applying cache modifier)
        T *d_in_unqualified = const_cast<T *>(input_it) +
                              block_offset +
                              (threadIdx.x * VECTOR_LOAD_LENGTH);
        VectorLoadIt vec_load_it(reinterpret_cast<Vector *>(d_in_unqualified));

#pragma unroll
        for (int i = 0; i < WORDS; ++i)
        {
          vec_items[i] = vec_load_it[BLOCK_THREADS * i];
        }


        // Reduce items within each thread stripe
        thread_aggregate =
            (IS_FIRST_TILE) ? cub::internal::ThreadReduce(items, reduction_op)
                            : cub::internal::ThreadReduce(items, reduction_op,
                                                          thread_aggregate);
      }


      // Consume a partial tile of input
      //
      template <int IS_FIRST_TILE, class CAN_VECTORIZE>
      THRUST_DEVICE_FUNCTION void
      consume_tile(T &  thread_aggregate,
                   Size block_offset,
                   int  valid_items,
                   thrust::detail::false_type /* is_full_tile */,
                   CAN_VECTORIZE)
      {
        // Partial tile
        int thread_offset = threadIdx.x;

        // Read first item
        if ((IS_FIRST_TILE) && (thread_offset < valid_items))
        {
          thread_aggregate = load_it[block_offset + thread_offset];
          thread_offset += BLOCK_THREADS;
        }

        // Continue reading items (block-striped)
        while (thread_offset < valid_items)
        {
          thread_aggregate = reduction_op(
              thread_aggregate,
              thrust::raw_reference_cast(load_it[block_offset + thread_offset]));
          thread_offset += BLOCK_THREADS;
        }
      }

      //---------------------------------------------------------------
      // Consume a contiguous segment of tiles
      //---------------------------------------------------------------------


      // Reduce a contiguous segment of input tiles
      //
      template <class CAN_VECTORIZE>
      THRUST_DEVICE_FUNCTION T
      consume_range_impl(Size          block_offset,
                         Size          block_end,
                         CAN_VECTORIZE can_vectorize)
      {
        T thread_aggregate;

        if (block_offset + ITEMS_PER_TILE > block_end)
        {
          // First tile isn't full (not all threads have valid items)
          int valid_items = block_end - block_offset;
          consume_tile<true>(thread_aggregate,
                             block_offset,
                             valid_items,
                             thrust::detail::false_type(),
                             can_vectorize);
          return BlockReduce(storage.reduce)
              .Reduce(thread_aggregate, reduction_op, valid_items);
        }

        // At least one full block
        consume_tile<true>(thread_aggregate,
                           block_offset,
                           ITEMS_PER_TILE,
                           thrust::detail::true_type(),
                           can_vectorize);
        block_offset += ITEMS_PER_TILE;

        // Consume subsequent full tiles of input
        while (block_offset + ITEMS_PER_TILE <= block_end)
        {
          consume_tile<false>(thread_aggregate,
                              block_offset,
                              ITEMS_PER_TILE,
                              thrust::detail::true_type(),
                              can_vectorize);
          block_offset += ITEMS_PER_TILE;
        }

        // Consume a partially-full tile
        if (block_offset < block_end)
        {
          int valid_items = block_end - block_offset;
          consume_tile<false>(thread_aggregate,
                              block_offset,
                              valid_items,
                              thrust::detail::false_type(),
                              can_vectorize);
        }

        // Compute block-wide reduction (all threads have valid items)
        return BlockReduce(storage.reduce)
            .Reduce(thread_aggregate, reduction_op);
      }

      // Reduce a contiguous segment of input tiles
      //
      THRUST_DEVICE_FUNCTION T consume_range(Size block_offset,
                                             Size block_end)
      {
        typedef is_true<ATTEMPT_VECTORIZATION>          attempt_vec;
        typedef is_true<true && ATTEMPT_VECTORIZATION>  path_a;
        typedef is_true<false && ATTEMPT_VECTORIZATION> path_b;

        return is_aligned(input_it + block_offset, attempt_vec())
                   ? consume_range_impl(block_offset, block_end, path_a())
                   : consume_range_impl(block_offset, block_end, path_b());
      }

      // Reduce a contiguous segment of input tiles
      //
      THRUST_DEVICE_FUNCTION T
      consume_tiles(Size /*num_items*/,
                    cub::GridEvenShare<Size> &even_share,
                    cub::GridQueue<UnsignedSize> & /*queue*/,
                    thrust::detail::integral_constant<cub::GridMappingStrategy, cub::GRID_MAPPING_RAKE> /*is_rake*/)
      {
        typedef is_true<ATTEMPT_VECTORIZATION>          attempt_vec;
        typedef is_true<true && ATTEMPT_VECTORIZATION>  path_a;
        typedef is_true<false && ATTEMPT_VECTORIZATION> path_b;

        // Initialize even-share descriptor for this thread block
        even_share
            .template BlockInit<ITEMS_PER_TILE, cub::GRID_MAPPING_RAKE>();

        return is_aligned(input_it, attempt_vec())
                   ? consume_range_impl(even_share.block_offset,
                                        even_share.block_end,
                                        path_a())
                   : consume_range_impl(even_share.block_offset,
                                        even_share.block_end,
                                        path_b());
      }


      //---------------------------------------------------------------------
      // Dynamically consume tiles
      //---------------------------------------------------------------------

      // Dequeue and reduce tiles of items as part of a inter-block reduction
      //
      template <class CAN_VECTORIZE>
      THRUST_DEVICE_FUNCTION T
      consume_tiles_impl(Size                         num_items,
                         cub::GridQueue<UnsignedSize> queue,
                         CAN_VECTORIZE                can_vectorize)
      {
        using core::sync_threadblock;

        // We give each thread block at least one tile of input.
        T    thread_aggregate;
        Size block_offset    = blockIdx.x * ITEMS_PER_TILE;
        Size even_share_base = gridDim.x * ITEMS_PER_TILE;

        if (block_offset + ITEMS_PER_TILE > num_items)
        {
          // First tile isn't full (not all threads have valid items)
          int valid_items = num_items - block_offset;
          consume_tile<true>(thread_aggregate,
                             block_offset,
                             valid_items,
                             thrust::detail::false_type(),
                             can_vectorize);
          return BlockReduce(storage.reduce)
              .Reduce(thread_aggregate, reduction_op, valid_items);
        }

        // Consume first full tile of input
        consume_tile<true>(thread_aggregate,
                           block_offset,
                           ITEMS_PER_TILE,
                           thrust::detail::true_type(),
                           can_vectorize);

        if (num_items > even_share_base)
        {
          // Dequeue a tile of items
          if (threadIdx.x == 0)
            storage.dequeue_offset = queue.Drain(ITEMS_PER_TILE) +
                                     even_share_base;

          sync_threadblock();

          // Grab tile offset and check if we're done with full tiles
          block_offset = storage.dequeue_offset;

          // Consume more full tiles
          while (block_offset + ITEMS_PER_TILE <= num_items)
          {
            consume_tile<false>(thread_aggregate,
                                block_offset,
                                ITEMS_PER_TILE,
                                thrust::detail::true_type(),
                                can_vectorize);

            sync_threadblock();

            // Dequeue a tile of items
            if (threadIdx.x == 0)
              storage.dequeue_offset = queue.Drain(ITEMS_PER_TILE) +
                                       even_share_base;

            sync_threadblock();

            // Grab tile offset and check if we're done with full tiles
            block_offset = storage.dequeue_offset;
          }

          // Consume partial tile
          if (block_offset < num_items)
          {
            int valid_items = num_items - block_offset;
            consume_tile<false>(thread_aggregate,
                                block_offset,
                                valid_items,
                                thrust::detail::false_type(),
                                can_vectorize);
          }
        }

        // Compute block-wide reduction (all threads have valid items)
        return BlockReduce(storage.reduce)
            .Reduce(thread_aggregate, reduction_op);
      }


      // Dequeue and reduce tiles of items as part of a inter-block reduction
      //
      THRUST_DEVICE_FUNCTION T
      consume_tiles(
          Size                              num_items,
          cub::GridEvenShare<Size> &/*even_share*/,
          cub::GridQueue<UnsignedSize> &    queue,
          thrust::detail::integral_constant<cub::GridMappingStrategy, cub::GRID_MAPPING_DYNAMIC>)
      {
        typedef is_true<ATTEMPT_VECTORIZATION>         attempt_vec;
        typedef is_true<true && ATTEMPT_VECTORIZATION> path_a;
        typedef is_true<false && ATTEMPT_VECTORIZATION> path_b;

        return is_aligned(input_it, attempt_vec())
                   ? consume_tiles_impl(num_items, queue, path_a())
                   : consume_tiles_impl(num_items, queue, path_b());
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry points
    //---------------------------------------------------------------------

    // single tile reduce entry point
    //
    THRUST_AGENT_ENTRY(InputIt     input_it,
                       OutputIt    output_it,
                       Size        num_items,
                       ReductionOp reduction_op,
                       char *      shmem)
    {
      TempStorage& storage = *reinterpret_cast<TempStorage*>(shmem);

      if (num_items == 0)
      {
        return;
      }

      T block_aggregate =
          impl(storage, input_it, reduction_op).consume_range((Size)0, num_items);

      if (threadIdx.x == 0)
        *output_it = block_aggregate;
    }

    // single tile reduce entry point
    //
    THRUST_AGENT_ENTRY(InputIt     input_it,
                       OutputIt    output_it,
                       Size        num_items,
                       ReductionOp reduction_op,
                       T           init,
                       char *      shmem)
    {
      TempStorage& storage = *reinterpret_cast<TempStorage*>(shmem);

      if (num_items == 0)
      {
        if (threadIdx.x == 0)
          *output_it = init;
        return;
      }

      T block_aggregate =
          impl(storage, input_it, reduction_op).consume_range((Size)0, num_items);

      if (threadIdx.x == 0)
        *output_it = reduction_op(init, block_aggregate);
    }

    THRUST_AGENT_ENTRY(InputIt                          input_it,
                       OutputIt                         output_it,
                       Size                             num_items,
                       cub::GridEvenShare<Size> even_share,
                       cub::GridQueue<UnsignedSize>     queue,
                       ReductionOp                      reduction_op,
                       char *                           shmem)
    {
      TempStorage& storage = *reinterpret_cast<TempStorage*>(shmem);

      typedef thrust::detail::integral_constant<cub::GridMappingStrategy, ptx_plan::GRID_MAPPING> grid_mapping;

      T block_aggregate =
          impl(storage, input_it, reduction_op)
              .consume_tiles(num_items, even_share, queue, grid_mapping());

      if (threadIdx.x == 0)
        output_it[blockIdx.x] = block_aggregate;
    }
  };    // struct ReduceAgent

  template<class Size>
  struct DrainAgent
  {
    typedef typename detail::make_unsigned_special<Size>::type UnsignedSize;

    template <class Arch>
    struct PtxPlan : PtxPolicy<1> {};
    typedef core::specialize_plan<PtxPlan> ptx_plan;

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    THRUST_AGENT_ENTRY(cub::GridQueue<UnsignedSize> grid_queue,
                       Size                         num_items,
                       char * /*shmem*/)
    {
      grid_queue.FillAndResetDrain(num_items);
    }
  };    // struct DrainAgent;


  template <class InputIt,
            class OutputIt,
            class Size,
            class ReductionOp,
            class T>
  cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *       d_temp_storage,
            size_t &     temp_storage_bytes,
            InputIt      input_it,
            Size         num_items,
            T            init,
            ReductionOp  reduction_op,
            OutputIt     output_it,
            cudaStream_t stream,
            bool         debug_sync)
  {
    using core::AgentPlan;
    using core::AgentLauncher;
    using core::get_agent_plan;
    using core::cuda_optional;

    typedef typename detail::make_unsigned_special<Size>::type UnsignedSize;

    if (num_items == 0)
      return cudaErrorNotSupported;

    typedef AgentLauncher<
        ReduceAgent<InputIt, OutputIt, T, Size, ReductionOp> >
        reduce_agent;

    typename reduce_agent::Plan reduce_plan = reduce_agent::get_plan(stream);

    cudaError_t status = cudaSuccess;


    if (num_items <= reduce_plan.items_per_tile)
    {
      size_t vshmem_size = core::vshmem_size(reduce_plan.shared_memory_size, 1);

      // small, single tile size
      if (d_temp_storage == NULL)
      {
        temp_storage_bytes = max<size_t>(1, vshmem_size);
        return status;
      }
      char *vshmem_ptr = vshmem_size > 0 ? (char*)d_temp_storage : NULL;

      reduce_agent ra(reduce_plan, num_items, stream, vshmem_ptr, "reduce_agent: single_tile only", debug_sync);
      ra.launch(input_it, output_it, num_items, reduction_op, init);
      CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
    }
    else
    {
      // regular size
      cuda_optional<int> sm_count = core::get_sm_count();
      CUDA_CUB_RET_IF_FAIL(sm_count.status());

      // reduction will not use more cta counts than requested
      cuda_optional<int> max_blocks_per_sm =
          reduce_agent::
              template get_max_blocks_per_sm<InputIt,
                                             OutputIt,
                                             Size,
                                             cub::GridEvenShare<Size>,
                                             cub::GridQueue<UnsignedSize>,
                                             ReductionOp>(reduce_plan);
      CUDA_CUB_RET_IF_FAIL(max_blocks_per_sm.status());



      int reduce_device_occupancy = (int)max_blocks_per_sm * sm_count;

      int sm_oversubscription = 5;
      int max_blocks          = reduce_device_occupancy * sm_oversubscription;

      cub::GridEvenShare<Size> even_share;
      even_share.DispatchInit(static_cast<int>(num_items), max_blocks,
                              reduce_plan.items_per_tile);

      // we will launch at most "max_blocks" blocks in a grid
      // so preallocate virtual shared memory storage for this if required
      //
      size_t vshmem_size = core::vshmem_size(reduce_plan.shared_memory_size,
                                             max_blocks);

      // Temporary storage allocation requirements
      void * allocations[3] = {NULL, NULL, NULL};
      size_t allocation_sizes[3] =
          {
              max_blocks * sizeof(T),                            // bytes needed for privatized block reductions
              cub::GridQueue<UnsignedSize>::AllocationSize(),    // bytes needed for grid queue descriptor0
              vshmem_size                                        // size of virtualized shared memory storage
          };
      status = cub::AliasTemporaries(d_temp_storage,
                                     temp_storage_bytes,
                                     allocations,
                                     allocation_sizes);
      CUDA_CUB_RET_IF_FAIL(status);
      if (d_temp_storage == NULL)
      {
        return status;
      }

      T *d_block_reductions = (T*) allocations[0];
      cub::GridQueue<UnsignedSize> queue(allocations[1]);
      char *vshmem_ptr = vshmem_size > 0 ? (char *)allocations[2] : NULL;


      // Get grid size for device_reduce_sweep_kernel
      int reduce_grid_size = 0;
      if (reduce_plan.grid_mapping == cub::GRID_MAPPING_RAKE)
      {
        // Work is distributed evenly
        reduce_grid_size = even_share.grid_size;
      }
      else if (reduce_plan.grid_mapping == cub::GRID_MAPPING_DYNAMIC)
      {
        // Work is distributed dynamically
        size_t num_tiles = cub::DivideAndRoundUp(num_items, reduce_plan.items_per_tile);

        // if not enough to fill the device with threadblocks
        // then fill the device with threadblocks
        reduce_grid_size = static_cast<int>((min)(num_tiles, static_cast<size_t>(reduce_device_occupancy)));

        typedef AgentLauncher<DrainAgent<Size> > drain_agent;
        AgentPlan drain_plan = drain_agent::get_plan();
        drain_plan.grid_size = 1;
        drain_agent da(drain_plan, stream, "__reduce::drain_agent", debug_sync);
        da.launch(queue, num_items);
        CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
      }
      else
      {
        CUDA_CUB_RET_IF_FAIL(cudaErrorNotSupported);
      }

      reduce_plan.grid_size = reduce_grid_size;
      reduce_agent ra(reduce_plan, stream, vshmem_ptr, "reduce_agent: regular size reduce", debug_sync);
      ra.launch(input_it,
                d_block_reductions,
                num_items,
                even_share,
                queue,
                reduction_op);
      CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());


      typedef AgentLauncher<
        ReduceAgent<T*, OutputIt, T, Size, ReductionOp> >
        reduce_agent_single;

      reduce_plan.grid_size = 1;
      reduce_agent_single ra1(reduce_plan, stream, vshmem_ptr, "reduce_agent: single tile reduce", debug_sync);

      ra1.launch(d_block_reductions, output_it, reduce_grid_size, reduction_op, init);
      CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
    }

    return status;
  }    // func doit_step


  template <typename Derived,
            typename InputIt,
            typename Size,
            typename T,
            typename BinaryOp>
  THRUST_RUNTIME_FUNCTION
  T reduce(execution_policy<Derived>& policy,
           InputIt                    first,
           Size                       num_items,
           T                          init,
           BinaryOp                   binary_op)
  {
    if (num_items == 0)
      return init;

    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    status = doit_step(NULL,
                       temp_storage_bytes,
                       first,
                       num_items,
                       init,
                       binary_op,
                       reinterpret_cast<T*>(NULL),
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "reduce failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(T*), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;
    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "reduce failed on 1st alias_storage");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "reduce failed on 2nd alias_storage");

    T* d_result = thrust::detail::aligned_reinterpret_cast<T*>(allocations[0]);

    status = doit_step(allocations[1],
                       temp_storage_bytes,
                       first,
                       num_items,
                       init,
                       binary_op,
                       d_result,
                       stream,
                       debug_sync);
    cuda_cub::throw_on_error(status, "reduce failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "reduce failed to synchronize");

    T result = cuda_cub::get_value(policy, d_result);

    return result;
  }
}    // namespace __reduce

namespace detail {

template <typename Derived,
          typename InputIt,
          typename Size,
          typename T,
          typename BinaryOp>
THRUST_RUNTIME_FUNCTION
T reduce_n_impl(execution_policy<Derived>& policy,
                InputIt                    first,
                Size                       num_items,
                T                          init,
                BinaryOp                   binary_op)
{
  cudaStream_t stream = cuda_cub::stream(policy);
  cudaError_t status;

  // Determine temporary device storage requirements.

  size_t tmp_size = 0;

  THRUST_INDEX_TYPE_DISPATCH2(status,
    cub::DeviceReduce::Reduce,
    (cub::DispatchReduce<
        InputIt, T*, Size, BinaryOp
    >::Dispatch),
    num_items,
    (NULL, tmp_size, first, reinterpret_cast<T*>(NULL),
        num_items_fixed, binary_op, init, stream,
        THRUST_DEBUG_SYNC_FLAG));
  cuda_cub::throw_on_error(status, "after reduction step 1");

  // Allocate temporary storage.

  thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
    tmp(policy, sizeof(T) + tmp_size);

  // Run reduction.

  // `tmp.begin()` yields a `normal_iterator`, which dereferences to a
  // `reference`, which has an `operator&` that returns a `pointer`, which
  // has a `.get` method that returns a raw pointer, which we can (finally)
  // `static_cast` to `void*`.
  //
  // The array was dynamically allocated, so we assume that it's suitably
  // aligned for any type of data. `malloc`/`cudaMalloc`/`new`/`std::allocator`
  // make this guarantee.
  T* ret_ptr = thrust::detail::aligned_reinterpret_cast<T*>(tmp.data().get());
  void* tmp_ptr = static_cast<void*>((tmp.data() + sizeof(T)).get());
  THRUST_INDEX_TYPE_DISPATCH2(status,
    cub::DeviceReduce::Reduce,
    (cub::DispatchReduce<
        InputIt, T*, Size, BinaryOp
    >::Dispatch),
    num_items,
    (tmp_ptr, tmp_size, first, ret_ptr,
        num_items_fixed, binary_op, init, stream,
        THRUST_DEBUG_SYNC_FLAG));
  cuda_cub::throw_on_error(status, "after reduction step 2");

  // Synchronize the stream and get the value.

  status = cuda_cub::synchronize(policy);
  cuda_cub::throw_on_error(status, "reduce failed to synchronize");

  // `tmp.begin()` yields a `normal_iterator`, which dereferences to a
  // `reference`, which has an `operator&` that returns a `pointer`, which
  // has a `.get` method that returns a raw pointer, which we can (finally)
  // `static_cast` to `void*`.
  //
  // The array was dynamically allocated, so we assume that it's suitably
  // aligned for any type of data. `malloc`/`cudaMalloc`/`new`/`std::allocator`
  // make this guarantee.
  return thrust::cuda_cub::get_value(policy,
    thrust::detail::aligned_reinterpret_cast<T*>(tmp.data().get()));
}

} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <typename Derived,
          typename InputIt,
          typename Size,
          typename T,
          typename BinaryOp>
__host__ __device__
T reduce_n(execution_policy<Derived>& policy,
           InputIt                    first,
           Size                       num_items,
           T                          init,
           BinaryOp                   binary_op)
{
  if (__THRUST_HAS_CUDART__)
    return thrust::cuda_cub::detail::reduce_n_impl(
      policy, first, num_items, init, binary_op);

  #if !__THRUST_HAS_CUDART__
    return thrust::reduce(
      cvt_to_seq(derived_cast(policy)), first, first + num_items, init, binary_op);
  #endif
}

template <class Derived, class InputIt, class T, class BinaryOp>
__host__ __device__
T reduce(execution_policy<Derived> &policy,
         InputIt                    first,
         InputIt                    last,
         T                          init,
         BinaryOp                   binary_op)
{
  typedef typename iterator_traits<InputIt>::difference_type size_type;
  // FIXME: Check for RA iterator.
  size_type num_items = static_cast<size_type>(thrust::distance(first, last));
  return cuda_cub::reduce_n(policy, first, num_items, init, binary_op);
}

template <class Derived,
          class InputIt,
          class T>
__host__ __device__
T reduce(execution_policy<Derived> &policy,
         InputIt                    first,
         InputIt                    last,
         T                          init)
{
  return cuda_cub::reduce(policy, first, last, init, plus<T>());
}

template <class Derived,
          class InputIt>
__host__ __device__
typename iterator_traits<InputIt>::value_type
reduce(execution_policy<Derived> &policy,
       InputIt                    first,
       InputIt                    last)
{
  typedef typename iterator_traits<InputIt>::value_type value_type;
  return cuda_cub::reduce(policy, first, last, value_type(0));
}


} // namespace cuda_cub

THRUST_NAMESPACE_END

#include <thrust/memory.h>
#include <thrust/reduce.h>

#endif
