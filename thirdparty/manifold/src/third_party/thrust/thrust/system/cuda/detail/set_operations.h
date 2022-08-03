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
#include <thrust/system/cuda/detail/util.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#include <thrust/system/cuda/detail/core/agent_launcher.h>
#include <thrust/system/cuda/detail/par_to_seq.h>
#include <thrust/system/cuda/detail/get_value.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/set_operations.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/distance.h>
#include <thrust/detail/alignment.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub {

namespace __set_operations {

  template <bool UpperBound,
            class IntT,
            class Size,
            class It,
            class T,
            class Comp>
  THRUST_DEVICE_FUNCTION void
  binary_search_iteration(It   data,
                          Size &begin,
                          Size &end,
                          T    key,
                          int  shift,
                          Comp comp)
  {

    IntT scale = (1 << shift) - 1;
    Size mid   = (begin + scale * end) >> shift;

    T    key2 = data[mid];
    bool pred = UpperBound ? !comp(key, key2) : comp(key2, key);
    if (pred)
      begin = mid + 1;
    else
      end = mid;
  }

  template <bool UpperBound, class Size, class T, class It, class Comp>
  THRUST_DEVICE_FUNCTION Size
  binary_search(It data, Size count, T key, Comp comp)
  {
    Size begin = 0;
    Size end   = count;
    while (begin < end)
      binary_search_iteration<UpperBound, int>(data,
                                               begin,
                                               end,
                                               key,
                                               1,
                                               comp);
    return begin;
  }

  template <bool UpperBound, class IntT, class Size, class T, class It, class Comp>
  THRUST_DEVICE_FUNCTION Size
  biased_binary_search(It data, Size count, T key, IntT levels, Comp comp)
  {
    Size begin = 0;
    Size end   = count;

    if (levels >= 4 && begin < end)
      binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 9, comp);
    if (levels >= 3 && begin < end)
      binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 7, comp);
    if (levels >= 2 && begin < end)
      binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 5, comp);
    if (levels >= 1 && begin < end)
      binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 4, comp);

    while (begin < end)
      binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 1, comp);
    return begin;
  }

  template <bool UpperBound, class Size, class It1, class It2, class Comp>
  THRUST_DEVICE_FUNCTION Size
  merge_path(It1 a, Size aCount, It2 b, Size bCount, Size diag, Comp comp)
  {
    typedef typename thrust::iterator_traits<It1>::value_type T;

    Size begin = thrust::max<Size>(0, diag - bCount);
    Size end   = thrust::min<Size>(diag, aCount);

    while (begin < end)
    {
      Size  mid  = (begin + end) >> 1;
      T    aKey = a[mid];
      T    bKey = b[diag - 1 - mid];
      bool pred = UpperBound ? comp(aKey, bKey) : !comp(bKey, aKey);
      if (pred)
        begin = mid + 1;
      else
        end = mid;
    }
    return begin;
  }

  template <class It1, class It2, class Size, class Size2, class CompareOp>
  THRUST_DEVICE_FUNCTION pair<Size, Size>
  balanced_path(It1       keys1,
                It2       keys2,
                Size      num_keys1,
                Size      num_keys2,
                Size      diag,
                Size2     levels,
                CompareOp compare_op)
  {
    typedef typename iterator_traits<It1>::value_type T;

    Size index1 = merge_path<false>(keys1,
                                    num_keys1,
                                    keys2,
                                    num_keys2,
                                    diag,
                                    compare_op);
    Size index2 = diag - index1;

    bool star = false;
    if (index2 < num_keys2)
    {
      T x = keys2[index2];

      // Search for the beginning of the duplicate run in both A and B.
      Size start1 = biased_binary_search<false>(keys1,
                                                index1,
                                                x,
                                                levels,
                                                compare_op);
      Size start2 = biased_binary_search<false>(keys2,
                                                index2,
                                                x,
                                                levels,
                                                compare_op);

      // The distance between x's merge path and its lower_bound is its rank.
      // We add up the a and b ranks and evenly distribute them to
      // get a stairstep path.
      Size run1      = index1 - start1;
      Size run2      = index2 - start2;
      Size total_run = run1 + run2;

      // Attempt to advance b and regress a.
      Size advance2 = max<Size>(total_run >> 1, total_run - run1);
      Size end2     = min<Size>(num_keys2, start2 + advance2 + 1);

      Size run_end2 = index2 + binary_search<true>(keys2 + index2,
                                                   end2 - index2,
                                                   x,
                                                   compare_op);
      run2 = run_end2 - start2;

      advance2      = min<Size>(advance2, run2);
      Size advance1 = total_run - advance2;

      bool round_up      = (advance1 == advance2 + 1) && (advance2 < run2);
      if (round_up) star = true;

      index1 = start1 + advance1;
    }
    return thrust::make_pair(index1, (diag - index1) + star);
  }    // func balanced_path

  template <int                      _BLOCK_THREADS,
            int                      _ITEMS_PER_THREAD = 1,
            cub::BlockLoadAlgorithm  _LOAD_ALGORITHM   = cub::BLOCK_LOAD_DIRECT,
            cub::CacheLoadModifier   _LOAD_MODIFIER    = cub::LOAD_LDG,
            cub::BlockScanAlgorithm  _SCAN_ALGORITHM   = cub::BLOCK_SCAN_WARP_SCANS>
  struct PtxPolicy
  {
    enum
    {
      BLOCK_THREADS    = _BLOCK_THREADS,
      ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
      ITEMS_PER_TILE   = _BLOCK_THREADS * _ITEMS_PER_THREAD - 1
    };

    static const cub::BlockLoadAlgorithm  LOAD_ALGORITHM  = _LOAD_ALGORITHM;
    static const cub::CacheLoadModifier   LOAD_MODIFIER   = _LOAD_MODIFIER;
    static const cub::BlockScanAlgorithm  SCAN_ALGORITHM  = _SCAN_ALGORITHM;
  };    // PtxPolicy

  template<class Arch, class T, class U>
  struct Tuning;

  namespace mpl = thrust::detail::mpl::math;

  template<class T, class U>
  struct Tuning<sm30,T,U>
  {
    enum
    {
      MAX_INPUT_BYTES             = mpl::max<size_t, sizeof(T), sizeof(U)>::value,
      COMBINED_INPUT_BYTES        = sizeof(T),    // + sizeof(Value),
      NOMINAL_4B_ITEMS_PER_THREAD = 7,
      ITEMS_PER_THREAD            = mpl::min<
          int,
          NOMINAL_4B_ITEMS_PER_THREAD,
          mpl::max<
              int,
              1,
              static_cast<int>(((NOMINAL_4B_ITEMS_PER_THREAD * 4) +
               COMBINED_INPUT_BYTES - 1) /
                  COMBINED_INPUT_BYTES)>::value>::value,
    };

    typedef PtxPolicy<128,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_DEFAULT,
                      cub::BLOCK_SCAN_WARP_SCANS>
        type;
  }; // tuning sm30

  template<class T, class U>
  struct Tuning<sm52,T,U>
  {
    enum
    {
      MAX_INPUT_BYTES             = mpl::max<size_t, sizeof(T), sizeof(U)>::value,
      COMBINED_INPUT_BYTES        = sizeof(T), // + sizeof(U),
      NOMINAL_4B_ITEMS_PER_THREAD = 15,
      ITEMS_PER_THREAD            = mpl::min<
          int,
          NOMINAL_4B_ITEMS_PER_THREAD,
          mpl::max<
              int,
              1,
              static_cast<int>(((NOMINAL_4B_ITEMS_PER_THREAD * 4) +
               COMBINED_INPUT_BYTES - 1) /
                  COMBINED_INPUT_BYTES)>::value>::value,
    };

    typedef PtxPolicy<256,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_DEFAULT,
                      cub::BLOCK_SCAN_WARP_SCANS>
        type;
  }; // tuning sm52

  template<class T, class U>
  struct Tuning<sm60,T,U>
  {
    enum
    {
      MAX_INPUT_BYTES             = mpl::max<size_t, sizeof(T), sizeof(U)>::value,
      COMBINED_INPUT_BYTES        = sizeof(T), // + sizeof(U),
      NOMINAL_4B_ITEMS_PER_THREAD = 19,
      ITEMS_PER_THREAD            = mpl::min<
          int,
          NOMINAL_4B_ITEMS_PER_THREAD,
          mpl::max<
              int,
              1,
              static_cast<int>(((NOMINAL_4B_ITEMS_PER_THREAD * 4) +
               COMBINED_INPUT_BYTES - 1) /
                  COMBINED_INPUT_BYTES)>::value>::value,
    };

    typedef PtxPolicy<512,
                      ITEMS_PER_THREAD,
                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                      cub::LOAD_DEFAULT,
                      cub::BLOCK_SCAN_WARP_SCANS>
        type;
  }; // tuning sm60

  template <class KeysIt1,
            class KeysIt2,
            class ValuesIt1,
            class ValuesIt2,
            class KeysOutputIt,
            class ValuesOutputIt,
            class Size,
            class CompareOp,
            class SetOp,
            class HAS_VALUES>
  struct SetOpAgent
  {
    typedef typename iterator_traits<KeysIt1>::value_type  key1_type;
    typedef typename iterator_traits<KeysIt2>::value_type  key2_type;
    typedef typename iterator_traits<ValuesIt1>::value_type value1_type;
    typedef typename iterator_traits<ValuesIt2>::value_type value2_type;

    typedef key1_type  key_type;
    typedef value1_type value_type;

    typedef cub::ScanTileState<Size> ScanTileState;

    template <class Arch>
    struct PtxPlan : Tuning<Arch, key_type, value_type>::type
    {
      typedef Tuning<Arch, key_type, value_type> tuning;

      typedef typename core::LoadIterator<PtxPlan, KeysIt1>::type   KeysLoadIt1;
      typedef typename core::LoadIterator<PtxPlan, KeysIt2>::type   KeysLoadIt2;
      typedef typename core::LoadIterator<PtxPlan, ValuesIt1>::type ValuesLoadIt1;
      typedef typename core::LoadIterator<PtxPlan, ValuesIt2>::type ValuesLoadIt2;

      typedef typename core::BlockLoad<PtxPlan, KeysLoadIt1>::type   BlockLoadKeys1;
      typedef typename core::BlockLoad<PtxPlan, KeysLoadIt2>::type   BlockLoadKeys2;
      typedef typename core::BlockLoad<PtxPlan, ValuesLoadIt1>::type BlockLoadValues1;
      typedef typename core::BlockLoad<PtxPlan, ValuesLoadIt2>::type BlockLoadValues2;

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

      // gather required temporary storage in a union
      //
      union TempStorage
      {
        struct ScanStorage
        {
          typename BlockScan::TempStorage          scan;
          typename TilePrefixCallback::TempStorage prefix;
        } scan_storage;

        struct LoadStorage
        {
          core::uninitialized_array<int, PtxPlan::BLOCK_THREADS> offset;
          union
          {
            // FIXME These don't appear to be used anywhere?
            typename BlockLoadKeys1::TempStorage   load_keys1;
            typename BlockLoadKeys2::TempStorage   load_keys2;
            typename BlockLoadValues1::TempStorage load_values1;
            typename BlockLoadValues2::TempStorage load_values2;

            // Allocate extra shmem than truely neccessary
            // This will permit to avoid range checks in
            // serial set operations, e.g. serial_set_difference
            core::uninitialized_array<
                key_type,
                PtxPlan::ITEMS_PER_TILE + PtxPlan::BLOCK_THREADS>
                keys_shared;

            core::uninitialized_array<
                value_type,
                PtxPlan::ITEMS_PER_TILE + PtxPlan::BLOCK_THREADS>
                values_shared;
          }; // anon union
        } load_storage; // struct LoadStorage
      };    // union TempStorage
    };      // struct PtxPlan

    typedef typename core::specialize_plan_msvc10_war<PtxPlan>::type::type ptx_plan;

    typedef typename ptx_plan::KeysLoadIt1   KeysLoadIt1;
    typedef typename ptx_plan::KeysLoadIt2   KeysLoadIt2;
    typedef typename ptx_plan::ValuesLoadIt1 ValuesLoadIt1;
    typedef typename ptx_plan::ValuesLoadIt2 ValuesLoadIt2;

    typedef typename ptx_plan::BlockLoadKeys1   BlockLoadKeys1;
    typedef typename ptx_plan::BlockLoadKeys2   BlockLoadKeys2;
    typedef typename ptx_plan::BlockLoadValues1 BlockLoadValues1;
    typedef typename ptx_plan::BlockLoadValues2 BlockLoadValues2;

    typedef typename ptx_plan::TilePrefixCallback TilePrefixCallback;
    typedef typename ptx_plan::BlockScan BlockScan;

    typedef typename ptx_plan::TempStorage TempStorage;

    enum
    {
      ITEMS_PER_THREAD = ptx_plan::ITEMS_PER_THREAD,
      BLOCK_THREADS    = ptx_plan::BLOCK_THREADS,
    };

    struct impl
    {
      //---------------------------------------------------------------------
      // Per-thread fields
      //---------------------------------------------------------------------

      TempStorage &  storage;
      ScanTileState &tile_state;
      KeysLoadIt1    keys1_in;
      KeysLoadIt2    keys2_in;
      ValuesLoadIt1  values1_in;
      ValuesLoadIt2  values2_in;
      Size           keys1_count;
      Size           keys2_count;
      KeysOutputIt   keys_out;
      ValuesOutputIt values_out;
      CompareOp      compare_op;
      SetOp          set_op;
      pair<Size, Size> *partitions;
      std::size_t *output_count;

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
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD - 1; ++ITEM)
          {
            int idx      = BLOCK_THREADS * ITEM + threadIdx.x;
            output[ITEM] = (idx < count1)
                               ? static_cast<T>(input1[idx])
                               : static_cast<T>(input2[idx - count1]);
          }

          // last ITEM might be a conditional load even for full tiles
          // please check first before attempting to load.
          int ITEM = ITEMS_PER_THREAD - 1;
          int idx  = BLOCK_THREADS * ITEM + threadIdx.x;
          if (idx < count1 + count2)
            output[ITEM] = (idx < count1)
                               ? static_cast<T>(input1[idx])
                               : static_cast<T>(input2[idx - count1]);
        }
        else
        {
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            int idx = BLOCK_THREADS * ITEM + threadIdx.x;
            if (idx < count1 + count2)
            {
              output[ITEM] = (idx < count1)
                                 ? static_cast<T>(input1[idx])
                                 : static_cast<T>(input2[idx - count1]);
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

      template <class OutputIt, class T, class SharedIt>
      void THRUST_DEVICE_FUNCTION
      scatter(OutputIt output,
              T (&input)[ITEMS_PER_THREAD],
              SharedIt shared,
              int      active_mask,
              Size     thread_output_prefix,
              Size     tile_output_prefix,
              int      tile_output_count)
      {
        using core::sync_threadblock;



        int local_scatter_idx = thread_output_prefix - tile_output_prefix;
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
          if (active_mask & (1 << ITEM))
          {
            shared[local_scatter_idx++] = input[ITEM];
          }
        }
        sync_threadblock();

        for (int item = threadIdx.x;
             item < tile_output_count;
             item += BLOCK_THREADS)
        {
          output[tile_output_prefix + item] = shared[item];
        }
      }

      int THRUST_DEVICE_FUNCTION
      serial_set_op(key_type *keys,
                    int       keys1_beg,
                    int       keys2_beg,
                    int       keys1_count,
                    int       keys2_count,
                    key_type (&output)[ITEMS_PER_THREAD],
                    int (&indices)[ITEMS_PER_THREAD],
                    CompareOp compare_op,
                    SetOp     set_op)
      {
        int active_mask = set_op(keys,
                                 keys1_beg,
                                 keys2_beg,
                                 keys1_count,
                                 keys2_count,
                                 output,
                                 indices,
                                 compare_op);

        return active_mask;
      }

      //---------------------------------------------------------------------
      // Tile operations
      //---------------------------------------------------------------------

      template <bool IS_LAST_TILE>
      void THRUST_DEVICE_FUNCTION
      consume_tile(Size tile_idx)
      {
        using core::sync_threadblock;
        using core::uninitialized_array;

        pair<Size, Size> partition_beg = partitions[tile_idx + 0];
        pair<Size, Size> partition_end = partitions[tile_idx + 1];

        Size keys1_beg = partition_beg.first;
        Size keys1_end = partition_end.first;
        Size keys2_beg = partition_beg.second;
        Size keys2_end = partition_end.second;

        // number of keys per tile
        //
        int num_keys1 = static_cast<int>(keys1_end - keys1_beg);
        int num_keys2 = static_cast<int>(keys2_end - keys2_beg);


       // load keys into shared memory for further processing
        key_type keys_loc[ITEMS_PER_THREAD];

        gmem_to_reg<!IS_LAST_TILE>(keys_loc,
                                   keys1_in + keys1_beg,
                                   keys2_in + keys2_beg,
                                   num_keys1,
                                   num_keys2);

        reg_to_shared(&storage.load_storage.keys_shared[0], keys_loc);

        sync_threadblock();

        int diag_loc = min<int>(ITEMS_PER_THREAD * threadIdx.x,
                                num_keys1 + num_keys2);

        pair<int, int> partition_loc =
            balanced_path(&storage.load_storage.keys_shared[0],
                          &storage.load_storage.keys_shared[num_keys1],
                          num_keys1,
                          num_keys2,
                          diag_loc,
                          4,
                          compare_op);

        int keys1_beg_loc = partition_loc.first;
        int keys2_beg_loc = partition_loc.second;

        // compute difference between next and current thread
        // to obtain number of elements per thread
        int value = threadIdx.x == 0
                        ? (num_keys1 << 16) | num_keys2
                        : (partition_loc.first << 16) | partition_loc.second;

        int dst = threadIdx.x == 0 ? BLOCK_THREADS - 1 : threadIdx.x - 1;
        storage.load_storage.offset[dst] = value;

        core::sync_threadblock();

        pair<int,int> partition1_loc = thrust::make_pair(
          storage.load_storage.offset[threadIdx.x] >> 16,
          storage.load_storage.offset[threadIdx.x] & 0xFFFF);

        int keys1_end_loc = partition1_loc.first;
        int keys2_end_loc = partition1_loc.second;

        int num_keys1_loc = keys1_end_loc - keys1_beg_loc;
        int num_keys2_loc = keys2_end_loc - keys2_beg_loc;

        // perform serial set operation
        //
        int indices[ITEMS_PER_THREAD];

        int active_mask = serial_set_op(&storage.load_storage.keys_shared[0],
                                        keys1_beg_loc,
                                        keys2_beg_loc + num_keys1,
                                        num_keys1_loc,
                                        num_keys2_loc,
                                        keys_loc,
                                        indices,
                                        compare_op,
                                        set_op);
        sync_threadblock();
#if 0
        if (ITEMS_PER_THREAD*threadIdx.x >= num_keys1 + num_keys2)
          active_mask = 0;
#endif

        // look-back scan over thread_output_count
        // to compute global thread_output_base and tile_otput_count;
        Size tile_output_count    = 0;
        Size thread_output_prefix = 0;
        Size tile_output_prefix   = 0;
        Size thread_output_count = static_cast<Size>(__popc(active_mask));

        if (tile_idx == 0)    // first tile
        {
          BlockScan(storage.scan_storage.scan)
              .ExclusiveSum(thread_output_count,
                            thread_output_prefix,
                            tile_output_count);
          if (threadIdx.x == 0)
          {
            // Update tile status if this is not the last tile
            if (!IS_LAST_TILE)
            {
              tile_state.SetInclusive(0, tile_output_count);
            }
          }
        }
        else
        {
          TilePrefixCallback prefix_cb(tile_state,
                                       storage.scan_storage.prefix,
                                       cub::Sum(),
                                       tile_idx);

          BlockScan(storage.scan_storage.scan)
              .ExclusiveSum(thread_output_count,
                            thread_output_prefix,
                            prefix_cb);
          tile_output_count  = prefix_cb.GetBlockAggregate();
          tile_output_prefix = prefix_cb.GetExclusivePrefix();
        }

        sync_threadblock();

        // scatter results
        //
        scatter(keys_out,
                keys_loc,
                &storage.load_storage.keys_shared[0],
                active_mask,
                thread_output_prefix,
                tile_output_prefix,
                tile_output_count);

        if (HAS_VALUES::value)
        {
          value_type values_loc[ITEMS_PER_THREAD];
          gmem_to_reg<!IS_LAST_TILE>(values_loc,
                                     values1_in + keys1_beg,
                                     values2_in + keys2_beg,
                                     num_keys1,
                                     num_keys2);

          sync_threadblock();

          reg_to_shared(&storage.load_storage.values_shared[0], values_loc);

          sync_threadblock();

          // gather items from shared mem
          //
#pragma unroll
          for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
          {
            if (active_mask & (1 << ITEM))
            {
              values_loc[ITEM] = storage.load_storage.values_shared[indices[ITEM]];
            }
          }

          sync_threadblock();

          scatter(values_out,
                  values_loc,
                  &storage.load_storage.values_shared[0],
                  active_mask,
                  thread_output_prefix,
                  tile_output_prefix,
                  tile_output_count);
        }

        if (IS_LAST_TILE && threadIdx.x == 0)
        {
          *output_count = tile_output_prefix + tile_output_count;
        }
      }

      //---------------------------------------------------------------------
      // Constructor
      //---------------------------------------------------------------------

      THRUST_DEVICE_FUNCTION
      impl(TempStorage &  storage_,
           ScanTileState &tile_state_,
           KeysIt1        keys1_,
           KeysIt2        keys2_,
           ValuesIt1      values1_,
           ValuesIt2      values2_,
           Size           keys1_count_,
           Size           keys2_count_,
           KeysOutputIt   keys_out_,
           ValuesOutputIt values_out_,
           CompareOp      compare_op_,
           SetOp          set_op_,
           pair<Size, Size> *partitions_,
           std::size_t * output_count_)
          : storage(storage_),
            tile_state(tile_state_),
            keys1_in(core::make_load_iterator(ptx_plan(), keys1_)),
            keys2_in(core::make_load_iterator(ptx_plan(), keys2_)),
            values1_in(core::make_load_iterator(ptx_plan(), values1_)),
            values2_in(core::make_load_iterator(ptx_plan(), values2_)),
            keys1_count(keys1_count_),
            keys2_count(keys2_count_),
            keys_out(keys_out_),
            values_out(values_out_),
            compare_op(compare_op_),
            set_op(set_op_),
            partitions(partitions_),
            output_count(output_count_)
      {
        int  tile_idx      = blockIdx.x;
        int  num_tiles     = gridDim.x;

        if (tile_idx < num_tiles-1)
        {
          consume_tile<false>(tile_idx);
        }
        else
        {
          consume_tile<true>(tile_idx);
        }
      }
    };    // struct impl

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    THRUST_AGENT_ENTRY(KeysIt1        keys1,
                       KeysIt2        keys2,
                       ValuesIt1      values1,
                       ValuesIt2      values2,
                       Size           keys1_count,
                       Size           keys2_count,
                       KeysOutputIt   keys_output,
                       ValuesOutputIt values_output,
                       CompareOp      compare_op,
                       SetOp          set_op,
                       pair<Size, Size> *partitions,
                       std::size_t *  output_count,
                       ScanTileState tile_state,
                       char *        shmem)
    {
      TempStorage &storage = *reinterpret_cast<TempStorage *>(shmem);

      impl(storage,
           tile_state,
           keys1,
           keys2,
           values1,
           values2,
           keys1_count,
           keys2_count,
           keys_output,
           values_output,
           compare_op,
           set_op,
           partitions,
           output_count);
    }
  };    // struct SetOpAgent

  template <class KeysIt1,
            class KeysIt2,
            class Size,
            class CompareOp>
  struct PartitionAgent
  {
    template <class Arch>
    struct PtxPlan : PtxPolicy<256> {};

    typedef core::specialize_plan<PtxPlan> ptx_plan;

    //---------------------------------------------------------------------
    // Agent entry point
    //---------------------------------------------------------------------

    THRUST_AGENT_ENTRY(KeysIt1 keys1,
                       KeysIt2 keys2,
                       Size    keys1_count,
                       Size    keys2_count,
                       Size    num_partitions,
                       pair<Size, Size> *partitions,
                       CompareOp compare_op,
                       int       items_per_tile,
                       char * /*shmem*/)
    {
      Size partition_idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (partition_idx < num_partitions)
      {
        Size partition_at = min<Size>(partition_idx * items_per_tile,
                                      keys1_count + keys2_count);
        pair<Size, Size> diag = balanced_path(keys1,
                                              keys2,
                                              keys1_count,
                                              keys2_count,
                                              partition_at,
                                              4ll,
                                              compare_op);
        partitions[partition_idx] = diag;
      }
    }
  };    // struct PartitionAgent

  template <class ScanTileState,
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
                       char * /*shmem*/)
    {
      tile_state.InitializeStatus(num_tiles);
    }
  }; // struct InitAgent

  //---------------------------------------------------------------------
  // Serial set operations
  //---------------------------------------------------------------------

  // serial_set_intersection
  // -----------------------
  // emit A if A and B are in range and equal.
  struct serial_set_intersection
  {
    // max_input_size <= 32
    template <class T, class CompareOp, int ITEMS_PER_THREAD>
    int THRUST_DEVICE_FUNCTION
    operator()(T * keys,
               int keys1_beg,
               int keys2_beg,
               int keys1_count,
               int keys2_count,
               T (&output)[ITEMS_PER_THREAD],
               int (&indices)[ITEMS_PER_THREAD],
               CompareOp compare_op)
    {
      int active_mask = 0;

      int aBegin = keys1_beg;
      int bBegin = keys2_beg;
      int aEnd   = keys1_beg + keys1_count;
      int bEnd   = keys2_beg + keys2_count;

      T aKey = keys[aBegin];
      T bKey = keys[bBegin];

#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; ++i)
      {
        bool pA = compare_op(aKey, bKey);
        bool pB = compare_op(bKey, aKey);

        // The outputs must come from A by definition of set interection.
        output[i]  = aKey;
        indices[i] = aBegin;

        if ((aBegin < aEnd) && (bBegin < bEnd) && pA == pB)
          active_mask |= 1 << i;

        if (!pB) {aKey = keys[++aBegin]; }
        if (!pA) {bKey = keys[++bBegin]; }
      }
      return active_mask;
    }
  };    // struct serial_set_intersection

  // serial_set_symmetric_difference
  // ---------------------
  // emit A if A < B and emit B if B < A.
  struct serial_set_symmetric_difference
  {
    // max_input_size <= 32
    template <class T, class CompareOp, int ITEMS_PER_THREAD>
    int THRUST_DEVICE_FUNCTION
    operator()(T * keys,
               int keys1_beg,
               int keys2_beg,
               int keys1_count,
               int keys2_count,
               T (&output)[ITEMS_PER_THREAD],
               int (&indices)[ITEMS_PER_THREAD],
               CompareOp compare_op)
    {
      int active_mask = 0;

      int aBegin = keys1_beg;
      int bBegin = keys2_beg;
      int aEnd   = keys1_beg + keys1_count;
      int bEnd   = keys2_beg + keys2_count;
      int end    = aEnd + bEnd;

      T aKey = keys[aBegin];
      T bKey = keys[bBegin];


#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; ++i)
      {
        bool pB = aBegin >= aEnd;
        bool pA = !pB && bBegin >= bEnd;

        if (!pA && !pB)
        {
          pA = compare_op(aKey, bKey);
          pB = !pA && compare_op(bKey, aKey);
        }

        // The outputs must come from A by definition of set difference.
        output[i]  = pA ? aKey : bKey;
        indices[i] = pA ? aBegin : bBegin;

        if (aBegin + bBegin < end && pA != pB)
          active_mask |= 1 << i;

        if (!pB) {aKey = keys[++aBegin]; }
        if (!pA) {bKey = keys[++bBegin]; }

      }
      return active_mask;
    }
  };    // struct set_symmetric_difference

  // serial_set_difference
  // ---------------------
  // emit A if A < B
  struct serial_set_difference
  {
    // max_input_size <= 32
    template <class T, class CompareOp, int ITEMS_PER_THREAD>
    int THRUST_DEVICE_FUNCTION
    operator()(T * keys,
               int keys1_beg,
               int keys2_beg,
               int keys1_count,
               int keys2_count,
               T (&output)[ITEMS_PER_THREAD],
               int (&indices)[ITEMS_PER_THREAD],
               CompareOp compare_op)
    {
      int active_mask = 0;

      int aBegin = keys1_beg;
      int bBegin = keys2_beg;
      int aEnd   = keys1_beg + keys1_count;
      int bEnd   = keys2_beg + keys2_count;
      int end    = aEnd + bEnd;

      T aKey = keys[aBegin];
      T bKey = keys[bBegin];

#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; ++i)
      {
        bool pB = aBegin >= aEnd;
        bool pA = !pB && bBegin >= bEnd;

        if (!pA && !pB)
        {
          pA = compare_op(aKey, bKey);
          pB = !pA && compare_op(bKey, aKey);
        }

        // The outputs must come from A by definition of set difference.
        output[i]  = aKey;
        indices[i] = aBegin;

        if (aBegin + bBegin < end && pA)
          active_mask |= 1 << i;

        if (!pB) { aKey = keys[++aBegin]; }
        if (!pA) { bKey = keys[++bBegin]; }
      }
      return active_mask;
    }
  };    // struct set_difference

  // serial_set_union
  // ----------------
  // emit A if A <= B else emit B
  struct serial_set_union
  {
    // max_input_size <= 32
    template <class T, class CompareOp, int ITEMS_PER_THREAD>
    int THRUST_DEVICE_FUNCTION
    operator()(T * keys,
               int keys1_beg,
               int keys2_beg,
               int keys1_count,
               int keys2_count,
               T (&output)[ITEMS_PER_THREAD],
               int (&indices)[ITEMS_PER_THREAD],
               CompareOp compare_op)
    {
      int active_mask = 0;

      int aBegin = keys1_beg;
      int bBegin = keys2_beg;
      int aEnd   = keys1_beg + keys1_count;
      int bEnd   = keys2_beg + keys2_count;
      int end    = aEnd + bEnd;

      T aKey = keys[aBegin];
      T bKey = keys[bBegin];

#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; ++i)
      {
        bool pB = aBegin >= aEnd;
        bool pA = !pB && bBegin >= bEnd;

        if (!pA && !pB)
        {
          pA = compare_op(aKey, bKey);
          pB = !pA && compare_op(bKey, aKey);
        }

        // Output A in case of a tie, so check if b < a.
        output[i]  = pB ? bKey : aKey;
        indices[i] = pB ? bBegin : aBegin;

        if (aBegin + bBegin < end)
          active_mask |= 1 << i;

        if (!pB) { aKey = keys[++aBegin]; }
        if (!pA) { bKey = keys[++bBegin]; }

      }
      return active_mask;
    }
  };    // struct set_union

  template <class HAS_VALUES,
            class KeysIt1,
            class KeysIt2,
            class ValuesIt1,
            class ValuesIt2,
            class Size,
            class KeysOutputIt,
            class ValuesOutputIt,
            class CompareOp,
            class SetOp>
  cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *         d_temp_storage,
            size_t &       temp_storage_size,
            KeysIt1        keys1,
            KeysIt2        keys2,
            ValuesIt1      values1,
            ValuesIt2      values2,
            Size           num_keys1,
            Size           num_keys2,
            KeysOutputIt   keys_output,
            ValuesOutputIt values_output,
            std::size_t *  output_count,
            CompareOp      compare_op,
            SetOp          set_op,
            cudaStream_t   stream,
            bool           debug_sync)
  {
    Size keys_total = num_keys1 + num_keys2;
    if (keys_total == 0)
      return cudaErrorNotSupported;

    cudaError_t status = cudaSuccess;

    using core::AgentPlan;
    using core::AgentLauncher;

    typedef AgentLauncher<
        SetOpAgent<KeysIt1,
                   KeysIt2,
                   ValuesIt1,
                   ValuesIt2,
                   KeysOutputIt,
                   ValuesOutputIt,
                   Size,
                   CompareOp,
                   SetOp,
                   HAS_VALUES> >
        set_op_agent;

    typedef AgentLauncher<PartitionAgent<KeysIt1, KeysIt2, Size, CompareOp> >
        partition_agent;

    typedef typename set_op_agent::ScanTileState ScanTileState;
    typedef AgentLauncher<InitAgent<ScanTileState, Size> > init_agent;


    AgentPlan set_op_plan    = set_op_agent::get_plan(stream);
    AgentPlan init_plan      = init_agent::get_plan();
    AgentPlan partition_plan = partition_agent::get_plan();

    int  tile_size = set_op_plan.items_per_tile;
    Size num_tiles = (keys_total + tile_size - 1) / tile_size;

    size_t tile_agent_storage;
    status = ScanTileState::AllocationSize(static_cast<int>(num_tiles),
                                           tile_agent_storage);
    CUDA_CUB_RET_IF_FAIL(status);

    size_t vshmem_storage = core::vshmem_size(set_op_plan.shared_memory_size,
                                              num_tiles);
    size_t partition_agent_storage = (num_tiles + 1) * sizeof(Size) * 2;

    void *allocations[3] = {NULL, NULL, NULL};
    size_t allocation_sizes[3] = {tile_agent_storage,
                                  partition_agent_storage,
                                  vshmem_storage};

    status = core::alias_storage(d_temp_storage,
                                 temp_storage_size,
                                 allocations,
                                 allocation_sizes);
    CUDA_CUB_RET_IF_FAIL(status);

    if (d_temp_storage == NULL)
    {
      return status;
    }

    ScanTileState tile_state;
    status = tile_state.Init(static_cast<int>(num_tiles),
                             allocations[0],
                             allocation_sizes[0]);
    CUDA_CUB_RET_IF_FAIL(status);

    pair<Size, Size> *partitions = (pair<Size, Size> *)allocations[1];
    char *vshmem_ptr = vshmem_storage > 0 ? (char *)allocations[2] : NULL;

    init_agent ia(init_plan, num_tiles, stream, "set_op::init_agent", debug_sync);
    ia.launch(tile_state, num_tiles);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    partition_agent pa(partition_plan, num_tiles+1, stream, "set_op::partition agent", debug_sync);
    pa.launch(keys1,
              keys2,
              num_keys1,
              num_keys2,
              num_tiles+1,
              partitions,
              compare_op,
              tile_size);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    set_op_agent sa(set_op_plan, keys_total, stream, vshmem_ptr, "set_op::set_op_agent", debug_sync);
    sa.launch(keys1,
              keys2,
              values1,
              values2,
              num_keys1,
              num_keys2,
              keys_output,
              values_output,
              compare_op,
              set_op,
              partitions,
              output_count,
              tile_state);
    CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    return status;
 }

 template <typename HAS_VALUES,
           typename Derived,
           typename KeysIt1,
           typename KeysIt2,
           typename ValuesIt1,
           typename ValuesIt2,
           typename KeysOutputIt,
           typename ValuesOutputIt,
           typename CompareOp,
           typename SetOp>
  THRUST_RUNTIME_FUNCTION
  pair<KeysOutputIt, ValuesOutputIt>
  set_operations(execution_policy<Derived>& policy,
                 KeysIt1                    keys1_first,
                 KeysIt1                    keys1_last,
                 KeysIt2                    keys2_first,
                 KeysIt2                    keys2_last,
                 ValuesIt1                  values1_first,
                 ValuesIt2                  values2_first,
                 KeysOutputIt               keys_output,
                 ValuesOutputIt             values_output,
                 CompareOp                  compare_op,
                 SetOp                      set_op)
  {
    typedef typename iterator_traits<KeysIt1>::difference_type size_type;

    size_type num_keys1 = static_cast<size_type>(thrust::distance(keys1_first, keys1_last));
    size_type num_keys2 = static_cast<size_type>(thrust::distance(keys2_first, keys2_last));

    if (num_keys1 + num_keys2 == 0)
      return thrust::make_pair(keys_output, values_output);

    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    THRUST_DOUBLE_INDEX_TYPE_DISPATCH(status, doit_step<HAS_VALUES>,
        num_keys1, num_keys2, (NULL,
                                   temp_storage_bytes,
                                   keys1_first,
                                   keys2_first,
                                   values1_first,
                                   values2_first,
                                   num_keys1_fixed,
                                   num_keys2_fixed,
                                   keys_output,
                                   values_output,
                                   reinterpret_cast<std::size_t*>(NULL),
                                   compare_op,
                                   set_op,
                                   stream,
                                   debug_sync));
    cuda_cub::throw_on_error(status, "set_operations failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(std::size_t), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;

    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "set_operations failed on 1st alias_storage");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "set_operations failed on 2nd alias_storage");

    std::size_t* d_output_count
      = thrust::detail::aligned_reinterpret_cast<std::size_t*>(allocations[0]);

    THRUST_DOUBLE_INDEX_TYPE_DISPATCH(status, doit_step<HAS_VALUES>,
        num_keys1, num_keys2, (allocations[1],
                                   temp_storage_bytes,
                                   keys1_first,
                                   keys2_first,
                                   values1_first,
                                   values2_first,
                                   num_keys1_fixed,
                                   num_keys2_fixed,
                                   keys_output,
                                   values_output,
                                   d_output_count,
                                   compare_op,
                                   set_op,
                                   stream,
                                   debug_sync));
    cuda_cub::throw_on_error(status, "set_operations failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "set_operations failed to synchronize");

    std::size_t output_count = cuda_cub::get_value(policy, d_output_count);

    return thrust::make_pair(keys_output + output_count, values_output + output_count);
  }
}    // namespace __set_operations

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt1,
          class ItemsIt2,
          class OutputIt,
          class CompareOp>
OutputIt __host__ __device__
set_difference(execution_policy<Derived> &policy,
               ItemsIt1                   items1_first,
               ItemsIt1                   items1_last,
               ItemsIt2                   items2_first,
               ItemsIt2                   items2_last,
               OutputIt                   result,
               CompareOp                  compare)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    typename thrust::iterator_value<ItemsIt1>::type *null_ = NULL;
    //
    ret = __set_operations::set_operations<thrust::detail::false_type>(
              policy,
              items1_first,
              items1_last,
              items2_first,
              items2_last,
              null_,
              null_,
              result,
              null_,
              compare,
              __set_operations::serial_set_difference())
              .first;
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::set_difference(cvt_to_seq(derived_cast(policy)),
                                 items1_first,
                                 items1_last,
                                 items2_first,
                                 items2_last,
                                 result,
                                 compare);
#endif
  }
  return ret;
}

template <class Derived,
          class ItemsIt1,
          class ItemsIt2,
          class OutputIt>
OutputIt __host__ __device__
set_difference(execution_policy<Derived> &policy,
               ItemsIt1                   items1_first,
               ItemsIt1                   items1_last,
               ItemsIt2                   items2_first,
               ItemsIt2                   items2_last,
               OutputIt                   result)
{
  typedef typename thrust::iterator_value<ItemsIt1>::type value_type;
  return cuda_cub::set_difference(policy,
                                  items1_first,
                                  items1_last,
                                  items2_first,
                                  items2_last,
                                  result,
                                  less<value_type>());
}

/*****************************/


__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt1,
          class ItemsIt2,
          class OutputIt,
          class CompareOp>
OutputIt __host__ __device__
set_intersection(execution_policy<Derived> &policy,
                 ItemsIt1                   items1_first,
                 ItemsIt1                   items1_last,
                 ItemsIt2                   items2_first,
                 ItemsIt2                   items2_last,
                 OutputIt                   result,
                 CompareOp                  compare)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    typename thrust::iterator_value<ItemsIt1>::type *null_ = NULL;
    //
    ret = __set_operations::set_operations<thrust::detail::false_type>(
              policy,
              items1_first,
              items1_last,
              items2_first,
              items2_last,
              null_,
              null_,
              result,
              null_,
              compare,
              __set_operations::serial_set_intersection())
              .first;
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::set_intersection(cvt_to_seq(derived_cast(policy)),
                                   items1_first,
                                   items1_last,
                                   items2_first,
                                   items2_last,
                                   result,
                                   compare);
#endif
  }
  return ret;
}

template <class Derived,
          class ItemsIt1,
          class ItemsIt2,
          class OutputIt>
OutputIt __host__ __device__
set_intersection(execution_policy<Derived> &policy,
                 ItemsIt1                   items1_first,
                 ItemsIt1                   items1_last,
                 ItemsIt2                   items2_first,
                 ItemsIt2                   items2_last,
                 OutputIt                   result)
{
  typedef typename thrust::iterator_value<ItemsIt1>::type value_type;
  return cuda_cub::set_intersection(policy,
                                    items1_first,
                                    items1_last,
                                    items2_first,
                                    items2_last,
                                    result,
                                    less<value_type>());
}


/*****************************/

__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt1,
          class ItemsIt2,
          class OutputIt,
          class CompareOp>
OutputIt __host__ __device__
set_symmetric_difference(execution_policy<Derived> &policy,
                         ItemsIt1                   items1_first,
                         ItemsIt1                   items1_last,
                         ItemsIt2                   items2_first,
                         ItemsIt2                   items2_last,
                         OutputIt                   result,
                         CompareOp                  compare)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    typename thrust::iterator_value<ItemsIt1>::type *null_ = NULL;
    //
    ret = __set_operations::set_operations<thrust::detail::false_type>(
              policy,
              items1_first,
              items1_last,
              items2_first,
              items2_last,
              null_,
              null_,
              result,
              null_,
              compare,
              __set_operations::serial_set_symmetric_difference())
              .first;
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::set_symmetric_difference(cvt_to_seq(derived_cast(policy)),
                                           items1_first,
                                           items1_last,
                                           items2_first,
                                           items2_last,
                                           result,
                                           compare);
#endif
  }
  return ret;
}


template <class Derived,
          class ItemsIt1,
          class ItemsIt2,
          class OutputIt>
OutputIt __host__ __device__
set_symmetric_difference(execution_policy<Derived> &policy,
                         ItemsIt1                   items1_first,
                         ItemsIt1                   items1_last,
                         ItemsIt2                   items2_first,
                         ItemsIt2                   items2_last,
                         OutputIt                   result)
{
  typedef typename thrust::iterator_value<ItemsIt1>::type value_type;
  return cuda_cub::set_symmetric_difference(policy,
                                            items1_first,
                                            items1_last,
                                            items2_first,
                                            items2_last,
                                            result,
                                            less<value_type>());
}

/*****************************/

__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt1,
          class ItemsIt2,
          class OutputIt,
          class CompareOp>
OutputIt __host__ __device__
set_union(execution_policy<Derived> &policy,
          ItemsIt1                   items1_first,
          ItemsIt1                   items1_last,
          ItemsIt2                   items2_first,
          ItemsIt2                   items2_last,
          OutputIt                   result,
          CompareOp                  compare)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    typename thrust::iterator_value<ItemsIt1>::type *null_ = NULL;
    //
    ret = __set_operations::set_operations<thrust::detail::false_type>(
              policy,
              items1_first,
              items1_last,
              items2_first,
              items2_last,
              null_,
              null_,
              result,
              null_,
              compare,
              __set_operations::serial_set_union())
              .first;
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::set_union(cvt_to_seq(derived_cast(policy)),
                            items1_first,
                            items1_last,
                            items2_first,
                            items2_last,
                            result,
                            compare);
#endif
  }
  return ret;
}


template <class Derived,
          class ItemsIt1,
          class ItemsIt2,
          class OutputIt>
OutputIt __host__ __device__
set_union(execution_policy<Derived> &policy,
          ItemsIt1                   items1_first,
          ItemsIt1                   items1_last,
          ItemsIt2                   items2_first,
          ItemsIt2                   items2_last,
          OutputIt                   result)
{
  typedef typename thrust::iterator_value<ItemsIt1>::type value_type;
  return cuda_cub::set_union(policy,
                             items1_first,
                             items1_last,
                             items2_first,
                             items2_last,
                             result,
                             less<value_type>());
}


/*****************************/
/*****************************/
/*****     *_by_key      *****/
/*****************************/
/*****************************/

/*****************************/

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
set_difference_by_key(execution_policy<Derived> &policy,
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
    ret = __set_operations::set_operations<thrust::detail::true_type>(
        policy,
        keys1_first,
        keys1_last,
        keys2_first,
        keys2_last,
        items1_first,
        items2_first,
        keys_result,
        items_result,
        compare_op,
        __set_operations::serial_set_difference());
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::set_difference_by_key(cvt_to_seq(derived_cast(policy)),
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
set_difference_by_key(execution_policy<Derived> &policy,
                      KeysIt1                    keys1_first,
                      KeysIt1                    keys1_last,
                      KeysIt2                    keys2_first,
                      KeysIt2                    keys2_last,
                      ItemsIt1                   items1_first,
                      ItemsIt2                   items2_first,
                      KeysOutputIt               keys_result,
                      ItemsOutputIt              items_result)
{
  typedef typename thrust::iterator_value<KeysIt1>::type value_type;
  return cuda_cub::set_difference_by_key(policy,
                                         keys1_first,
                                         keys1_last,
                                         keys2_first,
                                         keys2_last,
                                         items1_first,
                                         items2_first,
                                         keys_result,
                                         items_result,
                                         less<value_type>());
}

/*****************************/

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
set_intersection_by_key(execution_policy<Derived> &policy,
                        KeysIt1                    keys1_first,
                        KeysIt1                    keys1_last,
                        KeysIt2                    keys2_first,
                        KeysIt2                    keys2_last,
                        ItemsIt1                   items1_first,
                        KeysOutputIt               keys_result,
                        ItemsOutputIt              items_result,
                        CompareOp                  compare_op)
{
  pair<KeysOutputIt, ItemsOutputIt> ret = thrust::make_pair(keys_result, items_result);
  if (__THRUST_HAS_CUDART__)
  {
    ret = __set_operations::set_operations<thrust::detail::true_type>(
        policy,
        keys1_first,
        keys1_last,
        keys2_first,
        keys2_last,
        items1_first,
        items1_first,
        keys_result,
        items_result,
        compare_op,
        __set_operations::serial_set_intersection());
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::set_intersection_by_key(cvt_to_seq(derived_cast(policy)),
                                          keys1_first,
                                          keys1_last,
                                          keys2_first,
                                          keys2_last,
                                          items1_first,
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
set_intersection_by_key(execution_policy<Derived> &policy,
                        KeysIt1                    keys1_first,
                        KeysIt1                    keys1_last,
                        KeysIt2                    keys2_first,
                        KeysIt2                    keys2_last,
                        ItemsIt1                   items1_first,
                        KeysOutputIt               keys_result,
                        ItemsOutputIt              items_result)
{
  typedef typename thrust::iterator_value<KeysIt1>::type value_type;
  return cuda_cub::set_intersection_by_key(policy,
                                           keys1_first,
                                           keys1_last,
                                           keys2_first,
                                           keys2_last,
                                           items1_first,
                                           keys_result,
                                           items_result,
                                           less<value_type>());
}

/*****************************/

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
set_symmetric_difference_by_key(execution_policy<Derived> &policy,
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
    ret = __set_operations::set_operations<thrust::detail::true_type>(
        policy,
        keys1_first,
        keys1_last,
        keys2_first,
        keys2_last,
        items1_first,
        items2_first,
        keys_result,
        items_result,
        compare_op,
        __set_operations::serial_set_symmetric_difference());
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::set_symmetric_difference_by_key(cvt_to_seq(derived_cast(policy)),
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
set_symmetric_difference_by_key(execution_policy<Derived> &policy,
                                KeysIt1                    keys1_first,
                                KeysIt1                    keys1_last,
                                KeysIt2                    keys2_first,
                                KeysIt2                    keys2_last,
                                ItemsIt1                   items1_first,
                                ItemsIt2                   items2_first,
                                KeysOutputIt               keys_result,
                                ItemsOutputIt              items_result)
{
  typedef typename thrust::iterator_value<KeysIt1>::type value_type;
  return cuda_cub::set_symmetric_difference_by_key(policy,
                                                   keys1_first,
                                                   keys1_last,
                                                   keys2_first,
                                                   keys2_last,
                                                   items1_first,
                                                   items2_first,
                                                   keys_result,
                                                   items_result,
                                                   less<value_type>());
}

/*****************************/

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
set_union_by_key(execution_policy<Derived> &policy,
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
    ret = __set_operations::set_operations<thrust::detail::true_type>(
        policy,
        keys1_first,
        keys1_last,
        keys2_first,
        keys2_last,
        items1_first,
        items2_first,
        keys_result,
        items_result,
        compare_op,
        __set_operations::serial_set_union());
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::set_union_by_key(cvt_to_seq(derived_cast(policy)),
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
set_union_by_key(execution_policy<Derived> &policy,
                 KeysIt1                    keys1_first,
                 KeysIt1                    keys1_last,
                 KeysIt2                    keys2_first,
                 KeysIt2                    keys2_last,
                 ItemsIt1                   items1_first,
                 ItemsIt2                   items2_first,
                 KeysOutputIt               keys_result,
                 ItemsOutputIt              items_result)
{
  typedef typename thrust::iterator_value<KeysIt1>::type value_type;
  return cuda_cub::set_union_by_key(policy,
                                    keys1_first,
                                    keys1_last,
                                    keys2_first,
                                    keys2_last,
                                    items1_first,
                                    items2_first,
                                    keys_result,
                                    items_result,
                                    less<value_type>());
}

}    // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
