/*******************************************************************************
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
#include <thrust/system/cuda/detail/reduce.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/distance.h>

#include <cub/util_math.cuh>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub {

namespace __extrema {

  template <class InputType, class IndexType, class Predicate>
  struct arg_min_f
  {
    Predicate predicate;
    typedef tuple<InputType, IndexType> pair_type;

    __host__ __device__
    arg_min_f(Predicate p) : predicate(p) {}

    pair_type __device__
    operator()(pair_type const &lhs, pair_type const &rhs)
    {
      InputType const &rhs_value = get<0>(rhs);
      InputType const &lhs_value = get<0>(lhs);
      IndexType const &rhs_key   = get<1>(rhs);
      IndexType const &lhs_key   = get<1>(lhs);

      // check values first
      if (predicate(lhs_value, rhs_value))
        return lhs;
      else if (predicate(rhs_value, lhs_value))
        return rhs;

      // values are equivalent, prefer smaller index
      if (lhs_key < rhs_key)
        return lhs;
      else
        return rhs;
    }
  };    // struct arg_min_f

  template <class InputType, class IndexType, class Predicate>
  struct arg_max_f
  {
    Predicate predicate;
    typedef tuple<InputType, IndexType> pair_type;

    __host__ __device__
    arg_max_f(Predicate p) : predicate(p) {}

    pair_type __device__
    operator()(pair_type const &lhs, pair_type const &rhs)
    {
      InputType const &rhs_value = get<0>(rhs);
      InputType const &lhs_value = get<0>(lhs);
      IndexType const &rhs_key   = get<1>(rhs);
      IndexType const &lhs_key   = get<1>(lhs);

      // check values first
      if (predicate(lhs_value, rhs_value))
        return rhs;
      else if (predicate(rhs_value, lhs_value))
        return lhs;

      // values are equivalent, prefer smaller index
      if (lhs_key < rhs_key)
        return lhs;
      else
        return rhs;
    }
  };    // struct arg_max_f

  template<class InputType, class IndexType, class Predicate>
  struct arg_minmax_f
  {
    Predicate predicate;

    typedef tuple<InputType, IndexType> pair_type;
    typedef tuple<pair_type, pair_type> two_pairs_type;

    typedef arg_min_f<InputType, IndexType, Predicate> arg_min_t;
    typedef arg_max_f<InputType, IndexType, Predicate> arg_max_t;

    __host__ __device__
    arg_minmax_f(Predicate p) : predicate(p)
    {
    }

    two_pairs_type __device__
    operator()(two_pairs_type const &lhs, two_pairs_type const &rhs)
    {
      pair_type const &rhs_min = get<0>(rhs);
      pair_type const &lhs_min = get<0>(lhs);
      pair_type const &rhs_max = get<1>(rhs);
      pair_type const &lhs_max = get<1>(lhs);

      auto result = thrust::make_tuple(arg_min_t(predicate)(lhs_min, rhs_min),
                                       arg_max_t(predicate)(lhs_max, rhs_max));

      return result;
    }

    struct duplicate_tuple
    {
      __device__ two_pairs_type
      operator()(pair_type const &t)
      {
        return thrust::make_tuple(t, t);
      }
    };
  }; // struct arg_minmax_f

  template <class T,
            class InputIt,
            class OutputIt,
            class Size,
            class ReductionOp>
  cudaError_t THRUST_RUNTIME_FUNCTION
  doit_step(void *       d_temp_storage,
            size_t &     temp_storage_bytes,
            InputIt      input_it,
            Size         num_items,
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
        __reduce::ReduceAgent<InputIt, OutputIt, T, Size, ReductionOp> >
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
      ra.launch(input_it, output_it, num_items, reduction_op);
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
      even_share.DispatchInit(num_items, max_blocks,
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

        typedef AgentLauncher<__reduce::DrainAgent<Size> > drain_agent;
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
        __reduce::ReduceAgent<T*, OutputIt, T, Size, ReductionOp> >
        reduce_agent_single;

      reduce_plan.grid_size = 1;
      reduce_agent_single ra1(reduce_plan, stream, vshmem_ptr, "reduce_agent: single tile reduce", debug_sync);

      ra1.launch(d_block_reductions, output_it, reduce_grid_size, reduction_op);
      CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
    }

    return status;
  }    // func doit_step

  // this is an init-less reduce, needed for min/max-element functionality
  // this will avoid copying the first value from device->host
  template <typename Derived,
            typename InputIt,
            typename Size,
            typename BinaryOp,
            typename T>
  THRUST_RUNTIME_FUNCTION
  T extrema(execution_policy<Derived>& policy,
            InputIt                    first,
            Size                       num_items,
            BinaryOp                   binary_op,
            T*)
  {
    size_t       temp_storage_bytes = 0;
    cudaStream_t stream             = cuda_cub::stream(policy);
    bool         debug_sync         = THRUST_DEBUG_SYNC_FLAG;

    cudaError_t status;
    THRUST_INDEX_TYPE_DISPATCH(status, doit_step<T>, num_items,
        (NULL, temp_storage_bytes, first, num_items_fixed,
            binary_op, reinterpret_cast<T*>(NULL), stream,
            debug_sync));
    cuda_cub::throw_on_error(status, "extrema failed on 1st step");

    size_t allocation_sizes[2] = {sizeof(T*), temp_storage_bytes};
    void * allocations[2]      = {NULL, NULL};

    size_t storage_size = 0;
    status = core::alias_storage(NULL,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "extrema failed on 1st alias storage");

    // Allocate temporary storage.
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
      tmp(policy, storage_size);
    void *ptr = static_cast<void*>(tmp.data().get());

    status = core::alias_storage(ptr,
                                 storage_size,
                                 allocations,
                                 allocation_sizes);
    cuda_cub::throw_on_error(status, "extrema failed on 2nd alias storage");

    T* d_result = thrust::detail::aligned_reinterpret_cast<T*>(allocations[0]);

    THRUST_INDEX_TYPE_DISPATCH(status, doit_step<T>, num_items,
        (allocations[1], temp_storage_bytes, first,
            num_items_fixed, binary_op, d_result, stream,
            debug_sync));
    cuda_cub::throw_on_error(status, "extrema failed on 2nd step");

    status = cuda_cub::synchronize(policy);
    cuda_cub::throw_on_error(status, "extrema failed to synchronize");

    T result = cuda_cub::get_value(policy, d_result);

    return result;
  }

  template <template <class, class, class> class ArgFunctor,
            class Derived,
            class ItemsIt,
            class BinaryPred>
  ItemsIt THRUST_RUNTIME_FUNCTION
  element(execution_policy<Derived> &policy,
          ItemsIt                    first,
          ItemsIt                    last,
          BinaryPred                 binary_pred)
  {
    if (first == last)
      return last;

    typedef typename iterator_traits<ItemsIt>::value_type      InputType;
    typedef typename iterator_traits<ItemsIt>::difference_type IndexType;

    IndexType num_items = static_cast<IndexType>(thrust::distance(first, last));

    typedef tuple<ItemsIt, counting_iterator_t<IndexType> > iterator_tuple;
    typedef zip_iterator<iterator_tuple> zip_iterator;

    iterator_tuple iter_tuple = thrust::make_tuple(first, counting_iterator_t<IndexType>(0));


    typedef ArgFunctor<InputType, IndexType, BinaryPred> arg_min_t;
    typedef tuple<InputType, IndexType> T;

    zip_iterator begin = make_zip_iterator(iter_tuple);

    T result = extrema(policy,
                       begin,
                       num_items,
                       arg_min_t(binary_pred),
                       (T *)(NULL));
    return first + thrust::get<1>(result);
  }


}    // namespace __extrema

/// min element

__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt,
          class BinaryPred>
ItemsIt __host__ __device__
min_element(execution_policy<Derived> &policy,
            ItemsIt                    first,
            ItemsIt                    last,
            BinaryPred                 binary_pred)
{
  ItemsIt ret = first;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __extrema::element<__extrema::arg_min_f>(policy,
                                                   first,
                                                   last,
                                                   binary_pred);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::min_element(cvt_to_seq(derived_cast(policy)),
                              first,
                              last,
                              binary_pred);
#endif
  }
  return ret;
}

template <class Derived,
          class ItemsIt>
ItemsIt __host__ __device__
min_element(execution_policy<Derived> &policy,
            ItemsIt                    first,
            ItemsIt                    last)
{
  typedef typename iterator_value<ItemsIt>::type value_type;
  return cuda_cub::min_element(policy, first, last, less<value_type>());
}

/// max element

__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt,
          class BinaryPred>
ItemsIt __host__ __device__
max_element(execution_policy<Derived> &policy,
            ItemsIt                    first,
            ItemsIt                    last,
            BinaryPred                 binary_pred)
{
  ItemsIt ret = first;
  if (__THRUST_HAS_CUDART__)
  {
    ret = __extrema::element<__extrema::arg_max_f>(policy,
                                                   first,
                                                   last,
                                                   binary_pred);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::max_element(cvt_to_seq(derived_cast(policy)),
                              first,
                              last,
                              binary_pred);
#endif
  }
  return ret;
}

template <class Derived,
          class ItemsIt>
ItemsIt __host__ __device__
max_element(execution_policy<Derived> &policy,
            ItemsIt                    first,
            ItemsIt                    last)
{
  typedef typename iterator_value<ItemsIt>::type value_type;
  return cuda_cub::max_element(policy, first, last, less<value_type>());
}

/// minmax element

__thrust_exec_check_disable__
template <class Derived,
          class ItemsIt,
          class BinaryPred>
pair<ItemsIt, ItemsIt> __host__ __device__
minmax_element(execution_policy<Derived> &policy,
               ItemsIt                    first,
               ItemsIt                    last,
               BinaryPred                 binary_pred)
{
  pair<ItemsIt, ItemsIt> ret = thrust::make_pair(first, first);

  if (__THRUST_HAS_CUDART__)
  {
    if (first == last)
      return thrust::make_pair(last, last);

    typedef typename iterator_traits<ItemsIt>::value_type      InputType;
    typedef typename iterator_traits<ItemsIt>::difference_type IndexType;

    IndexType num_items = static_cast<IndexType>(thrust::distance(first, last));


    typedef tuple<ItemsIt, counting_iterator_t<IndexType> > iterator_tuple;
    typedef zip_iterator<iterator_tuple> zip_iterator;

    iterator_tuple iter_tuple = thrust::make_tuple(first, counting_iterator_t<IndexType>(0));


    typedef __extrema::arg_minmax_f<InputType, IndexType, BinaryPred> arg_minmax_t;
    typedef typename arg_minmax_t::two_pairs_type  two_pairs_type;
    typedef typename arg_minmax_t::duplicate_tuple duplicate_t;
    typedef transform_input_iterator_t<two_pairs_type,
                                       zip_iterator,
                                       duplicate_t>
        transform_t;

    zip_iterator   begin  = make_zip_iterator(iter_tuple);
    two_pairs_type result = __extrema::extrema(policy,
                                               transform_t(begin, duplicate_t()),
                                               num_items,
                                               arg_minmax_t(binary_pred),
                                               (two_pairs_type *)(NULL));
    ret = thrust::make_pair(first + get<1>(get<0>(result)),
                    first + get<1>(get<1>(result)));
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::minmax_element(cvt_to_seq(derived_cast(policy)),
                                 first,
                                 last,
                                 binary_pred);
#endif
  }
  return ret;
}

template <class Derived,
          class ItemsIt>
pair<ItemsIt, ItemsIt> __host__ __device__
minmax_element(execution_policy<Derived> &policy,
               ItemsIt                    first,
               ItemsIt                    last)
{
  typedef typename iterator_value<ItemsIt>::type value_type;
  return cuda_cub::minmax_element(policy, first, last, less<value_type>());
}


} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
