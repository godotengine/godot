// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../tasking/taskscheduler.h"
#include "../sys/array.h"
#include "../math/emath.h"
#include "../math/range.h"

namespace embree
{
  /* parallel_for without range */
  template<typename Index, typename Func>
    __forceinline void parallel_for( const Index N, const Func& func)
  {
#if defined(TASKING_INTERNAL) && !defined(TASKING_TBB)
    if (N) {
      TaskScheduler::TaskGroupContext context;
      TaskScheduler::spawn(Index(0),N,Index(1),[&] (const range<Index>& r) {
          assert(r.size() == 1);
          func(r.begin());
        },&context);
      TaskScheduler::wait();
      if (context.cancellingException != nullptr) {
        std::rethrow_exception(context.cancellingException);
      }
    }
#elif defined(TASKING_TBB)
  #if TBB_INTERFACE_VERSION >= 12002
    tbb::task_group_context context;
    tbb::parallel_for(Index(0),N,Index(1),[&](Index i) {
        func(i);
      },context);
    if (context.is_group_execution_cancelled())
      throw std::runtime_error("task cancelled");
  #else
    tbb::parallel_for(Index(0),N,Index(1),[&](Index i) {
        func(i);
      });
    if (tbb::task::self().is_cancelled())
      throw std::runtime_error("task cancelled");
  #endif

#elif defined(TASKING_PPL)
    concurrency::parallel_for(Index(0),N,Index(1),[&](Index i) { 
        func(i);
      });
#else
#  error "no tasking system enabled"
#endif
  }
  
  /* parallel for with range and granulatity */
  template<typename Index, typename Func>
    __forceinline void parallel_for( const Index first, const Index last, const Index minStepSize, const Func& func)
  {
    assert(first <= last);
#if defined(TASKING_INTERNAL) && !defined(TASKING_TBB)
    TaskScheduler::TaskGroupContext context;
    TaskScheduler::spawn(first,last,minStepSize,func,&context);
    TaskScheduler::wait();
    if (context.cancellingException != nullptr) {
      std::rethrow_exception(context.cancellingException);
    }

#elif defined(TASKING_TBB)
  #if TBB_INTERFACE_VERSION >= 12002
    tbb::task_group_context context;
    tbb::parallel_for(tbb::blocked_range<Index>(first,last,minStepSize),[&](const tbb::blocked_range<Index>& r) {
        func(range<Index>(r.begin(),r.end()));
      },context);
    if (context.is_group_execution_cancelled())
      throw std::runtime_error("task cancelled");
  #else
    tbb::parallel_for(tbb::blocked_range<Index>(first,last,minStepSize),[&](const tbb::blocked_range<Index>& r) {
        func(range<Index>(r.begin(),r.end()));
      });
    if (tbb::task::self().is_cancelled())
      throw std::runtime_error("task cancelled");
  #endif

#elif defined(TASKING_PPL)
    concurrency::parallel_for(first, last, Index(1) /*minStepSize*/, [&](Index i) { 
        func(range<Index>(i,i+1)); 
      });

#else
#  error "no tasking system enabled"
#endif
  }
  
  /* parallel for with range */
  template<typename Index, typename Func>
    __forceinline void parallel_for( const Index first, const Index last, const Func& func)
  {
    assert(first <= last);
    parallel_for(first,last,(Index)1,func);
  }

#if defined(TASKING_TBB) && (TBB_INTERFACE_VERSION > 4001)

  template<typename Index, typename Func>
    __forceinline void parallel_for_static( const Index N, const Func& func)
  {
    #if TBB_INTERFACE_VERSION >= 12002
      tbb::task_group_context context;
      tbb::parallel_for(Index(0),N,Index(1),[&](Index i) {
          func(i);
        },tbb::simple_partitioner(),context);
      if (context.is_group_execution_cancelled())
        throw std::runtime_error("task cancelled");
    #else
      tbb::parallel_for(Index(0),N,Index(1),[&](Index i) {
          func(i);
        },tbb::simple_partitioner());
      if (tbb::task::self().is_cancelled())
        throw std::runtime_error("task cancelled");
    #endif
  }

  typedef tbb::affinity_partitioner affinity_partitioner;

  template<typename Index, typename Func>
    __forceinline void parallel_for_affinity( const Index N, const Func& func, tbb::affinity_partitioner& ap)
  {
    #if TBB_INTERFACE_VERSION >= 12002
      tbb::task_group_context context;
      tbb::parallel_for(Index(0),N,Index(1),[&](Index i) {
          func(i);
        },ap,context);
      if (context.is_group_execution_cancelled())
        throw std::runtime_error("task cancelled");
    #else
      tbb::parallel_for(Index(0),N,Index(1),[&](Index i) {
          func(i);
        },ap);
      if (tbb::task::self().is_cancelled())
        throw std::runtime_error("task cancelled");
    #endif
  }

#else

  template<typename Index, typename Func>
    __forceinline void parallel_for_static( const Index N, const Func& func) 
  {
    parallel_for(N,func);
  }

  struct affinity_partitioner {
  };

  template<typename Index, typename Func>
    __forceinline void parallel_for_affinity( const Index N, const Func& func, affinity_partitioner& ap) 
  {
    parallel_for(N,func);
  }

#endif
}
