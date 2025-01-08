// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "parallel_for.h"

namespace embree
{
  template<typename ArrayArray, typename Func>
    __forceinline void sequential_for_for( ArrayArray& array2, const size_t minStepSize, const Func& func ) 
  {
    size_t k=0;
    for (size_t i=0; i!=array2.size(); ++i) {
      const size_t N = array2[i]->size();
      if (N) func(array2[i],range<size_t>(0,N),k);
      k+=N;
    }
  }

  class ParallelForForState
  {
  public:

    enum { MAX_TASKS = 64 };

    __forceinline ParallelForForState () 
      : taskCount(0) {}

    template<typename ArrayArray>
      __forceinline ParallelForForState (ArrayArray& array2, const size_t minStepSize) {
      init(array2,minStepSize);
    }

    template<typename SizeFunc>
    __forceinline ParallelForForState (const size_t numArrays, const SizeFunc& getSize, const size_t minStepSize) {
      init(numArrays,getSize,minStepSize);
    } 

    template<typename SizeFunc>
    __forceinline void init ( const size_t numArrays, const SizeFunc& getSize, const size_t minStepSize )
    {
      /* first calculate total number of elements */
      size_t N = 0;
      for (size_t i=0; i<numArrays; i++) {
	N += getSize(i);
      }
      this->N = N;

      /* calculate number of tasks to use */
      const size_t numThreads = TaskScheduler::threadCount();
      const size_t numBlocks  = (N+minStepSize-1)/minStepSize;
      taskCount = max(size_t(1),min(numThreads,numBlocks,size_t(ParallelForForState::MAX_TASKS)));
      
      /* calculate start (i,j) for each task */
      size_t taskIndex = 0;
      i0[taskIndex] = 0;
      j0[taskIndex] = 0;
      size_t k0 = (++taskIndex)*N/taskCount;
      for (size_t i=0, k=0; taskIndex < taskCount; i++) 
      {
	assert(i<numArrays);
	size_t j=0, M = getSize(i);
	while (j<M && k+M-j >= k0 && taskIndex < taskCount) {
	  assert(taskIndex<taskCount);
	  i0[taskIndex] = i;
	  j0[taskIndex] = j += k0-k;
	  k=k0;
	  k0 = (++taskIndex)*N/taskCount;
	}
	k+=M-j;
      }
    }

    template<typename ArrayArray>
      __forceinline void init ( ArrayArray& array2, const size_t minStepSize )
    {
      init(array2.size(),[&](size_t i) { return array2[i] ? array2[i]->size() : 0; },minStepSize);
    }
    
    __forceinline size_t size() const {
      return N;
    }
    
  public:
    size_t i0[MAX_TASKS];
    size_t j0[MAX_TASKS];
    size_t taskCount;
    size_t N;
  };

  template<typename ArrayArray, typename Func>
    __forceinline void parallel_for_for( ArrayArray& array2, const size_t minStepSize, const Func& func )
  {
    ParallelForForState state(array2,minStepSize);
    
    parallel_for(state.taskCount, [&](const size_t taskIndex) 
    {
      /* calculate range */
      const size_t k0 = (taskIndex+0)*state.size()/state.taskCount;
      const size_t k1 = (taskIndex+1)*state.size()/state.taskCount;
      size_t i0 = state.i0[taskIndex];
      size_t j0 = state.j0[taskIndex];

      /* iterate over arrays */
      size_t k=k0;
      for (size_t i=i0; k<k1; i++) {
        const size_t N =  array2[i] ? array2[i]->size() : 0;
        const size_t r0 = j0, r1 = min(N,r0+k1-k);
        if (r1 > r0) func(array2[i],range<size_t>(r0,r1),k);
        k+=r1-r0; j0 = 0;
      }
    });
  }

  template<typename ArrayArray, typename Func>
    __forceinline void parallel_for_for( ArrayArray& array2, const Func& func )
  {
    parallel_for_for(array2,1,func);
  }

  template<typename ArrayArray, typename Value, typename Func, typename Reduction>
    __forceinline Value parallel_for_for_reduce( ArrayArray& array2, const size_t minStepSize, const Value& identity, const Func& func, const Reduction& reduction )
  {
    ParallelForForState state(array2,minStepSize);
    Value temp[ParallelForForState::MAX_TASKS];

    for (size_t i=0; i<state.taskCount; i++)
      temp[i] = identity;
    
    parallel_for(state.taskCount, [&](const size_t taskIndex) 
    {
      /* calculate range */
      const size_t k0 = (taskIndex+0)*state.size()/state.taskCount;
      const size_t k1 = (taskIndex+1)*state.size()/state.taskCount;
      size_t i0 = state.i0[taskIndex];
      size_t j0 = state.j0[taskIndex];

      /* iterate over arrays */
      size_t k=k0;
      for (size_t i=i0; k<k1; i++) {
        const size_t N =  array2[i] ? array2[i]->size() : 0;
        const size_t r0 = j0, r1 = min(N,r0+k1-k);
        if (r1 > r0) temp[taskIndex] = reduction(temp[taskIndex],func(array2[i],range<size_t>(r0,r1),k));
        k+=r1-r0; j0 = 0;
      }
    });

    Value ret = identity;
    for (size_t i=0; i<state.taskCount; i++)
      ret = reduction(ret,temp[i]);
    return ret;
  }

  template<typename ArrayArray, typename Value, typename Func, typename Reduction>
    __forceinline Value parallel_for_for_reduce( ArrayArray& array2, const Value& identity, const Func& func, const Reduction& reduction)
  {
    return parallel_for_for_reduce(array2,1,identity,func,reduction);
  }
}
