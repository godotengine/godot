// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "../tasking/taskscheduler.h"
#include "../sys/array.h"
#include "../math/math.h"
#include "../math/range.h"

namespace embree
{
  /* parallel_for without range */
  template<typename Index, typename Func>
    __forceinline void parallel_for( const Index N, const Func& func)
  {
#if defined(TASKING_INTERNAL)
    if (N) {
      TaskScheduler::spawn(Index(0),N,Index(1),[&] (const range<Index>& r) {
          assert(r.size() == 1);
          func(r.begin());
        });
      if (!TaskScheduler::wait())
        throw std::runtime_error("task cancelled");
    }
    
#elif defined(TASKING_TBB)
    tbb::parallel_for(Index(0),N,Index(1),[&](Index i) { 
	func(i);
      });
    if (tbb::task::self().is_cancelled())
      throw std::runtime_error("task cancelled");

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
#if defined(TASKING_INTERNAL)
    TaskScheduler::spawn(first,last,minStepSize,func);
    if (!TaskScheduler::wait())
      throw std::runtime_error("task cancelled");

#elif defined(TASKING_TBB)
    tbb::parallel_for(tbb::blocked_range<Index>(first,last,minStepSize),[&](const tbb::blocked_range<Index>& r) { 
        func(range<Index>(r.begin(),r.end()));
      });
    if (tbb::task::self().is_cancelled())
      throw std::runtime_error("task cancelled");

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
    tbb::parallel_for(Index(0),N,Index(1),[&](Index i) { 
	func(i);
      },tbb::simple_partitioner());
    if (tbb::task::self().is_cancelled())
      throw std::runtime_error("task cancelled");
  }

  typedef tbb::affinity_partitioner affinity_partitioner;

  template<typename Index, typename Func>
    __forceinline void parallel_for_affinity( const Index N, const Func& func, tbb::affinity_partitioner& ap)
  {
    tbb::parallel_for(Index(0),N,Index(1),[&](Index i) { 
	func(i);
      },ap);
    if (tbb::task::self().is_cancelled())
      throw std::runtime_error("task cancelled");
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
