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

#include "parallel_for.h"

namespace embree
{
  template<typename Ty, typename Index, typename Predicate>
    inline Index sequential_filter( Ty* data, const Index first, const Index last, const Predicate& predicate)
  {
    Index j = first;
    for (Index i=first; i<last; i++)
      if (predicate(data[i]))
        data[j++] = data[i];

    return j;
  }

  template<typename Ty, typename Index, typename Predicate>
    inline Index parallel_filter( Ty* data, const Index begin, const Index end, const Index minStepSize, const Predicate& predicate)
  {
    /* sequential fallback */
    if (end-begin <= minStepSize)
      return sequential_filter(data,begin,end,predicate);

    /* calculate number of tasks to use */
    enum { MAX_TASKS = 64 };
    const Index numThreads = TaskScheduler::threadCount();
    const Index numBlocks  = (end-begin+minStepSize-1)/minStepSize;
    const Index taskCount  = min(numThreads,numBlocks,(Index)MAX_TASKS);

    /* filter blocks */
    Index nused[MAX_TASKS];
    Index nfree[MAX_TASKS];
    parallel_for(taskCount, [&](const Index taskIndex)
    {
      const Index i0 = begin+(taskIndex+0)*(end-begin)/taskCount;
      const Index i1 = begin+(taskIndex+1)*(end-begin)/taskCount;
      const Index i2 = sequential_filter(data,i0,i1,predicate);
      nused[taskIndex] = i2-i0;
      nfree[taskIndex] = i1-i2;
    });

    /* calculate offsets */
    Index sused=0;
    Index sfree=0;
    Index pfree[MAX_TASKS];
    for (Index i=0; i<taskCount; i++) 
    {
      sused+=nused[i];
      Index cfree = nfree[i]; pfree[i] = sfree; sfree+=cfree;
    }

    /* return if we did not filter out any element */
    assert(sfree <= end-begin);
    assert(sused <= end-begin);
    if (sused == end-begin)
      return end;

    /* otherwise we have to copy misplaced elements around */
    parallel_for(taskCount, [&](const Index taskIndex)
    {
      /* destination to write elements to */
      Index dst = begin+(taskIndex+0)*(end-begin)/taskCount+nused[taskIndex];
      Index dst_end = min(dst+nfree[taskIndex],begin+sused);
      if (dst_end <= dst) return;

      /* range of misplaced elements to copy to destination */
      Index r0 = pfree[taskIndex];
      Index r1 = r0+dst_end-dst;

      /* find range in misplaced elements in back to front order */
      Index k0=0;
      for (Index i=taskCount-1; i>0; i--)
      {
        if (k0 > r1) break;
        Index k1 = k0+nused[i];
        Index src = begin+(i+0)*(end-begin)/taskCount+nused[i];
        for (Index i=max(r0,k0); i<min(r1,k1); i++) {
          Index isrc = src-i+k0-1;
          assert(dst >= begin && dst < end);
          assert(isrc >= begin && isrc < end);
          data[dst++] = data[isrc];
        }
        k0 = k1;
      }
    });

    return begin+sused;
  }
}
