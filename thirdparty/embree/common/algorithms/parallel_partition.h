// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "parallel_for.h"
#include "../math/range.h"

namespace embree
{
  /* serial partitioning */
  template<typename T, typename V, typename IsLeft, typename Reduction_T>
    __forceinline size_t serial_partitioning(T* array, 
                                             const size_t begin,
                                             const size_t end, 
                                             V& leftReduction,
                                             V& rightReduction,
                                             const IsLeft& is_left, 
                                             const Reduction_T& reduction_t)
  {
    T* l = array + begin;
    T* r = array + end - 1;
    
    while(1)
    {
      /* *l < pivot */
      while (likely(l <= r && is_left(*l) )) 
      {
        //prefetchw(l+4); // FIXME: enable?
        reduction_t(leftReduction,*l);
        ++l;
      }
      /* *r >= pivot) */
      while (likely(l <= r && !is_left(*r)))
      {
        //prefetchw(r-4); FIXME: enable?
        reduction_t(rightReduction,*r);
        --r;
      }
      if (r<l) break;
      
      reduction_t(leftReduction ,*r);
      reduction_t(rightReduction,*l);
      xchg(*l,*r);
      l++; r--;
    }
    
    return l - array;        
  }

  template<typename T, typename V, typename Vi, typename IsLeft, typename Reduction_T, typename Reduction_V>
    class __aligned(64) parallel_partition_task
  {
    ALIGNED_CLASS_(64);
  private:

    static const size_t MAX_TASKS = 64;

    T* array;
    size_t N;
    const IsLeft& is_left;
    const Reduction_T& reduction_t;
    const Reduction_V& reduction_v;
    const Vi& identity;

    size_t numTasks; 
    __aligned(64) size_t counter_start[MAX_TASKS+1]; 
    __aligned(64) size_t counter_left[MAX_TASKS+1];  
    __aligned(64) range<ssize_t> leftMisplacedRanges[MAX_TASKS];  
    __aligned(64) range<ssize_t> rightMisplacedRanges[MAX_TASKS]; 
    __aligned(64) V leftReductions[MAX_TASKS];           
    __aligned(64) V rightReductions[MAX_TASKS];    

  public:
     
    __forceinline parallel_partition_task(T* array, 
                                          const size_t N, 
                                          const Vi& identity, 
                                          const IsLeft& is_left, 
                                          const Reduction_T& reduction_t, 
                                          const Reduction_V& reduction_v,
                                          const size_t BLOCK_SIZE) 

      : array(array), N(N), is_left(is_left), reduction_t(reduction_t), reduction_v(reduction_v), identity(identity),
      numTasks(min((N+BLOCK_SIZE-1)/BLOCK_SIZE,min(TaskScheduler::threadCount(),MAX_TASKS))) {}

    __forceinline const range<ssize_t>* findStartRange(size_t& index, const range<ssize_t>* const r, const size_t numRanges)
    {
      size_t i = 0;
      while(index >= (size_t)r[i].size())
      {
        assert(i < numRanges);
        index -= (size_t)r[i].size();
        i++;
      }	    
      return &r[i];
    }

    __forceinline void swapItemsInMisplacedRanges(const size_t numLeftMisplacedRanges,
                                                  const size_t numRightMisplacedRanges,
                                                  const size_t startID,
                                                  const size_t endID)
    {
      size_t leftLocalIndex  = startID;
      size_t rightLocalIndex = startID;
      const range<ssize_t>* l_range = findStartRange(leftLocalIndex,leftMisplacedRanges,numLeftMisplacedRanges);
      const range<ssize_t>* r_range = findStartRange(rightLocalIndex,rightMisplacedRanges,numRightMisplacedRanges);
      
      size_t l_left = l_range->size() - leftLocalIndex;
      size_t r_left = r_range->size() - rightLocalIndex;
      T *__restrict__ l = &array[l_range->begin() + leftLocalIndex];
      T *__restrict__ r = &array[r_range->begin() + rightLocalIndex];
      size_t size = endID - startID;
      size_t items = min(size,min(l_left,r_left)); 
     
      while (size)
      {
        if (unlikely(l_left == 0))
        {
          l_range++;
          l_left = l_range->size();
          l = &array[l_range->begin()];
          items = min(size,min(l_left,r_left));
        }

        if (unlikely(r_left == 0))
        {		
          r_range++;
          r_left = r_range->size();
          r = &array[r_range->begin()];          
          items = min(size,min(l_left,r_left));
        }

        size   -= items;
        l_left -= items;
        r_left -= items;

        while(items) {
          items--;
          xchg(*l++,*r++);
        }
      }
    }

    __forceinline size_t partition(V& leftReduction, V& rightReduction)
    {
      /* partition the individual ranges for each task */
      parallel_for(numTasks,[&] (const size_t taskID) {
          const size_t startID = (taskID+0)*N/numTasks;
          const size_t endID   = (taskID+1)*N/numTasks;
          V local_left(identity);
          V local_right(identity);
          const size_t mid = serial_partitioning(array,startID,endID,local_left,local_right,is_left,reduction_t);
          counter_start[taskID] = startID;
          counter_left [taskID] = mid-startID;
          leftReductions[taskID]  = local_left;
          rightReductions[taskID] = local_right;
        });
      counter_start[numTasks] = N;
      counter_left[numTasks]  = 0;
      
      /* finalize the reductions */
      for (size_t i=0; i<numTasks; i++) {
        reduction_v(leftReduction,leftReductions[i]);
        reduction_v(rightReduction,rightReductions[i]);
      }

      /* calculate mid point for partitioning */
      size_t mid = counter_left[0];
      for (size_t i=1; i<numTasks; i++)
        mid += counter_left[i];
      const range<ssize_t> globalLeft (0,mid);
      const range<ssize_t> globalRight(mid,N);

      /* calculate all left and right ranges that are on the wrong global side */
      size_t numMisplacedRangesLeft  = 0;
      size_t numMisplacedRangesRight = 0;
      size_t numMisplacedItemsLeft   = 0;
      size_t numMisplacedItemsRight  = 0;

      for (size_t i=0; i<numTasks; i++)
      {	    
        const range<ssize_t> left_range (counter_start[i], counter_start[i] + counter_left[i]);
        const range<ssize_t> right_range(counter_start[i] + counter_left[i], counter_start[i+1]);
        const range<ssize_t> left_misplaced  = globalLeft. intersect(right_range);
        const range<ssize_t> right_misplaced = globalRight.intersect(left_range);

        if (!left_misplaced.empty())  
        {
          numMisplacedItemsLeft += left_misplaced.size();
          leftMisplacedRanges[numMisplacedRangesLeft++] = left_misplaced;
        }

        if (!right_misplaced.empty()) 
        {
          numMisplacedItemsRight += right_misplaced.size();
          rightMisplacedRanges[numMisplacedRangesRight++] = right_misplaced;
        }
      }
      assert( numMisplacedItemsLeft == numMisplacedItemsRight );

      /* if no items are misplaced we are done */
      if (numMisplacedItemsLeft == 0)
        return mid;

      /* otherwise we copy the items to the right place in parallel */
      parallel_for(numTasks,[&] (const size_t taskID) {
          const size_t startID = (taskID+0)*numMisplacedItemsLeft/numTasks;
          const size_t endID   = (taskID+1)*numMisplacedItemsLeft/numTasks;
          swapItemsInMisplacedRanges(numMisplacedRangesLeft,numMisplacedRangesRight,startID,endID);	                             
        });

      return mid;
    }
  };

  template<typename T, typename V, typename Vi, typename IsLeft, typename Reduction_T, typename Reduction_V>
    __noinline size_t parallel_partitioning(T* array, 
                                            const size_t begin,
                                            const size_t end, 
                                            const Vi &identity,
                                            V &leftReduction,
                                            V &rightReduction,
                                            const IsLeft& is_left, 
                                            const Reduction_T& reduction_t,
                                            const Reduction_V& reduction_v,
                                            size_t BLOCK_SIZE = 128)
  {
    /* fall back to single threaded partitioning for small N */
    if (unlikely(end-begin < BLOCK_SIZE))
      return serial_partitioning(array,begin,end,leftReduction,rightReduction,is_left,reduction_t);

    /* otherwise use parallel code */
    else {
      typedef parallel_partition_task<T,V,Vi,IsLeft,Reduction_T,Reduction_V> partition_task;
      std::unique_ptr<partition_task> p(new partition_task(&array[begin],end-begin,identity,is_left,reduction_t,reduction_v,BLOCK_SIZE));
      return begin+p->partition(leftReduction,rightReduction);    
    }
  }

  template<typename T, typename V, typename Vi, typename IsLeft, typename Reduction_T, typename Reduction_V>
    __noinline size_t parallel_partitioning(T* array, 
                                            const size_t begin,
                                            const size_t end, 
                                            const Vi &identity,
                                            V &leftReduction,
                                            V &rightReduction,
                                            const IsLeft& is_left, 
                                            const Reduction_T& reduction_t,
                                            const Reduction_V& reduction_v,
                                            size_t BLOCK_SIZE,
                                            size_t PARALLEL_THRESHOLD)
  {
    /* fall back to single threaded partitioning for small N */
    if (unlikely(end-begin < PARALLEL_THRESHOLD))
      return serial_partitioning(array,begin,end,leftReduction,rightReduction,is_left,reduction_t);

    /* otherwise use parallel code */
    else {
      typedef parallel_partition_task<T,V,Vi,IsLeft,Reduction_T,Reduction_V> partition_task;
      std::unique_ptr<partition_task> p(new partition_task(&array[begin],end-begin,identity,is_left,reduction_t,reduction_v,BLOCK_SIZE));
      return begin+p->partition(leftReduction,rightReduction);    
    }
  }


  template<typename T, typename IsLeft>
    inline size_t parallel_partitioning(T* array, 
                                        const size_t begin,
                                        const size_t end, 
                                        const IsLeft& is_left, 
                                        size_t BLOCK_SIZE = 128)
  {
    size_t leftReduction = 0;
    size_t rightReduction = 0;
    return parallel_partitioning(
      array,begin,end,0,leftReduction,rightReduction,is_left,
      [] (size_t& t,const T& ref) {  },
      [] (size_t& t0,size_t& t1) { },
      BLOCK_SIZE);
  }

}
