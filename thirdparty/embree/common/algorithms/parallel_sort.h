// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../simd/simd.h"
#include "parallel_for.h"
#include <algorithm>

namespace embree
{
  template<class T>
    __forceinline void insertionsort_ascending(T *__restrict__ array, const size_t length)
  {
    for(size_t i = 1;i<length;++i)
    {
      T v = array[i];
      size_t j = i;
      while(j > 0 && v < array[j-1])
      {
        array[j] = array[j-1];
        --j;
      }
      array[j] = v;
    }
  }
  
  template<class T>
    __forceinline void insertionsort_decending(T *__restrict__ array, const size_t length)
  {
    for(size_t i = 1;i<length;++i)
    {
      T v = array[i];
      size_t j = i;
      while(j > 0 && v > array[j-1])
      {
        array[j] = array[j-1];
        --j;
      }
      array[j] = v;
    }
  }
  
  template<class T> 
    void quicksort_ascending(T *__restrict__ t, 
			     const ssize_t begin, 
			     const ssize_t end)
  {
    if (likely(begin < end)) 
    {      
      const T pivotvalue = t[begin];
      ssize_t left  = begin - 1;
      ssize_t right = end   + 1;
      
      while(1) 
      {
        while (t[--right] > pivotvalue);
        while (t[++left] < pivotvalue);
        
        if (left >= right) break;
        
        const T temp = t[right];
        t[right] = t[left];
        t[left] = temp;
      }
      
      const int pivot = right;
      quicksort_ascending(t, begin, pivot);
      quicksort_ascending(t, pivot + 1, end);
    }
  }
  
  template<class T> 
    void quicksort_decending(T *__restrict__ t, 
			     const ssize_t begin, 
			     const ssize_t end)
  {
    if (likely(begin < end)) 
    {
      const T pivotvalue = t[begin];
      ssize_t left  = begin - 1;
      ssize_t right = end   + 1;
      
      while(1) 
      {
        while (t[--right] < pivotvalue);
        while (t[++left] > pivotvalue);
        
        if (left >= right) break;
        
        const T temp = t[right];
        t[right] = t[left];
        t[left] = temp;
      }
      
      const int pivot = right;
      quicksort_decending(t, begin, pivot);
      quicksort_decending(t, pivot + 1, end);
    }
  }
  
  
  template<class T, ssize_t THRESHOLD> 
    void quicksort_insertionsort_ascending(T *__restrict__ t, 
					   const ssize_t begin, 
					   const ssize_t end)
  {
    if (likely(begin < end)) 
    {      
      const ssize_t size = end-begin+1;
      if (likely(size <= THRESHOLD))
      {
        insertionsort_ascending<T>(&t[begin],size);
      }
      else
      {
        const T pivotvalue = t[begin];
        ssize_t left  = begin - 1;
        ssize_t right = end   + 1;
        
        while(1) 
        {
          while (t[--right] > pivotvalue);
          while (t[++left] < pivotvalue);
          
          if (left >= right) break;
          
          const T temp = t[right];
          t[right] = t[left];
          t[left] = temp;
        }
        
        const ssize_t pivot = right;
        quicksort_insertionsort_ascending<T,THRESHOLD>(t, begin, pivot);
        quicksort_insertionsort_ascending<T,THRESHOLD>(t, pivot + 1, end);
      }
    }
  }
  
  
  template<class T, ssize_t THRESHOLD> 
    void quicksort_insertionsort_decending(T *__restrict__ t, 
					   const ssize_t begin, 
					   const ssize_t end)
  {
    if (likely(begin < end)) 
    {
      const ssize_t size = end-begin+1;
      if (likely(size <= THRESHOLD))
      {
        insertionsort_decending<T>(&t[begin],size);
      }
      else
      {
        
        const T pivotvalue = t[begin];
        ssize_t left  = begin - 1;
        ssize_t right = end   + 1;
        
        while(1) 
        {
          while (t[--right] < pivotvalue);
          while (t[++left] > pivotvalue);
          
          if (left >= right) break;
          
          const T temp = t[right];
          t[right] = t[left];
          t[left] = temp;
        }
        
        const ssize_t pivot = right;
        quicksort_insertionsort_decending<T,THRESHOLD>(t, begin, pivot);
        quicksort_insertionsort_decending<T,THRESHOLD>(t, pivot + 1, end);
      }
    }
  }
  
  template<typename T>
    static void radixsort32(T* const morton, const size_t num, const unsigned int shift = 3*8)
  {
    static const unsigned int BITS = 8;
    static const unsigned int BUCKETS = (1 << BITS);
    static const unsigned int CMP_SORT_THRESHOLD = 16;
    
    __aligned(64) unsigned int count[BUCKETS];
    
    /* clear buckets */
    for (size_t i=0;i<BUCKETS;i++) count[i] = 0;
    
    /* count buckets */
#if defined(__INTEL_COMPILER)
#pragma nounroll
#endif
    for (size_t i=0;i<num;i++)
      count[(unsigned(morton[i]) >> shift) & (BUCKETS-1)]++;
    
    /* prefix sums */
    __aligned(64) unsigned int head[BUCKETS];
    __aligned(64) unsigned int tail[BUCKETS];
    
    head[0] = 0;
    for (size_t i=1; i<BUCKETS; i++)    
      head[i] = head[i-1] + count[i-1];
    
    for (size_t i=0; i<BUCKETS-1; i++)    
      tail[i] = head[i+1];
    
    tail[BUCKETS-1] = head[BUCKETS-1] + count[BUCKETS-1];
    
    assert(tail[BUCKETS-1] == head[BUCKETS-1] + count[BUCKETS-1]);      
    assert(tail[BUCKETS-1] == num);      
    
    /* in-place swap */      
    for (size_t i=0;i<BUCKETS;i++)
    {
      /* process bucket */
      while(head[i] < tail[i])
      {
        T v = morton[head[i]];
        while(1)
        {
          const size_t b = (unsigned(v) >> shift) & (BUCKETS-1);
          if (b == i) break;
          std::swap(v,morton[head[b]++]);
        }
        assert((unsigned(v) >> shift & (BUCKETS-1)) == i);
        morton[head[i]++] = v;
      }
    }
    if (shift == 0) return;
    
    size_t offset = 0;
    for (size_t i=0;i<BUCKETS;i++)
      if (count[i])
      {
        
        for (size_t j=offset;j<offset+count[i]-1;j++)
          assert(((unsigned(morton[j]) >> shift) & (BUCKETS-1)) == i);
        
        if (unlikely(count[i] < CMP_SORT_THRESHOLD))
          insertionsort_ascending(morton + offset, count[i]);
        else
          radixsort32(morton + offset, count[i], shift-BITS);
        
        for (size_t j=offset;j<offset+count[i]-1;j++)
          assert(morton[j] <= morton[j+1]);
        
        offset += count[i];
      }      
  }    

  template<typename Ty, typename Key>
    class ParallelRadixSort
  {
    static const size_t MAX_TASKS = 64;
    static const size_t BITS = 8;
    static const size_t BUCKETS = (1 << BITS);
    typedef unsigned int TyRadixCount[BUCKETS];
    
    template<typename T>
      static bool compare(const T& v0, const T& v1) {
      return (Key)v0 < (Key)v1;
    }

  private:
    ParallelRadixSort (const ParallelRadixSort& other) DELETED; // do not implement
    ParallelRadixSort& operator= (const ParallelRadixSort& other) DELETED; // do not implement

    
  public:
    ParallelRadixSort (Ty* const src, Ty* const tmp, const size_t N)
      : radixCount(nullptr), src(src), tmp(tmp), N(N) {}

    void sort(const size_t blockSize)
    {
      assert(blockSize > 0);
      
      /* perform single threaded sort for small N */
      if (N<=blockSize) // handles also special case of 0!
      {	  
        /* do inplace sort inside destination array */
        std::sort(src,src+N,compare<Ty>);
      }
      
      /* perform parallel sort for large N */
      else 
      {
        const size_t numThreads = min((N+blockSize-1)/blockSize,TaskScheduler::threadCount(),size_t(MAX_TASKS));
        tbbRadixSort(numThreads);
      }
    }

    ~ParallelRadixSort()
    {
      alignedFree(radixCount); 
      radixCount = nullptr;
    }
    
  private:
    
    void tbbRadixIteration0(const Key shift, 
                            const Ty* __restrict const src, 
                            Ty* __restrict const dst, 
                            const size_t threadIndex, const size_t threadCount)
    {
      const size_t startID = (threadIndex+0)*N/threadCount;
      const size_t endID   = (threadIndex+1)*N/threadCount;
      
      /* mask to extract some number of bits */
      const Key mask = BUCKETS-1;
      
      /* count how many items go into the buckets */
      for (size_t i=0; i<BUCKETS; i++)
        radixCount[threadIndex][i] = 0;

      /* iterate over src array and count buckets */
      unsigned int * __restrict const count = radixCount[threadIndex];
#if defined(__INTEL_COMPILER)
#pragma nounroll      
#endif
      for (size_t i=startID; i<endID; i++) {
#if defined(__64BIT__)
        const size_t index = ((size_t)(Key)src[i] >> (size_t)shift) & (size_t)mask;
#else
        const Key index = ((Key)src[i] >> shift) & mask;
#endif
        count[index]++;
      }
    }
    
    void tbbRadixIteration1(const Key shift, 
                            const Ty* __restrict const src, 
                            Ty* __restrict const dst, 
                            const size_t threadIndex, const size_t threadCount)
    {
      const size_t startID = (threadIndex+0)*N/threadCount;
      const size_t endID   = (threadIndex+1)*N/threadCount;
      
      /* mask to extract some number of bits */
      const Key mask = BUCKETS-1;
      
      /* calculate total number of items for each bucket */
      __aligned(64) unsigned int total[BUCKETS];
      /*
      for (size_t i=0; i<BUCKETS; i++)
        total[i] = 0;
      */
      for (size_t i=0; i<BUCKETS; i+=VSIZEX)
        vintx::store(&total[i], zero);
      
      for (size_t i=0; i<threadCount; i++)
      {
        /*
        for (size_t j=0; j<BUCKETS; j++)
          total[j] += radixCount[i][j];
        */
        for (size_t j=0; j<BUCKETS; j+=VSIZEX)
          vintx::store(&total[j], vintx::load(&total[j]) + vintx::load(&radixCount[i][j]));
      }
      
      /* calculate start offset of each bucket */
      __aligned(64) unsigned int offset[BUCKETS];
      offset[0] = 0;
      for (size_t i=1; i<BUCKETS; i++)    
        offset[i] = offset[i-1] + total[i-1];
      
      /* calculate start offset of each bucket for this thread */
      for (size_t i=0; i<threadIndex; i++)
      {
        /*
        for (size_t j=0; j<BUCKETS; j++)
          offset[j] += radixCount[i][j];
        */
        for (size_t j=0; j<BUCKETS; j+=VSIZEX)
          vintx::store(&offset[j], vintx::load(&offset[j]) + vintx::load(&radixCount[i][j]));
      }
      
      /* copy items into their buckets */
#if defined(__INTEL_COMPILER)
#pragma nounroll
#endif
      for (size_t i=startID; i<endID; i++) {
        const Ty elt = src[i];
#if defined(__64BIT__)
        const size_t index = ((size_t)(Key)src[i] >> (size_t)shift) & (size_t)mask;
#else
        const size_t index = ((Key)src[i] >> shift) & mask;
#endif
        dst[offset[index]++] = elt;
      }
    }
    
    void tbbRadixIteration(const Key shift, const bool last,
                           const Ty* __restrict src, Ty* __restrict dst,
                           const size_t numTasks)
    {
      affinity_partitioner ap;
      parallel_for_affinity(numTasks,[&] (size_t taskIndex) { tbbRadixIteration0(shift,src,dst,taskIndex,numTasks); },ap);
      parallel_for_affinity(numTasks,[&] (size_t taskIndex) { tbbRadixIteration1(shift,src,dst,taskIndex,numTasks); },ap);
    }
    
    void tbbRadixSort(const size_t numTasks)
    {
      radixCount = (TyRadixCount*) alignedMalloc(MAX_TASKS*sizeof(TyRadixCount),64);
      
      if (sizeof(Key) == sizeof(uint32_t)) {
        tbbRadixIteration(0*BITS,0,src,tmp,numTasks);
        tbbRadixIteration(1*BITS,0,tmp,src,numTasks);
        tbbRadixIteration(2*BITS,0,src,tmp,numTasks);
        tbbRadixIteration(3*BITS,1,tmp,src,numTasks);
      }
      else if (sizeof(Key) == sizeof(uint64_t))
      {
        tbbRadixIteration(0*BITS,0,src,tmp,numTasks);
        tbbRadixIteration(1*BITS,0,tmp,src,numTasks);
        tbbRadixIteration(2*BITS,0,src,tmp,numTasks);
        tbbRadixIteration(3*BITS,0,tmp,src,numTasks);
        tbbRadixIteration(4*BITS,0,src,tmp,numTasks);
        tbbRadixIteration(5*BITS,0,tmp,src,numTasks);
        tbbRadixIteration(6*BITS,0,src,tmp,numTasks);
        tbbRadixIteration(7*BITS,1,tmp,src,numTasks);
      }
    }
    
  private:
    TyRadixCount* radixCount;
    Ty* const src;
    Ty* const tmp;
    const size_t N;
  };

  template<typename Ty>
    void radix_sort(Ty* const src, Ty* const tmp, const size_t N, const size_t blockSize = 8192)
  {
    ParallelRadixSort<Ty,Ty>(src,tmp,N).sort(blockSize);
  }
  
  template<typename Ty, typename Key>
    void radix_sort(Ty* const src, Ty* const tmp, const size_t N, const size_t blockSize = 8192)
  {
    ParallelRadixSort<Ty,Key>(src,tmp,N).sort(blockSize);
  }
  
  template<typename Ty>
    void radix_sort_u32(Ty* const src, Ty* const tmp, const size_t N, const size_t blockSize = 8192) {
    radix_sort<Ty,uint32_t>(src,tmp,N,blockSize);
  }
  
  template<typename Ty>
    void radix_sort_u64(Ty* const src, Ty* const tmp, const size_t N, const size_t blockSize = 8192) {
    radix_sort<Ty,uint64_t>(src,tmp,N,blockSize);
  }
}
