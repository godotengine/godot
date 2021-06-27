// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/default.h"

/* force a complete cache invalidation when running out of allocation space */
#define FORCE_SIMPLE_FLUSH 0

#define THREAD_BLOCK_ATOMIC_ADD 4

#if defined(DEBUG)
#define CACHE_STATS(x) 
#else
#define CACHE_STATS(x) 
#endif

namespace embree
{
  class SharedTessellationCacheStats
  {
  public:
    /* stats */
    static std::atomic<size_t> cache_accesses;
    static std::atomic<size_t> cache_hits;
    static std::atomic<size_t> cache_misses;
    static std::atomic<size_t> cache_flushes;                
    static size_t        cache_num_patches;
    __aligned(64) static SpinLock mtx;
    
    /* print stats for debugging */                 
    static void printStats();
    static void clearStats();
  };
  
  void resizeTessellationCache(size_t new_size);
  void resetTessellationCache();
  
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////

 struct __aligned(64) ThreadWorkState 
 {
   ALIGNED_STRUCT_(64);

   std::atomic<size_t> counter;
   ThreadWorkState* next;
   bool allocated;

   __forceinline ThreadWorkState(bool allocated = false) 
     : counter(0), next(nullptr), allocated(allocated) 
   {
     assert( ((size_t)this % 64) == 0 ); 
   }   
 };

 class __aligned(64) SharedLazyTessellationCache 
 {
 public:
   
   static const size_t NUM_CACHE_SEGMENTS              = 8;
   static const size_t NUM_PREALLOC_THREAD_WORK_STATES = 512;
   static const size_t COMMIT_INDEX_SHIFT              = 32+8;
#if defined(__64BIT__)
   static const size_t REF_TAG_MASK                    = 0xffffffffff;
#else
   static const size_t REF_TAG_MASK                    = 0x7FFFFFFF;
#endif
   static const size_t MAX_TESSELLATION_CACHE_SIZE     = REF_TAG_MASK+1;
   static const size_t BLOCK_SIZE                      = 64;
   

    /*! Per thread tessellation ref cache */
   static __thread ThreadWorkState* init_t_state;
   static ThreadWorkState* current_t_state;
   
   static __forceinline ThreadWorkState *threadState() 
   {
     if (unlikely(!init_t_state))
       /* sets init_t_state, can't return pointer due to macosx icc bug*/
       SharedLazyTessellationCache::sharedLazyTessellationCache.getNextRenderThreadWorkState();
     return init_t_state;
   }

   struct Tag
   {
     __forceinline Tag() : data(0) {}

     __forceinline Tag(void* ptr, size_t combinedTime) { 
       init(ptr,combinedTime);
     }

     __forceinline Tag(size_t ptr, size_t combinedTime) {
       init((void*)ptr,combinedTime); 
     }

     __forceinline void init(void* ptr, size_t combinedTime)
     {
       if (ptr == nullptr) {
         data = 0;
         return;
       }
       int64_t new_root_ref = (int64_t) ptr;
       new_root_ref -= (int64_t)SharedLazyTessellationCache::sharedLazyTessellationCache.getDataPtr();                                
       assert( new_root_ref <= (int64_t)REF_TAG_MASK );
       new_root_ref |= (int64_t)combinedTime << COMMIT_INDEX_SHIFT; 
       data = new_root_ref;
     }

     __forceinline int64_t get() const { return data.load(); }
     __forceinline void set( int64_t v ) { data.store(v); }
     __forceinline void reset() { data.store(0); }

   private:
     atomic<int64_t> data;
   };

   static __forceinline size_t extractCommitIndex(const int64_t v) { return v >> SharedLazyTessellationCache::COMMIT_INDEX_SHIFT; }

   struct CacheEntry
   {
     Tag tag;
     SpinLock mutex;
   };

 private:

   float *data;
   bool hugepages;
   size_t size;
   size_t maxBlocks;
   ThreadWorkState *threadWorkState;
      
   __aligned(64) std::atomic<size_t> localTime;
   __aligned(64) std::atomic<size_t> next_block;
   __aligned(64) SpinLock   reset_state;
   __aligned(64) SpinLock   linkedlist_mtx;
   __aligned(64) std::atomic<size_t> switch_block_threshold;
   __aligned(64) std::atomic<size_t> numRenderThreads;


 public:

      
   SharedLazyTessellationCache();
   ~SharedLazyTessellationCache();

   void getNextRenderThreadWorkState();

   __forceinline size_t maxAllocSize() const {
     return switch_block_threshold;
   }

   __forceinline size_t getCurrentIndex() { return localTime.load(); }
   __forceinline void   addCurrentIndex(const size_t i=1) { localTime.fetch_add(i); }

   __forceinline size_t getTime(const size_t globalTime) {
     return localTime.load()+NUM_CACHE_SEGMENTS*globalTime;
   }


   __forceinline size_t lockThread  (ThreadWorkState *const t_state, const ssize_t plus=1) { return t_state->counter.fetch_add(plus);  }
   __forceinline size_t unlockThread(ThreadWorkState *const t_state, const ssize_t plus=-1) { assert(isLocked(t_state)); return t_state->counter.fetch_add(plus); }

   __forceinline bool isLocked(ThreadWorkState *const t_state) { return t_state->counter.load() != 0; }

   static __forceinline void lock  () { sharedLazyTessellationCache.lockThread(threadState()); }
   static __forceinline void unlock() { sharedLazyTessellationCache.unlockThread(threadState()); }
   static __forceinline bool isLocked() { return sharedLazyTessellationCache.isLocked(threadState()); }
   static __forceinline size_t getState() { return threadState()->counter.load(); }
   static __forceinline void lockThreadLoop() { sharedLazyTessellationCache.lockThreadLoop(threadState()); }

   static __forceinline size_t getTCacheTime(const size_t globalTime) {
     return sharedLazyTessellationCache.getTime(globalTime);
   }

   /* per thread lock */
   __forceinline void lockThreadLoop (ThreadWorkState *const t_state) 
   { 
     while(1)
     {
       size_t lock = SharedLazyTessellationCache::sharedLazyTessellationCache.lockThread(t_state,1);
       if (unlikely(lock >= THREAD_BLOCK_ATOMIC_ADD))
       {
         /* lock failed wait until sync phase is over */
         sharedLazyTessellationCache.unlockThread(t_state,-1);	       
         sharedLazyTessellationCache.waitForUsersLessEqual(t_state,0);
       }
       else
         break;
     }
   }

   static __forceinline void* lookup(CacheEntry& entry, size_t globalTime)
   {   
     const int64_t subdiv_patch_root_ref = entry.tag.get(); 
     CACHE_STATS(SharedTessellationCacheStats::cache_accesses++);
     
     if (likely(subdiv_patch_root_ref != 0)) 
     {
       const size_t subdiv_patch_root = (subdiv_patch_root_ref & REF_TAG_MASK) + (size_t)sharedLazyTessellationCache.getDataPtr();
       const size_t subdiv_patch_cache_index = extractCommitIndex(subdiv_patch_root_ref);
       
       if (likely( sharedLazyTessellationCache.validCacheIndex(subdiv_patch_cache_index,globalTime) ))
       {
         CACHE_STATS(SharedTessellationCacheStats::cache_hits++);
         return (void*) subdiv_patch_root;
       }
     }
     CACHE_STATS(SharedTessellationCacheStats::cache_misses++);
     return nullptr;
   }

   template<typename Constructor>
     static __forceinline auto lookup (CacheEntry& entry, size_t globalTime, const Constructor constructor, const bool before=false) -> decltype(constructor())
   {
     ThreadWorkState *t_state = SharedLazyTessellationCache::threadState();

     while (true)
     {
       sharedLazyTessellationCache.lockThreadLoop(t_state);
       void* patch = SharedLazyTessellationCache::lookup(entry,globalTime);
       if (patch) return (decltype(constructor())) patch;
       
       if (entry.mutex.try_lock())
       {
         if (!validTag(entry.tag,globalTime)) 
         {
           auto timeBefore = sharedLazyTessellationCache.getTime(globalTime);
           auto ret = constructor(); // thread is locked here!
           assert(ret);
           /* this should never return nullptr */
           auto timeAfter = sharedLazyTessellationCache.getTime(globalTime);
           auto time = before ? timeBefore : timeAfter;
           __memory_barrier();
           entry.tag = SharedLazyTessellationCache::Tag(ret,time);
           __memory_barrier();
           entry.mutex.unlock();
           return ret;
         }
         entry.mutex.unlock();
       }
       SharedLazyTessellationCache::sharedLazyTessellationCache.unlockThread(t_state);
     }
   }
   
   __forceinline bool validCacheIndex(const size_t i, const size_t globalTime)
   {
#if FORCE_SIMPLE_FLUSH == 1
     return i == getTime(globalTime);
#else
     return i+(NUM_CACHE_SEGMENTS-1) >= getTime(globalTime);
#endif
   }

   static __forceinline bool validTime(const size_t oldtime, const size_t newTime)
   {
     return oldtime+(NUM_CACHE_SEGMENTS-1) >= newTime;
   }


    static __forceinline bool validTag(const Tag& tag, size_t globalTime)
    {
      const int64_t subdiv_patch_root_ref = tag.get(); 
      if (subdiv_patch_root_ref == 0) return false;
      const size_t subdiv_patch_cache_index = extractCommitIndex(subdiv_patch_root_ref);
      return sharedLazyTessellationCache.validCacheIndex(subdiv_patch_cache_index,globalTime);
    }

   void waitForUsersLessEqual(ThreadWorkState *const t_state,
			      const unsigned int users);
    
   __forceinline size_t alloc(const size_t blocks)
   {
     if (unlikely(blocks >= switch_block_threshold))
       throw_RTCError(RTC_ERROR_INVALID_OPERATION,"allocation exceeds size of tessellation cache segment");

     assert(blocks < switch_block_threshold);
     size_t index = next_block.fetch_add(blocks);
     if (unlikely(index + blocks >= switch_block_threshold)) return (size_t)-1;
     return index;
   }

   static __forceinline void* malloc(const size_t bytes)
   {
     size_t block_index = -1;
     ThreadWorkState *const t_state = threadState();
     while (true)
     {
       block_index = sharedLazyTessellationCache.alloc((bytes+BLOCK_SIZE-1)/BLOCK_SIZE);
       if (block_index == (size_t)-1)
       {
         sharedLazyTessellationCache.unlockThread(t_state);		  
         sharedLazyTessellationCache.allocNextSegment();
         sharedLazyTessellationCache.lockThread(t_state);
         continue; 
       }
       break;
     }
     return sharedLazyTessellationCache.getBlockPtr(block_index);
   }

   __forceinline void *getBlockPtr(const size_t block_index)
   {
     assert(block_index < maxBlocks);
     assert(data);
     assert(block_index*16 <= size);
     return (void*)&data[block_index*16];
   }

   __forceinline void*  getDataPtr()      { return data; }
   __forceinline size_t getNumUsedBytes() { return next_block * BLOCK_SIZE; }
   __forceinline size_t getMaxBlocks()    { return maxBlocks; }
   __forceinline size_t getSize()         { return size; }

   void allocNextSegment();
   void realloc(const size_t newSize);

   void reset();

   static SharedLazyTessellationCache sharedLazyTessellationCache;
 };
}
