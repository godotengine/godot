#pragma once

#include "../sys/platform.h"
#include "../sys/alloc.h"
#include "../sys/barrier.h"
#include "../sys/thread.h"
#include "../sys/mutex.h"
#include "../sys/condition.h"
#include "../sys/ref.h"

#include <dispatch/dispatch.h>

namespace embree
{
  struct TaskScheduler
  {
    /*! initializes the task scheduler */
    static void create(size_t numThreads, bool set_affinity, bool start_threads);

    /*! destroys the task scheduler again */
    static void destroy() {}

    /* returns the ID of the current thread */
    static __forceinline size_t threadID()
    {
      return threadIndex();
    }

    /* returns the index (0..threadCount-1) of the current thread */
    static __forceinline size_t threadIndex()
    {
        currentThreadIndex = (currentThreadIndex + 1) % GCDNumThreads;
        return currentThreadIndex;
    }

    /* returns the total number of threads */
    static __forceinline size_t threadCount()
    {
        return GCDNumThreads;
    }

    private:
      static size_t GCDNumThreads;
      static size_t currentThreadIndex;

  };

};

