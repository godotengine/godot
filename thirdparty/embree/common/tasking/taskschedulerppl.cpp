// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "taskschedulerppl.h"

namespace embree
{
  static bool g_ppl_threads_initialized = false;
    
  void TaskScheduler::create(size_t numThreads, bool set_affinity, bool start_threads)
  {
    assert(numThreads);
    
    /* first terminate threads in case we configured them */
    if (g_ppl_threads_initialized) {
      g_ppl_threads_initialized = false;
    }
    
    /* now either keep default settings or configure number of threads */
    if (numThreads == std::numeric_limits<size_t>::max())
    {
      g_ppl_threads_initialized = false;
      numThreads = threadCount();
    }
    else 
    {
      g_ppl_threads_initialized = true;
      try {
        concurrency::Scheduler::SetDefaultSchedulerPolicy(concurrency::SchedulerPolicy(2, concurrency::MinConcurrency, numThreads, concurrency::MaxConcurrency, numThreads));
      }
      catch(concurrency::default_scheduler_exists &) { 
      }
    }
  }
  
  void TaskScheduler::destroy()
  {
    if (g_ppl_threads_initialized) {
      g_ppl_threads_initialized = false;
    }
  }
}
