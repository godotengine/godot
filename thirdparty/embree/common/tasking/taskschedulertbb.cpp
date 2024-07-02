// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "taskschedulertbb.h"

namespace embree
{
  static bool g_tbb_threads_initialized = false;

#if TBB_INTERFACE_VERSION >= 11005
  static tbb::global_control* g_tbb_thread_control = nullptr;
#else
  static tbb::task_scheduler_init g_tbb_threads(tbb::task_scheduler_init::deferred);
#endif

  class TBBAffinity: public tbb::task_scheduler_observer
  {
  public:

    void on_scheduler_entry( bool ) {
      setAffinity(TaskScheduler::threadIndex());
    }

  } tbb_affinity;

  void TaskScheduler::create(size_t numThreads, bool set_affinity, bool start_threads)
  {
    assert(numThreads);

    /* first terminate threads in case we configured them */
    if (g_tbb_threads_initialized) {
#if TBB_INTERFACE_VERSION >= 11005
      delete g_tbb_thread_control;
      g_tbb_thread_control = nullptr;
#else
      g_tbb_threads.terminate();
#endif
      g_tbb_threads_initialized = false;
    }

    /* only set affinity if requested by the user */
#if TBB_INTERFACE_VERSION >= 9000 // affinity not properly supported by older TBB versions
    if (set_affinity)
      tbb_affinity.observe(true);
#endif

    /* now either keep default settings or configure number of threads */
    if (numThreads == std::numeric_limits<size_t>::max()) {
      numThreads = threadCount();
    }
    else {
      g_tbb_threads_initialized = true;
      const size_t max_concurrency = threadCount();
      if (numThreads > max_concurrency) numThreads = max_concurrency;
#if TBB_INTERFACE_VERSION >= 11005
      g_tbb_thread_control = new tbb::global_control(tbb::global_control::max_allowed_parallelism,numThreads);
#else
      g_tbb_threads.initialize(int(numThreads));
#endif
    }

    /* start worker threads */
    if (start_threads)
    {
      BarrierSys barrier(numThreads);
      tbb::parallel_for(size_t(0), size_t(numThreads), size_t(1), [&] ( size_t i ) {
          barrier.wait();
        });
    }
  }

  void TaskScheduler::destroy()
  {
#if TBB_INTERFACE_VERSION >= 9000 // affinity not properly supported by older TBB versions
    // Stop observe to prevent calling on_scheduler_entry
    // when static objects are already destroyed.
    if (tbb_affinity.is_observing())
      tbb_affinity.observe(false);
#endif

    if (g_tbb_threads_initialized) {
#if TBB_INTERFACE_VERSION >= 11005
      delete g_tbb_thread_control;
      g_tbb_thread_control = nullptr;
#else
      g_tbb_threads.terminate();
#endif
      g_tbb_threads_initialized = false;
    }
  }
}
