/*
 * Copyright © 2016 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS, AUTHORS
 * AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 */

/* Job queue with execution in a separate thread.
 *
 * Jobs can be added from any thread. After that, the wait call can be used
 * to wait for completion of the job.
 */

#ifndef U_QUEUE_H
#define U_QUEUE_H

#include <string.h>

#include "simple_mtx.h"
#include "util/futex.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_thread.h"

#ifdef __cplusplus
extern "C" {
#endif

#define UTIL_QUEUE_INIT_USE_MINIMUM_PRIORITY      (1 << 0)
#define UTIL_QUEUE_INIT_RESIZE_IF_FULL            (1 << 1)
#define UTIL_QUEUE_INIT_SET_FULL_THREAD_AFFINITY  (1 << 2)
#define UTIL_QUEUE_INIT_SCALE_THREADS             (1 << 3)

#if UTIL_FUTEX_SUPPORTED
#define UTIL_QUEUE_FENCE_FUTEX
#else
#define UTIL_QUEUE_FENCE_STANDARD
#endif

#ifdef UTIL_QUEUE_FENCE_FUTEX
/* Job completion fence.
 * Put this into your job structure.
 */
struct util_queue_fence {
   /* The fence can be in one of three states:
    *  0 - signaled
    *  1 - unsignaled
    *  2 - unsignaled, may have waiters
    */
   uint32_t val;
};

static inline void
util_queue_fence_init(struct util_queue_fence *fence)
{
   fence->val = 0;
}

static inline void
util_queue_fence_destroy(struct util_queue_fence *fence)
{
   assert(p_atomic_read_relaxed(&fence->val) == 0);
   /* no-op */
}

static inline void
util_queue_fence_signal(struct util_queue_fence *fence)
{
   uint32_t val = p_atomic_xchg(&fence->val, 0);

   assert(val != 0);

   if (val == 2)
      futex_wake(&fence->val, INT_MAX);
}

/**
 * Move \p fence back into unsignalled state.
 *
 * \warning The caller must ensure that no other thread may currently be
 *          waiting (or about to wait) on the fence.
 */
static inline void
util_queue_fence_reset(struct util_queue_fence *fence)
{
#ifdef NDEBUG
   fence->val = 1;
#else
   uint32_t v = p_atomic_xchg(&fence->val, 1);
   assert(v == 0);
#endif
}

static inline bool
util_queue_fence_is_signalled(struct util_queue_fence *fence)
{
   return p_atomic_read_relaxed(&fence->val) == 0;
}
#endif

#ifdef UTIL_QUEUE_FENCE_STANDARD
/* Job completion fence.
 * Put this into your job structure.
 */
struct util_queue_fence {
   mtx_t mutex;
   cnd_t cond;
   int signalled;
};

void util_queue_fence_init(struct util_queue_fence *fence);
void util_queue_fence_destroy(struct util_queue_fence *fence);
void util_queue_fence_signal(struct util_queue_fence *fence);

/**
 * Move \p fence back into unsignalled state.
 *
 * \warning The caller must ensure that no other thread may currently be
 *          waiting (or about to wait) on the fence.
 */
static inline void
util_queue_fence_reset(struct util_queue_fence *fence)
{
   assert(fence->signalled);
   fence->signalled = 0;
}

static inline bool
util_queue_fence_is_signalled(struct util_queue_fence *fence)
{
   return fence->signalled != 0;
}
#endif

void
_util_queue_fence_wait(struct util_queue_fence *fence);

static inline void
util_queue_fence_wait(struct util_queue_fence *fence)
{
   if (unlikely(!util_queue_fence_is_signalled(fence)))
      _util_queue_fence_wait(fence);
}

bool
_util_queue_fence_wait_timeout(struct util_queue_fence *fence,
                               int64_t abs_timeout);

/**
 * Wait for the fence to be signaled with a timeout.
 *
 * \param fence the fence
 * \param abs_timeout the absolute timeout in nanoseconds, relative to the
 *                    clock provided by os_time_get_nano.
 *
 * \return true if the fence was signaled, false if the timeout occurred.
 */
static inline bool
util_queue_fence_wait_timeout(struct util_queue_fence *fence,
                              int64_t abs_timeout)
{
   if (util_queue_fence_is_signalled(fence))
      return true;

   if (abs_timeout == (int64_t)OS_TIMEOUT_INFINITE) {
      _util_queue_fence_wait(fence);
      return true;
   }

   return _util_queue_fence_wait_timeout(fence, abs_timeout);
}

typedef void (*util_queue_execute_func)(void *job, void *gdata, int thread_index);

struct util_queue_job {
   void *job;
   void *global_data;
   size_t job_size;
   struct util_queue_fence *fence;
   util_queue_execute_func execute;
   util_queue_execute_func cleanup;
};

/* Put this into your context. */
struct util_queue {
   char name[14]; /* 13 characters = the thread name without the index */
   simple_mtx_t finish_lock; /* for util_queue_finish and protects threads/num_threads */
   mtx_t lock;
   cnd_t has_queued_cond;
   cnd_t has_space_cond;
   thrd_t *threads;
   unsigned flags;
   int num_queued;
   unsigned max_threads;
   unsigned num_threads; /* decreasing this number will terminate threads */
   int max_jobs;
   int write_idx, read_idx; /* ring buffer pointers */
   size_t total_jobs_size;  /* memory use of all jobs in the queue */
   struct util_queue_job *jobs;
   void *global_data;

   /* for cleanup at exit(), protected by exit_mutex */
   struct list_head head;
};

bool util_queue_init(struct util_queue *queue,
                     const char *name,
                     unsigned max_jobs,
                     unsigned num_threads,
                     unsigned flags,
                     void *global_data);
void util_queue_destroy(struct util_queue *queue);

/* optional cleanup callback is called after fence is signaled: */
void util_queue_add_job(struct util_queue *queue,
                        void *job,
                        struct util_queue_fence *fence,
                        util_queue_execute_func execute,
                        util_queue_execute_func cleanup,
                        const size_t job_size);
void util_queue_drop_job(struct util_queue *queue,
                         struct util_queue_fence *fence);

void util_queue_finish(struct util_queue *queue);

/* Adjust the number of active threads. The new number of threads can't be
 * greater than the initial number of threads at the creation of the queue,
 * and it can't be less than 1.
 */
void
util_queue_adjust_num_threads(struct util_queue *queue, unsigned num_threads);

int64_t util_queue_get_thread_time_nano(struct util_queue *queue,
                                        unsigned thread_index);

/* util_queue needs to be cleared to zeroes for this to work */
static inline bool
util_queue_is_initialized(struct util_queue *queue)
{
   return queue->threads != NULL;
}

/* Convenient structure for monitoring the queue externally and passing
 * the structure between Mesa components. The queue doesn't use it directly.
 */
struct util_queue_monitoring
{
   /* For querying the thread busyness. */
   struct util_queue *queue;

   /* Counters updated by the user of the queue. */
   unsigned num_offloaded_items;
   unsigned num_direct_items;
   unsigned num_syncs;
   unsigned num_batches;
};

#ifdef __cplusplus
}
#endif

#endif
