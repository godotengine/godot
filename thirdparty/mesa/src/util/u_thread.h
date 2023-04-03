/**************************************************************************
 *
 * Copyright 1999-2006 Brian Paul
 * Copyright 2008 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef U_THREAD_H_
#define U_THREAD_H_

#include <errno.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "c11/threads.h"
#include "detect_os.h"

/* Some highly performance-sensitive thread-local variables like the current GL
 * context are declared with the initial-exec model on Linux.  glibc allocates a
 * fixed number of extra slots for initial-exec TLS variables at startup, and
 * Mesa relies on (even if it's dlopen()ed after init) being able to fit into
 * those.  This model saves the call to look up the address of the TLS variable.
 *
 * However, if we don't have this TLS model available on the platform, then we
 * still want to use normal TLS (which involves a function call, but not the
 * expensive pthread_getspecific() or its equivalent).
 */
#if DETECT_OS_APPLE
/* Apple Clang emits wrappers when using thread_local that break module linkage,
 * but not with __thread
 */
#define __THREAD_INITIAL_EXEC __thread
#elif defined(__GLIBC__)
#define __THREAD_INITIAL_EXEC thread_local __attribute__((tls_model("initial-exec")))
#define REALLY_INITIAL_EXEC
#else
#define __THREAD_INITIAL_EXEC thread_local
#endif

#ifdef __cplusplus
extern "C" {
#endif

int
util_get_current_cpu(void);

int u_thread_create(thrd_t *thrd, int (*routine)(void *), void *param);

void u_thread_setname( const char *name );

/**
 * Set thread affinity.
 *
 * \param thread         Thread
 * \param mask           Set this affinity mask
 * \param old_mask       Previous affinity mask returned if not NULL
 * \param num_mask_bits  Number of bits in both masks
 * \return  true on success
 */
bool
util_set_thread_affinity(thrd_t thread,
                         const uint32_t *mask,
                         uint32_t *old_mask,
                         unsigned num_mask_bits);

static inline bool
util_set_current_thread_affinity(const uint32_t *mask,
                                 uint32_t *old_mask,
                                 unsigned num_mask_bits)
{
   return util_set_thread_affinity(thrd_current(), mask, old_mask,
                                   num_mask_bits);
}

/*
 * Thread statistics.
 */

/* Return the time of a thread's CPU time clock. */
int64_t
util_thread_get_time_nano(thrd_t thread);

/* Return the time of the current thread's CPU time clock. */
static inline int64_t
util_current_thread_get_time_nano(void)
{
   return util_thread_get_time_nano(thrd_current());
}

static inline bool u_thread_is_self(thrd_t thread)
{
   return thrd_equal(thrd_current(), thread);
}

/*
 * util_barrier
 */

#if defined(HAVE_PTHREAD) && !defined(__APPLE__) && !defined(__HAIKU__)

typedef pthread_barrier_t util_barrier;

#else /* If the OS doesn't have its own, implement barriers using a mutex and a condvar */

typedef struct {
   unsigned count;
   unsigned waiters;
   uint64_t sequence;
   mtx_t mutex;
   cnd_t condvar;
} util_barrier;

#endif

void util_barrier_init(util_barrier *barrier, unsigned count);

void util_barrier_destroy(util_barrier *barrier);

bool util_barrier_wait(util_barrier *barrier);

/*
 * Semaphores
 */

typedef struct
{
   mtx_t mutex;
   cnd_t cond;
   int counter;
} util_semaphore;


static inline void
util_semaphore_init(util_semaphore *sema, int init_val)
{
   (void) mtx_init(&sema->mutex, mtx_plain);
   cnd_init(&sema->cond);
   sema->counter = init_val;
}

static inline void
util_semaphore_destroy(util_semaphore *sema)
{
   mtx_destroy(&sema->mutex);
   cnd_destroy(&sema->cond);
}

/** Signal/increment semaphore counter */
static inline void
util_semaphore_signal(util_semaphore *sema)
{
   mtx_lock(&sema->mutex);
   sema->counter++;
   cnd_signal(&sema->cond);
   mtx_unlock(&sema->mutex);
}

/** Wait for semaphore counter to be greater than zero */
static inline void
util_semaphore_wait(util_semaphore *sema)
{
   mtx_lock(&sema->mutex);
   while (sema->counter <= 0) {
      cnd_wait(&sema->cond, &sema->mutex);
   }
   sema->counter--;
   mtx_unlock(&sema->mutex);
}

#ifdef __cplusplus
}
#endif

#endif /* U_THREAD_H_ */
