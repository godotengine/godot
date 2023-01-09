/**************************************************************************
 *
 * Copyright 2020 Lag Free Games, LLC
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef CND_MONOTONIC_H
#define CND_MONOTONIC_H

#include "c11/threads.h"
#include "util/os_time.h"

#ifdef __cplusplus
extern "C" {
#endif

struct u_cnd_monotonic
{
#ifdef _WIN32
   CONDITION_VARIABLE condvar;
#else
   pthread_cond_t cond;
#endif
};

static inline int
u_cnd_monotonic_init(struct u_cnd_monotonic *cond)
{
   assert(cond != NULL);

#ifdef _WIN32
   InitializeConditionVariable(&cond->condvar);
   return thrd_success;
#else
   int ret = thrd_error;
   pthread_condattr_t condattr;
   if (pthread_condattr_init(&condattr) == 0) {
      if ((pthread_condattr_setclock(&condattr, CLOCK_MONOTONIC) == 0) &&
         (pthread_cond_init(&cond->cond, &condattr) == 0)) {
         ret = thrd_success;
      }

      pthread_condattr_destroy(&condattr);
   }

   return ret;
#endif
}

static inline void
u_cnd_monotonic_destroy(struct u_cnd_monotonic *cond)
{
   assert(cond != NULL);

#ifdef _WIN32
   // Do nothing
#else
   pthread_cond_destroy(&cond->cond);
#endif
}

static inline int
u_cnd_monotonic_broadcast(struct u_cnd_monotonic *cond)
{
   assert(cond != NULL);

#ifdef _WIN32
   WakeAllConditionVariable(&cond->condvar);
   return thrd_success;
#else
   return (pthread_cond_broadcast(&cond->cond) == 0) ? thrd_success : thrd_error;
#endif
}

static inline int
u_cnd_monotonic_signal(struct u_cnd_monotonic *cond)
{
   assert(cond != NULL);

#ifdef _WIN32
   WakeConditionVariable(&cond->condvar);
   return thrd_success;
#else
   return (pthread_cond_signal(&cond->cond) == 0) ? thrd_success : thrd_error;
#endif
}

static inline int
u_cnd_monotonic_timedwait(struct u_cnd_monotonic *cond, mtx_t *mtx, const struct timespec *abs_time)
{
   assert(cond != NULL);
   assert(mtx != NULL);
   assert(abs_time != NULL);

#ifdef _WIN32
   const uint64_t future = (abs_time->tv_sec * 1000) + (abs_time->tv_nsec / 1000000);
   const uint64_t now = os_time_get_nano() / 1000000;
   const DWORD timeout = (future > now) ? (DWORD)(future - now) : 0;
   if (SleepConditionVariableCS(&cond->condvar, mtx, timeout))
      return thrd_success;
   return (GetLastError() == ERROR_TIMEOUT) ? thrd_timedout : thrd_error;
#else
   int rt = pthread_cond_timedwait(&cond->cond, mtx, abs_time);
   if (rt == ETIMEDOUT)
      return thrd_timedout;
   return (rt == 0) ? thrd_success : thrd_error;
#endif
}

static inline int
u_cnd_monotonic_wait(struct u_cnd_monotonic *cond, mtx_t *mtx)
{
   assert(cond != NULL);
   assert(mtx != NULL);

#ifdef _WIN32
   SleepConditionVariableCS(&cond->condvar, mtx, INFINITE);
   return thrd_success;
#else
   return (pthread_cond_wait(&cond->cond, mtx) == 0) ? thrd_success : thrd_error;
#endif
}

#ifdef __cplusplus
}
#endif

#endif
