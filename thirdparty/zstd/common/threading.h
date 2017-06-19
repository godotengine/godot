
/**
 * Copyright (c) 2016 Tino Reichardt
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 *
 * You can contact the author at:
 * - zstdmt source repository: https://github.com/mcmilk/zstdmt
 */

#ifndef THREADING_H_938743
#define THREADING_H_938743

#if defined (__cplusplus)
extern "C" {
#endif

#if defined(ZSTD_MULTITHREAD) && defined(_WIN32)

/**
 * Windows minimalist Pthread Wrapper, based on :
 * http://www.cse.wustl.edu/~schmidt/win32-cv-1.html
 */
#ifdef WINVER
#  undef WINVER
#endif
#define WINVER       0x0600

#ifdef _WIN32_WINNT
#  undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0600

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

/* mutex */
#define pthread_mutex_t           CRITICAL_SECTION
#define pthread_mutex_init(a,b)   InitializeCriticalSection((a))
#define pthread_mutex_destroy(a)  DeleteCriticalSection((a))
#define pthread_mutex_lock(a)     EnterCriticalSection((a))
#define pthread_mutex_unlock(a)   LeaveCriticalSection((a))

/* condition variable */
#define pthread_cond_t             CONDITION_VARIABLE
#define pthread_cond_init(a, b)    InitializeConditionVariable((a))
#define pthread_cond_destroy(a)    /* No delete */
#define pthread_cond_wait(a, b)    SleepConditionVariableCS((a), (b), INFINITE)
#define pthread_cond_signal(a)     WakeConditionVariable((a))
#define pthread_cond_broadcast(a)  WakeAllConditionVariable((a))

/* pthread_create() and pthread_join() */
typedef struct {
    HANDLE handle;
    void* (*start_routine)(void*);
    void* arg;
} pthread_t;

int pthread_create(pthread_t* thread, const void* unused,
                   void* (*start_routine) (void*), void* arg);

#define pthread_join(a, b) _pthread_join(&(a), (b))
int _pthread_join(pthread_t* thread, void** value_ptr);

/**
 * add here more wrappers as required
 */


#elif defined(ZSTD_MULTITHREAD)   /* posix assumed ; need a better detection method */
/* ===   POSIX Systems   === */
#  include <pthread.h>

#else  /* ZSTD_MULTITHREAD not defined */
/* No multithreading support */

#define pthread_mutex_t int   /* #define rather than typedef, as sometimes pthread support is implicit, resulting in duplicated symbols */
#define pthread_mutex_init(a,b)
#define pthread_mutex_destroy(a)
#define pthread_mutex_lock(a)
#define pthread_mutex_unlock(a)

#define pthread_cond_t int
#define pthread_cond_init(a,b)
#define pthread_cond_destroy(a)
#define pthread_cond_wait(a,b)
#define pthread_cond_signal(a)
#define pthread_cond_broadcast(a)

/* do not use pthread_t */

#endif /* ZSTD_MULTITHREAD */

#if defined (__cplusplus)
}
#endif

#endif /* THREADING_H_938743 */
