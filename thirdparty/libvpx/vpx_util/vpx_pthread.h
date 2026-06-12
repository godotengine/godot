// Copyright 2024 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// pthread.h wrapper

#ifndef VPX_VPX_UTIL_VPX_PTHREAD_H_
#define VPX_VPX_UTIL_VPX_PTHREAD_H_

#include "./vpx_config.h"

#if CONFIG_MULTITHREAD

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && !HAVE_PTHREAD_H
// Prevent leaking max/min macros.
#undef NOMINMAX
#define NOMINMAX
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#include <process.h>  // NOLINT
#include <stddef.h>   // NOLINT
#include <windows.h>  // NOLINT
typedef HANDLE pthread_t;
typedef SRWLOCK pthread_mutex_t;

#if _WIN32_WINNT < 0x0600
#error _WIN32_WINNT must target Windows Vista / Server 2008 or newer.
#endif
typedef CONDITION_VARIABLE pthread_cond_t;

#ifndef WINAPI_FAMILY_PARTITION
#define WINAPI_PARTITION_DESKTOP 1
#define WINAPI_FAMILY_PARTITION(x) x
#endif

#if !WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#define USE_CREATE_THREAD
#endif

//------------------------------------------------------------------------------
// simplistic pthread emulation layer

// _beginthreadex requires __stdcall
#if defined(__GNUC__) && \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2))
#define THREADFN __attribute__((force_align_arg_pointer)) unsigned int __stdcall
#else
#define THREADFN unsigned int __stdcall
#endif
#define THREAD_EXIT_SUCCESS 0

static INLINE int pthread_create(pthread_t *const thread, const void *attr,
                                 unsigned int(__stdcall *start)(void *),
                                 void *arg) {
  (void)attr;
#ifdef USE_CREATE_THREAD
  *thread = CreateThread(NULL,          /* lpThreadAttributes */
                         0,             /* dwStackSize */
                         start, arg, 0, /* dwStackSize */
                         NULL);         /* lpThreadId */
#else
  *thread = (pthread_t)_beginthreadex(NULL,          /* void *security */
                                      0,             /* unsigned stack_size */
                                      start, arg, 0, /* unsigned initflag */
                                      NULL);         /* unsigned *thrdaddr */
#endif
  if (*thread == NULL) return 1;
  SetThreadPriority(*thread, THREAD_PRIORITY_ABOVE_NORMAL);
  return 0;
}

static INLINE int pthread_join(pthread_t thread, void **value_ptr) {
  (void)value_ptr;
  return (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0 ||
          CloseHandle(thread) == 0);
}

// Mutex
static INLINE int pthread_mutex_init(pthread_mutex_t *const mutex,
                                     void *mutexattr) {
  (void)mutexattr;
  InitializeSRWLock(mutex);
  return 0;
}

static INLINE int pthread_mutex_lock(pthread_mutex_t *const mutex) {
  AcquireSRWLockExclusive(mutex);
  return 0;
}

static INLINE int pthread_mutex_unlock(pthread_mutex_t *const mutex) {
  ReleaseSRWLockExclusive(mutex);
  return 0;
}

static INLINE int pthread_mutex_destroy(pthread_mutex_t *const mutex) {
  (void)mutex;
  return 0;
}

// Condition
static INLINE int pthread_cond_destroy(pthread_cond_t *const condition) {
  (void)condition;
  return 0;
}

static INLINE int pthread_cond_init(pthread_cond_t *const condition,
                                    void *cond_attr) {
  (void)cond_attr;
  InitializeConditionVariable(condition);
  return 0;
}

static INLINE int pthread_cond_signal(pthread_cond_t *const condition) {
  WakeConditionVariable(condition);
  return 0;
}

static INLINE int pthread_cond_broadcast(pthread_cond_t *const condition) {
  WakeAllConditionVariable(condition);
  return 0;
}

static INLINE int pthread_cond_wait(pthread_cond_t *const condition,
                                    pthread_mutex_t *const mutex) {
  const int ok = SleepConditionVariableSRW(condition, mutex, INFINITE, 0);
  return !ok;
}
#else                 // _WIN32
#include <pthread.h>  // NOLINT
#define THREADFN void *
#define THREAD_EXIT_SUCCESS NULL
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CONFIG_MULTITHREAD

#endif  // VPX_VPX_UTIL_VPX_PTHREAD_H_
