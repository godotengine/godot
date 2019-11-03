// Copyright 2013 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Multi-threaded worker
//
// Original source:
//  http://git.chromium.org/webm/libwebp.git
//  100644 blob 7bd451b124ae3b81596abfbcc823e3cb129d3a38  src/utils/thread.h

#ifndef VPX_THREAD_H_
#define VPX_THREAD_H_

#include "./vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Set maximum decode threads to be 8 due to the limit of frame buffers
// and not enough semaphores in the emulation layer on windows.
#define MAX_DECODE_THREADS 8

#if CONFIG_MULTITHREAD

#if defined(_WIN32) && !HAVE_PTHREAD_H
#include <errno.h>  // NOLINT
#include <process.h>  // NOLINT
#include <windows.h>  // NOLINT
typedef HANDLE pthread_t;
typedef CRITICAL_SECTION pthread_mutex_t;
typedef struct {
  HANDLE waiting_sem_;
  HANDLE received_sem_;
  HANDLE signal_event_;
} pthread_cond_t;

//------------------------------------------------------------------------------
// simplistic pthread emulation layer

// _beginthreadex requires __stdcall
#define THREADFN unsigned int __stdcall
#define THREAD_RETURN(val) (unsigned int)((DWORD_PTR)val)

static INLINE int pthread_create(pthread_t* const thread, const void* attr,
                                 unsigned int (__stdcall *start)(void*),
                                 void* arg) {
  (void)attr;
  *thread = (pthread_t)_beginthreadex(NULL,   /* void *security */
                                      0,      /* unsigned stack_size */
                                      start,
                                      arg,
                                      0,      /* unsigned initflag */
                                      NULL);  /* unsigned *thrdaddr */
  if (*thread == NULL) return 1;
  SetThreadPriority(*thread, THREAD_PRIORITY_ABOVE_NORMAL);
  return 0;
}

static INLINE int pthread_join(pthread_t thread, void** value_ptr) {
  (void)value_ptr;
  return (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0 ||
          CloseHandle(thread) == 0);
}

// Mutex
static INLINE int pthread_mutex_init(pthread_mutex_t *const mutex,
                                     void* mutexattr) {
  (void)mutexattr;
  InitializeCriticalSection(mutex);
  return 0;
}

static INLINE int pthread_mutex_trylock(pthread_mutex_t *const mutex) {
  return TryEnterCriticalSection(mutex) ? 0 : EBUSY;
}

static INLINE int pthread_mutex_lock(pthread_mutex_t *const mutex) {
  EnterCriticalSection(mutex);
  return 0;
}

static INLINE int pthread_mutex_unlock(pthread_mutex_t *const mutex) {
  LeaveCriticalSection(mutex);
  return 0;
}

static INLINE int pthread_mutex_destroy(pthread_mutex_t *const mutex) {
  DeleteCriticalSection(mutex);
  return 0;
}

// Condition
static INLINE int pthread_cond_destroy(pthread_cond_t *const condition) {
  int ok = 1;
  ok &= (CloseHandle(condition->waiting_sem_) != 0);
  ok &= (CloseHandle(condition->received_sem_) != 0);
  ok &= (CloseHandle(condition->signal_event_) != 0);
  return !ok;
}

static INLINE int pthread_cond_init(pthread_cond_t *const condition,
                                    void* cond_attr) {
  (void)cond_attr;
  condition->waiting_sem_ = CreateSemaphore(NULL, 0, MAX_DECODE_THREADS, NULL);
  condition->received_sem_ = CreateSemaphore(NULL, 0, MAX_DECODE_THREADS, NULL);
  condition->signal_event_ = CreateEvent(NULL, FALSE, FALSE, NULL);
  if (condition->waiting_sem_ == NULL ||
      condition->received_sem_ == NULL ||
      condition->signal_event_ == NULL) {
    pthread_cond_destroy(condition);
    return 1;
  }
  return 0;
}

static INLINE int pthread_cond_signal(pthread_cond_t *const condition) {
  int ok = 1;
  if (WaitForSingleObject(condition->waiting_sem_, 0) == WAIT_OBJECT_0) {
    // a thread is waiting in pthread_cond_wait: allow it to be notified
    ok = SetEvent(condition->signal_event_);
    // wait until the event is consumed so the signaler cannot consume
    // the event via its own pthread_cond_wait.
    ok &= (WaitForSingleObject(condition->received_sem_, INFINITE) !=
           WAIT_OBJECT_0);
  }
  return !ok;
}

static INLINE int pthread_cond_wait(pthread_cond_t *const condition,
                                    pthread_mutex_t *const mutex) {
  int ok;
  // note that there is a consumer available so the signal isn't dropped in
  // pthread_cond_signal
  if (!ReleaseSemaphore(condition->waiting_sem_, 1, NULL))
    return 1;
  // now unlock the mutex so pthread_cond_signal may be issued
  pthread_mutex_unlock(mutex);
  ok = (WaitForSingleObject(condition->signal_event_, INFINITE) ==
        WAIT_OBJECT_0);
  ok &= ReleaseSemaphore(condition->received_sem_, 1, NULL);
  pthread_mutex_lock(mutex);
  return !ok;
}
#elif defined(__OS2__)
#define INCL_DOS
#include <os2.h>    // NOLINT

#include <errno.h>  // NOLINT
#include <stdlib.h> // NOLINT
#include <sys/builtin.h> // NOLINT

#define pthread_t TID
#define pthread_mutex_t HMTX

typedef struct {
  HEV event_sem_;
  HEV ack_sem_;
  volatile unsigned wait_count_;
} pthread_cond_t;

//------------------------------------------------------------------------------
// simplistic pthread emulation layer

#define THREADFN void *
#define THREAD_RETURN(val) (val)

typedef struct {
  void* (*start_)(void*);
  void* arg_;
} thread_arg;

static void thread_start(void* arg) {
  thread_arg targ = *(thread_arg *)arg;
  free(arg);

  targ.start_(targ.arg_);
}

static INLINE int pthread_create(pthread_t* const thread, const void* attr,
                                 void* (*start)(void*),
                                 void* arg) {
  int tid;
  thread_arg *targ = (thread_arg *)malloc(sizeof(*targ));
  if (targ == NULL) return 1;

  (void)attr;

  targ->start_ = start;
  targ->arg_ = arg;
  tid = (pthread_t)_beginthread(thread_start, NULL, 1024 * 1024, targ);
  if (tid == -1) {
    free(targ);
    return 1;
  }

  *thread = tid;
  return 0;
}

static INLINE int pthread_join(pthread_t thread, void** value_ptr) {
  (void)value_ptr;
  return DosWaitThread(&thread, DCWW_WAIT) != 0;
}

// Mutex
static INLINE int pthread_mutex_init(pthread_mutex_t *const mutex,
                                     void* mutexattr) {
  (void)mutexattr;
  return DosCreateMutexSem(NULL, mutex, 0, FALSE) != 0;
}

static INLINE int pthread_mutex_trylock(pthread_mutex_t *const mutex) {
  return DosRequestMutexSem(*mutex, SEM_IMMEDIATE_RETURN) == 0 ? 0 : EBUSY;
}

static INLINE int pthread_mutex_lock(pthread_mutex_t *const mutex) {
  return DosRequestMutexSem(*mutex, SEM_INDEFINITE_WAIT) != 0;
}

static INLINE int pthread_mutex_unlock(pthread_mutex_t *const mutex) {
  return DosReleaseMutexSem(*mutex) != 0;
}

static INLINE int pthread_mutex_destroy(pthread_mutex_t *const mutex) {
  return DosCloseMutexSem(*mutex) != 0;
}

// Condition
static INLINE int pthread_cond_destroy(pthread_cond_t *const condition) {
  int ok = 1;
  ok &= DosCloseEventSem(condition->event_sem_) == 0;
  ok &= DosCloseEventSem(condition->ack_sem_) == 0;
  return !ok;
}

static INLINE int pthread_cond_init(pthread_cond_t *const condition,
                                    void* cond_attr) {
  int ok = 1;
  (void)cond_attr;

  ok &= DosCreateEventSem(NULL, &condition->event_sem_, DCE_POSTONE, FALSE)
          == 0;
  ok &= DosCreateEventSem(NULL, &condition->ack_sem_, DCE_POSTONE, FALSE) == 0;
  if (!ok) {
    pthread_cond_destroy(condition);
    return 1;
  }
  condition->wait_count_ = 0;
  return 0;
}

static INLINE int pthread_cond_signal(pthread_cond_t *const condition) {
  int ok = 1;

  if (!__atomic_cmpxchg32(&condition->wait_count_, 0, 0)) {
    ok &= DosPostEventSem(condition->event_sem_) == 0;
    ok &= DosWaitEventSem(condition->ack_sem_, SEM_INDEFINITE_WAIT) == 0;
  }

  return !ok;
}

static INLINE int pthread_cond_broadcast(pthread_cond_t *const condition) {
  int ok = 1;

  while (!__atomic_cmpxchg32(&condition->wait_count_, 0, 0))
      ok &= pthread_cond_signal(condition) == 0;

  return !ok;
}

static INLINE int pthread_cond_wait(pthread_cond_t *const condition,
                                    pthread_mutex_t *const mutex) {
  int ok = 1;

  __atomic_increment(&condition->wait_count_);

  ok &= pthread_mutex_unlock(mutex) == 0;

  ok &= DosWaitEventSem(condition->event_sem_, SEM_INDEFINITE_WAIT) == 0;

  __atomic_decrement(&condition->wait_count_);

  ok &= DosPostEventSem(condition->ack_sem_) == 0;

  pthread_mutex_lock(mutex);

  return !ok;
}
#else  // _WIN32
#include <pthread.h> // NOLINT
# define THREADFN void*
# define THREAD_RETURN(val) val
#endif

#endif  // CONFIG_MULTITHREAD

// State of the worker thread object
typedef enum {
  NOT_OK = 0,   // object is unusable
  OK,           // ready to work
  WORK          // busy finishing the current task
} VPxWorkerStatus;

// Function to be called by the worker thread. Takes two opaque pointers as
// arguments (data1 and data2), and should return false in case of error.
typedef int (*VPxWorkerHook)(void*, void*);

// Platform-dependent implementation details for the worker.
typedef struct VPxWorkerImpl VPxWorkerImpl;

// Synchronization object used to launch job in the worker thread
typedef struct {
  VPxWorkerImpl *impl_;
  VPxWorkerStatus status_;
  VPxWorkerHook hook;     // hook to call
  void *data1;            // first argument passed to 'hook'
  void *data2;            // second argument passed to 'hook'
  int had_error;          // return value of the last call to 'hook'
} VPxWorker;

// The interface for all thread-worker related functions. All these functions
// must be implemented.
typedef struct {
  // Must be called first, before any other method.
  void (*init)(VPxWorker *const worker);
  // Must be called to initialize the object and spawn the thread. Re-entrant.
  // Will potentially launch the thread. Returns false in case of error.
  int (*reset)(VPxWorker *const worker);
  // Makes sure the previous work is finished. Returns true if worker->had_error
  // was not set and no error condition was triggered by the working thread.
  int (*sync)(VPxWorker *const worker);
  // Triggers the thread to call hook() with data1 and data2 arguments. These
  // hook/data1/data2 values can be changed at any time before calling this
  // function, but not be changed afterward until the next call to Sync().
  void (*launch)(VPxWorker *const worker);
  // This function is similar to launch() except that it calls the
  // hook directly instead of using a thread. Convenient to bypass the thread
  // mechanism while still using the VPxWorker structs. sync() must
  // still be called afterward (for error reporting).
  void (*execute)(VPxWorker *const worker);
  // Kill the thread and terminate the object. To use the object again, one
  // must call reset() again.
  void (*end)(VPxWorker *const worker);
} VPxWorkerInterface;

// Install a new set of threading functions, overriding the defaults. This
// should be done before any workers are started, i.e., before any encoding or
// decoding takes place. The contents of the interface struct are copied, it
// is safe to free the corresponding memory after this call. This function is
// not thread-safe. Return false in case of invalid pointer or methods.
int vpx_set_worker_interface(const VPxWorkerInterface *const winterface);

// Retrieve the currently set thread worker interface.
const VPxWorkerInterface *vpx_get_worker_interface(void);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // VPX_THREAD_H_
