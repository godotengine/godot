// Copyright 2011 Google Inc. All Rights Reserved.
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
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <string.h>   // for memset()
#include "src/utils/thread_utils.h"
#include "src/utils/utils.h"

#ifdef WEBP_USE_THREAD

#if defined(_WIN32)

#include <windows.h>
typedef HANDLE pthread_t;
typedef CRITICAL_SECTION pthread_mutex_t;

#if _WIN32_WINNT >= 0x0600  // Windows Vista / Server 2008 or greater
#define USE_WINDOWS_CONDITION_VARIABLE
typedef CONDITION_VARIABLE pthread_cond_t;
#else
typedef struct {
  HANDLE waiting_sem_;
  HANDLE received_sem_;
  HANDLE signal_event_;
} pthread_cond_t;
#endif  // _WIN32_WINNT >= 0x600

#ifndef WINAPI_FAMILY_PARTITION
#define WINAPI_PARTITION_DESKTOP 1
#define WINAPI_FAMILY_PARTITION(x) x
#endif

#if !WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#define USE_CREATE_THREAD
#endif

#else  // !_WIN32

#include <pthread.h>

#endif  // _WIN32

typedef struct {
  pthread_mutex_t mutex_;
  pthread_cond_t  condition_;
  pthread_t       thread_;
} WebPWorkerImpl;

#if defined(_WIN32)

//------------------------------------------------------------------------------
// simplistic pthread emulation layer

#include <process.h>

// _beginthreadex requires __stdcall
#define THREADFN unsigned int __stdcall
#define THREAD_RETURN(val) (unsigned int)((DWORD_PTR)val)

#if _WIN32_WINNT >= 0x0501  // Windows XP or greater
#define WaitForSingleObject(obj, timeout) \
  WaitForSingleObjectEx(obj, timeout, FALSE /*bAlertable*/)
#endif

static int pthread_create(pthread_t* const thread, const void* attr,
                          unsigned int (__stdcall *start)(void*), void* arg) {
  (void)attr;
#ifdef USE_CREATE_THREAD
  *thread = CreateThread(NULL,   /* lpThreadAttributes */
                         0,      /* dwStackSize */
                         start,
                         arg,
                         0,      /* dwStackSize */
                         NULL);  /* lpThreadId */
#else
  *thread = (pthread_t)_beginthreadex(NULL,   /* void *security */
                                      0,      /* unsigned stack_size */
                                      start,
                                      arg,
                                      0,      /* unsigned initflag */
                                      NULL);  /* unsigned *thrdaddr */
#endif
  if (*thread == NULL) return 1;
  SetThreadPriority(*thread, THREAD_PRIORITY_ABOVE_NORMAL);
  return 0;
}

static int pthread_join(pthread_t thread, void** value_ptr) {
  (void)value_ptr;
  return (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0 ||
          CloseHandle(thread) == 0);
}

// Mutex
static int pthread_mutex_init(pthread_mutex_t* const mutex, void* mutexattr) {
  (void)mutexattr;
#if _WIN32_WINNT >= 0x0600  // Windows Vista / Server 2008 or greater
  InitializeCriticalSectionEx(mutex, 0 /*dwSpinCount*/, 0 /*Flags*/);
#else
  InitializeCriticalSection(mutex);
#endif
  return 0;
}

static int pthread_mutex_lock(pthread_mutex_t* const mutex) {
  EnterCriticalSection(mutex);
  return 0;
}

static int pthread_mutex_unlock(pthread_mutex_t* const mutex) {
  LeaveCriticalSection(mutex);
  return 0;
}

static int pthread_mutex_destroy(pthread_mutex_t* const mutex) {
  DeleteCriticalSection(mutex);
  return 0;
}

// Condition
static int pthread_cond_destroy(pthread_cond_t* const condition) {
  int ok = 1;
#ifdef USE_WINDOWS_CONDITION_VARIABLE
  (void)condition;
#else
  ok &= (CloseHandle(condition->waiting_sem_) != 0);
  ok &= (CloseHandle(condition->received_sem_) != 0);
  ok &= (CloseHandle(condition->signal_event_) != 0);
#endif
  return !ok;
}

static int pthread_cond_init(pthread_cond_t* const condition, void* cond_attr) {
  (void)cond_attr;
#ifdef USE_WINDOWS_CONDITION_VARIABLE
  InitializeConditionVariable(condition);
#else
  condition->waiting_sem_ = CreateSemaphore(NULL, 0, 1, NULL);
  condition->received_sem_ = CreateSemaphore(NULL, 0, 1, NULL);
  condition->signal_event_ = CreateEvent(NULL, FALSE, FALSE, NULL);
  if (condition->waiting_sem_ == NULL ||
      condition->received_sem_ == NULL ||
      condition->signal_event_ == NULL) {
    pthread_cond_destroy(condition);
    return 1;
  }
#endif
  return 0;
}

static int pthread_cond_signal(pthread_cond_t* const condition) {
  int ok = 1;
#ifdef USE_WINDOWS_CONDITION_VARIABLE
  WakeConditionVariable(condition);
#else
  if (WaitForSingleObject(condition->waiting_sem_, 0) == WAIT_OBJECT_0) {
    // a thread is waiting in pthread_cond_wait: allow it to be notified
    ok = SetEvent(condition->signal_event_);
    // wait until the event is consumed so the signaler cannot consume
    // the event via its own pthread_cond_wait.
    ok &= (WaitForSingleObject(condition->received_sem_, INFINITE) !=
           WAIT_OBJECT_0);
  }
#endif
  return !ok;
}

static int pthread_cond_wait(pthread_cond_t* const condition,
                             pthread_mutex_t* const mutex) {
  int ok;
#ifdef USE_WINDOWS_CONDITION_VARIABLE
  ok = SleepConditionVariableCS(condition, mutex, INFINITE);
#else
  // note that there is a consumer available so the signal isn't dropped in
  // pthread_cond_signal
  if (!ReleaseSemaphore(condition->waiting_sem_, 1, NULL)) return 1;
  // now unlock the mutex so pthread_cond_signal may be issued
  pthread_mutex_unlock(mutex);
  ok = (WaitForSingleObject(condition->signal_event_, INFINITE) ==
        WAIT_OBJECT_0);
  ok &= ReleaseSemaphore(condition->received_sem_, 1, NULL);
  pthread_mutex_lock(mutex);
#endif
  return !ok;
}

#else  // !_WIN32
# define THREADFN void*
# define THREAD_RETURN(val) val
#endif  // _WIN32

//------------------------------------------------------------------------------

static THREADFN ThreadLoop(void* ptr) {
  WebPWorker* const worker = (WebPWorker*)ptr;
  WebPWorkerImpl* const impl = (WebPWorkerImpl*)worker->impl_;
  int done = 0;
  while (!done) {
    pthread_mutex_lock(&impl->mutex_);
    while (worker->status_ == OK) {   // wait in idling mode
      pthread_cond_wait(&impl->condition_, &impl->mutex_);
    }
    if (worker->status_ == WORK) {
      WebPGetWorkerInterface()->Execute(worker);
      worker->status_ = OK;
    } else if (worker->status_ == NOT_OK) {   // finish the worker
      done = 1;
    }
    // signal to the main thread that we're done (for Sync())
    pthread_cond_signal(&impl->condition_);
    pthread_mutex_unlock(&impl->mutex_);
  }
  return THREAD_RETURN(NULL);    // Thread is finished
}

// main thread state control
static void ChangeState(WebPWorker* const worker, WebPWorkerStatus new_status) {
  // No-op when attempting to change state on a thread that didn't come up.
  // Checking status_ without acquiring the lock first would result in a data
  // race.
  WebPWorkerImpl* const impl = (WebPWorkerImpl*)worker->impl_;
  if (impl == NULL) return;

  pthread_mutex_lock(&impl->mutex_);
  if (worker->status_ >= OK) {
    // wait for the worker to finish
    while (worker->status_ != OK) {
      pthread_cond_wait(&impl->condition_, &impl->mutex_);
    }
    // assign new status and release the working thread if needed
    if (new_status != OK) {
      worker->status_ = new_status;
      pthread_cond_signal(&impl->condition_);
    }
  }
  pthread_mutex_unlock(&impl->mutex_);
}

#endif  // WEBP_USE_THREAD

//------------------------------------------------------------------------------

static void Init(WebPWorker* const worker) {
  memset(worker, 0, sizeof(*worker));
  worker->status_ = NOT_OK;
}

static int Sync(WebPWorker* const worker) {
#ifdef WEBP_USE_THREAD
  ChangeState(worker, OK);
#endif
  assert(worker->status_ <= OK);
  return !worker->had_error;
}

static int Reset(WebPWorker* const worker) {
  int ok = 1;
  worker->had_error = 0;
  if (worker->status_ < OK) {
#ifdef WEBP_USE_THREAD
    WebPWorkerImpl* const impl =
        (WebPWorkerImpl*)WebPSafeCalloc(1, sizeof(WebPWorkerImpl));
    worker->impl_ = (void*)impl;
    if (worker->impl_ == NULL) {
      return 0;
    }
    if (pthread_mutex_init(&impl->mutex_, NULL)) {
      goto Error;
    }
    if (pthread_cond_init(&impl->condition_, NULL)) {
      pthread_mutex_destroy(&impl->mutex_);
      goto Error;
    }
    pthread_mutex_lock(&impl->mutex_);
    ok = !pthread_create(&impl->thread_, NULL, ThreadLoop, worker);
    if (ok) worker->status_ = OK;
    pthread_mutex_unlock(&impl->mutex_);
    if (!ok) {
      pthread_mutex_destroy(&impl->mutex_);
      pthread_cond_destroy(&impl->condition_);
 Error:
      WebPSafeFree(impl);
      worker->impl_ = NULL;
      return 0;
    }
#else
    worker->status_ = OK;
#endif
  } else if (worker->status_ > OK) {
    ok = Sync(worker);
  }
  assert(!ok || (worker->status_ == OK));
  return ok;
}

static void Execute(WebPWorker* const worker) {
  if (worker->hook != NULL) {
    worker->had_error |= !worker->hook(worker->data1, worker->data2);
  }
}

static void Launch(WebPWorker* const worker) {
#ifdef WEBP_USE_THREAD
  ChangeState(worker, WORK);
#else
  Execute(worker);
#endif
}

static void End(WebPWorker* const worker) {
#ifdef WEBP_USE_THREAD
  if (worker->impl_ != NULL) {
    WebPWorkerImpl* const impl = (WebPWorkerImpl*)worker->impl_;
    ChangeState(worker, NOT_OK);
    pthread_join(impl->thread_, NULL);
    pthread_mutex_destroy(&impl->mutex_);
    pthread_cond_destroy(&impl->condition_);
    WebPSafeFree(impl);
    worker->impl_ = NULL;
  }
#else
  worker->status_ = NOT_OK;
  assert(worker->impl_ == NULL);
#endif
  assert(worker->status_ == NOT_OK);
}

//------------------------------------------------------------------------------

static WebPWorkerInterface g_worker_interface = {
  Init, Reset, Sync, Launch, Execute, End
};

int WebPSetWorkerInterface(const WebPWorkerInterface* const winterface) {
  if (winterface == NULL ||
      winterface->Init == NULL || winterface->Reset == NULL ||
      winterface->Sync == NULL || winterface->Launch == NULL ||
      winterface->Execute == NULL || winterface->End == NULL) {
    return 0;
  }
  g_worker_interface = *winterface;
  return 1;
}

const WebPWorkerInterface* WebPGetWorkerInterface(void) {
  return &g_worker_interface;
}

//------------------------------------------------------------------------------
