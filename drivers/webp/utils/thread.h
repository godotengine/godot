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

#ifndef WEBP_UTILS_THREAD_H_
#define WEBP_UTILS_THREAD_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef WEBP_USE_THREAD

#if defined(_WIN32)

#include <windows.h>
typedef HANDLE pthread_t;
typedef CRITICAL_SECTION pthread_mutex_t;
typedef struct {
  HANDLE waiting_sem_;
  HANDLE received_sem_;
  HANDLE signal_event_;
} pthread_cond_t;

#else

#include <pthread.h>

#endif    /* _WIN32 */
#endif    /* WEBP_USE_THREAD */

// State of the worker thread object
typedef enum {
  NOT_OK = 0,   // object is unusable
  OK,           // ready to work
  WORK          // busy finishing the current task
} WebPWorkerStatus;

// Function to be called by the worker thread. Takes two opaque pointers as
// arguments (data1 and data2), and should return false in case of error.
typedef int (*WebPWorkerHook)(void*, void*);

// Synchronize object used to launch job in the worker thread
typedef struct {
#ifdef WEBP_USE_THREAD
  pthread_mutex_t mutex_;
  pthread_cond_t  condition_;
  pthread_t       thread_;
#endif
  WebPWorkerStatus status_;
  WebPWorkerHook hook;    // hook to call
  void* data1;            // first argument passed to 'hook'
  void* data2;            // second argument passed to 'hook'
  int had_error;          // return value of the last call to 'hook'
} WebPWorker;

// Must be called first, before any other method.
void WebPWorkerInit(WebPWorker* const worker);
// Must be called to initialize the object and spawn the thread. Re-entrant.
// Will potentially launch the thread. Returns false in case of error.
int WebPWorkerReset(WebPWorker* const worker);
// Makes sure the previous work is finished. Returns true if worker->had_error
// was not set and no error condition was triggered by the working thread.
int WebPWorkerSync(WebPWorker* const worker);
// Triggers the thread to call hook() with data1 and data2 argument. These
// hook/data1/data2 can be changed at any time before calling this function,
// but not be changed afterward until the next call to WebPWorkerSync().
void WebPWorkerLaunch(WebPWorker* const worker);
// This function is similar to WebPWorkerLaunch() except that it calls the
// hook directly instead of using a thread. Convenient to bypass the thread
// mechanism while still using the WebPWorker structs. WebPWorkerSync() must
// still be called afterward (for error reporting).
void WebPWorkerExecute(WebPWorker* const worker);
// Kill the thread and terminate the object. To use the object again, one
// must call WebPWorkerReset() again.
void WebPWorkerEnd(WebPWorker* const worker);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_THREAD_H_ */
