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
#include "../webp/config.h"
#endif

#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// State of the worker thread object
typedef enum {
  NOT_OK = 0,   // object is unusable
  OK,           // ready to work
  WORK          // busy finishing the current task
} WebPWorkerStatus;

// Function to be called by the worker thread. Takes two opaque pointers as
// arguments (data1 and data2), and should return false in case of error.
typedef int (*WebPWorkerHook)(void*, void*);

// Platform-dependent implementation details for the worker.
typedef struct WebPWorkerImpl WebPWorkerImpl;

// Synchronization object used to launch job in the worker thread
typedef struct {
  WebPWorkerImpl* impl_;
  WebPWorkerStatus status_;
  WebPWorkerHook hook;    // hook to call
  void* data1;            // first argument passed to 'hook'
  void* data2;            // second argument passed to 'hook'
  int had_error;          // return value of the last call to 'hook'
} WebPWorker;

// The interface for all thread-worker related functions. All these functions
// must be implemented.
typedef struct {
  // Must be called first, before any other method.
  void (*Init)(WebPWorker* const worker);
  // Must be called to initialize the object and spawn the thread. Re-entrant.
  // Will potentially launch the thread. Returns false in case of error.
  int (*Reset)(WebPWorker* const worker);
  // Makes sure the previous work is finished. Returns true if worker->had_error
  // was not set and no error condition was triggered by the working thread.
  int (*Sync)(WebPWorker* const worker);
  // Triggers the thread to call hook() with data1 and data2 arguments. These
  // hook/data1/data2 values can be changed at any time before calling this
  // function, but not be changed afterward until the next call to Sync().
  void (*Launch)(WebPWorker* const worker);
  // This function is similar to Launch() except that it calls the
  // hook directly instead of using a thread. Convenient to bypass the thread
  // mechanism while still using the WebPWorker structs. Sync() must
  // still be called afterward (for error reporting).
  void (*Execute)(WebPWorker* const worker);
  // Kill the thread and terminate the object. To use the object again, one
  // must call Reset() again.
  void (*End)(WebPWorker* const worker);
} WebPWorkerInterface;

// Install a new set of threading functions, overriding the defaults. This
// should be done before any workers are started, i.e., before any encoding or
// decoding takes place. The contents of the interface struct are copied, it
// is safe to free the corresponding memory after this call. This function is
// not thread-safe. Return false in case of invalid pointer or methods.
WEBP_EXTERN(int) WebPSetWorkerInterface(
    const WebPWorkerInterface* const winterface);

// Retrieve the currently set thread worker interface.
WEBP_EXTERN(const WebPWorkerInterface*) WebPGetWorkerInterface(void);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_THREAD_H_ */
