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
//  https://chromium.googlesource.com/webm/libwebp

#ifndef VPX_VPX_UTIL_VPX_THREAD_H_
#define VPX_VPX_UTIL_VPX_THREAD_H_

#ifdef __cplusplus
extern "C" {
#endif

// State of the worker thread object
typedef enum {
  VPX_WORKER_STATUS_NOT_OK = 0,  // object is unusable
  VPX_WORKER_STATUS_OK,          // ready to work
  VPX_WORKER_STATUS_WORKING      // busy finishing the current task
} VPxWorkerStatus;

// Function to be called by the worker thread. Takes two opaque pointers as
// arguments (data1 and data2). Should return true on success and return false
// in case of error.
typedef int (*VPxWorkerHook)(void *, void *);

// Platform-dependent implementation details for the worker.
typedef struct VPxWorkerImpl VPxWorkerImpl;

// Synchronization object used to launch job in the worker thread
typedef struct {
  VPxWorkerImpl *impl_;
  VPxWorkerStatus status_;
  // Thread name for the debugger. If not NULL, must point to a string that
  // outlives the worker thread. For portability, use a name <= 15 characters
  // long (not including the terminating NUL character).
  const char *thread_name;
  VPxWorkerHook hook;  // hook to call
  void *data1;         // first argument passed to 'hook'
  void *data2;         // second argument passed to 'hook'
  int had_error;       // true if a call to 'hook' returned false
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
}  // extern "C"
#endif

#endif  // VPX_VPX_UTIL_VPX_THREAD_H_
