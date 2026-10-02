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

// Enable GNU extensions in glibc so that we can call pthread_setname_np().
// This must be before any #include statements.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <assert.h>
#include <string.h>  // for memset()
#include "./vpx_config.h"
#include "./vpx_thread.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_util/vpx_pthread.h"

#if CONFIG_MULTITHREAD

struct VPxWorkerImpl {
  pthread_mutex_t mutex_;
  pthread_cond_t condition_;
  pthread_t thread_;
};

//------------------------------------------------------------------------------

static void execute(VPxWorker *const worker);  // Forward declaration.

static THREADFN thread_loop(void *ptr) {
  VPxWorker *const worker = (VPxWorker *)ptr;
#ifdef __APPLE__
  if (worker->thread_name != NULL) {
    // Apple's version of pthread_setname_np takes one argument and operates on
    // the current thread only. The maximum size of the thread_name buffer was
    // noted in the Chromium source code and was confirmed by experiments. If
    // thread_name is too long, pthread_setname_np returns -1 with errno
    // ENAMETOOLONG (63).
    char thread_name[64];
    strncpy(thread_name, worker->thread_name, sizeof(thread_name) - 1);
    thread_name[sizeof(thread_name) - 1] = '\0';
    pthread_setname_np(thread_name);
  }
#elif (defined(__GLIBC__) && !defined(__GNU__)) || defined(__BIONIC__)
  if (worker->thread_name != NULL) {
    // Linux and Android require names (with nul) fit in 16 chars, otherwise
    // pthread_setname_np() returns ERANGE (34).
    char thread_name[16];
    strncpy(thread_name, worker->thread_name, sizeof(thread_name) - 1);
    thread_name[sizeof(thread_name) - 1] = '\0';
    pthread_setname_np(pthread_self(), thread_name);
  }
#endif
  pthread_mutex_lock(&worker->impl_->mutex_);
  for (;;) {
    while (worker->status_ == VPX_WORKER_STATUS_OK) {  // wait in idling mode
      pthread_cond_wait(&worker->impl_->condition_, &worker->impl_->mutex_);
    }
    if (worker->status_ == VPX_WORKER_STATUS_WORKING) {
      // When worker->status_ is VPX_WORKER_STATUS_WORKING, the main thread
      // doesn't change worker->status_ and will wait until the worker changes
      // worker->status_ to VPX_WORKER_STATUS_OK. See change_state(). So the
      // worker can safely call execute() without holding worker->impl_->mutex_.
      // When the worker reacquires worker->impl_->mutex_, worker->status_ must
      // still be VPX_WORKER_STATUS_WORKING.
      pthread_mutex_unlock(&worker->impl_->mutex_);
      execute(worker);
      pthread_mutex_lock(&worker->impl_->mutex_);
      assert(worker->status_ == VPX_WORKER_STATUS_WORKING);
      worker->status_ = VPX_WORKER_STATUS_OK;
      // signal to the main thread that we're done (for sync())
      pthread_cond_signal(&worker->impl_->condition_);
    } else {
      assert(worker->status_ == VPX_WORKER_STATUS_NOT_OK);  // finish the worker
      break;
    }
  }
  pthread_mutex_unlock(&worker->impl_->mutex_);
  return THREAD_EXIT_SUCCESS;  // Thread is finished
}

// main thread state control
static void change_state(VPxWorker *const worker, VPxWorkerStatus new_status) {
  // No-op when attempting to change state on a thread that didn't come up.
  // Checking status_ without acquiring the lock first would result in a data
  // race.
  if (worker->impl_ == NULL) return;

  pthread_mutex_lock(&worker->impl_->mutex_);
  if (worker->status_ >= VPX_WORKER_STATUS_OK) {
    // wait for the worker to finish
    while (worker->status_ != VPX_WORKER_STATUS_OK) {
      pthread_cond_wait(&worker->impl_->condition_, &worker->impl_->mutex_);
    }
    // assign new status and release the working thread if needed
    if (new_status != VPX_WORKER_STATUS_OK) {
      worker->status_ = new_status;
      pthread_cond_signal(&worker->impl_->condition_);
    }
  }
  pthread_mutex_unlock(&worker->impl_->mutex_);
}

#endif  // CONFIG_MULTITHREAD

//------------------------------------------------------------------------------

static void init(VPxWorker *const worker) {
  memset(worker, 0, sizeof(*worker));
  worker->status_ = VPX_WORKER_STATUS_NOT_OK;
}

static int sync(VPxWorker *const worker) {
#if CONFIG_MULTITHREAD
  change_state(worker, VPX_WORKER_STATUS_OK);
#endif
  assert(worker->status_ <= VPX_WORKER_STATUS_OK);
  return !worker->had_error;
}

static int reset(VPxWorker *const worker) {
  int ok = 1;
  worker->had_error = 0;
  if (worker->status_ < VPX_WORKER_STATUS_OK) {
#if CONFIG_MULTITHREAD
    worker->impl_ = (VPxWorkerImpl *)vpx_calloc(1, sizeof(*worker->impl_));
    if (worker->impl_ == NULL) {
      return 0;
    }
    if (pthread_mutex_init(&worker->impl_->mutex_, NULL)) {
      goto Error;
    }
    if (pthread_cond_init(&worker->impl_->condition_, NULL)) {
      pthread_mutex_destroy(&worker->impl_->mutex_);
      goto Error;
    }
    pthread_mutex_lock(&worker->impl_->mutex_);
    ok = !pthread_create(&worker->impl_->thread_, NULL, thread_loop, worker);
    if (ok) worker->status_ = VPX_WORKER_STATUS_OK;
    pthread_mutex_unlock(&worker->impl_->mutex_);
    if (!ok) {
      pthread_mutex_destroy(&worker->impl_->mutex_);
      pthread_cond_destroy(&worker->impl_->condition_);
    Error:
      vpx_free(worker->impl_);
      worker->impl_ = NULL;
      return 0;
    }
#else
    worker->status_ = VPX_WORKER_STATUS_OK;
#endif
  } else if (worker->status_ > VPX_WORKER_STATUS_OK) {
    ok = sync(worker);
  }
  assert(!ok || (worker->status_ == VPX_WORKER_STATUS_OK));
  return ok;
}

static void execute(VPxWorker *const worker) {
  if (worker->hook != NULL) {
    worker->had_error |= !worker->hook(worker->data1, worker->data2);
  }
}

static void launch(VPxWorker *const worker) {
#if CONFIG_MULTITHREAD
  change_state(worker, VPX_WORKER_STATUS_WORKING);
#else
  execute(worker);
#endif
}

static void end(VPxWorker *const worker) {
#if CONFIG_MULTITHREAD
  if (worker->impl_ != NULL) {
    change_state(worker, VPX_WORKER_STATUS_NOT_OK);
    pthread_join(worker->impl_->thread_, NULL);
    pthread_mutex_destroy(&worker->impl_->mutex_);
    pthread_cond_destroy(&worker->impl_->condition_);
    vpx_free(worker->impl_);
    worker->impl_ = NULL;
  }
#else
  worker->status_ = VPX_WORKER_STATUS_NOT_OK;
  assert(worker->impl_ == NULL);
#endif
  assert(worker->status_ == VPX_WORKER_STATUS_NOT_OK);
}

//------------------------------------------------------------------------------

static VPxWorkerInterface g_worker_interface = { init,   reset,   sync,
                                                 launch, execute, end };

int vpx_set_worker_interface(const VPxWorkerInterface *const winterface) {
  if (winterface == NULL || winterface->init == NULL ||
      winterface->reset == NULL || winterface->sync == NULL ||
      winterface->launch == NULL || winterface->execute == NULL ||
      winterface->end == NULL) {
    return 0;
  }
  g_worker_interface = *winterface;
  return 1;
}

const VPxWorkerInterface *vpx_get_worker_interface(void) {
  return &g_worker_interface;
}

//------------------------------------------------------------------------------
