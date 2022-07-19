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
//  100644 blob 264210ba2807e4da47eb5d18c04cf869d89b9784  src/utils/thread.c

#include <assert.h>
#include <string.h>   // for memset()
#include "./vpx_thread.h"
#include "vpx_mem/vpx_mem.h"

#if CONFIG_MULTITHREAD

struct VPxWorkerImpl {
  pthread_mutex_t mutex_;
  pthread_cond_t  condition_;
  pthread_t       thread_;
};

//------------------------------------------------------------------------------

static void execute(VPxWorker *const worker);  // Forward declaration.

static THREADFN thread_loop(void *ptr) {
  VPxWorker *const worker = (VPxWorker*)ptr;
  int done = 0;
  while (!done) {
    pthread_mutex_lock(&worker->impl_->mutex_);
    while (worker->status_ == OK) {   // wait in idling mode
      pthread_cond_wait(&worker->impl_->condition_, &worker->impl_->mutex_);
    }
    if (worker->status_ == WORK) {
      execute(worker);
      worker->status_ = OK;
    } else if (worker->status_ == NOT_OK) {   // finish the worker
      done = 1;
    }
    // signal to the main thread that we're done (for sync())
    pthread_cond_signal(&worker->impl_->condition_);
    pthread_mutex_unlock(&worker->impl_->mutex_);
  }
  return THREAD_RETURN(NULL);    // Thread is finished
}

// main thread state control
static void change_state(VPxWorker *const worker,
                         VPxWorkerStatus new_status) {
  // No-op when attempting to change state on a thread that didn't come up.
  // Checking status_ without acquiring the lock first would result in a data
  // race.
  if (worker->impl_ == NULL) return;

  pthread_mutex_lock(&worker->impl_->mutex_);
  if (worker->status_ >= OK) {
    // wait for the worker to finish
    while (worker->status_ != OK) {
      pthread_cond_wait(&worker->impl_->condition_, &worker->impl_->mutex_);
    }
    // assign new status and release the working thread if needed
    if (new_status != OK) {
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
  worker->status_ = NOT_OK;
}

static int sync(VPxWorker *const worker) {
#if CONFIG_MULTITHREAD
  change_state(worker, OK);
#endif
  assert(worker->status_ <= OK);
  return !worker->had_error;
}

static int reset(VPxWorker *const worker) {
  int ok = 1;
  worker->had_error = 0;
  if (worker->status_ < OK) {
#if CONFIG_MULTITHREAD
    worker->impl_ = (VPxWorkerImpl*)vpx_calloc(1, sizeof(*worker->impl_));
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
    if (ok) worker->status_ = OK;
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
    worker->status_ = OK;
#endif
  } else if (worker->status_ > OK) {
    ok = sync(worker);
  }
  assert(!ok || (worker->status_ == OK));
  return ok;
}

static void execute(VPxWorker *const worker) {
  if (worker->hook != NULL) {
    worker->had_error |= !worker->hook(worker->data1, worker->data2);
  }
}

static void launch(VPxWorker *const worker) {
#if CONFIG_MULTITHREAD
  change_state(worker, WORK);
#else
  execute(worker);
#endif
}

static void end(VPxWorker *const worker) {
#if CONFIG_MULTITHREAD
  if (worker->impl_ != NULL) {
    change_state(worker, NOT_OK);
    pthread_join(worker->impl_->thread_, NULL);
    pthread_mutex_destroy(&worker->impl_->mutex_);
    pthread_cond_destroy(&worker->impl_->condition_);
    vpx_free(worker->impl_);
    worker->impl_ = NULL;
  }
#else
  worker->status_ = NOT_OK;
  assert(worker->impl_ == NULL);
#endif
  assert(worker->status_ == NOT_OK);
}

//------------------------------------------------------------------------------

static VPxWorkerInterface g_worker_interface = {
  init, reset, sync, launch, execute, end
};

int vpx_set_worker_interface(const VPxWorkerInterface* const winterface) {
  if (winterface == NULL ||
      winterface->init == NULL || winterface->reset == NULL ||
      winterface->sync == NULL || winterface->launch == NULL ||
      winterface->execute == NULL || winterface->end == NULL) {
    return 0;
  }
  g_worker_interface = *winterface;
  return 1;
}

const VPxWorkerInterface *vpx_get_worker_interface(void) {
  return &g_worker_interface;
}

//------------------------------------------------------------------------------
