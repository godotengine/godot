/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


/* ======   Dependencies   ======= */
#include <stddef.h>  /* size_t */
#include <stdlib.h>  /* malloc, calloc, free */
#include "pool.h"

/* ======   Compiler specifics   ====== */
#if defined(_MSC_VER)
#  pragma warning(disable : 4204)        /* disable: C4204: non-constant aggregate initializer */
#endif


#ifdef ZSTD_MULTITHREAD

#include "threading.h"   /* pthread adaptation */

/* A job is a function and an opaque argument */
typedef struct POOL_job_s {
  POOL_function function;
  void *opaque;
} POOL_job;

struct POOL_ctx_s {
    /* Keep track of the threads */
    pthread_t *threads;
    size_t numThreads;

    /* The queue is a circular buffer */
    POOL_job *queue;
    size_t queueHead;
    size_t queueTail;
    size_t queueSize;
    /* The mutex protects the queue */
    pthread_mutex_t queueMutex;
    /* Condition variable for pushers to wait on when the queue is full */
    pthread_cond_t queuePushCond;
    /* Condition variables for poppers to wait on when the queue is empty */
    pthread_cond_t queuePopCond;
    /* Indicates if the queue is shutting down */
    int shutdown;
};

/* POOL_thread() :
   Work thread for the thread pool.
   Waits for jobs and executes them.
   @returns : NULL on failure else non-null.
*/
static void* POOL_thread(void* opaque) {
    POOL_ctx* const ctx = (POOL_ctx*)opaque;
    if (!ctx) { return NULL; }
    for (;;) {
        /* Lock the mutex and wait for a non-empty queue or until shutdown */
        pthread_mutex_lock(&ctx->queueMutex);
        while (ctx->queueHead == ctx->queueTail && !ctx->shutdown) {
            pthread_cond_wait(&ctx->queuePopCond, &ctx->queueMutex);
        }
        /* empty => shutting down: so stop */
        if (ctx->queueHead == ctx->queueTail) {
            pthread_mutex_unlock(&ctx->queueMutex);
            return opaque;
        }
        /* Pop a job off the queue */
        {   POOL_job const job = ctx->queue[ctx->queueHead];
            ctx->queueHead = (ctx->queueHead + 1) % ctx->queueSize;
            /* Unlock the mutex, signal a pusher, and run the job */
            pthread_mutex_unlock(&ctx->queueMutex);
            pthread_cond_signal(&ctx->queuePushCond);
            job.function(job.opaque);
        }
    }
    /* Unreachable */
}

POOL_ctx *POOL_create(size_t numThreads, size_t queueSize) {
    POOL_ctx *ctx;
    /* Check the parameters */
    if (!numThreads || !queueSize) { return NULL; }
    /* Allocate the context and zero initialize */
    ctx = (POOL_ctx *)calloc(1, sizeof(POOL_ctx));
    if (!ctx) { return NULL; }
    /* Initialize the job queue.
     * It needs one extra space since one space is wasted to differentiate empty
     * and full queues.
     */
    ctx->queueSize = queueSize + 1;
    ctx->queue = (POOL_job *)malloc(ctx->queueSize * sizeof(POOL_job));
    ctx->queueHead = 0;
    ctx->queueTail = 0;
    pthread_mutex_init(&ctx->queueMutex, NULL);
    pthread_cond_init(&ctx->queuePushCond, NULL);
    pthread_cond_init(&ctx->queuePopCond, NULL);
    ctx->shutdown = 0;
    /* Allocate space for the thread handles */
    ctx->threads = (pthread_t *)malloc(numThreads * sizeof(pthread_t));
    ctx->numThreads = 0;
    /* Check for errors */
    if (!ctx->threads || !ctx->queue) { POOL_free(ctx); return NULL; }
    /* Initialize the threads */
    {   size_t i;
        for (i = 0; i < numThreads; ++i) {
            if (pthread_create(&ctx->threads[i], NULL, &POOL_thread, ctx)) {
                ctx->numThreads = i;
                POOL_free(ctx);
                return NULL;
        }   }
        ctx->numThreads = numThreads;
    }
    return ctx;
}

/*! POOL_join() :
    Shutdown the queue, wake any sleeping threads, and join all of the threads.
*/
static void POOL_join(POOL_ctx *ctx) {
    /* Shut down the queue */
    pthread_mutex_lock(&ctx->queueMutex);
    ctx->shutdown = 1;
    pthread_mutex_unlock(&ctx->queueMutex);
    /* Wake up sleeping threads */
    pthread_cond_broadcast(&ctx->queuePushCond);
    pthread_cond_broadcast(&ctx->queuePopCond);
    /* Join all of the threads */
    {   size_t i;
        for (i = 0; i < ctx->numThreads; ++i) {
            pthread_join(ctx->threads[i], NULL);
    }   }
}

void POOL_free(POOL_ctx *ctx) {
    if (!ctx) { return; }
    POOL_join(ctx);
    pthread_mutex_destroy(&ctx->queueMutex);
    pthread_cond_destroy(&ctx->queuePushCond);
    pthread_cond_destroy(&ctx->queuePopCond);
    if (ctx->queue) free(ctx->queue);
    if (ctx->threads) free(ctx->threads);
    free(ctx);
}

void POOL_add(void *ctxVoid, POOL_function function, void *opaque) {
    POOL_ctx *ctx = (POOL_ctx *)ctxVoid;
    if (!ctx) { return; }

    pthread_mutex_lock(&ctx->queueMutex);
    {   POOL_job const job = {function, opaque};
        /* Wait until there is space in the queue for the new job */
        size_t newTail = (ctx->queueTail + 1) % ctx->queueSize;
        while (ctx->queueHead == newTail && !ctx->shutdown) {
          pthread_cond_wait(&ctx->queuePushCond, &ctx->queueMutex);
          newTail = (ctx->queueTail + 1) % ctx->queueSize;
        }
        /* The queue is still going => there is space */
        if (!ctx->shutdown) {
            ctx->queue[ctx->queueTail] = job;
            ctx->queueTail = newTail;
        }
    }
    pthread_mutex_unlock(&ctx->queueMutex);
    pthread_cond_signal(&ctx->queuePopCond);
}

#else  /* ZSTD_MULTITHREAD  not defined */
/* No multi-threading support */

/* We don't need any data, but if it is empty malloc() might return NULL. */
struct POOL_ctx_s {
  int data;
};

POOL_ctx *POOL_create(size_t numThreads, size_t queueSize) {
  (void)numThreads;
  (void)queueSize;
  return (POOL_ctx *)malloc(sizeof(POOL_ctx));
}

void POOL_free(POOL_ctx *ctx) {
  if (ctx) free(ctx);
}

void POOL_add(void *ctx, POOL_function function, void *opaque) {
  (void)ctx;
  function(opaque);
}

#endif  /* ZSTD_MULTITHREAD */
