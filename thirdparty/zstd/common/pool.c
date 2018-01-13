/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */


/* ======   Dependencies   ======= */
#include <stddef.h>  /* size_t */
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
    ZSTD_customMem customMem;
    /* Keep track of the threads */
    ZSTD_pthread_t *threads;
    size_t numThreads;

    /* The queue is a circular buffer */
    POOL_job *queue;
    size_t queueHead;
    size_t queueTail;
    size_t queueSize;

    /* The number of threads working on jobs */
    size_t numThreadsBusy;
    /* Indicates if the queue is empty */
    int queueEmpty;

    /* The mutex protects the queue */
    ZSTD_pthread_mutex_t queueMutex;
    /* Condition variable for pushers to wait on when the queue is full */
    ZSTD_pthread_cond_t queuePushCond;
    /* Condition variables for poppers to wait on when the queue is empty */
    ZSTD_pthread_cond_t queuePopCond;
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
        ZSTD_pthread_mutex_lock(&ctx->queueMutex);

        while (ctx->queueEmpty && !ctx->shutdown) {
            ZSTD_pthread_cond_wait(&ctx->queuePopCond, &ctx->queueMutex);
        }
        /* empty => shutting down: so stop */
        if (ctx->queueEmpty) {
            ZSTD_pthread_mutex_unlock(&ctx->queueMutex);
            return opaque;
        }
        /* Pop a job off the queue */
        {   POOL_job const job = ctx->queue[ctx->queueHead];
            ctx->queueHead = (ctx->queueHead + 1) % ctx->queueSize;
            ctx->numThreadsBusy++;
            ctx->queueEmpty = ctx->queueHead == ctx->queueTail;
            /* Unlock the mutex, signal a pusher, and run the job */
            ZSTD_pthread_mutex_unlock(&ctx->queueMutex);
            ZSTD_pthread_cond_signal(&ctx->queuePushCond);

            job.function(job.opaque);

            /* If the intended queue size was 0, signal after finishing job */
            if (ctx->queueSize == 1) {
                ZSTD_pthread_mutex_lock(&ctx->queueMutex);
                ctx->numThreadsBusy--;
                ZSTD_pthread_mutex_unlock(&ctx->queueMutex);
                ZSTD_pthread_cond_signal(&ctx->queuePushCond);
        }   }
    }  /* for (;;) */
    /* Unreachable */
}

POOL_ctx* POOL_create(size_t numThreads, size_t queueSize) {
    return POOL_create_advanced(numThreads, queueSize, ZSTD_defaultCMem);
}

POOL_ctx* POOL_create_advanced(size_t numThreads, size_t queueSize, ZSTD_customMem customMem) {
    POOL_ctx* ctx;
    /* Check the parameters */
    if (!numThreads) { return NULL; }
    /* Allocate the context and zero initialize */
    ctx = (POOL_ctx*)ZSTD_calloc(sizeof(POOL_ctx), customMem);
    if (!ctx) { return NULL; }
    /* Initialize the job queue.
     * It needs one extra space since one space is wasted to differentiate empty
     * and full queues.
     */
    ctx->queueSize = queueSize + 1;
    ctx->queue = (POOL_job*)ZSTD_malloc(ctx->queueSize * sizeof(POOL_job), customMem);
    ctx->queueHead = 0;
    ctx->queueTail = 0;
    ctx->numThreadsBusy = 0;
    ctx->queueEmpty = 1;
    (void)ZSTD_pthread_mutex_init(&ctx->queueMutex, NULL);
    (void)ZSTD_pthread_cond_init(&ctx->queuePushCond, NULL);
    (void)ZSTD_pthread_cond_init(&ctx->queuePopCond, NULL);
    ctx->shutdown = 0;
    /* Allocate space for the thread handles */
    ctx->threads = (ZSTD_pthread_t*)ZSTD_malloc(numThreads * sizeof(ZSTD_pthread_t), customMem);
    ctx->numThreads = 0;
    ctx->customMem = customMem;
    /* Check for errors */
    if (!ctx->threads || !ctx->queue) { POOL_free(ctx); return NULL; }
    /* Initialize the threads */
    {   size_t i;
        for (i = 0; i < numThreads; ++i) {
            if (ZSTD_pthread_create(&ctx->threads[i], NULL, &POOL_thread, ctx)) {
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
static void POOL_join(POOL_ctx* ctx) {
    /* Shut down the queue */
    ZSTD_pthread_mutex_lock(&ctx->queueMutex);
    ctx->shutdown = 1;
    ZSTD_pthread_mutex_unlock(&ctx->queueMutex);
    /* Wake up sleeping threads */
    ZSTD_pthread_cond_broadcast(&ctx->queuePushCond);
    ZSTD_pthread_cond_broadcast(&ctx->queuePopCond);
    /* Join all of the threads */
    {   size_t i;
        for (i = 0; i < ctx->numThreads; ++i) {
            ZSTD_pthread_join(ctx->threads[i], NULL);
    }   }
}

void POOL_free(POOL_ctx *ctx) {
    if (!ctx) { return; }
    POOL_join(ctx);
    ZSTD_pthread_mutex_destroy(&ctx->queueMutex);
    ZSTD_pthread_cond_destroy(&ctx->queuePushCond);
    ZSTD_pthread_cond_destroy(&ctx->queuePopCond);
    ZSTD_free(ctx->queue, ctx->customMem);
    ZSTD_free(ctx->threads, ctx->customMem);
    ZSTD_free(ctx, ctx->customMem);
}

size_t POOL_sizeof(POOL_ctx *ctx) {
    if (ctx==NULL) return 0;  /* supports sizeof NULL */
    return sizeof(*ctx)
        + ctx->queueSize * sizeof(POOL_job)
        + ctx->numThreads * sizeof(ZSTD_pthread_t);
}

/**
 * Returns 1 if the queue is full and 0 otherwise.
 *
 * If the queueSize is 1 (the pool was created with an intended queueSize of 0),
 * then a queue is empty if there is a thread free and no job is waiting.
 */
static int isQueueFull(POOL_ctx const* ctx) {
    if (ctx->queueSize > 1) {
        return ctx->queueHead == ((ctx->queueTail + 1) % ctx->queueSize);
    } else {
        return ctx->numThreadsBusy == ctx->numThreads ||
               !ctx->queueEmpty;
    }
}

void POOL_add(void* ctxVoid, POOL_function function, void *opaque) {
    POOL_ctx* const ctx = (POOL_ctx*)ctxVoid;
    if (!ctx) { return; }

    ZSTD_pthread_mutex_lock(&ctx->queueMutex);
    {   POOL_job const job = {function, opaque};

        /* Wait until there is space in the queue for the new job */
        while (isQueueFull(ctx) && !ctx->shutdown) {
          ZSTD_pthread_cond_wait(&ctx->queuePushCond, &ctx->queueMutex);
        }
        /* The queue is still going => there is space */
        if (!ctx->shutdown) {
            ctx->queueEmpty = 0;
            ctx->queue[ctx->queueTail] = job;
            ctx->queueTail = (ctx->queueTail + 1) % ctx->queueSize;
        }
    }
    ZSTD_pthread_mutex_unlock(&ctx->queueMutex);
    ZSTD_pthread_cond_signal(&ctx->queuePopCond);
}

#else  /* ZSTD_MULTITHREAD  not defined */
/* No multi-threading support */

/* We don't need any data, but if it is empty malloc() might return NULL. */
struct POOL_ctx_s {
    int dummy;
};
static POOL_ctx g_ctx;

POOL_ctx* POOL_create(size_t numThreads, size_t queueSize) {
    return POOL_create_advanced(numThreads, queueSize, ZSTD_defaultCMem);
}

POOL_ctx* POOL_create_advanced(size_t numThreads, size_t queueSize, ZSTD_customMem customMem) {
    (void)numThreads;
    (void)queueSize;
    (void)customMem;
    return &g_ctx;
}

void POOL_free(POOL_ctx* ctx) {
    assert(!ctx || ctx == &g_ctx);
    (void)ctx;
}

void POOL_add(void* ctx, POOL_function function, void* opaque) {
    (void)ctx;
    function(opaque);
}

size_t POOL_sizeof(POOL_ctx* ctx) {
    if (ctx==NULL) return 0;  /* supports sizeof NULL */
    assert(ctx == &g_ctx);
    return sizeof(*ctx);
}

#endif  /* ZSTD_MULTITHREAD */
