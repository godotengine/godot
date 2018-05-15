/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */


/* ======   Tuning parameters   ====== */
#define ZSTDMT_NBWORKERS_MAX 200
#define ZSTDMT_JOBSIZE_MAX  (MEM_32bits() ? (512 MB) : (2 GB))  /* note : limited by `jobSize` type, which is `unsigned` */
#define ZSTDMT_OVERLAPLOG_DEFAULT 6


/* ======   Compiler specifics   ====== */
#if defined(_MSC_VER)
#  pragma warning(disable : 4204)   /* disable: C4204: non-constant aggregate initializer */
#endif


/* ======   Dependencies   ====== */
#include <string.h>      /* memcpy, memset */
#include <limits.h>      /* INT_MAX */
#include "pool.h"        /* threadpool */
#include "threading.h"   /* mutex */
#include "zstd_compress_internal.h"  /* MIN, ERROR, ZSTD_*, ZSTD_highbit32 */
#include "zstd_ldm.h"
#include "zstdmt_compress.h"

/* Guards code to support resizing the SeqPool.
 * We will want to resize the SeqPool to save memory in the future.
 * Until then, comment the code out since it is unused.
 */
#define ZSTD_RESIZE_SEQPOOL 0

/* ======   Debug   ====== */
#if defined(ZSTD_DEBUG) && (ZSTD_DEBUG>=2)

#  include <stdio.h>
#  include <unistd.h>
#  include <sys/times.h>
#  define DEBUGLOGRAW(l, ...) if (l<=ZSTD_DEBUG) { fprintf(stderr, __VA_ARGS__); }

#  define DEBUG_PRINTHEX(l,p,n) {            \
    unsigned debug_u;                        \
    for (debug_u=0; debug_u<(n); debug_u++)  \
        DEBUGLOGRAW(l, "%02X ", ((const unsigned char*)(p))[debug_u]); \
    DEBUGLOGRAW(l, " \n");                   \
}

static unsigned long long GetCurrentClockTimeMicroseconds(void)
{
   static clock_t _ticksPerSecond = 0;
   if (_ticksPerSecond <= 0) _ticksPerSecond = sysconf(_SC_CLK_TCK);

   { struct tms junk; clock_t newTicks = (clock_t) times(&junk);
     return ((((unsigned long long)newTicks)*(1000000))/_ticksPerSecond); }
}

#define MUTEX_WAIT_TIME_DLEVEL 6
#define ZSTD_PTHREAD_MUTEX_LOCK(mutex) {          \
    if (ZSTD_DEBUG >= MUTEX_WAIT_TIME_DLEVEL) {   \
        unsigned long long const beforeTime = GetCurrentClockTimeMicroseconds(); \
        ZSTD_pthread_mutex_lock(mutex);           \
        {   unsigned long long const afterTime = GetCurrentClockTimeMicroseconds(); \
            unsigned long long const elapsedTime = (afterTime-beforeTime); \
            if (elapsedTime > 1000) {  /* or whatever threshold you like; I'm using 1 millisecond here */ \
                DEBUGLOG(MUTEX_WAIT_TIME_DLEVEL, "Thread took %llu microseconds to acquire mutex %s \n", \
                   elapsedTime, #mutex);          \
        }   }                                     \
    } else {                                      \
        ZSTD_pthread_mutex_lock(mutex);           \
    }                                             \
}

#else

#  define ZSTD_PTHREAD_MUTEX_LOCK(m) ZSTD_pthread_mutex_lock(m)
#  define DEBUG_PRINTHEX(l,p,n) {}

#endif


/* =====   Buffer Pool   ===== */
/* a single Buffer Pool can be invoked from multiple threads in parallel */

typedef struct buffer_s {
    void* start;
    size_t capacity;
} buffer_t;

static const buffer_t g_nullBuffer = { NULL, 0 };

typedef struct ZSTDMT_bufferPool_s {
    ZSTD_pthread_mutex_t poolMutex;
    size_t bufferSize;
    unsigned totalBuffers;
    unsigned nbBuffers;
    ZSTD_customMem cMem;
    buffer_t bTable[1];   /* variable size */
} ZSTDMT_bufferPool;

static ZSTDMT_bufferPool* ZSTDMT_createBufferPool(unsigned nbWorkers, ZSTD_customMem cMem)
{
    unsigned const maxNbBuffers = 2*nbWorkers + 3;
    ZSTDMT_bufferPool* const bufPool = (ZSTDMT_bufferPool*)ZSTD_calloc(
        sizeof(ZSTDMT_bufferPool) + (maxNbBuffers-1) * sizeof(buffer_t), cMem);
    if (bufPool==NULL) return NULL;
    if (ZSTD_pthread_mutex_init(&bufPool->poolMutex, NULL)) {
        ZSTD_free(bufPool, cMem);
        return NULL;
    }
    bufPool->bufferSize = 64 KB;
    bufPool->totalBuffers = maxNbBuffers;
    bufPool->nbBuffers = 0;
    bufPool->cMem = cMem;
    return bufPool;
}

static void ZSTDMT_freeBufferPool(ZSTDMT_bufferPool* bufPool)
{
    unsigned u;
    DEBUGLOG(3, "ZSTDMT_freeBufferPool (address:%08X)", (U32)(size_t)bufPool);
    if (!bufPool) return;   /* compatibility with free on NULL */
    for (u=0; u<bufPool->totalBuffers; u++) {
        DEBUGLOG(4, "free buffer %2u (address:%08X)", u, (U32)(size_t)bufPool->bTable[u].start);
        ZSTD_free(bufPool->bTable[u].start, bufPool->cMem);
    }
    ZSTD_pthread_mutex_destroy(&bufPool->poolMutex);
    ZSTD_free(bufPool, bufPool->cMem);
}

/* only works at initialization, not during compression */
static size_t ZSTDMT_sizeof_bufferPool(ZSTDMT_bufferPool* bufPool)
{
    size_t const poolSize = sizeof(*bufPool)
                          + (bufPool->totalBuffers - 1) * sizeof(buffer_t);
    unsigned u;
    size_t totalBufferSize = 0;
    ZSTD_pthread_mutex_lock(&bufPool->poolMutex);
    for (u=0; u<bufPool->totalBuffers; u++)
        totalBufferSize += bufPool->bTable[u].capacity;
    ZSTD_pthread_mutex_unlock(&bufPool->poolMutex);

    return poolSize + totalBufferSize;
}

/* ZSTDMT_setBufferSize() :
 * all future buffers provided by this buffer pool will have _at least_ this size
 * note : it's better for all buffers to have same size,
 * as they become freely interchangeable, reducing malloc/free usages and memory fragmentation */
static void ZSTDMT_setBufferSize(ZSTDMT_bufferPool* const bufPool, size_t const bSize)
{
    ZSTD_pthread_mutex_lock(&bufPool->poolMutex);
    DEBUGLOG(4, "ZSTDMT_setBufferSize: bSize = %u", (U32)bSize);
    bufPool->bufferSize = bSize;
    ZSTD_pthread_mutex_unlock(&bufPool->poolMutex);
}

/** ZSTDMT_getBuffer() :
 *  assumption : bufPool must be valid
 * @return : a buffer, with start pointer and size
 *  note: allocation may fail, in this case, start==NULL and size==0 */
static buffer_t ZSTDMT_getBuffer(ZSTDMT_bufferPool* bufPool)
{
    size_t const bSize = bufPool->bufferSize;
    DEBUGLOG(5, "ZSTDMT_getBuffer: bSize = %u", (U32)bufPool->bufferSize);
    ZSTD_pthread_mutex_lock(&bufPool->poolMutex);
    if (bufPool->nbBuffers) {   /* try to use an existing buffer */
        buffer_t const buf = bufPool->bTable[--(bufPool->nbBuffers)];
        size_t const availBufferSize = buf.capacity;
        bufPool->bTable[bufPool->nbBuffers] = g_nullBuffer;
        if ((availBufferSize >= bSize) & ((availBufferSize>>3) <= bSize)) {
            /* large enough, but not too much */
            DEBUGLOG(5, "ZSTDMT_getBuffer: provide buffer %u of size %u",
                        bufPool->nbBuffers, (U32)buf.capacity);
            ZSTD_pthread_mutex_unlock(&bufPool->poolMutex);
            return buf;
        }
        /* size conditions not respected : scratch this buffer, create new one */
        DEBUGLOG(5, "ZSTDMT_getBuffer: existing buffer does not meet size conditions => freeing");
        ZSTD_free(buf.start, bufPool->cMem);
    }
    ZSTD_pthread_mutex_unlock(&bufPool->poolMutex);
    /* create new buffer */
    DEBUGLOG(5, "ZSTDMT_getBuffer: create a new buffer");
    {   buffer_t buffer;
        void* const start = ZSTD_malloc(bSize, bufPool->cMem);
        buffer.start = start;   /* note : start can be NULL if malloc fails ! */
        buffer.capacity = (start==NULL) ? 0 : bSize;
        if (start==NULL) {
            DEBUGLOG(5, "ZSTDMT_getBuffer: buffer allocation failure !!");
        } else {
            DEBUGLOG(5, "ZSTDMT_getBuffer: created buffer of size %u", (U32)bSize);
        }
        return buffer;
    }
}

#if ZSTD_RESIZE_SEQPOOL
/** ZSTDMT_resizeBuffer() :
 * assumption : bufPool must be valid
 * @return : a buffer that is at least the buffer pool buffer size.
 *           If a reallocation happens, the data in the input buffer is copied.
 */
static buffer_t ZSTDMT_resizeBuffer(ZSTDMT_bufferPool* bufPool, buffer_t buffer)
{
    size_t const bSize = bufPool->bufferSize;
    if (buffer.capacity < bSize) {
        void* const start = ZSTD_malloc(bSize, bufPool->cMem);
        buffer_t newBuffer;
        newBuffer.start = start;
        newBuffer.capacity = start == NULL ? 0 : bSize;
        if (start != NULL) {
            assert(newBuffer.capacity >= buffer.capacity);
            memcpy(newBuffer.start, buffer.start, buffer.capacity);
            DEBUGLOG(5, "ZSTDMT_resizeBuffer: created buffer of size %u", (U32)bSize);
            return newBuffer;
        }
        DEBUGLOG(5, "ZSTDMT_resizeBuffer: buffer allocation failure !!");
    }
    return buffer;
}
#endif

/* store buffer for later re-use, up to pool capacity */
static void ZSTDMT_releaseBuffer(ZSTDMT_bufferPool* bufPool, buffer_t buf)
{
    if (buf.start == NULL) return;   /* compatible with release on NULL */
    DEBUGLOG(5, "ZSTDMT_releaseBuffer");
    ZSTD_pthread_mutex_lock(&bufPool->poolMutex);
    if (bufPool->nbBuffers < bufPool->totalBuffers) {
        bufPool->bTable[bufPool->nbBuffers++] = buf;  /* stored for later use */
        DEBUGLOG(5, "ZSTDMT_releaseBuffer: stored buffer of size %u in slot %u",
                    (U32)buf.capacity, (U32)(bufPool->nbBuffers-1));
        ZSTD_pthread_mutex_unlock(&bufPool->poolMutex);
        return;
    }
    ZSTD_pthread_mutex_unlock(&bufPool->poolMutex);
    /* Reached bufferPool capacity (should not happen) */
    DEBUGLOG(5, "ZSTDMT_releaseBuffer: pool capacity reached => freeing ");
    ZSTD_free(buf.start, bufPool->cMem);
}


/* =====   Seq Pool Wrapper   ====== */

static rawSeqStore_t kNullRawSeqStore = {NULL, 0, 0, 0};

typedef ZSTDMT_bufferPool ZSTDMT_seqPool;

static size_t ZSTDMT_sizeof_seqPool(ZSTDMT_seqPool* seqPool)
{
    return ZSTDMT_sizeof_bufferPool(seqPool);
}

static rawSeqStore_t bufferToSeq(buffer_t buffer)
{
    rawSeqStore_t seq = {NULL, 0, 0, 0};
    seq.seq = (rawSeq*)buffer.start;
    seq.capacity = buffer.capacity / sizeof(rawSeq);
    return seq;
}

static buffer_t seqToBuffer(rawSeqStore_t seq)
{
    buffer_t buffer;
    buffer.start = seq.seq;
    buffer.capacity = seq.capacity * sizeof(rawSeq);
    return buffer;
}

static rawSeqStore_t ZSTDMT_getSeq(ZSTDMT_seqPool* seqPool)
{
    if (seqPool->bufferSize == 0) {
        return kNullRawSeqStore;
    }
    return bufferToSeq(ZSTDMT_getBuffer(seqPool));
}

#if ZSTD_RESIZE_SEQPOOL
static rawSeqStore_t ZSTDMT_resizeSeq(ZSTDMT_seqPool* seqPool, rawSeqStore_t seq)
{
  return bufferToSeq(ZSTDMT_resizeBuffer(seqPool, seqToBuffer(seq)));
}
#endif

static void ZSTDMT_releaseSeq(ZSTDMT_seqPool* seqPool, rawSeqStore_t seq)
{
  ZSTDMT_releaseBuffer(seqPool, seqToBuffer(seq));
}

static void ZSTDMT_setNbSeq(ZSTDMT_seqPool* const seqPool, size_t const nbSeq)
{
  ZSTDMT_setBufferSize(seqPool, nbSeq * sizeof(rawSeq));
}

static ZSTDMT_seqPool* ZSTDMT_createSeqPool(unsigned nbWorkers, ZSTD_customMem cMem)
{
    ZSTDMT_seqPool* seqPool = ZSTDMT_createBufferPool(nbWorkers, cMem);
    ZSTDMT_setNbSeq(seqPool, 0);
    return seqPool;
}

static void ZSTDMT_freeSeqPool(ZSTDMT_seqPool* seqPool)
{
    ZSTDMT_freeBufferPool(seqPool);
}



/* =====   CCtx Pool   ===== */
/* a single CCtx Pool can be invoked from multiple threads in parallel */

typedef struct {
    ZSTD_pthread_mutex_t poolMutex;
    unsigned totalCCtx;
    unsigned availCCtx;
    ZSTD_customMem cMem;
    ZSTD_CCtx* cctx[1];   /* variable size */
} ZSTDMT_CCtxPool;

/* note : all CCtx borrowed from the pool should be released back to the pool _before_ freeing the pool */
static void ZSTDMT_freeCCtxPool(ZSTDMT_CCtxPool* pool)
{
    unsigned u;
    for (u=0; u<pool->totalCCtx; u++)
        ZSTD_freeCCtx(pool->cctx[u]);  /* note : compatible with free on NULL */
    ZSTD_pthread_mutex_destroy(&pool->poolMutex);
    ZSTD_free(pool, pool->cMem);
}

/* ZSTDMT_createCCtxPool() :
 * implies nbWorkers >= 1 , checked by caller ZSTDMT_createCCtx() */
static ZSTDMT_CCtxPool* ZSTDMT_createCCtxPool(unsigned nbWorkers,
                                              ZSTD_customMem cMem)
{
    ZSTDMT_CCtxPool* const cctxPool = (ZSTDMT_CCtxPool*) ZSTD_calloc(
        sizeof(ZSTDMT_CCtxPool) + (nbWorkers-1)*sizeof(ZSTD_CCtx*), cMem);
    assert(nbWorkers > 0);
    if (!cctxPool) return NULL;
    if (ZSTD_pthread_mutex_init(&cctxPool->poolMutex, NULL)) {
        ZSTD_free(cctxPool, cMem);
        return NULL;
    }
    cctxPool->cMem = cMem;
    cctxPool->totalCCtx = nbWorkers;
    cctxPool->availCCtx = 1;   /* at least one cctx for single-thread mode */
    cctxPool->cctx[0] = ZSTD_createCCtx_advanced(cMem);
    if (!cctxPool->cctx[0]) { ZSTDMT_freeCCtxPool(cctxPool); return NULL; }
    DEBUGLOG(3, "cctxPool created, with %u workers", nbWorkers);
    return cctxPool;
}

/* only works during initialization phase, not during compression */
static size_t ZSTDMT_sizeof_CCtxPool(ZSTDMT_CCtxPool* cctxPool)
{
    ZSTD_pthread_mutex_lock(&cctxPool->poolMutex);
    {   unsigned const nbWorkers = cctxPool->totalCCtx;
        size_t const poolSize = sizeof(*cctxPool)
                                + (nbWorkers-1) * sizeof(ZSTD_CCtx*);
        unsigned u;
        size_t totalCCtxSize = 0;
        for (u=0; u<nbWorkers; u++) {
            totalCCtxSize += ZSTD_sizeof_CCtx(cctxPool->cctx[u]);
        }
        ZSTD_pthread_mutex_unlock(&cctxPool->poolMutex);
        assert(nbWorkers > 0);
        return poolSize + totalCCtxSize;
    }
}

static ZSTD_CCtx* ZSTDMT_getCCtx(ZSTDMT_CCtxPool* cctxPool)
{
    DEBUGLOG(5, "ZSTDMT_getCCtx");
    ZSTD_pthread_mutex_lock(&cctxPool->poolMutex);
    if (cctxPool->availCCtx) {
        cctxPool->availCCtx--;
        {   ZSTD_CCtx* const cctx = cctxPool->cctx[cctxPool->availCCtx];
            ZSTD_pthread_mutex_unlock(&cctxPool->poolMutex);
            return cctx;
    }   }
    ZSTD_pthread_mutex_unlock(&cctxPool->poolMutex);
    DEBUGLOG(5, "create one more CCtx");
    return ZSTD_createCCtx_advanced(cctxPool->cMem);   /* note : can be NULL, when creation fails ! */
}

static void ZSTDMT_releaseCCtx(ZSTDMT_CCtxPool* pool, ZSTD_CCtx* cctx)
{
    if (cctx==NULL) return;   /* compatibility with release on NULL */
    ZSTD_pthread_mutex_lock(&pool->poolMutex);
    if (pool->availCCtx < pool->totalCCtx)
        pool->cctx[pool->availCCtx++] = cctx;
    else {
        /* pool overflow : should not happen, since totalCCtx==nbWorkers */
        DEBUGLOG(4, "CCtx pool overflow : free cctx");
        ZSTD_freeCCtx(cctx);
    }
    ZSTD_pthread_mutex_unlock(&pool->poolMutex);
}

/* ====   Serial State   ==== */

typedef struct {
    void const* start;
    size_t size;
} range_t;

typedef struct {
    /* All variables in the struct are protected by mutex. */
    ZSTD_pthread_mutex_t mutex;
    ZSTD_pthread_cond_t cond;
    ZSTD_CCtx_params params;
    ldmState_t ldmState;
    XXH64_state_t xxhState;
    unsigned nextJobID;
    /* Protects ldmWindow.
     * Must be acquired after the main mutex when acquiring both.
     */
    ZSTD_pthread_mutex_t ldmWindowMutex;
    ZSTD_pthread_cond_t ldmWindowCond;  /* Signaled when ldmWindow is udpated */
    ZSTD_window_t ldmWindow;  /* A thread-safe copy of ldmState.window */
} serialState_t;

static int ZSTDMT_serialState_reset(serialState_t* serialState, ZSTDMT_seqPool* seqPool, ZSTD_CCtx_params params)
{
    /* Adjust parameters */
    if (params.ldmParams.enableLdm) {
        DEBUGLOG(4, "LDM window size = %u KB", (1U << params.cParams.windowLog) >> 10);
        params.ldmParams.windowLog = params.cParams.windowLog;
        ZSTD_ldm_adjustParameters(&params.ldmParams, &params.cParams);
        assert(params.ldmParams.hashLog >= params.ldmParams.bucketSizeLog);
        assert(params.ldmParams.hashEveryLog < 32);
        serialState->ldmState.hashPower =
                ZSTD_ldm_getHashPower(params.ldmParams.minMatchLength);
    } else {
        memset(&params.ldmParams, 0, sizeof(params.ldmParams));
    }
    serialState->nextJobID = 0;
    if (params.fParams.checksumFlag)
        XXH64_reset(&serialState->xxhState, 0);
    if (params.ldmParams.enableLdm) {
        ZSTD_customMem cMem = params.customMem;
        unsigned const hashLog = params.ldmParams.hashLog;
        size_t const hashSize = ((size_t)1 << hashLog) * sizeof(ldmEntry_t);
        unsigned const bucketLog =
            params.ldmParams.hashLog - params.ldmParams.bucketSizeLog;
        size_t const bucketSize = (size_t)1 << bucketLog;
        unsigned const prevBucketLog =
            serialState->params.ldmParams.hashLog -
            serialState->params.ldmParams.bucketSizeLog;
        /* Size the seq pool tables */
        ZSTDMT_setNbSeq(seqPool, ZSTD_ldm_getMaxNbSeq(params.ldmParams, params.jobSize));
        /* Reset the window */
        ZSTD_window_clear(&serialState->ldmState.window);
        serialState->ldmWindow = serialState->ldmState.window;
        /* Resize tables and output space if necessary. */
        if (serialState->ldmState.hashTable == NULL || serialState->params.ldmParams.hashLog < hashLog) {
            ZSTD_free(serialState->ldmState.hashTable, cMem);
            serialState->ldmState.hashTable = (ldmEntry_t*)ZSTD_malloc(hashSize, cMem);
        }
        if (serialState->ldmState.bucketOffsets == NULL || prevBucketLog < bucketLog) {
            ZSTD_free(serialState->ldmState.bucketOffsets, cMem);
            serialState->ldmState.bucketOffsets = (BYTE*)ZSTD_malloc(bucketSize, cMem);
        }
        if (!serialState->ldmState.hashTable || !serialState->ldmState.bucketOffsets)
            return 1;
        /* Zero the tables */
        memset(serialState->ldmState.hashTable, 0, hashSize);
        memset(serialState->ldmState.bucketOffsets, 0, bucketSize);
    }
    serialState->params = params;
    return 0;
}

static int ZSTDMT_serialState_init(serialState_t* serialState)
{
    int initError = 0;
    memset(serialState, 0, sizeof(*serialState));
    initError |= ZSTD_pthread_mutex_init(&serialState->mutex, NULL);
    initError |= ZSTD_pthread_cond_init(&serialState->cond, NULL);
    initError |= ZSTD_pthread_mutex_init(&serialState->ldmWindowMutex, NULL);
    initError |= ZSTD_pthread_cond_init(&serialState->ldmWindowCond, NULL);
    return initError;
}

static void ZSTDMT_serialState_free(serialState_t* serialState)
{
    ZSTD_customMem cMem = serialState->params.customMem;
    ZSTD_pthread_mutex_destroy(&serialState->mutex);
    ZSTD_pthread_cond_destroy(&serialState->cond);
    ZSTD_pthread_mutex_destroy(&serialState->ldmWindowMutex);
    ZSTD_pthread_cond_destroy(&serialState->ldmWindowCond);
    ZSTD_free(serialState->ldmState.hashTable, cMem);
    ZSTD_free(serialState->ldmState.bucketOffsets, cMem);
}

static void ZSTDMT_serialState_update(serialState_t* serialState,
                                      ZSTD_CCtx* jobCCtx, rawSeqStore_t seqStore,
                                      range_t src, unsigned jobID)
{
    /* Wait for our turn */
    ZSTD_PTHREAD_MUTEX_LOCK(&serialState->mutex);
    while (serialState->nextJobID < jobID) {
        ZSTD_pthread_cond_wait(&serialState->cond, &serialState->mutex);
    }
    /* A future job may error and skip our job */
    if (serialState->nextJobID == jobID) {
        /* It is now our turn, do any processing necessary */
        if (serialState->params.ldmParams.enableLdm) {
            size_t error;
            assert(seqStore.seq != NULL && seqStore.pos == 0 &&
                   seqStore.size == 0 && seqStore.capacity > 0);
            ZSTD_window_update(&serialState->ldmState.window, src.start, src.size);
            error = ZSTD_ldm_generateSequences(
                &serialState->ldmState, &seqStore,
                &serialState->params.ldmParams, src.start, src.size);
            /* We provide a large enough buffer to never fail. */
            assert(!ZSTD_isError(error)); (void)error;
            /* Update ldmWindow to match the ldmState.window and signal the main
             * thread if it is waiting for a buffer.
             */
            ZSTD_PTHREAD_MUTEX_LOCK(&serialState->ldmWindowMutex);
            serialState->ldmWindow = serialState->ldmState.window;
            ZSTD_pthread_cond_signal(&serialState->ldmWindowCond);
            ZSTD_pthread_mutex_unlock(&serialState->ldmWindowMutex);
        }
        if (serialState->params.fParams.checksumFlag && src.size > 0)
            XXH64_update(&serialState->xxhState, src.start, src.size);
    }
    /* Now it is the next jobs turn */
    serialState->nextJobID++;
    ZSTD_pthread_cond_broadcast(&serialState->cond);
    ZSTD_pthread_mutex_unlock(&serialState->mutex);

    if (seqStore.size > 0) {
        size_t const err = ZSTD_referenceExternalSequences(
            jobCCtx, seqStore.seq, seqStore.size);
        assert(serialState->params.ldmParams.enableLdm);
        assert(!ZSTD_isError(err));
        (void)err;
    }
}

static void ZSTDMT_serialState_ensureFinished(serialState_t* serialState,
                                              unsigned jobID, size_t cSize)
{
    ZSTD_PTHREAD_MUTEX_LOCK(&serialState->mutex);
    if (serialState->nextJobID <= jobID) {
        assert(ZSTD_isError(cSize)); (void)cSize;
        DEBUGLOG(5, "Skipping past job %u because of error", jobID);
        serialState->nextJobID = jobID + 1;
        ZSTD_pthread_cond_broadcast(&serialState->cond);

        ZSTD_PTHREAD_MUTEX_LOCK(&serialState->ldmWindowMutex);
        ZSTD_window_clear(&serialState->ldmWindow);
        ZSTD_pthread_cond_signal(&serialState->ldmWindowCond);
        ZSTD_pthread_mutex_unlock(&serialState->ldmWindowMutex);
    }
    ZSTD_pthread_mutex_unlock(&serialState->mutex);

}


/* ------------------------------------------ */
/* =====          Worker thread         ===== */
/* ------------------------------------------ */

static const range_t kNullRange = { NULL, 0 };

typedef struct {
    size_t   consumed;                   /* SHARED - set0 by mtctx, then modified by worker AND read by mtctx */
    size_t   cSize;                      /* SHARED - set0 by mtctx, then modified by worker AND read by mtctx, then set0 by mtctx */
    ZSTD_pthread_mutex_t job_mutex;      /* Thread-safe - used by mtctx and worker */
    ZSTD_pthread_cond_t job_cond;        /* Thread-safe - used by mtctx and worker */
    ZSTDMT_CCtxPool* cctxPool;           /* Thread-safe - used by mtctx and (all) workers */
    ZSTDMT_bufferPool* bufPool;          /* Thread-safe - used by mtctx and (all) workers */
    ZSTDMT_seqPool* seqPool;             /* Thread-safe - used by mtctx and (all) workers */
    serialState_t* serial;               /* Thread-safe - used by mtctx and (all) workers */
    buffer_t dstBuff;                    /* set by worker (or mtctx), then read by worker & mtctx, then modified by mtctx => no barrier */
    range_t prefix;                      /* set by mtctx, then read by worker & mtctx => no barrier */
    range_t src;                         /* set by mtctx, then read by worker & mtctx => no barrier */
    unsigned jobID;                      /* set by mtctx, then read by worker => no barrier */
    unsigned firstJob;                   /* set by mtctx, then read by worker => no barrier */
    unsigned lastJob;                    /* set by mtctx, then read by worker => no barrier */
    ZSTD_CCtx_params params;             /* set by mtctx, then read by worker => no barrier */
    const ZSTD_CDict* cdict;             /* set by mtctx, then read by worker => no barrier */
    unsigned long long fullFrameSize;    /* set by mtctx, then read by worker => no barrier */
    size_t   dstFlushed;                 /* used only by mtctx */
    unsigned frameChecksumNeeded;        /* used only by mtctx */
} ZSTDMT_jobDescription;

/* ZSTDMT_compressionJob() is a POOL_function type */
void ZSTDMT_compressionJob(void* jobDescription)
{
    ZSTDMT_jobDescription* const job = (ZSTDMT_jobDescription*)jobDescription;
    ZSTD_CCtx_params jobParams = job->params;   /* do not modify job->params ! copy it, modify the copy */
    ZSTD_CCtx* const cctx = ZSTDMT_getCCtx(job->cctxPool);
    rawSeqStore_t rawSeqStore = ZSTDMT_getSeq(job->seqPool);
    buffer_t dstBuff = job->dstBuff;

    /* Don't compute the checksum for chunks, since we compute it externally,
     * but write it in the header.
     */
    if (job->jobID != 0) jobParams.fParams.checksumFlag = 0;
    /* Don't run LDM for the chunks, since we handle it externally */
    jobParams.ldmParams.enableLdm = 0;

    /* ressources */
    if (cctx==NULL) {
        job->cSize = ERROR(memory_allocation);
        goto _endJob;
    }
    if (dstBuff.start == NULL) {   /* streaming job : doesn't provide a dstBuffer */
        dstBuff = ZSTDMT_getBuffer(job->bufPool);
        if (dstBuff.start==NULL) {
            job->cSize = ERROR(memory_allocation);
            goto _endJob;
        }
        job->dstBuff = dstBuff;   /* this value can be read in ZSTDMT_flush, when it copies the whole job */
    }

    /* init */
    if (job->cdict) {
        size_t const initError = ZSTD_compressBegin_advanced_internal(cctx, NULL, 0, ZSTD_dct_auto, job->cdict, jobParams, job->fullFrameSize);
        assert(job->firstJob);  /* only allowed for first job */
        if (ZSTD_isError(initError)) { job->cSize = initError; goto _endJob; }
    } else {  /* srcStart points at reloaded section */
        U64 const pledgedSrcSize = job->firstJob ? job->fullFrameSize : job->src.size;
        {   size_t const forceWindowError = ZSTD_CCtxParam_setParameter(&jobParams, ZSTD_p_forceMaxWindow, !job->firstJob);
            if (ZSTD_isError(forceWindowError)) {
                job->cSize = forceWindowError;
                goto _endJob;
        }   }
        {   size_t const initError = ZSTD_compressBegin_advanced_internal(cctx,
                                        job->prefix.start, job->prefix.size, ZSTD_dct_rawContent, /* load dictionary in "content-only" mode (no header analysis) */
                                        NULL, /*cdict*/
                                        jobParams, pledgedSrcSize);
            if (ZSTD_isError(initError)) {
                job->cSize = initError;
                goto _endJob;
    }   }   }

    /* Perform serial step as early as possible, but after CCtx initialization */
    ZSTDMT_serialState_update(job->serial, cctx, rawSeqStore, job->src, job->jobID);

    if (!job->firstJob) {  /* flush and overwrite frame header when it's not first job */
        size_t const hSize = ZSTD_compressContinue(cctx, dstBuff.start, dstBuff.capacity, job->src.start, 0);
        if (ZSTD_isError(hSize)) { job->cSize = hSize; /* save error code */ goto _endJob; }
        DEBUGLOG(5, "ZSTDMT_compressionJob: flush and overwrite %u bytes of frame header (not first job)", (U32)hSize);
        ZSTD_invalidateRepCodes(cctx);
    }

    /* compress */
    {   size_t const chunkSize = 4*ZSTD_BLOCKSIZE_MAX;
        int const nbChunks = (int)((job->src.size + (chunkSize-1)) / chunkSize);
        const BYTE* ip = (const BYTE*) job->src.start;
        BYTE* const ostart = (BYTE*)dstBuff.start;
        BYTE* op = ostart;
        BYTE* oend = op + dstBuff.capacity;
        int chunkNb;
        if (sizeof(size_t) > sizeof(int)) assert(job->src.size < ((size_t)INT_MAX) * chunkSize);   /* check overflow */
        DEBUGLOG(5, "ZSTDMT_compressionJob: compress %u bytes in %i blocks", (U32)job->src.size, nbChunks);
        assert(job->cSize == 0);
        for (chunkNb = 1; chunkNb < nbChunks; chunkNb++) {
            size_t const cSize = ZSTD_compressContinue(cctx, op, oend-op, ip, chunkSize);
            if (ZSTD_isError(cSize)) { job->cSize = cSize; goto _endJob; }
            ip += chunkSize;
            op += cSize; assert(op < oend);
            /* stats */
            ZSTD_PTHREAD_MUTEX_LOCK(&job->job_mutex);
            job->cSize += cSize;
            job->consumed = chunkSize * chunkNb;
            DEBUGLOG(5, "ZSTDMT_compressionJob: compress new block : cSize==%u bytes (total: %u)",
                        (U32)cSize, (U32)job->cSize);
            ZSTD_pthread_cond_signal(&job->job_cond);   /* warns some more data is ready to be flushed */
            ZSTD_pthread_mutex_unlock(&job->job_mutex);
        }
        /* last block */
        assert(chunkSize > 0); assert((chunkSize & (chunkSize - 1)) == 0);  /* chunkSize must be power of 2 for mask==(chunkSize-1) to work */
        if ((nbChunks > 0) | job->lastJob /*must output a "last block" flag*/ ) {
            size_t const lastBlockSize1 = job->src.size & (chunkSize-1);
            size_t const lastBlockSize = ((lastBlockSize1==0) & (job->src.size>=chunkSize)) ? chunkSize : lastBlockSize1;
            size_t const cSize = (job->lastJob) ?
                 ZSTD_compressEnd     (cctx, op, oend-op, ip, lastBlockSize) :
                 ZSTD_compressContinue(cctx, op, oend-op, ip, lastBlockSize);
            if (ZSTD_isError(cSize)) { job->cSize = cSize; goto _endJob; }
            /* stats */
            ZSTD_PTHREAD_MUTEX_LOCK(&job->job_mutex);
            job->cSize += cSize;
            ZSTD_pthread_mutex_unlock(&job->job_mutex);
    }   }

_endJob:
    ZSTDMT_serialState_ensureFinished(job->serial, job->jobID, job->cSize);
    if (job->prefix.size > 0)
        DEBUGLOG(5, "Finished with prefix: %zx", (size_t)job->prefix.start);
    DEBUGLOG(5, "Finished with source: %zx", (size_t)job->src.start);
    /* release resources */
    ZSTDMT_releaseSeq(job->seqPool, rawSeqStore);
    ZSTDMT_releaseCCtx(job->cctxPool, cctx);
    /* report */
    ZSTD_PTHREAD_MUTEX_LOCK(&job->job_mutex);
    job->consumed = job->src.size;
    ZSTD_pthread_cond_signal(&job->job_cond);
    ZSTD_pthread_mutex_unlock(&job->job_mutex);
}


/* ------------------------------------------ */
/* =====   Multi-threaded compression   ===== */
/* ------------------------------------------ */

typedef struct {
    range_t prefix;         /* read-only non-owned prefix buffer */
    buffer_t buffer;
    size_t filled;
} inBuff_t;

typedef struct {
  BYTE* buffer;     /* The round input buffer. All jobs get references
                     * to pieces of the buffer. ZSTDMT_tryGetInputRange()
                     * handles handing out job input buffers, and makes
                     * sure it doesn't overlap with any pieces still in use.
                     */
  size_t capacity;  /* The capacity of buffer. */
  size_t pos;       /* The position of the current inBuff in the round
                     * buffer. Updated past the end if the inBuff once
                     * the inBuff is sent to the worker thread.
                     * pos <= capacity.
                     */
} roundBuff_t;

static const roundBuff_t kNullRoundBuff = {NULL, 0, 0};

struct ZSTDMT_CCtx_s {
    POOL_ctx* factory;
    ZSTDMT_jobDescription* jobs;
    ZSTDMT_bufferPool* bufPool;
    ZSTDMT_CCtxPool* cctxPool;
    ZSTDMT_seqPool* seqPool;
    ZSTD_CCtx_params params;
    size_t targetSectionSize;
    size_t targetPrefixSize;
    roundBuff_t roundBuff;
    inBuff_t inBuff;
    int jobReady;        /* 1 => one job is already prepared, but pool has shortage of workers. Don't create another one. */
    serialState_t serial;
    unsigned singleBlockingThread;
    unsigned jobIDMask;
    unsigned doneJobID;
    unsigned nextJobID;
    unsigned frameEnded;
    unsigned allJobsCompleted;
    unsigned long long frameContentSize;
    unsigned long long consumed;
    unsigned long long produced;
    ZSTD_customMem cMem;
    ZSTD_CDict* cdictLocal;
    const ZSTD_CDict* cdict;
};

static void ZSTDMT_freeJobsTable(ZSTDMT_jobDescription* jobTable, U32 nbJobs, ZSTD_customMem cMem)
{
    U32 jobNb;
    if (jobTable == NULL) return;
    for (jobNb=0; jobNb<nbJobs; jobNb++) {
        ZSTD_pthread_mutex_destroy(&jobTable[jobNb].job_mutex);
        ZSTD_pthread_cond_destroy(&jobTable[jobNb].job_cond);
    }
    ZSTD_free(jobTable, cMem);
}

/* ZSTDMT_allocJobsTable()
 * allocate and init a job table.
 * update *nbJobsPtr to next power of 2 value, as size of table */
static ZSTDMT_jobDescription* ZSTDMT_createJobsTable(U32* nbJobsPtr, ZSTD_customMem cMem)
{
    U32 const nbJobsLog2 = ZSTD_highbit32(*nbJobsPtr) + 1;
    U32 const nbJobs = 1 << nbJobsLog2;
    U32 jobNb;
    ZSTDMT_jobDescription* const jobTable = (ZSTDMT_jobDescription*)
                ZSTD_calloc(nbJobs * sizeof(ZSTDMT_jobDescription), cMem);
    int initError = 0;
    if (jobTable==NULL) return NULL;
    *nbJobsPtr = nbJobs;
    for (jobNb=0; jobNb<nbJobs; jobNb++) {
        initError |= ZSTD_pthread_mutex_init(&jobTable[jobNb].job_mutex, NULL);
        initError |= ZSTD_pthread_cond_init(&jobTable[jobNb].job_cond, NULL);
    }
    if (initError != 0) {
        ZSTDMT_freeJobsTable(jobTable, nbJobs, cMem);
        return NULL;
    }
    return jobTable;
}

/* ZSTDMT_CCtxParam_setNbWorkers():
 * Internal use only */
size_t ZSTDMT_CCtxParam_setNbWorkers(ZSTD_CCtx_params* params, unsigned nbWorkers)
{
    if (nbWorkers > ZSTDMT_NBWORKERS_MAX) nbWorkers = ZSTDMT_NBWORKERS_MAX;
    params->nbWorkers = nbWorkers;
    params->overlapSizeLog = ZSTDMT_OVERLAPLOG_DEFAULT;
    params->jobSize = 0;
    return nbWorkers;
}

ZSTDMT_CCtx* ZSTDMT_createCCtx_advanced(unsigned nbWorkers, ZSTD_customMem cMem)
{
    ZSTDMT_CCtx* mtctx;
    U32 nbJobs = nbWorkers + 2;
    int initError;
    DEBUGLOG(3, "ZSTDMT_createCCtx_advanced (nbWorkers = %u)", nbWorkers);

    if (nbWorkers < 1) return NULL;
    nbWorkers = MIN(nbWorkers , ZSTDMT_NBWORKERS_MAX);
    if ((cMem.customAlloc!=NULL) ^ (cMem.customFree!=NULL))
        /* invalid custom allocator */
        return NULL;

    mtctx = (ZSTDMT_CCtx*) ZSTD_calloc(sizeof(ZSTDMT_CCtx), cMem);
    if (!mtctx) return NULL;
    ZSTDMT_CCtxParam_setNbWorkers(&mtctx->params, nbWorkers);
    mtctx->cMem = cMem;
    mtctx->allJobsCompleted = 1;
    mtctx->factory = POOL_create_advanced(nbWorkers, 0, cMem);
    mtctx->jobs = ZSTDMT_createJobsTable(&nbJobs, cMem);
    assert(nbJobs > 0); assert((nbJobs & (nbJobs - 1)) == 0);  /* ensure nbJobs is a power of 2 */
    mtctx->jobIDMask = nbJobs - 1;
    mtctx->bufPool = ZSTDMT_createBufferPool(nbWorkers, cMem);
    mtctx->cctxPool = ZSTDMT_createCCtxPool(nbWorkers, cMem);
    mtctx->seqPool = ZSTDMT_createSeqPool(nbWorkers, cMem);
    initError = ZSTDMT_serialState_init(&mtctx->serial);
    mtctx->roundBuff = kNullRoundBuff;
    if (!mtctx->factory | !mtctx->jobs | !mtctx->bufPool | !mtctx->cctxPool | !mtctx->seqPool | initError) {
        ZSTDMT_freeCCtx(mtctx);
        return NULL;
    }
    DEBUGLOG(3, "mt_cctx created, for %u threads", nbWorkers);
    return mtctx;
}

ZSTDMT_CCtx* ZSTDMT_createCCtx(unsigned nbWorkers)
{
    return ZSTDMT_createCCtx_advanced(nbWorkers, ZSTD_defaultCMem);
}


/* ZSTDMT_releaseAllJobResources() :
 * note : ensure all workers are killed first ! */
static void ZSTDMT_releaseAllJobResources(ZSTDMT_CCtx* mtctx)
{
    unsigned jobID;
    DEBUGLOG(3, "ZSTDMT_releaseAllJobResources");
    for (jobID=0; jobID <= mtctx->jobIDMask; jobID++) {
        DEBUGLOG(4, "job%02u: release dst address %08X", jobID, (U32)(size_t)mtctx->jobs[jobID].dstBuff.start);
        ZSTDMT_releaseBuffer(mtctx->bufPool, mtctx->jobs[jobID].dstBuff);
        mtctx->jobs[jobID].dstBuff = g_nullBuffer;
        mtctx->jobs[jobID].cSize = 0;
    }
    memset(mtctx->jobs, 0, (mtctx->jobIDMask+1)*sizeof(ZSTDMT_jobDescription));
    mtctx->inBuff.buffer = g_nullBuffer;
    mtctx->inBuff.filled = 0;
    mtctx->allJobsCompleted = 1;
}

static void ZSTDMT_waitForAllJobsCompleted(ZSTDMT_CCtx* mtctx)
{
    DEBUGLOG(4, "ZSTDMT_waitForAllJobsCompleted");
    while (mtctx->doneJobID < mtctx->nextJobID) {
        unsigned const jobID = mtctx->doneJobID & mtctx->jobIDMask;
        ZSTD_PTHREAD_MUTEX_LOCK(&mtctx->jobs[jobID].job_mutex);
        while (mtctx->jobs[jobID].consumed < mtctx->jobs[jobID].src.size) {
            DEBUGLOG(5, "waiting for jobCompleted signal from job %u", mtctx->doneJobID);   /* we want to block when waiting for data to flush */
            ZSTD_pthread_cond_wait(&mtctx->jobs[jobID].job_cond, &mtctx->jobs[jobID].job_mutex);
        }
        ZSTD_pthread_mutex_unlock(&mtctx->jobs[jobID].job_mutex);
        mtctx->doneJobID++;
    }
}

size_t ZSTDMT_freeCCtx(ZSTDMT_CCtx* mtctx)
{
    if (mtctx==NULL) return 0;   /* compatible with free on NULL */
    POOL_free(mtctx->factory);   /* stop and free worker threads */
    ZSTDMT_releaseAllJobResources(mtctx);  /* release job resources into pools first */
    ZSTDMT_freeJobsTable(mtctx->jobs, mtctx->jobIDMask+1, mtctx->cMem);
    ZSTDMT_freeBufferPool(mtctx->bufPool);
    ZSTDMT_freeCCtxPool(mtctx->cctxPool);
    ZSTDMT_freeSeqPool(mtctx->seqPool);
    ZSTDMT_serialState_free(&mtctx->serial);
    ZSTD_freeCDict(mtctx->cdictLocal);
    if (mtctx->roundBuff.buffer)
        ZSTD_free(mtctx->roundBuff.buffer, mtctx->cMem);
    ZSTD_free(mtctx, mtctx->cMem);
    return 0;
}

size_t ZSTDMT_sizeof_CCtx(ZSTDMT_CCtx* mtctx)
{
    if (mtctx == NULL) return 0;   /* supports sizeof NULL */
    return sizeof(*mtctx)
            + POOL_sizeof(mtctx->factory)
            + ZSTDMT_sizeof_bufferPool(mtctx->bufPool)
            + (mtctx->jobIDMask+1) * sizeof(ZSTDMT_jobDescription)
            + ZSTDMT_sizeof_CCtxPool(mtctx->cctxPool)
            + ZSTDMT_sizeof_seqPool(mtctx->seqPool)
            + ZSTD_sizeof_CDict(mtctx->cdictLocal)
            + mtctx->roundBuff.capacity;
}

/* Internal only */
size_t ZSTDMT_CCtxParam_setMTCtxParameter(ZSTD_CCtx_params* params,
                                ZSTDMT_parameter parameter, unsigned value) {
    DEBUGLOG(4, "ZSTDMT_CCtxParam_setMTCtxParameter");
    switch(parameter)
    {
    case ZSTDMT_p_jobSize :
        DEBUGLOG(4, "ZSTDMT_CCtxParam_setMTCtxParameter : set jobSize to %u", value);
        if ( (value > 0)  /* value==0 => automatic job size */
           & (value < ZSTDMT_JOBSIZE_MIN) )
            value = ZSTDMT_JOBSIZE_MIN;
        params->jobSize = value;
        return value;
    case ZSTDMT_p_overlapSectionLog :
        if (value > 9) value = 9;
        DEBUGLOG(4, "ZSTDMT_p_overlapSectionLog : %u", value);
        params->overlapSizeLog = (value >= 9) ? 9 : value;
        return value;
    default :
        return ERROR(parameter_unsupported);
    }
}

size_t ZSTDMT_setMTCtxParameter(ZSTDMT_CCtx* mtctx, ZSTDMT_parameter parameter, unsigned value)
{
    DEBUGLOG(4, "ZSTDMT_setMTCtxParameter");
    switch(parameter)
    {
    case ZSTDMT_p_jobSize :
        return ZSTDMT_CCtxParam_setMTCtxParameter(&mtctx->params, parameter, value);
    case ZSTDMT_p_overlapSectionLog :
        return ZSTDMT_CCtxParam_setMTCtxParameter(&mtctx->params, parameter, value);
    default :
        return ERROR(parameter_unsupported);
    }
}

/* Sets parameters relevant to the compression job,
 * initializing others to default values. */
static ZSTD_CCtx_params ZSTDMT_initJobCCtxParams(ZSTD_CCtx_params const params)
{
    ZSTD_CCtx_params jobParams;
    memset(&jobParams, 0, sizeof(jobParams));

    jobParams.cParams = params.cParams;
    jobParams.fParams = params.fParams;
    jobParams.compressionLevel = params.compressionLevel;
    jobParams.disableLiteralCompression = params.disableLiteralCompression;

    return jobParams;
}

/*! ZSTDMT_updateCParams_whileCompressing() :
 *  Updates only a selected set of compression parameters, to remain compatible with current frame.
 *  New parameters will be applied to next compression job. */
void ZSTDMT_updateCParams_whileCompressing(ZSTDMT_CCtx* mtctx, const ZSTD_CCtx_params* cctxParams)
{
    U32 const saved_wlog = mtctx->params.cParams.windowLog;   /* Do not modify windowLog while compressing */
    int const compressionLevel = cctxParams->compressionLevel;
    DEBUGLOG(5, "ZSTDMT_updateCParams_whileCompressing (level:%i)",
                compressionLevel);
    mtctx->params.compressionLevel = compressionLevel;
    {   ZSTD_compressionParameters cParams = ZSTD_getCParamsFromCCtxParams(cctxParams, 0, 0);
        cParams.windowLog = saved_wlog;
        mtctx->params.cParams = cParams;
    }
}

/* ZSTDMT_getNbWorkers():
 * @return nb threads currently active in mtctx.
 * mtctx must be valid */
unsigned ZSTDMT_getNbWorkers(const ZSTDMT_CCtx* mtctx)
{
    assert(mtctx != NULL);
    return mtctx->params.nbWorkers;
}

/* ZSTDMT_getFrameProgression():
 * tells how much data has been consumed (input) and produced (output) for current frame.
 * able to count progression inside worker threads.
 * Note : mutex will be acquired during statistics collection. */
ZSTD_frameProgression ZSTDMT_getFrameProgression(ZSTDMT_CCtx* mtctx)
{
    ZSTD_frameProgression fps;
    DEBUGLOG(6, "ZSTDMT_getFrameProgression");
    fps.consumed = mtctx->consumed;
    fps.produced = mtctx->produced;
    fps.ingested = mtctx->consumed + mtctx->inBuff.filled;
    {   unsigned jobNb;
        unsigned lastJobNb = mtctx->nextJobID + mtctx->jobReady; assert(mtctx->jobReady <= 1);
        DEBUGLOG(6, "ZSTDMT_getFrameProgression: jobs: from %u to <%u (jobReady:%u)",
                    mtctx->doneJobID, lastJobNb, mtctx->jobReady)
        for (jobNb = mtctx->doneJobID ; jobNb < lastJobNb ; jobNb++) {
            unsigned const wJobID = jobNb & mtctx->jobIDMask;
            ZSTD_pthread_mutex_lock(&mtctx->jobs[wJobID].job_mutex);
            {   size_t const cResult = mtctx->jobs[wJobID].cSize;
                size_t const produced = ZSTD_isError(cResult) ? 0 : cResult;
                fps.consumed += mtctx->jobs[wJobID].consumed;
                fps.ingested += mtctx->jobs[wJobID].src.size;
                fps.produced += produced;
            }
            ZSTD_pthread_mutex_unlock(&mtctx->jobs[wJobID].job_mutex);
        }
    }
    return fps;
}


/* ------------------------------------------ */
/* =====   Multi-threaded compression   ===== */
/* ------------------------------------------ */

static size_t ZSTDMT_computeTargetJobLog(ZSTD_CCtx_params const params)
{
    if (params.ldmParams.enableLdm)
        return MAX(21, params.cParams.chainLog + 4);
    return MAX(20, params.cParams.windowLog + 2);
}

static size_t ZSTDMT_computeOverlapLog(ZSTD_CCtx_params const params)
{
    unsigned const overlapRLog = (params.overlapSizeLog>9) ? 0 : 9-params.overlapSizeLog;
    if (params.ldmParams.enableLdm)
        return (MIN(params.cParams.windowLog, ZSTDMT_computeTargetJobLog(params) - 2) - overlapRLog);
    return overlapRLog >= 9 ? 0 : (params.cParams.windowLog - overlapRLog);
}

static unsigned ZSTDMT_computeNbJobs(ZSTD_CCtx_params params, size_t srcSize, unsigned nbWorkers) {
    assert(nbWorkers>0);
    {   size_t const jobSizeTarget = (size_t)1 << ZSTDMT_computeTargetJobLog(params);
        size_t const jobMaxSize = jobSizeTarget << 2;
        size_t const passSizeMax = jobMaxSize * nbWorkers;
        unsigned const multiplier = (unsigned)(srcSize / passSizeMax) + 1;
        unsigned const nbJobsLarge = multiplier * nbWorkers;
        unsigned const nbJobsMax = (unsigned)(srcSize / jobSizeTarget) + 1;
        unsigned const nbJobsSmall = MIN(nbJobsMax, nbWorkers);
        return (multiplier>1) ? nbJobsLarge : nbJobsSmall;
}   }

/* ZSTDMT_compress_advanced_internal() :
 * This is a blocking function : it will only give back control to caller after finishing its compression job.
 */
static size_t ZSTDMT_compress_advanced_internal(
                ZSTDMT_CCtx* mtctx,
                void* dst, size_t dstCapacity,
          const void* src, size_t srcSize,
          const ZSTD_CDict* cdict,
                ZSTD_CCtx_params params)
{
    ZSTD_CCtx_params const jobParams = ZSTDMT_initJobCCtxParams(params);
    size_t const overlapSize = (size_t)1 << ZSTDMT_computeOverlapLog(params);
    unsigned const nbJobs = ZSTDMT_computeNbJobs(params, srcSize, params.nbWorkers);
    size_t const proposedJobSize = (srcSize + (nbJobs-1)) / nbJobs;
    size_t const avgJobSize = (((proposedJobSize-1) & 0x1FFFF) < 0x7FFF) ? proposedJobSize + 0xFFFF : proposedJobSize;   /* avoid too small last block */
    const char* const srcStart = (const char*)src;
    size_t remainingSrcSize = srcSize;
    unsigned const compressWithinDst = (dstCapacity >= ZSTD_compressBound(srcSize)) ? nbJobs : (unsigned)(dstCapacity / ZSTD_compressBound(avgJobSize));  /* presumes avgJobSize >= 256 KB, which should be the case */
    size_t frameStartPos = 0, dstBufferPos = 0;
    assert(jobParams.nbWorkers == 0);
    assert(mtctx->cctxPool->totalCCtx == params.nbWorkers);

    params.jobSize = (U32)avgJobSize;
    DEBUGLOG(4, "ZSTDMT_compress_advanced_internal: nbJobs=%2u (rawSize=%u bytes; fixedSize=%u) ",
                nbJobs, (U32)proposedJobSize, (U32)avgJobSize);

    if ((nbJobs==1) | (params.nbWorkers<=1)) {   /* fallback to single-thread mode : this is a blocking invocation anyway */
        ZSTD_CCtx* const cctx = mtctx->cctxPool->cctx[0];
        DEBUGLOG(4, "ZSTDMT_compress_advanced_internal: fallback to single-thread mode");
        if (cdict) return ZSTD_compress_usingCDict_advanced(cctx, dst, dstCapacity, src, srcSize, cdict, jobParams.fParams);
        return ZSTD_compress_advanced_internal(cctx, dst, dstCapacity, src, srcSize, NULL, 0, jobParams);
    }

    assert(avgJobSize >= 256 KB);  /* condition for ZSTD_compressBound(A) + ZSTD_compressBound(B) <= ZSTD_compressBound(A+B), required to compress directly into Dst (no additional buffer) */
    ZSTDMT_setBufferSize(mtctx->bufPool, ZSTD_compressBound(avgJobSize) );
    if (ZSTDMT_serialState_reset(&mtctx->serial, mtctx->seqPool, params))
        return ERROR(memory_allocation);

    if (nbJobs > mtctx->jobIDMask+1) {  /* enlarge job table */
        U32 jobsTableSize = nbJobs;
        ZSTDMT_freeJobsTable(mtctx->jobs, mtctx->jobIDMask+1, mtctx->cMem);
        mtctx->jobIDMask = 0;
        mtctx->jobs = ZSTDMT_createJobsTable(&jobsTableSize, mtctx->cMem);
        if (mtctx->jobs==NULL) return ERROR(memory_allocation);
        assert((jobsTableSize != 0) && ((jobsTableSize & (jobsTableSize - 1)) == 0));  /* ensure jobsTableSize is a power of 2 */
        mtctx->jobIDMask = jobsTableSize - 1;
    }

    {   unsigned u;
        for (u=0; u<nbJobs; u++) {
            size_t const jobSize = MIN(remainingSrcSize, avgJobSize);
            size_t const dstBufferCapacity = ZSTD_compressBound(jobSize);
            buffer_t const dstAsBuffer = { (char*)dst + dstBufferPos, dstBufferCapacity };
            buffer_t const dstBuffer = u < compressWithinDst ? dstAsBuffer : g_nullBuffer;
            size_t dictSize = u ? overlapSize : 0;

            mtctx->jobs[u].prefix.start = srcStart + frameStartPos - dictSize;
            mtctx->jobs[u].prefix.size = dictSize;
            mtctx->jobs[u].src.start = srcStart + frameStartPos;
            mtctx->jobs[u].src.size = jobSize; assert(jobSize > 0);  /* avoid job.src.size == 0 */
            mtctx->jobs[u].consumed = 0;
            mtctx->jobs[u].cSize = 0;
            mtctx->jobs[u].cdict = (u==0) ? cdict : NULL;
            mtctx->jobs[u].fullFrameSize = srcSize;
            mtctx->jobs[u].params = jobParams;
            /* do not calculate checksum within sections, but write it in header for first section */
            mtctx->jobs[u].dstBuff = dstBuffer;
            mtctx->jobs[u].cctxPool = mtctx->cctxPool;
            mtctx->jobs[u].bufPool = mtctx->bufPool;
            mtctx->jobs[u].seqPool = mtctx->seqPool;
            mtctx->jobs[u].serial = &mtctx->serial;
            mtctx->jobs[u].jobID = u;
            mtctx->jobs[u].firstJob = (u==0);
            mtctx->jobs[u].lastJob = (u==nbJobs-1);

            DEBUGLOG(5, "ZSTDMT_compress_advanced_internal: posting job %u  (%u bytes)", u, (U32)jobSize);
            DEBUG_PRINTHEX(6, mtctx->jobs[u].prefix.start, 12);
            POOL_add(mtctx->factory, ZSTDMT_compressionJob, &mtctx->jobs[u]);

            frameStartPos += jobSize;
            dstBufferPos += dstBufferCapacity;
            remainingSrcSize -= jobSize;
    }   }

    /* collect result */
    {   size_t error = 0, dstPos = 0;
        unsigned jobID;
        for (jobID=0; jobID<nbJobs; jobID++) {
            DEBUGLOG(5, "waiting for job %u ", jobID);
            ZSTD_PTHREAD_MUTEX_LOCK(&mtctx->jobs[jobID].job_mutex);
            while (mtctx->jobs[jobID].consumed < mtctx->jobs[jobID].src.size) {
                DEBUGLOG(5, "waiting for jobCompleted signal from job %u", jobID);
                ZSTD_pthread_cond_wait(&mtctx->jobs[jobID].job_cond, &mtctx->jobs[jobID].job_mutex);
            }
            ZSTD_pthread_mutex_unlock(&mtctx->jobs[jobID].job_mutex);
            DEBUGLOG(5, "ready to write job %u ", jobID);

            {   size_t const cSize = mtctx->jobs[jobID].cSize;
                if (ZSTD_isError(cSize)) error = cSize;
                if ((!error) && (dstPos + cSize > dstCapacity)) error = ERROR(dstSize_tooSmall);
                if (jobID) {   /* note : job 0 is written directly at dst, which is correct position */
                    if (!error)
                        memmove((char*)dst + dstPos, mtctx->jobs[jobID].dstBuff.start, cSize);  /* may overlap when job compressed within dst */
                    if (jobID >= compressWithinDst) {  /* job compressed into its own buffer, which must be released */
                        DEBUGLOG(5, "releasing buffer %u>=%u", jobID, compressWithinDst);
                        ZSTDMT_releaseBuffer(mtctx->bufPool, mtctx->jobs[jobID].dstBuff);
                }   }
                mtctx->jobs[jobID].dstBuff = g_nullBuffer;
                mtctx->jobs[jobID].cSize = 0;
                dstPos += cSize ;
            }
        }  /* for (jobID=0; jobID<nbJobs; jobID++) */

        DEBUGLOG(4, "checksumFlag : %u ", params.fParams.checksumFlag);
        if (params.fParams.checksumFlag) {
            U32 const checksum = (U32)XXH64_digest(&mtctx->serial.xxhState);
            if (dstPos + 4 > dstCapacity) {
                error = ERROR(dstSize_tooSmall);
            } else {
                DEBUGLOG(4, "writing checksum : %08X \n", checksum);
                MEM_writeLE32((char*)dst + dstPos, checksum);
                dstPos += 4;
        }   }

        if (!error) DEBUGLOG(4, "compressed size : %u  ", (U32)dstPos);
        return error ? error : dstPos;
    }
}

size_t ZSTDMT_compress_advanced(ZSTDMT_CCtx* mtctx,
                               void* dst, size_t dstCapacity,
                         const void* src, size_t srcSize,
                         const ZSTD_CDict* cdict,
                               ZSTD_parameters params,
                               unsigned overlapLog)
{
    ZSTD_CCtx_params cctxParams = mtctx->params;
    cctxParams.cParams = params.cParams;
    cctxParams.fParams = params.fParams;
    cctxParams.overlapSizeLog = overlapLog;
    return ZSTDMT_compress_advanced_internal(mtctx,
                                             dst, dstCapacity,
                                             src, srcSize,
                                             cdict, cctxParams);
}


size_t ZSTDMT_compressCCtx(ZSTDMT_CCtx* mtctx,
                           void* dst, size_t dstCapacity,
                     const void* src, size_t srcSize,
                           int compressionLevel)
{
    U32 const overlapLog = (compressionLevel >= ZSTD_maxCLevel()) ? 9 : ZSTDMT_OVERLAPLOG_DEFAULT;
    ZSTD_parameters params = ZSTD_getParams(compressionLevel, srcSize, 0);
    params.fParams.contentSizeFlag = 1;
    return ZSTDMT_compress_advanced(mtctx, dst, dstCapacity, src, srcSize, NULL, params, overlapLog);
}


/* ====================================== */
/* =======      Streaming API     ======= */
/* ====================================== */

size_t ZSTDMT_initCStream_internal(
        ZSTDMT_CCtx* mtctx,
        const void* dict, size_t dictSize, ZSTD_dictContentType_e dictContentType,
        const ZSTD_CDict* cdict, ZSTD_CCtx_params params,
        unsigned long long pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTDMT_initCStream_internal (pledgedSrcSize=%u, nbWorkers=%u, cctxPool=%u, disableLiteralCompression=%i)",
                (U32)pledgedSrcSize, params.nbWorkers, mtctx->cctxPool->totalCCtx, params.disableLiteralCompression);
    /* params are supposed to be fully validated at this point */
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));
    assert(!((dict) && (cdict)));  /* either dict or cdict, not both */
    assert(mtctx->cctxPool->totalCCtx == params.nbWorkers);

    /* init */
    if (params.jobSize == 0) {
        params.jobSize = 1U << ZSTDMT_computeTargetJobLog(params);
    }
    if (params.jobSize > ZSTDMT_JOBSIZE_MAX) params.jobSize = ZSTDMT_JOBSIZE_MAX;

    mtctx->singleBlockingThread = (pledgedSrcSize <= ZSTDMT_JOBSIZE_MIN);  /* do not trigger multi-threading when srcSize is too small */
    if (mtctx->singleBlockingThread) {
        ZSTD_CCtx_params const singleThreadParams = ZSTDMT_initJobCCtxParams(params);
        DEBUGLOG(5, "ZSTDMT_initCStream_internal: switch to single blocking thread mode");
        assert(singleThreadParams.nbWorkers == 0);
        return ZSTD_initCStream_internal(mtctx->cctxPool->cctx[0],
                                         dict, dictSize, cdict,
                                         singleThreadParams, pledgedSrcSize);
    }

    DEBUGLOG(4, "ZSTDMT_initCStream_internal: %u workers", params.nbWorkers);

    if (mtctx->allJobsCompleted == 0) {   /* previous compression not correctly finished */
        ZSTDMT_waitForAllJobsCompleted(mtctx);
        ZSTDMT_releaseAllJobResources(mtctx);
        mtctx->allJobsCompleted = 1;
    }

    mtctx->params = params;
    mtctx->frameContentSize = pledgedSrcSize;
    if (dict) {
        ZSTD_freeCDict(mtctx->cdictLocal);
        mtctx->cdictLocal = ZSTD_createCDict_advanced(dict, dictSize,
                                                    ZSTD_dlm_byCopy, dictContentType, /* note : a loadPrefix becomes an internal CDict */
                                                    params.cParams, mtctx->cMem);
        mtctx->cdict = mtctx->cdictLocal;
        if (mtctx->cdictLocal == NULL) return ERROR(memory_allocation);
    } else {
        ZSTD_freeCDict(mtctx->cdictLocal);
        mtctx->cdictLocal = NULL;
        mtctx->cdict = cdict;
    }

    mtctx->targetPrefixSize = (size_t)1 << ZSTDMT_computeOverlapLog(params);
    DEBUGLOG(4, "overlapLog=%u => %u KB", params.overlapSizeLog, (U32)(mtctx->targetPrefixSize>>10));
    mtctx->targetSectionSize = params.jobSize;
    if (mtctx->targetSectionSize < ZSTDMT_JOBSIZE_MIN) mtctx->targetSectionSize = ZSTDMT_JOBSIZE_MIN;
    if (mtctx->targetSectionSize < mtctx->targetPrefixSize) mtctx->targetSectionSize = mtctx->targetPrefixSize;  /* job size must be >= overlap size */
    DEBUGLOG(4, "Job Size : %u KB (note : set to %u)", (U32)(mtctx->targetSectionSize>>10), params.jobSize);
    DEBUGLOG(4, "inBuff Size : %u KB", (U32)(mtctx->targetSectionSize>>10));
    ZSTDMT_setBufferSize(mtctx->bufPool, ZSTD_compressBound(mtctx->targetSectionSize));
    {
        /* If ldm is enabled we need windowSize space. */
        size_t const windowSize = mtctx->params.ldmParams.enableLdm ? (1U << mtctx->params.cParams.windowLog) : 0;
        /* Two buffers of slack, plus extra space for the overlap
         * This is the minimum slack that LDM works with. One extra because
         * flush might waste up to targetSectionSize-1 bytes. Another extra
         * for the overlap (if > 0), then one to fill which doesn't overlap
         * with the LDM window.
         */
        size_t const nbSlackBuffers = 2 + (mtctx->targetPrefixSize > 0);
        size_t const slackSize = mtctx->targetSectionSize * nbSlackBuffers;
        /* Compute the total size, and always have enough slack */
        size_t const nbWorkers = MAX(mtctx->params.nbWorkers, 1);
        size_t const sectionsSize = mtctx->targetSectionSize * nbWorkers;
        size_t const capacity = MAX(windowSize, sectionsSize) + slackSize;
        if (mtctx->roundBuff.capacity < capacity) {
            if (mtctx->roundBuff.buffer)
                ZSTD_free(mtctx->roundBuff.buffer, mtctx->cMem);
            mtctx->roundBuff.buffer = (BYTE*)ZSTD_malloc(capacity, mtctx->cMem);
            if (mtctx->roundBuff.buffer == NULL) {
                mtctx->roundBuff.capacity = 0;
                return ERROR(memory_allocation);
            }
            mtctx->roundBuff.capacity = capacity;
        }
    }
    DEBUGLOG(4, "roundBuff capacity : %u KB", (U32)(mtctx->roundBuff.capacity>>10));
    mtctx->roundBuff.pos = 0;
    mtctx->inBuff.buffer = g_nullBuffer;
    mtctx->inBuff.filled = 0;
    mtctx->inBuff.prefix = kNullRange;
    mtctx->doneJobID = 0;
    mtctx->nextJobID = 0;
    mtctx->frameEnded = 0;
    mtctx->allJobsCompleted = 0;
    mtctx->consumed = 0;
    mtctx->produced = 0;
    if (ZSTDMT_serialState_reset(&mtctx->serial, mtctx->seqPool, params))
        return ERROR(memory_allocation);
    return 0;
}

size_t ZSTDMT_initCStream_advanced(ZSTDMT_CCtx* mtctx,
                             const void* dict, size_t dictSize,
                                   ZSTD_parameters params,
                                   unsigned long long pledgedSrcSize)
{
    ZSTD_CCtx_params cctxParams = mtctx->params;  /* retrieve sticky params */
    DEBUGLOG(4, "ZSTDMT_initCStream_advanced (pledgedSrcSize=%u)", (U32)pledgedSrcSize);
    cctxParams.cParams = params.cParams;
    cctxParams.fParams = params.fParams;
    return ZSTDMT_initCStream_internal(mtctx, dict, dictSize, ZSTD_dct_auto, NULL,
                                       cctxParams, pledgedSrcSize);
}

size_t ZSTDMT_initCStream_usingCDict(ZSTDMT_CCtx* mtctx,
                               const ZSTD_CDict* cdict,
                                     ZSTD_frameParameters fParams,
                                     unsigned long long pledgedSrcSize)
{
    ZSTD_CCtx_params cctxParams = mtctx->params;
    if (cdict==NULL) return ERROR(dictionary_wrong);   /* method incompatible with NULL cdict */
    cctxParams.cParams = ZSTD_getCParamsFromCDict(cdict);
    cctxParams.fParams = fParams;
    return ZSTDMT_initCStream_internal(mtctx, NULL, 0 /*dictSize*/, ZSTD_dct_auto, cdict,
                                       cctxParams, pledgedSrcSize);
}


/* ZSTDMT_resetCStream() :
 * pledgedSrcSize can be zero == unknown (for the time being)
 * prefer using ZSTD_CONTENTSIZE_UNKNOWN,
 * as `0` might mean "empty" in the future */
size_t ZSTDMT_resetCStream(ZSTDMT_CCtx* mtctx, unsigned long long pledgedSrcSize)
{
    if (!pledgedSrcSize) pledgedSrcSize = ZSTD_CONTENTSIZE_UNKNOWN;
    return ZSTDMT_initCStream_internal(mtctx, NULL, 0, ZSTD_dct_auto, 0, mtctx->params,
                                       pledgedSrcSize);
}

size_t ZSTDMT_initCStream(ZSTDMT_CCtx* mtctx, int compressionLevel) {
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, ZSTD_CONTENTSIZE_UNKNOWN, 0);
    ZSTD_CCtx_params cctxParams = mtctx->params;   /* retrieve sticky params */
    DEBUGLOG(4, "ZSTDMT_initCStream (cLevel=%i)", compressionLevel);
    cctxParams.cParams = params.cParams;
    cctxParams.fParams = params.fParams;
    return ZSTDMT_initCStream_internal(mtctx, NULL, 0, ZSTD_dct_auto, NULL, cctxParams, ZSTD_CONTENTSIZE_UNKNOWN);
}


/* ZSTDMT_writeLastEmptyBlock()
 * Write a single empty block with an end-of-frame to finish a frame.
 * Job must be created from streaming variant.
 * This function is always successfull if expected conditions are fulfilled.
 */
static void ZSTDMT_writeLastEmptyBlock(ZSTDMT_jobDescription* job)
{
    assert(job->lastJob == 1);
    assert(job->src.size == 0);   /* last job is empty -> will be simplified into a last empty block */
    assert(job->firstJob == 0);   /* cannot be first job, as it also needs to create frame header */
    assert(job->dstBuff.start == NULL);   /* invoked from streaming variant only (otherwise, dstBuff might be user's output) */
    job->dstBuff = ZSTDMT_getBuffer(job->bufPool);
    if (job->dstBuff.start == NULL) {
      job->cSize = ERROR(memory_allocation);
      return;
    }
    assert(job->dstBuff.capacity >= ZSTD_blockHeaderSize);   /* no buffer should ever be that small */
    job->src = kNullRange;
    job->cSize = ZSTD_writeLastEmptyBlock(job->dstBuff.start, job->dstBuff.capacity);
    assert(!ZSTD_isError(job->cSize));
    assert(job->consumed == 0);
}

static size_t ZSTDMT_createCompressionJob(ZSTDMT_CCtx* mtctx, size_t srcSize, ZSTD_EndDirective endOp)
{
    unsigned const jobID = mtctx->nextJobID & mtctx->jobIDMask;
    int const endFrame = (endOp == ZSTD_e_end);

    if (mtctx->nextJobID > mtctx->doneJobID + mtctx->jobIDMask) {
        DEBUGLOG(5, "ZSTDMT_createCompressionJob: will not create new job : table is full");
        assert((mtctx->nextJobID & mtctx->jobIDMask) == (mtctx->doneJobID & mtctx->jobIDMask));
        return 0;
    }

    if (!mtctx->jobReady) {
        BYTE const* src = (BYTE const*)mtctx->inBuff.buffer.start;
        DEBUGLOG(5, "ZSTDMT_createCompressionJob: preparing job %u to compress %u bytes with %u preload ",
                    mtctx->nextJobID, (U32)srcSize, (U32)mtctx->inBuff.prefix.size);
        mtctx->jobs[jobID].src.start = src;
        mtctx->jobs[jobID].src.size = srcSize;
        assert(mtctx->inBuff.filled >= srcSize);
        mtctx->jobs[jobID].prefix = mtctx->inBuff.prefix;
        mtctx->jobs[jobID].consumed = 0;
        mtctx->jobs[jobID].cSize = 0;
        mtctx->jobs[jobID].params = mtctx->params;
        mtctx->jobs[jobID].cdict = mtctx->nextJobID==0 ? mtctx->cdict : NULL;
        mtctx->jobs[jobID].fullFrameSize = mtctx->frameContentSize;
        mtctx->jobs[jobID].dstBuff = g_nullBuffer;
        mtctx->jobs[jobID].cctxPool = mtctx->cctxPool;
        mtctx->jobs[jobID].bufPool = mtctx->bufPool;
        mtctx->jobs[jobID].seqPool = mtctx->seqPool;
        mtctx->jobs[jobID].serial = &mtctx->serial;
        mtctx->jobs[jobID].jobID = mtctx->nextJobID;
        mtctx->jobs[jobID].firstJob = (mtctx->nextJobID==0);
        mtctx->jobs[jobID].lastJob = endFrame;
        mtctx->jobs[jobID].frameChecksumNeeded = endFrame && (mtctx->nextJobID>0) && mtctx->params.fParams.checksumFlag;
        mtctx->jobs[jobID].dstFlushed = 0;

        /* Update the round buffer pos and clear the input buffer to be reset */
        mtctx->roundBuff.pos += srcSize;
        mtctx->inBuff.buffer = g_nullBuffer;
        mtctx->inBuff.filled = 0;
        /* Set the prefix */
        if (!endFrame) {
            size_t const newPrefixSize = MIN(srcSize, mtctx->targetPrefixSize);
            mtctx->inBuff.prefix.start = src + srcSize - newPrefixSize;
            mtctx->inBuff.prefix.size = newPrefixSize;
        } else {   /* endFrame==1 => no need for another input buffer */
            mtctx->inBuff.prefix = kNullRange;
            mtctx->frameEnded = endFrame;
            if (mtctx->nextJobID == 0) {
                /* single job exception : checksum is already calculated directly within worker thread */
                mtctx->params.fParams.checksumFlag = 0;
        }   }

        if ( (srcSize == 0)
          && (mtctx->nextJobID>0)/*single job must also write frame header*/ ) {
            DEBUGLOG(5, "ZSTDMT_createCompressionJob: creating a last empty block to end frame");
            assert(endOp == ZSTD_e_end);  /* only possible case : need to end the frame with an empty last block */
            ZSTDMT_writeLastEmptyBlock(mtctx->jobs + jobID);
            mtctx->nextJobID++;
            return 0;
        }
    }

    DEBUGLOG(5, "ZSTDMT_createCompressionJob: posting job %u : %u bytes  (end:%u, jobNb == %u (mod:%u))",
                mtctx->nextJobID,
                (U32)mtctx->jobs[jobID].src.size,
                mtctx->jobs[jobID].lastJob,
                mtctx->nextJobID,
                jobID);
    if (POOL_tryAdd(mtctx->factory, ZSTDMT_compressionJob, &mtctx->jobs[jobID])) {
        mtctx->nextJobID++;
        mtctx->jobReady = 0;
    } else {
        DEBUGLOG(5, "ZSTDMT_createCompressionJob: no worker available for job %u", mtctx->nextJobID);
        mtctx->jobReady = 1;
    }
    return 0;
}


/*! ZSTDMT_flushProduced() :
 * `output` : `pos` will be updated with amount of data flushed .
 * `blockToFlush` : if >0, the function will block and wait if there is no data available to flush .
 * @return : amount of data remaining within internal buffer, 0 if no more, 1 if unknown but > 0, or an error code */
static size_t ZSTDMT_flushProduced(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output, unsigned blockToFlush, ZSTD_EndDirective end)
{
    unsigned const wJobID = mtctx->doneJobID & mtctx->jobIDMask;
    DEBUGLOG(5, "ZSTDMT_flushProduced (blocking:%u , job %u <= %u)",
                blockToFlush, mtctx->doneJobID, mtctx->nextJobID);
    assert(output->size >= output->pos);

    ZSTD_PTHREAD_MUTEX_LOCK(&mtctx->jobs[wJobID].job_mutex);
    if (  blockToFlush
      && (mtctx->doneJobID < mtctx->nextJobID) ) {
        assert(mtctx->jobs[wJobID].dstFlushed <= mtctx->jobs[wJobID].cSize);
        while (mtctx->jobs[wJobID].dstFlushed == mtctx->jobs[wJobID].cSize) {  /* nothing to flush */
            if (mtctx->jobs[wJobID].consumed == mtctx->jobs[wJobID].src.size) {
                DEBUGLOG(5, "job %u is completely consumed (%u == %u) => don't wait for cond, there will be none",
                            mtctx->doneJobID, (U32)mtctx->jobs[wJobID].consumed, (U32)mtctx->jobs[wJobID].src.size);
                break;
            }
            DEBUGLOG(5, "waiting for something to flush from job %u (currently flushed: %u bytes)",
                        mtctx->doneJobID, (U32)mtctx->jobs[wJobID].dstFlushed);
            ZSTD_pthread_cond_wait(&mtctx->jobs[wJobID].job_cond, &mtctx->jobs[wJobID].job_mutex);  /* block when nothing to flush but some to come */
    }   }

    /* try to flush something */
    {   size_t cSize = mtctx->jobs[wJobID].cSize;                  /* shared */
        size_t const srcConsumed = mtctx->jobs[wJobID].consumed;   /* shared */
        size_t const srcSize = mtctx->jobs[wJobID].src.size;        /* read-only, could be done after mutex lock, but no-declaration-after-statement */
        ZSTD_pthread_mutex_unlock(&mtctx->jobs[wJobID].job_mutex);
        if (ZSTD_isError(cSize)) {
            DEBUGLOG(5, "ZSTDMT_flushProduced: job %u : compression error detected : %s",
                        mtctx->doneJobID, ZSTD_getErrorName(cSize));
            ZSTDMT_waitForAllJobsCompleted(mtctx);
            ZSTDMT_releaseAllJobResources(mtctx);
            return cSize;
        }
        /* add frame checksum if necessary (can only happen once) */
        assert(srcConsumed <= srcSize);
        if ( (srcConsumed == srcSize)   /* job completed -> worker no longer active */
          && mtctx->jobs[wJobID].frameChecksumNeeded ) {
            U32 const checksum = (U32)XXH64_digest(&mtctx->serial.xxhState);
            DEBUGLOG(4, "ZSTDMT_flushProduced: writing checksum : %08X \n", checksum);
            MEM_writeLE32((char*)mtctx->jobs[wJobID].dstBuff.start + mtctx->jobs[wJobID].cSize, checksum);
            cSize += 4;
            mtctx->jobs[wJobID].cSize += 4;  /* can write this shared value, as worker is no longer active */
            mtctx->jobs[wJobID].frameChecksumNeeded = 0;
        }
        if (cSize > 0) {   /* compression is ongoing or completed */
            size_t const toFlush = MIN(cSize - mtctx->jobs[wJobID].dstFlushed, output->size - output->pos);
            DEBUGLOG(5, "ZSTDMT_flushProduced: Flushing %u bytes from job %u (completion:%u/%u, generated:%u)",
                        (U32)toFlush, mtctx->doneJobID, (U32)srcConsumed, (U32)srcSize, (U32)cSize);
            assert(mtctx->doneJobID < mtctx->nextJobID);
            assert(cSize >= mtctx->jobs[wJobID].dstFlushed);
            assert(mtctx->jobs[wJobID].dstBuff.start != NULL);
            memcpy((char*)output->dst + output->pos,
                   (const char*)mtctx->jobs[wJobID].dstBuff.start + mtctx->jobs[wJobID].dstFlushed,
                   toFlush);
            output->pos += toFlush;
            mtctx->jobs[wJobID].dstFlushed += toFlush;  /* can write : this value is only used by mtctx */

            if ( (srcConsumed == srcSize)    /* job completed */
              && (mtctx->jobs[wJobID].dstFlushed == cSize) ) {   /* output buffer fully flushed => free this job position */
                DEBUGLOG(5, "Job %u completed (%u bytes), moving to next one",
                        mtctx->doneJobID, (U32)mtctx->jobs[wJobID].dstFlushed);
                ZSTDMT_releaseBuffer(mtctx->bufPool, mtctx->jobs[wJobID].dstBuff);
                mtctx->jobs[wJobID].dstBuff = g_nullBuffer;
                mtctx->jobs[wJobID].cSize = 0;   /* ensure this job slot is considered "not started" in future check */
                mtctx->consumed += srcSize;
                mtctx->produced += cSize;
                mtctx->doneJobID++;
        }   }

        /* return value : how many bytes left in buffer ; fake it to 1 when unknown but >0 */
        if (cSize > mtctx->jobs[wJobID].dstFlushed) return (cSize - mtctx->jobs[wJobID].dstFlushed);
        if (srcSize > srcConsumed) return 1;   /* current job not completely compressed */
    }
    if (mtctx->doneJobID < mtctx->nextJobID) return 1;   /* some more jobs ongoing */
    if (mtctx->jobReady) return 1;      /* one job is ready to push, just not yet in the list */
    if (mtctx->inBuff.filled > 0) return 1;   /* input is not empty, and still needs to be converted into a job */
    mtctx->allJobsCompleted = mtctx->frameEnded;   /* all jobs are entirely flushed => if this one is last one, frame is completed */
    if (end == ZSTD_e_end) return !mtctx->frameEnded;  /* for ZSTD_e_end, question becomes : is frame completed ? instead of : are internal buffers fully flushed ? */
    return 0;   /* internal buffers fully flushed */
}

/**
 * Returns the range of data used by the earliest job that is not yet complete.
 * If the data of the first job is broken up into two segments, we cover both
 * sections.
 */
static range_t ZSTDMT_getInputDataInUse(ZSTDMT_CCtx* mtctx)
{
    unsigned const firstJobID = mtctx->doneJobID;
    unsigned const lastJobID = mtctx->nextJobID;
    unsigned jobID;

    for (jobID = firstJobID; jobID < lastJobID; ++jobID) {
        unsigned const wJobID = jobID & mtctx->jobIDMask;
        size_t consumed;

        ZSTD_PTHREAD_MUTEX_LOCK(&mtctx->jobs[wJobID].job_mutex);
        consumed = mtctx->jobs[wJobID].consumed;
        ZSTD_pthread_mutex_unlock(&mtctx->jobs[wJobID].job_mutex);

        if (consumed < mtctx->jobs[wJobID].src.size) {
            range_t range = mtctx->jobs[wJobID].prefix;
            if (range.size == 0) {
                /* Empty prefix */
                range = mtctx->jobs[wJobID].src;
            }
            /* Job source in multiple segments not supported yet */
            assert(range.start <= mtctx->jobs[wJobID].src.start);
            return range;
        }
    }
    return kNullRange;
}

/**
 * Returns non-zero iff buffer and range overlap.
 */
static int ZSTDMT_isOverlapped(buffer_t buffer, range_t range)
{
    BYTE const* const bufferStart = (BYTE const*)buffer.start;
    BYTE const* const bufferEnd = bufferStart + buffer.capacity;
    BYTE const* const rangeStart = (BYTE const*)range.start;
    BYTE const* const rangeEnd = rangeStart + range.size;

    if (rangeStart == NULL || bufferStart == NULL)
        return 0;
    /* Empty ranges cannot overlap */
    if (bufferStart == bufferEnd || rangeStart == rangeEnd)
        return 0;

    return bufferStart < rangeEnd && rangeStart < bufferEnd;
}

static int ZSTDMT_doesOverlapWindow(buffer_t buffer, ZSTD_window_t window)
{
    range_t extDict;
    range_t prefix;

    extDict.start = window.dictBase + window.lowLimit;
    extDict.size = window.dictLimit - window.lowLimit;

    prefix.start = window.base + window.dictLimit;
    prefix.size = window.nextSrc - (window.base + window.dictLimit);
    DEBUGLOG(5, "extDict [0x%zx, 0x%zx)",
                (size_t)extDict.start,
                (size_t)extDict.start + extDict.size);
    DEBUGLOG(5, "prefix  [0x%zx, 0x%zx)",
                (size_t)prefix.start,
                (size_t)prefix.start + prefix.size);

    return ZSTDMT_isOverlapped(buffer, extDict)
        || ZSTDMT_isOverlapped(buffer, prefix);
}

static void ZSTDMT_waitForLdmComplete(ZSTDMT_CCtx* mtctx, buffer_t buffer)
{
    if (mtctx->params.ldmParams.enableLdm) {
        ZSTD_pthread_mutex_t* mutex = &mtctx->serial.ldmWindowMutex;
        DEBUGLOG(5, "source  [0x%zx, 0x%zx)",
                    (size_t)buffer.start,
                    (size_t)buffer.start + buffer.capacity);
        ZSTD_PTHREAD_MUTEX_LOCK(mutex);
        while (ZSTDMT_doesOverlapWindow(buffer, mtctx->serial.ldmWindow)) {
            DEBUGLOG(6, "Waiting for LDM to finish...");
            ZSTD_pthread_cond_wait(&mtctx->serial.ldmWindowCond, mutex);
        }
        DEBUGLOG(6, "Done waiting for LDM to finish");
        ZSTD_pthread_mutex_unlock(mutex);
    }
}

/**
 * Attempts to set the inBuff to the next section to fill.
 * If any part of the new section is still in use we give up.
 * Returns non-zero if the buffer is filled.
 */
static int ZSTDMT_tryGetInputRange(ZSTDMT_CCtx* mtctx)
{
    range_t const inUse = ZSTDMT_getInputDataInUse(mtctx);
    size_t const spaceLeft = mtctx->roundBuff.capacity - mtctx->roundBuff.pos;
    size_t const target = mtctx->targetSectionSize;
    buffer_t buffer;

    assert(mtctx->inBuff.buffer.start == NULL);
    assert(mtctx->roundBuff.capacity >= target);

    if (spaceLeft < target) {
        /* ZSTD_invalidateRepCodes() doesn't work for extDict variants.
         * Simply copy the prefix to the beginning in that case.
         */
        BYTE* const start = (BYTE*)mtctx->roundBuff.buffer;
        size_t const prefixSize = mtctx->inBuff.prefix.size;

        buffer.start = start;
        buffer.capacity = prefixSize;
        if (ZSTDMT_isOverlapped(buffer, inUse)) {
            DEBUGLOG(6, "Waiting for buffer...");
            return 0;
        }
        ZSTDMT_waitForLdmComplete(mtctx, buffer);
        memmove(start, mtctx->inBuff.prefix.start, prefixSize);
        mtctx->inBuff.prefix.start = start;
        mtctx->roundBuff.pos = prefixSize;
    }
    buffer.start = mtctx->roundBuff.buffer + mtctx->roundBuff.pos;
    buffer.capacity = target;

    if (ZSTDMT_isOverlapped(buffer, inUse)) {
        DEBUGLOG(6, "Waiting for buffer...");
        return 0;
    }
    assert(!ZSTDMT_isOverlapped(buffer, mtctx->inBuff.prefix));

    ZSTDMT_waitForLdmComplete(mtctx, buffer);

    DEBUGLOG(5, "Using prefix range [%zx, %zx)",
                (size_t)mtctx->inBuff.prefix.start,
                (size_t)mtctx->inBuff.prefix.start + mtctx->inBuff.prefix.size);
    DEBUGLOG(5, "Using source range [%zx, %zx)",
                (size_t)buffer.start,
                (size_t)buffer.start + buffer.capacity);


    mtctx->inBuff.buffer = buffer;
    mtctx->inBuff.filled = 0;
    assert(mtctx->roundBuff.pos + buffer.capacity <= mtctx->roundBuff.capacity);
    return 1;
}


/** ZSTDMT_compressStream_generic() :
 *  internal use only - exposed to be invoked from zstd_compress.c
 *  assumption : output and input are valid (pos <= size)
 * @return : minimum amount of data remaining to flush, 0 if none */
size_t ZSTDMT_compressStream_generic(ZSTDMT_CCtx* mtctx,
                                     ZSTD_outBuffer* output,
                                     ZSTD_inBuffer* input,
                                     ZSTD_EndDirective endOp)
{
    unsigned forwardInputProgress = 0;
    DEBUGLOG(5, "ZSTDMT_compressStream_generic (endOp=%u, srcSize=%u)",
                (U32)endOp, (U32)(input->size - input->pos));
    assert(output->pos <= output->size);
    assert(input->pos  <= input->size);

    if (mtctx->singleBlockingThread) {  /* delegate to single-thread (synchronous) */
        return ZSTD_compressStream_generic(mtctx->cctxPool->cctx[0], output, input, endOp);
    }

    if ((mtctx->frameEnded) && (endOp==ZSTD_e_continue)) {
        /* current frame being ended. Only flush/end are allowed */
        return ERROR(stage_wrong);
    }

    /* single-pass shortcut (note : synchronous-mode) */
    if ( (mtctx->nextJobID == 0)      /* just started */
      && (mtctx->inBuff.filled == 0)  /* nothing buffered */
      && (!mtctx->jobReady)           /* no job already created */
      && (endOp == ZSTD_e_end)        /* end order */
      && (output->size - output->pos >= ZSTD_compressBound(input->size - input->pos)) ) { /* enough space in dst */
        size_t const cSize = ZSTDMT_compress_advanced_internal(mtctx,
                (char*)output->dst + output->pos, output->size - output->pos,
                (const char*)input->src + input->pos, input->size - input->pos,
                mtctx->cdict, mtctx->params);
        if (ZSTD_isError(cSize)) return cSize;
        input->pos = input->size;
        output->pos += cSize;
        mtctx->allJobsCompleted = 1;
        mtctx->frameEnded = 1;
        return 0;
    }

    /* fill input buffer */
    if ( (!mtctx->jobReady)
      && (input->size > input->pos) ) {   /* support NULL input */
        if (mtctx->inBuff.buffer.start == NULL) {
            assert(mtctx->inBuff.filled == 0); /* Can't fill an empty buffer */
            if (!ZSTDMT_tryGetInputRange(mtctx)) {
                /* It is only possible for this operation to fail if there are
                 * still compression jobs ongoing.
                 */
                assert(mtctx->doneJobID != mtctx->nextJobID);
            }
        }
        if (mtctx->inBuff.buffer.start != NULL) {
            size_t const toLoad = MIN(input->size - input->pos, mtctx->targetSectionSize - mtctx->inBuff.filled);
            assert(mtctx->inBuff.buffer.capacity >= mtctx->targetSectionSize);
            DEBUGLOG(5, "ZSTDMT_compressStream_generic: adding %u bytes on top of %u to buffer of size %u",
                        (U32)toLoad, (U32)mtctx->inBuff.filled, (U32)mtctx->targetSectionSize);
            memcpy((char*)mtctx->inBuff.buffer.start + mtctx->inBuff.filled, (const char*)input->src + input->pos, toLoad);
            input->pos += toLoad;
            mtctx->inBuff.filled += toLoad;
            forwardInputProgress = toLoad>0;
        }
        if ((input->pos < input->size) && (endOp == ZSTD_e_end))
            endOp = ZSTD_e_flush;   /* can't end now : not all input consumed */
    }

    if ( (mtctx->jobReady)
      || (mtctx->inBuff.filled >= mtctx->targetSectionSize)  /* filled enough : let's compress */
      || ((endOp != ZSTD_e_continue) && (mtctx->inBuff.filled > 0))  /* something to flush : let's go */
      || ((endOp == ZSTD_e_end) && (!mtctx->frameEnded)) ) {   /* must finish the frame with a zero-size block */
        size_t const jobSize = mtctx->inBuff.filled;
        assert(mtctx->inBuff.filled <= mtctx->targetSectionSize);
        CHECK_F( ZSTDMT_createCompressionJob(mtctx, jobSize, endOp) );
    }

    /* check for potential compressed data ready to be flushed */
    {   size_t const remainingToFlush = ZSTDMT_flushProduced(mtctx, output, !forwardInputProgress, endOp); /* block if there was no forward input progress */
        if (input->pos < input->size) return MAX(remainingToFlush, 1);  /* input not consumed : do not end flush yet */
        return remainingToFlush;
    }
}


size_t ZSTDMT_compressStream(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output, ZSTD_inBuffer* input)
{
    CHECK_F( ZSTDMT_compressStream_generic(mtctx, output, input, ZSTD_e_continue) );

    /* recommended next input size : fill current input buffer */
    return mtctx->targetSectionSize - mtctx->inBuff.filled;   /* note : could be zero when input buffer is fully filled and no more availability to create new job */
}


static size_t ZSTDMT_flushStream_internal(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output, ZSTD_EndDirective endFrame)
{
    size_t const srcSize = mtctx->inBuff.filled;
    DEBUGLOG(5, "ZSTDMT_flushStream_internal");

    if ( mtctx->jobReady     /* one job ready for a worker to pick up */
      || (srcSize > 0)       /* still some data within input buffer */
      || ((endFrame==ZSTD_e_end) && !mtctx->frameEnded)) {  /* need a last 0-size block to end frame */
           DEBUGLOG(5, "ZSTDMT_flushStream_internal : create a new job (%u bytes, end:%u)",
                        (U32)srcSize, (U32)endFrame);
        CHECK_F( ZSTDMT_createCompressionJob(mtctx, srcSize, endFrame) );
    }

    /* check if there is any data available to flush */
    return ZSTDMT_flushProduced(mtctx, output, 1 /* blockToFlush */, endFrame);
}


size_t ZSTDMT_flushStream(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output)
{
    DEBUGLOG(5, "ZSTDMT_flushStream");
    if (mtctx->singleBlockingThread)
        return ZSTD_flushStream(mtctx->cctxPool->cctx[0], output);
    return ZSTDMT_flushStream_internal(mtctx, output, ZSTD_e_flush);
}

size_t ZSTDMT_endStream(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output)
{
    DEBUGLOG(4, "ZSTDMT_endStream");
    if (mtctx->singleBlockingThread)
        return ZSTD_endStream(mtctx->cctxPool->cctx[0], output);
    return ZSTDMT_flushStream_internal(mtctx, output, ZSTD_e_end);
}
