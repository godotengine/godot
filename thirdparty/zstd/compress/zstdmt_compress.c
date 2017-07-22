/**
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


/* ======   Tuning parameters   ====== */
#define ZSTDMT_NBTHREADS_MAX 128


/* ======   Compiler specifics   ====== */
#if defined(_MSC_VER)
#  pragma warning(disable : 4204)   /* disable: C4204: non-constant aggregate initializer */
#endif


/* ======   Dependencies   ====== */
#include <string.h>      /* memcpy, memset */
#include "pool.h"        /* threadpool */
#include "threading.h"   /* mutex */
#include "zstd_internal.h"  /* MIN, ERROR, ZSTD_*, ZSTD_highbit32 */
#include "zstdmt_compress.h"


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
#define PTHREAD_MUTEX_LOCK(mutex) {               \
    if (ZSTD_DEBUG>=MUTEX_WAIT_TIME_DLEVEL) {   \
        unsigned long long const beforeTime = GetCurrentClockTimeMicroseconds(); \
        pthread_mutex_lock(mutex);                \
        {   unsigned long long const afterTime = GetCurrentClockTimeMicroseconds(); \
            unsigned long long const elapsedTime = (afterTime-beforeTime); \
            if (elapsedTime > 1000) {  /* or whatever threshold you like; I'm using 1 millisecond here */ \
                DEBUGLOG(MUTEX_WAIT_TIME_DLEVEL, "Thread took %llu microseconds to acquire mutex %s \n", \
                   elapsedTime, #mutex);          \
        }   }                                     \
    } else pthread_mutex_lock(mutex);             \
}

#else

#  define PTHREAD_MUTEX_LOCK(m) pthread_mutex_lock(m)
#  define DEBUG_PRINTHEX(l,p,n) {}

#endif


/* =====   Buffer Pool   ===== */

typedef struct buffer_s {
    void* start;
    size_t size;
} buffer_t;

static const buffer_t g_nullBuffer = { NULL, 0 };

typedef struct ZSTDMT_bufferPool_s {
    unsigned totalBuffers;
    unsigned nbBuffers;
    ZSTD_customMem cMem;
    buffer_t bTable[1];   /* variable size */
} ZSTDMT_bufferPool;

static ZSTDMT_bufferPool* ZSTDMT_createBufferPool(unsigned nbThreads, ZSTD_customMem cMem)
{
    unsigned const maxNbBuffers = 2*nbThreads + 2;
    ZSTDMT_bufferPool* const bufPool = (ZSTDMT_bufferPool*)ZSTD_calloc(
        sizeof(ZSTDMT_bufferPool) + (maxNbBuffers-1) * sizeof(buffer_t), cMem);
    if (bufPool==NULL) return NULL;
    bufPool->totalBuffers = maxNbBuffers;
    bufPool->nbBuffers = 0;
    bufPool->cMem = cMem;
    return bufPool;
}

static void ZSTDMT_freeBufferPool(ZSTDMT_bufferPool* bufPool)
{
    unsigned u;
    if (!bufPool) return;   /* compatibility with free on NULL */
    for (u=0; u<bufPool->totalBuffers; u++)
        ZSTD_free(bufPool->bTable[u].start, bufPool->cMem);
    ZSTD_free(bufPool, bufPool->cMem);
}

/* only works at initialization, not during compression */
static size_t ZSTDMT_sizeof_bufferPool(ZSTDMT_bufferPool* bufPool)
{
    size_t const poolSize = sizeof(*bufPool)
                            + (bufPool->totalBuffers - 1) * sizeof(buffer_t);
    unsigned u;
    size_t totalBufferSize = 0;
    for (u=0; u<bufPool->totalBuffers; u++)
        totalBufferSize += bufPool->bTable[u].size;

    return poolSize + totalBufferSize;
}

/** ZSTDMT_getBuffer() :
 *  assumption : invocation from main thread only ! */
static buffer_t ZSTDMT_getBuffer(ZSTDMT_bufferPool* pool, size_t bSize)
{
    if (pool->nbBuffers) {   /* try to use an existing buffer */
        buffer_t const buf = pool->bTable[--(pool->nbBuffers)];
        size_t const availBufferSize = buf.size;
        if ((availBufferSize >= bSize) & (availBufferSize <= 10*bSize))
            /* large enough, but not too much */
            return buf;
        /* size conditions not respected : scratch this buffer, create new one */
        ZSTD_free(buf.start, pool->cMem);
    }
    /* create new buffer */
    {   buffer_t buffer;
        void* const start = ZSTD_malloc(bSize, pool->cMem);
        if (start==NULL) bSize = 0;
        buffer.start = start;   /* note : start can be NULL if malloc fails ! */
        buffer.size = bSize;
        return buffer;
    }
}

/* store buffer for later re-use, up to pool capacity */
static void ZSTDMT_releaseBuffer(ZSTDMT_bufferPool* pool, buffer_t buf)
{
    if (buf.start == NULL) return;   /* release on NULL */
    if (pool->nbBuffers < pool->totalBuffers) {
        pool->bTable[pool->nbBuffers++] = buf;   /* store for later re-use */
        return;
    }
    /* Reached bufferPool capacity (should not happen) */
    ZSTD_free(buf.start, pool->cMem);
}


/* =====   CCtx Pool   ===== */

typedef struct {
    unsigned totalCCtx;
    unsigned availCCtx;
    ZSTD_customMem cMem;
    ZSTD_CCtx* cctx[1];   /* variable size */
} ZSTDMT_CCtxPool;

/* assumption : CCtxPool invocation only from main thread */

/* note : all CCtx borrowed from the pool should be released back to the pool _before_ freeing the pool */
static void ZSTDMT_freeCCtxPool(ZSTDMT_CCtxPool* pool)
{
    unsigned u;
    for (u=0; u<pool->totalCCtx; u++)
        ZSTD_freeCCtx(pool->cctx[u]);  /* note : compatible with free on NULL */
    ZSTD_free(pool, pool->cMem);
}

/* ZSTDMT_createCCtxPool() :
 * implies nbThreads >= 1 , checked by caller ZSTDMT_createCCtx() */
static ZSTDMT_CCtxPool* ZSTDMT_createCCtxPool(unsigned nbThreads,
                                              ZSTD_customMem cMem)
{
    ZSTDMT_CCtxPool* const cctxPool = (ZSTDMT_CCtxPool*) ZSTD_calloc(
        sizeof(ZSTDMT_CCtxPool) + (nbThreads-1)*sizeof(ZSTD_CCtx*), cMem);
    if (!cctxPool) return NULL;
    cctxPool->cMem = cMem;
    cctxPool->totalCCtx = nbThreads;
    cctxPool->availCCtx = 1;   /* at least one cctx for single-thread mode */
    cctxPool->cctx[0] = ZSTD_createCCtx_advanced(cMem);
    if (!cctxPool->cctx[0]) { ZSTDMT_freeCCtxPool(cctxPool); return NULL; }
    DEBUGLOG(3, "cctxPool created, with %u threads", nbThreads);
    return cctxPool;
}

/* only works during initialization phase, not during compression */
static size_t ZSTDMT_sizeof_CCtxPool(ZSTDMT_CCtxPool* cctxPool)
{
    unsigned const nbThreads = cctxPool->totalCCtx;
    size_t const poolSize = sizeof(*cctxPool)
                            + (nbThreads-1)*sizeof(ZSTD_CCtx*);
    unsigned u;
    size_t totalCCtxSize = 0;
    for (u=0; u<nbThreads; u++)
        totalCCtxSize += ZSTD_sizeof_CCtx(cctxPool->cctx[u]);

    return poolSize + totalCCtxSize;
}

static ZSTD_CCtx* ZSTDMT_getCCtx(ZSTDMT_CCtxPool* pool)
{
    if (pool->availCCtx) {
        pool->availCCtx--;
        return pool->cctx[pool->availCCtx];
    }
    return ZSTD_createCCtx();   /* note : can be NULL, when creation fails ! */
}

static void ZSTDMT_releaseCCtx(ZSTDMT_CCtxPool* pool, ZSTD_CCtx* cctx)
{
    if (cctx==NULL) return;   /* compatibility with release on NULL */
    if (pool->availCCtx < pool->totalCCtx)
        pool->cctx[pool->availCCtx++] = cctx;
    else
        /* pool overflow : should not happen, since totalCCtx==nbThreads */
        ZSTD_freeCCtx(cctx);
}


/* =====   Thread worker   ===== */

typedef struct {
    buffer_t buffer;
    size_t filled;
} inBuff_t;

typedef struct {
    ZSTD_CCtx* cctx;
    buffer_t src;
    const void* srcStart;
    size_t   srcSize;
    size_t   dictSize;
    buffer_t dstBuff;
    size_t   cSize;
    size_t   dstFlushed;
    unsigned firstChunk;
    unsigned lastChunk;
    unsigned jobCompleted;
    unsigned jobScanned;
    pthread_mutex_t* jobCompleted_mutex;
    pthread_cond_t* jobCompleted_cond;
    ZSTD_parameters params;
    const ZSTD_CDict* cdict;
    unsigned long long fullFrameSize;
} ZSTDMT_jobDescription;

/* ZSTDMT_compressChunk() : POOL_function type */
void ZSTDMT_compressChunk(void* jobDescription)
{
    ZSTDMT_jobDescription* const job = (ZSTDMT_jobDescription*)jobDescription;
    const void* const src = (const char*)job->srcStart + job->dictSize;
    buffer_t const dstBuff = job->dstBuff;
    DEBUGLOG(5, "job (first:%u) (last:%u) : dictSize %u, srcSize %u",
                 job->firstChunk, job->lastChunk, (U32)job->dictSize, (U32)job->srcSize);
    if (job->cdict) {  /* should only happen for first segment */
        size_t const initError = ZSTD_compressBegin_usingCDict_advanced(job->cctx, job->cdict, job->params.fParams, job->fullFrameSize);
        DEBUGLOG(5, "using CDict");
        if (ZSTD_isError(initError)) { job->cSize = initError; goto _endJob; }
    } else {  /* srcStart points at reloaded section */
        if (!job->firstChunk) job->params.fParams.contentSizeFlag = 0;  /* ensure no srcSize control */
        {   size_t const dictModeError = ZSTD_setCCtxParameter(job->cctx, ZSTD_p_forceRawDict, 1);  /* Force loading dictionary in "content-only" mode (no header analysis) */
            size_t const initError = ZSTD_compressBegin_advanced(job->cctx, job->srcStart, job->dictSize, job->params, job->fullFrameSize);
            if (ZSTD_isError(initError) || ZSTD_isError(dictModeError)) { job->cSize = initError; goto _endJob; }
            ZSTD_setCCtxParameter(job->cctx, ZSTD_p_forceWindow, 1);
    }   }
    if (!job->firstChunk) {  /* flush and overwrite frame header when it's not first segment */
        size_t const hSize = ZSTD_compressContinue(job->cctx, dstBuff.start, dstBuff.size, src, 0);
        if (ZSTD_isError(hSize)) { job->cSize = hSize; goto _endJob; }
        ZSTD_invalidateRepCodes(job->cctx);
    }

    DEBUGLOG(5, "Compressing : ");
    DEBUG_PRINTHEX(4, job->srcStart, 12);
    job->cSize = (job->lastChunk) ?
                 ZSTD_compressEnd     (job->cctx, dstBuff.start, dstBuff.size, src, job->srcSize) :
                 ZSTD_compressContinue(job->cctx, dstBuff.start, dstBuff.size, src, job->srcSize);
    DEBUGLOG(5, "compressed %u bytes into %u bytes   (first:%u) (last:%u)",
                (unsigned)job->srcSize, (unsigned)job->cSize, job->firstChunk, job->lastChunk);
    DEBUGLOG(5, "dstBuff.size : %u ; => %s", (U32)dstBuff.size, ZSTD_getErrorName(job->cSize));

_endJob:
    PTHREAD_MUTEX_LOCK(job->jobCompleted_mutex);
    job->jobCompleted = 1;
    job->jobScanned = 0;
    pthread_cond_signal(job->jobCompleted_cond);
    pthread_mutex_unlock(job->jobCompleted_mutex);
}


/* ------------------------------------------ */
/* =====   Multi-threaded compression   ===== */
/* ------------------------------------------ */

struct ZSTDMT_CCtx_s {
    POOL_ctx* factory;
    ZSTDMT_jobDescription* jobs;
    ZSTDMT_bufferPool* buffPool;
    ZSTDMT_CCtxPool* cctxPool;
    pthread_mutex_t jobCompleted_mutex;
    pthread_cond_t jobCompleted_cond;
    size_t targetSectionSize;
    size_t marginSize;
    size_t inBuffSize;
    size_t dictSize;
    size_t targetDictSize;
    inBuff_t inBuff;
    ZSTD_parameters params;
    XXH64_state_t xxhState;
    unsigned nbThreads;
    unsigned jobIDMask;
    unsigned doneJobID;
    unsigned nextJobID;
    unsigned frameEnded;
    unsigned allJobsCompleted;
    unsigned overlapRLog;
    unsigned long long frameContentSize;
    size_t sectionSize;
    ZSTD_customMem cMem;
    ZSTD_CDict* cdictLocal;
    const ZSTD_CDict* cdict;
};

static ZSTDMT_jobDescription* ZSTDMT_allocJobsTable(U32* nbJobsPtr, ZSTD_customMem cMem)
{
    U32 const nbJobsLog2 = ZSTD_highbit32(*nbJobsPtr) + 1;
    U32 const nbJobs = 1 << nbJobsLog2;
    *nbJobsPtr = nbJobs;
    return (ZSTDMT_jobDescription*) ZSTD_calloc(
                            nbJobs * sizeof(ZSTDMT_jobDescription), cMem);
}

ZSTDMT_CCtx* ZSTDMT_createCCtx_advanced(unsigned nbThreads, ZSTD_customMem cMem)
{
    ZSTDMT_CCtx* mtctx;
    U32 nbJobs = nbThreads + 2;
    DEBUGLOG(3, "ZSTDMT_createCCtx_advanced");

    if ((nbThreads < 1) | (nbThreads > ZSTDMT_NBTHREADS_MAX)) return NULL;
    if ((cMem.customAlloc!=NULL) ^ (cMem.customFree!=NULL))
        /* invalid custom allocator */
        return NULL;

    mtctx = (ZSTDMT_CCtx*) ZSTD_calloc(sizeof(ZSTDMT_CCtx), cMem);
    if (!mtctx) return NULL;
    mtctx->cMem = cMem;
    mtctx->nbThreads = nbThreads;
    mtctx->allJobsCompleted = 1;
    mtctx->sectionSize = 0;
    mtctx->overlapRLog = 3;
    mtctx->factory = POOL_create(nbThreads, 1);
    mtctx->jobs = ZSTDMT_allocJobsTable(&nbJobs, cMem);
    mtctx->jobIDMask = nbJobs - 1;
    mtctx->buffPool = ZSTDMT_createBufferPool(nbThreads, cMem);
    mtctx->cctxPool = ZSTDMT_createCCtxPool(nbThreads, cMem);
    if (!mtctx->factory | !mtctx->jobs | !mtctx->buffPool | !mtctx->cctxPool) {
        ZSTDMT_freeCCtx(mtctx);
        return NULL;
    }
    pthread_mutex_init(&mtctx->jobCompleted_mutex, NULL);   /* Todo : check init function return */
    pthread_cond_init(&mtctx->jobCompleted_cond, NULL);
    DEBUGLOG(3, "mt_cctx created, for %u threads", nbThreads);
    return mtctx;
}

ZSTDMT_CCtx* ZSTDMT_createCCtx(unsigned nbThreads)
{
    return ZSTDMT_createCCtx_advanced(nbThreads, ZSTD_defaultCMem);
}

/* ZSTDMT_releaseAllJobResources() :
 * note : ensure all workers are killed first ! */
static void ZSTDMT_releaseAllJobResources(ZSTDMT_CCtx* mtctx)
{
    unsigned jobID;
    DEBUGLOG(3, "ZSTDMT_releaseAllJobResources");
    for (jobID=0; jobID <= mtctx->jobIDMask; jobID++) {
        ZSTDMT_releaseBuffer(mtctx->buffPool, mtctx->jobs[jobID].dstBuff);
        mtctx->jobs[jobID].dstBuff = g_nullBuffer;
        ZSTDMT_releaseBuffer(mtctx->buffPool, mtctx->jobs[jobID].src);
        mtctx->jobs[jobID].src = g_nullBuffer;
        ZSTDMT_releaseCCtx(mtctx->cctxPool, mtctx->jobs[jobID].cctx);
        mtctx->jobs[jobID].cctx = NULL;
    }
    memset(mtctx->jobs, 0, (mtctx->jobIDMask+1)*sizeof(ZSTDMT_jobDescription));
    ZSTDMT_releaseBuffer(mtctx->buffPool, mtctx->inBuff.buffer);
    mtctx->inBuff.buffer = g_nullBuffer;
    mtctx->allJobsCompleted = 1;
}

size_t ZSTDMT_freeCCtx(ZSTDMT_CCtx* mtctx)
{
    if (mtctx==NULL) return 0;   /* compatible with free on NULL */
    POOL_free(mtctx->factory);
    if (!mtctx->allJobsCompleted) ZSTDMT_releaseAllJobResources(mtctx); /* stop workers first */
    ZSTDMT_freeBufferPool(mtctx->buffPool);  /* release job resources into pools first */
    ZSTD_free(mtctx->jobs, mtctx->cMem);
    ZSTDMT_freeCCtxPool(mtctx->cctxPool);
    ZSTD_freeCDict(mtctx->cdictLocal);
    pthread_mutex_destroy(&mtctx->jobCompleted_mutex);
    pthread_cond_destroy(&mtctx->jobCompleted_cond);
    ZSTD_free(mtctx, mtctx->cMem);
    return 0;
}

size_t ZSTDMT_sizeof_CCtx(ZSTDMT_CCtx* mtctx)
{
    if (mtctx == NULL) return 0;   /* supports sizeof NULL */
    return sizeof(*mtctx)
        + POOL_sizeof(mtctx->factory)
        + ZSTDMT_sizeof_bufferPool(mtctx->buffPool)
        + (mtctx->jobIDMask+1) * sizeof(ZSTDMT_jobDescription)
        + ZSTDMT_sizeof_CCtxPool(mtctx->cctxPool)
        + ZSTD_sizeof_CDict(mtctx->cdictLocal);
}

size_t ZSTDMT_setMTCtxParameter(ZSTDMT_CCtx* mtctx, ZSDTMT_parameter parameter, unsigned value)
{
    switch(parameter)
    {
    case ZSTDMT_p_sectionSize :
        mtctx->sectionSize = value;
        return 0;
    case ZSTDMT_p_overlapSectionLog :
        DEBUGLOG(5, "ZSTDMT_p_overlapSectionLog : %u", value);
        mtctx->overlapRLog = (value >= 9) ? 0 : 9 - value;
        return 0;
    default :
        return ERROR(compressionParameter_unsupported);
    }
}


/* ------------------------------------------ */
/* =====   Multi-threaded compression   ===== */
/* ------------------------------------------ */

static unsigned computeNbChunks(size_t srcSize, unsigned windowLog, unsigned nbThreads) {
    size_t const chunkSizeTarget = (size_t)1 << (windowLog + 2);
    size_t const chunkMaxSize = chunkSizeTarget << 2;
    size_t const passSizeMax = chunkMaxSize * nbThreads;
    unsigned const multiplier = (unsigned)(srcSize / passSizeMax) + 1;
    unsigned const nbChunksLarge = multiplier * nbThreads;
    unsigned const nbChunksMax = (unsigned)(srcSize / chunkSizeTarget) + 1;
    unsigned const nbChunksSmall = MIN(nbChunksMax, nbThreads);
    return (multiplier>1) ? nbChunksLarge : nbChunksSmall;
}


size_t ZSTDMT_compress_advanced(ZSTDMT_CCtx* mtctx,
                           void* dst, size_t dstCapacity,
                     const void* src, size_t srcSize,
                     const ZSTD_CDict* cdict,
                           ZSTD_parameters const params,
                           unsigned overlapRLog)
{
    size_t const overlapSize = (overlapRLog>=9) ? 0 : (size_t)1 << (params.cParams.windowLog - overlapRLog);
    unsigned nbChunks = computeNbChunks(srcSize, params.cParams.windowLog, mtctx->nbThreads);
    size_t const proposedChunkSize = (srcSize + (nbChunks-1)) / nbChunks;
    size_t const avgChunkSize = ((proposedChunkSize & 0x1FFFF) < 0x7FFF) ? proposedChunkSize + 0xFFFF : proposedChunkSize;   /* avoid too small last block */
    const char* const srcStart = (const char*)src;
    size_t remainingSrcSize = srcSize;
    unsigned const compressWithinDst = (dstCapacity >= ZSTD_compressBound(srcSize)) ? nbChunks : (unsigned)(dstCapacity / ZSTD_compressBound(avgChunkSize));  /* presumes avgChunkSize >= 256 KB, which should be the case */
    size_t frameStartPos = 0, dstBufferPos = 0;

    DEBUGLOG(4, "nbChunks  : %2u   (chunkSize : %u bytes)   ", nbChunks, (U32)avgChunkSize);
    if (nbChunks==1) {   /* fallback to single-thread mode */
        ZSTD_CCtx* const cctx = mtctx->cctxPool->cctx[0];
        if (cdict) return ZSTD_compress_usingCDict_advanced(cctx, dst, dstCapacity, src, srcSize, cdict, params.fParams);
        return ZSTD_compress_advanced(cctx, dst, dstCapacity, src, srcSize, NULL, 0, params);
    }
    assert(avgChunkSize >= 256 KB);  /* condition for ZSTD_compressBound(A) + ZSTD_compressBound(B) <= ZSTD_compressBound(A+B), which is useful to avoid allocating extra buffers */

    if (nbChunks > mtctx->jobIDMask+1) {  /* enlarge job table */
        U32 nbJobs = nbChunks;
        ZSTD_free(mtctx->jobs, mtctx->cMem);
        mtctx->jobIDMask = 0;
        mtctx->jobs = ZSTDMT_allocJobsTable(&nbJobs, mtctx->cMem);
        if (mtctx->jobs==NULL) return ERROR(memory_allocation);
        mtctx->jobIDMask = nbJobs - 1;
    }

    {   unsigned u;
        for (u=0; u<nbChunks; u++) {
            size_t const chunkSize = MIN(remainingSrcSize, avgChunkSize);
            size_t const dstBufferCapacity = ZSTD_compressBound(chunkSize);
            buffer_t const dstAsBuffer = { (char*)dst + dstBufferPos, dstBufferCapacity };
            buffer_t const dstBuffer = u < compressWithinDst ? dstAsBuffer : ZSTDMT_getBuffer(mtctx->buffPool, dstBufferCapacity);
            ZSTD_CCtx* const cctx = ZSTDMT_getCCtx(mtctx->cctxPool);
            size_t dictSize = u ? overlapSize : 0;

            if ((cctx==NULL) || (dstBuffer.start==NULL)) {
                mtctx->jobs[u].cSize = ERROR(memory_allocation);   /* job result */
                mtctx->jobs[u].jobCompleted = 1;
                nbChunks = u+1;   /* only wait and free u jobs, instead of initially expected nbChunks ones */
                break;   /* let's wait for previous jobs to complete, but don't start new ones */
            }

            mtctx->jobs[u].srcStart = srcStart + frameStartPos - dictSize;
            mtctx->jobs[u].dictSize = dictSize;
            mtctx->jobs[u].srcSize = chunkSize;
            mtctx->jobs[u].cdict = mtctx->nextJobID==0 ? cdict : NULL;
            mtctx->jobs[u].fullFrameSize = srcSize;
            mtctx->jobs[u].params = params;
            /* do not calculate checksum within sections, but write it in header for first section */
            if (u!=0) mtctx->jobs[u].params.fParams.checksumFlag = 0;
            mtctx->jobs[u].dstBuff = dstBuffer;
            mtctx->jobs[u].cctx = cctx;
            mtctx->jobs[u].firstChunk = (u==0);
            mtctx->jobs[u].lastChunk = (u==nbChunks-1);
            mtctx->jobs[u].jobCompleted = 0;
            mtctx->jobs[u].jobCompleted_mutex = &mtctx->jobCompleted_mutex;
            mtctx->jobs[u].jobCompleted_cond = &mtctx->jobCompleted_cond;

            DEBUGLOG(5, "posting job %u   (%u bytes)", u, (U32)chunkSize);
            DEBUG_PRINTHEX(6, mtctx->jobs[u].srcStart, 12);
            POOL_add(mtctx->factory, ZSTDMT_compressChunk, &mtctx->jobs[u]);

            frameStartPos += chunkSize;
            dstBufferPos += dstBufferCapacity;
            remainingSrcSize -= chunkSize;
    }   }

    /* collect result */
    {   unsigned chunkID;
        size_t error = 0, dstPos = 0;
        for (chunkID=0; chunkID<nbChunks; chunkID++) {
            DEBUGLOG(5, "waiting for chunk %u ", chunkID);
            PTHREAD_MUTEX_LOCK(&mtctx->jobCompleted_mutex);
            while (mtctx->jobs[chunkID].jobCompleted==0) {
                DEBUGLOG(5, "waiting for jobCompleted signal from chunk %u", chunkID);
                pthread_cond_wait(&mtctx->jobCompleted_cond, &mtctx->jobCompleted_mutex);
            }
            pthread_mutex_unlock(&mtctx->jobCompleted_mutex);
            DEBUGLOG(5, "ready to write chunk %u ", chunkID);

            ZSTDMT_releaseCCtx(mtctx->cctxPool, mtctx->jobs[chunkID].cctx);
            mtctx->jobs[chunkID].cctx = NULL;
            mtctx->jobs[chunkID].srcStart = NULL;
            {   size_t const cSize = mtctx->jobs[chunkID].cSize;
                if (ZSTD_isError(cSize)) error = cSize;
                if ((!error) && (dstPos + cSize > dstCapacity)) error = ERROR(dstSize_tooSmall);
                if (chunkID) {   /* note : chunk 0 is written directly at dst, which is correct position */
                    if (!error)
                        memmove((char*)dst + dstPos, mtctx->jobs[chunkID].dstBuff.start, cSize);  /* may overlap when chunk compressed within dst */
                    if (chunkID >= compressWithinDst) {  /* chunk compressed into its own buffer, which must be released */
                        DEBUGLOG(5, "releasing buffer %u>=%u", chunkID, compressWithinDst);
                        ZSTDMT_releaseBuffer(mtctx->buffPool, mtctx->jobs[chunkID].dstBuff);
                    }
                    mtctx->jobs[chunkID].dstBuff = g_nullBuffer;
                }
                dstPos += cSize ;
            }
        }
        if (!error) DEBUGLOG(4, "compressed size : %u  ", (U32)dstPos);
        return error ? error : dstPos;
    }
}


size_t ZSTDMT_compressCCtx(ZSTDMT_CCtx* mtctx,
                           void* dst, size_t dstCapacity,
                     const void* src, size_t srcSize,
                           int compressionLevel)
{
    U32 const overlapRLog = (compressionLevel >= ZSTD_maxCLevel()) ? 0 : 3;
    ZSTD_parameters params = ZSTD_getParams(compressionLevel, srcSize, 0);
    params.fParams.contentSizeFlag = 1;
    return ZSTDMT_compress_advanced(mtctx, dst, dstCapacity, src, srcSize, NULL, params, overlapRLog);
}


/* ====================================== */
/* =======      Streaming API     ======= */
/* ====================================== */

static void ZSTDMT_waitForAllJobsCompleted(ZSTDMT_CCtx* zcs)
{
    DEBUGLOG(4, "ZSTDMT_waitForAllJobsCompleted");
    while (zcs->doneJobID < zcs->nextJobID) {
        unsigned const jobID = zcs->doneJobID & zcs->jobIDMask;
        PTHREAD_MUTEX_LOCK(&zcs->jobCompleted_mutex);
        while (zcs->jobs[jobID].jobCompleted==0) {
            DEBUGLOG(5, "waiting for jobCompleted signal from chunk %u", zcs->doneJobID);   /* we want to block when waiting for data to flush */
            pthread_cond_wait(&zcs->jobCompleted_cond, &zcs->jobCompleted_mutex);
        }
        pthread_mutex_unlock(&zcs->jobCompleted_mutex);
        zcs->doneJobID++;
    }
}


/** ZSTDMT_initCStream_internal() :
 *  internal usage only */
size_t ZSTDMT_initCStream_internal(ZSTDMT_CCtx* zcs,
                    const void* dict, size_t dictSize, const ZSTD_CDict* cdict,
                    ZSTD_parameters params, unsigned long long pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTDMT_initCStream_internal");
    /* params are supposed to be fully validated at this point */
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));
    assert(!((dict) && (cdict)));  /* either dict or cdict, not both */

    if (zcs->nbThreads==1) {
        DEBUGLOG(4, "single thread mode");
        return ZSTD_initCStream_internal(zcs->cctxPool->cctx[0],
                                dict, dictSize, cdict,
                                params, pledgedSrcSize);
    }

    if (zcs->allJobsCompleted == 0) {   /* previous compression not correctly finished */
        ZSTDMT_waitForAllJobsCompleted(zcs);
        ZSTDMT_releaseAllJobResources(zcs);
        zcs->allJobsCompleted = 1;
    }

    zcs->params = params;
    zcs->frameContentSize = pledgedSrcSize;
    if (dict) {
        DEBUGLOG(4,"cdictLocal: %08X", (U32)(size_t)zcs->cdictLocal);
        ZSTD_freeCDict(zcs->cdictLocal);
        zcs->cdictLocal = ZSTD_createCDict_advanced(dict, dictSize,
                                                    0 /* byRef */, ZSTD_dm_auto,   /* note : a loadPrefix becomes an internal CDict */
                                                    params.cParams, zcs->cMem);
        zcs->cdict = zcs->cdictLocal;
        if (zcs->cdictLocal == NULL) return ERROR(memory_allocation);
    } else {
        DEBUGLOG(4,"cdictLocal: %08X", (U32)(size_t)zcs->cdictLocal);
        ZSTD_freeCDict(zcs->cdictLocal);
        zcs->cdictLocal = NULL;
        zcs->cdict = cdict;
    }

    zcs->targetDictSize = (zcs->overlapRLog>=9) ? 0 : (size_t)1 << (zcs->params.cParams.windowLog - zcs->overlapRLog);
    DEBUGLOG(4, "overlapRLog : %u ", zcs->overlapRLog);
    DEBUGLOG(4, "overlap Size : %u KB", (U32)(zcs->targetDictSize>>10));
    zcs->targetSectionSize = zcs->sectionSize ? zcs->sectionSize : (size_t)1 << (zcs->params.cParams.windowLog + 2);
    zcs->targetSectionSize = MAX(ZSTDMT_SECTION_SIZE_MIN, zcs->targetSectionSize);
    zcs->targetSectionSize = MAX(zcs->targetDictSize, zcs->targetSectionSize);
    DEBUGLOG(4, "Section Size : %u KB", (U32)(zcs->targetSectionSize>>10));
    zcs->marginSize = zcs->targetSectionSize >> 2;
    zcs->inBuffSize = zcs->targetDictSize + zcs->targetSectionSize + zcs->marginSize;
    zcs->inBuff.buffer = ZSTDMT_getBuffer(zcs->buffPool, zcs->inBuffSize);
    if (zcs->inBuff.buffer.start == NULL) return ERROR(memory_allocation);
    zcs->inBuff.filled = 0;
    zcs->dictSize = 0;
    zcs->doneJobID = 0;
    zcs->nextJobID = 0;
    zcs->frameEnded = 0;
    zcs->allJobsCompleted = 0;
    if (params.fParams.checksumFlag) XXH64_reset(&zcs->xxhState, 0);
    return 0;
}

size_t ZSTDMT_initCStream_advanced(ZSTDMT_CCtx* mtctx,
                                const void* dict, size_t dictSize,
                                ZSTD_parameters params, unsigned long long pledgedSrcSize)
{
    DEBUGLOG(5, "ZSTDMT_initCStream_advanced");
    return ZSTDMT_initCStream_internal(mtctx, dict, dictSize, NULL, params, pledgedSrcSize);
}

size_t ZSTDMT_initCStream_usingCDict(ZSTDMT_CCtx* mtctx,
                               const ZSTD_CDict* cdict,
                                     ZSTD_frameParameters fParams,
                                     unsigned long long pledgedSrcSize)
{
    ZSTD_parameters params = ZSTD_getParamsFromCDict(cdict);
    if (cdict==NULL) return ERROR(dictionary_wrong);   /* method incompatible with NULL cdict */
    params.fParams = fParams;
    return ZSTDMT_initCStream_internal(mtctx, NULL, 0 /*dictSize*/, cdict,
                                        params, pledgedSrcSize);
}


/* ZSTDMT_resetCStream() :
 * pledgedSrcSize is optional and can be zero == unknown */
size_t ZSTDMT_resetCStream(ZSTDMT_CCtx* zcs, unsigned long long pledgedSrcSize)
{
    if (zcs->nbThreads==1)
        return ZSTD_resetCStream(zcs->cctxPool->cctx[0], pledgedSrcSize);
    return ZSTDMT_initCStream_internal(zcs, NULL, 0, 0, zcs->params, pledgedSrcSize);
}

size_t ZSTDMT_initCStream(ZSTDMT_CCtx* zcs, int compressionLevel) {
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, 0, 0);
    return ZSTDMT_initCStream_internal(zcs, NULL, 0, NULL, params, 0);
}


static size_t ZSTDMT_createCompressionJob(ZSTDMT_CCtx* zcs, size_t srcSize, unsigned endFrame)
{
    size_t const dstBufferCapacity = ZSTD_compressBound(srcSize);
    buffer_t const dstBuffer = ZSTDMT_getBuffer(zcs->buffPool, dstBufferCapacity);
    ZSTD_CCtx* const cctx = ZSTDMT_getCCtx(zcs->cctxPool);
    unsigned const jobID = zcs->nextJobID & zcs->jobIDMask;

    if ((cctx==NULL) || (dstBuffer.start==NULL)) {
        zcs->jobs[jobID].jobCompleted = 1;
        zcs->nextJobID++;
        ZSTDMT_waitForAllJobsCompleted(zcs);
        ZSTDMT_releaseAllJobResources(zcs);
        return ERROR(memory_allocation);
    }

    DEBUGLOG(4, "preparing job %u to compress %u bytes with %u preload ",
                zcs->nextJobID, (U32)srcSize, (U32)zcs->dictSize);
    zcs->jobs[jobID].src = zcs->inBuff.buffer;
    zcs->jobs[jobID].srcStart = zcs->inBuff.buffer.start;
    zcs->jobs[jobID].srcSize = srcSize;
    zcs->jobs[jobID].dictSize = zcs->dictSize;
    assert(zcs->inBuff.filled >= srcSize + zcs->dictSize);
    zcs->jobs[jobID].params = zcs->params;
    /* do not calculate checksum within sections, but write it in header for first section */
    if (zcs->nextJobID) zcs->jobs[jobID].params.fParams.checksumFlag = 0;
    zcs->jobs[jobID].cdict = zcs->nextJobID==0 ? zcs->cdict : NULL;
    zcs->jobs[jobID].fullFrameSize = zcs->frameContentSize;
    zcs->jobs[jobID].dstBuff = dstBuffer;
    zcs->jobs[jobID].cctx = cctx;
    zcs->jobs[jobID].firstChunk = (zcs->nextJobID==0);
    zcs->jobs[jobID].lastChunk = endFrame;
    zcs->jobs[jobID].jobCompleted = 0;
    zcs->jobs[jobID].dstFlushed = 0;
    zcs->jobs[jobID].jobCompleted_mutex = &zcs->jobCompleted_mutex;
    zcs->jobs[jobID].jobCompleted_cond = &zcs->jobCompleted_cond;

    /* get a new buffer for next input */
    if (!endFrame) {
        size_t const newDictSize = MIN(srcSize + zcs->dictSize, zcs->targetDictSize);
        DEBUGLOG(5, "ZSTDMT_createCompressionJob::endFrame = %u", endFrame);
        zcs->inBuff.buffer = ZSTDMT_getBuffer(zcs->buffPool, zcs->inBuffSize);
        if (zcs->inBuff.buffer.start == NULL) {   /* not enough memory to allocate next input buffer */
            zcs->jobs[jobID].jobCompleted = 1;
            zcs->nextJobID++;
            ZSTDMT_waitForAllJobsCompleted(zcs);
            ZSTDMT_releaseAllJobResources(zcs);
            return ERROR(memory_allocation);
        }
        DEBUGLOG(5, "inBuff currently filled to %u", (U32)zcs->inBuff.filled);
        zcs->inBuff.filled -= srcSize + zcs->dictSize - newDictSize;
        DEBUGLOG(5, "new job : inBuff filled to %u, with %u dict and %u src",
                    (U32)zcs->inBuff.filled, (U32)newDictSize,
                    (U32)(zcs->inBuff.filled - newDictSize));
        memmove(zcs->inBuff.buffer.start,
            (const char*)zcs->jobs[jobID].srcStart + zcs->dictSize + srcSize - newDictSize,
            zcs->inBuff.filled);
        DEBUGLOG(5, "new inBuff pre-filled");
        zcs->dictSize = newDictSize;
    } else {   /* if (endFrame==1) */
        DEBUGLOG(5, "ZSTDMT_createCompressionJob::endFrame = %u", endFrame);
        zcs->inBuff.buffer = g_nullBuffer;
        zcs->inBuff.filled = 0;
        zcs->dictSize = 0;
        zcs->frameEnded = 1;
        if (zcs->nextJobID == 0)
            /* single chunk exception : checksum is calculated directly within worker thread */
            zcs->params.fParams.checksumFlag = 0;
    }

    DEBUGLOG(4, "posting job %u : %u bytes  (end:%u) (note : doneJob = %u=>%u)",
                zcs->nextJobID,
                (U32)zcs->jobs[jobID].srcSize,
                zcs->jobs[jobID].lastChunk,
                zcs->doneJobID,
                zcs->doneJobID & zcs->jobIDMask);
    POOL_add(zcs->factory, ZSTDMT_compressChunk, &zcs->jobs[jobID]);   /* this call is blocking when thread worker pool is exhausted */
    zcs->nextJobID++;
    return 0;
}


/* ZSTDMT_flushNextJob() :
 * output : will be updated with amount of data flushed .
 * blockToFlush : if >0, the function will block and wait if there is no data available to flush .
 * @return : amount of data remaining within internal buffer, 1 if unknown but > 0, 0 if no more, or an error code */
static size_t ZSTDMT_flushNextJob(ZSTDMT_CCtx* zcs, ZSTD_outBuffer* output, unsigned blockToFlush)
{
    unsigned const wJobID = zcs->doneJobID & zcs->jobIDMask;
    if (zcs->doneJobID == zcs->nextJobID) return 0;   /* all flushed ! */
    PTHREAD_MUTEX_LOCK(&zcs->jobCompleted_mutex);
    while (zcs->jobs[wJobID].jobCompleted==0) {
        DEBUGLOG(5, "waiting for jobCompleted signal from job %u", zcs->doneJobID);
        if (!blockToFlush) { pthread_mutex_unlock(&zcs->jobCompleted_mutex); return 0; }  /* nothing ready to be flushed => skip */
        pthread_cond_wait(&zcs->jobCompleted_cond, &zcs->jobCompleted_mutex);  /* block when nothing available to flush */
    }
    pthread_mutex_unlock(&zcs->jobCompleted_mutex);
    /* compression job completed : output can be flushed */
    {   ZSTDMT_jobDescription job = zcs->jobs[wJobID];
        if (!job.jobScanned) {
            if (ZSTD_isError(job.cSize)) {
                DEBUGLOG(5, "compression error detected ");
                ZSTDMT_waitForAllJobsCompleted(zcs);
                ZSTDMT_releaseAllJobResources(zcs);
                return job.cSize;
            }
            ZSTDMT_releaseCCtx(zcs->cctxPool, job.cctx);
            zcs->jobs[wJobID].cctx = NULL;
            DEBUGLOG(5, "zcs->params.fParams.checksumFlag : %u ", zcs->params.fParams.checksumFlag);
            if (zcs->params.fParams.checksumFlag) {
                XXH64_update(&zcs->xxhState, (const char*)job.srcStart + job.dictSize, job.srcSize);
                if (zcs->frameEnded && (zcs->doneJobID+1 == zcs->nextJobID)) {  /* write checksum at end of last section */
                    U32 const checksum = (U32)XXH64_digest(&zcs->xxhState);
                    DEBUGLOG(5, "writing checksum : %08X \n", checksum);
                    MEM_writeLE32((char*)job.dstBuff.start + job.cSize, checksum);
                    job.cSize += 4;
                    zcs->jobs[wJobID].cSize += 4;
            }   }
            ZSTDMT_releaseBuffer(zcs->buffPool, job.src);
            zcs->jobs[wJobID].srcStart = NULL;
            zcs->jobs[wJobID].src = g_nullBuffer;
            zcs->jobs[wJobID].jobScanned = 1;
        }
        {   size_t const toWrite = MIN(job.cSize - job.dstFlushed, output->size - output->pos);
            DEBUGLOG(5, "Flushing %u bytes from job %u ", (U32)toWrite, zcs->doneJobID);
            memcpy((char*)output->dst + output->pos, (const char*)job.dstBuff.start + job.dstFlushed, toWrite);
            output->pos += toWrite;
            job.dstFlushed += toWrite;
        }
        if (job.dstFlushed == job.cSize) {   /* output buffer fully flushed => move to next one */
            ZSTDMT_releaseBuffer(zcs->buffPool, job.dstBuff);
            zcs->jobs[wJobID].dstBuff = g_nullBuffer;
            zcs->jobs[wJobID].jobCompleted = 0;
            zcs->doneJobID++;
        } else {
            zcs->jobs[wJobID].dstFlushed = job.dstFlushed;
        }
        /* return value : how many bytes left in buffer ; fake it to 1 if unknown but >0 */
        if (job.cSize > job.dstFlushed) return (job.cSize - job.dstFlushed);
        if (zcs->doneJobID < zcs->nextJobID) return 1;   /* still some buffer to flush */
        zcs->allJobsCompleted = zcs->frameEnded;   /* frame completed and entirely flushed */
        return 0;   /* everything flushed */
}   }


/** ZSTDMT_compressStream_generic() :
 *  internal use only
 *  assumption : output and input are valid (pos <= size)
 * @return : minimum amount of data remaining to flush, 0 if none */
size_t ZSTDMT_compressStream_generic(ZSTDMT_CCtx* mtctx,
                                     ZSTD_outBuffer* output,
                                     ZSTD_inBuffer* input,
                                     ZSTD_EndDirective endOp)
{
    size_t const newJobThreshold = mtctx->dictSize + mtctx->targetSectionSize + mtctx->marginSize;
    assert(output->pos <= output->size);
    assert(input->pos  <= input->size);
    if ((mtctx->frameEnded) && (endOp==ZSTD_e_continue)) {
        /* current frame being ended. Only flush/end are allowed. Or start new frame with init */
        return ERROR(stage_wrong);
    }
    if (mtctx->nbThreads==1) {
        return ZSTD_compressStream_generic(mtctx->cctxPool->cctx[0], output, input, endOp);
    }

    /* single-pass shortcut (note : this is blocking-mode) */
    if ( (mtctx->nextJobID==0)      /* just started */
      && (mtctx->inBuff.filled==0)  /* nothing buffered */
      && (endOp==ZSTD_e_end)        /* end order */
      && (output->size - output->pos >= ZSTD_compressBound(input->size - input->pos)) ) { /* enough room */
        size_t const cSize = ZSTDMT_compress_advanced(mtctx,
                (char*)output->dst + output->pos, output->size - output->pos,
                (const char*)input->src + input->pos, input->size - input->pos,
                mtctx->cdict, mtctx->params, mtctx->overlapRLog);
        if (ZSTD_isError(cSize)) return cSize;
        input->pos = input->size;
        output->pos += cSize;
        ZSTDMT_releaseBuffer(mtctx->buffPool, mtctx->inBuff.buffer);  /* was allocated in initStream */
        mtctx->allJobsCompleted = 1;
        mtctx->frameEnded = 1;
        return 0;
    }

    /* fill input buffer */
    if ((input->src) && (mtctx->inBuff.buffer.start)) {   /* support NULL input */
        size_t const toLoad = MIN(input->size - input->pos, mtctx->inBuffSize - mtctx->inBuff.filled);
        DEBUGLOG(2, "inBuff:%08X;  inBuffSize=%u;  ToCopy=%u", (U32)(size_t)mtctx->inBuff.buffer.start, (U32)mtctx->inBuffSize, (U32)toLoad);
        memcpy((char*)mtctx->inBuff.buffer.start + mtctx->inBuff.filled, (const char*)input->src + input->pos, toLoad);
        input->pos += toLoad;
        mtctx->inBuff.filled += toLoad;
    }

    if ( (mtctx->inBuff.filled >= newJobThreshold)  /* filled enough : let's compress */
      && (mtctx->nextJobID <= mtctx->doneJobID + mtctx->jobIDMask) ) {   /* avoid overwriting job round buffer */
        CHECK_F( ZSTDMT_createCompressionJob(mtctx, mtctx->targetSectionSize, 0 /* endFrame */) );
    }

    /* check for potential compressed data ready to be flushed */
    CHECK_F( ZSTDMT_flushNextJob(mtctx, output, (mtctx->inBuff.filled == mtctx->inBuffSize) /* blockToFlush */) ); /* block if it wasn't possible to create new job due to saturation */

    if (input->pos < input->size)  /* input not consumed : do not flush yet */
        endOp = ZSTD_e_continue;

    switch(endOp)
    {
        case ZSTD_e_flush:
            return ZSTDMT_flushStream(mtctx, output);
        case ZSTD_e_end:
            return ZSTDMT_endStream(mtctx, output);
        case ZSTD_e_continue:
            return 1;
        default:
            return ERROR(GENERIC);   /* invalid endDirective */
    }
}


size_t ZSTDMT_compressStream(ZSTDMT_CCtx* zcs, ZSTD_outBuffer* output, ZSTD_inBuffer* input)
{
    CHECK_F( ZSTDMT_compressStream_generic(zcs, output, input, ZSTD_e_continue) );

    /* recommended next input size : fill current input buffer */
    return zcs->inBuffSize - zcs->inBuff.filled;   /* note : could be zero when input buffer is fully filled and no more availability to create new job */
}


static size_t ZSTDMT_flushStream_internal(ZSTDMT_CCtx* zcs, ZSTD_outBuffer* output, unsigned endFrame)
{
    size_t const srcSize = zcs->inBuff.filled - zcs->dictSize;

    if ( ((srcSize > 0) || (endFrame && !zcs->frameEnded))
       && (zcs->nextJobID <= zcs->doneJobID + zcs->jobIDMask) ) {
        CHECK_F( ZSTDMT_createCompressionJob(zcs, srcSize, endFrame) );
    }

    /* check if there is any data available to flush */
    return ZSTDMT_flushNextJob(zcs, output, 1 /* blockToFlush */);
}


size_t ZSTDMT_flushStream(ZSTDMT_CCtx* zcs, ZSTD_outBuffer* output)
{
    DEBUGLOG(5, "ZSTDMT_flushStream");
    if (zcs->nbThreads==1)
        return ZSTD_flushStream(zcs->cctxPool->cctx[0], output);
    return ZSTDMT_flushStream_internal(zcs, output, 0 /* endFrame */);
}

size_t ZSTDMT_endStream(ZSTDMT_CCtx* zcs, ZSTD_outBuffer* output)
{
    DEBUGLOG(4, "ZSTDMT_endStream");
    if (zcs->nbThreads==1)
        return ZSTD_endStream(zcs->cctxPool->cctx[0], output);
    return ZSTDMT_flushStream_internal(zcs, output, 1 /* endFrame */);
}
