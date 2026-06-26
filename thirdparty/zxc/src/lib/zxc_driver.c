/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_driver.c
 * @brief Userspace @c FILE*-flavored driver: multi-threaded streaming and
 *        the seekable @c FILE* open helper.
 *
 * Two distinct subsystems live in this translation unit because they share
 * the same userspace-only host requirements (@c <stdio.h>, threading, and
 * platform file-descriptor extraction): keeping them together means a
 * single TU to exclude when building for kernel / freestanding targets.
 *
 *   1. Streaming engine: a ring-buffer producer / worker / consumer
 *      pipeline that parallelises block processing over @c FILE* streams.
 *      Public API: @ref zxc_stream_compress, @ref zxc_stream_decompress,
 *      @ref zxc_stream_get_decompressed_size.
 *
 *   2. Seekable @c FILE* wrapper: builds a @ref zxc_reader_t whose
 *      @c read_at uses @c pread / @c ReadFile on the file descriptor
 *      extracted from a @c FILE*, then delegates to
 *      @ref zxc_seekable_open_reader.  Public API:
 *      @ref zxc_seekable_open_file.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_dict.h"
#include "../../include/zxc_error.h"
#include "../../include/zxc_seekable.h"
#include "../../include/zxc_stream.h"
#include "zxc_internal.h"

/*
 * ============================================================================
 * WINDOWS THREADING EMULATION
 * ============================================================================
 * Maps POSIX pthread calls to Windows Native API (CriticalSection,
 * ConditionVariable, Threads). Allows the same threading logic to compile on
 * Linux/macOS and Windows.
 */
#if defined(_WIN32)
#include <io.h> /* _get_osfhandle, _fileno (used by zxc_seekable_open_file) */
#include <malloc.h>
#include <process.h>
#include <sys/types.h>
#include <windows.h>

// Map POSIX file positioning functions to Windows equivalents
#define fseeko _fseeki64
#define ftello _ftelli64

/**
 * @brief Returns the logical-processor count (backs the @c sysconf shim below).
 * @return Number of processors reported by @c GetSystemInfo.
 */
static int zxc_get_num_procs(void) {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;

#define pthread_mutex_init(m, a) InitializeCriticalSection(m)
#define pthread_mutex_destroy(m) DeleteCriticalSection(m)
#define pthread_mutex_lock(m) EnterCriticalSection(m)
#define pthread_mutex_unlock(m) LeaveCriticalSection(m)

#define pthread_cond_init(c, a) InitializeConditionVariable(c)
#define pthread_cond_destroy(c) (void)(0)
#define pthread_cond_wait(c, m) SleepConditionVariableCS(c, m, INFINITE)
#define pthread_cond_signal(c) WakeConditionVariable(c)
#define pthread_cond_broadcast(c) WakeAllConditionVariable(c)

/**
 * @brief Trampoline payload bridging the POSIX @c void*(*)(void*) worker
 *        signature to the @c _beginthreadex entry point.
 *
 * Heap-allocated by the @c pthread_create shim and freed by
 * @ref zxc_win_thread_entry once the captured worker has started.
 */
typedef struct {
    void* (*func)(void*); /* worker to invoke */
    void* arg;            /* argument forwarded to @c func */
} zxc_win_thread_arg_t;

/**
 * @brief @c _beginthreadex entry point: unpacks the trampoline payload, frees
 *        it, then runs the captured POSIX-style worker.
 *
 * @param[in] p  Heap @ref zxc_win_thread_arg_t handed over by the creator;
 *               ownership transfers to this function.
 * @return Always 0 (the worker's @c void* result is discarded, as on POSIX).
 */
static unsigned __stdcall zxc_win_thread_entry(void* p) {
    zxc_win_thread_arg_t* a = (zxc_win_thread_arg_t*)p;
    void* (*f)(void*) = a->func;
    void* arg = a->arg;
    ZXC_FREE(a);
    f(arg);
    return 0;
}

/**
 * @brief @c pthread_create shim: spawns @p start_routine(@p arg) via
 *        @c _beginthreadex, matching the POSIX prototype.
 *
 * @param[out] thread        Receives the thread handle on success.
 * @param[in]  attr          Unused (POSIX attribute object); ignored.
 * @param[in]  start_routine Worker to run on the new thread.
 * @param[in]  arg           Opaque argument forwarded to @p start_routine.
 * @return 0 on success, @ref ZXC_ERROR_MEMORY on allocation or spawn failure.
 */
static int pthread_create(pthread_t* thread, const void* attr, void* (*start_routine)(void*),
                          void* arg) {
    (void)attr;
    zxc_win_thread_arg_t* wrapper = ZXC_MALLOC(sizeof(zxc_win_thread_arg_t));
    if (UNLIKELY(!wrapper)) return ZXC_ERROR_MEMORY;
    wrapper->func = start_routine;
    wrapper->arg = arg;
    uintptr_t handle = _beginthreadex(NULL, 0, zxc_win_thread_entry, wrapper, 0, NULL);
    if (UNLIKELY(handle == 0)) {
        ZXC_FREE(wrapper);
        return ZXC_ERROR_MEMORY;
    }
    *thread = (HANDLE)handle;
    return 0;
}

/**
 * @brief @c pthread_join shim: blocks until @p thread finishes, then closes its
 *        handle.
 *
 * @param[in] thread  Handle from a successful @c pthread_create.
 * @param[in] retval  Unused (POSIX exit-value out-param); ignored.
 * @return Always 0.
 */
static int pthread_join(pthread_t thread, void** retval) {
    (void)retval;
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

#define sysconf(x) zxc_get_num_procs()
#define _SC_NPROCESSORS_ONLN 0

#else
#include <pthread.h>
#include <unistd.h>
#endif

/*
 * ============================================================================
 * STREAMING ENGINE (Producer / Worker / Consumer)
 * ============================================================================
 * Implements a Ring Buffer architecture to parallelize block processing.
 */

/**
 * @enum job_status_t
 * @brief Represents the lifecycle states of a processing job within the ring
 * buffer.
 *
 * @var JOB_STATUS_FREE
 *      The job slot is empty and available to be filled with new data by the
 * writer.
 * @var JOB_STATUS_FILLED
 *      The job slot has been populated with input data and is ready for
 * processing by a worker.
 * @var JOB_STATUS_PROCESSED
 *      The worker has finished processing the data; the result is ready to be
 * consumed/written out.
 */
typedef enum { JOB_STATUS_FREE, JOB_STATUS_FILLED, JOB_STATUS_PROCESSED } job_status_t;

/**
 * @struct zxc_stream_job_t
 * @brief Represents a single unit of work (a chunk of data) to be processed.
 *
 * This structure holds the input and output buffers for a specific chunk of
 * data, along with its processing status. It is padded to align with cache
 * lines to prevent false sharing in a multi-threaded environment.
 *
 * @var zxc_stream_job_t::in_buf
 *      Pointer to the buffer containing raw input data.
 * @var zxc_stream_job_t::in_cap
 *      The total allocated capacity of the input buffer.
 * @var zxc_stream_job_t::in_sz
 *      The actual size of the valid data currently in the input buffer.
 * @var zxc_stream_job_t::out_buf
 *      Pointer to the buffer where processed (compressed/decompressed) data is
 * stored.
 * @var zxc_stream_job_t::out_cap
 *      The total allocated capacity of the output buffer.
 * @var zxc_stream_job_t::result_sz
 *      The actual size of the valid data produced in the output buffer.
 * @var zxc_stream_job_t::job_id
 *      A unique identifier for the job, often used for ordering or debugging.
 * @var zxc_stream_job_t::status
 *      The current state of this job (Free, Filled, or Processed).
 * @var zxc_stream_job_t::pad
 *      Padding bytes to ensure the structure size aligns with the cache line
 * size (@c ZXC_CACHE_LINE_SIZE), minimizing cache contention between threads
 * accessing adjacent jobs.
 */
typedef struct {
    uint8_t* in_buf;
    size_t in_cap;
    size_t in_sz;
    uint8_t* out_buf;
    size_t out_cap;
    size_t result_sz;
    int job_id;
    ZXC_ATOMIC job_status_t status;  // Atomic for lock-free status updates
    char pad[ZXC_CACHE_LINE_SIZE];   // Prevent False Sharing
} zxc_stream_job_t;

/**
 * @typedef zxc_chunk_processor_t
 * @brief Function pointer type for processing a chunk of data.
 *
 * This type defines the signature for internal functions responsible for
 * processing (compressing or transforming) a specific chunk of input data.
 *
 * @param ctx     Pointer to the compression context containing state and
 * configuration.
 * @param in      Pointer to the input data buffer.
 * @param in_sz   Size of the input data in bytes.
 * @param out     Pointer to the output buffer where processed data will be
 * written.
 * @param out_cap Capacity of the output buffer in bytes.
 *
 * @return The number of bytes written to the output buffer on success, or a
 * negative error code on failure.
 */
typedef int (*zxc_chunk_processor_t)(zxc_cctx_t* RESTRICT ctx, const uint8_t* RESTRICT in,
                                     const size_t in_sz, uint8_t* RESTRICT out,
                                     const size_t out_cap);

/**
 * @struct zxc_stream_ctx_t
 * @brief The main context structure managing the streaming
 * compression/decompression state.
 *
 * This structure orchestrates the producer-consumer workflow. It manages the
 * ring buffer of jobs, the worker queue, synchronization primitives (mutexes
 * and condition variables), and configuration settings for the compression
 * algorithm.
 *
 * @var zxc_stream_ctx_t::jobs
 *      Array of job structures acting as the ring buffer.
 * @var zxc_stream_ctx_t::ring_size
 *      The total number of slots in the jobs array.
 * @var zxc_stream_ctx_t::worker_queue
 *      A circular queue containing indices of jobs ready to be picked up by
 * worker threads.
 * @var zxc_stream_ctx_t::wq_head
 *      Index of the head of the worker queue (where workers take jobs).
 * @var zxc_stream_ctx_t::wq_tail
 *      Index of the tail of the worker queue (where the writer adds jobs).
 * @var zxc_stream_ctx_t::wq_count
 *      Current number of items in the worker queue.
 * @var zxc_stream_ctx_t::lock
 *      Mutex used to protect access to shared resources (queue indices, status
 * changes).
 * @var zxc_stream_ctx_t::cond_reader
 *      Condition variable to signal the output thread (reader) that processed
 * data is available.
 * @var zxc_stream_ctx_t::cond_worker
 *      Condition variable to signal worker threads that new work is available.
 * @var zxc_stream_ctx_t::cond_writer
 *      Condition variable to signal the input thread (writer) that job slots
 * are free.
 * @var zxc_stream_ctx_t::shutdown_workers
 *      Flag indicating that worker threads should terminate.
 * @var zxc_stream_ctx_t::compression_mode
 *      Indicates the operation mode (e.g., compression or decompression).
 * @var zxc_stream_ctx_t::io_error
 *      Atomic flag to signal if an I/O error occurred during processing.
 * @var zxc_stream_ctx_t::processor
 *      Function pointer or object responsible for the actual chunk processing
 * logic.
 * @var zxc_stream_ctx_t::write_idx
 *      The index of the next job slot to be written to by the main thread.
 * @var zxc_stream_ctx_t::compression_level
 *      The configured level of compression (trading off speed vs. ratio).
 * @var zxc_stream_ctx_t::chunk_size
 *      The size of each data chunk to be processed.
 * @var zxc_stream_ctx_t::checksum_enabled
 *      Flag indicating whether checksum verification/generation is active.
 * @var zxc_stream_ctx_t::file_has_checksum
 *     Flag indicating whether the input file includes checksums.
 * @var zxc_stream_ctx_t::progress_cb
 *     Optional callback function for reporting progress during processing.
 * @var zxc_stream_ctx_t::progress_user_data
 *    User data pointer to be passed to the progress callback function.
 * @var zxc_stream_ctx_t::total_input_bytes
 *     Total size of the input data in bytes, used for progress tracking.
 * @var zxc_stream_ctx_t::dict
 *     Pointer to the optional dictionary buffer used to prime
 *     compression/decompression, NULL when no dictionary is in use.
 * @var zxc_stream_ctx_t::dict_size
 *     Size of the dictionary in bytes, 0 when no dictionary is in use.
 * @var zxc_stream_ctx_t::dict_huf
 *     Shared dictionary literal Huffman table (128-byte packed code-lengths
 *     header), NULL when absent.
 */
typedef struct {
    zxc_stream_job_t* jobs;
    size_t ring_size;
    int* worker_queue;
    int wq_head;
    int wq_tail;
    int wq_count;
    pthread_mutex_t lock;
    pthread_cond_t cond_reader;
    pthread_cond_t cond_worker;
    pthread_cond_t cond_writer;
    int shutdown_workers;
    int compression_mode;
    ZXC_ATOMIC int io_error;
    zxc_chunk_processor_t processor;
    int write_idx;
    int compression_level;
    size_t chunk_size;
    int checksum_enabled;
    int file_has_checksum;
    zxc_progress_callback_t progress_cb;
    void* progress_user_data;
    uint64_t total_input_bytes;
    const uint8_t* dict;
    size_t dict_size;
    const uint8_t* dict_huf; /**< Shared dictionary literal table (128-byte packed
                                  code-lengths header), NULL when absent. */
} zxc_stream_ctx_t;

/**
 * @struct writer_args_t
 * @brief Structure containing arguments for the writer callback function.
 *
 * This structure is used to pass necessary context and state information
 * to the function responsible for writing compressed or decompressed data
 * to a file stream.
 *
 * @var writer_args_t::ctx
 * Pointer to the ZXC stream context, holding the state of the
 * compression/decompression stream.
 *
 * @var writer_args_t::f
 * Pointer to the output file stream where data will be written.
 *
 * @var writer_args_t::total_bytes
 * Accumulator for the total number of bytes written to the file so far.
 *
 * @var writer_args_t::global_hash
 * The global hash accumulated during processing.
 *
 * @var writer_args_t::bytes_processed
 * The number of bytes processed so far, used for progress reporting.
 *
 * @var writer_args_t::seek_comp
 * Array of compressed block sizes for seek table construction.
 *
 * @var writer_args_t::seek_count
 * Number of entries in the seek table.
 *
 * @var writer_args_t::seek_cap
 * Capacity of the seek table array.
 */
typedef struct {
    zxc_stream_ctx_t* ctx;
    FILE* f;
    int64_t total_bytes;
    uint32_t global_hash;
    uint64_t bytes_processed;  // For progress callback
    uint32_t* seek_comp;
    uint32_t seek_count;
    uint32_t seek_cap;
} writer_args_t;

/**
 * @brief Worker thread function for parallel stream processing.
 *
 * This function serves as the entry point for worker threads in the ZXC
 * streaming compression/decompression context. It continuously retrieves jobs
 * from a shared work queue, processes them using a thread-local compression
 * context (`zxc_cctx_t`), and signals the writer thread upon completion.
 *
 * **Worker Lifecycle & Synchronization:**
 * 1. **Initialization:** Allocates a thread-local `zxc_cctx_t` to avoid lock
 * contention during compression/decompression.
 * 2. **Wait Loop:** Uses `pthread_cond_wait` on `cond_worker` to sleep until a
 * job is available in the `worker_queue`.
 * 3. **Job Retrieval:** Dequeues a job ID from the ring buffer. The
 * `worker_queue` acts as a load balancer.
 * 4. **Processing:** Calls `ctx->processor` (the compression/decompression
 * function) on the job's data. This is the CPU-intensive part and runs in
 * parallel.
 * 5. **Completion:** Updates `job->status` to `JOB_STATUS_PROCESSED`.
 * 6. **Signaling:** If the processed job is the *next* one expected by the
 * writer
 *    (`jid == ctx->write_idx`), it signals `cond_writer`. This optimization
 * prevents unnecessary wake-ups of the writer thread for out-of-order
 * completions.
 *
 * @param[in] arg A pointer to the shared stream context (`zxc_stream_ctx_t`).
 * @return Always returns NULL.
 */
static void* zxc_stream_worker(void* arg) {
    zxc_stream_ctx_t* const ctx = (zxc_stream_ctx_t*)arg;
    zxc_cctx_t cctx;

    const int unified_chk = (ctx->compression_mode == 1)
                                ? ctx->checksum_enabled
                                : (ctx->file_has_checksum && ctx->checksum_enabled);

    const size_t eff_chunk = (ctx->dict_size > 0 && ctx->compression_mode == 1)
                                 ? zxc_block_size_ceil(ctx->dict_size + ctx->chunk_size)
                                 : ctx->chunk_size;
    if (UNLIKELY(zxc_cctx_init(&cctx, eff_chunk, ctx->compression_mode, ctx->compression_level,
                               unified_chk, ctx->dict_size) != ZXC_OK ||
                 zxc_cctx_attach_dict_huf(&cctx, ctx->dict_huf) != ZXC_OK)) {
        // LCOV_EXCL_START
        zxc_cctx_free(&cctx);
        pthread_mutex_lock(&ctx->lock);
        ctx->io_error = 1;
        pthread_cond_broadcast(&ctx->cond_writer);
        pthread_cond_broadcast(&ctx->cond_reader);
        pthread_mutex_unlock(&ctx->lock);
        return NULL;
        // LCOV_EXCL_STOP
    }

    cctx.compression_level = ctx->compression_level;

    /* Per-worker dict buffer for assembling [dict | block_data] */
    const size_t dsz = ctx->dict_size;
    uint8_t* const dict_work = cctx.dict_buffer;
    if (dict_work) ZXC_MEMCPY(dict_work, ctx->dict, dsz);

    while (1) {
        zxc_stream_job_t* job = NULL;
        pthread_mutex_lock(&ctx->lock);
        while (ctx->wq_count == 0 && !ctx->shutdown_workers) {
            pthread_cond_wait(&ctx->cond_worker, &ctx->lock);
        }
        if (ctx->shutdown_workers && ctx->wq_count == 0) {
            pthread_mutex_unlock(&ctx->lock);
            break;
        }
        const int jid = ctx->worker_queue[ctx->wq_tail];
        ctx->wq_tail = (ctx->wq_tail + 1) % ctx->ring_size;
        ctx->wq_count--;
        job = &ctx->jobs[jid];
        pthread_mutex_unlock(&ctx->lock);

        int res;
        if (dict_work && ctx->compression_mode == 1) {
            ZXC_MEMCPY(dict_work + dsz, job->in_buf, job->in_sz);
            res = ctx->processor(&cctx, dict_work, dsz + job->in_sz, job->out_buf, job->out_cap);
        } else if (dict_work && ctx->compression_mode == 0) {
            res = ctx->processor(&cctx, job->in_buf, job->in_sz, dict_work + dsz,
                                 ctx->chunk_size + ZXC_DECOMPRESS_TAIL_PAD);
            if (LIKELY(res > 0)) ZXC_MEMCPY(job->out_buf, dict_work + dsz, (size_t)res);
        } else {
            res = ctx->processor(&cctx, job->in_buf, job->in_sz, job->out_buf, job->out_cap);
        }

        pthread_mutex_lock(&ctx->lock);
        job->result_sz = UNLIKELY(res < 0) ? 0 : (size_t)res;
        job->status = JOB_STATUS_PROCESSED;
        if (UNLIKELY(res < 0)) {
            ctx->io_error = 1;
            pthread_cond_broadcast(&ctx->cond_writer);
            pthread_cond_broadcast(&ctx->cond_reader);
        } else if (jid == ctx->write_idx) {
            pthread_cond_signal(&ctx->cond_writer);
        }
        pthread_mutex_unlock(&ctx->lock);
    }
    zxc_cctx_free(&cctx);
    return NULL;
}

/**
 * @brief Asynchronous writer thread function.
 *
 * This function runs as a separate thread responsible for writing processed
 * data chunks to the output file. It operates on a ring buffer of jobs shared
 * with the reader and worker threads.
 *
 * **Ordering Enforcement:**
 * The writer MUST write blocks in the exact order they were read. Even if
 * worker threads finish jobs out of order (e.g., job 2 finishes before job 1),
 * the writer waits for `ctx->write_idx` (job 1) to be `JOB_STATUS_PROCESSED`.
 *
 * **Workflow:**
 * 1. **Wait:** Sleeps on `cond_writer` until the job at `ctx->write_idx` is
 * ready.
 * 2. **Write:** Writes the `out_buf` to the file.
 * 3. **Release:** Sets the job status to `JOB_STATUS_FREE` and signals
 * `cond_reader`, allowing the main thread to reuse this slot for new input.
 * 4. **Advance:** Increments `ctx->write_idx` to wait for the next sequential
 * block.
 *
 * @param[in] arg Pointer to a `writer_args_t` structure containing the stream
 * context, the output file handle, and a counter for total bytes written.
 * @return Always returns NULL.
 */
static void* zxc_async_writer(void* arg) {
    writer_args_t* const args = (writer_args_t*)arg;
    zxc_stream_ctx_t* const ctx = args->ctx;
    while (1) {
        zxc_stream_job_t* const job = &ctx->jobs[ctx->write_idx];
        pthread_mutex_lock(&ctx->lock);
        while (job->status != JOB_STATUS_PROCESSED && !ctx->io_error)
            pthread_cond_wait(&ctx->cond_writer, &ctx->lock);

        const size_t result_sz = job->result_sz;
        const size_t in_sz = job->in_sz;
        pthread_mutex_unlock(&ctx->lock);

        if (result_sz == (size_t)-1) break;

        if (args->f && result_sz > 0) {
            if (fwrite(job->out_buf, 1, result_sz, args->f) != result_sz) {
                pthread_mutex_lock(&ctx->lock);
                ctx->io_error = 1;
                pthread_cond_signal(&ctx->cond_reader);
                pthread_mutex_unlock(&ctx->lock);
            } else if (ctx->checksum_enabled && ctx->compression_mode == 1) {
                // Update Global Hash (Rotation + XOR)
                if (LIKELY(result_sz >= ZXC_GLOBAL_CHECKSUM_SIZE)) {
                    uint32_t block_hash =
                        zxc_le32(job->out_buf + result_sz - ZXC_GLOBAL_CHECKSUM_SIZE);
                    args->global_hash = zxc_hash_combine_rotate(args->global_hash, block_hash);
                }
            }
        }
        if (UNLIKELY(ctx->io_error)) {
            pthread_mutex_lock(&ctx->lock);
            job->status = JOB_STATUS_FREE;
            pthread_cond_signal(&ctx->cond_reader);
            pthread_mutex_unlock(&ctx->lock);
            break;
        }
        args->total_bytes += (int64_t)result_sz;

        /* Seekable: record compressed block size */
        if (args->seek_comp && ctx->compression_mode == 1) {
            if (UNLIKELY(args->seek_count >= args->seek_cap)) {
                args->seek_cap = args->seek_cap * 2;
                uint32_t* nc =
                    (uint32_t*)ZXC_REALLOC(args->seek_comp, args->seek_cap * sizeof(uint32_t));
                // LCOV_EXCL_START
                if (UNLIKELY(!nc)) {
                    pthread_mutex_lock(&ctx->lock);
                    ctx->io_error = 1;
                    job->status = JOB_STATUS_FREE;
                    pthread_cond_signal(&ctx->cond_reader);
                    pthread_mutex_unlock(&ctx->lock);
                    break;
                }
                // LCOV_EXCL_STOP
                args->seek_comp = nc;
            }
            args->seek_comp[args->seek_count++] = (uint32_t)result_sz;
        }

        // Update progress callback
        if (ctx->progress_cb) {
            // LCOV_EXCL_START
            args->bytes_processed += ctx->compression_mode == 1 ? in_sz : result_sz;
            ctx->progress_cb(args->bytes_processed, ctx->total_input_bytes,
                             ctx->progress_user_data);
            // LCOV_EXCL_STOP
        }

        pthread_mutex_lock(&ctx->lock);
        job->status = JOB_STATUS_FREE;
        ctx->write_idx = (ctx->write_idx + 1) % ctx->ring_size;
        pthread_cond_signal(&ctx->cond_reader);
        pthread_mutex_unlock(&ctx->lock);
    }
    return NULL;
}

/**
 * @brief Orchestrates the multithreaded streaming compression or decompression
 * engine.
 *
 * This function initializes the stream context, allocates the necessary ring
 * buffer memory for jobs and I/O buffers, and spawns the worker threads and the
 * asynchronous writer thread. It acts as the main "producer" (reader) loop.
 *
 * **Architecture: Producer-Consumer with Ring Buffer**
 * - **Ring Buffer:** A fixed-size array of `zxc_stream_job_t` structures.
 * - **Producer (Main Thread):** Reads chunks from `f_in` and fills "Free" slots
 *   in the ring buffer. It blocks if no slots are free (backpressure).
 * - **Workers:** Pick up "Filled" jobs from a queue, process them, and mark
 * them as "Processed".
 * - **Consumer (Writer Thread):** Waits for the *next sequential* job to be
 *   "Processed", writes it to `f_out`, and marks the slot as "Free".
 *
 * **Double-Buffering & Zero-Copy:**
 * We allocate `alloc_in` and `alloc_out` buffers for each job. The reader reads
 * directly into `in_buf`, and the writer writes directly from `out_buf`,
 * minimizing memory copies.
 *
 * @param[in]  f_in             Input file stream (source).
 * @param[out] f_out            Output file stream (destination).
 * @param[in]  n_threads        Worker thread count; 0 or less auto-detects the
 *                              number of online processors.
 * @param[in]  mode             1 for compression, 0 for decompression.
 * @param[in]  level            Compression level (compression mode only).
 * @param[in]  block_size       Block size in bytes (compression mode).
 * @param[in]  checksum_enabled Non-zero to generate / verify checksums.
 * @param[in]  seekable         Non-zero to emit a seek table (compression mode).
 * @param[in]  func             Chunk processor (compression or decompression).
 * @param[in]  progress_cb      Optional progress callback, or NULL.
 * @param[in]  user_data        Opaque pointer passed to @p progress_cb.
 * @param[in]  dict             Optional dictionary content, or NULL.
 * @param[in]  dict_size        Dictionary length in bytes (0 if none).
 * @param[in]  dict_huf         Optional shared literal Huffman table, or NULL.
 * @return Total bytes written to the output on success, or a negative
 *         @ref zxc_error_t code.
 */
static int64_t zxc_stream_engine_run(FILE* f_in, FILE* f_out, const int n_threads, const int mode,
                                     const int level, const size_t block_size,
                                     const int checksum_enabled, const int seekable,
                                     zxc_chunk_processor_t func,
                                     zxc_progress_callback_t progress_cb, void* user_data,
                                     const uint8_t* dict, const size_t dict_size,
                                     const uint8_t* dict_huf) {
    zxc_stream_ctx_t ctx;
    ZXC_MEMSET(&ctx, 0, sizeof(ctx));

    size_t runtime_chunk_sz = (block_size > 0) ? block_size : ZXC_BLOCK_SIZE_DEFAULT;
    int file_has_chk = 0;

    // Try to get input file size for progress tracking (compression mode only)
    // For decompression, the CLI precomputes the size and passes it via user_data
    uint64_t total_file_size = 0;
    if (mode == 1 && progress_cb) {
        // LCOV_EXCL_START
        const long long saved_pos = ftello(f_in);
        if (saved_pos >= 0 && fseeko(f_in, 0, SEEK_END) == 0) {
            const long long size = ftello(f_in);
            if (size > 0) total_file_size = (uint64_t)size;
            fseeko(f_in, saved_pos, SEEK_SET);
        }
        // LCOV_EXCL_STOP
    }

    if (mode == 0) {
        // Decompression Mode: Read and validate file header
        uint8_t h[ZXC_FILE_HEADER_SIZE];
        uint32_t header_dict_id = 0;
        if (UNLIKELY(fread(h, 1, ZXC_FILE_HEADER_SIZE, f_in) != ZXC_FILE_HEADER_SIZE ||
                     zxc_read_file_header(h, ZXC_FILE_HEADER_SIZE, &runtime_chunk_sz, &file_has_chk,
                                          &header_dict_id) != ZXC_OK))
            return ZXC_ERROR_BAD_HEADER;

        if (header_dict_id != 0) {
            if (UNLIKELY(!dict || dict_size == 0)) return ZXC_ERROR_DICT_REQUIRED;
            if (UNLIKELY(zxc_dict_id(dict, dict_size, dict_huf) != header_dict_id))
                return ZXC_ERROR_DICT_MISMATCH;
        }
    }

    int num_threads = (n_threads > 0) ? n_threads : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads > ZXC_MAX_THREADS) num_threads = ZXC_MAX_THREADS;
    // Reserve 1 thread for Writer/Reader overhead if possible
    const int num_workers = (num_threads > 1) ? num_threads - 1 : 1;

    ctx.compression_mode = mode;
    ctx.processor = func;
    ctx.io_error = 0;
    ctx.compression_level = level;
    ctx.ring_size = (size_t)num_workers * 4U;
    ctx.chunk_size = runtime_chunk_sz;
    ctx.checksum_enabled = checksum_enabled;
    ctx.file_has_checksum = mode == 1 ? checksum_enabled : file_has_chk;
    ctx.progress_cb = progress_cb;
    ctx.progress_user_data = user_data;
    ctx.total_input_bytes = total_file_size;
    ctx.dict = dict;
    ctx.dict_size = dict_size;
    ctx.dict_huf = dict_huf;

    uint32_t d_global_hash = 0;

    const uint64_t max_out = zxc_compress_bound(runtime_chunk_sz);
    const size_t raw_alloc_in = (size_t)((mode ? runtime_chunk_sz : max_out) + ZXC_PAD_SIZE);
    const size_t alloc_in = (raw_alloc_in + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;

    const size_t raw_alloc_out =
        (size_t)((mode ? max_out : runtime_chunk_sz + ZXC_DECOMPRESS_TAIL_PAD) + ZXC_PAD_SIZE);
    const size_t alloc_out = (raw_alloc_out + ZXC_ALIGNMENT_MASK) & ~ZXC_ALIGNMENT_MASK;

    const size_t per_job_sz = sizeof(zxc_stream_job_t) + sizeof(int) + alloc_in + alloc_out;
    const size_t alloc_size = ctx.ring_size * per_job_sz;
    uint8_t* const mem_block = ZXC_ALIGNED_MALLOC(alloc_size, ZXC_CACHE_LINE_SIZE);
    if (UNLIKELY(!mem_block || per_job_sz > SIZE_MAX / ctx.ring_size)) {
        // LCOV_EXCL_START
        ZXC_ALIGNED_FREE(mem_block);
        return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
    }

    uint8_t* ptr = mem_block;
    ctx.jobs = (zxc_stream_job_t*)ptr;
    ptr += ctx.ring_size * sizeof(zxc_stream_job_t);
    ctx.worker_queue = (int*)ptr;
    ptr += ctx.ring_size * sizeof(int);
    uint8_t* buf_in = ptr;
    ptr += ctx.ring_size * alloc_in;
    uint8_t* buf_out = ptr;

    ZXC_MEMSET(mem_block, 0, alloc_size);

    for (size_t i = 0; i < ctx.ring_size; i++) {
        ctx.jobs[i].job_id = (int)i;
        ctx.jobs[i].status = JOB_STATUS_FREE;
        ctx.jobs[i].in_buf = buf_in + (i * alloc_in);
        ctx.jobs[i].in_cap = alloc_in - ZXC_PAD_SIZE;
        ctx.jobs[i].in_sz = 0;
        ctx.jobs[i].out_buf = buf_out + (i * alloc_out);
        ctx.jobs[i].out_cap = alloc_out - ZXC_PAD_SIZE;
        ctx.jobs[i].result_sz = 0;
    }

    pthread_mutex_init(&ctx.lock, NULL);
    pthread_cond_init(&ctx.cond_reader, NULL);
    pthread_cond_init(&ctx.cond_worker, NULL);
    pthread_cond_init(&ctx.cond_writer, NULL);

    pthread_t* const workers = ZXC_MALLOC((size_t)num_workers * sizeof(pthread_t));
    if (UNLIKELY(!workers)) {
        // LCOV_EXCL_START
        ZXC_ALIGNED_FREE(mem_block);
        return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
    }
    int started_workers = 0;
    for (int i = 0; i < num_workers; i++) {
        if (UNLIKELY(pthread_create(&workers[i], NULL, zxc_stream_worker, &ctx) != 0)) break;
        started_workers++;
    }
    if (UNLIKELY(started_workers == 0)) {
        // LCOV_EXCL_START
        pthread_cond_destroy(&ctx.cond_writer);
        pthread_cond_destroy(&ctx.cond_worker);
        pthread_cond_destroy(&ctx.cond_reader);
        pthread_mutex_destroy(&ctx.lock);
        ZXC_FREE(workers);
        ZXC_ALIGNED_FREE(mem_block);
        return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
    }

    writer_args_t w_args = {&ctx, f_out, 0, 0, 0, NULL, 0, 0};

    /* Seekable: allocate initial block-size tracking array */
    if (mode == 1 && seekable) {
        w_args.seek_cap = 64;
        w_args.seek_comp = (uint32_t*)ZXC_MALLOC(w_args.seek_cap * sizeof(uint32_t));
        // LCOV_EXCL_START
        if (UNLIKELY(!w_args.seek_comp)) {
            pthread_mutex_lock(&ctx.lock);
            ctx.shutdown_workers = 1;
            pthread_cond_broadcast(&ctx.cond_worker);
            pthread_mutex_unlock(&ctx.lock);
            for (int i = 0; i < started_workers; i++) pthread_join(workers[i], NULL);
            pthread_cond_destroy(&ctx.cond_writer);
            pthread_cond_destroy(&ctx.cond_worker);
            pthread_cond_destroy(&ctx.cond_reader);
            pthread_mutex_destroy(&ctx.lock);
            ZXC_FREE(workers);
            ZXC_ALIGNED_FREE(mem_block);
            return ZXC_ERROR_MEMORY;
        }
        // LCOV_EXCL_STOP
    }

    if (mode == 1 && f_out) {
        uint8_t h[ZXC_FILE_HEADER_SIZE];
        zxc_write_file_header(h, ZXC_FILE_HEADER_SIZE, runtime_chunk_sz, checksum_enabled,
                              (dict && dict_size) ? zxc_dict_id(dict, dict_size, dict_huf) : 0);
        if (UNLIKELY(fwrite(h, 1, ZXC_FILE_HEADER_SIZE, f_out) != ZXC_FILE_HEADER_SIZE))
            ctx.io_error = 1;

        w_args.total_bytes = ZXC_FILE_HEADER_SIZE;
    }
    pthread_t writer_th;
    if (UNLIKELY(pthread_create(&writer_th, NULL, zxc_async_writer, &w_args) != 0)) {
        // LCOV_EXCL_START
        pthread_mutex_lock(&ctx.lock);
        ctx.shutdown_workers = 1;
        pthread_cond_broadcast(&ctx.cond_worker);
        pthread_mutex_unlock(&ctx.lock);
        for (int i = 0; i < started_workers; i++) pthread_join(workers[i], NULL);
        pthread_cond_destroy(&ctx.cond_writer);
        pthread_cond_destroy(&ctx.cond_worker);
        pthread_cond_destroy(&ctx.cond_reader);
        pthread_mutex_destroy(&ctx.lock);
        ZXC_FREE(workers);
        ZXC_ALIGNED_FREE(mem_block);
        return ZXC_ERROR_MEMORY;
        // LCOV_EXCL_STOP
    }

    int read_idx = 0;
    int read_eof = 0;
    uint64_t total_src_bytes = 0;

    // Reader Loop: Reads from file, prepares jobs, pushes to worker queue.
    while (!read_eof && !ctx.io_error) {
        zxc_stream_job_t* const job = &ctx.jobs[read_idx];
        pthread_mutex_lock(&ctx.lock);
        while (job->status != JOB_STATUS_FREE && !ctx.io_error)
            pthread_cond_wait(&ctx.cond_reader, &ctx.lock);
        pthread_mutex_unlock(&ctx.lock);

        if (UNLIKELY(ctx.io_error)) break;

        size_t read_sz = 0;
        if (mode == 1) {
            read_sz = fread(job->in_buf, 1, runtime_chunk_sz, f_in);
            total_src_bytes += read_sz;
            if (UNLIKELY(read_sz == 0)) read_eof = 1;
        } else {
            uint8_t bh_buf[ZXC_BLOCK_HEADER_SIZE];
            size_t h_read = fread(bh_buf, 1, ZXC_BLOCK_HEADER_SIZE, f_in);
            if (UNLIKELY(h_read < ZXC_BLOCK_HEADER_SIZE)) {
                read_eof = 1;
            } else {
                zxc_block_header_t bh;
                if (UNLIKELY(zxc_read_block_header(bh_buf, ZXC_BLOCK_HEADER_SIZE, &bh) != ZXC_OK)) {
                    read_eof = 1;
                    goto _job_prepared;
                }

                if (bh.block_type == ZXC_BLOCK_EOF) {
                    if (UNLIKELY(bh.comp_size != 0)) {
                        ctx.io_error = 1;
                        goto _job_prepared;
                    }
                    read_eof = 1;
                    read_sz = 0;
                    goto _job_prepared;
                }

                const int has_crc = ctx.file_has_checksum;
                const size_t checksum_sz = (has_crc ? ZXC_BLOCK_CHECKSUM_SIZE : 0);
                const size_t body_total = bh.comp_size + checksum_sz;
                const size_t total_len = ZXC_BLOCK_HEADER_SIZE + body_total;

                if (UNLIKELY(total_len > job->in_cap)) {
                    ctx.io_error = 1;
                    break;
                }

                ZXC_MEMCPY(job->in_buf, bh_buf, ZXC_BLOCK_HEADER_SIZE);

                // Single fread for body + checksum (reduces syscalls)
                const size_t body_read =
                    fread(job->in_buf + ZXC_BLOCK_HEADER_SIZE, 1, body_total, f_in);

                if (UNLIKELY(body_read != body_total)) {
                    ctx.io_error = 1;
                    break;
                } else if (has_crc) {
                    // Update Global Hash for Decompression
                    const uint32_t b_crc =
                        zxc_le32(job->in_buf + ZXC_BLOCK_HEADER_SIZE + bh.comp_size);
                    d_global_hash = zxc_hash_combine_rotate(d_global_hash, b_crc);
                }
                read_sz = ZXC_BLOCK_HEADER_SIZE + body_read;
            }
        }
    _job_prepared:
        if (UNLIKELY(read_eof && read_sz == 0)) break;

        job->in_sz = read_sz;
        pthread_mutex_lock(&ctx.lock);
        job->status = JOB_STATUS_FILLED;
        ctx.worker_queue[ctx.wq_head] = read_idx;
        ctx.wq_head = (ctx.wq_head + 1) % ctx.ring_size;
        ctx.wq_count++;
        read_idx = (read_idx + 1) % ctx.ring_size;
        pthread_cond_signal(&ctx.cond_worker);
        pthread_mutex_unlock(&ctx.lock);

        if (UNLIKELY(read_sz < runtime_chunk_sz && mode == 1)) read_eof = 1;
    }

    zxc_stream_job_t* const end_job = &ctx.jobs[read_idx];
    pthread_mutex_lock(&ctx.lock);
    while (end_job->status != JOB_STATUS_FREE && !ctx.io_error)
        pthread_cond_wait(&ctx.cond_reader, &ctx.lock);
    end_job->result_sz = (size_t)-1;
    end_job->status = JOB_STATUS_PROCESSED;
    pthread_cond_broadcast(&ctx.cond_writer);
    pthread_mutex_unlock(&ctx.lock);

    pthread_join(writer_th, NULL);
    pthread_mutex_lock(&ctx.lock);
    ctx.shutdown_workers = 1;
    pthread_cond_broadcast(&ctx.cond_worker);
    pthread_mutex_unlock(&ctx.lock);
    for (int i = 0; i < started_workers; i++) pthread_join(workers[i], NULL);

    pthread_cond_destroy(&ctx.cond_writer);
    pthread_cond_destroy(&ctx.cond_worker);
    pthread_cond_destroy(&ctx.cond_reader);
    pthread_mutex_destroy(&ctx.lock);

    // Write EOF Block + optional Seek Table + Footer if compression and no error
    if (mode == 1 && !ctx.io_error && w_args.total_bytes >= 0) {
        /* EOF block */
        uint8_t eof_buf[ZXC_BLOCK_HEADER_SIZE];
        const zxc_block_header_t eof_bh = {
            .block_type = ZXC_BLOCK_EOF, .block_flags = 0, .reserved = 0, .comp_size = 0};
        zxc_write_block_header(eof_buf, ZXC_BLOCK_HEADER_SIZE, &eof_bh);
        if (UNLIKELY(f_out &&
                     fwrite(eof_buf, 1, ZXC_BLOCK_HEADER_SIZE, f_out) != ZXC_BLOCK_HEADER_SIZE))
            ctx.io_error = 1;
        else
            w_args.total_bytes += ZXC_BLOCK_HEADER_SIZE;

        /* Seekable: write SEK block between EOF and footer */
        if (!ctx.io_error && w_args.seek_comp && w_args.seek_count > 0) {
            const size_t st_size = zxc_seek_table_size(w_args.seek_count);
            uint8_t* const st_buf = (uint8_t*)ZXC_MALLOC(st_size);
            if (st_buf) {
                const int64_t st_val =
                    zxc_write_seek_table(st_buf, st_size, w_args.seek_comp, w_args.seek_count);
                if (st_val > 0 && f_out &&
                    fwrite(st_buf, 1, (size_t)st_val, f_out) == (size_t)st_val)
                    w_args.total_bytes += st_val;
                ZXC_FREE(st_buf);
            }
        }

        /* Footer */
        uint8_t footer_buf[ZXC_FILE_FOOTER_SIZE];
        zxc_write_file_footer(footer_buf, ZXC_FILE_FOOTER_SIZE, total_src_bytes, w_args.global_hash,
                              checksum_enabled);
        if (UNLIKELY(f_out &&
                     fwrite(footer_buf, 1, ZXC_FILE_FOOTER_SIZE, f_out) != ZXC_FILE_FOOTER_SIZE))
            ctx.io_error = 1;
        else
            w_args.total_bytes += ZXC_FILE_FOOTER_SIZE;
    } else if (mode == 0 && !ctx.io_error) {
        /*
         * After the EOF block, the stream may contain:
         *   (a) [FOOTER 12B]                  - no seekable table
         *   (b) [SEK header 8B] [payload] [FOOTER 12B] - seekable archive
         */
        uint8_t peek_buf[ZXC_BLOCK_HEADER_SIZE];
        uint8_t footer[ZXC_FILE_FOOTER_SIZE];

        if (UNLIKELY(fread(peek_buf, 1, ZXC_BLOCK_HEADER_SIZE, f_in) != ZXC_BLOCK_HEADER_SIZE)) {
            ctx.io_error = 1;
        } else {
            zxc_block_header_t peek_bh;
            const int is_sek =
                (zxc_read_block_header(peek_buf, ZXC_BLOCK_HEADER_SIZE, &peek_bh) == ZXC_OK &&
                 peek_bh.block_type == ZXC_BLOCK_SEK);

            if (is_sek) {
                /* Drain the SEK payload (read + discard) */
                size_t remaining = (size_t)peek_bh.comp_size;
                uint8_t discard[512];
                while (remaining > 0 && !ctx.io_error) {
                    const size_t chunk = remaining < sizeof(discard) ? remaining : sizeof(discard);
                    if (UNLIKELY(fread(discard, 1, chunk, f_in) != chunk)) ctx.io_error = 1;
                    remaining -= chunk;
                }
                /* Read full 12-byte footer */
                if (!ctx.io_error &&
                    UNLIKELY(fread(footer, 1, ZXC_FILE_FOOTER_SIZE, f_in) != ZXC_FILE_FOOTER_SIZE))
                    ctx.io_error = 1;
            } else {
                /* peek_buf contains the first 8 bytes of the 12-byte footer.
                 * Read the remaining 4 bytes and assemble. */
                ZXC_MEMCPY(footer, peek_buf, ZXC_BLOCK_HEADER_SIZE);
                const size_t tail = ZXC_FILE_FOOTER_SIZE - ZXC_BLOCK_HEADER_SIZE; /* 4 */
                if (UNLIKELY(fread(footer + ZXC_BLOCK_HEADER_SIZE, 1, tail, f_in) != tail))
                    ctx.io_error = 1;
            }
        }

        /* Verify Footer Content: Source Size and Global Checksum */
        if (!ctx.io_error) {
            int valid = (zxc_le64(footer) == (uint64_t)w_args.total_bytes);
            if (valid && checksum_enabled && ctx.file_has_checksum)
                valid = (zxc_le32(footer + sizeof(uint64_t)) == d_global_hash);
            if (UNLIKELY(!valid)) ctx.io_error = 1;
        }
    }

    ZXC_FREE(w_args.seek_comp);
    ZXC_FREE(workers);
    ZXC_ALIGNED_FREE(mem_block);

    if (UNLIKELY(ctx.io_error)) return ZXC_ERROR_IO;

    return w_args.total_bytes;
}

/**
 * @brief Compresses a @c FILE* stream to another @c FILE* stream.
 *
 * Public API; full contract in @c zxc_stream.h. Resolves the options (threads,
 * level, block size, checksums, seekable, dictionary) with their defaults, then
 * drives @ref zxc_stream_engine_run in compression mode with the
 * compress chunk processor.
 *
 * @param[in]  f_in   Input stream (must be non-NULL).
 * @param[out] f_out  Output stream (NULL performs a dry run / size estimate).
 * @param[in]  opts   Compression options, or NULL for all defaults.
 * @return Total bytes written on success, or a negative @ref zxc_error_t.
 */
int64_t zxc_stream_compress(FILE* f_in, FILE* f_out, const zxc_compress_opts_t* opts) {
    if (UNLIKELY(!f_in)) return ZXC_ERROR_NULL_INPUT;

    const int n_threads = opts ? opts->n_threads : 0;
    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const int seekable = opts ? opts->seekable : 0;
    const int level = (opts && opts->level > 0) ? opts->level : ZXC_LEVEL_DEFAULT;
    const size_t block_size =
        (opts && opts->block_size > 0) ? opts->block_size : ZXC_BLOCK_SIZE_DEFAULT;
    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = (opts && opts->dict) ? opts->dict_size : 0;
    zxc_progress_callback_t cb = opts ? opts->progress_cb : NULL;
    void* ud = opts ? opts->user_data : NULL;

    if (UNLIKELY(!zxc_validate_block_size(block_size))) return ZXC_ERROR_BAD_BLOCK_SIZE;
    if (UNLIKELY(dict_size > ZXC_DICT_SIZE_MAX)) return ZXC_ERROR_DICT_TOO_LARGE;

    const uint8_t* dict_huf = (opts && opts->dict) ? (const uint8_t*)opts->dict_huf : NULL;
    return zxc_stream_engine_run(f_in, f_out, n_threads, 1, level, block_size, checksum_enabled,
                                 seekable, zxc_compress_chunk_wrapper, cb, ud, dict, dict_size,
                                 dict_huf);
}

/**
 * @brief Decompresses a @c FILE* stream to another @c FILE* stream.
 *
 * Public API; full contract in @c zxc_stream.h. Resolves the options (threads,
 * checksums, dictionary), then drives @ref zxc_stream_engine_run in
 * decompression mode with the decompress chunk processor. The block size and
 * level are recovered from the archive header, not from @p opts.
 *
 * @param[in]  f_in   Input (compressed) stream (must be non-NULL).
 * @param[out] f_out  Output (decompressed) stream.
 * @param[in]  opts   Decompression options, or NULL for all defaults.
 * @return Total bytes written on success, or a negative @ref zxc_error_t.
 */
int64_t zxc_stream_decompress(FILE* f_in, FILE* f_out, const zxc_decompress_opts_t* opts) {
    if (UNLIKELY(!f_in)) return ZXC_ERROR_NULL_INPUT;

    const int n_threads = opts ? opts->n_threads : 0;
    const int checksum_enabled = opts ? opts->checksum_enabled : 0;
    const uint8_t* dict = opts ? (const uint8_t*)opts->dict : NULL;
    const size_t dict_size = (opts && opts->dict) ? opts->dict_size : 0;
    zxc_progress_callback_t cb = opts ? opts->progress_cb : NULL;
    void* ud = opts ? opts->user_data : NULL;

    const uint8_t* dict_huf = (opts && opts->dict) ? (const uint8_t*)opts->dict_huf : NULL;
    return zxc_stream_engine_run(f_in, f_out, n_threads, 0, 0, 0, checksum_enabled, 0,
                                 (zxc_chunk_processor_t)zxc_decompress_chunk_wrapper, cb, ud, dict,
                                 dict_size, dict_huf);
}

/**
 * @brief Reads the total decompressed size from an archive's footer.
 *
 * Public API; see @c zxc_stream.h. Validates the file magic, reads the 64-bit
 * decompressed-size field from the footer, and restores the caller's original
 * stream position before returning. Does not decompress any data.
 *
 * @param[in] f_in  Compressed stream (must be non-NULL and seekable).
 * @return Decompressed size in bytes, or a negative @ref zxc_error_t
 *         (@ref ZXC_ERROR_BAD_MAGIC, @ref ZXC_ERROR_SRC_TOO_SMALL,
 *         @ref ZXC_ERROR_IO).
 */
int64_t zxc_stream_get_decompressed_size(FILE* f_in) {
    if (UNLIKELY(!f_in)) return ZXC_ERROR_NULL_INPUT;

    const long long saved_pos = ftello(f_in);
    if (UNLIKELY(saved_pos < 0)) return ZXC_ERROR_IO;

    // Get file size
    if (fseeko(f_in, 0, SEEK_END) != 0) return ZXC_ERROR_IO;
    const long long file_size = ftello(f_in);
    if (UNLIKELY(file_size < (long long)(ZXC_FILE_HEADER_SIZE + ZXC_FILE_FOOTER_SIZE))) {
        fseeko(f_in, saved_pos, SEEK_SET);
        return ZXC_ERROR_SRC_TOO_SMALL;
    }

    uint8_t header[ZXC_FILE_HEADER_SIZE];
    if (UNLIKELY(fseeko(f_in, 0, SEEK_SET) != 0 ||
                 fread(header, 1, ZXC_FILE_HEADER_SIZE, f_in) != ZXC_FILE_HEADER_SIZE)) {
        fseeko(f_in, saved_pos, SEEK_SET);
        return ZXC_ERROR_IO;
    }

    if (UNLIKELY(zxc_le32(header) != ZXC_MAGIC_WORD)) {
        fseeko(f_in, saved_pos, SEEK_SET);
        return ZXC_ERROR_BAD_MAGIC;
    }

    uint8_t footer[ZXC_FILE_FOOTER_SIZE];
    if (UNLIKELY(fseeko(f_in, file_size - ZXC_FILE_FOOTER_SIZE, SEEK_SET) != 0 ||
                 fread(footer, 1, ZXC_FILE_FOOTER_SIZE, f_in) != ZXC_FILE_FOOTER_SIZE)) {
        fseeko(f_in, saved_pos, SEEK_SET);
        return ZXC_ERROR_IO;
    }

    fseeko(f_in, saved_pos, SEEK_SET);

    return (int64_t)zxc_le64(footer);
}

/*
 * ============================================================================
 * SEEKABLE FILE* WRAPPER
 * ============================================================================
 * Adapts a FILE* into a thread-safe zxc_reader_t (pread on POSIX, ReadFile +
 * OVERLAPPED on Windows) and delegates to zxc_seekable_open_reader.  Keeping
 * this entry point alongside the stream driver, rather than in the kernel-
 * safe zxc_seekable.c, means zxc_seekable.c stays freestanding.
 */

#if defined(_WIN32)
/** @brief Reader context for the Win32 @c FILE* adapter (OS file handle + size). */
typedef struct {
    HANDLE handle; /* OS handle from _get_osfhandle(_fileno(f)) */
    uint64_t size; /* total file size in bytes */
} zxc_stdio_ctx_t;

/**
 * @brief Thread-safe positioned read backing the seekable @c FILE* reader.
 *
 * Win32 implementation: a positioned @c ReadFile via @c OVERLAPPED, so
 * concurrent worker threads never race on a shared file cursor.
 *
 * @param[in]  vctx    @ref zxc_stdio_ctx_t carrying the file handle.
 * @param[out] dst     Destination buffer (at least @p len bytes).
 * @param[in]  len     Number of bytes to read.
 * @param[in]  offset  Absolute byte offset to read from.
 * @return @p len on a full read, otherwise @ref ZXC_ERROR_IO.
 */
// LCOV_EXCL_START - Windows I/O path, not reachable on POSIX CI
static int64_t zxc_stdio_read_at(void* vctx, void* dst, size_t len, uint64_t offset) {
    zxc_stdio_ctx_t* const ctx = (zxc_stdio_ctx_t*)vctx;
    OVERLAPPED ov;
    ZXC_MEMSET(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD)(offset & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD)(offset >> 32);
    DWORD bytes_read = 0;
    if (!ReadFile(ctx->handle, dst, (DWORD)len, &bytes_read, &ov)) return ZXC_ERROR_IO;
    return (bytes_read == (DWORD)len) ? (int64_t)len : ZXC_ERROR_IO;
}
// LCOV_EXCL_STOP

#else  /* POSIX */
/** @brief Reader context for the POSIX @c FILE* adapter (file descriptor + size). */
typedef struct {
    int fd;        /* descriptor from fileno(f) */
    uint64_t size; /* total file size in bytes */
} zxc_stdio_ctx_t;

/**
 * @brief Thread-safe positioned read backing the seekable @c FILE* reader.
 *
 * POSIX implementation: a single @c pread, which carries its own offset and so
 * is safe to call concurrently from multiple worker threads on one descriptor.
 *
 * @param[in]  vctx    @ref zxc_stdio_ctx_t carrying the file descriptor.
 * @param[out] dst     Destination buffer (at least @p len bytes).
 * @param[in]  len     Number of bytes to read.
 * @param[in]  offset  Absolute byte offset to read from.
 * @return @p len on a full read, otherwise @ref ZXC_ERROR_IO.
 */
static int64_t zxc_stdio_read_at(void* vctx, void* dst, size_t len, uint64_t offset) {
    zxc_stdio_ctx_t* const ctx = (zxc_stdio_ctx_t*)vctx;
    const ssize_t r = pread(ctx->fd, dst, len, (off_t)offset);
    return (r == (ssize_t)len) ? (int64_t)len : ZXC_ERROR_IO;
}
#endif /* _WIN32 */

/**
 * @brief Opens a seekable archive backed by an open @c FILE*.
 *
 * Public API; full contract in @c zxc_stream.h. Snapshots and restores the
 * file position, measures the file, wraps it in a thread-safe positioned
 * reader (@c pread on POSIX, @c ReadFile + @c OVERLAPPED on Windows), and
 * delegates to @ref zxc_seekable_open_reader. The reader context is heap-owned
 * and handed to the returned handle via @ref zxc_seekable_attach_owned_ctx, so
 * @ref zxc_seekable_free releases it.
 *
 * @param[in] f  Open, seekable file handle.
 * @return A handle to release with @ref zxc_seekable_free, or NULL on bad input,
 *         an I/O error, or a missing / malformed seek table.
 */
zxc_seekable* zxc_seekable_open_file(FILE* f) {
    if (UNLIKELY(!f)) return NULL;

    /* Snapshot the caller's file position so we can restore it. */
    const long long saved_pos = ftello(f);
    if (UNLIKELY(saved_pos < 0)) return NULL;  // LCOV_EXCL_LINE

    // LCOV_EXCL_START - ftello/fseeko failure paths not reachable in CI
    if (UNLIKELY(fseeko(f, 0, SEEK_END) != 0)) return NULL;
    const long long file_size = ftello(f);
    (void)fseeko(f, saved_pos, SEEK_SET);
    if (UNLIKELY(file_size <= 0)) return NULL;
    // LCOV_EXCL_STOP

    zxc_stdio_ctx_t* const ctx = (zxc_stdio_ctx_t*)ZXC_MALLOC(sizeof(*ctx));
    if (UNLIKELY(!ctx)) return NULL;  // LCOV_EXCL_LINE

#if defined(_WIN32)
    ctx->handle = (HANDLE)(intptr_t)_get_osfhandle(_fileno(f));  // LCOV_EXCL_LINE
#else
    ctx->fd = fileno(f);
#endif
    ctx->size = (uint64_t)file_size;

    const zxc_reader_t reader = {
        .read_at = zxc_stdio_read_at, .ctx = ctx, .size = (uint64_t)file_size};

    zxc_seekable* const s = zxc_seekable_open_reader(&reader);
    if (UNLIKELY(!s)) {
        ZXC_FREE(ctx);
        return NULL;
    }

    /* Hand the ctx lifetime over to the seekable handle. */
    zxc_seekable_attach_owned_ctx(s, ctx);
    return s;
}
