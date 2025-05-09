/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

// The Linux backend uses io_uring for asynchronous i/o, and falls back to
// the "generic" threadpool implementation if liburing isn't available or
// fails for some other reason.

#include "SDL_internal.h"

#ifdef HAVE_LIBURING_H

#include "../SDL_sysasyncio.h"

#include <liburing.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>  // for strerror()

static SDL_InitState liburing_init;

// We could add a whole bootstrap thing like the audio/video/etc subsystems use, but let's keep this simple for now.
static bool (*CreateAsyncIOQueue)(SDL_AsyncIOQueue *queue);
static void (*QuitAsyncIO)(void);
static bool (*AsyncIOFromFile)(const char *file, const char *mode, SDL_AsyncIO *asyncio);

// we never link directly to liburing.
// (this says "-ffi" which sounds like a scripting language binding thing, but the non-ffi version
// is static-inline code we can't lookup with dlsym. This is by design.)
static const char *liburing_library = "liburing-ffi.so.2";
static void *liburing_handle = NULL;

#define SDL_LIBURING_FUNCS \
    SDL_LIBURING_FUNC(int, io_uring_queue_init, (unsigned entries, struct io_uring *ring, unsigned flags)) \
    SDL_LIBURING_FUNC(struct io_uring_probe *,io_uring_get_probe,(void)) \
    SDL_LIBURING_FUNC(void, io_uring_free_probe, (struct io_uring_probe *probe)) \
    SDL_LIBURING_FUNC(int, io_uring_opcode_supported, (const struct io_uring_probe *p, int op)) \
    SDL_LIBURING_FUNC(struct io_uring_sqe *, io_uring_get_sqe, (struct io_uring *ring)) \
    SDL_LIBURING_FUNC(void, io_uring_prep_read,(struct io_uring_sqe *sqe, int fd, void *buf, unsigned nbytes, __u64 offset)) \
    SDL_LIBURING_FUNC(void, io_uring_prep_write,(struct io_uring_sqe *sqe, int fd, const void *buf, unsigned nbytes, __u64 offset)) \
    SDL_LIBURING_FUNC(void, io_uring_prep_close, (struct io_uring_sqe *sqe, int fd)) \
    SDL_LIBURING_FUNC(void, io_uring_prep_fsync, (struct io_uring_sqe *sqe, int fd, unsigned fsync_flags)) \
    SDL_LIBURING_FUNC(void, io_uring_prep_cancel, (struct io_uring_sqe *sqe, void *user_data, int flags)) \
    SDL_LIBURING_FUNC(void, io_uring_prep_timeout, (struct io_uring_sqe *sqe, struct __kernel_timespec *ts, unsigned count, unsigned flags)) \
    SDL_LIBURING_FUNC(void, io_uring_prep_nop, (struct io_uring_sqe *sqe)) \
    SDL_LIBURING_FUNC(void, io_uring_sqe_set_data, (struct io_uring_sqe *sqe, void *data)) \
    SDL_LIBURING_FUNC(void, io_uring_sqe_set_flags, (struct io_uring_sqe *sqe, unsigned flags)) \
    SDL_LIBURING_FUNC(int, io_uring_submit, (struct io_uring *ring)) \
    SDL_LIBURING_FUNC(int, io_uring_peek_cqe, (struct io_uring *ring, struct io_uring_cqe **cqe_ptr)) \
    SDL_LIBURING_FUNC(int, io_uring_wait_cqe, (struct io_uring *ring, struct io_uring_cqe **cqe_ptr)) \
    SDL_LIBURING_FUNC(int, io_uring_wait_cqe_timeout, (struct io_uring *ring, struct io_uring_cqe **cqe_ptr, struct __kernel_timespec *ts)) \
    SDL_LIBURING_FUNC(void, io_uring_cqe_seen, (struct io_uring *ring, struct io_uring_cqe *cqe)) \
    SDL_LIBURING_FUNC(void, io_uring_queue_exit, (struct io_uring *ring)) \


#define SDL_LIBURING_FUNC(ret, fn, args) typedef ret (*SDL_fntype_##fn) args;
SDL_LIBURING_FUNCS
#undef SDL_LIBURING_FUNC

typedef struct SDL_LibUringFunctions
{
    #define SDL_LIBURING_FUNC(ret, fn, args) SDL_fntype_##fn fn;
    SDL_LIBURING_FUNCS
    #undef SDL_LIBURING_FUNC
} SDL_LibUringFunctions;

static SDL_LibUringFunctions liburing;


typedef struct LibUringAsyncIOQueueData
{
    SDL_Mutex *sqe_lock;
    SDL_Mutex *cqe_lock;
    struct io_uring ring;
    SDL_AtomicInt num_waiting;
} LibUringAsyncIOQueueData;


static void UnloadLibUringLibrary(void)
{
    if (liburing_library) {
        SDL_UnloadObject(liburing_handle);
        liburing_library = NULL;
    }
    SDL_zero(liburing);
}

static bool LoadLibUringSyms(void)
{
    #define SDL_LIBURING_FUNC(ret, fn, args) { \
        liburing.fn = (SDL_fntype_##fn) SDL_LoadFunction(liburing_handle, #fn); \
        if (!liburing.fn) { \
            return false; \
        } \
    }
    SDL_LIBURING_FUNCS
    #undef SDL_LIBURING_FUNC
    return true;
}

// we rely on the presence of liburing to handle io_uring for us. The alternative is making
// direct syscalls into the kernel, which is undesirable. liburing both shields us from this,
// but also smooths over some kernel version differences, etc.
static bool LoadLibUring(void)
{
    bool result = true;

    if (!liburing_handle) {
        liburing_handle = SDL_LoadObject(liburing_library);
        if (!liburing_handle) {
            result = false;
            // Don't call SDL_SetError(): SDL_LoadObject already did.
        } else {
            result = LoadLibUringSyms();
            if (result) {
                static const int needed_ops[] = {
                    IORING_OP_NOP,
                    IORING_OP_FSYNC,
                    IORING_OP_TIMEOUT,
                    IORING_OP_CLOSE,
                    IORING_OP_READ,
                    IORING_OP_WRITE,
                    IORING_OP_ASYNC_CANCEL
                };

                struct io_uring_probe *probe = liburing.io_uring_get_probe();
                if (!probe) {
                    result = false;
                } else {
                    for (int i = 0; i < SDL_arraysize(needed_ops); i++) {
                        if (!io_uring_opcode_supported(probe, needed_ops[i])) {
                            result = false;
                            break;
                        }
                    }
                    liburing.io_uring_free_probe(probe);
                }
            }

            if (!result) {
                UnloadLibUringLibrary();
            }
        }
    }
    return result;
}

static bool liburing_SetError(const char *what, int err)
{
    SDL_assert(err <= 0);
    return SDL_SetError("%s failed: %s", what, strerror(-err));
}

static Sint64 liburing_asyncio_size(void *userdata)
{
    const int fd = (int) (intptr_t) userdata;
    struct stat statbuf;
    if (fstat(fd, &statbuf) < 0) {
        SDL_SetError("fstat failed: %s", strerror(errno));
        return -1;
    }
    return ((Sint64) statbuf.st_size);
}

// you must hold sqe_lock when calling this!
static bool liburing_asyncioqueue_queue_task(void *userdata, SDL_AsyncIOTask *task)
{
    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) userdata;
    const int rc = liburing.io_uring_submit(&queuedata->ring);
    return (rc < 0) ? liburing_SetError("io_uring_submit", rc) : true;
}

static void liburing_asyncioqueue_cancel_task(void *userdata, SDL_AsyncIOTask *task)
{
    SDL_AsyncIOTask *cancel_task = (SDL_AsyncIOTask *) SDL_calloc(1, sizeof (*cancel_task));
    if (!cancel_task) {
        return;  // oh well, the task can just finish on its own.
    }

    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) userdata;

    // have to hold a lock because otherwise two threads could get_sqe and submit while one request isn't fully set up.
    SDL_LockMutex(queuedata->sqe_lock);
    struct io_uring_sqe *sqe = liburing.io_uring_get_sqe(&queuedata->ring);
    if (!sqe) {
        SDL_UnlockMutex(queuedata->sqe_lock);
        SDL_free(cancel_task);  // oh well, the task can just finish on its own.
        return;
    }

    cancel_task->app_userdata = task;
    liburing.io_uring_prep_cancel(sqe, task, 0);
    liburing.io_uring_sqe_set_data(sqe, cancel_task);
    liburing_asyncioqueue_queue_task(userdata, task);
    SDL_UnlockMutex(queuedata->sqe_lock);
}

static SDL_AsyncIOTask *ProcessCQE(LibUringAsyncIOQueueData *queuedata, struct io_uring_cqe *cqe)
{
    if (!cqe) {
        return NULL;
    }

    SDL_AsyncIOTask *task = (SDL_AsyncIOTask *) io_uring_cqe_get_data(cqe);
    if (task) {  // can be NULL if this was just a wakeup message, a NOP, etc.
        if (!task->queue) {  // We leave `queue` blank to signify this was a task cancellation.
            SDL_AsyncIOTask *cancel_task = task;
            task = (SDL_AsyncIOTask *) cancel_task->app_userdata;
            SDL_free(cancel_task);
            if (cqe->res >= 0) {  // cancel was successful?
                task->result = SDL_ASYNCIO_CANCELED;
            } else {
                task = NULL; // it already finished or was too far along to cancel, so we'll pick up the actual results later.
            }
        } else if (cqe->res < 0) {
            task->result = SDL_ASYNCIO_FAILURE;
            // !!! FIXME: fill in task->error.
        } else {
            if ((task->type == SDL_ASYNCIO_TASK_WRITE) && (((Uint64) cqe->res) < task->requested_size)) {
                task->result = SDL_ASYNCIO_FAILURE;  // it's always a failure on short writes.
            }

            // don't explicitly mark it as COMPLETE; that's the default value and a linked task might have failed in an earlier operation and this would overwrite it.

            if ((task->type == SDL_ASYNCIO_TASK_READ) || (task->type == SDL_ASYNCIO_TASK_WRITE)) {
                task->result_size = (Uint64) cqe->res;
            }
        }

        if ((task->type == SDL_ASYNCIO_TASK_CLOSE) && task->flush) {
            task->flush = false;
            task = NULL;  // don't return this one, it's a linked task, so it'll arrive in a later CQE.
        }
    }

    return task;
}

static SDL_AsyncIOTask *liburing_asyncioqueue_get_results(void *userdata)
{
    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) userdata;

    // have to hold a lock because otherwise two threads will get the same cqe until we mark it "seen". Copy and mark it right away, then process further.
    SDL_LockMutex(queuedata->cqe_lock);
    struct io_uring_cqe *cqe = NULL;
    const int rc = liburing.io_uring_peek_cqe(&queuedata->ring, &cqe);
    if (rc != 0) {
        SDL_assert(rc == -EAGAIN);  // should only fail because nothing is available at the moment.
        SDL_UnlockMutex(queuedata->cqe_lock);
        return NULL;
    }

    struct io_uring_cqe cqe_copy;
    SDL_copyp(&cqe_copy, cqe);  // this is only a few bytes.
    liburing.io_uring_cqe_seen(&queuedata->ring, cqe);  // let io_uring use this slot again.
    SDL_UnlockMutex(queuedata->cqe_lock);

    return ProcessCQE(queuedata, &cqe_copy);
}

static SDL_AsyncIOTask *liburing_asyncioqueue_wait_results(void *userdata, Sint32 timeoutMS)
{
    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) userdata;
    struct io_uring_cqe *cqe = NULL;

    SDL_AddAtomicInt(&queuedata->num_waiting, 1);
    if (timeoutMS < 0) {
        liburing.io_uring_wait_cqe(&queuedata->ring, &cqe);
    } else {
        struct __kernel_timespec ts = { (Sint64) timeoutMS / SDL_MS_PER_SECOND, (Sint64) SDL_MS_TO_NS(timeoutMS % SDL_MS_PER_SECOND) };
        liburing.io_uring_wait_cqe_timeout(&queuedata->ring, &cqe, &ts);
    }
    SDL_AddAtomicInt(&queuedata->num_waiting, -1);

    // (we don't care if the wait failed for any reason, as the upcoming peek_cqe will report valid information. We just wanted the wait operation to block.)

    // each thing that peeks or waits for a completion _gets the same cqe_ until we mark it as seen. So when we wake up from the wait, lock the mutex and
    // then use peek to make sure we have a unique cqe, and other competing threads either get their own or nothing.
    return liburing_asyncioqueue_get_results(userdata);  // this just happens to do all those things.
}

static void liburing_asyncioqueue_signal(void *userdata)
{
    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) userdata;
    const int num_waiting = SDL_GetAtomicInt(&queuedata->num_waiting);

    SDL_LockMutex(queuedata->sqe_lock);
    for (int i = 0; i < num_waiting; i++) {  // !!! FIXME: is there a better way to do this than pushing a zero-timeout request for everything waiting?
        struct io_uring_sqe *sqe = liburing.io_uring_get_sqe(&queuedata->ring);
        if (sqe) {
            static struct __kernel_timespec ts;   // no wait, just wake a thread as fast as this can land in the completion queue.
            liburing.io_uring_prep_timeout(sqe, &ts, 0, 0);
            liburing.io_uring_sqe_set_data(sqe, NULL);
        }
    }
    liburing.io_uring_submit(&queuedata->ring);

    SDL_UnlockMutex(queuedata->sqe_lock);
}

static void liburing_asyncioqueue_destroy(void *userdata)
{
    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) userdata;
    liburing.io_uring_queue_exit(&queuedata->ring);
    SDL_DestroyMutex(queuedata->sqe_lock);
    SDL_DestroyMutex(queuedata->cqe_lock);
    SDL_free(queuedata);
}

static bool SDL_SYS_CreateAsyncIOQueue_liburing(SDL_AsyncIOQueue *queue)
{
    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) SDL_calloc(1, sizeof (*queuedata));
    if (!queuedata) {
        return false;
    }

    SDL_SetAtomicInt(&queuedata->num_waiting, 0);

    queuedata->sqe_lock = SDL_CreateMutex();
    if (!queuedata->sqe_lock) {
        SDL_free(queuedata);
        return false;
    }

    queuedata->cqe_lock = SDL_CreateMutex();
    if (!queuedata->cqe_lock) {
        SDL_DestroyMutex(queuedata->sqe_lock);
        SDL_free(queuedata);
        return false;
    }

    // !!! FIXME: no idea how large the queue should be. Is 128 overkill or too small?
    const int rc = liburing.io_uring_queue_init(128, &queuedata->ring, 0);
    if (rc != 0) {
        SDL_DestroyMutex(queuedata->sqe_lock);
        SDL_DestroyMutex(queuedata->cqe_lock);
        SDL_free(queuedata);
        return liburing_SetError("io_uring_queue_init", rc);
    }

    static const SDL_AsyncIOQueueInterface SDL_AsyncIOQueue_liburing = {
        liburing_asyncioqueue_queue_task,
        liburing_asyncioqueue_cancel_task,
        liburing_asyncioqueue_get_results,
        liburing_asyncioqueue_wait_results,
        liburing_asyncioqueue_signal,
        liburing_asyncioqueue_destroy
    };

    SDL_copyp(&queue->iface, &SDL_AsyncIOQueue_liburing);
    queue->userdata = queuedata;
    return true;
}


static bool liburing_asyncio_read(void *userdata, SDL_AsyncIOTask *task)
{
    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) task->queue->userdata;
    const int fd = (int) (intptr_t) userdata;

    // !!! FIXME: `unsigned` is likely smaller than requested_size's Uint64. If we overflow it, we could try submitting multiple SQEs
    // !!! FIXME:  and make a note in the task that there are several in sequence.
    if (task->requested_size > ((Uint64) ~((unsigned) 0))) {
        return SDL_SetError("io_uring: i/o task is too large");
    }

    // have to hold a lock because otherwise two threads could get_sqe and submit while one request isn't fully set up.
    SDL_LockMutex(queuedata->sqe_lock);
    bool retval;
    struct io_uring_sqe *sqe = liburing.io_uring_get_sqe(&queuedata->ring);
    if (!sqe) {
        retval = SDL_SetError("io_uring: submission queue is full");
    } else {
        liburing.io_uring_prep_read(sqe, fd, task->buffer, (unsigned) task->requested_size, task->offset);
        liburing.io_uring_sqe_set_data(sqe, task);
        retval = task->queue->iface.queue_task(task->queue->userdata, task);
    }
    SDL_UnlockMutex(queuedata->sqe_lock);
    return retval;
}

static bool liburing_asyncio_write(void *userdata, SDL_AsyncIOTask *task)
{
    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) task->queue->userdata;
    const int fd = (int) (intptr_t) userdata;

    // !!! FIXME: `unsigned` is likely smaller than requested_size's Uint64. If we overflow it, we could try submitting multiple SQEs
    // !!! FIXME:  and make a note in the task that there are several in sequence.
    if (task->requested_size > ((Uint64) ~((unsigned) 0))) {
        return SDL_SetError("io_uring: i/o task is too large");
    }

    // have to hold a lock because otherwise two threads could get_sqe and submit while one request isn't fully set up.
    SDL_LockMutex(queuedata->sqe_lock);
    bool retval;
    struct io_uring_sqe *sqe = liburing.io_uring_get_sqe(&queuedata->ring);
    if (!sqe) {
        retval = SDL_SetError("io_uring: submission queue is full");
    } else {
        liburing.io_uring_prep_write(sqe, fd, task->buffer, (unsigned) task->requested_size, task->offset);
        liburing.io_uring_sqe_set_data(sqe, task);
        retval = task->queue->iface.queue_task(task->queue->userdata, task);
    }
    SDL_UnlockMutex(queuedata->sqe_lock);
    return retval;
}

static bool liburing_asyncio_close(void *userdata, SDL_AsyncIOTask *task)
{
    LibUringAsyncIOQueueData *queuedata = (LibUringAsyncIOQueueData *) task->queue->userdata;
    const int fd = (int) (intptr_t) userdata;

    // have to hold a lock because otherwise two threads could get_sqe and submit while one request isn't fully set up.
    SDL_LockMutex(queuedata->sqe_lock);
    bool retval;
    struct io_uring_sqe *sqe = liburing.io_uring_get_sqe(&queuedata->ring);
    if (!sqe) {
        retval = SDL_SetError("io_uring: submission queue is full");
    } else {
        if (task->flush) {
            struct io_uring_sqe *flush_sqe = sqe;
            sqe = liburing.io_uring_get_sqe(&queuedata->ring);  // this will be our actual close task.
            if (!sqe) {
                liburing.io_uring_prep_nop(flush_sqe);  // we already have the first sqe, just make it a NOP.
                liburing.io_uring_sqe_set_data(flush_sqe, NULL);
                task->queue->iface.queue_task(task->queue->userdata, task);
                SDL_UnlockMutex(queuedata->sqe_lock);
                return SDL_SetError("io_uring: submission queue is full");
            }
            liburing.io_uring_prep_fsync(flush_sqe, fd, IORING_FSYNC_DATASYNC);
            liburing.io_uring_sqe_set_data(flush_sqe, task);
            liburing.io_uring_sqe_set_flags(flush_sqe, IOSQE_IO_HARDLINK);  // must complete before next sqe starts, and next sqe should run even if this fails.
        }

        liburing.io_uring_prep_close(sqe, fd);
        liburing.io_uring_sqe_set_data(sqe, task);

        retval = task->queue->iface.queue_task(task->queue->userdata, task);
    }
    SDL_UnlockMutex(queuedata->sqe_lock);
    return retval;
}

static void liburing_asyncio_destroy(void *userdata)
{
    // this is only a Unix file descriptor, should have been closed elsewhere.
}

static int PosixOpenModeFromString(const char *mode)
{
    // this is exactly the set of strings that SDL_AsyncIOFromFile promises will work.
    static const struct { const char *str; int flags; } mappings[] = {
        { "rb", O_RDONLY },
        { "wb", O_WRONLY | O_CREAT | O_TRUNC },
        { "r+b", O_RDWR },
        { "w+b", O_RDWR | O_CREAT | O_TRUNC }
    };

    for (int i = 0; i < SDL_arraysize(mappings); i++) {
        if (SDL_strcmp(mappings[i].str, mode) == 0) {
            return mappings[i].flags;
        }
    }

    SDL_assert(!"Shouldn't have reached this code");
    return 0;
}

static bool SDL_SYS_AsyncIOFromFile_liburing(const char *file, const char *mode, SDL_AsyncIO *asyncio)
{
    const int fd = open(file, PosixOpenModeFromString(mode), 0644);
    if (fd == -1) {
        return SDL_SetError("open failed: %s", strerror(errno));
    }

    static const SDL_AsyncIOInterface SDL_AsyncIOFile_liburing = {
        liburing_asyncio_size,
        liburing_asyncio_read,
        liburing_asyncio_write,
        liburing_asyncio_close,
        liburing_asyncio_destroy
    };

    SDL_copyp(&asyncio->iface, &SDL_AsyncIOFile_liburing);
    asyncio->userdata = (void *) (intptr_t) fd;
    return true;
}

static void SDL_SYS_QuitAsyncIO_liburing(void)
{
    UnloadLibUringLibrary();
}

static void MaybeInitializeLibUring(void)
{
    if (SDL_ShouldInit(&liburing_init)) {
        if (LoadLibUring()) {
            CreateAsyncIOQueue = SDL_SYS_CreateAsyncIOQueue_liburing;
            QuitAsyncIO = SDL_SYS_QuitAsyncIO_liburing;
            AsyncIOFromFile = SDL_SYS_AsyncIOFromFile_liburing;
        } else {  // can't use liburing? Use the "generic" threadpool implementation instead.
            CreateAsyncIOQueue = SDL_SYS_CreateAsyncIOQueue_Generic;
            QuitAsyncIO = SDL_SYS_QuitAsyncIO_Generic;
            AsyncIOFromFile = SDL_SYS_AsyncIOFromFile_Generic;
        }
        SDL_SetInitialized(&liburing_init, true);
    }
}

bool SDL_SYS_CreateAsyncIOQueue(SDL_AsyncIOQueue *queue)
{
    MaybeInitializeLibUring();
    return CreateAsyncIOQueue(queue);
}

bool SDL_SYS_AsyncIOFromFile(const char *file, const char *mode, SDL_AsyncIO *asyncio)
{
    MaybeInitializeLibUring();
    return AsyncIOFromFile(file, mode, asyncio);
}

void SDL_SYS_QuitAsyncIO(void)
{
    if (SDL_ShouldQuit(&liburing_init)) {
        QuitAsyncIO();
        CreateAsyncIOQueue = NULL;
        QuitAsyncIO = NULL;
        AsyncIOFromFile = NULL;
        SDL_SetInitialized(&liburing_init, false);
    }
}

#endif  // defined HAVE_LIBURING_H

