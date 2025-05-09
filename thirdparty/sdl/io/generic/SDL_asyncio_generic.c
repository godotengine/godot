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

// The generic backend uses a threadpool to block on synchronous i/o.
// This is not ideal, it's meant to be used if there isn't a platform-specific
// backend that can do something more efficient!

#include "SDL_internal.h"
#include "../SDL_sysasyncio.h"

// on Emscripten without threads, async i/o is synchronous. Sorry. Almost
// everything is MEMFS, so it's just a memcpy anyhow, and the Emscripten
// filesystem APIs don't offer async. In theory, directly accessing
// persistent storage _does_ offer async APIs at the browser level, but
// that's not exposed in Emscripten's filesystem abstraction.
#if defined(SDL_PLATFORM_EMSCRIPTEN) && !defined(__EMSCRIPTEN_PTHREADS__)
#define SDL_ASYNCIO_USE_THREADPOOL 0
#else
#define SDL_ASYNCIO_USE_THREADPOOL 1
#endif

typedef struct GenericAsyncIOQueueData
{
    SDL_Mutex *lock;
    SDL_Condition *condition;
    SDL_AsyncIOTask completed_tasks;
} GenericAsyncIOQueueData;

typedef struct GenericAsyncIOData
{
    SDL_Mutex *lock;  // !!! FIXME: we can skip this lock if we have an equivalent of pread/pwrite
    SDL_IOStream *io;
} GenericAsyncIOData;

static void AsyncIOTaskComplete(SDL_AsyncIOTask *task)
{
    SDL_assert(task->queue);
    GenericAsyncIOQueueData *data = (GenericAsyncIOQueueData *) task->queue->userdata;
    SDL_LockMutex(data->lock);
    LINKED_LIST_PREPEND(task, data->completed_tasks, queue);
    SDL_SignalCondition(data->condition);  // wake a thread waiting on the queue.
    SDL_UnlockMutex(data->lock);
}

// synchronous i/o is offloaded onto the threadpool. This function does the threaded work.
// This is called directly, without a threadpool, if !SDL_ASYNCIO_USE_THREADPOOL.
static void SynchronousIO(SDL_AsyncIOTask *task)
{
    SDL_assert(task->result != SDL_ASYNCIO_CANCELED);  // shouldn't have gotten in here if canceled!

    GenericAsyncIOData *data = (GenericAsyncIOData *) task->asyncio->userdata;
    SDL_IOStream *io = data->io;
    const size_t size = (size_t) task->requested_size;
    void *ptr = task->buffer;

    // this seek won't work if two tasks are reading from the same file at the same time,
    // so we lock here. This makes multiple reads from a single file serialize, but different
    // files will still run in parallel. An app can also open the same file twice to avoid this.
    SDL_LockMutex(data->lock);
    if (task->type == SDL_ASYNCIO_TASK_CLOSE) {
        bool okay = true;
        if (task->flush) {
            okay = SDL_FlushIO(data->io);
        }
        okay = SDL_CloseIO(data->io) && okay;
        task->result = okay ? SDL_ASYNCIO_COMPLETE : SDL_ASYNCIO_FAILURE;
    } else if (SDL_SeekIO(io, (Sint64) task->offset, SDL_IO_SEEK_SET) < 0) {
        task->result = SDL_ASYNCIO_FAILURE;
    } else {
        const bool writing = (task->type == SDL_ASYNCIO_TASK_WRITE);
        task->result_size = (Uint64) (writing ? SDL_WriteIO(io, ptr, size) : SDL_ReadIO(io, ptr, size));
        if (task->result_size == task->requested_size) {
            task->result = SDL_ASYNCIO_COMPLETE;
        } else {
            if (writing) {
                task->result = SDL_ASYNCIO_FAILURE;  // it's always a failure on short writes.
            } else {
                const SDL_IOStatus status = SDL_GetIOStatus(io);
                SDL_assert(status != SDL_IO_STATUS_READY);  // this should have either failed or been EOF.
                SDL_assert(status != SDL_IO_STATUS_NOT_READY);  // these should not be non-blocking reads!
                task->result = (status == SDL_IO_STATUS_EOF) ? SDL_ASYNCIO_COMPLETE : SDL_ASYNCIO_FAILURE;
            }
        }
    }
    SDL_UnlockMutex(data->lock);

    AsyncIOTaskComplete(task);
}

#if SDL_ASYNCIO_USE_THREADPOOL
static SDL_InitState threadpool_init;
static SDL_Mutex *threadpool_lock = NULL;
static bool stop_threadpool = false;
static SDL_AsyncIOTask threadpool_tasks;
static SDL_Condition *threadpool_condition = NULL;
static int max_threadpool_threads = 0;
static int running_threadpool_threads = 0;
static int idle_threadpool_threads = 0;
static int threadpool_threads_spun = 0;

static int SDLCALL AsyncIOThreadpoolWorker(void *data)
{
    SDL_LockMutex(threadpool_lock);

    while (!stop_threadpool) {
        SDL_AsyncIOTask *task = LINKED_LIST_START(threadpool_tasks, threadpool);
        if (!task) {
            // if we go 30 seconds without a new task, terminate unless we're the only thread left.
            idle_threadpool_threads++;
            const bool rc = SDL_WaitConditionTimeout(threadpool_condition, threadpool_lock, 30000);
            idle_threadpool_threads--;

            if (!rc) {
                // decide if we have too many idle threads, and if so, quit to let thread pool shrink when not busy.
                if (idle_threadpool_threads) {
                    break;
                }
            }

            continue;
        }

        LINKED_LIST_UNLINK(task, threadpool);

        SDL_UnlockMutex(threadpool_lock);

        // bookkeeping is done, so we drop the mutex and fire the work.
        SynchronousIO(task);

        SDL_LockMutex(threadpool_lock);  // take the lock again and see if there's another task (if not, we'll wait on the Condition).
    }

    running_threadpool_threads--;

    // this is kind of a hack, but this lets us reuse threadpool_condition to block on shutdown until all threads have exited.
    if (stop_threadpool) {
        SDL_BroadcastCondition(threadpool_condition);
    }

    SDL_UnlockMutex(threadpool_lock);

    return 0;
}

static bool MaybeSpinNewWorkerThread(void)
{
    // if all existing threads are busy and the pool of threads isn't maxed out, make a new one.
    if ((idle_threadpool_threads == 0) && (running_threadpool_threads < max_threadpool_threads)) {
        char threadname[32];
        SDL_snprintf(threadname, sizeof (threadname), "SDLasyncio%d", threadpool_threads_spun);
        SDL_Thread *thread = SDL_CreateThread(AsyncIOThreadpoolWorker, threadname, NULL);
        if (thread == NULL) {
            return false;
        }
        SDL_DetachThread(thread);  // these terminate themselves when idle too long, so we never WaitThread.
        running_threadpool_threads++;
        threadpool_threads_spun++;
    }
    return true;
}

static void QueueAsyncIOTask(SDL_AsyncIOTask *task)
{
    SDL_assert(task != NULL);

    SDL_LockMutex(threadpool_lock);

    if (stop_threadpool) {  // just in case.
        task->result = SDL_ASYNCIO_CANCELED;
        AsyncIOTaskComplete(task);
    } else {
        LINKED_LIST_PREPEND(task, threadpool_tasks, threadpool);
        MaybeSpinNewWorkerThread();  // okay if this fails or the thread pool is maxed out. Something will get there eventually.

        // tell idle threads to get to work.
        // This is a broadcast because we want someone from the thread pool to wake up, but
        // also shutdown might also be blocking on this. One of the threads will grab
        // it, the others will go back to sleep.
        SDL_BroadcastCondition(threadpool_condition);
    }

    SDL_UnlockMutex(threadpool_lock);
}

// We don't initialize async i/o at all until it's used, so
//  JUST IN CASE two things try to start at the same time,
//  this will make sure everything gets the same mutex.
static bool PrepareThreadpool(void)
{
    bool okay = true;
    if (SDL_ShouldInit(&threadpool_init)) {
        max_threadpool_threads = (SDL_GetNumLogicalCPUCores() * 2) + 1;  // !!! FIXME: this should probably have a hint to override.
        max_threadpool_threads = SDL_clamp(max_threadpool_threads, 1, 8);  // 8 is probably more than enough.

        okay = (okay && ((threadpool_lock = SDL_CreateMutex()) != NULL));
        okay = (okay && ((threadpool_condition = SDL_CreateCondition()) != NULL));
        okay = (okay && MaybeSpinNewWorkerThread());  // make sure at least one thread is going, since we'll need it.

        if (!okay) {
            if (threadpool_condition) {
                SDL_DestroyCondition(threadpool_condition);
                threadpool_condition = NULL;
            }
            if (threadpool_lock) {
                SDL_DestroyMutex(threadpool_lock);
                threadpool_lock = NULL;
            }
        }

        SDL_SetInitialized(&threadpool_init, okay);
    }
    return okay;
}

static void ShutdownThreadpool(void)
{
    if (SDL_ShouldQuit(&threadpool_init)) {
        SDL_LockMutex(threadpool_lock);

        // cancel anything that's still pending.
        SDL_AsyncIOTask *task;
        while ((task = LINKED_LIST_START(threadpool_tasks, threadpool)) != NULL) {
            LINKED_LIST_UNLINK(task, threadpool);
            task->result = SDL_ASYNCIO_CANCELED;
            AsyncIOTaskComplete(task);
        }

        stop_threadpool = true;
        SDL_BroadcastCondition(threadpool_condition);  // tell the whole threadpool to wake up and quit.

        while (running_threadpool_threads > 0) {
            // each threadpool thread will broadcast this condition before it terminates if stop_threadpool is set.
            // we can't just join the threads because they are detached, so the thread pool can automatically shrink as necessary.
            SDL_WaitCondition(threadpool_condition, threadpool_lock);
        }

        SDL_UnlockMutex(threadpool_lock);

        SDL_DestroyMutex(threadpool_lock);
        threadpool_lock = NULL;
        SDL_DestroyCondition(threadpool_condition);
        threadpool_condition = NULL;

        max_threadpool_threads = running_threadpool_threads = idle_threadpool_threads = threadpool_threads_spun = 0;

        stop_threadpool = false;
        SDL_SetInitialized(&threadpool_init, false);
    }
}
#endif


static Sint64 generic_asyncio_size(void *userdata)
{
    GenericAsyncIOData *data = (GenericAsyncIOData *) userdata;
    return SDL_GetIOSize(data->io);
}

static bool generic_asyncio_io(void *userdata, SDL_AsyncIOTask *task)
{
    return task->queue->iface.queue_task(task->queue->userdata, task);
}

static void generic_asyncio_destroy(void *userdata)
{
    GenericAsyncIOData *data = (GenericAsyncIOData *) userdata;
    SDL_DestroyMutex(data->lock);
    SDL_free(data);
}


static bool generic_asyncioqueue_queue_task(void *userdata, SDL_AsyncIOTask *task)
{
    #if SDL_ASYNCIO_USE_THREADPOOL
    QueueAsyncIOTask(task);
    #else
    SynchronousIO(task);  // oh well. Get a better platform.
    #endif
    return true;
}

static void generic_asyncioqueue_cancel_task(void *userdata, SDL_AsyncIOTask *task)
{
    #if !SDL_ASYNCIO_USE_THREADPOOL  // in theory, this was all synchronous and should never call this, but just in case.
    task->result = SDL_ASYNCIO_CANCELED;
    AsyncIOTaskComplete(task);
    #else
    // we can't stop i/o that's in-flight, but we _can_ just refuse to start it if the threadpool hadn't picked it up yet.
    SDL_LockMutex(threadpool_lock);
    if (LINKED_LIST_PREV(task, threadpool) != NULL) {  // still in the queue waiting to be run? Take it out.
        LINKED_LIST_UNLINK(task, threadpool);
        task->result = SDL_ASYNCIO_CANCELED;
        AsyncIOTaskComplete(task);
    }
    SDL_UnlockMutex(threadpool_lock);
    #endif
}

static SDL_AsyncIOTask *generic_asyncioqueue_get_results(void *userdata)
{
    GenericAsyncIOQueueData *data = (GenericAsyncIOQueueData *) userdata;
    SDL_LockMutex(data->lock);
    SDL_AsyncIOTask *task = LINKED_LIST_START(data->completed_tasks, queue);
    if (task) {
        LINKED_LIST_UNLINK(task, queue);
    }
    SDL_UnlockMutex(data->lock);
    return task;
}

static SDL_AsyncIOTask *generic_asyncioqueue_wait_results(void *userdata, Sint32 timeoutMS)
{
    GenericAsyncIOQueueData *data = (GenericAsyncIOQueueData *) userdata;
    SDL_LockMutex(data->lock);
    SDL_AsyncIOTask *task = LINKED_LIST_START(data->completed_tasks, queue);
    if (!task) {
        SDL_WaitConditionTimeout(data->condition, data->lock, timeoutMS);
        task = LINKED_LIST_START(data->completed_tasks, queue);
    }
    if (task) {
        LINKED_LIST_UNLINK(task, queue);
    }
    SDL_UnlockMutex(data->lock);
    return task;
}

static void generic_asyncioqueue_signal(void *userdata)
{
    GenericAsyncIOQueueData *data = (GenericAsyncIOQueueData *) userdata;
    SDL_LockMutex(data->lock);
    SDL_BroadcastCondition(data->condition);
    SDL_UnlockMutex(data->lock);
}

static void generic_asyncioqueue_destroy(void *userdata)
{
    GenericAsyncIOQueueData *data = (GenericAsyncIOQueueData *) userdata;
    SDL_DestroyMutex(data->lock);
    SDL_DestroyCondition(data->condition);
    SDL_free(data);
}

bool SDL_SYS_CreateAsyncIOQueue_Generic(SDL_AsyncIOQueue *queue)
{
    #if SDL_ASYNCIO_USE_THREADPOOL
    if (!PrepareThreadpool()) {
        return false;
    }
    #endif

    GenericAsyncIOQueueData *data = (GenericAsyncIOQueueData *) SDL_calloc(1, sizeof (*data));
    if (!data) {
        return false;
    }

    data->lock = SDL_CreateMutex();
    if (!data->lock) {
        SDL_free(data);
        return false;
    }

    data->condition = SDL_CreateCondition();
    if (!data->condition) {
        SDL_DestroyMutex(data->lock);
        SDL_free(data);
        return false;
    }

    static const SDL_AsyncIOQueueInterface SDL_AsyncIOQueue_Generic = {
        generic_asyncioqueue_queue_task,
        generic_asyncioqueue_cancel_task,
        generic_asyncioqueue_get_results,
        generic_asyncioqueue_wait_results,
        generic_asyncioqueue_signal,
        generic_asyncioqueue_destroy
    };

    SDL_copyp(&queue->iface, &SDL_AsyncIOQueue_Generic);
    queue->userdata = data;
    return true;
}


bool SDL_SYS_AsyncIOFromFile_Generic(const char *file, const char *mode, SDL_AsyncIO *asyncio)
{
    #if SDL_ASYNCIO_USE_THREADPOOL
    if (!PrepareThreadpool()) {
        return false;
    }
    #endif

    GenericAsyncIOData *data = (GenericAsyncIOData *) SDL_calloc(1, sizeof (*data));
    if (!data) {
        return false;
    }

    data->lock = SDL_CreateMutex();
    if (!data->lock) {
        SDL_free(data);
        return false;
    }

    data->io = SDL_IOFromFile(file, mode);
    if (!data->io) {
        SDL_DestroyMutex(data->lock);
        SDL_free(data);
        return false;
    }

    static const SDL_AsyncIOInterface SDL_AsyncIOFile_Generic = {
        generic_asyncio_size,
        generic_asyncio_io,
        generic_asyncio_io,
        generic_asyncio_io,
        generic_asyncio_destroy
    };

    SDL_copyp(&asyncio->iface, &SDL_AsyncIOFile_Generic);
    asyncio->userdata = data;
    return true;
}

void SDL_SYS_QuitAsyncIO_Generic(void)
{
    #if SDL_ASYNCIO_USE_THREADPOOL
    ShutdownThreadpool();
    #endif
}


#if SDL_ASYNCIO_ONLY_HAVE_GENERIC
bool SDL_SYS_AsyncIOFromFile(const char *file, const char *mode, SDL_AsyncIO *asyncio)
{
    return SDL_SYS_AsyncIOFromFile_Generic(file, mode, asyncio);
}

bool SDL_SYS_CreateAsyncIOQueue(SDL_AsyncIOQueue *queue)
{
    return SDL_SYS_CreateAsyncIOQueue_Generic(queue);
}

void SDL_SYS_QuitAsyncIO(void)
{
    SDL_SYS_QuitAsyncIO_Generic();
}
#endif

