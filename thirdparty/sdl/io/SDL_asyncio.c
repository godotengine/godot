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

#include "SDL_internal.h"
#include "SDL_sysasyncio.h"
#include "SDL_asyncio_c.h"

static const char *AsyncFileModeValid(const char *mode)
{
    static const struct { const char *valid; const char *with_binary; } mode_map[] = {
        { "r", "rb" },
        { "w", "wb" },
        { "r+","r+b" },
        { "w+", "w+b" }
    };

    for (int i = 0; i < SDL_arraysize(mode_map); i++) {
        if (SDL_strcmp(mode, mode_map[i].valid) == 0) {
            return mode_map[i].with_binary;
        }
    }
    return NULL;
}

SDL_AsyncIO *SDL_AsyncIOFromFile(const char *file, const char *mode)
{
    if (!file) {
        SDL_InvalidParamError("file");
        return NULL;
    } else if (!mode) {
        SDL_InvalidParamError("mode");
        return NULL;
    }

    const char *binary_mode = AsyncFileModeValid(mode);
    if (!binary_mode) {
        SDL_SetError("Unsupported file mode");
        return NULL;
    }

    SDL_AsyncIO *asyncio = (SDL_AsyncIO *)SDL_calloc(1, sizeof(*asyncio));
    if (!asyncio) {
        return NULL;
    }

    asyncio->lock = SDL_CreateMutex();
    if (!asyncio->lock) {
        SDL_free(asyncio);
        return NULL;
    }

    if (!SDL_SYS_AsyncIOFromFile(file, binary_mode, asyncio)) {
        SDL_DestroyMutex(asyncio->lock);
        SDL_free(asyncio);
        return NULL;
    }

    return asyncio;
}

Sint64 SDL_GetAsyncIOSize(SDL_AsyncIO *asyncio)
{
    if (!asyncio) {
        SDL_InvalidParamError("asyncio");
        return -1;
    }
    return asyncio->iface.size(asyncio->userdata);
}

static bool RequestAsyncIO(bool reading, SDL_AsyncIO *asyncio, void *ptr, Uint64 offset, Uint64 size, SDL_AsyncIOQueue *queue, void *userdata)
{
    if (!asyncio) {
        return SDL_InvalidParamError("asyncio");
    } else if (!ptr) {
        return SDL_InvalidParamError("ptr");
    } else if (!queue) {
        return SDL_InvalidParamError("queue");
    }

    SDL_AsyncIOTask *task = (SDL_AsyncIOTask *) SDL_calloc(1, sizeof (*task));
    if (!task) {
        return false;
    }

    task->asyncio = asyncio;
    task->type = reading ? SDL_ASYNCIO_TASK_READ : SDL_ASYNCIO_TASK_WRITE;
    task->offset = offset;
    task->buffer = ptr;
    task->requested_size = size;
    task->app_userdata = userdata;
    task->queue = queue;

    SDL_LockMutex(asyncio->lock);
    if (asyncio->closing) {
        SDL_free(task);
        SDL_UnlockMutex(asyncio->lock);
        return SDL_SetError("SDL_AsyncIO is closing, can't start new tasks");
    }
    LINKED_LIST_PREPEND(task, asyncio->tasks, asyncio);
    SDL_AddAtomicInt(&queue->tasks_inflight, 1);
    SDL_UnlockMutex(asyncio->lock);

    const bool queued = reading ? asyncio->iface.read(asyncio->userdata, task) : asyncio->iface.write(asyncio->userdata, task);
    if (!queued) {
        SDL_AddAtomicInt(&queue->tasks_inflight, -1);
        SDL_LockMutex(asyncio->lock);
        LINKED_LIST_UNLINK(task, asyncio);
        SDL_UnlockMutex(asyncio->lock);
        SDL_free(task);
        task = NULL;
    }

    return (task != NULL);
}

bool SDL_ReadAsyncIO(SDL_AsyncIO *asyncio, void *ptr, Uint64 offset, Uint64 size, SDL_AsyncIOQueue *queue, void *userdata)
{
    return RequestAsyncIO(true, asyncio, ptr, offset, size, queue, userdata);
}

bool SDL_WriteAsyncIO(SDL_AsyncIO *asyncio, void *ptr, Uint64 offset, Uint64 size, SDL_AsyncIOQueue *queue, void *userdata)
{
    return RequestAsyncIO(false, asyncio, ptr, offset, size, queue, userdata);
}

bool SDL_CloseAsyncIO(SDL_AsyncIO *asyncio, bool flush, SDL_AsyncIOQueue *queue, void *userdata)
{
    if (!asyncio) {
        return SDL_InvalidParamError("asyncio");
    } else if (!queue) {
        return SDL_InvalidParamError("queue");
    }

    SDL_LockMutex(asyncio->lock);
    if (asyncio->closing) {
        SDL_UnlockMutex(asyncio->lock);
        return SDL_SetError("Already closing");
    }

    SDL_AsyncIOTask *task = (SDL_AsyncIOTask *) SDL_calloc(1, sizeof (*task));
    if (task) {
        task->asyncio = asyncio;
        task->type = SDL_ASYNCIO_TASK_CLOSE;
        task->app_userdata = userdata;
        task->queue = queue;
        task->flush = flush;

        asyncio->closing = task;

        if (LINKED_LIST_START(asyncio->tasks, asyncio) == NULL) { // no tasks? Queue the close task now.
            LINKED_LIST_PREPEND(task, asyncio->tasks, asyncio);
            SDL_AddAtomicInt(&queue->tasks_inflight, 1);
            if (!asyncio->iface.close(asyncio->userdata, task)) {
                // uhoh, maybe they can try again later...?
                SDL_AddAtomicInt(&queue->tasks_inflight, -1);
                LINKED_LIST_UNLINK(task, asyncio);
                SDL_free(task);
                task = asyncio->closing = NULL;
            }
        }
    }

    SDL_UnlockMutex(asyncio->lock);

    return (task != NULL);
}

SDL_AsyncIOQueue *SDL_CreateAsyncIOQueue(void)
{
    SDL_AsyncIOQueue *queue = SDL_calloc(1, sizeof (*queue));
    if (queue) {
        SDL_SetAtomicInt(&queue->tasks_inflight, 0);
        if (!SDL_SYS_CreateAsyncIOQueue(queue)) {
            SDL_free(queue);
            return NULL;
        }
    }
    return queue;
}

static bool GetAsyncIOTaskOutcome(SDL_AsyncIOTask *task, SDL_AsyncIOOutcome *outcome)
{
    if (!task || !outcome) {
        return false;
    }

    SDL_AsyncIO *asyncio = task->asyncio;

    SDL_zerop(outcome);
    outcome->asyncio = asyncio->oneshot ? NULL : asyncio;
    outcome->result = task->result;
    outcome->type = task->type;
    outcome->buffer = task->buffer;
    outcome->offset = task->offset;
    outcome->bytes_requested = task->requested_size;
    outcome->bytes_transferred = task->result_size;
    outcome->userdata = task->app_userdata;

    // Take the completed task out of the SDL_AsyncIO that created it.
    SDL_LockMutex(asyncio->lock);
    LINKED_LIST_UNLINK(task, asyncio);
    // see if it's time to queue a pending close request (close requested and no other pending tasks)
    SDL_AsyncIOTask *closing = asyncio->closing;
    if (closing && (task != closing) && (LINKED_LIST_START(asyncio->tasks, asyncio) == NULL)) {
        LINKED_LIST_PREPEND(closing, asyncio->tasks, asyncio);
        SDL_AddAtomicInt(&closing->queue->tasks_inflight, 1);
        const bool async_close_task_was_queued = asyncio->iface.close(asyncio->userdata, closing);
        SDL_assert(async_close_task_was_queued);  // !!! FIXME: if this fails to queue the task, we're leaking resources!
        if (!async_close_task_was_queued) {
            SDL_AddAtomicInt(&closing->queue->tasks_inflight, -1);
        }
    }
    SDL_UnlockMutex(task->asyncio->lock);

    // was this the result of a closing task? Finally destroy the asyncio.
    bool retval = true;
    if (closing && (task == closing)) {
        if (asyncio->oneshot) {
            retval = false;  // don't send the close task results on to the app, just the read task for these.
        }
        asyncio->iface.destroy(asyncio->userdata);
        SDL_DestroyMutex(asyncio->lock);
        SDL_free(asyncio);
    }

    SDL_AddAtomicInt(&task->queue->tasks_inflight, -1);
    SDL_free(task);

    return retval;
}

bool SDL_GetAsyncIOResult(SDL_AsyncIOQueue *queue, SDL_AsyncIOOutcome *outcome)
{
    if (!queue || !outcome) {
        return false;
    }
    return GetAsyncIOTaskOutcome(queue->iface.get_results(queue->userdata), outcome);
}

bool SDL_WaitAsyncIOResult(SDL_AsyncIOQueue *queue, SDL_AsyncIOOutcome *outcome, Sint32 timeoutMS)
{
    if (!queue || !outcome) {
        return false;
    }
    return GetAsyncIOTaskOutcome(queue->iface.wait_results(queue->userdata, timeoutMS), outcome);
}

void SDL_SignalAsyncIOQueue(SDL_AsyncIOQueue *queue)
{
    if (queue) {
        queue->iface.signal(queue->userdata);
    }
}

void SDL_DestroyAsyncIOQueue(SDL_AsyncIOQueue *queue)
{
    if (queue) {
        // block until any pending tasks complete.
        while (SDL_GetAtomicInt(&queue->tasks_inflight) > 0) {
            SDL_AsyncIOTask *task = queue->iface.wait_results(queue->userdata, -1);
            if (task) {
                if (task->asyncio->oneshot) {
                    SDL_free(task->buffer);  // throw away the buffer from SDL_LoadFileAsync that will never be consumed/freed by app.
                    task->buffer = NULL;
                }
                SDL_AsyncIOOutcome outcome;
                GetAsyncIOTaskOutcome(task, &outcome);  // this frees the task, and does other upkeep.
            }
        }

        queue->iface.destroy(queue->userdata);
        SDL_free(queue);
    }
}

void SDL_QuitAsyncIO(void)
{
    SDL_SYS_QuitAsyncIO();
}

bool SDL_LoadFileAsync(const char *file, SDL_AsyncIOQueue *queue, void *userdata)
{
    if (!file) {
        return SDL_InvalidParamError("file");
    } else if (!queue) {
        return SDL_InvalidParamError("queue");
    }

    bool retval = false;
    SDL_AsyncIO *asyncio = SDL_AsyncIOFromFile(file, "r");
    if (asyncio) {
        asyncio->oneshot = true;

        Uint8 *ptr = NULL;
        const Sint64 flen = SDL_GetAsyncIOSize(asyncio);
        if (flen >= 0) {
            // !!! FIXME: check if flen > address space, since it'll truncate and we'll just end up with an incomplete buffer or a crash.
            ptr = (Uint8 *) SDL_malloc((size_t) (flen + 1));  // over-allocate by one so we can add a null-terminator.
            if (ptr) {
                ptr[flen] = '\0';
                retval = SDL_ReadAsyncIO(asyncio, ptr, 0, (Uint64) flen, queue, userdata);
                if (!retval) {
                    SDL_free(ptr);
                }
            }
        }

        SDL_CloseAsyncIO(asyncio, false, queue, userdata);  // if this fails, we'll have a resource leak, but this would already be a dramatic system failure.
    }

    return retval;
}

