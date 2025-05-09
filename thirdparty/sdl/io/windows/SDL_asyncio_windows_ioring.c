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

// The Windows backend uses IoRing for asynchronous i/o, and falls back to
// the "generic" threadpool implementation if it isn't available or
// fails for some other reason. IoRing was introduced in Windows 11.

#include "SDL_internal.h"
#include "../SDL_sysasyncio.h"

#ifdef HAVE_IORINGAPI_H

#include "../../core/windows/SDL_windows.h"
#include <ioringapi.h>

// Don't know what the lowest usable version is, but this seems safe.
#define SDL_REQUIRED_IORING_VERSION IORING_VERSION_3

static SDL_InitState ioring_init;

// We could add a whole bootstrap thing like the audio/video/etc subsystems use, but let's keep this simple for now.
static bool (*CreateAsyncIOQueue)(SDL_AsyncIOQueue *queue);
static void (*QuitAsyncIO)(void);
static bool (*AsyncIOFromFile)(const char *file, const char *mode, SDL_AsyncIO *asyncio);

// we never link directly to ioring.
static const char *ioring_library = "KernelBase.dll";
static void *ioring_handle = NULL;

#define SDL_IORING_FUNCS \
    SDL_IORING_FUNC(HRESULT, QueryIoRingCapabilities, (IORING_CAPABILITIES *capabilities)) \
    SDL_IORING_FUNC(BOOL, IsIoRingOpSupported, (HIORING ioRing, IORING_OP_CODE op)) \
    SDL_IORING_FUNC(HRESULT, CreateIoRing, (IORING_VERSION ioringVersion, IORING_CREATE_FLAGS flags, UINT32 submissionQueueSize, UINT32 completionQueueSize, HIORING* h)) \
    SDL_IORING_FUNC(HRESULT, GetIoRingInfo, (HIORING ioRing, IORING_INFO* info)) \
    SDL_IORING_FUNC(HRESULT, SubmitIoRing, (HIORING ioRing, UINT32 waitOperations, UINT32 milliseconds, UINT32* submittedEntries)) \
    SDL_IORING_FUNC(HRESULT, CloseIoRing, (HIORING ioRing)) \
    SDL_IORING_FUNC(HRESULT, PopIoRingCompletion, (HIORING ioRing, IORING_CQE* cqe)) \
    SDL_IORING_FUNC(HRESULT, SetIoRingCompletionEvent, (HIORING ioRing, HANDLE hEvent)) \
    SDL_IORING_FUNC(HRESULT, BuildIoRingCancelRequest, (HIORING ioRing, IORING_HANDLE_REF file, UINT_PTR opToCancel, UINT_PTR userData)) \
    SDL_IORING_FUNC(HRESULT, BuildIoRingReadFile, (HIORING ioRing, IORING_HANDLE_REF fileRef, IORING_BUFFER_REF dataRef, UINT32 numberOfBytesToRead, UINT64 fileOffset, UINT_PTR userData, IORING_SQE_FLAGS sqeFlags)) \
    SDL_IORING_FUNC(HRESULT, BuildIoRingWriteFile, (HIORING ioRing, IORING_HANDLE_REF fileRef, IORING_BUFFER_REF bufferRef, UINT32 numberOfBytesToWrite, UINT64 fileOffset, FILE_WRITE_FLAGS writeFlags, UINT_PTR userData, IORING_SQE_FLAGS sqeFlags)) \
    SDL_IORING_FUNC(HRESULT, BuildIoRingFlushFile, (HIORING ioRing, IORING_HANDLE_REF fileRef, FILE_FLUSH_MODE flushMode, UINT_PTR userData, IORING_SQE_FLAGS sqeFlags)) \

#define SDL_IORING_FUNC(ret, fn, args) typedef ret (WINAPI *SDL_fntype_##fn) args;
SDL_IORING_FUNCS
#undef SDL_IORING_FUNC

typedef struct SDL_WinIoRingFunctions
{
    #define SDL_IORING_FUNC(ret, fn, args) SDL_fntype_##fn fn;
    SDL_IORING_FUNCS
    #undef SDL_IORING_FUNC
} SDL_WinIoRingFunctions;

static SDL_WinIoRingFunctions ioring;


typedef struct WinIoRingAsyncIOQueueData
{
    SDL_Mutex *sqe_lock;
    SDL_Mutex *cqe_lock;
    HANDLE event;
    HIORING ring;
    SDL_AtomicInt num_waiting;
} WinIoRingAsyncIOQueueData;


static void UnloadWinIoRingLibrary(void)
{
    if (ioring_library) {
        SDL_UnloadObject(ioring_handle);
        ioring_library = NULL;
    }
    SDL_zero(ioring);
}

static bool LoadWinIoRingSyms(void)
{
    #define SDL_IORING_FUNC(ret, fn, args) { \
        ioring.fn = (SDL_fntype_##fn) SDL_LoadFunction(ioring_handle, #fn); \
        if (!ioring.fn) { \
            return false; \
        } \
    }
    SDL_IORING_FUNCS
    #undef SDL_IORING_FUNC
    return true;
}

static bool LoadWinIoRing(void)
{
    bool result = true;

    if (!ioring_handle) {
        ioring_handle = SDL_LoadObject(ioring_library);
        if (!ioring_handle) {
            result = false;
            // Don't call SDL_SetError(): SDL_LoadObject already did.
        } else {
            result = LoadWinIoRingSyms();
            if (result) {
                IORING_CAPABILITIES caps;
                HRESULT hr = ioring.QueryIoRingCapabilities(&caps);
                if (FAILED(hr)) {
                    result = false;
                } else if (caps.MaxVersion < SDL_REQUIRED_IORING_VERSION) {
                    result = false;
                }
            }

            if (!result) {
                UnloadWinIoRingLibrary();
            }
        }
    }
    return result;
}

static Sint64 ioring_asyncio_size(void *userdata)
{
    HANDLE handle = (HANDLE) userdata;
    LARGE_INTEGER size;
    if (!GetFileSizeEx(handle, &size)) {
        WIN_SetError("GetFileSizeEx");
        return -1;
    }
    return (Sint64) size.QuadPart;
}

// you must hold sqe_lock when calling this!
static bool ioring_asyncioqueue_queue_task(void *userdata, SDL_AsyncIOTask *task)
{
    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *) userdata;
    const HRESULT hr = ioring.SubmitIoRing(queuedata->ring, 0, 0, NULL);
    return (FAILED(hr) ? WIN_SetErrorFromHRESULT("SubmitIoRing", hr) : true);
}

static void ioring_asyncioqueue_cancel_task(void *userdata, SDL_AsyncIOTask *task)
{
    if (!task->asyncio || !task->asyncio->userdata) {
        return;  // Windows IoRing needs the file handle in question, so we'll just have to let it complete if unknown.
    }

    SDL_AsyncIOTask *cancel_task = (SDL_AsyncIOTask *) SDL_calloc(1, sizeof (*cancel_task));
    if (!cancel_task) {
        return;  // oh well, the task can just finish on its own.
    }

    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *) userdata;
    HANDLE handle = (HANDLE) task->asyncio->userdata;
    IORING_HANDLE_REF href = IoRingHandleRefFromHandle(handle);

    // have to hold a lock because otherwise two threads could get_sqe and submit while one request isn't fully set up.
    SDL_LockMutex(queuedata->sqe_lock);
    const HRESULT hr = ioring.BuildIoRingCancelRequest(queuedata->ring, href, (UINT_PTR) task, (UINT_PTR) cancel_task);
    if (FAILED(hr)) {
        SDL_UnlockMutex(queuedata->sqe_lock);
        SDL_free(cancel_task);  // oh well, the task can just finish on its own.
        return;
    }

    cancel_task->app_userdata = task;
    ioring_asyncioqueue_queue_task(userdata, task);
    SDL_UnlockMutex(queuedata->sqe_lock);
}

static SDL_AsyncIOTask *ProcessCQE(WinIoRingAsyncIOQueueData *queuedata, IORING_CQE *cqe)
{
    if (!cqe) {
        return NULL;
    }

    SDL_AsyncIOTask *task = (SDL_AsyncIOTask *) cqe->UserData;
    if (task) {  // can be NULL if this was just a wakeup message, a NOP, etc.
        if (!task->queue) {  // We leave `queue` blank to signify this was a task cancellation.
            SDL_AsyncIOTask *cancel_task = task;
            task = (SDL_AsyncIOTask *) cancel_task->app_userdata;
            SDL_free(cancel_task);
            if (SUCCEEDED(cqe->ResultCode)) {  // cancel was successful?
                task->result = SDL_ASYNCIO_CANCELED;
            } else {
                task = NULL; // it already finished or was too far along to cancel, so we'll pick up the actual results later.
            }
        } else if (FAILED(cqe->ResultCode)) {
            task->result = SDL_ASYNCIO_FAILURE;
            // !!! FIXME: fill in task->error.
        } else {
            if ((task->type == SDL_ASYNCIO_TASK_WRITE) && (((Uint64) cqe->Information) < task->requested_size)) {
                task->result = SDL_ASYNCIO_FAILURE;  // it's always a failure on short writes.
            }

            // don't explicitly mark it as COMPLETE; that's the default value and a linked task might have failed in an earlier operation and this would overwrite it.

            if ((task->type == SDL_ASYNCIO_TASK_READ) || (task->type == SDL_ASYNCIO_TASK_WRITE)) {
                task->result_size = (Uint64) cqe->Information;
            }
        }

        // we currently send all close operations through as flushes, requested or not, so the actually closing is (in theory) fast. We do that here.
        // if a later IoRing interface version offers an asynchronous close operation, revisit this to only flush if requested, like we do in the Linux io_uring code.
        if (task->type == SDL_ASYNCIO_TASK_CLOSE) {
            SDL_assert(task->asyncio != NULL);
            SDL_assert(task->asyncio->userdata != NULL);
            HANDLE handle = (HANDLE) task->asyncio->userdata;
            if (!CloseHandle(handle)) {
                task->result = SDL_ASYNCIO_FAILURE;  // shrug.
            }
        }
    }

    return task;
}

static SDL_AsyncIOTask *ioring_asyncioqueue_get_results(void *userdata)
{
    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *) userdata;

    // unlike liburing's io_uring_peek_cqe(), it's possible PopIoRingCompletion() is thread safe, but for now we wrap it in a mutex just in case.
    SDL_LockMutex(queuedata->cqe_lock);
    IORING_CQE cqe;
    const HRESULT hr = ioring.PopIoRingCompletion(queuedata->ring, &cqe);
    SDL_UnlockMutex(queuedata->cqe_lock);

    if ((hr == S_FALSE) || FAILED(hr)) {
        return NULL;  // nothing available at the moment.
    }

    return ProcessCQE(queuedata, &cqe);
}

static SDL_AsyncIOTask *ioring_asyncioqueue_wait_results(void *userdata, Sint32 timeoutMS)
{
    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *) userdata;

    // the event only signals when the IoRing moves from empty to non-empty, so you have to try a (non-blocking) get_results first or risk eternal hangs.
    SDL_AsyncIOTask *task = ioring_asyncioqueue_get_results(userdata);
    if (!task) {
        SDL_AddAtomicInt(&queuedata->num_waiting, 1);
        WaitForSingleObject(queuedata->event, (timeoutMS < 0) ? INFINITE : (DWORD) timeoutMS);
        SDL_AddAtomicInt(&queuedata->num_waiting, -1);

        // (we don't care if the wait failed for any reason, as the upcoming get_results will report valid information. We just wanted the wait operation to block.)
        task = ioring_asyncioqueue_get_results(userdata);
    }

    return task;
}

static void ioring_asyncioqueue_signal(void *userdata)
{
    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *) userdata;
    const int num_waiting = SDL_GetAtomicInt(&queuedata->num_waiting);
    for (int i = 0; i < num_waiting; i++) {
        SetEvent(queuedata->event);
    }
}

static void ioring_asyncioqueue_destroy(void *userdata)
{
    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *) userdata;
    ioring.CloseIoRing(queuedata->ring);
    CloseHandle(queuedata->event);
    SDL_DestroyMutex(queuedata->sqe_lock);
    SDL_DestroyMutex(queuedata->cqe_lock);
    SDL_free(queuedata);
}

static bool SDL_SYS_CreateAsyncIOQueue_ioring(SDL_AsyncIOQueue *queue)
{
    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *) SDL_calloc(1, sizeof (*queuedata));
    if (!queuedata) {
        return false;
    }

    HRESULT hr;
    IORING_CREATE_FLAGS flags;

    SDL_SetAtomicInt(&queuedata->num_waiting, 0);

    queuedata->sqe_lock = SDL_CreateMutex();
    if (!queuedata->sqe_lock) {
        goto failed;
    }

    queuedata->cqe_lock = SDL_CreateMutex();
    if (!queuedata->cqe_lock) {
        goto failed;
    }

    queuedata->event = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!queuedata->event) {
        WIN_SetError("CreateEventW");
        goto failed;
    }

    // !!! FIXME: no idea how large the queue should be. Is 128 overkill or too small?
    flags.Required = IORING_CREATE_REQUIRED_FLAGS_NONE;
    flags.Advisory = IORING_CREATE_ADVISORY_FLAGS_NONE;
    hr = ioring.CreateIoRing(SDL_REQUIRED_IORING_VERSION, flags, 128, 128, &queuedata->ring);
    if (FAILED(hr)) {
        WIN_SetErrorFromHRESULT("CreateIoRing", hr);
        goto failed;
    }

    hr = ioring.SetIoRingCompletionEvent(queuedata->ring, queuedata->event);
    if (FAILED(hr)) {
        WIN_SetErrorFromHRESULT("SetIoRingCompletionEvent", hr);
        goto failed;
    }

    static const IORING_OP_CODE needed_ops[] = {
        IORING_OP_NOP,
        IORING_OP_FLUSH,
        IORING_OP_READ,
        IORING_OP_WRITE,
        IORING_OP_CANCEL
    };

    for (int i = 0; i < SDL_arraysize(needed_ops); i++) {
        if (!ioring.IsIoRingOpSupported(queuedata->ring, needed_ops[i])) {
            SDL_SetError("Created IoRing doesn't support op %u", (unsigned int) needed_ops[i]);
            goto failed;
        }
    }

    static const SDL_AsyncIOQueueInterface SDL_AsyncIOQueue_ioring = {
        ioring_asyncioqueue_queue_task,
        ioring_asyncioqueue_cancel_task,
        ioring_asyncioqueue_get_results,
        ioring_asyncioqueue_wait_results,
        ioring_asyncioqueue_signal,
        ioring_asyncioqueue_destroy
    };

    SDL_copyp(&queue->iface, &SDL_AsyncIOQueue_ioring);
    queue->userdata = queuedata;
    return true;

failed:
    if (queuedata->ring) {
        ioring.CloseIoRing(queuedata->ring);
    }
    if (queuedata->event) {
        CloseHandle(queuedata->event);
    }
    if (queuedata->sqe_lock) {
        SDL_DestroyMutex(queuedata->sqe_lock);
    }
    if (queuedata->cqe_lock) {
        SDL_DestroyMutex(queuedata->cqe_lock);
    }
    SDL_free(queuedata);
    return false;
}

static bool ioring_asyncio_read(void *userdata, SDL_AsyncIOTask *task)
{
    // !!! FIXME: UINT32 smaller than requested_size's Uint64. If we overflow it, we could try submitting multiple SQEs
    // !!! FIXME:  and make a note in the task that there are several in sequence.
    if (task->requested_size > 0xFFFFFFFF) {
        return SDL_SetError("ioring: i/o task is too large");
    }

    HANDLE handle = (HANDLE) userdata;
    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *) task->queue->userdata;
    IORING_HANDLE_REF href = IoRingHandleRefFromHandle(handle);
    IORING_BUFFER_REF bref = IoRingBufferRefFromPointer(task->buffer);

    // have to hold a lock because otherwise two threads could get_sqe and submit while one request isn't fully set up.
    SDL_LockMutex(queuedata->sqe_lock);
    bool retval;
    const HRESULT hr = ioring.BuildIoRingReadFile(queuedata->ring, href, bref, (UINT32) task->requested_size, task->offset, (UINT_PTR) task, IOSQE_FLAGS_NONE);
    if (FAILED(hr)) {
        retval = WIN_SetErrorFromHRESULT("BuildIoRingReadFile", hr);
    } else {
        retval = task->queue->iface.queue_task(task->queue->userdata, task);
    }
    SDL_UnlockMutex(queuedata->sqe_lock);
    return retval;
}

static bool ioring_asyncio_write(void *userdata, SDL_AsyncIOTask *task)
{
    // !!! FIXME: UINT32 smaller than requested_size's Uint64. If we overflow it, we could try submitting multiple SQEs
    // !!! FIXME:  and make a note in the task that there are several in sequence.
    if (task->requested_size > 0xFFFFFFFF) {
        return SDL_SetError("ioring: i/o task is too large");
    }

    HANDLE handle = (HANDLE) userdata;
    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *) task->queue->userdata;
    IORING_HANDLE_REF href = IoRingHandleRefFromHandle(handle);
    IORING_BUFFER_REF bref = IoRingBufferRefFromPointer(task->buffer);

    // have to hold a lock because otherwise two threads could get_sqe and submit while one request isn't fully set up.
    SDL_LockMutex(queuedata->sqe_lock);
    bool retval;
    const HRESULT hr = ioring.BuildIoRingWriteFile(queuedata->ring, href, bref, (UINT32) task->requested_size, task->offset, 0 /*FILE_WRITE_FLAGS_NONE*/, (UINT_PTR) task, IOSQE_FLAGS_NONE);
    if (FAILED(hr)) {
        retval = WIN_SetErrorFromHRESULT("BuildIoRingWriteFile", hr);
    } else {
        retval = task->queue->iface.queue_task(task->queue->userdata, task);
    }
    SDL_UnlockMutex(queuedata->sqe_lock);
    return retval;
}

static bool ioring_asyncio_close(void *userdata, SDL_AsyncIOTask *task)
{
    // current IoRing operations don't offer asynchronous closing, but let's assume most of the potential work is flushing to disk, so just do it for everything, explicit flush or not. We'll close when it finishes.
    HANDLE handle = (HANDLE) userdata;
    WinIoRingAsyncIOQueueData *queuedata = (WinIoRingAsyncIOQueueData *)task->queue->userdata;
    IORING_HANDLE_REF href = IoRingHandleRefFromHandle(handle);

    // have to hold a lock because otherwise two threads could get_sqe and submit while one request isn't fully set up.
    SDL_LockMutex(queuedata->sqe_lock);
    bool retval;
    const HRESULT hr = ioring.BuildIoRingFlushFile(queuedata->ring, href, FILE_FLUSH_DEFAULT, (UINT_PTR) task, IOSQE_FLAGS_NONE);
    if (FAILED(hr)) {
        retval = WIN_SetErrorFromHRESULT("BuildIoRingFlushFile", hr);
    } else {
        retval = task->queue->iface.queue_task(task->queue->userdata, task);
    }
    SDL_UnlockMutex(queuedata->sqe_lock);
    return retval;
}

static void ioring_asyncio_destroy(void *userdata)
{
    // this is only a Win32 file HANDLE, should have been closed elsewhere.
}

static bool Win32OpenModeFromString(const char *mode, DWORD *access_mode, DWORD *create_mode)
{
    // this is exactly the set of strings that SDL_AsyncIOFromFile promises will work.
    static const struct { const char *str; DWORD amode; WORD cmode; } mappings[] = {
        { "rb", GENERIC_READ, OPEN_EXISTING },
        { "wb", GENERIC_WRITE, CREATE_ALWAYS },
        { "r+b", GENERIC_READ | GENERIC_WRITE, OPEN_EXISTING },
        { "w+b", GENERIC_READ | GENERIC_WRITE, CREATE_ALWAYS }
    };

    for (int i = 0; i < SDL_arraysize(mappings); i++) {
        if (SDL_strcmp(mappings[i].str, mode) == 0) {
            *access_mode = mappings[i].amode;
            *create_mode = mappings[i].cmode;
            return true;
        }
    }

    SDL_assert(!"Shouldn't have reached this code");
    return SDL_SetError("Invalid file open mode");
}

static bool SDL_SYS_AsyncIOFromFile_ioring(const char *file, const char *mode, SDL_AsyncIO *asyncio)
{
    DWORD access_mode, create_mode;
    if (!Win32OpenModeFromString(mode, &access_mode, &create_mode)) {
        return false;
    }

    LPWSTR wstr = WIN_UTF8ToStringW(file);
    if (!wstr) {
        return false;
    }

    HANDLE handle = CreateFileW(wstr, access_mode, FILE_SHARE_READ, NULL, create_mode, FILE_ATTRIBUTE_NORMAL, NULL);
    SDL_free(wstr);
    if (!handle) {
        return WIN_SetError("CreateFileW");
    }

    static const SDL_AsyncIOInterface SDL_AsyncIOFile_ioring = {
        ioring_asyncio_size,
        ioring_asyncio_read,
        ioring_asyncio_write,
        ioring_asyncio_close,
        ioring_asyncio_destroy
    };

    SDL_copyp(&asyncio->iface, &SDL_AsyncIOFile_ioring);

    asyncio->userdata = (void *) handle;
    return true;
}

static void SDL_SYS_QuitAsyncIO_ioring(void)
{
    UnloadWinIoRingLibrary();
}

static void MaybeInitializeWinIoRing(void)
{
    if (SDL_ShouldInit(&ioring_init)) {
        if (LoadWinIoRing()) {
            CreateAsyncIOQueue = SDL_SYS_CreateAsyncIOQueue_ioring;
            QuitAsyncIO = SDL_SYS_QuitAsyncIO_ioring;
            AsyncIOFromFile = SDL_SYS_AsyncIOFromFile_ioring;
        } else {  // can't use ioring? Use the "generic" threadpool implementation instead.
            CreateAsyncIOQueue = SDL_SYS_CreateAsyncIOQueue_Generic;
            QuitAsyncIO = SDL_SYS_QuitAsyncIO_Generic;
            AsyncIOFromFile = SDL_SYS_AsyncIOFromFile_Generic;
        }
        SDL_SetInitialized(&ioring_init, true);
    }
}

bool SDL_SYS_CreateAsyncIOQueue(SDL_AsyncIOQueue *queue)
{
    MaybeInitializeWinIoRing();
    return CreateAsyncIOQueue(queue);
}

bool SDL_SYS_AsyncIOFromFile(const char *file, const char *mode, SDL_AsyncIO *asyncio)
{
    MaybeInitializeWinIoRing();
    return AsyncIOFromFile(file, mode, asyncio);
}

void SDL_SYS_QuitAsyncIO(void)
{
    if (SDL_ShouldQuit(&ioring_init)) {
        QuitAsyncIO();
        CreateAsyncIOQueue = NULL;
        QuitAsyncIO = NULL;
        AsyncIOFromFile = NULL;
        SDL_SetInitialized(&ioring_init, false);
    }
}

#endif  // defined HAVE_IORINGAPI_H

