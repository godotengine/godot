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

// System independent thread management routines for SDL

#include "SDL_thread_c.h"
#include "SDL_systhread.h"
#include "../SDL_error_c.h"

// The storage is local to the thread, but the IDs are global for the process

static SDL_AtomicInt SDL_tls_allocated;
static SDL_AtomicInt SDL_tls_id;

void SDL_InitTLSData(void)
{
    SDL_SYS_InitTLSData();
}

void *SDL_GetTLS(SDL_TLSID *id)
{
    SDL_TLSData *storage;
    int storage_index;

    if (id == NULL) {
        SDL_InvalidParamError("id");
        return NULL;
    }

    storage_index = SDL_GetAtomicInt(id) - 1;
    storage = SDL_SYS_GetTLSData();
    if (!storage || storage_index < 0 || storage_index >= storage->limit) {
        return NULL;
    }
    return storage->array[storage_index].data;
}

bool SDL_SetTLS(SDL_TLSID *id, const void *value, SDL_TLSDestructorCallback destructor)
{
    SDL_TLSData *storage;
    int storage_index;

    if (id == NULL) {
        return SDL_InvalidParamError("id");
    }

    /* Make sure TLS is initialized.
     * There's a race condition here if you are calling this from non-SDL threads
     * and haven't called SDL_Init() on your main thread, but such is life.
     */
    SDL_InitTLSData();

    // Get the storage index associated with the ID in a thread-safe way
    storage_index = SDL_GetAtomicInt(id) - 1;
    if (storage_index < 0) {
        int new_id = (SDL_AtomicIncRef(&SDL_tls_id) + 1);

        SDL_CompareAndSwapAtomicInt(id, 0, new_id);

        /* If there was a race condition we'll have wasted an ID, but every thread
         * will have the same storage index for this id.
         */
        storage_index = SDL_GetAtomicInt(id) - 1;
    } else {
        // Make sure we don't allocate an ID clobbering this one
        int tls_id = SDL_GetAtomicInt(&SDL_tls_id);
        while (storage_index >= tls_id) {
            if (SDL_CompareAndSwapAtomicInt(&SDL_tls_id, tls_id, storage_index + 1)) {
                break;
            }
            tls_id = SDL_GetAtomicInt(&SDL_tls_id);
        }
    }

    // Get the storage for the current thread
    storage = SDL_SYS_GetTLSData();
    if (!storage || storage_index >= storage->limit) {
        unsigned int i, oldlimit, newlimit;
        SDL_TLSData *new_storage;

        oldlimit = storage ? storage->limit : 0;
        newlimit = (storage_index + TLS_ALLOC_CHUNKSIZE);
        new_storage = (SDL_TLSData *)SDL_realloc(storage, sizeof(*storage) + (newlimit - 1) * sizeof(storage->array[0]));
        if (!new_storage) {
            return false;
        }
        storage = new_storage;
        storage->limit = newlimit;
        for (i = oldlimit; i < newlimit; ++i) {
            storage->array[i].data = NULL;
            storage->array[i].destructor = NULL;
        }
        if (!SDL_SYS_SetTLSData(storage)) {
            SDL_free(storage);
            return false;
        }
        SDL_AtomicIncRef(&SDL_tls_allocated);
    }

    storage->array[storage_index].data = SDL_const_cast(void *, value);
    storage->array[storage_index].destructor = destructor;
    return true;
}

void SDL_CleanupTLS(void)
{
    SDL_TLSData *storage;

    // Cleanup the storage for the current thread
    storage = SDL_SYS_GetTLSData();
    if (storage) {
        int i;
        for (i = 0; i < storage->limit; ++i) {
            if (storage->array[i].destructor) {
                storage->array[i].destructor(storage->array[i].data);
            }
        }
        SDL_SYS_SetTLSData(NULL);
        SDL_free(storage);
        (void)SDL_AtomicDecRef(&SDL_tls_allocated);
    }
}

void SDL_QuitTLSData(void)
{
    SDL_CleanupTLS();

    if (SDL_GetAtomicInt(&SDL_tls_allocated) == 0) {
        SDL_SYS_QuitTLSData();
    } else {
        // Some thread hasn't called SDL_CleanupTLS()
    }
}

/* This is a generic implementation of thread-local storage which doesn't
   require additional OS support.

   It is not especially efficient and doesn't clean up thread-local storage
   as threads exit.  If there is a real OS that doesn't support thread-local
   storage this implementation should be improved to be production quality.
*/

typedef struct SDL_TLSEntry
{
    SDL_ThreadID thread;
    SDL_TLSData *storage;
    struct SDL_TLSEntry *next;
} SDL_TLSEntry;

static SDL_Mutex *SDL_generic_TLS_mutex;
static SDL_TLSEntry *SDL_generic_TLS;

void SDL_Generic_InitTLSData(void)
{
    if (!SDL_generic_TLS_mutex) {
        SDL_generic_TLS_mutex = SDL_CreateMutex();
    }
}

SDL_TLSData *SDL_Generic_GetTLSData(void)
{
    SDL_ThreadID thread = SDL_GetCurrentThreadID();
    SDL_TLSEntry *entry;
    SDL_TLSData *storage = NULL;

    SDL_LockMutex(SDL_generic_TLS_mutex);
    for (entry = SDL_generic_TLS; entry; entry = entry->next) {
        if (entry->thread == thread) {
            storage = entry->storage;
            break;
        }
    }
    SDL_UnlockMutex(SDL_generic_TLS_mutex);

    return storage;
}

bool SDL_Generic_SetTLSData(SDL_TLSData *data)
{
    SDL_ThreadID thread = SDL_GetCurrentThreadID();
    SDL_TLSEntry *prev, *entry;
    bool result = true;

    SDL_LockMutex(SDL_generic_TLS_mutex);
    prev = NULL;
    for (entry = SDL_generic_TLS; entry; entry = entry->next) {
        if (entry->thread == thread) {
            if (data) {
                entry->storage = data;
            } else {
                if (prev) {
                    prev->next = entry->next;
                } else {
                    SDL_generic_TLS = entry->next;
                }
                SDL_free(entry);
            }
            break;
        }
        prev = entry;
    }
    if (!entry && data) {
        entry = (SDL_TLSEntry *)SDL_malloc(sizeof(*entry));
        if (entry) {
            entry->thread = thread;
            entry->storage = data;
            entry->next = SDL_generic_TLS;
            SDL_generic_TLS = entry;
        } else {
            result = false;
        }
    }
    SDL_UnlockMutex(SDL_generic_TLS_mutex);

    return result;
}

void SDL_Generic_QuitTLSData(void)
{
    SDL_TLSEntry *entry;

    // This should have been cleaned up by the time we get here
    SDL_assert(!SDL_generic_TLS);
    if (SDL_generic_TLS) {
        SDL_LockMutex(SDL_generic_TLS_mutex);
        for (entry = SDL_generic_TLS; entry; ) {
            SDL_TLSEntry *next = entry->next;
            SDL_free(entry->storage);
            SDL_free(entry);
            entry = next;
        }
        SDL_generic_TLS = NULL;
        SDL_UnlockMutex(SDL_generic_TLS_mutex);
    }

    if (SDL_generic_TLS_mutex) {
        SDL_DestroyMutex(SDL_generic_TLS_mutex);
        SDL_generic_TLS_mutex = NULL;
    }
}

// Non-thread-safe global error variable
static SDL_error *SDL_GetStaticErrBuf(void)
{
    static SDL_error SDL_global_error;
    static char SDL_global_error_str[128];
    SDL_global_error.str = SDL_global_error_str;
    SDL_global_error.len = sizeof(SDL_global_error_str);
    return &SDL_global_error;
}

#ifndef SDL_THREADS_DISABLED
static void SDLCALL SDL_FreeErrBuf(void *data)
{
    SDL_error *errbuf = (SDL_error *)data;

    if (errbuf->str) {
        errbuf->free_func(errbuf->str);
    }
    errbuf->free_func(errbuf);
}
#endif

// Routine to get the thread-specific error variable
SDL_error *SDL_GetErrBuf(bool create)
{
#ifdef SDL_THREADS_DISABLED
    return SDL_GetStaticErrBuf();
#else
    static SDL_TLSID tls_errbuf;
    SDL_error *errbuf;

    errbuf = (SDL_error *)SDL_GetTLS(&tls_errbuf);
    if (!errbuf) {
        if (!create) {
            return NULL;
        }

        /* Get the original memory functions for this allocation because the lifetime
         * of the error buffer may span calls to SDL_SetMemoryFunctions() by the app
         */
        SDL_realloc_func realloc_func;
        SDL_free_func free_func;
        SDL_GetOriginalMemoryFunctions(NULL, NULL, &realloc_func, &free_func);

        errbuf = (SDL_error *)realloc_func(NULL, sizeof(*errbuf));
        if (!errbuf) {
            return SDL_GetStaticErrBuf();
        }
        SDL_zerop(errbuf);
        errbuf->realloc_func = realloc_func;
        errbuf->free_func = free_func;
        SDL_SetTLS(&tls_errbuf, errbuf, SDL_FreeErrBuf);
    }
    return errbuf;
#endif // SDL_THREADS_DISABLED
}

static bool ThreadValid(SDL_Thread *thread)
{
    return SDL_ObjectValid(thread, SDL_OBJECT_TYPE_THREAD);
}

void SDL_RunThread(SDL_Thread *thread)
{
    void *userdata = thread->userdata;
    int(SDLCALL *userfunc)(void *) = thread->userfunc;

    int *statusloc = &thread->status;

    // Perform any system-dependent setup - this function may not fail
    SDL_SYS_SetupThread(thread->name);

    // Get the thread id
    thread->threadid = SDL_GetCurrentThreadID();

    // Run the function
    *statusloc = userfunc(userdata);

    // Clean up thread-local storage
    SDL_CleanupTLS();

    // Mark us as ready to be joined (or detached)
    if (!SDL_CompareAndSwapAtomicInt(&thread->state, SDL_THREAD_ALIVE, SDL_THREAD_COMPLETE)) {
        // Clean up if something already detached us.
        if (SDL_GetAtomicInt(&thread->state) == SDL_THREAD_DETACHED) {
            SDL_free(thread->name); // Can't free later, we've already cleaned up TLS
            SDL_free(thread);
        }
    }
}

SDL_Thread *SDL_CreateThreadWithPropertiesRuntime(SDL_PropertiesID props,
                              SDL_FunctionPointer pfnBeginThread,
                              SDL_FunctionPointer pfnEndThread)
{
    // rather than check this in every backend, just make sure it's correct upfront. Only allow non-NULL if Windows, or Microsoft GDK.
    #if !defined(SDL_PLATFORM_WINDOWS)
    if (pfnBeginThread || pfnEndThread) {
        SDL_SetError("_beginthreadex/_endthreadex not supported on this platform");
        return NULL;
    }
    #endif

    SDL_ThreadFunction fn = (SDL_ThreadFunction) SDL_GetPointerProperty(props, SDL_PROP_THREAD_CREATE_ENTRY_FUNCTION_POINTER, NULL);
    const char *name = SDL_GetStringProperty(props, SDL_PROP_THREAD_CREATE_NAME_STRING, NULL);
    const size_t stacksize = (size_t) SDL_GetNumberProperty(props, SDL_PROP_THREAD_CREATE_STACKSIZE_NUMBER, 0);
    void *userdata = SDL_GetPointerProperty(props, SDL_PROP_THREAD_CREATE_USERDATA_POINTER, NULL);

    if (!fn) {
        SDL_SetError("Thread entry function is NULL");
        return NULL;
    }

    SDL_InitMainThread();

    SDL_Thread *thread = (SDL_Thread *)SDL_calloc(1, sizeof(*thread));
    if (!thread) {
        return NULL;
    }
    thread->status = -1;
    SDL_SetAtomicInt(&thread->state, SDL_THREAD_ALIVE);

    // Set up the arguments for the thread
    if (name) {
        thread->name = SDL_strdup(name);
        if (!thread->name) {
            SDL_free(thread);
            return NULL;
        }
    }

    thread->userfunc = fn;
    thread->userdata = userdata;
    thread->stacksize = stacksize;

    SDL_SetObjectValid(thread, SDL_OBJECT_TYPE_THREAD, true);

    // Create the thread and go!
    if (!SDL_SYS_CreateThread(thread, pfnBeginThread, pfnEndThread)) {
        // Oops, failed.  Gotta free everything
        SDL_SetObjectValid(thread, SDL_OBJECT_TYPE_THREAD, false);
        SDL_free(thread->name);
        SDL_free(thread);
        thread = NULL;
    }

    // Everything is running now
    return thread;
}

SDL_Thread *SDL_CreateThreadRuntime(SDL_ThreadFunction fn,
                 const char *name, void *userdata,
                 SDL_FunctionPointer pfnBeginThread,
                 SDL_FunctionPointer pfnEndThread)
{
    const SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_THREAD_CREATE_ENTRY_FUNCTION_POINTER, (void *) fn);
    SDL_SetStringProperty(props, SDL_PROP_THREAD_CREATE_NAME_STRING, name);
    SDL_SetPointerProperty(props, SDL_PROP_THREAD_CREATE_USERDATA_POINTER, userdata);
    SDL_Thread *thread = SDL_CreateThreadWithPropertiesRuntime(props, pfnBeginThread, pfnEndThread);
    SDL_DestroyProperties(props);
    return thread;
}

// internal helper function, not in the public API.
SDL_Thread *SDL_CreateThreadWithStackSize(SDL_ThreadFunction fn, const char *name, size_t stacksize, void *userdata)
{
    const SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_THREAD_CREATE_ENTRY_FUNCTION_POINTER, (void *) fn);
    SDL_SetStringProperty(props, SDL_PROP_THREAD_CREATE_NAME_STRING, name);
    SDL_SetPointerProperty(props, SDL_PROP_THREAD_CREATE_USERDATA_POINTER, userdata);
    SDL_SetNumberProperty(props, SDL_PROP_THREAD_CREATE_STACKSIZE_NUMBER, (Sint64) stacksize);
    SDL_Thread *thread = SDL_CreateThreadWithProperties(props);
    SDL_DestroyProperties(props);
    return thread;
}

SDL_ThreadID SDL_GetThreadID(SDL_Thread *thread)
{
    SDL_ThreadID id = 0;

    if (thread) {
        if (ThreadValid(thread)) {
            id = thread->threadid;
        }
    } else {
        id = SDL_GetCurrentThreadID();
    }
    return id;
}

const char *SDL_GetThreadName(SDL_Thread *thread)
{
    if (ThreadValid(thread)) {
        return SDL_GetPersistentString(thread->name);
    } else {
        return NULL;
    }
}

bool SDL_SetCurrentThreadPriority(SDL_ThreadPriority priority)
{
    return SDL_SYS_SetThreadPriority(priority);
}

void SDL_WaitThread(SDL_Thread *thread, int *status)
{
    if (!ThreadValid(thread)) {
        if (status) {
            *status = -1;
        }
        return;
    }

    SDL_SYS_WaitThread(thread);
    if (status) {
        *status = thread->status;
    }
    SDL_SetObjectValid(thread, SDL_OBJECT_TYPE_THREAD, false);
    SDL_free(thread->name);
    SDL_free(thread);
}

SDL_ThreadState SDL_GetThreadState(SDL_Thread *thread)
{
    if (!ThreadValid(thread)) {
        return SDL_THREAD_UNKNOWN;
    }

    return (SDL_ThreadState)SDL_GetAtomicInt(&thread->state);
}

void SDL_DetachThread(SDL_Thread *thread)
{
    if (!ThreadValid(thread)) {
        return;
    }

    // Grab dibs if the state is alive+joinable.
    if (SDL_CompareAndSwapAtomicInt(&thread->state, SDL_THREAD_ALIVE, SDL_THREAD_DETACHED)) {
        // The thread may vanish at any time, it's no longer valid
        SDL_SetObjectValid(thread, SDL_OBJECT_TYPE_THREAD, false);
        SDL_SYS_DetachThread(thread);
    } else {
        // all other states are pretty final, see where we landed.
        SDL_ThreadState thread_state = SDL_GetThreadState(thread);
        if (thread_state == SDL_THREAD_DETACHED) {
            return; // already detached (you shouldn't call this twice!)
        } else if (thread_state == SDL_THREAD_COMPLETE) {
            SDL_WaitThread(thread, NULL); // already done, clean it up.
        }
    }
}

void SDL_WaitSemaphore(SDL_Semaphore *sem)
{
    SDL_WaitSemaphoreTimeoutNS(sem, -1);
}

bool SDL_TryWaitSemaphore(SDL_Semaphore *sem)
{
    return SDL_WaitSemaphoreTimeoutNS(sem, 0);
}

bool SDL_WaitSemaphoreTimeout(SDL_Semaphore *sem, Sint32 timeoutMS)
{
    Sint64 timeoutNS;

    if (timeoutMS >= 0) {
        timeoutNS = SDL_MS_TO_NS(timeoutMS);
    } else {
        timeoutNS = -1;
    }
    return SDL_WaitSemaphoreTimeoutNS(sem, timeoutNS);
}

void SDL_WaitCondition(SDL_Condition *cond, SDL_Mutex *mutex)
{
    SDL_WaitConditionTimeoutNS(cond, mutex, -1);
}

bool SDL_WaitConditionTimeout(SDL_Condition *cond, SDL_Mutex *mutex, Sint32 timeoutMS)
{
    Sint64 timeoutNS;

    if (timeoutMS >= 0) {
        timeoutNS = SDL_MS_TO_NS(timeoutMS);
    } else {
        timeoutNS = -1;
    }
    return SDL_WaitConditionTimeoutNS(cond, mutex, timeoutNS);
}

bool SDL_ShouldInit(SDL_InitState *state)
{
    while (SDL_GetAtomicInt(&state->status) != SDL_INIT_STATUS_INITIALIZED) {
        if (SDL_CompareAndSwapAtomicInt(&state->status, SDL_INIT_STATUS_UNINITIALIZED, SDL_INIT_STATUS_INITIALIZING)) {
            state->thread = SDL_GetCurrentThreadID();
            return true;
        }

        // Wait for the other thread to complete transition
        SDL_Delay(1);
    }
    return false;
}

bool SDL_ShouldQuit(SDL_InitState *state)
{
    while (SDL_GetAtomicInt(&state->status) != SDL_INIT_STATUS_UNINITIALIZED) {
        if (SDL_CompareAndSwapAtomicInt(&state->status, SDL_INIT_STATUS_INITIALIZED, SDL_INIT_STATUS_UNINITIALIZING)) {
            state->thread = SDL_GetCurrentThreadID();
            return true;
        }

        // Wait for the other thread to complete transition
        SDL_Delay(1);
    }
    return false;
}

void SDL_SetInitialized(SDL_InitState *state, bool initialized)
{
    SDL_assert(state->thread == SDL_GetCurrentThreadID());

    if (initialized) {
        SDL_SetAtomicInt(&state->status, SDL_INIT_STATUS_INITIALIZED);
    } else {
        SDL_SetAtomicInt(&state->status, SDL_INIT_STATUS_UNINITIALIZED);
    }
}

