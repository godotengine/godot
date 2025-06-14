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

#ifndef SDL_sysasyncio_h_
#define SDL_sysasyncio_h_

#if defined(SDL_PLATFORM_WINDOWS) && defined(NTDDI_WIN10_NI)
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP) && NTDDI_VERSION >= NTDDI_WIN10_NI
#define HAVE_IORINGAPI_H
#endif
#endif

// If your platform has an option other than the "generic" code, make sure this
// is #defined to 0 instead and implement the SDL_SYS_* functions below in your
// backend (having them maybe call into the SDL_SYS_*_Generic versions as a
// fallback if the platform has functionality that isn't always available).
#if defined(HAVE_LIBURING_H) || defined(HAVE_IORINGAPI_H)
#define SDL_ASYNCIO_ONLY_HAVE_GENERIC 0
#else
#define SDL_ASYNCIO_ONLY_HAVE_GENERIC 1
#endif

// this entire thing is just juggling doubly-linked lists, so make some helper macros.
#define LINKED_LIST_DECLARE_FIELDS(type, prefix) \
    type *prefix##prev; \
    type *prefix##next

#define LINKED_LIST_PREPEND(item, list, prefix) do { \
    item->prefix##prev = &list; \
    item->prefix##next = list.prefix##next; \
    if (item->prefix##next) { \
        item->prefix##next->prefix##prev = item; \
    } \
    list.prefix##next = item; \
} while (false)

#define LINKED_LIST_UNLINK(item, prefix) do { \
    if (item->prefix##next) { \
        item->prefix##next->prefix##prev = item->prefix##prev; \
    } \
    item->prefix##prev->prefix##next = task->prefix##next; \
    item->prefix##prev = item->prefix##next = NULL; \
} while (false)

#define LINKED_LIST_START(list, prefix) (list.prefix##next)
#define LINKED_LIST_NEXT(item, prefix) (item->prefix##next)
#define LINKED_LIST_PREV(item, prefix) (item->prefix##prev)

typedef struct SDL_AsyncIOTask SDL_AsyncIOTask;

struct SDL_AsyncIOTask
{
    SDL_AsyncIO *asyncio;
    SDL_AsyncIOTaskType type;
    SDL_AsyncIOQueue *queue;
    Uint64 offset;
    bool flush;
    void *buffer;
    char *error;
    SDL_AsyncIOResult result;
    Uint64 requested_size;
    Uint64 result_size;
    void *app_userdata;
    LINKED_LIST_DECLARE_FIELDS(struct SDL_AsyncIOTask, asyncio);
    LINKED_LIST_DECLARE_FIELDS(struct SDL_AsyncIOTask, queue);      // the generic backend uses this, so I've added it here to avoid the extra allocation.
    LINKED_LIST_DECLARE_FIELDS(struct SDL_AsyncIOTask, threadpool); // the generic backend uses this, so I've added it here to avoid the extra allocation.
};

typedef struct SDL_AsyncIOQueueInterface
{
    bool (*queue_task)(void *userdata, SDL_AsyncIOTask *task);
    void (*cancel_task)(void *userdata, SDL_AsyncIOTask *task);
    SDL_AsyncIOTask * (*get_results)(void *userdata);
    SDL_AsyncIOTask * (*wait_results)(void *userdata, Sint32 timeoutMS);
    void (*signal)(void *userdata);
    void (*destroy)(void *userdata);
} SDL_AsyncIOQueueInterface;

struct SDL_AsyncIOQueue
{
    SDL_AsyncIOQueueInterface iface;
    void *userdata;
    SDL_AtomicInt tasks_inflight;
};

// this interface is kept per-object, even though generally it's going to decide
// on a single interface that is the same for the entire process, but I've kept
// the abstraction in case we start exposing more types of async i/o, like
// sockets, in the future.
typedef struct SDL_AsyncIOInterface
{
    Sint64 (*size)(void *userdata);
    bool (*read)(void *userdata, SDL_AsyncIOTask *task);
    bool (*write)(void *userdata, SDL_AsyncIOTask *task);
    bool (*close)(void *userdata, SDL_AsyncIOTask *task);
    void (*destroy)(void *userdata);
} SDL_AsyncIOInterface;

struct SDL_AsyncIO
{
    SDL_AsyncIOInterface iface;
    void *userdata;
    SDL_Mutex *lock;
    SDL_AsyncIOTask tasks;
    SDL_AsyncIOTask *closing;  // The close task, which isn't queued until all pending work for this file is done.
    bool oneshot;  // true if this is a SDL_LoadFileAsync open.
};

// This is implemented for various platforms; param validation is done before calling this. Open file, fill in iface and userdata.
extern bool SDL_SYS_AsyncIOFromFile(const char *file, const char *mode, SDL_AsyncIO *asyncio);

// This is implemented for various platforms. Call SDL_OpenAsyncIOQueue from in here.
extern bool SDL_SYS_CreateAsyncIOQueue(SDL_AsyncIOQueue *queue);

// This is called during SDL_QuitAsyncIO, after all tasks have completed and all files are closed, to let the platform clean up global backend details.
extern void SDL_SYS_QuitAsyncIO(void);

// the "generic" version is always available, since it is almost always needed as a fallback even on platforms that might offer something better.
extern bool SDL_SYS_AsyncIOFromFile_Generic(const char *file, const char *mode, SDL_AsyncIO *asyncio);
extern bool SDL_SYS_CreateAsyncIOQueue_Generic(SDL_AsyncIOQueue *queue);
extern void SDL_SYS_QuitAsyncIO_Generic(void);

#endif

