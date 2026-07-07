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

#include "../../core/windows/SDL_windows.h"

typedef SDL_Mutex *(*pfnSDL_CreateMutex)(void);
typedef void (*pfnSDL_LockMutex)(SDL_Mutex *);
typedef bool (*pfnSDL_TryLockMutex)(SDL_Mutex *);
typedef void (*pfnSDL_UnlockMutex)(SDL_Mutex *);
typedef void (*pfnSDL_DestroyMutex)(SDL_Mutex *);

typedef enum
{
    SDL_MUTEX_INVALID = 0,
    SDL_MUTEX_SRW,
    SDL_MUTEX_CS,
} SDL_MutexType;

typedef struct SDL_mutex_impl_t
{
    pfnSDL_CreateMutex Create;
    pfnSDL_DestroyMutex Destroy;
    pfnSDL_LockMutex Lock;
    pfnSDL_TryLockMutex TryLock;
    pfnSDL_UnlockMutex Unlock;
    // Needed by SDL_Condition:
    SDL_MutexType Type;
} SDL_mutex_impl_t;

extern SDL_mutex_impl_t SDL_mutex_impl_active;

#ifndef SRWLOCK_INIT
#define SRWLOCK_INIT \
    {                \
        0            \
    }
typedef struct _SRWLOCK
{
    PVOID Ptr;
} SRWLOCK, *PSRWLOCK;
#endif

typedef struct SDL_mutex_srw
{
    SRWLOCK srw;
    // SRW Locks are not recursive, that has to be handled by SDL:
    DWORD count;
    DWORD owner;
} SDL_mutex_srw;

typedef struct SDL_mutex_cs
{
    CRITICAL_SECTION cs;
} SDL_mutex_cs;
