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

/**
 * Implementation based on Slim Reader/Writer (SRW) Locks for Win 7 and newer.
 */

// This header makes sure SRWLOCK is actually declared, even on ancient WinSDKs.
#include "SDL_sysmutex_c.h"

typedef VOID(WINAPI *pfnInitializeSRWLock)(PSRWLOCK);
typedef VOID(WINAPI *pfnReleaseSRWLockShared)(PSRWLOCK);
typedef VOID(WINAPI *pfnAcquireSRWLockShared)(PSRWLOCK);
typedef BOOLEAN(WINAPI *pfnTryAcquireSRWLockShared)(PSRWLOCK);
typedef VOID(WINAPI *pfnReleaseSRWLockExclusive)(PSRWLOCK);
typedef VOID(WINAPI *pfnAcquireSRWLockExclusive)(PSRWLOCK);
typedef BOOLEAN(WINAPI *pfnTryAcquireSRWLockExclusive)(PSRWLOCK);

static pfnInitializeSRWLock pInitializeSRWLock = NULL;
static pfnReleaseSRWLockShared pReleaseSRWLockShared = NULL;
static pfnAcquireSRWLockShared pAcquireSRWLockShared = NULL;
static pfnTryAcquireSRWLockShared  pTryAcquireSRWLockShared = NULL;
static pfnReleaseSRWLockExclusive pReleaseSRWLockExclusive = NULL;
static pfnAcquireSRWLockExclusive pAcquireSRWLockExclusive = NULL;
static pfnTryAcquireSRWLockExclusive pTryAcquireSRWLockExclusive = NULL;

typedef SDL_RWLock *(*pfnSDL_CreateRWLock)(void);
typedef void (*pfnSDL_DestroyRWLock)(SDL_RWLock *);
typedef void (*pfnSDL_LockRWLockForReading)(SDL_RWLock *);
typedef void (*pfnSDL_LockRWLockForWriting)(SDL_RWLock *);
typedef bool (*pfnSDL_TryLockRWLockForReading)(SDL_RWLock *);
typedef bool (*pfnSDL_TryLockRWLockForWriting)(SDL_RWLock *);
typedef void (*pfnSDL_UnlockRWLock)(SDL_RWLock *);

typedef struct SDL_rwlock_impl_t
{
    pfnSDL_CreateRWLock Create;
    pfnSDL_DestroyRWLock Destroy;
    pfnSDL_LockRWLockForReading LockForReading;
    pfnSDL_LockRWLockForWriting LockForWriting;
    pfnSDL_TryLockRWLockForReading TryLockForReading;
    pfnSDL_TryLockRWLockForWriting TryLockForWriting;
    pfnSDL_UnlockRWLock Unlock;
} SDL_rwlock_impl_t;

// Implementation will be chosen at runtime based on available Kernel features
static SDL_rwlock_impl_t SDL_rwlock_impl_active = { 0 };

// rwlock implementation using Win7+ slim read/write locks (SRWLOCK)

typedef struct SDL_rwlock_srw
{
    SRWLOCK srw;
    SDL_ThreadID write_owner;
} SDL_rwlock_srw;

static SDL_RWLock *SDL_CreateRWLock_srw(void)
{
    SDL_rwlock_srw *rwlock = (SDL_rwlock_srw *)SDL_calloc(1, sizeof(*rwlock));
    if (rwlock) {
        pInitializeSRWLock(&rwlock->srw);
    }
    return (SDL_RWLock *)rwlock;
}

static void SDL_DestroyRWLock_srw(SDL_RWLock *_rwlock)
{
    SDL_rwlock_srw *rwlock = (SDL_rwlock_srw *) _rwlock;
    // There are no kernel allocated resources
    SDL_free(rwlock);
}

static void SDL_LockRWLockForReading_srw(SDL_RWLock *_rwlock) SDL_NO_THREAD_SAFETY_ANALYSIS  // clang doesn't know about NULL mutexes
{
    SDL_rwlock_srw *rwlock = (SDL_rwlock_srw *) _rwlock;
    pAcquireSRWLockShared(&rwlock->srw);
}

static void SDL_LockRWLockForWriting_srw(SDL_RWLock *_rwlock) SDL_NO_THREAD_SAFETY_ANALYSIS  // clang doesn't know about NULL mutexes
{
    SDL_rwlock_srw *rwlock = (SDL_rwlock_srw *) _rwlock;
    pAcquireSRWLockExclusive(&rwlock->srw);
    rwlock->write_owner = SDL_GetCurrentThreadID();
}

static bool SDL_TryLockRWLockForReading_srw(SDL_RWLock *_rwlock)
{
    SDL_rwlock_srw *rwlock = (SDL_rwlock_srw *) _rwlock;
    return pTryAcquireSRWLockShared(&rwlock->srw);
}

static bool SDL_TryLockRWLockForWriting_srw(SDL_RWLock *_rwlock)
{
    SDL_rwlock_srw *rwlock = (SDL_rwlock_srw *) _rwlock;
    if (pTryAcquireSRWLockExclusive(&rwlock->srw)) {
        rwlock->write_owner = SDL_GetCurrentThreadID();
        return true;
    } else {
        return false;
    }
}

static void SDL_UnlockRWLock_srw(SDL_RWLock *_rwlock) SDL_NO_THREAD_SAFETY_ANALYSIS  // clang doesn't know about NULL mutexes
{
    SDL_rwlock_srw *rwlock = (SDL_rwlock_srw *) _rwlock;
    if (rwlock->write_owner == SDL_GetCurrentThreadID()) {
        rwlock->write_owner = 0;
        pReleaseSRWLockExclusive(&rwlock->srw);
    } else {
        pReleaseSRWLockShared(&rwlock->srw);
    }
}

static const SDL_rwlock_impl_t SDL_rwlock_impl_srw = {
    &SDL_CreateRWLock_srw,
    &SDL_DestroyRWLock_srw,
    &SDL_LockRWLockForReading_srw,
    &SDL_LockRWLockForWriting_srw,
    &SDL_TryLockRWLockForReading_srw,
    &SDL_TryLockRWLockForWriting_srw,
    &SDL_UnlockRWLock_srw
};


#include "../generic/SDL_sysrwlock_c.h"

// Generic rwlock implementation using SDL_Mutex, SDL_Condition, and SDL_AtomicInt
static const SDL_rwlock_impl_t SDL_rwlock_impl_generic = {
    &SDL_CreateRWLock_generic,
    &SDL_DestroyRWLock_generic,
    &SDL_LockRWLockForReading_generic,
    &SDL_LockRWLockForWriting_generic,
    &SDL_TryLockRWLockForReading_generic,
    &SDL_TryLockRWLockForWriting_generic,
    &SDL_UnlockRWLock_generic
};

SDL_RWLock *SDL_CreateRWLock(void)
{
    if (!SDL_rwlock_impl_active.Create) {
        // Default to generic implementation, works with all mutex implementations
        const SDL_rwlock_impl_t *impl = &SDL_rwlock_impl_generic;
        {
            HMODULE kernel32 = GetModuleHandle(TEXT("kernel32.dll"));
            if (kernel32) {
                bool okay = true;
                #define LOOKUP_SRW_SYM(sym) if (okay) { if ((p##sym = (pfn##sym)GetProcAddress(kernel32, #sym)) == NULL) { okay = false; } }
                LOOKUP_SRW_SYM(InitializeSRWLock);
                LOOKUP_SRW_SYM(ReleaseSRWLockShared);
                LOOKUP_SRW_SYM(AcquireSRWLockShared);
                LOOKUP_SRW_SYM(TryAcquireSRWLockShared);
                LOOKUP_SRW_SYM(ReleaseSRWLockExclusive);
                LOOKUP_SRW_SYM(AcquireSRWLockExclusive);
                LOOKUP_SRW_SYM(TryAcquireSRWLockExclusive);
                #undef LOOKUP_SRW_SYM
                if (okay) {
                    impl = &SDL_rwlock_impl_srw;  // Use the Windows provided API instead of generic fallback
                }
            }
        }

        SDL_copyp(&SDL_rwlock_impl_active, impl);
    }
    return SDL_rwlock_impl_active.Create();
}

void SDL_DestroyRWLock(SDL_RWLock *rwlock)
{
    if (rwlock) {
        SDL_rwlock_impl_active.Destroy(rwlock);
    }
}

void SDL_LockRWLockForReading(SDL_RWLock *rwlock) SDL_NO_THREAD_SAFETY_ANALYSIS  // clang doesn't know about NULL mutexes
{
    if (rwlock) {
        SDL_rwlock_impl_active.LockForReading(rwlock);
    }
}

void SDL_LockRWLockForWriting(SDL_RWLock *rwlock) SDL_NO_THREAD_SAFETY_ANALYSIS  // clang doesn't know about NULL mutexes
{
    if (rwlock) {
        SDL_rwlock_impl_active.LockForWriting(rwlock);
    }
}

bool SDL_TryLockRWLockForReading(SDL_RWLock *rwlock)
{
    bool result = true;
    if (rwlock) {
        result = SDL_rwlock_impl_active.TryLockForReading(rwlock);
    }
    return result;
}

bool SDL_TryLockRWLockForWriting(SDL_RWLock *rwlock)
{
    bool result = true;
    if (rwlock) {
        result = SDL_rwlock_impl_active.TryLockForWriting(rwlock);
    }
    return result;
}

void SDL_UnlockRWLock(SDL_RWLock *rwlock) SDL_NO_THREAD_SAFETY_ANALYSIS  // clang doesn't know about NULL mutexes
{
    if (rwlock) {
        SDL_rwlock_impl_active.Unlock(rwlock);
    }
}

