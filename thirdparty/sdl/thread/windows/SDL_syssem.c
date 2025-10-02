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

#ifdef SDL_THREAD_WINDOWS

/**
 * Semaphore functions using the Win32 API
 * There are two implementations available based on:
 * - Kernel Semaphores. Available on all OS versions. (kern)
 *   Heavy-weight inter-process kernel objects.
 * - Atomics and WaitOnAddress API. (atom)
 *   Faster due to significantly less context switches.
 *   Requires Windows 8 or newer.
 * which are chosen at runtime.
 */

#include "../../core/windows/SDL_windows.h"

typedef SDL_Semaphore *(*pfnSDL_CreateSemaphore)(Uint32);
typedef void (*pfnSDL_DestroySemaphore)(SDL_Semaphore *);
typedef bool (*pfnSDL_WaitSemaphoreTimeoutNS)(SDL_Semaphore *, Sint64);
typedef Uint32 (*pfnSDL_GetSemaphoreValue)(SDL_Semaphore *);
typedef void (*pfnSDL_SignalSemaphore)(SDL_Semaphore *);

typedef struct SDL_semaphore_impl_t
{
    pfnSDL_CreateSemaphore Create;
    pfnSDL_DestroySemaphore Destroy;
    pfnSDL_WaitSemaphoreTimeoutNS WaitTimeoutNS;
    pfnSDL_GetSemaphoreValue Value;
    pfnSDL_SignalSemaphore Signal;
} SDL_sem_impl_t;

// Implementation will be chosen at runtime based on available Kernel features
static SDL_sem_impl_t SDL_sem_impl_active = { 0 };

/**
 * Atomic + WaitOnAddress implementation
 */

// APIs not available on WinPhone 8.1
// https://www.microsoft.com/en-us/download/details.aspx?id=47328

typedef BOOL(WINAPI *pfnWaitOnAddress)(volatile VOID *, PVOID, SIZE_T, DWORD);
typedef VOID(WINAPI *pfnWakeByAddressSingle)(PVOID);

static pfnWaitOnAddress pWaitOnAddress = NULL;
static pfnWakeByAddressSingle pWakeByAddressSingle = NULL;

typedef struct SDL_semaphore_atom
{
    LONG count;
} SDL_sem_atom;

static SDL_Semaphore *SDL_CreateSemaphore_atom(Uint32 initial_value)
{
    SDL_sem_atom *sem;

    sem = (SDL_sem_atom *)SDL_malloc(sizeof(*sem));
    if (sem) {
        sem->count = initial_value;
    }
    return (SDL_Semaphore *)sem;
}

static void SDL_DestroySemaphore_atom(SDL_Semaphore *sem)
{
    SDL_free(sem);
}

static bool SDL_WaitSemaphoreTimeoutNS_atom(SDL_Semaphore *_sem, Sint64 timeoutNS)
{
    SDL_sem_atom *sem = (SDL_sem_atom *)_sem;
    LONG count;
    Uint64 now;
    Uint64 deadline;
    DWORD timeout_eff;

    if (!sem) {
        return true;
    }

    if (timeoutNS == 0) {
        count = sem->count;
        if (count == 0) {
            return false;
        }

        if (InterlockedCompareExchange(&sem->count, count - 1, count) == count) {
            return true;
        }

        return false;
    }

    if (timeoutNS < 0) {
        for (;;) {
            count = sem->count;
            while (count == 0) {
                if (!pWaitOnAddress(&sem->count, &count, sizeof(sem->count), INFINITE)) {
                    return false;
                }
                count = sem->count;
            }

            if (InterlockedCompareExchange(&sem->count, count - 1, count) == count) {
                return true;
            }
        }
    }

    /**
     * WaitOnAddress is subject to spurious and stolen wakeups so we
     * need to recalculate the effective timeout before every wait
     */
    now = SDL_GetTicksNS();
    deadline = now + timeoutNS;

    for (;;) {
        count = sem->count;
        // If no semaphore is available we need to wait
        while (count == 0) {
            now = SDL_GetTicksNS();
            if (deadline > now) {
                timeout_eff = (DWORD)SDL_NS_TO_MS(deadline - now);
            } else {
                return false;
            }
            if (!pWaitOnAddress(&sem->count, &count, sizeof(count), timeout_eff)) {
                return false;
            }
            count = sem->count;
        }

        // Actually the semaphore is only consumed if this succeeds
        // If it doesn't we need to do everything again
        if (InterlockedCompareExchange(&sem->count, count - 1, count) == count) {
            return true;
        }
    }
}

static Uint32 SDL_GetSemaphoreValue_atom(SDL_Semaphore *_sem)
{
    SDL_sem_atom *sem = (SDL_sem_atom *)_sem;

    if (!sem) {
        return 0;
    }

    return (Uint32)sem->count;
}

static void SDL_SignalSemaphore_atom(SDL_Semaphore *_sem)
{
    SDL_sem_atom *sem = (SDL_sem_atom *)_sem;

    if (!sem) {
        return;
    }

    InterlockedIncrement(&sem->count);
    pWakeByAddressSingle(&sem->count);
}

static const SDL_sem_impl_t SDL_sem_impl_atom = {
    &SDL_CreateSemaphore_atom,
    &SDL_DestroySemaphore_atom,
    &SDL_WaitSemaphoreTimeoutNS_atom,
    &SDL_GetSemaphoreValue_atom,
    &SDL_SignalSemaphore_atom,
};

/**
 * Fallback Semaphore implementation using Kernel Semaphores
 */

typedef struct SDL_semaphore_kern
{
    HANDLE id;
    LONG count;
} SDL_sem_kern;

// Create a semaphore
static SDL_Semaphore *SDL_CreateSemaphore_kern(Uint32 initial_value)
{
    SDL_sem_kern *sem;

    // Allocate sem memory
    sem = (SDL_sem_kern *)SDL_malloc(sizeof(*sem));
    if (sem) {
        // Create the semaphore, with max value 32K
        sem->id = CreateSemaphore(NULL, initial_value, 32 * 1024, NULL);
        sem->count = initial_value;
        if (!sem->id) {
            SDL_SetError("Couldn't create semaphore");
            SDL_free(sem);
            sem = NULL;
        }
    }
    return (SDL_Semaphore *)sem;
}

// Free the semaphore
static void SDL_DestroySemaphore_kern(SDL_Semaphore *_sem)
{
    SDL_sem_kern *sem = (SDL_sem_kern *)_sem;
    if (sem) {
        if (sem->id) {
            CloseHandle(sem->id);
            sem->id = 0;
        }
        SDL_free(sem);
    }
}

static bool SDL_WaitSemaphoreTimeoutNS_kern(SDL_Semaphore *_sem, Sint64 timeoutNS)
{
    SDL_sem_kern *sem = (SDL_sem_kern *)_sem;
    DWORD dwMilliseconds;

    if (!sem) {
        return true;
    }

    if (timeoutNS < 0) {
        dwMilliseconds = INFINITE;
    } else {
        dwMilliseconds = (DWORD)SDL_NS_TO_MS(timeoutNS);
    }
    switch (WaitForSingleObjectEx(sem->id, dwMilliseconds, FALSE)) {
    case WAIT_OBJECT_0:
        InterlockedDecrement(&sem->count);
        return true;
    default:
        return false;
    }
}

// Returns the current count of the semaphore
static Uint32 SDL_GetSemaphoreValue_kern(SDL_Semaphore *_sem)
{
    SDL_sem_kern *sem = (SDL_sem_kern *)_sem;
    if (!sem) {
        return 0;
    }
    return (Uint32)sem->count;
}

static void SDL_SignalSemaphore_kern(SDL_Semaphore *_sem)
{
    SDL_sem_kern *sem = (SDL_sem_kern *)_sem;

    if (!sem) {
        return;
    }

    /* Increase the counter in the first place, because
     * after a successful release the semaphore may
     * immediately get destroyed by another thread which
     * is waiting for this semaphore.
     */
    InterlockedIncrement(&sem->count);
    if (ReleaseSemaphore(sem->id, 1, NULL) == FALSE) {
        InterlockedDecrement(&sem->count); // restore
    }
}

static const SDL_sem_impl_t SDL_sem_impl_kern = {
    &SDL_CreateSemaphore_kern,
    &SDL_DestroySemaphore_kern,
    &SDL_WaitSemaphoreTimeoutNS_kern,
    &SDL_GetSemaphoreValue_kern,
    &SDL_SignalSemaphore_kern,
};

/**
 * Runtime selection and redirection
 */

SDL_Semaphore *SDL_CreateSemaphore(Uint32 initial_value)
{
    if (!SDL_sem_impl_active.Create) {
        // Default to fallback implementation
        const SDL_sem_impl_t *impl = &SDL_sem_impl_kern;

        if (!SDL_GetHintBoolean(SDL_HINT_WINDOWS_FORCE_SEMAPHORE_KERNEL, false)) {
            /* We already statically link to features from this Api
             * Set (e.g. WaitForSingleObject). Dynamically loading
             * API Sets is not explicitly documented but according to
             * Microsoft our specific use case is legal and correct:
             * https://github.com/microsoft/STL/pull/593#issuecomment-655799859
             */
            HMODULE synch120 = GetModuleHandle(TEXT("api-ms-win-core-synch-l1-2-0.dll"));
            if (synch120) {
                // Try to load required functions provided by Win 8 or newer
                pWaitOnAddress = (pfnWaitOnAddress)GetProcAddress(synch120, "WaitOnAddress");
                pWakeByAddressSingle = (pfnWakeByAddressSingle)GetProcAddress(synch120, "WakeByAddressSingle");

                if (pWaitOnAddress && pWakeByAddressSingle) {
                    impl = &SDL_sem_impl_atom;
                }
            }
        }

        // Copy instead of using pointer to save one level of indirection
        SDL_copyp(&SDL_sem_impl_active, impl);
    }
    return SDL_sem_impl_active.Create(initial_value);
}

void SDL_DestroySemaphore(SDL_Semaphore *sem)
{
    SDL_sem_impl_active.Destroy(sem);
}

bool SDL_WaitSemaphoreTimeoutNS(SDL_Semaphore *sem, Sint64 timeoutNS)
{
    return SDL_sem_impl_active.WaitTimeoutNS(sem, timeoutNS);
}

Uint32 SDL_GetSemaphoreValue(SDL_Semaphore *sem)
{
    return SDL_sem_impl_active.Value(sem);
}

void SDL_SignalSemaphore(SDL_Semaphore *sem)
{
    SDL_sem_impl_active.Signal(sem);
}

#endif // SDL_THREAD_WINDOWS
