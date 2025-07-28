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

#include "../generic/SDL_syscond_c.h"
#include "SDL_sysmutex_c.h"

typedef SDL_Condition *(*pfnSDL_CreateCondition)(void);
typedef void (*pfnSDL_DestroyCondition)(SDL_Condition *);
typedef void (*pfnSDL_SignalCondition)(SDL_Condition *);
typedef void (*pfnSDL_BroadcastCondition)(SDL_Condition *);
typedef bool (*pfnSDL_WaitConditionTimeoutNS)(SDL_Condition *, SDL_Mutex *, Sint64);

typedef struct SDL_cond_impl_t
{
    pfnSDL_CreateCondition Create;
    pfnSDL_DestroyCondition Destroy;
    pfnSDL_SignalCondition Signal;
    pfnSDL_BroadcastCondition Broadcast;
    pfnSDL_WaitConditionTimeoutNS WaitTimeoutNS;
} SDL_cond_impl_t;

// Implementation will be chosen at runtime based on available Kernel features
static SDL_cond_impl_t SDL_cond_impl_active = { 0 };

/**
 * Native Windows Condition Variable (SRW Locks)
 */

#ifndef CONDITION_VARIABLE_INIT
#define CONDITION_VARIABLE_INIT \
    {                           \
        0                       \
    }
typedef struct CONDITION_VARIABLE
{
    PVOID Ptr;
} CONDITION_VARIABLE, *PCONDITION_VARIABLE;
#endif

typedef VOID(WINAPI *pfnWakeConditionVariable)(PCONDITION_VARIABLE);
typedef VOID(WINAPI *pfnWakeAllConditionVariable)(PCONDITION_VARIABLE);
typedef BOOL(WINAPI *pfnSleepConditionVariableSRW)(PCONDITION_VARIABLE, PSRWLOCK, DWORD, ULONG);
typedef BOOL(WINAPI *pfnSleepConditionVariableCS)(PCONDITION_VARIABLE, PCRITICAL_SECTION, DWORD);

static pfnWakeConditionVariable pWakeConditionVariable = NULL;
static pfnWakeAllConditionVariable pWakeAllConditionVariable = NULL;
static pfnSleepConditionVariableSRW pSleepConditionVariableSRW = NULL;
static pfnSleepConditionVariableCS pSleepConditionVariableCS = NULL;

typedef struct SDL_cond_cv
{
    CONDITION_VARIABLE cond;
} SDL_cond_cv;

static SDL_Condition *SDL_CreateCondition_cv(void)
{
    // Relies on CONDITION_VARIABLE_INIT == 0.
    return (SDL_Condition *)SDL_calloc(1, sizeof(SDL_cond_cv));
}

static void SDL_DestroyCondition_cv(SDL_Condition *cond)
{
    // There are no kernel allocated resources
    SDL_free(cond);
}

static void SDL_SignalCondition_cv(SDL_Condition *_cond)
{
    SDL_cond_cv *cond = (SDL_cond_cv *)_cond;
    pWakeConditionVariable(&cond->cond);
}

static void SDL_BroadcastCondition_cv(SDL_Condition *_cond)
{
    SDL_cond_cv *cond = (SDL_cond_cv *)_cond;
    pWakeAllConditionVariable(&cond->cond);
}

static bool SDL_WaitConditionTimeoutNS_cv(SDL_Condition *_cond, SDL_Mutex *_mutex, Sint64 timeoutNS)
{
    SDL_cond_cv *cond = (SDL_cond_cv *)_cond;
    DWORD timeout;
    bool result;

    if (timeoutNS < 0) {
        timeout = INFINITE;
    } else {
        timeout = (DWORD)SDL_NS_TO_MS(timeoutNS);
    }

    if (SDL_mutex_impl_active.Type == SDL_MUTEX_SRW) {
        SDL_mutex_srw *mutex = (SDL_mutex_srw *)_mutex;

        if (mutex->count != 1 || mutex->owner != GetCurrentThreadId()) {
            // Passed mutex is not locked or locked recursively"
            return false;
        }

        // The mutex must be updated to the released state
        mutex->count = 0;
        mutex->owner = 0;

        result = (pSleepConditionVariableSRW(&cond->cond, &mutex->srw, timeout, 0) == TRUE);

        // The mutex is owned by us again, regardless of status of the wait
        SDL_assert(mutex->count == 0 && mutex->owner == 0);
        mutex->count = 1;
        mutex->owner = GetCurrentThreadId();
    } else {
        SDL_mutex_cs *mutex = (SDL_mutex_cs *)_mutex;

        SDL_assert(SDL_mutex_impl_active.Type == SDL_MUTEX_CS);

        result = (pSleepConditionVariableCS(&cond->cond, &mutex->cs, timeout) == TRUE);
    }

    return result;
}

static const SDL_cond_impl_t SDL_cond_impl_cv = {
    &SDL_CreateCondition_cv,
    &SDL_DestroyCondition_cv,
    &SDL_SignalCondition_cv,
    &SDL_BroadcastCondition_cv,
    &SDL_WaitConditionTimeoutNS_cv,
};


// Generic Condition Variable implementation using SDL_Mutex and SDL_Semaphore
static const SDL_cond_impl_t SDL_cond_impl_generic = {
    &SDL_CreateCondition_generic,
    &SDL_DestroyCondition_generic,
    &SDL_SignalCondition_generic,
    &SDL_BroadcastCondition_generic,
    &SDL_WaitConditionTimeoutNS_generic,
};

SDL_Condition *SDL_CreateCondition(void)
{
    if (!SDL_cond_impl_active.Create) {
        const SDL_cond_impl_t *impl = NULL;

        if (SDL_mutex_impl_active.Type == SDL_MUTEX_INVALID) {
            // The mutex implementation isn't decided yet, trigger it
            SDL_Mutex *mutex = SDL_CreateMutex();
            if (!mutex) {
                return NULL;
            }
            SDL_DestroyMutex(mutex);

            SDL_assert(SDL_mutex_impl_active.Type != SDL_MUTEX_INVALID);
        }

        // Default to generic implementation, works with all mutex implementations
        impl = &SDL_cond_impl_generic;
        {
            HMODULE kernel32 = GetModuleHandle(TEXT("kernel32.dll"));
            if (kernel32) {
                pWakeConditionVariable = (pfnWakeConditionVariable)GetProcAddress(kernel32, "WakeConditionVariable");
                pWakeAllConditionVariable = (pfnWakeAllConditionVariable)GetProcAddress(kernel32, "WakeAllConditionVariable");
                pSleepConditionVariableSRW = (pfnSleepConditionVariableSRW)GetProcAddress(kernel32, "SleepConditionVariableSRW");
                pSleepConditionVariableCS = (pfnSleepConditionVariableCS)GetProcAddress(kernel32, "SleepConditionVariableCS");
                if (pWakeConditionVariable && pWakeAllConditionVariable && pSleepConditionVariableSRW && pSleepConditionVariableCS) {
                    // Use the Windows provided API
                    impl = &SDL_cond_impl_cv;
                }
            }
        }

        SDL_copyp(&SDL_cond_impl_active, impl);
    }
    return SDL_cond_impl_active.Create();
}

void SDL_DestroyCondition(SDL_Condition *cond)
{
    if (cond) {
        SDL_cond_impl_active.Destroy(cond);
    }
}

void SDL_SignalCondition(SDL_Condition *cond)
{
    if (!cond) {
        return;
    }

    SDL_cond_impl_active.Signal(cond);
}

void SDL_BroadcastCondition(SDL_Condition *cond)
{
    if (!cond) {
        return;
    }

    SDL_cond_impl_active.Broadcast(cond);
}

bool SDL_WaitConditionTimeoutNS(SDL_Condition *cond, SDL_Mutex *mutex, Sint64 timeoutNS)
{
    if (!cond || !mutex) {
        return true;
    }

    return SDL_cond_impl_active.WaitTimeoutNS(cond, mutex, timeoutNS);
}
