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

// An implementation of semaphores using mutexes and condition variables

#include "SDL_systhread_c.h"

#ifdef SDL_THREADS_DISABLED

SDL_Semaphore *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_SetError("SDL not built with thread support");
    return NULL;
}

void SDL_DestroySemaphore(SDL_Semaphore *sem)
{
}

bool SDL_WaitSemaphoreTimeoutNS(SDL_Semaphore *sem, Sint64 timeoutNS)
{
    return true;
}

Uint32 SDL_GetSemaphoreValue(SDL_Semaphore *sem)
{
    return 0;
}

void SDL_SignalSemaphore(SDL_Semaphore *sem)
{
    return;
}

#else

struct SDL_Semaphore
{
    Uint32 count;
    Uint32 waiters_count;
    SDL_Mutex *count_lock;
    SDL_Condition *count_nonzero;
};

SDL_Semaphore *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_Semaphore *sem;

    sem = (SDL_Semaphore *)SDL_malloc(sizeof(*sem));
    if (!sem) {
        return NULL;
    }
    sem->count = initial_value;
    sem->waiters_count = 0;

    sem->count_lock = SDL_CreateMutex();
    sem->count_nonzero = SDL_CreateCondition();
    if (!sem->count_lock || !sem->count_nonzero) {
        SDL_DestroySemaphore(sem);
        return NULL;
    }

    return sem;
}

/* WARNING:
   You cannot call this function when another thread is using the semaphore.
*/
void SDL_DestroySemaphore(SDL_Semaphore *sem)
{
    if (sem) {
        sem->count = 0xFFFFFFFF;
        while (sem->waiters_count > 0) {
            SDL_SignalCondition(sem->count_nonzero);
            SDL_Delay(10);
        }
        SDL_DestroyCondition(sem->count_nonzero);
        if (sem->count_lock) {
            SDL_LockMutex(sem->count_lock);
            SDL_UnlockMutex(sem->count_lock);
            SDL_DestroyMutex(sem->count_lock);
        }
        SDL_free(sem);
    }
}

bool SDL_WaitSemaphoreTimeoutNS(SDL_Semaphore *sem, Sint64 timeoutNS)
{
    bool result = false;

    if (!sem) {
        return true;
    }

    if (timeoutNS == 0) {
        SDL_LockMutex(sem->count_lock);
        if (sem->count > 0) {
            --sem->count;
            result = true;
        }
        SDL_UnlockMutex(sem->count_lock);
    } else if (timeoutNS < 0) {
        SDL_LockMutex(sem->count_lock);
        ++sem->waiters_count;
        while (sem->count == 0) {
            SDL_WaitConditionTimeoutNS(sem->count_nonzero, sem->count_lock, -1);
        }
        --sem->waiters_count;
        --sem->count;
        result = true;
        SDL_UnlockMutex(sem->count_lock);
    } else {
        Uint64 stop_time = SDL_GetTicksNS() + timeoutNS;

        SDL_LockMutex(sem->count_lock);
        ++sem->waiters_count;
        while (sem->count == 0) {
            Sint64 waitNS = (Sint64)(stop_time - SDL_GetTicksNS());
            if (waitNS > 0) {
                SDL_WaitConditionTimeoutNS(sem->count_nonzero, sem->count_lock, waitNS);
            } else {
                break;
            }
        }
        --sem->waiters_count;

        if (sem->count > 0) {
            --sem->count;
            result = true;
        }
        SDL_UnlockMutex(sem->count_lock);
    }

    return result;
}

Uint32 SDL_GetSemaphoreValue(SDL_Semaphore *sem)
{
    Uint32 value;

    value = 0;
    if (sem) {
        SDL_LockMutex(sem->count_lock);
        value = sem->count;
        SDL_UnlockMutex(sem->count_lock);
    }
    return value;
}

void SDL_SignalSemaphore(SDL_Semaphore *sem)
{
    if (!sem) {
        return;
    }

    SDL_LockMutex(sem->count_lock);
    if (sem->waiters_count > 0) {
        SDL_SignalCondition(sem->count_nonzero);
    }
    ++sem->count;
    SDL_UnlockMutex(sem->count_lock);
}

#endif // SDL_THREADS_DISABLED
