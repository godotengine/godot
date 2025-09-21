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

// An implementation of mutexes using semaphores

#include "SDL_systhread_c.h"

struct SDL_Mutex
{
    int recursive;
    SDL_ThreadID owner;
    SDL_Semaphore *sem;
};

SDL_Mutex *SDL_CreateMutex(void)
{
    SDL_Mutex *mutex = (SDL_Mutex *)SDL_calloc(1, sizeof(*mutex));

#ifndef SDL_THREADS_DISABLED
    if (mutex) {
        // Create the mutex semaphore, with initial value 1
        mutex->sem = SDL_CreateSemaphore(1);
        mutex->recursive = 0;
        mutex->owner = 0;
        if (!mutex->sem) {
            SDL_free(mutex);
            mutex = NULL;
        }
    }
#endif // !SDL_THREADS_DISABLED

    return mutex;
}

void SDL_DestroyMutex(SDL_Mutex *mutex)
{
    if (mutex) {
        if (mutex->sem) {
            SDL_DestroySemaphore(mutex->sem);
        }
        SDL_free(mutex);
    }
}

void SDL_LockMutex(SDL_Mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS  // clang doesn't know about NULL mutexes
{
#ifndef SDL_THREADS_DISABLED
    if (mutex != NULL) {
        SDL_ThreadID this_thread = SDL_GetCurrentThreadID();
        if (mutex->owner == this_thread) {
            ++mutex->recursive;
        } else {
            /* The order of operations is important.
               We set the locking thread id after we obtain the lock
               so unlocks from other threads will fail.
             */
            SDL_WaitSemaphore(mutex->sem);
            mutex->owner = this_thread;
            mutex->recursive = 0;
        }
    }
#endif // SDL_THREADS_DISABLED
}

bool SDL_TryLockMutex(SDL_Mutex *mutex)
{
    bool result = true;
#ifndef SDL_THREADS_DISABLED
    if (mutex) {
        SDL_ThreadID this_thread = SDL_GetCurrentThreadID();
        if (mutex->owner == this_thread) {
            ++mutex->recursive;
        } else {
            /* The order of operations is important.
               We set the locking thread id after we obtain the lock
               so unlocks from other threads will fail.
             */
            result = SDL_TryWaitSemaphore(mutex->sem);
            if (result) {
                mutex->owner = this_thread;
                mutex->recursive = 0;
            }
        }
    }
#endif // SDL_THREADS_DISABLED
    return result;
}

void SDL_UnlockMutex(SDL_Mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS  // clang doesn't know about NULL mutexes
{
#ifndef SDL_THREADS_DISABLED
    if (mutex != NULL) {
        // If we don't own the mutex, we can't unlock it
        if (SDL_GetCurrentThreadID() != mutex->owner) {
            SDL_assert(!"Tried to unlock a mutex we don't own!");
            return; // (undefined behavior!) SDL_SetError("mutex not owned by this thread");
        }

        if (mutex->recursive) {
            --mutex->recursive;
        } else {
            /* The order of operations is important.
               First reset the owner so another thread doesn't lock
               the mutex and set the ownership before we reset it,
               then release the lock semaphore.
             */
            mutex->owner = 0;
            SDL_SignalSemaphore(mutex->sem);
        }
    }
#endif // SDL_THREADS_DISABLED
}

