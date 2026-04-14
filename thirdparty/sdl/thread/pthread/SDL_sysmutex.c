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

#include <errno.h>
#include <pthread.h>

#include "SDL_sysmutex_c.h"

SDL_Mutex *SDL_CreateMutex(void)
{
    SDL_Mutex *mutex;
    pthread_mutexattr_t attr;

    // Allocate the structure
    mutex = (SDL_Mutex *)SDL_calloc(1, sizeof(*mutex));
    if (mutex) {
        pthread_mutexattr_init(&attr);
#ifdef SDL_THREAD_PTHREAD_RECURSIVE_MUTEX
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
#elif defined(SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP)
        pthread_mutexattr_setkind_np(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
#else
        // No extra attributes necessary
#endif
        if (pthread_mutex_init(&mutex->id, &attr) != 0) {
            SDL_SetError("pthread_mutex_init() failed");
            SDL_free(mutex);
            mutex = NULL;
        }
    }
    return mutex;
}

void SDL_DestroyMutex(SDL_Mutex *mutex)
{
    if (mutex) {
        pthread_mutex_destroy(&mutex->id);
        SDL_free(mutex);
    }
}

void SDL_LockMutex(SDL_Mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS // clang doesn't know about NULL mutexes
{
    if (mutex) {
#ifdef FAKE_RECURSIVE_MUTEX
        pthread_t this_thread = pthread_self();
        if (mutex->owner == this_thread) {
            ++mutex->recursive;
        } else {
            /* The order of operations is important.
               We set the locking thread id after we obtain the lock
               so unlocks from other threads will fail.
             */
            const int rc = pthread_mutex_lock(&mutex->id);
            SDL_assert(rc == 0);  // assume we're in a lot of trouble if this assert fails.
            mutex->owner = this_thread;
            mutex->recursive = 0;
        }
#else
        const int rc = pthread_mutex_lock(&mutex->id);
        SDL_assert(rc == 0);  // assume we're in a lot of trouble if this assert fails.
#endif
    }
}

bool SDL_TryLockMutex(SDL_Mutex *mutex)
{
    bool result = true;

    if (mutex) {
#ifdef FAKE_RECURSIVE_MUTEX
        pthread_t this_thread = pthread_self();
        if (mutex->owner == this_thread) {
            ++mutex->recursive;
        } else {
            /* The order of operations is important.
               We set the locking thread id after we obtain the lock
               so unlocks from other threads will fail.
             */
            const int rc = pthread_mutex_trylock(&mutex->id);
            if (rc == 0) {
                mutex->owner = this_thread;
                mutex->recursive = 0;
            } else if (rc == EBUSY) {
                result = false;
            } else {
                SDL_assert(!"Error trying to lock mutex");  // assume we're in a lot of trouble if this assert fails.
                result = false;
            }
        }
#else
        const int rc = pthread_mutex_trylock(&mutex->id);
        if (rc != 0) {
            if (rc == EBUSY) {
                result = false;
            } else {
                SDL_assert(!"Error trying to lock mutex");  // assume we're in a lot of trouble if this assert fails.
                result = false;
            }
        }
#endif
    }

    return result;
}

void SDL_UnlockMutex(SDL_Mutex *mutex) SDL_NO_THREAD_SAFETY_ANALYSIS // clang doesn't know about NULL mutexes
{
    if (mutex) {
#ifdef FAKE_RECURSIVE_MUTEX
        // We can only unlock the mutex if we own it
        if (pthread_self() == mutex->owner) {
            if (mutex->recursive) {
                --mutex->recursive;
            } else {
                /* The order of operations is important.
                   First reset the owner so another thread doesn't lock
                   the mutex and set the ownership before we reset it,
                   then release the lock semaphore.
                 */
                mutex->owner = 0;
                pthread_mutex_unlock(&mutex->id);
            }
        } else {
            SDL_SetError("mutex not owned by this thread");
            return;
        }

#else
        const int rc = pthread_mutex_unlock(&mutex->id);
        SDL_assert(rc == 0);  // assume we're in a lot of trouble if this assert fails.
#endif // FAKE_RECURSIVE_MUTEX
    }
}

