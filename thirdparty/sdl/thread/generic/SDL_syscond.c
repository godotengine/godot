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

// An implementation of condition variables using semaphores and mutexes
/*
   This implementation borrows heavily from the BeOS condition variable
   implementation, written by Christopher Tate and Owen Smith.  Thanks!
 */

#include "../generic/SDL_syscond_c.h"

/* If two implementations are to be compiled into SDL (the active one
 * will be chosen at runtime), the function names need to be
 * suffixed
 */
#ifndef SDL_THREAD_GENERIC_COND_SUFFIX
#define SDL_CreateCondition_generic      SDL_CreateCondition
#define SDL_DestroyCondition_generic     SDL_DestroyCondition
#define SDL_SignalCondition_generic      SDL_SignalCondition
#define SDL_BroadcastCondition_generic   SDL_BroadcastCondition
#endif

typedef struct SDL_cond_generic
{
    SDL_Semaphore *sem;
    SDL_Semaphore *handshake_sem;
    SDL_Semaphore *signal_sem;
    int num_waiting;
    int num_signals;
} SDL_cond_generic;

// Create a condition variable
SDL_Condition *SDL_CreateCondition_generic(void)
{
    SDL_cond_generic *cond = (SDL_cond_generic *)SDL_calloc(1, sizeof(*cond));

#ifndef SDL_THREADS_DISABLED
    if (cond) {
        cond->sem = SDL_CreateSemaphore(0);
        cond->handshake_sem = SDL_CreateSemaphore(0);
        cond->signal_sem = SDL_CreateSemaphore(1);
        if (!cond->sem || !cond->handshake_sem || !cond->signal_sem) {
            SDL_DestroyCondition_generic((SDL_Condition *)cond);
            cond = NULL;
        }
    }
#endif

    return (SDL_Condition *)cond;
}

// Destroy a condition variable
void SDL_DestroyCondition_generic(SDL_Condition *_cond)
{
    SDL_cond_generic *cond = (SDL_cond_generic *)_cond;
    if (cond) {
        if (cond->sem) {
            SDL_DestroySemaphore(cond->sem);
        }
        if (cond->handshake_sem) {
            SDL_DestroySemaphore(cond->handshake_sem);
        }
        if (cond->signal_sem) {
            SDL_DestroySemaphore(cond->signal_sem);
        }
        SDL_free(cond);
    }
}

// Restart one of the threads that are waiting on the condition variable
void SDL_SignalCondition_generic(SDL_Condition *_cond)
{
    SDL_cond_generic *cond = (SDL_cond_generic *)_cond;
    if (!cond) {
        return;
    }

#ifndef SDL_THREADS_DISABLED
    /* If there are waiting threads not already signalled, then
       signal the condition and wait for the thread to respond.
     */
    SDL_WaitSemaphore(cond->signal_sem);
    if (cond->num_waiting > cond->num_signals) {
        cond->num_signals++;
        SDL_SignalSemaphore(cond->sem);
        SDL_SignalSemaphore(cond->signal_sem);
        SDL_WaitSemaphore(cond->handshake_sem);
    } else {
        SDL_SignalSemaphore(cond->signal_sem);
    }
#endif
}

// Restart all threads that are waiting on the condition variable
void SDL_BroadcastCondition_generic(SDL_Condition *_cond)
{
    SDL_cond_generic *cond = (SDL_cond_generic *)_cond;
    if (!cond) {
        return;
    }

#ifndef SDL_THREADS_DISABLED
    /* If there are waiting threads not already signalled, then
       signal the condition and wait for the thread to respond.
     */
    SDL_WaitSemaphore(cond->signal_sem);
    if (cond->num_waiting > cond->num_signals) {
        const int num_waiting = (cond->num_waiting - cond->num_signals);
        cond->num_signals = cond->num_waiting;
        for (int i = 0; i < num_waiting; i++) {
            SDL_SignalSemaphore(cond->sem);
        }
        /* Now all released threads are blocked here, waiting for us.
           Collect them all (and win fabulous prizes!) :-)
         */
        SDL_SignalSemaphore(cond->signal_sem);
        for (int i = 0; i < num_waiting; i++) {
            SDL_WaitSemaphore(cond->handshake_sem);
        }
    } else {
        SDL_SignalSemaphore(cond->signal_sem);
    }
#endif
}

/* Wait on the condition variable for at most 'timeoutNS' nanoseconds.
   The mutex must be locked before entering this function!
   The mutex is unlocked during the wait, and locked again after the wait.

Typical use:

Thread A:
    SDL_LockMutex(lock);
    while ( ! condition ) {
        SDL_WaitCondition(cond, lock);
    }
    SDL_UnlockMutex(lock);

Thread B:
    SDL_LockMutex(lock);
    ...
    condition = true;
    ...
    SDL_SignalCondition(cond);
    SDL_UnlockMutex(lock);
 */
bool SDL_WaitConditionTimeoutNS_generic(SDL_Condition *_cond, SDL_Mutex *mutex, Sint64 timeoutNS)
{
    SDL_cond_generic *cond = (SDL_cond_generic *)_cond;
    bool result = true;

    if (!cond || !mutex) {
        return true;
    }

#ifndef SDL_THREADS_DISABLED
    /* Obtain the protection mutex, and increment the number of waiters.
       This allows the signal mechanism to only perform a signal if there
       are waiting threads.
     */
    SDL_WaitSemaphore(cond->signal_sem);
    cond->num_waiting++;
    SDL_SignalSemaphore(cond->signal_sem);

    // Unlock the mutex, as is required by condition variable semantics
    SDL_UnlockMutex(mutex);

    // Wait for a signal
    result = SDL_WaitSemaphoreTimeoutNS(cond->sem, timeoutNS);

    /* Let the signaler know we have completed the wait, otherwise
       the signaler can race ahead and get the condition semaphore
       if we are stopped between the mutex unlock and semaphore wait,
       giving a deadlock.  See the following URL for details:
       http://web.archive.org/web/20010914175514/http://www-classic.be.com/aboutbe/benewsletter/volume_III/Issue40.html#Workshop
     */
    SDL_WaitSemaphore(cond->signal_sem);
    if (cond->num_signals > 0) {
        SDL_SignalSemaphore(cond->handshake_sem);
        cond->num_signals--;
    }
    cond->num_waiting--;
    SDL_SignalSemaphore(cond->signal_sem);

    // Lock the mutex, as is required by condition variable semantics
    SDL_LockMutex(mutex);
#endif

    return result;
}

#ifndef SDL_THREAD_GENERIC_COND_SUFFIX
bool SDL_WaitConditionTimeoutNS(SDL_Condition *cond, SDL_Mutex *mutex, Sint64 timeoutNS)
{
    return SDL_WaitConditionTimeoutNS_generic(cond, mutex, timeoutNS);
}
#endif
