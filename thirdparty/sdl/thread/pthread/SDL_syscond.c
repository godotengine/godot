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

#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

#include "SDL_sysmutex_c.h"

struct SDL_Condition
{
    pthread_cond_t cond;
};

// Create a condition variable
SDL_Condition *SDL_CreateCondition(void)
{
    SDL_Condition *cond;

    cond = (SDL_Condition *)SDL_malloc(sizeof(SDL_Condition));
    if (cond) {
        if (pthread_cond_init(&cond->cond, NULL) != 0) {
            SDL_SetError("pthread_cond_init() failed");
            SDL_free(cond);
            cond = NULL;
        }
    }
    return cond;
}

// Destroy a condition variable
void SDL_DestroyCondition(SDL_Condition *cond)
{
    if (cond) {
        pthread_cond_destroy(&cond->cond);
        SDL_free(cond);
    }
}

// Restart one of the threads that are waiting on the condition variable
void SDL_SignalCondition(SDL_Condition *cond)
{
    if (!cond) {
        return;
    }

    pthread_cond_signal(&cond->cond);
}

// Restart all threads that are waiting on the condition variable
void SDL_BroadcastCondition(SDL_Condition *cond)
{
    if (!cond) {
        return;
    }

    pthread_cond_broadcast(&cond->cond);
}

bool SDL_WaitConditionTimeoutNS(SDL_Condition *cond, SDL_Mutex *mutex, Sint64 timeoutNS)
{
#ifndef HAVE_CLOCK_GETTIME
    struct timeval delta;
#endif
    struct timespec abstime;

    if (!cond || !mutex) {
        return true;
    }

    if (timeoutNS < 0) {
        return (pthread_cond_wait(&cond->cond, &mutex->id) == 0);
    }

#ifdef HAVE_CLOCK_GETTIME
    clock_gettime(CLOCK_REALTIME, &abstime);

    abstime.tv_sec += (timeoutNS / SDL_NS_PER_SECOND);
    abstime.tv_nsec += (timeoutNS % SDL_NS_PER_SECOND);
#else
    gettimeofday(&delta, NULL);

    abstime.tv_sec = delta.tv_sec + (timeoutNS / SDL_NS_PER_SECOND);
    abstime.tv_nsec = SDL_US_TO_NS(delta.tv_usec) + (timeoutNS % SDL_NS_PER_SECOND);
#endif
    while (abstime.tv_nsec >= 1000000000) {
        abstime.tv_sec += 1;
        abstime.tv_nsec -= 1000000000;
    }

    bool result;
    int rc;
tryagain:
    rc = pthread_cond_timedwait(&cond->cond, &mutex->id, &abstime);
    switch (rc) {
    case EINTR:
        goto tryagain;
        // break; -Wunreachable-code-break
    case ETIMEDOUT:
        result = false;
        break;
    default:
        result = true;
        break;
    }
    return result;
}
