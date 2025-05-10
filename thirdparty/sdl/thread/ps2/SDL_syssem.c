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

#ifdef SDL_THREAD_PS2

// Semaphore functions for the PS2.

#include <stdio.h>
#include <stdlib.h>
#include <kernel_util.h>

#include <kernel.h>

struct SDL_Semaphore
{
    s32 semid;
};

// Create a semaphore
SDL_Semaphore *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_Semaphore *sem;
    ee_sema_t sema;

    sem = (SDL_Semaphore *)SDL_malloc(sizeof(*sem));
    if (sem) {
        // TODO: Figure out the limit on the maximum value.
        sema.init_count = initial_value;
        sema.max_count = 255;
        sema.option = 0;
        sem->semid = CreateSema(&sema);

        if (sem->semid < 0) {
            SDL_SetError("Couldn't create semaphore");
            SDL_free(sem);
            sem = NULL;
        }
    }

    return sem;
}

// Free the semaphore
void SDL_DestroySemaphore(SDL_Semaphore *sem)
{
    if (sem) {
        if (sem->semid > 0) {
            DeleteSema(sem->semid);
            sem->semid = 0;
        }

        SDL_free(sem);
    }
}

bool SDL_WaitSemaphoreTimeoutNS(SDL_Semaphore *sem, Sint64 timeoutNS)
{
    u64 timeout_usec;
    u64 *timeout_ptr;

    if (!sem) {
        return true;
    }

    if (timeoutNS == 0) {
        return (PollSema(sem->semid) == 0);
    }

    timeout_ptr = NULL;

    if (timeoutNS != -1) {  // -1 == wait indefinitely.
        timeout_usec = SDL_NS_TO_US(timeoutNS);
        timeout_ptr = &timeout_usec;
    }

    return (WaitSemaEx(sem->semid, 1, timeout_ptr) == 0);
}

// Returns the current count of the semaphore
Uint32 SDL_GetSemaphoreValue(SDL_Semaphore *sem)
{
    ee_sema_t info;

    if (!sem) {
        return 0;
    }

    if (ReferSemaStatus(sem->semid, &info) == 0) {
        return info.count;
    }
    return 0;
}

void SDL_SignalSemaphore(SDL_Semaphore *sem)
{
    if (!sem) {
        return;
    }

    SignalSema(sem->semid);
}

#endif // SDL_THREAD_PS2
