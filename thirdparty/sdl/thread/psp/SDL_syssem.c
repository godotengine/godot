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

#ifdef SDL_THREAD_PSP

// Semaphore functions for the PSP.

#include <stdio.h>
#include <stdlib.h>

#include <pspthreadman.h>
#include <pspkerror.h>

struct SDL_Semaphore
{
    SceUID semid;
};

// Create a semaphore
SDL_Semaphore *SDL_CreateSemaphore(Uint32 initial_value)
{
    SDL_Semaphore *sem;

    sem = (SDL_Semaphore *)SDL_malloc(sizeof(*sem));
    if (sem) {
        // TODO: Figure out the limit on the maximum value.
        sem->semid = sceKernelCreateSema("SDL sema", 0, initial_value, 255, NULL);
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
            sceKernelDeleteSema(sem->semid);
            sem->semid = 0;
        }

        SDL_free(sem);
    }
}

/* TODO: This routine is a bit overloaded.
 * If the timeout is 0 then just poll the semaphore; if it's -1, pass
 * NULL to sceKernelWaitSema() so that it waits indefinitely; and if the timeout
 * is specified, convert it to microseconds. */
bool SDL_WaitSemaphoreTimeoutNS(SDL_Semaphore *sem, Sint64 timeoutNS)
{
	SceUInt timeoutUS;
    SceUInt *pTimeout = NULL;

    if (!sem) {
        return true;
    }

    if (timeoutNS == 0) {
        return (sceKernelPollSema(sem->semid, 1) == 0);
    }

    if (timeoutNS > 0) {
        timeoutUS = (SceUInt)SDL_NS_TO_US(timeoutNS); // Convert to microseconds.
        pTimeout = &timeoutUS;
    }

    return (sceKernelWaitSema(sem->semid, 1, pTimeout) == 0);
}

// Returns the current count of the semaphore
Uint32 SDL_GetSemaphoreValue(SDL_Semaphore *sem)
{
    SceKernelSemaInfo info;

    if (!sem) {
        return 0;
    }

    if (sceKernelReferSemaStatus(sem->semid, &info) == 0) {
        return info.currentCount;
    }
    return 0;
}

void SDL_SignalSemaphore(SDL_Semaphore *sem)
{
    if (!sem) {
        return;
    }

    sceKernelSignalSema(sem->semid, 1);
}

#endif // SDL_THREAD_PSP
