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

// PSP thread management routines for SDL

#include <stdio.h>
#include <stdlib.h>

#include "../SDL_systhread.h"
#include "../SDL_thread_c.h"
#include <pspkerneltypes.h>
#include <pspthreadman.h>

static int ThreadEntry(SceSize args, void *argp)
{
    SDL_RunThread(*(SDL_Thread **)argp);
    return 0;
}

bool SDL_SYS_CreateThread(SDL_Thread *thread,
                          SDL_FunctionPointer pfnBeginThread,
                          SDL_FunctionPointer pfnEndThread)
{
    SceKernelThreadInfo status;
    int priority = 32;

    // Set priority of new thread to the same as the current thread
    status.size = sizeof(SceKernelThreadInfo);
    if (sceKernelReferThreadStatus(sceKernelGetThreadId(), &status) == 0) {
        priority = status.currentPriority;
    }

    thread->handle = sceKernelCreateThread(thread->name, ThreadEntry,
                                           priority, thread->stacksize ? ((int)thread->stacksize) : 0x8000,
                                           PSP_THREAD_ATTR_VFPU, NULL);
    if (thread->handle < 0) {
        return SDL_SetError("sceKernelCreateThread() failed");
    }

    sceKernelStartThread(thread->handle, 4, &thread);
    return true;
}

void SDL_SYS_SetupThread(const char *name)
{
    // Do nothing.
}

SDL_ThreadID SDL_GetCurrentThreadID(void)
{
    return (SDL_ThreadID)sceKernelGetThreadId();
}

void SDL_SYS_WaitThread(SDL_Thread *thread)
{
    sceKernelWaitThreadEnd(thread->handle, NULL);
    sceKernelDeleteThread(thread->handle);
}

void SDL_SYS_DetachThread(SDL_Thread *thread)
{
    // !!! FIXME: is this correct?
    sceKernelDeleteThread(thread->handle);
}

void SDL_SYS_KillThread(SDL_Thread *thread)
{
    sceKernelTerminateDeleteThread(thread->handle);
}

bool SDL_SYS_SetThreadPriority(SDL_ThreadPriority priority)
{
    int value;

    if (priority == SDL_THREAD_PRIORITY_LOW) {
        value = 111;
    } else if (priority == SDL_THREAD_PRIORITY_HIGH) {
        value = 32;
    } else if (priority == SDL_THREAD_PRIORITY_TIME_CRITICAL) {
        value = 16;
    } else {
        value = 50;
    }

    if (sceKernelChangeThreadPriority(sceKernelGetThreadId(), value) < 0) {
        return SDL_SetError("sceKernelChangeThreadPriority() failed");
    }
    return true;
}

#endif // SDL_THREAD_PSP
