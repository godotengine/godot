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

// PS2 thread management routines for SDL

#include <stdio.h>
#include <stdlib.h>

#include "../SDL_systhread.h"
#include "../SDL_thread_c.h"
#include <kernel.h>

static void FinishThread(SDL_Thread *thread)
{
    ee_thread_status_t info;
    int res;

    res = ReferThreadStatus(thread->handle, &info);
    TerminateThread(thread->handle);
    DeleteThread(thread->handle);
    DeleteSema((int)thread->endfunc);

    if (res > 0) {
        SDL_free(info.stack);
    }
}

static int childThread(void *arg)
{
    SDL_Thread *thread = (SDL_Thread *)arg;
    int res = thread->userfunc(thread->userdata);
    SignalSema((int)thread->endfunc);
    return res;
}

bool SDL_SYS_CreateThread(SDL_Thread *thread,
                          SDL_FunctionPointer pfnBeginThread,
                          SDL_FunctionPointer pfnEndThread)
{
    ee_thread_status_t status;
    ee_thread_t eethread;
    ee_sema_t sema;
    size_t stack_size;
    int priority = 32;

    // Set priority of new thread to the same as the current thread
    // status.size = sizeof(ee_thread_t);
    if (ReferThreadStatus(GetThreadId(), &status) == 0) {
        priority = status.current_priority;
    }

    stack_size = thread->stacksize ? ((int)thread->stacksize) : 0x1800;

    // Create EE Thread
    eethread.attr = 0;
    eethread.option = 0;
    eethread.func = &childThread;
    eethread.stack = SDL_malloc(stack_size);
    eethread.stack_size = stack_size;
    eethread.gp_reg = &_gp;
    eethread.initial_priority = priority;
    thread->handle = CreateThread(&eethread);

    if (thread->handle < 0) {
        return SDL_SetError("CreateThread() failed");
    }

    // Prepare el semaphore for the ending function
    sema.init_count = 0;
    sema.max_count = 1;
    sema.option = 0;
    thread->endfunc = (void *)CreateSema(&sema);

    if (StartThread(thread->handle, thread) < 0) {
        return SDL_SetError("StartThread() failed");
    }
    return true;
}

void SDL_SYS_SetupThread(const char *name)
{
    // Do nothing.
}

SDL_ThreadID SDL_GetCurrentThreadID(void)
{
    return (SDL_ThreadID)GetThreadId();
}

void SDL_SYS_WaitThread(SDL_Thread *thread)
{
    WaitSema((int)thread->endfunc);
    ReleaseWaitThread(thread->handle);
    FinishThread(thread);
}

void SDL_SYS_DetachThread(SDL_Thread *thread)
{
    // Do nothing.
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

    if (ChangeThreadPriority(GetThreadId(), value) < 0) {
        return SDL_SetError("ChangeThreadPriority() failed");
    }
    return true;
}

#endif // SDL_THREAD_PS2
