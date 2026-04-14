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

// These are functions that need to be implemented by a port of SDL

#ifndef SDL_systhread_h_
#define SDL_systhread_h_

#include "SDL_thread_c.h"

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

/* This function creates a thread, passing args to SDL_RunThread(),
   saves a system-dependent thread id in thread->id, and returns 0
   on success.
*/
extern bool SDL_SYS_CreateThread(SDL_Thread *thread,
                                 SDL_FunctionPointer pfnBeginThread,
                                 SDL_FunctionPointer pfnEndThread);

// This function does any necessary setup in the child thread
extern void SDL_SYS_SetupThread(const char *name);

// This function sets the current thread priority
extern bool SDL_SYS_SetThreadPriority(SDL_ThreadPriority priority);

/* This function waits for the thread to finish and frees any data
   allocated by SDL_SYS_CreateThread()
 */
extern void SDL_SYS_WaitThread(SDL_Thread *thread);

// Mark thread as cleaned up as soon as it exits, without joining.
extern void SDL_SYS_DetachThread(SDL_Thread *thread);

// Initialize the global TLS data
extern void SDL_SYS_InitTLSData(void);

// Get the thread local storage for this thread
extern SDL_TLSData *SDL_SYS_GetTLSData(void);

// Set the thread local storage for this thread
extern bool SDL_SYS_SetTLSData(SDL_TLSData *data);

// Quit the global TLS data
extern void SDL_SYS_QuitTLSData(void);

// A helper function for setting up a thread with a stack size.
extern SDL_Thread *SDL_CreateThreadWithStackSize(SDL_ThreadFunction fn, const char *name, size_t stacksize, void *data);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif // SDL_systhread_h_
