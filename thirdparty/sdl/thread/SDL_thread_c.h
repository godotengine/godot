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

#ifndef SDL_thread_c_h_
#define SDL_thread_c_h_

// Need the definitions of SYS_ThreadHandle
#ifdef SDL_THREADS_DISABLED
#include "generic/SDL_systhread_c.h"
#elif defined(SDL_THREAD_PTHREAD)
#include "pthread/SDL_systhread_c.h"
#elif defined(SDL_THREAD_WINDOWS)
#include "windows/SDL_systhread_c.h"
#elif defined(SDL_THREAD_PS2)
#include "ps2/SDL_systhread_c.h"
#elif defined(SDL_THREAD_PSP)
#include "psp/SDL_systhread_c.h"
#elif defined(SDL_THREAD_VITA)
#include "vita/SDL_systhread_c.h"
#elif defined(SDL_THREAD_N3DS)
#include "n3ds/SDL_systhread_c.h"
#else
#error Need thread implementation for this platform
#include "generic/SDL_systhread_c.h"
#endif
#include "../SDL_error_c.h"

// This is the system-independent thread info structure
struct SDL_Thread
{
    SDL_ThreadID threadid;
    SYS_ThreadHandle handle;
    int status;
    SDL_AtomicInt state; /* SDL_ThreadState */
    SDL_error errbuf;
    char *name;
    size_t stacksize; // 0 for default, >0 for user-specified stack size.
    int(SDLCALL *userfunc)(void *);
    void *userdata;
    void *data;
    SDL_FunctionPointer endfunc; // only used on some platforms.
};

// This is the function called to run a thread
extern void SDL_RunThread(SDL_Thread *thread);

// This is the system-independent thread local storage structure
typedef struct
{
    int limit;
    struct
    {
        void *data;
        void(SDLCALL *destructor)(void *);
    } array[1];
} SDL_TLSData;

// This is how many TLS entries we allocate at once
#define TLS_ALLOC_CHUNKSIZE 4

extern void SDL_InitTLSData(void);
extern void SDL_QuitTLSData(void);

/* Generic TLS support.
   This is only intended as a fallback if getting real thread-local
   storage fails or isn't supported on this platform.
 */
extern void SDL_Generic_InitTLSData(void);
extern SDL_TLSData *SDL_Generic_GetTLSData(void);
extern bool SDL_Generic_SetTLSData(SDL_TLSData *data);
extern void SDL_Generic_QuitTLSData(void);

#endif // SDL_thread_c_h_
