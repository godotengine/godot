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

#ifdef SDL_THREAD_WINDOWS

// Win32 thread management routines for SDL

#include "../SDL_thread_c.h"
#include "../SDL_systhread.h"
#include "SDL_systhread_c.h"

#ifndef STACK_SIZE_PARAM_IS_A_RESERVATION
#define STACK_SIZE_PARAM_IS_A_RESERVATION 0x00010000
#endif

#define SDL_DEBUGGER_NAME_EXCEPTION_CODE 0x406D1388

typedef void (__cdecl * SDL_EndThreadExCallback) (unsigned retval);
typedef uintptr_t (__cdecl * SDL_BeginThreadExCallback)
                   (void *security, unsigned stacksize, unsigned (__stdcall *startaddr)(void *),
                    void * arglist, unsigned initflag, unsigned *threadaddr);

static DWORD RunThread(void *data)
{
    SDL_Thread *thread = (SDL_Thread *)data;
    SDL_EndThreadExCallback pfnEndThread = (SDL_EndThreadExCallback)thread->endfunc;
    SDL_RunThread(thread);
    if (pfnEndThread) {
        pfnEndThread(0);
    }
    return 0;
}

static DWORD WINAPI MINGW32_FORCEALIGN RunThreadViaCreateThread(LPVOID data)
{
    return RunThread(data);
}

static unsigned __stdcall MINGW32_FORCEALIGN RunThreadViaBeginThreadEx(void *data)
{
    return (unsigned)RunThread(data);
}

bool SDL_SYS_CreateThread(SDL_Thread *thread,
                          SDL_FunctionPointer vpfnBeginThread,
                          SDL_FunctionPointer vpfnEndThread)
{
    SDL_BeginThreadExCallback pfnBeginThread = (SDL_BeginThreadExCallback) vpfnBeginThread;

    const DWORD flags = thread->stacksize ? STACK_SIZE_PARAM_IS_A_RESERVATION : 0;

    // Save the function which we will have to call to clear the RTL of calling app!
    thread->endfunc = vpfnEndThread;

    // thread->stacksize == 0 means "system default", same as win32 expects
    if (pfnBeginThread) {
        unsigned threadid = 0;
        thread->handle = (SYS_ThreadHandle)((size_t)pfnBeginThread(NULL, (unsigned int)thread->stacksize,
                                                                   RunThreadViaBeginThreadEx,
                                                                   thread, flags, &threadid));
    } else {
        DWORD threadid = 0;
        thread->handle = CreateThread(NULL, thread->stacksize,
                                      RunThreadViaCreateThread,
                                      thread, flags, &threadid);
    }
    if (!thread->handle) {
        return SDL_SetError("Not enough resources to create thread");
    }
    return true;
}

#pragma pack(push, 8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType;     // must be 0x1000
    LPCSTR szName;    // pointer to name (in user addr space)
    DWORD dwThreadID; // thread ID (-1=caller thread)
    DWORD dwFlags;    // reserved for future use, must be zero
} THREADNAME_INFO;
#pragma pack(pop)

static LONG NTAPI EmptyVectoredExceptionHandler(EXCEPTION_POINTERS *info)
{
    if (info != NULL && info->ExceptionRecord != NULL && info->ExceptionRecord->ExceptionCode == SDL_DEBUGGER_NAME_EXCEPTION_CODE) {
        return EXCEPTION_CONTINUE_EXECUTION;
    } else {
        return EXCEPTION_CONTINUE_SEARCH;
    }
}

typedef HRESULT(WINAPI *pfnSetThreadDescription)(HANDLE, PCWSTR);

void SDL_SYS_SetupThread(const char *name)
{
    if (name) {
        PVOID exceptionHandlerHandle;
        static pfnSetThreadDescription pSetThreadDescription = NULL;
        static HMODULE kernel32 = NULL;

        if (!kernel32) {
            kernel32 = GetModuleHandle(TEXT("kernel32.dll"));
            if (kernel32) {
                pSetThreadDescription = (pfnSetThreadDescription)GetProcAddress(kernel32, "SetThreadDescription");
            }
            if (!kernel32 || !pSetThreadDescription) {
                HMODULE kernelBase = GetModuleHandle(TEXT("KernelBase.dll"));
                if (kernelBase) {
                    pSetThreadDescription = (pfnSetThreadDescription)GetProcAddress(kernelBase, "SetThreadDescription");
                }
            }
        }

        if (pSetThreadDescription) {
            WCHAR *strw = WIN_UTF8ToStringW(name);
            if (strw) {
                pSetThreadDescription(GetCurrentThread(), strw);
                SDL_free(strw);
            }
        }

        /* Presumably some version of Visual Studio will understand SetThreadDescription(),
           but we still need to deal with older OSes and debuggers. Set it with the arcane
           exception magic, too. */

        exceptionHandlerHandle = AddVectoredExceptionHandler(1, EmptyVectoredExceptionHandler);
        if (exceptionHandlerHandle) {
            THREADNAME_INFO inf;
            // This magic tells the debugger to name a thread if it's listening.
            SDL_zero(inf);
            inf.dwType = 0x1000;
            inf.szName = name;
            inf.dwThreadID = (DWORD)-1;
            inf.dwFlags = 0;

            // The debugger catches this, renames the thread, continues on.
            RaiseException(SDL_DEBUGGER_NAME_EXCEPTION_CODE, 0, sizeof(inf) / sizeof(ULONG_PTR), (const ULONG_PTR *)&inf);
            RemoveVectoredExceptionHandler(exceptionHandlerHandle);
        }
    }
}

SDL_ThreadID SDL_GetCurrentThreadID(void)
{
    return (SDL_ThreadID)GetCurrentThreadId();
}

bool SDL_SYS_SetThreadPriority(SDL_ThreadPriority priority)
{
    int value;

    if (priority == SDL_THREAD_PRIORITY_LOW) {
        value = THREAD_PRIORITY_LOWEST;
    } else if (priority == SDL_THREAD_PRIORITY_HIGH) {
        value = THREAD_PRIORITY_HIGHEST;
    } else if (priority == SDL_THREAD_PRIORITY_TIME_CRITICAL) {
        value = THREAD_PRIORITY_TIME_CRITICAL;
    } else {
        value = THREAD_PRIORITY_NORMAL;
    }
    if (!SetThreadPriority(GetCurrentThread(), value)) {
        return WIN_SetError("SetThreadPriority()");
    }
    return true;
}

void SDL_SYS_WaitThread(SDL_Thread *thread)
{
    WaitForSingleObjectEx(thread->handle, INFINITE, FALSE);
    CloseHandle(thread->handle);
}

void SDL_SYS_DetachThread(SDL_Thread *thread)
{
    CloseHandle(thread->handle);
}

#endif // SDL_THREAD_WINDOWS
