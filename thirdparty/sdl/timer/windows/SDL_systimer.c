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

#ifdef SDL_TIMER_WINDOWS

#include "../../core/windows/SDL_windows.h"

/* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION flag was added in Windows 10 version 1803. */
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x2
#endif

typedef HANDLE (WINAPI *CreateWaitableTimerExW_t)(LPSECURITY_ATTRIBUTES lpTimerAttributes, LPCWSTR lpTimerName, DWORD dwFlags, DWORD dwDesiredAccess);
static CreateWaitableTimerExW_t pCreateWaitableTimerExW;

typedef BOOL (WINAPI *SetWaitableTimerEx_t)(HANDLE hTimer, const LARGE_INTEGER *lpDueTime, LONG lPeriod, PTIMERAPCROUTINE pfnCompletionRoutine, LPVOID lpArgToCompletionRoutine, PREASON_CONTEXT WakeContext, ULONG TolerableDelay);
static SetWaitableTimerEx_t pSetWaitableTimerEx;

static void SDL_CleanupWaitableHandle(void *handle)
{
    CloseHandle(handle);
}

static HANDLE SDL_GetWaitableTimer(void)
{
    static SDL_TLSID TLS_timer_handle;
    HANDLE timer;

    if (!pCreateWaitableTimerExW || !pSetWaitableTimerEx) {
        static bool initialized;

        if (!initialized) {
            HMODULE module = GetModuleHandle(TEXT("kernel32.dll"));
            if (module) {
                pCreateWaitableTimerExW = (CreateWaitableTimerExW_t)GetProcAddress(module, "CreateWaitableTimerExW");
                pSetWaitableTimerEx = (SetWaitableTimerEx_t)GetProcAddress(module, "SetWaitableTimerEx");
            }
            initialized = true;
        }

        if (!pCreateWaitableTimerExW || !pSetWaitableTimerEx) {
            return NULL;
        }
    }

    timer = SDL_GetTLS(&TLS_timer_handle);
    if (!timer) {
        timer = pCreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (timer) {
            SDL_SetTLS(&TLS_timer_handle, timer, SDL_CleanupWaitableHandle);
        }
    }
    return timer;
}

static HANDLE SDL_GetWaitableEvent(void)
{
    static SDL_TLSID TLS_event_handle;
    HANDLE event;

    event = SDL_GetTLS(&TLS_event_handle);
    if (!event) {
        event = CreateEvent(NULL, FALSE, FALSE, NULL);
        if (event) {
            SDL_SetTLS(&TLS_event_handle, event, SDL_CleanupWaitableHandle);
        }
    }
    return event;
}

Uint64 SDL_GetPerformanceCounter(void)
{
    LARGE_INTEGER counter;
    const BOOL rc = QueryPerformanceCounter(&counter);
    SDL_assert(rc != 0); // this should _never_ fail if you're on XP or later.
    return (Uint64)counter.QuadPart;
}

Uint64 SDL_GetPerformanceFrequency(void)
{
    LARGE_INTEGER frequency;
    const BOOL rc = QueryPerformanceFrequency(&frequency);
    SDL_assert(rc != 0); // this should _never_ fail if you're on XP or later.
    return (Uint64)frequency.QuadPart;
}

void SDL_SYS_DelayNS(Uint64 ns)
{
    HANDLE timer = SDL_GetWaitableTimer();
    if (timer) {
        LARGE_INTEGER due_time;
        due_time.QuadPart = -((LONGLONG)ns / 100);
        if (pSetWaitableTimerEx(timer, &due_time, 0, NULL, NULL, NULL, 0)) {
            WaitForSingleObject(timer, INFINITE);
        }
        return;
    }

    const Uint64 max_delay = 0xffffffffLL * SDL_NS_PER_MS;
    if (ns > max_delay) {
        ns = max_delay;
    }
    const DWORD delay = (DWORD)SDL_NS_TO_MS(ns);

    HANDLE event = SDL_GetWaitableEvent();
    if (event) {
        WaitForSingleObjectEx(event, delay, FALSE);
        return;
    }

    Sleep(delay);
}

#endif // SDL_TIMER_WINDOWS
