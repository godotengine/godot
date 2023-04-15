/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_machine_windows_api_H
#define __TBB_machine_windows_api_H

#if _WIN32 || _WIN64

#include <windows.h>

#if _WIN32_WINNT < 0x0600
// The following Windows API function is declared explicitly;
// otherwise it fails to compile by VS2005.
#if !defined(WINBASEAPI) || (_WIN32_WINNT < 0x0501 && _MSC_VER == 1400)
#define __TBB_WINBASEAPI extern "C"
#else
#define __TBB_WINBASEAPI WINBASEAPI
#endif
__TBB_WINBASEAPI BOOL WINAPI TryEnterCriticalSection( LPCRITICAL_SECTION );
__TBB_WINBASEAPI BOOL WINAPI InitializeCriticalSectionAndSpinCount( LPCRITICAL_SECTION, DWORD );
// Overloading WINBASEAPI macro and using local functions missing in Windows XP/2003
#define InitializeCriticalSectionEx inlineInitializeCriticalSectionEx
#define CreateSemaphoreEx inlineCreateSemaphoreEx
#define CreateEventEx inlineCreateEventEx
inline BOOL WINAPI inlineInitializeCriticalSectionEx( LPCRITICAL_SECTION lpCriticalSection, DWORD dwSpinCount, DWORD )
{
    return InitializeCriticalSectionAndSpinCount( lpCriticalSection, dwSpinCount );
}
inline HANDLE WINAPI inlineCreateSemaphoreEx( LPSECURITY_ATTRIBUTES lpSemaphoreAttributes, LONG lInitialCount, LONG lMaximumCount, LPCTSTR lpName, DWORD, DWORD )
{
    return CreateSemaphore( lpSemaphoreAttributes, lInitialCount, lMaximumCount, lpName );
}
inline HANDLE WINAPI inlineCreateEventEx( LPSECURITY_ATTRIBUTES lpEventAttributes, LPCTSTR lpName, DWORD dwFlags, DWORD )
{
    BOOL manual_reset = dwFlags&0x00000001 ? TRUE : FALSE; // CREATE_EVENT_MANUAL_RESET
    BOOL initial_set  = dwFlags&0x00000002 ? TRUE : FALSE; // CREATE_EVENT_INITIAL_SET
    return CreateEvent( lpEventAttributes, manual_reset, initial_set, lpName );
}
#endif

#if defined(RTL_SRWLOCK_INIT)
#ifndef __TBB_USE_SRWLOCK
// TODO: turn it on when bug 1952 will be fixed
#define __TBB_USE_SRWLOCK 0
#endif
#endif

#else
#error tbb/machine/windows_api.h should only be used for Windows based platforms
#endif // _WIN32 || _WIN64

#endif // __TBB_machine_windows_api_H
