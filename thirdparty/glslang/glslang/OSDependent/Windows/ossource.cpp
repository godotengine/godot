//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "../osinclude.h"

#define STRICT
#define VC_EXTRALEAN 1
#include <windows.h>
#include <cassert>
#include <process.h>
#include <psapi.h>
#include <cstdio>
#include <cstdint>

//
// This file contains the Window-OS-specific functions
//

#if !(defined(_WIN32) || defined(_WIN64))
#error Trying to build a windows specific file in a non windows build.
#endif

namespace glslang {

inline OS_TLSIndex ToGenericTLSIndex (DWORD handle)
{
    return (OS_TLSIndex)((uintptr_t)handle + 1);
}

inline DWORD ToNativeTLSIndex (OS_TLSIndex nIndex)
{
    return (DWORD)((uintptr_t)nIndex - 1);
}

//
// Thread Local Storage Operations
//
OS_TLSIndex OS_AllocTLSIndex()
{
    DWORD dwIndex = TlsAlloc();
    if (dwIndex == TLS_OUT_OF_INDEXES) {
        assert(0 && "OS_AllocTLSIndex(): Unable to allocate Thread Local Storage");
        return OS_INVALID_TLS_INDEX;
    }

    return ToGenericTLSIndex(dwIndex);
}

bool OS_SetTLSValue(OS_TLSIndex nIndex, void *lpvValue)
{
    if (nIndex == OS_INVALID_TLS_INDEX) {
        assert(0 && "OS_SetTLSValue(): Invalid TLS Index");
        return false;
    }

    if (TlsSetValue(ToNativeTLSIndex(nIndex), lpvValue))
        return true;
    else
        return false;
}

void* OS_GetTLSValue(OS_TLSIndex nIndex)
{
    assert(nIndex != OS_INVALID_TLS_INDEX);
    return TlsGetValue(ToNativeTLSIndex(nIndex));
}

bool OS_FreeTLSIndex(OS_TLSIndex nIndex)
{
    if (nIndex == OS_INVALID_TLS_INDEX) {
        assert(0 && "OS_SetTLSValue(): Invalid TLS Index");
        return false;
    }

    if (TlsFree(ToNativeTLSIndex(nIndex)))
        return true;
    else
        return false;
}

HANDLE GlobalLock;

void InitGlobalLock()
{
    GlobalLock = CreateMutex(nullptr, false, nullptr);
}

void GetGlobalLock()
{
    WaitForSingleObject(GlobalLock, INFINITE);
}

void ReleaseGlobalLock()
{
    ReleaseMutex(GlobalLock);
}

unsigned int __stdcall EnterGenericThread (void* entry)
{
    return ((TThreadEntrypoint)entry)(nullptr);
}

//#define DUMP_COUNTERS

void OS_DumpMemoryCounters()
{
#ifdef DUMP_COUNTERS
    PROCESS_MEMORY_COUNTERS counters;
    GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
    printf("Working set size: %d\n", counters.WorkingSetSize);
#else
    printf("Recompile with DUMP_COUNTERS defined to see counters.\n");
#endif
}

} // namespace glslang
