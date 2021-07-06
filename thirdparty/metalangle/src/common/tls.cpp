//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// tls.cpp: Simple cross-platform interface for thread local storage.

#include "common/tls.h"

#include <assert.h>

#ifdef ANGLE_ENABLE_WINDOWS_UWP
#    include <map>
#    include <mutex>
#    include <set>
#    include <vector>

#    include <Windows.System.Threading.h>
#    include <wrl/async.h>
#    include <wrl/client.h>

using namespace std;
using namespace Windows::Foundation;
using namespace ABI::Windows::System::Threading;

// Thread local storage for Windows Store support
typedef vector<void *> ThreadLocalData;

static __declspec(thread) ThreadLocalData *currentThreadData = nullptr;
static set<ThreadLocalData *> allThreadData;
static DWORD nextTlsIndex = 0;
static vector<DWORD> freeTlsIndices;

#endif

TLSIndex CreateTLSIndex()
{
    TLSIndex index;

#ifdef ANGLE_PLATFORM_WINDOWS
#    ifdef ANGLE_ENABLE_WINDOWS_UWP
    if (!freeTlsIndices.empty())
    {
        DWORD result = freeTlsIndices.back();
        freeTlsIndices.pop_back();
        index = result;
    }
    else
    {
        index = nextTlsIndex++;
    }
#    else
    index = TlsAlloc();
#    endif

#elif defined(ANGLE_PLATFORM_POSIX)
    // Create global pool key
    if ((pthread_key_create(&index, nullptr)) != 0)
    {
        index = TLS_INVALID_INDEX;
    }
#endif

    assert(index != TLS_INVALID_INDEX &&
           "CreateTLSIndex(): Unable to allocate Thread Local Storage");
    return index;
}

bool DestroyTLSIndex(TLSIndex index)
{
    assert(index != TLS_INVALID_INDEX && "DestroyTLSIndex(): Invalid TLS Index");
    if (index == TLS_INVALID_INDEX)
    {
        return false;
    }

#ifdef ANGLE_PLATFORM_WINDOWS
#    ifdef ANGLE_ENABLE_WINDOWS_UWP
    assert(index < nextTlsIndex);
    assert(find(freeTlsIndices.begin(), freeTlsIndices.end(), index) == freeTlsIndices.end());

    freeTlsIndices.push_back(index);
    for (auto threadData : allThreadData)
    {
        if (threadData->size() > index)
        {
            threadData->at(index) = nullptr;
        }
    }
    return true;
#    else
    return (TlsFree(index) == TRUE);
#    endif
#elif defined(ANGLE_PLATFORM_POSIX)
    return (pthread_key_delete(index) == 0);
#endif
}

bool SetTLSValue(TLSIndex index, void *value)
{
    assert(index != TLS_INVALID_INDEX && "SetTLSValue(): Invalid TLS Index");
    if (index == TLS_INVALID_INDEX)
    {
        return false;
    }

#ifdef ANGLE_PLATFORM_WINDOWS
#    ifdef ANGLE_ENABLE_WINDOWS_UWP
    ThreadLocalData *threadData = currentThreadData;
    if (!threadData)
    {
        threadData = new ThreadLocalData(index + 1, nullptr);
        allThreadData.insert(threadData);
        currentThreadData = threadData;
    }
    else if (threadData->size() <= index)
    {
        threadData->resize(index + 1, nullptr);
    }

    threadData->at(index) = value;
    return true;
#    else
    return (TlsSetValue(index, value) == TRUE);
#    endif
#elif defined(ANGLE_PLATFORM_POSIX)
    return (pthread_setspecific(index, value) == 0);
#endif
}

void *GetTLSValue(TLSIndex index)
{
    assert(index != TLS_INVALID_INDEX && "GetTLSValue(): Invalid TLS Index");
    if (index == TLS_INVALID_INDEX)
    {
        return nullptr;
    }

#ifdef ANGLE_PLATFORM_WINDOWS
#    ifdef ANGLE_ENABLE_WINDOWS_UWP
    ThreadLocalData *threadData = currentThreadData;
    if (threadData && threadData->size() > index)
    {
        return threadData->at(index);
    }
    else
    {
        return nullptr;
    }
#    else
    return TlsGetValue(index);
#    endif
#elif defined(ANGLE_PLATFORM_POSIX)
    return pthread_getspecific(index);
#endif
}
