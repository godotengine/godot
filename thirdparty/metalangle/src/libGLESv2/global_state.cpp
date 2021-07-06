//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// global_state.cpp : Implements functions for querying the thread-local GL and EGL state.

#include "libGLESv2/global_state.h"

#include "common/debug.h"
#include "common/platform.h"
#include "common/tls.h"

namespace gl
{
// In single-threaded cases we can avoid a TLS lookup for the current Context.
//
// Let a global single-threaded context have 3 states: unset, set, and multi-threaded.
// Initially it is unset. Then, on MakeCurrent:
//
//  * if the ST context is unset                      -> set the global context.
//  * if the ST context is set and matches the TLS    -> set the global context.
//  * if the ST context is set and does not match TLS -> set multi-threaded mode.
//  * if in multi-threaded mode, unset and subsequently ignore the global context.
//
// Implementation-wise we can use a pointer and a boolean to represent the three modes.
Context *gSingleThreadedContext = nullptr;
bool gIsMultiThreadedContext    = false;
}  // namespace gl

namespace egl
{
namespace
{
static TLSIndex threadTLS = TLS_INVALID_INDEX;
Debug *g_Debug            = nullptr;

ANGLE_REQUIRE_CONSTANT_INIT std::atomic<std::mutex *> g_Mutex(nullptr);
static_assert(std::is_trivially_destructible<decltype(g_Mutex)>::value,
              "global mutex is not trivially destructible");

Thread *AllocateCurrentThread()
{
    ASSERT(threadTLS != TLS_INVALID_INDEX);
    if (threadTLS == TLS_INVALID_INDEX)
    {
        return nullptr;
    }

    Thread *thread = new Thread();
    if (!SetTLSValue(threadTLS, thread))
    {
        ERR() << "Could not set thread local storage.";
        return nullptr;
    }

    return thread;
}

void AllocateDebug()
{
    // All EGL calls use a global lock, this is thread safe
    if (g_Debug == nullptr)
    {
        g_Debug = new Debug();
    }
}

void AllocateMutex()
{
    if (g_Mutex == nullptr)
    {
        std::unique_ptr<std::mutex> newMutex(new std::mutex());
        std::mutex *expected = nullptr;
        if (g_Mutex.compare_exchange_strong(expected, newMutex.get()))
        {
            newMutex.release();
        }
    }
}

}  // anonymous namespace

std::mutex &GetGlobalMutex()
{
    AllocateMutex();
    return *g_Mutex;
}

Thread *GetCurrentThread()
{
    // Create a TLS index if one has not been created for this DLL
    if (threadTLS == TLS_INVALID_INDEX)
    {
        threadTLS = CreateTLSIndex();
    }

    Thread *current = static_cast<Thread *>(GetTLSValue(threadTLS));

    // ANGLE issue 488: when the dll is loaded after thread initialization,
    // thread local storage (current) might not exist yet.
    return (current ? current : AllocateCurrentThread());
}

Debug *GetDebug()
{
    AllocateDebug();
    return g_Debug;
}

void SetContextCurrent(Thread *thread, gl::Context *context)
{
    // See above comment on gGlobalContext.
    // If the context is in multi-threaded mode, ignore the global context.
    if (!gl::gIsMultiThreadedContext)
    {
        // If the global context is unset or matches the current TLS, set the global context.
        if (gl::gSingleThreadedContext == nullptr ||
            gl::gSingleThreadedContext == thread->getContext())
        {
            gl::gSingleThreadedContext = context;
        }
        else
        {
            // If the global context is set and does not match TLS, set multi-threaded mode.
            gl::gSingleThreadedContext  = nullptr;
            gl::gIsMultiThreadedContext = true;
        }
    }
    thread->setCurrent(context);
}
}  // namespace egl

#ifdef ANGLE_PLATFORM_WINDOWS
namespace egl
{

namespace
{

bool DeallocateCurrentThread()
{
    Thread *thread = static_cast<Thread *>(GetTLSValue(threadTLS));
    SafeDelete(thread);
    return SetTLSValue(threadTLS, nullptr);
}

void DeallocateDebug()
{
    SafeDelete(g_Debug);
}

void DeallocateMutex()
{
    std::mutex *mutex = g_Mutex.exchange(nullptr);
    {
        // Wait for the mutex to become released by other threads before deleting.
        std::lock_guard<std::mutex> lock(*mutex);
    }
    SafeDelete(mutex);
}

bool InitializeProcess()
{
    ASSERT(g_Debug == nullptr);
    AllocateDebug();

    AllocateMutex();

    threadTLS = CreateTLSIndex();
    if (threadTLS == TLS_INVALID_INDEX)
    {
        return false;
    }

    return AllocateCurrentThread() != nullptr;
}

bool TerminateProcess()
{
    DeallocateDebug();

    DeallocateMutex();

    if (!DeallocateCurrentThread())
    {
        return false;
    }

    if (threadTLS != TLS_INVALID_INDEX)
    {
        TLSIndex tlsCopy = threadTLS;
        threadTLS        = TLS_INVALID_INDEX;

        if (!DestroyTLSIndex(tlsCopy))
        {
            return false;
        }
    }

    return true;
}

}  // anonymous namespace

}  // namespace egl

extern "C" BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID)
{
    switch (reason)
    {
        case DLL_PROCESS_ATTACH:
            return static_cast<BOOL>(egl::InitializeProcess());

        case DLL_THREAD_ATTACH:
            return static_cast<BOOL>(egl::AllocateCurrentThread() != nullptr);

        case DLL_THREAD_DETACH:
            return static_cast<BOOL>(egl::DeallocateCurrentThread());

        case DLL_PROCESS_DETACH:
            return static_cast<BOOL>(egl::TerminateProcess());
    }

    return TRUE;
}
#endif  // ANGLE_PLATFORM_WINDOWS
