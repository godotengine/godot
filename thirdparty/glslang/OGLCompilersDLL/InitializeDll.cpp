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

#define SH_EXPORTING

#include <cassert>

#include "InitializeDll.h"
#include "../glslang/Include/InitializeGlobals.h"
#include "../glslang/Public/ShaderLang.h"
#include "../glslang/Include/PoolAlloc.h"

namespace glslang {

OS_TLSIndex ThreadInitializeIndex = OS_INVALID_TLS_INDEX;

// Per-process initialization.
// Needs to be called at least once before parsing, etc. is done.
// Will also do thread initialization for the calling thread; other
// threads will need to do that explicitly.
bool InitProcess()
{
    glslang::GetGlobalLock();

    if (ThreadInitializeIndex != OS_INVALID_TLS_INDEX) {
        //
        // Function is re-entrant.
        //

        glslang::ReleaseGlobalLock();
        return true;
    }

    ThreadInitializeIndex = OS_AllocTLSIndex();

    if (ThreadInitializeIndex == OS_INVALID_TLS_INDEX) {
        assert(0 && "InitProcess(): Failed to allocate TLS area for init flag");

        glslang::ReleaseGlobalLock();
        return false;
    }

    if (! InitializePoolIndex()) {
        assert(0 && "InitProcess(): Failed to initialize global pool");

        glslang::ReleaseGlobalLock();
        return false;
    }

    if (! InitThread()) {
        assert(0 && "InitProcess(): Failed to initialize thread");

        glslang::ReleaseGlobalLock();
        return false;
    }

    glslang::ReleaseGlobalLock();
    return true;
}

// Per-thread scoped initialization.
// Must be called at least once by each new thread sharing the
// symbol tables, etc., needed to parse.
bool InitThread()
{
    //
    // This function is re-entrant
    //
    if (ThreadInitializeIndex == OS_INVALID_TLS_INDEX) {
        assert(0 && "InitThread(): Process hasn't been initalised.");
        return false;
    }

    if (OS_GetTLSValue(ThreadInitializeIndex) != nullptr)
        return true;

    if (! OS_SetTLSValue(ThreadInitializeIndex, (void *)1)) {
        assert(0 && "InitThread(): Unable to set init flag.");
        return false;
    }

    glslang::SetThreadPoolAllocator(nullptr);

    return true;
}

// Not necessary to call this: InitThread() is reentrant, and the need
// to do per thread tear down has been removed.
//
// This is kept, with memory management removed, to satisfy any exiting
// calls to it that rely on it.
bool DetachThread()
{
    bool success = true;

    if (ThreadInitializeIndex == OS_INVALID_TLS_INDEX)
        return true;

    //
    // Function is re-entrant and this thread may not have been initialized.
    //
    if (OS_GetTLSValue(ThreadInitializeIndex) != nullptr) {
        if (!OS_SetTLSValue(ThreadInitializeIndex, nullptr)) {
            assert(0 && "DetachThread(): Unable to clear init flag.");
            success = false;
        }
    }

    return success;
}

// Not necessary to call this: InitProcess() is reentrant.
//
// This is kept, with memory management removed, to satisfy any exiting
// calls to it that rely on it.
//
// Users of glslang should call shFinalize() or glslang::FinalizeProcess() for
// process-scoped memory tear down.
bool DetachProcess()
{
    bool success = true;

    if (ThreadInitializeIndex == OS_INVALID_TLS_INDEX)
        return true;

    success = DetachThread();

    OS_FreeTLSIndex(ThreadInitializeIndex);
    ThreadInitializeIndex = OS_INVALID_TLS_INDEX;

    return success;
}

} // end namespace glslang
