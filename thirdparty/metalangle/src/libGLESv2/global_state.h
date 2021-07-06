//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// global_state.h : Defines functions for querying the thread-local GL and EGL state.

#ifndef LIBGLESV2_GLOBALSTATE_H_
#define LIBGLESV2_GLOBALSTATE_H_

#include "libANGLE/Context.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Thread.h"
#include "libANGLE/features.h"

#include <mutex>

namespace egl
{
class Debug;
class Thread;

std::mutex &GetGlobalMutex();
Thread *GetCurrentThread();
Debug *GetDebug();
void SetContextCurrent(Thread *thread, gl::Context *context);
}  // namespace egl

#define ANGLE_SCOPED_GLOBAL_LOCK() \
    std::lock_guard<std::mutex> globalMutexLock(egl::GetGlobalMutex())

namespace gl
{
extern Context *gSingleThreadedContext;

ANGLE_INLINE Context *GetGlobalContext()
{
    if (gSingleThreadedContext)
    {
        return gSingleThreadedContext;
    }

    egl::Thread *thread = egl::GetCurrentThread();
    return thread->getContext();
}

ANGLE_INLINE Context *GetValidGlobalContext()
{
    if (gSingleThreadedContext && !gSingleThreadedContext->isContextLost())
    {
        return gSingleThreadedContext;
    }

    egl::Thread *thread = egl::GetCurrentThread();
    return thread->getValidContext();
}

ANGLE_INLINE std::unique_lock<std::mutex> GetShareGroupLock(const Context *context)
{
    return context->isShared() ? std::unique_lock<std::mutex>(egl::GetGlobalMutex())
                               : std::unique_lock<std::mutex>();
}
}  // namespace gl

#endif  // LIBGLESV2_GLOBALSTATE_H_
