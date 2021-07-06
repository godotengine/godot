//
// Copyright (c) 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SyncMtl:
//    Defines the class interface for SyncMtl, implementing SyncImpl.
//

#include "libANGLE/renderer/metal/SyncMtl.h"

#include <chrono>

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"

namespace rx
{
namespace mtl
{
// SharedEvent is only available on iOS 12.0+ or mac 10.14+
#if defined(__IPHONE_12_0) || defined(__MAC_10_14)
Sync::Sync() {}
Sync::~Sync() {}

void Sync::onDestroy()
{
    mMetalSharedEvent = nil;
    mCv               = nullptr;
    mLock             = nullptr;
}

angle::Result Sync::initialize(ContextMtl *contextMtl)
{
    ANGLE_MTL_OBJC_SCOPE
    {
        mMetalSharedEvent = [[contextMtl->getMetalDevice() newSharedEvent] ANGLE_MTL_AUTORELEASE];
    }

    mSetCounter = mMetalSharedEvent.get().signaledValue;

    mCv.reset(new std::condition_variable());
    mLock.reset(new std::mutex());
    return angle::Result::Continue;
}

angle::Result Sync::set(ContextMtl *contextMtl, GLenum condition, GLbitfield flags)
{
    if (!mMetalSharedEvent)
    {
        ANGLE_TRY(initialize(contextMtl));
    }
    ASSERT(condition == GL_SYNC_GPU_COMMANDS_COMPLETE);
    ASSERT(flags == 0);

    mSetCounter++;
    contextMtl->queueEventSignal(mMetalSharedEvent, mSetCounter);
    return angle::Result::Continue;
}
angle::Result Sync::clientWait(ContextMtl *contextMtl,
                               bool flushCommands,
                               uint64_t timeout,
                               GLenum *outResult)
{
    std::unique_lock<std::mutex> lg(*mLock);
    if (mMetalSharedEvent.get().signaledValue >= mSetCounter)
    {
        *outResult = GL_ALREADY_SIGNALED;
        return angle::Result::Continue;
    }
    if (flushCommands)
    {
        contextMtl->flushCommandBufer();
    }

    if (timeout == 0)
    {
        *outResult = GL_TIMEOUT_EXPIRED;

        return angle::Result::Continue;
    }

    // Create references to mutex and condition variable since they might be released in
    // onDestroy(), but the callback might still not be fired yet.
    std::shared_ptr<std::condition_variable> cvRef = mCv;
    std::shared_ptr<std::mutex> lockRef            = mLock;

    AutoObjCObj<MTLSharedEventListener> eventListener =
        contextMtl->getDisplay()->getOrCreateSharedEventListener();
    [mMetalSharedEvent.get() notifyListener:eventListener
                                    atValue:mSetCounter
                                      block:^(id<MTLSharedEvent> sharedEvent, uint64_t value) {
                                        std::unique_lock<std::mutex> lg(*lockRef);
                                        cvRef->notify_one();
                                      }];

    if (!mCv->wait_for(lg, std::chrono::nanoseconds(timeout),
                       [this] { return mMetalSharedEvent.get().signaledValue >= mSetCounter; }))
    {
        *outResult = GL_TIMEOUT_EXPIRED;
        return angle::Result::Incomplete;
    }

    ASSERT(mMetalSharedEvent.get().signaledValue >= mSetCounter);
    *outResult = GL_CONDITION_SATISFIED;

    return angle::Result::Continue;
}
void Sync::serverWait(ContextMtl *contextMtl)
{
    contextMtl->serverWaitEvent(mMetalSharedEvent, mSetCounter);
}
angle::Result Sync::getStatus(bool *signaled)
{
    *signaled = mMetalSharedEvent.get().signaledValue >= mSetCounter;
    return angle::Result::Continue;
}
#endif  // #if defined(__IPHONE_12_0) || defined(__MAC_10_14)
}  // namespace mtl

// FenceNVMtl implementation
FenceNVMtl::FenceNVMtl() : FenceNVImpl() {}

FenceNVMtl::~FenceNVMtl() {}

angle::Result FenceNVMtl::set(const gl::Context *context, GLenum condition)
{
    ASSERT(condition == GL_ALL_COMPLETED_NV);
    ContextMtl *contextMtl = mtl::GetImpl(context);
    return mSync.set(contextMtl, GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

angle::Result FenceNVMtl::test(const gl::Context *context, GLboolean *outFinished)
{
    bool signaled = false;
    ANGLE_TRY(mSync.getStatus(&signaled));

    *outFinished = signaled ? GL_TRUE : GL_FALSE;
    return angle::Result::Continue;
}

angle::Result FenceNVMtl::finish(const gl::Context *context)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    uint64_t timeout       = 1000000000ul;
    GLenum result;
    do
    {
        ANGLE_TRY(mSync.clientWait(contextMtl, true, timeout, &result));
    } while (result == GL_TIMEOUT_EXPIRED);

    if (result == GL_WAIT_FAILED)
    {
        UNREACHABLE();
        return angle::Result::Stop;
    }

    return angle::Result::Continue;
}

// SyncMtl implementation
SyncMtl::SyncMtl() : SyncImpl() {}

SyncMtl::~SyncMtl() {}

void SyncMtl::onDestroy(const gl::Context *context)
{
    mSync.onDestroy();
}

angle::Result SyncMtl::set(const gl::Context *context, GLenum condition, GLbitfield flags)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    return mSync.set(contextMtl, condition, flags);
}

angle::Result SyncMtl::clientWait(const gl::Context *context,
                                  GLbitfield flags,
                                  GLuint64 timeout,
                                  GLenum *outResult)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);

    ASSERT((flags & ~GL_SYNC_FLUSH_COMMANDS_BIT) == 0);

    bool flush = (flags & GL_SYNC_FLUSH_COMMANDS_BIT) != 0;

    return mSync.clientWait(contextMtl, flush, timeout, outResult);
}

angle::Result SyncMtl::serverWait(const gl::Context *context, GLbitfield flags, GLuint64 timeout)
{
    ASSERT(flags == 0);
    ASSERT(timeout == GL_TIMEOUT_IGNORED);

    ContextMtl *contextMtl = mtl::GetImpl(context);
    mSync.serverWait(contextMtl);
    return angle::Result::Continue;
}

angle::Result SyncMtl::getStatus(const gl::Context *context, GLint *outResult)
{
    bool signaled = false;
    ANGLE_TRY(mSync.getStatus(&signaled));

    *outResult = signaled ? GL_SIGNALED : GL_UNSIGNALED;
    return angle::Result::Continue;
}

// EGLSyncMtl implementation
EGLSyncMtl::EGLSyncMtl(const egl::AttributeMap &attribs) : EGLSyncImpl()
{
    ASSERT(attribs.isEmpty());
}

EGLSyncMtl::~EGLSyncMtl() {}

void EGLSyncMtl::onDestroy(const egl::Display *display)
{
    mSync.onDestroy();
}

egl::Error EGLSyncMtl::initialize(const egl::Display *display,
                                  const gl::Context *context,
                                  EGLenum type)
{
    ASSERT(type == EGL_SYNC_FENCE_KHR);
    ASSERT(context != nullptr);

    ContextMtl *contextMtl = mtl::GetImpl(context);
    if (IsError(mSync.set(contextMtl, GL_SYNC_GPU_COMMANDS_COMPLETE, 0)))
    {
        return egl::Error(EGL_BAD_ALLOC, "eglCreateSyncKHR failed to create sync object");
    }

    return egl::NoError();
}

egl::Error EGLSyncMtl::clientWait(const egl::Display *display,
                                  const gl::Context *context,
                                  EGLint flags,
                                  EGLTime timeout,
                                  EGLint *outResult)
{
    ASSERT((flags & ~EGL_SYNC_FLUSH_COMMANDS_BIT_KHR) == 0);

    bool flush = (flags & EGL_SYNC_FLUSH_COMMANDS_BIT_KHR) != 0;
    GLenum result;
    ContextMtl *contextMtl = mtl::GetImpl(context);
    if (IsError(mSync.clientWait(contextMtl, flush, static_cast<uint64_t>(timeout), &result)))
    {
        return egl::Error(EGL_BAD_ALLOC);
    }

    switch (result)
    {
        case GL_ALREADY_SIGNALED:
            // fall through.  EGL doesn't differentiate between event being already set, or set
            // before timeout.
        case GL_CONDITION_SATISFIED:
            *outResult = EGL_CONDITION_SATISFIED_KHR;
            return egl::NoError();

        case GL_TIMEOUT_EXPIRED:
            *outResult = EGL_TIMEOUT_EXPIRED_KHR;
            return egl::NoError();

        default:
            UNREACHABLE();
            *outResult = EGL_FALSE;
            return egl::Error(EGL_BAD_ALLOC);
    }
}

egl::Error EGLSyncMtl::serverWait(const egl::Display *display,
                                  const gl::Context *context,
                                  EGLint flags)
{
    // Server wait requires a valid bound context.
    ASSERT(context);

    // No flags are currently implemented.
    ASSERT(flags == 0);

    ContextMtl *contextMtl = mtl::GetImpl(context);
    mSync.serverWait(contextMtl);
    return egl::NoError();
}

egl::Error EGLSyncMtl::getStatus(const egl::Display *display, EGLint *outStatus)
{
    bool signaled = false;
    if (IsError(mSync.getStatus(&signaled)))
    {
        return egl::Error(EGL_BAD_ALLOC);
    }

    *outStatus = signaled ? EGL_SIGNALED_KHR : EGL_UNSIGNALED_KHR;
    return egl::NoError();
}

egl::Error EGLSyncMtl::dupNativeFenceFD(const egl::Display *display, EGLint *result) const
{
    UNREACHABLE();
    return egl::EglBadDisplay();
}

}
