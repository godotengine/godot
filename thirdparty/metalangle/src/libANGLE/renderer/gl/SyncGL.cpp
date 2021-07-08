//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SyncGL.cpp: Implements the class methods for SyncGL.

#include "libANGLE/renderer/gl/SyncGL.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"

namespace rx
{

SyncGL::SyncGL(const FunctionsGL *functions) : SyncImpl(), mFunctions(functions), mSyncObject(0)
{
    ASSERT(mFunctions);
}

SyncGL::~SyncGL()
{
    ASSERT(mSyncObject == 0);
}

void SyncGL::onDestroy(const gl::Context *context)
{
    ASSERT(mSyncObject != 0);
    mFunctions->deleteSync(mSyncObject);
    mSyncObject = 0;
}

angle::Result SyncGL::set(const gl::Context *context, GLenum condition, GLbitfield flags)
{
    ASSERT(condition == GL_SYNC_GPU_COMMANDS_COMPLETE && flags == 0);
    mSyncObject = mFunctions->fenceSync(condition, flags);
    ANGLE_CHECK(GetImplAs<ContextGL>(context), mSyncObject != 0,
                "glFenceSync failed to create a GLsync object.", GL_OUT_OF_MEMORY);
    return angle::Result::Continue;
}

angle::Result SyncGL::clientWait(const gl::Context *context,
                                 GLbitfield flags,
                                 GLuint64 timeout,
                                 GLenum *outResult)
{
    ASSERT(mSyncObject != 0);
    *outResult = mFunctions->clientWaitSync(mSyncObject, flags, timeout);
    return angle::Result::Continue;
}

angle::Result SyncGL::serverWait(const gl::Context *context, GLbitfield flags, GLuint64 timeout)
{
    ASSERT(mSyncObject != 0);
    mFunctions->waitSync(mSyncObject, flags, timeout);
    return angle::Result::Continue;
}

angle::Result SyncGL::getStatus(const gl::Context *context, GLint *outResult)
{
    ASSERT(mSyncObject != 0);
    mFunctions->getSynciv(mSyncObject, GL_SYNC_STATUS, 1, nullptr, outResult);
    return angle::Result::Continue;
}
}  // namespace rx
