//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FenceNVGL.cpp: Implements the class methods for FenceNVGL.

#include "libANGLE/renderer/gl/FenceNVGL.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"

namespace rx
{

FenceNVGL::FenceNVGL(const FunctionsGL *functions) : FenceNVImpl(), mFunctions(functions)
{
    mFunctions->genFencesNV(1, &mFence);
}

FenceNVGL::~FenceNVGL()
{
    mFunctions->deleteFencesNV(1, &mFence);
    mFence = 0;
}

angle::Result FenceNVGL::set(const gl::Context *context, GLenum condition)
{
    ASSERT(condition == GL_ALL_COMPLETED_NV);
    mFunctions->setFenceNV(mFence, condition);
    return angle::Result::Continue;
}

angle::Result FenceNVGL::test(const gl::Context *context, GLboolean *outFinished)
{
    ASSERT(outFinished);
    *outFinished = mFunctions->testFenceNV(mFence);
    return angle::Result::Continue;
}

angle::Result FenceNVGL::finish(const gl::Context *context)
{
    mFunctions->finishFenceNV(mFence);
    return angle::Result::Continue;
}

// static
bool FenceNVGL::Supported(const FunctionsGL *functions)
{
    return functions->hasGLESExtension("GL_NV_fence") || functions->hasGLExtension("GL_NV_fence");
}

FenceNVSyncGL::FenceNVSyncGL(const FunctionsGL *functions)
    : FenceNVImpl(), mSyncObject(0), mFunctions(functions)
{}

FenceNVSyncGL::~FenceNVSyncGL()
{
    if (mSyncObject != 0)
    {
        mFunctions->deleteSync(mSyncObject);
        mSyncObject = 0;
    }
}

angle::Result FenceNVSyncGL::set(const gl::Context *context, GLenum condition)
{
    ASSERT(condition == GL_ALL_COMPLETED_NV);
    mSyncObject = mFunctions->fenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    ANGLE_CHECK(GetImplAs<ContextGL>(context), mSyncObject != 0,
                "glFenceSync failed to create a GLsync object.", GL_OUT_OF_MEMORY);
    return angle::Result::Continue;
}

angle::Result FenceNVSyncGL::test(const gl::Context *context, GLboolean *outFinished)
{
    ASSERT(mFunctions->isSync(mSyncObject));
    GLint result = 0;
    mFunctions->getSynciv(mSyncObject, GL_SYNC_STATUS, 1, nullptr, &result);
    *outFinished = (result == GL_SIGNALED);
    return angle::Result::Continue;
}

angle::Result FenceNVSyncGL::finish(const gl::Context *context)
{
    ASSERT(mFunctions->isSync(mSyncObject));
    GLenum result =
        mFunctions->clientWaitSync(mSyncObject, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
    ANGLE_CHECK(GetImplAs<ContextGL>(context),
                result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED,
                "glClientWaitSync did not return GL_ALREADY_SIGNALED or GL_CONDITION_SATISFIED.",
                GL_OUT_OF_MEMORY);
    return angle::Result::Continue;
}

// static
bool FenceNVSyncGL::Supported(const FunctionsGL *functions)
{
    return functions->isAtLeastGL(gl::Version(3, 2)) ||
           functions->isAtLeastGLES(gl::Version(3, 0)) || functions->hasGLExtension("GL_ARB_sync");
}

}  // namespace rx
