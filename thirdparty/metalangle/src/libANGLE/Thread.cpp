//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Thread.cpp : Defines the Thread class which represents a global EGL thread.

#include "libANGLE/Thread.h"

#include "libANGLE/Context.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/ErrorStrings.h"

namespace egl
{

Thread::Thread()
    : mLabel(nullptr),
      mError(EGL_SUCCESS),
      mAPI(EGL_OPENGL_ES_API),
      mContext(static_cast<gl::Context *>(EGL_NO_CONTEXT))
{}

void Thread::setLabel(EGLLabelKHR label)
{
    mLabel = label;
}

EGLLabelKHR Thread::getLabel() const
{
    return mLabel;
}

void Thread::setSuccess()
{
    mError = EGL_SUCCESS;
}

void Thread::setError(const Error &error,
                      const Debug *debug,
                      const char *command,
                      const LabeledObject *object)
{
    ASSERT(debug != nullptr);

    mError = error.getCode();
    if (error.isError() && !error.getMessage().empty())
    {
        debug->insertMessage(error.getCode(), command, ErrorCodeToMessageType(error.getCode()),
                             getLabel(), object ? object->getLabel() : nullptr, error.getMessage());
    }
}

EGLint Thread::getError() const
{
    return mError;
}

void Thread::setAPI(EGLenum api)
{
    mAPI = api;
}

EGLenum Thread::getAPI() const
{
    return mAPI;
}

void Thread::setCurrent(gl::Context *context)
{
    mContext = context;
}

Surface *Thread::getCurrentDrawSurface() const
{
    if (mContext)
    {
        return mContext->getCurrentDrawSurface();
    }
    return nullptr;
}

Surface *Thread::getCurrentReadSurface() const
{
    if (mContext)
    {
        return mContext->getCurrentReadSurface();
    }
    return nullptr;
}

gl::Context *Thread::getContext() const
{
    return mContext;
}

gl::Context *Thread::getValidContext() const
{
    if (mContext && mContext->isContextLost())
    {
        mContext->handleError(GL_OUT_OF_MEMORY, gl::err::kContextLost, __FILE__, ANGLE_FUNCTION,
                              __LINE__);
        return nullptr;
    }

    return mContext;
}

Display *Thread::getDisplay() const
{
    if (mContext)
    {
        return mContext->getDisplay();
    }
    return nullptr;
}

}  // namespace egl
