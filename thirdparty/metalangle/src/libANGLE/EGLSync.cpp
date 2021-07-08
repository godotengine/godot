//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// EGLSync.cpp: Implements the egl::Sync class.

#include "libANGLE/EGLSync.h"

#include "angle_gl.h"

#include "common/utilities.h"
#include "libANGLE/renderer/EGLImplFactory.h"
#include "libANGLE/renderer/EGLSyncImpl.h"

namespace egl
{

Sync::Sync(rx::EGLImplFactory *factory, EGLenum type, const AttributeMap &attribs)
    : mFence(factory->createSync(attribs)),
      mLabel(nullptr),
      mType(type),
      mNativeFenceFD(
          attribs.getAsInt(EGL_SYNC_NATIVE_FENCE_FD_ANDROID, EGL_NO_NATIVE_FENCE_FD_ANDROID))
{}

void Sync::onDestroy(const Display *display)
{
    ASSERT(mFence);
    mFence->onDestroy(display);
    mFence.reset();
}

Sync::~Sync() {}

Error Sync::initialize(const Display *display, const gl::Context *context)
{
    return mFence->initialize(display, context, mType);
}

void Sync::setLabel(EGLLabelKHR label)
{
    mLabel = label;
}

EGLLabelKHR Sync::getLabel() const
{
    return mLabel;
}

Error Sync::clientWait(const Display *display,
                       const gl::Context *context,
                       EGLint flags,
                       EGLTime timeout,
                       EGLint *outResult)
{
    return mFence->clientWait(display, context, flags, timeout, outResult);
}

Error Sync::serverWait(const Display *display, const gl::Context *context, EGLint flags)
{
    return mFence->serverWait(display, context, flags);
}

Error Sync::getStatus(const Display *display, EGLint *outStatus) const
{
    return mFence->getStatus(display, outStatus);
}

Error Sync::dupNativeFenceFD(const Display *display, EGLint *result) const
{
    return mFence->dupNativeFenceFD(display, result);
}

}  // namespace egl
