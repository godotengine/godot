//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SurfaceImpl.cpp: Implementation of Surface stub method class

#include "libANGLE/renderer/SurfaceImpl.h"

namespace rx
{

SurfaceImpl::SurfaceImpl(const egl::SurfaceState &state) : mState(state) {}

SurfaceImpl::~SurfaceImpl() {}

egl::Error SurfaceImpl::makeCurrent(const gl::Context *context)
{
    return egl::NoError();
}

egl::Error SurfaceImpl::unMakeCurrent(const gl::Context *context)
{
    return egl::NoError();
}

egl::Error SurfaceImpl::swapWithDamage(const gl::Context *context, EGLint *rects, EGLint n_rects)
{
    UNREACHABLE();
    return egl::EglBadSurface() << "swapWithDamage implementation missing.";
}

egl::Error SurfaceImpl::setPresentationTime(EGLnsecsANDROID time)
{
    UNREACHABLE();
    return egl::EglBadSurface() << "setPresentationTime implementation missing.";
}

void SurfaceImpl::setSwapBehavior(EGLint behavior)
{
    UNIMPLEMENTED();
}

void SurfaceImpl::setFixedWidth(EGLint width)
{
    UNREACHABLE();
}

void SurfaceImpl::setFixedHeight(EGLint height)
{
    UNREACHABLE();
}

void SurfaceImpl::setTimestampsEnabled(bool enabled)
{
    UNREACHABLE();
}

const angle::Format *SurfaceImpl::getD3DTextureColorFormat() const
{
    UNREACHABLE();
    return nullptr;
}

egl::SupportedCompositorTimings SurfaceImpl::getSupportedCompositorTimings() const
{
    UNREACHABLE();
    return egl::SupportedCompositorTimings();
}

egl::Error SurfaceImpl::getCompositorTiming(EGLint numTimestamps,
                                            const EGLint *names,
                                            EGLnsecsANDROID *values) const
{
    UNREACHABLE();
    return egl::EglBadDisplay();
}

egl::Error SurfaceImpl::getNextFrameId(EGLuint64KHR *frameId) const
{
    UNREACHABLE();
    return egl::EglBadDisplay();
}

egl::SupportedTimestamps SurfaceImpl::getSupportedTimestamps() const
{
    UNREACHABLE();
    return egl::SupportedTimestamps();
}

egl::Error SurfaceImpl::getFrameTimestamps(EGLuint64KHR frameId,
                                           EGLint numTimestamps,
                                           const EGLint *timestamps,
                                           EGLnsecsANDROID *values) const
{
    UNREACHABLE();
    return egl::EglBadDisplay();
}

}  // namespace rx
