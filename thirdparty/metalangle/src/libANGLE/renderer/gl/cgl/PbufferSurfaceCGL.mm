//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// PBufferSurfaceCGL.cpp: an implementation of egl::Surface for PBuffers for the CLG backend,
//                      currently implemented using renderbuffers

#include "libANGLE/renderer/gl/cgl/PbufferSurfaceCGL.h"

#include "common/debug.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/FramebufferGL.h"
#include "libANGLE/renderer/gl/RendererGL.h"
#include "libANGLE/renderer/gl/StateManagerGL.h"

namespace rx
{

PbufferSurfaceCGL::PbufferSurfaceCGL(const egl::SurfaceState &state,
                                     RendererGL *renderer,
                                     EGLint width,
                                     EGLint height)
    : SurfaceGL(state),
      mWidth(width),
      mHeight(height),
      mFunctions(renderer->getFunctions()),
      mStateManager(renderer->getStateManager()),
      mColorRenderbuffer(0),
      mDSRenderbuffer(0)
{
}

PbufferSurfaceCGL::~PbufferSurfaceCGL()
{
    if (mColorRenderbuffer != 0)
    {
        mFunctions->deleteRenderbuffers(1, &mColorRenderbuffer);
        mColorRenderbuffer = 0;
    }
    if (mDSRenderbuffer != 0)
    {
        mFunctions->deleteRenderbuffers(1, &mDSRenderbuffer);
        mDSRenderbuffer = 0;
    }
}

egl::Error PbufferSurfaceCGL::initialize(const egl::Display *display)
{
    mFunctions->genRenderbuffers(1, &mColorRenderbuffer);
    mStateManager->bindRenderbuffer(GL_RENDERBUFFER, mColorRenderbuffer);
    mFunctions->renderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, mWidth, mHeight);

    mFunctions->genRenderbuffers(1, &mDSRenderbuffer);
    mStateManager->bindRenderbuffer(GL_RENDERBUFFER, mDSRenderbuffer);
    mFunctions->renderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, mWidth, mHeight);

    return egl::NoError();
}

egl::Error PbufferSurfaceCGL::makeCurrent(const gl::Context *context)
{
    return egl::NoError();
}

egl::Error PbufferSurfaceCGL::swap(const gl::Context *context)
{
    return egl::NoError();
}

egl::Error PbufferSurfaceCGL::postSubBuffer(const gl::Context *context,
                                            EGLint x,
                                            EGLint y,
                                            EGLint width,
                                            EGLint height)
{
    return egl::NoError();
}

egl::Error PbufferSurfaceCGL::querySurfacePointerANGLE(EGLint attribute, void **value)
{
    UNIMPLEMENTED();
    return egl::NoError();
}

egl::Error PbufferSurfaceCGL::bindTexImage(const gl::Context *context,
                                           gl::Texture *texture,
                                           EGLint buffer)
{
    UNIMPLEMENTED();
    return egl::NoError();
}

egl::Error PbufferSurfaceCGL::releaseTexImage(const gl::Context *context, EGLint buffer)
{
    UNIMPLEMENTED();
    return egl::NoError();
}

void PbufferSurfaceCGL::setSwapInterval(EGLint interval)
{
}

EGLint PbufferSurfaceCGL::getWidth() const
{
    return mWidth;
}

EGLint PbufferSurfaceCGL::getHeight() const
{
    return mHeight;
}

EGLint PbufferSurfaceCGL::isPostSubBufferSupported() const
{
    UNIMPLEMENTED();
    return EGL_FALSE;
}

EGLint PbufferSurfaceCGL::getSwapBehavior() const
{
    return EGL_BUFFER_PRESERVED;
}

FramebufferImpl *PbufferSurfaceCGL::createDefaultFramebuffer(const gl::Context *context,
                                                             const gl::FramebufferState &state)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    StateManagerGL *stateManager = GetStateManagerGL(context);

    GLuint framebuffer = 0;
    functions->genFramebuffers(1, &framebuffer);
    stateManager->bindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    functions->framebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                       mColorRenderbuffer);
    functions->framebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
                                       mDSRenderbuffer);

    return new FramebufferGL(state, framebuffer, true, false);
}

}  // namespace rx
