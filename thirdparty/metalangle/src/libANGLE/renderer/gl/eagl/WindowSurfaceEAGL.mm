/*
 * Copyright (C) 2019 Apple Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY APPLE INC. ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL APPLE INC. OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// WindowSurfaceEAGL.cpp: EAGL implementation of egl::Surface

#import "common/platform.h"

#if defined(ANGLE_PLATFORM_IOS) && !defined(ANGLE_PLATFORM_MACCATALYST)

#    import "libANGLE/renderer/gl/eagl/WindowSurfaceEAGL.h"

#    import "common/debug.h"
#    import "libANGLE/Context.h"
#    import "libANGLE/renderer/gl/FramebufferGL.h"
#    import "libANGLE/renderer/gl/RendererGL.h"
#    import "libANGLE/renderer/gl/StateManagerGL.h"
#    import "libANGLE/renderer/gl/eagl/DisplayEAGL.h"

#    import <OpenGLES/EAGL.h>
#    import <QuartzCore/QuartzCore.h>

namespace rx
{

WindowSurfaceEAGL::WindowSurfaceEAGL(const egl::SurfaceState &state,
                                     RendererGL *renderer,
                                     EGLNativeWindowType layer,
                                     EAGLContextObj context)
    : SurfaceGL(state),
      mSwapLayer(nil),
      mLayer(reinterpret_cast<CALayer *>(layer)),
      mContext(context),
      mFunctions(renderer->getFunctions()),
      mStateManager(renderer->getStateManager()),
      mColorRenderbuffer(0),
      mDSRenderbuffer(0),
      mDSBufferWidth(0),
      mDSBufferHeight(0)
{}

WindowSurfaceEAGL::~WindowSurfaceEAGL()
{
    if (mColorRenderbuffer)
    {
        mFunctions->deleteRenderbuffers(1, &mColorRenderbuffer);
        mColorRenderbuffer = 0;
    }
    if (mDSRenderbuffer != 0)
    {
        mFunctions->deleteRenderbuffers(1, &mDSRenderbuffer);
        mDSRenderbuffer = 0;
    }

    if (mSwapLayer != nil)
    {
        [mSwapLayer removeFromSuperlayer];
        [mSwapLayer release];
        mSwapLayer = nil;
    }
}

egl::Error WindowSurfaceEAGL::initialize(const egl::Display *display)
{
    unsigned width = mDSBufferWidth = getWidth();
    unsigned height = mDSBufferHeight = getHeight();

    mSwapLayer       = [[CAEAGLLayer alloc] init];
    mSwapLayer.frame = mLayer.frame;
    [mLayer addSublayer:mSwapLayer];
    [mSwapLayer setContentsScale:[mLayer contentsScale]];

    mFunctions->genRenderbuffers(1, &mColorRenderbuffer);
    mStateManager->bindRenderbuffer(GL_RENDERBUFFER, mColorRenderbuffer);
    [mContext renderbufferStorage:GL_RENDERBUFFER fromDrawable:mSwapLayer];

    mFunctions->genRenderbuffers(1, &mDSRenderbuffer);
    mStateManager->bindRenderbuffer(GL_RENDERBUFFER, mDSRenderbuffer);
    mFunctions->renderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

    return egl::Error(EGL_SUCCESS);
}

egl::Error WindowSurfaceEAGL::makeCurrent(const gl::Context *context)
{
    [EAGLContext setCurrentContext:mContext];
    return egl::Error(EGL_SUCCESS);
}

egl::Error WindowSurfaceEAGL::unMakeCurrent(const gl::Context *context)
{
    [EAGLContext setCurrentContext:nil];
    return egl::Error(EGL_SUCCESS);
}

egl::Error WindowSurfaceEAGL::swap(const gl::Context *context)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    StateManagerGL *stateManager = GetStateManagerGL(context);

    functions->flush();

    stateManager->bindRenderbuffer(GL_RENDERBUFFER, mColorRenderbuffer);
    [mContext presentRenderbuffer:GL_RENDERBUFFER];

    unsigned width  = getWidth();
    unsigned height = getHeight();

    if (mDSBufferWidth != width || mDSBufferHeight != height)
    {
        // Resize color, depth stencil buffer
        mSwapLayer.frame = mLayer.frame;

        mStateManager->bindRenderbuffer(GL_RENDERBUFFER, mColorRenderbuffer);
        [mContext renderbufferStorage:GL_RENDERBUFFER fromDrawable:mSwapLayer];

        stateManager->bindRenderbuffer(GL_RENDERBUFFER, mDSRenderbuffer);
        functions->renderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

        mDSBufferWidth  = width;
        mDSBufferHeight = height;
    }

    return egl::Error(EGL_SUCCESS);
}

egl::Error WindowSurfaceEAGL::postSubBuffer(const gl::Context *context,
                                            EGLint x,
                                            EGLint y,
                                            EGLint width,
                                            EGLint height)
{
    UNIMPLEMENTED();
    return egl::Error(EGL_SUCCESS);
}

egl::Error WindowSurfaceEAGL::querySurfacePointerANGLE(EGLint attribute, void **value)
{
    UNIMPLEMENTED();
    return egl::Error(EGL_SUCCESS);
}

egl::Error WindowSurfaceEAGL::bindTexImage(const gl::Context *context,
                                           gl::Texture *texture,
                                           EGLint buffer)
{
    UNIMPLEMENTED();
    return egl::Error(EGL_SUCCESS);
}

egl::Error WindowSurfaceEAGL::releaseTexImage(const gl::Context *context, EGLint buffer)
{
    UNIMPLEMENTED();
    return egl::Error(EGL_SUCCESS);
}

void WindowSurfaceEAGL::setSwapInterval(EGLint interval)
{
    // TODO(cwallez) investigate implementing swap intervals other than 0
}

EGLint WindowSurfaceEAGL::getWidth() const
{
    return static_cast<EGLint>(CGRectGetWidth([mLayer frame]) * [mLayer contentsScale]);
}

EGLint WindowSurfaceEAGL::getHeight() const
{
    return static_cast<EGLint>(CGRectGetHeight([mLayer frame]) * [mLayer contentsScale]);
}

EGLint WindowSurfaceEAGL::isPostSubBufferSupported() const
{
    UNIMPLEMENTED();
    return EGL_FALSE;
}

EGLint WindowSurfaceEAGL::getSwapBehavior() const
{
    return EGL_BUFFER_DESTROYED;
}

FramebufferImpl *WindowSurfaceEAGL::createDefaultFramebuffer(const gl::Context *context,
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

#endif  // defined(ANGLE_PLATFORM_IOS)
