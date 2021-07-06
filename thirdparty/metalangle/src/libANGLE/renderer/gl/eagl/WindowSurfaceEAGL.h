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

// WindowSurfaceEAGL.h: EAGL implementation of egl::Surface

#ifndef LIBANGLE_RENDERER_GL_EAGL_WINDOWSURFACEEAGL_H_
#define LIBANGLE_RENDERER_GL_EAGL_WINDOWSURFACEEAGL_H_

#include "libANGLE/renderer/gl/SurfaceGL.h"

#ifdef __OBJC__
@class EAGLContext;
typedef EAGLContext *EAGLContextObj;
#else
typedef void *EAGLContextObj;
#endif
@class CALayer;
@class CAEAGLLayer;
struct __IOSurface;
typedef __IOSurface *IOSurfaceRef;

namespace rx
{

class DisplayEAGL;
class FramebufferGL;
class FunctionsGL;
class RendererGL;
class StateManagerGL;

class WindowSurfaceEAGL : public SurfaceGL
{
  public:
    WindowSurfaceEAGL(const egl::SurfaceState &state,
                      RendererGL *renderer,
                      EGLNativeWindowType layer,
                      EAGLContextObj context);
    ~WindowSurfaceEAGL() override;

    egl::Error initialize(const egl::Display *display) override;
    egl::Error makeCurrent(const gl::Context *context) override;
    egl::Error unMakeCurrent(const gl::Context *context) override;

    egl::Error swap(const gl::Context *context) override;
    egl::Error postSubBuffer(const gl::Context *context,
                             EGLint x,
                             EGLint y,
                             EGLint width,
                             EGLint height) override;
    egl::Error querySurfacePointerANGLE(EGLint attribute, void **value) override;
    egl::Error bindTexImage(const gl::Context *context,
                            gl::Texture *texture,
                            EGLint buffer) override;
    egl::Error releaseTexImage(const gl::Context *context, EGLint buffer) override;
    void setSwapInterval(EGLint interval) override;

    EGLint getWidth() const override;
    EGLint getHeight() const override;

    EGLint isPostSubBufferSupported() const override;
    EGLint getSwapBehavior() const override;

    FramebufferImpl *createDefaultFramebuffer(const gl::Context *context,
                                              const gl::FramebufferState &state) override;

  private:
    CAEAGLLayer *mSwapLayer;
    CALayer *mLayer;
    EAGLContextObj mContext;
    const FunctionsGL *mFunctions;
    StateManagerGL *mStateManager;

    GLuint mColorRenderbuffer;
    GLuint mDSRenderbuffer;
    EGLint mDSBufferWidth;
    EGLint mDSBufferHeight;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EAGL_WINDOWSURFACEEAGL_H_
