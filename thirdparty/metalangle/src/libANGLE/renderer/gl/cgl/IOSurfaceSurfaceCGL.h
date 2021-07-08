//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// PBufferSurfaceCGL.h: an implementation of PBuffers created from IOSurfaces using
//                      EGL_ANGLE_iosurface_client_buffer

#ifndef LIBANGLE_RENDERER_GL_CGL_IOSURFACESURFACECGL_H_
#define LIBANGLE_RENDERER_GL_CGL_IOSURFACESURFACECGL_H_

#include "libANGLE/renderer/gl/SurfaceGL.h"
#include "libANGLE/renderer/gl/cgl/DisplayCGL.h"

struct __IOSurface;
typedef __IOSurface *IOSurfaceRef;

namespace egl
{
class AttributeMap;
}  // namespace egl

namespace rx
{

class DisplayCGL;
class FunctionsGL;
class StateManagerGL;

class IOSurfaceSurfaceCGL : public SurfaceGL
{
  public:
    IOSurfaceSurfaceCGL(const egl::SurfaceState &state,
                        CGLContextObj cglContext,
                        EGLClientBuffer buffer,
                        const egl::AttributeMap &attribs);
    ~IOSurfaceSurfaceCGL() override;

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

    static bool validateAttributes(EGLClientBuffer buffer, const egl::AttributeMap &attribs);
    FramebufferImpl *createDefaultFramebuffer(const gl::Context *context,
                                              const gl::FramebufferState &state) override;

    bool hasEmulatedAlphaChannel() const override;

  private:
    angle::Result initializeAlphaChannel(const gl::Context *context, GLuint texture);

    CGLContextObj mCGLContext;
    IOSurfaceRef mIOSurface;
    int mWidth;
    int mHeight;
    int mPlane;
    int mFormatIndex;

    bool mAlphaInitialized;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_CGL_IOSURFACESURFACECGL_H_
