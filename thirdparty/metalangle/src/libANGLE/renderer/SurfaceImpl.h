//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SurfaceImpl.h: Implementation methods of egl::Surface

#ifndef LIBANGLE_RENDERER_SURFACEIMPL_H_
#define LIBANGLE_RENDERER_SURFACEIMPL_H_

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/FramebufferAttachment.h"
#include "libANGLE/renderer/FramebufferAttachmentObjectImpl.h"

namespace angle
{
struct Format;
}

namespace gl
{
class Context;
class FramebufferState;
}  // namespace gl

namespace egl
{
class Display;
struct Config;
struct SurfaceState;
class Thread;

using SupportedTimestamps        = angle::PackedEnumBitSet<Timestamp>;
using SupportedCompositorTimings = angle::PackedEnumBitSet<CompositorTiming>;
}  // namespace egl

namespace rx
{
class FramebufferImpl;

class SurfaceImpl : public FramebufferAttachmentObjectImpl
{
  public:
    SurfaceImpl(const egl::SurfaceState &surfaceState);
    ~SurfaceImpl() override;
    virtual void destroy(const egl::Display *display) {}

    virtual egl::Error initialize(const egl::Display *display)                           = 0;
    virtual FramebufferImpl *createDefaultFramebuffer(const gl::Context *context,
                                                      const gl::FramebufferState &state) = 0;
    virtual egl::Error makeCurrent(const gl::Context *context);
    virtual egl::Error unMakeCurrent(const gl::Context *context);
    virtual egl::Error swap(const gl::Context *context) = 0;
    virtual egl::Error swapWithDamage(const gl::Context *context, EGLint *rects, EGLint n_rects);
    virtual egl::Error postSubBuffer(const gl::Context *context,
                                     EGLint x,
                                     EGLint y,
                                     EGLint width,
                                     EGLint height) = 0;
    virtual egl::Error setPresentationTime(EGLnsecsANDROID time);
    virtual egl::Error querySurfacePointerANGLE(EGLint attribute, void **value)               = 0;
    virtual egl::Error bindTexImage(const gl::Context *context,
                                    gl::Texture *texture,
                                    EGLint buffer)                                            = 0;
    virtual egl::Error releaseTexImage(const gl::Context *context, EGLint buffer)             = 0;
    virtual egl::Error getSyncValues(EGLuint64KHR *ust, EGLuint64KHR *msc, EGLuint64KHR *sbc) = 0;
    virtual void setSwapInterval(EGLint interval)                                             = 0;
    virtual void setSwapBehavior(EGLint behavior);
    virtual void setFixedWidth(EGLint width);
    virtual void setFixedHeight(EGLint height);

    // width and height can change with client window resizing
    virtual EGLint getWidth() const  = 0;
    virtual EGLint getHeight() const = 0;

    virtual EGLint isPostSubBufferSupported() const = 0;
    virtual EGLint getSwapBehavior() const          = 0;

    // Used to query color format from pbuffers created from D3D textures.
    virtual const angle::Format *getD3DTextureColorFormat() const;

    // EGL_ANDROID_get_frame_timestamps
    virtual void setTimestampsEnabled(bool enabled);
    virtual egl::SupportedCompositorTimings getSupportedCompositorTimings() const;
    virtual egl::Error getCompositorTiming(EGLint numTimestamps,
                                           const EGLint *names,
                                           EGLnsecsANDROID *values) const;
    virtual egl::Error getNextFrameId(EGLuint64KHR *frameId) const;
    virtual egl::SupportedTimestamps getSupportedTimestamps() const;
    virtual egl::Error getFrameTimestamps(EGLuint64KHR frameId,
                                          EGLint numTimestamps,
                                          const EGLint *timestamps,
                                          EGLnsecsANDROID *values) const;

  protected:
    const egl::SurfaceState &mState;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_SURFACEIMPL_H_
