//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SurfaceGL.h: Defines the class interface for SurfaceGL.

#ifndef LIBANGLE_RENDERER_GL_SURFACEGL_H_
#define LIBANGLE_RENDERER_GL_SURFACEGL_H_

#include "libANGLE/renderer/SurfaceImpl.h"

namespace rx
{

class SurfaceGL : public SurfaceImpl
{
  public:
    SurfaceGL(const egl::SurfaceState &state);
    ~SurfaceGL() override;

    FramebufferImpl *createDefaultFramebuffer(const gl::Context *context,
                                              const gl::FramebufferState &data) override;
    egl::Error getSyncValues(EGLuint64KHR *ust, EGLuint64KHR *msc, EGLuint64KHR *sbc) override;

    angle::Result initializeContents(const gl::Context *context,
                                     const gl::ImageIndex &imageIndex) override;

    virtual bool hasEmulatedAlphaChannel() const;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_SURFACEGL_H_
