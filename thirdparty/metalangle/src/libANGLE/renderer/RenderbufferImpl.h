//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RenderbufferImpl.h: Defines the abstract class gl::RenderbufferImpl

#ifndef LIBANGLE_RENDERER_RENDERBUFFERIMPL_H_
#define LIBANGLE_RENDERER_RENDERBUFFERIMPL_H_

#include "angle_gl.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/renderer/FramebufferAttachmentObjectImpl.h"

namespace gl
{
class RenderbufferState;
}  // namespace gl

namespace egl
{
class Image;
}  // namespace egl

namespace rx
{

class RenderbufferImpl : public FramebufferAttachmentObjectImpl
{
  public:
    RenderbufferImpl(const gl::RenderbufferState &state) : mState(state) {}
    ~RenderbufferImpl() override {}
    virtual void onDestroy(const gl::Context *context) {}

    virtual angle::Result setStorage(const gl::Context *context,
                                     GLenum internalformat,
                                     size_t width,
                                     size_t height)                   = 0;
    virtual angle::Result setStorageMultisample(const gl::Context *context,
                                                size_t samples,
                                                GLenum internalformat,
                                                size_t width,
                                                size_t height)        = 0;
    virtual angle::Result setStorageEGLImageTarget(const gl::Context *context,
                                                   egl::Image *image) = 0;

    // Override if accurate native memory size information is available
    virtual GLint getMemorySize() const;

  protected:
    const gl::RenderbufferState &mState;
};

inline GLint RenderbufferImpl::getMemorySize() const
{
    return 0;
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_RENDERBUFFERIMPL_H_
