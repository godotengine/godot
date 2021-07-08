//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FramebufferAttachmentObjectImpl.h:
//   Common ancenstor for all implementations of FBO attachable-objects.
//   This means Surfaces, Textures and Renderbuffers.
//

#ifndef LIBANGLE_RENDERER_FRAMEBUFFER_ATTACHMENT_OBJECT_IMPL_H_
#define LIBANGLE_RENDERER_FRAMEBUFFER_ATTACHMENT_OBJECT_IMPL_H_

#include "libANGLE/ImageIndex.h"
#include "libANGLE/Observer.h"

namespace gl
{
class Context;
}  // namespace gl

namespace rx
{
class FramebufferAttachmentRenderTarget;

class FramebufferAttachmentObjectImpl : angle::NonCopyable
{
  public:
    FramebufferAttachmentObjectImpl() {}
    virtual ~FramebufferAttachmentObjectImpl() {}

    virtual angle::Result getAttachmentRenderTarget(const gl::Context *context,
                                                    GLenum binding,
                                                    const gl::ImageIndex &imageIndex,
                                                    GLsizei samples,
                                                    FramebufferAttachmentRenderTarget **rtOut);

    virtual angle::Result initializeContents(const gl::Context *context,
                                             const gl::ImageIndex &imageIndex);
};

inline angle::Result FramebufferAttachmentObjectImpl::getAttachmentRenderTarget(
    const gl::Context *context,
    GLenum binding,
    const gl::ImageIndex &imageIndex,
    GLsizei samples,
    FramebufferAttachmentRenderTarget **rtOut)
{
    UNIMPLEMENTED();
    return angle::Result::Stop;
}

inline angle::Result FramebufferAttachmentObjectImpl::initializeContents(
    const gl::Context *context,
    const gl::ImageIndex &imageIndex)
{
    UNIMPLEMENTED();
    return angle::Result::Stop;
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_FRAMEBUFFER_ATTACHMENT_OBJECT_IMPL_H_
