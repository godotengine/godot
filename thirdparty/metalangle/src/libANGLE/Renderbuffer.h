//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Renderbuffer.h: Defines the renderer-agnostic container class gl::Renderbuffer.
// Implements GL renderbuffer objects and related functionality.
// [OpenGL ES 2.0.24] section 4.4.3 page 108.

#ifndef LIBANGLE_RENDERBUFFER_H_
#define LIBANGLE_RENDERBUFFER_H_

#include "angle_gl.h"
#include "common/angleutils.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/FramebufferAttachment.h"
#include "libANGLE/Image.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/RenderbufferImpl.h"

namespace rx
{
class GLImplFactory;
}  // namespace rx

namespace gl
{
// A GL renderbuffer object is usually used as a depth or stencil buffer attachment
// for a framebuffer object. The renderbuffer itself is a distinct GL object, see
// FramebufferAttachment and Framebuffer for how they are applied to an FBO via an
// attachment point.

class RenderbufferState final : angle::NonCopyable
{
  public:
    RenderbufferState();
    ~RenderbufferState();

    GLsizei getWidth() const;
    GLsizei getHeight() const;
    const Format &getFormat() const;
    GLsizei getSamples() const;

  private:
    friend class Renderbuffer;

    void update(GLsizei width,
                GLsizei height,
                const Format &format,
                GLsizei samples,
                InitState initState);

    GLsizei mWidth;
    GLsizei mHeight;
    Format mFormat;
    GLsizei mSamples;

    // For robust resource init.
    InitState mInitState;
};

class Renderbuffer final : public RefCountObject<RenderbufferID>,
                           public egl::ImageSibling,
                           public LabeledObject
{
  public:
    Renderbuffer(rx::GLImplFactory *implFactory, RenderbufferID id);
    ~Renderbuffer() override;

    void onDestroy(const Context *context) override;

    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    angle::Result setStorage(const Context *context,
                             GLenum internalformat,
                             size_t width,
                             size_t height);
    angle::Result setStorageMultisample(const Context *context,
                                        size_t samples,
                                        GLenum internalformat,
                                        size_t width,
                                        size_t height);
    angle::Result setStorageEGLImageTarget(const Context *context, egl::Image *imageTarget);

    rx::RenderbufferImpl *getImplementation() const;

    GLsizei getWidth() const;
    GLsizei getHeight() const;
    const Format &getFormat() const;
    GLsizei getSamples() const;
    GLuint getRedSize() const;
    GLuint getGreenSize() const;
    GLuint getBlueSize() const;
    GLuint getAlphaSize() const;
    GLuint getDepthSize() const;
    GLuint getStencilSize() const;

    GLint getMemorySize() const;

    // FramebufferAttachmentObject Impl
    Extents getAttachmentSize(const ImageIndex &imageIndex) const override;
    Format getAttachmentFormat(GLenum binding, const ImageIndex &imageIndex) const override;
    GLsizei getAttachmentSamples(const ImageIndex &imageIndex) const override;
    bool isRenderable(const Context *context,
                      GLenum binding,
                      const ImageIndex &imageIndex) const override;

    void onAttach(const Context *context) override;
    void onDetach(const Context *context) override;
    GLuint getId() const override;

    InitState initState(const ImageIndex &imageIndex) const override;
    void setInitState(const ImageIndex &imageIndex, InitState initState) override;

  private:
    rx::FramebufferAttachmentObjectImpl *getAttachmentImpl() const override;

    RenderbufferState mState;
    std::unique_ptr<rx::RenderbufferImpl> mImplementation;

    std::string mLabel;
};

}  // namespace gl

#endif  // LIBANGLE_RENDERBUFFER_H_
