//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FramebufferGL.h: Defines the class interface for FramebufferGL.

#ifndef LIBANGLE_RENDERER_GL_FRAMEBUFFERGL_H_
#define LIBANGLE_RENDERER_GL_FRAMEBUFFERGL_H_

#include "libANGLE/Context.h"
#include "libANGLE/renderer/FramebufferImpl.h"

namespace rx
{

class BlitGL;
class ClearMultiviewGL;
class FunctionsGL;
class StateManagerGL;

class FramebufferGL : public FramebufferImpl
{
  public:
    FramebufferGL(const gl::FramebufferState &data, GLuint id, bool isDefault, bool emulatedAlpha);
    ~FramebufferGL() override;

    void destroy(const gl::Context *context) override;

    angle::Result discard(const gl::Context *context,
                          size_t count,
                          const GLenum *attachments) override;
    angle::Result invalidate(const gl::Context *context,
                             size_t count,
                             const GLenum *attachments) override;
    angle::Result invalidateSub(const gl::Context *context,
                                size_t count,
                                const GLenum *attachments,
                                const gl::Rectangle &area) override;

    angle::Result clear(const gl::Context *context, GLbitfield mask) override;
    angle::Result clearBufferfv(const gl::Context *context,
                                GLenum buffer,
                                GLint drawbuffer,
                                const GLfloat *values) override;
    angle::Result clearBufferuiv(const gl::Context *context,
                                 GLenum buffer,
                                 GLint drawbuffer,
                                 const GLuint *values) override;
    angle::Result clearBufferiv(const gl::Context *context,
                                GLenum buffer,
                                GLint drawbuffer,
                                const GLint *values) override;
    angle::Result clearBufferfi(const gl::Context *context,
                                GLenum buffer,
                                GLint drawbuffer,
                                GLfloat depth,
                                GLint stencil) override;

    GLenum getImplementationColorReadFormat(const gl::Context *context) const override;
    GLenum getImplementationColorReadType(const gl::Context *context) const override;

    angle::Result readPixels(const gl::Context *context,
                             const gl::Rectangle &area,
                             GLenum format,
                             GLenum type,
                             void *pixels) override;

    angle::Result blit(const gl::Context *context,
                       const gl::Rectangle &sourceArea,
                       const gl::Rectangle &destArea,
                       GLbitfield mask,
                       GLenum filter) override;

    angle::Result getSamplePosition(const gl::Context *context,
                                    size_t index,
                                    GLfloat *xy) const override;

    // The GL back-end requires a full sync state before we call checkStatus.
    bool shouldSyncStateBeforeCheckStatus() const override;

    bool checkStatus(const gl::Context *context) const override;

    angle::Result syncState(const gl::Context *context,
                            const gl::Framebuffer::DirtyBits &dirtyBits) override;

    GLuint getFramebufferID() const;
    bool isDefault() const;

    bool hasEmulatedAlphaChannelTextureAttachment() const;

  private:
    void syncClearState(const gl::Context *context, GLbitfield mask);
    void syncClearBufferState(const gl::Context *context, GLenum buffer, GLint drawBuffer);

    bool modifyInvalidateAttachmentsForEmulatedDefaultFBO(
        size_t count,
        const GLenum *attachments,
        std::vector<GLenum> *modifiedAttachments) const;

    angle::Result readPixelsRowByRow(const gl::Context *context,
                                     const gl::Rectangle &area,
                                     GLenum format,
                                     GLenum type,
                                     const gl::PixelPackState &pack,
                                     GLubyte *pixels) const;

    angle::Result readPixelsAllAtOnce(const gl::Context *context,
                                      const gl::Rectangle &area,
                                      GLenum format,
                                      GLenum type,
                                      const gl::PixelPackState &pack,
                                      GLubyte *pixels,
                                      bool readLastRowSeparately) const;

    void maskOutInactiveOutputDrawBuffersImpl(const gl::Context *context,
                                              gl::DrawBufferMask targetAppliedDrawBuffers);

    angle::Result adjustSrcDstRegion(const gl::Context *context,
                                     const gl::Rectangle &sourceArea,
                                     const gl::Rectangle &destArea,
                                     gl::Rectangle *newSourceArea,
                                     gl::Rectangle *newDestArea);

    angle::Result clipSrcRegion(const gl::Context *context,
                                const gl::Rectangle &sourceArea,
                                const gl::Rectangle &destArea,
                                gl::Rectangle *newSourceArea,
                                gl::Rectangle *newDestArea);

    GLuint mFramebufferID;
    bool mIsDefault;

    bool mHasEmulatedAlphaAttachment;

    gl::DrawBufferMask mAppliedEnabledDrawBuffers;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_FRAMEBUFFERGL_H_
