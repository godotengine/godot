//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Framebuffer.h: Defines the gl::Framebuffer class. Implements GL framebuffer
// objects and related functionality. [OpenGL ES 2.0.24] section 4.4 page 105.

#ifndef LIBANGLE_FRAMEBUFFER_H_
#define LIBANGLE_FRAMEBUFFER_H_

#include <vector>

#include "common/FixedVector.h"
#include "common/Optional.h"
#include "common/angleutils.h"
#include "libANGLE/Constants.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/FramebufferAttachment.h"
#include "libANGLE/Observer.h"
#include "libANGLE/RefCountObject.h"

namespace rx
{
class GLImplFactory;
class FramebufferImpl;
class RenderbufferImpl;
class SurfaceImpl;
}  // namespace rx

namespace egl
{
class Display;
class Surface;
}  // namespace egl

namespace gl
{
struct Caps;
class Context;
struct Extensions;
class Framebuffer;
class ImageIndex;
struct Rectangle;
class Renderbuffer;
class State;
class Texture;
class TextureCapsMap;

enum class AttachmentSampleType
{
    // The sample count of the actual resource
    Resource,
    // If render_to_texture is used, this is the sample count of the multisampled
    // texture that is created behind the scenes.
    Emulated
};

class FramebufferState final : angle::NonCopyable
{
  public:
    FramebufferState();
    explicit FramebufferState(const Caps &caps, FramebufferID id);
    ~FramebufferState();

    const std::string &getLabel();
    size_t getReadIndex() const;

    const FramebufferAttachment *getAttachment(const Context *context, GLenum attachment) const;
    const FramebufferAttachment *getReadAttachment() const;
    const FramebufferAttachment *getFirstNonNullAttachment() const;
    const FramebufferAttachment *getFirstColorAttachment() const;
    const FramebufferAttachment *getDepthOrStencilAttachment() const;
    const FramebufferAttachment *getStencilOrDepthStencilAttachment() const;
    const FramebufferAttachment *getColorAttachment(size_t colorAttachment) const;
    const FramebufferAttachment *getDepthAttachment() const;
    const FramebufferAttachment *getStencilAttachment() const;
    const FramebufferAttachment *getDepthStencilAttachment() const;

    const std::vector<GLenum> &getDrawBufferStates() const { return mDrawBufferStates; }
    DrawBufferMask getEnabledDrawBuffers() const { return mEnabledDrawBuffers; }
    GLenum getReadBufferState() const { return mReadBufferState; }
    const std::vector<FramebufferAttachment> &getColorAttachments() const
    {
        return mColorAttachments;
    }

    bool attachmentsHaveSameDimensions() const;
    bool hasSeparateDepthAndStencilAttachments() const;
    bool colorAttachmentsAreUniqueImages() const;
    Box getDimensions() const;
    Extents getExtents() const;

    const FramebufferAttachment *getDrawBuffer(size_t drawBufferIdx) const;
    size_t getDrawBufferCount() const;

    GLint getDefaultWidth() const { return mDefaultWidth; }
    GLint getDefaultHeight() const { return mDefaultHeight; }
    GLint getDefaultSamples() const { return mDefaultSamples; }
    bool getDefaultFixedSampleLocations() const { return mDefaultFixedSampleLocations; }
    GLint getDefaultLayers() const { return mDefaultLayers; }

    bool hasDepth() const;
    bool hasStencil() const;

    bool isMultiview() const;

    ANGLE_INLINE GLsizei getNumViews() const
    {
        const FramebufferAttachment *attachment = getFirstNonNullAttachment();
        if (attachment == nullptr)
        {
            return FramebufferAttachment::kDefaultNumViews;
        }
        return attachment->getNumViews();
    }

    GLint getBaseViewIndex() const;

    FramebufferID id() const { return mId; }

    bool isDefault() const;

  private:
    const FramebufferAttachment *getWebGLDepthStencilAttachment() const;
    const FramebufferAttachment *getWebGLDepthAttachment() const;
    const FramebufferAttachment *getWebGLStencilAttachment() const;

    friend class Framebuffer;

    FramebufferID mId;
    std::string mLabel;

    std::vector<FramebufferAttachment> mColorAttachments;
    FramebufferAttachment mDepthAttachment;
    FramebufferAttachment mStencilAttachment;

    std::vector<GLenum> mDrawBufferStates;
    GLenum mReadBufferState;
    DrawBufferMask mEnabledDrawBuffers;
    ComponentTypeMask mDrawBufferTypeMask;

    GLint mDefaultWidth;
    GLint mDefaultHeight;
    GLint mDefaultSamples;
    bool mDefaultFixedSampleLocations;
    GLint mDefaultLayers;

    // It's necessary to store all this extra state so we can restore attachments
    // when DEPTH_STENCIL/DEPTH/STENCIL is unbound in WebGL 1.
    FramebufferAttachment mWebGLDepthStencilAttachment;
    FramebufferAttachment mWebGLDepthAttachment;
    FramebufferAttachment mWebGLStencilAttachment;
    bool mWebGLDepthStencilConsistent;

    // Tracks if we need to initialize the resources for each attachment.
    angle::BitSet<IMPLEMENTATION_MAX_FRAMEBUFFER_ATTACHMENTS + 2> mResourceNeedsInit;

    bool mDefaultFramebufferReadAttachmentInitialized;
    FramebufferAttachment mDefaultFramebufferReadAttachment;
};

class Framebuffer final : public angle::ObserverInterface,
                          public LabeledObject,
                          public angle::Subject
{
  public:
    // Constructor to build application-defined framebuffers
    Framebuffer(const Caps &caps, rx::GLImplFactory *factory, FramebufferID id);
    // Constructor to build default framebuffers for a surface and context pair
    Framebuffer(const Context *context, egl::Surface *surface, egl::Surface *readSurface);
    // Constructor to build a fake default framebuffer when surfaceless
    Framebuffer(const Context *context, rx::GLImplFactory *factory, egl::Surface *readSurface);

    ~Framebuffer() override;
    void onDestroy(const Context *context);

    void setReadSurface(const Context *context, egl::Surface *readSurface);
    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    rx::FramebufferImpl *getImplementation() const { return mImpl; }

    FramebufferID id() const { return mState.mId; }

    void setAttachment(const Context *context,
                       GLenum type,
                       GLenum binding,
                       const ImageIndex &textureIndex,
                       FramebufferAttachmentObject *resource);
    void setAttachmentMultisample(const Context *context,
                                  GLenum type,
                                  GLenum binding,
                                  const ImageIndex &textureIndex,
                                  FramebufferAttachmentObject *resource,
                                  GLsizei samples);
    void setAttachmentMultiview(const Context *context,
                                GLenum type,
                                GLenum binding,
                                const ImageIndex &textureIndex,
                                FramebufferAttachmentObject *resource,
                                GLsizei numViews,
                                GLint baseViewIndex);
    void resetAttachment(const Context *context, GLenum binding);

    bool detachTexture(const Context *context, TextureID texture);
    bool detachRenderbuffer(const Context *context, RenderbufferID renderbuffer);

    const FramebufferAttachment *getColorAttachment(size_t colorAttachment) const;
    const FramebufferAttachment *getDepthAttachment() const;
    const FramebufferAttachment *getStencilAttachment() const;
    const FramebufferAttachment *getDepthStencilAttachment() const;
    const FramebufferAttachment *getDepthOrStencilAttachment() const;
    const FramebufferAttachment *getStencilOrDepthStencilAttachment() const;
    const FramebufferAttachment *getReadColorAttachment() const;
    GLenum getReadColorAttachmentType() const;
    const FramebufferAttachment *getFirstColorAttachment() const;
    const FramebufferAttachment *getFirstNonNullAttachment() const;

    const FramebufferAttachment *getAttachment(const Context *context, GLenum attachment) const;
    bool isMultiview() const;
    bool readDisallowedByMultiview() const;
    GLsizei getNumViews() const;
    GLint getBaseViewIndex() const;
    Extents getExtents() const;

    size_t getDrawbufferStateCount() const;
    GLenum getDrawBufferState(size_t drawBuffer) const;
    const std::vector<GLenum> &getDrawBufferStates() const;
    void setDrawBuffers(size_t count, const GLenum *buffers);
    const FramebufferAttachment *getDrawBuffer(size_t drawBuffer) const;
    ComponentType getDrawbufferWriteType(size_t drawBuffer) const;
    ComponentTypeMask getDrawBufferTypeMask() const;
    DrawBufferMask getDrawBufferMask() const;
    bool hasEnabledDrawBuffer() const;

    GLenum getReadBufferState() const;
    void setReadBuffer(GLenum buffer);

    size_t getNumColorAttachments() const;
    bool hasDepth() const;
    bool hasStencil() const;

    bool usingExtendedDrawBuffers() const;

    // This method calls checkStatus.
    int getSamples(const Context *context);
    int getResourceSamples(const Context *context);

    angle::Result getSamplePosition(const Context *context, size_t index, GLfloat *xy) const;

    GLint getDefaultWidth() const;
    GLint getDefaultHeight() const;
    GLint getDefaultSamples() const;
    bool getDefaultFixedSampleLocations() const;
    GLint getDefaultLayers() const;
    void setDefaultWidth(const Context *context, GLint defaultWidth);
    void setDefaultHeight(const Context *context, GLint defaultHeight);
    void setDefaultSamples(const Context *context, GLint defaultSamples);
    void setDefaultFixedSampleLocations(const Context *context, bool defaultFixedSampleLocations);
    void setDefaultLayers(GLint defaultLayers);

    void invalidateCompletenessCache();
    ANGLE_INLINE bool cachedStatusValid() { return mCachedStatus.valid(); }

    ANGLE_INLINE GLenum checkStatus(const Context *context)
    {
        // The default framebuffer is always complete except when it is surfaceless in which
        // case it is always unsupported.
        ASSERT(!isDefault() || mCachedStatus.valid());
        if (isDefault() || (!hasAnyDirtyBit() && mCachedStatus.valid()))
        {
            return mCachedStatus.value();
        }

        return checkStatusImpl(context);
    }

    // For when we don't want to check completeness in getSamples().
    int getCachedSamples(const Context *context, AttachmentSampleType sampleType) const;

    // Helper for checkStatus == GL_FRAMEBUFFER_COMPLETE.
    ANGLE_INLINE bool isComplete(const Context *context)
    {
        return (checkStatus(context) == GL_FRAMEBUFFER_COMPLETE);
    }

    bool hasValidDepthStencil() const;

    angle::Result discard(const Context *context, size_t count, const GLenum *attachments);
    angle::Result invalidate(const Context *context, size_t count, const GLenum *attachments);
    angle::Result invalidateSub(const Context *context,
                                size_t count,
                                const GLenum *attachments,
                                const Rectangle &area);

    angle::Result clear(const Context *context, GLbitfield mask);
    angle::Result clearBufferfv(const Context *context,
                                GLenum buffer,
                                GLint drawbuffer,
                                const GLfloat *values);
    angle::Result clearBufferuiv(const Context *context,
                                 GLenum buffer,
                                 GLint drawbuffer,
                                 const GLuint *values);
    angle::Result clearBufferiv(const Context *context,
                                GLenum buffer,
                                GLint drawbuffer,
                                const GLint *values);
    angle::Result clearBufferfi(const Context *context,
                                GLenum buffer,
                                GLint drawbuffer,
                                GLfloat depth,
                                GLint stencil);

    // These two methods call syncState() internally.
    angle::Result getImplementationColorReadFormat(const Context *context, GLenum *formatOut);
    angle::Result getImplementationColorReadType(const Context *context, GLenum *typeOut);

    angle::Result readPixels(const Context *context,
                             const Rectangle &area,
                             GLenum format,
                             GLenum type,
                             void *pixels);

    angle::Result blit(const Context *context,
                       const Rectangle &sourceArea,
                       const Rectangle &destArea,
                       GLbitfield mask,
                       GLenum filter);
    bool isDefault() const { return mState.isDefault(); }

    enum DirtyBitType : size_t
    {
        DIRTY_BIT_COLOR_ATTACHMENT_0,
        DIRTY_BIT_COLOR_ATTACHMENT_MAX =
            DIRTY_BIT_COLOR_ATTACHMENT_0 + IMPLEMENTATION_MAX_FRAMEBUFFER_ATTACHMENTS,
        DIRTY_BIT_DEPTH_ATTACHMENT = DIRTY_BIT_COLOR_ATTACHMENT_MAX,
        DIRTY_BIT_STENCIL_ATTACHMENT,
        DIRTY_BIT_COLOR_BUFFER_CONTENTS_0,
        DIRTY_BIT_COLOR_BUFFER_CONTENTS_MAX =
            DIRTY_BIT_COLOR_BUFFER_CONTENTS_0 + IMPLEMENTATION_MAX_FRAMEBUFFER_ATTACHMENTS,
        DIRTY_BIT_DEPTH_BUFFER_CONTENTS = DIRTY_BIT_COLOR_BUFFER_CONTENTS_MAX,
        DIRTY_BIT_STENCIL_BUFFER_CONTENTS,
        DIRTY_BIT_DRAW_BUFFERS,
        DIRTY_BIT_READ_BUFFER,
        DIRTY_BIT_DEFAULT_WIDTH,
        DIRTY_BIT_DEFAULT_HEIGHT,
        DIRTY_BIT_DEFAULT_SAMPLES,
        DIRTY_BIT_DEFAULT_FIXED_SAMPLE_LOCATIONS,
        DIRTY_BIT_DEFAULT_LAYERS,
        DIRTY_BIT_UNKNOWN,
        DIRTY_BIT_MAX = DIRTY_BIT_UNKNOWN
    };

    using DirtyBits = angle::BitSet<DIRTY_BIT_MAX>;
    bool hasAnyDirtyBit() const { return mDirtyBits.any(); }

    bool hasActiveFloat32ColorAttachment() const
    {
        return (mFloat32ColorAttachmentBits & getDrawBufferMask()).any();
    }

    bool hasResourceThatNeedsInit() const { return mState.mResourceNeedsInit.any(); }

    angle::Result syncState(const Context *context);

    // Observer implementation
    void onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message) override;

    bool formsRenderingFeedbackLoopWith(const Context *context) const;
    bool formsCopyingFeedbackLoopWith(TextureID copyTextureID,
                                      GLint copyTextureLevel,
                                      GLint copyTextureLayer) const;

    angle::Result ensureClearAttachmentsInitialized(const Context *context, GLbitfield mask);
    angle::Result ensureClearBufferAttachmentsInitialized(const Context *context,
                                                          GLenum buffer,
                                                          GLint drawbuffer);
    angle::Result ensureDrawAttachmentsInitialized(const Context *context);

    // Conservatively initializes both read color and depth. Blit can access the depth buffer.
    angle::Result ensureReadAttachmentsInitialized(const Context *context);
    Box getDimensions() const;

    static const FramebufferID kDefaultDrawFramebufferHandle;

  private:
    bool detachResourceById(const Context *context, GLenum resourceType, GLuint resourceId);
    bool detachMatchingAttachment(const Context *context,
                                  FramebufferAttachment *attachment,
                                  GLenum matchType,
                                  GLuint matchId);
    GLenum checkStatusWithGLFrontEnd(const Context *context);
    GLenum checkStatusImpl(const Context *context);
    void setAttachment(const Context *context,
                       GLenum type,
                       GLenum binding,
                       const ImageIndex &textureIndex,
                       FramebufferAttachmentObject *resource,
                       GLsizei numViews,
                       GLuint baseViewIndex,
                       bool isMultiview,
                       GLsizei samples);
    void commitWebGL1DepthStencilIfConsistent(const Context *context,
                                              GLsizei numViews,
                                              GLuint baseViewIndex,
                                              bool isMultiview,
                                              GLsizei samples);
    void setAttachmentImpl(const Context *context,
                           GLenum type,
                           GLenum binding,
                           const ImageIndex &textureIndex,
                           FramebufferAttachmentObject *resource,
                           GLsizei numViews,
                           GLuint baseViewIndex,
                           bool isMultiview,
                           GLsizei samples);
    void updateAttachment(const Context *context,
                          FramebufferAttachment *attachment,
                          size_t dirtyBit,
                          angle::ObserverBinding *onDirtyBinding,
                          GLenum type,
                          GLenum binding,
                          const ImageIndex &textureIndex,
                          FramebufferAttachmentObject *resource,
                          GLsizei numViews,
                          GLuint baseViewIndex,
                          bool isMultiview,
                          GLsizei samples);

    void markDrawAttachmentsInitialized(bool color, bool depth, bool stencil);
    void markBufferInitialized(GLenum bufferType, GLint bufferIndex);
    angle::Result ensureBufferInitialized(const Context *context,
                                          GLenum bufferType,
                                          GLint bufferIndex);

    // Checks that we have a partially masked clear:
    // * some color channels are masked out
    // * some stencil values are masked out
    // * scissor test partially overlaps the framebuffer
    bool partialClearNeedsInit(const Context *context, bool color, bool depth, bool stencil);
    bool partialBufferClearNeedsInit(const Context *context, GLenum bufferType);

    FramebufferAttachment *getAttachmentFromSubjectIndex(angle::SubjectIndex index);

    ANGLE_INLINE void updateFloat32ColorAttachmentBits(size_t index,
                                                       const gl::InternalFormat *format)
    {
        mFloat32ColorAttachmentBits.set(index, format->type == GL_FLOAT);
    }

    FramebufferState mState;
    rx::FramebufferImpl *mImpl;

    Optional<GLenum> mCachedStatus;
    std::vector<angle::ObserverBinding> mDirtyColorAttachmentBindings;
    angle::ObserverBinding mDirtyDepthAttachmentBinding;
    angle::ObserverBinding mDirtyStencilAttachmentBinding;

    DirtyBits mDirtyBits;
    DrawBufferMask mFloat32ColorAttachmentBits;
    DrawBufferMask mColorAttachmentBits;

    // The dirty bits guard is checked when we get a dependent state change message. We verify that
    // we don't set a dirty bit that isn't already set, when inside the dirty bits syncState.
    Optional<DirtyBits> mDirtyBitsGuard;
};

}  // namespace gl

#endif  // LIBANGLE_FRAMEBUFFER_H_
