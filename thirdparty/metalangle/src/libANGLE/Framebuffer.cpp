//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Framebuffer.cpp: Implements the gl::Framebuffer class. Implements GL framebuffer
// objects and related functionality. [OpenGL ES 2.0.24] section 4.4 page 105.

#include "libANGLE/Framebuffer.h"

#include "common/Optional.h"
#include "common/bitset_utils.h"
#include "common/utilities.h"
#include "libANGLE/Config.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/FramebufferAttachment.h"
#include "libANGLE/Renderbuffer.h"
#include "libANGLE/Surface.h"
#include "libANGLE/Texture.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/ContextImpl.h"
#include "libANGLE/renderer/FramebufferImpl.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/RenderbufferImpl.h"
#include "libANGLE/renderer/SurfaceImpl.h"

using namespace angle;

namespace gl
{

namespace
{

bool CheckMultiviewStateMatchesForCompleteness(const FramebufferAttachment *firstAttachment,
                                               const FramebufferAttachment *secondAttachment)
{
    ASSERT(firstAttachment && secondAttachment);
    ASSERT(firstAttachment->isAttached() && secondAttachment->isAttached());

    if (firstAttachment->getNumViews() != secondAttachment->getNumViews())
    {
        return false;
    }
    if (firstAttachment->getBaseViewIndex() != secondAttachment->getBaseViewIndex())
    {
        return false;
    }
    if (firstAttachment->isMultiview() != secondAttachment->isMultiview())
    {
        return false;
    }
    return true;
}

bool CheckAttachmentCompleteness(const Context *context, const FramebufferAttachment &attachment)
{
    ASSERT(attachment.isAttached());

    const Extents &size = attachment.getSize();
    if (size.width == 0 || size.height == 0)
    {
        return false;
    }

    if (!attachment.isRenderable(context))
    {
        return false;
    }

    if (attachment.type() == GL_TEXTURE)
    {
        // [EXT_geometry_shader] Section 9.4.1, "Framebuffer Completeness"
        // If <image> is a three-dimensional texture or a two-dimensional array texture and the
        // attachment is not layered, the selected layer is less than the depth or layer count,
        // respectively, of the texture.
        if (!attachment.isLayered())
        {
            if (attachment.layer() >= size.depth)
            {
                return false;
            }
        }
        // If <image> is a three-dimensional texture or a two-dimensional array texture and the
        // attachment is layered, the depth or layer count, respectively, of the texture is less
        // than or equal to the value of MAX_FRAMEBUFFER_LAYERS_EXT.
        else
        {
            if (static_cast<GLuint>(size.depth) >= context->getCaps().maxFramebufferLayers)
            {
                return false;
            }
        }

        // ES3 specifies that cube map texture attachments must be cube complete.
        // This language is missing from the ES2 spec, but we enforce it here because some
        // desktop OpenGL drivers also enforce this validation.
        // TODO(jmadill): Check if OpenGL ES2 drivers enforce cube completeness.
        const Texture *texture = attachment.getTexture();
        ASSERT(texture);
        if (texture->getType() == TextureType::CubeMap &&
            !texture->getTextureState().isCubeComplete())
        {
            return false;
        }

        if (!texture->getImmutableFormat())
        {
            GLuint attachmentMipLevel = static_cast<GLuint>(attachment.mipLevel());

            // From the ES 3.0 spec, pg 213:
            // If the value of FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is TEXTURE and the value of
            // FRAMEBUFFER_ATTACHMENT_OBJECT_NAME does not name an immutable-format texture,
            // then the value of FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL must be in the
            // range[levelbase, q], where levelbase is the value of TEXTURE_BASE_LEVEL and q is
            // the effective maximum texture level defined in the Mipmapping discussion of
            // section 3.8.10.4.
            if (attachmentMipLevel < texture->getBaseLevel() ||
                attachmentMipLevel > texture->getMipmapMaxLevel())
            {
                return false;
            }

            // Form the ES 3.0 spec, pg 213/214:
            // If the value of FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is TEXTURE and the value of
            // FRAMEBUFFER_ATTACHMENT_OBJECT_NAME does not name an immutable-format texture and
            // the value of FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL is not levelbase, then the
            // texture must be mipmap complete, and if FRAMEBUFFER_ATTACHMENT_OBJECT_NAME names
            // a cubemap texture, the texture must also be cube complete.
            if (attachmentMipLevel != texture->getBaseLevel() && !texture->isMipmapComplete())
            {
                return false;
            }
        }
    }

    return true;
}

bool CheckAttachmentSampleCounts(const Context *context,
                                 GLsizei currAttachmentSamples,
                                 GLsizei samples,
                                 bool colorAttachment)
{
    if (currAttachmentSamples != samples)
    {
        if (colorAttachment)
        {
            // APPLE_framebuffer_multisample, which EXT_draw_buffers refers to, requires that
            // all color attachments have the same number of samples for the FBO to be complete.
            return false;
        }
        else
        {
            // CHROMIUM_framebuffer_mixed_samples allows a framebuffer to be considered complete
            // when its depth or stencil samples are a multiple of the number of color samples.
            if (!context->getExtensions().framebufferMixedSamples)
            {
                return false;
            }

            if ((currAttachmentSamples % std::max(samples, 1)) != 0)
            {
                return false;
            }
        }
    }
    return true;
}

bool CheckAttachmentSampleCompleteness(const Context *context,
                                       const FramebufferAttachment &attachment,
                                       bool colorAttachment,
                                       Optional<int> *samples,
                                       Optional<bool> *fixedSampleLocations,
                                       Optional<int> *renderToTextureSamples)
{
    ASSERT(attachment.isAttached());

    if (attachment.type() == GL_TEXTURE)
    {
        const Texture *texture = attachment.getTexture();
        ASSERT(texture);
        GLenum internalFormat         = attachment.getFormat().info->internalFormat;
        const TextureCaps &formatCaps = context->getTextureCaps().get(internalFormat);
        if (static_cast<GLuint>(attachment.getSamples()) > formatCaps.getMaxSamples())
        {
            return false;
        }

        const ImageIndex &attachmentImageIndex = attachment.getTextureImageIndex();
        bool fixedSampleloc = texture->getAttachmentFixedSampleLocations(attachmentImageIndex);
        if (fixedSampleLocations->valid() && fixedSampleloc != fixedSampleLocations->value())
        {
            return false;
        }
        else
        {
            *fixedSampleLocations = fixedSampleloc;
        }
    }

    if (renderToTextureSamples->valid())
    {
        // Only check against RenderToTextureSamples if they actually exist.
        if (renderToTextureSamples->value() !=
            FramebufferAttachment::kDefaultRenderToTextureSamples)
        {
            if (!CheckAttachmentSampleCounts(context, attachment.getRenderToTextureSamples(),
                                             renderToTextureSamples->value(), colorAttachment))
            {
                return false;
            }
        }
    }
    else
    {
        *renderToTextureSamples = attachment.getRenderToTextureSamples();
    }

    if (samples->valid())
    {
        // RenderToTextureSamples takes precedence if they exist.
        if (renderToTextureSamples->value() ==
            FramebufferAttachment::kDefaultRenderToTextureSamples)
        {
            if (!CheckAttachmentSampleCounts(context, attachment.getSamples(), samples->value(),
                                             colorAttachment))
            {
                return false;
            }
        }
    }
    else
    {
        *samples = attachment.getSamples();
    }

    return true;
}

// Needed to index into the attachment arrays/bitsets.
static_assert(static_cast<size_t>(IMPLEMENTATION_MAX_FRAMEBUFFER_ATTACHMENTS) ==
                  Framebuffer::DIRTY_BIT_COLOR_ATTACHMENT_MAX,
              "Framebuffer Dirty bit mismatch");
static_assert(static_cast<size_t>(IMPLEMENTATION_MAX_FRAMEBUFFER_ATTACHMENTS) ==
                  Framebuffer::DIRTY_BIT_DEPTH_ATTACHMENT,
              "Framebuffer Dirty bit mismatch");
static_assert(static_cast<size_t>(IMPLEMENTATION_MAX_FRAMEBUFFER_ATTACHMENTS + 1) ==
                  Framebuffer::DIRTY_BIT_STENCIL_ATTACHMENT,
              "Framebuffer Dirty bit mismatch");

angle::Result InitAttachment(const Context *context, FramebufferAttachment *attachment)
{
    ASSERT(attachment->isAttached());
    if (attachment->initState() == InitState::MayNeedInit)
    {
        ANGLE_TRY(attachment->initializeContents(context));
    }
    return angle::Result::Continue;
}

bool IsColorMaskedOut(const BlendState &blend)
{
    return (!blend.colorMaskRed && !blend.colorMaskGreen && !blend.colorMaskBlue &&
            !blend.colorMaskAlpha);
}

bool IsDepthMaskedOut(const DepthStencilState &depthStencil)
{
    return !depthStencil.depthMask;
}

bool IsStencilMaskedOut(const DepthStencilState &depthStencil)
{
    return ((depthStencil.stencilMask & depthStencil.stencilWritemask) == 0);
}

bool IsClearBufferMaskedOut(const Context *context, GLenum buffer)
{
    switch (buffer)
    {
        case GL_COLOR:
            return IsColorMaskedOut(context->getState().getBlendState());
        case GL_DEPTH:
            return IsDepthMaskedOut(context->getState().getDepthStencilState());
        case GL_STENCIL:
            return IsStencilMaskedOut(context->getState().getDepthStencilState());
        case GL_DEPTH_STENCIL:
            return IsDepthMaskedOut(context->getState().getDepthStencilState()) &&
                   IsStencilMaskedOut(context->getState().getDepthStencilState());
        default:
            UNREACHABLE();
            return true;
    }
}

}  // anonymous namespace

// This constructor is only used for default framebuffers.
FramebufferState::FramebufferState()
    : mId(Framebuffer::kDefaultDrawFramebufferHandle),
      mLabel(),
      mColorAttachments(1),
      mDrawBufferStates(1, GL_BACK),
      mReadBufferState(GL_BACK),
      mDrawBufferTypeMask(),
      mDefaultWidth(0),
      mDefaultHeight(0),
      mDefaultSamples(0),
      mDefaultFixedSampleLocations(GL_FALSE),
      mDefaultLayers(0),
      mWebGLDepthStencilConsistent(true),
      mDefaultFramebufferReadAttachmentInitialized(false)
{
    ASSERT(mDrawBufferStates.size() > 0);
    mEnabledDrawBuffers.set(0);
}

FramebufferState::FramebufferState(const Caps &caps, FramebufferID id)
    : mId(id),
      mLabel(),
      mColorAttachments(caps.maxColorAttachments),
      mDrawBufferStates(caps.maxDrawBuffers, GL_NONE),
      mReadBufferState(GL_COLOR_ATTACHMENT0_EXT),
      mDrawBufferTypeMask(),
      mDefaultWidth(0),
      mDefaultHeight(0),
      mDefaultSamples(0),
      mDefaultFixedSampleLocations(GL_FALSE),
      mDefaultLayers(0),
      mWebGLDepthStencilConsistent(true),
      mDefaultFramebufferReadAttachmentInitialized(false)
{
    ASSERT(mId != Framebuffer::kDefaultDrawFramebufferHandle);
    ASSERT(mDrawBufferStates.size() > 0);
    mDrawBufferStates[0] = GL_COLOR_ATTACHMENT0_EXT;
}

FramebufferState::~FramebufferState() {}

const std::string &FramebufferState::getLabel()
{
    return mLabel;
}

const FramebufferAttachment *FramebufferState::getAttachment(const Context *context,
                                                             GLenum attachment) const
{
    if (attachment >= GL_COLOR_ATTACHMENT0 && attachment <= GL_COLOR_ATTACHMENT15)
    {
        return getColorAttachment(attachment - GL_COLOR_ATTACHMENT0);
    }

    // WebGL1 allows a developer to query for attachment parameters even when "inconsistant" (i.e.
    // multiple conflicting attachment points) and requires us to return the framebuffer attachment
    // associated with WebGL.
    switch (attachment)
    {
        case GL_COLOR:
        case GL_BACK:
            return getColorAttachment(0);
        case GL_DEPTH:
        case GL_DEPTH_ATTACHMENT:
            if (context->isWebGL1())
            {
                return getWebGLDepthAttachment();
            }
            else
            {
                return getDepthAttachment();
            }
        case GL_STENCIL:
        case GL_STENCIL_ATTACHMENT:
            if (context->isWebGL1())
            {
                return getWebGLStencilAttachment();
            }
            else
            {
                return getStencilAttachment();
            }
        case GL_DEPTH_STENCIL:
        case GL_DEPTH_STENCIL_ATTACHMENT:
            if (context->isWebGL1())
            {
                return getWebGLDepthStencilAttachment();
            }
            else
            {
                return getDepthStencilAttachment();
            }
        default:
            UNREACHABLE();
            return nullptr;
    }
}

size_t FramebufferState::getReadIndex() const
{
    ASSERT(mReadBufferState == GL_BACK ||
           (mReadBufferState >= GL_COLOR_ATTACHMENT0 && mReadBufferState <= GL_COLOR_ATTACHMENT15));
    size_t readIndex = (mReadBufferState == GL_BACK
                            ? 0
                            : static_cast<size_t>(mReadBufferState - GL_COLOR_ATTACHMENT0));
    ASSERT(readIndex < mColorAttachments.size());
    return readIndex;
}

const FramebufferAttachment *FramebufferState::getReadAttachment() const
{
    if (mReadBufferState == GL_NONE)
    {
        return nullptr;
    }

    size_t readIndex = getReadIndex();
    const gl::FramebufferAttachment &framebufferAttachment =
        isDefault() ? mDefaultFramebufferReadAttachment : mColorAttachments[readIndex];

    return framebufferAttachment.isAttached() ? &framebufferAttachment : nullptr;
}

const FramebufferAttachment *FramebufferState::getFirstNonNullAttachment() const
{
    auto *colorAttachment = getFirstColorAttachment();
    if (colorAttachment)
    {
        return colorAttachment;
    }
    return getDepthOrStencilAttachment();
}

const FramebufferAttachment *FramebufferState::getFirstColorAttachment() const
{
    for (const FramebufferAttachment &colorAttachment : mColorAttachments)
    {
        if (colorAttachment.isAttached())
        {
            return &colorAttachment;
        }
    }

    return nullptr;
}

const FramebufferAttachment *FramebufferState::getDepthOrStencilAttachment() const
{
    if (mDepthAttachment.isAttached())
    {
        return &mDepthAttachment;
    }
    if (mStencilAttachment.isAttached())
    {
        return &mStencilAttachment;
    }
    return nullptr;
}

const FramebufferAttachment *FramebufferState::getStencilOrDepthStencilAttachment() const
{
    if (mStencilAttachment.isAttached())
    {
        return &mStencilAttachment;
    }
    return getDepthStencilAttachment();
}

const FramebufferAttachment *FramebufferState::getColorAttachment(size_t colorAttachment) const
{
    ASSERT(colorAttachment < mColorAttachments.size());
    return mColorAttachments[colorAttachment].isAttached() ? &mColorAttachments[colorAttachment]
                                                           : nullptr;
}

const FramebufferAttachment *FramebufferState::getDepthAttachment() const
{
    return mDepthAttachment.isAttached() ? &mDepthAttachment : nullptr;
}

const FramebufferAttachment *FramebufferState::getWebGLDepthAttachment() const
{
    return mWebGLDepthAttachment.isAttached() ? &mWebGLDepthAttachment : nullptr;
}

const FramebufferAttachment *FramebufferState::getWebGLDepthStencilAttachment() const
{
    return mWebGLDepthStencilAttachment.isAttached() ? &mWebGLDepthStencilAttachment : nullptr;
}

const FramebufferAttachment *FramebufferState::getStencilAttachment() const
{
    return mStencilAttachment.isAttached() ? &mStencilAttachment : nullptr;
}

const FramebufferAttachment *FramebufferState::getWebGLStencilAttachment() const
{
    return mWebGLStencilAttachment.isAttached() ? &mWebGLStencilAttachment : nullptr;
}

const FramebufferAttachment *FramebufferState::getDepthStencilAttachment() const
{
    // A valid depth-stencil attachment has the same resource bound to both the
    // depth and stencil attachment points.
    if (mDepthAttachment.isAttached() && mStencilAttachment.isAttached() &&
        mDepthAttachment == mStencilAttachment)
    {
        return &mDepthAttachment;
    }

    return nullptr;
}

bool FramebufferState::attachmentsHaveSameDimensions() const
{
    Optional<Extents> attachmentSize;

    auto hasMismatchedSize = [&attachmentSize](const FramebufferAttachment &attachment) {
        if (!attachment.isAttached())
        {
            return false;
        }

        if (!attachmentSize.valid())
        {
            attachmentSize = attachment.getSize();
            return false;
        }

        const auto &prevSize = attachmentSize.value();
        const auto &curSize  = attachment.getSize();
        return (curSize.width != prevSize.width || curSize.height != prevSize.height);
    };

    for (const auto &attachment : mColorAttachments)
    {
        if (hasMismatchedSize(attachment))
        {
            return false;
        }
    }

    if (hasMismatchedSize(mDepthAttachment))
    {
        return false;
    }

    return !hasMismatchedSize(mStencilAttachment);
}

bool FramebufferState::hasSeparateDepthAndStencilAttachments() const
{
    // if we have both a depth and stencil buffer, they must refer to the same object
    // since we only support packed_depth_stencil and not separate depth and stencil
    return (getDepthAttachment() != nullptr && getStencilAttachment() != nullptr &&
            getDepthStencilAttachment() == nullptr);
}

const FramebufferAttachment *FramebufferState::getDrawBuffer(size_t drawBufferIdx) const
{
    ASSERT(drawBufferIdx < mDrawBufferStates.size());
    if (mDrawBufferStates[drawBufferIdx] != GL_NONE)
    {
        // ES3 spec: "If the GL is bound to a draw framebuffer object, the ith buffer listed in bufs
        // must be COLOR_ATTACHMENTi or NONE"
        ASSERT(mDrawBufferStates[drawBufferIdx] == GL_COLOR_ATTACHMENT0 + drawBufferIdx ||
               (drawBufferIdx == 0 && mDrawBufferStates[drawBufferIdx] == GL_BACK));

        if (mDrawBufferStates[drawBufferIdx] == GL_BACK)
        {
            return getColorAttachment(0);
        }
        else
        {
            return getColorAttachment(mDrawBufferStates[drawBufferIdx] - GL_COLOR_ATTACHMENT0);
        }
    }
    else
    {
        return nullptr;
    }
}

size_t FramebufferState::getDrawBufferCount() const
{
    return mDrawBufferStates.size();
}

bool FramebufferState::colorAttachmentsAreUniqueImages() const
{
    for (size_t firstAttachmentIdx = 0; firstAttachmentIdx < mColorAttachments.size();
         firstAttachmentIdx++)
    {
        const FramebufferAttachment &firstAttachment = mColorAttachments[firstAttachmentIdx];
        if (!firstAttachment.isAttached())
        {
            continue;
        }

        for (size_t secondAttachmentIdx = firstAttachmentIdx + 1;
             secondAttachmentIdx < mColorAttachments.size(); secondAttachmentIdx++)
        {
            const FramebufferAttachment &secondAttachment = mColorAttachments[secondAttachmentIdx];
            if (!secondAttachment.isAttached())
            {
                continue;
            }

            if (firstAttachment == secondAttachment)
            {
                return false;
            }
        }
    }

    return true;
}

bool FramebufferState::hasDepth() const
{
    return (mDepthAttachment.isAttached() && mDepthAttachment.getDepthSize() > 0);
}

bool FramebufferState::hasStencil() const
{
    return (mStencilAttachment.isAttached() && mStencilAttachment.getStencilSize() > 0);
}

bool FramebufferState::isMultiview() const
{
    const FramebufferAttachment *attachment = getFirstNonNullAttachment();
    if (attachment == nullptr)
    {
        return false;
    }
    return attachment->isMultiview();
}

int FramebufferState::getBaseViewIndex() const
{
    const FramebufferAttachment *attachment = getFirstNonNullAttachment();
    if (attachment == nullptr)
    {
        return GL_NONE;
    }
    return attachment->getBaseViewIndex();
}

Box FramebufferState::getDimensions() const
{
    Extents extents = getExtents();
    return Box(0, 0, 0, extents.width, extents.height, extents.depth);
}

Extents FramebufferState::getExtents() const
{
    ASSERT(attachmentsHaveSameDimensions());
    const FramebufferAttachment *first = getFirstNonNullAttachment();
    if (first)
    {
        return first->getSize();
    }
    return Extents(getDefaultWidth(), getDefaultHeight(), 0);
}

bool FramebufferState::isDefault() const
{
    return mId == Framebuffer::kDefaultDrawFramebufferHandle;
}

const FramebufferID Framebuffer::kDefaultDrawFramebufferHandle = {0};

Framebuffer::Framebuffer(const Caps &caps, rx::GLImplFactory *factory, FramebufferID id)
    : mState(caps, id),
      mImpl(factory->createFramebuffer(mState)),
      mCachedStatus(),
      mDirtyDepthAttachmentBinding(this, DIRTY_BIT_DEPTH_ATTACHMENT),
      mDirtyStencilAttachmentBinding(this, DIRTY_BIT_STENCIL_ATTACHMENT)
{
    ASSERT(mImpl != nullptr);
    ASSERT(mState.mColorAttachments.size() == static_cast<size_t>(caps.maxColorAttachments));

    for (uint32_t colorIndex = 0;
         colorIndex < static_cast<uint32_t>(mState.mColorAttachments.size()); ++colorIndex)
    {
        mDirtyColorAttachmentBindings.emplace_back(this, DIRTY_BIT_COLOR_ATTACHMENT_0 + colorIndex);
    }
    mDirtyBits.set(DIRTY_BIT_READ_BUFFER);
}

Framebuffer::Framebuffer(const Context *context, egl::Surface *surface, egl::Surface *readSurface)
    : mState(),
      mImpl(surface->getImplementation()->createDefaultFramebuffer(context, mState)),
      mCachedStatus(GL_FRAMEBUFFER_COMPLETE),
      mDirtyDepthAttachmentBinding(this, DIRTY_BIT_DEPTH_ATTACHMENT),
      mDirtyStencilAttachmentBinding(this, DIRTY_BIT_STENCIL_ATTACHMENT)
{
    ASSERT(mImpl != nullptr);

    mDirtyColorAttachmentBindings.emplace_back(this, DIRTY_BIT_COLOR_ATTACHMENT_0);
    setAttachmentImpl(context, GL_FRAMEBUFFER_DEFAULT, GL_BACK, ImageIndex(), surface,
                      FramebufferAttachment::kDefaultNumViews,
                      FramebufferAttachment::kDefaultBaseViewIndex, false,
                      FramebufferAttachment::kDefaultRenderToTextureSamples);

    setReadSurface(context, readSurface);

    if (surface->getConfig()->depthSize > 0)
    {
        setAttachmentImpl(context, GL_FRAMEBUFFER_DEFAULT, GL_DEPTH, ImageIndex(), surface,
                          FramebufferAttachment::kDefaultNumViews,
                          FramebufferAttachment::kDefaultBaseViewIndex, false,
                          FramebufferAttachment::kDefaultRenderToTextureSamples);
    }

    if (surface->getConfig()->stencilSize > 0)
    {
        setAttachmentImpl(context, GL_FRAMEBUFFER_DEFAULT, GL_STENCIL, ImageIndex(), surface,
                          FramebufferAttachment::kDefaultNumViews,
                          FramebufferAttachment::kDefaultBaseViewIndex, false,
                          FramebufferAttachment::kDefaultRenderToTextureSamples);
    }
    SetComponentTypeMask(getDrawbufferWriteType(0), 0, &mState.mDrawBufferTypeMask);

    // Ensure the backend has a chance to synchronize its content for a new backbuffer.
    mDirtyBits.set(DIRTY_BIT_COLOR_BUFFER_CONTENTS_0);
}

Framebuffer::Framebuffer(const Context *context,
                         rx::GLImplFactory *factory,
                         egl::Surface *readSurface)
    : mState(),
      mImpl(factory->createFramebuffer(mState)),
      mCachedStatus(GL_FRAMEBUFFER_UNDEFINED_OES),
      mDirtyDepthAttachmentBinding(this, DIRTY_BIT_DEPTH_ATTACHMENT),
      mDirtyStencilAttachmentBinding(this, DIRTY_BIT_STENCIL_ATTACHMENT)
{
    mDirtyColorAttachmentBindings.emplace_back(this, DIRTY_BIT_COLOR_ATTACHMENT_0);
    SetComponentTypeMask(getDrawbufferWriteType(0), 0, &mState.mDrawBufferTypeMask);

    setReadSurface(context, readSurface);
}

Framebuffer::~Framebuffer()
{
    SafeDelete(mImpl);
}

void Framebuffer::onDestroy(const Context *context)
{
    if (isDefault())
    {
        mState.mDefaultFramebufferReadAttachment.detach(context);
        mState.mDefaultFramebufferReadAttachmentInitialized = false;
    }

    for (auto &attachment : mState.mColorAttachments)
    {
        attachment.detach(context);
    }
    mState.mDepthAttachment.detach(context);
    mState.mStencilAttachment.detach(context);
    mState.mWebGLDepthAttachment.detach(context);
    mState.mWebGLStencilAttachment.detach(context);
    mState.mWebGLDepthStencilAttachment.detach(context);

    mImpl->destroy(context);
}

void Framebuffer::setReadSurface(const Context *context, egl::Surface *readSurface)
{
    // updateAttachment() without mState.mResourceNeedsInit.set()
    mState.mDefaultFramebufferReadAttachment.attach(
        context, GL_FRAMEBUFFER_DEFAULT, GL_BACK, ImageIndex(), readSurface,
        FramebufferAttachment::kDefaultNumViews, FramebufferAttachment::kDefaultBaseViewIndex,
        false, FramebufferAttachment::kDefaultRenderToTextureSamples);
    mDirtyBits.set(DIRTY_BIT_READ_BUFFER);
}

void Framebuffer::setLabel(const Context *context, const std::string &label)
{
    mState.mLabel = label;
}

const std::string &Framebuffer::getLabel() const
{
    return mState.mLabel;
}

bool Framebuffer::detachTexture(const Context *context, TextureID textureId)
{
    return detachResourceById(context, GL_TEXTURE, textureId.value);
}

bool Framebuffer::detachRenderbuffer(const Context *context, RenderbufferID renderbufferId)
{
    return detachResourceById(context, GL_RENDERBUFFER, renderbufferId.value);
}

bool Framebuffer::detachResourceById(const Context *context, GLenum resourceType, GLuint resourceId)
{
    bool found = false;

    for (size_t colorIndex = 0; colorIndex < mState.mColorAttachments.size(); ++colorIndex)
    {
        if (detachMatchingAttachment(context, &mState.mColorAttachments[colorIndex], resourceType,
                                     resourceId))
        {
            found = true;
        }
    }

    if (context->isWebGL1())
    {
        const std::array<FramebufferAttachment *, 3> attachments = {
            {&mState.mWebGLDepthStencilAttachment, &mState.mWebGLDepthAttachment,
             &mState.mWebGLStencilAttachment}};
        for (FramebufferAttachment *attachment : attachments)
        {
            if (detachMatchingAttachment(context, attachment, resourceType, resourceId))
            {
                found = true;
            }
        }
    }
    else
    {
        if (detachMatchingAttachment(context, &mState.mDepthAttachment, resourceType, resourceId))
        {
            found = true;
        }
        if (detachMatchingAttachment(context, &mState.mStencilAttachment, resourceType, resourceId))
        {
            found = true;
        }
    }

    return found;
}

bool Framebuffer::detachMatchingAttachment(const Context *context,
                                           FramebufferAttachment *attachment,
                                           GLenum matchType,
                                           GLuint matchId)
{
    if (attachment->isAttached() && attachment->type() == matchType && attachment->id() == matchId)
    {
        // We go through resetAttachment to make sure that all the required bookkeeping will be done
        // such as updating enabled draw buffer state.
        resetAttachment(context, attachment->getBinding());
        return true;
    }

    return false;
}

const FramebufferAttachment *Framebuffer::getColorAttachment(size_t colorAttachment) const
{
    return mState.getColorAttachment(colorAttachment);
}

const FramebufferAttachment *Framebuffer::getDepthAttachment() const
{
    return mState.getDepthAttachment();
}

const FramebufferAttachment *Framebuffer::getStencilAttachment() const
{
    return mState.getStencilAttachment();
}

const FramebufferAttachment *Framebuffer::getDepthStencilAttachment() const
{
    return mState.getDepthStencilAttachment();
}

const FramebufferAttachment *Framebuffer::getDepthOrStencilAttachment() const
{
    return mState.getDepthOrStencilAttachment();
}

const FramebufferAttachment *Framebuffer::getStencilOrDepthStencilAttachment() const
{
    return mState.getStencilOrDepthStencilAttachment();
}

const FramebufferAttachment *Framebuffer::getReadColorAttachment() const
{
    return mState.getReadAttachment();
}

GLenum Framebuffer::getReadColorAttachmentType() const
{
    const FramebufferAttachment *readAttachment = mState.getReadAttachment();
    return (readAttachment != nullptr ? readAttachment->type() : GL_NONE);
}

const FramebufferAttachment *Framebuffer::getFirstColorAttachment() const
{
    return mState.getFirstColorAttachment();
}

const FramebufferAttachment *Framebuffer::getFirstNonNullAttachment() const
{
    return mState.getFirstNonNullAttachment();
}

const FramebufferAttachment *Framebuffer::getAttachment(const Context *context,
                                                        GLenum attachment) const
{
    return mState.getAttachment(context, attachment);
}

size_t Framebuffer::getDrawbufferStateCount() const
{
    return mState.mDrawBufferStates.size();
}

GLenum Framebuffer::getDrawBufferState(size_t drawBuffer) const
{
    ASSERT(drawBuffer < mState.mDrawBufferStates.size());
    return mState.mDrawBufferStates[drawBuffer];
}

const std::vector<GLenum> &Framebuffer::getDrawBufferStates() const
{
    return mState.getDrawBufferStates();
}

void Framebuffer::setDrawBuffers(size_t count, const GLenum *buffers)
{
    auto &drawStates = mState.mDrawBufferStates;

    ASSERT(count <= drawStates.size());
    std::copy(buffers, buffers + count, drawStates.begin());
    std::fill(drawStates.begin() + count, drawStates.end(), GL_NONE);
    mDirtyBits.set(DIRTY_BIT_DRAW_BUFFERS);

    mState.mEnabledDrawBuffers.reset();
    mState.mDrawBufferTypeMask.reset();

    for (size_t index = 0; index < count; ++index)
    {
        SetComponentTypeMask(getDrawbufferWriteType(index), index, &mState.mDrawBufferTypeMask);

        if (drawStates[index] != GL_NONE && mState.mColorAttachments[index].isAttached())
        {
            mState.mEnabledDrawBuffers.set(index);
        }
    }
}

const FramebufferAttachment *Framebuffer::getDrawBuffer(size_t drawBuffer) const
{
    return mState.getDrawBuffer(drawBuffer);
}

ComponentType Framebuffer::getDrawbufferWriteType(size_t drawBuffer) const
{
    const FramebufferAttachment *attachment = mState.getDrawBuffer(drawBuffer);
    if (attachment == nullptr)
    {
        return ComponentType::NoType;
    }

    GLenum componentType = attachment->getFormat().info->componentType;
    switch (componentType)
    {
        case GL_INT:
            return ComponentType::Int;
        case GL_UNSIGNED_INT:
            return ComponentType::UnsignedInt;

        default:
            return ComponentType::Float;
    }
}

ComponentTypeMask Framebuffer::getDrawBufferTypeMask() const
{
    return mState.mDrawBufferTypeMask;
}

DrawBufferMask Framebuffer::getDrawBufferMask() const
{
    return mState.mEnabledDrawBuffers;
}

bool Framebuffer::hasEnabledDrawBuffer() const
{
    for (size_t drawbufferIdx = 0; drawbufferIdx < mState.mDrawBufferStates.size(); ++drawbufferIdx)
    {
        if (getDrawBuffer(drawbufferIdx) != nullptr)
        {
            return true;
        }
    }

    return false;
}

GLenum Framebuffer::getReadBufferState() const
{
    return mState.mReadBufferState;
}

void Framebuffer::setReadBuffer(GLenum buffer)
{
    ASSERT(buffer == GL_BACK || buffer == GL_NONE ||
           (buffer >= GL_COLOR_ATTACHMENT0 &&
            (buffer - GL_COLOR_ATTACHMENT0) < mState.mColorAttachments.size()));
    mState.mReadBufferState = buffer;
    mDirtyBits.set(DIRTY_BIT_READ_BUFFER);
}

size_t Framebuffer::getNumColorAttachments() const
{
    return mState.mColorAttachments.size();
}

bool Framebuffer::hasDepth() const
{
    return mState.hasDepth();
}

bool Framebuffer::hasStencil() const
{
    return mState.hasStencil();
}

bool Framebuffer::usingExtendedDrawBuffers() const
{
    for (size_t drawbufferIdx = 1; drawbufferIdx < mState.mDrawBufferStates.size(); ++drawbufferIdx)
    {
        if (getDrawBuffer(drawbufferIdx) != nullptr)
        {
            return true;
        }
    }

    return false;
}

void Framebuffer::invalidateCompletenessCache()
{
    if (!isDefault())
    {
        mCachedStatus.reset();
    }
    onStateChange(angle::SubjectMessage::DirtyBitsFlagged);
}

GLenum Framebuffer::checkStatusImpl(const Context *context)
{
    ASSERT(!isDefault());
    ASSERT(hasAnyDirtyBit() || !mCachedStatus.valid());

    mCachedStatus = checkStatusWithGLFrontEnd(context);

    if (mCachedStatus.value() == GL_FRAMEBUFFER_COMPLETE)
    {
        // We can skip syncState on several back-ends.
        if (mImpl->shouldSyncStateBeforeCheckStatus())
        {
            angle::Result err = syncState(context);
            if (err != angle::Result::Continue)
            {
                return 0;
            }
        }

        if (!mImpl->checkStatus(context))
        {
            mCachedStatus = GL_FRAMEBUFFER_UNSUPPORTED;
        }
    }

    return mCachedStatus.value();
}

GLenum Framebuffer::checkStatusWithGLFrontEnd(const Context *context)
{
    const State &state = context->getState();

    ASSERT(!isDefault());

    bool hasAttachments = false;
    Optional<unsigned int> colorbufferSize;
    Optional<int> samples;
    Optional<bool> fixedSampleLocations;
    bool hasRenderbuffer = false;
    Optional<int> renderToTextureSamples;

    const FramebufferAttachment *firstAttachment = getFirstNonNullAttachment();

    Optional<bool> isLayered;
    Optional<TextureType> colorAttachmentsTextureType;

    for (const FramebufferAttachment &colorAttachment : mState.mColorAttachments)
    {
        if (colorAttachment.isAttached())
        {
            if (!CheckAttachmentCompleteness(context, colorAttachment))
            {
                return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
            }

            const InternalFormat &format = *colorAttachment.getFormat().info;
            if (format.depthBits > 0 || format.stencilBits > 0)
            {
                return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
            }

            if (!CheckAttachmentSampleCompleteness(context, colorAttachment, true, &samples,
                                                   &fixedSampleLocations, &renderToTextureSamples))
            {
                return GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;
            }

            // in GLES 2.0, all color attachments attachments must have the same number of bitplanes
            // in GLES 3.0, there is no such restriction
            if (state.getClientMajorVersion() < 3)
            {
                if (colorbufferSize.valid())
                {
                    if (format.pixelBytes != colorbufferSize.value())
                    {
                        return GL_FRAMEBUFFER_UNSUPPORTED;
                    }
                }
                else
                {
                    colorbufferSize = format.pixelBytes;
                }
            }

            if (!CheckMultiviewStateMatchesForCompleteness(firstAttachment, &colorAttachment))
            {
                return GL_FRAMEBUFFER_INCOMPLETE_VIEW_TARGETS_OVR;
            }

            hasRenderbuffer = hasRenderbuffer || (colorAttachment.type() == GL_RENDERBUFFER);

            if (!hasAttachments)
            {
                isLayered = colorAttachment.isLayered();
                if (isLayered.value())
                {
                    colorAttachmentsTextureType = colorAttachment.getTextureImageIndex().getType();
                }
                hasAttachments = true;
            }
            else
            {
                // [EXT_geometry_shader] section 9.4.1, "Framebuffer Completeness"
                // If any framebuffer attachment is layered, all populated attachments
                // must be layered. Additionally, all populated color attachments must
                // be from textures of the same target. {FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT }
                ASSERT(isLayered.valid());
                if (isLayered.value() != colorAttachment.isLayered())
                {
                    return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT;
                }
                else if (isLayered.value())
                {
                    ASSERT(colorAttachmentsTextureType.valid());
                    if (colorAttachmentsTextureType.value() !=
                        colorAttachment.getTextureImageIndex().getType())
                    {
                        return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT;
                    }
                }
            }
        }
    }

    const FramebufferAttachment &depthAttachment = mState.mDepthAttachment;
    if (depthAttachment.isAttached())
    {
        if (!CheckAttachmentCompleteness(context, depthAttachment))
        {
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }

        const InternalFormat &format = *depthAttachment.getFormat().info;
        if (format.depthBits == 0)
        {
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }

        if (!CheckAttachmentSampleCompleteness(context, depthAttachment, false, &samples,
                                               &fixedSampleLocations, &renderToTextureSamples))
        {
            return GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;
        }

        if (!CheckMultiviewStateMatchesForCompleteness(firstAttachment, &depthAttachment))
        {
            return GL_FRAMEBUFFER_INCOMPLETE_VIEW_TARGETS_OVR;
        }

        hasRenderbuffer = hasRenderbuffer || (depthAttachment.type() == GL_RENDERBUFFER);

        if (!hasAttachments)
        {
            isLayered      = depthAttachment.isLayered();
            hasAttachments = true;
        }
        else
        {
            // [EXT_geometry_shader] section 9.4.1, "Framebuffer Completeness"
            // If any framebuffer attachment is layered, all populated attachments
            // must be layered. {FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT }
            ASSERT(isLayered.valid());
            if (isLayered.value() != depthAttachment.isLayered())
            {
                return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT;
            }
        }
    }

    const FramebufferAttachment &stencilAttachment = mState.mStencilAttachment;
    if (stencilAttachment.isAttached())
    {
        if (!CheckAttachmentCompleteness(context, stencilAttachment))
        {
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }

        const InternalFormat &format = *stencilAttachment.getFormat().info;
        if (format.stencilBits == 0)
        {
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }

        if (!CheckAttachmentSampleCompleteness(context, stencilAttachment, false, &samples,
                                               &fixedSampleLocations, &renderToTextureSamples))
        {
            return GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;
        }

        if (!CheckMultiviewStateMatchesForCompleteness(firstAttachment, &stencilAttachment))
        {
            return GL_FRAMEBUFFER_INCOMPLETE_VIEW_TARGETS_OVR;
        }

        hasRenderbuffer = hasRenderbuffer || (stencilAttachment.type() == GL_RENDERBUFFER);

        if (!hasAttachments)
        {
            hasAttachments = true;
        }
        else
        {
            // [EXT_geometry_shader] section 9.4.1, "Framebuffer Completeness"
            // If any framebuffer attachment is layered, all populated attachments
            // must be layered.
            // {FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT }
            ASSERT(isLayered.valid());
            if (isLayered.value() != stencilAttachment.isLayered())
            {
                return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT;
            }
        }
    }

    // Starting from ES 3.0 stencil and depth, if present, should be the same image
    if (state.getClientMajorVersion() >= 3 && depthAttachment.isAttached() &&
        stencilAttachment.isAttached() && stencilAttachment != depthAttachment)
    {
        return GL_FRAMEBUFFER_UNSUPPORTED;
    }

    // Special additional validation for WebGL 1 DEPTH/STENCIL/DEPTH_STENCIL.
    if (state.isWebGL1())
    {
        if (!mState.mWebGLDepthStencilConsistent)
        {
            return GL_FRAMEBUFFER_UNSUPPORTED;
        }

        if (mState.mWebGLDepthStencilAttachment.isAttached())
        {
            if (mState.mWebGLDepthStencilAttachment.getDepthSize() == 0 ||
                mState.mWebGLDepthStencilAttachment.getStencilSize() == 0)
            {
                return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
            }

            if (!CheckMultiviewStateMatchesForCompleteness(firstAttachment,
                                                           &mState.mWebGLDepthStencilAttachment))
            {
                return GL_FRAMEBUFFER_INCOMPLETE_VIEW_TARGETS_OVR;
            }
        }
        else if (mState.mStencilAttachment.isAttached() &&
                 mState.mStencilAttachment.getDepthSize() > 0)
        {
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }
        else if (mState.mDepthAttachment.isAttached() &&
                 mState.mDepthAttachment.getStencilSize() > 0)
        {
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }
    }

    // ES3.1(section 9.4) requires that if no image is attached to the framebuffer, and either the
    // value of the framebuffer's FRAMEBUFFER_DEFAULT_WIDTH or FRAMEBUFFER_DEFAULT_HEIGHT parameters
    // is zero, the framebuffer is considered incomplete.
    GLint defaultWidth  = mState.getDefaultWidth();
    GLint defaultHeight = mState.getDefaultHeight();
    if (!hasAttachments && (defaultWidth == 0 || defaultHeight == 0))
    {
        return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
    }

    // In ES 2.0 and WebGL, all color attachments must have the same width and height.
    // In ES 3.0, there is no such restriction.
    if ((state.getClientMajorVersion() < 3 || state.getExtensions().webglCompatibility) &&
        !mState.attachmentsHaveSameDimensions())
    {
        return GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
    }

    // ES3.1(section 9.4) requires that if the attached images are a mix of renderbuffers and
    // textures, the value of TEXTURE_FIXED_SAMPLE_LOCATIONS must be TRUE for all attached textures.
    if (fixedSampleLocations.valid() && hasRenderbuffer && !fixedSampleLocations.value())
    {
        return GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;
    }

    // The WebGL conformance tests implicitly define that all framebuffer
    // attachments must be unique. For example, the same level of a texture can
    // not be attached to two different color attachments.
    if (state.getExtensions().webglCompatibility)
    {
        if (!mState.colorAttachmentsAreUniqueImages())
        {
            return GL_FRAMEBUFFER_UNSUPPORTED;
        }
    }

    return GL_FRAMEBUFFER_COMPLETE;
}

angle::Result Framebuffer::discard(const Context *context, size_t count, const GLenum *attachments)
{
    // Back-ends might make the contents of the FBO undefined. In WebGL 2.0, invalidate operations
    // can be no-ops, so we should probably do that to ensure consistency.
    // TODO(jmadill): WebGL behaviour, and robust resource init behaviour without WebGL.

    return mImpl->discard(context, count, attachments);
}

angle::Result Framebuffer::invalidate(const Context *context,
                                      size_t count,
                                      const GLenum *attachments)
{
    // Back-ends might make the contents of the FBO undefined. In WebGL 2.0, invalidate operations
    // can be no-ops, so we should probably do that to ensure consistency.
    // TODO(jmadill): WebGL behaviour, and robust resource init behaviour without WebGL.

    return mImpl->invalidate(context, count, attachments);
}

bool Framebuffer::partialClearNeedsInit(const Context *context,
                                        bool color,
                                        bool depth,
                                        bool stencil)
{
    const auto &glState = context->getState();

    if (!glState.isRobustResourceInitEnabled())
    {
        return false;
    }

    // Scissors can affect clearing.
    // TODO(jmadill): Check for complete scissor overlap.
    if (glState.isScissorTestEnabled())
    {
        return true;
    }

    // If colors masked, we must clear before we clear. Do a simple check.
    // TODO(jmadill): Filter out unused color channels from the test.
    if (color)
    {
        const auto &blend = glState.getBlendState();
        if (!(blend.colorMaskRed && blend.colorMaskGreen && blend.colorMaskBlue &&
              blend.colorMaskAlpha))
        {
            return true;
        }
    }

    const auto &depthStencil = glState.getDepthStencilState();
    if (stencil && (depthStencil.stencilMask != depthStencil.stencilWritemask ||
                    depthStencil.stencilBackMask != depthStencil.stencilBackWritemask))
    {
        return true;
    }

    return false;
}

angle::Result Framebuffer::invalidateSub(const Context *context,
                                         size_t count,
                                         const GLenum *attachments,
                                         const Rectangle &area)
{
    // Back-ends might make the contents of the FBO undefined. In WebGL 2.0, invalidate operations
    // can be no-ops, so we should probably do that to ensure consistency.
    // TODO(jmadill): Make a invalidate no-op in WebGL 2.0.

    return mImpl->invalidateSub(context, count, attachments, area);
}

angle::Result Framebuffer::clear(const Context *context, GLbitfield mask)
{
    const auto &glState = context->getState();
    if (glState.isRasterizerDiscardEnabled())
    {
        return angle::Result::Continue;
    }

    // Remove clear bits that are ineffective. An effective clear changes at least one fragment. If
    // color/depth/stencil masks make the clear ineffective we skip it altogether.

    // If all color channels are masked, don't attempt to clear color.
    if (context->getState().getBlendState().allChannelsMasked())
    {
        mask &= ~GL_COLOR_BUFFER_BIT;
    }

    // If depth write is disabled, don't attempt to clear depth.
    if (!context->getState().getDepthStencilState().depthMask)
    {
        mask &= ~GL_DEPTH_BUFFER_BIT;
    }

    // If all stencil bits are masked, don't attempt to clear stencil.
    if (context->getState().getDepthStencilState().stencilWritemask == 0)
    {
        mask &= ~GL_STENCIL_BUFFER_BIT;
    }

    if (mask != 0)
    {
        ANGLE_TRY(mImpl->clear(context, mask));
    }

    return angle::Result::Continue;
}

angle::Result Framebuffer::clearBufferfv(const Context *context,
                                         GLenum buffer,
                                         GLint drawbuffer,
                                         const GLfloat *values)
{
    if (context->getState().isRasterizerDiscardEnabled() || IsClearBufferMaskedOut(context, buffer))
    {
        return angle::Result::Continue;
    }

    if (buffer == GL_DEPTH)
    {
        // If depth write is disabled, don't attempt to clear depth.
        if (!context->getState().getDepthStencilState().depthMask)
        {
            return angle::Result::Continue;
        }
    }
    else
    {
        // If all color channels are masked, don't attempt to clear color.
        if (context->getState().getBlendState().allChannelsMasked())
        {
            return angle::Result::Continue;
        }
    }

    ANGLE_TRY(mImpl->clearBufferfv(context, buffer, drawbuffer, values));

    return angle::Result::Continue;
}

angle::Result Framebuffer::clearBufferuiv(const Context *context,
                                          GLenum buffer,
                                          GLint drawbuffer,
                                          const GLuint *values)
{
    if (context->getState().isRasterizerDiscardEnabled() || IsClearBufferMaskedOut(context, buffer))
    {
        return angle::Result::Continue;
    }

    // If all color channels are masked, don't attempt to clear color.
    if (context->getState().getBlendState().allChannelsMasked())
    {
        return angle::Result::Continue;
    }

    ANGLE_TRY(mImpl->clearBufferuiv(context, buffer, drawbuffer, values));

    return angle::Result::Continue;
}

angle::Result Framebuffer::clearBufferiv(const Context *context,
                                         GLenum buffer,
                                         GLint drawbuffer,
                                         const GLint *values)
{
    if (context->getState().isRasterizerDiscardEnabled() || IsClearBufferMaskedOut(context, buffer))
    {
        return angle::Result::Continue;
    }

    if (buffer == GL_STENCIL)
    {
        // If all stencil bits are masked, don't attempt to clear stencil.
        if (context->getState().getDepthStencilState().stencilWritemask == 0)
        {
            return angle::Result::Continue;
        }
    }
    else
    {
        // If all color channels are masked, don't attempt to clear color.
        if (context->getState().getBlendState().allChannelsMasked())
        {
            return angle::Result::Continue;
        }
    }

    ANGLE_TRY(mImpl->clearBufferiv(context, buffer, drawbuffer, values));

    return angle::Result::Continue;
}

angle::Result Framebuffer::clearBufferfi(const Context *context,
                                         GLenum buffer,
                                         GLint drawbuffer,
                                         GLfloat depth,
                                         GLint stencil)
{
    if (context->getState().isRasterizerDiscardEnabled() || IsClearBufferMaskedOut(context, buffer))
    {
        return angle::Result::Continue;
    }

    bool clearDepth   = context->getState().getDepthStencilState().depthMask;
    bool clearStencil = context->getState().getDepthStencilState().stencilWritemask != 0;

    if (clearDepth && clearStencil)
    {
        ASSERT(buffer == GL_DEPTH_STENCIL);
        ANGLE_TRY(mImpl->clearBufferfi(context, GL_DEPTH_STENCIL, drawbuffer, depth, stencil));
    }
    else if (clearDepth && !clearStencil)
    {
        ANGLE_TRY(mImpl->clearBufferfv(context, GL_DEPTH, drawbuffer, &depth));
    }
    else if (!clearDepth && clearStencil)
    {
        ANGLE_TRY(mImpl->clearBufferiv(context, GL_STENCIL, drawbuffer, &stencil));
    }

    return angle::Result::Continue;
}

angle::Result Framebuffer::getImplementationColorReadFormat(const Context *context,
                                                            GLenum *formatOut)
{
    ANGLE_TRY(syncState(context));
    *formatOut = mImpl->getImplementationColorReadFormat(context);
    return angle::Result::Continue;
}

angle::Result Framebuffer::getImplementationColorReadType(const Context *context, GLenum *typeOut)
{
    ANGLE_TRY(syncState(context));
    *typeOut = mImpl->getImplementationColorReadType(context);
    return angle::Result::Continue;
}

angle::Result Framebuffer::readPixels(const Context *context,
                                      const Rectangle &area,
                                      GLenum format,
                                      GLenum type,
                                      void *pixels)
{
    ANGLE_TRY(mImpl->readPixels(context, area, format, type, pixels));

    Buffer *unpackBuffer = context->getState().getTargetBuffer(BufferBinding::PixelUnpack);
    if (unpackBuffer)
    {
        unpackBuffer->onDataChanged();
    }

    return angle::Result::Continue;
}

angle::Result Framebuffer::blit(const Context *context,
                                const Rectangle &sourceArea,
                                const Rectangle &destArea,
                                GLbitfield mask,
                                GLenum filter)
{
    GLbitfield blitMask = mask;

    // Note that blitting is called against draw framebuffer.
    // See the code in gl::Context::blitFramebuffer.
    if ((mask & GL_COLOR_BUFFER_BIT) && !hasEnabledDrawBuffer())
    {
        blitMask &= ~GL_COLOR_BUFFER_BIT;
    }

    if ((mask & GL_STENCIL_BUFFER_BIT) && mState.getStencilAttachment() == nullptr)
    {
        blitMask &= ~GL_STENCIL_BUFFER_BIT;
    }

    if ((mask & GL_DEPTH_BUFFER_BIT) && mState.getDepthAttachment() == nullptr)
    {
        blitMask &= ~GL_DEPTH_BUFFER_BIT;
    }

    if (!blitMask)
    {
        return angle::Result::Continue;
    }

    return mImpl->blit(context, sourceArea, destArea, blitMask, filter);
}

int Framebuffer::getSamples(const Context *context)
{
    return (isComplete(context) ? getCachedSamples(context, AttachmentSampleType::Emulated) : 0);
}

int Framebuffer::getResourceSamples(const Context *context)
{
    return (isComplete(context) ? getCachedSamples(context, AttachmentSampleType::Resource) : 0);
}

int Framebuffer::getCachedSamples(const Context *context, AttachmentSampleType sampleType) const
{
    ASSERT(mCachedStatus.valid() && mCachedStatus.value() == GL_FRAMEBUFFER_COMPLETE);

    // For a complete framebuffer, all attachments must have the same sample count.
    // In this case return the first nonzero sample size.
    const auto *firstNonNullAttachment = mState.getFirstNonNullAttachment();
    if (firstNonNullAttachment)
    {
        ASSERT(firstNonNullAttachment->isAttached());
        if (sampleType == AttachmentSampleType::Resource)
        {
            return firstNonNullAttachment->getResourceSamples();
        }
        else
        {
            ASSERT(sampleType == AttachmentSampleType::Emulated);
            return firstNonNullAttachment->getSamples();
        }
    }

    // No attachments found.
    return 0;
}

angle::Result Framebuffer::getSamplePosition(const Context *context,
                                             size_t index,
                                             GLfloat *xy) const
{
    ANGLE_TRY(mImpl->getSamplePosition(context, index, xy));
    return angle::Result::Continue;
}

bool Framebuffer::hasValidDepthStencil() const
{
    return mState.getDepthStencilAttachment() != nullptr;
}

void Framebuffer::setAttachment(const Context *context,
                                GLenum type,
                                GLenum binding,
                                const ImageIndex &textureIndex,
                                FramebufferAttachmentObject *resource)
{
    setAttachment(context, type, binding, textureIndex, resource,
                  FramebufferAttachment::kDefaultNumViews,
                  FramebufferAttachment::kDefaultBaseViewIndex, false,
                  FramebufferAttachment::kDefaultRenderToTextureSamples);
}

void Framebuffer::setAttachmentMultisample(const Context *context,
                                           GLenum type,
                                           GLenum binding,
                                           const ImageIndex &textureIndex,
                                           FramebufferAttachmentObject *resource,
                                           GLsizei samples)
{
    setAttachment(context, type, binding, textureIndex, resource,
                  FramebufferAttachment::kDefaultNumViews,
                  FramebufferAttachment::kDefaultBaseViewIndex, false, samples);
}

void Framebuffer::setAttachment(const Context *context,
                                GLenum type,
                                GLenum binding,
                                const ImageIndex &textureIndex,
                                FramebufferAttachmentObject *resource,
                                GLsizei numViews,
                                GLuint baseViewIndex,
                                bool isMultiview,
                                GLsizei samples)
{
    // Context may be null in unit tests.
    if (!context || !context->isWebGL1())
    {
        setAttachmentImpl(context, type, binding, textureIndex, resource, numViews, baseViewIndex,
                          isMultiview, samples);
        return;
    }

    switch (binding)
    {
        case GL_DEPTH_STENCIL:
        case GL_DEPTH_STENCIL_ATTACHMENT:
            mState.mWebGLDepthStencilAttachment.attach(context, type, binding, textureIndex,
                                                       resource, numViews, baseViewIndex,
                                                       isMultiview, samples);
            break;
        case GL_DEPTH:
        case GL_DEPTH_ATTACHMENT:
            mState.mWebGLDepthAttachment.attach(context, type, binding, textureIndex, resource,
                                                numViews, baseViewIndex, isMultiview, samples);
            break;
        case GL_STENCIL:
        case GL_STENCIL_ATTACHMENT:
            mState.mWebGLStencilAttachment.attach(context, type, binding, textureIndex, resource,
                                                  numViews, baseViewIndex, isMultiview, samples);
            break;
        default:
            setAttachmentImpl(context, type, binding, textureIndex, resource, numViews,
                              baseViewIndex, isMultiview, samples);
            return;
    }

    commitWebGL1DepthStencilIfConsistent(context, numViews, baseViewIndex, isMultiview, samples);
}

void Framebuffer::setAttachmentMultiview(const Context *context,
                                         GLenum type,
                                         GLenum binding,
                                         const ImageIndex &textureIndex,
                                         FramebufferAttachmentObject *resource,
                                         GLsizei numViews,
                                         GLint baseViewIndex)
{
    setAttachment(context, type, binding, textureIndex, resource, numViews, baseViewIndex, true,
                  FramebufferAttachment::kDefaultRenderToTextureSamples);
}

void Framebuffer::commitWebGL1DepthStencilIfConsistent(const Context *context,
                                                       GLsizei numViews,
                                                       GLuint baseViewIndex,
                                                       bool isMultiview,
                                                       GLsizei samples)
{
    int count = 0;

    std::array<FramebufferAttachment *, 3> attachments = {{&mState.mWebGLDepthStencilAttachment,
                                                           &mState.mWebGLDepthAttachment,
                                                           &mState.mWebGLStencilAttachment}};
    for (FramebufferAttachment *attachment : attachments)
    {
        if (attachment->isAttached())
        {
            count++;
        }
    }

    mState.mWebGLDepthStencilConsistent = (count <= 1);
    if (!mState.mWebGLDepthStencilConsistent)
    {
        // Inconsistent.
        return;
    }

    auto getImageIndexIfTextureAttachment = [](const FramebufferAttachment &attachment) {
        if (attachment.type() == GL_TEXTURE)
        {
            return attachment.getTextureImageIndex();
        }
        else
        {
            return ImageIndex();
        }
    };

    if (mState.mWebGLDepthAttachment.isAttached())
    {
        const auto &depth = mState.mWebGLDepthAttachment;
        setAttachmentImpl(context, depth.type(), GL_DEPTH_ATTACHMENT,
                          getImageIndexIfTextureAttachment(depth), depth.getResource(), numViews,
                          baseViewIndex, isMultiview, samples);
        setAttachmentImpl(context, GL_NONE, GL_STENCIL_ATTACHMENT, ImageIndex(), nullptr, numViews,
                          baseViewIndex, isMultiview, samples);
    }
    else if (mState.mWebGLStencilAttachment.isAttached())
    {
        const auto &stencil = mState.mWebGLStencilAttachment;
        setAttachmentImpl(context, GL_NONE, GL_DEPTH_ATTACHMENT, ImageIndex(), nullptr, numViews,
                          baseViewIndex, isMultiview, samples);
        setAttachmentImpl(context, stencil.type(), GL_STENCIL_ATTACHMENT,
                          getImageIndexIfTextureAttachment(stencil), stencil.getResource(),
                          numViews, baseViewIndex, isMultiview, samples);
    }
    else if (mState.mWebGLDepthStencilAttachment.isAttached())
    {
        const auto &depthStencil = mState.mWebGLDepthStencilAttachment;
        setAttachmentImpl(context, depthStencil.type(), GL_DEPTH_ATTACHMENT,
                          getImageIndexIfTextureAttachment(depthStencil),
                          depthStencil.getResource(), numViews, baseViewIndex, isMultiview,
                          samples);
        setAttachmentImpl(context, depthStencil.type(), GL_STENCIL_ATTACHMENT,
                          getImageIndexIfTextureAttachment(depthStencil),
                          depthStencil.getResource(), numViews, baseViewIndex, isMultiview,
                          samples);
    }
    else
    {
        setAttachmentImpl(context, GL_NONE, GL_DEPTH_ATTACHMENT, ImageIndex(), nullptr, numViews,
                          baseViewIndex, isMultiview, samples);
        setAttachmentImpl(context, GL_NONE, GL_STENCIL_ATTACHMENT, ImageIndex(), nullptr, numViews,
                          baseViewIndex, isMultiview, samples);
    }
}

void Framebuffer::setAttachmentImpl(const Context *context,
                                    GLenum type,
                                    GLenum binding,
                                    const ImageIndex &textureIndex,
                                    FramebufferAttachmentObject *resource,
                                    GLsizei numViews,
                                    GLuint baseViewIndex,
                                    bool isMultiview,
                                    GLsizei samples)
{
    switch (binding)
    {
        case GL_DEPTH_STENCIL:
        case GL_DEPTH_STENCIL_ATTACHMENT:
            updateAttachment(context, &mState.mDepthAttachment, DIRTY_BIT_DEPTH_ATTACHMENT,
                             &mDirtyDepthAttachmentBinding, type, binding, textureIndex, resource,
                             numViews, baseViewIndex, isMultiview, samples);
            updateAttachment(context, &mState.mStencilAttachment, DIRTY_BIT_STENCIL_ATTACHMENT,
                             &mDirtyStencilAttachmentBinding, type, binding, textureIndex, resource,
                             numViews, baseViewIndex, isMultiview, samples);
            break;

        case GL_DEPTH:
        case GL_DEPTH_ATTACHMENT:
            updateAttachment(context, &mState.mDepthAttachment, DIRTY_BIT_DEPTH_ATTACHMENT,
                             &mDirtyDepthAttachmentBinding, type, binding, textureIndex, resource,
                             numViews, baseViewIndex, isMultiview, samples);
            break;

        case GL_STENCIL:
        case GL_STENCIL_ATTACHMENT:
            updateAttachment(context, &mState.mStencilAttachment, DIRTY_BIT_STENCIL_ATTACHMENT,
                             &mDirtyStencilAttachmentBinding, type, binding, textureIndex, resource,
                             numViews, baseViewIndex, isMultiview, samples);
            break;

        case GL_BACK:
            updateAttachment(context, &mState.mColorAttachments[0], DIRTY_BIT_COLOR_ATTACHMENT_0,
                             &mDirtyColorAttachmentBindings[0], type, binding, textureIndex,
                             resource, numViews, baseViewIndex, isMultiview, samples);
            break;

        default:
        {
            size_t colorIndex = binding - GL_COLOR_ATTACHMENT0;
            ASSERT(colorIndex < mState.mColorAttachments.size());
            size_t dirtyBit = DIRTY_BIT_COLOR_ATTACHMENT_0 + colorIndex;
            updateAttachment(context, &mState.mColorAttachments[colorIndex], dirtyBit,
                             &mDirtyColorAttachmentBindings[colorIndex], type, binding,
                             textureIndex, resource, numViews, baseViewIndex, isMultiview, samples);

            if (!resource)
            {
                mColorAttachmentBits.reset(colorIndex);
                mFloat32ColorAttachmentBits.reset(colorIndex);
            }
            else
            {
                mColorAttachmentBits.set(colorIndex);
                updateFloat32ColorAttachmentBits(
                    colorIndex, resource->getAttachmentFormat(binding, textureIndex).info);
            }

            // TODO(jmadill): ASSERT instead of checking the attachment exists in
            // formsRenderingFeedbackLoopWith
            bool enabled = (type != GL_NONE && getDrawBufferState(colorIndex) != GL_NONE);
            mState.mEnabledDrawBuffers.set(colorIndex, enabled);
            SetComponentTypeMask(getDrawbufferWriteType(colorIndex), colorIndex,
                                 &mState.mDrawBufferTypeMask);
        }
        break;
    }
}

void Framebuffer::updateAttachment(const Context *context,
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
                                   GLsizei samples)
{
    attachment->attach(context, type, binding, textureIndex, resource, numViews, baseViewIndex,
                       isMultiview, samples);
    mDirtyBits.set(dirtyBit);
    mState.mResourceNeedsInit.set(dirtyBit, attachment->initState() == InitState::MayNeedInit);
    onDirtyBinding->bind(resource);

    invalidateCompletenessCache();
}

void Framebuffer::resetAttachment(const Context *context, GLenum binding)
{
    setAttachment(context, GL_NONE, binding, ImageIndex(), nullptr);
}

angle::Result Framebuffer::syncState(const Context *context)
{
    if (mDirtyBits.any())
    {
        mDirtyBitsGuard = mDirtyBits;
        ANGLE_TRY(mImpl->syncState(context, mDirtyBits));
        mDirtyBits.reset();
        mDirtyBitsGuard.reset();
    }
    return angle::Result::Continue;
}

void Framebuffer::onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message)
{
    if (message != angle::SubjectMessage::SubjectChanged)
    {
        // This can be triggered by SubImage calls for Textures.
        if (message == angle::SubjectMessage::ContentsChanged)
        {
            mDirtyBits.set(DIRTY_BIT_COLOR_BUFFER_CONTENTS_0 + index);
            onStateChange(angle::SubjectMessage::DirtyBitsFlagged);
            return;
        }

        // This can be triggered by the GL back-end TextureGL class.
        ASSERT(message == angle::SubjectMessage::DirtyBitsFlagged);
        return;
    }

    ASSERT(!mDirtyBitsGuard.valid() || mDirtyBitsGuard.value().test(index));
    mDirtyBits.set(index);

    invalidateCompletenessCache();

    FramebufferAttachment *attachment = getAttachmentFromSubjectIndex(index);

    // Mark the appropriate init flag.
    mState.mResourceNeedsInit.set(index, attachment->initState() == InitState::MayNeedInit);

    // Update mFloat32ColorAttachmentBits Cache
    if (index < DIRTY_BIT_COLOR_ATTACHMENT_MAX)
    {
        ASSERT(index != DIRTY_BIT_DEPTH_ATTACHMENT);
        ASSERT(index != DIRTY_BIT_STENCIL_ATTACHMENT);
        updateFloat32ColorAttachmentBits(index - DIRTY_BIT_COLOR_ATTACHMENT_0,
                                         attachment->getFormat().info);
    }
}

FramebufferAttachment *Framebuffer::getAttachmentFromSubjectIndex(angle::SubjectIndex index)
{
    switch (index)
    {
        case DIRTY_BIT_DEPTH_ATTACHMENT:
            return &mState.mDepthAttachment;
        case DIRTY_BIT_STENCIL_ATTACHMENT:
            return &mState.mStencilAttachment;
        default:
            size_t colorIndex = (index - DIRTY_BIT_COLOR_ATTACHMENT_0);
            ASSERT(colorIndex < mState.mColorAttachments.size());
            return &mState.mColorAttachments[colorIndex];
    }
}

bool Framebuffer::formsRenderingFeedbackLoopWith(const Context *context) const
{
    const State &state     = context->getState();
    const Program *program = state.getProgram();

    // TODO(jmadill): Default framebuffer feedback loops.
    if (mState.isDefault())
    {
        return false;
    }

    const FramebufferAttachment *depth   = getDepthAttachment();
    const FramebufferAttachment *stencil = getStencilAttachment();

    const bool checkDepth = depth && depth->type() == GL_TEXTURE;
    // Skip the feedback loop check for stencil if depth/stencil point to the same resource.
    const bool checkStencil =
        (stencil && stencil->type() == GL_TEXTURE) && (!depth || *stencil != *depth);

    const gl::ActiveTextureMask &activeTextures   = program->getActiveSamplersMask();
    const gl::ActiveTexturePointerArray &textures = state.getActiveTexturesCache();

    for (size_t textureUnit : activeTextures)
    {
        Texture *texture = textures[textureUnit];

        if (texture == nullptr)
        {
            continue;
        }

        // Depth and stencil attachment form feedback loops
        // Regardless of if enabled or masked.
        if (checkDepth)
        {
            if (texture->getId() == depth->id())
            {
                return true;
            }
        }

        if (checkStencil)
        {
            if (texture->getId() == stencil->id())
            {
                return true;
            }
        }

        // Check if any color attachment forms a feedback loop.
        for (size_t drawIndex : mColorAttachmentBits)
        {
            const FramebufferAttachment &attachment = mState.mColorAttachments[drawIndex];
            ASSERT(attachment.isAttached());

            if (attachment.isTextureWithId(texture->id()))
            {
                // TODO(jmadill): Check for appropriate overlap.
                return true;
            }
        }
    }

    return false;
}

bool Framebuffer::formsCopyingFeedbackLoopWith(TextureID copyTextureID,
                                               GLint copyTextureLevel,
                                               GLint copyTextureLayer) const
{
    if (mState.isDefault())
    {
        // It seems impossible to form a texture copying feedback loop with the default FBO.
        return false;
    }

    const FramebufferAttachment *readAttachment = getReadColorAttachment();
    ASSERT(readAttachment);

    if (readAttachment->isTextureWithId(copyTextureID))
    {
        const auto &imageIndex = readAttachment->getTextureImageIndex();
        if (imageIndex.getLevelIndex() == copyTextureLevel)
        {
            // Check 3D/Array texture layers.
            return !imageIndex.hasLayer() || copyTextureLayer == ImageIndex::kEntireLevel ||
                   imageIndex.getLayerIndex() == copyTextureLayer;
        }
    }
    return false;
}

GLint Framebuffer::getDefaultWidth() const
{
    return mState.getDefaultWidth();
}

GLint Framebuffer::getDefaultHeight() const
{
    return mState.getDefaultHeight();
}

GLint Framebuffer::getDefaultSamples() const
{
    return mState.getDefaultSamples();
}

bool Framebuffer::getDefaultFixedSampleLocations() const
{
    return mState.getDefaultFixedSampleLocations();
}

GLint Framebuffer::getDefaultLayers() const
{
    return mState.getDefaultLayers();
}

void Framebuffer::setDefaultWidth(const Context *context, GLint defaultWidth)
{
    mState.mDefaultWidth = defaultWidth;
    mDirtyBits.set(DIRTY_BIT_DEFAULT_WIDTH);
    invalidateCompletenessCache();
}

void Framebuffer::setDefaultHeight(const Context *context, GLint defaultHeight)
{
    mState.mDefaultHeight = defaultHeight;
    mDirtyBits.set(DIRTY_BIT_DEFAULT_HEIGHT);
    invalidateCompletenessCache();
}

void Framebuffer::setDefaultSamples(const Context *context, GLint defaultSamples)
{
    mState.mDefaultSamples = defaultSamples;
    mDirtyBits.set(DIRTY_BIT_DEFAULT_SAMPLES);
    invalidateCompletenessCache();
}

void Framebuffer::setDefaultFixedSampleLocations(const Context *context,
                                                 bool defaultFixedSampleLocations)
{
    mState.mDefaultFixedSampleLocations = defaultFixedSampleLocations;
    mDirtyBits.set(DIRTY_BIT_DEFAULT_FIXED_SAMPLE_LOCATIONS);
    invalidateCompletenessCache();
}

void Framebuffer::setDefaultLayers(GLint defaultLayers)
{
    mState.mDefaultLayers = defaultLayers;
    mDirtyBits.set(DIRTY_BIT_DEFAULT_LAYERS);
}

GLsizei Framebuffer::getNumViews() const
{
    return mState.getNumViews();
}

GLint Framebuffer::getBaseViewIndex() const
{
    return mState.getBaseViewIndex();
}

bool Framebuffer::isMultiview() const
{
    return mState.isMultiview();
}

bool Framebuffer::readDisallowedByMultiview() const
{
    return (mState.isMultiview() && mState.getNumViews() > 1);
}

angle::Result Framebuffer::ensureClearAttachmentsInitialized(const Context *context,
                                                             GLbitfield mask)
{
    const auto &glState = context->getState();
    if (!context->isRobustResourceInitEnabled() || glState.isRasterizerDiscardEnabled())
    {
        return angle::Result::Continue;
    }

    const BlendState &blend               = glState.getBlendState();
    const DepthStencilState &depthStencil = glState.getDepthStencilState();

    bool color   = (mask & GL_COLOR_BUFFER_BIT) != 0 && !IsColorMaskedOut(blend);
    bool depth   = (mask & GL_DEPTH_BUFFER_BIT) != 0 && !IsDepthMaskedOut(depthStencil);
    bool stencil = (mask & GL_STENCIL_BUFFER_BIT) != 0 && !IsStencilMaskedOut(depthStencil);

    if (!color && !depth && !stencil)
    {
        return angle::Result::Continue;
    }

    if (partialClearNeedsInit(context, color, depth, stencil))
    {
        ANGLE_TRY(ensureDrawAttachmentsInitialized(context));
    }

    // If the impl encounters an error during a a full (non-partial) clear, the attachments will
    // still be marked initialized. This simplifies design, allowing this method to be called before
    // the clear.
    markDrawAttachmentsInitialized(color, depth, stencil);

    return angle::Result::Continue;
}

angle::Result Framebuffer::ensureClearBufferAttachmentsInitialized(const Context *context,
                                                                   GLenum buffer,
                                                                   GLint drawbuffer)
{
    if (!context->isRobustResourceInitEnabled() ||
        context->getState().isRasterizerDiscardEnabled() || IsClearBufferMaskedOut(context, buffer))
    {
        return angle::Result::Continue;
    }

    if (partialBufferClearNeedsInit(context, buffer))
    {
        ANGLE_TRY(ensureBufferInitialized(context, buffer, drawbuffer));
    }

    // If the impl encounters an error during a a full (non-partial) clear, the attachments will
    // still be marked initialized. This simplifies design, allowing this method to be called before
    // the clear.
    markBufferInitialized(buffer, drawbuffer);

    return angle::Result::Continue;
}

angle::Result Framebuffer::ensureDrawAttachmentsInitialized(const Context *context)
{
    if (!context->isRobustResourceInitEnabled())
    {
        return angle::Result::Continue;
    }

    // Note: we don't actually filter by the draw attachment enum. Just init everything.
    for (size_t bit : mState.mResourceNeedsInit)
    {
        switch (bit)
        {
            case DIRTY_BIT_DEPTH_ATTACHMENT:
                ANGLE_TRY(InitAttachment(context, &mState.mDepthAttachment));
                break;
            case DIRTY_BIT_STENCIL_ATTACHMENT:
                ANGLE_TRY(InitAttachment(context, &mState.mStencilAttachment));
                break;
            default:
                ANGLE_TRY(InitAttachment(context, &mState.mColorAttachments[bit]));
                break;
        }
    }

    mState.mResourceNeedsInit.reset();
    return angle::Result::Continue;
}

angle::Result Framebuffer::ensureReadAttachmentsInitialized(const Context *context)
{
    ASSERT(context->isRobustResourceInitEnabled());

    if (mState.mResourceNeedsInit.none())
    {
        return angle::Result::Continue;
    }

    if (mState.mReadBufferState != GL_NONE)
    {
        if (isDefault())
        {
            if (!mState.mDefaultFramebufferReadAttachmentInitialized)
            {
                ANGLE_TRY(InitAttachment(context, &mState.mDefaultFramebufferReadAttachment));
                mState.mDefaultFramebufferReadAttachmentInitialized = true;
            }
        }
        else
        {
            size_t readIndex = mState.getReadIndex();
            if (mState.mResourceNeedsInit[readIndex])
            {
                ANGLE_TRY(InitAttachment(context, &mState.mColorAttachments[readIndex]));
                mState.mResourceNeedsInit.reset(readIndex);
            }
        }
    }

    // Conservatively init depth since it can be read by BlitFramebuffer.
    if (hasDepth())
    {
        if (mState.mResourceNeedsInit[DIRTY_BIT_DEPTH_ATTACHMENT])
        {
            ANGLE_TRY(InitAttachment(context, &mState.mDepthAttachment));
            mState.mResourceNeedsInit.reset(DIRTY_BIT_DEPTH_ATTACHMENT);
        }
    }

    // Conservatively init stencil since it can be read by BlitFramebuffer.
    if (hasStencil())
    {
        if (mState.mResourceNeedsInit[DIRTY_BIT_STENCIL_ATTACHMENT])
        {
            ANGLE_TRY(InitAttachment(context, &mState.mStencilAttachment));
            mState.mResourceNeedsInit.reset(DIRTY_BIT_STENCIL_ATTACHMENT);
        }
    }

    return angle::Result::Continue;
}

void Framebuffer::markDrawAttachmentsInitialized(bool color, bool depth, bool stencil)
{
    // Mark attachments as initialized.
    if (color)
    {
        for (auto colorIndex : mState.mEnabledDrawBuffers)
        {
            auto &colorAttachment = mState.mColorAttachments[colorIndex];
            ASSERT(colorAttachment.isAttached());
            colorAttachment.setInitState(InitState::Initialized);
            mState.mResourceNeedsInit.reset(colorIndex);
        }
    }

    if (depth && mState.mDepthAttachment.isAttached())
    {
        mState.mDepthAttachment.setInitState(InitState::Initialized);
        mState.mResourceNeedsInit.reset(DIRTY_BIT_DEPTH_ATTACHMENT);
    }

    if (stencil && mState.mStencilAttachment.isAttached())
    {
        mState.mStencilAttachment.setInitState(InitState::Initialized);
        mState.mResourceNeedsInit.reset(DIRTY_BIT_STENCIL_ATTACHMENT);
    }
}

void Framebuffer::markBufferInitialized(GLenum bufferType, GLint bufferIndex)
{
    switch (bufferType)
    {
        case GL_COLOR:
        {
            ASSERT(bufferIndex < static_cast<GLint>(mState.mColorAttachments.size()));
            if (mState.mColorAttachments[bufferIndex].isAttached())
            {
                mState.mColorAttachments[bufferIndex].setInitState(InitState::Initialized);
                mState.mResourceNeedsInit.reset(bufferIndex);
            }
            break;
        }
        case GL_DEPTH:
        {
            if (mState.mDepthAttachment.isAttached())
            {
                mState.mDepthAttachment.setInitState(InitState::Initialized);
                mState.mResourceNeedsInit.reset(DIRTY_BIT_DEPTH_ATTACHMENT);
            }
            break;
        }
        case GL_STENCIL:
        {
            if (mState.mStencilAttachment.isAttached())
            {
                mState.mStencilAttachment.setInitState(InitState::Initialized);
                mState.mResourceNeedsInit.reset(DIRTY_BIT_STENCIL_ATTACHMENT);
            }
            break;
        }
        case GL_DEPTH_STENCIL:
        {
            if (mState.mDepthAttachment.isAttached())
            {
                mState.mDepthAttachment.setInitState(InitState::Initialized);
                mState.mResourceNeedsInit.reset(DIRTY_BIT_DEPTH_ATTACHMENT);
            }
            if (mState.mStencilAttachment.isAttached())
            {
                mState.mStencilAttachment.setInitState(InitState::Initialized);
                mState.mResourceNeedsInit.reset(DIRTY_BIT_STENCIL_ATTACHMENT);
            }
            break;
        }
        default:
            UNREACHABLE();
            break;
    }
}

Box Framebuffer::getDimensions() const
{
    return mState.getDimensions();
}

Extents Framebuffer::getExtents() const
{
    return mState.getExtents();
}

angle::Result Framebuffer::ensureBufferInitialized(const Context *context,
                                                   GLenum bufferType,
                                                   GLint bufferIndex)
{
    ASSERT(context->isRobustResourceInitEnabled());

    if (mState.mResourceNeedsInit.none())
    {
        return angle::Result::Continue;
    }

    switch (bufferType)
    {
        case GL_COLOR:
        {
            ASSERT(bufferIndex < static_cast<GLint>(mState.mColorAttachments.size()));
            if (mState.mResourceNeedsInit[bufferIndex])
            {
                ANGLE_TRY(InitAttachment(context, &mState.mColorAttachments[bufferIndex]));
                mState.mResourceNeedsInit.reset(bufferIndex);
            }
            break;
        }
        case GL_DEPTH:
        {
            if (mState.mResourceNeedsInit[DIRTY_BIT_DEPTH_ATTACHMENT])
            {
                ANGLE_TRY(InitAttachment(context, &mState.mDepthAttachment));
                mState.mResourceNeedsInit.reset(DIRTY_BIT_DEPTH_ATTACHMENT);
            }
            break;
        }
        case GL_STENCIL:
        {
            if (mState.mResourceNeedsInit[DIRTY_BIT_STENCIL_ATTACHMENT])
            {
                ANGLE_TRY(InitAttachment(context, &mState.mStencilAttachment));
                mState.mResourceNeedsInit.reset(DIRTY_BIT_STENCIL_ATTACHMENT);
            }
            break;
        }
        case GL_DEPTH_STENCIL:
        {
            if (mState.mResourceNeedsInit[DIRTY_BIT_DEPTH_ATTACHMENT])
            {
                ANGLE_TRY(InitAttachment(context, &mState.mDepthAttachment));
                mState.mResourceNeedsInit.reset(DIRTY_BIT_DEPTH_ATTACHMENT);
            }
            if (mState.mResourceNeedsInit[DIRTY_BIT_STENCIL_ATTACHMENT])
            {
                ANGLE_TRY(InitAttachment(context, &mState.mStencilAttachment));
                mState.mResourceNeedsInit.reset(DIRTY_BIT_STENCIL_ATTACHMENT);
            }
            break;
        }
        default:
            UNREACHABLE();
            break;
    }

    return angle::Result::Continue;
}

bool Framebuffer::partialBufferClearNeedsInit(const Context *context, GLenum bufferType)
{
    if (!context->isRobustResourceInitEnabled() || mState.mResourceNeedsInit.none())
    {
        return false;
    }

    switch (bufferType)
    {
        case GL_COLOR:
            return partialClearNeedsInit(context, true, false, false);
        case GL_DEPTH:
            return partialClearNeedsInit(context, false, true, false);
        case GL_STENCIL:
            return partialClearNeedsInit(context, false, false, true);
        case GL_DEPTH_STENCIL:
            return partialClearNeedsInit(context, false, true, true);
        default:
            UNREACHABLE();
            return false;
    }
}
}  // namespace gl
