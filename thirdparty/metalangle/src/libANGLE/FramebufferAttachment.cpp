//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FramebufferAttachment.cpp: the gl::FramebufferAttachment class and its derived classes
// objects and related functionality. [OpenGL ES 2.0.24] section 4.4.3 page 108.

#include "libANGLE/FramebufferAttachment.h"

#include "common/utilities.h"
#include "libANGLE/Config.h"
#include "libANGLE/Context.h"
#include "libANGLE/Renderbuffer.h"
#include "libANGLE/Surface.h"
#include "libANGLE/Texture.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/FramebufferAttachmentObjectImpl.h"
#include "libANGLE/renderer/FramebufferImpl.h"

namespace gl
{

////// FramebufferAttachment::Target Implementation //////

const GLsizei FramebufferAttachment::kDefaultNumViews             = 1;
const GLint FramebufferAttachment::kDefaultBaseViewIndex          = 0;
const GLint FramebufferAttachment::kDefaultRenderToTextureSamples = 0;

FramebufferAttachment::Target::Target() : mBinding(GL_NONE), mTextureIndex() {}

FramebufferAttachment::Target::Target(GLenum binding, const ImageIndex &imageIndex)
    : mBinding(binding), mTextureIndex(imageIndex)
{}

FramebufferAttachment::Target::Target(const Target &other)
    : mBinding(other.mBinding), mTextureIndex(other.mTextureIndex)
{}

FramebufferAttachment::Target &FramebufferAttachment::Target::operator=(const Target &other)
{
    this->mBinding      = other.mBinding;
    this->mTextureIndex = other.mTextureIndex;
    return *this;
}

////// FramebufferAttachment Implementation //////

FramebufferAttachment::FramebufferAttachment()
    : mType(GL_NONE),
      mResource(nullptr),
      mNumViews(kDefaultNumViews),
      mIsMultiview(false),
      mBaseViewIndex(kDefaultBaseViewIndex),
      mRenderToTextureSamples(kDefaultRenderToTextureSamples)
{}

FramebufferAttachment::FramebufferAttachment(const Context *context,
                                             GLenum type,
                                             GLenum binding,
                                             const ImageIndex &textureIndex,
                                             FramebufferAttachmentObject *resource)
    : mResource(nullptr)
{
    attach(context, type, binding, textureIndex, resource, kDefaultNumViews, kDefaultBaseViewIndex,
           false, kDefaultRenderToTextureSamples);
}

FramebufferAttachment::FramebufferAttachment(FramebufferAttachment &&other)
    : FramebufferAttachment()
{
    *this = std::move(other);
}

FramebufferAttachment &FramebufferAttachment::operator=(FramebufferAttachment &&other)
{
    std::swap(mType, other.mType);
    std::swap(mTarget, other.mTarget);
    std::swap(mResource, other.mResource);
    std::swap(mNumViews, other.mNumViews);
    std::swap(mIsMultiview, other.mIsMultiview);
    std::swap(mBaseViewIndex, other.mBaseViewIndex);
    std::swap(mRenderToTextureSamples, other.mRenderToTextureSamples);
    return *this;
}

FramebufferAttachment::~FramebufferAttachment()
{
    ASSERT(!isAttached());
}

void FramebufferAttachment::detach(const Context *context)
{
    mType = GL_NONE;
    if (mResource != nullptr)
    {
        mResource->onDetach(context);
        mResource = nullptr;
    }
    mNumViews      = kDefaultNumViews;
    mIsMultiview   = false;
    mBaseViewIndex = kDefaultBaseViewIndex;

    // not technically necessary, could omit for performance
    mTarget = Target();
}

void FramebufferAttachment::attach(const Context *context,
                                   GLenum type,
                                   GLenum binding,
                                   const ImageIndex &textureIndex,
                                   FramebufferAttachmentObject *resource,
                                   GLsizei numViews,
                                   GLuint baseViewIndex,
                                   bool isMultiview,
                                   GLsizei samples)
{
    if (resource == nullptr)
    {
        detach(context);
        return;
    }

    mType                   = type;
    mTarget                 = Target(binding, textureIndex);
    mNumViews               = numViews;
    mBaseViewIndex          = baseViewIndex;
    mIsMultiview            = isMultiview;
    mRenderToTextureSamples = samples;
    resource->onAttach(context);

    if (mResource != nullptr)
    {
        mResource->onDetach(context);
    }

    mResource = resource;
}

GLuint FramebufferAttachment::getRedSize() const
{
    return getSize().empty() ? 0 : getFormat().info->redBits;
}

GLuint FramebufferAttachment::getGreenSize() const
{
    return getSize().empty() ? 0 : getFormat().info->greenBits;
}

GLuint FramebufferAttachment::getBlueSize() const
{
    return getSize().empty() ? 0 : getFormat().info->blueBits;
}

GLuint FramebufferAttachment::getAlphaSize() const
{
    return getSize().empty() ? 0 : getFormat().info->alphaBits;
}

GLuint FramebufferAttachment::getDepthSize() const
{
    return getSize().empty() ? 0 : getFormat().info->depthBits;
}

GLuint FramebufferAttachment::getStencilSize() const
{
    return getSize().empty() ? 0 : getFormat().info->stencilBits;
}

GLenum FramebufferAttachment::getComponentType() const
{
    return getFormat().info->componentType;
}

GLenum FramebufferAttachment::getColorEncoding() const
{
    return getFormat().info->colorEncoding;
}

GLuint FramebufferAttachment::id() const
{
    return mResource->getId();
}

TextureTarget FramebufferAttachment::cubeMapFace() const
{
    ASSERT(mType == GL_TEXTURE);

    const auto &index = mTarget.textureIndex();
    return index.getType() == TextureType::CubeMap ? index.getTarget() : TextureTarget::InvalidEnum;
}

GLint FramebufferAttachment::mipLevel() const
{
    ASSERT(type() == GL_TEXTURE);
    return mTarget.textureIndex().getLevelIndex();
}

GLint FramebufferAttachment::layer() const
{
    ASSERT(mType == GL_TEXTURE);

    const gl::ImageIndex &index = mTarget.textureIndex();
    return (index.has3DLayer() ? index.getLayerIndex() : 0);
}

bool FramebufferAttachment::isLayered() const
{
    return mTarget.textureIndex().isLayered();
}

bool FramebufferAttachment::isMultiview() const
{
    return mIsMultiview;
}

GLint FramebufferAttachment::getBaseViewIndex() const
{
    return mBaseViewIndex;
}

Texture *FramebufferAttachment::getTexture() const
{
    return rx::GetAs<Texture>(mResource);
}

Renderbuffer *FramebufferAttachment::getRenderbuffer() const
{
    return rx::GetAs<Renderbuffer>(mResource);
}

const egl::Surface *FramebufferAttachment::getSurface() const
{
    return rx::GetAs<egl::Surface>(mResource);
}

FramebufferAttachmentObject *FramebufferAttachment::getResource() const
{
    return mResource;
}

bool FramebufferAttachment::operator==(const FramebufferAttachment &other) const
{
    if (mResource != other.mResource || mType != other.mType || mNumViews != other.mNumViews ||
        mIsMultiview != other.mIsMultiview || mBaseViewIndex != other.mBaseViewIndex ||
        mRenderToTextureSamples != other.mRenderToTextureSamples)
    {
        return false;
    }

    if (mType == GL_TEXTURE && getTextureImageIndex() != other.getTextureImageIndex())
    {
        return false;
    }

    return true;
}

bool FramebufferAttachment::operator!=(const FramebufferAttachment &other) const
{
    return !(*this == other);
}

InitState FramebufferAttachment::initState() const
{
    return mResource ? mResource->initState(mTarget.textureIndex()) : InitState::Initialized;
}

angle::Result FramebufferAttachment::initializeContents(const Context *context)
{
    ASSERT(mResource);
    ANGLE_TRY(mResource->initializeContents(context, mTarget.textureIndex()));
    setInitState(InitState::Initialized);
    return angle::Result::Continue;
}

void FramebufferAttachment::setInitState(InitState initState) const
{
    ASSERT(mResource);
    mResource->setInitState(mTarget.textureIndex(), initState);
}

////// FramebufferAttachmentObject Implementation //////

FramebufferAttachmentObject::FramebufferAttachmentObject() {}

FramebufferAttachmentObject::~FramebufferAttachmentObject() {}

angle::Result FramebufferAttachmentObject::getAttachmentRenderTarget(
    const Context *context,
    GLenum binding,
    const ImageIndex &imageIndex,
    GLsizei samples,
    rx::FramebufferAttachmentRenderTarget **rtOut) const
{
    return getAttachmentImpl()->getAttachmentRenderTarget(context, binding, imageIndex, samples,
                                                          rtOut);
}

angle::Result FramebufferAttachmentObject::initializeContents(const Context *context,
                                                              const ImageIndex &imageIndex)
{
    ASSERT(context->isRobustResourceInitEnabled());

    // Because gl::Texture cannot support tracking individual layer dirtiness, we only handle
    // initializing entire mip levels for 2D array textures.
    if (imageIndex.getType() == TextureType::_2DArray && imageIndex.hasLayer())
    {
        ImageIndex fullMipIndex =
            ImageIndex::Make2DArray(imageIndex.getLevelIndex(), ImageIndex::kEntireLevel);
        return getAttachmentImpl()->initializeContents(context, fullMipIndex);
    }
    else if (imageIndex.getType() == TextureType::_2DMultisampleArray && imageIndex.hasLayer())
    {
        ImageIndex fullMipIndex = ImageIndex::Make2DMultisampleArray(ImageIndex::kEntireLevel);
        return getAttachmentImpl()->initializeContents(context, fullMipIndex);
    }
    else
    {
        return getAttachmentImpl()->initializeContents(context, imageIndex);
    }
}

}  // namespace gl
