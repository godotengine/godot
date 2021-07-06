//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Image.cpp: Implements the egl::Image class representing the EGLimage object.

#include "libANGLE/Image.h"

#include "common/debug.h"
#include "common/utilities.h"
#include "libANGLE/Context.h"
#include "libANGLE/Renderbuffer.h"
#include "libANGLE/Texture.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/EGLImplFactory.h"
#include "libANGLE/renderer/ImageImpl.h"

namespace egl
{

namespace
{
gl::ImageIndex GetImageIndex(EGLenum eglTarget, const egl::AttributeMap &attribs)
{
    if (!IsTextureTarget(eglTarget))
    {
        return gl::ImageIndex();
    }

    gl::TextureTarget target = egl_gl::EGLImageTargetToTextureTarget(eglTarget);
    GLint mip                = static_cast<GLint>(attribs.get(EGL_GL_TEXTURE_LEVEL_KHR, 0));
    GLint layer              = static_cast<GLint>(attribs.get(EGL_GL_TEXTURE_ZOFFSET_KHR, 0));

    if (target == gl::TextureTarget::_3D)
    {
        return gl::ImageIndex::Make3D(mip, layer);
    }
    else
    {
        ASSERT(layer == 0);
        return gl::ImageIndex::MakeFromTarget(target, mip, 1);
    }
}

const Display *DisplayFromContext(const gl::Context *context)
{
    return (context ? context->getDisplay() : nullptr);
}
}  // anonymous namespace

ImageSibling::ImageSibling() : FramebufferAttachmentObject(), mSourcesOf(), mTargetOf() {}

ImageSibling::~ImageSibling()
{
    // EGL images should hold a ref to their targets and siblings, a Texture should not be deletable
    // while it is attached to an EGL image.
    // Child class should orphan images before destruction.
    ASSERT(mSourcesOf.empty());
    ASSERT(mTargetOf.get() == nullptr);
}

void ImageSibling::setTargetImage(const gl::Context *context, egl::Image *imageTarget)
{
    ASSERT(imageTarget != nullptr);
    mTargetOf.set(DisplayFromContext(context), imageTarget);
    imageTarget->addTargetSibling(this);
}

angle::Result ImageSibling::orphanImages(const gl::Context *context)
{
    if (mTargetOf.get() != nullptr)
    {
        // Can't be a target and have sources.
        ASSERT(mSourcesOf.empty());

        ANGLE_TRY(mTargetOf->orphanSibling(context, this));
        mTargetOf.set(DisplayFromContext(context), nullptr);
    }
    else
    {
        for (Image *sourceImage : mSourcesOf)
        {
            ANGLE_TRY(sourceImage->orphanSibling(context, this));
        }
        mSourcesOf.clear();
    }

    return angle::Result::Continue;
}

void ImageSibling::addImageSource(egl::Image *imageSource)
{
    ASSERT(imageSource != nullptr);
    mSourcesOf.insert(imageSource);
}

void ImageSibling::removeImageSource(egl::Image *imageSource)
{
    ASSERT(mSourcesOf.find(imageSource) != mSourcesOf.end());
    mSourcesOf.erase(imageSource);
}

bool ImageSibling::isEGLImageTarget() const
{
    return (mTargetOf.get() != nullptr);
}

gl::InitState ImageSibling::sourceEGLImageInitState() const
{
    ASSERT(isEGLImageTarget());
    return mTargetOf->sourceInitState();
}

void ImageSibling::setSourceEGLImageInitState(gl::InitState initState) const
{
    ASSERT(isEGLImageTarget());
    mTargetOf->setInitState(initState);
}

bool ImageSibling::isRenderable(const gl::Context *context,
                                GLenum binding,
                                const gl::ImageIndex &imageIndex) const
{
    ASSERT(isEGLImageTarget());
    return mTargetOf->isRenderable(context);
}

void ImageSibling::notifySiblings(angle::SubjectMessage message)
{
    if (mTargetOf.get())
    {
        mTargetOf->notifySiblings(this, message);
    }
    for (Image *source : mSourcesOf)
    {
        source->notifySiblings(this, message);
    }
}

ExternalImageSibling::ExternalImageSibling(rx::EGLImplFactory *factory,
                                           const gl::Context *context,
                                           EGLenum target,
                                           EGLClientBuffer buffer,
                                           const AttributeMap &attribs)
    : mImplementation(factory->createExternalImageSibling(context, target, buffer, attribs))
{}

ExternalImageSibling::~ExternalImageSibling() = default;

void ExternalImageSibling::onDestroy(const egl::Display *display)
{
    mImplementation->onDestroy(display);
}

Error ExternalImageSibling::initialize(const egl::Display *display)
{
    return mImplementation->initialize(display);
}

gl::Extents ExternalImageSibling::getAttachmentSize(const gl::ImageIndex &imageIndex) const
{
    return mImplementation->getSize();
}

gl::Format ExternalImageSibling::getAttachmentFormat(GLenum binding,
                                                     const gl::ImageIndex &imageIndex) const
{
    return mImplementation->getFormat();
}

GLsizei ExternalImageSibling::getAttachmentSamples(const gl::ImageIndex &imageIndex) const
{
    return static_cast<GLsizei>(mImplementation->getSamples());
}

bool ExternalImageSibling::isRenderable(const gl::Context *context,
                                        GLenum binding,
                                        const gl::ImageIndex &imageIndex) const
{
    return mImplementation->isRenderable(context);
}

bool ExternalImageSibling::isTextureable(const gl::Context *context) const
{
    return mImplementation->isTexturable(context);
}

void ExternalImageSibling::onAttach(const gl::Context *context) {}

void ExternalImageSibling::onDetach(const gl::Context *context) {}

GLuint ExternalImageSibling::getId() const
{
    UNREACHABLE();
    return 0;
}

gl::InitState ExternalImageSibling::initState(const gl::ImageIndex &imageIndex) const
{
    return gl::InitState::Initialized;
}

void ExternalImageSibling::setInitState(const gl::ImageIndex &imageIndex, gl::InitState initState)
{}

rx::ExternalImageSiblingImpl *ExternalImageSibling::getImplementation() const
{
    return mImplementation.get();
}

rx::FramebufferAttachmentObjectImpl *ExternalImageSibling::getAttachmentImpl() const
{
    return mImplementation.get();
}

ImageState::ImageState(EGLenum target, ImageSibling *buffer, const AttributeMap &attribs)
    : label(nullptr),
      target(target),
      imageIndex(GetImageIndex(target, attribs)),
      source(buffer),
      targets(),
      format(GL_NONE),
      size(),
      samples(),
      sourceType(target)
{}

ImageState::~ImageState() {}

Image::Image(rx::EGLImplFactory *factory,
             const gl::Context *context,
             EGLenum target,
             ImageSibling *buffer,
             const AttributeMap &attribs)
    : mState(target, buffer, attribs),
      mImplementation(factory->createImage(mState, context, target, attribs)),
      mOrphanedAndNeedsInit(false)
{
    ASSERT(mImplementation != nullptr);
    ASSERT(buffer != nullptr);

    mState.source->addImageSource(this);
}

void Image::onDestroy(const Display *display)
{
    // All targets should hold a ref to the egl image and it should not be deleted until there are
    // no siblings left.
    ASSERT(mState.targets.empty());

    // Make sure the implementation gets a chance to clean up before we delete the source.
    mImplementation->onDestroy(display);

    // Tell the source that it is no longer used by this image
    if (mState.source != nullptr)
    {
        mState.source->removeImageSource(this);

        // If the source is an external object, delete it
        if (IsExternalImageTarget(mState.sourceType))
        {
            ExternalImageSibling *externalSibling = rx::GetAs<ExternalImageSibling>(mState.source);
            externalSibling->onDestroy(display);
            delete externalSibling;
        }

        mState.source = nullptr;
    }
}

Image::~Image()
{
    SafeDelete(mImplementation);
}

void Image::setLabel(EGLLabelKHR label)
{
    mState.label = label;
}

EGLLabelKHR Image::getLabel() const
{
    return mState.label;
}

void Image::addTargetSibling(ImageSibling *sibling)
{
    mState.targets.insert(sibling);
}

angle::Result Image::orphanSibling(const gl::Context *context, ImageSibling *sibling)
{
    ASSERT(sibling != nullptr);

    // notify impl
    ANGLE_TRY(mImplementation->orphan(context, sibling));

    if (mState.source == sibling)
    {
        // The external source of an image cannot be redefined so it cannot be orpahend.
        ASSERT(!IsExternalImageTarget(mState.sourceType));

        // If the sibling is the source, it cannot be a target.
        ASSERT(mState.targets.find(sibling) == mState.targets.end());
        mState.source = nullptr;
        mOrphanedAndNeedsInit =
            (sibling->initState(mState.imageIndex) == gl::InitState::MayNeedInit);
    }
    else
    {
        mState.targets.erase(sibling);
    }

    return angle::Result::Continue;
}

const gl::Format &Image::getFormat() const
{
    return mState.format;
}

bool Image::isRenderable(const gl::Context *context) const
{
    if (IsTextureTarget(mState.sourceType))
    {
        return mState.format.info->textureAttachmentSupport(context->getClientVersion(),
                                                            context->getExtensions());
    }
    else if (IsRenderbufferTarget(mState.sourceType))
    {
        return mState.format.info->renderbufferSupport(context->getClientVersion(),
                                                       context->getExtensions());
    }
    else if (IsExternalImageTarget(mState.sourceType))
    {
        ASSERT(mState.source != nullptr);
        return mState.source->isRenderable(context, GL_NONE, gl::ImageIndex());
    }

    UNREACHABLE();
    return false;
}

bool Image::isTexturable(const gl::Context *context) const
{
    if (IsTextureTarget(mState.sourceType))
    {
        return mState.format.info->textureSupport(context->getClientVersion(),
                                                  context->getExtensions());
    }
    else if (IsRenderbufferTarget(mState.sourceType))
    {
        return true;
    }
    else if (IsExternalImageTarget(mState.sourceType))
    {
        ASSERT(mState.source != nullptr);
        return rx::GetAs<ExternalImageSibling>(mState.source)->isTextureable(context);
    }

    UNREACHABLE();
    return false;
}

size_t Image::getWidth() const
{
    return mState.size.width;
}

size_t Image::getHeight() const
{
    return mState.size.height;
}

size_t Image::getSamples() const
{
    return mState.samples;
}

rx::ImageImpl *Image::getImplementation() const
{
    return mImplementation;
}

Error Image::initialize(const Display *display)
{
    if (IsExternalImageTarget(mState.sourceType))
    {
        ANGLE_TRY(rx::GetAs<ExternalImageSibling>(mState.source)->initialize(display));
    }

    mState.format  = mState.source->getAttachmentFormat(GL_NONE, mState.imageIndex);
    mState.size    = mState.source->getAttachmentSize(mState.imageIndex);
    mState.samples = mState.source->getAttachmentSamples(mState.imageIndex);

    return mImplementation->initialize(display);
}

bool Image::orphaned() const
{
    return (mState.source == nullptr);
}

gl::InitState Image::sourceInitState() const
{
    if (orphaned())
    {
        return mOrphanedAndNeedsInit ? gl::InitState::MayNeedInit : gl::InitState::Initialized;
    }

    return mState.source->initState(mState.imageIndex);
}

void Image::setInitState(gl::InitState initState)
{
    if (orphaned())
    {
        mOrphanedAndNeedsInit = false;
    }

    return mState.source->setInitState(mState.imageIndex, initState);
}

void Image::notifySiblings(const ImageSibling *notifier, angle::SubjectMessage message)
{
    if (mState.source && mState.source != notifier)
    {
        mState.source->onSubjectStateChange(rx::kTextureImageSiblingMessageIndex, message);
    }

    for (ImageSibling *target : mState.targets)
    {
        if (target != notifier)
        {
            target->onSubjectStateChange(rx::kTextureImageSiblingMessageIndex, message);
        }
    }
}

}  // namespace egl
