//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Renderbuffer.cpp: Implements the renderer-agnostic gl::Renderbuffer class,
// GL renderbuffer objects and related functionality.
// [OpenGL ES 2.0.24] section 4.4.3 page 108.

#include "libANGLE/Renderbuffer.h"

#include "common/utilities.h"
#include "libANGLE/Context.h"
#include "libANGLE/FramebufferAttachment.h"
#include "libANGLE/Image.h"
#include "libANGLE/Renderbuffer.h"
#include "libANGLE/Texture.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/d3d/RenderTargetD3D.h"

namespace gl
{
// RenderbufferState implementation.
RenderbufferState::RenderbufferState()
    : mWidth(0), mHeight(0), mFormat(GL_RGBA4), mSamples(0), mInitState(InitState::MayNeedInit)
{}

RenderbufferState::~RenderbufferState() {}

GLsizei RenderbufferState::getWidth() const
{
    return mWidth;
}

GLsizei RenderbufferState::getHeight() const
{
    return mHeight;
}

const Format &RenderbufferState::getFormat() const
{
    return mFormat;
}

GLsizei RenderbufferState::getSamples() const
{
    return mSamples;
}

void RenderbufferState::update(GLsizei width,
                               GLsizei height,
                               const Format &format,
                               GLsizei samples,
                               InitState initState)
{
    mWidth     = static_cast<GLsizei>(width);
    mHeight    = static_cast<GLsizei>(height);
    mFormat    = format;
    mSamples   = samples;
    mInitState = InitState::MayNeedInit;
}

// Renderbuffer implementation.
Renderbuffer::Renderbuffer(rx::GLImplFactory *implFactory, RenderbufferID id)
    : RefCountObject(id),
      mState(),
      mImplementation(implFactory->createRenderbuffer(mState)),
      mLabel()
{}

void Renderbuffer::onDestroy(const Context *context)
{
    (void)(orphanImages(context));

    if (mImplementation)
    {
        mImplementation->onDestroy(context);
    }
}

Renderbuffer::~Renderbuffer() {}

void Renderbuffer::setLabel(const Context *context, const std::string &label)
{
    mLabel = label;
}

const std::string &Renderbuffer::getLabel() const
{
    return mLabel;
}

angle::Result Renderbuffer::setStorage(const Context *context,
                                       GLenum internalformat,
                                       size_t width,
                                       size_t height)
{
    ANGLE_TRY(orphanImages(context));
    ANGLE_TRY(mImplementation->setStorage(context, internalformat, width, height));

    mState.update(static_cast<GLsizei>(width), static_cast<GLsizei>(height), Format(internalformat),
                  0, InitState::MayNeedInit);
    onStateChange(angle::SubjectMessage::SubjectChanged);

    return angle::Result::Continue;
}

angle::Result Renderbuffer::setStorageMultisample(const Context *context,
                                                  size_t samples,
                                                  GLenum internalformat,
                                                  size_t width,
                                                  size_t height)
{
    ANGLE_TRY(orphanImages(context));
    ANGLE_TRY(
        mImplementation->setStorageMultisample(context, samples, internalformat, width, height));

    mState.update(static_cast<GLsizei>(width), static_cast<GLsizei>(height), Format(internalformat),
                  static_cast<GLsizei>(samples), InitState::MayNeedInit);
    onStateChange(angle::SubjectMessage::SubjectChanged);

    return angle::Result::Continue;
}

angle::Result Renderbuffer::setStorageEGLImageTarget(const Context *context, egl::Image *image)
{
    ANGLE_TRY(orphanImages(context));
    ANGLE_TRY(mImplementation->setStorageEGLImageTarget(context, image));

    setTargetImage(context, image);

    mState.update(static_cast<GLsizei>(image->getWidth()), static_cast<GLsizei>(image->getHeight()),
                  Format(image->getFormat()), 0, image->sourceInitState());
    onStateChange(angle::SubjectMessage::SubjectChanged);

    return angle::Result::Continue;
}

rx::RenderbufferImpl *Renderbuffer::getImplementation() const
{
    ASSERT(mImplementation);
    return mImplementation.get();
}

GLsizei Renderbuffer::getWidth() const
{
    return mState.mWidth;
}

GLsizei Renderbuffer::getHeight() const
{
    return mState.mHeight;
}

const Format &Renderbuffer::getFormat() const
{
    return mState.mFormat;
}

GLsizei Renderbuffer::getSamples() const
{
    return mState.mSamples;
}

GLuint Renderbuffer::getRedSize() const
{
    return mState.mFormat.info->redBits;
}

GLuint Renderbuffer::getGreenSize() const
{
    return mState.mFormat.info->greenBits;
}

GLuint Renderbuffer::getBlueSize() const
{
    return mState.mFormat.info->blueBits;
}

GLuint Renderbuffer::getAlphaSize() const
{
    return mState.mFormat.info->alphaBits;
}

GLuint Renderbuffer::getDepthSize() const
{
    return mState.mFormat.info->depthBits;
}

GLuint Renderbuffer::getStencilSize() const
{
    return mState.mFormat.info->stencilBits;
}

GLint Renderbuffer::getMemorySize() const
{
    GLint implSize = mImplementation->getMemorySize();
    if (implSize > 0)
    {
        return implSize;
    }

    // Assume allocated size is around width * height * samples * pixelBytes
    angle::CheckedNumeric<GLint> size = 1;
    size *= mState.mFormat.info->pixelBytes;
    size *= mState.mWidth;
    size *= mState.mHeight;
    size *= std::max(mState.mSamples, 1);
    return size.ValueOrDefault(std::numeric_limits<GLint>::max());
}

void Renderbuffer::onAttach(const Context *context)
{
    addRef();
}

void Renderbuffer::onDetach(const Context *context)
{
    release(context);
}

GLuint Renderbuffer::getId() const
{
    return id().value;
}

Extents Renderbuffer::getAttachmentSize(const gl::ImageIndex & /*imageIndex*/) const
{
    return Extents(mState.mWidth, mState.mHeight, 1);
}

Format Renderbuffer::getAttachmentFormat(GLenum /*binding*/,
                                         const ImageIndex & /*imageIndex*/) const
{
    return getFormat();
}
GLsizei Renderbuffer::getAttachmentSamples(const ImageIndex & /*imageIndex*/) const
{
    return getSamples();
}

bool Renderbuffer::isRenderable(const Context *context,
                                GLenum binding,
                                const ImageIndex &imageIndex) const
{
    if (isEGLImageTarget())
    {
        return ImageSibling::isRenderable(context, binding, imageIndex);
    }
    return getFormat().info->renderbufferSupport(context->getClientVersion(),
                                                 context->getExtensions());
}

InitState Renderbuffer::initState(const gl::ImageIndex & /*imageIndex*/) const
{
    if (isEGLImageTarget())
    {
        return sourceEGLImageInitState();
    }

    return mState.mInitState;
}

void Renderbuffer::setInitState(const gl::ImageIndex & /*imageIndex*/, InitState initState)
{
    if (isEGLImageTarget())
    {
        setSourceEGLImageInitState(initState);
    }
    else
    {
        mState.mInitState = initState;
    }
}

rx::FramebufferAttachmentObjectImpl *Renderbuffer::getAttachmentImpl() const
{
    return mImplementation.get();
}

}  // namespace gl
