//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ImageMtl.cpp:
//    Implements the class methods for ImageMtl.
//

#include "libANGLE/renderer/metal/ImageMtl.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"
#include "libANGLE/renderer/metal/RenderBufferMtl.h"
#include "libANGLE/renderer/metal/TextureMtl.h"

namespace rx
{

// TextureImageSiblingMtl implementation
TextureImageSiblingMtl::TextureImageSiblingMtl(EGLClientBuffer buffer)
    : mBuffer(buffer), mGLFormat(GL_NONE)
{}

TextureImageSiblingMtl::~TextureImageSiblingMtl() {}

// Static
bool TextureImageSiblingMtl::ValidateClientBuffer(const DisplayMtl *display, EGLClientBuffer buffer)
{
    id<MTLTexture> texture = (__bridge id<MTLTexture>)(buffer);
    if (!texture || texture.device != display->getMetalDevice())
    {
        return false;
    }

    if (texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureTypeCube)
    {
        return false;
    }

    angle::FormatID angleFormatId = mtl::Format::MetalToAngleFormatID(texture.pixelFormat);
    const mtl::Format &format     = display->getPixelFormat(angleFormatId);
    if (!format.valid())
    {
        ERR() << "Unrecognized format";
        // Not supported
        return false;
    }

    return true;
}

egl::Error TextureImageSiblingMtl::initialize(const egl::Display *display)
{
    DisplayMtl *displayMtl = mtl::GetImpl(display);
    if (initImpl(displayMtl) != angle::Result::Continue)
    {
        return egl::EglBadParameter();
    }

    return egl::NoError();
}

angle::Result TextureImageSiblingMtl::initImpl(DisplayMtl *displayMtl)
{
    mNativeTexture = mtl::Texture::MakeFromMetal((__bridge id<MTLTexture>)(mBuffer));

    angle::FormatID angleFormatId =
        mtl::Format::MetalToAngleFormatID(mNativeTexture->pixelFormat());
    mFormat = displayMtl->getPixelFormat(angleFormatId);

    mGLFormat = gl::Format(mFormat.intendedAngleFormat().glInternalFormat);

    mRenderable = mFormat.getCaps().depthRenderable || mFormat.getCaps().colorRenderable;

    mTextureable = mFormat.getCaps().filterable || mFormat.hasDepthOrStencilBits();

    return angle::Result::Continue;
}

void TextureImageSiblingMtl::onDestroy(const egl::Display *display)
{
    mNativeTexture = nullptr;
}

gl::Format TextureImageSiblingMtl::getFormat() const
{
    return mGLFormat;
}

bool TextureImageSiblingMtl::isRenderable(const gl::Context *context) const
{
    return mRenderable;
}

bool TextureImageSiblingMtl::isTexturable(const gl::Context *context) const
{
    return mTextureable;
}

gl::Extents TextureImageSiblingMtl::getSize() const
{
    return mNativeTexture ? mNativeTexture->size() : gl::Extents(0, 0, 0);
}

size_t TextureImageSiblingMtl::getSamples() const
{
    uint32_t samples = mNativeTexture ? mNativeTexture->samples() : 0;
    return samples > 1 ? samples : 0;
}

// ImageMtl implementation
ImageMtl::ImageMtl(const egl::ImageState &state, const gl::Context *context)
    : ImageImpl(state)
{}

ImageMtl::~ImageMtl() {}

void ImageMtl::onDestroy(const egl::Display *display)
{
    mNativeTexture = nullptr;
}

egl::Error ImageMtl::initialize(const egl::Display *display)
{
    if (mState.target == EGL_MTL_TEXTURE_MGL)
    {
        const TextureImageSiblingMtl *externalImageSibling =
            GetImplAs<TextureImageSiblingMtl>(GetAs<egl::ExternalImageSibling>(mState.source));

        mNativeTexture = externalImageSibling->getTexture();

        switch (mNativeTexture->textureType())
        {
            case MTLTextureType2D:
                mImageTextureType = gl::TextureType::_2D;
                break;
            case MTLTextureTypeCube:
                mImageTextureType = gl::TextureType::CubeMap;
                break;
            default:
                UNREACHABLE();
        }

        mImageLevel = 0;
        mImageLayer = 0;
    }
    else
    {
        UNREACHABLE();
        return egl::EglBadAccess();
    }

    return egl::NoError();
}

angle::Result ImageMtl::orphan(const gl::Context *context, egl::ImageSibling *sibling)
{
    if (sibling == mState.source)
    {
        mNativeTexture = nullptr;
    }

    return angle::Result::Continue;
}

}  // namespace rx
