//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// TextureImpl.cpp: Defines the abstract rx::TextureImpl classes.

#include "libANGLE/renderer/TextureImpl.h"

namespace rx
{
TextureImpl::TextureImpl(const gl::TextureState &state) : mState(state) {}

TextureImpl::~TextureImpl() {}

void TextureImpl::onDestroy(const gl::Context *context) {}

angle::Result TextureImpl::copyTexture(const gl::Context *context,
                                       const gl::ImageIndex &index,
                                       GLenum internalFormat,
                                       GLenum type,
                                       size_t sourceLevel,
                                       bool unpackFlipY,
                                       bool unpackPremultiplyAlpha,
                                       bool unpackUnmultiplyAlpha,
                                       const gl::Texture *source)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result TextureImpl::copySubTexture(const gl::Context *context,
                                          const gl::ImageIndex &index,
                                          const gl::Offset &destOffset,
                                          size_t sourceLevel,
                                          const gl::Box &sourceBox,
                                          bool unpackFlipY,
                                          bool unpackPremultiplyAlpha,
                                          bool unpackUnmultiplyAlpha,
                                          const gl::Texture *source)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result TextureImpl::copyCompressedTexture(const gl::Context *context,
                                                 const gl::Texture *source)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result TextureImpl::copy3DTexture(const gl::Context *context,
                                         gl::TextureTarget target,
                                         GLenum internalFormat,
                                         GLenum type,
                                         size_t sourceLevel,
                                         size_t destLevel,
                                         bool unpackFlipY,
                                         bool unpackPremultiplyAlpha,
                                         bool unpackUnmultiplyAlpha,
                                         const gl::Texture *source)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result TextureImpl::copy3DSubTexture(const gl::Context *context,
                                            const gl::TextureTarget target,
                                            const gl::Offset &destOffset,
                                            size_t sourceLevel,
                                            size_t destLevel,
                                            const gl::Box &srcBox,
                                            bool unpackFlipY,
                                            bool unpackPremultiplyAlpha,
                                            bool unpackUnmultiplyAlpha,
                                            const gl::Texture *source)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

angle::Result TextureImpl::setImageExternal(const gl::Context *context,
                                            const gl::ImageIndex &index,
                                            GLenum internalFormat,
                                            const gl::Extents &size,
                                            GLenum format,
                                            GLenum type)
{
    UNREACHABLE();
    return angle::Result::Stop;
}

GLint TextureImpl::getMemorySize() const
{
    return 0;
}

GLint TextureImpl::getLevelMemorySize(gl::TextureTarget target, GLint level)
{
    return 0;
}

GLint TextureImpl::getNativeID() const
{
    UNREACHABLE();
    return 0;
}

}  // namespace rx
