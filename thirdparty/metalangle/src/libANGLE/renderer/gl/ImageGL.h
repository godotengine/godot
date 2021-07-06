//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ImageGL.h: Defines the rx::ImageGL class, the GL implementation of EGL images

#ifndef LIBANGLE_RENDERER_GL_IMAGEGL_H_
#define LIBANGLE_RENDERER_GL_IMAGEGL_H_

#include "common/PackedEnums.h"
#include "libANGLE/renderer/ImageImpl.h"

namespace rx
{
class TextureGL;
class RenderbufferGL;

class ImageGL : public ImageImpl
{
  public:
    ImageGL(const egl::ImageState &state);
    ~ImageGL() override;

    // TextureGL does not have access to all the parameters needed to implement
    // glEGLImageTargetTexture2DOES or glEGLImageTargetRenderbufferStorageOES. This allows the Image
    // to implement these functions because it holds the native EGLimage or emulated object.
    virtual angle::Result setTexture2D(const gl::Context *context,
                                       gl::TextureType type,
                                       TextureGL *texture,
                                       GLenum *outInternalFormat)           = 0;
    virtual angle::Result setRenderbufferStorage(const gl::Context *context,
                                                 RenderbufferGL *renderbuffer,
                                                 GLenum *outInternalFormat) = 0;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_IMAGEGL_H_
