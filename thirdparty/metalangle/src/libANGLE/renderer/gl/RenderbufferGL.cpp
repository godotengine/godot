//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RenderbufferGL.cpp: Implements the class methods for RenderbufferGL.

#include "libANGLE/renderer/gl/RenderbufferGL.h"

#include "common/debug.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Context.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/gl/BlitGL.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/ImageGL.h"
#include "libANGLE/renderer/gl/StateManagerGL.h"
#include "libANGLE/renderer/gl/formatutilsgl.h"
#include "libANGLE/renderer/gl/renderergl_utils.h"
#include "platform/FeaturesGL.h"

namespace rx
{
RenderbufferGL::RenderbufferGL(const gl::RenderbufferState &state, GLuint id)
    : RenderbufferImpl(state), mRenderbufferID(id)
{
}

RenderbufferGL::~RenderbufferGL()
{
    ASSERT(mRenderbufferID == 0);
}

void RenderbufferGL::onDestroy(const gl::Context *context)
{
    StateManagerGL *stateManager = GetStateManagerGL(context);
    stateManager->deleteRenderbuffer(mRenderbufferID);
    mRenderbufferID = 0;
}

angle::Result RenderbufferGL::setStorage(const gl::Context *context,
                                         GLenum internalformat,
                                         size_t width,
                                         size_t height)
{
    const FunctionsGL *functions      = GetFunctionsGL(context);
    StateManagerGL *stateManager      = GetStateManagerGL(context);
    const angle::FeaturesGL &features = GetFeaturesGL(context);

    stateManager->bindRenderbuffer(GL_RENDERBUFFER, mRenderbufferID);

    nativegl::RenderbufferFormat renderbufferFormat =
        nativegl::GetRenderbufferFormat(functions, features, internalformat);
    ANGLE_GL_TRY_ALWAYS_CHECK(
        context,
        functions->renderbufferStorage(GL_RENDERBUFFER, renderbufferFormat.internalFormat,
                                       static_cast<GLsizei>(width), static_cast<GLsizei>(height)));

    mNativeInternalFormat = renderbufferFormat.internalFormat;

    return angle::Result::Continue;
}

angle::Result RenderbufferGL::setStorageMultisample(const gl::Context *context,
                                                    size_t samples,
                                                    GLenum internalformat,
                                                    size_t width,
                                                    size_t height)
{
    const FunctionsGL *functions      = GetFunctionsGL(context);
    StateManagerGL *stateManager      = GetStateManagerGL(context);
    const angle::FeaturesGL &features = GetFeaturesGL(context);

    stateManager->bindRenderbuffer(GL_RENDERBUFFER, mRenderbufferID);

    nativegl::RenderbufferFormat renderbufferFormat =
        nativegl::GetRenderbufferFormat(functions, features, internalformat);
    ANGLE_GL_TRY_ALWAYS_CHECK(
        context,
        functions->renderbufferStorageMultisample(
            GL_RENDERBUFFER, static_cast<GLsizei>(samples), renderbufferFormat.internalFormat,
            static_cast<GLsizei>(width), static_cast<GLsizei>(height)));

    mNativeInternalFormat = renderbufferFormat.internalFormat;

    return angle::Result::Continue;
}

angle::Result RenderbufferGL::setStorageEGLImageTarget(const gl::Context *context,
                                                       egl::Image *image)
{
    ImageGL *imageGL = GetImplAs<ImageGL>(image);
    return imageGL->setRenderbufferStorage(context, this, &mNativeInternalFormat);
}

GLuint RenderbufferGL::getRenderbufferID() const
{
    return mRenderbufferID;
}

angle::Result RenderbufferGL::initializeContents(const gl::Context *context,
                                                 const gl::ImageIndex &imageIndex)
{
    BlitGL *blitter = GetBlitGL(context);
    return blitter->clearRenderbuffer(context, this, mNativeInternalFormat);
}

GLenum RenderbufferGL::getNativeInternalFormat() const
{
    return mNativeInternalFormat;
}

}  // namespace rx
