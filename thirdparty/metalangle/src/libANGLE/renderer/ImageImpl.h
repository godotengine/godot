//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ImageImpl.h: Defines the rx::ImageImpl class representing the EGLimage object.

#ifndef LIBANGLE_RENDERER_IMAGEIMPL_H_
#define LIBANGLE_RENDERER_IMAGEIMPL_H_

#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/FramebufferAttachmentObjectImpl.h"

namespace gl
{
class Context;
}  // namespace gl

namespace egl
{
class Display;
class ImageSibling;
struct ImageState;
}  // namespace egl

namespace rx
{
class ExternalImageSiblingImpl : public FramebufferAttachmentObjectImpl
{
  public:
    ~ExternalImageSiblingImpl() override {}

    virtual egl::Error initialize(const egl::Display *display) = 0;
    virtual void onDestroy(const egl::Display *display) {}

    virtual gl::Format getFormat() const                        = 0;
    virtual bool isRenderable(const gl::Context *context) const = 0;
    virtual bool isTexturable(const gl::Context *context) const = 0;
    virtual gl::Extents getSize() const                         = 0;
    virtual size_t getSamples() const                           = 0;
};

class ImageImpl : angle::NonCopyable
{
  public:
    ImageImpl(const egl::ImageState &state) : mState(state) {}
    virtual ~ImageImpl() {}
    virtual void onDestroy(const egl::Display *display) {}

    virtual egl::Error initialize(const egl::Display *display) = 0;

    virtual angle::Result orphan(const gl::Context *context, egl::ImageSibling *sibling) = 0;

  protected:
    const egl::ImageState &mState;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_IMAGEIMPL_H_
