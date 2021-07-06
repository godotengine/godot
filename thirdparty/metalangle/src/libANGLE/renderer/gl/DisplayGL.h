//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// DisplayGL.h: Defines the class interface for DisplayGL.

#ifndef LIBANGLE_RENDERER_GL_DISPLAYGL_H_
#define LIBANGLE_RENDERER_GL_DISPLAYGL_H_

#include "libANGLE/renderer/DisplayImpl.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"

namespace egl
{
class Surface;
}

namespace rx
{

class RendererGL;

class DisplayGL : public DisplayImpl
{
  public:
    DisplayGL(const egl::DisplayState &state);
    ~DisplayGL() override;

    egl::Error initialize(egl::Display *display) override;
    void terminate() override;

    ImageImpl *createImage(const egl::ImageState &state,
                           const gl::Context *context,
                           EGLenum target,
                           const egl::AttributeMap &attribs) override;

    StreamProducerImpl *createStreamProducerD3DTexture(egl::Stream::ConsumerType consumerType,
                                                       const egl::AttributeMap &attribs) override;

    egl::Error makeCurrent(egl::Surface *drawSurface,
                           egl::Surface *readSurface,
                           gl::Context *context) override;

    gl::Version getMaxConformantESVersion() const override;

  protected:
    void generateExtensions(egl::DisplayExtensions *outExtensions) const override;

  private:
    virtual egl::Error makeCurrentSurfaceless(gl::Context *context);
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_DISPLAYGL_H_
