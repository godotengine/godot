//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RendererCGL.h: implements createWorkerContext for RendererGL.

#ifndef LIBANGLE_RENDERER_GL_CGL_RENDERERCGL_H_
#define LIBANGLE_RENDERER_GL_CGL_RENDERERCGL_H_

#include "libANGLE/renderer/gl/RendererGL.h"

namespace rx
{

class DisplayCGL;

class RendererCGL : public RendererGL
{
  public:
    RendererCGL(std::unique_ptr<FunctionsGL> functions,
                const egl::AttributeMap &attribMap,
                DisplayCGL *display);
    ~RendererCGL() override;

  private:
    WorkerContext *createWorkerContext(std::string *infoLog) override;

    DisplayCGL *mDisplay;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_GLX_RENDERERGLX_H_
