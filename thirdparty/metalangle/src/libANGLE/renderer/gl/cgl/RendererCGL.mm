//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RendererCGL.mm: Implements the class methods for RendererCGL.

#include "libANGLE/renderer/gl/cgl/RendererCGL.h"

#include "libANGLE/renderer/gl/cgl/DisplayCGL.h"

namespace rx
{

RendererCGL::RendererCGL(std::unique_ptr<FunctionsGL> functions,
                         const egl::AttributeMap &attribMap,
                         DisplayCGL *display)
    : RendererGL(std::move(functions), attribMap, display), mDisplay(display)
{}

RendererCGL::~RendererCGL() {}

WorkerContext *RendererCGL::createWorkerContext(std::string *infoLog)
{
    return mDisplay->createWorkerContext(infoLog);
}

}  // namespace rx
