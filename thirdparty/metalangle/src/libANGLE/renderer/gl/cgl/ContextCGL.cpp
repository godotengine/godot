//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ContextCGL:
//   Mac-specific subclass of ContextGL.
//

#include "libANGLE/renderer/gl/cgl/ContextCGL.h"

#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/renderer/gl/cgl/DisplayCGL.h"

namespace rx
{

ContextCGL::ContextCGL(const gl::State &state,
                       gl::ErrorSet *errorSet,
                       const std::shared_ptr<RendererGL> &renderer,
                       bool usesDiscreteGPU)
    : ContextGL(state, errorSet, renderer), mUsesDiscreteGpu(usesDiscreteGPU)
{}

void ContextCGL::onDestroy(const gl::Context *context)
{
    if (mUsesDiscreteGpu)
    {
        egl::Display *display = context->getDisplay();
        // TODO(kbr): if the context is created and destroyed without ever
        // making it current, it is possible to leak retentions of the
        // discrete GPU.
        if (display)
        {
            GetImplAs<DisplayCGL>(display)->unreferenceDiscreteGPU();
        }
    }
}

}  // namespace rx
