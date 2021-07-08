//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// DeviceCGL.cpp: CGL implementation of egl::Device

#include "libANGLE/renderer/gl/cgl/DeviceCGL.h"

#include "libANGLE/renderer/gl/cgl/DisplayCGL.h"

#include <EGL/eglext.h>

namespace rx
{

DeviceCGL::DeviceCGL() {}

DeviceCGL::~DeviceCGL() {}

egl::Error DeviceCGL::initialize()
{
    return egl::NoError();
}

egl::Error DeviceCGL::getAttribute(const egl::Display *display, EGLint attribute, void **outValue)
{
    DisplayCGL *displayImpl = GetImplAs<DisplayCGL>(display);

    switch (attribute)
    {
        case EGL_CGL_CONTEXT_ANGLE:
            *outValue = displayImpl->getCGLContext();
            break;
        case EGL_CGL_PIXEL_FORMAT_ANGLE:
            *outValue = displayImpl->getCGLPixelFormat();
            break;
        default:
            return egl::EglBadAttribute();
    }

    return egl::NoError();
}

EGLint DeviceCGL::getType()
{
    return 0;
}

void DeviceCGL::generateExtensions(egl::DeviceExtensions *outExtensions) const
{
    outExtensions->deviceCGL = true;
}

}  // namespace rx
