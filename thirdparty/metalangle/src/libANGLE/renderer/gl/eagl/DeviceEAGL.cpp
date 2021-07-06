/*
 * Copyright (C) 2019 Apple Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY APPLE INC. ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL APPLE INC. OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// DeviceEAGL.cpp: EAGL implementation of egl::Device

#import "common/platform.h"

#if defined(ANGLE_PLATFORM_IOS)

#    include "libANGLE/renderer/gl/eagl/DeviceEAGL.h"

#    include "libANGLE/renderer/gl/eagl/DisplayEAGL.h"

#    include <EGL/eglext.h>

namespace rx
{

DeviceEAGL::DeviceEAGL() {}

DeviceEAGL::~DeviceEAGL() {}

egl::Error DeviceEAGL::initialize()
{
    return egl::NoError();
}

egl::Error DeviceEAGL::getAttribute(const egl::Display *display, EGLint attribute, void **outValue)
{
    switch (attribute)
    {
        default:
            return egl::EglBadAttribute();
    }

    return egl::NoError();
}

EGLint DeviceEAGL::getType()
{
    return 0;
}

void DeviceEAGL::generateExtensions(egl::DeviceExtensions *outExtensions) const {}

}  // namespace rx

#endif  // defined(ANGLE_PLATFORM_IOS)
