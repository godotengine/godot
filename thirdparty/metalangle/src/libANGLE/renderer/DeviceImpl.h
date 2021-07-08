//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// DeviceImpl.h: Implementation methods of egl::Device

#ifndef LIBANGLE_RENDERER_DEVICEIMPL_H_
#define LIBANGLE_RENDERER_DEVICEIMPL_H_

#include "common/angleutils.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Error.h"

namespace egl
{
class Display;
}

namespace rx
{
class DisplayImpl;

class DeviceImpl : angle::NonCopyable
{
  public:
    DeviceImpl();
    virtual ~DeviceImpl();

    virtual egl::Error initialize() = 0;

    virtual egl::Error getAttribute(const egl::Display *display,
                                    EGLint attribute,
                                    void **outValue)                            = 0;
    virtual EGLint getType()                                                    = 0;
    virtual void generateExtensions(egl::DeviceExtensions *outExtensions) const = 0;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_DEVICEIMPL_H_
