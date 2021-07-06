//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// EGLSyncImpl.h: Defines the rx::EGLSyncImpl class.

#ifndef LIBANGLE_RENDERER_EGLSYNCIMPL_H_
#define LIBANGLE_RENDERER_EGLSYNCIMPL_H_

#include "libANGLE/Error.h"

#include "common/angleutils.h"

#include "angle_gl.h"

namespace egl
{
class Display;
}  // namespace egl

namespace gl
{
class Context;
}  // namespace gl

namespace rx
{
class EGLSyncImpl : angle::NonCopyable
{
  public:
    EGLSyncImpl() {}
    virtual ~EGLSyncImpl() {}

    virtual void onDestroy(const egl::Display *display) {}

    virtual egl::Error initialize(const egl::Display *display,
                                  const gl::Context *context,
                                  EGLenum type)                                  = 0;
    virtual egl::Error clientWait(const egl::Display *display,
                                  const gl::Context *context,
                                  EGLint flags,
                                  EGLTime timeout,
                                  EGLint *outResult)                             = 0;
    virtual egl::Error serverWait(const egl::Display *display,
                                  const gl::Context *context,
                                  EGLint flags)                                  = 0;
    virtual egl::Error getStatus(const egl::Display *display, EGLint *outStatus) = 0;

    virtual egl::Error dupNativeFenceFD(const egl::Display *display, EGLint *result) const = 0;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_EGLSYNCIMPL_H_
