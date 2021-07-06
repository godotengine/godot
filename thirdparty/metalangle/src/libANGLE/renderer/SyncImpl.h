//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SyncImpl.h: Defines the rx::SyncImpl class.

#ifndef LIBANGLE_RENDERER_FENCESYNCIMPL_H_
#define LIBANGLE_RENDERER_FENCESYNCIMPL_H_

#include "libANGLE/Error.h"

#include "common/angleutils.h"

#include "angle_gl.h"

namespace gl
{
class Context;
}  // namespace gl

namespace rx
{
class SyncImpl : angle::NonCopyable
{
  public:
    SyncImpl() {}
    virtual ~SyncImpl() {}

    virtual void onDestroy(const gl::Context *context) {}

    virtual angle::Result set(const gl::Context *context, GLenum condition, GLbitfield flags) = 0;
    virtual angle::Result clientWait(const gl::Context *context,
                                     GLbitfield flags,
                                     GLuint64 timeout,
                                     GLenum *outResult)                                       = 0;
    virtual angle::Result serverWait(const gl::Context *context,
                                     GLbitfield flags,
                                     GLuint64 timeout)                                        = 0;
    virtual angle::Result getStatus(const gl::Context *context, GLint *outResult)             = 0;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_FENCESYNCIMPL_H_
