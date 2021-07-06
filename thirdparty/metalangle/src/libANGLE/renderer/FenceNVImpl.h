//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FenceNVImpl.h: Defines the rx::FenceNVImpl class.

#ifndef LIBANGLE_RENDERER_FENCENVIMPL_H_
#define LIBANGLE_RENDERER_FENCENVIMPL_H_

#include "libANGLE/Error.h"

#include "common/angleutils.h"

#include "angle_gl.h"

namespace gl
{
class Context;
}  // namespace gl

namespace rx
{
class FenceNVImpl : angle::NonCopyable
{
  public:
    FenceNVImpl() {}
    virtual ~FenceNVImpl() {}

    virtual angle::Result set(const gl::Context *context, GLenum condition)        = 0;
    virtual angle::Result test(const gl::Context *context, GLboolean *outFinished) = 0;
    virtual angle::Result finish(const gl::Context *context)                       = 0;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_FENCENVIMPL_H_
