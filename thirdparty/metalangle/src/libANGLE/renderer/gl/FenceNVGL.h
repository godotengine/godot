//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FenceNVGL.h: Defines the class interface for FenceNVGL.

#ifndef LIBANGLE_RENDERER_GL_FENCENVGL_H_
#define LIBANGLE_RENDERER_GL_FENCENVGL_H_

#include "libANGLE/renderer/FenceNVImpl.h"

namespace rx
{
class FunctionsGL;

// FenceNV implemented with the native GL_NV_fence extension
class FenceNVGL : public FenceNVImpl
{
  public:
    explicit FenceNVGL(const FunctionsGL *functions);
    ~FenceNVGL() override;

    angle::Result set(const gl::Context *context, GLenum condition) override;
    angle::Result test(const gl::Context *context, GLboolean *outFinished) override;
    angle::Result finish(const gl::Context *context) override;

    static bool Supported(const FunctionsGL *functions);

  private:
    GLuint mFence;

    const FunctionsGL *mFunctions;
};

// FenceNV implemented with the GLsync API
class FenceNVSyncGL : public FenceNVImpl
{
  public:
    explicit FenceNVSyncGL(const FunctionsGL *functions);
    ~FenceNVSyncGL() override;

    angle::Result set(const gl::Context *context, GLenum condition) override;
    angle::Result test(const gl::Context *context, GLboolean *outFinished) override;
    angle::Result finish(const gl::Context *context) override;

    static bool Supported(const FunctionsGL *functions);

  private:
    GLsync mSyncObject;

    const FunctionsGL *mFunctions;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_FENCENVGL_H_
