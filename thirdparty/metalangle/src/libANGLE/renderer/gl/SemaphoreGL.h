//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SempahoreGL.h: Defines the rx::SempahoreGL class, an implementation of SemaphoreImpl.

#ifndef LIBANGLE_RENDERER_GL_SEMAPHOREGL_H_
#define LIBANGLE_RENDERER_GL_SEMAPHOREGL_H_

#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/SemaphoreImpl.h"

namespace rx
{
class SemaphoreGL : public SemaphoreImpl
{
  public:
    SemaphoreGL(GLuint semaphoreID);
    ~SemaphoreGL() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result importFd(gl::Context *context, gl::HandleType handleType, GLint fd) override;

    angle::Result wait(gl::Context *context,
                       const gl::BufferBarrierVector &bufferBarriers,
                       const gl::TextureBarrierVector &textureBarriers) override;

    angle::Result signal(gl::Context *context,
                         const gl::BufferBarrierVector &bufferBarriers,
                         const gl::TextureBarrierVector &textureBarriers) override;

    GLuint getSemaphoreID() const;

  private:
    GLuint mSemaphoreID;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_SEMAPHOREGL_H_
