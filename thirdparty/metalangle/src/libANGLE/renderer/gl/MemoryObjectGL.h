// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// MemoryObjectGL.h: Defines the class interface for MemoryObjectGL,
// implementing MemoryObjectImpl.

#ifndef LIBANGLE_RENDERER_GL_MEMORYOBJECTGL_H_
#define LIBANGLE_RENDERER_GL_MEMORYOBJECTGL_H_

#include "libANGLE/renderer/MemoryObjectImpl.h"

namespace rx
{

class MemoryObjectGL : public MemoryObjectImpl
{
  public:
    MemoryObjectGL(GLuint memoryObject);
    ~MemoryObjectGL() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result importFd(gl::Context *context,
                           GLuint64 size,
                           gl::HandleType handleType,
                           GLint fd) override;

    GLuint getMemoryObjectID() const;

  private:
    GLuint mMemoryObject;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_MEMORYOBJECTGL_H_
