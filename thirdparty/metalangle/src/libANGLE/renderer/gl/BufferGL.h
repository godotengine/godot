//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// BufferGL.h: Defines the class interface for BufferGL.

#ifndef LIBANGLE_RENDERER_GL_BUFFERGL_H_
#define LIBANGLE_RENDERER_GL_BUFFERGL_H_

#include "common/MemoryBuffer.h"
#include "libANGLE/renderer/BufferImpl.h"

namespace rx
{

class FunctionsGL;
class StateManagerGL;

class BufferGL : public BufferImpl
{
  public:
    BufferGL(const gl::BufferState &state,
             const FunctionsGL *functions,
             StateManagerGL *stateManager);
    ~BufferGL() override;

    angle::Result setData(const gl::Context *context,
                          gl::BufferBinding target,
                          const void *data,
                          size_t size,
                          gl::BufferUsage usage) override;
    angle::Result setSubData(const gl::Context *context,
                             gl::BufferBinding target,
                             const void *data,
                             size_t size,
                             size_t offset) override;
    angle::Result copySubData(const gl::Context *context,
                              BufferImpl *source,
                              GLintptr sourceOffset,
                              GLintptr destOffset,
                              GLsizeiptr size) override;
    angle::Result map(const gl::Context *context, GLenum access, void **mapPtr) override;
    angle::Result mapRange(const gl::Context *context,
                           size_t offset,
                           size_t length,
                           GLbitfield access,
                           void **mapPtr) override;
    angle::Result unmap(const gl::Context *context, GLboolean *result) override;

    angle::Result getIndexRange(const gl::Context *context,
                                gl::DrawElementsType type,
                                size_t offset,
                                size_t count,
                                bool primitiveRestartEnabled,
                                gl::IndexRange *outRange) override;

    GLuint getBufferID() const;

  private:
    bool mIsMapped;
    size_t mMapOffset;
    size_t mMapSize;

    bool mShadowBufferData;
    angle::MemoryBuffer mShadowCopy;

    size_t mBufferSize;

    const FunctionsGL *mFunctions;
    StateManagerGL *mStateManager;

    GLuint mBufferID;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_BUFFERGL_H_
