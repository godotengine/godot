//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// BufferImpl.h: Defines the abstract rx::BufferImpl class.

#ifndef LIBANGLE_RENDERER_BUFFERIMPL_H_
#define LIBANGLE_RENDERER_BUFFERIMPL_H_

#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "common/mathutil.h"
#include "libANGLE/Error.h"
#include "libANGLE/Observer.h"

#include <stdint.h>

namespace gl
{
class BufferState;
class Context;
}  // namespace gl

namespace rx
{
// We use two set of Subject messages. The CONTENTS_CHANGED message is signaled whenever data
// changes, to trigger re-translation or other events. Some buffers only need to be updated when the
// underlying driver object changes - this is notified via the STORAGE_CHANGED message.
class BufferImpl : public angle::Subject
{
  public:
    BufferImpl(const gl::BufferState &state) : mState(state) {}
    ~BufferImpl() override {}
    virtual void destroy(const gl::Context *context) {}

    virtual angle::Result setData(const gl::Context *context,
                                  gl::BufferBinding target,
                                  const void *data,
                                  size_t size,
                                  gl::BufferUsage usage)                                = 0;
    virtual angle::Result setSubData(const gl::Context *context,
                                     gl::BufferBinding target,
                                     const void *data,
                                     size_t size,
                                     size_t offset)                                     = 0;
    virtual angle::Result copySubData(const gl::Context *context,
                                      BufferImpl *source,
                                      GLintptr sourceOffset,
                                      GLintptr destOffset,
                                      GLsizeiptr size)                                  = 0;
    virtual angle::Result map(const gl::Context *context, GLenum access, void **mapPtr) = 0;
    virtual angle::Result mapRange(const gl::Context *context,
                                   size_t offset,
                                   size_t length,
                                   GLbitfield access,
                                   void **mapPtr)                                       = 0;
    virtual angle::Result unmap(const gl::Context *context, GLboolean *result)          = 0;

    virtual angle::Result getIndexRange(const gl::Context *context,
                                        gl::DrawElementsType type,
                                        size_t offset,
                                        size_t count,
                                        bool primitiveRestartEnabled,
                                        gl::IndexRange *outRange) = 0;

    // Override if accurate native memory size information is available
    virtual GLint64 getMemorySize() const;

    virtual void onDataChanged() {}

  protected:
    const gl::BufferState &mState;
};

inline GLint64 BufferImpl::getMemorySize() const
{
    return 0;
}

}  // namespace rx

#endif  // LIBANGLE_RENDERER_BUFFERIMPL_H_
