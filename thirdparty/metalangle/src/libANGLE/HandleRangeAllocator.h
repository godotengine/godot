//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// HandleRangeAllocator.h: Defines the gl::HandleRangeAllocator class, which is used to
// allocate contiguous ranges of GL path handles.

#ifndef LIBANGLE_HANDLERANGEALLOCATOR_H_
#define LIBANGLE_HANDLERANGEALLOCATOR_H_

#include <map>

#include "angle_gl.h"
#include "common/angleutils.h"

namespace gl
{

// Allocates contiguous ranges of path object handles.
class HandleRangeAllocator final : angle::NonCopyable
{
  public:
    static const GLuint kInvalidHandle;

    HandleRangeAllocator();
    ~HandleRangeAllocator();

    // Allocates a new path handle.
    GLuint allocate();

    // Allocates a handle starting at or above the value of |wanted|.
    // Note: may wrap if it starts near limit.
    GLuint allocateAtOrAbove(GLuint wanted);

    // Allocates |range| amount of contiguous paths.
    // Returns the first id to |first_id| or |kInvalidHandle| if
    // allocation failed.
    GLuint allocateRange(GLuint range);

    // Marks an id as used. Returns false if handle was already used.
    bool markAsUsed(GLuint handle);

    // Release handle.
    void release(GLuint handle);

    // Release a |range| amount of contiguous handles, starting from |first|
    void releaseRange(GLuint first, GLuint range);

    // Checks whether or not a resource ID is in use.
    bool isUsed(GLuint handle) const;

  private:
    std::map<GLuint, GLuint> mUsed;
};

}  // namespace gl

#endif  // LIBANGLE_HANDLERANGEALLOCATOR_H_
