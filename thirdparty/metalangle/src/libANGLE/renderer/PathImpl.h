//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// PathImpl.h: Defines the Path implementation interface for
// CHROMIUM_path_rendering path objects.

#ifndef LIBANGLE_RENDERER_PATHIMPL_H_
#define LIBANGLE_RENDERER_PATHIMPL_H_

#include "angle_gl.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"

namespace rx
{

class PathImpl : angle::NonCopyable
{
  public:
    virtual ~PathImpl() {}

    virtual angle::Result setCommands(GLsizei numCommands,
                                      const GLubyte *commands,
                                      GLsizei numCoords,
                                      GLenum coordType,
                                      const void *coords) = 0;

    virtual void setPathParameter(GLenum pname, GLfloat value) = 0;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_PATHIMPL_H_
