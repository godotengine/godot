//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Path.h: Defines the gl::Path class, representing CHROMIUM_path_rendering
// path object.

#ifndef LIBANGLE_PATH_H_
#define LIBANGLE_PATH_H_

#include "angle_gl.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"

namespace rx
{
class PathImpl;
}

namespace gl
{
class Path final : angle::NonCopyable
{
  public:
    Path(rx::PathImpl *impl);

    ~Path();

    angle::Result setCommands(GLsizei numCommands,
                              const GLubyte *commands,
                              GLsizei numCoords,
                              GLenum coordType,
                              const void *coords);

    void setStrokeWidth(GLfloat width);
    void setStrokeBound(GLfloat bound);
    void setEndCaps(GLenum type);
    void setJoinStyle(GLenum type);
    void setMiterLimit(GLfloat value);

    GLfloat getStrokeWidth() const { return mStrokeWidth; }
    GLfloat getStrokeBound() const { return mStrokeBound; }
    GLfloat getMiterLimit() const { return mMiterLimit; }
    GLenum getEndCaps() const { return mEndCaps; }
    GLenum getJoinStyle() const { return mJoinStyle; }

    bool hasPathData() const { return mHasData; }

    rx::PathImpl *getImplementation() const { return mPath; }

  private:
    rx::PathImpl *mPath;

    // a Path object is not actually considered "a path"
    // untill it has been specified with data. So we'll
    // keep this flag to support this semantics.
    bool mHasData;

    GLenum mEndCaps;
    GLenum mJoinStyle;
    GLfloat mStrokeWidth;
    GLfloat mStrokeBound;
    GLfloat mMiterLimit;
};

}  // namespace gl

#endif  // LIBANGLE_PATH_H_
