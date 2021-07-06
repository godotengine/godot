//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Path.h: Defines the gl::Path class, representing CHROMIUM_path_rendering
// path object.

#include "libANGLE/Path.h"
#include "libANGLE/renderer/PathImpl.h"

#include "common/debug.h"
#include "common/mathutil.h"

namespace gl
{

Path::Path(rx::PathImpl *impl)
    : mPath(impl),
      mHasData(false),
      mEndCaps(GL_FLAT_CHROMIUM),
      mJoinStyle(GL_MITER_REVERT_CHROMIUM),
      mStrokeWidth(1.0f),
      mStrokeBound(0.2f),
      mMiterLimit(4.0f)
{}

Path::~Path()
{
    delete mPath;
}

angle::Result Path::setCommands(GLsizei numCommands,
                                const GLubyte *commands,
                                GLsizei numCoords,
                                GLenum coordType,
                                const void *coords)
{
    ANGLE_TRY(mPath->setCommands(numCommands, commands, numCoords, coordType, coords));

    mHasData = true;

    return angle::Result::Continue;
}

void Path::setStrokeWidth(GLfloat width)
{
    mStrokeWidth = width;
    mPath->setPathParameter(GL_PATH_STROKE_WIDTH_CHROMIUM, mStrokeWidth);
}

void Path::setStrokeBound(GLfloat bound)
{
    mStrokeBound = clamp(bound, 0.0f, 1.0f);
    mPath->setPathParameter(GL_PATH_STROKE_BOUND_CHROMIUM, mStrokeBound);
}

void Path::setEndCaps(GLenum type)
{
    mEndCaps = type;
    mPath->setPathParameter(GL_PATH_END_CAPS_CHROMIUM, static_cast<GLfloat>(type));
}

void Path::setJoinStyle(GLenum type)
{
    mJoinStyle = type;
    mPath->setPathParameter(GL_PATH_JOIN_STYLE_CHROMIUM, static_cast<GLfloat>(type));
}

void Path::setMiterLimit(GLfloat value)
{
    mMiterLimit = value;
    mPath->setPathParameter(GL_PATH_MITER_LIMIT_CHROMIUM, value);
}

}  // namespace gl
