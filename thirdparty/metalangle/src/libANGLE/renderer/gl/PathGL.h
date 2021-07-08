//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// PathGL.h: Class definition for CHROMIUM_path_rendering path object for the
// GL backend.

#ifndef LIBANGLE_RENDERER_GL_PATHIMPL_H_
#define LIBANGLE_RENDERER_GL_PATHIMPL_H_

#include "libANGLE/renderer/PathImpl.h"

namespace rx
{

class FunctionsGL;

class PathGL : public PathImpl
{
  public:
    PathGL(const FunctionsGL *functions, GLuint path);
    ~PathGL() override;

    angle::Result setCommands(GLsizei numCommands,
                              const GLubyte *commands,
                              GLsizei numCoords,
                              GLenum coordType,
                              const void *coords) override;

    void setPathParameter(GLenum pname, GLfloat value) override;

    GLuint getPathID() const { return mPathID; }

  private:
    const FunctionsGL *mFunctions;

    GLuint mPathID;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_PATHIMPL_H_
