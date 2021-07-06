//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ProgramPipelineGL.h:
//     Defines the class interface for ProgramPipelineGL, implementing ProgramPipelineImpl.
//

#ifndef LIBANGLE_RENDERER_GL_PROGRAMPIPELINEGL_H_
#define LIBANGLE_RENDERER_GL_PROGRAMPIPELINEGL_H_

#include "libANGLE/renderer/ProgramPipelineImpl.h"

namespace rx
{

class FunctionsGL;

class ProgramPipelineGL : public ProgramPipelineImpl
{
  public:
    ProgramPipelineGL(const gl::ProgramPipelineState &data, const FunctionsGL *functions);
    ~ProgramPipelineGL() override;

    GLuint getID() const { return mProgramPipelineID; }

  private:
    const FunctionsGL *mFunctions;

    GLuint mProgramPipelineID;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_PROGRAMPIPELINEGL_H_
