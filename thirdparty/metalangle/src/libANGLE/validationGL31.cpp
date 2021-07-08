//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationGL31.cpp: Validation functions for OpenGL 3.1 entry point parameters

#include "libANGLE/validationGL31_autogen.h"

namespace gl
{

bool ValidateGetActiveUniformName(Context *context,
                                  ShaderProgramID program,
                                  GLuint uniformIndex,
                                  GLsizei bufSize,
                                  GLsizei *length,
                                  GLchar *uniformName)
{
    return true;
}

bool ValidatePrimitiveRestartIndex(Context *context, GLuint index)
{
    return true;
}

bool ValidateTexBuffer(Context *context, GLenum target, GLenum internalformat, BufferID buffer)
{
    return true;
}

}  // namespace gl
