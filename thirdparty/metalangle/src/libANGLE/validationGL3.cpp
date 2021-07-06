//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// validationGL3.cpp: Validation functions for OpenGL 3.0 entry point parameters

#include "libANGLE/validationGL3_autogen.h"

namespace gl
{

bool ValidateBeginConditionalRender(Context *context, GLuint id, GLenum mode)
{
    return true;
}

bool ValidateBindFragDataLocation(Context *context,
                                  ShaderProgramID program,
                                  GLuint color,
                                  const GLchar *name)
{
    return true;
}

bool ValidateClampColor(Context *context, GLenum target, GLenum clamp)
{
    return true;
}

bool ValidateColorMaski(Context *context,
                        GLuint index,
                        GLboolean r,
                        GLboolean g,
                        GLboolean b,
                        GLboolean a)
{
    return true;
}

bool ValidateDisablei(Context *context, GLenum target, GLuint index)
{
    return true;
}

bool ValidateEnablei(Context *context, GLenum target, GLuint index)
{
    return true;
}

bool ValidateEndConditionalRender(Context *context)
{
    return true;
}

bool ValidateFramebufferTexture1D(Context *context,
                                  GLenum target,
                                  GLenum attachment,
                                  TextureTarget textargetPacked,
                                  TextureID texture,
                                  GLint level)
{
    return true;
}

bool ValidateFramebufferTexture3D(Context *context,
                                  GLenum target,
                                  GLenum attachment,
                                  TextureTarget textargetPacked,
                                  TextureID texture,
                                  GLint level,
                                  GLint zoffset)
{
    return true;
}

bool ValidateGetTexParameterIiv(Context *context,
                                TextureType targetPacked,
                                GLenum pname,
                                GLint *params)
{
    return true;
}

bool ValidateGetTexParameterIuiv(Context *context,
                                 TextureType targetPacked,
                                 GLenum pname,
                                 GLuint *params)
{
    return true;
}

bool ValidateIsEnabledi(Context *context, GLenum target, GLuint index)
{
    return true;
}

bool ValidateTexParameterIiv(Context *context,
                             TextureType targetPacked,
                             GLenum pname,
                             const GLint *params)
{
    return true;
}

bool ValidateTexParameterIuiv(Context *context,
                              TextureType targetPacked,
                              GLenum pname,
                              const GLuint *params)
{
    return true;
}

bool ValidateVertexAttribI1i(Context *context, GLuint index, GLint x)
{
    return true;
}

bool ValidateVertexAttribI1iv(Context *context, GLuint index, const GLint *v)
{
    return true;
}

bool ValidateVertexAttribI1ui(Context *context, GLuint index, GLuint x)
{
    return true;
}

bool ValidateVertexAttribI1uiv(Context *context, GLuint index, const GLuint *v)
{
    return true;
}

bool ValidateVertexAttribI2i(Context *context, GLuint index, GLint x, GLint y)
{
    return true;
}

bool ValidateVertexAttribI2iv(Context *context, GLuint index, const GLint *v)
{
    return true;
}

bool ValidateVertexAttribI2ui(Context *context, GLuint index, GLuint x, GLuint y)
{
    return true;
}

bool ValidateVertexAttribI2uiv(Context *context, GLuint index, const GLuint *v)
{
    return true;
}

bool ValidateVertexAttribI3i(Context *context, GLuint index, GLint x, GLint y, GLint z)
{
    return true;
}

bool ValidateVertexAttribI3iv(Context *context, GLuint index, const GLint *v)
{
    return true;
}

bool ValidateVertexAttribI3ui(Context *context, GLuint index, GLuint x, GLuint y, GLuint z)
{
    return true;
}

bool ValidateVertexAttribI3uiv(Context *context, GLuint index, const GLuint *v)
{
    return true;
}

bool ValidateVertexAttribI4bv(Context *context, GLuint index, const GLbyte *v)
{
    return true;
}

bool ValidateVertexAttribI4sv(Context *context, GLuint index, const GLshort *v)
{
    return true;
}

bool ValidateVertexAttribI4ubv(Context *context, GLuint index, const GLubyte *v)
{
    return true;
}

bool ValidateVertexAttribI4usv(Context *context, GLuint index, const GLushort *v)
{
    return true;
}

}  // namespace gl
