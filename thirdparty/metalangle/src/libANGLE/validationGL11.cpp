//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationGL11.cpp: Validation functions for OpenGL 1.1 entry point parameters

#include "libANGLE/validationGL11_autogen.h"

namespace gl
{

bool ValidateAreTexturesResident(Context *context,
                                 GLsizei n,
                                 const GLuint *textures,
                                 GLboolean *residences)
{
    return true;
}

bool ValidateArrayElement(Context *context, GLint i)
{
    return true;
}

bool ValidateCopyTexImage1D(Context *context,
                            GLenum target,
                            GLint level,
                            GLenum internalformat,
                            GLint x,
                            GLint y,
                            GLsizei width,
                            GLint border)
{
    return true;
}

bool ValidateCopyTexSubImage1D(Context *context,
                               GLenum target,
                               GLint level,
                               GLint xoffset,
                               GLint x,
                               GLint y,
                               GLsizei width)
{
    return true;
}

bool ValidateEdgeFlagPointer(Context *context, GLsizei stride, const void *pointer)
{
    return true;
}

bool ValidateIndexPointer(Context *context, GLenum type, GLsizei stride, const void *pointer)
{
    return true;
}

bool ValidateIndexub(Context *context, GLubyte c)
{
    return true;
}

bool ValidateIndexubv(Context *context, const GLubyte *c)
{
    return true;
}

bool ValidateInterleavedArrays(Context *context, GLenum format, GLsizei stride, const void *pointer)
{
    return true;
}

bool ValidatePopClientAttrib(Context *context)
{
    return true;
}

bool ValidatePrioritizeTextures(Context *context,
                                GLsizei n,
                                const GLuint *textures,
                                const GLfloat *priorities)
{
    return true;
}

bool ValidatePushClientAttrib(Context *context, GLbitfield mask)
{
    return true;
}

bool ValidateTexSubImage1D(Context *context,
                           GLenum target,
                           GLint level,
                           GLint xoffset,
                           GLsizei width,
                           GLenum format,
                           GLenum type,
                           const void *pixels)
{
    return true;
}

}  // namespace gl
