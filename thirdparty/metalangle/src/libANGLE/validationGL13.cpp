//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationGL13.cpp: Validation functions for OpenGL 1.3 entry point parameters

#include "libANGLE/validationGL13_autogen.h"

namespace gl
{

bool ValidateCompressedTexImage1D(Context *context,
                                  GLenum target,
                                  GLint level,
                                  GLenum internalformat,
                                  GLsizei width,
                                  GLint border,
                                  GLsizei imageSize,
                                  const void *data)
{
    return true;
}

bool ValidateCompressedTexSubImage1D(Context *context,
                                     GLenum target,
                                     GLint level,
                                     GLint xoffset,
                                     GLsizei width,
                                     GLenum format,
                                     GLsizei imageSize,
                                     const void *data)
{
    return true;
}

bool ValidateGetCompressedTexImage(Context *context, GLenum target, GLint level, void *img)
{
    return true;
}

bool ValidateLoadTransposeMatrixd(Context *context, const GLdouble *m)
{
    return true;
}

bool ValidateLoadTransposeMatrixf(Context *context, const GLfloat *m)
{
    return true;
}

bool ValidateMultTransposeMatrixd(Context *context, const GLdouble *m)
{
    return true;
}

bool ValidateMultTransposeMatrixf(Context *context, const GLfloat *m)
{
    return true;
}

bool ValidateMultiTexCoord1d(Context *context, GLenum target, GLdouble s)
{
    return true;
}

bool ValidateMultiTexCoord1dv(Context *context, GLenum target, const GLdouble *v)
{
    return true;
}

bool ValidateMultiTexCoord1f(Context *context, GLenum target, GLfloat s)
{
    return true;
}

bool ValidateMultiTexCoord1fv(Context *context, GLenum target, const GLfloat *v)
{
    return true;
}

bool ValidateMultiTexCoord1i(Context *context, GLenum target, GLint s)
{
    return true;
}

bool ValidateMultiTexCoord1iv(Context *context, GLenum target, const GLint *v)
{
    return true;
}

bool ValidateMultiTexCoord1s(Context *context, GLenum target, GLshort s)
{
    return true;
}

bool ValidateMultiTexCoord1sv(Context *context, GLenum target, const GLshort *v)
{
    return true;
}

bool ValidateMultiTexCoord2d(Context *context, GLenum target, GLdouble s, GLdouble t)
{
    return true;
}

bool ValidateMultiTexCoord2dv(Context *context, GLenum target, const GLdouble *v)
{
    return true;
}

bool ValidateMultiTexCoord2f(Context *context, GLenum target, GLfloat s, GLfloat t)
{
    return true;
}

bool ValidateMultiTexCoord2fv(Context *context, GLenum target, const GLfloat *v)
{
    return true;
}

bool ValidateMultiTexCoord2i(Context *context, GLenum target, GLint s, GLint t)
{
    return true;
}

bool ValidateMultiTexCoord2iv(Context *context, GLenum target, const GLint *v)
{
    return true;
}

bool ValidateMultiTexCoord2s(Context *context, GLenum target, GLshort s, GLshort t)
{
    return true;
}

bool ValidateMultiTexCoord2sv(Context *context, GLenum target, const GLshort *v)
{
    return true;
}

bool ValidateMultiTexCoord3d(Context *context, GLenum target, GLdouble s, GLdouble t, GLdouble r)
{
    return true;
}

bool ValidateMultiTexCoord3dv(Context *context, GLenum target, const GLdouble *v)
{
    return true;
}

bool ValidateMultiTexCoord3f(Context *context, GLenum target, GLfloat s, GLfloat t, GLfloat r)
{
    return true;
}

bool ValidateMultiTexCoord3fv(Context *context, GLenum target, const GLfloat *v)
{
    return true;
}

bool ValidateMultiTexCoord3i(Context *context, GLenum target, GLint s, GLint t, GLint r)
{
    return true;
}

bool ValidateMultiTexCoord3iv(Context *context, GLenum target, const GLint *v)
{
    return true;
}

bool ValidateMultiTexCoord3s(Context *context, GLenum target, GLshort s, GLshort t, GLshort r)
{
    return true;
}

bool ValidateMultiTexCoord3sv(Context *context, GLenum target, const GLshort *v)
{
    return true;
}

bool ValidateMultiTexCoord4d(Context *context,
                             GLenum target,
                             GLdouble s,
                             GLdouble t,
                             GLdouble r,
                             GLdouble q)
{
    return true;
}

bool ValidateMultiTexCoord4dv(Context *context, GLenum target, const GLdouble *v)
{
    return true;
}

bool ValidateMultiTexCoord4fv(Context *context, GLenum target, const GLfloat *v)
{
    return true;
}

bool ValidateMultiTexCoord4i(Context *context, GLenum target, GLint s, GLint t, GLint r, GLint q)
{
    return true;
}

bool ValidateMultiTexCoord4iv(Context *context, GLenum target, const GLint *v)
{
    return true;
}

bool ValidateMultiTexCoord4s(Context *context,
                             GLenum target,
                             GLshort s,
                             GLshort t,
                             GLshort r,
                             GLshort q)
{
    return true;
}

bool ValidateMultiTexCoord4sv(Context *context, GLenum target, const GLshort *v)
{
    return true;
}

}  // namespace gl
