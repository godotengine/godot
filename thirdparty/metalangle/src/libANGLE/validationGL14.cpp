//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationGL14.cpp: Validation functions for OpenGL 1.4 entry point parameters

#include "libANGLE/validationGL14_autogen.h"

namespace gl
{

bool ValidateFogCoordPointer(Context *context, GLenum type, GLsizei stride, const void *pointer)
{
    return true;
}

bool ValidateFogCoordd(Context *context, GLdouble coord)
{
    return true;
}

bool ValidateFogCoorddv(Context *context, const GLdouble *coord)
{
    return true;
}

bool ValidateFogCoordf(Context *context, GLfloat coord)
{
    return true;
}

bool ValidateFogCoordfv(Context *context, const GLfloat *coord)
{
    return true;
}

bool ValidateMultiDrawArrays(Context *context,
                             PrimitiveMode modePacked,
                             const GLint *first,
                             const GLsizei *count,
                             GLsizei drawcount)
{
    return true;
}

bool ValidateMultiDrawElements(Context *context,
                               PrimitiveMode modePacked,
                               const GLsizei *count,
                               DrawElementsType typePacked,
                               const void *const *indices,
                               GLsizei drawcount)
{
    return true;
}

bool ValidatePointParameteri(Context *context, GLenum pname, GLint param)
{
    return true;
}

bool ValidatePointParameteriv(Context *context, GLenum pname, const GLint *params)
{
    return true;
}

bool ValidateSecondaryColor3b(Context *context, GLbyte red, GLbyte green, GLbyte blue)
{
    return true;
}

bool ValidateSecondaryColor3bv(Context *context, const GLbyte *v)
{
    return true;
}

bool ValidateSecondaryColor3d(Context *context, GLdouble red, GLdouble green, GLdouble blue)
{
    return true;
}

bool ValidateSecondaryColor3dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateSecondaryColor3f(Context *context, GLfloat red, GLfloat green, GLfloat blue)
{
    return true;
}

bool ValidateSecondaryColor3fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateSecondaryColor3i(Context *context, GLint red, GLint green, GLint blue)
{
    return true;
}

bool ValidateSecondaryColor3iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateSecondaryColor3s(Context *context, GLshort red, GLshort green, GLshort blue)
{
    return true;
}

bool ValidateSecondaryColor3sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateSecondaryColor3ub(Context *context, GLubyte red, GLubyte green, GLubyte blue)
{
    return true;
}

bool ValidateSecondaryColor3ubv(Context *context, const GLubyte *v)
{
    return true;
}

bool ValidateSecondaryColor3ui(Context *context, GLuint red, GLuint green, GLuint blue)
{
    return true;
}

bool ValidateSecondaryColor3uiv(Context *context, const GLuint *v)
{
    return true;
}

bool ValidateSecondaryColor3us(Context *context, GLushort red, GLushort green, GLushort blue)
{
    return true;
}

bool ValidateSecondaryColor3usv(Context *context, const GLushort *v)
{
    return true;
}

bool ValidateSecondaryColorPointer(Context *context,
                                   GLint size,
                                   GLenum type,
                                   GLsizei stride,
                                   const void *pointer)
{
    return true;
}

bool ValidateWindowPos2d(Context *context, GLdouble x, GLdouble y)
{
    return true;
}

bool ValidateWindowPos2dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateWindowPos2f(Context *context, GLfloat x, GLfloat y)
{
    return true;
}

bool ValidateWindowPos2fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateWindowPos2i(Context *context, GLint x, GLint y)
{
    return true;
}

bool ValidateWindowPos2iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateWindowPos2s(Context *context, GLshort x, GLshort y)
{
    return true;
}

bool ValidateWindowPos2sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateWindowPos3d(Context *context, GLdouble x, GLdouble y, GLdouble z)
{
    return true;
}

bool ValidateWindowPos3dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateWindowPos3f(Context *context, GLfloat x, GLfloat y, GLfloat z)
{
    return true;
}

bool ValidateWindowPos3fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateWindowPos3i(Context *context, GLint x, GLint y, GLint z)
{
    return true;
}

bool ValidateWindowPos3iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateWindowPos3s(Context *context, GLshort x, GLshort y, GLshort z)
{
    return true;
}

bool ValidateWindowPos3sv(Context *context, const GLshort *v)
{
    return true;
}

}  // namespace gl
