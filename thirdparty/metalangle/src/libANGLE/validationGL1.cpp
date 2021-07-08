//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// validationGL1.cpp: Validation functions for OpenGL 1.0 entry point parameters

#include "libANGLE/validationGL1_autogen.h"

namespace gl
{

bool ValidateAccum(Context *context, GLenum op, GLfloat value)
{
    return true;
}

bool ValidateBegin(Context *context, GLenum mode)
{
    return true;
}

bool ValidateBitmap(Context *context,
                    GLsizei width,
                    GLsizei height,
                    GLfloat xorig,
                    GLfloat yorig,
                    GLfloat xmove,
                    GLfloat ymove,
                    const GLubyte *bitmap)
{
    return true;
}

bool ValidateCallList(Context *context, GLuint list)
{
    return true;
}

bool ValidateCallLists(Context *context, GLsizei n, GLenum type, const void *lists)
{
    return true;
}

bool ValidateClearAccum(Context *context, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    return true;
}

bool ValidateClearDepth(Context *context, GLdouble depth)
{
    return true;
}

bool ValidateClearIndex(Context *context, GLfloat c)
{
    return true;
}

bool ValidateClipPlane(Context *context, GLenum plane, const GLdouble *equation)
{
    return true;
}

bool ValidateColor3b(Context *context, GLbyte red, GLbyte green, GLbyte blue)
{
    return true;
}

bool ValidateColor3bv(Context *context, const GLbyte *v)
{
    return true;
}

bool ValidateColor3d(Context *context, GLdouble red, GLdouble green, GLdouble blue)
{
    return true;
}

bool ValidateColor3dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateColor3f(Context *context, GLfloat red, GLfloat green, GLfloat blue)
{
    return true;
}

bool ValidateColor3fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateColor3i(Context *context, GLint red, GLint green, GLint blue)
{
    return true;
}

bool ValidateColor3iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateColor3s(Context *context, GLshort red, GLshort green, GLshort blue)
{
    return true;
}

bool ValidateColor3sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateColor3ub(Context *context, GLubyte red, GLubyte green, GLubyte blue)
{
    return true;
}

bool ValidateColor3ubv(Context *context, const GLubyte *v)
{
    return true;
}

bool ValidateColor3ui(Context *context, GLuint red, GLuint green, GLuint blue)
{
    return true;
}

bool ValidateColor3uiv(Context *context, const GLuint *v)
{
    return true;
}

bool ValidateColor3us(Context *context, GLushort red, GLushort green, GLushort blue)
{
    return true;
}

bool ValidateColor3usv(Context *context, const GLushort *v)
{
    return true;
}

bool ValidateColor4b(Context *context, GLbyte red, GLbyte green, GLbyte blue, GLbyte alpha)
{
    return true;
}

bool ValidateColor4bv(Context *context, const GLbyte *v)
{
    return true;
}

bool ValidateColor4d(Context *context, GLdouble red, GLdouble green, GLdouble blue, GLdouble alpha)
{
    return true;
}

bool ValidateColor4dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateColor4fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateColor4i(Context *context, GLint red, GLint green, GLint blue, GLint alpha)
{
    return true;
}

bool ValidateColor4iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateColor4s(Context *context, GLshort red, GLshort green, GLshort blue, GLshort alpha)
{
    return true;
}

bool ValidateColor4sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateColor4ubv(Context *context, const GLubyte *v)
{
    return true;
}

bool ValidateColor4ui(Context *context, GLuint red, GLuint green, GLuint blue, GLuint alpha)
{
    return true;
}

bool ValidateColor4uiv(Context *context, const GLuint *v)
{
    return true;
}

bool ValidateColor4us(Context *context, GLushort red, GLushort green, GLushort blue, GLushort alpha)
{
    return true;
}

bool ValidateColor4usv(Context *context, const GLushort *v)
{
    return true;
}

bool ValidateColorMaterial(Context *context, GLenum face, GLenum mode)
{
    return true;
}

bool ValidateCopyPixels(Context *context,
                        GLint x,
                        GLint y,
                        GLsizei width,
                        GLsizei height,
                        GLenum type)
{
    return true;
}

bool ValidateDeleteLists(Context *context, GLuint list, GLsizei range)
{
    return true;
}

bool ValidateDepthRange(Context *context, GLdouble n, GLdouble f)
{
    return true;
}

bool ValidateDrawBuffer(Context *context, GLenum buf)
{
    return true;
}

bool ValidateDrawPixels(Context *context,
                        GLsizei width,
                        GLsizei height,
                        GLenum format,
                        GLenum type,
                        const void *pixels)
{
    return true;
}

bool ValidateEdgeFlag(Context *context, GLboolean flag)
{
    return true;
}

bool ValidateEdgeFlagv(Context *context, const GLboolean *flag)
{
    return true;
}

bool ValidateEnd(Context *context)
{
    return true;
}

bool ValidateEndList(Context *context)
{
    return true;
}

bool ValidateEvalCoord1d(Context *context, GLdouble u)
{
    return true;
}

bool ValidateEvalCoord1dv(Context *context, const GLdouble *u)
{
    return true;
}

bool ValidateEvalCoord1f(Context *context, GLfloat u)
{
    return true;
}

bool ValidateEvalCoord1fv(Context *context, const GLfloat *u)
{
    return true;
}

bool ValidateEvalCoord2d(Context *context, GLdouble u, GLdouble v)
{
    return true;
}

bool ValidateEvalCoord2dv(Context *context, const GLdouble *u)
{
    return true;
}

bool ValidateEvalCoord2f(Context *context, GLfloat u, GLfloat v)
{
    return true;
}

bool ValidateEvalCoord2fv(Context *context, const GLfloat *u)
{
    return true;
}

bool ValidateEvalMesh1(Context *context, GLenum mode, GLint i1, GLint i2)
{
    return true;
}

bool ValidateEvalMesh2(Context *context, GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2)
{
    return true;
}

bool ValidateEvalPoint1(Context *context, GLint i)
{
    return true;
}

bool ValidateEvalPoint2(Context *context, GLint i, GLint j)
{
    return true;
}

bool ValidateFeedbackBuffer(Context *context, GLsizei size, GLenum type, GLfloat *buffer)
{
    return true;
}

bool ValidateFogi(Context *context, GLenum pname, GLint param)
{
    return true;
}

bool ValidateFogiv(Context *context, GLenum pname, const GLint *params)
{
    return true;
}

bool ValidateFrustum(Context *context,
                     GLdouble left,
                     GLdouble right,
                     GLdouble bottom,
                     GLdouble top,
                     GLdouble zNear,
                     GLdouble zFar)
{
    return true;
}

bool ValidateGenLists(Context *context, GLsizei range)
{
    return true;
}

bool ValidateGetClipPlane(Context *context, GLenum plane, GLdouble *equation)
{
    return true;
}

bool ValidateGetDoublev(Context *context, GLenum pname, GLdouble *data)
{
    return true;
}

bool ValidateGetLightiv(Context *context, GLenum light, GLenum pname, GLint *params)
{
    return true;
}

bool ValidateGetMapdv(Context *context, GLenum target, GLenum query, GLdouble *v)
{
    return true;
}

bool ValidateGetMapfv(Context *context, GLenum target, GLenum query, GLfloat *v)
{
    return true;
}

bool ValidateGetMapiv(Context *context, GLenum target, GLenum query, GLint *v)
{
    return true;
}

bool ValidateGetMaterialiv(Context *context, GLenum face, GLenum pname, GLint *params)
{
    return true;
}

bool ValidateGetPixelMapfv(Context *context, GLenum map, GLfloat *values)
{
    return true;
}

bool ValidateGetPixelMapuiv(Context *context, GLenum map, GLuint *values)
{
    return true;
}

bool ValidateGetPixelMapusv(Context *context, GLenum map, GLushort *values)
{
    return true;
}

bool ValidateGetPolygonStipple(Context *context, GLubyte *mask)
{
    return true;
}

bool ValidateGetTexGendv(Context *context, GLenum coord, GLenum pname, GLdouble *params)
{
    return true;
}

bool ValidateGetTexGenfv(Context *context, GLenum coord, GLenum pname, GLfloat *params)
{
    return true;
}

bool ValidateGetTexGeniv(Context *context, GLenum coord, GLenum pname, GLint *params)
{
    return true;
}

bool ValidateGetTexImage(Context *context,
                         GLenum target,
                         GLint level,
                         GLenum format,
                         GLenum type,
                         void *pixels)
{
    return true;
}

bool ValidateIndexMask(Context *context, GLuint mask)
{
    return true;
}

bool ValidateIndexd(Context *context, GLdouble c)
{
    return true;
}

bool ValidateIndexdv(Context *context, const GLdouble *c)
{
    return true;
}

bool ValidateIndexf(Context *context, GLfloat c)
{
    return true;
}

bool ValidateIndexfv(Context *context, const GLfloat *c)
{
    return true;
}

bool ValidateIndexi(Context *context, GLint c)
{
    return true;
}

bool ValidateIndexiv(Context *context, const GLint *c)
{
    return true;
}

bool ValidateIndexs(Context *context, GLshort c)
{
    return true;
}

bool ValidateIndexsv(Context *context, const GLshort *c)
{
    return true;
}

bool ValidateInitNames(Context *context)
{
    return true;
}

bool ValidateIsList(Context *context, GLuint list)
{
    return true;
}

bool ValidateLightModeli(Context *context, GLenum pname, GLint param)
{
    return true;
}

bool ValidateLightModeliv(Context *context, GLenum pname, const GLint *params)
{
    return true;
}

bool ValidateLighti(Context *context, GLenum light, GLenum pname, GLint param)
{
    return true;
}

bool ValidateLightiv(Context *context, GLenum light, GLenum pname, const GLint *params)
{
    return true;
}

bool ValidateLineStipple(Context *context, GLint factor, GLushort pattern)
{
    return true;
}

bool ValidateListBase(Context *context, GLuint base)
{
    return true;
}

bool ValidateLoadMatrixd(Context *context, const GLdouble *m)
{
    return true;
}

bool ValidateLoadName(Context *context, GLuint name)
{
    return true;
}

bool ValidateMap1d(Context *context,
                   GLenum target,
                   GLdouble u1,
                   GLdouble u2,
                   GLint stride,
                   GLint order,
                   const GLdouble *points)
{
    return true;
}

bool ValidateMap1f(Context *context,
                   GLenum target,
                   GLfloat u1,
                   GLfloat u2,
                   GLint stride,
                   GLint order,
                   const GLfloat *points)
{
    return true;
}

bool ValidateMap2d(Context *context,
                   GLenum target,
                   GLdouble u1,
                   GLdouble u2,
                   GLint ustride,
                   GLint uorder,
                   GLdouble v1,
                   GLdouble v2,
                   GLint vstride,
                   GLint vorder,
                   const GLdouble *points)
{
    return true;
}

bool ValidateMap2f(Context *context,
                   GLenum target,
                   GLfloat u1,
                   GLfloat u2,
                   GLint ustride,
                   GLint uorder,
                   GLfloat v1,
                   GLfloat v2,
                   GLint vstride,
                   GLint vorder,
                   const GLfloat *points)
{
    return true;
}

bool ValidateMapGrid1d(Context *context, GLint un, GLdouble u1, GLdouble u2)
{
    return true;
}

bool ValidateMapGrid1f(Context *context, GLint un, GLfloat u1, GLfloat u2)
{
    return true;
}

bool ValidateMapGrid2d(Context *context,
                       GLint un,
                       GLdouble u1,
                       GLdouble u2,
                       GLint vn,
                       GLdouble v1,
                       GLdouble v2)
{
    return true;
}

bool ValidateMapGrid2f(Context *context,
                       GLint un,
                       GLfloat u1,
                       GLfloat u2,
                       GLint vn,
                       GLfloat v1,
                       GLfloat v2)
{
    return true;
}

bool ValidateMateriali(Context *context, GLenum face, GLenum pname, GLint param)
{
    return true;
}

bool ValidateMaterialiv(Context *context, GLenum face, GLenum pname, const GLint *params)
{
    return true;
}

bool ValidateMultMatrixd(Context *context, const GLdouble *m)
{
    return true;
}

bool ValidateNewList(Context *context, GLuint list, GLenum mode)
{
    return true;
}

bool ValidateNormal3b(Context *context, GLbyte nx, GLbyte ny, GLbyte nz)
{
    return true;
}

bool ValidateNormal3bv(Context *context, const GLbyte *v)
{
    return true;
}

bool ValidateNormal3d(Context *context, GLdouble nx, GLdouble ny, GLdouble nz)
{
    return true;
}

bool ValidateNormal3dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateNormal3fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateNormal3i(Context *context, GLint nx, GLint ny, GLint nz)
{
    return true;
}

bool ValidateNormal3iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateNormal3s(Context *context, GLshort nx, GLshort ny, GLshort nz)
{
    return true;
}

bool ValidateNormal3sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateOrtho(Context *context,
                   GLdouble left,
                   GLdouble right,
                   GLdouble bottom,
                   GLdouble top,
                   GLdouble zNear,
                   GLdouble zFar)
{
    return true;
}

bool ValidatePassThrough(Context *context, GLfloat token)
{
    return true;
}

bool ValidatePixelMapfv(Context *context, GLenum map, GLsizei mapsize, const GLfloat *values)
{
    return true;
}

bool ValidatePixelMapuiv(Context *context, GLenum map, GLsizei mapsize, const GLuint *values)
{
    return true;
}

bool ValidatePixelMapusv(Context *context, GLenum map, GLsizei mapsize, const GLushort *values)
{
    return true;
}

bool ValidatePixelStoref(Context *context, GLenum pname, GLfloat param)
{
    return true;
}

bool ValidatePixelTransferf(Context *context, GLenum pname, GLfloat param)
{
    return true;
}

bool ValidatePixelTransferi(Context *context, GLenum pname, GLint param)
{
    return true;
}

bool ValidatePixelZoom(Context *context, GLfloat xfactor, GLfloat yfactor)
{
    return true;
}

bool ValidatePolygonMode(Context *context, GLenum face, GLenum mode)
{
    return true;
}

bool ValidatePolygonStipple(Context *context, const GLubyte *mask)
{
    return true;
}

bool ValidatePopAttrib(Context *context)
{
    return true;
}

bool ValidatePopName(Context *context)
{
    return true;
}

bool ValidatePushAttrib(Context *context, GLbitfield mask)
{
    return true;
}

bool ValidatePushName(Context *context, GLuint name)
{
    return true;
}

bool ValidateRasterPos2d(Context *context, GLdouble x, GLdouble y)
{
    return true;
}

bool ValidateRasterPos2dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateRasterPos2f(Context *context, GLfloat x, GLfloat y)
{
    return true;
}

bool ValidateRasterPos2fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateRasterPos2i(Context *context, GLint x, GLint y)
{
    return true;
}

bool ValidateRasterPos2iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateRasterPos2s(Context *context, GLshort x, GLshort y)
{
    return true;
}

bool ValidateRasterPos2sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateRasterPos3d(Context *context, GLdouble x, GLdouble y, GLdouble z)
{
    return true;
}

bool ValidateRasterPos3dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateRasterPos3f(Context *context, GLfloat x, GLfloat y, GLfloat z)
{
    return true;
}

bool ValidateRasterPos3fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateRasterPos3i(Context *context, GLint x, GLint y, GLint z)
{
    return true;
}

bool ValidateRasterPos3iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateRasterPos3s(Context *context, GLshort x, GLshort y, GLshort z)
{
    return true;
}

bool ValidateRasterPos3sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateRasterPos4d(Context *context, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    return true;
}

bool ValidateRasterPos4dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateRasterPos4f(Context *context, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    return true;
}

bool ValidateRasterPos4fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateRasterPos4i(Context *context, GLint x, GLint y, GLint z, GLint w)
{
    return true;
}

bool ValidateRasterPos4iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateRasterPos4s(Context *context, GLshort x, GLshort y, GLshort z, GLshort w)
{
    return true;
}

bool ValidateRasterPos4sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateRectd(Context *context, GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2)
{
    return true;
}

bool ValidateRectdv(Context *context, const GLdouble *v1, const GLdouble *v2)
{
    return true;
}

bool ValidateRectf(Context *context, GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2)
{
    return true;
}

bool ValidateRectfv(Context *context, const GLfloat *v1, const GLfloat *v2)
{
    return true;
}

bool ValidateRecti(Context *context, GLint x1, GLint y1, GLint x2, GLint y2)
{
    return true;
}

bool ValidateRectiv(Context *context, const GLint *v1, const GLint *v2)
{
    return true;
}

bool ValidateRects(Context *context, GLshort x1, GLshort y1, GLshort x2, GLshort y2)
{
    return true;
}

bool ValidateRectsv(Context *context, const GLshort *v1, const GLshort *v2)
{
    return true;
}

bool ValidateRenderMode(Context *context, GLenum mode)
{
    return true;
}

bool ValidateRotated(Context *context, GLdouble angle, GLdouble x, GLdouble y, GLdouble z)
{
    return true;
}

bool ValidateScaled(Context *context, GLdouble x, GLdouble y, GLdouble z)
{
    return true;
}

bool ValidateSelectBuffer(Context *context, GLsizei size, GLuint *buffer)
{
    return true;
}

bool ValidateTexCoord1d(Context *context, GLdouble s)
{
    return true;
}

bool ValidateTexCoord1dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateTexCoord1f(Context *context, GLfloat s)
{
    return true;
}

bool ValidateTexCoord1fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateTexCoord1i(Context *context, GLint s)
{
    return true;
}

bool ValidateTexCoord1iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateTexCoord1s(Context *context, GLshort s)
{
    return true;
}

bool ValidateTexCoord1sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateTexCoord2d(Context *context, GLdouble s, GLdouble t)
{
    return true;
}

bool ValidateTexCoord2dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateTexCoord2f(Context *context, GLfloat s, GLfloat t)
{
    return true;
}

bool ValidateTexCoord2fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateTexCoord2i(Context *context, GLint s, GLint t)
{
    return true;
}

bool ValidateTexCoord2iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateTexCoord2s(Context *context, GLshort s, GLshort t)
{
    return true;
}

bool ValidateTexCoord2sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateTexCoord3d(Context *context, GLdouble s, GLdouble t, GLdouble r)
{
    return true;
}

bool ValidateTexCoord3dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateTexCoord3f(Context *context, GLfloat s, GLfloat t, GLfloat r)
{
    return true;
}

bool ValidateTexCoord3fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateTexCoord3i(Context *context, GLint s, GLint t, GLint r)
{
    return true;
}

bool ValidateTexCoord3iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateTexCoord3s(Context *context, GLshort s, GLshort t, GLshort r)
{
    return true;
}

bool ValidateTexCoord3sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateTexCoord4d(Context *context, GLdouble s, GLdouble t, GLdouble r, GLdouble q)
{
    return true;
}

bool ValidateTexCoord4dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateTexCoord4f(Context *context, GLfloat s, GLfloat t, GLfloat r, GLfloat q)
{
    return true;
}

bool ValidateTexCoord4fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateTexCoord4i(Context *context, GLint s, GLint t, GLint r, GLint q)
{
    return true;
}

bool ValidateTexCoord4iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateTexCoord4s(Context *context, GLshort s, GLshort t, GLshort r, GLshort q)
{
    return true;
}

bool ValidateTexCoord4sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateTexGend(Context *context, GLenum coord, GLenum pname, GLdouble param)
{
    return true;
}

bool ValidateTexGendv(Context *context, GLenum coord, GLenum pname, const GLdouble *params)
{
    return true;
}

bool ValidateTexGenf(Context *context, GLenum coord, GLenum pname, GLfloat param)
{
    return true;
}
bool ValidateTexGenfv(Context *context, GLenum coord, GLenum pname, const GLfloat *params)
{
    return true;
}

bool ValidateTexGeni(Context *context, GLenum coord, GLenum pname, GLint param)
{
    return true;
}

bool ValidateTexGeniv(Context *context, GLenum coord, GLenum pname, const GLint *params)
{
    return true;
}

bool ValidateTexImage1D(Context *context,
                        GLenum target,
                        GLint level,
                        GLint internalformat,
                        GLsizei width,
                        GLint border,
                        GLenum format,
                        GLenum type,
                        const void *pixels)
{
    return true;
}

bool ValidateTranslated(Context *context, GLdouble x, GLdouble y, GLdouble z)
{
    return true;
}

bool ValidateVertex2d(Context *context, GLdouble x, GLdouble y)
{
    return true;
}

bool ValidateVertex2dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateVertex2f(Context *context, GLfloat x, GLfloat y)
{
    return true;
}

bool ValidateVertex2fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateVertex2i(Context *context, GLint x, GLint y)
{
    return true;
}

bool ValidateVertex2iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateVertex2s(Context *context, GLshort x, GLshort y)
{
    return true;
}

bool ValidateVertex2sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateVertex3d(Context *context, GLdouble x, GLdouble y, GLdouble z)
{
    return true;
}

bool ValidateVertex3dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateVertex3f(Context *context, GLfloat x, GLfloat y, GLfloat z)
{
    return true;
}

bool ValidateVertex3fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateVertex3i(Context *context, GLint x, GLint y, GLint z)
{
    return true;
}

bool ValidateVertex3iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateVertex3s(Context *context, GLshort x, GLshort y, GLshort z)
{
    return true;
}

bool ValidateVertex3sv(Context *context, const GLshort *v)
{
    return true;
}

bool ValidateVertex4d(Context *context, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    return true;
}

bool ValidateVertex4dv(Context *context, const GLdouble *v)
{
    return true;
}

bool ValidateVertex4f(Context *context, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    return true;
}

bool ValidateVertex4fv(Context *context, const GLfloat *v)
{
    return true;
}

bool ValidateVertex4i(Context *context, GLint x, GLint y, GLint z, GLint w)
{
    return true;
}

bool ValidateVertex4iv(Context *context, const GLint *v)
{
    return true;
}

bool ValidateVertex4s(Context *context, GLshort x, GLshort y, GLshort z, GLshort w)
{
    return true;
}

bool ValidateVertex4sv(Context *context, const GLshort *v)
{
    return true;
}

}  // namespace gl
