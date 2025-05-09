/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/* list of OpenGL functions sorted alphabetically
   If you need to use a GL function from the SDL video subsystem,
   change its entry from SDL_PROC_UNUSED to SDL_PROC and rebuild.
*/
#define SDL_PROC_UNUSED(ret, func, params)

SDL_PROC_UNUSED(void, glAccum, (GLenum, GLfloat))
SDL_PROC_UNUSED(void, glAlphaFunc, (GLenum, GLclampf))
SDL_PROC_UNUSED(GLboolean, glAreTexturesResident,
                (GLsizei, const GLuint *, GLboolean *))
SDL_PROC_UNUSED(void, glArrayElement, (GLint))
SDL_PROC(void, glBegin, (GLenum))
SDL_PROC(void, glBindTexture, (GLenum, GLuint))
SDL_PROC_UNUSED(void, glBitmap,
                (GLsizei, GLsizei, GLfloat, GLfloat, GLfloat, GLfloat,
                 const GLubyte *))
SDL_PROC(void, glBlendEquation, (GLenum))
SDL_PROC_UNUSED(void, glBlendFunc, (GLenum, GLenum))
SDL_PROC(void, glBlendFuncSeparate, (GLenum, GLenum, GLenum, GLenum))
SDL_PROC_UNUSED(void, glCallList, (GLuint))
SDL_PROC_UNUSED(void, glCallLists, (GLsizei, GLenum, const GLvoid *))
SDL_PROC(void, glClear, (GLbitfield))
SDL_PROC_UNUSED(void, glClearAccum, (GLfloat, GLfloat, GLfloat, GLfloat))
SDL_PROC(void, glClearColor, (GLclampf, GLclampf, GLclampf, GLclampf))
SDL_PROC_UNUSED(void, glClearDepth, (GLclampd))
SDL_PROC_UNUSED(void, glClearIndex, (GLfloat))
SDL_PROC_UNUSED(void, glClearStencil, (GLint))
SDL_PROC_UNUSED(void, glClipPlane, (GLenum, const GLdouble *))
SDL_PROC_UNUSED(void, glColor3b, (GLbyte, GLbyte, GLbyte))
SDL_PROC_UNUSED(void, glColor3bv, (const GLbyte *))
SDL_PROC_UNUSED(void, glColor3d, (GLdouble, GLdouble, GLdouble))
SDL_PROC_UNUSED(void, glColor3dv, (const GLdouble *))
SDL_PROC_UNUSED(void, glColor3f, (GLfloat, GLfloat, GLfloat))
SDL_PROC(void, glColor3fv, (const GLfloat *))
SDL_PROC_UNUSED(void, glColor3i, (GLint, GLint, GLint))
SDL_PROC_UNUSED(void, glColor3iv, (const GLint *))
SDL_PROC_UNUSED(void, glColor3s, (GLshort, GLshort, GLshort))
SDL_PROC_UNUSED(void, glColor3sv, (const GLshort *))
SDL_PROC_UNUSED(void, glColor3ub, (GLubyte, GLubyte, GLubyte))
SDL_PROC_UNUSED(void, glColor3ubv, (const GLubyte *))
SDL_PROC_UNUSED(void, glColor3ui, (GLuint, GLuint, GLuint))
SDL_PROC_UNUSED(void, glColor3uiv, (const GLuint *))
SDL_PROC_UNUSED(void, glColor3us, (GLushort, GLushort, GLushort))
SDL_PROC_UNUSED(void, glColor3usv, (const GLushort *))
SDL_PROC_UNUSED(void, glColor4b, (GLbyte, GLbyte, GLbyte, GLbyte))
SDL_PROC_UNUSED(void, glColor4bv, (const GLbyte *))
SDL_PROC_UNUSED(void, glColor4d, (GLdouble, GLdouble, GLdouble, GLdouble))
SDL_PROC_UNUSED(void, glColor4dv, (const GLdouble *))
SDL_PROC(void, glColor4f, (GLfloat, GLfloat, GLfloat, GLfloat))
SDL_PROC_UNUSED(void, glColor4fv, (const GLfloat *))
SDL_PROC_UNUSED(void, glColor4i, (GLint, GLint, GLint, GLint))
SDL_PROC_UNUSED(void, glColor4iv, (const GLint *))
SDL_PROC_UNUSED(void, glColor4s, (GLshort, GLshort, GLshort, GLshort))
SDL_PROC_UNUSED(void, glColor4sv, (const GLshort *))
SDL_PROC(void, glColor4ub,
         (GLubyte red, GLubyte green, GLubyte blue, GLubyte alpha))
SDL_PROC_UNUSED(void, glColor4ubv, (const GLubyte *v))
SDL_PROC_UNUSED(void, glColor4ui,
                (GLuint red, GLuint green, GLuint blue, GLuint alpha))
SDL_PROC_UNUSED(void, glColor4uiv, (const GLuint *v))
SDL_PROC_UNUSED(void, glColor4us,
                (GLushort red, GLushort green, GLushort blue, GLushort alpha))
SDL_PROC_UNUSED(void, glColor4usv, (const GLushort *v))
SDL_PROC_UNUSED(void, glColorMask,
                (GLboolean red, GLboolean green, GLboolean blue,
                 GLboolean alpha))
SDL_PROC_UNUSED(void, glColorMaterial, (GLenum face, GLenum mode))
SDL_PROC(void, glColorPointer,
         (GLint size, GLenum type, GLsizei stride,
          const GLvoid *pointer))
SDL_PROC_UNUSED(void, glCopyPixels,
                (GLint x, GLint y, GLsizei width, GLsizei height,
                 GLenum type))
SDL_PROC_UNUSED(void, glCopyTexImage1D,
                (GLenum target, GLint level, GLenum internalFormat, GLint x,
                 GLint y, GLsizei width, GLint border))
SDL_PROC_UNUSED(void, glCopyTexImage2D,
                (GLenum target, GLint level, GLenum internalFormat, GLint x,
                 GLint y, GLsizei width, GLsizei height, GLint border))
SDL_PROC_UNUSED(void, glCopyTexSubImage1D,
                (GLenum target, GLint level, GLint xoffset, GLint x, GLint y,
                 GLsizei width))
SDL_PROC_UNUSED(void, glCopyTexSubImage2D,
                (GLenum target, GLint level, GLint xoffset, GLint yoffset,
                 GLint x, GLint y, GLsizei width, GLsizei height))
SDL_PROC_UNUSED(void, glCullFace, (GLenum mode))
SDL_PROC_UNUSED(void, glDeleteLists, (GLuint list, GLsizei range))
SDL_PROC(void, glDeleteTextures, (GLsizei n, const GLuint *textures))
SDL_PROC(void, glDepthFunc, (GLenum func))
SDL_PROC_UNUSED(void, glDepthMask, (GLboolean flag))
SDL_PROC_UNUSED(void, glDepthRange, (GLclampd zNear, GLclampd zFar))
SDL_PROC(void, glDisable, (GLenum cap))
SDL_PROC(void, glDisableClientState, (GLenum array))
SDL_PROC(void, glDrawArrays, (GLenum mode, GLint first, GLsizei count))
SDL_PROC_UNUSED(void, glDrawBuffer, (GLenum mode))
SDL_PROC_UNUSED(void, glDrawElements,
                (GLenum mode, GLsizei count, GLenum type,
                 const GLvoid *indices))
SDL_PROC(void, glDrawPixels,
         (GLsizei width, GLsizei height, GLenum format, GLenum type,
          const GLvoid *pixels))
SDL_PROC_UNUSED(void, glEdgeFlag, (GLboolean flag))
SDL_PROC_UNUSED(void, glEdgeFlagPointer,
                (GLsizei stride, const GLvoid *pointer))
SDL_PROC_UNUSED(void, glEdgeFlagv, (const GLboolean *flag))
SDL_PROC(void, glEnable, (GLenum cap))
SDL_PROC(void, glEnableClientState, (GLenum array))
SDL_PROC(void, glEnd, (void))
SDL_PROC_UNUSED(void, glEndList, (void))
SDL_PROC_UNUSED(void, glEvalCoord1d, (GLdouble u))
SDL_PROC_UNUSED(void, glEvalCoord1dv, (const GLdouble *u))
SDL_PROC_UNUSED(void, glEvalCoord1f, (GLfloat u))
SDL_PROC_UNUSED(void, glEvalCoord1fv, (const GLfloat *u))
SDL_PROC_UNUSED(void, glEvalCoord2d, (GLdouble u, GLdouble v))
SDL_PROC_UNUSED(void, glEvalCoord2dv, (const GLdouble *u))
SDL_PROC_UNUSED(void, glEvalCoord2f, (GLfloat u, GLfloat v))
SDL_PROC_UNUSED(void, glEvalCoord2fv, (const GLfloat *u))
SDL_PROC_UNUSED(void, glEvalMesh1, (GLenum mode, GLint i1, GLint i2))
SDL_PROC_UNUSED(void, glEvalMesh2,
                (GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2))
SDL_PROC_UNUSED(void, glEvalPoint1, (GLint i))
SDL_PROC_UNUSED(void, glEvalPoint2, (GLint i, GLint j))
SDL_PROC_UNUSED(void, glFeedbackBuffer,
                (GLsizei size, GLenum type, GLfloat *buffer))
SDL_PROC_UNUSED(void, glFinish, (void))
SDL_PROC_UNUSED(void, glFlush, (void))
SDL_PROC_UNUSED(void, glFogf, (GLenum pname, GLfloat param))
SDL_PROC_UNUSED(void, glFogfv, (GLenum pname, const GLfloat *params))
SDL_PROC_UNUSED(void, glFogi, (GLenum pname, GLint param))
SDL_PROC_UNUSED(void, glFogiv, (GLenum pname, const GLint *params))
SDL_PROC_UNUSED(void, glFrontFace, (GLenum mode))
SDL_PROC_UNUSED(void, glFrustum,
                (GLdouble left, GLdouble right, GLdouble bottom,
                 GLdouble top, GLdouble zNear, GLdouble zFar))
SDL_PROC_UNUSED(GLuint, glGenLists, (GLsizei range))
SDL_PROC(void, glGenTextures, (GLsizei n, GLuint *textures))
SDL_PROC_UNUSED(void, glGetBooleanv, (GLenum pname, GLboolean *params))
SDL_PROC_UNUSED(void, glGetClipPlane, (GLenum plane, GLdouble *equation))
SDL_PROC_UNUSED(void, glGetDoublev, (GLenum pname, GLdouble *params))
SDL_PROC(GLenum, glGetError, (void))
SDL_PROC(void, glGetFloatv, (GLenum pname, GLfloat *params))
SDL_PROC(void, glGetIntegerv, (GLenum pname, GLint *params))
SDL_PROC_UNUSED(void, glGetLightfv,
                (GLenum light, GLenum pname, GLfloat *params))
SDL_PROC_UNUSED(void, glGetLightiv,
                (GLenum light, GLenum pname, GLint *params))
SDL_PROC_UNUSED(void, glGetMapdv, (GLenum target, GLenum query, GLdouble *v))
SDL_PROC_UNUSED(void, glGetMapfv, (GLenum target, GLenum query, GLfloat *v))
SDL_PROC_UNUSED(void, glGetMapiv, (GLenum target, GLenum query, GLint *v))
SDL_PROC_UNUSED(void, glGetMaterialfv,
                (GLenum face, GLenum pname, GLfloat *params))
SDL_PROC_UNUSED(void, glGetMaterialiv,
                (GLenum face, GLenum pname, GLint *params))
SDL_PROC_UNUSED(void, glGetPixelMapfv, (GLenum map, GLfloat *values))
SDL_PROC_UNUSED(void, glGetPixelMapuiv, (GLenum map, GLuint *values))
SDL_PROC_UNUSED(void, glGetPixelMapusv, (GLenum map, GLushort *values))
SDL_PROC(void, glGetPointerv, (GLenum pname, GLvoid **params))
SDL_PROC_UNUSED(void, glGetPolygonStipple, (GLubyte * mask))
SDL_PROC(const GLubyte *, glGetString, (GLenum name))
SDL_PROC_UNUSED(void, glGetTexEnvfv,
                (GLenum target, GLenum pname, GLfloat *params))
SDL_PROC_UNUSED(void, glGetTexEnviv,
                (GLenum target, GLenum pname, GLint *params))
SDL_PROC_UNUSED(void, glGetTexGendv,
                (GLenum coord, GLenum pname, GLdouble *params))
SDL_PROC_UNUSED(void, glGetTexGenfv,
                (GLenum coord, GLenum pname, GLfloat *params))
SDL_PROC_UNUSED(void, glGetTexGeniv,
                (GLenum coord, GLenum pname, GLint *params))
SDL_PROC_UNUSED(void, glGetTexImage,
                (GLenum target, GLint level, GLenum format, GLenum type,
                 GLvoid *pixels))
SDL_PROC_UNUSED(void, glGetTexLevelParameterfv,
                (GLenum target, GLint level, GLenum pname, GLfloat *params))
SDL_PROC_UNUSED(void, glGetTexLevelParameteriv,
                (GLenum target, GLint level, GLenum pname, GLint *params))
SDL_PROC_UNUSED(void, glGetTexParameterfv,
                (GLenum target, GLenum pname, GLfloat *params))
SDL_PROC_UNUSED(void, glGetTexParameteriv,
                (GLenum target, GLenum pname, GLint *params))
SDL_PROC_UNUSED(void, glHint, (GLenum target, GLenum mode))
SDL_PROC_UNUSED(void, glIndexMask, (GLuint mask))
SDL_PROC_UNUSED(void, glIndexPointer,
                (GLenum type, GLsizei stride, const GLvoid *pointer))
SDL_PROC_UNUSED(void, glIndexd, (GLdouble c))
SDL_PROC_UNUSED(void, glIndexdv, (const GLdouble *c))
SDL_PROC_UNUSED(void, glIndexf, (GLfloat c))
SDL_PROC_UNUSED(void, glIndexfv, (const GLfloat *c))
SDL_PROC_UNUSED(void, glIndexi, (GLint c))
SDL_PROC_UNUSED(void, glIndexiv, (const GLint *c))
SDL_PROC_UNUSED(void, glIndexs, (GLshort c))
SDL_PROC_UNUSED(void, glIndexsv, (const GLshort *c))
SDL_PROC_UNUSED(void, glIndexub, (GLubyte c))
SDL_PROC_UNUSED(void, glIndexubv, (const GLubyte *c))
SDL_PROC_UNUSED(void, glInitNames, (void))
SDL_PROC_UNUSED(void, glInterleavedArrays,
                (GLenum format, GLsizei stride, const GLvoid *pointer))
SDL_PROC_UNUSED(GLboolean, glIsEnabled, (GLenum cap))
SDL_PROC_UNUSED(GLboolean, glIsList, (GLuint list))
SDL_PROC_UNUSED(GLboolean, glIsTexture, (GLuint texture))
SDL_PROC_UNUSED(void, glLightModelf, (GLenum pname, GLfloat param))
SDL_PROC_UNUSED(void, glLightModelfv, (GLenum pname, const GLfloat *params))
SDL_PROC_UNUSED(void, glLightModeli, (GLenum pname, GLint param))
SDL_PROC_UNUSED(void, glLightModeliv, (GLenum pname, const GLint *params))
SDL_PROC_UNUSED(void, glLightf, (GLenum light, GLenum pname, GLfloat param))
SDL_PROC_UNUSED(void, glLightfv,
                (GLenum light, GLenum pname, const GLfloat *params))
SDL_PROC_UNUSED(void, glLighti, (GLenum light, GLenum pname, GLint param))
SDL_PROC_UNUSED(void, glLightiv,
                (GLenum light, GLenum pname, const GLint *params))
SDL_PROC_UNUSED(void, glLineStipple, (GLint factor, GLushort pattern))
SDL_PROC(void, glLineWidth, (GLfloat width))
SDL_PROC_UNUSED(void, glListBase, (GLuint base))
SDL_PROC(void, glLoadIdentity, (void))
SDL_PROC_UNUSED(void, glLoadMatrixd, (const GLdouble *m))
SDL_PROC_UNUSED(void, glLoadMatrixf, (const GLfloat *m))
SDL_PROC_UNUSED(void, glLoadName, (GLuint name))
SDL_PROC_UNUSED(void, glLogicOp, (GLenum opcode))
SDL_PROC_UNUSED(void, glMap1d,
                (GLenum target, GLdouble u1, GLdouble u2, GLint stride,
                 GLint order, const GLdouble *points))
SDL_PROC_UNUSED(void, glMap1f,
                (GLenum target, GLfloat u1, GLfloat u2, GLint stride,
                 GLint order, const GLfloat *points))
SDL_PROC_UNUSED(void, glMap2d,
                (GLenum target, GLdouble u1, GLdouble u2, GLint ustride,
                 GLint uorder, GLdouble v1, GLdouble v2, GLint vstride,
                 GLint vorder, const GLdouble *points))
SDL_PROC_UNUSED(void, glMap2f,
                (GLenum target, GLfloat u1, GLfloat u2, GLint ustride,
                 GLint uorder, GLfloat v1, GLfloat v2, GLint vstride,
                 GLint vorder, const GLfloat *points))
SDL_PROC_UNUSED(void, glMapGrid1d, (GLint un, GLdouble u1, GLdouble u2))
SDL_PROC_UNUSED(void, glMapGrid1f, (GLint un, GLfloat u1, GLfloat u2))
SDL_PROC_UNUSED(void, glMapGrid2d,
                (GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1,
                 GLdouble v2))
SDL_PROC_UNUSED(void, glMapGrid2f,
                (GLint un, GLfloat u1, GLfloat u2, GLint vn, GLfloat v1,
                 GLfloat v2))
SDL_PROC_UNUSED(void, glMaterialf, (GLenum face, GLenum pname, GLfloat param))
SDL_PROC_UNUSED(void, glMaterialfv,
                (GLenum face, GLenum pname, const GLfloat *params))
SDL_PROC_UNUSED(void, glMateriali, (GLenum face, GLenum pname, GLint param))
SDL_PROC_UNUSED(void, glMaterialiv,
                (GLenum face, GLenum pname, const GLint *params))
SDL_PROC(void, glMatrixMode, (GLenum mode))
SDL_PROC_UNUSED(void, glMultMatrixd, (const GLdouble *m))
SDL_PROC_UNUSED(void, glMultMatrixf, (const GLfloat *m))
SDL_PROC_UNUSED(void, glNewList, (GLuint list, GLenum mode))
SDL_PROC_UNUSED(void, glNormal3b, (GLbyte nx, GLbyte ny, GLbyte nz))
SDL_PROC_UNUSED(void, glNormal3bv, (const GLbyte *v))
SDL_PROC_UNUSED(void, glNormal3d, (GLdouble nx, GLdouble ny, GLdouble nz))
SDL_PROC_UNUSED(void, glNormal3dv, (const GLdouble *v))
SDL_PROC_UNUSED(void, glNormal3f, (GLfloat nx, GLfloat ny, GLfloat nz))
SDL_PROC_UNUSED(void, glNormal3fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glNormal3i, (GLint nx, GLint ny, GLint nz))
SDL_PROC_UNUSED(void, glNormal3iv, (const GLint *v))
SDL_PROC_UNUSED(void, glNormal3s, (GLshort nx, GLshort ny, GLshort nz))
SDL_PROC_UNUSED(void, glNormal3sv, (const GLshort *v))
SDL_PROC_UNUSED(void, glNormalPointer,
                (GLenum type, GLsizei stride, const GLvoid *pointer))
SDL_PROC(void, glOrtho,
         (GLdouble left, GLdouble right, GLdouble bottom, GLdouble top,
          GLdouble zNear, GLdouble zFar))
SDL_PROC_UNUSED(void, glPassThrough, (GLfloat token))
SDL_PROC_UNUSED(void, glPixelMapfv,
                (GLenum map, GLsizei mapsize, const GLfloat *values))
SDL_PROC_UNUSED(void, glPixelMapuiv,
                (GLenum map, GLsizei mapsize, const GLuint *values))
SDL_PROC_UNUSED(void, glPixelMapusv,
                (GLenum map, GLsizei mapsize, const GLushort *values))
SDL_PROC_UNUSED(void, glPixelStoref, (GLenum pname, GLfloat param))
SDL_PROC(void, glPixelStorei, (GLenum pname, GLint param))
SDL_PROC_UNUSED(void, glPixelTransferf, (GLenum pname, GLfloat param))
SDL_PROC_UNUSED(void, glPixelTransferi, (GLenum pname, GLint param))
SDL_PROC_UNUSED(void, glPixelZoom, (GLfloat xfactor, GLfloat yfactor))
SDL_PROC(void, glPointSize, (GLfloat size))
SDL_PROC_UNUSED(void, glPolygonMode, (GLenum face, GLenum mode))
SDL_PROC_UNUSED(void, glPolygonOffset, (GLfloat factor, GLfloat units))
SDL_PROC_UNUSED(void, glPolygonStipple, (const GLubyte *mask))
SDL_PROC_UNUSED(void, glPopAttrib, (void))
SDL_PROC_UNUSED(void, glPopClientAttrib, (void))
SDL_PROC_UNUSED(void, glPopMatrix, (void))
SDL_PROC_UNUSED(void, glPopName, (void))
SDL_PROC_UNUSED(void, glPrioritizeTextures,
                (GLsizei n, const GLuint *textures,
                 const GLclampf *priorities))
SDL_PROC_UNUSED(void, glPushAttrib, (GLbitfield mask))
SDL_PROC_UNUSED(void, glPushClientAttrib, (GLbitfield mask))
SDL_PROC_UNUSED(void, glPushMatrix, (void))
SDL_PROC_UNUSED(void, glPushName, (GLuint name))
SDL_PROC_UNUSED(void, glRasterPos2d, (GLdouble x, GLdouble y))
SDL_PROC_UNUSED(void, glRasterPos2dv, (const GLdouble *v))
SDL_PROC_UNUSED(void, glRasterPos2f, (GLfloat x, GLfloat y))
SDL_PROC_UNUSED(void, glRasterPos2fv, (const GLfloat *v))
SDL_PROC(void, glRasterPos2i, (GLint x, GLint y))
SDL_PROC_UNUSED(void, glRasterPos2iv, (const GLint *v))
SDL_PROC_UNUSED(void, glRasterPos2s, (GLshort x, GLshort y))
SDL_PROC_UNUSED(void, glRasterPos2sv, (const GLshort *v))
SDL_PROC_UNUSED(void, glRasterPos3d, (GLdouble x, GLdouble y, GLdouble z))
SDL_PROC_UNUSED(void, glRasterPos3dv, (const GLdouble *v))
SDL_PROC_UNUSED(void, glRasterPos3f, (GLfloat x, GLfloat y, GLfloat z))
SDL_PROC_UNUSED(void, glRasterPos3fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glRasterPos3i, (GLint x, GLint y, GLint z))
SDL_PROC_UNUSED(void, glRasterPos3iv, (const GLint *v))
SDL_PROC_UNUSED(void, glRasterPos3s, (GLshort x, GLshort y, GLshort z))
SDL_PROC_UNUSED(void, glRasterPos3sv, (const GLshort *v))
SDL_PROC_UNUSED(void, glRasterPos4d,
                (GLdouble x, GLdouble y, GLdouble z, GLdouble w))
SDL_PROC_UNUSED(void, glRasterPos4dv, (const GLdouble *v))
SDL_PROC_UNUSED(void, glRasterPos4f,
                (GLfloat x, GLfloat y, GLfloat z, GLfloat w))
SDL_PROC_UNUSED(void, glRasterPos4fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glRasterPos4i, (GLint x, GLint y, GLint z, GLint w))
SDL_PROC_UNUSED(void, glRasterPos4iv, (const GLint *v))
SDL_PROC_UNUSED(void, glRasterPos4s,
                (GLshort x, GLshort y, GLshort z, GLshort w))
SDL_PROC_UNUSED(void, glRasterPos4sv, (const GLshort *v))
SDL_PROC(void, glReadBuffer, (GLenum mode))
SDL_PROC(void, glReadPixels,
         (GLint x, GLint y, GLsizei width, GLsizei height,
          GLenum format, GLenum type, GLvoid *pixels))
SDL_PROC_UNUSED(void, glRectd,
                (GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2))
SDL_PROC_UNUSED(void, glRectdv, (const GLdouble *v1, const GLdouble *v2))
SDL_PROC(void, glRectf,
         (GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2))
SDL_PROC_UNUSED(void, glRectfv, (const GLfloat *v1, const GLfloat *v2))
SDL_PROC_UNUSED(void, glRecti, (GLint x1, GLint y1, GLint x2, GLint y2))
SDL_PROC_UNUSED(void, glRectiv, (const GLint *v1, const GLint *v2))
SDL_PROC_UNUSED(void, glRects,
                (GLshort x1, GLshort y1, GLshort x2, GLshort y2))
SDL_PROC_UNUSED(void, glRectsv, (const GLshort *v1, const GLshort *v2))
SDL_PROC_UNUSED(GLint, glRenderMode, (GLenum mode))
SDL_PROC_UNUSED(void, glRotated,
                (GLdouble angle, GLdouble x, GLdouble y, GLdouble z))
SDL_PROC(void, glRotatef,
         (GLfloat angle, GLfloat x, GLfloat y, GLfloat z))
SDL_PROC_UNUSED(void, glScaled, (GLdouble x, GLdouble y, GLdouble z))
SDL_PROC_UNUSED(void, glScalef, (GLfloat x, GLfloat y, GLfloat z))
SDL_PROC(void, glScissor, (GLint x, GLint y, GLsizei width, GLsizei height))
SDL_PROC_UNUSED(void, glSelectBuffer, (GLsizei size, GLuint *buffer))
SDL_PROC(void, glShadeModel, (GLenum mode))
SDL_PROC_UNUSED(void, glStencilFunc, (GLenum func, GLint ref, GLuint mask))
SDL_PROC_UNUSED(void, glStencilMask, (GLuint mask))
SDL_PROC_UNUSED(void, glStencilOp, (GLenum fail, GLenum zfail, GLenum zpass))
SDL_PROC_UNUSED(void, glTexCoord1d, (GLdouble s))
SDL_PROC_UNUSED(void, glTexCoord1dv, (const GLdouble *v))
SDL_PROC_UNUSED(void, glTexCoord1f, (GLfloat s))
SDL_PROC_UNUSED(void, glTexCoord1fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glTexCoord1i, (GLint s))
SDL_PROC_UNUSED(void, glTexCoord1iv, (const GLint *v))
SDL_PROC_UNUSED(void, glTexCoord1s, (GLshort s))
SDL_PROC_UNUSED(void, glTexCoord1sv, (const GLshort *v))
SDL_PROC_UNUSED(void, glTexCoord2d, (GLdouble s, GLdouble t))
SDL_PROC_UNUSED(void, glTexCoord2dv, (const GLdouble *v))
SDL_PROC(void, glTexCoord2f, (GLfloat s, GLfloat t))
SDL_PROC_UNUSED(void, glTexCoord2fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glTexCoord2i, (GLint s, GLint t))
SDL_PROC_UNUSED(void, glTexCoord2iv, (const GLint *v))
SDL_PROC_UNUSED(void, glTexCoord2s, (GLshort s, GLshort t))
SDL_PROC_UNUSED(void, glTexCoord2sv, (const GLshort *v))
SDL_PROC_UNUSED(void, glTexCoord3d, (GLdouble s, GLdouble t, GLdouble r))
SDL_PROC_UNUSED(void, glTexCoord3dv, (const GLdouble *v))
SDL_PROC_UNUSED(void, glTexCoord3f, (GLfloat s, GLfloat t, GLfloat r))
SDL_PROC_UNUSED(void, glTexCoord3fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glTexCoord3i, (GLint s, GLint t, GLint r))
SDL_PROC_UNUSED(void, glTexCoord3iv, (const GLint *v))
SDL_PROC_UNUSED(void, glTexCoord3s, (GLshort s, GLshort t, GLshort r))
SDL_PROC_UNUSED(void, glTexCoord3sv, (const GLshort *v))
SDL_PROC_UNUSED(void, glTexCoord4d,
                (GLdouble s, GLdouble t, GLdouble r, GLdouble q))
SDL_PROC_UNUSED(void, glTexCoord4dv, (const GLdouble *v))
SDL_PROC_UNUSED(void, glTexCoord4f,
                (GLfloat s, GLfloat t, GLfloat r, GLfloat q))
SDL_PROC_UNUSED(void, glTexCoord4fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glTexCoord4i, (GLint s, GLint t, GLint r, GLint q))
SDL_PROC_UNUSED(void, glTexCoord4iv, (const GLint *v))
SDL_PROC_UNUSED(void, glTexCoord4s,
                (GLshort s, GLshort t, GLshort r, GLshort q))
SDL_PROC_UNUSED(void, glTexCoord4sv, (const GLshort *v))
SDL_PROC(void, glTexCoordPointer,
         (GLint size, GLenum type, GLsizei stride,
          const GLvoid *pointer))
SDL_PROC(void, glTexEnvf, (GLenum target, GLenum pname, GLfloat param))
SDL_PROC_UNUSED(void, glTexEnvfv,
                (GLenum target, GLenum pname, const GLfloat *params))
SDL_PROC_UNUSED(void, glTexEnvi, (GLenum target, GLenum pname, GLint param))
SDL_PROC_UNUSED(void, glTexEnviv,
                (GLenum target, GLenum pname, const GLint *params))
SDL_PROC_UNUSED(void, glTexGend, (GLenum coord, GLenum pname, GLdouble param))
SDL_PROC_UNUSED(void, glTexGendv,
                (GLenum coord, GLenum pname, const GLdouble *params))
SDL_PROC_UNUSED(void, glTexGenf, (GLenum coord, GLenum pname, GLfloat param))
SDL_PROC_UNUSED(void, glTexGenfv,
                (GLenum coord, GLenum pname, const GLfloat *params))
SDL_PROC_UNUSED(void, glTexGeni, (GLenum coord, GLenum pname, GLint param))
SDL_PROC_UNUSED(void, glTexGeniv,
                (GLenum coord, GLenum pname, const GLint *params))
SDL_PROC_UNUSED(void, glTexImage1D,
                (GLenum target, GLint level, GLint internalformat,
                 GLsizei width, GLint border, GLenum format, GLenum type,
                 const GLvoid *pixels))
SDL_PROC(void, glTexImage2D,
         (GLenum target, GLint level, GLint internalformat, GLsizei width,
          GLsizei height, GLint border, GLenum format, GLenum type,
          const GLvoid *pixels))
SDL_PROC_UNUSED(void, glTexParameterf,
                (GLenum target, GLenum pname, GLfloat param))
SDL_PROC_UNUSED(void, glTexParameterfv,
                (GLenum target, GLenum pname, const GLfloat *params))
SDL_PROC(void, glTexParameteri, (GLenum target, GLenum pname, GLint param))
SDL_PROC_UNUSED(void, glTexParameteriv,
                (GLenum target, GLenum pname, const GLint *params))
SDL_PROC_UNUSED(void, glTexSubImage1D,
                (GLenum target, GLint level, GLint xoffset, GLsizei width,
                 GLenum format, GLenum type, const GLvoid *pixels))
SDL_PROC(void, glTexSubImage2D,
         (GLenum target, GLint level, GLint xoffset, GLint yoffset,
          GLsizei width, GLsizei height, GLenum format, GLenum type,
          const GLvoid *pixels))
SDL_PROC_UNUSED(void, glTranslated, (GLdouble x, GLdouble y, GLdouble z))
SDL_PROC_UNUSED(void, glTranslatef, (GLfloat x, GLfloat y, GLfloat z))
SDL_PROC_UNUSED(void, glVertex2d, (GLdouble x, GLdouble y))
SDL_PROC_UNUSED(void, glVertex2dv, (const GLdouble *v))
SDL_PROC(void, glVertex2f, (GLfloat x, GLfloat y))
SDL_PROC_UNUSED(void, glVertex2fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glVertex2i, (GLint x, GLint y))
SDL_PROC_UNUSED(void, glVertex2iv, (const GLint *v))
SDL_PROC_UNUSED(void, glVertex2s, (GLshort x, GLshort y))
SDL_PROC_UNUSED(void, glVertex2sv, (const GLshort *v))
SDL_PROC_UNUSED(void, glVertex3d, (GLdouble x, GLdouble y, GLdouble z))
SDL_PROC_UNUSED(void, glVertex3dv, (const GLdouble *v))
SDL_PROC_UNUSED(void, glVertex3f, (GLfloat x, GLfloat y, GLfloat z))
SDL_PROC(void, glVertex3fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glVertex3i, (GLint x, GLint y, GLint z))
SDL_PROC_UNUSED(void, glVertex3iv, (const GLint *v))
SDL_PROC_UNUSED(void, glVertex3s, (GLshort x, GLshort y, GLshort z))
SDL_PROC_UNUSED(void, glVertex3sv, (const GLshort *v))
SDL_PROC_UNUSED(void, glVertex4d,
                (GLdouble x, GLdouble y, GLdouble z, GLdouble w))
SDL_PROC_UNUSED(void, glVertex4dv, (const GLdouble *v))
SDL_PROC_UNUSED(void, glVertex4f,
                (GLfloat x, GLfloat y, GLfloat z, GLfloat w))
SDL_PROC_UNUSED(void, glVertex4fv, (const GLfloat *v))
SDL_PROC_UNUSED(void, glVertex4i, (GLint x, GLint y, GLint z, GLint w))
SDL_PROC_UNUSED(void, glVertex4iv, (const GLint *v))
SDL_PROC_UNUSED(void, glVertex4s,
                (GLshort x, GLshort y, GLshort z, GLshort w))
SDL_PROC_UNUSED(void, glVertex4sv, (const GLshort *v))
SDL_PROC(void, glVertexPointer,
         (GLint size, GLenum type, GLsizei stride,
          const GLvoid *pointer))
SDL_PROC(void, glViewport, (GLint x, GLint y, GLsizei width, GLsizei height))
