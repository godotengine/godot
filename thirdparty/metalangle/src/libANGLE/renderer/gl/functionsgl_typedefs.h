//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// functionsgl_typedefs.h: Typedefs of OpenGL types and functions for versions 1.0 through 4.5.

#ifndef LIBANGLE_RENDERER_GL_FUNCTIONSGLTYPEDEFS_H_
#define LIBANGLE_RENDERER_GL_FUNCTIONSGLTYPEDEFS_H_

#include "common/platform.h"

#include <KHR/khrplatform.h>
#include <stdint.h>

#ifndef INTERNAL_GL_APIENTRY
#    ifdef ANGLE_PLATFORM_WINDOWS
#        define INTERNAL_GL_APIENTRY __stdcall
#    else
#        define INTERNAL_GL_APIENTRY
#    endif
#endif

typedef void GLvoid;
typedef char GLchar;
typedef unsigned int GLenum;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef khronos_int8_t GLbyte;
typedef short GLshort;
typedef int GLint;
typedef int GLsizei;
typedef khronos_uint8_t GLubyte;
typedef unsigned short GLushort;
typedef unsigned int GLuint;
typedef khronos_float_t GLfloat;
typedef khronos_float_t GLclampf;
typedef double GLdouble;
typedef double GLclampd;
typedef khronos_int32_t GLfixed;
typedef khronos_intptr_t GLintptr;
typedef khronos_ssize_t GLsizeiptr;
typedef unsigned short GLhalf;
typedef khronos_int64_t GLint64;
typedef khronos_uint64_t GLuint64;
typedef struct __GLsync *GLsync;

// TODO(jmadill): It's likely we can auto-generate this file from gl.xml.

namespace rx
{
typedef void(INTERNAL_GL_APIENTRY *GLDEBUGPROC)(GLenum source,
                                                GLenum type,
                                                GLuint id,
                                                GLenum severity,
                                                GLsizei length,
                                                const GLchar *message,
                                                const void *userParam);
typedef void(INTERNAL_GL_APIENTRY *GLDEBUGPROCARB)(GLenum source,
                                                   GLenum type,
                                                   GLuint id,
                                                   GLenum severity,
                                                   GLsizei length,
                                                   const GLchar *message,
                                                   const void *userParam);
typedef void(INTERNAL_GL_APIENTRY *GLDEBUGPROCAMD)(GLuint id,
                                                   GLenum category,
                                                   GLenum severity,
                                                   GLsizei length,
                                                   const GLchar *message,
                                                   void *userParam);

// 1.0
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLENDFUNCPROC)(GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARPROC)(GLbitfield);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARCOLORPROC)(GLfloat, GLfloat, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARDEPTHPROC)(GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARSTENCILPROC)(GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOLORMASKPROC)(GLboolean, GLboolean, GLboolean, GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCULLFACEPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDEPTHFUNCPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDEPTHMASKPROC)(GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDEPTHRANGEPROC)(GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDISABLEPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWBUFFERPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLENABLEPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLFINISHPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLFLUSHPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLFRONTFACEPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETBOOLEANVPROC)(GLenum, GLboolean *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETDOUBLEVPROC)(GLenum, GLdouble *);
typedef GLenum(INTERNAL_GL_APIENTRY *PFNGLGETERRORPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETFLOATVPROC)(GLenum, GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETINTEGERVPROC)(GLenum, GLint *);
typedef const GLubyte *(INTERNAL_GL_APIENTRY *PFNGLGETSTRINGPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXIMAGEPROC)(GLenum, GLint, GLenum, GLenum, GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXLEVELPARAMETERFVPROC)(GLenum,
                                                                    GLint,
                                                                    GLenum,
                                                                    GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXLEVELPARAMETERIVPROC)(GLenum, GLint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXPARAMETERFVPROC)(GLenum, GLenum, GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXPARAMETERIVPROC)(GLenum, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLHINTPROC)(GLenum, GLenum);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISENABLEDPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLLINEWIDTHPROC)(GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLLOGICOPPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPIXELSTOREFPROC)(GLenum, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPIXELSTOREIPROC)(GLenum, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPOINTSIZEPROC)(GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPOLYGONMODEPROC)(GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLREADBUFFERPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY
                 *PFNGLREADPIXELSPROC)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSCISSORPROC)(GLint, GLint, GLsizei, GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILFUNCPROC)(GLenum, GLint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILMASKPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILOPPROC)(GLenum, GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXIMAGE1DPROC)(GLenum,
                                                        GLint,
                                                        GLint,
                                                        GLsizei,
                                                        GLint,
                                                        GLenum,
                                                        GLenum,
                                                        const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXIMAGE2DPROC)(GLenum,
                                                        GLint,
                                                        GLint,
                                                        GLsizei,
                                                        GLsizei,
                                                        GLint,
                                                        GLenum,
                                                        GLenum,
                                                        const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXPARAMETERFPROC)(GLenum, GLenum, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXPARAMETERFVPROC)(GLenum, GLenum, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXPARAMETERIPROC)(GLenum, GLenum, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXPARAMETERIVPROC)(GLenum, GLenum, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVIEWPORTPROC)(GLint, GLint, GLsizei, GLsizei);

// 1.1
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDTEXTUREPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY
                 *PFNGLCOPYTEXIMAGE1DPROC)(GLenum, GLint, GLenum, GLint, GLint, GLsizei, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOPYTEXIMAGE2DPROC)(GLenum,
                                                            GLint,
                                                            GLenum,
                                                            GLint,
                                                            GLint,
                                                            GLsizei,
                                                            GLsizei,
                                                            GLint);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLCOPYTEXSUBIMAGE1DPROC)(GLenum, GLint, GLint, GLint, GLint, GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOPYTEXSUBIMAGE2DPROC)(GLenum,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLsizei,
                                                               GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETETEXTURESPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWARRAYSPROC)(GLenum, GLint, GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWELEMENTSPROC)(GLenum, GLsizei, GLenum, const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENTEXTURESPROC)(GLsizei, GLuint *);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISTEXTUREPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPOLYGONOFFSETPROC)(GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSUBIMAGE1DPROC)(GLenum,
                                                           GLint,
                                                           GLint,
                                                           GLsizei,
                                                           GLenum,
                                                           GLenum,
                                                           const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSUBIMAGE2DPROC)(GLenum,
                                                           GLint,
                                                           GLint,
                                                           GLint,
                                                           GLsizei,
                                                           GLsizei,
                                                           GLenum,
                                                           GLenum,
                                                           const GLvoid *);

// 1.2
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLENDCOLORPROC)(GLfloat, GLfloat, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLENDEQUATIONPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOPYTEXSUBIMAGE3DPROC)(GLenum,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLsizei,
                                                               GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWRANGEELEMENTSPROC)(GLenum,
                                                               GLuint,
                                                               GLuint,
                                                               GLsizei,
                                                               GLenum,
                                                               const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXIMAGE3DPROC)(GLenum,
                                                        GLint,
                                                        GLint,
                                                        GLsizei,
                                                        GLsizei,
                                                        GLsizei,
                                                        GLint,
                                                        GLenum,
                                                        GLenum,
                                                        const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSUBIMAGE3DPROC)(GLenum,
                                                           GLint,
                                                           GLint,
                                                           GLint,
                                                           GLint,
                                                           GLsizei,
                                                           GLsizei,
                                                           GLsizei,
                                                           GLenum,
                                                           GLenum,
                                                           const GLvoid *);

// 1.2 Extensions
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETEFENCESNVPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENFENCESNVPROC)(GLsizei, GLuint *);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISFENCENVPROC)(GLuint);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLTESTFENCENVPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETFENCEIVNVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLFINISHFENCENVPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSETFENCENVPROC)(GLuint, GLenum);

// 1.3
typedef void(INTERNAL_GL_APIENTRY *PFNGLACTIVETEXTUREPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPRESSEDTEXIMAGE1DPROC)(GLenum,
                                                                  GLint,
                                                                  GLenum,
                                                                  GLsizei,
                                                                  GLint,
                                                                  GLsizei,
                                                                  const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPRESSEDTEXIMAGE2DPROC)(GLenum,
                                                                  GLint,
                                                                  GLenum,
                                                                  GLsizei,
                                                                  GLsizei,
                                                                  GLint,
                                                                  GLsizei,
                                                                  const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPRESSEDTEXIMAGE3DPROC)(GLenum,
                                                                  GLint,
                                                                  GLenum,
                                                                  GLsizei,
                                                                  GLsizei,
                                                                  GLsizei,
                                                                  GLint,
                                                                  GLsizei,
                                                                  const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC)(GLenum,
                                                                     GLint,
                                                                     GLint,
                                                                     GLsizei,
                                                                     GLenum,
                                                                     GLsizei,
                                                                     const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC)(GLenum,
                                                                     GLint,
                                                                     GLint,
                                                                     GLint,
                                                                     GLsizei,
                                                                     GLsizei,
                                                                     GLenum,
                                                                     GLsizei,
                                                                     const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC)(GLenum,
                                                                     GLint,
                                                                     GLint,
                                                                     GLint,
                                                                     GLint,
                                                                     GLsizei,
                                                                     GLsizei,
                                                                     GLsizei,
                                                                     GLenum,
                                                                     GLsizei,
                                                                     const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETCOMPRESSEDTEXIMAGEPROC)(GLenum, GLint, GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSAMPLECOVERAGEPROC)(GLfloat, GLboolean);

// 1.4
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLENDFUNCSEPARATEPROC)(GLenum, GLenum, GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMULTIDRAWARRAYSPROC)(GLenum,
                                                             const GLint *,
                                                             const GLsizei *,
                                                             GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMULTIDRAWELEMENTSPROC)(GLenum,
                                                               const GLsizei *,
                                                               GLenum,
                                                               const GLvoid *const *,
                                                               GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPOINTPARAMETERFPROC)(GLenum, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPOINTPARAMETERFVPROC)(GLenum, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPOINTPARAMETERIPROC)(GLenum, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPOINTPARAMETERIVPROC)(GLenum, const GLint *);

// 1.5
typedef void(INTERNAL_GL_APIENTRY *PFNGLBEGINQUERYPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDBUFFERPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBUFFERDATAPROC)(GLenum, GLsizeiptr, const GLvoid *, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBUFFERSUBDATAPROC)(GLenum,
                                                           GLintptr,
                                                           GLsizeiptr,
                                                           const GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETEQUERIESPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLENDQUERYPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENQUERIESPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETBUFFERPARAMETERIVPROC)(GLenum, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETBUFFERPOINTERVPROC)(GLenum, GLenum, GLvoid **);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETBUFFERSUBDATAPROC)(GLenum,
                                                              GLintptr,
                                                              GLsizeiptr,
                                                              GLvoid *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYOBJECTIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYOBJECTUIVPROC)(GLuint, GLenum, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYIVPROC)(GLenum, GLenum, GLint *);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISBUFFERPROC)(GLuint);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISQUERYPROC)(GLuint);
typedef void *(INTERNAL_GL_APIENTRY *PFNGLMAPBUFFERPROC)(GLenum, GLenum);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLUNMAPBUFFERPROC)(GLenum);

// 2.0
typedef void(INTERNAL_GL_APIENTRY *PFNGLATTACHSHADERPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDATTRIBLOCATIONPROC)(GLuint, GLuint, const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLENDEQUATIONSEPARATEPROC)(GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint);
typedef GLuint(INTERNAL_GL_APIENTRY *PFNGLCREATEPROGRAMPROC)();
typedef GLuint(INTERNAL_GL_APIENTRY *PFNGLCREATESHADERPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETEPROGRAMPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETESHADERPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDETACHSHADERPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDISABLEVERTEXATTRIBARRAYPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWBUFFERSPROC)(GLsizei, const GLenum *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVEATTRIBPROC)(GLuint,
                                                             GLuint,
                                                             GLsizei,
                                                             GLsizei *,
                                                             GLint *,
                                                             GLenum *,
                                                             GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVEUNIFORMPROC)(GLuint,
                                                              GLuint,
                                                              GLsizei,
                                                              GLsizei *,
                                                              GLint *,
                                                              GLenum *,
                                                              GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETATTACHEDSHADERSPROC)(GLuint,
                                                                GLsizei,
                                                                GLsizei *,
                                                                GLuint *);
typedef GLint(INTERNAL_GL_APIENTRY *PFNGLGETATTRIBLOCATIONPROC)(GLuint, const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)(GLuint,
                                                               GLsizei,
                                                               GLsizei *,
                                                               GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint, GLsizei, GLsizei *, GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSHADERSOURCEPROC)(GLuint, GLsizei, GLsizei *, GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSHADERIVPROC)(GLuint, GLenum, GLint *);
typedef GLint(INTERNAL_GL_APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)(GLuint, const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETUNIFORMFVPROC)(GLuint, GLint, GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETUNIFORMIVPROC)(GLuint, GLint, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXATTRIBPOINTERVPROC)(GLuint, GLenum, GLvoid **);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXATTRIBDVPROC)(GLuint, GLenum, GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXATTRIBFVPROC)(GLuint, GLenum, GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXATTRIBIVPROC)(GLuint, GLenum, GLint *);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISPROGRAMPROC)(GLuint);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISSHADERPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint,
                                                          GLsizei,
                                                          const GLchar *const *,
                                                          const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILFUNCSEPARATEPROC)(GLenum, GLenum, GLint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILMASKSEPARATEPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILOPSEPARATEPROC)(GLenum, GLenum, GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM1FPROC)(GLint, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM1FVPROC)(GLint, GLsizei, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM1IPROC)(GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM1IVPROC)(GLint, GLsizei, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM2FPROC)(GLint, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM2FVPROC)(GLint, GLsizei, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM2IPROC)(GLint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM2IVPROC)(GLint, GLsizei, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM3FPROC)(GLint, GLfloat, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM3FVPROC)(GLint, GLsizei, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM3IPROC)(GLint, GLint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM3IVPROC)(GLint, GLsizei, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM4FPROC)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM4FVPROC)(GLint, GLsizei, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM4IPROC)(GLint, GLint, GLint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM4IVPROC)(GLint, GLsizei, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX2FVPROC)(GLint,
                                                              GLsizei,
                                                              GLboolean,
                                                              const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX3FVPROC)(GLint,
                                                              GLsizei,
                                                              GLboolean,
                                                              const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX4FVPROC)(GLint,
                                                              GLsizei,
                                                              GLboolean,
                                                              const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVALIDATEPROGRAMPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB1DPROC)(GLuint, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB1DVPROC)(GLuint, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB1FPROC)(GLuint, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB1FVPROC)(GLuint, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB1SPROC)(GLuint, GLshort);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB1SVPROC)(GLuint, const GLshort *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB2DPROC)(GLuint, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB2DVPROC)(GLuint, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB2FPROC)(GLuint, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB2FVPROC)(GLuint, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB2SPROC)(GLuint, GLshort, GLshort);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB2SVPROC)(GLuint, const GLshort *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB3DPROC)(GLuint, GLdouble, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB3DVPROC)(GLuint, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB3FPROC)(GLuint, GLfloat, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB3FVPROC)(GLuint, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB3SPROC)(GLuint, GLshort, GLshort, GLshort);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB3SVPROC)(GLuint, const GLshort *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4NBVPROC)(GLuint, const GLbyte *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4NIVPROC)(GLuint, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4NSVPROC)(GLuint, const GLshort *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4NUBPROC)(GLuint, GLubyte, GLubyte, GLubyte, GLubyte);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4NUBVPROC)(GLuint, const GLubyte *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4NUIVPROC)(GLuint, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4NUSVPROC)(GLuint, const GLushort *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4BVPROC)(GLuint, const GLbyte *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4DPROC)(GLuint, GLdouble, GLdouble, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4DVPROC)(GLuint, const GLdouble *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4FPROC)(GLuint, GLfloat, GLfloat, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4FVPROC)(GLuint, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4IVPROC)(GLuint, const GLint *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4SPROC)(GLuint, GLshort, GLshort, GLshort, GLshort);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4SVPROC)(GLuint, const GLshort *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4UBVPROC)(GLuint, const GLubyte *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4UIVPROC)(GLuint, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIB4USVPROC)(GLuint, const GLushort *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint,
                                                                 GLint,
                                                                 GLenum,
                                                                 GLboolean,
                                                                 GLsizei,
                                                                 const GLvoid *);

// 2.1
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX2X3FVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX2X4FVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX3X2FVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX3X4FVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX4X2FVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX4X3FVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLfloat *);

// 3.0
typedef void(INTERNAL_GL_APIENTRY *PFNGLBEGINCONDITIONALRENDERPROC)(GLuint, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBEGINTRANSFORMFEEDBACKPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDBUFFERBASEPROC)(GLenum, GLuint, GLuint);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLBINDBUFFERRANGEPROC)(GLenum, GLuint, GLuint, GLintptr, GLsizeiptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDFRAGDATALOCATIONPROC)(GLuint, GLuint, const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDFRAMEBUFFERPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDRENDERBUFFERPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDVERTEXARRAYPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLITFRAMEBUFFERPROC)(GLint,
                                                             GLint,
                                                             GLint,
                                                             GLint,
                                                             GLint,
                                                             GLint,
                                                             GLint,
                                                             GLint,
                                                             GLbitfield,
                                                             GLenum);
typedef GLenum(INTERNAL_GL_APIENTRY *PFNGLCHECKFRAMEBUFFERSTATUSPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLAMPCOLORPROC)(GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARBUFFERFIPROC)(GLenum, GLint, GLfloat, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARBUFFERFVPROC)(GLenum, GLint, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARBUFFERIVPROC)(GLenum, GLint, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARBUFFERUIVPROC)(GLenum, GLint, const GLuint *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLCOLORMASKIPROC)(GLuint, GLboolean, GLboolean, GLboolean, GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETERENDERBUFFERSPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETEVERTEXARRAYSPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDISABLEIPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLENABLEIPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLENDCONDITIONALRENDERPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLENDTRANSFORMFEEDBACKPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLFLUSHMAPPEDBUFFERRANGEPROC)(GLenum, GLintptr, GLsizeiptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLFRAMEBUFFERRENDERBUFFERPROC)(GLenum,
                                                                     GLenum,
                                                                     GLenum,
                                                                     GLuint);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLFRAMEBUFFERTEXTURE1DPROC)(GLenum, GLenum, GLenum, GLuint, GLint);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum, GLenum, GLenum, GLuint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLFRAMEBUFFERTEXTURE3DPROC)(GLenum,
                                                                  GLenum,
                                                                  GLenum,
                                                                  GLuint,
                                                                  GLint,
                                                                  GLint);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLFRAMEBUFFERTEXTURELAYERPROC)(GLenum, GLenum, GLuint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENFRAMEBUFFERSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENRENDERBUFFERSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENVERTEXARRAYSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENERATEMIPMAPPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETBOOLEANI_VPROC)(GLenum, GLuint, GLboolean *);
typedef GLint(INTERNAL_GL_APIENTRY *PFNGLGETFRAGDATALOCATIONPROC)(GLuint, const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC)(GLenum,
                                                                                 GLenum,
                                                                                 GLenum,
                                                                                 GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETINTEGERI_VPROC)(GLenum, GLuint, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETRENDERBUFFERPARAMETERIVPROC)(GLenum, GLenum, GLint *);
typedef const GLubyte *(INTERNAL_GL_APIENTRY *PFNGLGETSTRINGIPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXPARAMETERIIVPROC)(GLenum, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXPARAMETERIUIVPROC)(GLenum, GLenum, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTRANSFORMFEEDBACKVARYINGPROC)(GLuint,
                                                                         GLuint,
                                                                         GLsizei,
                                                                         GLsizei *,
                                                                         GLsizei *,
                                                                         GLenum *,
                                                                         GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETUNIFORMUIVPROC)(GLuint, GLint, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXATTRIBIIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXATTRIBIUIVPROC)(GLuint, GLenum, GLuint *);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISENABLEDIPROC)(GLenum, GLuint);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISFRAMEBUFFERPROC)(GLuint);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISRENDERBUFFERPROC)(GLuint);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISVERTEXARRAYPROC)(GLuint);
typedef void *(INTERNAL_GL_APIENTRY *PFNGLMAPBUFFERRANGEPROC)(GLenum,
                                                              GLintptr,
                                                              GLsizeiptr,
                                                              GLbitfield);
typedef void(INTERNAL_GL_APIENTRY *PFNGLRENDERBUFFERSTORAGEPROC)(GLenum, GLenum, GLsizei, GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC)(GLenum,
                                                                            GLsizei,
                                                                            GLenum,
                                                                            GLsizei,
                                                                            GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXPARAMETERIIVPROC)(GLenum, GLenum, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXPARAMETERIUIVPROC)(GLenum, GLenum, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTRANSFORMFEEDBACKVARYINGSPROC)(GLuint,
                                                                       GLsizei,
                                                                       const GLchar *const *,
                                                                       GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM1UIPROC)(GLint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM1UIVPROC)(GLint, GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM2UIPROC)(GLint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM2UIVPROC)(GLint, GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM3UIPROC)(GLint, GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM3UIVPROC)(GLint, GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM4UIPROC)(GLint, GLuint, GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM4UIVPROC)(GLint, GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI1IPROC)(GLuint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI1IVPROC)(GLuint, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI1UIPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI1UIVPROC)(GLuint, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI2IPROC)(GLuint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI2IVPROC)(GLuint, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI2UIPROC)(GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI2UIVPROC)(GLuint, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI3IPROC)(GLuint, GLint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI3IVPROC)(GLuint, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI3UIPROC)(GLuint, GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI3UIVPROC)(GLuint, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI4BVPROC)(GLuint, const GLbyte *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI4IPROC)(GLuint, GLint, GLint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI4IVPROC)(GLuint, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI4SVPROC)(GLuint, const GLshort *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI4UBVPROC)(GLuint, const GLubyte *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI4UIPROC)(GLuint, GLuint, GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI4UIVPROC)(GLuint, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBI4USVPROC)(GLuint, const GLushort *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBIPOINTERPROC)(GLuint,
                                                                  GLint,
                                                                  GLenum,
                                                                  GLsizei,
                                                                  const GLvoid *);

// 3.1
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOPYBUFFERSUBDATAPROC)(GLenum,
                                                               GLenum,
                                                               GLintptr,
                                                               GLintptr,
                                                               GLsizeiptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWARRAYSINSTANCEDPROC)(GLenum, GLint, GLsizei, GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWELEMENTSINSTANCEDPROC)(GLenum,
                                                                   GLsizei,
                                                                   GLenum,
                                                                   const GLvoid *,
                                                                   GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC)(GLuint,
                                                                       GLuint,
                                                                       GLsizei,
                                                                       GLsizei *,
                                                                       GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVEUNIFORMBLOCKIVPROC)(GLuint,
                                                                     GLuint,
                                                                     GLenum,
                                                                     GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVEUNIFORMNAMEPROC)(GLuint,
                                                                  GLuint,
                                                                  GLsizei,
                                                                  GLsizei *,
                                                                  GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVEUNIFORMSIVPROC)(GLuint,
                                                                 GLsizei,
                                                                 const GLuint *,
                                                                 GLenum,
                                                                 GLint *);
typedef GLuint(INTERNAL_GL_APIENTRY *PFNGLGETUNIFORMBLOCKINDEXPROC)(GLuint, const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETUNIFORMINDICESPROC)(GLuint,
                                                               GLsizei,
                                                               const GLchar *const *,
                                                               GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPRIMITIVERESTARTINDEXPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXBUFFERPROC)(GLenum, GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMBLOCKBINDINGPROC)(GLuint, GLuint, GLuint);

// 3.2
typedef GLenum(INTERNAL_GL_APIENTRY *PFNGLCLIENTWAITSYNCPROC)(GLsync, GLbitfield, GLuint64);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETESYNCPROC)(GLsync);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWELEMENTSBASEVERTEXPROC)(GLenum,
                                                                    GLsizei,
                                                                    GLenum,
                                                                    const GLvoid *,
                                                                    GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC)(GLenum,
                                                                             GLsizei,
                                                                             GLenum,
                                                                             const GLvoid *,
                                                                             GLsizei,
                                                                             GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC)(GLenum,
                                                                         GLuint,
                                                                         GLuint,
                                                                         GLsizei,
                                                                         GLenum,
                                                                         const GLvoid *,
                                                                         GLint);
typedef GLsync(INTERNAL_GL_APIENTRY *PFNGLFENCESYNCPROC)(GLenum, GLbitfield);
typedef void(INTERNAL_GL_APIENTRY *PFNGLFRAMEBUFFERTEXTUREPROC)(GLenum, GLenum, GLuint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETBUFFERPARAMETERI64VPROC)(GLenum, GLenum, GLint64 *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETINTEGER64I_VPROC)(GLenum, GLuint, GLint64 *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETINTEGER64VPROC)(GLenum, GLint64 *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETMULTISAMPLEFVPROC)(GLenum, GLuint, GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSYNCIVPROC)(GLsync, GLenum, GLsizei, GLsizei *, GLint *);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISSYNCPROC)(GLsync);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC)(GLenum,
                                                                         const GLsizei *,
                                                                         GLenum,
                                                                         const GLvoid *const *,
                                                                         GLsizei,
                                                                         const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROVOKINGVERTEXPROC)(GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSAMPLEMASKIPROC)(GLuint, GLbitfield);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXIMAGE2DMULTISAMPLEPROC)(GLenum,
                                                                   GLsizei,
                                                                   GLenum,
                                                                   GLsizei,
                                                                   GLsizei,
                                                                   GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXIMAGE3DMULTISAMPLEPROC)(GLenum,
                                                                   GLsizei,
                                                                   GLenum,
                                                                   GLsizei,
                                                                   GLsizei,
                                                                   GLsizei,
                                                                   GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLWAITSYNCPROC)(GLsync, GLbitfield, GLuint64);

// 3.3
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDFRAGDATALOCATIONINDEXEDPROC)(GLuint,
                                                                         GLuint,
                                                                         GLuint,
                                                                         const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDSAMPLERPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETESAMPLERSPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENSAMPLERSPROC)(GLsizei, GLuint *);
typedef GLint(INTERNAL_GL_APIENTRY *PFNGLGETFRAGDATAINDEXPROC)(GLuint, const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYOBJECTI64VPROC)(GLuint, GLenum, GLint64 *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYOBJECTUI64VPROC)(GLuint, GLenum, GLuint64 *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSAMPLERPARAMETERIIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSAMPLERPARAMETERIUIVPROC)(GLuint, GLenum, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSAMPLERPARAMETERFVPROC)(GLuint, GLenum, GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSAMPLERPARAMETERIVPROC)(GLuint, GLenum, GLint *);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISSAMPLERPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLQUERYCOUNTERPROC)(GLuint, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSAMPLERPARAMETERIIVPROC)(GLuint, GLenum, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSAMPLERPARAMETERIUIVPROC)(GLuint, GLenum, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSAMPLERPARAMETERFPROC)(GLuint, GLenum, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSAMPLERPARAMETERFVPROC)(GLuint, GLenum, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSAMPLERPARAMETERIPROC)(GLuint, GLenum, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSAMPLERPARAMETERIVPROC)(GLuint, GLenum, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBDIVISORPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBP1UIPROC)(GLuint, GLenum, GLboolean, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBP1UIVPROC)(GLuint,
                                                               GLenum,
                                                               GLboolean,
                                                               const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBP2UIPROC)(GLuint, GLenum, GLboolean, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBP2UIVPROC)(GLuint,
                                                               GLenum,
                                                               GLboolean,
                                                               const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBP3UIPROC)(GLuint, GLenum, GLboolean, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBP3UIVPROC)(GLuint,
                                                               GLenum,
                                                               GLboolean,
                                                               const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBP4UIPROC)(GLuint, GLenum, GLboolean, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBP4UIVPROC)(GLuint,
                                                               GLenum,
                                                               GLboolean,
                                                               const GLuint *);

// 4.0
typedef void(INTERNAL_GL_APIENTRY *PFNGLBEGINQUERYINDEXEDPROC)(GLenum, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDTRANSFORMFEEDBACKPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLENDEQUATIONSEPARATEIPROC)(GLuint, GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLENDEQUATIONIPROC)(GLuint, GLenum);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLBLENDFUNCSEPARATEIPROC)(GLuint, GLenum, GLenum, GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLENDFUNCIPROC)(GLuint, GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETETRANSFORMFEEDBACKSPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWARRAYSINDIRECTPROC)(GLenum, const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWELEMENTSINDIRECTPROC)(GLenum, GLenum, const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWTRANSFORMFEEDBACKPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC)(GLenum, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLENDQUERYINDEXEDPROC)(GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENTRANSFORMFEEDBACKSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVESUBROUTINENAMEPROC)(GLuint,
                                                                     GLenum,
                                                                     GLuint,
                                                                     GLsizei,
                                                                     GLsizei *,
                                                                     GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC)(GLuint,
                                                                            GLenum,
                                                                            GLuint,
                                                                            GLsizei,
                                                                            GLsizei *,
                                                                            GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC)(GLuint,
                                                                          GLenum,
                                                                          GLuint,
                                                                          GLenum,
                                                                          GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMSTAGEIVPROC)(GLuint, GLenum, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYINDEXEDIVPROC)(GLenum, GLuint, GLenum, GLint *);
typedef GLuint(INTERNAL_GL_APIENTRY *PFNGLGETSUBROUTINEINDEXPROC)(GLuint, GLenum, const GLchar *);
typedef GLint(INTERNAL_GL_APIENTRY *PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC)(GLuint,
                                                                           GLenum,
                                                                           const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETUNIFORMSUBROUTINEUIVPROC)(GLenum, GLint, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETUNIFORMDVPROC)(GLuint, GLint, GLdouble *);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISTRANSFORMFEEDBACKPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMINSAMPLESHADINGPROC)(GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPATCHPARAMETERFVPROC)(GLenum, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPATCHPARAMETERIPROC)(GLenum, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPAUSETRANSFORMFEEDBACKPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLRESUMETRANSFORMFEEDBACKPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM1DPROC)(GLint, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM1DVPROC)(GLint, GLsizei, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM2DPROC)(GLint, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM2DVPROC)(GLint, GLsizei, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM3DPROC)(GLint, GLdouble, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM3DVPROC)(GLint, GLsizei, const GLdouble *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLUNIFORM4DPROC)(GLint, GLdouble, GLdouble, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORM4DVPROC)(GLint, GLsizei, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX2DVPROC)(GLint,
                                                              GLsizei,
                                                              GLboolean,
                                                              const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX2X3DVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX2X4DVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX3DVPROC)(GLint,
                                                              GLsizei,
                                                              GLboolean,
                                                              const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX3X2DVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX3X4DVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX4DVPROC)(GLint,
                                                              GLsizei,
                                                              GLboolean,
                                                              const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX4X2DVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMMATRIX4X3DVPROC)(GLint,
                                                                GLsizei,
                                                                GLboolean,
                                                                const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUNIFORMSUBROUTINESUIVPROC)(GLenum, GLsizei, const GLuint *);

// 4.1
typedef void(INTERNAL_GL_APIENTRY *PFNGLACTIVESHADERPROGRAMPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDPROGRAMPIPELINEPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARDEPTHFPROC)(GLfloat);
typedef GLuint(INTERNAL_GL_APIENTRY *PFNGLCREATESHADERPROGRAMVPROC)(GLenum,
                                                                    GLsizei,
                                                                    const GLchar *const *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETEPROGRAMPIPELINESPROC)(GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDEPTHRANGEARRAYVPROC)(GLuint, GLsizei, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDEPTHRANGEINDEXEDPROC)(GLuint, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDEPTHRANGEFPROC)(GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENPROGRAMPIPELINESPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETDOUBLEI_VPROC)(GLenum, GLuint, GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETFLOATI_VPROC)(GLenum, GLuint, GLfloat *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMBINARYPROC)(GLuint, GLsizei, GLsizei *, GLenum *, void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMPIPELINEINFOLOGPROC)(GLuint,
                                                                       GLsizei,
                                                                       GLsizei *,
                                                                       GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMPIPELINEIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSHADERPRECISIONFORMATPROC)(GLenum,
                                                                      GLenum,
                                                                      GLint *,
                                                                      GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXATTRIBLDVPROC)(GLuint, GLenum, GLdouble *);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISPROGRAMPIPELINEPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMBINARYPROC)(GLuint, GLenum, const void *, GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMPARAMETERIPROC)(GLuint, GLenum, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM1DPROC)(GLuint, GLint, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM1DVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM1FPROC)(GLuint, GLint, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM1FVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM1IPROC)(GLuint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM1IVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM1UIPROC)(GLuint, GLint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM1UIVPROC)(GLuint,
                                                                GLint,
                                                                GLsizei,
                                                                const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM2DPROC)(GLuint, GLint, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM2DVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM2FPROC)(GLuint, GLint, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM2FVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM2IPROC)(GLuint, GLint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM2IVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM2UIPROC)(GLuint, GLint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM2UIVPROC)(GLuint,
                                                                GLint,
                                                                GLsizei,
                                                                const GLuint *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM3DPROC)(GLuint, GLint, GLdouble, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM3DVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLdouble *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM3FPROC)(GLuint, GLint, GLfloat, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM3FVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM3IPROC)(GLuint, GLint, GLint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM3IVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLint *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM3UIPROC)(GLuint, GLint, GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM3UIVPROC)(GLuint,
                                                                GLint,
                                                                GLsizei,
                                                                const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM4DPROC)(GLuint,
                                                              GLint,
                                                              GLdouble,
                                                              GLdouble,
                                                              GLdouble,
                                                              GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM4DVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM4FPROC)(GLuint,
                                                              GLint,
                                                              GLfloat,
                                                              GLfloat,
                                                              GLfloat,
                                                              GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM4FVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLfloat *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM4IPROC)(GLuint, GLint, GLint, GLint, GLint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM4IVPROC)(GLuint,
                                                               GLint,
                                                               GLsizei,
                                                               const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM4UIPROC)(GLuint,
                                                               GLint,
                                                               GLuint,
                                                               GLuint,
                                                               GLuint,
                                                               GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORM4UIVPROC)(GLuint,
                                                                GLint,
                                                                GLsizei,
                                                                const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX2DVPROC)(GLuint,
                                                                     GLint,
                                                                     GLsizei,
                                                                     GLboolean,
                                                                     const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX2FVPROC)(GLuint,
                                                                     GLint,
                                                                     GLsizei,
                                                                     GLboolean,
                                                                     const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X3DVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X4DVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX3DVPROC)(GLuint,
                                                                     GLint,
                                                                     GLsizei,
                                                                     GLboolean,
                                                                     const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX3FVPROC)(GLuint,
                                                                     GLint,
                                                                     GLsizei,
                                                                     GLboolean,
                                                                     const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X2DVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X4DVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX4DVPROC)(GLuint,
                                                                     GLint,
                                                                     GLsizei,
                                                                     GLboolean,
                                                                     const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX4FVPROC)(GLuint,
                                                                     GLint,
                                                                     GLsizei,
                                                                     GLboolean,
                                                                     const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X2DVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X3DVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       GLboolean,
                                                                       const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLRELEASESHADERCOMPILERPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLSCISSORARRAYVPROC)(GLuint, GLsizei, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSCISSORINDEXEDPROC)(GLuint, GLint, GLint, GLsizei, GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSCISSORINDEXEDVPROC)(GLuint, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSHADERBINARYPROC)(GLsizei,
                                                          const GLuint *,
                                                          GLenum,
                                                          const void *,
                                                          GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLUSEPROGRAMSTAGESPROC)(GLuint, GLbitfield, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVALIDATEPROGRAMPIPELINEPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBL1DPROC)(GLuint, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBL1DVPROC)(GLuint, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBL2DPROC)(GLuint, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBL2DVPROC)(GLuint, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBL3DPROC)(GLuint, GLdouble, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBL3DVPROC)(GLuint, const GLdouble *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBL4DPROC)(GLuint, GLdouble, GLdouble, GLdouble, GLdouble);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBL4DVPROC)(GLuint, const GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBLPOINTERPROC)(GLuint,
                                                                  GLint,
                                                                  GLenum,
                                                                  GLsizei,
                                                                  const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVIEWPORTARRAYVPROC)(GLuint, GLsizei, const GLfloat *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVIEWPORTINDEXEDFPROC)(GLuint, GLfloat, GLfloat, GLfloat, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVIEWPORTINDEXEDFVPROC)(GLuint, const GLfloat *);

// 4.2
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDIMAGETEXTUREPROC)(GLuint,
                                                              GLuint,
                                                              GLint,
                                                              GLboolean,
                                                              GLint,
                                                              GLenum,
                                                              GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC)(GLenum,
                                                                             GLint,
                                                                             GLsizei,
                                                                             GLsizei,
                                                                             GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC)(GLenum,
                                                                               GLsizei,
                                                                               GLenum,
                                                                               const void *,
                                                                               GLsizei,
                                                                               GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC)(
    GLenum,
    GLsizei,
    GLenum,
    const void *,
    GLsizei,
    GLint,
    GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC)(GLenum,
                                                                            GLuint,
                                                                            GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC)(GLenum,
                                                                                  GLuint,
                                                                                  GLuint,
                                                                                  GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETACTIVEATOMICCOUNTERBUFFERIVPROC)(GLuint,
                                                                            GLuint,
                                                                            GLenum,
                                                                            GLint *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLGETINTERNALFORMATIVPROC)(GLenum, GLenum, GLenum, GLsizei, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMEMORYBARRIERPROC)(GLbitfield);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGE1DPROC)(GLenum, GLsizei, GLenum, GLsizei);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGE2DPROC)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGE3DPROC)(GLenum,
                                                          GLsizei,
                                                          GLenum,
                                                          GLsizei,
                                                          GLsizei,
                                                          GLsizei);

// 4.3
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDVERTEXBUFFERPROC)(GLuint, GLuint, GLintptr, GLsizei);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLCLEARBUFFERDATAPROC)(GLenum, GLenum, GLenum, GLenum, const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARBUFFERSUBDATAPROC)(GLenum,
                                                                GLenum,
                                                                GLintptr,
                                                                GLsizeiptr,
                                                                GLenum,
                                                                GLenum,
                                                                const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOPYIMAGESUBDATAPROC)(GLuint,
                                                              GLenum,
                                                              GLint,
                                                              GLint,
                                                              GLint,
                                                              GLint,
                                                              GLuint,
                                                              GLenum,
                                                              GLint,
                                                              GLint,
                                                              GLint,
                                                              GLint,
                                                              GLsizei,
                                                              GLsizei,
                                                              GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDEBUGMESSAGECALLBACKPROC)(GLDEBUGPROC, const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDEBUGMESSAGECONTROLPROC)(GLenum,
                                                                 GLenum,
                                                                 GLenum,
                                                                 GLsizei,
                                                                 const GLuint *,
                                                                 GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDEBUGMESSAGEINSERTPROC)(GLenum,
                                                                GLenum,
                                                                GLuint,
                                                                GLenum,
                                                                GLsizei,
                                                                const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDISPATCHCOMPUTEPROC)(GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDISPATCHCOMPUTEINDIRECTPROC)(GLintptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLFRAMEBUFFERPARAMETERIPROC)(GLenum, GLenum, GLint);
typedef GLuint(INTERNAL_GL_APIENTRY *PFNGLGETDEBUGMESSAGELOGPROC)(GLuint,
                                                                  GLsizei,
                                                                  GLenum *,
                                                                  GLenum *,
                                                                  GLuint *,
                                                                  GLenum *,
                                                                  GLsizei *,
                                                                  GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETFRAMEBUFFERPARAMETERIVPROC)(GLenum, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETINTERNALFORMATI64VPROC)(GLenum,
                                                                   GLenum,
                                                                   GLenum,
                                                                   GLsizei,
                                                                   GLint64 *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPOINTERVPROC)(GLenum, void **);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLGETOBJECTLABELPROC)(GLenum, GLuint, GLsizei, GLsizei *, GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETOBJECTPTRLABELPROC)(const void *,
                                                               GLsizei,
                                                               GLsizei *,
                                                               GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMINTERFACEIVPROC)(GLuint, GLenum, GLenum, GLint *);
typedef GLuint(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMRESOURCEINDEXPROC)(GLuint,
                                                                       GLenum,
                                                                       const GLchar *);
typedef GLint(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMRESOURCELOCATIONPROC)(GLuint,
                                                                         GLenum,
                                                                         const GLchar *);
typedef GLint(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMRESOURCELOCATIONINDEXPROC)(GLuint,
                                                                              GLenum,
                                                                              const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMRESOURCENAMEPROC)(GLuint,
                                                                    GLenum,
                                                                    GLuint,
                                                                    GLsizei,
                                                                    GLsizei *,
                                                                    GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPROGRAMRESOURCEIVPROC)(GLuint,
                                                                  GLenum,
                                                                  GLuint,
                                                                  GLsizei,
                                                                  const GLenum *,
                                                                  GLsizei,
                                                                  GLsizei *,
                                                                  GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLINVALIDATEBUFFERDATAPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLINVALIDATEBUFFERSUBDATAPROC)(GLuint, GLintptr, GLsizeiptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLINVALIDATEFRAMEBUFFERPROC)(GLenum, GLsizei, const GLenum *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLINVALIDATESUBFRAMEBUFFERPROC)(GLenum,
                                                                      GLsizei,
                                                                      const GLenum *,
                                                                      GLint,
                                                                      GLint,
                                                                      GLsizei,
                                                                      GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLINVALIDATETEXIMAGEPROC)(GLuint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLINVALIDATETEXSUBIMAGEPROC)(GLuint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLsizei,
                                                                   GLsizei,
                                                                   GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMULTIDRAWARRAYSINDIRECTPROC)(GLenum,
                                                                     const void *,
                                                                     GLsizei,
                                                                     GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMULTIDRAWELEMENTSINDIRECTPROC)(GLenum,
                                                                       GLenum,
                                                                       const void *,
                                                                       GLsizei,
                                                                       GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLOBJECTLABELPROC)(GLenum, GLuint, GLsizei, const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLOBJECTPTRLABELPROC)(const void *, GLsizei, const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPOPDEBUGGROUPPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLPUSHDEBUGGROUPPROC)(GLenum,
                                                            GLuint,
                                                            GLsizei,
                                                            const GLchar *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSHADERSTORAGEBLOCKBINDINGPROC)(GLuint, GLuint, GLuint);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLTEXBUFFERRANGEPROC)(GLenum, GLenum, GLuint, GLintptr, GLsizeiptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGE2DMULTISAMPLEPROC)(GLenum,
                                                                     GLsizei,
                                                                     GLenum,
                                                                     GLsizei,
                                                                     GLsizei,
                                                                     GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGE3DMULTISAMPLEPROC)(GLenum,
                                                                     GLsizei,
                                                                     GLenum,
                                                                     GLsizei,
                                                                     GLsizei,
                                                                     GLsizei,
                                                                     GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREVIEWPROC)(GLuint,
                                                         GLenum,
                                                         GLuint,
                                                         GLenum,
                                                         GLuint,
                                                         GLuint,
                                                         GLuint,
                                                         GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBBINDINGPROC)(GLuint, GLuint);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBFORMATPROC)(GLuint, GLint, GLenum, GLboolean, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBIFORMATPROC)(GLuint, GLint, GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXATTRIBLFORMATPROC)(GLuint, GLint, GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXBINDINGDIVISORPROC)(GLuint, GLuint);

// NV_framebuffer_mixed_samples
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOVERAGEMODULATIONNVPROC)(GLenum);

// 4.4
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDBUFFERSBASEPROC)(GLenum,
                                                             GLuint,
                                                             GLsizei,
                                                             const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDBUFFERSRANGEPROC)(GLenum,
                                                              GLuint,
                                                              GLsizei,
                                                              const GLuint *,
                                                              const GLintptr *,
                                                              const GLsizeiptr *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDIMAGETEXTURESPROC)(GLuint, GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDSAMPLERSPROC)(GLuint, GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDTEXTURESPROC)(GLuint, GLsizei, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDVERTEXBUFFERSPROC)(GLuint,
                                                               GLsizei,
                                                               const GLuint *,
                                                               const GLintptr *,
                                                               const GLsizei *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBUFFERSTORAGEPROC)(GLenum,
                                                           GLsizeiptr,
                                                           const void *,
                                                           GLbitfield);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLCLEARTEXIMAGEPROC)(GLuint, GLint, GLenum, GLenum, const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARTEXSUBIMAGEPROC)(GLuint,
                                                              GLint,
                                                              GLint,
                                                              GLint,
                                                              GLint,
                                                              GLsizei,
                                                              GLsizei,
                                                              GLsizei,
                                                              GLenum,
                                                              GLenum,
                                                              const void *);

// 4.5
typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDTEXTUREUNITPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLITNAMEDFRAMEBUFFERPROC)(GLuint,
                                                                  GLuint,
                                                                  GLint,
                                                                  GLint,
                                                                  GLint,
                                                                  GLint,
                                                                  GLint,
                                                                  GLint,
                                                                  GLint,
                                                                  GLint,
                                                                  GLbitfield,
                                                                  GLenum);
typedef GLenum(INTERNAL_GL_APIENTRY *PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC)(GLuint, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARNAMEDBUFFERDATAPROC)(GLuint,
                                                                  GLenum,
                                                                  GLenum,
                                                                  GLenum,
                                                                  const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARNAMEDBUFFERSUBDATAPROC)(GLuint,
                                                                     GLenum,
                                                                     GLintptr,
                                                                     GLsizeiptr,
                                                                     GLenum,
                                                                     GLenum,
                                                                     const void *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLCLEARNAMEDFRAMEBUFFERFIPROC)(GLuint, GLenum, GLint, GLfloat, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARNAMEDFRAMEBUFFERFVPROC)(GLuint,
                                                                     GLenum,
                                                                     GLint,
                                                                     const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARNAMEDFRAMEBUFFERIVPROC)(GLuint,
                                                                     GLenum,
                                                                     GLint,
                                                                     const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC)(GLuint,
                                                                      GLenum,
                                                                      GLint,
                                                                      const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCLIPCONTROLPROC)(GLenum, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC)(GLuint,
                                                                         GLint,
                                                                         GLint,
                                                                         GLsizei,
                                                                         GLenum,
                                                                         GLsizei,
                                                                         const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC)(GLuint,
                                                                         GLint,
                                                                         GLint,
                                                                         GLint,
                                                                         GLsizei,
                                                                         GLsizei,
                                                                         GLenum,
                                                                         GLsizei,
                                                                         const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC)(GLuint,
                                                                         GLint,
                                                                         GLint,
                                                                         GLint,
                                                                         GLint,
                                                                         GLsizei,
                                                                         GLsizei,
                                                                         GLsizei,
                                                                         GLenum,
                                                                         GLsizei,
                                                                         const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOPYNAMEDBUFFERSUBDATAPROC)(GLuint,
                                                                    GLuint,
                                                                    GLintptr,
                                                                    GLintptr,
                                                                    GLsizeiptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOPYTEXTURESUBIMAGE1DPROC)(GLuint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOPYTEXTURESUBIMAGE2DPROC)(GLuint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLsizei,
                                                                   GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOPYTEXTURESUBIMAGE3DPROC)(GLuint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLint,
                                                                   GLsizei,
                                                                   GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATEBUFFERSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATEFRAMEBUFFERSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATEPROGRAMPIPELINESPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATEQUERIESPROC)(GLenum, GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATERENDERBUFFERSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATESAMPLERSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATETEXTURESPROC)(GLenum, GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATETRANSFORMFEEDBACKSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATEVERTEXARRAYSPROC)(GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDISABLEVERTEXARRAYATTRIBPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLENABLEVERTEXARRAYATTRIBPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC)(GLuint,
                                                                         GLintptr,
                                                                         GLsizeiptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENERATETEXTUREMIPMAPPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC)(GLuint,
                                                                       GLint,
                                                                       GLsizei,
                                                                       void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC)(GLuint,
                                                                          GLint,
                                                                          GLint,
                                                                          GLint,
                                                                          GLint,
                                                                          GLsizei,
                                                                          GLsizei,
                                                                          GLsizei,
                                                                          GLsizei,
                                                                          void *);
typedef GLenum(INTERNAL_GL_APIENTRY *PFNGLGETGRAPHICSRESETSTATUSPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNAMEDBUFFERPARAMETERI64VPROC)(GLuint, GLenum, GLint64 *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNAMEDBUFFERPARAMETERIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNAMEDBUFFERPOINTERVPROC)(GLuint, GLenum, void **);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNAMEDBUFFERSUBDATAPROC)(GLuint,
                                                                   GLintptr,
                                                                   GLsizeiptr,
                                                                   void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVPROC)(GLuint,
                                                                                      GLenum,
                                                                                      GLenum,
                                                                                      GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVPROC)(GLuint,
                                                                            GLenum,
                                                                            GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNAMEDRENDERBUFFERPARAMETERIVPROC)(GLuint,
                                                                             GLenum,
                                                                             GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYBUFFEROBJECTI64VPROC)(GLuint,
                                                                      GLuint,
                                                                      GLenum,
                                                                      GLintptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYBUFFEROBJECTIVPROC)(GLuint,
                                                                    GLuint,
                                                                    GLenum,
                                                                    GLintptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYBUFFEROBJECTUI64VPROC)(GLuint,
                                                                       GLuint,
                                                                       GLenum,
                                                                       GLintptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETQUERYBUFFEROBJECTUIVPROC)(GLuint,
                                                                     GLuint,
                                                                     GLenum,
                                                                     GLintptr);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLGETTEXTUREIMAGEPROC)(GLuint, GLint, GLenum, GLenum, GLsizei, void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXTURELEVELPARAMETERFVPROC)(GLuint,
                                                                        GLint,
                                                                        GLenum,
                                                                        GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXTURELEVELPARAMETERIVPROC)(GLuint,
                                                                        GLint,
                                                                        GLenum,
                                                                        GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXTUREPARAMETERIIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXTUREPARAMETERIUIVPROC)(GLuint, GLenum, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXTUREPARAMETERFVPROC)(GLuint, GLenum, GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXTUREPARAMETERIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTEXTURESUBIMAGEPROC)(GLuint,
                                                                GLint,
                                                                GLint,
                                                                GLint,
                                                                GLint,
                                                                GLsizei,
                                                                GLsizei,
                                                                GLsizei,
                                                                GLenum,
                                                                GLenum,
                                                                GLsizei,
                                                                void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTRANSFORMFEEDBACKI64_VPROC)(GLuint,
                                                                       GLenum,
                                                                       GLuint,
                                                                       GLint64 *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTRANSFORMFEEDBACKI_VPROC)(GLuint,
                                                                     GLenum,
                                                                     GLuint,
                                                                     GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETTRANSFORMFEEDBACKIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXARRAYINDEXED64IVPROC)(GLuint,
                                                                       GLuint,
                                                                       GLenum,
                                                                       GLint64 *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXARRAYINDEXEDIVPROC)(GLuint,
                                                                     GLuint,
                                                                     GLenum,
                                                                     GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETVERTEXARRAYIVPROC)(GLuint, GLenum, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNCOMPRESSEDTEXIMAGEPROC)(GLenum, GLint, GLsizei, void *);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLGETNTEXIMAGEPROC)(GLenum, GLint, GLenum, GLenum, GLsizei, void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNUNIFORMDVPROC)(GLuint, GLint, GLsizei, GLdouble *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNUNIFORMFVPROC)(GLuint, GLint, GLsizei, GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNUNIFORMIVPROC)(GLuint, GLint, GLsizei, GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETNUNIFORMUIVPROC)(GLuint, GLint, GLsizei, GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLINVALIDATENAMEDFRAMEBUFFERDATAPROC)(GLuint,
                                                                            GLsizei,
                                                                            const GLenum *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLINVALIDATENAMEDFRAMEBUFFERSUBDATAPROC)(GLuint,
                                                                               GLsizei,
                                                                               const GLenum *,
                                                                               GLint,
                                                                               GLint,
                                                                               GLsizei,
                                                                               GLsizei);
typedef void *(INTERNAL_GL_APIENTRY *PFNGLMAPNAMEDBUFFERPROC)(GLuint, GLenum);
typedef void *(INTERNAL_GL_APIENTRY *PFNGLMAPNAMEDBUFFERRANGEPROC)(GLuint,
                                                                   GLintptr,
                                                                   GLsizeiptr,
                                                                   GLbitfield);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMEMORYBARRIERBYREGIONPROC)(GLbitfield);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDBUFFERDATAPROC)(GLuint,
                                                             GLsizeiptr,
                                                             const void *,
                                                             GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDBUFFERSTORAGEPROC)(GLuint,
                                                                GLsizeiptr,
                                                                const void *,
                                                                GLbitfield);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDBUFFERSUBDATAPROC)(GLuint,
                                                                GLintptr,
                                                                GLsizeiptr,
                                                                const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC)(GLuint, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC)(GLuint,
                                                                         GLsizei,
                                                                         const GLenum *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDFRAMEBUFFERPARAMETERIPROC)(GLuint, GLenum, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC)(GLuint, GLenum);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC)(GLuint,
                                                                          GLenum,
                                                                          GLenum,
                                                                          GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTUREPROC)(GLuint, GLenum, GLuint, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC)(GLuint,
                                                                          GLenum,
                                                                          GLuint,
                                                                          GLint,
                                                                          GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDRENDERBUFFERSTORAGEPROC)(GLuint,
                                                                      GLenum,
                                                                      GLsizei,
                                                                      GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEPROC)(GLuint,
                                                                                 GLsizei,
                                                                                 GLenum,
                                                                                 GLsizei,
                                                                                 GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLREADNPIXELSPROC)(GLint,
                                                         GLint,
                                                         GLsizei,
                                                         GLsizei,
                                                         GLenum,
                                                         GLenum,
                                                         GLsizei,
                                                         void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREBARRIERPROC)();
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREBUFFERPROC)(GLuint, GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREBUFFERRANGEPROC)(GLuint,
                                                                GLenum,
                                                                GLuint,
                                                                GLintptr,
                                                                GLsizeiptr);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREPARAMETERIIVPROC)(GLuint, GLenum, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREPARAMETERIUIVPROC)(GLuint, GLenum, const GLuint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREPARAMETERFPROC)(GLuint, GLenum, GLfloat);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREPARAMETERFVPROC)(GLuint, GLenum, const GLfloat *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREPARAMETERIPROC)(GLuint, GLenum, GLint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTUREPARAMETERIVPROC)(GLuint, GLenum, const GLint *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGE1DPROC)(GLuint, GLsizei, GLenum, GLsizei);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGE2DPROC)(GLuint, GLsizei, GLenum, GLsizei, GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC)(GLuint,
                                                                         GLsizei,
                                                                         GLenum,
                                                                         GLsizei,
                                                                         GLsizei,
                                                                         GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGE3DPROC)(GLuint,
                                                              GLsizei,
                                                              GLenum,
                                                              GLsizei,
                                                              GLsizei,
                                                              GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC)(GLuint,
                                                                         GLsizei,
                                                                         GLenum,
                                                                         GLsizei,
                                                                         GLsizei,
                                                                         GLsizei,
                                                                         GLboolean);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESUBIMAGE1DPROC)(GLuint,
                                                               GLint,
                                                               GLint,
                                                               GLsizei,
                                                               GLenum,
                                                               GLenum,
                                                               const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESUBIMAGE2DPROC)(GLuint,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLsizei,
                                                               GLsizei,
                                                               GLenum,
                                                               GLenum,
                                                               const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESUBIMAGE3DPROC)(GLuint,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLint,
                                                               GLsizei,
                                                               GLsizei,
                                                               GLsizei,
                                                               GLenum,
                                                               GLenum,
                                                               const void *);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC)(GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC)(GLuint,
                                                                          GLuint,
                                                                          GLuint,
                                                                          GLintptr,
                                                                          GLsizeiptr);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLUNMAPNAMEDBUFFERPROC)(GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXARRAYATTRIBBINDINGPROC)(GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXARRAYATTRIBFORMATPROC)(GLuint,
                                                                     GLuint,
                                                                     GLint,
                                                                     GLenum,
                                                                     GLboolean,
                                                                     GLuint);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVERTEXARRAYATTRIBIFORMATPROC)(GLuint, GLuint, GLint, GLenum, GLuint);
typedef void(
    INTERNAL_GL_APIENTRY *PFNGLVERTEXARRAYATTRIBLFORMATPROC)(GLuint, GLuint, GLint, GLenum, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXARRAYBINDINGDIVISORPROC)(GLuint, GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXARRAYELEMENTBUFFERPROC)(GLuint, GLuint);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXARRAYVERTEXBUFFERPROC)(GLuint,
                                                                     GLuint,
                                                                     GLuint,
                                                                     GLintptr,
                                                                     GLsizei);
typedef void(INTERNAL_GL_APIENTRY *PFNGLVERTEXARRAYVERTEXBUFFERSPROC)(GLuint,
                                                                      GLuint,
                                                                      GLsizei,
                                                                      const GLuint *,
                                                                      const GLintptr *,
                                                                      const GLsizei *);

// GL_EXT_discard_framebuffer
typedef void(INTERNAL_GL_APIENTRY *PFNGLDISCARDFRAMEBUFFEREXTPROC)(GLenum target,
                                                                   GLsizei numAttachments,
                                                                   const GLenum *attachments);

// GL_OES_EGL_image
typedef void *GLeglImageOES;
typedef void(INTERNAL_GL_APIENTRY *PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)(GLenum target,
                                                                        GLeglImageOES image);
typedef void(INTERNAL_GL_APIENTRY *PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC)(
    GLenum target,
    GLeglImageOES image);

// NV_path_rendering (originally written against 3.2 compatibility profile)
typedef void(INTERNAL_GL_APIENTRY *PFNGLMATRIXLOADFEXTPROC)(GLenum matrixMode, const GLfloat *m);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMATRIXLOADFNVPROC)(GLenum matrixMode, const GLfloat *m);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMATRIXLOADIDENTITYNVPROC)(GLenum matrixMode);
typedef GLuint(INTERNAL_GL_APIENTRY *PFNGLGENPATHSNVPROC)(GLsizei range);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETEPATHSNVPROC)(GLuint path, GLsizei range);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISPATHNVPROC)(GLuint path);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPATHCOMMANDSNVPROC)(GLuint path,
                                                            GLsizei numCommands,
                                                            const GLubyte *commands,
                                                            GLsizei numCoords,
                                                            GLenum coordType,
                                                            const void *coords);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPATHPARAMETERINVPROC)(GLuint path,
                                                              GLenum pname,
                                                              GLint value);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPATHPARAMETERFNVPROC)(GLuint path,
                                                              GLenum pname,
                                                              GLfloat value);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPATHPARAMETERIVNVPROC)(GLuint path,
                                                                  GLenum pname,
                                                                  GLint *value);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETPATHPARAMETERFVNVPROC)(GLuint path,
                                                                  GLenum pname,
                                                                  GLfloat *value);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPATHSTENCILFUNCNVPROC)(GLenum func, GLint ref, GLuint mask);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILFILLPATHNVPROC)(GLuint path,
                                                               GLenum fillMode,
                                                               GLuint mask);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILSTROKEPATHNVPROC)(GLuint path,
                                                                 GLint reference,
                                                                 GLuint mask);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOVERFILLPATHNVPROC)(GLuint path, GLenum coverMode);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOVERSTROKEPATHNVPROC)(GLuint path, GLenum coverMode);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILTHENCOVERFILLPATHNVPROC)(GLuint path,
                                                                        GLenum fillMode,
                                                                        GLuint mask,
                                                                        GLenum coverMode);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILTHENCOVERSTROKEPATHNVPROC)(GLuint path,
                                                                          GLint reference,
                                                                          GLuint mask,
                                                                          GLenum coverMode);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOVERFILLPATHINSTANCEDNVPROC)(
    GLsizei numPaths,
    GLenum pathNameType,
    const void *paths,
    GLuint pathBase,
    GLenum coverMode,
    GLenum transformType,
    const GLfloat *transformValues);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCOVERSTROKEPATHINSTANCEDNVPROC)(
    GLsizei numPaths,
    GLenum pathNameType,
    const void *paths,
    GLuint pathBase,
    GLenum coverMode,
    GLenum transformType,
    const GLfloat *transformValues);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILFILLPATHINSTANCEDNVPROC)(
    GLsizei numPaths,
    GLenum pathNameType,
    const void *paths,
    GLuint pathBase,
    GLenum fillMode,
    GLuint mask,
    GLenum transformType,
    const GLfloat *transformValues);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILSTROKEPATHINSTANCEDNVPROC)(
    GLsizei numPaths,
    GLenum pathNameType,
    const void *paths,
    GLuint pathBase,
    GLint reference,
    GLuint mask,
    GLenum transformType,
    const GLfloat *transformValues);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILTHENCOVERFILLPATHINSTANCEDNVPROC)(
    GLsizei numPaths,
    GLenum pathNameType,
    const void *paths,
    GLuint pathBase,
    GLenum fillMode,
    GLuint mask,
    GLenum coverMode,
    GLenum transformType,
    const GLfloat *transformValues);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSTENCILTHENCOVERSTROKEPATHINSTANCEDNVPROC)(
    GLsizei numPaths,
    GLenum pathNameType,
    const void *paths,
    GLuint pathBase,
    GLint reference,
    GLuint mask,
    GLenum coverMode,
    GLenum transformType,
    const GLfloat *transformValues);

typedef void(INTERNAL_GL_APIENTRY *PFNGLBINDFRAGMENTINPUTLOCATIONNVPROC)(GLuint program,
                                                                         GLint location,
                                                                         const GLchar *name);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPROGRAMPATHFRAGMENTINPUTGENNVPROC)(GLuint program,
                                                                           GLint location,
                                                                           GLenum genMode,
                                                                           GLint components,
                                                                           const GLfloat *coeffs);

// ES 3.2
typedef void(INTERNAL_GL_APIENTRY *PFNGLBLENDBARRIERPROC)(void);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPRIMITIVEBOUNDINGBOXPROC)(GLfloat minX,
                                                                  GLfloat minY,
                                                                  GLfloat minZ,
                                                                  GLfloat minW,
                                                                  GLfloat maxX,
                                                                  GLfloat maxY,
                                                                  GLfloat maxZ,
                                                                  GLfloat maxW);

// GL_NV_internalformat_sample_query
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETINTERNALFORMATSAMPLEIVNVPROC)(GLenum target,
                                                                         GLenum internalformat,
                                                                         GLsizei samples,
                                                                         GLenum pname,
                                                                         GLsizei bufSize,
                                                                         GLint *params);

// GL_OVR_multiview2
typedef void(INTERNAL_GL_APIENTRY *PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVRPROC)(GLenum target,
                                                                            GLenum attachment,
                                                                            GLuint texture,
                                                                            GLint level,
                                                                            GLint baseViewIndex,
                                                                            GLsizei numViews);
// EXT_debug_marker
typedef void(INTERNAL_GL_APIENTRY *PFNGLINSERTEVENTMARKEREXTPROC)(GLsizei length,
                                                                  const GLchar *marker);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPUSHGROUPMARKEREXTPROC)(GLsizei length,
                                                                const GLchar *marker);
typedef void(INTERNAL_GL_APIENTRY *PFNGLPOPGROUPMARKEREXTPROC)(void);

// KHR_parallel_shader_compile
typedef void(INTERNAL_GL_APIENTRY *PFNGLMAXSHADERCOMPILERTHREADSKHRPROC)(GLuint count);

// ARB_parallel_shader_compile
typedef void(INTERNAL_GL_APIENTRY *PFNGLMAXSHADERCOMPILERTHREADSARBPROC)(GLuint count);

// GL_EXT_memory_object
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETUNSIGNEDBYTEVEXTPROC)(GLenum pname, GLubyte *data);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETUNSIGNEDBYTEI_VEXTPROC)(GLenum target,
                                                                   GLuint index,
                                                                   GLubyte *data);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETEMEMORYOBJECTSEXTPROC)(GLsizei n,
                                                                    const GLuint *memoryObjects);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISMEMORYOBJECTEXTPROC)(GLuint memoryObject);
typedef void(INTERNAL_GL_APIENTRY *PFNGLCREATEMEMORYOBJECTSEXTPROC)(GLsizei n,
                                                                    GLuint *memoryObjects);
typedef void(INTERNAL_GL_APIENTRY *PFNGLMEMORYOBJECTPARAMETERIVEXTPROC)(GLuint memoryObject,
                                                                        GLenum pname,
                                                                        const GLint *params);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETMEMORYOBJECTPARAMETERIVEXTPROC)(GLuint memoryObject,
                                                                           GLenum pname,
                                                                           GLint *params);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGEMEM2DEXTPROC)(GLenum target,
                                                                GLsizei levels,
                                                                GLenum internalFormat,
                                                                GLsizei width,
                                                                GLsizei height,
                                                                GLuint memory,
                                                                GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGEMEM2DMULTISAMPLEEXTPROC)(
    GLenum target,
    GLsizei samples,
    GLenum internalFormat,
    GLsizei width,
    GLsizei height,
    GLboolean fixedSampleLocations,
    GLuint memory,
    GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGEMEM3DEXTPROC)(GLenum target,
                                                                GLsizei levels,
                                                                GLenum internalFormat,
                                                                GLsizei width,
                                                                GLsizei height,
                                                                GLsizei depth,
                                                                GLuint memory,
                                                                GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGEMEM3DMULTISAMPLEEXTPROC)(
    GLenum target,
    GLsizei samples,
    GLenum internalFormat,
    GLsizei width,
    GLsizei height,
    GLsizei depth,
    GLboolean fixedSampleLocations,
    GLuint memory,
    GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLBUFFERSTORAGEMEMEXTPROC)(GLenum target,
                                                                 GLsizeiptr size,
                                                                 GLuint memory,
                                                                 GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGEMEM2DEXTPROC)(GLuint texture,
                                                                    GLsizei levels,
                                                                    GLenum internalFormat,
                                                                    GLsizei width,
                                                                    GLsizei height,
                                                                    GLuint memory,
                                                                    GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGEMEM2DMULTISAMPLEEXTPROC)(
    GLuint texture,
    GLsizei samples,
    GLenum internalFormat,
    GLsizei width,
    GLsizei height,
    GLboolean fixedSampleLocations,
    GLuint memory,
    GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGEMEM3DEXTPROC)(GLuint texture,
                                                                    GLsizei levels,
                                                                    GLenum internalFormat,
                                                                    GLsizei width,
                                                                    GLsizei height,
                                                                    GLsizei depth,
                                                                    GLuint memory,
                                                                    GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGEMEM3DMULTISAMPLEEXTPROC)(
    GLuint texture,
    GLsizei samples,
    GLenum internalFormat,
    GLsizei width,
    GLsizei height,
    GLsizei depth,
    GLboolean fixedSampleLocations,
    GLuint memory,
    GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLNAMEDBUFFERSTORAGEMEMEXTPROC)(GLuint buffer,
                                                                      GLsizeiptr size,
                                                                      GLuint memory,
                                                                      GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXSTORAGEMEM1DEXTPROC)(GLenum target,
                                                                GLsizei levels,
                                                                GLenum internalFormat,
                                                                GLsizei width,
                                                                GLuint memory,
                                                                GLuint64 offset);
typedef void(INTERNAL_GL_APIENTRY *PFNGLTEXTURESTORAGEMEM1DEXTPROC)(GLuint texture,
                                                                    GLsizei levels,
                                                                    GLenum internalFormat,
                                                                    GLsizei width,
                                                                    GLuint memory,
                                                                    GLuint64 offset);

// GL_EXT_semaphore
typedef void(INTERNAL_GL_APIENTRY *PFNGLGENSEMAPHORESEXTPROC)(GLsizei n, GLuint *semaphores);
typedef void(INTERNAL_GL_APIENTRY *PFNGLDELETESEMAPHORESEXTPROC)(GLsizei n,
                                                                 const GLuint *semaphores);
typedef GLboolean(INTERNAL_GL_APIENTRY *PFNGLISSEMAPHOREEXTPROC)(GLuint semaphore);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSEMAPHOREPARAMETERUI64VEXTPROC)(GLuint semaphore,
                                                                        GLenum pname,
                                                                        const GLuint64 *params);
typedef void(INTERNAL_GL_APIENTRY *PFNGLGETSEMAPHOREPARAMETERUI64VEXTPROC)(GLuint semaphore,
                                                                           GLenum pname,
                                                                           GLuint64 *params);
typedef void(INTERNAL_GL_APIENTRY *PFNGLWAITSEMAPHOREEXTPROC)(GLuint semaphore,
                                                              GLuint numBufferBarriers,
                                                              const GLuint *buffers,
                                                              GLuint numTextureBarriers,
                                                              const GLuint *textures,
                                                              const GLenum *srcLayouts);
typedef void(INTERNAL_GL_APIENTRY *PFNGLSIGNALSEMAPHOREEXTPROC)(GLuint semaphore,
                                                                GLuint numBufferBarriers,
                                                                const GLuint *buffers,
                                                                GLuint numTextureBarriers,
                                                                const GLuint *textures,
                                                                const GLenum *dstLayouts);

// GL_EXT_memory_object_fd
typedef void(INTERNAL_GL_APIENTRY *PFNGLIMPORTMEMORYFDEXTPROC)(GLuint memory,
                                                               GLuint64 size,
                                                               GLenum handleType,
                                                               GLint fd);

// GL_EXT_semaphore_fd
typedef void(INTERNAL_GL_APIENTRY *PFNGLIMPORTSEMAPHOREFDEXTPROC)(GLuint semaphore,
                                                                  GLenum handleType,
                                                                  GLint fd);

// GL_EXT_memory_object_win32
typedef void(INTERNAL_GL_APIENTRY *PFNGLIMPORTMEMORYWIN32HANDLEEXTPROC)(GLuint memory,
                                                                        GLuint64 size,
                                                                        GLenum handleType,
                                                                        void *handle);
typedef void(INTERNAL_GL_APIENTRY *PFNGLIMPORTMEMORYWIN32NAMEEXTPROC)(GLuint memory,
                                                                      GLuint64 size,
                                                                      GLenum handleType,
                                                                      const void *name);

// GL_EXT_semaphore_win32
typedef void(INTERNAL_GL_APIENTRY *PFNGLIMPORTSEMAPHOREWIN32HANDLEEXTPROC)(GLuint semaphore,
                                                                           GLenum handleType,
                                                                           void *handle);
typedef void(INTERNAL_GL_APIENTRY *PFNGLIMPORTSEMAPHOREWIN32NAMEEXTPROC)(GLuint semaphore,
                                                                         GLenum handleType,
                                                                         const void *name);

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_FUNCTIONSGLTYPEDEFS_H_
